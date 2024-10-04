import Mathlib
import Mathlib.Algebra.ArithmeticDerivative.Basic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Combinatorics.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Order.Roots
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.LocalExtr
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Combinatorics.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Pi
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic

namespace problem_1_problem_2_l504_504436

def f (a x : ℝ) : ℝ :=
  if x < 1 then 2 * x + a else -x - 2 * a

theorem problem_1 (h : a = -3) : f a 10 = -4 ∧ f a (f a 10) = -11 :=
by {
  simp [f, h],
  split;
  simp
}

theorem problem_2 (h : f a (1 - a) = f a (1 + a)) : a = -3 / 4 :=
by {
  sorry
}

end problem_1_problem_2_l504_504436


namespace annual_interest_rate_l504_504107

variable (P : ℝ) (t : ℝ)
variable (h1 : t = 25)
variable (h2 : ∀ r : ℝ, P * 2 = P * (1 + r * t))

theorem annual_interest_rate : ∃ r : ℝ, P * 2 = P * (1 + r * t) ∧ r = 0.04 := by
  sorry

end annual_interest_rate_l504_504107


namespace solve_for_y_l504_504661

theorem solve_for_y (y : ℚ) : y - 1 / 2 = 1 / 6 - 2 / 3 + 1 / 4 → y = 1 / 4 := by
  intro h
  sorry

end solve_for_y_l504_504661


namespace coin_flip_probability_l504_504670

def total_outcomes := 2^6
def favorable_outcomes := 2^3
def probability := favorable_outcomes / total_outcomes

theorem coin_flip_probability :
  probability = 1 / 8 :=
by
  unfold probability total_outcomes favorable_outcomes
  sorry

end coin_flip_probability_l504_504670


namespace triangle_perimeter_l504_504429

theorem triangle_perimeter (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (AB AC : ℝ) (angle_A : ℝ)
  (h1 : AB = 4) (h2 : AC = 4) (h3 : angle_A = 60) : 
  AB + AC + AB = 12 :=
by {
  sorry
}

end triangle_perimeter_l504_504429


namespace quadratic_inequality_solution_l504_504921

theorem quadratic_inequality_solution {a b : ℝ} 
  (h1 : (∀ x : ℝ, ax^2 - bx - 1 ≥ 0 ↔ (x = 1/3 ∨ x = 1/2))) : 
  ∃ a b : ℝ, (∀ x : ℝ, x^2 - b * x - a < 0 ↔ (-3 < x ∧ x < -2)) :=
by
  sorry

end quadratic_inequality_solution_l504_504921


namespace simplify_nested_sqrt_l504_504643

-- Define the expressions under the square roots
def expr1 : ℝ := 12 + 8 * real.sqrt 3
def expr2 : ℝ := 12 - 8 * real.sqrt 3

-- Problem statement to prove
theorem simplify_nested_sqrt : real.sqrt expr1 + real.sqrt expr2 = 4 * real.sqrt 2 :=
by
  sorry

end simplify_nested_sqrt_l504_504643


namespace find_first_part_l504_504787

variable (x y : ℕ)

theorem find_first_part (h₁ : x + y = 24) (h₂ : 7 * x + 5 * y = 146) : x = 13 :=
by
  -- The proof is omitted
  sorry

end find_first_part_l504_504787


namespace solve_equation_l504_504663

theorem solve_equation :
  ∃ x : ℝ, (sqrt (x + 16) - (8 * Real.cos (Real.pi / 6)) / (sqrt (x + 16)) = 4) ∧
            (x = (2 + 2 * sqrt (1 + sqrt 3))^2 - 16) :=
by
  sorry

end solve_equation_l504_504663


namespace smallest_k_for_quadratic_l504_504973

noncomputable def quadratic_has_two_distinct_real_roots (k : ℝ) : Prop :=
  let a := k in
  let b := -3 in
  let c := -9 / 4 in
  b^2 - 4*a*c > 0

theorem smallest_k_for_quadratic : 
  ∃ k : ℤ, quadratic_has_two_distinct_real_roots k ∧ k > -1 ∧ k ≠ 0 ∧ ∀ m : ℤ, quadratic_has_two_distinct_real_roots m → m > -1 → m ≠ 0 → k ≤ m :=
sorry

end smallest_k_for_quadratic_l504_504973


namespace sum_of_powers_of_i_l504_504373

theorem sum_of_powers_of_i : 
  ∀ (i : ℂ), i^2 = -1 → (∑ k in range 601, i^k) = 1 := by
  sorry

end sum_of_powers_of_i_l504_504373


namespace function_domain_function_composition_function_range_l504_504452

noncomputable def f (x : ℝ) : ℝ := 2 / (|x| - 1)

theorem function_domain : ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 ↔ (∃ y : ℝ, y = f x) := sorry

theorem function_composition : f (f (-5)) ≠ 4 := sorry

theorem function_range : ∀ y : ℝ, y ∈ set.range f ↔ (y ≤ -2 ∨ y > 0) := sorry

end function_domain_function_composition_function_range_l504_504452


namespace necessary_but_not_sufficient_l504_504306

theorem necessary_but_not_sufficient (a b : ℝ) : (a > b) → (a + 1 > b - 2) :=
by sorry

end necessary_but_not_sufficient_l504_504306


namespace haley_balls_l504_504801

theorem haley_balls (bags : ℕ) (balls_per_bag : ℕ) (h_bags : bags = 9) (h_balls_per_bag : balls_per_bag = 4) : 
    bags * balls_per_bag = 36 :=
by 
  simp [h_bags, h_balls_per_bag]
  sorry

end haley_balls_l504_504801


namespace event_tv_weather_forecast_is_random_l504_504230

namespace EventClassification

inductive Event
| certain : Event
| impossible : Event
| random : Event

open Event

def is_certain (e : Event) : Prop := e = certain
def is_impossible (e : Event) : Prop := e = impossible

noncomputable def classify_event : Event :=
if ¬is_certain random ∧ ¬is_impossible random then random else sorry

theorem event_tv_weather_forecast_is_random :
  ¬is_certain random ∧ ¬is_impossible random → classify_event = random :=
by
  intros h,
  unfold classify_event,
  simp [h],
  sorry

end EventClassification

end event_tv_weather_forecast_is_random_l504_504230


namespace prime_count_between_50_and_70_l504_504082

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_50_and_70 : List ℕ :=
  [53, 59, 61, 67]

theorem prime_count_between_50_and_70 : 
  (primes_between_50_and_70.filter is_prime).length = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504082


namespace investment_of_b_l504_504773

variable (a b c : ℕ)
variable (A_share B_share : ℕ)

-- Conditions as given in the problem
def investments (a b _: ℕ) := a = 7000 ∧ c = 18000
def shares (A_share B_share : ℕ) := A_share = 1400 ∧ B_share = 2200

-- The main theorem to be proved
theorem investment_of_b (B : ℕ) (h_investments : investments a b _) (h_shares : shares A_share B_share)
    (h_time : ∀ (n : ℕ), n = 8) :
    B = 11000 := by
    sorry

end investment_of_b_l504_504773


namespace replace_asterisk_with_monomial_l504_504196

theorem replace_asterisk_with_monomial (x : ℝ) :
  (∀ asterisk : ℝ, ((x^4 - 3)^2 + (x^3 + asterisk)^2) = (x^8 + x^6 + 9x^2 + 9)) ↔ asterisk = 3x :=
by sorry

end replace_asterisk_with_monomial_l504_504196


namespace max_possible_difference_l504_504605

theorem max_possible_difference :
  ∃ a b c d e f g h i : ℕ,
    (a, b, c) ∈ [{3, 5, 9}, {2, 3, 7}, {3, 4, 8, 9}] ∧
    (d, e, f) ∈ [{2, 3, 7}, {3, 5, 9}, {1, 4, 7}] ∧
    (g, h, i) ∈ [{4, 5, 9}, {2}, {4, 5, 9}] ∧
    (g * 100 + h * 10 + i = 529) ∧
    (a * 100 + b * 10 + c = d * 100 + e * 10 + f + 529) ∧
    (a * 100 + b * 10 + c = 923) ∧
    (d * 100 + e * 10 + f = 394) :=
begin
  sorry
end

end max_possible_difference_l504_504605


namespace circle_intersection_l504_504941

-- Definitions of the given conditions
def circle_1_polar : ℝ → ℝ → Prop := λ ρ θ, ρ = 2
def circle_2_polar : ℝ → ℝ → Prop := λ ρ θ, ρ^2 - 2 * sqrt 2 * ρ * cos (θ - π / 4) = 2

-- Required equations in Cartesian coordinates and polar coordinates for the line
def circle_1_cartesian (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_2_cartesian (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 2 * y - 2 = 0
def intersection_line_polar (ρ θ : ℝ) : Prop := ρ * sin (θ + π / 4) = sqrt 2 / 2

-- The theorem we need to prove
theorem circle_intersection (ρ θ x y : ℝ) :
  (∀ θ, circle_1_polar ρ θ → circle_1_cartesian x y) ∧
  (∀ θ, circle_2_polar ρ θ → circle_2_cartesian x y) ∧
  (∀ x y, circle_1_cartesian x y ∧ circle_2_cartesian x y → intersection_line_polar ρ θ) := by
  sorry

end circle_intersection_l504_504941


namespace sum_of_plane_angles_eq_twice_edges_l504_504165

theorem sum_of_plane_angles_eq_twice_edges (E F : ℕ) (m : Fin F → ℕ) :
  (∑ i, m i) = 2 * E :=
by
  sorry

end sum_of_plane_angles_eq_twice_edges_l504_504165


namespace max_distance_9_l504_504694

theorem max_distance_9 (a : ℕ → ℕ) (h_sum : (∑ i in finset.range 10, a i = 76))
  (h1 : ∀ i, i < 9 → a i + a (i + 1) ≤ 16)
  (h2 : ∀ i, i < 8 → a i + a (i + 1) + a (i + 2) ≥ 23) :
  ∃ i, a i ≤ 9 :=
by
  sorry

end max_distance_9_l504_504694


namespace max_unique_sundaes_l504_504827

theorem max_unique_sundaes (n : ℕ) (h : n = 8) : 
  (n + (n.choose 2) = 36) :=
by
  rw h
  simp [Nat.choose]
  sorry

end max_unique_sundaes_l504_504827


namespace contrapositive_true_l504_504894

theorem contrapositive_true (q p : Prop) (h : q → p) : ¬p → ¬q :=
by sorry

end contrapositive_true_l504_504894


namespace smallest_3cut_4cut_l504_504150

theorem smallest_3cut_4cut : ∃ (n : ℕ), n > 2 ∧ (n - 2) % 3 = 0 ∧ (n - 2) % 4 = 0 ∧ n = 14 :=
by {
  use 14,
  split,
  { exact Nat.lt_of_succ_lt_succ (Nat.zero_lt_succ 2) },
  split,
  { exact Nat.mod_eq_zero_of_dvd (DVD.intro 4 rfl) },
  split,
  { exact Nat.mod_eq_zero_of_dvd (DVD.intro 3 rfl) },
  refl,
}

end smallest_3cut_4cut_l504_504150


namespace three_points_aligned_nec_but_not_suf_four_points_coplanar_l504_504489

theorem three_points_aligned_nec_but_not_suf_four_points_coplanar
  (A B C D : Point) :
  (∃ l : Line, A ∈ l ∧ B ∈ l ∧ C ∈ l) → 
  (∃ P : Plane, A ∈ P ∧ B ∈ P ∧ C ∈ P ∧ D ∈ P) ∧ 
  (¬ (∃ l : Line, A ∈ l ∧ B ∈ l ∧ C ∈ l ∧ D ∈ l)) :=
sorry

end three_points_aligned_nec_but_not_suf_four_points_coplanar_l504_504489


namespace trig_identity_l504_504925

theorem trig_identity :
  (2 * Real.sin (46 * Real.pi / 180) - Real.sqrt 3 * Real.cos (74 * Real.pi / 180)) / Real.cos (16 * Real.pi / 180) = 1 :=
by
  sorry

end trig_identity_l504_504925


namespace base_six_to_base_ten_equivalent_l504_504750

theorem base_six_to_base_ten_equivalent :
  let n := 12345
  (5 * 6^0 + 4 * 6^1 + 3 * 6^2 + 2 * 6^3 + 1 * 6^4) = 1865 :=
by
  sorry

end base_six_to_base_ten_equivalent_l504_504750


namespace julien_contribution_l504_504553

def exchange_rate : ℝ := 1.5
def cost_of_pie : ℝ := 12
def lucas_cad : ℝ := 10

theorem julien_contribution : (cost_of_pie - lucas_cad / exchange_rate) = 16 / 3 := by
  sorry

end julien_contribution_l504_504553


namespace grade_S_percentage_correct_l504_504346

def class_scores : List ℕ := [95, 88, 70, 100, 75, 90, 80, 77, 67, 78, 85, 65, 72, 82, 96]

def is_grade_S (score : ℕ) : Bool := score >= 95 ∧ score <= 100

def count_students_with_grade_S (scores : List ℕ) : ℕ :=
  scores.countp is_grade_S

def total_students (scores : List ℕ) : ℕ := scores.length

noncomputable def percentage_S_students (scores : List ℕ) : ℕ :=
  (count_students_with_grade_S scores * 100) / total_students scores

theorem grade_S_percentage_correct :
  percentage_S_students class_scores = 20 := by
  sorry

end grade_S_percentage_correct_l504_504346


namespace construct_convex_solid_l504_504304

-- Given conditions about the shapes and the arrangement
def perpendicular_squares (ABCD CDEF : Prop) := ABCD ∧ CDEF
def isosceles_right_triangles (ABG EFH : Prop) := ABG ∧ EFH
def planes_meeting_at_K (GA AD GB BC DE EH CF FH K : Prop) := 
    (GA ∧ AD) ∧ (GB ∧ BC) ∧ (DE ∧ EH) ∧ (CF ∧ FH) ∧ K

-- Definition for congruent bricks forming a convex solid
def construction_possible (convex_solid : Prop) := 
  ∃ (shaped_bricks : Prop), 
  ∀ (ABCD CDEF ABG EFH GA AD GB BC DE EH CF FH K : Prop), 
  perpendicular_squares ABCD CDEF →
  isosceles_right_triangles ABG EFH →
  planes_meeting_at_K GA AD GB BC DE EH CF FH K →
  shaped_bricks ∧ convex_solid

-- Main statement
theorem construct_convex_solid:
  ∃ (convex_solid : Prop),
    construction_possible convex_solid :=
begin
  sorry
end

end construct_convex_solid_l504_504304


namespace proof_problem_l504_504109

variables (A B C D P : Type) [AffineSpace ℝ A]
variables (AD BC AB AP BP CD h : ℝ)

-- Given
axiom h_nonneg : 0 < h
axiom convex_quad : AB = AD + BC
axiom AP_eq : AP = h + AD
axiom BP_eq : BP = h + BC

-- The goal is to prove the following inequality:
theorem proof_problem (H : AB = AD + BC) (H1: AP = h + AD) (H2: BP = h + BC) (H3 : 0 < h) : 
  1 / sqrt h ≥ 1 / sqrt AD + 1 / sqrt BC := 
sorry

end proof_problem_l504_504109


namespace count_pairs_s_eq_3t_l504_504774

def S : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

theorem count_pairs_s_eq_3t : 
  Finset.card 
    ((Finset.filter (λ (pair : ℕ × ℕ), pair.fst = 3 * pair.snd ∧ pair.fst ∈ S ∧ pair.snd ∈ S) 
      (Finset.product (Finset.ofFinset S) (Finset.ofFinset S)))) = 5 := by
sorry

end count_pairs_s_eq_3t_l504_504774


namespace cistern_fill_time_l504_504792

variable (C : ℝ) -- Volume of the cistern
variable (X Y Z : ℝ) -- Rates at which pipes X, Y, and Z fill the cistern

-- Pipes X and Y together, pipes X and Z together, and pipes Y and Z together conditions
def condition1 := X + Y = C / 3
def condition2 := X + Z = C / 4
def condition3 := Y + Z = C / 5

theorem cistern_fill_time (h1 : condition1 C X Y) (h2 : condition2 C X Z) (h3 : condition3 C Y Z) :
  1 / (X + Y + Z) = 120 / 47 :=
by
  sorry

end cistern_fill_time_l504_504792


namespace recurring_decimal_to_fraction_l504_504388

theorem recurring_decimal_to_fraction :
  (0.3 + (0.\overline{07} : Real)) = (367 / 990 : Real) :=
by
  sorry

end recurring_decimal_to_fraction_l504_504388


namespace fifteenth_entry_is_24_l504_504884

open Nat

-- Definition stating the condition: the remainder when 3n is divided by 7 is at most 3
def satisfies_condition (n : ℕ) : Prop := n % 7 ∈ {0, 1, 3, 5}

-- Define a sequence that generates all n satisfying the condition in order
def sequence : ℕ → ℕ
| 0       := 0
| (n + 1) := Nat.find_x (λ k, satisfies_condition k ∧ n < Nat.count satisfies_condition k)

-- Prove that the 15th element in this sequence is 24
theorem fifteenth_entry_is_24 : sequence 14 = 24 := by
  sorry

end fifteenth_entry_is_24_l504_504884


namespace perpendicular_vectors_l504_504456

-- Define the vectors a and b
def a : (ℝ × ℝ) := (1, -3)
def b : (ℝ × ℝ) := (4, -2)

-- Dot product function
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Main statement
theorem perpendicular_vectors (λ : ℝ) (h : dot_product (λ • a + b) a = 0) : 
  λ = -1 := 
by
  sorry

end perpendicular_vectors_l504_504456


namespace shaded_region_area_l504_504509

open Real

noncomputable def semicircle_area (d : ℝ) : ℝ := (1 / 8) * π * d^2

theorem shaded_region_area :
  let UV := 3
  let VW := 5
  let WX := 4
  let XY := 6
  let YZ := 7
  let UZ := UV + VW + WX + XY + YZ
  let area_UZ := semicircle_area UZ
  let area_UV := semicircle_area UV
  let area_VW := semicircle_area VW
  let area_WX := semicircle_area WX
  let area_XY := semicircle_area XY
  let area_YZ := semicircle_area YZ
  area_UZ - (area_UV + area_VW + area_WX + area_XY + area_YZ) = (247/4) * π :=
sorry

end shaded_region_area_l504_504509


namespace number_of_integers_satisfying_abs_inequality_l504_504954

theorem number_of_integers_satisfying_abs_inequality :
  (setOf (λ x : ℤ, abs (4 * x - 5) < 9)).finite.card = 4 :=
by sorry

end number_of_integers_satisfying_abs_inequality_l504_504954


namespace intersection_in_quadrant_II_l504_504375

theorem intersection_in_quadrant_II (x y : ℝ) 
  (h1: y ≥ -2 * x + 3) 
  (h2: y ≤ 3 * x + 6) 
  (h_intersection: x = -3 / 5 ∧ y = 21 / 5) :
  x < 0 ∧ y > 0 := 
sorry

end intersection_in_quadrant_II_l504_504375


namespace lengthMC_equals_3sqrt10_l504_504208

noncomputable def lengthSegmentMC (A B M C : ℝ) : ℝ :=
  if AB_length : B - A = 6
  ∧ midpoint_arc_M : M = (A + B) / 2
  then real.sqrt (90)
  else 0

theorem lengthMC_equals_3sqrt10 (A B M C : ℝ) (AB_length : B - A = 6)
  (midpoint_arc_M : M = (A + B) / 2) :
  lengthSegmentMC A B M C = 3 * real.sqrt 10 := by
  sorry

end lengthMC_equals_3sqrt10_l504_504208


namespace binomial_coefficient_7_5_permutation_7_5_l504_504368

-- Define function for binomial coefficient
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define function for permutation calculation
def permutation (n k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

theorem binomial_coefficient_7_5 : binomial_coefficient 7 5 = 21 :=
by
  sorry

theorem permutation_7_5 : permutation 7 5 = 2520 :=
by
  sorry

end binomial_coefficient_7_5_permutation_7_5_l504_504368


namespace exists_digit_sum_divisible_by_11_l504_504612

/-- The sum of digits of a natural number. -/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_digit_sum_divisible_by_11 :
  ∀ (a : ℕ), ∃ n ∈ (set.Icc a (a+38) : set ℕ), sum_of_digits n % 11 = 0 :=
by
  sorry

end exists_digit_sum_divisible_by_11_l504_504612


namespace sequence_sum_l504_504374

-- Define the sequence with the given conditions
def sequence (a : ℕ → ℕ) : Prop :=
  (a 1 = 1) ∧ (a 2 = 1) ∧ (a 3 = 2) ∧
  (∀ n : ℕ, a n * a (n + 1) * a (n + 2) ≠ 1) ∧
  (∀ n : ℕ, a n * a (n + 1) * a (n + 2) * a (n + 3) = a n + a (n + 1) + a (n + 2) + a (n + 3))

-- Prove the sum of the first 100 terms equals 200
theorem sequence_sum (a : ℕ → ℕ) (h : sequence a) :
  (Finset.range 100).sum a = 200 :=
sorry

end sequence_sum_l504_504374


namespace solve_equation_sqrt_inv_l504_504878

theorem solve_equation_sqrt_inv (x : ℝ) (h : 3 * Real.sqrt x + 3 * (x ^ (-1 / 2)) = 6) :
  x = 1 :=
sorry

end solve_equation_sqrt_inv_l504_504878


namespace replace_with_monomial_produces_four_terms_l504_504175

-- Define the initial expression
def initialExpression (k : ℤ) (x : ℤ) : ℤ :=
  ((x^4 - 3)^2 + (x^3 + k)^2)

-- Proof statement
theorem replace_with_monomial_produces_four_terms (x : ℤ) :
  ∃ (k : ℤ), initialExpression k x = (x^8 + x^6 + 9x^2 + 9) :=
  exists.intro (3 * x) sorry

end replace_with_monomial_produces_four_terms_l504_504175


namespace odd_function_properties_l504_504099

variable {ℝ : Type} [LinearOrderedField ℝ]

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_properties (h_odd : ∀ x, f (-x) = -f x)
    (h_inc_1_3 : ∀ x y : ℝ, 1 ≤ x → x ≤ 3 → 1 ≤ y → y ≤ 3 → x < y → f x < f y)
    (h_min_val : ∀ x : ℝ, 1 ≤ x → x ≤ 3 → f x ≥ 7)
    (h_min_7 : ∃ x : ℝ, 1 ≤ x → x ≤ 3 → f x = 7) :
    (∀ x y : ℝ, -3 ≤ x → x ≤ -1 → -3 ≤ y → y ≤ -1 → x < y → f x < f y) ∧ 
    (∀ x : ℝ, -3 ≤ x → x ≤ -1 → f x ≤ -7) ∧ 
    (∃ x : ℝ, -3 ≤ x → x ≤ -1 → f x = -7) :=
  sorry

end odd_function_properties_l504_504099


namespace master_craftsman_quota_l504_504534

theorem master_craftsman_quota (N : ℕ) (initial_rate increased_rate : ℕ) (additional_hours extra_hours : ℝ) :
  initial_rate = 35 →
  increased_rate = initial_rate + 15 →
  additional_hours = 0.5 →
  extra_hours = 1 →
  N / initial_rate - N / increased_rate = additional_hours + extra_hours →
  N = 175 →
  (initial_rate + N) = 210 :=
by {
  intros h1 h2 h3 h4 h5 h6,
  rw h6,
  exact rfl,
}

end master_craftsman_quota_l504_504534


namespace cube_of_720_diamond_1001_l504_504243

-- Define the operation \diamond
def diamond (a b : ℕ) : ℕ :=
  (Nat.factors (a * b)).toFinset.card

-- Define the specific numbers 720 and 1001
def n1 : ℕ := 720
def n2 : ℕ := 1001

-- Calculate the cubic of the result of diamond operation
def cube_of_diamond : ℕ := (diamond n1 n2) ^ 3

-- The statement to be proved
theorem cube_of_720_diamond_1001 : cube_of_diamond = 216 :=
by {
  sorry
}

end cube_of_720_diamond_1001_l504_504243


namespace kim_time_away_from_home_l504_504556

noncomputable def time_away_from_home (distance_to_friend : ℕ) (detour_percent : ℕ) (stay_time : ℕ) (speed_mph : ℕ) : ℕ :=
  let return_distance := distance_to_friend * (1 + detour_percent / 100)
  let total_distance := distance_to_friend + return_distance
  let driving_time := total_distance / speed_mph
  let driving_time_minutes := driving_time * 60
  driving_time_minutes + stay_time

theorem kim_time_away_from_home : 
  time_away_from_home 30 20 30 44 = 120 := 
by
  -- We will handle the proof here
  sorry

end kim_time_away_from_home_l504_504556


namespace flower_bee_difference_l504_504709

theorem flower_bee_difference : ∀ (flowers bees : ℕ), flowers = 5 → bees = 3 → (flowers - bees) = 2 := by
  intros flowers bees hflowers hbees
  rw [hflowers, hbees]
  exact (Nat.sub_self 3).symm
  sorry

end flower_bee_difference_l504_504709


namespace ineq_condition_l504_504309

theorem ineq_condition (a b : ℝ) : (a + 1 > b - 2) ↔ (a > b - 3 ∧ ¬(a > b)) :=
by
  sorry

end ineq_condition_l504_504309


namespace quadratic_function_eqn_l504_504703

theorem quadratic_function_eqn (a : ℝ) (x1 x2 : ℝ) :
  let y := -1 / 3 * (x + 2)^2 + 3 in
  y = -1 / 3 * (x + 2)^2 + 3 ∧ |x1 - x2| = 6 :=
by { sorry }

end quadratic_function_eqn_l504_504703


namespace tan_ratio_identity_l504_504443

theorem tan_ratio_identity
  (x y : ℝ)
  (h1 : sin x / cos y + sin y / cos x = 2)
  (h2 : cos x / sin y + cos y / sin x = 5) :
  (tan x / tan y) + (tan y / tan x) = 10 := by
  sorry

end tan_ratio_identity_l504_504443


namespace constant_term_expansion_2x_1_over_x_4_eq_24_l504_504217

theorem constant_term_expansion_2x_1_over_x_4_eq_24 : 
  (let a := 2; let b := (1 : ℝ); let n := 4 in
  ∑ r in finset.range (n + 1), binomial n r * (a * a) ^ (n - r) * (b / a) ^ r * x ^ (n - 2 * r)) = 24 :=
begin
  sorry
end

end constant_term_expansion_2x_1_over_x_4_eq_24_l504_504217


namespace exists_valid_coloring_8_no_valid_coloring_9_l504_504581

def is_valid_coloring (n : ℕ) (coloring : Fin n → Prop) : Prop :=
  ∀ (x y : Fin n), x + y < n → (x + y) % 2 = 0 → coloring x ≠ coloring ((x + y) / 2) ∨ coloring ((x + y) / 2) ≠ coloring y ∨ coloring x ≠ coloring y

-- Problem (a): Exists a valid coloring for n = 8
theorem exists_valid_coloring_8 : ∃ coloring : Fin 8 → Prop, is_valid_coloring 8 coloring := sorry

-- Problem (b): No valid coloring for n = 9
theorem no_valid_coloring_9 : ¬ ∃ coloring : Fin 9 → Prop, is_valid_coloring 9 coloring := sorry

end exists_valid_coloring_8_no_valid_coloring_9_l504_504581


namespace derivative_of_f_l504_504012

def f(x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem derivative_of_f :
  Deriv f = fun x => Real.exp x * (Real.sin x + Real.cos x) :=
by
  sorry

end derivative_of_f_l504_504012


namespace tan_sum_l504_504961

theorem tan_sum (α β : ℝ) 
  (h1: Real.tan α + Real.tan β = 25) 
  (h2: Real.cot α + Real.cot β = 30) 
  : Real.tan (α + β) = 150 :=
sorry

end tan_sum_l504_504961


namespace object_speed_is_68_18_mph_l504_504775

noncomputable def speed_in_mph (distance_ft : ℕ) (time_sec : ℕ) : ℝ :=
  let distance_miles := (distance_ft : ℝ) / 5280
  let time_hours := (time_sec : ℝ) / 3600
  distance_miles / time_hours

theorem object_speed_is_68_18_mph :
  speed_in_mph 200 2 ≈ 68.18 := sorry

end object_speed_is_68_18_mph_l504_504775


namespace replace_star_with_3x_l504_504178

theorem replace_star_with_3x (x : ℝ) :
  (x^4 - 3)^2 + (x^3 + 3x)^2 = x^8 + x^6 + 9x^2 + 9 :=
by
  sorry

end replace_star_with_3x_l504_504178


namespace max_total_time_l504_504598

theorem max_total_time :
  ∀ (time_mowing time_fertilizing total_time : ℕ), 
    time_mowing = 40 ∧ time_fertilizing = 2 * time_mowing ∧ total_time = time_mowing + time_fertilizing → 
    total_time = 120 :=
by
  intros time_mowing time_fertilizing total_time h
  have h1: time_mowing = 40 := h.1
  have h2: time_fertilizing = 2 * time_mowing := h.2.1
  have h3: total_time = time_mowing + time_fertilizing := h.2.2
  rw[h1] at h2
  rw[h1, h2] at h3
  simp at h3
  exact h3.symm

end max_total_time_l504_504598


namespace simplify_sqrt_sum_l504_504650

noncomputable def sqrt_expr_1 : ℝ := Real.sqrt (12 + 8 * Real.sqrt 3)
noncomputable def sqrt_expr_2 : ℝ := Real.sqrt (12 - 8 * Real.sqrt 3)

theorem simplify_sqrt_sum : sqrt_expr_1 + sqrt_expr_2 = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_sum_l504_504650


namespace malcolm_instagram_followers_l504_504593

variables (I facebook twitter tiktok youtube : ℕ)

-- Define the conditions as variables
def facebook_followers := 500
def twitter_followers := (I + facebook_followers) / 2
def tiktok_followers := 3 * twitter_followers
def youtube_followers := tiktok_followers + 510
def total_followers := I + facebook_followers + twitter_followers + tiktok_followers + youtube_followers

theorem malcolm_instagram_followers :
  total_followers I facebook_followers twitter_followers tiktok_followers youtube_followers = 3840 →
  I = 240 :=
by
  sorry

end malcolm_instagram_followers_l504_504593


namespace simplify_sqrt_sum_l504_504651

noncomputable def sqrt_expr_1 : ℝ := Real.sqrt (12 + 8 * Real.sqrt 3)
noncomputable def sqrt_expr_2 : ℝ := Real.sqrt (12 - 8 * Real.sqrt 3)

theorem simplify_sqrt_sum : sqrt_expr_1 + sqrt_expr_2 = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_sum_l504_504651


namespace range_absolute_difference_l504_504849

theorem range_absolute_difference : ∀ y, y = |x + 5| - |x - 3| → y ∈ set.Icc (-8) 8 :=
by
  sorry

end range_absolute_difference_l504_504849


namespace ellipse_with_foci_on_y_axis_l504_504674

theorem ellipse_with_foci_on_y_axis (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1) ↔ (m > n ∧ n > 0) := 
sorry

end ellipse_with_foci_on_y_axis_l504_504674


namespace min_value_of_ratio_l504_504951

noncomputable def min_ratio (m : ℝ) (hm : m > 0) : ℝ :=
  let xA := 2 ^ -m
  let xB := 2 ^ m
  let xC := 2 ^ -(8 / (2 * m + 1))
  let xD := 2 ^ (8 / (2 * m + 1))
  let a := |xA - xC|
  let b := |xB - xD|
  b / a

theorem min_value_of_ratio : 
  (∀ m : ℝ, m > 0 → (min_ratio m (by assumption)) ≥ 8 * Real.sqrt 2) :=
by 
  sorry

end min_value_of_ratio_l504_504951


namespace prime_numbers_between_50_and_70_l504_504055

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_between_50_and_70 : 
  (finset.filter is_prime (finset.range 71)).count (λ n, 50 ≤ n ∧ n ≤ 70) = 4 := 
sorry

end prime_numbers_between_50_and_70_l504_504055


namespace additional_terms_induction_l504_504166

theorem additional_terms_induction {n : ℕ} (h₁ : n ≥ 2) (h₂ : 0 < n) : 
  ∀ k : ℕ, (1 + ∑ i in Finset.range (2^k), 1 / (↑i + 1) = 1 + ∑ i in Finset.range (2^k), 1 / (↑i + 1) 
           + ∑ j in Finset.range' (2^k) (2^(k+1) - 1 + 1), 1 / (↑j + 1)) :=
begin
  sorry
end

end additional_terms_induction_l504_504166


namespace simplify_sqrt_sum_l504_504648

noncomputable def sqrt_expr_1 : ℝ := Real.sqrt (12 + 8 * Real.sqrt 3)
noncomputable def sqrt_expr_2 : ℝ := Real.sqrt (12 - 8 * Real.sqrt 3)

theorem simplify_sqrt_sum : sqrt_expr_1 + sqrt_expr_2 = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_sum_l504_504648


namespace avg_time_car_in_storm_l504_504318

-- Define the conditions in Lean
def car_position (t : ℝ) : ℝ × ℝ := ((5/4) * t, 0)
def storm_position (t : ℝ) : ℝ × ℝ := (0, 110 - (1/2) * t)
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the time when the distance is exactly the storm radius
def times_when_car_enters_and_leaves_storm (t : ℝ) : Prop :=
  distance (car_position t) (storm_position t) = 51

-- Prove the average of these times is 880/29
theorem avg_time_car_in_storm :
  ∃ t1 t2 : ℝ, times_when_car_enters_and_leaves_storm t1 ∧ times_when_car_enters_and_leaves_storm t2 ∧
    (t1 + t2) / 2 = 880 / 29 :=
by
  sorry

end avg_time_car_in_storm_l504_504318


namespace average_k_value_l504_504008

theorem average_k_value (k : ℕ) (r1 r2 : ℕ) 
  (h1 : r1 + r2 = k) 
  (h2 : r1 * r2 = 24)
  (h3 : r1 > 0)
  (h4 : r2 > 0) :
  (1 + 24 = k ∨ 2 + 12 = k ∨ 3 + 8 = k ∨ 4 + 6 = k) →
  (∑ k in {25, 14, 11, 10}, k) / (card {25, 14, 11, 10}) = 15 :=
by
  sorry

end average_k_value_l504_504008


namespace count_primes_between_fifty_and_seventy_l504_504063

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_fifty_and_seventy : List ℕ :=
  [53, 59, 61, 67]

theorem count_primes_between_fifty_and_seventy :
  (primes_between_fifty_and_seventy.count is_prime = 4) :=
by
  sorry

end count_primes_between_fifty_and_seventy_l504_504063


namespace distance_center_to_point_l504_504751

theorem distance_center_to_point :
  let center : ℝ × ℝ := (3, 5)
  let point : ℝ × ℝ := (-4, -2)
  let distance : ℝ := Real.sqrt ((-4 - 3)^2 + (-2 - 5)^2)
  x^2 + y^2 = 6x + 10y + 9 → distance = 7 * Real.sqrt 2 :=
by
  sorry

end distance_center_to_point_l504_504751


namespace intersection_M_N_l504_504947

def M : Set ℝ := { x | x / (x - 1) ≥ 0 }
def N : Set ℝ := { y | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_M_N :
  { x | x / (x - 1) ≥ 0 } ∩ { y | ∃ x : ℝ, y = 3 * x^2 + 1 } = { x | x > 1 } :=
sorry

end intersection_M_N_l504_504947


namespace max_value_m_l504_504458

noncomputable def sequence (m : ℝ) : ℕ → ℝ
| 0       := 1
| (n + 1) := (1 / 8) * (sequence m n) ^ 2 + m

theorem max_value_m (m : ℝ) (h : ∀ (n : ℕ), sequence m (n + 1) < 4) : m ≤ 2 := 
sorry

end max_value_m_l504_504458


namespace local_extrema_of_f_l504_504127

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * x^3 - (1 / 2) * x^2 - 6 * x + (8 / 3)

theorem local_extrema_of_f :
  (f (-2) = 10) ∧ (f (3) = -10 - 5 / 6) :=
by
  have h1 : f (-2) = (1 / 3) * (-2)^3 - (1 / 2) * (-2)^2 - 6 * (-2) + (8 / 3) := rfl
  have h2 : f (3) = (1 / 3) * 3^3 - (1 / 2) * 3^2 - 6 * 3 + (8 / 3) := rfl
  have max_value : f (-2) = 10 := by
    simp [f] at h1
    rw h1
    norm_num
  have min_value : f (3) = -10 - 5 / 6 := by
    simp [f] at h2
    rw h2
    norm_num
  exact ⟨max_value, min_value⟩

end local_extrema_of_f_l504_504127


namespace tv_weight_difference_l504_504365

noncomputable def BillTV_width : ℝ := 48
noncomputable def BillTV_height : ℝ := 100
noncomputable def BobTV_width : ℝ := 70
noncomputable def BobTV_height : ℝ := 60
noncomputable def weight_per_sq_inch : ℝ := 4
noncomputable def ounces_per_pound : ℝ := 16

theorem tv_weight_difference :
  let BillTV_area := BillTV_width * BillTV_height,
      BillTV_weight_oz := BillTV_area * weight_per_sq_inch,
      BillTV_weight_lb := BillTV_weight_oz / ounces_per_pound,
      BobTV_area := BobTV_width * BobTV_height,
      BobTV_weight_oz := BobTV_area * weight_per_sq_inch,
      BobTV_weight_lb := BobTV_weight_oz / ounces_per_pound
  in BillTV_weight_lb - BobTV_weight_lb = 150 :=
by
  sorry

end tv_weight_difference_l504_504365


namespace prime_count_between_50_and_70_l504_504047

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def primes_in_range (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem prime_count_between_50_and_70 : primes_in_range 50 70 = [53, 59, 61, 67] :=
sorry

end prime_count_between_50_and_70_l504_504047


namespace triangle_sine_half_angle_inequality_triangle_sine_cos_half_angle_inequality_l504_504981

-- Part 1: Proof of the sine inequality
theorem triangle_sine_half_angle_inequality
  {A B C : ℝ}
  (hABC : A + B + C = π) -- Condition that A, B, and C are angles in a triangle
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) -- Angles are positive
  : (sin A) * (sin (A / 2)) + (sin B) * (sin (B / 2)) + (sin C) * (sin (C / 2)) ≤ 4 / (sqrt 3) :=
by
  sorry

-- Part 2: Proof of the cosine inequality
theorem triangle_sine_cos_half_angle_inequality
  {A B C : ℝ}
  (hABC : A + B + C = π) -- Condition that A, B, and C are angles in a triangle
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) -- Angles are positive
  : (sin A) * (cos (A / 2)) + (sin B) * (cos (B / 2)) + (sin C) * (cos (C / 2)) ≤ 4 / (sqrt 3) :=
by
  sorry

end triangle_sine_half_angle_inequality_triangle_sine_cos_half_angle_inequality_l504_504981


namespace hershel_fish_remaining_l504_504466

def betta_initial := 10
def goldfish_initial := 15
def betta_bexley := 2.5 * betta_initial
def goldfish_bexley := (2 / 3) * goldfish_initial
def betta_total := betta_initial + betta_bexley
def goldfish_total := goldfish_initial + goldfish_bexley
def betta_sister := (3 / 5) * betta_total
def goldfish_sister := (2 / 5) * goldfish_total
def betta_remaining := betta_total - betta_sister
def goldfish_remaining := goldfish_total - goldfish_sister
def total_remaining := betta_remaining + goldfish_remaining
def cousin_receives := (1 / 4) * total_remaining
def final_remaining := total_remaining - cousin_receives

theorem hershel_fish_remaining : final_remaining = 22 := by
  sorry

end hershel_fish_remaining_l504_504466


namespace repeating_decimal_to_fraction_l504_504391

noncomputable def repeating_decimal_sum (x y z : ℚ) : ℚ := x + y + z

theorem repeating_decimal_to_fraction :
  let x := 4 / 33
  let y := 34 / 999
  let z := 567 / 99999
  repeating_decimal_sum x y z = 134255 / 32929667 := by
  -- proofs are omitted
  sorry

end repeating_decimal_to_fraction_l504_504391


namespace shaded_area_a_length_EF_b_length_EF_c_ratio_ab_d_l504_504786

-- (a) Prove that the area of the shaded region is 36 cm^2
theorem shaded_area_a (AB EF : ℕ) (h1 : AB = 10) (h2 : EF = 8) : (AB ^ 2) - (EF ^ 2) = 36 :=
by
  sorry

-- (b) Prove that the length of EF is 7 cm
theorem length_EF_b (AB : ℕ) (shaded_area : ℕ) (h1 : AB = 13) (h2 : shaded_area = 120)
  : ∃ EF, (AB ^ 2) - (EF ^ 2) = shaded_area ∧ EF = 7 :=
by
  sorry

-- (c) Prove that the length of EF is 9 cm
theorem length_EF_c (AB : ℕ) (h1 : AB = 18)
  : ∃ EF, (AB ^ 2) - ((1 / 4) * AB ^ 2) = (3 / 4) * AB ^ 2 ∧ EF = 9 :=
by
  sorry

-- (d) Prove that a / b = 5 / 3
theorem ratio_ab_d (a b : ℕ) (shaded_percent : ℚ) (h1 : shaded_percent = 0.64)
  : (a ^ 2) - ((0.36) * a ^ 2) = (a ^ 2) * shaded_percent ∧ (a / b) = (5 / 3) :=
by
  sorry

end shaded_area_a_length_EF_b_length_EF_c_ratio_ab_d_l504_504786


namespace find_angle_BPC_l504_504578

variables (A B C D E P : Type) [RegularPentagon A B C D E] 
  (P_inside : PointInsidePentagon P A B C D E)
  (angle_PAB : Angle P A B = 48)
  (angle_PDC : Angle P D C = 42)

theorem find_angle_BPC : Angle B P C = 84 := 
sorry

end find_angle_BPC_l504_504578


namespace smallest_sum_of_20_consecutive_integers_twice_perfect_square_l504_504256

theorem smallest_sum_of_20_consecutive_integers_twice_perfect_square :
  ∃ n : ℕ, ∃ k : ℕ, (∀ m : ℕ, m ≥ n → 0 < m) ∧ 10 * (2 * n + 19) = 2 * k^2 ∧ 10 * (2 * n + 19) = 450 :=
by
  sorry

end smallest_sum_of_20_consecutive_integers_twice_perfect_square_l504_504256


namespace part_I_part_II_part_III_l504_504933

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.exp x - a * x

theorem part_I (a : ℝ) (h_increasing : ∀ x y : ℝ, x < y → f x a < f y a) :
  a ∈ Set.Iic (-1/Real.exp 2) :=
sorry

theorem part_II (h_critical_1 : f.deriv 0 1 = 0 ∧ (∀ x < 0, f.deriv x 1 < 0) ∧ (∀ x > 0, f.deriv x 1 > 0)) :
  IsLocalMinOn (f 0 1) {0} :=
sorry

theorem part_III (a : ℝ) : 
  (a ≤ 0 ∨ a = 1 → ∃! x, f x a = 0) ∧ (0 < a ∧ a < 1 ∨ a > 1 → ∃ x₁ x₂, f x₁ a = 0 ∧ f x₂ a = 0 ∧ x₁ ≠ x₂) :=
sorry

end part_I_part_II_part_III_l504_504933


namespace like_terms_exponent_equality_l504_504959

theorem like_terms_exponent_equality (m n : ℕ) (a b : ℝ) 
    (H : 3 * a^m * b^2 = 2/3 * a * b^n) : m = 1 ∧ n = 2 :=
by
  sorry

end like_terms_exponent_equality_l504_504959


namespace prime_count_between_50_and_70_l504_504069

open Nat

theorem prime_count_between_50_and_70 : 
  (finset.filter Nat.prime (finset.range 71 \ finset.range 51).card = 4) := 
begin
  sorry
end

end prime_count_between_50_and_70_l504_504069


namespace angle_BAC_is_20_degrees_l504_504268

theorem angle_BAC_is_20_degrees
  (A B C O : Point)
  (isosceles_triangle_ABC : is_isosceles_triangle A B C)
  (circumcenter_O : is_circumcenter O (triangle A B C))
  (angle_OAC : angle O A C = 20) :
  angle B A C = 20 := 
sorry

end angle_BAC_is_20_degrees_l504_504268


namespace count_primes_between_fifty_and_seventy_l504_504060

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_fifty_and_seventy : List ℕ :=
  [53, 59, 61, 67]

theorem count_primes_between_fifty_and_seventy :
  (primes_between_fifty_and_seventy.count is_prime = 4) :=
by
  sorry

end count_primes_between_fifty_and_seventy_l504_504060


namespace new_person_weight_l504_504776

theorem new_person_weight (increase : ℝ) (n : ℕ) (old_weight new_weight : ℝ) : 
  n = 15 → increase = 2.3 → old_weight = 80 → 
  new_weight = old_weight + (n * increase) →
  new_weight = 114.5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end new_person_weight_l504_504776


namespace max_stamps_purchase_l504_504485

theorem max_stamps_purchase (price_per_stamp : ℕ) (total_money : ℕ) (h_price : price_per_stamp = 45) (h_money : total_money = 5000) : 
  (total_money / price_per_stamp) = 111 := 
by 
  rw [h_price, h_money]
  rfl

sorry

end max_stamps_purchase_l504_504485


namespace angle_AZY_eq_60_l504_504841

-- Define the angles in terms of Lean's angle operations
variables {A B C X Y Z : Type} [incircle ABC] [circumcircle XYZ]

-- Assume the given conditions in the triangle ABC
variables (angle_A : ∠A = 50) 
          (angle_B : ∠B = 70) 
          (angle_C : ∠C = 60)

-- State the theorem to be proven
theorem angle_AZY_eq_60 (h1 : incircle ABC) 
                        (h2 : circumcircle XYZ) 
                        (h3 : X ∈ (BC : Set Type)) 
                        (h4 : Y ∈ (AB : Set Type)) 
                        (h5 : Z ∈ (AC : Set Type)) 
                        (hA : ∠A = 50) 
                        (hB : ∠B = 70) 
                        (hC : ∠C = 60) : 
  ∠AZY = 60 :=
sorry

end angle_AZY_eq_60_l504_504841


namespace find_integer_solutions_l504_504872

theorem find_integer_solutions (x y : ℕ) (h1 : 1 ≤ x) (h2 : 1 ≤ y) (h3 : x^y^2 = y^x) :
    (x = 1 ∧ y = 1) ∨ (x = 16 ∧ y = 2) ∨ (x = 27 ∧ y = 3) :=
begin
  sorry
end

end find_integer_solutions_l504_504872


namespace replace_star_with_3x_l504_504181

theorem replace_star_with_3x (x : ℝ) :
  (x^4 - 3)^2 + (x^3 + 3x)^2 = x^8 + x^6 + 9x^2 + 9 :=
by
  sorry

end replace_star_with_3x_l504_504181


namespace lina_uphill_comparison_l504_504760

noncomputable theory

def teenager_time_per_mile (distance_teenager : ℕ) (time_teenager_hours : ℕ) : ℕ :=
  (time_teenager_hours * 60) / distance_teenager

def adult_time_per_mile_uphill (distance_uphill : ℕ) (time_uphill_hours : ℕ) : ℕ :=
  (time_uphill_hours * 60) / distance_uphill

theorem lina_uphill_comparison (d_teenager t_teenager d_uphill t_uphill : ℕ) (h1 : d_teenager = 30) (h2 : t_teenager = 2) (h3 : d_uphill = 20) (h4 : t_uphill = 3) :
  adult_time_per_mile_uphill d_uphill t_uphill = teenager_time_per_mile d_teenager t_teenager + 5 := 
by 
  rw [h1, h2, h3, h4]
  rw [teenager_time_per_mile, adult_time_per_mile_uphill]
  norm_num

end lina_uphill_comparison_l504_504760


namespace triangle_third_side_lengths_product_l504_504741

def hypotenuse (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

def leg (c b : ℝ) : ℝ :=
  real.sqrt (c^2 - b^2)

theorem triangle_third_side_lengths_product :
  let a := 6
  let b := 8
  let hyp := hypotenuse a b
  let leg := leg b a
  real.round (hyp * leg * 10) / 10 = 52.9 :=
by {
  -- Definitions and calculations have been provided in the problem statement
  sorry
}

end triangle_third_side_lengths_product_l504_504741


namespace find_a4_l504_504425

noncomputable def sequence_a (n : ℕ) : ℤ := 
if n = 1 then 1 else if n = 2 then -2 else -2 * sequence_a (n - 1)

theorem find_a4 
  (S : ℕ → ℤ) 
  (h_condition : ∀ n ≥ 2, 2 * S (n-1) = S n + S (n+1))
  (h_S_def : ∀ n, S (n+1) = S n + sequence_a (n+1)) 
  (h_a2 : sequence_a 2 = -2) : 
  sequence_a 4 = -8 :=
sorry

end find_a4_l504_504425


namespace value_of_expression_l504_504279

theorem value_of_expression : 3 ^ (0 ^ (2 ^ 11)) + ((3 ^ 0) ^ 2) ^ 11 = 2 := by
  sorry

end value_of_expression_l504_504279


namespace count_permissible_arrangements_l504_504891

structure Sibling (α : Type) := (left : α) (right : α)

def is_permissible_arrangement {α : Type} [DecidableEq α] (first_row second_row : list α) (siblings : list (Sibling α)) : Prop :=
  first_row.length = second_row.length ∧
  first_row.length = siblings.length ∧
  (∀ (sib : Sibling α), (sib.left ∈ first_row → sib.right ∉ first_row ∧ list.indexOf sib.left first_row ≠ list.indexOf sib.right first_row - 4) ∧
                         (sib.right ∈ first_row → sib.left ∉ first_row ∧ list.indexOf sib.right first_row ≠ list.indexOf sib.left first_row + 4))

def sibling_swap_factor (n : ℕ) : ℕ :=
  2 ^ n

def derangement_count (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 0
  | 2     => 1
  | 3     => 2
  | 4     => 9
  | _     => sorry  -- Derangement function for arbitrary n needs proof

theorem count_permissible_arrangements : 4! * derangement_count 4 * sibling_swap_factor 4 = 3456 :=
by
  -- Proof steps to show the count is 3456
  sorry

end count_permissible_arrangements_l504_504891


namespace max_total_time_l504_504597

theorem max_total_time :
  ∀ (time_mowing time_fertilizing total_time : ℕ), 
    time_mowing = 40 ∧ time_fertilizing = 2 * time_mowing ∧ total_time = time_mowing + time_fertilizing → 
    total_time = 120 :=
by
  intros time_mowing time_fertilizing total_time h
  have h1: time_mowing = 40 := h.1
  have h2: time_fertilizing = 2 * time_mowing := h.2.1
  have h3: total_time = time_mowing + time_fertilizing := h.2.2
  rw[h1] at h2
  rw[h1, h2] at h3
  simp at h3
  exact h3.symm

end max_total_time_l504_504597


namespace prob_six_games_170_729_prob_dist_X_correct_expected_value_X_326_81_l504_504270

noncomputable def probability_six_games : ℚ :=
  let p_A_win := 1 / 3 in
  let p_B_win := 2 / 3 in
  (binomial 5 4) * (p_A_win ^ 4) * (p_B_win) * (p_A_win) +
  (binomial 5 4) * (p_B_win ^ 5) * (p_A_win)
  
theorem prob_six_games_170_729 : probability_six_games = 170 / 729 :=
  sorry

noncomputable def prob_dist_X : list (ℚ × ℚ) :=
  [(2, 1 / 9), (3, 4 / 27), (4, 28 / 81), (5, 32 / 81)]

noncomputable def expected_value_X : ℚ :=
  2 * (1 / 9) + 3 * (4 / 27) + 4 * (28 / 81) + 5 * (32 / 81)

theorem prob_dist_X_correct : prob_dist_X = [(2, 1 / 9), (3, 4 / 27), (4, 28 / 81), (5, 32 / 81)] :=
  sorry

theorem expected_value_X_326_81 : expected_value_X = 326 / 81 :=
  sorry

end prob_six_games_170_729_prob_dist_X_correct_expected_value_X_326_81_l504_504270


namespace smallest_period_and_max_value_intervals_of_monotonicity_l504_504011

noncomputable def f (x : ℝ) : ℝ := sin (x - 3 * Real.pi / 2) * sin x - sqrt 3 * cos x ^ 2

theorem smallest_period_and_max_value :
  (∀ x, f (x + Real.pi) = f x) ∧ (∀ x, f x ≤ 1 - sqrt 3 / 2) :=
by
  -- (proof omitted)
  sorry

theorem intervals_of_monotonicity :
  (∀ x ∈ Icc (Real.pi / 6) (5 * Real.pi / 12), 
    ∀ y ∈ Icc (Real.pi / 6) (5 * Real.pi / 12), x < y → f x < f y) ∧
  (∀ x ∈ Icc (5 * Real.pi / 12) (2 * Real.pi / 3), 
    ∀ y ∈ Icc (5 * Real.pi / 12) (2 * Real.pi / 3), x < y → f x > f y) :=
by
  -- (proof omitted)
  sorry

end smallest_period_and_max_value_intervals_of_monotonicity_l504_504011


namespace simplify_f_value_of_f_l504_504417

noncomputable def f (α : ℝ) : ℝ := 
  (Real.sin (π - α) * Real.cos (2 * π - α) * Real.tan (- α + 3 * π / 2)) / 
  (Real.cot (- α - π) * Real.sin (- π - α))

-- Proof that f(α) = -cos(α)
theorem simplify_f (α : ℝ) : f(α) = -Real.cos α := 
  sorry

-- Given α is an angle in the third quadrant and cos(α - 3π/2) = 1/5, proof f(α) = 2√6/5 
theorem value_of_f (α : ℝ) (h1 : 3 * π / 2 < α ∧ α < 2 * π)
  (h2 : Real.cos (α - 3 * π / 2) = 1 / 5) : f(α) = 2 * Real.sqrt 6 / 5 := 
  sorry

end simplify_f_value_of_f_l504_504417


namespace smallest_sum_of_20_consecutive_integers_twice_perfect_square_l504_504257

theorem smallest_sum_of_20_consecutive_integers_twice_perfect_square :
  ∃ n : ℕ, ∃ k : ℕ, (∀ m : ℕ, m ≥ n → 0 < m) ∧ 10 * (2 * n + 19) = 2 * k^2 ∧ 10 * (2 * n + 19) = 450 :=
by
  sorry

end smallest_sum_of_20_consecutive_integers_twice_perfect_square_l504_504257


namespace prove_percent_liquid_X_in_new_solution_l504_504828

variable (initial_solution total_weight_x total_weight_y total_weight_new)

def percent_liquid_X_in_new_solution : Prop :=
  let liquid_X_in_initial := 0.45 * 12
  let water_in_initial := 0.55 * 12
  let remaining_liquid_X := liquid_X_in_initial
  let remaining_water := water_in_initial - 5
  let liquid_X_in_added := 0.45 * 7
  let water_in_added := 0.55 * 7
  let total_liquid_X := remaining_liquid_X + liquid_X_in_added
  let total_water := remaining_water + water_in_added
  let total_weight := total_liquid_X + total_water
  (total_liquid_X / total_weight) * 100 = 61.07

theorem prove_percent_liquid_X_in_new_solution :
  percent_liquid_X_in_new_solution := by
  sorry

end prove_percent_liquid_X_in_new_solution_l504_504828


namespace trigonometric_identity_l504_504898

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -2) : 
  2 * Real.sin α * Real.cos α - (Real.cos α)^2 = -1 := 
by
  sorry

end trigonometric_identity_l504_504898


namespace tan_double_angle_l504_504471

theorem tan_double_angle (α : ℝ) (h : 3 * Real.cos α + Real.sin α = 0) : 
    Real.tan (2 * α) = 3 / 4 := 
by
  sorry

end tan_double_angle_l504_504471


namespace b1_plus_b2_l504_504871

-- Define the Fibonacci sequence
def fibonacci (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ ∀ n, a (n+2) = a (n+1) + a n

-- Sequence b satisfying the given conditions
def sequence_b (a b : ℕ → ℕ) : Prop :=
  (∀ n, b (n+3) + (-1) ^ (a n) * b n = n) ∧
  (∑ i in finset.range 12, b (i+1) = 86)

-- Prove that b1 + b2 = 8
theorem b1_plus_b2 {a b : ℕ → ℕ} :
  fibonacci a →
  sequence_b a b →
  b 1 + b 2 = 8 :=
by
  sorry

end b1_plus_b2_l504_504871


namespace count_4_primable_numbers_lt_1000_l504_504342

def is_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_n_primable (n k : ℕ) : Prop :=
  k % n = 0 ∧ ∀ d, d ∈ (k.digits 10) → is_digit_prime d

def is_4_primable (k : ℕ) : Prop :=
  is_n_primable 4 k

theorem count_4_primable_numbers_lt_1000 : 
  ∃ n, n = 21 ∧ n = (Finset.filter is_4_primable (Finset.range 1000)).card :=
sorry

end count_4_primable_numbers_lt_1000_l504_504342


namespace count_4_primable_numbers_lt_1000_l504_504341

def is_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_n_primable (n k : ℕ) : Prop :=
  k % n = 0 ∧ ∀ d, d ∈ (k.digits 10) → is_digit_prime d

def is_4_primable (k : ℕ) : Prop :=
  is_n_primable 4 k

theorem count_4_primable_numbers_lt_1000 : 
  ∃ n, n = 21 ∧ n = (Finset.filter is_4_primable (Finset.range 1000)).card :=
sorry

end count_4_primable_numbers_lt_1000_l504_504341


namespace book_readings_statements_l504_504030

-- Define the list of numbers representing the books Mia read
def book_readings := [4, 0, 1, 0, 4, 4, 0, 1, 4]

-- Prove that median < mean < mode
theorem book_readings_statements :
  let mean := (book_readings.sum : ℝ) / book_readings.length
  let sorted_readings := book_readings.qsort (≤)
  let median := sorted_readings.nth_le (sorted_readings.length / 2) sorry
  let mode := (book_readings.foldr (λ x counts, counts.insertWith (+) x 1) ∅).max_dfst sorry in
  median < mean ∧ mean < mode := 
sorry

end book_readings_statements_l504_504030


namespace max_self_crossings_2010_gon_l504_504905

-- Problem Statement:
-- Given a convex 2010-gon with no three diagonals intersecting except at vertices,
-- If a closed broken line has 2010 diagonals passing through each vertex exactly once,
-- Prove that the maximum number of self-crossings is 2016031.

theorem max_self_crossings_2010_gon :
  ∀ (polygon : polygonal) (vertices_cnt : ℕ)
    (convex : is_convex polygon)
    (no_intersect_diagonals : ∀ (d1 d2 d3 : diagonal),
      distinct_vertices d1 d2 d3 → non_intersect_vertices d1 d2 d3)
    (closed_broken_lines : list broken_line)
    (n : ℕ),
    vertices_cnt = 2010 →
    ∀ l ∈ closed_broken_lines, count_diagonals l = n →
    closed l →
    (∃ l, count_self_crossings l ≤ 2016031) := by
  sorry

end max_self_crossings_2010_gon_l504_504905


namespace smallest_p_condition_l504_504778

theorem smallest_p_condition (n p : ℕ) (hn1 : n % 2 = 1) (hn2 : n % 7 = 5) (hp : (n + p) % 10 = 0) : p = 1 := by
  sorry

end smallest_p_condition_l504_504778


namespace machine_A_produces_2_sprockets_per_hour_l504_504157

def A : ℝ := 220 / (((220 / A) + 10) * (1.1 * A))

theorem machine_A_produces_2_sprockets_per_hour 
  (A: ℝ) (B: ℝ) (T_A: ℝ) (T_B: ℝ)
  (h1: B = 1.1 * A)
  (h2: T_A = T_B + 10)
  (h3: 220 = A * T_A)
  (h4: 220 = B * T_B) :
  A = 2 :=
by
  sorry

end machine_A_produces_2_sprockets_per_hour_l504_504157


namespace tens_place_of_smallest_even_number_is_8_l504_504747

def digits := [1, 3, 5, 6, 8]

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_valid_number (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ,
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  n = a * 10000 + b * 1000 + c * 100 + d * 10 + e

noncomputable def smallest_even_number : ℕ :=
  (list.permutations digits).map (λ l, l.head! * 10000 + l.tail.head! * 1000 + l.tail.tail.head! * 100 + l.tail.tail.tail.head! * 10 + l.tail.tail.tail.tail.head!)
  |>.filter (λ n, is_even n ∧ is_valid_number n)
  |>.min

def tens_place_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

theorem tens_place_of_smallest_even_number_is_8 :
  tens_place_digit smallest_even_number = 8 :=
by
  sorry

end tens_place_of_smallest_even_number_is_8_l504_504747


namespace complex_exponentiation_l504_504845

def de_moivres_theorem (θ : ℝ) (n : ℕ) : ℂ :=
  (complex.cos θ + complex.sin θ * complex.I)^n

theorem complex_exponentiation :
  de_moivres_theorem 185 54 = -complex.I := by
sorry

end complex_exponentiation_l504_504845


namespace cos_double_angle_of_perpendicular_line_l504_504920

theorem cos_double_angle_of_perpendicular_line (α : ℝ) 
  (h_perpendicular : ∀ {l m : ℝ}, (l = tan α) ∧ (m = -2) → tan α * -2 = -1) : 
  cos (2 * α) = -3/5 := 
by 
  sorry

end cos_double_angle_of_perpendicular_line_l504_504920


namespace _l504_504118

lemma angle_y_value (A B C D : Point) 
  (h1 : collinear A B C)
  (h2 : ∠ ABD = 148)
  (h3 : ∠ BCD = 52)
  (h4 : exterior_angle_theorem : ∀ {a b c d : Point}, ∠ ABD = ∠ BCD + ∠ BDC) :
  ∠ BDC = 96 :=
by
  sorry

end _l504_504118


namespace trig_inequality_2016_l504_504145

theorem trig_inequality_2016 :
  let a := Real.sin (Real.cos (2016 * Real.pi / 180))
  let b := Real.sin (Real.sin (2016 * Real.pi / 180))
  let c := Real.cos (Real.sin (2016 * Real.pi / 180))
  let d := Real.cos (Real.cos (2016 * Real.pi / 180))
  c > d ∧ d > b ∧ b > a := by
  sorry

end trig_inequality_2016_l504_504145


namespace evaluate_f_f_neg2_l504_504930

def f (x : ℝ) : ℝ :=
  if x >= 0 then 1 - real.sqrt x else 2^x

theorem evaluate_f_f_neg2 : f (f (-2)) = 1 / 2 :=
by sorry

end evaluate_f_f_neg2_l504_504930


namespace find_line_eq_l504_504017

-- Definitions of the given hyperbola and midpoint conditions
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 2 = 1

def midpoint (x1 y1 x2 y2: ℝ) : Prop := (x1 + x2 = 1) ∧ (y1 + y2 = -2)

-- The main theorem proving that the equation of the line is 2x + 8y + 7 = 0
theorem find_line_eq (x1 y1 x2 y2 : ℝ) (h1 : hyperbola x1 y1) (h2 : hyperbola x2 y2) 
    (h_mid : midpoint x1 y1 x2 y2) : ∃ (a b c : ℝ), a = 2 ∧ b = 8 ∧ c = 7 ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 :=
by
  -- The proof would demonstrate that the line equation is 2x + 8y + 7 = 0
  sorry

end find_line_eq_l504_504017


namespace prime_numbers_between_50_and_70_l504_504057

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_between_50_and_70 : 
  (finset.filter is_prime (finset.range 71)).count (λ n, 50 ≤ n ∧ n ≤ 70) = 4 := 
sorry

end prime_numbers_between_50_and_70_l504_504057


namespace right_angled_triangle_l504_504097

-- Definitions of the sides of the triangle and the given equations
variables {a b c x_0 : ℝ}
axiom eq1 : x_0^2 + 2 * a * x_0 + b^2 = 0
axiom eq2 : x_0^2 + 2 * c * x_0 - b^2 = 0
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0

-- The final theorem to prove
theorem right_angled_triangle (h1 : eq1) (h2 : eq2) (h3 : triangle_sides) : a^2 = b^2 + c^2 := 
sorry

end right_angled_triangle_l504_504097


namespace right_triangle_third_side_product_l504_504734

theorem right_triangle_third_side_product :
  let a := 6
  let b := 8
  let c1 := Real.sqrt (a^2 + b^2)     -- Hypotenuse when a and b are legs
  let c2 := Real.sqrt (b^2 - a^2)     -- Other side when b is the hypotenuse
  20 * Real.sqrt 7 ≈ 52.7 := 
by
  sorry

end right_triangle_third_side_product_l504_504734


namespace a_2017_eq_2_l504_504944

-- Sequence definition and recurrence relation
noncomputable def a : ℕ → ℚ
| 0     := 2
| (n+1) := 1 - 1 / a n

-- The statement to be proven
theorem a_2017_eq_2 : a 2016 = 2 :=
sorry

end a_2017_eq_2_l504_504944


namespace kids_ticket_price_is_correct_l504_504204

noncomputable def price_of_kids_ticket (A: ℝ) : ℝ :=
  let K := 0.5 * A in
  let adults_discounted := 4 * A * 0.9 in
  let kids_cost := 6 * K in
  let total_before_tax := adults_discounted + kids_cost in
  let total_with_tax := total_before_tax * 1.05 in
  let premium_charge := 2 * 10 in
  let total_cost := total_with_tax + premium_charge in
  if total_cost = 100 then K + 2 else 0

#eval price_of_kids_ticket (100 / (6.93)) -- We determined A to be approximately 11.54 from the previous calculations

theorem kids_ticket_price_is_correct : price_of_kids_ticket (80 / 6.93) = 7.77 := 
by {
  sorry
}

end kids_ticket_price_is_correct_l504_504204


namespace function_properties_l504_504291

noncomputable def f (x : ℝ) : ℝ := x^2

theorem function_properties :
  (∀ x1 x2 : ℝ, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x : ℝ, 0 < x → deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end function_properties_l504_504291


namespace domain_of_f_not_symmetric_about_x_1_f_of_f_neg5_range_of_f_l504_504449

noncomputable def f (x : ℝ) : ℝ := 2 / (|x| - 1)

theorem domain_of_f : ∀ x : ℝ, x ≠ 1 → x ≠ -1 → (|x| - 1 ≠ 0) :=
by
  intro x h1 h2
  apply ne_of_ne_of_eq
  exact h1
  exact h2
  sorry

theorem not_symmetric_about_x_1 : ¬ (∀ x : ℝ, f(x + 1) = f(1 - x)) :=
by
  intro h
  specialize h 0
  have := f(0)
  have := f(2)
  sorry

theorem f_of_f_neg5 : f (f (-5)) = -4 :=
by
  have h1 : f (-5) = 2 / (|(-5)| - 1) := rfl
  have h2 : |(-5)| = 5 := abs_neg 5
  have h3 : 5 - 1 = 4 := sub_self 4
  have h4 : f (1/2) = -4 := rfl
  sorry

theorem range_of_f : (set.range f) = set.union (set.Iic (-2)) (set.Ioi 0):=
by
  sorry

end domain_of_f_not_symmetric_about_x_1_f_of_f_neg5_range_of_f_l504_504449


namespace constant_term_expansion_eq_24_l504_504215

theorem constant_term_expansion_eq_24 :
  let a := (2 : ℝ) * X
  let b := (1 : ℝ) / X
  let n := 4
  (∀ X : ℝ, (2 * X + 1 / X)^4).constant_term = 24 :=
by
  sorry

end constant_term_expansion_eq_24_l504_504215


namespace product_of_possible_lengths_approx_l504_504739

noncomputable def hypotenuse (a b : ℝ) : ℝ :=
  real.sqrt (a * a + b * b)

noncomputable def other_leg (hypotenuse a : ℝ) : ℝ :=
  real.sqrt (hypotenuse * hypotenuse - a * a)

noncomputable def product_of_possible_lengths (a b : ℝ) : ℝ :=
  hypotenuse a b * other_leg (max a b) (min a b)

theorem product_of_possible_lengths_approx (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  Float.round (product_of_possible_lengths a b) 1 = 52.7 :=
by
  sorry

end product_of_possible_lengths_approx_l504_504739


namespace exterior_angle_regular_octagon_l504_504991

theorem exterior_angle_regular_octagon : 
  ∀ {θ : ℝ}, 
  (8 - 2) * 180 / 8 = θ →
  180 - θ = 45 := 
by 
  intro θ hθ
  sorry

end exterior_angle_regular_octagon_l504_504991


namespace range_of_abs_function_l504_504846

theorem range_of_abs_function : ∀ (y : ℝ), (∃ (x : ℝ), y = |x + 5| - |x - 3|) ↔ y ∈ Set.Icc (-8) 8 :=
by
  sorry

end range_of_abs_function_l504_504846


namespace master_craftsman_parts_l504_504518

/-- 
Given:
  (1) the master craftsman produces 35 parts in the first hour,
  (2) at the rate of 35 parts/hr, he would be one hour late to meet the quota,
  (3) by increasing his speed by 15 parts/hr, he finishes the quota 0.5 hours early,
Prove that the total number of parts manufactured during the shift is 210.
-/
theorem master_craftsman_parts (N : ℕ) (quota : ℕ) 
  (initial_rate : ℕ := 35)
  (increased_rate_diff : ℕ := 15)
  (extra_time_slow : ℕ := 1)
  (time_saved_fast : ℕ := 1/2) :
  (quota = initial_rate * (extra_time_slow + 1) + N ∧
   increased_rate_diff = 15 ∧
   increased_rate_diff = λ (x : ℕ), initial_rate + x ∧
   time_saved_fast = 1/2 ∧
   N = 35) →
  quota = 210 := 
by
  sorry

end master_craftsman_parts_l504_504518


namespace max_expected_expenditure_l504_504478

theorem max_expected_expenditure
  (a b : ℝ) (ε : ℝ) (x : ℝ)
  (h_a : a = 2)
  (h_b : b = 0.8)
  (h_eps : |ε| ≤ 0.5)
  (h_x : x = 10) :
  let y := a + b * x + ε in y ≤ 10.5 :=
by
  sorry

end max_expected_expenditure_l504_504478


namespace prime_count_between_50_and_70_l504_504070

open Nat

theorem prime_count_between_50_and_70 : 
  (finset.filter Nat.prime (finset.range 71 \ finset.range 51).card = 4) := 
begin
  sorry
end

end prime_count_between_50_and_70_l504_504070


namespace number_of_digits_in_sum_l504_504474

theorem number_of_digits_in_sum
  (A B : ℕ)
  (hA : 1 ≤ A ∧ A ≤ 9)
  (hB : 1 ≤ B ∧ B ≤ 9) :
  nat.digits 10 (9876 + A * 100 + 32 + B * 10 + 1) = [5] :=
sorry

end number_of_digits_in_sum_l504_504474


namespace intersection_M_N_l504_504414

def set_M : Set (ℝ × ℝ) := {p | ∃ x : ℝ, p = (x, x^2 + 1)}
def set_N : Set (ℝ × ℝ) := {p | ∃ x : ℝ, p = (x, x + 1)}

theorem intersection_M_N : set_M ∩ set_N = {(0 : ℝ, 1 : ℝ), (1 : ℝ, 2 : ℝ)} :=
by
  sorry

end intersection_M_N_l504_504414


namespace simplify_sqrt_expression_l504_504653

theorem simplify_sqrt_expression :
  √(12 + 8 * √3) + √(12 - 8 * √3) = 4 * √3 :=
sorry

end simplify_sqrt_expression_l504_504653


namespace simplify_sqrt_sum_l504_504647

noncomputable def sqrt_expr_1 : ℝ := Real.sqrt (12 + 8 * Real.sqrt 3)
noncomputable def sqrt_expr_2 : ℝ := Real.sqrt (12 - 8 * Real.sqrt 3)

theorem simplify_sqrt_sum : sqrt_expr_1 + sqrt_expr_2 = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_sum_l504_504647


namespace simplify_nested_sqrt_l504_504638

-- Define the expressions under the square roots
def expr1 : ℝ := 12 + 8 * real.sqrt 3
def expr2 : ℝ := 12 - 8 * real.sqrt 3

-- Problem statement to prove
theorem simplify_nested_sqrt : real.sqrt expr1 + real.sqrt expr2 = 4 * real.sqrt 2 :=
by
  sorry

end simplify_nested_sqrt_l504_504638


namespace trajectory_eq_is_ellipse_ratio_DE_AB_l504_504438

-- Define the conditions and constants
def circle := { p : ℝ × ℝ | (p.1)^2 + (p.2)^2 = 18 }
def perpendicular (P Q : ℝ × ℝ) : Prop := Q.2 = 0
def vector_relation (O M P Q : ℝ × ℝ) : Prop :=
  (M.1, M.2) = (1/3 * P.1 + 2/3 * Q.1, 1/3 * P.2 + 2/3 * Q.2)

-- Define trajectory equation of M
def trajectory_eq (M : ℝ × ℝ) : Prop :=
  (M.1)^2 / 18 + (M.2)^2 / 2 = 1

-- Define the line equation and its intersections
def line_eq (M : ℝ) (y : ℝ) : ℝ := M * y - 4
def intersect_points (A B : ℝ × ℝ) (C : ℝ × ℝ) (m : ℝ) : Prop :=
  A.1 = line_eq m A.2 ∧ B.1 = line_eq m B.2 ∧
  A ∈ C ∧ B ∈ C

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def perpendicular_bisector_eq (A B : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ D, D.2 = (- (B.1 - A.1) / (B.2 - A.2)) * (D.1 - midpoint A B).1 + midpoint A B.2

def x_intersect (A B : ℝ × ℝ) : ℝ :=
  (-32 / ((B.1)^2 + 9))

-- Prove the trajectory is the given ellipse
theorem trajectory_eq_is_ellipse (P Q : ℝ × ℝ) (hP : P ∈ circle) (hQ : perpendicular P Q) (M : ℝ × ℝ)
    (hV : vector_relation (0, 0) M P Q) :
  trajectory_eq M :=
sorry

-- Prove the ratio DE/AB is sqrt(2)/3
theorem ratio_DE_AB (E : ℝ × ℝ) (hE : E = (-4, 0)) (A B : ℝ × ℝ)
    (hM : ∀ m ≠ 0, intersect_points A B (0,0) m) :
  let D := (x_intersect A B, 0) in
  |E.1 - D.1| / dist A B = (Real.sqrt 2) / 3 :=
sorry

end trajectory_eq_is_ellipse_ratio_DE_AB_l504_504438


namespace geometric_sequence_common_ratio_l504_504883

-- Definitions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a 1 * q^n

def sum_terms (a : ℕ → ℝ) : ℕ → ℝ 
| 0       := a 0
| (n + 1) := a (n + 1) + sum_terms n

-- Problem statement
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) : 
  geometric_sequence a q → 
  (a 1 * (2 * q + 1 + q^2) = 0) → 
  q = -1 := 
by {
  intros h_geom h_cond,
  sorry
}

end geometric_sequence_common_ratio_l504_504883


namespace place_numbers_1_to_10_in_table_l504_504607

def exists_balanced_table (s : ℕ) (arr : Fin 3 → Fin 3 → ℕ) : Prop :=
  arr (Fin.mk 0 (by simp)) (Fin.mk 0 (by simp)) +
  arr (Fin.mk 0 (by simp)) (Fin.mk 1 (by simp)) +
  arr (Fin.mk 0 (by simp)) (Fin.mk 2 (by simp)) = s ∧
  arr (Fin.mk 1 (by simp)) (Fin.mk 0 (by simp)) +
  arr (Fin.mk 1 (by simp)) (Fin.mk 1 (by simp)) +
  arr (Fin.mk 1 (by simp)) (Fin.mk 2 (by simp)) = s ∧
  arr (Fin.mk 2 (by simp)) (Fin.mk 0 (by simp)) +
  arr (Fin.mk 2 (by simp)) (Fin.mk 1 (by simp)) +
  arr (Fin.mk 2 (by simp)) (Fin.mk 2 (by simp)) +
  arr (Fin.mk 3 (by simp)) (Fin.mk 0 (by simp)) +
  arr (Fin.mk 3 (by simp)) (Fin.mk 1 (by simp)) +
  arr (Fin.mk 3 (by simp)) (Fin.mk 2 (by simp)) = s

theorem place_numbers_1_to_10_in_table :
  ∃ (arr : Fin 3 → Fin 3 → ℕ) (s : ℕ), (∀ i j, arr i j ∈ Finset.range 10) ∧
  exists_balanced_table s arr :=
sorry

end place_numbers_1_to_10_in_table_l504_504607


namespace minimum_weight_each_crate_l504_504352

-- Define constants and conditions
constant crates_trip1 : Nat := 3
constant crates_trip2 : Nat := 4
constant crates_trip3 : Nat := 5
constant max_weight : Nat := 750

-- Define theorem statement
theorem minimum_weight_each_crate :
  ∃ w : Nat, w = max_weight / crates_trip3 := by
  use 150
  sorry

end minimum_weight_each_crate_l504_504352


namespace abs_distance_equation_1_abs_distance_equation_2_l504_504273

theorem abs_distance_equation_1 (x : ℚ) : |x - (3 : ℚ)| = 5 ↔ x = 8 ∨ x = -2 := 
sorry

theorem abs_distance_equation_2 (x : ℚ) : |x - (3 : ℚ)| = |x + (1 : ℚ)| ↔ x = 1 :=
sorry

end abs_distance_equation_1_abs_distance_equation_2_l504_504273


namespace find_f_of_3_l504_504475

theorem find_f_of_3 : 
  (∀ x : ℝ, f (x + 1 / x) = x^2 + (1 / x)^2) → f 3 = 7 :=
by
  sorry

end find_f_of_3_l504_504475


namespace function_properties_l504_504290

noncomputable def f (x : ℝ) : ℝ := x^2

theorem function_properties :
  (∀ x1 x2 : ℝ, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x : ℝ, 0 < x → deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end function_properties_l504_504290


namespace range_of_abs_function_l504_504847

theorem range_of_abs_function : ∀ (y : ℝ), (∃ (x : ℝ), y = |x + 5| - |x - 3|) ↔ y ∈ Set.Icc (-8) 8 :=
by
  sorry

end range_of_abs_function_l504_504847


namespace sequence_sum_l504_504426

/--
Let \(S_n\) be the sum of the first \(n\) terms of the sequence \(\{a_n\}\).
Given that \(S_n = n^2 - 1\) for \(n \in \mathbb{N}_{+}\), prove that
\(a_1 + a_3 + a_5 + a_7 + a_9 = 44\).
-/
theorem sequence_sum :
  (∀ n : ℕ, 0 < n → S n = n^2 - 1) →
  let a (n : ℕ) := if n = 0 then 0 else S n - S (n - 1) in
  a 1 + a 3 + a 5 + a 7 + a 9 = 44 :=
by
  intro hS
  have S_zero : S 0 = 0 := by sorry
  have a_def : ∀ n, a n = if n = 0 then 0 else S n - S (n - 1) := by sorry
  rw [a_def 1, a_def 3, a_def 5, a_def 7, a_def 9]
  simp
  sorry

end sequence_sum_l504_504426


namespace area_of_triangle_is_correct_l504_504228

noncomputable def area_of_triangle_with_roots (a b c : ℝ) : ℝ :=
  let q := (a + b + c) / 2
  q * (q - a) * (q - b) * (q - c)

theorem area_of_triangle_is_correct (a b c : ℝ) (h : a + b + c = 3)
    (e : (3 / 2 - a) * (3 / 2 - b) * (3 / 2 - c) = 1 / 5) :
  area_of_triangle_with_roots a b c = real.sqrt (3 / 10) := by
  sorry

end area_of_triangle_is_correct_l504_504228


namespace percentage_difference_in_gain_correct_l504_504826

-- Define the conditions
def selling_price1 : ℝ := 360
def selling_price2 : ℝ := 340
def cost_price : ℝ := 400

-- Calculate gains and the percentage difference in gain
def gain1 : ℝ := selling_price1 - cost_price
def gain2 : ℝ := selling_price2 - cost_price
def difference_in_gain : ℝ := gain1 - gain2
def percentage_difference_in_gain : ℝ := (difference_in_gain / cost_price) * 100

-- The statement we need to prove
theorem percentage_difference_in_gain_correct :
  percentage_difference_in_gain = 5 := by
  sorry

end percentage_difference_in_gain_correct_l504_504826


namespace log_equation_solution_l504_504473

theorem log_equation_solution : 
  (∀ (x : ℝ), (2^3 * 3^(x + 1) = 48)) ∧ (log 2 = 0.3010) ∧ (log 3 = 0.4771) → 
  x = 0.63 := 
by
  sorry

end log_equation_solution_l504_504473


namespace median_of_list_of_numbers_l504_504276

def list_of_numbers : List ℕ :=
  (List.range 925).map(λ x, x + 1) ++ (List.range 925).map(λ x, (x + 1) ^ 2)

def is_sorted (l : List ℕ) : Prop :=
  ∀ i j, i < j → l.nth i ≤ l.nth j

def median (l : List ℕ) : ℝ :=
  let len := l.length
  if len % 2 = 0 then
    let mid1 := l.nth (len / 2 - 1)
    let mid2 := l.nth (len / 2)
    (mid1.getD 0 + mid2.getD 0) / 2
  else
    l.nth (len / 2).getD 0

theorem median_of_list_of_numbers :
  median list_of_numbers = 895.5 :=
by {
  sorry
}

end median_of_list_of_numbers_l504_504276


namespace at_least_one_real_root_l504_504614

theorem at_least_one_real_root (m : ℝ) :
  ∃ x : ℝ, (x^2 - 5 * x + m = 0) ∨ (2 * x^2 + x + 6 - m = 0) :=
begin
  sorry
end

end at_least_one_real_root_l504_504614


namespace area_under_curve_is_pi_div_16_l504_504901

section
  open Real
  
  /-- Define the function g(x) -/
  def g (x : ℝ) : ℝ := sqrt (x * (1 - x))
  
  /-- Function f(x) to be integrated -/
  def f (x : ℝ) : ℝ := x * g x 
  
  /-- Statement of the problem -/
  theorem area_under_curve_is_pi_div_16 :
    ∫ x in 0..1, f x = π / 16 :=
  by
    sorry
end

end area_under_curve_is_pi_div_16_l504_504901


namespace not_two_thousand_in_A_num_sets_with_property_P_l504_504001

-- Define the conditions and the function for Property P
def has_property_P (A : Finset ℕ) : Prop :=
  ∀ m n ∈ (Finset.range 1000).map Finset.coeSort, (A m) + (A n) ∈ A

-- Define the set A satisfying property P
example (A : Finset ℕ) (h : A = (Finset.range 1000).map Finset.coeSort) : has_property_P A :=
  by sorry

-- Prove that 2000 ∉ A for any A with property P
theorem not_two_thousand_in_A (A : Finset ℕ) (hP : has_property_P A) : 2000 ∉ A :=
  by sorry

-- Find the number of sets A with property P
theorem num_sets_with_property_P : ∃ n, n = 2^19 ∧
  ∀ A : Finset ℕ, (has_property_P A ↔ (∃ B : Finset ℕ, B.card ≤ 19 ∧ A = (Finset.range 1000).map Finset.coeSort ∪ B)) :=
  by sorry

end not_two_thousand_in_A_num_sets_with_property_P_l504_504001


namespace correct_weight_of_misread_boy_l504_504672

variable (num_boys : ℕ) (avg_weight_incorrect : ℝ) (misread_weight : ℝ) (avg_weight_correct : ℝ)

theorem correct_weight_of_misread_boy
  (h1 : num_boys = 20)
  (h2 : avg_weight_incorrect = 58.4)
  (h3 : misread_weight = 56)
  (h4 : avg_weight_correct = 58.6) : 
  misread_weight + (num_boys * avg_weight_correct - num_boys * avg_weight_incorrect) / num_boys = 60 := 
by 
  -- skipping proof
  sorry

end correct_weight_of_misread_boy_l504_504672


namespace correct_vector_equation_l504_504280

-- Defining vector spaces and vector addition
variable {V : Type*} [AddCommGroup V] [Module ℝ V] -- Assuming vector space V

-- Define points A, B, C, D and their respective vectors
variables (A B C D : V)

-- Definitions based on given conditions
def vector_eq_A_to_B_C_D : Prop := 
  ∀ (A B C D : V), 
  (\overrightarrow {AB} = B - A) → 
  (\overrightarrow {BC} = C - B) → 
  (\overrightarrow {CD} = D - C) → 
  \overrightarrow {AB} + \overrightarrow {BC} + \overrightarrow {CD} = \overrightarrow {AD}

-- Proving the mathematically equivalent proof problem
theorem correct_vector_equation : vector_eq_A_to_B_C_D A B C D :=
by
  -- Use the given properties and vector addition rules
  intros A B C D h_AB h_BC h_CD
  simp [h_AB, h_BC, h_CD]
  sorry -- skipping the detailed proof steps

end correct_vector_equation_l504_504280


namespace verify_quadratic_eq_l504_504763

def is_quadratic (eq : String) : Prop :=
  eq = "ax^2 + bx + c = 0"

theorem verify_quadratic_eq :
  is_quadratic "x^2 - 1 = 0" :=
by
  -- Auxiliary functions or steps can be introduced if necessary, but proof is omitted here.
  sorry

end verify_quadratic_eq_l504_504763


namespace sum_f_powers_of_2_l504_504927

def f (x : ℝ) : ℝ := 4 * (log x / log 3) * log 2 + 233

theorem sum_f_powers_of_2 : 
  (∑ i in Finset.range 1 (← 9), f (2^i)) = 2241 :=
by
  sorry

end sum_f_powers_of_2_l504_504927


namespace tens_digit_36_pow_12_l504_504759

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def tens_digit (n : ℕ) : ℕ :=
  (last_two_digits n) / 10

theorem tens_digit_36_pow_12 : tens_digit (36^12) = 3 :=
by
  sorry

end tens_digit_36_pow_12_l504_504759


namespace primes_between_50_and_70_l504_504081

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).filter is_prime |>.length

theorem primes_between_50_and_70 : count_primes_in_range 50 70 = 4 :=
by
  sorry

end primes_between_50_and_70_l504_504081


namespace area_of_triangle_EOF_correct_l504_504242

noncomputable def area_of_triangle_eof : ℝ :=
  let line := λ x y : ℝ, x - 2 * y - 3 = 0
  let circle := λ x y : ℝ, (x - 2)^2 + (y + 3)^2 = 9
  let O := (0 : ℝ, 0 : ℝ)
  let E := sorry  -- placeholder for the intersection point
  let F := sorry  -- placeholder for the intersection point
  let OE := real.sqrt ((E.1 - O.1)^2 + (E.2 - O.2)^2)
  let OF := real.sqrt ((F.1 - O.1)^2 + (F.2 - O.2)^2)
  let EF := real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)
  let p := (OE + OF + EF) / 2
  real.sqrt (p * (p - OE) * (p - OF) * (p - EF))

theorem area_of_triangle_EOF_correct : area_of_triangle_eof = 6 * real.sqrt 5 / 5 := sorry

end area_of_triangle_EOF_correct_l504_504242


namespace union_A_B_complement_union_l504_504895

-- Define \( U \), \( A \), and \( B \)
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 2 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}

-- Define complement in the universe \( U \)
def complement_U (s : Set ℝ) : Set ℝ := {x | x ∉ s}

-- Statements to prove
theorem union_A_B : A ∪ B = {x | 2 ≤ x ∧ x ≤ 7} :=
  sorry

theorem complement_union : complement_U A ∪ complement_U B = {x | x < 3 ∨ x ≥ 5} :=
  sorry

end union_A_B_complement_union_l504_504895


namespace replacement_sequence_finite_l504_504160

-- Define the finite set of points and initial conditions.
noncomputable def points (M : Type) [Fintype M] (is_non_collinear : ∀ (A B C : M), ¬ collinear A B C) : Prop :=
  ∀ (P Q : M), ∃ (S : set (M × M)), (∀ x ∈ S, x.1 ≠ x.2) ∧
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y → disjointSg x y) ∧
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y → ¬(intersects x y && is_permitted_replacement x y))

-- Define the operation of replacing intersecting line segments
def replace_segments (A B C D : M) (intersects : (M × M) → (M × M) → Bool) : Prop :=
  intersects (A, B) (C, D) → ∃ (AC BD : set (M × M)),
  (AC = {(A, C)}) ∧ (BD = {(B, D)}) ∧ (length (A, C) + length (B, D) < length (A, B) + length (C, D))

-- Define the sum of the segment lengths and its behavior
def decreasing_sum (M : Type) [Fintype M] (S : set (M × M)) : Prop :=
  ∀ (x y : S), (length x.1 + length x.2) > (length y.1 + length y.2)

-- Main theorem
theorem replacement_sequence_finite
  (M : Type) [Fintype M] (is_non_collinear : ∀ (A B C : M), ¬ collinear A B C)
  (intersects : (M × M) → (M × M) → Bool)
  (replace : ∀ (A B C D : M), replace_segments A B C D intersects)
  (S : set (M × M))
  (finite_steps : decreasing_sum M S) :
  ∀ replacements : ℕ, replacements < Fintype.card M := sorry

end replacement_sequence_finite_l504_504160


namespace evaluate_product_l504_504387

theorem evaluate_product (m : ℕ) (h : m = 3) : (m - 2) * (m - 1) * m * (m + 1) * (m + 2) * (m + 3) = 720 :=
by {
  sorry
}

end evaluate_product_l504_504387


namespace constant_function_inequality_l504_504396

variable {R : Type} [LinearOrderedField R]

theorem constant_function_inequality (f : R → R) (h : ∀ x y z : R, f(x + y) + f(y + z) + f(z + x) ≥ 3 * f(x + 2 * y + 3 * z)) :
  ∃ c : R, ∀ x : R, f x = c :=
sorry

end constant_function_inequality_l504_504396


namespace cyclic_quadrilateral_IJ_length_l504_504113

-- Define cyclic quadrilateral and necessary properties
variable {A B C D I J : Type}
variable [CyclicQuadrilateral A B C D] -- Assume presence of necessary structures and properties
variable (AB AD AC : ℝ) -- lengths
variable (AB_eq_AD : AB = 49) 
variable (AC_eq_73 : AC = 73)
variable (I J : Point) -- incenters of triangles ABD and CBD
variable (BD_bisects_IJ : Bisects BD IJ)

-- Define the length of IJ
def length_IJ : ℝ := (28 * Real.sqrt 69) / 5

-- Lean statement to prove that under the given conditions, IJ has the specified length
theorem cyclic_quadrilateral_IJ_length :
  ∀ (A B C D I J : Point),
  CyclicQuadrilateral A B C D →
  AB = 49 →
  AD = 49 →
  AC = 73 →
  Incenter A B D I →
  Incenter C B D J →
  Bisects BD IJ →
  length_IJ = (28 * Real.sqrt 69) / 5 := 
sorry

end cyclic_quadrilateral_IJ_length_l504_504113


namespace books_read_in_common_l504_504717

theorem books_read_in_common (T D B total X : ℕ) 
  (hT : T = 23) 
  (hD : D = 12) 
  (hB : B = 17) 
  (htotal : total = 47)
  (h_eq : (T - X) + (D - X) + B + 1 = total) : 
  X = 3 :=
by
  -- Here would go the proof details.
  sorry

end books_read_in_common_l504_504717


namespace no_analytic_roots_l504_504880

theorem no_analytic_roots : ¬∃ x : ℝ, (x - 2) * (x + 5)^3 * (5 - x) = 8 := 
sorry

end no_analytic_roots_l504_504880


namespace tan_alpha_eq_four_thirds_l504_504896

variable (α : Real)

def condition1 : Prop := sin (π / 2 + α) = -3 / 5
def condition2 : Prop := α ∈ Set.Ioo (-π) 0

theorem tan_alpha_eq_four_thirds (h1 : condition1 α) (h2 : condition2 α) : tan α = 4 / 3 := 
  sorry

end tan_alpha_eq_four_thirds_l504_504896


namespace floor_abs_sum_l504_504668

theorem floor_abs_sum (x : Fin 2010 → ℝ) 
  (h : ∀ k, 1 ≤ k + 1 ∧ k < 2010 → x ⟨k, (Nat.div_lt_self k 2010 (Nat.zero_lt_succ _))⟩ + k + 1 = (Finset.univ.sum (λ i, x i) : ℝ) + 2011) :
  ⌊abs (Finset.univ.sum (λ (i : Fin 2010), x i))⌋ = 1005 :=
by sorry

end floor_abs_sum_l504_504668


namespace sum_of_digits_of_repeating_decimal_l504_504249

theorem sum_of_digits_of_repeating_decimal : 
  ∃ (c d : ℕ), 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ (5/13 : ℚ) = 0 + Nat.digits 10 (c + d*10).sum.to_rnor ≠ 0 → c + d = 11 := 
by sorry

end sum_of_digits_of_repeating_decimal_l504_504249


namespace master_craftsman_quota_l504_504524

theorem master_craftsman_quota (parts_first_hour : ℕ)
  (extra_hour_needed : ℕ)
  (increased_speed : ℕ)
  (time_diff : ℕ)
  (total_parts : ℕ) :
  parts_first_hour = 35 →
  extra_hour_needed = 1 →
  increased_speed = 15 →
  time_diff = 1.5 →
  total_parts = parts_first_hour + (175 : ℕ) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3]
  norm_num
  rw [add_comm]
  exact sorry

end master_craftsman_quota_l504_504524


namespace desserts_left_to_sell_l504_504316

-- Declare all the conditions as constants
constant cheesecakes_display : ℕ := 10
constant cheesecakes_fridge : ℕ := 15
constant cherry_pies_ready : ℕ := 12
constant cherry_pies_oven : ℕ := 20
constant chocolate_eclairs_counter : ℕ := 20
constant chocolate_eclairs_pantry : ℕ := 10

constant cheesecakes_sold : ℕ := 7
constant cherry_pies_sold : ℕ := 8
constant chocolate_eclairs_sold : ℕ := 10

-- Define the total number of desserts left in a structured manner
theorem desserts_left_to_sell : 
  (cheesecakes_display + cheesecakes_fridge - cheesecakes_sold) + 
  (cherry_pies_ready + cherry_pies_oven - cherry_pies_sold) + 
  (chocolate_eclairs_counter + chocolate_eclairs_pantry - chocolate_eclairs_sold) = 62 := 
by 
  -- Proof goes here
  sorry

end desserts_left_to_sell_l504_504316


namespace exterior_angle_regular_octagon_l504_504992

theorem exterior_angle_regular_octagon : ∀ (n : ℕ), n = 8 → (180 - (1080 / n)) = 45 :=
by
  intros n h
  rw h
  sorry

end exterior_angle_regular_octagon_l504_504992


namespace solution_set_of_inequality_l504_504007

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

theorem solution_set_of_inequality (H1 : f 1 = 1)
  (H2 : ∀ x : ℝ, x * f' x < 1 / 2) :
  {x : ℝ | f (Real.log x ^ 2) < (Real.log x ^ 2) / 2 + 1 / 2} = 
  {x : ℝ | 0 < x ∧ x < 1 / 10} ∪ {x : ℝ | x > 10} :=
sorry

end solution_set_of_inequality_l504_504007


namespace gym_apartment_ratio_l504_504225

theorem gym_apartment_ratio (dist_apartment_to_work : ℕ) (dist_apartment_to_gym : ℕ) :
  dist_apartment_to_work = 10 ∧
  dist_apartment_to_gym = 7 ∧
  dist_apartment_to_gym > dist_apartment_to_work / 2 →
  dist_apartment_to_gym * 10 = 7 * dist_apartment_to_work :=
by {
  sorry,
}

end gym_apartment_ratio_l504_504225


namespace master_craftsman_quota_l504_504527

theorem master_craftsman_quota (parts_first_hour : ℕ)
  (extra_hour_needed : ℕ)
  (increased_speed : ℕ)
  (time_diff : ℕ)
  (total_parts : ℕ) :
  parts_first_hour = 35 →
  extra_hour_needed = 1 →
  increased_speed = 15 →
  time_diff = 1.5 →
  total_parts = parts_first_hour + (175 : ℕ) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3]
  norm_num
  rw [add_comm]
  exact sorry

end master_craftsman_quota_l504_504527


namespace olga_fish_count_at_least_l504_504603

def number_of_fish (yellow blue green : ℕ) : ℕ :=
  yellow + blue + green

theorem olga_fish_count_at_least :
  ∃ (fish_count : ℕ), 
  (∃ (yellow blue green : ℕ), 
       yellow = 12 ∧ blue = yellow / 2 ∧ green = yellow * 2 ∧ fish_count = number_of_fish yellow blue green) ∧
  fish_count = 42 :=
by
  let yellow := 12
  let blue := yellow / 2
  let green := yellow * 2
  let fish_count := number_of_fish yellow blue green
  have h : fish_count = 42 := sorry
  use fish_count, yellow, blue, green
  repeat {constructor}
  assumption
  assumption
  assumption
  assumption
  assumption
  assumption

end olga_fish_count_at_least_l504_504603


namespace simplify_sqrt_sum_l504_504637

theorem simplify_sqrt_sum :
  sqrt (12 + 8 * sqrt 3) + sqrt (12 - 8 * sqrt 3) = 4 * sqrt 3 := 
by
  -- Proof would go here
  sorry

end simplify_sqrt_sum_l504_504637


namespace right_triangle_side_product_l504_504721

theorem right_triangle_side_product :
  let a := 6
  let b := 8
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let other_leg := Real.sqrt (b^2 - a^2)
  (hypotenuse * 2 * Real.sqrt 7).round = 53 := -- using 53 to consider rounding to the nearest tenth

by
  let a := 6
  let b := 8
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let other_leg := Real.sqrt (b^2 - a^2)
  have h1 : hypotenuse = 10 := by sorry
  have h2 : other_leg = 2 * Real.sqrt 7 := by sorry
  have h_prod : (hypotenuse * 2 * Real.sqrt 7).round = 53 := by sorry
  exact h_prod

end right_triangle_side_product_l504_504721


namespace find_a_l504_504093

theorem find_a (a x : ℝ) : 
  ((x + a)^2 / (3 * x + 65) = 2) 
  ∧ (∃ x1 x2 : ℝ,  x1 ≠ x2 ∧ (x1 = x2 + 22 ∨ x2 = x1 + 22 )) 
  → a = 3 := 
sorry

end find_a_l504_504093


namespace good_subsets_count_proof_l504_504851

def set_S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def even_numbers := {2, 4, 6, 8, 10}
def odd_numbers := {1, 3, 5, 7, 9}

def binomial_coefficient (n k : ℕ) : ℕ := n.choose k

def good_subset_count : ℕ :=
  ∑ i in finset.range 6 \ finset.singleton 0, 
    binomial_coefficient 5 i * (∑ j in finset.range i, binomial_coefficient 5 j)

theorem good_subsets_count_proof : good_subset_count = 637 := 
by sorry

end good_subsets_count_proof_l504_504851


namespace max_n_positive_sum_l504_504434

variable {a : ℕ → ℝ}
variable {a1 : ℝ} (h1 : a 1 = a1) (h2 : a1 > 0)
variable (h3 : a 5 + a 6 > 0) (h4 : a 5 * a 6 < 0)
variable (h_seq : ∀ n : ℕ, a n = a1 + (n - 1) * (a 2 - a1))

theorem max_n_positive_sum :
  ∃ n, n = 10 ∧ (∀ m < n, (∑ k in Finset.range m, a (k + 1)) > 0) ∧
         (∑ k in Finset.range n, a (k + 1)) = 0
:= sorry

end max_n_positive_sum_l504_504434


namespace percentage_snacks_at_least_wifi_percentage_l504_504314

variable (A : Type) [Finite A]
variable (airlines : Finset A)
variable (offer_wifi : A → Prop)
variable (offer_snacks : A → Prop)

noncomputable def percentage (p : A → Prop) : ℝ :=
  ((airlines.filter p).card.toReal / airlines.card.toReal) * 100

theorem percentage_snacks_at_least_wifi_percentage :
  percentage airlines offer_wifi = 35 →
  percentage airlines (λ a, offer_wifi a ∧ offer_snacks a) ≤ 35 →
  percentage airlines offer_snacks ≥ 35 := sorry

end percentage_snacks_at_least_wifi_percentage_l504_504314


namespace replace_asterisk_with_monomial_l504_504190

theorem replace_asterisk_with_monomial :
  ∀ (x : ℝ), (x^4 - 3)^2 + (x^3 + 3x)^2 = x^8 + x^6 + 9x^2 + 9 := 
by
  intro x
  calc
    (x^4 - 3)^2 + (x^3 + 3x)^2
        = (x^4)^2 - 2 * x^4 * 3 + 3^2 + (x^3)^2 + 2 * x^3 * 3x + (3x)^2 : by ring
    ... = x^8 - 6 * x^4 + 9 + x^6 + 6 * x^4 + 9 * x^2 : by ring
    ... = x^8 + x^6 + 9 * x^2 + 9 : by ring
  sorry

end replace_asterisk_with_monomial_l504_504190


namespace modulus_complex_l504_504420

theorem modulus_complex (z : ℂ) (h : z * complex.I = 1 - complex.I) : complex.abs z = real.sqrt 2 := by
  sorry

end modulus_complex_l504_504420


namespace recurring_sum_l504_504390

noncomputable def recurring_to_fraction (a b : ℕ) : ℚ := a / (10 ^ b - 1)

def r1 := recurring_to_fraction 12 2
def r2 := recurring_to_fraction 34 3
def r3 := recurring_to_fraction 567 5

theorem recurring_sum : r1 + r2 + r3 = 16133 / 99999 := by
  sorry

end recurring_sum_l504_504390


namespace beth_score_l504_504027

theorem beth_score 
  (g m a : ℕ) (num_bowlers : ℕ) 
  (h1 : g = 120)
  (h2 : m = 113)
  (h3 : a = 106)
  (h4 : num_bowlers = 3)
  (h_avg : a = (g + m + b) / num_bowlers) 
  : b = 85 :=
begin
  sorry
end

end beth_score_l504_504027


namespace random_events_l504_504926

-- Definitions for conditions
def event1 : Prop := ∃ x, x > 0 ∧ x = 9.8 -- A simplified representation for free fall
def event2 : Prop := (x^2 + 2 * x + 8 = 0)
def event3 : Prop := ∃ n : ℕ, n > 10  -- Representing the information desk receiving more than 10 requests
def event4 : Prop := ∃ t : ℕ, t ≥ 0    -- Representing the possibility of rain next Saturday

-- Definition for randomness (as an example, considering it requires some randomness aspect)
def is_random (e : Prop) : Prop := ¬(∃ c : ℝ, ∀ t : ℝ, e = (t > c))

-- Statement asserting that events 3 and 4 are random
theorem random_events :
  is_random event3 ∧ is_random event4 :=
sorry

end random_events_l504_504926


namespace coffee_merchant_mixture_price_l504_504793

theorem coffee_merchant_mixture_price
  (c1 c2 : ℝ) (w1 w2 total_cost mixture_price : ℝ)
  (h_c1 : c1 = 9)
  (h_c2 : c2 = 12)
  (h_w1w2 : w1 = 25 ∧ w2 = 25)
  (h_total_weight : w1 + w2 = 100)
  (h_total_cost : total_cost = w1 * c1 + w2 * c2)
  (h_mixture_price : mixture_price = total_cost / (w1 + w2)) :
  mixture_price = 5.25 :=
by sorry

end coffee_merchant_mixture_price_l504_504793


namespace solve_inequality_l504_504665

theorem solve_inequality (a x : ℝ) :
  (a^2 - 4) * x^2 + 4 * x - 1 > 0 ↔ 
    ((a = 2 ∨ a = -2) ∧ x > 1 / 4) ∨
    (a > 2 ∧ (x > 1 / (a + 2) ∨ x < 1 / (2 - a))) ∨
    (a < -2 ∧ (x < 1 / (a + 2) ∨ x > 1 / (2 - a))) ∨
    (-2 < a ∧ a < 2 ∧ 1 / (a + 2) < x ∧ x < 1 / (2 - a)) :=
begin
  sorry
end

end solve_inequality_l504_504665


namespace pencils_before_buying_l504_504358

theorem pencils_before_buying (x total bought : Nat) 
  (h1 : bought = 7) 
  (h2 : total = 10) 
  (h3 : total = x + bought) : x = 3 :=
by
  sorry

end pencils_before_buying_l504_504358


namespace parts_manufactured_l504_504536

variable (initial_parts : ℕ) (initial_rate : ℕ) (increased_speed : ℕ) (time_diff : ℝ)
variable (N : ℕ)

-- initial conditions
def initial_parts := 35
def initial_rate := 35
def increased_speed := 15
def time_diff := 1.5

-- additional parts to be manufactured
noncomputable def additional_parts := N

-- equation representing the time differences
noncomputable def equation := (N / initial_rate) - (N / (initial_rate + increased_speed)) = time_diff

-- state the proof problem
theorem parts_manufactured : initial_parts + additional_parts = 210 :=
by
  -- Use the given conditions to solve the problem
  sorry

end parts_manufactured_l504_504536


namespace prime_count_between_50_and_70_l504_504049

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def primes_in_range (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem prime_count_between_50_and_70 : primes_in_range 50 70 = [53, 59, 61, 67] :=
sorry

end prime_count_between_50_and_70_l504_504049


namespace length_of_EF_l504_504103

theorem length_of_EF (DE DF : ℕ) (D E F : Type) (EY FY EF : ℕ)
  (h1 : DE = 105) (h2 : DF = 110)
  (h3 : circle_radius (center D) DE = 105)
  (h4 : ∃ x y : ℕ, (x = FY ∧ y = EY) ∧ (DE = y) ∧ (y * x + x * y * y = 1075 * y)) :
  EF = 43 :=
sorry

end length_of_EF_l504_504103


namespace juice_consumption_l504_504498

theorem juice_consumption 
  (n : ℕ) (h_n : n ≥ 3) 
  (J : Fin n → ℝ) 
  (hJ_nonneg : ∀ i, J i ≥ 0) 
  (h_total_pos : ∑ i, J i > 0) :
  (∀ i, J i ≤ (1 / 3) * (∑ i, J i)) ↔ 
  (∀ i, (∃ tr : Finset (Fin n), tr.card = 3 ∧ ∀ j ∈ tr, J j > 0)) :=
sorry

end juice_consumption_l504_504498


namespace count_incorrect_statements_l504_504687

theorem count_incorrect_statements : 
  (∀ (a b : ℝ) (unit_vector_a unit_vector_b : a ≠ 0 ∧ b ≠ 0 ∧ a^2 + b^2 = 1 ∧ a = b → false)) →
  (∀ (A B C D : ℝ × ℝ) (AB CD : A ≠ B ∧ C ≠ D ∧ (AB = CD ∨ AB = -CD) → false)) →
  (∀ (a b c : ℝ) (a_parallel_b b_parallel_c : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ (a = b ∨ b = 0 ∨ b = c) → false)) →
  (∀ (A B C : ℝ × ℝ) (collinear_v different_start points_to_different_end : A ≠ B ∧ (A = B ∨ A = B) → false)) →
  true :=
by
  intros
  sorry

end count_incorrect_statements_l504_504687


namespace domain_range_of_g_l504_504799

variable (f : ℝ → ℝ)
variable (dom_f : Set.Icc 1 3)
variable (rng_f : Set.Icc 0 1)
variable (g : ℝ → ℝ)
variable (g_eq : ∀ x, g x = 2 - f (x - 1))

theorem domain_range_of_g :
  (Set.Icc 2 4) = { x | ∃ y, x = y ∧ g y = (g y) } ∧ Set.Icc 1 2 = { z | ∃ w, z = g w} :=
  sorry

end domain_range_of_g_l504_504799


namespace k_works_iff_l504_504406

noncomputable def find_k (r : ℕ) (k : ℕ) (m : ℕ) (n : ℕ) : Prop :=
  let h := 2^r in
  k ∣ m^h - 1 ∧ m ∣ n^((m^h - 1) / k) + 1

theorem k_works_iff (r k : ℕ) (m n : ℕ) (hm_odd : m % 2 = 1) (hm_gt_one : m > 1) (hn_nat : n ∈ ℕ) :
  (∃ m n, find_k r k m n ∧ m % 2 = 1 ∧ m > 1 ∧ n ∈ ℕ) ↔ 2^(r+1) ∣ k :=
by
  sorry

end k_works_iff_l504_504406


namespace percentage_good_condition_l504_504296

theorem percentage_good_condition (Oranges : ℕ) (Bananas : ℕ) (RottenOrangesPercentage : ℕ) (RottenBananasPercentage : ℕ) :
  Oranges = 600 → Bananas = 400 → RottenOrangesPercentage = 15 → RottenBananasPercentage = 3 → 
  ((Oranges * (100 - RottenOrangesPercentage) / 100 + Bananas * (100 - RottenBananasPercentage) / 100) / (Oranges + Bananas) * 100 = 89.8) :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end percentage_good_condition_l504_504296


namespace exterior_angle_regular_octagon_l504_504990

theorem exterior_angle_regular_octagon : 
  ∀ {θ : ℝ}, 
  (8 - 2) * 180 / 8 = θ →
  180 - θ = 45 := 
by 
  intro θ hθ
  sorry

end exterior_angle_regular_octagon_l504_504990


namespace simson_lines_form_reflected_similar_triangle_l504_504125

open EuclideanGeometry

/-- Given points A, B, and C defining a triangle, M a point on the circumcircle of triangle ABC,
   with A'_1, B'_1, C'_1 projections of M onto BC, CA, and AB respectively forming a collinear
   Simson line. Let A', B', C' be the second intersections of the perpendiculars from M to BC, CA, and
   AB with the circumcircle. We aim to prove the Simson lines of A', B', C' form a triangle
   mirrored similar to triangle ABC. -/
theorem simson_lines_form_reflected_similar_triangle
(A B C M A1 B1 C1 A' B' C' : Point)
(hABC : triangle A B C)
(hM : M ∈ circumcircle A B C)
(hA1 : proj M BC = A1)
(hB1 : proj M CA = B1)
(hC1 : proj M AB = C1)
(hcollinear : collinear {A1, B1, C1})
(hA' : second_intersection_perpendicular M BC (circumcircle A B C) = A')
(hB' : second_intersection_perpendicular M CA (circumcircle A B C) = B')
(hC' : second_intersection_perpendicular M AB (circumcircle A B C) = C') :
mirrored_similar_triangle (simson_line A') (simson_line B') (simson_line C') (triangle A B C)
:= sorry

end simson_lines_form_reflected_similar_triangle_l504_504125


namespace master_craftsman_parts_l504_504519

/-- 
Given:
  (1) the master craftsman produces 35 parts in the first hour,
  (2) at the rate of 35 parts/hr, he would be one hour late to meet the quota,
  (3) by increasing his speed by 15 parts/hr, he finishes the quota 0.5 hours early,
Prove that the total number of parts manufactured during the shift is 210.
-/
theorem master_craftsman_parts (N : ℕ) (quota : ℕ) 
  (initial_rate : ℕ := 35)
  (increased_rate_diff : ℕ := 15)
  (extra_time_slow : ℕ := 1)
  (time_saved_fast : ℕ := 1/2) :
  (quota = initial_rate * (extra_time_slow + 1) + N ∧
   increased_rate_diff = 15 ∧
   increased_rate_diff = λ (x : ℕ), initial_rate + x ∧
   time_saved_fast = 1/2 ∧
   N = 35) →
  quota = 210 := 
by
  sorry

end master_craftsman_parts_l504_504519


namespace prime_numbers_between_50_and_70_l504_504056

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_between_50_and_70 : 
  (finset.filter is_prime (finset.range 71)).count (λ n, 50 ≤ n ∧ n ≤ 70) = 4 := 
sorry

end prime_numbers_between_50_and_70_l504_504056


namespace simplify_sqrt_sum_l504_504646

noncomputable def sqrt_expr_1 : ℝ := Real.sqrt (12 + 8 * Real.sqrt 3)
noncomputable def sqrt_expr_2 : ℝ := Real.sqrt (12 - 8 * Real.sqrt 3)

theorem simplify_sqrt_sum : sqrt_expr_1 + sqrt_expr_2 = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_sum_l504_504646


namespace sum_g_h_k_l504_504691

def polynomial_product_constants (d g h k : ℤ) : Prop :=
  ((5 * d^2 + 4 * d + g) * (4 * d^2 + h * d - 5) = 20 * d^4 + 11 * d^3 - 9 * d^2 + k * d - 20)

theorem sum_g_h_k (d g h k : ℤ) (h1 : polynomial_product_constants d g h k) : g + h + k = -16 :=
by
  sorry

end sum_g_h_k_l504_504691


namespace solution_of_inequality_l504_504861

theorem solution_of_inequality (x : ℝ) : 
  (x+2)/(x+4) ≤ 3 ↔ x ∈ set.Ioo (-5 : ℝ) (-4 : ℝ) :=
by
  sorry

end solution_of_inequality_l504_504861


namespace sin_alpha_plus_beta_alpha_plus_two_beta_l504_504432

variables {α β : ℝ} (hα_acute : 0 < α ∧ α < π / 2) (hβ_acute : 0 < β ∧ β < π / 2)
          (h_tan_α : Real.tan α = 1 / 7) (h_sin_β : Real.sin β = Real.sqrt 10 / 10)

theorem sin_alpha_plus_beta : 
    Real.sin (α + β) = Real.sqrt 5 / 5 :=
by
  sorry

theorem alpha_plus_two_beta : 
    α + 2 * β = π / 4 :=
by
  sorry

end sin_alpha_plus_beta_alpha_plus_two_beta_l504_504432


namespace probability_of_zero_point_probability_of_increasing_on_interval_l504_504943

-- Definition of the quadratic function
def f (a b x : ℝ) : ℝ := a * x^2 - b * x + 1

-- Definitions of sets P and Q
def P : Set ℝ := {1, 2, 3}
def Q : Set ℝ := {-1, 1, 2, 3, 4}

-- All pairs (a, b)
def pairs := { (a, b) | a ∈ P, b ∈ Q }

-- Define the condition for the function to have a zero point
def has_zero_point (a b : ℝ) : Prop :=
  b^2 - 4 * a ≥ 0

-- Define the condition for the function to be increasing on [1, +∞)
def increasing_on_interval (a b : ℝ) : Prop :=
  b / (2 * a) ≤ 1

-- Prove that the probability of the function having a zero point is 2/5
theorem probability_of_zero_point : 
  (∃ six_cases ∈ pairs, has_zero_point (six_cases.fst) (six_cases.snd)).card / (pairs.card : ℝ) = 2 / 5 := sorry

-- Prove that the probability of the function being increasing on [1, +∞) is 13/15
theorem probability_of_increasing_on_interval : 
  (∃ cases ∈ pairs, increasing_on_interval (cases.fst) (cases.snd)).card / (pairs.card : ℝ) = 13 / 15 := sorry

end probability_of_zero_point_probability_of_increasing_on_interval_l504_504943


namespace measure_angle_DGF_l504_504494

noncomputable def point : Type := ℝ

structure Triangle :=
(a b c : point)

structure IsIsosceles (T : Triangle) :=
(isosceles : true)

def angle (A B C : point) : ℝ := sorry

def CFE : Triangle := ⟨(-2), 0, 50⟩
def DFG : Triangle := ⟨0, 1, 180 - 50⟩

axiom CFE_is_isosceles : IsIsosceles CFE
axiom DFG_is_isosceles : IsIsosceles DFG
axiom angle_CFE_50 : angle (-2) 50 0 = 50
axiom angle_EFG_3_CFE (angle_CFE : ℝ) : angle 50 (180 - 3 * angle_CFE) 0 = 3 * angle_CFE
axiom F_on_CD : true

theorem measure_angle_DGF : angle 50 (180 - 50 - 50) 0 = 50 := by
  sorry

end measure_angle_DGF_l504_504494


namespace train_delay_l504_504814

theorem train_delay (
  (distance : ℝ) 
  (speed_on_time : ℝ)
  (speed_late : ℝ)
  (time_on_time : ℝ := distance / speed_on_time)
  (time_late : ℝ := distance / speed_late)
  (distance_eq : distance = 70)
  (speed_on_time_eq : speed_on_time = 40)
  (speed_late_eq : speed_late = 35) 
  ) : 
  ((time_late - time_on_time) * 60 = 15) := 
by
  rw [distance_eq, speed_on_time_eq, speed_late_eq]
  rw [←div_eq_mul_inv]
  have t1 : time_on_time = 1.75 := by norm_num[time_on_time, distance_eq, speed_on_time_eq]
  have t2 : time_late = 2 := by norm_num[time_late, distance_eq, speed_late_eq]
  have time_diff : time_late - time_on_time = 0.25 := by norm_num[t2, t1]
  norm_num [time_diff]

end train_delay_l504_504814


namespace smallest_C_l504_504864

theorem smallest_C (a : fin 5 → ℝ) :
  ∃ (i j k l : fin 5), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
  |a i / a j - a k / a l| ≤ 1 / 2 :=
sorry

end smallest_C_l504_504864


namespace polynomial_remainder_l504_504756

theorem polynomial_remainder :
  let p := (x + 1)^2021 in
  let divisor := x^3 - x^2 + x - 1 in
  ∀ (x : ℝ), (p % divisor) = (x^2 - x + 1) :=
by {
  intro x,
  sorry
}

end polynomial_remainder_l504_504756


namespace constant_term_expansion_l504_504222

theorem constant_term_expansion :
  let x := λ : Type, (2 * x + 1 / x)^4 = 24

end constant_term_expansion_l504_504222


namespace seq_an_identity_l504_504544

theorem seq_an_identity (n : ℕ) (a : ℕ → ℕ) 
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) > a n)
  (h₃ : ∀ n, a (n + 1)^2 + a n^2 + 1 = 2 * (a (n + 1) * a n + a (n + 1) + a n)) 
  : a n = n^2 := sorry

end seq_an_identity_l504_504544


namespace slope_of_line_I_l504_504328

-- Line I intersects y = 1 at point P
def intersects_y_eq_one (I P : ℝ × ℝ → Prop) : Prop :=
∀ x y : ℝ, P (x, 1) ↔ I (x, y) ∧ y = 1

-- Line I intersects x - y - 7 = 0 at point Q
def intersects_x_minus_y_eq_seven (I Q : ℝ × ℝ → Prop) : Prop :=
∀ x y : ℝ, Q (x, y) ↔ I (x, y) ∧ x - y - 7 = 0

-- The coordinates of the midpoint of segment PQ are (1, -1)
def midpoint_eq (P Q : ℝ × ℝ) : Prop :=
∃ x1 y1 x2 y2 : ℝ,
  P = (x1, y1) ∧ Q = (x2, y2) ∧ ((x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1)

-- We need to show that the slope of line I is -2/3
def slope_of_I (I : ℝ × ℝ → Prop) (k : ℝ) : Prop :=
∀ x y : ℝ, I (x, y) → y + 1 = k * (x - 1)

theorem slope_of_line_I :
  ∃ I P Q : (ℝ × ℝ → Prop),
    intersects_y_eq_one I P ∧
    intersects_x_minus_y_eq_seven I Q ∧
    (∃ x1 y1 x2 y2 : ℝ, P (x1, y1) ∧ Q (x2, y2) ∧ ((x1 + x2) / 2 = 1 ∧ (y1 + y2) / 2 = -1)) →
    slope_of_I I (-2/3) :=
by
  sorry

end slope_of_line_I_l504_504328


namespace is_isosceles_right_triangle_l504_504909

-- Define the main objects in the problem
variable {α β γ : ℝ}
variable {a b c : ℝ}

-- The conditions from the problem
variable (cond1 : tan(α - β) * cos γ = 0)
variable (cond2 : sin(β + γ) * cos(β - γ) = 1)
variable (cond3 : a * cos α = b * cos β)
variable (cond4 : sin (α - β) ^ 2 + cos γ ^ 2 = 0)

-- The proposition we want to prove
theorem is_isosceles_right_triangle (cond1 : tan (α - β) * cos γ = 0)
                                    (cond2 : sin (β + γ) * cos (β - γ) = 1)
                                    (cond3 : a * cos α = b * cos β)
                                    (cond4 : sin (α - β) ^ 2 + cos γ ^ 2 = 0) : 
  (α = β ∧ γ = π / 2) ↔ (α = β ∧ γ = 90 * (π / 180)) :=
by
  -- Insert the proof here
  sorry

end is_isosceles_right_triangle_l504_504909


namespace find_m_n_l504_504667

noncomputable def m_n_sum (x : ℝ) (m n : ℤ) : Prop :=
  ∃ (m n : ℤ), (sec x + tan x = 3) ∧ (csc x + cot x = (m / n)) ∧ (gcd m n = 1) 

theorem find_m_n : ∀ (x : ℝ) (m n : ℤ), m_n_sum x m n → m + n = 3 :=
  begin
    sorry
  end

end find_m_n_l504_504667


namespace right_to_left_evaluation_l504_504999

variable (a b c d : ℝ)

theorem right_to_left_evaluation :
  a / b - c + d = a / (b - c - d) :=
sorry

end right_to_left_evaluation_l504_504999


namespace area_between_curves_l504_504315

-- Function definitions:
def quartic (a b c d e x : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e
def line (p q x : ℝ) : ℝ := p * x + q

-- Conditions:
variables (a b c d e p q α β : ℝ)
variable (a_ne_zero : a ≠ 0)
variable (α_lt_β : α < β)
variable (touch_at_α : quartic a b c d e α = line p q α ∧ deriv (quartic a b c d e) α = p)
variable (touch_at_β : quartic a b c d e β = line p q β ∧ deriv (quartic a b c d e) β = p)

-- Theorem:
theorem area_between_curves :
  ∫ x in α..β, |quartic a b c d e x - line p q x| = (a * (β - α)^5) / 30 :=
by sorry

end area_between_curves_l504_504315


namespace radius_of_circle_intersecting_parabola_l504_504497

theorem radius_of_circle_intersecting_parabola 
  (r : ℝ)
  (h1 : ∀ (A B C D : ℝ × ℝ), 
          (A.1 - 1)^2 + A.2^2 = r^2 ∧ (B.1 - 1)^2 + B.2^2 = r^2 ∧ 
          (C.1 - 1)^2 + C.2^2 = r^2 ∧ (D.1 - 1)^2 + D.2^2 = r^2) 
  (h2 : ∀ (A B C D : ℝ × ℝ),
          A.2^2 = A.1 ∧ B.2^2 = B.1 ∧ C.2^2 = C.1 ∧ D.2^2 = D.1)
  (h3 : ∀ (A B C D F : ℝ × ℝ), 
          (F = (1/4, 0)) → 
          (A.1 * D.1 = F.1 ∧ B.1 * C.1 = F.1 ∧ 
          A.2 * D.2 = F.2 ∧ B.2 * C.2 = F.2)) : 
  r = sqrt 15 / 4 :=
sorry

end radius_of_circle_intersecting_parabola_l504_504497


namespace smallest_k_for_distinct_real_roots_l504_504975

noncomputable def discriminant (a b c : ℝ) := b^2 - 4 * a * c

theorem smallest_k_for_distinct_real_roots :
  ∃ k : ℤ, (k > 0) ∧ discriminant (k : ℝ) (-3) (-9/4) > 0 ∧ (∀ m : ℤ, discriminant (m : ℝ) (-3) (-9/4) > 0 → m ≥ k) := 
by
  sorry

end smallest_k_for_distinct_real_roots_l504_504975


namespace absolute_value_difference_l504_504873

-- Define the conditions in base 6
def DDC_base6 (C D : ℕ) : ℕ := D * 6^2 + D * 6 + C
def D52D_base6 (D : ℕ) : ℕ := 5 * 6^2 + 2 * 6 + D
def C34_base6 (C : ℕ) : ℕ := C * 6^2 + 3 * 6 + 4
def C213_base6 (C: ℕ): ℕ := C * 6^3 + 2 * 6^2 + 1 * 6 + 3

theorem absolute_value_difference {C D : ℕ} (h1: DDC_base6 C D + D52D_base6 D + C34_base6 C = C213_base6 C)
                                  (h2 : C + D = 6): |C - D| = 2 :=
by sorry

end absolute_value_difference_l504_504873


namespace simplify_sqrt_expression_l504_504654

theorem simplify_sqrt_expression :
  √(12 + 8 * √3) + √(12 - 8 * √3) = 4 * √3 :=
sorry

end simplify_sqrt_expression_l504_504654


namespace parabola_focus_l504_504675

def parabola_focus_equation (a b : ℝ) : Prop :=
  ∃ (P : ℝ), y^2 = -4 * x ∧ b = 0 ∧ a = -P

theorem parabola_focus {a b : ℝ} (h : parabola_focus_equation a b) : 
  (a, b) = (-1, 0) := sorry

end parabola_focus_l504_504675


namespace sum_of_first_10_terms_of_arithmetic_sequence_l504_504433

theorem sum_of_first_10_terms_of_arithmetic_sequence :
  ∀ (a n : ℕ) (a₁ : ℤ) (d : ℤ),
  (d = -2) →
  (a₇ : ℤ := a₁ + 6 * d) →
  (a₃ : ℤ := a₁ + 2 * d) →
  (a₁₀ : ℤ := a₁ + 9 * d) →
  (a₇ * a₇ = a₃ * a₁₀) →
  (S₁₀ : ℤ := 10 * a₁ + 45 * d) →
  S₁₀ = 270 :=
by
  intros a n a₁ d hd ha₇ ha₃ ha₁₀ hgm hS₁₀
  sorry

end sum_of_first_10_terms_of_arithmetic_sequence_l504_504433


namespace total_volume_l504_504147

-- Defining the volumes for different parts as per the conditions.
variables (V_A V_C V_B' V_C' : ℝ)
variables (V : ℝ)

-- The given conditions
axiom V_A_eq_40 : V_A = 40
axiom V_C_eq_300 : V_C = 300
axiom V_B'_eq_360 : V_B' = 360
axiom V_C'_eq_90 : V_C' = 90

-- The proof goal: total volume of the parallelepiped
theorem total_volume (V_A V_C V_B' V_C' : ℝ) 
  (V_A_eq_40 : V_A = 40) (V_C_eq_300 : V_C = 300) 
  (V_B'_eq_360 : V_B' = 360) (V_C'_eq_90 : V_C' = 90) :
  V = V_A + V_C + V_B' + V_C' :=
by
  sorry

end total_volume_l504_504147


namespace find_a_for_odd_function_l504_504444

def is_odd_function {α β : Type*} [AddGroup β] (f : α → β) : Prop :=
∀ x, f (-x) = -f x

def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 2 * x else -x^2 + a * x

theorem find_a_for_odd_function :
  (∀ a : ℝ, is_odd_function (λ x, f x a) → a = -2) :=
by
  intro a h
  sorry

end find_a_for_odd_function_l504_504444


namespace product_of_possible_lengths_approx_l504_504736

noncomputable def hypotenuse (a b : ℝ) : ℝ :=
  real.sqrt (a * a + b * b)

noncomputable def other_leg (hypotenuse a : ℝ) : ℝ :=
  real.sqrt (hypotenuse * hypotenuse - a * a)

noncomputable def product_of_possible_lengths (a b : ℝ) : ℝ :=
  hypotenuse a b * other_leg (max a b) (min a b)

theorem product_of_possible_lengths_approx (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  Float.round (product_of_possible_lengths a b) 1 = 52.7 :=
by
  sorry

end product_of_possible_lengths_approx_l504_504736


namespace fraction_of_water_l504_504319

theorem fraction_of_water (total_weight sand_ratio water_weight gravel_weight : ℝ)
  (htotal : total_weight = 49.99999999999999)
  (hsand_ratio : sand_ratio = 1/2)
  (hwater : water_weight = total_weight - total_weight * sand_ratio - gravel_weight)
  (hgravel : gravel_weight = 15)
  : (water_weight / total_weight) = 1/5 :=
by
  sorry

end fraction_of_water_l504_504319


namespace vertex_angle_is_120_in_isosceles_triangle_l504_504997

variable {A B C : Type*}

-- Definition of an isosceles triangle with one angle being 120 degrees
def isosceles_triangle_with_angle_120 (A B C : Type*) [triangle A B C] : Prop := 
(angle A B C = 120 ∨ angle B A C = 120 ∨ angle B C A = 120) ∧ 
((angle A B C = angle B C A) ∨ (angle B A C = angle B C A) ∨ (angle A B C = angle B A C)) ∧ 
(angle A B C + angle B A C + angle B C A = 180)

-- Lean theorem stating that in such a triangle, the vertex angle is necessarily 120 degrees
theorem vertex_angle_is_120_in_isosceles_triangle (A B C : Type*) [triangle A B C] (h : isosceles_triangle_with_angle_120 A B C) :
  ∃ α β, (angle A B C = 120 ∧ angle B A C = α ∧ angle B C A = β ∧ α = β) :=
sorry

end vertex_angle_is_120_in_isosceles_triangle_l504_504997


namespace x_in_terms_of_y_y_in_terms_of_x_l504_504120

-- Define the main equation
variable (x y : ℝ)

-- First part: Expressing x in terms of y given the condition
theorem x_in_terms_of_y (h : x + 3 * y = 3) : x = 3 - 3 * y :=
by
  sorry

-- Second part: Expressing y in terms of x given the condition
theorem y_in_terms_of_x (h : x + 3 * y = 3) : y = (3 - x) / 3 :=
by
  sorry

end x_in_terms_of_y_y_in_terms_of_x_l504_504120


namespace minimum_value_expression_ge_5_minimum_value_expression_eq_5_at_pi_4_l504_504875

noncomputable def minimum_value_expression (x : ℝ) : ℝ :=
  (\sin x + tan x)^2 + (\cos x + cot x)^2

theorem minimum_value_expression_ge_5 (x : ℝ) (hx : 0 < x ∧ x < π / 2) : 
  minimum_value_expression x ≥ 5 :=
sorry

theorem minimum_value_expression_eq_5_at_pi_4 : 
  minimum_value_expression (π / 4) = 5 :=
sorry

end minimum_value_expression_ge_5_minimum_value_expression_eq_5_at_pi_4_l504_504875


namespace prime_count_between_50_and_70_l504_504085

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_50_and_70 : List ℕ :=
  [53, 59, 61, 67]

theorem prime_count_between_50_and_70 : 
  (primes_between_50_and_70.filter is_prime).length = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504085


namespace valid_c_values_count_l504_504887

theorem valid_c_values_count : 
  let valid_c (c : ℤ) := ∃ x : ℝ, 0 ≤ x ∧ 10 * (⌊x⌋ : ℤ) + 3 * (⌈x⌉ : ℤ) = c in
  (finset.Icc 0 1500).filter valid_c).card = 231 := 
by
  sorry

end valid_c_values_count_l504_504887


namespace square_in_triangle_l504_504771

-- Definition of vertices and scaling factor
def exists_square_in_triangle (A B C : Point) (vertAB : [Point]) (vertAC : [Point]) (vertBC : [Point]) : Prop :=
  ∃ K'L'M'N' : Square, 
  (K'L' ∈ vertAC) ∧ (L'M' ∈ vertAB) ∧ (M'N' ∈ vertBC) ∧ 
  ∃ (k : ℝ), k = (length AL') / (length AL) ∧ ∀ P in K'L'M'N', P = homothety k A P 

-- Theorem Statement
theorem square_in_triangle (A B C : Point) (vertAB : [Point]) (vertAC : [Point]) (vertBC : [Point]) : exists_square_in_triangle A B C vertAB vertAC vertBC :=
sorry

end square_in_triangle_l504_504771


namespace math_proof_problem_l504_504006

section 
variables (a b c : ℝ) (k : ℕ)

-- Proposition ①: Prove if a > b then 1/a < 1/b
def prop1 : Prop := a > b → 1 / a < 1 / b

-- Proposition ②: Prove if a > b and k ∈ ℕ* then a^k > b^k
def prop2 : Prop := a > b → (k > 0 → a^k > b^k)

-- Proposition ③: Prove if ac > bc^2 then a > b
def prop3 : Prop := (a * c > b * c^2) → a > b

-- Proposition ④: Prove if c > a > b > 0 then a / (c - a) > b / (c - b)
def prop4 : Prop := (c > a ∧ a > b ∧ b > 0) → (a / (c - a) > b / (c - b))

-- Evaluate all propositions
def evaluate_propositions : Prop :=
  ¬ prop1 ∧ prop2 ∧ ¬ prop3 ∧ prop4

-- Statement to prove
theorem math_proof_problem : evaluate_propositions :=
by sorry
end

end math_proof_problem_l504_504006


namespace sum_g_equals_1001_l504_504568

def g (x : ℝ) : ℝ := 5 / (25^x + 5)

theorem sum_g_equals_1001 :
  (Finset.range 2002).sum (λ i, g ((i + 1) / 2003)) = 1001 := 
sorry

end sum_g_equals_1001_l504_504568


namespace find_range_of_a_l504_504479

def quadratic_function (a x : ℝ) : ℝ :=
  x^2 + 2 * (a - 1) * x + 2

def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≥ f y

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

def is_monotonic_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  is_decreasing_on f I ∨ is_increasing_on f I

theorem find_range_of_a (a : ℝ) :
  is_monotonic_on (quadratic_function a) (Set.Icc (-4) 4) ↔ (a ≤ -3 ∨ a ≥ 5) :=
sorry

end find_range_of_a_l504_504479


namespace triangle_area_l504_504244

-- Definitions for the given problem conditions:
def perimeter : ℝ := 36
def inradius : ℝ := 2.5

-- Semiperimeter definition based on the given perimeter.
def semiperimeter (perimeter : ℝ) : ℝ := perimeter / 2

-- Area of a triangle with the given inradius and semiperimeter.
def area (inradius semiperimeter : ℝ) : ℝ := inradius * semiperimeter

-- The theorem stating the necessary proof:
theorem triangle_area (h_perimeter : perimeter = 36) (h_inradius : inradius = 2.5) : 
  area inradius (semiperimeter perimeter) = 45 := by
  -- Skipping the exact proof steps, inserting sorry to represent the skipped proof.
  sorry

end triangle_area_l504_504244


namespace divides_sqrt_of_perfect_square_and_condition_l504_504561

theorem divides_sqrt_of_perfect_square_and_condition (a p q : ℕ) (hpq_pos : 1 ≤ p ∧ 1 ≤ q)
  (h_perfect_square : ∃ k : ℕ, a = k * k) (h_eq : a = p * q)
  (h_div : 2021 ∣ (p^3 + q^3 + p^2 * q + p * q^2)) :
  2021 ∣ (nat.sqrt a) :=
sorry

end divides_sqrt_of_perfect_square_and_condition_l504_504561


namespace inequality_sum_l504_504584

open Real
open BigOperators

theorem inequality_sum 
  (n : ℕ) 
  (h : n > 1) 
  (x : Fin n → ℝ)
  (hx1 : ∀ i, 0 < x i) 
  (hx2 : ∑ i, x i = 1) :
  ∑ i, x i / sqrt (1 - x i) ≥ (∑ i, sqrt (x i)) / sqrt (n - 1) :=
sorry

end inequality_sum_l504_504584


namespace point_B_on_line_point_A_not_on_line_point_C_not_on_line_point_D_not_on_line_l504_504783

def is_on_line (x y : ℝ) : Prop :=
  2 * x - 3 * y + 7 = 0

def point_A := (0, -2)
def point_B := (-2, 1)
def point_C := (0, 0)
def point_D := (2, -9)

theorem point_B_on_line : is_on_line (-2) 1 := by
  -- Substitute values of B into the line equation
  have h : 2 * (-2) - 3 * 1 + 7 = 0 := by linarith
  -- Conclude that the point B satisfies is_on_line
  exact h

theorem point_A_not_on_line : ¬ is_on_line 0 (-2) := by
  -- Substitute values of A into the line equation
  have h : 2 * 0 - 3 * (-2) + 7 = 13 := by linarith
  -- Conclude that the point A does not satisfy is_on_line
  intro hA
  rw is_on_line at hA
  contradiction

theorem point_C_not_on_line : ¬ is_on_line 0 0 := by
  -- Substitute values of C into the line equation
  have h : 2 * 0 - 3 * 0 + 7 = 7 := by linarith
  -- Conclude that the point C does not satisfy is_on_line
  intro hC
  rw is_on_line at hC
  contradiction

theorem point_D_not_on_line : ¬ is_on_line 2 (-9) := by
  -- Substitute values of D into the line equation
  have h : 2 * 2 - 3 * (-9) + 7 = 38 := by linarith
  -- Conclude that the point D does not satisfy is_on_line
  intro hD
  rw is_on_line at hD
  contradiction

end point_B_on_line_point_A_not_on_line_point_C_not_on_line_point_D_not_on_line_l504_504783


namespace general_formula_sum_of_first_10_terms_l504_504505

variable (a : ℕ → ℝ) (d : ℝ) (S_10 : ℝ)
variable (h1 : a 5 = 11) (h2 : a 8 = 5)

theorem general_formula (n : ℕ) : a n = -2 * n + 21 :=
sorry

theorem sum_of_first_10_terms : S_10 = 100 :=
sorry

end general_formula_sum_of_first_10_terms_l504_504505


namespace herring_invariant_l504_504323

/--
A circle is divided into six sectors. Each sector contains one herring. 
In one move, you can move any two herrings in adjacent sectors moving them in opposite directions.
Prove that it is impossible to gather all herrings into one sector using these operations.
-/
theorem herring_invariant (herring : Fin 6 → Bool) :
  ¬ ∃ i : Fin 6, ∀ j : Fin 6, herring j = herring i := 
sorry

end herring_invariant_l504_504323


namespace satisfies_properties_l504_504286

noncomputable def f (x : ℝ) : ℝ := x^2

theorem satisfies_properties :
  (∀ x1 x2 : ℝ, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x : ℝ, 0 < x → 0 < (f' x)) ∧
  (∀ x : ℝ, f' (-x) = - f' x) := 
sorry

end satisfies_properties_l504_504286


namespace inequality_property_l504_504764

theorem inequality_property (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : ac < bc :=
by sorry

end inequality_property_l504_504764


namespace simplify_sqrt_expression_is_correct_l504_504625

-- Definition for the given problem
def simplify_sqrt_expression (a b : ℝ) :=
  a = Real.sqrt (12 + 8 * Real.sqrt 3) → 
  b = Real.sqrt (12 - 8 * Real.sqrt 3) → 
  a + b = 4 * Real.sqrt 3

-- The theorem to be proven
theorem simplify_sqrt_expression_is_correct : simplify_sqrt_expression :=
begin
  intros a b ha hb,
  rw [ha, hb],
  -- Step-by-step simplification approach would occur here
  sorry
end

end simplify_sqrt_expression_is_correct_l504_504625


namespace right_triangle_side_product_l504_504723

theorem right_triangle_side_product :
  let a := 6
  let b := 8
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let other_leg := Real.sqrt (b^2 - a^2)
  (hypotenuse * 2 * Real.sqrt 7).round = 53 := -- using 53 to consider rounding to the nearest tenth

by
  let a := 6
  let b := 8
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let other_leg := Real.sqrt (b^2 - a^2)
  have h1 : hypotenuse = 10 := by sorry
  have h2 : other_leg = 2 * Real.sqrt 7 := by sorry
  have h_prod : (hypotenuse * 2 * Real.sqrt 7).round = 53 := by sorry
  exact h_prod

end right_triangle_side_product_l504_504723


namespace equilateral_triangle_perimeter_and_area_l504_504508

noncomputable def equilateral_triangle_side : ℝ := 12

theorem equilateral_triangle_perimeter_and_area :
  let s := equilateral_triangle_side in
  (3 * s = 36) ∧ (sqrt 3 / 4 * s^2 = 36 * sqrt 3) :=
by
  sorry

end equilateral_triangle_perimeter_and_area_l504_504508


namespace average_rates_of_change_order_l504_504681

-- Define the function y = 1/x
def f (x : ℝ) : ℝ := 1 / x

-- Define the intervals
def interval1 := (1 : ℝ, 2 : ℝ)
def interval2 := (2 : ℝ, 3 : ℝ)
def interval3 := (3 : ℝ, 4 : ℝ)

-- Define the average rate of change over an interval
def average_rate_of_change (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (f b - f a) / (b - a)

-- Compute the average rates of change over the respective intervals
def k1 := average_rate_of_change f 1 2
def k2 := average_rate_of_change f 2 3
def k3 := average_rate_of_change f 3 4

-- The proof statement
theorem average_rates_of_change_order : k3 < k2 ∧ k2 < k1 :=
by sorry

end average_rates_of_change_order_l504_504681


namespace master_craftsman_total_parts_l504_504517

theorem master_craftsman_total_parts
  (N : ℕ) -- Additional parts to be produced after the first hour
  (initial_rate : ℕ := 35) -- Initial production rate (35 parts/hour)
  (increased_rate : ℕ := initial_rate + 15) -- Increased production rate (50 parts/hour)
  (time_difference : ℝ := 1.5) -- Time difference in hours between the rates
  (eq_time_diff : (N / initial_rate) - (N / increased_rate) = time_difference) -- The given time difference condition
  : 35 + N = 210 := -- Conclusion we need to prove
sorry

end master_craftsman_total_parts_l504_504517


namespace problem1_problem2_l504_504913

variable (a : ℝ)
def p : Prop := 1^2 < a
def q : Prop := 2^2 < a

-- Problem (1)
theorem problem1 : (p ∨ q) → a > 1 := by 
  sorry

-- Problem (2)
theorem problem2 : (p ∧ q) → a > 4 := by
  sorry

end problem1_problem2_l504_504913


namespace domain_of_f_not_symmetric_about_x_1_f_of_f_neg5_range_of_f_l504_504450

noncomputable def f (x : ℝ) : ℝ := 2 / (|x| - 1)

theorem domain_of_f : ∀ x : ℝ, x ≠ 1 → x ≠ -1 → (|x| - 1 ≠ 0) :=
by
  intro x h1 h2
  apply ne_of_ne_of_eq
  exact h1
  exact h2
  sorry

theorem not_symmetric_about_x_1 : ¬ (∀ x : ℝ, f(x + 1) = f(1 - x)) :=
by
  intro h
  specialize h 0
  have := f(0)
  have := f(2)
  sorry

theorem f_of_f_neg5 : f (f (-5)) = -4 :=
by
  have h1 : f (-5) = 2 / (|(-5)| - 1) := rfl
  have h2 : |(-5)| = 5 := abs_neg 5
  have h3 : 5 - 1 = 4 := sub_self 4
  have h4 : f (1/2) = -4 := rfl
  sorry

theorem range_of_f : (set.range f) = set.union (set.Iic (-2)) (set.Ioi 0):=
by
  sorry

end domain_of_f_not_symmetric_about_x_1_f_of_f_neg5_range_of_f_l504_504450


namespace percentage_increase_in_area_l504_504305

variable (L W : Real)

theorem percentage_increase_in_area (hL : L > 0) (hW : W > 0) :
  ((1 + 0.25) * L * (1 + 0.25) * W - L * W) / (L * W) * 100 = 56.25 := by
  sorry

end percentage_increase_in_area_l504_504305


namespace binomial_expansion_integer_exponents_terms_l504_504680

theorem binomial_expansion_integer_exponents_terms :
  let n := 8 in
  let terms_with_integer_exponents := 
    {r : ℕ | ∃ k : ℤ, n - 2 * r = 2 * k} in
  (terms_with_integer_exponents.card = 3) :=
by sorry

end binomial_expansion_integer_exponents_terms_l504_504680


namespace jason_safe_combination_count_l504_504551

theorem jason_safe_combination_count : 
  let digits := {1, 2, 3, 4, 5, 6}
  let even_digits := {2, 4, 6}
  let odd_digits := {1, 3, 5}
  let valid_combination (c : List ℕ) := 
    c.length = 5 ∧ 
    ∀ i, i < 4 → ((c.nth i ∈ even_digits) → (c.nth (i+1) ∈ odd_digits)) ∧ 
              ((c.nth i ∈ odd_digits) → (c.nth (i+1) ∈ even_digits))
  (∃ c, c ∈ List.replicate 5 digits ∧ valid_combination c).card = 486 := 
sorry

end jason_safe_combination_count_l504_504551


namespace ryan_hours_on_english_l504_504869

-- Given the conditions
def hours_on_chinese := 2
def hours_on_spanish := 4
def extra_hours_between_english_and_spanish := 3

-- We want to find out the hours on learning English
def hours_on_english := hours_on_spanish + extra_hours_between_english_and_spanish

-- Proof statement
theorem ryan_hours_on_english : hours_on_english = 7 := by
  -- This is where the proof would normally go.
  sorry

end ryan_hours_on_english_l504_504869


namespace probability_hit_10_or_7_ring_probability_below_7_ring_l504_504784

noncomputable def P_hit_10_ring : ℝ := 0.21
noncomputable def P_hit_9_ring : ℝ := 0.23
noncomputable def P_hit_8_ring : ℝ := 0.25
noncomputable def P_hit_7_ring : ℝ := 0.28
noncomputable def P_below_7_ring : ℝ := 0.03

theorem probability_hit_10_or_7_ring :
  P_hit_10_ring + P_hit_7_ring = 0.49 :=
  by sorry

theorem probability_below_7_ring :
  P_below_7_ring = 0.03 :=
  by sorry

end probability_hit_10_or_7_ring_probability_below_7_ring_l504_504784


namespace total_money_shared_l504_504825

/-- Assume there are four people Amanda, Ben, Carlos, and David, sharing an amount of money.
    Their portions are in the ratio 1:2:7:3.
    Amanda's portion is $20.
    Prove that the total amount of money shared by them is $260. -/
theorem total_money_shared (A B C D : ℕ) (h_ratio : A = 20 ∧ B = 2 * A ∧ C = 7 * A ∧ D = 3 * A) :
  A + B + C + D = 260 := by 
  sorry

end total_money_shared_l504_504825


namespace fish_count_l504_504499

variables
  (x g s r : ℕ)
  (h1 : x - g = (2 / 3 : ℚ) * x - 1)
  (h2 : x - r = (2 / 3 : ℚ) * x + 4)
  (h3 : x = g + s + r)

theorem fish_count :
  s - g = 2 :=
by
  sorry

end fish_count_l504_504499


namespace range_of_f_l504_504929

def f (x : ℕ) : ℤ := 2 * x - 3

def domain := {x : ℕ | 1 ≤ x ∧ x ≤ 5}

def range (f : ℕ → ℤ) (s : Set ℕ) : Set ℤ :=
  {y : ℤ | ∃ x ∈ s, f x = y}

theorem range_of_f :
  range f domain = {-1, 1, 3, 5, 7} :=
by
  sorry

end range_of_f_l504_504929


namespace parallel_vector_magnitude_l504_504460

open Real
open Vector

-- Given conditions
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, -4)

-- Parallel condition (cross product zero)
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 - u.2 * v.1 = 0

-- Problem statement in Lean
theorem parallel_vector_magnitude : parallel a b → ∥(3 • a + b)∥ = sqrt 5 := by
  sorry

end parallel_vector_magnitude_l504_504460


namespace students_in_class_l504_504812

theorem students_in_class (total_spent: ℝ) (packs_per_student: ℝ) (sausages_per_student: ℝ) (cost_pack_noodles: ℝ) (cost_sausage: ℝ) (cost_per_student: ℝ) (num_students: ℝ):
  total_spent = 290 → 
  packs_per_student = 2 → 
  sausages_per_student = 1 → 
  cost_pack_noodles = 3.5 → 
  cost_sausage = 7.5 → 
  cost_per_student = packs_per_student * cost_pack_noodles + sausages_per_student * cost_sausage →
  total_spent = cost_per_student * num_students →
  num_students = 20 := 
by
  sorry

end students_in_class_l504_504812


namespace inclination_angle_l504_504238

theorem inclination_angle (x y : ℝ) (α : ℝ) (h : sqrt 3 * x + 3 * y + 2 = 0) : α = 150 :=
sorry

end inclination_angle_l504_504238


namespace max_stamps_purchase_l504_504484

theorem max_stamps_purchase (price_per_stamp : ℕ) (total_money : ℕ) (h_price : price_per_stamp = 45) (h_money : total_money = 5000) : 
  (total_money / price_per_stamp) = 111 := 
by 
  rw [h_price, h_money]
  rfl

sorry

end max_stamps_purchase_l504_504484


namespace product_of_p_yi_l504_504574

noncomputable def h (y : ℝ) := y^5 - y^3 + 2 * y + 3
noncomputable def p (y : ℝ) := y^2 - 3

theorem product_of_p_yi {y₁ y₂ y₃ y₄ y₅ : ℝ}
  (h_root : ∀ {y}, h y = 0 ↔ y = y₁ ∨ y = y₂ ∨ y = y₃ ∨ y = y₄ ∨ y = y₅) :
  p(y₁) * p(y₂) * p(y₃) * p(y₄) * p(y₅) = -183 :=
sorry

end product_of_p_yi_l504_504574


namespace loom_weaving_rate_l504_504320

theorem loom_weaving_rate :
  ∀ (time : ℝ) (cloth : ℝ), time = 117.1875 → cloth = 15 → (cloth / time) = 0.128 :=
by
  intros time cloth h_time h_cloth
  rw [h_time, h_cloth]
  norm_num
  sorry

end loom_weaving_rate_l504_504320


namespace right_triangle_at_n_12_l504_504144

noncomputable def angle_sequence (x₀ : ℝ) : ℕ → ℝ
| 0       := x₀
| (n + 1) := 180 - 2 * angle_sequence n

theorem right_triangle_at_n_12 :
  let x₀ := 60.2 in
  let y₀ := 59.8 in
  let z₀ := 60 in
  ∃ n : ℕ, (triangle_is_right (angle_sequence x₀ n) (angle_sequence y₀ n) (angle_sequence z₀ n)) ∧ n = 12 :=
sorry

def triangle_is_right (x : ℝ) (y : ℝ) (z : ℝ) : Prop :=
x = 90 ∨ y = 90 ∨ z = 90

end right_triangle_at_n_12_l504_504144


namespace sum_of_interior_angles_and_regularity_l504_504142

theorem sum_of_interior_angles_and_regularity (n : ℕ) 
  (h1 : ∀ i e, i = 9 * e) 
  (h2 : ∑ k in finset.range n, exterior_angle Q k = 360) :
  ∃ T, T = 3240 ∧ (∀ i e, i = 9 * e → Q regular ∨ Q not_regular) := 
sorry

end sum_of_interior_angles_and_regularity_l504_504142


namespace farmer_apples_count_l504_504234

theorem farmer_apples_count (initial : ℕ) (given : ℕ) (remaining : ℕ) 
  (h1 : initial = 127) (h2 : given = 88) : remaining = initial - given := 
by
  sorry

end farmer_apples_count_l504_504234


namespace sequence_inequality_l504_504153

theorem sequence_inequality
  (a : ℕ → ℝ)
  (h_cond : ∀ k m : ℕ, |a (k + m) - a k - a m| ≤ 1) :
  ∀ p q : ℕ, |a p / p - a q / q| < 1 / p + 1 / q :=
by
  intros p q
  sorry

end sequence_inequality_l504_504153


namespace compare_sizes_l504_504372

theorem compare_sizes :
  -(+(5/9)) > -|-4/7| :=
sorry

end compare_sizes_l504_504372


namespace number_of_packages_l504_504282

theorem number_of_packages (total_tshirts : ℕ) (tshirts_per_package : ℕ) 
  (h1 : total_tshirts = 56) (h2 : tshirts_per_package = 2) : 
  (total_tshirts / tshirts_per_package) = 28 := 
  by
    sorry

end number_of_packages_l504_504282


namespace intersection_x_solutions_l504_504404

noncomputable def intersection_x_values : set ℝ :=
  {x | ∃ y, y = 10 / (x^2 + 1) ∧ x^2 + y = 3}

theorem intersection_x_solutions :
  intersection_x_values = {sqrt (1 + 2 * sqrt 2), - sqrt (1 + 2 * sqrt 2)} :=
by
  sorry

end intersection_x_solutions_l504_504404


namespace prime_count_between_50_and_70_l504_504073

open Nat

theorem prime_count_between_50_and_70 : 
  (finset.filter Nat.prime (finset.range 71 \ finset.range 51).card = 4) := 
begin
  sorry
end

end prime_count_between_50_and_70_l504_504073


namespace find_a_l504_504013

def f (x : ℝ) : ℝ :=
  if x > 0 then 2 * x else x + 1

theorem find_a (a : ℝ) (h : f a + f 1 = 0) : a = -3 :=
by
  have f_1 : f 1 = 2 := by simp [f]
  unfold f at h
  split_ifs at h with h_pos
  · rw [f_1] at h
    simp at h
    contradiction
  · rw [f_1] at h
    simp at h
    assumption

end find_a_l504_504013


namespace prime_count_between_50_and_70_l504_504034

theorem prime_count_between_50_and_70 : 
  (finset.filter nat.prime (finset.range 71).filter (λ n, 50 < n ∧ n < 71)).card = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504034


namespace prime_count_between_50_and_70_l504_504039

theorem prime_count_between_50_and_70 : 
  (finset.filter nat.prime (finset.range 71).filter (λ n, 50 < n ∧ n < 71)).card = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504039


namespace retailer_profit_percent_l504_504769

noncomputable def purchase_price : ℝ := 225
noncomputable def overhead_expenses : ℝ := 15
noncomputable def selling_price : ℝ := 300

def total_cost_price : ℝ := purchase_price + overhead_expenses
def profit : ℝ := selling_price - total_cost_price
def profit_percent : ℝ := (profit / total_cost_price) * 100

theorem retailer_profit_percent : profit_percent = 25 := by
  sorry

end retailer_profit_percent_l504_504769


namespace smallest_sum_of_20_consecutive_integers_twice_perfect_square_l504_504258

theorem smallest_sum_of_20_consecutive_integers_twice_perfect_square :
  ∃ n : ℕ, ∃ k : ℕ, (∀ m : ℕ, m ≥ n → 0 < m) ∧ 10 * (2 * n + 19) = 2 * k^2 ∧ 10 * (2 * n + 19) = 450 :=
by
  sorry

end smallest_sum_of_20_consecutive_integers_twice_perfect_square_l504_504258


namespace line_passes_through_fixed_point_l504_504455

-- Definition of line l
def line (m : ℝ) (x y : ℝ) : Prop := m * (x - 1) - y - 2 = 0

-- Definition of circle C
def circle (x y : ℝ) : Prop := (x + 1) ^ 2 + (y + 2) ^ 2 = 9

-- Prove that the line passes through the point (1, -2)
theorem line_passes_through_fixed_point (m : ℝ) :
  line m 1 (-2) :=
by
  unfold line
  rw sub_self
  norm_num
  -- This leaves us with a proof that -2 = -2, trivially true.
  exact eq.refl (-2)
  sorry

end line_passes_through_fixed_point_l504_504455


namespace cashback_discount_percentage_l504_504383

noncomputable def iphoneOriginalPrice : ℝ := 800
noncomputable def iwatchOriginalPrice : ℝ := 300
noncomputable def iphoneDiscountRate : ℝ := 0.15
noncomputable def iwatchDiscountRate : ℝ := 0.10
noncomputable def finalPrice : ℝ := 931

noncomputable def iphoneDiscountedPrice : ℝ := iphoneOriginalPrice * (1 - iphoneDiscountRate)
noncomputable def iwatchDiscountedPrice : ℝ := iwatchOriginalPrice * (1 - iwatchDiscountRate)
noncomputable def totalDiscountedPrice : ℝ := iphoneDiscountedPrice + iwatchDiscountedPrice
noncomputable def cashbackAmount : ℝ := totalDiscountedPrice - finalPrice
noncomputable def cashbackRate : ℝ := (cashbackAmount / totalDiscountedPrice) * 100

theorem cashback_discount_percentage : cashbackRate = 2 := by
  sorry

end cashback_discount_percentage_l504_504383


namespace intersection_of_sets_l504_504948

def M : Set ℝ := { x | 3 * x - 6 ≥ 0 }
def N : Set ℝ := { x | x^2 < 16 }

theorem intersection_of_sets : M ∩ N = { x | 2 ≤ x ∧ x < 4 } :=
by {
  sorry
}

end intersection_of_sets_l504_504948


namespace debby_total_bottles_l504_504859

theorem debby_total_bottles :
  (let daily_bottles := [20, 15, 25, 18, 22, 30, 28] in
   let days := 74 in
   let total_weekly_bottles := List.sum daily_bottles in
   let weeks := days / 7 in
   let remaining_days := days % 7 in
   let full_weeks_bottles := weeks * total_weekly_bottles in
   let remaining_bottles := daily_bottles.take remaining_days |> List.sum in
   let total_drunk := full_weeks_bottles + remaining_bottles in
   let percent_given_away := 0.05 in
   let given_away := Float.ceil (percent_given_away * total_drunk) in
   nat_total := total_drunk + given_away.toNat; nat_total)
  = 1741 := sorry

end debby_total_bottles_l504_504859


namespace intersection_count_l504_504380

-- Define the polar equations as conditions
def r1 (θ : ℝ) : ℝ := 3 * Real.cos θ
def r2 (θ : ℝ) : ℝ := 6 * Real.sin θ

-- Define the task of determining the number of intersection points
theorem intersection_count : ∃ θ1 θ2 : ℝ, (r1 θ1 = r2 θ1) ∧ (r1 θ2 = r2 θ2) ∧ θ1 ≠ θ2 ∧ 
                           (∀ θ : ℝ, (r1 θ = r2 θ) → (θ = θ1 ∨ θ = θ2)) :=
by
  sorry

end intersection_count_l504_504380


namespace staircase_sum_of_digits_l504_504853

def cozy_jumps (n : ℕ) : ℕ := (n + 1) / 2
def dash_jumps (n : ℕ) : ℕ := (n + 4) / 5

theorem staircase_sum_of_digits :
  let steps := (finset.range 1000).filter (λ n => cozy_jumps n - dash_jumps n = 19) in
  let s := steps.sum id in
  s.digits.sum = 13 :=
by
  sorry

end staircase_sum_of_digits_l504_504853


namespace quadratic_discriminant_constraint_l504_504096

theorem quadratic_discriminant_constraint (c : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 4*x1 + c = 0 ∧ x2^2 - 4*x2 + c = 0) ↔ c < 4 := 
by
  sorry

end quadratic_discriminant_constraint_l504_504096


namespace f_D_not_mapping_to_B_l504_504152

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 2}
def B := {y : ℝ | 1 ≤ y ∧ y <= 4}
def f_D (x : ℝ) := 4 - x^2

theorem f_D_not_mapping_to_B : ¬ (∀ x ∈ A, f_D x ∈ B) := sorry

end f_D_not_mapping_to_B_l504_504152


namespace cartesian_equation_of_parametric_l504_504399

variable (t : ℝ) (x y : ℝ)

open Real

theorem cartesian_equation_of_parametric 
  (h1 : x = sqrt t)
  (h2 : y = 2 * sqrt (1 - t))
  (h3 : 0 ≤ t ∧ t ≤ 1) :
  (x^2 / 1) + (y^2 / 4) = 1 := by 
  sorry

end cartesian_equation_of_parametric_l504_504399


namespace base8_to_base10_l504_504716

theorem base8_to_base10 (n : ℕ) : n = 4 * 8^3 + 3 * 8^2 + 7 * 8^1 + 2 * 8^0 → n = 2298 :=
by 
  sorry

end base8_to_base10_l504_504716


namespace fair_die_odd_probability_l504_504617

theorem fair_die_odd_probability :
  let outcomes := {1, 2, 3, 4, 5, 6}
  let odd_outcomes := {1, 3, 5}
  (set.card odd_outcomes / set.card outcomes : ℝ) = 1 / 2 :=
by
  sorry

end fair_die_odd_probability_l504_504617


namespace _l504_504423

noncomputable def positive_sequence (a : ℕ → ℝ) : Prop := ∀ n, a n > 0

noncomputable def sum_first_n_terms (a : ℕ → ℝ) : ℕ → ℝ
| 0 => 0
| (n + 1) => sum_first_n_terms n + a (n + 1)

noncomputable def arithmetic_sequence (S a : ℕ → ℝ) : Prop :=
  ∀ n, 2 * a n = S n + 1 / 2

noncomputable theorem problem
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_positive : positive_sequence a)
  (h_sum : ∀ n, S n = sum_first_n_terms a n)
  (h_arith_seq : arithmetic_sequence S a) :
  (∀ n, a n = (1 / 2) * 2 ^ n) ∧
  (let b n := 3 + log2 (a n) in
    ∀ n, let T_n := ∑ i in finset.range (n + 1), 1 / (b i * b (i + 1)) in
          T_n < 1 / 2) := sorry

end _l504_504423


namespace cell_phone_bill_l504_504789

theorem cell_phone_bill
  (base_cost : ℕ := 25) 
  (text_cost_per_message : ℕ := 3) 
  (extra_minute_cost : ℕ := 15) 
  (texts : ℕ := 150) 
  (talk_time_hours : ℕ := 26) :
  let text_cost := (texts * text_cost_per_message) / 100
  let extra_minutes := max 0 ((talk_time_hours - 25) * 60)
  let extra_cost := (extra_minutes * extra_minute_cost) / 100
  let total_cost := base_cost + text_cost + extra_cost
  total_cost = 38.50
:= by
  sorry

end cell_phone_bill_l504_504789


namespace train_passenger_problem_l504_504837

theorem train_passenger_problem :
  let trains_per_hour := 60 / 5,
      passengers_taken_per_hour := 320 * trains_per_hour,
      total_passengers := 6240
  in (total_passengers - passengers_taken_per_hour) / trains_per_hour = 200 := by
  sorry

end train_passenger_problem_l504_504837


namespace p_sufficient_not_necessary_q_l504_504570

-- Definitions for conditions p and q
def p (x : ℝ) : Prop := log x / log 2 < 0
def q (x : ℝ) : Prop := (1/2)^(x - 1) > 1

-- Theorem: p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  -- Proof is skipped
  sorry

end p_sufficient_not_necessary_q_l504_504570


namespace slope_tangent_at_pi_div_six_l504_504935

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x - 2 * Real.cos x

theorem slope_tangent_at_pi_div_six : (deriv f π / 6) = 3 / 2 := 
by 
  sorry

end slope_tangent_at_pi_div_six_l504_504935


namespace solution_set_inequality_l504_504698

   theorem solution_set_inequality (a : ℝ) : (∀ x : ℝ, x^2 - 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) :=
   sorry
   
end solution_set_inequality_l504_504698


namespace solve_congruence_l504_504662

theorem solve_congruence : ∃ n : ℕ, 0 ≤ n ∧ n < 43 ∧ 11 * n % 43 = 7 :=
by
  sorry

end solve_congruence_l504_504662


namespace primes_between_50_and_70_l504_504078

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).filter is_prime |>.length

theorem primes_between_50_and_70 : count_primes_in_range 50 70 = 4 :=
by
  sorry

end primes_between_50_and_70_l504_504078


namespace darma_peanut_consumption_l504_504858

theorem darma_peanut_consumption :
  ∀ (t : ℕ) (rate : ℕ),
  (rate = 20 / 15) →  -- Given the rate of peanut consumption
  (t = 6 * 60) →     -- Given that the total time is 6 minutes
  (rate * t = 480) :=  -- Prove that the total number of peanuts eaten in 6 minutes is 480
by
  intros t rate h_rate h_time
  sorry

end darma_peanut_consumption_l504_504858


namespace min_abs_diff_l504_504903

theorem min_abs_diff (a b c d : ℝ) (h1 : |a - b| = 5) (h2 : |b - c| = 8) (h3 : |c - d| = 10) : 
  ∃ m, m = |a - d| ∧ m = 3 := 
by 
  sorry

end min_abs_diff_l504_504903


namespace part1_solution_count_part2_k_solutions_count_l504_504785

-- Part (1)
theorem part1_solution_count :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), 
    (∀ (m n r) ∈ solutions, mn + nr + mr = 2 * (m + n + r)) ∧
    solutions.card = 7 := 
sorry

-- Part (2)
theorem part2_k_solutions_count (k : ℕ) (hk : k > 1) :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), 
    (∀ (m n r) ∈ solutions, mn + nr + mr = k * (m + n + r)) ∧
    solutions.card ≥ 3 * k + 1 := 
sorry

end part1_solution_count_part2_k_solutions_count_l504_504785


namespace rectangle_circle_intersection_area_l504_504713

theorem rectangle_circle_intersection_area :
  let vertices := [(2, 7), (13, 7), (13, -6), (2, -6)]
  let circle_eq := ∀ (x y : ℝ), (x - 2)^2 + (y + 6)^2 = 25
  ∃ (area : ℝ), area = (25 / 4) * Real.pi :=
by
  let vertices := [(2, 7), (13, 7), (13, -6), (2, -6)]
  let circle_eq := (λ (x y : ℝ), (x - 2)^2 + (y + 6)^2 = 25)
  use (25 / 4) * Real.pi
  sorry

end rectangle_circle_intersection_area_l504_504713


namespace probability_real_roots_l504_504690

/-- 
Given that \( n \in (0,1) \), the probability that the equation \( x^2 + x + n = 0 \) has real roots is \( \frac{1}{4} \).
-/
theorem probability_real_roots (n : ℝ) (h0 : 0 < n) (h1 : n < 1) : 
    let Δ := 1 - 4 * n in
    Δ ≥ 0 →  n ≤ 1 / 4 ∧ 
    ((∀ n, (0 < n ∧ n ≤ 1 / 4)) → 
    (∀ n, 0 < n ∧ n < 1) → 
    (real_interval_length : ℝ := 1 / 4) → 
    sorry :=
begin
    sorry,
end

end probability_real_roots_l504_504690


namespace negation_of_forall_l504_504685

theorem negation_of_forall (x : ℝ) : ¬ (∀ (x : ℝ), x^2 ≥ 0) ↔ ∃ (x : ℝ), x^2 < 0 :=
begin
  sorry
end

end negation_of_forall_l504_504685


namespace fraction_of_age_when_babysitting_l504_504129

noncomputable theory

-- Define the constants and conditions
def janeStartedAge : ℕ := 20
def currentJaneAge : ℕ := 32
def yearsSinceStoppedBabysitting : ℕ := 10
def currentOldestChildAge : ℕ := 22

-- Define derived ages
def stoppedJaneAge : ℕ := currentJaneAge - yearsSinceStoppedBabysitting
def babysittingOldestChildAge : ℕ := currentOldestChildAge - yearsSinceStoppedBabysitting

-- The fraction of Jane's age that the child could be at most
def babysittingAgeFraction : ℚ := babysittingOldestChildAge / stoppedJaneAge

-- The theorem statement
theorem fraction_of_age_when_babysitting :
  babysittingAgeFraction = 6 / 11 :=
by {
  -- The proof is skipped
  sorry
}

end fraction_of_age_when_babysitting_l504_504129


namespace expected_value_of_X_eq_6_l504_504400

theorem expected_value_of_X_eq_6 : 
  let X := [-4, 6, 10]
  let p := [0.2, 0.3, 0.5]
  (\sum i, p[i] * X[i]) = 6 := 
by
  sorry

end expected_value_of_X_eq_6_l504_504400


namespace prime_numbers_between_50_and_70_l504_504054

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_between_50_and_70 : 
  (finset.filter is_prime (finset.range 71)).count (λ n, 50 ≤ n ∧ n ≤ 70) = 4 := 
sorry

end prime_numbers_between_50_and_70_l504_504054


namespace ellipse_equation_l504_504454

noncomputable def hyperbola_foci (a b : ℝ) : set (ℝ × ℝ) :=
  {p | ∃ h, ((p.1, p.2) = (0, a) ∨ (p.1, p.2) = (0, -a))
     ∧ h = p.1^2 / b^2 - p.2^2 / a^2 = -1}

noncomputable def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
  {p | p.1^2 / a^2 + p.2^2 / b^2 = 1}

theorem ellipse_equation 
  (a b : ℝ) 
  (h : hyperbola_foci 4 (2 * Real.sqrt 3) (0, -4))
  (ellipse_major_axis : ellipse 4 2) :
  ellipse 4 2 := 
  sorry

end ellipse_equation_l504_504454


namespace conditions_not_sufficient_nor_necessary_l504_504435

theorem conditions_not_sufficient_nor_necessary (a : ℝ) (b : ℝ) :
  (a ≠ 5) ∧ (b ≠ -5) ↔ ¬((a ≠ 5) ∨ (b ≠ -5)) ∧ (a + b ≠ 0) := 
sorry

end conditions_not_sufficient_nor_necessary_l504_504435


namespace substitute_monomial_to_simplify_expr_l504_504186

theorem substitute_monomial_to_simplify_expr (k : ℤ) : 
  ( ∃ k : ℤ, (x^4 - 3)^2 + (x^3 + k * x)^2 after expanding has exactly four terms) := 
begin
  use 3,
  sorry
end

end substitute_monomial_to_simplify_expr_l504_504186


namespace gcd_of_lengths_l504_504683

-- Definitions of lengths in cm
def length1 : ℕ := 315
def length2 : ℕ := 4 * 100 + 58  -- 4 meters and 58 centimeters
def length3 : ℕ := 9 * 100 + 212 -- 9 meters and 212 centimeters

-- GCD function for Nat
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Lean statement to prove that the GCD of the three lengths is 1
theorem gcd_of_lengths :
  gcd (gcd length1 length2) length3 = 1 :=
sorry

end gcd_of_lengths_l504_504683


namespace solve_inequality_l504_504207

theorem solve_inequality (x : ℝ) (hx1 : x > 0) (hx2 : x ≠ 1) :
  8^(sqrt(log 2 x)) - 7 * 2^(1 + sqrt(4 * log 2 x)) + 60 * x * sqrt(log x 2) ≤ 72 ↔ 
  x ∈ (Set.Ioo 1 2 ∪ {6^(log 2 6)}) :=
by
  sorry

end solve_inequality_l504_504207


namespace find_the_number_l504_504250

theorem find_the_number (x : ℝ) (h : 150 - x = x + 68) : x = 41 :=
sorry

end find_the_number_l504_504250


namespace x_add_y_values_l504_504912

theorem x_add_y_values (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : (2 * x - 5) * (2 * y - 5) = 25) :
  x + y = 10 ∨ x + y = 18 :=
begin
  sorry
end

end x_add_y_values_l504_504912


namespace triangle_angle_symmetry_l504_504577

theorem triangle_angle_symmetry
  {A B C Q P A1 C1 C2 : Type*}
  (triangle_ABC : Triangle A B C)
  (on_angle_bisector : AngleBisector Q (Angle B A C))
  (circumscribed_BAQ : CircumscribedCircle Q [B, A, Q])
  (intersect_AC : IntersectPoint_on_Segment P A C)
  (P_neq_C : P ≠ C)
  (circumscribed_CQP : CircumscribedCircle Q [C, Q, P])
  (r_omega1_gt_r_omega2 : Radius circumscribed_BAQ > Radius circumscribed_CQP)
  (circle_Q_radius_QA_intersect_omega1 : CircleCenterRadiusIntersects Q (Distance Q A) (CircumscribedCircle Q [B, A, Q]) A [A, A1])
  (circle_Q_radius_QC_intersect_omega1 : CircleCenterRadiusIntersects Q (Distance Q C) (CircumscribedCircle Q [B, A, Q]) C [C1, C2]) :
  Angle A1 B C1 = Angle C2 P A :=
by
  sorry

end triangle_angle_symmetry_l504_504577


namespace divide_square_l504_504866

theorem divide_square (ABCD : Square) (s : ℝ) :
  ∃ (EFGH : Square), EFGH ⊆ ABCD ∧ side_len EFGH = s ∧ 
  area ABCD = area EFGH + 4 * area_of_right_triangle (s / √2) := 
sorry

end divide_square_l504_504866


namespace Mrs_Hilt_nickels_l504_504600

def total_value (pennies dimes nickels : ℕ) : ℕ :=
  pennies * 1 + dimes * 10 + nickels * 5

theorem Mrs_Hilt_nickels :
  let MrsHilt_pennies := 2 in
  let MrsHilt_dimes := 2 in
  let Jacob_pennies := 4 in
  let Jacob_nickels := 1 in
  let Jacob_dimes := 1 in
  let diff := 13 in
  ∃ (MrsHilt_nickels : ℕ), 
  total_value MrsHilt_pennies MrsHilt_dimes MrsHilt_nickels - 
  total_value Jacob_pennies Jacob_dimes Jacob_nickels = diff ∧
  MrsHilt_nickels = 2 :=
begin
  sorry
end

end Mrs_Hilt_nickels_l504_504600


namespace triangle_third_side_lengths_product_l504_504744

def hypotenuse (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

def leg (c b : ℝ) : ℝ :=
  real.sqrt (c^2 - b^2)

theorem triangle_third_side_lengths_product :
  let a := 6
  let b := 8
  let hyp := hypotenuse a b
  let leg := leg b a
  real.round (hyp * leg * 10) / 10 = 52.9 :=
by {
  -- Definitions and calculations have been provided in the problem statement
  sorry
}

end triangle_third_side_lengths_product_l504_504744


namespace fold_paper_problem_l504_504334

theorem fold_paper_problem :
  ∃ (p q : ℚ), 
    (∃ m1 m2 : ℚ × ℚ, m1 = (1,3) ∧ m2 = (5,1) ∧ is_perpendicular_bisector (line_through m1 m2) (line_y_equals_2x_minus4)) ∧
    (∃ m3 m4 : ℚ × ℚ, m3 = (8,4) ∧ m4 = (p,q) ∧ is_perpendicular_bisector (line_through m3 m4) (line_y_equals_2x_minus4)) ∧
    p + q = 20 / 3 :=
sorry

end fold_paper_problem_l504_504334


namespace sum_gt_product_iff_l504_504967

theorem sum_gt_product_iff (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : m + n > m * n ↔ m = 1 ∨ n = 1 :=
sorry

end sum_gt_product_iff_l504_504967


namespace greatest_possible_sum_of_digits_l504_504405

theorem greatest_possible_sum_of_digits 
  (n : ℕ) (a b d : ℕ) 
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_d : d ≠ 0)
  (h1 : ∃ n1 n2 : ℕ, n1 ≠ n2 ∧ (d * ((10 ^ (3 * n1) - 1) / 9) - b * ((10 ^ n1 - 1) / 9) = a^3 * ((10^n1 - 1) / 9)^3) 
                      ∧ (d * ((10 ^ (3 * n2) - 1) / 9) - b * ((10 ^ n2 - 1) / 9) = a^3 * ((10^n2 - 1) / 9)^3)) : 
  a + b + d = 12 := 
sorry

end greatest_possible_sum_of_digits_l504_504405


namespace product_of_possible_lengths_approx_l504_504737

noncomputable def hypotenuse (a b : ℝ) : ℝ :=
  real.sqrt (a * a + b * b)

noncomputable def other_leg (hypotenuse a : ℝ) : ℝ :=
  real.sqrt (hypotenuse * hypotenuse - a * a)

noncomputable def product_of_possible_lengths (a b : ℝ) : ℝ :=
  hypotenuse a b * other_leg (max a b) (min a b)

theorem product_of_possible_lengths_approx (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  Float.round (product_of_possible_lengths a b) 1 = 52.7 :=
by
  sorry

end product_of_possible_lengths_approx_l504_504737


namespace f_properties_l504_504285

open Real

-- Define the function f(x) = x^2
noncomputable def f (x : ℝ) : ℝ := x^2

-- Define the statement to be proved
theorem f_properties (x₁ x₂ : ℝ) (x : ℝ) (h : 0 < x) :
  (f (x₁ * x₂) = f x₁ * f x₂) ∧ 
  (deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end f_properties_l504_504285


namespace find_a6_l504_504506

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem find_a6 (a : ℕ → ℝ) (h : arithmetic_sequence a) (h2 : a 2 = 4) (h4 : a 4 = 2) : a 6 = 0 :=
by sorry

end find_a6_l504_504506


namespace imag_part_of_complex_exp_l504_504923

theorem imag_part_of_complex_exp :
  let θ := (2 * Real.pi) / 3
  in Complex.im (Complex.exp (Complex.I * θ)) = Real.sqrt 3 / 2 := sorry

end imag_part_of_complex_exp_l504_504923


namespace equal_perpendiculars_on_angle_bisector_l504_504579

variable {A B C D E F : Type}
variables [linear_ordered_field A] [metric_space A] [finite_dimensional A]

-- Conditions
variable (triangle_ABC : ∀ A B C : A, A ≠ B ∧ B ≠ C ∧ A ≠ C)
variable (angle_bisector_AD : ∀ (A B C : A), ∃ D : A, D ∈ line {A, B, C} ∧ 
                                               D ∈ angle_bisector {A, B, C})
variable (perp_bisector_BC : ∀ (B C : A), ∃ D : A, D ∈ perp_bisector {B, C})
variable (DE_perp_AB : ∀ (D E : A), D ∈ E ∧ perpendicular D E)
variable (DF_perp_AC : ∀ (D F : A), D ∈ F ∧ perpendicular D F)

-- Proof Problem Statement
theorem equal_perpendiculars_on_angle_bisector
  (ABC Δ : triangle_ABC A B C)
  (D_intersection : angle_bisector_AD A B C ∧ perp_bisector_BC B C)
  (DE_is_perp : DE_perp_AB D E)
  (DF_is_perp : DF_perp_AC D F) :
  dist B E = dist C F := 
sorry

end equal_perpendiculars_on_angle_bisector_l504_504579


namespace smallest_sum_of_consecutive_integers_l504_504260

theorem smallest_sum_of_consecutive_integers:
  ∃ (n m : ℕ), (n > 0) ∧ (20 * n + 190 = 2 * m^2) ∧ (20 * n + 190 = 450)  :=
by
  use 13, 15
  split; norm_num
  -- the proof steps would then follow
  sorry

end smallest_sum_of_consecutive_integers_l504_504260


namespace NH4OH_moles_formed_l504_504876

theorem NH4OH_moles_formed
  (moles_NH4Cl : ℕ)
  (moles_NaOH : ℕ)
  (moles_H2SO4 : ℕ)
  (moles_KOH : ℕ)
  (r1 : NH4Cl + NaOH → NH4OH + NaCl)
  (r2 : 2 * NH4Cl + H2SO4 → (NH4)2SO4 + 2 * HCl)
  (r3 : 2 * NaOH + H2SO4 → Na2SO4 + 2 * H2O)
  (r4 : KOH + H2SO4 → KHSO4 + H2O)
  (r5 : NH4OH + KOH → NH4K + H2O) :
  (moles_NH4Cl = 2) →
  (moles_NaOH = 2) →
  (moles_H2SO4 = 3) →
  (moles_KOH = 4) →
  count_moles_NH4OH = 0 :=
by sorry

end NH4OH_moles_formed_l504_504876


namespace simplify_sqrt_expression_is_correct_l504_504630

-- Definition for the given problem
def simplify_sqrt_expression (a b : ℝ) :=
  a = Real.sqrt (12 + 8 * Real.sqrt 3) → 
  b = Real.sqrt (12 - 8 * Real.sqrt 3) → 
  a + b = 4 * Real.sqrt 3

-- The theorem to be proven
theorem simplify_sqrt_expression_is_correct : simplify_sqrt_expression :=
begin
  intros a b ha hb,
  rw [ha, hb],
  -- Step-by-step simplification approach would occur here
  sorry
end

end simplify_sqrt_expression_is_correct_l504_504630


namespace find_n_for_log_sum_l504_504408

theorem find_n_for_log_sum :
  ∃ (n : ℕ), (∑ k in Finset.range (n + 1), Nat.floor (Real.log k / Real.log 2) ) = 1994 ∧ n = 312 :=
sorry

end find_n_for_log_sum_l504_504408


namespace rectangle_base_length_l504_504816

theorem rectangle_base_length
  (h : ℝ) (b : ℝ)
  (common_height_nonzero : h ≠ 0)
  (triangle_base : ℝ := 24)
  (same_area : (1/2) * triangle_base * h = b * h) :
  b = 12 :=
by
  sorry

end rectangle_base_length_l504_504816


namespace subset_of_intervals_l504_504025

def A (x : ℝ) := -2 ≤ x ∧ x ≤ 5
def B (m x : ℝ) := m + 1 ≤ x ∧ x ≤ 2 * m - 1
def is_subset_of (B A : ℝ → Prop) := ∀ x, B x → A x
def possible_values_m (m : ℝ) := m ≤ 3

theorem subset_of_intervals (m : ℝ) :
  is_subset_of (B m) A ↔ possible_values_m m := by
  sorry

end subset_of_intervals_l504_504025


namespace value_of_y_l504_504101

def sum_arithmetic_series (a1 an n : ℕ) : ℕ := n * (a1 + an) / 2

def num_even_integers (a1 an : ℕ) : ℕ := (an - a1) / 2 + 1

theorem value_of_y :
  let x := sum_arithmetic_series 10 30 21 in
  let y := num_even_integers 10 30 in
  x + y = 431 → y = 11 :=
by
  intros x y h
  sorry

end value_of_y_l504_504101


namespace sin_x_eq_2ab_div_a2_plus_b2_l504_504419

theorem sin_x_eq_2ab_div_a2_plus_b2
  (a b : ℝ) (x : ℝ)
  (h_tan : Real.tan x = 2 * a * b / (a^2 - b^2))
  (h_pos : 0 < b) (h_lt : b < a) (h_x : 0 < x ∧ x < Real.pi / 2) :
  Real.sin x = 2 * a * b / (a^2 + b^2) :=
by sorry

end sin_x_eq_2ab_div_a2_plus_b2_l504_504419


namespace simplify_sqrt_sum_l504_504649

noncomputable def sqrt_expr_1 : ℝ := Real.sqrt (12 + 8 * Real.sqrt 3)
noncomputable def sqrt_expr_2 : ℝ := Real.sqrt (12 - 8 * Real.sqrt 3)

theorem simplify_sqrt_sum : sqrt_expr_1 + sqrt_expr_2 = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_sum_l504_504649


namespace prove_percent_liquid_X_in_new_solution_l504_504829

variable (initial_solution total_weight_x total_weight_y total_weight_new)

def percent_liquid_X_in_new_solution : Prop :=
  let liquid_X_in_initial := 0.45 * 12
  let water_in_initial := 0.55 * 12
  let remaining_liquid_X := liquid_X_in_initial
  let remaining_water := water_in_initial - 5
  let liquid_X_in_added := 0.45 * 7
  let water_in_added := 0.55 * 7
  let total_liquid_X := remaining_liquid_X + liquid_X_in_added
  let total_water := remaining_water + water_in_added
  let total_weight := total_liquid_X + total_water
  (total_liquid_X / total_weight) * 100 = 61.07

theorem prove_percent_liquid_X_in_new_solution :
  percent_liquid_X_in_new_solution := by
  sorry

end prove_percent_liquid_X_in_new_solution_l504_504829


namespace distinguishable_tetrahedron_colorings_l504_504889

noncomputable def tetrahedron_colorings_equivalence : ℕ :=
  let num_faces := 4
  let colors := {red, green, blue, yellow}
  let num_permutations := nat.factorial (num_faces - 1) -- 3! permutations for the remaining faces
  let num_rotations := 3  -- 3 rotations around the fixed face
  in num_permutations / num_rotations

theorem distinguishable_tetrahedron_colorings : tetrahedron_colorings_equivalence = 2 := by
  sorry

end distinguishable_tetrahedron_colorings_l504_504889


namespace number_of_principals_in_oxford_high_school_l504_504606

-- Define the conditions
def numberOfTeachers : ℕ := 48
def numberOfClasses : ℕ := 15
def studentsPerClass : ℕ := 20
def totalStudents : ℕ := numberOfClasses * studentsPerClass
def totalPeople : ℕ := 349
def numberOfPrincipals : ℕ := totalPeople - (numberOfTeachers + totalStudents)

-- Proposition: Prove the number of principals in Oxford High School
theorem number_of_principals_in_oxford_high_school :
  numberOfPrincipals = 1 := by sorry

end number_of_principals_in_oxford_high_school_l504_504606


namespace total_cost_all_children_l504_504269

-- Defining the constants and conditions
def regular_tuition : ℕ := 45
def early_bird_discount : ℕ := 15
def first_sibling_discount : ℕ := 15
def additional_sibling_discount : ℕ := 10
def weekend_class_extra_cost : ℕ := 20
def multi_instrument_discount : ℕ := 10

def Ali_cost : ℕ := regular_tuition - early_bird_discount
def Matt_cost : ℕ := regular_tuition - first_sibling_discount
def Jane_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra_cost - multi_instrument_discount
def Sarah_cost : ℕ := regular_tuition - additional_sibling_discount + weekend_class_extra_cost - multi_instrument_discount

-- Proof statement
theorem total_cost_all_children : Ali_cost + Matt_cost + Jane_cost + Sarah_cost = 150 := by
  sorry

end total_cost_all_children_l504_504269


namespace sum_powers_of_two_l504_504271

theorem sum_powers_of_two (n : ℕ) : (finset.sum (finset.range n) (λ i, 2^i) = 2^n - 1) := by
  induction n with
  | zero => 
    simp [finset.sum, pow_succ, pow_zero]   -- Base case: 0
  | succ k ih => 
    simp [finset.range_succ, finset.sum_insert finset.not_mem_range_self, ih, pow_succ]
    ring                                 -- Inductive step: k+1
  sorry

end sum_powers_of_two_l504_504271


namespace sum_of_squares_of_rates_l504_504715

theorem sum_of_squares_of_rates :
  ∃ (b j s : ℕ), 3 * b + j + 5 * s = 89 ∧ 4 * b + 3 * j + 2 * s = 106 ∧ b^2 + j^2 + s^2 = 821 := 
by
  sorry

end sum_of_squares_of_rates_l504_504715


namespace intersection_polar_sum_l504_504116

variables {α ρ θ : ℝ}

def curve_C1 (α : ℝ) : Prop :=
  (λ (x y : ℝ), x = 2 + Real.cos α ∧ y = 2 + Real.sin α)

def line_C2 (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * x

noncomputable def polar_coordinates_oa_ob (A B : ℝ × ℝ) (O : ℝ × ℝ) :=
  let OA := Real.sqrt ((A.1 - O.1)^2 + (A.2 - O.2)^2) in
  let OB := Real.sqrt ((B.1 - O.1)^2 + (B.2 - O.2)^2) in
  1 / |OA| + 1 / |OB|

theorem intersection_polar_sum (OA OB : ℝ) (O : ℝ × ℝ) :
  (∀ α, ∃ x y, curve_C1 α x y) →
  (∀ x y, line_C2 x y) →
  polar_coordinates_oa_ob (2 + Real.cos OA, 2 + Real.sin OA) (2 + Real.cos OB, 2 + Real.sin OB) O = (2 + 2 * Real.sqrt 3) / 7 :=
sorry

end intersection_polar_sum_l504_504116


namespace exists_integers_for_S_geq_100_l504_504167

theorem exists_integers_for_S_geq_100 (S : ℤ) (hS : S ≥ 100) :
  ∃ (T C B : ℤ) (P : ℤ),
    T > 0 ∧ C > 0 ∧ B > 0 ∧
    T > C ∧ C > B ∧
    T + C + B = S ∧
    T * C * B = P ∧
    (∀ (T₁ C₁ B₁ T₂ C₂ B₂ : ℤ), 
      T₁ > 0 ∧ C₁ > 0 ∧ B₁ > 0 ∧ 
      T₂ > 0 ∧ C₂ > 0 ∧ B₂ > 0 ∧ 
      T₁ > C₁ ∧ C₁ > B₁ ∧ 
      T₂ > C₂ ∧ C₂ > B₂ ∧ 
      T₁ + C₁ + B₁ = S ∧ 
      T₂ + C₂ + B₂ = S ∧ 
      T₁ * C₁ * B₁ = T₂ * C₂ * B₂ → 
      (T₁ = T₂) ∧ (C₁ = C₂) ∧ (B₁ = B₂) → false) :=
sorry

end exists_integers_for_S_geq_100_l504_504167


namespace relationship_among_mean_median_modes_l504_504599

-- Defining the conditions
def occurrences_1_to_29 : Nat := 12
def occurrences_1_to_28 : Nat := 12
def occurrences_30 : Nat := 11
def occurrences_31 : Nat := 7

-- Given the number of occurrences and the values, prove the relationship between mean (μ), median (M), and median of the modes (d).
theorem relationship_among_mean_median_modes
    (μ M d : ℝ)
    (hM : M = 16)
    (hμ : μ ≈ 15.764)
    (hd : d = 15)
    (h_occurrences : occurrences_1_to_29 = 12 ∧ occurrences_1_to_28 = 12 ∧ occurrences_30 = 11 ∧ occurrences_31 = 7) :
    d < μ ∧ μ < M := by
  sorry

end relationship_among_mean_median_modes_l504_504599


namespace sum_of_squares_of_ranks_l504_504413

theorem sum_of_squares_of_ranks (n : ℕ) :
  let rank (r : ℕ) := r
  let sum_of_squares (regions : list ℕ) := regions.foldr (λ r acc, acc + r^2) 0
  let regions : list ℕ := -- Placeholder for the list of ranks of regions
    sorry
  (sum_of_squares regions) ≤ 10 * n^2 := 
sorry

end sum_of_squares_of_ranks_l504_504413


namespace smallest_k_for_distinct_real_roots_l504_504974

noncomputable def discriminant (a b c : ℝ) := b^2 - 4 * a * c

theorem smallest_k_for_distinct_real_roots :
  ∃ k : ℤ, (k > 0) ∧ discriminant (k : ℝ) (-3) (-9/4) > 0 ∧ (∀ m : ℤ, discriminant (m : ℝ) (-3) (-9/4) > 0 → m ≥ k) := 
by
  sorry

end smallest_k_for_distinct_real_roots_l504_504974


namespace total_interest_10_years_l504_504298

def principal (P : ℝ) : Prop := true
def interest_rate (R : ℝ) : Prop := true
def initial_interest (SI T : ℝ) : Prop := SI = (P * R * T) / 100
def principal_trebled (P : ℝ) : Prop := true
def increased_interest (SI' : ℝ) : Prop := SI' = ((3 * P) * R * 5) / 100
def final_interest (total : ℝ) : Prop := total = SI + SI'

theorem total_interest_10_years (P R SI SI' total : ℝ) (h1 : initial_interest SI 10) (h2 : initial_interest 400 10) (h3 : principal_trebled P) (h4 : increased_interest SI') (h5 : increased_interest 600) (h6 : final_interest total) : 
  total = 1000 :=
begin
  sorry
end

end total_interest_10_years_l504_504298


namespace length_of_BD_l504_504998

-- Definitions based on conditions
variable (A B C D E : Type) 
variable [right_triangle ABC]
variable [midpoint D BC]
variable [midpoint D AE]
variable [segment BC = 15]
variable [BEAC_parallelogram]

-- Target theorem to prove
theorem length_of_BD : length BD = 7.5 := sorry

end length_of_BD_l504_504998


namespace prime_count_between_50_and_70_l504_504066

open Nat

theorem prime_count_between_50_and_70 : 
  (finset.filter Nat.prime (finset.range 71 \ finset.range 51).card = 4) := 
begin
  sorry
end

end prime_count_between_50_and_70_l504_504066


namespace distance_between_x_intercepts_l504_504329

theorem distance_between_x_intercepts :
  ∀ (x1 x2 : ℝ),
  (∀ x, x1 = 8 → x2 = 20 → 20 = 4 * (x - 8)) → 
  (∀ x, x1 = 8 → x2 = 20 → 20 = 7 * (x - 8)) → 
  abs ((3 : ℝ) - (36 / 7)) = (15 / 7) :=
by
  intros x1 x2 h1 h2
  sorry

end distance_between_x_intercepts_l504_504329


namespace stock_decrease_required_l504_504349

theorem stock_decrease_required (x : ℝ) (h : x > 0) : 
  (∃ (p : ℝ), (1 - p) * 1.40 * x = x ∧ p * 100 = 28.57) :=
sorry

end stock_decrease_required_l504_504349


namespace rational_expression_nonnegative_l504_504476

theorem rational_expression_nonnegative (x : ℚ) : 2 * |x| + x ≥ 0 :=
  sorry

end rational_expression_nonnegative_l504_504476


namespace substitute_monomial_to_simplify_expr_l504_504183

theorem substitute_monomial_to_simplify_expr (k : ℤ) : 
  ( ∃ k : ℤ, (x^4 - 3)^2 + (x^3 + k * x)^2 after expanding has exactly four terms) := 
begin
  use 3,
  sorry
end

end substitute_monomial_to_simplify_expr_l504_504183


namespace positive_integer_solutions_count_l504_504143

def floor (x : ℝ) : ℤ := Int.floor x

theorem positive_integer_solutions_count :
  ({(x, y) | x > 0 ∧ y > 0 ∧ floor (1.9 * x) + floor (8.8 * y) = 36}.to_finset.card = 5) :=
by
  sorry

end positive_integer_solutions_count_l504_504143


namespace reflection_on_circumcircle_segment_ratio_half_l504_504137

variables {A B C P V O D E : Type} [Nonempty A] [Nonempty B] [Nonempty C]
variable [Nonempty P] [Nonempty V] [Nonempty O] [Nonempty D] [Nonempty E]

/-- Let ABC be a triangle where AC ≠ BC. Let P be the foot of the altitude taken from C to AB.
Let V be the orthocenter, O the circumcenter of ABC, and D the point of intersection between the
radius OC and the side AB. The midpoint of CD is E. -/
variables (AC_ne_BC : A ≠ B)
variable (foot_P : P)
variable (orthocenter_V : V)
variable (circumcenter_O : O)
variable (intersection_D : D)
variable (midpoint_E : E)

/-- Part a: Prove that the reflection V' of V in AB is on the circumcircle of the triangle ABC. -/
theorem reflection_on_circumcircle :
  ∃ V' : Type, V' ∈ circumcircle A B C :=
sorry

/-- Part b: Prove that the segment EP divides the segment OV in the ratio 1:1. -/
theorem segment_ratio_half :
  ∃ N9 : Type, midpoint E P N9 → divides_ratio EP OV 1 1 :=
sorry

end reflection_on_circumcircle_segment_ratio_half_l504_504137


namespace kai_manny_ratio_l504_504594

theorem kai_manny_ratio 
  (pieces_total manny_ate raphael_ate lisa_ate kai_ate: ℝ) 
  (H1 : pieces_total = 6)
  (H2 : manny_ate = 1)
  (H3 : raphael_ate = manny_ate / 2)
  (H4 : lisa_ate = 2 + raphael_ate)
  (H5 : ((pieces_total - (manny_ate + raphael_ate + lisa_ate)):ℝ) = kai_ate) :
  kai_ate / manny_ate = 2 :=
by 
  amt sorry

end kai_manny_ratio_l504_504594


namespace graph_of_g_shift_is_C_l504_504016

-- Define the piecewise function g
def g (x : ℝ) : ℝ :=
  if -2 ≤ x ∧ x ≤ 1 then -x
  else if 1 < x ∧ x ≤ 4 then (x - 3)^2 + 1
  else if 4 < x ∧ x ≤ 5 then x - 2
  else 0  -- out-of-domain behavior is automatic to avoid non-total function

-- Define the transformation function g_shift by shifting g by 2 units upward
def g_shift (x : ℝ) : ℝ := g(x) + 2

-- We need to prove that the graph of g_shift corresponds to the labeled graph "C"
-- This can be interpreted as follows:
theorem graph_of_g_shift_is_C : ∀ x ∈ Icc (-2) 5, true := by
  sorry

end graph_of_g_shift_is_C_l504_504016


namespace smallest_sum_of_consecutive_integers_l504_504261

theorem smallest_sum_of_consecutive_integers:
  ∃ (n m : ℕ), (n > 0) ∧ (20 * n + 190 = 2 * m^2) ∧ (20 * n + 190 = 450)  :=
by
  use 13, 15
  split; norm_num
  -- the proof steps would then follow
  sorry

end smallest_sum_of_consecutive_integers_l504_504261


namespace coefficient_of_x4_in_binomial_expansion_l504_504511

theorem coefficient_of_x4_in_binomial_expansion :
  (∃ n : ℕ, (Nat.choose n 0 + Nat.choose n 1 + Nat.choose n (n-1) + Nat.choose n n = 18) ∧
  ∑ i in (Finset.range (n+1)), binomial n i * (2x) ^ i * (1/(2x)) ^ (n - i) = C_8^6 * 2^4) :=
sorry

end coefficient_of_x4_in_binomial_expansion_l504_504511


namespace monotonicity_g_minimum_m_l504_504010

variable {a : ℝ}
noncomputable def f (x : ℝ) := Real.log x
noncomputable def g (x : ℝ) := f x - a * x

theorem monotonicity_g :
  (∀ x : ℝ, a ≤ 0 → (g x).derivative > 0) ∧
  (∀ x : ℝ, a > 0 → 
    (∀ x ∈ Ioo 0 (1/a : ℝ), (g x).derivative > 0) ∧ 
    (∀ x > (1/a : ℝ), (g x).derivative < 0)) :=
sorry

variable {m : ℤ}
noncomputable def F (x : ℝ) := (x - 2) * Real.exp x + f x - x - m

theorem minimum_m (m : ℤ) : 
  (∀ x ∈ Ioo (1/4) 1, F x ≤ 0) → m ≥ -3 :=
sorry

end monotonicity_g_minimum_m_l504_504010


namespace minimize_MP_projection_l504_504501

/-- Let ABC be an obtuse triangle with c being the obtuse angle,
    A point D is selected on side BC distinct from points B and C,
    A line AM passes through an interior point M (distinct from D) on segment BC,
    and intersects the circumcircle S of ABC at point N,
    A circle is drawn through points M, D, and N, intersecting S at N and another point P,
    Then point M should be the projection of P_0 onto BC to minimize the length of segment MP.-/
theorem minimize_MP_projection (A B C D P M N P_0 : Point)
  (h_acute : ∠C > pi / 2)
  (h_D_on_BC : D ∈ segment B C)
  (h_M_on_BC : M ∈ interior (segment B C))
  (h_AM_intersects_N : line A M ∩ circumcircle (triangle ABC) = {N})
  (h_circum_MDN : circle_through M D N ∩ circumcircle (triangle ABC) = {N, P})
  (h_parallel : parallel (line A P_0) (line B C))
  (h_kd_intersects_P0 : N ∈ interior (segment K D)) :
  M = projection P_0 (segment B C) :=
sorry

end minimize_MP_projection_l504_504501


namespace min_value_g_l504_504919

noncomputable theory

def f (x : ℝ) (φ : ℝ) : ℝ := √3 * sin (2 * x + φ) + cos (2 * x + φ)

def g (x : ℝ) (φ : ℝ) : ℝ := f (x - 3 * π / 4) φ

theorem min_value_g :
  ∀ (φ : ℝ), 0 < φ ∧ φ < π →
  (∀ x : ℝ, g x φ = -g (-x + π/6 + 3π/4) φ) →  -- This expresses symmetry around π/12 for f and the shift to the right by 3π/4 for g
  ∃ x : ℝ, x ∈ Icc (-π/4) (π/6) ∧ g x φ = -1 :=
by
  intros φ hφ hsymm
  sorry

end min_value_g_l504_504919


namespace arith_seq_int_ratio_l504_504950

theorem arith_seq_int_ratio (a b : ℕ → ℕ) (S T : ℕ → ℕ)
  (h_sum_S : ∀ n, S n = (n * (2 * n - 1)) / 2 + 30)
  (h_sum_T : ∀ n, T n = (n * (n - 1)) / 2 + 3)
  (h_ratio : ∀ n, S n / T n = (2 * n + 30) / (n + 3)) :
  (finset.range 12).filter (λ n, (a n / b n).nat_ceiling = a n / b n).card = 5 := 
sorry

end arith_seq_int_ratio_l504_504950


namespace probability_at_least_two_white_balls_l504_504788

theorem probability_at_least_two_white_balls :
  let total_ways := Nat.factorial 17 / (Nat.factorial 3 * Nat.factorial (17 - 3)) in
  let exactly_two_white := (Nat.factorial 8 / (Nat.factorial 2 * Nat.factorial (8 - 2))) *
                           (Nat.factorial 9 / (Nat.factorial 1 * Nat.factorial (9 - 1))) in
  let exactly_three_white := Nat.factorial 8 / (Nat.factorial 3 * Nat.factorial (8 - 3)) in
  let favorable_ways := exactly_two_white + exactly_three_white in
  let probability := favorable_ways / total_ways in
  probability = 154 / 340 :=
by
  sorry

end probability_at_least_two_white_balls_l504_504788


namespace principal_amount_l504_504697

def simple_interest (P R T : ℝ) : ℝ := (P * R * T) / 100

theorem principal_amount (SI : ℝ) (T : ℝ) (R : ℝ) (P : ℝ) :
  SI = simple_interest P R T → T = 5 → R = 4 → SI = 160 → P = 800 :=
by
  intros
  sorry

end principal_amount_l504_504697


namespace existintegers_inc_gcd_l504_504910

theorem existintegers_inc_gcd {d m : ℤ} (hd : d > 1) : 
  ∃ k l : ℤ, k > l ∧ l > 0 ∧ m < Int.gcd (2 ^ 2 ^ k + d) (2 ^ 2 ^ l + d) := 
sorry

end existintegers_inc_gcd_l504_504910


namespace f_properties_l504_504284

open Real

-- Define the function f(x) = x^2
noncomputable def f (x : ℝ) : ℝ := x^2

-- Define the statement to be proved
theorem f_properties (x₁ x₂ : ℝ) (x : ℝ) (h : 0 < x) :
  (f (x₁ * x₂) = f x₁ * f x₂) ∧ 
  (deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end f_properties_l504_504284


namespace problem_solution_l504_504464

noncomputable def collinear (m n : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ n.1 = k * m.1 ∧ n.2 = k * m.2

theorem problem_solution
  (A : ℝ)
  (BC : ℝ)
  (m : ℝ × ℝ)
  (n : ℝ × ℝ)
  (condition1 : m = (Real.sin A, 1/2))
  (condition2 : n = (3, Real.sin A + Real.sqrt 3 * Real.cos A))
  (condition3 : BC = 2)
  (collinear_mn : collinear m n)
  (interior_angle_A : 0 < A ∧ A < Real.pi) :
  A = Real.pi / 3 ∧
  ∃ S : ℝ,
  let b := 2 / (Real.sqrt 3)
      c := 2 / (Real.sqrt 3) in
  let area := b * c * (Real.sqrt 3) / 4 in
  S = Real.sqrt 3 ∧
  (S = area ∧ b = c) :=
sorry

end problem_solution_l504_504464


namespace substitute_monomial_to_simplify_expr_l504_504187

theorem substitute_monomial_to_simplify_expr (k : ℤ) : 
  ( ∃ k : ℤ, (x^4 - 3)^2 + (x^3 + k * x)^2 after expanding has exactly four terms) := 
begin
  use 3,
  sorry
end

end substitute_monomial_to_simplify_expr_l504_504187


namespace integral_solution_l504_504300

noncomputable def integral_problem : Prop := 
  ∫ (x : ℝ) in 0..x, (x^3 + 6 * x^2 + 18 * x - 4) / ((x - 2) * (x + 2)^3) = 
    ln |x - 2| - 3 / (x + 2)^2 + C

theorem integral_solution : integral_problem :=
sorry

end integral_solution_l504_504300


namespace average_greater_than_median_l504_504029

/-- Assuming Hammie weighs 120 pounds and his triplet siblings weigh 4, 7, and 10 pounds respectively,
prove that the average weight is greater than the median weight by 26.75 pounds.
-/
theorem average_greater_than_median : 
  let weights := [120, 4, 7, 10] in
  let sorted_weights := List.sort weights in
  let median := (sorted_weights.nth 1 + sorted_weights.nth 2) / 2 in
  let average := (weights.sum) / (weights.length) in
  average - median = 26.75 :=
by
  sorry

end average_greater_than_median_l504_504029


namespace xiao_wang_total_amount_l504_504292

variable (P : ℝ) (x : ℝ)

def annualInterest (x : ℝ) : ℝ :=
  1 + x / 100

def totalAmountAfterTwoYears (P : ℝ) (x : ℝ) : ℝ :=
  P * (annualInterest x)^2

theorem xiao_wang_total_amount (P : ℝ) (x : ℝ) (hP : P = 5000) : 
    totalAmountAfterTwoYears P x = 5000 * (1 + x / 100)^2 := by
  sorry

end xiao_wang_total_amount_l504_504292


namespace outlets_per_room_l504_504325

theorem outlets_per_room
  (rooms : ℕ)
  (total_outlets : ℕ)
  (h1 : rooms = 7)
  (h2 : total_outlets = 42) :
  total_outlets / rooms = 6 :=
by sorry

end outlets_per_room_l504_504325


namespace prime_count_between_50_and_70_l504_504038

theorem prime_count_between_50_and_70 : 
  (finset.filter nat.prime (finset.range 71).filter (λ n, 50 < n ∧ n < 71)).card = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504038


namespace gcd_3060_561_l504_504752

theorem gcd_3060_561 : Nat.gcd 3060 561 = 51 :=
by
  sorry

end gcd_3060_561_l504_504752


namespace solve_trig_identity_problem_l504_504415

noncomputable def trig_identity_proof_problem (α : ℝ) : Prop :=
  tan (α + (Real.pi / 4)) = 1 / 2 ∧ - (Real.pi / 2) < α ∧ α < 0 →
  (2 * (Real.sin α) ^ 2 + Real.sin (2 * α)) / (Real.cos (α - (Real.pi / 4))) = - (2 * Real.sqrt 5) / 5

-- Statement to assert the theorem
theorem solve_trig_identity_problem (α : ℝ) : trig_identity_proof_problem α :=
by 
  sorry

end solve_trig_identity_problem_l504_504415


namespace count_4_primable_below_1000_is_4_l504_504338

def is_one_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_4_primable (n : ℕ) : Prop :=
  (n % 4 = 0) ∧ (∀ d ∈ n.digits 10, is_one_digit_prime d)

def count_4_primable_below_1000 : ℕ :=
  (List.range 1000).countp is_4_primable

theorem count_4_primable_below_1000_is_4 : count_4_primable_below_1000 = 4 :=
by sorry

end count_4_primable_below_1000_is_4_l504_504338


namespace worstPlayerIsGrandson_l504_504800

-- Define the four players as types
inductive Player
| grandfather : Player
| son : Player
| granddaughter : Player
| grandson : Player

-- Define the conditions given in the problem as definitions
def playingChess : Player → Bool :=
  λ p, p ∈ [Player.grandfather, Player.son, Player.granddaughter, Player.grandson]

def oppositeSex (p1 p2 : Player) : Bool :=
  p1 ≠ p2 ∧ (p1 = Player.granddaughter ∨ p1 = Player.grandson) ∧ (p2 = Player.son ∨ p2 = Player.grandfather)

def sameAge (p1 p2 : Player) : Bool :=
  p1 ≠ p2 ∧ ((p1 = Player.grandson ∧ p2 = Player.son) ∨ (p1 = Player.son ∧ p2 = Player.grandson))

-- Define the predicate to check if someone is the worst player
def isWorst (p : Player) : Bool :=
  match p with
  | Player.grandfather => False
  | Player.son => False
  | Player.granddaughter => False
  | Player.grandson => True

-- State the theorem
theorem worstPlayerIsGrandson : ∀ (p : Player),
  playingChess p →
  (∃ (w b : Player), isWorst w ∧ oppositeSex w b ∧ sameAge w b) →
  p = Player.grandson :=
by
  intros p hp hex
  have : p = Player.grandson, from sorry
  exact this

end worstPlayerIsGrandson_l504_504800


namespace measure_angle_B_function_period_function_max_min_l504_504491

theorem measure_angle_B 
  (a b c : ℝ) (A B C : ℝ) (h1 : 0 < A) (h2 : A < π) 
  (h3 : 0 < B) (h4 : B < π) 
  (h5 : 0 < C) (h6 : C < π) 
  (h7 : real.sin A / a = real.sin B / b) 
  (h8 : real.sin A / a = real.sin C / c)
  (h_cond : b * real.cos C + c * real.cos B = 2 * a * real.cos B) : 
  B = π / 3 := 
sorry

theorem function_period
  (x : ℝ) :
  let f := λ x : ℝ, real.sin (2 * x + π / 3) + real.sin (2 * x - π / 3) + 2 * real.cos x ^ 2 - 1
  in real.is_periodic f π :=
sorry

theorem function_max_min 
  (x : ℝ) 
  (h1 : -π / 4 <= x) 
  (h2 : x <= π / 4) :
  let f := λ x : ℝ, real.sin (2 * x + π / 3) + real.sin (2 * x - π / 3) + 2 * real.cos x ^ 2 - 1
  in (∀ x, -π / 4 <= x → x <= π / 4 → f x ≤ sqrt 2) ∧ 
     (∀ x, -π / 4 <= x → x <= π / 4 → f x >= -1) :=
sorry

end measure_angle_B_function_period_function_max_min_l504_504491


namespace count_4_primable_numbers_lt_1000_l504_504343

def is_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_n_primable (n k : ℕ) : Prop :=
  k % n = 0 ∧ ∀ d, d ∈ (k.digits 10) → is_digit_prime d

def is_4_primable (k : ℕ) : Prop :=
  is_n_primable 4 k

theorem count_4_primable_numbers_lt_1000 : 
  ∃ n, n = 21 ∧ n = (Finset.filter is_4_primable (Finset.range 1000)).card :=
sorry

end count_4_primable_numbers_lt_1000_l504_504343


namespace line_through_origin_and_intersection_of_lines_l504_504398

theorem line_through_origin_and_intersection_of_lines 
  (x y : ℝ)
  (h1 : x - 3 * y + 4 = 0)
  (h2 : 2 * x + y + 5 = 0) :
  3 * x + 19 * y = 0 :=
sorry

end line_through_origin_and_intersection_of_lines_l504_504398


namespace master_craftsman_total_parts_l504_504513

theorem master_craftsman_total_parts
  (N : ℕ) -- Additional parts to be produced after the first hour
  (initial_rate : ℕ := 35) -- Initial production rate (35 parts/hour)
  (increased_rate : ℕ := initial_rate + 15) -- Increased production rate (50 parts/hour)
  (time_difference : ℝ := 1.5) -- Time difference in hours between the rates
  (eq_time_diff : (N / initial_rate) - (N / increased_rate) = time_difference) -- The given time difference condition
  : 35 + N = 210 := -- Conclusion we need to prove
sorry

end master_craftsman_total_parts_l504_504513


namespace range_of_a_l504_504689

theorem range_of_a (a : ℝ) : 
  ((-1 + a) ^ 2 + (-1 - a) ^ 2 < 4) ↔ (-1 < a ∧ a < 1) := 
by
  sorry

end range_of_a_l504_504689


namespace sequence_term_position_l504_504023

theorem sequence_term_position :
  ∀ n : ℕ, (∀ k, k = 3 * n + 1 → sqrt k = 2 * sqrt 7 → n = 9) :=
by
  sorry

end sequence_term_position_l504_504023


namespace farmer_apples_count_l504_504232

-- Definitions from the conditions in step a)
def initial_apples : ℕ := 127
def apples_given_away : ℕ := 88

-- Proof goal from step c)
theorem farmer_apples_count : initial_apples - apples_given_away = 39 :=
by
  sorry

end farmer_apples_count_l504_504232


namespace primes_between_50_and_70_l504_504079

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).filter is_prime |>.length

theorem primes_between_50_and_70 : count_primes_in_range 50 70 = 4 :=
by
  sorry

end primes_between_50_and_70_l504_504079


namespace f_iterated_l504_504585

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 3 else 4 * x + 1

theorem f_iterated (h : f (f (f (f 1)))) = 341 : Proposition := by
  sorry

end f_iterated_l504_504585


namespace a_seq_2015_l504_504907

noncomputable def a_seq : ℕ → ℚ
| 0       := 4/5
| (n + 1) := if 0 ≤ a_seq n ∧ a_seq n ≤ 1/2 then 2 * a_seq n
             else if 1/2 < a_seq n ∧ a_seq n ≤ 1 then 2 * a_seq n - 1
             else 0 -- Fallback if out of bounds

theorem a_seq_2015 : a_seq 2014 = 1/5 := by
  sorry

end a_seq_2015_l504_504907


namespace shaded_area_of_rotated_semicircle_l504_504874

-- Definitions and conditions from the problem
def radius (R : ℝ) : Prop := R > 0
def central_angle (α : ℝ) : Prop := α = 30 * (Real.pi / 180)

-- Lean theorem statement for the proof problem
theorem shaded_area_of_rotated_semicircle (R : ℝ) (hR : radius R) (hα : central_angle 30) : 
  ∃ (area : ℝ), area = (Real.pi * R^2) / 3 :=
by
  -- using proofs of radius and angle conditions
  sorry

end shaded_area_of_rotated_semicircle_l504_504874


namespace luca_milk_calculation_l504_504155

def milk_needed (flour mL: ℕ) (milk_per_300mL: ℕ) (batch_count: ℕ) : ℕ :=
  (flour mL / 300) * milk_per_300mL * batch_count

theorem luca_milk_calculation:
  ∀ (flour mL: ℕ) (milk_per_300mL: ℕ) (batch_count: ℕ),
  flour mL = 1200 ∧ milk_per_300mL = 60 →
  milk_needed flour mL milk_per_300mL 1 = 240 ∧ milk_needed flour mL milk_per_300mL 2 = 480 :=
by
  intros
  sorry

end luca_milk_calculation_l504_504155


namespace part_a_part_b_l504_504447

variable {f : ℝ → ℝ} 

-- Given conditions
axiom condition1 (x y : ℝ) : f (x + y) + 1 = f x + f y
axiom condition2 : f (1/2) = 0
axiom condition3 (x : ℝ) : x > 1/2 → f x < 0

-- Part (a)
theorem part_a (x : ℝ) : f x = 1/2 + 1/2 * f (2 * x) :=
sorry

-- Part (b)
theorem part_b (n : ℕ) (hn : n > 0) (x : ℝ) 
  (hx : 1 / 2^(n + 1) ≤ x ∧ x ≤ 1 / 2^n) : f x ≤ 1 - 1 / 2^n :=
sorry

end part_a_part_b_l504_504447


namespace combinatorial_arithmetic_combinatorial_sum_l504_504313

/-- 7C_6^3 - 4C_7^4 -/
theorem combinatorial_arithmetic :
  (7 * (nat.comb 6 3)) - (4 * (nat.comb 7 4)) = 0 :=
by
  sorry

/-- Given m, n ∈ ℕ* and n ≥ m, prove 
(m+1)C_m^m + (m+2)C_{m+1}^m + ... + (n+1)C_n^m = (m+1)C_{n+2}^{m+2} -/
theorem combinatorial_sum (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) (h : n ≥ m) :
  ((list.range (n+1)).sum (λ k, (k + m).choose m)) =
  (m+1) * (n + 3).choose (m + 2) :=
by
  sorry

end combinatorial_arithmetic_combinatorial_sum_l504_504313


namespace Q_has_at_least_2n_minus_1_roots_l504_504560

noncomputable def P (x : ℝ) : ℝ := sorry

def Q (P : ℝ → ℝ) (P' : ℝ → ℝ) (x : ℝ) : ℝ :=
  (x^2 + 1) * P(x) * P'(x) + x * (P(x)^2 + P'(x)^2)

theorem Q_has_at_least_2n_minus_1_roots (P : ℝ → ℝ) (P' : ℝ → ℝ) (n : ℕ) (hP : ∀ d > 1, ∃ e, P(e) = 0) 
  : ∃ x : ℝ, Q P P' x = 0 ∧ ∃ S : set ℝ, S.card ≥ 2 * n - 1 ∧ ∀ x ∈ S, Q P P' x = 0 :=
by
  sorry

end Q_has_at_least_2n_minus_1_roots_l504_504560


namespace simplify_sqrt_expression_is_correct_l504_504629

-- Definition for the given problem
def simplify_sqrt_expression (a b : ℝ) :=
  a = Real.sqrt (12 + 8 * Real.sqrt 3) → 
  b = Real.sqrt (12 - 8 * Real.sqrt 3) → 
  a + b = 4 * Real.sqrt 3

-- The theorem to be proven
theorem simplify_sqrt_expression_is_correct : simplify_sqrt_expression :=
begin
  intros a b ha hb,
  rw [ha, hb],
  -- Step-by-step simplification approach would occur here
  sorry
end

end simplify_sqrt_expression_is_correct_l504_504629


namespace simplify_sqrt_sum_l504_504633

theorem simplify_sqrt_sum :
  sqrt (12 + 8 * sqrt 3) + sqrt (12 - 8 * sqrt 3) = 4 * sqrt 3 := 
by
  -- Proof would go here
  sorry

end simplify_sqrt_sum_l504_504633


namespace simplify_sqrt_sum_l504_504631

theorem simplify_sqrt_sum :
  sqrt (12 + 8 * sqrt 3) + sqrt (12 - 8 * sqrt 3) = 4 * sqrt 3 := 
by
  -- Proof would go here
  sorry

end simplify_sqrt_sum_l504_504631


namespace divide_pyramid_volume_l504_504609

/-- Given points on a pyramid and certain ratios, prove the volume division ratio. -/
theorem divide_pyramid_volume (A B C D N K M : Point) 
  (h₁ : segment_ratio C N B = 2)
  (h₂ : segment_ratio D K C = 3 / 2)
  (h3 : is_centroid M A B D) :
  divides_volume_ratio (plane M N K) ABCD = 37 / 68 :=
begin
  sorry
end

end divide_pyramid_volume_l504_504609


namespace right_triangle_third_side_product_l504_504735

theorem right_triangle_third_side_product :
  let a := 6
  let b := 8
  let c1 := Real.sqrt (a^2 + b^2)     -- Hypotenuse when a and b are legs
  let c2 := Real.sqrt (b^2 - a^2)     -- Other side when b is the hypotenuse
  20 * Real.sqrt 7 ≈ 52.7 := 
by
  sorry

end right_triangle_third_side_product_l504_504735


namespace right_triangle_side_product_l504_504725

theorem right_triangle_side_product :
  let a := 6
  let b := 8
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let other_leg := Real.sqrt (b^2 - a^2)
  (hypotenuse * 2 * Real.sqrt 7).round = 53 := -- using 53 to consider rounding to the nearest tenth

by
  let a := 6
  let b := 8
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let other_leg := Real.sqrt (b^2 - a^2)
  have h1 : hypotenuse = 10 := by sorry
  have h2 : other_leg = 2 * Real.sqrt 7 := by sorry
  have h_prod : (hypotenuse * 2 * Real.sqrt 7).round = 53 := by sorry
  exact h_prod

end right_triangle_side_product_l504_504725


namespace circumcircle_PLQ_tangent_BC_l504_504575

theorem circumcircle_PLQ_tangent_BC 
  (A B C P Q L: Type*)
  [Triangle A B C] 
  [AngleBisector A B C L] 
  [PerpendicularBisector A L (Circumcircle A B C) P Q] :
  Tangent (Circumcircle P L Q) (Side B C) :=
sorry

end circumcircle_PLQ_tangent_BC_l504_504575


namespace largest_n_l504_504564

variable {a : ℕ → ℤ}
variable {S : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

axiom a1_gt_zero : a 1 > 0
axiom a2011_a2012_sum_gt_zero : a 2011 + a 2012 > 0
axiom a2011_a2012_prod_lt_zero : a 2011 * a 2012 < 0

-- Sum of first n terms of an arithmetic sequence
def sequence_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + a n) / 2

-- Problem statement to prove
theorem largest_n (H : is_arithmetic_sequence a) :
  ∀ n, (sequence_sum a 4022 > 0) ∧ (sequence_sum a 4023 < 0) → n = 4022 := by
  sorry

end largest_n_l504_504564


namespace parts_manufactured_l504_504541

variable (initial_parts : ℕ) (initial_rate : ℕ) (increased_speed : ℕ) (time_diff : ℝ)
variable (N : ℕ)

-- initial conditions
def initial_parts := 35
def initial_rate := 35
def increased_speed := 15
def time_diff := 1.5

-- additional parts to be manufactured
noncomputable def additional_parts := N

-- equation representing the time differences
noncomputable def equation := (N / initial_rate) - (N / (initial_rate + increased_speed)) = time_diff

-- state the proof problem
theorem parts_manufactured : initial_parts + additional_parts = 210 :=
by
  -- Use the given conditions to solve the problem
  sorry

end parts_manufactured_l504_504541


namespace trapezoid_perimeter_l504_504832

noncomputable def isosceles_trapezoid_perimeter (R : ℝ) (α : ℝ) (hα : α < π / 2) : ℝ :=
  8 * R / (Real.sin α)

theorem trapezoid_perimeter (R : ℝ) (α : ℝ) (hα : α < π / 2) :
  ∃ (P : ℝ), P = isosceles_trapezoid_perimeter R α hα := by
    sorry

end trapezoid_perimeter_l504_504832


namespace total_time_proof_l504_504596

variable (mow_time : ℕ) (fertilize_time : ℕ) (total_time : ℕ)

-- Based on the problem conditions.
axiom mow_time_def : mow_time = 40
axiom fertilize_time_def : fertilize_time = 2 * mow_time
axiom total_time_def : total_time = mow_time + fertilize_time

-- The proof goal
theorem total_time_proof : total_time = 120 := by
  sorry

end total_time_proof_l504_504596


namespace solve_trigonometric_equation_l504_504664

noncomputable theory
open Real

theorem solve_trigonometric_equation (t : ℝ) (k : ℤ) :
  2 * (sin (2 * t))^5 - (sin (2 * t))^3 - 6 * (sin (2 * t))^2 + 3 = 0 ↔ 
  t = π / 8 * (2 * k + 1) :=
by {
  sorry
}

end solve_trigonometric_equation_l504_504664


namespace largest_interior_angle_of_triangle_l504_504918

-- Definitions for the sides of the triangle and the given conditions
variables (a b c : ℝ) (triangle_ABC : ∀ (a b c : ℝ), (c = 5) → (√(a - 4) + (b - 3)^2 = 0) → True)

namespace triangle_proof

-- Statement for the largest interior angle
theorem largest_interior_angle_of_triangle:
  ∀ (a b c : ℝ), c=5 → (√(a-4) + (b-3)^2 = 0) → a = 4 ∧ b = 3 → ∃ (θ : ℝ), θ = 90 :=
by
  intros a b c h1 h2 h3,
  use 90,
  exact sorry

end triangle_proof

end largest_interior_angle_of_triangle_l504_504918


namespace anna_pays_total_l504_504833

-- Define the conditions
def daily_rental_cost : ℝ := 35
def cost_per_mile : ℝ := 0.25
def rental_days : ℝ := 3
def miles_driven : ℝ := 300

-- Define the total cost function
def total_cost (daily_rental_cost cost_per_mile rental_days miles_driven : ℝ) : ℝ :=
  (daily_rental_cost * rental_days) + (cost_per_mile * miles_driven)

-- The statement to be proved
theorem anna_pays_total : total_cost daily_rental_cost cost_per_mile rental_days miles_driven = 180 :=
by
  sorry

end anna_pays_total_l504_504833


namespace ganesh_average_speed_l504_504779

variable (D : ℝ) -- distance between the two towns in kilometers
variable (V : ℝ) -- average speed from x to y in km/hr

-- Conditions
variable (h1 : V > 0) -- Speed must be positive
variable (h2 : 30 > 0) -- Speed must be positive
variable (h3 : 40 = (2 * D) / ((D / V) + (D / 30))) -- Average speed formula

theorem ganesh_average_speed : V = 60 :=
by {
  sorry
}

end ganesh_average_speed_l504_504779


namespace replace_asterisk_with_monomial_l504_504192

theorem replace_asterisk_with_monomial :
  ∀ (x : ℝ), (x^4 - 3)^2 + (x^3 + 3x)^2 = x^8 + x^6 + 9x^2 + 9 := 
by
  intro x
  calc
    (x^4 - 3)^2 + (x^3 + 3x)^2
        = (x^4)^2 - 2 * x^4 * 3 + 3^2 + (x^3)^2 + 2 * x^3 * 3x + (3x)^2 : by ring
    ... = x^8 - 6 * x^4 + 9 + x^6 + 6 * x^4 + 9 * x^2 : by ring
    ... = x^8 + x^6 + 9 * x^2 + 9 : by ring
  sorry

end replace_asterisk_with_monomial_l504_504192


namespace trapezoid_midsegment_length_l504_504124

theorem trapezoid_midsegment_length (PQRS : Type*) [trapezoid PQRS]
  (QR PS : ℝ) (angleP angleS : ℝ) (X Y : PQRS)
  (hpqrs : QR ∥ PS)
  (hqr : QR = 1500)
  (hps : PS = 3000)
  (hp : angleP = 27)
  (hs : angleS = 63)
  (hmx : midpoint X QR)
  (hmy : midpoint Y PS) :
  dist X Y = 750 :=
sorry

end trapezoid_midsegment_length_l504_504124


namespace count_4_primable_below_1000_is_4_l504_504340

def is_one_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_4_primable (n : ℕ) : Prop :=
  (n % 4 = 0) ∧ (∀ d ∈ n.digits 10, is_one_digit_prime d)

def count_4_primable_below_1000 : ℕ :=
  (List.range 1000).countp is_4_primable

theorem count_4_primable_below_1000_is_4 : count_4_primable_below_1000 = 4 :=
by sorry

end count_4_primable_below_1000_is_4_l504_504340


namespace imaginary_part_of_z_l504_504441

noncomputable theory

open Complex

theorem imaginary_part_of_z : 
  ∃ z : ℂ, z = 1 / (1 + I) ∧ im z = -1/2 :=
begin
  use 1 / (1 + I),
  split,
  { 
    simp, -- This is a placeholder hint for simplification
  },
  {
    simp, -- This is a placeholder hint for simplification
    sorry
  }
end

end imaginary_part_of_z_l504_504441


namespace circle_intersection_count_l504_504378

theorem circle_intersection_count :
  let r1 θ := 3 * Real.cos θ
  let r2 θ := 6 * Real.sin θ
  ∀ θ₁ θ₂, r1 θ₁ = r2 θ₁ ∧ r1 θ₂ = r2 θ₂ →
  (θ₁ ≠ θ₂) → 2 :=
sorry

end circle_intersection_count_l504_504378


namespace min_positive_period_f_range_f_l504_504448

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * Real.sin (2 * x - Real.pi / 3)

theorem min_positive_period_f : ∀ x : ℝ, f (x + π) = f x :=
by
  intro x
  unfold f
  sorry

theorem range_f : set.range f = {y | y ≥ -(1/2) ∧ y ≤ 1/2} :=
by
  sorry

end min_positive_period_f_range_f_l504_504448


namespace ratio_of_areas_l504_504616

-- Define a regular hexagon ABCDEF and its properties
def regular_hexagon (A B C D E F : Type) := 
  -- Attributes or conditions that make it regular can be included here
  sorry

-- Triangle ABM, where M is the midpoint of BC
def triangle_ABM (A B M : Type) :=
  -- Attributes or conditions for triangle ABM can be included here
  sorry

-- Triangle ADF formed by connecting points A, D, and F
def triangle_ADF (A D F : Type) :=
  -- Attributes or conditions for triangle ADF can be included here
  sorry

-- Ratio of the areas
theorem ratio_of_areas (k : ℝ) (A B C D E F M : Type)
  (hexagon : regular_hexagon A B C D E F)
  (triangle1 : triangle_ABM A B M)
  (triangle2 : triangle_ADF A D F) :
  (1 / 12 : ℝ) := 
begin
  -- Proof omitted
  sorry
end

end ratio_of_areas_l504_504616


namespace count_4_primable_below_1000_is_4_l504_504339

def is_one_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_4_primable (n : ℕ) : Prop :=
  (n % 4 = 0) ∧ (∀ d ∈ n.digits 10, is_one_digit_prime d)

def count_4_primable_below_1000 : ℕ :=
  (List.range 1000).countp is_4_primable

theorem count_4_primable_below_1000_is_4 : count_4_primable_below_1000 = 4 :=
by sorry

end count_4_primable_below_1000_is_4_l504_504339


namespace simplify_sqrt_sum_l504_504635

theorem simplify_sqrt_sum :
  sqrt (12 + 8 * sqrt 3) + sqrt (12 - 8 * sqrt 3) = 4 * sqrt 3 := 
by
  -- Proof would go here
  sorry

end simplify_sqrt_sum_l504_504635


namespace range_of_a_for_monotonicity_l504_504480

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

theorem range_of_a_for_monotonicity
  (f : ℝ → ℝ)
  (a : ℝ)
  (h_a_pos : 0 < a)
  (h_a_ne_one : a ≠ 1)
  (h_f : ∀ x, f x = Real.log (2 - a * x))
  (h_incr : is_monotonically_increasing f { x | 1 < x ∧ x < 3 }) :
  0 < a ∧ a ≤ 2 / 3 :=
sorry

end range_of_a_for_monotonicity_l504_504480


namespace sum_of_box_dimensions_l504_504344

theorem sum_of_box_dimensions (X Y Z : ℝ) (h1 : X * Y = 32) (h2 : X * Z = 50) (h3 : Y * Z = 80) :
  X + Y + Z = 25.5 * Real.sqrt 2 :=
by sorry

end sum_of_box_dimensions_l504_504344


namespace decode_SAGE_l504_504702

def code_map : String → ℕ
| "M" := 0
| "A" := 1
| "G" := 2
| "I" := 3
| "C" := 4
| "H" := 5
| "O" := 6
| "R" := 7
| "S" := 8
| "E" := 9
| _ := 0  -- default case, but should not occur given the conditions

def decode_word (word : String) : ℕ :=
let digits := word.toList.map code_map in
digits.foldl (λ acc digit, acc * 10 + digit) 0

theorem decode_SAGE : decode_word "SAGE" = 8129 :=
sorry

end decode_SAGE_l504_504702


namespace exists_k_inequality_l504_504422

theorem exists_k_inequality (m : ℕ) : 
  ∃ k : ℕ, 1 ≤ (∑ i in Finset.range (k-1), (i + 1)^m) / (k^m) ∧ (∑ i in Finset.range (k-1), (i + 1)^m) / (k^m) < 2 := 
sorry

end exists_k_inequality_l504_504422


namespace replace_asterisk_with_monomial_l504_504200

theorem replace_asterisk_with_monomial (x : ℝ) :
  (∀ asterisk : ℝ, ((x^4 - 3)^2 + (x^3 + asterisk)^2) = (x^8 + x^6 + 9x^2 + 9)) ↔ asterisk = 3x :=
by sorry

end replace_asterisk_with_monomial_l504_504200


namespace angle_C_cos_B_l504_504982

-- Definitions based on conditions
section
variables (A B C : ℝ) -- angles of the triangle
variables (a b c : ℝ)  -- sides opposite to the angles A, B, and C

-- sin values of the angles
variables (sinA sinB sinC cosA cosB : ℝ)

-- given vectors in conditions
def m : ℝ × ℝ := (sinB + sinC, sinA - sinB)
def n : ℝ × ℝ := (sinB - sinC, sinA)

-- condition that vectors are perpendicular
axiom perpendicular_vectors : (m.1 * n.1 + m.2 * n.2) = 0

-- condition that sinA = 4/5
axiom sinA_condition : sinA = 4 / 5

-- main theorem to prove angle C and cos B
theorem angle_C_cos_B :
  (C = π / 3) ∧ (cosB = (4 * real.sqrt 3 - 3) / 10) :=
sorry
end

end angle_C_cos_B_l504_504982


namespace primes_between_50_and_70_l504_504077

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).filter is_prime |>.length

theorem primes_between_50_and_70 : count_primes_in_range 50 70 = 4 :=
by
  sorry

end primes_between_50_and_70_l504_504077


namespace ratio_of_square_to_rectangle_perimeters_l504_504809

noncomputable def ratio_of_perimeters : ℚ :=
let side_length := 6
let x := (side_length / 4 : ℚ) -- x = 1.5 inches
let perimeter_square := 4 * x -- Perimeter of the small square
let perimeter_rectangle := 2 * (x + 3 * x) -- Perimeter of the long rectangle
in perimeter_square / perimeter_rectangle

theorem ratio_of_square_to_rectangle_perimeters : ratio_of_perimeters = 1 / 2 := 
by
  sorry

end ratio_of_square_to_rectangle_perimeters_l504_504809


namespace coronavirus_case_ratio_l504_504686

theorem coronavirus_case_ratio (n_first_wave_cases : ℕ) (total_second_wave_cases : ℕ) (n_days : ℕ) 
  (h1 : n_first_wave_cases = 300) (h2 : total_second_wave_cases = 21000) (h3 : n_days = 14) :
  (total_second_wave_cases / n_days) / n_first_wave_cases = 5 :=
by sorry

end coronavirus_case_ratio_l504_504686


namespace master_craftsman_quota_l504_504525

theorem master_craftsman_quota (parts_first_hour : ℕ)
  (extra_hour_needed : ℕ)
  (increased_speed : ℕ)
  (time_diff : ℕ)
  (total_parts : ℕ) :
  parts_first_hour = 35 →
  extra_hour_needed = 1 →
  increased_speed = 15 →
  time_diff = 1.5 →
  total_parts = parts_first_hour + (175 : ℕ) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3]
  norm_num
  rw [add_comm]
  exact sorry

end master_craftsman_quota_l504_504525


namespace find_mplusn_l504_504245

-- Define relatively prime
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the conditions for the problem
variable {A P M : Type}
variable (perimeter_AM : ℕ) (angle_PAM_right : Prop) (circle_radius : ℕ)
variable (center_O_on_AP : Prop) (circle_tangent_AM_PM : Prop)
variable (m n : ℕ) (h_rel_prime : relatively_prime m n) := 
  (152 : perimeter_AM = 152) ∧
  (19 : circle_radius = 19) ∧ 
  angle_PAM_right ∧ 
  center_O_on_AP ∧ 
  circle_tangent_AM_PM

-- The statement to prove
theorem find_mplusn : 
  ∃ (m n : ℕ), relatively_prime m n ∧ 
               (OP : ℚ) = m / n ∧ 
               (m + n = 98) := 
begin
  sorry,
end

end find_mplusn_l504_504245


namespace infinitely_many_n_divisible_by_expr_l504_504908

theorem infinitely_many_n_divisible_by_expr (k : ℕ) : ∃ᶠ n in at_top, (∀ a b : ℕ, a ≤ b → (b - a + 1 = n) → ((∏ i in finset.range (n + 1), (a + i)) % ((n + k)^2 + 1) = 0)) :=
sorry

end infinitely_many_n_divisible_by_expr_l504_504908


namespace limit_f_l504_504015

open Real

noncomputable def f (x : ℝ) := (5 / 3) * x - log (2 * x + 1)

theorem limit_f (f' : ℝ → ℝ) (h : ∀ x, f' x = (5 / 3) - (2 / (2 * x + 1))) :
  Tendsto (λ Δx, (f (1 + Δx) - f 1) / Δx) (𝓝 0) (𝓝 1) :=
begin
  have f'_def : f' 1 = 1,
  { rw h, simp },
  rw ← f'_def,
  exact has_deriv_at_iff_tendsto_slope.mp (Deriv.deriv f (1 : ℝ)),
  sorry -- Proof required here
end

end limit_f_l504_504015


namespace allen_change_l504_504822

-- Define the cost per box and the number of boxes
def cost_per_box : ℕ := 7
def num_boxes : ℕ := 5

-- Define the total cost including the tip
def total_cost := num_boxes * cost_per_box
def tip := total_cost / 7
def total_paid := total_cost + tip

-- Define the amount given to the delivery person
def amount_given : ℕ := 100

-- Define the change received
def change := amount_given - total_paid

-- The statement to prove
theorem allen_change : change = 60 :=
by
  -- sorry is used here to skip the proof, as per the instruction
  sorry

end allen_change_l504_504822


namespace find_T_l504_504714

theorem find_T (T : ℝ) (h : (3/4) * (1/8) * T = (1/2) * (1/6) * 72) : T = 64 :=
by {
  -- proof goes here
  sorry
}

end find_T_l504_504714


namespace find_probability_P_l504_504592

noncomputable def game_ends_after_second_round : Prop :=
  ∃ P : ℝ, P > (1 / 2) ∧ (P^2 + (1 - P)^2 = (5 / 8))

theorem find_probability_P : 
  ∃ P : ℝ, game_ends_after_second_round ∧ P = (3 / 4) :=
begin
  use 3 / 4,
  split,
  { unfold game_ends_after_second_round,
    use 3 / 4,
    split,
    { norm_num, },
    { norm_num, }, },
  refl,
end

end find_probability_P_l504_504592


namespace book_arrangement_l504_504610

/-- Given ten different mathematics books consisting of:
- three calculus books,
- four algebra books, and
- three statistics books,
and the condition that all calculus books must be grouped together and all statistics books must be grouped together,
there are 25920 ways to arrange these books on a shelf. --/
theorem book_arrangement :
  let total_books := 10
  let calculus_books := 3
  let algebra_books := 4
  let statistics_books := 3
  (∀ (arr : Fin set (total_books)) (cals, stats, algs : Finset) (cals.size = calculus_books ∧ stats.size = statistics_books ∧ algs.size = algebra_books)
  , (Finset.disjoint cals stats ∧ Finset.disjoint cals algs ∧ Finset.disjoint stats algs)
  ∧ ∀ (s : Finset).__List.perm (arr.toList) (cals :: stats :: algs :: Nil) => 25920) :=
sorry

end book_arrangement_l504_504610


namespace repeating_decimal_to_fraction_l504_504392

noncomputable def repeating_decimal_sum (x y z : ℚ) : ℚ := x + y + z

theorem repeating_decimal_to_fraction :
  let x := 4 / 33
  let y := 34 / 999
  let z := 567 / 99999
  repeating_decimal_sum x y z = 134255 / 32929667 := by
  -- proofs are omitted
  sorry

end repeating_decimal_to_fraction_l504_504392


namespace find_T_coordinates_l504_504562

def Point : Type := ℝ × ℝ

def O : Point := (0, 0)
def Q : Point := (4, 2)
def P : Point := (4, 0)
def R : Point := (0, 2)

def is_on_x_axis (T : Point) : Prop := T.snd = 0

def area_rectangle (A B : Point) : ℝ := abs ((B.fst - A.fst) * (B.snd - A.snd))

def area_triangle (A B C : Point) : ℝ := abs ((B.fst - A.fst) * (C.snd - A.snd) - (C.fst - A.fst) * (B.snd - A.snd)) / 2

theorem find_T_coordinates : ∃ T : Point, is_on_x_axis T
  ∧ area_triangle P Q T = (area_rectangle O Q) / 2
  ∧ T = (8, 0) :=
by
  sorry

end find_T_coordinates_l504_504562


namespace area_between_chords_l504_504720

variable (circle_radius : ℝ) (chord_distance : ℝ) (chord_length : ℝ)

-- Predicate expressing the conditions
def conditions_holds : Prop :=
  circle_radius = 10 ∧ chord_distance = 10 ∧ ∃ chord_length, true

theorem area_between_chords : conditions_holds circle_radius chord_distance chord_length → 
  (area_between_chords circle_radius chord_distance chord_length) = (100 * Real.pi / 3 - 50 * Real.sqrt 3) :=
by
  sorry

end area_between_chords_l504_504720


namespace find_modulus_of_product_of_sums_l504_504843

-- Formalizing conditions in Lean 4
variables {p q r z : ℂ}

-- Assuming p, q, r form an equilateral triangle with side length 24
-- and have been translated by the same complex number z
def is_equilateral_triangle (p q r : ℂ) (side_length : ℝ) : Prop :=
  (abs (p - q) = side_length) ∧ (abs (q - r) = side_length) ∧ (abs (r - p) = side_length)

def translated_complex_numbers (p q r z : ℂ) : ℂ × ℂ × ℂ :=
  ((p + z : ℂ), (q + z : ℂ), (r + z : ℂ))

-- Assuming the modulus of the sum of translated complex numbers is 48
def sum_modulus_is (p q r z : ℂ) (modulus : ℝ) : Prop :=
  abs (p + q + r + 3 * z) = modulus

-- The main proof statement
theorem find_modulus_of_product_of_sums
  (h1 : is_equilateral_triangle p q r 24)
  (h2 : sum_modulus_is p q r z 48) :
  abs (p * q + p * r + q * r) = 768 :=
sorry

end find_modulus_of_product_of_sums_l504_504843


namespace fractional_expression_evaluation_l504_504963

theorem fractional_expression_evaluation (a : ℝ) (h : a^3 + 3 * a^2 + a = 0) :
  ∃ b : ℝ, b = 0 ∨ b = 1 ∧ b = 2022 * a^2 / (a^4 + 2015 * a^2 + 1) :=
by
  sorry

end fractional_expression_evaluation_l504_504963


namespace length_of_CB_l504_504980

-- Let ΔABC be a triangle with points D, E, F, G
variables {A B C D E F G : Type}
variables (DE_parallel_AB : ∀ {l m n : Type}, l ∥ m → m ∥ n → l ∥ n)
variables (FG_parallel_AB : ∀ {l m n : Type}, l ∥ m → m ∥ n → l ∥ n)
variables (CD_length : ℝ)
variables (DA_length : ℝ)
variables (CF_length : ℝ)
variables [linear_ordered_field ℝ]

-- Given conditions
def given_conditions := 
  (CD_length = 5) ∧ 
  (DA_length = 15) ∧ 
  (CF_length = 10)

-- Proof goal
theorem length_of_CB :
  given_conditions → (CB : ℝ) = 20 :=
begin
  sorry
end

end length_of_CB_l504_504980


namespace tap_open_duration_l504_504351

theorem tap_open_duration
  (t : ℝ)
  (h_filling_rate1 : 1/20) -- First tap fills in 20 minutes
  (h_filling_rate2 : 1/60) -- Second tap fills in 60 minutes
  (h_eq : t * (1/20 + 1/60) + 20.000000000000004 * (1/60) = 1) -- Combined equation for the filling process
  : t = 10 :=
sorry -- This skips the proof.

end tap_open_duration_l504_504351


namespace truck_catches_up_with_bus_l504_504317

noncomputable def timeToCatchUp(truck_speed bus_speed car_speed : ℕ) (initial_dist_car_truck initial_dist_truck_bus : ℕ) : ℕ :=
  if h : (car_speed - truck_speed = truck_speed - bus_speed) then
    15
  else
    sorry

theorem truck_catches_up_with_bus : ∀ (bus_speed truck_speed car_speed : ℕ)
  (initial_dist_car_truck initial_dist_truck_bus : ℕ),
  car_speed > truck_speed ∧ truck_speed > bus_speed ∧
  initial_dist_car_truck = initial_dist_truck_bus →
  timeToCatchUp truck_speed bus_speed car_speed initial_dist_car_truck initial_dist_truck_bus = 15 :=
begin
  intros,
  have h₁ : car_speed - truck_speed = truck_speed - bus_speed,
    sorry,
  rw timeToCatchUp,
  simp [h₁],
  exact sorry,
end

end truck_catches_up_with_bus_l504_504317


namespace part1_part2_1_part2_2_l504_504906

-- Definitions and conditions
def y1 (x a : ℝ) : ℝ := x - a + 2
def y2 (x k : ℝ) (hk : k ≠ 0) : ℝ := k / x
def passes_through (y : ℝ → ℝ) (pt : ℝ × ℝ) : Prop := y pt.1 = pt.2

-- Conditions provided in the problem
variable (a k : ℝ) (ha : 2 * a + k = 5) (hk : k ≠ 0)

-- Proof problem statements
theorem part1 : passes_through (y2 k hk) (k, 1) := sorry

noncomputable def y2_expression : ℝ → ℝ :=
  let a := 2
  let k := 1
  (y2)


theorem part2_1 : y2_expression = λ x, 1 / x := sorry

theorem part2_2 (x : ℝ) (hx : x > 0) : 
  if 0 < x ∧ x < 1 then y1 x a < y2 x k hk else if x = 1 then y1 x a = y2 x k hk else y1 x a > y2 x k hk := sorry

end part1_part2_1_part2_2_l504_504906


namespace bottle_display_total_l504_504995

theorem bottle_display_total (a d l n S : ℕ) 
  (h_a : a = 3)
  (h_d : d = 3)
  (h_l : l = 30)
  (h_n : l = a + (n - 1) * d)
  (h_n_10 : n = 10)
  (h_S : S = n * (a + l) / 2) :
  S = 165 :=
by
  rw [h_a, h_d, h_l, h_n, h_n_10, h_S]
  norm_num
  sorry

end bottle_display_total_l504_504995


namespace simplify_sqrt_sum_l504_504634

theorem simplify_sqrt_sum :
  sqrt (12 + 8 * sqrt 3) + sqrt (12 - 8 * sqrt 3) = 4 * sqrt 3 := 
by
  -- Proof would go here
  sorry

end simplify_sqrt_sum_l504_504634


namespace replace_asterisk_with_monomial_l504_504195

theorem replace_asterisk_with_monomial (x : ℝ) :
  (∀ asterisk : ℝ, ((x^4 - 3)^2 + (x^3 + asterisk)^2) = (x^8 + x^6 + 9x^2 + 9)) ↔ asterisk = 3x :=
by sorry

end replace_asterisk_with_monomial_l504_504195


namespace line_equation_l504_504440

theorem line_equation (θ : Real) (b : Real) (h1 : θ = 45) (h2 : b = 2) : (y = x + b) :=
by
  -- Assume θ = 45°. The corresponding slope is k = tan(θ) = 1.
  -- Since the y-intercept b = 2, the equation of the line y = mx + b = x + 2.
  sorry

end line_equation_l504_504440


namespace subsets_P_count_l504_504946

def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {2, 4, 6}
def P : Set ℕ := M ∩ N

theorem subsets_P_count : (Set.powerset P).card = 4 := by
  sorry

end subsets_P_count_l504_504946


namespace jason_total_spending_l504_504130

def cost_of_shorts : ℝ := 14.28
def cost_of_jacket : ℝ := 4.74
def total_spent : ℝ := 19.02

theorem jason_total_spending : cost_of_shorts + cost_of_jacket = total_spent :=
by
  sorry

end jason_total_spending_l504_504130


namespace master_craftsman_quota_l504_504535

theorem master_craftsman_quota (N : ℕ) (initial_rate increased_rate : ℕ) (additional_hours extra_hours : ℝ) :
  initial_rate = 35 →
  increased_rate = initial_rate + 15 →
  additional_hours = 0.5 →
  extra_hours = 1 →
  N / initial_rate - N / increased_rate = additional_hours + extra_hours →
  N = 175 →
  (initial_rate + N) = 210 :=
by {
  intros h1 h2 h3 h4 h5 h6,
  rw h6,
  exact rfl,
}

end master_craftsman_quota_l504_504535


namespace find_set_A_find_t_range_l504_504924

theorem find_set_A (a : ℝ) : 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → log 2 (x^2 - 2*x + 5) - |2*a - 1| = 0) → 
  (0 < a) → 
  (a ∈ set.Icc (3/2) 2) :=
by sorry

theorem find_t_range (t : ℝ) : 
  (∀ (a : ℝ), a ∈ set.Icc (3/2) 2 → t^2 - a*t - 3 ≥ 0) → 
  (t ≥ (real.sqrt 57 - 3)/4 ∨ t ≥ 3) :=
by sorry

end find_set_A_find_t_range_l504_504924


namespace dasha_strip_dimensions_l504_504782

theorem dasha_strip_dimensions (a b c : ℕ) (h1 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43) : 
  (a = 1 ∧ (b + c = 22)) ∨ (a = 22 ∧ (b + c = 1)) :=
by sorry

end dasha_strip_dimensions_l504_504782


namespace find_sum_of_numbers_in_sequences_l504_504266

theorem find_sum_of_numbers_in_sequences 
  (d : ℝ) (a b : ℝ) (r : ℝ)
  (h1 : a = 4 + d)
  (h2 : b = 4 + 2d)
  (h3 : b = a * r)
  (h4 : 16 = b * r) :
  a + b = 20 := 
  sorry

end find_sum_of_numbers_in_sequences_l504_504266


namespace sum_of_first_nine_terms_l504_504699

noncomputable def arithmetic_sequence_sum (a₁ d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def a_n (a₁ d n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem sum_of_first_nine_terms (a₁ d : ℕ) (h : a_n a₁ d 2 + a_n a₁ d 6 + a_n a₁ d 7 = 18) :
  arithmetic_sequence_sum a₁ d 9 = 54 :=
sorry

end sum_of_first_nine_terms_l504_504699


namespace simplify_sqrt_expression_l504_504655

theorem simplify_sqrt_expression :
  √(12 + 8 * √3) + √(12 - 8 * √3) = 4 * √3 :=
sorry

end simplify_sqrt_expression_l504_504655


namespace rectangle_and_square_problems_l504_504385

theorem rectangle_and_square_problems :
  ∃ (length width : ℝ), 
    (length / width = 2) ∧ 
    (length * width = 50) ∧ 
    (length = 10) ∧
    (width = 5) ∧
    ∃ (side_length : ℝ), 
      (side_length ^ 2 = 50) ∧ 
      (side_length - width = 5 * (Real.sqrt 2 - 1)) := 
by
  sorry

end rectangle_and_square_problems_l504_504385


namespace quadratic_bc_value_l504_504021

-- Define the context where the roots of the quadratic equation are known
variables {x b c : ℝ}

-- Define the equations using Vieta's formulas
def roots_quadratic (b c : ℝ) := 2 + 4 = -b ∧ 2 * 4 = c

-- Goal: to prove bc = -48
theorem quadratic_bc_value (b c : ℝ) (h : roots_quadratic b c) : b * c = -48 :=
by {
  -- Use provided conditions to derive b and c
  obtain ⟨hb, hc⟩ := h,
  -- Explicitly substitute these back into the desired result
  have b_value : b = -6 := by linarith,
  have c_value : c = 8 := by linarith,
  rw [b_value, c_value],
  -- Compute b * c
  norm_num,
  -- Expected bc = -48
  exact rfl
}

end quadratic_bc_value_l504_504021


namespace range_of_k_l504_504938

theorem range_of_k (k : ℝ) : (∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 - 2*k*x + k)) ↔ (k ∈ Set.Iic 0 ∨ k ∈ Set.Ici 1) :=
by
  sorry

end range_of_k_l504_504938


namespace simplify_sqrt_sum_l504_504636

theorem simplify_sqrt_sum :
  sqrt (12 + 8 * sqrt 3) + sqrt (12 - 8 * sqrt 3) = 4 * sqrt 3 := 
by
  -- Proof would go here
  sorry

end simplify_sqrt_sum_l504_504636


namespace largest_angle_right_triangle_l504_504916

theorem largest_angle_right_triangle
  (a b c : ℝ)
  (h1 : c = 5)
  (h2 : sqrt (a - 4) + (b - 3) ^ 2 = 0) :
  ∃ angle : ℝ, angle = 90 :=
by
  sorry

end largest_angle_right_triangle_l504_504916


namespace master_craftsman_total_parts_l504_504512

theorem master_craftsman_total_parts
  (N : ℕ) -- Additional parts to be produced after the first hour
  (initial_rate : ℕ := 35) -- Initial production rate (35 parts/hour)
  (increased_rate : ℕ := initial_rate + 15) -- Increased production rate (50 parts/hour)
  (time_difference : ℝ := 1.5) -- Time difference in hours between the rates
  (eq_time_diff : (N / initial_rate) - (N / increased_rate) = time_difference) -- The given time difference condition
  : 35 + N = 210 := -- Conclusion we need to prove
sorry

end master_craftsman_total_parts_l504_504512


namespace primes_between_50_and_70_l504_504075

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).filter is_prime |>.length

theorem primes_between_50_and_70 : count_primes_in_range 50 70 = 4 :=
by
  sorry

end primes_between_50_and_70_l504_504075


namespace monotonic_intervals_range_f_l504_504937

def f (x : ℝ) : ℝ := (Real.cos x)^2 + (Real.sqrt 3) * (Real.sin x) * (Real.cos x) + 1

theorem monotonic_intervals (k : ℤ) :
  ∀ x, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 →
  (∃ x₀ x₁, x₀ ≤ x₁ ∧ x₀ ∈ Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)
    ∧ x₁ ∈ Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6)
    ∧ f x₀ ≤ f x₁) := sorry

theorem range_f (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) : 
  1 ≤ f x ∧ f x ≤ 5 / 2 := sorry

end monotonic_intervals_range_f_l504_504937


namespace constant_term_expansion_l504_504221

theorem constant_term_expansion :
  let x := λ : Type, (2 * x + 1 / x)^4 = 24

end constant_term_expansion_l504_504221


namespace derivative_of_g_l504_504567

noncomputable def f (x : ℝ) : ℝ := x^3

def g (a b x : ℝ) : ℝ := f (a - b * x)

theorem derivative_of_g (a b : ℝ) :
  ∀ x : ℝ, deriv (g a b) x = -3 * b * (a - b * x)^2 :=
by
  intro x
  sorry

end derivative_of_g_l504_504567


namespace recipe_flour_amount_l504_504159

open Nat

theorem recipe_flour_amount :
  ∃ f, (∃ s, s = 8 ∧ ∃ t, t = 7 ∧ ∃ x, x = 5 ∧ s - x = 1) → (∃ y, y = 8 ∧ ∃ z, z = 7 ∧ y - z = 1) → y - z = 1 → (∃ f1 : ℕ, sorry) :=
begin
  sorry
end

end recipe_flour_amount_l504_504159


namespace find_a_l504_504014

open Real

def f (a x : ℝ) : ℝ := log x + a * x

theorem find_a (a : ℝ) : (∃ x, f a x = 0 ∧ ∀ y, f a y ≤ f a x) → a = -1 / exp 1 :=
by
  sorry

end find_a_l504_504014


namespace fraction_white_surface_area_l504_504796

theorem fraction_white_surface_area : 
  let total_surface_area := 96
  let black_faces_corners := 6
  let black_faces_centers := 6
  let black_faces_total := 12
  let white_faces_total := total_surface_area - black_faces_total
  white_faces_total / total_surface_area = 7 / 8 :=
by
  sorry

end fraction_white_surface_area_l504_504796


namespace parts_manufactured_l504_504540

variable (initial_parts : ℕ) (initial_rate : ℕ) (increased_speed : ℕ) (time_diff : ℝ)
variable (N : ℕ)

-- initial conditions
def initial_parts := 35
def initial_rate := 35
def increased_speed := 15
def time_diff := 1.5

-- additional parts to be manufactured
noncomputable def additional_parts := N

-- equation representing the time differences
noncomputable def equation := (N / initial_rate) - (N / (initial_rate + increased_speed)) = time_diff

-- state the proof problem
theorem parts_manufactured : initial_parts + additional_parts = 210 :=
by
  -- Use the given conditions to solve the problem
  sorry

end parts_manufactured_l504_504540


namespace smallest_d_l504_504403

noncomputable def smallestPositiveD : ℝ := 1

theorem smallest_d (d : ℝ) : 
  (0 < d) →
  (∀ x y : ℝ, 0 ≤ x → 0 ≤ y → 
    (Real.sqrt (x * y) + d * (x^2 - y^2)^2 ≥ x + y)) →
  d ≥ smallestPositiveD :=
by
  intros h1 h2
  sorry

end smallest_d_l504_504403


namespace code_cracked_probability_l504_504162

theorem code_cracked_probability:
  (pa pb : ℚ) (hpa : pa = 1 / 5) (hpb : pb = 1 / 4) :
  1 - (1 - pa) * (1 - pb) = 2 / 5 :=
by
  sorry

end code_cracked_probability_l504_504162


namespace integral_points_sum_l504_504117

noncomputable def gcd (a b : ℕ) : ℕ := Nat.gcd a b
noncomputable def f (n : ℕ) : ℕ :=
  if gcd n (n + 3) = 3 then 2 else 0

theorem integral_points_sum :
  (Finset.range 1990).sum (f ∘ (λ n, n + 1)) = 1326 := by
  -- The detailed proof can be constructed here.
  sorry

end integral_points_sum_l504_504117


namespace father_l504_504798

theorem father's_age (S F : ℕ) (h1 : S - 5 = 17) (h2 : F - S = S) : F = 44 :=
by
  have h3 : S = 22 := by linarith
  have h4 : F - 22 = 22 := by linarith
  linarith

end father_l504_504798


namespace replace_with_monomial_produces_four_terms_l504_504172

-- Define the initial expression
def initialExpression (k : ℤ) (x : ℤ) : ℤ :=
  ((x^4 - 3)^2 + (x^3 + k)^2)

-- Proof statement
theorem replace_with_monomial_produces_four_terms (x : ℤ) :
  ∃ (k : ℤ), initialExpression k x = (x^8 + x^6 + 9x^2 + 9) :=
  exists.intro (3 * x) sorry

end replace_with_monomial_produces_four_terms_l504_504172


namespace competition_math_problem_l504_504989

theorem competition_math_problem :
  let scores := [60, 75, 85, 95]
  let percentages := [0.15, 0.25, 0.40, 0.20]
  let mean := (0.15 * 60 + 0.25 * 75 + 0.40 * 85 + 0.20 * 95)
  let median := 85
  in abs (median - mean) = 4 :=
by
  sorry

end competition_math_problem_l504_504989


namespace product_bc_l504_504264

theorem product_bc (b c : ℤ)
    (h1 : ∀ s : ℤ, s^2 = 2 * s + 1 → s^6 - b * s - c = 0) :
    b * c = 2030 :=
sorry

end product_bc_l504_504264


namespace floor_arithmetic_sum_l504_504844

   variable (a : ℝ)
   variable (d : ℝ)
   variable (n : ℕ)

   noncomputable def floor_sum (a d : ℝ) (n : ℕ) : ℝ :=
     (Finset.range n).sum (λ i => Real.floor (a + d * i))

   theorem floor_arithmetic_sum :
     let a := 2
     let d := 0.8
     let n := 124
     floor_sum a d n + 2 = 6300 :=
   by
     sorry
   
end floor_arithmetic_sum_l504_504844


namespace simplify_sqrt_expression_l504_504652

theorem simplify_sqrt_expression :
  √(12 + 8 * √3) + √(12 - 8 * √3) = 4 * √3 :=
sorry

end simplify_sqrt_expression_l504_504652


namespace primes_between_50_and_70_l504_504080

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).filter is_prime |>.length

theorem primes_between_50_and_70 : count_primes_in_range 50 70 = 4 :=
by
  sorry

end primes_between_50_and_70_l504_504080


namespace domain_of_f_l504_504121

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x + 1) + (x-2)^0

theorem domain_of_f : ∀ x : ℝ, (x ≥ -1 ∧ x ≠ 2) ↔ ∀ y : ℝ, y = f x → True :=
by 
  sorry

end domain_of_f_l504_504121


namespace f_3_minus_f_4_l504_504965

def f : ℝ → ℝ := sorry  -- we assume the existence of such a function

-- Define f being odd and having a period of 5
axiom odd_function (x : ℝ) : f(-x) = -f(x)
axiom periodic_function (x : ℝ) : f(x + 5) = f(x)

-- Given conditions
axiom f_at_1 : f(1) = 1
axiom f_at_2 : f(2) = 2

theorem f_3_minus_f_4 : f(3) - f(4) = -1 := by
  sorry

end f_3_minus_f_4_l504_504965


namespace count_primes_between_fifty_and_seventy_l504_504058

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_fifty_and_seventy : List ℕ :=
  [53, 59, 61, 67]

theorem count_primes_between_fifty_and_seventy :
  (primes_between_fifty_and_seventy.count is_prime = 4) :=
by
  sorry

end count_primes_between_fifty_and_seventy_l504_504058


namespace ineq_condition_l504_504308

theorem ineq_condition (a b : ℝ) : (a + 1 > b - 2) ↔ (a > b - 3 ∧ ¬(a > b)) :=
by
  sorry

end ineq_condition_l504_504308


namespace part_a_part_b_l504_504781

namespace FlyingSaucerProof

-- Consider the setup for the lunar orbit
def Earth : Type := ℝ
def Moon : Type := ℝ
def LunarOrbit := set (euclidean_space ℝ 2)
variable (orbit : LunarOrbit)

-- Definition for the midpoint and jumping conditions
def midpoint (A A' : euclidean_space ℝ 2) (B : Earth) : Prop :=
  (A + A') / 2 = B

def can_jump (A A' : euclidean_space ℝ 2) : Prop :=
  ∃ M : Moon, midpoint A A' M ∨ midpoint A A' 0 -- assuming Earth centered at origin

-- Problem Setup
variable (P Q : euclidean_space ℝ 2)

-- Part (a)
theorem part_a : ∃ A1 A2, can_jump P A1 ∧ can_jump A1 A2 ∧ A2 = Q :=
  sorry

-- Part (b)
theorem part_b : ∀ ε > 0, ∃ A₀ ... Aₙ, n : ℕ, can_jump_sequence : Π (i : fin n), can_jump Aᵢ Aᵢ₊₁, A₀ = P, Aₙ = Q, 
  (Aₙ - A₀).norm < ε :=
  sorry

end FlyingSaucerProof

end part_a_part_b_l504_504781


namespace quadratic_factors_even_b_l504_504693

theorem quadratic_factors_even_b (b : ℤ) : 
  (∃ (d e f g : ℤ), 
    15 = d * f ∧ 
    45 = e * g ∧ 
    b = d * g + e * f
    ∧ (∀ (d : ℕ) e f g, Nat.prime d → Nat.prime e → Nat.prime f → Nat.prime g → 
      15 = d * f ∧ 45 = e * g ∧ b = d * g + e * f)) → 
    Even b := 
begin
  sorry
end

end quadratic_factors_even_b_l504_504693


namespace jony_stop_block_correct_l504_504133

-- Jony's walk parameters
def start_time : ℕ := 7 -- In hours, but it is not used directly
def start_block : ℕ := 10
def end_block : ℕ := 90
def stop_time : ℕ := 40 -- Jony stops walking after 40 minutes starting from 07:00
def speed : ℕ := 100 -- meters per minute
def block_length : ℕ := 40 -- meters

-- Function to calculate the stop block given the parameters
def stop_block (start_block end_block stop_time speed block_length : ℕ) : ℕ :=
  let total_distance := stop_time * speed
  let outbound_distance := (end_block - start_block) * block_length
  let remaining_distance := total_distance - outbound_distance
  let blocks_walked_back := remaining_distance / block_length
  end_block - blocks_walked_back

-- The statement to prove
theorem jony_stop_block_correct :
  stop_block start_block end_block stop_time speed block_length = 70 :=
by
  sorry

end jony_stop_block_correct_l504_504133


namespace median_of_donatedAmounts_is_8_l504_504984

def donatedAmounts : List ℕ := [8, 10, 10, 4, 6]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get! (sorted.length / 2)

theorem median_of_donatedAmounts_is_8 : median donatedAmounts = 8 := by
  sorry

end median_of_donatedAmounts_is_8_l504_504984


namespace right_triangle_third_side_product_l504_504731

theorem right_triangle_third_side_product :
  let a := 6
  let b := 8
  let c1 := Real.sqrt (a^2 + b^2)     -- Hypotenuse when a and b are legs
  let c2 := Real.sqrt (b^2 - a^2)     -- Other side when b is the hypotenuse
  20 * Real.sqrt 7 ≈ 52.7 := 
by
  sorry

end right_triangle_third_side_product_l504_504731


namespace polynomial_solution_l504_504877

noncomputable def q (x : ℝ) : ℝ := 2 * Real.sqrt 3 * x^4

theorem polynomial_solution (x : ℝ) : q(x^4) - q(x^4 - 3) = (q x)^3 + 18 :=
by
  sorry

end polynomial_solution_l504_504877


namespace video_down_votes_l504_504818

theorem video_down_votes 
  (up_votes : ℕ)
  (ratio_up_down : up_votes / 1394 = 45 / 17)
  (up_votes_known : up_votes = 3690) : 
  3690 / 1394 = 45 / 17 :=
by
  sorry

end video_down_votes_l504_504818


namespace valid_arrangement_count_l504_504608

-- Definitions derived from conditions:
-- Competition types and gyms are represented as finite sets
def CompetitionType := { Volleyball, Basketball, TableTennis }
def Gym := { Gym1, Gym2, Gym3, Gym4 }

-- Define a function that models the number of valid arrangements:
def validArrangements : Nat :=
  let scenario1 := 4 * 3 * 2
  let scenario2 := (3.choose 2) * (4 * 3)
  scenario1 + scenario2

-- Theorem stating the total number of valid arrangements
theorem valid_arrangement_count : validArrangements = 60 := by
  sorry

end valid_arrangement_count_l504_504608


namespace trigonometric_identity_l504_504780

-- Definitions for the conditions
def f1 (α : ℝ) := cos (5 / 2 * π - 6 * α) * (sin (π - 2 * α))^3 - cos (6 * α - π) * (sin (π / 2 - 2 * α))^3
def f2 (α : ℝ) := (cos (4 * α))^3

-- Theorem to be proved
theorem trigonometric_identity (α : ℝ) : f1(α) = f2(α) := by sorry -- proof goes here

end trigonometric_identity_l504_504780


namespace count_primes_between_fifty_and_seventy_l504_504062

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_fifty_and_seventy : List ℕ :=
  [53, 59, 61, 67]

theorem count_primes_between_fifty_and_seventy :
  (primes_between_fifty_and_seventy.count is_prime = 4) :=
by
  sorry

end count_primes_between_fifty_and_seventy_l504_504062


namespace isosceles_triangle_circumscribed_radius_and_height_l504_504241

/-
Conditions:
- The isosceles triangle has two equal sides of 20 inches.
- The base of the triangle is 24 inches.

Prove:
1. The radius of the circumscribed circle is 5 inches.
2. The height of the triangle is 16 inches.
-/

theorem isosceles_triangle_circumscribed_radius_and_height 
  (h_eq_sides : ∀ A B C : Type, ∀ (AB AC : ℝ), ∀ (BC : ℝ), AB = 20 → AC = 20 → BC = 24) 
  (R : ℝ) (h : ℝ) : 
  R = 5 ∧ h = 16 := 
sorry

end isosceles_triangle_circumscribed_radius_and_height_l504_504241


namespace decreasing_geometric_sums_implications_l504_504122

variable (X : Type)
variable (a1 q : ℝ)
variable (S : ℕ → ℝ)

def is_geometric_sequence (a : ℕ → ℝ) :=
∀ n : ℕ, a (n + 1) = a1 * q^n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) :=
S 0 = a 0 ∧ ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

def is_decreasing_sequence (S : ℕ → ℝ) :=
∀ n : ℕ, S (n + 1) < S n

theorem decreasing_geometric_sums_implications (a1 q : ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, S (n + 1) < S n) → a1 < 0 ∧ q > 0 := 
by 
  sorry

end decreasing_geometric_sums_implications_l504_504122


namespace tournament_committees_l504_504495

-- Assuming each team has 7 members
def team_members : Nat := 7

-- There are 5 teams
def total_teams : Nat := 5

-- The host team selects 3 members including at least one woman
def select_host_team_members (w m : Nat) : ℕ :=
  let total_combinations := Nat.choose team_members 3
  let all_men_combinations := Nat.choose (team_members - 1) 3
  total_combinations - all_men_combinations

-- Each non-host team selects 2 members including at least one woman
def select_non_host_team_members (w m : Nat) : ℕ :=
  let total_combinations := Nat.choose team_members 2
  let all_men_combinations := Nat.choose (team_members - 1) 2
  total_combinations - all_men_combinations

-- Total number of committees when one team is the host
def one_team_host_total_combinations (w m : Nat) : ℕ :=
  select_host_team_members w m * (select_non_host_team_members w m) ^ (total_teams - 1)

-- Total number of possible 11-member tournament committees
def total_committees (w m : Nat) : ℕ :=
  one_team_host_total_combinations w m * total_teams

theorem tournament_committees (w m : Nat) (hw : w ≥ 1) (hm : m ≤ 6) :
  total_committees w m = 97200 :=
by
  sorry

end tournament_committees_l504_504495


namespace arithmetic_mean_not_prime_l504_504601

open Nat

def is_prime (p : ℕ) : Prop := 2 ≤ p ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def primes (n : ℕ) : ℕ := if n = 1 then 2 else if n = 2 then 3 else sorry

theorem arithmetic_mean_not_prime (n : ℕ) (hn : 2 ≤ n) :
  ¬ is_prime ((∑ i in Finset.range n, primes (i + 1)) / n) :=
sorry

end arithmetic_mean_not_prime_l504_504601


namespace replace_asterisk_with_monomial_l504_504194

theorem replace_asterisk_with_monomial :
  ∀ (x : ℝ), (x^4 - 3)^2 + (x^3 + 3x)^2 = x^8 + x^6 + 9x^2 + 9 := 
by
  intro x
  calc
    (x^4 - 3)^2 + (x^3 + 3x)^2
        = (x^4)^2 - 2 * x^4 * 3 + 3^2 + (x^3)^2 + 2 * x^3 * 3x + (3x)^2 : by ring
    ... = x^8 - 6 * x^4 + 9 + x^6 + 6 * x^4 + 9 * x^2 : by ring
    ... = x^8 + x^6 + 9 * x^2 + 9 : by ring
  sorry

end replace_asterisk_with_monomial_l504_504194


namespace quadratic_has_real_roots_find_specific_k_l504_504942

-- Part 1: Prove the range of values for k
theorem quadratic_has_real_roots (k : ℝ) : (k ≥ 2) ↔ ∃ x1 x2 : ℝ, x1 ^ 2 - 4 * x1 - 2 * k + 8 = 0 ∧ x2 ^ 2 - 4 * x2 - 2 * k + 8 = 0 := 
sorry

-- Part 2: Prove the specific value of k given the additional condition
theorem find_specific_k (k : ℝ) (x1 x2 : ℝ) : (x1 ^ 3 * x2 + x1 * x2 ^ 3 = 24) ∧ x1 ^ 2 - 4 * x1 - 2 * k + 8 = 0 ∧ x2 ^ 2 - 4 * x2 - 2 * k + 8 = 0 → k = 3 :=
sorry

end quadratic_has_real_roots_find_specific_k_l504_504942


namespace _l504_504003

variables {α x α_minus_pi_over4 : ℝ}

def f (x α : ℝ) := sin (x + α) + sin (α - x) - 2 * sin α

lemma sin_alpha_minus_pi_over4 :
  (tan (2 * α) = 3 / 4) →
  (α ∈ Ioo (- (π / 2)) (π / 2)) →
  (∀ x : ℝ, f x α ≥ 0) →
  (sin (α - π / 4) = - 2 * real.sqrt 5 / 5) :=
by
  -- We only state the theorem without proving it.
  intros,
  sorry

end _l504_504003


namespace arithmetic_mean_of_numbers_l504_504852

theorem arithmetic_mean_of_numbers (n : ℕ) (h : n > 1) :
  let one_special_number := (1 / n) + (2 / n ^ 2)
  let other_numbers := (n - 1) * 1
  (other_numbers + one_special_number) / n = 1 + 2 / n ^ 2 :=
by
  sorry

end arithmetic_mean_of_numbers_l504_504852


namespace sum_tan_square_eq_sqrt_two_l504_504563

open Real

theorem sum_tan_square_eq_sqrt_two :
  let S := {x : ℝ | 0 < x ∧ x < π / 2 ∧
                (∃ a b c : ℝ, {sin x, cos x, tan x} = {a, b, c} ∧ a^2 + b^2 = c^2)} in
  ∑ x in S, tan x * tan x = sqrt 2 :=
by sorry

end sum_tan_square_eq_sqrt_two_l504_504563


namespace lcm_3_4_6_15_l504_504275

noncomputable def lcm_is_60 : ℕ := 60

theorem lcm_3_4_6_15 : lcm (lcm (lcm 3 4) 6) 15 = lcm_is_60 := 
by 
    sorry

end lcm_3_4_6_15_l504_504275


namespace equation_of_parabola_l504_504115

noncomputable def parabola_symmetric_x_axis_vertex_origin_passing_point (p : ℝ) : Prop :=
  ∃ (a : ℝ), a ≠ 0 ∧ p = (λ (x y : ℝ), y^2 = 2 * a * x)

theorem equation_of_parabola (a : ℝ) : 
  (parabola_symmetric_x_axis_vertex_origin_passing_point (2, 4)) →
  a = 4 → 
  ∃ (x y : ℝ), (x, y) = (2, 4) →
  y^2 = 8 * x :=
by
  sorry

end equation_of_parabola_l504_504115


namespace equal_roots_quadratic_l504_504971

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

/--
If the quadratic equation 2x^2 - ax + 2 = 0 has two equal real roots,
then the value of a is ±4.
-/
theorem equal_roots_quadratic (a : ℝ) (h : quadratic_discriminant 2 (-a) 2 = 0) :
  a = 4 ∨ a = -4 :=
sorry

end equal_roots_quadratic_l504_504971


namespace farmer_land_l504_504808

theorem farmer_land (initial_land remaining_land : ℚ) (h1 : initial_land - initial_land / 10 = remaining_land) (h2 : remaining_land = 10) : initial_land = 100 / 9 := by
  sorry

end farmer_land_l504_504808


namespace calc_series_l504_504377

noncomputable def y_seq : ℕ → ℝ
| 1     := 50
| (n+1) := y_seq n^2 - 2*y_seq n

def sum_term (k : ℕ) : ℝ := 1 / (y_seq k + 2)

theorem calc_series : (∑' k, sum_term k) = 1 / 50 :=
by
  sorry

end calc_series_l504_504377


namespace div_complex_lands_in_fourth_quadrant_l504_504442

open Complex

def z1 : ℂ := 2 + I
def z2 : ℂ := 1 + I

theorem div_complex_lands_in_fourth_quadrant :
  let z := z1 / z2 in
  z.re > 0 ∧ z.im < 0 := by
  sorry

end div_complex_lands_in_fourth_quadrant_l504_504442


namespace general_term_formula_sum_first_n_terms_exists_m_l504_504669

-- Given conditions
variable (a_n : ℕ → ℚ)
variable (b_n T_n : ℕ → ℚ)
hypothesis (a_seq_arith : ∀ n, a_n n = 2 * n - 1)
hypothesis (a3 : a_n 3 = 5)
hypothesis (geo_seq : a_n 1 * a_n 5 = (a_n 2) ^ 2)
hypothesis (bn_def : ∀ n, b_n n = 1 / ((a_n n + 1) * (a_n (n + 1) + 1)))
hypothesis (Tn_def : ∀ n, T_n n = ∑ k in finset.range n, b_n k)

-- Proof targets
theorem general_term_formula :
  ∀ n, a_n n = 2 * n - 1 := sorry

theorem sum_first_n_terms :
  ∀ n, T_n n = n / (4 * (n + 1)) := sorry

theorem exists_m :
  ∃ m : ℕ, m = 2 ∧ ∀ n, ∀ hn: n > 0, (m - 2) / 4 < T_n n ∧ T_n n < m / 5 := sorry

end general_term_formula_sum_first_n_terms_exists_m_l504_504669


namespace angle_CBA_max_area_l504_504718

open Real EuclideanGeometry

noncomputable def triangle_ABC : Triangle := sorry -- Define triangle ABC

-- Given conditions
def angle_BAC : Real := 60
def angle_CBA (B : Real) : Prop := B ≤ 90
def BC_length : Real := 1
def AC_geq_AB : Prop := sorry -- Define condition AC >= AB

-- Centers of triangle
def H := orthocenter triangle_ABC
def I := incenter triangle_ABC
def O := circumcenter triangle_ABC

-- Area maximization condition for pentagon BCOIH
def max_area_pentagon_BCOIH (B : Real) : Prop := sorry -- Define that the area of pentagon BCOIH is maximized

-- Main theorem
theorem angle_CBA_max_area : ∃ (B : Real), (angle_BAC = 60 ∧ angle_CBA B ∧ BC_length = 1 ∧ AC_geq_AB ∧ max_area_pentagon_BCOIH B) -> B = 80 :=
by 
  sorry

end angle_CBA_max_area_l504_504718


namespace division_remainder_l504_504111

-- Define the conditions
def dividend : ℝ := 9087.42
def divisor : ℝ := 417.35
def quotient : ℝ := 21

-- Define the expected remainder
def expected_remainder : ℝ := 323.07

-- Statement of the problem
theorem division_remainder : dividend - divisor * quotient = expected_remainder :=
by
  sorry

end division_remainder_l504_504111


namespace find_x_l504_504092

theorem find_x (x : ℝ) (h₀ : 0 < x) (h₁ : real.sqrt(12 * x) * real.sqrt(20 * x) * real.sqrt(5 * x) * real.sqrt(30 * x) = 30) :
  x = 1 / real.sqrt(20) :=
sorry

end find_x_l504_504092


namespace product_of_possible_lengths_approx_l504_504738

noncomputable def hypotenuse (a b : ℝ) : ℝ :=
  real.sqrt (a * a + b * b)

noncomputable def other_leg (hypotenuse a : ℝ) : ℝ :=
  real.sqrt (hypotenuse * hypotenuse - a * a)

noncomputable def product_of_possible_lengths (a b : ℝ) : ℝ :=
  hypotenuse a b * other_leg (max a b) (min a b)

theorem product_of_possible_lengths_approx (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  Float.round (product_of_possible_lengths a b) 1 = 52.7 :=
by
  sorry

end product_of_possible_lengths_approx_l504_504738


namespace inverse_fraction_coefficient_ratio_l504_504850

theorem inverse_fraction_coefficient_ratio (g : ℝ → ℝ) (h : ∀ x, g x = (3 * x - 2) / (x - 4)) :
  let g_inv := λ y, (4 * y - 2) / (y - 3) in ∃ g_inv, ∀ x, x ≠ 4 → g (g_inv x) = x ∧ ∃ (a b c d : ℝ), g_inv = λ y, (a * y + b) / (c * y + d) ∧ a = 4 ∧ c = 1 := by
  sorry

end inverse_fraction_coefficient_ratio_l504_504850


namespace tangent_line_equation_l504_504229

theorem tangent_line_equation
  (x y : ℝ)
  (h1 : (x - 2)^2 + y^2 = 4)
  (P : x = 1 ∧ y = real.sqrt 3) :
  x - real.sqrt 3 * y + 2 = 0 :=
sorry

end tangent_line_equation_l504_504229


namespace prime_count_between_50_and_70_l504_504067

open Nat

theorem prime_count_between_50_and_70 : 
  (finset.filter Nat.prime (finset.range 71 \ finset.range 51).card = 4) := 
begin
  sorry
end

end prime_count_between_50_and_70_l504_504067


namespace probability_same_color_probability_different_color_and_odd_l504_504263

open Finset

def balls := {1, 2, 3, 4, 5}
def red_balls := {1, 2, 3}
def white_balls := {4, 5}
def events := balls.powerset.filter (λ s, s.card = 2)

-- Defining event A: Draw two balls of the same color
def event_A := ({1, 2}, {1, 3}, {2, 3}, {4, 5} : Finset (Finset ℕ))

-- Defining event B: Draw two balls of different colors, and at least one has an odd number
def event_B := ({1, 4}, {1, 5}, {2, 5}, {3, 4}, {3, 5} : Finset (Finset ℕ))

-- Calculate the probability of the events
noncomputable def probability {α} (e : Finset α) (Ω : Finset α) : ℚ :=
  (e.card : ℚ) / Ω.card

theorem probability_same_color : 
  probability event_A events = 2 / 5 := by
  sorry

theorem probability_different_color_and_odd : 
  probability event_B events = 1 / 2 := by 
  sorry

end probability_same_color_probability_different_color_and_odd_l504_504263


namespace function_properties_l504_504289

noncomputable def f (x : ℝ) : ℝ := x^2

theorem function_properties :
  (∀ x1 x2 : ℝ, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x : ℝ, 0 < x → deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end function_properties_l504_504289


namespace replace_asterisk_with_monomial_l504_504197

theorem replace_asterisk_with_monomial (x : ℝ) :
  (∀ asterisk : ℝ, ((x^4 - 3)^2 + (x^3 + asterisk)^2) = (x^8 + x^6 + 9x^2 + 9)) ↔ asterisk = 3x :=
by sorry

end replace_asterisk_with_monomial_l504_504197


namespace mutually_exclusive_not_complementary_l504_504706

-- Definitions for the conditions
def bag : List (ℕ × String) := [(3, "red"), (2, "white"), (1, "black")]

def at_least_one_white_ball (balls : List (String × String)) : Prop :=
  ∃ b, b.fst = "white" ∨ b.snd = "white"

def one_red_one_black (balls : List (String × String)) : Prop :=
  (balls.head.fst = "red" ∧ balls.head.snd = "black") ∨
  (balls.head.fst = "black" ∧ balls.head.snd = "red")

-- Problem statement to prove the mutually exclusive but not complementary events
theorem mutually_exclusive_not_complementary :
  ∀ (balls : List (String × String)),
  (at_least_one_white_ball balls ∧ one_red_one_black balls → False) ∧
  ¬((¬ at_least_one_white_ball balls) ↔ one_red_one_black balls) :=
by
  sorry

end mutually_exclusive_not_complementary_l504_504706


namespace visual_range_increase_l504_504322

def percent_increase (original new : ℕ) : ℕ :=
  ((new - original) * 100) / original

theorem visual_range_increase :
  percent_increase 50 150 = 200 := 
by
  -- the proof would go here
  sorry

end visual_range_increase_l504_504322


namespace simplify_sqrt_expression_is_correct_l504_504624

-- Definition for the given problem
def simplify_sqrt_expression (a b : ℝ) :=
  a = Real.sqrt (12 + 8 * Real.sqrt 3) → 
  b = Real.sqrt (12 - 8 * Real.sqrt 3) → 
  a + b = 4 * Real.sqrt 3

-- The theorem to be proven
theorem simplify_sqrt_expression_is_correct : simplify_sqrt_expression :=
begin
  intros a b ha hb,
  rw [ha, hb],
  -- Step-by-step simplification approach would occur here
  sorry
end

end simplify_sqrt_expression_is_correct_l504_504624


namespace Jake_peaches_l504_504549

variables (Jake Steven Jill : ℕ)

def peaches_relation : Prop :=
  (Jake = Steven - 6) ∧
  (Steven = Jill + 18) ∧
  (Jill = 5)

theorem Jake_peaches : peaches_relation Jake Steven Jill → Jake = 17 := by
  sorry

end Jake_peaches_l504_504549


namespace fraction_equality_l504_504572

theorem fraction_equality
  (a b : ℝ)
  (x : ℝ)
  (h1 : x = (a^2) / (b^2))
  (h2 : a ≠ b)
  (h3 : b ≠ 0) :
  (a^2 + b^2) / (a^2 - b^2) = (x + 1) / (x - 1) :=
by
  sorry

end fraction_equality_l504_504572


namespace total_coins_in_30_layers_l504_504994

theorem total_coins_in_30_layers : 
  let T (n : ℕ) := n * (n + 1) / 2 in T 30 = 465 :=
by
  let T := λ n : ℕ, n * (n + 1) / 2
  show T 30 = 465
  sorry

end total_coins_in_30_layers_l504_504994


namespace series_convergence_implies_ratio_series_convergence_l504_504623

open Classical -- for noncomputable theory

noncomputable section
-- Define convergence of a series
def series_converges (a : ℕ → ℝ) : Prop :=
  ∃ l : ℝ, Filter.Tendsto (fun n => ∑ i in Finset.range n, a i) Filter.atTop (Filter.Tendsto.lim (fun (l : ℝ) => Filter.atTop.eventually (fun n => ∑ i in Finset.range n, a i) ≤ l))

-- Main theorem statement
theorem series_convergence_implies_ratio_series_convergence
  (a : ℕ → ℝ) :
  series_converges a →
  series_converges (fun n => a n / n) :=
sorry

end series_convergence_implies_ratio_series_convergence_l504_504623


namespace loaned_books_count_l504_504295

theorem loaned_books_count (X Y : ℝ) (Z : ℕ)
  (h1 : X = 0.35 * Y)
  (h2 : Z = 300 - 244):
  Y = 160 := by
  have h3 : X = 56 := by
    rw [h2]
  rw [h3] at h1
  sorry

end loaned_books_count_l504_504295


namespace unique_positions_of_triangle_l504_504345

def rotation_angle (k : Nat) : Real :=
  3^k

def num_unique_positions : Nat :=
  4

theorem unique_positions_of_triangle :
  ∃ N : Nat, (∀ k : Nat, rotation_angle k ≡ rotation_angle (k + N) [MOD 120]) ∧ N = num_unique_positions := by
  sorry

end unique_positions_of_triangle_l504_504345


namespace product_of_possible_lengths_approx_l504_504740

noncomputable def hypotenuse (a b : ℝ) : ℝ :=
  real.sqrt (a * a + b * b)

noncomputable def other_leg (hypotenuse a : ℝ) : ℝ :=
  real.sqrt (hypotenuse * hypotenuse - a * a)

noncomputable def product_of_possible_lengths (a b : ℝ) : ℝ :=
  hypotenuse a b * other_leg (max a b) (min a b)

theorem product_of_possible_lengths_approx (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  Float.round (product_of_possible_lengths a b) 1 = 52.7 :=
by
  sorry

end product_of_possible_lengths_approx_l504_504740


namespace distance_to_city_l504_504356

variable (d : ℝ)  -- Define d as a real number

theorem distance_to_city (h1 : ¬ (d ≥ 13)) (h2 : ¬ (d ≤ 10)) :
  10 < d ∧ d < 13 :=
by
  -- Here we will formalize the proof in Lean syntax
  sorry

end distance_to_city_l504_504356


namespace sum_i_powers_l504_504659

theorem sum_i_powers : (∑ k in Finset.range 2014, (complex.I) ^ k) = 1 + complex.I := 
by sorry

end sum_i_powers_l504_504659


namespace altitude_less_than_half_hypotenuse_l504_504613

theorem altitude_less_than_half_hypotenuse 
  {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (ABC : Triangle A B C) (right_angle_C : ∠ABC = 90°)
  (H : PointOnSegment C (LineSegment A B)) :
  let H := AltitudeFromVertex ABC C in
  let M := Midpoint (Segment A B) in
  let CM := MedianToHypotenuse ABC M in
  AltitudeFromVertex ABC C < (1/2) * Hypotenuse ABC :=
  sorry

end altitude_less_than_half_hypotenuse_l504_504613


namespace constant_term_expansion_2x_1_over_x_4_eq_24_l504_504218

theorem constant_term_expansion_2x_1_over_x_4_eq_24 : 
  (let a := 2; let b := (1 : ℝ); let n := 4 in
  ∑ r in finset.range (n + 1), binomial n r * (a * a) ^ (n - r) * (b / a) ^ r * x ^ (n - 2 * r)) = 24 :=
begin
  sorry
end

end constant_term_expansion_2x_1_over_x_4_eq_24_l504_504218


namespace smallest_sum_twice_perfect_square_l504_504254

-- Definitions based directly on conditions:
def sum_of_20_consecutive_integers (n : ℕ) : ℕ := (2 * n + 19) * 10

def twice_perfect_square (x : ℕ) : Prop := ∃ m : ℕ, x = 2 * m^2

-- Proposition to prove the smallest possible value satisfying these conditions:
theorem smallest_sum_twice_perfect_square : 
  ∃ n S, S = sum_of_20_consecutive_integers n ∧ twice_perfect_square S ∧ S = 450 :=
begin
  sorry
end

end smallest_sum_twice_perfect_square_l504_504254


namespace last_digit_factorial_difference_l504_504407

theorem last_digit_factorial_difference : (2014! - 3!) % 10 = 4 := by
  sorry

end last_digit_factorial_difference_l504_504407


namespace negation_of_p_l504_504020

noncomputable def p : Prop := ∀ x : ℝ, x > 0 → 2 * x^2 + 1 > 0

theorem negation_of_p : (∃ x : ℝ, x > 0 ∧ 2 * x^2 + 1 ≤ 0) ↔ ¬p :=
by
  sorry

end negation_of_p_l504_504020


namespace point_outside_circle_l504_504482

theorem point_outside_circle (a b : ℝ) (h : ∃ (x y : ℝ), (a * x + b * y = 1) ∧ (x^2 + y^2 = 1)) : a^2 + b^2 > 1 :=
by
  sorry

end point_outside_circle_l504_504482


namespace tv_weight_difference_l504_504367

-- Definitions for the given conditions
def bill_tv_length : ℕ := 48
def bill_tv_width : ℕ := 100
def bob_tv_length : ℕ := 70
def bob_tv_width : ℕ := 60
def weight_per_square_inch : ℕ := 4
def ounces_per_pound : ℕ := 16

-- The statement to prove
theorem tv_weight_difference : (bill_tv_length * bill_tv_width * weight_per_square_inch)
                               - (bob_tv_length * bob_tv_width * weight_per_square_inch)
                               = 150 * ounces_per_pound := by
  sorry

end tv_weight_difference_l504_504367


namespace length_of_other_train_l504_504746

-- Define the lengths and speeds of the trains
def speed_train_1 := 90 -- km/hr
def speed_train_2 := 90 -- km/hr
def length_train_1 := 0.9 -- km

-- Define the crossing time
def crossing_time := 40 -- seconds

-- Define the conversion factor from hours to seconds
def hr_to_s := 3600 -- seconds in one hour

-- Define the relative speed calculation in km/hr
def relative_speed_km_hr : Real := speed_train_1 + speed_train_2

-- Convert relative speed to km/s
def relative_speed_km_s : Real := relative_speed_km_hr / hr_to_s

-- Total distance covered during the crossing
def total_distance : Real := relative_speed_km_s * crossing_time

-- Prove the length of the other train
theorem length_of_other_train : 
  (total_distance - length_train_1 = 1.1) := by
  sorry

end length_of_other_train_l504_504746


namespace simplify_sqrt_expression_l504_504657

theorem simplify_sqrt_expression :
  √(12 + 8 * √3) + √(12 - 8 * √3) = 4 * √3 :=
sorry

end simplify_sqrt_expression_l504_504657


namespace angle_B_is_arcsin_l504_504546

-- Define the triangle and its conditions
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  ∀ (A B C : ℝ), 
    a = 8 ∧ b = Real.sqrt 3 ∧ 
    (2 * Real.cos (A - B) / 2 ^ 2 * Real.cos B - Real.sin (A - B) * Real.sin B + Real.cos (A + C) = -3 / 5)

-- Prove that the measure of ∠B is arcsin(√3 / 10)
theorem angle_B_is_arcsin (A B C : ℝ) (a b c : ℝ) (h : triangle_ABC A B C a b c) : 
  B = Real.arcsin (Real.sqrt 3 / 10) :=
sorry

end angle_B_is_arcsin_l504_504546


namespace snow_at_least_once_l504_504247

noncomputable def prob_snow_at_least_once (p1 p2 p3: ℚ) : ℚ :=
  1 - (1 - p1) * (1 - p2) * (1 - p3)

theorem snow_at_least_once : 
  prob_snow_at_least_once (1/2) (2/3) (3/4) = 23 / 24 := 
by
  sorry

end snow_at_least_once_l504_504247


namespace find_largest_angle_l504_504493

noncomputable def largest_angle_in_convex_pentagon (x : ℝ) : Prop :=
  let angle1 := 2 * x + 2
  let angle2 := 3 * x - 3
  let angle3 := 4 * x + 4
  let angle4 := 6 * x - 6
  let angle5 := x + 5
  angle1 + angle2 + angle3 + angle4 + angle5 = 540 ∧
  max (max angle1 (max angle2 (max angle3 angle4))) angle5 = angle4 ∧
  angle4 = 195.75

theorem find_largest_angle (x : ℝ) : largest_angle_in_convex_pentagon x := by
  sorry

end find_largest_angle_l504_504493


namespace meeting_handshakes_l504_504496

theorem meeting_handshakes :
  ∃ (n m k : ℕ), 
  n = 25 ∧
  m = 15 ∧
  (k = n - 5) ∧
  let within_group1_handshakes := (k.choose 2) in
  let total_interactions := (m * (n + (m - 1))) in
  within_group1_handshakes + total_interactions = 595 :=
sorry

end meeting_handshakes_l504_504496


namespace right_triangle_third_side_product_l504_504730

theorem right_triangle_third_side_product :
  ∀ (a b : ℝ), (a = 6 ∧ b = 8 ∧ (a^2 + b^2 = c^2 ∨ a^2 = b^2 - c^2)) →
  (a * b = 53.0) :=
by
  intros a b h
  sorry

end right_triangle_third_side_product_l504_504730


namespace chessboard_impossible_l504_504547

theorem chessboard_impossible :
  ∃ (f : Fin 8 × Fin 8 → Fin 65), (∀ i j, f ⟨i, j⟩ ∈ Finset.range 64) ∧
    (∀ (i j : Fin 7), ¬∃ (k₁ k₂ k₃ k₄ : ℕ), k₁ + k₂ + k₃ + k₄ ≡ 0 [MOD 5] 
    ∧ k₁ = f ⟨i, j⟩ 
    ∧ k₂ = f ⟨i, (j + 1) % 8⟩ 
    ∧ k₃ = f ⟨(i + 1) % 8, j⟩ 
    ∧ k₄ = f ⟨(i + 1) % 8, (j + 1) % 8⟩ ∧ k₁ ≠ k₂ ≠ k₃ ≠ k₄) ↔ False := sorry

end chessboard_impossible_l504_504547


namespace strictly_increasing_on_interval_l504_504382

noncomputable def log_base_2 (x : ℝ) := log x / log 2

open Real  -- Open the real number namespace to access sqrt, log, etc.

-- Define the function y = log_2(2x - x^2)
def f (x : ℝ) : ℝ := log_base_2 (2 * x - x^2)

-- Define the domain condition for the function
def domain_condition (x : ℝ) : Prop := 0 < x ∧ x < 2

-- Theorem: The function y = log_2(2x - x^2) is strictly increasing on (0, 1)
theorem strictly_increasing_on_interval : 
  ∀ x : ℝ, domain_condition x → 0 < x ∧ x < 1 → StrictMono (λ x, f x) :=
sorry

end strictly_increasing_on_interval_l504_504382


namespace num_valid_permutations_l504_504131

-- Type for representing the indices of problems
def ProblemIndex := Fin 5

-- Function to express the conditions
def validOrdering (p: Fin 5 → ProblemIndex) : Prop :=
  (p 3 > p 0) ∧ (p 4 > p 1) ∧ (p 4 > p 0)

-- Define the main theorem to prove the number of valid permutations
theorem num_valid_permutations :
  (Finset.univ.filter validOrdering).card = 25 :=
  sorry

end num_valid_permutations_l504_504131


namespace pairing_animals_l504_504394

def animals := ["cow", "pig", "horse"]

def num_cows : ℕ := 5
def num_pigs : ℕ := 4
def num_horses : ℕ := 7

theorem pairing_animals : 
  let total_pairs := 5 * 4 * 7! in
  total_pairs = 100800 := 
by
  sorry

end pairing_animals_l504_504394


namespace probability_x_lt_2y_l504_504335

open Set

theorem probability_x_lt_2y :
  let rect := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
  let tri := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 ∧ p.1 < 2 * p.2}
  (∃ (area_rect area_tri : ℝ), 
    area_rect = 12 ∧ area_tri = 6 ∧ ((area_tri / area_rect) = 1/2)) :=
by
  let rect := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
  let tri := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 ∧ p.1 < 2 * p.2}
  exists (12), (6),
  split,
  -- Proof of the step that area_rect = 12
  sorry,
  split,
  -- Proof of the step that area_tri = 6
  sorry,
  -- Proof of the step that the probability is 1/2
  exact (6 / 12 = 1 / 2)

end probability_x_lt_2y_l504_504335


namespace find_a_m_l504_504002

noncomputable def A := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) := {x | x^2 - a * x + a - 1 = 0}
noncomputable def C (m : ℝ) := {x | x^2 - m * x + 2 = 0}

theorem find_a_m (a m : ℝ) :
  ((B a ⊆ A ∧ A ∪ B a = A) ∧ (C m ⊆ A ∧ A ∩ C m = C m)) →
  (a = 2 ∨ a = 3) ∧ ((m = 3) ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) :=
by
  sorry

end find_a_m_l504_504002


namespace circle_area_l504_504502

theorem circle_area (x y : ℝ) :
  (3 * x ^ 2 + 3 * y ^ 2 + 12 * x - 9 * y - 27 = 0) →
  (∃ r : ℝ, r = sqrt 15.25 ∧ π * r ^ 2 = 15.25 * π) :=
by
  intro h
  sorry

end circle_area_l504_504502


namespace prime_count_between_50_and_70_l504_504089

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_50_and_70 : List ℕ :=
  [53, 59, 61, 67]

theorem prime_count_between_50_and_70 : 
  (primes_between_50_and_70.filter is_prime).length = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504089


namespace parts_manufactured_l504_504539

variable (initial_parts : ℕ) (initial_rate : ℕ) (increased_speed : ℕ) (time_diff : ℝ)
variable (N : ℕ)

-- initial conditions
def initial_parts := 35
def initial_rate := 35
def increased_speed := 15
def time_diff := 1.5

-- additional parts to be manufactured
noncomputable def additional_parts := N

-- equation representing the time differences
noncomputable def equation := (N / initial_rate) - (N / (initial_rate + increased_speed)) = time_diff

-- state the proof problem
theorem parts_manufactured : initial_parts + additional_parts = 210 :=
by
  -- Use the given conditions to solve the problem
  sorry

end parts_manufactured_l504_504539


namespace digits_of_fraction_l504_504468

theorem digits_of_fraction (n : ℕ) :
    let x := (5^7 : ℚ) / (10^5 * 8)
    let decimals := decimal_expansion x
    count_right_of_decimal decimals = 8 :=
by
  sorry

end digits_of_fraction_l504_504468


namespace replace_asterisk_with_monomial_l504_504199

theorem replace_asterisk_with_monomial (x : ℝ) :
  (∀ asterisk : ℝ, ((x^4 - 3)^2 + (x^3 + asterisk)^2) = (x^8 + x^6 + 9x^2 + 9)) ↔ asterisk = 3x :=
by sorry

end replace_asterisk_with_monomial_l504_504199


namespace smallest_integer_satisfying_inequality_l504_504758

theorem smallest_integer_satisfying_inequality :
  ∃ y : ℤ, (∀ k : ℤ, (3 * k - 6 < 15) → (y ≤ k)) ∧ (3 * y - 6 < 15) :=
begin
  use 6,
  split,
  { intros k hk,
    linarith, },
  { linarith, },
end

end smallest_integer_satisfying_inequality_l504_504758


namespace symmetric_graphs_l504_504682

def f (x : ℝ) := -exp(-x) - 2
def g (x : ℝ) := exp(x) + 2

theorem symmetric_graphs :
  ∀ x : ℝ, f(x) = -(g(-x)) := by
  sorry

end symmetric_graphs_l504_504682


namespace binomial_expansion_constant_term_l504_504510

theorem binomial_expansion_constant_term : 
  (∃ k : ℕ, (x ^ (6 * k - 6) = 1) ∧ (C(6, k) * (-2) ^ k = -12)) :=
by
  -- Setup the binomial expression and the constant term condition
  let expansion_general_term := λ k : ℕ, (binomial 6 k) * ((1/x) ^ (6 - k)) * ((-2) * x^5)^k
  -- Define the condition for the constant term
  have h_constant : ∀ k, (x ^ (6 * k - 6) = 1) ↔ k = 1 := by
     sorry
  -- prove the constant term calculation
  use 1
  split
  { simp [h_constant] },
  { simp [binomial, pow_succ] }

end binomial_expansion_constant_term_l504_504510


namespace prime_numbers_between_50_and_70_l504_504050

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_between_50_and_70 : 
  (finset.filter is_prime (finset.range 71)).count (λ n, 50 ≤ n ∧ n ≤ 70) = 4 := 
sorry

end prime_numbers_between_50_and_70_l504_504050


namespace simplify_nested_sqrt_l504_504640

-- Define the expressions under the square roots
def expr1 : ℝ := 12 + 8 * real.sqrt 3
def expr2 : ℝ := 12 - 8 * real.sqrt 3

-- Problem statement to prove
theorem simplify_nested_sqrt : real.sqrt expr1 + real.sqrt expr2 = 4 * real.sqrt 2 :=
by
  sorry

end simplify_nested_sqrt_l504_504640


namespace pq_sum_is_neg2x2_minus_4_l504_504237

-- Define p(x) and q(x) based on given conditions
def p (x : ℝ) : ℝ := -2 * x + 2
def q (x : ℝ) : ℝ := -2 * x ^ 2 + 2 * x + 4

-- Given conditions transformed to Lean assumptions
axiom p_at_0 : p 0 = 2
axiom q_at_0 : q 0 = 4

-- Define the theorem stating the final result to be proved
theorem pq_sum_is_neg2x2_minus_4 : 
  ∀ x : ℝ, p(x) + q(x) = -2 * x^2 - 4 :=
by 
  -- Provide the statement, without the proof steps
  sorry

end pq_sum_is_neg2x2_minus_4_l504_504237


namespace sum_of_first_6_terms_l504_504251

noncomputable def sequence (a : ℕ → ℝ) : ℕ → ℝ
| 0     := sorry -- undefined
| 1     := sorry -- undefined
| (n+2) := a n + a (n+1)

theorem sum_of_first_6_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (h₁ : ∀ n ≥ 2, a (n + 1) = a n + a (n - 1)) (h₂ : a 5 = 1) :
  S 6 = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 → S 6 = 4 :=
begin
  sorry,
end

end sum_of_first_6_terms_l504_504251


namespace karting_routes_10_min_l504_504677

-- Define the recursive function for M_{n, A}
def num_routes : ℕ → ℕ
| 0 => 1   -- Starting point at A for 0 minutes (0 routes)
| 1 => 0   -- Impossible to end at A in just 1 move
| 2 => 1   -- Only one way to go A -> B -> A in 2 minutes
| n + 1 =>
  if n = 1 then 0 -- Additional base case for n=2 as defined
  else if n = 2 then 1
  else num_routes (n - 1) + num_routes (n - 2)

theorem karting_routes_10_min : num_routes 10 = 34 := by
  -- Proof steps go here
  sorry

end karting_routes_10_min_l504_504677


namespace replace_star_with_3x_l504_504179

theorem replace_star_with_3x (x : ℝ) :
  (x^4 - 3)^2 + (x^3 + 3x)^2 = x^8 + x^6 + 9x^2 + 9 :=
by
  sorry

end replace_star_with_3x_l504_504179


namespace octahedron_side_length_l504_504797

structure Cuboid where
  P1 P2 P3 P4 P1' P2' P3' P4' : ℝ × ℝ × ℝ
  adj_P1 : P2 = (2, 0, 0) ∧ P3 = (0, 2, 0) ∧ P4 = (0, 0, 2)
  opp_P1 : P1' = (2, 2, 2) ∧ P2' = (0, 2, 2) ∧ P3' = (2, 0, 2) ∧ P4' = (2, 2, 0)

def octahedron_vertices (P1 P2 P3 P4 P1' P2' P3' P4': ℝ × ℝ × ℝ) :=
  (P1.x + (2/3) * (P2.x - P1.x), P1.y + (2/3) * (P2.y - P1.y), P1.z + (2/3) * (P2.z - P1.z)) ::
  (P1.x + (2/3) * (P3.x - P1.x), P1.y + (2/3) * (P3.y - P1.y), P1.z + (2/3) * (P3.z - P1.z)) ::
  (P1.x + (2/3) * (P4.x - P1.x), P1.y + (2/3) * (P4.y - P1.y), P1.z + (2/3) * (P4.z - P1.z)) ::
  (P1'.x + (2/3) * (P2'.x - P1'.x), P1'.y + (2/3) * (P2'.y - P1'.y), P1'.z + (2/3) * (P2'.z - P1'.z)) ::
  (P1'.x + (2/3) * (P3'.x - P1'.x), P1'.y + (2/3) * (P3'.y - P1'.y), P1'.z + (2/3) * (P3'.z - P1'.z)) ::
  (P1'.x + (2/3) * (P4'.x - P1'.x), P1'.y + (2/3) * (P4'.y - P1'.y), P1'.z + (2/3) * (P4'.z - P1'.z)) :: []

theorem octahedron_side_length : 
  ∀ (P1 P2 P3 P4 P1' P2' P3' P4': ℝ × ℝ × ℝ), 
  P1 = (0, 0, 0) → P1' = (2, 2, 2) → 
  Cuboid P1 P2 P3 P4 P1' P2' P3' P4' P2 P3 P4 P1' P2' P3' P4' →
  let v := octahedron_vertices P1 P2 P3 P4 P1' P2' P3' P4' in 
  dist (v.head) (v.tail.head) = 4 * ℝ.sqrt 2 / 3 := 
by 
  intros P1 P2 P3 P4 P1' P2' P3' P4' h1 h2 c
  simp [octahedron_vertices, dist]
  sorry

end octahedron_side_length_l504_504797


namespace factorization1_factorization2_factorization3_factorization4_l504_504393

-- Question 1
theorem factorization1 (a b : ℝ) :
  4 * a^2 * b - 6 * a * b^2 = 2 * a * b * (2 * a - 3 * b) :=
by 
  sorry

-- Question 2
theorem factorization2 (x y : ℝ) :
  25 * x^2 - 9 * y^2 = (5 * x + 3 * y) * (5 * x - 3 * y) :=
by 
  sorry

-- Question 3
theorem factorization3 (a b : ℝ) :
  2 * a^2 * b - 8 * a * b^2 + 8 * b^3 = 2 * b * (a - 2 * b)^2 :=
by 
  sorry

-- Question 4
theorem factorization4 (x : ℝ) :
  (x + 2) * (x - 8) + 25 = (x - 3)^2 :=
by 
  sorry

end factorization1_factorization2_factorization3_factorization4_l504_504393


namespace min_value_of_3a_plus_b_l504_504091

theorem min_value_of_3a_plus_b 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a + 2 * b = a * b) : 3 * a + b ≥ 7 + 2 * real.sqrt 6 := 
by {
  sorry
}

end min_value_of_3a_plus_b_l504_504091


namespace primes_between_50_and_70_l504_504074

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).filter is_prime |>.length

theorem primes_between_50_and_70 : count_primes_in_range 50 70 = 4 :=
by
  sorry

end primes_between_50_and_70_l504_504074


namespace train_crossing_time_l504_504815

def speed := 60 -- in km/hr
def length := 300 -- in meters
def speed_in_m_per_s := (60 * 1000) / 3600 -- converting speed from km/hr to m/s
def expected_time := 18 -- in seconds

theorem train_crossing_time :
  (300 / (speed_in_m_per_s)) = expected_time :=
sorry

end train_crossing_time_l504_504815


namespace prime_count_between_50_and_70_l504_504072

open Nat

theorem prime_count_between_50_and_70 : 
  (finset.filter Nat.prime (finset.range 71 \ finset.range 51).card = 4) := 
begin
  sorry
end

end prime_count_between_50_and_70_l504_504072


namespace combinations_no_equilateral_triangle_l504_504660

def isEquilateralTriangle (a b c : ℕ) : Prop :=
  if (b - a) % 6 = (c - b) % 6 then true else false

def count_combinations_without_equilateral (n : ℕ) : ℕ :=
  let combinations := (list.combinations (list.range 6) n).filter (λ comb, ¬ (isEquilateralTriangle comb[0] comb[1] comb[2]))
  combinations.length

theorem combinations_no_equilateral_triangle : count_combinations_without_equilateral 3 = 18 :=
by {
  sorry
}

end combinations_no_equilateral_triangle_l504_504660


namespace triangle_third_side_lengths_product_l504_504745

def hypotenuse (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

def leg (c b : ℝ) : ℝ :=
  real.sqrt (c^2 - b^2)

theorem triangle_third_side_lengths_product :
  let a := 6
  let b := 8
  let hyp := hypotenuse a b
  let leg := leg b a
  real.round (hyp * leg * 10) / 10 = 52.9 :=
by {
  -- Definitions and calculations have been provided in the problem statement
  sorry
}

end triangle_third_side_lengths_product_l504_504745


namespace sum_of_first_1985_bad_numbers_l504_504337

-- Define the concept of "bad number"
def is_bad_number (n : ℕ) : Prop :=
  (nat.popcount n) % 2 = 0

-- Define the sum of the first n bad numbers
noncomputable def sum_first_n_bad_numbers (n : ℕ) : ℕ :=
  (list.range (2 * n)).filter is_bad_number |>.take n |>.sum

-- The theorem statement
theorem sum_of_first_1985_bad_numbers :
  sum_first_n_bad_numbers 1985 = 2^21 + 2^20 + 2^19 + 2^18 + 2^13 + 2^11 + 2^9 + 2^8 + 2^5 + 2 + 1 :=
by
  sorry

end sum_of_first_1985_bad_numbers_l504_504337


namespace select_second_grade_students_l504_504106

variable {first_grade_students second_grade_students total_selected first_grade_selected : ℕ}

axiom students_data :
  first_grade_students = 400 ∧
  second_grade_students = 360 ∧
  total_selected = 56 ∧
  first_grade_selected = 20

theorem select_second_grade_students (x : ℕ) :
  students_data →
  (x : ℕ) = (second_grade_students * first_grade_selected) / first_grade_students :=
by
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  sorry

end select_second_grade_students_l504_504106


namespace inequality_proof_l504_504902

noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  x^2 + (Real.sqrt m) * x + m + 1

theorem inequality_proof (m : ℝ) (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 < x i) (hm : 0 ≤ m) :
  f m (Real.geomMean x) ≤ Real.geomMean (λ i, f m (x i)) ∧ 
  (f m (Real.geomMean x) = Real.geomMean (λ i, f m (x i)) ↔ ∀ i j, x i = x j) :=
by
  sorry

end inequality_proof_l504_504902


namespace right_triangle_third_side_product_l504_504733

theorem right_triangle_third_side_product :
  let a := 6
  let b := 8
  let c1 := Real.sqrt (a^2 + b^2)     -- Hypotenuse when a and b are legs
  let c2 := Real.sqrt (b^2 - a^2)     -- Other side when b is the hypotenuse
  20 * Real.sqrt 7 ≈ 52.7 := 
by
  sorry

end right_triangle_third_side_product_l504_504733


namespace simplify_and_evaluate_l504_504203

theorem simplify_and_evaluate :
  let a := 1
  let b := 2
  (a - b) ^ 2 - a * (a - b) + (a + b) * (a - b) = -1 := by
  sorry

end simplify_and_evaluate_l504_504203


namespace kate_change_l504_504134

def first_candy_cost : ℝ := 0.54
def second_candy_cost : ℝ := 0.35
def third_candy_cost : ℝ := 0.68
def amount_given : ℝ := 5.00

theorem kate_change : amount_given - (first_candy_cost + second_candy_cost + third_candy_cost) = 3.43 := by
  sorry

end kate_change_l504_504134


namespace necessary_sufficient_condition_geometric_sequence_l504_504009

noncomputable def an_geometric (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = r * a n

theorem necessary_sufficient_condition_geometric_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ) (p q : ℝ) (h_sum : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
  (h_eq : ∀ n : ℕ, a (n + 1) = p * S n + q) :
  (a 1 = q) ↔ (∃ r : ℝ, an_geometric a r) :=
sorry

end necessary_sufficient_condition_geometric_sequence_l504_504009


namespace problem_solution_l504_504794

-- Define the problem
variables (total_employees male_employees female_employees team_size : ℕ)
variable (male_in_team : ℕ)
variable (female_in_team : ℕ)
variable (select_probability : ℚ)
variable (female_probability : ℚ)
variables (exp1_data exp2_data : list ℚ)
variable (exp1_variance exp2_variance : ℚ)

-- Given conditions
def conditions := (total_employees = 60) ∧ 
                  (male_employees = 45) ∧ 
                  (female_employees = 15) ∧ 
                  (team_size = 4) ∧ 
                  (male_in_team = 3) ∧ 
                  (female_in_team = 1) ∧ 
                  (exp1_data = [68, 70, 71, 72, 74]) ∧ 
                  (exp2_data = [69, 70, 70, 72, 74])

-- Prove the solution
theorem problem_solution : 
    conditions →
    (select_probability = 1 / 15) ∧ 
    (female_probability = 1 / 2) ∧ 
    (exp1_variance = 4) ∧ 
    (exp2_variance = 3.2) ∧ 
    (exp1_variance > exp2_variance) :=
begin
  -- Sorry proof part is ignored as per the requirement
  sorry
end

end problem_solution_l504_504794


namespace prime_count_between_50_and_70_l504_504083

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_50_and_70 : List ℕ :=
  [53, 59, 61, 67]

theorem prime_count_between_50_and_70 : 
  (primes_between_50_and_70.filter is_prime).length = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504083


namespace find_a_l504_504899

-- Define the function f and its derivative
def f (a : ℝ) (x : ℝ) := Real.exp (-x) - a * x^2 * (deriv (f a) x)

-- Given f'(x) is -e^{-x} - 2ax f'(x)
def f' (a : ℝ) (x : ℝ) := -Real.exp (-x) - 2 * a * x * (deriv (f a) x)

-- The condition given f'(1) = 1/e
theorem find_a (a : ℝ) (h : deriv (f a) 1 = 1 / Real.exp 1) : a = -1 := 
by
  sorry

end find_a_l504_504899


namespace monotonically_increasing_intervals_triangle_area_l504_504465

open Real

def vec_a (x : ℝ) : ℝ × ℝ := (sin x, -1)
def vec_b (x : ℝ) : ℝ × ℝ := (sqrt 3 * cos x, -1 / 2)

def f (x : ℝ) : ℝ := (vec_a x + vec_b x) • (vec_a x) - 2

theorem monotonically_increasing_intervals :
  ∀ k : ℤ, monotone_on f (Icc (k * π - π / 6) (k * π + π / 3)) := sorry

theorem triangle_area :
  let A := π / 3 in
  let a := sqrt 3 in
  let c := 1 in
  f A = 1 → 
  acute A →
  (∃ b : ℝ, b = 2 ∧ ∃ S : ℝ, S = (1 / 2) * b * c * sin A ∧ S = sqrt 3 / 2) := sorry

end monotonically_increasing_intervals_triangle_area_l504_504465


namespace smallest_k_integer_product_l504_504376

noncomputable def seq_b : ℕ → ℝ
| 0     := 1
| 1     := real.root 23 3
| (n+2) := seq_b(n + 1) * (seq_b n) ^ 3

def is_integer (x : ℝ) : Prop :=
  ∃ (k : ℤ), x = k

theorem smallest_k_integer_product :
  ∃ k, (∀ n, 0 ≤ n ∧ n ≤ k → seq_b (n+1) * seq_b n ^ 3 = seq_b n *
            (seq_b (n-1)) ^ 3) →
  is_integer (finset.prod (finset.range k) seq_b) :=
sorry

end smallest_k_integer_product_l504_504376


namespace simplify_nested_sqrt_l504_504639

-- Define the expressions under the square roots
def expr1 : ℝ := 12 + 8 * real.sqrt 3
def expr2 : ℝ := 12 - 8 * real.sqrt 3

-- Problem statement to prove
theorem simplify_nested_sqrt : real.sqrt expr1 + real.sqrt expr2 = 4 * real.sqrt 2 :=
by
  sorry

end simplify_nested_sqrt_l504_504639


namespace prove_lambda_prove_zeros_of_g_l504_504932

-- Define the interval
def interval_0_pi (x : Real) : Prop := 0 < x ∧ x < π

-- Define the function f
def f (x λ : Real) : Real := Real.exp x - λ * Real.sin x

-- Define the function g with the interval for a
def g (x a : Real) (ha : 1 ≤ a ∧ a < 4) : Real := Real.exp x + 3 * Real.sin x - 1 - a * x

theorem prove_lambda (λ : ℕ) (h1 : λ = 3) :
  (∀ x : Real, interval_0_pi x → f x λ > 0) := 
  by
  sorry

theorem prove_zeros_of_g (a : Real) (ha : 1 ≤ a ∧ a < 4) :
  ∃ z1 z2 : Real, z1 ≠ z2 ∧ g z1 a ha = 0 ∧ g z2 a ha = 0 ∧ 
  (∀ z : Real, g z a ha = 0 → z = z1 ∨ z = z2) :=
  by
  sorry

end prove_lambda_prove_zeros_of_g_l504_504932


namespace substitute_monomial_to_simplify_expr_l504_504185

theorem substitute_monomial_to_simplify_expr (k : ℤ) : 
  ( ∃ k : ℤ, (x^4 - 3)^2 + (x^3 + k * x)^2 after expanding has exactly four terms) := 
begin
  use 3,
  sorry
end

end substitute_monomial_to_simplify_expr_l504_504185


namespace no_sum_of_two_squares_l504_504772

theorem no_sum_of_two_squares (n : ℤ) (h : n % 4 = 3) : ¬∃ a b : ℤ, n = a^2 + b^2 := 
by
  sorry

end no_sum_of_two_squares_l504_504772


namespace train_probability_correct_l504_504355

/-- Define the necessary parameters and conditions --/
noncomputable def train_arrival_prob (train_start train_wait max_time_Alex max_time_train : ℝ) : ℝ :=
  let total_possible_area := max_time_Alex * max_time_train
  let overlap_area := (max_time_train - train_wait) * train_wait + (train_wait) * max_time_train / 2
  overlap_area / total_possible_area

/-- Main theorem stating that the probability is 3/10 --/
theorem train_probability_correct :
  train_arrival_prob 0 15 75 60 = 3 / 10 :=
by sorry

end train_probability_correct_l504_504355


namespace possible_value_of_a_eq_neg1_l504_504294

theorem possible_value_of_a_eq_neg1 (a : ℝ) : (-6 * a ^ 2 = 3 * (4 * a + 2)) → (a = -1) :=
by
  intro h
  have H : a^2 + 2*a + 1 = 0
  · sorry
  show a = -1
  · sorry

end possible_value_of_a_eq_neg1_l504_504294


namespace parts_manufactured_l504_504538

variable (initial_parts : ℕ) (initial_rate : ℕ) (increased_speed : ℕ) (time_diff : ℝ)
variable (N : ℕ)

-- initial conditions
def initial_parts := 35
def initial_rate := 35
def increased_speed := 15
def time_diff := 1.5

-- additional parts to be manufactured
noncomputable def additional_parts := N

-- equation representing the time differences
noncomputable def equation := (N / initial_rate) - (N / (initial_rate + increased_speed)) = time_diff

-- state the proof problem
theorem parts_manufactured : initial_parts + additional_parts = 210 :=
by
  -- Use the given conditions to solve the problem
  sorry

end parts_manufactured_l504_504538


namespace f_properties_l504_504283

open Real

-- Define the function f(x) = x^2
noncomputable def f (x : ℝ) : ℝ := x^2

-- Define the statement to be proved
theorem f_properties (x₁ x₂ : ℝ) (x : ℝ) (h : 0 < x) :
  (f (x₁ * x₂) = f x₁ * f x₂) ∧ 
  (deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end f_properties_l504_504283


namespace master_craftsman_parts_l504_504522

/-- 
Given:
  (1) the master craftsman produces 35 parts in the first hour,
  (2) at the rate of 35 parts/hr, he would be one hour late to meet the quota,
  (3) by increasing his speed by 15 parts/hr, he finishes the quota 0.5 hours early,
Prove that the total number of parts manufactured during the shift is 210.
-/
theorem master_craftsman_parts (N : ℕ) (quota : ℕ) 
  (initial_rate : ℕ := 35)
  (increased_rate_diff : ℕ := 15)
  (extra_time_slow : ℕ := 1)
  (time_saved_fast : ℕ := 1/2) :
  (quota = initial_rate * (extra_time_slow + 1) + N ∧
   increased_rate_diff = 15 ∧
   increased_rate_diff = λ (x : ℕ), initial_rate + x ∧
   time_saved_fast = 1/2 ∧
   N = 35) →
  quota = 210 := 
by
  sorry

end master_craftsman_parts_l504_504522


namespace darma_peanut_consumption_l504_504857

theorem darma_peanut_consumption :
  ∀ (t : ℕ) (rate : ℕ),
  (rate = 20 / 15) →  -- Given the rate of peanut consumption
  (t = 6 * 60) →     -- Given that the total time is 6 minutes
  (rate * t = 480) :=  -- Prove that the total number of peanuts eaten in 6 minutes is 480
by
  intros t rate h_rate h_time
  sorry

end darma_peanut_consumption_l504_504857


namespace totalSolutions_l504_504956

noncomputable def systemOfEquations (a b c d a1 b1 c1 d1 x y : ℝ) : Prop :=
  a * x^2 + b * x * y + c * y^2 = d ∧ a1 * x^2 + b1 * x * y + c1 * y^2 = d1

theorem totalSolutions 
  (a b c d a1 b1 c1 d1 : ℝ) 
  (h₀ : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)
  (h₁ : a1 ≠ 0 ∨ b1 ≠ 0 ∨ c1 ≠ 0) :
  ∃ x y : ℝ, systemOfEquations a b c d a1 b1 c1 d1 x y :=
sorry

end totalSolutions_l504_504956


namespace inverse_of_f_at_27_l504_504445

-- Define the function f(x)
def f (x : ℝ) : ℝ := 5 * x ^ 2 + 7

-- Define the proposition we want to prove
theorem inverse_of_f_at_27 :
  f(2) = 27 :=
by
  -- skipped proof
  sorry

end inverse_of_f_at_27_l504_504445


namespace midpoint_CH_l504_504836

-- Define points and their positions
variables {A B C H D P O1 O2 : Type} 
-- Orthocenter condition
variable (is_orthocenter : Orthocenter H A B C)
-- Circles with given diameters and circumcircle constraints
variable (circle1 : Circle (diameter A B))
variable (circle2 : Circle (circumcircle B C H))
-- Intersection of circles
variable (D_on_intersection : OnCircle D circle1 ∧ OnCircle D circle2)
-- Relationship with line AD
variable (AD_line : Line A D)
variable (CH_line : Line C H)
variable (P_on_intersection : Intersection (extended AD_line) (CH_line) P)

-- The proposition to prove
theorem midpoint_CH : midpoint P C H :=
by
  sorry

end midpoint_CH_l504_504836


namespace probability_of_selection_of_Ravi_l504_504719

theorem probability_of_selection_of_Ravi :
  let P_Ram := 5/7
  let P_Ram_and_Ravi := 0.14285714285714288
  ∃ P_Ravi, P_Ram_and_Ravi = P_Ram * P_Ravi ∧ P_Ravi = 0.2 :=
begin
  sorry
end

end probability_of_selection_of_Ravi_l504_504719


namespace parabola_intersects_x_axis_at_one_point_l504_504123

theorem parabola_intersects_x_axis_at_one_point (k : ℝ) : (∃ x : ℝ, x^2 + 2 * x + k = 0) → k = 1 :=
by
  have h : 4 - 4 * k = 0 := sorry
  exact h

end parabola_intersects_x_axis_at_one_point_l504_504123


namespace value_v3_correct_using_horners_method_l504_504272

theorem value_v3_correct_using_horners_method :
  let x := -2
  let f := (λ x, (((((x) - 5) * x + 6) * x + 0) * x + 1) * x + 3) * x + 2
  let v0 := 1
  let v1 := x + -5
  let v2 := v1 * x + 6
  let v3 := v2 * x + 0
  in v3 = -40 := by
    sorry

end value_v3_correct_using_horners_method_l504_504272


namespace area_of_cod_l504_504114

open Finset

theorem area_of_cod (A B C D O : Type) [real.of_eq] :
  (area A O B = 12) ∧ (area A O D = 16) ∧ (area B O C = 18) → (area C O D = 24) := by
  sorry

end area_of_cod_l504_504114


namespace range_of_a_l504_504481

noncomputable def inequality_always_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0

theorem range_of_a (a : ℝ) : inequality_always_holds a ↔ 0 ≤ a ∧ a < 1 := 
by
  sorry

end range_of_a_l504_504481


namespace equidistant_from_circumcenter_l504_504149

-- Define the isosceles triangle ABC with AB = AC
variables {A B C : Type*} [linear_ordered_field A]

-- Define M on line segment [BC]
variable (M : Type*)

-- Define intersections
variables (E F : Type*)

-- Define the circumcircle center O
variable (O : Type*)

-- Conditions covered in types and variables above.
-- Now, let's state the theorem that E and F are equidistant from O
theorem equidistant_from_circumcenter 
  (is_isosceles : AB = AC)
  (intersect_E : parallel_to_AC_through_M_intersects_AB E M)
  (intersect_F : parallel_to_AB_through_M_intersects_AC F M)
  (power_of_E : EA * EB = k)
  (power_of_F : FA * FC = k) -- Powers are equal implies equal distance from O
  : distance O E = distance O F :=
sorry

end equidistant_from_circumcenter_l504_504149


namespace problem_c_minus_3d_l504_504274

-- Definitions of c and d
def c : ℂ := 6 - 3 * complex.i
def d : ℂ := 2 - 5 * complex.i

-- The theorem to be proved
theorem problem_c_minus_3d : c - 3 * d = 12 * complex.i :=
by {
  -- Proof will be provided here
  sorry
}

end problem_c_minus_3d_l504_504274


namespace pain_subsided_days_l504_504128

-- Define the problem conditions in Lean
variable (x : ℕ) -- the number of days it takes for the pain to subside

-- Condition 1: The injury takes 5 times the pain subsiding period to fully heal
def injury_healing_days := 5 * x

-- Condition 2: James waits an additional 3 days after the injury is fully healed
def workout_waiting_days := injury_healing_days + 3

-- Condition 3: James waits another 3 weeks (21 days) before lifting heavy
def total_days_until_lifting_heavy := workout_waiting_days + 21

-- Given the total days until James can lift heavy is 39 days, prove x = 3
theorem pain_subsided_days : 
    total_days_until_lifting_heavy x = 39 → x = 3 := by
  sorry

end pain_subsided_days_l504_504128


namespace math_problem_statement_l504_504503

noncomputable def fixed_point_F : ℝ × ℝ := (sqrt 3, 0)

def line_through_fixed_point := 
  ∀ k : ℝ, (sqrt 3 * k + 1) * x + (k - sqrt 3) * y - (3 * k + sqrt 3) = 0

def max_distance_from_point_to_ellipse (F : ℝ × ℝ) (C : ℝ × ℝ → Prop) (d : ℝ) :=
  ∀ p : ℝ × ℝ, C p → dist p F ≤ d

def ellipse_equation (p : ℝ × ℝ) :=
  let (x, y) := p in 
  x^2 / 4 + y^2 = 1

def circle_with_four_intersections := 
  ∃ r : ℝ, r > 1 ∧ r < 2 ∧ ∀ p : ℝ × ℝ, ellipse_equation p → circle r p

def positional_relationship (m n : ℝ) (r : ℝ) :=
  let d1 := 1 / sqrt (m^2 + n^2) in
  let d2 := 4 / sqrt (m^2 + n^2) in
  (d1 ≤ 1 ∧ 1 < r ∧ d2 ≥ 2 ∧ 2 > r)

theorem math_problem_statement :
  (line_through_fixed_point) → 
  max_distance_from_point_to_ellipse fixed_point_F ellipse_equation (2 + sqrt 3) →
  circle_with_four_intersections →
  ∀ (m n : ℝ), ellipse_equation (m, n) → 
  (1 < sqrt (m^2 / 4 + n^2) ∧ sqrt (m^2 / 4 + n^2) < 2) →
  positional_relationship m n) :=
sorry

end math_problem_statement_l504_504503


namespace probability_A_union_not_B_l504_504500

-- Definitions of the events A and B.
def events : set ℕ := {1, 2, 3, 4, 5, 6}
def event_A : set ℕ := {2, 4}
def event_B : set ℕ := {1, 2, 3, 4}
def event_not_B : set ℕ := events \ event_B -- Complementary event of B

-- Probability function given total outcomes and successful outcomes
def probability (target total : set ℕ) : ℚ := target.card / total.card

-- Conditions
lemma event_A_probability : probability event_A events = 1 / 3 := by sorry
lemma event_B_probability : probability event_B events = 2 / 3 := by sorry
lemma event_not_B_probability : probability event_not_B events = 1 / 3 := by sorry
lemma mutually_exclusive : disjoint event_A event_not_B := by sorry

-- Prove the final probability
theorem probability_A_union_not_B : 
  probability (event_A ∪ event_not_B) events = 2 / 3 := by 
{
  rw [←set.card_union_add_card_inter_eq_card_add_card event_A event_not_B,
      set.disjoint_iff_inter_eq_empty.mp mutually_exclusive],
  simp [event_A_probability, event_not_B_probability],
  norm_num,
  sorry
}

end probability_A_union_not_B_l504_504500


namespace area_enclosed_between_circles_is_correct_l504_504705

open Real

noncomputable def area_enclosed_between_circles 
  (r1 r2 r3 : ℝ) : ℝ :=
  let a := r1 + r2
  let b := r2 + r3
  let c := r3 + r1
  let s := (a + b + c) / 2
  let T := sqrt (s * (s - a) * (s - b) * (s - c))
  let alpha := 2 * asin ((2 * T) / a * b)
  let beta := 2 * asin ((2 * T) / b * c)
  let gamma := 2 * asin ((2 * T) / c * a)
  let t1 := (pi * r1^2 * alpha) / (2 * pi)
  let t2 := (pi * r2^2 * beta) / (2 * pi)
  let t3 := (pi * r3^2 * gamma) / (2 * pi)
  T - (t1 + t2 + t3)

theorem area_enclosed_between_circles_is_correct : 
  area_enclosed_between_circles 4 9 36 = 11.5 := by
sorry

end area_enclosed_between_circles_is_correct_l504_504705


namespace intersection_plane_is_regular_hexagon_l504_504507

def is_midpoint (X Y Z : Point) : Prop :=
  dist X Z = dist Y Z ∧ 
  ∀ W, dist X W = dist Y W → W = Z

noncomputable def cube_points (A B C D A1 B1 C1 D1 P M N : Point) : Prop :=
  is_midpoint A A1 P ∧
  is_midpoint B C M ∧
  is_midpoint C C1 N

theorem intersection_plane_is_regular_hexagon (A B C D A1 B1 C1 D1 P M N : Point) :
  cube_points A B C D A1 B1 C1 D1 P M N →
  (∃ hexagon : List Point, is_regular_hexagon hexagon ∧ 
    intersects_plane (Plane.of_points P M N) (cube_surface A B C D A1 B1 C1 D1) = hexagon) :=
by sorry

end intersection_plane_is_regular_hexagon_l504_504507


namespace proof_problem_l504_504472

noncomputable def n : ℕ :=
7

def C_n_2 (n : ℕ) : ℕ :=
n.factorial / (2.factorial * (n - 2).factorial)

def A_2_2 : ℕ :=
2

def given_condition (n : ℕ) : Prop :=
(C_n_2 n) * A_2_2 = 42

def desired_expression (n : ℕ) : ℕ :=
n.factorial / (3.factorial * (n - 3).factorial)

theorem proof_problem : given_condition n → desired_expression n = 35 :=
by sorry

end proof_problem_l504_504472


namespace selling_price_correct_l504_504326

-- Define the conditions
def cost_price : ℝ := 900
def gain_percentage : ℝ := 0.2222222222222222

-- Define the selling price calculation
def profit := cost_price * gain_percentage
def selling_price := cost_price + profit

-- The problem statement in Lean 4
theorem selling_price_correct : selling_price = 1100 := 
by
  -- Proof to be filled in later
  sorry

end selling_price_correct_l504_504326


namespace count_decreasing_functions_l504_504032

theorem count_decreasing_functions :
  let n := 2013
  let f : {i : ℕ // 1 ≤ i ∧ i ≤ n} → {j : ℕ // 1 ≤ j ∧ j ≤ n}
  in (∀ (i j : {m : ℕ // 1 ≤ m ∧ m ≤ n}), (i < j) → (f j).val < (f i).val + j.val - i.val)
     →
     (finset.card (finset.univ.filter (λ (f : {i : ℕ // 1 ≤ i ∧ i ≤ n} → {j : ℕ // 1 ≤ j ∧ j ≤ n}),
       (∀ (i j : {m : ℕ // 1 ≤ m ∧ m ≤ n}), i < j → (f j).val < (f i).val + (j.val - i.val))))) = nat.choose 4025 2013 :=
begin
  sorry
end

end count_decreasing_functions_l504_504032


namespace suitable_n_l504_504860

theorem suitable_n (n : ℕ) (hn : 2 ≤ n)
  (S : Finset ℤ) (hS₁ : S.card = n)
  (hS₂ : ∀ (α : Fin n → ℤ), α.to_finset = S → (Finset.univ.sum (λ i => α i)) % n ≠ 0 → 
    ∃ (σ : Fin n → Fin n), (Finset.univ.sum (λ i => (σ i + 1) * (α (σ i)))) % n = 0) :
  ∃ (k : ℕ), n = 2 ^ k ∨ ∃ (m : ℕ), odd m ∧ n = m := 
sorry

end suitable_n_l504_504860


namespace number_of_even_whole_numbers_between_9_and_27_l504_504031

theorem number_of_even_whole_numbers_between_9_and_27 : 
  ∃ n : ℕ, n = 9 ∧ count_even_between (3^2 + 1) (3^3 - 1) = n := by
  sorry

noncomputable def count_even_between (a b : ℕ) : ℕ :=
  List.length (List.filter (λ x, x % 2 = 0) (List.range' (a + 1) (b - a - 1) ))

#eval count_even_between 10 26  -- This should return 9

end number_of_even_whole_numbers_between_9_and_27_l504_504031


namespace darma_eats_peanuts_l504_504855

/--
Darma can eat 20 peanuts in 15 seconds. Prove that she can eat 480 peanuts in 6 minutes.
-/
theorem darma_eats_peanuts (rate: ℕ) (per_seconds: ℕ) (minutes: ℕ) (conversion: ℕ) : 
  (rate = 20) → (per_seconds = 15) → (minutes = 6) → (conversion = 60) → 
  rate * (conversion * minutes / per_seconds) = 480 :=
by
  intros hrate hseconds hminutes hconversion
  rw [hrate, hseconds, hminutes, hconversion]
  -- skipping the detailed proof
  sorry

end darma_eats_peanuts_l504_504855


namespace center_of_mass_exists_unique_vector_relation_to_center_of_mass_l504_504297

variable {n : ℕ}
variable {m : Fin n → ℝ}
variable {X : Fin n → EuclideanSpace ℝ (Fin 3)}

noncomputable def center_of_mass (n : ℕ) (m : Fin n → ℝ) (X : Fin n → EuclideanSpace ℝ (Fin 3)) :=
  (∑ i, m i • X i) / (∑ i, m i)

theorem center_of_mass_exists_unique (n : ℕ) (m : Fin n → ℝ) (X : Fin n → EuclideanSpace ℝ (Fin 3)) :
  ∃! O : EuclideanSpace ℝ (Fin 3), O = center_of_mass n m X :=
sorry

theorem vector_relation_to_center_of_mass (X : EuclideanSpace ℝ (Fin 3)) (O : EuclideanSpace ℝ (Fin 3))
  (n : ℕ) (m : Fin n → ℝ) (X_pts : Fin n → EuclideanSpace ℝ (Fin 3)) 
  (hO : O = center_of_mass n m X_pts) :
  (X - O) = (1 / (∑ i, m i)) • (∑ i, m i • (X_pts i - X)) :=
sorry

end center_of_mass_exists_unique_vector_relation_to_center_of_mass_l504_504297


namespace lost_codes_count_l504_504770

theorem lost_codes_count : 
  let total_codes_with_leading_zeros := (10 : ℕ) ^ 5
  let total_codes_without_leading_zeros := 9 * (10 : ℕ) ^ 4
  total_codes_with_leading_zeros - total_codes_without_leading_zeros = 10_000 :=
by
  let total_codes_with_leading_zeros := (10 : ℕ) ^ 5;
  let total_codes_without_leading_zeros := 9 * (10 : ℕ) ^ 4;
  have h : total_codes_with_leading_zeros - total_codes_without_leading_zeros = 10_000 := sorry;
  exact h

end lost_codes_count_l504_504770


namespace real_solution_count_l504_504402

noncomputable def equation (x : ℝ) : ℝ :=
  (x^1010 + 1) * (∑ i in (range 0 505).map (λ n, if n % 2 = 0 then x^(1008 - 2*n) else 0)) - 1010 * x^1009

theorem real_solution_count :
  ∃! x : ℝ, equation x = 0 :=
by
  sorry

end real_solution_count_l504_504402


namespace primes_between_50_and_70_l504_504076

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def count_primes_in_range (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).filter is_prime |>.length

theorem primes_between_50_and_70 : count_primes_in_range 50 70 = 4 :=
by
  sorry

end primes_between_50_and_70_l504_504076


namespace find_c_l504_504205

-- Define variables
variables (c : ℝ)

-- Define the conditions
def side_length : ℝ := 2
def total_area : ℝ := 6 * (side_length * side_length)
def half_area : ℝ := total_area / 2

-- Define the equation of the line and the area of the triangle
def triangle_area (c : ℝ) : ℝ := (1 / 2) * (6 - c) * 6

-- The proof statement
theorem find_c (h : triangle_area c = half_area) : c = 2 := sorry

end find_c_l504_504205


namespace area_of_unpainted_face_is_correct_l504_504348

-- Definition of the given conditions as a structure.
structure Cylinder where
  radius : ℝ
  height : ℝ
  painted_red : Prop
  P Q : ℝ -- Assume P and Q are positions on the circular face for simplicity.
  arc_length_PQ : ℝ
  angle_POQ : ℝ
  cut_along_plane : Prop

-- Example instance matching the given problem.
def given_cylinder : Cylinder :=
  { radius := 5,
    height := 10,
    painted_red := true,
    P := 0, -- example value, this can represent any point on the top face
    Q := 1, -- example value, this can represent any point on the top face
    arc_length_PQ := 5 * π / 2,
    angle_POQ := 90,
    cut_along_plane := true }

-- Mathematically equivalent problem statement
theorem area_of_unpainted_face_is_correct
    (cyl : Cylinder)
    (h1 : cyl.radius = 5)
    (h2 : cyl.height = 10)
    (h3 : cyl.arc_length_PQ = 5 * π / 2)
    (h4 : cyl.angle_POQ = 90)
    (h5 : cyl.painted_red)
    (h6 : cyl.cut_along_plane) :
  ∃ (d e f : ℝ), 
    f = 1 ∧ 
    d = 12.5 ∧ 
    e = 0 ∧ 
    d + e + f = 13.5 :=
by
  -- The proof goes here (not required as per instructions)
  sorry

end area_of_unpainted_face_is_correct_l504_504348


namespace count_primes_between_fifty_and_seventy_l504_504059

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_fifty_and_seventy : List ℕ :=
  [53, 59, 61, 67]

theorem count_primes_between_fifty_and_seventy :
  (primes_between_fifty_and_seventy.count is_prime = 4) :=
by
  sorry

end count_primes_between_fifty_and_seventy_l504_504059


namespace average_of_r_s_t_l504_504090

theorem average_of_r_s_t (r s t : ℝ) (h : (5/4) * (r + s + t) = 20) : (r + s + t) / 3 = 16 / 3 :=
by
  sorry

end average_of_r_s_t_l504_504090


namespace trigonometric_identity_l504_504416

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) : 
  (1 - Real.sin θ) / Real.cos θ - Real.cos θ / (1 + Real.sin θ) = 0 := 
by 
  sorry

end trigonometric_identity_l504_504416


namespace equilateral_triangle_side_length_l504_504979

theorem equilateral_triangle_side_length (t : ℝ)
  (A B C Q : Type*) [IsTriangle A B C]
  (hABC : IsEquilateralTriangle A B C t)
  (hAQ : Distance A Q = 2)
  (hBQ : Distance B Q = 1)
  (hCQ : Distance C Q = Real.sqrt 2) :
  t = Real.sqrt 6 := sorry

end equilateral_triangle_side_length_l504_504979


namespace find_N_l504_504700

theorem find_N : 
  ∀ (a b c N : ℝ), 
  a + b + c = 80 → 
  2 * a = N → 
  b - 10 = N → 
  3 * c = N → 
  N = 38 := 
by sorry

end find_N_l504_504700


namespace exists_infinitely_many_distinct_a_l504_504163

theorem exists_infinitely_many_distinct_a (n : ℕ) (h_n_pos : 0 < n) :
  ∃ (a : ℕ → ℕ), (∀ i j, i ≠ j → a i ≠ a j) ∧
  ∀ k, a k ∈ ℕ ∧ (a 1)^2 * (a 2)^2 * … * (a n)^2 - 4 * (a 1)^2 - 4 * (a 2)^2 - … - 4 * (a n)^2 IsPerfectSquare :=
sorry

end exists_infinitely_many_distinct_a_l504_504163


namespace min_trucks_required_l504_504707

theorem min_trucks_required {a b c d cap trucks: ℕ} 
  (ha : a = 4) (wa: a = 3)
  (hb : b = 5) (wb: b = 2.5)
  (hc : c = 14) (wc: c = 1.5)
  (hd : d = 7) (wd: d = 1)
  (hcp: cap = 4.5)
  (htr: trucks = 12)
  : ⌈(a * wa + b * wb + c * wc + d * wd) / cap⌉ = trucks :=
  sorry

end min_trucks_required_l504_504707


namespace coefficient_of_x_cubed_is_31_l504_504840

def expression (x : ℝ) : ℝ :=
  2 * (x^2 - 2 * x^3 + x) + 4 * (x + 3 * x^3 - 2 * x^2 + 2 * x^5 + 2 * x^3) - 3 * (2 + x - 5 * x^3 - x^2)

theorem coefficient_of_x_cubed_is_31 : coeff (simplify (expression x)) 3 = 31 :=
sorry

end coefficient_of_x_cubed_is_31_l504_504840


namespace prob_smallest_odd_l504_504277

-- Definition to encapsulate the problem in Lean
def hungarian_lottery_smallest_odd_probability : Prop :=
  let pool_size := 90
  let draw_size := 5
  let total_ways := (Finset.range (pool_size + 1)).choose(draw_size).card
  let T := ∑ k in Finset.range 43, ((Finset.range (2*k + 3)).choose(3).card)
  (1 / 2) + (44 * Finset.range 46).choose(3).card / (2 * total_ways) ≈ 0.5142

-- Theorem stating the proof problem
theorem prob_smallest_odd : hungarian_lottery_smallest_odd_probability := 
  by sorry

end prob_smallest_odd_l504_504277


namespace vector_B_not_unit_l504_504281

-- Definition of a vector and magnitude
structure Vector2D where
  x : ℝ
  y : ℝ

def magnitude (v : Vector2D) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2)

-- Vectors given in the problem
def A : Vector2D := ⟨-1, 0⟩
def B : Vector2D := ⟨1, 1⟩
def C (a : ℝ) : Vector2D := ⟨Real.cos a, Real.sin a⟩
def D (v : Vector2D) (h : (magnitude v) ≠ 0) : Vector2D := 
  let mag := magnitude v
  ⟨v.x / mag, v.y / mag⟩

-- Conditions: Magnitude of vectors A, C, and D is 1
lemma mag_A : magnitude A = 1 := by
  sorry

lemma mag_C (a : ℝ) : magnitude (C a) = 1 := by
  sorry

lemma mag_D (v : Vector2D) (h : (magnitude v) ≠ 0) : magnitude (D v h) = 1 := by
  sorry

-- Proof that the magnitude of vector B is not 1
theorem vector_B_not_unit : magnitude B ≠ 1 := by
  rw [magnitude]
  -- Calculate magnitude of B which is sqrt(1^2 + 1^2) = sqrt(2)
  have mag_B : (Real.sqrt(1^2 + 1^2)) = Real.sqrt 2 := by rfl
  rw [mag_B]
  -- sqrt(2) is not equal to 1
  exact Real.sqrt_ne 2 1 (by norm_num)

end vector_B_not_unit_l504_504281


namespace master_craftsman_total_parts_l504_504515

theorem master_craftsman_total_parts
  (N : ℕ) -- Additional parts to be produced after the first hour
  (initial_rate : ℕ := 35) -- Initial production rate (35 parts/hour)
  (increased_rate : ℕ := initial_rate + 15) -- Increased production rate (50 parts/hour)
  (time_difference : ℝ := 1.5) -- Time difference in hours between the rates
  (eq_time_diff : (N / initial_rate) - (N / increased_rate) = time_difference) -- The given time difference condition
  : 35 + N = 210 := -- Conclusion we need to prove
sorry

end master_craftsman_total_parts_l504_504515


namespace sum_arithmetic_seq_nine_terms_l504_504005

theorem sum_arithmetic_seq_nine_terms
  (a_n : ℕ → ℝ)
  (S_n : ℕ → ℝ)
  (k : ℝ)
  (h1 : ∀ n, a_n n = k * n + 4 - 5 * k)
  (h2 : ∀ n, S_n n = (n / 2) * (a_n 1 + a_n n))
  : S_n 9 = 36 :=
sorry

end sum_arithmetic_seq_nine_terms_l504_504005


namespace function_domain_function_composition_function_range_l504_504451

noncomputable def f (x : ℝ) : ℝ := 2 / (|x| - 1)

theorem function_domain : ∀ x : ℝ, x ≠ 1 ∧ x ≠ -1 ↔ (∃ y : ℝ, y = f x) := sorry

theorem function_composition : f (f (-5)) ≠ 4 := sorry

theorem function_range : ∀ y : ℝ, y ∈ set.range f ↔ (y ≤ -2 ∨ y > 0) := sorry

end function_domain_function_composition_function_range_l504_504451


namespace largest_angle_right_triangle_l504_504915

theorem largest_angle_right_triangle
  (a b c : ℝ)
  (h1 : c = 5)
  (h2 : sqrt (a - 4) + (b - 3) ^ 2 = 0) :
  ∃ angle : ℝ, angle = 90 :=
by
  sorry

end largest_angle_right_triangle_l504_504915


namespace digit_not_in_mean_l504_504369

def sequence : List Nat := [1, 22, 333, 4444, 55555, 666666]

noncomputable def arithmetic_mean (seq : List Nat) : Rat :=
  seq.foldl (+) 0 / seq.length

def contains_digit (n : Nat) (d : Char) : Bool :=
  d ∈ (n.repr.toList)

theorem digit_not_in_mean : ¬ contains_digit (arithmetic_mean sequence).floor '3' :=
by
  sorry

end digit_not_in_mean_l504_504369


namespace students_less_than_20_total_students_students_more_than_40_majority_students_more_than_20_l504_504213

def bar_chart_data : List ℕ := [90, 60, 10, 20]

theorem students_less_than_20 (data : List ℕ) (h : data = bar_chart_data) :
  List.nth data 0 = some 90 :=
by
  sorry

theorem total_students (data : List ℕ) (h : data = bar_chart_data) :
  data.sum = 180 :=
by
  sorry

theorem students_more_than_40 (data : List ℕ) (h : data = bar_chart_data) :
  (data.get! 2 + data.get! 3) = 30 :=
by
  sorry

theorem majority_students_more_than_20 (data : List ℕ) (h : data = bar_chart_data) :
  ¬ ((data.get! 1 + data.get! 2 + data.get! 3) > data.get! 0) :=
by
  sorry

end students_less_than_20_total_students_students_more_than_40_majority_students_more_than_20_l504_504213


namespace prime_square_mod_12_l504_504970

-- Definition of the sequence {a_n}
def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 2 * Nat.prime (n + 1)

-- Declaration of the main problem statement
theorem prime_square_mod_12 (p : ℕ) (a : ℕ → ℕ) (h1 : sequence a) (h2 : Prime p) (h3 : 3 < p) (h4 : ∃ n : ℕ, a n = p) : 
  (p ^ 2 + 15) % 12 = 4 :=
by
  sorry

end prime_square_mod_12_l504_504970


namespace line_mb_calculation_l504_504679

theorem line_mb_calculation :
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b) ∧ (b = -3 ∧ m = -3) ∧ (m * b = 9) :=
by
  use [-3, -3]
  split
  sorry -- use (y = mx+b) form to define the equation
  split
  { exact ⟨rfl, rfl⟩ }
  ...


end line_mb_calculation_l504_504679


namespace triangle_third_side_lengths_product_l504_504743

def hypotenuse (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

def leg (c b : ℝ) : ℝ :=
  real.sqrt (c^2 - b^2)

theorem triangle_third_side_lengths_product :
  let a := 6
  let b := 8
  let hyp := hypotenuse a b
  let leg := leg b a
  real.round (hyp * leg * 10) / 10 = 52.9 :=
by {
  -- Definitions and calculations have been provided in the problem statement
  sorry
}

end triangle_third_side_lengths_product_l504_504743


namespace pool_capacity_l504_504156

theorem pool_capacity (hose_rate leak_rate : ℝ) (fill_time : ℝ) (net_rate := hose_rate - leak_rate) (total_water := net_rate * fill_time) :
  hose_rate = 1.6 → 
  leak_rate = 0.1 → 
  fill_time = 40 → 
  total_water = 60 := by
  intros
  sorry

end pool_capacity_l504_504156


namespace n_value_l504_504569

theorem n_value (n : ℕ) (hn : 0 < n) :
  (∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x * y + z = n) →
  56 →
  n = 34 ∨ n = 35 := by sorry

end n_value_l504_504569


namespace replace_star_with_3x_l504_504177

theorem replace_star_with_3x (x : ℝ) :
  (x^4 - 3)^2 + (x^3 + 3x)^2 = x^8 + x^6 + 9x^2 + 9 :=
by
  sorry

end replace_star_with_3x_l504_504177


namespace master_craftsman_quota_l504_504529

theorem master_craftsman_quota (parts_first_hour : ℕ)
  (extra_hour_needed : ℕ)
  (increased_speed : ℕ)
  (time_diff : ℕ)
  (total_parts : ℕ) :
  parts_first_hour = 35 →
  extra_hour_needed = 1 →
  increased_speed = 15 →
  time_diff = 1.5 →
  total_parts = parts_first_hour + (175 : ℕ) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3]
  norm_num
  rw [add_comm]
  exact sorry

end master_craftsman_quota_l504_504529


namespace estimated_high_quality_probability_l504_504882

theorem estimated_high_quality_probability :
  let frequencies := [0.93, 0.96, 0.95, 0.935, 0.938, 0.942, 0.939] in
  real.round ((list.sum frequencies) / (list.length frequencies) * 100) / 100 = 0.94 :=
by sorry

end estimated_high_quality_probability_l504_504882


namespace convex_polygon_acute_angles_l504_504467

-- Define a polygon
structure Polygon (n : ℕ) :=
  (vertices : Fin n → ℝ × ℝ)
  (convex : ∀ i j k : Fin n, i ≠ j → j ≠ k → k ≠ i → ∠ (vertices i) (vertices j) (vertices k) < 180)

-- Define opposite sides being pairwise parallel
def pairwise_parallel (P : Polygon n) : Prop :=
  ∃ (m : ℕ), 2 * m = n ∧
  ∀ i : Fin m, ∃ a b : ℝ × ℝ, a ≠ b ∧
  (P.vertices (Fin.cast (by rw [Nat.mod_add_mod])) = a ∧
   P.vertices (Fin.cast (by rw [Nat.mod_add_mod])) = a) ∧
  (P.vertices (Fin.cast (by rw [Nat.mod_add_mod])) = b ∧
   P.vertices (Fin.cast (by rw [Nat.mod_add_mod])) = b)

-- Main theorem: A convex polygon with opposite sides pairwise parallel has at most two acute angles
theorem convex_polygon_acute_angles (P : Polygon n) (h : pairwise_parallel P) : 
  ∃ k : ℕ, k = 0 ∨ k = 2 :=
sorry

end convex_polygon_acute_angles_l504_504467


namespace range_absolute_difference_l504_504848

theorem range_absolute_difference : ∀ y, y = |x + 5| - |x - 3| → y ∈ set.Icc (-8) 8 :=
by
  sorry

end range_absolute_difference_l504_504848


namespace exist_circle_passing_through_intersection_l504_504462

-- Definitions: Lines l1 and l2, a point A, and an angle alpha
variable {ℝ : Type*} [LinearOrderedField ℝ]
variables {l1 l2 : ℝ → ℝ} (A : ℝ × ℝ) (α : ℝ)

-- Theorem: Existence of a circle centered at A such that l1 and l2 intercept an arc of angle α
theorem exist_circle_passing_through_intersection 
  (h_non_parallel : ¬(l1 = l2)) (h_point_A : (0 : ℝ) ≤ A.1 ∧ (0 : ℝ) ≤ A.2)
  (h_angle_size : 0 < α ∧ α < (2 * Real.pi)) :
  ∃ (r : ℝ), ∃ (C : Metric.sphere A r), 
  ∃ (M : ℝ × ℝ), M ∈ C ∧ ∃ (M' : ℝ × ℝ), M' ∈ C ∧
  ∃ (P1 P2 : ℝ × ℝ), 
  P1 = Metric.project l1 M' ∧ P2 = Metric.project l2 M ∧
  ∠ P1 A P2 = α :=
sorry

end exist_circle_passing_through_intersection_l504_504462


namespace range_of_m_l504_504914

noncomputable def p (x : ℝ) : Prop := x^2 - 2 * x - 15 ≤ 0
noncomputable def q (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x - m^2 + 1 ≤ 0

theorem range_of_m (m : ℝ) : 
  (¬∃ x : ℝ, p x) -> ¬∃ x : ℝ, q x m → (m <-4 ∨ m > 4) :=
by
  sorry

end range_of_m_l504_504914


namespace xiao_ming_total_score_l504_504766

-- Definitions for the given conditions
def score_regular : ℝ := 70
def score_midterm : ℝ := 80
def score_final : ℝ := 85

def weight_regular : ℝ := 0.3
def weight_midterm : ℝ := 0.3
def weight_final : ℝ := 0.4

-- The statement that we need to prove
theorem xiao_ming_total_score : 
  (score_regular * weight_regular) + (score_midterm * weight_midterm) + (score_final * weight_final) = 79 := 
by
  sorry

end xiao_ming_total_score_l504_504766


namespace radius_of_circumcircle_l504_504302

-- Definitions of sides of a triangle and its area
variables {a b c t : ℝ}

-- Condition that t is the area of a triangle with sides a, b, and c
def is_triangle_area (a b c t : ℝ) : Prop := -- Placeholder condition stating these values form a triangle
sorry

-- Statement to prove the given radius formula for the circumscribed circle
theorem radius_of_circumcircle (h : is_triangle_area a b c t) : 
  ∃ r : ℝ, r = abc / (4 * t) :=
sorry

end radius_of_circumcircle_l504_504302


namespace simplify_sqrt_expression_is_correct_l504_504627

-- Definition for the given problem
def simplify_sqrt_expression (a b : ℝ) :=
  a = Real.sqrt (12 + 8 * Real.sqrt 3) → 
  b = Real.sqrt (12 - 8 * Real.sqrt 3) → 
  a + b = 4 * Real.sqrt 3

-- The theorem to be proven
theorem simplify_sqrt_expression_is_correct : simplify_sqrt_expression :=
begin
  intros a b ha hb,
  rw [ha, hb],
  -- Step-by-step simplification approach would occur here
  sorry
end

end simplify_sqrt_expression_is_correct_l504_504627


namespace smallest_k_for_quadratic_l504_504972

noncomputable def quadratic_has_two_distinct_real_roots (k : ℝ) : Prop :=
  let a := k in
  let b := -3 in
  let c := -9 / 4 in
  b^2 - 4*a*c > 0

theorem smallest_k_for_quadratic : 
  ∃ k : ℤ, quadratic_has_two_distinct_real_roots k ∧ k > -1 ∧ k ≠ 0 ∧ ∀ m : ℤ, quadratic_has_two_distinct_real_roots m → m > -1 → m ≠ 0 → k ≤ m :=
sorry

end smallest_k_for_quadratic_l504_504972


namespace reciprocal_of_3_div_2_l504_504755

def reciprocal (a : ℚ) : ℚ := a⁻¹

theorem reciprocal_of_3_div_2 : reciprocal (3 / 2) = 2 / 3 :=
by
  -- proof would go here
  sorry

end reciprocal_of_3_div_2_l504_504755


namespace farmer_apples_count_l504_504233

theorem farmer_apples_count (initial : ℕ) (given : ℕ) (remaining : ℕ) 
  (h1 : initial = 127) (h2 : given = 88) : remaining = initial - given := 
by
  sorry

end farmer_apples_count_l504_504233


namespace smaller_solution_of_quadratic_l504_504881

theorem smaller_solution_of_quadratic :
  let a := 1
  let b := -15
  let c := -56
  (∀ (x : ℝ), x^2 - 15 * x - 56 = 0 → x = (15 + Real.sqrt 449) / 2 ∨ x = (15 - Real.sqrt 449) / 2) →
  ∃ x : ℝ, x = (15 - Real.sqrt 449) / 2 ∧ (∀ y : ℝ, y = (15 + Real.sqrt 449) / 2 → x < y) :=
by
  sorry

end smaller_solution_of_quadratic_l504_504881


namespace simplify_nested_sqrt_l504_504642

-- Define the expressions under the square roots
def expr1 : ℝ := 12 + 8 * real.sqrt 3
def expr2 : ℝ := 12 - 8 * real.sqrt 3

-- Problem statement to prove
theorem simplify_nested_sqrt : real.sqrt expr1 + real.sqrt expr2 = 4 * real.sqrt 2 :=
by
  sorry

end simplify_nested_sqrt_l504_504642


namespace exterior_angle_regular_octagon_l504_504993

theorem exterior_angle_regular_octagon : ∀ (n : ℕ), n = 8 → (180 - (1080 / n)) = 45 :=
by
  intros n h
  rw h
  sorry

end exterior_angle_regular_octagon_l504_504993


namespace distinct_possible_results_l504_504701

def jia_jia_numbers := {3, 6, 7}
def fang_fang_numbers := {4, 5, 6}
def ming_ming_numbers := {4, 5, 8}

def initial_number_jia_jia := 234
def initial_number_fang_fang := 235
def initial_number_ming_ming := 236

def possible_calculations (nums : Set ℕ) (initial : ℕ) : Set ℕ :=
  {prod + initial | prod in {a * b | a in nums, b in nums, a ≠ b}}

def all_possible_calculations : Set ℕ :=
  (possible_calculations jia_jia_numbers initial_number_jia_jia) ∪ 
  (possible_calculations fang_fang_numbers initial_number_fang_fang) ∪ 
  (possible_calculations ming_ming_numbers initial_number_ming_ming)

theorem distinct_possible_results : all_possible_calculations.card = 7 := 
sorry

end distinct_possible_results_l504_504701


namespace find_integer_l504_504969

theorem find_integer
  (x y : ℤ)
  (h1 : 4 * x + y = 34)
  (h2 : 2 * x - y = 20)
  (h3 : y^2 = 4) :
  y = -2 :=
by
  sorry

end find_integer_l504_504969


namespace substitute_monomial_to_simplify_expr_l504_504184

theorem substitute_monomial_to_simplify_expr (k : ℤ) : 
  ( ∃ k : ℤ, (x^4 - 3)^2 + (x^3 + k * x)^2 after expanding has exactly four terms) := 
begin
  use 3,
  sorry
end

end substitute_monomial_to_simplify_expr_l504_504184


namespace distance_from_circumcenter_to_orthocenter_l504_504112

variables {A B C A1 H O : Type}

-- Condition Definitions
variable (acute_triangle : Prop)
variable (is_altitude : Prop)
variable (is_orthocenter : Prop)
variable (AH_dist : ℝ := 3)
variable (A1H_dist : ℝ := 2)
variable (circum_radius : ℝ := 4)

-- Prove the distance from O to H
theorem distance_from_circumcenter_to_orthocenter
  (h1 : acute_triangle)
  (h2 : is_altitude)
  (h3 : is_orthocenter)
  (h4 : AH_dist = 3)
  (h5 : A1H_dist = 2)
  (h6 : circum_radius = 4) : 
  ∃ (d : ℝ), d = 2 := 
sorry

end distance_from_circumcenter_to_orthocenter_l504_504112


namespace average_age_of_population_l504_504987

theorem average_age_of_population
  (k : ℕ)
  (ratio_women_men : 7 * (k : ℕ) = 7 * (k : ℕ) + 5 * (k : ℕ) - 5 * (k : ℕ))
  (avg_age_women : ℝ := 38)
  (avg_age_men : ℝ := 36)
  : ( (7 * k * avg_age_women) + (5 * k * avg_age_men) ) / (12 * k) = 37 + (1 / 6) :=
by
  sorry

end average_age_of_population_l504_504987


namespace sum_distances_proof_l504_504888

theorem sum_distances_proof (r_A r_B r_C r_D : ℝ) (A B C D R P Q : ℝ×ℝ) 
  (h_r_A : r_A = (3/4) * r_B) (h_r_C : r_C = (3/4) * r_D)
  (h_AB : dist A B = 40) (h_CD : dist C D = 50)
  (h_PQ : dist P Q = 52) (h_mid_R : R = midpoint P Q) : 
  dist A R + dist B R + dist C R + dist D R = 126.2 := 
by 
  sorry

end sum_distances_proof_l504_504888


namespace Simson_line_through_Feuerbach_center_l504_504026

theorem Simson_line_through_Feuerbach_center
  (A B C : Point)
  (circumcircle : Circle)
  (D E : Point)
  (K : Point)
  (Feuerbach_center : Point)
  (h1 : Triangle A B C) 
  (h2 : PerpendicularBisector A B ∩ circumcircle = {D, E})
  (h3 : D ∈ Arc circumcircle A C B)
  (h4 : K ∈ DivideArc circumcircle D C 3)
  (h5 : Closer K D K E)
  (Simson_line : Line) :
  SimsonLine K (Triangle A B C) = Simson_line →
  PassesThrough Simson_line Feuerbach_center :=
begin
  sorry
end

end Simson_line_through_Feuerbach_center_l504_504026


namespace BP_value_l504_504311

theorem BP_value 
  (A B C D P : Point)
  (h_circle : ∀ (X Y : Point), X ∈ circle(A B D) → Y ∈ circle(A B D) → X = Y)
  (h_AP : distance(A, P) = 12)
  (h_PC : distance(P, C) = 3)
  (h_BD : distance(B, D) = 10)
  (h_BP_lt_DP : distance(B, P) < distance(P, D)) : 
  distance(B, P) = 4 :=
begin
  sorry
end

end BP_value_l504_504311


namespace master_craftsman_parts_l504_504523

/-- 
Given:
  (1) the master craftsman produces 35 parts in the first hour,
  (2) at the rate of 35 parts/hr, he would be one hour late to meet the quota,
  (3) by increasing his speed by 15 parts/hr, he finishes the quota 0.5 hours early,
Prove that the total number of parts manufactured during the shift is 210.
-/
theorem master_craftsman_parts (N : ℕ) (quota : ℕ) 
  (initial_rate : ℕ := 35)
  (increased_rate_diff : ℕ := 15)
  (extra_time_slow : ℕ := 1)
  (time_saved_fast : ℕ := 1/2) :
  (quota = initial_rate * (extra_time_slow + 1) + N ∧
   increased_rate_diff = 15 ∧
   increased_rate_diff = λ (x : ℕ), initial_rate + x ∧
   time_saved_fast = 1/2 ∧
   N = 35) →
  quota = 210 := 
by
  sorry

end master_craftsman_parts_l504_504523


namespace cyclic_IQPR_l504_504559

theorem cyclic_IQPR 
  (ABC : Triangle)
  (hAB_AC : ABC.AB < ABC.AC)
  (I : Point)
  (hI_incenter : incenter I ABC)
  (Gamma : Circle)
  (hGamma : circumcircle_of (Triangle.mk I ABC.B ABC.C) Gamma)
  (P : Point)
  (hP_on_AIGamma : AI_intersect_again_on P ABC.A I Gamma)
  (Q : Point)
  (hQ_on_AC : on_side Q ABC.AC)
  (hAB_EQ_AQ : ABC.AB = AQ)
  (R : Point)
  (hR_on_AB : on_side R ABC.AB)
  (hB_between_AR : between B ABC.A R)
  (hAR_EQ_AC : AR = ABC.AC)
: cyclic I Q P R := sorry

end cyclic_IQPR_l504_504559


namespace prime_count_between_50_and_70_l504_504037

theorem prime_count_between_50_and_70 : 
  (finset.filter nat.prime (finset.range 71).filter (λ n, 50 < n ∧ n < 71)).card = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504037


namespace unit_digit_of_product_of_nine_consecutive_numbers_is_zero_l504_504865

theorem unit_digit_of_product_of_nine_consecutive_numbers_is_zero (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6) * (n + 7) * (n + 8)) % 10 = 0 :=
by
  sorry

end unit_digit_of_product_of_nine_consecutive_numbers_is_zero_l504_504865


namespace compute_f_seven_halves_l504_504146

theorem compute_f_seven_halves 
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_shift : ∀ x, f (x + 2) = -f x)
  (h_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x) :
  f (7 / 2) = -1 / 2 :=
  sorry

end compute_f_seven_halves_l504_504146


namespace prime_count_between_50_and_70_l504_504046

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def primes_in_range (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem prime_count_between_50_and_70 : primes_in_range 50 70 = [53, 59, 61, 67] :=
sorry

end prime_count_between_50_and_70_l504_504046


namespace master_craftsman_total_parts_l504_504516

theorem master_craftsman_total_parts
  (N : ℕ) -- Additional parts to be produced after the first hour
  (initial_rate : ℕ := 35) -- Initial production rate (35 parts/hour)
  (increased_rate : ℕ := initial_rate + 15) -- Increased production rate (50 parts/hour)
  (time_difference : ℝ := 1.5) -- Time difference in hours between the rates
  (eq_time_diff : (N / initial_rate) - (N / increased_rate) = time_difference) -- The given time difference condition
  : 35 + N = 210 := -- Conclusion we need to prove
sorry

end master_craftsman_total_parts_l504_504516


namespace abs_sub_abs_l504_504757

theorem abs_sub_abs : |(13 - 3)| - |(4 - 10)| = 4 := by
  sorry

end abs_sub_abs_l504_504757


namespace sum_f_mod_1000_l504_504886

def f (n : ℕ) : ℕ :=
  if n % 2 = 0 then n - 1 else n

theorem sum_f_mod_1000 : ∑ n in Finset.range 2022 | (f (n + 1)) % 1000 = 242 :=
by
  sorry

end sum_f_mod_1000_l504_504886


namespace max_non_managers_l504_504985

theorem max_non_managers (N : ℕ) (h : (9:ℝ) / (N:ℝ) > (7:ℝ) / (32:ℝ)) : N ≤ 41 :=
by
  -- Proof skipped
  sorry

end max_non_managers_l504_504985


namespace intersection_complement_eq_l504_504459

def A : Set ℝ := { x | 1 ≤ x ∧ x < 3 }

def B : Set ℝ := { x | x^2 ≥ 4 }

def complementB : Set ℝ := { x | -2 < x ∧ x < 2 }

def intersection (A : Set ℝ) (B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∈ B }

theorem intersection_complement_eq : 
  intersection A complementB = { x | 1 ≤ x ∧ x < 2 } := 
sorry

end intersection_complement_eq_l504_504459


namespace two_xy_value_l504_504749

theorem two_xy_value (x y : ℝ) 
  (h1 : (8^x) / (2^(x+y)) = 64)
  (h2 : (9^(x+y)) / (3^(4*y)) = 243) : 
  2 * x * y = 7 :=
by
  sorry

end two_xy_value_l504_504749


namespace range_of_a_l504_504483

theorem range_of_a (a : ℝ) : (0, 0).1 + (0, 0).2 - a > 0 → (1, 1).1 + (1, 1).2 - a < 0 ↔ 0 < a ∧ a < 2 := by
  intro h₀ h₁
  split
  { intros
    sorry }
  { intros
    sorry }

end range_of_a_l504_504483


namespace count_primes_between_fifty_and_seventy_l504_504065

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_fifty_and_seventy : List ℕ :=
  [53, 59, 61, 67]

theorem count_primes_between_fifty_and_seventy :
  (primes_between_fifty_and_seventy.count is_prime = 4) :=
by
  sorry

end count_primes_between_fifty_and_seventy_l504_504065


namespace sum_f_alpha_beta_gamma_neg_l504_504453

theorem sum_f_alpha_beta_gamma_neg (f : ℝ → ℝ)
  (h_f : ∀ x, f x = -x - x^3)
  (α β γ : ℝ)
  (h1 : α + β > 0)
  (h2 : β + γ > 0)
  (h3 : γ + α > 0) :
  f α + f β + f γ < 0 := 
sorry

end sum_f_alpha_beta_gamma_neg_l504_504453


namespace total_time_proof_l504_504595

variable (mow_time : ℕ) (fertilize_time : ℕ) (total_time : ℕ)

-- Based on the problem conditions.
axiom mow_time_def : mow_time = 40
axiom fertilize_time_def : fertilize_time = 2 * mow_time
axiom total_time_def : total_time = mow_time + fertilize_time

-- The proof goal
theorem total_time_proof : total_time = 120 := by
  sorry

end total_time_proof_l504_504595


namespace probability_of_choosing_A_on_second_day_l504_504363

-- Definitions of the probabilities given in the problem conditions.
def p_first_day_A := 0.5
def p_first_day_B := 0.5
def p_second_day_A_given_first_day_A := 0.6
def p_second_day_A_given_first_day_B := 0.5

-- Define the problem to be proved in Lean 4
theorem probability_of_choosing_A_on_second_day :
  (p_first_day_A * p_second_day_A_given_first_day_A) +
  (p_first_day_B * p_second_day_A_given_first_day_B) = 0.55 :=
by
  sorry

end probability_of_choosing_A_on_second_day_l504_504363


namespace smallest_positive_integer_l504_504278

theorem smallest_positive_integer (n : ℕ) : 3 * n ≡ 568 [MOD 34] → n = 18 := 
sorry

end smallest_positive_integer_l504_504278


namespace intersection_count_l504_504381

-- Define the polar equations as conditions
def r1 (θ : ℝ) : ℝ := 3 * Real.cos θ
def r2 (θ : ℝ) : ℝ := 6 * Real.sin θ

-- Define the task of determining the number of intersection points
theorem intersection_count : ∃ θ1 θ2 : ℝ, (r1 θ1 = r2 θ1) ∧ (r1 θ2 = r2 θ2) ∧ θ1 ≠ θ2 ∧ 
                           (∀ θ : ℝ, (r1 θ = r2 θ) → (θ = θ1 ∨ θ = θ2)) :=
by
  sorry

end intersection_count_l504_504381


namespace ants_species_c_count_l504_504362

noncomputable def ants : ℕ × ℕ × ℕ := sorry

variable (a b c : ℕ)
variable h1 : a + b + c = 50
variable h2 : 16 * a + 81 * b + 256 * c = 6561

theorem ants_species_c_count : c = 22 := 
by
  obtain ⟨a, b, c⟩ := ants,
  exact sorry

end ants_species_c_count_l504_504362


namespace find_m_value_l504_504684

theorem find_m_value (m n : ℝ) (h₀ : m ≠ 0) (h₁ : m ≠ 1) :
  (∀ n : ℝ, ∃ A, A = (λ (x : ℝ), (x, x + n)) ∧ A.2 = (x, m * x + 3 * n) ∧ (A - line_y_eq_3_over_4x_sub_3).constant_distance) → 
  m = 3 / 2 :=
by
  sorry

end find_m_value_l504_504684


namespace betty_harvest_l504_504839

/-- 
Given the following conditions for Betty's vegetable harvest, 
- Boxes for parsnips hold 20 units, with 5/8 of the boxes full and 3/8 half-full, averaging 18 boxes per harvest.
- Boxes for carrots hold 25 units, with 7/12 of the boxes full and 5/12 half-full, averaging 12 boxes per harvest.
- Boxes for potatoes hold 30 units, with 3/5 of the boxes full and 2/5 half-full, averaging 15 boxes per harvest.
Prove Betty's average harvest yields:
- 290 parsnips
- 237.5 carrots
- 360 potatoes
--/

theorem betty_harvest :
  let full_boxes (capacity : ℕ) (ratio : ℚ) (total_boxes : ℚ) : ℚ :=
    (ratio * total_boxes) * capacity

  let half_boxes (capacity : ℕ) (ratio : ℚ) (total_boxes : ℚ) : ℚ :=
    (ratio * total_boxes) * (capacity / 2)

  full_boxes 20 (5/8) 18 + half_boxes 20 (3/8) 18 = 290 ∧
  full_boxes 25 (7/12) 12 + half_boxes 25 (5/12) 12 = 237.5 ∧
  full_boxes 30 (3/5) 15 + half_boxes 30 (2/5) 15 = 360 :=
by 
  sorry

end betty_harvest_l504_504839


namespace molecular_weight_of_compound_l504_504753

-- Definitions of the atomic weights.
def atomic_weight_K : ℝ := 39.10
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

-- Proof statement of the molecular weight of the compound.
theorem molecular_weight_of_compound :
  (1 * atomic_weight_K) + (1 * atomic_weight_Br) + (3 * atomic_weight_O) = 167.00 :=
  by
    sorry

end molecular_weight_of_compound_l504_504753


namespace wire_length_l504_504353

theorem wire_length (S L : ℝ) (h1 : S = 10) (h2 : S = (2 / 5) * L) : S + L = 35 :=
by
  sorry

end wire_length_l504_504353


namespace paint_calculation_l504_504158

theorem paint_calculation (x : ℕ) 
  (h1 : ∀ mary mike sun, mary = x → mike = x + 2 → sun = 5 → mary + mike + sun = 13) :
  x = 3 :=
by
  -- We assume conditions mary, mike, and sun to satisfy the theorem h1
  have h2 : x + (x + 2) + 5 = 13 := h1 x (x + 2) 5 rfl rfl rfl
  -- Now we simplify the equation
  have t : 2 * x + 7 = 13 := by linarith
  -- We isolate x
  have t2 : 2 * x = 6 := by linarith
  -- We solve for x
  exact t2 ▸ Nat.div_self_two

end paint_calculation_l504_504158


namespace prime_count_between_50_and_70_l504_504042

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def primes_in_range (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem prime_count_between_50_and_70 : primes_in_range 50 70 = [53, 59, 61, 67] :=
sorry

end prime_count_between_50_and_70_l504_504042


namespace train_average_speed_l504_504870

theorem train_average_speed (speed : ℕ) (stop_time : ℕ) (running_time : ℕ) (total_time : ℕ)
  (h1 : speed = 60)
  (h2 : stop_time = 24)
  (h3 : running_time = total_time - stop_time)
  (h4 : running_time = 36)
  (h5 : total_time = 60) :
  (speed * running_time / total_time = 36) :=
by {
  -- Sorry is used here to skip the proof
  sorry
}

end train_average_speed_l504_504870


namespace number_of_valid_subsets_l504_504957

def setA : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def oddSet : Finset ℕ := {1, 3, 5, 7}
def evenSet : Finset ℕ := {2, 4, 6}

theorem number_of_valid_subsets : 
  (oddSet.powerset.card * (evenSet.powerset.card - 1) - oddSet.powerset.card) = 96 :=
by sorry

end number_of_valid_subsets_l504_504957


namespace find_z_l504_504463

open Matrix

noncomputable def a : Matrix (Fin 3) (Fin 1) ℝ := ![![2], ![3], ![-1]]
noncomputable def b : Matrix (Fin 3) (Fin 1) ℝ := ![![1], ![1], ![0]]
noncomputable def c : Matrix (Fin 3) (Fin 1) ℝ := ![![3], ![0], ![-3]]

theorem find_z (x y z : ℝ)
  (h : c = x • a + y • b + z • (crossProduct a b)) : z = 0 :=
sorry

def crossProduct (u v : Matrix (Fin 3) (Fin 1) ℝ) : Matrix (Fin 3) (Fin 1) ℝ :=
  ![![u 2 0 * v 1 0 - u 1 0 * v 2 0], 
    ![u 0 0 * v 2 0 - u 2 0 * v 0 0],
    ![u 1 0 * v 0 0 - u 0 0 * v 1 0]]

end find_z_l504_504463


namespace g3_squared_eq_27_l504_504209

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
axiom fgx_eq_x3 : ∀ x : ℝ, x ≥ 1 → f(g(x)) = x^3
axiom gfx_eq_x2 : ∀ x : ℝ, x ≥ 1 → g(f(x)) = x^2
axiom g27_eq_27 : g(27) = 27

theorem g3_squared_eq_27 : [g(3)]^2 = 27 := by
  sorry

end g3_squared_eq_27_l504_504209


namespace total_songs_l504_504412

open Nat

/-- Define the overall context and setup for the problem --/
def girls : List String := ["Mary", "Alina", "Tina", "Hanna"]

def hanna_songs : ℕ := 7
def mary_songs : ℕ := 4

def alina_songs (a : ℕ) : Prop := a > mary_songs ∧ a < hanna_songs
def tina_songs (t : ℕ) : Prop := t > mary_songs ∧ t < hanna_songs

theorem total_songs (a t : ℕ) (h_alina : alina_songs a) (h_tina : tina_songs t) : 
  (11 + a + t) % 3 = 0 → (7 + 4 + a + t) / 3 = 7 := by
  sorry

end total_songs_l504_504412


namespace plotted_points_parabola_l504_504409

theorem plotted_points_parabola (t : ℝ) : 
  ∃ a b c : ℝ, ∀ t : ℝ, let x := 3^t - 4 in let y := 9^t - 7 * 3^t + 2 in y = a * x^2 + b * x + c := 
begin
  use [1, 1, -10],  -- These are the coefficients of the parabola from the solution
  intro t,
  let x := 3^t - 4,
  let y := 9^t - 7 * 3^t + 2,
  calc 
    y = (3^t)^2 - 7 * 3^t + 2 : by rw pow_two (3^t)
    ... = (x + 4)^2 - 7 * (x + 4) + 2 : by rw [show 3^t = x + 4, by sorry]
    ... = x^2 + 8x + 16 - 7x - 28 + 2 : by sorry
    ... = x^2 + x - 10 : by sorry
end

end plotted_points_parabola_l504_504409


namespace symmetric_circle_eq_l504_504095

theorem symmetric_circle_eq :
  (∀ {x y : ℝ}, (x + 2)^2 + (y - 1)^2 = 1 ↔ (x - 2)^2 + (y + 1)^2 = 1) → 
  ∀ {x y : ℝ}, circle_symmetric_to_origin (x+2)^2 (y-1)^2 = (x-2)^2 + (y+1)^2 :=
by
  sorry

-- Definitions to set up the Lean environment for the problem.
def circle_eq (h k r : ℝ) : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | (p.1 - h)^2 + (p.2 - k)^2 = r^2}

def circle_symmetric_to_origin (h k : ℝ) : set (ℝ × ℝ) := 
  {p : ℝ × ℝ | (p.1 + h)^2 + (p.2 - k)^2 = 1}

-- Example of how circle_symmetric_to_origin would work in practice
example : circle_symmetric_to_origin (-2) 1 ={p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 1} :=
by
  sorry

end symmetric_circle_eq_l504_504095


namespace necessary_but_not_sufficient_l504_504307

theorem necessary_but_not_sufficient (a b : ℝ) : (a > b) → (a + 1 > b - 2) :=
by sorry

end necessary_but_not_sufficient_l504_504307


namespace simplify_nested_sqrt_l504_504644

-- Define the expressions under the square roots
def expr1 : ℝ := 12 + 8 * real.sqrt 3
def expr2 : ℝ := 12 - 8 * real.sqrt 3

-- Problem statement to prove
theorem simplify_nested_sqrt : real.sqrt expr1 + real.sqrt expr2 = 4 * real.sqrt 2 :=
by
  sorry

end simplify_nested_sqrt_l504_504644


namespace average_of_all_digits_l504_504710

theorem average_of_all_digits (d : List ℕ) (h_len : d.length = 9)
  (h1 : (d.take 4).sum = 32)
  (h2 : (d.drop 4).sum = 130) : 
  (d.sum / d.length : ℚ) = 18 := 
by
  sorry

end average_of_all_digits_l504_504710


namespace replace_asterisk_with_monomial_l504_504193

theorem replace_asterisk_with_monomial :
  ∀ (x : ℝ), (x^4 - 3)^2 + (x^3 + 3x)^2 = x^8 + x^6 + 9x^2 + 9 := 
by
  intro x
  calc
    (x^4 - 3)^2 + (x^3 + 3x)^2
        = (x^4)^2 - 2 * x^4 * 3 + 3^2 + (x^3)^2 + 2 * x^3 * 3x + (3x)^2 : by ring
    ... = x^8 - 6 * x^4 + 9 + x^6 + 6 * x^4 + 9 * x^2 : by ring
    ... = x^8 + x^6 + 9 * x^2 + 9 : by ring
  sorry

end replace_asterisk_with_monomial_l504_504193


namespace cross_product_of_vectors_l504_504397

theorem cross_product_of_vectors :
  let v := ![0, -2, 4]
  let w := ![-1, 0, 6]
  v × w = ![-12, -4, 0] := by
  sorry

end cross_product_of_vectors_l504_504397


namespace tv_weight_difference_l504_504364

noncomputable def BillTV_width : ℝ := 48
noncomputable def BillTV_height : ℝ := 100
noncomputable def BobTV_width : ℝ := 70
noncomputable def BobTV_height : ℝ := 60
noncomputable def weight_per_sq_inch : ℝ := 4
noncomputable def ounces_per_pound : ℝ := 16

theorem tv_weight_difference :
  let BillTV_area := BillTV_width * BillTV_height,
      BillTV_weight_oz := BillTV_area * weight_per_sq_inch,
      BillTV_weight_lb := BillTV_weight_oz / ounces_per_pound,
      BobTV_area := BobTV_width * BobTV_height,
      BobTV_weight_oz := BobTV_area * weight_per_sq_inch,
      BobTV_weight_lb := BobTV_weight_oz / ounces_per_pound
  in BillTV_weight_lb - BobTV_weight_lb = 150 :=
by
  sorry

end tv_weight_difference_l504_504364


namespace trajectory_of_point_C_l504_504439

noncomputable def ellipse_trajectory (C : ℝ × ℝ) : Prop :=
  (C.1, C.2) = (7, 0) ∨ (C.1, C.2) = (-7, 0)

theorem trajectory_of_point_C :
  ∀ (C : ℝ × ℝ), C ∉ set_of ellipse_trajectory -> 
    (x : ℝ), (y : ℝ), 
      C = (x, y) -> 
      (x^2 / 49) + (y^2 / 13) = 1 :=
by
  sorry

end trajectory_of_point_C_l504_504439


namespace sum_x_y_l504_504102

theorem sum_x_y :
  let x := (finset.range (70 - 50 + 1)).sum (λ i, 50 + i)
      y := (finset.range (70 - 50 + 1)).filter (λ i, (50 + i) % 2 = 0).card
  in x + y = 1271 :=
by
  let x := (finset.range (70 - 50 + 1)).sum (λ i, 50 + i)
  let y := (finset.range (70 - 50 + 1)).filter (λ i, (50 + i) % 2 = 0).card
  have h_x : x = 1260 := sorry -- Detailed proof of x
  have h_y : y = 11 := sorry -- Detailed proof of y
  rw [h_x, h_y]
  norm_num

end sum_x_y_l504_504102


namespace farmhands_work_hours_l504_504028

def apples_per_pint (variety: String) : ℕ :=
  match variety with
  | "golden_delicious" => 20
  | "pink_lady" => 40
  | _ => 0

def total_apples_for_pints (pints: ℕ) : ℕ :=
  (apples_per_pint "golden_delicious") * pints + (apples_per_pint "pink_lady") * pints

def apples_picked_per_hour_per_farmhand : ℕ := 240

def num_farmhands : ℕ := 6

def total_apples_picked_per_hour : ℕ :=
  num_farmhands * apples_picked_per_hour_per_farmhand

def ratio_golden_to_pink : ℕ × ℕ := (1, 2)

def haley_cider_pints : ℕ := 120

def hours_worked (pints: ℕ) (picked_per_hour: ℕ): ℕ :=
  (total_apples_for_pints pints) / picked_per_hour

theorem farmhands_work_hours :
  hours_worked haley_cider_pints total_apples_picked_per_hour = 5 := by
  sorry

end farmhands_work_hours_l504_504028


namespace extreme_points_l504_504936

noncomputable def f (a x : ℝ) : ℝ := Real.log (1 / (2 * x)) - a * x^2 + x

theorem extreme_points (
  a : ℝ
) (h : 0 < a ∧ a < (1 : ℝ) / 8) :
  ∃ x1 x2 : ℝ, f a x1 + f a x2 > 3 - 4 * Real.log 2 :=
sorry

end extreme_points_l504_504936


namespace master_craftsman_total_parts_l504_504514

theorem master_craftsman_total_parts
  (N : ℕ) -- Additional parts to be produced after the first hour
  (initial_rate : ℕ := 35) -- Initial production rate (35 parts/hour)
  (increased_rate : ℕ := initial_rate + 15) -- Increased production rate (50 parts/hour)
  (time_difference : ℝ := 1.5) -- Time difference in hours between the rates
  (eq_time_diff : (N / initial_rate) - (N / increased_rate) = time_difference) -- The given time difference condition
  : 35 + N = 210 := -- Conclusion we need to prove
sorry

end master_craftsman_total_parts_l504_504514


namespace abs_sum_bound_l504_504477

theorem abs_sum_bound (x : ℝ) (a : ℝ) (h : |x - 4| + |x - 3| < a) (ha : 0 < a) : 1 < a :=
by
  sorry

end abs_sum_bound_l504_504477


namespace zero_is_a_root_of_polynomial_l504_504879

theorem zero_is_a_root_of_polynomial :
  (12 * (0 : ℝ)^4 + 38 * (0)^3 - 51 * (0)^2 + 40 * (0) = 0) :=
by simp

end zero_is_a_root_of_polynomial_l504_504879


namespace contrapositive_proposition_l504_504223

theorem contrapositive_proposition (x y : ℝ) :
  (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
sorry

end contrapositive_proposition_l504_504223


namespace simplify_sqrt_sum_l504_504632

theorem simplify_sqrt_sum :
  sqrt (12 + 8 * sqrt 3) + sqrt (12 - 8 * sqrt 3) = 4 * sqrt 3 := 
by
  -- Proof would go here
  sorry

end simplify_sqrt_sum_l504_504632


namespace quotient_remainder_scaled_l504_504164

theorem quotient_remainder_scaled (a b q r k : ℤ) (hb : b > 0) (hk : k ≠ 0) (h1 : a = b * q + r) (h2 : 0 ≤ r) (h3 : r < b) :
  a * k = (b * k) * q + (r * k) ∧ (k ∣ r → (a / k = (b / k) * q + (r / k) ∧ 0 ≤ (r / k) ∧ (r / k) < (b / k))) :=
by
  sorry

end quotient_remainder_scaled_l504_504164


namespace master_craftsman_quota_l504_504526

theorem master_craftsman_quota (parts_first_hour : ℕ)
  (extra_hour_needed : ℕ)
  (increased_speed : ℕ)
  (time_diff : ℕ)
  (total_parts : ℕ) :
  parts_first_hour = 35 →
  extra_hour_needed = 1 →
  increased_speed = 15 →
  time_diff = 1.5 →
  total_parts = parts_first_hour + (175 : ℕ) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3]
  norm_num
  rw [add_comm]
  exact sorry

end master_craftsman_quota_l504_504526


namespace darma_eats_peanuts_l504_504856

/--
Darma can eat 20 peanuts in 15 seconds. Prove that she can eat 480 peanuts in 6 minutes.
-/
theorem darma_eats_peanuts (rate: ℕ) (per_seconds: ℕ) (minutes: ℕ) (conversion: ℕ) : 
  (rate = 20) → (per_seconds = 15) → (minutes = 6) → (conversion = 60) → 
  rate * (conversion * minutes / per_seconds) = 480 :=
by
  intros hrate hseconds hminutes hconversion
  rw [hrate, hseconds, hminutes, hconversion]
  -- skipping the detailed proof
  sorry

end darma_eats_peanuts_l504_504856


namespace nail_cannot_fix_strip_l504_504331

noncomputable def point_in_plane_has_infinite_lines : Prop :=
∀ (P : Point), ∃ (L : Line), P ∈ L

theorem nail_cannot_fix_strip (P : Point) : 
  point_in_plane_has_infinite_lines :=
by
  -- Mathematically reason there are infinite lines passing through a point
  sorry

end nail_cannot_fix_strip_l504_504331


namespace sum_of_reciprocals_squares_lt_two_l504_504622

theorem sum_of_reciprocals_squares_lt_two (n : ℕ) (h : n > 0) : (∑ k in Finset.range(n) + 1, 1 / (k.succ : ℝ)^2) < 2 := 
by 
  sorry

end sum_of_reciprocals_squares_lt_two_l504_504622


namespace prime_numbers_between_50_and_70_l504_504051

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_between_50_and_70 : 
  (finset.filter is_prime (finset.range 71)).count (λ n, 50 ≤ n ∧ n ≤ 70) = 4 := 
sorry

end prime_numbers_between_50_and_70_l504_504051


namespace master_craftsman_parts_l504_504520

/-- 
Given:
  (1) the master craftsman produces 35 parts in the first hour,
  (2) at the rate of 35 parts/hr, he would be one hour late to meet the quota,
  (3) by increasing his speed by 15 parts/hr, he finishes the quota 0.5 hours early,
Prove that the total number of parts manufactured during the shift is 210.
-/
theorem master_craftsman_parts (N : ℕ) (quota : ℕ) 
  (initial_rate : ℕ := 35)
  (increased_rate_diff : ℕ := 15)
  (extra_time_slow : ℕ := 1)
  (time_saved_fast : ℕ := 1/2) :
  (quota = initial_rate * (extra_time_slow + 1) + N ∧
   increased_rate_diff = 15 ∧
   increased_rate_diff = λ (x : ℕ), initial_rate + x ∧
   time_saved_fast = 1/2 ∧
   N = 35) →
  quota = 210 := 
by
  sorry

end master_craftsman_parts_l504_504520


namespace initial_concentration_is_correct_l504_504817

-- Define the initial conditions
def vessel1_capacity : ℕ := 2 -- in liters
def vessel1_alcohol_concentration : ℝ := 0.20 -- 20%

def vessel2_capacity : ℕ := 6 -- in liters
def total_capacity : ℕ := 10 -- in liters (final mixed vessel capacity)
def final_concentration : ℝ := 0.37 -- 37%

-- Define the unknown concentration in the second vessel as x
noncomputable def initial_concentration_vessel2 (x : ℝ) : Prop :=
  0.4 + 0.06 * x = 3.7

-- Main statement to prove
theorem initial_concentration_is_correct :
  initial_concentration_vessel2 55 :=
by
  -- Automatically solve using the given conditions
  unfold initial_concentration_vessel2
  norm_num
  exact (0.4 + 0.06 * 55 = 3.7)
  sorry

end initial_concentration_is_correct_l504_504817


namespace fifth_square_area_is_36_l504_504119

noncomputable def area_of_fifth_square (right_square1 right_square2 left_square1 left_square2 : ℕ) : ℕ :=
  let s := (right_square1 + right_square2) - (left_square1 + left_square2)
  in s * s

theorem fifth_square_area_is_36 : area_of_fifth_square 3 8 1 4 = 36 := by
  let right_sum := 3 + 8
  let left_sum := 1 + 4
  let s := right_sum - left_sum
  have : s = 6 := by decide
  show area_of_fifth_square 3 8 1 4 = 36
  rw [area_of_fifth_square, ← this]
  norm_num
  sorry

end fifth_square_area_is_36_l504_504119


namespace allen_change_l504_504821

-- Define the cost per box and the number of boxes
def cost_per_box : ℕ := 7
def num_boxes : ℕ := 5

-- Define the total cost including the tip
def total_cost := num_boxes * cost_per_box
def tip := total_cost / 7
def total_paid := total_cost + tip

-- Define the amount given to the delivery person
def amount_given : ℕ := 100

-- Define the change received
def change := amount_given - total_paid

-- The statement to prove
theorem allen_change : change = 60 :=
by
  -- sorry is used here to skip the proof, as per the instruction
  sorry

end allen_change_l504_504821


namespace sqrt_3_is_irrational_l504_504863

theorem sqrt_3_is_irrational : irrational (real.sqrt 3) :=
by {
  sorry
}

end sqrt_3_is_irrational_l504_504863


namespace solve_equation_l504_504206

theorem solve_equation (x : ℂ) (h : x ≠ -1) :
  (x^2 + x + 1) / (x + 1) = x^2 + 2x + 3 ↔ x = -2 ∨ x = complex.I * real.sqrt 2 ∨ x = -complex.I * real.sqrt 2 :=
by
  sorry

end solve_equation_l504_504206


namespace max_subset_no_diff_6_or_8_l504_504140

theorem max_subset_no_diff_6_or_8 : 
  ∃ (T : Finset ℕ), (∀ x y ∈ T, (x ≠ y) → (|x - y| ≠ 6 ∧ |x - y| ≠ 8)) ∧ T ⊆ Finset.range 2001 ∧ T.card = 715 :=
sorry

end max_subset_no_diff_6_or_8_l504_504140


namespace master_craftsman_parts_l504_504521

/-- 
Given:
  (1) the master craftsman produces 35 parts in the first hour,
  (2) at the rate of 35 parts/hr, he would be one hour late to meet the quota,
  (3) by increasing his speed by 15 parts/hr, he finishes the quota 0.5 hours early,
Prove that the total number of parts manufactured during the shift is 210.
-/
theorem master_craftsman_parts (N : ℕ) (quota : ℕ) 
  (initial_rate : ℕ := 35)
  (increased_rate_diff : ℕ := 15)
  (extra_time_slow : ℕ := 1)
  (time_saved_fast : ℕ := 1/2) :
  (quota = initial_rate * (extra_time_slow + 1) + N ∧
   increased_rate_diff = 15 ∧
   increased_rate_diff = λ (x : ℕ), initial_rate + x ∧
   time_saved_fast = 1/2 ∧
   N = 35) →
  quota = 210 := 
by
  sorry

end master_craftsman_parts_l504_504521


namespace sum_of_squares_not_divisible_by_4_or_8_l504_504141

theorem sum_of_squares_not_divisible_by_4_or_8 (n : ℤ) (h : n % 2 = 1) :
  let a := n - 2
  let b := n
  let c := n + 2
  let sum_squares := a^2 + b^2 + c^2
  ¬(4 ∣ sum_squares ∨ 8 ∣ sum_squares) :=
by
  let a := n - 2
  let b := n
  let c := n + 2
  let sum_squares := a^2 + b^2 + c^2
  sorry

end sum_of_squares_not_divisible_by_4_or_8_l504_504141


namespace fourth_term_is_fifteen_l504_504488

-- Define the problem parameters
variables (a d : ℕ)

-- Define the conditions
def sum_first_third_term : Prop := (a + (a + 2 * d) = 10)
def fourth_term_def : ℕ := a + 3 * d

-- Declare the theorem to be proved
theorem fourth_term_is_fifteen (h1 : sum_first_third_term a d) : fourth_term_def a d = 15 :=
sorry

end fourth_term_is_fifteen_l504_504488


namespace find_m_l504_504139

theorem find_m (m : ℝ) :
  ({m, 2} : set ℝ) = {m^2 - 2, 2} → m = -1 := by
  sorry

end find_m_l504_504139


namespace smallest_sum_twice_perfect_square_l504_504255

-- Definitions based directly on conditions:
def sum_of_20_consecutive_integers (n : ℕ) : ℕ := (2 * n + 19) * 10

def twice_perfect_square (x : ℕ) : Prop := ∃ m : ℕ, x = 2 * m^2

-- Proposition to prove the smallest possible value satisfying these conditions:
theorem smallest_sum_twice_perfect_square : 
  ∃ n S, S = sum_of_20_consecutive_integers n ∧ twice_perfect_square S ∧ S = 450 :=
begin
  sorry
end

end smallest_sum_twice_perfect_square_l504_504255


namespace minimum_value_expression_l504_504573

theorem minimum_value_expression (x y z : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) (hz : -1 < z ∧ z < 1) :
  (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) ≥ 2) ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → (1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) = 2)) :=
sorry

end minimum_value_expression_l504_504573


namespace Allen_change_l504_504824

theorem Allen_change (boxes_cost : ℕ) (total_boxes : ℕ) (tip_fraction : ℚ) (money_given : ℕ) :
  boxes_cost = 7 → total_boxes = 5 → tip_fraction = 1/7 → money_given = 100 →
  let total_cost := boxes_cost * total_boxes in
  let tip := total_cost * tip_fraction in
  let total_spent := total_cost + tip in
  let change := money_given - total_spent in
  change = 60 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  unfold total_cost tip total_spent change
  norm_num
  unfold tip_fraction
  norm_num
  sorry -- Proof can be completed

end Allen_change_l504_504824


namespace count_primes_between_fifty_and_seventy_l504_504064

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_fifty_and_seventy : List ℕ :=
  [53, 59, 61, 67]

theorem count_primes_between_fifty_and_seventy :
  (primes_between_fifty_and_seventy.count is_prime = 4) :=
by
  sorry

end count_primes_between_fifty_and_seventy_l504_504064


namespace kim_time_away_from_home_l504_504555

noncomputable def time_away_from_home (distance_to_friend : ℕ) (detour_percent : ℕ) (stay_time : ℕ) (speed_mph : ℕ) : ℕ :=
  let return_distance := distance_to_friend * (1 + detour_percent / 100)
  let total_distance := distance_to_friend + return_distance
  let driving_time := total_distance / speed_mph
  let driving_time_minutes := driving_time * 60
  driving_time_minutes + stay_time

theorem kim_time_away_from_home : 
  time_away_from_home 30 20 30 44 = 120 := 
by
  -- We will handle the proof here
  sorry

end kim_time_away_from_home_l504_504555


namespace replace_with_monomial_produces_four_terms_l504_504174

-- Define the initial expression
def initialExpression (k : ℤ) (x : ℤ) : ℤ :=
  ((x^4 - 3)^2 + (x^3 + k)^2)

-- Proof statement
theorem replace_with_monomial_produces_four_terms (x : ℤ) :
  ∃ (k : ℤ), initialExpression k x = (x^8 + x^6 + 9x^2 + 9) :=
  exists.intro (3 * x) sorry

end replace_with_monomial_produces_four_terms_l504_504174


namespace combined_salaries_l504_504695

variable {A B C E : ℝ}
variable (D : ℝ := 7000)
variable (average_salary : ℝ := 8400)
variable (n : ℕ := 5)

theorem combined_salaries (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ E) 
  (h4 : B ≠ C) (h5 : B ≠ E) (h6 : C ≠ E)
  (h7 : average_salary = (A + B + C + D + E) / n) :
  A + B + C + E = 35000 :=
by
  sorry

end combined_salaries_l504_504695


namespace square_area_increase_l504_504696

theorem square_area_increase (s : ℝ) : 
  let A1 := s^2 in
  let A2 := (1.25 * s)^2 in
  (A2 - A1) / A1 * 100 = 56.25 :=
by
  sorry

end square_area_increase_l504_504696


namespace prime_count_between_50_and_70_l504_504048

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def primes_in_range (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem prime_count_between_50_and_70 : primes_in_range 50 70 = [53, 59, 61, 67] :=
sorry

end prime_count_between_50_and_70_l504_504048


namespace prime_count_between_50_and_70_l504_504084

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_50_and_70 : List ℕ :=
  [53, 59, 61, 67]

theorem prime_count_between_50_and_70 : 
  (primes_between_50_and_70.filter is_prime).length = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504084


namespace value_of_composition_l504_504566

def f (x : ℝ) : ℝ := 3 * x - 2
def g (x : ℝ) : ℝ := x - 1

theorem value_of_composition : g (f (1 + 2 * g 3)) = 12 := by
  sorry

end value_of_composition_l504_504566


namespace width_of_domain_l504_504966

theorem width_of_domain (h : ℝ → ℝ) (h_dom : ∀ x, x ∈ Set.Icc (-9 : ℝ) 9 → h x = h x) :
  let g (x: ℝ) := h (x / 3) in
  ∃ (a b : ℝ), Set.Icc a b = {x | -9 ≤ x / 3 ∧ x / 3 ≤ 9} ∧ b - a = 54 := 
sorry

end width_of_domain_l504_504966


namespace coefficient_x5_y2_l504_504673

-- We define the polynomial expression
def polynomial := (x^2 - x - 2 * y) ^ 5

-- We need to prove the coefficient of x^5 y^2 in the polynomial expansion is -120
theorem coefficient_x5_y2 : coefficient (expand polynomial) x^5 y^2 = -120 :=
begin
  sorry
end

end coefficient_x5_y2_l504_504673


namespace find_z_y_l504_504922

variable {Ω : Type} {P : Ω → Prop} -- Universe and a predicate P representing the probability space

-- Assume the existence of functions z representing the probability measure
variable (z : Set Ω → ℝ)

-- Conditions provided in the problem
variables (x y : Set Ω) (hx : z x = 0.02) (hxy : z (x ∩ y) = 0.10) (hcond : z x | y = 0.2)

-- Conclusion to be shown: \( z(y) = 0.5 \)
theorem find_z_y (z_y : z y = 0.5) : z y = 0.5 :=
begin
  sorry -- Proof to be provided
end

end find_z_y_l504_504922


namespace find_first_number_l504_504321

theorem find_first_number :
  ∃ x : ℝ, (x + (1 / 4) * 48 = 27) ∧ x = 15 :=
by
  use 15
  split
  · rw [← eq_sub_iff_add_eq] at h
    norm_num at h
  · refl

end find_first_number_l504_504321


namespace no_participation_l504_504347

variable (students : Finset ℕ)
variable (A B : Finset ℕ) -- A and B are the sets of students participating in volleyball and track respectively.
variable (total_students : ℕ) (card_A : ℕ) (card_B : ℕ) (card_A_inter_B : ℕ)

theorem no_participation (h_total : total_students = 45)
    (h_card_A : card_A = 12)
    (h_card_B : card_B = 20)
    (h_card_intersect : card_A_inter_B = 6) :
    total_students - (card_A + card_B - card_A_inter_B) = 19 :=
by 
    rw [h_total, h_card_A, h_card_B, h_card_intersect]
    simp
    sorry

end no_participation_l504_504347


namespace replace_asterisk_with_monomial_l504_504198

theorem replace_asterisk_with_monomial (x : ℝ) :
  (∀ asterisk : ℝ, ((x^4 - 3)^2 + (x^3 + asterisk)^2) = (x^8 + x^6 + 9x^2 + 9)) ↔ asterisk = 3x :=
by sorry

end replace_asterisk_with_monomial_l504_504198


namespace master_craftsman_quota_l504_504528

theorem master_craftsman_quota (parts_first_hour : ℕ)
  (extra_hour_needed : ℕ)
  (increased_speed : ℕ)
  (time_diff : ℕ)
  (total_parts : ℕ) :
  parts_first_hour = 35 →
  extra_hour_needed = 1 →
  increased_speed = 15 →
  time_diff = 1.5 →
  total_parts = parts_first_hour + (175 : ℕ) :=
by
  intros h1 h2 h3 h4
  rw [h1, h3]
  norm_num
  rw [add_comm]
  exact sorry

end master_craftsman_quota_l504_504528


namespace largest_interior_angle_of_triangle_l504_504917

-- Definitions for the sides of the triangle and the given conditions
variables (a b c : ℝ) (triangle_ABC : ∀ (a b c : ℝ), (c = 5) → (√(a - 4) + (b - 3)^2 = 0) → True)

namespace triangle_proof

-- Statement for the largest interior angle
theorem largest_interior_angle_of_triangle:
  ∀ (a b c : ℝ), c=5 → (√(a-4) + (b-3)^2 = 0) → a = 4 ∧ b = 3 → ∃ (θ : ℝ), θ = 90 :=
by
  intros a b c h1 h2 h3,
  use 90,
  exact sorry

end triangle_proof

end largest_interior_angle_of_triangle_l504_504917


namespace rationalizing_factor_sqrt3_plus_1_simplify_fraction1_simplify_fraction2_calculate_expression_l504_504615

theorem rationalizing_factor_sqrt3_plus_1 : (sqrt 3 + 1) * (sqrt 3 - 1) = 2 := 
by sorry

theorem simplify_fraction1 : 3 / (2 * sqrt 3) = sqrt 3 / 2 := 
by sorry

theorem simplify_fraction2 : (3 / (3 + sqrt 6)) = 3 - sqrt 6 := 
by sorry

theorem calculate_expression (n : ℕ) : 
  (\sum (i : ℕ) in (finset.range (n + 1) \ 
  finset.range 2).filter (λ (i : ℕ), i % 1 = 0), 
  (1 / sqrt (i + 1) + 1)) * (sqrt (n + 1) + 1) = 
  n * (n + 1) := 
by sorry

/- For n = 2008 -/

example : 
  (\sum (i : ℕ) in (finset.range 2009 \ 
  finset.range 1).filter (λ (i : ℕ), i % 1 = 0), 
  (1 / sqrt (i + 1) + 1)) * (sqrt 2009 + 1) = 
  2008 := 
by sorry

end rationalizing_factor_sqrt3_plus_1_simplify_fraction1_simplify_fraction2_calculate_expression_l504_504615


namespace bases_for_204_base_b_l504_504862

theorem bases_for_204_base_b (b : ℕ) : (∃ n : ℤ, 2 * b^2 + 4 = n^2) ↔ b = 4 ∨ b = 6 ∨ b = 8 ∨ b = 10 :=
by
  sorry

end bases_for_204_base_b_l504_504862


namespace kamal_average_score_l504_504554

-- Define the marks
def marks : List ℝ := [76, 60, 72, 65, 82]

-- Define the weights
def weights : List ℝ := [0.20, 0.30, 0.25, 0.15, 0.10]

-- Define a function to calculate the weighted average
def weighted_average (marks weights : List ℝ) : ℝ :=
  (List.zipWith (· * ·) marks weights).sum

-- Define the proof statement
theorem kamal_average_score :
  weighted_average marks weights = 69.15 :=
by
  sorry

end kamal_average_score_l504_504554


namespace constant_term_expansion_l504_504220

theorem constant_term_expansion :
  let x := λ : Type, (2 * x + 1 / x)^4 = 24

end constant_term_expansion_l504_504220


namespace replace_with_monomial_produces_four_terms_l504_504176

-- Define the initial expression
def initialExpression (k : ℤ) (x : ℤ) : ℤ :=
  ((x^4 - 3)^2 + (x^3 + k)^2)

-- Proof statement
theorem replace_with_monomial_produces_four_terms (x : ℤ) :
  ∃ (k : ℤ), initialExpression k x = (x^8 + x^6 + 9x^2 + 9) :=
  exists.intro (3 * x) sorry

end replace_with_monomial_produces_four_terms_l504_504176


namespace probability_diff_colors_l504_504357

theorem probability_diff_colors (total_balls red_balls white_balls selected_balls : ℕ) 
  (h_total : total_balls = 4)
  (h_red : red_balls = 2)
  (h_white : white_balls = 2)
  (h_selected : selected_balls = 2) :
  (∃ P : ℚ, P = (red_balls.choose (selected_balls / 2) * white_balls.choose (selected_balls / 2)) / total_balls.choose selected_balls ∧ P = 2 / 3) :=
by 
  sorry

end probability_diff_colors_l504_504357


namespace find_p_l504_504802

-- Define the parabola and its focus
def parabola (p : ℝ) := ∀ x y : ℝ, y^2 = 2 * p * x
def focus (p : ℝ) := (p / 2, 0)

-- Define the line passing through the focus with slope 1
def line_through_focus (p : ℝ) := ∀ x y : ℝ, y = x - p / 2

-- Define the circle
def circle (x y : ℝ) := (x - 5)^2 + y^2 = 8

-- Define the tangency condition
def tangent_condition (p : ℝ) := (|5 - p / 2| / Real.sqrt 2) = 2 * Real.sqrt 2

-- Theorem statement proving the required possible values of p
theorem find_p (p : ℝ) (hp : 0 < p) :
  (parabola p ∧ line_through_focus p ∧ tangent_condition p) → (p = 2 ∨ p = 18) :=
by
  sorry

end find_p_l504_504802


namespace distinct_numbers_exists_l504_504384

-- Defining the main proposition
noncomputable def exists_100_distinct_numbers : Prop :=
  ∃ a : Fin 100 → ℕ,
    (∀ i j, i ≠ j → a i ≠ a j) ∧
    (∀ S ⊆ Finset.univ, S.card = 52 →
      (∏ i in S, a i) ∣ (∏ i in (Finset.univ \ S), a i))

-- Statement asserting the existence of such 100 numbers
theorem distinct_numbers_exists : exists_100_distinct_numbers :=
sorry

end distinct_numbers_exists_l504_504384


namespace constant_term_expansion_eq_24_l504_504214

theorem constant_term_expansion_eq_24 :
  let a := (2 : ℝ) * X
  let b := (1 : ℝ) / X
  let n := 4
  (∀ X : ℝ, (2 * X + 1 / X)^4).constant_term = 24 :=
by
  sorry

end constant_term_expansion_eq_24_l504_504214


namespace find_fraction_l504_504867

theorem find_fraction (x : ℚ) (h : (1 / x) * (5 / 9) = 1 / 1.4814814814814814) : x = 740 / 999 := sorry

end find_fraction_l504_504867


namespace compute_difference_l504_504688

-- Define the operations
def op_⊕ (a b : ℚ) : ℚ := a^2 + b
def op_⊗ (a b : ℚ) : ℚ := a^3 / (b ⊕ 1)

-- The main proof goal
theorem compute_difference : 
  ((1 ⊗ 4) ⊗ 6) - (1 ⊗ (4 ⊗ 6)) = -68316 / 181781 := 
by
  sorry

end compute_difference_l504_504688


namespace max_stamps_l504_504486

theorem max_stamps (price_per_stamp : ℕ) (total_money : ℕ) (h1 : price_per_stamp = 45) (h2 : total_money = 5000) : ∃ n : ℕ, n = 111 ∧ 45 * n ≤ 5000 ∧ ∀ m : ℕ, (45 * m ≤ 5000) → m ≤ n := 
by
  sorry

end max_stamps_l504_504486


namespace simplify_sqrt_expression_l504_504658

theorem simplify_sqrt_expression :
  √(12 + 8 * √3) + √(12 - 8 * √3) = 4 * √3 :=
sorry

end simplify_sqrt_expression_l504_504658


namespace prime_count_between_50_and_70_l504_504045

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def primes_in_range (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem prime_count_between_50_and_70 : primes_in_range 50 70 = [53, 59, 61, 67] :=
sorry

end prime_count_between_50_and_70_l504_504045


namespace find_a_l504_504504

open Real

/-- Define the curve C in polar coordinates -/
def curve_polar (ρ θ : ℝ) (a : ℝ) (ha : a > 0) : Prop :=
  ρ * (sin θ)^2 = 2 * a * cos θ

/-- Define the parametric equations of the line l -/
def line_parametric (t : ℝ) : ℝ × ℝ :=
  (-2 + (sqrt 2)/2 * t, -4 + (sqrt 2)/2 * t)

/-- Define the Cartesian coordinate equation of the line l -/
def line_cartesian : ℝ × ℝ → Prop :=
  λ p, p.1 - p.2 - 2 = 0

/-- Define the Cartesian coordinate equation of the curve C -/
def curve_cartesian (x y a : ℝ) (ha : a > 0) : Prop :=
  y^2 = 2 * a * x

/-- Prove that given the conditions, the value of a = 1 -/
theorem find_a (a : ℝ) (ha : a > 0) :
  (∀ (t : ℝ) (x y : ℝ), (x, y) = line_parametric t → y^2 = 2 * a * x) →
  (∀ t1 t2 : ℝ, 
    t1 + t2 = 2 * sqrt 2 * (4 + a) ∧ t1 * t2 = 32 + 8 * a) →
  (∀ PM MN PN : ℝ, PM * PN = MN^2) →
  a = 1 :=
sorry

end find_a_l504_504504


namespace S_n_maximized_at_n_eq_50_l504_504986

open Nat

noncomputable def S_n_maximized_at (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∃ n_max, ∀ m, (∑ i in range (m + 1), a i) ≤ (∑ i in range (n_max + 1), a i)

theorem S_n_maximized_at_n_eq_50 (a : ℕ → ℝ) (h_decreasing : ∀ n, a n > a (n + 1)) 
  (h_condition : a 1 + a 100 = 0) : S_n_maximized_at a 50 :=
sorry

end S_n_maximized_at_n_eq_50_l504_504986


namespace dimensions_satisfy_condition_l504_504239
-- Import the required library

-- Define the basic structures and conditions
variables (m n : ℕ)
variables (x y z : ℕ)

-- Define the conditions
def lateral_surface_area (x y z : ℕ) := 2 * (x + y) * z
def sum_of_bases_areas (x y : ℕ) := 2 * x * y

theorem dimensions_satisfy_condition (m n : ℕ) (h : m < n) :
  let x := n * (n - m),
      y := m * n,
      z := m * (n - m)
  in lateral_surface_area x y z = sum_of_bases_areas x y :=
by {
  sorry
}

end dimensions_satisfy_condition_l504_504239


namespace range_of_k_l504_504589

noncomputable def problem (λ m α : ℝ) : Prop :=
  let a := (λ + 2, λ^2 - cos α ^ 2)
  let b := (m, m / 2 + sin α)
  a = (2 * b.1, 2 * b.2)

theorem range_of_k (λ m α : ℝ) (h : problem λ m α) :
  (∃ k : ℝ, k = λ / m ∧ k ∈ Icc (-6 : ℝ) 1) :=
sorry

end range_of_k_l504_504589


namespace log_sin_eq_3c_div_2_l504_504964

variable (b x c : ℝ)
variable (h1 : b > 1) (h2 : sin x > 0) (h3 : cos x > 0)
variable (h4 : log b (cos x) = c)
variable (h5 : log b (cos x) = (1 / 3) * log b (1 - b^(3 * c)))

theorem log_sin_eq_3c_div_2 : log b (sin x) = 3 * c / 2 :=
by
  sorry

end log_sin_eq_3c_div_2_l504_504964


namespace geometric_sequence_b_l504_504246

theorem geometric_sequence_b (b : ℝ) (h : b > 0) (s : ℝ) 
  (h1 : 30 * s = b) (h2 : b * s = 15 / 4) : 
  b = 15 * Real.sqrt 2 / 2 := 
by
  sorry

end geometric_sequence_b_l504_504246


namespace part1_part2_part3_l504_504587

variable {f : ℝ → ℝ}

-- Condition declarations
lemma cond1 (x y : ℝ) : f(x - y) = f(x) - f(y) := sorry
axiom cond2 : f(2) = 1
axiom cond3 (x : ℝ) : x > 0 → f(x) > 0

-- Prove f(0) = 0
theorem part1 : f(0) = 0 := 
by
  have h := cond1 0 0
  simp at h
  exact h

-- Prove f(x) is strictly increasing
theorem part2 : ∀ x1 x2 : ℝ, x1 > x2 → f(x1) > f(x2) := sorry

-- Prove the range of x such that f(x) + f(x+2) < 2 is {x | x < 1}
theorem part3 (x : ℝ) (h : f(x) + f(x + 2) < 2) : x < 1 := sorry

end part1_part2_part3_l504_504587


namespace simplify_sqrt_expression_is_correct_l504_504628

-- Definition for the given problem
def simplify_sqrt_expression (a b : ℝ) :=
  a = Real.sqrt (12 + 8 * Real.sqrt 3) → 
  b = Real.sqrt (12 - 8 * Real.sqrt 3) → 
  a + b = 4 * Real.sqrt 3

-- The theorem to be proven
theorem simplify_sqrt_expression_is_correct : simplify_sqrt_expression :=
begin
  intros a b ha hb,
  rw [ha, hb],
  -- Step-by-step simplification approach would occur here
  sorry
end

end simplify_sqrt_expression_is_correct_l504_504628


namespace prime_count_between_50_and_70_l504_504044

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def primes_in_range (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem prime_count_between_50_and_70 : primes_in_range 50 70 = [53, 59, 61, 67] :=
sorry

end prime_count_between_50_and_70_l504_504044


namespace equal_angles_of_tangents_and_midpoint_l504_504893

section Geometry

variables {Point : Type*} [MetricSpace Point] [MetricSpace.isMetric Point]
-- Hypotheses: 
variables (A M N P Q L : Point)
variable (circle : Point → Prop)
variable (tangent_to_circle : Point → Point → Prop)
variable [h_circle : MetricSpace.is_circle circle]
variable (midpoint : Point → Point → Point → Prop)

-- Tangency conditions
variables (h_tangent_AM : tangent_to_circle A M)
variable (h_tangent_AN : tangent_to_circle A N)

-- Secant line intersects circle at P and Q, and L is the midpoint
variables (secant_intersection : (P ∈ circle ∧ Q ∈ circle) ∧ h_midpoint_L : midpoint P Q L)

-- To be proven:
theorem equal_angles_of_tangents_and_midpoint
  (h_offset_A : ¬circle A)
  (h_midpoint_L : midpoint P Q L)
  (h_tangent_AM : tangent_to_circle A M)
  (h_tangent_AN : tangent_to_circle A N) :
  ∠MLA = ∠NLA :=
sorry

end Geometry

end equal_angles_of_tangents_and_midpoint_l504_504893


namespace substitute_monomial_to_simplify_expr_l504_504188

theorem substitute_monomial_to_simplify_expr (k : ℤ) : 
  ( ∃ k : ℤ, (x^4 - 3)^2 + (x^3 + k * x)^2 after expanding has exactly four terms) := 
begin
  use 3,
  sorry
end

end substitute_monomial_to_simplify_expr_l504_504188


namespace gasoline_price_reduction_l504_504108

theorem gasoline_price_reduction (P0 : ℝ)
    (h1 : P1 = (P0 + 0.30 * P0))
    (h2 : P2' = (P1 - 0.10 * P1))
    (h3 : P2 = (P2' + 0.15 * P2'))
    (h4 : P3 = (P2 + 0.15 * P2))
    (h5 : P4 = P3 - x / 100 * P3)
    (h6 : P4 = P0) :
  x ≈ 35 := by
  sorry

end gasoline_price_reduction_l504_504108


namespace sufficient_and_necessary_condition_l504_504571

def isMonotonicallyIncreasing {R : Type _} [LinearOrderedField R] (f : R → R) :=
  ∀ x y, x < y → f x < f y

def fx {R : Type _} [LinearOrderedField R] (x m : R) :=
  x^3 + 2*x^2 + m*x + 1

theorem sufficient_and_necessary_condition (m : ℝ) :
  (isMonotonicallyIncreasing (λ x => fx x m) ↔ m ≥ 4/3) :=
  sorry

end sufficient_and_necessary_condition_l504_504571


namespace inequality_ab_bc_ca_l504_504583

open Real

theorem inequality_ab_bc_ca (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b / (a + b) + b * c / (b + c) + c * a / (c + a)) ≤ (3 * (a * b + b * c + c * a) / (2 * (a + b + c))) := by
sorry

end inequality_ab_bc_ca_l504_504583


namespace percentage_X_correct_l504_504830

def initial_solution_Y : ℝ := 12.0
def percentage_X_in_Y : ℝ := 45.0 / 100.0
def percentage_water_in_Y : ℝ := 55.0 / 100.0
def evaporated_water : ℝ := 5.0
def added_solution_Y : ℝ := 7.0

def calculate_percentage_X_in_new_solution : ℝ :=
  let initial_X := 0.45 * 12.0
  let initial_water := 0.55 * 12.0
  let remaining_water := initial_water - evaporated_water

  let added_X := 0.45 * added_solution_Y
  let added_water := 0.55 * added_solution_Y

  let total_X := initial_X + added_X
  let total_water := remaining_water + added_water

  let total_weight := total_X + total_water
  (total_X / total_weight) * 100.0

theorem percentage_X_correct :
  abs (calculate_percentage_X_in_new_solution - 61.07) < 0.01 := 
  sorry

end percentage_X_correct_l504_504830


namespace cylinder_surface_area_proof_l504_504094

variable (height : ℝ) (circumference : ℝ)

def cylinder_surface_area (height circumference : ℝ) : ℝ :=
  let r := circumference / (2 * Real.pi)
  (2 * Real.pi * r * height) + (2 * Real.pi * r^2)

theorem cylinder_surface_area_proof (h_height : height = 4) (h_circumference : circumference = 2 * Real.pi) :
  cylinder_surface_area height circumference = 10 * Real.pi := by
  -- Proof omitted
  sorry

end cylinder_surface_area_proof_l504_504094


namespace approx_sin_two_l504_504586

-- Define the function and its derivatives at zero
def f (x : ℝ) : ℝ := Real.sin x
def f' (x : ℝ) : ℝ := Real.cos x
def f'' (x : ℝ) : ℝ := -Real.sin x
def f''' (x : ℝ) : ℝ := -Real.cos x
def f'''' (x : ℝ) : ℝ := Real.sin x
def f''''' (x : ℝ) : ℝ := Real.cos x

-- Axioms for derivative evaluations at zero
axiom f_0 : f 0 = 0
axiom f'_0 : f' 0 = 1
axiom f''_0 : f'' 0 = 0
axiom f'''_0 : f''' 0 = -1
axiom f''''_0 : f'''' 0 = 0
axiom f'''''_0 : f''''' 0 = 1

-- Statement of the proof problem
theorem approx_sin_two : 
  Real.sin 2 ≈ f 0 + (f' 0 / 1!) * 2 + (f'' 0 / 2!) * 2^2 + (f''' 0 / 3!) * 2^3 + (f'''' 0 / 4!) * 2^4 + (f''''' 0 / 5!) * 2^5 :=
by
  sorry

end approx_sin_two_l504_504586


namespace kim_total_time_away_l504_504558

noncomputable def total_time_away (d : ℝ) (detour_percentage : ℝ) (time_at_friends : ℝ) (speed : ℝ) : ℝ :=
  let detour_distance := d * detour_percentage
  let total_return_distance := d + detour_distance
  let total_distance := d + total_return_distance
  let driving_time := total_distance / speed
  driving_time + time_at_friends

theorem kim_total_time_away :
  total_time_away 30 0.2 (30 / 60) 44 = 2 :=
by
  delta total_time_away -- unfold the definition of total_time_away
  simp only [div_eq_mul_inv]
  norm_num
  sorry

end kim_total_time_away_l504_504558


namespace prime_count_between_50_and_70_l504_504087

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_50_and_70 : List ℕ :=
  [53, 59, 61, 67]

theorem prime_count_between_50_and_70 : 
  (primes_between_50_and_70.filter is_prime).length = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504087


namespace fraction_of_Charlie_circumference_l504_504548

/-- Definitions for the problem conditions -/
def Jack_head_circumference : ℕ := 12
def Charlie_head_circumference : ℕ := 9 + Jack_head_circumference / 2
def Bill_head_circumference : ℕ := 10

/-- Statement of the theorem to be proved -/
theorem fraction_of_Charlie_circumference :
  Bill_head_circumference / Charlie_head_circumference = 2 / 3 :=
sorry

end fraction_of_Charlie_circumference_l504_504548


namespace satisfies_properties_l504_504287

noncomputable def f (x : ℝ) : ℝ := x^2

theorem satisfies_properties :
  (∀ x1 x2 : ℝ, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x : ℝ, 0 < x → 0 < (f' x)) ∧
  (∀ x : ℝ, f' (-x) = - f' x) := 
sorry

end satisfies_properties_l504_504287


namespace lucky_numbers_below_2010_l504_504748

def is_lucky (b : ℕ) : Prop :=
  ∀ (a : ℕ), (a^5) % (b^2) = 0 → (a^2) % b = 0

theorem lucky_numbers_below_2010 : 
  { b : ℕ // b < 2010 ∧ is_lucky b }.card = 1961 :=
by
  sorry

end lucky_numbers_below_2010_l504_504748


namespace exists_t_for_f_inequality_l504_504424

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * (x + 1) ^ 2

theorem exists_t_for_f_inequality :
  ∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ 9 → f (x + t) ≤ x := by
  sorry

end exists_t_for_f_inequality_l504_504424


namespace butterflies_let_go_l504_504552

theorem butterflies_let_go (H₁ : original_butterflies = 93) (H₂ : butterflies_left = 82) : 
  let butterflies_let_go := original_butterflies - butterflies_left in
  butterflies_let_go = 11 :=
by 
  sorry

end butterflies_let_go_l504_504552


namespace singleton_intersection_condition_l504_504431

-- Define A and B based on given conditions
def A (a : ℝ) : set (ℝ × ℝ) := { p : ℝ × ℝ | p.2 = a * p.1 + 1 }
def B : set (ℝ × ℝ) := { p : ℝ × ℝ | p.2 = |p.1| }

-- State the theorem
theorem singleton_intersection_condition (a : ℝ) :
  (∃! (p : ℝ × ℝ), p ∈ A a ∧ p ∈ B) ↔ (a ≥ 1 ∨ a ≤ -1) :=
by
  sorry

end singleton_intersection_condition_l504_504431


namespace parts_manufactured_l504_504537

variable (initial_parts : ℕ) (initial_rate : ℕ) (increased_speed : ℕ) (time_diff : ℝ)
variable (N : ℕ)

-- initial conditions
def initial_parts := 35
def initial_rate := 35
def increased_speed := 15
def time_diff := 1.5

-- additional parts to be manufactured
noncomputable def additional_parts := N

-- equation representing the time differences
noncomputable def equation := (N / initial_rate) - (N / (initial_rate + increased_speed)) = time_diff

-- state the proof problem
theorem parts_manufactured : initial_parts + additional_parts = 210 :=
by
  -- Use the given conditions to solve the problem
  sorry

end parts_manufactured_l504_504537


namespace strictly_increasing_difference_l504_504236

variable {a b : ℝ}
variable {f g : ℝ → ℝ}

theorem strictly_increasing_difference
  (h_diff : ∀ x ∈ Set.Icc a b, DifferentiableAt ℝ f x ∧ DifferentiableAt ℝ g x)
  (h_eq : f a = g a)
  (h_diff_ineq : ∀ x ∈ Set.Ioo a b, (deriv f x : ℝ) > (deriv g x : ℝ)) :
  ∀ x ∈ Set.Ioo a b, f x > g x := by
  sorry

end strictly_increasing_difference_l504_504236


namespace inscribed_circle_area_l504_504911

theorem inscribed_circle_area (a : ℝ) :
  let l1 := λ x y : ℝ, x + 2 * y = a + 2,
      l2 := λ x y : ℝ, 2 * x - y = 2 * a - 1,
      circle := λ x y : ℝ, (x - a)^2 + (y - 1)^2 = 16,
      point_of_intersection := (a, 1),
      radius_of_inscribed_circle := 2 * sqrt 2,
      area_of_inscribed_circle := (radius_of_inscribed_circle)^2 * π in
  area_of_inscribed_circle = 8 * π :=
sorry

end inscribed_circle_area_l504_504911


namespace master_craftsman_quota_l504_504532

theorem master_craftsman_quota (N : ℕ) (initial_rate increased_rate : ℕ) (additional_hours extra_hours : ℝ) :
  initial_rate = 35 →
  increased_rate = initial_rate + 15 →
  additional_hours = 0.5 →
  extra_hours = 1 →
  N / initial_rate - N / increased_rate = additional_hours + extra_hours →
  N = 175 →
  (initial_rate + N) = 210 :=
by {
  intros h1 h2 h3 h4 h5 h6,
  rw h6,
  exact rfl,
}

end master_craftsman_quota_l504_504532


namespace yoongi_division_l504_504791

theorem yoongi_division (x : ℕ) (h : 5 * x = 100) : x / 10 = 2 := by
  sorry

end yoongi_division_l504_504791


namespace sum_possible_n_l504_504810

theorem sum_possible_n :
  let n_values := {n : ℕ | 4 < n ∧ n < 18}
  ∑ n in n_values, n = 143 :=
by
  sorry

end sum_possible_n_l504_504810


namespace annual_interest_rate_l504_504136

-- Define the conditions as constants
constant principal : ℝ := 35    -- The original amount charged
constant total_amount : ℝ := 37.1 -- The total amount owed after a year
constant time : ℝ := 1 -- Time period in years

-- Define the simple interest formula
def simple_interest (P R T : ℝ) : ℝ :=
  P * R * T

-- Calculate the interest charged over the year
def interest := total_amount - principal

-- The theorem we want to prove
theorem annual_interest_rate : 
  ∃ R : ℝ, simple_interest principal R time = interest ∧ R * 100 = 6 :=
by
  sorry

end annual_interest_rate_l504_504136


namespace smallest_number_is_42_l504_504262

theorem smallest_number_is_42 (x : ℤ) 
  (h1 : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 225)
  (h2 : x % 7 = 0) : 
  x = 42 := 
sorry

end smallest_number_is_42_l504_504262


namespace part_a_part_b_l504_504885

-- Let p(n) be the product of its digits
def p (n : ℕ) : ℕ :=
  (n.digits 10).prod

/- 
  Part (a): Prove that for all positive integers n, 
  p(n) ≤ n 
-/
theorem part_a (n : ℕ) (h_pos : 0 < n) : p(n) ≤ n := 
  sorry

/- 
  Part (b): Find all positive integers n such that 
  10 * p(n) = n^2 + 4 * n - 2005.
-/
theorem part_b (n : ℕ) (h_eq : 10 * p n = n^2 + 4 * n - 2005) : n = 45 :=
  sorry

end part_a_part_b_l504_504885


namespace Allen_change_l504_504823

theorem Allen_change (boxes_cost : ℕ) (total_boxes : ℕ) (tip_fraction : ℚ) (money_given : ℕ) :
  boxes_cost = 7 → total_boxes = 5 → tip_fraction = 1/7 → money_given = 100 →
  let total_cost := boxes_cost * total_boxes in
  let tip := total_cost * tip_fraction in
  let total_spent := total_cost + tip in
  let change := money_given - total_spent in
  change = 60 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  unfold total_cost tip total_spent change
  norm_num
  unfold tip_fraction
  norm_num
  sorry -- Proof can be completed

end Allen_change_l504_504823


namespace quadrilateral_not_parallelogram_l504_504430

theorem quadrilateral_not_parallelogram (a b c d : ℕ) (h₁ : a = 16) 
  (h₂ : b = 13) (h₃ : c = 10) (h₄ : d = 6) (h₅ : a ≠ c) (h₆ : a ∣ c ∨ c ∣ a) : ¬ (a ∥ c ∧ b ∥ d) :=
by {
  sorry
}

end quadrilateral_not_parallelogram_l504_504430


namespace find_f_prime_zero_l504_504580

noncomputable def smooth_fn (f : ℝ → ℝ) : Prop :=
∀ n, continuous_diff ℝ n f

theorem find_f_prime_zero
  (f : ℝ → ℝ)
  (h_smooth : smooth_fn f)
  (h_diff_eq : ∀ x, (derivative f x)^2 = f x * (second_derivative f x))
  (h_f_zero : f 0 = 1)
  (h_f_fourth_zero : (derivative^[4] f) 0 = 9) :
  ∃ a : ℝ, (derivative f 0 = a ∧ (a = sqrt 3 ∨ a = -sqrt 3)) := sorry

end find_f_prime_zero_l504_504580


namespace round_2741836_to_nearest_integer_l504_504618

theorem round_2741836_to_nearest_integer :
  (2741836.4928375).round = 2741836 := 
by
  -- Explanation that 0.4928375 < 0.5 leading to rounding down
  sorry

end round_2741836_to_nearest_integer_l504_504618


namespace sequence_sum_100_l504_504588

-- Definitions of the sequence and conditions
def sequence (a : ℕ → ℕ) :=
  a 1 = 1 ∧ 
  a 2 = 1 ∧ 
  a 3 = 2 ∧ 
  (∀ n : ℕ, a n * a (n + 1) * a (n + 2) ≠ 1) ∧ 
  (∀ n : ℕ, a n * a (n + 1) * a (n + 2) * a (n + 3) = a n + a (n + 1) + a (n + 2) + a (n + 3))

-- The theorem to be proved
theorem sequence_sum_100 (a : ℕ → ℕ) (h : sequence a) : (∑ i in Ico 1 101 , a i) = 200 :=
by
  sorry

end sequence_sum_100_l504_504588


namespace inequality_holds_l504_504928

noncomputable def f (x : ℝ) : ℝ := -x^2 - 2*x
def a : ℝ := Real.log 2
def b : ℝ := Real.log 2 / Real.log (1 / 3)
def c : ℝ := Real.sqrt 3

theorem inequality_holds :
  f(b) > f(a) ∧ f(a) > f(c) :=
by 
  -- conditions
  have h1 : f = λ x, -x^2 - 2*x := rfl,
  have ha : 0 < a ∧ a < 1 := sorry,
  have hb : -1 < b ∧ b < 0 := sorry,
  have hc : 1 < c ∧ c < 2 := sorry,
  -- Given conditions on the behavior of f on specified intervals
  have hf_decreasing : ∀ x y, (-1 ≤ x ∧ -1 ≤ y ∧ x < y) → f(x) > f(y) := sorry,
  -- Proof of the inequality using aforementioned conditions
  sorry

end inequality_holds_l504_504928


namespace valid_votes_for_candidate_E_l504_504988

theorem valid_votes_for_candidate_E : 
  let total_registered_voters := 20000
  let urban_voters := 10000
  let rural_voters := 10000
  let urban_turnout := 0.8 * urban_voters
  let rural_turnout := 0.6 * rural_voters
  let total_votes := urban_turnout + rural_turnout
  let invalid_percentage := 0.15
  let valid_vote_percentage := 1 - invalid_percentage
  let valid_urban_votes := valid_vote_percentage * urban_turnout
  let valid_rural_votes := valid_vote_percentage * rural_turnout
  let total_valid_votes := valid_urban_votes + valid_rural_votes
  let votes_A := 0.4 * total_valid_votes
  let votes_B := 0.3 * total_valid_votes
  let votes_C := 0.15 * total_valid_votes
  let votes_D := 0.1 * total_valid_votes
  let votes_E := total_valid_votes - (votes_A + votes_B + votes_C + votes_D)
  in votes_E = 595 :=
sorry

end valid_votes_for_candidate_E_l504_504988


namespace batsman_average_after_17th_inning_l504_504767

theorem batsman_average_after_17th_inning 
    (A : ℕ)  -- assuming A (the average before the 17th inning) is a natural number
    (h₁ : 16 * A + 85 = 17 * (A + 3)) : 
    A + 3 = 37 := by
  sorry

end batsman_average_after_17th_inning_l504_504767


namespace minimize_cost_l504_504795

theorem minimize_cost 
  (k1 k2 : ℝ)
  (d1 : ℝ := 10) (y1 : ℝ := 20000) (y2 : ℝ := 80000)
  (h1 : k1 = y1 * d1) (h2 : k2 = y2 / d1) :
  ∃ d : ℝ, d = 5 :=
by
  use 5
  sorry

end minimize_cost_l504_504795


namespace find_AD_AF_l504_504996

-- Define the geometric configuration and the required properties
namespace TriangleProblem

variables {A B C D E F : Type} [EuclideanGeometry A B C D E F]

-- Conditions stated in the problem
def is_isosceles_triangle (ABC : Triangle A B C) : Prop :=
  ABC.isosceles ∧ ABC.a = ABC.b ∧ ABC.a = √5 ∧ ABC.b = √5

def on_side_not_midpoint (D : Point) (BC : Line B C) : Prop :=
  D ∈ BC ∧ ¬(D.is_midpoint_of BC)

def reflected_about (E : Point) (C : Point) (AD : Line A D) : Prop :=
  E.is_reflection_of C about AD

def intersection (F : Point) (EB : Line E B) (AD : Line A D) : Prop :=
  F ∈ (EB ∩ AD)

-- Question translated into a Lean 4 statement
theorem find_AD_AF (ABC : Triangle A B C) (D : Point) (E : Point) (F : Point)
  (h_iso : is_isosceles_triangle ABC)
  (h_on_side : on_side_not_midpoint D BC)
  (h_reflect : reflected_about E C AD)
  (h_intersect : intersection F EB AD) :
  AD.length * AF.length = 5 := 
sorry

end TriangleProblem

end find_AD_AF_l504_504996


namespace point_between_lines_l504_504100

theorem point_between_lines (b : ℝ) (h1 : 6 * 5 - 8 * b + 1 < 0) (h2 : 3 * 5 - 4 * b + 5 > 0) : b = 4 :=
  sorry

end point_between_lines_l504_504100


namespace sam_bikes_speed_l504_504386

noncomputable def EugeneSpeed : ℝ := 5
noncomputable def ClaraSpeed : ℝ := (3/4) * EugeneSpeed
noncomputable def SamSpeed : ℝ := (4/3) * ClaraSpeed

theorem sam_bikes_speed :
  SamSpeed = 5 :=
by
  -- Proof will be filled here.
  sorry

end sam_bikes_speed_l504_504386


namespace right_triangle_third_side_product_l504_504728

theorem right_triangle_third_side_product :
  ∀ (a b : ℝ), (a = 6 ∧ b = 8 ∧ (a^2 + b^2 = c^2 ∨ a^2 = b^2 - c^2)) →
  (a * b = 53.0) :=
by
  intros a b h
  sorry

end right_triangle_third_side_product_l504_504728


namespace satisfies_properties_l504_504288

noncomputable def f (x : ℝ) : ℝ := x^2

theorem satisfies_properties :
  (∀ x1 x2 : ℝ, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x : ℝ, 0 < x → 0 < (f' x)) ∧
  (∀ x : ℝ, f' (-x) = - f' x) := 
sorry

end satisfies_properties_l504_504288


namespace x_y_multiple_of_3_l504_504210

-- Conditions: x is a multiple of 6, y is a multiple of 9
variables {m n : ℤ}
def x : ℤ := 6 * m
def y : ℤ := 9 * n

-- Proof that x + y is a multiple of 3
theorem x_y_multiple_of_3 : ∃ k : ℤ, x + y = 3 * k :=
by {
  sorry
}

end x_y_multiple_of_3_l504_504210


namespace right_triangle_third_side_product_l504_504732

theorem right_triangle_third_side_product :
  let a := 6
  let b := 8
  let c1 := Real.sqrt (a^2 + b^2)     -- Hypotenuse when a and b are legs
  let c2 := Real.sqrt (b^2 - a^2)     -- Other side when b is the hypotenuse
  20 * Real.sqrt 7 ≈ 52.7 := 
by
  sorry

end right_triangle_third_side_product_l504_504732


namespace selection_ways_l504_504324

theorem selection_ways :
  let volunteers : Finset ℕ := {0, 1, 2, 3, 4}
  let ABC : Finset ℕ := {0, 1, 2}
  ∃ translators guides flexible_staff,
  translators.card = 2 ∧ guides.card = 2 ∧ flexible_staff.card = 1 ∧
  translators ⊆ volunteers ∧ guides ⊆ volunteers ∧ flexible_staff ⊆ volunteers ∧
  (translators ∩ ABC).nonempty ∧ (guides ∩ ABC).nonempty ∧
  (volunteers.card - (translators ∪ guides).card = 1) ∧
  (Finset.card (translators ∪ guides ∪ flexible_staff) = 5) →
  (∀ C : ∀ {(translators guides flexible_staff)},
   translators.card = 2 ∧ guides.card = 2 ∧ flexible_staff.card = 1 ∧
   translators ⊆ volunteers ∧ guides ⊆ volunteers ∧ flexible_staff ⊆ volunteers ∧
   (translators ∩ ABC).nonempty ∧ (guides ∩ ABC).nonempty ∧
   (volunteers.card - (translators ∪ guides).card = 1) ∧
   (Finset.card (translators ∪ guides ∪ flexible_staff) = 5), 
    36)
:= sorry

end selection_ways_l504_504324


namespace sequence_an_square_l504_504542

theorem sequence_an_square (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) > a n) 
  (h3 : ∀ n : ℕ, a (n + 1)^2 + a n^2 + 1 = 2 * (a (n + 1) * a n + a (n + 1) + a n)) :
  ∀ n : ℕ, a n = n^2 :=
by
  sorry

end sequence_an_square_l504_504542


namespace weight_difference_l504_504708

theorem weight_difference:
  (∃ (a : Fin 40 → ℝ), 
    (∀ i j, |a i - a j| ≤ 45) ∧
    (∀ (s : Finset (Fin 40)) (hcard : s.card = 10), 
      ∃ (s1 s2 : Finset (Fin 40)) (h1 : s1.card = 5) (h2 : s2.card = 5), 
        s1 ∪ s2 = s ∧ s1 ∩ s2 = ∅ ∧ |∑ i in s1, a i - ∑ i in s2, a i| ≤ 11)) 
    → (∃ i j, |a i - a j| ≤ 1)) :=
sorry

end weight_difference_l504_504708


namespace circle_intersection_count_l504_504379

theorem circle_intersection_count :
  let r1 θ := 3 * Real.cos θ
  let r2 θ := 6 * Real.sin θ
  ∀ θ₁ θ₂, r1 θ₁ = r2 θ₁ ∧ r1 θ₂ = r2 θ₂ →
  (θ₁ ≠ θ₂) → 2 :=
sorry

end circle_intersection_count_l504_504379


namespace cost_price_of_each_clock_l504_504768

theorem cost_price_of_each_clock
  (C : ℝ)
  (h1 : 40 * C * 1.1 + 50 * C * 1.2 - 90 * C * 1.15 = 40) :
  C = 80 :=
sorry

end cost_price_of_each_clock_l504_504768


namespace increase_in_radius_l504_504604

theorem increase_in_radius
  (pi : ℝ)
  (miles_to_inches : ℝ)
  (original_radius : ℝ)
  (odometer_reading_trip : ℝ)
  (odometer_reading_snow : ℝ) :
  let circumference_original := 2 * pi * original_radius in 
  let distance_per_rotation := circumference_original / miles_to_inches in 
  let trips_rotations := odometer_reading_trip / distance_per_rotation in 
  let radius_new := (odometer_reading_trip * distance_per_rotation * miles_to_inches) / (2 * pi * odometer_reading_snow) in 
  let increase := radius_new - original_radius in 
  increase ≈ 0.39  :=
by
  sorry

end increase_in_radius_l504_504604


namespace sum_of_digits_of_d_l504_504161

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem sum_of_digits_of_d (d : ℕ) 
  (h_exchange : 15 * d = 9 * (d * 5 / 3)) 
  (h_spending : (5 * d / 3) - 120 = d) 
  (h_d_eq : d = 180) : sum_of_digits d = 9 := by
  -- This is where the proof would go
  sorry

end sum_of_digits_of_d_l504_504161


namespace abs_add_lt_abs_sub_l504_504962

variable {a b : ℝ}

theorem abs_add_lt_abs_sub (h1 : a * b < 0) : |a + b| < |a - b| :=
sorry

end abs_add_lt_abs_sub_l504_504962


namespace prime_numbers_between_50_and_70_l504_504053

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_between_50_and_70 : 
  (finset.filter is_prime (finset.range 71)).count (λ n, 50 ≤ n ∧ n ≤ 70) = 4 := 
sorry

end prime_numbers_between_50_and_70_l504_504053


namespace old_oranges_thrown_away_l504_504350

theorem old_oranges_thrown_away (x : ℕ) : 
  31 - x + 38 = 60 → x = 9 :=
by 
  intros h
  have h1 : 69 - x = 60 := by { rw [←add_assoc, add_comm, add_assoc] at h, exact h }
  have h2 : -x = -9 := by linarith
  exact neg_inj.mp h2

end old_oranges_thrown_away_l504_504350


namespace area_sector_approx_l504_504671

-- Definition of the given conditions
def radius : ℝ := 12
def theta_deg : ℝ := 41

-- Definition of the area formula for a sector
def area_of_sector (r : ℝ) (θ : ℝ) : ℝ :=
  (θ / 360) * Real.pi * (r ^ 2)

-- Proving the area of the sector given the conditions
theorem area_sector_approx :
  |area_of_sector radius theta_deg - 51.57| < 0.01 :=
by
  -- The proof is skipped, here we assert the statement is approximately true
  sorry

end area_sector_approx_l504_504671


namespace triangle_angle_A_eq_pi_div_3_find_side_a_l504_504437

-- Definitions for a triangle with sides opposite angles A, B, and C as a, b, and c respectively
variables {a b c : ℝ}
-- Additional definitions and conditions
variable (h1: b^2 + c^2 - a^2 = b * c)
variable (h2: sin B + sqrt 3 * cos B = 2)
variable (area_abc : ℝ)
variable (h3 : area_abc = 3 * sqrt 3 / 2)

-- Proofs required
theorem triangle_angle_A_eq_pi_div_3 (h1: b^2 + c^2 - a^2 = b * c) : 
  ∠A = π / 3 := sorry

theorem find_side_a (h2: sin C + sqrt 3 * cos C = 2) (area_abc : area_abc = 3 * sqrt 3 / 2) :
  a = 3 := sorry

end triangle_angle_A_eq_pi_div_3_find_side_a_l504_504437


namespace overall_profit_percentage_l504_504803

noncomputable def cost_A := 50
noncomputable def cost_B := 75
noncomputable def cost_C := 100
noncomputable def cost_D := 150
noncomputable def cost_E := 200

noncomputable def markup_A := 15
noncomputable def markup_B := 20
noncomputable def markup_C := 40
noncomputable def markup_D := 60
noncomputable def markup_E := 80

noncomputable def discount_A := 10
noncomputable def discount_B := 5
noncomputable def discount_C := 15
noncomputable def discount_D := 20
noncomputable def discount_E := 30

noncomputable def selling_price_A := cost_A + markup_A - discount_A
noncomputable def selling_price_B := cost_B + markup_B - discount_B
noncomputable def selling_price_C := cost_C + markup_C - discount_C
noncomputable def selling_price_D := cost_D + markup_D - discount_D
noncomputable def selling_price_E := cost_E + markup_E - discount_E

noncomputable def profit_A := selling_price_A - cost_A
noncomputable def profit_B := selling_price_B - cost_B
noncomputable def profit_C := selling_price_C - cost_C
noncomputable def profit_D := selling_price_D - cost_D
noncomputable def profit_E := selling_price_E - cost_E

noncomputable def total_cost := cost_A + cost_B + cost_C + cost_D + cost_E
noncomputable def total_profit := profit_A + profit_B + profit_C + profit_D + profit_E

noncomputable def profit_percentage := (total_profit / total_cost.to_real) * 100

theorem overall_profit_percentage :
  profit_percentage = 23.48 := by
  sorry

end overall_profit_percentage_l504_504803


namespace reassemble_equilateral_triangle_l504_504854

-- Defining conditions for the problem

def triangle (a : ℝ) : Prop := a > 0

def cut_into_three_parts (a : ℝ) : Prop := 
  triangle a ∧ 
  ⟦ ∃ T1 T2 T3 : ℝ, T1 = a / 2 ∧ T2 = a / 4 ∧ T3 = 3 * (a / 4) ⟧

-- Problem statement in Lean 4
theorem reassemble_equilateral_triangle :
  ∀ (a b : ℝ), triangle a ∧ triangle b ∧ cut_into_three_parts 2 ∧ cut_into_three_parts 3 →
  ∃ c : ℝ, c = Real.sqrt (2^2 + 3^2) :=
by 
  sorry

end reassemble_equilateral_triangle_l504_504854


namespace ordered_pairs_count_l504_504469

noncomputable def num_ordered_pairs (x y : ℝ) : ℕ :=
if hx : x^2 + y^2 = 200 ∧ (∃ (n : ℤ), n = ⟨(Real.sqrt ((x - 5)^2 + (y - 5)^2) + Real.sqrt ((x + 5)^2 + (y + 5)^2))⟩) then 
  1
else 
  0

theorem ordered_pairs_count : ∑ x y in ({ p | num_ordered_pairs p.1 p.2 = 1} : set (ℝ × ℝ)), 1 = 12 := by
  sorry

end ordered_pairs_count_l504_504469


namespace work_completion_time_l504_504666

-- Conditions:
variable (P W : ℝ)
variable (h : (2 * P) * 5 = W / 2)

-- Statement of the problem:
theorem work_completion_time :
  let d := (10 * 2) in
  P * d = W := by
  sorry

end work_completion_time_l504_504666


namespace percentage_X_correct_l504_504831

def initial_solution_Y : ℝ := 12.0
def percentage_X_in_Y : ℝ := 45.0 / 100.0
def percentage_water_in_Y : ℝ := 55.0 / 100.0
def evaporated_water : ℝ := 5.0
def added_solution_Y : ℝ := 7.0

def calculate_percentage_X_in_new_solution : ℝ :=
  let initial_X := 0.45 * 12.0
  let initial_water := 0.55 * 12.0
  let remaining_water := initial_water - evaporated_water

  let added_X := 0.45 * added_solution_Y
  let added_water := 0.55 * added_solution_Y

  let total_X := initial_X + added_X
  let total_water := remaining_water + added_water

  let total_weight := total_X + total_water
  (total_X / total_weight) * 100.0

theorem percentage_X_correct :
  abs (calculate_percentage_X_in_new_solution - 61.07) < 0.01 := 
  sorry

end percentage_X_correct_l504_504831


namespace sale_in_second_month_l504_504327

def sale_first_month : ℕ := 6435
def sale_third_month : ℕ := 6855
def sale_fourth_month : ℕ := 7230
def sale_fifth_month : ℕ := 6562
def sale_sixth_month : ℕ := 6191
def average_sale : ℕ := 6700

theorem sale_in_second_month : 
  ∀ (sale_second_month : ℕ), 
    (sale_first_month + sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month = 6700 * 6) → 
    sale_second_month = 6927 :=
by
  intro sale_second_month h
  sorry

end sale_in_second_month_l504_504327


namespace trigonometric_cos_value_l504_504897

open Real

theorem trigonometric_cos_value (α : ℝ) (h : sin (α + π / 6) = 1 / 3) : 
  cos (2 * α - 2 * π / 3) = -7 / 9 := 
sorry

end trigonometric_cos_value_l504_504897


namespace no_natural_number_n_exists_l504_504868

theorem no_natural_number_n_exists :
  ∀ (n : ℕ), ¬ ∃ (a b : ℕ), 3 * n + 1 = a * b := by
  sorry

end no_natural_number_n_exists_l504_504868


namespace journey_distance_l504_504354

theorem journey_distance 
  (T : ℝ) 
  (s1 s2 s3 : ℝ) 
  (hT : T = 36) 
  (hs1 : s1 = 21)
  (hs2 : s2 = 45)
  (hs3 : s3 = 24) : ∃ (D : ℝ), D = 972 :=
  sorry

end journey_distance_l504_504354


namespace exists_point_in_half_of_sets_l504_504621

open Set

-- Definition of sets A_i each as a union of two segments [a_i, b_i] ∪ [c_i, d_i]
variable {n : ℕ} (A : Fin n → Set ℝ) 
variable (a b c d : Fin n → ℝ)

-- Condition on any three sets having a common point
axiom common_point_exists (hcomm: ∀ i j k, i ≠ j → j ≠ k → i ≠ k → ∃ x, x ∈ A i ∧ x ∈ A j ∧ x ∈ A k)

-- Prove there exists a point in at least half of the sets
theorem exists_point_in_half_of_sets : ∃ x : ℝ, ∃ S : Finset (Fin n), S.card ≥ n / 2 ∧ (∀ i ∈ S, x ∈ A i) := by
  sorry

end exists_point_in_half_of_sets_l504_504621


namespace math_problem_l504_504939

noncomputable def f (x : ℝ) := Real.logBase 2 x
noncomputable def g (x : ℝ) := x ^ 2 + 2 * x
noncomputable def S (n : ℕ) := n ^ 2 + 2 * n
noncomputable def b (n : ℕ) := 2 ^ n
noncomputable def a (n : ℕ) := S n - S (n - 1)
noncomputable def C (n : ℕ) := 1 / (a n * f (b (2 * n - 1)))

def prob1 : Prop := ∀ n : ℕ, 1 ≤ n → a n = 2 * n + 1
def prob2 : Prop := ∀ n : ℕ, 1 ≤ n → 
  (∑ i in Finset.range n, C (i + 1)) = n / (2 * n + 1)

theorem math_problem : prob1 ∧ prob2 := by
  sorry

end math_problem_l504_504939


namespace angle_A_prime_equiv_l504_504004

theorem angle_A_prime_equiv (ABC A'B'C' : Triangle) (cong : Congruent ABC A'B'C') (h1 : ABC.sideAB = A'B'C'.sideAB) 
(h2 : ABC.angleB = 50) (h3 : A'B'C'.angleC = 70) : A'B'C'.angleA = 60 := 
by sorry

end angle_A_prime_equiv_l504_504004


namespace prove_m_set_l504_504945

-- Define set A
def A : Set ℝ := {x | x^2 - 6*x + 8 = 0}

-- Define set B as dependent on m
def B (m : ℝ) : Set ℝ := {x | m*x - 4 = 0}

-- The main proof statement
theorem prove_m_set : {m : ℝ | B m ∩ A = B m} = {0, 1, 2} :=
by
  -- Code here would prove the above theorem
  sorry

end prove_m_set_l504_504945


namespace triangle_third_side_lengths_product_l504_504742

def hypotenuse (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

def leg (c b : ℝ) : ℝ :=
  real.sqrt (c^2 - b^2)

theorem triangle_third_side_lengths_product :
  let a := 6
  let b := 8
  let hyp := hypotenuse a b
  let leg := leg b a
  real.round (hyp * leg * 10) / 10 = 52.9 :=
by {
  -- Definitions and calculations have been provided in the problem statement
  sorry
}

end triangle_third_side_lengths_product_l504_504742


namespace articles_produced_l504_504968

theorem articles_produced (x y : ℕ) :
  (x * x * x * (1 / (x^2 : ℝ))) = x → (y * y * y * (1 / (x^2 : ℝ))) = (y^3 / x^2 : ℝ) :=
by
  sorry

end articles_produced_l504_504968


namespace prime_count_between_50_and_70_l504_504068

open Nat

theorem prime_count_between_50_and_70 : 
  (finset.filter Nat.prime (finset.range 71 \ finset.range 51).card = 4) := 
begin
  sorry
end

end prime_count_between_50_and_70_l504_504068


namespace kenny_jumps_l504_504135

theorem kenny_jumps (M : ℕ) (h : 34 + M + 0 + 123 + 64 + 23 + 61 = 325) : M = 20 :=
by
  sorry

end kenny_jumps_l504_504135


namespace number_division_remainder_l504_504332

theorem number_division_remainder (N k m : ℤ) (h1 : N = 281 * k + 160) (h2 : N = D * m + 21) : D = 139 :=
by sorry

end number_division_remainder_l504_504332


namespace right_triangle_third_side_product_l504_504726

theorem right_triangle_third_side_product :
  ∀ (a b : ℝ), (a = 6 ∧ b = 8 ∧ (a^2 + b^2 = c^2 ∨ a^2 = b^2 - c^2)) →
  (a * b = 53.0) :=
by
  intros a b h
  sorry

end right_triangle_third_side_product_l504_504726


namespace cost_of_painting_floor_l504_504240

-- Conditions
def length := 15.491933384829668
def breadth := length / 3
def area := length * breadth
def cost_per_sqm := 3
def total_cost := area * cost_per_sqm

-- Proof Statement
theorem cost_of_painting_floor : total_cost = 240 := by
  sorry

end cost_of_painting_floor_l504_504240


namespace rhombus_side_length_l504_504676

theorem rhombus_side_length (d1 d2 : ℕ) (h1 : d1 = 24) (h2 : d2 = 70) : 
  ∃ (a : ℕ), a^2 = (d1 / 2)^2 + (d2 / 2)^2 ∧ a = 37 :=
by
  sorry

end rhombus_side_length_l504_504676


namespace bullfinches_are_50_l504_504265

theorem bullfinches_are_50 :
  ∃ N : ℕ, (N > 50 ∨ N < 50 ∨ N ≥ 1) ∧ (¬(N > 50) ∨ ¬(N < 50) ∨ ¬(N ≥ 1)) ∧
  (N > 50 ∧ ¬(N < 50) ∨ N < 50 ∧ ¬(N > 50) ∨ N ≥ 1 ∧ (¬(N > 50) ∧ ¬(N < 50))) ∧
  N = 50 :=
by
  sorry

end bullfinches_are_50_l504_504265


namespace right_triangle_third_side_product_l504_504729

theorem right_triangle_third_side_product :
  ∀ (a b : ℝ), (a = 6 ∧ b = 8 ∧ (a^2 + b^2 = c^2 ∨ a^2 = b^2 - c^2)) →
  (a * b = 53.0) :=
by
  intros a b h
  sorry

end right_triangle_third_side_product_l504_504729


namespace best_model_fitting_l504_504761

theorem best_model_fitting :
  let r1 := -0.98
  let r2 := 0.80
  let r3 := -0.50
  let r4 := 0.25
  abs r1 > abs r2 ∧ abs r1 > abs r3 ∧ abs r1 > abs r4 → 
  "Model I has the best fitting effect" = "Model I has the best fitting effect" :=
by
  intros
  sorry

end best_model_fitting_l504_504761


namespace prime_count_between_50_and_70_l504_504040

theorem prime_count_between_50_and_70 : 
  (finset.filter nat.prime (finset.range 71).filter (λ n, 50 < n ∧ n < 71)).card = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504040


namespace arithmetic_square_root_of_4_is_2_l504_504212

theorem arithmetic_square_root_of_4_is_2 : sqrt 4 = 2 := 
by
  sorry

end arithmetic_square_root_of_4_is_2_l504_504212


namespace annual_interest_rate_is_correct_l504_504360

-- Define conditions
def principal : ℝ := 900
def finalAmount : ℝ := 992.25
def compoundingPeriods : ℕ := 2
def timeYears : ℕ := 1

-- Compound interest formula
def compound_interest (P A r : ℝ) (n t : ℕ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

-- Statement to prove
theorem annual_interest_rate_is_correct :
  ∃ r : ℝ, compound_interest principal finalAmount r compoundingPeriods timeYears ∧ r = 0.10 :=
by 
  sorry

end annual_interest_rate_is_correct_l504_504360


namespace replace_with_monomial_produces_four_terms_l504_504173

-- Define the initial expression
def initialExpression (k : ℤ) (x : ℤ) : ℤ :=
  ((x^4 - 3)^2 + (x^3 + k)^2)

-- Proof statement
theorem replace_with_monomial_produces_four_terms (x : ℤ) :
  ∃ (k : ℤ), initialExpression k x = (x^8 + x^6 + 9x^2 + 9) :=
  exists.intro (3 * x) sorry

end replace_with_monomial_produces_four_terms_l504_504173


namespace intended_profit_l504_504807

variables (C P : ℝ)

theorem intended_profit (L S : ℝ) (h1 : L = C * (1 + P)) (h2 : S = 0.90 * L) (h3 : S = 1.17 * C) :
  P = 0.3 + 1 / 3 :=
by
  sorry

end intended_profit_l504_504807


namespace find_line_and_area_l504_504018

noncomputable def l1 : (ℝ → ℝ → Prop) := λ x y, 2 * x - 3 * y + 1 = 0
noncomputable def point_l2 : ℝ × ℝ := (1,1)
noncomputable def perpendicular_slope (k : ℝ) : ℝ := -1 / k

theorem find_line_and_area :
  (∃ (a b c : ℝ), a * 1 + b * 1 + c = 0 ∧ 
   a * 0 + b * (5 / 2) + c = 0 ∧
   a * (5 / 3) + b * 0 + c = 0 ∧
   ∀ x y, 2 * x - 3 * y + 1 = 0 → ((3 * x) + (2 * y) - 5 = 0) ∧
   (1 / 2) * (5 / 3) * (5 / 2) = 25 / 12) := sorry

end find_line_and_area_l504_504018


namespace problem_solution_l504_504960

open Real

/-- If (y / 6) / 3 = 6 / (y / 3), then y is ±18. -/
theorem problem_solution (y : ℝ) (h : (y / 6) / 3 = 6 / (y / 3)) : y = 18 ∨ y = -18 :=
by
  sorry

end problem_solution_l504_504960


namespace ratio_is_correct_l504_504201

-- Define the given conditions
def total_amount : ℝ := 782
def first_part : ℝ := 246.95

-- Define the ratio of the first part to the total amount
def ratio (a b : ℝ) := a / b

-- State and formulate the theorem
theorem ratio_is_correct : ratio 246.95 782 = 4939 / 15640 := by
  sorry

end ratio_is_correct_l504_504201


namespace number_of_tests_initially_l504_504132

-- Given conditions
variables (n S : ℕ)
variables (h1 : S / n = 70)
variables (h2 : S = 70 * n)
variables (h3 : (S - 55) / (n - 1) = 75)

-- Prove the number of tests initially, n, is 4.
theorem number_of_tests_initially (n : ℕ) (S : ℕ)
  (h1 : S / n = 70) (h2 : S = 70 * n) (h3 : (S - 55) / (n - 1) = 75) :
  n = 4 :=
sorry

end number_of_tests_initially_l504_504132


namespace farmer_apples_count_l504_504231

-- Definitions from the conditions in step a)
def initial_apples : ℕ := 127
def apples_given_away : ℕ := 88

-- Proof goal from step c)
theorem farmer_apples_count : initial_apples - apples_given_away = 39 :=
by
  sorry

end farmer_apples_count_l504_504231


namespace probability_at_least_one_box_with_four_blocks_of_same_color_l504_504411

theorem probability_at_least_one_box_with_four_blocks_of_same_color :
  (∃ (friends : List (List (Color → Box))),
    friends.length = 4 ∧
    ∀ (f : List (Color → Box)), f ∈ friends → f.length = 6 ) →
  (real.rat.of_int 517 / real.rat.of_int 7776) = (1 - (5 / 6)^6) := 
by
  -- The proof needs to be filled in
  sorry

end probability_at_least_one_box_with_four_blocks_of_same_color_l504_504411


namespace garden_breadth_l504_504977

theorem garden_breadth (P L B : ℕ) (h1 : P = 700) (h2 : L = 250) (h3 : P = 2 * (L + B)) : B = 100 :=
by
  sorry

end garden_breadth_l504_504977


namespace recurring_sum_l504_504389

noncomputable def recurring_to_fraction (a b : ℕ) : ℚ := a / (10 ^ b - 1)

def r1 := recurring_to_fraction 12 2
def r2 := recurring_to_fraction 34 3
def r3 := recurring_to_fraction 567 5

theorem recurring_sum : r1 + r2 + r3 = 16133 / 99999 := by
  sorry

end recurring_sum_l504_504389


namespace sally_received_quarters_l504_504620

theorem sally_received_quarters : 
  ∀ (original_quarters total_quarters received_quarters : ℕ), 
  original_quarters = 760 → 
  total_quarters = 1178 → 
  received_quarters = total_quarters - original_quarters → 
  received_quarters = 418 :=
by 
  intros original_quarters total_quarters received_quarters h_original h_total h_received
  rw [h_original, h_total] at h_received
  exact h_received

end sally_received_quarters_l504_504620


namespace olga_fish_count_at_least_l504_504602

def number_of_fish (yellow blue green : ℕ) : ℕ :=
  yellow + blue + green

theorem olga_fish_count_at_least :
  ∃ (fish_count : ℕ), 
  (∃ (yellow blue green : ℕ), 
       yellow = 12 ∧ blue = yellow / 2 ∧ green = yellow * 2 ∧ fish_count = number_of_fish yellow blue green) ∧
  fish_count = 42 :=
by
  let yellow := 12
  let blue := yellow / 2
  let green := yellow * 2
  let fish_count := number_of_fish yellow blue green
  have h : fish_count = 42 := sorry
  use fish_count, yellow, blue, green
  repeat {constructor}
  assumption
  assumption
  assumption
  assumption
  assumption
  assumption

end olga_fish_count_at_least_l504_504602


namespace drivers_distance_difference_l504_504777

theorem drivers_distance_difference
  (initial_distance : ℕ)
  (driver_A_speed : ℕ)
  (driver_B_speed : ℕ)
  (driver_A_start_time : ℕ)
  (driver_B_start_time : ℕ) :
  initial_distance = 940 →
  driver_A_speed = 90 →
  driver_B_speed = 80 →
  driver_A_start_time = 0 →
  driver_B_start_time = 1 →
  let t := (initial_distance - driver_A_speed) / (driver_A_speed + driver_B_speed) in
  (driver_A_speed * (t + 1) - driver_B_speed * t) = 140 :=
by
  intros h_initial h_A_speed h_B_speed h_A_start h_B_start
  let t := 5
  have h_t : t = (940 - 90) / (90 + 80) := by simp [h_initial, h_A_speed, h_B_speed, t]
  simp [h_t]
  sorry

end drivers_distance_difference_l504_504777


namespace cos_B_of_triangle_l504_504126

theorem cos_B_of_triangle (A B : ℝ) (a b : ℝ) (h1 : A = 2 * B) (h2 : a = 6) (h3 : b = 4) :
  Real.cos B = 3 / 4 :=
by
  sorry

end cos_B_of_triangle_l504_504126


namespace master_craftsman_quota_l504_504531

theorem master_craftsman_quota (N : ℕ) (initial_rate increased_rate : ℕ) (additional_hours extra_hours : ℝ) :
  initial_rate = 35 →
  increased_rate = initial_rate + 15 →
  additional_hours = 0.5 →
  extra_hours = 1 →
  N / initial_rate - N / increased_rate = additional_hours + extra_hours →
  N = 175 →
  (initial_rate + N) = 210 :=
by {
  intros h1 h2 h3 h4 h5 h6,
  rw h6,
  exact rfl,
}

end master_craftsman_quota_l504_504531


namespace train_speed_in_kmh_l504_504813

-- Definitions of the given problem conditions
def train_length_m : ℕ := 500
def time_seconds : ℕ := 10
def speed_conversion_factor : ℝ := 3.6

-- The theorem stating the question and the deduced answer
theorem train_speed_in_kmh : 
  let distance := (train_length_m : ℝ)
  let time := (time_seconds : ℝ)
  let speed_mps := distance / time  -- speed in meters/second
  let speed_kmh := speed_mps * speed_conversion_factor  -- convert to km/hour
  speed_kmh = 180 := 
by {
  -- speed_mps = distance / time
  have h1 : speed_mps = 50 := by sorry,
  -- speed_kmh = speed_mps * speed_conversion_factor
  have h2 : speed_kmh = 50 * speed_conversion_factor := by sorry,
  -- speed_kmh = 50 * 3.6 = 180
  show speed_kmh = 180
  from sorry
}

end train_speed_in_kmh_l504_504813


namespace find_m_l504_504949

theorem find_m (x y m : ℤ) 
  (h1 : x + 2 * y = 5 * m) 
  (h2 : x - 2 * y = 9 * m) 
  (h3 : 3 * x + 2 * y = 19) : 
  m = 1 := 
by 
  sorry

end find_m_l504_504949


namespace master_craftsman_quota_l504_504533

theorem master_craftsman_quota (N : ℕ) (initial_rate increased_rate : ℕ) (additional_hours extra_hours : ℝ) :
  initial_rate = 35 →
  increased_rate = initial_rate + 15 →
  additional_hours = 0.5 →
  extra_hours = 1 →
  N / initial_rate - N / increased_rate = additional_hours + extra_hours →
  N = 175 →
  (initial_rate + N) = 210 :=
by {
  intros h1 h2 h3 h4 h5 h6,
  rw h6,
  exact rfl,
}

end master_craftsman_quota_l504_504533


namespace prime_numbers_between_50_and_70_l504_504052

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_between_50_and_70 : 
  (finset.filter is_prime (finset.range 71)).count (λ n, 50 ≤ n ∧ n ≤ 70) = 4 := 
sorry

end prime_numbers_between_50_and_70_l504_504052


namespace collinear_C_P_Q_l504_504252

variables {Point : Type} [incidence_geometry Point]

-- Definitions for the points in the problem
variables (A B C D N P L K M Q : Point)

-- Conditions of the problem
variables (h_square : square ABCD)
variables (h_division1 : square NPLD)
variables (h_division2 : square KBMP)
variables (h_parallel1 : parallel_side_division ABCD NPLD KBMP)
variables (h_parallel2 : parallel_side_division ABCD KBMP NPLD)
variables (h_P_intersection : intersects_at KL MN P)
variables (h_Q_intersection : intersects_at BN DK Q)

-- The theorem to be proven
theorem collinear_C_P_Q : collinear {C, P, Q} :=
sorry

end collinear_C_P_Q_l504_504252


namespace triangle_area_sum_of_squares_l504_504267

theorem triangle_area_sum_of_squares (ABC Q : Type) (side_parallel_lines_through_Q : Point → Triangle → List Line)
  (A B C Q: Point) [InTriangle Q ABC] 
  (DE FG HI : Line) (S1 S2 S3 : ℝ) (S : ℝ) 
  (h1 : LineParallel DE B C) (h2 : LineParallel FG C A) (h3 : LineParallel HI A B)
  (h4 : Area (Triangle.from_points G D Q) = S1)
  (h5 : Area (Triangle.from_points I E Q) = S2)
  (h6 : Area (Triangle.from_points H F Q) = S3)
  (h7 : Area ABC = S) :
  S = (sqrt S1 + sqrt S2 + sqrt S3) ^ 2 :=
by
  sorry

end triangle_area_sum_of_squares_l504_504267


namespace tangent_circles_l504_504301

-- Definitions of the conditions in the problem
variables {A B C M D H T : Point}
variable {triangle_ABC : Triangle}
variable (A B C M D H T : Point)
variable (triangle_ABC : AcuteAngledTriangle A B C)
variable (M_mid_BC : Midpoint M B C)
variable (AD_perp_BC : Perpendicular AD BC)
variable (H_orthocenter_ABC : Orthocenter H triangle_ABC)
variable (radius_A : Real)
variable (circle_A : Circle A radius_A)
variable (sqrt_AH_AD : radius_A = sqrt (distance A H * distance A D))
variable (extension_MH_inter_T : Intersects (LineSegment M H) circle_A T)

-- Theorem statement proving the tangency of the necessary circles
theorem tangent_circles
  (hcond1 : acute_angled_triangle A B C)
  (hcond2 : midpoint M B C)
  (hcond3 : perpendicular AD BC)
  (hcond4 : orthocenter H triangle_ABC)
  (hcond5 : radius_A = sqrt (distance A H * distance A D))
  (hcond6 : intersects (line_segment M H) (circle.mk A radius_A) T) :
  tangent (circle.mk A radius_A) (circumcircle T B C) :=
sorry

end tangent_circles_l504_504301


namespace beads_probability_l504_504890

/-
  Four red beads, three white beads, and two blue beads are placed in a line in random order.
  Prove that the probability that no two neighboring beads are the same color is 1/70.
-/
theorem beads_probability :
  let total_permutations := Nat.factorial 9 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 2)
  let valid_permutations := 18 -- conservative estimate from the solution
  (valid_permutations : ℚ) / total_permutations = 1 / 70 :=
by
  let total_permutations := Nat.factorial 9 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 2)
  let valid_permutations := 18
  show (valid_permutations : ℚ) / total_permutations = 1 / 70
  -- skipping proof details
  sorry

end beads_probability_l504_504890


namespace problem1_l504_504312

variable (A B C a b c : ℝ)
variable (A_eq : A = 60 * Real.pi / 180) -- converting degrees to radians for Lean
variable (a_eq : a = Real.sqrt 3)
variable (sin_rule : ∀ {x y z : ℝ}, y / sin x = z / sin y)

theorem problem1 (h : 2 * sin A = 1) : (sin A + sin B + sin C) / (a + b + c) = 1 / 2 :=
sorry

end problem1_l504_504312


namespace find_f2014_l504_504931

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (π * x + α) + b * Real.cos (π * x + β)

theorem find_f2014 (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : f 2013 a α b β = -1) :
  f 2014 a α b β = 1 :=
sorry

end find_f2014_l504_504931


namespace total_ice_cream_l504_504820

variable (friday saturday total : ℝ)
variable (h_friday : friday = 3.25)
variable (h_saturday : saturday = 0.25)

theorem total_ice_cream : total = 3.5 :=
by
  have h_add : total = friday + saturday,
  exact eq_add_of_sub_eq h_add,
  rw [h_friday, h_saturday],
  norm_num,
  rfl

end total_ice_cream_l504_504820


namespace simplify_nested_sqrt_l504_504641

-- Define the expressions under the square roots
def expr1 : ℝ := 12 + 8 * real.sqrt 3
def expr2 : ℝ := 12 - 8 * real.sqrt 3

-- Problem statement to prove
theorem simplify_nested_sqrt : real.sqrt expr1 + real.sqrt expr2 = 4 * real.sqrt 2 :=
by
  sorry

end simplify_nested_sqrt_l504_504641


namespace prime_count_between_50_and_70_l504_504041

theorem prime_count_between_50_and_70 : 
  (finset.filter nat.prime (finset.range 71).filter (λ n, 50 < n ∧ n < 71)).card = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504041


namespace largest_smallest_difference_l504_504576

theorem largest_smallest_difference : ∃ (A B C : ℕ), 
    (C = 0 ∨ C = 5) ∧ 
    (14 + A + B + C) % 9 = 0 ∧
    (A > 0 ∧ B > 0 ∧ A < 10 ∧ B < 10) ∧
    ∑ (a, b), A + B = (λ x y, if (C = 0 ∧ x + y = 4) ∨ (C = 0 ∧ x + y = 13) ∨ (C = 5 ∧ x + y = 8) ∨ (C = 5 ∧ x + y = 17) then (× a.x) elif else 0)  85
 

end largest_smallest_difference_l504_504576


namespace combined_total_score_l504_504105

-- Define the conditions
def num_single_answer_questions : ℕ := 50
def num_multiple_answer_questions : ℕ := 20
def single_answer_score : ℕ := 2
def multiple_answer_score : ℕ := 4
def wrong_single_penalty : ℕ := 1
def wrong_multiple_penalty : ℕ := 2
def jose_wrong_single : ℕ := 10
def jose_wrong_multiple : ℕ := 5
def jose_lost_marks : ℕ := (jose_wrong_single * wrong_single_penalty) + (jose_wrong_multiple * wrong_multiple_penalty)
def jose_correct_single : ℕ := num_single_answer_questions - jose_wrong_single
def jose_correct_multiple : ℕ := num_multiple_answer_questions - jose_wrong_multiple
def jose_single_score : ℕ := jose_correct_single * single_answer_score
def jose_multiple_score : ℕ := jose_correct_multiple * multiple_answer_score
def jose_score : ℕ := (jose_single_score + jose_multiple_score) - jose_lost_marks
def alison_score : ℕ := jose_score - 50
def meghan_score : ℕ := jose_score - 30

-- Prove the combined total score
theorem combined_total_score :
  jose_score + alison_score + meghan_score = 280 :=
by
  sorry

end combined_total_score_l504_504105


namespace replace_asterisk_with_monomial_l504_504189

theorem replace_asterisk_with_monomial :
  ∀ (x : ℝ), (x^4 - 3)^2 + (x^3 + 3x)^2 = x^8 + x^6 + 9x^2 + 9 := 
by
  intro x
  calc
    (x^4 - 3)^2 + (x^3 + 3x)^2
        = (x^4)^2 - 2 * x^4 * 3 + 3^2 + (x^3)^2 + 2 * x^3 * 3x + (3x)^2 : by ring
    ... = x^8 - 6 * x^4 + 9 + x^6 + 6 * x^4 + 9 * x^2 : by ring
    ... = x^8 + x^6 + 9 * x^2 + 9 : by ring
  sorry

end replace_asterisk_with_monomial_l504_504189


namespace find_xy_l504_504900

theorem find_xy (x y : ℝ) (h : |x^3 - 1/8| + Real.sqrt (y - 4) = 0) : x * y = 2 :=
by
  sorry

end find_xy_l504_504900


namespace area_decrease_l504_504359

theorem area_decrease (A : ℝ) (s : ℝ) (d : ℝ) (new_s : ℝ) (new_A : ℝ) (A_dec : ℝ) :
  A = 100 * Real.sqrt 3 →
  s = Real.sqrt(400 : ℝ) →
  d = 3 →
  new_s = s - d →
  new_A = (new_s^2 * Real.sqrt 3) / 4 →
  A_dec = A - new_A →
  A_dec = 27.75 * Real.sqrt 3 :=
begin
  sorry
end

end area_decrease_l504_504359


namespace ant_return_probability_after_2006_moves_l504_504838

noncomputable def probability_of_ant_at_start_vertex (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 1 else 1 / 6 * (1 + 1 / 2 ^ (n - 1))

theorem ant_return_probability_after_2006_moves :
  probability_of_ant_at_start_vertex (λ n, 2 * (1 - (-1 / 2) ^ n)) 2006 = 
    (2 ^ 2005 + 1) / (3 * 2 ^ 2006) :=
by sorry

end ant_return_probability_after_2006_moves_l504_504838


namespace proof_problem_l504_504427

open Classical
open Real

-- Definition of the given conditions
def point_on_terminal_side_of_angle (x y : ℝ) (α : ℝ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ tan α = y / x

-- Definition of the trigonometric expression
def trig_expression (α : ℝ) : ℝ :=
  cos (π / 2 + α) * sin (-π - α) / (cos (11 * π / 2 - α) * sin (9 * π / 2 + α))

-- The proof goal statement
theorem proof_problem (α : ℝ) (h : point_on_terminal_side_of_angle (-4) 3 α) :
    trig_expression α = -3 / 4 := by
  sorry

end proof_problem_l504_504427


namespace well_depth_l504_504330

def daily_climb_up : ℕ := 4
def daily_slip_down : ℕ := 3
def total_days : ℕ := 27

theorem well_depth : (daily_climb_up * (total_days - 1) - daily_slip_down * (total_days - 1)) + daily_climb_up = 30 := by
  -- conditions
  let net_daily_progress := daily_climb_up - daily_slip_down
  let net_26_days_progress := net_daily_progress * (total_days - 1)

  -- proof to be completed
  sorry

end well_depth_l504_504330


namespace prime_count_between_50_and_70_l504_504071

open Nat

theorem prime_count_between_50_and_70 : 
  (finset.filter Nat.prime (finset.range 71 \ finset.range 51).card = 4) := 
begin
  sorry
end

end prime_count_between_50_and_70_l504_504071


namespace parallel_lines_necessary_not_sufficient_l504_504461

theorem parallel_lines_necessary_not_sufficient (a b c : ℝ) :
  let l1 := λ x : ℝ, ax + y = 3
  let l2 := λ x : ℝ, x + by - c = 0
  ab = 1 → (l1 ∥ l2 ↔ true) ∧ ¬(l1 ∥ l2 ↔ ab = 1) := 
sorry

end parallel_lines_necessary_not_sufficient_l504_504461


namespace sara_initial_pears_l504_504202

theorem sara_initial_pears (given_to_dan : ℕ) (left_with_sara : ℕ) (total : ℕ) :
  given_to_dan = 28 ∧ left_with_sara = 7 ∧ total = given_to_dan + left_with_sara → total = 35 :=
by
  sorry

end sara_initial_pears_l504_504202


namespace find_larger_number_l504_504678

theorem find_larger_number (L S : ℕ) (h1 : L - S = 2415) (h2 : L = 21 * S + 15) : L = 2535 := 
by
  sorry

end find_larger_number_l504_504678


namespace count_primes_between_fifty_and_seventy_l504_504061

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_fifty_and_seventy : List ℕ :=
  [53, 59, 61, 67]

theorem count_primes_between_fifty_and_seventy :
  (primes_between_fifty_and_seventy.count is_prime = 4) :=
by
  sorry

end count_primes_between_fifty_and_seventy_l504_504061


namespace wire_wraps_around_square_field_15_times_l504_504401

noncomputable def area : ℝ := 69696
noncomputable def wire_length : ℝ := 15840

theorem wire_wraps_around_square_field_15_times (h : area = 69696) (w : wire_length = 15840) :
  (wire_length / (4 * real.sqrt area)) = 15 :=
by
  sorry

end wire_wraps_around_square_field_15_times_l504_504401


namespace roots_sum_l504_504591

-- Let's express our quadratic equation.
def quadratic_equation : ℝ → ℝ := λ x, 4 * x^2 - 9 * x + 4

-- Prove that the roots of the quadratic equation can be written in the specified form and their sum.
theorem roots_sum (m n p : ℕ) (h1 : m = 9) (h2 : n = 17) (h3 : p = 8) :
    (∀ x, quadratic_equation x = 0 → x = (9 + real.sqrt 17) / 8 ∨ x = (9 - real.sqrt 17) / 8) →
    (nat.gcd m n = 1) →
    (nat.gcd m p = 1) →
    (nat.gcd n p = 1) →
    m + n + p = 34 :=
by
  intros
  rw [h1, h2, h3]
  norm_num
  sorry

end roots_sum_l504_504591


namespace prime_count_between_50_and_70_l504_504043

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def primes_in_range (a b : ℕ) : List ℕ :=
  List.filter is_prime (List.range' a (b - a + 1))

theorem prime_count_between_50_and_70 : primes_in_range 50 70 = [53, 59, 61, 67] :=
sorry

end prime_count_between_50_and_70_l504_504043


namespace max_stamps_l504_504487

theorem max_stamps (price_per_stamp : ℕ) (total_money : ℕ) (h1 : price_per_stamp = 45) (h2 : total_money = 5000) : ∃ n : ℕ, n = 111 ∧ 45 * n ≤ 5000 ∧ ∀ m : ℕ, (45 * m ≤ 5000) → m ≤ n := 
by
  sorry

end max_stamps_l504_504487


namespace total_area_rectangles_l504_504169

theorem total_area_rectangles :
  let R0 : ℝ := 3 * 4,
      R1 : ℝ := (1 / 2) * R0,
      R2 : ℝ := (1 / 2) * R1,
      R3 : ℝ := (1 / 2) * R2
  in R0 + R1 + R2 + R3 = 30 :=
by
  let R0 : ℝ := 3 * 4
  let R1 : ℝ := (1 / 2) * R0
  let R2 : ℝ := (1 / 2) * R1
  let R3 : ℝ := (1 / 2) * R2
  sorry

end total_area_rectangles_l504_504169


namespace number_of_distinct_products_l504_504033

def distinct_products (A : Finset ℕ) : Finset ℕ :=
  (Finset.powerset A).filter (λ s, 2 ≤ s.card).image (λ s, s.prod id)

theorem number_of_distinct_products : 
  distinct_products (Finset.of_list [1, 2, 3, 5, 11]).card = 15 :=
by sorry

end number_of_distinct_products_l504_504033


namespace integral_equals_expected_l504_504370

noncomputable def integral_expr (x : ℝ) : ℝ :=
  ∫ (t : ℝ) in 0..x, ((t^3 - 6*t^2 + 14*t - 4) / ((t - 2) * (t + 2)^3))

noncomputable def expected_expr (x : ℝ) : ℝ :=
  (1 / 8) * log (abs (x - 2)) + (7 / 8) * log (abs (x + 2)) + (17 * x + 18) / (2 * (x + 2)^2)

theorem integral_equals_expected (C : ℝ) :
  ∃ (f : ℝ → ℝ), (∀ x, integral_expr x = expected_expr x + f C) := by
  sorry

end integral_equals_expected_l504_504370


namespace function_value_l504_504098

noncomputable def f (x : ℝ) : ℝ :=
  if x % 2 = 0 then 0 else
  if 0 < x ∧ x < 1 then 4^x else
  -f (-x)

theorem function_value :
  f (-5 / 2) + f 2 = -2 :=
by
  sorry

end function_value_l504_504098


namespace hyperbola_triangle_area_proof_l504_504940

noncomputable def hyperbola_area (x y : ℝ) : Prop :=
  (x^2 / 9 - y^2 / 16 = 1) ∧
  let a := 3 in 
  let b := 4 in
  let c := Real.sqrt (a^2 + b^2) in
  let F1 := (-c, 0) in
  let F2 := (c, 0) in
  ∀ P : ℝ × ℝ, 
    P = (x, y) → 
    (P.1^2 / 9 - P.2^2 / 16 = 1) →
    (Real.angle F1 P F2 = Real.pi / 2) →
    (Real.dist P F1) * (Real.dist P F2) = 32 →
    let S := (1 / 2) * (Real.dist P F1) * (Real.dist P F2) in
    S = 16

theorem hyperbola_triangle_area_proof (x y : ℝ) : 
  hyperbola_area x y :=
sorry

end hyperbola_triangle_area_proof_l504_504940


namespace perimeter_of_square_garden_eq_40sqrt2_l504_504211

-- Define the conditions
def side_of_square (area : ℝ) := real.sqrt area

-- Define the given condition
def area_of_garden : ℝ := 200

-- Define the solution requirement
def perimeter_of_garden (side_length : ℝ) := 4 * side_length

-- State the theorem
theorem perimeter_of_square_garden_eq_40sqrt2 : perimeter_of_garden (side_of_square area_of_garden) = 40 * real.sqrt 2 :=
begin
  sorry
end

end perimeter_of_square_garden_eq_40sqrt2_l504_504211


namespace part_I_part_II_l504_504446

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - 6 * a * x^2

-- Part (I): Equation of the tangent line
theorem part_I (a : ℝ) (ha : a = -1) :
  let k := deriv (f 1 a),
      y1 := f 1 a in
  (15 * (1 : ℝ) - y1 - 8 = 0) :=
by
  sorry

-- Part (II): Monotonicity discussion
theorem part_II (a : ℝ) :
  (a = 0 → ∀ x y, x ≤ y → f x a ≤ f y a) ∧
  (a < 0 → ∀ x y, 
    ((x < 4 * a ∧ y < 4 * a) ∨ (x > 0 ∧ y > 0)) ∧ x ≤ y → f x a ≤ f y a) ∧
  (a > 0 → ∀ x y, 
    ((x < 0 ∧ y < 0) ∨ (x > 4 * a ∧ y > 4 * a)) ∧ x ≤ y → f x a ≤ f y a) :=
by
  sorry

end part_I_part_II_l504_504446


namespace correct_cell_description_l504_504224

noncomputable def cell_state := 
  ∃ (cell : Type) 
    (from_testes : cell → Prop)
    (colorblind_patient : cell → Prop)
    (late_stage_meiosis_ii : cell → Prop)
    (contains_46_chromosomes : cell → Prop)
    (contains_44_autosomes : cell → Prop)
    (contains_2_x_chromosomes : cell → Prop)
    (contains_2_colorblind_genes : cell → Prop), 
    from_testes cell ∧ 
    colorblind_patient cell ∧ 
    late_stage_meiosis_ii cell ∧ 
    contains_46_chromosomes cell ∧ 
    contains_44_autosomes cell ∧ 
    contains_2_x_chromosomes cell ∧ 
    contains_2_colorblind_genes cell

theorem correct_cell_description : cell_state := 
  sorry

end correct_cell_description_l504_504224


namespace acute_triangle_conditions_l504_504762

-- Definitions exclusively from the conditions provided.
def condition_A (AB AC : ℝ) : Prop :=
  AB * AC > 0

def condition_B (sinA sinB sinC : ℝ) : Prop :=
  sinA / sinB = 4 / 5 ∧ sinA / sinC = 4 / 6 ∧ sinB / sinC = 5 / 6

def condition_C (cosA cosB cosC : ℝ) : Prop :=
  cosA * cosB * cosC > 0

def condition_D (tanA tanB : ℝ) : Prop :=
  tanA * tanB = 2

-- Prove which conditions guarantee that triangle ABC is acute.
theorem acute_triangle_conditions (AB AC sinA sinB sinC cosA cosB cosC tanA tanB : ℝ) :
  (condition_B sinA sinB sinC ∨ condition_C cosA cosB cosC ∨ condition_D tanA tanB) →
  (∀ (A B C : ℝ), A < π / 2 ∧ B < π / 2 ∧ C < π / 2) :=
sorry

end acute_triangle_conditions_l504_504762


namespace replace_asterisk_with_monomial_l504_504191

theorem replace_asterisk_with_monomial :
  ∀ (x : ℝ), (x^4 - 3)^2 + (x^3 + 3x)^2 = x^8 + x^6 + 9x^2 + 9 := 
by
  intro x
  calc
    (x^4 - 3)^2 + (x^3 + 3x)^2
        = (x^4)^2 - 2 * x^4 * 3 + 3^2 + (x^3)^2 + 2 * x^3 * 3x + (3x)^2 : by ring
    ... = x^8 - 6 * x^4 + 9 + x^6 + 6 * x^4 + 9 * x^2 : by ring
    ... = x^8 + x^6 + 9 * x^2 + 9 : by ring
  sorry

end replace_asterisk_with_monomial_l504_504191


namespace length_of_diagonal_is_17_l504_504410

noncomputable def num_whole_number_lengths : ℕ :=
let possible_lengths := {x // 5 ≤ x ∧ x ≤ 21} in
Finset.card (Finset.filter (λ x, true) (Finset.range 22)) - Finset.filter (λ b, b < 5) (Finset.range 22)

theorem length_of_diagonal_is_17 :
  num_whole_number_lengths = 17 :=
by
  unfold num_whole_number_lengths
  simp
  sorry

end length_of_diagonal_is_17_l504_504410


namespace replace_star_with_3x_l504_504182

theorem replace_star_with_3x (x : ℝ) :
  (x^4 - 3)^2 + (x^3 + 3x)^2 = x^8 + x^6 + 9x^2 + 9 :=
by
  sorry

end replace_star_with_3x_l504_504182


namespace seq_an_identity_l504_504545

theorem seq_an_identity (n : ℕ) (a : ℕ → ℕ) 
  (h₁ : a 1 = 1)
  (h₂ : ∀ n, a (n + 1) > a n)
  (h₃ : ∀ n, a (n + 1)^2 + a n^2 + 1 = 2 * (a (n + 1) * a n + a (n + 1) + a n)) 
  : a n = n^2 := sorry

end seq_an_identity_l504_504545


namespace ones_digit_of_9_pow_47_l504_504754

theorem ones_digit_of_9_pow_47 : (9 ^ 47) % 10 = 9 := 
by
  sorry

end ones_digit_of_9_pow_47_l504_504754


namespace sequence_arithmetic_l504_504457

theorem sequence_arithmetic (a : ℕ → Real)
    (h₁ : a 3 = 2)
    (h₂ : a 7 = 1)
    (h₃ : ∃ d, ∀ n, 1 / (1 + a (n + 1)) = 1 / (1 + a n) + d):
    a 11 = 1 / 2 := by
  sorry

end sequence_arithmetic_l504_504457


namespace right_triangle_side_product_l504_504724

theorem right_triangle_side_product :
  let a := 6
  let b := 8
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let other_leg := Real.sqrt (b^2 - a^2)
  (hypotenuse * 2 * Real.sqrt 7).round = 53 := -- using 53 to consider rounding to the nearest tenth

by
  let a := 6
  let b := 8
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let other_leg := Real.sqrt (b^2 - a^2)
  have h1 : hypotenuse = 10 := by sorry
  have h2 : other_leg = 2 * Real.sqrt 7 := by sorry
  have h_prod : (hypotenuse * 2 * Real.sqrt 7).round = 53 := by sorry
  exact h_prod

end right_triangle_side_product_l504_504724


namespace constant_term_expansion_eq_24_l504_504216

theorem constant_term_expansion_eq_24 :
  let a := (2 : ℝ) * X
  let b := (1 : ℝ) / X
  let n := 4
  (∀ X : ℝ, (2 * X + 1 / X)^4).constant_term = 24 :=
by
  sorry

end constant_term_expansion_eq_24_l504_504216


namespace maximum_rectangle_area_l504_504804

theorem maximum_rectangle_area (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 44) : 
  ∃ (l_max w_max : ℕ), l_max * w_max = 121 :=
by
  sorry

end maximum_rectangle_area_l504_504804


namespace stratified_sampling_third_grade_l504_504492

theorem stratified_sampling_third_grade
  (n1 n2 n3 : ℕ) (s : ℕ)
  (h1 : n1 = 300)
  (h2 : n2 = 300)
  (h3 : n3 = 400)
  (h4 : s = 40) :
  let total := n1 + n2 + n3,
      proportion := n3 / total.to_rat,
      k := (proportion * s).to_nat
  in k = 16 := by
  sorry

end stratified_sampling_third_grade_l504_504492


namespace sum_ninth_row_yanghui_triangle_l504_504835

theorem sum_ninth_row_yanghui_triangle : 
  let ninth_row_sum := (∑ k in finset.range (9 + 1), nat.choose 9 k)
  in ninth_row_sum = 2^8 :=
by
  -- Definitions translating the given conditions
  let yanghui_row := λ n : ℕ, (finset.range (n + 1)).map (nat.choose n),
      sum_elements := λ n : ℕ, (∑ k in finset.range (n + 1), nat.choose n k)

  -- State the theorem to be proved
  have ninth_row_sum := sum_elements 9
  
  -- Apply known formula: sum of the elements of the n-th row is 2^n
  -- Proving ninth_row_sum = 2^8
  exact sorry -- Proof will go here

end sum_ninth_row_yanghui_triangle_l504_504835


namespace kim_total_time_away_l504_504557

noncomputable def total_time_away (d : ℝ) (detour_percentage : ℝ) (time_at_friends : ℝ) (speed : ℝ) : ℝ :=
  let detour_distance := d * detour_percentage
  let total_return_distance := d + detour_distance
  let total_distance := d + total_return_distance
  let driving_time := total_distance / speed
  driving_time + time_at_friends

theorem kim_total_time_away :
  total_time_away 30 0.2 (30 / 60) 44 = 2 :=
by
  delta total_time_away -- unfold the definition of total_time_away
  simp only [div_eq_mul_inv]
  norm_num
  sorry

end kim_total_time_away_l504_504557


namespace replace_star_with_3x_l504_504180

theorem replace_star_with_3x (x : ℝ) :
  (x^4 - 3)^2 + (x^3 + 3x)^2 = x^8 + x^6 + 9x^2 + 9 :=
by
  sorry

end replace_star_with_3x_l504_504180


namespace sum_of_squares_of_roots_l504_504692

theorem sum_of_squares_of_roots :
  (∃ (x₁ x₂ : ℝ), 5 * x₁^2 + 3 * x₁ - 7 = 0 ∧ 5 * x₂^2 + 3 * x₂ - 7 = 0 ∧ x₁ ≠ x₂) →
  (∃ (x₁ x₂ : ℝ), 5 * x₁^2 + 3 * x₁ - 7 = 0 ∧ 5 * x₂^2 + 3 * x₂ - 7 = 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + x₂^2 = 79 / 25) :=
by
  sorry

end sum_of_squares_of_roots_l504_504692


namespace smallest_sum_twice_perfect_square_l504_504253

-- Definitions based directly on conditions:
def sum_of_20_consecutive_integers (n : ℕ) : ℕ := (2 * n + 19) * 10

def twice_perfect_square (x : ℕ) : Prop := ∃ m : ℕ, x = 2 * m^2

-- Proposition to prove the smallest possible value satisfying these conditions:
theorem smallest_sum_twice_perfect_square : 
  ∃ n S, S = sum_of_20_consecutive_integers n ∧ twice_perfect_square S ∧ S = 450 :=
begin
  sorry
end

end smallest_sum_twice_perfect_square_l504_504253


namespace simplify_sqrt_expression_is_correct_l504_504626

-- Definition for the given problem
def simplify_sqrt_expression (a b : ℝ) :=
  a = Real.sqrt (12 + 8 * Real.sqrt 3) → 
  b = Real.sqrt (12 - 8 * Real.sqrt 3) → 
  a + b = 4 * Real.sqrt 3

-- The theorem to be proven
theorem simplify_sqrt_expression_is_correct : simplify_sqrt_expression :=
begin
  intros a b ha hb,
  rw [ha, hb],
  -- Step-by-step simplification approach would occur here
  sorry
end

end simplify_sqrt_expression_is_correct_l504_504626


namespace number_of_symmetric_scanning_codes_l504_504805

-- Define the grid size
def grid_size := 5

-- Define the conditions for the scanning codes
structure ScanningCode (n : Nat) :=
  (grid : Array (Array Bool)) -- Bool represents black or white
  (size : Array.size grid = n ∧ ∀ row in grid, Array.size row = n)
  (contains_both_colors : grid.exists (fun row => row.any id) ∧ grid.exists (fun row => row.any not)) -- at least one black and one white
  (symmetric : ∀ i j, grid[i][j] = grid[j][i] ∧ grid[grid_size - 1 - i][grid_size - 1 - j] = grid[j][i]) -- symmetry conditions

-- Theorem statement
theorem number_of_symmetric_scanning_codes : 
  ∃ (count : Nat), count = 30 ∧ 
  ∀ code : ScanningCode grid_size, code.contains_both_colors → code.symmetric := sorry

end number_of_symmetric_scanning_codes_l504_504805


namespace ratio_shaded_pentagon_normalized_l504_504170

-- Define the given conditions
def rectangle (l w : ℕ) := l * w

-- Define the areas of the given rectangles
def rectangle_ABCD : ℕ := rectangle 2 1
def rectangle_EFGH : ℕ := rectangle 3 1
def rectangle_GHIJ : ℕ := rectangle 2 1

-- Total area of all rectangles
def total_area : ℕ := rectangle_ABCD + rectangle_EFGH + rectangle_GHIJ

-- Area of shaded pentagon (based on problem condition, assume correctly estimated as half)
def shaded_pentagon_area : ℕ := 7 / 2

-- The proof statement
theorem ratio_shaded_pentagon_normalized :
  shaded_pentagon_area.to_rat / total_area.to_rat = (1 : ℚ) / 2 := by
  sorry -- Proof not required as per instructions

end ratio_shaded_pentagon_normalized_l504_504170


namespace inequality_xy_l504_504582

-- Defining the constants and conditions
variables {x y : ℝ}

-- Main theorem to prove the inequality and find pairs for equality
theorem inequality_xy (h : (x + 1) * (y + 2) = 8) :
  (xy - 10)^2 ≥ 64 ∧ ((xy - 10)^2 = 64 → (x, y) = (1, 2) ∨ (x, y) = (-3, -6)) :=
sorry

end inequality_xy_l504_504582


namespace general_form_of_curve_minimum_distance_midpoint_to_line_l504_504019

-- Conditions
def parametric_curve (a : ℝ) : ℝ × ℝ := (sqrt 3 * cos a, sin a)
def point_P : ℝ × ℝ := (0, 2)
def polar_line (p θ : ℝ) : Prop := p * cos θ + √3 * p * sin θ + 2 * √3 = 0

-- Questions
theorem general_form_of_curve :
  ∀ x y a, (x = sqrt 3 * cos a ∧ y = sin a) → (x^2 / 3 + y^2 = 1) :=
by
  sorry

theorem minimum_distance_midpoint_to_line :
  ∀ α : ℝ, let Q := (sqrt 3 * cos α, sin α),
               M := (sqrt 3 / 2 * cos α, 1 + 1 / 2 * sin α),
               A := 1, B := sqrt 3, C := 2 * sqrt 3 in
           (| A * (sqrt 3 / 2 * cos α) + B * (1 + 1 / 2 * sin α) + C | / sqrt (A^2 + B^2)) = (6 * sqrt 3 - sqrt 6) / 4 :=
by
  sorry

end general_form_of_curve_minimum_distance_midpoint_to_line_l504_504019


namespace sum_of_products_leq_one_fourth_l504_504310

-- Define the problem conditions and the statement to prove
theorem sum_of_products_leq_one_fourth (n : ℕ) (x : ℕ → ℝ)
    (h1 : n = 24)
    (h2 : ∀ i, 0 ≤ x i)
    (h3 : (Finset.range n).sum x = 1) :
  (Finset.range n).sum (λ i, x i * x ((i + 1) % n)) ≤ 1 / 4 :=
sorry

end sum_of_products_leq_one_fourth_l504_504310


namespace jason_safe_combination_count_l504_504550

theorem jason_safe_combination_count : 
  let digits := {1, 2, 3, 4, 5, 6}
  let even_digits := {2, 4, 6}
  let odd_digits := {1, 3, 5}
  let valid_combination (c : List ℕ) := 
    c.length = 5 ∧ 
    ∀ i, i < 4 → ((c.nth i ∈ even_digits) → (c.nth (i+1) ∈ odd_digits)) ∧ 
              ((c.nth i ∈ odd_digits) → (c.nth (i+1) ∈ even_digits))
  (∃ c, c ∈ List.replicate 5 digits ∧ valid_combination c).card = 486 := 
sorry

end jason_safe_combination_count_l504_504550


namespace function_identity_l504_504151

theorem function_identity (f : ℕ → ℕ) :
  (∀ m n, f(m) + f(n) ∣ m + n) ↔ (∀ m, f(m) = m) := by
  sorry

end function_identity_l504_504151


namespace min_value_ellipse_min_frac_value_l504_504148

def on_ellipse (x y : ℝ) := (x^2 / 16) + (y^2 / 12) = 1 ∧ 0 < x ∧ 0 < y

theorem min_value_ellipse (x y : ℝ) (h : on_ellipse x y) : ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ (x = 4 * Real.cos α) ∧ (y = 2 * Real.sqrt 3 * Real.sin α) ∧ (∀ α : ℝ, 0 < α ∧ α < π / 2 → (4 / (4 - 4 * Real.cos α) + 18 / (6 - 2 * Real.sqrt 3 * Real.sin α)) ≥ 8) := 
by
  sorry

theorem min_frac_value (x y : ℝ) (h : on_ellipse x y) : (∃ c d : ℝ, 0 < x ∧ 0 < y ∧ (x = 4 * Real.cos c) ∧ (y = 2 * Real.sqrt 3 * Real.sin d) ∧ 0 < c ∧ c < π / 2 ∧ 0 < d ∧ d < π / 2 ∧ (4 / (4 - 4 * Real.cos c) + 18 / (6 - 2 * Real.sqrt 3 * Real.sin d) = 8))
  := 
by
  sorry
  
lemma min_frac (P : ℝ × ℝ) (hx : on_ellipse P.1 P.2) : 
  ∃ z : ℝ, z = 4 := 
by
  exactExists.elim h (fun a ha => exists.intro (4 / (4 - 4 * Real.cos a) + 18 / (6 - 2 * Real.sqrt 3 * Real.sin a)) 
lemma min_frac_interim (a : ℝ) (ha : 0 < a ∧ a < π / 2) : 
  4 / (4 - 4 * Real.cos a) + 18 / (6 - 2 * Real.sqrt 3 * Real.sin a) = 8 := 
by
  exact sorry

end min_value_ellipse_min_frac_value_l504_504148


namespace trajectory_of_moving_circle_l504_504421

-- Define the conditions
def passes_through (M : ℝ × ℝ) (A : ℝ × ℝ) : Prop :=
  M = A

def tangent_to_line (M : ℝ × ℝ) (l : ℝ) : Prop :=
  M.1 = -l

noncomputable def equation_of_trajectory (M : ℝ × ℝ) : Prop :=
  M.2 ^ 2 = 12 * M.1

theorem trajectory_of_moving_circle 
  (M : ℝ × ℝ)
  (A : ℝ × ℝ)
  (l : ℝ)
  (h1 : passes_through M (3, 0))
  (h2 : tangent_to_line M 3)
  : equation_of_trajectory M := 
sorry

end trajectory_of_moving_circle_l504_504421


namespace monotonically_increasing_interval_l504_504235

noncomputable def f (x : ℝ) : ℝ := real.sqrt (4 + 3 * x - x^2)

theorem monotonically_increasing_interval :
  ∀ x y : ℝ, -1 ≤ x ∧ x ≤ (3 / 2) ∧ x ≤ y ∧ y ≤ (3 / 2) → f x ≤ f y := by
  sorry

end monotonically_increasing_interval_l504_504235


namespace prime_count_between_50_and_70_l504_504086

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_50_and_70 : List ℕ :=
  [53, 59, 61, 67]

theorem prime_count_between_50_and_70 : 
  (primes_between_50_and_70.filter is_prime).length = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504086


namespace cos_alpha_minus_pi_over_4_l504_504904

theorem cos_alpha_minus_pi_over_4 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (h_tan : Real.tan α = 2) :
  Real.cos (α - π / 4) = (3 * Real.sqrt 10) / 10 := 
  sorry

end cos_alpha_minus_pi_over_4_l504_504904


namespace sequence_an_square_l504_504543

theorem sequence_an_square (a : ℕ → ℝ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, a (n + 1) > a n) 
  (h3 : ∀ n : ℕ, a (n + 1)^2 + a n^2 + 1 = 2 * (a (n + 1) * a n + a (n + 1) + a n)) :
  ∀ n : ℕ, a n = n^2 :=
by
  sorry

end sequence_an_square_l504_504543


namespace thirdRowIs4213_l504_504395

def isLatinSquare (grid : list (list ℕ)) : Prop :=
  (∀ row, row ∈ grid → (∀ n ∈ row, n ∈ [1, 2, 3, 4])) ∧
  (∀ row, row ∈ grid → (∀ n, n ∈ [1, 2, 3, 4] → n ∈ row)) ∧
  (∀ col, col ∈ (list.transpose grid) → (∀ n ∈ col, n ∈ [1, 2, 3, 4])) ∧
  (∀ col, col ∈ (list.transpose grid) → (∀ n, n ∈ [1, 2, 3, 4] → n ∈ col))

def gridSatisfiesExternalConstraints : Prop := 
  -- Define constraints based on external numbers
  sorry

theorem thirdRowIs4213 (grid : list (list ℕ))
  (h_latin : isLatinSquare grid)
  (h_constraints : gridSatisfiesExternalConstraints grid)
  (h_structure : grid = [[3, 1, 2, 4], [1, 4, 3, 2], [4, 2, 1, 3], [2, 3, 4, 1]]) :
  (grid.nth 2).get = [4, 2, 1, 3] :=
begin
  -- Proof steps go here, but are omitted as per the instructions
  sorry
end

end thirdRowIs4213_l504_504395


namespace sample_quantities_and_probability_l504_504819

-- Define the given quantities from each workshop
def q_A := 10
def q_B := 20
def q_C := 30

-- Total sample size
def n := 6

-- Given conditions, the total quantity and sample ratio
def total_quantity := q_A + q_B + q_C
def ratio := n / total_quantity

-- Derived quantities in the samples based on the proportion
def sample_A := q_A * ratio
def sample_B := q_B * ratio
def sample_C := q_C * ratio

-- Combinatorial calculations
def C (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def total_combinations := C 6 2
def workshop_C_combinations := C 3 2
def probability_C_samples := workshop_C_combinations / total_combinations

-- Theorem to prove the quantities and probability
theorem sample_quantities_and_probability :
  sample_A = 1 ∧ sample_B = 2 ∧ sample_C = 3 ∧ probability_C_samples = 1 / 5 :=
by
  sorry

end sample_quantities_and_probability_l504_504819


namespace age_of_youngest_child_l504_504299

theorem age_of_youngest_child (x : ℕ) (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 :=
by
  sorry

end age_of_youngest_child_l504_504299


namespace master_craftsman_quota_l504_504530

theorem master_craftsman_quota (N : ℕ) (initial_rate increased_rate : ℕ) (additional_hours extra_hours : ℝ) :
  initial_rate = 35 →
  increased_rate = initial_rate + 15 →
  additional_hours = 0.5 →
  extra_hours = 1 →
  N / initial_rate - N / increased_rate = additional_hours + extra_hours →
  N = 175 →
  (initial_rate + N) = 210 :=
by {
  intros h1 h2 h3 h4 h5 h6,
  rw h6,
  exact rfl,
}

end master_craftsman_quota_l504_504530


namespace initial_price_of_speakers_l504_504953

theorem initial_price_of_speakers (price_paid : ℝ) (discount_saved : ℝ) (initial_price : ℝ) : 
  price_paid = 199 → 
  discount_saved = 276 → 
  initial_price = price_paid + discount_saved → 
  initial_price = 475 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end initial_price_of_speakers_l504_504953


namespace intersect_setA_setB_l504_504024

def setA : Set ℝ := {x | x < 2}
def setB : Set ℝ := {x | 3 - 2 * x > 0}

theorem intersect_setA_setB :
  setA ∩ setB = {x | x < 3 / 2} :=
by
  -- proof goes here
  sorry

end intersect_setA_setB_l504_504024


namespace gold_coins_percentage_l504_504361

-- Define the percentage of beads, rings, and silver coins in the urn.
variables (percentage_beads percentage_rings percentage_silver_coins : ℝ)

-- Given conditions:
-- 1. 30% of the objects are beads 
-- 2. 10% of the objects are rings
-- 3. 35% of the coins are silver
axiom beads_condition : percentage_beads = 0.30
axiom rings_condition : percentage_rings = 0.10
axiom silver_coins_condition : percentage_silver_coins = 0.35

-- Prove that 39% of the objects are gold coins
theorem gold_coins_percentage :
  let percentage_coins := 1 - percentage_beads - percentage_rings in
  let percentage_gold_coins := (1 - percentage_silver_coins) * percentage_coins in
  percentage_gold_coins = 0.39 :=
by
  let percentage_coins := 1 - percentage_beads - percentage_rings
  let percentage_gold_coins := (1 - percentage_silver_coins) * percentage_coins
  have h1 : percentage_coins = 1 - 0.30 - 0.10 := by rw [beads_condition, rings_condition]
  have h2 : percentage_coins = 0.60 := by norm_num [h1]
  have h3 : percentage_gold_coins = (1 - 0.35) * 0.60 := by rw [silver_coins_condition, h2]
  have h4 : percentage_gold_coins = 0.65 * 0.60 := by norm_num [h3]
  have h5 : percentage_gold_coins = 0.39 := by norm_num [h4]
  exact h5

end gold_coins_percentage_l504_504361


namespace tv_weight_difference_l504_504366

-- Definitions for the given conditions
def bill_tv_length : ℕ := 48
def bill_tv_width : ℕ := 100
def bob_tv_length : ℕ := 70
def bob_tv_width : ℕ := 60
def weight_per_square_inch : ℕ := 4
def ounces_per_pound : ℕ := 16

-- The statement to prove
theorem tv_weight_difference : (bill_tv_length * bill_tv_width * weight_per_square_inch)
                               - (bob_tv_length * bob_tv_width * weight_per_square_inch)
                               = 150 * ounces_per_pound := by
  sorry

end tv_weight_difference_l504_504366


namespace single_elimination_games_l504_504811

theorem single_elimination_games (n : Nat) (h : n = 21) : games_needed = n - 1 :=
by
  sorry

end single_elimination_games_l504_504811


namespace tangent_line_eq_interval_min_value_l504_504934

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a
noncomputable def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

theorem tangent_line_eq (x y : ℝ) (h : y = f x (-10)) : x = 2 → (9 * x - y = 6) :=
  by
    intro h2
    rw [h2] at h
    sorry

theorem interval_min_value (a : ℝ) (h_max : ∃ x ∈ Icc (-2 : ℝ) (2 : ℝ), f x a = 20 ) :
  ∃ y ∈ Icc (-2 : ℝ) (2 : ℝ), f y a = -7 :=
  by
    sorry

end tangent_line_eq_interval_min_value_l504_504934


namespace correct_calculation_l504_504711

theorem correct_calculation :
  ∃ x : ℝ, (63 / x = 9) → (36 - x = 29) :=
by
  existsi (63 / 9)
  intros h
  rw [h]
  norm_num
  have hx : 63 / 9 = 7 := by norm_num
  rw [hx]
  norm_num

end correct_calculation_l504_504711


namespace percentage_game_of_thrones_l504_504892

def cheat_votes (original: ℕ) (percent: ℚ) : ℕ :=
  original - (original * percent).to_nat

def switch_votes (original: ℕ) (percent: ℚ) : ℕ :=
  (original * percent).to_nat

theorem percentage_game_of_thrones:
  let original_game_of_thrones := 10
  let original_twilight := 12
  let original_war_and_peace := 15
  let original_moby_dick := 25

  let moby_dick_remaining := cheat_votes original_moby_dick (4/5)
  let war_and_peace_switched := switch_votes original_war_and_peace (65/100)
  let twilight_remaining := cheat_votes original_twilight (1/2)
  let twilight_switched := original_twilight / 2

  let game_of_thrones_total := original_game_of_thrones + war_and_peace_switched + twilight_switched

  let total_votes := game_of_thrones_total + moby_dick_remaining + (original_war_and_peace - war_and_peace_switched) + twilight_remaining

  (game_of_thrones_total : ℚ) / total_votes * 100 ≈ 72.22 :=
by sorry

end percentage_game_of_thrones_l504_504892


namespace count_sequences_l504_504470

def is_binary_sequence (a : Fin 8 → ℕ) : Prop :=
  ∀ i, a i = 0 ∨ a i = 1

def sum_adjacent_products (a : Fin 8 → ℕ) : ℕ :=
  ∑ i in (Finset.range 7), a i * a (i+1)

theorem count_sequences : 
  (Finset.card (Finset.filter (λ a : Fin 8 → ℕ, is_binary_sequence a ∧ sum_adjacent_products a = 5) (Finset.univ : Finset (Fin 8 → ℕ)))) = 9 :=
by
  sorry

end count_sequences_l504_504470


namespace correct_proposition_l504_504565

variable (a b : ℝ)
variable (a_nonzero : a ≠ 0)
variable (b_nonzero : b ≠ 0)
variable (a_gt_b : a > b)

theorem correct_proposition : 1 / (a * b^2) > 1 / (a^2 * b) :=
sorry

end correct_proposition_l504_504565


namespace simplify_sqrt_sum_l504_504645

noncomputable def sqrt_expr_1 : ℝ := Real.sqrt (12 + 8 * Real.sqrt 3)
noncomputable def sqrt_expr_2 : ℝ := Real.sqrt (12 - 8 * Real.sqrt 3)

theorem simplify_sqrt_sum : sqrt_expr_1 + sqrt_expr_2 = 4 * Real.sqrt 3 := by
  sorry

end simplify_sqrt_sum_l504_504645


namespace partial_fraction_decomposition_product_correct_l504_504371

noncomputable def partial_fraction_decomposition_product : ℚ :=
let A := (4 : ℚ) in
let B := (-3 / 2 : ℚ) in
let C := (-3 / 2 : ℚ) in
A * B * C

theorem partial_fraction_decomposition_product_correct :
  partial_fraction_decomposition_product = (9 : ℚ) :=
by
  -- Using the calculated values of A, B, and C from the conditions
  have A : ℚ := 4
  have B : ℚ := -3 / 2
  have C : ℚ := -3 / 2
  show A * B * C = 9 by
    simp [A, B, C]
    norm_num
    sorry -- Complete the proof

end partial_fraction_decomposition_product_correct_l504_504371


namespace reuleaux_cannot_rotate_in_hexagon_reuleaux_cannot_rotate_in_triangle_l504_504611

-- Definitions required for the problem
noncomputable def ReuleauxShape (a r : ℝ) := sorry -- Define the Reuleaux shape
noncomputable def Hexagon (side_length : ℝ) := sorry -- Define the regular hexagon
noncomputable def Triangle (side_length : ℝ) := sorry -- Define the regular triangle

variables (a r : ℝ)
axiom h1 : (a > 0) (r > 0)

-- The first part: Reuleaux shape cannot be rotated inside a hexagon
theorem reuleaux_cannot_rotate_in_hexagon (R : ReuleauxShape a r) (H : Hexagon ((a + 2 * r) / Real.sqrt 3)) :
  ∀ (θ : ℝ), ∃ p ∈ (boundary H), ¬(p ∈ (boundary (rotate R θ))) := sorry

-- The second part: Reuleaux shape cannot be rotated inside a triangle while maintaining touch
theorem reuleaux_cannot_rotate_in_triangle (R : ReuleauxShape a r) (T : Triangle (a + 2 * r)) :
  ∀ (θ : ℝ), ¬(∃ p q r ∈ (boundary T), p ∈ (boundary (rotate R θ)) → q ∈ (boundary (rotate R θ)) → r ∈ (boundary (rotate R θ))) := sorry

end reuleaux_cannot_rotate_in_hexagon_reuleaux_cannot_rotate_in_triangle_l504_504611


namespace right_triangle_side_product_l504_504722

theorem right_triangle_side_product :
  let a := 6
  let b := 8
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let other_leg := Real.sqrt (b^2 - a^2)
  (hypotenuse * 2 * Real.sqrt 7).round = 53 := -- using 53 to consider rounding to the nearest tenth

by
  let a := 6
  let b := 8
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let other_leg := Real.sqrt (b^2 - a^2)
  have h1 : hypotenuse = 10 := by sorry
  have h2 : other_leg = 2 * Real.sqrt 7 := by sorry
  have h_prod : (hypotenuse * 2 * Real.sqrt 7).round = 53 := by sorry
  exact h_prod

end right_triangle_side_product_l504_504722


namespace tiles_difference_between_tenth_and_eleventh_square_l504_504806

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ :=
  3 + 2 * (n - 1)

-- Define the area of the nth square
def area (n : ℕ) : ℕ :=
  (side_length n) ^ 2

-- The math proof statement
theorem tiles_difference_between_tenth_and_eleventh_square : area 11 - area 10 = 88 :=
by 
  -- Proof goes here, but we use sorry to skip it for now
  sorry

end tiles_difference_between_tenth_and_eleventh_square_l504_504806


namespace right_triangle_third_side_product_l504_504727

theorem right_triangle_third_side_product :
  ∀ (a b : ℝ), (a = 6 ∧ b = 8 ∧ (a^2 + b^2 = c^2 ∨ a^2 = b^2 - c^2)) →
  (a * b = 53.0) :=
by
  intros a b h
  sorry

end right_triangle_third_side_product_l504_504727


namespace solution_count_l504_504955

noncomputable def number_of_solutions : ℕ :=
  { ⟨x : ℝ, ⟨y : ℝ, x^4 - 2^(-y^2) * x^2 - ⌊ x^2 ⌋ + 1 = 0 ⟩ ⟩ }.card

theorem solution_count : number_of_solutions = 2 := 
sorry

end solution_count_l504_504955


namespace area_of_intersection_of_three_circles_l504_504712

theorem area_of_intersection_of_three_circles (r : ℝ) (h : r > 0) :
  let A := (Real.sqrt 3 / 4) * r^2 in
  let S := (Real.pi * r^2) / 6 in
  3 * S - 2 * A = (1 / 2) * r^2 * (Real.pi - Real.sqrt 3) := 
by
  -- Proof steps would go here
  sorry

end area_of_intersection_of_three_circles_l504_504712


namespace machines_needed_l504_504765

theorem machines_needed (x Y : ℝ) (R : ℝ) :
  (4 * R * 6 = x) → (M * R * 6 = Y) → M = 4 * Y / x :=
by
  intros h1 h2
  sorry

end machines_needed_l504_504765


namespace a_lt_sqrt3b_l504_504428

open Int

theorem a_lt_sqrt3b (a b : ℤ) (h1 : a > b) (h2 : b > 1) 
    (h3 : a + b ∣ a * b + 1) (h4 : a - b ∣ a * b - 1) : a < sqrt 3 * b :=
  sorry

end a_lt_sqrt3b_l504_504428


namespace common_root_conds_l504_504227

theorem common_root_conds (α a b c d : ℝ) (h₁ : a ≠ c)
  (h₂ : α^2 + a * α + b = 0)
  (h₃ : α^2 + c * α + d = 0) :
  α = (d - b) / (a - c) :=
by 
  sorry

end common_root_conds_l504_504227


namespace iodine_solution_percentage_l504_504333

theorem iodine_solution_percentage :
  ∀ (x : ℝ),
  (0.40 * 4.5 + (x / 100) * 4.5 = 0.50 * 6) → x = 26.67 := 
by 
  assume x,
  assume h,
  sorry

end iodine_solution_percentage_l504_504333


namespace find_y_l504_504978

theorem find_y (x y : ℤ) (h1 : x + y = 250) (h2 : x - y = 200) : y = 25 :=
by
  sorry

end find_y_l504_504978


namespace distance_sum_leq_sqrt_two_sum_squares_equality_possibility_l504_504226

theorem distance_sum_leq_sqrt_two_sum_squares
  (a b c : ℝ) (e : Line) (D A B C : Point) 
  (DA : segment D A, DB : segment D B, DC : segment D C)
  (line_through_vertex : e = Line.mk D) :
  distance A e + distance B e + distance C e ≤ sqrt (2 * (a^2 + b^2 + c^2)) :=
sorry

theorem equality_possibility
  (a b c : ℝ) (e : Line) (D A B C : Point) 
  (DA : segment D A, DB : segment D B, DC : segment D C)
  (line_through_vertex : e = Line.mk D) :
  (distance A e + distance B e + distance C e = sqrt (2 * (a^2 + b^2 + c^2))) ↔ 
  ∃ (λ : ℝ) (hλ : λ > 0), sin α / a = sin β / b = sin γ / c :=
sorry

end distance_sum_leq_sqrt_two_sum_squares_equality_possibility_l504_504226


namespace smallest_sum_of_consecutive_integers_l504_504259

theorem smallest_sum_of_consecutive_integers:
  ∃ (n m : ℕ), (n > 0) ∧ (20 * n + 190 = 2 * m^2) ∧ (20 * n + 190 = 450)  :=
by
  use 13, 15
  split; norm_num
  -- the proof steps would then follow
  sorry

end smallest_sum_of_consecutive_integers_l504_504259


namespace prime_count_between_50_and_70_l504_504036

theorem prime_count_between_50_and_70 : 
  (finset.filter nat.prime (finset.range 71).filter (λ n, 50 < n ∧ n < 71)).card = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504036


namespace range_of_function_l504_504248

open Set

noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem range_of_function (S : Set ℝ) : 
    S = {y : ℝ | ∃ x : ℝ, x ≥ 1 ∧ y = 2 + log_base_2 x} 
    ↔ S = {y : ℝ | y ≥ 2} :=
by 
  sorry

end range_of_function_l504_504248


namespace find_angle_CDB_l504_504303

/-
Translate given geometric problem and conditions into a Lean statement.

Given:
1) \( \angle CAD = \angle DBA = 40^\circ \)
2) \( \angle CAB = 60^\circ \)
3) \( \angle CBD = 20^\circ \)

To prove:
\( \angle CDB = 30^\circ \)
-/

variables {α : Type*}
variables [inner_product_space ℝ α]

open_locale euclidean_geometry

-- Defining the vertices of the quadrilateral and the angles as given
variables (A B C D : α)

-- Given conditions
def angle_CAD := 40
def angle_DBA := 40
def angle_CAB := 60
def angle_CBD := 20

-- Required proof statement
theorem find_angle_CDB :
  ∠CDB = 30 := sorry

end find_angle_CDB_l504_504303


namespace combination_of_15_3_l504_504958

open Nat

theorem combination_of_15_3 : choose 15 3 = 455 :=
by
  -- The statement describes that the number of ways to choose 3 books out of 15 is 455
  sorry

end combination_of_15_3_l504_504958


namespace math_problem_l504_504104

noncomputable def total_grid_area : ℝ := 7 * 7

noncomputable def small_circle_area : ℝ := 5 * (Real.pi * (0.5)^2)

noncomputable def large_circle_area : ℝ := 2 * (Real.pi * 1^2)

noncomputable def unshaded_square_area : ℝ := 4 * (0.5^2)

noncomputable def unshaded_area : ℝ :=
  small_circle_area + large_circle_area + unshaded_square_area

noncomputable def visible_shaded_area : ℝ :=
  total_grid_area - unshaded_area

noncomputable def A : ℝ := 48
noncomputable def B : ℝ := 3.25
noncomputable def C : ℝ := 1
noncomputable def total_formula : ℝ := A + B + C

theorem math_problem : total_formula = 52.25 :=
by
  have h1 : A = 48 := rfl
  have h2 : B = 3.25 := rfl
  have h3 : C = 1 := rfl
  have h4 : total_formula = 48 + 3.25 + 1 := by rw [h1, h2, h3]
  have h5 : total_formula = 52.25 := by norm_num
  exact h5

end math_problem_l504_504104


namespace prime_count_between_50_and_70_l504_504035

theorem prime_count_between_50_and_70 : 
  (finset.filter nat.prime (finset.range 71).filter (λ n, 50 < n ∧ n < 71)).card = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504035


namespace transformed_coordinates_l504_504336

-- Given conditions
def x₀ : ℝ := 15
def y₀ : ℝ := 8
def r : ℝ := Real.sqrt ((x₀^2) + (y₀^2))
def θ : ℝ := Real.atan2 y₀ x₀

-- New polar coordinates
def r₃ : ℝ := r^3
def θ₃ : ℝ := 3 * θ

-- Triple angle cosine and sine
def cos3θ : ℝ :=
  4 * (Real.cos θ)^3 - 3 * (Real.cos θ)
def sin3θ : ℝ :=
  3 * (Real.sin θ) - 4 * (Real.sin θ)^3

-- New rectangular coordinates
def x₃ : ℝ := r₃ * cos3θ
def y₃ : ℝ := r₃ * sin3θ

theorem transformed_coordinates :
  ( (x₃, y₃) =
    ( r₃ * ( 4 * ( (x₀ / r)^3 ) - 3 * (x₀ / r) ),
      r₃ * ( 3 * (y₀ / r) - 4 * ( (y₀ / r)^3 ) ) ) ) := 
sorry

end transformed_coordinates_l504_504336


namespace percentage_y_of_x_l504_504590

theorem percentage_y_of_x (x y : ℝ) (h : sqrt(0.3 * (x - y)) = sqrt(0.2 * (x + y))) :
  y = 0.2 * x :=
sorry

end percentage_y_of_x_l504_504590


namespace yoojung_namjoon_total_flowers_l504_504293

theorem yoojung_namjoon_total_flowers
  (yoojung_flowers : ℕ)
  (namjoon_flowers : ℕ)
  (yoojung_condition : yoojung_flowers = 4 * namjoon_flowers)
  (yoojung_count : yoojung_flowers = 32) :
  yoojung_flowers + namjoon_flowers = 40 :=
by
  sorry

end yoojung_namjoon_total_flowers_l504_504293


namespace replace_with_monomial_produces_four_terms_l504_504171

-- Define the initial expression
def initialExpression (k : ℤ) (x : ℤ) : ℤ :=
  ((x^4 - 3)^2 + (x^3 + k)^2)

-- Proof statement
theorem replace_with_monomial_produces_four_terms (x : ℤ) :
  ∃ (k : ℤ), initialExpression k x = (x^8 + x^6 + 9x^2 + 9) :=
  exists.intro (3 * x) sorry

end replace_with_monomial_produces_four_terms_l504_504171


namespace max_z_value_l504_504952

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 0) (h2 : x * y + y * z + z * x = -3) : z ≤ 2 := sorry

end max_z_value_l504_504952


namespace phone_prices_purchase_plans_l504_504790

noncomputable def modelA_price : ℝ := 2000
noncomputable def modelB_price : ℝ := 1000

theorem phone_prices :
  (∀ x y : ℝ, (2 * x + y = 5000 ∧ 3 * x + 2 * y = 8000) → x = modelA_price ∧ y = modelB_price) :=
by
    intro x y
    intro h
    have h1 := h.1
    have h2 := h.2
    -- We would provide the detailed proof here
    sorry

theorem purchase_plans :
  (∀ a : ℕ, (4 ≤ a ∧ a ≤ 6) ↔ (24000 ≤ 2000 * a + 1000 * (20 - a) ∧ 2000 * a + 1000 * (20 - a) ≤ 26000)) :=
by
    intro a
    -- We would provide the detailed proof here
    sorry

end phone_prices_purchase_plans_l504_504790


namespace prime_count_between_50_and_70_l504_504088

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def primes_between_50_and_70 : List ℕ :=
  [53, 59, 61, 67]

theorem prime_count_between_50_and_70 : 
  (primes_between_50_and_70.filter is_prime).length = 4 :=
by
  sorry

end prime_count_between_50_and_70_l504_504088


namespace sine_of_angle_between_u_v_l504_504490

open Real

def u : ℝ × ℝ × ℝ := (1, 2, 2)
def v : ℝ × ℝ × ℝ := (2, -1, 2)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ := 
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

noncomputable def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (a.1^2 + a.2^2 + a.3^2)

noncomputable def cos_angle (a b : ℝ × ℝ × ℝ) : ℝ :=
  dot_product a b / (magnitude a * magnitude b)

noncomputable def sin_angle (a b : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (1 - (cos_angle a b)^2)

theorem sine_of_angle_between_u_v : sin_angle u v = (sqrt 65) / 9 := 
  by sorry

end sine_of_angle_between_u_v_l504_504490


namespace problem_part1_problem_part2_l504_504022

noncomputable def seq_a (n : ℕ) : ℚ :=
  if n = 1 then 3 else sorry

noncomputable def seq_b (n : ℕ) : ℚ := seq_a n - 2

noncomputable def seq_c (n : ℕ) : ℚ := 1 / seq_b n

theorem problem_part1 : 
  ∀ n : ℕ, n > 1 → seq_c (n + 1) - seq_c n = 2 :=
sorry

noncomputable def S (n : ℕ) : ℚ :=
  ∑ i in finset.range n, seq_b i * seq_b (i + 1)

theorem problem_part2 : 
  ∀ n : ℕ, n ≥ 1 → (2 * n + 1) * 2^(n+2) * S n > (2 * n - 3) * 2^(n+1) + 192 ↔ n ≥ 6 :=
sorry

end problem_part1_problem_part2_l504_504022


namespace rachel_pool_fill_time_l504_504168

theorem rachel_pool_fill_time :
  ∀ (pool_volume : ℕ) (num_hoses : ℕ) (hose_rate : ℕ),
  pool_volume = 30000 →
  num_hoses = 5 →
  hose_rate = 3 →
  (pool_volume / (num_hoses * hose_rate * 60) : ℤ) = 33 :=
by
  intros pool_volume num_hoses hose_rate h1 h2 h3
  sorry

end rachel_pool_fill_time_l504_504168


namespace ruby_candies_l504_504619

theorem ruby_candies (number_of_friends : ℕ) (candies_per_friend : ℕ) (total_candies : ℕ)
  (h1 : number_of_friends = 9)
  (h2 : candies_per_friend = 4)
  (h3 : total_candies = number_of_friends * candies_per_friend) :
  total_candies = 36 :=
by {
  sorry
}

end ruby_candies_l504_504619


namespace not_possible_to_form_polygon_with_sticks_l504_504418

theorem not_possible_to_form_polygon_with_sticks :
  ¬ (∃ (subset : Finset ℕ), subset ⊆ (Finset.range 100) ∧ 
    (∀ x ∈ subset, ∃ k : ℕ, x = 2^k) ∧
    ∃ s ∈ subset, s.1 = 2^{99} ∧ 
    Finset.sum (subset.erase s) (λ x, x) > s.1) :=
sorry

end not_possible_to_form_polygon_with_sticks_l504_504418


namespace constant_term_expansion_2x_1_over_x_4_eq_24_l504_504219

theorem constant_term_expansion_2x_1_over_x_4_eq_24 : 
  (let a := 2; let b := (1 : ℝ); let n := 4 in
  ∑ r in finset.range (n + 1), binomial n r * (a * a) ^ (n - r) * (b / a) ^ r * x ^ (n - 2 * r)) = 24 :=
begin
  sorry
end

end constant_term_expansion_2x_1_over_x_4_eq_24_l504_504219


namespace inequality_medCircum_l504_504138

-- Definitions for the sides of the triangle and the medians
variables (a b c m_a m_b m_c D : ℝ)

-- Assume the relationships given in the problem
def is_triangle : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)
def median_a : Prop := m_a = 1 / 2 * real.sqrt (2 * b^2 + 2 * c^2 - a^2)
def median_b : Prop := m_b = 1 / 2 * real.sqrt (2 * a^2 + 2 * c^2 - b^2)
def median_c : Prop := m_c = 1 / 2 * real.sqrt (2 * a^2 + 2 * b^2 - c^2)
def diameter_of_circumcircle : Prop := D = 2 * a * b * c / (4 * real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)))

-- Problem statement to be proved
theorem inequality_medCircum:
  is_triangle a b c → median_a a b c m_a → median_b a b c m_b → median_c a b c m_c → diameter_of_circumcircle a b c D →
  (a^2 + b^2) / m_c + (a^2 + c^2) / m_b + (b^2 + c^2) / m_a ≤ 6 * D :=
by
  sorry

end inequality_medCircum_l504_504138


namespace find_quotient_l504_504110

theorem find_quotient (D d R Q : ℤ) (hD : D = 729) (hd : d = 38) (hR : R = 7)
  (h : D = d * Q + R) : Q = 19 := by
  sorry

end find_quotient_l504_504110


namespace simplify_sqrt_expression_l504_504656

theorem simplify_sqrt_expression :
  √(12 + 8 * √3) + √(12 - 8 * √3) = 4 * √3 :=
sorry

end simplify_sqrt_expression_l504_504656


namespace center_circumcircle_on_XY_l504_504000

theorem center_circumcircle_on_XY
    (ω1 ω2 : Circle) (O1 O2 : Point) (r1 r2 : Real)
    (X Y P Q R S : Point) :
    intersect ω1 ω2 X Y →
    lies_on_line_through_center ω1 O1 (Line.mk X Y) →
    lies_on_line_through_center ω2 O2 (Line.mk X Y) →
    line_intersects_circle O1 ω2 P Q →
    line_intersects_circle O2 ω1 R S →
    points_lie_on_same_circle [P, Q, R, S] →
    ∃ O3 : Point, center_circumcircle [P, Q, R, S] = O3 ∧ lies_on_line O3 (Line.mk X Y) :=
by
  sorry

end center_circumcircle_on_XY_l504_504000


namespace percentage_increase_in_price_l504_504834

theorem percentage_increase_in_price (initial_price : ℝ) (total_cost : ℝ) (num_family_members : ℕ) 
  (pounds_per_person : ℝ) (new_price : ℝ) (percentage_increase : ℝ) :
  initial_price = 1.6 → 
  total_cost = 16 → 
  num_family_members = 4 → 
  pounds_per_person = 2 → 
  (total_cost / (num_family_members * pounds_per_person)) = new_price → 
  percentage_increase = ((new_price - initial_price) / initial_price) * 100 → 
  percentage_increase = 25 :=
by
  intros h_initial h_total h_members h_pounds h_new_price h_percentage
  sorry

end percentage_increase_in_price_l504_504834


namespace total_eggs_found_l504_504976

theorem total_eggs_found (eggs_club_house : ℕ) (eggs_park : ℕ) (eggs_town_hall : ℕ) (eggs_library : ℕ) (eggs_community_center : ℕ) :
  eggs_club_house = 60 → eggs_park = 40 → eggs_town_hall = 30 → eggs_library = 50 → eggs_community_center = 35 → 
  eggs_club_house + eggs_park + eggs_town_hall + eggs_library + eggs_community_center = 215 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  -- sorry

end total_eggs_found_l504_504976


namespace probability_of_rain_at_most_3_days_in_july_l504_504983

open Nat

def probability_of_rain (k n : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

def total_probability_of_rain (n : ℕ) (p : ℚ) (max_days : ℕ) : ℚ :=
  (Finset.range (max_days + 1)).sum (λ k, probability_of_rain k n p)

theorem probability_of_rain_at_most_3_days_in_july :
  (total_probability_of_rain 31 (1 / 5) 3).toReal ≈ 0.191 :=
  sorry

end probability_of_rain_at_most_3_days_in_july_l504_504983


namespace milk_jars_good_for_sale_l504_504154

noncomputable def good_whole_milk_jars : ℕ := 
  let initial_jars := 60 * 30
  let short_deliveries := 20 * 30 * 2
  let damaged_jars_1 := 3 * 5
  let damaged_jars_2 := 4 * 6
  let totally_damaged_cartons := 2 * 30
  let received_jars := initial_jars - short_deliveries - damaged_jars_1 - damaged_jars_2 - totally_damaged_cartons
  let spoilage := (5 * received_jars) / 100
  received_jars - spoilage

noncomputable def good_skim_milk_jars : ℕ := 
  let initial_jars := 40 * 40
  let short_delivery := 10 * 40
  let damaged_jars := 5 * 4
  let totally_damaged_carton := 1 * 40
  let received_jars := initial_jars - short_delivery - damaged_jars - totally_damaged_carton
  let spoilage := (3 * received_jars) / 100
  received_jars - spoilage

noncomputable def good_almond_milk_jars : ℕ := 
  let initial_jars := 30 * 20
  let short_delivery := 5 * 20
  let damaged_jars := 2 * 3
  let received_jars := initial_jars - short_delivery - damaged_jars
  let spoilage := (1 * received_jars) / 100
  received_jars - spoilage

theorem milk_jars_good_for_sale : 
  good_whole_milk_jars = 476 ∧
  good_skim_milk_jars = 1106 ∧
  good_almond_milk_jars = 489 :=
by
  sorry

end milk_jars_good_for_sale_l504_504154


namespace shaded_area_in_2_foot_length_l504_504842

-- Definitions based on the conditions
def radius : ℝ := 3 -- in inches
def length : ℝ := 24 -- 2 feet in inches
def arcsInLength : ℝ := length / (2 * radius) -- Each arc diameter is 2 * radius

-- Prove the area of the shaded region in a 2-foot length
theorem shaded_area_in_2_foot_length 
  (r : ℝ)
  (l : ℝ)
  (arc_length : ℝ)
  (num_arcs : ℝ) 
  (circle_area : ℝ) : 
  r = 3 → l = 24 → arc_length = 2 * r → num_arcs = l / arc_length → circle_area = π * r^2 →
  2 * (num_arcs / 2) * circle_area / 2 = 18 * π :=
by
  intros hr hl h_arc_length h_num_arcs h_circle_area
  subst hr
  subst hl
  subst h_arc_length
  subst h_num_arcs
  subst h_circle_area
  sorry

end shaded_area_in_2_foot_length_l504_504842


namespace product_of_next_palindromic_year_l504_504704

   def is_palindrome (n : Nat) : Prop :=
     let s := n.repr
     s = s.reverse

   noncomputable def product_of_digits (n : Nat) : Nat :=
     (n.digits 10).foldl (· * ·) 1

   theorem product_of_next_palindromic_year (h : is_palindrome 1991) : 
     product_of_digits 2002 = 0 :=
   by
     sorry
   
end product_of_next_palindromic_year_l504_504704
