import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Binomial
import Mathlib.Algebra.Fraction
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Perm.Basic
import Mathlib.Analysis.Calculus.Fderiv.Basic
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Fintype.Prod
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Base
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial.Basic
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.InfiniteSum
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Circle.Basic
import Mathlib.Geometry.Euclidean.Circumcircle
import Mathlib.Geometry.Triangle.Basic
import Mathlib.MeasureTheory.ProbabilityMassFunction
import Mathlib.NumberTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.MetricSpace.Basic

namespace evaluate_pow_l187_187250

theorem evaluate_pow : 64^(5/6 : ℝ) = 32 := by
  sorry

end evaluate_pow_l187_187250


namespace identity_permutation_unique_l187_187623

open Nat

theorem identity_permutation_unique :
  ∀ (a : Fin 2021 → ℕ), 
    (∀ m n : ℕ, n > m + 20^21 →
      ∑ k in Finset.range 2021, 
        gcd (m + k + 1) (n + a ⟨k, sorry⟩) < 2 * (n - m)) → 
    ∀ i : Fin 2021, a i = (i : ℕ) + 1 :=
sorry

end identity_permutation_unique_l187_187623


namespace number_of_integer_points_in_intersection_l187_187182

def sphere1 (x y z : ℤ) := x^2 + y^2 + (z - 5)^2 ≤ 25
def sphere2 (x y z : ℤ) := x^2 + y^2 + z^2 ≤ 9

theorem number_of_integer_points_in_intersection : 
  let points := { p : ℤ × ℤ × ℤ | sphere1 p.1 p.2 p.3 ∧ sphere2 p.1 p.2 p.3 } in 
  points.to_finset.card = 13 :=
by
  sorry

end number_of_integer_points_in_intersection_l187_187182


namespace absolute_value_simplify_l187_187374

variable (a : ℝ)

theorem absolute_value_simplify
  (h : a < 3) : |a - 3| = 3 - a := sorry

end absolute_value_simplify_l187_187374


namespace conjugate_of_given_complex_number_l187_187332

theorem conjugate_of_given_complex_number :
  (∃ z : ℂ, z = (1 - complex.i) / (1 + complex.i) ∧ conj z = complex.i) :=
by
  let z : ℂ := (1 - complex.i) / (1 + complex.i)
  use z
  split
  exact rfl
  sorry

end conjugate_of_given_complex_number_l187_187332


namespace eval_power_l187_187210

-- Given condition
def sixty_four : ℕ := 64

-- Given condition rewritten in Lean
def sixty_four_as_two_powersix : sixty_four = 2^6 := by
  sorry

-- Prove that 64^(5/6) = 32
theorem eval_power : real.exp (5/6 * real.log 64) = 32 := by
  have h1 : 64 = 2^6 := sixty_four_as_two_powersix
  sorry

end eval_power_l187_187210


namespace partitioning_staircase_l187_187444

def number_of_ways_to_partition_staircase (n : ℕ) : ℕ :=
  2^(n-1)

theorem partitioning_staircase (n : ℕ) : 
  number_of_ways_to_partition_staircase n = 2^(n-1) :=
by 
  sorry

end partitioning_staircase_l187_187444


namespace correct_option_C_l187_187701

def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem correct_option_C : ∀ {x1 x2 : ℝ}, 0 < x1 → x1 < x2 → x1 * f x1 < x2 * f x2 :=
by
  sorry

end correct_option_C_l187_187701


namespace jordan_has_11_oreos_l187_187771

-- Define the conditions
def jamesOreos (x : ℕ) : ℕ := 3 + 2 * x
def totalOreos (jordanOreos : ℕ) : ℕ := 36

-- Theorem stating the problem that Jordan has 11 Oreos given the conditions
theorem jordan_has_11_oreos (x : ℕ) (h1 : jamesOreos x + x = totalOreos x) : x = 11 :=
by
  sorry

end jordan_has_11_oreos_l187_187771


namespace cos_of_angle_with_point_l187_187323

theorem cos_of_angle_with_point (α : ℝ) (h : ∃ P : ℝ × ℝ, P = (1, 3) ∧ P = (1, 3) ∧ α = real.atan2 3 1) :
  real.cos α = 1 / real.sqrt 10 :=
by
  sorry

end cos_of_angle_with_point_l187_187323


namespace dart_game_l187_187540

theorem dart_game (x y z : ℕ) (h1 : x + y + z > 11) (h2 : 8 * x + 9 * y + 10 * z = 100) :
  x = 10 ∨ x = 9 ∨ x = 8 :=
by {-- Proof omitted --}

end dart_game_l187_187540


namespace infinite_solutions_geometric_sequence_l187_187740

-- Define the sequence as a geometric progression
def is_geometric_progression (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → q ≠ 0 → a (n + 1) / a n = q

-- State the theorem
theorem infinite_solutions_geometric_sequence
  {a : ℕ → ℝ} {q : ℝ}
  (hg : is_geometric_progression a q)
  (h : q ≠ 0) :
  let A : Matrix (Fin 2) (Fin 3) ℝ := ![
    ![a 1, a 2, a 4],
    ![a 5, a 6, a 8]
  ] in
  let b : Vector (Fin 2) ℝ := ![
    0, 0
  ] in
  (A ⬝ b = 0) ∧ (∃ x y : ℝ, a 1 * x + a 2 * y = a 4 ∧ a 5 * x + a 6 * y = a 8) := sorry

end infinite_solutions_geometric_sequence_l187_187740


namespace repair_cost_l187_187881

-- Define the purchase price, selling price, and gain as given conditions
def purchase_price : ℝ := 900
def selling_price : ℝ := 1260
def gain_percent : ℝ := 5

-- Define the proof to show the repair cost
theorem repair_cost (purchase_cost selling_price : ℝ) (gain_percent: ℝ)
    (h1 : purchase_cost = purchase_price)
    (h2 : selling_price = 1260)
    (h3 : gain_percent = 5) :
    let total_cost := (selling_price * 100) / (100 + gain_percent),
        repair_cost := total_cost - purchase_cost
    in repair_cost = 300 :=
by
  sorry

end repair_cost_l187_187881


namespace range_of_p_l187_187349

theorem range_of_p (p : ℝ) (h_discriminant : (p - 1)^2 - 4 * p * (p + 1) > 0)
                   (h_positive_sum : (1 - p) / p > 0)
                   (h_positive_product : (p + 1) / p > 0) 
                   (h_one_root_greater : let x1 := (1 - p - real.sqrt ((p - 1)^2 - 4 * p * (p + 1))) / (2 * p),
                                              x2 := (1 - p + real.sqrt ((p - 1)^2 - 4 * p * (p + 1))) / (2 * p)
                                         in x2 > 2 * x1) : 
                   0 < p ∧ p < 1 / 7 := 
by {
    sorry
}

end range_of_p_l187_187349


namespace evaluate_64_pow_5_div_6_l187_187263

theorem evaluate_64_pow_5_div_6 : (64 : ℝ) ^ (5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ) ^ 6 := by norm_num
  have h2 : (64 : ℝ) ^ (5 / 6) = ((2 : ℝ) ^ 6) ^ (5 / 6) := by rw h1
  have h3 : ((2 : ℝ) ^ 6) ^ (5 / 6) = (2 : ℝ) ^ (6 * (5 / 6)) := by rw [Real.rpow_mul]
  have h4 : (2 : ℝ) ^ (6 * (5 / 6)) = (2 : ℝ) ^ 5 := by norm_num
  rw [h2, h3, h4]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187263


namespace average_time_per_student_l187_187594

theorem average_time_per_student (
  num_students : ℕ,
  total_hours : ℕ,
  total_problems : ℕ,
  students_per_problem : ℕ
) : num_students = 8 → total_hours = 2 → total_problems = 30 → students_per_problem = 2 → 
      (120 / (60 / 8) = 16) := 
by
  intros
  sorry

end average_time_per_student_l187_187594


namespace tan_7pi_over_6_eq_1_over_sqrt_3_l187_187599

theorem tan_7pi_over_6_eq_1_over_sqrt_3 : 
  ∀ θ : ℝ, θ = (7 * Real.pi) / 6 → Real.tan θ = 1 / Real.sqrt 3 :=
by
  intros θ hθ
  rw [hθ]
  sorry  -- Proof to be completed

end tan_7pi_over_6_eq_1_over_sqrt_3_l187_187599


namespace min_value_expr_l187_187301

-- Define the conditions
variables (x y : ℝ)
variables (hx_pos : 0 < x)
variables (hy_pos : 0 < y)
variables (h_eq : 2 * x + y = 1)

-- Define the expression
def expr := (x^2 + y^2 + x) / (x * y)

-- State the theorem
theorem min_value_expr : expr x y = 2 * real.sqrt 3 + 1 :=
by
  -- The proof of this theorem is omitted.
  sorry

end min_value_expr_l187_187301


namespace log_sum_example_l187_187601

open Real

theorem log_sum_example :
    2 * (log 10 9) + 3 * (log 10 4) + 4 * (log 10 3) + 5 * (log 10 2) + (log 10 16) = log 10 215233856 := by
  sorry

end log_sum_example_l187_187601


namespace intersection_complement_A_B_l187_187354

open Set

variable (x : ℝ)

def U := ℝ
def A := {x | -2 ≤ x ∧ x ≤ 3}
def B := {x | x < -1 ∨ x > 4}

theorem intersection_complement_A_B :
  {x | -2 ≤ x ∧ x ≤ 3} ∩ compl {x | x < -1 ∨ x > 4} = {x | -1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end intersection_complement_A_B_l187_187354


namespace integral_evaluation_l187_187604

def integral_expression : ℝ := ∫ x in (0 : ℝ)..(2 : ℝ), real.sqrt(4 - x^2)

theorem integral_evaluation : integral_expression = real.pi :=
sorry

end integral_evaluation_l187_187604


namespace value_at_neg_one_l187_187848

-- Definitions and conditions
def odd_function (f: ℝ → ℝ) : Prop := ∀ x, f (-x) = - f x

def f (x : ℝ) : ℝ :=
  if h : x > 0 then x + 1 else sorry -- We use sorry to handle the incomplete definition.

-- The proof problem statement
theorem value_at_neg_one (f_odd : odd_function f)
                      (f_pos : ∀ x, 0 < x → f x = x + 1) :
  f (-1) = -2 :=
by
  sorry

end value_at_neg_one_l187_187848


namespace quadrilateral_proof_problem_l187_187742

theorem quadrilateral_proof_problem
    (ABCD_area : ℝ)
    (PQ : ℝ)
    (RS : ℝ)
    (BD : ℝ)
    (m n p : ℕ)
    (h1 : ABCD_area = 15)
    (h2 : PQ = 6)
    (h3 : RS = 8)
    (h4 : BD^2 = m + n * real.sqrt p)
    (h5 : p > 0) :
    m + n + p = 81 :=
sorry

end quadrilateral_proof_problem_l187_187742


namespace equation_has_2020_real_solutions_l187_187464

noncomputable def G (x : ℝ) (c : ℝ) : ℝ := 
  x^2 * (x - 1)^2 * (x - 2)^2 * ... * (x - 1008)^2 * (x - 1009)^2 - c

theorem equation_has_2020_real_solutions (c : ℝ) 
  (h : 0 < c ∧ c < (List.product (List.range 1009 |>.map Real.ofNat))^4 / 2 ^ 2020) : 
  ∃ l : List ℝ, l.length = 2020 ∧ ∀ x ∈ l, G x c = 0 :=
begin
  sorry
end

end equation_has_2020_real_solutions_l187_187464


namespace measure_angle_BCQ_l187_187461

/-- Given:
  - Segment AB has a length of 12 units.
  - Segment AC is 9 units long.
  - Segment AC : CB = 3 : 1.
  - A semi-circle is constructed with diameter AB.
  - Another smaller semi-circle is constructed with diameter CB.
  - A line segment CQ divides the combined area of the two semi-circles into two equal areas.

  Prove: The degree measure of angle BCQ is 11.25°.
-/ 
theorem measure_angle_BCQ (AB AC CB : ℝ) (hAB : AB = 12) (hAC : AC = 9) (hRatio : AC / CB = 3) :
  ∃ θ : ℝ, θ = 11.25 :=
by
  sorry

end measure_angle_BCQ_l187_187461


namespace dihedral_angle_PACB_l187_187593

-- Given conditions
variables {A B C P : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space P]
variables [innermost B] [innermost C] [innermost P]

-- Defining lengths of sides in a right triangle
def AC_length := 2
def BC_length := 3
def AB_length := real.sqrt 7

-- Assuming point P on hypotenuse AB
axiom point_P_on_AB (P : Type*) (AB : Type*) : P ∈ AB

-- Main theorem to prove
theorem dihedral_angle_PACB : 
  dihedral_angle (P-AC-B) = real.arctan (real.sqrt 2) :=
by sorry

end dihedral_angle_PACB_l187_187593


namespace range_of_k_l187_187343

-- Definitions for the condition
def inequality_holds (k : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + (k-1)*x^2 + 1 ≥ 0

-- Theorem statement
theorem range_of_k (k : ℝ) : inequality_holds k → k ≥ 1 :=
sorry

end range_of_k_l187_187343


namespace students_not_yes_for_either_subject_l187_187555

variable (total_students yes_m no_m unsure_m yes_r no_r unsure_r yes_only_m : ℕ)

theorem students_not_yes_for_either_subject :
  total_students = 800 →
  yes_m = 500 →
  no_m = 200 →
  unsure_m = 100 →
  yes_r = 400 →
  no_r = 100 →
  unsure_r = 300 →
  yes_only_m = 150 →
  ∃ students_not_yes, students_not_yes = total_students - (yes_only_m + (yes_m - yes_only_m) + (yes_r - (yes_m - yes_only_m))) ∧ students_not_yes = 400 :=
by
  intros ht yt1 nnm um ypr ynr ur yom
  sorry

end students_not_yes_for_either_subject_l187_187555


namespace nathaniel_tickets_l187_187858

theorem nathaniel_tickets :
  ∀ (B S : ℕ),
  (7 * B + 4 * S + 11 = 128) →
  (B + S = 20) :=
by
  intros B S h
  sorry

end nathaniel_tickets_l187_187858


namespace total_money_collected_l187_187856

def hourly_wage : ℕ := 10 -- Marta's hourly wage 
def tips_collected : ℕ := 50 -- Tips collected by Marta
def hours_worked : ℕ := 19 -- Hours Marta worked

theorem total_money_collected : (hourly_wage * hours_worked + tips_collected = 240) :=
  sorry

end total_money_collected_l187_187856


namespace tan_30_l187_187148

theorem tan_30 : Real.tan (Real.pi / 6) = Real.sqrt 3 / 3 := 
by 
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry
  have h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2 := by sorry
  calc
    Real.tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6) : Real.tan_eq_sin_div_cos _
    ... = (1 / 2) / (Real.sqrt 3 / 2) : by rw [h1, h2]
    ... = (1 / 2) * (2 / Real.sqrt 3) : by rw Div.div_eq_mul_inv
    ... = 1 / Real.sqrt 3 : by norm_num
    ... = Real.sqrt 3 / 3 : by rw [Div.inv_eq_inv, Mul.comm, Mul.assoc, Div.mul_inv_cancel (Real.sqrt_ne_zero _), one_div Real.sqrt 3, inv_mul_eq_div]

-- Additional necessary function apologies for the unproven theorems.
noncomputable def _root_.Real.sqrt (x:ℝ) : ℝ := sorry

noncomputable def _root_.Real.tan (x : ℝ) : ℝ :=
  (Real.sin x) / (Real.cos x)

#eval tan_30 -- check result

end tan_30_l187_187148


namespace find_xy_l187_187455

-- Definition of the conditions
variables (x y : ℝ)
-- Define the given conditions
axiom positive_x : 0 < x
axiom positive_y : 0 < y
axiom equation1 : x^2 + y^2 = 1
axiom equation2 : x^4 + y^4 = 3 / 4

-- The problem we need to prove
theorem find_xy : x * y = Real.sqrt(2) / 4 :=
by
  sorry

end find_xy_l187_187455


namespace minimum_revisit_l187_187908

def rail_network (stations : Type) := set (prod stations stations)

variables (stations : Type) [decidable_eq stations]
variable (R : rail_network stations)

def path_exists (R: rail_network stations) (start end: stations) : Prop :=
∃ path : list stations, path.head = some start ∧ path.last = some end ∧ ∀ i, (i < path.length - 1) → (path.nth i, path.nth (i + 1)) ∈ R

def visit_all_stations (R : rail_network stations) (stations : list stations) : Prop :=
∀ s, s ∈ stations → ∃ (s' s'' : stations), s' ≠ s'' ∧ path_exists R s' s''

theorem minimum_revisit (stations : Type) [decidable_eq stations] (R : rail_network stations) :
  (visit_all_stations R (list.of_fn (λ i, stations)) → ∃ s1 s2 s3: stations, s1 ≠ s2 ∧ s2 ≠ s3 ∧ s1 ≠ s3 ∧
  (∃ path : list stations, (s1 ∈ path ∧ s2 ∈ path ∧ s3 ∈ path ∧ (∃ i, ∃ j, ∃ k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ paths.nth i = some s1 ∧ paths.nth j = some s2 ∧ paths.nth k = some s3 ∧ ∃ path2:list stations, s1::s2::s3::path2 ⊆ path ∧ ∀ m, m ∈ path → (m ∈ path2) ))))
  :=
sorry

end minimum_revisit_l187_187908


namespace max_value_of_q_l187_187812

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_q_l187_187812


namespace neither_maximum_nor_minimum_value_l187_187735

noncomputable def f (x : ℝ) : ℝ := x^3 - (3 / 2) * x^2 + 1

theorem neither_maximum_nor_minimum_value 
: ¬ ∃ y, ∀ x, f(x) ≤ y ∨ f(x) ≥ y := 
sorry

end neither_maximum_nor_minimum_value_l187_187735


namespace symmetric_difference_identity_l187_187088

variables {α : Type*} (A1 A2 A3 : set α)

-- Statement to prove
theorem symmetric_difference_identity :
  |A1 \triangle A2| + |A2ᶜ \triangle A3| - |A1ᶜ \triangle A3| = 
  2 * |A1 ∩ A2ᶜ ∩ A3ᶜ| + 2 * |A1ᶜ ∩ A2 ∩ A3| :=
sorry

end symmetric_difference_identity_l187_187088


namespace correct_option_C_l187_187699

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem correct_option_C : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → x1 * f x1 < x2 * f x2 :=
by
  intro x1 x2 hx1 hx12
  sorry

end correct_option_C_l187_187699


namespace smallest_n_for_4n_square_and_5n_cube_l187_187051

theorem smallest_n_for_4n_square_and_5n_cube :
  ∃ (n : ℕ), (n > 0 ∧ (∃ k : ℕ, 4 * n = k^2) ∧ (∃ m : ℕ, 5 * n = m^3)) ∧ n = 400 :=
by
  sorry

end smallest_n_for_4n_square_and_5n_cube_l187_187051


namespace deposited_percentage_is_24_l187_187112

noncomputable def percentage_deposited_to_wife : ℝ :=
  let total_income := 1000
  let amount_to_children := 0.2 * total_income
  let remaining_after_children := total_income - amount_to_children
  let final_amount := 500
  let donation (x: ℝ) := 0.1 * (remaining_after_children - (x / 100) * total_income)
  fun (x : ℝ) => remaining_after_children - (x / 100) * total_income - donation x

theorem deposited_percentage_is_24.44 : percentage_deposited_to_wife (24.44) = 500 := by
  sorry

end deposited_percentage_is_24_l187_187112


namespace second_train_speed_l187_187957

theorem second_train_speed
  (v : ℕ)
  (h1 : 8 * v - 8 * 11 = 160) :
  v = 31 :=
sorry

end second_train_speed_l187_187957


namespace seven_lines_regions_l187_187882

theorem seven_lines_regions :
  let n := 7 in
  (∀ (l1 l2 : ℝ → ℝ → Prop), ¬(∀ x, l1 x = l2 x)) ∧
  (∀ (l1 l2 l3 : ℝ → ℝ → Prop), ¬(∀ x, l1 x = l2 x ∧ l2 x = l3 x)) →
  number_of_regions n = 29 :=
by
  sorry

end seven_lines_regions_l187_187882


namespace polynomial_value_at_8_l187_187436

noncomputable def p : ℝ → ℝ := sorry

theorem polynomial_value_at_8 (p : ℝ → ℝ) (h_monic : ∀ c, (λ x, p x - x) (c + x^7) = 0) : 
  p 1 = 1 → p 2 = 2 → p 3 = 3 → p 4 = 4 → p 5 = 5 → p 6 = 6 → p 7 = 7 → p 8 = 5048 :=
by
  sorry

end polynomial_value_at_8_l187_187436


namespace max_q_value_l187_187803

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end max_q_value_l187_187803


namespace volume_of_solid_from_triangle_rotation_l187_187508

theorem volume_of_solid_from_triangle_rotation :
  let a := 1
  let b := 1
  let hypotenuse := Real.sqrt (a^2 + b^2) 
  let cone_radius := hypotenuse / 2 
  let cone_height := Real.sqrt (a^2 / 2)
  let cone_volume := (1 / 3) * Real.pi * cone_radius^2 * cone_height
  in 2 * cone_volume = (Real.sqrt 2 / 6) * Real.pi :=
by {
  sorry
}

end volume_of_solid_from_triangle_rotation_l187_187508


namespace eq_tangent_line_at_1_l187_187911

def f (x : ℝ) : ℝ := x^4 - 2 * x^3

def tangent_line (x : ℝ) : ℝ := -2 * x + 1

theorem eq_tangent_line_at_1 : 
  ∃ (m : ℝ) (c : ℝ), m = -2 ∧ c = 1 ∧ ∀ x, tangent_line x = m * x + c :=
by
  use -2
  use 1
  split
  . rfl
  split
  . rfl
  intro x
  rfl

end eq_tangent_line_at_1_l187_187911


namespace find_k_value_l187_187993

noncomputable def find_k (k : ℝ) : Prop :=
  let circle_center := (2 : ℝ, 0 : ℝ)
  let circle_radius := (2 : ℝ)
  let line_l1_distance := (sqrt 3 : ℝ)
  let line_l2_distance := (abs (2 * k - 1) / sqrt (k^2 + 1) : ℝ)
  let chord_l1_length := (2 : ℝ)
  let chord_l2_length := (2 * sqrt (4 - (2 * k - 1)^2 / (k^2 + 1)) : ℝ)
  chord_l1_length / chord_l2_length = 1 / 2

theorem find_k_value : ∃ k : ℝ, find_k k ∧ k = 1 / 2 :=
by {
  use (1 / 2 : ℝ),
  unfold find_k,
  sorry
}

end find_k_value_l187_187993


namespace money_left_l187_187873

def calories_per_orange := 80
def cost_per_orange := 1.2
def total_money := 10
def required_calories := 400

theorem money_left (n_oranges : ℕ) (money_spent : ℝ) :
  n_oranges = required_calories / calories_per_orange ∧
  money_spent = n_oranges * cost_per_orange →
  total_money - money_spent = 4 := by
  sorry

end money_left_l187_187873


namespace ratio_books_purchased_l187_187782

-- Definitions based on the conditions
def books_last_year : ℕ := 50
def books_before_purchase : ℕ := 100
def books_now : ℕ := 300

-- Let x be the multiple of the books purchased this year
def multiple_books_purchased_this_year (x : ℕ) : Prop :=
  books_now = books_before_purchase + books_last_year + books_last_year * x

-- Prove the ratio is 3:1
theorem ratio_books_purchased (x : ℕ) (h : multiple_books_purchased_this_year x) : x = 3 :=
  by sorry

end ratio_books_purchased_l187_187782


namespace apples_in_box_l187_187473

variable (A : ℕ)

-- Translation of the given conditions into Lean definitions and statements
def initial_oranges : ℕ := 26
def removed_oranges : ℕ := 20
def remaining_oranges : ℕ := initial_oranges - removed_oranges
def ratio_apples : ℚ := 0.70
def ratio_oranges : ℚ := 0.30

-- The proof statement
theorem apples_in_box : A = 14 :=
  have h1 : remaining_oranges = 6 := by
    sorry, -- This placeholder represents the calculation: 26 - 20 = 6
  have h2 : remaining_oranges = ratio_oranges * (A + remaining_oranges) := by
    sorry, -- This placeholder represents the equation: 6 = 0.30 * (A + 6)
  have h3 : 6 = 0.30 * (A + 6) := by
    exact h2,
  have h4 : 20 = A + 6 := by
    sorry, -- This placeholder represents the rearrangement: 6 / 0.30 = A + 6
  have h5 : A = 14 := by
    sorry, -- This placeholder represents the result: A = 20 - 6 = 14
  h5

end apples_in_box_l187_187473


namespace intersection_A_B_l187_187663

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | x^2 - 2 * x < 0 }

theorem intersection_A_B : A ∩ B = { x | 0 < x ∧ x < 2 } :=
by
  -- We are going to skip the proof for now
  sorry

end intersection_A_B_l187_187663


namespace angle_EFG_is_60_l187_187298

-- Given: AD is parallel to FG
variable (AD FG : Line)
variable (parallel_AD_FG : AD ∥ FG)

-- Definitions of angles
variable (x : ℝ)
variable (angle_CFG EFG CEA : Angle)

-- Conditions from the problem
def corresponding_angles (parallel_AD_FG : AD ∥ FG) : Prop :=
  angle_CFG + angle_CEA = π

-- Theorem to prove
theorem angle_EFG_is_60 (parallel_AD_FG : AD ∥ FG) (hx : 6 * x = 180) :
  EFG = 2 * 30 :=
by
  sorry

end angle_EFG_is_60_l187_187298


namespace tan_30_eq_sqrt3_div_3_l187_187160

/-- Statement that proves the value of tang of 30 degrees, given the cosine
    and sine values. -/
theorem tan_30_eq_sqrt3_div_3 
  (cos_30 : Real) (sin_30 : Real) 
  (hcos : cos_30 = Real.sqrt 3 / 2) 
  (hsin : sin_30 = 1 / 2) : 
    Real.tan 30 = Real.sqrt 3 / 3 := 
by 
  sorry

end tan_30_eq_sqrt3_div_3_l187_187160


namespace polynomial_expansion_l187_187085

open Real
open Nat

noncomputable def f (x : ℝ) : ℝ := 
  let a_0 : ℝ := sorry
  let a_1 : ℝ := sorry
  let a_2 : ℝ := sorry
  let a_n : ℝ := sorry
  a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n

theorem polynomial_expansion (a : ℝ) (f : ℝ → ℝ) :
  ∃ A : Fin n → ℝ, 
    (f (x) = A 0 + A 1 * (x - a) + A 2 * (x - a)^2 + ... + A n * (x - a)^n ∧
    (A 0 = f a) ∧
    (A 1 = deriv f a) ∧
    (A 2 = (deriv^[2] f a) / 2!) ∧
    ... ∧
    (A n = (deriv^[n] f a) / n!)) :=
sorry

end polynomial_expansion_l187_187085


namespace tan_30_eq_sqrt3_div_3_l187_187176

theorem tan_30_eq_sqrt3_div_3 :
  let opposite := 1
  let adjacent := sqrt (3 : ℝ) 
  tan (real.pi / 6) = opposite / adjacent := by 
    sorry

end tan_30_eq_sqrt3_div_3_l187_187176


namespace points_in_rectangle_l187_187295

-- Define the rectangle dimensions
def length := 4 -- The length of the rectangle
def width := 3 -- The width of the rectangle

-- Define the Euclidean distance function
def euclidean_distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Define the main theorem to be proven
theorem points_in_rectangle (points : Fin 6 → (ℝ × ℝ)) (h : ∀ i, points i.1 ∈ set.Icc (0,0) (length,width)) :
  ∃ (i j : Fin 6), i ≠ j ∧ euclidean_distance (points i) (points j) ≤ real.sqrt 5 :=
begin
  sorry
end

end points_in_rectangle_l187_187295


namespace eq_tangent_line_at_1_l187_187913

def f (x : ℝ) : ℝ := x^4 - 2 * x^3

def tangent_line (x : ℝ) : ℝ := -2 * x + 1

theorem eq_tangent_line_at_1 : 
  ∃ (m : ℝ) (c : ℝ), m = -2 ∧ c = 1 ∧ ∀ x, tangent_line x = m * x + c :=
by
  use -2
  use 1
  split
  . rfl
  split
  . rfl
  intro x
  rfl

end eq_tangent_line_at_1_l187_187913


namespace dice_probability_l187_187039

theorem dice_probability :
  let outcomes : List (ℕ × ℕ) := [(4,1), (4,2), (4,3), (4,4), (4,5), (4,6), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6)],
      favorable : List (ℕ × ℕ) := [(4,6), (6,4), (6,5), (6,6)]
  in ((List.length favorable).toRat / (List.length outcomes).toRat = (1 / 3)) :=
by
  sorry

end dice_probability_l187_187039


namespace number_of_integer_values_of_a_l187_187651

theorem number_of_integer_values_of_a (a : ℤ) : 
  (∃ x : ℤ, x^2 + a * x + 9 * a = 0) ↔ 
  (∃ (a_values : Finset ℤ), a_values.card = 6 ∧ ∀ a ∈ a_values, ∃ x : ℤ, x^2 + a * x + 9 * a = 0) :=
by
  sorry

end number_of_integer_values_of_a_l187_187651


namespace tan_30_eq_sqrt3_div_3_l187_187161

/-- Statement that proves the value of tang of 30 degrees, given the cosine
    and sine values. -/
theorem tan_30_eq_sqrt3_div_3 
  (cos_30 : Real) (sin_30 : Real) 
  (hcos : cos_30 = Real.sqrt 3 / 2) 
  (hsin : sin_30 = 1 / 2) : 
    Real.tan 30 = Real.sqrt 3 / 3 := 
by 
  sorry

end tan_30_eq_sqrt3_div_3_l187_187161


namespace eval_64_pow_5_over_6_l187_187197

theorem eval_64_pow_5_over_6 (h : 64 = 2^6) : 64^(5/6) = 32 := 
by 
  sorry

end eval_64_pow_5_over_6_l187_187197


namespace derivative_of_x_squared_plus_sin_x_l187_187629

noncomputable def derivative := sorry

theorem derivative_of_x_squared_plus_sin_x (x : ℝ) : 
    derivative (λ x => x^2 + Real.sin x) = (λ x => 2*x + Real.cos x) :=
by
  sorry

end derivative_of_x_squared_plus_sin_x_l187_187629


namespace length_QS_l187_187762

variables (P Q R S T : Type)
variables [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S] [MetricSpace T]
variables [MetricSpace.dist] [MetricSpace.alt PS]
variables [MetricSpace.dist QR] = 10
variables [MetricSpace.dist PS] = 6
variables [MetricSpace.dist PT] = 5

def is_obtuse_triangle (P Q R : Point) := ∃ β, β > 90
def altitude (P Q : Point) (PS : Point) := ∃ PS, ∠ PQR = 90
def altitudes_for_triangle :=
  altitude PS QR ∧ altitude PT PR

theorem length_QS (P Q R S T : Point) :
  is_obtuse_triangle P Q R → altitudes_for_triangle P S T →
  dist QR = 10 → dist PS = 6 → dist PT = 5 →
  dist QS = 4.45 :=
sorry

end length_QS_l187_187762


namespace students_with_both_l187_187944

/-- There are 28 students in a class -/
def total_students : ℕ := 28

/-- Number of students with a cat -/
def students_with_cat : ℕ := 17

/-- Number of students with a dog -/
def students_with_dog : ℕ := 10

/-- Number of students with neither a cat nor a dog -/
def students_with_neither : ℕ := 5

/-- Number of students having both a cat and a dog -/
theorem students_with_both :
  students_with_cat + students_with_dog - (total_students - students_with_neither) = 4 :=
sorry

end students_with_both_l187_187944


namespace tan_30_deg_l187_187173

theorem tan_30_deg : 
  let θ := (30 : ℝ) * (Real.pi / 180)
  in Real.sin θ = 1 / 2 ∧ Real.cos θ = Real.sqrt 3 / 2 → Real.tan θ = Real.sqrt 3 / 3 :=
by
  intro h
  let th := θ
  have h1 : Real.sin th = 1 / 2 := And.left h
  have h2 : Real.cos th = Real.sqrt 3 / 2 := And.right h
  sorry

end tan_30_deg_l187_187173


namespace smallest_x_cubed_ends_and_begins_with_1_l187_187636

def first_three_digits_one (n : ℕ) : Prop :=
  let str_n := n.toString
  str_n.length ≥ 3 ∧ str_n.take 3 = "111"

def last_four_digits_one_one_one_one (n : ℕ) : Prop :=
  let str_n := n.toString
  str_n.length ≥ 4 ∧ str_n.drop (str_n.length - 4) = "1111"

theorem smallest_x_cubed_ends_and_begins_with_1 : 
  ∃ x : ℕ, 
    (∀ y : ℕ, (first_three_digits_one (y ^ 3) ∧ last_four_digits_one_one_one_one (y ^ 3) → y = x)) ∧ 
    first_three_digits_one (x ^ 3) ∧ last_four_digits_one_one_one_one (x ^ 3) :=
by 
  use 1038471
  -- proof steps here
  sorry

end smallest_x_cubed_ends_and_begins_with_1_l187_187636


namespace x1_xn_leq_neg_inv_n_l187_187432

theorem x1_xn_leq_neg_inv_n (n : ℕ) (hn : 0 < n)
  (x : ℕ → ℝ)
  (hx_sorted : ∀ i j, i < j → x i ≤ x j)
  (hx_sum_zero : ∑ i in Finset.range n, x i = 0)
  (hx_sq_sum_one : ∑ i in Finset.range n, (x i)^2 = 1) :
  x 0 * x (n - 1) ≤ - (1 / n) := 
sorry

end x1_xn_leq_neg_inv_n_l187_187432


namespace rain_at_least_once_l187_187491

noncomputable def rain_probability (day_prob : ℚ) (days : ℕ) : ℚ :=
  1 - (1 - day_prob)^days

theorem rain_at_least_once :
  ∀ (day_prob : ℚ) (days : ℕ),
    day_prob = 3/4 → days = 4 →
    rain_probability day_prob days = 255/256 :=
by
  intros day_prob days h1 h2
  sorry

end rain_at_least_once_l187_187491


namespace distinct_integer_values_of_a_l187_187647

theorem distinct_integer_values_of_a : 
  let eq_has_integer_solutions (a : ℤ) : Prop := 
    ∃ (x y : ℤ), (x^2 + a*x + 9*a = 0) ∧ (y^2 + a*y + 9*a = 0) in
  (finset.univ.filter eq_has_integer_solutions).card = 5 := 
sorry

end distinct_integer_values_of_a_l187_187647


namespace Jerry_final_answer_l187_187119

theorem Jerry_final_answer (x : ℕ) (h : x = 8) : 
  let y := (x - 2) * 3 in 
  let z := (y + 3) * 3 in
  z = 63 :=
by
  split
  case intro a =>
    have aux_1 : y = (x - 2) * 3 := by rfl
    have aux_2 : z = (y + 3) * 3 := by rfl
    rw [h] at aux_1
    rw [aux_1, h]
    norm_num at *
    sorry

end Jerry_final_answer_l187_187119


namespace evaluate_pow_l187_187257

theorem evaluate_pow : (64 : ℝ) = (8 : ℝ) ^ 2 → (8 : ℝ) = (2 : ℝ) ^ 3 → (64 : ℝ) ^ (5 / 6) = 32 :=
by
  intros h1 h2
  rw h1
  rw h2
  have h3 : (2 : ℝ)^3 ^ 2 = (2 : ℝ) ^ 6 := by ring_exp
  rw h3
  sorry

end evaluate_pow_l187_187257


namespace geom_series_sum_l187_187847

theorem geom_series_sum (n : ℕ) :
  (∑ k in finset.range (n + 4), 2 * 8^k) = (2 / 7) * (8 ^ (n + 4) - 1) :=
by
  sorry

end geom_series_sum_l187_187847


namespace student_weight_loss_l187_187118

theorem student_weight_loss :
  let S := 71 in
  let combined_weight := 104 in
  let R := combined_weight - S in
  let desired_student_weight := 2 * R in
  let weight_to_lose := S - desired_student_weight in
  weight_to_lose = 5 :=
by
  sorry

end student_weight_loss_l187_187118


namespace extras_after_drinking_l187_187459

theorem extras_after_drinking (total_sodas drank_sodas : ℕ) (h1 : total_sodas = 11) (h2 : drank_sodas = 3) : total_sodas - drank_sodas = 8 :=
by
  rw [h1, h2]
  exact rfl

end extras_after_drinking_l187_187459


namespace last_place_team_wins_l187_187750

theorem last_place_team_wins
  (teams : Finset ℕ)
  (plays : ∀ (i j : ℕ), i ∈ teams ∧ j ∈ teams → Prop)
  (no_ties : ∀ (i j : ℕ), i ∈ teams ∧ j ∈ teams → plays i j ∨ plays j i)
  (first_place : ℕ)
  (second_place third_place fourth_place fifth_place : ℕ)
  (ranked_teams : List ℕ)
  (ranking_cond : ∀ (i j : ℕ), i ∈ teams ∧ j ∈ teams ∧ i ≠ j →
    (i ∈ ranked_teams ∧ j ∈ ranked_teams ∧ plays i j → ranked_teams.index_of i < ranked_teams.index_of j)):
  (first_place_wins_all : ∀ i ∈ teams, i ≠ first_place → plays first_place i)
  (second_place_wins_two : set.count plays second_place = 2)
  (third_place_wins_two : set.count plays third_place = 2)
  (total_teams : teams.card = 5)
  (all_teams_ranking: [first_place, second_place, third_place, fourth_place, fifth_place] ~ ranked_teams)
  : set.count plays fifth_place = 1 := 
by
  sorry

end last_place_team_wins_l187_187750


namespace four_m0_as_sum_of_primes_l187_187690

theorem four_m0_as_sum_of_primes (m0 : ℕ) (h1 : m0 > 1) 
  (h2 : ∀ n : ℕ, ∃ p : ℕ, Prime p ∧ n ≤ p ∧ p ≤ 2 * n) 
  (h3 : ∀ p1 p2 : ℕ, Prime p1 → Prime p2 → (2 * m0 ≠ p1 + p2)) : 
  ∃ p1 p2 p3 p4 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ Prime p4 ∧ (4 * m0 = p1 + p2 + p3 + p4) ∨ (∃ p1 p2 p3 : ℕ, Prime p1 ∧ Prime p2 ∧ Prime p3 ∧ 4 * m0 = p1 + p2 + p3) :=
by sorry

end four_m0_as_sum_of_primes_l187_187690


namespace max_value_amc_am_mc_ca_l187_187792

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end max_value_amc_am_mc_ca_l187_187792


namespace evaluate_64_pow_5_div_6_l187_187270

theorem evaluate_64_pow_5_div_6 : (64 : ℝ) ^ (5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ) ^ 6 := by norm_num
  have h2 : (64 : ℝ) ^ (5 / 6) = ((2 : ℝ) ^ 6) ^ (5 / 6) := by rw h1
  have h3 : ((2 : ℝ) ^ 6) ^ (5 / 6) = (2 : ℝ) ^ (6 * (5 / 6)) := by rw [Real.rpow_mul]
  have h4 : (2 : ℝ) ^ (6 * (5 / 6)) = (2 : ℝ) ^ 5 := by norm_num
  rw [h2, h3, h4]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187270


namespace john_vacation_funds_l187_187413

def octal_to_decimal (o : List ℕ) : ℕ :=
o.foldr (λ (d acc : ℕ), d + 8 * acc) 0

-- Definition of John's savings in base 8
def john_savings_octal : List ℕ := [5, 4, 3, 2]

-- Convert John's savings from base 8 to base 10
def john_savings_decimal : ℕ := octal_to_decimal john_savings_octal

-- Cost of the round-trip ticket in base 10
def round_trip_ticket_cost : ℕ := 1200

-- John's remaining money after buying the ticket
def john_remaining_money : ℕ := john_savings_decimal - round_trip_ticket_cost

-- Theorem to be proved
theorem john_vacation_funds : john_remaining_money = 1642 :=
by
    have h1 : john_savings_decimal = 2842 := by
        -- Proof of conversion from octal to decimal
        sorry
    have h2 : john_remaining_money = 2842 - 1200 := by
        -- Proof of subtraction
        sorry
    rw h1 at h2
    norm_num at h2
    assumption

end john_vacation_funds_l187_187413


namespace non_negative_root_condition_l187_187475

noncomputable theory -- Declare noncomputable theory

open Real -- Open the real numbers scope

-- Define the equation root definition and prove that it has one non-negative root under given conditions
def has_non_negative_root (a: ℝ) : Prop :=
    ∃ (x : ℝ), x = a + sqrt (a^2 - 4*a + 3) ∨ x = a - sqrt (a^2 - 4*a + 3)

theorem non_negative_root_condition (a: ℝ) : 
    (a ∈ Set.Icc (3 / 4) 1 ∪ Set.Ici 3 ∨ 0 < a ∧ a < 3 / 4) → has_non_negative_root a :=
by
    sorry -- Full proof is omitted

end non_negative_root_condition_l187_187475


namespace rolling_sphere_points_coplanar_l187_187574

theorem rolling_sphere_points_coplanar
    (sphere : ℝ^3)
    (box : set ℝ^3)
    (initial_corner : ℝ^3)
    (X X1 X2 X3 : ℝ^3)
    (h1 : sphere \in initial_corner)
    (h2 : sphere ⊆ box)
    (h3 : ∀ (p ∈ {X, X1, X2, X3}), p ∈ sphere) :
  ∃ (plane : set ℝ^3), {X, X1, X2, X3} ⊆ plane :=
by
  sorry

end rolling_sphere_points_coplanar_l187_187574


namespace evaluate_root_l187_187239

theorem evaluate_root : 64 ^ (5 / 6 : ℝ) = 32 :=
by sorry

end evaluate_root_l187_187239


namespace find_BD_l187_187831

-- Define the given conditions in Lean:
variables (A B C D : Type) -- the four points
variables [real_space A B C D] -- these are real-space points
variables (BD : real) -- Line segment BD's length

-- Necessary conditions:
axiom right_triangle_ABC : is_right_triangle A B C
axiom BC_diameter : diameter_of_circle (B, C)
axiom circle_intersect_AC : circle_has_point (AC) D
axiom area_triangle_ABC : triangle_area A B C = 180
axiom side_length_BC : segment_length B C = 30

-- Define the main proof statement:
theorem find_BD : BD = 12 :=
sorry

end find_BD_l187_187831


namespace max_value_of_expression_l187_187808

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l187_187808


namespace tan_30_l187_187149

theorem tan_30 : Real.tan (Real.pi / 6) = Real.sqrt 3 / 3 := 
by 
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry
  have h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2 := by sorry
  calc
    Real.tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6) : Real.tan_eq_sin_div_cos _
    ... = (1 / 2) / (Real.sqrt 3 / 2) : by rw [h1, h2]
    ... = (1 / 2) * (2 / Real.sqrt 3) : by rw Div.div_eq_mul_inv
    ... = 1 / Real.sqrt 3 : by norm_num
    ... = Real.sqrt 3 / 3 : by rw [Div.inv_eq_inv, Mul.comm, Mul.assoc, Div.mul_inv_cancel (Real.sqrt_ne_zero _), one_div Real.sqrt 3, inv_mul_eq_div]

-- Additional necessary function apologies for the unproven theorems.
noncomputable def _root_.Real.sqrt (x:ℝ) : ℝ := sorry

noncomputable def _root_.Real.tan (x : ℝ) : ℝ :=
  (Real.sin x) / (Real.cos x)

#eval tan_30 -- check result

end tan_30_l187_187149


namespace smallest_n_l187_187060

theorem smallest_n (n : ℕ) : (∃ (m1 m2 : ℕ), 4 * n = m1^2 ∧ 5 * n = m2^3) ↔ n = 500 := 
begin
  sorry
end

end smallest_n_l187_187060


namespace mass_of_4_moles_l187_187967

theorem mass_of_4_moles (molecular_weight : ℕ) (n : ℕ) (h1 : molecular_weight = 312) (h2 : n = 4) : 
  mass n molecular_weight = 1248 :=
by
  sorry

def mass (n : ℕ) (molecular_weight : ℕ) : ℕ := n * molecular_weight

end mass_of_4_moles_l187_187967


namespace tan_30_eq_sqrt3_div_3_l187_187175

theorem tan_30_eq_sqrt3_div_3 :
  let opposite := 1
  let adjacent := sqrt (3 : ℝ) 
  tan (real.pi / 6) = opposite / adjacent := by 
    sorry

end tan_30_eq_sqrt3_div_3_l187_187175


namespace opposite_of_one_fourth_l187_187934

/-- The opposite of the fraction 1/4 is -1/4 --/
theorem opposite_of_one_fourth : - (1 / 4) = -1 / 4 :=
by
  sorry

end opposite_of_one_fourth_l187_187934


namespace jordan_has_11_oreos_l187_187772

-- Define the conditions
def jamesOreos (x : ℕ) : ℕ := 3 + 2 * x
def totalOreos (jordanOreos : ℕ) : ℕ := 36

-- Theorem stating the problem that Jordan has 11 Oreos given the conditions
theorem jordan_has_11_oreos (x : ℕ) (h1 : jamesOreos x + x = totalOreos x) : x = 11 :=
by
  sorry

end jordan_has_11_oreos_l187_187772


namespace measure_of_angle_Z_l187_187405

theorem measure_of_angle_Z (X Y Z : ℝ) (h_sum : X + Y + Z = 180) (h_XY : X + Y = 80) : Z = 100 := 
by
  -- The proof is not required.
  sorry

end measure_of_angle_Z_l187_187405


namespace find_x3_l187_187042

noncomputable def x3 : ℝ :=
  Real.log ((2 / 3) + (1 / 3) * Real.exp 2)

theorem find_x3 
  (x1 x2 : ℝ)
  (h1 : x1 = 0)
  (h2 : x2 = 2)
  (A : ℝ × ℝ := (x1, Real.exp x1))
  (B : ℝ × ℝ := (x2, Real.exp x2))
  (C : ℝ × ℝ := ((2 * A.1 + B.1) / 3, (2 * A.2 + B.2) / 3))
  (yC : ℝ := (2 / 3) * A.2 + (1 / 3) * B.2)
  (E : ℝ × ℝ := (x3, yC)) :
  E.1 = Real.log ((2 / 3) + (1 / 3) * Real.exp x2) := sorry

end find_x3_l187_187042


namespace find_a1_plus_a13_l187_187755

variable (a : ℕ → ℝ) (d : ℝ)

-- The sequence is an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given condition: a_4 + a_6 + a_8 + a_10 = 80
def given_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 + a 7 + a 9 = 80

-- To prove: a_1 + a_13 = 40
theorem find_a1_plus_a13 (a : ℕ → ℝ) (d : ℝ) [arithmetic_sequence a d] [given_condition a] : 
  a 0 + a 12 = 40 :=
by {
  sorry
}

end find_a1_plus_a13_l187_187755


namespace largest_d_l187_187373

theorem largest_d (a b c d : ℤ) 
  (h₁ : a + 1 = b - 2) 
  (h₂ : a + 1 = c + 3) 
  (h₃ : a + 1 = d - 4) : 
  d > a ∧ d > b ∧ d > c := 
by 
  -- Here we would provide the proof, but for now we'll skip it
  sorry

end largest_d_l187_187373


namespace square_side_length_equals_nine_l187_187948

-- Definitions based on the conditions
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 8
def rectangle_perimeter (length width : ℕ) : ℕ := 2 * length + 2 * width
def side_length_of_square (perimeter : ℕ) : ℕ := perimeter / 4

-- The theorem we want to prove
theorem square_side_length_equals_nine : 
  side_length_of_square (rectangle_perimeter rectangle_length rectangle_width) = 9 :=
by
  -- proof goes here
  sorry

end square_side_length_equals_nine_l187_187948


namespace sqrt_equation_l187_187722

theorem sqrt_equation {m n : ℕ} (h : sqrt (5 - 2 * sqrt 6) = sqrt m - sqrt n) : m = 3 ∧ n = 2 := 
by 
  sorry

end sqrt_equation_l187_187722


namespace slope_of_parallel_line_l187_187049

theorem slope_of_parallel_line (m : ℚ) (b : ℚ) :
  (∀ x y : ℚ, 5 * x - 3 * y = 21 → y = (5 / 3) * x + b) →
  m = 5 / 3 :=
by
  intros hyp
  sorry

end slope_of_parallel_line_l187_187049


namespace max_dot_product_l187_187714

variable (θ : ℝ)

def a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
def b : ℝ × ℝ := (3, -4)

noncomputable def dot_product : ℝ := a θ.1 * b.1 + a θ.2 * b.2

theorem max_dot_product : ∃ θ : ℝ, dot_product θ = 5 := by
  sorry

end max_dot_product_l187_187714


namespace determine_range_of_k_l187_187346

noncomputable def inequality_holds_for_all_x (k : ℝ) : Prop :=
  ∀ (x : ℝ), x^4 + (k - 1) * x^2 + 1 ≥ 0

theorem determine_range_of_k (k : ℝ) : inequality_holds_for_all_x k ↔ k ≥ 1 := sorry

end determine_range_of_k_l187_187346


namespace ratio_is_one_fifth_l187_187111

-- Axioms corresponding to the conditions of the problem
axiom number : ℕ 
axiom part_increased_by_four : ℕ 

-- Definition of N in the problem
def N := 280

-- Equation from the problem
axiom equation : part_increased_by_four + 4 = (N / 4) - 10

-- Definition of the ratio
def ratio := part_increased_by_four / N

-- The theorem to prove the ratio is 1/5
theorem ratio_is_one_fifth : ratio = 1 / 5 := 
sorry

end ratio_is_one_fifth_l187_187111


namespace length_AB_l187_187324

noncomputable def ellipse_length_AB : ℝ :=
  let E_center := (0, 0 : ℝ × ℝ)
  let e := Real.sqrt 3 / 2
  let right_focus := (3, 0 : ℝ × ℝ)
  let parabola_eqn (x y : ℝ) : Prop := y^2 = 12 * x
  let latus_rectum_x := -3
  let ellipse_eqn (x y : ℝ) : Prop := (x^2 / 12) + (y^2 / 3) = 1
  let A := (-3, Real.sqrt 3 / 2) : ℝ × ℝ
  let B := (-3, -Real.sqrt 3 / 2) : ℝ × ℝ
  dist A B

theorem length_AB : ellipse_length_AB = Real.sqrt 3 := by
  sorry

end length_AB_l187_187324


namespace not_perpendicular_l187_187108

noncomputable def line := sorry
noncomputable def plane := sorry

def intersects (a : line) (α : plane) : Prop := sorry
def equidistant_lines (a : line) (α : plane) : Set line := sorry
def does_not_intersect (l1 l2 : line) : Prop := sorry
def count {α : Type*} (s : Set α) : ℕ := sorry

-- Given conditions
variable (a : line) (α : plane)
variable h_intersects : intersects a α
variable h_count : count (equidistant_lines a α) = 2011
variable h_no_intersect : ∀ l ∈ (equidistant_lines a α), does_not_intersect l a

-- Proof problem statement
theorem not_perpendicular (a : line) (α : plane) 
    (h_intersects : intersects a α)
    (h_count : count (equidistant_lines a α) = 2011)
    (h_no_intersect : ∀ l ∈ (equidistant_lines a α), does_not_intersect l a) 
    : ¬ perpendicular a α :=
sorry

end not_perpendicular_l187_187108


namespace tangent_line_at_1_l187_187915

noncomputable def f (x : ℝ) : ℝ := x^4 - 2 * x^3

theorem tangent_line_at_1 :
  let p := (1 : ℝ, f 1)
  in ∃ m c : ℝ, (∀ x : ℝ, y : ℝ, y = m * x + c ↔ y = -2 * x + 1) ∧ ∀ x : ℝ, f x = x^4 - 2 * x^3 :=
sorry

end tangent_line_at_1_l187_187915


namespace zero_members_in_all_departments_l187_187117

-- Definitions for the sets involved
variable (X : Type) -- Universe of all members
variable (L U S : Set X) -- Sets for soccer players (L), swimmers (U), and chess players (S)

-- Conditions
axiom a1 : S ⊆ U -- No chess player who does not swim.
axiom a2 : L \ U ⊆ S -- Soccer players who do not swim all play chess.
axiom a3 : (L ∪ S).card = U.card -- Soccer and chess departments together have as many members as the swimming department.
axiom a4 : ∀ x, x ∈ L ∨ x ∈ S → x ∈ U -- Every member participates in at least two departments

-- Prove that the size of the intersection of all three departments is 0
theorem zero_members_in_all_departments :
  (L ∩ S ∩ U).card = 0 :=
sorry

end zero_members_in_all_departments_l187_187117


namespace units_digit_product_of_even_multiples_of_4_between_20_and_120_l187_187076

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

def even_multiples_of_4_between (a b : ℕ) : list ℕ :=
(list.range (b - a + 1)).map (λ n, a + n).filter (λ n, is_even n ∧ is_multiple_of_4 n)

theorem units_digit_product_of_even_multiples_of_4_between_20_and_120 :
  (even_multiples_of_4_between 20 120).prod % 10 = 0 :=
by
  -- Proof goes here
  sorry

end units_digit_product_of_even_multiples_of_4_between_20_and_120_l187_187076


namespace range_of_a_l187_187696

def f (x : ℝ) := -x^5 - 3*x^3 - 5*x + 3

theorem range_of_a (a : ℝ) : f(a) + f(a - 2) > 6 → a < 1 := 
by
  sorry -- The proof is omitted

end range_of_a_l187_187696


namespace smallest_yellow_marbles_l187_187854

theorem smallest_yellow_marbles :
  ∃ n : ℕ, (n ≡ 0 [MOD 20]) ∧
           (∃ b : ℕ, b = n / 4) ∧
           (∃ r : ℕ, r = n / 5) ∧
           (∃ g : ℕ, g = 10) ∧
           (∃ y : ℕ, y = n - (b + r + g) ∧ y = 1) :=
sorry

end smallest_yellow_marbles_l187_187854


namespace eval_64_pow_5_over_6_l187_187201

theorem eval_64_pow_5_over_6 (h : 64 = 2^6) : 64^(5/6) = 32 := 
by 
  sorry

end eval_64_pow_5_over_6_l187_187201


namespace part1_part2_l187_187530

def is_sum_solution_equation (a b x : ℝ) : Prop :=
  x = b + a

def part1_statement := ¬ is_sum_solution_equation 3 4.5 (4.5 / 3)

def part2_statement (m : ℝ) : Prop :=
  is_sum_solution_equation 5 (m + 1) (m + 6) → m = (-29 / 4)

theorem part1 : part1_statement :=
by 
  -- Proof here
  sorry

theorem part2 (m : ℝ) : part2_statement m :=
by 
  -- Proof here
  sorry

end part1_part2_l187_187530


namespace smallest_n_for_perfect_square_and_cube_l187_187071

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, 0 < n ∧ (∃ a1 b1 : ℕ, 4 * n = a1 ^ 2 ∧ 5 * n = b1 ^ 3 ∧ n = 50) :=
begin
  use 50,
  split,
  { norm_num, },
  { use [10, 5],
    split,
    { norm_num, },
    { split, 
      { norm_num, },
      { refl, }, },
  },
  sorry
end

end smallest_n_for_perfect_square_and_cube_l187_187071


namespace solve_QE_l187_187822

noncomputable def QE : ℝ := 16

variables {Q C : Type}
variables [IsCircle C]
variables (Q : Point) (R D E : Point) 
variables [OutsidePoint Q C] [Tangent Q Q R] [Secant Q Q D Q E]
variables (QD QR QE DE : ℝ)
variables (PowerOfPoint : QD * QE = QR * QR)
variables (hQR : QR = DE - QD) (hQD : QD = 4) (hLT : QD < QE)

theorem solve_QE : QE = 16 := 
by 
  sorry

end solve_QE_l187_187822


namespace solve_for_z_l187_187667

theorem solve_for_z (z : ℂ) (h : z + 5 - 6 * complex.I = 3 + 4 * complex.I) : z = -2 + 10 * complex.I :=
by
  sorry

end solve_for_z_l187_187667


namespace timmy_money_left_after_oranges_l187_187876

-- Define the conditions
def orange_calories : ℕ := 80
def orange_cost : ℝ := 1.20
def timmy_money : ℝ := 10
def required_calories : ℕ := 400

-- Define the proof problem
theorem timmy_money_left_after_oranges :
  (timmy_money - (real.of_nat (required_calories / orange_calories)) * orange_cost = 4) :=
by
  sorry

end timmy_money_left_after_oranges_l187_187876


namespace smallest_possible_s_l187_187826

noncomputable def all_x_i (x: ℕ → set ℕ): Prop :=
  ∀ i: ℕ, 1 ≤ i → i < 100 → (x i ∩ x (i + 1) = ∅) ∧ (x i ∪ x (i + 1) ≠ {1, 2, 3, 4, 5, 6, 7, 8}) ∧
    ∀ j: ℕ, 1 ≤ j → j < i → (x i ≠ x j)

theorem smallest_possible_s : 
  ∃ S: set ℕ, S = {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
  ∃ x: ℕ → set ℕ, all_x_i x → S.card = 8 :=
begin
  sorry
end

end smallest_possible_s_l187_187826


namespace large_prime_divisor_l187_187313

theorem large_prime_divisor {p x : ℕ} (hp : p ≥ 3) (H_pprime : nat.prime p)
  (H_sufficiently_large_x : ∃ N, ∀ x ≥ N, x > 0) :
  ∃ i, i ∈ (list.range (p + 3) / 2).map (λ n, x + n + 1) ∧ ∃ q, nat.prime q ∧ q > p ∧ q ∣ (x + i) := 
sorry

end large_prime_divisor_l187_187313


namespace geometric_prog_last_term_l187_187285

theorem geometric_prog_last_term (a r : ℝ) (S_n T_n : ℝ) (n : ℕ) 
  (ha : a = 9)
  (hr : r = 1/3)
  (hS_n : S_n = 40/3)
  (hsum : S_n = a * (1 - r^n) / (1 - r)) :
  T_n = a * r^(n-1) :=
by
  -- Bring in the necessary constraints
  have ar := ha ▸ hr ▸ rfl,
  have sn := ha ▸ hr ▸ hS_n ▸ hsum ▸ rfl,
  sorry

end geometric_prog_last_term_l187_187285


namespace sum_of_reciprocals_is_two_l187_187505

variable (x y : ℝ)
variable (h1 : x + y = 50)
variable (h2 : x * y = 25)

theorem sum_of_reciprocals_is_two (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1/x + 1/y) = 2 :=
by
  sorry

end sum_of_reciprocals_is_two_l187_187505


namespace smallest_n_condition_l187_187066

theorem smallest_n_condition (n : ℕ) : (4 * n) ∣ (n^2) ∧ (5 * n) ∣ (u^3) → n = 100 :=
by
  sorry

end smallest_n_condition_l187_187066


namespace max_dot_product_l187_187711

open Real

theorem max_dot_product (θ : ℝ) :
  let a := (cos θ, sin θ)
  let b := (3, -4)
  ∃ θ, a.1 * b.1 + a.2 * b.2 = 5 :=
by
  sorry

end max_dot_product_l187_187711


namespace rain_at_least_once_l187_187493

theorem rain_at_least_once (p : ℚ) (h : p = 3/4) : 
    (1 - (1 - p)^4) = 255/256 :=
by
  sorry

end rain_at_least_once_l187_187493


namespace line_through_fixed_point_and_equation_l187_187671

/-- Definitions and conditions -/
def circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5
def line (m x y : ℝ) : Prop := m * x - y + 1 - m = 0
def distance_between_points (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- Main theorem statement -/

theorem line_through_fixed_point_and_equation {m x y : ℝ} {A B : ℝ × ℝ} :
  (∀ m, (∃ x y, line m x y ∧ x = 1 ∧ y = 1)) ∧
  (distance_between_points A B = real.sqrt 17 →
  ∃ m, line m x y ∧ (line m = λ x y, sqrt 3 * x - y + 1 - sqrt 3) ∨
                  (line m = λ x y, -sqrt 3 * x + y + 1 - sqrt 3)) :=
by
  split
  · -- proof that line l always passes through (1,1)
    sorry
  · -- proof to find the required line equation(s)
    sorry

end line_through_fixed_point_and_equation_l187_187671


namespace total_elements_C_l187_187462

-- Define the sets C and D
variables {C D : Type} [fintype C] [fintype D]

-- We know that the cardinality of C is three times the cardinality of D
def card_C_eq_3_card_D (c d : ℕ) : Prop :=
  c = 3 * d

-- Given the conditions
def union_card (c d : ℕ) : Prop :=
  (c + d - 1200) = 4500

-- Goal is to prove the total number of elements in set C is 4275
theorem total_elements_C (c d : ℕ) (h1 : card_C_eq_3_card_D c d)
  (h2 : union_card c d) : c = 4275 :=
by
  -- we claim c is 4275
  sorry

end total_elements_C_l187_187462


namespace evaluate_root_l187_187241

theorem evaluate_root : 64 ^ (5 / 6 : ℝ) = 32 :=
by sorry

end evaluate_root_l187_187241


namespace max_value_amc_am_mc_ca_l187_187791

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end max_value_amc_am_mc_ca_l187_187791


namespace quadratic_condition_l187_187936

theorem quadratic_condition (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (0 ≤ a ∧ a < 3) :=
sorry

end quadratic_condition_l187_187936


namespace number_of_girls_who_left_the_auditorium_l187_187035

theorem number_of_girls_who_left_the_auditorium :
  ∀ (B G remaining_students : ℕ), 
  B = 24 → G = 14 → (B + G - remaining_students) % 2 = 0 → remaining_students = 30 → 
  (B + G - remaining_students) / 2 = 4 :=
by
  intros B G remaining_students hB hG hEqual hRemaining
  have hTotal : B + G = 38 := by rw [hB, hG]; exact rfl
  have hLeft : B + G - remaining_students = 8 := by rw [hTotal, hRemaining]; exact rfl
  have hDiv : (B + G - remaining_students) / 2 = 4 := by rw [hLeft]; exact rfl
  exact hDiv

end number_of_girls_who_left_the_auditorium_l187_187035


namespace count_valid_base_three_numbers_l187_187633

/--
Finds the number of positive integers less than or equal to 2017 whose base-three 
representation contains no digit equal to 0.
-/
theorem count_valid_base_three_numbers : 
  (finset.filter (λ n, ∀ d ∈ ((nat.digits 3 n).to_list : list ℕ), d ≠ 0) 
  (finset.range_succ 2017)).card = 222 :=
sorry

end count_valid_base_three_numbers_l187_187633


namespace P_eq_Q_l187_187843

variable (f : ℝ → ℝ)

noncomputable def is_bijective : Prop :=
  Function.Bijective f

noncomputable def is_strictly_increasing : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f(x₁) < f(x₂)

noncomputable def P : Set ℝ := {x | x > f(x)}
noncomputable def Q : Set ℝ := {x | x > f(f(x))}

theorem P_eq_Q (H1 : is_bijective f) (H2 : is_strictly_increasing f) :
  P f = Q f :=
sorry

end P_eq_Q_l187_187843


namespace integer_values_of_a_count_integer_values_of_a_l187_187655

theorem integer_values_of_a (a : ℤ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ (x1 * x1 + a * x1 + 9 * a = 0) ∧ (x2 * x2 + a * x2 + 9 * a = 0)) →
  a ∈ {0, -12, -64} :=
by
  sorry

theorem count_integer_values_of_a : 
  {a : ℤ | ∃ x1 x2 : ℤ, x1 ≠ x2 ∧ (x1 * x1 + a * x1 + 9 * a = 0) ∧ (x2 * x2 + a * x2 + 9 * a = 0)}.to_finset.card = 3 :=
by
  sorry

end integer_values_of_a_count_integer_values_of_a_l187_187655


namespace max_value_of_q_l187_187813

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_q_l187_187813


namespace correct_answer_b_l187_187745

statement : Prop :=
∀ (candidates population sample_size : nat),
    (candidates = 30000) →
    (sample_size = 600) →
    (population = candidates) →
    True  -- This represents the correctness of the statement "The math score of each candidate is an individual."
    
theorem correct_answer_b : statement :=
by
    intros candidates population sample_size h1 h2 h3
    exact trivial

end correct_answer_b_l187_187745


namespace remainder_of_geometric_series_l187_187600

theorem remainder_of_geometric_series :
  let S := (1 + 11 + 11^2 + ⋯ + 11^2500)
  S % 500 = 1 :=
sorry

end remainder_of_geometric_series_l187_187600


namespace coeff_x4_in_expansion_l187_187903

-- We are tasked with proving that the coefficient of x^4 in the expansion of x(1 + x)(1 + x^2)^10 is 10.

-- We'll define the polynomial and express its expansion
def polynomial : Polynomial ℤ := Polynomial.X * (1 + Polynomial.X) * (1 + Polynomial.X^2) ^ 10

-- The goal is to state that the coefficient of x^4 in the expansion of "polynomial" is 10
theorem coeff_x4_in_expansion : polynomial.coeff 4 = 10 :=
by sorry

end coeff_x4_in_expansion_l187_187903


namespace problem_1_problem_2_l187_187704

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 + a) * x^2 - Real.log x - a + 1

theorem problem_1 (a : ℝ) : (a ≤ -1 → ∀ x > 0, f x a < f x 0) ∧ 
                             (a > -1 → ∃ c > 0, c = Real.sqrt (2 * (1 + a)) / (2 * (1 + a)) ∧ 
                              (∀ x ∈ Ioo 0 c, f x a < f x 0) ∧ 
                              (∀ x ∈ Ioo c +∞, f x a > f x 0)) :=
sorry

theorem problem_2 (a : ℝ) (h : a < 1) : ∀ x > 0, x * f x a > Real.log x + (1 + a) * x^3 - x^2 :=
sorry

end problem_1_problem_2_l187_187704


namespace first_row_is_53124_l187_187274

noncomputable def SudokuGrid : Type := Matrix (Fin 5) (Fin 5) (Fin 5)

def valid_sudoku_grid (grid: SudokuGrid) : Prop :=
  (∀ i : Fin 5, ∀ j : Fin 5, ∀ k : Fin 5, i ≠ j → grid i k ≠ grid j k) ∧ -- no repeats in any column
  (∀ i : Fin 5, ∀ j : Fin 5, ∀ k : Fin 5, i ≠ j → grid k i ≠ grid k j) ∧ -- no repeats in any row
  (∀ b : Fin 5, ∀ i : Fin (√5), ∀ j : Fin (√5),
    let block: Fin 5 → Fin 5 → Fin 5 :=
      λ m n, grid ((((b / sqrt 5) * sqrt 5) + m) % 5) ((((b % sqrt 5) * sqrt 5) + n) % 5) in
        ∀ k l : Fin 5, k ≠ l → block k i ≠ block l j) ∧ -- no repeats in any block
  (∀ i j : Fin 5, ∀ k l : Fin 5, (i ≠ k ∧ j ≠ l) → (|i - k| == 1 ∧ |j - l| == 1) → grid i j ≠ grid k l) -- diagonal constraint

theorem first_row_is_53124 (grid: SudokuGrid) (h : valid_sudoku_grid grid) : 
  (grid 0 0, grid 0 1, grid 0 2, grid 0 3, grid 0 4) = (5, 3, 1, 2, 4) :=
sorry

end first_row_is_53124_l187_187274


namespace x1_xn_leq_neg_inv_n_l187_187433

theorem x1_xn_leq_neg_inv_n (n : ℕ) (hn : 0 < n)
  (x : ℕ → ℝ)
  (hx_sorted : ∀ i j, i < j → x i ≤ x j)
  (hx_sum_zero : ∑ i in Finset.range n, x i = 0)
  (hx_sq_sum_one : ∑ i in Finset.range n, (x i)^2 = 1) :
  x 0 * x (n - 1) ≤ - (1 / n) := 
sorry

end x1_xn_leq_neg_inv_n_l187_187433


namespace evaluate_pow_l187_187260

theorem evaluate_pow : (64 : ℝ) = (8 : ℝ) ^ 2 → (8 : ℝ) = (2 : ℝ) ^ 3 → (64 : ℝ) ^ (5 / 6) = 32 :=
by
  intros h1 h2
  rw h1
  rw h2
  have h3 : (2 : ℝ)^3 ^ 2 = (2 : ℝ) ^ 6 := by ring_exp
  rw h3
  sorry

end evaluate_pow_l187_187260


namespace sin_cos_relation_l187_187613

variable (a b : ℝ) (θ : ℝ)
variable (h : (sin θ) ^ 6 / a + (cos θ) ^ 6 / b = 1 / (a + b))

theorem sin_cos_relation : 
  (sin θ) ^ 12 / a ^ 5 + (cos θ) ^ 12 / b ^ 5 = 1 / (a + b) ^ 5 :=
by
  sorry

end sin_cos_relation_l187_187613


namespace prime_related_divisors_circle_l187_187955

variables (n : ℕ)

-- Definitions of prime-related and conditions for n
def is_prime (p: ℕ): Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p
def prime_related (a b : ℕ) : Prop := 
  ∃ p : ℕ, is_prime p ∧ (a = p * b ∨ b = p * a)

-- The main statement to be proven
theorem prime_related_divisors_circle (n : ℕ) : 
  (n ≥ 3) ∧ (∀ a b, a ≠ b → (a ∣ n ∧ b ∣ n) → prime_related a b) ↔ ¬ (
    ∃ (p : ℕ) (k : ℕ), is_prime p ∧ (n = p ^ k) ∨ 
    ∃ (m : ℕ), n = m ^ 2 ) :=
sorry

end prime_related_divisors_circle_l187_187955


namespace correct_option_C_l187_187698

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem correct_option_C : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → x1 * f x1 < x2 * f x2 :=
by
  intro x1 x2 hx1 hx12
  sorry

end correct_option_C_l187_187698


namespace quadratic_expression_value_l187_187309

theorem quadratic_expression_value (a : ℝ) :
  (∃ x : ℝ, (3 * a - 1) * x^2 - a * x + 1 / 4 = 0 ∧ 
  (3 * a - 1) * x^2 - a * x + 1 / 4 = 0 ∧ 
  a^2 - 3 * a + 1 = 0) → 
  a^2 - 2 * a + 2021 + 1 / a = 2023 := 
sorry

end quadratic_expression_value_l187_187309


namespace angle_BNM_is_40_l187_187353

theorem angle_BNM_is_40
  (A B C M N : Type)
  (triangle : Triangle A B C)
  (hC_eq_B : ∠ C = ∠ B)
  (hC_eq_50 : ∠ C = 50)
  (hMAB_eq_50 : ∠ MAB = 50)
  (hABN_eq_30 : ∠ ABN = 30) :
  ∠ BNM = 40 := by
  sorry

end angle_BNM_is_40_l187_187353


namespace evaluate_pow_l187_187245

theorem evaluate_pow : 64^(5/6 : ℝ) = 32 := by
  sorry

end evaluate_pow_l187_187245


namespace rectangle_diagonal_length_l187_187485

theorem rectangle_diagonal_length (P : ℝ) (L W D : ℝ) 
  (hP : P = 72) 
  (h_ratio : 3 * W = 2 * L) 
  (h_perimeter : 2 * (L + W) = P) :
  D = Real.sqrt (L * L + W * W) :=
sorry

end rectangle_diagonal_length_l187_187485


namespace smallest_n_condition_l187_187065

theorem smallest_n_condition (n : ℕ) : (4 * n) ∣ (n^2) ∧ (5 * n) ∣ (u^3) → n = 100 :=
by
  sorry

end smallest_n_condition_l187_187065


namespace minimum_revenue_maximum_marginal_cost_minimum_profit_l187_187395

noncomputable def R (x : ℕ) : ℝ := x^2 + 16 / x^2 + 40
noncomputable def C (x : ℕ) : ℝ := 10 * x + 40 / x
noncomputable def MC (x : ℕ) : ℝ := C (x + 1) - C x
noncomputable def z (x : ℕ) : ℝ := R x - C x

theorem minimum_revenue :
  ∀ x : ℕ, 1 ≤ x → x ≤ 10 → R x ≥ 72 :=
sorry

theorem maximum_marginal_cost :
  ∀ x : ℕ, 1 ≤ x → x ≤ 9 → MC x ≤ 86 / 9 :=
sorry

theorem minimum_profit :
  ∀ x : ℕ, 1 ≤ x → x ≤ 10 → (x = 1 ∨ x = 4) → z x ≥ 7 :=
sorry

end minimum_revenue_maximum_marginal_cost_minimum_profit_l187_187395


namespace root_in_interval_l187_187925

def f (x : ℝ) : ℝ := x^3 + x - 3

lemma monotone_f : ∀ x, f x ≥ f 0 :=
  by
  intro x
  calc
    f x = x^3 + x - 3   : by rfl
    ... ≥ (0^3 + 0 - 3) : by sorry -- Show monotonicity using derivative f'(x) ≥ 0
    ... = -3            : by norm_num

theorem root_in_interval : ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
  by
  have h1 : f 1 < 0 := by norm_num
  have h2 : f 2 > 0 := by norm_num
  use (exists_between (by linarith)).some
  simp_rw [lt_iff_le_not_le]
  exact ⟨(exists_between (by linarith)).some_spec.left, (exists_between (by linarith)).some_spec.right, sorry⟩

end root_in_interval_l187_187925


namespace pencils_problem_l187_187743

theorem pencils_problem (red_pencils blue_pencils : ℕ) (h_red : red_pencils = 7) (h_blue : blue_pencils = 5) :
  ∃ n, n = 10 ∧
       (∀ taken : ℕ, taken ≥ n → 
                     (∃ r b : ℕ, r ≥ 2 ∧ b ≥ 3 ∧ r + b = taken ∧ r + b ≤ red_pencils + blue_pencils)) :=
begin
  sorry
end

end pencils_problem_l187_187743


namespace quadrilateral_sides_l187_187115

noncomputable def circle_radius : ℝ := 25
noncomputable def diagonal1_length : ℝ := 48
noncomputable def diagonal2_length : ℝ := 40

theorem quadrilateral_sides :
  ∃ (a b c d : ℝ),
    (a = 5 * Real.sqrt 10 ∧ 
    b = 9 * Real.sqrt 10 ∧ 
    c = 13 * Real.sqrt 10 ∧ 
    d = 15 * Real.sqrt 10) ∧ 
    (diagonal1_length = 48 ∧ 
    diagonal2_length = 40 ∧ 
    circle_radius = 25) :=
sorry

end quadrilateral_sides_l187_187115


namespace eval_power_l187_187209

-- Given condition
def sixty_four : ℕ := 64

-- Given condition rewritten in Lean
def sixty_four_as_two_powersix : sixty_four = 2^6 := by
  sorry

-- Prove that 64^(5/6) = 32
theorem eval_power : real.exp (5/6 * real.log 64) = 32 := by
  have h1 : 64 = 2^6 := sixty_four_as_two_powersix
  sorry

end eval_power_l187_187209


namespace tangent_circle_radius_is_one_l187_187036

noncomputable def radius_of_tangent_circle (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 1) (h₂ : r₂ = 2) (h₃ : r₃ = 3) : ℝ :=
  let a := r₁ + r₂
  let b := r₂ + r₃
  let c := r₃ + r₁
  let s := (a + b + c) / 2
  let A := real.sqrt (s * (s - a) * (s - b) * (s - c))
  A / s

theorem tangent_circle_radius_is_one :
  radius_of_tangent_circle 1 2 3 1 2 3 = 1 :=
by 
  sorry

end tangent_circle_radius_is_one_l187_187036


namespace sequence_bound_l187_187938

-- Define the sequence using recurrence relation
def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n > 0 → a (n+1) = Real.sqrt (a n ^ 2 + 1 / (a n))

-- The hypothesis that such a sequence exists
axiom sequence_exists : ∃ a : ℕ → ℝ, sequence a

-- The theorem to prove
theorem sequence_bound :
  ∃ α : ℝ, α = 1 / 3 ∧
  ∃ a : ℕ → ℝ, sequence a ∧ ∀ n : ℕ, n > 0 →
  1 / 2 ≤ a n / n ^ (1 / 3 : ℝ) ∧ a n / n ^ (1 / 3 : ℝ) ≤ 2 :=
by
  sorry

end sequence_bound_l187_187938


namespace cong_sum_l187_187183

open Nat

def f (n : ℕ) : ℕ :=
  ∑ t in {t : Fin n → Fin n | ∑ j in Fin.range n, j * t j = n}, 
  (1 + ∑ i in Fin.range n, t i)! / ((1 + t 0) * ∏ i in Fin.range n, (t i)!)

theorem cong_sum (p : ℕ) (hp : Nat.Prime p ∧ 3 < p) : 
  (∑ i in Finset.range (p - 1), ∑ j in Finset.Ico (i + 1) (p - 1), ∑ k in Finset.Ico (j + 1) (p - 1), 
    f i / (i * j * k)) % p = 
  (∑ i in Finset.range (p - 1), ∑ j in Finset.Ico (i + 1) (p - 1), ∑ k in Finset.Ico (j + 1) (p - 1), 
    2^i / (i * j * k)) % p := by 
  sorry

end cong_sum_l187_187183


namespace fiona_reaches_pad_thirteen_without_predators_l187_187032

noncomputable def probability_reach_pad_thirteen : ℚ := sorry

theorem fiona_reaches_pad_thirteen_without_predators :
  probability_reach_pad_thirteen = 3 / 2048 :=
sorry

end fiona_reaches_pad_thirteen_without_predators_l187_187032


namespace median_school_A_correct_l187_187952

noncomputable def median_score_school_A : ℝ :=
  let scores := [2, 3, 5, 10, 10]  -- This corresponds to the number of students in each score range bin for School A.
  let sub_range_scores := [96, 96.5, 97, 97.5, 96.5, 96.5, 97.5, 96, 96.5, 96.5]
  let n := 30  -- Total number of students sampled.
  let median_position := (n / 2).to_nat -- This corresponds to calculating the position of the median.
  let sorted_scores := sub_range_scores.sort -- Sort the sub-range scores.
  if n % 2 == 0 then
      (sorted_scores.get_median median_position - 1 + 
       sorted_scores.get_median median_position) / 2
  else
      sorted_scores.get_median median_position

theorem median_school_A_correct : median_score_school_A = 96.5 :=
by sorry

end median_school_A_correct_l187_187952


namespace DE_eq_2KL_l187_187783

noncomputable def TriangleABC := Type

noncomputable def incenter (T : TriangleABC) : Type := sorry

noncomputable def projection (P : Type) (line : Type) : Type := sorry 

noncomputable def reflection (P : Type) (line : Type) : Type := sorry

noncomputable def circumcircle (T : TriangleABC) : Type := sorry

theorem DE_eq_2KL
  (ABC : TriangleABC)
  (I : Type := incenter ABC)
  (D : Type := projection I (BC : Type))
  (E : Type := projection I (CA : Type))
  (F : Type := projection I (AB : Type))
  (K : Type := reflection D (AI : Type))
  (circum_BFK : Type := circumcircle (TriangleABC.mk B F K))
  (circum_CEK : Type := circumcircle (TriangleABC.mk C E K))
  (L : Type := second_intersection circum_BFK circum_CEK)
  (h : (1 / 3 : ℝ) * BC = AC - AB)
  : distance D E = 2 * distance K L := 
  sorry

end DE_eq_2KL_l187_187783


namespace cube_loop_probability_l187_187564

-- Define the number of faces and alignments for a cube
def total_faces := 6
def stripe_orientations_per_face := 2

-- Define the total possible stripe combinations
def total_stripe_combinations := stripe_orientations_per_face ^ total_faces

-- Define the combinations for both vertical and horizontal loops
def vertical_and_horizontal_loop_combinations := 64

-- Define the probability space
def probability_at_least_one_each := vertical_and_horizontal_loop_combinations / total_stripe_combinations

-- The main theorem to state the probability of having at least one vertical and one horizontal loop
theorem cube_loop_probability : probability_at_least_one_each = 1 := by
  sorry

end cube_loop_probability_l187_187564


namespace smallest_S_value_l187_187932

def num_list := {x : ℕ // 1 ≤ x ∧ x ≤ 9}

def S (a b c : num_list) (d e f : num_list) (g h i : num_list) : ℕ :=
  a.val * b.val * c.val + d.val * e.val * f.val + g.val * h.val * i.val

theorem smallest_S_value :
  ∃ a b c d e f g h i : num_list,
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
  g ≠ h ∧ g ≠ i ∧
  h ≠ i ∧
  S a b c d e f g h i = 214 :=
sorry

end smallest_S_value_l187_187932


namespace probability_of_popped_white_is_12_over_17_l187_187748

noncomputable def probability_white_given_popped (white_kernels yellow_kernels : ℚ) (pop_white pop_yellow : ℚ) : ℚ :=
  let p_white_popped := white_kernels * pop_white
  let p_yellow_popped := yellow_kernels * pop_yellow
  let p_popped := p_white_popped + p_yellow_popped
  p_white_popped / p_popped

theorem probability_of_popped_white_is_12_over_17 :
  probability_white_given_popped (3/4) (1/4) (3/5) (3/4) = 12/17 :=
by
  sorry

end probability_of_popped_white_is_12_over_17_l187_187748


namespace polynomial_functional_equation_solution_l187_187275

noncomputable def is_solution (P : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ x : ℝ, P(x) = x^2 + c

theorem polynomial_functional_equation_solution (P : ℝ → ℝ) :
  (∀ x : ℝ, P(x+1) = P(x) + 2*x + 1) → is_solution P :=
by
  intro h
  sorry

end polynomial_functional_equation_solution_l187_187275


namespace gcd_max_value_l187_187017

theorem gcd_max_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1008) : 
  ∃ d, d = Nat.gcd a b ∧ d = 504 :=
by
  sorry

end gcd_max_value_l187_187017


namespace pet_store_cages_l187_187114

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 78) (h2 : sold_puppies = 30) (h3 : puppies_per_cage = 8) : 
  (initial_puppies - sold_puppies) / puppies_per_cage = 6 := 
by 
  sorry

end pet_store_cages_l187_187114


namespace sum_of_digits_is_18_l187_187757

-- Define the distinct digits
variables {a b e f t : ℕ}

-- Given conditions
axiom distinct_digits : a ≠ b ∧ a ≠ e ∧ a ≠ f ∧ a ≠ t ∧ b ≠ e ∧ b ≠ f ∧ b ≠ t ∧ e ≠ f ∧ e ≠ t ∧ f ≠ t
axiom six_equals_6 : (6 : ℕ)

-- The initial equation condition
axiom equation_condition :
  (10 * a + b) * (10000 * a + 1000 * b + 100 * six_equals_6 + 10 * e + f) =
    (100000 * six_equals_6 + 10000 * e + 1000 * f + 100 * t + 10 * b + a)

-- Known values based on the solution
axiom fly_equals_1 : f = 1
axiom god_equals_2 : a = 2
axiom boat_equals_5 : b = 5
axiom number_equals_4 : e = 4
axiom heaven_equals_0 : t = 0

-- The proof problem
theorem sum_of_digits_is_18 : a + b + six_equals_6 + e + f + t = 18 :=
by
  sorry

end sum_of_digits_is_18_l187_187757


namespace basketball_substitution_remainder_l187_187988

theorem basketball_substitution_remainder :
  let n := 1 + 50 + 2250 + 90000 + 3150000 in
  n % 1000 = 301 :=
by 
  let n := 1 + 50 + 2250 + 90000 + 3150000;
  have h : n = 3182301 := by rfl;
  exact (nat.mod_eq_of_lt (by norm_num : 3182301 < 1000000)) ▸ (show 3182301 % 1000 = 301, by norm_num)

end basketball_substitution_remainder_l187_187988


namespace Juanita_spends_more_l187_187360

def Grant_annual_spend : ℝ := 200
def weekday_spend : ℝ := 0.50
def sunday_spend : ℝ := 2.00
def weeks_in_year : ℝ := 52

def Juanita_weekly_spend : ℝ := (6 * weekday_spend) + sunday_spend
def Juanita_annual_spend : ℝ := weeks_in_year * Juanita_weekly_spend
def spending_difference : ℝ := Juanita_annual_spend - Grant_annual_spend

theorem Juanita_spends_more : spending_difference = 60 := by
  sorry

end Juanita_spends_more_l187_187360


namespace tan_30_deg_l187_187172

theorem tan_30_deg : 
  let θ := (30 : ℝ) * (Real.pi / 180)
  in Real.sin θ = 1 / 2 ∧ Real.cos θ = Real.sqrt 3 / 2 → Real.tan θ = Real.sqrt 3 / 3 :=
by
  intro h
  let th := θ
  have h1 : Real.sin th = 1 / 2 := And.left h
  have h2 : Real.cos th = Real.sqrt 3 / 2 := And.right h
  sorry

end tan_30_deg_l187_187172


namespace tan_30_deg_l187_187165

theorem tan_30_deg : 
  let θ := 30 * (Float.pi / 180) in  -- Conversion from degrees to radians
  Float.sin θ = 1 / 2 ∧ Float.cos θ = Float.sqrt 3 / 2 →
  Float.tan θ = Float.sqrt 3 / 3 := by
  intro h
  sorry

end tan_30_deg_l187_187165


namespace abs_a_plus_2_always_positive_l187_187539

theorem abs_a_plus_2_always_positive (a : ℝ) : |a| + 2 > 0 := 
sorry

end abs_a_plus_2_always_positive_l187_187539


namespace series_sum_eq_three_halves_l187_187617

noncomputable def series (k : ℕ) : ℝ := k * (k + 1) / (2 * 3^k)

theorem series_sum_eq_three_halves :
  has_sum (λ k, series k) (3 / 2) :=
sorry

end series_sum_eq_three_halves_l187_187617


namespace carina_coffee_total_l187_187603

theorem carina_coffee_total :
  (∃ (n5 n10 : ℕ), n10 = 7 ∧ n5 = n10 + 2 ∧ 5 * n5 + 10 * n10 = 115) :=
by
  let n10 := 7
  let n5 := n10 + 2
  have h5 : 5 * n5 = 5 * (n10 + 2) := rfl
  have h10 : 10 * n10 = 10 * 7 := rfl
  have h : 5 * (n10 + 2) + 10 * 7 = 115 := by sorry
  use [n5, n10]
  exact ⟨rfl, rfl, h⟩

end carina_coffee_total_l187_187603


namespace coeff_x2_in_expansion_l187_187402

theorem coeff_x2_in_expansion : 
  binomial_expansion.coeff (x - (1 / (4 * x))) 6 x 2 = 15 / 16 :=
by
  sorry

end coeff_x2_in_expansion_l187_187402


namespace least_number_to_add_l187_187047

theorem least_number_to_add (a b : ℤ) (d : ℤ) (h : a = 1054) (hb : b = 47) (hd : d = 27) :
  ∃ n : ℤ, (a + d) % b = 0 :=
by
  sorry

end least_number_to_add_l187_187047


namespace smallest_possible_s_l187_187827

noncomputable def all_x_i (x: ℕ → set ℕ): Prop :=
  ∀ i: ℕ, 1 ≤ i → i < 100 → (x i ∩ x (i + 1) = ∅) ∧ (x i ∪ x (i + 1) ≠ {1, 2, 3, 4, 5, 6, 7, 8}) ∧
    ∀ j: ℕ, 1 ≤ j → j < i → (x i ≠ x j)

theorem smallest_possible_s : 
  ∃ S: set ℕ, S = {1, 2, 3, 4, 5, 6, 7, 8} ∧ 
  ∃ x: ℕ → set ℕ, all_x_i x → S.card = 8 :=
begin
  sorry
end

end smallest_possible_s_l187_187827


namespace ball_box_distribution_l187_187457

theorem ball_box_distribution:
  ∃ (C : ℕ → ℕ → ℕ) (A : ℕ → ℕ → ℕ),
  C 4 2 * A 3 3 = sorry := 
by sorry

end ball_box_distribution_l187_187457


namespace absolute_value_of_sum_C_D_base5_l187_187279

theorem absolute_value_of_sum_C_D_base5 (C D : ℕ) (h1 : C - 3 = 1) (h2 : C = 4) (h3 : 2 - 2 = 3 . -. (7 - 2) = 3) (h4 : D - C = 2) (h5 : D = 0) : |C + D| = 4 := by
sorry

end absolute_value_of_sum_C_D_base5_l187_187279


namespace smallest_n_condition_l187_187069

theorem smallest_n_condition (n : ℕ) : (4 * n) ∣ (n^2) ∧ (5 * n) ∣ (u^3) → n = 100 :=
by
  sorry

end smallest_n_condition_l187_187069


namespace eq_tangent_line_at_1_l187_187912

def f (x : ℝ) : ℝ := x^4 - 2 * x^3

def tangent_line (x : ℝ) : ℝ := -2 * x + 1

theorem eq_tangent_line_at_1 : 
  ∃ (m : ℝ) (c : ℝ), m = -2 ∧ c = 1 ∧ ∀ x, tangent_line x = m * x + c :=
by
  use -2
  use 1
  split
  . rfl
  split
  . rfl
  intro x
  rfl

end eq_tangent_line_at_1_l187_187912


namespace remainder_of_sum_mod_13_l187_187536

theorem remainder_of_sum_mod_13 {a b c d e : ℕ} 
  (h1 : a % 13 = 3) 
  (h2 : b % 13 = 5)
  (h3 : c % 13 = 7) 
  (h4 : d % 13 = 9) 
  (h5 : e % 13 = 11) : 
  (a + b + c + d + e) % 13 = 9 :=
by
  sorry

end remainder_of_sum_mod_13_l187_187536


namespace evaluate_64_pow_fifth_sixth_l187_187217

theorem evaluate_64_pow_fifth_sixth : 64 ^ (5 / 6) = 32 := by
  have h : 64 = 2 ^ 6 := by sorry
  calc 64 ^ (5 / 6) = (2 ^ 6) ^ (5 / 6) : by rw [h]
              ...   = 2 ^ (6 * (5 / 6))  : by sorry
              ...   = 2 ^ 5              : by sorry
              ...   = 32                 : by sorry

end evaluate_64_pow_fifth_sixth_l187_187217


namespace constants_for_sin_identity_l187_187186

theorem constants_for_sin_identity :
  (∃ (a b : ℝ), (∀ θ : ℝ, sin θ ^ 3 = a * sin (3 * θ) + b * sin θ)) → 
  (a = -1/4 ∧ b = 3/4) := 
by
  intros h
  have h' := h 0
  sorry

end constants_for_sin_identity_l187_187186


namespace bus_total_people_l187_187863

def number_of_boys : ℕ := 50
def additional_girls (b : ℕ) : ℕ := (2 * b) / 5
def number_of_girls (b : ℕ) : ℕ := b + additional_girls b
def total_people (b g : ℕ) : ℕ := b + g + 3  -- adding 3 for the driver, assistant, and teacher

theorem bus_total_people : total_people number_of_boys (number_of_girls number_of_boys) = 123 :=
by
  sorry

end bus_total_people_l187_187863


namespace cos_triple_sum_div_l187_187732

theorem cos_triple_sum_div {A B C : ℝ} (h : Real.cos A + Real.cos B + Real.cos C = 0) : 
  (Real.cos (3 * A) + Real.cos (3 * B) + Real.cos (3 * C)) / (Real.cos A * Real.cos B * Real.cos C) = 12 :=
by
  sorry

end cos_triple_sum_div_l187_187732


namespace smallest_positive_period_increasing_interval_center_of_symmetry_l187_187341

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x)^2 + 4 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem smallest_positive_period : (∀ x : ℝ, f (x + Real.pi) = f x) :=
by sorry

theorem increasing_interval (k : ℤ) : 
  (∀ x : ℝ, k * Real.pi - Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 3 → f' x > 0) :=
by sorry

theorem center_of_symmetry (k : ℤ) : 
  (f (k * Real.pi / 2 + Real.pi / 12) = 2) :=
by sorry

end smallest_positive_period_increasing_interval_center_of_symmetry_l187_187341


namespace stacy_grew_more_l187_187887

variable (initial_height_stacy current_height_stacy brother_growth stacy_growth_more : ℕ)

-- Conditions
def stacy_initial_height : initial_height_stacy = 50 := by sorry
def stacy_current_height : current_height_stacy = 57 := by sorry
def brother_growth_last_year : brother_growth = 1 := by sorry

-- Compute Stacy's growth
def stacy_growth : ℕ := current_height_stacy - initial_height_stacy

-- Prove the difference in growth
theorem stacy_grew_more :
  stacy_growth - brother_growth = stacy_growth_more → stacy_growth_more = 6 := 
by sorry

end stacy_grew_more_l187_187887


namespace unique_rhombus_property_not_in_rectangle_l187_187972

-- Definitions of properties for a rhombus and a rectangle
def is_rhombus (sides_equal : Prop) (opposite_sides_parallel : Prop) (opposite_angles_equal : Prop)
  (diagonals_perpendicular_and_bisect : Prop) : Prop :=
  sides_equal ∧ opposite_sides_parallel ∧ opposite_angles_equal ∧ diagonals_perpendicular_and_bisect

def is_rectangle (opposite_sides_equal_and_parallel : Prop) (all_angles_right : Prop)
  (diagonals_equal_and_bisect : Prop) : Prop :=
  opposite_sides_equal_and_parallel ∧ all_angles_right ∧ diagonals_equal_and_bisect

-- Proof objective: Prove that the unique property of a rhombus is the perpendicular and bisecting nature of its diagonals
theorem unique_rhombus_property_not_in_rectangle :
  ∀ (sides_equal opposite_sides_parallel opposite_angles_equal
      diagonals_perpendicular_and_bisect opposite_sides_equal_and_parallel
      all_angles_right diagonals_equal_and_bisect : Prop),
  is_rhombus sides_equal opposite_sides_parallel opposite_angles_equal diagonals_perpendicular_and_bisect →
  is_rectangle opposite_sides_equal_and_parallel all_angles_right diagonals_equal_and_bisect →
  diagonals_perpendicular_and_bisect ∧ ¬diagonals_equal_and_bisect :=
by
  sorry

end unique_rhombus_property_not_in_rectangle_l187_187972


namespace number_of_integer_values_of_a_l187_187649

theorem number_of_integer_values_of_a (a : ℤ) : 
  (∃ x : ℤ, x^2 + a * x + 9 * a = 0) ↔ 
  (∃ (a_values : Finset ℤ), a_values.card = 6 ∧ ∀ a ∈ a_values, ∃ x : ℤ, x^2 + a * x + 9 * a = 0) :=
by
  sorry

end number_of_integer_values_of_a_l187_187649


namespace jordan_oreos_l187_187774

theorem jordan_oreos 
  (x : ℕ) 
  (h1 : let james := 3 + 2 * x in james + x = 36) : 
  x = 11 :=
by 
  -- Proof will go here
  sorry

end jordan_oreos_l187_187774


namespace jordan_oreos_l187_187775

theorem jordan_oreos 
  (x : ℕ) 
  (h1 : let james := 3 + 2 * x in james + x = 36) : 
  x = 11 :=
by 
  -- Proof will go here
  sorry

end jordan_oreos_l187_187775


namespace coin_distribution_possible_l187_187099

theorem coin_distribution_possible (n : ℕ) (k : ℕ) :
  (n = 5 * k) ↔
  ∃ (boxes : Fin 5 → Finset (Fin n)),
    (∀ (i : Fin 5), boxes i.card = m) ∧
    (∀ (i : Fin 5), (∑ j in boxes i, j.val) = S) ∧
    (∀ (i j : Fin 5), i ≠ j → Finset.univ ⊆ boxes i ∪ boxes j) ∧
    (∀ (d : Fin n), ∃! (b : Fin 5), d ∈ boxes b → (∀ (b1 b2 b3 b4 : Fin 5), d ∉ boxes b1 ∩ boxes b2 ∩ boxes b3 ∩ boxes b4)) := sorry

end coin_distribution_possible_l187_187099


namespace sam_added_later_buckets_l187_187450

variable (initial_buckets : ℝ) (total_buckets : ℝ)

def buckets_added_later (initial_buckets total_buckets : ℝ) : ℝ :=
  total_buckets - initial_buckets

theorem sam_added_later_buckets :
  initial_buckets = 1 ∧ total_buckets = 9.8 → buckets_added_later initial_buckets total_buckets = 8.8 := by
  sorry

end sam_added_later_buckets_l187_187450


namespace harmonic_sum_divisibility_l187_187428

-- Definitions based on conditions
def harmonic_sum (n : ℕ) : ℚ := (∑ i in (finset.range (n+1)).filter(λ k, k > 0), 1 / i)

def rel_prime_pos (a b : ℕ) : Prop := nat.coprime a b ∧ a > 0 ∧ b > 0

-- The theorem based on equivalent proof statement
theorem harmonic_sum_divisibility (n : ℕ) (p_n q_n : ℕ) (h : harmonic_sum n = p_n / q_n) (hpq : rel_prime_pos p_n q_n) : 
  ¬ (5 ∣ q_n) ↔ n ∈ finset.range 5 ∪ (finset.range (25) \ finset.range (20)) ∪ (finset.range (105) \ finset.range (100)) ∪ (finset.range (125) \ finset.range (120)) :=
sorry

end harmonic_sum_divisibility_l187_187428


namespace tan_30_deg_l187_187166

theorem tan_30_deg : 
  let θ := 30 * (Float.pi / 180) in  -- Conversion from degrees to radians
  Float.sin θ = 1 / 2 ∧ Float.cos θ = Float.sqrt 3 / 2 →
  Float.tan θ = Float.sqrt 3 / 3 := by
  intro h
  sorry

end tan_30_deg_l187_187166


namespace number_of_solutions_eq_1190_l187_187632

theorem number_of_solutions_eq_1190 :
  let w' = w - 2,
      x' = x - 3,
      y' = y - 1,
      z' = z - 2,
      t' = t - 4 in
  (∃ (w x y z t : ℕ), 
    w ≥ 2 ∧ 
    x ≥ 3 ∧ 
    y ≥ 1 ∧ 
    z ≥ 2 ∧ 
    t ≥ 4 ∧ 
    w + x + y + z + t = 25) ↔
  (∃ (w' x' y' z' t' : ℕ), 
    w' + x' + y' + z' + t' = 13 ∧ 
    ∑ i in finset.range 14, w' = 1190) := sorry

end number_of_solutions_eq_1190_l187_187632


namespace vertex_A_not_unique_l187_187406

-- Define a structure for the midpoint and angle bisector conditions in triangle ABC
structure TriangleData where
  A B C D E F L : Type
  midpoint_AB : D = (A + B) / 2
  midpoint_AC : E = (A + C) / 2
  midpoint_AL : F = (A + L) / 2
  DF_len : ∥D - F∥ = 1
  EF_len : ∥E - F∥ = 2
  DF_horizontal : D.y = F.y
  A_above_DF : A.y > D.y

-- Prove that the position of vertex A cannot be uniquely determined
theorem vertex_A_not_unique (data : TriangleData) : 
  ¬ ∃ (A₁ A₂ : data.A), A₁ = A₂ := 
sorry

end vertex_A_not_unique_l187_187406


namespace water_left_after_experiment_l187_187583

theorem water_left_after_experiment (initial_water : ℝ) (used_water : ℝ) (result_water : ℝ) 
  (h1 : initial_water = 3) 
  (h2 : used_water = 9 / 4) 
  (h3 : result_water = 3 / 4) : 
  initial_water - used_water = result_water := by
  sorry

end water_left_after_experiment_l187_187583


namespace tan_30_eq_sqrt3_div_3_l187_187162

/-- Statement that proves the value of tang of 30 degrees, given the cosine
    and sine values. -/
theorem tan_30_eq_sqrt3_div_3 
  (cos_30 : Real) (sin_30 : Real) 
  (hcos : cos_30 = Real.sqrt 3 / 2) 
  (hsin : sin_30 = 1 / 2) : 
    Real.tan 30 = Real.sqrt 3 / 3 := 
by 
  sorry

end tan_30_eq_sqrt3_div_3_l187_187162


namespace bryden_total_l187_187100

def face_value_quarter : ℝ := 0.25
def number_of_quarters : ℝ := 6
def percentage_multiplier : ℝ := 30

theorem bryden_total (face_value_quarter = 0.25) (number_of_quarters = 6) (percentage_multiplier = 30) :
  percentage_multiplier * (number_of_quarters * face_value_quarter) = 45 :=
by
  sorry

end bryden_total_l187_187100


namespace total_people_on_bus_l187_187859

-- Definitions of the conditions
def num_boys : ℕ := 50
def num_girls : ℕ := (2 / 5 : ℚ) * num_boys
def num_students : ℕ := num_boys + num_girls.toNat
def num_non_students : ℕ := 3 -- Mr. Gordon, the driver, and the assistant

-- The theorem to be proven
theorem total_people_on_bus : num_students + num_non_students = 123 := by
  sorry

end total_people_on_bus_l187_187859


namespace Alfred_profit_percentage_l187_187129

-- Define the conditions
def purchase_price : ℝ := 4700
def initial_repair_cost : ℝ := 0.1 * purchase_price
def maintenance_cost : ℝ := 500
def total_repair_cost : ℝ := initial_repair_cost + maintenance_cost
def safety_upgrade_cost : ℝ := 0.05 * total_repair_cost
def total_cost : ℝ := purchase_price + total_repair_cost + safety_upgrade_cost
def selling_price_before_tax : ℝ := 5800
def sales_tax : ℝ := 0.12 * selling_price_before_tax
def total_selling_price : ℝ := selling_price_before_tax + sales_tax
def profit : ℝ := total_selling_price - total_cost
def profit_percentage : ℝ := (profit / total_cost) * 100

-- State the theorem
theorem Alfred_profit_percentage : profit_percentage ≈ 13.60 := by
  sorry

end Alfred_profit_percentage_l187_187129


namespace find_divisor_l187_187971

-- Definitions of conditions
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

variable {x y : ℕ}

-- Condition 1: When x is divided by 61, the remainder is 24
def condition1 : Prop := ∃ k : ℕ, x = 61 * k + 24

-- Condition 2: When x is divided by y, the remainder is 4
def condition2 : Prop := ∃ m : ℕ, x = y * m + 4

-- Prove that y = 9
theorem find_divisor (H1 : condition1) (H2 : condition2) : y = 9 :=
sorry

end find_divisor_l187_187971


namespace foreman_can_establish_corr_foreman_cannot_with_less_l187_187393

-- Define the given conditions:
def num_rooms (n : ℕ) := 2^n
def num_checks (n : ℕ) := 2 * n

-- Part (a)
theorem foreman_can_establish_corr (n : ℕ) : 
  ∃ (c : ℕ), c = num_checks n ∧ (c ≥ 2 * n) :=
by
  sorry

-- Part (b)
theorem foreman_cannot_with_less (n : ℕ) : 
  ¬ (∃ (c : ℕ), c = 2 * n - 1 ∧ (c < 2 * n)) :=
by
  sorry

end foreman_can_establish_corr_foreman_cannot_with_less_l187_187393


namespace integer_values_of_a_l187_187643

theorem integer_values_of_a : 
  ∃ (a : Set ℤ), (∀ x, x ∈ a → ∃ (y z : ℤ), x^2 + x * y + 9 * y = 0) ∧ (a.card = 6) :=
by
  sorry

end integer_values_of_a_l187_187643


namespace problem_statement_l187_187702

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - Real.log x

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : ∀ x, 0 < x → f a b x ≥ f a b 1) : 
  Real.log a < -2 * b :=
by
  sorry

end problem_statement_l187_187702


namespace find_weight_of_b_l187_187897

theorem find_weight_of_b (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) : B = 31 :=
sorry

end find_weight_of_b_l187_187897


namespace machine_production_l187_187595

theorem machine_production:
  ∃ (n : ℕ) (productions : Fin n → ℕ),
    n = 5 ∧
    ({35, 39, 40, 49, 44, 46, 30, 41, 32, 36} = { i.2 + j.2 | (i, j) ∈ (Finset.univ.product Finset.univ).filter (λ p, p.fst < p.snd) }) ∧
    (Set.ofFin (Fin.map productions Finset.univ)) = {13, 17, 19, 22, 27} :=
begin
  sorry  -- Proof to be filled here
end

end machine_production_l187_187595


namespace sum_of_roots_l187_187684

theorem sum_of_roots (α β : ℝ)
  (hα : α^3 - 3*α^2 + 5*α - 4 = 0)
  (hβ : β^3 - 3*β^2 + 5*β - 2 = 0) :
  α + β = 2 :=
sorry

end sum_of_roots_l187_187684


namespace option_d_correct_l187_187838

theorem option_d_correct (q : ℕ) (hq : q.prime) (hodd : q % 2 = 1) : 
  (q - 1)^(q - 2) + 1 ≡ 0 [MOD q] :=
  sorry

end option_d_correct_l187_187838


namespace downstream_speed_is_35_l187_187568

variables (upstream downstream still_water : ℕ)

-- Define the given conditions
def upstream_speed : upstream = 15 := sorry
def still_water_speed : still_water = 25 := sorry

-- Define the speed of the man rowing downstream
def downstream_speed : downstream = still_water + (still_water - upstream) := sorry

-- The proof goal statement
theorem downstream_speed_is_35 : downstream = 35 :=
by
  rw [downstream_speed, upstream_speed, still_water_speed]
  sorry

end downstream_speed_is_35_l187_187568


namespace elevation_angle_second_ship_l187_187526

-- Assume h is the height of the lighthouse.
def h : ℝ := 100

-- Assume d_total is the distance between the two ships.
def d_total : ℝ := 273.2050807568877

-- Assume θ₁ is the angle of elevation from the first ship.
def θ₁ : ℝ := 30

-- Assume θ₂ is the angle of elevation from the second ship.
def θ₂ : ℝ := 45

-- Prove that angle of elevation from the second ship is 45 degrees.
theorem elevation_angle_second_ship : θ₂ = 45 := by
  sorry

end elevation_angle_second_ship_l187_187526


namespace sum_of_roots_l187_187724

theorem sum_of_roots (x : ℝ) :
  (x + 2) * (x - 3) = 16 →
  ∃ a b : ℝ, (a ≠ x ∧ b ≠ x ∧ (x - a) * (x - b) = 0) ∧
             (a + b = 1) :=
by
  intro h
  sorry

end sum_of_roots_l187_187724


namespace evaluate_pow_l187_187253

theorem evaluate_pow : (64 : ℝ) = (8 : ℝ) ^ 2 → (8 : ℝ) = (2 : ℝ) ^ 3 → (64 : ℝ) ^ (5 / 6) = 32 :=
by
  intros h1 h2
  rw h1
  rw h2
  have h3 : (2 : ℝ)^3 ^ 2 = (2 : ℝ) ^ 6 := by ring_exp
  rw h3
  sorry

end evaluate_pow_l187_187253


namespace five_b_value_l187_187731

theorem five_b_value (a b : ℚ) 
  (h1 : 3 * a + 4 * b = 4) 
  (h2 : a = b - 3) : 
  5 * b = 65 / 7 := 
by
  sorry

end five_b_value_l187_187731


namespace probability_at_least_half_of_six_children_are_girls_l187_187415

-- Define the probability space and binomial distribution
noncomputable def probability_at_least_half_are_girls : ℚ :=
  let n := 6
  let p := 1 / 2
  (∑ k in finset.range (n + 1), if 3 ≤ k then (nat.choose n k * (p ^ k) * ((1 - p) ^ (n - k))) else 0 : ℚ)

theorem probability_at_least_half_of_six_children_are_girls :
  probability_at_least_half_are_girls = 21 / 32 :=
by
  sorry

end probability_at_least_half_of_six_children_are_girls_l187_187415


namespace reciprocal_of_2022_l187_187005

theorem reciprocal_of_2022 : 1 / 2022 = (1 : ℝ) / 2022 :=
sorry

end reciprocal_of_2022_l187_187005


namespace centroid_positions_count_l187_187889

open Set

def point := (ℝ × ℝ)

def rectangle_vertices : Set point := { (0,0), (12,0), (12,8), (0,8) }

def points_on_perimeter : Set point :=
  { (0.8 * i, 0) | i in 0..15 } ∪
  { (12, (8 / 7) * j) | j in 0..7 } ∪
  { (12 - 0.8 * k, 8) | k in 0..15 } ∪
  { (0, 8 - (8 / 7) * l) | l in 0..7 }

def non_collinear (a b c : point) : Prop :=
  ¬ ∃ k : ℝ, (b.1 - a.1) = k * (c.1 - a.1) ∧ (b.2 - a.2) = k * (c.2 - a.2)

theorem centroid_positions_count :
  (finset.univ : finset point).filter (λ abc, non_collinear abc.1 abc.2 abc.3) .image (λ abc, ((abc.1.1 + abc.2.1 + abc.3.1) / 3, (abc.1.2 + abc.2.2 + abc.3.2) / 3)).card = 925 :=
sorry

end centroid_positions_count_l187_187889


namespace tan_30_eq_sqrt3_div3_l187_187154

theorem tan_30_eq_sqrt3_div3 (sin_30_cos_30 : ℝ → ℝ → Prop)
  (h1 : sin_30_cos_30 (1 / 2) (Real.sqrt 3 / 2)) :
  ∃ t, t = Real.tan (Real.pi / 6) ∧ t = Real.sqrt 3 / 3 :=
by
  existsi Real.tan (Real.pi / 6)
  sorry

end tan_30_eq_sqrt3_div3_l187_187154


namespace eval_64_pow_5_over_6_l187_187202

theorem eval_64_pow_5_over_6 (h : 64 = 2^6) : 64^(5/6) = 32 := 
by 
  sorry

end eval_64_pow_5_over_6_l187_187202


namespace find_side_length_b_l187_187760

-- Define the conditions given in the problem
variables (a c b : ℝ) (C : ℝ)
def is_side_length (length : ℝ) := length > 0

-- Given conditions
def given_conditions : Prop :=
  a = 2 ∧ c = 2 * real.sqrt 3 ∧ C = real.pi / 3

-- Prove that, given the conditions, b equals 4
theorem find_side_length_b (h : given_conditions) : b = 4 :=
sorry

end find_side_length_b_l187_187760


namespace exchange_needed_probability_l187_187581

theorem exchange_needed_probability :
  let toys := List.finRange 10
  let prices := List.map (fun i => 30 * (i + 1)) toys
  let favorite_toy_price := 240
  let total_quarters := 10
  let has_change_probability := (1 : ℚ) - (40320 + 5040) / 3628800
  toys.length = 10 ∧
  (∀ i, prices.nth i < prices.nth (i + 1) ∧ prices.nth i = 30 * (i + 1)) ∧
  favorite_toy_price = 240 ∧
  total_quarters = 10 ->
  has_change_probability = 79 / 80 :=
by
  intros,
  sorry

end exchange_needed_probability_l187_187581


namespace parallel_vectors_equal_ratios_l187_187045

theorem parallel_vectors_equal_ratios (x : ℝ) :
  let a := (2 : ℝ, 1 : ℝ),
      b := (x, -1 : ℝ) in
  (a.1 * b.2 = a.2 * b.1) → x = -2 :=
by
  sorry

end parallel_vectors_equal_ratios_l187_187045


namespace smaller_of_x_and_y_l187_187525

theorem smaller_of_x_and_y 
  (x y a b c d : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < b + 1) 
  (h3 : x + y = c) 
  (h4 : x - y = d) 
  (h5 : x / y = a / (b + 1)) :
  min x y = (ac/(a + b + 1)) := 
by
  sorry

end smaller_of_x_and_y_l187_187525


namespace trigonometric_identity_l187_187728

noncomputable def cos_value (α : ℝ) : Prop :=
  cos (5 * π / 6 - α) = -1 / 3

theorem trigonometric_identity (α : ℝ) (h : sin (π / 3 - α) = 1 / 3) : cos_value α :=
by
  sorry

end trigonometric_identity_l187_187728


namespace curve_is_circle_l187_187625

theorem curve_is_circle (r : ℝ) (r_eq_5 : r = 5) : 
  (∃ (c : ℝ), c = 0 ∧ ∀ (θ : ℝ), (r, θ) ∈ set_of (λ (p : ℝ × ℝ), p.1 = 5)) :=
sorry

end curve_is_circle_l187_187625


namespace effective_rate_proof_l187_187549

noncomputable def nominal_rate : ℝ := 0.08
noncomputable def compounding_periods : ℕ := 2
noncomputable def effective_annual_rate (i : ℝ) (n : ℕ) : ℝ := (1 + i / n) ^ n - 1

theorem effective_rate_proof :
  effective_annual_rate nominal_rate compounding_periods = 0.0816 :=
by
  sorry

end effective_rate_proof_l187_187549


namespace find_radius_of_third_circle_l187_187522

noncomputable def radius_of_third_circle_equals_shaded_region (r1 r2 r3 : ℝ) : Prop :=
  let area_large := Real.pi * (r2 ^ 2)
  let area_small := Real.pi * (r1 ^ 2)
  let area_shaded := area_large - area_small
  let area_third_circle := Real.pi * (r3 ^ 2)
  area_shaded = area_third_circle

theorem find_radius_of_third_circle (r1 r2 : ℝ) (r1_eq : r1 = 17) (r2_eq : r2 = 27) : ∃ r3 : ℝ, r3 = 10 * Real.sqrt 11 ∧ radius_of_third_circle_equals_shaded_region r1 r2 r3 := 
by
  sorry

end find_radius_of_third_circle_l187_187522


namespace value_taken_away_l187_187733

theorem value_taken_away (n x : ℕ) (h1 : n = 4) (h2 : 2 * n + 20 = 8 * n - x) : x = 4 :=
by
  sorry

end value_taken_away_l187_187733


namespace k_sq_geq_25_over_4_l187_187192

theorem k_sq_geq_25_over_4
  (a1 a2 a3 a4 a5 k : ℝ)
  (h1 : |a1 - a2| ≥ 1 ∧ |a1 - a3| ≥ 1 ∧ |a1 - a4| ≥ 1 ∧ |a1 - a5| ≥ 1 ∧
       |a2 - a3| ≥ 1 ∧ |a2 - a4| ≥ 1 ∧ |a2 - a5| ≥ 1 ∧
       |a3 - a4| ≥ 1 ∧ |a3 - a5| ≥ 1 ∧
       |a4 - a5| ≥ 1)
  (h2 : a1 + a2 + a3 + a4 + a5 = 2 * k)
  (h3 : a1^2 + a2^2 + a3^2 + a4^2 + a5^2 = 2 * k^2) :
  k^2 ≥ 25 / 4 :=
sorry

end k_sq_geq_25_over_4_l187_187192


namespace euler_line_vertex_C_l187_187890

theorem euler_line_vertex_C :
  (∀ C : ℝ × ℝ, 
    let A := (-4, 0)
    let B := (0, 4)
    let euler_line := λ p : ℝ × ℝ, p.1 - p.2 + 2 = 0
    let centroid := λ (A B C : ℝ × ℝ), ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
    let circumcenter_line := λ (A B : ℝ × ℝ), (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
    in circumcenter_line A B C ∧ euler_line (centroid A B C) →
      (C = (0, -2)) ∨ (C = (2, 0))
  ) :=
begin
  sorry
end

end euler_line_vertex_C_l187_187890


namespace sufficient_but_not_necessary_l187_187665

variable (m : ℝ)

def P : Prop := ∀ x : ℝ, x^2 - 4*x + 3*m > 0
def Q : Prop := ∀ x : ℝ, 3*x^2 + 4*x + m ≥ 0

theorem sufficient_but_not_necessary : (P m → Q m) ∧ ¬(Q m → P m) :=
by
  sorry

end sufficient_but_not_necessary_l187_187665


namespace num_false_propositions_l187_187587

theorem num_false_propositions (a : ℝ) : 
  let P := (a > -3 → a > -6),
      Q := (a > -6 → a > -3),
      R := (a ≤ -3 → a ≤ -6),
      S := (a ≤ -6 → a ≤ -3)
  in P ∧ ¬Q ∧ ¬R ∧ S → 2 :=
by
  intros P Q R S hyp,
  sorry

end num_false_propositions_l187_187587


namespace find_k_l187_187734

noncomputable def k := 3

theorem find_k :
  (∀ x : ℝ, (Real.sin x ^ k) * (Real.sin (k * x)) + (Real.cos x ^ k) * (Real.cos (k * x)) = Real.cos (2 * x) ^ k) ↔ k = 3 :=
sorry

end find_k_l187_187734


namespace tangent_line_acute_probability_l187_187674

-- Define the function and interval
def f (x : ℝ) := x^2 + x

-- Derivative of the function
def f' (x : ℝ) := 2 * x + 1

noncomputable def acute_angle_probability : ℝ :=
  (1 - (-1/2)) / (1 - (-1))

theorem tangent_line_acute_probability :
  ∀ (a : ℝ), (a ∈ set.Icc (-1 : ℝ) (1 : ℝ)) → 
  (∃ p : ℝ, p = acute_angle_probability ∧ p = 3 / 4) := by
  -- Define probable regions and conditions, followed by its calculation
  sorry

end tangent_line_acute_probability_l187_187674


namespace three_digit_integers_with_conditions_l187_187368

theorem three_digit_integers_with_conditions :
  ∃ count : ℕ, count = 16 ∧ ∀ n : ℕ,
    100 ≤ n ∧ n < 1000 ∧ 
    (∀ i j k : ℕ, n = 100 * i + 10 * j + k → i ≠ j ∧ j ≠ k ∧ i ≠ k) ∧
    (∀ i j k : ℕ, n = 100 * i + 10 * j + k → i ≠ 0) ∧
    (n % 4 = 0) ∧
    (∀ i j k : ℕ, n = 100 * i + 10 * j + k → i ≤ 4 ∧ j ≤ 4 ∧ k ≤ 4 ∧ (i = 4 ∨ j = 4 ∨ k = 4))
    :=
begin
  sorry
end

end three_digit_integers_with_conditions_l187_187368


namespace part1_part2_l187_187306

variable (a : ℝ)
def f (x : ℝ) : ℝ := a * Real.log (x - 1) + x
def f' (x : ℝ) : ℝ := a / (x - 1) + 1

theorem part1 (h1 : f' 2 = 2) : a = 1 ∧ ∃ g : ℝ → ℝ, (∀ x, g x = x - 1) :=
by
  sorry

def g (x : ℝ) : ℝ := x - 1
def h (m : ℝ) (x : ℝ) : ℝ := m * f' x + g x + 1

theorem part2 (h2 : ∀ x, 2 ≤ x → x ≤ 4 → h  _ x > 0) : ∀ m, m > -1 :=
by
  sorry

end part1_part2_l187_187306


namespace probability_white_ball_l187_187385

theorem probability_white_ball :
  ∀ (balls : List String) (first_draw : String),
    (balls.length = 20 ∧
    balls.count (:="red") = 10 ∧
    balls.count (:="white") = 10 ∧
    first_draw = "red" ∧
    (balls.erase_first "red").length = 19 ∧
    (balls.erase_first "red").count (:="red") = 9 ∧
    (balls.erase_first "red").count (:="white") = 10) →
    (probability (balls.erase_first "red") "white" = 10 / 19) :=
sorry

end probability_white_ball_l187_187385


namespace rain_at_least_once_l187_187494

theorem rain_at_least_once (p : ℚ) (h : p = 3/4) : 
    (1 - (1 - p)^4) = 255/256 :=
by
  sorry

end rain_at_least_once_l187_187494


namespace tan_30_eq_sqrt3_div3_l187_187152

theorem tan_30_eq_sqrt3_div3 (sin_30_cos_30 : ℝ → ℝ → Prop)
  (h1 : sin_30_cos_30 (1 / 2) (Real.sqrt 3 / 2)) :
  ∃ t, t = Real.tan (Real.pi / 6) ∧ t = Real.sqrt 3 / 3 :=
by
  existsi Real.tan (Real.pi / 6)
  sorry

end tan_30_eq_sqrt3_div3_l187_187152


namespace james_needs_to_sell_12_coins_l187_187766

theorem james_needs_to_sell_12_coins:
  ∀ (num_coins : ℕ) (initial_price new_price : ℝ),
  num_coins = 20 ∧ initial_price = 15 ∧ new_price = initial_price + (2 / 3) * initial_price →
  (num_coins * initial_price) / new_price = 12 :=
by
  intros num_coins initial_price new_price h
  obtain ⟨hc1, hc2, hc3⟩ := h
  sorry

end james_needs_to_sell_12_coins_l187_187766


namespace evaluate_pow_l187_187256

theorem evaluate_pow : (64 : ℝ) = (8 : ℝ) ^ 2 → (8 : ℝ) = (2 : ℝ) ^ 3 → (64 : ℝ) ^ (5 / 6) = 32 :=
by
  intros h1 h2
  rw h1
  rw h2
  have h3 : (2 : ℝ)^3 ^ 2 = (2 : ℝ) ^ 6 := by ring_exp
  rw h3
  sorry

end evaluate_pow_l187_187256


namespace parabola_intersection_sum_l187_187484

theorem parabola_intersection_sum :
  let intersections := {(x, y) | y = (x + 1)^2 ∧ x + 4 = (y - 3)^2} in
  let xs := {x | ∃ y, (x, y) ∈ intersections} in
  let ys := {y | ∃ x, (x, y) ∈ intersections} in
  ∑ x in xs, x + ∑ y in ys, y = 8 :=
by
  sorry

end parabola_intersection_sum_l187_187484


namespace max_two_scoop_sundaes_l187_187132

theorem max_two_scoop_sundaes (V C S M : Type) (vanilla chocolate strawberry mint : V) (kinds : Finset V)
  (h_total : kinds.card = 8)
  (h_v_c : ¬(vanilla, chocolate) ∈ kinds.product kinds)
  (h_s_m : ¬(strawberry, mint) ∈ kinds.product kinds) :
  ∀ (sundaes : Finset (V × V)), sundae.card = 26 := 
by sorry

end max_two_scoop_sundaes_l187_187132


namespace lateral_surface_area_volume_larger_part_l187_187922

-- Define the cylinder height and radius
def cylinder_radius_height (a : ℝ) : Prop :=
  a > 0

-- Define the condition of the arc division
def arc_division_condition (r : ℝ) : Prop :=
  r > 0

-- Prove the lateral surface area of the larger part of the cylinder
theorem lateral_surface_area (a : ℝ) (hcyl : cylinder_radius_height a) (hdiv : arc_division_condition a) :
  2 * π * a^2 = 2 * π * a^2 :=
by
  sorry

-- Prove the volume of the larger part of the cylinder
theorem volume_larger_part (a : ℝ) (hcyl : cylinder_radius_height a) (hdiv : arc_division_condition a) :
  ((2 * π * a^3 + 3 * real.sqrt 3 * a^3) / 6) = (a^3 / 6) * (2 * π + 3 * real.sqrt 3) :=
by
  sorry

end lateral_surface_area_volume_larger_part_l187_187922


namespace max_value_amc_am_mc_ca_l187_187793

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end max_value_amc_am_mc_ca_l187_187793


namespace money_left_l187_187874

def calories_per_orange := 80
def cost_per_orange := 1.2
def total_money := 10
def required_calories := 400

theorem money_left (n_oranges : ℕ) (money_spent : ℝ) :
  n_oranges = required_calories / calories_per_orange ∧
  money_spent = n_oranges * cost_per_orange →
  total_money - money_spent = 4 := by
  sorry

end money_left_l187_187874


namespace number_of_correct_conclusions_l187_187130

-- Define the functions mentioned in the problem
def sqrt_squared (x : ℝ) : ℝ := real.sqrt (x^2)
def squared_sqrt (x : ℝ) : ℝ := (real.sqrt x)^2

-- Define the domain constraints from the problem's condition
def domain_f_x_minus_1 : Set ℝ := set.Icc 1 2 -- [1, 2]

-- Calculate the derived domain for f(3x^2)
def calc_domain_f_3x_squared : Set ℝ := 
  {x | -real.sqrt 3 / 3 ≤ x ∧ x ≤ real.sqrt 3 / 3}

-- The function and its increasing interval
def log_func (x : ℝ) : ℝ := real.log 2 (x^2 + 2*x - 3)
def increasing_interval : Set ℝ := set.Ioi 1 -- (1, +∞)

-- The main theorem to prove
theorem number_of_correct_conclusions :
  (sqrt_squared ≠ squared_sqrt) ∧
  (domain_f_x_minus_1  ≠ calc_domain_f_3x_squared) ∧
  (increasing_interval ≠ set.Ioi (-1)) →
  0 = 0 := 
by
  intros
  sorry

end number_of_correct_conclusions_l187_187130


namespace minimum_distance_point_l187_187543

noncomputable def point_on_curve_1 (theta : ℝ) : ℝ × ℝ :=
  (3 * Real.cos theta, Real.sin theta)

noncomputable def curve_1 (M : ℝ × ℝ) : Prop :=
  let x := M.1
  let y := M.2
  (x^2 / 9 + y^2 = 1)

noncomputable def curve_2 (N : ℝ × ℝ) : Prop :=
  let x := N.1
  let y := N.2
  (x - y - 4 = 0)

theorem minimum_distance_point :
  ∃ (M : ℝ × ℝ), curve_1 M ∧ ∀ (N : ℝ × ℝ), curve_2 N → ‖M - N‖ = min (λ z, ‖z - N‖) :=
begin
  let M := (9 * Real.sqrt 10 / 10, -Real.sqrt 10 / 10),
  use M,
  split,
  { unfold curve_1,
    sorry }, -- Proof that M is on curve_1
  { intros N HN,
    unfold curve_2 at HN,
    sorry } -- Proof that the distance is minimized
end

end minimum_distance_point_l187_187543


namespace relationship_y1_y2_y3_l187_187370

variable (y1 y2 y3 : ℝ)

def quadratic_function (x : ℝ) : ℝ := -x^2 + 4 * x - 5

theorem relationship_y1_y2_y3
  (h1 : quadratic_function (-4) = y1)
  (h2 : quadratic_function (-3) = y2)
  (h3 : quadratic_function (1) = y3) :
  y1 < y2 ∧ y2 < y3 :=
sorry

end relationship_y1_y2_y3_l187_187370


namespace evaluate_expression_l187_187140

theorem evaluate_expression : (-1:ℤ)^2022 + |(-2:ℤ)| - (1/2 : ℚ)^0 - 2 * Real.tan (Real.pi / 4) = 0 := 
by
  sorry

end evaluate_expression_l187_187140


namespace initial_investment_to_reach_100000_l187_187855

def compound_interest (P A r : ℝ) (n t : ℕ) : Prop :=
  A = P * (1 + r/n)^(n * t)

theorem initial_investment_to_reach_100000 :
  ∃ P : ℝ, P ≈ 45639 ∧ compound_interest P 100000 0.08 2 10 :=
by
  use 100000 / (1 + 0.08/2)^(2 * 10)
  split
  · norm_num
    sorry  -- Numerical verification of approximation to 45639 left as an exercise
  · unfold compound_interest
    norm_num
    sorry  -- Full calculation left as an exercise

end initial_investment_to_reach_100000_l187_187855


namespace number_of_natural_points_on_parabola_l187_187631

def y (x : ℕ) : Int := - (x^2 : Int) / 9 + 50

theorem number_of_natural_points_on_parabola :
  (∃ (x y : ℕ), y = -((x: Int)^2) / 9 + 50) -> (Finset.filter (λ x, x % 3 = 0 ∧ 1 ≤ x ∧ x ≤ 21) (Finset.range 22)).card = 7 :=
by
  sorry

end number_of_natural_points_on_parabola_l187_187631


namespace sum_of_constants_l187_187612

theorem sum_of_constants :
  (∃ x : ℝ, 
    (1 / x - 1 / (x + 2) + 1 / (x + 4) + 1 / (x + 6) - 1 / (x + 8) - 1 / (x + 12) + 1 / (x + 14)) = 0) →
  let a := 7 in
  let b := 27 in
  let c := 8 in
  let d := 2 in
  a + b + c + d = 44 :=
by
  sorry

end sum_of_constants_l187_187612


namespace slope_of_line_l187_187502

def line_equation (x y : ℝ) : Prop := x + y - 1 = 0

theorem slope_of_line :
  ∀ (x y : ℝ), line_equation x y → ∃ k b, y = k * x + b ∧ k = -1 :=
by { intros, use [-1, 1], simp [line_equation] at *, linarith, sorry }

end slope_of_line_l187_187502


namespace bisector_theorem_l187_187973

noncomputable def is_bisector (D E : Point) (A B C : Triangle) : Prop :=
  ∠(D, E, _) = ∠(D, _, E)

variables {A B C D E K : Point}
variable [TriangleABC : Triangle A B C]
variables (bisector_C : is_bisector D C A B) 
          (bisector_DE : is_bisector E D A C) 
          (bisector_DK : is_bisector K D B C)

theorem bisector_theorem :
  AD^2 + BD^2 = (AE + BK)^2 :=
by
  sorry

end bisector_theorem_l187_187973


namespace values_of_y_l187_187941

variable {x y : ℝ}

def equation1 := 2 * x ^ 2 + 6 * x + 5 * y + 1 = 0
def equation2 := 2 * x + y + 3 = 0

theorem values_of_y : equation1 → equation2 → y ^ 2 + 10 * y - 7 = 0 :=
by
  sorry

end values_of_y_l187_187941


namespace expression_equality_l187_187477

theorem expression_equality : (2 + Real.sqrt 2 + 1 / (2 + Real.sqrt 2) + 1 / (Real.sqrt 2 - 2) = 2) :=
sorry

end expression_equality_l187_187477


namespace total_number_of_people_on_bus_l187_187866

theorem total_number_of_people_on_bus (boys girls : ℕ)
    (driver assistant teacher : ℕ) 
    (h1 : boys = 50)
    (h2 : girls = boys + (2 * boys / 5))
    (h3 : driver = 1)
    (h4 : assistant = 1)
    (h5 : teacher = 1) :
    (boys + girls + driver + assistant + teacher = 123) :=
by
    sorry

end total_number_of_people_on_bus_l187_187866


namespace equal_probabilities_partitioned_nonpartitioned_conditions_for_equal_probabilities_l187_187377

variable (v1 v2 f1 f2 : ℝ)

theorem equal_probabilities_partitioned_nonpartitioned :
  (v1 * (v2 + f2) + v2 * (v1 + f1)) / (2 * (v1 + f1) * (v2 + f2)) =
  (v1 + v2) / ((v1 + f1) + (v2 + f2)) :=
by sorry

theorem conditions_for_equal_probabilities :
  (v1 * f2 = v2 * f1) ∨ (v1 + f1 = v2 + f2) :=
by sorry

end equal_probabilities_partitioned_nonpartitioned_conditions_for_equal_probabilities_l187_187377


namespace relationship_between_events_uncertain_l187_187727

theorem relationship_between_events_uncertain
  (A B : Event)
  (h : P(A ∪ B) = P(A) + P(B) ∧ P(A) + P(B) = 1) :
  (¬ MutuallyExclusive A B ∧ ¬ Complementary A B) ∨ 
  (¬ MutuallyExclusive A B ∧ Complementary A B) ∨ 
  (MutuallyExclusive A B ∧ ¬ Complementary A B) ∨ 
  (¬ Complementary A B ∧ MutuallyExclusive A B) := 
sorry

end relationship_between_events_uncertain_l187_187727


namespace S_subset_T_l187_187842

def is_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 2*k + 1

def S : Set (ℝ × ℝ) :=
  {p | let (x, y) := p in is_odd (x^2 - y^2)}

def T : Set (ℝ × ℝ) :=
  {p | let (x, y) := p in sin (2 * Real.pi * x^2) - sin (2 * Real.pi * y^2) = cos (2 * Real.pi * x^2) - cos (2 * Real.pi * y^2)}

theorem S_subset_T : S ⊆ T := sorry

end S_subset_T_l187_187842


namespace min_value_x4_y3_z2_l187_187846

theorem min_value_x4_y3_z2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : 1/x + 1/y + 1/z = 9) : 
  x^4 * y^3 * z^2 ≥ 1 / 9^9 :=
by 
  -- Proof goes here
  sorry

end min_value_x4_y3_z2_l187_187846


namespace determine_x_l187_187605

theorem determine_x (x : ℝ) (h : 9 * x^2 + 2 * x^2 + 3 * x^2 / 2 = 300) : x = 2 * Real.sqrt 6 :=
by sorry

end determine_x_l187_187605


namespace evaluate_64_pow_fifth_sixth_l187_187220

theorem evaluate_64_pow_fifth_sixth : 64 ^ (5 / 6) = 32 := by
  have h : 64 = 2 ^ 6 := by sorry
  calc 64 ^ (5 / 6) = (2 ^ 6) ^ (5 / 6) : by rw [h]
              ...   = 2 ^ (6 * (5 / 6))  : by sorry
              ...   = 2 ^ 5              : by sorry
              ...   = 32                 : by sorry

end evaluate_64_pow_fifth_sixth_l187_187220


namespace slant_asymptote_correct_sum_m_c_l187_187572

open Real

noncomputable def rational_expression (x : ℝ) : ℝ :=
  (3 * x^2 + 8 * x - 20) / (x - 5)

def slant_asymptote (x : ℝ) : ℝ :=
  3 * x + 23

theorem slant_asymptote_correct :
  ∀ x : ℝ, (rational_expression x - slant_asymptote x) → 0 as x → ∞ ∨ x → -∞ := sorry

theorem sum_m_c :
  let m := 3
  let c := 23
  m + c = 26 := by
  rw [m, c]
  rfl

end slant_asymptote_correct_sum_m_c_l187_187572


namespace exists_intersecting_line_l187_187682

/-- Represents a segment as a pair of endpoints in a 2D plane. -/
structure Segment where
  x : ℝ
  y1 : ℝ
  y2 : ℝ

open Segment

/-- Given several parallel segments with the property that for any three of these segments, 
there exists a line that intersects all three of them, prove that 
there is a line that intersects all the segments. -/
theorem exists_intersecting_line (segments : List Segment)
  (h : ∀ s1 s2 s3 : Segment, s1 ∈ segments → s2 ∈ segments → s3 ∈ segments → 
       ∃ a b : ℝ, (s1.y1 <= a * s1.x + b) ∧ (a * s1.x + b <= s1.y2) ∧ 
                   (s2.y1 <= a * s2.x + b) ∧ (a * s2.x + b <= s2.y2) ∧ 
                   (s3.y1 <= a * s3.x + b) ∧ (a * s3.x + b <= s3.y2)) :
  ∃ a b : ℝ, ∀ s : Segment, s ∈ segments → (s.y1 <= a * s.x + b) ∧ (a * s.x + b <= s.y2) := 
sorry

end exists_intersecting_line_l187_187682


namespace tan_30_deg_l187_187168

theorem tan_30_deg : 
  let θ := 30 * (Float.pi / 180) in  -- Conversion from degrees to radians
  Float.sin θ = 1 / 2 ∧ Float.cos θ = Float.sqrt 3 / 2 →
  Float.tan θ = Float.sqrt 3 / 3 := by
  intro h
  sorry

end tan_30_deg_l187_187168


namespace eval_power_l187_187211

-- Given condition
def sixty_four : ℕ := 64

-- Given condition rewritten in Lean
def sixty_four_as_two_powersix : sixty_four = 2^6 := by
  sorry

-- Prove that 64^(5/6) = 32
theorem eval_power : real.exp (5/6 * real.log 64) = 32 := by
  have h1 : 64 = 2^6 := sixty_four_as_two_powersix
  sorry

end eval_power_l187_187211


namespace sequence_sum_l187_187116

theorem sequence_sum (a b : ℕ → ℂ) (h : ∀ n, a (n + 1) = (√3 * a n - 2 * b n) ∧ b (n + 1) = (2 * √3 * b n + a n)) 
  (h50 : a 50 = 1 ∧ b 50 = 3) :
  a 1 + b 1 = (1 + 3 * complex.I) / (√7)^(49 : ℂ) * complex.exp(49 * complex.I * complex.atan (2 / √3)) :=
sorry

end sequence_sum_l187_187116


namespace tan_30_eq_sqrt3_div_3_l187_187158

/-- Statement that proves the value of tang of 30 degrees, given the cosine
    and sine values. -/
theorem tan_30_eq_sqrt3_div_3 
  (cos_30 : Real) (sin_30 : Real) 
  (hcos : cos_30 = Real.sqrt 3 / 2) 
  (hsin : sin_30 = 1 / 2) : 
    Real.tan 30 = Real.sqrt 3 / 3 := 
by 
  sorry

end tan_30_eq_sqrt3_div_3_l187_187158


namespace number_of_subsets_of_M_equals_eight_l187_187303

noncomputable def number_of_subsets (a : ℝ) : ℕ :=
  let M := {x : ℝ | (|x| * (x^2 - 3 * x - a^2 + 2)) = 0 } in
  2 ^ (Set.card M)

theorem number_of_subsets_of_M_equals_eight (a : ℝ) : number_of_subsets a = 8 :=
by
  sorry

end number_of_subsets_of_M_equals_eight_l187_187303


namespace num_distinct_three_digit_integers_l187_187363

namespace ProofProblem

def available_digits : Multiset ℕ := {2, 3, 3, 5, 5, 5, 6, 6}

def is_valid_integer (n : ℕ) : Prop :=
  ∀ d, Multiset.count d (Multiset.of_list (nat.digits 10 n)) ≤ Multiset.count d available_digits

theorem num_distinct_three_digit_integers : 
  ∃! n, (n = 52) ∧ (∑ k in {i | is_valid_integer i ∧ 100 ≤ i ∧ i ≤ 999}.to_finset, 1) = n 
:= 
  sorry

end num_distinct_three_digit_integers_l187_187363


namespace simplest_quadratic_radical_l187_187131

theorem simplest_quadratic_radical :
  (is_simplest_quadratic_radical (sqrt 7)) :=
by
  sorry

def is_simplest_quadratic_radical (r : ℝ) : Prop :=
  (∀ x : ℝ, r ≠ x ^ 2) ∧ ¬ (∃ y : ℝ, r = y / z ∧ z ≠ 0)
  sorry

end simplest_quadratic_radical_l187_187131


namespace max_q_value_l187_187804

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end max_q_value_l187_187804


namespace distinct_sequences_count_l187_187717

-- Definitions based on the conditions
def letters : Finset Char := {'E', 'Q', 'U', 'A', 'L', 'I', 'T', 'Y'}

-- Definition of a valid sequence
def valid_seq (s : List Char) : Prop :=
  s.head = 'E' ∧ s.getLast s ≠ none ∧ s.getLast s = some 'Y' ∧ s.to_finset ⊆ letters ∧ s.nodup ∧ s.length = 5

-- Statement of the problem
theorem distinct_sequences_count : 
  {s : List Char // valid_seq s}.to_finset.card = 120 := sorry

end distinct_sequences_count_l187_187717


namespace evaluate_64_pow_5_div_6_l187_187224

theorem evaluate_64_pow_5_div_6 : (64 : ℝ)^(5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ)^6 := by norm_num
  rw [← h1]
  have h2 : ((2 : ℝ)^6)^(5 / 6) = (2 : ℝ)^(6 * (5 / 6)) := by rw [Real.rpow_mul]
  rw [h2]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187224


namespace sum_of_two_numbers_l187_187870

theorem sum_of_two_numbers (L S : ℕ) (hL : L = 22) (hExceeds : L = S + 10) : L + S = 34 := by
  sorry

end sum_of_two_numbers_l187_187870


namespace max_value_of_q_l187_187817

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_q_l187_187817


namespace domain_of_f_monotonicity_of_f_l187_187664

noncomputable def f (a x : ℝ) := Real.log (a ^ x - 1) / Real.log a

theorem domain_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (a > 1 → ∀ x : ℝ, f a x ∈ Set.Ioi 0) ∧ (0 < a ∧ a < 1 → ∀ x : ℝ, f a x ∈ Set.Iio 0) :=
sorry

theorem monotonicity_of_f (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  (a > 1 → StrictMono (f a)) ∧ (0 < a ∧ a < 1 → StrictMono (f a)) :=
sorry

end domain_of_f_monotonicity_of_f_l187_187664


namespace triangle_BXN_property_l187_187819

theorem triangle_BXN_property
  (A B C M N X : Type)
  [AddGroup B] 
  [AddCommGroup B]
  [Module ℝ B]
  (AC_length : Real)
  (M_midpoint : midpt M A C)
  (CN_angle_bisector : angle_bisector C N A B)
  (X_intersection : intersection X B M C N)
  (right_triangle_BXN : right_triangle B X N)
  (angle_BXN_90 : angle B X N = π / 2)
  (AC_eq_4 : AC_length = 4) :
  sq (dist B X) = 4 := 
begin
  sorry
end

end triangle_BXN_property_l187_187819


namespace max_value_expression_l187_187798

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end max_value_expression_l187_187798


namespace pages_per_30_dollars_l187_187408

/-- Define the cost per set of pages and the flat fee -/
def cost_per_4_pages : ℕ := 7
def flat_fee : ℕ := 100

/-- Define the conversion from dollars to cents -/
def dollars_to_cents (dollars : ℕ) : ℕ := dollars * 100

/-- Define the main statement -/
theorem pages_per_30_dollars : 
  ∃ (pages : ℕ), 
  let total_cents := dollars_to_cents 30 in  -- Convert $30 to cents
  let remaining_cents := total_cents - flat_fee in  -- Subtract the flat fee
  let pages_per_cent := (4 : ℝ) / (7 : ℝ) in       -- Pages per 7 cents
  pages = ⌊remaining_cents * pages_per_cent⌋₊ ∧   -- Calculate total pages
  pages = 1657 :=                                  -- Assert the correct answer
by sorry

end pages_per_30_dollars_l187_187408


namespace number_of_even_four_digit_numbers_l187_187529

def digits := {1, 2, 3, 4}

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def four_digit_numbers (d : Set ℕ) (p : ℕ → Prop) := 
  { n | n ∈ (d.to_list.perms.map (λ l, l.head! * 1000 + l[1]! * 100 + l[2]! * 10 + l[3]!) : Set ℕ) ∧ p n }

theorem number_of_even_four_digit_numbers : 
  ∃ n, n = 12 ∧ n = (four_digit_numbers digits is_even).size := 
by
  sorry

end number_of_even_four_digit_numbers_l187_187529


namespace timmy_money_left_after_oranges_l187_187875

-- Define the conditions
def orange_calories : ℕ := 80
def orange_cost : ℝ := 1.20
def timmy_money : ℝ := 10
def required_calories : ℕ := 400

-- Define the proof problem
theorem timmy_money_left_after_oranges :
  (timmy_money - (real.of_nat (required_calories / orange_calories)) * orange_cost = 4) :=
by
  sorry

end timmy_money_left_after_oranges_l187_187875


namespace alpha_beta_sum_l187_187336

theorem alpha_beta_sum (a : ℝ) (h_a : 1 < a) (α β : ℝ)
    (h1 : α ∈ Ioo (-π / 2) (π / 2))
    (h2 : β ∈ Ioo (-π / 2) (π / 2))
    (h3 : (tan(α) + tan(β) = -3 * a))
    (h4 : (tan(α) * tan(β) = 3 * a + 1)) :
    α + β = - 3 * π / 4 :=
sorry

end alpha_beta_sum_l187_187336


namespace find_unique_five_digit_number_with_property_l187_187621

theorem find_unique_five_digit_number_with_property :
  ∃ (N : ℕ), 10000 ≤ N ∧ N < 100000 ∧
  (∀ a b c d e : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
   N = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 →
   N = 1332 * (a + b + c + d + e)) ∧
  N = 35964 :=
begin
  sorry
end

end find_unique_five_digit_number_with_property_l187_187621


namespace initial_amount_invested_l187_187659

-- Conditions
def initial_investment : ℝ := 367.36
def annual_interest_rate : ℝ := 0.08
def accumulated_amount : ℝ := 500
def years : ℕ := 4

-- Required to prove that the initial investment satisfies the given equation
theorem initial_amount_invested :
  initial_investment * (1 + annual_interest_rate) ^ years = accumulated_amount :=
by
  sorry

end initial_amount_invested_l187_187659


namespace max_value_of_q_l187_187815

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_q_l187_187815


namespace ellipse_equation_l187_187691

theorem ellipse_equation 
  (a b c : ℝ) 
  (h_positive : a > b ∧ b > 0 ∧ c > 0)
  (h_foci : c = 4)
  (h_point : (5, 0) ∈ set_of (λ p, ∃ x y, p = (x, y) ∧ (x^2 / a^2 + y^2 / b^2 = 1))) : 
  (a = 5 ∧ b^2 = a^2 - c^2) → (∀ x y : ℝ, x^2 / 25 + y^2 / 9 = 1) :=
by
  sorry

end ellipse_equation_l187_187691


namespace unique_solution_a_l187_187635

theorem unique_solution_a (a x y : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : 4 * y + a ^ 2 > 0) 
  (h4 : |x| + real.log 2 (4 * y + a ^ 2) = 3) 
  (h5 : real.log (2 * a ^ 2) (x ^ 2 + 1) + y ^ 2 = 1) : 
  a = 2 * real.sqrt 3 :=
sorry

end unique_solution_a_l187_187635


namespace problem_I_problem_II_l187_187350

variable (a : ℕ → ℤ)
variable (T : ℕ → ℤ)

-- Problem I: Prove that the sequence {a_n + 4} is a geometric sequence
theorem problem_I (h₀ : a 1 = -2) 
                (h₁ : ∀ n, a (n + 1) = 2 * a n + 4) :
  ∃ r, ∀ n, a n + 4 = (r : ℤ) ^ n :=
by
  sorry

-- Problem II: Prove that the sum T_n of the first n terms of the sequence {|a_n|} is 2^(n+1) - 4n + 2
theorem problem_II (h₀ : a 1 = -2) 
                   (h₁ : ∀ n, a (n + 1) = 2 * a n + 4) :
  T = λ n, sorry -> T n = 2 ^ (n + 1) - 4 * n + 2 :=
by
  sorry

end problem_I_problem_II_l187_187350


namespace find_f8_l187_187479

theorem find_f8 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, f(x + y) = f(x) + f(y)) (h2 : f 7 = 8) : f 8 = 64 / 7 := 
sorry

end find_f8_l187_187479


namespace fraction_c_over_d_l187_187506

-- Assume that we have a polynomial equation ax^3 + bx^2 + cx + d = 0 with roots 1, 2, 3
def polynomial (a b c d x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

-- The roots of the polynomial are 1, 2, 3
def roots (a b c d : ℝ) : Prop := polynomial a b c d 1 ∧ polynomial a b c d 2 ∧ polynomial a b c d 3

-- Vieta's formulas give us the relation for c and d in terms of the roots
theorem fraction_c_over_d (a b c d : ℝ) (h : roots a b c d) : c / d = -11 / 6 :=
sorry

end fraction_c_over_d_l187_187506


namespace general_formula_for_sequences_c_seq_is_arithmetic_fn_integer_roots_l187_187709

noncomputable def a_seq (n : ℕ) : ℕ :=
  if h : n > 0 then n else 1

noncomputable def b_seq (n : ℕ) : ℚ :=
  if h : n > 0 then n * (n - 1) / 4 else 0

noncomputable def c_seq (n : ℕ) : ℚ :=
  a_seq n ^ 2 - 4 * b_seq n

theorem general_formula_for_sequences (n : ℕ) (h : n > 0) :
  a_seq n = n ∧ b_seq n = (n * (n - 1)) / 4 :=
sorry

theorem c_seq_is_arithmetic (n : ℕ) (h : n > 0) : 
  ∀ m : ℕ, (h2 : m > 0) -> c_seq (m+1) - c_seq m = 1 :=
sorry

theorem fn_integer_roots (n : ℕ) : 
  ∃ k : ℤ, n = k ^ 2 ∧ k ≠ 0 :=
sorry

end general_formula_for_sequences_c_seq_is_arithmetic_fn_integer_roots_l187_187709


namespace evaluate_pow_l187_187255

theorem evaluate_pow : (64 : ℝ) = (8 : ℝ) ^ 2 → (8 : ℝ) = (2 : ℝ) ^ 3 → (64 : ℝ) ^ (5 / 6) = 32 :=
by
  intros h1 h2
  rw h1
  rw h2
  have h3 : (2 : ℝ)^3 ^ 2 = (2 : ℝ) ^ 6 := by ring_exp
  rw h3
  sorry

end evaluate_pow_l187_187255


namespace odd_function_value_at_neg_four_l187_187322

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^(-1/2) else - if (-x > 0) then (-x)^(-1/2) else 0

theorem odd_function_value_at_neg_four :
  f(-4) = -1/2 := by
sorry

end odd_function_value_at_neg_four_l187_187322


namespace intersection_P_Q_l187_187851

open Set

def P : Set ℝ := {1, 2}
def Q : Set ℝ := {x | abs x < 2}

theorem intersection_P_Q : P ∩ Q = {1} :=
by
  sorry

end intersection_P_Q_l187_187851


namespace no_valid_numbering_of_second_octagon_l187_187962

theorem no_valid_numbering_of_second_octagon :
  ¬ ∃ (a : Fin 8 → Fin 8), 
    ∀ (k : Fin 8), 
      ∃ (i : Fin 8), 
        (i + k) % 8 = a i := 
begin
  sorry
end

end no_valid_numbering_of_second_octagon_l187_187962


namespace jenna_driving_speed_l187_187778

open Real

/-- Jenna's driving speed calculation based on given conditions -/
def jenna_speed (driving_distance_jenna driving_distance_friend friend_speed total_time breaks time_factor : ℝ) : ℝ :=
  let total_break_time := breaks * time_factor
  let actual_driving_time := total_time - total_break_time
  let friend_driving_time := driving_distance_friend / friend_speed
  let jenna_driving_time := actual_driving_time - friend_driving_time
  driving_distance_jenna / jenna_driving_time

theorem jenna_driving_speed :
  jenna_speed 200 100 20 10 2 (30 / 60) = 50 :=
by
  -- Here we will prove that the calculated speed for Jenna is 50 mph.
  sorry

end jenna_driving_speed_l187_187778


namespace tan_30_eq_sqrt3_div_3_l187_187159

/-- Statement that proves the value of tang of 30 degrees, given the cosine
    and sine values. -/
theorem tan_30_eq_sqrt3_div_3 
  (cos_30 : Real) (sin_30 : Real) 
  (hcos : cos_30 = Real.sqrt 3 / 2) 
  (hsin : sin_30 = 1 / 2) : 
    Real.tan 30 = Real.sqrt 3 / 3 := 
by 
  sorry

end tan_30_eq_sqrt3_div_3_l187_187159


namespace current_age_l187_187397

theorem current_age (A B S Y : ℕ) 
  (h1: Y = 4) 
  (h2: S = 2 * Y) 
  (h3: B = S + 3) 
  (h4: A + 10 = 2 * (B + 10))
  (h5: A + 10 = 3 * (S + 10))
  (h6: A + 10 = 4 * (Y + 10)) 
  (h7: (A + 10) + (B + 10) + (S + 10) + (Y + 10) = 88) : 
  A = 46 :=
sorry

end current_age_l187_187397


namespace det_matrix_zero_l187_187379

theorem det_matrix_zero (x : ℝ) : 
  (det (Matrix.kronecker (λ (i j : Fin 2), (if i = 0 ∧ j = 0 then 2^(x-1) else 0) +
                                         (if i = 0 ∧ j = 1 then 4 else 0) +
                                         (if i = 1 ∧ j = 0 then 1 else 0) +
                                         (if i = 1 ∧ j = 1 then 2 else 0))) = 0) → x = 2 :=
sorry

end det_matrix_zero_l187_187379


namespace desired_cost_of_mixture_l187_187562

theorem desired_cost_of_mixture 
  (w₈ : ℝ) (c₈ : ℝ) -- weight and cost per pound of the $8 candy
  (w₅ : ℝ) (c₅ : ℝ) -- weight and cost per pound of the $5 candy
  (total_w : ℝ) (desired_cost : ℝ) -- total weight and desired cost per pound of the mixture
  (h₁ : w₈ = 30) (h₂ : c₈ = 8) 
  (h₃ : w₅ = 60) (h₄ : c₅ = 5)
  (h₅ : total_w = w₈ + w₅)
  (h₆ : desired_cost = (w₈ * c₈ + w₅ * c₅) / total_w) :
  desired_cost = 6 := 
by
  sorry

end desired_cost_of_mixture_l187_187562


namespace primes_at_ends_of_sequence_l187_187844

theorem primes_at_ends_of_sequence {k : ℕ} 
  (h : ∑ i in finset.range 29, nat.prime (30 * k + i + 1) = 7) :
  nat.prime (30 * k + 1) ∧ nat.prime (30 * k + 29) :=
by
  sorry

end primes_at_ends_of_sequence_l187_187844


namespace total_cost_correct_l187_187579

variable (O B : ℕ) (cost_orchestra cost_balcony : ℕ)
variable (total_tickets : ℕ) (balcony_more : ℕ)

-- Conditions
def condition1 : Prop := O + B = total_tickets
def condition2 : Prop := B = O + balcony_more
def cost_orchestra := 12
def cost_balcony := 8
def total_tickets := 355
def balcony_more := 115

-- Target Proof
theorem total_cost_correct (O B : ℕ) 
  (h1 : condition1 O B) (h2 : condition2 O B) :
  O * cost_orchestra + B * cost_balcony = 3320 := 
by
  sorry

end total_cost_correct_l187_187579


namespace teacher_arrangement_l187_187517

theorem teacher_arrangement : 
  let total_teachers := 6
  let max_teachers_per_class := 4
  (∑ i in (finset.range max_teachers_per_class).filter (λ x, x ≤ total_teachers - max_teachers_per_class + 1), 
    nat.choose total_teachers (total_teachers - i) * (if i = total_teachers - i then 1 else 2)) = 31 :=
by sorry

end teacher_arrangement_l187_187517


namespace frosting_cans_needed_l187_187618

theorem frosting_cans_needed :
  let daily_cakes := 10
  let days := 5
  let total_cakes := daily_cakes * days
  let eaten_cakes := 12
  let remaining_cakes := total_cakes - eaten_cakes
  let cans_per_cake := 2
  let total_cans := remaining_cakes * cans_per_cake
  total_cans = 76 := 
by
  sorry

end frosting_cans_needed_l187_187618


namespace jesse_fraction_of_book_read_l187_187779

theorem jesse_fraction_of_book_read :
  let pages_read := 10 + 15 + 27 + 12 + 19
  let pages_left := 166
  let total_pages := pages_read + pages_left
  (pages_read / total_pages : ℚ) = 1 / 3 :=
by
  let pages_read := 10 + 15 + 27 + 12 + 19
  let pages_left := 166
  let total_pages := pages_read + pages_left
  have : (pages_read / total_pages : ℚ) = 83 / 249 := by sorry
  have : 83 / 249 = 1 / 3 := by sorry
  exact Eq.trans this this

end jesse_fraction_of_book_read_l187_187779


namespace smallest_n_for_perfect_square_and_cube_l187_187059

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, (∃ a : ℕ, 4 * n = a^2) ∧ (∃ b : ℕ, 5 * n = b^3) ∧ n = 125 :=
by
  sorry

end smallest_n_for_perfect_square_and_cube_l187_187059


namespace sum_of_sequence_l187_187675

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = (-1)^n * (a n + 1)

def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  Finset.sum (Finset.range n) a

theorem sum_of_sequence {a : ℕ → ℤ} (h : sequence a) :
  S a 2013 = -1005 :=
sorry

end sum_of_sequence_l187_187675


namespace evaluate_root_l187_187236

theorem evaluate_root : 64 ^ (5 / 6 : ℝ) = 32 :=
by sorry

end evaluate_root_l187_187236


namespace jordan_oreos_l187_187773

theorem jordan_oreos 
  (x : ℕ) 
  (h1 : let james := 3 + 2 * x in james + x = 36) : 
  x = 11 :=
by 
  -- Proof will go here
  sorry

end jordan_oreos_l187_187773


namespace ball_distribution_l187_187518

theorem ball_distribution (n k : ℕ) (h₁ : 1 ≤ k) (h₂ : k ≤ n - 1) :
  (binomial n k) * (binomial (n - 1) k) = number_of_ways_to_distribute_balls :=
by
  sorry

end ball_distribution_l187_187518


namespace eval_64_pow_5_over_6_l187_187193

theorem eval_64_pow_5_over_6 (h : 64 = 2^6) : 64^(5/6) = 32 := 
by 
  sorry

end eval_64_pow_5_over_6_l187_187193


namespace log_expression_eval_l187_187685

theorem log_expression_eval (x : ℝ) (hx : x < 1)
  (h : (Real.log10 x)^2 - Real.log10 (x^2) = 48) :
  (Real.log10 x)^3 - Real.log10 (x^3) = -198 := by
  sorry

end log_expression_eval_l187_187685


namespace gcd_max_value_l187_187015

theorem gcd_max_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1008) : 
  ∃ d, d = Nat.gcd a b ∧ d = 504 :=
by
  sorry

end gcd_max_value_l187_187015


namespace smallest_n_condition_l187_187067

theorem smallest_n_condition (n : ℕ) : (4 * n) ∣ (n^2) ∧ (5 * n) ∣ (u^3) → n = 100 :=
by
  sorry

end smallest_n_condition_l187_187067


namespace order_abc_l187_187316

noncomputable def a : ℝ := Real.log 0.8 / Real.log 0.7
noncomputable def b : ℝ := Real.log 0.9 / Real.log 1.1
noncomputable def c : ℝ := Real.exp (0.9 * Real.log 1.1)

theorem order_abc : b < a ∧ a < c := by
  sorry

end order_abc_l187_187316


namespace radius_of_circle_is_sqrt2_l187_187606

-- Define the conditions
def spherical_coord (ρ θ φ : ℝ) := (ρ, θ, φ)

-- Define the circle radius function given the spherical coordinates
noncomputable def circle_radius (ρ φ : ℝ) : ℝ :=
  sqrt (ρ^2 * (sin φ)^2)

-- The problem statement, asserting the radius for the given conditions
theorem radius_of_circle_is_sqrt2 (θ : ℝ) :
  circle_radius 2 (π / 4) = sqrt 2 :=
by sorry

end radius_of_circle_is_sqrt2_l187_187606


namespace lines_parallel_iff_a_eq_one_l187_187317

theorem lines_parallel_iff_a_eq_one (a : ℝ) : 
  (a x - y + 1 = 0 ∧ x - a y - 1 = 0 → parallel (a x - y + 1 = 0) (x - a y - 1 = 0))
  ↔ a = 1 := 
by 
  sorry

end lines_parallel_iff_a_eq_one_l187_187317


namespace max_value_expression_l187_187796

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end max_value_expression_l187_187796


namespace zero_in_interval_l187_187528

noncomputable def f (x : ℝ) : ℝ := x - 2 * log (1 / sqrt x) - 3

theorem zero_in_interval : f 2 < 0 ∧ f 3 > 0 :=
by
  have h1 : f 2 = log 2 - 1, sorry
  have h2 : f 3 = (3 / 2) * log 3, sorry
  split
  { show f 2 < 0, sorry }
  { show f 3 > 0, sorry }

end zero_in_interval_l187_187528


namespace visits_correct_l187_187857

-- Define the fact that February 1st is a Tuesday in a non-leap year
axiom feb_1_is_tuesday : ∀ (year : ℕ), ¬ (year % 4 = 0 ∧ (year % 100 ≠ 0 ∨ year % 400 = 0)) → true

-- Define visits
def city := Type
def Smolensk : city := sorry
def Vologda : city := sorry
def Pskov : city := sorry
def Vladimir : city := sorry

def visit (c : city) (d : ℕ) (m : ℕ) (y : ℕ) : Prop := sorry

-- Using non-leap year 2021 as an example
axiom year_2021 : ℕ := 2021

-- Given conditions in the non-leap year
axioms 
  (H1 : visit Smolensk 1 2 year_2021)
  (H2 : visit Vologda 8 2 year_2021)
  (H3: visit Pskov 1 3 year_2021)
  (H4 : visit Vladimir 8 3 year_2021)

-- The main theorem statement
theorem visits_correct : 
  ∀ year : ℕ, ¬ (year % 4 = 0 ∧ (year % 100 ≠ 0 ∨ year % 400 = 0)) →
    visit Smolensk 1 2 year ∧
    visit Vologda 8 2 year ∧
    visit Pskov 1 3 year ∧
    visit Vladimir 8 3 year :=
begin
  intro year,
  intro non_leap_year,
  exact ⟨
    H1,
    H2,
    H3,
    H4
  ⟩,
end

end visits_correct_l187_187857


namespace plane_equation_of_projection_l187_187829

-- Definitions of vectors and the equation of the plane
def w : ℝ × ℝ × ℝ := (3, -2, 1)

def proj_w_v (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let num := v.1 * w.1 + v.2 * w.2 + v.3 * w.3 in
  (num / (w.1 * w.1 + w.2 * w.2 + w.3 * w.3) * w.1,
   num / (w.1 * w.1 + w.2 * w.2 + w.3 * w.3) * w.2,
   num / (w.1 * w.1 + w.2 * w.2 + w.3 * w.3) * w.3)

theorem plane_equation_of_projection (v : ℝ × ℝ × ℝ) :
  proj_w_v v = w → 3 * v.1 - 2 * v.2 + v.3 - 14 = 0 :=
by
  intro h
  sorry

end plane_equation_of_projection_l187_187829


namespace woman_waits_time_until_man_catches_up_l187_187976

theorem woman_waits_time_until_man_catches_up
  (woman_speed : ℝ)
  (man_speed : ℝ)
  (wait_time : ℝ)
  (woman_slows_after : ℝ)
  (h_man_speed : man_speed = 5 / 60) -- man's speed in miles per minute
  (h_woman_speed : woman_speed = 25 / 60) -- woman's speed in miles per minute
  (h_wait_time : woman_slows_after = 5) -- the time in minutes after which the woman waits for man
  (h_woman_waits : wait_time = 25) : wait_time = (woman_slows_after * woman_speed) / man_speed :=
sorry

end woman_waits_time_until_man_catches_up_l187_187976


namespace no_play_days_in_june_l187_187961

theorem no_play_days_in_june :
  (let d := 396 / 18 in 30 - d = 8) :=
by
  let d := 396 / 18
  have h1: 18 * d = 396 := sorry
  have h2: d = 22 := by sorry
  show 30 - d = 8, from sorry

end no_play_days_in_june_l187_187961


namespace tank_depth_is_six_l187_187578

-- Definitions derived from the conditions
def tank_length : ℝ := 25
def tank_width : ℝ := 12
def plastering_cost_per_sq_meter : ℝ := 0.45
def total_cost : ℝ := 334.8

-- Compute the surface area to be plastered
def surface_area (d : ℝ) : ℝ := (tank_length * tank_width) + 2 * (tank_length * d) + 2 * (tank_width * d)

-- Equation relating the plastering cost to the surface area
def cost_equation (d : ℝ) : ℝ := plastering_cost_per_sq_meter * (surface_area d)

-- The mathematical result we need to prove
theorem tank_depth_is_six : ∃ d : ℝ, cost_equation d = total_cost ∧ d = 6 := by
  sorry

end tank_depth_is_six_l187_187578


namespace master_bedroom_suite_size_correct_l187_187104

noncomputable def living_dining_kitchen_area : ℝ := 1200
noncomputable def total_area : ℝ := 3500
noncomputable def library_area : ℝ := living_dining_kitchen_area - 300

def master_bedroom_suite_size (M : ℝ) : Prop :=
  let guest_bedroom := (1 / 3) * M
  let office := (1 / 6) * M
  living_dining_kitchen_area + guest_bedroom + M + office + library_area = total_area

theorem master_bedroom_suite_size_correct :
  ∃ M : ℝ, master_bedroom_suite_size M ∧ M = 933.33 :=
begin
  use 933.33,
  unfold master_bedroom_suite_size,
  sorry,
end

end master_bedroom_suite_size_correct_l187_187104


namespace largest_gcd_l187_187019

theorem largest_gcd (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 1008) : 
  ∃ d : ℕ, d = Int.gcd a b ∧ d = 504 :=
by
  sorry

end largest_gcd_l187_187019


namespace center_of_grid_is_five_l187_187586

theorem center_of_grid_is_five :
  -- Define the grid as a 3x3 matrix of integers
  ∃ (grid : (Fin 3) × (Fin 3) → ℕ),
  -- Condition 1: grid contains numbers 2 through 10
  (∀ n, n ∈ (grid '' {p : (Fin 3) × (Fin 3) | true}) ↔ n ∈ ({2, 3, 4, 5, 6, 7, 8, 9, 10} : Set ℕ)) ∧ 
  -- Condition 2: consecutive numbers share an edge
  (∀ n m, abs (n - m) = 1 → ∃ (p q : (Fin 3) × (Fin 3)), grid p = n ∧ grid q = m ∧ (p.1 = q.1 ∧ abs (p.2 - q.2) = 1 ∨ p.2 = q.2 ∧ abs (p.1 - q.1) = 1)) ∧
  -- Condition 3: the numbers in the four corners sum up to 26
  (grid (Fin.mk 0 _, Fin.mk 0 _) + grid (Fin.mk 0 _, Fin.mk 2 _) + grid (Fin.mk 2 _, Fin.mk 0 _) + grid (Fin.mk 2 _, Fin.mk 2 _) = 26) ∧ 
  -- Prove: the number in the center of the grid is 5
  grid (Fin.mk 1 _, Fin.mk 1 _) = 5 :=
sorry

end center_of_grid_is_five_l187_187586


namespace number_of_integer_values_of_a_l187_187650

theorem number_of_integer_values_of_a (a : ℤ) : 
  (∃ x : ℤ, x^2 + a * x + 9 * a = 0) ↔ 
  (∃ (a_values : Finset ℤ), a_values.card = 6 ∧ ∀ a ∈ a_values, ∃ x : ℤ, x^2 + a * x + 9 * a = 0) :=
by
  sorry

end number_of_integer_values_of_a_l187_187650


namespace range_of_a_l187_187739

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, ∃ y ∈ Set.Ici a, y = (x^2 + 2*x + a) / (x + 1)) ↔ a ≤ 2 :=
by
  sorry

end range_of_a_l187_187739


namespace same_cost_duration_l187_187544

-- Define the cost function for Plan A
def cost_plan_a (x : ℕ) : ℚ :=
 if x ≤ 8 then 0.60 else 0.60 + 0.06 * (x - 8)

-- Define the cost function for Plan B
def cost_plan_b (x : ℕ) : ℚ :=
 0.08 * x

-- The duration of a call for which the company charges the same under Plan A and Plan B is 14 minutes
theorem same_cost_duration (x : ℕ) : cost_plan_a x = cost_plan_b x ↔ x = 14 :=
by
  -- The proof is not required, using sorry to skip the proof steps
  sorry

end same_cost_duration_l187_187544


namespace tan_30_eq_sqrt3_div3_l187_187153

theorem tan_30_eq_sqrt3_div3 (sin_30_cos_30 : ℝ → ℝ → Prop)
  (h1 : sin_30_cos_30 (1 / 2) (Real.sqrt 3 / 2)) :
  ∃ t, t = Real.tan (Real.pi / 6) ∧ t = Real.sqrt 3 / 3 :=
by
  existsi Real.tan (Real.pi / 6)
  sorry

end tan_30_eq_sqrt3_div3_l187_187153


namespace exists_positive_integers_for_equation_l187_187885

theorem exists_positive_integers_for_equation (k n : ℕ) (hk : 0 < k) (hn : 0 < n) :
  ∃ (m : Fin k → ℕ), (∀ i, 0 < m i) ∧ 
  (1 + (2 ^ k - 1) / n : ℚ) = (∏ i : Fin k, (1 + 1 / (m i) : ℚ)) :=
by
  sorry

end exists_positive_integers_for_equation_l187_187885


namespace positive_root_exists_l187_187290

theorem positive_root_exists : ∃ x : ℝ, x > 0 ∧ (x^3 - 3 * x^2 - 4 * x - 2 * Real.sqrt 2 = 0) :=
  let x := 2 - Real.sqrt 2 in
  ⟨x, by sorry, by sorry⟩

end positive_root_exists_l187_187290


namespace top_square_after_five_folds_l187_187891

def initial_grid : List (List ℕ) :=
  [[1, 2, 3, 4, 5],
   [6, 7, 8, 9, 10],
   [11, 12, 13, 14, 15],
   [16, 17, 18, 19, 20],
   [21, 22, 23, 24, 25]]

def fold1 (grid : List (List ℕ)) : List (List ℕ) :=
  [[21, 22, 23, 24, 25],
   [16, 17, 18, 19, 20],
   [11, 12, 13, 14, 15],
   [6, 7, 8, 9, 10],
   [1, 2, 3, 4, 5]]

def fold2 (grid : List (List ℕ)) : List (List ℕ) :=
  [[1, 2, 3, 4, 5],
   [6, 7, 8, 9, 10],
   [11, 12, 13, 14, 15],
   [16, 17, 18, 19, 20],
   [21, 22, 23, 24, 25]]

def fold3 (grid : List (List ℕ)) : List (List ℕ) :=
  [[4, 5, 3, 2, 1],
   [9, 10, 8, 7, 6],
   [14, 15, 13, 12, 11],
   [19, 20, 18, 17, 16],
   [24, 25, 23, 22, 21]]

def fold4 (grid : List (List ℕ)) : List (List ℕ) :=
  [[3, 2, 1, 4, 5],
   [8, 7, 6, 9, 10],
   [13, 12, 11, 14, 15],
   [18, 17, 16, 19, 20],
   [23, 22, 21, 24, 25]]

def fold5 (grid : List (List ℕ)) : List (List ℕ) :=
  [[5, 4, 3, 2, 1],
   [10, 9, 8, 7, 6],
   [15, 14, 13, 12, 11],
   [20, 19, 18, 17, 16],
   [25, 24, 23, 22, 21]]

def final_grid : List (List ℕ) :=
  fold5 (fold4 (fold3 (fold2 (fold1 initial_grid))))

theorem top_square_after_five_folds : final_grid.head.head = 13 :=
by simp [final_grid, fold1, fold2, fold3, fold4, fold5]
  ; sorry

end top_square_after_five_folds_l187_187891


namespace number_of_ways_to_divide_friends_l187_187369

theorem number_of_ways_to_divide_friends :
  let friends := 8
  let teams := 4
  (teams ^ friends) = 65536 := by
  sorry

end number_of_ways_to_divide_friends_l187_187369


namespace least_plates_to_ensure_match_l187_187943

-- Define the condition as a structure for clarity
structure PlateCabinet where
  white : Nat
  green : Nat
  red : Nat
  pink : Nat
  purple : Nat

-- There are 2 white, 6 green, some red, 4 pink, and 10 purple plates
def exampleCabinet : PlateCabinet :=
  { white := 2,
    green := 6,
    red := 0,  -- "some" can be interpreted as arbitrary non-negative number, here simplified
    pink := 4,
    purple := 10 }

-- The main theorem stating you need to pull 6 plates to ensure a matching pair
theorem least_plates_to_ensure_match (cabinet : PlateCabinet) : ∃ n, (n ≥ 6) ∧ (∀ n' < n, ∃ p1 p2, p1 ≠ p2 ∧ (p1.1 = p1.2) ≠ (p2.1 = p2.2)) :=
  sorry

end least_plates_to_ensure_match_l187_187943


namespace coins_probability_l187_187094

theorem coins_probability :
  let pennies := 3
  let nickels := 5
  let dimes := 7
  let quarters := 4
  let total_coins := pennies + nickels + dimes + quarters
  let total_ways := Nat.choose total_coins 8
  let successful_ways := 2345
  let probability := (successful_ways : ℚ) / total_ways
  probability = 2345 / 75582 :=
by {
  let pennies := 3
  let nickels := 5
  let dimes := 7
  let quarters := 4
  let total_coins := pennies + nickels + dimes + quarters
  let total_ways := Nat.choose total_coins 8
  let successful_ways := 2345
  let probability := (successful_ways : ℚ) / total_ways
  show probability = 2345 / 75582, from sorry
}

end coins_probability_l187_187094


namespace min_distance_between_circle_and_parabola_l187_187839

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 1
def parabola_eq (x y : ℝ) : Prop := y^2 = 4 * x

theorem min_distance_between_circle_and_parabola :
  ∃ (P Q : ℝ × ℝ), 
    (circle_eq P.1 P.2) ∧ (parabola_eq Q.1 Q.2) ∧ (dist P Q = 2 * real.sqrt 3 - 1) :=
sorry

end min_distance_between_circle_and_parabola_l187_187839


namespace max_value_amc_am_mc_ca_l187_187789

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end max_value_amc_am_mc_ca_l187_187789


namespace count_possible_sets_l187_187683

theorem count_possible_sets (A : Set ℤ) (h : A ∪ {-1, 1} = {-1, 0, 1}) : 
    (∃ s, s = ({∅, {0}, {0, 1}, {0, -1}, {0, 1, -1}} : Set (Set ℤ)) ∧ A ∈ s) → 
    (({ ∅, {0}, {0, 1}, {0, -1}, {0, 1, -1} } : Set (Set ℤ)).card = 4) :=
by
    sorry

end count_possible_sets_l187_187683


namespace tan_30_eq_sqrt3_div_3_l187_187157

/-- Statement that proves the value of tang of 30 degrees, given the cosine
    and sine values. -/
theorem tan_30_eq_sqrt3_div_3 
  (cos_30 : Real) (sin_30 : Real) 
  (hcos : cos_30 = Real.sqrt 3 / 2) 
  (hsin : sin_30 = 1 / 2) : 
    Real.tan 30 = Real.sqrt 3 / 3 := 
by 
  sorry

end tan_30_eq_sqrt3_div_3_l187_187157


namespace smallest_yellow_marbles_l187_187585

-- Define the number of blue marbles, red marbles, green marbles, and the equation for yellow marbles
def blue_marbles (n : ℕ) : ℕ := (2 * n) / 5
def red_marbles (n : ℕ) : ℕ := (n) / 3
def green_marbles : ℕ := 4
def yellow_marbles (n : ℕ) : ℕ := n - (blue_marbles n + red_marbles n + green_marbles)

-- Main theorem: The smallest number of yellow marbles is 0 when the total number of marbles is 15
theorem smallest_yellow_marbles : ∃ (n : ℕ), n % 15 = 0 ∧ yellow_marbles n = 0 :=
by {
  use 15,
  -- Check that 15 is a multiple of 15
  simp,
  -- Check the number of yellow marbles
  have h_blue : blue_marbles 15 = 6, by simp [blue_marbles],
  have h_red : red_marbles 15 = 5, by simp [red_marbles],
  have h_green : green_marbles = 4, by simp [green_marbles],
  have h_yellow := by {
    have := yellow_marbles,
    rw [h_blue, h_red, h_green],
    norm_num,
  },
  exact h_yellow,
}

end smallest_yellow_marbles_l187_187585


namespace repeating_decimal_45_eq_5_div_11_l187_187764

theorem repeating_decimal_45_eq_5_div_11 :
  (0 • 1 / 10 + 4 • 1 / 10 + 5 / 100 + 5 / 1000 + 5 / 10000 + ...) = 5 / 11 :=
sorry

end repeating_decimal_45_eq_5_div_11_l187_187764


namespace evaluate_64_pow_5_div_6_l187_187265

theorem evaluate_64_pow_5_div_6 : (64 : ℝ) ^ (5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ) ^ 6 := by norm_num
  have h2 : (64 : ℝ) ^ (5 / 6) = ((2 : ℝ) ^ 6) ^ (5 / 6) := by rw h1
  have h3 : ((2 : ℝ) ^ 6) ^ (5 / 6) = (2 : ℝ) ^ (6 * (5 / 6)) := by rw [Real.rpow_mul]
  have h4 : (2 : ℝ) ^ (6 * (5 / 6)) = (2 : ℝ) ^ 5 := by norm_num
  rw [h2, h3, h4]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187265


namespace find_original_number_l187_187569

theorem find_original_number (x : ℤ) : 4 * (3 * x + 29) = 212 → x = 8 :=
by
  intro h
  sorry

end find_original_number_l187_187569


namespace vectors_not_coplanar_l187_187591

def vector3 := (ℝ × ℝ × ℝ)

-- Define the vectors
def a : vector3 := (3, 3, 1)
def b : vector3 := (1, -2, 1)
def c : vector3 := (1, 1, 1)

-- Function to compute the determinant of a 3x3 matrix formed by three vectors
def scalarTripleProduct (u v w : vector3) : ℝ :=
  let (ux, uy, uz) := u
  let (vx, vy, vz) := v
  let (wx, wy, wz) := w
  ux * (vy * wz - vz * wy) - uy * (vx * wz - vz * wx) + uz * (vx * wy - vy * wx)

-- Statement to prove the non-coplanarity of the vectors
theorem vectors_not_coplanar :
  scalarTripleProduct a b c ≠ 0 :=
by {
  -- calculations of the determinant
  let det_abc := scalarTripleProduct a b c,
  show det_abc ≠ 0, from sorry
}

end vectors_not_coplanar_l187_187591


namespace product_ends_in_36_l187_187041

theorem product_ends_in_36 (a b : ℕ) (ha : a < 10) (hb : b < 10) :
  ((10 * a + 6) * (10 * b + 6)) % 100 = 36 ↔ (a + b = 0 ∨ a + b = 5 ∨ a + b = 10 ∨ a + b = 15) :=
by
  sorry

end product_ends_in_36_l187_187041


namespace angle_between_vectors_is_pi_over_4_l187_187358

variables (a b : ℝ^3)

def magnitude (v : ℝ^3) : ℝ := real.sqrt (v.dot_product v)

noncomputable def angle_between (u v : ℝ^3) : ℝ :=
real.acos ((u.dot_product v) / (magnitude u * magnitude v))

theorem angle_between_vectors_is_pi_over_4
  (ha : magnitude a = real.sqrt 2)
  (hb : magnitude b = 1)
  (h_ab : magnitude (a - (2 : ℝ) • b) = real.sqrt 2) :
  angle_between a b = π / 4 :=
sorry

end angle_between_vectors_is_pi_over_4_l187_187358


namespace sum_of_divisors_inequality_l187_187786

def d (n : ℕ) : ℕ :=
  ∑ i in (finset.range n).filter (λ d, d ∣ n), 1

theorem sum_of_divisors_inequality (n : ℕ) (h : n > 1) :
  (∑ i in (finset.range n).filter (λ i, 2 ≤ i), (1 : ℝ) / i) ≤
  (∑ i in (finset.range n).filter (λ i, 1 ≤ i), (d i : ℝ) / n) ∧
  (∑ i in (finset.range n).filter (λ i, 1 ≤ i), (d i : ℝ) / n) ≤
  (∑ i in (finset.range n).filter (λ i, 1 ≤ i), (1 : ℝ) / i) :=
  sorry

end sum_of_divisors_inequality_l187_187786


namespace specialPermutationCount_l187_187719

def countSpecialPerms (n : ℕ) : ℕ := 2 ^ (n - 1)

theorem specialPermutationCount (n : ℕ) : 
  (countSpecialPerms n = 2 ^ (n - 1)) := 
by 
  sorry

end specialPermutationCount_l187_187719


namespace find_x_for_divisibility_l187_187291

theorem find_x_for_divisibility (x : ℕ) : (x24x : ℕ) % 30 = 0 → x = 0 := by
  -- define the four-digit number x24x
  let n := 1000 * x + 240 + x
  have : n % 30 = 0 → (x % 2 = 0 ∧ (2 * x + 6) % 3 = 0 ∧ (x = 0 ∨ x = 5)), from sorry
  intro h
  exact sorry

end find_x_for_divisibility_l187_187291


namespace gcd_largest_divisor_l187_187023

theorem gcd_largest_divisor (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 1008) : 
  ∃ d, nat.gcd a b = d ∧ d = 504 :=
begin
  sorry
end

end gcd_largest_divisor_l187_187023


namespace integer_values_of_a_count_integer_values_of_a_l187_187654

theorem integer_values_of_a (a : ℤ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ (x1 * x1 + a * x1 + 9 * a = 0) ∧ (x2 * x2 + a * x2 + 9 * a = 0)) →
  a ∈ {0, -12, -64} :=
by
  sorry

theorem count_integer_values_of_a : 
  {a : ℤ | ∃ x1 x2 : ℤ, x1 ≠ x2 ∧ (x1 * x1 + a * x1 + 9 * a = 0) ∧ (x2 * x2 + a * x2 + 9 * a = 0)}.to_finset.card = 3 :=
by
  sorry

end integer_values_of_a_count_integer_values_of_a_l187_187654


namespace eval_64_pow_5_over_6_l187_187199

theorem eval_64_pow_5_over_6 (h : 64 = 2^6) : 64^(5/6) = 32 := 
by 
  sorry

end eval_64_pow_5_over_6_l187_187199


namespace highest_score_batsman_l187_187981

variable (H L : ℕ)

theorem highest_score_batsman :
  (60 * 46) = (58 * 44 + H + L) ∧ (H - L = 190) → H = 199 :=
by
  intros h
  sorry

end highest_score_batsman_l187_187981


namespace same_solution_k_value_l187_187658

theorem same_solution_k_value 
  (x : ℝ)
  (k : ℝ)
  (m : ℝ)
  (h₁ : 2 * x + 4 = 4 * (x - 2))
  (h₂ : k * x + m = 2 * x - 1) 
  (h₃ : k = 17) : 
  k = 17 ∧ m = -91 :=
by
  sorry

end same_solution_k_value_l187_187658


namespace time_for_dry_cleaning_l187_187410

def total_time_minutes : ℕ := 180
def commute_time_minutes : ℕ := 30
def grocery_time_minutes : ℕ := 30
def dog_grooming_time_minutes : ℕ := 20
def cooking_dinner_time_minutes : ℕ := 90

theorem time_for_dry_cleaning : ∀ (x : ℕ),
  total_time_minutes - commute_time_minutes - grocery_time_minutes
  - dog_grooming_time_minutes - cooking_dinner_time_minutes = x →
  x = 10 :=
by
  intros x h
  lwho rwa [h, Nat.sub_sub_assoc, Nat.sub_sub_assoc, Nat.sub_sub_assoc]
  sorry

end time_for_dry_cleaning_l187_187410


namespace evaluate_pow_l187_187254

theorem evaluate_pow : (64 : ℝ) = (8 : ℝ) ^ 2 → (8 : ℝ) = (2 : ℝ) ^ 3 → (64 : ℝ) ^ (5 / 6) = 32 :=
by
  intros h1 h2
  rw h1
  rw h2
  have h3 : (2 : ℝ)^3 ^ 2 = (2 : ℝ) ^ 6 := by ring_exp
  rw h3
  sorry

end evaluate_pow_l187_187254


namespace smallest_n_l187_187061

theorem smallest_n (n : ℕ) : (∃ (m1 m2 : ℕ), 4 * n = m1^2 ∧ 5 * n = m2^3) ↔ n = 500 := 
begin
  sorry
end

end smallest_n_l187_187061


namespace point_in_triangle_l187_187400

variables {m a b n : ℝ}
variables (c1 c2 c3 z : ℂ) (S1 S2 S3 : ℂ)

-- Given the complex points and their constraints
def c1 : ℂ := m + b * complex.I
def c2 : ℂ := a + b * complex.I
def c3 : ℂ := a + n * complex.I

-- The condition a > m and n > b indicating the points form a triangle
variables (h1 : a > m) (h2 : n > b)

-- The equation to solve
def equation (z : ℂ) : Prop :=
  1 / (z - c1) + 1 / (z - c2) + 1 / (z - c3) = 0

-- The statement theorem, z corresponds to a point Z located inside the triangle formed by points corresponding to c1, c2, c3
theorem point_in_triangle 
  (hz : equation z)
  (h1 : a > m)
  (h2 : n > b) :
  ∃ (Z : ℂ), Z = z ∧ Z ∈ triangle c1 c2 c3 :=
sorry

end point_in_triangle_l187_187400


namespace evaluate_64_pow_5_div_6_l187_187267

theorem evaluate_64_pow_5_div_6 : (64 : ℝ) ^ (5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ) ^ 6 := by norm_num
  have h2 : (64 : ℝ) ^ (5 / 6) = ((2 : ℝ) ^ 6) ^ (5 / 6) := by rw h1
  have h3 : ((2 : ℝ) ^ 6) ^ (5 / 6) = (2 : ℝ) ^ (6 * (5 / 6)) := by rw [Real.rpow_mul]
  have h4 : (2 : ℝ) ^ (6 * (5 / 6)) = (2 : ℝ) ^ 5 := by norm_num
  rw [h2, h3, h4]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187267


namespace intersection_in_fourth_quadrant_l187_187924

theorem intersection_in_fourth_quadrant :
  (∃ x y : ℝ, y = -x ∧ y = 2 * x - 1 ∧ x = 1 ∧ y = -1) ∧ (1 > 0 ∧ -1 < 0) :=
by
  sorry

end intersection_in_fourth_quadrant_l187_187924


namespace correct_option_C_l187_187700

def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem correct_option_C : ∀ {x1 x2 : ℝ}, 0 < x1 → x1 < x2 → x1 * f x1 < x2 * f x2 :=
by
  sorry

end correct_option_C_l187_187700


namespace negation_of_p_l187_187348

variable (p : ∀ x : ℝ, x^2 + x - 6 ≤ 0)

theorem negation_of_p : (∃ x : ℝ, x^2 + x - 6 > 0) :=
sorry

end negation_of_p_l187_187348


namespace total_number_of_people_on_bus_l187_187865

theorem total_number_of_people_on_bus (boys girls : ℕ)
    (driver assistant teacher : ℕ) 
    (h1 : boys = 50)
    (h2 : girls = boys + (2 * boys / 5))
    (h3 : driver = 1)
    (h4 : assistant = 1)
    (h5 : teacher = 1) :
    (boys + girls + driver + assistant + teacher = 123) :=
by
    sorry

end total_number_of_people_on_bus_l187_187865


namespace installation_cost_l187_187458

-- Definitions
variables (LP : ℝ) (P : ℝ := 16500) (D : ℝ := 0.2) (T : ℝ := 125) (SP : ℝ := 23100) (I : ℝ)

-- Conditions
def purchase_price := P = (1 - D) * LP
def selling_price := SP = 1.1 * LP
def total_cost := P + T + I = SP

-- Proof Statement
theorem installation_cost : I = 6350 :=
  by
    -- sorry is used to skip the proof
    sorry

end installation_cost_l187_187458


namespace evaluate_pow_l187_187259

theorem evaluate_pow : (64 : ℝ) = (8 : ℝ) ^ 2 → (8 : ℝ) = (2 : ℝ) ^ 3 → (64 : ℝ) ^ (5 / 6) = 32 :=
by
  intros h1 h2
  rw h1
  rw h2
  have h3 : (2 : ℝ)^3 ^ 2 = (2 : ℝ) ^ 6 := by ring_exp
  rw h3
  sorry

end evaluate_pow_l187_187259


namespace tan_30_l187_187147

theorem tan_30 : Real.tan (Real.pi / 6) = Real.sqrt 3 / 3 := 
by 
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry
  have h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2 := by sorry
  calc
    Real.tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6) : Real.tan_eq_sin_div_cos _
    ... = (1 / 2) / (Real.sqrt 3 / 2) : by rw [h1, h2]
    ... = (1 / 2) * (2 / Real.sqrt 3) : by rw Div.div_eq_mul_inv
    ... = 1 / Real.sqrt 3 : by norm_num
    ... = Real.sqrt 3 / 3 : by rw [Div.inv_eq_inv, Mul.comm, Mul.assoc, Div.mul_inv_cancel (Real.sqrt_ne_zero _), one_div Real.sqrt 3, inv_mul_eq_div]

-- Additional necessary function apologies for the unproven theorems.
noncomputable def _root_.Real.sqrt (x:ℝ) : ℝ := sorry

noncomputable def _root_.Real.tan (x : ℝ) : ℝ :=
  (Real.sin x) / (Real.cos x)

#eval tan_30 -- check result

end tan_30_l187_187147


namespace max_q_value_l187_187800

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end max_q_value_l187_187800


namespace evaluate_root_l187_187240

theorem evaluate_root : 64 ^ (5 / 6 : ℝ) = 32 :=
by sorry

end evaluate_root_l187_187240


namespace tan_30_deg_l187_187163

theorem tan_30_deg : 
  let θ := 30 * (Float.pi / 180) in  -- Conversion from degrees to radians
  Float.sin θ = 1 / 2 ∧ Float.cos θ = Float.sqrt 3 / 2 →
  Float.tan θ = Float.sqrt 3 / 3 := by
  intro h
  sorry

end tan_30_deg_l187_187163


namespace jung_kook_blue_balls_l187_187191

def num_boxes := 2
def blue_balls_per_box := 5
def total_blue_balls := num_boxes * blue_balls_per_box

theorem jung_kook_blue_balls : total_blue_balls = 10 :=
by
  sorry

end jung_kook_blue_balls_l187_187191


namespace rectangle_diagonal_length_l187_187488

theorem rectangle_diagonal_length {k : ℝ} (h1 : 2 * (3 * k + 2 * k) = 72)
  (h2 : k = 7.2) : 
  let length := 3 * k in
  let width := 2 * k in
  let diagonal := real.sqrt ((length ^ 2) + (width ^ 2)) in
  diagonal = 25.96 :=
by
  sorry

end rectangle_diagonal_length_l187_187488


namespace find_x_l187_187357

theorem find_x (x : ℝ) :
  let a := (x, 2)
  let b := (3, -1)
  let a_plus_b := (x + 3, 1)
  let a_minus_2b := (x - 6, 4)
  (∃ k : ℝ, a_plus_b = (k * a_minus_2b)) → x = -6 :=
by
  sorry

end find_x_l187_187357


namespace find_GH_l187_187438

-- Let ABC and FGH be similar triangles
variables (A B C F G H : Type) 
variables [metric_space A] [metric_space B] [metric_space C]
variables [metric_space F] [metric_space G] [metric_space H]
variables (triangle_ABC_sim_triangle_FGH : similar_triangles A B C F G H)
variables (BC : ℝ) (FG : ℝ) (AB : ℝ) (GH : ℝ)
variables (h1 : BC = 30) (h2 : FG = 9) (h3 : AB = 18)
variables (prop_ratio : AB / FG = BC / GH)

-- Finally, we express what we need to prove
theorem find_GH : GH = 15 :=
sorry

end find_GH_l187_187438


namespace smallest_m_for_no_real_solution_l187_187693

theorem smallest_m_for_no_real_solution : 
  (∀ x : ℝ, ∀ m : ℝ, (m * x^2 - 3 * x + 1 = 0) → false) ↔ (m ≥ 3) :=
by
  sorry

end smallest_m_for_no_real_solution_l187_187693


namespace distinct_integer_values_of_a_l187_187644

theorem distinct_integer_values_of_a : 
  let eq_has_integer_solutions (a : ℤ) : Prop := 
    ∃ (x y : ℤ), (x^2 + a*x + 9*a = 0) ∧ (y^2 + a*y + 9*a = 0) in
  (finset.univ.filter eq_has_integer_solutions).card = 5 := 
sorry

end distinct_integer_values_of_a_l187_187644


namespace number_of_integer_values_of_a_l187_187648

theorem number_of_integer_values_of_a (a : ℤ) : 
  (∃ x : ℤ, x^2 + a * x + 9 * a = 0) ↔ 
  (∃ (a_values : Finset ℤ), a_values.card = 6 ∧ ∀ a ∈ a_values, ∃ x : ℤ, x^2 + a * x + 9 * a = 0) :=
by
  sorry

end number_of_integer_values_of_a_l187_187648


namespace smallest_result_is_2784_l187_187453

def smallest_possible_result (d1 d2 d3 d4 : ℕ) : ℕ :=
  let pairs := [(d1 * 10 + d2, d3 * 10 + d4), 
                (d1 * 10 + d3, d2 * 10 + d4), 
                (d1 * 10 + d4, d2 * 10 + d3), 
                (d2 * 10 + d1, d3 * 10 + d4), 
                (d2 * 10 + d3, d1 * 10 + d4), 
                (d2 * 10 + d4, d1 * 10 + d3), 
                (d3 * 10 + d1, d2 * 10 + d4), 
                (d3 * 10 + d2, d1 * 10 + d4), 
                (d3 * 10 + d4, d1 * 10 + d2), 
                (d4 * 10 + d1, d2 * 10 + d3), 
                (d4 * 10 + d2, d1 * 10 + d3), 
                (d4 * 10 + d3, d1 * 10 + d2)]
  List.foldl min (10^10) (pairs.map (λ (x, y), x * y - min x y))

theorem smallest_result_is_2784 : smallest_possible_result 4 5 8 9 = 2784 :=
by
  sorry

end smallest_result_is_2784_l187_187453


namespace range_of_a_l187_187315

variables {R : Type*} [LinearOrder R]

/-- Given two distinct points (x₁, y₁) and (x₂, y₂) on the linear function y = (a - 1)x + 1,
     if the slope of the line joining these points is negative, then 'a' must be less than '1'. -/
theorem range_of_a (x₁ y₁ x₂ y₂ a : R) (h₁ : y₁ = (a - 1) * x₁ + 1) (h₂ : y₂ = (a - 1) * x₂ + 1) 
                   (h₃ : x₁ ≠ x₂) (h₄ : (y₁ - y₂) / (x₁ - x₂) < 0) : a < 1 :=
begin
  sorry,
end

end range_of_a_l187_187315


namespace partial_fraction_sum_inverse_l187_187837

theorem partial_fraction_sum_inverse (p q r A B C : ℝ)
  (hroots : (∀ s, s^3 - 20 * s^2 + 96 * s - 91 = (s - p) * (s - q) * (s - r)))
  (hA : ∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 20 * s^2 + 96 * s - 91) = A / (s - p) + B / (s - q) + C / (s - r)) :
  1 / A + 1 / B + 1 / C = 225 :=
sorry

end partial_fraction_sum_inverse_l187_187837


namespace a3_plus_a4_value_l187_187662

theorem a3_plus_a4_value
  (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : (1 - 2*x)^5 = a_0 + a_1*(1 + x) + a_2*(1 + x)^2 + a_3*(1 + x)^3 + a_4*(1 + x)^4 + a_5*(1 + x)^5) :
  a_3 + a_4 = -480 := 
sorry

end a3_plus_a4_value_l187_187662


namespace value_of_y_l187_187968

theorem value_of_y : 
  ∀ y : ℚ, y = (2010^2 - 2010 + 1 : ℚ) / 2010 → y = (2009 + 1 / 2010 : ℚ) := by
  sorry

end value_of_y_l187_187968


namespace Jimin_weight_l187_187383

variable (T J : ℝ)

theorem Jimin_weight (h1 : T - J = 4) (h2 : T + J = 88) : J = 42 :=
sorry

end Jimin_weight_l187_187383


namespace smallest_n_condition_l187_187068

theorem smallest_n_condition (n : ℕ) : (4 * n) ∣ (n^2) ∧ (5 * n) ∣ (u^3) → n = 100 :=
by
  sorry

end smallest_n_condition_l187_187068


namespace average_tomatoes_per_day_l187_187044

theorem average_tomatoes_per_day :
  let t₁ := 120
  let t₂ := t₁ + 50
  let t₃ := 2 * t₂
  let t₄ := t₁ / 2
  (t₁ + t₂ + t₃ + t₄) / 4 = 172.5 := by
  sorry

end average_tomatoes_per_day_l187_187044


namespace combined_area_of_WIN_sectors_l187_187560

-- Define the radius of the circular spinner.
def radius : ℝ := 15

-- Define the total area of the circle.
def total_area : ℝ := Real.pi * radius^2

-- Define the probability of winning for each WIN sector.
def win_probability_per_sector : ℝ := 1 / 6

-- Define the total winning probability from two WIN sectors.
def total_win_probability : ℝ := win_probability_per_sector + win_probability_per_sector

theorem combined_area_of_WIN_sectors :
  ∃ area : ℝ, area = total_area * total_win_probability := 
begin
  use (total_area * total_win_probability),
  sorry,
end

end combined_area_of_WIN_sectors_l187_187560


namespace min_value_y_of_parabola_l187_187181

theorem min_value_y_of_parabola :
  ∃ y : ℝ, ∃ x : ℝ, (∀ y' x', (y' + x') = (y' - x')^2 + 3 * (y' - x') + 3 → y' ≥ y) ∧
            y = -1/2 :=
by
  sorry

end min_value_y_of_parabola_l187_187181


namespace a_share_is_2500_l187_187577

theorem a_share_is_2500
  (x : ℝ)
  (h1 : 4 * x = 3 * x + 500)
  (h2 : 6 * x = 2 * 2 * x) : 5 * x = 2500 :=
by 
  sorry

end a_share_is_2500_l187_187577


namespace solve_diamond_l187_187372

theorem solve_diamond : 
  (∃ (Diamond : ℤ), Diamond * 5 + 3 = Diamond * 6 + 2) →
  (∃ (Diamond : ℤ), Diamond = 1) :=
by
  sorry

end solve_diamond_l187_187372


namespace line_intercepts_l187_187567

theorem line_intercepts (x y : ℝ) (P : ℝ × ℝ) (h1 : P = (1, 4)) (h2 : ∃ k : ℝ, (x + y = k ∨ 4 * x - y = 0) ∧ 
  ∃ intercepts_p : ℝ × ℝ, intercepts_p = (k / 2, k / 2)) :
  ∃ k : ℝ, (x + y - k = 0 ∧ k = 5) ∨ (4 * x - y = 0) :=
sorry

end line_intercepts_l187_187567


namespace t50_mod_7_l187_187480

def sequence_T : ℕ → ℕ 
| 0       := 3  
| (n + 1) := 3 ^ sequence_T n

theorem t50_mod_7 : (sequence_T 49) % 7 = 6 :=
sorry

end t50_mod_7_l187_187480


namespace coordinates_of_point_in_fourth_quadrant_l187_187909

theorem coordinates_of_point_in_fourth_quadrant 
  (P : ℝ × ℝ)
  (h₁ : P.1 > 0) -- P is in the fourth quadrant, so x > 0
  (h₂ : P.2 < 0) -- P is in the fourth quadrant, so y < 0
  (dist_x_axis : P.2 = -5) -- Distance from P to x-axis is 5 (absolute value of y)
  (dist_y_axis : P.1 = 3)  -- Distance from P to y-axis is 3 (absolute value of x)
  : P = (3, -5) :=
sorry

end coordinates_of_point_in_fourth_quadrant_l187_187909


namespace max_value_expression_l187_187799

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end max_value_expression_l187_187799


namespace garage_sale_items_l187_187545

theorem garage_sale_items (h9: ∃ (radio_price: ℕ), nth_highest radio_price 9) 
                          (h35: ∃ (radio_price: ℕ), nth_lowest radio_price 35) :
                          ∃ (n: ℕ), n = 43 :=
by
  -- Placeholder proof
  sorry

end garage_sale_items_l187_187545


namespace sum_of_two_numbers_l187_187872

theorem sum_of_two_numbers (a b : ℕ) (h1 : a - b = 10) (h2 : a = 22) : a + b = 34 :=
sorry

end sum_of_two_numbers_l187_187872


namespace eval_power_l187_187203

-- Given condition
def sixty_four : ℕ := 64

-- Given condition rewritten in Lean
def sixty_four_as_two_powersix : sixty_four = 2^6 := by
  sorry

-- Prove that 64^(5/6) = 32
theorem eval_power : real.exp (5/6 * real.log 64) = 32 := by
  have h1 : 64 = 2^6 := sixty_four_as_two_powersix
  sorry

end eval_power_l187_187203


namespace PC_equals_QC_l187_187845

open Classical
open Geometry
open RealInnerProductSpace

variables {A B C D E P Q : Point} 

-- Given an acute-angled triangle ABC
axiom acute_triangle (hABC : Triangle A B C) (h_acute : acute hABC)

-- Points D and E on BC and AC respectively
axiom point_D_on_BC (D : Point) (hD : lies_on D (line B C))
axiom point_E_on_AC (E : Point) (hE : lies_on E (line A C))

-- AD ⊥ BC and BE ⊥ AC
axiom perp_AD_BC : perpendicular (line A D) (line B C)
axiom perp_BE_AC : perpendicular (line B E) (line A C)

-- P is the point where AD meets the semicircle on BC
axiom point_P (P : Point) (hP : lies_on P (semicircle_outward B C) ∧ lies_on P (line A D))

-- Q is the point where BE meets the semicircle on AC
axiom point_Q (Q : Point) (hQ : lies_on Q (semicircle_outward A C) ∧ lies_on Q (line B E))

-- The theorem to prove: PC = QC
theorem PC_equals_QC : dist P C = dist Q C := 
sorry

end PC_equals_QC_l187_187845


namespace evaluate_64_pow_5_div_6_l187_187271

theorem evaluate_64_pow_5_div_6 : (64 : ℝ) ^ (5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ) ^ 6 := by norm_num
  have h2 : (64 : ℝ) ^ (5 / 6) = ((2 : ℝ) ^ 6) ^ (5 / 6) := by rw h1
  have h3 : ((2 : ℝ) ^ 6) ^ (5 / 6) = (2 : ℝ) ^ (6 * (5 / 6)) := by rw [Real.rpow_mul]
  have h4 : (2 : ℝ) ^ (6 * (5 / 6)) = (2 : ℝ) ^ 5 := by norm_num
  rw [h2, h3, h4]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187271


namespace avg_age_increases_by_6_l187_187392

-- Definition of the problem's conditions
def original_avg_age (A : ℝ) : Prop :=
  -- Total age of 10 men is 10 * A
  ∀ a, total_age a = 10 * A

def replaced_age_diff : ℝ :=
  -- Difference caused by replacing two men with ages 18 and 22 with two women with average age 50
  (50 + 50) - (18 + 22)

def new_avg_age_increase (increase : ℝ) : Prop :=
  -- Check if the increase in average age is 6 years
  increase = replaced_age_diff / 10

-- Theorem stating the increase in average age
theorem avg_age_increases_by_6 : ∃ inc, new_avg_age_increase inc ∧ inc = 6 := by
  sorry

end avg_age_increases_by_6_l187_187392


namespace probability_first_4_second_club_third_2_l187_187949

theorem probability_first_4_second_club_third_2 :
  let deck_size := 52
  let prob_4_first := 4 / deck_size
  let deck_minus_first_card := deck_size - 1
  let prob_club_second := 13 / deck_minus_first_card
  let deck_minus_two_cards := deck_minus_first_card - 1
  let prob_2_third := 4 / deck_minus_two_cards
  prob_4_first * prob_club_second * prob_2_third = 1 / 663 :=
by
  sorry

end probability_first_4_second_club_third_2_l187_187949


namespace total_cups_l187_187003

theorem total_cups (r_b r_f r_s : ℕ) (s : ℕ) (h_rb : r_b = 2) (h_rf : r_f = 5) (h_rs : r_s = 3) (h_s : s = 9) :
  s * (r_b + r_f + r_s) / r_s = 30 :=
by
  rw [h_rb, h_rf, h_rs, h_s]
  norm_num
  sorry

end total_cups_l187_187003


namespace rectangle_perimeter_increase_l187_187381

theorem rectangle_perimeter_increase (l w : ℝ) :
  let initial_perimeter := 2 * (l + w),
      new_perimeter := 2 * (1.1 * l + 1.1 * w) in
  new_perimeter = 1.1 * initial_perimeter :=
by 
  sorry

end rectangle_perimeter_increase_l187_187381


namespace x1_xn_le_neg_one_div_n_l187_187434

theorem x1_xn_le_neg_one_div_n
    (n : ℕ)
    (x : ℕ → ℝ)
    (h : n > 0)
    (h_sorted : ∀ i j, i < j → x i ≤ x j)
    (h_sum_zero : ∑ i in finset.range n, x i = 0)
    (h_sum_sq_one : ∑ i in finset.range n, (x i)^2 = 1) :
    x 0 * x (n-1) ≤ -1 / n :=
by
  sorry

end x1_xn_le_neg_one_div_n_l187_187434


namespace ratio_not_greater_than_one_twentieth_l187_187998

noncomputable def total_cost_intersections (m k a x : ℝ) : ℝ :=
  m * k * (a * x + 5)

noncomputable def ratio_costs (x y : ℝ) : ℝ :=
  x / y

theorem ratio_not_greater_than_one_twentieth
  (a x m k : ℝ) (h1 : x = 0.2 * a) (h2 : k ≥ 3) :
  let y := total_cost_intersections m k a x in
  ratio_costs (m * x) y ≤ 1 / 20 :=
by
  sorry

end ratio_not_greater_than_one_twentieth_l187_187998


namespace circumcircles_touch_l187_187038

theorem circumcircles_touch
  {A B1 C1 B2 C2 : Point}
  (h1 : IntersectTwoCircles at A)
  (h2 : TouchesAtPoints S1 B1 C1)
  (h3 : TouchesAtPoints S2 B2 C2)
  (h4 : TangencyEquality B1 C1 B2 C2) :
  Touches (Circumcircle A B1 C1) (Circumcircle A B2 C2) := sorry

end circumcircles_touch_l187_187038


namespace integer_values_of_a_l187_187641

theorem integer_values_of_a : 
  ∃ (a : Set ℤ), (∀ x, x ∈ a → ∃ (y z : ℤ), x^2 + x * y + 9 * y = 0) ∧ (a.card = 6) :=
by
  sorry

end integer_values_of_a_l187_187641


namespace max_value_expression_l187_187797

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end max_value_expression_l187_187797


namespace x1_xn_le_neg_one_div_n_l187_187435

theorem x1_xn_le_neg_one_div_n
    (n : ℕ)
    (x : ℕ → ℝ)
    (h : n > 0)
    (h_sorted : ∀ i j, i < j → x i ≤ x j)
    (h_sum_zero : ∑ i in finset.range n, x i = 0)
    (h_sum_sq_one : ∑ i in finset.range n, (x i)^2 = 1) :
    x 0 * x (n-1) ≤ -1 / n :=
by
  sorry

end x1_xn_le_neg_one_div_n_l187_187435


namespace strike_12_oclock_time_l187_187561

-- Define the conditions
def strike_intervals (t : ℝ) : list ℝ :=
  [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

def total_time_9_oclock (t : ℝ) : ℝ :=
  t + t + 0.2 + t + 0.4 + t + 0.6 + t + 0.8 + t + 1.0 + t + 1.2 + t + 1.4

def total_time_12_oclock (t : ℝ) : ℝ :=
  11 * t + (strike_intervals t).sum

theorem strike_12_oclock_time : 
  ∃ t : ℝ, total_time_9_oclock t = 7 →
  total_time_12_oclock t = 12.925 :=
by
  -- We declare the value of t calculated in the solution steps
  let t : ℝ := 0.175
  use t
  intros ht9
  rw [total_time_9_oclock] at ht9
  have ht9s : 8 * t + 5.6 = 7 := by exact ht9
  rw [mul_add] at ht9s
  have ht : t = 0.175 := by linarith
  
  -- Compatibility check for the 12 o'clock
  have hsum_intervals :
    (strike_intervals t).sum = 11 := by
    simp [strike_intervals]
    sorry

  rw [total_time_12_oclock, ht, hsum_intervals]
  linarith
  sorry

end strike_12_oclock_time_l187_187561


namespace range_of_sin_cos_sin_l187_187499

theorem range_of_sin_cos_sin (x : ℝ) :
  let y := sin x * (cos x - sin x)
  in ∃ l u, (l = (-1 - sqrt 2) / 2) ∧ (u = (-1 + sqrt 2) / 2) ∧ (∀ y, ∃ x, y = sin x * (cos x - sin x) ↔ l ≤ y ∧ y ≤ u) :=
sorry

end range_of_sin_cos_sin_l187_187499


namespace angle_DAB_eq_2_angle_ADB_l187_187678

-- Definitions based on the conditions
variables 
  (ABC : Triangle)
  (A B C D : Point)
  [Circle α]
  [Circle β]
  (hA : α ∈ A)
  (hB : α ∈ B)
  (hC : β ∈ C)
  (hTangB : tangent (side BC) B)
  (hTangC : tangent (side BC) C)
  (hIntersection : second_intersection_point α β D A)
  (hBC_eq_2BD : BC = 2 * BD)

-- Prove the statement
theorem angle_DAB_eq_2_angle_ADB :
  ∠ DAB = 2 * ∠ ADB := 
sorry

end angle_DAB_eq_2_angle_ADB_l187_187678


namespace trajectory_midpoint_trajectory_l187_187326

noncomputable theory

open Real

def point := ℝ × ℝ

def dist (p1 p2 : point) : ℝ :=
  sqrt ((p2.1-p1.1)^2 + (p2.2-p1.2)^2)

def circle (center : point) (radius : ℝ) (p : point) : Prop :=
  dist center p = radius

def is_midpoint (m A B : point) : Prop :=
  m.1 = (A.1 + B.1) / 2 ∧ m.2 = (A.2 + B.2) / 2

theorem trajectory_midpoint_trajectory (A B M : point)
  (hB : B = (4, 3))
  (hA_on_circle : circle (-1, 0) 2 A)
  (hM_midpoint : is_midpoint M A B) :
  circle (3/2, 3/2) 1 M := by
  sorry

end trajectory_midpoint_trajectory_l187_187326


namespace rectangle_diagonal_length_l187_187487

theorem rectangle_diagonal_length {k : ℝ} (h1 : 2 * (3 * k + 2 * k) = 72)
  (h2 : k = 7.2) : 
  let length := 3 * k in
  let width := 2 * k in
  let diagonal := real.sqrt ((length ^ 2) + (width ^ 2)) in
  diagonal = 25.96 :=
by
  sorry

end rectangle_diagonal_length_l187_187487


namespace tan_30_l187_187146

theorem tan_30 : Real.tan (Real.pi / 6) = Real.sqrt 3 / 3 := 
by 
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry
  have h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2 := by sorry
  calc
    Real.tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6) : Real.tan_eq_sin_div_cos _
    ... = (1 / 2) / (Real.sqrt 3 / 2) : by rw [h1, h2]
    ... = (1 / 2) * (2 / Real.sqrt 3) : by rw Div.div_eq_mul_inv
    ... = 1 / Real.sqrt 3 : by norm_num
    ... = Real.sqrt 3 / 3 : by rw [Div.inv_eq_inv, Mul.comm, Mul.assoc, Div.mul_inv_cancel (Real.sqrt_ne_zero _), one_div Real.sqrt 3, inv_mul_eq_div]

-- Additional necessary function apologies for the unproven theorems.
noncomputable def _root_.Real.sqrt (x:ℝ) : ℝ := sorry

noncomputable def _root_.Real.tan (x : ℝ) : ℝ :=
  (Real.sin x) / (Real.cos x)

#eval tan_30 -- check result

end tan_30_l187_187146


namespace problem_statement_l187_187307

noncomputable def f (x m : ℝ) := (1/2)^(|x - m|) - 1

theorem problem_statement (m : ℝ) (a b c : ℝ)
  (h_even : ∀ x, f x m = f (-x) m)
  (h_a : a = f (Real.logb 0.5 3) m)
  (h_b : b = f (Real.logb 2 5) m)
  (h_c : c = f 0 m)
  (h_log_comparison : 0 < Real.logb 2 3 ∧ Real.logb 2 3 < Real.logb 2 5) :
  c > a ∧ a > b :=
by
  sorry

end problem_statement_l187_187307


namespace smallest_integer_with_10_divisors_l187_187533

theorem smallest_integer_with_10_divisors :
  ∃ n : ℕ, (∀ m : ℕ, (m > 0 ∧ ∃ d : ℕ → Prop, (∀ k, d k ↔ k ∣ m) ∧ (nat.card { k // d k } = 10)) → n ≤ m) ∧ (n = 48) := by
sorry

end smallest_integer_with_10_divisors_l187_187533


namespace tyesha_correct_balance_l187_187958

def tyesha_hourly_rate : ℕ := 5
def tyesha_hours_worked : ℕ := 7
def tyesha_initial_balance : ℕ := 20

def tyesha_total_earnings : ℕ := tyesha_hourly_rate * tyesha_hours_worked :=
by 
  let earnings := tyesha_hourly_rate * tyesha_hours_worked
  exact earnings

def tyesha_final_balance : ℕ := tyesha_initial_balance + tyesha_total_earnings :=
by 
  let final_balance := tyesha_initial_balance + tyesha_total_earnings
  exact final_balance

theorem tyesha_correct_balance : tyesha_final_balance = 55 :=
by
  rw [tyesha_total_earnings, nat.mul_comm]
  rw [tyesha_final_balance, nat.add_comm]
  exact rfl

end tyesha_correct_balance_l187_187958


namespace vertex_of_quadratic_l187_187905

theorem vertex_of_quadratic :
  ∀ x : ℝ, let y := -2 * (x + 1) ^ 2 + 5
  in y = (λ y : ℝ, if x = -1 then y = 5 else False) :=
by
  sorry

end vertex_of_quadratic_l187_187905


namespace power_vs_square_l187_187463

theorem power_vs_square (n : ℕ) (h : n ≥ 4) : 2^n ≥ n^2 := by
  sorry

end power_vs_square_l187_187463


namespace eccentricity_of_ellipse_line_AB_tangent_to_circle_l187_187335

open Real

-- Define the ellipse
def ellipse (x y: ℝ) : Prop := x^2 + 2 * y^2 = 4

-- Define the circle
def circle (x y: ℝ) : Prop := x^2 + y^2 = 2

-- Define the perpendicularity condition OA ⊥ OB
def perpendicular (xa ya xb yb: ℝ): Prop := xa * xb + ya * yb = 0

-- Problem statements
theorem eccentricity_of_ellipse : 
  (∀ (x y : ℝ), ellipse x y → (sqrt 2 / 2)) := 
sorry

theorem line_AB_tangent_to_circle :
  (∀ (xa ya xb yb : ℝ), ellipse xa ya → (yb = 2) → perpendicular xa ya xb yb → 
    (∀ (x y : ℝ), circle x y → x = abs (sqrt 2))) :=
sorry

end eccentricity_of_ellipse_line_AB_tangent_to_circle_l187_187335


namespace evaluate_64_pow_5_div_6_l187_187228

theorem evaluate_64_pow_5_div_6 : (64 : ℝ)^(5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ)^6 := by norm_num
  rw [← h1]
  have h2 : ((2 : ℝ)^6)^(5 / 6) = (2 : ℝ)^(6 * (5 / 6)) := by rw [Real.rpow_mul]
  rw [h2]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187228


namespace triangle_is_right_l187_187729

theorem triangle_is_right (n : ℝ) (h : n > 1) :
  let a := n^2 - 1
  let b := 2 * n
  let c := n^2 + 1
  a^2 + b^2 = c^2 →
  is_right_triangle a b c :=
sorry

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

end triangle_is_right_l187_187729


namespace smallest_n_l187_187063

theorem smallest_n (n : ℕ) : (∃ (m1 m2 : ℕ), 4 * n = m1^2 ∧ 5 * n = m2^3) ↔ n = 500 := 
begin
  sorry
end

end smallest_n_l187_187063


namespace player_one_wins_with_optimal_play_l187_187034

-- Definitions
def rectangular_table (l w : ℕ) := l > 0 ∧ w > 0

def euro_coin := ℕ -- Assume we define coins as natural numbers

def valid_move (table : ℕ × ℕ) (coins : list (ℕ × ℕ)) (move : ℕ × ℕ) : Prop :=
  move.1 >= 0 ∧ move.1 < table.1 ∧ move.2 >= 0 ∧ move.2 < table.2 ∧ (move ∉ coins)

-- Conditions
variables {l w : ℕ} (table : ℕ × ℕ := (l, w)) (coins : list (ℕ × ℕ))

-- Proof Statement
theorem player_one_wins_with_optimal_play (h_table : rectangular_table l w)
  (h_coins : ∀ move, valid_move table coins move → valid_move table (move :: coins) move) :
  ∃ strategy : (list (ℕ × ℕ) → ℕ × ℕ), 
    (∀ coins, strategy coins ∈ table ∧ valid_move table coins (strategy coins)) → 
    (strategy coins ∉ coins) :=
sorry

end player_one_wins_with_optimal_play_l187_187034


namespace problem_solution_l187_187668

noncomputable def a_solution : ℝ :=
  let (x1, y1, x2, y2, b : ℝ) := (0, 0, 0, 0, 0) -- placeholders for the intersection points and parameter b
  let C1 := x1^2 + y1^2 - 2*x1 + 4*y1 - b^2 + 5 = 0 ∧ x2^2 + y2^2 - 2*x2 + 4*y2 - b^2 + 5 = 0
  let C2 := x1^2 + y1^2 - 2*(4 - 6)*x1 - 2*4*y1 + 2*4^2 - 12*4 + 27 = 0 ∧ x2^2 + y2^2 - 2*(4 - 6)*x2 - 2*4*y2 + 2*4^2 - 12*4 + 27 = 0
  let intersection_condition := (y1 + y2) / (x1 + x2) + (x1 - x2) / (y1 - y2) = 0
  if C1 ∧ C2 ∧ intersection_condition then 4 else sorry

theorem problem_solution : a_solution = 4 :=
by {
  -- Conditions
  let a : ℝ := 4,
  let b : ℝ := 0,
  let C1 := ∀ (x y : ℝ), x^2 + y^2 - 2*x + 4*y - b^2 + 5 = 0,
  let C2 := ∀ (x y : ℝ), x^2 + y^2 - 2*(a-6)*x - 2*a*y + 2*a^2 - 12*a + 27 = 0,
  let intersection_condition := ∀ (a x y : ℝ), (a, x, y) ≠ (0, 0, 0) ∧ (y / x) * (y / x) = -1,
  sorry -- The proof
}

end problem_solution_l187_187668


namespace evaluate_root_l187_187238

theorem evaluate_root : 64 ^ (5 / 6 : ℝ) = 32 :=
by sorry

end evaluate_root_l187_187238


namespace largest_gcd_l187_187013

theorem largest_gcd (a b : ℕ) (h : a + b = 1008) : ∃ d, d = gcd a b ∧ (∀ d', d' = gcd a b → d' ≤ d) ∧ d = 504 :=
by
  sorry

end largest_gcd_l187_187013


namespace probability_continuous_stripe_l187_187616

def tetrahedron_faces : Type := fin 4
def stripe_orientations : Type := fin 2

def stripe_combination : Type := tetrahedron_faces → stripe_orientations

axiom continuous_stripe_encircle (sc : stripe_combination) : Prop

def count_favorable_outcomes : ℕ := 2
def total_possible_combinations : ℕ := 16

theorem probability_continuous_stripe :
  (count_favorable_outcomes : ℚ) / total_possible_combinations = 1 / 8 :=
by
  sorry

end probability_continuous_stripe_l187_187616


namespace kiwi_count_initial_l187_187472

-- Define the initial conditions and what we need to prove.
theorem kiwi_count_initial (K : ℕ) 
  (h1 : 24 = 0.30 * (24 + K + 26)) :
  K = 30 :=
by
  -- Since detailed proof is not required, we skip it.
  sorry

end kiwi_count_initial_l187_187472


namespace max_value_of_expression_l187_187811

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l187_187811


namespace geom_seq_sum_l187_187747

theorem geom_seq_sum (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 3)
  (h2 : a 1 + a 2 + a 3 = 21)
  (h3 : ∀ n, a (n + 1) = a n * q) : a 4 + a 5 + a 6 = 168 :=
sorry

end geom_seq_sum_l187_187747


namespace binomial_expansion_term_eight_l187_187305

theorem binomial_expansion_term_eight:
  (∑ i in Finset.range 11, ((1:ℝ) + x) ^ i * ((1:ℝ) - x) ^ (10 - i) * (Nat.choose 10 i)) = (1 + x) ^ 10 →
  (∑ i in Finset.range 9, (1 - x) ^ 8 * 4 * (Nat.choose 10 8)) = 180 :=
begin
  sorry
end

end binomial_expansion_term_eight_l187_187305


namespace tan_30_deg_l187_187174

theorem tan_30_deg : 
  let θ := (30 : ℝ) * (Real.pi / 180)
  in Real.sin θ = 1 / 2 ∧ Real.cos θ = Real.sqrt 3 / 2 → Real.tan θ = Real.sqrt 3 / 3 :=
by
  intro h
  let th := θ
  have h1 : Real.sin th = 1 / 2 := And.left h
  have h2 : Real.cos th = Real.sqrt 3 / 2 := And.right h
  sorry

end tan_30_deg_l187_187174


namespace invalid_votes_percentage_l187_187751

variable (T : ℕ) (V : ℕ)

theorem invalid_votes_percentage 
  (h1 : T = 7000) 
  (h2 : 0.55 * V = 0.55 * (2520 / 0.45)) 
  (h3 : 2520 = 0.45 * V) :
  ((T - V).toRat / T.toRat) * 100 = 20 := 
by
  sorry

end invalid_votes_percentage_l187_187751


namespace cups_added_l187_187989

/--
A bowl was half full of water. Some cups of water were then added to the bowl, filling the bowl to 70% of its capacity. There are now 14 cups of water in the bowl.
Prove that the number of cups of water added to the bowl is 4.
-/
theorem cups_added (C : ℚ) (h1 : C / 2 + 0.2 * C = 14) : 
  14 - C / 2 = 4 :=
by
  sorry

end cups_added_l187_187989


namespace total_people_on_bus_l187_187860

-- Definitions of the conditions
def num_boys : ℕ := 50
def num_girls : ℕ := (2 / 5 : ℚ) * num_boys
def num_students : ℕ := num_boys + num_girls.toNat
def num_non_students : ℕ := 3 -- Mr. Gordon, the driver, and the assistant

-- The theorem to be proven
theorem total_people_on_bus : num_students + num_non_students = 123 := by
  sorry

end total_people_on_bus_l187_187860


namespace line_passing_quadrants_l187_187726

-- Define the conditions as assumptions in Lean
variables {A B C : ℝ}

-- Define the main theorem statement
theorem line_passing_quadrants (h1 : A * C < 0) (h2 : B * C < 0) :
  ∃ Q : set ℕ, Q = {1, 2, 4} ∧ ∀ (x y : ℝ), A * x + B * y + C = 0 → 
  ((x > 0 ∧ y > 0 → 1 ∈ Q) ∧ 
   (x < 0 ∧ y > 0 → 2 ∈ Q) ∧ 
   (x > 0 ∧ y < 0 → 4 ∈ Q)) := 
sorry

end line_passing_quadrants_l187_187726


namespace evaluate_pow_l187_187249

theorem evaluate_pow : 64^(5/6 : ℝ) = 32 := by
  sorry

end evaluate_pow_l187_187249


namespace largest_gcd_l187_187014

theorem largest_gcd (a b : ℕ) (h : a + b = 1008) : ∃ d, d = gcd a b ∧ (∀ d', d' = gcd a b → d' ≤ d) ∧ d = 504 :=
by
  sorry

end largest_gcd_l187_187014


namespace evaluate_64_pow_5_div_6_l187_187227

theorem evaluate_64_pow_5_div_6 : (64 : ℝ)^(5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ)^6 := by norm_num
  rw [← h1]
  have h2 : ((2 : ℝ)^6)^(5 / 6) = (2 : ℝ)^(6 * (5 / 6)) := by rw [Real.rpow_mul]
  rw [h2]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187227


namespace family_reunion_weight_gain_l187_187138

def orlando_gained : ℕ := 5

def jose_gained (orlando: ℕ) : ℕ := 2 * orlando + 2

def fernando_gained (jose: ℕ) : ℕ := jose / 2 - 3

def total_weight_gained : ℕ := 
  let orlando := orlando_gained in
  let jose := jose_gained orlando in
  let fernando := fernando_gained jose in
  orlando + jose + fernando

theorem family_reunion_weight_gain : total_weight_gained = 20 := by
  sorry

end family_reunion_weight_gain_l187_187138


namespace range_of_k_l187_187344

-- Definitions for the condition
def inequality_holds (k : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + (k-1)*x^2 + 1 ≥ 0

-- Theorem statement
theorem range_of_k (k : ℝ) : inequality_holds k → k ≥ 1 :=
sorry

end range_of_k_l187_187344


namespace juanita_spends_more_l187_187362

-- Define the expenditures
def grant_yearly_expenditure : ℝ := 200.00

def juanita_weekday_expenditure : ℝ := 0.50

def juanita_sunday_expenditure : ℝ := 2.00

def weeks_per_year : ℕ := 52

-- Given conditions translated to Lean
def juanita_weekly_expenditure : ℝ :=
  (juanita_weekday_expenditure * 6) + juanita_sunday_expenditure

def juanita_yearly_expenditure : ℝ :=
  juanita_weekly_expenditure * weeks_per_year

-- The statement we need to prove
theorem juanita_spends_more : (juanita_yearly_expenditure - grant_yearly_expenditure) = 60.00 :=
by
  sorry

end juanita_spends_more_l187_187362


namespace largest_gcd_l187_187022

theorem largest_gcd (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 1008) : 
  ∃ d : ℕ, d = Int.gcd a b ∧ d = 504 :=
by
  sorry

end largest_gcd_l187_187022


namespace only_one_proposition_true_l187_187680

open Set Real

def prop_p (a : ℝ) := ∀ x ∈ Icc (1 : ℝ) 2, x^2 - a ≥ 0
def prop_q (a : ℝ) := ∃ x : ℝ, x^2 + 2 * a * x + 4 = 0

theorem only_one_proposition_true (a : ℝ) :
  xor (prop_p a) (prop_q a) ↔ a ∈ Ioc (-2) 1 ∪ Ici 2 := sorry

end only_one_proposition_true_l187_187680


namespace length_of_platform_l187_187974

variables (t L T_p T_s : ℝ)
def train_length := 200  -- length of the train in meters
def platform_cross_time := 50  -- time in seconds to cross the platform
def pole_cross_time := 42  -- time in seconds to cross the signal pole

theorem length_of_platform :
  T_p = platform_cross_time ->
  T_s = pole_cross_time ->
  t = train_length ->
  (L = 38) :=
by
  intros hp hsp ht
  sorry  -- proof goes here

end length_of_platform_l187_187974


namespace tangent_line_to_curve_l187_187928

theorem tangent_line_to_curve (a : ℝ) : (∀ (x : ℝ), y = x → y = a + Real.log x) → a = 1 := 
sorry

end tangent_line_to_curve_l187_187928


namespace percentage_increase_l187_187907

variables (a b t d : ℝ)
definition original_cost : ℝ := a * (t * b^4) / d
definition new_cost : ℝ := a * ((t/2) * (2*b)^4) / d

theorem percentage_increase : 
  (new_cost a b t d - original_cost a b t d) / original_cost a b t d * 100 = 700 := 
by 
  sorry

end percentage_increase_l187_187907


namespace tangent_line_equation_l187_187630

theorem tangent_line_equation (e x y : ℝ) (h_curve : y = x^3 / e) (h_point : x = e ∧ y = e^2) :
  3 * e * x - y - 2 * e^2 = 0 :=
sorry

end tangent_line_equation_l187_187630


namespace determine_q_l187_187610

noncomputable def q (x : ℝ) := (8/3) * x^3 - (16/3) * x^2 - (40/3) * x + 16

theorem determine_q :
  (∀ x, ∃ a, q x = a * (x - 3) * (x - 1) * (x + 2)) ∧
  (∀ x, ∃ b, x ≠ 3 ∧ x ≠ 1 ∧ x ≠ (-2) → q x = (x^4 - x^3 - 6x^2 + x + 6) / b) ∧
  q(4) = 48 :=
by
  sorry

end determine_q_l187_187610


namespace six_vertices_one_circle_l187_187588

-- Define a structure for acute triangles
structure AcuteTriangle (A B C : Type) := 
(angle_ABC : acute_angle A B C)
(angle_BCA : acute_angle B C A)
(angle_CAB : acute_angle C A B)

-- Define a condition for points on sides
structure PointsOnSides (A B C A1 A2 B1 B2 C1 C2 : Type) :=
(between_A2_A1C : between A2 A1 C)
(between_B2_B1A : between B2 B1 A)
(between_C2_C1B : between C2 C1 B)

-- Define the angle equality condition
structure AngleConditions (A B C A1 A2 B1 B2 C1 C2 : Type) :=
(angle_AAA_1A_2 : ∠ A A1 A2 = ∠ A A2 A1)
(angle_BBB_1B_2 : ∠ B B1 B2 = ∠ B B2 B1)
(angle_CCC_1C_2 : ∠ C C1 C2 = ∠ C C2 C1)

-- Define the triangle intersections
def Triangles (A B C A1 A2 B1 B2 C1 C2 : Type) :=
let X1 := intersection BB1 CC1,
    Y1 := intersection CC1 AA1,
    Z1 := intersection AA1 BB1,
    X2 := intersection BB2 CC2,
    Y2 := intersection CC2 AA2,
    Z2 := intersection AA2 BB2
in (X1, Y1, Z1, X2, Y2, Z2)

-- Prove all six vertices lie on a single circle.
theorem six_vertices_one_circle 
  (A B C A1 A2 B1 B2 C1 C2 : Type)
  [AcuteTriangle A B C] 
  [PointsOnSides A B C A1 A2 B1 B2 C1 C2] 
  [AngleConditions A B C A1 A2 B1 B2 C1 C2] :
  let (X1, Y1, Z1, X2, Y2, Z2) := Triangles A B C A1 A2 B1 B2 C1 C2 
  in cyclic X1 Y1 Z1 ∧ cyclic X2 Y2 Z2 :=
  sorry

end six_vertices_one_circle_l187_187588


namespace cubs_win_series_prob_is_79percent_l187_187892

def cubs_win_series_probability (prob_win : ℚ) (required_wins: Nat) (num_games: Nat) : ℚ :=
  let lose_prob := 1 - prob_win
  let prob_k_games (k: Nat) := choose (required_wins + k - 1) k * (prob_win ^ required_wins) * (lose_prob ^ k)
  (List.sum $ List.map (λ k => prob_k_games k) (List.range (required_wins - 1 + 1))) / 100

theorem cubs_win_series_prob_is_79percent:
  cubs_win_series_probability (2/3) 3 5 = 0.79 := sorry

end cubs_win_series_prob_is_79percent_l187_187892


namespace total_days_2000_to_2003_correct_l187_187720

-- Define the days in each type of year
def days_in_leap_year : ℕ := 366
def days_in_common_year : ℕ := 365

-- Define each year and its corresponding number of days
def year_2000 := days_in_leap_year
def year_2001 := days_in_common_year
def year_2002 := days_in_common_year
def year_2003 := days_in_common_year

-- Calculate the total number of days from 2000 to 2003
def total_days_2000_to_2003 : ℕ := year_2000 + year_2001 + year_2002 + year_2003

theorem total_days_2000_to_2003_correct : total_days_2000_to_2003 = 1461 := 
by
  unfold total_days_2000_to_2003 year_2000 year_2001 year_2002 year_2003 
        days_in_leap_year days_in_common_year 
  exact rfl

end total_days_2000_to_2003_correct_l187_187720


namespace not_must_be_even_number_of_even_scores_l187_187566

-- Define the conditions of the problem
def round_robin_tournament (teams games : ℕ) (scores : Fin teams → ℕ) : Prop :=
  teams = 14 ∧
  games = 91 ∧
  (∀ (i j : Fin teams), i ≠ j → scores i + scores j = 3 ∨ scores i + scores j = 4)

-- The problem we need to prove
theorem not_must_be_even_number_of_even_scores (scores : Fin 14 → ℕ) :
  round_robin_tournament 14 91 scores →
  ¬(∃ S, S = {n : ℕ | ∃ i, scores i = n ∧ n % 2 = 0} ∧ S.card % 2 = 0) := 
by {
  intro h,
  sorry
}

end not_must_be_even_number_of_even_scores_l187_187566


namespace evaluate_pow_l187_187258

theorem evaluate_pow : (64 : ℝ) = (8 : ℝ) ^ 2 → (8 : ℝ) = (2 : ℝ) ^ 3 → (64 : ℝ) ^ (5 / 6) = 32 :=
by
  intros h1 h2
  rw h1
  rw h2
  have h3 : (2 : ℝ)^3 ^ 2 = (2 : ℝ) ^ 6 := by ring_exp
  rw h3
  sorry

end evaluate_pow_l187_187258


namespace standard_equation_of_ellipse_max_area_triangle_l187_187333

-- The conditions for the ellipse
def minor_axis_length (b : ℝ) : Prop := 2 * b = 2
def ellipse_eccentricity (a b : ℝ) : Prop := (b > 0 ∧ a > b) ∧ (sqrt (a^2 - b^2) / a = sqrt 2 / 2)

-- The conditions for the line and its intersection
def line_intersects_ellipse (k m x1 x2 y1 y2 : ℝ) : Prop :=
  y1 = k * x1 + m ∧
  y2 = k * x2 + m ∧
  (x1^2 / 2 + y1^2 = 1) ∧
  (x2^2 / 2 + y2^2 = 1)

def perpendicular_bisector_condition (k m x1 x2 y1 y2 : ℝ) : Prop :=
  let midpoint_x := (x1 + x2) / 2 in
  let midpoint_y := (y1 + y2) / 2 in
  (midpoint_y + 1/2) / midpoint_x = -1 / k

-- Prove the standard equation of the ellipse
theorem standard_equation_of_ellipse (a b : ℝ) (h_minor : minor_axis_length b) 
  (h_ecc : ellipse_eccentricity a b) : (x y : ℝ) -> (x^2 / a^2 + y^2 / b^2 = 1) :=
begin
  sorry
end

-- Prove the maximum area of the triangle
theorem max_area_triangle (k m x1 x2 y1 y2 : ℝ)
  (h_line : line_intersects_ellipse k m x1 x2 y1 y2)
  (h_perp : perpendicular_bisector_condition k m x1 x2 y1 y2)
  : (area : ℝ) -> (area = sqrt 2 / 2) :=
begin
  sorry
end

end standard_equation_of_ellipse_max_area_triangle_l187_187333


namespace ellipse_equation_range_OQ_l187_187351

-- Conditions for the problem
variables {a b : ℝ} (h₀ : a > b) (h₁ : b > 1) 
variable (P : ℝ × ℝ)  -- Assuming P = (P.1, P.2) lies on the unit circle
variables {A F O : ℝ × ℝ}  -- Points A, F, O

-- Additional geometric conditions
variable h₂ : a = 2
variable h₃ : b = Real.sqrt 3

-- Given specific angle and distances
variable h4 : (angle P O A = Real.pi / 3)
variable h5 : dist P F = 1
variable h6 : dist P A = Real.sqrt 3 

-- We need to prove that the equation of C is as follows:
theorem ellipse_equation (h₀ : a > b) (h₁ : b > 1) (h₂ : a = 2) (h₃ : b = Real.sqrt 3) : 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) := 
by
  -- Equation of the ellipse using the given lengths
  sorry

-- We need to prove the range for |OQ| 
theorem range_OQ (P : ℝ × ℝ) (A F O : ℝ × ℝ) (h4: (angle P O A = Real.pi / 3)) (h5: dist P F = 1) (h6: dist P A = Real.sqrt 3) 
  (H : P.1^2 + P.2^2 = 1): (range_OQ (A F P O) = [3, 4]) :=
by
  -- Derivation based on given tangent conditions and distance formula
  sorry

end ellipse_equation_range_OQ_l187_187351


namespace total_people_on_bus_l187_187861

-- Definitions of the conditions
def num_boys : ℕ := 50
def num_girls : ℕ := (2 / 5 : ℚ) * num_boys
def num_students : ℕ := num_boys + num_girls.toNat
def num_non_students : ℕ := 3 -- Mr. Gordon, the driver, and the assistant

-- The theorem to be proven
theorem total_people_on_bus : num_students + num_non_students = 123 := by
  sorry

end total_people_on_bus_l187_187861


namespace find_second_number_l187_187895

theorem find_second_number (x : ℝ) 
    (h : (14 + x + 53) / 3 = (21 + 47 + 22) / 3 + 3) : 
    x = 32 := 
by 
    sorry

end find_second_number_l187_187895


namespace _l187_187883

noncomputable theorem no_2023_liars_in_circle (n : ℕ) : (n % 2 = 1) → (∀ (k l : ℕ), k ≠ l → ∃ (H : set (ℕ × ℕ)), 
  (∃ knights : finset ℕ, 
    (∀ knight ∈ knights, knight_truth knight H) ∧ 
    (∀ liar ∈ (finset.range n) \ knights, liar_lie liar H)) → 
  n = 2023) → false :=
by
  sorry

-- Definitions being used
def knight_truth (k : ℕ) (H : set (ℕ × ℕ)) : Prop :=
  ∃ (n₁ n₂ : ℕ), (k, n₁) ∈ H ∧ (k, n₂) ∈ H ∧ n₁ ≠ n₂ ∧ (n₁ < k ↔ k < n₂)

def liar_lie (l : ℕ) (H : set (ℕ × ℕ)) : Prop :=
  ∃ (n₁ n₂ : ℕ), (l, n₁) ∈ H ∧ (l, n₂) ∈ H ∧ n₁ ≠ n₂ ∧ ¬(n₁ < l ↔ l < n₂)

end _l187_187883


namespace y2_odd_y4_odd_l187_187189

variables {f : ℝ → ℝ}  -- declaring the function f

-- Definitions of the given functions
def y1 (x : ℝ) := -|f x|
def y2 (x : ℝ) := x * f (x^2)
def y3 (x : ℝ) := -f (-x)
def y4 (x : ℝ) := f x - f (-x)

-- Test case for y2 being odd
theorem y2_odd : ∀ x : ℝ, y2 (-x) = -y2 x := 
by 
  sorry

-- Test case for y4 being odd
theorem y4_odd : ∀ x : ℝ, y4 (-x) = -y4 x := 
by 
  sorry

end y2_odd_y4_odd_l187_187189


namespace language_spoken_by_at_least_three_scientists_l187_187596

/-- 
There are 9 scientists, none of them speaks more than three languages, and among any 
three of them, there are at least two who speak a common language. Prove that there 
is a language spoken by at least three of the scientists.
-/
theorem language_spoken_by_at_least_three_scientists
  (S : Finset ℕ) (languages : ℕ → Finset ℕ)
  (h_count : S.card = 9)
  (h_max_languages : ∀ s ∈ S, (languages s).card ≤ 3)
  (h_pairwise : ∀ (s1 s2 s3 ∈ S), (↑s1 ≠ ↑s2) → (↑s2 ≠ ↑s3) → (↑s1 ≠ ↑s3) → 
    (¬ (languages s1) ∩ (languages s2) = ∅ ∨ ¬ (languages s2) ∩ (languages s3) = ∅ ∨ 
      ¬ (languages s1) ∩ (languages s3) = ∅)) :
  ∃ l, 3 ≤ S.count (∈ λ s, l ∈ languages s) :=
sorry  -- Proof is to be provided

end language_spoken_by_at_least_three_scientists_l187_187596


namespace number_of_ways_correct_l187_187956

noncomputable def number_of_ways_to_choose_subsets (S : finset ℕ) (n : ℕ) : ℕ :=
if h : S.card = n then sorry else 0

theorem number_of_ways_correct : 
  number_of_ways_to_choose_subsets ({1, 2, 3, 4, 5, 6} : finset ℕ) 6 = 80 :=
sorry

end number_of_ways_correct_l187_187956


namespace number_of_positive_integer_solutions_l187_187982

theorem number_of_positive_integer_solutions :
  ∃ n : ℕ, n = 84 ∧ (∀ x y z t : ℕ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < t ∧ x + y + z + t = 10 → true) :=
sorry

end number_of_positive_integer_solutions_l187_187982


namespace final_answer_l187_187836

def is_valid_5_digit_number (n : ℕ) : Prop :=
  n >= 10000 ∧ n <= 99999

def div_by_80_quot_rem (n : ℕ) : ℕ × ℕ :=
  (n / 80, n % 80)

def condition (q r : ℕ) : Prop :=
  (q + 2 * r) % 7 = 0

noncomputable def count_valid_n : ℕ :=
  let valid_q_range := list.range' 125 1125 -- 125 to 1249 inclusive
  val valid_q_r_pairs :=
    valid_q_range.map (λ q, ((list.range' 0 80).filter (λ r, condition q r)).length)
  valid_q_r_pairs.sum

theorem final_answer : count_valid_n = 13596 :=
  sorry

end final_answer_l187_187836


namespace intersection_A_B_l187_187818

def A : Set ℤ := {x | x ≥ -1 ∧ x ≤ 2}
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x}

theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end intersection_A_B_l187_187818


namespace inverse_proportion_decreasing_l187_187868

theorem inverse_proportion_decreasing (k : ℝ) (x : ℝ) (hx : x > 0) :
  (y = (k - 1) / x) → (k > 1) :=
by
  sorry

end inverse_proportion_decreasing_l187_187868


namespace total_tissues_l187_187481

-- define the number of students in each group
def g1 : Nat := 9
def g2 : Nat := 10
def g3 : Nat := 11

-- define the number of tissues per mini tissue box
def t : Nat := 40

-- state the main theorem
theorem total_tissues : (g1 + g2 + g3) * t = 1200 := by
  sorry

end total_tissues_l187_187481


namespace sequence_sum_l187_187937

theorem sequence_sum :
  ∃ a : ℕ → ℝ, 
    a 1 = 1 ∧ 
    (∃ λ > 0, 
      (∀ n : ℕ, a n * a (n + 1) = λ * 2^n) ∧
      (∀ k : ℕ, 2 * a (2 * k) = a (2 * k - 1) + a (2 * k + 1)) ∧
      ∑ n in (range 20), a n = 5115 / 2) := sorry

end sequence_sum_l187_187937


namespace gcd_largest_divisor_l187_187024

theorem gcd_largest_divisor (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 1008) : 
  ∃ d, nat.gcd a b = d ∧ d = 504 :=
begin
  sorry
end

end gcd_largest_divisor_l187_187024


namespace evaluate_64_pow_fifth_sixth_l187_187222

theorem evaluate_64_pow_fifth_sixth : 64 ^ (5 / 6) = 32 := by
  have h : 64 = 2 ^ 6 := by sorry
  calc 64 ^ (5 / 6) = (2 ^ 6) ^ (5 / 6) : by rw [h]
              ...   = 2 ^ (6 * (5 / 6))  : by sorry
              ...   = 2 ^ 5              : by sorry
              ...   = 32                 : by sorry

end evaluate_64_pow_fifth_sixth_l187_187222


namespace complex_quadrant_l187_187756

def quadrant (z : ℂ) : String :=
  if z.re > 0 ∧ z.im > 0 then "First"
  else if z.re < 0 ∧ z.im > 0 then "Second"
  else if z.re < 0 ∧ z.im < 0 then "Third"
  else if z.re > 0 ∧ z.im < 0 then "Fourth"
  else "Origin"

theorem complex_quadrant :
  quadrant ((2 : ℂ) - (1 : ℂ)*complex.I)^2 = "Fourth" :=
by sorry

end complex_quadrant_l187_187756


namespace sum_squares_wins_equals_sum_squares_losses_l187_187512

theorem sum_squares_wins_equals_sum_squares_losses (n : ℕ) (h1 : n > 1)
  (w l : Fin n → ℕ)
  (h2 : ∀ i, w i + l i = n - 1)
  (h3 : ∑ i, w i = ∑ i, l i) :
  ∑ i, (w i)^2 = ∑ i, (l i)^2 := 
by 
  sorry

end sum_squares_wins_equals_sum_squares_losses_l187_187512


namespace speed_of_rest_distance_l187_187113

theorem speed_of_rest_distance (D V : ℝ) (h1 : D = 26.67)
                                (h2 : (D / 2) / 5 + (D / 2) / V = 6) : 
  V = 20 :=
by
  sorry

end speed_of_rest_distance_l187_187113


namespace functional_inequality_solution_l187_187431

theorem functional_inequality_solution (n : ℕ) (hn : n > 0) : 
  (∀ x ∈ D, x^n + (1 - x)^n > 1) ↔ (n > 1 ∧ D = set.Iio 0 ∪ set.Ioi 1) :=
by
  sorry

end functional_inequality_solution_l187_187431


namespace zinc_weight_correct_l187_187542

def zinc_and_copper_mixture_weight (zinc_ratio : ℕ) (copper_ratio : ℕ) (total_weight : ℚ) : ℚ :=
  let total_parts := zinc_ratio + copper_ratio in
  let weight_per_part := total_weight / total_parts in
  weight_per_part * zinc_ratio

theorem zinc_weight_correct :
  zinc_and_copper_mixture_weight 9 11 74 = 33.3 := 
sorry

end zinc_weight_correct_l187_187542


namespace shortest_path_AH_l187_187933

theorem shortest_path_AH 
  (d : ℝ) (diameter_eq : d = 100)
  (R : ℝ) (radius_eq : R = d / 2)
  (A B : ℝ) (AB_eq : A = -R ∧ B = R)
  (C : ℝ) (C_eq : C = sqrt (3) * R)
  (H : ℝ) (H_eq : H = (B + C) / 2) :
  sqrt (R^2 + ((B - H)*2 / 2)^2) = 50 * sqrt 5 := 
by
  sorry

end shortest_path_AH_l187_187933


namespace max_value_amc_am_mc_ca_l187_187788

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end max_value_amc_am_mc_ca_l187_187788


namespace smallest_n_l187_187062

theorem smallest_n (n : ℕ) : (∃ (m1 m2 : ℕ), 4 * n = m1^2 ∧ 5 * n = m2^3) ↔ n = 500 := 
begin
  sorry
end

end smallest_n_l187_187062


namespace smallest_n_for_perfect_square_and_cube_l187_187058

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, (∃ a : ℕ, 4 * n = a^2) ∧ (∃ b : ℕ, 5 * n = b^3) ∧ n = 125 :=
by
  sorry

end smallest_n_for_perfect_square_and_cube_l187_187058


namespace correct_removal_of_parentheses_C_incorrect_removal_of_parentheses_A_incorrect_removal_of_parentheses_B_incorrect_removal_of_parentheses_D_l187_187537

theorem correct_removal_of_parentheses_C (a : ℝ) :
    -(2 * a - 1) = -2 * a + 1 :=
by sorry

theorem incorrect_removal_of_parentheses_A (a : ℝ) :
    -(7 * a - 5) ≠ -7 * a - 5 :=
by sorry

theorem incorrect_removal_of_parentheses_B (a : ℝ) :
    -(-1 / 2 * a + 2) ≠ -1 / 2 * a - 2 :=
by sorry

theorem incorrect_removal_of_parentheses_D (a : ℝ) :
    -(-3 * a + 2) ≠ 3 * a + 2 :=
by sorry

end correct_removal_of_parentheses_C_incorrect_removal_of_parentheses_A_incorrect_removal_of_parentheses_B_incorrect_removal_of_parentheses_D_l187_187537


namespace exists_eight_consecutive_nonrepresentable_integers_l187_187879

noncomputable def f (x y : ℤ) : ℤ := 7 * x^2 + 9 * x * y - 5 * y^2

theorem exists_eight_consecutive_nonrepresentable_integers :
  ∃ (n : ℤ), ∀ (x y : ℤ), (n ≤ 11) → (19 ≤ n + 7) → ∀ k ∈ set.Icc n (n + 7), f x y ≠ k ∧ f x y ≠ -k :=
by sorry

end exists_eight_consecutive_nonrepresentable_integers_l187_187879


namespace lime_bottom_means_magenta_top_l187_187465

-- Define the colors as an enumeration for clarity
inductive Color
| Purple : Color
| Cyan : Color
| Magenta : Color
| Lime : Color
| Silver : Color
| Black : Color

open Color

-- Define the function representing the question
def opposite_top_face_given_bottom (bottom : Color) : Color :=
  match bottom with
  | Lime => Magenta
  | _ => Lime  -- For simplicity, we're only handling the Lime case as specified

-- State the theorem
theorem lime_bottom_means_magenta_top : 
  opposite_top_face_given_bottom Lime = Magenta :=
by
  -- This theorem states exactly what we need: if Lime is the bottom face, then Magenta is the top face.
  sorry

end lime_bottom_means_magenta_top_l187_187465


namespace smallest_n_solution_l187_187532

theorem smallest_n_solution : ∃ n : ℕ, 3 * n ≡ 2412 [MOD 30] ∧ n = 4 :=
by
  use 4
  split
  · norm_num
  · reflexivity

end smallest_n_solution_l187_187532


namespace evaluate_64_pow_5_div_6_l187_187223

theorem evaluate_64_pow_5_div_6 : (64 : ℝ)^(5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ)^6 := by norm_num
  rw [← h1]
  have h2 : ((2 : ℝ)^6)^(5 / 6) = (2 : ℝ)^(6 * (5 / 6)) := by rw [Real.rpow_mul]
  rw [h2]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187223


namespace prob_win_all_6_games_prob_win_exactly_5_out_of_6_games_l187_187893

noncomputable def prob_win_single_game : ℚ := 7 / 10
noncomputable def prob_lose_single_game : ℚ := 3 / 10

theorem prob_win_all_6_games : (prob_win_single_game ^ 6) = 117649 / 1000000 :=
by
  sorry

theorem prob_win_exactly_5_out_of_6_games : (6 * (prob_win_single_game ^ 5) * prob_lose_single_game) = 302526 / 1000000 :=
by
  sorry

end prob_win_all_6_games_prob_win_exactly_5_out_of_6_games_l187_187893


namespace bounded_sequence_l187_187294

def p : ℤ → ℤ
| 0        := ∞
| (1)      := 1
| (-1)     := 1
| (n - a) := -- Implement proper definition for greatest prime divisor of m
  if n = 2 then 2 else
    let f := factors n in list.max' (filter (λ x, x.prime) f)

noncomputable def f (n : ℤ) : ℤ := -- Implement proper polynomial definition

theorem bounded_sequence (f : ℤ → ℤ) (hf : ∀ n : ℤ, f n ≠ 0)
  : (∃ C : ℤ, ∀ n, p (f (n^2)) - 2n ≤ C) ↔ 
    ∃ c : ℤ, ∀ n : ℤ, (f n).is_product_of_linear_factors (λ k, (4 * n - a k ^ 2)) := sorry

end bounded_sequence_l187_187294


namespace general_formula_and_sum_l187_187673

theorem general_formula_and_sum (a₁ q : ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) (S : ℕ → ℝ)
  (h1 : a 3 = 3 / 2)
  (h2 : S 3 = 9 / 2)
  (hs : ∀ n, S n = a₁ * (1 - q ^ n) / (1 - q))
  (ha : ∀ n, a n = a₁ * q ^ (n - 1))
  (hb : ∀ n, b n = log 2 (6 / a (2 * n + 1)))
  (h_inc : ∀ n, b n < b (n + 1))
  (hc : ∀ n, c n = 1 / (b n * b (n + 1))) :
  (∀ n, a n = if q = 1 then 3 / 2 else 6 * (-1 / 2) ^ (n - 1)) ∧
  (∀ n, (finset.range n).sum c = n / (4 * (n + 1))) :=
by
  sorry

end general_formula_and_sum_l187_187673


namespace total_seats_in_movie_theater_l187_187947

theorem total_seats_in_movie_theater (sections : ℕ) (seats_per_section : ℕ) (h1 : sections = 9) (h2 : seats_per_section = 30) : sections * seats_per_section = 270 :=
by
  rw [h1, h2]
  norm_num

end total_seats_in_movie_theater_l187_187947


namespace original_savings_l187_187853

variable {S : ℝ} (tv_cost : ℝ) (furniture_fraction : ℝ) (tv_fraction : ℝ)

-- Conditions
def condition_furniture : Prop := furniture_fraction = 3 / 4
def condition_tv : Prop := tv_fraction = 1 / 4
def condition_tv_cost : Prop := tv_cost = 250
def condition_savings : Prop := tv_fraction * S = tv_cost

-- Proposition to prove
theorem original_savings (h1 : condition_furniture) (h2 : condition_tv) (h3 : condition_tv_cost) (h4 : condition_savings) : S = 1000 :=
sorry

end original_savings_l187_187853


namespace price_of_sugar_l187_187002

-- Define the variables and conditions
variables (S L : ℝ)

-- Define the conditions based on the problem statement
def cond1 : Prop := 2 * S + 5 * L = 5.50
def cond2 : Prop := 3 * S + L = 5

-- Theorem to prove the price of a kilogram of sugar is $1.50
theorem price_of_sugar (h1 : cond1) (h2 : cond2) : S = 1.50 :=
sorry

end price_of_sugar_l187_187002


namespace rain_at_least_once_prob_l187_187496

theorem rain_at_least_once_prob (p : ℚ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 :=
by {
  -- Implementation of Lean code is not required as per instructions.
  sorry
}

end rain_at_least_once_prob_l187_187496


namespace find_a_b_max_profit_allocation_l187_187095

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (a * Real.log x) / x + 5 / x - b

theorem find_a_b :
  (∃ (a b : ℝ), f 1 a b = 5 ∧ f 10 a b = 16.515) :=
sorry

noncomputable def g (x : ℝ) := 2 * Real.sqrt x / x

noncomputable def profit (x : ℝ) := x * (5 * Real.log x / x + 5 / x) + (50 - x) * (2 * Real.sqrt (50 - x) / (50 - x))

theorem max_profit_allocation :
  (∃ (x : ℝ), 10 ≤ x ∧ x ≤ 40 ∧ ∀ y, (10 ≤ y ∧ y ≤ 40) → profit x ≥ profit y)
  ∧ profit 25 = 31.09 :=
sorry

end find_a_b_max_profit_allocation_l187_187095


namespace cooks_choice_l187_187565

theorem cooks_choice (n k : ℕ) (A B : Fin n) (hA : A.val < n) (hB : B.val < n) :
  n = 10 →
  k = 3 →
  (∀ (S : Finset (Fin n)), S.card = k → A ∈ S → B ∉ S) → 
  Finset.card (Finset.filter (λ S, A ∈ S ∧ B ∉ S) (Finset.powerset_len k (Finset.univ : Finset (Fin n)))) = 112 :=
by
  sorry

end cooks_choice_l187_187565


namespace nikes_sold_l187_187584

theorem nikes_sold (x : ℕ) : 
  let adidas_price := 45
  let nike_price := 60
  let reebok_price := 35
  let adidas_count := 6
  let reebok_count := 9
  let quota := 1000
  let surplus := 65 in
  quotient := adidas_count * adidas_price + reebok_count * reebok_price + x * nike_price
  quota_total := quota + surplus in
  quotient = quota_total → x = 8 :=
by
  intros adidas_price nike_price reebok_price adidas_count reebok_count quota surplus
  sorry

end nikes_sold_l187_187584


namespace mean_transformed_l187_187347

variable (n : ℕ) (x : Fin n → ℝ)

-- Condition: Mean of the original sample data is 10
def mean_original := (∑ i, x i) / n = 10

-- Goal: Prove that the mean of the transformed data is 29
theorem mean_transformed (h : mean_original n x) : 
  (∑ i, (3 * x i - 1)) / n = 29 := 
by
  sorry

end mean_transformed_l187_187347


namespace tan_30_deg_l187_187167

theorem tan_30_deg : 
  let θ := 30 * (Float.pi / 180) in  -- Conversion from degrees to radians
  Float.sin θ = 1 / 2 ∧ Float.cos θ = Float.sqrt 3 / 2 →
  Float.tan θ = Float.sqrt 3 / 3 := by
  intro h
  sorry

end tan_30_deg_l187_187167


namespace triangle_perimeter_l187_187927

-- Definitions for the conditions
def side_length1 : ℕ := 3
def side_length2 : ℕ := 6
def equation (x : ℤ) := x^2 - 6 * x + 8 = 0

-- Perimeter calculation given the sides form a triangle
theorem triangle_perimeter (x : ℤ) (h₁ : equation x) (h₂ : 3 + 6 > x) (h₃ : 3 + x > 6) (h₄ : 6 + x > 3) :
  3 + 6 + x = 13 :=
by sorry

end triangle_perimeter_l187_187927


namespace remainder_of_sum_of_integers_mod_15_l187_187078

theorem remainder_of_sum_of_integers_mod_15 (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end remainder_of_sum_of_integers_mod_15_l187_187078


namespace part_a_l187_187888

noncomputable def sequence (α : ℝ) (n : ℕ) : ℕ := by sorry

noncomputable def golden_prime (α : ℝ) (n : ℕ) : ℕ := by sorry

theorem part_a (α : ℝ) (q_n : ℕ) (n : ℕ)
  (h1 : α = 1.5)
  (h2 : ∀ n : ℕ, sequence α n ≤ n^α)
  (h3 : ∀ m : ℕ, ∃ a ∈ sequence α m, ∃ (q : ℕ), prime q ∧ q ∣ a)
  (h4 : golden_prime α n = q_n)
  : q_n ≤ 35^n := by sorry

end part_a_l187_187888


namespace triangle_right_angle_maximum_area_triangle_l187_187328

-- Definitions from conditions in (a)
variable {A B C : ℝ}
variable {a b c : ℝ}
variable {α β γ : ℝ}

-- Definition: The perimeter of triangle ABC is 1
def perimeter_triangle_ABC_is_one : Prop :=
  a + b + c = 1

-- Definition: Trigonometric condition
def trigonometric_condition : Prop :=
  Real.sin (2*A) + Real.sin (2*B) = 4 * Real.sin A * Real.sin B

-- Question 1: \( \triangle ABC \) is a right triangle.
theorem triangle_right_angle (h1 : perimeter_triangle_ABC_is_one) (h2 : trigonometric_condition) :
  A + B + C = π ∧ (A = π / 2 ∨ B = π / 2 ∨ C = π / 2) :=
  sorry  -- Proof to be provided

-- Question 2: Maximum area of \( \triangle ABC \)
theorem maximum_area_triangle (h1 : perimeter_triangle_ABC_is_one) (h3 : A = π / 2 ∨ B = π / 2 ∨ C = π / 2) :
  ∃ max_area, 
  max_area = (3 - 2 * Real.sqrt 2) / 4 ∧
  (∀ area, area ≤ max_area) :=
  sorry  -- Proof to be provided

end triangle_right_angle_maximum_area_triangle_l187_187328


namespace reciprocal_of_2022_l187_187008

noncomputable def reciprocal (x : ℝ) := 1 / x

theorem reciprocal_of_2022 : reciprocal 2022 = 1 / 2022 :=
by
  -- Define reciprocal
  sorry

end reciprocal_of_2022_l187_187008


namespace win_sector_area_l187_187997

-- Define the conditions
def radius : ℝ := 12
def win_probability : ℝ := 1 / 3
def total_area : ℝ := Real.pi * radius ^ 2

-- State the theorem
theorem win_sector_area : 
  (win_probability * total_area) = 48 * Real.pi := 
by 
  sorry

end win_sector_area_l187_187997


namespace number_of_ways_to_pick_three_cards_l187_187573

-- Define the conditions of the problem
def num_cards : ℕ := 60
def cards_per_suit : ℕ := 12
def num_suits : ℕ := 5
def order_matters (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0 else ∏ i in (list.range' (n - k + 1) k), i + (n - k)

-- State the problem as a theorem
theorem number_of_ways_to_pick_three_cards :
  order_matters num_cards 3 = 205320 :=
by 
  -- The proof is skipped
  sorry

end number_of_ways_to_pick_three_cards_l187_187573


namespace tangent_line_at_1_l187_187917

noncomputable def f (x : ℝ) : ℝ := x^4 - 2 * x^3

theorem tangent_line_at_1 :
  let p := (1 : ℝ, f 1)
  in ∃ m c : ℝ, (∀ x : ℝ, y : ℝ, y = m * x + c ↔ y = -2 * x + 1) ∧ ∀ x : ℝ, f x = x^4 - 2 * x^3 :=
sorry

end tangent_line_at_1_l187_187917


namespace class_duration_l187_187098

theorem class_duration (h1 : 8 * 60 + 30 = 510) (h2 : 9 * 60 + 5 = 545) : (545 - 510 = 35) :=
by
  sorry

end class_duration_l187_187098


namespace min_val_proof_l187_187286

noncomputable def min_expression_value : ℝ :=
  let expr (x : ℝ) := 27^x - 6 * 3^x + 10
  in -4 * Real.sqrt 2 + 10

theorem min_val_proof : 
  ∀ x : ℝ, ∃ y : ℝ, y = 3^x ∧ expr(y) = -4 * real.sqrt 2 + 10 := 
  sorry

end min_val_proof_l187_187286


namespace y_intercept_correct_four_units_left_of_x_intercept_l187_187501

-- Define the slope and x-intercept point
def slope : ℝ := -3
def x_intercept : ℝ × ℝ := (8, 0)

-- Proof for y-intercept of the line
theorem y_intercept_correct :
  let x := 0 in let y := -3 * x + 24 in (x, y) = (0, 24) :=
by
  sorry

-- Proof for coordinates of the point 4 units left from the x-intercept
theorem four_units_left_of_x_intercept :
  let x := 8 - 4 in let y := -3 * x + 24 in (x, y) = (4, 12) := 
by 
  sorry

end y_intercept_correct_four_units_left_of_x_intercept_l187_187501


namespace max_value_ac_bd_ca_db_l187_187001

theorem max_value_ac_bd_ca_db (a b c d : ℕ) (h : {a, b, c, d} = {2, 4, 6, 8}) : 
  (a * c + b * d + c * a + d * b) ≤ 100 :=
by {
  sorry
}

end max_value_ac_bd_ca_db_l187_187001


namespace probability_of_nana_l187_187028

open ProbabilityTheory

noncomputable def card_set : set (char) := { 'a', 'a', 'a', 'n', 'n', 'x' }

def drawn_cards (set : set (char)) : list char := ['n', 'a', 'n', 'a']

-- Define the probability of each event
def event_A : ℝ := 2 / 6
def event_B_given_A : ℝ := 3 / 5
def event_C_given_AB : ℝ := 1 / 4
def event_D_given_ABC : ℝ := 2 / 3

-- Define the chain rule for conditional probabilities
def probability_nana : ℝ := event_A * event_B_given_A * event_C_given_AB * event_D_given_ABC

theorem probability_of_nana :
  probability_nana = 1 / 30 :=
by 
  -- This is a placeholder for the proof
  sorry

end probability_of_nana_l187_187028


namespace line_perpendicular_through_p0_l187_187476

-- Definitions based on conditions
variables {A B C x0 y0 : ℝ}

-- Definition of the given line and point
def given_line (x y : ℝ) : Prop := A * x + B * y + C = 0
def point_p0 : Prop := true -- Point P0 (x0, y0) is assumed to exist

-- Statement to prove
theorem line_perpendicular_through_p0 :
  (∀ x y : ℝ, given_line x y → B * x - A * y - B * x0 + A * y0 = 0) :=
begin
  sorry
end

end line_perpendicular_through_p0_l187_187476


namespace wicket_keeper_older_than_captain_l187_187390

variables (captain_age : ℕ) (team_avg_age : ℕ) (num_players : ℕ) (remaining_avg_age : ℕ)

def x_older_than_captain (captain_age team_avg_age num_players remaining_avg_age : ℕ) : ℕ :=
  team_avg_age * num_players - remaining_avg_age * (num_players - 2) - 2 * captain_age

theorem wicket_keeper_older_than_captain 
  (captain_age : ℕ) (team_avg_age : ℕ) (num_players : ℕ) (remaining_avg_age : ℕ) 
  (h1 : captain_age = 25) (h2 : team_avg_age = 23) (h3 : num_players = 11) (h4 : remaining_avg_age = 22) :
  x_older_than_captain captain_age team_avg_age num_players remaining_avg_age = 5 :=
by sorry

end wicket_keeper_older_than_captain_l187_187390


namespace find_MN_sum_l187_187371

noncomputable def M : ℝ := sorry -- Placeholder for the actual non-zero solution M
noncomputable def N : ℝ := M ^ 2

theorem find_MN_sum :
  (M^2 = N) ∧ (Real.log N / Real.log M = Real.log M / Real.log N) ∧ (M ≠ N) ∧ (M ≠ 1) ∧ (N ≠ 1) → (M + N = 6) :=
by
  intros h
  exact sorry -- Will be replaced by the actual proof


end find_MN_sum_l187_187371


namespace terms_to_add_l187_187527

theorem terms_to_add (k : ℕ) (h : k > 1) : 
  (2^k - 1) + 2^k - (2^k - 1) = 2^k 
    ∧ 1 + ∑ i in range k, 1/(2^i) < k -> 1 + ∑ i in range (k+1), 1/(2^i) < k+1 :=
begin
  sorry
end

end terms_to_add_l187_187527


namespace value_of_k_l187_187954

theorem value_of_k : 
  (k : ℝ) 
  (h1 : (P : ℝ × ℝ) (P = (10, 6))) 
  (h2 : (S : ℝ × ℝ) (S = (0, k))) 
  (h3 : (O : ℝ × ℝ) (O = (0, 0)))
  (QR : ℝ) 
  (h4 : QR = 4)
  (h5 : dist O P = sqrt 136)
  (h6 : dist O S = sqrt 136 - 4) 
  : k = 2 * sqrt 34 - 4 := 
sorry

end value_of_k_l187_187954


namespace math_problem_statement_l187_187763

-- Geometry setup
variable {A B C P Q S R D : Type}
variable [IncidenceGeometry A B C]

-- Definitions based on the given conditions
def is_on_side_AB (P : A) (B : A) := P ∈ line A B  -- P is on line segment AB
def is_on_side_AC (Q : A) (C : A) := Q ∈ line A C  -- Q is on line segment AC
def angle_APC := angle A P C = 45
def angle_AQB := angle A Q B = 45
def is_perpendicular (P : A) (AB : line A B) := is_perp P AB -- P is perpendicular to AB
def intersects (P : A) (BQ : line B Q) := intersects P BQ -- intersects line BQ at S
def is_altitude (D : A) := is_alt AD D BC -- altitude from A to BC

-- Goal: Prove concurrency of PS, AD, QR and parallelism of SR || BC
noncomputable def concurrency_and_parallelism : Prop :=
  concurrency (line PS AD QR) ∧ parallel (line SR BC)

theorem math_problem_statement :
  (∀ (A B C P Q S R D : Type) [IncidenceGeometry A B C],
  is_on_side_AB P B →
  is_on_side_AC Q C →
  angle_APC P C →
  angle_AQB Q B →
  is_perpendicular P (line A B) →
  intersects P (line B Q) →
  intersects Q (line C P) →
  is_altitude D) →
  concurrency_and_parallelism :=
sorry

end math_problem_statement_l187_187763


namespace bruce_total_payment_l187_187598

def cost_of_grapes (quantity rate : ℕ) : ℕ := quantity * rate
def cost_of_mangoes (quantity rate : ℕ) : ℕ := quantity * rate

theorem bruce_total_payment : 
  cost_of_grapes 8 70 + cost_of_mangoes 11 55 = 1165 :=
by 
  sorry

end bruce_total_payment_l187_187598


namespace rain_at_least_once_l187_187492

noncomputable def rain_probability (day_prob : ℚ) (days : ℕ) : ℚ :=
  1 - (1 - day_prob)^days

theorem rain_at_least_once :
  ∀ (day_prob : ℚ) (days : ℕ),
    day_prob = 3/4 → days = 4 →
    rain_probability day_prob days = 255/256 :=
by
  intros day_prob days h1 h2
  sorry

end rain_at_least_once_l187_187492


namespace reflected_line_equation_l187_187107

noncomputable def circle_center : ℝ × ℝ := (3, 2)
noncomputable def circle_radius : ℝ := 1

def point_A : ℝ × ℝ := (-2, 3)

def point_symmetric_about_x_axis (pt : ℝ × ℝ) : ℝ × ℝ :=
  (pt.1, -pt.2)

def general_line (k : ℝ) (pt : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ p, p.2 = k * (p.1 - pt.1) + pt.2

lemma correct_slope (k : ℝ) :
  let d := |3 * k - 2 + 2 * k - 3| / (real.sqrt (k^2 + 1))
  in d = 1 → (k = 4 / 3 ∨ k = 3 / 4) :=
sorry

theorem reflected_line_equation :
  ∃ k : ℝ, (k = 4 / 3 ∨ k = 3 / 4) ∧ 
    ((general_line k (-2, -3)) (x, y) ↔ 4 * x - 3 * y - 1 = 0 ∨ 3 * x - 4 * y - 6 = 0) :=
sorry

end reflected_line_equation_l187_187107


namespace max_dot_product_l187_187712

open Real

theorem max_dot_product (θ : ℝ) :
  let a := (cos θ, sin θ)
  let b := (3, -4)
  ∃ θ, a.1 * b.1 + a.2 * b.2 = 5 :=
by
  sorry

end max_dot_product_l187_187712


namespace eccentricity_hyperbola_correctness_l187_187516

noncomputable def eccentricity_of_hyperbola (a b c : ℝ) (e : ℝ) : Prop :=
  ∀ (A B : ℝ × ℝ),
    (a > 0) ∧ (b > 0) ∧ 
    B = (2 * (a^2 / c) - c, 2 * (a * b / c)) ∧
    A = (a^2 / c, a * b / c) ∧ 
    (B.1^2 / a^2 - B.2^2 / b^2 = 1) ∧
    (c^2 = 5 * a^2) ∧ 
    (e = c / a) → 
    e = ℝ.sqrt 5

theorem eccentricity_hyperbola_correctness : 
  ∀ (a b c : ℝ), (a > 0) → (b > 0) → c^2 = 5 * a^2 → eccentricity_of_hyperbola a b c (ℝ.sqrt 5) :=
begin 
  intros,
  unfold eccentricity_of_hyperbola,
  intros,
  sorry
end

end eccentricity_hyperbola_correctness_l187_187516


namespace min_ratio_CD_to_AD_l187_187437

/-- 
  Given a right trapezoid ABCD with bases AB and CD, and right angles at vertices A and D.
  Additionally, the shorter diagonal BD is perpendicular to the side BC.
  We aim to prove that the minimum possible value for the ratio CD / AD  is 2.
-/
theorem min_ratio_CD_to_AD (α : ℝ) (h1 : 0 < α) (h2 : α < π/2) :
  let CD := λ BD : ℝ, BD / real.cos α,
      AD := λ BD : ℝ, BD * real.sin α,
      ratio := λ BD : ℝ, (CD BD) / (AD BD) in
  ∃ BD : ℝ, BD > 0 ∧ ratio BD ≥ 2 :=
begin
  use 1,  -- Using BD = 1 for simplification
  split,
  { linarith, },  -- Prove that 1 > 0
  { sorry }
end

end min_ratio_CD_to_AD_l187_187437


namespace largest_t_value_l187_187284

theorem largest_t_value : 
  ∃ t : ℝ, 
    (∃ s : ℝ, s > 0 ∧ t = 3 ∧
    ∀ u : ℝ, 
      (u = 3 →
        (15 * u^2 - 40 * u + 18) / (4 * u - 3) + 3 * u = 4 * u + 2 ∧
        u ≤ 3) ∧
      (u ≠ 3 → 
        (15 * u^2 - 40 * u + 18) / (4 * u - 3) + 3 * u = 4 * u + 2 → 
        u ≤ 3)) :=
sorry

end largest_t_value_l187_187284


namespace lauren_change_l187_187416

-- Define all the conditions stated
def meat_price_per_pound := 3.50
def meat_discount := 0.15
def meat_weight := 2.0
def buns_price := 1.50
def lettuce_price := 1.00
def tomato_price_per_pound := 2.00
def tomato_weight := 1.5
def onion_price_per_pound := 0.75
def onion_weight := 0.5
def pickles_price := 2.50
def pickles_coupon := 1.00
def potatoes_price := 4.00
def soda_price := 5.99
def soda_discount := 0.07
def sales_tax := 0.06
def payment := 50.00

-- Translate the correct answer into a Lean theorem
theorem lauren_change : 
  let total_cost := (meat_weight * meat_price_per_pound * (1 - meat_discount)) + 
                    buns_price + 
                    lettuce_price + 
                    (tomato_price_per_pound * tomato_weight) + 
                    (onion_price_per_pound * onion_weight).round + 
                    (pickles_price - pickles_coupon) + 
                    potatoes_price + 
                    (soda_price * (1 - soda_discount)) in
  let total_cost_with_tax := total_cost * (1 + sales_tax) in
  payment - total_cost_with_tax = 24.67 :=
by show 50.0 - ((2.0 * 3.50 * (1 - 0.15)) + 1.50 + 1.00 + (2.00 * 1.5) + ((0.75 * 0.5).round : ℝ) + (2.50 - 1.00) + 4.00 + (5.99 * (1 - 0.07))) * (1 + 0.06) = 24.67 sorry

end lauren_change_l187_187416


namespace arithmetic_sqrt_9_l187_187894

def arithmetic_sqrt (x : ℕ) : ℕ :=
  if h : 0 ≤ x then Nat.sqrt x else 0

theorem arithmetic_sqrt_9 : arithmetic_sqrt 9 = 3 :=
by {
  sorry
}

end arithmetic_sqrt_9_l187_187894


namespace A_is_guilty_l187_187093

-- Define the conditions
variables (A B C : Prop)  -- A, B, C are the propositions that represent the guilt of the individuals A, B, and C
variable  (car : Prop)    -- car represents the fact that the crime involved a car
variable  (C_never_alone : C → A)  -- C never commits a crime without A

-- Facts:
variables (crime_committed : A ∨ B ∨ C) -- the crime was committed by A, B, or C (or a combination)
variable  (B_knows_drive : B → car)     -- B knows how to drive

-- The proof goal: Show that A is guilty.
theorem A_is_guilty : A :=
sorry

end A_is_guilty_l187_187093


namespace intersection_in_second_quadrant_l187_187327

theorem intersection_in_second_quadrant (k : ℝ) 
    (h₀ : 0 < k) 
    (h₁ : k < 1 / 2) :
    let x := k / (k - 1)
    let y := (2 * k ^ 2 + k - 1) / (k ^ 2 - 1)
    in x < 0 ∧ y > 0 := 
by
    let x := k / (k - 1)
    let y := (2 * k ^ 2 + k - 1) / (k ^ 2 - 1)
    sorry

end intersection_in_second_quadrant_l187_187327


namespace tan_30_l187_187145

theorem tan_30 : Real.tan (Real.pi / 6) = Real.sqrt 3 / 3 := 
by 
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry
  have h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2 := by sorry
  calc
    Real.tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6) : Real.tan_eq_sin_div_cos _
    ... = (1 / 2) / (Real.sqrt 3 / 2) : by rw [h1, h2]
    ... = (1 / 2) * (2 / Real.sqrt 3) : by rw Div.div_eq_mul_inv
    ... = 1 / Real.sqrt 3 : by norm_num
    ... = Real.sqrt 3 / 3 : by rw [Div.inv_eq_inv, Mul.comm, Mul.assoc, Div.mul_inv_cancel (Real.sqrt_ne_zero _), one_div Real.sqrt 3, inv_mul_eq_div]

-- Additional necessary function apologies for the unproven theorems.
noncomputable def _root_.Real.sqrt (x:ℝ) : ℝ := sorry

noncomputable def _root_.Real.tan (x : ℝ) : ℝ :=
  (Real.sin x) / (Real.cos x)

#eval tan_30 -- check result

end tan_30_l187_187145


namespace power_modulo_one_l187_187531

theorem power_modulo_one :
  ∃ n : ℕ, 7 ^ n % 100 = 1 :=
by {
  use 4,
  sorry
}

end power_modulo_one_l187_187531


namespace P_on_S_center_S_on_HN_PE_equals_side_length_l187_187418

-- A regular dodecagon denoted by its vertices
variables (A B C D E F G H I L M N : Point)
-- The circle S passing through A and H with the same radius as the circumcircle of the dodecagon
variable (S : Circle)
-- The point P is the intersection of diagonals AF and DH
variable (P : Point)

-- Conditions
variables
  (is_regular_dodecagon : RegularDodecagon A B C D E F G H I L M N)
  (P_intersection_AF_DH : IntersectionPoint P A F D H)
  (circle_S : Circle S A H radius equals CircumcircleRadius A B C D E F G H I L M N)

-- Proof 1: P lies on S
theorem P_on_S : LiesOn P S :=
  sorry

-- Proof 2: Center of S lies on the diagonal HN
theorem center_S_on_HN : LiesOn (Center S) (Line H N) :=
  sorry

-- Proof 3: The length of PE equals the length of the side of the dodecagon
theorem PE_equals_side_length : Length (Segment P E) = SideLength (RegularDodecagon A B C D E F G H I L M N) :=
  sorry

end P_on_S_center_S_on_HN_PE_equals_side_length_l187_187418


namespace rain_at_least_once_l187_187495

theorem rain_at_least_once (p : ℚ) (h : p = 3/4) : 
    (1 - (1 - p)^4) = 255/256 :=
by
  sorry

end rain_at_least_once_l187_187495


namespace area_under_curve_l187_187471

theorem area_under_curve : 
  (∫ x in 0..1, (x^2 + 1) : ℝ) = 4 / 3 :=
by
  sorry

end area_under_curve_l187_187471


namespace smallest_n_for_perfect_square_and_cube_l187_187072

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, 0 < n ∧ (∃ a1 b1 : ℕ, 4 * n = a1 ^ 2 ∧ 5 * n = b1 ^ 3 ∧ n = 50) :=
begin
  use 50,
  split,
  { norm_num, },
  { use [10, 5],
    split,
    { norm_num, },
    { split, 
      { norm_num, },
      { refl, }, },
  },
  sorry
end

end smallest_n_for_perfect_square_and_cube_l187_187072


namespace minimize_third_side_l187_187708

theorem minimize_third_side (a b gamma : ℝ) (h : a + b = d) : (minimize_third_side a b gamma = a) := sorry

end minimize_third_side_l187_187708


namespace find_weight_of_b_l187_187896

theorem find_weight_of_b (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) : B = 31 :=
sorry

end find_weight_of_b_l187_187896


namespace max_value_of_q_l187_187816

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_q_l187_187816


namespace evaluate_64_pow_5_div_6_l187_187269

theorem evaluate_64_pow_5_div_6 : (64 : ℝ) ^ (5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ) ^ 6 := by norm_num
  have h2 : (64 : ℝ) ^ (5 / 6) = ((2 : ℝ) ^ 6) ^ (5 / 6) := by rw h1
  have h3 : ((2 : ℝ) ^ 6) ^ (5 / 6) = (2 : ℝ) ^ (6 * (5 / 6)) := by rw [Real.rpow_mul]
  have h4 : (2 : ℝ) ^ (6 * (5 / 6)) = (2 : ℝ) ^ 5 := by norm_num
  rw [h2, h3, h4]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187269


namespace remainder_of_sum_of_integers_mod_15_l187_187077

theorem remainder_of_sum_of_integers_mod_15 (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end remainder_of_sum_of_integers_mod_15_l187_187077


namespace complex_number_properties_l187_187300

open Complex

theorem complex_number_properties (z z1 : ℂ) (i : ℂ) 
  (hi : i^2 = -1) (hz1 : z1 = 1 - i) (hz : z * (2 / z1) = z1) : 
  abs z = 1 :=
by
  -- Further proof steps needed here
  sorry

end complex_number_properties_l187_187300


namespace S_320_eq_10_l187_187676

-- Definitions for the sequences and their properties
noncomputable def a (n : ℕ) : ℝ := 
  if n = 1 then 1 else if n = 2 then 2 else sqrt(3 * n - 2)

noncomputable def b (n : ℕ) : ℝ := 
  1 / (a n + a (n + 1))

-- Partial sum of the first n terms of b_n
noncomputable def S (n : ℕ) : ℝ := 
  ∑ i in finset.range n, b (i + 1)

-- The proof statement
theorem S_320_eq_10 : S 320 = 10 := 
  sorry

end S_320_eq_10_l187_187676


namespace bus_total_people_l187_187864

def number_of_boys : ℕ := 50
def additional_girls (b : ℕ) : ℕ := (2 * b) / 5
def number_of_girls (b : ℕ) : ℕ := b + additional_girls b
def total_people (b g : ℕ) : ℕ := b + g + 3  -- adding 3 for the driver, assistant, and teacher

theorem bus_total_people : total_people number_of_boys (number_of_girls number_of_boys) = 123 :=
by
  sorry

end bus_total_people_l187_187864


namespace liquid_ratio_l187_187556

-- Defining the initial state of the container.
def initial_volume : ℝ := 37.5
def removed_and_replaced_volume : ℝ := 15

-- Defining the process steps.
def fraction_remaining (total: ℝ) (removed: ℝ) : ℝ := (total - removed) / total
def final_volume_A (initial: ℝ) (removed: ℝ) : ℝ := (fraction_remaining initial removed)^2 * initial

-- The given problem and its conclusion as a theorem.
theorem liquid_ratio (initial_V : ℝ) (remove_replace_V : ℝ) 
  (h1 : initial_V = 37.5) (h2 : remove_replace_V = 15) :
  let final_A := final_volume_A initial_V remove_replace_V in
  let final_B := initial_V - final_A in
  final_A / final_B = 9 / 16 :=
by
  sorry

end liquid_ratio_l187_187556


namespace evaluate_64_pow_5_div_6_l187_187229

theorem evaluate_64_pow_5_div_6 : (64 : ℝ)^(5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ)^6 := by norm_num
  rw [← h1]
  have h2 : ((2 : ℝ)^6)^(5 / 6) = (2 : ℝ)^(6 * (5 / 6)) := by rw [Real.rpow_mul]
  rw [h2]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187229


namespace expression_simplifies_l187_187375

variable {x y : Real}

-- Conditions
def cond1 : Prop := x ≠ 0
def cond2 : Prop := y ≠ 0
def cond3 : Prop := 3 * x - y / 3 ≠ 0

-- Statement
theorem expression_simplifies 
  (h1 : cond1) 
  (h2 : cond2) 
  (h3 : cond3) 
  : (3 * x - y / 3)⁻¹ * ((3 * x)⁻¹ - (y / 3)⁻¹) = -(x * y)⁻¹ :=
by 
  sorry

end expression_simplifies_l187_187375


namespace jordan_has_11_oreos_l187_187770

-- Define the conditions
def jamesOreos (x : ℕ) : ℕ := 3 + 2 * x
def totalOreos (jordanOreos : ℕ) : ℕ := 36

-- Theorem stating the problem that Jordan has 11 Oreos given the conditions
theorem jordan_has_11_oreos (x : ℕ) (h1 : jamesOreos x + x = totalOreos x) : x = 11 :=
by
  sorry

end jordan_has_11_oreos_l187_187770


namespace required_amount_of_water_l187_187000

/-- 
Given:
- A solution of 12 ounces with 60% alcohol,
- A desired final concentration of 40% alcohol,

Prove:
- The required amount of water to add is 6 ounces.
-/
theorem required_amount_of_water 
    (original_volume : ℚ)
    (initial_concentration : ℚ)
    (desired_concentration : ℚ)
    (final_volume : ℚ)
    (amount_of_water : ℚ)
    (h1 : original_volume = 12)
    (h2 : initial_concentration = 0.6)
    (h3 : desired_concentration = 0.4)
    (h4 : final_volume = original_volume + amount_of_water)
    (h5 : amount_of_alcohol = original_volume * initial_concentration)
    (h6 : desired_amount_of_alcohol = final_volume * desired_concentration)
    (h7 : amount_of_alcohol = desired_amount_of_alcohol) : 
  amount_of_water = 6 := 
sorry

end required_amount_of_water_l187_187000


namespace trigonometric_identity_l187_187688

open Real

theorem trigonometric_identity (θ : ℝ) (h₁ : 0 < θ ∧ θ < π/2) (h₂ : cos θ = sqrt 10 / 10) :
  (cos (2 * θ) / (sin (2 * θ) + (cos θ)^2)) = -8 / 7 := 
sorry

end trigonometric_identity_l187_187688


namespace main_theorem_l187_187754

-- Given ellipse C1 with the condition on eccentricity
def ellipse1 (a b : ℝ) : Prop := b > 0 ∧ a > b ∧ ( ∃ e: ℝ, e = (Real.sqrt 2) / 2 ∧ e = (Real.sqrt (a^2 - b^2)) / a)

-- Given sum of perpendicular chords through the right focus F
def sum_of_perpendicular_chords (a b : ℝ) : Prop := 2 * a + (2 * b^2) / a = 6

-- Given points A and B on parabola C2 with perpendicular tangents
def points_on_parabola (A B : ℝ × ℝ) : Prop := 
  A.1^2 = 4 * A.2 ∧ B.1^2 = 4 * B.2 ∧ (∃ kA kB : ℝ, kA = A.1 / 2 ∧ kB = B.1 / 2 ∧ kA * kB = -1)

-- Maximum value of length of chord CD
def max_length_chord_CD (k : ℝ) : Prop := 
  ∃ CD_length : ℝ, CD_length = Real.sqrt (1 + k^2) * Real.sqrt (8 * (1 + 4 * k^2)) / (1 + 2 * k^2) ∧ CD_length ≤ 3

-- The main theorem to prove
theorem main_theorem :
  ∃ a b : ℝ, ellipse1 a b ∧ sum_of_perpendicular_chords a b ∧ a = 2 ∧ b = Real.sqrt 2 ∧
  (∀ A B : ℝ × ℝ, points_on_parabola A B → ∃ k : ℝ, max_length_chord_CD k → k = Real.sqrt 2 / 2 ∨ k = -Real.sqrt 2 / 2) :=
begin
  sorry,
end

end main_theorem_l187_187754


namespace jordan_Oreos_count_l187_187769

variable (J : ℕ)
variable (OreosTotal : ℕ)
variable (JamesOreos : ℕ)

axiom James_Oreos_condition : JamesOreos = 2 * J + 3
axiom Oreos_total_condition : J + JamesOreos = OreosTotal
axiom Oreos_total_value : OreosTotal = 36

theorem jordan_Oreos_count : J = 11 :=
by 
  unfold OreosTotal JamesOreos
  sorry

end jordan_Oreos_count_l187_187769


namespace gymnastics_problem_l187_187570

def arithmetic_sum (a n : ℕ) : ℕ :=
  (n * (2 * a + n - 1)) / 2

theorem gymnastics_problem 
  (1000_team_members : ℕ) 
  (n : ℕ) 
  (a : ℕ) 
  (hn : n > 16) 
  (hsum : arithmetic_sum a n = 1000_team_members) :
  n = 25 ∧ a = 28 := 
by
  let 1000_team_members := 1000
  let total_people := 1000_team_members
  have h1 : arithmetic_sum 28 25 = total_people, by sorry -- This is where the proof would occur
  have h2 : n = 25, by sorry -- Prove n
  have h3 : a = 28, by sorry -- Prove a
  exact ⟨h2, h3⟩

end gymnastics_problem_l187_187570


namespace darts_score_l187_187609

theorem darts_score (First_score Second_score Third_score Fourth_score : ℕ) 
  (h1 : First_score = 30)
  (h2 : Second_score = 38)
  (h3 : Third_score = 41)
  (h4 : First_score + Second_score = 2 * Fourth_score) :
  Fourth_score = 34 := 
  by 
  rw [h1, h2] at h4
  simp at h4
  assumption -- Since this conclusion follows mathematically from the simplified condition

end darts_score_l187_187609


namespace bus_total_people_l187_187862

def number_of_boys : ℕ := 50
def additional_girls (b : ℕ) : ℕ := (2 * b) / 5
def number_of_girls (b : ℕ) : ℕ := b + additional_girls b
def total_people (b g : ℕ) : ℕ := b + g + 3  -- adding 3 for the driver, assistant, and teacher

theorem bus_total_people : total_people number_of_boys (number_of_girls number_of_boys) = 123 :=
by
  sorry

end bus_total_people_l187_187862


namespace evaluate_64_pow_5_div_6_l187_187226

theorem evaluate_64_pow_5_div_6 : (64 : ℝ)^(5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ)^6 := by norm_num
  rw [← h1]
  have h2 : ((2 : ℝ)^6)^(5 / 6) = (2 : ℝ)^(6 * (5 / 6)) := by rw [Real.rpow_mul]
  rw [h2]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187226


namespace difference_of_squares_is_149_l187_187141

-- Definitions of the conditions
def are_consecutive (n m : ℤ) : Prop := m = n + 1
def sum_less_than_150 (n : ℤ) : Prop := (n + (n + 1)) < 150

-- The difference of their squares
def difference_of_squares (n m : ℤ) : ℤ := (m * m) - (n * n)

-- Stating the problem where the answer expected is 149
theorem difference_of_squares_is_149 :
  ∀ n : ℤ, 
  ∀ m : ℤ,
  are_consecutive n m →
  sum_less_than_150 n →
  difference_of_squares n m = 149 :=
by
  sorry

end difference_of_squares_is_149_l187_187141


namespace sum_approx_289_857_l187_187834

def f (n : ℕ) : ℤ :=
  let root := (n : ℚ)^(1 / 3 : ℚ)
  if root.fract = 0.5 then root.floor.toNat else root.round.toNat

theorem sum_approx_289_857 :
  (∑ k in Finset.range 2500 | k > 0, (1 / (f k : ℚ))) ≈ 289.857 := by
  sorry

end sum_approx_289_857_l187_187834


namespace number_of_possible_integral_values_l187_187419

-- Definitions related to the conditions
variable {A B C D E F : Type}
variables {AC BC : ℝ}
variable (AB : ℝ)
variable (BC : ℝ)
variable (AD : ℝ)

-- Conditions as hypotheses
axiom angle_bisector_theorem : BD = x ∧ DC = 2x
axiom line_EF_parallel_AD : EF ∥ AD
axiom EF_divides_triangle_ABC_into_three_parts_of_equal_area : 
  ∀ area_ABC : ℝ, area_ABC / 3 = area_ABD
axiom triangle_inequality_1 : 7 + 14 > BC
axiom triangle_inequality_2 : 7 + BC > 14
axiom triangle_inequality_3 : 14 + BC > 7

-- Prove the number of possible integral values
theorem number_of_possible_integral_values :
  7 < BC ∧ BC < 21 → 
  ∃ n : ℕ, n = 13 :=
by
  -- End the proof with sorry, as the focus here is to establish the statement
  sorry

end number_of_possible_integral_values_l187_187419


namespace perpendicular_line_through_point_l187_187282

theorem perpendicular_line_through_point (M : ℝ × ℝ) (l : ℝ × ℝ × ℝ) :
  M = (4, -1) ∧ l = (3, -4, 6) →
  ∃ a b c : ℝ,
    a = 4 ∧ b = 3 ∧ c = -13 ∧
    ∀ x y : ℝ, a*x + b*y + c = 0 ↔ 
      ∃ m x₁ y₁ : ℝ, 
      x₁ = 4 ∧ y₁ = -1 ∧ 
      m = -4/3 ∧ 
      y - y₁ = m * (x - x₁) :=
begin
  sorry,
end

end perpendicular_line_through_point_l187_187282


namespace area_ratio_of_similar_pentagons_l187_187820

-- Definitions of centroids for pentagon vertices and centroids of corresponding triangles
variables {A B C D E G_A G_B G_C G_D G_E : Point}
variables 
  (hGA : G_A = centroid (B :: C :: D :: E :: []))
  (hGB : G_B = centroid (A :: C :: D :: E :: []))
  (hGC : G_C = centroid (A :: B :: D :: E :: []))
  (hGD : G_D = centroid (A :: B :: C :: E :: []))
  (hGE : G_E = centroid (A :: B :: C :: D :: []))

-- Prove that the area ratio is 1/16
theorem area_ratio_of_similar_pentagons (A B C D E G_A G_B G_C G_D G_E : Point):
  G_A = centroid [B, C, D, E] →
  G_B = centroid [A, C, D, E] →
  G_C = centroid [A, B, D, E] →
  G_D = centroid [A, B, C, E] →
  G_E = centroid [A, B, C, D] →
  (area (polygon [G_A, G_B, G_C, G_D, G_E])) / (area (polygon [A, B, C, D, E])) = 1 / 16 :=
by {
  sorry
}

end area_ratio_of_similar_pentagons_l187_187820


namespace competition_inequality_l187_187386

theorem competition_inequality (a b k : ℕ) (hb : b ≥ 3) (hob : odd b) (hmax : ∀ (i j : ℕ) (hij: i ≠ j), ∃ m : ℕ, m ≤ k ∧ (
  ( ∀ c : ℕ, c ≤ m → judge i c = judge j c) ∨
  ( ∀ c : ℕ, c ≤ m → judge i c ≠ judge j c)
)) : 
k / a ≥ (b - 1) / (2 * b) := 
sorry

end competition_inequality_l187_187386


namespace smallest_n_for_4n_square_and_5n_cube_l187_187053

theorem smallest_n_for_4n_square_and_5n_cube :
  ∃ (n : ℕ), (n > 0 ∧ (∃ k : ℕ, 4 * n = k^2) ∧ (∃ m : ℕ, 5 * n = m^3)) ∧ n = 400 :=
by
  sorry

end smallest_n_for_4n_square_and_5n_cube_l187_187053


namespace range_of_z_l187_187004

theorem range_of_z (x y : ℝ) (h : x^2 / 16 + y^2 / 9 = 1) : -5 ≤ x + y ∧ x + y ≤ 5 :=
sorry

end range_of_z_l187_187004


namespace tan_30_l187_187150

theorem tan_30 : Real.tan (Real.pi / 6) = Real.sqrt 3 / 3 := 
by 
  have h1 : Real.sin (Real.pi / 6) = 1 / 2 := by sorry
  have h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2 := by sorry
  calc
    Real.tan (Real.pi / 6) = Real.sin (Real.pi / 6) / Real.cos (Real.pi / 6) : Real.tan_eq_sin_div_cos _
    ... = (1 / 2) / (Real.sqrt 3 / 2) : by rw [h1, h2]
    ... = (1 / 2) * (2 / Real.sqrt 3) : by rw Div.div_eq_mul_inv
    ... = 1 / Real.sqrt 3 : by norm_num
    ... = Real.sqrt 3 / 3 : by rw [Div.inv_eq_inv, Mul.comm, Mul.assoc, Div.mul_inv_cancel (Real.sqrt_ne_zero _), one_div Real.sqrt 3, inv_mul_eq_div]

-- Additional necessary function apologies for the unproven theorems.
noncomputable def _root_.Real.sqrt (x:ℝ) : ℝ := sorry

noncomputable def _root_.Real.tan (x : ℝ) : ℝ :=
  (Real.sin x) / (Real.cos x)

#eval tan_30 -- check result

end tan_30_l187_187150


namespace matching_book_lists_l187_187511

theorem matching_book_lists (n : ℕ) (algebra_books geometry_books : ℕ) (students : ℕ) 
  (h1 : n = 3) (h2 : algebra_books = 6) (h3 : geometry_books = 5) (h4 : students = 210) :
  ∃ (s1 s2 : ℕ), s1 ≠ s2 ∧ 
  ∃ (a1 a2 a3 b1 b2 b3 : ℕ), 
    (a1 ≠ a2 ∧ a1 ≠ a3 ∧ a2 ≠ a3 ∧ 1 ≤ a1 ∧ a1 ≤ algebra_books ∧ 
     1 ≤ a2 ∧ a2 ≤ algebra_books ∧ 1 ≤ a3 ∧ a3 <= algebra_books ∧ 
     b1 ≠ b2 ∧ b1 ≠ b3 ∧ b2 ≠ b3 ∧ 1 ≤ b1 ∧ b1 ≤ geometry_books ∧ 
     1 ≤ b2 ∧ b2 ≤ geometry_books ∧ 1 ≤ b3 ∧ b3 ≠ 0)  ∧
    ∃ (books_choice : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → (Finset ℕ)),
      (books_choice a1 a2 a3 b1 b2 b3).card = 6 ∧
      (books_choice a1 a2 a3 b1 b2 b3 s1) = (books_choice a1 a2 a3 b1 b2 b3 s2) :=
by
  sorry

end matching_book_lists_l187_187511


namespace maximum_accuracy_intersection_l187_187849

theorem maximum_accuracy_intersection (C1 C2 : Set ℝ) (P : ℝ) 
(h1 : ∃ x y : ℝ, C1 = {p | p = (x, y)} ∧ C2 = {q | q = (y, x)} ∧ P ∈ C1 ∩ C2) :
  (∃ θ : ℝ, θ = 90 ∧ tangent_angle_at P C1 = θ ∧ tangent_angle_at P C2 = θ) → 
  accuracy_of_intersection P = maximum_accuracy := 
sorry

end maximum_accuracy_intersection_l187_187849


namespace eval_64_pow_5_over_6_l187_187200

theorem eval_64_pow_5_over_6 (h : 64 = 2^6) : 64^(5/6) = 32 := 
by 
  sorry

end eval_64_pow_5_over_6_l187_187200


namespace unique_three_digit_numbers_unique_four_digit_even_numbers_l187_187513

def digits : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem unique_three_digit_numbers : 
  (List.permutations digits).count (λ l, l.length = 3 ∧ l.head ≠ 0) = 648 :=
by
  sorry

theorem unique_four_digit_even_numbers : 
  (List.permutations digits).count (λ l, l.length = 4 ∧ l.last ∈ [0, 2, 4, 6, 8]) = 2296 :=
by
  sorry

end unique_three_digit_numbers_unique_four_digit_even_numbers_l187_187513


namespace shaded_percentage_of_square_is_correct_l187_187535

theorem shaded_percentage_of_square_is_correct :
  ∀ (a side1 side2 side3 : ℕ),
  a = 6 →
  side1 = 1 →
  side2 = 2 →
  side3 = 3 →
  ((side1^2 + (1 / 2) * side2 * side3 + side2^2) / a^2) * 100 = 22.22 :=
by
  intro a side1 side2 side3 ha hside1 hside2 hside3
  have area_square : ℕ := ha ▸ (a * a)
  have area_small_square : ℕ := hside1 ▸ (side1 * side1)
  have area_triangle : ℕ := hside2 ▸ hside3 ▸ ((1 / 2) * side2 * side3)
  have area_large_square : ℕ := hside2 ▸ (side2 * side2)
  have total_area_shaded : ℕ := area_small_square + area_triangle + area_large_square
  have fraction_shaded : ℕ := (total_area_shaded / area_square)
  have percentage_shaded : ℕ := fraction_shaded * 100
  exact percentage_shaded = 22.22
  sorry

end shaded_percentage_of_square_is_correct_l187_187535


namespace tan_30_deg_l187_187170

theorem tan_30_deg : 
  let θ := (30 : ℝ) * (Real.pi / 180)
  in Real.sin θ = 1 / 2 ∧ Real.cos θ = Real.sqrt 3 / 2 → Real.tan θ = Real.sqrt 3 / 3 :=
by
  intro h
  let th := θ
  have h1 : Real.sin th = 1 / 2 := And.left h
  have h2 : Real.cos th = Real.sqrt 3 / 2 := And.right h
  sorry

end tan_30_deg_l187_187170


namespace annual_income_of_A_l187_187931

variable (Cm : ℝ)
variable (Bm : ℝ)
variable (Am : ℝ)
variable (Aa : ℝ)

-- Given conditions
axiom h1 : Cm = 12000
axiom h2 : Bm = Cm + 0.12 * Cm
axiom h3 : (Am / Bm) = 5 / 2

-- Statement to prove
theorem annual_income_of_A : Aa = 403200 := by
  sorry

end annual_income_of_A_l187_187931


namespace max_q_value_l187_187801

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end max_q_value_l187_187801


namespace sum_of_solutions_l187_187833

noncomputable def f (x : ℝ) : ℝ := x^2 - x + 2

theorem sum_of_solutions (z : ℝ) :
  (∀ z, f (4 * z) = 10 → z = (2 + Real.sqrt 132) / 64 ∨ z = (2 - Real.sqrt 132) / 64) →
  ∑ (h : f (4 * z) = 10), z = 1 / 16 :=
by
  sorry

end sum_of_solutions_l187_187833


namespace tan_30_eq_sqrt3_div3_l187_187151

theorem tan_30_eq_sqrt3_div3 (sin_30_cos_30 : ℝ → ℝ → Prop)
  (h1 : sin_30_cos_30 (1 / 2) (Real.sqrt 3 / 2)) :
  ∃ t, t = Real.tan (Real.pi / 6) ∧ t = Real.sqrt 3 / 3 :=
by
  existsi Real.tan (Real.pi / 6)
  sorry

end tan_30_eq_sqrt3_div3_l187_187151


namespace largest_gcd_l187_187011

theorem largest_gcd (a b : ℕ) (h : a + b = 1008) : ∃ d, d = gcd a b ∧ (∀ d', d' = gcd a b → d' ≤ d) ∧ d = 504 :=
by
  sorry

end largest_gcd_l187_187011


namespace find_n_from_digits_sum_l187_187877

theorem find_n_from_digits_sum (n : ℕ) (h1 : 777 = (9 * 1) + ((99 - 10 + 1) * 2) + (n - 99) * 3) : n = 295 :=
sorry

end find_n_from_digits_sum_l187_187877


namespace quadratic_residue_pairs_count_l187_187288

theorem quadratic_residue_pairs_count :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (a b : ℕ), (a ∈ Finset.range 36) ∧ (b ∈ Finset.range 36) → ((a, b) ∈ S ↔ is_quadratic_residue (a * x + b) (x^2 + 1) ∧ gcd 35 = 1)) ∧ S.card = 225 :=
  sorry

end quadratic_residue_pairs_count_l187_187288


namespace gcd_max_value_l187_187018

theorem gcd_max_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1008) : 
  ∃ d, d = Nat.gcd a b ∧ d = 504 :=
by
  sorry

end gcd_max_value_l187_187018


namespace evaluate_pow_l187_187247

theorem evaluate_pow : 64^(5/6 : ℝ) = 32 := by
  sorry

end evaluate_pow_l187_187247


namespace triangle_area_l187_187524

open Real

noncomputable def line1 (x : ℝ) : ℝ := -1/3 * x + 4
noncomputable def line2 (x : ℝ) : ℝ := 3 * x - 6
noncomputable def line3 (x : ℝ) : ℝ := 12 - x

def pointA : ℝ × ℝ := (3, 3)
def pointC : ℝ × ℝ := (12, 0)
def pointB : ℝ × ℝ := (4.5, 7.5)

theorem triangle_area :
  let (x₁, y₁) := pointA
  let (x₂, y₂) := pointB
  let (x₃, y₃) := pointC
  ½ * abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)) = 22.5 :=
by
  sorry

end triangle_area_l187_187524


namespace min_painted_cells_l187_187966

noncomputable def knight_moves (x y : ℕ) : set (ℕ × ℕ) :=
  {(x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1),
   (x + 1, y + 2), (x + 1, y - 2), (x - 1, y + 2), (x - 1, y - 2)}

noncomputable def grid_size : ℕ := 35

noncomputable def num_cells (N : ℕ) : ℕ := N * N

theorem min_painted_cells : ∃ (painted_cells : ℕ), painted_cells = 612 ∧
  ∀ (x y : ℕ), 
  (x < grid_size ∧ y < grid_size) →
  ∀ (move ∈ knight_moves x y), 
  let (nx, ny) := move in (nx < grid_size ∧ ny < grid_size) →
  (∃! c, c ∈ {(x, y), (nx, ny)} ∧ c.fst % 2 = 0 ∧ c.snd % 2 = 0) :=
sorry

end min_painted_cells_l187_187966


namespace complement_intersection_l187_187331

-- Definitions of the sets
def R := Set ℝ
def M : Set ℝ := {x | x ≤ Real.sqrt 5}
def N : Set ℝ := {1, 2, 3, 4}
def compM := {x : ℝ | x ∈ R ∧ x > Real.sqrt 5}

-- Theorem stating the required intersection
theorem complement_intersection :
  (compM ∩ N) = {3, 4} :=
sorry

end complement_intersection_l187_187331


namespace deep_champion_probability_l187_187296

-- Probabilities are represented as real numbers between 0 and 1
noncomputable def equal_prob : ℝ := 1 / 2

-- Define the probability of specific sequences leading to Deep's victory
def prob_deep_wins_in_6_games : ℝ := (equal_prob ^ 4)

def prob_deep_wins_in_7_games : ℝ := 4 * (equal_prob ^ 5)

-- Define total probability
def total_prob_deep_wins : ℝ := prob_deep_wins_in_6_games + prob_deep_wins_in_7_games

theorem deep_champion_probability :
  total_prob_deep_wins = 3 / 16 :=
by
  -- Proof omitted
  sorry

end deep_champion_probability_l187_187296


namespace find_number_l187_187986

theorem find_number (x : ℝ) (h : 0.70 * x - 40 = 30) : x = 100 :=
sorry

end find_number_l187_187986


namespace find_ratio_l187_187607

theorem find_ratio (x y c d : ℝ) (h1 : 8 * x - 6 * y = c) (h2 : 12 * y - 18 * x = d) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) : c / d = -1 := by
  sorry

end find_ratio_l187_187607


namespace max_value_of_expression_l187_187806

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l187_187806


namespace evaluate_64_pow_5_div_6_l187_187272

theorem evaluate_64_pow_5_div_6 : (64 : ℝ) ^ (5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ) ^ 6 := by norm_num
  have h2 : (64 : ℝ) ^ (5 / 6) = ((2 : ℝ) ^ 6) ^ (5 / 6) := by rw h1
  have h3 : ((2 : ℝ) ^ 6) ^ (5 / 6) = (2 : ℝ) ^ (6 * (5 / 6)) := by rw [Real.rpow_mul]
  have h4 : (2 : ℝ) ^ (6 * (5 / 6)) = (2 : ℝ) ^ 5 := by norm_num
  rw [h2, h3, h4]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187272


namespace evaluate_64_pow_fifth_sixth_l187_187215

theorem evaluate_64_pow_fifth_sixth : 64 ^ (5 / 6) = 32 := by
  have h : 64 = 2 ^ 6 := by sorry
  calc 64 ^ (5 / 6) = (2 ^ 6) ^ (5 / 6) : by rw [h]
              ...   = 2 ^ (6 * (5 / 6))  : by sorry
              ...   = 2 ^ 5              : by sorry
              ...   = 32                 : by sorry

end evaluate_64_pow_fifth_sixth_l187_187215


namespace sequence_formula_and_sum_l187_187697

-- The definition of function f
def f (x : ℝ) := x / (x + 3)

-- The sequence a_n defined recursively
def a : ℕ → ℝ
| 0     := 1 -- Adjusting to 0-based indexing for Lean's natural numbers
| (n+1) := f (a n)

-- The definition of b_n based on a_n
def b (n : ℕ) := (1/2) * a n * a (n + 1) * 3^n

-- The partial sum S_n definition
def S : ℕ → ℝ
| 0     := b 0
| (n+1) := S n + b (n+1)

-- The statement to prove
theorem sequence_formula_and_sum (n : ℕ) :
  (a n = 2 / (3^n - 1)) ∧ (S n = 1/2 - 1/(3^(n+1) - 1)) :=
sorry

end sequence_formula_and_sum_l187_187697


namespace evaluate_64_pow_fifth_sixth_l187_187219

theorem evaluate_64_pow_fifth_sixth : 64 ^ (5 / 6) = 32 := by
  have h : 64 = 2 ^ 6 := by sorry
  calc 64 ^ (5 / 6) = (2 ^ 6) ^ (5 / 6) : by rw [h]
              ...   = 2 ^ (6 * (5 / 6))  : by sorry
              ...   = 2 ^ 5              : by sorry
              ...   = 32                 : by sorry

end evaluate_64_pow_fifth_sixth_l187_187219


namespace positive_multiples_of_11_ending_in_5_less_than_2000_l187_187366

theorem positive_multiples_of_11_ending_in_5_less_than_2000 :
  ∃ (n : ℕ), (∀ (k < n), 
    let a := 55 + k * 110 in 
      a < 2000 ∧ 
      a % 11 = 0 ∧ 
      a % 10 = 5) ∧ n = 18 :=
by
  sorry

end positive_multiples_of_11_ending_in_5_less_than_2000_l187_187366


namespace even_function_a_value_l187_187736

theorem even_function_a_value (a : ℝ) :
  (∀ x : ℝ, (x + 1) * (x - a) = (-x + 1) * (-x - a)) → a = 1 :=
by
  sorry

end even_function_a_value_l187_187736


namespace max_people_seated_l187_187521

-- Define the total number of chairs
def chairs : ℕ := 12

-- Conditions:
-- 1. There are 12 chairs in a row, initially empty.
-- 2. If a person sits in an unoccupied chair, one of their neighbors stands up and leaves.

-- Define a function to determine the maximum number of people who can sit given the conditions
theorem max_people_seated :
  ∃ max_people : ℕ, max_people ≤ chairs ∧
  (∀ (seating_arrangement : Fin chairs → Bool), 
   (seating_arrangement.toList.count true = max_people → 
    (∀ (i : Fin (chairs - 1)),
      (seating_arrangement i → ¬ seating_arrangement (i + 1) → ¬ seating_arrangement (i + 2))))) := sorry

end max_people_seated_l187_187521


namespace reciprocal_of_2022_l187_187006

theorem reciprocal_of_2022 : 1 / 2022 = (1 : ℝ) / 2022 :=
sorry

end reciprocal_of_2022_l187_187006


namespace find_function_satisfying_equation_l187_187091

theorem find_function_satisfying_equation :
  ∃ f : ℝ → ℝ, (∀ x y : ℝ, f (x + y) + f (x - y) = x^2 + y^2) ∧ (∀ x : ℝ, f x = x^2 / 2) :=
by
  let f : ℝ → ℝ := λ x, x^2 / 2
  existsi f
  split
  {
    intros x y
    calc
      f (x + y) + f (x - y) = (x + y)^2 / 2 + (x - y)^2 / 2 : by simp [f]
      ... = (x^2 + 2 * x * y + y^2) / 2 + (x^2 - 2 * x * y + y^2) / 2 : by ring
      ... = (x^2 + y^2) : by ring
  }
  {
    intro x
    refl
  }

end find_function_satisfying_equation_l187_187091


namespace evaluate_root_l187_187242

theorem evaluate_root : 64 ^ (5 / 6 : ℝ) = 32 :=
by sorry

end evaluate_root_l187_187242


namespace gcd_max_value_l187_187016

theorem gcd_max_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1008) : 
  ∃ d, d = Nat.gcd a b ∧ d = 504 :=
by
  sorry

end gcd_max_value_l187_187016


namespace find_a_b_find_relationship_range_maximize_profit_l187_187127

variables (a b x y : ℕ)

-- Purchased amounts and costs
def purchase_equation_1 (a b : ℕ) : Prop := 2 * a + b = 110
def purchase_equation_2 (a b : ℕ) : Prop := 4 * a + 3 * b = 260

-- Profit function
def profit_function (a b x y : ℕ) : Prop :=
  y = (50 - a) * x + (60 - b) * (300 - x)

-- Range of x
def range_of_x (x : ℕ) : Prop :=
  100 ≤ x ∧ x ≤ 150 ∧ x ≥ (300 - x) / 2

-- Maximum profit conditions
def maximize_profit_a_lt_5 (a x : ℕ) : Prop := a < 5 ∧ x = 100
def maximize_profit_a_eq_5 (a y : ℕ) : Prop := a = 5 ∧ y = 6000
def maximize_profit_a_gt_5 (a x : ℕ) : Prop := a > 5 ∧ x = 150

theorem find_a_b :
  purchase_equation_1 a b ∧ purchase_equation_2 a b → a = 35 ∧ b = 40 :=
sorry

theorem find_relationship_range :
  profit_function 35 40 x y ∧ range_of_x x → y = -5 * x + 6000 :=
sorry

theorem maximize_profit :
  (maximize_profit_a_lt_5 a x ∨ maximize_profit_a_eq_5 a y ∨ maximize_profit_a_gt_5 a x) :=
sorry

end find_a_b_find_relationship_range_maximize_profit_l187_187127


namespace juanita_spends_more_l187_187361

-- Define the expenditures
def grant_yearly_expenditure : ℝ := 200.00

def juanita_weekday_expenditure : ℝ := 0.50

def juanita_sunday_expenditure : ℝ := 2.00

def weeks_per_year : ℕ := 52

-- Given conditions translated to Lean
def juanita_weekly_expenditure : ℝ :=
  (juanita_weekday_expenditure * 6) + juanita_sunday_expenditure

def juanita_yearly_expenditure : ℝ :=
  juanita_weekly_expenditure * weeks_per_year

-- The statement we need to prove
theorem juanita_spends_more : (juanita_yearly_expenditure - grant_yearly_expenditure) = 60.00 :=
by
  sorry

end juanita_spends_more_l187_187361


namespace last_8_digits_of_product_l187_187965

theorem last_8_digits_of_product :
  let p := 11 * 101 * 1001 * 10001 * 1000001 * 111
  (p % 100000000) = 87654321 :=
by
  let p := 11 * 101 * 1001 * 10001 * 1000001 * 111
  have : p % 100000000 = 87654321 := sorry
  exact this

end last_8_digits_of_product_l187_187965


namespace max_q_value_l187_187802

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end max_q_value_l187_187802


namespace angleACB_l187_187830

open Point
open Vector

variables {A B C D E F : Point ℝ}
          {α β γ : Angle}
          {hA : altitude A B C = D}
          {hB : altitude B C A = E}
          {hC : altitude C A B = F}
          {vAD : Vector ℝ}
          {vBE : Vector ℝ}
          {vCF : Vector ℝ}

def altitudes (α β γ : Angle) (AD BE CF : Vector ℝ) := 
  7 * AD + 3 * BE + 5 * CF = 0

theorem angleACB (h1 : altitudes α β γ vAD vBE vCF) : α = 60 := 
sorry

end angleACB_l187_187830


namespace coefficient_of_x2_in_expansion_of_x_plus_1_pow_4_l187_187902

theorem coefficient_of_x2_in_expansion_of_x_plus_1_pow_4 :
  let expansion := (x + 1) ^ 4 in
  let term_coefficient := (4.choose 2) in
  term_coefficient = 6 :=
by
  sorry

end coefficient_of_x2_in_expansion_of_x_plus_1_pow_4_l187_187902


namespace number_of_true_statements_l187_187365

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m
def is_odd (n : ℕ) : Prop := ∃ m : ℕ, n = 2 * m + 1
def is_even (n : ℕ) : Prop := ∃ m : ℕ, n = 2 * m

theorem number_of_true_statements : 3 = (ite ((∀ p q : ℕ, is_prime p → is_prime q → is_prime (p * q)) = false) 0 1) +
                                     (ite ((∀ a b : ℕ, is_square a → is_square b → is_square (a * b)) = true) 1 0) +
                                     (ite ((∀ x y : ℕ, is_odd x → is_odd y → is_odd (x * y)) = true) 1 0) +
                                     (ite ((∀ u v : ℕ, is_even u → is_even v → is_even (u * v)) = true) 1 0) :=
by
  sorry

end number_of_true_statements_l187_187365


namespace rain_at_least_once_prob_l187_187497

theorem rain_at_least_once_prob (p : ℚ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 :=
by {
  -- Implementation of Lean code is not required as per instructions.
  sorry
}

end rain_at_least_once_prob_l187_187497


namespace gcd_largest_divisor_l187_187026

theorem gcd_largest_divisor (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 1008) : 
  ∃ d, nat.gcd a b = d ∧ d = 504 :=
begin
  sorry
end

end gcd_largest_divisor_l187_187026


namespace find_m_for_given_eccentricity_l187_187334

theorem find_m_for_given_eccentricity (m : ℝ) (h1 : m > 0) (h2 : eccentricity (ellipse 1 m) = 1/2) : 
  m = 3/4 ∨ m = 4/3 := by
s-tasophy

end find_m_for_given_eccentricity_l187_187334


namespace func_positive_range_l187_187639

theorem func_positive_range (a : ℝ) : 
  (∀ x : ℝ, (5 - a) * x^2 - 6 * x + a + 5 > 0) → (-4 < a ∧ a < 4) := 
by 
  sorry

end func_positive_range_l187_187639


namespace negation_of_inverse_true_l187_187380

variables (P : Prop)

theorem negation_of_inverse_true (h : ¬P → false) : ¬P := by
  sorry

end negation_of_inverse_true_l187_187380


namespace limits_imply_equality_l187_187420

open Filter Topology

noncomputable def bounded_seq (a : ℕ → ℝ) := ∃ B, ∀ n, |a n| ≤ B

theorem limits_imply_equality {a : ℕ → ℝ} 
  (h_bounded : bounded_seq a) 
  (h1 : Tendsto (λ N, (1 : ℝ) / N * (Finset.sum (Finset.range N) a)) atTop (nhds b)) 
  (h2 : Tendsto (λ N, (1 : ℝ) / (Real.log N) * (Finset.sum (Finset.range N) (λ n, a n / n))) atTop (nhds c)) 
  : b = c := 
by 
  sorry

end limits_imply_equality_l187_187420


namespace tangent_circle_distance_relation_l187_187994

theorem tangent_circle_distance_relation
  (A P Q : Point)
  (u v w : ℝ)
  (h1 : Tangent (CircleThrough [P, Q]) (LineThrough P Q))
  (h2 : dist (Foot A (LineThrough P Q)) A = w)
  (h3 : dist P (LineThrough P Q) = u)
  (h4 : dist Q (LineThrough P Q) = v) :
  (u * v) / w^2 = Real.sin (AngleBetween (RayFrom A P) (RayFrom A Q) / 2)^2 := 
sorry

end tangent_circle_distance_relation_l187_187994


namespace max_dot_product_l187_187713

variable (θ : ℝ)

def a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
def b : ℝ × ℝ := (3, -4)

noncomputable def dot_product : ℝ := a θ.1 * b.1 + a θ.2 * b.2

theorem max_dot_product : ∃ θ : ℝ, dot_product θ = 5 := by
  sorry

end max_dot_product_l187_187713


namespace find_triplets_l187_187278

theorem find_triplets (x y z : ℕ) :
  (x^2 + y^2 = 3 * 2016^z + 77) →
  (x, y, z) = (77, 14, 1) ∨ (x, y, z) = (14, 77, 1) ∨ 
  (x, y, z) = (70, 35, 1) ∨ (x, y, z) = (35, 70, 1) ∨ 
  (x, y, z) = (8, 4, 0) ∨ (x, y, z) = (4, 8, 0) :=
by
  sorry

end find_triplets_l187_187278


namespace evaluate_pow_l187_187248

theorem evaluate_pow : 64^(5/6 : ℝ) = 32 := by
  sorry

end evaluate_pow_l187_187248


namespace round_robin_max_tied_teams_l187_187394

-- Definitions and the main theorem statement
theorem round_robin_max_tied_teams (n : ℕ) (h : n = 8) :
  let total_games := nat.choose n 2 in
  let win_loss_distinct := ∀ i j : ℕ, i ≠ j → i ≠ j in
  (total_games = 28) →
  (∀ k, k ∈ finset.range 0 8 → ∃ m, m ∈ finset.range (nat.div 28 k.succ)) -/
  7 = maximum_teams_tied_for_most_wins n :=
sorry

end round_robin_max_tied_teams_l187_187394


namespace smallest_number_of_elements_l187_187824

theorem smallest_number_of_elements (S : Type) (X : Fin 100 → set S) :
  (∀ i : Fin 99, X i ≠ ∅ ∧ X (i + 1).val.succ ≠ ∅ ∧ X i ∩ X (i + 1).val.succ = ∅ ∧ (X i ∪ X (i + 1).val.succ) ≠ set.univ) →
  (∀ i j : Fin 100, i ≠ j → X i ≠ X j) →
  8 ≤ Fintype.card S :=
sorry

end smallest_number_of_elements_l187_187824


namespace exists_set_of_25_distinct_integers_with_properties_l187_187407

theorem exists_set_of_25_distinct_integers_with_properties :
  ∃ (S : Finset ℤ), S.card = 25 ∧ (∑ i in S, i = 0) ∧ 
  ∀ i ∈ S, abs ((∑ j in S \ {i}, j)) > 25 :=
by sorry

end exists_set_of_25_distinct_integers_with_properties_l187_187407


namespace sin_is_odd_l187_187449

theorem sin_is_odd : ∀ x : ℝ, sin (-x) = -sin x :=
by
  intro x
  sorry

end sin_is_odd_l187_187449


namespace find_function_l187_187622

noncomputable def f : ℝ → ℝ := sorry

theorem find_function :
  (∀ x y : ℝ, f (1 + x * y) - f (x + y) = f x * f y) ∧ f (-1) ≠ 0 →
  (∀ x : ℝ, f x = x - 1) :=
begin
  sorry
end

end find_function_l187_187622


namespace line_passes_through_specific_quadrants_l187_187318

theorem line_passes_through_specific_quadrants
  (a b c : ℝ)
  (hab : a * b < 0)
  (hbc : b * c < 0) :
  ∃ q1 q3 q4 : ℝ × ℝ, 
  (line_eq a b c q1 ∧ q1.1 > 0 ∧ q1.2 > 0) ∧
  (line_eq a b c q3 ∧ q3.1 < 0 ∧ q3.2 < 0) ∧
  (line_eq a b c q4 ∧ q4.1 > 0 ∧ q4.2 < 0) :=
sorry

def line_eq (a b c : ℝ) (p: ℝ × ℝ) : Prop :=
a * p.1 + b * p.2 = c

end line_passes_through_specific_quadrants_l187_187318


namespace AD_bisects_CE_l187_187134

variables (A O B C D E M : Type)
variable (circle_O : Set O) -- Represents circle O
variable [MetricSpace O] -- Assuming O lies in a metric space for distance definitions

-- Definitions of geometric entities
variable (A_outside_O : ∀ x ∈ circle_O, x ≠ A)
variable (tangents_AB_AC : LineSegment O → Bool) -- AB and AC are tangent to circle O 
variable (B_C_tangency : B ∈ circle_O ∧ C ∈ circle_O)
variable (BD_diameter : ∃ D, isDiameter BD circle_O) -- BD is the diameter
variable (CE_perp_BD_at_E : ∃ E, E ≠ D ∧ E ≠ C ∧ isPerpendicular CE BD E) -- CE is perpendicular to BD at E
variable (AD_intersects_CE_at_M : ∃ M, isIntersection AD CE M) -- AD intersects CE at M

theorem AD_bisects_CE :
  B_C_tangency B C →
  BD_diameter BD → 
  CE_perp_BD_at_E E →
  AD_intersects_CE_at_M M →
  bisects AD CE :=
begin
  intros hBC hBD hCE hAM,
  sorry
end

end AD_bisects_CE_l187_187134


namespace pentagon_coverage_percentage_l187_187489

def area_of_large_square (b : ℕ) : ℕ := 16 * b^2

def area_covered_by_pentagons (b : ℕ) : ℕ := 8 * b^2

theorem pentagon_coverage_percentage :
  let b := 1 in
  let total_area := area_of_large_square b in
  let pentagon_area := area_covered_by_pentagons b in
  (pentagon_area * 100) / total_area = 50 :=
by
  sorry

end pentagon_coverage_percentage_l187_187489


namespace sum_of_two_numbers_l187_187871

theorem sum_of_two_numbers (a b : ℕ) (h1 : a - b = 10) (h2 : a = 22) : a + b = 34 :=
sorry

end sum_of_two_numbers_l187_187871


namespace raghu_investment_l187_187551

noncomputable def R := 2299.65

theorem raghu_investment (V T : ℝ) 
  (h1 : T = 0.9 * R)
  (h2 : V = 1.1 * T)
  (h3 : R + T + V = 6647) : 
  R = 2299.65 :=
sorry

end raghu_investment_l187_187551


namespace range_of_a_l187_187470

theorem range_of_a (a : ℝ) : (∃ x ∈ Ioo 1 2, log 3 x - a = 0) ↔ a ∈ Ioo 0 (log 3 2) := sorry

end range_of_a_l187_187470


namespace evaluate_pow_l187_187261

theorem evaluate_pow : (64 : ℝ) = (8 : ℝ) ^ 2 → (8 : ℝ) = (2 : ℝ) ^ 3 → (64 : ℝ) ^ (5 / 6) = 32 :=
by
  intros h1 h2
  rw h1
  rw h2
  have h3 : (2 : ℝ)^3 ^ 2 = (2 : ℝ) ^ 6 := by ring_exp
  rw h3
  sorry

end evaluate_pow_l187_187261


namespace evaluate_64_pow_5_div_6_l187_187264

theorem evaluate_64_pow_5_div_6 : (64 : ℝ) ^ (5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ) ^ 6 := by norm_num
  have h2 : (64 : ℝ) ^ (5 / 6) = ((2 : ℝ) ^ 6) ^ (5 / 6) := by rw h1
  have h3 : ((2 : ℝ) ^ 6) ^ (5 / 6) = (2 : ℝ) ^ (6 * (5 / 6)) := by rw [Real.rpow_mul]
  have h4 : (2 : ℝ) ^ (6 * (5 / 6)) = (2 : ℝ) ^ 5 := by norm_num
  rw [h2, h3, h4]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187264


namespace y_relationship_l187_187378

theorem y_relationship :
  ∀ (y1 y2 y3 : ℝ), 
  (y1 = (-2)^2 - 4*(-2) - 3) ∧ 
  (y2 = 1^2 - 4*1 - 3) ∧ 
  (y3 = 4^2 - 4*4 - 3) → 
  y1 > y3 ∧ y3 > y2 := 
by sorry

end y_relationship_l187_187378


namespace smallest_n_for_perfect_square_and_cube_l187_187073

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, 0 < n ∧ (∃ a1 b1 : ℕ, 4 * n = a1 ^ 2 ∧ 5 * n = b1 ^ 3 ∧ n = 50) :=
begin
  use 50,
  split,
  { norm_num, },
  { use [10, 5],
    split,
    { norm_num, },
    { split, 
      { norm_num, },
      { refl, }, },
  },
  sorry
end

end smallest_n_for_perfect_square_and_cube_l187_187073


namespace min_value_of_a_l187_187110

theorem min_value_of_a (a m n : ℕ) (h1 : 0.2 * a = m * m) (h2 : 0.5 * a = n * n * n) : a = 2000 :=
sorry

end min_value_of_a_l187_187110


namespace weight_of_b_l187_187898

theorem weight_of_b (A B C : ℝ) : 
  (A + B + C = 135) → 
  (A + B = 80) → 
  (B + C = 86) → 
  B = 31 :=
by {
  intros h1 h2 h3,
  sorry
}

end weight_of_b_l187_187898


namespace max_value_expression_l187_187794

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end max_value_expression_l187_187794


namespace curve_is_circle_l187_187627

theorem curve_is_circle (r : ℝ) (h : r = 5) : ∃ c : ℝ, ∀ θ : ℝ, r = c ∧ c = 5 :=
by {
  use 5,
  intro θ,
  exact ⟨h, rfl⟩,
  sorry
}

end curve_is_circle_l187_187627


namespace complex_sum_inequality_l187_187421

open Complex Real

variables {n : ℕ} {r : ℝ}
variables {z : Fin n → ℂ}

noncomputable def sum_z (z : Fin n → ℂ) := ∑ i : Fin n, z i
noncomputable def sum_inv_z (z : Fin n → ℂ) := ∑ i : Fin n, 1 / z i

theorem complex_sum_inequality 
  (hz : ∀ i : Fin n, abs (z i - 1) ≤ r)
  (hr : r ∈ Set.Ioo 0 1) :
  abs (sum_z z) * abs (sum_inv_z z) ≥ n^2 * (1 - r^2) :=
sorry

end complex_sum_inequality_l187_187421


namespace tangent_line_at_1_l187_187918

noncomputable def f (x : ℝ) : ℝ := x^4 - 2 * x^3

theorem tangent_line_at_1 :
  let p := (1 : ℝ, f 1)
  in ∃ m c : ℝ, (∀ x : ℝ, y : ℝ, y = m * x + c ↔ y = -2 * x + 1) ∧ ∀ x : ℝ, f x = x^4 - 2 * x^3 :=
sorry

end tangent_line_at_1_l187_187918


namespace total_time_is_correct_l187_187990

/-- A car trip problem from A to B, B to C, and back to A, including break times. We need to prove
that the total journey time is approximately 24.79 hours.
Conditions include: 
  - Segments from A to B: 4 hours at 75 mph, 3 hours at 60 mph, 2 hours at 50 mph.
  - Break in Town B: 0.5 hours.
  - Segments from B to C: 1 hour at 40 mph, 1 hour at 55 mph, 1.5 hours at 65 mph.
  - Break in Town C: 0.75 hours.
  - Return from C to A: The total distance is the sum of distances from A to B and B to C, with an average speed of 70 mph.
-/

namespace CarTrip

def time_A_to_B : ℝ := (4 + 3 + 2) -- hours
def speed_A_to_B: ℝ := (4 * 75 + 3 * 60 + 2 * 50) / (4 + 3 + 2) -- mph

def time_B_to_C : ℝ := (1 + 1 + 1.5) -- hours
def speed_B_to_C: ℝ := (1 * 40 + 1 * 55 + 1.5 * 65) / (1 + 1 + 1.5) -- mph

def break_B : ℝ := 0.5 -- hours
def break_C : ℝ := 0.75 -- hours

def distance_A_to_B : ℝ := (4 * 75 + 3 * 60 + 2 * 50) -- miles
def distance_B_to_C : ℝ := (1 * 40 + 1 * 55 + 1.5 * 65) -- miles 

def total_distance : ℝ := distance_A_to_B + distance_B_to_C -- miles
def speed_C_to_A : ℝ := 70 -- mph

def time_C_to_A : ℝ := total_distance / speed_C_to_A -- hours

def total_driving_time : ℝ := time_A_to_B + time_B_to_C + time_C_to_A -- hours
def total_break_time : ℝ := break_B + break_C -- hours

def total_journey_time : ℝ := total_driving_time + total_break_time -- hours

theorem total_time_is_correct : abs (total_journey_time - 24.79) < 0.01 := sorry

end CarTrip

end total_time_is_correct_l187_187990


namespace trig_identity_proof_l187_187382

theorem trig_identity_proof (θ : ℝ) :
  (∃ P : ℝ × ℝ, P = (-3/5, 4/5) ∧
  let x := P.1; let y := P.2; let r := 1;
  let cos_θ := x / r; let tan_θ := y / x in
  ∃ cos_θ = -3/5 ∧ tan_θ = -4/3) →
  sin (π/2 + θ) + cos (π - θ) + tan (2 * π - θ) = 4/3 :=
by
  sorry

end trig_identity_proof_l187_187382


namespace hyperbola_sufficiency_l187_187187

open Real

theorem hyperbola_sufficiency (k : ℝ) : 
  (9 - k < 0 ∧ k - 4 > 0) → 
  (∃ x y : ℝ, (x^2) / (9 - k) + (y^2) / (k - 4) = 1) :=
by
  intro hk
  sorry

end hyperbola_sufficiency_l187_187187


namespace find_r_l187_187620

theorem find_r (r : ℚ) (h : log 64 (3 * r - 2) = -1 / 2) : r = 17 / 24 :=
sorry

end find_r_l187_187620


namespace rain_at_least_once_prob_l187_187498

theorem rain_at_least_once_prob (p : ℚ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 :=
by {
  -- Implementation of Lean code is not required as per instructions.
  sorry
}

end rain_at_least_once_prob_l187_187498


namespace num_ways_to_select_five_crayons_including_red_l187_187510

noncomputable def num_ways_select_five_crayons (total_crayons : ℕ) (selected_crayons : ℕ) (fixed_red_crayon : ℕ) : ℕ :=
  Nat.choose (total_crayons - fixed_red_crayon) selected_crayons

theorem num_ways_to_select_five_crayons_including_red
  (total_crayons : ℕ) 
  (fixed_red_crayon : ℕ)
  (selected_crayons : ℕ)
  (h1 : total_crayons = 15)
  (h2 : fixed_red_crayon = 1)
  (h3 : selected_crayons = 4) : 
  num_ways_select_five_crayons total_crayons selected_crayons fixed_red_crayon = 1001 := by
  sorry

end num_ways_to_select_five_crayons_including_red_l187_187510


namespace locus_of_center_of_circle_through_points_l187_187314

theorem locus_of_center_of_circle_through_points :
  ∃ P : ℝ × ℝ, (∃ k : ℝ, P = (0, k) ∧ k ≠ 0) ∧
  ((circle (0, -5) r).center = P ∧ (circle (0, 5) r).center = P) →
  ∀ x y : ℝ, x ≠ 0 → (y^2 / 169) + (x^2 / 144) = 1 := 
by
  sorry

end locus_of_center_of_circle_through_points_l187_187314


namespace largest_gcd_l187_187021

theorem largest_gcd (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 1008) : 
  ∃ d : ℕ, d = Int.gcd a b ∧ d = 504 :=
by
  sorry

end largest_gcd_l187_187021


namespace volume_of_pyramid_l187_187900

theorem volume_of_pyramid (c : ℝ) : 
  (∃ (BC AC : ℝ), BC = c / 2 ∧ AC = (c * real.sqrt 3) / 2 ∧ 
  ∃ (S_base : ℝ), S_base = (c^2 * real.sqrt 3) / 8 ∧ 
  ∃ (SO : ℝ), SO = c / 2 ∧ 
  ∃ (V : ℝ), V = 1 / 3 * S_base * SO ∧ 
  V = c^3 * real.sqrt 3 / 48) :=
begin
  sorry
end

end volume_of_pyramid_l187_187900


namespace find_principal_l187_187741

theorem find_principal (x y : ℝ) : 
  (2 * x * y / 100 = 400) → 
  (2 * x * y + x * y^2 / 100 = 41000) → 
  x = 4000 := 
by
  sorry

end find_principal_l187_187741


namespace evaluate_64_pow_5_div_6_l187_187266

theorem evaluate_64_pow_5_div_6 : (64 : ℝ) ^ (5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ) ^ 6 := by norm_num
  have h2 : (64 : ℝ) ^ (5 / 6) = ((2 : ℝ) ^ 6) ^ (5 / 6) := by rw h1
  have h3 : ((2 : ℝ) ^ 6) ^ (5 / 6) = (2 : ℝ) ^ (6 * (5 / 6)) := by rw [Real.rpow_mul]
  have h4 : (2 : ℝ) ^ (6 * (5 / 6)) = (2 : ℝ) ^ 5 := by norm_num
  rw [h2, h3, h4]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187266


namespace solve_porters_transportation_l187_187043

variable (x : ℝ)

def porters_transportation_equation : Prop :=
  (5000 / x = 8000 / (x + 600))

theorem solve_porters_transportation (x : ℝ) (h₁ : 600 > 0) (h₂ : x > 0):
  porters_transportation_equation x :=
sorry

end solve_porters_transportation_l187_187043


namespace k_less_than_zero_l187_187759

variable (k : ℝ)

def function_decreases (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂

theorem k_less_than_zero (h : function_decreases (λ x => k * x - 5)) : k < 0 :=
sorry

end k_less_than_zero_l187_187759


namespace geom_seq_product_l187_187672

variable {α : Type*} [OrderedField α]

-- Geometric sequence definition
def geom_sequence (a : ℕ → α) := ∃ r, ∀ n, a (n+1) = a n * r

-- Given conditions
variables {a : ℕ → α}
variable (h_geom : geom_sequence a)
variable (h_pos : ∀ n, a n > 0)
variable (h_mul : a 1 * a 9 = 16)

-- Statement to be proved
theorem geom_seq_product (h_geom : geom_sequence a) (h_pos : ∀ n, a n > 0) (h_mul : a 1 * a 9 = 16) : 
  a 2 * a 5 * a 8 = 64 := 
sorry

end geom_seq_product_l187_187672


namespace tan_30_eq_sqrt3_div_3_l187_187179

theorem tan_30_eq_sqrt3_div_3 :
  let opposite := 1
  let adjacent := sqrt (3 : ℝ) 
  tan (real.pi / 6) = opposite / adjacent := by 
    sorry

end tan_30_eq_sqrt3_div_3_l187_187179


namespace quadratic_complete_square_l187_187615

theorem quadratic_complete_square (x : ℝ) (m t : ℝ) :
  (4 * x^2 - 16 * x - 448 = 0) → ((x + m) ^ 2 = t) → (t = 116) :=
by
  sorry

end quadratic_complete_square_l187_187615


namespace factor_expression_l187_187273

theorem factor_expression (b : ℝ) : 180 * b ^ 2 + 36 * b = 36 * b * (5 * b + 1) :=
by
  -- actual proof is omitted
  sorry

end factor_expression_l187_187273


namespace weight_of_b_l187_187899

theorem weight_of_b (A B C : ℝ) : 
  (A + B + C = 135) → 
  (A + B = 80) → 
  (B + C = 86) → 
  B = 31 :=
by {
  intros h1 h2 h3,
  sorry
}

end weight_of_b_l187_187899


namespace sampling_method_tv_sampling_method_hall_sampling_method_school_l187_187083

-- Definitions for the conditions
structure PopulationTV (n : ℕ) := 
(tv_count : ℕ) (sample_size : ℕ)
example : PopulationTV :=
{ tv_count := 20, sample_size := 4 }

structure PopulationHall (rows seats_per_row : ℕ) :=
example : PopulationHall :=
{ rows := 32, seats_per_row := 40 }

structure PopulationSchool (teachers admin logistics total sample_size : ℕ) :=
example : PopulationSchool :=
{ teachers := 136, admin := 20, logistics := 24, total := 180, sample_size := 15 }

-- Theorem stating the appropriate sampling method
theorem sampling_method_tv (p : PopulationTV) : 
  p.tv_count = 20 ∧ p.sample_size = 4 → "simple random sampling" := 
by
  sorry

theorem sampling_method_hall (p : PopulationHall) :
  p.rows = 32 ∧ p.seats_per_row = 40 → "systematic sampling" :=
by
  sorry

theorem sampling_method_school (p : PopulationSchool) :
  p.teachers = 136 ∧ p.admin = 20 ∧ p.logistics = 24 ∧ p.total = 180 ∧ p.sample_size = 15 → 
  "stratified sampling" :=
by
  sorry

end sampling_method_tv_sampling_method_hall_sampling_method_school_l187_187083


namespace equation_of_circle_with_AB_diameter_l187_187738

theorem equation_of_circle_with_AB_diameter :
  let x_A : ℝ := -4
      y_A : ℝ := 0
      x_B : ℝ := 0
      y_B : ℝ := 3
      center_x := (x_A + x_B) / 2
      center_y := (y_A + y_B) / 2
      radius_sq := ((x_B - x_A) ^ 2 + (y_B - y_A) ^ 2) / 4
  in  ∀ x y : ℝ, (x + center_x)^2 + (y - center_y)^2 = radius_sq ↔ x^2 + y^2 + 4*x - 3*y = 0 :=
by
  -- Proof is omitted
  sorry   

end equation_of_circle_with_AB_diameter_l187_187738


namespace unique_integer_with_18_factors_l187_187923

/-- Prove that if x is an integer with 18 positive factors, 
       and both 18 and 24 are factors of x, 
       then x = 288 -/
theorem unique_integer_with_18_factors (x : ℤ) 
  (h1 : 18 ∣ x) 
  (h2 : 24 ∣ x) 
  (h3 : Nat.totient x = 18) : 
  x = 288 := 
sorry

end unique_integer_with_18_factors_l187_187923


namespace largest_equal_cost_under_2000_l187_187951

def option1_cost (n : ℕ) : ℕ :=
(n.digits 10).sum + (n.digits 10).length

def option2_cost (n : ℕ) : ℕ :=
(n.digits 2).sum + (n.digits 2).length

def equal_cost_number (limit : ℕ) : ℕ :=
  List.maximum (List.filter (λ n, option1_cost n = option2_cost n) (List.range limit)).get_or_else 0

theorem largest_equal_cost_under_2000 : equal_cost_number 2000 = 1539 := sorry

end largest_equal_cost_under_2000_l187_187951


namespace radius_of_circle_P_l187_187689

noncomputable theory
open_locale classical
open Real

-- Define the given points and line
def A := (3, 1 : ℝ)
def M := (0, 2 : ℝ)
def R1 := 2 -- Radius of circle \odot M
def l (x : ℝ) := -3/4 * x - 11/4

-- Definitions relevant to the problem
def passes_through_point (P : ℝ × ℝ) (r : ℝ) (point : ℝ × ℝ) : Prop :=
  (P.1 - point.1)^2 + (P.2 - point.2)^2 = r^2

def tangent_to_circle (P : ℝ × ℝ) (r : ℝ) (Q : ℝ × ℝ) (R : ℝ) : Prop :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = r + R ∨ sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = abs (r - R)

def perpendicular_distance (P : ℝ × ℝ) (line : ℝ → ℝ) (r : ℝ) : Prop :=
  abs (-3 / 4 * P.1 + P.2 + 11 / 4) / sqrt ((-3 / 4)^2 + 1) = r

-- Main theorem statement
theorem radius_of_circle_P (P : ℝ × ℝ) (r : ℝ) :
  passes_through_point P r A ∧
  tangent_to_circle P r M R1 ∧
  perpendicular_distance P l r →
  r = 3 ∨ r = 7.8 :=
by sorry

end radius_of_circle_P_l187_187689


namespace evaluate_root_l187_187233

theorem evaluate_root : 64 ^ (5 / 6 : ℝ) = 32 :=
by sorry

end evaluate_root_l187_187233


namespace eval_64_pow_5_over_6_l187_187194

theorem eval_64_pow_5_over_6 (h : 64 = 2^6) : 64^(5/6) = 32 := 
by 
  sorry

end eval_64_pow_5_over_6_l187_187194


namespace program_output_l187_187721

theorem program_output (a : ℕ) (ha : a = 35) :
  let b := a // 10 - a / 10 + a % 10
  b = 4.5 :=
by
  sorry

end program_output_l187_187721


namespace score_difference_l187_187417

-- Definitions of the given conditions
def Layla_points : ℕ := 70
def Total_points : ℕ := 112

-- The statement to be proven
theorem score_difference : (Layla_points - (Total_points - Layla_points)) = 28 :=
by sorry

end score_difference_l187_187417


namespace evaluate_64_pow_5_div_6_l187_187232

theorem evaluate_64_pow_5_div_6 : (64 : ℝ)^(5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ)^6 := by norm_num
  rw [← h1]
  have h2 : ((2 : ℝ)^6)^(5 / 6) = (2 : ℝ)^(6 * (5 / 6)) := by rw [Real.rpow_mul]
  rw [h2]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187232


namespace compute_fraction_l187_187144

noncomputable def floor_cbrt (n : ℕ) : ℕ :=
  ⌊ (n : ℝ)^(1/3) ⌋

theorem compute_fraction :
  ( ∏ i in (Finset.range 27).filter (λ i, i % 3 == 1), floor_cbrt i )
  / ( ∏ i in (Finset.range 27).filter (λ i, i % 3 == 2), floor_cbrt i ) = 1 / 3 :=
by
  sorry

end compute_fraction_l187_187144


namespace eval_64_pow_5_over_6_l187_187195

theorem eval_64_pow_5_over_6 (h : 64 = 2^6) : 64^(5/6) = 32 := 
by 
  sorry

end eval_64_pow_5_over_6_l187_187195


namespace intersection_l187_187422

noncomputable def triangle := Type*

structure Point (triangle : Type*) :=
(x y : ℝ)

structure Triangle (triangle : Type*) :=
(A B C : Point triangle)

def angle_bisector {triangle : Type*} (B : Point triangle) : Prop := sorry -- Define the angle bisector property

def perp_bisector {triangle : Type*} (AC : set (Point triangle)) : Prop := sorry -- Define the perpendicular bisector property

def circumcircle_of_triangle {triangle : Type*} (Δ : Triangle triangle) : set (Point triangle) := sorry -- Define circumcircle

theorem intersection lies_on_circumcircle_and_perpbisector
(triangle : Type*)
(Δ : Triangle triangle)
(h1 : ∃ D : Point triangle, D ∈ circumcircle_of_triangle Δ ∧ angle_bisector Δ.B)
(h2 : ∃ D : Point triangle, perp_bisector {Δ.A, Δ.C})
: ∃ D : Point triangle, D ∈ circumcircle_of_triangle Δ ∧ perp_bisector {Δ.A, Δ.C} :=
by
  -- Proof is omitted, to be filled in with steps from the solution
  sorry

end intersection_l187_187422


namespace wheel_radius_approximation_l187_187125

noncomputable def radius_of_wheel (total_distance : ℝ) (num_revolutions : ℕ) : ℝ :=
  let circumference := total_distance / num_revolutions
  circumference / (2 * Real.pi)

theorem wheel_radius_approximation :
  radius_of_wheel 2112 1500 ≈ 0.224 :=
by
  sorry

end wheel_radius_approximation_l187_187125


namespace overall_profit_percentage_l187_187037

theorem overall_profit_percentage :
  let SP_A := 80
  let SP_B := 100
  let SP_C := 125
  let CP_A := 0.24 * SP_A
  let CP_B := 0.25 * SP_B
  let CP_C := 0.28 * SP_C
  let TCP := CP_A + CP_B + CP_C
  let TSP := SP_A + SP_B + SP_C
  let TP := TSP - TCP
  let PP := (TP / TCP) * 100
  PP = 285 := by simp [SP_A, SP_B, SP_C, CP_A, CP_B, CP_C, TCP, TSP, TP, PP]; sorry

end overall_profit_percentage_l187_187037


namespace triangle_BDF_angles_l187_187389

-- Given conditions for the convex hexagon
variables (A B C D E F : Type)
variables (AB BC CD DE EF FA : ℝ)
variables (α β γ : ℝ)
variable (h : α + β + γ = 2 * Real.pi)

-- Assuming equal sides in the hexagon
axiom AB_eq_BC : AB = BC
axiom CD_eq_DE : CD = DE
axiom EF_eq_FA : EF = FA

-- Given angles at B, D, and F
axiom angleB : ∠B = α
axiom angleD : ∠D = β
axiom angleF : ∠F = γ

-- The statement to prove
theorem triangle_BDF_angles :
  ∠BDF = α / 2 ∧ ∠DBF = β / 2 ∧ ∠DFB = γ / 2 :=
sorry

end triangle_BDF_angles_l187_187389


namespace longest_segment_DE_l187_187478

noncomputable def point := (ℝ × ℝ)

constant A : point := (-4, 0)
constant B : point := (0, 3)
constant C : point := (4, 0)
constant D : point := (0, -2)
constant E : point := (2, 2)

constant angle_BAD : ℝ := 50
constant angle_ABD : ℝ := 60
constant angle_BDE : ℝ := 70
constant angle_EDB : ℝ := 80
constant angle_DEC : ℝ := 90
constant angle_ECD : ℝ := 30

theorem longest_segment_DE : 
  segment_length A B < segment_length B D ∧ 
  segment_length B D < segment_length D E ∧ 
  segment_length D E > segment_length E C ∧ 
  segment_length D E > segment_length C A :=
sorry

end longest_segment_DE_l187_187478


namespace evaluate_64_pow_fifth_sixth_l187_187218

theorem evaluate_64_pow_fifth_sixth : 64 ^ (5 / 6) = 32 := by
  have h : 64 = 2 ^ 6 := by sorry
  calc 64 ^ (5 / 6) = (2 ^ 6) ^ (5 / 6) : by rw [h]
              ...   = 2 ^ (6 * (5 / 6))  : by sorry
              ...   = 2 ^ 5              : by sorry
              ...   = 32                 : by sorry

end evaluate_64_pow_fifth_sixth_l187_187218


namespace solve_for_x_l187_187547

theorem solve_for_x (x : ℝ) (h : 144 / 0.144 = 14.4 / x) : x = 0.0144 := 
by
  sorry

end solve_for_x_l187_187547


namespace cos_alpha_plus_beta_l187_187319

variable {α β : ℝ}

theorem cos_alpha_plus_beta (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : (sin (2 * α)) / ((cos (2 * α)) - 1) = (1 - tan β) / (1 + tan β)) :
  cos (α + β) = - (Real.sqrt 2 / 2) :=
by
  sorry

end cos_alpha_plus_beta_l187_187319


namespace least_positive_linear_combination_l187_187715

theorem least_positive_linear_combination :
  ∃ x y z : ℤ, 0 < 24 * x + 20 * y + 12 * z ∧ ∀ n : ℤ, (∃ x y z : ℤ, n = 24 * x + 20 * y + 12 * z) → 0 < n → 4 ≤ n :=
by
  sorry

end least_positive_linear_combination_l187_187715


namespace collinear_P_H_Q_l187_187784

variables {A B C P Q H: Type} [triangle_with_orthocenter A B C H] 
          {ω : circle (diameter B C)}
          (tangent_AP : tangent (line_through A P) ω)
          (tangent_AQ : tangent (line_through A Q) ω)

theorem collinear_P_H_Q :
  collinear ({P, H, Q} : set Point) :=
by
  sorry

end collinear_P_H_Q_l187_187784


namespace rectangle_diagonal_length_l187_187486

theorem rectangle_diagonal_length (P : ℝ) (L W D : ℝ) 
  (hP : P = 72) 
  (h_ratio : 3 * W = 2 * L) 
  (h_perimeter : 2 * (L + W) = P) :
  D = Real.sqrt (L * L + W * W) :=
sorry

end rectangle_diagonal_length_l187_187486


namespace principal_calculation_l187_187109

-- Define the given conditions
def rate : ℝ := 13
def time : ℝ := 3
def simple_interest : ℝ := 5400

-- Define the principal amount to be proved
def principal_amount : ℝ := 13846.15

-- Define the formula for simple interest
def simple_interest_formula (P R T : ℝ) : ℝ := (P * R * T) / 100

-- State the theorem to prove the principal amount
theorem principal_calculation (SI R T : ℝ) (h : SI = simple_interest_formula principal_amount R T) :
  SI = 5400 → R = 13 → T = 3 → principal_amount = 13846.15 :=
by
  intros h1 h2 h3
  sorry

end principal_calculation_l187_187109


namespace percentage_of_boys_from_schoolA_who_study_science_l187_187744

-- Define the conditions
def total_boys := 150
def perc_schoolA := 0.20
def boys_no_science := 21

-- Define the theorem
theorem percentage_of_boys_from_schoolA_who_study_science :
  let boys_from_schoolA := total_boys * perc_schoolA
  let boys_study_science := boys_from_schoolA - boys_no_science in
  (boys_study_science / boys_from_schoolA) * 100 = 30 :=
by
  sorry

end percentage_of_boys_from_schoolA_who_study_science_l187_187744


namespace roots_opposite_numbers_l187_187657

-- Define the polynomial P
def P (k : ℝ) : ℝ → ℝ := λ x => x^2 + (k^2 - 4) * x + k - 1

-- Define the statement that if the roots are opposite numbers, then k = -2
theorem roots_opposite_numbers (k : ℝ) :
  (∃ x1 x2 : ℝ, P(k) x1 = 0 ∧ P(k) x2 = 0 ∧ x1 = -x2) → k = -2 :=
by
  intros h
  sorry

end roots_opposite_numbers_l187_187657


namespace num_players_with_exactly_two_wins_l187_187509

variables {Player : Type} [fintype Player] (rel : Player → Player → Prop)
  (cond1 : fintype.card Player = 10)
  (cond2 : ∀ x y, x ≠ y → rel x y ∨ rel y x)
  (cond3 : ∀ S : finset Player, S.card = 5 → (∃ p ∈ S, ∀ q ∈ S, q ≠ p → rel p q) ∧ (∃ q ∈ S, ∀ p ∈ S, p ≠ q → rel p q))

theorem num_players_with_exactly_two_wins : ∃! p : Player, 
  (∃ s : finset Player, s.card = 2 ∧ (∀ q ∈ s, rel p q ∧ q ≠ p)) :=
sorry

end num_players_with_exactly_two_wins_l187_187509


namespace exam_rule_l187_187448

variable (P R Q : Prop)

theorem exam_rule (hp : P ∧ R → Q) : ¬ Q → ¬ P ∨ ¬ R :=
by
  sorry

end exam_rule_l187_187448


namespace solve_system_eq_l187_187886

theorem solve_system_eq (x y : ℝ) (h1 : x - y = 1) (h2 : 2 * x + 3 * y = 7) :
  x = 2 ∧ y = 1 := by
  sorry

end solve_system_eq_l187_187886


namespace volume_of_cube_is_9261_l187_187906

-- Conditions in Lean
def cost_painting_rupees : ℝ := 343.98
def rate_painting_paise_per_sq_cm : ℝ := 13
def conversion_rate_paise_per_rupee : ℝ := 100

-- Given the conditions, prove that: the volume of the cube is 9261 cubic cm
theorem volume_of_cube_is_9261 :
  (∀ (cost_painting_rupees rate_painting_paise_per_sq_cm conversion_rate_paise_per_rupee : ℝ),
    let total_cost_paise := cost_painting_rupees * conversion_rate_paise_per_rupee in
    let total_surface_area := total_cost_paise / rate_painting_paise_per_sq_cm in
    let area_one_face := total_surface_area / 6 in
    let side_length := real.sqrt area_one_face in
    let volume := side_length^3 in
    volume = 9261) :=
sorry

end volume_of_cube_is_9261_l187_187906


namespace evaluate_64_pow_fifth_sixth_l187_187216

theorem evaluate_64_pow_fifth_sixth : 64 ^ (5 / 6) = 32 := by
  have h : 64 = 2 ^ 6 := by sorry
  calc 64 ^ (5 / 6) = (2 ^ 6) ^ (5 / 6) : by rw [h]
              ...   = 2 ^ (6 * (5 / 6))  : by sorry
              ...   = 2 ^ 5              : by sorry
              ...   = 32                 : by sorry

end evaluate_64_pow_fifth_sixth_l187_187216


namespace Juanita_spends_more_l187_187359

def Grant_annual_spend : ℝ := 200
def weekday_spend : ℝ := 0.50
def sunday_spend : ℝ := 2.00
def weeks_in_year : ℝ := 52

def Juanita_weekly_spend : ℝ := (6 * weekday_spend) + sunday_spend
def Juanita_annual_spend : ℝ := weeks_in_year * Juanita_weekly_spend
def spending_difference : ℝ := Juanita_annual_spend - Grant_annual_spend

theorem Juanita_spends_more : spending_difference = 60 := by
  sorry

end Juanita_spends_more_l187_187359


namespace math_problem_proof_l187_187010

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ (∀ n, a (n + 1) = 2019 * a n + 1)

def inequality (a : ℕ → ℕ) (x : ℕ → ℝ) : Prop :=
  x 1 = a 2019 ∧ x 2019 = a 1 ∧
  (∑ k in Finset.range 2018, (x (k + 2) - 2019 * x (k + 1) - 1)^2) ≥
  (∑ k in Finset.range 2018, (a (2019 - k) - 2019 * a (2020 - k) - 1)^2)

theorem math_problem_proof (a : ℕ → ℕ) (x : ℕ → ℝ) 
  (h : sequence a) : inequality a x :=
sorry

end math_problem_proof_l187_187010


namespace gcd_largest_divisor_l187_187025

theorem gcd_largest_divisor (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 1008) : 
  ∃ d, nat.gcd a b = d ∧ d = 504 :=
begin
  sorry
end

end gcd_largest_divisor_l187_187025


namespace probability_gt_two_in_die_throw_l187_187548

theorem probability_gt_two_in_die_throw : 
  ∀ (faces: Finset ℕ), faces = {1, 2, 3, 4, 5, 6} →
  (∑ x in faces, if x > 2 then 1 else 0 : ℚ) / (∑ _ in faces, 1 : ℚ) = 2 / 3 := 
by
  intro faces h_faces
  sorry

end probability_gt_two_in_die_throw_l187_187548


namespace derivative_of_f_l187_187281

noncomputable def C : ℝ := Real.sin (Real.tan (1 / 7))
def f (x : ℝ) : ℝ := (C * (Real.cos (16 * x))^2) / (32 * (Real.sin (32 * x)))

theorem derivative_of_f (x : ℝ) :
  deriv f x = - C / (4 * (Real.sin (16 * x))^2) := by
  sorry

end derivative_of_f_l187_187281


namespace sum_powers_mod_m_l187_187429

theorem sum_powers_mod_m
  (n r : ℤ) (a x : Fin n.succ → ℤ)
  (h1 : n ≥ 2)
  (h2 : r ≥ 2)
  (h3 : ∀ k : ℤ, 1 ≤ k ∧ k ≤ r → (Finset.univ.sum (λ j : Fin n.succ, a j * (x j) ^ k) = 0))
  (m : ℤ) (hm : m ∈ Finset.range (2 * r + 2) \ Finset.range (r + 1)) :
  Finset.univ.sum (λ j : Fin n.succ, a j * (x j) ^ m) ≡ 0 [ZMOD m] := 
sorry

end sum_powers_mod_m_l187_187429


namespace teaching_arrangements_l187_187752

theorem teaching_arrangements : 
  let teachers := ["A", "B", "C", "D", "E", "F"]
  let lessons := ["L1", "L2", "L3", "L4"]
  let valid_first_lesson := ["A", "B"]
  let valid_fourth_lesson := ["A", "C"]
  ∃ arrangements : ℕ, 
    (arrangements = 36) ∧
    (∀ (l1 l2 l3 l4 : String), (l1 ∈ valid_first_lesson) → (l4 ∈ valid_fourth_lesson) → 
      (l2 ≠ l1 ∧ l2 ≠ l4 ∧ l3 ≠ l1 ∧ l3 ≠ l4) ∧ 
      (List.length teachers - (if (l1 == "A") then 1 else 0) - (if (l4 == "A") then 1 else 0) = 4)) :=
by {
  -- This is just the theorem statement; no proof is required.
  sorry
}

end teaching_arrangements_l187_187752


namespace evaluate_pow_l187_187252

theorem evaluate_pow : 64^(5/6 : ℝ) = 32 := by
  sorry

end evaluate_pow_l187_187252


namespace hyperbola_focus_cosine_l187_187687

theorem hyperbola_focus_cosine (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
  (h₂ : 5 = (real.sqrt (a^2 + b^2) / a)) :
  ∃ P : ℝ × ℝ, (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧ P.1 < 0 ∧ P.2 > 0 ∧ (cos (angle (P, (5*a, 0), (-5*a, 0))) = 4 / 5) :=
by patch sorry

end hyperbola_focus_cosine_l187_187687


namespace product_of_solutions_product_of_all_values_of_t_l187_187634

theorem product_of_solutions (t : ℤ) (h : t^2 = 36) : t = 6 ∨ t = -6 :=
by sorry

theorem product_of_all_values_of_t : (∏ t in {t : ℤ | t^2 = 36}, t) = -36 :=
by sorry

end product_of_solutions_product_of_all_values_of_t_l187_187634


namespace eval_power_l187_187204

-- Given condition
def sixty_four : ℕ := 64

-- Given condition rewritten in Lean
def sixty_four_as_two_powersix : sixty_four = 2^6 := by
  sorry

-- Prove that 64^(5/6) = 32
theorem eval_power : real.exp (5/6 * real.log 64) = 32 := by
  have h1 : 64 = 2^6 := sixty_four_as_two_powersix
  sorry

end eval_power_l187_187204


namespace not_all_terms_positive_l187_187884

variable (a b c d : ℝ)
variable (e f g h : ℝ)

theorem not_all_terms_positive
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (he : e < 0) (hf : f < 0) (hg : g < 0) (hh : h < 0) :
  ¬ ((a * e + b * c > 0) ∧ (e * f + c * g > 0) ∧ (f * d + g * h > 0) ∧ (d * a + h * b > 0)) :=
sorry

end not_all_terms_positive_l187_187884


namespace count_sortable_permutations_l187_187120

-- Define the general condition
def is_fixed (p : ℕ → ℕ) (i : ℕ) : Prop :=
  p i ≤ p (i + 1)

def fix_permutation (p : ℕ → ℕ) (n : ℕ) :=
  ∀ a, 1 ≤ a ∧ a ≤ n - 1 → ∀ i, a ≤ i ∧ i ≤ n - 1 → is_fixed p i

-- Define the specific problem statement
theorem count_sortable_permutations : 
  let n := 2018 
  in (∀ p : ℕ → ℕ, (∀ i, 1 ≤ p i ∧ p i ≤ n) → fix_permutation p n → p = id) →
  (1009.factorial * 1010.factorial) :=
by
  sorry

end count_sortable_permutations_l187_187120


namespace correct_option_B_l187_187403

-- Assigning the variables
variables (y : ℝ)

-- The Lean statement of the problem
theorem correct_option_B : 
  (-2 * y^3) * (-y) = 2 * y^4 :=
by 
  have h1 : (-2 * y^3) * (-y) = (-2) * y^3 * (-y),
  ring,
  sorry

end correct_option_B_l187_187403


namespace proof_problem_l187_187321

variable {ℝ : Type} [LinearOrderedField ℝ]

-- Definitions for the problem conditions

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g (x)
def increasing_on_pos (f' : ℝ → ℝ) : Prop := ∀ x > 0, 0 < f' x
def decreasing_on_neg (g' : ℝ → ℝ) : Prop := ∀ x > 0, 0 < g' (-x)

-- Mathematically equivalent proof problem in Lean

theorem proof_problem
  {f g : ℝ → ℝ}
  {f' g' : ℝ → ℝ}
  (H1 : is_odd f)
  (H2 : is_even g)
  (H3 : increasing_on_pos f')
  (H4 : decreasing_on_neg g') :
  ∀ x < 0, 0 < f' x ∧ g' (-x) < 0 := 
by
  sorry


end proof_problem_l187_187321


namespace tan_30_deg_l187_187164

theorem tan_30_deg : 
  let θ := 30 * (Float.pi / 180) in  -- Conversion from degrees to radians
  Float.sin θ = 1 / 2 ∧ Float.cos θ = Float.sqrt 3 / 2 →
  Float.tan θ = Float.sqrt 3 / 3 := by
  intro h
  sorry

end tan_30_deg_l187_187164


namespace problem1_problem2_l187_187325

/-- Given that the distance from point M to point F(4,0) is 2 less than its distance to the line l: x = -6, 
    prove that the trajectory of point M is described by the parabola y^2 = 16x -/
theorem problem1 (M F : ℝ × ℝ) (l : ℝ → Prop) (h1 : ∀ M : ℝ × ℝ, ∃ F : ℝ × ℝ, (dist M F) = 2 + (dist M (λ x : ℝ, l (x + 6)))) :
  (∃ x y : ℝ, y^2 = 16 * x) :=
sorry

/-- Given that OA and OB are chords on the parabola y^2 = 16x that are perpendicular to each other,
    prove that the line AB passes through a fixed point on the x-axis, say (16,0) -/
theorem problem2 (A B : ℝ × ℝ) (h2 : ∃ OA OB : ℝ × ℝ, 
  (OA.2^2 = 16 * OA.1 ∧ OB.2^2 = 16 * OB.1) ∧ OA.1 * OB.1 + OA.2 * OB.2 = 0) :
  ∃ x0 : ℝ, ∀ A B : ℝ × ℝ, ( ( (A.2^2 = 16 * A.1 ∧ B.2^2 = 16 * B.1) ∧ A.1 * B.1 + A.2 * B.2 = 0) → (x0 = 16 ∧ x0 ∈ ℝ) := 
sorry

end problem1_problem2_l187_187325


namespace evaluate_64_pow_5_div_6_l187_187268

theorem evaluate_64_pow_5_div_6 : (64 : ℝ) ^ (5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ) ^ 6 := by norm_num
  have h2 : (64 : ℝ) ^ (5 / 6) = ((2 : ℝ) ^ 6) ^ (5 / 6) := by rw h1
  have h3 : ((2 : ℝ) ^ 6) ^ (5 / 6) = (2 : ℝ) ^ (6 * (5 / 6)) := by rw [Real.rpow_mul]
  have h4 : (2 : ℝ) ^ (6 * (5 / 6)) = (2 : ℝ) ^ 5 := by norm_num
  rw [h2, h3, h4]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187268


namespace midpoint_of_AB_l187_187840

open EuclideanGeometry

-- Define the two circles intersecting at points A and B
variables {P : Type*} [metric_space P] [normed_add_torsor ℝ P]
variables (C1 C2 : Circle P) (A B M N I : P)

-- Indicating M lies on C1 and N lies on C2, and I is the intersection of lines MN and AB
variables (hAC1 : A ∈ C1) (hBC1 : B ∈ C1) (hMC1 : M ∈ C1)
variables (hAC2 : A ∈ C2) (hBC2 : B ∈ C2) (hNC2 : N ∈ C2)
variables (hMN : tangent C1 C2 M N) (hI : collinear {I, M, N} ∨ collinear {I, A, B})

theorem midpoint_of_AB : midpoint I A B :=
sorry

end midpoint_of_AB_l187_187840


namespace max_distance_AB_tangent_l187_187692

noncomputable def max_distance_AB (a b R : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : b < R) (h4 : R < a)
  (ellipse : ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)
  (circle : ∀ (x y : ℝ), x^2 + y^2 = R^2) :
  ℝ :=
a - b

theorem max_distance_AB_tangent
  {a b R : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : b < R) (h4 : R < a)
  (ellipse : ∀ (x y : ℝ), (a ≠ 0 → b ≠ 0) → x^2 / a^2 + y^2 / b^2 = 1)
  (circle : ∀ (x y : ℝ), R ≠ 0 → x^2 + y^2 = R^2)
  (tangent : ∀ A B : ℝ × ℝ, ellipse A.1 A.2 (λ ha hb, true) → circle B.1 B.2 (λ hR, true) → 
  ∃ k m : ℝ, B.2 = k * B.1 + m ∧ A.2 = k * A.1 + m ∧  -- Line equation
  4 * k^2 * m^2 = 4 * (1 + k^2) * (m^2 - R^2) ∧
  4 * (k * m * a^2)^2 - 4 * (a^2 * k^2 + b^2) * a^2 * (m^2 - b^2) = 0 ): 
  max_distance_AB a b R h1 h2 h3 h4 ellipse circle = a - b :=
sorry

end max_distance_AB_tangent_l187_187692


namespace smallest_n_for_4n_square_and_5n_cube_l187_187050

theorem smallest_n_for_4n_square_and_5n_cube :
  ∃ (n : ℕ), (n > 0 ∧ (∃ k : ℕ, 4 * n = k^2) ∧ (∃ m : ℕ, 5 * n = m^3)) ∧ n = 400 :=
by
  sorry

end smallest_n_for_4n_square_and_5n_cube_l187_187050


namespace negation_of_prop_l187_187706

variable (x : ℝ)
def prop (x : ℝ) := x ∈ Set.Ici 0 → Real.exp x ≥ 1

theorem negation_of_prop :
  (¬ ∀ x ∈ Set.Ici 0, Real.exp x ≥ 1) = ∃ x ∈ Set.Ici 0, Real.exp x < 1 :=
by
  sorry

end negation_of_prop_l187_187706


namespace false_weight_approx_l187_187999

theorem false_weight_approx (profit_percent : ℝ) (true_weight : ℝ) (false_weight : ℝ):
  profit_percent = 4.166666666666674 / 100 → 
  true_weight = 1000 →
  false_weight = true_weight / (1 + 4.166666666666674 / 100) →
  false_weight ≈ 960 :=
by
  intros h_profit h_true_weight h_false_weight
  sorry

end false_weight_approx_l187_187999


namespace correct_calculation_l187_187080

theorem correct_calculation (A : 3 + 2 * Real.sqrt 3 ≠ 5 * Real.sqrt 3)
                            (B : Real.sqrt 3 / Real.sqrt 5 = Real.sqrt (3 * 5) / 5)
                            (C : 5 * Real.sqrt 3 * 2 * Real.sqrt 3 ≠ 10 * Real.sqrt 3)
                            (D : 4 * Real.sqrt 3 - 3 * Real.sqrt 3 ≠ 1) :
  B := sorry

end correct_calculation_l187_187080


namespace evaluate_64_pow_fifth_sixth_l187_187221

theorem evaluate_64_pow_fifth_sixth : 64 ^ (5 / 6) = 32 := by
  have h : 64 = 2 ^ 6 := by sorry
  calc 64 ^ (5 / 6) = (2 ^ 6) ^ (5 / 6) : by rw [h]
              ...   = 2 ^ (6 * (5 / 6))  : by sorry
              ...   = 2 ^ 5              : by sorry
              ...   = 32                 : by sorry

end evaluate_64_pow_fifth_sixth_l187_187221


namespace area_of_triangle_ABC_l187_187311

-- Definitions of sides and conditions given in the problem
variables (A B C : Type) (AB : ℝ) (BC : ℝ) (angle_C : ℝ)
variables (sin_C : ℝ) (cos_C : ℝ)

-- Conditions from the problem
axiom AB_Condition : AB = real.sqrt 3
axiom BC_Condition : BC = 1
axiom SinCos_Relation : sin_C = real.sqrt 3 * cos_C

-- The area of triangle ABC
noncomputable def area_of_triangle : ℝ := 
  (1 / 2) * 2 * 1 * (real.sqrt 3 / 2)

-- The proof statement
theorem area_of_triangle_ABC (AB : A -> B -> C)
  (AB_Condition : AB = real.sqrt 3)
  (BC_Condition : BC = 1)
  (SinCos_Relation : sin_C = real.sqrt 3 * cos_C)
  : 
  area_of_triangle = real.sqrt 3 / 2 :=
sorry

end area_of_triangle_ABC_l187_187311


namespace like_terms_sum_l187_187725

theorem like_terms_sum (m n : ℕ) (h1 : m = 3) (h2 : 4 = n + 2) : m + n = 5 :=
by
  sorry

end like_terms_sum_l187_187725


namespace family_reunion_weight_gain_l187_187137

def orlando_gained : ℕ := 5

def jose_gained (orlando: ℕ) : ℕ := 2 * orlando + 2

def fernando_gained (jose: ℕ) : ℕ := jose / 2 - 3

def total_weight_gained : ℕ := 
  let orlando := orlando_gained in
  let jose := jose_gained orlando in
  let fernando := fernando_gained jose in
  orlando + jose + fernando

theorem family_reunion_weight_gain : total_weight_gained = 20 := by
  sorry

end family_reunion_weight_gain_l187_187137


namespace dartboard_even_score_probability_l187_187404

open Real

/-- A structure defining a dartboard region with a specific radius and point value -/
structure DartboardRegion where
  radius : ℝ
  point_value : ℕ

/-- Conditions of the problem defined as data -/
def outer_circle_radius : ℝ := 9
def inner_circle_radius : ℝ := 4
def inner_regions : List ℕ := [3, 5, 5]
def outer_regions : List ℕ := [5, 3, 3]

/-- Calculate Areas of the Regions -/
def area_of_circle (r : ℝ) : ℝ := π * r ^ 2

def inner_circle_area : ℝ := area_of_circle inner_circle_radius
def outer_circle_area : ℝ := area_of_circle outer_circle_radius
def outer_ring_area : ℝ := outer_circle_area - inner_circle_area
def inner_region_area : ℝ := inner_circle_area / 3
def outer_region_area : ℝ := outer_ring_area / 3

/-- Given the areas and probabilities, define the probability that the score is even 
when two darts are thrown -/
def probability_even_score : ℚ :=
(let prob_even := (inner_region_area + 2 * outer_region_area) / outer_circle_area
    prob_odd := 1 - prob_even 
 in prob_even ^ 2 + prob_odd ^ 2)

theorem dartboard_even_score_probability : probability_even_score = 30725 / 59049 :=
by
  -- Proof details would go here
  sorry

end dartboard_even_score_probability_l187_187404


namespace probability_triplet_1_2_3_in_10_rolls_l187_187964

noncomputable def probability_of_triplet (n : ℕ) : ℝ :=
  let A0 := (6^10 : ℝ)
  let A1 := (8 * 6^7 : ℝ)
  let A2 := (15 * 6^4 : ℝ)
  let A3 := (4 * 6 : ℝ)
  let total := A0
  let p := (A0 - (total - (A1 - A2 + A3))) / total
  p

theorem probability_triplet_1_2_3_in_10_rolls : 
  abs (probability_of_triplet 10 - 0.0367) < 0.0001 :=
by
  sorry

end probability_triplet_1_2_3_in_10_rolls_l187_187964


namespace win_lottery_amount_l187_187538

theorem win_lottery_amount (W : ℝ) (cond1 : W * 0.20 + 5 = 35) : W = 50 := by
  sorry

end win_lottery_amount_l187_187538


namespace arithmetic_mean_odd_primes_lt_30_l187_187081

theorem arithmetic_mean_odd_primes_lt_30 : 
  (3 + 5 + 7 + 11 + 13 + 17 + 19 + 23 + 29) / 9 = 14 :=
by
  sorry

end arithmetic_mean_odd_primes_lt_30_l187_187081


namespace sophia_can_create_8_different_squares_l187_187467

-- Definitions of conditions
def is_isosceles_right_triangle (t : Type) : Prop :=
  ∃ a, t = (a, a, a * sqrt 2)

def can_create_square (triangles : ℕ) (sizes : ℕ) : Prop :=
  sizes = 8

-- Theorem statement
theorem sophia_can_create_8_different_squares :
  ∀ (triangles : ℕ), triangles = 52 → can_create_square triangles 8 :=
by
  intros triangles h
  rw h
  exact sorry

end sophia_can_create_8_different_squares_l187_187467


namespace disc_distinct_colorings_l187_187996

open Nat

-- Definitions for given conditions
def sectors : Nat := 12
def colors : Nat := 6

-- Euler's Totient Function
def euler_totient (n : Nat) : Nat :=
  (Finset.range (n + 1)).filter (λ i => gcd n i = 1).card

-- Burnside's lemma calculation
noncomputable def num_distinct_colorings : Nat :=
  let sum := (Finset.divisors sectors).sum (λ d =>
    euler_totient d * ((colors - 1) ^ (sectors / d) + if d % 2 = 0 then -1 else (colors - 1)))
  (1 / sectors) * sum

theorem disc_distinct_colorings : num_distinct_colorings sectors colors = 20346485 := by
  sorry

end disc_distinct_colorings_l187_187996


namespace pencils_ratio_l187_187959

theorem pencils_ratio (T S Ti : ℕ) 
  (h1 : T = 6 * S)
  (h2 : T = 12)
  (h3 : Ti = 16) : Ti / S = 8 := by
  sorry

end pencils_ratio_l187_187959


namespace min_radius_of_circumcircle_l187_187124

theorem min_radius_of_circumcircle {a b : ℝ} (ha : a = 3) (hb : b = 4) : 
∃ R : ℝ, R = 2.5 ∧ (∃ c : ℝ, c = Real.sqrt (a^2 + b^2) ∧ a^2 + b^2 = c^2 ∧ 2 * R = c) :=
by 
  sorry

end min_radius_of_circumcircle_l187_187124


namespace rod_segment_weight_l187_187398

variables (a_1 a_2 a_10 : ℝ) (M : ℝ) 

def weight_sequence := ∀ i : ℕ, 1 ≤ i ∧ i ≤ 10 → a_1 + (i - 1) * ((a_10 - a_1) / 9)

theorem rod_segment_weight (h1 : a_1 + a_10 = 4) 
                          (h2 : a_1 + a_2 = 2)
                          (hM : M = 10 * a_1 + 10 * 9 / 2 * ((a_10 - a_1) / 9))
                          (h48ai : ∀ i : ℕ, 1 ≤ i ∧ i ≤ 10 → 48 * (a_1 + (i - 1) * ((a_10 - a_1) / 9)) = 5 * M) :
  (∃ i : ℕ, 1 ≤ i ∧ i ≤ 10 ∧ 48 * (a_1 + (i - 1) * ((a_10 - a_1) / 9)) = 75) :=
begin
  use 6,
  sorry
end

end rod_segment_weight_l187_187398


namespace number_of_six_digit_integers_l187_187364

-- Define the problem conditions
def digits := [1, 1, 3, 3, 7, 8]

-- State the theorem
theorem number_of_six_digit_integers : 
  (List.permutations digits).length = 180 := 
by sorry

end number_of_six_digit_integers_l187_187364


namespace PB_distance_eq_l187_187089

theorem PB_distance_eq {
  A B C D P : Type
} (PA PD PC : ℝ) (hPA: PA = 6) (hPD: PD = 8) (hPC: PC = 10)
  (h_equidistant: ∃ y : ℝ, PA^2 + y^2 = PB^2 ∧ PD^2 + y^2 = PC^2) :
  ∃ PB : ℝ, PB = 6 * Real.sqrt 2 := 
by
  sorry

end PB_distance_eq_l187_187089


namespace probability_of_dime_l187_187105

noncomputable def value_of_dimes : ℝ := 8.0
noncomputable def value_of_nickels : ℝ := 7.0
noncomputable def value_of_pennies : ℝ := 5.0
noncomputable def value_per_dime : ℝ := 0.10
noncomputable def value_per_nickel : ℝ := 0.05
noncomputable def value_per_penny : ℝ := 0.01

noncomputable def number_of_dimes : ℕ := (value_of_dimes / value_per_dime).to_nat
noncomputable def number_of_nickels : ℕ := (value_of_nickels / value_per_nickel).to_nat
noncomputable def number_of_pennies : ℕ := (value_of_pennies / value_per_penny).to_nat

noncomputable def total_number_of_coins : ℕ := number_of_dimes + number_of_nickels + number_of_pennies

theorem probability_of_dime : (number_of_dimes : ℝ) / (total_number_of_coins : ℝ) = 1 / 9 := by
  sorry

end probability_of_dime_l187_187105


namespace order_four_packages_l187_187031

theorem order_four_packages (m : Fin 4 → ℝ) (h_distinct : ∀ i j, i ≠ j → m i ≠ m j) :
  ∃ (order : Fin 4 → Fin 4), 
    (∀ i j, i < j → m (order i) < m (order j)) ∧
    (∃ n ≤ 5, ∀ k, k < n → (∃ i j, i ≠ j ∧ (m (order i) = m i ∧ m (order j) = m j))) :=
begin
  sorry
end

end order_four_packages_l187_187031


namespace num_positive_integers_l187_187656

theorem num_positive_integers (n : ℕ) :
    (0 < n ∧ n < 40 ∧ ∃ k : ℕ, k > 0 ∧ n = 40 * k / (k + 1)) ↔ 
    (n = 20 ∨ n = 30 ∨ n = 32 ∨ n = 35 ∨ n = 36 ∨ n = 38 ∨ n = 39) :=
sorry

end num_positive_integers_l187_187656


namespace evaluate_64_pow_fifth_sixth_l187_187213

theorem evaluate_64_pow_fifth_sixth : 64 ^ (5 / 6) = 32 := by
  have h : 64 = 2 ^ 6 := by sorry
  calc 64 ^ (5 / 6) = (2 ^ 6) ^ (5 / 6) : by rw [h]
              ...   = 2 ^ (6 * (5 / 6))  : by sorry
              ...   = 2 ^ 5              : by sorry
              ...   = 32                 : by sorry

end evaluate_64_pow_fifth_sixth_l187_187213


namespace distance_between_parallel_lines_l187_187705

-- Define the equations of the lines
def l1 (x y : ℝ) := 2 * x + y - 1 = 0
def l2 (x y : ℝ) := 2 * x + y + 1 = 0

-- Define A, B, C1, C2
def A : ℝ := 2
def B : ℝ := 1
def C1 : ℝ := -1
def C2 : ℝ := 1

-- Prove the distance between the lines
theorem distance_between_parallel_lines : 
  (|C2 - C1| / real.sqrt (A^2 + B^2)) = (2 * real.sqrt 5 / 5) := 
  by
  sorry

end distance_between_parallel_lines_l187_187705


namespace eval_power_l187_187205

-- Given condition
def sixty_four : ℕ := 64

-- Given condition rewritten in Lean
def sixty_four_as_two_powersix : sixty_four = 2^6 := by
  sorry

-- Prove that 64^(5/6) = 32
theorem eval_power : real.exp (5/6 * real.log 64) = 32 := by
  have h1 : 64 = 2^6 := sixty_four_as_two_powersix
  sorry

end eval_power_l187_187205


namespace win_loss_ratio_l187_187597

theorem win_loss_ratio (won lost : ℕ) (hw : won = 28) (hl : lost = 7) : won / lost = 4 :=
by {
  rw [hw, hl],
  norm_num,
  sorry
}

end win_loss_ratio_l187_187597


namespace max_distance_from_circle_to_line_l187_187427

noncomputable def distance_from_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  (abs (a * p.1 + b * p.2 + c)) / (Real.sqrt (a^2 + b^2))

theorem max_distance_from_circle_to_line :
  let center := (2 : ℝ, 2 : ℝ)
  let radius := 1
  let line := (1, -1, -5)  -- Corresponding to 'x - y - 5 = 0'
  let circle_equation := (x^2 + y^2 - 4 * x - 4 * y + 7 = 0)
  ∀ A : ℝ × ℝ,
    A ∈ ({ p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 2)^2 = radius^2 }) →
    (distance_from_point_to_line center line.1 line.2 line.3) + radius = (5 * Real.sqrt 2 / 2) + 1 := 
by
  sorry

end max_distance_from_circle_to_line_l187_187427


namespace jordan_Oreos_count_l187_187768

variable (J : ℕ)
variable (OreosTotal : ℕ)
variable (JamesOreos : ℕ)

axiom James_Oreos_condition : JamesOreos = 2 * J + 3
axiom Oreos_total_condition : J + JamesOreos = OreosTotal
axiom Oreos_total_value : OreosTotal = 36

theorem jordan_Oreos_count : J = 11 :=
by 
  unfold OreosTotal JamesOreos
  sorry

end jordan_Oreos_count_l187_187768


namespace problem_f_minus3_log2_3_l187_187425

def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (1 - x) / Real.log 2 else 2 ^ x

theorem problem_f_minus3_log2_3 :
  f (-3) + f (Real.log 3 / Real.log 2) = 5 := by
  sorry

end problem_f_minus3_log2_3_l187_187425


namespace total_cost_of_goods_l187_187474

theorem total_cost_of_goods :
  let F := 20.50 in
  let R := (6 * F) / 2 in
  let M := R / 10 in
  (4 * M) + (3 * R) + (5 * F) = 311.60 :=
by
  sorry

end total_cost_of_goods_l187_187474


namespace least_subtraction_for_divisibility_l187_187546

def original_number : ℕ := 5474827

def required_subtraction : ℕ := 7

theorem least_subtraction_for_divisibility :
  ∃ k : ℕ, (original_number - required_subtraction) = 12 * k :=
sorry

end least_subtraction_for_divisibility_l187_187546


namespace line_BC_eq_altitude_A_eq_l187_187352

section
  open Real

  -- Define the vertices of the triangle
  def A := (-4: ℝ, 0: ℝ)
  def B := (0: ℝ, -3: ℝ)
  def C := (-2: ℝ, 1: ℝ)

  -- Define the slope of a line given two points
  def slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)

  -- Define the equation of a line in the form ax + by + c = 0
  def line_eq (a b c : ℝ) (P : ℝ × ℝ) : Prop := a * P.1 + b * P.2 + c = 0

  -- Prove the equation of line BC
  theorem line_BC_eq :
    line_eq 2 1 3 B ∧ line_eq 2 1 3 C :=
  sorry

  -- Prove the equation of the altitude from A to BC
  theorem altitude_A_eq :
    line_eq 1 -2 4 A ∧ line_eq 1 -2 4 (C.1, (1 / 2) * C.1 + 2) :=
  sorry

end

end line_BC_eq_altitude_A_eq_l187_187352


namespace probability_of_sum_15_l187_187850

open Set

def s : Set ℕ := {2, 3, 4, 5, 9, 12}
def b : Set ℕ := {4, 5, 6, 7, 8, 11, 14}

def is_odd (n : ℕ) := n % 2 = 1
def is_even (n : ℕ) := n % 2 = 0

def pairs (s b : Set ℕ) := { (x, y) | x ∈ s ∧ y ∈ b }

def valid_pairs (pair : ℕ × ℕ) := 
  let (x, y) := pair
  in (is_odd x ∧ is_even y ∨ is_even x ∧ is_odd y) ∧ x + y = 15

theorem probability_of_sum_15 :
  (∑ p in (pairs s b).to_finset, if valid_pairs p then 1 else 0).to_nat / ((pairs s b).to_finset.card) = 2 / 21 := 
sorry

end probability_of_sum_15_l187_187850


namespace probability_f_l187_187320

open ProbabilityTheory

variables {Ω : Type*} [ProbabilitySpace Ω] (e f : Event Ω)

theorem probability_f (h1 : P(e) = 25) (h2 : P(e ∩ f) = 75) (h3 : P(e | f) = 3) : P(f) = 25 :=
by
  sorry

end probability_f_l187_187320


namespace pyramid_volume_l187_187482

theorem pyramid_volume (α l : ℝ) : 
  let SO := l * Real.sin α in
  let AO := l * Real.cos α in
  let AB := l * Real.sqrt 3 * Real.cos α in
  let base_area := (AB ^ 2 * Real.sqrt 3) / 4 in
  let volume := (1 / 3) * base_area * SO in
  volume = (Real.sqrt 3 * l^3 * Real.cos(α)^2 * Real.sin(α)) / 4 := 
by 
  sorry

end pyramid_volume_l187_187482


namespace tan_30_eq_sqrt3_div_3_l187_187180

theorem tan_30_eq_sqrt3_div_3 :
  let opposite := 1
  let adjacent := sqrt (3 : ℝ) 
  tan (real.pi / 6) = opposite / adjacent := by 
    sorry

end tan_30_eq_sqrt3_div_3_l187_187180


namespace one_cow_one_bag_in_forty_days_l187_187391

theorem one_cow_one_bag_in_forty_days
    (total_cows : ℕ)
    (total_bags : ℕ)
    (total_days : ℕ)
    (husk_consumption : total_cows * total_bags = total_cows * total_days) :
  total_days = 40 :=
by sorry

end one_cow_one_bag_in_forty_days_l187_187391


namespace train_passes_man_in_3_seconds_l187_187580

noncomputable def train_passes_man_time 
  (train_length : ℝ)
  (train_speed_kmph : ℝ)
  (man_speed_kmph : ℝ) 
  (opposite_direction : Bool) : ℝ :=
if opposite_direction then 
  let relative_speed_kmph := train_speed_kmph + man_speed_kmph
  let relative_speed_mps := relative_speed_kmph * (1000 / 3600)
  train_length / relative_speed_mps
else
  let relative_speed_kmph := train_speed_kmph - man_speed_kmph
  let relative_speed_mps := relative_speed_kmph * (1000 / 3600)
  train_length / relative_speed_mps

theorem train_passes_man_in_3_seconds : train_passes_man_time 55 60 6 true ≈ 3 := 
by 
  sorry

end train_passes_man_in_3_seconds_l187_187580


namespace smallest_number_of_students_l187_187660

def numberOfDivisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ x => x > 0 ∧ n % x = 0).length

theorem smallest_number_of_students : ∃ n : ℕ, n = 72 ∧ (numberOfDivisors n = 6) ∧ (18 ∣ n) := 
by 
  sorry

end smallest_number_of_students_l187_187660


namespace tan_30_eq_sqrt3_div_3_l187_187177

theorem tan_30_eq_sqrt3_div_3 :
  let opposite := 1
  let adjacent := sqrt (3 : ℝ) 
  tan (real.pi / 6) = opposite / adjacent := by 
    sorry

end tan_30_eq_sqrt3_div_3_l187_187177


namespace determine_abc_l187_187452

theorem determine_abc (a b c : ℕ) (h1 : a * b * c = 2^4 * 3^2 * 5^3) 
  (h2 : gcd a b = 15) (h3 : gcd a c = 5) (h4 : gcd b c = 20) : 
  a = 15 ∧ b = 60 ∧ c = 20 :=
by
  sorry

end determine_abc_l187_187452


namespace integer_values_of_a_count_integer_values_of_a_l187_187652

theorem integer_values_of_a (a : ℤ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ (x1 * x1 + a * x1 + 9 * a = 0) ∧ (x2 * x2 + a * x2 + 9 * a = 0)) →
  a ∈ {0, -12, -64} :=
by
  sorry

theorem count_integer_values_of_a : 
  {a : ℤ | ∃ x1 x2 : ℤ, x1 ≠ x2 ∧ (x1 * x1 + a * x1 + 9 * a = 0) ∧ (x2 * x2 + a * x2 + 9 * a = 0)}.to_finset.card = 3 :=
by
  sorry

end integer_values_of_a_count_integer_values_of_a_l187_187652


namespace integer_values_of_a_count_integer_values_of_a_l187_187653

theorem integer_values_of_a (a : ℤ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ (x1 * x1 + a * x1 + 9 * a = 0) ∧ (x2 * x2 + a * x2 + 9 * a = 0)) →
  a ∈ {0, -12, -64} :=
by
  sorry

theorem count_integer_values_of_a : 
  {a : ℤ | ∃ x1 x2 : ℤ, x1 ≠ x2 ∧ (x1 * x1 + a * x1 + 9 * a = 0) ∧ (x2 * x2 + a * x2 + 9 * a = 0)}.to_finset.card = 3 :=
by
  sorry

end integer_values_of_a_count_integer_values_of_a_l187_187653


namespace max_value_of_expression_l187_187807

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l187_187807


namespace value_of_5_l187_187758

def q' (q : ℤ) : ℤ := 3 * q - 3

theorem value_of_5'_prime : q' (q' 5) = 33 :=
by
  sorry

end value_of_5_l187_187758


namespace evaluate_64_pow_5_div_6_l187_187230

theorem evaluate_64_pow_5_div_6 : (64 : ℝ)^(5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ)^6 := by norm_num
  rw [← h1]
  have h2 : ((2 : ℝ)^6)^(5 / 6) = (2 : ℝ)^(6 * (5 / 6)) := by rw [Real.rpow_mul]
  rw [h2]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187230


namespace father_ate_oranges_l187_187446

theorem father_ate_oranges (initial_oranges : ℝ) (remaining_oranges : ℝ) (eaten_oranges : ℝ) : 
  initial_oranges = 77.0 → remaining_oranges = 75 → eaten_oranges = initial_oranges - remaining_oranges → eaten_oranges = 2.0 :=
by
  intros h1 h2 h3
  sorry

end father_ate_oranges_l187_187446


namespace convex_value_m_max_b_minus_a_l187_187852

noncomputable section

def f (x : ℝ) (m : ℝ) : ℝ := (1 / 12) * x^4 - (1 / 6) * m * x^3 - (3 / 2) * x^2
def f'' (x : ℝ) (m : ℝ) : ℝ := x^2 - m * x - 3

def isConvexOn (f'' : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a < x ∧ x < b → f'' x < 0

theorem convex_value_m (f'' : ℝ → ℝ) :
  isConvexOn (λ x, x^2 - 2 * x - 3) (-1) 3 → 2 = 2 :=
by
  sorry

theorem max_b_minus_a (f'' : ℝ → ℝ) (a b : ℝ) :
  (∀ m : ℝ, |m| ≤ 2 → isConvexOn (λ x, x^2 - m * x - 3) a b) → b - a = 2 :=
by
  sorry

end convex_value_m_max_b_minus_a_l187_187852


namespace exists_infinitely_many_n_for_sum_of_squares_l187_187456

theorem exists_infinitely_many_n_for_sum_of_squares:
  ∃ (n : ℕ) (infinitely_many : Prop),
    let sum_of_squares := (1^2 + 2^2 + ... + n^2) / n
    in (sum_of_squares = (k^2: ℕ)) ∧ (1 = k: ℕ)
      ∧ (3 = ∃ m : ℕ, n = 337 ∧ n = 65521) := by
  sorry

end exists_infinitely_many_n_for_sum_of_squares_l187_187456


namespace calculate_expression_l187_187602

theorem calculate_expression : 
  2 * (3 + 1) * (3^2 + 1) * (3^4 + 1) * (3^8 + 1) * (3^16 + 1) * (3^32 + 1) * (3^64 + 1) + 1 = 3^128 :=
sorry

end calculate_expression_l187_187602


namespace set_A_is_expected_set_l187_187939

-- Define the set A using the given condition
def set_A : Set ℕ := {x | (6 / (6 - x)).Nat}

-- Define the expected set in roster form
def expected_set : Set ℕ := {0, 2, 3, 4, 5}

-- The main theorem that proves the equality of the two sets
theorem set_A_is_expected_set : set_A = expected_set := 
by {
  sorry -- the proof will be skipped
}

end set_A_is_expected_set_l187_187939


namespace curve_is_circle_l187_187626

theorem curve_is_circle (r : ℝ) (r_eq_5 : r = 5) : 
  (∃ (c : ℝ), c = 0 ∧ ∀ (θ : ℝ), (r, θ) ∈ set_of (λ (p : ℝ × ℝ), p.1 = 5)) :=
sorry

end curve_is_circle_l187_187626


namespace curve_is_circle_l187_187628

theorem curve_is_circle (r : ℝ) (h : r = 5) : ∃ c : ℝ, ∀ θ : ℝ, r = c ∧ c = 5 :=
by {
  use 5,
  intro θ,
  exact ⟨h, rfl⟩,
  sorry
}

end curve_is_circle_l187_187628


namespace logarithmic_exponential_equivalence_l187_187337

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 3^x + b

theorem logarithmic_exponential_equivalence (a : ℝ) (b : ℝ) :
  (0 < a ∧ a ≠ 1) ∧
  (∀ x, ∀ y, (y = log a (x + 3) - 8/9) → (x = -2) → (y = -8/9)) ∧
  (f (-2) b = -8/9) →
  f (log 3 2) (-1) = 1 :=
by
  sorry

end logarithmic_exponential_equivalence_l187_187337


namespace max_sum_digits_digital_clock_l187_187102

theorem max_sum_digits_digital_clock : 
  ∃ (max_sum : ℕ), max_sum = 24 ∧ 
  (∀ (h m : ℕ), (h < 24) → (m < 60) → 
    max_sum ≥ nat.sum_digits h + nat.sum_digits m) :=
begin
  sorry
end

end max_sum_digits_digital_clock_l187_187102


namespace expected_value_of_die_is_475_l187_187143

-- Define the given probabilities
def prob_1 : ℚ := 1 / 12
def prob_2 : ℚ := 1 / 12
def prob_3 : ℚ := 1 / 6
def prob_4 : ℚ := 1 / 12
def prob_5 : ℚ := 1 / 12
def prob_6 : ℚ := 7 / 12

-- Define the expected value calculation
def expected_value := 
  prob_1 * 1 + prob_2 * 2 + prob_3 * 3 +
  prob_4 * 4 + prob_5 * 5 + prob_6 * 6

-- The problem statement to prove
theorem expected_value_of_die_is_475 : expected_value = 4.75 := by
  sorry

end expected_value_of_die_is_475_l187_187143


namespace Mandy_yoga_time_l187_187765

theorem Mandy_yoga_time :
  (∀ (G B E Y : ℝ),
    G / B = 2 / 3 ∧
    Y / E = 2 / 3 ∧
    B = 12 →
    G + B = E →
    Y = (E / 3) * 2 →
    Y ≈ 13.33) :=
by sorry

end Mandy_yoga_time_l187_187765


namespace jack_marbles_l187_187409

theorem jack_marbles (initial_marbles : ℕ) (shared_marbles : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 62 → shared_marbles = 33 → remaining_marbles = initial_marbles - shared_marbles → remaining_marbles = 29 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

-- This adds a simple proof environment to verify the theorem.
#eval jack_marbles 62 33 29 (by rfl) (by rfl) (by norm_num)

end jack_marbles_l187_187409


namespace probability_not_late_probability_late_and_misses_bus_l187_187935

variable (P_Sam_late : ℚ)
variable (P_miss_bus_given_late : ℚ)

theorem probability_not_late (h1 : P_Sam_late = 5/9) :
  1 - P_Sam_late = 4/9 := by
  rw [h1]
  norm_num

theorem probability_late_and_misses_bus (h1 : P_Sam_late = 5/9) (h2 : P_miss_bus_given_late = 1/3) :
  P_Sam_late * P_miss_bus_given_late = 5/27 := by
  rw [h1, h2]
  norm_num

#check probability_not_late
#check probability_late_and_misses_bus

end probability_not_late_probability_late_and_misses_bus_l187_187935


namespace max_kings_on_chessboard_no_check_l187_187552

-- Definitions regarding the chessboard and king's movement.
def Chessboard := fin 8 × fin 8
def King_moves (p q : Chessboard) : Prop :=
  abs (p.1.val - q.1.val) ≤ 1 ∧ abs (p.2.val - q.2.val) ≤ 1

-- Main theorem statement
theorem max_kings_on_chessboard_no_check : 
  ∃ (S : finset Chessboard), S.card = 16 ∧ ∀ (k1 k2 ∈ S), k1 ≠ k2 → ¬ King_moves k1 k2 :=
sorry

end max_kings_on_chessboard_no_check_l187_187552


namespace odd_even_function_probability_l187_187694

-- Define the given functions
def f1 (x : ℝ) : ℝ := x + 1 / x
def f2 (x : ℝ) : ℝ := x^2 + x
def f3 (x : ℝ) : ℝ := 2^|x|
def f4 (x : ℝ) : ℝ := x^(2 / 3)
def f5 (x : ℝ) : ℝ := Real.tan x
def f6 (x : ℝ) : ℝ := Real.sin (Real.arccos x)
def f7 (x : ℝ) : ℝ := Real.log (x + Real.sqrt (x^2 + 4)) - Real.log 2

-- Predicate to check if a function is odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Predicate to check if a function is even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- List of functions and their classification
def functions : List (ℝ → ℝ) := [f1, f2, f3, f4, f5, f6, f7]

def odd_functions : List (ℝ → ℝ) := [f1, f5, f7]
def even_functions : List (ℝ → ℝ) := [f3, f4, f6]

-- Proof that the probability that one randomly selected function is odd and the other is even is 3/7
theorem odd_even_function_probability : 
  (choose 3 1) * (choose 3 1) / (choose 7 2) = 3 / 7 := 
by {
  sorry
}

end odd_even_function_probability_l187_187694


namespace find_pos_int_l187_187276

theorem find_pos_int (n p : ℕ) (h_prime : Nat.Prime p) (h_pos_n : 0 < n) (h_pos_p : 0 < p) : 
  n^8 - p^5 = n^2 + p^2 → (n = 2 ∧ p = 3) :=
by
  sorry

end find_pos_int_l187_187276


namespace set_intersection_eq_l187_187440

def A : Set ℝ := {x | |x - 1| ≤ 2}
def B : Set ℝ := {x | x^2 - 4 * x > 0}

theorem set_intersection_eq :
  A ∩ (Set.univ \ B) = {x | 0 ≤ x ∧ x ≤ 3} := by
  sorry

end set_intersection_eq_l187_187440


namespace cubic_root_sum_l187_187423

noncomputable def poly : Polynomial ℝ := Polynomial.C (-4) + Polynomial.C 3 * X + Polynomial.C (-2) * X^2 + X^3

theorem cubic_root_sum (a b c : ℝ) (h1 : poly.eval a = 0) (h2 : poly.eval b = 0) (h3 : poly.eval c = 0)
  (h_sum : a + b + c = 2) (h_prod : a * b + b * c + c * a = 3) (h_triple_prod : a * b * c = 4) :
  a^3 + b^3 + c^3 = 2 := 
by {
  sorry
}

end cubic_root_sum_l187_187423


namespace adi_change_l187_187126

theorem adi_change (pencil notebook colored_pencils amount_paid : ℝ)
  (h1 : pencil = 0.35)
  (h2 : notebook = 1.50)
  (h3 : colored_pencils = 2.75)
  (h4 : amount_paid = 20.00) :
  amount_paid - (pencil + notebook + colored_pencils) = 15.40 :=
by {
  intro h1,
  intro h2,
  intro h3,
  intro h4,
  rw [h1, h2, h3, h4],
  linarith,
}

end adi_change_l187_187126


namespace original_price_of_article_l187_187979

theorem original_price_of_article (new_price : ℝ) (reduction_percentage : ℝ) (original_price : ℝ) 
  (h_reduction : reduction_percentage = 56/100) (h_new_price : new_price = 4400) :
  original_price = 10000 :=
sorry

end original_price_of_article_l187_187979


namespace find_lambda_l187_187308

open Real

noncomputable def P : (ℝ × ℝ) := (-1, 2)
noncomputable def M : (ℝ × ℝ) := (1, -1)

theorem find_lambda (λ : ℝ) (Q : ℝ × ℝ)
    (hQx : 1 = (-1 + Q.1) / 2)
    (hQy : -1 = (2 + Q.2) / 2)
    (hCollinear : ∃ k : ℝ, (4, -6) = (k * λ, k)) :
    λ = -2 / 3 :=
sorry

end find_lambda_l187_187308


namespace find_transformation_matrix_l187_187828

def projection_matrix_onto (u : ℝ × ℝ) : ℝ × ℝ → ℝ × ℝ :=
  λ v, let dot_uu := u.1 * u.1 + u.2 * u.2 in
       let dot_uv := u.1 * v.1 + u.2 * v.2 in
       (dot_uv / dot_uu) • u

theorem find_transformation_matrix :
  let v0 := (x : ℝ × ℝ) in
  let u1 : ℝ × ℝ := (4, 2) in
  let u2 : ℝ × ℝ := (2, 3) in
  let v1 := projection_matrix_onto u1 v0 in
  let v2 := projection_matrix_onto u2 v1 in
  let T := (44 / 65, 22 / 65),
           (66 / 65, 33 / 65) in
  v2 =
    ((proj_matrix_onto u2).comp (proj_matrix_onto u1)) v0 :=
sorry

end find_transformation_matrix_l187_187828


namespace number_in_pattern_l187_187592

theorem number_in_pattern (m n : ℕ) (h : 8 * m - 5 = 2023) (hn : n = 5) : m + n = 258 :=
by
  sorry

end number_in_pattern_l187_187592


namespace staff_work_schedule_l187_187128

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem staff_work_schedule :
  ∀ (n : ℕ),
    (∃ (a b c d : ℕ), a = 5 ∧ b = 6 ∧ c = 8 ∧ d = 9 ∧ lcm (lcm (lcm a b) c) d = n) ↔ n = 360 :=
by
  sorry

end staff_work_schedule_l187_187128


namespace tangent_perpendicular_extreme_values_range_of_m_l187_187703

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * log x + (3 / 2 : ℝ) * x^2 - 4 * x
noncomputable def g (x : ℝ) : ℝ := x^3 - 4
noncomputable def h (m : ℝ) (x : ℝ) : ℝ := f m x - g x

theorem tangent_perpendicular_extreme_values 
  (m : ℝ)
  (hf_tangent : deriv (f m) 1 = 0)
  (hf : ∀ x > 0, deriv (f m) x = m / x + 3 * x - 4)
  : ∃ x_min x_max : ℝ, x_min = 1 ∧ f m x_min = -5 / 2 ∧ x_max = 1 / 3 ∧ f m x_max = -7 / 6 - log 3 :=
begin
  sorry
end

theorem range_of_m (m : ℝ)
  (hh : ∀ x ∈ Ioi 1, deriv (h m) x ≤ 0)
  : m ≤ 4 :=
begin
  sorry
end

end tangent_perpendicular_extreme_values_range_of_m_l187_187703


namespace eccentricity_of_ellipse_l187_187677

-- Define the geometric setup: A square ABCD with certain properties
noncomputable def side_length : ℝ := 1
noncomputable def A : ℝ := 0
noncomputable def B : ℝ := 1
noncomputable def C : ℝ := -1
noncomputable def D : ℝ := 1

-- Define the values c and a, derived from the problem's conditions
noncomputable def c : ℝ := side_length / 2
noncomputable def a : ℝ := (Real.sqrt (2 : ℝ) + 1) / 2

-- The statement we want to prove
theorem eccentricity_of_ellipse : 
  let e := c / a in 
  e = Real.sqrt (2 : ℝ) - 1 :=
by
  sorry

end eccentricity_of_ellipse_l187_187677


namespace unique_parallelograms_l187_187451

def is_lattice_point (p : ℕ × ℕ) : Prop :=
  ∃ k : ℕ, p = (k, k * m)

def line_condition (m : ℕ) (p : ℕ × ℕ) : Prop :=
  m > 1 ∧ is_lattice_point p

theorem unique_parallelograms (m n : ℕ) (A B D : ℕ × ℕ)
  (hA : A = (0, 0))
  (hB : line_condition m B)
  (hD : line_condition n D)
  (area_condition : 500000 = 1 / 2 * abs ((m + n - 2) * (B.1 * D.1))) :
  (number_of_unique_parallelograms A B D) = 392 :=
sorry

end unique_parallelograms_l187_187451


namespace triangles_in_square_l187_187910

theorem triangles_in_square (P : Fin 8 → Fin 3 → Point) (Q : ∀ i, P i 0 = side_points_on_square i)
  (h_divides_sides : ∀ i, divides (line_segment_length (P i 0) (P (i + 1) 0)) 3)
  : (count_right_triangles P = 24) :=
sorry

end triangles_in_square_l187_187910


namespace eq_tangent_line_at_1_l187_187914

def f (x : ℝ) : ℝ := x^4 - 2 * x^3

def tangent_line (x : ℝ) : ℝ := -2 * x + 1

theorem eq_tangent_line_at_1 : 
  ∃ (m : ℝ) (c : ℝ), m = -2 ∧ c = 1 ∧ ∀ x, tangent_line x = m * x + c :=
by
  use -2
  use 1
  split
  . rfl
  split
  . rfl
  intro x
  rfl

end eq_tangent_line_at_1_l187_187914


namespace eval_64_pow_5_over_6_l187_187198

theorem eval_64_pow_5_over_6 (h : 64 = 2^6) : 64^(5/6) = 32 := 
by 
  sorry

end eval_64_pow_5_over_6_l187_187198


namespace find_k_values_l187_187469

noncomputable def problem (a b c d k : ℂ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  (a * k^3 + b * k^2 + c * k + d = 0) ∧
  (b * k^3 + c * k^2 + d * k + a = 0)

theorem find_k_values (a b c d k : ℂ) (h : problem a b c d k) : 
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end find_k_values_l187_187469


namespace area_ratio_PQS_PRS_l187_187761

theorem area_ratio_PQS_PRS (P Q R S : Type) [Inhabited P] [Inhabited Q] [Inhabited R] 
    [Inhabited S] (distPQ : ℝ) (distPR : ℝ) (distQR : ℝ) 
    (angleBisector : ∀ (PQR : triangle P Q R), is_angle_bisector P S (Q, R)) :
    distPQ = 18 →
    distPR = 30 →
    distQR = 25 →
    ratio_area_PQS_PRS P Q R S = 3 / 5 :=
by
  intros
  sorry

end area_ratio_PQS_PRS_l187_187761


namespace max_value_expression_l187_187795

theorem max_value_expression (A M C : ℕ) (h₁ : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
  sorry

end max_value_expression_l187_187795


namespace loci_difference_in_distances_to_lines_l187_187355

variable (a b : Line) (s : ℝ)

theorem loci_difference_in_distances_to_lines (M : Point) :
  (distance M a - distance M b = s) ↔ 
  (∃ P1 P2 P3 P4 : Point, P1 ∈ lineParallelAtDistance a s b ∧ 
                        P2 ∈ lineParallelAtDistance a s b ∧
                        P3 ∈ lineParallelAtDistance b s a ∧
                        P4 ∈ lineParallelAtDistance b s a ∧
                        M ∈ eightRaysThrough P1 P2 P3 P4) :=
sorry

end loci_difference_in_distances_to_lines_l187_187355


namespace length_of_AB_l187_187571

-- Defining the radius and the dimensions
def r : ℝ := 4
def h : ℝ := 4

-- Defining the volume function for a cylinder
def volume_cylinder (r L : ℝ) : ℝ := π * r^2 * L

-- Defining the volume function for a cone
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

-- Define the total volume given in the problem
def V_total : ℝ := 288 * π

-- Stating the theorem: The length of AB (which is L) is 46 / 3
theorem length_of_AB (L : ℝ) 
  (h_r : r = 4)
  (h_h : h = 4)
  (h_V : V_total = (volume_cylinder r L) + 2 * (volume_cone r h)) : 
  L = 46 / 3 := by
  -- Proof goes here
  sorry

end length_of_AB_l187_187571


namespace correct_calculation_l187_187079

theorem correct_calculation (A : 3 + 2 * Real.sqrt 3 ≠ 5 * Real.sqrt 3)
                            (B : Real.sqrt 3 / Real.sqrt 5 = Real.sqrt (3 * 5) / 5)
                            (C : 5 * Real.sqrt 3 * 2 * Real.sqrt 3 ≠ 10 * Real.sqrt 3)
                            (D : 4 * Real.sqrt 3 - 3 * Real.sqrt 3 ≠ 1) :
  B := sorry

end correct_calculation_l187_187079


namespace log_product_sequence_l187_187087

open Real

theorem log_product_sequence :
  (∏ i in Finset.range 78, log (i + 3) ((i + 3) + 1)) = 4 :=
by
  sorry

end log_product_sequence_l187_187087


namespace evaluate_64_pow_5_div_6_l187_187231

theorem evaluate_64_pow_5_div_6 : (64 : ℝ)^(5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ)^6 := by norm_num
  rw [← h1]
  have h2 : ((2 : ℝ)^6)^(5 / 6) = (2 : ℝ)^(6 * (5 / 6)) := by rw [Real.rpow_mul]
  rw [h2]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187231


namespace smallest_angle_of_triangle_l187_187930

noncomputable def smallest_angle (a b : ℝ) (c : ℝ) (h_sum : a + b + c = 180) : ℝ :=
  min a (min b c)

theorem smallest_angle_of_triangle :
  smallest_angle 60 65 (180 - (60 + 65)) (by norm_num) = 55 :=
by
  -- The correct proof steps should be provided for the result
  sorry

end smallest_angle_of_triangle_l187_187930


namespace smallest_number_of_elements_l187_187825

theorem smallest_number_of_elements (S : Type) (X : Fin 100 → set S) :
  (∀ i : Fin 99, X i ≠ ∅ ∧ X (i + 1).val.succ ≠ ∅ ∧ X i ∩ X (i + 1).val.succ = ∅ ∧ (X i ∪ X (i + 1).val.succ) ≠ set.univ) →
  (∀ i j : Fin 100, i ≠ j → X i ≠ X j) →
  8 ≤ Fintype.card S :=
sorry

end smallest_number_of_elements_l187_187825


namespace tan_30_eq_sqrt3_div_3_l187_187178

theorem tan_30_eq_sqrt3_div_3 :
  let opposite := 1
  let adjacent := sqrt (3 : ℝ) 
  tan (real.pi / 6) = opposite / adjacent := by 
    sorry

end tan_30_eq_sqrt3_div_3_l187_187178


namespace sahil_purchase_price_l187_187460

def purchase_price (P : ℝ) : Prop :=
  let repair_cost := 5000
  let transportation_charges := 1000
  let total_cost := repair_cost + transportation_charges
  let selling_price := 27000
  let profit_factor := 1.5
  profit_factor * (P + total_cost) = selling_price

theorem sahil_purchase_price : ∃ P : ℝ, purchase_price P ∧ P = 12000 :=
by
  use 12000
  unfold purchase_price
  simp
  sorry

end sahil_purchase_price_l187_187460


namespace sum_of_roots_l187_187723

theorem sum_of_roots (x : ℝ) :
  (x + 2) * (x - 3) = 16 →
  ∃ a b : ℝ, (a ≠ x ∧ b ≠ x ∧ (x - a) * (x - b) = 0) ∧
             (a + b = 1) :=
by
  intro h
  sorry

end sum_of_roots_l187_187723


namespace six_sided_dice_sum_twenty_l187_187969

theorem six_sided_dice_sum_twenty :
  let number_of_ways := (choose (10 + 5 - 1) (5 - 1)) - (5 * (choose (5 + 4 - 1) (4 - 1))) in
  number_of_ways = 721 := by
  sorry

end six_sided_dice_sum_twenty_l187_187969


namespace sum_f_1_to_2013_l187_187184

noncomputable def f : ℝ → ℝ
| x := if -3 ≤ x ∧ x < -1 then -(x + 2) ^ 2
       else if -1 ≤ x ∧ x < 3 then x
       else f (x - 3)

theorem sum_f_1_to_2013 : ∑ k in finset.range 2013, f (k + 1) = 337 :=
  sorry

end sum_f_1_to_2013_l187_187184


namespace jessie_weight_after_first_week_l187_187780

variable (initial_weight : ℕ) (weight_lost : ℕ)

def final_weight_after_first_week (initial_weight weight_lost : ℕ) : ℕ :=
  initial_weight - weight_lost

theorem jessie_weight_after_first_week :
  initial_weight = 92 → 
  weight_lost = 56 → 
  final_weight_after_first_week initial_weight weight_lost = 36 :=
by
  intros h1 h2
  simp [final_weight_after_first_week, h1, h2]
  sorry

end jessie_weight_after_first_week_l187_187780


namespace evaluate_root_l187_187237

theorem evaluate_root : 64 ^ (5 / 6 : ℝ) = 32 :=
by sorry

end evaluate_root_l187_187237


namespace largest_David_number_l187_187776

-- Define Jana's and David's numbers
variables (Jana David : ℝ)

-- We need some hypothesis
hypothesis H_sum : Jana + David = 11.11
hypothesis H_david_digits : (string.length (David.to_string.split '.'[0]) = string.length (David.to_string.split '.'[1]))
hypothesis H_jana_repeats : ∃ d, (string.count d (Jana.to_string)) = 2
hypothesis H_david_unique : (string.to_list (David.to_string)).nodup

-- The main statement
theorem largest_David_number : David = 0.9 :=
by
  sorry

end largest_David_number_l187_187776


namespace boxes_neither_pens_nor_pencils_l187_187880

theorem boxes_neither_pens_nor_pencils :
  ∀ (total_boxes pens boxes pencils both: ℕ),
  total_boxes = 12 ->
  pencils = 8 ->
  pens = 5 ->
  both = 3 ->
  (total_boxes - ((pencils + pens) - both)) = 2 :=
by
  intros total_boxes pens boxes pencils both ht_boxes hpencils hpens hboth
  rw [ht_boxes, hpencils, hpens, hboth]
  norm_num
  rw [Nat.sub_sub, Nat.add_sub_swap, Nat.sub_self, Nat.add_zero]
  trivial
  sorry

end boxes_neither_pens_nor_pencils_l187_187880


namespace evaluate_pow_l187_187243

theorem evaluate_pow : 64^(5/6 : ℝ) = 32 := by
  sorry

end evaluate_pow_l187_187243


namespace false_statement1_true_statement2_true_statement3_true_statement4_false_statement5_l187_187920

open Real

theorem false_statement1 (x : ℝ) : ¬ (|x| + x = 0 → x < 0) := sorry

theorem true_statement2 (a : ℝ) : (-a ≥ 0 → a ≤ 0) := sorry

theorem true_statement3 (a : ℝ) : | -a^2 | = (-a)^2 := by
  simp

theorem true_statement4 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : (a/|a| + b/|b| = 0 → ab/|ab| = -1) := sorry

theorem false_statement5 (a b : ℝ) : ¬ (|a| = -b ∧ |b| = b → a = b) := sorry

end false_statement1_true_statement2_true_statement3_true_statement4_false_statement5_l187_187920


namespace distinct_real_roots_l187_187835

noncomputable def g (x d : ℝ) : ℝ := x^2 + 4*x + d

theorem distinct_real_roots (d : ℝ) :
  (∃! x : ℝ, g (g x d) d = 0) ↔ d = 0 :=
sorry

end distinct_real_roots_l187_187835


namespace max_value_of_func_l187_187929

def func (x : ℝ) : ℝ :=
  1 - 8 * Real.cos x - 2 * (Real.sin x)^2

theorem max_value_of_func : ∃ x : ℝ, func x = 1 := by
  sorry

end max_value_of_func_l187_187929


namespace discount_percentage_l187_187749

theorem discount_percentage (tshirt_cost pants_cost shoes_cost : ℕ) (tshirts_bought pants_bought shoes_bought amount_paid : ℕ) :
  tshirt_cost = 20 →
  pants_cost = 80 →
  shoes_cost = 150 →
  tshirts_bought = 4 →
  pants_bought = 3 →
  shoes_bought = 2 →
  amount_paid = 558 →
  let total_cost := tshirts_bought * tshirt_cost + pants_bought * pants_cost + shoes_bought * shoes_cost in
  let discount_amount := total_cost - amount_paid in
  let discount_percentage := (discount_amount * 100 / total_cost : ℕ) in
  discount_percentage = 10 := by
  sorry

end discount_percentage_l187_187749


namespace evaluate_pow_l187_187244

theorem evaluate_pow : 64^(5/6 : ℝ) = 32 := by
  sorry

end evaluate_pow_l187_187244


namespace problem_l187_187523

noncomputable def areaBetweenChords (r : ℝ) (d : ℝ) : ℝ :=
  let OC := Real.sqrt (r^2 - (d/2)^2)
  let theta := 2 * Real.asin (OC / r)
  let sectorArea := (theta / (2 * Real.pi)) * (Real.pi * r^2)
  let triangleArea := OC * d
  (Real.pi * r^2) - 2 * (sectorArea - triangleArea/2)

theorem problem (h1 : r = 10) (h2 : d = 6) : 
  areaBetweenChords r d = 100 * Real.pi - 12 * Real.sqrt 91 := by
  rw [h1, h2]
  sorry

end problem_l187_187523


namespace yogurt_combinations_l187_187103

theorem yogurt_combinations :
  let flavors := 6
  let toppings := 8
  let choose_3_toppings := Nat.choose toppings 3
  in flavors * choose_3_toppings = 336 :=
by
  let flavors := 6
  let toppings := 8
  let choose_3_toppings := Nat.choose toppings 3
  have : flavors * choose_3_toppings = 336 := by
    sorry
  exact this

end yogurt_combinations_l187_187103


namespace min_f_value_l187_187304

noncomputable def f (a θ : ℝ) : ℝ := (Real.cos θ)^3 + 4 / (3 * a * (Real.cos θ)^2 - a^3)

theorem min_f_value (a θ : ℝ) (h1 : 0 < a) (h2 : a < Real.sqrt 3 * Real.cos θ)
  (h3 : θ ∈ Set.Icc (-Real.pi/4) (Real.pi/3)) :
  Exists (λ m, ∀ (a θ : ℝ), 0 < a ∧ a < Real.sqrt 3 * Real.cos θ ∧ θ ∈ Set.Icc (-Real.pi/4) (Real.pi/3) 
    → f(a, θ) ≥ m ∧ f(a, θ) = m ↔ f(a, θ) =  17 * Real.sqrt 2 / 4) :=
  sorry

end min_f_value_l187_187304


namespace graph_symmetry_l187_187009

/-- Theorem:
The functions y = 2^x and y = 2^{-x} are symmetric about the y-axis.
-/
theorem graph_symmetry :
  ∀ (x : ℝ), (∃ (y : ℝ), y = 2^x) →
  (∃ (y' : ℝ), y' = 2^(-x)) →
  (∀ (y : ℝ), ∃ (x : ℝ), (y = 2^x ↔ y = 2^(-x)) → y = 2^x → y = 2^(-x)) :=
by
  intro x
  intro h1
  intro h2
  intro y
  exists x
  intro h3
  intro hy
  sorry

end graph_symmetry_l187_187009


namespace monthly_rent_l187_187777

-- Definition
def total_amount_saved := 2225
def extra_amount_needed := 775
def deposit := 500

-- Total amount required
def total_amount_required := total_amount_saved + extra_amount_needed
def total_rent_plus_deposit (R : ℝ) := 2 * R + deposit

-- The statement to prove
theorem monthly_rent (R : ℝ) : total_rent_plus_deposit R = total_amount_required → R = 1250 :=
by
  intros h
  exact sorry -- Proof is omitted.

end monthly_rent_l187_187777


namespace total_students_at_concert_l187_187514

-- Define the number of buses
def num_buses : ℕ := 8

-- Define the number of students per bus
def students_per_bus : ℕ := 45

-- State the theorem with the conditions and expected result
theorem total_students_at_concert : (num_buses * students_per_bus) = 360 := by
  -- Proof is not required as per the instructions; replace with 'sorry'
  sorry

end total_students_at_concert_l187_187514


namespace GCD_of_n_pow_13_sub_n_l187_187624

theorem GCD_of_n_pow_13_sub_n :
  ∀ n : ℤ, gcd (n^13 - n) 2730 = gcd (n^13 - n) n := sorry

end GCD_of_n_pow_13_sub_n_l187_187624


namespace intersection_eq_l187_187443

-- Definitions for M and N
def M : Set ℤ := Set.univ
def N : Set ℤ := {x : ℤ | x^2 - x - 2 < 0}

-- The theorem to be proved
theorem intersection_eq : M ∩ N = {0, 1} := 
  sorry

end intersection_eq_l187_187443


namespace range_of_fraction_sum_l187_187730

theorem range_of_fraction_sum (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 2 * y = 1) : 
  ∀ z, (∃ x y, 0 < x ∧ 0 < y ∧ x + 2 * y = 1 ∧ z = 1/x + 1/y) ↔ z ∈ set.Ici (3 + 2 * Real.sqrt 2) :=
by sorry

end range_of_fraction_sum_l187_187730


namespace base_ten_representation_15_factorial_l187_187901

/-- Base ten representation problem for 15! involves unknown digits 
    T, M, and H whose sum we want to determine. -/
theorem base_ten_representation_15_factorial (T M H : ℕ) 
  (h1: 15! = 1 * 10^6 + 3 * 10^5 + 0 * 10^4 + 7 * 10^3 + M * 10^2 + 7 * 10 + T * 10 + 2000 + H * 1000)
  (h2: 1 + 3 + 0 + 7 + 2 == 13)
  (h3 : M + T + 13 % 3 = 0) :
  T + M + H = 2 := 
sorry

end base_ten_representation_15_factorial_l187_187901


namespace area_of_inscribed_rectangle_l187_187995

theorem area_of_inscribed_rectangle (r l w : ℝ) (h1 : r = 8) (h2 : l / w = 3) (h3 : w = 2 * r) : l * w = 768 :=
by
  sorry

end area_of_inscribed_rectangle_l187_187995


namespace sufficient_necessary_condition_l187_187576

theorem sufficient_necessary_condition (a : ℝ) :
  (∃ x : ℝ, 2 * x + 1 = a ∧ x > 2) ↔ a > 5 :=
by
  sorry

end sufficient_necessary_condition_l187_187576


namespace find_speed_of_stream_l187_187507

theorem find_speed_of_stream (x : ℝ) (h1 : ∃ x, 1 / (39 - x) = 2 * (1 / (39 + x))) : x = 13 :=
by
sorry

end find_speed_of_stream_l187_187507


namespace integer_values_of_a_l187_187640

theorem integer_values_of_a : 
  ∃ (a : Set ℤ), (∀ x, x ∈ a → ∃ (y z : ℤ), x^2 + x * y + 9 * y = 0) ∧ (a.card = 6) :=
by
  sorry

end integer_values_of_a_l187_187640


namespace eval_64_pow_5_over_6_l187_187196

theorem eval_64_pow_5_over_6 (h : 64 = 2^6) : 64^(5/6) = 32 := 
by 
  sorry

end eval_64_pow_5_over_6_l187_187196


namespace necessarily_negative_b_ab_l187_187515

theorem necessarily_negative_b_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : -2 < b) (h4 : b < 0) : 
  b + a * b < 0 := by 
  sorry

end necessarily_negative_b_ab_l187_187515


namespace always_exists_perpendicular_line_on_ground_l187_187746

-- Definition of the problem using Lean 4
def line_on_ground_perpendicular_to_ruler (ground : Type) [plane ground] (ruler_line : Type) [line ruler_line] : Prop :=
  ∃ ground_line : ground → Prop, ∀ placement : ruler_line → ground, perpendicular(ground_line, placement)

axiom perpendicular : ∀ (l1 l2 : Type) [line l1] [line l2], Prop

theorem always_exists_perpendicular_line_on_ground (ground : Type) [plane ground] (ruler_line : Type) [line ruler_line] :
  (∀ placement : ruler_line → ground, ∃ ground_line : ground → Prop, perpendicular(ground_line, placement)) :=
by
  sorry

end always_exists_perpendicular_line_on_ground_l187_187746


namespace find_x_l187_187534

theorem find_x (x : ℝ) : x - (502 / 100.4) = 5015 → x = 5020 :=
by
  sorry

end find_x_l187_187534


namespace count_congruent_to_5_mod_7_l187_187718

theorem count_congruent_to_5_mod_7 (n : ℕ) :
  (∀ x : ℕ, 1 ≤ x ∧ x ≤ 300 ∧ x % 7 = 5) → ∃ count : ℕ, count = 43 := by
  sorry

end count_congruent_to_5_mod_7_l187_187718


namespace evaluate_root_l187_187234

theorem evaluate_root : 64 ^ (5 / 6 : ℝ) = 32 :=
by sorry

end evaluate_root_l187_187234


namespace cauchy_schwarz_inequality_l187_187439

variables {n : ℕ} (x y : Fin n → ℝ)

theorem cauchy_schwarz_inequality : 
  abs (∑ i, x i * y i) ≤ sqrt (∑ i, (x i)^2) * sqrt (∑ i, (y i)^2) :=
sorry

end cauchy_schwarz_inequality_l187_187439


namespace problem_k__l187_187608

def operation : ℕ → ℕ → ℕ := λ x y, x^3 + 3 - y

theorem problem_k_⊗_(k_⊗_(k_⊗_k)) (k : ℕ) : 
  operation k (operation k (operation k k)) = k^3 + 3 - k := 
by 
  sorry

end problem_k__l187_187608


namespace distance_from_point_to_plane_l187_187188

-- Define the point and plane equation
def point : ℝ × ℝ × ℝ := (2, 4, 1)
def plane (x y z : ℝ) : ℝ := x + 2 * y + 3 * z + 3

-- Define the distance function from a point to a plane
def distance_to_plane (P : ℝ × ℝ × ℝ) (A B C D : ℝ) : ℝ :=
  abs (A * P.1 + B * P.2 + C * P.3 + D) / real.sqrt (A^2 + B^2 + C^2)

-- The expected distance
def expected_distance : ℝ := (8 * real.sqrt 14) / 7

-- The main statement: distance from the point to the plane equals the expected value
theorem distance_from_point_to_plane :
  distance_to_plane point 1 2 3 3 = expected_distance :=
by
  sorry

end distance_from_point_to_plane_l187_187188


namespace max_value_amc_am_mc_ca_l187_187790

theorem max_value_amc_am_mc_ca (A M C : ℕ) 
  (h : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 := 
sorry

end max_value_amc_am_mc_ca_l187_187790


namespace maximum_ones_in_triangular_table_l187_187430

def is_b_k (a : ℕ → ℕ) (k : ℕ) : ℕ :=
  if a k = a (k + 1) then 0 else 1

def process_repeatedly {n : ℕ} (a : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ k < n, b k = is_b_k a k

def triangular_table_maximum_ones (n : ℕ) : ℕ :=
  (n^2 + n + 1) / 3

theorem maximum_ones_in_triangular_table (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) :
  (∀ i < n, a i = 0 ∨ a i = 1) →
  process_repeatedly a b →
  (count_ones_in_table a b n ≤ triangular_table_maximum_ones n) :=
sorry

end maximum_ones_in_triangular_table_l187_187430


namespace rotation_150_moves_rectangle_to_smaller_circle_l187_187097

theorem rotation_150_moves_rectangle_to_smaller_circle :
  ∀ (figures : ℕ → string), 
    (figures 0 = "rectangle" ∧ figures 1 = "smaller circle" ∧ figures 2 = "pentagon") →
    (by rotation_150 figures = figures.rotate 1 ∧ figures.rotate 1 = figures.rotate 2) →
    figures 0 = "smaller circle" :=
by
  intro figures h_pos h_rot
  sorry

end rotation_150_moves_rectangle_to_smaller_circle_l187_187097


namespace brenda_more_than_jeff_l187_187985

def emma_amount : ℕ := 8
def daya_amount : ℕ := emma_amount + (emma_amount * 25 / 100)
def jeff_amount : ℕ := (2 / 5) * daya_amount
def brenda_amount : ℕ := 8

theorem brenda_more_than_jeff :
  brenda_amount - jeff_amount = 4 :=
sorry

end brenda_more_than_jeff_l187_187985


namespace closest_vector_l187_187614

noncomputable section

def v (s : ℚ) : ℚ × ℚ × ℚ :=
  (3 + 5 * s, -2 + 3 * s, -4 - 2 * s)

def a : ℚ × ℚ × ℚ :=
  (5, 5, 6)

def direction : ℚ × ℚ × ℚ :=
  (5, 3, -2)

def orthogonal (u v : ℚ × ℚ × ℚ) : Prop :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3 = 0

theorem closest_vector :
  ∃ s : ℚ, orthogonal (v s - a) direction ∧ s = 11 / 38 := sorry

end closest_vector_l187_187614


namespace complex_power_equality_l187_187619

-- Define the complex number (1 - i) / sqrt(2)
def base_expr : ℂ := (1 - complex.I) / real.sqrt 2

-- Define the power 52
def exponent := 52

-- State the main theorem
theorem complex_power_equality : (base_expr ^ exponent) = -1 := 
by 
{
  sorry
}

end complex_power_equality_l187_187619


namespace evaluate_64_pow_fifth_sixth_l187_187214

theorem evaluate_64_pow_fifth_sixth : 64 ^ (5 / 6) = 32 := by
  have h : 64 = 2 ^ 6 := by sorry
  calc 64 ^ (5 / 6) = (2 ^ 6) ^ (5 / 6) : by rw [h]
              ...   = 2 ^ (6 * (5 / 6))  : by sorry
              ...   = 2 ^ 5              : by sorry
              ...   = 32                 : by sorry

end evaluate_64_pow_fifth_sixth_l187_187214


namespace unit_prices_min_type_a_l187_187582

theorem unit_prices (x : ℕ) :
  (1.5 * x) * 10 + 5400 = 7200 → x = 360 :=
by
  assume h : (1.5 * x) * 10 + 5400 = 7200
  sorry

theorem min_type_a (x y : ℕ) :
  x = 360 → y = 540 →
  (50 - y) ≤ 50 →
  360 * 34 + 540 * (50 - 34) ≤ 21000 :=
by
  assume h1 : x = 360
  assume h2 : y = 540
  assume h3 : (50 - y) ≤ 50
  sorry

end unit_prices_min_type_a_l187_187582


namespace geometric_seq_l187_187310

def seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  a 1 = 3 ∧ (∀ n : ℕ, S (n + 1) + a n = S n + 5 * 4 ^ n)

theorem geometric_seq (a S : ℕ → ℝ) (h : seq a S) :
  ∃ r : ℝ, ∃ a1 : ℝ, (∀ n : ℕ, (a (n + 1) - 4 ^ (n + 1)) = r * (a n - 4 ^ n)) :=
by
  sorry

end geometric_seq_l187_187310


namespace pump_A_time_l187_187084

noncomputable def pump_rate_B := 1 / 6  -- Pool per hour
noncomputable def combined_time := 144 / 60  -- 144 minutes equivalent to 2.4 hours
noncomputable def combined_rate := 1 / combined_time  -- Pool per hour

theorem pump_A_time : ∃ (A : ℝ), A = 4 ∧ (1 / A + pump_rate_B = combined_rate) :=
by
  existsi 4
  have h : (1:ℝ) / 4 + 1 / 6 = 5 / 12 := by norm_num
  have h_combined_rate : (1:ℝ) / 2.4 = 5 / 12 := by norm_num
  rw [h_combined_rate] at h
  exact ⟨rfl, h⟩
-- sorry

end pump_A_time_l187_187084


namespace number_of_ways_to_place_numbers_l187_187396

theorem number_of_ways_to_place_numbers :
  ∃ (f : Fin 9 → Fin 9) (f_inj : Function.Injective f),
    let sums := [ (f 0 + f 1 + f 2), (f 3 + f 4 + f 5), (f 6 + f 7 + f 8) ] in
    (sums !! 0 = 7) ∧
    (sums !! 1 = 8) ∧
    (sums !! 2 = 9) ∧
    (sums !! 0 + sums !! 1 + sums !! 2 = 24) ∧
    count_ways(f) = 32 :=
sorry

end number_of_ways_to_place_numbers_l187_187396


namespace total_weight_gain_l187_187135

def orlando_gained : ℕ := 5

def jose_gained (orlando : ℕ) : ℕ :=
  2 * orlando + 2

def fernando_gained (jose : ℕ) : ℕ :=
  jose / 2 - 3

theorem total_weight_gain (O J F : ℕ) 
  (ho : O = orlando_gained) 
  (hj : J = jose_gained O) 
  (hf : F = fernando_gained J) :
  O + J + F = 20 :=
by
  sorry

end total_weight_gain_l187_187135


namespace problem_statement_l187_187339

open Real

variable (a : ℝ) (θ : ℝ) (α : ℝ)

def f (x : ℝ) : ℝ := a + 2 * cos x ^ 2 * cos (2 * x + θ)

theorem problem_statement :
  (∀ x, f x = -f (-x)) ∧
  f (π / 4) = 0 ∧
  a ∈ Set.univ ∧
  θ ∈ Ioo 0 π →
  a = -1 ∧ θ = π / 2 ∧
  (f (α / 4) = -2 / 5 ∧ α ∈ Ioo (π / 2) π → sin (α + π / 3) = (4 - 3 * sqrt 3) / 10) :=
by
  sorry

end problem_statement_l187_187339


namespace value_of_b_l187_187399

theorem value_of_b (a b m : ℤ) (h₁ : a = ∑ i in finset.range 21, nat.choose 20 i * 2^i) (h₂ : a % 10 = b % 10) (h₃ : m > 0) :
  b = 2011 := by
  sorry

end value_of_b_l187_187399


namespace smallest_n_for_perfect_square_and_cube_l187_187055

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, (∃ a : ℕ, 4 * n = a^2) ∧ (∃ b : ℕ, 5 * n = b^3) ∧ n = 125 :=
by
  sorry

end smallest_n_for_perfect_square_and_cube_l187_187055


namespace tan_30_eq_sqrt3_div3_l187_187155

theorem tan_30_eq_sqrt3_div3 (sin_30_cos_30 : ℝ → ℝ → Prop)
  (h1 : sin_30_cos_30 (1 / 2) (Real.sqrt 3 / 2)) :
  ∃ t, t = Real.tan (Real.pi / 6) ∧ t = Real.sqrt 3 / 3 :=
by
  existsi Real.tan (Real.pi / 6)
  sorry

end tan_30_eq_sqrt3_div3_l187_187155


namespace hyperbola_foci_l187_187904

noncomputable def foci_coordinates (a b : ℝ) : ℝ × ℝ :=
  let c := Real.sqrt (a^2 + b^2)
  (c, 0)

theorem hyperbola_foci :
  let a := 3
  let b := 4
  foci_coordinates a b = (5, 0) :=
by
  let a := 3
  let b := 4
  show foci_coordinates a b = (5, 0)
  sorry

end hyperbola_foci_l187_187904


namespace big_container_capacity_l187_187559

-- Defining the conditions
def big_container_initial_fraction : ℚ := 0.30
def second_container_initial_fraction : ℚ := 0.50
def big_container_added_water : ℚ := 18
def second_container_added_water : ℚ := 12
def big_container_final_fraction : ℚ := 3 / 4
def second_container_final_fraction : ℚ := 0.90

-- Defining the capacity of the containers
variable (C_b C_s : ℚ)

-- Defining the equations based on the conditions
def big_container_equation : Prop :=
  big_container_initial_fraction * C_b + big_container_added_water = big_container_final_fraction * C_b

def second_container_equation : Prop :=
  second_container_initial_fraction * C_s + second_container_added_water = second_container_final_fraction * C_s

-- Proof statement to prove the capacity of the big container
theorem big_container_capacity : big_container_equation C_b → C_b = 40 :=
by
  intro H
  -- Skipping the proof steps
  sorry

end big_container_capacity_l187_187559


namespace greatest_length_measures_exactly_l187_187980

theorem greatest_length_measures_exactly 
    (a b c : ℕ) 
    (ha : a = 700)
    (hb : b = 385)
    (hc : c = 1295) : 
    Nat.gcd (Nat.gcd a b) c = 35 := 
by
  sorry

end greatest_length_measures_exactly_l187_187980


namespace train_pass_time_approx_l187_187122

def length_of_train : ℝ := 400
def length_of_bridge : ℝ := 800
def speed_of_train_kmh : ℝ := 60
def speed_of_train_mps : ℝ := (speed_of_train_kmh * 1000) / 3600
def total_distance_to_cover : ℝ := length_of_train + length_of_bridge
def approximate_time_to_pass_bridge : ℝ := total_distance_to_cover / speed_of_train_mps

theorem train_pass_time_approx :
  approximate_time_to_pass_bridge ≈ 72 := 
sorry

end train_pass_time_approx_l187_187122


namespace cafe_problem_l187_187412

def total_items_to_buy (budget : ℝ) (sandwich_cost : ℝ) (hot_chocolate_cost : ℝ) : ℕ :=
  let s := ⌊budget / sandwich_cost⌋ 
  let remaining_money := budget - (s * sandwich_cost)
  let h := ⌊remaining_money / hot_chocolate_cost⌋ in
  s + h

theorem cafe_problem :
  total_items_to_buy 35 5 1.5 = 7 :=
by
  simp [total_items_to_buy]
  norm_num
  sorry

end cafe_problem_l187_187412


namespace smallest_n_for_4n_square_and_5n_cube_l187_187052

theorem smallest_n_for_4n_square_and_5n_cube :
  ∃ (n : ℕ), (n > 0 ∧ (∃ k : ℕ, 4 * n = k^2) ∧ (∃ m : ℕ, 5 * n = m^3)) ∧ n = 400 :=
by
  sorry

end smallest_n_for_4n_square_and_5n_cube_l187_187052


namespace smallest_n_for_perfect_square_and_cube_l187_187056

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, (∃ a : ℕ, 4 * n = a^2) ∧ (∃ b : ℕ, 5 * n = b^3) ∧ n = 125 :=
by
  sorry

end smallest_n_for_perfect_square_and_cube_l187_187056


namespace number_of_welders_left_l187_187984

-- Define the constants and variables
def welders_total : ℕ := 36
def days_to_complete : ℕ := 5
def rate : ℝ := 1  -- Assume the rate per welder is 1 for simplicity
def total_work : ℝ := welders_total * days_to_complete * rate

def days_after_first : ℕ := 6
def work_done_in_first_day : ℝ := welders_total * 1 * rate
def remaining_work : ℝ := total_work - work_done_in_first_day

-- Define the theorem to solve for the number of welders x that started to work on another project
theorem number_of_welders_left (x : ℕ) : (welders_total - x) * days_after_first * rate = remaining_work → x = 12 := by
  intros h
  sorry

end number_of_welders_left_l187_187984


namespace total_journey_distance_l187_187086

theorem total_journey_distance : 
  ∃ D : ℝ, 
    (∀ (T : ℝ), T = 10) →
    ((D/2) / 21 + (D/2) / 24 = 10) →
    D = 224 := 
by
  sorry

end total_journey_distance_l187_187086


namespace zinc_percentage_in_1_gram_antacid_l187_187411

theorem zinc_percentage_in_1_gram_antacid :
  ∀ (z1 z2 : ℕ → ℤ) (total_zinc : ℤ),
    z1 0 = 2 ∧ z2 0 = 2 ∧ z1 1 = 1 ∧ total_zinc = 650 ∧
    (z1 0) * 2 * 5 / 100 + (z2 1) * 3 = total_zinc / 100 →
    (z2 1) * 100 = 15 :=
by
  sorry

end zinc_percentage_in_1_gram_antacid_l187_187411


namespace problem_statement1_problem_statement2_problem_statement3_problem_statement4_l187_187297

-- Definitions based on given conditions
def f (x : ℤ) : ℤ := (3 * x - 1) ^ 4

theorem problem_statement1 :
  f 1 = (3 * 1 - 1) ^ 4 = 16 :=
by sorry

theorem problem_statement2 :
  f (-1) = (3 * (-1) - 1) ^ 4 = 256 :=
by sorry

theorem problem_statement3 :
  f 0 = (3 * 0 - 1) ^ 4 = 1 :=
by sorry

theorem problem_statement4 :
  ( ∑ k in finset.range 28 \ {0}, nat.choose 27 k) % 9 = 7 :=
by sorry

end problem_statement1_problem_statement2_problem_statement3_problem_statement4_l187_187297


namespace eval_power_l187_187212

-- Given condition
def sixty_four : ℕ := 64

-- Given condition rewritten in Lean
def sixty_four_as_two_powersix : sixty_four = 2^6 := by
  sorry

-- Prove that 64^(5/6) = 32
theorem eval_power : real.exp (5/6 * real.log 64) = 32 := by
  have h1 : 64 = 2^6 := sixty_four_as_two_powersix
  sorry

end eval_power_l187_187212


namespace normal_distribution_test_l187_187519

noncomputable def normal_distribution_at_least_90 : Prop :=
  let μ := 78
  let σ := 4
  -- Given reference data
  let p_within_3_sigma := 0.9974
  -- Calculate P(X >= 90)
  let p_at_least_90 := (1 - p_within_3_sigma) / 2
  -- The expected answer 0.13% ⇒ 0.0013
  p_at_least_90 = 0.0013

theorem normal_distribution_test :
  normal_distribution_at_least_90 :=
by
  sorry

end normal_distribution_test_l187_187519


namespace tan_30_deg_l187_187171

theorem tan_30_deg : 
  let θ := (30 : ℝ) * (Real.pi / 180)
  in Real.sin θ = 1 / 2 ∧ Real.cos θ = Real.sqrt 3 / 2 → Real.tan θ = Real.sqrt 3 / 3 :=
by
  intro h
  let th := θ
  have h1 : Real.sin th = 1 / 2 := And.left h
  have h2 : Real.cos th = Real.sqrt 3 / 2 := And.right h
  sorry

end tan_30_deg_l187_187171


namespace largest_gcd_l187_187012

theorem largest_gcd (a b : ℕ) (h : a + b = 1008) : ∃ d, d = gcd a b ∧ (∀ d', d' = gcd a b → d' ≤ d) ∧ d = 504 :=
by
  sorry

end largest_gcd_l187_187012


namespace evaluate_pow_l187_187262

theorem evaluate_pow : (64 : ℝ) = (8 : ℝ) ^ 2 → (8 : ℝ) = (2 : ℝ) ^ 3 → (64 : ℝ) ^ (5 / 6) = 32 :=
by
  intros h1 h2
  rw h1
  rw h2
  have h3 : (2 : ℝ)^3 ^ 2 = (2 : ℝ) ^ 6 := by ring_exp
  rw h3
  sorry

end evaluate_pow_l187_187262


namespace train_crosses_platform_in_35_seconds_l187_187123

noncomputable def train_crossing_platform_time (v : ℝ) (t : ℝ) (P : ℝ) : ℝ :=
  let v_mps := (v * 1000) / 3600
  let L := v_mps * t
  let D := L + P
  D / v_mps

theorem train_crosses_platform_in_35_seconds :
  train_crossing_platform_time 72 18 340 = 35 := by
  unfold train_crossing_platform_time
  rw [show 72 * 1000 / 3600 = 20, by norm_num]
  rw [show 20 * 18 = 360, by norm_num]
  rw [show 360 + 340 = 700, by norm_num]
  rw [show 700 / 20 = 35, by norm_num]
  sorry

end train_crosses_platform_in_35_seconds_l187_187123


namespace angle_BAC_l187_187190

theorem angle_BAC
  (elevation_angle_B_from_A : ℝ)
  (depression_angle_C_from_A : ℝ)
  (h₁ : elevation_angle_B_from_A = 60)
  (h₂ : depression_angle_C_from_A = 70) :
  elevation_angle_B_from_A + depression_angle_C_from_A = 130 :=
by
  sorry

end angle_BAC_l187_187190


namespace units_digit_difference_l187_187075

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_difference :
  units_digit (72^3) - units_digit (24^3) = 4 :=
by
  sorry

end units_digit_difference_l187_187075


namespace find_constants_l187_187710

variables {V : Type*} [inner_product_space ℝ V]
variables (a b p : V)
variables (t u : ℝ)

theorem find_constants
  (h : ∥p - b∥ = 3 * ∥p - a∥) :
  ∃ t u : ℝ, t = 9 / 8 ∧ u = -1 / 8 :=
begin
  use [9 / 8, -1 / 8],
  split;
  { refl, }
end

end find_constants_l187_187710


namespace smallest_n_for_perfect_square_and_cube_l187_187057

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, (∃ a : ℕ, 4 * n = a^2) ∧ (∃ b : ℕ, 5 * n = b^3) ∧ n = 125 :=
by
  sorry

end smallest_n_for_perfect_square_and_cube_l187_187057


namespace exists_separating_line_l187_187963

structure Points :=
  (white : Finset Point)
  (black : Finset Point)

variables (P : Points)

def separates_line (l : Line) (w : Finset Point) (b : Finset Point) :=
  ∀ p ∈ w, ∀ q ∈ b, (l.separates p q)

theorem exists_separating_line (h : ∀ a b c d : Point, ∃ l : Line, separates_line l (Finset.insert a (Finset.insert b (Finset.insert c {d}))) {d}) :
  ∃ l : Line, separates_line l P.white P.black :=
begin
  sorry
end

end exists_separating_line_l187_187963


namespace alpha_plus_beta_value_l187_187299

theorem alpha_plus_beta_value :
  ∀ (α β: ℝ),
  (sin (2 * α) = sqrt 5 / 5) →
  (sin (β - α) = sqrt 10 / 10) →
  (α ∈ set.Icc (π / 4) π) →
  (β ∈ set.Icc π (3 * π / 2)) →
  α + β = 7 * π / 4 :=
by intros α β h1 h2 h3 h4; sorry

end alpha_plus_beta_value_l187_187299


namespace proof_part1_proof_part2_l187_187384

noncomputable
def part1 (R r : ℝ) : Prop :=
  R - r = 2 ∧ π * (R^2 - r^2) = 96 * π ∧ (π = 3) → π * R^2 = 1875

theorem proof_part1 (R r : ℝ) : part1 R r :=
  by sorry

noncomputable
def part2 (R Q : ℝ) : Prop :=
  Q = 1875 → (R : ℕ) < (5 : ℝ)^(2/15) → R = 2

theorem proof_part2 (R : ℕ) (Q : ℝ) : part2 R Q :=
  by sorry

end proof_part1_proof_part2_l187_187384


namespace part_1_part_2_l187_187441

def f (x m : ℝ) : ℝ := |x + 8 / m| + |x - 2 * m|

theorem part_1 (m x : ℝ) (h : m > 0) : f x m ≥ 8 := 
by sorry

theorem part_2 (m : ℝ) (h : 0 < m ∧ m < 1 ∨ m > 4) : f 1 m > 10 := 
by sorry

end part_1_part_2_l187_187441


namespace solid_of_revolution_volume_l187_187921

noncomputable def volume_of_solid_of_revolution (a α : ℝ) : ℝ :=
  (π * a^3 / 3) * (sin (2 * α))^2 * (sin (α + π / 6)) * (sin (α - π / 6))

theorem solid_of_revolution_volume (a α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  let EC := a * cos α
  let CF := EC * sin α
  let EF := EC * cos α
  let V1 := π * (CF^2) * a
  let V2 := (1 / 3) * π * (EF^2) * a
  V1 + 2 * V2 = volume_of_solid_of_revolution a α :=
by
  unfold volume_of_solid_of_revolution
  sorry

end solid_of_revolution_volume_l187_187921


namespace valid_selections_one_female_l187_187945

-- Define the conditions
def GroupA_males : ℕ := 5
def GroupA_females : ℕ := 3
def GroupB_males : ℕ := 6
def GroupB_females : ℕ := 2
def students_selected_each_group : ℕ := 2

-- Define the selection problem and prove that the number of valid selections is 345
theorem valid_selections_one_female :
  let scenario1 := choose GroupA_males 1 * choose GroupA_females 1 * choose GroupB_males 2,
      scenario2 := choose GroupA_males 2 * choose GroupB_males 1 * choose GroupB_females 1
  in scenario1 + scenario2 = 345 := by
  sorry

end valid_selections_one_female_l187_187945


namespace smallest_n_for_perfect_square_and_cube_l187_187074

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, 0 < n ∧ (∃ a1 b1 : ℕ, 4 * n = a1 ^ 2 ∧ 5 * n = b1 ^ 3 ∧ n = 50) :=
begin
  use 50,
  split,
  { norm_num, },
  { use [10, 5],
    split,
    { norm_num, },
    { split, 
      { norm_num, },
      { refl, }, },
  },
  sorry
end

end smallest_n_for_perfect_square_and_cube_l187_187074


namespace determine_range_of_k_l187_187345

noncomputable def inequality_holds_for_all_x (k : ℝ) : Prop :=
  ∀ (x : ℝ), x^4 + (k - 1) * x^2 + 1 ≥ 0

theorem determine_range_of_k (k : ℝ) : inequality_holds_for_all_x k ↔ k ≥ 1 := sorry

end determine_range_of_k_l187_187345


namespace evaluate_64_pow_5_div_6_l187_187225

theorem evaluate_64_pow_5_div_6 : (64 : ℝ)^(5 / 6) = 32 := by
  have h1 : (64 : ℝ) = (2 : ℝ)^6 := by norm_num
  rw [← h1]
  have h2 : ((2 : ℝ)^6)^(5 / 6) = (2 : ℝ)^(6 * (5 / 6)) := by rw [Real.rpow_mul]
  rw [h2]
  norm_num
  sorry

end evaluate_64_pow_5_div_6_l187_187225


namespace find_a_l187_187695

def f : ℝ → ℝ :=
λ x, if x > 0 then (1/2) * x - 1 else (1/2) ^ x

theorem find_a (a : ℝ) : (f a = 1) ↔ (a = 0 ∨ a = 4) :=
begin
  sorry
end

end find_a_l187_187695


namespace employee_payments_l187_187950

theorem employee_payments :
  ∃ (A B C : ℤ), A = 900 ∧ B = 600 ∧ C = 500 ∧
    A + B + C = 2000 ∧
    A = 3 * B / 2 ∧
    C = 400 + 100 := 
by
  sorry

end employee_payments_l187_187950


namespace total_number_of_people_on_bus_l187_187867

theorem total_number_of_people_on_bus (boys girls : ℕ)
    (driver assistant teacher : ℕ) 
    (h1 : boys = 50)
    (h2 : girls = boys + (2 * boys / 5))
    (h3 : driver = 1)
    (h4 : assistant = 1)
    (h5 : teacher = 1) :
    (boys + girls + driver + assistant + teacher = 123) :=
by
    sorry

end total_number_of_people_on_bus_l187_187867


namespace smallest_n_for_4n_square_and_5n_cube_l187_187054

theorem smallest_n_for_4n_square_and_5n_cube :
  ∃ (n : ℕ), (n > 0 ∧ (∃ k : ℕ, 4 * n = k^2) ∧ (∃ m : ℕ, 5 * n = m^3)) ∧ n = 400 :=
by
  sorry

end smallest_n_for_4n_square_and_5n_cube_l187_187054


namespace find_subtracted_number_l187_187500

theorem find_subtracted_number (x y : ℕ) (h1 : 6 * x - 5 * x = 5) (h2 : (30 - y) * 4 = (25 - y) * 5) : y = 5 :=
sorry

end find_subtracted_number_l187_187500


namespace neva_time_difference_l187_187970

theorem neva_time_difference :
  let young_cycling_time_per_mile := 165 / 20
  let older_walking_time_per_mile := 180 / 8
  (older_walking_time_per_mile - young_cycling_time_per_mile = 14.25) := by
  unfold young_cycling_time_per_mile older_walking_time_per_mile
  sorry

end neva_time_difference_l187_187970


namespace number_of_valid_permutations_l187_187133

open Finset

-- Define the finite set of permutations of the numbers 1 to 6
def permutations_6 : Finset (Fin 6 → Fin 6) :=
  univ.filter (λ σ : (Fin 6 → Fin 6), 
    σ 0 ≠ 1 ∧ σ 2 ≠ 3 ∧ σ 4 ≠ 5 ∧ σ 0 < σ 2 ∧ σ 2 < σ 4)

-- Prove that the number of such permutations is 30
theorem number_of_valid_permutations : 
  (permutations_6.card = 30) :=
by 
  sorry

end number_of_valid_permutations_l187_187133


namespace find_scalar_m_l187_187426

variables (p q r : EuclideanSpace ℝ (Fin 3))
variables (m : ℝ)

theorem find_scalar_m (h1 : p + q + r = 0) (h2 : m * (q × p) + 2 * (q × r) + 2 * (r × p) = 0) : m = 4 :=
sorry

end find_scalar_m_l187_187426


namespace sum_of_two_numbers_l187_187869

theorem sum_of_two_numbers (L S : ℕ) (hL : L = 22) (hExceeds : L = S + 10) : L + S = 34 := by
  sorry

end sum_of_two_numbers_l187_187869


namespace find_resistance_l187_187040

-- Given conditions
variables (R U: ℝ) (Uv : ℝ := 15) (IA : ℝ := 20)

-- Hypotheses derived from the condition
-- 1. Total voltage when two resistors in series
hypothesis hU : U = 2 * Uv
-- 2. Applying Ohm's law
hypothesis hOhm : U = IA * 2 * R

theorem find_resistance : R = 0.75 :=
by
  sorry

end find_resistance_l187_187040


namespace liu_hui_approx_pi_l187_187445

-- Assuming the necessary data and constants
variable (R : ℝ) (cos15deg : ℝ) (sqrt068 : ℝ)

-- The conditions provided
axiom h_cos15deg : cos15deg ≈ 0.966
axiom h_sqrt068 : sqrt068 ≈ 0.26

-- Definition of the problem statement
theorem liu_hui_approx_pi : 
    (π : ℝ) ≈ 3.12 :=
by
  -- Mark the proof as skipped
  sorry

end liu_hui_approx_pi_l187_187445


namespace largest_square_exists_l187_187312

theorem largest_square_exists (a b c m1 m2 m3 : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : m1 ≠ 0) (h₅ : m2 ≠ 0) (h₆ : m3 ≠ 0)
    (h_triangle : a > b ∧ b > c) :
    let x := (a * m1) / (m1 + a)
    let y := (b * m2) / (m2 + b)
    let z := (c * m3) / (m3 + c)
  in z > y ∧ y > x :=
by
  sorry

end largest_square_exists_l187_187312


namespace min_f_eq_2_m_n_inequality_l187_187342

def f (x : ℝ) := abs (x + 1) + abs (x - 1)

theorem min_f_eq_2 : (∀ x, f x ≥ 2) ∧ (∃ x, f x = 2) :=
by
  sorry

theorem m_n_inequality (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m^3 + n^3 = 2) : m + n ≤ 2 :=
by
  sorry

end min_f_eq_2_m_n_inequality_l187_187342


namespace sequence_a_not_periodic_l187_187787

def sequence_a (x : ℝ) (n : ℕ) : ℤ :=
  Int.floor (x^(n+1)) - x * Int.floor (x^n)

theorem sequence_a_not_periodic (x : ℝ) (h1 : 1 < x) (h2 : ¬ ∃ n : ℤ, x = n) :
  ¬ ∃ p : ℕ, ∀ n : ℕ, sequence_a x (n + p) = sequence_a x n :=
sorry

end sequence_a_not_periodic_l187_187787


namespace moles_of_NaOH_l187_187287

-- Define the reaction and stoichiometry
def balanced_reaction : String := "NaH + H2O → NaOH + H2"

def stoichiometry (NaH H2O NaOH H2 : ℕ) : Prop :=
  NaH = H2O ∧ H2O = NaOH ∧ NaOH = H2

-- Conditions
def NaH_moles : ℕ := 3
def H2O_moles : ℕ := 3

-- Define the proof problem
theorem moles_of_NaOH (NaH_moles H2O_moles NaOH_moles H2_moles : ℕ) 
  (h : stoichiometry NaH_moles H2O_moles NaOH_moles H2_moles) : NaOH_moles = 3 :=
  by
    -- assumptions based on the problem statement
    have h1 : NaH_moles = 3 := rfl
    have h2 : H2O_moles = 3 := rfl
    have h3 : stoichiometry NaH_moles H2O_moles NaOH_moles H2_moles := 
      by rw [h1, h2]; exact h
    -- conclusion
    sorry

end moles_of_NaOH_l187_187287


namespace john_total_distance_l187_187414

def speed : ℕ := 45
def time1 : ℕ := 2
def time2 : ℕ := 3

theorem john_total_distance:
  speed * (time1 + time2) = 225 := by
  sorry

end john_total_distance_l187_187414


namespace sum_of_digits_0_to_99_l187_187504

def sum_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_digits (n / 10)

theorem sum_of_digits_0_to_99 :
  (∀ n, 18 <= n → n <= 21 → sum_digits n = 24) →
  ∑ i in finset.range 100, sum_digits i = 900 :=
by
  sorry

end sum_of_digits_0_to_99_l187_187504


namespace topsoil_cost_correct_l187_187520

noncomputable def topsoilCost (price_per_cubic_foot : ℝ) (yard_to_foot : ℝ) (discount_threshold : ℝ) (discount_rate : ℝ) (volume_in_yards : ℝ) : ℝ :=
  let volume_in_feet := volume_in_yards * yard_to_foot
  let cost_without_discount := volume_in_feet * price_per_cubic_foot
  if volume_in_feet > discount_threshold then
    cost_without_discount * (1 - discount_rate)
  else
    cost_without_discount

theorem topsoil_cost_correct:
  topsoilCost 8 27 100 0.10 7 = 1360.8 :=
by
  sorry

end topsoil_cost_correct_l187_187520


namespace circumference_and_area_correct_l187_187575

noncomputable def circumference_of_semicircle_and_area_of_triangle : ℝ × ℝ :=
  let length := 14
  let breadth := 10
  let perimeter_rectangle := 2 * (length + breadth)
  let side_square := perimeter_rectangle / 4
  let diameter_semicircle := side_square
  let pi := 3.14
  let circumference_semicircle := (pi * diameter_semicircle) / 2 + diameter_semicircle
  let height_triangle := 15
  let area_triangle := (diameter_semicircle * height_triangle) / 2
  (Float.round_cent2 circumference_semicircle, Float.round_cent2 area_triangle)

theorem circumference_and_area_correct :
  let ans := circumference_of_semicircle_and_area_of_triangle
  ans.1 = 30.84 ∧ ans.2 = 90 := by
  sorry

end circumference_and_area_correct_l187_187575


namespace value_of_neg2_neg4_l187_187185

def operation (a b x y : ℤ) : ℤ := a * x - b * y

theorem value_of_neg2_neg4 (a b : ℤ) (h : operation a b 1 2 = 8) : operation a b (-2) (-4) = -16 := by
  sorry

end value_of_neg2_neg4_l187_187185


namespace max_value_of_expression_l187_187809

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l187_187809


namespace standard_eq_of_ellipse_area_triangle_ABF2_l187_187330

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  (y^2 / 10) + (x^2 / 6) = 1

theorem standard_eq_of_ellipse :
  ∃ (a b : ℝ), (∀ (x y : ℝ), ellipse_eq x y) ∧ a = sqrt 10 ∧ b = sqrt 6 := sorry

noncomputable def area_triangle (x1 y1 x2 y2 : ℝ) : ℝ :=
  1 / 2 * |y1 - y2| * (sqrt 15)

theorem area_triangle_ABF2 :
  ∃ (a b : ℝ), 
  let A := (-sqrt 15 / 2, sqrt 15 / 2) in
  let B := (sqrt 15 / 2, -sqrt 15 / 2) in 
  a = A ∧ b = B ∧ area_triangle (A.1) (A.2) (B.1) (B.2) = sqrt 15 := sorry

end standard_eq_of_ellipse_area_triangle_ABF2_l187_187330


namespace volume_of_cone_with_lateral_surface_unfolding_to_semicircle_l187_187563

theorem volume_of_cone_with_lateral_surface_unfolding_to_semicircle :
  ∀ (r_semicircle : ℝ) (r_base : ℝ) (h : ℝ), 
  r_semicircle = 2 → 
  r_base = 1 → 
  h = sqrt(3) → 
  (1/3) * π * r_base^2 * h = (sqrt(3) / 3) * π :=
by
  intros r_semicircle r_base h h_semicircle_eq h_base_eq h_eq
  sorry

end volume_of_cone_with_lateral_surface_unfolding_to_semicircle_l187_187563


namespace total_weight_gain_l187_187136

def orlando_gained : ℕ := 5

def jose_gained (orlando : ℕ) : ℕ :=
  2 * orlando + 2

def fernando_gained (jose : ℕ) : ℕ :=
  jose / 2 - 3

theorem total_weight_gain (O J F : ℕ) 
  (ho : O = orlando_gained) 
  (hj : J = jose_gained O) 
  (hf : F = fernando_gained J) :
  O + J + F = 20 :=
by
  sorry

end total_weight_gain_l187_187136


namespace bus_passengers_l187_187033

variable (P : ℕ) -- P represents the initial number of passengers

theorem bus_passengers (h1 : P + 16 - 17 = 49) : P = 50 :=
by
  sorry

end bus_passengers_l187_187033


namespace complex_product_solution_l187_187661

theorem complex_product_solution (x y : ℝ) (h : (1 + x * complex.i) * (1 - 2 * complex.i) = y) 
  : x = 2 ∧ y = 5 :=
sorry

end complex_product_solution_l187_187661


namespace tan_30_deg_l187_187169

theorem tan_30_deg : 
  let θ := (30 : ℝ) * (Real.pi / 180)
  in Real.sin θ = 1 / 2 ∧ Real.cos θ = Real.sqrt 3 / 2 → Real.tan θ = Real.sqrt 3 / 3 :=
by
  intro h
  let th := θ
  have h1 : Real.sin th = 1 / 2 := And.left h
  have h2 : Real.cos th = Real.sqrt 3 / 2 := And.right h
  sorry

end tan_30_deg_l187_187169


namespace savings_percentage_l187_187975

variables (I S : ℝ)
-- Conditions
-- A man saves a certain portion S of his income I during the first year.
-- He spends the remaining portion (I - S) on his personal expenses.
-- In the second year, his income increases by 50%, so his new income is 1.5I.
-- His savings increase by 100%, so his new savings are 2S.
-- His total expenditure in 2 years is double his expenditure in the first year.

def first_year_expenditure (I S : ℝ) : ℝ := I - S
def second_year_income (I : ℝ) : ℝ := 1.5 * I
def second_year_savings (S : ℝ) : ℝ := 2 * S
def second_year_expenditure (I S : ℝ) : ℝ := second_year_income I - second_year_savings S
def total_expenditure (I S : ℝ) : ℝ := first_year_expenditure I S + second_year_expenditure I S

theorem savings_percentage :
  total_expenditure I S = 2 * first_year_expenditure I S → S / I = 0.5 :=
by
  sorry

end savings_percentage_l187_187975


namespace smallest_n_l187_187064

theorem smallest_n (n : ℕ) : (∃ (m1 m2 : ℕ), 4 * n = m1^2 ∧ 5 * n = m2^3) ↔ n = 500 := 
begin
  sorry
end

end smallest_n_l187_187064


namespace largest_integer_y_l187_187046

theorem largest_integer_y (y : ℤ) : 
  (∃ k : ℤ, (y^2 + 3*y + 10) = k * (y - 4)) → y ≤ 42 :=
sorry

end largest_integer_y_l187_187046


namespace evaluate_pow_l187_187251

theorem evaluate_pow : 64^(5/6 : ℝ) = 32 := by
  sorry

end evaluate_pow_l187_187251


namespace symmetric_point_line_intercept_l187_187027

noncomputable def line_intercept_x_axis (k b : ℝ) : ℝ :=
-b / k

theorem symmetric_point_line_intercept 
  (k b : ℝ)
  (A B : ℝ × ℝ)
  (hA : A = (1, 3))
  (hB : B = (-2, 1))
  (midpoint_symmetric : (1/2 * (fst A + fst B), 1/2 * (snd A + snd B)) = (-1/2, 2))
  (line_equation : ∀ x y : ℝ, y = k * x + b → (y - snd A) / (x - fst A) = -1/k) :
  line_intercept_x_axis k b = 5/6 := 
sorry

end symmetric_point_line_intercept_l187_187027


namespace reciprocal_of_2022_l187_187007

noncomputable def reciprocal (x : ℝ) := 1 / x

theorem reciprocal_of_2022 : reciprocal 2022 = 1 / 2022 :=
by
  -- Define reciprocal
  sorry

end reciprocal_of_2022_l187_187007


namespace jordan_Oreos_count_l187_187767

variable (J : ℕ)
variable (OreosTotal : ℕ)
variable (JamesOreos : ℕ)

axiom James_Oreos_condition : JamesOreos = 2 * J + 3
axiom Oreos_total_condition : J + JamesOreos = OreosTotal
axiom Oreos_total_value : OreosTotal = 36

theorem jordan_Oreos_count : J = 11 :=
by 
  unfold OreosTotal JamesOreos
  sorry

end jordan_Oreos_count_l187_187767


namespace eval_power_l187_187206

-- Given condition
def sixty_four : ℕ := 64

-- Given condition rewritten in Lean
def sixty_four_as_two_powersix : sixty_four = 2^6 := by
  sorry

-- Prove that 64^(5/6) = 32
theorem eval_power : real.exp (5/6 * real.log 64) = 32 := by
  have h1 : 64 = 2^6 := sixty_four_as_two_powersix
  sorry

end eval_power_l187_187206


namespace max_q_value_l187_187805

theorem max_q_value (A M C : ℕ) (h : A + M + C = 15) : 
  (A * M * C + A * M + M * C + C * A) ≤ 200 :=
sorry

end max_q_value_l187_187805


namespace max_value_of_expression_l187_187810

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_expression_l187_187810


namespace exponential_solution_set_l187_187503

-- Definition and assumptions
def exponential_inequality (x : ℝ) : Prop :=
  2^(x^2 + 2*x - 4) ≤ 1/2

def quadratic_inequality (x : ℝ) : Prop :=
  -3 ≤ x ∧ x ≤ 1

-- Main statement
theorem exponential_solution_set (x : ℝ) :
  exponential_inequality x ↔ quadratic_inequality x := 
sorry

end exponential_solution_set_l187_187503


namespace projection_of_a_on_b_l187_187686

variable (a b : EuclideanSpace ℝ 3)
variable (ha : ‖a‖ = 6)
variable (hb : ‖b‖ = 3)
variable (hab : inner a b = -12)

theorem projection_of_a_on_b : (inner a b) / ‖b‖ = -4 := by
  sorry

end projection_of_a_on_b_l187_187686


namespace distance_C_to_D_l187_187940

-- Define the conditions
def smaller_square_perimeter := 8
def larger_square_area := 81

-- Define the side length of squares based on given conditions
def smaller_square_side := smaller_square_perimeter / 4
def larger_square_side := Real.sqrt larger_square_area

-- Define the length of the triangle sides
def horizontal_length := smaller_square_side + larger_square_side
def vertical_length := larger_square_side - smaller_square_side

-- Define the distance from C to D using Pythagoras' Theorem
def distance_CD := Real.sqrt (horizontal_length^2 + vertical_length^2)

-- Theorem to prove
theorem distance_C_to_D :
  (distance_CD = 13.0) := sorry

end distance_C_to_D_l187_187940


namespace cos2_theta_plus_2m_sin_theta_minus_2m_minus_2_neg_l187_187637

theorem cos2_theta_plus_2m_sin_theta_minus_2m_minus_2_neg (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π / 2) (m : ℝ) :
  cos θ ^ 2 + 2 * m * sin θ - 2 * m - 2 < 0 ↔ m > -1 / 2 :=
sorry

end cos2_theta_plus_2m_sin_theta_minus_2m_minus_2_neg_l187_187637


namespace smallest_maxima_d_l187_187082

def a (x : ℝ) := x ^ 2 / (1 + x ^ 12)
def b (x : ℝ) := x ^ 3 / (1 + x ^ 11)
def c (x : ℝ) := x ^ 4 / (1 + x ^ 10)
def d (x : ℝ) := x ^ 5 / (1 + x ^ 9)
def e (x : ℝ) := x ^ 6 / (1 + x ^ 8)

theorem smallest_maxima_d (x : ℝ) (hx : 0 < x) :
  (∀ x > 0, d x ≤ a x) ∧ (∀ x > 0, d x ≤ b x) ∧ (∀ x > 0, d x ≤ c x) ∧ (∀ x > 0, d x ≤ e x) :=
by
  sorry

end smallest_maxima_d_l187_187082


namespace determine_defective_coin_l187_187946

-- Define the properties of the coins
structure Coin :=
(denomination : ℕ)
(weight : ℕ)

-- Given coins
def c1 : Coin := ⟨1, 1⟩
def c2 : Coin := ⟨2, 2⟩
def c3 : Coin := ⟨3, 3⟩
def c5 : Coin := ⟨5, 5⟩

-- Assume one coin is defective
variable (defective : Coin)
variable (differing_weight : ℕ)
#check differing_weight

theorem determine_defective_coin :
  (∃ (defective : Coin), ∀ (c : Coin), 
    c ≠ defective → c.weight = c.denomination) → 
  ((c2.weight + c3.weight = c5.weight → defective = c1) ∧
   (c1.weight + c2.weight = c3.weight → defective = c5) ∧
   (c2.weight ≠ 2 → defective = c2) ∧
   (c3.weight ≠ 3 → defective = c3)) :=
by
  sorry

end determine_defective_coin_l187_187946


namespace ratio_shorter_longer_l187_187558

theorem ratio_shorter_longer (total_length shorter_length longer_length : ℝ)
  (h1 : total_length = 21) 
  (h2 : shorter_length = 6) 
  (h3 : longer_length = total_length - shorter_length) 
  (h4 : shorter_length / longer_length = 2 / 5) : 
  shorter_length / longer_length = 2 / 5 :=
by sorry

end ratio_shorter_longer_l187_187558


namespace evaluate_root_l187_187235

theorem evaluate_root : 64 ^ (5 / 6 : ℝ) = 32 :=
by sorry

end evaluate_root_l187_187235


namespace smallest_n_for_perfect_square_and_cube_l187_187070

theorem smallest_n_for_perfect_square_and_cube :
  ∃ n : ℕ, 0 < n ∧ (∃ a1 b1 : ℕ, 4 * n = a1 ^ 2 ∧ 5 * n = b1 ^ 3 ∧ n = 50) :=
begin
  use 50,
  split,
  { norm_num, },
  { use [10, 5],
    split,
    { norm_num, },
    { split, 
      { norm_num, },
      { refl, }, },
  },
  sorry
end

end smallest_n_for_perfect_square_and_cube_l187_187070


namespace alcohol_percentage_new_mixture_l187_187983

/--
Given:
1. The initial mixture has 15 liters.
2. The mixture contains 20% alcohol.
3. 5 liters of water is added to the mixture.

Prove:
The percentage of alcohol in the new mixture is 15%.
-/
theorem alcohol_percentage_new_mixture :
  let initial_mixture_volume := 15 -- in liters
  let initial_alcohol_percentage := 20 / 100
  let initial_alcohol_volume := initial_alcohol_percentage * initial_mixture_volume
  let added_water_volume := 5 -- in liters
  let new_total_volume := initial_mixture_volume + added_water_volume
  let new_alcohol_percentage := (initial_alcohol_volume / new_total_volume) * 100
  new_alcohol_percentage = 15 := 
by
  -- Proof steps go here
  sorry

end alcohol_percentage_new_mixture_l187_187983


namespace division_remainder_correct_l187_187048

def polynomial_div_remainder (x : ℝ) : ℝ :=
  3 * x^4 + 14 * x^3 - 50 * x^2 - 72 * x + 55

def divisor (x : ℝ) : ℝ :=
  x^2 + 8 * x - 4

theorem division_remainder_correct :
  ∀ x : ℝ, polynomial_div_remainder x % divisor x = 224 * x - 113 :=
by
  sorry

end division_remainder_correct_l187_187048


namespace tangent_line_at_1_l187_187916

noncomputable def f (x : ℝ) : ℝ := x^4 - 2 * x^3

theorem tangent_line_at_1 :
  let p := (1 : ℝ, f 1)
  in ∃ m c : ℝ, (∀ x : ℝ, y : ℝ, y = m * x + c ↔ y = -2 * x + 1) ∧ ∀ x : ℝ, f x = x^4 - 2 * x^3 :=
sorry

end tangent_line_at_1_l187_187916


namespace domain_of_function_l187_187611

/-- Determine the domain of the function y = (log base 10 (x + 1)) / (x - 2) --/
theorem domain_of_function (x : ℝ) : (∃ y, y = (Real.log (x + 1)) / (x - 2)) ↔ 
  ((x > -1) ∧ (x ≠ 2)) :=
begin
  sorry
end

end domain_of_function_l187_187611


namespace rain_at_least_once_l187_187490

noncomputable def rain_probability (day_prob : ℚ) (days : ℕ) : ℚ :=
  1 - (1 - day_prob)^days

theorem rain_at_least_once :
  ∀ (day_prob : ℚ) (days : ℕ),
    day_prob = 3/4 → days = 4 →
    rain_probability day_prob days = 255/256 :=
by
  intros day_prob days h1 h2
  sorry

end rain_at_least_once_l187_187490


namespace seq_eq_n_seq_shifted_l187_187707

-- Define the sequences a_n and b_n
def a (n : ℕ) : ℕ := sorry
def b (n : ℕ) : ℕ := sorry

-- Define the conditions for the problem
axiom seq_cond_1 (n : ℕ) (hn : n > 0) : (finset.range (n+1)).filter (λ k, a k ≤ n).card = b n
axiom seq_cond_2 (n : ℕ) (hn : n > 0) : (finset.range (n+1)).filter (λ k, b k ≤ n).card = a n

-- The first proof problem
theorem seq_eq_n (h : a 1 = b 1) : ∀ n, a n = n ∧ b n = n :=
by
  intro n
  sorry

-- The second proof problem
theorem seq_shifted (h : a 1 = b 1 + 2014) : 
  (∀ n, a n = 2013 + n) ∧ (∀ n, (1 ≤ n ∧ n ≤ 2013 → b n = 0) ∧ (n ≥ 2014 → b n = n - 2013)) :=
by
  split
  -- Proof for a n
  { intro n
    sorry }
  -- Proof for b n
  { intro n
    split
    -- Proof for 1 ≤ n ≤ 2013
    { intro hn
      sorry }
    -- Proof for n ≥ 2014
    { intro hn
      sorry }
  }

end seq_eq_n_seq_shifted_l187_187707


namespace eval_power_l187_187207

-- Given condition
def sixty_four : ℕ := 64

-- Given condition rewritten in Lean
def sixty_four_as_two_powersix : sixty_four = 2^6 := by
  sorry

-- Prove that 64^(5/6) = 32
theorem eval_power : real.exp (5/6 * real.log 64) = 32 := by
  have h1 : 64 = 2^6 := sixty_four_as_two_powersix
  sorry

end eval_power_l187_187207


namespace meet_time_l187_187550

noncomputable def kmph_to_mps (kmph : ℝ) : ℝ :=
  kmph * 1000 / 3600

def length_track : ℝ := 250
def speed1_kmph : ℝ := 20
def speed2_kmph : ℝ := 40
def speed1_mps : ℝ := kmph_to_mps speed1_kmph
def speed2_mps : ℝ := kmph_to_mps speed2_kmph

def relative_speed_mps : ℝ := speed1_mps + speed2_mps
def time_to_meet (length : ℝ) (relative_speed : ℝ) : ℝ := length / relative_speed

theorem meet_time :
  time_to_meet length_track relative_speed_mps = 15 := by
  sorry

end meet_time_l187_187550


namespace cubic_root_sum_l187_187424

noncomputable def poly : Polynomial ℝ := Polynomial.C (-4) + Polynomial.C 3 * X + Polynomial.C (-2) * X^2 + X^3

theorem cubic_root_sum (a b c : ℝ) (h1 : poly.eval a = 0) (h2 : poly.eval b = 0) (h3 : poly.eval c = 0)
  (h_sum : a + b + c = 2) (h_prod : a * b + b * c + c * a = 3) (h_triple_prod : a * b * c = 4) :
  a^3 + b^3 + c^3 = 2 := 
by {
  sorry
}

end cubic_root_sum_l187_187424


namespace compute_difference_square_l187_187832

-- Definitions
def numMultiplesOf4LessThan50 : ℕ := (List.range' 1 50).countp (λ n => n % 4 = 0)
def numMultiplesOf2And4LessThan50 : ℕ := (List.range' 1 50).countp (λ n => n % 2 = 0 ∧ n % 4 = 0)

-- Theorem statement
theorem compute_difference_square : (numMultiplesOf4LessThan50 - numMultiplesOf2And4LessThan50) ^ 2 = 0 := by
  sorry

end compute_difference_square_l187_187832


namespace earnings_bc_l187_187978

variable (A B C : ℕ)

theorem earnings_bc :
  A + B + C = 600 →
  A + C = 400 →
  C = 100 →
  B + C = 300 :=
by
  intros h1 h2 h3
  sorry

end earnings_bc_l187_187978


namespace find_T_value_l187_187823

variable (P : ℚ) (T : ℚ) (h : ℚ)

theorem find_T_value (h_eq : ∀ P, T = h * P + 3)
                     (cond1 : T = 20)
                     (P1 : P = 7)
                     (P2 : P = 12) :
  ∃ T, T = 32 + 1/7 :=
by {
  let P1 := 7,
  let P2 := 12,
  have h_val : h = 17 / 7 := by { sorry },
  have T_val : T = 32 + 1 / 7 := by {
    calc T = (17 / 7) * 12 + 3 : by { sorry }
       ... = 32 + 1 / 7       : by { sorry }
  },
  use T,
  exact T_val
}

end find_T_value_l187_187823


namespace smallest_integer_proof_l187_187466

noncomputable def smallest_integer_satisfying_conditions :
  Nat := 3466

theorem smallest_integer_proof :
  ∃ x > 1, ( x > 1 ∧ (x / (5 / 7) - (x / (5 / 7)).floor = 2 / 5)
                  ∧ (x / (7 / 9) - (x / (7 / 9)).floor = 2 / 7)
                  ∧ (x / (9 / 11) - (x / (9 / 11)).floor = 2 / 9)
                  ∧ (x / (11 / 13) - (x / (11 / 13)).floor = 2 / 11)) → x = smallest_integer_satisfying_conditions := 
by 
-- Proof omitted
sorry

end smallest_integer_proof_l187_187466


namespace spherical_coordinates_cone_l187_187638

open Real

-- Define spherical coordinates and the equation φ = c
def spherical_coordinates (ρ θ φ : ℝ) : Prop := 
  ∃ (c : ℝ), φ = c

-- Prove that φ = c describes a cone
theorem spherical_coordinates_cone (ρ θ : ℝ) (c : ℝ) :
  spherical_coordinates ρ θ c → ∃ ρ' θ', spherical_coordinates ρ' θ' c :=
by
  sorry

end spherical_coordinates_cone_l187_187638


namespace correct_statement_l187_187554

variables {R : Type*} [linear_ordered_field R]

structure line (R : Type*) :=
(point : R)
(direction : R)

structure plane (R : Type*) :=
(normal : R → R)
(offset : R)

variables (m n l : line R)
variables (α β : plane R)

def is_parallel (m n : line R) : Prop :=
m.direction = n.direction

def is_perpendicular (m : line R) (α : plane R) : Prop :=
α.normal m.direction = 0

def is_contained_in (m : line R) (α : plane R) : Prop :=
α.normal m.point = 0

def are_parallel (α β : plane R) : Prop :=
α.normal = β.normal

theorem correct_statement (m_perp_alpha : is_perpendicular m α)
    (n_perp_beta : is_perpendicular n β)
    (alpha_parallel_beta : are_parallel α β) :
  is_parallel m n :=
sorry

end correct_statement_l187_187554


namespace total_apples_for_bobbing_l187_187447

theorem total_apples_for_bobbing (apples_per_bucket : ℕ) (buckets : ℕ) (total_apples : ℕ) : 
  apples_per_bucket = 9 → buckets = 7 → total_apples = apples_per_bucket * buckets → total_apples = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_apples_for_bobbing_l187_187447


namespace asymptotes_of_hyperbola_slope_of_line_l187_187483

-- Problem (1) lean statement
theorem asymptotes_of_hyperbola
  (b : ℝ) (hb : b > 0) (a : ℝ) (ha : a = 1) :
  (∀ (F1 F2 A B : ℝ × ℝ),
    let c := Math.sqrt (1 + b^2),
    F1 = (-c, 0) ∧ F2 = (c, 0) ∧
    A = (c, b^2) ∧ B = (c, -b^2) ∧
    angle A B = Math.pi / 2 ∧
    equilateral_triangle F1 A B) →
  asymptotes_eq (x^2 - y^2 / b^2 = 1) (y = ± Math.sqrt(2) * x) :=
sorry

-- Problem (2) lean statement
theorem slope_of_line
  (b : ℝ) (hb : b = Math.sqrt(3)) :
  (∃ (k : ℝ),
    hyperbola_eq x^2 - y^2 / 3 = 1 ∧
    F1 = (-2, 0) ∧ F2 = (2, 0) ∧
    midpoint M A B ∧
    ⟪FM, AB⟫ = 0) →
  k = ± Math.sqrt(15) / 5 :=
sorry

end asymptotes_of_hyperbola_slope_of_line_l187_187483


namespace four_digit_even_numbers_five_digit_multiples_of_5_four_digit_numbers_gt_1325_l187_187960

-- Definitions for conditions
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}
def four_digit_numbers := {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ ∀ d ∈ n.digits, d ∈ digits ∧ n.digits.nodup}
def five_digit_numbers := {n : ℕ | 10000 ≤ n ∧ n < 100000 ∧ ∀ d ∈ n.digits, d ∈ digits ∧ n.digits.nodup}

-- Problem (1): Four-digit even numbers
theorem four_digit_even_numbers : 
  ∀ n ∈ four_digit_numbers, n % 2 = 0 → four_digit_numbers.card = 156 := 
sorry

-- Problem (2): Five-digit numbers that are multiples of 5
theorem five_digit_multiples_of_5 : 
  ∀ n ∈ five_digit_numbers, n % 5 = 0 → five_digit_numbers.card = 216 := 
sorry

-- Problem (3): Four-digit numbers greater than 1325
theorem four_digit_numbers_gt_1325 : 
  ∀ n ∈ four_digit_numbers, n > 1325 → four_digit_numbers.card = 270 := 
sorry

end four_digit_even_numbers_five_digit_multiples_of_5_four_digit_numbers_gt_1325_l187_187960


namespace tan_30_eq_sqrt3_div3_l187_187156

theorem tan_30_eq_sqrt3_div3 (sin_30_cos_30 : ℝ → ℝ → Prop)
  (h1 : sin_30_cos_30 (1 / 2) (Real.sqrt 3 / 2)) :
  ∃ t, t = Real.tan (Real.pi / 6) ∧ t = Real.sqrt 3 / 3 :=
by
  existsi Real.tan (Real.pi / 6)
  sorry

end tan_30_eq_sqrt3_div3_l187_187156


namespace total_pages_in_storybook_l187_187541

theorem total_pages_in_storybook
  (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) (Sₙ : ℕ) 
  (h₁ : a₁ = 12)
  (h₂ : d = 1)
  (h₃ : aₙ = 26)
  (h₄ : aₙ = a₁ + (n - 1) * d)
  (h₅ : Sₙ = n * (a₁ + aₙ) / 2) :
  Sₙ = 285 :=
by
  sorry

end total_pages_in_storybook_l187_187541


namespace evaluate_pow_l187_187246

theorem evaluate_pow : 64^(5/6 : ℝ) = 32 := by
  sorry

end evaluate_pow_l187_187246


namespace largest_inscribed_equilateral_triangle_area_l187_187142

noncomputable def side_length_of_equilateral_triangle (r : ℝ) : ℝ :=
  r * real.sqrt 3

noncomputable def area_of_equilateral_triangle (s : ℝ) : ℝ :=
  (real.sqrt 3 / 4) * s^2

theorem largest_inscribed_equilateral_triangle_area (r : ℝ) (h : r = 10) : 
  area_of_equilateral_triangle (side_length_of_equilateral_triangle r) = 75 * real.sqrt 3 :=
by 
  sorry

end largest_inscribed_equilateral_triangle_area_l187_187142


namespace sum_f_series_l187_187292

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 else 0

theorem sum_f_series : (∑ n in Finset.range 2007, f n) = 1336 := by
  sorry

end sum_f_series_l187_187292


namespace integer_values_of_a_l187_187642

theorem integer_values_of_a : 
  ∃ (a : Set ℤ), (∀ x, x ∈ a → ∃ (y z : ℤ), x^2 + x * y + 9 * y = 0) ∧ (a.card = 6) :=
by
  sorry

end integer_values_of_a_l187_187642


namespace sequence_non_zero_l187_187669

theorem sequence_non_zero :
  ∀ n : ℕ, ∃ a : ℕ → ℤ,
  (a 1 = 1) ∧
  (a 2 = 2) ∧
  (∀ n : ℕ, (a (n+1) % 2 = 1 ∧ a n % 2 = 1) → (a (n+2) = 5 * a (n+1) - 3 * a n)) ∧
  (∀ n : ℕ, (a (n+1) % 2 = 0 ∧ a n % 2 = 0) → (a (n+2) = a (n+1) - a n)) ∧
  (a n ≠ 0) :=
by
  sorry

end sequence_non_zero_l187_187669


namespace complex_norm_equality_l187_187101

theorem complex_norm_equality (z : ℂ) (h1 : ∥z∥ = ∥z - 1∥) (h2 : ∥z∥ = ∥z - complex.I∥) : 
  ∥z∥ = (Real.sqrt 2) / 2 := 
by 
  sorry

end complex_norm_equality_l187_187101


namespace equilateral_triangle_side_length_l187_187454

noncomputable def side_length_of_triangle (PQ PR PS : ℕ) : ℝ := 
  let s := 8 * Real.sqrt 3
  s

theorem equilateral_triangle_side_length (PQ PR PS : ℕ) (P_inside_triangle : true) 
  (Q_foot : true) (R_foot : true) (S_foot : true)
  (hPQ : PQ = 2) (hPR : PR = 4) (hPS : PS = 6) : 
  side_length_of_triangle PQ PR PS = 8 * Real.sqrt 3 := 
sorry

end equilateral_triangle_side_length_l187_187454


namespace initial_investment_eq_1000_l187_187589

theorem initial_investment_eq_1000 :
  ∃ P : ℝ, 
    (let r := 0.10 in
     let n := 2 in
     let t := 1 in
     let A := 1102.5 in
     P * ((1 + r / n) ^ (n * t)) = A) ↔ P = 1000 :=
by
  sorry

end initial_investment_eq_1000_l187_187589


namespace max_value_of_q_l187_187814

theorem max_value_of_q (A M C : ℕ) (h_sum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end max_value_of_q_l187_187814


namespace heaviest_lightest_difference_total_excess_shortfall_total_earnings_l187_187029

-- Condition: Differences from standard weight and number of baskets
def differences : List ℝ := [-3, -4, -4.5, 0, 2, 3.5]
def basket_counts : List ℕ := [5, 3, 2, 4, 3, 7]

-- Part 1: Prove the difference between heaviest and lightest basket
theorem heaviest_lightest_difference :
  let heaviest := 3.5
  let lightest := -4.5
  heaviest - lightest = 8 := by
  sorry

-- Part 2: Prove the total excess/shortfall compared to standard weight
theorem total_excess_shortfall :
  let total_excess := (differences.zip basket_counts).map (λ (diff, count), diff * count).sum
  total_excess = -5.5 := by
  sorry

-- Part 3: Prove the total earnings from selling the fruits
theorem total_earnings :
  let standard_weight := 24 * 40
  let shortfall := -5.5
  let price_per_kg := 2
  let total_weight := standard_weight + shortfall
  total_weight * price_per_kg = 1909 := by
  sorry

end heaviest_lightest_difference_total_excess_shortfall_total_earnings_l187_187029


namespace count_valid_numbers_l187_187367

open Nat

theorem count_valid_numbers : 
  let valid_numbers := 
    [ (x, y) | x ∈ [1, 3, 5, 7, 9], y ∈ [0..9],
    2 * x + y < 17 ∧ ¬ (2 * x + y) % 3 = 0 ] 
  in
  length valid_numbers = 24 :=
by
  -- We'll need to define the set comprehension manually in Lean
  let valid_x : List Nat := [1, 3, 5, 7, 9]
  let valid_y := range 10
  let valid_pairs := do
    x <- valid_x
    y <- valid_y
    guard (2 * x + y < 17 ∧ ¬ (2 * x + y) % 3 = 0)
    return (x, y)
  have h : valid_pairs.length = 24 := sorry
  exact h

end count_valid_numbers_l187_187367


namespace minimal_polynomial_with_roots_and_rational_coeffs_l187_187289

theorem minimal_polynomial_with_roots_and_rational_coeffs :
  ∃ p : Polynomial ℚ,
    p = Polynomial.monic (Polynomial.X^4 - 10 * Polynomial.X^3 + 27 * Polynomial.X^2 - 14 * Polynomial.X + 2) ∧
    (Polynomial.aeval (2 + Real.sqrt 5) p = 0) ∧
    (Polynomial.aeval (3 + Real.sqrt 7) p = 0) ∧
    Polynomial.monic p := 
by
  sorry

end minimal_polynomial_with_roots_and_rational_coeffs_l187_187289


namespace rate_is_15_l187_187096

variable (sum : ℝ) (interest12 : ℝ) (interest_r : ℝ) (r : ℝ)

-- Given conditions
def conditions : Prop :=
  sum = 7000 ∧
  interest12 = 7000 * 0.12 * 2 ∧
  interest_r = 7000 * (r / 100) * 2 ∧
  interest_r = interest12 + 420

-- The rate to prove
def rate_to_prove : Prop := r = 15

theorem rate_is_15 : conditions sum interest12 interest_r r → rate_to_prove r := 
by
  sorry

end rate_is_15_l187_187096


namespace find_minimum_value_l187_187340

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := x^2 + a * |x - 1| + 1

-- The statement of the proof problem
theorem find_minimum_value (a : ℝ) (h : a ≥ 0) :
  (a = 0 → ∀ x, f x a ≥ 1 ∧ ∃ x, f x a = 1) ∧
  ((0 < a ∧ a < 2) → ∀ x, f x a ≥ -a^2 / 4 + a + 1 ∧ ∃ x, f x a = -a^2 / 4 + a + 1) ∧
  (a ≥ 2 → ∀ x, f x a ≥ 2 ∧ ∃ x, f x a = 2) := 
by
  sorry

end find_minimum_value_l187_187340


namespace largest_gcd_l187_187020

theorem largest_gcd (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + b = 1008) : 
  ∃ d : ℕ, d = Int.gcd a b ∧ d = 504 :=
by
  sorry

end largest_gcd_l187_187020


namespace arabella_learning_time_l187_187590

noncomputable def arabella_total_time (step1 step2 step3 step4 step5 step6 step7 : ℝ) : ℝ := 
step1 + step2 + step3 + step4 + step5 + step6 + step7

theorem arabella_learning_time : 
  ∃ (step1 step2 step3 step4 step5 step6 step7 : ℝ),
    step1 = 50 ∧
    step2 = (1/3) * step1 ∧
    step3 = step1 + step2 ∧
    step4 = 1.75 * step1 ∧
    step5 = step2 + 25 ∧
    step6 = step3 + step5 - 40 ∧
    step7 = step1 + step2 + step4 + 10 ∧
    arabella_total_time step1 step2 step3 step4 step5 step6 step7 = 495.02 :=
by
  let step1 := 50
  let step2 := (1/3) * step1
  let step3 := step1 + step2
  let step4 := 1.75 * step1
  let step5 := step2 + 25
  let step6 := step3 + step5 - 40
  let step7 := step1 + step2 + step4 + 10
  exact ⟨step1, step2, step3, step4, step5, step6, step7, 
       by
         split; refl,
       by
         split; refl,
       by
         split; refl,
       by
         split; refl,
       by
         split; refl,
       by
         split; refl,
       by
         split; refl,
       by
         refl⟩

end arabella_learning_time_l187_187590


namespace ratio_AM_MC_l187_187841

variables (A B C D M : Type) [IsMidpoint D B C] [RightTriangle A B C] (angle_AMB angle_CMD : Real)
variable [Angle angle_AMB = Angle angle_CMD]
variable [PointOnSegment M A C]

theorem ratio_AM_MC (h : angle_AMB = angle_CMD) : AM / MC = 1 / 2 := sorry

end ratio_AM_MC_l187_187841


namespace min_frac_l187_187681

noncomputable def min_value (a b : ℝ) (h₁ : a + 2 * b = 1) (h₂ : b > 0) : ℝ :=
  (7 + 2 * Real.sqrt 6)

theorem min_frac : ∀ (a b : ℝ), a + 2 * b = 1 → b > 0 → 
  ∃ c, (c = 7 + 2 * Real.sqrt 6) ∧ (∀ x, x = c → x = min_value a b (by assumption) (by assumption)) :=
by
  intros a b h₁ h₂
  use 7 + 2* Real.sqrt 6
  split
  { refl }
  { intros x hx
    rw hx
    refl }
  sorry

end min_frac_l187_187681


namespace fermats_little_theorem_l187_187878

open Nat

theorem fermats_little_theorem (p a : ℕ) (hp : Prime p) (hdiv : ¬ p ∣ a) :
  a ^ (p - 1) ≡ 1 [MOD p] :=
sorry

end fermats_little_theorem_l187_187878


namespace train_b_speed_l187_187953

variable (v : ℝ) -- the speed of Train B

theorem train_b_speed 
  (speedA : ℝ := 30) -- speed of Train A
  (head_start_hours : ℝ := 2) -- head start time in hours
  (overtake_distance : ℝ := 285) -- distance at which Train B overtakes Train A
  (train_a_travel_distance : ℝ := speedA * head_start_hours) -- distance Train A travels in the head start time
  (total_distance : ℝ := 345) -- total distance Train B travels to overtake Train A
  (train_a_travel_time : ℝ := overtake_distance / speedA) -- time taken by Train A to travel the overtake distance
  : v * train_a_travel_time = total_distance → v = 36.32 :=
by
  sorry

end train_b_speed_l187_187953


namespace range_of_a_l187_187338

def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 + 2 * x else x - 1

theorem range_of_a (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (f x - a^2 + 2 * a = 0) ∧ 
  (f y - a^2 + 2 * a = 0) ∧ (f z - a^2 + 2 * a = 0)) ↔ (0 < a ∧ a < 1 ∨ 1 < a ∧ a < 2) := 
sorry

end range_of_a_l187_187338


namespace remainders_inequalities_l187_187356

theorem remainders_inequalities
  (X Y M A B s t u : ℕ)
  (h1 : X > Y)
  (h2 : X = Y + 8)
  (h3 : X % M = A)
  (h4 : Y % M = B)
  (h5 : s = (X^2) % M)
  (h6 : t = (Y^2) % M)
  (h7 : u = (A * B)^2 % M) :
  s ≠ t ∧ t ≠ u ∧ s ≠ u :=
sorry

end remainders_inequalities_l187_187356


namespace lemons_for_6_gallons_l187_187553

variable (lemons gallons : ℕ)
variable (constant_ratio : lemons.toFloat / gallons.toFloat = 36.toFloat / 48.toFloat)
variable (gallons_needed : ℕ := 6)
noncomputable def lemons_needed : ℝ := 4.5

theorem lemons_for_6_gallons (h : lemons.toFloat / gallons.toFloat = 36.toFloat / 48.toFloat) :
  lemons_needed = (lemons_needed * gallons_needed) / 6.toFloat :=
sorry

end lemons_for_6_gallons_l187_187553


namespace cuboid_surface_area_l187_187280

/--
Given a cuboid with length 10 cm, breadth 8 cm, and height 6 cm, the surface area is 376 cm².
-/
theorem cuboid_surface_area 
  (length : ℝ) 
  (breadth : ℝ) 
  (height : ℝ) 
  (h_length : length = 10) 
  (h_breadth : breadth = 8) 
  (h_height : height = 6) : 
  2 * (length * height + length * breadth + breadth * height) = 376 := 
by 
  -- Replace these placeholders with the actual proof steps.
  sorry

end cuboid_surface_area_l187_187280


namespace intersect_at_one_point_l187_187092

-- Definitions of points and circles
variable (Point : Type)
variable (Circle : Type)
variable (A : Point)
variable (C1 C2 C3 C4 : Circle)

-- Definition of intersection points
variable (B12 B13 B14 B23 B24 B34 : Point)

-- Note: Assumptions around the geometry structure axioms need to be defined
-- Assuming we have a function that checks if three points are collinear:
variable (are_collinear : Point → Point → Point → Prop)
-- Assuming we have a function that checks if a point is part of a circle:
variable (on_circle : Point → Circle → Prop)

-- Axioms related to the conditions
axiom collinear_B12_B34_B (hC1 : on_circle B12 C1) (hC2 : on_circle B12 C2) (hC3 : on_circle B34 C3) (hC4 : on_circle B34 C4) : 
  ∃ P : Point, are_collinear B12 P B34 

axiom collinear_B13_B24_B (hC1 : on_circle B13 C1) (hC2 : on_circle B13 C3) (hC3 : on_circle B24 C2) (hC4 : on_circle B24 C4) : 
  ∃ P : Point, are_collinear B13 P B24 

axiom collinear_B14_B23_B (hC1 : on_circle B14 C1) (hC2 : on_circle B14 C4) (hC3 : on_circle B23 C2) (hC4 : on_circle B23 C3) : 
  ∃ P : Point, are_collinear B14 P B23 

-- The theorem to be proved
theorem intersect_at_one_point :
  ∃ P : Point, 
    are_collinear B12 P B34 ∧ are_collinear B13 P B24 ∧ are_collinear B14 P B23 := 
sorry

end intersect_at_one_point_l187_187092


namespace ellipse_standard_eq_find_k1_line_MN_fixed_point_l187_187679

noncomputable theory

-- Given conditions and derived proofs:

-- Definition of the ellipse
def ellipse_eq (x y : ℝ) : Prop :=
  (x^2) / 3 + (y^2) / 2 = 1

-- Given point (1, 2√3/3) lies on the ellipse
axiom point_on_ellipse : ellipse_eq 1 (2 * real.sqrt 3 / 3)

-- Midpoint P(1,1) condition for chords with slopes k1 and k2
axiom midpoint_P (A B : ℝ × ℝ) (k1 : ℝ) : Prop :=
  let (x1, y1) := A in
  let (x2, y2) := B in
  x1 + x2 = 2 ∧ y1 + y2 = 2 ∧ k1 = (y2 - y1) / (x2 - x1)

-- k1 + k2 = 1 condition
axiom slopes_sum_one (k1 k2 : ℝ) : k1 + k2 = 1

-- Proof of the standard equation of the ellipse
theorem ellipse_standard_eq : 
  ∀ x y : ℝ, ellipse_eq x y ↔ x^2 / 3 + y^2 / 2 = 1 :=
by sorry

-- Proof that k1 = -2/3 if P is the midpoint of AB
theorem find_k1 (A B : ℝ × ℝ) (k1 : ℝ) (h : midpoint_P A B k1) : k1 = -2 / 3 :=
by sorry

-- Proof that line MN always passes through (0, -2/3)
theorem line_MN_fixed_point (k1 k2 : ℝ) (h1 : slopes_sum_one k1 k2) :
  ∀ M N : ℝ × ℝ, (some_function_to_define_MN M N k1 k2) (0, -2/3) :=
by sorry

end ellipse_standard_eq_find_k1_line_MN_fixed_point_l187_187679


namespace omega_value_l187_187737

noncomputable def smallestPositivePeriod (f : ℝ → ℝ) : ℝ := sorry

theorem omega_value (ω : ℝ) (h1 : ω > 0)
  (h2 : smallestPositivePeriod (λ x, Real.cos (ω * x - Real.pi / 6)) = Real.pi / 5) :
  ω = 10 :=
by
  sorry

end omega_value_l187_187737


namespace sum_four_digit_even_numbers_l187_187302

-- Define the digits set
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

-- Define the set of valid units digits for even numbers
def even_units : Finset ℕ := {0, 2, 4}

-- Define the set of all four-digit numbers using the provided digits
def four_digit_even_numbers : Finset ℕ :=
  (Finset.range (10000) \ Finset.range (1000)).filter (λ n =>
    n % 10 ∈ even_units ∧
    (n / 1000) ∈ digits ∧
    ((n / 100) % 10) ∈ digits ∧
    ((n / 10) % 10) ∈ digits)

theorem sum_four_digit_even_numbers :
  (four_digit_even_numbers.sum (λ x => x)) = 1769580 :=
  sorry

end sum_four_digit_even_numbers_l187_187302


namespace problem_statement_l187_187821

noncomputable theory

open_locale classical

variables {A B C P A' B' C' : Point}

axiom Point_on_circumcircle (A B C P : Point) : Prop
axiom Parallel (l1 l2 : Line) : Prop
axiom Line_through_points (P1 P2 : Point) : Line

def circumcircle_of_triangle (A B C : Point) : Circle :=
  sorry -- Placeholder for the definition of a circumcircle

def points_on_circle (C : Circle) (P : Point) : Prop :=
  sorry -- Placeholder for the definition of a point being on a circle

def parallel_lines (P1 P2 Q1 Q2 : Point) : Prop :=
  Parallel (Line_through_points P1 P2) (Line_through_points Q1 Q2)

theorem problem_statement
  (hp : Point_on_circumcircle A B C P)
  (hpa : parallel_lines P A' B C)
  (hpb : parallel_lines P B' C A)
  (hpc : parallel_lines P C' A B) :
  parallel_lines A A' B B' ∧ parallel_lines A A' C C' :=
sorry

end problem_statement_l187_187821


namespace monotonic_increase_interval_l187_187926

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem monotonic_increase_interval :
  {x : ℝ | ∀ a b, a ≤ x → x ≤ b → f(a) ≤ f(b)} = (Set.Ioi 2) :=
by
  sorry

end monotonic_increase_interval_l187_187926


namespace total_tires_l187_187293

def vehicles_data : Type := 
  (cars : ℕ) × 
  (bicycles : ℕ) × 
  (pickup_trucks : ℕ) × 
  (tricycles : ℕ)

def tire_counts (data : vehicles_data) : ℕ :=
  let cars := data.1
  let bicycles := data.2
  let pickup_trucks := data.3
  let tricycles := data.4
  (cars * 4) + (bicycles * 2) + (pickup_trucks * 4) + (tricycles * 3)

theorem total_tires (d : vehicles_data) : 
  d = ((15, 3, 8, 1) : vehicles_data) → 
  tire_counts d = 101 :=
by
  intro h
  simp [tire_counts, h]
  sorry

end total_tires_l187_187293


namespace odds_against_Z_winning_l187_187716

theorem odds_against_Z_winning
  (P_X_winning : ℚ := 1/5)
  (P_Y_winning : ℚ := 2/3)
  (P_total : P_X_winning + P_Y_winning + (1 - P_X_winning - P_Y_winning) = 1)
  (odds_X : P_X_winning = 1/(4 + 1))
  (odds_Y : P_Y_winning = 2/(1 + 2)) :
  (let P_Z_winning := (1 - P_X_winning - P_Y_winning) in
   let P_Z_losing := 1 - P_Z_winning in
   (P_Z_losing / P_Z_winning) = 13/2) :=
by
  sorry

end odds_against_Z_winning_l187_187716


namespace maximum_students_l187_187942

-- Definitions for conditions
def students (n : ℕ) := Fin n → Prop

-- Condition: Among any six students, there are two who are not friends
def not_friend_in_six (n : ℕ) (friend : Fin n → Fin n → Prop) : Prop :=
  ∀ (s : Finset (Fin n)), s.card = 6 → ∃ (a b : Fin n), a ∈ s ∧ b ∈ s ∧ ¬ friend a b

-- Condition: For any pair of students not friends, there is a student who is friends with both
def friend_of_two_not_friends (n : ℕ) (friend : Fin n → Fin n → Prop) : Prop :=
  ∀ (a b : Fin n), ¬ friend a b → ∃ (c : Fin n), c ≠ a ∧ c ≠ b ∧ friend c a ∧ friend c b

-- Theorem stating the main result
theorem maximum_students (n : ℕ) (friend : Fin n → Fin n → Prop) :
  not_friend_in_six n friend ∧ friend_of_two_not_friends n friend → n ≤ 25 := 
sorry

end maximum_students_l187_187942


namespace range_of_a_l187_187442

noncomputable def f (x a b : ℝ) : ℝ := (2 * x^2 - a * x + b) * Real.log (x - 1)

theorem range_of_a (a b : ℝ) (h1 : ∀ x > 1, f x a b ≥ 0) : a ≤ 6 :=
by 
  let x := 2
  have hb_eq : b = 2 * a - 8 :=
    by sorry
  have ha_le_6 : a ≤ 6 :=
    by sorry
  exact ha_le_6

end range_of_a_l187_187442


namespace minimum_t_is_2_l187_187670

noncomputable def minimum_t_value (t : ℝ) : Prop :=
  let A := (-t, 0)
  let B := (t, 0)
  let C := (Real.sqrt 3, Real.sqrt 6)
  let r := 1
  ∃ P : ℝ × ℝ, 
    (P.1 - (Real.sqrt 3))^2 + (P.2 - (Real.sqrt 6))^2 = r^2 ∧ 
    (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

theorem minimum_t_is_2 : (∃ t : ℝ, t > 0 ∧ minimum_t_value t) → ∃ t : ℝ, t = 2 :=
sorry

end minimum_t_is_2_l187_187670


namespace car_distance_l187_187991

theorem car_distance (initial_time new_ratio new_speed : ℕ) (h1 : initial_time = 6) (h2 : new_ratio = 3 / 2) (h3 : new_speed = 40) : 
  let new_time := initial_time * new_ratio
  let distance := new_speed * new_time
  distance = 360 :=
by 
  dunfold new_time distance 
  rw [h1, h2, h3]
  -- Sorry here indicates the proof step, which is omitted.
  sorry

end car_distance_l187_187991


namespace probability_of_one_red_ball_drawn_l187_187030

theorem probability_of_one_red_ball_drawn
  (red_balls black_balls white_balls : ℕ)
  (total_drawn : ℕ)
  (h_red : red_balls = 3)
  (h_black : black_balls = 4)
  (h_white : white_balls = 5)
  (h_total_drawn : total_drawn = 2) :
  let total_balls := red_balls + black_balls + white_balls in
  let total_ways := total_balls * (total_balls - 1) / 2 in
  let successful_ways := red_balls * (black_balls + white_balls) in
  (successful_ways : ℚ) / (total_ways : ℚ) = 9 / 22 :=
by
  sorry

end probability_of_one_red_ball_drawn_l187_187030


namespace liquid_ratio_l187_187557

-- Defining the initial state of the container.
def initial_volume : ℝ := 37.5
def removed_and_replaced_volume : ℝ := 15

-- Defining the process steps.
def fraction_remaining (total: ℝ) (removed: ℝ) : ℝ := (total - removed) / total
def final_volume_A (initial: ℝ) (removed: ℝ) : ℝ := (fraction_remaining initial removed)^2 * initial

-- The given problem and its conclusion as a theorem.
theorem liquid_ratio (initial_V : ℝ) (remove_replace_V : ℝ) 
  (h1 : initial_V = 37.5) (h2 : remove_replace_V = 15) :
  let final_A := final_volume_A initial_V remove_replace_V in
  let final_B := initial_V - final_A in
  final_A / final_B = 9 / 16 :=
by
  sorry

end liquid_ratio_l187_187557


namespace sequence_properties_l187_187329

variable {α : Type*} [AddCommGroup α] [Module ℕ α] [HasSmul ℕ α] [OrderedRing ℕ]

-- Given sequence and its sum
variable (a : ℕ → α) (S : ℕ → α)

-- Conditions
variable (is_arithmetic : ∀ n, S n = a n)
variable (initial_condition : a 1 = 2)
variable (recurrence_relation : ∀ n, a (n + 1) = 2 * a n + 3)
variable (arithmetic_sequence : ∀ n, a (n + 1) - a n = a (2 * n + 1) - a (2 * n))

def is_geometric_sequence (b : ℕ → α) := ∃ q : α, ∀ n, b (n + 1) = q * b n

-- Prove
theorem sequence_properties :
  (∀ n, S n = a n → ∃ d, ∀ n, a (n + 1) - a n = d) ∧
  (a 1 = 2 → (∀ n, a (n + 1) = 2 * a n + 3) → is_geometric_sequence (λ n, a n + 3)) ∧
  (∀ n, a (n + 1) - a n = a (2 * n + 1) - a (2 * n) → ∃ c, ∀ n, S (2 * n) - S n = c * n) := 
by sorry

end sequence_properties_l187_187329


namespace six_digit_number_unique_solution_l187_187468

theorem six_digit_number_unique_solution
    (a b c d e f : ℕ)
    (hN : (N : ℕ) = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f)
    (hM : (M : ℕ) = 100000 * d + 10000 * e + 1000 * f + 100 * a + 10 * b + c)
    (h_eq : 7 * N = 6 * M) :
    N = 461538 :=
by
  sorry

end six_digit_number_unique_solution_l187_187468


namespace distinct_integer_values_of_a_l187_187645

theorem distinct_integer_values_of_a : 
  let eq_has_integer_solutions (a : ℤ) : Prop := 
    ∃ (x y : ℤ), (x^2 + a*x + 9*a = 0) ∧ (y^2 + a*y + 9*a = 0) in
  (finset.univ.filter eq_has_integer_solutions).card = 5 := 
sorry

end distinct_integer_values_of_a_l187_187645


namespace sqrt_infinite_series_eq_two_l187_187919

theorem sqrt_infinite_series_eq_two (m : ℝ) (hm : 0 < m) :
  (m ^ 2 = 2 + m) → m = 2 :=
by {
  sorry
}

end sqrt_infinite_series_eq_two_l187_187919


namespace total_students_next_year_l187_187388

theorem total_students_next_year
  (current_girls : ℕ)
  (current_boys : ℕ)
  (ratio_boys_girls : ℕ → ℕ → Prop := λ b g, b * 5 = g * 8)
  (expected_girls_increase : ℕ → ℕ := λ g, g * 2 / 10)
  (expected_boys_increase : ℕ → ℕ := λ b, b * 15 / 100)
  (total_students_next_year : ℕ := current_girls + (expected_girls_increase current_girls) + current_boys + (expected_boys_increase current_boys)) :
  current_girls = 190 → ratio_boys_girls current_boys current_girls → total_students_next_year = 577 :=
by
  intro hgirls hrato
  sorry

end total_students_next_year_l187_187388


namespace distinct_integer_values_of_a_l187_187646

theorem distinct_integer_values_of_a : 
  let eq_has_integer_solutions (a : ℤ) : Prop := 
    ∃ (x y : ℤ), (x^2 + a*x + 9*a = 0) ∧ (y^2 + a*y + 9*a = 0) in
  (finset.univ.filter eq_has_integer_solutions).card = 5 := 
sorry

end distinct_integer_values_of_a_l187_187646


namespace speed_of_man_l187_187121

/-
  Problem Statement:
  A train 100 meters long takes 6 seconds to cross a man walking at a certain speed in the direction opposite to that of the train. The speed of the train is 54.99520038396929 kmph. What is the speed of the man in kmph?
-/
 
theorem speed_of_man :
  ∀ (length_of_train : ℝ) (time_to_cross : ℝ) (speed_of_train_kmph : ℝ) (relative_speed_mps : ℝ),
    length_of_train = 100 →
    time_to_cross = 6 →
    speed_of_train_kmph = 54.99520038396929 →
    relative_speed_mps = length_of_train / time_to_cross →
    (relative_speed_mps - (speed_of_train_kmph * (1000 / 3600))) * (3600 / 1000) = 5.00479961403071 :=
by
  intros length_of_train time_to_cross speed_of_train_kmph relative_speed_mps
  intros h1 h2 h3 h4
  sorry

end speed_of_man_l187_187121


namespace lara_savings_exceed_4_on_sunday_l187_187781

def deposit_sequence_first_exceed_4 (initial_deposit : ℕ) (multiplier : ℕ) : ℕ :=
  Nat.find (λ n, (initial_deposit * multiplier^n) > 400)

theorem lara_savings_exceed_4_on_sunday :
  deposit_sequence_first_exceed_4 2 3 = 6 := sorry

end lara_savings_exceed_4_on_sunday_l187_187781


namespace twice_shorter_vs_longer_l187_187987

-- Definitions and conditions
def total_length : ℝ := 20
def shorter_length : ℝ := 8
def longer_length : ℝ := total_length - shorter_length

-- Statement to prove
theorem twice_shorter_vs_longer :
  2 * shorter_length - longer_length = 4 :=
by
  sorry

end twice_shorter_vs_longer_l187_187987


namespace percentage_problem_l187_187376

theorem percentage_problem (x : ℝ)
  (h : 0.70 * 600 = 0.40 * x) : x = 1050 :=
sorry

end percentage_problem_l187_187376


namespace relative_height_of_mountain_l187_187753

/-- In summer, the temperature on the mountain decreases by 0.7°C for every 100 meters increase in altitude.
    Given that the temperature at the summit is 14.1°C and the temperature at the foot of the mountain is 26°C,
    prove that the relative height of the mountain is 1700 meters. -/
theorem relative_height_of_mountain
  (decrease_rate : ℝ) (summit_temp : ℝ) (foot_temp : ℝ) (relative_height : ℝ) :
  decrease_rate = 0.7 →
  summit_temp = 14.1 →
  foot_temp = 26 →
  relative_height = (foot_temp - summit_temp) / (decrease_rate / 100) →
  relative_height = 1700 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end relative_height_of_mountain_l187_187753


namespace no_integer_solution_l187_187139

theorem no_integer_solution (n : ℤ) : ¬ (∃ n : ℤ, (((n : ℂ) + complex.I)^5).im = 0) :=
by
  sorry

end no_integer_solution_l187_187139


namespace total_percent_reduction_l187_187106

theorem total_percent_reduction (P : ℝ) :
  let price_after_first_discount := 0.8 * P,
      price_after_second_discount := 0.7 * price_after_first_discount,
      price_after_third_discount := 0.9 * price_after_second_discount,
      total_reduction := P - price_after_third_discount,
      percent_reduction := (total_reduction / P) * 100 in
  percent_reduction = 49.6 := by
  sorry

end total_percent_reduction_l187_187106


namespace a_investment_l187_187977

theorem a_investment
  (b_investment : ℝ) (c_investment : ℝ) (c_share_profit : ℝ) (total_profit : ℝ)
  (h1 : b_investment = 45000)
  (h2 : c_investment = 50000)
  (h3 : c_share_profit = 36000)
  (h4 : total_profit = 90000) :
  ∃ A : ℝ, A = 30000 :=
by {
  sorry
}

end a_investment_l187_187977


namespace percentage_decrease_l187_187992

theorem percentage_decrease (initial_price : ℝ) (increase_rate : ℝ) (stabilized_price : ℝ)
  (h1 : initial_price = 50) (h2 : increase_rate = 0.30) (h3 : stabilized_price = 52) :
  let increased_price := initial_price * (1 + increase_rate) in
  let decrease := increased_price - stabilized_price in
  let percentage_decrease := (decrease / increased_price) * 100 in
  percentage_decrease = 20 :=
by
  rw [h1, h2, h3]
  let increased_price := 50 * (1 + 0.30)
  let decrease := increased_price - 52
  let percentage_decrease := (decrease / increased_price) * 100
  have h1 : increased_price = 65 by norm_num
  have h2 : decrease = 13 by norm_num
  have h3 : percentage_decrease = 20 by norm_num
  exact h3

end percentage_decrease_l187_187992


namespace linear_regression_prediction_l187_187387

-- Define the problem conditions
variables (x : ℝ) (y : ℝ) (b : ℝ) (a : ℝ) (x_bar : ℝ) (y_bar : ℝ)

-- Given conditions
def regression_equation : Prop := y = b * x + a
def x_bar_value : Prop := x_bar = 5
def y_bar_value : Prop := y_bar = 56
def b_value : Prop := b = 10.5

-- The target value to prove
def y_prediction : Prop := y = 108.5

-- Statement of the problem in Lean (theorem without proof)
theorem linear_regression_prediction (h1 : regression_equation)
    (h2 : x_bar_value) (h3 : y_bar_value) (h4 : b_value) :
    y_prediction :=
  sorry

end linear_regression_prediction_l187_187387


namespace problem1_problem2_problem3_problem4_l187_187090

-- Statement for the first problem
theorem problem1 (α : ℝ) (h1 : α ∈ Icc (Real.pi / 2) Real.pi) (h2 : Real.tan α = -1 / 2) : Real.cos α = -2 * Real.sqrt 5 / 5 :=
sorry

-- Statement for the second problem
theorem problem2 (x α : ℝ) (hx : 0 < x ∧ x < 1) (hα : 0 < α ∧ α < 1) 
  (a b c : ℝ) (ha : a = x^α) (hb : b = x^(α / 2)) (hc : c = x^(1 / α)) : c < a ∧ a < b :=
sorry

-- Statement for the third problem
theorem problem3 (A B C D : ℝ) (BC BD AD : ℝ) 
  (hBC : BC = 3 * BD) (hAD : AD = Real.sqrt 2) (h_angle : Real.angle A D B = 135) (hAC : AC = Real.sqrt 2 * AB) : BD = 2 + Real.sqrt 5 :=
sorry

-- Statement for the fourth problem
def is_convex (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
∃ f'' : ℝ → ℝ, ∀ x ∈ D, Real.derivative (Real.derivative f x) < 0

theorem problem4 : 
  (is_convex (λ x => Real.sin x + Real.cos x) (Set.Ioo 0 (Real.pi / 2)) ∧ 
  is_convex (λ x => Real.ln x - 2 * x) (Set.Ioo 0 (Real.pi / 2)) ∧
  is_convex (λ x => -x^3 + 2*x - 1) (Set.Ioo 0 (Real.pi / 2)) 
  ∧ ¬ is_convex (λ x => x * Real.exp x) (Set.Ioo 0 (Real.pi / 2))) :=
sorry

end problem1_problem2_problem3_problem4_l187_187090


namespace concurrent_lines_l187_187785

theorem concurrent_lines {A B C D : Type} [ConvexQuadrilateral A B C D]
  (I J K L : Type) [Incenter I A B C] [Excenter J A B C] [Incenter K A C D] [Excenter L A C D] :
  ConcurrentLines IL JK (angle_bisector B C D) :=
sorry

end concurrent_lines_l187_187785


namespace eval_power_l187_187208

-- Given condition
def sixty_four : ℕ := 64

-- Given condition rewritten in Lean
def sixty_four_as_two_powersix : sixty_four = 2^6 := by
  sorry

-- Prove that 64^(5/6) = 32
theorem eval_power : real.exp (5/6 * real.log 64) = 32 := by
  have h1 : 64 = 2^6 := sixty_four_as_two_powersix
  sorry

end eval_power_l187_187208


namespace isosceles_triangle_angle_l187_187401

theorem isosceles_triangle_angle (A B C : Type*) (AB BC : A) (t : ℝ) 
  (h_isosceles : AB = BC) (h_angle : ∠ABC = t) : 
  ∠BAC = 90 - (1/2) * t :=
by
  sorry

end isosceles_triangle_angle_l187_187401


namespace composite_divisors_non_coprime_circle_l187_187277

open Nat

theorem composite_divisors_non_coprime_circle (n : ℕ) (h : n > 1) (composite_n : ¬ Prime n) 
    : (∃ p : ℕ, Prime p ∧ ∃ m : ℕ, m ≥ 2 ∧ n = p^m) 
      ∨ (∃ k : ℕ, k ≥ 2 ∧ (∃ f : Fin k → ℕ, (∀ i, Prime (f i)) ∧ n = (Finset.univ.prod f) 
      ∧ (k > 2 ∨ ∧ ∃ m1 m2 : ℕ, m1 ≥ 1 ∧ m2 ≥ 1 ∧ n = (f 0)^m1 * (f 1)^m2))) :=
sorry

end composite_divisors_non_coprime_circle_l187_187277


namespace flux_of_a_through_surface_l187_187283

noncomputable def flux_of_vector_field (a : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ) 
  (surface : ℝ × ℝ → ℝ) (z_cutoff : ℝ) (vector_field : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem flux_of_a_through_surface :
  let vector_field (p : ℝ × ℝ × ℝ) := (0, (p.2) ^ 2, p.3)
  let surface (p : ℝ × ℝ) := (p.1)^2 + (p.2)^2
  let z_cutoff := 2
  flux_of_vector_field vector_field surface z_cutoff = -2 * Real.pi := by
  sorry

end flux_of_a_through_surface_l187_187283


namespace find_a_l187_187666

theorem find_a (a : ℝ) : 
  (∀ x ∈ set.Icc (1 : ℝ) 2, (λ x : ℝ, a * x + 1) x ≤ (2 * a + 1) ∧ (λ x : ℝ, a * x + 1) x ≥ (a + 1)) →
  (2 * a + 1) - (a + 1) = 2 → a = 2 ∨ a = -2 :=
by
  sorry

end find_a_l187_187666
