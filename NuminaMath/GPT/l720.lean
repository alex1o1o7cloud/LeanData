import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Group.Enum
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Lcm
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Int.Lemmas
import Mathlib.Data.List
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Seq.Combinatorics
import Mathlib.Geometry.Polygon
import Mathlib.Init.Data.Int.Basic
import Mathlib.LinearAlgebra.VectorSpace
import Mathlib.NumberTheory.Prime
import Mathlib.Parametricity.Init
import Mathlib.Probability.Chebyshev
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic

namespace triangle_ineq_bisector_circumcircle_l720_720450

theorem triangle_ineq_bisector_circumcircle
  (A B C W : Type*)
  [field A]
  [ordered_field B]
  [metric_space C]
  [add_comm_group W]
  (b c AW : ℝ) 
  (triangle : IsTriangle A B C) 
  (circumcircle : IsCircumcircle A B C)
  (angle_bisector : IsAngleBisector A B C W)
  (intersection : IntersectsAtBisector A B C W (circumcircle)) :
  b + c ≤ 2 * AW :=
by
  sorry

end triangle_ineq_bisector_circumcircle_l720_720450


namespace smallest_five_digit_number_divisible_l720_720690

def smallest_prime_divisible (n: ℕ) : Prop :=
  ∃ k: ℕ, n = 2310 * k ∧ 10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_number_divisible :
  ∃ (n: ℕ), smallest_prime_divisible n ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_l720_720690


namespace set_intersection_complement_l720_720331

open Set

noncomputable def A : Set ℝ := { x | abs (x - 1) > 2 }
noncomputable def B : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }
noncomputable def notA : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }
noncomputable def targetSet : Set ℝ := { x | 2 < x ∧ x ≤ 3 }

theorem set_intersection_complement :
  (notA ∩ B) = targetSet :=
  by
  sorry

end set_intersection_complement_l720_720331


namespace percent_increase_l720_720426

variable {F : Type} [Field F]

theorem percent_increase (F F' : F) (h1 : 0.3 * F <-> 0.45 * F) (h2 : (3/7) * F' = 0.45 * F) : 
  F' = 1.05 * F := 
  sorry

end percent_increase_l720_720426


namespace decreasing_in_interval_l720_720599

-- Define the function f(x) = ax^2 - b
def f (a b : ℝ) (x : ℝ) := a * x^2 - b

-- The main theorem statement:
theorem decreasing_in_interval (a b : ℝ) :
  (∀ x : ℝ, x < 0 → f a b x ≤ f a b (x + 1)) ↔ (a > 0 ∧ b ∈ set.univ) := 
sorry

end decreasing_in_interval_l720_720599


namespace total_spent_by_friends_l720_720603

noncomputable def discounted_price (original_price : ℚ) (discount_percentage : ℚ) : ℚ :=
  original_price - (original_price * discount_percentage / 100)

noncomputable def total_cost_with_tax (initial_cost : ℚ) (tax_percentage : ℚ) : ℚ :=
  initial_cost * (1 + tax_percentage / 100)

theorem total_spent_by_friends :
  let tshirt_price := discounted_price 20 40 in
  let hat_price := discounted_price 15 60 in
  let bracelet_price := discounted_price 10 30 in
  let belt_price := discounted_price 10 50 in
  let friend1_cost := tshirt_price + hat_price + bracelet_price in
  let friend2_3_4_cost := 3 * (tshirt_price + hat_price + belt_price) in
  let total_cost_before_tax := friend1_cost + friend2_3_4_cost in
  let total_cost_after_tax := total_cost_with_tax total_cost_before_tax 5 in
  total_cost_after_tax = 98.70 :=
by
  sorry

end total_spent_by_friends_l720_720603


namespace integral_f_l720_720980

noncomputable def f (x m : ℝ) : ℝ := x^2 + 2*x + m

theorem integral_f (m : ℝ) (h : ∀ x : ℝ, f x m = x^2 + 2*x + m ∧ ∃ y : ℝ, f y m = -1) :
  ∫ x in 1..2, f x m = 16 / 3 := by
  -- f(x) = x^2 + 2x + m
  -- ∀ x ∈ ℝ, f(x) has a minimum value of -1
  sorry

end integral_f_l720_720980


namespace probability_of_circle_in_square_l720_720862

open Real Set

theorem probability_of_circle_in_square :
  ∃ (p : ℝ), (∀ x y : ℝ, x ∈ Icc (-1 : ℝ) 1 → y ∈ Icc (-1 : ℝ) 1 → (x^2 + y^2 < 1/4) → True)
  → p = π / 16 :=
by
  use π / 16
  sorry

end probability_of_circle_in_square_l720_720862


namespace MN_parallel_PQ_l720_720946

-- Definitions and conditions
variable {A B C I Z M N P Q : Type}
variable [is_center_of_circumcircle I A B C]
variable [midpoint_of_arc I A C]
variable [angle_bisector_of IZ (∠ B)]
variable [perpendicular PQ IZ]
variable [isosceles_triangle M BI]
variable [isosceles_triangle N BI]

-- The theorem statement
theorem MN_parallel_PQ (h_center: is_center_of_circumcircle I A B C) 
  (h_midpoint: midpoint_of_arc I A C) (h_bisector: angle_bisector_of IZ (∠ B)) 
  (h_perpendicular: perpendicular PQ IZ) (h_mbi: isosceles_triangle M BI) 
  (h_nbi: isosceles_triangle N BI) : 
  parallel MN PQ :=
sorry

end MN_parallel_PQ_l720_720946


namespace infinite_divisibility_l720_720393

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ (∀ n, odd n → a (n+1) = 2 + a n) ∧ (∀ n, even n → a (n+1) = 2 * a n)

def div_by_b (a : ℕ → ℕ) (p : ℕ) (b : ℕ) : Prop :=
  ∃ n, a n % b = 0

theorem infinite_divisibility (a : ℕ → ℕ) (p : ℕ) (b : ℕ) :
  sequence a → p.prime ∧ p > 3 → b = (2^(2*p) - 1) / 3 → (∃∞ n, div_by_b a p b) :=
sorry

end infinite_divisibility_l720_720393


namespace triangles_cover_base_l720_720293

-- Definitions
variable {S : Type} {A : Type} [Inhabited S] [Inhabited A] [RealSpace S] [ConvexPolygon A]
variables (A1 A2 An : A) (SA1 SA2 : S) (n : ℕ)

-- Conditions
def is_pyramid (S A1 A2 … An : Type) : Prop :=
  convex_polygon (A1 A2 … An)

def is_congruent_triangle (X A B : Type) (S A B : Type) : Prop :=
  ∃ X, congruent (triangle X A B) (triangle S A B) ∧
       same_side (line A B) (polygon_base A B)

def base_covered_by_triangles (X1 X2 ... Xn : Type) (A1 A2 ... An : Type) : Prop :=
  ∀ P ∈ polygon_base (A1 A2 ... An), ∃ i, P ∈ triangle (Xi Ai A(i+1))

-- Lean statement
theorem triangles_cover_base (S A1 A2 ... An : Type) (X1 X2 ... Xn : Type)
  (hp : is_pyramid S A1 A2 ... An)
  (ht : ∀ i, is_congruent_triangle (Xi Ai A(i+1)) (S Ai A(i+1))) :
  base_covered_by_triangles (X1 X2 ... Xn) (A1 A2 ... An) :=
sorry

end triangles_cover_base_l720_720293


namespace decimal_to_fraction_equivalence_l720_720048

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l720_720048


namespace smallest_degree_of_polynomial_l720_720969

theorem smallest_degree_of_polynomial :
  ∃ (p : Polynomial ℚ), 
    (3 - Real.sqrt 6) ∈ p.roots ∧
    (3 + Real.sqrt 6) ∈ p.roots ∧
    (5 + Real.sqrt 15) ∈ p.roots ∧
    (5 - Real.sqrt 15) ∈ p.roots ∧
    (16 - 2 * Real.sqrt 10) ∈ p.roots ∧
    (16 + 2 * Real.sqrt 10) ∈ p.roots ∧
    (- Real.sqrt 3) ∈ p.roots ∧
    (Real.sqrt 3) ∈ p.roots ∧
    p.degree = 8 :=
sorry

end smallest_degree_of_polynomial_l720_720969


namespace number_of_valid_n_l720_720829

theorem number_of_valid_n : 
  ∃ (c : Nat), (∀ n : Nat, (n + 9) * (n - 4) * (n - 13) < 0 → n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 11 ∨ n = 12) ∧ c = 11 :=
by
  sorry

end number_of_valid_n_l720_720829


namespace part_a_part_c_part_d_l720_720507

-- Define the variables
variables {a b : ℝ}

-- Define the conditions and statements
def cond := a + b > 0

theorem part_a (h : cond) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem part_c (h : cond) : a^21 + b^21 > 0 :=
sorry

theorem part_d (h : cond) : (a + 2) * (b + 2) > a * b :=
sorry

end part_a_part_c_part_d_l720_720507


namespace fencing_cost_200_dollars_l720_720995

noncomputable def cost_of_fencing_park (area : ℝ) (ratio_len : ℝ) (ratio_wid : ℝ) (cost_per_meter_paise : ℝ) (paise_to_dollar : ℝ) : ℝ :=
  let (x : ℝ) := real.sqrt (area / (ratio_len * ratio_wid))
  let len := ratio_len * x
  let wid := ratio_wid * x
  let perimeter := 2 * (len + wid)
  let cost_per_meter_dollars := cost_per_meter_paise / paise_to_dollar
  perimeter * cost_per_meter_dollars

theorem fencing_cost_200_dollars :
  cost_of_fencing_park 3750 3 2 80 100 = 200 := 
by 
  sorry

end fencing_cost_200_dollars_l720_720995


namespace decimal_to_fraction_equivalence_l720_720051

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l720_720051


namespace petya_time_comparison_l720_720216

variables (D V : ℝ) (hD_pos : D > 0) (hV_pos : V > 0)

theorem petya_time_comparison (hD_pos : D > 0) (hV_pos : V > 0) :
  (41 * D / (40 * V)) > (D / V) :=
by
  sorry

end petya_time_comparison_l720_720216


namespace number_of_intersection_points_l720_720245

-- Definitions of the given lines
def line1 (x y : ℝ) : Prop := 6 * y - 4 * x = 2
def line2 (x y : ℝ) : Prop := x + 2 * y = 2
def line3 (x y : ℝ) : Prop := -4 * x + 6 * y = 3

-- Definitions of the intersection points
def intersection1 (x y : ℝ) : Prop := line1 x y ∧ line2 x y
def intersection2 (x y : ℝ) : Prop := line2 x y ∧ line3 x y

-- Definition of the problem
theorem number_of_intersection_points : 
  (∃ x y : ℝ, intersection1 x y) ∧
  (∃ x y : ℝ, intersection2 x y) ∧
  (¬ ∃ x y : ℝ, line1 x y ∧ line3 x y) →
  (∃ z : ℕ, z = 2) :=
sorry

end number_of_intersection_points_l720_720245


namespace frank_problems_each_type_l720_720645

theorem frank_problems_each_type (bill_total : ℕ) (ryan_ratio bill_total_ratio : ℕ) (frank_ratio ryan_total : ℕ) (types : ℕ)
  (h1 : bill_total = 20)
  (h2 : ryan_ratio = 2)
  (h3 : bill_total_ratio = bill_total * ryan_ratio)
  (h4 : ryan_total = bill_total_ratio)
  (h5 : frank_ratio = 3)
  (h6 : ryan_total * frank_ratio = ryan_total) :
  (ryan_total * frank_ratio) / types = 30 :=
by
  sorry

end frank_problems_each_type_l720_720645


namespace log_eq_solutions_l720_720441

noncomputable def solve_log_equation : Set ℝ := 
  { x | log x 10 + 2 * log (10 * x) 10 + 3 * log (100 * x) 10 = 0 }

noncomputable def excluded_values : Set ℝ := 
  {0, 1, 1/10, 1/100}

theorem log_eq_solutions : 
  solve_log_equation \ excluded_values = 
  {10^(((-5 + Real.sqrt 13) / 6)), 10^(((-5 - Real.sqrt 13) / 6))} := sorry

end log_eq_solutions_l720_720441


namespace systematic_sampling_fifth_student_l720_720362

theorem systematic_sampling_fifth_student {
  -- Define the total number of students.
  total_students : ℕ,
  -- Define the selected students.
  selected_students : set ℕ
} (h_total : total_students = 55)
  (h_selected : selected_students = {4, 15, 26, 48}) :
  ∃ n, n ∉ selected_students ∧ n = 37 ∧ set.count _ (selected_students ∪ {n}) = 5 :=
by
  sorry

end systematic_sampling_fifth_student_l720_720362


namespace equation_solution_l720_720965

theorem equation_solution (x : ℝ) (h : x ≠ 2) :
  (3 * x + 6) / (x^2 + 5 * x - 14) = (3 - x) / (x - 2) ↔ x = 3 ∨ x = -5 :=
by 
  sorry

end equation_solution_l720_720965


namespace diff_PA_AQ_const_l720_720661

open Real

def point := (ℝ × ℝ)

noncomputable def distance (p1 p2 : point) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem diff_PA_AQ_const (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  let P := (0, -sqrt 2)
  let Q := (0, sqrt 2)
  let A := (a, sqrt (a^2 + 1))
  distance P A - distance A Q = 2 := 
sorry

end diff_PA_AQ_const_l720_720661


namespace lena_glued_friends_pictures_l720_720879

-- Define the conditions
def clippings_per_friend : ℕ := 3
def glue_per_clipping : ℕ := 6
def total_glue : ℕ := 126

-- Define the proof problem statement
theorem lena_glued_friends_pictures : 
    ∃ (F : ℕ), F * (clippings_per_friend * glue_per_clipping) = total_glue ∧ F = 7 := 
by
  sorry

end lena_glued_friends_pictures_l720_720879


namespace train_length_l720_720629

theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) (speed_kmh_val : speed_kmh = 180) (time_sec_val : time_sec = 7) :
  let speed_ms := (speed_kmh * 1000) / 3600 in
  let distance := speed_ms * time_sec in
  distance = 350 :=
by
  -- Given values
  rw [speed_kmh_val, time_sec_val]
  -- Convert speed from km/hr to m/s
  let speed_ms := (180 * 1000) / 3600
  -- Calculate distance
  let distance := speed_ms * 7
  -- Prove distance is 350 meters
  have speed_computation : speed_ms = 50 := by sorry
  rw speed_computation
  let distance := 50 * 7
  have length_computation : distance = 350 := by sorry
  exact length_computation

end train_length_l720_720629


namespace verify_toothpick_count_l720_720553

def toothpick_problem : Prop :=
  let L := 45
  let W := 25
  let Mv := 8
  let Mh := 5
  -- Calculate the total number of vertical toothpicks
  let verticalToothpicks := (L + 1 - Mv) * W
  -- Calculate the total number of horizontal toothpicks
  let horizontalToothpicks := (W + 1 - Mh) * L
  -- Calculate the total number of toothpicks
  let totalToothpicks := verticalToothpicks + horizontalToothpicks
  -- Ensure the total matches the expected result
  totalToothpicks = 1895

theorem verify_toothpick_count : toothpick_problem :=
by
  sorry

end verify_toothpick_count_l720_720553


namespace math_problem_l720_720467

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end math_problem_l720_720467


namespace cupcakes_initial_count_l720_720745

theorem cupcakes_initial_count (x : ℕ) (h1 : x - 5 + 10 = 24) : x = 19 :=
by sorry

end cupcakes_initial_count_l720_720745


namespace angle_between_bisectors_l720_720433

theorem angle_between_bisectors (β γ : ℝ) (h_sum : β + γ = 130) : (β / 2) + (γ / 2) = 65 :=
by
  have h : β + γ = 130 := h_sum
  sorry

end angle_between_bisectors_l720_720433


namespace range_of_a_l720_720802

def f (x : ℝ) : ℝ := x^3 - 6 * x + 5

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f(x₁) = a ∧ f(x₂) = a ∧ f(x₃) = a) ↔ 
  5 - 4 * Real.sqrt 2 < a ∧ a < 5 + 4 * Real.sqrt 2 :=
sorry

end range_of_a_l720_720802


namespace longTengNumber2012th_l720_720357

def isLongTengNumber (a : ℕ) : Prop :=
  (a.digits 10).sum = 5

theorem longTengNumber2012th : ∃ (a : ℕ), isLongTengNumber a ∧ ∃ (seq : ℕ → ℕ), (∀ n, isLongTengNumber (seq n)) ∧ 
  (strict_mono seq) ∧ (seq 2012 = 300200) :=
sorry

end longTengNumber2012th_l720_720357


namespace circle_radius_l720_720608

theorem circle_radius (P Q : ℝ) (h1 : P = π * r^2) (h2 : Q = 2 * π * r) (h3 : P / Q = 15) : r = 30 :=
by
  sorry

end circle_radius_l720_720608


namespace incorrect_statement_C_l720_720199

-- Define the predicates for the statements in the problem
def is_doubling_DNA_during_interphase_of_mitosis : Prop :=
  ∀ (cell : Type), during_interphase_of_mitosis(cell) → number_of_DNA_molecules(cell) = 2 * initial_number_of_DNA_molecules(cell)

def is_doubling_chromosomes_in_late_phase_of_mitosis_due_to_centromere_division : Prop :=
  ∀ (cell : Type), in_late_phase_of_mitosis(cell) → number_of_chromosomes(cell) = 2 * initial_number_of_chromosomes(cell)

def is_unchanged_DNA_molecules_in_second_meiotic_division : Prop :=
  ∀ (cell : Type), in_late_phase_of_second_meiotic_division(cell) → number_of_DNA_molecules(cell) = initial_number_of_DNA_molecules(cell)

def is_halving_chromosomes_in_first_meiotic_division_due_to_homologous_chromosome_separation : Prop :=
  ∀ (cell : Type), in_late_phase_of_first_meiotic_division(cell) → number_of_chromosomes(cell) = initial_number_of_chromosomes(cell) / 2

-- Theorem to prove that the incorrect statement is statement C
theorem incorrect_statement_C :
  is_doubling_DNA_during_interphase_of_mitosis →
  is_doubling_chromosomes_in_late_phase_of_mitosis_due_to_centromere_division →
  is_unchanged_DNA_molecules_in_second_meiotic_division →
  ¬ is_halving_chromosomes_in_first_meiotic_division_due_to_homologous_chromosome_separation :=
by
  intros
  sorry

end incorrect_statement_C_l720_720199


namespace part_a_part_c_part_d_l720_720512

-- Define the variables
variables {a b : ℝ}

-- Define the conditions and statements
def cond := a + b > 0

theorem part_a (h : cond) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem part_c (h : cond) : a^21 + b^21 > 0 :=
sorry

theorem part_d (h : cond) : (a + 2) * (b + 2) > a * b :=
sorry

end part_a_part_c_part_d_l720_720512


namespace sum_of_integers_a_is_13_l720_720998

noncomputable def sum_of_all_integers_a : ℤ :=
  let a_values := { a : ℤ | a > 2 ∧ a < 7 ∧ a ≠ 5 }
  a_values.to_finset.sum

theorem sum_of_integers_a_is_13 : sum_of_all_integers_a = 13 := 
  sorry

end sum_of_integers_a_is_13_l720_720998


namespace periodic_points_1989_l720_720141

section

variables {m : ℕ} (hm : m > 1)
def unit_circle := {z : ℂ // complex.abs z = 1}
def f (z : unit_circle) : unit_circle := ⟨z.val^m, by { rw [complex.abs_pow, z.property, one_pow], exact one_eq_one }⟩
def f_iter (k : ℕ) (z : unit_circle) : unit_circle := nat.iterate f k z

def is_periodic_point (n : ℕ) (z : unit_circle) : Prop :=
(f_iter m hm n z = z) ∧ ∀ i < n, f_iter m hm i z ≠ z

def count_periodic_points (n : ℕ) : ℕ :=
(fix : ℕ := (m ^ n - 1)) - (fix 117) - (fix 153) - (fix 663) + (fix 51) + (fix 39) + (fix 9) - (fix 3)

theorem periodic_points_1989 :
  count_periodic_points 1989 m = m ^ 1989 - m ^ 117 - m ^ 153 - m ^ 663 + m ^ 51 + m ^ 39 + m ^ 9 - m ^ 3 := 
sorry

end

end periodic_points_1989_l720_720141


namespace trillion_in_scientific_notation_l720_720857

theorem trillion_in_scientific_notation :
  (10^4) * (10^4) * (10^4) = 10^(12) := 
by sorry

end trillion_in_scientific_notation_l720_720857


namespace kim_cousins_l720_720875

theorem kim_cousins (pieces_per_cousin : ℕ) (total_pieces : ℕ) (h_pieces_per_cousin : pieces_per_cousin = 5) (h_total_pieces : total_pieces = 20) :
  total_pieces / pieces_per_cousin = 4 :=
by
  rw [h_pieces_per_cousin, h_total_pieces]
  norm_num

end kim_cousins_l720_720875


namespace number_of_correct_propositions_l720_720823

-- Definitions of the entities
variables {m n : Type}  -- m and n are types representing the lines
variables {α β : Type}  -- α and β are types representing the planes

-- Defining parallelism and perpendicularity predicates
variables (parallel_line_plane : m → α → Prop)
variables (perpendicular_line_plane : m → α → Prop)
variables (parallel_plane_plane : α → β → Prop)
variables (perpendicular_plane_plane : α → β → Prop)
variables (parallel_line_line : m → n → Prop)
variables (perpendicular_line_line : m → n → Prop)

-- Proving that the number of correct propositions is 2
theorem number_of_correct_propositions : 
  let p1 := ∀ m n α β, (parallel_line_plane m α) → (parallel_line_plane n β) → (parallel_plane_plane α β) → (parallel_line_line m n) in
  let p2 := ∀ m n α β, (perpendicular_line_plane m α) → (parallel_line_plane n β) → (parallel_plane_plane α β) → (perpendicular_line_line m n) in
  let p3 := ∀ m n α β, (parallel_line_plane m α) → (perpendicular_line_plane n β) → (perpendicular_plane_plane α β) → (parallel_line_line m n) in
  let p4 := ∀ m n α β, (perpendicular_line_plane m α) → (perpendicular_line_plane n β) → (perpendicular_plane_plane α β) → (perpendicular_line_line m n) in
  (¬p1) ∧ p2 ∧ (¬p3) ∧ p4 ↔ 2 :=
sorry

end number_of_correct_propositions_l720_720823


namespace lines_parallel_l720_720399

variables {A B C : ℝ} {a b c : ℝ}

-- Assuming the conditions
def triangle_side_lengths (a b c : ℝ) (A B C : ℝ) := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  ∃ s : ℝ, s = (a + b + c) / 2 ∧
  a + b > c ∧ b + c > a ∧ a + c > b ∧
  (sin A / a) = (sin B / b)

-- Two lines under analysis
def line1 (x y : ℝ) := sin A * x + a * y + c = 0
def line2 (x y : ℝ) := sin B * x + b * y = 0

theorem lines_parallel 
  (h_triangle : triangle_side_lengths a b c A B C) :
  Parallel_lines (∃ x y : ℝ, line1 x y) (∃ x y : ℝ, line2 x y) :=
sorry


end lines_parallel_l720_720399


namespace max_tan_A_minus_B_l720_720379

open Real

-- Given conditions
variables {A B C a b c : ℝ}

-- Assume the triangle ABC with sides a, b, c opposite to angles A, B, C respectively
-- and the equation a * cos B - b * cos A = (3 / 5) * c holds.
def condition (a b c A B C : ℝ) : Prop :=
  a * cos B - b * cos A = (3 / 5) * c

-- Prove that the maximum value of tan(A - B) is 3/4
theorem max_tan_A_minus_B (a b c A B C : ℝ) (h : condition a b c A B C) :
  ∃ t : ℝ, t = tan (A - B) ∧ 0 ≤ t ∧ t ≤ 3 / 4 :=
sorry

end max_tan_A_minus_B_l720_720379


namespace total_cost_of_mangoes_l720_720541

-- Definition of prices per dozen in one box
def prices_per_dozen : List ℕ := [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

-- Number of dozens per box (constant for all boxes)
def dozens_per_box : ℕ := 10

-- Number of boxes
def number_of_boxes : ℕ := 36

-- Calculate the total cost of mangoes in all boxes
theorem total_cost_of_mangoes :
  (prices_per_dozen.sum * number_of_boxes = 3060) := by
  -- Proof goes here
  sorry

end total_cost_of_mangoes_l720_720541


namespace avg_annual_reduction_l720_720632

theorem avg_annual_reduction (x : ℝ) (hx : (1 - x)^2 = 0.64) : x = 0.2 :=
by
  sorry

end avg_annual_reduction_l720_720632


namespace minimum_value_2x_plus_y_l720_720349

theorem minimum_value_2x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + y + 6 = x * y) : 
  2 * x + y ≥ 12 := 
sorry

end minimum_value_2x_plus_y_l720_720349


namespace math_problem_l720_720468

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end math_problem_l720_720468


namespace log_sequence_sum_l720_720818

def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, 5^(a (n+1)) = 25 * 5^(a n)

theorem log_sequence_sum (a : ℕ → ℝ) (h1 : sequence a)
  (h2 : a 2 + a 4 + a 6 = 9) :
  Real.logBase (1 / 3) (a 5 + a 7 + a 9) = -3 - Real.logBase 3 11 :=
  sorry

end log_sequence_sum_l720_720818


namespace area_comparison_l720_720395

-- Define the side lengths of the triangles
def a₁ := 17
def b₁ := 17
def c₁ := 12

def a₂ := 17
def b₂ := 17
def c₂ := 16

-- Define the semiperimeters
def s₁ := (a₁ + b₁ + c₁) / 2
def s₂ := (a₂ + b₂ + c₂) / 2

-- Define the areas using Heron's formula
noncomputable def area₁ := (s₁ * (s₁ - a₁) * (s₁ - b₁) * (s₁ - c₁)).sqrt
noncomputable def area₂ := (s₂ * (s₂ - a₂) * (s₂ - b₂) * (s₂ - c₂)).sqrt

-- The theorem to prove
theorem area_comparison : area₁ < area₂ := sorry

end area_comparison_l720_720395


namespace reciprocal_of_neg_one_seventh_l720_720991

theorem reciprocal_of_neg_one_seventh :
  (∃ x : ℚ, - (1 / 7) * x = 1) → (-7) * (- (1 / 7)) = 1 :=
by
  sorry

end reciprocal_of_neg_one_seventh_l720_720991


namespace number_of_classical_probability_models_is_one_l720_720859

noncomputable def classical_probability_model (condition : ℕ) : Prop :=
  match condition with
  | 2 => true 
  | _ => false

theorem number_of_classical_probability_models_is_one :
  (card {n : ℕ // classical_probability_model n}) = 1 :=
  by
    sorry

end number_of_classical_probability_models_is_one_l720_720859


namespace line_intersects_circle_l720_720531

theorem line_intersects_circle (a : ℝ) :
  ∃ t : ℝ × ℝ, (ax - t.snd + 2a + 1 = 0) ∧ (t.fst^2 + t.snd^2 = 9) := sorry

end line_intersects_circle_l720_720531


namespace smallest_five_digit_number_divisible_by_first_five_primes_l720_720706

theorem smallest_five_digit_number_divisible_by_first_five_primes : 
  ∃ n, (n >= 10000) ∧ (n < 100000) ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_first_five_primes_l720_720706


namespace sphere_radius_same_volume_l720_720159

noncomputable def tent_radius : ℝ := 3
noncomputable def tent_height : ℝ := 9

theorem sphere_radius_same_volume : 
  (4 / 3) * Real.pi * ( (20.25)^(1/3) )^3 = (1 / 3) * Real.pi * tent_radius^2 * tent_height :=
by
  sorry

end sphere_radius_same_volume_l720_720159


namespace olaf_travels_miles_l720_720914

-- Define the given conditions
def men : ℕ := 25
def per_day_water_per_man : ℚ := 1 / 2
def boat_mileage_per_day : ℕ := 200
def total_water : ℚ := 250

-- Define the daily water consumption for the crew
def daily_water_consumption : ℚ := men * per_day_water_per_man

-- Define the number of days the water will last
def days_water_lasts : ℚ := total_water / daily_water_consumption

-- Define the total miles traveled
def total_miles_traveled : ℚ := days_water_lasts * boat_mileage_per_day

-- Theorem statement to prove the total miles traveled is 4000 miles
theorem olaf_travels_miles : total_miles_traveled = 4000 := by
  sorry

end olaf_travels_miles_l720_720914


namespace max_chord_length_l720_720794

theorem max_chord_length 
  (θ : ℝ)
  (curve : ℝ → ℝ → Prop := λ x y, 2 * (2 * sin θ - cos θ + 3) * x^2 - (8 * sin θ + cos θ + 1) * y = 0)
  (line : ℝ → ℝ → Prop := λ x y, y = 2 * x) :
  ∃ l_max, l_max = 8 * sqrt 5 :=
by
  sorry

end max_chord_length_l720_720794


namespace angle_BAC_measure_l720_720867

theorem angle_BAC_measure {A X Y B C : Type} [InnerProductSpace ℝ (A × B × C)] 
    (AX XY YB BC : ℝ) (h1 : AX = XY) (h2 : XY = YB) (h3 : YB = BC) 
    (angle_ABC : ℝ) (h4 : angle_ABC = 150) : 
    ∃ t : ℝ, t = 7.5 ∧ angle (B - A) (C - A) = t :=
by
  sorry

end angle_BAC_measure_l720_720867


namespace proof_angle_A_proof_area_l720_720279

variables {A B C a b c : ℝ}
variables {m n : ℝ × ℝ}
variables {AM : ℝ}

def m := (Real.cos A, Real.cos C)
def n := (c - 2*b, a)

def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Given conditions: m and n are perpendicular
axiom h1 : perpendicular m n

-- Result 1: Determine the measure of angle A
def A_is_120 (A : ℝ) : Prop := A = 120 * Real.pi / 180

-- Given conditions for part 2
axiom h2 : B = 60 * Real.pi / 180
axiom h3 : AM = Real.sqrt 3

-- Result 2: Determine the area of triangle ABC
def area_triangle (a b c : ℝ) (A : ℝ) : ℝ :=
  1 / 2 * b * c * Real.sin A

def area_is_correct (area : ℝ) : Prop :=
  area = 3 * Real.sqrt 3 / 7

theorem proof_angle_A : A_is_120 A := sorry

theorem proof_area : area_is_correct (area_triangle a b c A) := sorry

end proof_angle_A_proof_area_l720_720279


namespace germination_probability_l720_720981

noncomputable def binomial_dist (n : ℕ) (p : ℝ) : pmf ℕ := sorry

theorem germination_probability (n : ℕ) (p : ℝ) (a b : ℕ) (μ σ² : ℝ) (ε : ℝ) :
  n = 1000 →
  p = 0.75 →
  a = 700 →
  b = 800 →
  μ = n * p →
  σ² = n * p * (1 - p) →
  ε = 50 →
  a ≤ μ - ε →
  b ≥ μ + ε →
  Pr (X ∈ Ioc a b) ≥ 0.925 :=
sorry

end germination_probability_l720_720981


namespace symmetric_circle_C_l720_720313

variables {a b : ℝ}

-- Condition: Symmetry of points with respect to line l
def symmetric_point (P P' : ℝ × ℝ) : Prop :=
  P'.1 = P.2 + 1 ∧ P'.2 = P.1 - 1

-- Given the original circle equation
def circle_C := (x y : ℝ) → x^2 + y^2 - 6 * x - 2 * y = 0

-- The standard form of circle C
def standard_form_C := (x y : ℝ) → (x - 3)^2 + (y - 1)^2 = 10

-- Condition translated to Lean: Center of C is symmetric to a point (2,2)
lemma center_symmetric_to_line (l : ℝ) :=
  symmetric_point (3, 1) (2, 2)

-- Prove that equation of the circle C' is (x-2)^2+(y-2)^2=10
theorem symmetric_circle_C' : (x : ℝ) (y : ℝ) → ((x - 2)^2 + (y - 2)^2 = 10) := by
  sorry

end symmetric_circle_C_l720_720313


namespace find_minimum_value_l720_720807

open Real

theorem find_minimum_value (x y z : ℝ) (hx : 0 ≤ x) (hx' : x ≤ π) (hy : 0 ≤ y) (hy' : y ≤ π) (hz : 0 ≤ z) (hz' : z ≤ π) :
  ∃ (min_val : ℝ), min_val = -1 ∧ min_val = min (cos (x - y) + cos (y - z) + cos (z - x)) := 
sorry

end find_minimum_value_l720_720807


namespace smallest_positive_five_digit_number_divisible_by_first_five_primes_l720_720719

theorem smallest_positive_five_digit_number_divisible_by_first_five_primes :
  ∃ n : ℕ, (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ 10000 ≤ n ∧ n = 11550 :=
by
  use 11550
  split
  · intros p hp
    fin_cases hp <;> norm_num
  split
  · norm_num
  rfl

end smallest_positive_five_digit_number_divisible_by_first_five_primes_l720_720719


namespace base7_difference_l720_720265

def from_base7 (s : string) : ℕ :=
  let digits := s.toList.reverse
  digits.enum.map (λ ⟨i, c⟩, (c.toNat - '0'.toNat) * 7 ^ i).sum

noncomputable def to_base7 (n : ℕ) : string :=
  if n = 0 then "0"
  else
    let rec aux (n : ℕ) : string :=
      if n = 0 then "" else
        let (q, r) := n.divMod 7
        (char.ofNat (r + '0'.toNat)).toString ++ aux q
    aux n |>.reverse

theorem base7_difference :
  to_base7 (from_base7 "5321" - from_base7 "1234") = "4054" := by
  sorry

end base7_difference_l720_720265


namespace maria_profit_l720_720911

theorem maria_profit (cost_per_5 : ℕ) (sell_per_4 : ℕ) (cost_amount : ℕ) (sell_amount : ℕ) (profit : ℕ) :
  cost_per_5 = 5 → sell_per_4 = 4 → cost_amount = 6 → sell_amount = 7 → profit = 150 → 
  ceil ((profit : ℚ) / ((sell_amount * cost_per_5 - cost_amount * sell_per_4 : ℚ) / (cost_per_5 * sell_per_4 : ℚ))) = 273 :=
by
  intros
  sorry

end maria_profit_l720_720911


namespace quadratic_is_perfect_square_l720_720666

theorem quadratic_is_perfect_square (c : ℝ) :
  (∃ b : ℝ, (3 * (x : ℝ) + b)^2 = 9 * x^2 - 24 * x + c) ↔ c = 16 :=
by sorry

end quadratic_is_perfect_square_l720_720666


namespace math_problem_l720_720470

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end math_problem_l720_720470


namespace smallest_positive_five_digit_number_divisible_by_first_five_primes_l720_720720

theorem smallest_positive_five_digit_number_divisible_by_first_five_primes :
  ∃ n : ℕ, (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ 10000 ≤ n ∧ n = 11550 :=
by
  use 11550
  split
  · intros p hp
    fin_cases hp <;> norm_num
  split
  · norm_num
  rfl

end smallest_positive_five_digit_number_divisible_by_first_five_primes_l720_720720


namespace mutually_exclusive_not_contradictory_l720_720755

namespace BallProbability
  -- Definitions of events based on the conditions
  def at_least_two_white (outcome : Multiset (String)) : Prop := 
    Multiset.count "white" outcome ≥ 2

  def all_red (outcome : Multiset (String)) : Prop := 
    Multiset.count "red" outcome = 3

  -- Problem statement
  theorem mutually_exclusive_not_contradictory :
    ∀ outcome : Multiset (String),
    Multiset.card outcome = 3 →
    (at_least_two_white outcome → ¬all_red outcome) ∧
    ¬(∀ outcome, at_least_two_white outcome ↔ ¬all_red outcome) := 
  by
    intros
    sorry
end BallProbability

end mutually_exclusive_not_contradictory_l720_720755


namespace quadratic_real_roots_l720_720294

theorem quadratic_real_roots (m : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + 2 * x1 - 1 + m = 0 ∧ x2^2 + 2 * x2 - 1 + m = 0) ↔ m ≤ 2 :=
by
  sorry

end quadratic_real_roots_l720_720294


namespace average_score_l720_720364

theorem average_score (M F : ℕ) (h1 : M = 0.4 * (M + F))
  (h2 : ¬ (M + F = 0)) -- Avoid division by zero
  (h3 : ∀ s, s = 75 ∨ s = 80 -> s ∈ finset.univ.filter (λ s, s ∈ finset.univ)) : 
  (75 * M + 80 * F) / (M + F) = 78 := 
by 
  have hF : F = 0.6 * (M + F), from show F, by sorry 
  have M + F ≠ 0, from h2 
  calc
    (75 * M + 80 * F) / (M + F)
    = (75 * 0.4 * (M + F) + 80 * 0.6 * (M + F)) / (M + F) : by rw [h1, hF]
    have hcancel : (75 * 0.4 * (M + F) + 80 * 0.6 * (M + F)) / (M + F), from sorry  -- Calculation
    = (75 * 0.4 + 80 * 0.6) : by sorry  -- Simplification
    = 30 + 48 : by sorry  -- Substitution and addition
    = 78 : by sorry -- Final answer

end average_score_l720_720364


namespace simplify_fraction_l720_720343

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h_cond : y^3 - 1/x ≠ 0) :
  (x^3 - 1/y) / (y^3 - 1/x) = x / y :=
by
  sorry

end simplify_fraction_l720_720343


namespace collinear_vectors_λ_l720_720332

def vector (α : Type) [Add α] [Mul α] := (α × α)

variables {α : Type} [AddCommGroup α] [Mul α] [HasOne α] [OfNat α 2]

theorem collinear_vectors_λ (a b c : vector α) (λ : α) :
  a = (1, 2) → b = (2, 0) → c = (1, -2) →
  collinear (λ • a + b) c → λ = -1 :=
begin
  -- Definitions
  sorry
end

-- Definition to check collinearity
def collinear {α : Type} [Field α] (u v : vector α) : Prop :=
  ∃ (k : α), (u.1 = k * v.1) ∧ (u.2 = k * v.2)

end collinear_vectors_λ_l720_720332


namespace difference_of_squares_eval_l720_720675

-- Define the conditions
def a : ℕ := 81
def b : ℕ := 49

-- State the corresponding problem and its equivalence
theorem difference_of_squares_eval : (a^2 - b^2) = 4160 := by
  sorry -- Placeholder for the proof

end difference_of_squares_eval_l720_720675


namespace value_of_r_is_2_l720_720658

noncomputable def quadratic_function : ℝ → ℝ → ℝ
| x, r => 2*x^2 + 3*x + r

def max_value_within_interval (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
max (f a) (f b)

theorem value_of_r_is_2 : (∃ r : ℝ, max_value_within_interval (quadratic_function x) (-2) 0 = 4) → r = 2 :=
sorry

end value_of_r_is_2_l720_720658


namespace atmospheric_pressure_600m_l720_720641

noncomputable def c : ℝ := 1.01 * 10^5
noncomputable def k : ℝ := (1/1000) * Real.log (0.90 / 1.01)

theorem atmospheric_pressure_600m :
    let p : ℝ := c * Real.exp (600 * k) in
    p ≈ 9.43 * 10^4 :=
  sorry

end atmospheric_pressure_600m_l720_720641


namespace good_circles_count_l720_720753

/-- Definition of the "good circle" property as per the given problem. -/
def good_circle {P : Type} [metric_space P] (A B C D E : P) (circle : set P) : Prop :=
  (A ∈ circle ∧ B ∈ circle ∧ C ∈ circle ∧ 
  (D ∉ circle ∧ distance D (center circle) < radius circle) ∧ 
  (E ∉ circle ∧ distance E (center circle) > radius circle)) ∨
  (A ∈ circle ∧ B ∈ circle ∧ D ∈ circle ∧ 
  (C ∉ circle ∧ distance C (center circle) < radius circle) ∧ 
  (E ∉ circle ∧ distance E (center circle) > radius circle)) ∨
  (A ∈ circle ∧ B ∈ circle ∧ E ∈ circle ∧ 
  (C ∉ circle ∧ distance C (center circle) < radius circle) ∧ 
  (D ∉ circle ∧ distance D (center circle) > radius circle))

/-- Given five points no three of which are collinear and no four of which are concyclic,
we want to prove the number of possible "good circles" n is exactly 4. -/
theorem good_circles_count {P : Type} [metric_space P] (A B C D E : P) :
  n = 4 := sorry

end good_circles_count_l720_720753


namespace length_of_train_is_750m_l720_720016

-- Defining the conditions
def train_and_platform_equal_length : Prop := ∀ (L : ℝ), (Length_of_train = L ∧ Length_of_platform = L)
def train_speed := 90 * (1000 / 3600)  -- Convert speed from km/hr to m/s
def crossing_time := 60  -- Time given in seconds

-- Definition for the length of the train
def Length_of_train := sorry -- Given that it should be derived

-- The proof problem statement
theorem length_of_train_is_750m : (train_and_platform_equal_length ∧ train_speed ∧ crossing_time → Length_of_train = 750) :=
by
  -- Proof is skipped
  sorry

end length_of_train_is_750m_l720_720016


namespace trajectory_moving_point_l720_720008

theorem trajectory_moving_point (x y : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (y / (x + 1)) * (y / (x - 1)) = -1 ↔ x^2 + y^2 = 1 := by
  sorry

end trajectory_moving_point_l720_720008


namespace min_value_g_l720_720800

noncomputable def f (x a b : ℝ) : ℝ := Real.exp x - a * x^2 - b * x - 1
noncomputable def g (x a b : ℝ) : ℝ := Real.exp x - 2 * a * x - b

theorem min_value_g (a b : ℝ) :
  (a ≤ 1/2 → ∀ x ∈ set.Icc (0 : ℝ) 1, g 0 a b ≤ g x a b) ∧
  ((1/2 < a ∧ a < Real.exp 1 / 2) → (∃ x, x = Real.log (2 * a) ∧ x ∈ set.Icc (0 : ℝ) 1 ∧ ∀ y ∈ set.Icc (0 : ℝ) 1, g x a b ≤ g y a b)) ∧
  (a ≥ Real.exp 1 / 2 → ∀ x ∈ set.Icc (0 : ℝ) 1, g 1 a b ≤ g x a b) :=
begin
  sorry
end

end min_value_g_l720_720800


namespace four_statements_make_implication_true_l720_720659

theorem four_statements_make_implication_true 
  (p q r : Prop)
  (h1 : p ∧ ¬q ∧ r)
  (h2 : ¬p ∧ ¬q ∧ r)
  (h3 : p ∧ ¬q ∧ ¬r)
  (h4 : ¬p ∧ q ∧ r) :
  (if (p → q) → r then 1 else 0) + 
  (if (¬p → q) → r then 1 else 0) + 
  (if (p → ¬q) → ¬r then 1 else 0) + 
  (if (¬p → q) → r then 1 else 0) = 4 := 
  sorry

end four_statements_make_implication_true_l720_720659


namespace inequality_a_inequality_c_inequality_d_l720_720494

variable {a b : ℝ}

axiom (h : a + b > 0)

theorem inequality_a : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_c : a^21 + b^21 > 0 :=
sorry

theorem inequality_d : (a + 2) * (b + 2) > a * b :=
sorry

end inequality_a_inequality_c_inequality_d_l720_720494


namespace find_m_l720_720292

noncomputable def f : ℝ → ℝ := sorry  -- Given that an explicit form of f(x) isn't provided.

-- Conditions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x ≤ y → f(x) ≤ f(y)

-- Theorem
theorem find_m :
  (∀ (θ : ℝ), f (cos (2 * θ) - 3) + f (4 * m - 2 * m * cos θ) > 0) ↔ m > 3 :=
begin
  sorry  -- The proof is not required as per the instructions.
end

-- Assert the conditions for the function f
axiom f_odd : is_odd f
axiom f_increasing : is_increasing_on_nonneg f

end find_m_l720_720292


namespace inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720487

variable {a b : ℝ}

theorem inequality_a (hab : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_b_not_true (hab : a + b > 0) : ¬(a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

theorem inequality_c (hab : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem inequality_d (hab : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

theorem inequality_e_not_true (hab : a + b > 0) : ¬((a − 3) * (b − 3) < a * b) :=
sorry

theorem inequality_f_not_true (hab : a + b > 0) : ¬((a + 2) * (b + 3) > a * b + 5) :=
sorry

end inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720487


namespace central_angle_of_sector_l720_720310

theorem central_angle_of_sector (arc_length : ℝ) (radius : ℝ) (θ : ℝ) (h₁ : arc_length = 4) (h₂ : radius = 2) : θ = 2 :=
by
  have h : 4 = 2 * θ := by rw [h₁, h₂]
  linarith

#check central_angle_of_sector

end central_angle_of_sector_l720_720310


namespace alcohol_mixture_l720_720134

def alcohol_volume_solution_x (x_volume : ℝ) : ℝ := 0.10 * x_volume
def alcohol_volume_solution_y (y_volume : ℝ) : ℝ := 0.30 * y_volume
def total_volume (x_volume y_volume : ℝ) : ℝ := x_volume + y_volume
def total_alcohol_volume (x_volume y_volume : ℝ) : ℝ := 
  alcohol_volume_solution_x x_volume + alcohol_volume_solution_y y_volume

theorem alcohol_mixture (x_volume : ℝ) (y_volume : ℝ) :
  x_volume = 250 →
  (∀ v, total_alcohol_volume 250 v / total_volume 250 v = 0.25 → v = 750) :=
by
  intros h
  use 750
  sorry

end alcohol_mixture_l720_720134


namespace inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720483

variable {a b : ℝ}

theorem inequality_a (hab : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_b_not_true (hab : a + b > 0) : ¬(a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

theorem inequality_c (hab : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem inequality_d (hab : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

theorem inequality_e_not_true (hab : a + b > 0) : ¬((a − 3) * (b − 3) < a * b) :=
sorry

theorem inequality_f_not_true (hab : a + b > 0) : ¬((a + 2) * (b + 3) > a * b + 5) :=
sorry

end inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720483


namespace minimum_cost_verifying_all_diamonds_l720_720974

/-- 
Given:
1. The weighing method involves placing one diamond on one scale pan and weights on the other to verify the mass.
2. For each weight taken, the buyer has to pay the seller 100 coins.
3. If a weight is removed from the scale and does not participate in the next weighing, it is taken back by the seller.
4. Diamonds' masses are between 1 and 15 units.

Prove that the minimum amount of money required to verify the masses of all the diamonds is 800 coins.
-/
theorem minimum_cost_verifying_all_diamonds : 
  (∃ (weights : list ℕ), 
      (∀ d, d ∈ [1, 2, 4, 8] → ∃ n ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], weights.sum = d) 
      ∧ 
      (list.length weights) * 100 = 800) := 
sorry

end minimum_cost_verifying_all_diamonds_l720_720974


namespace decimal_to_fraction_l720_720091

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l720_720091


namespace square_area_in_ellipse_l720_720622

theorem square_area_in_ellipse : ∀ (s : ℝ), 
  (s > 0) → 
  (∀ x y, (x = s ∨ x = -s) ∧ (y = s ∨ y = -s) → (x^2) / 4 + (y^2) / 8 = 1) → 
  (2 * s)^2 = 32 / 3 := by
  sorry

end square_area_in_ellipse_l720_720622


namespace find_natural_number_l720_720268

open Nat

theorem find_natural_number (p q r n : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r)
  (h1 : r - q = 2 * p) (h2 : r * q + p^2 = 676) : 
  n = p * q * r → n = 2001 :=
by
  sorry

end find_natural_number_l720_720268


namespace photo_album_requirement_l720_720167

-- Definition of the conditions
def pages_per_album : ℕ := 32
def photos_per_page : ℕ := 5
def total_photos : ℕ := 900

-- Calculation of photos per album
def photos_per_album := pages_per_album * photos_per_page

-- Calculation of required albums
noncomputable def albums_needed := (total_photos + photos_per_album - 1) / photos_per_album

-- Theorem to prove the required number of albums is 6
theorem photo_album_requirement : albums_needed = 6 :=
  by sorry

end photo_album_requirement_l720_720167


namespace problem_a_problem_b_problem_c_problem_d_l720_720436

theorem problem_a : 37.3 / (1 / 2) = 74.6 := by
  sorry

theorem problem_b : 0.45 - (1 / 20) = 0.4 := by
  sorry

theorem problem_c : (33 / 40) * (10 / 11) = 0.75 := by
  sorry

theorem problem_d : 0.375 + (1 / 40) = 0.4 := by
  sorry

end problem_a_problem_b_problem_c_problem_d_l720_720436


namespace least_n_froods_score_l720_720860

theorem least_n_froods_score (n : ℕ) : (n * (n + 1) / 2 > 12 * n) ↔ (n > 23) := 
by 
  sorry

end least_n_froods_score_l720_720860


namespace decimal_to_fraction_l720_720117

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l720_720117


namespace equation_one_equation_two_l720_720440

-- Equation (1): Show that for the equation ⟦ ∀ x, (x / (2 * x - 1) + 2 / (1 - 2 * x) = 3 ↔ x = 1 / 5) ⟧
theorem equation_one (x : ℝ) : (x / (2 * x - 1) + 2 / (1 - 2 * x) = 3) ↔ (x = 1 / 5) :=
sorry

-- Equation (2): Show that for the equation ⟦ ∀ x, ((4 / (x^2 - 4) - 1 / (x - 2) = 0) ↔ false) ⟧
theorem equation_two (x : ℝ) : (4 / (x^2 - 4) - 1 / (x - 2) = 0) ↔ false :=
sorry

end equation_one_equation_two_l720_720440


namespace cos_angle_plus_pi_over_two_l720_720338

theorem cos_angle_plus_pi_over_two (α : ℝ) (h1 : Real.cos α = 1 / 5) (h2 : α ∈ Set.Icc (-2 * Real.pi) (-3 * Real.pi / 2) ∪ Set.Icc (0) (Real.pi / 2)) :
  Real.cos (α + Real.pi / 2) = 2 * Real.sqrt 6 / 5 :=
sorry

end cos_angle_plus_pi_over_two_l720_720338


namespace decimal_to_fraction_l720_720080

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l720_720080


namespace sqrt_y_to_the_fourth_eq_256_l720_720579

theorem sqrt_y_to_the_fourth_eq_256 (y : ℝ) (h : (sqrt y)^4 = 256) : y = 16 := by
  sorry

end sqrt_y_to_the_fourth_eq_256_l720_720579


namespace staircase_problem_l720_720034

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem staircase_problem (total_steps required_steps : ℕ) (num_two_steps : ℕ) :
  total_steps = 11 ∧ required_steps = 7 ∧ num_two_steps = 4 →
  C 7 4 = 35 :=
by
  intro h
  sorry

end staircase_problem_l720_720034


namespace sin_cos_cubic_diff_l720_720144

theorem sin_cos_cubic_diff (α n : ℝ) (h : sin α - cos α = n) : sin α ^ 3 - cos α ^ 3 = (3 * n - n ^ 3) / 2 :=
by
  sorry

end sin_cos_cubic_diff_l720_720144


namespace color_triplet_exists_l720_720670

theorem color_triplet_exists (color : ℕ → Prop) :
  (∀ n, color n ∨ ¬ color n) → ∃ x y z : ℕ, (x ≠ y ∧ y ≠ z ∧ x ≠ z) ∧ color x = color y ∧ color y = color z ∧ x * y = z ^ 2 :=
by
  sorry

end color_triplet_exists_l720_720670


namespace inversion_number_reverse_l720_720278

def inversion_number (l : List ℕ) : ℕ :=
  l.enumerate.foldr (λ ⟨i, a⟩ acc, acc +
    l.drop (i + 1) |>.filter (λ b, a > b) |>.length) 0

theorem inversion_number_reverse (a1 a2 a3 a4 : ℕ) 
  (h_distinct : List.Nodup [a1, a2, a3, a4])
  (h_pos : 0 < a1 ∧ 0 < a2 ∧ 0 < a3 ∧ 0 < a4)
  (h_inv_num : inversion_number [a1, a2, a3, a4] = 2) :
  inversion_number [a4, a3, a2, a1] = 4 := 
sorry

end inversion_number_reverse_l720_720278


namespace problem_a_problem_c_problem_d_l720_720500

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end problem_a_problem_c_problem_d_l720_720500


namespace sqrt2_irrational_l720_720583

theorem sqrt2_irrational : irrational (sqrt 2) :=
sorry

end sqrt2_irrational_l720_720583


namespace part1_part2_l720_720769

open Real

def g (x m : ℝ) : ℝ := x^2 - m * x + (m^2) / 2 + 2 * m - 3
def f (x m : ℝ) : ℝ := x + m

-- Statement for Part (1)
theorem part1 (a m : ℝ) (h : ∀ x, g x m < (m^2) / 2 + 1 ↔ 1 < x ∧ x < a) : a = 2 :=
sorry

-- Statement for Part (2)
theorem part2 : ¬ ∃ m : ℝ, ∀ x1 ∈ Icc (0 : ℝ) 1, ∀ x2 ∈ Icc (1 : ℝ) 2, f x1 m > g x2 m :=
sorry

end part1_part2_l720_720769


namespace sin_15_cos_15_l720_720273

theorem sin_15_cos_15 (h1: ∀ θ : ℝ, sin (2 * θ) = 2 * sin θ * cos θ)
  (h2: sin (30 * (π / 180)) = 1 / 2) :
  sin (15 * (π / 180)) * cos (15 * (π / 180)) = 1 / 4 :=
sorry

end sin_15_cos_15_l720_720273


namespace relationship_a_b_c_l720_720751

noncomputable def a : ℝ := Real.sin (Real.pi / 16)
noncomputable def b : ℝ := 0.25
noncomputable def c : ℝ := 2 * Real.log 2 - Real.log 3

theorem relationship_a_b_c : a < b ∧ b < c :=
by
  sorry

end relationship_a_b_c_l720_720751


namespace bridge_length_calculation_l720_720135

def length_of_bridge (train_length : ℝ) (speed_kmph : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_mps := speed_kmph * (1000 / 3600) in
  let distance_traveled := speed_mps * time_sec in
  distance_traveled - train_length

theorem bridge_length_calculation :
  length_of_bridge 295 75 45 = 642.5 :=
by
  unfold length_of_bridge
  -- Insert computation steps here.
  sorry

end bridge_length_calculation_l720_720135


namespace segment_length_BD_eq_CB_l720_720411

theorem segment_length_BD_eq_CB {AC CB BD x : ℝ}
  (h1 : AC = 4 * CB)
  (h2 : BD = CB)
  (h3 : CB = x) :
  BD = CB := 
by
  -- Proof omitted
  sorry

end segment_length_BD_eq_CB_l720_720411


namespace MN_parallel_PQ_l720_720930

theorem MN_parallel_PQ
  (A B C I Z M N P Q : Point)
  (circumcircle : Circle)
  (h1 : I = circumcenter A B C)
  (h2 : midpoint I (arc A C circumcircle))
  (h3 : angle_bisector I Z (angle B A C))
  (h4 : perpendicular PQ IZ)
  (h5 : isosceles MBI with base BI)
  (h6 : isosceles NBI with base BI)
  : parallel PQ MN :=
sorry -- Proof to be filled in later by the user

end MN_parallel_PQ_l720_720930


namespace orange_balloons_count_l720_720437

variable (original_orange_balloons : ℝ)
variable (found_orange_balloons : ℝ)
variable (total_orange_balloons : ℝ)

theorem orange_balloons_count :
  original_orange_balloons = 9.0 →
  found_orange_balloons = 2.0 →
  total_orange_balloons = original_orange_balloons + found_orange_balloons →
  total_orange_balloons = 11.0 := by
  sorry

end orange_balloons_count_l720_720437


namespace regular_polygon_exterior_angle_l720_720203

theorem regular_polygon_exterior_angle (n : ℕ) (h : 60 * n = 360) : n = 6 :=
sorry

end regular_polygon_exterior_angle_l720_720203


namespace unique_determination_of_x_given_sums_l720_720884

theorem unique_determination_of_x_given_sums {n : ℕ} (h_n : n ≥ 3) 
  (x : Fin n → ℝ) (y : Fin (Nat.choose n 2) → ℝ)
  (h_y : ∃ (f : {i : Fin n // i < n} → Fin (Nat.choose n 2)), ∀ ⟨i, hi⟩, y (f ⟨i, hi⟩) = x ⟨i, Nat.lt_of_lt_of_le hi (Nat.choose_two_le n (zero_lt_two_left.ne h_n))⟩)
  (hx : ∀ (a b : Fin n) (ha : a < b), y (h : Fin (Nat.choose n 2), h.1.val < n) = x a + x b -> ∃i, y i = x a + x b) :
  ∃ (x' : Fin n → ℝ), (∀ ⟨i, hi⟩, x' ⟨i, hi⟩ = x ⟨i, hi⟩) :=
sorry  -- Proof omitted

end unique_determination_of_x_given_sums_l720_720884


namespace cos_double_beta_eq_24_over_25_l720_720289

theorem cos_double_beta_eq_24_over_25
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 3 / 5)
  (h2 : Real.cos (α + β) = -3 / 5)
  (h3 : α - β ∈ Set.Ioo (π / 2) π)
  (h4 : α + β ∈ Set.Ioo (π / 2) π) :
  Real.cos (2 * β) = 24 / 25 := sorry

end cos_double_beta_eq_24_over_25_l720_720289


namespace distance_T_S_l720_720430

theorem distance_T_S : 
  let P := -14
  let Q := 46
  let S := P + (3 / 4:ℚ) * (Q - P)
  let T := P + (1 / 3:ℚ) * (Q - P)
  S - T = 25 :=
by
  let P := -14
  let Q := 46
  let S := P + (3 / 4:ℚ) * (Q - P)
  let T := P + (1 / 3:ℚ) * (Q - P)
  show S - T = 25
  sorry

end distance_T_S_l720_720430


namespace drawable_grid_l720_720148

theorem drawable_grid (n : ℕ) (h : n > 0) : 
  (∃ (f : (ℕ × ℕ) → (ℕ × ℕ)), 
      (∀ i j, 1 ≤ i ∧ i < n → 1 ≤ j ∧ j < n → 
        ((i + j) % 2 = 0 → 
         (f (i, j) = (i+1, j+1) ∨ 
          f (i, j) = (i-1, j-1))) ∧ 
        ((i + j) % 2 = 1 → 
         (f (i, j) = (i+1, j-1) ∨ 
          f (i, j) = (i-1, j+1))) ∧
        ((i + j) % 2 = 0 → (f (i+1, j) ≠ f (i, j-1))) ∧ 
        ((i + j) % 2 = 0 → (f (i-1, j) ≠ f (i, j+1)))))
↔ (n = 1 ∨ n = 2 ∨ n = 3) := 
begin
  -- Proof goes here
  sorry
end

end drawable_grid_l720_720148


namespace columns_contain_all_numbers_l720_720031

def rearrange (n m k : ℕ) (a : ℕ → ℕ) : ℕ → ℕ :=
  λ i => if i < n - m then a (i + m + 1)
         else if i < n - k - m then a (i - (n - m) + k + 1)
         else a (i - (n - k))

theorem columns_contain_all_numbers
  (n m k: ℕ)
  (h1 : n > 0)
  (h2 : m < n)
  (h3 : k < n)
  (a : ℕ → ℕ)
  (h4 : ∀ i : ℕ, i < n → a i = i + 1) :
  ∀ j : ℕ, j < n → ∃ i : ℕ, i < n ∧ rearrange n m k a i = j + 1 :=
by
  sorry

end columns_contain_all_numbers_l720_720031


namespace range_of_a_l720_720353

noncomputable def f (a x : ℝ) : ℝ :=
  x^3 + 3 * a * x^2 + 3 * ((a + 2) * x + 1)

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ deriv (f a) x = 0 ∧ deriv (f a) y = 0) ↔ a < -1 ∨ a > 2 :=
by
  sorry

end range_of_a_l720_720353


namespace central_angle_of_sector_l720_720351

theorem central_angle_of_sector (r α : ℝ) (h_arc_length : α * r = 5) (h_area : 0.5 * α * r^2 = 5): α = 5 / 2 := by
  sorry

end central_angle_of_sector_l720_720351


namespace smallest_five_digit_divisible_by_primes_l720_720712

theorem smallest_five_digit_divisible_by_primes : 
  let primes := [2, 3, 5, 7, 11] in
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11))) in
  let five_digit_threshold := 10000 in
  ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
by
  let primes := [2, 3, 5, 7, 11]
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))
  have lcm_2310 : lcm_primes = 2310 := sorry
  let five_digit_threshold := 10000
  have exists_n : ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
    sorry
  exists_intro 5
  have 5_condition : 5 > 0 := sorry
  have 2310_5_condition : 2310 * 5 >= five_digit_threshold := sorry
  have answer : 2310 * 5 = 11550 := sorry
  exact  ⟨5, 5_condition, 2310_5_condition, answer⟩
  exact ⟨5, 5 > 0, 2310 * 5 ≥ 10000, 2310 * 5 = 11550⟩
  sorry

end smallest_five_digit_divisible_by_primes_l720_720712


namespace exists_positive_integer_n_for_inequality_l720_720959

theorem exists_positive_integer_n_for_inequality :
  ∃ n : ℕ, n > 0 ∧ ∀ x : ℝ, x ≥ 0 → (x - 1) * (x ^ 2005 - 2005 * x ^ (n + 1) + 2005 * x ^ n - 1) ≥ 0 :=
begin
  use 2004,
  split,
  { norm_num, },
  { intros x hx,
    sorry, -- Proof here
  }
end

end exists_positive_integer_n_for_inequality_l720_720959


namespace selection_no_arithmetic_mean_l720_720435

theorem selection_no_arithmetic_mean (k : ℕ) : 
  ∃ S : set ℕ, S ⊆ finset.range (3^k) ∧ S.card = 2^k ∧ 
  ∀ p q r ∈ S, p ≠ (q + r) / 2 ∨ p = q ∧ q = r := by
  sorry

end selection_no_arithmetic_mean_l720_720435


namespace part_a_part_c_part_d_l720_720511

-- Define the variables
variables {a b : ℝ}

-- Define the conditions and statements
def cond := a + b > 0

theorem part_a (h : cond) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem part_c (h : cond) : a^21 + b^21 > 0 :=
sorry

theorem part_d (h : cond) : (a + 2) * (b + 2) > a * b :=
sorry

end part_a_part_c_part_d_l720_720511


namespace angle_CAB_is_60_l720_720368

open EuclideanGeometry

variables {A B C D H : Point}
variables [Triangle ABC]
variables (circumcenter_on_bisector : OnCircle (Circumcircle ABC) (Bisector (Angle DHB)) = true)
variables (is_acute_triangle : AcuteTriangle ABC)
variables (CD_is_altitude : Altitude CD)
variables (H_is_orthocenter : Orthocenter H)

theorem angle_CAB_is_60 (h1 : is_acute_triangle) (h2 : CD_is_altitude) (h3 : H_is_orthocenter) (h4 : circumcenter_on_bisector) :
  ∠CAB = 60 :=
sorry

end angle_CAB_is_60_l720_720368


namespace vector_properties_l720_720637

-- Define vectors and relevant properties
variables {V : Type*} [AddCommGroup V] [Module ℝ V] (a b : V)

-- Define magnitudes of vectors
def magnitude (v : V) : ℝ := ∥v∥

-- Define parallel vectors
def parallel (a b : V) : Prop := ∃ k : ℝ, a = k • b

-- Define same direction vectors
def same_direction (a b : V) : Prop := ∃ k : ℝ, k > 0 ∧ a = k • b

-- Define opposite direction vectors
def opposite_direction (a b : V) : Prop := ∃ k : ℝ, k < 0 ∧ a = k • b

-- Theorem statement
theorem vector_properties :
  (parallel a b ∧ magnitude a = magnitude b → a = b ∨ a = -b) ∧
  (¬ (parallel a b ∧ magnitude a = magnitude b → a ≠ b ∧ a ≠ -b)) ∧
  (same_direction a b ∧ magnitude a = magnitude b ↔ a = b) ∧
  (opposite_direction a b ∨ magnitude a ≠ magnitude b → a ≠ b) :=
by sorry

end vector_properties_l720_720637


namespace small_gifts_combinations_large_gifts_combinations_l720_720626

/-
  Definitions based on the given conditions:
  - 12 varieties of wrapping paper.
  - 3 colors of ribbon.
  - 6 types of gift cards.
  - Small gifts can use only 2 out of the 3 ribbon colors.
-/

def wrapping_paper_varieties : ℕ := 12
def ribbon_colors : ℕ := 3
def gift_card_types : ℕ := 6
def small_gift_ribbon_colors : ℕ := 2

/-
  Proof problems:

  - For small gifts, there are 12 * 2 * 6 combinations.
  - For large gifts, there are 12 * 3 * 6 combinations.
-/

theorem small_gifts_combinations :
  wrapping_paper_varieties * small_gift_ribbon_colors * gift_card_types = 144 :=
by
  sorry

theorem large_gifts_combinations :
  wrapping_paper_varieties * ribbon_colors * gift_card_types = 216 :=
by
  sorry

end small_gifts_combinations_large_gifts_combinations_l720_720626


namespace stickers_decorate_l720_720907

theorem stickers_decorate (initial_stickers bought_stickers birthday_stickers given_stickers remaining_stickers stickers_used : ℕ)
    (h1 : initial_stickers = 20)
    (h2 : bought_stickers = 12)
    (h3 : birthday_stickers = 20)
    (h4 : given_stickers = 5)
    (h5 : remaining_stickers = 39) :
    (initial_stickers + bought_stickers + birthday_stickers - given_stickers - remaining_stickers = stickers_used) →
    stickers_used = 8 
:= by {sorry}

end stickers_decorate_l720_720907


namespace decimal_to_fraction_l720_720073

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l720_720073


namespace geom_seq_S6_l720_720861

theorem geom_seq_S6 :
  ∃ (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ),
  (q = 2) →
  (S 3 = 7) →
  (∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) →
  S 6 = 63 :=
sorry

end geom_seq_S6_l720_720861


namespace num_valid_numbers_l720_720830

def is_valid_number (n : ℕ) : Prop :=
  n < 100000 ∧ (let digits : List ℕ := n.digits in 
    digits.length ≤ 5 ∧ (digits.nodup.erase_dup.length ≤ 2 ∧ digits.contains 1))

theorem num_valid_numbers : {n : ℕ | is_valid_number n}.card = 279 := by
  sorry

end num_valid_numbers_l720_720830


namespace solution_set_f_x_leq_m_solution_set_inequality_a_2_l720_720797

-- Part (I)
theorem solution_set_f_x_leq_m (a m : ℝ) (h : ∀ x : ℝ, |x - a| ≤ m ↔ -1 ≤ x ∧ x ≤ 5) :
  a = 2 ∧ m = 3 :=
sorry

-- Part (II)
theorem solution_set_inequality_a_2 (t : ℝ) (h_t : t ≥ 0) :
  (∀ x : ℝ, |x - 2| + t ≥ |x + 2 * t - 2| ↔ t = 0 ∧ (∀ x : ℝ, True) ∨ t > 0 ∧ ∀ x : ℝ, x ≤ 2 - t / 2) :=
sorry

end solution_set_f_x_leq_m_solution_set_inequality_a_2_l720_720797


namespace area_of_triangle_is_23_over_10_l720_720556

noncomputable def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  1/2 * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

theorem area_of_triangle_is_23_over_10 :
  let A : ℚ × ℚ := (3, 3)
  let B : ℚ × ℚ := (5, 3)
  let C : ℚ × ℚ := (21 / 5, 19 / 5)
  area_of_triangle A.1 A.2 B.1 B.2 C.1 C.2 = 23 / 10 :=
by
  sorry

end area_of_triangle_is_23_over_10_l720_720556


namespace find_n_l720_720304

theorem find_n 
  (x : ℝ) 
  (h1 : log 10 (sin x) + log 10 (cos x) = -0.5) 
  (h2 : log 10 (sin x + cos x) = 0.5 * (log 10 n - 0.5)) 
  : n = (1 + sqrt 10 / 5) * sqrt 10 := 
by 
  sorry

end find_n_l720_720304


namespace expression_value_l720_720578

theorem expression_value (x : ℝ) (h : x = 3) : 3 * x^2 - 4 * x + 2 = 17 := by
  rw [h]
  norm_num
  sorry

end expression_value_l720_720578


namespace printing_time_345_l720_720174

def printing_time (total_pages : ℕ) (rate : ℕ) : ℕ :=
  total_pages / rate

theorem printing_time_345 :
  printing_time 345 23 = 15 :=
by
  sorry

end printing_time_345_l720_720174


namespace complex_number_solution_l720_720684

theorem complex_number_solution (z : ℂ) : (|z - 2| = |z + 4 * complex.I| ∧ |z - 2| = |z + 2 * complex.I|) → z = 3 - 3 * complex.I :=
by
  sorry

end complex_number_solution_l720_720684


namespace projection_value_l720_720308

variables (a b : E) [inner_product_space ℝ E] [finite_dimensional ℝ E]
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 2) (θ : real.angle)
variables (hθ : θ = real.pi / 3) -- 60 degrees in radians

theorem projection_value (a b : E) [inner_product_space ℝ E] [finite_dimensional ℝ E]
  (ha : ∥a∥ = 1) (hb : ∥b∥ = 2) (hθ : real.angle.cos (real.pi / 3) = 1/2) :
  (2 • a + b) - (1 / ∥b∥ ^ 2) • ⟪2 • a + b, b⟫ • b = 3 :=
begin
  sorry
end

end projection_value_l720_720308


namespace number_of_real_solutions_l720_720271

-- The real question is simplified to finding the number of real solutions
def equation (x : ℝ) : Prop := (6 * x / (x^2 + 2 * x + 4) + 7 * x / (x^2 - 7 * x + 4) = -2)

-- Stating that there are exactly 2 real solutions to the given equation
theorem number_of_real_solutions : set.count_eq { x : ℝ | equation x } 2 :=
sorry

end number_of_real_solutions_l720_720271


namespace work_duration_l720_720133

theorem work_duration (W : ℝ) (p q : ℝ) (h1 : p = W / 40) (h2 : q = W / 24) : p alone started the work.
  q joined p after 16 days.
  The total time the work lasted is 25 days :=
by
  sorry

end work_duration_l720_720133


namespace coordinates_of_B_eq_l720_720300

-- Define the coordinates of point A
def A : (ℝ × ℝ × ℝ) := (3, -1, 0)

-- Define the vector AB
def vecAB : (ℝ × ℝ × ℝ) := (2, 5, -3)

-- Define point B and conditions to prove the coordinates of B
theorem coordinates_of_B_eq :
  ∃ B : (ℝ × ℝ × ℝ), B = (5, 4, -3) ∧
  (B.1 - A.1 = vecAB.1) ∧
  (B.2 - A.2 = vecAB.2) ∧
  (B.3 - A.3 = vecAB.3) :=
begin
  sorry
end

end coordinates_of_B_eq_l720_720300


namespace MN_parallel_PQ_l720_720945

-- Definitions and conditions
variable {A B C I Z M N P Q : Type}
variable [is_center_of_circumcircle I A B C]
variable [midpoint_of_arc I A C]
variable [angle_bisector_of IZ (∠ B)]
variable [perpendicular PQ IZ]
variable [isosceles_triangle M BI]
variable [isosceles_triangle N BI]

-- The theorem statement
theorem MN_parallel_PQ (h_center: is_center_of_circumcircle I A B C) 
  (h_midpoint: midpoint_of_arc I A C) (h_bisector: angle_bisector_of IZ (∠ B)) 
  (h_perpendicular: perpendicular PQ IZ) (h_mbi: isosceles_triangle M BI) 
  (h_nbi: isosceles_triangle N BI) : 
  parallel MN PQ :=
sorry

end MN_parallel_PQ_l720_720945


namespace find_varphi_l720_720982

theorem find_varphi 
  (f g : ℝ → ℝ) 
  (x1 x2 varphi : ℝ) 
  (h_f : ∀ x, f x = 2 * Real.cos (2 * x)) 
  (h_g : ∀ x, g x = 2 * Real.cos (2 * x - 2 * varphi)) 
  (h_varphi_range : 0 < varphi ∧ varphi < π / 2) 
  (h_diff_cos : |f x1 - g x2| = 4) 
  (h_min_dist : |x1 - x2| = π / 6) 
: varphi = π / 3 := 
sorry

end find_varphi_l720_720982


namespace problem_a_problem_b_problem_c_l720_720523

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end problem_a_problem_b_problem_c_l720_720523


namespace probability_two_balls_different_colors_probability_two_balls_different_colors_replacement_l720_720542

variable (whiteBalls blackBalls : ℕ)
variable (totalBalls : ℕ := whiteBalls + blackBalls)

/-- Probability of drawing two balls of different colors without replacement --/
theorem probability_two_balls_different_colors (h₁ : whiteBalls = 2) (h₂ : blackBalls = 3) :
  (Nat.choose totalBalls 2) ≠ 0 ∧
  (Nat.choose whiteBalls 1) * (Nat.choose blackBalls 1) / (Nat.choose totalBalls 2) = 3 / 5 :=
by
  sorry

/-- Probability of drawing two balls of different colors with replacement --/
theorem probability_two_balls_different_colors_replacement (h₁ : whiteBalls = 2) (h₂ : blackBalls = 3) :
  (whiteBalls / totalBalls * blackBalls / totalBalls + blackBalls / totalBalls * whiteBalls / totalBalls) = 0.48 :=
by
  sorry

end probability_two_balls_different_colors_probability_two_balls_different_colors_replacement_l720_720542


namespace slower_train_speed_l720_720560

-- Define the given conditions
def speed_faster_train : ℝ := 50  -- km/h
def length_faster_train : ℝ := 75.006  -- meters
def passing_time : ℝ := 15  -- seconds

-- Conversion factor
def mps_to_kmph : ℝ := 3.6

-- Define the problem to be proved
theorem slower_train_speed : 
  ∃ speed_slower_train : ℝ, 
    speed_slower_train = speed_faster_train - (75.006 / 15) * mps_to_kmph := 
  by
    exists 31.99856
    sorry

end slower_train_speed_l720_720560


namespace time_to_complete_round_of_larger_field_l720_720589

-- Variables definition
variables (W : ℝ) -- Width of the smaller field
def L : ℝ := 1.5 * W -- Length of the smaller field
def P_small : ℝ := 2 * (L + W) -- Perimeter of the smaller field
def P_large : ℝ := 2 * (3 * L + 4 * W) -- Perimeter of the larger field

-- Ratios and times
def time_for_larger_field (time_small : ℝ) : ℝ := time_small * (P_large / P_small)

-- Proof statement
theorem time_to_complete_round_of_larger_field (W : ℝ) (time_small : ℝ) (h: time_small = 20) :
  time_for_larger_field W time_small = 68 :=
by
  rw [time_for_larger_field, P_small, P_large, L]
  -- Simplify and compute accordingly
  sorry

end time_to_complete_round_of_larger_field_l720_720589


namespace smallest_five_digit_number_divisible_l720_720692

def smallest_prime_divisible (n: ℕ) : Prop :=
  ∃ k: ℕ, n = 2310 * k ∧ 10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_number_divisible :
  ∃ (n: ℕ), smallest_prime_divisible n ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_l720_720692


namespace MN_parallel_PQ_l720_720935

-- Define the given geometric entities and properties
variables (A B C I M N P Q Z : Type) [Inhabited A]

-- Definition of points being collinear
def collinear (a b c : Type) := ∃ (r : ℝ), r • (b - a) + a = c

-- Definition of parallel lines
def parallel (l1 l2 : Type) : Prop := 
  ∃ p1 p2 p3 p4 : Type, collinear p1 p2 p3 → collinear p2 p3 p4

-- Definition points on a circle
def on_circumcircle (A B C : Type) : Prop := sorry

-- Definition of angle bisector
def angle_bisector (I A B : Type) : Prop := sorry

-- Definition of perpendicular lines
def perpendicular (l1 l2 : Type) : Prop := sorry

-- Main theorem statement
theorem MN_parallel_PQ 
  (h1 : on_circumcircle A B C) 
  (h2 : angle_bisector I A B) 
  (h3 : perpendicular PQ IZ) 
  (h4 : perpendicular PQ BI) 
  (h5 : perpendicular MN BI) : parallel MN PQ :=
sorry

end MN_parallel_PQ_l720_720935


namespace M_eq_four_l720_720231

-- Definition of secant function
def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Definition of M based on the given problem
def M : ℝ :=
  sec (2 * Real.pi / 9) + sec (4 * Real.pi / 9) + sec (6 * Real.pi / 9) + sec (8 * Real.pi / 9)

-- The theorem to prove that M equals 4
theorem M_eq_four : M = 4 := by
  sorry

end M_eq_four_l720_720231


namespace correct_option_l720_720128

theorem correct_option (a b c d : ℝ) :
  (5 * real.sqrt 7 - 2 * real.sqrt 7 ≠ 2) ∧
  (2 + real.sqrt 2 ≠ 2 * real.sqrt 2) ∧
  (real.sqrt 3 * real.sqrt 6 = 3 * real.sqrt 2) ∧
  (real.sqrt 15 / real.sqrt 5 ≠ 3) :=
by {
  -- Conditions need to be used in the proof, which is not included here.
  sorry
}

end correct_option_l720_720128


namespace decimal_to_fraction_l720_720058

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l720_720058


namespace segment_division_ratio_l720_720558

theorem segment_division_ratio (α : ℝ) (R r : ℝ) :
  (∃ A B K : ℝ³, 
    A ≠ B ∧
    let cos_half_alpha := Real.cos (α / 2)
    let ratio := cos_half_alpha * cos_half_alpha : (1 / (cos_half_alpha * cos_half_alpha)) : cos_half_alpha * cos_half_alpha in
    segment_divided_ratio A B K ratio) :=
sorry

end segment_division_ratio_l720_720558


namespace total_sum_of_money_l720_720154

theorem total_sum_of_money (x : ℝ) (A B C D E : ℝ) (hA : A = x) (hB : B = 0.75 * x) 
  (hC : C = 0.60 * x) (hD : D = 0.50 * x) (hE1 : E = 0.40 * x) (hE2 : E = 84) : 
  A + B + C + D + E = 682.50 := 
by sorry

end total_sum_of_money_l720_720154


namespace depth_of_each_cut_l720_720036

theorem depth_of_each_cut (d : ℕ) (w h : ℕ)
  (original_area : 1200 = w * h)
  (total_cut_area : 30 * d)
  (remaining_area : 990)
  (total_area_equation : original_area - total_cut_area = remaining_area) :
  d = 7 :=
by
  -- Proof would go here
  sorry

end depth_of_each_cut_l720_720036


namespace zoo_visitors_l720_720428

theorem zoo_visitors (P : ℕ) (h : 3 * P = 3750) : P = 1250 :=
by 
  sorry

end zoo_visitors_l720_720428


namespace linear_term_coefficient_l720_720239

theorem linear_term_coefficient : (x - 1) * (1 / x + x) ^ 6 = a + b * x + c * x^2 + d * x^3 + e * x^4 + f * x^5 + g * x^6 →
  b = 20 :=
by
  sorry

end linear_term_coefficient_l720_720239


namespace chess_game_grandfather_wins_l720_720333

theorem chess_game_grandfather_wins (x : ℕ)
  (h1 : ∀ n, (n = 12 - x) → 3 * n = 12 * 3 - 3 * x) 
  (h2 : x + 3 * (12 - x) = 36) :
  x = 9 := 
by 
  -- Definitions and conditions
  have h3 : 4 * x = 36, from by 
    have := h2,
    linarith,
  -- Solve for x
  have h4 : x = 9, from by 
    have := h3,
    linarith,
  exact h4

#eval chess_game_grandfather_wins 9

end chess_game_grandfather_wins_l720_720333


namespace three_inv_mod_191_l720_720681

theorem three_inv_mod_191 : ∃ x, 0 ≤ x ∧ x ≤ 190 ∧ 3 * x ≡ 1 [MOD 191] :=
by{
  use 64,
  split,
  -- 0 ≤ x
  exact Nat.zero_le 64,
  split,
  -- x ≤ 190
  exact le_of_lt (by norm_num),
  -- 3 * x ≡ 1 [MOD 191]
  sorry
}

end three_inv_mod_191_l720_720681


namespace math_problem_l720_720466

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end math_problem_l720_720466


namespace sin_cos_term_side_l720_720787

theorem sin_cos_term_side (a : ℝ) (ha : a ≠ 0) :
  ∃ k : ℝ, (k = 2 * (if a > 0 then -3/5 else 3/5) + (if a > 0 then 4/5 else -4/5)) ∧ (k = 2/5 ∨ k = -2/5) := by
  sorry

end sin_cos_term_side_l720_720787


namespace chris_age_is_16_l720_720002

/- Definitions -/
variables (a b c d : ℕ)
variable h1 : (a + b + c + d) = 48
variable h2 : (c - 5) = 2 * (a - 5)
variable h3 : (b + 2) = 3 * (a + 2) / 4
variable h4 : d = 15

/- Theorem statement -/
theorem chris_age_is_16 (h1 : (a + b + c + d) = 48) (h2 : (c - 5) = 2 * (a - 5)) (h3 : (b + 2) = 3 * (a + 2) / 4) (h4 : d = 15) : c = 16 :=
by
  sorry

end chris_age_is_16_l720_720002


namespace range_of_m_l720_720339

theorem range_of_m (m : ℝ) (h : m ≠ 0) :
  (∀ x : ℝ, x ≥ 4 → (m^2 * x - 1) / (m * x + 1) < 0) →
  m < -1 / 2 :=
by
  sorry

end range_of_m_l720_720339


namespace chips_per_cookie_l720_720662

theorem chips_per_cookie (total_cookies : ℕ) (uneaten_chips : ℕ) (uneaten_cookies : ℕ) (h1 : total_cookies = 4 * 12) (h2 : uneaten_cookies = total_cookies / 2) (h3 : uneaten_chips = 168) : 
  uneaten_chips / uneaten_cookies = 7 :=
by sorry

end chips_per_cookie_l720_720662


namespace solve_for_x_l720_720283

theorem solve_for_x : ∀ (x : ℝ), 3^(x^2 - 6 * x + 9) = 3^(x^2 + 2 * x - 1) → x = 5 / 4 :=
by
  intro x h
  sorry

end solve_for_x_l720_720283


namespace smallest_five_digit_number_divisible_by_first_five_primes_l720_720702

theorem smallest_five_digit_number_divisible_by_first_five_primes : 
  ∃ n, (n >= 10000) ∧ (n < 100000) ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_first_five_primes_l720_720702


namespace odd_product_probability_l720_720671

theorem odd_product_probability :
  let chips := [1, 2, 3, 4] in
  let total_outcomes := 4 * 4 in
  let favorable_outcomes := {
    (1, 1), (1, 3), (3, 1), (3, 3)
  }.card in
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1/4 :=
by
  -- skip the proof
  sorry

end odd_product_probability_l720_720671


namespace B_gt_A_l720_720582

noncomputable def harmonic (n : ℕ) : ℝ :=
  ∑ k in finset.range(n+1), 1 / (k + 1 : ℝ)

def A : ℝ := (1 / 2015) * harmonic 2015

def B : ℝ := (1 / 2016) * harmonic 2016

theorem B_gt_A : B > A :=
  sorry

end B_gt_A_l720_720582


namespace fraction_of_second_eq_fifth_of_first_l720_720601

theorem fraction_of_second_eq_fifth_of_first 
  (a b x y : ℕ)
  (h1 : y = 40)
  (h2 : x + 35 = 4 * y)
  (h3 : (1 / 5) * x = (a / b) * y) 
  (hb : b ≠ 0):
  a / b = 5 / 8 := by
  sorry

end fraction_of_second_eq_fifth_of_first_l720_720601


namespace decimal_to_fraction_l720_720055

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l720_720055


namespace solution_set_proof_l720_720780

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

def domain_condition : Prop :=
  ∀ x, 0 < x → f x ≠ 0

def derivative_condition (x : ℝ) : Prop :=
  (x + 1) * (2 * f x + x * f' x) > x * f x

def value_condition : Prop :=
  f 6 = 7 / 12

def inequality_to_prove (x : ℝ) : Prop :=
  f (x + 4) < (3 * x + 15) / (x + 4) ^ 2

def solution_set_condition : Set ℝ :=
  {-4 < x ∧ x < 2}

theorem solution_set_proof :
  (∀ (x : ℝ), 0 < x →
    domain_condition ∧
    derivative_condition x ∧
    value_condition →
    inequality_to_prove x) ↔
    ∀ x, x ∈ solution_set_condition := 
sorry

end solution_set_proof_l720_720780


namespace inequality_a_inequality_c_inequality_d_l720_720492

variable {a b : ℝ}

axiom (h : a + b > 0)

theorem inequality_a : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_c : a^21 + b^21 > 0 :=
sorry

theorem inequality_d : (a + 2) * (b + 2) > a * b :=
sorry

end inequality_a_inequality_c_inequality_d_l720_720492


namespace expected_heads_l720_720415

-- Given conditions
def num_coins : ℕ := 80
def coin_fair : ℕ -> ℕ := λ i, 2 -- Each coin is fair, 2 outcomes (heads or tails)
def toss_probability (k : ℕ) : ℚ := 1 / 2^(k + 1)

-- Expected value statement
theorem expected_heads : (num_coins : ℚ) * ((toss_probability 0) + (toss_probability 1) + (toss_probability 2) + (toss_probability 3)) = 75 := 
by
  sorry

end expected_heads_l720_720415


namespace horner_V1_value_l720_720561

-- Definitions based on conditions
def f (x : ℕ) : ℕ := 3 * x^4 + 2 * x^2 + x + 4

def horner_rule_step1 (V_0 x : ℕ) : ℕ := V_0 * x + 2

-- Given conditions
def V_0 : ℕ := 3
def x_val : ℕ := 10
def V_1 : ℕ := horner_rule_step1 V_0 x_val

-- Goal to prove
theorem horner_V1_value :
  V_1 = 32 :=
by rw [V_1, horner_rule_step1, V_0, x_val]; sorry

end horner_V1_value_l720_720561


namespace parallel_lines_l720_720949

-- Definition of points and lines
variables {A B C M N P Q I Z : Type} [geometry_space : Geometry A B C M N P Q I Z]

-- Conditions
def is_center_of_circumcircle (A B C I : Type) : Prop := Geometry.is_center_of_circumcircle A B C I
def midpoint_of_arc_AC (A C I : Type) : Prop := Geometry.midpoint_ω A C I
def is_angle_bisector (IZ : Type) (B : Type) : Prop := Geometry.angle_bisector IZ B
def is_perpendicular (PQ IZ : Type) : Prop := Geometry.perpendicular PQ IZ
def is_isosceles (MBI NBI BI : Type) : Prop := Geometry.isosceles MBI NBI BI

-- Problem Statement
theorem parallel_lines (A B C M N P Q I Z : Type)
  [is_center_of_circumcircle A B C I]
  [midpoint_of_arc_AC A C I]
  [is_angle_bisector IZ B]
  [is_perpendicular PQ IZ]
  [is_isosceles MBI NBI BI] : PQ ∥ MN :=
by
  sorry

end parallel_lines_l720_720949


namespace geometric_sequence_at_t_l720_720535

theorem geometric_sequence_at_t (a : ℕ → ℕ) (S : ℕ → ℕ) (t : ℕ) :
  (∀ n, a n = a 1 * (3 ^ (n - 1))) →
  a 1 = 1 →
  S t = (a 1 * (1 - 3 ^ t)) / (1 - 3) →
  S t = 364 →
  a t = 243 :=
by {
  sorry
}

end geometric_sequence_at_t_l720_720535


namespace trigonometric_identity_l720_720737

theorem trigonometric_identity (α : ℝ) : 
  (1 + Real.cos (2 * α - 2 * Real.pi) + Real.cos (4 * α + 2 * Real.pi) - Real.cos (6 * α - Real.pi)) /
  (Real.cos (2 * Real.pi - 2 * α) + 2 * Real.cos (2 * α + Real.pi) ^ 2 - 1) = 
  2 * Real.cos (2 * α) :=
by sorry

end trigonometric_identity_l720_720737


namespace angle_alpha_not_2pi_over_9_l720_720752

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) * (Real.cos (2 * x)) * (Real.cos (4 * x))

theorem angle_alpha_not_2pi_over_9 (α : ℝ) (h : f α = 1 / 8) : α ≠ 2 * π / 9 :=
sorry

end angle_alpha_not_2pi_over_9_l720_720752


namespace interest_calculation_time_l720_720996

noncomputable
def P : ℝ := sorry

def r : ℝ := 0.20

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

def compound_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r) ^ t - P

theorem interest_calculation_time :
  (∃ (P t : ℝ), simple_interest P r t = 400 ∧ compound_interest P r t = 440) → ∃ t, t = 2 :=
by
  sorry

end interest_calculation_time_l720_720996


namespace find_m_l720_720330

variable (A B : Set ℝ)
variable m : ℝ

def setA : Set ℝ := { y : ℝ | ∃ (x : ℝ), (x ∈ Icc (-1/2 : ℝ) (2 : ℝ)) ∧ (y = x^2 - (3/2) * x + 1) }
def setB (m : ℝ) : Set ℝ := { x : ℝ | |x - m| ≥ 1 }

theorem find_m (H : ∀ t, t ∈ setA → t ∈ setB m):
  m ≤ (-9/16 : ℝ) ∨ m ≥ (3 : ℝ) := by sorry

end find_m_l720_720330


namespace hypotenuse_length_l720_720904

open Real

-- Definitions corresponding to the conditions
def right_triangle_vertex_length (ADC_length : ℝ) (AEC_length : ℝ) (x : ℝ) : Prop :=
  0 < x ∧ x < π / 2 ∧ ADC_length = sqrt 3 * sin x ∧ AEC_length = sin x

def trisect_hypotenuse (BD : ℝ) (DE : ℝ) (EC : ℝ) (c : ℝ) : Prop :=
  BD = c / 3 ∧ DE = c / 3 ∧ EC = c / 3

-- Main theorem definition
theorem hypotenuse_length (x hypotenuse ADC_length AEC_length : ℝ) :
  right_triangle_vertex_length ADC_length AEC_length x →
  trisect_hypotenuse (hypotenuse / 3) (hypotenuse / 3) (hypotenuse / 3) hypotenuse →
  hypotenuse = sqrt 3 * sin x :=
by
  intros h₁ h₂
  sorry

end hypotenuse_length_l720_720904


namespace compound_interest_correct_l720_720842

variable (P R T : ℝ)
variable (SI CI : ℝ)

def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

def compound_interest (P R T : ℝ) : ℝ := P * (1 + R / 100)^T - P

theorem compound_interest_correct :
  (R = 20) → (T = 2) → (simple_interest P R T = 400) → (compound_interest P R T = 440) :=
by
  intros hR hT hSI
  -- steps to prove will be added here
  sorry

end compound_interest_correct_l720_720842


namespace inequality_of_abc_l720_720409

theorem inequality_of_abc (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by {
  sorry
}

end inequality_of_abc_l720_720409


namespace fabric_nguyen_needs_l720_720919

-- Definitions for conditions
def fabric_per_pant : ℝ := 8.5
def total_pants : ℝ := 7
def yards_to_feet (yards : ℝ) : ℝ := yards * 3
def fabric_nguyen_has_yards : ℝ := 3.5

-- The proof we need to establish
theorem fabric_nguyen_needs : (total_pants * fabric_per_pant) - (yards_to_feet fabric_nguyen_has_yards) = 49 :=
by
  sorry

end fabric_nguyen_needs_l720_720919


namespace general_formula_l720_720397

noncomputable def N_pos : Set ℕ := {n | n > 0}

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n ∈ N_pos, a (n + 1) = (a (n + 1) + 3 * a n) / 2

theorem general_formula (a : ℕ → ℝ) (h1 : a 1 = 3) (h2 : sequence a) :
  ∀ n, a n = 3 ^ n :=
sorry

end general_formula_l720_720397


namespace distance_from_A_to_B_l720_720391

theorem distance_from_A_to_B :
  let A := (0, 0)
  let B := (0, -50) -- 50 yards south
  let B := (fst B - 80, snd B) -- 80 yards west
  let B := (fst B, snd B + 30) -- 30 yards north
  let B := (fst B + 40, snd B) -- 40 yards east
  let distance := Real.sqrt ((20:ℝ)^2 + (40:ℝ)^2)
  distance = 20 * Real.sqrt 5 :=
by
  sorry

end distance_from_A_to_B_l720_720391


namespace min_value_of_y_l720_720739

noncomputable def y (x : Real) : Real :=
  Real.cos (x + 10) + Real.cos (x + 70)

theorem min_value_of_y (x : Real) (h : 0 < x ∧ x < 180) (hx : x = 140) :
  y(x) = -(Real.sqrt 3) :=
  sorry

end min_value_of_y_l720_720739


namespace smallest_five_digit_divisible_by_primes_l720_720711

theorem smallest_five_digit_divisible_by_primes : 
  let primes := [2, 3, 5, 7, 11] in
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11))) in
  let five_digit_threshold := 10000 in
  ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
by
  let primes := [2, 3, 5, 7, 11]
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))
  have lcm_2310 : lcm_primes = 2310 := sorry
  let five_digit_threshold := 10000
  have exists_n : ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
    sorry
  exists_intro 5
  have 5_condition : 5 > 0 := sorry
  have 2310_5_condition : 2310 * 5 >= five_digit_threshold := sorry
  have answer : 2310 * 5 = 11550 := sorry
  exact  ⟨5, 5_condition, 2310_5_condition, answer⟩
  exact ⟨5, 5 > 0, 2310 * 5 ≥ 10000, 2310 * 5 = 11550⟩
  sorry

end smallest_five_digit_divisible_by_primes_l720_720711


namespace installment_payment_l720_720417

theorem installment_payment
  (cash_price : ℕ)
  (down_payment : ℕ)
  (first_four_months_payment : ℕ)
  (last_four_months_payment : ℕ)
  (installment_additional_cost : ℕ)
  (total_next_four_months_payment : ℕ)
  (H_cash_price : cash_price = 450)
  (H_down_payment : down_payment = 100)
  (H_first_four_months_payment : first_four_months_payment = 4 * 40)
  (H_last_four_months_payment : last_four_months_payment = 4 * 30)
  (H_installment_additional_cost : installment_additional_cost = 70)
  (H_total_next_four_months_payment_correct : 4 * total_next_four_months_payment = 4 * 35) :
  down_payment + first_four_months_payment + 4 * 35 + last_four_months_payment = cash_price + installment_additional_cost := 
by {
  sorry
}

end installment_payment_l720_720417


namespace union_sets_l720_720840

variable {α : Type}
variables A B : set α

def A := {x : ℝ | x < 2}
def B := {x : ℝ | 1 < x ∧ x < 7}

theorem union_sets (x : ℝ) : (x ∈ A ∪ B) ↔ (x < 7) := by
  sorry

end union_sets_l720_720840


namespace smallest_five_digit_number_divisible_by_first_five_primes_l720_720705

theorem smallest_five_digit_number_divisible_by_first_five_primes : 
  ∃ n, (n >= 10000) ∧ (n < 100000) ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_first_five_primes_l720_720705


namespace smallest_five_digit_number_divisible_by_five_primes_l720_720723

theorem smallest_five_digit_number_divisible_by_five_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  lcm = 2310 → (∃ n : ℕ, n = 5 ∧ 10000 ≤ lcm * n ∧ lcm * n = 11550) :=
by
  intros p1 p2 p3 p4 p5 
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  intro hlcm
  use (5 : ℕ)
  split
  { exact rfl }
  split
  { sorry }
  { sorry }

end smallest_five_digit_number_divisible_by_five_primes_l720_720723


namespace factor_added_when_moving_from_k_to_k_plus_one_l720_720043

open Nat

theorem factor_added_when_moving_from_k_to_k_plus_one 
  (k : ℕ) (h : k > 0) : 
  ((2 * k + 1) * (2 * k + 2)) / (k + 1) = ((k + 2) * (k + 3) * ... * (k + 1 + k + 1)) / ((k + 1) * (k + 2) * ... * (k + k)) :=
by sorry

end factor_added_when_moving_from_k_to_k_plus_one_l720_720043


namespace part1_no_two_women_sit_next_to_each_other_part2_no_two_spouses_no_two_women_no_two_men_sit_next_to_each_other_l720_720369

-- Part 1: No two women can sit next to each other
def numberOfWaysFirstPart (n : ℕ) : ℕ :=
  if n = 4 then 3! * 4! else 0

theorem part1_no_two_women_sit_next_to_each_other :
  numberOfWaysFirstPart 4 = 144 := by
  unfold numberOfWaysFirstPart
  -- Proof omitted
  sorry

-- Part 2: No two spouses, no two women, and no two men sit next to each other
def numberOfWaysSecondPart (n : ℕ) : ℕ :=
  if n = 4 then 3! * 2 else 0

theorem part2_no_two_spouses_no_two_women_no_two_men_sit_next_to_each_other :
  numberOfWaysSecondPart 4 = 12 := by
  unfold numberOfWaysSecondPart
  -- Proof omitted
  sorry

end part1_no_two_women_sit_next_to_each_other_part2_no_two_spouses_no_two_women_no_two_men_sit_next_to_each_other_l720_720369


namespace part_a_part_c_part_d_l720_720509

-- Define the variables
variables {a b : ℝ}

-- Define the conditions and statements
def cond := a + b > 0

theorem part_a (h : cond) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem part_c (h : cond) : a^21 + b^21 > 0 :=
sorry

theorem part_d (h : cond) : (a + 2) * (b + 2) > a * b :=
sorry

end part_a_part_c_part_d_l720_720509


namespace min_dot_product_l720_720845

variables {V : Type*} [inner_product_space ℝ V]

theorem min_dot_product
  {A B C : V}
  (BC : dist B C = 2)
  (h : ∀ t : ℝ, ∥t • (B - A) + (1 - t) • (C - A)∥ ≥ ∥t_0 • (B - A) + (1 - t_0) • (C - A)∥)
  (t0_condition : ∥t_0 • (B - A) + (1 - t_0) • (C - A)∥ = 3) :
  ∃ (t_0 : ℝ), t_0 = 1/2 ∧ inner (B - A) (C - A) = 8 :=
begin
  sorry
end

end min_dot_product_l720_720845


namespace AQ_parallel_BP_l720_720639

-- Definitions for given conditions:
variables {A B C I P Q : Point} {𝒪 : Circle}
variable [nonempty ℜ]

namespace geometry

structure Triangle (A B C : Point) : Prop := 
(incenter : Point) (incenter_def : incenter = I)

structure Circle (A B I : Point) : Prop :=
(circumcircle : Circle)
(circumcircle_def : circumcircle = 𝒪)

-- Points on intersection with circle
structure Intersection (A B C P Q : Point) : Prop :=
(intersect_CA : inter CA 𝒪 = P)
(intersect_CB : inter CB 𝒪 = Q)

-- Main theorem - prove AQ is parallel to BP
theorem AQ_parallel_BP 
(triangleABC : Triangle A B C)
(circumcircleAIB : Circle A B I)
(intersectionCircumcircle : Intersection C A B P Q) :
  parallel AQ BP := sorry

end geometry

end AQ_parallel_BP_l720_720639


namespace find_smallest_n_l720_720819

noncomputable theory

def sequence (a : ℕ → ℝ) : Prop :=
∀ n, n ≥ 1 → 3 * a (n + 1) + a n = 4

def initial_condition (a : ℕ → ℝ) : Prop :=
a 1 = 9

def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, a (i + 1)

def valid_n (n : ℕ) (a : ℕ → ℝ) : Prop :=
|S_n a n - n - 6| < 1 / 125

theorem find_smallest_n :
∃ n, sequence a ∧ initial_condition a ∧ valid_n n a → n = 7 :=
sorry

end find_smallest_n_l720_720819


namespace root_of_equation_l720_720838

theorem root_of_equation (x : ℝ) (h : real.cbrt (x + 9) - real.cbrt (x - 9) = 3) :
  75 ≤ x^2 ∧ x^2 < 85 :=
begin
  sorry
end

end root_of_equation_l720_720838


namespace gerald_poisoning_sentence_l720_720747

-- Definitions based on conditions
def assault_sentence : Nat := 3
def total_jail_time : Nat := 36
def extension_ratio : Rational := 1/3

-- Statement to be proved
theorem gerald_poisoning_sentence:
  ∃ (total_sentence : Nat) (poisoning_sentence : Nat),
    total_sentence + (total_sentence / 3) = total_jail_time ∧
    total_sentence - assault_sentence = poisoning_sentence ∧
    poisoning_sentence = 24 := 
begin
  sorry
end

end gerald_poisoning_sentence_l720_720747


namespace probability_xi_eq_1_l720_720447

-- Definitions based on conditions
def white_balls_bag_A := 8
def red_balls_bag_A := 4
def white_balls_bag_B := 6
def red_balls_bag_B := 6

-- Combinatorics function for choosing k items from n items
def C (n k : ℕ) : ℕ := Nat.choose n k

-- Definition for probability P(ξ = 1)
def P_xi_eq_1 := 
  (C white_balls_bag_A 1 * C white_balls_bag_B 1 + C red_balls_bag_A 1 * C white_balls_bag_B 1) /
  (C (white_balls_bag_A + red_balls_bag_A) 1 * C (white_balls_bag_B + red_balls_bag_B) 1)

theorem probability_xi_eq_1 :
  P_xi_eq_1 = (C 8 1 * C 6 1 + C 4 1 * C 6 1) / (C 12 1 * C 12 1) :=
by
  sorry

end probability_xi_eq_1_l720_720447


namespace max_sum_arith_seq_l720_720855

variable {α : Type*} [OrderedCommGroup α]

def arithmetic_seq (a : ℕ → α) := ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def sum_seq (a : ℕ → α) (S : ℕ → α) := ∀ n : ℕ, S n = List.sum (List.map a (List.range (n + 1)))

theorem max_sum_arith_seq (a : ℕ → α) (S : ℕ → α) (h_arith : arithmetic_seq a) (h_a1_pos : 0 < a 1) (h_sum_eq : S 3 = S 10) : 
  ∃ n : ℕ, n = 6 ∨ n = 7 ∧ ∀ m : ℕ, S m ≤ S n := 
sorry

end max_sum_arith_seq_l720_720855


namespace coordinates_of_point_A_l720_720922

    theorem coordinates_of_point_A (x y : ℝ) (h1 : y = 0) (h2 : abs x = 3) : (x, y) = (3, 0) ∨ (x, y) = (-3, 0) :=
    sorry
    
end coordinates_of_point_A_l720_720922


namespace smallest_five_digit_number_divisible_by_primes_l720_720734

theorem smallest_five_digit_number_divisible_by_primes : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
begin
  sorry
end

end smallest_five_digit_number_divisible_by_primes_l720_720734


namespace gnuff_tutoring_rate_l720_720825

theorem gnuff_tutoring_rate (flat_rate : ℕ) (total_paid : ℕ) (minutes : ℕ) :
  flat_rate = 20 → total_paid = 146 → minutes = 18 → (total_paid - flat_rate) / minutes = 7 :=
by
  intros
  sorry

end gnuff_tutoring_rate_l720_720825


namespace two_point_three_five_as_fraction_l720_720072

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l720_720072


namespace combined_weight_of_new_students_l720_720003

theorem combined_weight_of_new_students 
  (avg_weight_orig : ℝ) (num_students_orig : ℝ) 
  (new_avg_weight : ℝ) (num_new_students : ℝ) 
  (total_weight_gain_orig : ℝ) (total_weight_loss_orig : ℝ)
  (total_weight_orig : ℝ := avg_weight_orig * num_students_orig) 
  (net_weight_change_orig : ℝ := total_weight_gain_orig - total_weight_loss_orig)
  (total_weight_after_change_orig : ℝ := total_weight_orig + net_weight_change_orig) 
  (total_students_after : ℝ := num_students_orig + num_new_students) 
  (total_weight_class_after : ℝ := new_avg_weight * total_students_after) : 
  total_weight_class_after - total_weight_after_change_orig = 586 :=
by
  sorry

end combined_weight_of_new_students_l720_720003


namespace parallel_MN_PQ_l720_720940

variables {A B C I Z M N P Q : Point}
variables {BI IZ PQ MN : Line}

-- Conditions
variables (triangle_ABC : is_triangle A B C)
variables (incircle_center_I : is_incircle_center I A B C)
variables (angle_bisector_IZ : is_angle_bisector Z B I triangle_ABC)
variables (perpendicular_PQ_IZ : is_perpendicular PQ IZ)
variables (perpendicular_IZ_BI : is_perpendicular IZ BI)
variables (perpendicular_MN_BI : is_perpendicular MN BI)

theorem parallel_MN_PQ 
  (h1: is_angle_bisector Z B I triangle_ABC)
  (h2: is_perpendicular PQ IZ)
  (h3: is_perpendicular IZ BI)
  (h4: is_perpendicular MN BI) : 
  is_parallel MN PQ := 
sorry

end parallel_MN_PQ_l720_720940


namespace number_of_valid_rods_l720_720874

theorem number_of_valid_rods : ∃ n, n = 22 ∧
  (∀ (d : ℕ), 1 < d ∧ d < 25 ∧ d ≠ 4 ∧ d ≠ 9 ∧ d ≠ 12 → d ∈ {d | d > 0}) :=
by
  use 22
  sorry

end number_of_valid_rods_l720_720874


namespace smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l720_720698

theorem smallest_five_digit_number_divisible_by_prime_2_3_5_7_11 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l720_720698


namespace expression_rational_iff_x_rational_l720_720247

theorem expression_rational_iff_x_rational (x : ℝ) :
  (∃ r : ℚ, (x + sqrt (x^2 - 2) - 1 / (x + sqrt (x^2 - 2)) : ℝ) = r) ↔ (∃ q : ℚ, (x : ℝ) = q) :=
by
  sorry

end expression_rational_iff_x_rational_l720_720247


namespace decimal_to_fraction_l720_720102

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l720_720102


namespace relay_team_order_count_l720_720386

def num_ways_to_order_relay (total_members : Nat) (jordan_lap : Nat) : Nat :=
  if jordan_lap = total_members then (total_members - 1).factorial else 0

theorem relay_team_order_count : num_ways_to_order_relay 5 5 = 24 :=
by
  -- the proof would go here
  sorry

end relay_team_order_count_l720_720386


namespace minimum_value_of_I_l720_720276

noncomputable def I (a : ℝ) : ℝ :=
  ∫ x in 0..(Real.pi / 2), abs ((sin (2 * x) / (1 + (sin x)^2)) - a * (cos x))

theorem minimum_value_of_I : ∃ a : ℝ, I(a) = (3 * Real.log 2 - 2 * Real.log 5) :=
by
  sorry

end minimum_value_of_I_l720_720276


namespace part1_part2_l720_720817

open Function

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0       => 2
| (n+1) => a n + 2^n + 2

-- Define the shifted sequence {b_n = a_n - 2^n}
def b (n : ℕ) : ℕ := a n - 2^n

-- Prove that b_n is an arithmetic sequence with first term 0 and common difference 2
theorem part1 : b 0 = 0 ∧ ∀ n : ℕ, b (n+1) = b n + 2 :=
by
  sorry

-- Define the sequence {S_n} being the sum of the first n terms of {a_n}
def sum_of_an (n : ℕ) : ℕ := (Finset.range n).sum a

-- Prove that S_n = 2^(n+1) - 2 + n^2 - n
theorem part2 (n : ℕ) : sum_of_an n = 2^(n+1) - 2 + n^2 - n :=
by
  sorry

end part1_part2_l720_720817


namespace max_profit_l720_720158

noncomputable def profit_function (x : ℕ) : ℝ :=
  if x ≤ 400 then
    300 * x - (1 / 2) * x^2 - 20000
  else
    60000 - 100 * x

theorem max_profit : 
  (∀ x ≥ 0, profit_function x ≤ 25000) ∧ (profit_function 300 = 25000) :=
by 
  sorry

end max_profit_l720_720158


namespace gcd_7384_12873_l720_720267

theorem gcd_7384_12873 : Int.gcd 7384 12873 = 1 :=
by
  sorry

end gcd_7384_12873_l720_720267


namespace quadratic_inequality_solution_set_quadratic_inequality_solution_set2_l720_720813

-- Proof Problem 1 Statement
theorem quadratic_inequality_solution_set (a b : ℝ) (h : ∀ x : ℝ, b < x ∧ x < 1 → ax^2 + 3 * x + 2 > 0) : 
  a = -5 ∧ b = -2/5 := sorry

-- Proof Problem 2 Statement
theorem quadratic_inequality_solution_set2 (a : ℝ) (h_pos : a > 0) : 
  ((0 < a ∧ a < 3) → (∀ x : ℝ, x < -3 / a ∨ x > -1 → ax^2 + 3 * x + 2 > -ax - 1)) ∧
  (a = 3 → (∀ x : ℝ, x ≠ -1 → ax^2 + 3 * x + 2 > -ax - 1)) ∧
  (a > 3 → (∀ x : ℝ, x < -1 ∨ x > -3 / a → ax^2 + 3 * x + 2 > -ax - 1)) := sorry

end quadratic_inequality_solution_set_quadratic_inequality_solution_set2_l720_720813


namespace find_trajectory_l720_720454

noncomputable def trajectory_of_point (M : ℝ × ℝ) : Prop :=
  let F : ℝ × ℝ := (4, 0)
  let directrix := {x : ℝ // x = -6}
  let directrix2 := {x : ℝ // x = -4}
  (dist M F = dist M directrix2) → (M.snd^2 = 16 * M.fst)

theorem find_trajectory (M : ℝ × ℝ) : trajectory_of_point M :=
  sorry

end find_trajectory_l720_720454


namespace rational_coefficient_term_is_third_l720_720858

noncomputable def general_term (r : ℕ) : ℝ :=
  Nat.choose 4 r * (-1 : ℝ)^r * (2 : ℝ)^((4 - 2 * r) / 3 : ℝ)

theorem rational_coefficient_term_is_third :
  ∃ (r : ℕ), 0 ≤ r ∧ r ≤ 4 ∧ (general_term r) = 1 ∧ r = 2 
  :=
by
  use 2
  split
  { exact Nat.zero_le _ }
  split
  { -- We use the fact that r <= 4
    exact le_refl 4 }
  split
  { -- We compute the general term when r = 2
    simp [general_term]
    sorry } -- You would compute the term and show it is 1 here
  { -- Finally, we show r = 2
    exact rfl }

end rational_coefficient_term_is_third_l720_720858


namespace range_of_m_l720_720843

noncomputable def quadratic_expr (x : ℝ) : ℝ := x^2 - 2*x + 5

theorem range_of_m :
  ∃ x ∈ set.Icc (2 : ℝ) (4 : ℝ), quadratic_expr x < m ↔ m ∈ set.Ioi (5 : ℝ) :=
by
  sorry

end range_of_m_l720_720843


namespace printing_time_345_l720_720176

def printing_time (total_pages : ℕ) (rate : ℕ) : ℕ :=
  total_pages / rate

theorem printing_time_345 :
  printing_time 345 23 = 15 :=
by
  sorry

end printing_time_345_l720_720176


namespace count_triangles_l720_720337

-- Define the problem conditions
def num_small_triangles : ℕ := 11
def num_medium_triangles : ℕ := 4
def num_large_triangles : ℕ := 1

-- Define the main statement asserting the total number of triangles
theorem count_triangles (small : ℕ) (medium : ℕ) (large : ℕ) :
  small = num_small_triangles →
  medium = num_medium_triangles →
  large = num_large_triangles →
  small + medium + large = 16 :=
by
  intros h_small h_medium h_large
  rw [h_small, h_medium, h_large]
  sorry

end count_triangles_l720_720337


namespace decimal_to_fraction_l720_720103

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l720_720103


namespace minimum_percentage_increase_mean_l720_720136

def mean (s : List ℤ) : ℚ :=
  (s.sum : ℚ) / s.length

theorem minimum_percentage_increase_mean (F : List ℤ) (p1 p2 : ℤ) (F' : List ℤ)
  (hF : F = [ -4, -1, 0, 6, 9 ])
  (hp1 : p1 = 2) (hp2 : p2 = 3)
  (hF' : F' = [p1, p2, 0, 6, 9])
  : (mean F' - mean F) / mean F * 100 = 100 := 
sorry

end minimum_percentage_increase_mean_l720_720136


namespace determine_x_squared_plus_y_squared_l720_720832

theorem determine_x_squared_plus_y_squared (x y : ℝ) 
(h : (x^2 + y^2 + 2) * (x^2 + y^2 - 3) = 6) : x^2 + y^2 = 4 :=
sorry

end determine_x_squared_plus_y_squared_l720_720832


namespace poolDrainTime_l720_720971

/-- The problem parameters -/
structure PoolSection :=
  (width : ℝ)
  (length : ℝ)
  (depth : ℝ)
  (radius : ℝ := 0)
  (base : ℝ := 0)
  (height : ℝ := 0)

/-- Helper function to calculate volume of a rectangular section -/
def rectVolume (length width depth : ℝ) : ℝ := length * width * depth

/-- Helper function to calculate volume of a semicircular section -/
def semiCircVolume (radius depth : ℝ) : ℝ := (real.pi * radius ^ 2 * depth) / 2

/-- Helper function to calculate volume of a triangular section -/
def triVolume (base height depth : ℝ) : ℝ := (base * height * depth) / 2

/-- Total volume at full capacity -/
def totalVolume (rect semiCirc tri : ℝ) : ℝ := rect + semiCirc + tri

/-- Volume at 80% capacity -/
def volumeAt80Percent (totalVol : ℝ) : ℝ := 0.8 * totalVol

/-- Combined rate of two hoses -/
def combinedRate (rateA rateB : ℝ) : ℝ := rateA + rateB

/-- Time to drain the pool -/
def drainTime (vol rate : ℝ) : ℝ := vol / rate

/-- Problem statement -/
theorem poolDrainTime :
  let rectVol := rectVolume 100 60 10,
      semiCircVol := semiCircVolume 30 10,
      triVol := triVolume 50 40 10,
      fullVol := totalVolume rectVol semiCircVol triVol,
      vol80 := volumeAt80Percent fullVol,
      rate := combinedRate 60 45 in
  drainTime vol80 rate = 641.04 :=
by sorry

end poolDrainTime_l720_720971


namespace smallest_five_digit_number_divisible_by_first_five_primes_l720_720703

theorem smallest_five_digit_number_divisible_by_first_five_primes : 
  ∃ n, (n >= 10000) ∧ (n < 100000) ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_first_five_primes_l720_720703


namespace weighted_avg_correct_to_nearest_tenth_l720_720182

-- Definitions based on conditions
def year1_courses := 8
def year1_avg_grade := 92
def year2_courses := 6
def year2_avg_grade := 88
def year3_courses := 10
def year3_avg_grade := 76

-- Derived calculations for total points and number of courses
def total_points := year1_courses * year1_avg_grade + year2_courses * year2_avg_grade + year3_courses * year3_avg_grade
def total_courses := year1_courses + year2_courses + year3_courses

-- Weighted average calculation based on given conditions
def weighted_average := (total_points : ℚ) / total_courses

-- Proof statement asserting the calculated weighted average is as expected
theorem weighted_avg_correct_to_nearest_tenth :
  round (weighted_average * 10) / 10 = 84.3 := by sorry

end weighted_avg_correct_to_nearest_tenth_l720_720182


namespace number_of_right_angled_triangles_with_incenter_at_origin_l720_720372

def lattice_point (p : ℤ × ℤ) : Prop := true

def is_right_angle_at (A B C : ℤ × ℤ) : Prop :=
(B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

def is_incenter (A B C : ℤ × ℤ) (I : ℤ × ℤ) : Prop :=
  (A, B, C) → (I.1 * (B.2 - C.2) + I.2 * (C.1 - B.1) + I.1 * A.2 - I.2 * A.1 = 0)

theorem number_of_right_angled_triangles_with_incenter_at_origin :
  (A B C : ℤ × ℤ), 
  (lattice_point B) ∧ (lattice_point C) ∧ is_right_angle_at (12, 84) B C ∧ is_incenter (12, 84) B C (0, 0) → 
  ∃ n : ℕ, n = 18 :=
sorry

end number_of_right_angled_triangles_with_incenter_at_origin_l720_720372


namespace MN_parallel_PQ_l720_720927

theorem MN_parallel_PQ
  (A B C I Z M N P Q : Point)
  (circumcircle : Circle)
  (h1 : I = circumcenter A B C)
  (h2 : midpoint I (arc A C circumcircle))
  (h3 : angle_bisector I Z (angle B A C))
  (h4 : perpendicular PQ IZ)
  (h5 : isosceles MBI with base BI)
  (h6 : isosceles NBI with base BI)
  : parallel PQ MN :=
sorry -- Proof to be filled in later by the user

end MN_parallel_PQ_l720_720927


namespace hyperbola_equation_l720_720804

theorem hyperbola_equation
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : b = a / sqrt 3)
  (h₄ : a^2 + b^2 = 36)
  (y x : ℝ) :
  (y^2 / 27 - x^2 / 9 = 1) :=
by
  sorry

end hyperbola_equation_l720_720804


namespace heximal_to_binary_k_value_l720_720355

theorem heximal_to_binary_k_value (k : ℕ) (h : 10 * (6^3) + k * 6 + 5 = 239) : 
  k = 3 :=
by
  sorry

end heximal_to_binary_k_value_l720_720355


namespace max_interval_length_l720_720020

noncomputable def log_base (b x : ℝ) := log x / log b
noncomputable def abs_log_half (x : ℝ) := |log_base (0.5) x|

theorem max_interval_length :
  ∀ a b : ℝ, (∀ x : ℝ, x ∈ set.Icc a b ↔ abs_log_half x ∈ set.Icc 0 2) →
  (b - a = 15/4) :=
begin
  sorry
end

end max_interval_length_l720_720020


namespace train_length_problem_l720_720018

noncomputable def train_length (v : ℝ) (t : ℝ) (L : ℝ) : Prop :=
v = 90 / 3.6 ∧ t = 60 ∧ 2 * L = v * t

theorem train_length_problem : train_length 90 1 750 :=
by
  -- Define speed in m/s
  let v_m_s := 90 * (1000 / 3600)
  -- Calculate distance = speed * time
  let distance := 25 * 60
  -- Since distance = 2 * Length
  have h : 2 * 750 = 1500 := sorry
  show train_length 90 1 750
  simp [train_length, h]
  sorry

end train_length_problem_l720_720018


namespace triangle_area_inscribed_in_circle_l720_720186

theorem triangle_area_inscribed_in_circle :
  ∀ (x : ℝ), (2 * x)^2 + (3 * x)^2 = (4 * x)^2 → (5 = (4 * x) / 2) → (1/2 * (2 * x) * (3 * x) = 18.75) :=
by
  -- Assume all necessary conditions
  intros x h_ratio h_radius
  -- Skip the proof part using sorry
  sorry

end triangle_area_inscribed_in_circle_l720_720186


namespace decimal_to_fraction_l720_720094

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l720_720094


namespace max_smoothie_servings_l720_720178

-- Define the constants based on the problem conditions
def servings_per_recipe := 4
def bananas_per_recipe := 3
def yogurt_per_recipe := 1 -- cup
def honey_per_recipe := 2 -- tablespoons
def strawberries_per_recipe := 2 -- cups

-- Define the total amount of ingredients Lynn has
def total_bananas := 12
def total_yogurt := 6 -- cups
def total_honey := 16 -- tablespoons (since 1 cup = 16 tablespoons)
def total_strawberries := 8 -- cups

-- Define the calculation for the number of servings each ingredient can produce
def servings_from_bananas := (total_bananas / bananas_per_recipe) * servings_per_recipe
def servings_from_yogurt := (total_yogurt / yogurt_per_recipe) * servings_per_recipe
def servings_from_honey := (total_honey / honey_per_recipe) * servings_per_recipe
def servings_from_strawberries := (total_strawberries / strawberries_per_recipe) * servings_per_recipe

-- Define the minimum number of servings that can be made based on all ingredients
def max_servings := min servings_from_bananas (min servings_from_yogurt (min servings_from_honey servings_from_strawberries))

theorem max_smoothie_servings : max_servings = 16 :=
by
  sorry

end max_smoothie_servings_l720_720178


namespace largest_no_floating_plus_l720_720277

noncomputable def maxBlackSquares (n : ℕ) : ℕ := 4 * n - 4

theorem largest_no_floating_plus (n : ℕ) (h : n ≥ 3) : 
    ∃ k, k = maxBlackSquares n ∧ (∀ M L R A B : (Fin n) × (Fin n),
      (L.1 = M.1 ∧ L.2 < M.2) ∧ 
      (R.1 = M.1 ∧ R.2 > M.2) ∧ 
      (A.2 = M.2 ∧ A.1 < M.1) ∧ 
      (B.2 = M.2 ∧ B.1 > M.1) → 
      (¬(is_black M ∧ is_black L ∧ is_black R ∧ is_black A ∧ is_black B))) :=
begin
  existsi maxBlackSquares n,
  split,
  refl,
  sorry
end

end largest_no_floating_plus_l720_720277


namespace min_students_received_all_three_exams_l720_720434

theorem min_students_received_all_three_exams
  (M_1 M_2 M_3 M_{12} M_{13} M_{23} M_{123} : ℕ)
  (h1 : M_1 + M_{12} + M_{13} + M_{123} = 160)
  (h2 : M_2 + M_{12} + M_{23} + M_{123} = 140)
  (h3 : M_3 + M_{23} + M_{13} + M_{123} = 118)
  (h_total : M_1 + M_2 + M_3 + M_{12} + M_{13} + M_{23} + M_{123} ≤ 200)
  : 18 ≤ M_{123} :=
by
  sorry

end min_students_received_all_three_exams_l720_720434


namespace smallest_positive_five_digit_number_divisible_by_first_five_primes_l720_720718

theorem smallest_positive_five_digit_number_divisible_by_first_five_primes :
  ∃ n : ℕ, (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ 10000 ≤ n ∧ n = 11550 :=
by
  use 11550
  split
  · intros p hp
    fin_cases hp <;> norm_num
  split
  · norm_num
  rfl

end smallest_positive_five_digit_number_divisible_by_first_five_primes_l720_720718


namespace triangle_angle_ge_60_l720_720044

theorem triangle_angle_ge_60 {A B C : ℝ} (h : A + B + C = 180) :
  A < 60 ∧ B < 60 ∧ C < 60 → false :=
by
  sorry

end triangle_angle_ge_60_l720_720044


namespace pages_revised_once_l720_720533

theorem pages_revised_once (total_pages revised_twice cost_per_page cost_per_rev total_cost : ℕ) (h₁ : total_pages = 100) (h₂ : revised_twice = 20) (h₃ : cost_per_page = 10) (h₄ : cost_per_rev = 5) (h₅ : total_cost = 1350) : ∃ x : ℕ, 1000 + 5 * x + 200 = 1350 ∧ x = 30 :=
by 
  let x := 30
  use x
  split
  ; calc
    1000 + 5 * x + 200 = 1000 + 5 * 30 + 200 : by refl
    ... = 1000 + 150 + 200 : by refl
    ... = 1350 : by refl
  ; refl

end pages_revised_once_l720_720533


namespace cannot_form_triangle_sets_l720_720198

theorem cannot_form_triangle_sets (A B C D : ℕ × ℕ × ℕ) 
    (hA : A = (3, 4, 5)) 
    (hB : B = (5, 10, 8)) 
    (hC : C = (5, 4.5, 8)) 
    (hD : D = (7, 7, 15)) :
    (¬ (A.1 + A.2 > A.3 ∧ A.1 + A.3 > A.2 ∧ A.2 + A.3 > A.1) ∨
    ¬ (B.1 + B.2 > B.3 ∧ B.1 + B.3 > B.2 ∧ B.2 + B.3 > B.1) ∨
    ¬ (C.1 + C.2 > C.3 ∧ C.1 + C.3 > C.2 ∧ C.2 + C.3 > C.1) ∨
    (D.1 + D.2 ≤ D.3 ∨ D.1 + D.3 ≤ D.2 ∨ D.2 + D.3 ≤ D.1)) = true :=  
by
  sorry

end cannot_form_triangle_sets_l720_720198


namespace total_number_of_workers_l720_720591

variables (W N : ℕ)
variables (average_salary_workers average_salary_techs average_salary_non_techs : ℤ)
variables (num_techs total_salary total_salary_techs total_salary_non_techs : ℤ)

theorem total_number_of_workers (h1 : average_salary_workers = 8000)
                               (h2 : average_salary_techs = 14000)
                               (h3 : num_techs = 7)
                               (h4 : average_salary_non_techs = 6000)
                               (h5 : total_salary = W * 8000)
                               (h6 : total_salary_techs = 7 * 14000)
                               (h7 : total_salary_non_techs = N * 6000)
                               (h8 : total_salary = total_salary_techs + total_salary_non_techs)
                               (h9 : W = 7 + N) : 
                               W = 28 :=
sorry

end total_number_of_workers_l720_720591


namespace decimal_to_fraction_l720_720083

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l720_720083


namespace original_price_of_trouser_l720_720873

theorem original_price_of_trouser (sale_price : ℝ) (percent_decrease : ℝ) (original_price : ℝ) 
  (h1 : sale_price = 75) 
  (h2 : percent_decrease = 0.25) 
  (h3 : original_price - percent_decrease * original_price = sale_price) : 
  original_price = 100 :=
by
  sorry

end original_price_of_trouser_l720_720873


namespace log_base_eq_l720_720262

theorem log_base_eq (x : ℝ) : (logBase x 16 = (1 / 3)) → (logBase 64 4 = (1 / 3)) → x = 4096 :=
by
  sorry

end log_base_eq_l720_720262


namespace OM_geq_ON_l720_720597

variables {A B C D E F G H P Q M N O : Type*}

-- Definitions for geometrical concepts
def is_intersection_of_diagonals (M : Type*) (A B C D : Type*) : Prop :=
-- M is the intersection of the diagonals AC and BD
sorry

def is_intersection_of_midlines (N : Type*) (A B C D : Type*) : Prop :=
-- N is the intersection of the midlines connecting the midpoints of opposite sides
sorry

def is_center_of_circumscribed_circle (O : Type*) (A B C D : Type*) : Prop :=
-- O is the center of the circumscribed circle around quadrilateral ABCD
sorry

-- Proof problem
theorem OM_geq_ON (A B C D M N O : Type*) 
  (hm : is_intersection_of_diagonals M A B C D)
  (hn : is_intersection_of_midlines N A B C D)
  (ho : is_center_of_circumscribed_circle O A B C D) : 
  ∃ (OM ON : ℝ), OM ≥ ON :=
sorry

end OM_geq_ON_l720_720597


namespace MN_parallel_PQ_l720_720926

theorem MN_parallel_PQ
  (A B C I Z M N P Q : Point)
  (circumcircle : Circle)
  (h1 : I = circumcenter A B C)
  (h2 : midpoint I (arc A C circumcircle))
  (h3 : angle_bisector I Z (angle B A C))
  (h4 : perpendicular PQ IZ)
  (h5 : isosceles MBI with base BI)
  (h6 : isosceles NBI with base BI)
  : parallel PQ MN :=
sorry -- Proof to be filled in later by the user

end MN_parallel_PQ_l720_720926


namespace smallest_five_digit_number_divisible_by_five_primes_l720_720725

theorem smallest_five_digit_number_divisible_by_five_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  lcm = 2310 → (∃ n : ℕ, n = 5 ∧ 10000 ≤ lcm * n ∧ lcm * n = 11550) :=
by
  intros p1 p2 p3 p4 p5 
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  intro hlcm
  use (5 : ℕ)
  split
  { exact rfl }
  split
  { sorry }
  { sorry }

end smallest_five_digit_number_divisible_by_five_primes_l720_720725


namespace regular_decagon_interior_angle_sum_divided_by_number_is_144_l720_720738

theorem regular_decagon_interior_angle_sum_divided_by_number_is_144 :
  ∀ n : ℕ, n = 10 → (n - 2) * 180 / n = 144 := by
  intro n hn
  have h1 : (n - 2) * 180 = 1440 := by
    rw [hn]
    norm_num
  have h2 : (n - 2) * 180 / n = 144 := by
    rw [hn]
    norm_num
  exact h2

end regular_decagon_interior_angle_sum_divided_by_number_is_144_l720_720738


namespace sum_cubics_evaluation_equals_2016_l720_720404

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 3 * x - (5/12)

theorem sum_cubics_evaluation_equals_2016 :
  ∑ k in (Finset.range 2016).map (λ k, k.succ), f (k / 2017) = 2016 :=
begin
  sorry
end

end sum_cubics_evaluation_equals_2016_l720_720404


namespace largest_n_satisfying_equation_l720_720242

theorem largest_n_satisfying_equation :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ ∀ n : ℕ,
  (n * n = x * x + y * y + z * z + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 12) →
  n ≤ 2 :=
by
  sorry

end largest_n_satisfying_equation_l720_720242


namespace solve_system_eq_l720_720238

theorem solve_system_eq (a1 a2 : ℚ) :
  (a1 * 1 + a2 * 3 = 5) ∧ (a1 * 4 - a2 * 2 = 0) →
  a1 = 5/7 ∧ a2 = 25/21 :=
by
  intro h
  cases h with h1 h2
  have eq1 : a1 + 3 * a2 = 5 := h1
  have eq2 : 4 * a1 - 2 * a2 = 0 := h2
  sorry

end solve_system_eq_l720_720238


namespace petya_time_comparison_l720_720218

variables (D V : ℝ) (hD_pos : D > 0) (hV_pos : V > 0)

theorem petya_time_comparison (hD_pos : D > 0) (hV_pos : V > 0) :
  (41 * D / (40 * V)) > (D / V) :=
by
  sorry

end petya_time_comparison_l720_720218


namespace angle_XOY_invariant_l720_720983

-- Define the basic geometric entities and conditions
variables {O X Y A B P : Type} [circle : Circ O] [tangent_PA : TangentAt P A] [tangent_PB : TangentAt P B] [tangent_XY : TangentIntersection X Y]
variables (Angle : O × O × O → ℝ)

-- Define the statement
theorem angle_XOY_invariant (h1 : tangent O P A) (h2 : tangent O P B) (h3 : tangent O X O) (h4 : tangent O Y O)
  (h5 : ∠ X O A = ∠ X O (O : O)) : 
  ∠ X O Y = (∠ A O B) / 2 :=
sorry

end angle_XOY_invariant_l720_720983


namespace log_expression_equals_zero_l720_720650

-- Defining the properties as hypotheses
axiom log_property1 (a b : ℝ) (ha : a > 0) (hb : b > 0) : log 10 (a * b) = log 10 a + log 10 b
axiom log_property2 (a b : ℝ) (ha : a > 0) (hb : b > 0) : log 10 (a / b) = log 10 a - log 10 b
axiom log_property3 (a : ℝ) (b : ℝ) (ha : a > 0) : log 10 (a^b) = b * log 10 a
axiom log_property4 : log 10 1 = 0

-- The main theorem to be proved
theorem log_expression_equals_zero : log 10 14 - 2 * log 10 (7 / 3) + log 10 7 - log 10 18 = 0 :=
by {
  -- the proof would go here, but it is not required
  sorry
}

end log_expression_equals_zero_l720_720650


namespace tank_overflow_time_l720_720920

theorem tank_overflow_time :
  let rate_A := 1 / 24
  let rate_B := 6 * rate_A
  let rate_C := (3 * rate_A + (1 / 2) * rate_A) / 2
  let combined_rate := rate_A + rate_B + rate_C
  (1 / combined_rate) = 96 / 35 :=
by
  have rate_A := 1 / 24
  have rate_B := 6 * rate_A
  have rate_C := (3 * rate_A + (1 / 2) * rate_A) / 2
  have combined_rate := rate_A + rate_B + rate_C
  calc
    (1 / combined_rate) = (1 / ((1 / 24) + (6 * (1 / 24)) + ((3 * (1 / 24) + (1 / 2) * (1 / 24)) / 2))) : by rfl
    ... = 96 / 35 : sorry

end tank_overflow_time_l720_720920


namespace points_coplanar_x_eq_eight_l720_720287

theorem points_coplanar_x_eq_eight :
  ∃ (x : ℝ), ∀ (O A B C : ℝ × ℝ × ℝ), 
    O = (0, 0, 0) → A = (-2, 2, -2) → B = (1, 4, -6) → C = (x, -8, 8) →
    ∃ (λ μ : ℝ), C = λ • A + μ • B → x = 8 :=
begin
  sorry,
end

end points_coplanar_x_eq_eight_l720_720287


namespace ministers_receive_decree_l720_720548

variable (n : Nat)

def unique_decrees (ministers : Fin n → Set Nat) : Prop :=
  ∀ i j, i ≠ j → ministers i ≠ ministers j

def telegrams_exchange (ministers : Fin n → Set Nat) (telegrams : List (Fin n × Fin n)) : Prop :=
  ∀ t ∈ telegrams, (telegrams.filter (λ p, p.fst = t.fst)).length = 1

def all_ministers_informed (ministers : Fin n → Set Nat) : Prop :=
  ∀ i, ∀ m ∈ ministers i, (∃ t ∈ telegrams, t.snd = i)

theorem ministers_receive_decree :
  ∀ (ministers : Fin n → Set Nat) (telegrams : List (Fin n × Fin n)),
    unique_decrees ministers →
    telegrams_exchange ministers telegrams →
    all_ministers_informed ministers telegrams →
    ∃ k, k ≥ n - 1 ∧ (∃ received : Fin k → Bool, ∀ i : Fin k, received i = true) :=
by
  sorry

end ministers_receive_decree_l720_720548


namespace hyperbola_eccentricity_is_correct_l720_720327

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h : a > 0 ∧ b > 0) 
    (focus : c^2 = 8 * a^2)
    (parallelogram_area : ∃ (x₀ y₀ : ℝ), y₀ > 0 ∧ x₀ = c / 2 ∧ y₀ = b
        ∧ (x₀² / a² - y₀² / b² = 1)
        ∧ y₀ * (c / 2) = bc):
    ℝ :=
    2 * real.sqrt 2

theorem hyperbola_eccentricity_is_correct (a b c : ℝ) (h : a > 0 ∧ b > 0) 
    (focus : c^2 = 8 * a^2)
    (parallelogram_area : ∃ (x₀ y₀ : ℝ), y₀ > 0 ∧ x₀ = c / 2 ∧ y₀ = b
        ∧ (x₀² / a² - y₀² / b² = 1)
        ∧ y₀ * (c / 2) = bc):
    hyperbola_eccentricity a b c h focus parallelogram_area = 2 * real.sqrt 2 :=
sorry

end hyperbola_eccentricity_is_correct_l720_720327


namespace union_set_A_set_B_l720_720903

def set_A : Set ℝ := { x | x^2 - 5 * x - 6 < 0 }
def set_B : Set ℝ := { x | -3 < x ∧ x < 2 }
def set_union (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∨ x ∈ B }

theorem union_set_A_set_B : set_union set_A set_B = { x | -3 < x ∧ x < 6 } := 
by sorry

end union_set_A_set_B_l720_720903


namespace exists_three_players_with_losses_l720_720602

theorem exists_three_players_with_losses (players : Finset ℕ) (matchup : ℕ → ℕ → Prop) 
  (H_players : players.card = 14)
  (H_matchup : ∀ a b ∈ players, a ≠ b → (matchup a b ∨ matchup b a))
  (H_no_draws : ∀ a b ∈ players, a ≠ b → ¬ (matchup a b ∧ matchup b a)) : 
  ∃ (A B C : ℕ), 
  A ∈ players ∧ B ∈ players ∧ C ∈ players ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
  (∀ x ∈ players, x ≠ A ∧ x ≠ B ∧ x ≠ C → (matchup A x ∨ matchup B x ∨ matchup C x)) :=
sorry

end exists_three_players_with_losses_l720_720602


namespace problem_a_problem_c_problem_d_l720_720502

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end problem_a_problem_c_problem_d_l720_720502


namespace carol_mike_same_amount_in_weeks_l720_720236

-- Define the initial amounts and savings rates
def carol_initial_amount : ℝ := 40
def carol_saving_rate : ℝ := 12
def mike_initial_amount : ℝ := 150
def mike_saving_rate : ℝ := 2

-- Define the number of weeks when Carol and Mike have the same amount of money
def weeks_to_equal_amount : ℝ := by
  have h := carol_initial_amount + carol_saving_rate * 11 = mike_initial_amount + mike_saving_rate * 11
  exact 11

-- Prove that the calculated number of weeks is correct
theorem carol_mike_same_amount_in_weeks :
  ∀ w : ℝ, (carol_initial_amount + carol_saving_rate * w = mike_initial_amount + mike_saving_rate * w) ↔ (w = 11) :=
by
  intro w
  split
  · intro h
    -- Solving Carol's and Mike's money equation
    have eq1 : 12 * w - 2 * w = 150 - 40 := calc
      12 * w - 2 * w
        = (40 + 12 * w) - (150 + 2 * w) : by rw [h]
        ... = (40 - 150) + (12 * w - 2 * w) : by ring
        ... = 110 : by ring
    have eq2 : 10 * w = 110 := by {
      ring_nf at eq1,
      exact eq1 }
    exact (eq_of_mul_eq_mul_right (by norm_num : 10 ≠ 0) eq2).symm
  · intro h
    rw h
    ring_nf
    trivial

end carol_mike_same_amount_in_weeks_l720_720236


namespace smallest_five_digit_number_divisible_by_primes_l720_720733

theorem smallest_five_digit_number_divisible_by_primes : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
begin
  sorry
end

end smallest_five_digit_number_divisible_by_primes_l720_720733


namespace find_a_and_monotonicity_find_extrema_l720_720796

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x + 4

theorem find_a_and_monotonicity :
    (∃ a : ℝ, (∀ x : ℝ, ((1 / 3 : ℝ) * x^3 - a * x + 4).deriv x = x^2 - a) ∧ a = 4) ∧
    ∀ x : ℝ, monotone_on (λ x, (1 / 3) * x^3 - 4 * x + 4) (-∞, -2) ∧
              antitone_on (λ x, (1 / 3) * x^3 - 4 * x + 4) (-2, 2) ∧
              monotone_on (λ x, (1 / 3) * x^3 - 4 * x + 4) (2, +∞) :=
by sorry

theorem find_extrema :
    ∃ min_x max_x : ℝ, (∀ x : ℝ, x ∈ set.Icc 0 3 → (λ x, (1 / 3) * x^3 - 4 * x + 4 x = min_x) ∧
    (∀ x : ℝ, x ∈ set.Icc 0 3 → (λ x, (1 / 3) * x^3 - 4 * x + 4 x = max_x)) ∧
    min_x = -4 / 3 ∧ max_x = 1 :=
by sorry

end find_a_and_monotonicity_find_extrema_l720_720796


namespace decimal_to_fraction_l720_720112

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l720_720112


namespace area_of_sector_BAE_l720_720640

-- Given definitions and conditions
variables (A B C D E : Type) [square ℝ A B C D] 
variables (d : dist A B = 2) (r₁ : dist C D = 2) (r₂ : dist B A = 2) 
variables (θ : angle B A E = π / 3)

-- Prove the area of the sector BAE
theorem area_of_sector_BAE (h₁ : dist C D = 2) (h₂ : dist B A = 2) (h₃ : ∠ B A E = π / 3) :
  let r : ℝ := 2 in
  let θ_rad : ℝ := π / 3 in
  (θ_rad / (2 * π)) * π * r^2 = π / 3 :=
by 
  sorry

end area_of_sector_BAE_l720_720640


namespace simplify_expression_l720_720657

theorem simplify_expression (x : ℝ) (h : x = 9) : 
  ((x^9 - 27 * x^6 + 729) / (x^6 - 27) = 730 + 1 / 26) :=
by {
 sorry
}

end simplify_expression_l720_720657


namespace product_of_numbers_l720_720557

theorem product_of_numbers (a b : ℕ) (hcf : ℕ := 12) (lcm : ℕ := 205) (ha : Nat.gcd a b = hcf) (hb : Nat.lcm a b = lcm) : a * b = 2460 := by
  sorry

end product_of_numbers_l720_720557


namespace rhombus_perimeter_to_circle_circumference_ratio_l720_720865

theorem rhombus_perimeter_to_circle_circumference_ratio
  (a r : ℝ)
  (h_rhombus : sorry) -- Representation of ABCD as a rhombus
  (h_angle_ABC : ∠ABC = 60)
  (h_circle_tangent : circle (A, r).is_tangent_to AD)
  (h_center_inside: center_of_circle (A, r) ∈ interior_of_rhombus ABCD)
  (h_tangents_perpendicular : tangent_from_C_is_perpendicular_to_circle (C, ⟨M, N⟩)) :
  (4 * a) / (2 * π * r) = (√3 + √7) / π :=
sorry

end rhombus_perimeter_to_circle_circumference_ratio_l720_720865


namespace points_five_units_away_from_neg_one_l720_720024

theorem points_five_units_away_from_neg_one (x : ℝ) :
  |x + 1| = 5 ↔ x = 4 ∨ x = -6 :=
by
  sorry

end points_five_units_away_from_neg_one_l720_720024


namespace min_xyz_product_l720_720894

open Real

theorem min_xyz_product
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h_sum : x + y + z = 1)
  (h_no_more_than_twice : x ≤ 2 * y ∧ y ≤ 2 * x ∧ y ≤ 2 * z ∧ z ≤ 2 * y) :
  ∃ p : ℝ, (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x + y + z = 1 → x ≤ 2 * y ∧ y ≤ 2 * x ∧ y ≤ 2 * z ∧ z ≤ 2 * y → x * y * z ≥ p) ∧ p = 1 / 32 :=
by
  sorry

end min_xyz_product_l720_720894


namespace tangent_slope_at_one_l720_720026

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x + Real.sqrt x

theorem tangent_slope_at_one :
  (deriv f 1) = 3 / 2 :=
by
  sorry

end tangent_slope_at_one_l720_720026


namespace train_length_problem_l720_720019

noncomputable def train_length (v : ℝ) (t : ℝ) (L : ℝ) : Prop :=
v = 90 / 3.6 ∧ t = 60 ∧ 2 * L = v * t

theorem train_length_problem : train_length 90 1 750 :=
by
  -- Define speed in m/s
  let v_m_s := 90 * (1000 / 3600)
  -- Calculate distance = speed * time
  let distance := 25 * 60
  -- Since distance = 2 * Length
  have h : 2 * 750 = 1500 := sorry
  show train_length 90 1 750
  simp [train_length, h]
  sorry

end train_length_problem_l720_720019


namespace gcd_increase_by_9_l720_720125

theorem gcd_increase_by_9 (m n d : ℕ) (h1 : d = Nat.gcd m n) (h2 : 9 * d = Nat.gcd (m + 6) n) : d = 3 ∨ d = 6 :=
by
  sorry

end gcd_increase_by_9_l720_720125


namespace divisibility_of_exponential_sums_l720_720406

open Nat

theorem divisibility_of_exponential_sums (a b m n : ℕ) (ha : 1 < a) (h_gcd : gcd a b = 1) (h_div : a^n + b^n ∣ a^m + b^m) : n ∣ m := 
sorry

end divisibility_of_exponential_sums_l720_720406


namespace non_j_nice_count_l720_720281

theorem non_j_nice_count :
  let M := (1 to 499).filter (λ m, ¬ (m % 5 = 1 ∨ m % 6 = 1)).length
  in M = 333 := 
by sorry

end non_j_nice_count_l720_720281


namespace peya_time_comparison_l720_720213

variable (V D : ℝ) (hV : 0 < V) (hD : 0 < D)

def planned_time : ℝ := D / V
def increased_speed : ℝ := 1.25 * V
def decreased_speed : ℝ := 0.80 * V

def first_half_distance : ℝ := D / 2
def second_half_distance : ℝ := D / 2

def time_first_half : ℝ := first_half_distance / increased_speed
def time_second_half : ℝ := second_half_distance / decreased_speed

def actual_time : ℝ := time_first_half + time_second_half

theorem peya_time_comparison : actual_time V D = (41 * D) / (40 * V) > (D / V) :=
by {
  unfold actual_time,
  unfold time_first_half time_second_half,
  unfold first_half_distance second_half_distance,
  unfold increased_speed decreased_speed,
  unfold planned_time,
  sorry
}

end peya_time_comparison_l720_720213


namespace print_time_l720_720173

-- Define the conditions
def pages : ℕ := 345
def rate : ℕ := 23
def expected_minutes : ℕ := 15

-- State the problem as a theorem
theorem print_time (pages rate : ℕ) : (pages / rate = 15) :=
by
  sorry

end print_time_l720_720173


namespace regular_polygon_exterior_angle_l720_720201

theorem regular_polygon_exterior_angle (n : ℕ) (h : 1 ≤ n) :
  (360 : ℝ) / (n : ℝ) = 60 → n = 6 :=
by
  intro h1
  sorry

end regular_polygon_exterior_angle_l720_720201


namespace frank_problems_per_type_l720_720647

-- Definitions based on the problem conditions
def bill_problems : ℕ := 20
def ryan_problems : ℕ := 2 * bill_problems
def frank_problems : ℕ := 3 * ryan_problems
def types_of_problems : ℕ := 4

-- The proof statement equivalent to the math problem
theorem frank_problems_per_type : frank_problems / types_of_problems = 30 :=
by 
  -- skipping the proof steps
  sorry

end frank_problems_per_type_l720_720647


namespace hemisphere_surface_area_l720_720538

theorem hemisphere_surface_area (r : ℝ) (h : r = 10) : 
  (4 * Real.pi * r^2) / 2 + (Real.pi * r^2) = 300 * Real.pi := by
  sorry

end hemisphere_surface_area_l720_720538


namespace closest_whole_number_l720_720649

theorem closest_whole_number :
  let x := (10^2001 + 10^2003) / (10^2002 + 10^2002)
  abs ((x : ℝ) - 5) < 1 :=
by 
  sorry

end closest_whole_number_l720_720649


namespace circumcircle_of_DEF_area_l720_720895

noncomputable def circumcircle_area (A B C D E F : Type) [EuclideanGeometry A B C D E F]   
    (AB BC CA BD CE AF : ℝ) (hAB : AB = 5) (hBC : BC = 6) (hCA : CA = 7) 
    (hBD : BD = 7) (hCE : CE = 5) (hAF : AF = 6) : ℝ :=
    let area := (251 / 3) * Real.pi in
    area

-- We now create a theorem to prove that this value for the area is correct given the conditions.
theorem circumcircle_of_DEF_area {A B C D E F : Type} [EuclideanGeometry A B C D E F]
    {AB BC CA BD CE AF : ℝ} (hAB : AB = 5) (hBC : BC = 6) (hCA : CA = 7) 
    (hBD : BD = 7) (hCE : CE = 5) (hAF : AF = 6) :
    circumcircle_area A B C D E F AB BC CA BD CE AF hAB hBC hCA hBD hCE hAF = (251 / 3) * Real.pi :=
by
  sorry

end circumcircle_of_DEF_area_l720_720895


namespace random_event_proof_l720_720192

def is_certain_event (event: Prop) : Prop := ∃ h: event → true, ∃ h': true → event, true
def is_impossible_event (event: Prop) : Prop := event → false
def is_random_event (event: Prop) : Prop := ¬is_certain_event event ∧ ¬is_impossible_event event

def cond1 : Prop := sorry -- Yingying encounters a green light
def cond2 : Prop := sorry -- A non-transparent bag contains one ping-pong ball and two glass balls of the same size, and a ping-pong ball is drawn from it.
def cond3 : Prop := sorry -- You are currently answering question 12 of this test paper.
def cond4 : Prop := sorry -- The highest temperature in our city tomorrow will be 60°C.

theorem random_event_proof : 
  is_random_event cond1 ∧ 
  ¬is_random_event cond2 ∧ 
  ¬is_random_event cond3 ∧ 
  ¬is_random_event cond4 :=
by
  sorry

end random_event_proof_l720_720192


namespace remainder_of_86_l720_720346

theorem remainder_of_86 {m : ℕ} (h1 : m ≠ 1) 
  (h2 : 69 % m = 90 % m) (h3 : 90 % m = 125 % m) : 86 % m = 2 := 
by
  sorry

end remainder_of_86_l720_720346


namespace equilateral_triangle_count_l720_720791

def eleven_vertices := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

def is_equilateral_triangle (triangle : Set ℕ) : Prop :=
  -- Assume a placeholder definition to represent an equilateral triangle with the given vertices
  -- and the eleven-sided regular polygon properties.
  triangle \subset eleven_vertices ∧ triangle.card = 3 ∧ 
  -- Additional placeholder condition to ensure the triangle is equilateral and in the plane of the polygon
  sorry

theorem equilateral_triangle_count : 
  (∑ t in {triangle | is_equilateral_triangle triangle}.to_list, 1) = 88 :=
sorry

end equilateral_triangle_count_l720_720791


namespace problem_equivalent_proof_l720_720425

def sequence_row1 (n : ℕ) : ℤ := 2 * (-2)^(n - 1)
def sequence_row2 (n : ℕ) : ℤ := sequence_row1 n - 1
def sequence_row3 (n : ℕ) : ℤ := (-2)^n - sequence_row2 n

theorem problem_equivalent_proof :
  let a := sequence_row1 7
  let b := sequence_row2 7
  let c := sequence_row3 7
  a - b + c = -254 :=
by
  sorry

end problem_equivalent_proof_l720_720425


namespace smallest_five_digit_number_divisible_by_first_five_primes_l720_720701

theorem smallest_five_digit_number_divisible_by_first_five_primes : 
  ∃ n, (n >= 10000) ∧ (n < 100000) ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_first_five_primes_l720_720701


namespace decimal_to_fraction_l720_720060

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l720_720060


namespace solve_arithmetic_sequence_l720_720856

variables {α : Type*} [linear_ordered_field α]

-- Define the arithmetic sequence
def arithmetic_sequence (a1 d : α) (n : ℕ) : α :=
  a1 + n * d

-- State the condition given in the problem
def condition (a1 d : α) : Prop :=
  arithmetic_sequence a1 d 8 = 1 / 2 * arithmetic_sequence a1 d 11 + 6

-- Theorem to prove a_6 = 12 given the condition
theorem solve_arithmetic_sequence (a1 d : α) (h : condition a1 d) : 
  arithmetic_sequence a1 d 5 = 12 :=
sorry

end solve_arithmetic_sequence_l720_720856


namespace unique_products_count_l720_720643

theorem unique_products_count :
  let A := {1, 3, 5, 7}
  let B := {2, 4, 6, 8}
  ∃ P : Finset ℕ, (∀ a ∈ A, ∀ b ∈ B, a * b ∈ P) ∧ P.card = 15 :=
by
  let A := {1, 3, 5, 7}
  let B := {2, 4, 6, 8}
  let products := {2, 4, 6, 8, 10, 12, 14, 18, 20, 24, 28, 30, 40, 42, 56} -- manually listing the distinct products for clear understanding, although it should be logically derived 
  use {2, 4, 6, 8, 10, 12, 14, 18, 20, 24, 28, 30, 40, 42, 56}
  split
  {
    intros a a_mem b b_mem
    show a * b ∈ products
  }
  {
    show products.card = 15
  }
  sorry

end unique_products_count_l720_720643


namespace num_ways_assign_guests_l720_720610

theorem num_ways_assign_guests (rooms : ℕ) (friends : ℕ) (max_per_room : ℕ) (num_ways : ℕ) : 
  rooms = 6 → friends = 7 → max_per_room = 2 → num_ways = 92820 → 
  ∃ (f : ℕ → ℕ → Prop), (∀ r f, f r f ∧ r ≤ rooms ∧ f ≤ friends → num_ways = 92820) :=
by
  intros h_rooms h_friends h_max_per_room h_num_ways
  sorry

end num_ways_assign_guests_l720_720610


namespace decimal_to_fraction_l720_720057

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l720_720057


namespace number_of_stickers_used_to_decorate_l720_720906

def initial_stickers : ℕ := 20
def bought_stickers : ℕ := 12
def birthday_stickers : ℕ := 20
def given_stickers : ℕ := 5
def remaining_stickers : ℕ := 39

theorem number_of_stickers_used_to_decorate :
  (initial_stickers + bought_stickers + birthday_stickers - given_stickers - remaining_stickers) = 8 :=
by
  -- Proof goes here
  sorry

end number_of_stickers_used_to_decorate_l720_720906


namespace exp_add_l720_720979

-- Definition and conditions
def f (x : ℝ) : ℝ := Real.exp x

-- Theorem statement
theorem exp_add (x y : ℝ) : f (x + y) = f x * f y :=
by sorry

end exp_add_l720_720979


namespace sum_distances_from_point_to_faces_l720_720779

-- Define a regular tetrahedron with edge length 1
structure Tetrahedron :=
  (vertices : Fin 4 → ℝ × ℝ × ℝ)
  (is_regular : ∀ i j, i ≠ j → EuclideanDistance (vertices i) (vertices j) = 1)

-- Define a point P inside the tetrahedron
structure PointInTetrahedron (T : Tetrahedron) :=
  (P : ℝ × ℝ × ℝ)
  (inside_tetrahedron : ∀ plane, plane ∈ facets T → P_inside_plane plane)

-- Define a function to calculate the distance of a point from a plane
def distance_from_plane (P : ℝ × ℝ × ℝ) (plane : Plane ℝ) : ℝ :=
  sorry  -- Skipping the actual calculation of distance for brevity

-- Define a function that sums the distances from P to each face
def sum_distances (T : Tetrahedron) (P : PointInTetrahedron T) : ℝ :=
  ∑ face in facets T, distance_from_plane P.P face

-- The theorem to be proved
theorem sum_distances_from_point_to_faces (T : Tetrahedron) (P : PointInTetrahedron T) :
  sum_distances T P = (√6)/3 :=
by
  -- Import necessary background for distances in Euclidean geometry
  -- Define the facets of the tetrahedron, calculate distances, sum them up
  -- sorry as placeholder for the proof steps
  sorry

end sum_distances_from_point_to_faces_l720_720779


namespace line_equation_l720_720266

theorem line_equation (a b : ℝ) (h_a : a = 2 * b ∨ a = 0)
    (h_l : ∀ P Q : ℝ × ℝ, P = (3, -1) ∧ Q = (a, 0) → line_through P Q) : 
    ∃ (l : ℝ → ℝ → Prop), 
    (∀ x y, l x y ↔ (x + 2 * y - 1 = 0 ∨ x + 3 * y = 0)) :=
begin
  sorry
end

end line_equation_l720_720266


namespace decimal_to_fraction_equivalence_l720_720047

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l720_720047


namespace problem_a_problem_c_problem_d_l720_720503

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end problem_a_problem_c_problem_d_l720_720503


namespace capacity_of_first_bucket_is_3_l720_720152

variable (C : ℝ)

theorem capacity_of_first_bucket_is_3 
  (h1 : 48 / C = 48 / 3 - 4) : 
  C = 3 := 
  sorry

end capacity_of_first_bucket_is_3_l720_720152


namespace decimal_to_fraction_l720_720093

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l720_720093


namespace total_sand_correct_l720_720187

-- Define the conditions as variables and equations:
variables (x : ℕ) -- original days scheduled to complete
variables (total_sand : ℕ) -- total amount of sand in tons

-- Define the conditions in the problem:
def original_daily_amount := 15  -- tons per day as scheduled
def actual_daily_amount := 20  -- tons per day in reality
def days_ahead := 3  -- days finished ahead of schedule

-- Equation representing the planned and actual transportation:
def planned_sand := original_daily_amount * x
def actual_sand := actual_daily_amount * (x - days_ahead)

-- The goal is to prove:
theorem total_sand_correct : planned_sand = actual_sand → total_sand = 180 :=
by
  sorry

end total_sand_correct_l720_720187


namespace sum_of_undefined_values_l720_720576

-- Define the condition where the denominator is zero
def denominator_is_zero (x : ℝ) : Prop := x^2 - 7 * x + 10 = 0

-- Prove the sum of values of x where the expression is undefined
theorem sum_of_undefined_values : (∑ x in {x : ℝ | denominator_is_zero x}.to_finset, x) = 7 :=
by sorry

end sum_of_undefined_values_l720_720576


namespace gcd_divisibility_and_scaling_l720_720897

theorem gcd_divisibility_and_scaling (a b n : ℕ) (c : ℕ) (h₁ : a ≠ 0) (h₂ : c > 0) (d : ℕ := Nat.gcd a b) :
  (n ∣ a ∧ n ∣ b ↔ n ∣ d) ∧ Nat.gcd (a * c) (b * c) = c * d :=
by 
  sorry

end gcd_divisibility_and_scaling_l720_720897


namespace problem1_problem2_problem3_problem4_problem5_problem6_l720_720519

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end problem1_problem2_problem3_problem4_problem5_problem6_l720_720519


namespace floor_10a_equals_6_l720_720290

theorem floor_10a_equals_6 
  (a : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < 1)
  (h3 : ∑ k in (finset.range 29).map (λ k, k + 1), floor (a + k / 30 : ℝ) = 18) : 
  floor (10 * a) = 6 := 
  sorry

end floor_10a_equals_6_l720_720290


namespace sampling_interval_is_9_l720_720150

-- Conditions
def books_per_hour : ℕ := 362
def sampled_books_per_hour : ℕ := 40

-- Claim to prove
theorem sampling_interval_is_9 : (360 / sampled_books_per_hour = 9) := by
  sorry

end sampling_interval_is_9_l720_720150


namespace Sam_drove_same_rate_as_Marguerite_l720_720910

theorem Sam_drove_same_rate_as_Marguerite 
  (d_Marguerite : ℝ) (t_Marguerite : ℝ) (t_Sam : ℝ) 
  (h_Marguerite : d_Marguerite = 100) (h_t_Marguerite : t_Marguerite = 2.4) 
  (h_t_Sam : t_Sam = 3) : 
  (d_Marguerite / t_Marguerite) * t_Sam = 125 := 
by 
  rw [h_Marguerite, h_t_Marguerite, h_t_Sam]
  norm_num
  sorry

end Sam_drove_same_rate_as_Marguerite_l720_720910


namespace pipeA_fill_time_l720_720921

variable (t : ℕ) -- t is the time in minutes for Pipe A to fill the tank

-- Conditions
def pipeA_duration (t : ℕ) : Prop :=
  t > 0

def pipeB_duration (t : ℕ) : Prop :=
  t / 3 > 0

def combined_rate (t : ℕ) : Prop :=
  3 * (1 / (4 / t)) = t

-- Problem
theorem pipeA_fill_time (h1 : pipeA_duration t) (h2 : pipeB_duration t) (h3 : combined_rate t) : t = 12 :=
sorry

end pipeA_fill_time_l720_720921


namespace decimal_to_fraction_l720_720105

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l720_720105


namespace nonempty_even_subsets_count_l720_720335

theorem nonempty_even_subsets_count :
  let even_subset := {2, 4, 6, 8}
  ∃ n, n = 15 ∧ n = (2 ^ (even_subset.to_finset.card)) - 1 :=
by
  let even_subset : set ℕ := {2, 4, 6, 8}
  have h1 : even_subset.to_finset.card = 4 := by simp
  let total_subsets := 2 ^ even_subset.to_finset.card
  have h2 : total_subsets - 1 = 15 := by simp
  use 15
  simp [h2]
  sorry

end nonempty_even_subsets_count_l720_720335


namespace symmetric_graph_implies_range_of_a_l720_720801

theorem symmetric_graph_implies_range_of_a 
    (a : ℝ)
    (f g : ℝ → ℝ)
    (h_f : ∀ x, f x = x^2 + exp x - if x < 0 then 1/2 else 0)
    (h_g : ∀ x, g x = x^2 + log (x + a)) :
    (∃ x₀ < 0, f x₀ = g (-x₀)) → a < sqrt_real e :=
by 
  sorry

end symmetric_graph_implies_range_of_a_l720_720801


namespace total_percentage_reduction_l720_720532

theorem total_percentage_reduction (P : ℝ) (h : 0 < P) : 
  let first_reduction := 0.75 * P
  let second_reduction := 0.5 * first_reduction
  let final_price := first_reduction - second_reduction
  let reduction := P - final_price in
  (reduction / P) * 100 = 62.5 := 
by 
  sorry

end total_percentage_reduction_l720_720532


namespace max_value_sqrt_sum_l720_720837

theorem max_value_sqrt_sum (a b c : ℝ) (h : a + b + c = 1) : 
  sqrt (3 * a + 1) + sqrt (3 * b + 1) + sqrt (3 * c + 1) ≤ 3 * Real.sqrt 2 :=
sorry

end max_value_sqrt_sum_l720_720837


namespace smallest_five_digit_divisible_by_primes_l720_720713

theorem smallest_five_digit_divisible_by_primes : 
  let primes := [2, 3, 5, 7, 11] in
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11))) in
  let five_digit_threshold := 10000 in
  ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
by
  let primes := [2, 3, 5, 7, 11]
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))
  have lcm_2310 : lcm_primes = 2310 := sorry
  let five_digit_threshold := 10000
  have exists_n : ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
    sorry
  exists_intro 5
  have 5_condition : 5 > 0 := sorry
  have 2310_5_condition : 2310 * 5 >= five_digit_threshold := sorry
  have answer : 2310 * 5 = 11550 := sorry
  exact  ⟨5, 5_condition, 2310_5_condition, answer⟩
  exact ⟨5, 5 > 0, 2310 * 5 ≥ 10000, 2310 * 5 = 11550⟩
  sorry

end smallest_five_digit_divisible_by_primes_l720_720713


namespace inverse_function_l720_720566

def f (x : ℝ) : ℝ := 2 - 3 * x

def g (x : ℝ) : ℝ := (2 - x) / 3

theorem inverse_function (x : ℝ) : f (g x) = x :=
by
  calc
    f (g x) = 2 - 3 * (g x) : by rw f
        ... = 2 - 3 * ((2 - x) / 3) : by rw g
        ... = x : by sorry

end inverse_function_l720_720566


namespace trig_identity_proof_l720_720302

theorem trig_identity_proof (θ φ : ℝ) 
    (h : (cos θ)^6 / (cos φ)^2 + (sin θ)^6 / (sin φ)^2 = 1) :
    (sin φ)^6 / (sin θ)^2 + (cos φ)^6 / (cos θ)^2 = (1 + (cos (2 * φ))^2) / 2 :=
sorry

end trig_identity_proof_l720_720302


namespace count_values_g100_zero_l720_720888

noncomputable def g0 (x: ℝ) : ℝ :=
if x < -150 then x + 300
else if x < 150 then -x
else x - 300

noncomputable def gn : ℕ → ℝ → ℝ
| 0     x := g0 x
| (n+1) x := abs (gn n x) - 1

theorem count_values_g100_zero : 
  (finset.filter (λ x, gn 100 x = 0) (finset.range 1000)).card = 603 :=
sorry

end count_values_g100_zero_l720_720888


namespace decimal_to_fraction_l720_720085

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l720_720085


namespace measure_of_angle_A_max_area_triangle_l720_720309

namespace TriangleProblems

section AngleA

variables {a b c : ℝ} (A B C : ℝ)
variable (h : (2 * c - b) * Real.cos A = a * Real.cos B)

theorem measure_of_angle_A (h : (2 * c - b) * Real.cos A = a * Real.cos B) :
  A = π / 3 :=
sorry

end AngleA

section MaxArea

variables {a b c : ℝ} (A B C : ℝ)
variable (h : (2 * c - b) * Real.cos A = a * Real.cos B)
variable (ha : a = 2)

theorem max_area_triangle (h : (2 * c - b) * Real.cos A = a * Real.cos B) (ha : a = 2) :
  ∃ b c, ∀ A B C, b = c → (1 / 2) * b * c * Real.sin A ≤ sqrt 3 :=
sorry

end MaxArea
end TriangleProblems

end measure_of_angle_A_max_area_triangle_l720_720309


namespace woody_needs_14_weeks_l720_720584

noncomputable def weeks_needed : ℕ :=
  let console_cost := 282
  let game_cost := 75
  let tax_rate := 0.10
  let initial_amount := 42
  let weekly_allowance := 24
  let total_game_cost := game_cost * (1 + tax_rate)
  let total_cost := console_cost + total_game_cost
  let amount_needed := total_cost - initial_amount
  let weeks := (amount_needed / weekly_allowance).ceil
  weeks

theorem woody_needs_14_weeks : weeks_needed = 14 :=
by
  sorry

end woody_needs_14_weeks_l720_720584


namespace red_shoe_probability_l720_720593

variable (red_shoes : ℕ) (green_shoes : ℕ)

def total_shoes (red_shoes: ℕ) (green_shoes: ℕ) : ℕ := red_shoes + green_shoes
def first_red_prob (red_shoes: ℕ) (total_shoes: ℕ) : ℚ := red_shoes / total_shoes
def second_red_prob (red_shoes: ℕ) (total_shoes: ℕ) : ℚ := (red_shoes - 1) / (total_shoes - 1)

theorem red_shoe_probability (h1 : red_shoes = 5) (h2 : green_shoes = 4) :
  first_red_prob red_shoes (total_shoes red_shoes green_shoes) * second_red_prob red_shoes (total_shoes red_shoes green_shoes) = 5 / 18 := by
  let total := total_shoes red_shoes green_shoes
  have h_total : total = 9 := by
    rw [h1, h2]
    norm_num
  sorry

end red_shoe_probability_l720_720593


namespace range_of_m_l720_720412

def f (x : ℝ) : ℝ := x ^ 3 + x

theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, 0 < θ ∧ θ < π / 2 → f (m * sin θ) + f (1 - m) > 0) → m ≤ 1 :=
by
  sorry

end range_of_m_l720_720412


namespace total_weight_is_correct_l720_720552

noncomputable def A (B : ℝ) : ℝ := 12 + (1/2) * B
noncomputable def B (C : ℝ) : ℝ := 8 + (1/3) * C
noncomputable def C (A : ℝ) : ℝ := 20 + 2 * A
noncomputable def NewWeightB (A B : ℝ) : ℝ := B + 0.15 * A
noncomputable def NewWeightA (A C : ℝ) : ℝ := A - 0.10 * C

theorem total_weight_is_correct (B C : ℝ) (h1 : A B = (C - 20) / 2)
  (h2 : B = 8 + (1/3) * C) 
  (h3 : C = 20 + 2 * A B) 
  (h4 : NewWeightB (A B) B = 38.35) 
  (h5 : NewWeightA (A B) C = 21.2) :
  NewWeightA (A B) C + NewWeightB (A B) B + C = 139.55 :=
sorry

end total_weight_is_correct_l720_720552


namespace smallest_five_digit_number_divisible_by_primes_l720_720731

theorem smallest_five_digit_number_divisible_by_primes : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
begin
  sorry
end

end smallest_five_digit_number_divisible_by_primes_l720_720731


namespace sin_2α_eq_neg_four_fifths_tan_pi_over_3_plus_α_eq_five_root_three_minus_eight_l720_720305

variable (α : ℝ)
variable (h1 : sin α = (√5) / 5)
variable (h2 : α ∈ Ioo (π / 2) π)

theorem sin_2α_eq_neg_four_fifths : sin (2 * α) = -4 / 5 := sorry

theorem tan_pi_over_3_plus_α_eq_five_root_three_minus_eight : 
  tan ((π / 3) + α) = 5 * (√3) - 8 := sorry

end sin_2α_eq_neg_four_fifths_tan_pi_over_3_plus_α_eq_five_root_three_minus_eight_l720_720305


namespace sum_of_first_four_terms_geometric_sequence_is_15_l720_720758

variable {a : ℕ → ℕ} -- Geometric sequence {a_n}

-- Definitions based on the given conditions
def geometric_sequence (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

def arithmetic_sequence (x y z : ℕ) : Prop := 
  y - x = z - y

def a1 : ℕ := 1

-- Sum of the first n terms of a sequence
def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in finset.range (n+1), a i

-- Problem statement: Proof that S_4 = 15 given the conditions
theorem sum_of_first_four_terms_geometric_sequence_is_15 (r a2 a3 : ℕ) :
  (geometric_sequence a r) →
  arithmetic_sequence (4 * a1) a2 a3 →
  a 0 = a1 →
  a 1 = a2 →
  a 2 = a3 →
  S a 3 = 15 :=
by
  sorry

end sum_of_first_four_terms_geometric_sequence_is_15_l720_720758


namespace isosceles_if_interior_angles_equal_l720_720005

-- Definition for a triangle
structure Triangle :=
  (A B C : Type)

-- Defining isosceles triangle condition
def is_isosceles (T : Triangle) :=
  ∃ a b c : ℝ, (a = b) ∨ (b = c) ∨ (a = c)

-- Defining the angle equality condition
def interior_angles_equal (T : Triangle) :=
  ∃ a b c : ℝ, (a = b) ∨ (b = c) ∨ (a = c)

-- Main theorem stating the contrapositive
theorem isosceles_if_interior_angles_equal (T : Triangle) : 
  interior_angles_equal T → is_isosceles T :=
by sorry

end isosceles_if_interior_angles_equal_l720_720005


namespace not_identity_for_all_a_identity_for_specific_a_l720_720384

-- First problem: Prove that the equality is not an identity for all a
theorem not_identity_for_all_a (a : ℤ) : ¬ (∀ a, (a^4 - 1)^6 = (a^6 - 1)^4) := sorry

-- Second problem: Prove that the equality is an identity when a in {-1, 0, 1}
theorem identity_for_specific_a (a : ℤ) (h : a ∈ ({-1, 0, 1} : set ℤ)) : (a^4 - 1)^6 = (a^6 - 1)^4 := sorry

end not_identity_for_all_a_identity_for_specific_a_l720_720384


namespace problem1_problem2_problem3_problem4_problem5_problem6_l720_720514

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end problem1_problem2_problem3_problem4_problem5_problem6_l720_720514


namespace power_comparison_l720_720127

theorem power_comparison : (9^20 : ℝ) < (9999^10 : ℝ) :=
sorry

end power_comparison_l720_720127


namespace smallest_five_digit_number_divisible_by_primes_l720_720735

theorem smallest_five_digit_number_divisible_by_primes : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
begin
  sorry
end

end smallest_five_digit_number_divisible_by_primes_l720_720735


namespace Petya_running_time_l720_720222

theorem Petya_running_time (D V : ℝ) 
  (hV_pos : 0 < V) (hD_pos : 0 < D):
  let T := D / V in
  let V1 := 1.25 * V in
  let V2 := 0.8 * V in
  let T1 := (D / 2) / V1 in
  let T2 := (D / 2) / V2 in
  let T_actual := T1 + T2 in
  T_actual > T :=
by
  let T := D / V
  let V1 := 1.25 * V
  let V2 := 0.8 * V
  let T1 := (D / 2) / V1
  let T2 := (D / 2) / V2
  let T_actual := T1 + T2
  have : T_actual = (2 * D) / (5 * V) + (5 * D) / (8 * V) := by 
  sorry
  have : T_actual = 41 * D / (40 * V) := by 
  sorry
  have : T = D / V := by 
  sorry
  show 41 * D / (40 * V) > D / V := by 
  sorry

end Petya_running_time_l720_720222


namespace max_distance_from_M_to_C1_l720_720374

noncomputable def parametric_equations_C1 (t : ℝ) : ℝ × ℝ :=
  (1 + 2 * t, -2 + t)

noncomputable def polar_equation_C2 (ρ θ : ℝ) : Prop :=
  ρ^2 = 4 / (1 + 3 * (Real.sin θ)^2)

noncomputable def scaling_transformation (x y : ℝ) : ℝ × ℝ :=
  (2 * x, y)

def ordinary_eq_C1 (x y : ℝ) : Prop :=
  x - 2 * y - 5 = 0

def rectangular_eq_C2 (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

def rectangular_eq_C3 (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 = 1

noncomputable def distance_from_M_to_C1 (M : ℝ × ℝ) : ℝ :=
  let (x, y) := M in abs((2 * y) - (4 * x) + 5) / Real.sqrt(5)

theorem max_distance_from_M_to_C1 :
  ∃ M : ℝ × ℝ, (scaling_transformation M.1 M.2) ∈ rectangular_eq_C3
  ∧ distance_from_M_to_C1 M = 2 + Real.sqrt(5) :=
sorry

end max_distance_from_M_to_C1_l720_720374


namespace mushroom_children_count_l720_720975

variables {n : ℕ} {A V S R : ℕ}

-- Conditions:
def condition1 (n : ℕ) (A : ℕ) (V : ℕ) : Prop :=
  ∀ (k : ℕ), k < n → V + A / 2 = k

def condition2 (S : ℕ) (A : ℕ) (R : ℕ) (V : ℕ) : Prop :=
  S + A = R + V + A

-- Proof statement
theorem mushroom_children_count (n : ℕ) (A : ℕ) (V : ℕ) (S : ℕ) (R : ℕ) :
  condition1 n A V → condition2 S A R V → n = 6 :=
by
  intros hcondition1 hcondition2
  sorry

end mushroom_children_count_l720_720975


namespace probability_of_stones_in_bucket_l720_720673

def P (x y : ℕ) : ℚ := ⟨binom y ((y - x) / 2), 2 ^ y⟩

theorem probability_of_stones_in_bucket :
  P 1337 2017 = (binomial 2017 340) / 2 ^ 2017 := by
  sorry

end probability_of_stones_in_bucket_l720_720673


namespace roots_expression_eval_l720_720407

theorem roots_expression_eval (p q r : ℝ) 
  (h1 : p + q + r = 2)
  (h2 : p * q + q * r + r * p = -1)
  (h3 : p * q * r = -2)
  (hp : p^3 - 2 * p^2 - p + 2 = 0)
  (hq : q^3 - 2 * q^2 - q + 2 = 0)
  (hr : r^3 - 2 * r^2 - r + 2 = 0) :
  p * (q - r)^2 + q * (r - p)^2 + r * (p - q)^2 = 16 :=
sorry

end roots_expression_eval_l720_720407


namespace z_rate_per_rupee_of_x_l720_720183

-- Given conditions as definitions in Lean 4
def x_share := 1 -- x gets Rs. 1 for this proof
def y_rate_per_rupee_of_x := 0.45
def y_share := 27
def total_amount := 105

-- The statement to prove
theorem z_rate_per_rupee_of_x :
  (105 - (1 * 60) - 27) / 60 = 0.30 :=
by
  sorry

end z_rate_per_rupee_of_x_l720_720183


namespace decimal_to_fraction_equivalence_l720_720052

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l720_720052


namespace problem_a_problem_c_problem_d_l720_720501

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end problem_a_problem_c_problem_d_l720_720501


namespace inequality_a_inequality_c_inequality_d_l720_720493

variable {a b : ℝ}

axiom (h : a + b > 0)

theorem inequality_a : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_c : a^21 + b^21 > 0 :=
sorry

theorem inequality_d : (a + 2) * (b + 2) > a * b :=
sorry

end inequality_a_inequality_c_inequality_d_l720_720493


namespace positive_integer_iff_positive_real_l720_720284

theorem positive_integer_iff_positive_real (x : ℝ) (hx : x ≠ 0) :
  (∃ n : ℕ, n > 0 ∧ abs ((x - 2 * abs x) * abs x) / x = n) ↔ x > 0 :=
by
  sorry

end positive_integer_iff_positive_real_l720_720284


namespace find_f_3_l720_720401

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := x^2

theorem find_f_3
(h₀ : ∀ x : ℝ, x ≠ 0 → (f x) - 3 * (f (1 / x)) + g x = 3^x)
: f 3 = -9 - (real.cbrt 3) / 6 + 1 / 54 := 
sorry

end find_f_3_l720_720401


namespace trapezoids_in_rhombus_l720_720618

theorem trapezoids_in_rhombus (n : ℕ) (h : 0 < n) :
  let sides := n 
  let angle := 60 
  let triangles := 2 * n^2 
  s(n) = (n * (n^2 - 1) * (2 * n + 1)) / 3 := 
sorry

end trapezoids_in_rhombus_l720_720618


namespace length_TU_and_perimeter_STU_l720_720145

-- Define the lengths of the sides of the triangles
def PQ := 10
def QR := 12
def ST := 5

-- Define similarity condition
def triangleSimilar (a₁ a₂ a₃ b₁ b₂ b₃: ℝ) : Prop :=
  b₁ / a₁ = b₂ / a₂ ∧ b₁ / a₁ = b₃ / a₃ ∧ b₂ / a₂ = b₃ / a₃

-- Define the similarity of triangles PQR and STU
def trianglesSimilarPQR_STU :=
  triangleSimilar 10 12 (√(10^2 + 12^2)) 5 (5 * 12 / 10) ?

-- Define the target proof problems
theorem length_TU_and_perimeter_STU : 
  (∃ TU SU : ℝ, TU = 6 ∧ SU = 6 ∧ 5 + TU + SU = 17) :=
sorry

end length_TU_and_perimeter_STU_l720_720145


namespace distinct_pairs_disjoint_subsets_l720_720334

theorem distinct_pairs_disjoint_subsets (n : ℕ) : 
  ∃ k, k = (3^n + 1) / 2 := 
sorry

end distinct_pairs_disjoint_subsets_l720_720334


namespace complex_problem_l720_720165

theorem complex_problem (z : ℂ) (h : (i * z + z) = 2) : z = 1 - i :=
sorry

end complex_problem_l720_720165


namespace multiplication_trick_l720_720594

theorem multiplication_trick (a b c : ℕ) (h : b + c = 10) :
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c :=
by
  sorry

end multiplication_trick_l720_720594


namespace half_of_villagers_are_liars_l720_720630

noncomputable def proportion_of_liars (villagers : List Bool) : ℚ :=
  let n := villagers.length
  let liar_count := villagers.count (fun v => v = false)
  ↑liar_count / n

theorem half_of_villagers_are_liars (villagers : List Bool) (h1 : ∀ v ∈ villagers, v = true ∨ v = false)
  (h2 : villagers ≠ []) (h3 : villagers.head = villagers.last) :
  proportion_of_liars villagers = 1 / 2 :=
by
  sorry

end half_of_villagers_are_liars_l720_720630


namespace log_base_comparison_l720_720340

theorem log_base_comparison (m n : ℝ) (h : log m 9 < log n 9 ∧ log n 9 < 0) : 0 < m ∧ m < n ∧ n < 1 :=
sorry

end log_base_comparison_l720_720340


namespace possible_values_of_M_l720_720773

theorem possible_values_of_M (a b : ℚ) (h : a * b ≠ 0) : 
  let M := (2 * |a| / a) + (3 * b / |b|) in M = 1 ∨ M = -1 ∨ M = 5 ∨ M = -5 :=
by
  sorry

end possible_values_of_M_l720_720773


namespace min_cos_sum_l720_720808

theorem min_cos_sum (x y z : ℝ) (hx : 0 ≤ x) (hx' : x ≤ π) (hy : 0 ≤ y) (hy' : y ≤ π) (hz : 0 ≤ z) (hz' : z ≤ π) :
  ∃ x y z, (A = cos (x - y) + cos (y - z) + cos (z - x)) ∧  A = -1

end min_cos_sum_l720_720808


namespace cost_per_meter_fencing_is_25_paise_l720_720994

-- Conditions
def sides_in_ratio (a b : ℕ) := a = 3 * b ∧ b = 4 * c
def area_field (a b : ℕ) (A : ℕ) := A = a * b
def total_cost (cost : ℕ) := cost = 105
def cost_per_meter_in_paise (cost_per_meter : ℕ) (total_cost perimeter : ℕ) := cost_per_meter = (total_cost * 100) / perimeter

-- Problem: Given conditions, prove cost per meter in paise is 25
theorem cost_per_meter_fencing_is_25_paise
  (a b c perimeter : ℕ)
  (h1 : sides_in_ratio a b)
  (h2 : area_field a b 10800)
  (h3 : total_cost 105)
  (h4 : perimeter = 2 * (a + b)) :
  cost_per_meter_in_paise (25 : ℕ) 105 perimeter :=
sorry

end cost_per_meter_fencing_is_25_paise_l720_720994


namespace distance_points_l720_720119

/-- The distance between two points (-3, 5) and (4, -7) -/
theorem distance_points :
  let p1 : ℝ × ℝ := (-3, 5)
  let p2 : ℝ × ℝ := (4, -7)
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2) = Real.sqrt 193 :=
by
  sorry

end distance_points_l720_720119


namespace range_of_m_for_no_extreme_points_l720_720354

theorem range_of_m_for_no_extreme_points (m : ℝ) :
  (∀ x : ℝ, ∀ y : ℝ, (f'(x) := 3 * x^2 + 2 * x + m) ∧ (f'(y) := 3 * y^2 + 2 * y + m) ∧ (f'(x) = f'(y) → x = y)) →
  (m ≥ 1 / 3) :=
by
  sorry

end range_of_m_for_no_extreme_points_l720_720354


namespace Petya_running_time_l720_720220

theorem Petya_running_time (D V : ℝ) 
  (hV_pos : 0 < V) (hD_pos : 0 < D):
  let T := D / V in
  let V1 := 1.25 * V in
  let V2 := 0.8 * V in
  let T1 := (D / 2) / V1 in
  let T2 := (D / 2) / V2 in
  let T_actual := T1 + T2 in
  T_actual > T :=
by
  let T := D / V
  let V1 := 1.25 * V
  let V2 := 0.8 * V
  let T1 := (D / 2) / V1
  let T2 := (D / 2) / V2
  let T_actual := T1 + T2
  have : T_actual = (2 * D) / (5 * V) + (5 * D) / (8 * V) := by 
  sorry
  have : T_actual = 41 * D / (40 * V) := by 
  sorry
  have : T = D / V := by 
  sorry
  show 41 * D / (40 * V) > D / V := by 
  sorry

end Petya_running_time_l720_720220


namespace part_a_part_c_part_d_l720_720510

-- Define the variables
variables {a b : ℝ}

-- Define the conditions and statements
def cond := a + b > 0

theorem part_a (h : cond) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem part_c (h : cond) : a^21 + b^21 > 0 :=
sorry

theorem part_d (h : cond) : (a + 2) * (b + 2) > a * b :=
sorry

end part_a_part_c_part_d_l720_720510


namespace inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720486

variable {a b : ℝ}

theorem inequality_a (hab : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_b_not_true (hab : a + b > 0) : ¬(a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

theorem inequality_c (hab : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem inequality_d (hab : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

theorem inequality_e_not_true (hab : a + b > 0) : ¬((a − 3) * (b − 3) < a * b) :=
sorry

theorem inequality_f_not_true (hab : a + b > 0) : ¬((a + 2) * (b + 3) > a * b + 5) :=
sorry

end inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720486


namespace triangle_circle_ineq_l720_720156

theorem triangle_circle_ineq (A B C P Q M : Point) (hM_mid : M = midpoint B C)
  (h_circle : Circle M A intersects AB at P ∧ Circle M A intersects AC at Q)
  (h_angle : ∠ BAC = 60°) :
  AP + AQ + PQ < AB + AC + 1/2 * BC := 
by
  sorry

end triangle_circle_ineq_l720_720156


namespace complex_number_quadrant_l720_720341

theorem complex_number_quadrant (a : ℝ) : 
  let z := (a^2 - 4 * a + 5) - 6 * Complex.i in 
  (Complex.re z > 0) ∧ (Complex.im z < 0) := 
by
  sorry

end complex_number_quadrant_l720_720341


namespace smallest_five_digit_number_divisible_l720_720693

def smallest_prime_divisible (n: ℕ) : Prop :=
  ∃ k: ℕ, n = 2310 * k ∧ 10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_number_divisible :
  ∃ (n: ℕ), smallest_prime_divisible n ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_l720_720693


namespace smallest_n_satisfying_congruence_l720_720573

def smallest_four_digit_integer (n : ℕ) : Prop :=
  ∃ k : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n = 3 + 23 * k

theorem smallest_n_satisfying_congruence : ∃ n, smallest_four_digit_integer n ∧ (75 * n) % 345 = 225 % 345 :=
by
  use 1015
  split
  · unfold smallest_four_digit_integer
    use 44
    split
    · exact nat.le_refl 1000
    split
    · exact nat.lt_succ_self 10000
    rfl
  rfl

end smallest_n_satisfying_congruence_l720_720573


namespace values_of_m_and_n_l720_720775

theorem values_of_m_and_n (m n : ℕ) (h_cond1 : 2 * m + 3 = 5 * n - 2) (h_cond2 : 5 * n - 2 < 15) : m = 5 ∧ n = 3 :=
by
  sorry

end values_of_m_and_n_l720_720775


namespace initial_mean_of_observations_l720_720985

theorem initial_mean_of_observations :
  ∃ M : ℝ, 
  (Sum_initial = 50 * M ∧
  Sum_corrected = Sum_initial + 25 ∧
  Sum_corrected = 50 * 36.5) → 
  M = 36 :=
begin
  sorry
end

end initial_mean_of_observations_l720_720985


namespace ratio_of_sums_l720_720025

noncomputable def sum_of_squares (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def square_of_sum (n : ℕ) : ℚ :=
  ((n * (n + 1)) / 2) ^ 2

theorem ratio_of_sums (n : ℕ) (h : n = 25) :
  sum_of_squares n / square_of_sum n = 1 / 19 :=
by
  have hn : n = 25 := h
  rw [hn]
  dsimp [sum_of_squares, square_of_sum]
  have : (25 * (25 + 1) * (2 * 25 + 1)) / 6 = 5525 := by norm_num
  have : ((25 * (25 + 1)) / 2) ^ 2 = 105625 := by norm_num
  norm_num
  sorry

end ratio_of_sums_l720_720025


namespace find_students_that_got_As_l720_720252

variables (Emily Frank Grace Harry : Prop)

theorem find_students_that_got_As
  (cond1 : Emily → Frank)
  (cond2 : Frank → Grace)
  (cond3 : Grace → Harry)
  (cond4 : Harry → ¬ Emily)
  (three_A_students : ¬ (Emily ∧ Frank ∧ Grace ∧ Harry) ∧
                      (Emily ∧ Frank ∧ Grace ∧ ¬ Harry ∨
                       Emily ∧ Frank ∧ ¬ Grace ∧ Harry ∨
                       Emily ∧ ¬ Frank ∧ Grace ∧ Harry ∨
                       ¬ Emily ∧ Frank ∧ Grace ∧ Harry)) :
  (¬ Emily ∧ Frank ∧ Grace ∧ Harry) :=
by {
  sorry
}

end find_students_that_got_As_l720_720252


namespace smallest_five_digit_number_divisible_l720_720688

def smallest_prime_divisible (n: ℕ) : Prop :=
  ∃ k: ℕ, n = 2310 * k ∧ 10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_number_divisible :
  ∃ (n: ℕ), smallest_prime_divisible n ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_l720_720688


namespace decimal_to_fraction_l720_720079

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l720_720079


namespace volume_of_tetrahedron_equiv_l720_720741

noncomputable def volume_tetrahedron (D1 D2 D3 : ℝ) 
  (h1 : D1 = 24) (h2 : D3 = 20) (h3 : D2 = 16) : ℝ :=
  30 * Real.sqrt 6

theorem volume_of_tetrahedron_equiv (D1 D2 D3 : ℝ) 
  (h1 : D1 = 24) (h2 : D3 = 20) (h3 : D2 = 16) :
  volume_tetrahedron D1 D2 D3 h1 h2 h3 = 30 * Real.sqrt 6 :=
  sorry

end volume_of_tetrahedron_equiv_l720_720741


namespace a_n_general_formula_sum_a_n_gt_l720_720296

noncomputable theory

-- Given condition definitions
def a : ℕ → ℚ
| 0 => 8 / 9
| (n + 1) => a n + 8 * (n + 1) / ((2*n + 1)^2 * (2*n + 3)^2)

-- Prove the general formula using induction
theorem a_n_general_formula (n : ℕ) : a n = 1 - 1 / (2*n + 1)^2 :=
by
  induction n with
  | zero =>
    show a 0 = 1 - 1 / (2*0 + 1)^2
    -- Base case proof to be filled
    sorry
  | succ n ih =>
    show a (n + 1) = 1 - 1 / (2*(n + 1) + 1)^2
    -- Induction step proof to be filled
    sorry

-- Prove sum of sequence is greater than n - 1/4
theorem sum_a_n_gt (n : ℕ) : (∑ i in Finset.range (n + 1), a i) > n - 1/4 :=
by
  -- Proof to be filled
  sorry

end a_n_general_formula_sum_a_n_gt_l720_720296


namespace math_problem_l720_720473

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end math_problem_l720_720473


namespace inequality_a_inequality_c_inequality_d_l720_720490

variable {a b : ℝ}

axiom (h : a + b > 0)

theorem inequality_a : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_c : a^21 + b^21 > 0 :=
sorry

theorem inequality_d : (a + 2) * (b + 2) > a * b :=
sorry

end inequality_a_inequality_c_inequality_d_l720_720490


namespace trajectory_and_slope_range_l720_720788
-- Import necessary libraries

-- Define the given quantities and conditions
def circle_M_P (x y : ℝ) : Prop := (x + Real.sqrt 5)^2 + y^2 = 36
def fixed_point_N : ℝ × ℝ := (Real.sqrt 5, 0)
def line_l (k x : ℝ) := k * (x - 2)

-- Define the trajectory curve equation
def trajectory_C (x y : ℝ) : Prop := (x ^ 2) / 9 + (y ^ 2) / 4 = 1

-- Define the proof problem
theorem trajectory_and_slope_range :
  (∀ x y, circle_M_P x y → ∃ a b, 
    trajectory_C a b ∧ 
    (∃! k, ∀ x1 y1 x2 y2, line_l k x1 = y1 ∧ line_l k x2 = y2 ∧ trajectory_C x1 y1 ∧ trajectory_C x2 y2 → 
    (a * x1 + b * y1 ≤ -1) → (- (4 * Real.sqrt 130 / 65) < k ∧ k < 4 * Real.sqrt 130 / 65))) := sorry

end trajectory_and_slope_range_l720_720788


namespace construct_equilateral_triangle_is_equilateral_l720_720429

noncomputable def construct_equilateral_triangle (A B : Point) (d r : ℝ) (h : d ≠ r) : Triangle :=
  let D := point_on_arc A d r
  let C := point_on_arc B d r
  let E := intersection_of_arcs D C
  triangle A B E

theorem construct_equilateral_triangle_is_equilateral (A B : Point) (d r : ℝ) (h : d ≠ r)
  (distAB : dist A B = d) :
  let T := construct_equilateral_triangle A B d r h in equilateral_triangle T := 
sorry

end construct_equilateral_triangle_is_equilateral_l720_720429


namespace average_value_is_12_l720_720280

open Nat

def average_value_sum_permutations : ℚ :=
  ∑ b in perm.univ, (|b 0 - b 1| + |b 2 - b 3| + |b 4 - b 5| + |b 6 - b 7|) / perm.univ.card

theorem average_value_is_12 :
  let p := 12
  let q := 1
  p + q = 13 :=
by
  sorry

end average_value_is_12_l720_720280


namespace line_in_slope_intercept_form_l720_720612

def vec1 : ℝ × ℝ := (3, -7)
def point : ℝ × ℝ := (-2, 4)
def line_eq (x y : ℝ) : Prop := vec1.1 * (x - point.1) + vec1.2 * (y - point.2) = 0

theorem line_in_slope_intercept_form (x y : ℝ) : line_eq x y → y = (3 / 7) * x - (34 / 7) :=
by
  sorry

end line_in_slope_intercept_form_l720_720612


namespace A_eq_three_l720_720445

theorem A_eq_three (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (A : ℤ)
  (h : A = ((a + 1 : ℕ) / (b : ℕ)) + (b : ℕ) / (a : ℕ)) : A = 3 := by
  sorry

end A_eq_three_l720_720445


namespace parallel_MN_PQ_l720_720942

variables {A B C I Z M N P Q : Point}
variables {BI IZ PQ MN : Line}

-- Conditions
variables (triangle_ABC : is_triangle A B C)
variables (incircle_center_I : is_incircle_center I A B C)
variables (angle_bisector_IZ : is_angle_bisector Z B I triangle_ABC)
variables (perpendicular_PQ_IZ : is_perpendicular PQ IZ)
variables (perpendicular_IZ_BI : is_perpendicular IZ BI)
variables (perpendicular_MN_BI : is_perpendicular MN BI)

theorem parallel_MN_PQ 
  (h1: is_angle_bisector Z B I triangle_ABC)
  (h2: is_perpendicular PQ IZ)
  (h3: is_perpendicular IZ BI)
  (h4: is_perpendicular MN BI) : 
  is_parallel MN PQ := 
sorry

end parallel_MN_PQ_l720_720942


namespace find_x_given_log_eq_a_l720_720772

theorem find_x_given_log_eq_a (a : ℤ) (ha : a > 1) (x : ℝ) (hx : log x a = a) :
  x = 10 ^ (log 10 a / a) :=
sorry

end find_x_given_log_eq_a_l720_720772


namespace custom_op_evaluation_l720_720344

def custom_op (x y : ℕ) : ℕ := x * y + x - y

theorem custom_op_evaluation : (custom_op 7 4) - (custom_op 4 7) = 6 := by
  sorry

end custom_op_evaluation_l720_720344


namespace monotonicity_of_f_extreme_points_f_x2_l720_720321

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := Real.log x + (1/2) * x^2 - 2 * k * x

-- Prove monotonicity based on different values of k ∈ ℝ
theorem monotonicity_of_f (k : ℝ) : 
  (forall x : ℝ, x > 0 → f x k → f' x k ≥ 0) ∨
  (forall x : ℝ, x > 0 → k ≤ 1 → f x k → f' x k ≥ 0) ∨
  (forall x : ℝ, x > 0 → k > 1 → 
    (forall x ∈ (0, k - sqrt (k ^ 2 - 1)), f' x k > 0) ∧
    (forall x ∈ (k - sqrt (k ^ 2 - 1), k + sqrt (k ^ 2 - 1)), f' x k < 0) ∧
    (forall x ∈ (k + sqrt (k ^ 2 - 1), +∞), f' x k > 0)) :=
sorry

-- Prove f(x2) < -3/2 given f(x) has two extreme points x1, x2 with x1 < x2 and k > 1
theorem extreme_points_f_x2 (k : ℝ) (h : k > 1) (x1 x2 : ℝ) (h1 : x1 < x2) (h2 : f' x1 k = 0) (h3 : f' x2 k = 0) :
  f x2 k < -3 / 2 :=
sorry

end monotonicity_of_f_extreme_points_f_x2_l720_720321


namespace inscribed_circle_radius_l720_720569

theorem inscribed_circle_radius (A B C : Point) (AB AC BC : ℝ) (hAB : AB = 8) (hAC : AC = 8) (hBC : BC = 5) : 
  radius_of_inscribed_circle_in_triangle ABC AB AC BC = 76 * Real.sqrt 10 / 21 :=
by {
  sorry -- Proof goes here.
}

end inscribed_circle_radius_l720_720569


namespace length_of_train_is_750m_l720_720014

-- Defining the conditions
def train_and_platform_equal_length : Prop := ∀ (L : ℝ), (Length_of_train = L ∧ Length_of_platform = L)
def train_speed := 90 * (1000 / 3600)  -- Convert speed from km/hr to m/s
def crossing_time := 60  -- Time given in seconds

-- Definition for the length of the train
def Length_of_train := sorry -- Given that it should be derived

-- The proof problem statement
theorem length_of_train_is_750m : (train_and_platform_equal_length ∧ train_speed ∧ crossing_time → Length_of_train = 750) :=
by
  -- Proof is skipped
  sorry

end length_of_train_is_750m_l720_720014


namespace smallest_five_digit_divisible_by_primes_l720_720710

theorem smallest_five_digit_divisible_by_primes : 
  let primes := [2, 3, 5, 7, 11] in
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11))) in
  let five_digit_threshold := 10000 in
  ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
by
  let primes := [2, 3, 5, 7, 11]
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))
  have lcm_2310 : lcm_primes = 2310 := sorry
  let five_digit_threshold := 10000
  have exists_n : ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
    sorry
  exists_intro 5
  have 5_condition : 5 > 0 := sorry
  have 2310_5_condition : 2310 * 5 >= five_digit_threshold := sorry
  have answer : 2310 * 5 = 11550 := sorry
  exact  ⟨5, 5_condition, 2310_5_condition, answer⟩
  exact ⟨5, 5 > 0, 2310 * 5 ≥ 10000, 2310 * 5 = 11550⟩
  sorry

end smallest_five_digit_divisible_by_primes_l720_720710


namespace length_AB_l720_720924

theorem length_AB (x : ℝ) (h1 : 0 < x)
  (hG : G = (0 + 1) / 2)
  (hH : H = (0 + G) / 2)
  (hI : I = (0 + H) / 2)
  (hJ : J = (0 + I) / 2)
  (hAJ : J - 0 = 2) :
  x = 32 := by
  sorry

end length_AB_l720_720924


namespace disjoint_triangles_exists_l720_720563

theorem disjoint_triangles_exists (n : ℕ) (points : Fin (3 * n) → ℝ × ℝ)
  (h_no_three_collinear : ∀ (i j k : Fin (3 * n)), i ≠ j → j ≠ k → i ≠ k → 
    ¬ AffineMap.collinear_points (points i) (points j) (points k)) :
  ∃ (triangles : Fin n → Fin 3 → Fin (3 * n)),
    (∀ (i : Fin n), 
      ¬ AffineMap.collinear_points (points (triangles i 0)) 
                                 (points (triangles i 1))
                                 (points (triangles i 2))) ∧
    (∀ (i j : Fin n), i ≠ j → 
      ∀ (a b : Fin 3), (triangles i a) ≠ (triangles j b)) :=
by
  sorry

end disjoint_triangles_exists_l720_720563


namespace value_of_V3_l720_720562

def f (x : ℝ) : ℝ := 3 * x^5 + 8 * x^4 - 3 * x^3 + 5 * x^2 + 12 * x - 6

def horner (a : ℝ) : ℝ :=
  let V0 := 3
  let V1 := V0 * a + 8
  let V2 := V1 * a - 3
  let V3 := V2 * a + 5
  V3

theorem value_of_V3 : horner 2 = 55 :=
  by
    simp [horner]
    sorry

end value_of_V3_l720_720562


namespace domain_cannot_be_0_to_3_l720_720598

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2 * x + 2

-- Define the range of the function f
def range_f : Set ℝ := Set.Icc 1 2

-- Statement that the domain [0, 3] cannot be the domain of f given the range
theorem domain_cannot_be_0_to_3 :
  ∀ (f : ℝ → ℝ) (range_f : Set ℝ),
    (∀ x, 1 ≤ f x ∧ f x ≤ 2) →
    ¬ ∃ dom : Set ℝ, dom = Set.Icc 0 3 ∧ 
      (∀ x ∈ dom, f x ∈ range_f) :=
by
  sorry

end domain_cannot_be_0_to_3_l720_720598


namespace find_minimum_value_l720_720806

open Real

theorem find_minimum_value (x y z : ℝ) (hx : 0 ≤ x) (hx' : x ≤ π) (hy : 0 ≤ y) (hy' : y ≤ π) (hz : 0 ≤ z) (hz' : z ≤ π) :
  ∃ (min_val : ℝ), min_val = -1 ∧ min_val = min (cos (x - y) + cos (y - z) + cos (z - x)) := 
sorry

end find_minimum_value_l720_720806


namespace sign_selection_sum_zero_l720_720880

theorem sign_selection_sum_zero
  (n : ℕ) (h_n : 2 ≤ n)
  (a : ℕ → ℕ) (h_a : ∀ k, 1 ≤ k ∧ k ≤ n → 0 < a k ∧ a k ≤ k)
  (h_sum_even : ∑ k in finset.range n, a (k + 1) % 2 = 0) :
  ∃ (sign : ℕ → ℤ), (∀ k, 1 ≤ k ∧ k ≤ n → sign k = 1 ∨ sign k = -1) ∧ ∑ k in finset.range n, sign (k + 1) * a (k + 1) = 0 :=
by
  sorry

end sign_selection_sum_zero_l720_720880


namespace find_m_l720_720307

theorem find_m 
  (m : ℕ) 
  (hm_pos : 0 < m) 
  (h1 : Nat.lcm 30 m = 90) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 36 := 
sorry

end find_m_l720_720307


namespace complex_purely_imaginary_l720_720352

theorem complex_purely_imaginary (m : ℝ) :
  (m^2 - 3*m + 2 = 0) ∧ (m^2 - 2*m ≠ 0) → m = 1 :=
by {
  sorry
}

end complex_purely_imaginary_l720_720352


namespace sum_of_x_coords_of_points_above_line_l720_720810

def point := (ℝ × ℝ)

def points : List point := [(4, 15), (7, 25), (13, 38), (19, 45), (21, 52)]

def above_line (p : point) : Prop := p.2 > 3 * p.1 + 5

def sum_x_coords_above_line (pts : List point) : ℝ :=
  pts.filter above_line |>.foldr (λ p acc => p.1 + acc) 0

theorem sum_of_x_coords_of_points_above_line : sum_x_coords_above_line points = 0 :=
by
  sorry

end sum_of_x_coords_of_points_above_line_l720_720810


namespace negation_proposition_l720_720023

theorem negation_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by
  sorry

end negation_proposition_l720_720023


namespace P_eval_two_ge_three_pow_n_l720_720896

open Polynomial

variables (P : Polynomial ℝ) (n : ℕ)

-- Conditions
def isMonic : Prop := P.monic
def positiveRealCoefficients : Prop := ∀ i, P.coeff i > 0
def degree_n : Prop := P.natDegree = n
def has_n_real_roots : Prop := ∃ roots : Fin n → ℝ, ∀ x : ℝ, P.isRoot x → x ∈ Multiset.toFinset (Finset.image ((roots : Fin n → ℝ) ∘ Fin.ofMultiset) Multiset.finRange)
def P_at_0_one : Prop := P.eval 0 = 1

-- Theorem statement
theorem P_eval_two_ge_three_pow_n 
  (H1 : isMonic P) 
  (H2 : positiveRealCoefficients P) 
  (H3 : degree_n P n) 
  (H4 : has_n_real_roots P n) 
  (H5 : P_at_0_one P) : 
  P.eval 2 ≥ 3 ^ n := 
sorry

end P_eval_two_ge_three_pow_n_l720_720896


namespace f_2008_of_neg1_l720_720901

theorem f_2008_of_neg1 :
  ∃ (a b : ℝ), 
    (f_5 : ℝ → ℝ) = (λ x, 32 * x + 31) ∧
    (f : ℝ → ℝ) = (λ x, a * x + b) ∧
    (f_1 : ℝ → ℝ) = f ∧
    (∀ n : ℕ, f (f_n n x) = f_{n+1} (f_n n x)) ∧
    (f_2008 (-1) = -1)
:= sorry

end f_2008_of_neg1_l720_720901


namespace MN_parallel_PQ_l720_720948

-- Definitions and conditions
variable {A B C I Z M N P Q : Type}
variable [is_center_of_circumcircle I A B C]
variable [midpoint_of_arc I A C]
variable [angle_bisector_of IZ (∠ B)]
variable [perpendicular PQ IZ]
variable [isosceles_triangle M BI]
variable [isosceles_triangle N BI]

-- The theorem statement
theorem MN_parallel_PQ (h_center: is_center_of_circumcircle I A B C) 
  (h_midpoint: midpoint_of_arc I A C) (h_bisector: angle_bisector_of IZ (∠ B)) 
  (h_perpendicular: perpendicular PQ IZ) (h_mbi: isosceles_triangle M BI) 
  (h_nbi: isosceles_triangle N BI) : 
  parallel MN PQ :=
sorry

end MN_parallel_PQ_l720_720948


namespace sampling_probabilities_equal_l720_720032

variables (total_items first_grade_items second_grade_items equal_grade_items substandard_items : ℕ)
variables (p_1 p_2 p_3 : ℚ)

-- Conditions given in the problem
def conditions := 
  total_items = 160 ∧ 
  first_grade_items = 48 ∧ 
  second_grade_items = 64 ∧ 
  equal_grade_items = 3 ∧ 
  substandard_items = 1 ∧ 
  p_1 = 1 / 8 ∧ 
  p_2 = 1 / 8 ∧ 
  p_3 = 1 / 8

-- The theorem to be proved
theorem sampling_probabilities_equal (h : conditions total_items first_grade_items second_grade_items equal_grade_items substandard_items p_1 p_2 p_3) :
  p_1 = p_2 ∧ p_2 = p_3 :=
sorry

end sampling_probabilities_equal_l720_720032


namespace max_chord_length_l720_720792

def curve (θ : ℝ) (x y : ℝ) : Prop :=
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

def line (x y : ℝ) : Prop :=
  y = 2 * x

theorem max_chord_length (θ : ℝ) :
  let line_x := (8 * Real.sin θ + Real.cos θ + 1) / (2 * (2 * Real.sin θ - Real.cos θ + 3))
  let x_max := Real.abs line_x
  let chord_length := 2 * Real.sqrt 2 * x_max
  ∃ θ : ℝ, chord_length = 8 * Real.sqrt 5 :=
sorry

end max_chord_length_l720_720792


namespace problem_a_problem_b_problem_c_l720_720524

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end problem_a_problem_b_problem_c_l720_720524


namespace square_logarithm_base_l720_720443

theorem square_logarithm_base (b : ℝ) (E F G : ℝ × ℝ)
  (h1 : EFGH_area : (E.1 - G.1) * (E.2 - G.2) = 64)
  (h2 : EF_parallel_x : E.2 = F.2)
  (h3 : E_on_graph : ∃ x, E = (x, log b x))
  (h4 : F_on_graph : ∃ y, F = (y, log b y))
  (h5 : G_on_graph : ∃ z, G = (z, log b z))
  : b = 2 ^ (7 / 8) :=
sorry

end square_logarithm_base_l720_720443


namespace problem_a_problem_b_problem_c_l720_720522

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end problem_a_problem_b_problem_c_l720_720522


namespace decimal_to_fraction_l720_720075

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l720_720075


namespace decagon_diagonals_l720_720828

-- Define the number of sides of the polygon
def n : ℕ := 10

-- Calculate the number of diagonals in an n-sided polygon
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem that the number of diagonals in a decagon is 35
theorem decagon_diagonals : number_of_diagonals n = 35 := by
  sorry

end decagon_diagonals_l720_720828


namespace same_heads_probability_l720_720420

-- Define the number of coins Keiko tosses and the number of coins Ephraim tosses
def keiko_coins := 2
def ephraim_coins := 3

-- Probability that Ephraim gets the same number of heads as Keiko
theorem same_heads_probability : 
  (countSameHeadsOutcomes keiko_coins ephraim_coins) / (totalPossibleOutcomes keiko_coins ephraim_coins) = 3 / 16 := 
sorry

end same_heads_probability_l720_720420


namespace kevin_birth_year_l720_720009

theorem kevin_birth_year (year_first_amc: ℕ) (annual: ∀ n, year_first_amc + n = year_first_amc + n) (age_tenth_amc: ℕ) (year_tenth_amc: ℕ) (year_kevin_took_amc: ℕ) 
  (h_first_amc: year_first_amc = 1988) (h_age_tenth_amc: age_tenth_amc = 13) (h_tenth_amc: year_tenth_amc = year_first_amc + 9) (h_kevin_took_amc: year_kevin_took_amc = year_tenth_amc) :
  year_kevin_took_amc - age_tenth_amc = 1984 :=
by
  sorry

end kevin_birth_year_l720_720009


namespace clubsuit_comm_l720_720663

def clubsuit (a b : ℝ) : ℝ := a^2 * b - 2 * a * b^2

theorem clubsuit_comm (x y : ℝ) : clubsuit x y = clubsuit y x ↔ (x = 0 ∨ y = 0 ∨ x = y) :=
by 
  sorry

end clubsuit_comm_l720_720663


namespace sequence_properties_l720_720815

open Nat

theorem sequence_properties : 
  (∀ n : ℕ, n ≠ 0 → n * a (n + 1) - (n + 1) * a n = 2 * n ^ 2 + 2 * n) →
  a 1 = 1 →
  a 2 = 6 ∧
  a 3 = 15 ∧
  (∀ n : ℕ, n ≠ 0 → (a (n + 1) / (n + 1) - a n / n = 2) ∧
  a n = 2 * n ^ 2 - n) :=
sorry

end sequence_properties_l720_720815


namespace one_third_12x_plus_5_l720_720839

-- Define x as a real number
variable (x : ℝ)

-- Define the hypothesis
def h := 12 * x + 5

-- State the theorem
theorem one_third_12x_plus_5 : (1 / 3) * (12 * x + 5) = 4 * x + 5 / 3 :=
  by 
    sorry -- Proof is omitted

end one_third_12x_plus_5_l720_720839


namespace missing_number_l720_720254

theorem missing_number :
  ∀ (x : ℝ), 
  (| x - 8 * (3 - 12) | - | 5 - 11 | = 70) → (x = 4) := 
by
  intro x h
  have h1 : 8 * (3 - 12) = -72 := by norm_num
  have h2 : |5 - 11| = 6 := by norm_num
  rw [h1, h2] at h
  have h3 : |x + 72| - 6 = 70 := by assumption
  sorry

end missing_number_l720_720254


namespace speech_competition_l720_720993

theorem speech_competition (num_contestants : ℕ) (num_topics : ℕ) : 
  num_contestants = 4 → num_topics = 4 → 
  let scenarios := (num_topics - 1) ^ num_contestants * num_topics in
  scenarios = 324 := 
by 
  intros hc ht
  have : scenarios = (num_topics - 1) ^ num_contestants * num_topics,
  sorry

end speech_competition_l720_720993


namespace find_number_of_oranges_l720_720635

variables (apples_cost_per_item : ℕ) (oranges_cost_per_item : ℕ) (total_spent : ℕ)
variables (number_of_apples : ℕ) (number_of_oranges : ℕ)

-- Given conditions
def cost_of_apples := number_of_apples * apples_cost_per_item
def cost_of_oranges := number_of_oranges * oranges_cost_per_item
def total_expenditure := cost_of_apples + cost_of_oranges

-- Conditions provided in the problem
def conditions : Prop :=
  apples_cost_per_item = 1 ∧
  oranges_cost_per_item = 2 ∧
  total_spent = 9 ∧
  number_of_apples = 5

-- The statement to prove
theorem find_number_of_oranges (h : conditions) : number_of_oranges = 2 :=
by
  sorry

end find_number_of_oranges_l720_720635


namespace unique_intersection_value_of_b_l720_720208

theorem unique_intersection_value_of_b : 
  (∃ (x : ℝ), bx^2 + 5x + 2 = -2x - 3) ∧ (∀ (x₁ x₂ : ℝ), (bx^2 + 5x + 2 = -2x - 3 ∧ bx^2 + 5x + 2 = -2x - 3) → x₁ = x₂) → b = 49 / 20 :=
sorry

end unique_intersection_value_of_b_l720_720208


namespace probability_even_product_chips_l720_720672

theorem probability_even_product_chips :
  let chips := {1, 2, 4, 6} in
  let total_outcomes := fintype.card (prod (finset chips) (finset chips)) in
  let favorable_outcomes := total_outcomes - 1 in
  (favorable_outcomes : ℚ) / total_outcomes = 15 / 16 :=
by
  let chips := {1, 2, 4, 6}
  let total_outcomes := fintype.card (prod 𝓝 𝓝 chips chips)
  let favorable_outcomes := total_outcomes - 1
  have : total_outcomes = 16 := by sorry
  have : favorable_outcomes = 15 := by sorry
  rw [this, this, div_eq_div_iff]
  norm_num
  sorry

end probability_even_product_chips_l720_720672


namespace lateral_area_correct_l720_720786

-- Define the radius and height of the cone
def radius : ℕ := 3
def height : ℕ := 4

-- Define the slant height using the Pythagorean theorem
def slant_height : ℝ := Real.sqrt (radius ^ 2 + height ^ 2)

-- Define the circumference of the base
def circumference : ℝ := 2 * Real.pi * radius

-- Define the lateral area of the cone
def lateral_area : ℝ := (1 / 2) * circumference * slant_height

-- Statement to prove the lateral area is 15 * π
theorem lateral_area_correct : lateral_area = 15 * Real.pi := by
  sorry

end lateral_area_correct_l720_720786


namespace complex_quadrant_l720_720371

-- Definitions for the condition
def z : ℂ := (5 + 4 * complex.I) + (-1 + 2 * complex.I)

-- Proof statement
theorem complex_quadrant (hz : z = 4 + 6 * complex.I) : (complex.re z > 0) ∧ (complex.im z > 0) :=
by
  -- This is because a point (x,y) lies in the first quadrant if x > 0 and y > 0
  rw hz
  exact ⟨by norm_num, by norm_num⟩

end complex_quadrant_l720_720371


namespace fish_count_l720_720132

theorem fish_count (l r : ℕ) (hl : l = 10) (hr : r = 12) : l + r = 22 := by
  rw [hl, hr]
  rfl

end fish_count_l720_720132


namespace number_of_good_pairs_l720_720881

def is_good_pair (a b k : Nat) : Prop :=
  1 ≤ a ∧ a < b ∧ b ≤ 100 ∧ ab ∣ a^k + b^k

theorem number_of_good_pairs : 
  (∑ n in finset.range 101, ∑ m in finset.range n, ∀ k, is_good_pair n m k) = 132 := 
sorry

end number_of_good_pairs_l720_720881


namespace max_value_A_l720_720140

noncomputable def A (x y : ℝ) : ℝ :=
  ((x^2 - y) * Real.sqrt (y + x^3 - x * y) + (y^2 - x) * Real.sqrt (x + y^3 - x * y) + 1) /
  ((x - y)^2 + 1)

theorem max_value_A (x y : ℝ) (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) :
  A x y ≤ 1 :=
sorry

end max_value_A_l720_720140


namespace trajectory_passes_incenter_l720_720885

noncomputable def fixed_point_O : Type := sorry
noncomputable def point_A_on_plane : Type := sorry
noncomputable def point_B_on_plane : Type := sorry
noncomputable def point_C_on_plane : Type := sorry
noncomputable def non_collinear (A B C : Type) : Prop := sorry
noncomputable def moving_point_P (O A B C : Type) (λ : ℝ) : Prop := sorry -- Given the specific vector equation

axiom O_fixed : fixed_point_O
axiom A_non_collinear : point_A_on_plane
axiom B_non_collinear : point_B_on_plane
axiom C_non_collinear : point_C_on_plane
axiom ABC_non_collinear : non_collinear A_non_collinear B_non_collinear C_non_collinear
axiom trajectory_P_eq : ∀ (λ : ℝ) (P : Type), λ ∈ set.Ici 0 → 
  moving_point_P O_fixed A_non_collinear B_non_collinear C_non_collinear λ

theorem trajectory_passes_incenter : 
  ∃ P : Type, moving_point_P O_fixed A_non_collinear B_non_collinear C_non_collinear (λ : ℝ) ∈ set.Ici 0 → 
  incenter:
    ∀ (P O A B C : Type) (λ : ℝ), moving_point_P O A B C λ → 
    point_constant_on_path_being_incenter : sorry

end trajectory_passes_incenter_l720_720885


namespace leak_empties_cistern_in_90_hours_l720_720609

theorem leak_empties_cistern_in_90_hours
  (H1 : ∀ (t: ℕ), t = 9 → (1 / 9) = t⁻¹)
  (H2 : ∀ (t: ℕ), t = 10 → (1 / 10) = t⁻¹) :
  ∃ (L: ℝ), (1 / 9) - L = (1 / 10) ∧ ∀ T, T = (1 / L) → T = 90 :=
by sorry

end leak_empties_cistern_in_90_hours_l720_720609


namespace find_width_l720_720987

variable (a b : ℝ)

def perimeter : ℝ := 6 * a + 4 * b
def length : ℝ := 2 * a + b
def width : ℝ := a + b

theorem find_width (h : perimeter a b = 6 * a + 4 * b)
                   (h₂ : length a b = 2 * a + b) : width a b = (perimeter a b) / 2 - length a b := by
  sorry

end find_width_l720_720987


namespace boys_without_glasses_l720_720421

def total_students_with_glasses : ℕ := 36
def girls_with_glasses : ℕ := 21
def total_boys : ℕ := 30

theorem boys_without_glasses :
  total_boys - (total_students_with_glasses - girls_with_glasses) = 15 :=
by
  sorry

end boys_without_glasses_l720_720421


namespace shortest_path_length_l720_720686

-- Define the regular tetrahedron with unit edge lengths
structure Tetrahedron :=
  (A B C D : ℝ × ℝ × ℝ)
  (dist_AB : dist A B = 1)
  (dist_AC : dist A C = 1)
  (dist_AD : dist A D = 1)
  (dist_BC : dist B C = 1)
  (dist_BD : dist B D = 1)
  (dist_CD : dist C D = 1)

-- Define the midpoint function
def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

-- Define the theorem to prove the shortest path
theorem shortest_path_length (T : Tetrahedron) :
  let M := midpoint T.A T.B,
      N := midpoint T.C T.D in
  dist M N = 1 :=
by
  sorry

end shortest_path_length_l720_720686


namespace smallest_positive_five_digit_number_divisible_by_first_five_primes_l720_720721

theorem smallest_positive_five_digit_number_divisible_by_first_five_primes :
  ∃ n : ℕ, (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ 10000 ≤ n ∧ n = 11550 :=
by
  use 11550
  split
  · intros p hp
    fin_cases hp <;> norm_num
  split
  · norm_num
  rfl

end smallest_positive_five_digit_number_divisible_by_first_five_primes_l720_720721


namespace range_of_a_l720_720356

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 :=
by {
  sorry -- Proof is not required as per instructions.
}

end range_of_a_l720_720356


namespace profit_percent_is_33_33_l720_720590

-- Defining the cost price (CP) as 75% of the selling price (SP)
def cost_price (SP : ℝ) : ℝ := 0.75 * SP

-- Defining the profit calculation
def profit (SP : ℝ) (CP : ℝ) : ℝ := SP - CP

-- Defining the profit_percent function
def profit_percent (SP : ℝ) (CP : ℝ) : ℝ := (profit SP CP / CP) * 100

-- Theorem to be proved
theorem profit_percent_is_33_33 (SP : ℝ) (h₁: SP > 0) :
  profit_percent SP (cost_price SP) = 33.33 :=
by
  sorry

end profit_percent_is_33_33_l720_720590


namespace parallel_MN_PQ_l720_720937

variables {A B C I Z M N P Q : Point}
variables {BI IZ PQ MN : Line}

-- Conditions
variables (triangle_ABC : is_triangle A B C)
variables (incircle_center_I : is_incircle_center I A B C)
variables (angle_bisector_IZ : is_angle_bisector Z B I triangle_ABC)
variables (perpendicular_PQ_IZ : is_perpendicular PQ IZ)
variables (perpendicular_IZ_BI : is_perpendicular IZ BI)
variables (perpendicular_MN_BI : is_perpendicular MN BI)

theorem parallel_MN_PQ 
  (h1: is_angle_bisector Z B I triangle_ABC)
  (h2: is_perpendicular PQ IZ)
  (h3: is_perpendicular IZ BI)
  (h4: is_perpendicular MN BI) : 
  is_parallel MN PQ := 
sorry

end parallel_MN_PQ_l720_720937


namespace option_B_is_incorrect_l720_720820

-- Define the set A
def A := { x : ℤ | x ^ 2 - 4 = 0 }

-- Statement to prove that -2 is an element of A
theorem option_B_is_incorrect : -2 ∈ A :=
sorry

end option_B_is_incorrect_l720_720820


namespace derivative_of_reciprocal_l720_720453

-- Define the function f(x) = 1 / x
def f (x : ℝ) : ℝ := 1 / x

-- State the problem: Prove that the derivative of f(x) is -1 / x^2 given f(x) = 1 / x
theorem derivative_of_reciprocal :
  ∀ x ≠ 0, (deriv f x) = -1 / x^2 := by
sorry

end derivative_of_reciprocal_l720_720453


namespace jumping_rope_count_l720_720544

theorem jumping_rope_count : 
  let hacky_sack_players := 6 in
  let rope_jumping_players := 6 * hacky_sack_players in
  rope_jumping_players ≠ 12 :=
by
  -- skipping the proof for this example
  sorry

end jumping_rope_count_l720_720544


namespace find_m_value_l720_720303

theorem find_m_value
    (x y m : ℝ)
    (hx : x = -1)
    (hy : y = 2)
    (hxy : m * x + 2 * y = 1) :
    m = 3 :=
by
  sorry

end find_m_value_l720_720303


namespace smallest_five_digit_number_divisible_l720_720689

def smallest_prime_divisible (n: ℕ) : Prop :=
  ∃ k: ℕ, n = 2310 * k ∧ 10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_number_divisible :
  ∃ (n: ℕ), smallest_prime_divisible n ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_l720_720689


namespace smallest_five_digit_number_divisible_by_five_primes_l720_720724

theorem smallest_five_digit_number_divisible_by_five_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  lcm = 2310 → (∃ n : ℕ, n = 5 ∧ 10000 ≤ lcm * n ∧ lcm * n = 11550) :=
by
  intros p1 p2 p3 p4 p5 
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  intro hlcm
  use (5 : ℕ)
  split
  { exact rfl }
  split
  { sorry }
  { sorry }

end smallest_five_digit_number_divisible_by_five_primes_l720_720724


namespace curved_surface_area_of_cone_l720_720264

-- Definitions
def radius : ℝ := 7
def slant_height : ℝ := 14
def π := Real.pi

-- Problem Statement: Proof of CSA being 98π m²
theorem curved_surface_area_of_cone (r : ℝ) (l : ℝ) (CSA : ℝ) (h_r : r = radius) (h_l : l = slant_height) : CSA = π * r * l :=
by
  rw [h_r, h_l]
  sorry

end curved_surface_area_of_cone_l720_720264


namespace minimum_area_convex_quadrilateral_l720_720006

theorem minimum_area_convex_quadrilateral
  (S_AOB S_COD : ℝ) (h₁ : S_AOB = 4) (h₂ : S_COD = 9) :
  (∀ S_BOC S_AOD : ℝ, S_AOB * S_COD = S_BOC * S_AOD → 
    (S_AOB + S_BOC + S_COD + S_AOD) ≥ 25) := sorry

end minimum_area_convex_quadrilateral_l720_720006


namespace veridux_male_associates_l720_720642

theorem veridux_male_associates (total_employees female_employees total_managers female_managers : ℕ)
  (h1 : total_employees = 250)
  (h2 : female_employees = 90)
  (h3 : total_managers = 40)
  (h4 : female_managers = 40) :
  total_employees - female_employees = 160 :=
by
  sorry

end veridux_male_associates_l720_720642


namespace arithmetic_sequence_common_difference_l720_720029

-- Define the conditions
variables {S_3 a_1 a_3 : ℕ}
variables (d : ℕ)
axiom h1 : S_3 = 6
axiom h2 : a_3 = 4
axiom h3 : S_3 = 3 * (a_1 + a_3) / 2

-- Prove that the common difference d is 2
theorem arithmetic_sequence_common_difference :
  d = (a_3 - a_1) / 2 → d = 2 :=
by
  sorry -- Proof to be completed

end arithmetic_sequence_common_difference_l720_720029


namespace find_ordered_pair_l720_720021

theorem find_ordered_pair :
  ∃ p m : ℚ, 
  (∀ t : ℚ,
    y = (2/3) * x - 5 ↔ 
    (x, y) = (-6, p) + t * (m, 7)) → 
  (p, m) = (-9, 21/2) :=
sorry

end find_ordered_pair_l720_720021


namespace rational_points_colored_l720_720864

def rational_point (x : ℚ × ℚ) : Prop :=
  ∃ a b c d : ℤ, b > 0 ∧ d > 0 ∧ Int.gcd a b = 1 ∧ Int.gcd c d = 1 ∧ 
  x = (a / b, c / d)

def color_point (n : ℕ) (q : ℚ × ℚ) : ℕ :=
  let v_2 (x : ℤ) : ℕ := Nat.find (λ k => 2^k ∣ x ∧ ¬ 2^(k + 1) ∣ x) 
  natMod (max (v_2 q.1.den) (v_2 q.2.den)) n

theorem rational_points_colored {n : ℕ} (n_pos : 0 < n) :
  ∃ f : (ℚ × ℚ) → Fin n, (∀ (p1 p2 : ℚ × ℚ), rational_point p1 → rational_point p2 →
  ∃ q, rational_point q ∧ 
  (∃ (λ : ℚ), 0 < λ ∧ λ < 1 ∧ q = (λ * p1.1 + (1 - λ) * p2.1, λ * p1.2 + (1 - λ) * p2.2) ∧ 
  ∀ i : Fin n, ∃ q : ℚ × ℚ, rational_point q ∧ color_point n q = i)) :=
sorry

end rational_points_colored_l720_720864


namespace inequality_a_inequality_c_inequality_d_l720_720475

variable (a b : ℝ)

theorem inequality_a (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 := 
Sorry

theorem inequality_c (h : a + b > 0) : a^21 + b^21 > 0 := 
Sorry

theorem inequality_d (h : a + b > 0) : (a + 2) * (b + 2) > a * b := 
Sorry

end inequality_a_inequality_c_inequality_d_l720_720475


namespace tea_bags_count_l720_720913

-- Definitions based on the given problem
def valid_bags (b : ℕ) : Prop :=
  ∃ (a c d : ℕ), a + b - a = b ∧ c + d = b ∧ 3 * c + 2 * d = 41 ∧ 3 * a + 2 * (b - a) = 58

-- Statement of the problem, confirming the proof condition
theorem tea_bags_count (b : ℕ) : valid_bags b ↔ b = 20 :=
by {
  -- The proof is left for completion
  sorry
}

end tea_bags_count_l720_720913


namespace baseball_card_total_percent_decrease_l720_720149

theorem baseball_card_total_percent_decrease :
  ∀ (original_value first_year_decrease second_year_decrease : ℝ),
  first_year_decrease = 0.60 →
  second_year_decrease = 0.10 →
  original_value > 0 →
  (original_value - original_value * first_year_decrease - (original_value * (1 - first_year_decrease)) * second_year_decrease) =
  original_value * (1 - 0.64) :=
by
  intros original_value first_year_decrease second_year_decrease h_first_year h_second_year h_original_pos
  sorry

end baseball_card_total_percent_decrease_l720_720149


namespace parallel_lines_l720_720953

-- Definition of points and lines
variables {A B C M N P Q I Z : Type} [geometry_space : Geometry A B C M N P Q I Z]

-- Conditions
def is_center_of_circumcircle (A B C I : Type) : Prop := Geometry.is_center_of_circumcircle A B C I
def midpoint_of_arc_AC (A C I : Type) : Prop := Geometry.midpoint_ω A C I
def is_angle_bisector (IZ : Type) (B : Type) : Prop := Geometry.angle_bisector IZ B
def is_perpendicular (PQ IZ : Type) : Prop := Geometry.perpendicular PQ IZ
def is_isosceles (MBI NBI BI : Type) : Prop := Geometry.isosceles MBI NBI BI

-- Problem Statement
theorem parallel_lines (A B C M N P Q I Z : Type)
  [is_center_of_circumcircle A B C I]
  [midpoint_of_arc_AC A C I]
  [is_angle_bisector IZ B]
  [is_perpendicular PQ IZ]
  [is_isosceles MBI NBI BI] : PQ ∥ MN :=
by
  sorry

end parallel_lines_l720_720953


namespace inequality_a_inequality_c_inequality_d_l720_720479

variable (a b : ℝ)

theorem inequality_a (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 := 
Sorry

theorem inequality_c (h : a + b > 0) : a^21 + b^21 > 0 := 
Sorry

theorem inequality_d (h : a + b > 0) : (a + 2) * (b + 2) > a * b := 
Sorry

end inequality_a_inequality_c_inequality_d_l720_720479


namespace number_of_true_propositions_is_one_l720_720638

theorem number_of_true_propositions_is_one : 
  let p1 := ∀ x : ℝ, x^4 > x^2 in
  let p2 := ∀ (p q : Prop), ¬ (p ∧ q) → ¬ p ∧ ¬ q in
  let p3 := ∀ x : ℝ, ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) → (∃ x : ℝ, x^3 - x^2 + 1 > 0) in
  (p1 = false) ∧
  (p2 = false) ∧
  (p3 = true) →
  (∑ b in {p1, p2, p3}, if b then 1 else 0) = 1 := 
by
  sorry

end number_of_true_propositions_is_one_l720_720638


namespace cone_base_radius_l720_720997

theorem cone_base_radius {r : ℝ} (h : r > 0) :
  (∃ (s θ : ℝ), s = 12 ∧ θ = 150 ∧ (θ / 360 * 2 * real.pi * s = 2 * real.pi * r)) →
  r = 5 :=
by
  intros h_exists
  cases h_exists with s hs
  cases hs with theta htheta
  cases htheta with hs_eq htheta_eq
  cases htheta_eq with ht_eq arc_eq
  sorry

end cone_base_radius_l720_720997


namespace decimal_to_fraction_l720_720074

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l720_720074


namespace smallest_possible_value_of_a_l720_720988

theorem smallest_possible_value_of_a (r1 r2 r3 : ℕ) (h_roots : (Polynomial.C (1890 : ℤ) + Polynomial.X * (Polynomial.C (-b : ℤ) + Polynomial.X * (Polynomial.C (-a : ℤ) + Polynomial.X))).roots = [r1, r2, r3]) 
(htotal_roots : r1 * r2 * r3 = 1890) 
(h_pos : r1 > 0 ∧ r2 > 0 ∧ r3 > 0) : 
a = min (r1 + r2 + r3) := by 
sorry

end smallest_possible_value_of_a_l720_720988


namespace problem_a_problem_c_problem_d_l720_720498

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end problem_a_problem_c_problem_d_l720_720498


namespace distance_downstream_l720_720028

-- Define the conditions in Lean
variables (speed_boat : ℝ) (c : ℝ) (D : ℝ)
-- Condition: speed of boat in still water
axiom speed_boat_const : speed_boat = 12
-- Condition: downstream time
axiom downstream_time : 3 * (speed_boat + c) = D
-- Condition: upstream time
axiom upstream_time : 6 * (speed_boat - c) = D

-- Define the main theorem to prove the distance downstream
theorem distance_downstream : (D = 48) :=
by
  substitute speed_boat_const into downstream_time,
  substitute speed_boat_const into upstream_time,
  -- here we skip the proof steps, as mentioned
  sorry

end distance_downstream_l720_720028


namespace inverse_function_property_l720_720347

-- Define the function f
def f (x : ℝ) : ℝ := 18 / (8 + 2 * x)

-- Define the property that we want to prove
theorem inverse_function_property (x : ℝ) (h : f x = 3) : (x ^ (-2)) = 1 :=
by
  -- Since this is just the statement, we add 'sorry' to complete the definition
  sorry

end inverse_function_property_l720_720347


namespace smallest_positive_five_digit_number_divisible_by_first_five_primes_l720_720716

theorem smallest_positive_five_digit_number_divisible_by_first_five_primes :
  ∃ n : ℕ, (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ 10000 ≤ n ∧ n = 11550 :=
by
  use 11550
  split
  · intros p hp
    fin_cases hp <;> norm_num
  split
  · norm_num
  rfl

end smallest_positive_five_digit_number_divisible_by_first_five_primes_l720_720716


namespace no_hyperdeficient_numbers_l720_720887

def sum_of_divisors (n : ℕ) : ℕ :=
  (Finset.range (n+1)).filter (λ i, i > 0 ∧ n % i = 0).sum id

def is_hyperdeficient (n : ℕ) : Prop :=
  sum_of_divisors (sum_of_divisors n) = n + 3

theorem no_hyperdeficient_numbers : ∀ n : ℕ, ¬ is_hyperdeficient n := 
by
  sorry

end no_hyperdeficient_numbers_l720_720887


namespace part_a_part_c_part_d_l720_720508

-- Define the variables
variables {a b : ℝ}

-- Define the conditions and statements
def cond := a + b > 0

theorem part_a (h : cond) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem part_c (h : cond) : a^21 + b^21 > 0 :=
sorry

theorem part_d (h : cond) : (a + 2) * (b + 2) > a * b :=
sorry

end part_a_part_c_part_d_l720_720508


namespace intersection_complement_eq_l720_720821

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

-- Theorem
theorem intersection_complement_eq : (A ∩ (U \ B)) = {2, 3} :=
by 
  sorry

end intersection_complement_eq_l720_720821


namespace sum_inscribed_squares_less_half_area_l720_720869

open Real

noncomputable def inscribed_square_area_sum (A B C : Point) : ℝ :=
  sorry

theorem sum_inscribed_squares_less_half_area (A B C : Point) (h : angle A B C ≥ π / 2) :
  inscribed_square_area_sum A B C < (1 / 2) * area_of_triangle A B C :=
sorry

end sum_inscribed_squares_less_half_area_l720_720869


namespace correct_proposition_l720_720342

theorem correct_proposition (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
sorry

end correct_proposition_l720_720342


namespace circle_center_radius_fixedPoint_l720_720126

noncomputable def fixedPoint (a : ℝ) : Prop :=
  ∃ C : ℝ × ℝ, C = (1, -2) ∧ ∀ x y : ℝ, (a - 1) * x - y - a - 1 = 0 → (x, y) = C

noncomputable def circleEquation (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 5

theorem circle_center_radius_fixedPoint :
  (∀ a : ℝ, fixedPoint a) →
  ∀ x y : ℝ, circleEquation x y ↔ x^2 + y^2 - 2x + 4y = 0 := by
  sorry

end circle_center_radius_fixedPoint_l720_720126


namespace Petya_running_time_l720_720219

theorem Petya_running_time (D V : ℝ) 
  (hV_pos : 0 < V) (hD_pos : 0 < D):
  let T := D / V in
  let V1 := 1.25 * V in
  let V2 := 0.8 * V in
  let T1 := (D / 2) / V1 in
  let T2 := (D / 2) / V2 in
  let T_actual := T1 + T2 in
  T_actual > T :=
by
  let T := D / V
  let V1 := 1.25 * V
  let V2 := 0.8 * V
  let T1 := (D / 2) / V1
  let T2 := (D / 2) / V2
  let T_actual := T1 + T2
  have : T_actual = (2 * D) / (5 * V) + (5 * D) / (8 * V) := by 
  sorry
  have : T_actual = 41 * D / (40 * V) := by 
  sorry
  have : T = D / V := by 
  sorry
  show 41 * D / (40 * V) > D / V := by 
  sorry

end Petya_running_time_l720_720219


namespace flour_already_put_in_l720_720418

def total_flour : ℕ := 8
def additional_flour_needed : ℕ := 6

theorem flour_already_put_in : total_flour - additional_flour_needed = 2 := by
  sorry

end flour_already_put_in_l720_720418


namespace drama_club_ticket_sales_l720_720448

theorem drama_club_ticket_sales :
  ∃ (A S : ℕ), 
    (A + S = 1500 ∧ 12 * A + 6 * S = 16200) ∧ S = 300 :=
by {
  use 1200,
  use 300,
  split,
  { 
    split,
    exact rfl, 
    norm_num
  },
  norm_num
}

end drama_club_ticket_sales_l720_720448


namespace petya_time_comparison_l720_720226

open Real

noncomputable def petya_planned_time (D V : ℝ) := D / V

noncomputable def petya_actual_time (D V : ℝ) :=
  let V1 := 1.25 * V
  let V2 := 0.80 * V
  let T1 := (D / 2) / V1
  let T2 := (D / 2) / V2
  T1 + T2

theorem petya_time_comparison (D V : ℝ) (hV : V > 0) : 
  petya_actual_time D V > petya_planned_time D V :=
by {
  let T := petya_planned_time D V
  let T_actual := petya_actual_time D V
  have h1 : petya_planned_time D V = D / V, by unfold petya_planned_time
  have h2 : petya_actual_time D V = (D * 41) / (40 * V), by {
      unfold petya_actual_time,
      have h3 : 1.25 * V = 5 * V / 4, by linarith,
      have h4 : 0.80 * V = 4 * V / 5, by linarith,
      rw [h3, h4],
      simp,
      linarith,
  },
  rw h1,
  rw h2,
  have h3 : (41 * D) / (40 * V) > D / V, by linarith,
  exact h3,
}

end petya_time_comparison_l720_720226


namespace chess_competition_l720_720848

theorem chess_competition (W M : ℕ) 
  (hW : W * (W - 1) / 2 = 45) 
  (hM : M * 10 = 200) :
  M * (M - 1) / 2 = 190 :=
by
  sorry

end chess_competition_l720_720848


namespace find_t_l720_720824

open Real

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_t (t : ℝ) :
  let m := (t + 1, 1)
  let n := (t + 2, 2)
  dot_product (vector_add m n) (vector_sub m n) = 0 → 
  t = -3 :=
by
  intro h
  sorry

end find_t_l720_720824


namespace problem_a_problem_c_problem_d_l720_720505

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end problem_a_problem_c_problem_d_l720_720505


namespace eval_sum_of_i_powers_l720_720253

noncomputable def i : ℂ := complex.I

theorem eval_sum_of_i_powers :
  i ^ 13 + i ^ 18 + i ^ 23 + i ^ 28 + i ^ 33 + i ^ 38 = 0 := by
  have h1 : i ^ 2 = -1 := by
    rw [complex.I_sq, neg_one]
  have h2 : i ^ 3 = -i := by
    rw [pow_succ, h1, mul_neg_one]
  have h3 : i ^ 4 = 1 := by
    rw [pow_succ, h2, neg_neg, mul_one]
  have h4 : i ^ 5 = i := by
    rw [pow_succ, h3, mul_one]
  calc
    i ^ 13 + i ^ 18 + i ^ 23 + i ^ 28 + i ^ 33 + i ^ 38
      = i + -1 + -i + 1 + i + -1 : by
        rw [←pow_add, show 4 * 3 + 1 = 13, by norm_num]
        rw [←pow_add, show 4 * 4 + 2 = 18, by norm_num]
        rw [←pow_add, show 4 * 5 + 3 = 23, by norm_num]
        rw [←pow_add, show 4 * 7 = 28, by norm_num]
        rw [←pow_add, show 4 * 8 + 1 = 33, by norm_num]
        rw [←pow_add, show 4 * 9 + 2 = 38, by norm_num]
      = (i + i + -i) + (-1 + 1 + -1) : by
        ring
      = i : by
        ring
      = 0 : by
        ring
  sorry

end eval_sum_of_i_powers_l720_720253


namespace new_triangle_perimeter_l720_720380

theorem new_triangle_perimeter (PQ PR QR : ℝ) (h1 : PQ = 8) (h2 : PR = 5) (h3 : QR = 6) :
  let new_PQ := PQ + 4 in
  let new_PR := PR + 4 in
  let new_QR := QR + 1 in
  new_PQ + new_PR + new_QR = 28 :=
by
  sorry

end new_triangle_perimeter_l720_720380


namespace part_a_part_c_part_d_l720_720513

-- Define the variables
variables {a b : ℝ}

-- Define the conditions and statements
def cond := a + b > 0

theorem part_a (h : cond) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem part_c (h : cond) : a^21 + b^21 > 0 :=
sorry

theorem part_d (h : cond) : (a + 2) * (b + 2) > a * b :=
sorry

end part_a_part_c_part_d_l720_720513


namespace largest_of_four_consecutive_even_sum_140_l720_720592

theorem largest_of_four_consecutive_even_sum_140 : 
  ∃ (n : ℕ), (4 * n + 12 = 140) ∧ (n + 6 = 38) :=
by
  use 32
  simp
  unfold is_consecutive_even_numbers
  split
  repeat { sorry }

end largest_of_four_consecutive_even_sum_140_l720_720592


namespace math_problem_l720_720469

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end math_problem_l720_720469


namespace train_length_problem_l720_720017

noncomputable def train_length (v : ℝ) (t : ℝ) (L : ℝ) : Prop :=
v = 90 / 3.6 ∧ t = 60 ∧ 2 * L = v * t

theorem train_length_problem : train_length 90 1 750 :=
by
  -- Define speed in m/s
  let v_m_s := 90 * (1000 / 3600)
  -- Calculate distance = speed * time
  let distance := 25 * 60
  -- Since distance = 2 * Length
  have h : 2 * 750 = 1500 := sorry
  show train_length 90 1 750
  simp [train_length, h]
  sorry

end train_length_problem_l720_720017


namespace find_x_and_union_l720_720329

def A : Set ℕ := {1, 3, 5}
def B (x : ℤ) : Set ℕ := {1, 2, x^2 - 1}

theorem find_x_and_union (x : ℤ) (hx : A ∩ B x = {1, 3}) : x = -2 ∧ A ∪ B x = {1, 2, 3, 5} :=
by
  sorry

end find_x_and_union_l720_720329


namespace probability_at_least_one_cherry_is_correct_l720_720455

open Nat
open ProbabilityTheory

noncomputable def probability_at_least_one_cherry :=
  let total_cuttings := 20
  let cherry_cuttings := 8
  let non_cherry_cuttings := total_cuttings - cherry_cuttings
  let ways_to_choose_3 := combinatorial.nCr total_cuttings 3
  let ways_to_choose_3_non_cherry := combinatorial.nCr non_cherry_cuttings 3
  let p_no_cherry := ways_to_choose_3_non_cherry.toReal / ways_to_choose_3.toReal
  -- The probability of at least one cherry
  1 - p_no_cherry

theorem probability_at_least_one_cherry_is_correct :
  probability_at_least_one_cherry = 46 / 57 :=
by
  sorry

end probability_at_least_one_cherry_is_correct_l720_720455


namespace decimal_to_fraction_l720_720100

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l720_720100


namespace Kim_has_4_cousins_l720_720878

noncomputable def pieces_per_cousin : ℕ := 5
noncomputable def total_pieces : ℕ := 20
noncomputable def cousins : ℕ := total_pieces / pieces_per_cousin

theorem Kim_has_4_cousins : cousins = 4 := 
by
  show cousins = 4
  sorry

end Kim_has_4_cousins_l720_720878


namespace visitors_on_monday_l720_720872

theorem visitors_on_monday (M : ℕ) (h : M + 2 * M + 100 = 250) : M = 50 :=
by
  sorry

end visitors_on_monday_l720_720872


namespace bacteria_population_growth_l720_720530

noncomputable def final_population (initial_population doubling_period total_time : ℝ) : ℝ :=
  initial_population * 2^(total_time / doubling_period).round

theorem bacteria_population_growth :
  final_population 1000 4 35.86 = 512000 := by
  sorry

end bacteria_population_growth_l720_720530


namespace calculate_p_q_l720_720834

theorem calculate_p_q :
  let S₈ := -176 - 64 * complex.I  -- Define S_8
  let S₉ := 2 * S₈ + 144 * complex.I  -- Define S_9 based on S_8
  let p := S₉.re  -- Real part of S_9
  let q := S₉.im  -- Imaginary part of S_9
  |p| + |q| = 368 := by
  sorry -- Proof omitted for this example

end calculate_p_q_l720_720834


namespace decimal_to_fraction_equivalence_l720_720050

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l720_720050


namespace sum_prime_factors_165_plus_5_l720_720124

theorem sum_prime_factors_165_plus_5 : 
  let prime_factors := [3, 5, 11] in
  prime_factors.sum + 5 = 24 := 
by
  -- Definition of prime_factors
  let prime_factors : List ℕ := [3, 5, 11]
  -- Calculate the sum of prime_factors
  have h_sum : prime_factors.sum = 19 := by
    simp [prime_factors]
  -- Prove the final statement
  have h_final_sum : prime_factors.sum + 5 = 24 := by
    rw [h_sum]
    norm_num
  exact h_final_sum

end sum_prime_factors_165_plus_5_l720_720124


namespace part1_solution_set_part2_range_of_a_l720_720322

-- Define the function f for part 1 
def f_part1 (x : ℝ) : ℝ := |2*x + 1| + |2*x - 1|

-- Define the function f for part 2 
def f_part2 (x a : ℝ) : ℝ := |2*x + 1| + |a*x - 1|

-- Theorem for part 1
theorem part1_solution_set (x : ℝ) : 
  (f_part1 x) ≥ 3 ↔ x ∈ (Set.Iic (-3/4) ∪ Set.Ici (3/4)) :=
sorry

-- Theorem for part 2
theorem part2_range_of_a (a : ℝ) : 
  (a > 0) → (∃ x : ℝ, f_part2 x a < (a / 2) + 1) ↔ (a ∈ Set.Ioi 2) :=
sorry

end part1_solution_set_part2_range_of_a_l720_720322


namespace sqrt_inequality_l720_720146

variable {a b c : ℝ}

theorem sqrt_inequality (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : 
  ∃ d : ℝ, d^2 = b^2 - ac ∧ d < sqrt 3 * a := 
sorry

end sqrt_inequality_l720_720146


namespace area_ratio_trapezoid_l720_720205

theorem area_ratio_trapezoid
  (ABCD : Type) [has_area ABCD]
  (P : ABCD)
  (AD BC : ∀ (P Q : ABCD), Prop)
  (h : AD = λ P Q, AD P Q ∧ BC = λ P Q, BC P Q )
  (area_ADP area_BCP : ℚ)
  (h_ratio : area_ADP / area_BCP = 1/2) :
  let area_ABCD := area_ADP + area_BCP + 2 * area_BCP in
  area_ADP / area_ABCD = 3 - 2 * real.sqrt 2 :=
by
  sorry

end area_ratio_trapezoid_l720_720205


namespace max_value_A_l720_720139

noncomputable def A (x y : ℝ) : ℝ :=
  ((x^2 - y) * Real.sqrt (y + x^3 - x * y) + (y^2 - x) * Real.sqrt (x + y^3 - x * y) + 1) /
  ((x - y)^2 + 1)

theorem max_value_A (x y : ℝ) (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) :
  A x y ≤ 1 :=
sorry

end max_value_A_l720_720139


namespace necessary_and_sufficient_condition_l720_720990

variable (a : ℝ)

theorem necessary_and_sufficient_condition :
  (-16 ≤ a ∧ a ≤ 0) ↔ ∀ x : ℝ, ¬(x^2 + a * x - 4 * a < 0) :=
by
  sorry

end necessary_and_sufficient_condition_l720_720990


namespace problem_a_problem_c_problem_d_l720_720499

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end problem_a_problem_c_problem_d_l720_720499


namespace inequality_a_inequality_c_inequality_d_l720_720496

variable {a b : ℝ}

axiom (h : a + b > 0)

theorem inequality_a : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_c : a^21 + b^21 > 0 :=
sorry

theorem inequality_d : (a + 2) * (b + 2) > a * b :=
sorry

end inequality_a_inequality_c_inequality_d_l720_720496


namespace hemisphere_surface_area_l720_720537

theorem hemisphere_surface_area (r : ℝ) (h : r = 10) : 
  (4 * Real.pi * r^2) / 2 + (Real.pi * r^2) = 300 * Real.pi := by
  sorry

end hemisphere_surface_area_l720_720537


namespace balls_into_boxes_l720_720831

theorem balls_into_boxes : ∃ n : ℕ, (n = 22) ∧ (∃ (balls boxes : ℕ), 
  balls = 6 ∧ boxes = 4 ∧ (∀ (f : Fin 6 → Fin 4), 
    (∀ j : Fin 4, 0 < ∑ i, if f i = j then 1 else 0) ↔
    n = 22)) :=
by
  use 22
  split
  { reflexivity }
  { use 6, 4
    split
    { reflexivity }
    { split
      { reflexivity }
      { intros f
        split
        { intro h
          sorry -- proving that the distribution is valid
        }
        { intro h
          sorry -- proving that the distribution matches 22
        }
      }
    }
  }

end balls_into_boxes_l720_720831


namespace time_to_pass_l720_720042

def length_of_train := 500 -- meters
def speed_train1 := 45 -- km/hr
def speed_train2 := 30 -- km/hr
def relative_speed_kmhr := speed_train1 + speed_train2 -- km/hr
def relative_speed_mps := (relative_speed_kmhr * 1000) / 3600 -- m/s

theorem time_to_pass : 
  ∀ (length_of_train : ℝ) (relative_speed_mps : ℝ), 
  relative_speed_mps = 20.8333 →
  length_of_train = 500 →
  (length_of_train / relative_speed_mps) ≈ 24 :=
by
  sorry

end time_to_pass_l720_720042


namespace part1_part2_l720_720750

theorem part1 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + 4*b^2 = 1/(a*b) + 3) :
  a*b ≤ 1 := sorry

theorem part2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + 4*b^2 = 1/(a*b) + 3) (hba : b > a) :
  1/a^3 - 1/b^3 ≥ 3 * (1/a - 1/b) := sorry

end part1_part2_l720_720750


namespace circle_equation_l720_720617

-- Define the given conditions
def point_P : ℝ × ℝ := (-1, 0)
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1
def center_C : ℝ × ℝ := (1, 2)

-- Define the required equation of the circle and the claim
def required_circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 2

-- The Lean theorem statement
theorem circle_equation :
  ∃ (x y : ℝ), required_circle x y :=
sorry

end circle_equation_l720_720617


namespace evaluate_expression_l720_720410

noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 15
noncomputable def b : ℝ := -Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 15
noncomputable def c : ℝ := Real.sqrt 5 - Real.sqrt 3 + Real.sqrt 15
noncomputable def d : ℝ := -Real.sqrt 5 - Real.sqrt 3 + Real.sqrt 15

theorem evaluate_expression : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 240 / 961 := 
by 
  sorry

end evaluate_expression_l720_720410


namespace proof_f_f_half_eq_nine_half_l720_720317

def f (x : ℝ) : ℝ := 
  if -1 < x ∧ x < 2 then 2 * x 
  else if x ≥ 2 then (x ^ 2) / 2 
  else 0  -- It doesn't mention what happens when x <= -1, so we'll assume 0 for now.

theorem proof_f_f_half_eq_nine_half : f (f (3 / 2)) = 9 / 2 := 
by
  sorry

end proof_f_f_half_eq_nine_half_l720_720317


namespace min_digits_decimal_correct_l720_720122

noncomputable def min_digits_decimal : ℕ := 
  let n : ℕ := 123456789
  let d : ℕ := 2^26 * 5^4
  26 -- As per the problem statement

theorem min_digits_decimal_correct :
  let n := 123456789
  let d := 2^26 * 5^4
  ∀ x:ℕ, (∃ k:ℕ, n = k * 10^x) → x ≥ min_digits_decimal := 
by
  sorry

end min_digits_decimal_correct_l720_720122


namespace looms_employed_l720_720627

theorem looms_employed 
    (sales_value : ℕ) (manufacturing_expenses : ℕ) (establishment_charges : ℕ)
    (decrease_in_profit : ℕ)
    (equal_sales_contribution : Prop) (even_manufacturing_distribution : Prop)
    (one_loom_breaks_down_implication : Prop) :
    sales_value = 500000 →
    manufacturing_expenses = 150000 →
    establishment_charges = 75000 →
    decrease_in_profit = 3500 →
    equal_sales_contribution →
    even_manufacturing_distribution →
    one_loom_breaks_down_implication →
    ∃ L : ℕ, L = 100 :=
begin
    intros hs hm he hd _ _ _,
    use 100,
    sorry
end

end looms_employed_l720_720627


namespace a_2016_eq_4_div_5_l720_720375

noncomputable def a : ℕ → ℚ
| 0 := -1/4
| (n + 1) := 1 - 1 / a n

theorem a_2016_eq_4_div_5 : a 2015 = 4 / 5 :=
sorry

end a_2016_eq_4_div_5_l720_720375


namespace part_a_1_part_a_2_part_b_1_part_b_2_l720_720902

noncomputable theory

-- Define the random variable ξ uniformly distributed on [-1,1]
def ξ : Type := unit → ℝ

axiom ξ_uniform : ∀ x, -1 ≤ ξ x ∧ ξ x ≤ 1

-- Expected values and conditional expectations are defined
axiom E : (ξ -> ℝ) -> ℝ
axiom E_cond : (ξ -> ℝ) -> (ξ -> ℝ) -> ℝ

-- Definition of optimal linear estimate function
axiom linear_estimate : (ξ -> ℝ) -> (ξ -> ℝ) -> ℝ

-- Given ξ is uniformly distributed on [-1,1]
variable (ξ)

-- Prove the following:
theorem part_a_1 : E_cond (λ x, (ξ x) ^ 2) (λ x, ξ x) = λ x, (ξ x) ^ 2 := 
sorry

theorem part_a_2 : E_cond (λ x, ξ x) (λ x, (ξ x) ^ 2) = λ x, 0 := 
sorry

theorem part_b_1 : linear_estimate (λ x, (ξ x) ^ 2) (λ x, ξ x) = λ x, 1 / 3 := 
sorry

theorem part_b_2 : linear_estimate (λ x, ξ x) (λ x, (ξ x) ^ 2) = λ x, 0 := 
sorry

end part_a_1_part_a_2_part_b_1_part_b_2_l720_720902


namespace area_triangle_ABC_l720_720844

theorem area_triangle_ABC 
  (A B C D E F : ℝ × ℝ)
  (midpoint_D : D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (ratio_AE_EC : (E.1 - A.1) * 2 = (C.1 - E.1) ∧ (E.2 - A.2) * 2 = (C.2 - E.2))
  (ratio_AF_FD : (F.1 - A.1) * 3 = (D.1 - F.1) ∧ (F.2 - A.2) * 3 = (D.2 - F.2))
  (area_DEF : 17)
  (area_correct : 408) :
  let triangle_area := λ (P Q R : ℝ × ℝ), 0.5 * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2)) in
  triangle_area A B C = area_correct :=
by
  sorry

end area_triangle_ABC_l720_720844


namespace decimal_to_fraction_l720_720061

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l720_720061


namespace ratio_misses_hits_l720_720851

theorem ratio_misses_hits (misses hits total : ℕ) 
    (h_misses : misses = 50) 
    (h_total : total = 200) 
    (h_total_eq : total = misses + hits) : 
    (misses / nat.gcd misses hits = 1) ∧ (hits / nat.gcd misses hits = 3) :=
by
  sorry

end ratio_misses_hits_l720_720851


namespace f_positive_f_decreasing_solve_inequality_l720_720785

noncomputable def f : ℝ → ℝ := sorry -- as we don't have the explicit form of f

axiom functional_equation : ∀ a b : ℝ, f(a + b) = f(a) * f(b)
axiom positive_negatives : ∀ x : ℝ, x < 0 → f(x) > 1
axiom non_zero_function : ∀ x : ℝ, f(x) ≠ 0
axiom f_at_4 : f(4) = 1/16

theorem f_positive (x : ℝ) : f(x) > 0 := sorry

theorem f_decreasing (x1 x2 : ℝ) (h : x1 < x2) : f(x1) > f(x2) := sorry

theorem solve_inequality (x : ℝ) : 
  f(x - 3) * f(5 - x^2) ≤ 1 / 4 ↔ 0 ≤ x ∧ x ≤ 1 := sorry

end f_positive_f_decreasing_solve_inequality_l720_720785


namespace tetrahedron_parallelepiped_areas_tetrahedron_heights_distances_l720_720131

-- Definition for Part (a)
theorem tetrahedron_parallelepiped_areas 
  (S1 S2 S3 S4 P1 P2 P3 : ℝ)
  (h1 : true)
  (h2 : true) :
  S1^2 + S2^2 + S3^2 + S4^2 = P1^2 + P2^2 + P3^2 := 
sorry

-- Definition for Part (b)
theorem tetrahedron_heights_distances 
  (h1 h2 h3 h4 d1 d2 d3 : ℝ)
  (h : true) :
  (1/(h1^2)) + (1/(h2^2)) + (1/(h3^2)) + (1/(h4^2)) = (1/(d1^2)) + (1/(d2^2)) + (1/(d3^2)) := 
sorry

end tetrahedron_parallelepiped_areas_tetrahedron_heights_distances_l720_720131


namespace choose_50_boxes_contains_half_balls_100_choose_50_boxes_contains_half_balls_99_l720_720359

theorem choose_50_boxes_contains_half_balls_100 (boxes : Fin 100 → ℕ × ℕ) :
  ∃ subset : Finset (Fin 100), subset.card = 50 ∧ 
  (∑ i in subset, (boxes i).1) ≥ (∑ i, (boxes i).1) / 2 ∧
  (∑ i in subset, (boxes i).2) ≥ (∑ i, (boxes i).2) / 2 := 
    sorry

theorem choose_50_boxes_contains_half_balls_99 (boxes : Fin 99 → ℕ × ℕ) :
  ∃ subset : Finset (Fin 99), subset.card = 50 ∧ 
  (∑ i in subset, (boxes i).1) ≥ (∑ i, (boxes i).1) / 2 ∧
  (∑ i in subset, (boxes i).2) ≥ (∑ i, (boxes i).2) / 2 := 
    sorry

end choose_50_boxes_contains_half_balls_100_choose_50_boxes_contains_half_balls_99_l720_720359


namespace factor_poly_l720_720255

theorem factor_poly (x : ℤ) :
  (x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 - x^9 + x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1)) :=
by
  sorry

end factor_poly_l720_720255


namespace surface_area_tetrahedron_PABCD_l720_720376

-- Definitions based on the conditions (step a)
def side_length : ℝ := 3
def height_PD : ℝ := 4
def base_area : ℝ := side_length^2 -- 9

-- We use Pythagorean theorem to find the height of the lateral faces
def height_PA : ℝ := real.sqrt (height_PD^2 - side_length^2) -- √7

-- Area of each lateral face (each right-angled triangle)
def lateral_area : ℝ := 1/2 * side_length * height_PA -- 3/2 * √7

-- Total surface area based on the solution (step b)
def total_surface_area : ℝ := base_area + 4 * lateral_area -- 9 + 6√7

-- Theorem statement
theorem surface_area_tetrahedron_PABCD : total_surface_area = 9 + 6 * real.sqrt 7 :=
sorry

end surface_area_tetrahedron_PABCD_l720_720376


namespace jar_water_fraction_l720_720674

theorem jar_water_fraction
  (S L : ℝ)
  (h1 : S = (1 / 5) * S)
  (h2 : S = x * L)
  (h3 : (1 / 5) * S + x * L = (2 / 5) * L) :
  x = (1 / 10) :=
by
  sorry

end jar_water_fraction_l720_720674


namespace smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l720_720695

theorem smallest_five_digit_number_divisible_by_prime_2_3_5_7_11 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l720_720695


namespace smallest_five_digit_divisible_by_primes_l720_720709

theorem smallest_five_digit_divisible_by_primes : 
  let primes := [2, 3, 5, 7, 11] in
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11))) in
  let five_digit_threshold := 10000 in
  ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
by
  let primes := [2, 3, 5, 7, 11]
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))
  have lcm_2310 : lcm_primes = 2310 := sorry
  let five_digit_threshold := 10000
  have exists_n : ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
    sorry
  exists_intro 5
  have 5_condition : 5 > 0 := sorry
  have 2310_5_condition : 2310 * 5 >= five_digit_threshold := sorry
  have answer : 2310 * 5 = 11550 := sorry
  exact  ⟨5, 5_condition, 2310_5_condition, answer⟩
  exact ⟨5, 5 > 0, 2310 * 5 ≥ 10000, 2310 * 5 = 11550⟩
  sorry

end smallest_five_digit_divisible_by_primes_l720_720709


namespace decimal_to_fraction_l720_720076

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l720_720076


namespace num_pos_divisors_3465_l720_720270

theorem num_pos_divisors_3465 : (Nat.divisors 3465).length = 24 := 
by 
  sorry

end num_pos_divisors_3465_l720_720270


namespace distribution_ways_l720_720543

-- Definitions based on the problem conditions
def num_people : ℕ := 9
def math_books : ℕ := 6
def chinese_books : ℕ := 3
def total_books : ℕ := math_books + chinese_books

-- Prove that the number of ways to distribute the books is equal to C(num_people, chinese_books)
theorem distribution_ways :
  combinatorics.choose num_people chinese_books = combinatorics.choose 9 3 :=
by
  -- Proof skipped
  sorry

end distribution_ways_l720_720543


namespace inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720485

variable {a b : ℝ}

theorem inequality_a (hab : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_b_not_true (hab : a + b > 0) : ¬(a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

theorem inequality_c (hab : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem inequality_d (hab : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

theorem inequality_e_not_true (hab : a + b > 0) : ¬((a − 3) * (b − 3) < a * b) :=
sorry

theorem inequality_f_not_true (hab : a + b > 0) : ¬((a + 2) * (b + 3) > a * b + 5) :=
sorry

end inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720485


namespace fraction_black_after_seven_changes_l720_720457

theorem fraction_black_after_seven_changes (A : ℝ) (initially_black : A > 0) :
  let fraction_black := (8 / 9)^7 in
  fraction_black = 2097152 / 4782969 :=
by
  sorry

end fraction_black_after_seven_changes_l720_720457


namespace rationalize_denominator_l720_720960

theorem rationalize_denominator : 
  (1 : ℚ) / (real.cbrt 2 + real.cbrt 16) = real.cbrt 4 / 6 := by
  sorry

end rationalize_denominator_l720_720960


namespace problem_translation_l720_720539

-- Definition of values and conditions
def values_silver := ∃ (x y : ℕ), 5 * x + 2 * y = 19 ∧ 2 * x + 5 * y = 16

-- Definition of individual value solutions
def value_cow_and_sheep := ∃ (x y : ℕ), x = 3 ∧ y = 2

-- Definition of purchase solutions
def purchase_silver := 
  let x := 3 in
  let y := 2 in
  ∃ (SOLS : list (ℕ × ℕ)),
    SOLS = [(2, 7), (4, 4), (6, 1)] ∧
    ∀ (m n : ℕ), (m, n) ∈ SOLS → 3 * m + 2 * n = 20

-- Theorem stating the proof problem
theorem problem_translation : 
  values_silver →
  (value_cow_and_sheep ∧ purchase_silver) :=
sorry

end problem_translation_l720_720539


namespace percentage_saved_l720_720633

theorem percentage_saved (rent milk groceries education petrol misc savings : ℝ) 
  (salary : ℝ) 
  (h_rent : rent = 5000) 
  (h_milk : milk = 1500) 
  (h_groceries : groceries = 4500) 
  (h_education : education = 2500) 
  (h_petrol : petrol = 2000) 
  (h_misc : misc = 700) 
  (h_savings : savings = 1800) 
  (h_salary : salary = rent + milk + groceries + education + petrol + misc + savings) : 
  (savings / salary) * 100 = 10 :=
by
  sorry

end percentage_saved_l720_720633


namespace find_number_of_hens_l720_720163

def hens_and_cows_problem (H C : ℕ) : Prop :=
  (H + C = 50) ∧ (2 * H + 4 * C = 144)

theorem find_number_of_hens (H C : ℕ) (hc : hens_and_cows_problem H C) : H = 28 :=
by {
  -- We assume the problem conditions and skip the proof using sorry
  sorry
}

end find_number_of_hens_l720_720163


namespace problem1_problem2_problem3_problem4_problem5_problem6_l720_720521

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end problem1_problem2_problem3_problem4_problem5_problem6_l720_720521


namespace race_minimum_distance_avoiding_river_l720_720992

variable (A B : Point)
variable (C : Point) -- where the runner touches the wall
variable (A' B' : Point) -- adjusted points
variable (wall_length : ℝ)
variable (river_depth : ℝ)
variable (river_length : ℝ)
variable (initial_to_river : ℝ)
variable (river_width : ℝ)
variable (displacement : ℝ)
variable (vertical_distance : ℝ)
variable (A'_B' : ℝ)

def minimumDistance (A B C A' B' : Point) (wall_length river_depth river_length initial_to_river river_width displacement vertical_distance : ℝ) : Prop :=
  A'_B' = sqrt((wall_length + displacement)^2 + vertical_distance^2)

theorem race_minimum_distance_avoiding_river :
  minimumDistance A B C A' B' 1300 100 200 100 50 150 800 = 1570 :=
sorry

end race_minimum_distance_avoiding_river_l720_720992


namespace hexagon_longest_side_l720_720585

theorem hexagon_longest_side (x : ℝ) (h₁ : 6 * x = 20) (h₂ : x < 20 - x) : (10 / 3) ≤ x ∧ x < 10 :=
sorry

end hexagon_longest_side_l720_720585


namespace unique_function_solution_l720_720886

theorem unique_function_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f(x + f(x + y)) + f(x * y) = x + f(x + y) + y * f(x)) :
  (f = fun x => x) ∨ (f = fun x => 2 - x) :=
sorry

end unique_function_solution_l720_720886


namespace max_chord_length_l720_720793

def curve (θ : ℝ) (x y : ℝ) : Prop :=
  2 * (2 * Real.sin θ - Real.cos θ + 3) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0

def line (x y : ℝ) : Prop :=
  y = 2 * x

theorem max_chord_length (θ : ℝ) :
  let line_x := (8 * Real.sin θ + Real.cos θ + 1) / (2 * (2 * Real.sin θ - Real.cos θ + 3))
  let x_max := Real.abs line_x
  let chord_length := 2 * Real.sqrt 2 * x_max
  ∃ θ : ℝ, chord_length = 8 * Real.sqrt 5 :=
sorry

end max_chord_length_l720_720793


namespace closest_fraction_l720_720207

theorem closest_fraction :
  let won_france := (23 : ℝ) / 120
  let fractions := [ (1 : ℝ) / 4, (1 : ℝ) / 5, (1 : ℝ) / 6, (1 : ℝ) / 7, (1 : ℝ) / 8 ]
  ∃ closest : ℝ, closest ∈ fractions ∧ ∀ f ∈ fractions, abs (won_france - closest) ≤ abs (won_france - f)  :=
  sorry

end closest_fraction_l720_720207


namespace smallest_five_digit_number_divisible_by_primes_l720_720730

theorem smallest_five_digit_number_divisible_by_primes : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
begin
  sorry
end

end smallest_five_digit_number_divisible_by_primes_l720_720730


namespace find_x_l720_720316
noncomputable theory

def x_value (a b c d e f : ℝ) : ℝ :=
  ((0.47 * (1442 + a^2)) - (0.36 * (1412 - b^3))) + (65 + c * Real.log d) + e * Real.sin f

theorem find_x : x_value 3 2 8 7 4 (Real.pi / 3) ≈ 261.56138 := sorry

end find_x_l720_720316


namespace ellipse_foci_distance_2sqrt21_l720_720204

noncomputable def ellipse_foci_distance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance_2sqrt21 :
  let center : ℝ × ℝ := (5, 2)
  let a := 5
  let b := 2
  ellipse_foci_distance a b = 2 * Real.sqrt 21 :=
by
  sorry

end ellipse_foci_distance_2sqrt21_l720_720204


namespace petya_time_comparison_l720_720225

open Real

noncomputable def petya_planned_time (D V : ℝ) := D / V

noncomputable def petya_actual_time (D V : ℝ) :=
  let V1 := 1.25 * V
  let V2 := 0.80 * V
  let T1 := (D / 2) / V1
  let T2 := (D / 2) / V2
  T1 + T2

theorem petya_time_comparison (D V : ℝ) (hV : V > 0) : 
  petya_actual_time D V > petya_planned_time D V :=
by {
  let T := petya_planned_time D V
  let T_actual := petya_actual_time D V
  have h1 : petya_planned_time D V = D / V, by unfold petya_planned_time
  have h2 : petya_actual_time D V = (D * 41) / (40 * V), by {
      unfold petya_actual_time,
      have h3 : 1.25 * V = 5 * V / 4, by linarith,
      have h4 : 0.80 * V = 4 * V / 5, by linarith,
      rw [h3, h4],
      simp,
      linarith,
  },
  rw h1,
  rw h2,
  have h3 : (41 * D) / (40 * V) > D / V, by linarith,
  exact h3,
}

end petya_time_comparison_l720_720225


namespace train_length_equals_750_l720_720013

theorem train_length_equals_750
  (L : ℕ) -- length of the train in meters
  (v : ℕ) -- speed of the train in m/s
  (t : ℕ) -- time in seconds
  (h1 : v = 25) -- speed is 25 m/s
  (h2 : t = 60) -- time is 60 seconds
  (h3 : 2 * L = v * t) -- total distance covered by the train is 2L (train and platform) and equals speed * time
  : L = 750 := 
sorry

end train_length_equals_750_l720_720013


namespace find_x_log_eq_l720_720257

theorem find_x_log_eq (x : ℝ) : log x 16 = log 64 4 → x = 4096 :=
by
  sorry

end find_x_log_eq_l720_720257


namespace peya_time_comparison_l720_720212

variable (V D : ℝ) (hV : 0 < V) (hD : 0 < D)

def planned_time : ℝ := D / V
def increased_speed : ℝ := 1.25 * V
def decreased_speed : ℝ := 0.80 * V

def first_half_distance : ℝ := D / 2
def second_half_distance : ℝ := D / 2

def time_first_half : ℝ := first_half_distance / increased_speed
def time_second_half : ℝ := second_half_distance / decreased_speed

def actual_time : ℝ := time_first_half + time_second_half

theorem peya_time_comparison : actual_time V D = (41 * D) / (40 * V) > (D / V) :=
by {
  unfold actual_time,
  unfold time_first_half time_second_half,
  unfold first_half_distance second_half_distance,
  unfold increased_speed decreased_speed,
  unfold planned_time,
  sorry
}

end peya_time_comparison_l720_720212


namespace vol_first_body_vol_second_body_vol_third_body_l720_720274

-- Problem 1
theorem vol_first_body:
  let region := {p : ℝ × ℝ × ℝ | 
    (p.1 + p.2 + p.2 = 4) ∧ 
    (0 ≤ p.1 ∧ p.1 ≤ 3) ∧ 
    (0 ≤ p.2 ∧ p.2 ≤ 2) ∧ 
    (0 ≤ p.3 ≤ (4 - p.1 - p.2)) } in
  volume region = 17 / 6 :=
sorry

-- Problem 2
theorem vol_second_body:
  let sphere_cone_intersection := {p : ℝ × ℝ × ℝ | 
    (p.1^2 + p.2^2 + p.3^2 = 2 * p.3) ∧ 
    (p.1^2 + p.2^2 = p.3^2) } in
  volume sphere_cone_intersection = real.pi :=
sorry

-- Problem 3
theorem vol_third_body:
  let paraboloid_plane_intersection := {p : ℝ × ℝ × ℝ | 
    (2 * p.3 = p.1^2 + p.2^2) ∧ 
    (p.2 + p.3 = 4) } in
  volume paraboloid_plane_intersection = 81 * real.pi / 4 :=
sorry

end vol_first_body_vol_second_body_vol_third_body_l720_720274


namespace algebraic_inequality_l720_720766

theorem algebraic_inequality (n : ℕ) (h : n ≥ 3) (x : Fin n → ℝ) (h_nonneg : ∀ i, 0 ≤ x i) :
  (n + 1) * (∑ i, x i) ^ 2 * (∑ i, (x i) ^ 2) + (n - 2) * (∑ i, (x i) ^ 2) ^ 2 ≥ 
  (∑ i, x i) ^ 4 + (2 * n - 2) * (∑ i, x i) * (∑ i, (x i) ^ 3) :=
by
  sorry

end algebraic_inequality_l720_720766


namespace decimal_to_fraction_l720_720090

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l720_720090


namespace find_side_a_l720_720868

variable (a b c A B C : ℝ)
variable (sin cos : ℝ → ℝ)
variable area : ℝ

theorem find_side_a 
  (h1 : cos A * cos A - cos B * cos B + sin C * sin C = sin B * sin C)
  (h2 : sin B * sin C = 1 / 4)
  (h3 : area = 2 * sqrt 3)
  (h4 : area = 1 / 2 * b * c * sin A)
  (h5 : b * c = 8) :
  a = 2 * sqrt 6 :=
  sorry

end find_side_a_l720_720868


namespace decimal_to_fraction_l720_720089

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l720_720089


namespace dodecagon_area_proof_l720_720682

noncomputable def dodecagon_area (r : ℝ) : ℝ := 3 * r^2 * sqrt(3)

theorem dodecagon_area_proof : 100 * dodecagon_area 1 = 300 := by
  sorry

end dodecagon_area_proof_l720_720682


namespace unique_friendly_determination_l720_720413

def is_friendly (a b : ℕ → ℕ) : Prop :=
∀ n : ℕ, ∃ i j : ℕ, n = a i * b j ∧ ∀ (k l : ℕ), n = a k * b l → (i = k ∧ j = l)

theorem unique_friendly_determination {a b c : ℕ → ℕ} 
  (h_friend_a_b : is_friendly a b) 
  (h_friend_a_c : is_friendly a c) :
  b = c :=
sorry

end unique_friendly_determination_l720_720413


namespace smallest_five_digit_divisible_by_primes_l720_720714

theorem smallest_five_digit_divisible_by_primes : 
  let primes := [2, 3, 5, 7, 11] in
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11))) in
  let five_digit_threshold := 10000 in
  ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
by
  let primes := [2, 3, 5, 7, 11]
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))
  have lcm_2310 : lcm_primes = 2310 := sorry
  let five_digit_threshold := 10000
  have exists_n : ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
    sorry
  exists_intro 5
  have 5_condition : 5 > 0 := sorry
  have 2310_5_condition : 2310 * 5 >= five_digit_threshold := sorry
  have answer : 2310 * 5 = 11550 := sorry
  exact  ⟨5, 5_condition, 2310_5_condition, answer⟩
  exact ⟨5, 5 > 0, 2310 * 5 ≥ 10000, 2310 * 5 = 11550⟩
  sorry

end smallest_five_digit_divisible_by_primes_l720_720714


namespace f_increasing_infinite_f_decreasing_infinite_l720_720045

def f (n : ℕ) : ℝ :=
  if n = 0 then 0
  else (1 / n) * (List.sum (List.map (λ k, real.floor (n / k)) (List.range n)))

theorem f_increasing_infinite :
  ¬ (∃ N : ℕ, ∀ n ≥ N, f (n + 1) ≤ f n) :=
begin
  sorry
end

theorem f_decreasing_infinite :
  ¬ (∃ N : ℕ, ∀ n ≥ N, f (n + 1) ≥ f n) :=
begin
  sorry
end

end f_increasing_infinite_f_decreasing_infinite_l720_720045


namespace find_m_l720_720744

-- Define the pattern of splitting cubes into odd numbers
def split_cubes (m : ℕ) : List ℕ := 
  let rec odd_numbers (n : ℕ) : List ℕ :=
    if n = 0 then []
    else (2 * n - 1) :: odd_numbers (n - 1)
  odd_numbers m

-- Define the condition that 59 is part of the split numbers of m^3
def is_split_number (m : ℕ) (n : ℕ) : Prop :=
  n ∈ (split_cubes m)

-- Prove that if 59 is part of the split numbers of m^3, then m = 8
theorem find_m (m : ℕ) (h : is_split_number m 59) : m = 8 := 
sorry

end find_m_l720_720744


namespace problem1_problem2_problem3_problem4_problem5_problem6_l720_720518

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end problem1_problem2_problem3_problem4_problem5_problem6_l720_720518


namespace bijection_lcm_property_l720_720882

noncomputable def bijective_function {n : ℕ} : Fin n → Fin n := sorry

theorem bijection_lcm_property (n : ℕ) (f : Fin n → Fin n) (hf : bijective f) :
  ∃ M : ℕ, M > 0 ∧ ∀ i : Fin n, iterate f M i = f i :=
sorry

end bijection_lcm_property_l720_720882


namespace exists_composite_for_mul_add_one_infinite_composite_for_mul_add_one_l720_720956

-- Auxiliary definition to identify composite numbers
def is_composite (n : ℕ) : Prop := ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ n = p * q

-- Statement for part (a)
theorem exists_composite_for_mul_add_one (a : ℕ) : ∃ x : ℕ, is_composite (a * x + 1) := 
sorry

-- Statement for part (b)
theorem infinite_composite_for_mul_add_one (a : ℕ) : ∃ S : set ℕ, S.infinite ∧ ∀ x ∈ S, is_composite (a * x + 1) := 
sorry

end exists_composite_for_mul_add_one_infinite_composite_for_mul_add_one_l720_720956


namespace decimal_to_fraction_l720_720087

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l720_720087


namespace decimal_to_fraction_l720_720086

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l720_720086


namespace petya_time_comparison_l720_720217

variables (D V : ℝ) (hD_pos : D > 0) (hV_pos : V > 0)

theorem petya_time_comparison (hD_pos : D > 0) (hV_pos : V > 0) :
  (41 * D / (40 * V)) > (D / V) :=
by
  sorry

end petya_time_comparison_l720_720217


namespace inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720488

variable {a b : ℝ}

theorem inequality_a (hab : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_b_not_true (hab : a + b > 0) : ¬(a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

theorem inequality_c (hab : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem inequality_d (hab : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

theorem inequality_e_not_true (hab : a + b > 0) : ¬((a − 3) * (b − 3) < a * b) :=
sorry

theorem inequality_f_not_true (hab : a + b > 0) : ¬((a + 2) * (b + 3) > a * b + 5) :=
sorry

end inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720488


namespace range_of_a_l720_720291

variable (a : ℝ)

theorem range_of_a (h : ∀ x : ℤ, 2 * (x:ℝ)^2 - 17 * x + a ≤ 0 →  (x = 3 ∨ x = 4 ∨ x = 5)) : 
  30 < a ∧ a ≤ 33 :=
sorry

end range_of_a_l720_720291


namespace two_point_three_five_as_fraction_l720_720071

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l720_720071


namespace abc_positive_and_triangle_possible_l720_720405

theorem abc_positive_and_triangle_possible (a b c : ℝ) (α β : ℝ) 
  (h1 : β ≠ 0) (h2 : 0 < α) 
  (h3 : (α + β * Complex.i)) * (α - β * Complex.i) = (α^2 + β^2) 
  (h4 : x^2 - (a + b + c) * x + ab + bc + ca = 0) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ (√a + √b > √c) ∧ (√a + √c > √b) ∧ (√b + √c > √a) :=
by
  sorry

end abc_positive_and_triangle_possible_l720_720405


namespace smallest_five_digit_number_divisible_by_five_primes_l720_720726

theorem smallest_five_digit_number_divisible_by_five_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  lcm = 2310 → (∃ n : ℕ, n = 5 ∧ 10000 ≤ lcm * n ∧ lcm * n = 11550) :=
by
  intros p1 p2 p3 p4 p5 
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  intro hlcm
  use (5 : ℕ)
  split
  { exact rfl }
  split
  { sorry }
  { sorry }

end smallest_five_digit_number_divisible_by_five_primes_l720_720726


namespace slope_transformation_l720_720778

theorem slope_transformation :
  ∀ (b : ℝ), ∃ k : ℝ, 
  (∀ x : ℝ, k * x + b = k * (x + 4) + b + 1) → k = -1/4 :=
by
  intros b
  use -1/4
  intros h
  sorry

end slope_transformation_l720_720778


namespace sqrt_D_always_irrational_l720_720408

-- Definitions for consecutive even integers and D
def is_consecutive_even (p q : ℤ) : Prop :=
  ∃ k : ℤ, p = 2 * k ∧ q = 2 * k + 2

def D (p q : ℤ) : ℤ :=
  p^2 + q^2 + p * q^2

-- The main statement to prove
theorem sqrt_D_always_irrational (p q : ℤ) (h : is_consecutive_even p q) :
  ¬ ∃ r : ℤ, r * r = D p q :=
sorry

end sqrt_D_always_irrational_l720_720408


namespace parallel_lines_l720_720951

-- Definition of points and lines
variables {A B C M N P Q I Z : Type} [geometry_space : Geometry A B C M N P Q I Z]

-- Conditions
def is_center_of_circumcircle (A B C I : Type) : Prop := Geometry.is_center_of_circumcircle A B C I
def midpoint_of_arc_AC (A C I : Type) : Prop := Geometry.midpoint_ω A C I
def is_angle_bisector (IZ : Type) (B : Type) : Prop := Geometry.angle_bisector IZ B
def is_perpendicular (PQ IZ : Type) : Prop := Geometry.perpendicular PQ IZ
def is_isosceles (MBI NBI BI : Type) : Prop := Geometry.isosceles MBI NBI BI

-- Problem Statement
theorem parallel_lines (A B C M N P Q I Z : Type)
  [is_center_of_circumcircle A B C I]
  [midpoint_of_arc_AC A C I]
  [is_angle_bisector IZ B]
  [is_perpendicular PQ IZ]
  [is_isosceles MBI NBI BI] : PQ ∥ MN :=
by
  sorry

end parallel_lines_l720_720951


namespace probability_of_six_being_largest_l720_720151

noncomputable def probability_six_is_largest : ℚ := sorry

theorem probability_of_six_being_largest (cards : Finset ℕ) (selected_cards : Finset ℕ) :
  cards = {1, 2, 3, 4, 5, 6, 7} →
  selected_cards ⊆ cards →
  selected_cards.card = 4 →
  (probability_six_is_largest = 2 / 7) := sorry

end probability_of_six_being_largest_l720_720151


namespace ratio_cost_to_marked_price_l720_720164

theorem ratio_cost_to_marked_price (p : ℝ) (hp : p > 0) :
  let selling_price := (3 / 4) * p
  let cost_price := (5 / 6) * selling_price
  cost_price / p = 5 / 8 :=
by 
  sorry

end ratio_cost_to_marked_price_l720_720164


namespace sum_of_exterior_angles_of_regular_pentagon_l720_720999

theorem sum_of_exterior_angles_of_regular_pentagon : ∀ (P : Type) [polygon P] (h : sides P = 5), sum_exterior_angles P = 360 :=
by
  assume P
  assume _ : polygon P
  assume h : sides P = 5
  sorry

end sum_of_exterior_angles_of_regular_pentagon_l720_720999


namespace square_area_in_ellipse_l720_720623

theorem square_area_in_ellipse : ∀ (s : ℝ), 
  (s > 0) → 
  (∀ x y, (x = s ∨ x = -s) ∧ (y = s ∨ y = -s) → (x^2) / 4 + (y^2) / 8 = 1) → 
  (2 * s)^2 = 32 / 3 := by
  sorry

end square_area_in_ellipse_l720_720623


namespace coefficient_x3_in_product_l720_720118

def P (x : ℝ) : ℝ := x^5 - 4*x^3 + 3*x^2 - 2*x + 1
def Q (x : ℝ) : ℝ := 3*x^3 - 2*x^2 + x + 5

theorem coefficient_x3_in_product : 
  ∀ (x : ℝ), 
  (P x * Q x).coeff 3 = -10 :=
by 
  intro x
  sorry

end coefficient_x3_in_product_l720_720118


namespace find_simple_solutions_transform_equation_infinite_solutions_l720_720275

open int

axiom x y : ℕ
axiom equation_holds : (x - 1)^2 + (x + 1)^2 = y^2 + 1

theorem find_simple_solutions :
  ((x = 0 ∧ y = 1) ∨ (x = 2 ∧ y = 3)) →
  ∃ (x y : ℕ), (x - 1)^2 + (x + 1)^2 = y^2 + 1 :=
by
  intros h
  cases h
  case inl h1 =>
    existsi 0, 1
    simp [h1]
  case inr h2 =>
    existsi 2, 3
    simp [h2]

theorem transform_equation (x y : ℕ) :
  (x - 1)^2 + (x + 1)^2 = y^2 + 1 → (3 * x + 2 * y - 1)^2 + (3 * x + 2 * y + 1)^2 = (4 * x + 3 * y)^2 + 1 :=
sorry

theorem infinite_solutions :
  ∃ (f : ℕ → ℕ × ℕ), ∀ n, let (x, y) := f n in (x, y) is a solution :=
sorry

end find_simple_solutions_transform_equation_infinite_solutions_l720_720275


namespace MN_parallel_PQ_l720_720931

-- Define the given geometric entities and properties
variables (A B C I M N P Q Z : Type) [Inhabited A]

-- Definition of points being collinear
def collinear (a b c : Type) := ∃ (r : ℝ), r • (b - a) + a = c

-- Definition of parallel lines
def parallel (l1 l2 : Type) : Prop := 
  ∃ p1 p2 p3 p4 : Type, collinear p1 p2 p3 → collinear p2 p3 p4

-- Definition points on a circle
def on_circumcircle (A B C : Type) : Prop := sorry

-- Definition of angle bisector
def angle_bisector (I A B : Type) : Prop := sorry

-- Definition of perpendicular lines
def perpendicular (l1 l2 : Type) : Prop := sorry

-- Main theorem statement
theorem MN_parallel_PQ 
  (h1 : on_circumcircle A B C) 
  (h2 : angle_bisector I A B) 
  (h3 : perpendicular PQ IZ) 
  (h4 : perpendicular PQ BI) 
  (h5 : perpendicular MN BI) : parallel MN PQ :=
sorry

end MN_parallel_PQ_l720_720931


namespace largest_y_diff_zero_l720_720241

theorem largest_y_diff_zero :
  let f := λ x : ℝ, 4 - x^2 + x^4
  let g := λ x : ℝ, 2 + x^2 + x^4
  let intersections := {x : ℝ | f x = g x}
  let y_vals := {y | ∃ x ∈ intersections, y = f x}
  (Sup y_vals - Inf y_vals) = 0 :=
by
  sorry

end largest_y_diff_zero_l720_720241


namespace find_m_l720_720977

-- Define the conditions
variables {m x1 x2 : ℝ}

-- Given the equation x^2 + mx - 1 = 0 has roots x1 and x2:
-- The sum of the roots x1 + x2 is -m, and the product of the roots x1 * x2 is -1.
-- Furthermore, given that 1/x1 + 1/x2 = -3,
-- Prove that m = -3.

theorem find_m :
  (x1 + x2 = -m) →
  (x1 * x2 = -1) →
  (1 / x1 + 1 / x2 = -3) →
  m = -3 := by
  intros hSum hProd hRecip
  sorry

end find_m_l720_720977


namespace smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l720_720699

theorem smallest_five_digit_number_divisible_by_prime_2_3_5_7_11 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l720_720699


namespace math_problem_l720_720472

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end math_problem_l720_720472


namespace if_a_greater_b_then_ac_square_greater_bc_square_l720_720248

theorem if_a_greater_b_then_ac_square_greater_bc_square (a b c : ℝ) (h : a > b) : ac^2 > bc^2 :=
by {
  sorry,
}

end if_a_greater_b_then_ac_square_greater_bc_square_l720_720248


namespace bill_annual_healthcare_cost_l720_720229

def monthly_price : ℕ := 500

def bill_hourly_wage : ℕ := 25
def bill_hours_per_week : ℕ := 30
def weeks_per_month : ℕ := 4

def bill_age : ℕ := 38
def bill_family_size : ℕ := 3

def calculate_monthly_income (wage hours weeks : ℕ) : ℕ :=
  wage * hours * weeks

def calculate_annual_income (monthly_income : ℕ) : ℕ :=
  monthly_income * 12

def contribution_percentage (income age family_size : ℕ) : ℕ :=
  let income_contribution :=
    if income < 10000 then 90
    else if income ≤ 25000 then 75
    else if income ≤ 40000 then 50
    else if income ≤ 55000 then 35
    else if income ≤ 70000 then 20
    else 10
  let age_contribution :=
    if age > 55 then 10
    else if age > 45 then 5
    else 0
  let family_size_contribution :=
    if family_size >= 2 then 10
    else if family_size = 1 then 5
    else 0
  income_contribution + age_contribution + family_size_contribution

def calculate_monthly_payment (monthly_price contribution : ℕ) : ℕ :=
  monthly_price * (100 - contribution) / 100

def calculate_annual_payment (monthly_payment : ℕ) : ℕ :=
  monthly_payment * 12

theorem bill_annual_healthcare_cost :
  let monthly_income := calculate_monthly_income bill_hourly_wage bill_hours_per_week weeks_per_month in
  let annual_income := calculate_annual_income monthly_income in
  let contribution := contribution_percentage annual_income bill_age bill_family_size in
  let monthly_payment := calculate_monthly_payment monthly_price contribution in
  calculate_annual_payment monthly_payment = 2400 :=
by sorry

end bill_annual_healthcare_cost_l720_720229


namespace height_difference_correct_l720_720555

-- Definitions of given conditions
variables (d : ℝ) (num_pipes : ℕ) (rows_A : ℕ) (rows_B_sep : ℕ)

-- Given constants
def diameter : ℝ := 12
def pipes_count : ℕ := 200
def rows_in_A : ℕ := 20
def effective_row_separation_in_B : ℕ := 11 

-- Heights calculations
def height_A : ℝ := rows_in_A * diameter
def height_B : ℝ := (rows_B_sep * 6 * Real.sqrt 3) + diameter

-- Theorem Statement
theorem height_difference_correct :
  height_A - height_B = 228 - 66 * (Real.sqrt 3) :=
by
  -- You can add the proof here.
  sorry

end height_difference_correct_l720_720555


namespace Kim_has_4_cousins_l720_720877

noncomputable def pieces_per_cousin : ℕ := 5
noncomputable def total_pieces : ℕ := 20
noncomputable def cousins : ℕ := total_pieces / pieces_per_cousin

theorem Kim_has_4_cousins : cousins = 4 := 
by
  show cousins = 4
  sorry

end Kim_has_4_cousins_l720_720877


namespace tomato_picking_ratio_l720_720185

theorem tomato_picking_ratio
  (initial_tomatoes : ℕ)
  (first_week_fraction : ℚ)
  (second_week_pick : ℕ)
  (remaining_tomatoes : ℕ)
  (ratio : ℚ) :
  initial_tomatoes = 100 →
  first_week_fraction = 1 / 4 →
  second_week_pick = 20 →
  remaining_tomatoes = 15 →
  let first_week_pick := (first_week_fraction * initial_tomatoes).toNat,
      total_picked := first_week_pick + second_week_pick + (ratio * second_week_pick).toNat in
  total_picked = initial_tomatoes - remaining_tomatoes →
  ratio = 2 := by
  sorry

end tomato_picking_ratio_l720_720185


namespace selling_price_is_60_cents_l720_720387

-- Define the buying cost of oranges
def cost_per_orange (total_cost : ℝ) (total_oranges : ℕ) : ℝ :=
  total_cost / total_oranges

-- Define the selling price considering profit
def selling_price (cost : ℝ) (profit : ℝ) : ℝ :=
  cost + profit

-- Define conversion from dollars to cents
def dollars_to_cents (d : ℝ) : ℕ :=
  (d * 100).toNat

-- Given conditions
def total_cost : ℝ := 12.5
def total_oranges : ℕ := 25
def profit_per_orange : ℝ := 0.1

-- Calculate the selling price in cents
theorem selling_price_is_60_cents :
  dollars_to_cents (selling_price (cost_per_orange total_cost total_oranges) profit_per_orange) = 60 :=
by
  sorry

end selling_price_is_60_cents_l720_720387


namespace geom_seq_ineq_l720_720398

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

variables {a : ℕ → ℝ} {r : ℝ}

theorem geom_seq_ineq (h1: geometric_sequence a r) (h2: ∀ n, a n > 0) (h3: a 0 ≤ r) :
  ∀ n : ℕ, n > 0 → (∏ i in finset.range n, (a (i + 1) ^ (1 / (i + 2)))) ≥ 
           (real.sqrt ((a 1) ^ (1 / 2) * (a n) ^ (1 / (n + 1)))) ^ n :=
by
  sorry

end geom_seq_ineq_l720_720398


namespace bags_of_chips_count_l720_720540

theorem bags_of_chips_count :
  ∃ n : ℕ, n * 400 + 4 * 50 = 2200 ∧ n = 5 :=
by {
  sorry
}

end bags_of_chips_count_l720_720540


namespace MN_parallel_PQ_l720_720928

theorem MN_parallel_PQ
  (A B C I Z M N P Q : Point)
  (circumcircle : Circle)
  (h1 : I = circumcenter A B C)
  (h2 : midpoint I (arc A C circumcircle))
  (h3 : angle_bisector I Z (angle B A C))
  (h4 : perpendicular PQ IZ)
  (h5 : isosceles MBI with base BI)
  (h6 : isosceles NBI with base BI)
  : parallel PQ MN :=
sorry -- Proof to be filled in later by the user

end MN_parallel_PQ_l720_720928


namespace factorial_geometric_sequence_l720_720743

open Nat

def primality (p : ℕ) : Prop := p > 1 ∧ (∀ d : ℕ, d ∣ p → d = 1 ∨ d = p)

noncomputable def prime_exponent (n : ℕ) (p : ℕ) [fact (prime p)] : ℕ :=
∑ k in range (n.log p + 1), n / p^k

def primes_upto (n : ℕ) : List ℕ :=
(List.range (n + 1)).filter primality

def exponents (n : ℕ) : List ℕ :=
primes_upto (n).map (prime_exponent n)

def is_geometric_sequence (l : List ℕ) : Prop :=
∃ r : ℕ, ∀ i j k : ℕ, i < j ∧ j < k → l.getOrElse i 0 * l.getOrElse k 0 = (l.getOrElse j 0)^2

theorem factorial_geometric_sequence (n : ℕ) (h : n ≥ 3) :
  exponents n = [1,2,1] ∨ exponents n = [2,3,1] ∨ exponents n = [1,1,1] ∨ exponents n = [1,3,1] :=
sorry

end factorial_geometric_sequence_l720_720743


namespace decimal_to_fraction_l720_720108

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l720_720108


namespace problem_a_problem_c_problem_d_l720_720504

variables (a b : ℝ)

-- Given condition
def condition : Prop := a + b > 0

-- Proof problems
theorem problem_a (h : condition a b) : a^5 * b^2 + a^4 * b^3 ≥ 0 := sorry

theorem problem_c (h : condition a b) : a^21 + b^21 > 0 := sorry

theorem problem_d (h : condition a b) : (a + 2) * (b + 2) > a * b := sorry

end problem_a_problem_c_problem_d_l720_720504


namespace hyperbola_quadrilateral_area_l720_720759

theorem hyperbola_quadrilateral_area :
  ∀ (k : ℝ) (y₀ : ℝ), 
  (frac x ^ 2 / (16 + k) - frac y ^ 2 / (8 - k) = 1) →
  (-16 < k ∧ k < 8) →
  (y = -sqrt(3) * x) →
  ((3, y₀) is_on_hyperbola) →
  (symmetric_points_about_origin (3, y₀) Q) →
  area_of_quadrilateral F₁ Q F₂ P = 12 * sqrt(6) :=
by
  sorry

end hyperbola_quadrilateral_area_l720_720759


namespace isabella_paint_area_l720_720871

theorem isabella_paint_area 
    (bedrooms : ℕ) 
    (length width height doorway_window_area : ℕ) 
    (h1 : bedrooms = 4) 
    (h2 : length = 14) 
    (h3 : width = 12) 
    (h4 : height = 9)
    (h5 : doorway_window_area = 80) :
    (2 * (length * height) + 2 * (width * height) - doorway_window_area) * bedrooms = 1552 := by
       -- Calculate the area of the walls in one bedroom
       -- 2 * (length * height) + 2 * (width * height) - doorway_window_area = 388
       -- The total paintable area for 4 bedrooms = 388 * 4 = 1552
       sorry

end isabella_paint_area_l720_720871


namespace equivalent_operation_l720_720607

theorem equivalent_operation (x : ℚ) : 
  (x * (2 / 3)) / (4 / 7) = x * (7 / 6) :=
by sorry

end equivalent_operation_l720_720607


namespace length_OM_l720_720155

variables {O A B X C D M : Point}

-- Circle inscribed in an angle at vertex O touches sides at A and B
axiom circle_inscribed (O A B : Point) (α : Angle) : 
exists (circle : Circle), 
  (circle.tangent O A) ∧ 
  (circle.tangent O B) ∧ 
  (circle.contains α)

-- OX intersects the circle at points C and D
axiom ray_intersects_circle (O X C D : Point) (circle : Circle) :
  (ray O X).intersects circle [C, D] ∧
  (distance O C = 1) ∧ 
  (distance C D = 1)

-- M is the intersection point of the ray OX and the segment AB
axiom intersection_point (O X A B M : Point): 
  (on_ray O X M) ∧ 
  (on_segment A B M)

theorem length_OM {O A B X C D M : Point} 
  (circle : Circle)
  (h1 : circle_inscribed O A B (angle A O B)) 
  (h2 : ray_intersects_circle O X C D circle)
  (h3 : intersection_point O X A B M) :
  distance O M = 4/3 :=
sorry

end length_OM_l720_720155


namespace increase_mean_by_15_l720_720348

-- Define the original mean and the sum
variable (a : Fin 12 → ℝ)

def arithmetic_mean (a : Fin 12 → ℝ) : ℝ :=
  (∑ i, a i) / 12

-- Increase each number by 15 and calculate the new mean
def new_arithmetic_mean (a : Fin 12 → ℝ) : ℝ :=
  (∑ i, a i + 15) / 12

-- The problem statement to prove
theorem increase_mean_by_15 (a : Fin 12 → ℝ) :
  new_arithmetic_mean a = arithmetic_mean a + 15 :=
by
  sorry

end increase_mean_by_15_l720_720348


namespace min_AG_magnitude_l720_720923

variable (A B C G : Type) [HilbertSpace A] [HilbertSpace B] [HilbertSpace C]
variables (AB AC : A) 

-- Given conditions
def is_centroid (G A B C : A) : Prop :=
  ∃ (α β : ℝ), G = α * A + β * B ∧ α + β = 1

def angle_A_120 (A B C : A) : Prop :=
  ∃ (θ : ℝ), θ = 120 ∧ ∠BAC = θ

def ab_ac_dot_neg2 (AB AC : A) : Prop :=
  inner AB AC = -2

-- Prove that the minimum value of |AG| is 2/3 given the above conditions
theorem min_AG_magnitude (A B C G : A) (AB AC : A)
  (h1 : is_centroid G A B C)
  (h2 : angle_A_120 A B C)
  (h3 : ab_ac_dot_neg2 AB AC) :
  ∃ (m : ℝ), m = 2/3 ∧ ∥G - A∥ = m :=
sorry

end min_AG_magnitude_l720_720923


namespace ellipse_equation_line_BE_equation_l720_720765

section Problem

variables (a b : ℝ) (h_ab : a > b) (h_b0 : b > 0)
noncomputable def eccentricity : ℝ := (Real.sqrt 2) / 2
variable (h_ecc : (Real.sqrt (a^2 - b^2)) / a = eccentricity)

def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def point_A := (a, 0 : ℝ)
def point_B := (0, -b : ℝ)

def line_AB (x y : ℝ) : Prop := x / a + y / -b = 1
variable (h_dist_O_AB : dist (0,0) (line_AB) = 2 * Real.sqrt 3 / 3)

theorem ellipse_equation : ellipse x y ↔ x^2 / 4 + y^2 / 2 = 1 := sorry

variables (C P : ℝ × ℝ) (h_cp_neq : C ≠ P)
def line_PA (y : ℝ) : Prop := y = 2 * x - 4
variable (h_line_PA : line_PA y)

def vector_CP := (P.1 - C.1, P.2 - C.2)
def vector_BE := (x - (0 : ℝ), y + b)
variable (h_cp_be_orth : vector_CP ⋅ vector_BE = 0)

theorem line_BE_equation : y = 4 * x - Real.sqrt 2 := sorry

end Problem

end ellipse_equation_line_BE_equation_l720_720765


namespace decimal_to_fraction_l720_720088

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l720_720088


namespace find_x_log_eq_l720_720258

theorem find_x_log_eq (x : ℝ) : log x 16 = log 64 4 → x = 4096 :=
by
  sorry

end find_x_log_eq_l720_720258


namespace two_point_three_five_as_fraction_l720_720066

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l720_720066


namespace proposition_1_proposition_2_proposition_3_proposition_4_l720_720282

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) - 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem proposition_1 (x₁ x₂ : ℝ) (h : x₁ - x₂ = Real.pi) : f x₁ = f x₂ :=
sorry

theorem proposition_2 : ¬ (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 3), Function.monotone (fun x => f x)) :=
sorry

theorem proposition_3 : ∃ c, c = (Real.pi / 12, 0) ∧ Function.centrally_symmetric (fun x => (x, f x)) c :=
sorry

theorem proposition_4 : ¬ (∀ x, (f (x + Real.pi / 12 / 5)) = 2 * Real.sin (2 * x)) :=
sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l720_720282


namespace smallest_five_digit_number_divisible_l720_720691

def smallest_prime_divisible (n: ℕ) : Prop :=
  ∃ k: ℕ, n = 2310 * k ∧ 10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_number_divisible :
  ∃ (n: ℕ), smallest_prime_divisible n ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_l720_720691


namespace min_games_needed_l720_720449

theorem min_games_needed (N : ℕ) (initial_tigers_wins : ℕ := 3) (initial_sharks_wins : ℕ := 1) 
    (total_initial_games : ℕ := 4) 
    (total_sharks_games_won := initial_sharks_wins + N) 
    (total_games_played := total_initial_games + N) : 
    total_sharks_games_won ≥ 9 * total_games_played / 10 → N ≥ 26 := 
begin
  sorry
end

end min_games_needed_l720_720449


namespace concurrency_of_ABC_l720_720038

-- Define the setup of the problem
variables (A A' B B' C C' : Type) [has_line A A'] [has_line B B'] [has_line C C']
variables (P1 P2 P3 P4 P5 P6 : Type)
variables [has_segment A B P1] [has_segment C A P3] [has_segment B C P6]
variables [has_segment A C P4] [has_segment B A P2] [has_segment C B P5]
variables [segments_eq : ∀ (x y z : Type), x = y ∧ y = z ∧ z = x → x = z] 

-- Assume all given conditions
variables (h1 : A ⊂ A') (h2 : lines_meet A B P5 C' P1)
variables (h3 : lines_meet A C P3 B' B) (h4 : lines_meet B C P6 A' C)
variables (h5 : equal_segments AP1 AP4 BP2 BP5 CP3 CP6 (BP1 + CP2 + AP3))

-- State the theorem
theorem concurrency_of_ABC {A A' B B' C C' : Type}
    (TriangleABC : in_triangle ABC (A' B' C'))
    (extAB : intersection_of_lines AB C' B') (extAC : intersection_of_lines AC B' A')
    (extBC : intersection_of_lines BC A' C') (segment_conditions : equal_segments
    (AP1 AP4 BP2 BP5 CP3 CP6 (BP1 + CP2 + AP3))) :
    concurrent_lines (AA' BB' CC') :=
by sorry

end concurrency_of_ABC_l720_720038


namespace find_f_log2_20_l720_720458

noncomputable def f : ℝ → ℝ :=
  fun (x : ℝ) => if x ∈ Ioo (-1) 0 then 2^x + 1/5 else sorry

theorem find_f_log2_20 :
  (∀ x : ℝ, f(-x) = -f(x)) →
  (∀ x : ℝ, f(x - 4) = f(x)) →
  (∀ x : ℝ, x ∈ Ioo (-1 : ℝ) 0 → f(x) = 2^x + (1/5 : ℝ)) →
  f(log 20 / log 2) = -1 :=
by
  sorry

end find_f_log2_20_l720_720458


namespace find_q_l720_720549

theorem find_q (p q r : ℕ) (M : ℝ) (hp : p > 1) (hq : q > 1) (hr : r > 1) (hM : M ≠ 1)
  (h_eq : real.rpow M (1 / p + 1 / (p * q) + 1 / (p * q * r)) = real.rpow M (15 / 24)) :
  q = 2 :=
sorry

end find_q_l720_720549


namespace reciprocal_sum_l720_720123

theorem reciprocal_sum (a b c d : ℚ) (h1 : a = 2) (h2 : b = 5) (h3 : c = 3) (h4 : d = 4) : 
  (a / b + c / d)⁻¹ = (20 : ℚ) / 23 := 
by
  sorry

end reciprocal_sum_l720_720123


namespace jungkook_needs_more_paper_l720_720388

def bundles : Nat := 5
def pieces_per_bundle : Nat := 8
def rows : Nat := 9
def sheets_per_row : Nat := 6

def total_pieces : Nat := bundles * pieces_per_bundle
def pieces_needed : Nat := rows * sheets_per_row
def pieces_missing : Nat := pieces_needed - total_pieces

theorem jungkook_needs_more_paper : pieces_missing = 14 := by
  sorry

end jungkook_needs_more_paper_l720_720388


namespace books_leftover_l720_720366

theorem books_leftover :
  (1500 * 45) % 47 = 13 :=
by
  sorry

end books_leftover_l720_720366


namespace decimal_to_fraction_l720_720098

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l720_720098


namespace length_segment_UV_in_right_triangle_l720_720361

theorem length_segment_UV_in_right_triangle 
  {X Y Z : Type} [MetricSpace X] [MetricSpace Y] [MetricSpace Z]
  (XY_length : Real) (XZ_length : Real) (YZ_length : Real)
  (H : XY_length = 13 ∧ XZ_length = 5 ∧ YZ_length = 12) :
  ∃ (U V : Type) (S : Set (MetricSpace.X)),
    S = Metric.Sphere Z (60 / 13) ∧
    (U ∈ S ∧ V ∈ S ∧ U ≠ Z ∧ V ≠ Z ∧
      (Segment XZ).intersectionHi.state.states(U) ∧
      (Segment YZ).intersectionHi.state.states(V) ∧
      dist(U, V) = 120 / 13) :=
sorry

end length_segment_UV_in_right_triangle_l720_720361


namespace sum_rounded_nearest_thousandth_l720_720230

-- Definitions based on the conditions
def num1 : ℝ := 46.129
def num2 : ℝ := 37.9312

-- Proof problem statement
theorem sum_rounded_nearest_thousandth :
  Float.ofReal (num1 + num2) ≈ 84.106 with precision (10 : ℝ)⁻³ :=
by sorry

end sum_rounded_nearest_thousandth_l720_720230


namespace inequality_region_is_lower_left_l720_720010

def lower_left_region (x y : ℝ) : Prop := x + 3 * y - 1 < 0

theorem inequality_region_is_lower_left :
  ∀ x y : ℝ, lower_left_region x y → (x + 3 * y < 1) :=
begin
  sorry
end

end inequality_region_is_lower_left_l720_720010


namespace find_min_value_expression_l720_720736

theorem find_min_value_expression (a b c : ℕ) (hb : b ≠ 0) (ha : a > 0) (hc : c > 0) :
  (a + b ≠ 0) ∧ (b - c ≠ 0) ∧ (c - a ≠ 0) →
  (min ((↑(a + b)^3 + ↑(b - c)^3 + ↑(c - a)^3) / (↑b)^3) = 3.5) :=
by
  sorry

end find_min_value_expression_l720_720736


namespace inequality_a_inequality_c_inequality_d_l720_720491

variable {a b : ℝ}

axiom (h : a + b > 0)

theorem inequality_a : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_c : a^21 + b^21 > 0 :=
sorry

theorem inequality_d : (a + 2) * (b + 2) > a * b :=
sorry

end inequality_a_inequality_c_inequality_d_l720_720491


namespace least_number_to_palindrome_l720_720567

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

theorem least_number_to_palindrome (n : ℕ) : ∃ k : ℕ, (k + n = 124421) ∧ k = 965 := 
by
  sorry

#eval least_number_to_palindrome 123456

end least_number_to_palindrome_l720_720567


namespace decimal_to_fraction_equivalence_l720_720053

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l720_720053


namespace right_triangle_angle_relation_l720_720854

/-- Given:
- \( \triangle ABC \) is a right triangle with \( \angle BAC = 90^\circ \).
- Point \( D \) is on \( AC \) such that \( AD = DC \).
- A line through \( D \) meets \( BC \) at point \( E \).
- \( \angle ADE = x \).
- \( \angle BAE = y \).

Prove that \( x = y \).
-/
theorem right_triangle_angle_relation
  (A B C D E : Type)
  [right_triangle A B C]
  (h1 : ∠BAC = 90)
  (h2 : on_segment D A C)
  (h3 : D.midpoint A C)
  (h4 : on_line D E BC)
  (h5 : ∠ADE = x)
  (h6 : ∠BAE = y) :
  x = y := by
  sorry

end right_triangle_angle_relation_l720_720854


namespace number_of_ordered_pairs_l720_720244

/-- 
Determine the number of ordered pairs (b, c) of positive integers 
for which neither x^2 + bx + c = 0 nor x^2 + cx + b = 0 has two 
distinct real solutions, and both b, c are less than or equal to 5.
--/
theorem number_of_ordered_pairs : 
  (Finset.card 
    (Finset.filter 
      (λ (p : ℕ × ℕ), 
       (1 ≤ p.1 ∧ p.1 ≤ 5) ∧ (1 ≤ p.2 ∧ p.2 ≤ 5) ∧ 
       (p.1^2 - 4 * p.2 ≤ 0) ∧ (p.2^2 - 4 * p.1 ≤ 0)) 
      ((Finset.range 6).product (Finset.range 6)))) = 15 := 
sorry

end number_of_ordered_pairs_l720_720244


namespace smallest_five_digit_number_divisible_l720_720687

def smallest_prime_divisible (n: ℕ) : Prop :=
  ∃ k: ℕ, n = 2310 * k ∧ 10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_number_divisible :
  ∃ (n: ℕ), smallest_prime_divisible n ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_l720_720687


namespace minimum_experiments_fractional_method_l720_720166

/--
A pharmaceutical company needs to optimize the cultivation temperature for a certain medicinal liquid through bioassay.
The experimental range is set from 29℃ to 63℃, with an accuracy requirement of ±1℃.
Prove that the minimum number of experiments required to ensure the best cultivation temperature is found using the fractional method is 7.
-/
theorem minimum_experiments_fractional_method
  (range_start : ℕ)
  (range_end : ℕ)
  (accuracy : ℕ)
  (fractional_method : ∀ (range_start range_end accuracy: ℕ), ℕ) :
  range_start = 29 → range_end = 63 → accuracy = 1 → fractional_method range_start range_end accuracy = 7 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end minimum_experiments_fractional_method_l720_720166


namespace maximum_intersection_points_l720_720423

noncomputable def nine_circles : Prop := 
  ∃ (C : Fin 9 → Circle) (P : Fin 36 → Point) (L : Fin 36 → Line),
    (∀ (i j : Fin 9), i ≠ j → ∃ (p₁ p₂ : Point), C i ∩ C j = {p₁, p₂}) ∧
    ∀ (i j : Fin 9), i ≠ j → ∃ (l : Line), 
      (l = LineThrough (C i ∩ C j).fst (C i ∩ C j).snd) ∧
      (∀ (m n : Fin 36), m ≠ n → L m ≠ L n)

theorem maximum_intersection_points : nine_circles → 
  ∃ mtp, 
    mtp = (Finset.card (Finset.bind (Finset.univ : Finset (Fin 36)) 
                               (λm, Finset.bind (Finset.univ : Finset (Fin 36)) 
                                              (λn, if m ≠ n 
                                                   then (LineThrough ((C m ∩ C n).fst) ((C m ∩ C n).snd)) 
                                                   else ∅)))) - 2 * (Fin.choose 9 3) = 462 :=
by 
  sorry

end maximum_intersection_points_l720_720423


namespace alley_width_is_5_sqrt3_plus_1_l720_720852

-- Define the conditions
def alley_width (w : ℝ) : Prop :=
  ∃ (l : ℝ), l = 10 ∧
  ∃ (h1 : ℝ), h1 = 4 ∧
  ∃ (θ1 : ℝ), θ1 = 30 ∧
  ∃ (h2 : ℝ), h2 = 3 ∧
  ∃ (θ2 : ℝ), θ2 = 120 ∧
  let OA := l * Real.cos (θ1 * Real.pi / 180) in
  let OB := l * Real.cos (θ2 * Real.pi / 180) in
  w = Real.abs (OA + OB)

-- Statement of the proof problem
theorem alley_width_is_5_sqrt3_plus_1 : alley_width (5 * (Real.sqrt 3 + 1)) :=
sorry

end alley_width_is_5_sqrt3_plus_1_l720_720852


namespace problem_a_problem_b_problem_c_l720_720526

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end problem_a_problem_b_problem_c_l720_720526


namespace slope_angle_of_line_l720_720272

theorem slope_angle_of_line (x y : ℝ) (θ : ℝ) : (x - y + 3 = 0) → θ = 45 := 
sorry

end slope_angle_of_line_l720_720272


namespace goldbach_numbers_l720_720660

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def can_be_written_as_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = n

theorem goldbach_numbers : ∀ (n : ℕ), n ∈ {102, 144, 178, 200} → can_be_written_as_sum_of_two_primes n :=
  by
  intros n hn
  rcases hn with rfl | rfl | rfl | rfl | _
  all_goals { -- for each case, we need to show that the number can be written as sum of two primes
    use [5, 97],
    use [7, 137],
    use [5, 173],
    use [3, 197],
    repeat { split; try apply is_prime; try simp } }
  sorry -- skipped proof part for checking each case

end goldbach_numbers_l720_720660


namespace breadth_of_rectangle_l720_720841

theorem breadth_of_rectangle 
  (Perimeter Length Breadth : ℝ)
  (h_perimeter_eq : Perimeter = 2 * (Length + Breadth))
  (h_given_perimeter : Perimeter = 480)
  (h_given_length : Length = 140) :
  Breadth = 100 := 
by
  sorry

end breadth_of_rectangle_l720_720841


namespace f_negative_l720_720306

-- Let f be a function defined on the real numbers
variable (f : ℝ → ℝ)

-- Conditions: f is odd and given form for non-negative x
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom f_positive : ∀ x : ℝ, 0 ≤ x → f x = x^2 - 2 * x

theorem f_negative (x : ℝ) (hx : x < 0) : f x = -x^2 + 2 * x := by
  sorry

end f_negative_l720_720306


namespace sum_first_n_terms_l720_720762

theorem sum_first_n_terms 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (h₀ : a 1 = 2)
  (h₁ : ∀ n, a (n + 1) = 3 * a n + 2)
  (h₂ : ∀ n, S n = ∑ i in finset.range (n + 1), a (i + 1)) :
  ∀ n, S n = (3 ^ (n + 1) - 2 * n - 3) / 2 :=
by
  sorry

end sum_first_n_terms_l720_720762


namespace nguyen_fabric_needs_l720_720916

def yards_to_feet (yards : ℝ) := yards * 3
def total_fabric_needed (pairs : ℝ) (fabric_per_pair : ℝ) := pairs * fabric_per_pair
def fabric_still_needed (total_needed : ℝ) (already_have : ℝ) := total_needed - already_have

theorem nguyen_fabric_needs :
  let pairs := 7
  let fabric_per_pair := 8.5
  let yards_have := 3.5
  let feet_have := yards_to_feet yards_have
  let total_needed := total_fabric_needed pairs fabric_per_pair
  fabric_still_needed total_needed feet_have = 49 :=
by
  sorry

end nguyen_fabric_needs_l720_720916


namespace sin_alpha_in_second_quadrant_l720_720777

theorem sin_alpha_in_second_quadrant 
  (α : ℝ) 
  (h1 : π / 2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.tan α = -1 / 2)  -- tan α = -1/2
  : Real.sin α = Real.sqrt 5 / 5 :=
sorry

end sin_alpha_in_second_quadrant_l720_720777


namespace set_intersection_l720_720288

open Set

variables (U : Set ℝ) (A B : Set ℝ)

def set_problem_conditions :=
  U = univ ∧ A = Ioc (-1 : ℝ) 3 ∧ B = Ici 2

theorem set_intersection (h : set_problem_conditions U A B) : A ∩ (U \ B) = Ioc (-1 : ℝ) 2 :=
by
  unfold set_problem_conditions at h
  cases h with hU hAB
  cases hAB with hA hB
  rw [hU, hA, hB]
  sorry

end set_intersection_l720_720288


namespace maximum_clubs_l720_720365

theorem maximum_clubs (n : ℕ) :
  ∃ k : ℕ, k = ⌊(1/2 : ℝ) + real.sqrt (2 * n + 1/4)⌋ ∧
  (∀ c1 c2 : fin k, ∃ m : fin n, m ∈ c1.members ∧ m ∈ c2.members) ∧
  (∀ c1 c2 c3 : fin k, ¬ (∃ m : fin n, m ∈ c1.members ∧ m ∈ c2.members ∧ m ∈ c3.members)) :=
begin
  sorry
end

end maximum_clubs_l720_720365


namespace valid_three_digit_numbers_count_l720_720336

noncomputable def count_valid_numbers : ℕ :=
  let valid_first_digits := [2, 4, 6, 8].length
  let valid_other_digits := [0, 2, 4, 6, 8].length
  let total_even_digit_3_digit_numbers := valid_first_digits * valid_other_digits * valid_other_digits
  let no_4_or_8_first_digits := [2, 6].length
  let no_4_or_8_other_digits := [0, 2, 6].length
  let numbers_without_4_or_8 := no_4_or_8_first_digits * no_4_or_8_other_digits * no_4_or_8_other_digits
  let numbers_with_4_or_8 := total_even_digit_3_digit_numbers - numbers_without_4_or_8
  let valid_even_sum_count := 50  -- Assumed from the manual checking
  valid_even_sum_count

theorem valid_three_digit_numbers_count :
  count_valid_numbers = 50 :=
by
  sorry

end valid_three_digit_numbers_count_l720_720336


namespace MN_parallel_PQ_l720_720929

theorem MN_parallel_PQ
  (A B C I Z M N P Q : Point)
  (circumcircle : Circle)
  (h1 : I = circumcenter A B C)
  (h2 : midpoint I (arc A C circumcircle))
  (h3 : angle_bisector I Z (angle B A C))
  (h4 : perpendicular PQ IZ)
  (h5 : isosceles MBI with base BI)
  (h6 : isosceles NBI with base BI)
  : parallel PQ MN :=
sorry -- Proof to be filled in later by the user

end MN_parallel_PQ_l720_720929


namespace product_of_ten_proper_fractions_is_one_tenth_l720_720235

theorem product_of_ten_proper_fractions_is_one_tenth 
    (fractions : List (ℚ)) 
    (h_length : fractions.length = 10)
    (h_pos : ∀ (f : ℚ), f ∈ fractions → 0 < f)
    (h_proper : ∀ (f : ℚ), f ∈ fractions → f.num < f.denom) :
        ∃ (fractions : List (ℚ)), (fractions.length = 10) ∧ (∀ f ∈ fractions, 0 < f) ∧ (∀ f ∈ fractions, f.num < f.denom) ∧ (fractions.foldr (*) 1) = (1 / 10) :=
by
  sorry

end product_of_ten_proper_fractions_is_one_tenth_l720_720235


namespace xp1_xp2_xpn_plus_op1n_eq_oxn_l720_720424

open Complex

def nthRootOfUnity (n : ℕ) (k : ℕ) : Complex :=
  exp (2 * Real.pi * Complex.I * (k : ℂ) / (n : ℂ))

noncomputable def complex_distance_product (n : ℕ) (r : ℂ) : ℂ :=
  ∏ k in Finset.range n, (r - nthRootOfUnity n k)

theorem xp1_xp2_xpn_plus_op1n_eq_oxn 
  (n : ℕ) (r : ℂ) (hr : r.re > 1):
  complex_distance_product n r + 1 = r^n := by
  sorry

end xp1_xp2_xpn_plus_op1n_eq_oxn_l720_720424


namespace part_a_winner_part_b_winner_part_c_winner_a_part_c_winner_b_l720_720142

-- Define the game rules and conditions for the proof
def takeMatches (total_matches : Nat) (taken_matches : Nat) : Nat :=
  total_matches - taken_matches

-- Part (a) statement
theorem part_a_winner (total_matches : Nat) (m : Nat) : 
  (total_matches = 25) → (m = 3) → True := 
  sorry

-- Part (b) statement
theorem part_b_winner (total_matches : Nat) (m : Nat) : 
  (total_matches = 25) → (m = 3) → True := 
  sorry

-- Part (c) generalized statement for game type (a)
theorem part_c_winner_a (n : Nat) (m : Nat) : 
  (total_matches = 2 * n + 1) → True :=
  sorry

-- Part (c) generalized statement for game type (b)
theorem part_c_winner_b (n : Nat) (m : Nat) : 
  (total_matches = 2 * n + 1) → True :=
  sorry

end part_a_winner_part_b_winner_part_c_winner_a_part_c_winner_b_l720_720142


namespace smallest_five_digit_divisible_by_primes_l720_720708

theorem smallest_five_digit_divisible_by_primes : 
  let primes := [2, 3, 5, 7, 11] in
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11))) in
  let five_digit_threshold := 10000 in
  ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
by
  let primes := [2, 3, 5, 7, 11]
  let lcm_primes := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))
  have lcm_2310 : lcm_primes = 2310 := sorry
  let five_digit_threshold := 10000
  have exists_n : ∃ n : ℤ, n > 0 ∧ 2310 * n >= five_digit_threshold ∧ 2310 * n = 11550 :=
    sorry
  exists_intro 5
  have 5_condition : 5 > 0 := sorry
  have 2310_5_condition : 2310 * 5 >= five_digit_threshold := sorry
  have answer : 2310 * 5 = 11550 := sorry
  exact  ⟨5, 5_condition, 2310_5_condition, answer⟩
  exact ⟨5, 5 > 0, 2310 * 5 ≥ 10000, 2310 * 5 = 11550⟩
  sorry

end smallest_five_digit_divisible_by_primes_l720_720708


namespace odd_and_increasing_function_l720_720194

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f (x) ≤ f (y)

def function_D (x : ℝ) : ℝ := x * abs x

theorem odd_and_increasing_function : 
  (is_odd function_D) ∧ (is_increasing function_D) :=
sorry

end odd_and_increasing_function_l720_720194


namespace find_lambda_l720_720776

variable {a : ℕ → ℝ} {S : ℕ → ℝ} {λ : ℝ}

-- Condition 1: S_n is the sum of the first n terms of the sequence a_n
def sum_seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in finset.range (n + 1), a i

-- Condition 2: 3a_n = 2S_n + λ * n for all n in ℕ*
def seq_equation (a : ℕ → ℝ) (S : ℕ → ℝ) (λ : ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → 3 * a n = 2 * S n + λ * n

-- Condition 3: The sequence {a_{n+2}} is a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 2 → a n + 2 = r * a (n - 1 + 2)

-- Theorem to prove: λ = 4
theorem find_lambda (h1 : sum_seq a S) (h2 : seq_equation a S λ) (h3 : is_geometric a) : 
  λ = 4 := sorry

end find_lambda_l720_720776


namespace find_m_l720_720866

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x^2)
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -x^2 + m * x

theorem find_m (m : ℝ) :
  (∀ x y, y = f x → deriv f x * (x - 0) + f 0 = y) →
  (∀ x, y = 1 → is_tangent g x 1) →
  m = 2 ∨ m = -2 :=
sorry

end find_m_l720_720866


namespace decimal_to_fraction_l720_720116

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l720_720116


namespace polygon_area_is_correct_l720_720683

open Real

-- Define the vertices of the polygon as vectors
def v1 : ℝ × ℝ := (2, 1)
def v2 : ℝ × ℝ := (4, 3)
def v3 : ℝ × ℝ := (7, 1)
def v4 : ℝ × ℝ := (6, 6)

-- Define a function to calculate the area of a polygon using Shoelace Theorem
def shoelace_area (vertices : List (ℝ × ℝ)) : ℝ :=
  (0.5 * abs 
    (vertices.sum (λi, i.fst * i.snd) 
      + vertices.head.fst * vertices.last.snd 
      - vertices.sum (λi, i.snd * i.fst) 
      - vertices.head.snd * vertices.last.fst))

-- Prove that the area of the given polygon is equal to 7.5
theorem polygon_area_is_correct : shoelace_area [v1, v2, v3, v4] = 7.5 :=
  sorry

end polygon_area_is_correct_l720_720683


namespace sum_of_series_l720_720232

theorem sum_of_series : 
  (∑ k in Finset.range 9, 1 / ((2 * k + 1) * (2 * k + 3))) = 9 / 19 :=
by
  sorry

end sum_of_series_l720_720232


namespace option_C_represents_same_function_l720_720193

-- Definitions of the functions from option C
def f (x : ℝ) := x^2 - 1
def g (t : ℝ) := t^2 - 1

-- The proof statement that needs to be proven
theorem option_C_represents_same_function :
  f = g :=
sorry

end option_C_represents_same_function_l720_720193


namespace parallel_vectors_l720_720452

theorem parallel_vectors : 
  ∀ (a b c : ℝ), 
    let u := (a, b, c) in
    let v1 := (-1, -3, 2) in
    let v2 := (-1/2, 3/2, -1) in
    (∃ k1 : ℝ, v1 = (k1 * a, k1 * b, k1 * c)) ∧ (∃ k2 : ℝ, v2 = (k2 * a, k2 * b, k2 * c)) :=
by
  intros a b c
  let u := (a, b, c)
  let v1 := (-1, -3, 2)
  let v2 := (-1 / 2, 3 / 2, -1)
  split
  all_goals
    use 1 -- This indicates that for v1, k1 = 1 and for v2, k2 = 1.
    sorry

end parallel_vectors_l720_720452


namespace smallest_n_ultra_special_sum_l720_720651

def is_ultra_special (x : ℝ) : Prop := 
  ∃ (digits : ℕ → ℕ), (∀ n, digits n ∈ {0, 2}) ∧ 
  (x = ∑' n, (digits n) * (2⁻¹: ℝ)^(n+1))

theorem smallest_n_ultra_special_sum (n : ℕ) :
  (∀ {x : ℝ}, is_ultra_special x → 0 ≤ x ∧ x < 2) → 
  ∃ l : list ℝ, (∀ x ∈ l, is_ultra_special x) ∧ 
  list.sum l = 2 ∧ l.length = 16 :=
by 
  sorry

end smallest_n_ultra_special_sum_l720_720651


namespace smallest_five_digit_number_divisible_by_primes_l720_720729

theorem smallest_five_digit_number_divisible_by_primes : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
begin
  sorry
end

end smallest_five_digit_number_divisible_by_primes_l720_720729


namespace root_interval_l720_720665
open Real

noncomputable def f (x : ℝ) : ℝ := log x / log 3 + x - 3

theorem root_interval : ∃ c ∈ Ioo (2 : ℝ) 3, f c = 0 :=
by
  have h_cont : ContinuousOn f (Ioo 2 3) := by
    -- Prove continuity, which will require the log function to be defined.
    sorry
  have h_f2 : f 2 < 0 := by
    -- Prove f(2) < 0
    sorry
  have h_f3 : f 3 > 0 := by
    -- Prove f(3) > 0
    sorry
  -- Apply Intermediate Value Theorem
  exact IntermediateValueTheorem h_cont h_f2 h_f3

end root_interval_l720_720665


namespace AlexSumLargerBy105_l720_720634

/-- Definition of Alex's list of numbers -/
def AlexNumbers : List ℕ := List.range' 1 50

/-- Function that replaces digit '3' with digit '2' in a number -/
def replaceDigit (n : ℕ) : ℕ :=
  Nat.digits 10 n |>.reverse |>.map (λ d => if d = 3 then 2 else d) |>.reverse |>.foldl (λ acc d => acc * 10 + d) 0

/-- Definition of Tony's list of numbers -/
def TonyNumbers : List ℕ := AlexNumbers.map replaceDigit

/-- Summing up the numbers in a list -/
def listSum (lst : List ℕ) : ℕ := lst.foldl (λ acc x => acc + x) 0

/-- Theorem stating that the sum of Alex's numbers is 105 larger than Tony's numbers -/
theorem AlexSumLargerBy105 : listSum AlexNumbers - listSum TonyNumbers = 105 := by
  sorry

end AlexSumLargerBy105_l720_720634


namespace map_f_eq_0_neg3_4_neg1_l720_720315

theorem map_f_eq_0_neg3_4_neg1 : 
  ∀ (f : (ℕ, ℕ, ℕ, ℕ) → (ℕ, ℤ, ℕ, ℤ)), 
    (∀ (a_1 a_2 a_3 a_4 b_1 b_2 b_3 b_4: ℤ), 
      (x^4 + a_1 * x^3 + a_2 * x^2 + a_3 * x + a_4 = (x + 1)^4 + b_1 * (x + 1)^3 + b_2 * (x + 1)^2 + b_3 * (x + 1) + b_4) → 
      (b_1 + b_2 + b_3 + b_4 = 0) →
      f (a_1, a_2, a_3, a_4) = (b_1, b_2, b_3, b_4)) →
  f (4, 3, 2, 1) = (0, -3, 4, -1) :=
begin
  sorry
end

end map_f_eq_0_neg3_4_neg1_l720_720315


namespace problem_l720_720301

def p : Prop := 0 % 2 = 0
def q : Prop := ¬(3 % 2 = 0)

theorem problem : p ∨ q :=
by
  sorry

end problem_l720_720301


namespace smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l720_720700

theorem smallest_five_digit_number_divisible_by_prime_2_3_5_7_11 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l720_720700


namespace shaded_region_area_l720_720363

theorem shaded_region_area
  (A B C D E F G I H : Type)
  (square1: square A B C D 5)  -- Condition 1: ABCD is a square with side length 5 cm
  (square2: square C E F G 4)  -- Condition 1: CEFG is a square with side length 4 cm
  (remove_side_gc : remove_side GC)  -- Condition 2: Side GC is removed
  (intersect_AE_BF_at_H : intersect AE BF H)  -- Condition 3: AE and BF intersect at H
  (intersect_BG_AE_at_I : intersect BG AE I)  -- Condition 4: BG and AE intersect at I
  : area_shaded_region = 640 / 549 := sorry  -- The area of the shaded region is 640/549 square cm (to be proved)

end shaded_region_area_l720_720363


namespace jill_total_tax_percentage_l720_720427

variables {total_amount : ℝ} {tax_clothing tax_food tax_items tax_total : ℝ}
variables {percent_clothing percent_food percent_items : ℝ}

/-- 
  Problem statement: 

  Jill spent 50% of her total shopping amount on clothing, 
  20% on food, and 30% on other items. 
  Jill paid a 4% tax on clothing, no tax on food, and an 8% tax on other items. 
  What is the total tax that she paid as a percentage of the total amount that she spent (excluding taxes)?

  Conditions:
  - total_amount spent (excluding taxes) is $100.
  - percent_clothing = 50%, percent_food = 20%, percent_items = 30%
  - tax rate for clothing = 4%, for food = 0%, for other items = 8%
-/
theorem jill_total_tax_percentage
  (total_amount : ℝ)
  (percent_clothing percent_food percent_items : ℝ)
  (tax_clothing tax_food tax_items : ℝ)
  (h_clothing_percentage : percent_clothing = 0.50)
  (h_food_percentage : percent_food = 0.20)
  (h_items_percentage : percent_items = 0.30)
  (h_total_amount : total_amount = 100)
  (h_tax_clothing : tax_clothing = 4 / 100)
  (h_tax_food : tax_food = 0 / 100)
  (h_tax_items : tax_items = 8 / 100):
  let 
    amount_clothing := total_amount * percent_clothing,
    amount_food := total_amount * percent_food,
    amount_items := total_amount * percent_items,
    tax_paid_clothing := amount_clothing * tax_clothing,
    tax_paid_food := amount_food * tax_food,
    tax_paid_items := amount_items * tax_items,
    total_tax := tax_paid_clothing + tax_paid_food + tax_paid_items,
    tax_percentage := (total_tax / total_amount) * 100
  in tax_percentage = 4.40 :=
by
  sorry

end jill_total_tax_percentage_l720_720427


namespace smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l720_720694

theorem smallest_five_digit_number_divisible_by_prime_2_3_5_7_11 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l720_720694


namespace valid_permutations_count_l720_720964

theorem valid_permutations_count (n : ℕ) :
  let odd_count := n / 2,
      even_count := (n + 1) / 2
  in fact odd_count * fact even_count = (List.perm (List.range n)).filter
      (λ perm, ∀ i < n, (List.count (λ j, (j < i ∧ List.nthLe perm j sorry > List.nthLe perm i sorry) ∨ (j > i ∧ List.nthLe perm j sorry < List.nthLe perm i sorry)) (List.range n)) % 2 = 0).length := sorry


end valid_permutations_count_l720_720964


namespace ratio_of_angles_l720_720191

variables {A B C O E : Type} 

def inscribed_triangle (ABC : Triangle) (O : Circle) : Prop :=
  ∃ (A B C : Point), 
  (Triangle A B C = ABC) ∧ 
  (Circle.center O = (circumcenter (Triangle A B C)))

def minor_arc_angle {O : Circle} (A B : Point) : Angle := sorry

def perpendicular {O E A C : Point} : Prop := sorry

def angle_OBE (A B C O E : Type) : Angle := sorry
def angle_BAC (A B C O E : Type) : Angle := sorry

theorem ratio_of_angles (ABC : Triangle) (O : Circle) (E : Point)
  (h_inscribed : inscribed_triangle ABC O) 
  (h_minor_arc_AB : minor_arc_angle O A B = 150)
  (h_minor_arc_BC : minor_arc_angle O B C = 54)
  (h_perpendicular : perpendicular O E A C) : 
  (angle_OBE A B C O E) / (angle_BAC A B C O E) = 5 / 9 :=
sorry

end ratio_of_angles_l720_720191


namespace find_g_inverse_84_l720_720836

-- Definition of the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 3

-- Definition stating the goal
theorem find_g_inverse_84 : g⁻¹ 84 = 3 :=
sorry

end find_g_inverse_84_l720_720836


namespace problem1_problem2_problem3_problem4_problem5_problem6_l720_720515

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end problem1_problem2_problem3_problem4_problem5_problem6_l720_720515


namespace decimal_to_fraction_l720_720063

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l720_720063


namespace min_value_of_sum_range_of_x_l720_720774

noncomputable def ab_condition (a b : ℝ) : Prop := a + b = 1
noncomputable def ra_positive (a b : ℝ) : Prop := a > 0 ∧ b > 0

-- Problem 1: Minimum value of (1/a + 4/b)

theorem min_value_of_sum (a b : ℝ) (h_ab : ab_condition a b) (h_pos : ra_positive a b) : 
    ∃ m : ℝ, m = 9 ∧ ∀ a b, ab_condition a b → ra_positive a b → 
    (1 / a + 4 / b) ≥ m :=
by sorry

-- Problem 2: Range of x for which the inequality holds

theorem range_of_x (a b x : ℝ) (h_ab : ab_condition a b) (h_pos : ra_positive a b) : 
    (1 / a + 4 / b) ≥ |2 * x - 1| - |x + 1| → x ∈ Set.Icc (-7 : ℝ) 11 :=
by sorry

end min_value_of_sum_range_of_x_l720_720774


namespace Age_Of_Antonio_l720_720870

theorem Age_Of_Antonio
  (isabella_future_age : ℝ) (future_months : ℝ)
  (isabella_multiplier : ℝ) (months_in_year : ℝ))
  (isabella_age : ℝ := isabella_future_age - future_months / months_in_year)
  (antonio_age : ℝ := isabella_age / isabella_multiplier)
  (antonio_age_in_months : ℝ := antonio_age * months_in_year) :
  isabella_future_age = 10 ∧ future_months = 18 ∧ isabella_multiplier = 2 ∧ months_in_year = 12 →
  antonio_age_in_months = 51 :=
by
  intros
  sorry

end Age_Of_Antonio_l720_720870


namespace even_degree_partition_l720_720958

theorem even_degree_partition (G : SimpleGraph V) : 
  ∃ (H1 H2 : set V), (H1 ∩ H2 = ∅) ∧ (H1 ∪ H2 = univ) ∧
  (∀ v ∈ V, (degree (G.edge_set \ (H1 ×ˢ H2)) v) % 2 = 0) :=
sorry

end even_degree_partition_l720_720958


namespace find_n_tangent_l720_720685

theorem find_n_tangent (n : ℤ) (h₁ : -180 < n) (h₂ : n < 180) :
  (∀ k : ℤ, tan n = tan (1500 + k * 180)) → n = 60 := by
  sorry

end find_n_tangent_l720_720685


namespace max_distance_sin_cos_l720_720325

open Real

theorem max_distance_sin_cos : ∃ m : ℝ, 
  let f := λ x : ℝ, sin x,
      g := λ x : ℝ, sin (π / 2 - x) in
  (| f m - g m |) = √2 :=
begin
  sorry
end

end max_distance_sin_cos_l720_720325


namespace inscribed_square_area_l720_720625

-- Define the condition of the problem
def inscribed_square_condition (t : ℝ) : Prop :=
(∀ x y, (x = t ∨ x = -t) ∧ (y = t ∨ y = -t) →
( x^2 / 4 + y^2 / 8 = 1 ))

-- The theorem that proves the area of the square inscribed in the ellipse
theorem inscribed_square_area :
  ∃ t : ℝ, inscribed_square_condition t ∧ (2 * t)*(2 * t) = 32 / 3 :=
begin
  sorry
end

end inscribed_square_area_l720_720625


namespace smallest_prime_dividing_sum_l720_720574

theorem smallest_prime_dividing_sum (h1 : Odd 7) (h2 : Odd 9) 
    (h3 : ∀ {a b : ℤ}, Odd a → Odd b → Even (a + b)) :
  ∃ p : ℕ, Prime p ∧ p ∣ (7 ^ 15 + 9 ^ 7) ∧ p = 2 := 
by
  sorry

end smallest_prime_dividing_sum_l720_720574


namespace inequality_a_inequality_c_inequality_d_l720_720495

variable {a b : ℝ}

axiom (h : a + b > 0)

theorem inequality_a : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_c : a^21 + b^21 > 0 :=
sorry

theorem inequality_d : (a + 2) * (b + 2) > a * b :=
sorry

end inequality_a_inequality_c_inequality_d_l720_720495


namespace smallest_five_digit_number_divisible_by_five_primes_l720_720722

theorem smallest_five_digit_number_divisible_by_five_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  lcm = 2310 → (∃ n : ℕ, n = 5 ∧ 10000 ≤ lcm * n ∧ lcm * n = 11550) :=
by
  intros p1 p2 p3 p4 p5 
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  intro hlcm
  use (5 : ℕ)
  split
  { exact rfl }
  split
  { sorry }
  { sorry }

end smallest_five_digit_number_divisible_by_five_primes_l720_720722


namespace probability_log_inequality_l720_720799

noncomputable def f : ℝ → ℝ := λ x, Real.log x / Real.log 2

theorem probability_log_inequality :
  ∃ (p : ℚ), (1 ≤ f x ∧ f x ≤ 2) ∧ x ∈ Set.Icc 1 8 → p = 2 / 7 :=
begin
  sorry
end

end probability_log_inequality_l720_720799


namespace general_term_a_eq_1_geometric_sequence_a_value_sum_of_squares_geo_seq_l720_720816

-- Definitions for the problem
def sequence_sum (a : ℤ) (n : ℕ) : ℤ := 2^n + a

def a_n (a : ℤ) : ℕ → ℤ
| 0     := 0  -- Since sequences usually start from 1 in such problems
| 1     := 2 + a
| (n+2) := (sequence_sum a (n+2)) - (sequence_sum a (n+1))

-- Problem 1: Finding the general term when a = 1
theorem general_term_a_eq_1 :
  (∀ n : ℕ, a_n 1 n = if n = 1 then 3 else if n = 0 then 0 else 2^(n-1)) := 
sorry

-- Problem 2: Finding the value of a when the sequence is geometric
theorem geometric_sequence_a_value (a : ℤ) :
  (∃ r : ℤ, ∀ n : ℕ, a_n a (n+2) = r * a_n a (n+1)) → a = -1 := 
sorry

-- Problem 3: Finding the sum of squares when a = -1 and the sequence is geometric
theorem sum_of_squares_geo_seq (n : ℕ) :
  let a := -1 in 
  (∀ m : ℕ, a_n a m = 2^(m - 1) * a) → 
  a_1^2 + a_2^2 + a_3^2 + ... + a_n^2 = (1/3) * (4^n - 1) := 
sorry

end general_term_a_eq_1_geometric_sequence_a_value_sum_of_squares_geo_seq_l720_720816


namespace find_q_l720_720812

variable (p q : ℝ)
def y : ℝ → ℝ := λ x, x^2 + p * x + (q - p^2 / 4)

theorem find_q (hmin : ∃ x, y p q x = 0) : q = p^2 / 2 :=
  sorry

end find_q_l720_720812


namespace Tony_packs_of_pens_l720_720389

theorem Tony_packs_of_pens (T : ℕ) 
  (Kendra_packs : ℕ := 4) 
  (pens_per_pack : ℕ := 3) 
  (Kendra_keep : ℕ := 2) 
  (Tony_keep : ℕ := 2)
  (friends_pens : ℕ := 14) 
  (total_pens_given : Kendra_packs * pens_per_pack - Kendra_keep + 3 * T - Tony_keep = friends_pens) :
  T = 2 :=
by {
  sorry
}

end Tony_packs_of_pens_l720_720389


namespace range_of_a_l720_720370

theorem range_of_a (a : ℝ) :
  (∃ (M : ℝ × ℝ), (M.1 - a)^2 + (M.2 - a + 2)^2 = 1 ∧
    (M.1)^2 + (M.2 - 2)^2 + (M.1)^2 + (M.2)^2 = 10) → 
  0 ≤ a ∧ a ≤ 3 := 
sorry

end range_of_a_l720_720370


namespace second_smallest_four_prob_l720_720963

/-- Probability that, when selecting 7 distinct integers from the set {1, 2, ..., 12},
    the second smallest number is 4 --/
theorem second_smallest_four_prob :
  let total_ways := Nat.choose 12 7 in
  let valid_ways := (Nat.choose 3 1) * (Nat.choose 8 5) in
  (valid_ways : ℚ) / total_ways = 7 / 33 := by
  sorry

end second_smallest_four_prob_l720_720963


namespace hiker_total_distance_l720_720587

variables (D1 D2 D3 : ℕ) (SP1 SP2 SP3 : ℕ) (T1 T2 T3 : ℕ)

-- Conditions
def condition_1 : D1 = 18 := sorry
def condition_2 : SP1 = 3 := sorry
def condition_3 : T1 = D1 / SP1 := sorry
def condition_4 : SP2 = SP1 + 1 := sorry
def condition_5 : T2 = T1 - 1 := sorry
def condition_6 : D2 = SP2 * T2 := sorry
def condition_7 : SP3 = 5 := sorry
def condition_8 : T3 = 6 := sorry
def condition_9 : D3 = SP3 * T3 := sorry

-- The total distance
def total_distance := D1 + D2 + D3

theorem hiker_total_distance : total_distance D1 D2 D3 = 68 :=
by
  rw [condition_1, condition_2, condition_3, condition_4, condition_5, condition_6, condition_7, condition_8, condition_9]
  sorry

end hiker_total_distance_l720_720587


namespace MN_parallel_PQ_l720_720925

theorem MN_parallel_PQ
  (A B C I Z M N P Q : Point)
  (circumcircle : Circle)
  (h1 : I = circumcenter A B C)
  (h2 : midpoint I (arc A C circumcircle))
  (h3 : angle_bisector I Z (angle B A C))
  (h4 : perpendicular PQ IZ)
  (h5 : isosceles MBI with base BI)
  (h6 : isosceles NBI with base BI)
  : parallel PQ MN :=
sorry -- Proof to be filled in later by the user

end MN_parallel_PQ_l720_720925


namespace decimal_to_fraction_l720_720059

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l720_720059


namespace problem_one_problem_two_l720_720324

-- Problem 1 Statement in Lean 4
theorem problem_one {x : ℝ} (h1 : 0 < x) (h2 : x < 2 / 3) : abs(x + 1) + abs(2 * x - 1) < 2 := 
sorry

-- Problem 2 Statement in Lean 4
theorem problem_two {m n : ℝ} (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n):
  ∃ (a : ℝ), ∀ (x : ℝ), abs(x + a) + abs(2 * x - a) ≤ 1/m + 4/n ∧ |x| ≤ 3 :=
sorry

end problem_one_problem_two_l720_720324


namespace peya_time_comparison_l720_720209

variable (V D : ℝ) (hV : 0 < V) (hD : 0 < D)

def planned_time : ℝ := D / V
def increased_speed : ℝ := 1.25 * V
def decreased_speed : ℝ := 0.80 * V

def first_half_distance : ℝ := D / 2
def second_half_distance : ℝ := D / 2

def time_first_half : ℝ := first_half_distance / increased_speed
def time_second_half : ℝ := second_half_distance / decreased_speed

def actual_time : ℝ := time_first_half + time_second_half

theorem peya_time_comparison : actual_time V D = (41 * D) / (40 * V) > (D / V) :=
by {
  unfold actual_time,
  unfold time_first_half time_second_half,
  unfold first_half_distance second_half_distance,
  unfold increased_speed decreased_speed,
  unfold planned_time,
  sorry
}

end peya_time_comparison_l720_720209


namespace sara_spent_on_hotdog_l720_720438

def total_cost_of_lunch: ℝ := 10.46
def cost_of_salad: ℝ := 5.10
def cost_of_hotdog: ℝ := total_cost_of_lunch - cost_of_salad

theorem sara_spent_on_hotdog :
  cost_of_hotdog = 5.36 := by
  sorry

end sara_spent_on_hotdog_l720_720438


namespace decimal_to_fraction_l720_720062

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l720_720062


namespace main_reason_for_relocation_l720_720120

-- Define the conditions
def xiongAnDevelopment : Prop := 
  "Xiong'an New Area's development direction includes green and ecological city, innovation-driven development, coordinated development, and open development."

def notInLine : Prop := 
  "The traditional shoemaking industry in Santai Town is not in line with Xiong'an New Area's industrial development direction."

def santaiShoeCapital : Prop := 
  "Anxin County's Santai Town is the 'Northern Shoe Capital'."

def notLandCost : Prop := 
  "Increased land cost is not the reason for relocation."

def notMarketPotential : Prop := 
  "Greater market potential in Shijiazhuang is not the reason for relocation."

def notTransport : Prop := 
  "More convenient transportation in Shijiazhuang is not the reason for relocation."

-- The main theorem
theorem main_reason_for_relocation (
  xiongAnDevelopment : xiongAnDevelopment,
  notInLine : notInLine,
  santaiShoeCapital : santaiShoeCapital,
  notLandCost : notLandCost,
  notMarketPotential : notMarketPotential,
  notTransport : notTransport
) : 
  "Industrial structure adjustment in Anxin County is the main reason for the relocation of the 'Northern Shoe Capital' out of Anxin County." :=
  sorry

end main_reason_for_relocation_l720_720120


namespace Xiaoyu_group_A_probability_l720_720416

theorem Xiaoyu_group_A_probability : 
  ∀ (total_students : ℕ) (groups : ℕ) (students_per_group : ℕ) (x: ℕ),
  total_students = 48 ∧ groups = 4 ∧ students_per_group = 12 ∧ x ∈ finset.range total_students →
  (∃ group : ℕ, group < groups ∧ group = 0) →
  (∃ (probability : ℚ), probability = 1 / groups ∧ probability = 1 / 4) :=
begin
  -- sorry
end

end Xiaoyu_group_A_probability_l720_720416


namespace decimal_to_fraction_l720_720115

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l720_720115


namespace petya_time_comparison_l720_720228

open Real

noncomputable def petya_planned_time (D V : ℝ) := D / V

noncomputable def petya_actual_time (D V : ℝ) :=
  let V1 := 1.25 * V
  let V2 := 0.80 * V
  let T1 := (D / 2) / V1
  let T2 := (D / 2) / V2
  T1 + T2

theorem petya_time_comparison (D V : ℝ) (hV : V > 0) : 
  petya_actual_time D V > petya_planned_time D V :=
by {
  let T := petya_planned_time D V
  let T_actual := petya_actual_time D V
  have h1 : petya_planned_time D V = D / V, by unfold petya_planned_time
  have h2 : petya_actual_time D V = (D * 41) / (40 * V), by {
      unfold petya_actual_time,
      have h3 : 1.25 * V = 5 * V / 4, by linarith,
      have h4 : 0.80 * V = 4 * V / 5, by linarith,
      rw [h3, h4],
      simp,
      linarith,
  },
  rw h1,
  rw h2,
  have h3 : (41 * D) / (40 * V) > D / V, by linarith,
  exact h3,
}

end petya_time_comparison_l720_720228


namespace negative_sixty_represents_expenditure_l720_720653

def positive_represents_income (x : ℤ) : Prop := x > 0
def negative_represents_expenditure (x : ℤ) : Prop := x < 0

theorem negative_sixty_represents_expenditure :
  negative_represents_expenditure (-60) ∧ abs (-60) = 60 :=
by
  sorry

end negative_sixty_represents_expenditure_l720_720653


namespace _l720_720534

-- Define the problem conditions
structure PentagonalPrism :=
  (A B C D E A1 B1 C1 D1 E1 : Point)
  (relations : List (Plane × Plane))

-- Define the conjecture structure
structure Conjecture :=
  (types_count : Nat)
  (pieces_count : Nat)

-- The statement of the proof
def pentagonalPrismCuts : Prop :=
  ∀ (prism : PentagonalPrism),
    prism.relations = 
      [(ABD1, BCE1), (CDA1, DEB1), (EAC1, ABD1), (BCE1, CDA1), (DEB1, EAC1)] →
      ∃ (conclusion : Conjecture), 
         conclusion.types_count = 6 ∧ conclusion.pieces_count = 22

#check pentagonalPrismCuts
  
protected theorem prove_pentagonalPrismCuts : pentagonalPrismCuts := 
  sorry

end _l720_720534


namespace sum_even_terms_geometric_sequence_l720_720326
-- Import Mathlib for geometric sequence computation

-- Define the geometric sequence and the sum formula to prove
theorem sum_even_terms_geometric_sequence (n : ℕ) : 
  (∑ k in range n, (2 * 3^(2*k))) = (3/4) * (9^n - 1) := 
by 
  sorry

end sum_even_terms_geometric_sequence_l720_720326


namespace total_students_l720_720536

-- Define the condition that the sum of boys (75) and girls (G) is the total number of students (T)
def sum_boys_girls (G T : ℕ) := 75 + G = T

-- Define the condition that the number of girls (G) equals 75% of the total number of students (T)
def girls_percentage (G T : ℕ) := G = Nat.div (3 * T) 4

-- State the theorem that given the above conditions, the total number of students (T) is 300
theorem total_students (G T : ℕ) (h1 : sum_boys_girls G T) (h2 : girls_percentage G T) : T = 300 := 
sorry

end total_students_l720_720536


namespace cartesian_equation_circle_cartesian_equation_line_intersection_polar_coordinates_l720_720863

noncomputable def circle (ρ θ : ℝ) := ρ = cos θ + sin θ
noncomputable def line (ρ θ : ℝ) := ρ * sin (θ - π / 4) = sqrt 2 / 2

theorem cartesian_equation_circle (x y : ℝ) (ρ θ : ℝ) (h1 : circle ρ θ) :
    x^2 + y^2 - x - y = 0 := sorry

theorem cartesian_equation_line (x y : ℝ) (ρ θ : ℝ) (h2 : line ρ θ) :
    x - y + 1 = 0 := sorry

theorem intersection_polar_coordinates (ρ θ : ℝ) (h1 : circle ρ θ) (h2 : line ρ θ) (hθ : 0 < θ ∧ θ < π) :
    (ρ, θ) = (1, π / 2) := sorry

end cartesian_equation_circle_cartesian_equation_line_intersection_polar_coordinates_l720_720863


namespace neg_p_iff_a_in_0_1_l720_720811

theorem neg_p_iff_a_in_0_1 (a : ℝ) : 
  (¬ (∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0)) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) ∧ (0 < a ∧ a < 1) :=
sorry

end neg_p_iff_a_in_0_1_l720_720811


namespace lottery_most_frequent_number_l720_720581

noncomputable def m (i : ℕ) : ℚ :=
  ((i - 1) * (90 - i) * (89 - i) * (88 - i)) / 6

theorem lottery_most_frequent_number :
  ∀ (i : ℕ), 2 ≤ i ∧ i ≤ 87 → m 23 ≥ m i :=
by 
  sorry -- Proof goes here. This placeholder allows the file to compile.

end lottery_most_frequent_number_l720_720581


namespace point_in_second_quadrant_l720_720756

noncomputable def z : ℂ := (1 + 2 * complex.i) * complex.i

theorem point_in_second_quadrant : 
  let z := (1 + 2 * complex.i) * complex.i in
  ∀ (x y : ℝ), z = x + y * complex.i → x < 0 ∧ y > 0 :=
by
  sorry

end point_in_second_quadrant_l720_720756


namespace solve_for_a_l720_720378

variable (a b c : ℝ) (A : ℝ)

def cosine_law_condition : Prop := 
  cos A = 7/8 ∧ c - a = 2 ∧ b = 3

theorem solve_for_a (h : cosine_law_condition a b c A) : a = 1 :=
  sorry

end solve_for_a_l720_720378


namespace allocation_plans_l720_720249

theorem allocation_plans : 
  let classes := {class1, class2, class3} in
  ∃ (allocation : list (string × string)), 
  (∀ cls, cls ∈ classes → ∃ student, (student, cls) ∈ allocation) ∧
  (('studentA', 'class1') ∉ allocation) ∧
  (length allocation = 4 ∧ 
   ∀ p ∈ allocation, p.1 ∈ {"studentA", "studentB", "studentC", "studentD"} ∧ p.2 ∈ classes) →
  length ((#allocation plans satisfying conditions#)) = 24 :=
begin
  sorry
end

end allocation_plans_l720_720249


namespace maximum_value_A_l720_720137

theorem maximum_value_A (x y : ℝ) (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) :
  ( (x^2 - y) * real.sqrt (y + x^3 - x * y) + (y^2 - x) * real.sqrt (x + y^3 - x * y) + 1 ) /
  ( (x - y)^2 + 1 ) ≤ 1 :=
sorry

end maximum_value_A_l720_720137


namespace problem_statement_l720_720394

variables {A B C H I S : Point}
variable {d : ℝ}

-- Assuming necessary conditions for the points and triangle are given,
-- especially in terms of relationships and distances
axiom orthocenter (ABC : Triangle) : Point -- Orthocenter of triangle ABC is H
axiom incenter (ABC : Triangle) : Point -- Incenter of triangle ABC is I
axiom centroid (ABC : Triangle) : Point -- Centroid of triangle ABC is S
axiom circumcircle_diameter (ABC : Triangle) : ℝ -- Diameter of the circumcircle of triangle ABC is d

noncomputable def AH := dist A H
noncomputable def BH := dist B H
noncomputable def CH := dist C H
noncomputable def AI := dist A I
noncomputable def BI := dist B I
noncomputable def CI := dist C I
noncomputable def HS := dist H S

theorem problem_statement (ABC : Triangle) 
  (H := orthocenter ABC) 
  (I := incenter ABC)
  (S := centroid ABC)
  (d := circumcircle_diameter ABC) :
  9 * (HS ^ 2) + 4 * (AH * AI + BH * BI + CH * CI) ≥ 3 * d ^ 2 := 
  sorry

end problem_statement_l720_720394


namespace alcohol_concentration_after_mixing_l720_720989

theorem alcohol_concentration_after_mixing (A_water A_alcohol B_water B_alcohol : ℝ) (hA : A_water / A_alcohol = 4 / 1) (hB : B_water / B_alcohol = 2 / 3) (V_A V_B : ℝ) (h_equal_volumes : V_A = V_B) :
  let new_alcohol_concentration := (A_alcohol * V_A + B_alcohol * V_B) / (V_A + V_B)
  in new_alcohol_concentration = 0.4 :=
by
  -- Let denote "V_A = 1" and "V_B = 1" for simplification, matching with the assumption used in solution.
  have hA_liters : A_water = 4/5 ∧ A_alcohol = 1/5, from by sorry,
  have hB_liters : B_water = 2/5 ∧ B_alcohol = 3/5, from by sorry,
  let new_alcohol_concentration := (A_alcohol * 1 + B_alcohol * 1) / (1 + 1),
  show new_alcohol_concentration = 0.4, from by sorry

end alcohol_concentration_after_mixing_l720_720989


namespace pin_slot_alignment_l720_720004

theorem pin_slot_alignment :
  ∀ (pins slots : Fin 7), ∃ (rotation : Fin 7), pinned_slot_positions(pins, slots, rotation) := 
sorry

-- Define the pinned_slot_positions
def pinned_slot_positions (pins slots : Fin 7, rotation : Fin 7) : Prop :=
  -- Implementation that computes the positions of the pins with respect to the rotation and checks for alignment with slots
  sorry

end pin_slot_alignment_l720_720004


namespace range_of_a4_l720_720803

noncomputable def geometric_sequence (a1 a2 a3 : ℝ) (q : ℝ) (a4 : ℝ) : Prop :=
  ∃ (a1 q : ℝ), 0 < a1 ∧ a1 < 1 ∧ 
                1 < a1 * q ∧ a1 * q < 2 ∧ 
                2 < a1 * q^2 ∧ a1 * q^2 < 4 ∧ 
                a4 = (a1 * q^2) * q ∧ 
                2 * Real.sqrt 2 < a4 ∧ a4 < 16

theorem range_of_a4 (a1 a2 a3 a4 : ℝ) (q : ℝ) (h1 : 0 < a1) (h2 : a1 < 1) 
  (h3 : 1 < a2) (h4 : a2 < 2) (h5 : a2 = a1 * q)
  (h6 : 2 < a3) (h7 : a3 < 4) (h8 : a3 = a1 * q^2) :
  2 * Real.sqrt 2 < a4 ∧ a4 < 16 :=
by
  have hq1 : 2 * q^2 < 1 := sorry    -- Placeholder for necessary inequalities
  have hq2: 1 < q ∧ q < 4 := sorry   -- Placeholder for necessary inequalities
  sorry

end range_of_a4_l720_720803


namespace acute_triangle_cos_sin_l720_720846

theorem acute_triangle_cos_sin (A B C : ℝ) (h : ∃ A' B' C', A + B + C = π ∧ A' = A ∧ B' = B ∧ C' = C ∧ cos A * cos B > sin A * sin B) : 
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2 := 
by 
  sorry

end acute_triangle_cos_sin_l720_720846


namespace stickers_decorate_l720_720908

theorem stickers_decorate (initial_stickers bought_stickers birthday_stickers given_stickers remaining_stickers stickers_used : ℕ)
    (h1 : initial_stickers = 20)
    (h2 : bought_stickers = 12)
    (h3 : birthday_stickers = 20)
    (h4 : given_stickers = 5)
    (h5 : remaining_stickers = 39) :
    (initial_stickers + bought_stickers + birthday_stickers - given_stickers - remaining_stickers = stickers_used) →
    stickers_used = 8 
:= by {sorry}

end stickers_decorate_l720_720908


namespace no_real_solutions_l720_720439

open Real

theorem no_real_solutions :
  ¬(∃ x : ℝ, (3 * x^2) / (x - 2) - (x + 4) / 4 + (5 - 3 * x) / (x - 2) + 2 = 0) := by
  sorry

end no_real_solutions_l720_720439


namespace at_least_n_minus_one_ministers_receive_own_decree_l720_720546

theorem at_least_n_minus_one_ministers_receive_own_decree 
  (n : ℕ) (decrees : Fin n → Type) 
  (send_telegram : (i j : Fin n) → (known_decrees : Set (Fin n)) → Prop)
  (all_familiar : ∀ i : Fin n, ∃ known_decrees : Set (Fin n), known_decrees = Set.univ)
  (received_at_most_once : ∀ i j : Fin n, ∀ d : Fin n, send_telegram i j {d} → ¬ send_telegram j i {d})
  (some_received_own_decree : ∀ i : Fin n, ∃ j : Fin n, send_telegram j i (decrees j)) : 
  ∃ S : Finset (Fin n), S.card ≥ n-1 :=
sorry

end at_least_n_minus_one_ministers_receive_own_decree_l720_720546


namespace maximum_value_A_l720_720138

theorem maximum_value_A (x y : ℝ) (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) :
  ( (x^2 - y) * real.sqrt (y + x^3 - x * y) + (y^2 - x) * real.sqrt (x + y^3 - x * y) + 1 ) /
  ( (x - y)^2 + 1 ) ≤ 1 :=
sorry

end maximum_value_A_l720_720138


namespace product_pairwise_differences_divisible_by_12_l720_720968

-- Define that the given integers are distinct
def distinct {α : Type*} (s : list α) : Prop :=
  s.nodup

-- Main statement: Given a, b, c, d are four distinct integers,
-- prove that the product of their pairwise differences is divisible by 12.
theorem product_pairwise_differences_divisible_by_12 
  (a b c d : ℤ) (h_distinct : distinct [a, b, c, d]) : 
  ∃ k : ℤ, (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) = 12 * k :=
by
  sorry

end product_pairwise_differences_divisible_by_12_l720_720968


namespace conditional_probability_l720_720559

def num_attractions : ℕ := 5
def diff_attractions (a b : ℕ) : Prop := a ≠ b
def chooses_jiuzhaigou (a : ℕ) : Prop := a = 1
def one_chooses_jiuzhaigou (a b : ℕ) : Prop :=
  (chooses_jiuzhaigou a ∧ ¬chooses_jiuzhaigou b) ∨ (chooses_jiuzhaigou b ∧ ¬chooses_jiuzhaigou a)

theorem conditional_probability :
  ∀ (a b : ℕ), a ≠ b → one_chooses_jiuzhaigou a b → 
  (finset.card {x : ℕ × ℕ | diff_attractions x.1 x.2}.filter (λ x, one_chooses_jiuzhaigou x.1 x.2) /
   finset.card {x : ℕ × ℕ | diff_attractions x.1 x.2} 
   = 2 / 5) :=
by
  sorry

end conditional_probability_l720_720559


namespace harmonic_half_l720_720955

theorem harmonic_half (n : ℕ) (h : 2 ≤ n) : 
  (∑ k in Finset.range (2*n - n) + n, (1 / (k : ℝ))) > (1 / 2) := 
by sorry

end harmonic_half_l720_720955


namespace distance_symmetric_parabola_l720_720586

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def parabola (x : ℝ) : ℝ := 3 - x^2

theorem distance_symmetric_parabola (A B : ℝ × ℝ) 
  (hA : A.2 = parabola A.1) 
  (hB : B.2 = parabola B.1)
  (hSym : A.1 + A.2 = 0 ∧ B.1 + B.2 = 0) 
  (hDistinct : A ≠ B) :
  distance A B = 3 * sqrt 2 :=
by
  sorry

end distance_symmetric_parabola_l720_720586


namespace parallelogram_opposite_sides_parallel_lines_perpendicular_equation_solution_equidistant_points_bisector_l720_720250

-- Problem 1
theorem parallelogram_opposite_sides (Q : Type) [Quadrilateral Q] :
  (Parallelogram Q) ↔ (OppositeSidesPairwiseEqual Q) :=
sorry

-- Problem 2
theorem parallel_lines_perpendicular (L M N : Type) [Line L] [Line M] [Line N] :
  (Parallel L M) ↔ (PerpendicularToThirdLine L M N) :=
sorry

-- Problem 3
theorem equation_solution :
  (2 * x + 5 = 0) ↔ (x = -2.5) :=
sorry

-- Problem 4
theorem equidistant_points_bisector (P : Type) [Point P] (A B : Type) [Angle A B] :
  (EquidistantFromSidesOfAngle P A B) ↔ (LiesOnAngleBisector P A B) :=
sorry

end parallelogram_opposite_sides_parallel_lines_perpendicular_equation_solution_equidistant_points_bisector_l720_720250


namespace decimal_to_fraction_l720_720082

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l720_720082


namespace D_time_to_complete_job_l720_720129

-- Let A_rate be the rate at which A works (jobs per hour)
-- Let D_rate be the rate at which D works (jobs per hour)
def A_rate : ℚ := 1 / 3
def combined_rate : ℚ := 1 / 2

-- We need to prove that D_rate, the rate at which D works alone, is 1/6 jobs per hour
def D_rate := 1 / 6

-- And thus, that D can complete the job in 6 hours
theorem D_time_to_complete_job :
  (A_rate + D_rate = combined_rate) → (1 / D_rate) = 6 :=
by
  sorry

end D_time_to_complete_job_l720_720129


namespace q1_q2_qm_sum_at_3_l720_720392

theorem q1_q2_qm_sum_at_3 :
  ∃ (q : ℕ → polynomial ℤ) (m : ℕ),
    (∀ i, (q i).monic) ∧
    (∀ i, (q i).coeffs.forall (λ c, c ∈ ℤ)) ∧
    (∀ i, irreducible (q i)) ∧
    polynomial.eval 3 (polynomial.x^7 - polynomial.x^3 - 2 * polynomial.x^2 - polynomial.x - 1) =
    polynomial.eval 3 (∏ i in finset.range m, q i) ∧
    (finset.range m).sum (λ i, polynomial.eval 3 (q i)) = 42 :=
sorry

end q1_q2_qm_sum_at_3_l720_720392


namespace smaller_number_l720_720030

theorem smaller_number (x y : ℝ) (h1 : x + y = 16) (h2 : x - y = 4) (h3 : x * y = 60) : y = 6 :=
sorry

end smaller_number_l720_720030


namespace Zoe_given_card_6_l720_720847

-- Define the cards and friends
def cards : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
def friends : List String := ["Eliza", "Miguel", "Naomi", "Ivan", "Zoe"]

-- Define scores 
def scores (name : String) : ℕ :=
  match name with
  | "Eliza"  => 15
  | "Miguel" => 11
  | "Naomi"  => 9
  | "Ivan"   => 13
  | "Zoe"    => 10
  | _ => 0

-- Each friend is given a pair of cards
def cardAssignments (name : String) : List (ℕ × ℕ) :=
  match name with
  | "Eliza"  => [(6,9), (7,8), (5,10), (4,11), (3,12)]
  | "Miguel" => [(1,10), (2,9), (3,8), (4,7), (5,6)]
  | "Naomi"  => [(1,8), (2,7), (3,6), (4,5)]
  | "Ivan"   => [(1,12), (2,11), (3,10), (4,9), (5,8), (6,7)]
  | "Zoe"    => [(1,9), (2,8), (3,7), (4,6)]
  | _ => []

-- The proof statement
theorem Zoe_given_card_6 : ∃ c1 c2, (c1, c2) ∈ cardAssignments "Zoe" ∧ (c1 = 6 ∨ c2 = 6)
:= by
  sorry -- Proof omitted as per the instructions

end Zoe_given_card_6_l720_720847


namespace alice_muffins_l720_720636

theorem alice_muffins : 
  let M : ℕ := by sorry in
  (M < 90) ∧ (M % 9 = 5) ∧ (M % 5 = 2) → 
  (∃ sum_M : ℕ, sum_M = 250) :=
begin
  sorry
end

end alice_muffins_l720_720636


namespace cos_theta_solution_l720_720748

theorem cos_theta_solution (θ : ℝ) (h₁ : cos (θ - π / 4) = 3 / 5) (h₂ : θ ∈ Ioo (-π / 4) (π / 4)) : 
  cos θ = 7 * real.sqrt 2 / 10 :=
sorry

end cos_theta_solution_l720_720748


namespace westward_measurement_l720_720827

def east_mov (d : ℕ) : ℤ := - (d : ℤ)

def west_mov (d : ℕ) : ℤ := d

theorem westward_measurement :
  east_mov 50 = -50 →
  west_mov 60 = 60 :=
by
  intro h
  exact rfl

end westward_measurement_l720_720827


namespace Petya_running_time_l720_720223

theorem Petya_running_time (D V : ℝ) 
  (hV_pos : 0 < V) (hD_pos : 0 < D):
  let T := D / V in
  let V1 := 1.25 * V in
  let V2 := 0.8 * V in
  let T1 := (D / 2) / V1 in
  let T2 := (D / 2) / V2 in
  let T_actual := T1 + T2 in
  T_actual > T :=
by
  let T := D / V
  let V1 := 1.25 * V
  let V2 := 0.8 * V
  let T1 := (D / 2) / V1
  let T2 := (D / 2) / V2
  let T_actual := T1 + T2
  have : T_actual = (2 * D) / (5 * V) + (5 * D) / (8 * V) := by 
  sorry
  have : T_actual = 41 * D / (40 * V) := by 
  sorry
  have : T = D / V := by 
  sorry
  show 41 * D / (40 * V) > D / V := by 
  sorry

end Petya_running_time_l720_720223


namespace tenth_largest_number_is_531_l720_720655

theorem tenth_largest_number_is_531 : 
  let digits := {5, 3, 1, 9}
  ∃ nums : List ℕ, 
  ((∀ n ∈ nums, ∃ a b c, 
    a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ 
    a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
    n = 100 * a + 10 * b + c) ∧ 
    List.length nums = 24 ∧ 
    List.reverse (nums.quicksort (· ≤ ·)).nth 9 = some 531) := 
by
  -- Proof goes here
  sorry

end tenth_largest_number_is_531_l720_720655


namespace sum_three_digit_even_integers_l720_720575

theorem sum_three_digit_even_integers :
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  S = 247050 :=
by
  let a := 100
  let d := 2
  let l := 998
  let n := (l - a) / d + 1
  let S := n / 2 * (a + l)
  sorry

end sum_three_digit_even_integers_l720_720575


namespace two_point_three_five_as_fraction_l720_720067

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l720_720067


namespace packing_height_difference_l720_720040

noncomputable def height_difference (d : ℝ) (n : ℕ) : ℝ :=
let crate_A_height := n * d,
    crate_B_height := d + (n - 1) * (d * (Real.sqrt 3 / 2))
in abs (crate_A_height - crate_B_height)

theorem packing_height_difference :
  height_difference 12 200 = abs (2400 - (12 + 1194 * Real.sqrt 3)) :=
by
  sorry

end packing_height_difference_l720_720040


namespace mod_inv_3_191_l720_720678

theorem mod_inv_3_191 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 190 ∧ (3 * x) % 191 = 1 :=
by
  use 64
  split
  . exact nat.zero_le 64
  . split
  . exact dec_trivial
  . exact dec_trivial

end mod_inv_3_191_l720_720678


namespace Petya_running_time_l720_720221

theorem Petya_running_time (D V : ℝ) 
  (hV_pos : 0 < V) (hD_pos : 0 < D):
  let T := D / V in
  let V1 := 1.25 * V in
  let V2 := 0.8 * V in
  let T1 := (D / 2) / V1 in
  let T2 := (D / 2) / V2 in
  let T_actual := T1 + T2 in
  T_actual > T :=
by
  let T := D / V
  let V1 := 1.25 * V
  let V2 := 0.8 * V
  let T1 := (D / 2) / V1
  let T2 := (D / 2) / V2
  let T_actual := T1 + T2
  have : T_actual = (2 * D) / (5 * V) + (5 * D) / (8 * V) := by 
  sorry
  have : T_actual = 41 * D / (40 * V) := by 
  sorry
  have : T = D / V := by 
  sorry
  show 41 * D / (40 * V) > D / V := by 
  sorry

end Petya_running_time_l720_720221


namespace number_of_solutions_l720_720596

theorem number_of_solutions (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, k = 2 + 4 * n ∧ (∃ (x y : ℤ), x ^ 2 + 2016 * y ^ 2 = 2017 ^ n) :=
by
  sorry

end number_of_solutions_l720_720596


namespace parallel_lines_l720_720950

-- Definition of points and lines
variables {A B C M N P Q I Z : Type} [geometry_space : Geometry A B C M N P Q I Z]

-- Conditions
def is_center_of_circumcircle (A B C I : Type) : Prop := Geometry.is_center_of_circumcircle A B C I
def midpoint_of_arc_AC (A C I : Type) : Prop := Geometry.midpoint_ω A C I
def is_angle_bisector (IZ : Type) (B : Type) : Prop := Geometry.angle_bisector IZ B
def is_perpendicular (PQ IZ : Type) : Prop := Geometry.perpendicular PQ IZ
def is_isosceles (MBI NBI BI : Type) : Prop := Geometry.isosceles MBI NBI BI

-- Problem Statement
theorem parallel_lines (A B C M N P Q I Z : Type)
  [is_center_of_circumcircle A B C I]
  [midpoint_of_arc_AC A C I]
  [is_angle_bisector IZ B]
  [is_perpendicular PQ IZ]
  [is_isosceles MBI NBI BI] : PQ ∥ MN :=
by
  sorry

end parallel_lines_l720_720950


namespace time_difference_l720_720909

/-
Malcolm's speed: 5 minutes per mile
Joshua's speed: 7 minutes per mile
Race length: 12 miles
Question: Prove that the time difference between Joshua crossing the finish line after Malcolm is 24 minutes
-/
noncomputable def time_taken (speed: ℕ) (distance: ℕ) : ℕ :=
  speed * distance

theorem time_difference :
  let malcolm_speed := 5
  let joshua_speed := 7
  let race_length := 12
  let malcolm_time := time_taken malcolm_speed race_length
  let joshua_time := time_taken joshua_speed race_length
  malcolm_time < joshua_time →
  joshua_time - malcolm_time = 24 :=
by
  intros malcolm_speed joshua_speed race_length malcolm_time joshua_time malcolm_time_lt_joshua_time
  sorry

end time_difference_l720_720909


namespace parallel_MN_PQ_l720_720939

variables {A B C I Z M N P Q : Point}
variables {BI IZ PQ MN : Line}

-- Conditions
variables (triangle_ABC : is_triangle A B C)
variables (incircle_center_I : is_incircle_center I A B C)
variables (angle_bisector_IZ : is_angle_bisector Z B I triangle_ABC)
variables (perpendicular_PQ_IZ : is_perpendicular PQ IZ)
variables (perpendicular_IZ_BI : is_perpendicular IZ BI)
variables (perpendicular_MN_BI : is_perpendicular MN BI)

theorem parallel_MN_PQ 
  (h1: is_angle_bisector Z B I triangle_ABC)
  (h2: is_perpendicular PQ IZ)
  (h3: is_perpendicular IZ BI)
  (h4: is_perpendicular MN BI) : 
  is_parallel MN PQ := 
sorry

end parallel_MN_PQ_l720_720939


namespace max_sum_of_cubes_l720_720400

open Real

theorem max_sum_of_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * sqrt 5 :=
by
  sorry

end max_sum_of_cubes_l720_720400


namespace MN_parallel_PQ_l720_720936

-- Define the given geometric entities and properties
variables (A B C I M N P Q Z : Type) [Inhabited A]

-- Definition of points being collinear
def collinear (a b c : Type) := ∃ (r : ℝ), r • (b - a) + a = c

-- Definition of parallel lines
def parallel (l1 l2 : Type) : Prop := 
  ∃ p1 p2 p3 p4 : Type, collinear p1 p2 p3 → collinear p2 p3 p4

-- Definition points on a circle
def on_circumcircle (A B C : Type) : Prop := sorry

-- Definition of angle bisector
def angle_bisector (I A B : Type) : Prop := sorry

-- Definition of perpendicular lines
def perpendicular (l1 l2 : Type) : Prop := sorry

-- Main theorem statement
theorem MN_parallel_PQ 
  (h1 : on_circumcircle A B C) 
  (h2 : angle_bisector I A B) 
  (h3 : perpendicular PQ IZ) 
  (h4 : perpendicular PQ BI) 
  (h5 : perpendicular MN BI) : parallel MN PQ :=
sorry

end MN_parallel_PQ_l720_720936


namespace inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720484

variable {a b : ℝ}

theorem inequality_a (hab : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_b_not_true (hab : a + b > 0) : ¬(a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

theorem inequality_c (hab : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem inequality_d (hab : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

theorem inequality_e_not_true (hab : a + b > 0) : ¬((a − 3) * (b − 3) < a * b) :=
sorry

theorem inequality_f_not_true (hab : a + b > 0) : ¬((a + 2) * (b + 3) > a * b + 5) :=
sorry

end inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720484


namespace inequality_solution_l720_720967

theorem inequality_solution (x : ℝ) : 
  (1 + (1 + (1 + x) / (1 - 3 * x)) / (1 - 3 * (1 + (1 + x) / (1 - 3 * x)))) 
  / (1 - 3 * (1 + (1 + x) / (1 - 3 * x)) / (1 - 3 * (1 + (1 + x) / (1 - 3 * x)))) < 
  (1 - 3 * (1 + (1 + x) / (1 - 3 * x)) / (1 - 3 * (1 + (1 + x) / (1 - 3 * x)))) 
  / (1 + (1 + (1 + x) / (1 - 3 * x)) / (1 - 3 * (1 + (1 + x) / (1 - 3 * x)))) 
  ↔ x ∈ Set.Union (Set.Union Set.Iio (-1 : ℝ)) (Set.Union (Set.Ioi (0 : ℝ)) (Set.Iio (1 : ℝ))) := 
by
  sorry

end inequality_solution_l720_720967


namespace smallest_positive_five_digit_number_divisible_by_first_five_primes_l720_720715

theorem smallest_positive_five_digit_number_divisible_by_first_five_primes :
  ∃ n : ℕ, (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ 10000 ≤ n ∧ n = 11550 :=
by
  use 11550
  split
  · intros p hp
    fin_cases hp <;> norm_num
  split
  · norm_num
  rfl

end smallest_positive_five_digit_number_divisible_by_first_five_primes_l720_720715


namespace decimal_to_fraction_l720_720107

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l720_720107


namespace decimal_to_fraction_l720_720101

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l720_720101


namespace present_age_of_son_l720_720588

theorem present_age_of_son :
  (∃ (S F : ℕ), F = S + 22 ∧ (F + 2) = 2 * (S + 2)) → ∃ (S : ℕ), S = 20 :=
by
  sorry

end present_age_of_son_l720_720588


namespace range_of_m_l720_720757

theorem range_of_m (f : ℝ → ℝ) (h_deriv : ∀ x, differentiable_at ℝ f x)
  (h_fun : ∀ x, f(x) = 4 * x^2 - f(-x))
  (h_ineq1 : ∀ x ∈ Iio 0, deriv f x + 1 / 2 < 4 * x)
  (h_ineq2 : ∀ m, f(m + 1) ≤ f(-m) + 4 * m + 2) :
  ∀ m, m ≥ -1/2 :=
sorry

end range_of_m_l720_720757


namespace inscribed_square_area_l720_720624

-- Define the condition of the problem
def inscribed_square_condition (t : ℝ) : Prop :=
(∀ x y, (x = t ∨ x = -t) ∧ (y = t ∨ y = -t) →
( x^2 / 4 + y^2 / 8 = 1 ))

-- The theorem that proves the area of the square inscribed in the ellipse
theorem inscribed_square_area :
  ∃ t : ℝ, inscribed_square_condition t ∧ (2 * t)*(2 * t) = 32 / 3 :=
begin
  sorry
end

end inscribed_square_area_l720_720624


namespace smallest_five_digit_number_divisible_by_first_five_primes_l720_720704

theorem smallest_five_digit_number_divisible_by_first_five_primes : 
  ∃ n, (n >= 10000) ∧ (n < 100000) ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_first_five_primes_l720_720704


namespace cost_to_fill_half_of_can_B_l720_720234

theorem cost_to_fill_half_of_can_B (r h : ℝ) (cost_fill_V : ℝ) (cost_fill_V_eq : cost_fill_V = 16)
  (V_radius_eq : 2 * r = radius_of_can_V)
  (V_height_eq: h / 2 = height_of_can_V) :
  cost_fill_half_of_can_B = 4 :=
by
  sorry

end cost_to_fill_half_of_can_B_l720_720234


namespace no_real_roots_polynomial_l720_720957

noncomputable def f (n : ℕ) : ℝ → ℝ := λ x, (∑ k in (finset.range (2 * n)).filter (λ k, k % 2 == 0), (2 * n + 1 - k) * (x ^ k) * (-1) ^ (k % 2))

theorem no_real_roots_polynomial (n : ℕ) : ¬ ∃ x : ℝ, f n x = 0 :=
sorry

end no_real_roots_polynomial_l720_720957


namespace decimal_to_fraction_l720_720097

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l720_720097


namespace right_triangle_in_marked_cubes_l720_720161

theorem right_triangle_in_marked_cubes 
  (n : ℕ) 
  (marked : set (fin n × fin n × fin n)) 
  (h_marked : marked.card > (3 / 2) * (n ^ 2)) : 
  ∃ (a b c : fin n × fin n × fin n),
    a ∈ marked ∧ b ∈ marked ∧ c ∈ marked ∧
    ((a.1 = b.1 ∧ b.2 = c.2 ∧ c.1 = a.1) ∨
     (a.2 = b.2 ∧ b.3 = c.3 ∧ c.2 = a.2) ∨
     (a.1 = b.1 ∧ b.3 = c.3 ∧ c.1 = a.1)) :=
begin
  sorry
end

end right_triangle_in_marked_cubes_l720_720161


namespace polar_equation_parabola_l720_720240

def isParabola (x y : ℝ) : Prop :=
  y^2 = 2 * x + 1

theorem polar_equation_parabola :
  ∀ θ: ℝ, ∃ r: ℝ, r = 1 / (1 + cos θ) → ∃ x y: ℝ, x = r * cos θ ∧ y = r * sin θ ∧ isParabola x y :=
by
  intro θ
  use 1 / (1 + cos θ)
  intro h
  let r := 1 / (1 + cos θ)
  let x := r * cos θ
  let y := r * sin θ
  use x, y
  split; sorry

end polar_equation_parabola_l720_720240


namespace solve_x_l720_720577

noncomputable def x : ℝ := 4.7

theorem solve_x : (10 - x) ^ 2 = x ^ 2 + 6 :=
by
  sorry

end solve_x_l720_720577


namespace complex_identity_l720_720314

open Complex

noncomputable def z := 1 + 2 * I
noncomputable def z_inv := (1 - 2 * I) / 5
noncomputable def z_conj := 1 - 2 * I

theorem complex_identity : 
  (z + z_inv) * z_conj = (22 / 5 : ℂ) - (4 / 5) * I := 
by
  sorry

end complex_identity_l720_720314


namespace greatest_good_t_l720_720883

noncomputable def S (a t : ℕ) : Set ℕ := {x | ∃ n : ℕ, x = a + 1 + n ∧ n < t}

def is_good (S : Set ℕ) (k : ℕ) : Prop :=
∃ (coloring : ℕ → Fin k), ∀ (x y : ℕ), x ≠ y → x + y ∈ S → coloring x ≠ coloring y

theorem greatest_good_t {k : ℕ} (hk : k > 1) : ∃ t, ∀ a, is_good (S a t) k ∧ 
  ∀ t' > t, ¬ ∀ a, is_good (S a t') k := 
sorry

end greatest_good_t_l720_720883


namespace moles_of_CO2_combined_l720_720269
-- Import the necessary library

-- Define the assumptions and the theorem statement
open_locale classical

theorem moles_of_CO2_combined (a b c : ℕ) (h1 : a = 2) (h2 : c = 2) (h3 : a = c) : b = 2 :=
by sorry

end moles_of_CO2_combined_l720_720269


namespace part_1_part_2_l720_720320

noncomputable def f (m x : ℝ) : ℝ := 
  log (1 + m * x) + (x^2 / 2) - m * x

theorem part_1 (x : ℝ) (h1 : -1 < x) (h2 : x ≤ 0) (m : ℝ) (hm : m = 1) : 
  f m x ≤ x^3 / 3 := 
sorry

theorem part_2 (m : ℝ) (hm : 0 < m ∧ m ≤ 1) :
  (if m = 1 then ∃! x, f m x = 0 
   else ∃ x1 x2, x1 ≠ x2 ∧ f m x1 = 0 ∧ f m x2 = 0) :=
sorry

end part_1_part_2_l720_720320


namespace subtraction_contradiction_l720_720604

theorem subtraction_contradiction (k t : ℕ) (hk_non_zero : k ≠ 0) (ht_non_zero : t ≠ 0) : 
  ¬ ((8 * 100 + k * 10 + 8) - (k * 100 + 8 * 10 + 8) = 1 * 100 + 6 * 10 + t * 1) :=
by
  sorry

end subtraction_contradiction_l720_720604


namespace decimal_to_fraction_l720_720110

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l720_720110


namespace lambda_range_iff_inequality_l720_720328

noncomputable def inequality_all (x λ : ℝ) : Prop :=
(e^x + 1) * x > (log x - log λ) * (x / λ + 1)

theorem lambda_range_iff_inequality (x λ : ℝ) :
  (∀ x, inequality_all x λ) ↔ λ ∈ Ioi (1 / exp 1) :=
sorry

end lambda_range_iff_inequality_l720_720328


namespace triangle_circumcircle_angle_bisector_l720_720396

theorem triangle_circumcircle_angle_bisector (A B C D E F : Point) (α β γ : ℝ)
  (h_triangle : triangle A B C)
  (D_foot : ∃ D, angle_bisector A B C D)
  (E_inter : ∃ E, circumcircle A B D E ∧ E ∈ segment A C)
  (F_inter : ∃ F, circumcircle A D C F ∧ F ∈ segment A B) :
  distance B F = distance C E :=
sorry

end triangle_circumcircle_angle_bisector_l720_720396


namespace train_length_equals_750_l720_720011

theorem train_length_equals_750
  (L : ℕ) -- length of the train in meters
  (v : ℕ) -- speed of the train in m/s
  (t : ℕ) -- time in seconds
  (h1 : v = 25) -- speed is 25 m/s
  (h2 : t = 60) -- time is 60 seconds
  (h3 : 2 * L = v * t) -- total distance covered by the train is 2L (train and platform) and equals speed * time
  : L = 750 := 
sorry

end train_length_equals_750_l720_720011


namespace minimum_children_to_ensure_three_same_birthday_l720_720121

theorem minimum_children_to_ensure_three_same_birthday : ∀ (days : ℕ), days = 366 → ∃ (n : ℕ), n = 733 ∧ (∀ (f : fin n → fin days), ∃ (d : fin days), 3 ≤ (finset.filter (λ i, f i = d) (finset.range n)).card) :=
begin
  sorry
end

end minimum_children_to_ensure_three_same_birthday_l720_720121


namespace alternating_colors_probability_l720_720606

noncomputable def alternating_prob : ℚ := 
  let total_sequences := (nat.choose 9 4) in
  let success_sequences := 1 in
  success_sequences / total_sequences

theorem alternating_colors_probability :
  let white_balls := 5 in
  let black_balls := 4 in
  let draw_sequence := ["W", "B", "W", "B", "W", "B", "W", "B", "W"] in
  have valid_draw : draw_sequence.length = white_balls + black_balls := rfl,
  have condition_met : ∀ i < draw_sequence.length - 1, draw_sequence.nth i ≠ draw_sequence.nth (i+1) := by sorry,
  alternating_prob = 1 / 126 := by sorry

end alternating_colors_probability_l720_720606


namespace radius_of_each_circle_l720_720978

variables (ℓ m : ℝ) (A B : ℝ × ℝ) (r : ℝ)
def conditions : Prop :=
  ℓ = 0 ∧ m = 12 ∧ A = (0, 0) ∧ B = (5, 12) ∧
  dist A B = 13

theorem radius_of_each_circle (h : conditions ℓ m A B r) : r = 169 / 48 := 
  sorry

end radius_of_each_circle_l720_720978


namespace find_x_log_eq_l720_720259

theorem find_x_log_eq (x : ℝ) : log x 16 = log 64 4 → x = 4096 :=
by
  sorry

end find_x_log_eq_l720_720259


namespace problem1_problem2_problem3_problem4_problem5_problem6_l720_720517

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end problem1_problem2_problem3_problem4_problem5_problem6_l720_720517


namespace two_point_three_five_as_fraction_l720_720068

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l720_720068


namespace fabric_nguyen_needs_l720_720918

-- Definitions for conditions
def fabric_per_pant : ℝ := 8.5
def total_pants : ℝ := 7
def yards_to_feet (yards : ℝ) : ℝ := yards * 3
def fabric_nguyen_has_yards : ℝ := 3.5

-- The proof we need to establish
theorem fabric_nguyen_needs : (total_pants * fabric_per_pant) - (yards_to_feet fabric_nguyen_has_yards) = 49 :=
by
  sorry

end fabric_nguyen_needs_l720_720918


namespace print_time_correct_l720_720169

-- Define the conditions
def pages_per_minute : ℕ := 23
def total_pages : ℕ := 345

-- Define the expected result
def expected_minutes : ℕ := 15

-- Prove the equivalence
theorem print_time_correct :
  total_pages / pages_per_minute = expected_minutes :=
by 
  -- Proof will be provided here
  sorry

end print_time_correct_l720_720169


namespace max_expression_value_l720_720893

theorem max_expression_value 
  (x y z : ℝ) 
  (h_nonneg_x : 0 ≤ x)
  (h_nonneg_y : 0 ≤ y)
  (h_nonneg_z : 0 ≤ z)
  (h_sum_squares : x^2 + y^2 + z^2 = 1) :
  2 * x * y * real.sqrt 8 + 6 * y * z ≤ real.sqrt 2 := 
sorry

end max_expression_value_l720_720893


namespace right_triangle_condition_l720_720667

theorem right_triangle_condition (a d : ℝ) (h : d > 0) : 
  (a = d * (1 + Real.sqrt 7)) ↔ (a^2 + (a + 2 * d)^2 = (a + 4 * d)^2) := 
sorry

end right_triangle_condition_l720_720667


namespace parallel_lines_l720_720954

-- Definition of points and lines
variables {A B C M N P Q I Z : Type} [geometry_space : Geometry A B C M N P Q I Z]

-- Conditions
def is_center_of_circumcircle (A B C I : Type) : Prop := Geometry.is_center_of_circumcircle A B C I
def midpoint_of_arc_AC (A C I : Type) : Prop := Geometry.midpoint_ω A C I
def is_angle_bisector (IZ : Type) (B : Type) : Prop := Geometry.angle_bisector IZ B
def is_perpendicular (PQ IZ : Type) : Prop := Geometry.perpendicular PQ IZ
def is_isosceles (MBI NBI BI : Type) : Prop := Geometry.isosceles MBI NBI BI

-- Problem Statement
theorem parallel_lines (A B C M N P Q I Z : Type)
  [is_center_of_circumcircle A B C I]
  [midpoint_of_arc_AC A C I]
  [is_angle_bisector IZ B]
  [is_perpendicular PQ IZ]
  [is_isosceles MBI NBI BI] : PQ ∥ MN :=
by
  sorry

end parallel_lines_l720_720954


namespace decimal_to_fraction_l720_720096

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l720_720096


namespace petya_time_comparison_l720_720215

variables (D V : ℝ) (hD_pos : D > 0) (hV_pos : V > 0)

theorem petya_time_comparison (hD_pos : D > 0) (hV_pos : V > 0) :
  (41 * D / (40 * V)) > (D / V) :=
by
  sorry

end petya_time_comparison_l720_720215


namespace regular_polygon_exterior_angle_l720_720202

theorem regular_polygon_exterior_angle (n : ℕ) (h : 60 * n = 360) : n = 6 :=
sorry

end regular_polygon_exterior_angle_l720_720202


namespace christine_min_bottles_l720_720237

theorem christine_min_bottles
  (fluid_ounces_needed : ℕ)
  (bottle_volume_ml : ℕ)
  (fluid_ounces_per_liter : ℝ)
  (liters_in_milliliter : ℕ)
  (required_bottles : ℕ)
  (h1 : fluid_ounces_needed = 45)
  (h2 : bottle_volume_ml = 200)
  (h3 : fluid_ounces_per_liter = 33.8)
  (h4 : liters_in_milliliter = 1000)
  (h5 : required_bottles = 7) :
  required_bottles = ⌈(fluid_ounces_needed * liters_in_milliliter) / (bottle_volume_ml * fluid_ounces_per_liter)⌉ := by
  sorry

end christine_min_bottles_l720_720237


namespace exists_multiple_sum_divides_l720_720898

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_multiple_sum_divides {n : ℕ} (hn : n > 0) :
  ∃ (n_ast : ℕ), n ∣ n_ast ∧ sum_of_digits n_ast ∣ n_ast :=
by
  sorry

end exists_multiple_sum_divides_l720_720898


namespace problem_a_problem_b_problem_c_l720_720527

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end problem_a_problem_b_problem_c_l720_720527


namespace problem_a_problem_b_problem_c_l720_720529

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end problem_a_problem_b_problem_c_l720_720529


namespace number_of_stickers_used_to_decorate_l720_720905

def initial_stickers : ℕ := 20
def bought_stickers : ℕ := 12
def birthday_stickers : ℕ := 20
def given_stickers : ℕ := 5
def remaining_stickers : ℕ := 39

theorem number_of_stickers_used_to_decorate :
  (initial_stickers + bought_stickers + birthday_stickers - given_stickers - remaining_stickers) = 8 :=
by
  -- Proof goes here
  sorry

end number_of_stickers_used_to_decorate_l720_720905


namespace find_a_l720_720790

theorem find_a (a : ℝ) (h : (∀ r : ℕ, (r = 1 → (a * nat.choose 5 r) = -15))) : a = -3 := by
  have r_eq_1 : 10 - 3 * 1 = 7 := by norm_num
  specialize h 1
  rw r_eq_1 at h
  suffices (a * nat.choose 5 1 = -15) by
    rw complete at h
    simp at h
    exact sorry

end find_a_l720_720790


namespace min_cos_sum_l720_720809

theorem min_cos_sum (x y z : ℝ) (hx : 0 ≤ x) (hx' : x ≤ π) (hy : 0 ≤ y) (hy' : y ≤ π) (hz : 0 ≤ z) (hz' : z ≤ π) :
  ∃ x y z, (A = cos (x - y) + cos (y - z) + cos (z - x)) ∧  A = -1

end min_cos_sum_l720_720809


namespace print_time_l720_720172

-- Define the conditions
def pages : ℕ := 345
def rate : ℕ := 23
def expected_minutes : ℕ := 15

-- State the problem as a theorem
theorem print_time (pages rate : ℕ) : (pages / rate = 15) :=
by
  sorry

end print_time_l720_720172


namespace unique_non_congruent_rectangle_with_conditions_l720_720179

theorem unique_non_congruent_rectangle_with_conditions :
  ∃! (w h : ℕ), 2 * (w + h) = 80 ∧ w * h = 400 :=
by
  sorry

end unique_non_congruent_rectangle_with_conditions_l720_720179


namespace find_m_plus_n_volume_of_dirt_l720_720180

-- Definitions of the problem conditions
def length := 12
def width := 10
def height := 3
def slope_distance := 4

-- Definitions required for the proof problem
def m := 264 -- Volume from dirt along the sides
def n := 16 -- Volume from dirt in the corners

-- Statement proving m + n = 280
theorem find_m_plus_n : m + n = 280 := by
  -- Explicitly assigning values for clarity
  have h_m : m = 264 := rfl
  have h_n : n = 16 := rfl
  -- Adding the values to prove the statement
  show 264 + 16 = 280 from rfl

-- Provide a statement to represent the volume of dirt
theorem volume_of_dirt (m n : ℕ) (π : real) : m + n * π = 264 + 16 * π := 
by
  -- Explicitly stating the equality based on given problem condition
  have h_m : m = 264 := rfl
  have h_n : n = 16 := rfl
  show m + n * π = 264 + 16 * π from rfl

end find_m_plus_n_volume_of_dirt_l720_720180


namespace print_time_correct_l720_720168

-- Define the conditions
def pages_per_minute : ℕ := 23
def total_pages : ℕ := 345

-- Define the expected result
def expected_minutes : ℕ := 15

-- Prove the equivalence
theorem print_time_correct :
  total_pages / pages_per_minute = expected_minutes :=
by 
  -- Proof will be provided here
  sorry

end print_time_correct_l720_720168


namespace number_of_real_roots_l720_720891

noncomputable section

open Real

def determinant (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

theorem number_of_real_roots (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) :
  (∑ (x : ℝ) in (roots (λ x, determinant x p (-r) (-p) x q r (-q) x)), x).toFinset.card = 1 := 
  sorry

end number_of_real_roots_l720_720891


namespace engine_capacity_proof_l720_720605

-- the essential conditions
variables (d1 d2 : ℕ) (c : ℕ)

def direct_varies_as_engine_capacity (v : ℕ) (e : ℕ) : Prop :=
  ∃ k : ℕ, v = k * e

-- conditions
theorem engine_capacity_proof :
  (direct_varies_as_engine_capacity 170 1200) →
  (direct_varies_as_engine_capacity 85 c) →
  ∃ C : ℕ, C = 595 :=
by
  -- Introduce the variables and assumptions
  intros h1 h2,

  -- Extract proportionality constants from the conditions
  cases h1 with k1 h1_eq,
  cases h2 with k2 h2_eq,

  -- Setup equations derived from direct variation
  have h3 := congr_arg (λ x, x * 1200) h1_eq,
  have h4 := congr_arg (λ x, x * c) h2_eq,

  -- Solve the equations to find C
  sorry

end engine_capacity_proof_l720_720605


namespace green_apples_count_l720_720550

-- Definitions for the conditions
def total_apples : ℕ := 9
def red_apples : ℕ := 7

-- Theorem stating the number of green apples
theorem green_apples_count : total_apples - red_apples = 2 := by
  sorry

end green_apples_count_l720_720550


namespace fixed_point_on_parabola_l720_720899

-- Definition of the parabola
def parabola (x t : ℝ) := 4 * x^2 + t * x - 3 * t

-- The fixed point (3, 36) 
def fixed_point (p : ℝ × ℝ) := p = (3, 36)

-- The statement to prove that (3, 36) lies on any parabola defined by y = 4x^2 + tx - 3t
theorem fixed_point_on_parabola (t : ℝ) : fixed_point (3, parabola 3 t) :=
by {
  show fixed_point (3, parabola 3 t),
  calc 
    parabola 3 t = 4 * 3^2 + t * 3 - 3 * t : by refl
              ... = 36                    : by ring,
  exact congrArgProd (rfl) (by normNum)
}

end fixed_point_on_parabola_l720_720899


namespace smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l720_720696

theorem smallest_five_digit_number_divisible_by_prime_2_3_5_7_11 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l720_720696


namespace A_finishes_work_in_6_days_l720_720190

theorem A_finishes_work_in_6_days :
  (∃ A_work B_work C_work : ℕ → ℕ → Prop,
   ∀ (d : ℕ), A_work d 6) ∧
  (∀ (d : ℕ), B_work d 18) ∧
  (∀ (d : ℕ), C_work d 36) →
  (∀ (d : ℕ), (A_work d 4) ∧ (B_work d 18) ∧ (C_work d 36)) →
  (∃ (d : ℕ), d = 6) := sorry

end A_finishes_work_in_6_days_l720_720190


namespace jacket_markup_l720_720613

def markup_percentage (purchase_price selling_price gross_profit : ℕ) : ℕ :=
  100 * (selling_price - purchase_price) / selling_price

theorem jacket_markup
  (purchase_price discount_percent gross_profit selling_price0 : ℕ)
  (h_purchase_price : purchase_price = 48)
  (h_discount_percent : discount_percent = 20)
  (h_gross_profit : gross_profit = 16)
  (h_selling_price : selling_price0 = (64 * 5 / 4)) :
  markup_percentage purchase_price selling_price0 gross_profit = 40 :=
by
  have h : selling_price0 = 80 := by simp [h_selling_price]
  rw [markup_percentage, h]
  have h32 : 100 * (80 - 48) / 80 = 40 := by norm_num
  exact h32

end jacket_markup_l720_720613


namespace ladybugs_total_total_ladybugs_is_5_l720_720850

def num_ladybugs (x y : ℕ) : ℕ :=
  x + y

theorem ladybugs_total (x y n : ℕ) 
    (h_spot_calc_1: 6 * x + 4 * y = 30 ∨ 6 * x + 4 * y = 26)
    (h_total_spots_30: (6 * x + 4 * y = 30) ↔ 3 * x + 2 * y = 15)
    (h_total_spots_26: (6 * x + 4 * y = 26) ↔ 3 * x + 2 * y = 13)
    (h_truth_only_one: 
       (6 * x + 4 * y = 30 ∧ ¬(6 * x + 4 * y = 26)) ∨
       (¬(6 * x + 4 * y = 30) ∧ 6 * x + 4 * y = 26))
    : n = x + y :=
by 
  sorry

theorem total_ladybugs_is_5 : ∃ x y : ℕ, num_ladybugs x y = 5 :=
  ⟨3, 2, rfl⟩

end ladybugs_total_total_ladybugs_is_5_l720_720850


namespace units_digit_m_squared_plus_3_pow_m_l720_720889

def m := 2023^2 + 3^2023

theorem units_digit_m_squared_plus_3_pow_m : 
  (m^2 + 3^m) % 10 = 5 := sorry

end units_digit_m_squared_plus_3_pow_m_l720_720889


namespace negation_of_exists_l720_720568

theorem negation_of_exists {x : ℝ} (h : ∃ x : ℝ, 3^x + x < 0) : ∀ x : ℝ, 3^x + x ≥ 0 :=
sorry

end negation_of_exists_l720_720568


namespace inscribed_circle_radius_l720_720570

theorem inscribed_circle_radius (A B C : Point) (AB AC BC : ℝ) (hAB : AB = 8) (hAC : AC = 8) (hBC : BC = 5) : 
  radius_of_inscribed_circle_in_triangle ABC AB AC BC = 76 * Real.sqrt 10 / 21 :=
by {
  sorry -- Proof goes here.
}

end inscribed_circle_radius_l720_720570


namespace numberOfSuchTriangles_l720_720783

noncomputable def numberOfTriangles (b c : ℕ) := 
  b <= 5 ∧ 5 <= c ∧ c - b < 5

theorem numberOfSuchTriangles : 
  ∃ n, n = 15 ∧ (numberOfTriangles n b c) :=
sorry

end numberOfSuchTriangles_l720_720783


namespace symmetric_about_y_eq_x_l720_720771

noncomputable theory

open Real

variables {a b : ℝ}
-- Given conditions
def conditions (a b : ℝ) : Prop :=
  log 2 a + log 2 b = 0 ∧ a > 0 ∧ b > 0 ∧ a ≠ 1 ∧ b ≠ 1

-- function definitions
def f (x : ℝ) (a : ℝ) : ℝ := a^x
def g (x : ℝ) (b : ℝ) : ℝ := -log b x

-- The statement to prove the symmetry about the line y = x
theorem symmetric_about_y_eq_x (a b : ℝ) (cond : conditions a b) :
  (∀ x y : ℝ, f x a = y ↔ g y b = x) :=
begin
  sorry
end

end symmetric_about_y_eq_x_l720_720771


namespace max_subsets_of_N_l720_720350

def M : set ℕ := {0, 2, 3, 7}
def N : set ℕ := {x | ∃ a b, a ∈ M ∧ b ∈ M ∧ x = a * b }
def num_subsets (A : set ℕ) := 2 ^ A.to_finset.card

theorem max_subsets_of_N : num_subsets N = 128 :=
by
  sorry

end max_subsets_of_N_l720_720350


namespace points_bounds_l720_720754

theorem points_bounds (n k : ℕ) (h_n_pos : n > 0) (h_k_pos : k > 0)
  (S : Finset (ℝ × ℝ)) (hS_card : S.card = n)
  (h_no_three_collinear : ∀ (P₁ P₂ P₃ : ℝ × ℝ), P₁ ∈ S → P₂ ∈ S → P₃ ∈ S → 
    P₁ ≠ P₂ → P₁ ≠ P₃ → P₂ ≠ P₃ → ¬ collinear ({P₁, P₂, P₃} : Set (ℝ × ℝ)))
  (h_equidistant : ∀ P ∈ S, (Finset.filter (λ Q, dist P Q = (Finset.filter (λ R, R = Q) S).card) S).card = k) :
  k ≤ ⌊ (1 / 2) + sqrt (2 * n) ⌋ := sorry

end points_bounds_l720_720754


namespace systematic_sampling_interval_l720_720037

theorem systematic_sampling_interval 
  (N : ℕ) (n : ℕ) (hN : N = 630) (hn : n = 45) :
  N / n = 14 :=
by {
  sorry
}

end systematic_sampling_interval_l720_720037


namespace petya_time_comparison_l720_720224

open Real

noncomputable def petya_planned_time (D V : ℝ) := D / V

noncomputable def petya_actual_time (D V : ℝ) :=
  let V1 := 1.25 * V
  let V2 := 0.80 * V
  let T1 := (D / 2) / V1
  let T2 := (D / 2) / V2
  T1 + T2

theorem petya_time_comparison (D V : ℝ) (hV : V > 0) : 
  petya_actual_time D V > petya_planned_time D V :=
by {
  let T := petya_planned_time D V
  let T_actual := petya_actual_time D V
  have h1 : petya_planned_time D V = D / V, by unfold petya_planned_time
  have h2 : petya_actual_time D V = (D * 41) / (40 * V), by {
      unfold petya_actual_time,
      have h3 : 1.25 * V = 5 * V / 4, by linarith,
      have h4 : 0.80 * V = 4 * V / 5, by linarith,
      rw [h3, h4],
      simp,
      linarith,
  },
  rw h1,
  rw h2,
  have h3 : (41 * D) / (40 * V) > D / V, by linarith,
  exact h3,
}

end petya_time_comparison_l720_720224


namespace angle_EDF_is_120_degrees_l720_720377

theorem angle_EDF_is_120_degrees 
  (A B C D E F : Type) 
  [IncidenceGeometry A B C D E F]
  (midpoint_E : Midpoint E A C) 
  (midpoint_F : Midpoint F A B) 
  (circle_passing : CirclePassingThrough A E F) 
  (circle_tangent : CircleTangent D B C)
  (ratio_condition : (AB / AC) + (AC / AB) = 4) :
  ∠ EDF = 120 :=
begin
  sorry
end

end angle_EDF_is_120_degrees_l720_720377


namespace find_pool_length_l720_720001

noncomputable def pool_length : ℝ :=
  let drain_rate := 60 -- cubic feet per minute
  let width := 40 -- feet
  let depth := 10 -- feet
  let capacity_percent := 0.80
  let drain_time := 800 -- minutes
  let drained_volume := drain_rate * drain_time -- cubic feet
  let full_capacity := drained_volume / capacity_percent -- cubic feet
  let length := full_capacity / (width * depth) -- feet
  length

theorem find_pool_length : pool_length = 150 := by
  sorry

end find_pool_length_l720_720001


namespace triangle_area_24_l720_720565

open_locale big_operators

def abs_diff (a b : ℝ) : ℝ :=
|a - b|

def triangle_area (base height : ℝ) : ℝ :=
0.5 * base * height

theorem triangle_area_24 :
  let A := (3, 2)
  let B := (3, -4)
  let C := (11, 2)
  triangle_area (abs_diff 2 (-4)) (abs_diff 11 3) = 24 := 
by
  have base : ℝ := abs_diff 2 (-4)
  have height : ℝ := abs_diff 11 3
  sorry

end triangle_area_24_l720_720565


namespace pairs_xy_solution_sum_l720_720246

theorem pairs_xy_solution_sum :
  ∃ (x y : ℝ) (a b c d : ℕ), 
    x + y = 5 ∧ 2 * x * y = 5 ∧ 
    (x = (5 + Real.sqrt 15) / 2 ∨ x = (5 - Real.sqrt 15) / 2) ∧ 
    a = 5 ∧ b = 1 ∧ c = 15 ∧ d = 2 ∧ a + b + c + d = 23 :=
by
  sorry

end pairs_xy_solution_sum_l720_720246


namespace problem1_problem2_problem3_problem4_problem5_problem6_l720_720516

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end problem1_problem2_problem3_problem4_problem5_problem6_l720_720516


namespace smallest_five_digit_number_divisible_by_five_primes_l720_720728

theorem smallest_five_digit_number_divisible_by_five_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  lcm = 2310 → (∃ n : ℕ, n = 5 ∧ 10000 ≤ lcm * n ∧ lcm * n = 11550) :=
by
  intros p1 p2 p3 p4 p5 
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  intro hlcm
  use (5 : ℕ)
  split
  { exact rfl }
  split
  { sorry }
  { sorry }

end smallest_five_digit_number_divisible_by_five_primes_l720_720728


namespace length_of_train_is_750m_l720_720015

-- Defining the conditions
def train_and_platform_equal_length : Prop := ∀ (L : ℝ), (Length_of_train = L ∧ Length_of_platform = L)
def train_speed := 90 * (1000 / 3600)  -- Convert speed from km/hr to m/s
def crossing_time := 60  -- Time given in seconds

-- Definition for the length of the train
def Length_of_train := sorry -- Given that it should be derived

-- The proof problem statement
theorem length_of_train_is_750m : (train_and_platform_equal_length ∧ train_speed ∧ crossing_time → Length_of_train = 750) :=
by
  -- Proof is skipped
  sorry

end length_of_train_is_750m_l720_720015


namespace decimal_to_fraction_equivalence_l720_720049

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l720_720049


namespace train_length_l720_720041

def relative_speed (v_fast v_slow : ℕ) : ℚ :=
  v_fast - v_slow

def convert_speed (speed : ℚ) : ℚ :=
  (speed * 1000) / 3600

def covered_distance (speed : ℚ) (time_seconds : ℚ) : ℚ :=
  speed * time_seconds

theorem train_length (L : ℚ) (v_fast v_slow : ℕ) (time_seconds : ℚ)
    (hf : v_fast = 42) (hs : v_slow = 36) (ht : time_seconds = 36)
    (hc : relative_speed v_fast v_slow * 1000 / 3600 * time_seconds = 2 * L) :
    L = 30 := by
  sorry

end train_length_l720_720041


namespace largest_expression_l720_720403

noncomputable def x : ℝ := 10 ^ (-2024 : ℤ)

theorem largest_expression :
  let a := 5 + x
  let b := 5 - x
  let c := 5 * x
  let d := 5 / x
  let e := x / 5
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by
  sorry

end largest_expression_l720_720403


namespace inequality_a_inequality_c_inequality_d_l720_720474

variable (a b : ℝ)

theorem inequality_a (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 := 
Sorry

theorem inequality_c (h : a + b > 0) : a^21 + b^21 > 0 := 
Sorry

theorem inequality_d (h : a + b > 0) : (a + 2) * (b + 2) > a * b := 
Sorry

end inequality_a_inequality_c_inequality_d_l720_720474


namespace golden_ratio_dot_product_l720_720668

theorem golden_ratio_dot_product
  (A B C D : EuclideanGeometry.Point)
  (hB : B ≠ A) (hC : C ≠ A) (h_angle_60 : ∠ BAC = 60)
  (h_AB : dist A B = 2) (h_AC : dist A C = 3)
  (h_ratio : dist B D / dist B C = (Real.sqrt 5 - 1) / 2)
  (h_D : B ≠ D) :
  (A -ᵥ D) • (C -ᵥ B) = (7 * Real.sqrt 5 - 9) / 2 :=
begin
  sorry
end

end golden_ratio_dot_product_l720_720668


namespace unique_bad_configurations_l720_720465

def is_bad_configuration (arr : list ℕ) : Prop :=
  let sums := set.image (λ s : list ℕ, list.sum s) (list.sublists arr) in
  ∃ n ∈ (set.range (λ s : fin 22, s.val)), n ∉ sums

theorem unique_bad_configurations :
  let numbers := [1, 3, 4, 6, 7] in
  let circular_arrangements := list.permutations numbers in
  let unique_arrangements := 
    (list.nodup (list.erase_dup 
      (list.map (λ arr, if list.mem arr (circular_arrangements ++ list.map list.reverse circular_arrangements) 
                then arr else [] ) circular_arrangements ))) in
  list.countp is_bad_configuration unique_arrangements = 3 := 
sorry

end unique_bad_configurations_l720_720465


namespace coefficient_x3_in_expansion_l720_720373

theorem coefficient_x3_in_expansion :
  let f := (λ x : ℝ, (x^2 + 1)^2 * (x - 1)^6)
  (polynomial.coeff (polynomial.CX 3) (polynomial.eval 2 (f x))) = -32 :=
begin
  sorry
end

end coefficient_x3_in_expansion_l720_720373


namespace find_f_of_1_sub_sqrt_2_l720_720782

noncomputable def f (x : ℝ) : ℝ := 
if x ∈ (Set.Icc 0 1) then Real.log (x + 1) / Real.log 2 else
  if x > 0 then f (-x) else -f (-x)

theorem find_f_of_1_sub_sqrt_2 :
  (f (1 - Real.sqrt 2) = -1 / 2) :=
by
  -- applying the conditions and properties of the function
  sorry

end find_f_of_1_sub_sqrt_2_l720_720782


namespace sum_of_digits_l720_720402

noncomputable def g_k (k x : ℕ) : ℕ :=
  x^2 / 10^k

noncomputable def x_n (n : ℕ) : ℕ :=
  Nat.find (λ a, g_k (2*n) a - g_k (2*n) (a-1) ≥ 2)

noncomputable def f (k : ℕ) : ℕ :=
  if h : k % 2 = 0 then
    let n := k / 2
    in (25 * 10^(2 * n - 2) + 10^n)
  else 
    0

theorem sum_of_digits (k : ℕ) :
  (\sum i in Finset.range (k / 2), f (2 * (i + 1))) = 3024 := sorry

end sum_of_digits_l720_720402


namespace distinct_values_f_l720_720900

noncomputable def f (x : ℝ) : ℝ :=
  ∑ k in Finset.range 13 + 3, (Int.floor (k * x^2) - k * Int.floor (x^2))

theorem distinct_values_f : ∀ x ≥ 0, ∃ n, n = 67 ∧
  ∀ y, f x = y → y ∈ {0, ..., n - 1} :=
by
  sorry

end distinct_values_f_l720_720900


namespace find_t_l720_720833

theorem find_t (s t : ℤ) (h1 : 9 * s + 5 * t = 108) (h2 : s = t - 2) : t = 9 :=
sorry

end find_t_l720_720833


namespace range_x_l720_720311

variable {R : Type*} [LinearOrderedField R]

def monotone_increasing_on (f : R → R) (s : Set R) := ∀ ⦃a b⦄, a ≤ b → f a ≤ f b

theorem range_x 
    (f : R → R) 
    (h_mono : monotone_increasing_on f Set.univ) 
    (h_zero : f 1 = 0) 
    (h_ineq : ∀ x, f (x^2 + 3 * x - 3) < 0) :
  ∀ x, -4 < x ∧ x < 1 :=
by 
  sorry

end range_x_l720_720311


namespace min_cost_hand_sanitizer_l720_720251

theorem min_cost_hand_sanitizer (x : ℕ) (hx₁ : 15 ≤ x) (hx₂ : x ≤ 35) :
  let price_A := 202 - 2 * x,
      price_B := 100,
      qty_A := x,
      qty_B := 50 - x,
      total_cost := (202 - 2 * x) * x + 100 * (50 - x)
  in total_cost = -2 * x^2 + 102 * x + 5000 ∧ 
     (total_cost ≤ (202 - 2 * 15) * 15 + 100 * (50 - 15) ∧ 
      total_cost ≤ (202 - 2 * 35) * 35 + 100 * (50 - 35) →
      x = 35) :=
by
  sorry

end min_cost_hand_sanitizer_l720_720251


namespace frank_problems_each_type_l720_720644

theorem frank_problems_each_type (bill_total : ℕ) (ryan_ratio bill_total_ratio : ℕ) (frank_ratio ryan_total : ℕ) (types : ℕ)
  (h1 : bill_total = 20)
  (h2 : ryan_ratio = 2)
  (h3 : bill_total_ratio = bill_total * ryan_ratio)
  (h4 : ryan_total = bill_total_ratio)
  (h5 : frank_ratio = 3)
  (h6 : ryan_total * frank_ratio = ryan_total) :
  (ryan_total * frank_ratio) / types = 30 :=
by
  sorry

end frank_problems_each_type_l720_720644


namespace inequality_cannot_hold_l720_720318

noncomputable def f (x : ℝ) : ℝ := (1 / Real.exp 1) ^ x + Real.log x

variables {a b c x0 : ℝ}

theorem inequality_cannot_hold 
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) 
  (h_abc : a < b < c)
  (h_fabc : f(a) * f(b) * f(c) > 0)
  (h_x0_sol : f(x0) = 0) : 
  ¬ (x0 > c) :=
sorry

end inequality_cannot_hold_l720_720318


namespace kim_cousins_l720_720876

theorem kim_cousins (pieces_per_cousin : ℕ) (total_pieces : ℕ) (h_pieces_per_cousin : pieces_per_cousin = 5) (h_total_pieces : total_pieces = 20) :
  total_pieces / pieces_per_cousin = 4 :=
by
  rw [h_pieces_per_cousin, h_total_pieces]
  norm_num

end kim_cousins_l720_720876


namespace find_linear_function_l720_720760

theorem find_linear_function (k b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = k * x + b) 
  (h2 : -3 ≤ x ∧ x ≤ 1) 
  (h3 : ∀ x, -3 ≤ x ∧ x ≤ 1 → 1 ≤ f x ∧ f x ≤ 9): 
  f = (λ x, 2 * x + 7) ∨ f = (λ x, -2 * x + 3) :=
by {
  sorry
}

end find_linear_function_l720_720760


namespace monotonically_increasing_interval_max_min_values_on_interval_l720_720798

noncomputable def f (x : ℝ) : ℝ :=
  cos x * sin (x + π / 6) - cos (2 * x) - 1 / 4

theorem monotonically_increasing_interval (k : ℤ) :
  ∀ x, (x ≥ k * π - π / 12 ∧ x ≤ k * π + 5 * π / 12) →
  (∀ y, (y ≥ k * π - π / 12 ∧ y ≤ k * π + 5 * π / 12) → (x ≤ y → f x ≤ f y)) :=
sorry

theorem max_min_values_on_interval :
  let interval := set.Icc (-π / 6 : ℝ) (π / 4)
  let max_val := sqrt 3 / 4
  let min_val := -sqrt 3 / 2
  ∃ x y ∈ interval, f x = max_val ∧ f y = min_val :=
sorry

end monotonically_increasing_interval_max_min_values_on_interval_l720_720798


namespace part_a_part_b_l720_720383

def regular_polygon (n : ℕ) := {v : ℕ // 1 ≤ v ∧ v ≤ n}

-- Condition: Vertices belong to a regular 14-gon
def is_vertex_of_14_gon (v : ℕ) : Prop := v ∈ regular_polygon 14

-- Predicate for a quadrilateral being a rectangle
def is_rectangle (a b c d : ℕ) : Prop := sorry -- Define appropriate geometric properties

-- Check if a quadrilateral has two parallel sides
def has_two_parallel_sides (a b c d : ℕ) : Prop :=
  sorry -- Include logic for identifying parallel sides

-- Question translation: Given k vertices, can mark vertices make every quadrilateral with two parallel sides a rectangle?
def can_mark_vertices (k : ℕ) : Prop :=
  ∀ (v : Finset ℕ) (h : v.card = k),
  ∀ (a b c d ∈ v), has_two_parallel_sides a b c d → is_rectangle a b c d

theorem part_a : can_mark_vertices 6 := 
  sorry -- Proof that it is true for k = 6

theorem part_b : ∀ k, k ≥ 7 → ¬ can_mark_vertices k := 
  sorry -- Proof that it is false for k ≥ 7

end part_a_part_b_l720_720383


namespace domain_of_inverse_function_l720_720456

noncomputable def f (x : ℝ) : ℝ := log x / log 2 + 1

def domain_f_inv : set ℝ := {y | 3 ≤ y}

theorem domain_of_inverse_function :
  (∀ x : ℝ, x ≥ 4 → ∃ y, f y = x) →
  ∀ y : ℝ, y ∈ domain_f_inv ↔ (∃ x, f x = y) :=
by
  intros h y
  simp [domain_f_inv, f]
  sorry

end domain_of_inverse_function_l720_720456


namespace decimal_to_fraction_l720_720078

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l720_720078


namespace harold_savings_l720_720826

theorem harold_savings :
  let income_primary := 2500
  let income_freelance := 500
  let rent := 700
  let car_payment := 300
  let car_insurance := 125
  let electricity := 0.25 * car_payment
  let water := 0.15 * rent
  let internet := 75
  let groceries := 200
  let miscellaneous := 150
  let total_income := income_primary + income_freelance
  let total_expenses := rent + car_payment + car_insurance + electricity + water + internet + groceries + miscellaneous
  let amount_before_savings := total_income - total_expenses
  let retirement := (1/3) * amount_before_savings
  let emergency := (1/3) * amount_before_savings
  let amount_after_savings := amount_before_savings - retirement - emergency
  amount_after_savings = 423.34 := 
sorry

end harold_savings_l720_720826


namespace eval_expr_l720_720621

noncomputable def a : ℕ → ℤ
| 0       := 1
| (n + 1) := if n + 1 ≤ 5 then n + 1 else (a n * a 0 * a (n - 1) * a (n - 2) * a (n - 3)) - 1

def prod_a (n : ℕ) : ℤ := list.prod (list.map ((^) a) (list.range n))

def sum_a_sq (n : ℕ) : ℤ := list.sum (list.map (λ i, (a i) ^ 2) (list.range n))

theorem eval_expr : prod_a 2011 - sum_a_sq 2011 = -1941 :=
by
    sorry

end eval_expr_l720_720621


namespace calculate_value_l720_720233

theorem calculate_value :
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 :=
by
  sorry

end calculate_value_l720_720233


namespace print_time_l720_720171

-- Define the conditions
def pages : ℕ := 345
def rate : ℕ := 23
def expected_minutes : ℕ := 15

-- State the problem as a theorem
theorem print_time (pages rate : ℕ) : (pages / rate = 15) :=
by
  sorry

end print_time_l720_720171


namespace P_X_leq_36_eq_P_Y_leq_36_plan_arrive_before_734_bus_plan_arrive_before_740_bike_l720_720414

-- Definitions for the given conditions:
def mean_bus : ℝ := 30
def var_bus : ℝ := 36
def mean_bike : ℝ := 34
def var_bike : ℝ := 4

-- Assuming normal distributions for bus times and bike times:
def X : pmf ℝ := pmf.primitive (λ x, exp (-((x - mean_bus)^2) / (2 * var_bus)) / (sqrt (2 * π * var_bus)))
def Y : pmf ℝ := pmf.primitive (λ y, exp (-((y - mean_bike)^2) / (2 * var_bike)) / (sqrt (2 * π * var_bike)))

-- The theorem statements:
theorem P_X_leq_36_eq_P_Y_leq_36 : 
  (X.prob (λ x, x ≤ 36)) = (Y.prob (λ y, y ≤ 36)) :=
sorry

theorem plan_arrive_before_734_bus : 
  (X.prob (λ x, x ≤ 34)) > (Y.prob (λ y, y ≤ 34)) :=
sorry

theorem plan_arrive_before_740_bike : 
  (Y.prob (λ y, y ≤ 40)) > (X.prob (λ x, x ≤ 40)) :=
sorry

end P_X_leq_36_eq_P_Y_leq_36_plan_arrive_before_734_bus_plan_arrive_before_740_bike_l720_720414


namespace difference_longest_shortest_trip_l720_720461

variables (F S R G : Type) [metric_space F] [metric_space S] [metric_space R] [metric_space G]

axiom dist_FS : dist F S = 5
axiom dist_SR : dist S R = 12
axiom dist_SG : dist S G = 9
axiom dist_FR : dist F R = Math.sqrt (5^2 + 12^2)
axiom dist_GR : dist G R = Math.sqrt (9^2 + 12^2)

noncomputable def route1 : ℝ := dist F R + dist R S + dist S G
noncomputable def route2 : ℝ := dist F S + dist S G + dist G R
noncomputable def route3 : ℝ := dist F R + dist R G + dist G S
noncomputable def route4 : ℝ := dist F S + dist S R + dist R G

theorem difference_longest_shortest_trip : 
    max (max route1 route2) (max route3 route4) - min (min route1 route2) (min route3 route4) = 8 := 
sorry

end difference_longest_shortest_trip_l720_720461


namespace printing_time_345_l720_720175

def printing_time (total_pages : ℕ) (rate : ℕ) : ℕ :=
  total_pages / rate

theorem printing_time_345 :
  printing_time 345 23 = 15 :=
by
  sorry

end printing_time_345_l720_720175


namespace problem1_problem2_l720_720600

open Nat

def binomial (n k : ℕ) := n.factorial / (k.factorial * (n - k).factorial)
def permutation (n k : ℕ) := n.factorial / (n - k).factorial

theorem problem1 : binomial 10 4 - binomial 7 3 * permutation 3 3 = 0 := sorry

theorem problem2 (x : ℕ) (h : 3 * permutation 8 x = 4 * permutation 9 (x - 1)) : x = 6 := sorry

end problem1_problem2_l720_720600


namespace find_b_l720_720770

theorem find_b 
  (a b : ℚ)
  (h_root : (1 + Real.sqrt 5) ^ 3 + a * (1 + Real.sqrt 5) ^ 2 + b * (1 + Real.sqrt 5) - 60 = 0) :
  b = 26 :=
sorry

end find_b_l720_720770


namespace geometry_proof_l720_720431

noncomputable def problem_statement
  (A B C A₁ A₂ B₁ B₂ C₁ C₂ P O : Point)
  (BC CA AB : Line)
  (R : ℝ) : Prop :=
  (A₁ ∈ BC) ∧ (A₂ ∈ BC) ∧ (B₁ ∈ CA) ∧ (B₂ ∈ CA) ∧ (C₁ ∈ AB) ∧ (C₂ ∈ AB) ∧
  (parallel (line_through A₁ A₂) BC) ∧
  (parallel (line_through B₁ B₂) CA) ∧
  (parallel (line_through C₁ C₂) AB) ∧
  (intersects (line_through A₁ A₂) (line_through B₁ B₂) P) ∧
  (intersects (line_through B₁ B₂) (line_through C₁ C₂) P) ∧
  (circumcenter A B C O) ∧ (circumradius A B C R) →
  (dist P A₁ * dist P A₂ + dist P B₁ * dist P B₂ + dist P C₁ * dist P C₂ = R ^ 2 - dist O P ^ 2)

theorem geometry_proof (A B C A₁ A₂ B₁ B₂ C₁ C₂ P O : Point)
  (BC CA AB : Line)
  (R : ℝ) 
  (h1 : A₁ ∈ BC) (h2 : A₂ ∈ BC) (h3 : B₁ ∈ CA) (h4 : B₂ ∈ CA) 
  (h5 : C₁ ∈ AB) (h6 : C₂ ∈ AB) 
  (h7 : parallel (line_through A₁ A₂) BC)
  (h8 : parallel (line_through B₁ B₂) CA)
  (h9 : parallel (line_through C₁ C₂) AB)
  (h10 : intersects (line_through A₁ A₂) (line_through B₁ B₂) P)
  (h11 : intersects (line_through B₁ B₂) (line_through C₁ C₂) P)
  (h12 : circumcenter A B C O)
  (h13 : circumradius A B C R) :
  dist P A₁ * dist P A₂ + dist P B₁ * dist P B₂ + dist P C₁ * dist P C₂ = R ^ 2 - dist O P ^ 2 := 
sorry

end geometry_proof_l720_720431


namespace smallest_positive_five_digit_number_divisible_by_first_five_primes_l720_720717

theorem smallest_positive_five_digit_number_divisible_by_first_five_primes :
  ∃ n : ℕ, (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ 10000 ≤ n ∧ n = 11550 :=
by
  use 11550
  split
  · intros p hp
    fin_cases hp <;> norm_num
  split
  · norm_num
  rfl

end smallest_positive_five_digit_number_divisible_by_first_five_primes_l720_720717


namespace ministers_receive_decree_l720_720547

variable (n : Nat)

def unique_decrees (ministers : Fin n → Set Nat) : Prop :=
  ∀ i j, i ≠ j → ministers i ≠ ministers j

def telegrams_exchange (ministers : Fin n → Set Nat) (telegrams : List (Fin n × Fin n)) : Prop :=
  ∀ t ∈ telegrams, (telegrams.filter (λ p, p.fst = t.fst)).length = 1

def all_ministers_informed (ministers : Fin n → Set Nat) : Prop :=
  ∀ i, ∀ m ∈ ministers i, (∃ t ∈ telegrams, t.snd = i)

theorem ministers_receive_decree :
  ∀ (ministers : Fin n → Set Nat) (telegrams : List (Fin n × Fin n)),
    unique_decrees ministers →
    telegrams_exchange ministers telegrams →
    all_ministers_informed ministers telegrams →
    ∃ k, k ≥ n - 1 ∧ (∃ received : Fin k → Bool, ∀ i : Fin k, received i = true) :=
by
  sorry

end ministers_receive_decree_l720_720547


namespace MN_parallel_PQ_l720_720947

-- Definitions and conditions
variable {A B C I Z M N P Q : Type}
variable [is_center_of_circumcircle I A B C]
variable [midpoint_of_arc I A C]
variable [angle_bisector_of IZ (∠ B)]
variable [perpendicular PQ IZ]
variable [isosceles_triangle M BI]
variable [isosceles_triangle N BI]

-- The theorem statement
theorem MN_parallel_PQ (h_center: is_center_of_circumcircle I A B C) 
  (h_midpoint: midpoint_of_arc I A C) (h_bisector: angle_bisector_of IZ (∠ B)) 
  (h_perpendicular: perpendicular PQ IZ) (h_mbi: isosceles_triangle M BI) 
  (h_nbi: isosceles_triangle N BI) : 
  parallel MN PQ :=
sorry

end MN_parallel_PQ_l720_720947


namespace decimal_to_fraction_l720_720099

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l720_720099


namespace triangle_product_eq_21_l720_720039

noncomputable def triangle_product (A B C P Q D : ℝ) (BQ DQ : ℝ) :=
  ∃ (AD CD : ℝ), 
    (P = B) = (P = C) ∧
    ∠BPQ = 2 * ∠BCA ∧ 
    (D ∈ line AC) ∧ 
    (D ∈ line BQ) ∧ 
    BQ = 10 ∧ 
    DQ = 7 → 
    AD * CD = 21

-- Lean 4 statement
theorem triangle_product_eq_21 (A B C P Q D : ℝ) (BQ DQ : ℝ) :
  triangle_product A B C P Q D BQ DQ :=
sorry

end triangle_product_eq_21_l720_720039


namespace parallel_lines_l720_720952

-- Definition of points and lines
variables {A B C M N P Q I Z : Type} [geometry_space : Geometry A B C M N P Q I Z]

-- Conditions
def is_center_of_circumcircle (A B C I : Type) : Prop := Geometry.is_center_of_circumcircle A B C I
def midpoint_of_arc_AC (A C I : Type) : Prop := Geometry.midpoint_ω A C I
def is_angle_bisector (IZ : Type) (B : Type) : Prop := Geometry.angle_bisector IZ B
def is_perpendicular (PQ IZ : Type) : Prop := Geometry.perpendicular PQ IZ
def is_isosceles (MBI NBI BI : Type) : Prop := Geometry.isosceles MBI NBI BI

-- Problem Statement
theorem parallel_lines (A B C M N P Q I Z : Type)
  [is_center_of_circumcircle A B C I]
  [midpoint_of_arc_AC A C I]
  [is_angle_bisector IZ B]
  [is_perpendicular PQ IZ]
  [is_isosceles MBI NBI BI] : PQ ∥ MN :=
by
  sorry

end parallel_lines_l720_720952


namespace arithmetic_sequence_a4_l720_720853

theorem arithmetic_sequence_a4 (a1 : ℤ) (S3 : ℤ) (h1 : a1 = 3) (h2 : S3 = 15) : 
  ∃ (a4 : ℤ), a4 = 9 :=
by
  sorry

end arithmetic_sequence_a4_l720_720853


namespace correct_reaction_for_phosphoric_acid_l720_720615

-- Define the reactions
def reaction_A := "H₂ + 2OH⁻ - 2e⁻ = 2H₂O"
def reaction_B := "H₂ - 2e⁻ = 2H⁺"
def reaction_C := "O₂ + 4H⁺ + 4e⁻ = 2H₂O"
def reaction_D := "O₂ + 2H₂O + 4e⁻ = 4OH⁻"

-- Define the condition that the electrolyte used is phosphoric acid
def electrolyte := "phosphoric acid"

-- Define the correct reaction
def correct_negative_electrode_reaction := reaction_B

-- Theorem to state that given the conditions above, the correct reaction is B
theorem correct_reaction_for_phosphoric_acid :
  (∃ r, r = reaction_B ∧ electrolyte = "phosphoric acid") :=
by
  sorry

end correct_reaction_for_phosphoric_acid_l720_720615


namespace game_score_l720_720616

theorem game_score:
  ∃ (a b : ℕ), 0 < b ∧ b < a ∧ a < 1986 ∧
               (∀ x ≥ 1986, ∃ k m : ℕ, x = k * a + m * b) ∧
               ¬ (∃ k m : ℕ, 1985 = k * a + m * b) ∧
               ¬ (∃ k m : ℕ, 663 = k * a + m * b)  :=
begin
  use [332, 7],
  split,
  {
    exact nat.zero_lt_succ 6,
  },
  split,
  {
    exact lt_add_one 331,
  },
  split,
  {
    exact nat.lt_succ_self 1985,
  },
  split,
  {
    intros x hx,
    sorry,
  },
  split,
  {
    intro h,
    cases h with k m,
    sorry,
  },
  {
    intro h,
    cases h with k m,
    sorry,
  }
end

end game_score_l720_720616


namespace Tristan_wins_for_primality_l720_720890

noncomputable def Tristan_wins_game (p : ℕ) [hp : Fact (Nat.Prime p)] : Prop := 
  ∀ (X : ℕ) (a : ℕ → ℕ), (X ≥ 1) → (∀ n, a n > 0) → ∃ n, (¬(∃ k, let b := (a n * X % p) in b = k * p)) 

theorem Tristan_wins_for_primality (p1 p2 : ℕ) 
[Fact (p1 = 1000000007)] [Fact (p2 = 1000000009)] 
(hp1 : Nat.Prime p1) (hp2 : Nat.Prime p2) : 
  (Tristan_wins_game p1) ∧ (Tristan_wins_game p2) :=
by 
  sorry

end Tristan_wins_for_primality_l720_720890


namespace condition_sufficient_not_necessary_l720_720976

theorem condition_sufficient_not_necessary (x : ℝ) : (0 < x ∧ x < 5) → (|x - 2| < 3) ∧ (¬ ((|x - 2| < 3) → (0 < x ∧ x < 5))) :=
by
  sorry

end condition_sufficient_not_necessary_l720_720976


namespace max_chord_length_l720_720795

theorem max_chord_length 
  (θ : ℝ)
  (curve : ℝ → ℝ → Prop := λ x y, 2 * (2 * sin θ - cos θ + 3) * x^2 - (8 * sin θ + cos θ + 1) * y = 0)
  (line : ℝ → ℝ → Prop := λ x y, y = 2 * x) :
  ∃ l_max, l_max = 8 * sqrt 5 :=
by
  sorry

end max_chord_length_l720_720795


namespace smallest_five_digit_number_divisible_by_five_primes_l720_720727

theorem smallest_five_digit_number_divisible_by_five_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  lcm = 2310 → (∃ n : ℕ, n = 5 ∧ 10000 ≤ lcm * n ∧ lcm * n = 11550) :=
by
  intros p1 p2 p3 p4 p5 
  let lcm := Nat.lcm (Nat.lcm p1 p2) (Nat.lcm p3 (Nat.lcm p4 p5))
  intro hlcm
  use (5 : ℕ)
  split
  { exact rfl }
  split
  { sorry }
  { sorry }

end smallest_five_digit_number_divisible_by_five_primes_l720_720727


namespace decimal_to_fraction_l720_720104

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l720_720104


namespace march_five_mondays_l720_720444

theorem march_five_mondays (N : ℕ) (leap_year : N % 4 = 0) (feb_has_29_days : (leap_year → true)) (march_has_31_days : true) (feb_has_five_sundays : true) :
  ∃! d : string, d = "Monday" ∧ number_of_occurrences d N = 5 :=
sorry

end march_five_mondays_l720_720444


namespace triangle_is_obtuse_l720_720360

-- Definitions:
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides of the triangle

-- Given condition:
def sin_squared_cond (A B C : ℝ) : Prop :=
  Real.sin(A)^2 + Real.sin(B)^2 < Real.sin(C)^2

-- Desired conclusion:
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A + B + C = π ∧ -- Sum of angles in a triangle
  ∃ (A' B' C' : ℝ), 
    A' = A ∧ B' = B ∧ C' = C ∧ 
    (A > π / 2 ∨ B > π / 2 ∨ C > π / 2)

-- Theorem stating the proof problem:
theorem triangle_is_obtuse 
  (A B C : ℝ) 
  (h : sin_squared_cond A B C) : 
  is_obtuse_triangle A B C :=
sorry

end triangle_is_obtuse_l720_720360


namespace two_point_three_five_as_fraction_l720_720070

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l720_720070


namespace area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3_l720_720972

noncomputable def area_enclosed_by_sine_and_line : ℝ :=
  (∫ x in (Real.pi / 6)..(5 * Real.pi / 6), (Real.sin x - 1 / 2))

theorem area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3 :
  area_enclosed_by_sine_and_line = Real.sqrt 3 - Real.pi / 3 := by
  sorry

end area_between_sine_and_half_line_is_sqrt3_minus_pi_by_3_l720_720972


namespace three_inv_mod_191_l720_720680

theorem three_inv_mod_191 : ∃ x, 0 ≤ x ∧ x ≤ 190 ∧ 3 * x ≡ 1 [MOD 191] :=
by{
  use 64,
  split,
  -- 0 ≤ x
  exact Nat.zero_le 64,
  split,
  -- x ≤ 190
  exact le_of_lt (by norm_num),
  -- 3 * x ≡ 1 [MOD 191]
  sorry
}

end three_inv_mod_191_l720_720680


namespace decimal_to_fraction_l720_720084

theorem decimal_to_fraction (x : ℚ) (h : x = 2.35) : x = 47 / 20 :=
by sorry

end decimal_to_fraction_l720_720084


namespace quadrilateral_max_area_configuration_l720_720297

noncomputable def is_maximal_quadrilateral (K M N P : Point) (angle_K : Angle) : Prop :=
  let M_N := 1
  let N_P := 1
  in K M = K P ∧ is_angle_bisector K M N P

theorem quadrilateral_max_area_configuration (K M N P : Point) (angle_K : Angle)
  (hMN : dist M N = 1) (hNP : dist N P = 1) :
  is_maximal_quadrilateral K M N P angle_K :=
by
  sorry

end quadrilateral_max_area_configuration_l720_720297


namespace inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720482

variable {a b : ℝ}

theorem inequality_a (hab : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_b_not_true (hab : a + b > 0) : ¬(a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

theorem inequality_c (hab : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem inequality_d (hab : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

theorem inequality_e_not_true (hab : a + b > 0) : ¬((a − 3) * (b − 3) < a * b) :=
sorry

theorem inequality_f_not_true (hab : a + b > 0) : ¬((a + 2) * (b + 3) > a * b + 5) :=
sorry

end inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720482


namespace difference_of_squares_eval_l720_720676

-- Define the conditions
def a : ℕ := 81
def b : ℕ := 49

-- State the corresponding problem and its equivalence
theorem difference_of_squares_eval : (a^2 - b^2) = 4160 := by
  sorry -- Placeholder for the proof

end difference_of_squares_eval_l720_720676


namespace final_number_in_interval_l720_720432

theorem final_number_in_interval : 
  ∀ (k : ℝ), (∃ S : Finset ℝ, S = (Finset.range 2019).map (λ n, (n + 673 : ℝ)) ∧ ∀ n, n < 673 → perform_operation(S) = k → k ∈ (0, 1)) := 
sorry

end final_number_in_interval_l720_720432


namespace cannot_form_triangle_sets_l720_720197

theorem cannot_form_triangle_sets (A B C D : ℕ × ℕ × ℕ) 
    (hA : A = (3, 4, 5)) 
    (hB : B = (5, 10, 8)) 
    (hC : C = (5, 4.5, 8)) 
    (hD : D = (7, 7, 15)) :
    (¬ (A.1 + A.2 > A.3 ∧ A.1 + A.3 > A.2 ∧ A.2 + A.3 > A.1) ∨
    ¬ (B.1 + B.2 > B.3 ∧ B.1 + B.3 > B.2 ∧ B.2 + B.3 > B.1) ∨
    ¬ (C.1 + C.2 > C.3 ∧ C.1 + C.3 > C.2 ∧ C.2 + C.3 > C.1) ∨
    (D.1 + D.2 ≤ D.3 ∨ D.1 + D.3 ≤ D.2 ∨ D.2 + D.3 ≤ D.1)) = true :=  
by
  sorry

end cannot_form_triangle_sets_l720_720197


namespace inequality_a_inequality_c_inequality_d_l720_720497

variable {a b : ℝ}

axiom (h : a + b > 0)

theorem inequality_a : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_c : a^21 + b^21 > 0 :=
sorry

theorem inequality_d : (a + 2) * (b + 2) > a * b :=
sorry

end inequality_a_inequality_c_inequality_d_l720_720497


namespace find_k_for_inequality_l720_720263

theorem find_k_for_inequality {α : Type*} [Nonempty α] (a : ℕ → ℝ) (hpos : ∀ n, 0 < a n) :
  (∑ n : ℕ, n / (∑ i in finset.range (n + 1), a i)) ≤ 4 * (∑ n : ℕ, 1 / a n) := by
  sorry

end find_k_for_inequality_l720_720263


namespace regular_polygon_exterior_angle_l720_720200

theorem regular_polygon_exterior_angle (n : ℕ) (h : 1 ≤ n) :
  (360 : ℝ) / (n : ℝ) = 60 → n = 6 :=
by
  intro h1
  sorry

end regular_polygon_exterior_angle_l720_720200


namespace problem_a_problem_b_problem_c_l720_720525

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end problem_a_problem_b_problem_c_l720_720525


namespace train_length_equals_750_l720_720012

theorem train_length_equals_750
  (L : ℕ) -- length of the train in meters
  (v : ℕ) -- speed of the train in m/s
  (t : ℕ) -- time in seconds
  (h1 : v = 25) -- speed is 25 m/s
  (h2 : t = 60) -- time is 60 seconds
  (h3 : 2 * L = v * t) -- total distance covered by the train is 2L (train and platform) and equals speed * time
  : L = 750 := 
sorry

end train_length_equals_750_l720_720012


namespace peaches_picked_up_l720_720962

variable (initial_peaches : ℕ) (final_peaches : ℕ)

theorem peaches_picked_up :
  initial_peaches = 13 →
  final_peaches = 55 →
  final_peaches - initial_peaches = 42 :=
by
  intros
  sorry

end peaches_picked_up_l720_720962


namespace ordered_pairs_count_l720_720742

theorem ordered_pairs_count :
  let valid_pairs :=
    {p : ℕ × ℕ | 
      let b := p.fst in let c := p.snd in
        b > 0 ∧ c > 0 ∧
        b^2 = 4 * c ∧ c^2 ≤ 4 * b ∧ b + c ≤ 10} in
  finset.card valid_pairs = 2 :=
by
  let finset_pairs := finset.univ.filter (λ p : ℕ × ℕ, 
    let b := p.fst in let c := p.snd in
      b > 0 ∧ c > 0 ∧
      b^2 = 4 * c ∧ c^2 ≤ 4 * b ∧ b + c ≤ 10)
  show finset.card finset_pairs = 2
  sorry

end ordered_pairs_count_l720_720742


namespace range_a_two_zeros_l720_720781

-- Definition of the function f(x)
def f (a x : ℝ) : ℝ := a * x^3 - 3 * a * x + 3 * a - 5

-- The theorem statement about the range of a
theorem range_a_two_zeros (a : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) → 1 ≤ a ∧ a ≤ 5 := sorry

end range_a_two_zeros_l720_720781


namespace all_terms_are_perfect_squares_l720_720767

noncomputable def sequence (a : ℕ → ℤ) : Prop :=
(∀ n, 2 ≤ n → a (n+1) = 3 * a n - 3 * a (n-1) + a (n-2)) ∧
(2 * a 1 = a 0 + a 2 - 2) ∧
(∀ m, ∃ k, ∀ i, i < m → ∃ n, i = n ∧ ∃ t, a (k + i) = t * t)

theorem all_terms_are_perfect_squares (a : ℕ → ℤ) (h : sequence a) :
  ∀ n, ∃ t, a n = t * t := 
sorry

end all_terms_are_perfect_squares_l720_720767


namespace area_one_magnet_is_150_l720_720551

noncomputable def area_one_magnet : ℕ :=
  let length := 15
  let total_circumference := 70
  let combined_width := (total_circumference / 2 - length) / 2
  let width := combined_width
  length * width

theorem area_one_magnet_is_150 :
  area_one_magnet = 150 :=
by
  -- This will skip the actual proof for now
  sorry

end area_one_magnet_is_150_l720_720551


namespace sqrt_y_to_the_fourth_eq_256_l720_720580

theorem sqrt_y_to_the_fourth_eq_256 (y : ℝ) (h : (sqrt y)^4 = 256) : y = 16 := by
  sorry

end sqrt_y_to_the_fourth_eq_256_l720_720580


namespace part_a_part_c_part_d_l720_720506

-- Define the variables
variables {a b : ℝ}

-- Define the conditions and statements
def cond := a + b > 0

theorem part_a (h : cond) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem part_c (h : cond) : a^21 + b^21 > 0 :=
sorry

theorem part_d (h : cond) : (a + 2) * (b + 2) > a * b :=
sorry

end part_a_part_c_part_d_l720_720506


namespace peya_time_comparison_l720_720211

variable (V D : ℝ) (hV : 0 < V) (hD : 0 < D)

def planned_time : ℝ := D / V
def increased_speed : ℝ := 1.25 * V
def decreased_speed : ℝ := 0.80 * V

def first_half_distance : ℝ := D / 2
def second_half_distance : ℝ := D / 2

def time_first_half : ℝ := first_half_distance / increased_speed
def time_second_half : ℝ := second_half_distance / decreased_speed

def actual_time : ℝ := time_first_half + time_second_half

theorem peya_time_comparison : actual_time V D = (41 * D) / (40 * V) > (D / V) :=
by {
  unfold actual_time,
  unfold time_first_half time_second_half,
  unfold first_half_distance second_half_distance,
  unfold increased_speed decreased_speed,
  unfold planned_time,
  sorry
}

end peya_time_comparison_l720_720211


namespace grid_count_l720_720256

def is_valid_grid (grid : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  (∀ i, (∀ j₁ j₂, j₁ < j₂ → grid i j₁ < grid i j₂)) ∧
  (∀ j, (∀ i₁ i₂, i₁ < i₂ → grid i₁ j < grid i₂ j)) ∧
  {grid 0 0, grid 0 1, grid 0 2, grid 1 0, grid 1 1, grid 1 2, grid 2 0, grid 2 1, grid 2 2} = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  grid 0 2 = 3 ∧
  grid 2 2 = 4

theorem grid_count: (∃ grids : List (Matrix (Fin 3) (Fin 3) ℕ), 
  (∀ grid ∈ grids, is_valid_grid grid) ∧ 
  List.length grids = 6) :=
sorry

end grid_count_l720_720256


namespace problem_a_problem_b_problem_c_l720_720528

variable (a b : ℝ)

theorem problem_a {a b : ℝ} (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem problem_b {a b : ℝ} (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem problem_c {a b : ℝ} (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

end problem_a_problem_b_problem_c_l720_720528


namespace max_and_min_values_in_interval_l720_720984

noncomputable def f : ℝ → ℝ := λ x, 2 * x ^ 3 - 3 * x ^ 2 - 12 * x + 5

theorem max_and_min_values_in_interval :
  (∀ x ∈ set.Icc (0 : ℝ) 3, f x ≤ 5) ∧
  (∃ x ∈ set.Icc (0 : ℝ) 3, f x = 5) ∧
  (∀ x ∈ set.Icc (0 : ℝ) 3, f (-15) ≤ f x) ∧
  (∃ x ∈ set.Icc (0 : ℝ) 3, f x = -15) :=
by
  sorry

end max_and_min_values_in_interval_l720_720984


namespace unique_m_for_prime_condition_l720_720664

theorem unique_m_for_prime_condition :
  ∃ (m : ℕ), m > 0 ∧ (∀ (p : ℕ), Prime p → (∀ (n : ℕ), ¬ p ∣ (n^m - m))) ↔ m = 1 :=
sorry

end unique_m_for_prime_condition_l720_720664


namespace ratio_docking_to_license_l720_720419

noncomputable def Mitch_savings : ℕ := 20000
noncomputable def boat_cost_per_foot : ℕ := 1500
noncomputable def license_and_registration_fees : ℕ := 500
noncomputable def max_boat_length : ℕ := 12

theorem ratio_docking_to_license :
  let remaining_amount := Mitch_savings - license_and_registration_fees
  let cost_of_longest_boat := boat_cost_per_foot * max_boat_length
  let docking_fees := remaining_amount - cost_of_longest_boat
  docking_fees / license_and_registration_fees = 3 :=
by
  sorry

end ratio_docking_to_license_l720_720419


namespace students_with_one_talent_l720_720184

-- Define the given conditions
def total_students := 120
def cannot_sing := 30
def cannot_dance := 50
def both_skills := 10

-- Define the problem statement
theorem students_with_one_talent :
  (total_students - cannot_sing - both_skills) + (total_students - cannot_dance - both_skills) = 130 :=
by
  sorry

end students_with_one_talent_l720_720184


namespace MN_parallel_PQ_l720_720943

-- Definitions and conditions
variable {A B C I Z M N P Q : Type}
variable [is_center_of_circumcircle I A B C]
variable [midpoint_of_arc I A C]
variable [angle_bisector_of IZ (∠ B)]
variable [perpendicular PQ IZ]
variable [isosceles_triangle M BI]
variable [isosceles_triangle N BI]

-- The theorem statement
theorem MN_parallel_PQ (h_center: is_center_of_circumcircle I A B C) 
  (h_midpoint: midpoint_of_arc I A C) (h_bisector: angle_bisector_of IZ (∠ B)) 
  (h_perpendicular: perpendicular PQ IZ) (h_mbi: isosceles_triangle M BI) 
  (h_nbi: isosceles_triangle N BI) : 
  parallel MN PQ :=
sorry

end MN_parallel_PQ_l720_720943


namespace probability_laurent_greater_than_chloe_l720_720654

open ProbabilityTheory MeasurableSet

noncomputable def chloe_distribution : MeasureTheory.ProbabilityMeasure ℝ :=
  MeasureTheory.ProbabilityMeasure.uniform 0 1000

noncomputable def laurent_distribution : MeasureTheory.ProbabilityMeasure ℝ :=
  MeasureTheory.ProbabilityMeasure.uniform 0 3000

theorem probability_laurent_greater_than_chloe :
  ∫⁻ (x : ℝ) in chloe_distribution.toOuterMeasure.toMeasure, 
  ∫⁻ (y : ℝ) in laurent_distribution.toOuterMeasure.toMeasure,
  indicator (fun y => y > x) 1 = 5 / 6 :=
sorry

end probability_laurent_greater_than_chloe_l720_720654


namespace decimal_to_fraction_l720_720113

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l720_720113


namespace math_problem_l720_720471

def foo (a b : ℝ) (h : a + b > 0) : Prop :=
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧
  ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) ∧
  (a^21 + b^21 > 0) ∧
  ((a + 2) * (b + 2) > a * b) ∧
  ¬ ((a - 3) * (b - 3) < a * b) ∧
  ¬ ((a + 2) * (b + 3) > a * b + 5)

theorem math_problem (a b : ℝ) (h : a + b > 0) : foo a b h :=
by
  -- The proof will be here
  sorry

end math_problem_l720_720471


namespace mike_age_l720_720912

theorem mike_age : ∀ (m M : ℕ), m = M - 18 ∧ m + M = 54 → m = 18 :=
by
  intros m M
  intro h
  sorry

end mike_age_l720_720912


namespace smallest_five_digit_number_divisible_by_primes_l720_720732

theorem smallest_five_digit_number_divisible_by_primes : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
begin
  sorry
end

end smallest_five_digit_number_divisible_by_primes_l720_720732


namespace crafts_club_necklaces_l720_720422

theorem crafts_club_necklaces (members : ℕ) (total_beads : ℕ) (beads_per_necklace : ℕ)
  (h1 : members = 9) (h2 : total_beads = 900) (h3 : beads_per_necklace = 50) :
  (total_beads / beads_per_necklace) / members = 2 :=
by
  sorry

end crafts_club_necklaces_l720_720422


namespace radius_inequality_l720_720892

-- Definitions for the conditions
variables {T : Type} [Tetrahedron T]
# Prop to prove
theorem radius_inequality (r R : ℝ) (T : Tetrahedron) (insphere_radius : ℝ) (circumsphere_radius : ℝ) :
  insphere_radius = r → circumsphere_radius = R → 
  R ≥ 3 * r :=
sorry

end radius_inequality_l720_720892


namespace problem_f_2015_2016_l720_720768
-- Import all necessary Lean libraries

-- Define the function f along with the given conditions
axiom f : ℝ → ℝ
axiom odd_f : ∀ x : ℝ, f(-x) = -f(x)
axiom periodic_f : ∀ x : ℝ, f(x + 6) = f(x)
axiom f_one : f(1) = 1

-- Define the main theorem to prove
theorem problem_f_2015_2016 : f(2015) + f(2016) = -1 :=
by
  -- skip the proof
  sorry


end problem_f_2015_2016_l720_720768


namespace largest_prime_factor_of_sum_l720_720620

theorem largest_prime_factor_of_sum (s : List ℕ) (h₀ : ∀ x ∈ s, 1000 ≤ x ∧ x < 10000)
  (h₁ : ∀ i j, i < s.length → j < s.length → i + 1 = j → (s.get! i) % 1000 = (s.get! j) / 10)
  (h₂ : (s.last) % 1000 = (s.head) / 10) :
  ∃ m, 1111 * m = s.sum ∧ 101 ∣ s.sum :=
by sorry

end largest_prime_factor_of_sum_l720_720620


namespace peya_time_comparison_l720_720210

variable (V D : ℝ) (hV : 0 < V) (hD : 0 < D)

def planned_time : ℝ := D / V
def increased_speed : ℝ := 1.25 * V
def decreased_speed : ℝ := 0.80 * V

def first_half_distance : ℝ := D / 2
def second_half_distance : ℝ := D / 2

def time_first_half : ℝ := first_half_distance / increased_speed
def time_second_half : ℝ := second_half_distance / decreased_speed

def actual_time : ℝ := time_first_half + time_second_half

theorem peya_time_comparison : actual_time V D = (41 * D) / (40 * V) > (D / V) :=
by {
  unfold actual_time,
  unfold time_first_half time_second_half,
  unfold first_half_distance second_half_distance,
  unfold increased_speed decreased_speed,
  unfold planned_time,
  sorry
}

end peya_time_comparison_l720_720210


namespace find_common_ratio_l720_720459

variable {a : ℕ → ℝ}
variable {q : ℝ}

noncomputable def geometric_sequence_q (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 + a 4 = 20 ∧ a 3 + a 5 = 40

theorem find_common_ratio (h : geometric_sequence_q a q) : q = 2 :=
by
  sorry

end find_common_ratio_l720_720459


namespace drum_X_capacity_l720_720669

variable (C : ℝ) (hC : C > 0)
variable (X : ℝ)
variable (dualCapacity : ${1}/{2}$)
variable (initialFraction : ℝ := 2/5)
variable (finalFraction : ℝ := 0.45)

-- Given conditions
def drum_Y_initial : Prop := 2 * C * initialFraction
def drum_X_to_drum_Y : Prop := 2 * C * finalFraction

-- The goal to prove
theorem drum_X_capacity (h : X + drum_Y_initial = drum_X_to_drum_Y) : X = 0.5 * C :=
sorry

end drum_X_capacity_l720_720669


namespace two_point_three_five_as_fraction_l720_720065

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l720_720065


namespace petya_time_comparison_l720_720214

variables (D V : ℝ) (hD_pos : D > 0) (hV_pos : V > 0)

theorem petya_time_comparison (hD_pos : D > 0) (hV_pos : V > 0) :
  (41 * D / (40 * V)) > (D / V) :=
by
  sorry

end petya_time_comparison_l720_720214


namespace mod_inv_3_191_l720_720679

theorem mod_inv_3_191 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 190 ∧ (3 * x) % 191 = 1 :=
by
  use 64
  split
  . exact nat.zero_le 64
  . split
  . exact dec_trivial
  . exact dec_trivial

end mod_inv_3_191_l720_720679


namespace fibonacci_fifth_divisible_by_5_l720_720000

noncomputable def fibonacci : ℕ → ℕ
| 1 := 1
| 2 := 1
| (n + 1) := fibonacci n + fibonacci (n - 1)

theorem fibonacci_fifth_divisible_by_5 (k : ℕ) : 5 ∣ fibonacci (5 * k) :=
sorry

end fibonacci_fifth_divisible_by_5_l720_720000


namespace area_of_region_R_l720_720382

noncomputable def area_of_R_in_square : ℝ :=
  let square_side := 1
  let strip_area := (1 / 2 - 1 / 4) * square_side
  let triangle_height := sqrt 3 / 2
  let triangle_area := 1 / 2 * square_side * triangle_height
  let intersection_height := triangle_height - 1 / 4 * sqrt 3
  let intersection_area := 1 / 2 * (triangle_height + intersection_height) * (1 / 4)
  strip_area - intersection_area

theorem area_of_region_R : area_of_R_in_square = (4 - sqrt 3) / 16 :=
by
  sorry

end area_of_region_R_l720_720382


namespace part1_solution_set_part2_range_of_a_l720_720323

-- Define the function f for part 1 
def f_part1 (x : ℝ) : ℝ := |2*x + 1| + |2*x - 1|

-- Define the function f for part 2 
def f_part2 (x a : ℝ) : ℝ := |2*x + 1| + |a*x - 1|

-- Theorem for part 1
theorem part1_solution_set (x : ℝ) : 
  (f_part1 x) ≥ 3 ↔ x ∈ (Set.Iic (-3/4) ∪ Set.Ici (3/4)) :=
sorry

-- Theorem for part 2
theorem part2_range_of_a (a : ℝ) : 
  (a > 0) → (∃ x : ℝ, f_part2 x a < (a / 2) + 1) ↔ (a ∈ Set.Ioi 2) :=
sorry

end part1_solution_set_part2_range_of_a_l720_720323


namespace inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720489

variable {a b : ℝ}

theorem inequality_a (hab : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

theorem inequality_b_not_true (hab : a + b > 0) : ¬(a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

theorem inequality_c (hab : a + b > 0) : a^21 + b^21 > 0 :=
sorry

theorem inequality_d (hab : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

theorem inequality_e_not_true (hab : a + b > 0) : ¬((a − 3) * (b − 3) < a * b) :=
sorry

theorem inequality_f_not_true (hab : a + b > 0) : ¬((a + 2) * (b + 3) > a * b + 5) :=
sorry

end inequality_a_inequality_b_not_true_inequality_c_inequality_d_inequality_e_not_true_inequality_f_not_true_l720_720489


namespace MN_parallel_PQ_l720_720933

-- Define the given geometric entities and properties
variables (A B C I M N P Q Z : Type) [Inhabited A]

-- Definition of points being collinear
def collinear (a b c : Type) := ∃ (r : ℝ), r • (b - a) + a = c

-- Definition of parallel lines
def parallel (l1 l2 : Type) : Prop := 
  ∃ p1 p2 p3 p4 : Type, collinear p1 p2 p3 → collinear p2 p3 p4

-- Definition points on a circle
def on_circumcircle (A B C : Type) : Prop := sorry

-- Definition of angle bisector
def angle_bisector (I A B : Type) : Prop := sorry

-- Definition of perpendicular lines
def perpendicular (l1 l2 : Type) : Prop := sorry

-- Main theorem statement
theorem MN_parallel_PQ 
  (h1 : on_circumcircle A B C) 
  (h2 : angle_bisector I A B) 
  (h3 : perpendicular PQ IZ) 
  (h4 : perpendicular PQ BI) 
  (h5 : perpendicular MN BI) : parallel MN PQ :=
sorry

end MN_parallel_PQ_l720_720933


namespace distance_between_foci_l720_720035

theorem distance_between_foci :
  let p1 := (1, 5)
  let p2 := (4, -3)
  let p3 := (11, 5)
  ∀ (a b : ℝ), 
    a = 8 ∧ b = 5 →
    2 * real.sqrt ((a ^ 2) - (b ^ 2)) = 2 * real.sqrt 39 :=
by 
  intro p1 p2 p3 a b h
  cases h with ha hb
  rw [ha, hb]
  sorry

end distance_between_foci_l720_720035


namespace area_triangle_GCD_l720_720442

theorem area_triangle_GCD : 
  ∀ (ABCD : Type) [affine_space ABCD]
    (A B C D E F G : ABCD)
    (side : real)
    (H1 : square ABCD)
    (H2 : area ABCD = 180)
    (H3 : E ∈ segment B C)
    (H4 : ratio (segment B E) (segment E C) = 2/1)
    (H5 : midpoint A E = F)
    (H6 : midpoint D E = G)
    (H7 : area (quadrilateral B E G F) = 46),
  area (triangle G C D) = 36.5 := sorry

end area_triangle_GCD_l720_720442


namespace start_A_to_B_l720_720849

theorem start_A_to_B (x : ℝ)
  (A_to_C : x = 1000 * (1000 / 571.43) - 1000)
  (h1 : 1000 / (1000 - 600) = 1000 / (1000 - 428.57))
  (h2 : x = 1750 - 1000) :
  x = 750 :=
by
  rw [h2]
  sorry   -- Proof to be filled in.

end start_A_to_B_l720_720849


namespace nguyen_fabric_needs_l720_720917

def yards_to_feet (yards : ℝ) := yards * 3
def total_fabric_needed (pairs : ℝ) (fabric_per_pair : ℝ) := pairs * fabric_per_pair
def fabric_still_needed (total_needed : ℝ) (already_have : ℝ) := total_needed - already_have

theorem nguyen_fabric_needs :
  let pairs := 7
  let fabric_per_pair := 8.5
  let yards_have := 3.5
  let feet_have := yards_to_feet yards_have
  let total_needed := total_fabric_needed pairs fabric_per_pair
  fabric_still_needed total_needed feet_have = 49 :=
by
  sorry

end nguyen_fabric_needs_l720_720917


namespace ratio_of_work_efficiencies_l720_720961

-- Definition of work efficiencies and their ratios
def A_work_time : ℝ := 4
def B_work_time : ℝ := 5
def A_efficiency := 1 / A_work_time
def B_efficiency := 1 / B_work_time
def efficiency_ratio := A_efficiency / B_efficiency

-- Theorem to prove
theorem ratio_of_work_efficiencies : efficiency_ratio = 5 / 4 :=
by
  sorry

end ratio_of_work_efficiencies_l720_720961


namespace shaded_region_area_l720_720206

noncomputable def area_of_shaded_region (dodecagon_side_length: ℝ) (hexagon_side_length: ℝ): ℝ :=
  let base1 := dodecagon_side_length
  let height1 := dodecagon_side_length
  let area_triangle1 := 1/2 * base1 * height1
  
  let base2 := hexagon_side_length
  let height2 := hexagon_side_length / 2
  let area_triangle2 := 1/2 * base2 * height2

  3 * (area_triangle1 + area_triangle2)

theorem shaded_region_area :
  area_of_shaded_region 12 12 = 324 :=
by
  sorry

end shaded_region_area_l720_720206


namespace polygon_sides_div_by_4_l720_720157

-- Define lattice points
structure LatticePoint := 
  (x : ℤ) 
  (y : ℤ)

-- Define a vector with odd length
structure OddLengthVector := 
  (start : LatticePoint) 
  (end : LatticePoint) 
  (length_odd : (end.x - start.x)^2 + (end.y - start.y)^2 % 2 = 1)

-- Define a closed polygon with properties given in the problem
structure ClosedPolygon :=
  (vertices : List LatticePoint)
  (edges : List OddLengthVector)
  (closed : vertices.head = vertices.last)
  (vertex_lattice : ∀ v ∈ vertices, v.x ∈ ℤ ∧ v.y ∈ ℤ)
  (edges_odd : ∀ e ∈ edges, e.length_odd)

-- Theorem statement
theorem polygon_sides_div_by_4 (P : ClosedPolygon) : 
      ∃ k, 4 * k = P.edges.length := 
by 
  sorry -- Proof goes here

end polygon_sides_div_by_4_l720_720157


namespace smallest_five_digit_number_divisible_by_first_five_primes_l720_720707

theorem smallest_five_digit_number_divisible_by_first_five_primes : 
  ∃ n, (n >= 10000) ∧ (n < 100000) ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_first_five_primes_l720_720707


namespace parallel_MN_PQ_l720_720938

variables {A B C I Z M N P Q : Point}
variables {BI IZ PQ MN : Line}

-- Conditions
variables (triangle_ABC : is_triangle A B C)
variables (incircle_center_I : is_incircle_center I A B C)
variables (angle_bisector_IZ : is_angle_bisector Z B I triangle_ABC)
variables (perpendicular_PQ_IZ : is_perpendicular PQ IZ)
variables (perpendicular_IZ_BI : is_perpendicular IZ BI)
variables (perpendicular_MN_BI : is_perpendicular MN BI)

theorem parallel_MN_PQ 
  (h1: is_angle_bisector Z B I triangle_ABC)
  (h2: is_perpendicular PQ IZ)
  (h3: is_perpendicular IZ BI)
  (h4: is_perpendicular MN BI) : 
  is_parallel MN PQ := 
sorry

end parallel_MN_PQ_l720_720938


namespace problem1_problem2_l720_720740

-- Define the first problem
theorem problem1 : (Real.cos (25 / 3 * Real.pi) + Real.tan (-15 / 4 * Real.pi)) = 3 / 2 :=
by
  sorry

-- Define vector operations and the problem
variables (a b : ℝ)

theorem problem2 : 2 * (a - b) - (2 * a + b) + 3 * b = 0 :=
by
  sorry

end problem1_problem2_l720_720740


namespace decimal_to_fraction_l720_720056

theorem decimal_to_fraction (x : ℝ) (hx : x = 2.35) : x = 47 / 20 := by
  sorry

end decimal_to_fraction_l720_720056


namespace median_in_right_triangle_l720_720381

theorem median_in_right_triangle
  (A B C D E : Type)
  (AD AE AF : ℝ)
  (angle_A: ℝ)
  (h1: AD = 12)
  (h2 : AE = 13)
  (h3 : angle_A = 90)
  (h4 : ∀ (α : ℝ), α = 45 → cos α = 1/real.sqrt 2) :
  AF = 12 * real.sqrt 2 := 
sorry

end median_in_right_triangle_l720_720381


namespace digit_B_divisible_by_9_l720_720564

-- Defining the condition for B making 762B divisible by 9
theorem digit_B_divisible_by_9 (B : ℕ) : (15 + B) % 9 = 0 ↔ B = 3 := 
by
  sorry

end digit_B_divisible_by_9_l720_720564


namespace trig_identity_l720_720749

theorem trig_identity 
  (α : ℝ) 
  (tan_alpha : Real.tan α = 1 / 2) 
  (alpha_range : π < α ∧ α < 3 * π / 2) :
  Real.cos α - Real.sin α = - sqrt 5 / 5 :=
sorry

end trig_identity_l720_720749


namespace factorize_expression_l720_720677

-- Definition of the variables and the expression
variable (a b : ℝ)
def E : ℝ := -a^2 + 4b^2

-- Statement to prove
theorem factorize_expression : E a b = (2 * b + a) * (2 * b - a) :=
by
  sorry

end factorize_expression_l720_720677


namespace at_least_n_minus_one_ministers_receive_own_decree_l720_720545

theorem at_least_n_minus_one_ministers_receive_own_decree 
  (n : ℕ) (decrees : Fin n → Type) 
  (send_telegram : (i j : Fin n) → (known_decrees : Set (Fin n)) → Prop)
  (all_familiar : ∀ i : Fin n, ∃ known_decrees : Set (Fin n), known_decrees = Set.univ)
  (received_at_most_once : ∀ i j : Fin n, ∀ d : Fin n, send_telegram i j {d} → ¬ send_telegram j i {d})
  (some_received_own_decree : ∀ i : Fin n, ∃ j : Fin n, send_telegram j i (decrees j)) : 
  ∃ S : Finset (Fin n), S.card ≥ n-1 :=
sorry

end at_least_n_minus_one_ministers_receive_own_decree_l720_720545


namespace stratified_sampling_third_grade_l720_720619

def total_students : ℕ := 2000
def first_grade_students : ℕ := 750
def second_grade_boys : ℕ := 280
def probability_of_girl_2nd_grade : ℝ := 0.11
def selected_students : ℕ := 64

noncomputable def number_of_girls_in_2nd_grade : ℕ :=
  (total_students * (probability_of_girl_2nd_grade)).toNat

def total_students_in_1st_and_2nd_grades : ℕ :=
  first_grade_students + second_grade_boys + number_of_girls_in_2nd_grade

def third_grade_students : ℕ :=
  total_students - total_students_in_1st_and_2nd_grades

noncomputable def students_selected_from_third_grade : ℕ :=
  (selected_students * (third_grade_students.toRat / total_students.toRat)).toNat

theorem stratified_sampling_third_grade :
  students_selected_from_third_grade = 24 :=
by sorry

end stratified_sampling_third_grade_l720_720619


namespace conic_section_eq_ellipse_y_axis_l720_720007

theorem conic_section_eq_ellipse_y_axis :
  ∀ (x y : ℝ), 
  (1 = x^2 / (sin (Real.sqrt 2) - sin (Real.sqrt 3)) + y^2 / (cos (Real.sqrt 2) - cos (Real.sqrt 3))) → 
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a = (sin (Real.sqrt 2) - sin (Real.sqrt 3)) ∧ b = (cos (Real.sqrt 2) - cos (Real.sqrt 3)) ∧ 
  ((x^2) / a + (y^2) / b = 1) ∧ (a < b)) :=
sorry

end conic_section_eq_ellipse_y_axis_l720_720007


namespace fourth_vertex_of_square_l720_720822

theorem fourth_vertex_of_square {a b c d : ℂ} (h1 : a = 1 + 2 * complex.i) (h2 : b = -2 + complex.i) (h3 : c = -1 - 2 * complex.i) :
  d = 2 - complex.i :=
sorry

end fourth_vertex_of_square_l720_720822


namespace two_point_three_five_as_fraction_l720_720069

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l720_720069


namespace frank_problems_per_type_l720_720646

-- Definitions based on the problem conditions
def bill_problems : ℕ := 20
def ryan_problems : ℕ := 2 * bill_problems
def frank_problems : ℕ := 3 * ryan_problems
def types_of_problems : ℕ := 4

-- The proof statement equivalent to the math problem
theorem frank_problems_per_type : frank_problems / types_of_problems = 30 :=
by 
  -- skipping the proof steps
  sorry

end frank_problems_per_type_l720_720646


namespace count_of_sets_l720_720986

-- Define the universe in which we will work
universe u

-- Define the elements and sets
def S : Finset ℕ := {1, 2, 3, 4}
def subsetS := {T : Finset ℕ | T ⊆ S}
def M : Finset ℕ

-- Define the conditions
def condition1 := ∀ M : Finset ℕ, {1, 2} ⊆ M → M ⊆ S

-- Define the statement that must be proven
theorem count_of_sets (h : condition1) : Finset.card {M : Finset ℕ | {1, 2} ⊆ M ∧ M ⊆ S} = 4 :=
sorry

end count_of_sets_l720_720986


namespace customers_left_l720_720631

theorem customers_left (initial_customers remaining_tables people_per_table customers_left : ℕ)
    (h_initial : initial_customers = 62)
    (h_tables : remaining_tables = 5)
    (h_people : people_per_table = 9)
    (h_left : customers_left = initial_customers - remaining_tables * people_per_table) : 
    customers_left = 17 := 
    by 
        -- Provide the proof here 
        sorry

end customers_left_l720_720631


namespace car_speed_in_second_hour_l720_720027

theorem car_speed_in_second_hour (x : ℕ) : 84 = (98 + x) / 2 → x = 70 := 
sorry

end car_speed_in_second_hour_l720_720027


namespace data_transformation_equiv_l720_720784

variables {n : ℕ}
variables {x y : Fin n → ℝ}
variables {a1 b1 c1 d1 a2 b2 c2 d2 : ℝ}

-- Given condition for transformation
def transformation (y x : Fin n → ℝ) := ∀ i, y i = 3 * x i - 1

-- Definitions for mean, variance, and percentiles (placeholders for actual definitions)
noncomputable def mean (data : Fin n → ℝ) := (1 : ℝ) / n * (Finset.univ.sum (λ i, data i))
noncomputable def variance (data : Fin n → ℝ) := (1 : ℝ) / n * (Finset.univ.sum (λ i, (data i - mean data)^2))
noncomputable def percentile (data : Fin n → ℝ) (p : ℝ) := sorry -- placeholder for percentile definition
noncomputable def mode (data : Fin n → ℝ) := sorry -- placeholder for mode definition

-- Lean theorem for equivalence proof
theorem data_transformation_equiv (h_trans : transformation y x) 
  (mean_x : mean x = b1) (var_x : variance x = c1) (percentile_x : percentile x 0.8 = d1)
  (mean_y : mean y = b2) (var_y : variance y = c2) (percentile_y : percentile y 0.8 = d2) :
  b2 = 3 * b1 - 1 ∧ c2 = 9 * c1 ∧ d2 = 3 * d1 - 1 :=
begin
  sorry
end

end data_transformation_equiv_l720_720784


namespace parallel_MN_PQ_l720_720941

variables {A B C I Z M N P Q : Point}
variables {BI IZ PQ MN : Line}

-- Conditions
variables (triangle_ABC : is_triangle A B C)
variables (incircle_center_I : is_incircle_center I A B C)
variables (angle_bisector_IZ : is_angle_bisector Z B I triangle_ABC)
variables (perpendicular_PQ_IZ : is_perpendicular PQ IZ)
variables (perpendicular_IZ_BI : is_perpendicular IZ BI)
variables (perpendicular_MN_BI : is_perpendicular MN BI)

theorem parallel_MN_PQ 
  (h1: is_angle_bisector Z B I triangle_ABC)
  (h2: is_perpendicular PQ IZ)
  (h3: is_perpendicular IZ BI)
  (h4: is_perpendicular MN BI) : 
  is_parallel MN PQ := 
sorry

end parallel_MN_PQ_l720_720941


namespace common_ratio_of_geometric_sequence_l720_720298

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)
  (h3 : a 1 + 4 * d = (a 0 + 16 * d) * (a 0 + 4 * d) / a 0 ) :
  (a 1 + 4 * d) / a 0 = 3 :=
by
  sorry

end common_ratio_of_geometric_sequence_l720_720298


namespace inequality_a_inequality_c_inequality_d_l720_720478

variable (a b : ℝ)

theorem inequality_a (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 := 
Sorry

theorem inequality_c (h : a + b > 0) : a^21 + b^21 > 0 := 
Sorry

theorem inequality_d (h : a + b > 0) : (a + 2) * (b + 2) > a * b := 
Sorry

end inequality_a_inequality_c_inequality_d_l720_720478


namespace geometry_problem_l720_720143

-- Define the setting
variables {A B C P E F B1 C1 : Type} [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint P] [IsPoint E] [IsPoint F] [IsPoint B1] [IsPoint C1]

def circumcircle (A B C : Type) : Type := sorry  -- placeholder for circumcircle definition

noncomputable def inradius (A B C : Type) : Real := sorry -- placeholder for inradius definition
noncomputable def circumradius (A B C : Type) : Real := sorry -- placeholder for circumradius definition

-- State the problem
theorem geometry_problem (P : A → B → C → Prop)
                         (omega : circumcircle A B C)
                         (BP_inter_omega : BP ∧ B1 ∈ ω)
                         (CP_inter_omega : CP ∧ C1 ∈ ω)
                         (PE_perp_AC : PE ⊥ AC)
                         (PF_perp_AB : PF ⊥ AB)
                         (r : Real)
                         (R : Real)
                         (EF B1C1 : Real) :
                         (inradius A B C = r) →
                         (circumradius A B C = R) →
                         (\frac{EF}{B1C1} ≥ \frac{r}{R}) :=
begin
  sorry
end

end geometry_problem_l720_720143


namespace squares_sum_l720_720595

theorem squares_sum (a b c : ℝ) 
  (h1 : 36 - 4 * Real.sqrt 2 - 6 * Real.sqrt 3 + 12 * Real.sqrt 6 = (a * Real.sqrt 2 + b * Real.sqrt 3 + c) ^ 2) : 
  a^2 + b^2 + c^2 = 14 := 
by
  sorry

end squares_sum_l720_720595


namespace find_cat_video_length_l720_720648

variables (C : ℕ)

def cat_video_length (C : ℕ) : Prop :=
  C + 2 * C + 6 * C = 36

theorem find_cat_video_length : cat_video_length 4 :=
by
  sorry

end find_cat_video_length_l720_720648


namespace cannot_form_triangle_l720_720195

theorem cannot_form_triangle (a b c : ℝ) (h1 : a = 7) (h2 : b = 7) (h3 : c = 15) : a + b ≤ c :=
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end cannot_form_triangle_l720_720195


namespace line_passes_through_fixed_point_l720_720611

theorem line_passes_through_fixed_point 
  (m : ℝ) : ∃ x y : ℝ, y = m * x + (2 * m + 1) ∧ (x, y) = (-2, 1) :=
by
  use (-2), (1)
  sorry

end line_passes_through_fixed_point_l720_720611


namespace value_of_y_minus_x_l720_720345

theorem value_of_y_minus_x (x y : ℝ) (h1 : abs (x + 1) = 3) (h2 : abs y = 5) (h3 : -y / x > 0) :
  y - x = -7 ∨ y - x = 9 :=
sorry

end value_of_y_minus_x_l720_720345


namespace price_of_adult_ticket_l720_720188

/--
Given:
1. The price of a child's ticket is half the price of an adult's ticket.
2. Janet buys tickets for 10 people, 4 of whom are children.
3. Janet buys a soda for $5.
4. With the soda, Janet gets a 20% discount on the total admission price.
5. Janet paid $197 in total for everything.

Prove that the price of an adult admission ticket is $30.
-/
theorem price_of_adult_ticket : 
  ∃ (A : ℝ), 
  (∀ (childPrice adultPrice total : ℝ),
    adultPrice = A →
    childPrice = A / 2 →
    total = adultPrice * 6 + childPrice * 4 →
    totalPriceWithDiscount = 192 →
    total / 0.8 = total + 5 →
    A = 30) :=
sorry

end price_of_adult_ticket_l720_720188


namespace monotonic_decreasing_interval_l720_720462
open Real

def f (x : ℝ) := (1 / 2) * x^2 - log x

theorem monotonic_decreasing_interval : (∀ x, x > 0 → (1/2) * x^2 - log x = f x) →
                                      (∀ x, x > 0 → deriv f x = x - 1 / x) →
                                      (∀ x, x > 0 → x - 1 / x ≤ 0 ↔ 0 < x ∧ x ≤ 1) →
                                      (∀ x, 0 < x ∧ x < 1 → deriv f x ≤ 0) :=
by { intros h1 h2 h3, sorry }

end monotonic_decreasing_interval_l720_720462


namespace MN_parallel_PQ_l720_720932

-- Define the given geometric entities and properties
variables (A B C I M N P Q Z : Type) [Inhabited A]

-- Definition of points being collinear
def collinear (a b c : Type) := ∃ (r : ℝ), r • (b - a) + a = c

-- Definition of parallel lines
def parallel (l1 l2 : Type) : Prop := 
  ∃ p1 p2 p3 p4 : Type, collinear p1 p2 p3 → collinear p2 p3 p4

-- Definition points on a circle
def on_circumcircle (A B C : Type) : Prop := sorry

-- Definition of angle bisector
def angle_bisector (I A B : Type) : Prop := sorry

-- Definition of perpendicular lines
def perpendicular (l1 l2 : Type) : Prop := sorry

-- Main theorem statement
theorem MN_parallel_PQ 
  (h1 : on_circumcircle A B C) 
  (h2 : angle_bisector I A B) 
  (h3 : perpendicular PQ IZ) 
  (h4 : perpendicular PQ BI) 
  (h5 : perpendicular MN BI) : parallel MN PQ :=
sorry

end MN_parallel_PQ_l720_720932


namespace median_perpendicular_l720_720915
-- Import all necessary math libraries

-- Define the setup of the problem
variables {A B C D B1 C1 D1 A2 B2 D2 : Type*}
  [inner_product_space ℝ A]
  [inner_product_space ℝ B]
  [inner_product_space ℝ C]
  [inner_product_space ℝ D]
  [inner_product_space ℝ B1]
  [inner_product_space ℝ C1]
  [inner_product_space ℝ D1]
  [inner_product_space ℝ A2]
  [inner_product_space ℝ B2]
  [inner_product_space ℝ D2]

-- Define points and vectors
variables (ABCD : square A B C D)
          (AB1C1D1 : square A B1 C1 D1)
          (A2B2CD2 : square A2 B2 C D)
          
-- Noncomputable rotation by 90 degrees
noncomputable def rotate_90_deg {A B C : Type*} [inner_product_space ℝ A] 
  (v : A) : B :=
-- Define a rotation function (assuming it is properly defined elsewhere)
sorry

def vector_perpendicular (v1 v2 : A) : Prop :=
  inner_product_space.inner v1 v2 = 0

-- Theorem to prove
theorem median_perpendicular (h1 : equally_oriented_ABCD_ABC1D1 ABCD AB1C1D1)
  (h2 : equally_oriented_AB2CD2_ABC1D1 ABCD A2B2CD2) :
  vector_perpendicular (median BB1B2) (segment D1D2) :=
sorry

end median_perpendicular_l720_720915


namespace metallic_sheet_dimension_l720_720614

theorem metallic_sheet_dimension
  (length_cut : ℕ) (other_dim : ℕ) (volume : ℕ) (x : ℕ)
  (length_cut_eq : length_cut = 8)
  (other_dim_eq : other_dim = 36)
  (volume_eq : volume = 4800)
  (volume_formula : volume = (x - 2 * length_cut) * (other_dim - 2 * length_cut) * length_cut) :
  x = 46 :=
by
  sorry

end metallic_sheet_dimension_l720_720614


namespace decimal_to_fraction_l720_720114

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l720_720114


namespace decimal_to_fraction_l720_720109

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l720_720109


namespace decimal_to_fraction_equivalence_l720_720054

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l720_720054


namespace standard_deviation_of_sample_l720_720295

noncomputable def sample (a : ℝ) : List ℝ := [1, 3, 4, a, 7]

def mean (s : List ℝ) : ℝ :=
  s.sum / s.length

def stddev (s : List ℝ) : ℝ :=
  Real.sqrt ((s.map (λ x, (x - mean s) ^ 2)).sum / s.length)

theorem standard_deviation_of_sample (a : ℝ) (h : mean (sample a) = 4) : stddev (sample a) = 2 := 
  sorry

end standard_deviation_of_sample_l720_720295


namespace find_age_of_first_replaced_man_l720_720451

variables (a x : ℝ) -- a is the average age before replacement, x is the age of the first replaced man
variable (avg_new : ℝ) -- avg_new is the average age of the two new men; given as 34

-- Conditions
axiom avg_new_val : avg_new = 34
axiom increased_avg : a + 2 = a + (24 / 12)
axiom age_diff : 12 * (a + 2) - 12 * a = 2 * avg_new - (x + 23)

theorem find_age_of_first_replaced_man (a x : ℝ) (avg_new : ℝ) [avg_new_val : avg_new = 34] :
  (x = 21) :=
by
  have h : 24 = 68 - x - 23,
  { simp_rw [increased_avg, age_diff],
    ring, },
  linarith

end find_age_of_first_replaced_man_l720_720451


namespace swimming_area_probability_l720_720162

theorem swimming_area_probability :
  let radius_lake := 5
  let radius_swimming_area := 3
  let area_lake := N.pi * radius_lake^2
  let area_swimming_area := N.pi * radius_swimming_area^2
  (area_swimming_area / area_lake) = 9 / 25 := by
sorry

end swimming_area_probability_l720_720162


namespace single_elimination_matches_l720_720367

theorem single_elimination_matches (n : ℕ) (hn : n = 512) : 
  let matches := n - 1
  in matches = 511 :=
by {
  sorry
}

end single_elimination_matches_l720_720367


namespace area_triangle_PQR_zero_l720_720656

open Real

noncomputable def P := (-5 : ℝ, 0 : ℝ)
noncomputable def Q := (0 : ℝ, 0 : ℝ)
noncomputable def R := (6 : ℝ, 0 : ℝ)

theorem area_triangle_PQR_zero : 
  let P := (-5, 0)
  let Q := (0, 0)
  let R := (6, 0)
  let area (P Q R : ℝ × ℝ) := 
    0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))
  area P Q R = 0 :=
by simp [area, P, Q, R]
; sorry

end area_triangle_PQR_zero_l720_720656


namespace ratio_women_to_passengers_is_two_to_three_l720_720033

noncomputable def ratio_women_to_total (total_passengers : ℕ) (seated_men : ℕ) (fraction_standing : ℚ) : ℚ :=
  let total_men := (seated_men * (1 + (1 / fraction_standing))) in
  let total_women := total_passengers - total_men in
  total_women / total_passengers

theorem ratio_women_to_passengers_is_two_to_three :
  ratio_women_to_total 48 14 (1 / 8) = 2 / 3 :=
by sorry

end ratio_women_to_passengers_is_two_to_three_l720_720033


namespace decimal_to_fraction_l720_720106

theorem decimal_to_fraction (d : ℚ) (h : d = 2.35) : d = 47 / 20 := sorry

end decimal_to_fraction_l720_720106


namespace cylinder_radius_and_volume_l720_720460

theorem cylinder_radius_and_volume (h : ℝ) (S : ℝ) (r : ℝ) (V : ℝ) :
    h = 8 ∧ S = 130 * real.pi →
    2 * real.pi * r^2 + 2 * real.pi * r * 8 = 130 * real.pi →
    r = 5 ∧ V = real.pi * r^2 * h :=
by
  intros
  sorry

end cylinder_radius_and_volume_l720_720460


namespace max_lambda_eq_4_over_3_l720_720761

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(λ (a : ℕ → ℝ) (q : ℝ) n, if q = 1 then a 1 * n else a 1 * (q^n - 1) / (q - 1)) a (2 : ℝ) n

theorem max_lambda_eq_4_over_3 :
  ∃ (λ : ℝ), 
    (∀ (a : ℕ → ℝ) (q > 1),
      geometric_sequence a q →
      a 2 = 6 →
      a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 900 →
      (∀ (n : ℕ), λ * a n ≤ 1 + sum_first_n_terms a n)) ∧ λ = 4 / 3 := 
sorry

end max_lambda_eq_4_over_3_l720_720761


namespace two_point_three_five_as_fraction_l720_720064

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l720_720064


namespace inverse_proportion_in_quadrants_l720_720312

theorem inverse_proportion_in_quadrants (f : ℝ → ℝ) (hp : f (-2) = 1) :
  (∀ x : ℝ, f x * x = k) → (k < 0) → 
  ∀ y : ℝ, ((y > 0 ∧ f y < 0) ∨ (y < 0 ∧ f y > 0)) :=
by
  assume h1 : ∀ x : ℝ, f x * x = k
  assume h2 : k < 0
  sorry

end inverse_proportion_in_quadrants_l720_720312


namespace brick_height_l720_720153

/--

Prove that given the conditions:
1. The wall dimensions are 800 cm x 600 cm x 22.5 cm.
2. The brick dimensions are 80 cm x 11.25 cm x height (unknown).
3. 2000 bricks are needed to build the wall.

Then the height of each brick is 6 cm.

-/

theorem brick_height (volume_wall : ℤ) (num_bricks : ℤ)
  (length_wall width_wall height_wall : ℤ)
  (length_brick width_brick height_brick : ℤ) :
  length_wall = 800 → width_wall = 600 → height_wall = 22.5 →
  length_brick = 80 → width_brick = 11.25 →
  num_bricks = 2000 →
  volume_wall = length_wall * width_wall * height_wall →
  height_brick = volume_wall / (num_bricks * length_brick * width_brick) →
  height_brick = 6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end brick_height_l720_720153


namespace scenarios_one_route_not_visited_l720_720746

open Nat

theorem scenarios_one_route_not_visited :
  let families := 4
  let routes := 4
  ∃ (n : ℕ), n = choose families 2 * permutations (routes - 1) (routes - 1) ∧ n = 144 :=
by
  sorry

end scenarios_one_route_not_visited_l720_720746


namespace find_g_inverse_84_l720_720835

-- Definition of the function g
def g (x : ℝ) : ℝ := 3 * x^3 + 3

-- Definition stating the goal
theorem find_g_inverse_84 : g⁻¹ 84 = 3 :=
sorry

end find_g_inverse_84_l720_720835


namespace inequality_a_inequality_c_inequality_d_l720_720476

variable (a b : ℝ)

theorem inequality_a (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 := 
Sorry

theorem inequality_c (h : a + b > 0) : a^21 + b^21 > 0 := 
Sorry

theorem inequality_d (h : a + b > 0) : (a + 2) * (b + 2) > a * b := 
Sorry

end inequality_a_inequality_c_inequality_d_l720_720476


namespace inequality_a_inequality_c_inequality_d_l720_720481

variable (a b : ℝ)

theorem inequality_a (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 := 
Sorry

theorem inequality_c (h : a + b > 0) : a^21 + b^21 > 0 := 
Sorry

theorem inequality_d (h : a + b > 0) : (a + 2) * (b + 2) > a * b := 
Sorry

end inequality_a_inequality_c_inequality_d_l720_720481


namespace graph_symmetry_with_respect_to_point_l720_720319

noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(4 - x)

theorem graph_symmetry_with_respect_to_point : 
  ∀ x : ℝ, f(x) + f(4 - x) = 0 := 
by
  intro x
  sorry

end graph_symmetry_with_respect_to_point_l720_720319


namespace inequality_a_inequality_c_inequality_d_l720_720480

variable (a b : ℝ)

theorem inequality_a (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 := 
Sorry

theorem inequality_c (h : a + b > 0) : a^21 + b^21 > 0 := 
Sorry

theorem inequality_d (h : a + b > 0) : (a + 2) * (b + 2) > a * b := 
Sorry

end inequality_a_inequality_c_inequality_d_l720_720480


namespace decimal_to_fraction_equivalence_l720_720046

theorem decimal_to_fraction_equivalence :
  (∃ a b : ℤ, b ≠ 0 ∧ 2.35 = (a / b) ∧ a.gcd b = 5 ∧ a / b = 47 / 20) :=
sorry

# Check the result without proof
# eval 2.35 = 47/20

end decimal_to_fraction_equivalence_l720_720046


namespace decimal_to_fraction_l720_720111

theorem decimal_to_fraction (d : ℝ) (h : d = 2.35) : d = 47 / 20 :=
by {
  rw h,
  sorry
}

end decimal_to_fraction_l720_720111


namespace circumradius_inequality_l720_720763

section
variables {α : Type*} [EuclideanSpace α]

-- Define a triangle and its incenter
variables {A B C O : α}

-- Assume O is the incenter of triangle ABC
variable (hO : isIncenter O A B C)

-- Definitions for circumradius of triangles ABC and BOC
noncomputable def R_ABC := circumradius A B C
noncomputable def R_BOC := circumradius B O C

-- Statement to prove
theorem circumradius_inequality (hO : isIncenter O A B C) : R_BOC < 2 * R_ABC :=
sorry

end

end circumradius_inequality_l720_720763


namespace log_base_eq_l720_720261

theorem log_base_eq (x : ℝ) : (logBase x 16 = (1 / 3)) → (logBase 64 4 = (1 / 3)) → x = 4096 :=
by
  sorry

end log_base_eq_l720_720261


namespace hyperbola_eccentricity_l720_720805

variables {a b : ℝ}
variables (A C B O : (ℝ × ℝ))
def line_eq (b : ℝ) : ℝ × ℝ → Prop := λ p, p.2 = 2 * b
def hyperbola_eq (a b : ℝ) : ℝ × ℝ → Prop := λ p, (p.1 ^ 2) / (a ^ 2) - (p.2 ^ 2) / (b ^ 2) = 1

theorem hyperbola_eccentricity 
  (hA : A = (a, 0))
  (hO : O = (0, 0))
  (hC : C = (2 * real.sqrt(3) / 3 * b, 2 * b))
  (hAngle : ∠(A, O, C) = π / 3)  -- 60 degrees in radians
  (hIntersect : ∃ (B C : (ℝ × ℝ)), line_eq b B ∧ line_eq b C ∧ hyperbola_eq a b B ∧ hyperbola_eq a b C) :
  let c := real.sqrt (a^2 + b^2) in
  let e := c / a in
  e = real.sqrt(19) / 2 :=
sorry

end hyperbola_eccentricity_l720_720805


namespace efficiency_ratio_l720_720130

variable {A B : ℝ}

theorem efficiency_ratio (hA : A = 1 / 30) (hAB : A + B = 1 / 20) : A / B = 2 :=
by
  sorry

end efficiency_ratio_l720_720130


namespace decimal_to_fraction_l720_720092

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l720_720092


namespace count_integers_between_l720_720464

theorem count_integers_between :
  (∃ n : ℕ, ∀ x : ℕ, 
       (x ∈ {-4, -3, -2, -1, 0, 1, 2}) → x > -5 ∧ x < 3) → 
  {n : ℤ | -5 < n ∧ n < 3}.card = 7 :=
begin
  sorry
end

end count_integers_between_l720_720464


namespace weight_of_b_l720_720973

variable (Wa Wb Wc: ℝ)

-- Conditions
def avg_weight_abc : Prop := (Wa + Wb + Wc) / 3 = 45
def avg_weight_ab : Prop := (Wa + Wb) / 2 = 40
def avg_weight_bc : Prop := (Wb + Wc) / 2 = 43

-- Theorem to prove
theorem weight_of_b (Wa Wb Wc: ℝ) (h_avg_abc : avg_weight_abc Wa Wb Wc)
  (h_avg_ab : avg_weight_ab Wa Wb) (h_avg_bc : avg_weight_bc Wb Wc) : Wb = 31 :=
by
  sorry

end weight_of_b_l720_720973


namespace red_ball_probabilities_l720_720385

universe u

noncomputable def count_combinations : (α : Type u) → List α → Finset (Multiset α) := sorry

def probability_two_red (total_balls : List (String × Nat)) : ℚ :=
  let all_outcomes := count_combinations _ (total_balls.bind (fun x => List.replicate x.snd x.fst))
  let favorable_outcomes := all_outcomes.filter (fun m => m.count "red" = 2)
  (favorable_outcomes.card : ℚ) / all_outcomes.card

def probability_at_least_one_red (total_balls : List (String × Nat)) : ℚ :=
  let all_outcomes := count_combinations _ (total_balls.bind (fun x => List.replicate x.snd x.fst))
  let favorable_outcomes := all_outcomes.filter (fun m => m.count "red" > 0)
  (favorable_outcomes.card : ℚ) / all_outcomes.card

theorem red_ball_probabilities :
  let total_balls := [("red", 3), ("white", 2)]
  probability_two_red total_balls = 3 / 10 ∧ probability_at_least_one_red total_balls = 9 / 10 :=
by
  sorry

end red_ball_probabilities_l720_720385


namespace cos_sin_quadrant_third_l720_720789

theorem cos_sin_quadrant_third (θ : ℝ) (hθ : θ = (4/3) * Real.pi) :
  let z := Complex.mk (cos θ) (sin θ) in
  (-1/2) > 0 ∧ (-Real.sqrt 3 / 2) > 0 → (Complex.re z < 0) ∧ (Complex.im z < 0) := 
by
  sorry

end cos_sin_quadrant_third_l720_720789


namespace orchestra_admission_l720_720022

theorem orchestra_admission (x v c t: ℝ) 
  -- Conditions
  (h1 : v = 1.25 * 1.6 * x)
  (h2 : c = 0.8 * x)
  (h3 : t = 0.4 * x)
  (h4 : v + c + t = 32) :
  -- Conclusion
  v = 20 ∧ c = 8 ∧ t = 4 :=
sorry

end orchestra_admission_l720_720022


namespace number_of_possible_values_for_c_l720_720446

def isPossibileValueForC (c : ℕ) : Prop :=
  (c^2 ≤ 256) ∧ (256 < c^3)

theorem number_of_possible_values_for_c : {c : ℕ | c ≥ 2 ∧ isPossibileValueForC c}.card = 10 :=
by
  sorry

end number_of_possible_values_for_c_l720_720446


namespace decimal_to_fraction_l720_720095

theorem decimal_to_fraction (h : 2.35 = (47/20 : ℚ)) : 2.35 = 47/20 :=
by sorry

end decimal_to_fraction_l720_720095


namespace problem1_problem2_problem3_problem4_problem5_problem6_l720_720520

section
variables {a b : ℝ}

-- Problem 1
theorem problem1 (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 :=
sorry

-- Problem 2
theorem problem2 (h : a + b > 0) : ¬ (a^4 * b^3 + a^3 * b^4 ≥ 0) :=
sorry

-- Problem 3
theorem problem3 (h : a + b > 0) : a^21 + b^21 > 0 :=
sorry

-- Problem 4
theorem problem4 (h : a + b > 0) : (a + 2) * (b + 2) > a * b :=
sorry

-- Problem 5
theorem problem5 (h : a + b > 0) : ¬ (a - 3) * (b - 3) < a * b :=
sorry

-- Problem 6
theorem problem6 (h : a + b > 0) : ¬ (a + 2) * (b + 3) > a * b + 5 :=
sorry

end

end problem1_problem2_problem3_problem4_problem5_problem6_l720_720520


namespace negation_of_existential_proposition_l720_720463

theorem negation_of_existential_proposition :
  ¬ (∃ x : ℝ, 2^x < 1) ↔ ∀ x : ℝ, 2^x ≥ 1 :=
by
  sorry

end negation_of_existential_proposition_l720_720463


namespace flowmaster_pump_output_l720_720970

theorem flowmaster_pump_output (hourly_rate : ℕ) (time_minutes : ℕ) (output_gallons : ℕ) 
  (h1 : hourly_rate = 600) 
  (h2 : time_minutes = 30) 
  (h3 : output_gallons = (hourly_rate * time_minutes) / 60) : 
  output_gallons = 300 :=
by sorry

end flowmaster_pump_output_l720_720970


namespace inequality_a_inequality_c_inequality_d_l720_720477

variable (a b : ℝ)

theorem inequality_a (h : a + b > 0) : a^5 * b^2 + a^4 * b^3 ≥ 0 := 
Sorry

theorem inequality_c (h : a + b > 0) : a^21 + b^21 > 0 := 
Sorry

theorem inequality_d (h : a + b > 0) : (a + 2) * (b + 2) > a * b := 
Sorry

end inequality_a_inequality_c_inequality_d_l720_720477


namespace inscribed_circle_radius_l720_720571

noncomputable def radius_of_inscribed_circle (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem inscribed_circle_radius :
  radius_of_inscribed_circle 8 8 5 = 38 / 21 :=
by
  sorry

end inscribed_circle_radius_l720_720571


namespace print_time_correct_l720_720170

-- Define the conditions
def pages_per_minute : ℕ := 23
def total_pages : ℕ := 345

-- Define the expected result
def expected_minutes : ℕ := 15

-- Prove the equivalence
theorem print_time_correct :
  total_pages / pages_per_minute = expected_minutes :=
by 
  -- Proof will be provided here
  sorry

end print_time_correct_l720_720170


namespace smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l720_720697

theorem smallest_five_digit_number_divisible_by_prime_2_3_5_7_11 : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0 ∧ n = 11550 :=
by
  sorry

end smallest_five_digit_number_divisible_by_prime_2_3_5_7_11_l720_720697


namespace max_balls_49_adjacent_product_lt_100_l720_720286

def max_balls_with_adjacent_product_constraint (n : ℕ) (s : Set ℕ) : ℕ :=
  if ∀ (a b ∈ s), (a * b < n) then s.card else 0

theorem max_balls_49_adjacent_product_lt_100 :
  max_balls_with_adjacent_product_constraint 100 {i | i ∈ (Finset.range 49).erase 0} = 18 := 
sorry

end max_balls_49_adjacent_product_lt_100_l720_720286


namespace inequalities_count_l720_720299

def expr1 : Prop := 3 < 5
def expr2 (x : ℝ) : Prop := 4 * x + 5 > 0
def expr3 (x : ℝ) : Prop := x = 3
def expr4 (x : ℝ) : ℝ := x^2 + x 
def expr5 (x : ℝ) : Prop := x ≠ 4
def expr6 (x : ℝ) : Prop := x + 2 ≥ x + 1

theorem inequalities_count : 
  (expr1 ∧ expr2 0 ∧ ¬(expr3 3) ∧ ¬(true) ∧ expr5 0 ∧ expr6 0) →
  4 :=
sorry

end inequalities_count_l720_720299


namespace optimal_strategy_outcome_l720_720189

def chessboard : Type := (Σ (x : Fin 8), Fin 8) -- A chessboard with coordinates (x, y)

def knight_moves (c : chessboard) : List chessboard :=
  let (x, y) := c
  [(x + 1, y + 2), (x + 1, y - 2), (x - 1, y + 2), (x - 1, y - 2),
   (x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1)].filter
    (λ ⟨x, y⟩, x.val < 8 ∧ y.val < 8) -- valid knight moves on an 8x8 chessboard

/--
Player A and Player B take turns applying glue to one square of a chessboard, one square per turn starting from (0, 0).
The knight must always be able to jump to any unglued squares following standard chess rules without getting stuck.
If a player cannot make a valid move, that player loses.
-/
theorem optimal_strategy_outcome : ∀ (knight : chessboard) (turns : Fin 64 → chessboard) (glued_turns : Fin 64 → Bool),
  knight = ⟨0, 0⟩ → -- Initial position of the knight at a1 (0, 0)
  (∀ t, glued_turns t = false) → -- Initial board state, no squares are glued
  (∀ t, (knight_moves (turns t)).all (λ s, glued_turns s = false)) → -- Knight can always move to unglued squares
  (∃ (a_turn_strategy : Fin 64 → chessboard),
    (∃ (b_turn_strategy : Fin 64 → chessboard),
      (a_turn_strategy 0 = turns 0 ∧ b_turn_strategy 1 = turns 1) ∧ (player_wins a_turn_strategy b_turn_strategy))) := sorry

end optimal_strategy_outcome_l720_720189


namespace inscribed_circle_radius_l720_720572

noncomputable def radius_of_inscribed_circle (AB AC BC : ℝ) : ℝ :=
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  K / s

theorem inscribed_circle_radius :
  radius_of_inscribed_circle 8 8 5 = 38 / 21 :=
by
  sorry

end inscribed_circle_radius_l720_720572


namespace bugs_meet_at_point_P_in_6_minutes_l720_720554

-- Define the parameters given in the problem
def radius_large : ℕ := 6
def radius_small : ℕ := 3

def speed_large : ℕ := 4 * Nat.pi
def speed_small : ℕ := 3 * Nat.pi

def circumference (r : ℕ) : ℕ := 2 * Nat.pi * r

def time_to_complete (circumference speed : ℕ) : ℕ := circumference / speed

-- Calculate the circumferences
def circumference_large : ℕ := circumference radius_large
def circumference_small : ℕ := circumference radius_small

-- Calculate the times to complete one round
def time_large : ℕ := time_to_complete circumference_large speed_large
def time_small : ℕ := time_to_complete circumference_small speed_small

-- Least common multiple function for the natural numbers
def lcm (a b : ℕ) : ℕ := a * b / Nat.gcd a b

-- Theorem statement: The time at which both bugs meet again at point P is 6 minutes
theorem bugs_meet_at_point_P_in_6_minutes : lcm time_large time_small = 6 := 
  by
    sorry -- Proof is omitted

end bugs_meet_at_point_P_in_6_minutes_l720_720554


namespace decimal_to_fraction_l720_720081

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l720_720081


namespace find_circle_equation_l720_720147

noncomputable def point_A : ℝ × ℝ := (2, -1)

noncomputable def line_tangent (x y : ℝ) : Prop := x - y = 1 

noncomputable def center_line (x y : ℝ) : Prop := y = -2 * x 

noncomputable def is_circle_equation (x0 y0 r : ℝ) (x y : ℝ) : Prop := 
  (x - x0)^2 + (y - y0)^2 = r  

noncomputable def tangent_condition (x0 y0 : ℝ) (r : ℝ) : Prop :=
  (abs (x0 - y0 - 1) / sqrt 2) = sqrt r 

theorem find_circle_equation :
  ∃ (x0 y0 r : ℝ),
    is_circle_equation x0 y0 r 2 (-1) ∧
    tangent_condition x0 y0 r ∧
    center_line x0 y0 ∧
    ((is_circle_equation x0 y0 r = (λ x y, (x - 1)^2 + (y + 2)^2 = 2))
     ∨ (is_circle_equation x0 y0 r = (λ x y, (x - 9)^2 + (y + 18)^2 = 338))) :=
sorry

end find_circle_equation_l720_720147


namespace petya_time_comparison_l720_720227

open Real

noncomputable def petya_planned_time (D V : ℝ) := D / V

noncomputable def petya_actual_time (D V : ℝ) :=
  let V1 := 1.25 * V
  let V2 := 0.80 * V
  let T1 := (D / 2) / V1
  let T2 := (D / 2) / V2
  T1 + T2

theorem petya_time_comparison (D V : ℝ) (hV : V > 0) : 
  petya_actual_time D V > petya_planned_time D V :=
by {
  let T := petya_planned_time D V
  let T_actual := petya_actual_time D V
  have h1 : petya_planned_time D V = D / V, by unfold petya_planned_time
  have h2 : petya_actual_time D V = (D * 41) / (40 * V), by {
      unfold petya_actual_time,
      have h3 : 1.25 * V = 5 * V / 4, by linarith,
      have h4 : 0.80 * V = 4 * V / 5, by linarith,
      rw [h3, h4],
      simp,
      linarith,
  },
  rw h1,
  rw h2,
  have h3 : (41 * D) / (40 * V) > D / V, by linarith,
  exact h3,
}

end petya_time_comparison_l720_720227


namespace cannot_form_triangle_l720_720196

theorem cannot_form_triangle (a b c : ℝ) (h1 : a = 7) (h2 : b = 7) (h3 : c = 15) : a + b ≤ c :=
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end cannot_form_triangle_l720_720196


namespace proposition_alpha_is_false_proposition_beta_is_true_l720_720764

-- Conditions
variable {a : ℕ → ℕ}  -- a_n sequence
variable {S : ℕ → ℕ}  -- S_n sequence sum
variable {M : Set ℕ}  -- M set of indices

-- Hypotheses from the problem statement
def is_arithmetic_sequence (d : ℕ) : Prop := ∀ n : ℕ, a (n+1) = a n + d

def S_n_sum : Prop := ∀ n : ℕ, S n = (n + 1) * a 0 + n * (n + 1) // 2

def in_set_M (n : ℕ) : Prop :=
  (S n - 2022) * (S (n + 1) - 2022) < 0 ∧ (S n - 2023) * (S (n + 1) - 2023) < 0

theorem proposition_alpha_is_false :
  ¬(∀ M : Set ℕ, ∃ k : ℕ, k = 2 * |M|) :=
begin
  sorry
end

theorem proposition_beta_is_true {d : ℕ} (pos_d : d > 0) {n₀ : ℕ} (h : in_set_M n₀) :
  a (n₀ + 1) > 1 :=
begin
  sorry
end

end proposition_alpha_is_false_proposition_beta_is_true_l720_720764


namespace min_value_of_quadratic_l720_720243

theorem min_value_of_quadratic :
  ∀ (x : ℝ), ∃ (z : ℝ), z = 4 * x^2 + 8 * x + 16 ∧ z ≥ 12 ∧ (∃ c : ℝ, c = c → z = 12) :=
by
  sorry

end min_value_of_quadratic_l720_720243


namespace deck_width_l720_720181

theorem deck_width (w : ℝ) (A : ℝ) (L : ℝ) (W : ℝ) : 
  (A = 728) ∧ (L = 20) ∧ (W = 22) ∧ 
  ((L + 2 * w) * (W + 2 * w) = A) → 
  (w = 3) :=
by
  intros,
  sorry

end deck_width_l720_720181


namespace problem_quadrilateral_inscribed_in_circle_l720_720177

theorem problem_quadrilateral_inscribed_in_circle
  (r : ℝ)
  (AB BC CD DA : ℝ)
  (h_radius : r = 300 * Real.sqrt 2)
  (h_AB : AB = 300)
  (h_BC : BC = 150)
  (h_CD : CD = 150) :
  DA = 750 :=
sorry

end problem_quadrilateral_inscribed_in_circle_l720_720177


namespace decimal_to_fraction_l720_720077

theorem decimal_to_fraction (x : ℝ) (h : x = 2.35) : ∃ (a b : ℤ), (b ≠ 0) ∧ (a / b = x) ∧ (a = 47) ∧ (b = 20) := by
  sorry

end decimal_to_fraction_l720_720077


namespace MN_parallel_PQ_l720_720944

-- Definitions and conditions
variable {A B C I Z M N P Q : Type}
variable [is_center_of_circumcircle I A B C]
variable [midpoint_of_arc I A C]
variable [angle_bisector_of IZ (∠ B)]
variable [perpendicular PQ IZ]
variable [isosceles_triangle M BI]
variable [isosceles_triangle N BI]

-- The theorem statement
theorem MN_parallel_PQ (h_center: is_center_of_circumcircle I A B C) 
  (h_midpoint: midpoint_of_arc I A C) (h_bisector: angle_bisector_of IZ (∠ B)) 
  (h_perpendicular: perpendicular PQ IZ) (h_mbi: isosceles_triangle M BI) 
  (h_nbi: isosceles_triangle N BI) : 
  parallel MN PQ :=
sorry

end MN_parallel_PQ_l720_720944


namespace sequence_bound_l720_720814

noncomputable def a : ℕ → ℝ
| 0       := 1 / 2
| (n + 1) := a n + (1 / (n + 1)^2) * (a n)^2

theorem sequence_bound (n : ℕ) : 
  (n ≠ 0) →
  (↑(n + 1) / ↑(n + 2)) < a n ∧ a n < ↑n := 
by
  sorry

end sequence_bound_l720_720814


namespace solve_quadratic_l720_720966

theorem solve_quadratic : ∀ (x : ℝ), x * (x + 1) = 2014 * 2015 ↔ (x = 2014 ∨ x = -2015) := by
  sorry

end solve_quadratic_l720_720966


namespace find_multiplier_l720_720358

noncomputable def N : ℤ := 2

theorem find_multiplier (x : ℤ) (N : ℤ) (hx : x = 5) (H : N * 10^x < 220000) : N = 2 :=
by
  sorry

end find_multiplier_l720_720358


namespace total_games_single_elimination_l720_720628

-- Define the single elimination tournament structure
def single_elimination_tournament (n : ℕ) : Prop :=
  ∀ (k : ℕ), k < n → ∃! (i j : ℕ), i ≠ j

-- State the problem and the expected solution
theorem total_games_single_elimination {n : ℕ} (h1 : n = 27) (h2 : single_elimination_tournament n) : 
  ∃ g : ℕ, g = 26 :=
by {
  use 26,
  sorry
}

end total_games_single_elimination_l720_720628


namespace log_base_eq_l720_720260

theorem log_base_eq (x : ℝ) : (logBase x 16 = (1 / 3)) → (logBase 64 4 = (1 / 3)) → x = 4096 :=
by
  sorry

end log_base_eq_l720_720260


namespace MN_parallel_PQ_l720_720934

-- Define the given geometric entities and properties
variables (A B C I M N P Q Z : Type) [Inhabited A]

-- Definition of points being collinear
def collinear (a b c : Type) := ∃ (r : ℝ), r • (b - a) + a = c

-- Definition of parallel lines
def parallel (l1 l2 : Type) : Prop := 
  ∃ p1 p2 p3 p4 : Type, collinear p1 p2 p3 → collinear p2 p3 p4

-- Definition points on a circle
def on_circumcircle (A B C : Type) : Prop := sorry

-- Definition of angle bisector
def angle_bisector (I A B : Type) : Prop := sorry

-- Definition of perpendicular lines
def perpendicular (l1 l2 : Type) : Prop := sorry

-- Main theorem statement
theorem MN_parallel_PQ 
  (h1 : on_circumcircle A B C) 
  (h2 : angle_bisector I A B) 
  (h3 : perpendicular PQ IZ) 
  (h4 : perpendicular PQ BI) 
  (h5 : perpendicular MN BI) : parallel MN PQ :=
sorry

end MN_parallel_PQ_l720_720934


namespace kwameStudyTime_l720_720390

-- Kwame studied for some hours
variable {kwameStudyHours : ℕ}  -- The time Kwame studied in hours

-- Conditions in the given problem
def connorStudyMinutes := 1.5 * 60      -- Connor studied for 1.5 hours, converted to minutes
def lexiaStudyMinutes := 97             -- Lexia studied for 97 minutes
def combinedStudyMinutes := lexiaStudyMinutes + 143  -- Kwame and Connor studied 143 minutes more than Lexia

-- The final proof statement
theorem kwameStudyTime :
  kwameStudyHours * 60 = (combinedStudyMinutes - connorStudyMinutes) :=
sorry

end kwameStudyTime_l720_720390


namespace spell_casting_competition_order_count_l720_720285

theorem spell_casting_competition_order_count (participants : Fin 4 → String) :
  (∃! (order : List (Fin 4)), List.perm order [0, 1, 2, 3] ∧ List.nodup order) :=
by {
  let count_possible_orders := Nat.factorial 4,
  have h : count_possible_orders = 24 := by rfl,
  exact ⟨[], λ l _, h.symm⟩ -- This indicates that there is a unique order of length 4, permutations of [0,1,2,3] with no duplicates
}

end spell_casting_competition_order_count_l720_720285


namespace helical_stripe_length_correct_l720_720160

noncomputable def helical_stripe_length (circumference height : ℕ) : Real :=
  Real.sqrt ((circumference:Real)^2 + (height:Real)^2)

theorem helical_stripe_length_correct
  (circumference : ℕ) (height : ℕ) (hc : circumference = 20) (hh : height = 9) :
  helical_stripe_length circumference height = Real.sqrt 481 :=
by
  rw [hc, hh]
  unfold helical_stripe_length
  rw [Nat.cast_pow, Nat.cast_pow, ← Real.sqrt_add, Real.sqrt_eq_rfl]
  simp
  sorry

end helical_stripe_length_correct_l720_720160


namespace carol_rectangle_width_l720_720652

def carol_width (lengthC : ℕ) (widthJ : ℕ) (lengthJ : ℕ) (widthC : ℕ) :=
  lengthC * widthC = lengthJ * widthJ

theorem carol_rectangle_width 
  {lengthC widthJ lengthJ : ℕ} (h1 : lengthC = 8)
  (h2 : widthJ = 30) (h3 : lengthJ = 4)
  (h4 : carol_width lengthC widthJ lengthJ 15) : 
  widthC = 15 :=
by 
  subst h1
  subst h2
  subst h3
  sorry -- proof not required

end carol_rectangle_width_l720_720652
