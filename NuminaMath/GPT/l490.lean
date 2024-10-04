import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Powers
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Quadratic.Discriminant
import Mathlib.Analysis.Calculus.FDeriv
import Mathlib.Analysis.Geometry.Circle
import Mathlib.Analysis.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.PolynomialApplications
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Log
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Polynomial.RingDivision
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Zmod.Basic
import Mathlib.Geometry.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.GroupTheory.Perm.Basic
import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Topology.Basic
import Mathlib.Topology.Instances.Real

namespace sum_of_differences_squared_le_one_l490_490241

theorem sum_of_differences_squared_le_one
  {a1 a2 a3 b1 b2 b3 : ℝ}
  (ha_pos : 0 < a1 ∧ 0 < a2 ∧ 0 < a3)
  (hb_pos : 0 < b1 ∧ 0 < b2 ∧ 0 < b3)
  (ha : ∑ i in (finset.range 3).filter (≤ 3), a i * a j ≤ 1)
  (hb : ∑ i in (finset.range 3).filter (≤ 3), b i * b j ≤ 1) :
  ∑ i in (finset.range 3).filter (≤ 3), (a i - b i) * (a j - b j) ≤ 1 := by
  sorry

end sum_of_differences_squared_le_one_l490_490241


namespace total_gardening_time_l490_490485

-- Define the conditions given in the problem

def time_mowing (lines : ℕ) (time_per_line : ℕ) : ℕ :=
  lines * time_per_line

def time_planting (rows : ℕ) (flowers_per_row : ℕ) (time_per_flower : ℕ) : ℕ :=
  rows * flowers_per_row * (time_per_flower / 2)

def time_watering (sections : ℕ) (time_per_section : ℕ) : ℕ :=
  sections * time_per_section

def time_trimming (hedges : ℕ) (time_per_hedge : ℕ) : ℕ :=
  hedges * time_per_hedge

-- Main theorem to prove
theorem total_gardening_time :
  let mowing_time := time_mowing 40 2,
      planting_time := time_planting 10 8 1,
      watering_time := time_watering 4 3,
      trimming_time := time_trimming 5 6
  in mowing_time + planting_time + watering_time + trimming_time = 162 := 
by
  sorry

end total_gardening_time_l490_490485


namespace twelve_circles_five_neighbors_l490_490126

theorem twelve_circles_five_neighbors :
  ∃ (G : SimpleGraph (Fin 12)), G.IsRegularOfDegree 5 :=
sorry

end twelve_circles_five_neighbors_l490_490126


namespace domain_y_inequality_y_zero_y_l490_490239

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x + 1)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := log a (4 - 2 * x)

theorem domain_y (a : ℝ) (ha : 0 < a) (ha_ne : a ≠ 1) :
  { x : ℝ | -1 < x ∧ x < 2 } = 
    { x : ℝ | 0 < x + 1 ∧ 0 < 4 - 2 * x } :=
by sorry

theorem inequality_y (a : ℝ) (ha : 0 < a) (ha_ne : a ≠ 1) :
  iff (0 < a ∧ a < 1 ∧ (-1 < x ∧ x < 1)) 
      (f a x > g a x) :=
by sorry

theorem zero_y (a : ℝ) (ha : 0 < a) (ha_ne : a ≠ 1) : 
  2 * f a x - g a x - f a 1 = 0 → x = 1 :=
by sorry

end domain_y_inequality_y_zero_y_l490_490239


namespace eval_expression_l490_490514

theorem eval_expression : ∀a b c d : ℝ, 
  (a = 2) ∧ (b = 3) ∧ (c = 5) ∧ (d = 7) →
  (3 * Real.sqrt 8 / (Real.sqrt a + Real.sqrt b + Real.sqrt c + Real.sqrt d) = 
  -6 * (Real.sqrt a - Real.sqrt (a * b) - Real.sqrt (a * c) - Real.sqrt (a * d)) / 13) :=
by {
  intros a b c d,
  intro h,
  cases h with ha h1,
  cases h1 with hb h2,
  cases h2 with hc hd,
  rw [ha, hb, hc, hd],
  sorry -- proof omitted
}

end eval_expression_l490_490514


namespace periodic_f_pi_l490_490312

def f (x : ℝ) : ℝ := |Real.sin (x + (Real.pi / 3))|

theorem periodic_f_pi : ∃ p > 0, ∀ x, f (x + p) = f x ∧ p = Real.pi :=
by
  sorry

end periodic_f_pi_l490_490312


namespace binomial_div_eq_l490_490533

def binomial (b : ℝ) (m : ℕ) : ℝ :=
  b * (b - 1) * (b - 2) * ... * (b - (m - 1)) / (m * (m - 1) * ... * 2 * 1)

theorem binomial_div_eq :
  binomial (-3/2) 50 / binomial (3/2) 50 = -101 / 3 :=
by
  sorry

end binomial_div_eq_l490_490533


namespace concurrency_problem_l490_490655

theorem concurrency_problem
  (A B C : Point)
  (ABC_obtuse : ∠BAC > 90°)
  (O : Circle)
  (H : Circle O A B C)
  (tangent_A : Line)
  (tangent_B : Line)
  (tangent_C : Line)
  (H_tangent_A : tangent_A ⊥ (Circle.tangent O A))
  (H_tangent_B : tangent_B ⊥ (Circle.tangent O B))
  (H_tangent_C : tangent_C ⊥ (Circle.tangent O C))
  (P Q : Point)
  (H_intersect_P : P ∈ (tangent_A ∩ tangent_B))
  (H_intersect_Q : Q ∈ (tangent_A ∩ tangent_C))
  (D E : Point)
  (H_perpendicular_D : D = Foot P BC)
  (H_perpendicular_E : E = Foot Q BC)
  (F G : Point)
  (H_on_PQ_F : F ∈ PQ ∧ F ≠ A)
  (H_on_PQ_G : G ∈ PQ ∧ G ≠ A)
  (H_concyclic_AFBE : Cyclic A F B E)
  (H_concyclic_AGCD : Cyclic A G C D)
  (M : Point)
  (H_midpoint_M : M = Midpoint D E) :
  Concurrent (Line DF) (Line OM) (Line EG) :=
sorry

end concurrency_problem_l490_490655


namespace is_quadratic_f4_l490_490057

noncomputable def f1 (a b c : ℝ) : ℝ → ℝ := λ x, a * x^2 + b * x + c
noncomputable def f2 : ℝ → ℝ := λ x, (2 * x - 1)^2 - 4 * x^2
noncomputable def f3 (a b c : ℝ) (h : a ≠ 0) : ℝ → ℝ := λ x, a / x^2 + b / x + c
noncomputable def f4 : ℝ → ℝ := λ x, (x - 1) * (x - 2)

theorem is_quadratic_f4 : ∀ (x : ℝ), ∃ (a b c : ℝ), f4 x = a * x^2 + b * x + c :=
by
  intro x
  use 1, -3, 2
  sorry

end is_quadratic_f4_l490_490057


namespace sum_of_roots_l490_490858

theorem sum_of_roots (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) :
  ∑ x in {x | a * x^2 + b * x + c = 0}, x = 9 :=
by
  sorry

end sum_of_roots_l490_490858


namespace minimum_grid_points_l490_490103

/-- 
  If a square has an area that is four times the area of a cell,
  then the minimum number of grid points (nodes) that the square can cover 
  is 4.
-/
theorem minimum_grid_points (A_cell : ℝ) : 
  (∃ A_square, A_square = 4 * A_cell) →
  (∃ n, n = 4) :=
begin
  sorry
end

end minimum_grid_points_l490_490103


namespace find_slant_height_l490_490548

-- Definitions of the given conditions
variable (r1 r2 L A1 A2 : ℝ)
variable (π : ℝ := Real.pi)

-- The conditions as given in the problem
def conditions : Prop := 
  r1 = 3 ∧ r2 = 4 ∧ 
  (π * L * (r1 + r2) = A1 + A2) ∧ 
  (A1 = π * r1^2) ∧ 
  (A2 = π * r2^2)

-- The theorem stating the question and the correct answer
theorem find_slant_height (h : conditions r1 r2 L A1 A2) : 
  L = 5 := 
sorry

end find_slant_height_l490_490548


namespace sum_of_solutions_l490_490833

theorem sum_of_solutions (x : ℝ) : 
  (∀ x : ℝ, x^2 = 9*x - 20 → x = 4 ∨ x = 5) → (4 + 5 = 9) :=
by
  intros h
  calc 4 + 5 = 9 : by norm_num
  sorry

end sum_of_solutions_l490_490833


namespace person_A_work_days_l490_490412

theorem person_A_work_days (x : ℝ) (h1 : 0 < x) 
                                 (h2 : ∃ b_work_rate, b_work_rate = 1 / 30) 
                                 (h3 : 5 * (1 / x + 1 / 30) = 0.5) : 
  x = 15 :=
by
-- Proof omitted
sorry

end person_A_work_days_l490_490412


namespace monotonically_increasing_on_interval_l490_490183

theorem monotonically_increasing_on_interval
  (f : ℝ → ℝ) (h_diff : ∀ x, differentiable_at ℝ f x)
  (h_cond : ∀ x, (x^2 - 3 * x + 2) * deriv (deriv f x) < 0) :
  ∀ x ∈ set.Icc 1 2, f 1 ≤ f x ∧ f x ≤ f 2 :=
sorry

end monotonically_increasing_on_interval_l490_490183


namespace sum_of_solutions_eq_9_l490_490809

theorem sum_of_solutions_eq_9 (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20) :
  let (sum_roots : ℝ) := -b / a in 
  sum_roots = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490809


namespace proportional_stratified_sampling_probability_at_least_one_grade12_probability_classes_ab_l490_490636

def school_classes := {grade10 := 16, grade11 := 12, grade12 := 8}

-- Proving proportional stratified sampling
def stratified_sampling (total_classes: Fin 3 -> Nat) (selected: Fin 3 -> Nat) : Prop :=
  selected 0 = (16 * 9 / (16 + 12 + 8)) ∧
  selected 1 = (12 * 9 / (16 + 12 + 8)) ∧
  selected 2 = (8 * 9 / (16 + 12 + 8))

def selected_classes : Fin 3 -> Nat := fun i =>
  match i with
  | ⟨0, _⟩ => 4
  | ⟨1, _⟩ => 3
  | ⟨2, _⟩ => 2

theorem proportional_stratified_sampling :
  stratified_sampling (λ ⟨i, _⟩ => match i with
                              | 0 => school_classes.grade10
                              | 1 => school_classes.grade11
                              | 2 => school_classes.grade12
                      end) selected_classes := sorry

-- Proving probability that at least one of the 2 selected classes is from grade 12
def at_least_one_grade12 : Prop :=
  let grade11 := 3
  let grade12 := 2
  ∃ combs: Π n, Fin₃ (grade11 + grade12), true ∧
  (grade11 + grade12 = 5) ∧
  ∀ selections, ∃ count12, selections.filter (λ x, x < grade12) = count12 ∧
  (count12.toNat ≥ 1) = (7/10)

theorem probability_at_least_one_grade12 :
  at_least_one_grade12 := sorry

-- Proving probability that both class A from grade 11 and class B from grade 12 are selected
def classes_selection(A: Nat) (B: Nat) : Prop :=
  let total_combinations := 4 * 3 * 2
  ∃ prob: (4/total_combinations), true ∧
  prob = (1/6)

theorem probability_classes_ab :
  classes_selection 11 12 := sorry

end proportional_stratified_sampling_probability_at_least_one_grade12_probability_classes_ab_l490_490636


namespace son_l490_490099

variable (S M : ℕ)
variable h1 : M = S + 26
variable h2 : M + 2 = 2 * (S + 2)

theorem son's_age_is_24 : S = 24 :=
by
  sorry

end son_l490_490099


namespace triangle_ratio_l490_490208

theorem triangle_ratio {ABC : Triangle} (h_non_isosceles : ¬ is_isosceles ABC) 
    (h_angle_ABC : ∠ABC = 60) {T : Point} (h_point_T : inside_triangle ABC T)
    (h_angles_T : ∠ATC = 120 ∧ ∠BTC = 120 ∧ ∠BTA = 120)
    {M : Point} (h_M : is_centroid ABC M)
    {K : Point} (h_intersect : intersects (line TM) (circumcircle (triangle ATC)) K) :
    TM / MK = 2 := 
sorry

end triangle_ratio_l490_490208


namespace remainder_of_x13_plus_1_by_x_minus_1_l490_490868

-- Define the polynomial f(x) = x^13 + 1
def f (x : ℕ) : ℕ := x ^ 13 + 1

-- State the theorem using the Polynomial Remainder Theorem
theorem remainder_of_x13_plus_1_by_x_minus_1 : f 1 = 2 := by
  -- Skip the proof
  sorry

end remainder_of_x13_plus_1_by_x_minus_1_l490_490868


namespace sum_of_roots_l490_490861

theorem sum_of_roots (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) :
  ∑ x in {x | a * x^2 + b * x + c = 0}, x = 9 :=
by
  sorry

end sum_of_roots_l490_490861


namespace math_problem_l490_490565

theorem math_problem (x y : ℝ) (h1 : x + Real.sin y = 2023) (h2 : x + 2023 * Real.cos y = 2022) (h3 : Real.pi / 2 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2022 + Real.pi / 2 :=
sorry

end math_problem_l490_490565


namespace lauren_change_l490_490678

-- Define the given conditions as Lean terms.
def price_meat_per_pound : ℝ := 3.5
def pounds_meat : ℝ := 2.0
def price_buns : ℝ := 1.5
def price_lettuce : ℝ := 1.0
def pounds_tomato : ℝ := 1.5
def price_tomato_per_pound : ℝ := 2.0
def price_pickles : ℝ := 2.5
def coupon_value : ℝ := 1.0
def amount_paid : ℝ := 20.0

-- Define the total cost of each item.
def cost_meat : ℝ := pounds_meat * price_meat_per_pound
def cost_tomato : ℝ := pounds_tomato * price_tomato_per_pound
def total_cost_before_coupon : ℝ := cost_meat + price_buns + price_lettuce + cost_tomato + price_pickles

-- Define the final total cost after applying the coupon.
def final_total_cost : ℝ := total_cost_before_coupon - coupon_value

-- Define the expected change.
def expected_change : ℝ := amount_paid - final_total_cost

-- Prove that the expected change is $6.00.
theorem lauren_change : expected_change = 6.0 := by
  sorry

end lauren_change_l490_490678


namespace total_cost_proof_l490_490385

-- Definitions for the problem conditions
def basketball_cost : ℕ := 48
def volleyball_cost : ℕ := basketball_cost - 18
def basketball_quantity : ℕ := 3
def volleyball_quantity : ℕ := 5
def total_basketball_cost : ℕ := basketball_cost * basketball_quantity
def total_volleyball_cost : ℕ := volleyball_cost * volleyball_quantity
def total_cost : ℕ := total_basketball_cost + total_volleyball_cost

-- Theorem to be proved
theorem total_cost_proof : total_cost = 294 :=
by
  sorry

end total_cost_proof_l490_490385


namespace green_apples_count_l490_490714

-- Definitions for the conditions in the problem
def total_apples : ℕ := 19
def red_apples : ℕ := 3
def yellow_apples : ℕ := 14

-- Statement expressing that the number of green apples on the table is 2
theorem green_apples_count : (total_apples - red_apples - yellow_apples = 2) :=
by
  sorry

end green_apples_count_l490_490714


namespace sum_of_values_x_for_f_x_eq_0_l490_490309

def f (x : ℝ) : ℝ := if x ≤ 0 then -x - 5 else (x^2) / 3 + 1

theorem sum_of_values_x_for_f_x_eq_0 :
  ∑ x in {x : ℝ | f x = 0}.to_finset, id x = -5 :=
by
  sorry

end sum_of_values_x_for_f_x_eq_0_l490_490309


namespace students_more_than_pets_l490_490165

theorem students_more_than_pets
  (students_per_classroom : ℕ)
  (rabbits_per_classroom : ℕ)
  (birds_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (total_students : ℕ)
  (total_rabbits : ℕ)
  (total_birds : ℕ)
  (total_pets : ℕ)
  (difference : ℕ)
  : students_per_classroom = 22 → 
    rabbits_per_classroom = 3 → 
    birds_per_classroom = 2 → 
    number_of_classrooms = 5 → 
    total_students = students_per_classroom * number_of_classrooms → 
    total_rabbits = rabbits_per_classroom * number_of_classrooms → 
    total_birds = birds_per_classroom * number_of_classrooms → 
    total_pets = total_rabbits + total_birds → 
    difference = total_students - total_pets →
    difference = 85 := 
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end students_more_than_pets_l490_490165


namespace det_B_eq_one_l490_490306

variable {x y : ℝ}  -- Declaring x and y as real numbers

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![x, 3], ![-4, y]]  -- Defining the matrix B

theorem det_B_eq_one (hB_inv : B + inverse(B) = 0) : Matrix.det B = 1 :=
  sorry  -- Placeholder for the proof

end det_B_eq_one_l490_490306


namespace sum_of_solutions_l490_490993

-- Definitions derived from conditions
def f (x : ℝ) := abs (x^2 - 4 * x + 3)
def g (x : ℝ) := 21 / 4 - x

-- Mathematical statement to prove
theorem sum_of_solutions : 
  (∑ x in { x : ℝ | f x = g x }, x) = 11 / 2 :=
sorry

end sum_of_solutions_l490_490993


namespace area_of_triangle_PQS_l490_490646

-- Define a structure to capture the conditions of the trapezoid and its properties.
structure Trapezoid (P Q R S : Type) :=
(area : ℝ)
(PQ : ℝ)
(RS : ℝ)
(area_PQS : ℝ)
(condition1 : area = 18)
(condition2 : RS = 3 * PQ)

-- Here's the theorem we want to prove, stating the conclusion based on the given conditions.
theorem area_of_triangle_PQS {P Q R S : Type} (T : Trapezoid P Q R S) : T.area_PQS = 4.5 :=
by
  -- Proof will go here, but for now we use sorry.
  sorry

end area_of_triangle_PQS_l490_490646


namespace remove_two_points_preserve_property_l490_490440

noncomputable def problem_statement (n : ℕ) (hn : n ≥ 2) : Prop :=
  let points := list.range (2 * n) in
  let colorings := function.const ℕ (list.range n ++ list.range n) in
  ∀ arc : set ℕ, (1 ≤ arc.card ∧ arc.card < 2 * n) →
    ∃ c ∈ colorings, (arc ∩ {p | colorings p = c}).card = 1 →
  ∃ removed_points : set ℕ, removed_points.card = 2 → ∀ arc : set ℕ,
    (1 ≤ arc.card ∧ arc.card < 2 * (n - 1)) →
    ∃ c ∈ (colorings \ removed_points), (arc ∩ {p | (colorings \ removed_points) p = c}).card = 1

theorem remove_two_points_preserve_property (n : ℕ) (hn : n ≥ 2) :
  problem_statement n hn :=
sorry

end remove_two_points_preserve_property_l490_490440


namespace half_product_unique_l490_490031

theorem half_product_unique (x : ℕ) (n k : ℕ) 
  (hn : x = n * (n + 1) / 2) (hk : x = k * (k + 1) / 2) : 
  n = k := 
sorry

end half_product_unique_l490_490031


namespace probability_passes_through_C_and_D_l490_490954

theorem probability_passes_through_C_and_D
  (grid_size : ℕ)
  (start end : ℕ × ℕ)
  (C D : ℕ × ℕ)
  (path_probability : (ℕ × ℕ) → (ℕ × ℕ) → ℚ)
  (S : finset (list (ℕ × ℕ)))
  (total_paths : ℚ)
  (paths_through_C_and_D : ℚ)
  (probability_through_C_and_D : ℚ)
  (h1 : grid_size = 5)
  (h2 : start = (0, 0))
  (h3 : end = (5, 5))
  (h4 : C = (3, 2))
  (h5 : D = (4, 3))
  (h6 : path_probability = λ start end, (if start.fst + start.snd = end.fst + end.snd then S.card else 0) / total_paths)
  (h7 : total_paths = 252)
  (h8 : paths_through_C_and_D = 40) :
  probability_through_C_and_D = 10 / 63 :=
by
  sorry

end probability_passes_through_C_and_D_l490_490954


namespace base3_20121_to_base10_l490_490493

def base3_to_base10 (n : ℕ) : ℕ :=
  2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0

theorem base3_20121_to_base10 :
  base3_to_base10 20121 = 178 :=
by
  sorry

end base3_20121_to_base10_l490_490493


namespace greatest_increase_l490_490529

-- Define population in 1980 and 1990 for each city
def pop1980_F : ℕ := 60000
def pop1990_F : ℕ := 78000

def pop1980_G : ℕ := 80000
def pop1990_G : ℕ := 104000

def pop1980_H : ℕ := 55000
def pop1990_H : ℕ := 66000

def pop1980_I : ℕ := 90000
def pop1990_I : ℕ := 117000

def pop1980_J : ℕ := 75000
def pop1990_J : ℕ := 90000

-- Define the greatest percentage increase function
def percent_increase (pop1980 pop1990 : ℕ) : rat :=
  (pop1990 : rat) / (pop1980 : rat)

-- Prove that cities F, G, and I have the greatest percentage increase
theorem greatest_increase :
  let f_ratio := percent_increase pop1980_F pop1990_F,
      g_ratio := percent_increase pop1980_G pop1990_G,
      h_ratio := percent_increase pop1980_H pop1990_H,
      i_ratio := percent_increase pop1980_I pop1990_I,
      j_ratio := percent_increase pop1980_J pop1990_J in
  f_ratio = 1.3 ∧ g_ratio = 1.3 ∧ i_ratio = 1.3 ∧
  ∀ r, (r = f_ratio ∨ r = g_ratio ∨ r = i_ratio ∨ r = h_ratio ∨ r = j_ratio) → r ≤ 1.3 :=
by
  intro f_ratio g_ratio h_ratio i_ratio j_ratio
  have : f_ratio = 1.3 := sorry
  have : g_ratio = 1.3 := sorry
  have : i_ratio = 1.3 := sorry
  have : h_ratio = 1.2 := sorry
  have : j_ratio = 1.2 := sorry
  exact ⟨this_1.1, this_1.2, this_1.3,
          λ r hr, hr.elim 
            (λ hf, hf.symm ▸ le_of_eq this_1.1) 
            (λ hg, hg.symm ▸ le_of_eq this_1.2)
            (λ hi, hi.symm ▸ le_of_eq this_1.3)
            (λ hh, hh.symm ▸ this_1.4)
            (λ hj, hj.symm ▸ this_1.5)⟩
  sorry

end greatest_increase_l490_490529


namespace find_n_times_s_l490_490699

variable (f : ℝ → ℝ)
variable (h : ∀ x y : ℝ, f((x - y)^3) = f(x)^3 - 3 * x * f(y)^2 + 3 * y^2 * f(x) - y^3)

theorem find_n_times_s (n s : ℕ) (h1 : (f(1) = 1) ∨ (f(1) = 2))
  (h2 : f(1) = 1 ∧ f(1) = 2 → false)
  (h3 : s = 1 + 2) :
  n * s = 6 :=
by
  have h4 : n = 2 := sorry
  have h5 : s = 3 := sorry
  exact sorry

end find_n_times_s_l490_490699


namespace remainder_s_2022_mod_100_l490_490691

-- Define the polynomial conditions
def q (x : ℕ) : ℕ := x ^ 2012 + x ^ 2011 + x ^ 2010 + ... + x + 1

def s (x : ℕ) : ℕ := -- Division of q(x) by x^5 + x^4 + 2x^3 + x^2 + 1 would be implemented here

-- State the proof problem
theorem remainder_s_2022_mod_100 : s.abs 2022 % 100 = 41 :=
  sorry

end remainder_s_2022_mod_100_l490_490691


namespace value_V3_at_x_neg1_l490_490012

theorem value_V3_at_x_neg1 :
  let a := 1 * 3^3 + 2 * 3^2 + 0 * 3^1 + 2 * 3^0,
      b := Nat.gcd 8251 6105,
      x := -1,
      f := x^5 + a * x^4 - b * x^2 + 1
  in (((x + a) * x) * x - b) * x + 1 = 9 :=
by
  sorry

end value_V3_at_x_neg1_l490_490012


namespace fourth_term_is_neg6_term_is_150_at_16_positive_terms_after_7_l490_490368

-- Define the sequence {a_n} using the general term formula
def seq (n : ℕ) : ℤ := n^2 - 7 * n + 6

-- Prove that the fourth term of the sequence is -6
theorem fourth_term_is_neg6 : seq 4 = -6 := by
  sorry

-- Prove that there exists an n = 16 such that a_n = 150
theorem term_is_150_at_16 : ∃ n : ℕ, seq n = 150 := by
  use 16
  sorry

-- Prove that all terms of the sequence are positive for n ≥ 7
theorem positive_terms_after_7 (n : ℕ) : n ≥ 7 → seq n > 0 := by
  sorry

end fourth_term_is_neg6_term_is_150_at_16_positive_terms_after_7_l490_490368


namespace distance_between_trees_l490_490434

-- Lean 4 statement for the proof problem
theorem distance_between_trees (n : ℕ) (yard_length : ℝ) (h_n : n = 26) (h_length : yard_length = 600) :
  yard_length / (n - 1) = 24 :=
by
  sorry

end distance_between_trees_l490_490434


namespace sum_of_solutions_eq_9_l490_490810

theorem sum_of_solutions_eq_9 (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20) :
  let (sum_roots : ℝ) := -b / a in 
  sum_roots = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490810


namespace geometric_sequence_terms_sum_l490_490999

theorem geometric_sequence_terms_sum :
  ∀ (a_n : ℕ → ℝ) (q : ℝ),
    (∀ n, a_n (n + 1) = a_n n * q) ∧ a_n 1 = 3 ∧
    (a_n 1 + a_n 2 + a_n 3) = 21 →
    (a_n (1 + 2) + a_n (1 + 3) + a_n (1 + 4)) = 84 :=
by
  intros a_n q h
  sorry

end geometric_sequence_terms_sum_l490_490999


namespace range_of_f_gt_f_of_quadratic_l490_490156

-- Define the function f and its properties
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_increasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

-- Define the problem statement
theorem range_of_f_gt_f_of_quadratic (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_inc : is_increasing_on_pos f) :
  {x : ℝ | f x > f (x^2 - 2*x + 2)} = {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end range_of_f_gt_f_of_quadratic_l490_490156


namespace sum_of_roots_l490_490862

theorem sum_of_roots (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) :
  ∑ x in {x | a * x^2 + b * x + c = 0}, x = 9 :=
by
  sorry

end sum_of_roots_l490_490862


namespace distance_between_foci_l490_490487

def semi_major_axis : ℝ := 8
def semi_minor_axis : ℝ := 3

theorem distance_between_foci : 2 * real.sqrt (semi_major_axis^2 - semi_minor_axis^2) = 2 * real.sqrt 55 := by
  sorry

end distance_between_foci_l490_490487


namespace rectangular_to_cylindrical_l490_490958

theorem rectangular_to_cylindrical (x y z r θ : ℝ) (hx : x = 3) (hy : y = -3 * Real.sqrt 3) (hz : z = 2)
    (h_r : r = Real.sqrt (x^2 + y^2)) (h_θ : θ = Real.arctan2 y x) :
    (r = 6) ∧ (θ = 5 * Real.pi / 3) ∧ (z = 2) :=
by
  -- Definitions/conditions from problem.
  have hx : x = 3 := hx
  have hy : y = -3 * Real.sqrt 3 := hy
  have hz : z = 2 := hz
  have h_r : r = Real.sqrt (x^2 + y^2) := h_r
  have h_θ : θ = Real.arctan2 y x := h_θ
  sorry -- placeholder for detailed proof steps

end rectangular_to_cylindrical_l490_490958


namespace smallest_area_of_triangle_l490_490299

open Real

noncomputable def vec := (ℝ × ℝ × ℝ)

def A : vec := (-2, 0, 3)
def B : vec := (0, 3, 4)
def C (s : ℝ) : vec := (s, 0, 0)

def cross_product (u v : vec) : vec :=
  (u.2.2 * v.3 - u.3 * v.2.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2.2 - u.2.2 * v.1)

def vec_norm (v : vec) : ℝ :=
  sqrt (v.1 ^ 2 + v.2.2 ^ 2 + v.3 ^ 2)

def area_triangle (A B C : vec) : ℝ :=
  1 / 2 * vec_norm (cross_product (B.1 - A.1, B.2.2 - A.2.2, B.3 - A.3) (C.1 - A.1, C.2.2 - A.2.2, C.3 - A.3))

theorem smallest_area_of_triangle : ∀ s : ℝ, area_triangle A B (C s) = sqrt 61 / 2 :=
sorry

end smallest_area_of_triangle_l490_490299


namespace M_is_centroid_of_PQR_area_ratio_ABC_PQR_l490_490286

-- We assume that point M exists with the given parallel properties and segment lengths

variables {A B C M P Q R : Point}
variable [Plane]

-- All the points are distinct and well-defined in the plane
axiom h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ A ≠ M ∧ B ≠ M ∧ C ≠ M

-- Definitions for segments from M to P, Q, R being equal to sides of triangle ABC respectively
axiom h_segments : (segment M P) ≈ (segment C A) ∧ (segment M Q) ≈ (segment A B) ∧ (segment M R) ≈ (segment B C)

-- Parallels from point M
axiom h_parallel_mp : parallel (line M P) (line C A)
axiom h_parallel_mq : parallel (line M Q) (line A B)
axiom h_parallel_mr : parallel (line M R) (line B C)

-- To prove that M is the centroid of triangle PQR
theorem M_is_centroid_of_PQR :
  is_centroid M P Q R :=
begin
  sorry
end

-- To prove the area ratio of triangles ABC and PQR is 1:3
theorem area_ratio_ABC_PQR :
  area_ratio (triangle A B C) (triangle P Q R) = (1 : ℝ) / 3 :=
begin
  sorry
end

end M_is_centroid_of_PQR_area_ratio_ABC_PQR_l490_490286


namespace solutions_to_z3_eq_1_l490_490527

theorem solutions_to_z3_eq_1 (z : ℂ) : z^3 = 1 ↔ z = 1 ∨ z = (-1 + complex.I * real.sqrt 3) / 2 ∨ z = (-1 - complex.I * real.sqrt 3) / 2 :=
by
  sorry

end solutions_to_z3_eq_1_l490_490527


namespace buddy_baseball_cards_l490_490327

theorem buddy_baseball_cards :
  let tuesday_cards := 200 - 0.30 * 200 in
  let wednesday_cards := tuesday_cards + 0.20 * tuesday_cards in
  let thursday_cards := wednesday_cards - 0.25 * wednesday_cards in
  let friday_cards := thursday_cards + (1/3) * thursday_cards in
  let saturday_cards := friday_cards + 2 * friday_cards in
  let sunday_gain := 0.40 * saturday_cards in
  let sunday_cards := saturday_cards + sunday_gain - 15 in
  let monday_cards := sunday_cards + 3 * sunday_gain in
  monday_cards = 1297 :=
by
  let tuesday_cards := 200 - 0.30 * 200
  let wednesday_cards := tuesday_cards + 0.20 * tuesday_cards
  let thursday_cards := wednesday_cards - 0.25 * wednesday_cards
  let friday_cards := thursday_cards + (1/3) * thursday_cards
  let saturday_cards := friday_cards + 2 * friday_cards
  let sunday_gain := 0.40 * saturday_cards
  let sunday_cards := saturday_cards + sunday_gain - 15
  let monday_cards := sunday_cards + 3 * sunday_gain
  have : monday_cards = 1297 := sorry
  exact this

end buddy_baseball_cards_l490_490327


namespace probability_of_snow_during_holiday_l490_490182

theorem probability_of_snow_during_holiday
  (P_snow_Friday : ℝ)
  (P_snow_Monday : ℝ)
  (P_snow_independent : true) -- Placeholder since we assume independence
  (h_Friday: P_snow_Friday = 0.30)
  (h_Monday: P_snow_Monday = 0.45) :
  ∃ P_snow_holiday, P_snow_holiday = 0.615 :=
by
  sorry

end probability_of_snow_during_holiday_l490_490182


namespace coordinates_of_P_l490_490197

def point (x y : ℝ) := (x, y)

def A : (ℝ × ℝ) := point 1 1
def B : (ℝ × ℝ) := point 4 0

def vector_sub (p q : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 - q.1, p.2 - q.2)

def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (c * v.1, c * v.2)

theorem coordinates_of_P
  (P : ℝ × ℝ)
  (hP : vector_sub P A = scalar_mult 3 (vector_sub B P)) :
  P = (11 / 2, -1 / 2) :=
by
  sorry

end coordinates_of_P_l490_490197


namespace bonnie_roark_wire_length_ratio_l490_490949

-- Define the conditions
def bonnie_wire_pieces : ℕ := 12
def bonnie_wire_length_per_piece : ℕ := 8
def roark_wire_length_per_piece : ℕ := 2
def bonnie_cube_volume : ℕ := 8 * 8 * 8
def roark_total_cube_volume : ℕ := bonnie_cube_volume
def roark_unit_cube_volume : ℕ := 1
def roark_unit_cube_wires : ℕ := 12

-- Calculate Bonnie's total wire length
noncomputable def bonnie_total_wire_length : ℕ := bonnie_wire_pieces * bonnie_wire_length_per_piece

-- Calculate the number of Roark's unit cubes
noncomputable def roark_number_of_unit_cubes : ℕ := roark_total_cube_volume / roark_unit_cube_volume

-- Calculate the total wire used by Roark
noncomputable def roark_total_wire_length : ℕ := roark_number_of_unit_cubes * roark_unit_cube_wires * roark_wire_length_per_piece

-- Calculate the ratio of Bonnie's total wire length to Roark's total wire length
noncomputable def wire_length_ratio : ℚ := bonnie_total_wire_length / roark_total_wire_length

-- State the theorem
theorem bonnie_roark_wire_length_ratio : wire_length_ratio = 1 / 128 := 
by 
  sorry

end bonnie_roark_wire_length_ratio_l490_490949


namespace solve_functional_equation_l490_490697

noncomputable def functional_equation {f : ℚ → ℚ} (s r : ℚ) : Prop :=
  ∀ x y : ℚ, f(x + f(y)) = f(x + r) + y + s

theorem solve_functional_equation (f : ℚ → ℚ) (s r : ℚ)
  (h : functional_equation s r f) :
  f = (λ x, x + r + s) ∨ f = (λ x, -x + r - s) :=
sorry

end solve_functional_equation_l490_490697


namespace pentagon_AE_length_l490_490643

-- Define the problem with necessary conditions
-- Pentagon ABCDE where specific conditions are defined
theorem pentagon_AE_length :
  ∃ (a b c : ℕ), 
    let BC := 2,
    let CD := 2,
    let DE := 2,
    let angleE := 90,                     -- Angle at E is 90 degrees.
    let angleBCD := 120,                 -- Angles at B, C, and D are 120 degrees each.
    a + b * Real.sqrt c = 2 * Real.sqrt 3 ∧ a + b + c = 5
  :=
by
  -- The proof could be written here but is not required for this exercise.
  sorry

end pentagon_AE_length_l490_490643


namespace line_MN_parallel_to_y_axis_l490_490210

-- Definition of points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of vector between two points
def vector_between (P Q : Point) : Point :=
  { x := Q.x - P.x,
    y := Q.y - P.y,
    z := Q.z - P.z }

-- Given points M and N
def M : Point := { x := 3, y := -2, z := 1 }
def N : Point := { x := 3, y := 2, z := 1 }

-- The vector \overrightarrow{MN}
def vec_MN : Point := vector_between M N

-- Theorem: The vector between points M and N is parallel to the y-axis
theorem line_MN_parallel_to_y_axis : vec_MN = {x := 0, y := 4, z := 0} := by
  sorry

end line_MN_parallel_to_y_axis_l490_490210


namespace a_is_5_if_extreme_at_neg3_l490_490562

-- Define the function f with parameter a
def f (a x : ℝ) : ℝ := x^3 + a * x^2 + 3 * x - 9

-- Define the derivative of f
def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 3

-- Define the given condition that f reaches an extreme value at x = -3
def reaches_extreme_at (a : ℝ) : Prop := f_prime a (-3) = 0

-- Prove that a = 5 if f reaches an extreme value at x = -3
theorem a_is_5_if_extreme_at_neg3 : ∀ a : ℝ, reaches_extreme_at a → a = 5 :=
by
  intros a h
  -- Proof omitted
  sorry

end a_is_5_if_extreme_at_neg3_l490_490562


namespace basketball_substitution_modulo_736_l490_490447

theorem basketball_substitution_modulo_736 :
  let b : ℕ → ℕ 
    | 0       := 1
    | (n + 1) := (5 - n) * (11 + n) * b n 
  in ((b 0 + b 1 + b 2 + b 3 + b 4 + b 5) % 1000) = 736 := by
  sorry

end basketball_substitution_modulo_736_l490_490447


namespace cookies_to_pints_l490_490408

theorem cookies_to_pints (c18 : ℕ) (m18 : ℕ) (q : ℕ → ℕ) (c15 : ℕ) : 
  (c18 = 18) → 
  (q(3) = 6) → 
  (m18 = q(3)) → 
  (c15 = 15) → 
  (m18 / c18 * c15 = 5) :=
by
  intros h_c18 h_q h_m18 h_c15
  rw [h_c18, h_m18, h_q, h_c15]
  norm_num
  sorry

end cookies_to_pints_l490_490408


namespace all_positive_integers_occur_in_sequence_l490_490963

noncomputable def a : ℕ → ℕ
| 1     := 1
| 2     := 2
| (n+1) := (List.range (n+2)).find (λ m, m > 0 ∧ (∀ k < n, a k ≠ m) ∧ Nat.gcd m (a n) ≠ 1)

theorem all_positive_integers_occur_in_sequence : ∀ m : ℕ, ∃ n : ℕ, a (n + 1) = m :=
sorry

end all_positive_integers_occur_in_sequence_l490_490963


namespace lauren_change_l490_490677

theorem lauren_change :
  let meat_cost      := 2 * 3.50
  let buns_cost      := 1.50
  let lettuce_cost   := 1.00
  let tomato_cost    := 1.5 * 2.00
  let pickles_cost   := 2.50 - 1.00
  let total_cost     := meat_cost + buns_cost + lettuce_cost + tomato_cost + pickles_cost
  let payment        := 20.00
  let change         := payment - total_cost
  change = 6.00 :=
by
  unfold meat_cost buns_cost lettuce_cost tomato_cost pickles_cost total_cost payment change
  -- Prove the main statement.
  sorry

end lauren_change_l490_490677


namespace even_mult_expressions_divisible_by_8_l490_490724

theorem even_mult_expressions_divisible_by_8 {a : ℤ} (h : ∃ k : ℤ, a = 2 * k) :
  (8 ∣ a * (a^2 + 20)) ∧ (8 ∣ a * (a^2 - 20)) ∧ (8 ∣ a * (a^2 - 4)) := by
  sorry

end even_mult_expressions_divisible_by_8_l490_490724


namespace sum_of_roots_l490_490865

theorem sum_of_roots (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) :
  ∑ x in {x | a * x^2 + b * x + c = 0}, x = 9 :=
by
  sorry

end sum_of_roots_l490_490865


namespace arithmetic_sequence_a1a6_eq_l490_490283

noncomputable def a_1 : ℤ := 2
noncomputable def d : ℤ := 1
noncomputable def a_n (n : ℕ) : ℤ := a_1 + (n - 1) * d

theorem arithmetic_sequence_a1a6_eq :
  (a_1 * a_n 6) = 14 := by 
  sorry

end arithmetic_sequence_a1a6_eq_l490_490283


namespace line_parallel_plane_l490_490550

axiom line (m : Type) : Prop
axiom plane (α : Type) : Prop
axiom has_no_common_points (m : Type) (α : Type) : Prop
axiom parallel (m : Type) (α : Type) : Prop

theorem line_parallel_plane
  (m : Type) (α : Type)
  (h : has_no_common_points m α) : parallel m α := sorry

end line_parallel_plane_l490_490550


namespace find_sin_X_l490_490689

noncomputable def sin_value (X : ℝ) (a b : ℝ) (area : ℝ) (geom_mean : ℝ) : ℝ :=
if X < π / 2 ∧ a * b = 225 ∧ 1 / 2 * a * b * Real.sin X = area then
  Real.sin X 
else 
  0 -- If conditions are not met, return 0

theorem find_sin_X (a b : ℝ) (X : ℝ) :
  X < π / 2 ∧ a * b = 225 ∧ 1 / 2 * a * b * Real.sin X = 81 →
  Real.sin X = 18 / 25 :=
by
  intros h,
  cases h with h_angle h_rest,
  cases h_rest with h_product h_area,
  sorry -- Proof goes here

end find_sin_X_l490_490689


namespace daisies_per_bouquet_l490_490090

def total_bouquets := 20
def rose_bouquets := 10
def roses_per_rose_bouquet := 12
def total_flowers_sold := 190

def total_roses_sold := rose_bouquets * roses_per_rose_bouquet
def daisy_bouquets := total_bouquets - rose_bouquets
def total_daisies_sold := total_flowers_sold - total_roses_sold

theorem daisies_per_bouquet :
  (total_daisies_sold / daisy_bouquets = 7) := sorry

end daisies_per_bouquet_l490_490090


namespace mod_congruence_zero_iff_l490_490621

theorem mod_congruence_zero_iff
  (a b c d n : ℕ)
  (h1 : a * c ≡ 0 [MOD n])
  (h2 : b * c + a * d ≡ 0 [MOD n]) :
  b * c ≡ 0 [MOD n] ∧ a * d ≡ 0 [MOD n] :=
by
  sorry

end mod_congruence_zero_iff_l490_490621


namespace find_polynomial_sum_l490_490622

noncomputable def f (x : ℂ) : ℂ := (x ^ 2 + x + 2) ^ 2014

theorem find_polynomial_sum:
  let a := λ i: ℕ, ((x ^ i).coeff f) in
  (2 * a 0 - a 1 - a 2 + 2 * a 3 - a 4 - a 5 + ... + 2 * a 4020 - a 4027 - a 4028) = 2 :=
by
  sorry

end find_polynomial_sum_l490_490622


namespace find_x_l490_490268

-- Define the product of all even integers from 2 up to x
def evenProduct (x : Nat) : Nat :=
  (List.range' 2 x 2).prod

-- Define {12} as specified
def evenProduct12 : Nat := evenProduct 12

-- The main theorem to prove
theorem find_x (x : Nat) (hx : x % 2 = 0) :
  Prime (evenProduct x + evenProduct12) = 13 :=
sorry

end find_x_l490_490268


namespace ratio_of_areas_l490_490025

-- Define the sides of the triangle
def a : ℝ := 13
def b : ℝ := 14
def c : ℝ := 15

-- Define the semi-perimeter
def semi_perimeter : ℝ := (a + b + c) / 2

-- Define the area using Heron's formula
def area : ℝ := Real.sqrt (semi_perimeter * (semi_perimeter - a) * (semi_perimeter - b) * (semi_perimeter - c))

-- Define the radius of the incircle using the formula r = S/p
def inradius : ℝ := area / semi_perimeter

-- Define the radius of the circumcircle using the formula R = abc / (4S)
def circumradius : ℝ := (a * b * c) / (4 * area)

-- Define the ratio of the areas of the circumcircle to the incircle
def ratio : ℝ := (circumradius / inradius) ^ 2

-- State the theorem
theorem ratio_of_areas : ratio = (65 / 32) ^ 2 := by
  sorry

end ratio_of_areas_l490_490025


namespace marble_cut_percentage_l490_490936

theorem marble_cut_percentage (initial_weight : ℝ)
  (cut_first_week_percent : ℝ)
  (cut_second_week_percent : ℝ)
  (final_weight : ℝ) :
  initial_weight = 190 →
  cut_first_week_percent = 0.25 →
  cut_second_week_percent = 0.15 →
  final_weight = 109.0125 →
  let remaining_after_first_week := initial_weight * (1 - cut_first_week_percent) in
  let remaining_after_second_week := remaining_after_first_week * (1 - cut_second_week_percent) in
  let cut_third_week_percent := 1 - (final_weight / remaining_after_second_week) in
  cut_third_week_percent = 0.0999 :=
by
  intros initial_eq cut1_eq cut2_eq final_w_eq
  simp [initial_eq, cut1_eq, cut2_eq, final_w_eq]
  sorry

end marble_cut_percentage_l490_490936


namespace sum_of_solutions_eq_9_l490_490824

theorem sum_of_solutions_eq_9 :
  let roots := {x : ℝ | x^2 = 9 * x - 20}
  in ∑ x in roots, x = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490824


namespace rectangle_length_l490_490436

theorem rectangle_length (side_square length_rectangle width_rectangle wire_length : ℝ) 
    (h1 : side_square = 12) 
    (h2 : width_rectangle = 6) 
    (h3 : wire_length = 4 * side_square) 
    (h4 : wire_length = 2 * width_rectangle + 2 * length_rectangle) : 
    length_rectangle = 18 := 
by 
  sorry

end rectangle_length_l490_490436


namespace rotate_point_180_about_origin_l490_490720

theorem rotate_point_180_about_origin :
  let A : ℝ × ℝ := (-3, 2)
  ∃ B : ℝ × ℝ, B = (3, -2) ∧ 
  ∀ (x y : ℝ), (x, y) = A → B = (-x, -y) :=
by
  let A : ℝ × ℝ := (-3, 2)
  let B : ℝ × ℝ := (3, -2)
  have H : ∀ (x y : ℝ), (x, y) = A → B = (-x, -y)
  {
    intro x
    intro y
    intro h
    cases h
    unfold A B
    simp
  }
  use B
  split
  { 
    unfold B 
  }
  assumption

end rotate_point_180_about_origin_l490_490720


namespace movie_hours_sum_l490_490671

noncomputable def total_movie_hours 
  (Michael Joyce Nikki Ryn Sam : ℕ) 
  (h1 : Joyce = Michael + 2)
  (h2 : Nikki = 3 * Michael)
  (h3 : Ryn = (4 * Nikki) / 5)
  (h4 : Sam = (3 * Joyce) / 2)
  (h5 : Nikki = 30) : ℕ :=
  Joyce + Michael + Nikki + Ryn + Sam

theorem movie_hours_sum (Michael Joyce Nikki Ryn Sam : ℕ) 
  (h1 : Joyce = Michael + 2)
  (h2 : Nikki = 3 * Michael)
  (h3 : Ryn = (4 * Nikki) / 5)
  (h4 : Sam = (3 * Joyce) / 2)
  (h5 : Nikki = 30) : 
  total_movie_hours Michael Joyce Nikki Ryn Sam h1 h2 h3 h4 h5 = 94 :=
by 
  -- The actual proof will go here, to demonstrate the calculations resulting in 94 hours
  sorry

end movie_hours_sum_l490_490671


namespace area_of_circle_is_correct_l490_490418

/-- 
Prove that the area of a circle with center at point P (2, -1)
and passing through point Q (-4, 5) is 72π.
-/
noncomputable def area_of_circle : ℝ :=
  let P : ℝ × ℝ := (2, -1)
  let Q : ℝ × ℝ := (-4, 5)
  let radius := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let area := π * radius^2
  area

theorem area_of_circle_is_correct : area_of_circle = 72 * π :=
by
  sorry

end area_of_circle_is_correct_l490_490418


namespace base3_20121_to_base10_l490_490494

def base3_to_base10 (n : ℕ) : ℕ :=
  2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0

theorem base3_20121_to_base10 :
  base3_to_base10 20121 = 178 :=
by
  sorry

end base3_20121_to_base10_l490_490494


namespace plane_EAB_perpendicular_plane_ABCD_cosine_dihedral_angle_AEC_DCE_l490_490119

-- Conditions
variables {A B C D E : Type}
variables (angle_ABC : ℝ) [fact (angle_ABC = 60 * (π / 180))]
variables (AB EC : ℝ) [fact (AB = 2)] [fact (EC = 2)]
variables (AE BE : ℝ) [fact (AE = real.sqrt 2)] [fact (BE = real.sqrt 2)]

-- Proof of Part 1: Plane EAB ⊥ Plane ABCD
theorem plane_EAB_perpendicular_plane_ABCD : is_perpendicular (plane, [E, A, B]) (plane, [A, B, C, D]) :=
  sorry

-- Proof of Part 2: Find cosine of dihedral angle AEC-DCE
theorem cosine_dihedral_angle_AEC_DCE : 
  cos_dihedral_angle (plane, [A, E, C]) (plane, [D, C, E]) = (2 * sqrt 7 / 7) :=
  sorry

end plane_EAB_perpendicular_plane_ABCD_cosine_dihedral_angle_AEC_DCE_l490_490119


namespace drivers_sufficient_l490_490924

theorem drivers_sufficient
  (round_trip_duration : ℕ := 320)
  (rest_duration : ℕ := 60)
  (return_time_A : ℕ := 12 * 60 + 40)
  (depart_time_D : ℕ := 13 * 60 + 5)
  (next_depart_time_A : ℕ := 13 * 60 + 40)
  : (4 : ℕ) ∧ (21 * 60 + 30 = 1290) := 
  sorry

end drivers_sufficient_l490_490924


namespace perimeter_of_rectangle_is_correct_l490_490461

-- Define the sides of the right triangle
def side1 : ℝ := 9
def side2 : ℝ := 12
def hypotenuse : ℝ := 15

-- Verify the Pythagorean theorem
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- Define the area of the right triangle
def triangle_area (a b : ℝ) : ℝ := (1/2) * a * b

-- Define the length of one side of the rectangle
def rectangle_length : ℝ := hypotenuse / 2

-- Define the width of the rectangle using the area
def rectangle_width (A len : ℝ) : ℝ := A / len

-- Calculate the perimeter of the rectangle
def rectangle_perimeter (len wid : ℝ) : ℝ := 2 * (len + wid)

-- The target statement to prove
theorem perimeter_of_rectangle_is_correct :
  is_right_triangle side1 side2 hypotenuse →
  let A := triangle_area side1 side2 in
  let len := rectangle_length in
  let wid := rectangle_width A len in
  rectangle_perimeter len wid = 29.4 :=
by
  sorry

end perimeter_of_rectangle_is_correct_l490_490461


namespace continuity_f_at_x₀_l490_490441

namespace ContinuityProof

def f (x : ℝ) := 2 * x^2 - 3
def x₀ : ℝ := 4

theorem continuity_f_at_x₀ : ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, abs (x - x₀) < δ → abs (f x - f x₀) < ε :=
by
  intros ε ε_pos
  let δ := ε / 18
  have δ_pos : δ > 0 := ε_pos / 18
  use δ, δ_pos
  intros x h
  have : abs (f x - f x₀) = 2 * abs ((x - x₀) * (x + x₀)) := sorry
  sorry

end ContinuityProof

end continuity_f_at_x₀_l490_490441


namespace solve_first_equation_solve_second_equation_l490_490349

-- Problem 1: Solve x(x+2) - (x+2) = 0
theorem solve_first_equation : 
  ∀ x : ℝ, x * (x + 2) - (x + 2) = 0 ↔ (x = -2 ∨ x = 1) := 
by
  intro x
  sorry

-- Problem 2: Solve x^2 - 4x - 3 = 0
theorem solve_second_equation : 
  ∀ x : ℝ, x^2 - 4x - 3 = 0 ↔ (x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) := 
by
  intro x
  sorry

end solve_first_equation_solve_second_equation_l490_490349


namespace probability_palindrome_divisible_by_11_l490_490908

theorem probability_palindrome_divisible_by_11 :
  (∑ (a : ℕ) (b : ℕ) (c : ℕ) in finset.Icc (0:ℕ) 10, 
    if 2 * a - c % 11 = 0 then 1 else 0) / 900 = 1 / 5 :=
by {
  -- Using ∑ constexpr as inlinable form of finset.sum
  sorry
}

end probability_palindrome_divisible_by_11_l490_490908


namespace slope_of_vertical_line_l490_490393

theorem slope_of_vertical_line (x : ℝ) : (∀ y : ℝ, ¬ is_slope (x = 1) y) :=
by
  sorry

end slope_of_vertical_line_l490_490393


namespace sum_of_solutions_l490_490785

-- Define the quadratic equation and variable x
def quadratic_equation := ∀ x : ℝ, (x^2 - 9 * x + 20 = 0)

-- Define what we need to prove
theorem sum_of_solutions : ∃ s : ℝ, (∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → s = x1 + x2) ∧ s = 9 :=
by
  sorry -- Proof is omitted

end sum_of_solutions_l490_490785


namespace train_length_is_correct_l490_490112

noncomputable def train_length (speed_kmph : ℝ) (time_sec : ℝ) (bridge_length : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * time_sec
  total_distance - bridge_length

theorem train_length_is_correct :
  train_length 60 20.99832013438925 240 = 110 :=
by
  sorry

end train_length_is_correct_l490_490112


namespace four_drivers_suffice_l490_490917

theorem four_drivers_suffice
  (one_way_trip_time : ℕ := 160) -- in minutes
  (round_trip_time : ℕ := 320) -- in minutes
  (rest_time : ℕ := 60) -- in minutes
  (time_A_returns : ℕ := 760) -- 12:40 PM in minutes from midnight
  (time_A_next_start : ℕ := 820) -- 1:40 PM in minutes from midnight
  (time_D_departs : ℕ := 785) -- 1:05 PM in minutes from midnight
  (time_A_fifth_depart : ℕ := 970) -- 4:10 PM in minutes from midnight
  (time_B_returns : ℕ := 960) -- 4:00 PM in minutes from midnight
  (time_B_sixth_depart : ℕ := 1050) -- 5:30 PM in minutes from midnight
  (time_A_fifth_complete: ℕ := 1290) -- 9:30 PM in minutes from midnight
  : 4_drivers_sufficient : ℕ :=
    if time_A_fifth_complete = 1290 then 1 else 0
-- The theorem states that if the calculated trip completion time is 9:30 PM, then 4 drivers are sufficient.
  sorry

end four_drivers_suffice_l490_490917


namespace zach_fill_time_l490_490315

theorem zach_fill_time : 
  ∀ (t : ℕ), 
  (∀ (max_time max_rate zach_rate popped total : ℕ), 
    max_time = 30 → 
    max_rate = 2 → 
    zach_rate = 3 → 
    popped = 10 → 
    total = 170 → 
    (max_time * max_rate + t * zach_rate - popped = total) → 
    t = 40) := 
sorry

end zach_fill_time_l490_490315


namespace pentagon_same_parity_l490_490892

open Classical

theorem pentagon_same_parity (vertices : Fin 5 → ℤ × ℤ) : 
  ∃ i j : Fin 5, i ≠ j ∧ (vertices i).1 % 2 = (vertices j).1 % 2 ∧ (vertices i).2 % 2 = (vertices j).2 % 2 :=
by
  sorry

end pentagon_same_parity_l490_490892


namespace max_min_difference_x_l490_490947

noncomputable def condition_start_yes : ℕ := 40
noncomputable def condition_start_no : ℕ := 30
noncomputable def condition_start_undecided : ℕ := 30
noncomputable def condition_end_yes : ℕ := 60
noncomputable def condition_end_no : ℕ := 20
noncomputable def condition_end_undecided : ℕ := 20

theorem max_min_difference_x :
  let max_change := 50 in
  let min_change := 20 in
  max_change - min_change = 30 :=
by {
  let start_yes := condition_start_yes,
  let start_no := condition_start_no,
  let start_undecided := condition_start_undecided,
  let end_yes := condition_end_yes,
  let end_no := condition_end_no,
  let end_undecided := condition_end_undecided,
  -- The following are the given conditions in the problem statement
  have h1 : start_yes = 40 := rfl,
  have h2 : start_no = 30 := rfl,
  have h3 : start_undecided = 30 := rfl,
  have h4 : end_yes = 60 := rfl,
  have h5 : end_no = 20 := rfl,
  have h6 : end_undecided = 20 := rfl,

  -- Definitions of maximum and minimum changes based on given conditions
  let max_change := 50,
  let min_change := 20,
  
  show max_change - min_change = 30 from rfl,
}

end max_min_difference_x_l490_490947


namespace even_function_sum_l490_490366

theorem even_function_sum (a b : ℝ) (h_even : ∀ x : ℝ, f(x) = a * x^2 + b * x + 2 * a - b ∧ f (-x) = f (x)) 
                          (h_domain : ∀ x : ℝ, x ∈ Icc (a - 1) (2 * a)) : a + b = 1/3 :=
by
  sorry

noncomputable def f (x : ℝ) : ℝ := a * x^2 + b * x + 2 * a - b

end even_function_sum_l490_490366


namespace smallest_t_value_l490_490390

theorem smallest_t_value : ∀ (t : ℕ), (7.5 < 11 + t ∧ 
                                        t < 18.5 ∧
                                        t > 3.5) → t = 4 :=
by
  intros t h
  sorry

end smallest_t_value_l490_490390


namespace equation_not_expression_with_unknowns_l490_490474

def is_equation (expr : String) : Prop :=
  expr = "equation"

def contains_unknowns (expr : String) : Prop :=
  expr = "contains unknowns"

theorem equation_not_expression_with_unknowns (expr : String) (h1 : is_equation expr) (h2 : contains_unknowns expr) : 
  (is_equation expr) = False := 
sorry

end equation_not_expression_with_unknowns_l490_490474


namespace sum_of_solutions_eq_9_l490_490806

theorem sum_of_solutions_eq_9 (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20) :
  let (sum_roots : ℝ) := -b / a in 
  sum_roots = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490806


namespace multiplicative_inverse_modulo_l490_490700

theorem multiplicative_inverse_modulo :
  let A := 222222
  let B := 142857
  let M := 2000000
  let N := 126
  N < 1000000 ∧ N * (A * B) % M = 1 :=
by
  let A := 222222
  let B := 142857
  let M := 2000000
  let N := 126
  have h1 : A = 222222 := rfl
  have h2 : B = 142857 := rfl
  have h3 : M = 2000000 := rfl
  have h4 : N = 126 := rfl
  exact And.intro
    (show N < 1000000 from sorry)
    (show N * (A * B) % M = 1 from sorry)

end multiplicative_inverse_modulo_l490_490700


namespace factor_polynomial_l490_490986

theorem factor_polynomial : 
  (x : ℝ) → x^4 - 4 * x^2 + 16 = (x^2 - 4 * x + 4) * (x^2 + 2 * x + 4) :=
by
sorry

end factor_polynomial_l490_490986


namespace coefficient_x2_in_expansion_l490_490747

-- Define combinatorial binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the coefficient of the x^3 term in the expansion of (x-1)^6
def coeff_x3 := binom 6 3 * (-1)^3

-- The main theorem statement
theorem coefficient_x2_in_expansion :
  coeff_x3 = -20 := 
by
  sorry

end coefficient_x2_in_expansion_l490_490747


namespace part1_part2_l490_490311

def f (x : ℝ) : ℝ := abs (x - 2) - abs (2 * x + 1)

theorem part1 (x : ℝ) : f(x) ≤ 0 ↔ x ≥ 1 / 3 ∨ x ≤ -3 := sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, f(x) - 2 * m^2 ≤ 4 * m) ↔ (m ≥ 1 / 2 ∨ m ≤ -5 / 2) := sorry

end part1_part2_l490_490311


namespace sum_of_solutions_eq_9_l490_490825

theorem sum_of_solutions_eq_9 :
  let roots := {x : ℝ | x^2 = 9 * x - 20}
  in ∑ x in roots, x = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490825


namespace area_of_triangle_l490_490633

theorem area_of_triangle (A B C : Type) [EuclideanSpace (point A)] [EuclideanSpace (point B)] [EuclideanSpace (point C)] 
  (a b c : ℝ) (angle_B : ℝ) (h1 : a = 1) (h2 : c = 2) (h3 : angle_B = 60) : 
  area_of_triangle A B C = 1 :=
begin
  sorry
end

end area_of_triangle_l490_490633


namespace solve_for_t_l490_490346

theorem solve_for_t (t : ℝ) : sqrt(3 * sqrt(t - 1)) = (t + 9)^(1/4) → t = 9/4 := 
by 
    intros h
    sorry

end solve_for_t_l490_490346


namespace number_of_sides_of_polygon_l490_490274

theorem number_of_sides_of_polygon (sum_exterior_angles : ℝ) (exterior_angle : ℝ) (h1 : sum_exterior_angles = 360) (h2 : exterior_angle = 45) : (n : ℕ), n = sum_exterior_angles / exterior_angle :=
by
  have h3 : sum_exterior_angles = 360 := h1
  have h4 : exterior_angle = 45 := h2
  let n := 360 / 45
  exact sorry

end number_of_sides_of_polygon_l490_490274


namespace find_circle_eqn_l490_490519

open Real

def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4
def is_tangent (C1 C2 : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop := 
    ∃ (x y : ℝ), C1 x y ∧ C2 x y ∧ P = (x, y)

theorem find_circle_eqn :
  ∃ C : ℝ → ℝ → Prop, is_tangent circle1 C (4, -1) ∧ (∀ x y, C x y ↔ (x - 5)^2 + (y + 1)^2 = 1) := sorry

end find_circle_eqn_l490_490519


namespace trapezoid_ratio_l490_490682

variables {AB CD AD AC BM BD : ℝ}
variables (ABCD_trapezoid : AB ∥ CD)
variables (M_midpoint_CD : M = (CD / 2))
variables (AD_perp_CD : AD ⊥ CD)
variables (AC_perp_BM : AC ⊥ BM)
variables (BC_perp_BD : BC ⊥ BD)

theorem trapezoid_ratio : 
  (ABCD_trapezoid) →
  (M_midpoint_CD) →
  (AD_perp_CD) →
  (AC_perp_BM) →
  (BC_perp_BD) →
  AB / CD = sqrt(2) / 2 :=
begin
  sorry
end

end trapezoid_ratio_l490_490682


namespace volume_of_cube_l490_490405

theorem volume_of_cube (SA : ℝ) (H : SA = 600) : (10^3 : ℝ) = 1000 :=
by
  sorry

end volume_of_cube_l490_490405


namespace base_eight_to_ten_l490_490422

theorem base_eight_to_ten (n : Nat) (h : n = 52) : 8 * 5 + 2 = 42 :=
by
  -- Proof will be written here.
  sorry

end base_eight_to_ten_l490_490422


namespace probability_closer_to_5_than_0_l490_490458

theorem probability_closer_to_5_than_0 : 
  let interval := set.Icc 0 8,
      closer_to_5 := set.Ioc 2.5 8 in
  ∃ P : ℚ, P = 0.6875 ∧
    (measure_theory.measure_space.volume closer_to_5 / 
     measure_theory.measure_space.volume interval : ℚ) = P :=
by
  sorry

end probability_closer_to_5_than_0_l490_490458


namespace compare_fraction_product_l490_490135

theorem compare_fraction_product :
  (∏ i in Finset.range 461, (100 + 2*i : ℝ) / (101 + 2*i)) < (5/16 : ℝ) := 
sorry

end compare_fraction_product_l490_490135


namespace find_eccentricity_l490_490597

open Real

-- Define the parametric equations
def parametric_eq (α : ℝ) : ℝ × ℝ := (5 * cos α, 3 * sin α)

-- The rectangular form confirms the curve is an ellipse with given axes
def is_ellipse : Prop :=
  ∀ (x y : ℝ), (∃ α, x = 5 * cos α ∧ y = 3 * sin α) → (x^2 / 25 + y^2 / 9 = 1)

-- Compute the eccentricity of the ellipse
def eccentricity (a b : ℝ) : ℝ :=
  let c := sqrt (a^2 - b^2) in c / a

-- Given the specific ellipse parameters, calculate the eccentricity
theorem find_eccentricity : is_ellipse → eccentricity 5 3 = 4 / 5 :=
by
  sorry

end find_eccentricity_l490_490597


namespace translated_function_value_l490_490282

def initial_function (x : ℝ) : ℝ := sin (2 * x)

def translated_function (x : ℝ) : ℝ := sin (2 * (x - (π / 12)))

theorem translated_function_value : translated_function (π / 12) = 0 := by
  sorry

end translated_function_value_l490_490282


namespace angle_between_adjacent_triangles_l490_490531

-- Define the setup of the problem
def five_nonoverlapping_equilateral_triangles (angles : Fin 5 → ℝ) :=
  ∀ i, angles i = 60

def angles_between_adjacent_triangles (angles : Fin 5 → ℝ) :=
  ∀ i j, i ≠ j → angles i = angles j

-- State the main theorem
theorem angle_between_adjacent_triangles :
  ∀ (angles : Fin 5 → ℝ),
    five_nonoverlapping_equilateral_triangles angles →
    angles_between_adjacent_triangles angles →
    ((360 - 5 * 60) / 5) = 12 :=
by
  intros angles h1 h2
  sorry

end angle_between_adjacent_triangles_l490_490531


namespace cos_phi_value_l490_490264

theorem cos_phi_value
  (f : ℝ → ℝ)
  (φ x1 : ℝ)
  (h1 : ∀ x, f (-x) - sin (-x + φ) = f x - sin (x + φ))
  (h2 : ∀ x, f (-x) - cos (-x + φ) = - (f x - cos (x + φ)))
  (h3 : (cos x1 - sin x1) * cos φ * (cos (x1 + π / 2) - sin (x1 + π / 2)) * cos φ = 1) : 
  cos φ = 1 ∨ cos φ = -1 :=
by
  sorry

end cos_phi_value_l490_490264


namespace pagoda_lights_l490_490442

theorem pagoda_lights :
  ∃ a₁ : ℕ, (a₁ * (2^7 - 1) = 381) ∧ (a₁ = 3) :=
by {
  use 3,
  split,
  { -- a₁ * (2^7 - 1) = 381
    calc 3 * (2^7 - 1) = 3 * 127 : by rw pow_succ
                    ... = 381      : by norm_num },
  { -- a₁ = 3
    refl }
}

end pagoda_lights_l490_490442


namespace _l490_490115

noncomputable def waiter_fraction_from_tips (S T I : ℝ) : Prop :=
  T = (5 / 2) * S ∧
  I = S + T ∧
  T / I = 5 / 7

lemma waiter_tips_fraction_theorem (S T I : ℝ) : waiter_fraction_from_tips S T I → T / I = 5 / 7 :=
by
  intro h
  rw [waiter_fraction_from_tips] at h
  obtain ⟨h₁, h₂, h₃⟩ := h
  exact h₃

end _l490_490115


namespace scientific_notation_of_40000000_l490_490010

theorem scientific_notation_of_40000000 : ∃ a n : ℤ, (1 ≤ a ∧ a < 10) ∧ (40000000 = a * 10^n) ∧ (a = 4) ∧ (n = 7) :=
by
  sorry

end scientific_notation_of_40000000_l490_490010


namespace sum_of_solutions_l490_490834

theorem sum_of_solutions (x : ℝ) : 
  (∀ x : ℝ, x^2 = 9*x - 20 → x = 4 ∨ x = 5) → (4 + 5 = 9) :=
by
  intros h
  calc 4 + 5 = 9 : by norm_num
  sorry

end sum_of_solutions_l490_490834


namespace polygon_sides_l490_490276

theorem polygon_sides (e : ℝ) (h : e = 45) : 360 / e = 8 :=
by {
  rw [h],
  norm_num,
}

end polygon_sides_l490_490276


namespace proof_volume_and_bk_length_l490_490752

noncomputable def volume_prism_height_6 (a : ℝ) : ℝ :=
  let S := (a^2 * real.sqrt 3) / 4
  let h := 6
  S * h

noncomputable def inradius_eq (a : ℝ) : Prop :=
  let r := real.sqrt (8 / 3)
  r = (a * real.sqrt 3) / 6

def valid_bk_length (a : ℝ) (r : ℝ) : Prop :=
  let k := a * (real.sqrt 3) / 3
  k = 1 ∨ k = 5

theorem proof_volume_and_bk_length :
  ∃ a : ℝ, volume_prism_height_6 a = 48 * real.sqrt 3 ∧ inradius_eq a ∧ valid_bk_length a (real.sqrt (8 / 3)) :=
sorry

end proof_volume_and_bk_length_l490_490752


namespace valid_tickets_percentage_l490_490475

theorem valid_tickets_percentage (cars : ℕ) (people_without_payment : ℕ) (P : ℚ) 
  (h_cars : cars = 300) (h_people_without_payment : people_without_payment = 30) 
  (h_total_valid_or_passes : (cars - people_without_payment = 270)) :
  P + (P / 5) = 90 → P = 75 :=
by
  sorry

end valid_tickets_percentage_l490_490475


namespace triangle_area_equal_l490_490203

variable (A B C D : Type) [plane_geometry : PlaneGeometry A B C D]

theorem triangle_area_equal (tABC : triangle A B C) (pointD : point D) :
  let ortho_ABD := orthocenter (triangle A B pointD),
      ortho_BCD := orthocenter (triangle B C pointD),
      ortho_CAD := orthocenter (triangle C A pointD) in
  area (triangle ortho_ABD ortho_BCD ortho_CAD) = area tABC :=
sorry

end triangle_area_equal_l490_490203


namespace cost_of_fencing_proof_l490_490889

noncomputable def cost_of_fencing (area : ℝ) (ratio : ℝ × ℝ) (rate_per_meter_paise : ℝ) : ℝ :=
let x := real.sqrt (area / (ratio.1 * ratio.2)) in
let length := ratio.1 * x in
let width := ratio.2 * x in
let perimeter := 2 * (length + width) in
let rate_per_meter_rupees := rate_per_meter_paise / 100 in
perimeter * rate_per_meter_rupees

theorem cost_of_fencing_proof :
  cost_of_fencing 8748 (4, 3) 25 = 94.5 :=
by {
  -- Proof omitted
  sorry
}

end cost_of_fencing_proof_l490_490889


namespace base_three_to_base_ten_l490_490492

theorem base_three_to_base_ten : 
  (2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0 = 178) :=
by
  sorry

end base_three_to_base_ten_l490_490492


namespace geometric_sequence_max_product_l490_490206

noncomputable def max_geometric_product (a₁ a₂ a₃ : ℝ) (q: ℝ) : ℝ :=
  a₁ * a₂ * a₃

theorem geometric_sequence_max_product :
  ∃ (a₁ a₂ a₃ q : ℝ), a₁ + a₃ = 5 ∧ a₂ + a₄ = 5/2 ∧
  q = 1/2 ∧ a₁ = 4 ∧ a₂ = a₁ * q ∧ a₃ = a₁ * q^2 ∧
  max_geometric_product a₁ a₂ a₃ q = 8 := 
begin
  sorry
end

end geometric_sequence_max_product_l490_490206


namespace area_inner_triangle_l490_490242

-- Definitions: Semiperimeter, Area using Heron's formula
def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

def triangle_area (a b c : ℝ) (s : ℝ) : ℝ :=
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem: Area of the inner triangle
theorem area_inner_triangle
  (a b c d : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : d > 0) :
  let s := semiperimeter a b c in
  let t := triangle_area a b c s in
  let t' := (t - d * s) * (t - d * s) / t in
  t' = (t - d * s) * (t - d * s) / t :=
by
  sorry

end area_inner_triangle_l490_490242


namespace sum_of_solutions_l490_490786

-- Define the quadratic equation and variable x
def quadratic_equation := ∀ x : ℝ, (x^2 - 9 * x + 20 = 0)

-- Define what we need to prove
theorem sum_of_solutions : ∃ s : ℝ, (∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → s = x1 + x2) ∧ s = 9 :=
by
  sorry -- Proof is omitted

end sum_of_solutions_l490_490786


namespace training_distance_l490_490038

theorem training_distance (a₁ d n : ℕ) (hₐ₁ : a₁ = 5000) (hd : d = 200) (hn : n = 7) : 
  let S := n * a₁ + (n * (n - 1) / 2) * d in S = 39200 :=
by
  sorry

end training_distance_l490_490038


namespace find_error_page_l490_490377

theorem find_error_page (n : ℕ) (x : ℕ) (h1 : 1 ≤ n) (h2 : ∑ i in (range n).succ, i = n * (n + 1) / 2)
  (h3 : n * (n + 1) / 2 + x = 2076) : x = 60 := 
sorry

end find_error_page_l490_490377


namespace ellipse_foci_y_axis_l490_490008

theorem ellipse_foci_y_axis (k : ℝ) :
  (∃ a b : ℝ, a = 15 - k ∧ b = k - 9 ∧ a > 0 ∧ b > 0) ↔ (12 < k ∧ k < 15) :=
by
  sorry

end ellipse_foci_y_axis_l490_490008


namespace mrs_heine_dogs_l490_490710

theorem mrs_heine_dogs (total_biscuits biscuits_per_dog : ℕ) (h1 : total_biscuits = 6) (h2 : biscuits_per_dog = 3) :
  total_biscuits / biscuits_per_dog = 2 :=
by
  sorry

end mrs_heine_dogs_l490_490710


namespace identify_quadratic_equation_l490_490872

def is_quadratic_one_variable (eq : Type) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ eq = fun x : ℝ => a * x^2 + b * x + c = 0

theorem identify_quadratic_equation (E1 E2 E3 E4 : Type) :
  E1 = (λ x y : ℝ, x^2 - 4 * y = 0) →
  E2 = (λ x : ℝ, x^2 + x + 3 = 0) →
  E3 = (λ x : ℝ, 2 * x = 5) →
  E4 = (λ x : ℝ, x^2 + x⁻¹ - 2 = 0) →
  is_quadratic_one_variable E2 :=
begin
  sorry
end

end identify_quadratic_equation_l490_490872


namespace percentage_conveyance_l490_490729

def percentage_on_food := 40 / 100
def percentage_on_rent := 20 / 100
def percentage_on_entertainment := 10 / 100
def salary := 12500
def savings := 2500

def total_percentage_spent := percentage_on_food + percentage_on_rent + percentage_on_entertainment
def total_spent := salary - savings
def amount_spent_on_conveyance := total_spent - (salary * total_percentage_spent)
def percentage_spent_on_conveyance := (amount_spent_on_conveyance / salary) * 100

theorem percentage_conveyance : percentage_spent_on_conveyance = 10 :=
by sorry

end percentage_conveyance_l490_490729


namespace sufficient_drivers_and_correct_time_l490_490914

-- Conditions definitions
def one_way_minutes := 2 * 60 + 40  -- 2 hours 40 minutes in minutes
def round_trip_minutes := 2 * one_way_minutes  -- round trip in minutes
def rest_minutes := 60  -- mandatory rest period in minutes

-- Time checks for drivers
def driver_a_return := 12 * 60 + 40  -- Driver A returns at 12:40 PM in minutes
def driver_a_next_trip := driver_a_return + rest_minutes  -- Driver A's next trip time
def driver_d_departure := 13 * 60 + 5  -- Driver D departs at 13:05 in minutes

-- Verify sufficiency of four drivers and time correctness
theorem sufficient_drivers_and_correct_time : 
  4 = 4 ∧ (driver_a_next_trip + round_trip_minutes = 21 * 60 + 30) :=
by
  -- Explain the reasoning path that leads to this conclusion within this block
  sorry

end sufficient_drivers_and_correct_time_l490_490914


namespace second_number_is_sixty_l490_490064

theorem second_number_is_sixty (x : ℕ) (h_sum : 2 * x + x + (2 / 3) * x = 220) : x = 60 :=
by
  sorry

end second_number_is_sixty_l490_490064


namespace find_k_l490_490899

theorem find_k
  (k : ℕ)
  (h₁ : k > 0)
  (h₂ : (24 - k) / (8 + k) = 0.60) : k = 12 :=
sorry

end find_k_l490_490899


namespace k_values_possible_ranges_l490_490332

variable (r1 r2 : ℝ) (d : ℝ)
def k_values (r1 r2 d : ℝ) : ℕ :=
  if d = 0 then 0
  else if d = abs (r1 - r2) then 1
  else if abs (r1 - r2) < d ∧ d < (r1 + r2) then 2
  else if d = (r1 + r2) then 3
  else 4

theorem k_values_possible_ranges
  (r1 r2 : ℝ) (r1_pos : 0 < r1) (r2_pos : 0 < r2)
  (possible_ks : 0 ≤ k_values r1 r2 d ∧ k_values r1 r2 d ≤ 4)
  : 
  r1 = 4 → r2 = 5 → 
  {k : ℕ | ∃ d, k = k_values 4 5 d}.Finite ∧
  ({k : ℕ | ∃ d, k = k_values 4 5 d}.toFinset.card = 5) := 
by
  sorry

end k_values_possible_ranges_l490_490332


namespace return_to_freezer_probability_l490_490886

theorem return_to_freezer_probability :
  let cherry := 4
  let orange := 3
  let lemon_lime := 4
  let total := cherry + orange + lemon_lime
  (1 - (cherry / total * (cherry - 1) / (total - 1) 
     + orange / total * (orange - 1) / (total - 1)
     + lemon_lime / total * (lemon_lime - 1) / (total - 1)) : ℚ) = 8/11 :=
by
  let cherry := 4
  let orange := 3
  let lemon_lime := 4
  let total := cherry + orange + lemon_lime
  sorry

end return_to_freezer_probability_l490_490886


namespace find_number_l490_490193

theorem find_number : ∃ (x : ℤ), 45 + 3 * x = 72 ∧ x = 9 := by
  sorry

end find_number_l490_490193


namespace joan_gave_apples_l490_490669

theorem joan_gave_apples (initial_apples : ℕ) (remaining_apples : ℕ) (given_apples : ℕ) 
  (h1 : initial_apples = 43) (h2 : remaining_apples = 16) : given_apples = 27 :=
by
  -- Show that given_apples is obtained by subtracting remaining_apples from initial_apples
  sorry

end joan_gave_apples_l490_490669


namespace ellipse_and_tangent_proof_l490_490557

noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) := 
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def circle_equation (r : ℝ) (x y : ℝ) := 
  x^2 + y^2 = r

theorem ellipse_and_tangent_proof :
  (∃ a b : ℝ, a > b ∧ b > 0 ∧ 
              ∃ M : ℝ × ℝ, M = (real.sqrt 6, 1) ∧ 
              ∃ e : ℝ, e = real.sqrt 2 / 2 ∧ 
              ellipse_equation a b (real.sqrt 6) 1 ∧
              ∃ x y : ℝ, ellipse_equation a b x y ∧ 
                (∃ A B, (A = (x₁, y₁) ∧ B = (x₂, y₂) ∧ (x₁ * x₂ + y₁ * y₂ = 0)) → 
                ∃ d, d = abs m / real.sqrt (1 + k^2) ∧ d^2 = 8 / 3 ∧ 
                ∃ r, r = 8 / 3 ∧ d = r → 
                line_tangent_to_circle k m a b r)) :=
begin
  sorry
end

end ellipse_and_tangent_proof_l490_490557


namespace martha_final_cards_l490_490708

theorem martha_final_cards :
  (let initial_cards := 423 in
   let cards_from_emily := 3 * initial_cards in
   let total_cards := initial_cards + cards_from_emily in
   let given_away_cards := 213 in
   total_cards - given_away_cards = 1479) :=
by
  let initial_cards := 423
  let cards_from_emily := 3 * initial_cards
  let total_cards := initial_cards + cards_from_emily
  let given_away_cards := 213
  show total_cards - given_away_cards = 1479
  from sorry

end martha_final_cards_l490_490708


namespace sum_of_solutions_eq_9_l490_490822

theorem sum_of_solutions_eq_9 :
  let roots := {x : ℝ | x^2 = 9 * x - 20}
  in ∑ x in roots, x = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490822


namespace sum_of_solutions_l490_490848

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | x^2 = 9*x - 20}, x) = 9 := 
sorry

end sum_of_solutions_l490_490848


namespace number_of_routes_from_A_to_B_l490_490464

-- Define the points and paths
inductive Point
| A | B | Hex1 | Hex2 | Hex3 | Hex4 deriving DecidableEq, Repr

-- Define the directed edges
inductive Edge
| AB | Hex1Hex2 | Hex2Hex3 | Hex3Hex4 | BA | Hex4B

-- Define the path from A to B
def paths : List Edge → Bool
| [] => False
| (Edge.AB :: t)    => paths t
| _ => paths t

-- State that there are exactly 10 distinct routes from A to B
theorem number_of_routes_from_A_to_B :
  ∃ (paths: List Edge), (paths.filter paths).length = 10 :=
sorry

end number_of_routes_from_A_to_B_l490_490464


namespace loss_percentage_l490_490097

theorem loss_percentage (C S : ℕ) (H1 : C = 750) (H2 : S = 600) : (C - S) * 100 / C = 20 := by
  sorry

end loss_percentage_l490_490097


namespace quadratic_root_relationship_l490_490021

noncomputable def roots_of_quadratic (a b c: ℚ) (h_nonzero: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h_root_relation: ∀ (s₁ s₂ : ℚ), s₁ + s₂ = -c ∧ s₁ * s₂ = a → (3 * s₁) + (3 * s₂) = -a ∧ (3 * s₁) * (3 * s₂) = b) : Prop :=
  b / c = 27

theorem quadratic_root_relationship (a b c : ℚ) (h_nonzero: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_root_relation: ∀ (s₁ s₂ : ℚ), s₁ + s₂ = -c ∧ s₁ * s₂ = a → (3 * s₁) + (3 * s₂) = -a ∧ (3 * s₁) * (3 * s₂) = b) : 
  roots_of_quadratic a b c h_nonzero h_root_relation := 
by 
  sorry

end quadratic_root_relationship_l490_490021


namespace problem_statement_l490_490564

-- Definitions of the given conditions
variables {ℝ : Type*} [inner_product_space ℝ (euclidean_space ℝ (fin 3))]
variables (m n : euclidean_space ℝ (fin 3))
variables (α β γ : euclidean_space ℝ (fin 3))

-- m and n are direction vectors of lines m and n
-- α, β, γ are normal vectors of planes α, β, γ
variable (overline_m : euclidean_space ℝ (fin 3))
variable (overline_alpha overline_beta overline_gamma: euclidean_space ℝ (fin 3))

-- Condition: m is parallel to α and m is parallel to β implies α is parallel to β
theorem problem_statement
  (m_para_alpha : overline_m ∥ overline_alpha)
  (m_para_beta : overline_m ∥ overline_beta) : overline_alpha ∥ overline_beta :=
sorry

end problem_statement_l490_490564


namespace cylinder_cut_volume_l490_490528

variables {R h : ℝ}

-- Lean 4 proof statement
theorem cylinder_cut_volume (HR : 0 < R) (Hh : 0 < h) :
  volume (cylinder_cut R h) = (2 / 3) * h * R^2 :=
sorry

end cylinder_cut_volume_l490_490528


namespace drivers_sufficient_l490_490922

theorem drivers_sufficient
  (round_trip_duration : ℕ := 320)
  (rest_duration : ℕ := 60)
  (return_time_A : ℕ := 12 * 60 + 40)
  (depart_time_D : ℕ := 13 * 60 + 5)
  (next_depart_time_A : ℕ := 13 * 60 + 40)
  : (4 : ℕ) ∧ (21 * 60 + 30 = 1290) := 
  sorry

end drivers_sufficient_l490_490922


namespace eval_power_l490_490168

theorem eval_power {a m n : ℕ} : (a^m)^n = a^(m * n) := by
  sorry

example : (3^2)^4 = 6561 := by
  rw eval_power
  norm_num

end eval_power_l490_490168


namespace intersection_point_of_diagonals_l490_490379

noncomputable def intersection_of_diagonals (k m b : Real) : Real × Real :=
  let A := (0, b)
  let B := (0, -b)
  let C := (2 * b / (k - m), 2 * b * k / (k - m) - b)
  let D := (-2 * b / (k - m), -2 * b * k / (k - m) + b)
  (0, 0)

theorem intersection_point_of_diagonals (k m b : Real) :
  intersection_of_diagonals k m b = (0, 0) :=
sorry

end intersection_point_of_diagonals_l490_490379


namespace molecular_weight_correct_l490_490905

namespace MolecularWeight

-- Define the atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_Cl : ℝ := 35.45

-- Define the number of each atom in the compound
def n_N : ℝ := 1
def n_H : ℝ := 4
def n_Cl : ℝ := 1

-- Calculate the molecular weight of the compound
def molecular_weight : ℝ := (n_N * atomic_weight_N) + (n_H * atomic_weight_H) + (n_Cl * atomic_weight_Cl)

theorem molecular_weight_correct : molecular_weight = 53.50 := by
  -- Proof is omitted
  sorry

end MolecularWeight

end molecular_weight_correct_l490_490905


namespace smaller_prime_is_x_l490_490755

theorem smaller_prime_is_x (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (h1 : x + y = 36) (h2 : 4 * x + y = 87) : x = 17 :=
  sorry

end smaller_prime_is_x_l490_490755


namespace isosceles_triangle_angle_l490_490074

-- Definitions of the points and triangle
variables (A B C P Q : Type) [point A] [point B] [point C] [point P] [point Q]
variable [triangle ABC : A ∈ triangle B C]

-- Conditions
variable (isosceles_triangle : AB = BC)
variable (on_line_AP_Q : P ∈ line A C)
variable (on_line_PQ_QC : Q ∈ line P C)
variable (distance_relation : distance AP = 2 * distance PQ ∧ distance PQ = 2 * distance QC)

-- Definition of the angles
variable (angle_B : degrees 60)

-- Theorem to be proved
theorem isosceles_triangle_angle (isosceles_triangle : AB = BC)
  (on_line_AP_Q : P ∈ line A C)
  (on_line_PQ_QC : Q ∈ line P C)
  (distance_relation : distance AP = 2 * distance PQ ∧ distance PQ = 2 * distance QC) : 
  degrees angle B = 60 := 
sorry -- Proof to be provided

end isosceles_triangle_angle_l490_490074


namespace oblique_asymptote_of_rational_function_l490_490781

open_locale real

noncomputable def oblique_asymptote (f : ℝ → ℝ) : (ℝ → ℝ) := 
  λ x, x + 4/3

theorem oblique_asymptote_of_rational_function :
  let f := (λ x : ℝ, (3 * x^2 + 8 * x + 12) / (3 * x + 4)) in
  ∃ (a : ℝ → ℝ), a = oblique_asymptote f :=
begin
  let f := (λ x : ℝ, (3 * x^2 + 8 * x + 12) / (3 * x + 4)),
  use (λ x : ℝ, x + 4/3),
  sorry
end

end oblique_asymptote_of_rational_function_l490_490781


namespace remaining_gnomes_total_l490_490742

/--
The remaining number of gnomes in the three forests after the owner takes his specified percentages.
-/
theorem remaining_gnomes_total :
  let westerville_gnomes := 20
  let ravenswood_gnomes := 4 * westerville_gnomes
  let greenwood_grove_gnomes := ravenswood_gnomes + (25 * ravenswood_gnomes) / 100
  let remaining_ravenswood := ravenswood_gnomes - (40 * ravenswood_gnomes) / 100
  let remaining_westerville := westerville_gnomes - (30 * westerville_gnomes) / 100
  let remaining_greenwood_grove := greenwood_grove_gnomes - (50 * greenwood_grove_gnomes) / 100
  remaining_ravenswood + remaining_westerville + remaining_greenwood_grove = 112 := by
  sorry

end remaining_gnomes_total_l490_490742


namespace find_divisor_l490_490887

def dividend := 23
def quotient := 4
def remainder := 3

theorem find_divisor (d : ℕ) (h : dividend = (d * quotient) + remainder) : d = 5 :=
by {
  sorry
}

end find_divisor_l490_490887


namespace pentagon_area_l490_490456

variable (a b c d e : ℕ)
variable (r s : ℕ)

-- Given conditions
axiom H₁: a = 14
axiom H₂: b = 35
axiom H₃: c = 42
axiom H₄: d = 14
axiom H₅: e = 35
axiom H₆: r = 21
axiom H₇: s = 28
axiom H₈: r^2 + s^2 = e^2

-- Question: Prove that the area of the pentagon is 1176
theorem pentagon_area : b * c - (1 / 2) * r * s = 1176 := 
by 
  sorry

end pentagon_area_l490_490456


namespace max_probability_dice_difference_l490_490869

theorem max_probability_dice_difference : 
  ∃ p : ℚ, 
    (∀ (d : ℤ), d ∈ {-2, -1, 0, 1, 2} → 
      (∃ n : ℕ, ∃ m : ℕ, n ≤ m ∧ p = n / m ∧ 
      (∃ outcomes : set (ℕ × ℕ), set.finite outcomes ∧ 
       p = outcomes.card / 36 ∧ ∀ (x : ℕ × ℕ), x ∈ outcomes ↔ d = x.fst - x.snd))) ∧
    (∀ q : ℚ, q ∈ {p | ∃ (d : ℤ), d ∈ {-2, -1, 0, 1, 2} ∧ 
      ∃ n : ℕ, ∃ m : ℕ, n ≤ m ∧ q = n / m ∧ 
      (∃ outcomes : set (ℕ × ℕ), set.finite outcomes ∧ 
       q = outcomes.card / 36 ∧ ∀ (x : ℕ × ℕ), x ∈ outcomes ↔ d = x.fst - x.snd)} → q ≤ 1 / 6) :=
begin
  sorry
end

end max_probability_dice_difference_l490_490869


namespace cyclic_quadrilateral_implies_cyclic_k_l_m_n_l490_490416

theorem cyclic_quadrilateral_implies_cyclic_k_l_m_n 
  (A B C D K L M N : Point)
  (h_cyclic : CyclicQuadrilateral A B C D)
  (h_neq : A ≠ C)
  (h_rhombus_AKDL : Rhombus A K D L)
  (h_rhombus_CMBN : Rhombus C M B N)
  (h_equal_sides : side_length A K = side_length C M) :
  CyclicQuadrilateral K L M N :=
sorry

end cyclic_quadrilateral_implies_cyclic_k_l_m_n_l490_490416


namespace remainder_div_P_by_D_plus_D_l490_490188

theorem remainder_div_P_by_D_plus_D' 
  (P Q D R D' Q' R' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R') :
  P % (D + D') = R :=
by
  -- Proof is not required.
  sorry

end remainder_div_P_by_D_plus_D_l490_490188


namespace sum_of_solutions_l490_490835

theorem sum_of_solutions (x : ℝ) : 
  (∀ x : ℝ, x^2 = 9*x - 20 → x = 4 ∨ x = 5) → (4 + 5 = 9) :=
by
  intros h
  calc 4 + 5 = 9 : by norm_num
  sorry

end sum_of_solutions_l490_490835


namespace find_reflected_ray_equation_l490_490930

-- Define points A and B
def A : (ℝ × ℝ) := (-1/2, 0)
def B : (ℝ × ℝ) := (0, 1)
def A' : (ℝ × ℝ) := (1/2, 0)  -- Symmetric point A' of A with respect to the y-axis

-- Define the target equation of the line
def line_reflected (x y : ℝ) := 2 * x + y - 1 = 0

-- The theorem to be proved
theorem find_reflected_ray_equation (x y : ℝ) :
  (A = (-1/2, 0)) →
  (B = (0, 1)) →
  (A' = (1/2, 0)) →
  (2 * x + y - 1 = 0) :=
by
  intros hA hB hA'
  sorry

end find_reflected_ray_equation_l490_490930


namespace number_of_French_speaking_students_who_do_not_speak_English_l490_490094

variable (T : ℕ)
variable (F : ℕ)
variable (FE : ℕ)
variable (F_not_E : ℕ)

axiom T_def : T = 200
axiom percentage_not_French : 60% T = (3 * T) / 5
axiom percentage_French : 40% T = (2 * T) / 5
axiom FE_def : FE = 20
axiom F_def : F = 40% T

theorem number_of_French_speaking_students_who_do_not_speak_English : 
  F - FE = 60 :=
by
  rw [F_def, FE_def]
  sorry

end number_of_French_speaking_students_who_do_not_speak_English_l490_490094


namespace income_to_expenditure_ratio_l490_490753

-- Define the constants based on the conditions in step a)
def income : ℕ := 36000
def savings : ℕ := 4000

-- Define the expenditure as a function of income and savings
def expenditure (I S : ℕ) : ℕ := I - S

-- Define the ratio of two natural numbers
def ratio (a b : ℕ) : ℚ := a / b

-- Statement to be proved
theorem income_to_expenditure_ratio : 
  ratio income (expenditure income savings) = 9 / 8 :=
by
  sorry

end income_to_expenditure_ratio_l490_490753


namespace determine_pairs_l490_490499

theorem determine_pairs (a b : ℕ) (h : 2017^a = b^6 - 32 * b + 1) : 
  (a = 0 ∧ b = 0) ∨ (a = 0 ∧ b = 2) :=
by sorry

end determine_pairs_l490_490499


namespace sum_of_roots_quadratic_eq_l490_490846

theorem sum_of_roots_quadratic_eq :
  (∑ x in Finset.filter (λ x, x^2 = 9 * x - 20) (Finset.range 100), x) = 9 :=
begin
  sorry
end

end sum_of_roots_quadratic_eq_l490_490846


namespace least_multiple_of_24_gt_500_l490_490050

theorem least_multiple_of_24_gt_500 : ∃ x : ℕ, (x % 24 = 0) ∧ (x > 500) ∧ (∀ y : ℕ, (y % 24 = 0) ∧ (y > 500) → y ≥ x) ∧ (x = 504) := by
  sorry

end least_multiple_of_24_gt_500_l490_490050


namespace find_mnp_l490_490486

-- Definitions of given paths and conditions
def initial_area : ℝ := (ℝ.pi / 2) + 2
def turn_area : ℝ := (ℝ.pi / 4) + 1
def straight_area : ℝ := 2

-- Paths properties
def number_of_paths : ℕ := 2^20
def path_length : ℕ := 20
def expected_turns : ℕ := 19

-- Total area calculation
def total_area : ℝ := initial_area + (expected_turns * turn_area) + ((path_length - 1) * straight_area)

-- Average area
def average_area : ℝ := total_area / number_of_paths

-- Simplified form of average area
def simplified_average_area : ℝ := (236 + 21 * ℝ.pi) / 4

-- Problem statement: Prove that the average area matches the simplified form and find m + n + p
theorem find_mnp : average_area = simplified_average_area → 236 + 21 + 4 = 261 :=
by
  intro h
  rw [average_area, simplified_average_area] at h
  exact rfl

end find_mnp_l490_490486


namespace range_of_a_l490_490608

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 3 → x > Real.logBase 2 a) ↔ (0 < a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l490_490608


namespace magnitude_sum_vectors_l490_490567

variable (a b : ℝ^3) -- Assuming vectors in 3-dimensional real space

-- Given conditions
def norm_a : ℝ := 3
def norm_b : ℝ := 4
def dot_ab : ℝ := -2

-- Definition of norms in terms of ℝ^3 space
axiom norm_a_def : ‖a‖ = 3
axiom norm_b_def : ‖b‖ = 4
axiom dot_ab_def : inner a b = -2

-- Mathematical statement to prove
theorem magnitude_sum_vectors (a b : ℝ^3) (h1 : ‖a‖ = 3) (h2 : ‖b‖ = 4) (h3 : inner a b = -2) :
  ‖a + b‖ = Real.sqrt 21 := by
  sorry

end magnitude_sum_vectors_l490_490567


namespace initial_floors_l490_490121

-- Define the conditions given in the problem
def austin_time := 60 -- Time Austin takes in seconds to reach the ground floor
def jake_time := 90 -- Time Jake takes in seconds to reach the ground floor
def jake_steps_per_sec := 3 -- Jake descends 3 steps per second
def steps_per_floor := 30 -- There are 30 steps per floor

-- Define the total number of steps Jake descends
def total_jake_steps := jake_time * jake_steps_per_sec

-- Define the number of floors descended in terms of total steps and steps per floor
def num_floors := total_jake_steps / steps_per_floor

-- Theorem stating the number of floors is 9
theorem initial_floors : num_floors = 9 :=
by 
  -- Provide the basic proof structure
  sorry

end initial_floors_l490_490121


namespace product_of_roots_is_12_l490_490047

theorem product_of_roots_is_12 :
  (81 ^ (1 / 4) * 8 ^ (1 / 3) * 4 ^ (1 / 2)) = 12 := by
  sorry

end product_of_roots_is_12_l490_490047


namespace parabola_equation_l490_490626

def standard_equation_of_parabola (d : ℝ) (p : ℝ) : Prop :=
    d = 2 * p

theorem parabola_equation (d : ℝ) : d = sqrt 3 → standard_equation_of_parabola d (sqrt 3) :=
by
  intro hd
  rw hd
  sorry

end parabola_equation_l490_490626


namespace eccentricity_range_of_ellipse_l490_490555

noncomputable def ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  { e : ℝ // (e ∈ set.Ioo (real.sqrt 2 / 2) 1) } :=
sorry

theorem eccentricity_range_of_ellipse (a b x0 y0 : ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h3 : x0^2 / a^2 + y0^2 / b^2 = 1)
  (h4 : ∃ h : ℝ, h = -b^2 / a^2 ∧ (h > -1/2 ∧ h < 0)) :
  ∃ e : ℝ, e ∈ set.Ioo (real.sqrt 2 / 2) 1 := 
begin
  sorry
end

end eccentricity_range_of_ellipse_l490_490555


namespace adult_ticket_cost_l490_490036

theorem adult_ticket_cost (A Tc : ℝ) (T C : ℕ) (M : ℝ) 
  (hTc : Tc = 3.50) 
  (hT : T = 21) 
  (hC : C = 16) 
  (hM : M = 83.50) 
  (h_eq : 16 * Tc + (↑(T - C)) * A = M) : 
  A = 5.50 :=
by sorry

end adult_ticket_cost_l490_490036


namespace incenter_inequality_l490_490301

noncomputable def circumcircle_intersects (A B C I : Point) : Point × Point × Point := sorry
-- Assumption above that circumcircle intersects AI, BI, CI at A', B', C' respectively is abstracted as 'circumcircle_intersects'.

theorem incenter_inequality (ABC : Triangle) (I : Point) :
  (incenter ABC I) →
  let (A', B', C') := circumcircle_intersects ABC.A ABC.B ABC.C I in 
  (dist I ABC.A) * (dist I ABC.B) * (dist I ABC.C) ≤ (dist I A') * (dist I B') * (dist I C') :=
by
  sorry

end incenter_inequality_l490_490301


namespace meaningful_iff_x_ne_2_l490_490611

theorem meaningful_iff_x_ne_2 (x : ℝ) : (x ≠ 2) ↔ (∃ y : ℝ, y = (x - 3) / (x - 2)) := 
by
  sorry

end meaningful_iff_x_ne_2_l490_490611


namespace sum_of_solutions_l490_490793

-- Define the quadratic equation and variable x
def quadratic_equation := ∀ x : ℝ, (x^2 - 9 * x + 20 = 0)

-- Define what we need to prove
theorem sum_of_solutions : ∃ s : ℝ, (∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → s = x1 + x2) ∧ s = 9 :=
by
  sorry -- Proof is omitted

end sum_of_solutions_l490_490793


namespace polynomial_divisibility_result_l490_490998

theorem polynomial_divisibility_result :
  ∃ (A B : ℤ), (A = 1) ∧ (B = 1) ∧ (A + B = 2) ∧
  (∀ (x : ℂ), x^2 + x + 1 = 0 → x^103 + A * x^2 + B = 0) :=
by
  use 1, 1
  split
  rfl
  split
  rfl
  split
  norm_num
  intro x hx
  have h₁ : x^3 = 1,
  { calc x^3 = x * x^2 : by ring
         ... = x * (-x - 1) : by rw [hx]
         ... = -x^2 - x : by ring
         ... = -(-x - 1) - x : by rw [hx]
         ... = 1 : by ring }
  simp [h₁, hx]
  ring

end polynomial_divisibility_result_l490_490998


namespace imag_by_modulus_l490_490225

theorem imag_by_modulus (Z : ℂ) (h1 : Z.re = 1) (h2 : complex.abs Z = 2) : Z.im = √3 ∨ Z.im = -√3 :=
by
  sorry

end imag_by_modulus_l490_490225


namespace right_triangle_iff_area_eq_circles_relation_l490_490721

theorem right_triangle_iff_area_eq_circles_relation
  (r varrho d t : ℝ)
  (is_right_triangle : ∃ (A B C: Type) [Triangle A B C], ∠C = 90) :
  t = varrho^2 + r^2 - d^2 ↔ ∠C = 90 := sorry

end right_triangle_iff_area_eq_circles_relation_l490_490721


namespace sum_of_solutions_of_quadratic_eq_l490_490813

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 9 * x + 20 = 0

-- Prove that the sum of the solutions to this equation is 9
theorem sum_of_solutions_of_quadratic_eq : 
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 9) := 
begin
  -- Proof is omitted
  sorry
end

end sum_of_solutions_of_quadratic_eq_l490_490813


namespace strictly_increasing_arithmetic_seq_l490_490352

theorem strictly_increasing_arithmetic_seq 
  (s : ℕ → ℕ) 
  (hs_incr : ∀ n, s n < s (n + 1)) 
  (hs_seq1 : ∃ D1, ∀ n, s (s n) = s (s 0) + n * D1) 
  (hs_seq2 : ∃ D2, ∀ n, s (s n + 1) = s (s 0 + 1) + n * D2) : 
  ∃ d, ∀ n, s (n + 1) = s n + d :=
sorry

end strictly_increasing_arithmetic_seq_l490_490352


namespace area_ratio_ADGJ_dodecagon_l490_490342

variable (A B C D E F G H I J K L : Type)
variable [RegularDodecagon A B C D E F G H I J K L]

def area_dodecagon : ℝ := sorry
def area_rhombus_ADGJ : ℝ := sorry

theorem area_ratio_ADGJ_dodecagon (n m : ℝ) 
  (h₁ : area_dodecagon = n) 
  (h₂ : area_rhombus_ADGJ = m) :
  m / n = Real.sqrt 3 / 6 :=
  sorry

end area_ratio_ADGJ_dodecagon_l490_490342


namespace quadratic_equation_formulation_l490_490536

theorem quadratic_equation_formulation (a b c : ℝ) (x₁ x₂ : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a * x₁^2 + b * x₁ + c = 0)
  (h₃ : a * x₂^2 + b * x₂ + c = 0)
  (h₄ : x₁ + x₂ = -b / a)
  (h₅ : x₁ * x₂ = c / a) :
  ∃ (y : ℝ), a^2 * y^2 + a * (b - c) * y - b * c = 0 :=
by
  sorry

end quadratic_equation_formulation_l490_490536


namespace max_ab_cd_value_l490_490375

theorem max_ab_cd_value :
  ∃ (a b c d : ℕ), 
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) ∧ 
  ({a, b, c, d} = {2, 3, 5, 7}) ∧ 
  a + b + c + d = 17 ∧ 
  (a + b) * (c + d) = 72 :=
by
  sorry

end max_ab_cd_value_l490_490375


namespace identify_quadratic_equation_l490_490873

def is_quadratic_one_variable (eq : Type) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ eq = fun x : ℝ => a * x^2 + b * x + c = 0

theorem identify_quadratic_equation (E1 E2 E3 E4 : Type) :
  E1 = (λ x y : ℝ, x^2 - 4 * y = 0) →
  E2 = (λ x : ℝ, x^2 + x + 3 = 0) →
  E3 = (λ x : ℝ, 2 * x = 5) →
  E4 = (λ x : ℝ, x^2 + x⁻¹ - 2 = 0) →
  is_quadratic_one_variable E2 :=
begin
  sorry
end

end identify_quadratic_equation_l490_490873


namespace max_length_connected_sequence_yellow_l490_490713

def board : Type := fin 100 × fin 300

def first_player_color (c : board) : Prop := sorry -- Indicates if a cell is yellow
def second_player_color (c : board) : Prop := sorry -- Indicates if a cell is blue

def connected_sequence (seq : list board) : Prop :=
  ∀ i < seq.length - 1, (seq.nth i = some c1 ∧ seq.nth (i + 1) = some c2 → 
  (|c1.1.val - c2.1.val| + |c1.2.val - c2.2.val| = 1)) -- Sequence of cells where each pair of consecutive cells shares a side

def first_player_result (seq : list board) : Prop :=
  connected_sequence seq ∧ ∀ c ∈ seq, first_player_color c

theorem max_length_connected_sequence_yellow :
  ∀ seq, first_player_result seq → seq.length ≤ 200 :=
sorry

end max_length_connected_sequence_yellow_l490_490713


namespace g_at_3_l490_490592

def g (x : ℝ) : ℝ := -3 * x^4 + 4 * x^3 - 7 * x^2 + 5 * x - 2

theorem g_at_3 : g 3 = -185 := by
  sorry

end g_at_3_l490_490592


namespace arcsin_sin_eq_one_third_l490_490253

theorem arcsin_sin_eq_one_third (x : ℝ) (h1 : sin x = 1 / 3) (h2 : x ∈ Icc (-π / 2) (π / 2)) : 
  x = arcsin (1 / 3) := 
by
  sorry

end arcsin_sin_eq_one_third_l490_490253


namespace eval_log2_3_l490_490196

noncomputable def f : ℝ → ℝ 
| x := if h : x < 4 then f (x + 1) else 2^x

theorem eval_log2_3 : f (Real.log 3 / Real.log 2) = 24 :=
by
  sorry

end eval_log2_3_l490_490196


namespace cone_height_l490_490417

theorem cone_height
  (V1 V2 V : ℝ)
  (h1 h2 : ℝ)
  (fact1 : h1 = 10)
  (fact2 : h2 = 2)
  (h : ∀ m : ℝ, V1 = V * (10 ^ 3) / (m ^ 3) ∧ V2 = V * ((m - 2) ^ 3) / (m ^ 3))
  (equal_volumes : V1 + V2 = V) :
  (∃ m : ℝ, m = 13.897) :=
by
  sorry

end cone_height_l490_490417


namespace find_initial_time_l490_490081

-- The initial distance d
def distance : ℕ := 288

-- Conditions
def initial_condition (v t : ℕ) : Prop :=
  distance = v * t

def new_condition (t : ℕ) : Prop :=
  distance = 32 * (3 * t / 2)

-- Proof Problem Statement
theorem find_initial_time (v t : ℕ) (h1 : initial_condition v t)
  (h2 : new_condition t) : t = 6 := by
  sorry

end find_initial_time_l490_490081


namespace cupcakes_per_day_l490_490129

theorem cupcakes_per_day (goal cupcakes_to_Bonnie days : ℕ) (h1 : goal = 96) (h2 : cupcakes_to_Bonnie = 24) (h3 : days = 2) : 
  (goal + cupcakes_to_Bonnie) / days = 60 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end cupcakes_per_day_l490_490129


namespace max_d_for_kappa_labeling_l490_490712

-- Define the conditions as a Lean predicate
def canLabelAtLeast (d : ℝ) (n : ℕ) : Prop :=
  ∃ labelGrid : ℕ → ℕ → Prop,
  (∀ i j, labelGrid i j → (0 ≤ i ∧ i < n) ∧ (0 ≤ j ∧ j < n)) ∧
  (∀ i j, labelGrid i j → ¬ (
    (i + 1 < n ∧ labelGrid (i + 1) j) ∧ 
    (i + 2 < n ∧ labelGrid (i + 2) j)
  ) ∧ ¬ (
    (j + 1 < n ∧ labelGrid i (j + 1)) ∧ 
    (j + 2 < n ∧ labelGrid i (j + 2))
  ) ∧ ¬ (
    (i + 1 < n ∧ j + 1 < n ∧ labelGrid (i + 1) (j + 1)) ∧ 
    (i + 2 < n ∧ j + 2 < n ∧ labelGrid (i + 2) (j + 2))
  )) ∧
  (∃ count kappaCount, kappaCount = n * n * d ∧
   (∀ i j, labelGrid i j → count += 1)) ∧
  count ≥ d * n²

-- The theorem to prove
theorem max_d_for_kappa_labeling : ∀ (d : ℝ), d > 0 → d ≤ 1 / 2 := by
  assume d hd
  sorry

end max_d_for_kappa_labeling_l490_490712


namespace percentage_correct_l490_490055

noncomputable def part : ℝ := 172.8
noncomputable def whole : ℝ := 450.0
noncomputable def percentage (part whole : ℝ) := (part / whole) * 100

theorem percentage_correct : percentage part whole = 38.4 := by
  sorry

end percentage_correct_l490_490055


namespace limit_na_n_l490_490498

noncomputable def L (x : ℝ) : ℝ := x - x^2 / 2

noncomputable def a_n (n : ℕ) : ℝ :=
(λ (L : ℝ → ℝ) (x : ℝ), Nat.iterate (λ y, L y) n x) L (19 / n)

theorem limit_na_n : 
  tendsto (λ n : ℕ, n * a_n n) at_top (𝓝 (38 / 21)) :=
sorry

end limit_na_n_l490_490498


namespace factor_81_sub_27x3_l490_490974

theorem factor_81_sub_27x3 (x : ℝ) : 81 - 27 * x^3 = 3 * (3 - x) * (81 + 27 * x + 9 * x^2) :=
sorry

end factor_81_sub_27x3_l490_490974


namespace sum_of_solutions_l490_490853

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | x^2 = 9*x - 20}, x) = 9 := 
sorry

end sum_of_solutions_l490_490853


namespace sum_of_digits_of_a_l490_490298

theorem sum_of_digits_of_a
  (a : ℝ)
  (h : a = ∑ i in finset.range 101, (10^(i + 2) - 1) * 0.9) :
  ∑ d in a.digits, d = 891 :=
by
  sorry

end sum_of_digits_of_a_l490_490298


namespace calculation_correct_l490_490480

theorem calculation_correct : 200 * 19.9 * 1.99 * 100 = 791620 := by
  sorry

end calculation_correct_l490_490480


namespace proof1_proof2_proof3_l490_490243

noncomputable def question1 (m : ℝ) (h : m > 0) : Prop :=
  let x_A := 2^(-m)
  let x_B := 2^m
  let x_C := 8^(-3/(m+1))
  let x_D := 8^(3/(m+1))
  x_A * x_B = x_C * x_D

noncomputable def question2 (m : ℝ) (h : m > 0) (ha : ℝ) (hb : ℝ) : Prop :=
  let a := |2^(-m) - 8^(-3/(m+1))|
  let b := |2^m - 8^(3/(m+1))|
  a = b → m = (-1 + Real.sqrt 37) / 2

noncomputable def question3 (m : ℝ) (h : m > 0) (a_ne_0 : ℝ) : Prop :=
  let a := |2^(-m) - 8^(-3/(m+1))|
  let b := |2^m - 8^(3/(m+1))|
  let f := (b / a)
  f = 2^(m + 9/(m+1)) ∧ (∀ x, f ≥ 32) ∧ f = 32 at (m = 2)

theorem proof1 (m : ℝ) (h : m > 0) : question1 m h := 
sorry

theorem proof2 (m : ℝ) (h : m > 0) (ha : ℝ) (hb : ℝ) : question2 m h ha hb :=
sorry

theorem proof3 (m : ℝ) (h : m > 0) (a_ne_0 : ℝ) : question3 m h a_ne_0 :=
sorry

end proof1_proof2_proof3_l490_490243


namespace stock_price_decrease_in_may_l490_490477

theorem stock_price_decrease_in_may (S₀ : ℝ) (h : 
  let S₁ := S₀ * 0.85 in
  let S₂ := S₁ * 1.10 in
  let S₃ := S₂ * 1.30 in
  let S₄ := S₃ * 0.80 in
  let S₅ := S₄ * (1 - y / 100) in
  S₅ = S₀ 
) : y = 3 :=
sorry

end stock_price_decrease_in_may_l490_490477


namespace solve_log_eq_l490_490761

theorem solve_log_eq (x : ℝ) : log 2 x = -1 / 2 → x = real.sqrt 2 / 2 :=
by
  -- Note: Proof steps are omitted
  sorry

end solve_log_eq_l490_490761


namespace sin_neg_19pi_over_6_eq_one_half_l490_490162

theorem sin_neg_19pi_over_6_eq_one_half :
  ∀ (x : ℝ), (sin (-x)) = -sin x ∧ (sin (x + 2 * π)) = sin x ∧ (sin (π / 6) = 1 / 2) → sin (-19 * π / 6) = 1 / 2 :=
by 
  intros x hx
  sorry

end sin_neg_19pi_over_6_eq_one_half_l490_490162


namespace average_birds_per_site_l490_490674

theorem average_birds_per_site :
  let birds_seen (day_sites day_avg : ℕ) := day_sites * day_avg in
  let total_birds := birds_seen 5 7 + birds_seen 5 5 + birds_seen 10 8 in
  let total_sites := 5 + 5 + 10 in
  total_birds / total_sites = 7 :=
by
  sorry

end average_birds_per_site_l490_490674


namespace modulus_of_complex_l490_490216

noncomputable def imaginary_unit : ℂ := complex.I
noncomputable def sqrt_two : ℝ := real.sqrt 2

def condition (a : ℝ) : Prop :=
  let z := (a - sqrt_two) + imaginary_unit in
  (z / imaginary_unit).im = 0

theorem modulus_of_complex (a : ℝ) (ha : condition a) :
  complex.abs (2 * a + sqrt_two * imaginary_unit) = real.sqrt 10 :=
by
  sorry

end modulus_of_complex_l490_490216


namespace find_white_balls_l490_490278

-- Define a structure to hold the probabilities and total balls
structure BallProperties where
  totalBalls : Nat
  probRed : Real
  probBlack : Real

-- Given data as conditions
def givenData : BallProperties := 
  { totalBalls := 50, probRed := 0.15, probBlack := 0.45 }

-- The statement to prove the number of white balls
theorem find_white_balls (data : BallProperties) : 
  data.totalBalls = 50 →
  data.probRed = 0.15 →
  data.probBlack = 0.45 →
  ∃ whiteBalls : Nat, whiteBalls = 20 :=
by
  sorry

end find_white_balls_l490_490278


namespace cosine_third_quadrant_l490_490623

theorem cosine_third_quadrant 
  (B : ℝ) 
  (sin_B : ℝ) 
  (h1 : sin_B = 5 / 13)
  (h2 : π < B ∧ B < 3 * π / 2) :
  cos B = -12 / 13 :=
begin
  sorry
end

end cosine_third_quadrant_l490_490623


namespace simplify_expression_l490_490870

theorem simplify_expression :
  (Real.sqrt 2 * 2 ^ (1 / 2 : ℝ) + 18 / 3 * 3 - 8 ^ (3 / 2 : ℝ)) = (20 - 16 * Real.sqrt 2) :=
by sorry

end simplify_expression_l490_490870


namespace sum_of_two_digit_factors_l490_490161

theorem sum_of_two_digit_factors (a b : ℕ) (h : a * b = 5681) (h1 : 10 ≤ a) (h2 : a < 100) (h3 : 10 ≤ b) (h4 : b < 100) : a + b = 154 :=
by
  sorry

end sum_of_two_digit_factors_l490_490161


namespace math_problem_solution_l490_490039

noncomputable def problem_statement : Prop :=
  ∀ (X Y Z W U V : Type)
  (XY YZ ZX XV : ℕ)
  (XY_eq : XY = 13)
  (YZ_eq : YZ = 30)
  (ZX_eq : ZX = 26)
  (bisector_intersects : ∀ W, ∃ U, U ≠ X ∧ ∃ V, ∃ circum ∈ ∂ (triangle Y W U), intersects circum XY Y ∧ V ≠ Y)
  , XV = 29

theorem math_problem_solution : problem_statement :=
by {
  intros,
  sorry
}

end math_problem_solution_l490_490039


namespace graph_transformation_l490_490768

noncomputable def transformation_correct : Prop :=
  ∀ x : ℝ, sin (2 * x - π / 3) + 1 = sin (2 * (x - π / 6)) + 1

theorem graph_transformation :
  transformation_correct :=
by
  sorry

end graph_transformation_l490_490768


namespace shortest_distance_exp_to_line_l490_490760

-- Define the exponential function
def exp (x : ℝ) : ℝ := Real.exp x

-- Define the line y = x
def line (p : ℝ × ℝ) : Prop := p.2 = p.1 

-- Define the distance formula from a point to a line
def distance_from_point_to_line (a b c: ℝ) (p: ℝ × ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)

-- Define the shortest distance problem
theorem shortest_distance_exp_to_line : 
  let points_on_exp := λ x : ℝ, (x, exp x)
  let line_eq := λ p, line p
  let distance := λ p, distance_from_point_to_line (-1) 1 0 p
  ∃ (P : ℝ), distance (points_on_exp P) = Real.sqrt 2 / 2 :=
begin
  sorry 
end

end shortest_distance_exp_to_line_l490_490760


namespace perimeter_of_triangle_l490_490087

theorem perimeter_of_triangle (side_length : ℕ) (num_sides : ℕ) (h1 : side_length = 7) (h2 : num_sides = 3) : 
  num_sides * side_length = 21 :=
by
  sorry

end perimeter_of_triangle_l490_490087


namespace necessary_but_not_sufficient_condition_for_lnot_q_l490_490207

variables {a x : ℝ}

def p : Prop := 2 * x^2 - 3 * x + 1 ≤ 0
def q : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem necessary_but_not_sufficient_condition_for_lnot_q (h : (¬p → ¬q) ∧ ¬(¬q → ¬p)) :
  0 ≤ a ∧ a ≤ 1 / 2 :=
sorry

end necessary_but_not_sufficient_condition_for_lnot_q_l490_490207


namespace factor_expr_l490_490979

theorem factor_expr (x : ℝ) : 81 - 27 * x^3 = 27 * (3 - x) * (9 + 3 * x + x^2) := 
sorry

end factor_expr_l490_490979


namespace cd_percentage_cheaper_l490_490079

theorem cd_percentage_cheaper (cost_cd cost_book cost_album difference percentage : ℝ) 
  (h1 : cost_book = cost_cd + 4)
  (h2 : cost_book = 18)
  (h3 : cost_album = 20)
  (h4 : difference = cost_album - cost_cd)
  (h5 : percentage = (difference / cost_album) * 100) : 
  percentage = 30 :=
sorry

end cd_percentage_cheaper_l490_490079


namespace simplified_trig_expression_l490_490731

theorem simplified_trig_expression :
  (sin 400 * sin (-230)) / (cos 850 * tan (-50)) = sin 40 :=
by sorry

end simplified_trig_expression_l490_490731


namespace smallest_positive_a_exists_l490_490526

-- Definition of the equation to be solved
def equation (a : ℝ) : Prop := 
  (5 * real.sqrt ((3 * a)^2 + 2^2) - 3 * a^2 - 2) / (real.sqrt (1 + 5 * a^2) + 4) = 1

-- The proof goal is to find the smallest positive 'a' such that 'equation' holds true
theorem smallest_positive_a_exists : ∃ a : ℝ, a > 0 ∧ equation a ∧ ∀ b : ℝ, (b > 0 ∧ equation b) → a ≤ b :=
  sorry

end smallest_positive_a_exists_l490_490526


namespace triangle_third_side_length_l490_490034

theorem triangle_third_side_length 
  (a b c : ℝ) (ha : a = 7) (hb : b = 11) (hc : c = 3) :
  (4 < c ∧ c < 18) → c ≠ 3 :=
by
  sorry

end triangle_third_side_length_l490_490034


namespace perp_locus_theorem_l490_490035

noncomputable def locus_of_perpendiculars (O A B C K L : Point) (alpha beta : ℝ) 
  (interior_angle_O : angle C O A <= α)
  (angle_beta : angle O C A = angle O C B = β)
  (perpendicular_OK : perpendicular O K (line C A))
  (perpendicular_OL : perpendicular O L (line C B))
  : Prop :=
if α <= β then
  ∃ M : Point, M ∈ arc K L ∧ angle K M L = α + β
else
  ∃ M : Point, M ∈ circle_through K L ∧ (angle K M L = α + β ∨ angle K M L = α - β)

theorem perp_locus_theorem (O A B C K L : Point) (alpha beta : ℝ)
  (interior_angle_O : angle C O A <= α)
  (angle_beta : angle O C A = angle O C B = β)
  (perpendicular_OK : perpendicular O K (line C A))
  (perpendicular_OL : perpendicular O L (line C B)) :
  locus_of_perpendiculars O A B C K L alpha beta interior_angle_O angle_beta perpendicular_OK perpendicular_OL := 
sorry

end perp_locus_theorem_l490_490035


namespace sum_sequence_2018_eq_2018_l490_490576

noncomputable def a_n : ℕ → ℝ
| n := 2 * n - 1

def b_n (n : ℕ) : ℝ := (a_n n) * cos (n * π)

def sum_first_n_terms (f : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum f

theorem sum_sequence_2018_eq_2018 (h₁ : a_n 3 + a_n 5 = a_n 4 + 7)
                                  (h₂ : a_n 10 = 19) :
  sum_first_n_terms b_n 2018 = 2018 :=
by
  sorry

end sum_sequence_2018_eq_2018_l490_490576


namespace constant_term_expansion_l490_490748

noncomputable def binom : ℕ → ℕ → ℕ
| n, k => if h : k ≤ n then Nat.choose n k else 0

theorem constant_term_expansion :
    ∀ x: ℂ, (x ≠ 0) → ∃ term: ℂ, 
    term = (-1 : ℂ) * binom 6 4 ∧ term = -15 := 
by
  intros x hx
  use (-1 : ℂ) * binom 6 4
  constructor
  · rfl
  · sorry

end constant_term_expansion_l490_490748


namespace lauren_change_l490_490679

-- Define the given conditions as Lean terms.
def price_meat_per_pound : ℝ := 3.5
def pounds_meat : ℝ := 2.0
def price_buns : ℝ := 1.5
def price_lettuce : ℝ := 1.0
def pounds_tomato : ℝ := 1.5
def price_tomato_per_pound : ℝ := 2.0
def price_pickles : ℝ := 2.5
def coupon_value : ℝ := 1.0
def amount_paid : ℝ := 20.0

-- Define the total cost of each item.
def cost_meat : ℝ := pounds_meat * price_meat_per_pound
def cost_tomato : ℝ := pounds_tomato * price_tomato_per_pound
def total_cost_before_coupon : ℝ := cost_meat + price_buns + price_lettuce + cost_tomato + price_pickles

-- Define the final total cost after applying the coupon.
def final_total_cost : ℝ := total_cost_before_coupon - coupon_value

-- Define the expected change.
def expected_change : ℝ := amount_paid - final_total_cost

-- Prove that the expected change is $6.00.
theorem lauren_change : expected_change = 6.0 := by
  sorry

end lauren_change_l490_490679


namespace sum_of_solutions_l490_490798

theorem sum_of_solutions (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) : 
  -b / a = 9 :=
by
  -- The proof is omitted here (hence the 'sorry')
  sorry

end sum_of_solutions_l490_490798


namespace trains_meet_in_32_seconds_l490_490439

noncomputable def train_meeting_time
  (length_train1 : ℕ)
  (length_train2 : ℕ)
  (initial_distance : ℕ)
  (speed_train1_kmph : ℕ)
  (speed_train2_kmph : ℕ)
  : ℕ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600
  let relative_speed := speed_train1_mps + speed_train2_mps
  let total_distance := length_train1 + length_train2 + initial_distance
  total_distance / relative_speed

theorem trains_meet_in_32_seconds :
  train_meeting_time 400 200 200 54 36 = 32 := 
by
  sorry

end trains_meet_in_32_seconds_l490_490439


namespace right_triangle_area_l490_490177

theorem right_triangle_area (a : ℝ) (h1 : a > 0) :
  let b := (2 / 3) * a,
      c := (∛13 / 3) * a,
      area := a * b / 2 in 
  area = 8 / 3 :=
by {
  sorry
}

end right_triangle_area_l490_490177


namespace plane_speed_east_l490_490105

def plane_travel_problem (v : ℕ) : Prop :=
  let time : ℕ := 35 / 10 
  let distance_east := v * time
  let distance_west := 275 * time
  let total_distance := distance_east + distance_west
  total_distance = 2100

theorem plane_speed_east : ∃ v : ℕ, plane_travel_problem v ∧ v = 325 :=
sorry

end plane_speed_east_l490_490105


namespace calculate_g5_at_2_l490_490254

noncomputable def g (x : ℝ) : ℝ := (2 - x) / (3 + 2 * x)

def g_iter (n : ℕ) : ℝ → ℝ
| 0 => id
| (n + 1) => g ∘ g_iter n

theorem calculate_g5_at_2 :
  g_iter 0 2 = 2 ∧
  g_iter 1 2 = 0 ∧
  g_iter 2 2 = 2 / 3 ∧
  g_iter 3 2 = 4 / 13 ∧
  g_iter 4 2 = 2 / 5 ∧
  g_iter 5 2 = 8 / 19 :=
by
  unfold g_iter
  unfold g
  -- Calculation by Lean required here
  sorry

end calculate_g5_at_2_l490_490254


namespace bricks_needed_to_build_wall_l490_490433

-- Define the dimensions of the brick
def brick_length := 25 -- in cm
def brick_width := 11.25 -- in cm
def brick_height := 6 -- in cm

-- Define the dimensions of the wall (converted into centimeters)
def wall_length := 750 -- in cm (7.5 m converted to cm)
def wall_height := 600 -- in cm (6 m converted to cm)
def wall_width := 22.5 -- in cm

-- Define the volumes
def V_wall := wall_length * wall_height * wall_width
def V_brick := brick_length * brick_width * brick_height

-- Definition of the problem in Lean
theorem bricks_needed_to_build_wall : V_wall / V_brick = 6000 := by
  sorry

end bricks_needed_to_build_wall_l490_490433


namespace configuration_permutations_l490_490505

theorem configuration_permutations : (13.factorial / (2.factorial * 2.factorial * 2.factorial * 2.factorial)) = 389188800 := by
  sorry

end configuration_permutations_l490_490505


namespace digits_needed_l490_490086

theorem digits_needed (n : ℕ) (h : n = 710) : 
  let digits_single = 9 * 1 in
  let digits_double = 90 * 2 in
  let digits_triple = 611 * 3 in
  digits_single + digits_double + digits_triple = 2022 :=
by {
  -- Total pages derivation based on n
  have h1 : digits_single = 9 * 1 := rfl,
  have h2 : digits_double = (99 - 10 + 1) * 2 := rfl,
  have h3 : digits_triple = (n - 100 + 1) * 3, by rw h,
  rw h,
  calc 
    9 * 1 + (99 - 10 + 1) * 2 + (710 - 100 + 1) * 3 
    = 9 + 180 + 1833 : by { rw [h1, h2, h3] }
    ... = 2022 : by norm_num
}

end digits_needed_l490_490086


namespace log_inequality_l490_490539

theorem log_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : x^2 + y = 0) : 
  Real.log a (a^x + a^y) ≤ Real.log 2 a + 1 / 8 := 
by
  sorry

end log_inequality_l490_490539


namespace Trisha_test_scores_l490_490022

theorem Trisha_test_scores (t1 t2 t3 t4 t5 : ℕ) 
  (h1 : t1 = 88) (h2 : t2 = 73) (h3 : t3 = 70) 
  (hmean : (t1 + t2 + t3 + t4 + t5) / 5 = 81)
  (hless : t1 < 90 ∧ t2 < 90 ∧ t3 < 90 ∧ t4 < 90 ∧ t5 < 90)
  (hdistinct : t1 ≠ t2 ∧ t1 ≠ t3 ∧ t1 ≠ t4 ∧ t1 ≠ t5 ∧ 
               t2 ≠ t3 ∧ t2 ≠ t4 ∧ t2 ≠ t5 ∧ 
               t3 ≠ t4 ∧ t3 ≠ t5 ∧ 
               t4 ≠ t5) :
  {t1, t2, t3, t4, t5}.to_list.sorted = [89, 88, 85, 73, 70] := by
    sorry

end Trisha_test_scores_l490_490022


namespace apollo_total_cost_l490_490944

def hephaestus_first_half_months : ℕ := 6
def hephaestus_first_half_rate : ℕ := 3
def hephaestus_second_half_rate : ℕ := hephaestus_first_half_rate * 2

def athena_rate : ℕ := 5
def athena_months : ℕ := 12

def ares_first_period_months : ℕ := 9
def ares_first_period_rate : ℕ := 4
def ares_second_period_months : ℕ := 3
def ares_second_period_rate : ℕ := 6

def total_cost := hephaestus_first_half_months * hephaestus_first_half_rate
               + hephaestus_first_half_months * hephaestus_second_half_rate
               + athena_months * athena_rate
               + ares_first_period_months * ares_first_period_rate
               + ares_second_period_months * ares_second_period_rate

theorem apollo_total_cost : total_cost = 168 := by
  -- placeholder for the proof
  sorry

end apollo_total_cost_l490_490944


namespace quadratic_equation_with_one_variable_is_B_l490_490875

def is_quadratic_equation_with_one_variable (eq : String) : Prop :=
  eq = "x^2 + x + 3 = 0"

theorem quadratic_equation_with_one_variable_is_B :
  is_quadratic_equation_with_one_variable "x^2 + x + 3 = 0" :=
by
  sorry

end quadratic_equation_with_one_variable_is_B_l490_490875


namespace monica_study_ratio_l490_490325

theorem monica_study_ratio :
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let weekday_total := wednesday + thursday + friday
  let total := 22
  let weekend := total - weekday_total
  weekend = wednesday + thursday + friday :=
by
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let weekday_total := wednesday + thursday + friday
  let total := 22
  let weekend := total - weekday_total
  sorry

end monica_study_ratio_l490_490325


namespace sum_of_solutions_eq_9_l490_490803

theorem sum_of_solutions_eq_9 (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20) :
  let (sum_roots : ℝ) := -b / a in 
  sum_roots = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490803


namespace cheryl_used_total_material_correct_amount_l490_490131

def material_used (initial leftover : ℚ) : ℚ := initial - leftover

def total_material_used 
  (initial_a initial_b initial_c leftover_a leftover_b leftover_c : ℚ) : ℚ :=
  material_used initial_a leftover_a + material_used initial_b leftover_b + material_used initial_c leftover_c

theorem cheryl_used_total_material_correct_amount :
  total_material_used (2/9) (1/8) (3/10) (4/18) (1/12) (3/15) = 17/120 :=
by
  sorry

end cheryl_used_total_material_correct_amount_l490_490131


namespace find_x_lceil_l490_490983

theorem find_x_lceil (x : ℝ) (h : ⌈x⌉ * x = 186) : x ≈ 13.29 :=
by
  sorry

end find_x_lceil_l490_490983


namespace jack_shoes_time_l490_490294

theorem jack_shoes_time (J : ℝ) (h : J + 2 * (J + 3) = 18) : J = 4 :=
by
  sorry

end jack_shoes_time_l490_490294


namespace sum_of_roots_l490_490860

theorem sum_of_roots (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) :
  ∑ x in {x | a * x^2 + b * x + c = 0}, x = 9 :=
by
  sorry

end sum_of_roots_l490_490860


namespace convert_deg_to_rad_l490_490957

theorem convert_deg_to_rad (deg_to_rad : ℝ → ℝ) (conversion_factor : deg_to_rad 1 = π / 180) :
  deg_to_rad (-300) = - (5 * π) / 3 :=
by
  sorry

end convert_deg_to_rad_l490_490957


namespace trig_identity_l490_490612

theorem trig_identity (f : ℝ → ℝ) (x : ℝ) (h : f (Real.sin x) = 3 - Real.cos (2 * x)) : f (Real.cos x) = 3 + Real.cos (2 * x) :=
sorry

end trig_identity_l490_490612


namespace range_of_a_l490_490600

theorem range_of_a (M : Set ℝ) (a : ℝ) :
  (M = {x | x^2 - 4 * x + 4 * a < 0}) →
  ¬(2 ∈ M) →
  (1 ≤ a) :=
by
  -- Given assumptions
  intros hM h2_notin_M
  -- Convert h2_notin_M to an inequality and prove the desired result
  sorry

end range_of_a_l490_490600


namespace product_B_sampling_l490_490381

theorem product_B_sampling (a : ℕ) (h_seq : a > 0) :
  let A := a
  let B := 2 * a
  let C := 4 * a
  let total := A + B + C
  total = 7 * a →
  let total_drawn := 140
  B / total * total_drawn = 40 :=
by sorry

end product_B_sampling_l490_490381


namespace sum_of_roots_l490_490864

theorem sum_of_roots (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) :
  ∑ x in {x | a * x^2 + b * x + c = 0}, x = 9 :=
by
  sorry

end sum_of_roots_l490_490864


namespace volume_less_than_1000_l490_490881

noncomputable def volume (x : ℕ) : ℤ :=
(x + 3) * (x - 1) * (x^3 - 20)

theorem volume_less_than_1000 : ∃ (n : ℕ), n = 2 ∧ 
  ∃ x1 x2, x1 ≠ x2 ∧ 0 < x1 ∧ 
  0 < x2 ∧
  volume x1 < 1000 ∧
  volume x2 < 1000 ∧
  ∀ x, 0 < x → volume x < 1000 → (x = x1 ∨ x = x2) :=
by
  sorry

end volume_less_than_1000_l490_490881


namespace dish_heats_up_by_5_degrees_per_minute_l490_490667

theorem dish_heats_up_by_5_degrees_per_minute
  (final_temperature initial_temperature : ℕ)
  (time_taken : ℕ)
  (h1 : final_temperature = 100)
  (h2 : initial_temperature = 20)
  (h3 : time_taken = 16) :
  (final_temperature - initial_temperature) / time_taken = 5 :=
by
  sorry

end dish_heats_up_by_5_degrees_per_minute_l490_490667


namespace right_triangle_area_eq_8_over_3_l490_490174

-- Definitions arising from the conditions in the problem
variable (a b c : ℝ)

-- The conditions as Lean definitions
def condition1 : Prop := b = (2/3) * a
def condition2 : Prop := b = (2/3) * c

-- The question translated into a proof problem: proving that the area of the triangle equals 8/3
theorem right_triangle_area_eq_8_over_3 (h1 : condition1 a b) (h2 : condition2 b c) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 8/3 :=
by
  sorry

end right_triangle_area_eq_8_over_3_l490_490174


namespace cylindrical_coordinates_of_point_l490_490961
open Real

def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := sqrt (x * x + y * y)
  let θ := if y >= 0 then arccos (x / r) else 2 * π - arccos (x / r)
  (r, θ, z)

theorem cylindrical_coordinates_of_point :
  rectangular_to_cylindrical 3 (-3 * sqrt 3) 2 = (6, 5 * π / 3, 2) :=
by
  sorry

end cylindrical_coordinates_of_point_l490_490961


namespace smallest_positive_period_minimum_value_l490_490200

def f (x : ℝ) : ℝ := 2 * Real.sin (x + Real.pi / 2) + 3

theorem smallest_positive_period (x : ℝ) : ∃ T > 0, T = Real.pi * 2 ∧ ∀ x, f (x + T) = f x :=
by
  sorry

theorem minimum_value (x : ℝ) : ∀ x, f x ≥ 1 ∧ ∃ x, f x = 1 :=
by
  sorry

end smallest_positive_period_minimum_value_l490_490200


namespace cylindrical_coordinates_of_point_l490_490960
open Real

def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := sqrt (x * x + y * y)
  let θ := if y >= 0 then arccos (x / r) else 2 * π - arccos (x / r)
  (r, θ, z)

theorem cylindrical_coordinates_of_point :
  rectangular_to_cylindrical 3 (-3 * sqrt 3) 2 = (6, 5 * π / 3, 2) :=
by
  sorry

end cylindrical_coordinates_of_point_l490_490960


namespace sum_2_75_0_003_0_158_l490_490063

theorem sum_2_75_0_003_0_158 : 2.75 + 0.003 + 0.158 = 2.911 :=
by
  -- Lean proof goes here  
  sorry

end sum_2_75_0_003_0_158_l490_490063


namespace pencils_distributed_per_container_l490_490511

noncomputable def total_pencils (initial_pencils : ℕ) (additional_pencils : ℕ) : ℕ :=
  initial_pencils + additional_pencils

noncomputable def pencils_per_container (total_pencils : ℕ) (num_containers : ℕ) : ℕ :=
  total_pencils / num_containers

theorem pencils_distributed_per_container :
  let initial_pencils := 150
  let additional_pencils := 30
  let num_containers := 5
  let total := total_pencils initial_pencils additional_pencils
  let pencils_per_container := pencils_per_container total num_containers
  pencils_per_container = 36 :=
by {
  -- sorry is used to skip the proof
  -- the actual proof is not required
  sorry
}

end pencils_distributed_per_container_l490_490511


namespace range_of_alpha_l490_490407

noncomputable def parabola_condition (x y : ℝ) : Prop :=
  y^2 = 4 * x

noncomputable def ellipse_condition (x y : ℝ) : Prop :=
  3 * x^2 + 2 * y^2 = 2

noncomputable def chord_length_condition (α : ℝ) : Prop :=
  ∃ (x x1 x2 : ℝ), ((x - 1) * (tan α))^2 = 4 * x ∧ (x1 + 1 + x2 + 1) ≤ 8

theorem range_of_alpha (α : ℝ) :
  (∀ x y : ℝ, parabola_condition x y → chord_length_condition α → ellipse_condition x ((x - 1) * tan α)) →
  α ∈ (set.Icc (Real.pi/4) (Real.pi/3) ∪ set.Icc (2 * Real.pi/3) (3 * Real.pi/4)) :=
sorry

end range_of_alpha_l490_490407


namespace plums_in_basket_l490_490076

theorem plums_in_basket (initial : ℕ) (added : ℕ) (total : ℕ) (h_initial : initial = 17) (h_added : added = 4) : total = 21 := by
  sorry

end plums_in_basket_l490_490076


namespace sum_of_roots_quadratic_eq_l490_490842

theorem sum_of_roots_quadratic_eq :
  (∑ x in Finset.filter (λ x, x^2 = 9 * x - 20) (Finset.range 100), x) = 9 :=
begin
  sorry
end

end sum_of_roots_quadratic_eq_l490_490842


namespace number_of_possible_points_C_of_conditions_l490_490596

noncomputable def number_of_possible_points_C (line : ℝ × ℝ × ℝ) (circle_center : ℝ × ℝ) (circle_radius : ℝ) (area_triangle_ABC : ℝ) : ℕ :=
sorry

theorem number_of_possible_points_C_of_conditions :
  number_of_possible_points_C (3, 4, -15) (0, 0) 5 8 = 3 :=
by
  sorry

end number_of_possible_points_C_of_conditions_l490_490596


namespace sum_of_solutions_eq_9_l490_490804

theorem sum_of_solutions_eq_9 (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20) :
  let (sum_roots : ℝ) := -b / a in 
  sum_roots = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490804


namespace parallelogram_angle_proof_l490_490341

def quadrilateral (a b c d : Type*) :=
  { ab : int // a + b = 180 } × { bc : int // b + c = 180 } × { cd : int // c + d = 180 } × { da : int // d + a = 180 }

theorem parallelogram_angle_proof (EFGH : quadrilateral ℝ ℝ ℝ ℝ)
  (angle_FGH : ℝ) (angle_EFG : ℝ)
  (h_parallelogram : EFGH)
  (h_angle_FGH : angle_FGH = 75)
  (h_angle_EFG : angle_EFG = 105) :
  let E := 180 - angle_EFG in
  E = 75 :=
sorry

end parallelogram_angle_proof_l490_490341


namespace solve_for_x_l490_490347

theorem solve_for_x : 
  ∀ x : ℝ, (sqrt (9 + sqrt (27 + 9 * x)) + sqrt (6 + sqrt (9 + 3 * x)) = 3 + 3 * sqrt 3) → x = 48 := 
by
  sorry

end solve_for_x_l490_490347


namespace functional_equation_solution_l490_490692

noncomputable def f (x : ℝ) : ℝ := (x^3 - x^2 - x + 1) / (2 * (x^2 - x))

theorem functional_equation_solution : 
  ∀ x ∈ (Set.Univ \ {0, 1} : Set ℝ), 
  f(x) + f(1 - 1/x) = 1 + x :=
begin
  sorry
end

end functional_equation_solution_l490_490692


namespace right_triangle_area_l490_490176

theorem right_triangle_area (a : ℝ) (h1 : a > 0) :
  let b := (2 / 3) * a,
      c := (∛13 / 3) * a,
      area := a * b / 2 in 
  area = 8 / 3 :=
by {
  sorry
}

end right_triangle_area_l490_490176


namespace range_of_k_l490_490579

theorem range_of_k (k : ℝ) : (4 < k ∧ k < 9 ∧ k ≠ 13 / 2) ↔ (k ∈ Set.Ioo 4 (13 / 2) ∪ Set.Ioo (13 / 2) 9) :=
by
  sorry

end range_of_k_l490_490579


namespace smallest_natural_number_to_create_palindrome_l490_490784

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

theorem smallest_natural_number_to_create_palindrome :
  ∃ (n : ℕ), is_palindrome (25751 + n) ∧ ∀ m < n, ¬ is_palindrome (25751 + m) :=
sorry

end smallest_natural_number_to_create_palindrome_l490_490784


namespace tribe_leadership_count_l490_490473

theorem tribe_leadership_count : 
  ∃ (choose_chief : ℕ) (choose_support_chiefs : ℕ × ℕ × ℕ) (choose_inferior_officers : ℕ × ℕ × ℕ), 
  let (chief_count, support_chief_A, support_chief_B, support_chief_C) := (choose_chief, choose_support_chiefs.1, choose_support_chiefs.2.1, choose_support_chiefs.2.2) in
  let (support_A_count, support_B_count, support_C_count) := (choose_inferior_officers.1, choose_inferior_officers.2.1, choose_inferior_officers.2.2) in 
  chief_count = 13 ∧ 
  support_chief_A = 12 ∧ 
  support_chief_B = 11 ∧ 
  support_chief_C = 10 ∧ 
  support_A_count = nat.choose 9 2 ∧ 
  support_B_count = nat.choose 7 2 ∧ 
  support_C_count = nat.choose 5 2 ∧ 
  chief_count * support_chief_A * support_chief_B * support_chief_C * support_A_count * support_B_count * support_C_count = 18604800 :=
begin
  use 13,
  use (12, 11, 10),
  use (nat.choose 9 2, nat.choose 7 2, nat.choose 5 2),
  sorry 
end

end tribe_leadership_count_l490_490473


namespace Jack_time_to_school_l490_490154

universe u

-- Definitions from the conditions
def Dave_steps_per_minute : ℕ := 80
def Dave_step_length_cm : ℕ := 70
def Dave_time_minutes : ℕ := 20

def Jack_initial_steps_per_minute : ℕ := 110
def Jack_step_length_cm : ℕ := 65
def Jack_adjusted_steps_per_minute : ℕ := 90
def Jack_initial_time_minutes : ℕ := 3

-- Main theorem statement
theorem Jack_time_to_school : 
  (Jack_initial_time_minutes + 
  (112000 - (Jack_initial_steps_per_minute * Jack_step_length_cm * Jack_initial_time_minutes)) / 
  (Jack_adjusted_steps_per_minute * Jack_step_length_cm)) ≈ 18.47 := 
sorry

end Jack_time_to_school_l490_490154


namespace danny_steve_ratio_l490_490496

theorem danny_steve_ratio :
  ∀ (D S : ℝ),
  D = 29 →
  2 * (S / 2 - D / 2) = 29 →
  D / S = 1 / 2 :=
by
  intros D S hD h_eq
  sorry

end danny_steve_ratio_l490_490496


namespace part_a_part_b_part_c_l490_490683

-- Define the ring and the matrix structure
variable {K : Type*} [CommRing K]

def M : Type* := Matrix (Fin 2) (Fin 2) K

-- The proof that M is a ring with a unit and not commutative
theorem part_a : (∀ (a b : M), a + b = b + a ∧ (a * b) = (b * a) → false) ∧ ∃ (I : M), ∀ (A : M), A * I = A ∧ I * A = A := sorry

-- The proof for the invertibility condition for matrices in M when K is a commutative field
theorem part_b {K : Type*} [Field K] (A : M) : (∃ (A_inv : M), A * A_inv = 1 ∧ A_inv * A = 1) ↔ ((A 0 0 * A 1 1 - A 0 1 * A 1 0) ≠ 0) := sorry

-- The proof that the subset of M consisting of invertible matrices forms a multiplicative group
theorem part_c {K : Type*} [Field K] : ∃ (G : Set M), (∀ x ∈ G, ∃ y ∈ G, x * y = 1 ∧ y * x = 1) ∧ (∀ x y ∈ G, x * y ∈ G) := sorry

end part_a_part_b_part_c_l490_490683


namespace irrational_sum_contradiction_l490_490307

theorem irrational_sum_contradiction (x : ℝ) :
  ¬ (∃ a b : ℚ, x + real.sqrt 3 = a ∧ x^3 + 5 * real.sqrt 3 = b) :=
begin
  sorry
end

end irrational_sum_contradiction_l490_490307


namespace probability_of_both_selected_l490_490438

theorem probability_of_both_selected :
  let pX := 1 / 5
  let pY := 2 / 7
  (pX * pY) = 2 / 35 :=
by
  let pX := 1 / 5
  let pY := 2 / 7
  show (pX * pY) = 2 / 35
  sorry

end probability_of_both_selected_l490_490438


namespace sqrt_of_nine_l490_490395

theorem sqrt_of_nine : (sqrt 9 = 3 ∨ sqrt 9 = -3) := by
  sorry

end sqrt_of_nine_l490_490395


namespace set_inter_complement_eq_l490_490540

open Set

theorem set_inter_complement_eq :
  ∀ (U A B : Set ℕ),
    U = (finset.range 6).to_set ∧
    A = {2, 3} ∧
    B = {3, 5} →
    A ∩ (U \ B) = {2} :=
by
  intro U A B
  rintro ⟨hU, hA, hB⟩
  rw [hU, finset.range_succ_eq_of_n 6] at hU
  rw [hA, hB] at *
  sorry

end set_inter_complement_eq_l490_490540


namespace eight_step_paths_count_l490_490093

-- Define the conditions
def grid_size := 9
def total_squares := grid_size * grid_size
def start_on_white (pos : ℕ) := pos = 1
def end_on_any (pos : ℕ) := pos = grid_size

-- Define the problem
theorem eight_step_paths_count :
  ∃ paths : ℕ, 
    paths = 70 ∧ 
    valid_paths grid_size start_on_white end_on_any paths :=
begin
  sorry
end

-- Auxiliary definitions for validity of paths (placeholders)
def valid_paths (n : ℕ) (p1 : ℕ → Prop) (p2 : ℕ → Prop) (k : ℕ) : Prop := sorry

end eight_step_paths_count_l490_490093


namespace area_of_R2_l490_490211

theorem area_of_R2 (a1 a2 : ℝ) (d2 : ℝ) (h_area1 : a1 * a2 = 18) (h_side1 : a1 = 3) (h_diagonal : d2 = 20) (h_similar : a2 / a1 = 2) : 
  (let a := Real.sqrt 80; let b := 2 * a in a * b = 160) :=
by
  have : a2 = 6 := by sorry
  have : a = 4 * Real.sqrt 5 := by sorry
  have : b = 8 * Real.sqrt 5 := by sorry
  show 4 * Real.sqrt 5 * (8 * Real.sqrt 5) = 160 := by sorry

end area_of_R2_l490_490211


namespace minimum_value_expr_l490_490191

theorem minimum_value_expr (x : ℝ) (h : x > 2) :
  ∃ y, y = (x^2 - 6 * x + 8) / (2 * x - 4) ∧ y = -1/2 := sorry

end minimum_value_expr_l490_490191


namespace sum_of_solutions_l490_490830

theorem sum_of_solutions (x : ℝ) : 
  (∀ x : ℝ, x^2 = 9*x - 20 → x = 4 ∨ x = 5) → (4 + 5 = 9) :=
by
  intros h
  calc 4 + 5 = 9 : by norm_num
  sorry

end sum_of_solutions_l490_490830


namespace exists_common_point_l490_490331

-- Definitions: Rectangle and the problem conditions
structure Rectangle :=
(x_min y_min x_max y_max : ℝ)
(h_valid : x_min ≤ x_max ∧ y_min ≤ y_max)

def rectangles_intersect (R1 R2 : Rectangle) : Prop :=
¬(R1.x_max < R2.x_min ∨ R2.x_max < R1.x_min ∨ R1.y_max < R2.y_min ∨ R2.y_max < R1.y_min)

def all_rectangles_intersect (rects : List Rectangle) : Prop :=
∀ (R1 R2 : Rectangle), R1 ∈ rects → R2 ∈ rects → rectangles_intersect R1 R2

-- Theorem: Existence of a common point
theorem exists_common_point (rects : List Rectangle) (h_intersect : all_rectangles_intersect rects) : 
  ∃ (T : ℝ × ℝ), ∀ (R : Rectangle), R ∈ rects → 
    R.x_min ≤ T.1 ∧ T.1 ≤ R.x_max ∧ 
    R.y_min ≤ T.2 ∧ T.2 ≤ R.y_max := 
sorry

end exists_common_point_l490_490331


namespace solve_sqrt_equation_l490_490001

noncomputable def solve_equation (x : ℝ) : Prop :=
  sqrt (2 * x + 16) - (8 / sqrt (2 * x + 16)) = 4

theorem solve_sqrt_equation (x : ℝ) (h : solve_equation x) : x = 4 * real.sqrt 3 :=
by
  sorry

end solve_sqrt_equation_l490_490001


namespace eval_power_l490_490169

theorem eval_power {a m n : ℕ} : (a^m)^n = a^(m * n) := by
  sorry

example : (3^2)^4 = 6561 := by
  rw eval_power
  norm_num

end eval_power_l490_490169


namespace Miranda_can_stuff_pillows_l490_490321

theorem Miranda_can_stuff_pillows:
  let pounds_per_pillow := 2 in
  let goose_feathers_per_pound := 300 in
  let duck_feathers_per_pound := 500 in
  let total_goose_feathers := 3600 in
  let total_duck_feathers := 4000 in
  let goose_feathers_in_pounds := total_goose_feathers / goose_feathers_per_pound in
  let duck_feathers_in_pounds := total_duck_feathers / duck_feathers_per_pound in
  let total_feathers_in_pounds := goose_feathers_in_pounds + duck_feathers_in_pounds in
  let pillows_stuffed := total_feathers_in_pounds / pounds_per_pillow in
  pillows_stuffed = 10 := by
  sorry

end Miranda_can_stuff_pillows_l490_490321


namespace parts_from_blanks_9_parts_from_blanks_14_blanks_needed_for_40_parts_l490_490639

theorem parts_from_blanks_9 : ∀ (produced_parts : ℕ), produced_parts = 13 :=
by
  sorry

theorem parts_from_blanks_14 : ∀ (produced_parts : ℕ), produced_parts = 20 :=
by
  sorry

theorem blanks_needed_for_40_parts : ∀ (required_blanks : ℕ), required_blanks = 27 :=
by
  sorry

end parts_from_blanks_9_parts_from_blanks_14_blanks_needed_for_40_parts_l490_490639


namespace find_a_l490_490155

def F (a b c : ℝ) : ℝ := a * (b^2 + c^2) + b * c

theorem find_a (a : ℝ) (h : F a 3 4 = F a 2 5) : a = 1 / 2 :=
by
  sorry

end find_a_l490_490155


namespace double_angle_cosine_calculation_l490_490123

theorem double_angle_cosine_calculation :
    2 * (Real.cos (Real.pi / 12))^2 - 1 = Real.cos (Real.pi / 6) := 
by
    sorry

end double_angle_cosine_calculation_l490_490123


namespace number_of_possible_s_values_l490_490133

theorem number_of_possible_s_values : 
  let s_values := {s : ℕ | s < 144 ∧ 144 % s = 0} in
  s_values.to_finset.card = 14 :=
by
  sorry

end number_of_possible_s_values_l490_490133


namespace proposition_proof_l490_490880

def StatementA : Prop := ∃ (A B C D : Type), Line ∣ A ∣ B ∧ Line ∣ C ∣ D
def StatementB : Prop := false  -- This represents the "Is the weather good today?" question can't be a proposition
def StatementC : Prop := false  -- This represents the "Connect points A and B" can't be a proposition
def StatementD : Prop := ∀ {α β : ℝ}, Supplement α = Supplement β → Supplement α = Supplement β 

theorem proposition_proof : StatementD := by
  intro α β h
  exact h

end proposition_proof_l490_490880


namespace even_and_multiple_of_3_l490_490738

theorem even_and_multiple_of_3 (a b : ℤ) (h1 : ∃ k : ℤ, a = 2 * k) (h2 : ∃ n : ℤ, b = 6 * n) :
  (∃ m : ℤ, a + b = 2 * m) ∧ (∃ p : ℤ, a + b = 3 * p) :=
by
  sorry

end even_and_multiple_of_3_l490_490738


namespace union_A_B_subset_B_A_l490_490300

-- Condition definitions
def A : Set ℝ := {x | 2 * x - 8 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2 * (m + 1) * x + m^2 = 0}

-- Problem 1: If m = 4, prove A ∪ B = {2, 4, 8}
theorem union_A_B (m : ℝ) (h : m = 4) : A ∪ B m = {2, 4, 8} :=
sorry

-- Problem 2: If B ⊆ A, find the range for m
theorem subset_B_A (m : ℝ) (h : B m ⊆ A) : 
  m = 4 + 2 * Real.sqrt 2 ∨ m = 4 - 2 * Real.sqrt 2 ∨ m < -1 / 2 :=
sorry

end union_A_B_subset_B_A_l490_490300


namespace daisies_per_bouquet_l490_490089

def total_bouquets := 20
def rose_bouquets := 10
def roses_per_rose_bouquet := 12
def total_flowers_sold := 190

def total_roses_sold := rose_bouquets * roses_per_rose_bouquet
def daisy_bouquets := total_bouquets - rose_bouquets
def total_daisies_sold := total_flowers_sold - total_roses_sold

theorem daisies_per_bouquet :
  (total_daisies_sold / daisy_bouquets = 7) := sorry

end daisies_per_bouquet_l490_490089


namespace lauren_change_l490_490676

theorem lauren_change :
  let meat_cost      := 2 * 3.50
  let buns_cost      := 1.50
  let lettuce_cost   := 1.00
  let tomato_cost    := 1.5 * 2.00
  let pickles_cost   := 2.50 - 1.00
  let total_cost     := meat_cost + buns_cost + lettuce_cost + tomato_cost + pickles_cost
  let payment        := 20.00
  let change         := payment - total_cost
  change = 6.00 :=
by
  unfold meat_cost buns_cost lettuce_cost tomato_cost pickles_cost total_cost payment change
  -- Prove the main statement.
  sorry

end lauren_change_l490_490676


namespace sequence_2024th_term_l490_490598

theorem sequence_2024th_term :
  let a : ℕ → ℤ := λ n, (-1)^n * (2 / 3) * ((10 : ℤ)^n - 1)
  in a 2024 = (2 / 3) * (10^2024 - 1) :=
by sorry

end sequence_2024th_term_l490_490598


namespace sin_double_angle_in_terms_of_y_l490_490688

variables (α y : ℝ) (h1 : α > 0 ∧ α < π / 2) (h2 : cos (α / 2) = sqrt ((y + 1) / (2 * y)))

theorem sin_double_angle_in_terms_of_y : sin (2 * α) = (2 * sqrt (y^2 - 1)) / y :=
by 
  sorry

end sin_double_angle_in_terms_of_y_l490_490688


namespace ship_distance_plot_l490_490463

theorem ship_distance_plot (r : ℝ) (A B C X : Point) : 
  (∃ f : ℝ → ℝ, 
    (∀ t ∈ [0, T₁], f t = r) ∧
    (∀ t ∈ [T₁, T₂], f t > r ∧ f t ↑) ∧
    (∀ t ∈ [T₂, T₃], f t < r ∧ f t ↓) ∧
    f 0 = r ∧ f T₃ = r) 
:=
sorry

end ship_distance_plot_l490_490463


namespace magic_8_ball_probability_l490_490665

theorem magic_8_ball_probability :
  let num_questions := 7
  let num_positive := 3
  let positive_probability := 3 / 7
  let negative_probability := 4 / 7
  let binomial_coefficient := Nat.choose num_questions num_positive
  let total_probability := binomial_coefficient * (positive_probability ^ num_positive) * (negative_probability ^ (num_questions - num_positive))
  total_probability = 242112 / 823543 :=
by
  sorry

end magic_8_ball_probability_l490_490665


namespace smallest_possible_t_l490_490392

theorem smallest_possible_t (t : ℕ) (h₁ : 7.5 + t > 11) (h₂ : 7.5 + 11 > t) (h₃ : 11 + t > 7.5) : t = 4 :=
by
  sorry

end smallest_possible_t_l490_490392


namespace find_b_l490_490236

noncomputable def y (a : ℝ) (x : ℝ) : ℝ := log a (x + 3) - 8 / 9

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 3 ^ x + b

theorem find_b (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (hA : y a (-2) = -8 / 9) :
  ∃ b : ℝ, f (-2) b = -8 / 9 ∧ b = -1 :=
by sorry

end find_b_l490_490236


namespace euclidean_algorithm_iterations_l490_490740

theorem euclidean_algorithm_iterations (a b : ℕ) 
  (ha : a ≤ 1988) (hb : b ≤ 1988) : 
  ∃ n ≤ 6, (let rec algo : ℕ × ℕ → ℕ
  | (0, _) => 0
  | (_, 0) => 0
  | (m, n) => 1 + algo (n, m % n)
  in algo (a, b)) ≤ n :=
sorry

end euclidean_algorithm_iterations_l490_490740


namespace cupcakes_per_day_l490_490130

theorem cupcakes_per_day (goal cupcakes_to_Bonnie days : ℕ) (h1 : goal = 96) (h2 : cupcakes_to_Bonnie = 24) (h3 : days = 2) : 
  (goal + cupcakes_to_Bonnie) / days = 60 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end cupcakes_per_day_l490_490130


namespace equilateral_tri_area_ade_l490_490941

noncomputable def area_of_ade : ℝ :=
  let s := 6 in  -- side length of the equilateral triangle
  let h := 3 * Real.sqrt 3 in  -- height of the equilateral triangle
  1 / 2 * (s / 3) * h

theorem equilateral_tri_area_ade (A B C D E : ℝ) (h₁ : ∀ x y z, x + y + z = 27) 
(h₂ : ∀ P Q R, ∠PQR = 60) :
  area_of_ade = 3 * Real.sqrt 3 := by
  sorry

end equilateral_tri_area_ade_l490_490941


namespace f_monotonically_increasing_f_minimum_value_f_maximum_value_l490_490583

noncomputable def f (x : ℝ) := 2 * sin x * cos x + sqrt 3 * cos (2 * x) + 2

theorem f_monotonically_increasing (k : ℤ) :
  ∀ x, -5 * (π / 12) + k * π ≤ x ∧ x ≤ π / 12 + k * π → monotone_increasing (λ x, f x) :=
sorry

theorem f_minimum_value :
  ∀ x, -π / 3 ≤ x ∧ x ≤ π / 3 → 2 - sqrt 3 ≤ f x :=
sorry

theorem f_maximum_value :
  ∀ x, -π / 3 ≤ x ∧ x ≤ π / 3 → f x ≤ 4 :=
sorry

end f_monotonically_increasing_f_minimum_value_f_maximum_value_l490_490583


namespace sufficient_drivers_and_correct_time_l490_490916

-- Conditions definitions
def one_way_minutes := 2 * 60 + 40  -- 2 hours 40 minutes in minutes
def round_trip_minutes := 2 * one_way_minutes  -- round trip in minutes
def rest_minutes := 60  -- mandatory rest period in minutes

-- Time checks for drivers
def driver_a_return := 12 * 60 + 40  -- Driver A returns at 12:40 PM in minutes
def driver_a_next_trip := driver_a_return + rest_minutes  -- Driver A's next trip time
def driver_d_departure := 13 * 60 + 5  -- Driver D departs at 13:05 in minutes

-- Verify sufficiency of four drivers and time correctness
theorem sufficient_drivers_and_correct_time : 
  4 = 4 ∧ (driver_a_next_trip + round_trip_minutes = 21 * 60 + 30) :=
by
  -- Explain the reasoning path that leads to this conclusion within this block
  sorry

end sufficient_drivers_and_correct_time_l490_490916


namespace distribute_pencils_l490_490509

variables {initial_pencils : ℕ} {num_containers : ℕ} {additional_pencils : ℕ}

theorem distribute_pencils (h₁ : initial_pencils = 150) (h₂ : num_containers = 5)
                           (h₃ : additional_pencils = 30) :
  (initial_pencils + additional_pencils) / num_containers = 36 :=
by sorry

end distribute_pencils_l490_490509


namespace cafeteria_apples_count_l490_490386

def initial_apples : ℕ := 17
def used_monday : ℕ := 2
def bought_monday : ℕ := 23
def used_tuesday : ℕ := 4
def bought_tuesday : ℕ := 15
def used_wednesday : ℕ := 3

def final_apples (initial_apples used_monday bought_monday used_tuesday bought_tuesday used_wednesday : ℕ) : ℕ :=
  initial_apples - used_monday + bought_monday - used_tuesday + bought_tuesday - used_wednesday

theorem cafeteria_apples_count :
  final_apples initial_apples used_monday bought_monday used_tuesday bought_tuesday used_wednesday = 46 :=
by
  sorry

end cafeteria_apples_count_l490_490386


namespace daisies_per_bouquet_is_7_l490_490091

/-
Each bouquet of roses contains 12 roses.
Each bouquet of daisies contains an equal number of daisies.
The flower shop sells 20 bouquets today.
10 of the bouquets are rose bouquets and 10 are daisy bouquets.
The flower shop sold 190 flowers in total today.
-/

def num_daisies_per_bouquet (roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold : ℕ) : ℕ :=
  (total_flowers_sold - total_roses_sold) / bouquets_sold 

theorem daisies_per_bouquet_is_7 :
  ∀ (roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold : ℕ),
  (roses_per_bouquet = 12) →
  (bouquets_sold = 10) →
  (total_roses_sold = bouquets_sold * roses_per_bouquet) →
  (total_flowers_sold = 190) →
  num_daisies_per_bouquet roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold = 7 :=
by
  intros
  -- Placeholder for the actual proof
  sorry

end daisies_per_bouquet_is_7_l490_490091


namespace average_speed_round_trip_l490_490109

theorem average_speed_round_trip (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (2 * m * n) / (m + n) = (2 * (m * n)) / (m + n) :=
  sorry

end average_speed_round_trip_l490_490109


namespace savings_increase_l490_490336

-- Define initial income I
variable (I : ℝ) (hI : I > 0)

-- Define percentages as constants
def reg_expense_rate : ℝ := 0.75
def add_expense_rate : ℝ := 0.10
def income_increase_rate : ℝ := 0.20
def reg_expense_increase_rate : ℝ := 0.10
def add_expense_increase_rate : ℝ := 0.25

-- Calculate initial savings
def initial_savings : ℝ := I * (1 - reg_expense_rate - add_expense_rate)

-- Calculate the new income after increment
def new_income : ℝ := I * (1 + income_increase_rate)

-- Calculate the new regular and additional expenses after increments
def new_reg_expense : ℝ := I * reg_expense_rate * (1 + reg_expense_increase_rate)
def new_add_expense : ℝ := I * add_expense_rate * (1 + add_expense_increase_rate)

-- Calculate new savings
def new_savings : ℝ := new_income - (new_reg_expense + new_add_expense)

-- Calculate percentage increase in savings
def percentage_increase (initial new : ℝ) : ℝ := (new - initial) / initial * 100

-- Main statement: Prove the percentage increase in savings is 66.67%
theorem savings_increase : percentage_increase initial_savings new_savings ≈ 66.67 :=
by
  -- Proof is omitted, add sorry as a placeholder
  sorry

end savings_increase_l490_490336


namespace sum_of_solutions_eq_9_l490_490826

theorem sum_of_solutions_eq_9 :
  let roots := {x : ℝ | x^2 = 9 * x - 20}
  in ∑ x in roots, x = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490826


namespace measure_10_grams_coefficient_l490_490190

noncomputable def f (t : ℚ) : ℚ :=
  (t⁻¹ + 1 + t)^3 * (t⁻² + 1 + t²)^3 * (t⁻⁵ + 1 + t⁵)

theorem measure_10_grams_coefficient :
  (polynomial.coeff (polynomial.expand ℚ (f : ℚ → polynomial ℚ) 1) 10) = 29 :=
sorry

end measure_10_grams_coefficient_l490_490190


namespace kendra_birdwatching_l490_490673

theorem kendra_birdwatching :
  let birds_seen_monday := 5 * 7 in
  let birds_seen_tuesday := 5 * 5 in
  let birds_seen_wednesday := 10 * 8 in
  let total_birds_seen := birds_seen_monday + birds_seen_tuesday + birds_seen_wednesday in
  let total_sites_visited := 5 + 5 + 10 in
  (total_birds_seen / total_sites_visited : ℝ) = 7 :=
by
  sorry

end kendra_birdwatching_l490_490673


namespace hcf_of_three_numbers_l490_490751

def hcf (a b : ℕ) : ℕ := gcd a b

theorem hcf_of_three_numbers :
  let a := 136
  let b := 144
  let c := 168
  hcf (hcf a b) c = 8 :=
by
  sorry

end hcf_of_three_numbers_l490_490751


namespace sum_of_solutions_l490_490851

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | x^2 = 9*x - 20}, x) = 9 := 
sorry

end sum_of_solutions_l490_490851


namespace jezebel_red_roses_count_l490_490668

def cost_of_red_roses (R : ℕ) : ℝ := 1.50 * R
def cost_of_sunflowers : ℝ := 3 * 3
def total_cost (R : ℕ) : ℝ := cost_of_red_roses R + cost_of_sunflowers

theorem jezebel_red_roses_count : ∃ (R : ℕ), total_cost R = 45 ∧ R = 24 :=
by
  have h : total_cost 24 = 45 := by
    simp [cost_of_red_roses, cost_of_sunflowers, total_cost]
    norm_num
  use 24
  tauto

end jezebel_red_roses_count_l490_490668


namespace sum_of_solutions_l490_490797

theorem sum_of_solutions (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) : 
  -b / a = 9 :=
by
  -- The proof is omitted here (hence the 'sorry')
  sorry

end sum_of_solutions_l490_490797


namespace average_temperature_correct_l490_490744

theorem average_temperature_correct (W T : ℝ) :
  (38 + W + T) / 3 = 32 →
  44 = 44 →
  38 = 38 →
  (W + T + 44) / 3 = 34 :=
by
  intros h1 h2 h3
  sorry

end average_temperature_correct_l490_490744


namespace side_length_S2_l490_490726

-- Define the variables
variables (r s : ℕ)

-- Given conditions
def condition1 : Prop := 2 * r + s = 2300
def condition2 : Prop := 2 * r + 3 * s = 4000

-- The main statement to be proven
theorem side_length_S2 (h1 : condition1 r s) (h2 : condition2 r s) : s = 850 := sorry

end side_length_S2_l490_490726


namespace sufficient_not_necessary_l490_490443

theorem sufficient_not_necessary (x : ℝ) : (x > 3) → (abs (x - 3) > 0) ∧ (¬(abs (x - 3) > 0) → (¬(x > 3))) :=
by
  sorry

end sufficient_not_necessary_l490_490443


namespace cosine_value_l490_490213

theorem cosine_value (α : ℝ) (h : Real.sin (α - Real.pi / 3) = 1 / 3) :
  Real.cos (α + Real.pi / 6) = -1 / 3 :=
by
  sorry

end cosine_value_l490_490213


namespace rectangle_placement_l490_490690

theorem rectangle_placement (a b c d : ℝ)
  (h1 : a < c)
  (h2 : c < d)
  (h3 : d < b)
  (h4 : a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b * d - a * c)^2 + (b * c - a * d)^2 :=
sorry

end rectangle_placement_l490_490690


namespace surface_divides_volume_ratio_l490_490292

-- Definitions based on given conditions
variables (S A B C G : Type)
variables [RegularTriangularPyramid S A B C]
variables (height_SO : Ray S G)
variables [TrihedralAngle G]
variables (BisectorsPassThroughVertices : ∀ (A B C : Type), Triangle S A B C)

-- The statement we need to prove
theorem surface_divides_volume_ratio 
  (h₁ : LocatedOn G height_SO)
  (h₂ : DividesIntoEqualParts S A B C)
  : VolumeRatio S A B C G = 3 / 11 := 
sorry


end surface_divides_volume_ratio_l490_490292


namespace find_constant_in_line_eq_l490_490009

theorem find_constant_in_line_eq (c : ℝ) (h : (∀ x y : ℝ, y = (4 / 3) * x - c → sqrt (x^2 + y^2) ≥ 60)) : c = 100 :=
sorry

end find_constant_in_line_eq_l490_490009


namespace ways_companies_accept_students_l490_490765

theorem ways_companies_accept_students (students companies : ℕ) (h_students : students = 4) (h_each_student : ∀ s, s < students → ∃ c, c < companies ∧ accepts s c) (h_each_company : ∀ c, c < companies → ∃ s, s < students ∧ accepts s c) : 
  (number_of_ways students companies) = 60 :=
sorry

end ways_companies_accept_students_l490_490765


namespace Scarlett_adds_correct_amount_l490_490032

-- Define the problem with given conditions
def currentOilAmount : ℝ := 0.17
def desiredOilAmount : ℝ := 0.84

-- Prove that the amount of oil Scarlett needs to add is 0.67 cup
theorem Scarlett_adds_correct_amount : (desiredOilAmount - currentOilAmount) = 0.67 := by
  sorry

end Scarlett_adds_correct_amount_l490_490032


namespace average_birds_per_site_l490_490675

theorem average_birds_per_site :
  let birds_seen (day_sites day_avg : ℕ) := day_sites * day_avg in
  let total_birds := birds_seen 5 7 + birds_seen 5 5 + birds_seen 10 8 in
  let total_sites := 5 + 5 + 10 in
  total_birds / total_sites = 7 :=
by
  sorry

end average_birds_per_site_l490_490675


namespace sum_of_solutions_l490_490850

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | x^2 = 9*x - 20}, x) = 9 := 
sorry

end sum_of_solutions_l490_490850


namespace polygon_sides_l490_490275

theorem polygon_sides (e : ℝ) (h : e = 45) : 360 / e = 8 :=
by {
  rw [h],
  norm_num,
}

end polygon_sides_l490_490275


namespace class_percentage_calculation_l490_490903

noncomputable def percentage_of_class_answering_first_question_correctly :=
  ∀ (A B Neither A_and_B : ℝ),
    B = 0.5 → Neither = 0.2 → A_and_B = 0.33 →
    A = 0.63

theorem class_percentage_calculation :
  percentage_of_class_answering_first_question_correctly := by
  intros A B Neither A_and_B hB hN hA_and_B
  have h1 : A + B - A_and_B + Neither = 1 :=
    by linarith [hB, hN, hA_and_B]
  have h2 : A + (B - A_and_B) + Neither = 1 :=
    by rw [←add_assoc, add_comm (B - A_and_B), add_assoc] at h1; exact h1
  have h3 : A + (0.5 - 0.33) + 0.2 = 1 :=
    by rw [hB, hN, hA_and_B] at h2; exact h2
  have h4 : A + 0.17 + 0.2 = 1 :=
    by norm_num at h3; exact h3
  have h5 : A + 0.37 = 1 :=
    by rw [←add_assoc] at h4; exact h4
  have h6 : A = 0.63 :=
    by linarith [h5]
  exact h6

end class_percentage_calculation_l490_490903


namespace work_to_stretch_springs_l490_490043

theorem work_to_stretch_springs (k1 k2 : ℝ) (x : ℝ) (W : ℝ) 
  (hk1 : k1 = 6000) (hk2 : k2 = 12000) (hx : x = 0.1) : 
  W = 20 :=
by
  -- Assuming the conditions and variables are defined correctly
  have h_eq_k : 1/(1/k1 + 1/k2) = 4000, sorry
  have h_W : W = 0.5 * 4000 * (x^2), sorry
  exact h_W

end work_to_stretch_springs_l490_490043


namespace inequality_sqrt_gt_ax_by_l490_490069

variable (a b x y : ℝ)

theorem inequality_sqrt_gt_ax_by (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 < 1)
  (h4 : x > 0) (h5 : y > 0) : sqrt (x^2 + y^2) > a * x + b * y :=
sorry

end inequality_sqrt_gt_ax_by_l490_490069


namespace projection_of_a_minus_b_onto_b_is_neg_three_halves_b_l490_490569

variables (a b : ℝ^3)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)
variables (angle_ab : real.angle a b = real.angle.pi * 2 / 3)

theorem projection_of_a_minus_b_onto_b_is_neg_three_halves_b : 
  (a - b) ⬝ b / ∥b∥^2 • b = -3/2 • b :=
by sorry

end projection_of_a_minus_b_onto_b_is_neg_three_halves_b_l490_490569


namespace product_sqrt_50_l490_490398

theorem product_sqrt_50 (a b : ℕ) (h₁ : a = 7) (h₂ : b = 8) (h₃ : a^2 < 50) (h₄ : 50 < b^2) : a * b = 56 := by
  sorry

end product_sqrt_50_l490_490398


namespace max_frac_quot_l490_490552

noncomputable def a : ℕ → ℚ
| 1 => 1
| 2 => 3
| n + 1 => (n - 1) • (a (n - 1) / (2 * n)) + (n + 1) • (a (n + 1) / (2 * n))

theorem max_frac_quot {a: ℕ → ℚ} (h₀ : a 1 = 1)
  (h₁ : a 2 = 3)
  (h: ∀ n : ℕ, n ≥ 2 → 2 * n * a n = (n - 1) * a (n - 1) + (n + 1) * a (n + 1)) :
  ∃ n : ℕ, n ≥ 2 ∧ (a n / n = 3 / 2) :=
sorry

end max_frac_quot_l490_490552


namespace sum_of_solutions_l490_490787

-- Define the quadratic equation and variable x
def quadratic_equation := ∀ x : ℝ, (x^2 - 9 * x + 20 = 0)

-- Define what we need to prove
theorem sum_of_solutions : ∃ s : ℝ, (∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → s = x1 + x2) ∧ s = 9 :=
by
  sorry -- Proof is omitted

end sum_of_solutions_l490_490787


namespace find_integer_l490_490361

theorem find_integer (a b c d : ℕ) (h1 : a + b + c + d = 18) 
  (h2 : b + c = 11) (h3 : a - d = 3) (h4 : (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0) :
  10^3 * a + 10^2 * b + 10 * c + d = 5262 ∨ 10^3 * a + 10^2 * b + 10 * c + d = 5622 := 
by
  sorry

end find_integer_l490_490361


namespace P_Q_B_C_cyclic_l490_490308

variables (A B C D K L P Q : Type)

-- Hypotheses:
hypothesis h1 : is_trapezoid A B C D ∧ (A.1 = B.0) ∧ (D.1 = C.0) ∧ (A.2 > B.2)
hypothesis h2 : ∃ K, K ∈ AB ∧ (AK / KB = DL / LC)
hypothesis h3 : ∃ L, L ∈ CD ∧ (AK / KB = DL / LC)
hypothesis h4 : (K.1 = P.1) ∧ (L.1 = Q.1)
hypothesis h5 : (∠APB = ∠BCD) ∧ (∠CQD = ∠ABC)

-- The theorem we want to prove:
theorem P_Q_B_C_cyclic : is_cyclic P Q B C :=
begin
  sorry
end

end P_Q_B_C_cyclic_l490_490308


namespace roots_of_Q_are_fifth_powers_of_roots_of_P_l490_490297

def P (x : ℝ) : ℝ := x^3 - 3 * x + 1

noncomputable def Q (y : ℝ) : ℝ := y^3 + 15 * y^2 - 198 * y + 1

theorem roots_of_Q_are_fifth_powers_of_roots_of_P : 
  ∀ α β γ : ℝ, (P α = 0) ∧ (P β = 0) ∧ (P γ = 0) →
  (Q (α^5) = 0) ∧ (Q (β^5) = 0) ∧ (Q (γ^5) = 0) := 
by 
  intros α β γ h
  sorry

end roots_of_Q_are_fifth_powers_of_roots_of_P_l490_490297


namespace part_I_part_II_l490_490559

def intersection_of_two_lines (α : ℝ) (π : ℝ) : set (ℝ × ℝ) :=
  if α = π / 3 then
    { (1, 0), (1/2, -Real.sqrt 3 / 2) }
  else
    ∅

def trajectory_of_point_P (α : ℝ) : (ℝ × ℝ) :=
  (1 / 2 * Real.sin α ^ 2, -1 / 2 * Real.sin α * Real.cos α)

theorem part_I (α : ℝ) (π : ℝ) :
  intersection_of_two_lines α π = { (1, 0), (1/2, -Real.sqrt 3 / 2) } :=
sorry

theorem part_II (α : ℝ) :
  let P_trajectory := (1 / 2 * Real.sin α ^ 2, -1 / 2 * Real.sin α * Real.cos α)
  ∃ O A : ℝ × ℝ, P_trajectory = (1 / 4 * Real.sin α ^ 2, -1 / 2 * Real.sin α * Real.cos α)
  ∧ ( (fst O + fst A) / 2, (snd O + snd A) / 2) = (1 / 4, 0)
  ∧ ( (fst O - 1 / 4) ^ 2 + (snd A) ^ 2 = 1 / 16 ));
sorry

end part_I_part_II_l490_490559


namespace dmitriy_before_father_l490_490782

noncomputable def probability_dmitriy_before_father (m x y z : ℝ) : ℝ :=
  if 0 < x ∧ x < m ∧ 0 < y ∧ y < z ∧ z < m then 2/3 else 0

theorem dmitriy_before_father (m x y z : ℝ) (h : 0 < x ∧ x < m ∧ 0 < y ∧ y < z ∧ z < m) :
  probability_dmitriy_before_father m x y z = 2/3 :=
begin
  sorry  -- Proof goes here
end

end dmitriy_before_father_l490_490782


namespace base_three_to_base_ten_l490_490491

theorem base_three_to_base_ten : 
  (2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 2 * 3^1 + 1 * 3^0 = 178) :=
by
  sorry

end base_three_to_base_ten_l490_490491


namespace sufficient_drivers_and_correct_time_l490_490913

-- Conditions definitions
def one_way_minutes := 2 * 60 + 40  -- 2 hours 40 minutes in minutes
def round_trip_minutes := 2 * one_way_minutes  -- round trip in minutes
def rest_minutes := 60  -- mandatory rest period in minutes

-- Time checks for drivers
def driver_a_return := 12 * 60 + 40  -- Driver A returns at 12:40 PM in minutes
def driver_a_next_trip := driver_a_return + rest_minutes  -- Driver A's next trip time
def driver_d_departure := 13 * 60 + 5  -- Driver D departs at 13:05 in minutes

-- Verify sufficiency of four drivers and time correctness
theorem sufficient_drivers_and_correct_time : 
  4 = 4 ∧ (driver_a_next_trip + round_trip_minutes = 21 * 60 + 30) :=
by
  -- Explain the reasoning path that leads to this conclusion within this block
  sorry

end sufficient_drivers_and_correct_time_l490_490913


namespace vasya_petya_no_mistake_l490_490777

theorem vasya_petya_no_mistake :
  ∃ (x p q : ℝ), prime (nat_abs (10 * x)) ∧ prime (nat_abs (15 * x)) ∧ 10 * x = p ∧ 15 * x = q ∧ 3 * p = 2 * q :=
by
  sorry

end vasya_petya_no_mistake_l490_490777


namespace average_pastries_per_day_l490_490445

def monday_sales : ℕ := 2
def increment_weekday : ℕ := 2
def increment_weekend : ℕ := 3

def tuesday_sales : ℕ := monday_sales + increment_weekday
def wednesday_sales : ℕ := tuesday_sales + increment_weekday
def thursday_sales : ℕ := wednesday_sales + increment_weekday
def friday_sales : ℕ := thursday_sales + increment_weekday
def saturday_sales : ℕ := friday_sales + increment_weekend
def sunday_sales : ℕ := saturday_sales + increment_weekend

def total_sales_week : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales + sunday_sales
def average_sales_per_day : ℚ := total_sales_week / 7

theorem average_pastries_per_day : average_sales_per_day = 59 / 7 := by
  sorry

end average_pastries_per_day_l490_490445


namespace volume_divided_by_pi_of_cone_formed_from_sector_l490_490906

theorem volume_divided_by_pi_of_cone_formed_from_sector :
  let r := 15
  let sector_angle := 270
  let arc_length := (sector_angle / 360) * (2 * π * r)
  let cone_base_radius := arc_length / (2 * π)
  let slant_height := r
  let height := sqrt (r^2 - cone_base_radius^2)
  let volume := (1 / 3) * π * cone_base_radius^2 * height
  volume / π = 1184.59375 :=
by
  -- Proof would go here
  sorry

end volume_divided_by_pi_of_cone_formed_from_sector_l490_490906


namespace find_ellipse_equation_find_slope_l490_490556

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (∀ P : ℝ × ℝ, (P.1^2 / a^2 + P.2^2 / b^2 = 1) → dist P (1, 0) + dist P (-1, 0) = 2 * sqrt 2) ∧ 
  ((sqrt 2) / 2 = sqrt (a^2 - b^2) / a)

theorem find_ellipse_equation (a b : ℝ) (h : ellipse_equation a b) : 
  ∃ x y : ℝ, (x^2 / 2 + y^2 = 1) :=
sorry

def slope_possible_values (k : ℝ) : Prop :=
  k = 0 ∨ k = sqrt 3 ∨ k = sqrt 3 / 6

theorem find_slope (k : ℝ) (a b : ℝ) (h : ellipse_equation a b)
  (M : ℝ × ℝ) (hM : M = (0, sqrt 3 / 7))
  (F2 : ℝ × ℝ) (hF2 : F2 = (1, 0))
  (A B : ℝ × ℝ) (hAB : ∃ l : ℝ × ℝ, l = A ∧ l = B ∧ dist M A = dist M B) :
  slope_possible_values k :=
sorry

end find_ellipse_equation_find_slope_l490_490556


namespace find_function_l490_490984

theorem find_function (f : ℚ → ℚ) (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 :=
by
  sorry

end find_function_l490_490984


namespace categorize_numbers_l490_490515

-- Definitions based on the problem's conditions
def is_positive_integer (x : ℝ) : Prop := x > 0 ∧ ∃ n : ℤ, x = ↑n
def is_negative_integer (x : ℝ) : Prop := x < 0 ∧ ∃ n : ℤ, x = ↑n
def is_positive_fraction (x : ℝ) : Prop := x > 0 ∧ ∀ n : ℤ, x ≠ ↑n
def is_negative_fraction (x : ℝ) : Prop := x < 0 ∧ ∀ n : ℤ, x ≠ ↑n

-- Given numbers
def numbers : List ℝ := [1, -1/2, 8.9, -7, -3.2, 0.06, 28, -9, 0]

-- Problem statement
theorem categorize_numbers :
  ({x ∈ numbers | is_positive_integer x} = {1, 28}) ∧
  ({x ∈ numbers | is_negative_integer x} = {-7, -9}) ∧
  ({x ∈ numbers | is_positive_fraction x} = {8.9, 0.06}) ∧
  ({x ∈ numbers | is_negative_fraction x} = {-1/2, -3.2}) :=
by
  sorry

end categorize_numbers_l490_490515


namespace lateral_surface_area_of_solid_of_revolution_l490_490382

theorem lateral_surface_area_of_solid_of_revolution 
  (r α : ℝ) (α_pos : 0 < α ∧ α < π / 2) :
  let S := (4 * π * r^2 * (1 + sin α)) / (sin α)^2 in
  ∃ (trapezoid : Trapezoid ℝ), 
    trapezoid.is_right ∧ 
    trapezoid.inscribed_circle_radius = r ∧ 
    trapezoid.acute_angle = α ∧ 
    trapezoid.lateral_surface_area = S 
:= sorry

end lateral_surface_area_of_solid_of_revolution_l490_490382


namespace inequality_holds_equality_condition_l490_490723

noncomputable def pos_four (a b c d : ℝ) : Prop :=
0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d

theorem inequality_holds (a b c d : ℝ) (h : pos_four a b c d) :
  real.cbrt ((a * b * c + a * b * d + a * c * d + b * c * d) / 4) ≤ 
  real.sqrt ((a * b + a * c + a * d + b * c + b * d + c * d) / 6) :=
sorry

theorem equality_condition (a b c d : ℝ) (h : pos_four a b c d) :
  real.cbrt ((a * b * c + a * b * d + a * c * d + b * c * d) / 4) =
  real.sqrt ((a * b + a * c + a * d + b * c + b * d + c * d) / 6) ↔
  a = b ∧ b = c ∧ c = d :=
sorry

end inequality_holds_equality_condition_l490_490723


namespace S_6_eq_zero_l490_490704

variable (a₁ d : ℝ)

-- Definitions of terms in the arithmetic sequence
def a (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Definitions of conditions
axiom (h : (a 2) ^ 2 + (a 3) ^ 2 = (a 4) ^ 2 + (a 5) ^ 2)
axiom (hd : d ≠ 0)

-- Sum of first n terms of an arithmetic sequence
def S : ℕ → ℝ
| 0       := 0
| (n+1) := S n + a (n+1)

-- Problem statement in Lean
theorem S_6_eq_zero : S a₁ d 6 = 0 := by
  sorry

end S_6_eq_zero_l490_490704


namespace sum_of_solutions_l490_490837

theorem sum_of_solutions (x : ℝ) : 
  (∀ x : ℝ, x^2 = 9*x - 20 → x = 4 ∨ x = 5) → (4 + 5 = 9) :=
by
  intros h
  calc 4 + 5 = 9 : by norm_num
  sorry

end sum_of_solutions_l490_490837


namespace range_of_m_l490_490544

theorem range_of_m :
  (∀ x₁ ∈ Icc 0 3, ∃ x₂ ∈ Icc 1 2, ln (x₁^2 + 1) ≥ (1 / 2)^x₂ - m)
  ↔ m ∈ Icc (1 / 4) ∞ :=
by
  sorry

end range_of_m_l490_490544


namespace sum_of_solutions_l490_490794

theorem sum_of_solutions (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) : 
  -b / a = 9 :=
by
  -- The proof is omitted here (hence the 'sorry')
  sorry

end sum_of_solutions_l490_490794


namespace shaded_region_area_l490_490715

theorem shaded_region_area (a : ℝ) : 
  let semicircle_area := 4 * (1 / 2 * π * (a / 2)^2)
  let square_area := a^2
  let shaded_area := semicircle_area - square_area
  shaded_area ≈ (4 / 7) * a^2 :=
by 
  let semicircle_area := 4 * (1 / 2 * π * (a / 2)^2)
  let square_area := a^2
  let shaded_area := semicircle_area - square_area
  have approx_pi : π ≈ 22 / 7, from sorry
  have eq1 : semicircle_area ≈ (11 / 7) * a^2, from sorry
  show shaded_area ≈ (4 / 7) * a^2, from sorry

end shaded_region_area_l490_490715


namespace sum_of_roots_quadratic_eq_l490_490844

theorem sum_of_roots_quadratic_eq :
  (∑ x in Finset.filter (λ x, x^2 = 9 * x - 20) (Finset.range 100), x) = 9 :=
begin
  sorry
end

end sum_of_roots_quadratic_eq_l490_490844


namespace all_cards_eventually_face_up_l490_490890

-- Defining the concepts of card facing up and down
def Card := ℕ
def face_up : Card := 1
def face_down : Card := 2

-- Conditions given in the problem
def is_face_up (c : Card) : Prop := c = face_up
def is_face_down (c : Card) : Prop := c = face_down

-- Define the operation of flipping a section of cards
def flip_section (deck : List Card) (n m : ℕ) : List Card :=
  if n ≤ m ∧ m < deck.length then
    deck.take n ++
    (deck.drop n).take (m - n + 1).reverse ++
    deck.drop (m + 1)
  else deck

-- Define the invariant essentially saying "eventually all cards face up"
def all_face_up (deck : List Card) : Prop :=
  ∀ c ∈ deck, is_face_up c

-- Define the decreasing nature of the sequence in terms of number of face_down cards
def num_face_down (deck : List Card) : ℕ :=
  deck.countp is_face_down

-- The main theorem to be proved
theorem all_cards_eventually_face_up (deck : List Card) :
  ∃ n, (∀ m ≥ n, all_face_up (List.iterate flip_section m deck)) :=
sorry

end all_cards_eventually_face_up_l490_490890


namespace max_perimeter_of_triangle_l490_490578

theorem max_perimeter_of_triangle
  (F A : ℝ × ℝ)
  (x y : ℝ)
  (M N : ℝ × ℝ)
  (h1 : F.1^2 / 100 + F.2^2 / 64 = 1)
  (h2 : A.1 = -F.1 ∧ A.2 = F.2)
  (h3 : M.1^2 / 100 + M.2^2 / 64 = 1)
  (h4 : N.1^2 / 100 + N.2^2 / 64 = 1):
  let a := 10 -- semi-major axis of the ellipse
  in ∃ a = 10, 4 * a = 40 := by
  let a := 10 -- semi-major axis of the ellipse
  have h4 : 4 * a = 40 := by norm_num
  exact ⟨(a = 10), (4 * a = 40)⟩
  sorry

end max_perimeter_of_triangle_l490_490578


namespace marathon_distance_l490_490343

-- Definitions based on the conditions
def total_time_minutes : ℕ := 3 * 60 + 36 -- Total time Rich ran in minutes
def avg_time_per_mile : ℕ := 9 -- Average time to run a mile in minutes

-- Statement to prove
theorem marathon_distance : total_time_minutes / avg_time_per_mile = 24 := by
  -- Definitions
  have h1 : total_time_minutes = 216 := by
    unfold total_time_minutes
    norm_num
  have h2 : avg_time_per_mile = 9 := by
    unfold avg_time_per_mile
    norm_num
  -- Main proof
  rw [h1, h2]
  norm_num
  sorry

end marathon_distance_l490_490343


namespace sum_of_roots_quadratic_eq_l490_490840

theorem sum_of_roots_quadratic_eq :
  (∑ x in Finset.filter (λ x, x^2 = 9 * x - 20) (Finset.range 100), x) = 9 :=
begin
  sorry
end

end sum_of_roots_quadratic_eq_l490_490840


namespace tournament_problem_l490_490111

open scoped Classical

noncomputable def survival_probability (k : ℕ) : ℚ :=
1 - 2 / (k * (k - 1))

noncomputable def total_survival_prob : ℚ :=
(∏ k in finset.range (2021-2+1) + 2, survival_probability k)

noncomputable def second_best_eliminated_probability : ℚ :=
1 - total_survival_prob

theorem tournament_problem :
  ∑ k in finset.range 2020 + 2, (floor (2021 * second_best_eliminated_probability)) = 674 :=
by
  sorry

end tournament_problem_l490_490111


namespace min_coins_for_all_amounts_min_coins_with_half_dollar_l490_490415

def min_coins (m : ℕ) (use_half_dollar : Prop) (c : bool) : ℕ :=
  if c then min_coins_for_amount m else 0

noncomputable def min_coins_for_amount (n : ℕ) : ℕ :=
  if n < 50 then n % 10 + (n / 10)
  else if n < 60 then 1 + (n % 10)
  else if n < 99 then min_coins_for_amount (n - 50) + 1
  else 0

theorem min_coins_for_all_amounts : ∀ n, 1 ≤ n ∧ n < 100 →
  min_coins_for_amount n <= 13 :=
sorry

theorem min_coins_with_half_dollar : ∀ n, 1 ≤ n ∧ n < 100 →
  min_coins n True True = 14 := sorry

end min_coins_for_all_amounts_min_coins_with_half_dollar_l490_490415


namespace ratio_of_areas_l490_490296

-- Given a circle with diameter AB
variables (r : ℝ) (A B C D E : ℝ × ℝ)

-- Define points A and B at the ends of the diameter
def A := (-r, 0)
def B := (r, 0)

-- Point C lies on AB such that 2 * AC = BC
def C := (r / 3, 0)

-- Points D and E are on the circle with specific conditions
def D := (r / 3, (2 * real.sqrt 2 * r) / 3)
def E := (-r / 3, -(2 * real.sqrt 2 * r) / 3)

-- Areas of triangles DCE and ABD
def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
1 / 2 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

def area_DCE := area_triangle D C E
def area_ABD := area_triangle A B D

-- Theorem stating that the ratio of the areas is 1/6
theorem ratio_of_areas : area_DCE / area_ABD = 1 / 6 := 
sorry

end ratio_of_areas_l490_490296


namespace work_hours_required_l490_490252

theorem work_hours_required
  (hours_per_week_orig : ℝ) (weeks_orig : ℕ) (total_earnings : ℝ) (weeks_missed : ℕ) :
  hours_per_week_orig = 15 →
  weeks_orig = 10 →
  total_earnings = 1800 →
  weeks_missed = 1 →
  (total_earnings / (weeks_orig - weeks_missed)) = 16.67 :=
by
  intros hpo wo te wm h1 h2 h3 h4
  sorry

end work_hours_required_l490_490252


namespace daisies_per_bouquet_is_7_l490_490092

/-
Each bouquet of roses contains 12 roses.
Each bouquet of daisies contains an equal number of daisies.
The flower shop sells 20 bouquets today.
10 of the bouquets are rose bouquets and 10 are daisy bouquets.
The flower shop sold 190 flowers in total today.
-/

def num_daisies_per_bouquet (roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold : ℕ) : ℕ :=
  (total_flowers_sold - total_roses_sold) / bouquets_sold 

theorem daisies_per_bouquet_is_7 :
  ∀ (roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold : ℕ),
  (roses_per_bouquet = 12) →
  (bouquets_sold = 10) →
  (total_roses_sold = bouquets_sold * roses_per_bouquet) →
  (total_flowers_sold = 190) →
  num_daisies_per_bouquet roses_per_bouquet daisies_sold bouquets_sold total_roses_sold total_flowers_sold = 7 :=
by
  intros
  -- Placeholder for the actual proof
  sorry

end daisies_per_bouquet_is_7_l490_490092


namespace num_factors_of_n_multiples_of_360_l490_490616

theorem num_factors_of_n_multiples_of_360 (n : ℕ) (h : n = 2^12 * 3^15 * 5^9) : 
  (∃ count : ℕ, count = 1260 ∧ ∀ d : ℕ, d ∣ n → ∃ k : ℕ, d = 360 * k) :=
by {
  have h1 : 3 ≤ 12 := sorry,
  have h2 : 2 ≤ 15 := sorry,
  have h3 : 1 ≤ 9 := sorry,
  have num_factors := (12 - 3 + 1) * (15 - 2 + 1) * (9 - 1 + 1),
  exact ⟨num_factors, by simp [num_factors, h1, h2, h3], sorry⟩
}

end num_factors_of_n_multiples_of_360_l490_490616


namespace correct_final_counts_l490_490326

def initial_cards := (Nell : ℕ, Jeff : ℕ, Rick : ℕ)

def transactions := 
  (give_Nell_to_Jeff : ℕ, give_Nell_to_Rick : ℕ, give_Jeff_to_Rick : ℕ, give_Rick_to_Nell : ℕ, give_Rick_to_Jeff : ℕ)

def final_cards (init : initial_cards) (trans : transactions) : initial_cards :=
  let Nell_new := init.1 - trans.give_Nell_to_Jeff - trans.give_Nell_to_Rick + trans.give_Rick_to_Nell
  let Jeff_new := init.2 + trans.give_Nell_to_Jeff - trans.give_Jeff_to_Rick + trans.give_Rick_to_Jeff
  let Rick_new := init.3 + trans.give_Nell_to_Rick + trans.give_Jeff_to_Rick - trans.give_Rick_to_Nell - trans.give_Rick_to_Jeff
  (Nell_new, Jeff_new, Rick_new)

theorem correct_final_counts :
  final_cards (1052, 348, 512) (214, 115, 159, 95, 78) = (818, 481, 613) :=
by {
  -- Initial counts
  let Nell := 1052
  let Jeff := 348
  let Rick := 512

  -- Transactions
  Nell - 214 - 115 + 95 = 818,
  Jeff + 214 - 159 + 78 = 481,
  Rick + 115 + 159 - 95 - 78 = 613,
  
  sorry
}

end correct_final_counts_l490_490326


namespace flavoring_ratio_comparison_l490_490651

theorem flavoring_ratio_comparison (f_st cs_st w_st : ℕ) (f_sp cs_sp w_sp : ℕ) :
  f_st = 1 → cs_st = 12 → w_st = 30 →
  w_sp = 75 → cs_sp = 5 →
  f_sp / w_sp = f_st / (2 * w_st) →
  (f_st / cs_st) * 3 = f_sp / cs_sp :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end flavoring_ratio_comparison_l490_490651


namespace root_exists_in_interval_1_2_l490_490369

def f (x : ℝ) : ℝ := 3^x - 1/(Real.sqrt x + 1) - 6

theorem root_exists_in_interval_1_2 :
  (f 1) * (f 2) < 0 → ∃ ξ ∈ Ioo 1 2, f ξ = 0 :=
by
  sorry

end root_exists_in_interval_1_2_l490_490369


namespace mass_percentage_of_O_in_H2O_l490_490989

theorem mass_percentage_of_O_in_H2O :
  (let atomic_mass_H := 1.01
       atomic_mass_O := 16.00
       molar_mass_H2O := (2 * atomic_mass_H) + atomic_mass_O
       mass_percentage_O := (atomic_mass_O / molar_mass_H2O) * 100
   in mass_percentage_O ≈ 88.8) :=
by
  sorry

end mass_percentage_of_O_in_H2O_l490_490989


namespace sum_of_solutions_of_quadratic_eq_l490_490812

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 9 * x + 20 = 0

-- Prove that the sum of the solutions to this equation is 9
theorem sum_of_solutions_of_quadratic_eq : 
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 9) := 
begin
  -- Proof is omitted
  sorry
end

end sum_of_solutions_of_quadratic_eq_l490_490812


namespace sum_of_radii_greater_than_incircle_radius_l490_490642

variable (A B C D E : Point)
variable (AC BC AB : ℝ) -- sides of the triangle
variable (r r1 r2 : ℝ) -- radii of the in circles 
variable (p p1 p2 : ℝ) -- semiperimeters of the triangles
variable (S S1 S2 : ℝ) -- areas of the triangles

-- Let's assume all the necessary conditions
axiom acute_triangle : AcuteTriangle A B C
axiom circle1_touches_AC_BC : touchesCircle AC BC D r1
axiom circle2_touches_AB_BC : touchesCircle AB BC E r2
axiom incircle_radius : InCircleRadius A B C r

-- The theorem statement
theorem sum_of_radii_greater_than_incircle_radius :
  r1 + r2 > r := by
  sorry

end sum_of_radii_greater_than_incircle_radius_l490_490642


namespace intersection_property_l490_490291

variables {A B C P Q X Y : Type}

-- Definitions for the problem conditions
-- Assume "triangle ABC", "cevians AP and AQ are symmetric with respect to the angle bisector", and "points X and Y symmetric with respect to the angle bisector".

structure triangle (A B C : Type) : Type :=
(angle_bisector_symm : ∀ {AP AQ : Type}, symmetric_about_angle_bisector AP AQ)

-- Prove intersection property
theorem intersection_property (ABC : triangle A B C) :
  intersects_on_line ABC.ABC X P Q :=
sorry

end intersection_property_l490_490291


namespace geometric_sequence_term_number_l490_490648

theorem geometric_sequence_term_number 
  (a_n : ℕ → ℝ)
  (a1 : ℝ) (q : ℝ) (n : ℕ)
  (h1 : a1 = 1/2)
  (h2 : q = 1/2)
  (h3 : a_n n = 1/32)
  (h4 : ∀ n, a_n n = a1 * (q^(n-1))) :
  n = 5 := 
by
  sorry

end geometric_sequence_term_number_l490_490648


namespace sibling_pairs_count_l490_490902

theorem sibling_pairs_count :
  ∃ S : ℕ, S = 30 ∧
    let business_students := 500
    let law_students := 800
    let probability := (7.5 * 10 ^ (-5) : ℝ)
    let total_pairs := business_students * law_students
    probability = S / total_pairs :=
begin
  sorry
end

end sibling_pairs_count_l490_490902


namespace sum_of_solutions_eq_9_l490_490829

theorem sum_of_solutions_eq_9 :
  let roots := {x : ℝ | x^2 = 9 * x - 20}
  in ∑ x in roots, x = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490829


namespace average_age_increase_l490_490764

-- Define the conditions
def students : ℕ := 30
def avg_29_students : ℕ := 12
def age_30th_student : ℕ := 80

-- Define the theorem to prove the increase in average age
theorem average_age_increase :
  let total_age_29 := 29 * avg_29_students in
  let total_age_all := total_age_29 + age_30th_student in
  let new_avg_age := total_age_all / students in
  let original_avg_age := avg_29_students in
  let increase := new_avg_age - original_avg_age in
  increase = 2.27 :=
sorry

end average_age_increase_l490_490764


namespace polar_to_rectangular_correct_l490_490574

-- Define variables and constants
def rho : ℝ := 5
def theta : ℝ := 2 * Real.pi / 3

-- Define the rectangular coordinates
def x : ℝ := rho * Real.cos theta
def y : ℝ := rho * Real.sin theta

-- State the theorem
theorem polar_to_rectangular_correct :
  (x, y) = (-5 / 2, 5 * Real.sqrt 3 / 2) :=
by
  -- Proof will go here
  sorry

end polar_to_rectangular_correct_l490_490574


namespace two_box_even_sum_probability_l490_490972

theorem two_box_even_sum_probability : 
  let chips := {1, 2, 4}
  let draws := { (a, b) | a ∈ chips ∧ b ∈ chips }
  let even_sum := { (a, b) ∈ draws | (a + b) % 2 = 0 }
  (|even_sum| : ℚ) / (|draws| : ℚ) = 5 / 9 :=
by
  have chips_def : chips = {1, 2, 4} := rfl
  have draws_def : draws = { (a, b) | a ∈ chips ∧ b ∈ chips } := rfl
  have even_sum_def : even_sum = { (a, b) | (a + b) % 2 = 0 } := rfl
  sorry

end two_box_even_sum_probability_l490_490972


namespace count_correct_statements_l490_490011

-- Define the conditions as propositions
def condition1 : Prop := ∀ flowchart, flowchart.has_start_symbol ∧ flowchart.has_end_symbol
def condition2 : Prop := ∀ flowchart, flowchart.input_after_start ∧ flowchart.output_before_end
def condition3 : Prop := ∀ symbol, symbol.is_decision_symbol → symbol.has_more_than_one_exit
def condition4 : Prop := ∀ program1 program2, (program1.decision_condition ≠ program2.decision_condition) 

-- Define the statements based on conditions
def statement1 : Prop := condition1
def statement2 : Prop := condition2
def statement3 : Prop := condition3
def statement4 : Prop := ¬ condition4

-- Define the problem as proving the number of true statements
theorem count_correct_statements : 
  ((statement1 = true) + (statement2 = true) + (statement3 = true) + (statement4 = true) = 3) :=
by sorry

end count_correct_statements_l490_490011


namespace green_tetrahedron_volume_l490_490085

def side_length (s : ℕ) := s = 8
def alternately_colored_vertices {C : Type} [cube : C]
  (red green : C → Prop) := ∀ (v w : C), adjacent v w → ((red v ∧ green w) ∨ (green v ∧ red w))
def tetrahedron_volume (s : ℕ) := (s ^ 3) / 3 

theorem green_tetrahedron_volume :
  ∀ (C : Type) [cube : C] (red green : C → Prop) 
  (v1 v2 v3 v4 : C),
  side_length 8 →
  alternately_colored_vertices red green →
  green v1 → green v2 → green v3 → green v4 → 
  tetrahedron_volume 8 v1 v2 v3 v4 = 171 :=
by
  sorry

end green_tetrahedron_volume_l490_490085


namespace find_treasure_l490_490660

theorem find_treasure :
  ∃ i : ℕ, i = 2 ∧
  (i = 2 → (((1 ∨ 4) ≡ 0) ∧ ((1 → 2) ≡ 0) ∧ ((3 ∨ 5) ≡ 0) ∧ (¬((i = 4) ∧ ¬(¬(1 ∨ (4 ∧ 5)))))
  ∧ ¬((2 ∨ 3) ∧ ¬((¬ (i = 4 ∧ 5) ≡ 0) ∧ ((¬ (i = 5 ∧ 4)) ≡ ¬(i = 4 ∧ 3))))) ∧
    (2 = 1 ↔ true) ∧ (3 = 2 ↔ true) ∧ (5 = 4 ↔ ¬ true) →
  ∃ f : ℕ → ℕ, f 1 = 2 ∧ f 2 = 2 ∧ f 3 = 2 ∧ f 4 = 4 ∧ f 5 ≠ 5) :=
by
  sorry

end find_treasure_l490_490660


namespace base_eight_to_ten_l490_490423

theorem base_eight_to_ten (n : Nat) (h : n = 52) : 8 * 5 + 2 = 42 :=
by
  -- Proof will be written here.
  sorry

end base_eight_to_ten_l490_490423


namespace average_carnations_l490_490411

theorem average_carnations :
  let a : List ℝ := [9, 23, 13, 36, 28, 45, 18.5, 30, 22.5, 15, 12.75, 39] in
  (a.sum / a.length) = 24.396 :=
by
  sorry

end average_carnations_l490_490411


namespace minimum_triangle_area_l490_490762

noncomputable def triangle_least_possible_area : ℂ :=
  let z := (2: ℂ)^(1/2) * complex.exp((2 * real.pi * complex.I) / 12)
  let D := z * complex.cos 0, z * complex.sin 0 in
  let E := z * complex.cos (real.pi / 6), z * complex.sin (real.pi / 6) in
  let F := z * complex.cos (real.pi / 3), z * complex.sin (real.pi / 3) in
  let base := complex.abs(z * (complex.cos (real.pi / 6) - complex.cos (real.pi / 3))) in
  let height := complex.abs(z * complex.sin (real.pi / 3)) in
  (1/2 : ℂ) * base * height

theorem minimum_triangle_area : triangle_least_possible_area = (real.sqrt 6 * (real.sqrt 3 - 1)) / 4 := by
  sorry

end minimum_triangle_area_l490_490762


namespace ratio_of_areas_l490_490457

noncomputable def area_ratio (A B C D E F A' B' C' : Point) (plane : Plane) : ℝ :=
  if intersects plane D A = A' ∧ 
     intersects plane D B = B' ∧ 
     intersects plane D C = C' ∧ 
     intersects plane A B = E ∧ 
     intersects plane A C = F ∧ 
     ratio (D, A') (A', A) = 5 / 1 then
    area (triangle A E F) / area (triangle A B C)
  else
    0

theorem ratio_of_areas 
  (A B C D E F A' B' C' : Point) (plane : Plane)
  (h1 : intersects plane D A = A')
  (h2 : intersects plane D B = B')
  (h3 : intersects plane D C = C')
  (h4 : intersects plane A B = E)
  (h5 : intersects plane A C = F)
  (h6 : ratio (D, A') (A', A) = 5 / 1)
  :
  area_ratio A B C D E F A' B' C' plane = 1 / 576 :=
  sorry

end ratio_of_areas_l490_490457


namespace drivers_sufficient_l490_490921

theorem drivers_sufficient
  (round_trip_duration : ℕ := 320)
  (rest_duration : ℕ := 60)
  (return_time_A : ℕ := 12 * 60 + 40)
  (depart_time_D : ℕ := 13 * 60 + 5)
  (next_depart_time_A : ℕ := 13 * 60 + 40)
  : (4 : ℕ) ∧ (21 * 60 + 30 = 1290) := 
  sorry

end drivers_sufficient_l490_490921


namespace percentage_reduction_l490_490871

variable (P S : ℝ) -- original price and sales
variable (x : ℝ) -- percentage reduction in price
variable h1 : S * 1.80 = S * (1.26 / (1 - x / 100))
variable h2 : x = 30

-- Prove that the percentage reduction in the price of the article is 30%
theorem percentage_reduction (h3 : h2) : x = 30 := by
  sorry

end percentage_reduction_l490_490871


namespace dice_probability_l490_490940

-- Defining the problem conditions
def even_number_probability : ℝ := 1 / 2
def odd_number_probability : ℝ := 1 / 2

/-- The probability that exactly three out of five fair 6-sided dice show an even number is 5/16 -/
theorem dice_probability (n : ℕ) (k : ℕ) (p : ℝ) (X : ℕ → Prop) (h : ∀ i, i < n → (X i = true) ↔ (i < k)) :
  ∑ i in finset.range 6, p = 1 → 
  (∀ i, i < n → (X i = true) → i < 6) →
  (∑ i in finset.range n, X i = k) →
  (∏ i in finset.range n, if X i then p else (1 - p)) = (5/16) :=
by sorry

end dice_probability_l490_490940


namespace general_formula_l490_490202

open Nat

def sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, ((∑ i in range (n + 1), a i) = (n + 1) * a n / 2) 

def Sn (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in range (n + 1), a i

def geometric_seq (a1 k : ℕ) (Sk2 : ℕ) : Prop :=
  a1 * Sk2 = a1^2 * k^2

def Tn (a : ℕ → ℕ) (Sn : ℕ → ℕ) (n : ℕ) : ℝ := ∑ i in range (n + 1), 1 / Sn i

theorem general_formula (a : ℕ → ℕ) (Sn : ℕ → ℕ) (Tn : ℕ → ℝ) : 
  (sequence a) ∧ (a 1 = 1) ∧ (∀ n, Sn n = (n * (n + 1)) / 2) ∧ 
  (∃ k, 1 < k ∧ geometric_seq (a 1) k (Sn (k + 2)) ∧ k = 6) ∧ 
  (∀ n, Tn a Sn n < 2)
:=
begin
  -- Proof omitted
  sorry
end

end general_formula_l490_490202


namespace exists_monic_polynomial_degree_58_with_conditions_l490_490339

theorem exists_monic_polynomial_degree_58_with_conditions :
  ∃ (P : Polynomial ℝ), P.degree = 58 ∧ P.leading_coeff = 1 ∧
  (∃ positive_roots negative_roots : Finset ℝ,
    positive_roots.card = 29 ∧
    negative_roots.card = 29 ∧
    ∀ x ∈ positive_roots, P.xeval x = 0 ∧ x > 0 ∧
    ∀ x ∈ negative_roots, P.xeval x = 0 ∧ x < 0) ∧
  (∀ k ∈ Finset.range 59, ∃ n : ℕ, real.log (abs (P.coeff k)) / real.log 2017 = n) :=
sorry

end exists_monic_polynomial_degree_58_with_conditions_l490_490339


namespace first_person_amount_l490_490734

theorem first_person_amount (A B C : ℕ) (h1 : A = 28) (h2 : B = 72) (h3 : C = 98) (h4 : A + B + C = 198) (h5 : 99 ≤ max (A + B) (B + C) / 2) : 
  A = 28 :=
by
  -- placeholder for proof
  sorry

end first_person_amount_l490_490734


namespace find_colored_copies_l490_490134

variable (cost_c cost_w total_copies total_cost : ℝ)
variable (colored_copies white_copies : ℝ)

def colored_copies_condition (cost_c cost_w total_copies total_cost : ℝ) :=
  ∃ (colored_copies white_copies : ℝ),
    colored_copies + white_copies = total_copies ∧
    cost_c * colored_copies + cost_w * white_copies = total_cost

theorem find_colored_copies :
  colored_copies_condition 0.10 0.05 400 22.50 → 
  ∃ (c : ℝ), c = 50 :=
by 
  sorry

end find_colored_copies_l490_490134


namespace domain_of_f_l490_490502

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (2 * x - 4) + Mathlib.cbrt (2 * x - 6)

theorem domain_of_f :
  {x : ℝ | 2 * x - 4 ≥ 0} = {x : ℝ | x ≥ 2} :=
by
  sorry

end domain_of_f_l490_490502


namespace recovery_probability_A_le_14_distribution_X_expected_value_of_X_variance_relationship_l490_490041

def groupA : list ℕ := [10, 11, 12, 13, 14, 15, 16]

def groupB : list ℕ := [12, 13, 14, 15, 16, 17, 20]

def recovery_time_of_person_A_le_14 : ℝ := 5 / 7

noncomputable def probability_X_distribution : list (ℕ × ℝ) := [(0, 15 / 49), (1, 26 / 49), (2, 8 / 49)]

noncomputable def expected_value_X : ℝ := 6 / 7

theorem recovery_probability_A_le_14 :
  (5 : ℝ) / 7 = recovery_time_of_person_A_le_14 :=
sorry

theorem distribution_X :
  probability_X_distribution = 
  [(0, 15 / 49), (1, 26 / 49), (2, 8 / 49)] :=
sorry

theorem expected_value_of_X :
  expected_value_X = (6 : ℝ) / 7 :=
sorry

theorem variance_relationship :
  D_A < D_B :=
sorry

end recovery_probability_A_le_14_distribution_X_expected_value_of_X_variance_relationship_l490_490041


namespace max_watched_hours_l490_490317

-- Define the duration of one episode in minutes
def episode_duration : ℕ := 30

-- Define the number of weekdays Max watched the show
def weekdays_watched : ℕ := 4

-- Define the total minutes Max watched
def total_minutes_watched : ℕ := episode_duration * weekdays_watched

-- Define the conversion factor from minutes to hours
def minutes_to_hours_factor : ℕ := 60

-- Define the total hours watched
def total_hours_watched : ℕ := total_minutes_watched / minutes_to_hours_factor

-- Proof statement
theorem max_watched_hours : total_hours_watched = 2 :=
by
  sorry

end max_watched_hours_l490_490317


namespace min_time_needed_l490_490077

-- Define the conditions and required time for shoeing horses
def num_blacksmiths := 48
def num_horses := 60
def hooves_per_horse := 4
def time_per_hoof := 5
def total_hooves := num_horses * hooves_per_horse
def total_time_one_blacksmith := total_hooves * time_per_hoof
def min_time (num_blacksmiths : Nat) (total_time_one_blacksmith : Nat) : Nat :=
  total_time_one_blacksmith / num_blacksmiths

-- Prove that the minimum time needed is 25 minutes
theorem min_time_needed : min_time num_blacksmiths total_time_one_blacksmith = 25 :=
by
  sorry

end min_time_needed_l490_490077


namespace total_crayons_lost_or_given_away_l490_490335

-- Define the conditions
def crayons_given_away : ℕ := 213
def crayons_lost : ℕ := 16

-- State the proposition that the total number of crayons lost or given away is 229
theorem total_crayons_lost_or_given_away : crayons_given_away + crayons_lost = 229 :=
by
  rw [crayons_given_away, crayons_lost]
  rfl

end total_crayons_lost_or_given_away_l490_490335


namespace range_of_a_l490_490627

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ax^2 + ax + 3 > 0) ↔ (0 ≤ a ∧ a < 12) :=
by
  sorry

end range_of_a_l490_490627


namespace per_can_price_difference_cents_l490_490080

   theorem per_can_price_difference_cents :
     let bulk_warehouse_price_per_case := 12.0
     let bulk_warehouse_cans_per_case := 48
     let bulk_warehouse_discount := 0.10
     let local_store_price_per_case := 6.0
     let local_store_cans_per_case := 12
     let local_store_promotion_factor := 1.5 -- represents the effect of the promotion (3 cases for the price of 2.5 cases)
     let bulk_warehouse_price_per_can := (bulk_warehouse_price_per_case * (1 - bulk_warehouse_discount)) / bulk_warehouse_cans_per_case
     let local_store_price_per_can := (local_store_price_per_case * local_store_promotion_factor) / (local_store_cans_per_case * 3)
     let price_difference_cents := (local_store_price_per_can - bulk_warehouse_price_per_can) * 100
     price_difference_cents = 19.17 :=
   by
     sorry
   
end per_can_price_difference_cents_l490_490080


namespace range_of_a_l490_490629

theorem range_of_a (a : ℝ) : (∃ x : ℝ, 0 < x ∧ x^2 + 2 * a * x + 2 * a + 3 < 0) ↔ a < -1 :=
sorry

end range_of_a_l490_490629


namespace matrix_mult_correct_l490_490144

-- Definition of matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 1],
  ![4, -2]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![7, -3],
  ![2, 4]
]

-- The goal is to prove that A * B yields the matrix C
def matrix_product : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![23, -5],
  ![24, -20]
]

theorem matrix_mult_correct : A * B = matrix_product := by
  -- Proof omitted
  sorry

end matrix_mult_correct_l490_490144


namespace max_value_f_max_value_f_at_13_l490_490372

noncomputable def f (x : ℝ) : ℝ := |x| / (Real.sqrt (1 + x^2) * Real.sqrt (4 + x^2))

theorem max_value_f : ∀ x : ℝ, f x ≤ 1 / 3 := by
  sorry

theorem max_value_f_at_13 : ∃ x : ℝ, f x = 1 / 3 := by
  sorry

end max_value_f_max_value_f_at_13_l490_490372


namespace work_done_in_one_day_l490_490900

theorem work_done_in_one_day (A_time B_time : ℕ) (hA : A_time = 4) (hB : B_time = A_time / 2) : 
  (1 / A_time + 1 / B_time) = (3 / 4) :=
by
  -- Here we are setting up the conditions as per our identified steps
  rw [hA, hB]
  -- The remaining steps to prove will be omitted as per instructions
  sorry

end work_done_in_one_day_l490_490900


namespace lambda_value_l490_490279

variable (OA OB OC : Vector ℝ)

theorem lambda_value (M A B C : Vector ℝ)
  (h₁ : M = 1 / 2 • A + 1 / 6 • B + λ • C)
  (h₂ : coplanar {M - A, M - B, M - C}) :
  λ = 1 / 3 :=
by sorry

end lambda_value_l490_490279


namespace sum_of_solutions_of_quadratic_eq_l490_490818

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 9 * x + 20 = 0

-- Prove that the sum of the solutions to this equation is 9
theorem sum_of_solutions_of_quadratic_eq : 
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 9) := 
begin
  -- Proof is omitted
  sorry
end

end sum_of_solutions_of_quadratic_eq_l490_490818


namespace quadratic_equation_with_one_variable_is_B_l490_490874

def is_quadratic_equation_with_one_variable (eq : String) : Prop :=
  eq = "x^2 + x + 3 = 0"

theorem quadratic_equation_with_one_variable_is_B :
  is_quadratic_equation_with_one_variable "x^2 + x + 3 = 0" :=
by
  sorry

end quadratic_equation_with_one_variable_is_B_l490_490874


namespace max_minutes_sleep_without_missing_happy_moment_l490_490337

def isHappyMoment (h m : ℕ) : Prop :=
  (h = 4 * m ∨ m = 4 * h) ∧ h < 24 ∧ m < 60

def sleepDurationMax : ℕ :=
  239

theorem max_minutes_sleep_without_missing_happy_moment :
  ∀ (sleepDuration : ℕ), sleepDuration ≤ 239 :=
sorry

end max_minutes_sleep_without_missing_happy_moment_l490_490337


namespace part1_solution_set_eq_part2_a_range_l490_490591

theorem part1_solution_set_eq : {x : ℝ | |2 * x + 1| + |2 * x - 3| ≤ 6} = Set.Icc (-1) 2 :=
by sorry

theorem part2_a_range (a : ℝ) (h : a > 0) : 
  (∃ x : ℝ, |2 * x + 1| + |2 * x - 3| < |a - 2|) → 6 < a :=
by sorry

end part1_solution_set_eq_part2_a_range_l490_490591


namespace count_integer_values_abs_lt_seven_pi_l490_490607

theorem count_integer_values_abs_lt_seven_pi : 
  (finset.Icc (-⌊7*real.pi⌋) ⌊(7*real.pi)⌋).card = 43 := 
begin
  sorry
end

end count_integer_values_abs_lt_seven_pi_l490_490607


namespace sum_of_interior_angles_of_triangle_is_180_l490_490429

theorem sum_of_interior_angles_of_triangle_is_180 : 
  ∀ (A B C : ℝ),
  let α := A
  let β := B
  let γ := C
  interior_angles_of_triangle α β γ →
  α + β + γ = 180 := 
by
  sorry

end sum_of_interior_angles_of_triangle_is_180_l490_490429


namespace cupcakes_per_day_needed_l490_490128

-- Define the given conditions
def total_goal_cupcakes : ℕ := 96
def cupcakes_to_bonnie : ℕ := 24
def days : ℕ := 2

-- State the problem as a Lean theorem
theorem cupcakes_per_day_needed (goal : ℕ) (bonnie : ℕ) (days : ℕ) : ℕ :=
  (goal + bonnie) / days

-- Verify the conditions as true
example : cupcakes_per_day_needed total_goal_cupcakes cupcakes_to_bonnie days = 60 :=
by
  -- The calculation can be checked against the example conditions
  calc
    cupcakes_per_day_needed 96 24 2 = (96 + 24) / 2 : rfl
    ... = 120 / 2 : by norm_num
    ... = 60 : by norm_num

end cupcakes_per_day_needed_l490_490128


namespace sin_double_angle_l490_490609

theorem sin_double_angle (α : ℝ) (h : Real.cos (Real.pi / 4 - α) = 3 / 5) : Real.sin (2 * α) = -7 / 25 :=
  sorry

end sin_double_angle_l490_490609


namespace certain_number_sixth_powers_l490_490929

theorem certain_number_sixth_powers :
  ∃ N, (∀ n : ℕ, n < N → ∃ a : ℕ, n = a^6) ∧
       (∃ m ≤ N, (∀ n < m, ∃ k : ℕ, n = k^6) ∧ ¬ ∃ k : ℕ, m = k^6) :=
sorry

end certain_number_sixth_powers_l490_490929


namespace simplify_expression_l490_490150

theorem simplify_expression (x : ℝ) :
  (7 - Real.sqrt (x^2 - 49))^2 = x^2 - 14 * Real.sqrt (x^2 - 49) :=
sorry

end simplify_expression_l490_490150


namespace park_area_l490_490024

noncomputable def park_length (x : ℕ) : ℕ := 3 * x
noncomputable def park_width (x : ℕ) : ℕ := 2 * x
noncomputable def park_perimeter (x : ℕ) : ℕ := 2 * (park_length x + park_width x)
noncomputable def fence_cost (perimeter : ℕ) (cost_per_meter : ℕ) : ℕ := perimeter * cost_per_meter

theorem park_area (cost_per_fence_meter : ℕ) (total_cost : ℕ) :
  (let x := total_cost / (cost_per_fence_meter * 10) in 
   park_length x * park_width x) = 7350 :=
by
  sorry

end park_area_l490_490024


namespace perpendicular_sufficient_condition_l490_490396

variable (l : Line) (A B C : Point)

-- Conditions
def is_perpendicular_to_sides_ABC (l : Line) (A B C : Point) : Prop :=
  is_perpendicular_to_line l (line_through_points A B) ∧ is_perpendicular_to_line l (line_through_points A C)

def is_perpendicular_to_side_BC (l : Line) (B C : Point) : Prop :=
  is_perpendicular_to_line l (line_through_points B C)

-- Proof Statement
theorem perpendicular_sufficient_condition :
  is_perpendicular_to_sides_ABC l A B C → is_perpendicular_to_side_BC l B C ∧ ¬(is_perpendicular_to_side_BC l B C → is_perpendicular_to_sides_ABC l A B C) :=
by sorry

end perpendicular_sufficient_condition_l490_490396


namespace smallest_n_to_cover_convex_100gon_with_triangles_l490_490524

-- Defining convex 100-gon and triangles
def convex_100gon := { P : Type | P is a convex polygon with 100 sides }
def triangle := { T : Type | T is a triangle }

-- The smallest number n such that any convex 100-gon can be represented as an intersection of n triangles.
theorem smallest_n_to_cover_convex_100gon_with_triangles (n : ℕ) :
  (∀ P : convex_100gon, ∃ (T : fin n → triangle), (⋂ i, T i) = P) ↔ n = 50 :=
sorry

end smallest_n_to_cover_convex_100gon_with_triangles_l490_490524


namespace sum_minimal_area_k_l490_490378

def vertices_triangle_min_area (k : ℤ) : Prop :=
  let x1 := 1
  let y1 := 7
  let x2 := 13
  let y2 := 16
  let x3 := 5
  ((y1 - k) * (x2 - x1) ≠ (x1 - x3) * (y2 - y1))

def minimal_area_sum_k : ℤ :=
  9 + 11

theorem sum_minimal_area_k :
  ∃ k1 k2 : ℤ, vertices_triangle_min_area k1 ∧ vertices_triangle_min_area k2 ∧ k1 + k2 = 20 := 
sorry

end sum_minimal_area_k_l490_490378


namespace minimum_cubes_to_show_only_receptacle_holes_l490_490455

-- Define the properties of the cubes
structure Cube where
  top_snap : Bool
  bottom_snap : Bool
  receptacle_hole : Fin 4 → Bool

-- Define the condition of only showing receptacle holes
def all_receptacle_holes_visible (cubes : List Cube) : Bool :=
  cubes.length = 2 ∧
  cubes.head!.bottom_snap = false ∧
  cubes.head!.top_snap = false ∧
  cubes.tail.head!.bottom_snap = false ∧
  cubes.tail.head!.top_snap = false ∧
  (∀ i, i ∈ [0, 1, 2, 3] → cubes.head!.receptacle_hole i = true) ∧
  (∀ i, i ∈ [0, 1, 2, 3] → cubes.tail.head!.receptacle_hole i = true)

-- The target statement to be proved
theorem minimum_cubes_to_show_only_receptacle_holes :
  ∃ cubes : List Cube, all_receptacle_holes_visible cubes ∧ cubes.length = 2 :=
by
  sorry

end minimum_cubes_to_show_only_receptacle_holes_l490_490455


namespace right_triangle_area_eq_8_over_3_l490_490175

-- Definitions arising from the conditions in the problem
variable (a b c : ℝ)

-- The conditions as Lean definitions
def condition1 : Prop := b = (2/3) * a
def condition2 : Prop := b = (2/3) * c

-- The question translated into a proof problem: proving that the area of the triangle equals 8/3
theorem right_triangle_area_eq_8_over_3 (h1 : condition1 a b) (h2 : condition2 b c) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 8/3 :=
by
  sorry

end right_triangle_area_eq_8_over_3_l490_490175


namespace sheep_population_in_2000_l490_490451

-- Define the conditions related to sheep reproduction and initial purchase
def initial_sheep_count : ℕ := 2
def no_offspring_in_1996 : Prop := true
def first_sheep_birth_rate : ℕ := 3
def second_sheep_birth_rate : ℕ := 2
def annual_birth_rate : ℕ := 1

-- Define a function to model the population growth
noncomputable def sheep_population (year : ℕ) : ℕ :=
  if year < 1996 then 0
  else (match year with
  | _ => 2 -- The function needs to be properly defined to model the given conditions
  end)

-- State the theorem that the sheep population will be 12 at the end of the year 2000
theorem sheep_population_in_2000 :
  sheep_population 2000 = 12 :=
sorry -- proof goes here

end sheep_population_in_2000_l490_490451


namespace product_upto_10_equals_11_l490_490482

-- Define the product
def product_upto (n : ℕ) : ℚ :=
  ∏ k in finset.range n, (1 + 1/(k + 1))

-- Assert the product equals 11 when n = 10
theorem product_upto_10_equals_11 : product_upto 10 = 11 :=
by
  sorry

end product_upto_10_equals_11_l490_490482


namespace true_propositions_l490_490687

variables (α β : Plane) (m n : Line)

-- Conditions
axiom non_coincident_planes : α ≠ β 
axiom non_coincident_lines : m ≠ n

-- Propositions
def prop1 : Prop := (m ∥ n ∧ n ⊆ α) → m ∥ α
def prop2 : Prop := (m ⟂ n ∧ m ⟂ α) → n ∥ α
def prop3 : Prop := (α ⟂ β ∧ α ∩ β = m ∧ n ⊆ α ∧ n ⟂ m) → n ⟂ β
def prop4 : Prop := (m ∥ n ∧ n ⟂ α ∧ α ∥ β) → m ⟂ β

-- Proof statement
theorem true_propositions : prop3 α β m n ∧ prop4 α β m n ∧ ¬prop1 α β m n ∧ ¬prop2 α β m n :=
by
  -- Placeholder for the proof
  sorry

end true_propositions_l490_490687


namespace find_a_value_l490_490580

-- Definitions of conditions
def eq_has_positive_root (a : ℝ) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ (x / (x - 5) = 3 - (a / (x - 5)))

-- Statement of the theorem
theorem find_a_value (a : ℝ) (h : eq_has_positive_root a) : a = -5 := 
  sorry

end find_a_value_l490_490580


namespace det_my_matrix_l490_490122

def my_matrix : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![3, 0, 1], ![-5, 5, -4], ![3, 3, 6]]

theorem det_my_matrix : my_matrix.det = 96 := by
  sorry

end det_my_matrix_l490_490122


namespace sum_of_solutions_of_quadratic_eq_l490_490815

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 9 * x + 20 = 0

-- Prove that the sum of the solutions to this equation is 9
theorem sum_of_solutions_of_quadratic_eq : 
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 9) := 
begin
  -- Proof is omitted
  sorry
end

end sum_of_solutions_of_quadratic_eq_l490_490815


namespace junior_boys_to_junior_girls_ratio_l490_490280

theorem junior_boys_to_junior_girls_ratio (b g jb sb jg sg: ℕ) (hb : b = 0.55 * (b + g))
  (h_ratio : jb / sb = j / s) : jb / jg = 11 / 9 :=
by
  sorry

end junior_boys_to_junior_girls_ratio_l490_490280


namespace average_multiples_of_10_l490_490048

theorem average_multiples_of_10 (a b : ℕ) (h1 : a = 10) (h2 : b = 100) :
  (a + b) / 2 = 55 :=
by
  rw [h1, h2]
  norm_num
  sorry

end average_multiples_of_10_l490_490048


namespace math_problem_l490_490140

noncomputable def product_range : ℝ := 
  ∏ n in (finset.range 462).map (λ n => 100 + 2 * n), (n : ℝ) / (n + 1)

theorem math_problem : product_range < (5 : ℝ) / 16 := by
  sorry

end math_problem_l490_490140


namespace power_function_value_at_half_l490_490587

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2 - x) - 3/4

noncomputable def g (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_value_at_half (a : ℝ) (α : ℝ) 
  (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : f a 2 = 1 / 4) (h4 : g α 2 = 1 / 4) : 
  g α (1/2) = 4 := 
by
  sorry

end power_function_value_at_half_l490_490587


namespace inequality_factorial_l490_490696

theorem inequality_factorial (n k : ℕ) (h1 : 0 < k) (h2 : k < n) : 
  (1 / (n + 1) * (n ^ n) / (k ^ k * (n - k) ^ (n - k)) < nat.factorial n / (nat.factorial k * nat.factorial (n - k)))
  ∧ (nat.factorial n / (nat.factorial k * nat.factorial (n - k)) < n ^ n / (k ^ k * (n - k) ^ (n - k))) :=
by
  sorry

end inequality_factorial_l490_490696


namespace find_a_minus_b_l490_490610

variable (a b : ℚ) (x : ℚ)
def pos_rational (q: ℚ) := q > 0

-- Summarize the conditions as hypotheses
def condition1 := a / (2^x - 1) + b / (2^x + 2) = (2 * 2^x + 1) / ((2^x - 1) * (2^x + 2))
def condition2 := pos_rational x

theorem find_a_minus_b (h1 : condition1) (h2 : condition2) : a - b = 0 :=
by
  sorry

end find_a_minus_b_l490_490610


namespace select_disjoint_cells_l490_490329

theorem select_disjoint_cells (n : ℕ) (cells : set (ℕ × ℕ)) (h : cells.card = n) :
  ∃ (subset : set (ℕ × ℕ)), subset ⊆ cells ∧ subset.card ≥ n / 4 ∧
    ∀ (c1 c2 : ℕ × ℕ), c1 ∈ subset → c2 ∈ subset → c1 ≠ c2 → ¬(adjacent c1 c2) :=
sorry

end select_disjoint_cells_l490_490329


namespace base_arithmetic_problem_l490_490778

def base_convert_8_to_10 (n : ℕ) : ℕ :=
  let digits := [2, 4, 6, 8]
  digits.enum.sum (λ ⟨i, d⟩, d * 8^i)

def base_convert_4_to_10 (n : ℕ) : ℕ :=
  let digits := [1, 1, 0]
  digits.enum.sum (λ ⟨i, d⟩, d * 4^i)

def base_convert_9_to_10 (n : ℕ) : ℕ :=
  let digits := [3, 5, 7, 1]
  digits.enum.sum (λ ⟨i, d⟩, d * 9^i)

def base_convert_10_to_10 (n : ℕ) : ℕ := n

def num1 := base_convert_8_to_10 2468
def num2 := base_convert_4_to_10 110
def num3 := base_convert_9_to_10 3571
def num4 := base_convert_10_to_10 1357

theorem base_arithmetic_problem : (Int.ofNat (num1 / num2).round - Int.ofNat num3 + Int.ofNat num4) = -1232 :=
by
  sorry

end base_arithmetic_problem_l490_490778


namespace smallest_m_exists_l490_490201

theorem smallest_m_exists (p : ℕ) [hp : Fact (Nat.Prime p)] :
  ∃ m : ℕ, 1 ≤ m ∧ m ≤ p - 1 ∧ (∑ k in Finset.range (p - 1), k ^ m) % p = 0 ∧ m = p - 2 := sorry

end smallest_m_exists_l490_490201


namespace prime_sum_square_mod_3_l490_490546

theorem prime_sum_square_mod_3 (p : Fin 100 → ℕ) (h_prime : ∀ i, Nat.Prime (p i)) (h_distinct : Function.Injective p) :
  let N := (Finset.univ : Finset (Fin 100)).sum (λ i => (p i)^2)
  N % 3 = 1 := by
  sorry

end prime_sum_square_mod_3_l490_490546


namespace fraction_simplification_fraction_value_check_l490_490732

noncomputable def simplify_fraction (x y : ℝ) : ℝ :=
  (x^5 + y^5) / (x + y)

theorem fraction_simplification
  (x y : ℝ) (h_not_zero_denominator : x + y ≠ 0) :
  (x^8 + x^6 * y^2 + x^4 * y^4 + x^2 * y^6 + y^8) /
  (x^4 + x^3 * y + x^2 * y^2 + x * y^3 + y^4) = simplify_fraction x y := by
  sorry

theorem fraction_value_check :
  simplify_fraction 0.01 0.02 = 11 * 10^(-8) := by
  sorry

end fraction_simplification_fraction_value_check_l490_490732


namespace y_z_add_x_eq_160_l490_490632

theorem y_z_add_x_eq_160 (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * (y + z) = 132) (h5 : z * (x + y) = 180) (h6 : x * y * z = 160) :
  y * (z + x) = 160 := 
by 
  sorry

end y_z_add_x_eq_160_l490_490632


namespace smallest_N_l490_490465

theorem smallest_N (l m n N : ℕ) (hl : l > 1) (hm : m > 1) (hn : n > 1) :
  (l - 1) * (m - 1) * (n - 1) = 231 → l * m * n = N → N = 384 :=
sorry

end smallest_N_l490_490465


namespace find_value_of_fraction_l490_490684

noncomputable def a : ℝ := 5 * (Real.sqrt 2) + 7

theorem find_value_of_fraction (h : (20 * a) / (a^2 + 1) = Real.sqrt 2) (h1 : 1 < a) : 
  (14 * a) / (a^2 - 1) = 1 := 
by 
  have h_sqrt : 20 * a = Real.sqrt 2 * a^2 + Real.sqrt 2 := by sorry
  have h_rearrange : Real.sqrt 2 * a^2 - 20 * a + Real.sqrt 2 = 0 := by sorry
  have h_solution : a = 5 * (Real.sqrt 2) + 7 := by sorry
  have h_asquare : a^2 = 99 + 70 * (Real.sqrt 2) := by sorry
  exact sorry

end find_value_of_fraction_l490_490684


namespace factor_81_minus_27_x_cubed_l490_490976

theorem factor_81_minus_27_x_cubed (x : ℝ) : 
  81 - 27 * x ^ 3 = 27 * (3 - x) * (9 + 3 * x + x ^ 2) :=
by sorry

end factor_81_minus_27_x_cubed_l490_490976


namespace find_factor_l490_490937

theorem find_factor (f : ℝ) : (120 * f - 138 = 102) → f = 2 :=
by
  sorry

end find_factor_l490_490937


namespace jenny_total_wins_correct_l490_490666

theorem jenny_total_wins_correct :
    ∀ (games_mark games_jill games_sarah games_tom : ℕ)
    (mark_win_pct jill_win_pct sarah_win_pct tom_win_pct : ℚ),
    games_mark = 20 →
    mark_win_pct = 0.25 →
    games_jill = 30 →
    jill_win_pct = 0.65 →
    games_sarah = 25 →
    sarah_win_pct = 0.70 →
    games_tom = 15 →
    tom_win_pct = 0.45 →
    let jenny_wins_mark := games_mark - (mark_win_pct * games_mark).nat_trunc,
        jenny_wins_jill := games_jill - (jill_win_pct * games_jill).nat_trunc,
        jenny_wins_sarah := games_sarah - (sarah_win_pct * games_sarah).nat_trunc,
        jenny_wins_tom := games_tom - (tom_win_pct * games_tom).nat_trunc
    in jenny_wins_mark + jenny_wins_jill + jenny_wins_sarah + jenny_wins_tom = 43 := by
    sorry

end jenny_total_wins_correct_l490_490666


namespace linda_original_savings_l490_490437

theorem linda_original_savings (S : ℝ) (h1 : 3 / 4 * S = 300 + 300) :
  S = 1200 :=
by
  sorry -- The proof is not required.

end linda_original_savings_l490_490437


namespace collinear_and_half_perimeter_triangle_l490_490263

open EuclideanGeometry

variables {A B C A1 B1 C1 D E F G : Point}

-- Definitions based on the conditions
variables (h₁ : A1, B1, C1 are the feet of the altitudes of triangle ABC)
           (h₂ : circums (ba1, ca1) intersect the lines AB, CA, BB1, and CC1 at points D, E, F, and G respectively)

-- Lean 4 statement for the proof problem
theorem collinear_and_half_perimeter_triangle :
  Collinear ({D, E, F, G} : Set Point) ∧ Segment_Length(DE) = (half (Perimeter (PedalTriangle A1 B1 C1))) := 
by
  sorry

end collinear_and_half_perimeter_triangle_l490_490263


namespace find_atomic_weight_of_Na_l490_490990

def atomic_weight_of_Na_is_correct : Prop :=
  ∃ (atomic_weight_of_Na : ℝ),
    (atomic_weight_of_Na + 35.45 + 16.00 = 74) ∧ (atomic_weight_of_Na = 22.55)

theorem find_atomic_weight_of_Na : atomic_weight_of_Na_is_correct :=
by
  sorry

end find_atomic_weight_of_Na_l490_490990


namespace beginning_diving_classes_weekend_day_l490_490355

-- Definitions for the conditions
def weekday_classes_per_day := 2
def weekend_days_per_week := 2
def weeks := 3
def people_per_class := 5
def total_people := 270
def total_weekdays_per_week := 5

-- Theorem statement
theorem beginning_diving_classes_weekend_day :
  let weekday_classes_in_3_weeks := weekday_classes_per_day * total_weekdays_per_week * weeks
  let weekday_people_in_3_weeks := weekday_classes_in_3_weeks * people_per_class
  let weekend_people_in_3_weeks := total_people - weekday_people_in_3_weeks
  let total_weekend_classes_in_3_weeks := weekend_people_in_3_weeks / people_per_class in
  total_weekend_classes_in_3_weeks / (weekend_days_per_week * weeks) = 4 :=
by {
  sorry
}

end beginning_diving_classes_weekend_day_l490_490355


namespace math_problem_l490_490139

noncomputable def product_range : ℝ := 
  ∏ n in (finset.range 462).map (λ n => 100 + 2 * n), (n : ℝ) / (n + 1)

theorem math_problem : product_range < (5 : ℝ) / 16 := by
  sorry

end math_problem_l490_490139


namespace total_hours_difference_l490_490170

-- Definitions based on conditions
def hours_learning_english := 6
def hours_learning_chinese := 2
def hours_learning_spanish := 3
def hours_learning_french := 1

-- Calculation of total time spent on English and Chinese
def total_hours_english_chinese := hours_learning_english + hours_learning_chinese

-- Calculation of total time spent on Spanish and French
def total_hours_spanish_french := hours_learning_spanish + hours_learning_french

-- Calculation of the difference in hours spent
def hours_difference := total_hours_english_chinese - total_hours_spanish_french

-- Statement to prove
theorem total_hours_difference : hours_difference = 4 := by
  sorry

end total_hours_difference_l490_490170


namespace points_not_in_plane_l490_490551

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def in_plane (p M n : Point) : Prop :=
  (p.x - M.x) * n.x + (p.y - M.y) * n.y + (p.z - M.z) * n.z = 0

theorem points_not_in_plane :
  let M  := Point.mk 1 (-1) 1
  let n  := Point.mk 4 (-1) 0
  let B  := Point.mk (-2) 0 1
  let C  := Point.mk (-4) 4 0
  let D  := Point.mk 3 (-3) 4
  ¬ in_plane B M n ∧
  ¬ in_plane C M n ∧
  ¬ in_plane D M n :=
by {
  let M := Point.mk 1 (-1) 1,
  let n := Point.mk 4 (-1) 0,
  let B := Point.mk (-2) 0 1,
  let C := Point.mk (-4) 4 0,
  let D := Point.mk 3 (-3) 4,
  sorry
}

end points_not_in_plane_l490_490551


namespace gcd_of_polynomial_and_multiple_of_12321_l490_490737

theorem gcd_of_polynomial_and_multiple_of_12321 (k : ℤ) :
  let x := 12321 * k,
      g := (3 * x + 4) * (5 * x + 1) * (11 * x + 6) * (x + 11) in
  Int.gcd g x = 3 := by
sorry

end gcd_of_polynomial_and_multiple_of_12321_l490_490737


namespace sum_of_solutions_l490_490852

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | x^2 = 9*x - 20}, x) = 9 := 
sorry

end sum_of_solutions_l490_490852


namespace boys_needed_to_change_ratio_l490_490272

variables (x B G : ℕ)

theorem boys_needed_to_change_ratio (h1 : B + G = 48)
                                   (h2 : B * 5 = G * 3)
                                   (h3 : (B + x) * 3 = 5 * G) :
                                   x = 32 :=
by sorry

end boys_needed_to_change_ratio_l490_490272


namespace tangent_line_at_zero_range_of_a_l490_490234

-- Part 1: Tangent Line at x = 0 when a = 1
theorem tangent_line_at_zero (a : ℝ) (h_a : a = 1) :
  ∀ x : ℝ, (f : ℝ → ℝ) := λ x, exp x - a * sin x - 1,
  f' : ℝ → ℝ := λ x, exp x - cos x,
  f 0 = 0 ∧ f' 0 = 0 → ∀ y : ℝ, y = 0 :=
sorry

-- Part 2: f(x) ≥ 0 in [0, 1) implies a ≤ 1
theorem range_of_a (f : ℝ → ℝ) (h_f : ∀ x ∈ Ico 0 1, f x ≥ 0) :
  ∀ a : ℝ, ∀ x ∈ Ico 0 1, (f := λ x, exp x - a * sin x - 1) → a ≤ 1 :=
sorry

end tangent_line_at_zero_range_of_a_l490_490234


namespace find_diameters_l490_490359

theorem find_diameters (x y z : ℕ) (hx : x ≠ y) (hy : y ≠ z) (hz : x ≠ z) :
  x + y + z = 26 ∧ x^2 + y^2 + z^2 = 338 :=
  sorry

end find_diameters_l490_490359


namespace sum_of_solutions_l490_490791

-- Define the quadratic equation and variable x
def quadratic_equation := ∀ x : ℝ, (x^2 - 9 * x + 20 = 0)

-- Define what we need to prove
theorem sum_of_solutions : ∃ s : ℝ, (∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → s = x1 + x2) ∧ s = 9 :=
by
  sorry -- Proof is omitted

end sum_of_solutions_l490_490791


namespace find_q_l490_490151

theorem find_q (p q d : ℝ) (h₁ : (-p / 3) = q) (h₂ : 1 + p + q + d = q) (h₃ : d = 7) : q = 8 / 3 :=
by
  sorry

end find_q_l490_490151


namespace factor_expr_l490_490981

theorem factor_expr (x : ℝ) : 81 - 27 * x^3 = 27 * (3 - x) * (9 + 3 * x + x^2) := 
sorry

end factor_expr_l490_490981


namespace sum_of_digits_10a_minus_74_single_digit_l490_490257

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_10a_minus_74_single_digit (a : ℕ) (h_pos : a > 0) 
  (h_single_digit : sum_of_digits (10^a - 74) < 10) : a = 2 := by
  sorry

end sum_of_digits_10a_minus_74_single_digit_l490_490257


namespace total_profit_l490_490770

-- Define the investments and their durations
def Tom_investment : ℝ := 3000
def Jose_investment : ℝ := 4500
def Tom_duration : ℝ := 12
def Jose_duration : ℝ := 10

-- Define Jose's share of the profit
def Jose_share : ℝ := 3000

-- The main theorem we want to prove
theorem total_profit (T_invest : ℝ) (J_invest : ℝ) (T_months : ℝ) (J_months : ℝ) (J_profit : ℝ) :
  T_invest = 3000 → 
  J_invest = 4500 → 
  T_months = 12 → 
  J_months = 10 → 
  J_profit = 3000 →
  (let Tom_total := T_invest * T_months;
       Jose_total := J_invest * J_months;
       ratio := Tom_total / Jose_total;
       Jose_part := J_profit / (Jose_total / (Jose_total + Tom_total));
       Tom_part := ratio * Jose_part) in
  Tom_part + J_profit = 5400 :=
begin
  intros h1 h2 h3 h4 h5,
  simp [h1, h2, h3, h4, h5],
  sorry
end

end total_profit_l490_490770


namespace valid_coloring_exists_iff_n_is_odd_l490_490637

theorem valid_coloring_exists_iff_n_is_odd (n : ℕ) (P : Type) [convex_ngon P] 
  (coloring : (edges P) → fin n) : 
  (∀ (c1 c2 c3 : fin n), c1 ≠ c2 → c2 ≠ c3 → c1 ≠ c3 →
  ∃ (a b c : vertices P), 
  adjacent a b ∧ adjacent b c ∧ adjacent c a ∧ 
  coloring (edge a b) = c1 ∧ coloring (edge b c) = c2 ∧ coloring (edge c a) = c3)
  ↔ odd n :=
begin
  sorry -- Proof is not required.
end

end valid_coloring_exists_iff_n_is_odd_l490_490637


namespace minimum_straight_cuts_needed_l490_490661

-- Definitions of the problem
def pancake_radius : ℝ := 10
def coin_radius : ℝ := 1
def strip_width : ℝ := 2

theorem minimum_straight_cuts_needed : 
  ∀ (cuts : ℕ), (cuts * strip_width < 2 * pancake_radius → cuts < 10) := 
by
  intro cuts
  intro h
  have h_diameter : 2 * pancake_radius = 20 := by norm_num
  have h_strip_coverage : cuts * strip_width < 20 := h
  sorry

end minimum_straight_cuts_needed_l490_490661


namespace flour_needed_l490_490106

theorem flour_needed (flour_per_40_cookies : ℝ) (cookies : ℕ) (desired_cookies : ℕ) (flour_needed : ℝ) 
  (h1 : flour_per_40_cookies = 3) (h2 : cookies = 40) (h3 : desired_cookies = 100) :
  flour_needed = 7.5 :=
by
  sorry

end flour_needed_l490_490106


namespace black_white_tile_ratio_l490_490149

theorem black_white_tile_ratio :
  let original_black_tiles := 10
  let original_white_tiles := 15
  let total_tiles_in_original_square := original_black_tiles + original_white_tiles
  let side_length_of_original_square := Int.sqrt total_tiles_in_original_square -- this should be 5
  let side_length_of_extended_square := side_length_of_original_square + 2
  let total_black_tiles_in_border := 4 * (side_length_of_extended_square - 1) / 2 -- Each border side starts and ends with black
  let total_white_tiles_in_border := (side_length_of_extended_square * 4 - 4) - total_black_tiles_in_border 
  let new_total_black_tiles := original_black_tiles + total_black_tiles_in_border
  let new_total_white_tiles := original_white_tiles + total_white_tiles_in_border
  (new_total_black_tiles / gcd new_total_black_tiles new_total_white_tiles) / 
  (new_total_white_tiles / gcd new_total_black_tiles new_total_white_tiles) = 26 / 23 :=
by
  sorry

end black_white_tile_ratio_l490_490149


namespace count_possible_third_side_lengths_l490_490204

theorem count_possible_third_side_lengths : ∀ (n : ℤ), 2 < n ∧ n < 14 → ∃ s : Finset ℤ, s.card = 11 ∧ ∀ x ∈ s, 2 < x ∧ x < 14 := by
  sorry

end count_possible_third_side_lengths_l490_490204


namespace zero_point_interval_l490_490173

noncomputable def f (x : ℝ) : ℝ := log x - 3 / x

theorem zero_point_interval : ∃ c ∈ Ioo 2 3, f c = 0 :=
begin
  -- Function definition and domain condition
  have h1 : ∀ x ∈ Ioo 0 (1:ℝ) ∪ Ioo (1:ℝ) (2:ℝ) ∪ Ioo (2:ℝ) (3:ℝ) ∪ Ioo (3:ℝ) (∞), f x = log x - 3 / x :=
    by assume x hx; simp [f],
  -- Using intervals and the Intermediate Value Theorem
  have h2 : f 2 < 0 ∧ f 3 > 0 :=
    by {simp [f]; linarith [log_two, log_three]},
  let I := Ioo 2 3,
  exact exists_interval_zero f I h2.1 h2.2,
end

end zero_point_interval_l490_490173


namespace ages_l490_490452

-- Definitions of ages
variables (S M : ℕ) -- S: son's current age, M: mother's current age

-- Given conditions
def father_age : ℕ := 44
def son_father_relationship (S : ℕ) : Prop := father_age = S + S
def son_mother_relationship (S M : ℕ) : Prop := (S - 5) = (M - 10)

-- Theorem to prove the ages
theorem ages (S M : ℕ) (h1 : son_father_relationship S) (h2 : son_mother_relationship S M) :
  S = 22 ∧ M = 27 :=
by 
  sorry

end ages_l490_490452


namespace inscribed_triangle_area_l490_490469

noncomputable def triangle_area (r : ℝ) (A B C : ℝ) : ℝ :=
  (1 / 2) * r^2 * (Real.sin A + Real.sin B + Real.sin C)

theorem inscribed_triangle_area :
  ∀ (r : ℝ), r = 12 / Real.pi →
  ∀ (A B C : ℝ), A = 40 * Real.pi / 180 → B = 80 * Real.pi / 180 → C = 120 * Real.pi / 180 →
  triangle_area r A B C = 359.4384 / Real.pi^2 :=
by
  intros
  unfold triangle_area
  sorry

end inscribed_triangle_area_l490_490469


namespace find_time_when_acceleration_is_10_l490_490262

theorem find_time_when_acceleration_is_10 (t : ℝ) :
  (∃ (s : ℝ → ℝ), s = (λ t, 1/3 * t^3 - 3*t^2 + 9*t) ∧ 
  (∀ t, (s t)'' = 2*t - 6) ∧ 
  (2*t - 6 = 10)) → t = 8 :=
by sorry

end find_time_when_acceleration_is_10_l490_490262


namespace rolls_combinations_l490_490446

theorem rolls_combinations (x1 x2 x3 : ℕ) (h1 : x1 + x2 + x3 = 2) : 
  (Nat.choose (2 + 3 - 1) (3 - 1) = 6) :=
by
  sorry

end rolls_combinations_l490_490446


namespace austin_needs_to_polish_l490_490478

theorem austin_needs_to_polish (pairs_of_shoes : ℕ) (percentage_polished : ℚ)
  (h_paired_shoes : pairs_of_shoes = 10)
  (h_percentage : percentage_polished = (45 / 100)) :
  let individual_shoes := 2 * pairs_of_shoes in
  let polished_shoes := percentage_polished * individual_shoes in
  (individual_shoes - polished_shoes) = 11 := 
by
  sorry

end austin_needs_to_polish_l490_490478


namespace chord_length_l490_490756

noncomputable def circle_eq : (ℝ × ℝ) → ℝ := λ p, (p.1)^2 + (p.2)^2 - 8 * p.1 - 2 * p.2 + 1

noncomputable def line_eq : (ℝ × ℝ) → ℝ := λ p, p.2 - (Real.sqrt 3 * p.1 + 1)

theorem chord_length 
  (C : ℝ × ℝ) (r : ℝ) 
  (hC : C = (4, 1)) (hr : r = 4)
  (h_circle : circle_eq C = 0)
  (y_eq : line_eq C = 0) :
  ∃ A B : ℝ × ℝ, line_eq A = 0 ∧ line_eq B = 0 ∧ circle_eq A = 0 ∧ circle_eq B = 0 ∧ dist A B = 4 := 
  sorry

end chord_length_l490_490756


namespace remainder_division_l490_490991

theorem remainder_division (x : ℝ) :
  (x ^ 2021 + 1) % (x ^ 12 - x ^ 9 + x ^ 6 - x ^ 3 + 1) = -x ^ 4 + 1 :=
sorry

end remainder_division_l490_490991


namespace parallel_MN_AB_l490_490718

variables {A B C D M N : Type*}
variables [geometry A] [geometry B] [geometry C] [geometry D] [geometry M] [geometry N]
variables {ABCD : parallelogram A B C D}
variables (insideM : is_inside M ABCD) (insideN : is_inside N (triangle A M D))
variables (angle_cond1 : ∠M N A + ∠M C B = 180) (angle_cond2 : ∠M N D + ∠M B C = 180)

theorem parallel_MN_AB : parallel (line M N) (line A B) :=
by
  sorry

end parallel_MN_AB_l490_490718


namespace sum_smallest_largest_prime_l490_490295

  def primes_between_1_and_40 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

  theorem sum_smallest_largest_prime : 
    let smallest_prime := List.head primes_between_1_and_40
    let largest_prime := List.last primes_between_1_and_40

    smallest_prime + largest_prime = 39 := by
  sorry
  
end sum_smallest_largest_prime_l490_490295


namespace drivers_sufficient_l490_490923

theorem drivers_sufficient
  (round_trip_duration : ℕ := 320)
  (rest_duration : ℕ := 60)
  (return_time_A : ℕ := 12 * 60 + 40)
  (depart_time_D : ℕ := 13 * 60 + 5)
  (next_depart_time_A : ℕ := 13 * 60 + 40)
  : (4 : ℕ) ∧ (21 * 60 + 30 = 1290) := 
  sorry

end drivers_sufficient_l490_490923


namespace probability_Y_eq_2_l490_490702

noncomputable theory
open ProbabilityTheory

variable (p : ℝ)
variable (X : Ω → ℕ)
variable (Y : Ω → ℕ)

axiom h1 : X ∼ binomial 2 p
axiom h2 : Y ∼ binomial 3 p
axiom h3 : P(X ≥ 1) = 5 / 9

theorem probability_Y_eq_2 : P(Y = 2) = 2 / 9 := sorry

end probability_Y_eq_2_l490_490702


namespace count_ordered_triples_l490_490250

def lcm (a b : ℕ) : ℕ := sorry

theorem count_ordered_triples : 
  ∃ (f : ℕ × ℕ × ℕ → Prop), 
  (∀ x y z : ℕ, f (x, y, z) ↔ 
  x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  lcm x y = 180 ∧ 
  lcm x z = 450 ∧ 
  lcm y z = 1200) ∧ 
  (finset.card (finset.filter f ((finset.range 500) ×ˢ (finset.range 500) ×ˢ (finset.range 500)) = 2) :=
begin
  sorry
end

end count_ordered_triples_l490_490250


namespace product_inequality_l490_490142

theorem product_inequality :
  let A := (List.prod (List.map (λ n, (n / (n + 1))) (List.range' 100 923))) 
  in A < (5 / 16) :=
by
  sorry

end product_inequality_l490_490142


namespace number_of_ordered_pairs_l490_490521

theorem number_of_ordered_pairs : 
  {s : Set (ℝ × ℝ) | ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a = b^3 ∧ b = a^3}.finite.toFinset.card = 2 :=
by
  sorry

end number_of_ordered_pairs_l490_490521


namespace magician_performance_reappearances_l490_490104

theorem magician_performance_reappearances :
  let avg_audience := 275
  let disappear_per_50 := λ audience_size, audience_size / 50
  let disappearances := disappear_per_50 avg_audience
  let non_reappear_prob := 1 / 10
  let twice_reappear_prob := 1 / 5
  let thrice_reappear_prob := 1 / 20
  let num_shows := 100
  let num_non_reappear := num_shows * non_reappear_prob
  let num_twice_reappear := num_shows * twice_reappear_prob
  let num_thrice_reappear := num_shows * thrice_reappear_prob
  let base_reappearances := (num_shows * disappearances) - num_non_reappear
  let total_extra_reappearance := (num_twice_reappear * disappearances) + (num_thrice_reappear * (disappearances * 2))
in base_reappearances + total_extra_reappearance = 640 := 
by
  -- Based on the problem's conditions, the calculation confirms the total number of reappearances.
  let avg_audience := 275
  let disappear_per_50 := λ audience_size, audience_size / 50
  let disappearances := disappear_per_50 avg_audience
  let num_shows := 100
  let base_reappearances := (num_shows * disappearances)
  let num_non_reappear := num_shows / 10
  let num_twice_reappear := num_shows / 5
  let num_thrice_reappear := num_shows / 20
  let total_extra_reappearance := (num_twice_reappear * disappearances) + (num_thrice_reappear * (disappearances * 2))
  let total_reappeared := base_reappearances + total_extra_reappearance - num_non_reappear
  have h1 := (5 * 100) + ((20 * 5) + (5 * 10)) - 10
  have h2 := total_reappeared
  exact h1

end magician_performance_reappearances_l490_490104


namespace relationship_between_abc_l490_490950

theorem relationship_between_abc (u v a b c : ℝ)
  (h1 : u - v = a) 
  (h2 : u^2 - v^2 = b)
  (h3 : u^3 - v^3 = c) : 
  3 * b ^ 2 + a ^ 4 = 4 * a * c :=
sorry

end relationship_between_abc_l490_490950


namespace sum_of_solutions_eq_9_l490_490811

theorem sum_of_solutions_eq_9 (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20) :
  let (sum_roots : ℝ) := -b / a in 
  sum_roots = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490811


namespace minimum_value_of_c_l490_490630

open Real

noncomputable
def question (a b c : ℝ) (log_base2 : ℝ → ℝ) :=
  2^a + 4^b = 2^c ∧ 4^a + 2^b = 4^c

noncomputable
def answer (c : ℝ) (log_base2 : ℝ → ℝ) :=
  c = log_base2 3 - 5 / 3

theorem minimum_value_of_c (a b c : ℝ) (log_base2 : ℝ → ℝ)
  (h : question a b c log_base2) : answer c log_base2 :=
sorry

end minimum_value_of_c_l490_490630


namespace divisible_by_five_l490_490649

theorem divisible_by_five (a b : ℕ) (h₀ : 0 ≤ a ∧ a ≤ 9)
  (h₁ : 0 ≤ b ∧ b ≤ 9) (h₂ : a * b = 15) : ∃ k : ℕ, 110 * 1000 + a * 100 + b * 10 ∗ 1 = k * 5 :=
by
  sorry

end divisible_by_five_l490_490649


namespace train_distance_30_minutes_l490_490453

theorem train_distance_30_minutes (h : ∀ (t : ℝ), 0 < t → (1 / 2) * t = 1 / 2 * t) : 
  (1 / 2) * 30 = 15 :=
by
  sorry

end train_distance_30_minutes_l490_490453


namespace allocation_schemes_l490_490971

theorem allocation_schemes (students venues : ℕ) (H_students : students = 4) (H_venues : venues = 3) (H_nonempty : ∀ v, v < 3 → ∃ s, s < 4 ∧ ∃ v' : fin 3, v' = v) :
  ∃ allocation_schemes, allocation_schemes = 36 :=
by {
  -- Omitted proof
  sorry,
}

end allocation_schemes_l490_490971


namespace contradiction_example_l490_490413

theorem contradiction_example (a b c d : ℝ) 
(h1 : a + b = 1) 
(h2 : c + d = 1) 
(h3 : ac + bd > 1) : 
¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) :=
by 
  sorry

end contradiction_example_l490_490413


namespace prime_pairs_int_fraction_l490_490159

theorem prime_pairs_int_fraction (p n : ℕ) (h_prime : Nat.Prime p) (h_pos : n > 0) :
  Nat.divides (p^n + 1) (n^p + 1) ↔ (p = 2 ∧ (n = 2 ∨ n = 4)) ∨ (Nat.Prime p ∧ n = p) :=
by 
  sorry

end prime_pairs_int_fraction_l490_490159


namespace sequence_pattern_number_in_parentheses_is_99_l490_490759

-- Define the sequence as a function.
def sequence (n : Nat) : Nat :=
  if n = 1 then 3
  else if n = 2 then 15
  else if n = 3 then 35
  else if n = 4 then 63
  else if n = 6 then 143
  else 0   -- dummy value for indices not in the problem statement

-- Define the statement that the sequence follows the pattern of consecutive odd numbers multiplication.
theorem sequence_pattern (n : Nat) (h : n ∈ [1, 2, 3, 4, 6]) : 
  sequence n = match n with
               | 1 => 1 * 3
               | 2 => 3 * 5
               | 3 => 5 * 7
               | 4 => 7 * 9
               | 6 => 11 * 13
               | _ => 0
               end :=
by sorry

-- State the problem to prove that the number in parentheses is 99.
theorem number_in_parentheses_is_99 : sequence 5 = 99 :=
by sorry

end sequence_pattern_number_in_parentheses_is_99_l490_490759


namespace compare_payment_plans_l490_490901

noncomputable def carPrice : ℝ := 100000
def monthlyInterestRate : ℝ := 0.008
def ref1 : ℝ := 1.024
def ref2 : ℝ := 1.033
def ref3 : ℝ := 1.092
def ref4 : ℝ := 1.1

theorem compare_payment_plans :
  let a := (10 * ref4) / 3 in
  let S1 := 3 * a in
  let b := 10 / 12 in
  let S2 := 12 * b * ref4 in
  S2 < S1 :=
by
  sorry

end compare_payment_plans_l490_490901


namespace probability_juliet_supporter_capulet_l490_490896

theorem probability_juliet_supporter_capulet
  (P : ℕ)  -- Assume the total population is a natural number
  (h_montague : 5 / 8 < 1) -- Population proportion condition
  (h_montague_support_romeo : 80 / 100 ≤ 1) -- Montague's support percentage for Romeo
  (h_capulet_support_juliet : 70 / 100 ≤ 1) -- Capulet's support percentage for Juliet
  :
  let montague_population := (5 / 8 : ℚ) * P
  let capulet_population := (3 / 8 : ℚ) * P
  let juliet_supporters_montague := (1 / 5 : ℚ) * montague_population
  let juliet_supporters_capulet := (7 / 10 : ℚ) * capulet_population
  let total_juliet_supporters := juliet_supporters_montague + juliet_supporters_capulet
  let probability_capulet := juliet_supporters_capulet / total_juliet_supporters
  68 = Int.ofNat (Real.toRat likelihood_capulet).round :=
  sorry

end probability_juliet_supporter_capulet_l490_490896


namespace integer_solutions_count_l490_490247

theorem integer_solutions_count :
  {n : ℤ | (n + 4) * (n - 9) ≤ 0}.card = 14 :=
by sorry

end integer_solutions_count_l490_490247


namespace average_speed_approx_l490_490449

noncomputable def average_speed : ℝ :=
  let distance1 := 7
  let speed1 := 10
  let distance2 := 10
  let speed2 := 7
  let distance3 := 5
  let speed3 := 12
  let distance4 := 8
  let speed4 := 6
  let total_distance := distance1 + distance2 + distance3 + distance4
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let time3 := distance3 / speed3
  let time4 := distance4 / speed4
  let total_time := time1 + time2 + time3 + time4
  total_distance / total_time

theorem average_speed_approx : abs (average_speed - 7.73) < 0.01 := by
  -- The necessary definitions fulfill the conditions and hence we put sorry here
  sorry

end average_speed_approx_l490_490449


namespace pyramid_height_l490_490107

theorem pyramid_height (perimeter : ℝ) (apex_distance : ℝ) : 
  let s := perimeter / 4 in
  let diagonal := s * Real.sqrt 2 in
  let half_diagonal := diagonal / 2 in
  sqrt (apex_distance ^ 2 - half_diagonal ^ 2) = 3 * sqrt (47 / 3) :=
  by
    sorry

end pyramid_height_l490_490107


namespace sum_of_smallest_positive_solutions_l490_490488

noncomputable def decimal_part (x : ℝ) : ℝ := x - floor x

theorem sum_of_smallest_positive_solutions :
  let f := decimal_part in
  (∃ x₁ x₂ x₃ x₄ x₅ : ℝ,
    x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0 ∧
    f x₁ = 1 / floor x₁ ∧ f x₂ = 1 / floor x₂ ∧ f x₃ = 1 / floor x₃ ∧
    f x₄ = 1 / floor x₄ ∧ f x₅ = 1 / floor x₅ ∧
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧
    x₁ + x₂ + x₃ + x₄ + x₅ = 21.45) :=
sorry

end sum_of_smallest_positive_solutions_l490_490488


namespace range_of_y_coordinate_of_C_l490_490560

-- Define the given parabola equation
def on_parabola (x y : ℝ) : Prop := y^2 = x + 4

-- Define the coordinates for point A
def A : (ℝ × ℝ) := (0, 2)

-- Determine if points B and C lies on the parabola
def point_on_parabola (B C : ℝ × ℝ) : Prop :=
  on_parabola B.1 B.2 ∧ on_parabola C.1 C.2

-- Determine if lines AB and BC are perpendicular
def perpendicular_slopes (B C : ℝ × ℝ) : Prop :=
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let k_BC := (C.2 - B.2) / (C.1 - B.1)
  k_AB * k_BC = -1

-- Prove the range for y-coordinate of C
theorem range_of_y_coordinate_of_C (B C : ℝ × ℝ) (h1 : point_on_parabola B C) (h2 : perpendicular_slopes B C) :
  C.2 ≤ 0 ∨ C.2 ≥ 4 := sorry

end range_of_y_coordinate_of_C_l490_490560


namespace negation_of_exists_l490_490373

open Set Real

theorem negation_of_exists (x : Real) :
  ¬ (∃ x ∈ Icc 0 1, x^3 + x^2 > 1) ↔ ∀ x ∈ Icc 0 1, x^3 + x^2 ≤ 1 := 
by sorry

end negation_of_exists_l490_490373


namespace radius_increase_l490_490427

-- Definitions and conditions
def initial_circumference : ℝ := 24
def final_circumference : ℝ := 30
def circumference_radius_relation (C : ℝ) (r : ℝ) : Prop := C = 2 * Real.pi * r

-- Required proof statement
theorem radius_increase (r1 r2 Δr : ℝ)
  (h1 : circumference_radius_relation initial_circumference r1)
  (h2 : circumference_radius_relation final_circumference r2)
  (h3 : Δr = r2 - r1) :
  Δr = 3 / Real.pi :=
by
  sorry

end radius_increase_l490_490427


namespace effective_average_speed_l490_490101

def rowing_speed_with_stream := 16 -- km/h
def rowing_speed_against_stream := 6 -- km/h
def stream1_effect := 2 -- km/h
def stream2_effect := -1 -- km/h
def stream3_effect := 3 -- km/h
def opposing_wind := 1 -- km/h

theorem effective_average_speed :
  ((rowing_speed_with_stream + stream1_effect - opposing_wind) + 
   (rowing_speed_against_stream + stream2_effect - opposing_wind) + 
   (rowing_speed_with_stream + stream3_effect - opposing_wind)) / 3 = 13 := 
by
  sorry

end effective_average_speed_l490_490101


namespace maximize_profit_l490_490078

-- Define the problem conditions
variables (k : ℝ) (x : ℝ)
hypothesis (hk : k > 0)
hypothesis (hx : x ∈ Ioo 0 0.048)

-- Define the profit function y
def profit : ℝ := 0.048 * k * x^2 - k * x^3

-- State the theorem to find the maximum profit
theorem maximize_profit (h1 : k > 0) (h2 : x ∈ Ioo 0 0.048) : x = 0.032 :=
sorry

end maximize_profit_l490_490078


namespace compute_DF_l490_490027

-- Definitions for the points and triangle side lengths
variables {A B C D E F : Type}
variable [point A]
variable [point B]
variable [point C]
-- Represent segments as real numbers for the lengths
variables (AB BC AC) : ℝ
-- Conditions from the problem
variable (h1 : AB = 4)
variable (h2 : BC = 5)
variable (h3 : AC = 6)
-- D is the intersection of the angle bisector of ∠BAC with BC
variable (h4 : ∃ D, is_angle_bisector A D B C ∧ on_line D B C)
-- E is the foot of the perpendicular from B to the angle bisector AD
variable (h5 : ∃ E, perpendicular_from B E (line_through A D) ∧ on_line E A D)
-- F is the intersection of the line through E parallel to AC with BC
variable (h6 : ∃ F, parallel_line_through E F (line_through A C) ∧ on_line F B C)

-- Prove that DF = 1/2 under given conditions
theorem compute_DF :
  ∃ D F, is_intersection_point (line_through A (angle_bisector A B C)) (line_through B C) D ∧
            is_distance D E  (distance E (foot_perpendicular B (angle_bisector A (angle_bisector A (angle (B C)):real)) (angle (B C)))) ∧
            parallel_line_through E (line_through A C)) F ∧
            distance D F = 1/2 := sorry

end compute_DF_l490_490027


namespace distinct_colored_cube_patterns_l490_490414

-- Define the colors
inductive Color
| Yellow
| Black
| Red
| Pink

-- Define a coloring of the cube
structure Coloring :=
(faces : Fin 6 → Color)

-- Define the group of symmetries of a cube
-- This would involve defining the rotations and considering color blindness

-- Define what it means for Mr. Li to consider two colorings the same
-- This typically involves permutations of red and pink faces

structure Cube :=
(symmetry : Fin 6 ≃ Fin 6)

theorem distinct_colored_cube_patterns :
  -- There are 5 distinct colorings of the cube according to Mr. Li’s recognition method.
  ℕ := sorry -- Correct answer is 5

end distinct_colored_cube_patterns_l490_490414


namespace b_power_a_equals_nine_l490_490619

theorem b_power_a_equals_nine (a b : ℝ) (h : |a - 2| + (b + 3)^2 = 0) : b^a = 9 := by
  sorry

end b_power_a_equals_nine_l490_490619


namespace find_x_l490_490088

variable {x : ℕ}

-- Condition 1: Digits are 1, 3, 4, 6, and x without repetition
def valid_digits (n : ℕ) : Prop :=
  n = 1 ∨ n = 3 ∨ n = 4 ∨ n = 6 ∨ n = x

-- Condition 2: The sum of the digits of all these five-digit numbers is 2640
def sum_of_permutations (S : Finset (Finset ℕ)) : ℕ :=
  S.sum (λ d, d.sum id)

-- The specific sum we are given
def specific_sum := 2640

-- List of all five digits
def digits : Finset ℕ := {1, 3, 4, 6, x}

-- Definition used to count the sum and permutations
def permutations_count := 5.factorial

-- We use the conditions to prove x = 8
theorem find_x (h: valid_digits x) (hs: sum_of_permutations {digits} * permutations_count = specific_sum) : x = 8 := 
  sorry

end find_x_l490_490088


namespace inclination_angle_of_line_l490_490016

theorem inclination_angle_of_line (θ : ℝ) : 
  (∃ m : ℝ, ∀ x y : ℝ, 2 * x - y + 1 = 0 → m = 2) → θ = Real.arctan 2 :=
by
  sorry

end inclination_angle_of_line_l490_490016


namespace factor_expr_l490_490980

theorem factor_expr (x : ℝ) : 81 - 27 * x^3 = 27 * (3 - x) * (9 + 3 * x + x^2) := 
sorry

end factor_expr_l490_490980


namespace find_candies_l490_490506

variable (e : ℝ)

-- Given conditions
def candies_sum (e : ℝ) : ℝ := e + 4 * e + 16 * e + 96 * e

theorem find_candies (h : candies_sum e = 876) : e = 7.5 :=
by
  -- proof omitted
  sorry

end find_candies_l490_490506


namespace six_digit_squares_l490_490964

theorem six_digit_squares (x y : ℕ) 
  (h1 : y < 1000)
  (h2 : (1000 * x + y) < 1000000)
  (h3 : y * (y - 1) = 1000 * x)
  (mod8 : y * (y - 1) ≡ 0 [MOD 8])
  (mod125 : y * (y - 1) ≡ 0 [MOD 125]) :
  (1000 * x + y = 390625 ∨ 1000 * x + y = 141376) :=
sorry

end six_digit_squares_l490_490964


namespace proof_problem_l490_490577

noncomputable def z (a : ℂ) : ℂ := (a * complex.I) / (1 - 2 * complex.I)
def condition1 (a : ℂ) : Prop := a < 0
def condition2 (a : ℂ) : Prop := complex.abs (z a) = real.sqrt 5
def correct_answer (a : ℂ) : Prop := a = -5

theorem proof_problem (a : ℂ) : condition1 a → condition2 a → correct_answer a :=
by
  intro h1 h2
  sorry

end proof_problem_l490_490577


namespace find_a_l490_490230

theorem find_a (a : ℝ) :
  (∃ x : ℝ, f x = 0 ∧ deriv f x = 0) →
  f = λ x, x^3 + a * x + 1/4 →
  a = -3 / 4 :=
by {
  sorry
}

end find_a_l490_490230


namespace kendra_birdwatching_l490_490672

theorem kendra_birdwatching :
  let birds_seen_monday := 5 * 7 in
  let birds_seen_tuesday := 5 * 5 in
  let birds_seen_wednesday := 10 * 8 in
  let total_birds_seen := birds_seen_monday + birds_seen_tuesday + birds_seen_wednesday in
  let total_sites_visited := 5 + 5 + 10 in
  (total_birds_seen / total_sites_visited : ℝ) = 7 :=
by
  sorry

end kendra_birdwatching_l490_490672


namespace swim_ratio_l490_490454

theorem swim_ratio
  (V_m : ℝ) (h1 : V_m = 4.5)
  (V_s : ℝ) (h2 : V_s = 1.5)
  (V_u : ℝ) (h3 : V_u = V_m - V_s)
  (V_d : ℝ) (h4 : V_d = V_m + V_s)
  (T_u T_d : ℝ) (h5 : T_u / T_d = V_d / V_u) :
  T_u / T_d = 2 :=
by {
  sorry
}

end swim_ratio_l490_490454


namespace problem_division_count_problem_subtraction_count_l490_490428

noncomputable def gcd_division_count (a b : ℕ) : ℕ :=
if b = 0 then 0 else 1 + gcd_division_count b (a % b)

noncomputable def gcd_subtraction_count : ℕ → ℕ → ℕ
| a, b => if a = b then 0 else if a > b then 1 + gcd_subtraction_count (a - b) b else 1 + gcd_subtraction_count a (b - a)

theorem problem_division_count : gcd_division_count 240 288 = 2 :=
by sorry

theorem problem_subtraction_count : gcd_subtraction_count 36 48 = 3 :=
by sorry

end problem_division_count_problem_subtraction_count_l490_490428


namespace mandy_payment_l490_490314

theorem mandy_payment :
  let rate_per_room : ℚ := 15 / 4
  let rooms_cleaned : ℚ := 12 / 5
  let discount : ℚ := 10 / 100
  let original_total : ℚ := rate_per_room * rooms_cleaned
  let discounted_total : ℚ := if rooms_cleaned > 2 then original_total * (1 - discount) else original_total
  discounted_total = 81 / 10 :=
by
  -- Definitions
  let rate_per_room := (15 : ℚ) / 4
  let rooms_cleaned := (12 : ℚ) / 5
  let discount := (10 : ℚ) / 100
  -- Calculate original total without discount
  let original_total := rate_per_room * rooms_cleaned
  have h1 : original_total = 9, by
    calc
      original_total = (15 / 4) * (12 / 5) : by rfl
      ... = (15 * 12) / (4 * 5) : by simp [mul_div, mul_comm (12 : ℚ)]
      ... = 180 / 20 : by simp [mul_comm (15 : ℚ), mul_comm (12 : ℚ)]
      ... = 9 : by norm_num
  -- Apply discount if rooms_cleaned > 2
  let discounted_total := if rooms_cleaned > 2 then original_total * (1 - discount) else original_total
  have h2 : rooms_cleaned > 2, by
    calc
      rooms_cleaned = 12 / 5 : by rfl
      ... = 2.4 : by norm_num
      ... > 2 : by norm_num
  have h3 : original_total * (1 - discount) = 8.1, by
    calc
      original_total * (1 - discount) = 9 * (1 - 0.1) : by rw [h1, discount]
      ... = 9 * 0.9 : by norm_num
      ... = 8.1 : by norm_num
  have h4 : discounted_total = 8.1, by
    simp [h2, h3]
  -- Convert 8.1 to fraction form
  have h5 : (8.1 : ℚ) = 81 / 10, by norm_num
  show discounted_total = 81 / 10, by rw [h4, h5]; rfl

end mandy_payment_l490_490314


namespace count_valid_A_l490_490187

-- Definitions of the conditions and the final proof statement
def divisible_by (n : ℕ) (d : ℕ) : Prop := d ≠ 0 ∧ n % d = 0

theorem count_valid_A : (finset.filter (λ A, divisible_by 120 A ∧ divisible_by (208 + 40 * A) 8) 
  (finset.range 10)).card = 2 :=
by
  sorry

end count_valid_A_l490_490187


namespace num_factors_of_n_multiples_of_360_l490_490617

theorem num_factors_of_n_multiples_of_360 (n : ℕ) (h : n = 2^12 * 3^15 * 5^9) : 
  (∃ count : ℕ, count = 1260 ∧ ∀ d : ℕ, d ∣ n → ∃ k : ℕ, d = 360 * k) :=
by {
  have h1 : 3 ≤ 12 := sorry,
  have h2 : 2 ≤ 15 := sorry,
  have h3 : 1 ≤ 9 := sorry,
  have num_factors := (12 - 3 + 1) * (15 - 2 + 1) * (9 - 1 + 1),
  exact ⟨num_factors, by simp [num_factors, h1, h2, h3], sorry⟩
}

end num_factors_of_n_multiples_of_360_l490_490617


namespace max_sum_arithmetic_sequence_l490_490763

theorem max_sum_arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) (S : ℕ → ℝ) (h1 : (a + 2) ^ 2 = (a + 8) * (a - 2))
  (h2 : ∀ k, S k = (k * (2 * a + (k - 1) * d)) / 2)
  (h3 : 10 = a) (h4 : -2 = d) :
  S 10 = 90 :=
sorry

end max_sum_arithmetic_sequence_l490_490763


namespace polynomial_properties_l490_490686

-- Define Q as a polynomial
variables {Q : ℤ → ℤ}

-- Define the conditions
def condition1 (Q : ℤ → ℤ) : Prop :=
∀ x : ℤ, Q x = Q 0 + Q 1 * x + Q 2 * x^2 + Q 3 * x^3

def condition2 (Q : ℤ → ℤ) : Prop := Q (-1) = 2

-- The main theorem to be proved
theorem polynomial_properties (Q : ℤ → ℤ) :
  condition1 Q → condition2 Q → Q = (λ x, x^2 + 1) :=
by
  intros h1 h2
  sorry

end polynomial_properties_l490_490686


namespace rem_value_is_correct_l490_490147

def rem (x y : ℚ) : ℚ :=
  x - y * (Int.floor (x / y))

theorem rem_value_is_correct : rem (-5/9) (7/3) = 16/9 := by
  sorry

end rem_value_is_correct_l490_490147


namespace find_quadratic_function_l490_490573

noncomputable def quadratic_function (a h k : ℝ) : ℝ → ℝ := λ x, a * (x - h) ^ 2 + k

theorem find_quadratic_function : 
  ∃ a h k : ℝ, 
    (h = 4 ∧ k = -8) ∧ 
    (quadratic_function a h k 6 = 0) ∧ 
    (quadratic_function 2 4 (-8) = (λ x, 2 * (x - 4)^2 - 8)) :=
by
  sorry

end find_quadratic_function_l490_490573


namespace professors_women_tenured_or_both_l490_490945

variable (professors : ℝ) -- Total number of professors as percentage
variable (women tenured men_tenured tenured_women : ℝ) -- Given percentages

-- Conditions
variables (hw : women = 0.69 * professors) 
          (ht : tenured = 0.7 * professors)
          (hm_t : men_tenured = 0.52 * (1 - women) * professors)
          (htw : tenured_women = tenured - men_tenured)
          
-- The statement to prove
theorem professors_women_tenured_or_both :
  women + tenured - tenured_women = 0.8512 * professors :=
by
  sorry

end professors_women_tenured_or_both_l490_490945


namespace domain_of_f_is_R_f_is_periodic_f_has_max_and_min_values_l490_490613

def f (x : ℝ) := Real.tan (Real.sin x + Real.cos x)

theorem domain_of_f_is_R : ∀ x : ℝ, f x = Real.tan (Real.sin x + Real.cos x) := 
by
  intro x 
  exact rfl

theorem f_is_periodic : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := 
by
  intro x
  sorry

theorem f_has_max_and_min_values : ∃ m M : ℝ, ∀ x : ℝ, m ≤ f x ∧ f x ≤ M := 
by
  sorry


end domain_of_f_is_R_f_is_periodic_f_has_max_and_min_values_l490_490613


namespace sum_of_solutions_eq_9_l490_490828

theorem sum_of_solutions_eq_9 :
  let roots := {x : ℝ | x^2 = 9 * x - 20}
  in ∑ x in roots, x = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490828


namespace sum_of_solutions_l490_490789

-- Define the quadratic equation and variable x
def quadratic_equation := ∀ x : ℝ, (x^2 - 9 * x + 20 = 0)

-- Define what we need to prove
theorem sum_of_solutions : ∃ s : ℝ, (∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → s = x1 + x2) ∧ s = 9 :=
by
  sorry -- Proof is omitted

end sum_of_solutions_l490_490789


namespace identify_perpendicular_sets_l490_490736

def line := Type
def plane := Type

variables (a b : line) (α β : plane)

-- Definitions for the conditions
def set1 : Prop := (a ⊆ α) ∧ (b ∥ β) ∧ (α ⊥ β)
def set2 : Prop := (a ⊥ α) ∧ (b ⊥ β) ∧ (α ⊥ β)
def set3 : Prop := (a ⊆ α) ∧ (b ⊥ β) ∧ (α ∥ β)
def set4 : Prop := (a ⊥ α) ∧ (b ∥ β) ∧ (α ∥ β)

-- Definition of perpendicular lines
def perpendicular (a b : line) : Prop := sorry -- Assuming there exists a proper definition for perpendicular lines

-- Lean 4 statement for the proof problem
theorem identify_perpendicular_sets :
  { set2, set3, set4 } = { condition |
  (condition = set1 → False) ∧
  (condition = set2 → True) ∧
  (condition = set3 → True) ∧
  (condition = set4 → True) } :=
sorry

end identify_perpendicular_sets_l490_490736


namespace average_of_remaining_numbers_l490_490743

theorem average_of_remaining_numbers
  (avg : ℕ → ℕ) 
  (total_sum : ℕ) 
  (remove_nums : list ℕ) 
  (x : ℕ) 
  (sizes : ℕ × ℕ)
  (sum_diff : ℕ → ℕ → ℕ) 
  (avg_remaining : ℕ → ℕ → ℤ) :
  avg 12 = 90 →
  total_sum = 1080 →
  remove_nums = [82, 73, x] →
  sizes = (12, 9) →
  sum_diff total_sum (82 + 73 + x) = 925 - x →
  avg_remaining (925 - x) 9 = (925 - x) / 9 :=
by 
  sorry

end average_of_remaining_numbers_l490_490743


namespace segment_EP_length_l490_490350

theorem segment_EP_length (side len_EP : ℝ) (h_square : side = 4) 
  (h_partition : (side * side) / 3 = ((side * (len_EP / 2)) / 2)) :
  len_EP = real.sqrt (208 / 9) := 
sorry

end segment_EP_length_l490_490350


namespace common_difference_unique_l490_490554

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d a1 : ℝ, ∀ n : ℕ, a n = a1 + n * d

theorem common_difference_unique {a : ℕ → ℝ}
  (h1 : a 2 = 5)
  (h2 : a 3 + a 5 = 2) :
  ∃ d : ℝ, (∀ n : ℕ, a n = a 1 + (n - 1) * d) ∧ d = -2 :=
sorry

end common_difference_unique_l490_490554


namespace factor_81_sub_27x3_l490_490973

theorem factor_81_sub_27x3 (x : ℝ) : 81 - 27 * x^3 = 3 * (3 - x) * (81 + 27 * x + 9 * x^2) :=
sorry

end factor_81_sub_27x3_l490_490973


namespace prob_fx_leq_0_l490_490750

def f (x : ℝ) : ℝ := x^2 - x - 2

theorem prob_fx_leq_0 : 
  (∀ x0 ∈ set.Icc (-5 : ℝ) 5, x0 ∈ set.Icc (-1 : ℝ) 2 → f x0 ≤ 0) → 
  (f '' (set.Icc (-5 : ℝ) 5)).measure (set.Icc (-5 : ℝ) 5) = 3 / 10 := by
  sorry

end prob_fx_leq_0_l490_490750


namespace total_three_digit_numbers_l490_490030

-- Definitions of the conditions
def card1 := (1, 2)
def card2 := (3, 4)
def card3 := (5, 6)
def card4 := (7, 8)

-- Prove that the number of three-digit numbers formed is 192
theorem total_three_digit_numbers : 
  let chosen_cards := {card1, card2, card3, card4}
  let ways_to_choose_3_cards := 4
  let ways_to_decide_sides := 8
  let permutations := 6
  ways_to_choose_3_cards * ways_to_decide_sides * permutations = 192 :=
by sorry

end total_three_digit_numbers_l490_490030


namespace remainders_sum_l490_490054

theorem remainders_sum (a b c : ℕ) 
  (h1 : a % 30 = 15) 
  (h2 : b % 30 = 20) 
  (h3 : c % 30 = 10) : 
  (a + b + c) % 30 = 15 := 
by
  sorry

end remainders_sum_l490_490054


namespace sum_of_solutions_eq_9_l490_490807

theorem sum_of_solutions_eq_9 (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20) :
  let (sum_roots : ℝ) := -b / a in 
  sum_roots = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490807


namespace correct_choice_is_C_l490_490116

-- Define the conditions as given in the problem
def conditionA : Prop := (∛(-64) = 4)
def conditionB : Prop := (√(49) = 7) ∨ (√(49) = -7)
def conditionC : Prop := (∛(1 / 27) = 1 / 3)
def conditionD : Prop := (√(9) = 3) ∨ (√(9) = -3)

-- Prove that among the statements, the correct one is conditionC
theorem correct_choice_is_C : ¬conditionA ∧ ¬conditionB ∧ conditionC ∧ ¬conditionD :=
by
  sorry

end correct_choice_is_C_l490_490116


namespace triangle_cong_two_legs_l490_490340

variables (AB DE AC DF : ℝ)
variables (hAB_DE : AB = DE) (hAC_DF : AC = DF)

theorem triangle_cong_two_legs (ABC DEF : Type) [RightAngledTriangle ABC] [RightAngledTriangle DEF] :
  AB = DE → AC = DF → ABC ≃ DEF := 
by sorry

end triangle_cong_two_legs_l490_490340


namespace parabola_intersection_prob_l490_490772

noncomputable def prob_intersect_parabolas : ℚ :=
  57 / 64

theorem parabola_intersection_prob :
  ∀ (a b c d : ℤ), (1 ≤ a ∧ a ≤ 8) → (1 ≤ b ∧ b ≤ 8) →
  (1 ≤ c∧ c ≤ 8) → (1 ≤ d ∧ d ≤ 8) →
  prob_intersect_parabolas = 57 / 64 :=
by
  intros a b c d ha hb hc hd
  sorry

end parabola_intersection_prob_l490_490772


namespace eval_floor_sqrt_116_l490_490513
-- Lean 4 statement to prove the given mathematical problem


theorem eval_floor_sqrt_116:
  ∃ (x : ℝ), 10 ≤ x ∧ x < 11 ∧ x = real.sqrt 116 ∧ real.floor x = 10 :=
by
  sorry

end eval_floor_sqrt_116_l490_490513


namespace vector_parallel_probability_l490_490310

theorem vector_parallel_probability :
  let S := {0, 1, 2, 3, 4}
  let pairs := [(0, 0), (1, 2), (2, 4)] in
  let total_pairs := finset.card (finset.product S S) in
  let valid_pairs := finset.card (finset.filter (λ (p : ℕ × ℕ), p.snd = 2 * p.fst) (finset.product S S)) in
  valid_pairs / total_pairs = 3 / 25 :=
by
  sorry

end vector_parallel_probability_l490_490310


namespace domain_of_f_monotonicity_of_f_range_of_m_l490_490584

noncomputable def f (x : ℝ) : ℝ := log (1 / 2) (10 - 2 * x)

-- Domain proof
theorem domain_of_f : ∀ (x: ℝ), (10 - 2 * x) > 0 ↔ x < 5 := 
by 
  intro x
  rw [sub_pos, lt_div_iff']
  simp
  sorry

-- Monotonicity proof
theorem monotonicity_of_f : 
  ∀ (x y : ℝ), x < y → x < 5 → y < 5 → f x < f y :=
by
  intros x y hxy hx hy
  have hu := sub_lt_sub_left hxy 10
  simp at hu
  sorry

-- Range of m proof
theorem range_of_m : 
  ∀ (m : ℝ), (∀ x ∈ set.Icc 3 4, f x ≥ (1 / 2) ^ x + m) → m ≤ - (17 / 8) :=
by 
  intro m h
  have := h 3 ⟨le_refl 3, by norm_num⟩
  simp [f, pow3, log_div] at this
  linarith
  sorry

end domain_of_f_monotonicity_of_f_range_of_m_l490_490584


namespace four_drivers_suffice_l490_490918

theorem four_drivers_suffice
  (one_way_trip_time : ℕ := 160) -- in minutes
  (round_trip_time : ℕ := 320) -- in minutes
  (rest_time : ℕ := 60) -- in minutes
  (time_A_returns : ℕ := 760) -- 12:40 PM in minutes from midnight
  (time_A_next_start : ℕ := 820) -- 1:40 PM in minutes from midnight
  (time_D_departs : ℕ := 785) -- 1:05 PM in minutes from midnight
  (time_A_fifth_depart : ℕ := 970) -- 4:10 PM in minutes from midnight
  (time_B_returns : ℕ := 960) -- 4:00 PM in minutes from midnight
  (time_B_sixth_depart : ℕ := 1050) -- 5:30 PM in minutes from midnight
  (time_A_fifth_complete: ℕ := 1290) -- 9:30 PM in minutes from midnight
  : 4_drivers_sufficient : ℕ :=
    if time_A_fifth_complete = 1290 then 1 else 0
-- The theorem states that if the calculated trip completion time is 9:30 PM, then 4 drivers are sufficient.
  sorry

end four_drivers_suffice_l490_490918


namespace simplified_radical_sqrt6_l490_490876

def isSimplifiedRadical (n : ℕ) : Prop :=
  ∀ m : ℕ, m * m ∣ n → m = 1

def isFraction (n : ℚ) : Prop :=
  n.denom ≠ 1

theorem simplified_radical_sqrt6:
  isSimplifiedRadical 6 ∧ ¬isFraction 6 :=
by
  sorry

end simplified_radical_sqrt6_l490_490876


namespace function_even_compare_values_l490_490265

def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c

theorem function_even 
  (b c : ℝ)
  (h : ∀ x : ℝ, f (-x) b c = f x b c) :
  b = 0 :=
by
  have : f (-1) b c = f 1 b c := h 1
  have : (-1)^2 + b * (-1) + c = 1^2 + b * 1 + c := this
  linarith

theorem compare_values 
  (c : ℝ) 
  (h : ∀ x : ℝ, f (-x) 0 c = f x 0 c) :
  f 1 0 c < f (-2) 0 c ∧ f (-2) 0 c < f 3 0 c :=
by
  have hb : 0 = 0 := function_even 0 c h
  have h1 : f 1 0 c = 1^2 + c := by simp [f]
  have h_2 : f (-2) 0 c = (-2)^2 + c := by simp [f]
  have h2 : (-2)^2 = 4 := by norm_num
  have h_2' : f (-2) 0 c = 4 + c := by rw [h_2, h_2]
  have h3 : f 3 0 c = 3^2 + c := by simp [f]
  have h3' : 3^2 = 9 := by norm_num
  have h3'' : f 3 0 c = 9 + c := by rw [h3, h3']
  linarith

example (c : ℝ) (h : ∀ x : ℝ, f (-x) 0 c = f x 0 c) : f 1 0 c < f (-2) 0 c ∧ f (-2) 0 c < f 3 0 c :=
compare_values c h

end function_even_compare_values_l490_490265


namespace trapezoid_area_l490_490419

-- Define the lines bounding the trapezoid
def line1 (x : ℝ) : ℝ := x + 2
def line2 (x : ℝ) : ℝ := 12
def line3 (x : ℝ) : ℝ := 3
def y_axis (x : ℝ) : ℝ := 0

-- Define the points of intersection and vertices
def point1 : ℝ × ℝ := (10, 12) -- Intersection of line1 and line2
def point2 : ℝ × ℝ := (1, 3) -- Intersection of line1 and line3
def point3 : ℝ × ℝ := (0, 12) -- Line2 intersects y-axis
def point4 : ℝ × ℝ := (0, 3) -- Line3 intersects y-axis

-- Define bases and height
def lower_base : ℝ := 1
def upper_base : ℝ := 10
def height : ℝ := 9

-- Theorem: The area of the trapezoid is 49.5 square units
theorem trapezoid_area : (1 + 10) / 2 * 9 = 49.5 := by
  sorry

end trapezoid_area_l490_490419


namespace sufficient_drivers_and_completion_time_l490_490928

noncomputable def one_way_trip_minutes : ℕ := 2 * 60 + 40
noncomputable def round_trip_minutes : ℕ := 2 * one_way_trip_minutes
noncomputable def rest_period_minutes : ℕ := 60
noncomputable def twelve_forty_pm : ℕ := 12 * 60 + 40 -- in minutes from midnight
noncomputable def one_forty_pm : ℕ := twelve_forty_pm + rest_period_minutes
noncomputable def thirteen_five_pm : ℕ := 13 * 60 + 5 -- 1:05 PM
noncomputable def sixteen_ten_pm : ℕ := 16 * 60 + 10 -- 4:10 PM
noncomputable def sixteen_pm : ℕ := 16 * 60 -- 4:00 PM
noncomputable def seventeen_thirty_pm : ℕ := 17 * 60 + 30 -- 5:30 PM
noncomputable def twenty_one_thirty_pm : ℕ := sixteen_ten_pm + round_trip_minutes -- 9:30 PM (21:30)

theorem sufficient_drivers_and_completion_time :
  4 = 4 ∧ twenty_one_thirty_pm = 21 * 60 + 30 := by
  sorry 

end sufficient_drivers_and_completion_time_l490_490928


namespace sum_of_solutions_l490_490838

theorem sum_of_solutions (x : ℝ) : 
  (∀ x : ℝ, x^2 = 9*x - 20 → x = 4 ∨ x = 5) → (4 + 5 = 9) :=
by
  intros h
  calc 4 + 5 = 9 : by norm_num
  sorry

end sum_of_solutions_l490_490838


namespace sum_of_roots_eq_l490_490146

noncomputable def sum_of_roots : ℝ :=
  let roots := { x : ℝ | (2 * x + 3) * (x - 2) + (2 * x + 3) * (x - 8) = 0 }
  in roots.sum

theorem sum_of_roots_eq : sum_of_roots = 7 / 2 :=
by
  sorry

end sum_of_roots_eq_l490_490146


namespace lottery_ticket_increment_l490_490706

theorem lottery_ticket_increment (x : ℝ) :
  let price_sum := 1 + (1 + x) + (1 + 2 * x) + (1 + 3 * x) + (1 + 4 * x)
  price_sum = 15 → x = 1 :=
by
  intros price_sum_eq
  dsimp only [price_sum] at price_sum_eq
  linarith

end lottery_ticket_increment_l490_490706


namespace proof_a2_is_56_l490_490195

noncomputable def calculate_a2_eq_56 : Prop :=
  let lhs := (1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7)
  let rhs := (a_0 + a_1 * (x - 1) + a_2 * (x - 1) ^ 2 + a_3 * (x - 1) ^ 3 + a_4 * (x - 1) ^ 4 + a_5 * (x - 1) ^ 5 + a_6 * (x - 1) ^ 6 + a_7 * (x - 1) ^ 7)
  let diff_eq := (2 + 3 * 2 * x + 4 * 3 * x^2 + 5 * 4 * x^3 + 6 * 5 * x^4 + 7 * 6 * x^5)
  ∑ i in finset.range 8, x^i = ∑ j in finset.range 8, (a_j * (x - 1)^j) ∧ diff_eq = (2 * a_2 + 3 * 2 * a_3 * (x - 1) + 4 * 3 * a_4 * (x - 1)^2 + 5 * 4 * a_5 * (x - 1)^3 + 6 * 5 * a_6 * (x - 1)^4 + 7 * 6 * a_7 * (x - 1)^5) →
    a_2 = 56

theorem proof_a2_is_56 : calculate_a2_eq_56 := sorry

end proof_a2_is_56_l490_490195


namespace find_principal_amount_l490_490004

-- Define all conditions
def A : ℝ := 5324.000000000002
def r : ℝ := 0.10
def n : ℕ := 1
def t : ℕ := 3

-- Compound interest formula
def compound_interest (P : ℝ) : ℝ := P * (1 + r/n)^(n*t)

-- Assertion to prove
theorem find_principal_amount : ∃ P : ℝ, compound_interest P = A :=
by
  -- Replace with the proof
  sorry

end find_principal_amount_l490_490004


namespace isosceles_triangle_angles_l490_490517

open real

-- Define the isosceles triangle where the intersection of altitudes lies on the inscribed circle
noncomputable def isosceles_triangle_condition (α : ℝ) : Prop :=
  ∃ (A B C O H : ℝ), 
  let K := (A + B) / 2 in
  is_isosceles_triangle A B C ∧
  altitudes_intersect_at H A B C ∧
  altitudes_intersection_on_incircle H O A B C ∧
  α = arccos (2 / 3)

theorem isosceles_triangle_angles :
  ∀ {α : ℝ},
  isosceles_triangle_condition α →
  angles_of_isosceles_triangle α = [arccos (2 / 3), arccos (2 / 3), π - 2 * arccos (2 / 3)] :=
begin
  sorry
end

end isosceles_triangle_angles_l490_490517


namespace vertices_of_regular_pentagon_l490_490019

theorem vertices_of_regular_pentagon (z : ℕ → ℂ) (h_nonzero : ∀ i, z i ≠ 0) 
  (h_magnitude : ∀ i j, |z i| = |z j|) 
  (h_sum_zero : ∑ i in Finset.range 5, z i = 0)
  (h_sum_squares_zero : ∑ i in Finset.range 5, z i ^ 2 = 0) : 
  ∃ (e : ℂ), ∀ i, ∃ k ∈ Finset.range 5, z i = e * complex.exp (2 * real.pi * complex.I * k / 5) := 
sorry

end vertices_of_regular_pentagon_l490_490019


namespace projection_of_difference_l490_490571

variables (a b : ℝ^3)
variables (hab : ∥a∥ = 1 ∧ ∥b∥ = 1) (angle_ab : real.angle a b = real.angle.pi * 2 / 3)

theorem projection_of_difference :
  let proj := (a - b).dot b / b.dot b in
  proj • b = -3/2 • b :=
by 
  sorry

end projection_of_difference_l490_490571


namespace expected_value_prize_l490_490758

theorem expected_value_prize :
  ∃ (P1 P2 P3 ξ1 ξ2 ξ3 : ℝ),
    (P1 + P2 + P3 = 1) ∧ 
    (P2 = 2 * P1) ∧ 
    (P3 = 4 * P1) ∧ 
    (ξ1 = 700) ∧ 
    (ξ2 = 560) ∧ 
    (ξ3 = 420) ∧ 
    (P1 * ξ1 + P2 * ξ2 + P3 * ξ3 = 500) :=
by {
  let a := 1 / 7,
  have h1 : P1 = a := by sorry,
  have h2 : P2 = 2 * a := by sorry,
  have h3 : P3 = 4 * a := by sorry,
  have h4 : ξ1 = 700 := by sorry,
  have h5 : ξ2 = 560 := by sorry,
  have h6 : ξ3 = 420 := by sorry,
  use [a, 2 * a, 4 * a, 700, 560, 420],
  split; try { assumption },
  split; try { assumption },
  split; try { assumption },
  split; try { assumption },
  split; try { assumption },
  split; try { assumption },
  calc
    P1 * ξ1 + P2 * ξ2 + P3 * ξ3
      = a * 700 + 2 * a * 560 + 4 * a * 420 : by rw [h1, h2, h3, h4, h5, h6]
  ... = 100 + 160 + 240 : by sorry
  ... = 500 : by norm_num,
}

end expected_value_prize_l490_490758


namespace probability_of_multiples_of_3_from_1_to_10_l490_490538

-- Define the set of integers from 1 to 10
def integers_set := {x | 1 ≤ x ∧ x ≤ 10}

-- Define a predicate that checks if a number is a multiple of 3
def is_multiple_of_3 (n : ℕ) : Prop := n % 3 = 0

-- Extract the multiples of 3 from the set
def multiples_of_3 := {x ∈ integers_set | is_multiple_of_3 x}

-- Calculate the probability as a rational number
def probability_of_multiples_of_3 : ℚ := (multiples_of_3.to_finset.card : ℚ) / (integers_set.to_finset.card : ℚ)

-- State the theorem to be proved
theorem probability_of_multiples_of_3_from_1_to_10 :
  probability_of_multiples_of_3 = 3 / 10 :=
sorry

end probability_of_multiples_of_3_from_1_to_10_l490_490538


namespace find_m_l490_490644

open Real

noncomputable def value_of_m : ℝ :=
  let m : ℝ := 2
  m

theorem find_m : ∃ m > 0, 
  (∀ (ρ θ : ℝ), ρ * sin(θ)^2 = m * cos(θ) ↔ y^2 = m * x) ∧ 
  (∀ x, y = x - 2 = true) ∧ 
  (|AP| * |BP| = |BA|^2) -> 
  m = 2
  :=
  sorry

end find_m_l490_490644


namespace sum_of_roots_l490_490863

theorem sum_of_roots (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) :
  ∑ x in {x | a * x^2 + b * x + c = 0}, x = 9 :=
by
  sorry

end sum_of_roots_l490_490863


namespace rest_area_location_l490_490330

theorem rest_area_location (milepost_fourth_exit milepost_eighth_exit : ℕ) (h_fourth : milepost_fourth_exit = 50) (h_eighth : milepost_eighth_exit = 210) :
  ∃ milepost_rest_area, milepost_rest_area = milepost_fourth_exit + (milepost_eighth_exit - milepost_fourth_exit) / 2 ∧ milepost_rest_area = 130 :=
by
  use 50 + (210 - 50) / 2
  split
  case left =>
    -- simplify
    sorry
  case right =>
    -- prove the result
    sorry

end rest_area_location_l490_490330


namespace find_number_l490_490895

theorem find_number (x : ℝ) (h : 0.45 * x = 162) : x = 360 :=
sorry

end find_number_l490_490895


namespace vasya_petya_prime_l490_490775

theorem vasya_petya_prime (x : ℚ) (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (10 * x = p) ∧ (15 * x = q) → ∃ x : ℚ, (10 * x).Nat.Abs.Prime ∧ (15 * x).Nat.Abs.Prime := 
by
  intro h
  sorry

end vasya_petya_prime_l490_490775


namespace part_a_part_b_l490_490152

variables (k1 k2 : Type)
variables (T X A B S C Y I : k1)
variables (l m : k2)
variables [metric_space k1]

open_locale classical

-- Conditions
axiom h1 : circles_touch_externally_at k1 k2 T
axiom h2 : tangent_to_circle k2 l X
axiom h3 : intersects_circle_at l k1 A
axiom h4 : second_intersection_point k1 l T S
axiom h5 : chosen_on_arc_not_containing C (arc TS) A B
axiom h6 : tangent_at_point k2 m Y
axiom h7 : SEG_ne_intersect l m ST
axiom h8 : intersection_point l m XY SC I

-- Questions
theorem part_a : concyclic C T Y I :=
sorry -- Proof is not required

theorem part_b : excenter_of_triangle ABC I BC :=
sorry -- Proof is not required

end part_a_part_b_l490_490152


namespace Harold_XL_boxes_l490_490245

-- Given conditions as definitions
def rolls_needed_for_shirts (shirt_boxes : ℕ) (boxes_per_roll : ℕ) : ℕ :=
  shirt_boxes / boxes_per_roll

def rolls_bought (total_cost : ℕ) (cost_per_roll : ℕ) : ℕ :=
  total_cost / cost_per_roll

def rolls_left (total_rolls : ℕ) (used_rolls : ℕ) : ℕ :=
  total_rolls - used_rolls

def xl_boxes (rolls_left : ℕ) (boxes_per_roll : ℕ) : ℕ :=
  rolls_left * boxes_per_roll

-- Specific instance of the general definitions in the problem
def Harold : ℕ :=
  let shirt_boxes := 20
  let boxes_per_roll_for_shirts := 5
  let total_cost := 32
  let cost_per_roll := 4
  let boxes_per_roll_for_xl := 3
  let used_for_shirts := rolls_needed_for_shirts shirt_boxes boxes_per_roll_for_shirts
  let total_rolls := rolls_bought total_cost cost_per_roll
  let left_rolls := rolls_left total_rolls used_for_shirts
  in xl_boxes left_rolls boxes_per_roll_for_xl

theorem Harold_XL_boxes : Harold = 12 :=
by
  sorry

end Harold_XL_boxes_l490_490245


namespace xiaomings_hens_eggs_l490_490059

-- Definitions of conditions
def lays_one_egg_per_day (hen : Nat) : Prop := 
  hen = 1

def lays_one_egg_every_two_days (hen : Nat) : Prop := 
  hen = 2

def lays_one_egg_every_three_days (hen : Nat) : Prop := 
  hen = 3

-- General Property: Calculate eggs laid by a hen over 31 days given its laying frequency
def eggs_laid (hen : Nat) (days : Nat) : Nat :=
  if lays_one_egg_per_day hen then
    days
  else if lays_one_egg_every_two_days hen then
    days / 2
  else if lays_one_egg_every_three_days hen then
    days / 3
  else
    0

-- Calculate the total number of eggs laid by all hens in 31 days
def total_eggs_laid (days : Nat) : Nat :=
  eggs_laid 1 days + eggs_laid 2 days + eggs_laid 3 days

-- Conditions
def January_is_31_days : Nat := 31

theorem xiaomings_hens_eggs :
  total_eggs_laid January_is_31_days = 56 := by
  sorry

end xiaomings_hens_eggs_l490_490059


namespace combined_square_problems_l490_490767

theorem combined_square_problems
  (P : ℕ)
  (hP : P = 28) :
  let s := P / 4,
      A := s * s,
      rectangle_area := 3 * A,
      larger_side := 2 * s,
      larger_perimeter := 4 * larger_side in
  rectangle_area = 147 ∧ larger_perimeter = 56 :=
by
  let s := P / 4,
  let A := s * s,
  let rectangle_area := 3 * A,
  let larger_side := 2 * s,
  let larger_perimeter := 4 * larger_side,
  sorry

end combined_square_problems_l490_490767


namespace eccentricity_of_ellipse_l490_490701

-- Definition of an ellipse with given parameters
structure Ellipse (a b : ℝ) (h : a > b > 0) :=
(x y : ℝ)
(hP : (x^2 / a^2 + y^2 / b^2 = 1))

-- The foci of the ellipse
structure Foci (a b : ℝ) :=
(F1 F2 : ℝ × ℝ) -- Suppose these are the coordinates of the foci

-- The incenter of the triangle formed by the given points
structure Incenter (P F1 F2 : ℝ × ℝ) :=
(I : ℝ × ℝ)

-- The area relationship given in the problem
def area_relationship (P F1 F2 I : ℝ × ℝ) : Prop :=
  let S_IPF1 := 1/2 * ((fst P - fst F1) * snd I - (snd P - snd F1) * fst I) in
  let S_IPF2 := 1/2 * ((fst P - fst F2) * snd I - (snd P - snd F2) * fst I) in
  let S_IF1F2 := 1/2 * ((fst F1 - fst F2) * snd I - (snd F1 - snd F2) * fst I) in
  S_IPF1 + S_IPF2 = 2 * S_IF1F2

-- Definition of eccentricity of an ellipse
def eccentricity (a c : ℝ) : ℝ := c / a

-- The main theorem
theorem eccentricity_of_ellipse (a b : ℝ) (h : a > b > 0) (P : ℝ × ℝ)
  (hP : Ellipse a b h) (F1 F2 : ℝ × ℝ) (inc : Incenter P F1 F2)
  (hArea : area_relationship P F1 F2 inc.I) :
  eccentricity a ((F1.1 - F2.1) / 2) = 1 / 2 :=
by
  sorry

end eccentricity_of_ellipse_l490_490701


namespace shaded_area_l490_490466

noncomputable def side_length : ℝ := 12
noncomputable def radius : ℝ := side_length / 2
noncomputable def square_area : ℝ := side_length * side_length
noncomputable def circle_area : ℝ := π * radius * radius
noncomputable def quarter_circle_area : ℝ := circle_area / 4
noncomputable def total_quarter_circles_area : ℝ := 3 * quarter_circle_area

theorem shaded_area : (square_area - total_quarter_circles_area) = 144 - 27 * π := by
  sorry

end shaded_area_l490_490466


namespace power_function_value_at_half_l490_490588

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2 - x) - 3/4

noncomputable def g (α : ℝ) (x : ℝ) : ℝ := x^α

theorem power_function_value_at_half (a : ℝ) (α : ℝ) 
  (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : f a 2 = 1 / 4) (h4 : g α 2 = 1 / 4) : 
  g α (1/2) = 4 := 
by
  sorry

end power_function_value_at_half_l490_490588


namespace conditional_probability_correct_l490_490663

variable {Ω : Type*} [ProbabilitySpace Ω]

-- Define the events A and B
variable (A B : Event Ω)

-- Probability of A and AB
variable (P_A : ℝ) (P_AB : ℝ)

-- Given conditions: P(A) = 0.5 and P(AB) = 0.4
axiom P_A_def : P_A = 0.5
axiom P_AB_def : P_AB = 0.4

-- Define the conditional probability P(B|A)
noncomputable def P_B_given_A : ℝ := P_AB / P_A

-- Expected result: P(B|A) = 0.8
theorem conditional_probability_correct : P_B_given_A A B P_AB P_A = 0.8 :=
by
  -- Use the given conditions
  rw [P_A_def, P_AB_def]
  -- Simplify the expression
  sorry

end conditional_probability_correct_l490_490663


namespace man_work_alone_l490_490098

theorem man_work_alone (W: ℝ) (M S: ℝ)
  (hS: S = W / 6.67)
  (hMS: M + S = W / 4):
  W / M = 10 :=
by {
  -- This is a placeholder for the proof
  sorry
}

end man_work_alone_l490_490098


namespace integer_solutions_count_l490_490520

noncomputable def trigonometric_inequality_solutions_count : ℤ :=
  count_in_interval (1991, 2013) (λ x, 
    √ (1 + sin (π * x / 4) - 3 * cos (π * x / 2))
    + √ 6 * sin (π * x / 4) ≥ 0)

theorem integer_solutions_count :
  trigonometric_inequality_solutions_count = 9 :=
sorry

end integer_solutions_count_l490_490520


namespace max_cardinality_of_S_l490_490305

-- Define the set S
def S : Set ℕ := { n | 1 ≤ n ∧ n ≤ 21002 }

-- Define the property that the product of any two distinct elements is not in S
def product_property (a b : ℕ) (S : Set ℕ) : Prop :=
  a ≠ b → (a ∈ S) → (b ∈ S) → (a * b) ∉ S

-- Define the maximum size property of the set S
def max_size_S : ℕ := 1958

-- The theorem statement
theorem max_cardinality_of_S : 
  ∀ S : Set ℕ, 
  (∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b → (a * b) ∉ S) → 
  (∀ T : Set ℕ, (∀ t : ℕ, t ∈ T → t ∈ S) → T.to_finset.card ≤ max_size_S) :=
by
  sorry

end max_cardinality_of_S_l490_490305


namespace quadrilateral_tangent_equality_l490_490459

theorem quadrilateral_tangent_equality
  (A B C D M N K L : Point)
  (h_inscribed : Circle (inscribed_in A B D) intersects [B,C] at M)
  (h_inscribed : Circle (inscribed_in A B D) intersects [D,C] at N) :
  (dist A B + dist B C = dist A D + dist D C) :=
sorry

end quadrilateral_tangent_equality_l490_490459


namespace rain_over_weekend_probability_l490_490185

-- Define the given probabilities
def P_R_S : ℝ := 0.6
def P_not_R_S : ℝ := 0.4
def P_R_D_given_R_S : ℝ := 0.7
def P_R_D_given_not_R_S : ℝ := 0.4

-- Prove that the probability it rains on at least one day over the weekend is 0.76
theorem rain_over_weekend_probability :
  let P_R_D := P_R_D_given_R_S * P_R_S + P_R_D_given_not_R_S * P_not_R_S in
  let P_not_R_D_given_R_S := 1 - P_R_D_given_R_S in
  let P_not_R_D_given_not_R_S := 1 - P_R_D_given_not_R_S in
  let P_rains_both_not_days := P_not_R_S * P_not_R_D_given_not_R_S in
  1 - P_rains_both_not_days = 0.76 :=
by
  -- Insert proof here
  sorry

end rain_over_weekend_probability_l490_490185


namespace runner_distance_l490_490933

theorem runner_distance :
  ∃ x t d : ℕ,
    d = x * t ∧
    d = (x + 1) * (2 * t / 3) ∧
    d = (x - 1) * (t + 3) ∧
    d = 6 :=
by
  sorry

end runner_distance_l490_490933


namespace find_a_l490_490601

def A : Set ℝ := {1, 2}
def B (a : ℝ) : Set ℝ := {x | (x^2 + a * x) * (x^2 + a * x + 2) = 0}

def n (s : Set ℝ) [Fintype s] := Fintype.card s

def m (A B : Set ℝ) [Fintype A] [Fintype B] : ℝ :=
  if n A ≥ n B then n A - n B else n B - n A

variable (a : ℝ)
variable [Fact (m A (B a) = 1)]

theorem find_a :
  a = 2 * Real.sqrt 2 :=
sorry

end find_a_l490_490601


namespace area_of_region_l490_490500

open Set

theorem area_of_region:
  let R := {p : ℝ × ℝ | |p.1 - 2| ≤ p.2 ∧ p.2 ≤ 5 - |p.1 + 1|} in
  measurable_set R →
  ∫ p in R, 1 = 10 :=
by
  sorry

end area_of_region_l490_490500


namespace initial_red_pens_l490_490095

theorem initial_red_pens :
  ∀ (R : ℕ), 
    let initial_blue_pens := 9 in
    let initial_black_pens := 21 in
    let remaining_pens := 25 in
    let removed_blue_pens := 4 in
    let removed_black_pens := 7 in
    let remaining_blue_pens := initial_blue_pens - removed_blue_pens in
    let remaining_black_pens := initial_black_pens - removed_black_pens in
    remaining_blue_pens + remaining_black_pens + R = remaining_pens → R = 6 :=
sorry

end initial_red_pens_l490_490095


namespace sum_of_solutions_l490_490795

theorem sum_of_solutions (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) : 
  -b / a = 9 :=
by
  -- The proof is omitted here (hence the 'sorry')
  sorry

end sum_of_solutions_l490_490795


namespace distinct_four_digit_even_numbers_l490_490606

theorem distinct_four_digit_even_numbers : 
  let even_digits := {0, 2, 4, 6, 8} in
  let choices_thousands := {2, 4, 6, 8} in
  let choices_hundreds := even_digits in
  let choices_tens := even_digits in
  let choices_units := even_digits in
  ∃ total : ℕ, total = 4 * 5 * 5 * 5 ∧ total = 500 :=
by
  sorry

end distinct_four_digit_even_numbers_l490_490606


namespace sum_of_solutions_l490_490849

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | x^2 = 9*x - 20}, x) = 9 := 
sorry

end sum_of_solutions_l490_490849


namespace joy_pencils_count_l490_490670

theorem joy_pencils_count :
  ∃ J, J = 30 ∧ (∃ (pencils_cost_J pencils_cost_C : ℕ), 
  pencils_cost_C = 50 * 4 ∧ pencils_cost_J = pencils_cost_C - 80 ∧ J = pencils_cost_J / 4) := sorry

end joy_pencils_count_l490_490670


namespace range_of_g_l490_490968

noncomputable def g (x : ℝ) : ℝ := (Real.arccos (x / 3))^2 + (Real.pi / 2) * Real.arcsin (x / 3) - (Real.arcsin (x / 3))^2 + (Real.pi ^ 2 / 18) * (x^2 + 9 * x + 27)

theorem range_of_g : 
  ∀ x, x ∈ set.Icc (-3 : ℝ) (3 : ℝ) →
  (Real.arccos (x / 3) + Real.arcsin (x / 3) = Real.pi / 2) → 
  ∃ (y ∈ set.Icc (Real.pi ^ 2) (13 * (Real.pi ^ 2) / 4)), g x = y :=
sorry

end range_of_g_l490_490968


namespace relationship_among_abc_l490_490215

noncomputable def a : ℝ := (1/2)^(1/3)
noncomputable def b : ℝ := Real.log 2 / Real.log (1/3)
noncomputable def c : ℝ := Real.log 3 / Real.log (1/2)

theorem relationship_among_abc : a > b ∧ b > c :=
by {
  sorry
}

end relationship_among_abc_l490_490215


namespace rotation_180_counterclockwise_l490_490897

open Complex

theorem rotation_180_counterclockwise (z : ℂ) (h : z = -6 - 3 * I) :
    exp (π * I) * z = 6 + 3 * I :=
by
  rw [←h]
  simp
  norm_num
  sorry

end rotation_180_counterclockwise_l490_490897


namespace last_year_winner_time_l490_490397

/-- 
Given:
1. The length of the town square is 3/4 of a mile.
2. Participants run around the town square 7 times.
3. The winner finishes the race in 42 minutes.
4. This year's winner ran one mile of the race on average 1 minute faster compared to last year.

Prove:
Last year's winner took 47 minutes to finish the race.
-/
theorem last_year_winner_time 
  (length_of_town_square : ℚ)
  (laps : ℕ)
  (time_this_year : ℕ)
  (time_diff_per_mile : ℚ) : 
  (length_of_town_square = 3/4) →
  (laps = 7) →
  (time_this_year = 42) →
  (time_diff_per_mile = 1) →
  ∃ time_last_year : ℚ, time_last_year = 47 :=
begin
  intros h1 h2 h3 h4,
  -- detailed proof steps will be added here
  sorry
end

end last_year_winner_time_l490_490397


namespace bucket_capacity_l490_490894

theorem bucket_capacity :
  (∃ (x : ℝ), 30 * x = 45 * 9) → 13.5 = 13.5 :=
by
  -- proof needed
  sorry

end bucket_capacity_l490_490894


namespace sum_of_solutions_of_quadratic_eq_l490_490820

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 9 * x + 20 = 0

-- Prove that the sum of the solutions to this equation is 9
theorem sum_of_solutions_of_quadratic_eq : 
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 9) := 
begin
  -- Proof is omitted
  sorry
end

end sum_of_solutions_of_quadratic_eq_l490_490820


namespace tan_alpha_equals_neg_two_l490_490214

theorem tan_alpha_equals_neg_two (α : ℝ) 
  (h : sin (π - α) = -2 * sin (π / 2 + α)) : 
  tan α = -2 := 
by
  sorry

end tan_alpha_equals_neg_two_l490_490214


namespace correctNumberOfWays_l490_490717

open Nat

def binomial (n k : ℕ) := n.choose k

def derangement (n : ℕ) : ℕ := -- Assuming derangements formula provided somewhere
  nat.size (perm.derangements (fin n))

noncomputable def numberOfWays : ℕ := 
  binomial 10 3 * derangement 3

theorem correctNumberOfWays : numberOfWays = 240 := by
  sorry

end correctNumberOfWays_l490_490717


namespace arithmetic_sequence_card_draw_l490_490404

-- Declare the problem setup and assumptions.
constant Boxes : Type
constant A B C : Boxes
constant cards : Boxes → Finset ℕ
constant numbers : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Assume each box contains numbers 1 to 6.
axiom cards_in_A : cards A = numbers
axiom cards_in_B : cards B = numbers
axiom cards_in_C : cards C = numbers

-- Proof that the number of ways to draw cards forming an arithmetic sequence is 18.
theorem arithmetic_sequence_card_draw : 
  -- In three boxes, each with cards numbered {1, 2, 3, 4, 5, 6}, 
  -- there are exactly 18 ways to draw one card from each box 
  -- such that the three numbers form an arithmetic sequence.
  ∃ (draws : Finset (ℕ × ℕ × ℕ)), 
    (∀ (x ∈ draws), (∃ a : ℕ, ∃ d : ℕ, x = (a, a+d, a+2*d) ∨ x = (a, a-d, a-2*d)) ∧
     ∀ y ∈ {A, B, C}, y ∈ {x.1, x.2, x.3}) ∧ draws.card = 18 :=
sorry

end arithmetic_sequence_card_draw_l490_490404


namespace cone_prism_ratio_l490_490931

theorem cone_prism_ratio 
  (a b h_c h_p : ℝ) (hb_lt_a : b < a) : 
  (π * b * h_c) / (12 * a * h_p) = (1 / 3 * π * b^2 * h_c) / (4 * a * b * h_p) :=
by
  sorry

end cone_prism_ratio_l490_490931


namespace find_conjugate_product_l490_490261

-- Define the complex number z satisfying the given condition
def complex_num_condition (z : ℂ) : Prop :=
  z * (1 + complex.i) = 2 - complex.i

-- Prove that the complex conjugate of z multiplied by z equals 5/2
theorem find_conjugate_product (z : ℂ) (h: complex_num_condition z) : (conj z * z = 5 / 2) :=
by
-- Assuming the given condition here
admit

end find_conjugate_product_l490_490261


namespace phone_charges_equal_l490_490082

theorem phone_charges_equal (x : ℝ) : 
  (0.60 + 14 * x = 0.08 * 18) → (x = 0.06) :=
by
  intro h
  have : 14 * x = 1.44 - 0.60 := sorry
  have : 14 * x = 0.84 := sorry
  have : x = 0.06 := sorry
  exact this

end phone_charges_equal_l490_490082


namespace rectangular_to_cylindrical_l490_490959

theorem rectangular_to_cylindrical (x y z r θ : ℝ) (hx : x = 3) (hy : y = -3 * Real.sqrt 3) (hz : z = 2)
    (h_r : r = Real.sqrt (x^2 + y^2)) (h_θ : θ = Real.arctan2 y x) :
    (r = 6) ∧ (θ = 5 * Real.pi / 3) ∧ (z = 2) :=
by
  -- Definitions/conditions from problem.
  have hx : x = 3 := hx
  have hy : y = -3 * Real.sqrt 3 := hy
  have hz : z = 2 := hz
  have h_r : r = Real.sqrt (x^2 + y^2) := h_r
  have h_θ : θ = Real.arctan2 y x := h_θ
  sorry -- placeholder for detailed proof steps

end rectangular_to_cylindrical_l490_490959


namespace min_ap_pb_l490_490218

theorem min_ap_pb {A B P : Type} [MetricSpace A] [MetricSpace B] [MetricSpace P]
  (AB_length : dist A B = 8)
  (P_distance : ∃ H : Type, dist P H = 3 ∧ H ∈ line AB) :
  ∃ Q : ℝ, Q = 24 ∧ ∀ AP PB : ℝ, AP * PB ≥ Q :=
sorry

end min_ap_pb_l490_490218


namespace sum_of_solutions_l490_490792

-- Define the quadratic equation and variable x
def quadratic_equation := ∀ x : ℝ, (x^2 - 9 * x + 20 = 0)

-- Define what we need to prove
theorem sum_of_solutions : ∃ s : ℝ, (∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → s = x1 + x2) ∧ s = 9 :=
by
  sorry -- Proof is omitted

end sum_of_solutions_l490_490792


namespace part1_part2_part3_l490_490167

-- Part 1 of the problem
theorem part1 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (sqrt (a^3 * b^2 * cbrt (a * b^2)) / (a^(1/4) * b^(1/2))^4 * a^(-1/3) * b^(1/3)) = a / b :=
sorry

-- Part 2 of the problem (no specific conditions needed)
theorem part2 : 
  ((-27 / 8 : ℚ) ^ (-2 / 3) + (0.002 : ℚ) - (1 / 2 : ℚ) - 10 * (sqrt 5 - 2)⁻¹ + (sqrt 2 - sqrt 3) ^ 0) = - (167 / 9) :=
sorry

-- Part 3 of the problem
theorem part3 (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ((5 / 6 * a^(1/3) * b^(-2) * (-3 * a - (1 / 2) * b^(-1))) / (4 * a^(2/3) * b^(-3))^(1/2)) = - (5 * sqrt (a * b) / (4 * a * b^2)) :=
sorry

end part1_part2_part3_l490_490167


namespace sarah_bus_time_l490_490345

noncomputable def totalTimeAway : ℝ := (4 + 15/60) + (5 + 15/60)  -- 9.5 hours
noncomputable def totalTimeAwayInMinutes : ℝ := totalTimeAway * 60  -- 570 minutes

noncomputable def timeInClasses : ℝ := 8 * 45  -- 360 minutes
noncomputable def timeInLunch : ℝ := 30  -- 30 minutes
noncomputable def timeInExtracurricular : ℝ := 1.5 * 60  -- 90 minutes
noncomputable def totalTimeInSchoolActivities : ℝ := timeInClasses + timeInLunch + timeInExtracurricular  -- 480 minutes

noncomputable def timeOnBus : ℝ := totalTimeAwayInMinutes - totalTimeInSchoolActivities  -- 90 minutes

theorem sarah_bus_time : timeOnBus = 90 := by
  sorry

end sarah_bus_time_l490_490345


namespace factor_81_minus_27_x_cubed_l490_490978

theorem factor_81_minus_27_x_cubed (x : ℝ) : 
  81 - 27 * x ^ 3 = 27 * (3 - x) * (9 + 3 * x + x ^ 2) :=
by sorry

end factor_81_minus_27_x_cubed_l490_490978


namespace slowest_bailing_rate_correct_l490_490058

noncomputable def slowest_bailing_rate (distance : ℝ) (leak_rate : ℝ) (sink_threshold : ℝ) (rowing_speed : ℝ) : ℝ :=
  let t := distance / rowing_speed * 60 -- converting hours to minutes
  let total_intake := leak_rate * t
  let bailing_rate := (total_intake - sink_threshold) / t
  bailing_rate

theorem slowest_bailing_rate_correct :
  slowest_bailing_rate 2 8 50 3 = 7 :=
by
  -- Definitions
  let distance := 2
  let leak_rate := 8
  let sink_threshold := 50
  let rowing_speed := 3
  let t := distance / rowing_speed * 60 -- time in minutes
  let total_intake := leak_rate * t
  let bailing_rate := (total_intake - sink_threshold) / t

  -- Calculation
  have h1: t = 40 := by
    calc t = (2 / 3) * 60 : by rw [distance, rowing_speed]
       ... = 40 : by norm_num

  have h2: total_intake = 320 := by
    calc total_intake = 8 * 40 : by rw [leak_rate, h1]
                    ... = 320 : by norm_num

  have h3: bailing_rate = (320 - 50) / 40 := by
    calc bailing_rate = ((total_intake - sink_threshold) / t) : by rw [total_intake, sink_threshold, h1]

  have h4: bailing_rate = 6.75 := by
    calc bailing_rate = (320 - 50) / 40 : by rw h3
                    ... = 270 / 40 : by norm_num
                    ... = 6.75 : by norm_num

  -- Conclusion
  have answer := 7
  exact eq_of_le_of_lt (le_of_eq (floor_eq_iff.mpr ⟨le_refl 6.75, by norm_num⟩)) have rfl ➔ sorry

end slowest_bailing_rate_correct_l490_490058


namespace sum_of_roots_l490_490995

theorem sum_of_roots (x y : ℝ) (h : ∀ z, z^2 + 2023 * z - 2024 = 0 → z = x ∨ z = y) : x + y = -2023 := 
by
  sorry

end sum_of_roots_l490_490995


namespace minimal_polynomial_degree_l490_490522

theorem minimal_polynomial_degree (r1 r2 r3 r4 : ℝ)
  (h₁ : r1 = 2 + Real.sqrt 5)
  (h₂ : r2 = 2 - Real.sqrt 5)
  (h₃ : r3 = 3 + Real.sqrt 7)
  (h₄ : r4 = 3 - Real.sqrt 7) :
  ∃ (P : Polynomial ℚ), 
    (P.leadingCoeff = 1) ∧
    (P.eval r1 = 0) ∧
    (P.eval r2 = 0) ∧
    (P.eval r3 = 0) ∧
    (P.eval r4 = 0) ∧ 
    (P.degree = 4) ∧
    (P = Polynomial.X^4 - 10 * Polynomial.X^3 + 33 * Polynomial.X^2 - 26 * Polynomial.X + 2) := sorry

end minimal_polynomial_degree_l490_490522


namespace optimal_telegraph_pole_l490_490068

theorem optimal_telegraph_pole
  (intervals : ℕ := 28)
  (dodson_walk : ℚ := 9)
  (williams_walk : ℚ := 11)
  (ride : ℚ := 3)
  (k : ℕ) :
  (k ≤ intervals) →
  let dodson_time := dodson_walk - (6 * k / intervals)
  let williams_time := 3 + (8 * k / intervals)
  dodson_time = williams_time →
  k = 12 := by
  intros hk eq_time
  have : dodson_walk - (6 * k / intervals) = williams_time,
  { rw eq_time },
  have k_eq : 9 - 3 = (8 * k + 6 * k) / intervals,
  { exact eq_time },
  rw [intervals_def, mul_comm, mul_assoc, <-div_eq_mul_inv] at k_eq,
  have h : 6 = k / 2,
  { linarith },
  linarith

end optimal_telegraph_pole_l490_490068


namespace total_pieces_correct_l490_490969

-- Definitions based on conditions
def rods_in_row (n : ℕ) : ℕ := 3 * n
def connectors_in_row (n : ℕ) : ℕ := n

-- Sum of natural numbers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Total rods in ten rows
def total_rods : ℕ := 3 * sum_first_n 10

-- Total connectors in eleven rows
def total_connectors : ℕ := sum_first_n 11

-- Total pieces
def total_pieces : ℕ := total_rods + total_connectors

-- Theorem to prove
theorem total_pieces_correct : total_pieces = 231 :=
by
  sorry

end total_pieces_correct_l490_490969


namespace Kim_drink_amount_l490_490037

namespace MathProof

-- Define the conditions
variable (milk_initial t_drinks k_drinks : ℚ)
variable (H1 : milk_initial = 3/4)
variable (H2 : t_drinks = 1/3 * milk_initial)
variable (H3 : k_drinks = 1/2 * (milk_initial - t_drinks))

-- Theorem statement
theorem Kim_drink_amount : k_drinks = 1/4 :=
by
  sorry -- Proof steps would go here, but we're just setting up the statement

end MathProof

end Kim_drink_amount_l490_490037


namespace trains_crossing_time_l490_490065

section train_problem

  -- Define the lengths of the two trains
  def length_train_A : ℝ := 200 -- in meters
  def length_train_B : ℝ := 180 -- in meters

  -- Define the speeds of the two trains
  def speed_train_A : ℝ := 40 -- in km/h
  def speed_train_B : ℝ := 45 -- in km/h

  -- Convert the speeds to m/s
  def speed_in_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 1000 / 3600

  -- Define the relative speed in m/s
  def relative_speed : ℝ := speed_in_mps speed_train_B - speed_in_mps speed_train_A

  -- Define total distance to be covered
  def total_distance : ℝ := length_train_A + length_train_B

  -- Define the time taken for the trains to cross each other
  def time_to_cross : ℝ := total_distance / relative_speed

  -- Final theorem statement
  theorem trains_crossing_time : time_to_cross ≈ 273.38 := 
  by sorry

end train_problem

end trains_crossing_time_l490_490065


namespace projection_of_a_minus_b_onto_b_is_neg_three_halves_b_l490_490568

variables (a b : ℝ^3)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)
variables (angle_ab : real.angle a b = real.angle.pi * 2 / 3)

theorem projection_of_a_minus_b_onto_b_is_neg_three_halves_b : 
  (a - b) ⬝ b / ∥b∥^2 • b = -3/2 • b :=
by sorry

end projection_of_a_minus_b_onto_b_is_neg_three_halves_b_l490_490568


namespace total_litter_l490_490741

theorem total_litter (glass_bottles : ℕ) (aluminum_cans : ℕ) (plastic_bags : ℕ) (miscellaneous_fraction : ℚ) :
  glass_bottles = 10 →
  aluminum_cans = 8 →
  plastic_bags = 12 →
  miscellaneous_fraction = 1 / 4 →
  let non_miscellaneous := glass_bottles + aluminum_cans + plastic_bags in
  let total_litter := non_miscellaneous / (3/4) in
  total_litter = 40 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  let non_miscellaneous := 10 + 8 + 12
  have h5 : non_miscellaneous = 30 := by norm_num
  let total_litter := non_miscellaneous * 4 / 3
  have h6 : total_litter = 30 * 4 / 3 := rfl
  have h7 : total_litter = 40 := by norm_num
  exact h7

end total_litter_l490_490741


namespace monotonic_intervals_min_value_range_l490_490235

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 - 9 * x

theorem monotonic_intervals :
  (∀ x, x < -3 → (f' x > 0)) ∧ 
  (∀ x, -3 < x ∧ x < 1 → (f' x < 0)) ∧
  (∀ x, x > 1 → (f' x > 0)) := 
begin
  sorry
end

theorem min_value_range (c : ℝ) : 
  (∀ x, -4 ≤ x ∧ x ≤ c → f x ≥ -5) → c ∈ set.Ici 1 :=
begin
  sorry
end

end monotonic_intervals_min_value_range_l490_490235


namespace equal_arcs_in_semicircle_l490_490470

-- Some preliminary definitions might be necessary here
-- e.g., the definition of Equilateral triangle, semicircle, dividing points, and arcs on the semicircle

/-- 
Given an equilateral triangle ABC, 
with points P and Q on segment BC such that BP = PQ = QC = BC / 3,
and a semicircle K with diameter BC on the side opposite to vertex A,
where rays AP and AQ intersect K at points X and Y respectively,
prove that arcs BX, XY, and YC are all equal.
 -/
theorem equal_arcs_in_semicircle 
  (a b c p q x y : Point) -- Points in the problem
  (h_equilateral : EquilateralTriangle a b c)
  (h_bc_eq : ∥b - c∥ = 3 * ∥b - p∥)
  (h_pq_eq : ∥b - p∥ = ∥p - q∥)
  (h_qc_eq : ∥p - q∥ = ∥q - c∥)
  (h_k_semicircle : Semicircle b c)
  (h_ap_inter_x : Ray a p ∩ h_k_semicircle = {x})
  (h_aq_inter_y : Ray a q ∩ h_k_semicircle = {y}) : 
  ArcEqual (Arc b x) (Arc x y) ∧ ArcEqual (Arc x y) (Arc y c) := 
begin
  sorry
end

end equal_arcs_in_semicircle_l490_490470


namespace find_ac_bd_l490_490075

variable (a b c d : ℝ)

axiom cond1 : a^2 + b^2 = 1
axiom cond2 : c^2 + d^2 = 1
axiom cond3 : a * d - b * c = 1 / 7

theorem find_ac_bd : a * c + b * d = 4 * Real.sqrt 3 / 7 := by
  sorry

end find_ac_bd_l490_490075


namespace antonieta_tickets_needed_l490_490476

-- Definitions based on conditions:
def ferris_wheel_tickets : ℕ := 6
def roller_coaster_tickets : ℕ := 5
def log_ride_tickets : ℕ := 7
def antonieta_initial_tickets : ℕ := 2

-- Theorem to prove the required number of tickets Antonieta should buy
theorem antonieta_tickets_needed : ferris_wheel_tickets + roller_coaster_tickets + log_ride_tickets - antonieta_initial_tickets = 16 :=
by
  sorry

end antonieta_tickets_needed_l490_490476


namespace function_increasing_intervals_l490_490581

theorem function_increasing_intervals :
  ∀ (k : ℤ), 
  ∃ (f : ℝ → ℝ), 
  (f = λ x, sin (2 * x - π / 6)) ∧
  (∀ x, (k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3) ↔ (differentiable_at ℝ f x ∧ f' x > 0)) :=
by 
  sorry

end function_increasing_intervals_l490_490581


namespace field_trip_students_l490_490387

theorem field_trip_students 
  (seats_per_bus : ℕ) 
  (buses_needed : ℕ) 
  (total_students : ℕ) 
  (h1 : seats_per_bus = 2) 
  (h2 : buses_needed = 7) 
  (h3 : total_students = seats_per_bus * buses_needed) : 
  total_students = 14 :=
by 
  rw [h1, h2] at h3
  assumption

end field_trip_students_l490_490387


namespace evaluate_ninth_roots_of_unity_product_l490_490512

theorem evaluate_ninth_roots_of_unity_product : 
  (3 - Complex.exp (2 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (4 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (6 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (8 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (10 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (12 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (14 * Real.pi * Complex.I / 9)) *
  (3 - Complex.exp (16 * Real.pi * Complex.I / 9)) 
  = 9841 := 
by 
  sorry

end evaluate_ninth_roots_of_unity_product_l490_490512


namespace trig_values_l490_490575

theorem trig_values (x y : ℤ) (r : ℝ) (α : ℝ)
  (h1: x = -4)
  (h2: y = 3)
  (h3 : r = Real.sqrt (x^2 + y^2))
  (h4 : sin α = y / r)
  (h5 : cos α = x / r)
  (h6 : tan α = y / x) :
  sin α = 3 / 5 ∧ cos α = -4 / 5 ∧ tan α = -3 / 4 := by
  sorry

end trig_values_l490_490575


namespace variance_of_data_is_0_02_l490_490553

def data : List ℝ := [10.1, 9.8, 10, 9.8, 10.2]

theorem variance_of_data_is_0_02 (h : (10.1 + 9.8 + 10 + 9.8 + 10.2) / 5 = 10) : 
  (1 / 5) * ((10.1 - 10) ^ 2 + (9.8 - 10) ^ 2 + (10 - 10) ^ 2 + (9.8 - 10) ^ 2 + (10.2 - 10) ^ 2) = 0.02 :=
by
  sorry

end variance_of_data_is_0_02_l490_490553


namespace divisors_congruent_mod8_l490_490695

theorem divisors_congruent_mod8 (n : ℕ) (hn : n % 2 = 1) :
  ∀ d, d ∣ (2^n - 1) → d % 8 = 1 ∨ d % 8 = 7 :=
by
  sorry

end divisors_congruent_mod8_l490_490695


namespace new_rectangle_area_not_equal_circle_area_l490_490357

-- Define the dimensions of the given rectangle
variable (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)

def diagonal (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

def new_base (a b : ℝ) : ℝ := 2 * a * b

def new_altitude (a : ℝ) : ℝ := (a * diagonal a b) / 2

def new_area (a b : ℝ) : ℝ := 
  new_base a b * new_altitude a

def circle_area (b : ℝ) : ℝ := π * b^2

theorem new_rectangle_area_not_equal_circle_area :
  new_area a b ≠ circle_area b := by
  sorry

end new_rectangle_area_not_equal_circle_area_l490_490357


namespace reciprocal_of_one_fifth_l490_490020

theorem reciprocal_of_one_fifth : (∃ x : ℚ, (1/5) * x = 1 ∧ x = 5) :=
by
  -- The proof goes here, for now we assume it with sorry
  sorry

end reciprocal_of_one_fifth_l490_490020


namespace trig_expression_equality_l490_490125

-- Define the trigonometric values
def sin_30 : ℝ := 1/2
def cos_45 : ℝ := real.sqrt 2 / 2
def tan_60 : ℝ := real.sqrt 3

-- The theorem to prove
theorem trig_expression_equality : 
  sin_30 - real.sqrt 3 * cos_45 + real.sqrt 2 * tan_60 = (1 + real.sqrt 6) / 2 :=
by 
  -- Proof steps are skipped
  sorry

end trig_expression_equality_l490_490125


namespace number_of_elements_in_Z_l490_490260

variable (A B : Set ℤ)

def set_A : Set ℤ := {-1, 1}
def set_B : Set ℤ := {0, 2}
def set_Z : Set ℤ := {z | ∃ x ∈ set_A, ∃ y ∈ set_B, z = x + y}

theorem number_of_elements_in_Z : Set.card set_Z = 3 := by
  sorry

end number_of_elements_in_Z_l490_490260


namespace omega_range_l490_490303

open Real

theorem omega_range (ω : ℝ) (a b : ℝ) (h1 : 0 < ω)
  (h2 : π ≤ a) (h3 : a < b) (h4 : b ≤ 2 * π) (h5 : sin (ω * a) + sin (ω * b) = 2) :
  ω ∈ set.Icc (9 / 4) (5 / 2) ∪ set.Ici (13 / 4) :=
sorry

end omega_range_l490_490303


namespace compare_fraction_product_l490_490137

theorem compare_fraction_product :
  (∏ i in Finset.range 461, (100 + 2*i : ℝ) / (101 + 2*i)) < (5/16 : ℝ) := 
sorry

end compare_fraction_product_l490_490137


namespace cubes_99_l490_490365

theorem cubes_99 (x : ℕ) : (7 * x + 1 = 99) ↔ (x = 14) :=
by {
  split,
  { -- Assuming 7 * x + 1 = 99, show x = 14
    intro h,
    have h1 : 7 * x = 98 := by linarith,
    have h2 : x = 14 := by linarith,
    exact h2,
  },
  { -- Assuming x = 14, show 7 * x + 1 = 99
    intro h,
    rw h,
    exact eq.refl 99,
  }
}

end cubes_99_l490_490365


namespace problem_equiv_proof_l490_490481

theorem problem_equiv_proof :
  2015 * (1 + 1999 / 2015) * (1 / 4) - (2011 / 2015) = 503 := 
by
  sorry

end problem_equiv_proof_l490_490481


namespace speed_W_B_l490_490769

-- Definitions for the conditions
def distance_W_B (D : ℝ) := 2 * D
def average_speed := 36
def speed_B_C := 20

-- The problem statement to be verified in Lean
theorem speed_W_B (D : ℝ) (S : ℝ) (h1: distance_W_B D = 2 * D) (h2: S ≠ 0 ∧ D ≠ 0)
(h3: (3 * D) / ((2 * D) / S + D / speed_B_C) = average_speed) : S = 60 := by
sorry

end speed_W_B_l490_490769


namespace smallest_possible_t_l490_490391

theorem smallest_possible_t (t : ℕ) (h₁ : 7.5 + t > 11) (h₂ : 7.5 + 11 > t) (h₃ : 11 + t > 7.5) : t = 4 :=
by
  sorry

end smallest_possible_t_l490_490391


namespace train_crossing_time_correct_l490_490251

variables (length_train length_bridge : ℕ) (speed_kmph : ℕ)
variables (convert_to_mps : ℕ → ℕ := λ kmph, kmph * 1000 / 3600)

def total_distance (length_train length_bridge : ℕ) : ℕ :=
  length_train + length_bridge

def time_to_cross (total_distance speed_mps : ℕ) : ℕ :=
  total_distance / speed_mps

theorem train_crossing_time_correct :
  length_train = 100 →
  length_bridge = 140 →
  speed_kmph = 36 →
  time_to_cross (total_distance length_train length_bridge) (convert_to_mps speed_kmph) = 24
:= by
  intros h1 h2 h3
  rw [h1, h2, h3]
  dsimp [total_distance, convert_to_mps, time_to_cross]
  -- Calculation steps to show equivalence will go here
  sorry

end train_crossing_time_correct_l490_490251


namespace remaining_distance_l490_490739

-- Definitions of conditions
def distance_to_grandmother : ℕ := 300
def speed_per_hour : ℕ := 60
def time_elapsed : ℕ := 2

-- Statement of the proof problem
theorem remaining_distance : distance_to_grandmother - (speed_per_hour * time_elapsed) = 180 :=
by 
  sorry

end remaining_distance_l490_490739


namespace seating_arrangements_l490_490640

theorem seating_arrangements (total_seats : ℕ) (people : ℕ) (empty: ℕ) (h1 : total_seats = 8) (h2 : people = 3) (h3 : empty = total_seats - people) (h4 : ∀ p (h: p ≤ people), 1 ≤ (total_seats - people) / (people + 1)) : (∑ p in range(6), (choose (empty + people) empty) * (factorial people) = 24) :=
by
  sorry

end seating_arrangements_l490_490640


namespace sum_of_solutions_of_quadratic_eq_l490_490819

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 9 * x + 20 = 0

-- Prove that the sum of the solutions to this equation is 9
theorem sum_of_solutions_of_quadratic_eq : 
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 9) := 
begin
  -- Proof is omitted
  sorry
end

end sum_of_solutions_of_quadratic_eq_l490_490819


namespace factor_81_minus_27_x_cubed_l490_490977

theorem factor_81_minus_27_x_cubed (x : ℝ) : 
  81 - 27 * x ^ 3 = 27 * (3 - x) * (9 + 3 * x + x ^ 2) :=
by sorry

end factor_81_minus_27_x_cubed_l490_490977


namespace g_evaluation_l490_490680

def g (x y : ℝ) : ℝ :=
if x - y <= 1 then (x ^ 2 * y - x + 3) / (3 * x)
else (x ^ 2 * y - y - 3) / (-3 * y)

theorem g_evaluation : g 3 2 + g 4 1 = -2 := by
  sorry

end g_evaluation_l490_490680


namespace sum_of_roots_l490_490859

theorem sum_of_roots (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) :
  ∑ x in {x | a * x^2 + b * x + c = 0}, x = 9 :=
by
  sorry

end sum_of_roots_l490_490859


namespace find_x_l490_490647

theorem find_x 
  (ABC_angle : ∠ ABC = 90)
  (DBA_angle : ∠ DBA = 3 * x)
  (DBC_angle : ∠ DBC = 2 * x) :
  x = 18 := by
  sorry

end find_x_l490_490647


namespace largest_divisor_of_n4_minus_n2_is_12_l490_490370

theorem largest_divisor_of_n4_minus_n2_is_12 : ∀ n : ℤ, 12 ∣ (n^4 - n^2) :=
by
  intro n
  -- Placeholder for proof; the detailed steps of the proof go here
  sorry

end largest_divisor_of_n4_minus_n2_is_12_l490_490370


namespace sum_of_solutions_l490_490855

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | x^2 = 9*x - 20}, x) = 9 := 
sorry

end sum_of_solutions_l490_490855


namespace no_opposite_side_middle_hit_l490_490460

def hits_opposite_side_middle (T : ℝ) (rect : Type) [IsRectangular rect] (corner : rect) (angle : ℝ) (hit_middle : rect) : Prop :=
  -- The ball hits the middle of one side at time T1
  (hit_middle = middle_of_one_side T corner rect) ∧
  -- The ball hits the middle of the opposite side at time T2 > T1
  ∃ T_2 : ℝ, T_2 > T ∧ (hit_middle = middle_of_opposite_side T_2 corner rect)

theorem no_opposite_side_middle_hit (rect : Type) [IsRectangular rect] (corner : rect) :
  ∀ (T : ℝ) (angle : ℝ) (hit_middle : rect), 
  angle = 45 ∧ hits_opposite_side_middle T rect corner angle hit_middle → false := by
  sorry

end no_opposite_side_middle_hit_l490_490460


namespace wire_service_reporters_l490_490171

theorem wire_service_reporters (total_reporters covering_local_politics : ℕ) 
  (h1 : total_reporters = 100) 
  (h2 : covering_local_politics = 18) 
  (h3 : covering_local_politics = 0.6 * (total_reporters - covering_local_politics + covering_local_politics) ∧ covering_local_politics = 0.6 * covering_local_politics):
  (total_reporters - (covering_local_politics / 0.6)) / total_reporters = 0.7 :=
by
  sorry

end wire_service_reporters_l490_490171


namespace sum_of_roots_l490_490857

theorem sum_of_roots (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) :
  ∑ x in {x | a * x^2 + b * x + c = 0}, x = 9 :=
by
  sorry

end sum_of_roots_l490_490857


namespace bracelet_display_capacity_l490_490911

namespace JewelryStore

def necklace_capacity := 12
def necklace_current := 5
def necklace_needed := necklace_capacity - necklace_current

def ring_capacity := 30
def ring_current := 18
def ring_needed := ring_capacity - ring_current

def bracelet_current := 8
def bracelet_cost := 5
def total_payment := 183

def necklace_cost := 4
def ring_cost := 10

-- Calculate the total cost to fill the necklace stand and the ring display
def total_cost_necklaces_rings := (necklace_needed * necklace_cost) + (ring_needed * ring_cost)

-- Calculate the remaining amount for bracelets
def remaining_amount := total_payment - total_cost_necklaces_rings

-- Calculate the number of additional bracelets
def additional_bracelets := remaining_amount / bracelet_cost

-- Calculate the total number of bracelets the bracelet display can hold
def bracelet_total := bracelet_current + additional_bracelets

theorem bracelet_display_capacity : bracelet_total = 15 := 
by
  simp [necklace_needed, ring_needed, necklace_cost, ring_cost, total_cost_necklaces_rings, remaining_amount, additional_bracelets, bracelet_current, bracelet_cost]
  rfl
  -- sorry

end JewelryStore

end bracelet_display_capacity_l490_490911


namespace count_integers_in_range_l490_490249

theorem count_integers_in_range : 
  {n : ℤ | -4 ≤ n ∧ n ≤ 9}.toFinset.card = 14 :=
by
  sorry

end count_integers_in_range_l490_490249


namespace new_milk_to_water_ratio_is_six_to_five_l490_490435

noncomputable def milk_to_water_ratio_after_adding_water
  (initial_mixture_volume : ℕ)
  (initial_milk_to_water_ratio : ℕ × ℕ)
  (additional_water_volume : ℕ)
  : ℕ × ℕ :=
let (milk_ratio, water_ratio) := initial_milk_to_water_ratio in
let total_initial_parts := milk_ratio + water_ratio in
let part_volume := initial_mixture_volume / total_initial_parts in
let initial_milk_volume := milk_ratio * part_volume in
let initial_water_volume := water_ratio * part_volume in
let new_water_volume := initial_water_volume + additional_water_volume in
let gcd := Nat.gcd initial_milk_volume new_water_volume in
(initial_milk_volume / gcd, new_water_volume / gcd)

theorem new_milk_to_water_ratio_is_six_to_five :
  milk_to_water_ratio_after_adding_water 45 (4, 1) 21 = (6, 5) :=
sorry

end new_milk_to_water_ratio_is_six_to_five_l490_490435


namespace camera_sticker_price_l490_490602

theorem camera_sticker_price (p : ℝ)
  (h1 : p > 0)
  (hx : ∀ x, x = 0.80 * p - 50)
  (hy : ∀ y, y = 0.65 * p)
  (hs : 0.80 * p - 50 = 0.65 * p - 40) :
  p = 666.67 :=
by sorry

end camera_sticker_price_l490_490602


namespace number_of_sides_of_polygon_l490_490273

theorem number_of_sides_of_polygon (sum_exterior_angles : ℝ) (exterior_angle : ℝ) (h1 : sum_exterior_angles = 360) (h2 : exterior_angle = 45) : (n : ℕ), n = sum_exterior_angles / exterior_angle :=
by
  have h3 : sum_exterior_angles = 360 := h1
  have h4 : exterior_angle = 45 := h2
  let n := 360 / 45
  exact sorry

end number_of_sides_of_polygon_l490_490273


namespace flavoring_corn_syrup_ratio_comparison_l490_490653

-- Definitions and conditions derived from the problem
def standard_flavoring_to_water_ratio : ℝ := 1 / 30
def sport_flavoring_to_water_ratio : ℝ := standard_flavoring_to_water_ratio / 2
def sport_water_amount : ℝ := 75
def sport_flavoring_amount : ℝ := sport_water_amount / 60
def sport_corn_syrup_amount : ℝ := 5
def sport_flavoring_to_corn_syrup_ratio : ℝ := sport_flavoring_amount / sport_corn_syrup_amount
def standard_flavoring_to_corn_syrup_ratio : ℝ := 1 / 12

-- The statement to be proved
theorem flavoring_corn_syrup_ratio_comparison :
  sport_flavoring_to_corn_syrup_ratio / standard_flavoring_to_corn_syrup_ratio = 3 :=
by
  have h_sport_flavoring_to_corn_syrup : sport_flavoring_to_corn_syrup_ratio = 1 / 4,
  sorry

  have h_standard_flavoring_to_corn_syrup : standard_flavoring_to_corn_syrup_ratio = 1 / 12,
  sorry

  calc
    sport_flavoring_to_corn_syrup_ratio / standard_flavoring_to_corn_syrup_ratio
        = (1 / 4) / (1 / 12) : by rw [h_sport_flavoring_to_corn_syrup, h_standard_flavoring_to_corn_syrup]
    ... = (1 / 4) * 12 / 1 : by sorry
    ... = 12 / 4 : by sorry
    ... = 3 : by sorry

end flavoring_corn_syrup_ratio_comparison_l490_490653


namespace num_dissimilar_terms_expansion_l490_490160

theorem num_dissimilar_terms_expansion : 
  (∀ (i j k l : ℕ), i + j + k + l = 10) → 
  (finset.card {nusms : finset (ℕ × ℕ × ℕ × ℕ) | 
    ∃ (i j k l : ℕ), nusms = (i, j, k, l) ∧ i + j + k + l = 10}) = 286 :=
by 
  filter.hyp : (∀ (i j k l : ℕ), i + j + k + l = 10),
  exact finset.card
    {nusms : finset (ℕ × ℕ × ℕ × ℕ) | 
      ∃ (i j k l : ℕ), nusms = (i, j, k, l)
      ∧ i + j + k + l = 10} = 286,
  sorry

end num_dissimilar_terms_expansion_l490_490160


namespace sufficient_condition_for_two_zeros_l490_490231

theorem sufficient_condition_for_two_zeros
  {a : ℝ}
  (f : ℝ → ℝ := λ x, if x ≤ 1 then 2^x - a else -x + a) :
  (∃ x y, f x = 0 ∧ f y = 0 ∧ x ≠ y) → 1 < a ∧ a ≤ 2 :=
by
  sorry

end sufficient_condition_for_two_zeros_l490_490231


namespace sum_of_roots_quadratic_eq_l490_490845

theorem sum_of_roots_quadratic_eq :
  (∑ x in Finset.filter (λ x, x^2 = 9 * x - 20) (Finset.range 100), x) = 9 :=
begin
  sorry
end

end sum_of_roots_quadratic_eq_l490_490845


namespace box_paperclips_l490_490943

noncomputable def efficiency (v : ℝ) : ℝ := 
  1 - 0.05 * ((v / 36) - 1)

noncomputable def num_paperclips (v : ℝ) : ℝ :=
  (120 / 36) * (v * efficiency(v))

theorem box_paperclips :
  num_paperclips 90 = 263 :=
by
  unfold num_paperclips
  unfold efficiency
  have h1 : efficiency 90 = 0.875 := by norm_num
  rw [h1]
  norm_num
  have h2 : (120 / 36) * (90 * 0.875) = 262.5 := by norm_num
  rw [h2]
  norm_num
  sorry

end box_paperclips_l490_490943


namespace percentage_of_first_class_l490_490333

/-- A proof that the percentage of passengers in first class is 10%, given specific conditions. -/
theorem percentage_of_first_class 
  (total_passengers : ℕ)
  (percent_female : ℝ)
  (females_in_coach : ℕ)
  (frac_first_class_male : ℝ)
  (fraction_is_pct : 0 ≤ percent_female ∧ percent_female ≤ 1 ∧ 0 ≤ frac_first_class_male ∧ frac_first_class_male < 1)
  (h_total : total_passengers = 120)
  (h_female_pct : percent_female = 0.55)
  (h_females_in_coach : females_in_coach = 58)
  (h_frac_first_class_male : frac_first_class_male = 1/3) : 
  (total_passengers * ((8 : ℕ) / (2 : ℝ / 3))).to_nat / total_passengers * 100 = 10 := 
by 
  sorry

end percentage_of_first_class_l490_490333


namespace side_length_each_piece_l490_490051

-- Define the side length of the cake and the number of pieces.
def side_length_cake := 15
def num_pieces := 9

-- Prove the side length of each piece.
theorem side_length_each_piece (s : ℕ) (h : s * num_pieces = side_length_cake) : s = 5 := by
  have total_area := side_length_cake ^ 2
  have area_each_piece := s ^ 2
  have num_pieces_eq := total_area / area_each_piece
  have h_total : total_area = 225 := by sorry
  have h_piece : area_each_piece = 25 := by sorry
  exact h.symm.trans (nat.div_eq_of_eq_mul_left) 5 sorry

end side_length_each_piece_l490_490051


namespace base_eight_to_base_ten_l490_490420

theorem base_eight_to_base_ten : (5 * 8^1 + 2 * 8^0) = 42 := by
  sorry

end base_eight_to_base_ten_l490_490420


namespace distribute_pencils_l490_490508

variables {initial_pencils : ℕ} {num_containers : ℕ} {additional_pencils : ℕ}

theorem distribute_pencils (h₁ : initial_pencils = 150) (h₂ : num_containers = 5)
                           (h₃ : additional_pencils = 30) :
  (initial_pencils + additional_pencils) / num_containers = 36 :=
by sorry

end distribute_pencils_l490_490508


namespace frustum_volume_fraction_l490_490935

noncomputable def volume_pyramid (base_edge height : ℝ) : ℝ :=
  (1 / 3) * (base_edge ^ 2) * height

noncomputable def fraction_of_frustum (base_edge height : ℝ) : ℝ :=
  let original_volume := volume_pyramid base_edge height
  let smaller_volume := volume_pyramid (base_edge / 5) (height / 5)
  let frustum_volume := original_volume - smaller_volume
  frustum_volume / original_volume

theorem frustum_volume_fraction :
  fraction_of_frustum 40 20 = 63 / 64 :=
by sorry

end frustum_volume_fraction_l490_490935


namespace inequality_of_fractions_l490_490698

theorem inequality_of_fractions (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x / (x + y)) + (y / (y + z)) + (z / (z + x)) ≤ 2 := 
by 
  sorry

end inequality_of_fractions_l490_490698


namespace remainder_product_div_17_l490_490425

theorem remainder_product_div_17 :
  (2357 ≡ 6 [MOD 17]) → (2369 ≡ 4 [MOD 17]) → (2384 ≡ 0 [MOD 17]) →
  (2391 ≡ 9 [MOD 17]) → (3017 ≡ 9 [MOD 17]) → (3079 ≡ 0 [MOD 17]) →
  (3082 ≡ 3 [MOD 17]) →
  ((2357 * 2369 * 2384 * 2391) * (3017 * 3079 * 3082) ≡ 0 [MOD 17]) :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end remainder_product_div_17_l490_490425


namespace find_n_l490_490982

theorem find_n (n : ℕ) (hn : n * n! - n! = 5040 - n!) : n = 7 :=
by
  sorry

end find_n_l490_490982


namespace length_AB_l490_490224

open Real

-- Define the polar to rectangular coordinate conversion.
noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

-- Given condition: Polar coordinate equation.
def curve_C (ρ θ : ℝ) : Prop :=
  ρ = 8 * sin θ

-- Parametric equation of the line.
def param_line (t : ℝ) : ℝ × ℝ :=
  (t, t + 2)

-- Rectangular coordinate equation of the curve.
def rectangular_curve (x y : ℝ) : Prop :=
  x^2 + y^2 = 8 * y

-- Define the distance between two points.
def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- The main theorem to be proved
theorem length_AB :
  (∀ ρ θ, curve_C ρ θ → rectangular_curve (ρ * cos θ) (ρ * sin θ)) ∧
  (∀ t, let A := (t, t + 2), B := (t, t + 2) in
   let C := (0, 4 : ℝ × ℝ), radius := 4, d := sqrt 2 in 
   let AB := distance A B in
   AB = 2 * sqrt 14) :=
  sorry

end length_AB_l490_490224


namespace find_P_l490_490304

noncomputable def P (x : ℝ) : ℝ :=
  4 * x^3 - 6 * x^2 - 12 * x

theorem find_P (a b c : ℝ) (h_root : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_roots : ∀ x, x^3 - 2 * x^2 - 4 * x - 1 = 0 ↔ x = a ∨ x = b ∨ x = c)
  (h_Pa : P a = b + 2 * c)
  (h_Pb : P b = 2 * a + c)
  (h_Pc : P c = a + 2 * b)
  (h_Psum : P (a + b + c) = -20) :
  ∀ x, P x = 4 * x^3 - 6 * x^2 - 12 * x :=
by
  sorry

end find_P_l490_490304


namespace sum_of_vars_l490_490258

variables (a b c d k p : ℝ)

theorem sum_of_vars (h1 : a^2 + b^2 + c^2 + d^2 = 390)
                    (h2 : ab + bc + ca + ad + bd + cd = 5)
                    (h3 : ad + bd + cd = k)
                    (h4 : (a * b * c * d)^2 = p) :
                    a + b + c + d = 20 :=
by
  -- placeholder for the proof
  sorry

end sum_of_vars_l490_490258


namespace zero_point_of_function_l490_490401

-- We are given a function y = x + 1
-- We need to prove that (-1, 0) is a zero point of this function

theorem zero_point_of_function : ∃ x : ℝ, y = 0 ∧ x + 1 = 0 :=
by
  use -1
  split
  · -- show that y = 0
    exact sorry
  · -- show that x + 1 = 0
    exact sorry

end zero_point_of_function_l490_490401


namespace function_symmetry_origin_l490_490015

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - x

theorem function_symmetry_origin : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end function_symmetry_origin_l490_490015


namespace num_factors_of_n_multiples_of_360_l490_490614

-- Define n
def n : ℕ := 2^12 * 3^15 * 5^9

-- Define 360, its prime factorization for clarity
def c : ℕ := 360
def factorization_360 : ℕ × ℕ × ℕ := (3, 2, 1) -- (exponent of 2, exponent of 3, exponent of 5)

-- Define the problem as a theorem
theorem num_factors_of_n_multiples_of_360 : 
  let factors_2 := finset.range (12 + 1) \ finset.range 3,
      factors_3 := finset.range (15 + 1) \ finset.range 2,
      factors_5 := finset.range (9 + 1) \ finset.range 1 in
  factors_2.card * factors_3.card * factors_5.card = 1260 := 
by
  sorry

end num_factors_of_n_multiples_of_360_l490_490614


namespace min_magnitude_lambda_a_l490_490705

def vector := ℝ × ℝ

def a1 : vector := (1, 5)
def a2 : vector := (4, -1)
def a3 : vector := (2, 1)

inductive NonNegReal : Type
| mk : (r : ℝ) → 0 ≤ r → NonNegReal

def NonNegReal.val (x : NonNegReal) : ℝ := match x with
| NonNegReal.mk r _ => r

axiom lambda1 : NonNegReal
axiom lambda2 : NonNegReal
axiom lambda3 : NonNegReal
axiom condition : lambda1.val + lambda2.val / 2 + lambda3.val / 3 = 1

-- Proving statement: the minimum value of |lambda1 a1 + lambda2 a2 + lambda3 a3| is 3sqrt(2)
theorem min_magnitude_lambda_a : 
  ∃ min_val : ℝ, min_val = 3 * Real.sqrt 2 ∧
    (∀ λ1 λ2 λ3, 
      (λ1 ≥ 0) ∧ 
      (λ2 ≥ 0) ∧ 
      (λ3 ≥ 0) ∧
      (λ1 + λ2 / 2 + λ3 / 3 = 1) →
      |(λ1 * a1.1 + λ2 * a2.1 + λ3 * a3.1, λ1 * a1.2 + λ2 * a2.2 + λ3 * a3.2)| ≥ 3 * Real.sqrt 2) := sorry

end min_magnitude_lambda_a_l490_490705


namespace hyperbola_focus_to_asymptote_distance_l490_490007

theorem hyperbola_focus_to_asymptote_distance :
  ∀ (x y : ℝ), (x ^ 2 - y ^ 2 = 1) →
  ∃ c : ℝ, (c = 1) :=
by
  sorry

end hyperbola_focus_to_asymptote_distance_l490_490007


namespace four_drivers_suffice_l490_490919

theorem four_drivers_suffice
  (one_way_trip_time : ℕ := 160) -- in minutes
  (round_trip_time : ℕ := 320) -- in minutes
  (rest_time : ℕ := 60) -- in minutes
  (time_A_returns : ℕ := 760) -- 12:40 PM in minutes from midnight
  (time_A_next_start : ℕ := 820) -- 1:40 PM in minutes from midnight
  (time_D_departs : ℕ := 785) -- 1:05 PM in minutes from midnight
  (time_A_fifth_depart : ℕ := 970) -- 4:10 PM in minutes from midnight
  (time_B_returns : ℕ := 960) -- 4:00 PM in minutes from midnight
  (time_B_sixth_depart : ℕ := 1050) -- 5:30 PM in minutes from midnight
  (time_A_fifth_complete: ℕ := 1290) -- 9:30 PM in minutes from midnight
  : 4_drivers_sufficient : ℕ :=
    if time_A_fifth_complete = 1290 then 1 else 0
-- The theorem states that if the calculated trip completion time is 9:30 PM, then 4 drivers are sufficient.
  sorry

end four_drivers_suffice_l490_490919


namespace max_watches_two_hours_l490_490318

noncomputable def show_watched_each_day : ℕ := 30 -- Time in minutes
def days_watched : ℕ := 4 -- Monday to Thursday

theorem max_watches_two_hours :
  (days_watched * show_watched_each_day) / 60 = 2 := by
  sorry

end max_watches_two_hours_l490_490318


namespace triangle_side_b_l490_490289

theorem triangle_side_b (a b : ℝ) (B C : ℝ) (B_rad C_rad A_rad : ℝ)
  (h_a : a = 8)
  (h_B_deg : B = 30)
  (h_C_deg : C = 105)
  (h_B_rad : B_rad = B / 180 * real.pi)
  (h_C_rad : C_rad = C / 180 * real.pi)
  (h_A_deg : A = 180 - B - C)
  (h_A_rad : A_rad = A / 180 * real.pi)
  (sinB : real.sin B_rad = 1 / 2)
  (sinA : real.sin A_rad = real.sqrt 2 / 2) :
  b = 4 * real.sqrt 2 :=
sorry

end triangle_side_b_l490_490289


namespace quadrilateral_construction_possible_l490_490490

theorem quadrilateral_construction_possible
(a b c d : ℝ) (A B : ℝ) 
(h_a : a = 4) 
(h_b : b = 2)
(h_c : c = 8) 
(h_d : d = 5.5)
(h_angle_sum : A + B = 225) 
: ∃ (ABCD : Type) (quadrilateral : ABCD -> Prop), 
  quadrilateral = (λ ABCD, 
    ∃ (A B C D : ABCD → ABCD)
    (ab : dist A B = a)
    (bc : dist B C = b)
    (cd : dist C D = c)
    (da : dist D A = d)
    (angle_sum : ∠A + ∠B = 225), True) :=
sorry

end quadrilateral_construction_possible_l490_490490


namespace f_f_log2_12_l490_490229

/-- Definition of the piecewise function f(x) -/
def f : ℝ → ℝ
| x := if x < 1 then 1 + Real.logb 2 (2 - x) else -(2 ^ (x - 1))

/-- Statement to prove that f(f(log₂(12))) = 4 -/
theorem f_f_log2_12 : f (f (Real.logb 2 12)) = 4 := 
by 
  sorry

end f_f_log2_12_l490_490229


namespace max_value_of_function_l490_490371

theorem max_value_of_function (x : ℝ) : 
  (∀ x, (-1 : ℝ) ≤ real.cos x ∧ real.cos x ≤ (1 : ℝ)) → 
  (∃ y, y = (3 : ℝ) ∧ 
    ∀ y, y = (2 + real.cos x) / (2 - real.cos x) → 
    (1 / (3 : ℝ)) ≤ y ∧ y ≤ (3 : ℝ)) :=
begin
  intros h,
  sorry,
end

end max_value_of_function_l490_490371


namespace length_BD_in_isosceles_triangle_l490_490634

theorem length_BD_in_isosceles_triangle
    (A B C D H : Type)
    (AC BC : ℝ)
    (AB : ℝ)
    (CD : ℝ)
    (AH HB BD : ℝ)
    (isosceles : AC = BC)
    (ac_len : AC = 8)
    (bc_len : BC = 8)
    (ab_len : AB = 3)
    (cd_len : CD = 9)
    (H_midpoint : H = (A + B)/2)
    (AH_len : AH = 3/2)
    (HB_len : HB = 3/2)
    (CH : ℝ)
    (CH_eq : CH^2 = 64 - (3/2)^2)
    (BD_eq : CH^2 = 81 - (BD + AH)^2) :
    BD = 2.887 :=
  sorry

end length_BD_in_isosceles_triangle_l490_490634


namespace compute_factorial_ratio_l490_490148

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem compute_factorial_ratio : (factorial 45) / (factorial 42) = 85140 := by
  sorry

end compute_factorial_ratio_l490_490148


namespace sufficient_not_necessary_l490_490542

theorem sufficient_not_necessary (a b : ℝ) :
  (a = -1 ∧ b = 2 → a * b = -2) ∧ (a * b = -2 → ¬(a = -1 ∧ b = 2)) :=
by
  sorry

end sufficient_not_necessary_l490_490542


namespace sufficient_drivers_and_completion_time_l490_490927

noncomputable def one_way_trip_minutes : ℕ := 2 * 60 + 40
noncomputable def round_trip_minutes : ℕ := 2 * one_way_trip_minutes
noncomputable def rest_period_minutes : ℕ := 60
noncomputable def twelve_forty_pm : ℕ := 12 * 60 + 40 -- in minutes from midnight
noncomputable def one_forty_pm : ℕ := twelve_forty_pm + rest_period_minutes
noncomputable def thirteen_five_pm : ℕ := 13 * 60 + 5 -- 1:05 PM
noncomputable def sixteen_ten_pm : ℕ := 16 * 60 + 10 -- 4:10 PM
noncomputable def sixteen_pm : ℕ := 16 * 60 -- 4:00 PM
noncomputable def seventeen_thirty_pm : ℕ := 17 * 60 + 30 -- 5:30 PM
noncomputable def twenty_one_thirty_pm : ℕ := sixteen_ten_pm + round_trip_minutes -- 9:30 PM (21:30)

theorem sufficient_drivers_and_completion_time :
  4 = 4 ∧ twenty_one_thirty_pm = 21 * 60 + 30 := by
  sorry 

end sufficient_drivers_and_completion_time_l490_490927


namespace solution_set_of_g_inequality_l490_490205

noncomputable def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def derivative_condition (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 0 < x → x * f' x > -2 * f x

def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  x^2 * f x

theorem solution_set_of_g_inequality {f f' : ℝ → ℝ}
  (hf : even_function f)
  (hf_cond : derivative_condition f f') :
  {x : ℝ | g f x < g f (1 - x)} = {x : ℝ | x ∈ (set.Iio 0) ∪ set.Ioo 0 (1 / 2)} :=
sorry

end solution_set_of_g_inequality_l490_490205


namespace average_of_possible_values_of_x_l490_490620

theorem average_of_possible_values_of_x :
  (∀ x : ℝ, (sqrt (3 * x^2 + 4) = sqrt 31) → (x = 3 ∨ x = -3)) →
  (average : ℝ := (3 + (-3)) / 2) →
  (average = 0) :=
by
  intro h
  let average := (3 + (-3)) / 2
  show average = 0
  sorry

end average_of_possible_values_of_x_l490_490620


namespace pencils_distributed_per_container_l490_490510

noncomputable def total_pencils (initial_pencils : ℕ) (additional_pencils : ℕ) : ℕ :=
  initial_pencils + additional_pencils

noncomputable def pencils_per_container (total_pencils : ℕ) (num_containers : ℕ) : ℕ :=
  total_pencils / num_containers

theorem pencils_distributed_per_container :
  let initial_pencils := 150
  let additional_pencils := 30
  let num_containers := 5
  let total := total_pencils initial_pencils additional_pencils
  let pencils_per_container := pencils_per_container total num_containers
  pencils_per_container = 36 :=
by {
  -- sorry is used to skip the proof
  -- the actual proof is not required
  sorry
}

end pencils_distributed_per_container_l490_490510


namespace range_sum_eq_one_l490_490158

theorem range_sum_eq_one :
  ∃a b, (∀ x : ℝ, 0 < a ∧ a < f(x) ∧ f(x) ≤ b ∧ b = 1) ∧ (a + b = 1) :=
sorry

end range_sum_eq_one_l490_490158


namespace sunny_finishes_behind_l490_490635

-- Definitions based on the conditions:
variables (s w : ℝ) -- defining speeds of Sunny and Windy
constants (h1 : (120 : ℝ) / s = 102 / w) -- Sunny finishes 18 meters ahead of Windy in the first race

def w_reduced := 0.96 * w -- Windy's speed reduction factor
def sunny_dist := 138 -- Sunny's total distance to run in the second race

theorem sunny_finishes_behind :
  let time_sunny := sunny_dist / s in
  let distance_windy := w_reduced * time_sunny in
  distance_windy > sunny_dist :=
by sorry

end sunny_finishes_behind_l490_490635


namespace greatest_integer_m_l490_490049

theorem greatest_integer_m (N : ℕ) (hN : N = 20) : ∃ m, (∀ k, (N! / (10^k) : ℚ) ∈ ℚ → k ≤ m) ∧ m = 4 :=
by
  use 4
  intros k hk
  sorry

end greatest_integer_m_l490_490049


namespace range_of_p_l490_490535

noncomputable def a_n : ℕ → ℕ := λ n, 2 * n + 2

noncomputable def A_n (n : ℕ) : ℕ := (Finset.range n).sum (λ k, 2^k * a_n (k + 1))

theorem range_of_p (p : ℝ) (T : ℕ → ℝ) (n : ℕ) (hA : ∀ n, A_n n = n * 2^(n + 1))
    (hTn_le_T6 : ∀ n, T n ≤ T 6) : -7 / 3 ≤ p ∧ p ≤ -16 / 7 := 
sorry

end range_of_p_l490_490535


namespace sum_of_solutions_l490_490790

-- Define the quadratic equation and variable x
def quadratic_equation := ∀ x : ℝ, (x^2 - 9 * x + 20 = 0)

-- Define what we need to prove
theorem sum_of_solutions : ∃ s : ℝ, (∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → s = x1 + x2) ∧ s = 9 :=
by
  sorry -- Proof is omitted

end sum_of_solutions_l490_490790


namespace coefficient_of_x3_in_expansion_l490_490965

theorem coefficient_of_x3_in_expansion : 
  (∃ c : ℤ, c = -80 ∧ 
  ∑ k in Finset.range (6), (Nat.choose 5 k) * (-2 : ℤ)^k * (x : ℤ)^(5 - k) = (1 - 2*x)^5) :=
begin
  sorry
end

end coefficient_of_x3_in_expansion_l490_490965


namespace sum_of_solutions_l490_490788

-- Define the quadratic equation and variable x
def quadratic_equation := ∀ x : ℝ, (x^2 - 9 * x + 20 = 0)

-- Define what we need to prove
theorem sum_of_solutions : ∃ s : ℝ, (∀ x1 x2 : ℝ, quadratic_equation x1 → quadratic_equation x2 → s = x1 + x2) ∧ s = 9 :=
by
  sorry -- Proof is omitted

end sum_of_solutions_l490_490788


namespace even_and_monotonically_decreasing_l490_490877

noncomputable def f_B (x : ℝ) : ℝ := 1 / (x^2)

theorem even_and_monotonically_decreasing (x : ℝ) (h : x > 0) :
  (f_B x = f_B (-x)) ∧ (∀ {a b : ℝ}, a < b → a > 0 → b > 0 → f_B a > f_B b) :=
by
  sorry

end even_and_monotonically_decreasing_l490_490877


namespace math_problem_l490_490138

noncomputable def product_range : ℝ := 
  ∏ n in (finset.range 462).map (λ n => 100 + 2 * n), (n : ℝ) / (n + 1)

theorem math_problem : product_range < (5 : ℝ) / 16 := by
  sorry

end math_problem_l490_490138


namespace maximize_box_volume_l490_490495

noncomputable def volume (x : ℝ) := (16 - 2 * x) * (10 - 2 * x) * x

theorem maximize_box_volume :
  (∃ x : ℝ, volume x = 144 ∧ ∀ y : ℝ, 0 < y ∧ y < 5 → volume y ≤ volume 2) := 
by
  sorry

end maximize_box_volume_l490_490495


namespace inscribed_rectangle_area_l490_490117

variables (b h x : ℝ)
variables (h_isosceles_triangle : b > 0 ∧ h > 0 ∧ x > 0 ∧ x < h)

noncomputable def rectangle_area (b h x : ℝ) : ℝ :=
  (b * x / h) * (h - x)

theorem inscribed_rectangle_area :
  rectangle_area b h x = (b * x / h) * (h - x) :=
by
  unfold rectangle_area
  sorry

end inscribed_rectangle_area_l490_490117


namespace combined_share_is_1800_l490_490060

def total_money : ℕ := 4500
def ratio : (ℕ × ℕ × ℕ × ℕ) := (2, 4, 5, 4)

theorem combined_share_is_1800 : 
  let a_share : ℕ := (ratio.1 * total_money) / (ratio.1 + ratio.2 + ratio.3 + ratio.4)
  let b_share : ℕ := (ratio.2 * total_money) / (ratio.1 + ratio.2 + ratio.3 + ratio.4)
  a_share + b_share = 1800 :=
by
  sorry

end combined_share_is_1800_l490_490060


namespace proof_cond_3_proof_cond_4_l490_490563

variables {P : Type} [Plane P]
variables {L : Type} [Line L]

def distinct_lines (m n : L) : Prop := m ≠ n
def non_coincident_planes (α β : P) : Prop := α ≠ β

def parallel_plane (α β : P) : Prop := ∀ m : L, m ∈ α → m ∈ β
def parallel_line_to_plane (m : L) (α : P) : Prop := ∀ n : L, n ∈ α → n ∥ m
def perpendicular_line_to_plane (m : L) (α : P) : Prop := ∀ n : L, n ∈ α → m ⊥ n
def line_in_plane (m : L) (α : P) : Prop := ∀ n : L, m = n → n ∈ α
def skew_lines (m n : L) : Prop := ¬ (∀ α : P, line_in_plane m α ∧ line_in_plane n α) ∧ ∀ α : P, ¬ (parallel_line_to_plane m α ∧ parallel_line_to_plane n α)

theorem proof_cond_3 (m n : L) (α β : P) 
(hl1: distinct_lines m n) 
(hp1: non_coincident_planes α β) 
(h1: α ∥ β) 
(h2: line_in_plane m α) 
(h3: line_in_plane n α) 
(h4: parallel_line_to_plane m α) 
(h5: parallel_line_to_plane n α) : 
parallel_plane α β := 
sorry

theorem proof_cond_4 (m n : L) (α : P)
(hl1 : distinct_lines m n)
(h1 : perpendicular_line_to_plane m α)
(h2 : parallel_line_to_plane n α) :
m ⊥ n := 
sorry


end proof_cond_3_proof_cond_4_l490_490563


namespace sum_of_solutions_of_quadratic_eq_l490_490816

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 9 * x + 20 = 0

-- Prove that the sum of the solutions to this equation is 9
theorem sum_of_solutions_of_quadratic_eq : 
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 9) := 
begin
  -- Proof is omitted
  sorry
end

end sum_of_solutions_of_quadratic_eq_l490_490816


namespace product_inequality_l490_490143

theorem product_inequality :
  let A := (List.prod (List.map (λ n, (n / (n + 1))) (List.range' 100 923))) 
  in A < (5 / 16) :=
by
  sorry

end product_inequality_l490_490143


namespace f_periodicity_f_value_at_minus_100_l490_490694

noncomputable def f : ℝ → ℝ :=
λ x, if 0 ≤ x ∧ x < 7 then Real.log 2 (9 - x) else sorry

-- Define the functions and properties within Lean's environment
theorem f_periodicity : (∀ x : ℝ, f (x + 3) * f (x - 4) = -1) →
  (∀ x : ℝ, f (x + 14) = f x) :=
begin
  intro h,
  sorry -- Proof of periodicity property from the given condition
end

theorem f_value_at_minus_100 : 
  (∀ x : ℝ, f (x + 3) * f (x - 4) = -1) →
  (∀ x, 0 ≤ x → x < 7 → f x = Real.log 2 (9 - x)) →
  f (-100) = - (1/2) :=
begin
  intros h1 h2,
  -- First prove periodicity
  have period := f_periodicity h1,
  sorry -- Further steps to compute f(-100) using periodicity and defined conditions
end

end f_periodicity_f_value_at_minus_100_l490_490694


namespace fraction_b_not_whole_l490_490878

-- Defining the fractions as real numbers
def fraction_a := 60 / 12
def fraction_b := 60 / 8
def fraction_c := 60 / 5
def fraction_d := 60 / 4
def fraction_e := 60 / 3

-- Defining what it means to be a whole number
def is_whole_number (x : ℝ) : Prop := ∃ (n : ℤ), x = n

-- Theorem stating that fraction_b is not a whole number
theorem fraction_b_not_whole : ¬ is_whole_number fraction_b := 
by 
-- proof to be filled in
sorry

end fraction_b_not_whole_l490_490878


namespace g_half_equals_four_l490_490589

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a^(2 - x) - 3 / 4

def A : ℝ × ℝ :=
  (2, 1 / 4)

noncomputable def g (α : ℝ) (x : ℝ) : ℝ :=
  x^α

theorem g_half_equals_four (a : ℝ) (α : ℝ)
  (ha : a > 0) (ha' : a ≠ 1) (hA1 : f a 2 = 1/4) (hA2 : g α 2 = 1/4) :
  g α (1 / 2) = 4 :=
by
  sorry

end g_half_equals_four_l490_490589


namespace shift_and_symmetric_proof_l490_490910

-- Definitions and conditions based on the problem
def original_function (x : ℝ) : ℝ :=
  real.exp x

def symmetric_function (x : ℝ) : ℝ :=
  real.exp (-x)

def shifted_function (x : ℝ) : ℝ :=
  real.exp (-(x - 1))

-- Proof statement in Lean
theorem shift_and_symmetric_proof : (∀ x : ℝ, (shifted_function x = real.exp (-x - 1))) :=
by
  intro x
  rw [shifted_function, sub_add_eq_sub_sub]
  sorry

end shift_and_symmetric_proof_l490_490910


namespace neznayka_made_an_error_l490_490711

theorem neznayka_made_an_error :
  ∀ (a : fin 11 → ℕ), 
    (finset.card {i : fin 11 | abs (a i - a ((i + 1) % 11)) = 1} = 4 ∧ 
     finset.card {i : fin 11 | abs (a i - a ((i + 1) % 11)) = 2} = 4 ∧ 
     finset.card {i : fin 11 | abs (a i - a ((i + 1) % 11)) = 3} = 3) → 
    false :=
by
  sorry

end neznayka_made_an_error_l490_490711


namespace professors_women_tenured_or_both_l490_490946

noncomputable def total_professors : ℕ := 100
def percent_women : ℕ := 70
def percent_tenured : ℕ := 70
def percent_men_tenured : ℕ := 50

theorem professors_women_tenured_or_both :
  percent_women = 70 ∧
  percent_tenured = 70 ∧
  percent_men_tenured = 50 →
  85% of total_professors are women, tenured, or both
:= sorry

end professors_women_tenured_or_both_l490_490946


namespace number_of_rows_l490_490462

theorem number_of_rows (r : ℕ) (h1 : ∀ bus : ℕ, bus * (4 * r) = 240) : r = 10 :=
sorry

end number_of_rows_l490_490462


namespace sum_of_roots_l490_490996

theorem sum_of_roots : 
  ∀ x1 x2 : ℝ, 
  (x1^2 + 2023*x1 = 2024 ∧ x2^2 + 2023*x2 = 2024) → 
  x1 + x2 = -2023 := 
by 
  sorry

end sum_of_roots_l490_490996


namespace hypotenuse_length_l490_490932

theorem hypotenuse_length (a b : ℕ) (h : ℕ) (h0 : a = 30) (h1 : b = 40) : a * a + b * b = h * h → h = 50 :=
by
  intros
  rw [h0, h1] at *
  calc
    30 * 30 + 40 * 40 = 900 + 1600     : by norm_num
                       ... = 2500       : by norm_num
                       ... = 50 * 50    : by norm_num
  exact eq_of_sq_eq_sq h0

end hypotenuse_length_l490_490932


namespace sufficient_drivers_and_completion_time_l490_490925

noncomputable def one_way_trip_minutes : ℕ := 2 * 60 + 40
noncomputable def round_trip_minutes : ℕ := 2 * one_way_trip_minutes
noncomputable def rest_period_minutes : ℕ := 60
noncomputable def twelve_forty_pm : ℕ := 12 * 60 + 40 -- in minutes from midnight
noncomputable def one_forty_pm : ℕ := twelve_forty_pm + rest_period_minutes
noncomputable def thirteen_five_pm : ℕ := 13 * 60 + 5 -- 1:05 PM
noncomputable def sixteen_ten_pm : ℕ := 16 * 60 + 10 -- 4:10 PM
noncomputable def sixteen_pm : ℕ := 16 * 60 -- 4:00 PM
noncomputable def seventeen_thirty_pm : ℕ := 17 * 60 + 30 -- 5:30 PM
noncomputable def twenty_one_thirty_pm : ℕ := sixteen_ten_pm + round_trip_minutes -- 9:30 PM (21:30)

theorem sufficient_drivers_and_completion_time :
  4 = 4 ∧ twenty_one_thirty_pm = 21 * 60 + 30 := by
  sorry 

end sufficient_drivers_and_completion_time_l490_490925


namespace odd_numbers_one_l490_490220

theorem odd_numbers_one (a b c d k m : ℤ) :
  odd a → odd b → odd c → odd d → 
  0 < a → a < b → b < c → c < d → 
  a * d = b * c → 
  a + d = 2^k → 
  b + c = 2^m → 
  a = 1 :=
by 
  sorry

end odd_numbers_one_l490_490220


namespace polar_bear_trout_l490_490164

/-
Question: How many buckets of trout does the polar bear eat daily?
Conditions:
  1. The polar bear eats some amount of trout and 0.4 bucket of salmon daily.
  2. The polar bear eats a total of 0.6 buckets of fish daily.
Answer: 0.2 buckets of trout daily.
-/

theorem polar_bear_trout (trout salmon total : ℝ) 
  (h1 : salmon = 0.4)
  (h2 : total = 0.6)
  (h3 : trout + salmon = total) :
  trout = 0.2 :=
by
  -- The proof will be provided here
  sorry

end polar_bear_trout_l490_490164


namespace find_k_l490_490489

noncomputable def f (x : ℝ) : ℝ := 6 * x^2 + 4 * x - (1 / x) + 2

noncomputable def g (x : ℝ) (k : ℝ) : ℝ := x^2 + 3 * x - k

theorem find_k (k : ℝ) : 
  f 3 - g 3 k = 5 → 
  k = - 134 / 3 :=
by
  sorry

end find_k_l490_490489


namespace modulus_of_pure_imaginary_z_l490_490226

-- Given
variables (a : ℝ)

-- Definition of z
def complex_z (a : ℝ) : ℂ := (2 - a * complex.I) / complex.I

-- Conditions
axiom z_is_pure_imaginary : Im (complex_z a) = (complex_z a) ∧ Re (complex_z a) = 0

-- Proof problem
theorem modulus_of_pure_imaginary_z (h : ∀ a : ℝ, complex_z a = -2 * complex.I) : complex.abs (complex_z 0) = 2 := 
by 
-- proof is omitted according to instruction
sorry

end modulus_of_pure_imaginary_z_l490_490226


namespace slips_with_3_count_l490_490353

-- Conditions
def total_slips := 15
def expected_value := 5

-- Let y be the number of slips with a number 3
def slips_with_3 (y : ℕ) := y
def slips_with_8 (y : ℕ) := total_slips - y

-- Expected value calculation for verification
def expected_value_calculation (y : ℕ) : ℝ :=
  (slips_with_3 y / total_slips) * 3 + (slips_with_8 y / total_slips) * 8

-- The hypothesis
def hypothesis (y : ℕ) : Prop :=
  expected_value_calculation y = expected_value

-- Statement to prove
theorem slips_with_3_count : ∃ y, slips_with_3 y = 9 ∧ hypothesis y :=
by
  sorry

end slips_with_3_count_l490_490353


namespace least_number_to_subtract_l490_490066

theorem least_number_to_subtract {x : ℕ} (h : x = 13604) : 
    ∃ n : ℕ, n = 32 ∧ (13604 - n) % 87 = 0 :=
by
  sorry

end least_number_to_subtract_l490_490066


namespace product_inequality_l490_490141

theorem product_inequality :
  let A := (List.prod (List.map (λ n, (n / (n + 1))) (List.range' 100 923))) 
  in A < (5 / 16) :=
by
  sorry

end product_inequality_l490_490141


namespace range_of_omega_l490_490628

noncomputable def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x1 x2, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 < f x2

theorem range_of_omega
  (ω : ℝ) (hω_pos : ω > 0)
  (h_incr : is_increasing (λ x, Real.tan (ω * x)) (-π) π) :
  0 < ω ∧ ω ≤ 1 / 2 := 
sorry

end range_of_omega_l490_490628


namespace hyperbola_eccentricity_l490_490595

theorem hyperbola_eccentricity (a b : Real) (h_a_gt_zero : a > 0) (h_b_gt_zero : b > 0) 
  (h_asymptote : ∀ x : Real, y = 2 * x):
  ∃ e : Real, e = sqrt 5 := 
  sorry

end hyperbola_eccentricity_l490_490595


namespace smallest_possible_w_two_power_factor_l490_490448

def prime_factors (n : ℕ) : list ℕ := sorry

theorem smallest_possible_w (w : ℕ) (h_w : w = 3^2 * 13^3) :
  3^3 ∣ 1452 * w ∧ 13^3 ∣ 1452 * w :=
by {
  sorry
}

theorem two_power_factor (w : ℕ) (h : 1452 * w = 3^3 * 13^3 * 2^2 * w) (k : ℕ) :
  ∃ k, (1452 * 468) = 2^k * 3^3 * 13^3 :=
by {
  use 0,
  sorry
}

end smallest_possible_w_two_power_factor_l490_490448


namespace parabola_point_distance_to_focus_l490_490028

theorem parabola_point_distance_to_focus :
  ∀ (x y : ℝ), (y^2 = 12 * x) → (∃ (xf : ℝ), xf = 3 ∧ 0 ≤ y) → (∃ (d : ℝ), d = 7) → x = 4 :=
by
  intros x y parabola_focus distance_to_focus distance
  sorry

end parabola_point_distance_to_focus_l490_490028


namespace base7_number_divisible_by_19_l490_490501

theorem base7_number_divisible_by_19 :
  ∃ (x : ℕ), x ∈ finset.range 7 ∧ (934 + 7 * x) % 19 = 0 :=
by sorry

end base7_number_divisible_by_19_l490_490501


namespace darryl_cantaloupes_start_l490_490497

theorem darryl_cantaloupes_start (C : ℕ) : 
  (∃ C : ℕ, 2 * (C - 10) + 3 * 15 = 85) → C = 30 :=
by 
  intro hC,
  cases hC with C hEq,
  have h1 : 2 * C - 20 + 45 = 85 := hEq,
  have h2 : 2 * C + 25 = 85 := by linarith,
  have h3 : 2 * C = 60 := by linarith,
  have h4 : C = 30 := by linarith,
  exact h4

end darryl_cantaloupes_start_l490_490497


namespace angle_ADB_eq_45_l490_490645

/--
In the convex pentagon ABCDE, where ∠A = ∠B = ∠D = 90°, and a circle can be inscribed in the pentagon.
Prove that the angle ∠ADB is equal to 45°.
-/
theorem angle_ADB_eq_45 (A B C D E O K L M N T : Point)
  (h1 : ConvexPentagon A B C D E)
  (h2 : ∠ A = 90°)
  (h3 : ∠ B = 90°)
  (h4 : ∠ D = 90°)
  (h_inscribe : InscribedCircle A B C D E O K L M N T) :
  ∠ ADB = 45° :=
sorry

end angle_ADB_eq_45_l490_490645


namespace find_sum_l490_490882

-- Define the prime conditions
variables (P : ℝ) (SI15 SI12 : ℝ)

-- Assume conditions for the problem
axiom h1 : SI15 = P * 15 / 100 * 2
axiom h2 : SI12 = P * 12 / 100 * 2
axiom h3 : SI15 - SI12 = 840

-- Prove that P = 14000
theorem find_sum : P = 14000 :=
sorry

end find_sum_l490_490882


namespace area_enclosed_by_sin_curve_l490_490518

noncomputable def area_of_sin : ℝ := 
  (∫ x in - (Real.pi / 2) .. 0, Real.sin x) +
  (∫ x in 0 .. Real.pi, Real.sin x) + 
  (∫ x in Real.pi .. (5 * Real.pi / 4), Real.sin x)

theorem area_enclosed_by_sin_curve : area_of_sin = 4 - Real.sqrt 2 / 2 := 
by sorry

end area_enclosed_by_sin_curve_l490_490518


namespace formula_f_max_f_value_range_f_l490_490507

def ellipse (x y : ℝ) : Prop := x^2 + y^2 / 4 = 1 ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 2

noncomputable def unit_tangent_vector (x : ℝ) : ℝ × ℝ :=
let y := 2 * real.sqrt (1 - x^2) in
let magnitude := real.sqrt ((-real.sqrt (1 - x^2))^2 + (2 * x)^2) in
((-real.sqrt (1 - x^2) / magnitude), (2 * x / magnitude))

noncomputable def f (x : ℝ) : ℝ :=
let y := 2 * real.sqrt (1 - x^2) in
(unit_tangent_vector x).fst * x + (unit_tangent_vector x).snd * y

theorem formula_f (x : ℝ) (h : x ∈ Icc 0 1) : f x = 3 * x * real.sqrt (1 - x^2) / real.sqrt (1 + 3 * x^2) :=
sorry

theorem max_f_value : ∃ x ∈ Icc (0 : ℝ) 1, f x = 1 :=
sorry

theorem range_f : ∀ x ∈ Icc (0 : ℝ) 1, 0 ≤ f x ∧ f x ≤ 1 :=
sorry

end formula_f_max_f_value_range_f_l490_490507


namespace flavoring_corn_syrup_ratio_comparison_l490_490652

-- Definitions and conditions derived from the problem
def standard_flavoring_to_water_ratio : ℝ := 1 / 30
def sport_flavoring_to_water_ratio : ℝ := standard_flavoring_to_water_ratio / 2
def sport_water_amount : ℝ := 75
def sport_flavoring_amount : ℝ := sport_water_amount / 60
def sport_corn_syrup_amount : ℝ := 5
def sport_flavoring_to_corn_syrup_ratio : ℝ := sport_flavoring_amount / sport_corn_syrup_amount
def standard_flavoring_to_corn_syrup_ratio : ℝ := 1 / 12

-- The statement to be proved
theorem flavoring_corn_syrup_ratio_comparison :
  sport_flavoring_to_corn_syrup_ratio / standard_flavoring_to_corn_syrup_ratio = 3 :=
by
  have h_sport_flavoring_to_corn_syrup : sport_flavoring_to_corn_syrup_ratio = 1 / 4,
  sorry

  have h_standard_flavoring_to_corn_syrup : standard_flavoring_to_corn_syrup_ratio = 1 / 12,
  sorry

  calc
    sport_flavoring_to_corn_syrup_ratio / standard_flavoring_to_corn_syrup_ratio
        = (1 / 4) / (1 / 12) : by rw [h_sport_flavoring_to_corn_syrup, h_standard_flavoring_to_corn_syrup]
    ... = (1 / 4) * 12 / 1 : by sorry
    ... = 12 / 4 : by sorry
    ... = 3 : by sorry

end flavoring_corn_syrup_ratio_comparison_l490_490652


namespace prime_product_sum_91_l490_490026

theorem prime_product_sum_91 (p1 p2 : ℕ) (h1 : Nat.Prime p1) (h2 : Nat.Prime p2) (h3 : p1 + p2 = 91) : p1 * p2 = 178 :=
sorry

end prime_product_sum_91_l490_490026


namespace product_segments_eq_l490_490071

open_locale classical

variables {α : Type*} [real_space α]
variables (circle : set α) (O A B T C D E S : α)
variable (OB_perpendicular : line_through T ⊥ line_through O B)
variable (TS_on_circle : T ∈ segment O B)
variable (line_T_intersect_AB : line_through T ⊥ line_through O B ∧ line_through T ∩ AB = {C})
variable (line_T_intersect_DE : line_through T ⊥ line_through O B ∧ line_through T ∩ circle = {D, E})
variable (S_projection : orthogonal_projection T AB S)

theorem product_segments_eq :
  (segment A S).length * (segment B C).length = (segment T E).length * (segment T D).length :=
sorry

end product_segments_eq_l490_490071


namespace eighth_graders_ninth_grader_points_l490_490885

noncomputable def eighth_grader_points (y : ℚ) (x : ℕ) : Prop :=
  x * y + 8 = ((x + 2) * (x + 1)) / 2

theorem eighth_graders (x : ℕ) (y : ℚ) (hx : eighth_grader_points y x) :
  x = 7 ∨ x = 14 :=
sorry

noncomputable def tenth_grader_points (z y : ℚ) (x : ℕ) : Prop :=
  10 * z = 4.5 * y ∧ x * z = y

theorem ninth_grader_points (y : ℚ) (x : ℕ) (z : ℚ)
  (hx : tenth_grader_points z y x) :
  y = 10 :=
sorry

end eighth_graders_ninth_grader_points_l490_490885


namespace volume_prism_l490_490181

-- Define the parameters and conditions
variables (a : ℝ) (h1 : a > 0)

-- Define the height of the prism
def height_of_prism := (a * real.sin (real.pi / 3))

-- Define the area of the base (equilateral triangle)
def area_of_base := (real.sqrt 3 / 4) * a ^ 2

-- Define the volume of the oblique triangular prism
noncomputable def volume_of_prism := (area_of_base a) * (height_of_prism a)

-- Prove that the volume is equal to (3 * a ^ 3) / 8
theorem volume_prism (a > 0) : volume_of_prism a = (3 * a ^ 3) / 8 :=
sorry

end volume_prism_l490_490181


namespace window_design_ratio_l490_490114

theorem window_design_ratio (AB AD r : ℝ)
  (h1 : AB = 40)
  (h2 : AD / AB = 4 / 3)
  (h3 : r = AB / 2) :
  ((AD - AB) * AB) / (π * r^2 / 2) = 8 / (3 * π) :=
by
  sorry

end window_design_ratio_l490_490114


namespace smallest_integer_for_circle_angles_l490_490992

theorem smallest_integer_for_circle_angles :
  ∃ n : ℕ, (∀ (A : Fin n → ℝ × ℝ), (∃ (S : Finset (Fin n × Fin n)), S.card ≥ 2007 ∧ ∀ (i j : Fin n), i < j → (i, j) ∈ S → angle A[i] O A[j] ≤ 120)) ∧ n = 91 :=
sorry

end smallest_integer_for_circle_angles_l490_490992


namespace samia_walked_distance_l490_490730

-- Introduce all the givens
variables (x : ℝ) -- half the total distance

def total_distance := 2 * x
def cycling_speed := 20
def walking_speed := 6
def total_time_minutes := 48

-- Convert total time to hours
def total_time_hours := (total_time_minutes : ℝ) / 60

-- Calculate the time for each part of the journey
def time_cycling := x / cycling_speed
def time_walking := x / walking_speed

-- Total time equation
def total_time_calculated := time_cycling + time_walking

-- The main theorem that we need to prove
theorem samia_walked_distance :
  total_time_calculated = total_time_hours →
  x ≈ 2.1 :=
  sorry

end samia_walked_distance_l490_490730


namespace vasya_petya_prime_l490_490774

theorem vasya_petya_prime (x : ℚ) (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (10 * x = p) ∧ (15 * x = q) → ∃ x : ℚ, (10 * x).Nat.Abs.Prime ∧ (15 * x).Nat.Abs.Prime := 
by
  intro h
  sorry

end vasya_petya_prime_l490_490774


namespace power_function_decreasing_through_point_l490_490594

noncomputable def is_decreasing (f : Real → Real) : Prop :=
∀ x y : Real, x < y → f x > f y

noncomputable def power_function (α : Real) : Real → Real := λ x, x^α

theorem power_function_decreasing_through_point :
  (∃ α : Real, power_function α 2 = Real.sqrt 2 / 2) →
  ∃ α : Real, α = -1 / 2 ∧ is_decreasing (power_function α) :=
by
  intros h
  sorry

end power_function_decreasing_through_point_l490_490594


namespace min_inv_sum_four_l490_490259

theorem min_inv_sum_four (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  4 ≤ (1 / a + 1 / b) := 
sorry

end min_inv_sum_four_l490_490259


namespace cos_B_value_l490_490290

noncomputable theory
open Real

variables {a b c : ℝ} (q : ℝ) (A B C : ℝ)
variables (triangle_ABC : Triangle a b c)

-- Conditions
def condition1 := triangle_ABC
def condition2 := b = sqrt (a * c)
def condition3 := sin A = (sin (B - A) + sin C) / 2

-- Question: Prove cos B = (sqrt(5) - 1) / 2
theorem cos_B_value (h1 : condition1) (h2 : condition2) (h3 : condition3) : cos B = (sqrt 5 - 1) / 2 := 
sorry

end cos_B_value_l490_490290


namespace ratio_rise_liquid_levels_l490_490042

noncomputable def volume_cone (r h : ℝ) : ℝ :=
  (1 / 3) * real.pi * r^2 * h

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * real.pi * r^3

theorem ratio_rise_liquid_levels 
  (h : ℝ)
  (ha : h > 0)
  (ra : ℝ) (rb : ℝ)
  (hcA : ra = 4)
  (hcB : rb = 8)
  (rm : ℝ)
  (hm : rm = 2) :
  let Va := volume_cone ra h,
      Vb := volume_cone rb h,
      Vm := volume_sphere rm,
      new_Va := Va + Vm,
      new_Vb := Vb + Vm,
      rise_A := new_Va / (real.pi * ra^2) - Va / (real.pi * ra^2),
      rise_B := new_Vb / (real.pi * rb^2) - Vb / (real.pi * rb^2)
  in rise_A / rise_B = 4 :=
begin
  sorry
end

end ratio_rise_liquid_levels_l490_490042


namespace max_sum_value_l490_490351

-- Given conditions:
-- a sequence of non-decreasing positive integers
variables {a : ℕ → ℕ} (h_a : ∀ n, a n ≤ a (n + 1))

-- b_m = min {n | a_n ≥ m}
def b (m : ℕ) : ℕ := Inf {n | a n ≥ m}

-- Given a_19 = 85
axiom a_19_eq_85 : a 19 = 85

-- Prove the maximum value of the sum is 1700
theorem max_sum_value : 
  (∑ i in (Finset.range 19), a i) + (∑ j in (Finset.range 85), b j) = 1700 :=
sorry

end max_sum_value_l490_490351


namespace relationship_segments_l490_490284

-- Definitions and conditions
variables {ABC : Type} [triangle ABC]
variables {O' : point ABC}
variables {O : point ABC}
variables {D E : point ABC}
variables (incenter_O : is_incenter O ABC)
variables (circumscribed_O' : is_circumscribed_center O' ABC)
variables (diameter_DE : is_diameter D E (circumscribed_circle O' ABC))

-- Proof goal
theorem relationship_segments :
  segments_equal CD OE BD :=
sorry

end relationship_segments_l490_490284


namespace num_elements_in_A_l490_490547

-- Define the sum a
def a : ℕ := ∑ i in finset.image (λ i : ℕ, nat.floor (real.sqrt i)) (finset.range 24), id i

-- Define the set A
def A : finset ℕ := finset.filter (λ x, x ∣ a) (finset.range (a + 1))

-- The theorem statement
theorem num_elements_in_A : A.card = 8 :=
by
  sorry

end num_elements_in_A_l490_490547


namespace polynomial_real_root_symmetry_l490_490967

noncomputable def cubic_root_of_unity (ω : Complex) : Prop :=
  ω = Complex.exp (2 * Real.pi * Complex.I / 3)

theorem polynomial_real_root_symmetry :
  ∃ (a b c d e : ℝ), ∃ (P : Polynomial ℂ) (roots : Finset ℂ),
    (P = Polynomial.C 2023 + Polynomial.X + Polynomial.C a * Polynomial.X^5 + Polynomial.C b * Polynomial.X^4 + 
        Polynomial.C c * Polynomial.X^3 + Polynomial.C d * Polynomial.X^2 + Polynomial.C e * Polynomial.X) ∧
    (∀ s ∈ roots, cubic_root_of_unity (Complex.of_real (-1 + Real.sqrt (3)) / 2) * s ∈ roots) ∧
    (roots.card = 6 ∨ roots.card = 6 * 2) ∧
    (∃ s, P = Polynomial.prod (λ r, Polynomial.X - Polynomial.C r)) ∧
    (∃ s : ℂ, P.eval s = 0) →
    (roots.card = 2) :=
by sorry

end polynomial_real_root_symmetry_l490_490967


namespace bricks_required_l490_490084

-- Definitions
def courtyard_length : ℕ := 20  -- in meters
def courtyard_breadth : ℕ := 16  -- in meters
def brick_length : ℕ := 20  -- in centimeters
def brick_breadth : ℕ := 10  -- in centimeters

-- Statement to prove
theorem bricks_required :
  ((courtyard_length * 100) * (courtyard_breadth * 100)) / (brick_length * brick_breadth) = 16000 :=
sorry

end bricks_required_l490_490084


namespace num_correct_statements_is_three_l490_490471

theorem num_correct_statements_is_three 
    (h1 : ∀ a : ℝ, 0 ≤ a ∧ a ≤ 1 → 3 * a - 1 > 0 ↔ a > 1 / 3) 
    (h2 : ∀ x y : ℝ, x + y ≠ 0 → x ≠ 1 ∨ y ≠ -1) 
    (h3 : ∀ (α β : Type) [plane α] [plane β], 
           (¬ perpendicular α β) → ∃ l : line α, perpendicular l β → false) 
    (h4p : ∀ {a b c : Type} [vector_space a] [vector_space b] [vector_space c], 
            inner_product a b = 0 → inner_product b c = 0 → inner_product a c = 0 → false) 
    (h4q : ∀ {a b c : Type} [vector_space a] [vector_space b] [vector_space c], 
            parallel a b → parallel b c → parallel a c) :
    (2, 3, 4, : {1, 2, 3, 4}) = 3 := sorry

end num_correct_statements_is_three_l490_490471


namespace max_watches_two_hours_l490_490319

noncomputable def show_watched_each_day : ℕ := 30 -- Time in minutes
def days_watched : ℕ := 4 -- Monday to Thursday

theorem max_watches_two_hours :
  (days_watched * show_watched_each_day) / 60 = 2 := by
  sorry

end max_watches_two_hours_l490_490319


namespace find_a_parallel_lines_l490_490266

theorem find_a_parallel_lines (a : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, x * a + 2 * y + 2 = 0 ↔ 3 * x - y - 2 = k * (x * a + 2 * y + 2)) ↔ a = -6 := by
  sorry

end find_a_parallel_lines_l490_490266


namespace determine_m_l490_490163

theorem determine_m (m : ℝ) (f g : ℝ → ℝ)
  (h1 : f = λ x, x^3 - 3*x^2 + m)
  (h2 : g = λ x, x^3 - 3*x^2 + 6*m)
  (h3 : 2 * f 3 = 3 * g 3) :
  m = 0 := by
sorry

end determine_m_l490_490163


namespace point_on_transformed_graph_l490_490223

variable f : ℝ → ℝ

theorem point_on_transformed_graph (h : f 9 = 7) : 3 * (16 / 9) = (f (3 * 3)) / 3 + 3 ∧ 3 + (16 / 9) = 43 / 9 :=
by
  split
  · -- Proof that (3, 16/9) is on the graph of 3y = (f(3x))/3 + 3
    calc
      3 * (16 / 9) = (f 9 / 3) + 3     : by rw [h]; norm_num
                 ... = 7 / 3 + 3       : by rw [h]
                 ... = 7 / 3 + 9 / 3   : by norm_num
                 ... = (7 + 9) / 3     : by ring
                 ... = 16 / 3          : by norm_num
                 
  · -- Proof that the sum of the coordinates is 43/9
    calc
      3 + (16 / 9) = 27 / 9 + 16 / 9   : by norm_num
                 ... = (27 + 16) / 9   : by ring
                 ... = 43 / 9          : by norm_num

end point_on_transformed_graph_l490_490223


namespace unique_integer_n_l490_490189

def S_n (n : ℕ) (a : Fin n → ℝ) : ℝ :=
  ∑ k in Finset.range n, Real.sqrt ((2 * (k + 1) - 1)^2 + (a ⟨k, Fin.is_lt k⟩) ^ 2)

theorem unique_integer_n (n : ℕ) (a : Fin n → ℝ)
  (hn_pos : 0 < n)
  (ha_sum : (∑ k in Finset.range n, a ⟨k, Fin.is_lt k⟩) = 17)
  (h_integer : ∃ k : ℤ, S_n n a = k) :
  n = 12 :=
  sorry

end unique_integer_n_l490_490189


namespace decreasing_function_range_l490_490014

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (3 * a - 1) * x + 4 * a else a ^ x

theorem decreasing_function_range :
  { a : ℝ | ∀ x y : ℝ, x < y → f a x ≥ f a y } = Icc (1 / 6) (1 / 3) :=
by
  -- Proof is omitted as per guidelines
  sorry

end decreasing_function_range_l490_490014


namespace simplify_expression_and_evaluate_l490_490733

theorem simplify_expression_and_evaluate :
  ∀ x : ℤ, (-1 ≤ x ∧ x ≤ 3) → (∃ y : ℚ, y = (let e := (x^2 - 4) / (x^2 - 4*x + 4) + (x / (x^2 - x)) / ((x - 2) / (x - 1)) in
                                if h : x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2 then e else 0) ∧ y = -2 / 3 → x = -1) :=
by
  sorry

end simplify_expression_and_evaluate_l490_490733


namespace vasya_petya_no_mistake_l490_490776

theorem vasya_petya_no_mistake :
  ∃ (x p q : ℝ), prime (nat_abs (10 * x)) ∧ prime (nat_abs (15 * x)) ∧ 10 * x = p ∧ 15 * x = q ∧ 3 * p = 2 * q :=
by
  sorry

end vasya_petya_no_mistake_l490_490776


namespace find_k_l490_490240

theorem find_k (k : ℝ) (x₁ x₂ : ℝ) (h_distinct_roots : (2*k + 3)^2 - 4*k^2 > 0)
  (h_roots : ∀ (x : ℝ), x^2 + (2*k + 3)*x + k^2 = 0 ↔ x = x₁ ∨ x = x₂)
  (h_reciprocal_sum : 1/x₁ + 1/x₂ = -1) : k = 3 :=
by
  sorry

end find_k_l490_490240


namespace dot_product_orthogonal_necessary_not_sufficient_l490_490244

variables {V : Type*} [inner_product_space ℝ V] -- We consider a real inner product space for the vectors

def orthogonal (a b : V) : Prop := ⟪a, b⟫ = 0

theorem dot_product_orthogonal_necessary_not_sufficient (a b : V) :
  (⟪a, b⟫ = 0 → orthogonal a b) ∧ (orthogonal a b ↔ ⟪a, b⟫ = 0) :=
by
  sorry

end dot_product_orthogonal_necessary_not_sufficient_l490_490244


namespace base_eight_to_base_ten_l490_490421

theorem base_eight_to_base_ten : (5 * 8^1 + 2 * 8^0) = 42 := by
  sorry

end base_eight_to_base_ten_l490_490421


namespace rhombus_perimeter_l490_490625

theorem rhombus_perimeter (d1 d2 : ℝ) (θ : ℝ) :
  let s := sqrt (d1^2 + d2^2) / 2
  let P := 4 * s
  P = 2 * sqrt (d1^2 + d2^2) := by
  sorry

end rhombus_perimeter_l490_490625


namespace sum_of_solutions_eq_9_l490_490821

theorem sum_of_solutions_eq_9 :
  let roots := {x : ℝ | x^2 = 9 * x - 20}
  in ∑ x in roots, x = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490821


namespace bathroom_module_cost_l490_490157

theorem bathroom_module_cost : ∃ B : ℝ, let K := 20000 in
  let num_bathrooms := 2 in
  let bathroom_sqft := 150 in
  let other_sqft_cost := 100 in
  let total_sqft := 2000 in
  let total_cost := 174000 in
  let kitchen_sqft := 400 in
  let total_bathroom_cost := num_bathrooms * B in
  let remaining_sqft := total_sqft - kitchen_sqft - num_bathrooms * bathroom_sqft in
  let other_modules_cost := remaining_sqft * other_sqft_cost in
    (total_cost = K + total_bathroom_cost + other_modules_cost) → B = 12000 :=
begin
  sorry
end

end bathroom_module_cost_l490_490157


namespace unique_outfits_count_l490_490431

theorem unique_outfits_count (s : Fin 5) (p : Fin 6) (restricted_pairings : (Fin 1 × Fin 2) → Prop) 
  (r : restricted_pairings (0, 0) ∧ restricted_pairings (0, 1)) : ∃ n, n = 28 ∧ 
  ∃ (outfits : Fin 5 → Fin 6 → Prop), 
    (∀ s p, outfits s p) ∧ 
    (∀ p, ¬outfits 0 p ↔ p = 0 ∨ p = 1) := by
  sorry

end unique_outfits_count_l490_490431


namespace flavoring_ratio_comparison_l490_490650

theorem flavoring_ratio_comparison (f_st cs_st w_st : ℕ) (f_sp cs_sp w_sp : ℕ) :
  f_st = 1 → cs_st = 12 → w_st = 30 →
  w_sp = 75 → cs_sp = 5 →
  f_sp / w_sp = f_st / (2 * w_st) →
  (f_st / cs_st) * 3 = f_sp / cs_sp :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end flavoring_ratio_comparison_l490_490650


namespace possible_omega_values_l490_490233

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Math.sin (ω * x + Real.pi / 6) - Math.cos (ω * x)

def isSymmetrical (ω : ℝ) : Prop :=
  ∀ x, f ω x = f ω (x + 4 * Real.pi)

def isMonotonic (ω : ℝ) : Prop :=
  ∀ x, -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4 → (ω * Math.cos (ω * x + Real.pi / 6) + ω * Math.sin (ω * x) ≥ 0)

theorem possible_omega_values :
  {ω : ℝ | isSymmetrical ω ∧ isMonotonic ω} = {1 / 3, 5 / 6, 4 / 3} :=
sorry

end possible_omega_values_l490_490233


namespace treasure_in_box_2_l490_490658

open Classical

structure Chest (number : Nat) :=
  (material : String)
  (statement : String)

def chest1 := Chest.mk 1 "cedar" "The treasure is in me or in the 4th chest."
def chest2 := Chest.mk 2 "sandalwood" "The treasure is in the chest to the left of me."
def chest3 := Chest.mk 3 "sandalwood" "The treasure is in me or in the chest at the far right."
def chest4 := Chest.mk 4 "cedar" "There is no treasure in the chests to the left of me."
def chest5 := Chest.mk 5 "cedar" "All the inscriptions on other chests are false."

def chests := [chest1, chest2, chest3, chest4, chest5]

def is_true (chest : Chest) (actual : Nat) : Bool :=
  match chest.number with
  | 1 => actual = 1 ∨ actual = 4
  | 2 => actual = 1
  | 3 => actual = 3 ∨ actual = 5
  | 4 => ∀ i : Nat, (i < 4 → i ≠ actual)
  | 5 => ∀ i : Nat, (i ≠ 5 → ¬ is_true (chests.getD (i - 1) chest5) actual)

theorem treasure_in_box_2 :
  ∃ t, 
    (∀ chest, chest ∈ chests → 
      (if chest.material = "cedar" ∨ chest.material = "sandalwood" then
        is_true chest t = false 
      else true))  -- Number of false statements on cedar and sandalwood chests are equal
  ∧ 
    t = 2 :=
by
  sorry

end treasure_in_box_2_l490_490658


namespace ratio_to_percent_l490_490384

theorem ratio_to_percent (a b : ℕ) (h : a = 10 ∧ b = 20) : ((a : ℚ) / b) * 100 = 50 := 
by
  have h1 : a = 10 := h.left
  have h2 : b = 20 := h.right
  rw [h1, h2]
  norm_num
  sorry

end ratio_to_percent_l490_490384


namespace sum_of_roots_quadratic_eq_l490_490847

theorem sum_of_roots_quadratic_eq :
  (∑ x in Finset.filter (λ x, x^2 = 9 * x - 20) (Finset.range 100), x) = 9 :=
begin
  sorry
end

end sum_of_roots_quadratic_eq_l490_490847


namespace parabola_line_intersection_l490_490593

theorem parabola_line_intersection
  (a b c k m: ℝ)
  (A: ℝ × ℝ) (B: ℝ × ℝ)
  (hA: A = (1, m)) (hB: B = (4, 8))
  (h1: a ≠ 0)
  (h2: ∀ x: ℝ, x = A.1 ∨ x = B.1 → a * x^2 + b * x + c = k * x + 4)
  (h3: c = 0):
  
  (∀ x: ℝ, a * x^2 + b * x + c = -x^2 + 6 * x) ∧ 
  (∀ x: ℝ, k * x + 4 = x + 4) ∧ 
  (∃ D: ℝ × ℝ, D.2 > 0 ∧ (D = (3 + real.sqrt 5, 4) ∨ D = (3 - real.sqrt 5, 4))):=
sorry

end parabola_line_intersection_l490_490593


namespace extreme_values_and_min_value_l490_490586

noncomputable def f (x : ℝ) : ℝ := exp x - x + (1/2) * x^2

theorem extreme_values_and_min_value (a b : ℝ) :
  (∀ x : ℝ, (1/2) * x^2 - f x ≤ a * x + b) →
  (1 - a) * b ≥ - (1 - a)^2 + (1 - a)^2 * log (1 - a) →
  (∀ t > 0, let F (t : ℝ) : ℝ := - t^2 + t^2 * log t in F (real.sqrt real.exp 1) = - real.exp 1 / 2) →
  (1 - a) * b = - real.exp 1 / 2 :=
sorry

end extreme_values_and_min_value_l490_490586


namespace q_sum_at_1_l490_490681

noncomputable def polynomial : List (Polynomial ℤ) := 
  [x - 1, x^2 + x + 1, x^2 + 1]

theorem q_sum_at_1 : polynomial.sum (λ q, q.eval 1) = 5 := by
  sorry

end q_sum_at_1_l490_490681


namespace minyoung_position_from_front_l490_490893

-- Define the total number of students
def total_students : ℕ := 27

-- Define Minyoung's position from the back
def minyoung_position_from_back : ℕ := 13

-- Define the function to calculate Minyoung's position from the front
def position_from_front (total : ℕ) (from_back : ℕ) : ℕ :=
  total - from_back + 1

-- The theorem to prove Minyoung's position from the front
theorem minyoung_position_from_front : position_from_front total_students minyoung_position_from_back = 15 := 
by {
  unfold position_from_front,
  simp,
  sorry -- The actual proof
}

end minyoung_position_from_front_l490_490893


namespace chris_mixed_nuts_l490_490132

theorem chris_mixed_nuts :
  ∃ x : ℕ, (let R := 1 in let N := 2 * R in let cost_raisins := 3 * R in
            let cost_nuts := x * N in
            let total_cost_mixture := cost_raisins + cost_nuts in
            (cost_raisins : ℝ) = 0.2727272727272727 * total_cost_mixture ∧
            (N : ℝ) = 2 * (R : ℝ)) ∧ x = 4 :=
begin
  have fraction_form : 0.2727272727272727 = (3 : ℝ) / 11, 
  { norm_num },
  sorry
end

end chris_mixed_nuts_l490_490132


namespace ratio_a_to_c_l490_490062

theorem ratio_a_to_c {a b c : ℚ} (h1 : a / b = 4 / 3) (h2 : b / c = 1 / 5) :
  a / c = 4 / 5 := 
sorry

end ratio_a_to_c_l490_490062


namespace solution_set_of_inequality_l490_490394

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / (3 - x) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} :=
by
  sorry

end solution_set_of_inequality_l490_490394


namespace count_integers_in_range_l490_490248

theorem count_integers_in_range : 
  {n : ℤ | -4 ≤ n ∧ n ≤ 9}.toFinset.card = 14 :=
by
  sorry

end count_integers_in_range_l490_490248


namespace exactly_one_greater_than_one_l490_490380

theorem exactly_one_greater_than_one (x1 x2 x3 : ℝ) 
  (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3)
  (h4 : x1 * x2 * x3 = 1)
  (h5 : x1 + x2 + x3 > (1 / x1) + (1 / x2) + (1 / x3)) :
  (x1 > 1 ∧ x2 ≤ 1 ∧ x3 ≤ 1) ∨ 
  (x1 ≤ 1 ∧ x2 > 1 ∧ x3 ≤ 1) ∨ 
  (x1 ≤ 1 ∧ x2 ≤ 1 ∧ x3 > 1) :=
sorry

end exactly_one_greater_than_one_l490_490380


namespace area_of_triangle_AF1F2_l490_490073

-- Definitions for the ellipse and points
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 7) = 1

def F1 : (ℝ × ℝ) := (√2, 0)
def F2 : (ℝ × ℝ) := (-√2, 0)
def A : (ℝ × ℝ)

-- Given condition: ∠AF_1F_2 = 45°
def angle_AF1F2_is_45 (A : ℝ × ℝ) : Prop := 
  ∠ A F1 F2 = 45 * (π / 180)

-- Triangle area to prove
def area_AF1F2 (A F1 F2 : ℝ × ℝ) : ℝ := 
  0.5 * abs ((A.1 * (F1.2 - F2.2)) + (F1.1 * (F2.2 - A.2)) + (F2.1 * (A.2 - F1.2)))

theorem area_of_triangle_AF1F2 :
  forall (A : ℝ × ℝ), (ellipse_eq A.1 A.2) ∧ (angle_AF1F2_is_45 A) -> (area_AF1F2 A F1 F2 = 7/2) :=
  sorry

end area_of_triangle_AF1F2_l490_490073


namespace evaluate_diamond_l490_490186

variable {R : Type} [Field R]

-- Define the operation diamond
def diamond (x y : R) : R := (x + y) / (x + y - 2 * x * y)

-- Given conditions
variable (x y : R) (hx : x ≠ y)

-- Proof problem statement
theorem evaluate_diamond : diamond (diamond 3 4) 5 = 39 / 74 := by
  sorry

end evaluate_diamond_l490_490186


namespace greatest_value_of_sum_l490_490424

variable (x y : ℝ)

-- Conditions
axiom sum_of_squares : x^2 + y^2 = 130
axiom product : x * y = 36

-- Statement to prove
theorem greatest_value_of_sum : x + y ≤ Real.sqrt 202 := sorry

end greatest_value_of_sum_l490_490424


namespace race_head_start_l490_490884

theorem race_head_start (Va Vb L H : ℚ) (h : Va = 30 / 17 * Vb) :
  H = 13 / 30 * L :=
by
  sorry

end race_head_start_l490_490884


namespace sum_of_solutions_l490_490802

theorem sum_of_solutions (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) : 
  -b / a = 9 :=
by
  -- The proof is omitted here (hence the 'sorry')
  sorry

end sum_of_solutions_l490_490802


namespace residue_195_15_18_8_4_mod_17_l490_490523

theorem residue_195_15_18_8_4_mod_17 :
  (195 * 15 - 18 * 8 + 4) % 17 = 7 := by
  -- Introduce necessary equivalences and simplifications
  have h1 : 195 % 17 = 3 := sorry
  have h2 : 18 % 17 = 1 := sorry
  calc
    (195 * 15 - 18 * 8 + 4) % 17 = ((3 * 15) % 17 - (1 * 8) % 17 + 4 % 17) % 17 := by sorry
    ... = (45 % 17 - 8 % 17 + 4 % 17) % 17 := by sorry
    ... = (11 - 8 + 4) % 17 := by sorry
    ... = 7 := by sorry

end residue_195_15_18_8_4_mod_17_l490_490523


namespace longest_side_l490_490178

-- Definition of the conditions
def feasible_region (x y : ℝ) : Prop :=
  (x + 2 * y ≤ 6) ∧ (x + y ≥ 3) ∧ (x ≥ 0) ∧ (y ≥ 0)

-- Definition of the length of a segment given two points
def length (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Definition of the vertices of the triangle (using the solution)
def vert1 : ℝ × ℝ := (0, 3)
def vert2 : ℝ × ℝ := (3, 0)
def vert3 : ℝ × ℝ := (0, 0)

-- Prove the length of the longest side
theorem longest_side :
  length vert1 vert2 = 3 * real.sqrt 2 := by sorry

end longest_side_l490_490178


namespace integer_solutions_count_l490_490246

theorem integer_solutions_count :
  {n : ℤ | (n + 4) * (n - 9) ≤ 0}.card = 14 :=
by sorry

end integer_solutions_count_l490_490246


namespace miranda_can_stuff_10_pillows_l490_490324

def feathers_needed_per_pillow : ℕ := 2
def goose_feathers_per_pound : ℕ := 300
def duck_feathers_per_pound : ℕ := 500
def goose_total_feathers : ℕ := 3600
def duck_total_feathers : ℕ := 4000

theorem miranda_can_stuff_10_pillows :
  (goose_total_feathers / goose_feathers_per_pound + duck_total_feathers / duck_feathers_per_pound) / feathers_needed_per_pillow = 10 :=
by
  sorry

end miranda_can_stuff_10_pillows_l490_490324


namespace find_angle_and_perimeter_l490_490288

-- Define the given conditions
def triangle_sides (a b c : ℝ) (A B C : ℝ) : Prop :=
  2 * cos C * (a * cos B + b * cos A) = c

def area_triangle (a b : ℝ) : ℝ := (sqrt 3 / 4) * a * b

-- Define the proof problem
theorem find_angle_and_perimeter
  (a b c A B C : ℝ)
  (h1 : triangle_sides a b c A B C)
  (h2 : c = sqrt 7)
  (h3 : area_triangle a b = 3 * sqrt 3 / 2) :
  C = π / 3 ∧ a + b + c = 5 + sqrt 7 := 
  sorry

end find_angle_and_perimeter_l490_490288


namespace x_when_y_is_125_l490_490399

noncomputable def C : ℝ := (2^2) * (5^2)

theorem x_when_y_is_125 
  (x y : ℝ) 
  (h_pos : x > 0 ∧ y > 0) 
  (h_inv : x^2 * y^2 = C) 
  (h_initial : y = 5) 
  (h_x_initial : x = 2) 
  (h_y : y = 125) : 
  x = 2 / 25 :=
by
  sorry

end x_when_y_is_125_l490_490399


namespace problem1_problem2_l490_490444

-- Problem 1
theorem problem1 (l : Line) :
  (intersects l ⟨2, 1, -8⟩ ⟨1, -2, 1⟩) → (equal_intercepts l) → (equation l = "2x - 3y = 0" ∨ equation l = "x + y - 5 = 0") :=
sorry

-- Problem 2
theorem problem2 (l : Line) (A : Point) :
  (passes_through l A) → (distance_from_origin l = 3) → (equation l = "x = -3" ∨ equation l = "5x-12y+39=0") :=
sorry

end problem1_problem2_l490_490444


namespace vertical_angles_equal_l490_490727

-- Define what it means for two angles to be vertical angles.
def are_vertical_angles (α β : ℝ) : Prop :=
  ∃ (γ δ : ℝ), α + γ = 180 ∧ β + δ = 180 ∧ γ = β ∧ δ = α

-- The theorem statement:
theorem vertical_angles_equal (α β : ℝ) : are_vertical_angles α β → α = β := 
  sorry

end vertical_angles_equal_l490_490727


namespace remainder_same_l490_490725

def sum_digits_at_odd_positions (digits: List ℕ): ℕ :=
  digits.enum.filterMap (λ ⟨i, d⟩, if i % 2 = 1 then some d else none).sum

def sum_digits_at_even_positions (digits: List ℕ): ℕ :=
  digits.enum.filterMap (λ ⟨i, d⟩, if i % 2 = 0 then some d else none).sum

theorem remainder_same (digits: List ℕ) (h: ∀ i, i < digits.length → digits[i] < 10):
  let N := digits.enum.foldl (λ acc ⟨i, d⟩, acc + d * 10^i) 0
  let S := sum_digits_at_even_positions digits - sum_digits_at_odd_positions digits
  (N % 11) = (S % 11) :=
sorry

end remainder_same_l490_490725


namespace sufficient_drivers_and_correct_time_l490_490915

-- Conditions definitions
def one_way_minutes := 2 * 60 + 40  -- 2 hours 40 minutes in minutes
def round_trip_minutes := 2 * one_way_minutes  -- round trip in minutes
def rest_minutes := 60  -- mandatory rest period in minutes

-- Time checks for drivers
def driver_a_return := 12 * 60 + 40  -- Driver A returns at 12:40 PM in minutes
def driver_a_next_trip := driver_a_return + rest_minutes  -- Driver A's next trip time
def driver_d_departure := 13 * 60 + 5  -- Driver D departs at 13:05 in minutes

-- Verify sufficiency of four drivers and time correctness
theorem sufficient_drivers_and_correct_time : 
  4 = 4 ∧ (driver_a_next_trip + round_trip_minutes = 21 * 60 + 30) :=
by
  -- Explain the reasoning path that leads to this conclusion within this block
  sorry

end sufficient_drivers_and_correct_time_l490_490915


namespace sin_150_eq_one_half_l490_490970

theorem sin_150_eq_one_half :
  ∀ θ : ℝ, (∀ θ, sin (θ) = cos (90 - θ)) → (∀ θ, cos (-θ) = cos (θ)) → cos 60 = 1/2 → sin 150 = 1/2 :=
by
  sorry

end sin_150_eq_one_half_l490_490970


namespace find_total_amount_l490_490467

variables (A B C : ℕ) (total_amount : ℕ) 

-- Conditions
def condition1 : Prop := B = 36
def condition2 : Prop := 100 * B / 45 = A
def condition3 : Prop := 100 * C / 30 = A

-- Proof statement
theorem find_total_amount (h1 : condition1 B) (h2 : condition2 A B) (h3 : condition3 A C) :
  total_amount = 300 :=
sorry

end find_total_amount_l490_490467


namespace target_hit_probability_l490_490716

-- Define the probabilities of Person A and Person B hitting the target
def prob_A_hits := 0.8
def prob_B_hits := 0.7

-- Define the probability that the target is hit when both shoot independently at the same time
def prob_target_hit := 1 - (1 - prob_A_hits) * (1 - prob_B_hits)

theorem target_hit_probability : prob_target_hit = 0.94 := 
by
  sorry

end target_hit_probability_l490_490716


namespace matrix_vector_product_l490_490145

def mat : Matrix (Fin 2) (Fin 2) ℤ := ![![3, -2], ![-4, 5]]
def vec : Fin 2 → ℤ := ![4, -2]

theorem matrix_vector_product : mat.mul_vec vec = ![16, -26] :=
by
  sorry

end matrix_vector_product_l490_490145


namespace miranda_can_stuff_10_pillows_l490_490323

def feathers_needed_per_pillow : ℕ := 2
def goose_feathers_per_pound : ℕ := 300
def duck_feathers_per_pound : ℕ := 500
def goose_total_feathers : ℕ := 3600
def duck_total_feathers : ℕ := 4000

theorem miranda_can_stuff_10_pillows :
  (goose_total_feathers / goose_feathers_per_pound + duck_total_feathers / duck_feathers_per_pound) / feathers_needed_per_pillow = 10 :=
by
  sorry

end miranda_can_stuff_10_pillows_l490_490323


namespace solution_part1_solution_part2_l490_490582

variable (f : ℝ → ℝ) (a x m : ℝ)

def problem_statement :=
  (∀ x : ℝ, f x = abs (x - a)) ∧
  (∀ x : ℝ, f x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5)

theorem solution_part1 (x : ℝ) (h : problem_statement f a) : a = 2 :=
by
  sorry

theorem solution_part2 (x : ℝ) (h : problem_statement f a) :
  (∀ x : ℝ, f x + f (x + 5) ≥ m) → m ≤ 5 :=
by
  sorry

end solution_part1_solution_part2_l490_490582


namespace dog_speed_is_16_kmh_l490_490096

variable (man's_speed : ℝ := 4) -- man's speed in km/h
variable (total_path_length : ℝ := 625) -- total path length in meters
variable (remaining_distance : ℝ := 81) -- remaining distance in meters

theorem dog_speed_is_16_kmh :
  let total_path_length_km := total_path_length / 1000
  let remaining_distance_km := remaining_distance / 1000
  let man_covered_distance_km := total_path_length_km - remaining_distance_km
  let time := man_covered_distance_km / man's_speed
  let dog_total_distance_km := 4 * (2 * total_path_length_km)
  let dog_speed := dog_total_distance_km / time
  dog_speed = 16 :=
by
  sorry

end dog_speed_is_16_kmh_l490_490096


namespace min_number_of_squares_l490_490033

theorem min_number_of_squares (length width : ℕ) (h_length : length = 10) (h_width : width = 9) : 
  ∃ n, n = 10 :=
by
  sorry

end min_number_of_squares_l490_490033


namespace smallest_t_value_l490_490389

theorem smallest_t_value : ∀ (t : ℕ), (7.5 < 11 + t ∧ 
                                        t < 18.5 ∧
                                        t > 3.5) → t = 4 :=
by
  intros t h
  sorry

end smallest_t_value_l490_490389


namespace probability_one_second_class_l490_490450

def C (n k : ℕ) := nat.choose n k  -- Combination function

theorem probability_one_second_class :
  C 10 1 * C 90 3 / C 100 4 = (C 10 1 * C 90 3 : ℚ) / (C 100 4 : ℚ) :=
by {
  sorry
}

end probability_one_second_class_l490_490450


namespace crushing_load_value_l490_490277

variable (T H : ℝ)
def L (T H : ℝ) : ℝ := (36 * T^3) / (H^3)

theorem crushing_load_value (hT : T = 3) (hH : H = 9) : L T H = 4 / 3 := by
  rw [hT, hH]
  sorry

end crushing_load_value_l490_490277


namespace one_of_sasha_zhenya_valya_is_girl_l490_490358

variable (α : Type) [LinearOrder α]

-- Given a type α which represents the height type of children and a linear order on this type

-- Define a structure for children participants
structure Child where
  name : String
  height : α
  isGirl : Bool

-- Let children be a list of Child
variable (children : List Child)

-- Sasha, Zhenya, Valya as named children
noncomputable def Sasha : Child := sorry
noncomputable def Zhenya : Child := sorry
noncomputable def Valya : Child := sorry

-- Assume Sasha, Zhenya, and Valya received the same number of candies
axiom same_candies_received : ∀ (c : Child), (c = Sasha ∨ c = Zhenya ∨ c = Valya) → 
  candies_received c = candies_received Sasha

-- Assume all other children received fewer candies than Sasha, Zhenya, and Valya
axiom fewer_candies_received : ∀ (c : Child), (c ≠ Sasha ∧ c ≠ Zhenya ∧ c ≠ Valya) → 
  candies_received c < candies_received Sasha

-- Assume all children have different heights
axiom different_heights : ∀ (c1 c2 : Child), c1 ≠ c2 → c1.height ≠ c2.height

-- Define function to check if a child is a boy
def isBoy (c : Child) : Prop := ¬c.isGirl

-- Define candies received by each child (this is simplified assumption; actual computation may be complex)
noncomputable def candies_received : Child → ℕ := sorry

-- Now, the theorem based on given condition
theorem one_of_sasha_zhenya_valya_is_girl : Sasha.isGirl ∨ Zhenya.isGirl ∨ Valya.isGirl := sorry

end one_of_sasha_zhenya_valya_is_girl_l490_490358


namespace find_lambda_l490_490212

variables {E : Type*} [AddCommGroup E] [Module ℝ E]
variable (e1 e2 : E)
variable (a : E)
variable (b : E)
variable (λ : ℝ)

theorem find_lambda (h0 : e1 ≠ (0 : E)) (h1 : e2 ≠ (0 : E)) (h2 : a = 2 • e1 - e2) (h3 : a = (λ • e2 + e1)) :
  λ = -1 / 2 :=
by
  sorry

end find_lambda_l490_490212


namespace angle_A_range_l490_490287

open Real

theorem angle_A_range (A : ℝ) (h1 : sin A + cos A > 0) (h2 : tan A < sin A) (h3 : 0 < A ∧ A < π) : 
  π / 2 < A ∧ A < 3 * π / 4 :=
by
  sorry

end angle_A_range_l490_490287


namespace power_of_two_l490_490693

theorem power_of_two (b m n : ℕ) (hb : b > 1) (hmn : m ≠ n)
  (hpdiv : ∀ p : ℕ, p.prime ∧ p ∣ (b^m - 1) ↔ p.prime ∧ p ∣ (b^n - 1)) :
  ∃ k : ℕ, b + 1 = 2^k := 
sorry

end power_of_two_l490_490693


namespace scheduling_arrangements_l490_490083

-- Definitions
def Subject : Type := {Chinese, Mathematics, PhysicalEducation, English}
def validPeriod (p : Nat) : Prop := p ≠ 1 ∧ p ≠ 4

-- Proof statement
theorem scheduling_arrangements : 
  ∃ arrangements : Finset (Fin 4 → Subject), arrangements.card = 12 := 
by
  sorry

end scheduling_arrangements_l490_490083


namespace floor_sum_cubed_root_l490_490184

theorem floor_sum_cubed_root (n : ℕ) (h1 : n ≥ 3) (h2 : ∀ m : ℕ, m^2 ∣ n → m = 1) :
    ∑ k in Finset.range ((n-2)*(n-1)), ⌊(k * n)^(1/3)⌋ = (n-2)*(n-1)*(3*n-5)/4 := by
  sorry

end floor_sum_cubed_root_l490_490184


namespace smallest_number_sum_consecutive_is_105_l490_490525
noncomputable def smallest_number_sum_consecutive_numbers : ℕ :=
  let S5 := ∃ k : ℕ, ∃ n : ℕ, k = 5 ∧ 5 * n + 10 = k in
  let S6 := ∃ l : ℕ, ∃ n : ℕ, l = 3 ∧ 6 * n + 15 = l in
  let S7 := ∃ m : ℕ, ∃ n : ℕ, m = 7 ∧ 7 * n + 21 = m in
  (least_common_multiple 5 3 7)

theorem smallest_number_sum_consecutive_is_105 : smallest_number_sum_consecutive_numbers = 105 := sorry

end smallest_number_sum_consecutive_is_105_l490_490525


namespace proof_f_derivative_neg1_l490_490561

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  a * x ^ 4 + b * x ^ 2 + c

noncomputable def f_derivative (x : ℝ) (a b : ℝ) : ℝ :=
  4 * a * x ^ 3 + 2 * b * x

theorem proof_f_derivative_neg1
  (a b c : ℝ) (h : f_derivative 1 a b = 2) :
  f_derivative (-1) a b = -2 :=
by
  sorry

end proof_f_derivative_neg1_l490_490561


namespace trapezoid_area_PQS_l490_490285

-- Define the problem conditions:
variables {PQ RS : ℝ}
axiom PQRS_trapezoid_area : 24
axiom RS_eq_3PQ : RS = 3 * PQ

-- Define the area of the trapezoid and its relationship to the triangles:
noncomputable def area_trapezoid (PQ_area PQS_area : ℝ) : Prop :=
  PQ_area + 3 * PQ_area = 24

-- State the proof problem:
theorem trapezoid_area_PQS :
  ∃ (PQS_area : ℝ), area_trapezoid PQS_area ∧ PQS_area = 6 :=
by
  use 6
  simp [area_trapezoid]
  sorry

end trapezoid_area_PQS_l490_490285


namespace sequence_properties_l490_490388

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ a 2 = 20 ∧ a 3 = 56 ∧ ∀ n, a (n + 3) = 7 * a (n + 2) - 11 * a (n + 1) + 5 * a n - 3 * 2^n

theorem sequence_properties (a : ℕ → ℕ) (h : sequence a) :
  (∀ n > 0, a n > 0) ∧ a 673 % 673 = 663 := sorry

end sequence_properties_l490_490388


namespace find_a_of_line_slope_l490_490219

theorem find_a_of_line_slope (a : ℝ) (h1 : a > 0)
  (h2 : ∃ (b : ℝ), (a, 5) = (b * 1, b * 2) ∧ (2, a) = (b * 1, 2 * b) ∧ b = 1) 
  : a = 3 := 
sorry

end find_a_of_line_slope_l490_490219


namespace percent_of_x_is_y_l490_490061

variable {x y : ℝ}

theorem percent_of_x_is_y
  (h : 0.5 * (x - y) = 0.4 * (x + y)) :
  y = (1 / 9) * x :=
sorry

end percent_of_x_is_y_l490_490061


namespace length_of_train_is_400_l490_490468

noncomputable def speed_kmh := 90 -- Speed in km/h
noncomputable def time_seconds := 16 -- Time in seconds

-- Conversion factor from km/h to m/s:
def km_h_to_m_s (speed : ℝ) : ℝ := speed * (5 / 18)

-- Convert speed to m/s
noncomputable def speed_m_s : ℝ := km_h_to_m_s speed_kmh

-- Calculate the length of the train using the formula: Length = Speed * Time
noncomputable def length_of_train : ℝ := speed_m_s * time_seconds

theorem length_of_train_is_400 :
  length_of_train = 400 :=
sorry

end length_of_train_is_400_l490_490468


namespace orthocenter_of_triangle_MNQ_l490_490942

-- Definitions of points and triangles in a metric space
variable {Point : Type}
variable [MetricSpace Point]

-- Given conditions
variable {M N K Q : Point}
variable (MN SQ : real) (NK SQ : real) (MK SQ : real)
variable (MQ SQ : real) (QK SQ : real) (NQ SQ : real)

-- The theorem to be proven: Q is the orthocenter of triangle MNK given specific conditions
theorem orthocenter_of_triangle_MNQ :
  MN SQ + QK SQ = NK SQ + MQ SQ ∧ NK SQ + MQ SQ = MK SQ + NQ SQ ∧ MK SQ + NQ SQ = MN SQ + QK SQ →
    isOrthocenter Q M N K :=
sorry

end orthocenter_of_triangle_MNQ_l490_490942


namespace consecutive_odd_numbers_count_l490_490766

theorem consecutive_odd_numbers_count (n : ℕ) (avg : ℤ) (diff : ℤ) 
  (consecutive_odds : ℕ → ℤ)
  (h_avg : avg = 55)
  (h_diff : diff = 8) 
  (h_avg_def : avg = ∑ i in Finset.range n, consecutive_odds i / n)
  (h_diff_def : diff = consecutive_odds (n - 1) - consecutive_odds 0)
  (h_consecutive : ∀ i : ℕ, i < n - 1 → consecutive_odds (i + 1) - consecutive_odds i = 2) :
  n = 9 :=
by
  sorry

end consecutive_odd_numbers_count_l490_490766


namespace bow_area_fraction_l490_490638

noncomputable def midpoints (A B C : Point) : Point × Point :=
  let A1 := midpoint B C
  let B1 := midpoint A C
  (A1, B1)

noncomputable def quarters (A B : Point) : Point × Point × Point :=
  let C1 := point_divide A B (1/4)
  let C2 := point_divide A B (2/4)
  let C3 := point_divide A B (3/4)
  (C1, C2, C3)

theorem bow_area_fraction (A B C A1 B1 C1 C2 C3 : Point)
  (hA1 : A1 = midpoint B C) (hB1 : B1 = midpoint A C)
  (hC1 : C1 = point_divide A B (1/4)) (hC2 : C2 = point_divide A B (2/4)) (hC3 : C3 = point_divide A B (3/4)) :
  let ABC := triangle_area A B C in
  let bow := parallelogram_area C1 C3 A1 B1 in
  bow = (1/4) * ABC :=
sorry

end bow_area_fraction_l490_490638


namespace angle_ABN_36_degree_l490_490654

theorem angle_ABN_36_degree
  (O A K M N B : Point)
  (h1 : IsCenter O)
  (h2 : Perpendicular OK OA)
  (h3 : IsMidpoint M OK)
  (h4 : Parallel BN OK)
  (h5 : ∠AMN = ∠NMO) :
  ∠ABN = 36 :=
sorry

end angle_ABN_36_degree_l490_490654


namespace parametric_to_parabola_l490_490006

theorem parametric_to_parabola (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) :
  let x := sqrt (1 + sin θ)
  let y := cos^2 (π / 4 - θ / 2)
  y = (x^2) / 2 ∧ (x = 1 → y = 1 / 2) :=
by
  let x := sqrt (1 + sin θ)
  let y := cos^2 (π / 4 - θ / 2)
  have h1 : x^2 = 1 + sin θ := by sorry
  have h2 : y = (1 + sin θ) / 2 := by sorry
  have h3 : y = x^2 / 2 := by sorry
  have h4 : x = 1 → y = 1 / 2 := by sorry
  exact ⟨h3, h4⟩

end parametric_to_parabola_l490_490006


namespace num_factors_of_n_multiples_of_360_l490_490615

-- Define n
def n : ℕ := 2^12 * 3^15 * 5^9

-- Define 360, its prime factorization for clarity
def c : ℕ := 360
def factorization_360 : ℕ × ℕ × ℕ := (3, 2, 1) -- (exponent of 2, exponent of 3, exponent of 5)

-- Define the problem as a theorem
theorem num_factors_of_n_multiples_of_360 : 
  let factors_2 := finset.range (12 + 1) \ finset.range 3,
      factors_3 := finset.range (15 + 1) \ finset.range 2,
      factors_5 := finset.range (9 + 1) \ finset.range 1 in
  factors_2.card * factors_3.card * factors_5.card = 1260 := 
by
  sorry

end num_factors_of_n_multiples_of_360_l490_490615


namespace find_f_l490_490199

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 2) * x^2 + 2 * x * f'' 2017 - 2017 * Real.log x

theorem find_f''_2017 :
  f'' 2017 = -2016 :=
sorry

end find_f_l490_490199


namespace find_a6_l490_490221

noncomputable def a_n (n : ℕ) : ℝ := sorry
noncomputable def S_n (n : ℕ) : ℝ := sorry
noncomputable def r : ℝ := sorry

axiom h_pos : ∀ n, a_n n > 0
axiom h_s3 : S_n 3 = 14
axiom h_a3 : a_n 3 = 8

theorem find_a6 : a_n 6 = 64 := by sorry

end find_a6_l490_490221


namespace best_initial_estimate_l490_490166

def estimate_submission_score (E D : ℕ) : ℝ :=
  2 / (0.5 * |E - D| + 1)

theorem best_initial_estimate :
  ∃ E : ℕ, 1 ≤ E ∧ E ≤ 50 ∧ (∀ D : ℕ, 1 ≤ D ∧ D ≤ 50 → estimate_submission_score 30 D = 2 / (0.5 * abs (30 - D) + 1)) :=
begin
  use 30,
  split,
  exact nat.one_le_of_lt (lt_of_le_of_lt (zero_le 30) (lt_add_one 50)),
  split,
  exact nat.le_of_eq (rfl),
  intros D hD,
  simp [estimate_submission_score],
  sorry,
end

end best_initial_estimate_l490_490166


namespace sum_of_solutions_of_quadratic_eq_l490_490814

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 9 * x + 20 = 0

-- Prove that the sum of the solutions to this equation is 9
theorem sum_of_solutions_of_quadratic_eq : 
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 9) := 
begin
  -- Proof is omitted
  sorry
end

end sum_of_solutions_of_quadratic_eq_l490_490814


namespace extreme_points_range_of_b_l490_490585

/-- Define the function f as given in the problem -/
noncomputable def f (a x : ℝ) : ℝ := a * x - 1 - Real.log x

/-- Part (I) -/
theorem extreme_points (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, ¬∃ y > 0, f a' y < f a' x) ∧
  (a > 0 → ∃ x > 0, ∀ y > 0, (y ≠ x → f a' x < f a' y ∨ f a' x > f a' y)) :=
begin
  sorry
end

/-- Part (II) -/
theorem range_of_b :
  ∃ b : ℝ, ∀ x > 0, f 1 x ≥ b * x - 2 ↔ b ≤ 1 - 1 / Real.exp 2 :=
begin
  sorry
end

end extreme_points_range_of_b_l490_490585


namespace area_of_sector_l490_490779

noncomputable def circleAreaAboveXAxisAndRightOfLine : ℝ :=
  let radius := 10
  let area_of_circle := Real.pi * radius^2
  area_of_circle / 4

theorem area_of_sector :
  circleAreaAboveXAxisAndRightOfLine = 25 * Real.pi := sorry

end area_of_sector_l490_490779


namespace distance_from_point_to_line_l490_490362

theorem distance_from_point_to_line : 
  ∀ (x0 y0 a b c : ℝ), 
  (x0, y0) = (0, 5) → 
  (-a + b * y + c) = 0 → 
  (∃ d : ℝ, d = sqrt 5) := by
  intros x0 y0 a b c h h_line
  use sqrt 5
  sorry

end distance_from_point_to_line_l490_490362


namespace initial_cost_of_each_wand_l490_490120

theorem initial_cost_of_each_wand (W1 W2 W3 : ℝ) (W1_weight : W1 = 1.5) (W2_weight : W2 = 1.8) (W3_weight : W3 = 2.0)
  (discount : ℝ) (discount_rate : discount = 0.10)
  (shipping_rate : ℝ) (shipping_rate_value : shipping_rate = 3.0)
  (num_wands : ℝ) (num_wands_value : num_wands = 3.0)
  (extra_charge : ℝ) (extra_charge_value : extra_charge = 5.0)
  (total_collected : ℝ) (total_collected_value : total_collected = 160.0) :
  let total_shipping := (W1 + W2 + W3) * shipping_rate,
      discounted_cost := num_wands * (1 - discount) * C,
      total_cost := discounted_cost + total_shipping,
      total_sold := total_cost + num_wands * extra_charge
  in total_sold = total_collected → C = 47.81 :=
by
  intros h
  sorry

end initial_cost_of_each_wand_l490_490120


namespace die_odd_dots_probability_l490_490040

theorem die_odd_dots_probability (d : die) (e : events) :
  (∃ (d1 d2 : dot) (h1 : d.remove(d1)), (h2 : d.remove(d2)),
  probability (d.top_face.odd_dots) = 1 / 60 :=
sorry

end die_odd_dots_probability_l490_490040


namespace martin_big_bell_rings_l490_490709

theorem martin_big_bell_rings (B S : ℚ) (h1 : S = B / 3 + B^2 / 4) (h2 : S + B = 52) : B = 12 :=
by
  sorry

end martin_big_bell_rings_l490_490709


namespace chad_savings_correct_l490_490951

variable (earnings_mowing : ℝ := 600)
variable (earnings_birthday : ℝ := 250)
variable (earnings_video_games : ℝ := 150)
variable (earnings_odd_jobs : ℝ := 150)
variable (tax_rate : ℝ := 0.10)

noncomputable def total_earnings : ℝ := 
  earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs

noncomputable def taxes : ℝ := 
  tax_rate * total_earnings

noncomputable def money_after_taxes : ℝ := 
  total_earnings - taxes

noncomputable def savings_mowing : ℝ := 
  0.50 * earnings_mowing

noncomputable def savings_birthday : ℝ := 
  0.30 * earnings_birthday

noncomputable def savings_video_games : ℝ := 
  0.40 * earnings_video_games

noncomputable def savings_odd_jobs : ℝ := 
  0.20 * earnings_odd_jobs

noncomputable def total_savings : ℝ := 
  savings_mowing + savings_birthday + savings_video_games + savings_odd_jobs

theorem chad_savings_correct : total_savings = 465 := by
  sorry

end chad_savings_correct_l490_490951


namespace g_neg_one_l490_490566

variables (f : ℝ → ℝ) (g : ℝ → ℝ)
variables (h₀ : ∀ x : ℝ, f (-x) + x^2 = -(f x + x^2))
variables (h₁ : f 1 = 1)
variables (h₂ : ∀ x : ℝ, g x = f x + 2)

theorem g_neg_one : g (-1) = -1 :=
by
  sorry

end g_neg_one_l490_490566


namespace sum_of_solutions_of_quadratic_eq_l490_490817

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 9 * x + 20 = 0

-- Prove that the sum of the solutions to this equation is 9
theorem sum_of_solutions_of_quadratic_eq : 
  (∃ a b : ℝ, quadratic_eq a ∧ quadratic_eq b ∧ a + b = 9) := 
begin
  -- Proof is omitted
  sorry
end

end sum_of_solutions_of_quadratic_eq_l490_490817


namespace sum_of_solutions_l490_490856

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | x^2 = 9*x - 20}, x) = 9 := 
sorry

end sum_of_solutions_l490_490856


namespace trains_cross_time_l490_490773

noncomputable def time_to_cross : ℝ := 
  let length_train1 := 110 -- length of the first train in meters
  let length_train2 := 150 -- length of the second train in meters
  let speed_train1 := 60 * 1000 / 3600 -- speed of the first train in meters per second
  let speed_train2 := 45 * 1000 / 3600 -- speed of the second train in meters per second
  let bridge_length := 340 -- length of the bridge in meters
  let total_distance := length_train1 + length_train2 + bridge_length -- total distance to be covered
  let relative_speed := speed_train1 + speed_train2 -- relative speed in meters per second
  total_distance / relative_speed

theorem trains_cross_time :
  abs (time_to_cross - 20.57) < 0.01 :=
sorry

end trains_cross_time_l490_490773


namespace projection_correct_l490_490631

def vector_a : ℝ × ℝ × ℝ := (1, 0, 2)
def vector_b : ℝ × ℝ × ℝ := (0, 1, -1)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

noncomputable def projection (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
let scalar := (dot_product a b) / (magnitude b ^ 2) in
(scalar * b.1, scalar * b.2, scalar * b.3)

theorem projection_correct :
  projection vector_a vector_b = (0, -1, 1) :=
by
  sorry

end projection_correct_l490_490631


namespace horner_polynomial_value_l490_490044

theorem horner_polynomial_value :
  let f (x : ℝ) := 12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6 in
  let x := -4 in
  let V0 := 3 in
  let V1 := V0 * x + 5 in
  let V2 := V1 * x + 6 in
  let V3 := V2 * x + 79 in
  let V4 := V3 * x - 8 in
  V4 = 220 :=
by {
  let f (x : ℝ) := 12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6,
  let x := -4,
  let V0 := 3,
  let V1 := V0 * x + 5,
  let V2 := V1 * x + 6,
  let V3 := V2 * x + 79,
  let V4 := V3 * x - 8,
  have : V4 = 220,
  sorry
}

end horner_polynomial_value_l490_490044


namespace sum_f_values_l490_490543

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem sum_f_values : 
  f 1 + f 2 + f 3 + f 4 + f (1/2) + f (1/3) + f (1/4) = 7 / 2 := 
by
  sorry

end sum_f_values_l490_490543


namespace mandy_total_shirts_l490_490707

-- Condition definitions
def black_packs : ℕ := 6
def black_shirts_per_pack : ℕ := 7
def yellow_packs : ℕ := 8
def yellow_shirts_per_pack : ℕ := 4

theorem mandy_total_shirts : 
  (black_packs * black_shirts_per_pack + yellow_packs * yellow_shirts_per_pack) = 74 :=
by
  sorry

end mandy_total_shirts_l490_490707


namespace first_group_persons_l490_490735

-- We need to define the conditions as assumptions and connect the question to the answer

theorem first_group_persons (P : ℕ) :
  (∃ P : ℕ, (P * 12 * 5 = 30 * 14 * 6)) → P = 42 :=
by
  intro h
  cases h with n hn
  have h_eq : n * 60 = 30 * 84 := by 
    rw [←hn, mul_assoc, mul_assoc]
  have eq_2520 : 30 * 84 = 2520 := by norm_num
  rw [h_eq, eq_2520] at hn
  have n_eq : n = 42 := by linarith
  exact n_eq.symm

end first_group_persons_l490_490735


namespace number_of_remaining_elements_l490_490023

noncomputable def S : Finset ℕ := (Finset.range 51).filter (λ x, x > 0)

def multiples_of_2 (S : Finset ℕ) : Finset ℕ := S.filter (λ x, x % 2 = 0)
def multiples_of_3 (S : Finset ℕ) : Finset ℕ := S.filter (λ x, x % 3 = 0)

def remaining_elements (S : Finset ℕ) : Finset ℕ :=
  S \ multiples_of_2 S \ multiples_of_3 (S \ multiples_of_2 S)

theorem number_of_remaining_elements : remaining_elements S.card = 17 := by sorry

end number_of_remaining_elements_l490_490023


namespace triangle_inequality_l490_490102

-- Define the nondegenerate condition for the triangle's side lengths.
def nondegenerate_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the perimeter condition for the triangle.
def triangle_perimeter (a b c : ℝ) (p : ℝ) : Prop :=
  a + b + c = p

-- The main theorem to prove the given inequality.
theorem triangle_inequality (a b c : ℝ) (h_non_deg : nondegenerate_triangle a b c) (h_perim : triangle_perimeter a b c 1) :
  abs ((a - b) / (c + a * b)) + abs ((b - c) / (a + b * c)) + abs ((c - a) / (b + a * c)) < 2 :=
by
  sorry

end triangle_inequality_l490_490102


namespace probability_correct_l490_490406

noncomputable def probability_divisible_by_3 : ℚ :=
  let S := {x : ℕ | 1 ≤ x ∧ x ≤ 15}
  let multiples_of_3 := {x : ℕ | 1 ≤ x ∧ x ≤ 15 ∧ x % 3 = 0}
  let total_ways := (S.card.choose 3 : ℕ) -- Total ways to choose 3 distinct integers from {1, ..., 15}
  let valid_ways := (multiples_of_3.card.choose 3 : ℕ) -- Ways to choose 3 multiples of 3 from {3, 6, 9, 12, 15}
  valid_ways / total_ways

theorem probability_correct :
  probability_divisible_by_3 = 2 / 91 :=
sorry

end probability_correct_l490_490406


namespace harriett_found_3_dimes_l490_490867

-- Definitions and conditions
def num_quarters := 10
def num_nickels := 3
def num_pennies := 5
def total_money := 3 -- in dollars

-- Values of coins in dollars
def quarter_value := 0.25
def nickel_value := 0.05
def penny_value := 0.01

-- Amount of money found in each type of coin
def money_from_quarters := num_quarters * quarter_value
def money_from_nickels := num_nickels * nickel_value
def money_from_pennies := num_pennies * penny_value

-- Total money from quarters, nickels, and pennies
def total_other_coins := money_from_quarters + money_from_nickels + money_from_pennies

-- Formula to calculate the number of dimes
def dime_value := 0.10
def num_dimes := (total_money - total_other_coins) / dime_value

-- Theorem to prove the number of dimes is 3
theorem harriett_found_3_dimes : num_dimes = 3 := by
  simp [num_quarters, quarter_value, num_nickels, nickel_value, num_pennies, penny_value, total_money, dime_value, money_from_quarters, money_from_nickels, money_from_pennies, total_other_coins, num_dimes]
  sorry

end harriett_found_3_dimes_l490_490867


namespace sin_alpha_plus_7pi_over_6_l490_490545

variable (α : Real)

theorem sin_alpha_plus_7pi_over_6 (h : sin α + cos (α - π / 6) = sqrt 3 / 3) : 
  sin (α + 7 * π / 6) = -(1 / 3) :=
by
  sorry

end sin_alpha_plus_7pi_over_6_l490_490545


namespace find_treasure_l490_490659

theorem find_treasure :
  ∃ i : ℕ, i = 2 ∧
  (i = 2 → (((1 ∨ 4) ≡ 0) ∧ ((1 → 2) ≡ 0) ∧ ((3 ∨ 5) ≡ 0) ∧ (¬((i = 4) ∧ ¬(¬(1 ∨ (4 ∧ 5)))))
  ∧ ¬((2 ∨ 3) ∧ ¬((¬ (i = 4 ∧ 5) ≡ 0) ∧ ((¬ (i = 5 ∧ 4)) ≡ ¬(i = 4 ∧ 3))))) ∧
    (2 = 1 ↔ true) ∧ (3 = 2 ↔ true) ∧ (5 = 4 ↔ ¬ true) →
  ∃ f : ℕ → ℕ, f 1 = 2 ∧ f 2 = 2 ∧ f 3 = 2 ∧ f 4 = 4 ∧ f 5 ≠ 5) :=
by
  sorry

end find_treasure_l490_490659


namespace no_prime_p_satisfies_l490_490504

-- Define conditions based on the problem
def convert_from_base (p : ℕ) :=
  2 * p^3 + p + 7 + 5 * p^2 + 4 + 2 * p^2 + p + 7 + 2 * p^2 + 3 + p + 4 = 
  2 * p^2 + 4 * p + 5 + 4 * p^2 + p + 5 + 5 * p^2 + 3 * p + 1

-- Define the equation we need to prove has no prime solutions
def eq_to_solve (p : ℕ) :=
  2 * p^3 - 5 * p + 14 = 0

-- Statement: No prime value of p satisfies the equation
theorem no_prime_p_satisfies : ∀ p, prime p → eq_to_solve p → false :=
begin
  sorry
end

end no_prime_p_satisfies_l490_490504


namespace quadratic_function_solution_l490_490179

noncomputable def f (x : ℝ) : ℝ := x^2 + 1798 * x + 251

theorem quadratic_function_solution (x : ℝ) :
  (f(f(x) + x)) / (f(x)) = x^2 + 1800 * x + 2050 := by sorry

end quadratic_function_solution_l490_490179


namespace sarah_saves_5_dollars_l490_490907

noncomputable def price_per_pair : ℕ := 40

noncomputable def promotion_A_price (n : ℕ) : ℕ :=
if n % 2 = 0 then price_per_pair * n / 2 else price_per_pair

noncomputable def promotion_B_price (n : ℕ) : ℕ :=
if n % 2 = 0 then price_per_pair * n - (15 * (n / 2)) else price_per_pair

noncomputable def total_price_promotion_A : ℕ :=
price_per_pair + (price_per_pair / 2)

noncomputable def total_price_promotion_B : ℕ :=
price_per_pair + (price_per_pair - 15)

theorem sarah_saves_5_dollars : total_price_promotion_B - total_price_promotion_A = 5 :=
by
  rw [total_price_promotion_B, total_price_promotion_A]
  norm_num
  sorry

end sarah_saves_5_dollars_l490_490907


namespace sum_of_roots_Q_l490_490302

open Complex

noncomputable def Q (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 6) : Polynomial ℝ :=
let p₁ := polynomial.C (cos (2 * θ) + sin (2 * θ) * I) in
let p₂ := polynomial.C (cos (2 * θ) - sin (2 * θ) * I) in
let p₃ := polynomial.C (sin (2 * θ) + cos (2 * θ) * I) in
let p₄ := polynomial.C (sin (2 * θ) - cos (2 * θ) * I) in
(X - p₁) * (X - p₂) * (X - p₃) * (X - p₄) * (X - 1) * (X + 1) * (X^2 + polynomial.C 0)

theorem sum_of_roots_Q (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 6) : 
  (sum_roots (Q θ hθ) = 2 * sqrt 2) :=
sorry

end sum_of_roots_Q_l490_490302


namespace alternating_sum_binomials_100_alternating_sum_binomials_99_l490_490070

theorem alternating_sum_binomials_100 : 
  (∑ k in finset.range 51, (if k % 2 = 0 then 1 else -1) * nat.choose 100 (2 * k)) = -2^50 :=
by
  sorry

theorem alternating_sum_binomials_99 :
  (∑ k in finset.range 50, (if k % 2 = 1 then 1 else -1) * nat.choose 99 (2 * k + 1)) = 2^49 :=
by
  sorry

end alternating_sum_binomials_100_alternating_sum_binomials_99_l490_490070


namespace concurrency_l490_490558

-- Define the geometry
variable {K : Type*} [Field K]
variable {P : Type*} [MetricSpace K P]

-- Define initial conditions on the triangle ABC
variables (A B C H M D E F X Y : P)
variables [IsMetrizableSpace P] (circum_circle : Circle P)
variables [Altitude A B C H] [Median A B C M] [MeetLine OH AM D]
variables [Intersect AB CD E] [Intersect AC BD F]
variables [Intersect EH circum_circle X] [Intersect FH circum_circle Y]

-- Define the result we want to prove
theorem concurrency : Concur BY CX AH := sorry

end concurrency_l490_490558


namespace whale_sixth_hour_consumption_l490_490883

-- Definitions from conditions
def first_hour_consumption (x : ℕ) : ℕ := x
def subsequent_hour_consumption (x : ℕ) (h : ℕ) : ℕ := x + 4 * h
def total_consumption (x : ℕ) : ℕ := 9 / 2 * (2 * x + 32)

-- The statement to be proved
theorem whale_sixth_hour_consumption (x : ℕ) (h1 : first_hour_consumption x) (h2 : ∑ i in finset.range 9, subsequent_hour_consumption x i = 450) :
  subsequent_hour_consumption x 5 = 54 :=
sorry

end whale_sixth_hour_consumption_l490_490883


namespace solution_set_of_f_inequality_l490_490013

variable {f : ℝ → ℝ}
variable (h1 : f 1 = 1)
variable (h2 : ∀ x, f' x < 1/2)

theorem solution_set_of_f_inequality :
  {x : ℝ | f (x^2) < x^2 / 2 + 1 / 2} = {x : ℝ | x < -1 ∨ 1 < x} :=
sorry

end solution_set_of_f_inequality_l490_490013


namespace sum_faces_edges_vertices_of_octagonal_pyramid_l490_490180

-- We define an octagonal pyramid with the given geometric properties.
structure OctagonalPyramid :=
  (base_vertices : ℕ) -- the number of vertices of the base
  (base_edges : ℕ)    -- the number of edges of the base
  (apex : ℕ)          -- the single apex of the pyramid
  (faces : ℕ)         -- the total number of faces: base face + triangular faces
  (edges : ℕ)         -- the total number of edges
  (vertices : ℕ)      -- the total number of vertices

-- Now we instantiate the structure based on the conditions.
def octagonalPyramid : OctagonalPyramid :=
  { base_vertices := 8,
    base_edges := 8,
    apex := 1,
    faces := 9,
    edges := 16,
    vertices := 9 }

-- We prove that the total number of faces, edges, and vertices sum to 34.
theorem sum_faces_edges_vertices_of_octagonal_pyramid : 
  (octagonalPyramid.faces + octagonalPyramid.edges + octagonalPyramid.vertices = 34) :=
by
  -- The proof steps are omitted as per instruction.
  sorry

end sum_faces_edges_vertices_of_octagonal_pyramid_l490_490180


namespace train_cross_time_l490_490603

/-- Define train and bridge parameters -/
def train_length : ℝ := 110
def bridge_length : ℝ := 134
def train_speed_kmh : ℝ := 72

/-- Conversion from km/h to m/s -/
def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)

/-- Total distance to cross -/
def total_distance : ℝ := train_length + bridge_length

/-- Time calculation -/
def time_to_cross : ℝ := total_distance / train_speed_ms

/-- Theorem stating the time taken by the train to cross the bridge -/
theorem train_cross_time : time_to_cross = 12.2 := by
  sorry

end train_cross_time_l490_490603


namespace sum_of_solutions_l490_490801

theorem sum_of_solutions (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) : 
  -b / a = 9 :=
by
  -- The proof is omitted here (hence the 'sorry')
  sorry

end sum_of_solutions_l490_490801


namespace sum_of_roots_quadratic_eq_l490_490839

theorem sum_of_roots_quadratic_eq :
  (∑ x in Finset.filter (λ x, x^2 = 9 * x - 20) (Finset.range 100), x) = 9 :=
begin
  sorry
end

end sum_of_roots_quadratic_eq_l490_490839


namespace hyperbola_eccentricity_l490_490222

theorem hyperbola_eccentricity (m : ℝ) (h : m > 0) : 
  let F := (3, 0) in 
  let equation_hyperbola := 3 * x^2 - m * y^2 = 3 * m in 
  let eccentricity := c / a in 
  eccentricity = √6 / 2 :=
by
  sorry

end hyperbola_eccentricity_l490_490222


namespace tenth_finger_will_be_2_l490_490045

variable f : Nat → Nat

-- Conditions based on the problem description
axiom f_2 : f 2 = 1
axiom f_1 : f 1 = 8
axiom f_8 : f 8 = 7
axiom f_7 : f 7 = 2

-- Definition of string of applications
def sequence (n : Nat) : Nat :=
  match n with
  | 1 => 2
  | 2 => f 2
  | 3 => f (f 2)
  | 4 => f (f (f 2))
  | 5 => f (f (f (f 2)))
  | 6 => f (f (f (f (f 2))))
  | 7 => f (f (f (f (f (f 2)))))
  | 8 => f (f (f (f (f (f (f 2))))))
  | 9 => f (f (f (f (f (f (f (f 2)))))))
  | 10 => f (f (f (f (f (f (f (f (f 2))))))))
  | _ => f (f (f (f (f (f (f (f (f (f 2)))))))))

theorem tenth_finger_will_be_2 : sequence 10 = 2 := by
  -- Proof would go here
  sorry

end tenth_finger_will_be_2_l490_490045


namespace best_approximation_log9_49_l490_490409

theorem best_approximation_log9_49 :
  (log 2 3 ≈ 1.585 ∧ log 2 7 ≈ 2.807) → 
  (∃ x : ℚ, (log 9 49 ≈ x) ∧
    (x = 10/7 ∨ x = 11/7 ∨ x = 12/7 ∨ x = 13/7 ∨ x = 14/7) ∧ 
   best_approx x (13/7)) :=
begin
  intros h,
  sorry
end

end best_approximation_log9_49_l490_490409


namespace simple_interest_calculation_l490_490110

theorem simple_interest_calculation :
  let P := 133875
  let R := 1
  let T := 3
  (P * R * T) / 100 = 4016.25 :=
by simp; norm_num

end simple_interest_calculation_l490_490110


namespace vector_dot_product_l490_490656

variable {A B C : Point}
variable {a c : ℝ}

-- Define area and angle conditions
def area_condition : ℝ := 2 * real.sqrt 3
def angle_condition: RealAngle := real.pi / 3

-- Intermediate results
def side_product (area : ℝ) (sinB : ℝ): ℝ := 4 * area / sinB

-- Main theorem
theorem vector_dot_product
  (h_area : (1/2) * a * c * sin angle_condition = area_condition)
  (h_cosB : real.cos angle_condition = 1/2 ) :
  (|a| * |c| * (- real.cos angle_condition) = -4 : ℝ) :=
  by sorry

end vector_dot_product_l490_490656


namespace problem_statement_l490_490367

noncomputable def f : ℝ → ℝ := sorry

variable {a b : ℝ}

theorem problem_statement (h1 : ∀ x ∈ Ioo (0 : ℝ) +∞, f'' x < f'' (-x))
                         (h2 : a ≠ 0)
                         (h3 : b ≠ 0)
                         (h4 : f a - f b > f (-b) - f (-a)) :
                         a^2 < b^2 := 
sorry

end problem_statement_l490_490367


namespace main_world_population_transition_l490_490962

noncomputable def world_population_reproduction_transition
  (developing_majority : Prop)
  (developing_transition : Prop)
  (developed_small : Prop)
  (developed_modern : Prop)
  (minor_impact : Prop) : Prop :=
  developing_majority ∧ developing_transition ∧ developed_small ∧ developed_modern ∧ minor_impact →
  "Traditional" = "Traditional" ∧ "Modern" = "Modern" →
  "Traditional" = "Traditional"

theorem main_world_population_transition
  (developing_majority : Prop)
  (developing_transition : Prop)
  (developed_small : Prop)
  (developed_modern : Prop)
  (minor_impact : Prop) :
  developing_majority ∧ developing_transition ∧ developed_small ∧ developed_modern ∧ minor_impact →
  "Traditional" = "Traditional" ∧ "Modern" = "Modern" →
  "Traditional" = "Traditional" :=
by
  sorry

end main_world_population_transition_l490_490962


namespace coefficient_binomial_expansion_l490_490780

theorem coefficient_binomial_expansion :
  let f (x y : ℚ) := (4 / 3 * x - 2 / 7 * y) ^ 8 
  ∃ c : ℚ, c * x^3 * y^5 = (coeff (monomial 3 5) (f x y)) ∧ c = -114688 / 45927 :=
begin
  sorry
end

end coefficient_binomial_expansion_l490_490780


namespace technicians_count_l490_490641

theorem technicians_count (avg_all : ℕ) (avg_tech : ℕ) (avg_other : ℕ) (total_workers : ℕ)
  (h1 : avg_all = 750) (h2 : avg_tech = 900) (h3 : avg_other = 700) (h4 : total_workers = 20) :
  ∃ T O : ℕ, (T + O = total_workers) ∧ ((T * avg_tech + O * avg_other) = total_workers * avg_all) ∧ (T = 5) :=
by
  sorry

end technicians_count_l490_490641


namespace intervals_of_monotonic_decrease_range_of_f_on_interval_minimum_value_of_phi_l490_490232

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x * sin (x + π / 2) + cos² x - 1 / 2

theorem intervals_of_monotonic_decrease :
  ∀ k : ℤ, 
  {x : ℝ | k * π + π / 6 ≤ x ∧ x ≤ k * π + 2 * π / 3} ⊆
  {x : ℝ | is_monotonic_decreasing_on f x} := 
sorry

theorem range_of_f_on_interval :
  ∀ x : ℝ, (0 ≤ x ∧ x ≤ π / 2) → 
  -1 / 2 ≤ f x ∧ f x ≤ 1 := 
sorry

noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := f (x / 2 + φ)

theorem minimum_value_of_phi :
  ∀ φ : ℝ, 
  (∃ k : ℤ, φ = k * π - π / 6 ∧ φ > 0) → 
  φ = 5 * π / 6 := 
sorry

end intervals_of_monotonic_decrease_range_of_f_on_interval_minimum_value_of_phi_l490_490232


namespace reflection_across_y_axis_l490_490281

-- Define original point P in Cartesian coordinate system
def P := (3, -5 : ℚ)

-- Define the transformation: reflection across the y-axis
def reflect_y (point : ℚ × ℚ) : ℚ × ℚ :=
  (-point.1, point.2)

-- Prove that the coordinates of point P (3, -5) with respect to the y-axis are (-3, -5)
theorem reflection_across_y_axis : reflect_y P = (-3, -5 : ℚ) :=
by
  -- use the definition of reflect_y and compute it for the point P
  sorry

end reflection_across_y_axis_l490_490281


namespace treasure_in_box_2_l490_490657

open Classical

structure Chest (number : Nat) :=
  (material : String)
  (statement : String)

def chest1 := Chest.mk 1 "cedar" "The treasure is in me or in the 4th chest."
def chest2 := Chest.mk 2 "sandalwood" "The treasure is in the chest to the left of me."
def chest3 := Chest.mk 3 "sandalwood" "The treasure is in me or in the chest at the far right."
def chest4 := Chest.mk 4 "cedar" "There is no treasure in the chests to the left of me."
def chest5 := Chest.mk 5 "cedar" "All the inscriptions on other chests are false."

def chests := [chest1, chest2, chest3, chest4, chest5]

def is_true (chest : Chest) (actual : Nat) : Bool :=
  match chest.number with
  | 1 => actual = 1 ∨ actual = 4
  | 2 => actual = 1
  | 3 => actual = 3 ∨ actual = 5
  | 4 => ∀ i : Nat, (i < 4 → i ≠ actual)
  | 5 => ∀ i : Nat, (i ≠ 5 → ¬ is_true (chests.getD (i - 1) chest5) actual)

theorem treasure_in_box_2 :
  ∃ t, 
    (∀ chest, chest ∈ chests → 
      (if chest.material = "cedar" ∨ chest.material = "sandalwood" then
        is_true chest t = false 
      else true))  -- Number of false statements on cedar and sandalwood chests are equal
  ∧ 
    t = 2 :=
by
  sorry

end treasure_in_box_2_l490_490657


namespace area_of_rectangle_l490_490745

noncomputable def rectangle_area (length : ℕ) (side_square : ℕ) : ℕ :=
  let radius_circle := side_square in
  let breadth_rectangle := (3 * radius_circle) / 5 in
  length * breadth_rectangle

theorem area_of_rectangle :
  ∀ (side_square length : ℕ),
    side_square * side_square = 2025 →
    length = 10 →
    rectangle_area length side_square = 270 :=
by
  intros side_square length h1 h2
  -- proof omitted
  sorry

end area_of_rectangle_l490_490745


namespace hexagon_area_proof_l490_490052

-- Define the conditions
def base := 1
def height := 3
def rect_length := 6
def rect_width := 4

-- Define the key areas based on the conditions
def triangle_area := (1/2) * base * height
def total_triangle_area := 4 * triangle_area
def rect_area := rect_length * rect_width
def hexagon_area := rect_area - total_triangle_area

-- Formalize the theorem
theorem hexagon_area_proof : hexagon_area = 18 := by
  sorry

end hexagon_area_proof_l490_490052


namespace sum_of_solutions_eq_9_l490_490805

theorem sum_of_solutions_eq_9 (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20) :
  let (sum_roots : ℝ) := -b / a in 
  sum_roots = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490805


namespace sequence_sum_l490_490599

/-- Let {a_n} be a sequence of real numbers such that the sum of its first n terms is S_n.
Given that a_1 = 1 and for n ≥ 2, a_n = (2 * S_n^2) / (2 * S_n - 1).
Prove that S_2016 = 1 / 4031. -/
theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : ∀ n : ℕ, n ≥ 2 → a n = (2 * (S n) ^ 2) / (2 * (S n) - 1))
  (h₂ : ∀ n : ℕ, S n = (∑ i in finset.range (n + 1), a i)) :
  S 2016 = 1 / 4031 := sorry

end sequence_sum_l490_490599


namespace number_of_tangents_l490_490374

noncomputable def circle₁ : { x : ℝ × ℝ // pow x.1 2 + pow x.2 2 + 2 * x.1 + 4 * x.2 + 1 = 0 } := sorry
noncomputable def circle₂ : { x : ℝ × ℝ // pow x.1 2 + pow x.2 2 - 4 * x.1 - 4 * x.2 - 1 = 0 } := sorry

theorem number_of_tangents :
  let C₁ := { x : ℝ × ℝ // pow x.1 2 + pow x.2 2 + 2 * x.1 + 4 * x.2 + 1 = 0 };
  let C₂ := { x : ℝ × ℝ // pow x.1 2 + pow x.2 2 - 4 * x.1 - 4 * x.2 - 1 = 0 };
  ∃ t, (number_of_common_tangents C₁ C₂ = t) ∧ (t = 3) :=
sorry

end number_of_tangents_l490_490374


namespace sum_of_solutions_l490_490796

theorem sum_of_solutions (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) : 
  -b / a = 9 :=
by
  -- The proof is omitted here (hence the 'sorry')
  sorry

end sum_of_solutions_l490_490796


namespace math_problem_l490_490530

theorem math_problem (a b c d e : ℤ) (x : ℤ) (hx : x > 196)
  (h1 : a + b = 183) (h2 : a + c = 186) (h3 : d + e = x) (h4 : c + e = 196)
  (h5 : 183 < 186) (h6 : 186 < 187) (h7 : 187 < 190) (h8 : 190 < 191) (h9 : 191 < 192)
  (h10 : 192 < 193) (h11 : 193 < 194) (h12 : 194 < 196) (h13 : 196 < x) :
  (a = 91 ∧ b = 92 ∧ c = 95 ∧ d = 99 ∧ e = 101 ∧ x = 200) ∧ (∃ y, y = 10 * x + 3 ∧ y = 2003) :=
by
  sorry

end math_problem_l490_490530


namespace small_sphere_radius_l490_490029

theorem small_sphere_radius (r : ℚ) (A B C D : Point) (s₁ s₂ : Sphere) (s₃ s₄ s₅ s₆ : Sphere) : 
  s₁.radius = 2 ∧ s₂.radius = 2 ∧ s₃.radius = 3 ∧ s₄.radius = 3 ∧ 
  tangent s₁ s₂ ∧ tangent s₁ s₃ ∧ tangent s₁ s₄ ∧ 
  tangent s₂ s₃ ∧ tangent s₂ s₄ ∧ tangent s₃ s₄ ∧
  tangent s₅ s₁ ∧ tangent s₅ s₂ ∧ tangent s₅ s₃ ∧ tangent s₅ s₄ ->
  s₅.radius = 6 / 11 :=
by
  sorry

end small_sphere_radius_l490_490029


namespace hexagon_side_length_l490_490376

-- Define the conditions for the side length of a hexagon where the area equals the perimeter
theorem hexagon_side_length (s : ℝ) (h1 : (3 * Real.sqrt 3 / 2) * s^2 = 6 * s) :
  s = 4 * Real.sqrt 3 / 3 :=
sorry

end hexagon_side_length_l490_490376


namespace sum_of_possible_values_of_m_l490_490113

noncomputable def is_intersection_midpoint (m : ℝ) : Prop :=
  let midpoint_x := (6*m + 2) / 2 in
  let midpoint_y := 1 in
  midpoint_y = m * midpoint_x

noncomputable def is_divided_equally (m : ℝ) : Prop :=
  let line_bc := λ x : ℝ, -2 / (6*m - 2) * (x - 2) in
  ∃ x : ℝ, let y := line_bc x in y = m * x ∧ is_intersection_midpoint m

theorem sum_of_possible_values_of_m : (∑ m in {m : ℝ | is_divided_equally m}, m) = -1 / 3 :=
begin
  -- Proof goes here
  sorry
end

end sum_of_possible_values_of_m_l490_490113


namespace completion_time_workshop_3_l490_490002

-- Define the times for workshops
def time_in_workshop_3 : ℝ := 8
def time_in_workshop_1 : ℝ := time_in_workshop_3 + 10
def time_in_workshop_2 : ℝ := (time_in_workshop_3 + 10) - 3.6

-- Define the combined work equation
def combined_work_eq := (1 / time_in_workshop_1) + (1 / time_in_workshop_2) = (1 / time_in_workshop_3)

-- Final theorem statement
theorem completion_time_workshop_3 (h : combined_work_eq) : time_in_workshop_3 - 7 = 1 :=
by
  sorry

end completion_time_workshop_3_l490_490002


namespace exists_matrix_with_conditions_iff_n_odd_l490_490192

theorem exists_matrix_with_conditions_iff_n_odd (n : ℕ) (A : matrix (fin n) (fin n) ℤ) :
  (∀ i : fin n, (matrix.dot_product (A i) (A i)) % 2 = 0) ∧
  (∀ i j : fin n, i ≠ j → (matrix.dot_product (A i) (A j)) % 2 = 1) ↔ 
  odd n := 
sorry

end exists_matrix_with_conditions_iff_n_odd_l490_490192


namespace inconsistency_in_survey_data_l490_490939

open Set

noncomputable def number_of_households : ℕ := 1000
noncomputable def households_using_gas : ℕ := 265
noncomputable def households_using_oil : ℕ := 51
noncomputable def households_using_coal : ℕ := 803

noncomputable def households_using_gas_or_oil : ℕ := 287
noncomputable def households_using_oil_or_coal : ℕ := 843
noncomputable def households_using_gas_or_coal : ℕ := 919

theorem inconsistency_in_survey_data :
  let A₁ := households_using_gas,
      A₂ := households_using_oil,
      A₃ := households_using_coal,
      total := number_of_households,
      A₁U_A₂ := households_using_gas_or_oil,
      A₂U_A₃ := households_using_oil_or_coal,
      A₁U_A₃ := households_using_gas_or_coal,
      A₁I_A₂ := A₁ + A₂ - A₁U_A₂,
      A₂I_A₃ := A₂ + A₃ - A₂U_A₃,
      A₁I_A₃ := A₃ + A₁ - A₁U_A₃,
      A₁I_A₂I_A₃ := total - (A₁ + A₂ + A₃ - A₁I_A₂ - A₂I_A₃ - A₁I_A₃)
in A₁I_A₂I_A₃ = 70 ∧ 70 > A₂ :=
sorry

end inconsistency_in_survey_data_l490_490939


namespace problem_I_solution_problem_II_solution_l490_490238

variable {x a b : ℝ}

-- Problem I
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x + 3

-- Condition I: Solution set of f(x) ≤ -3 is [b, 3]
def eqn (b a : ℝ) : Prop := b + 3 = a ∧ 3 * b = 6

theorem problem_I_solution (b a : ℝ) (h : eqn b a) : b = 2 ∧ a = 5 := 
by
  sorry

-- Problem II
variable {x : ℝ}

-- Condition II: When x ∈ [1/2, +∞), inequality f(x) ≥ 1 - x² always true
def g (x : ℝ) : ℝ := 2 * x + 2 / x

theorem problem_II_solution (a : ℝ) (h1 : ∀ x, x ≥ 1/2 → f(x, a) ≥ 1 - x^2) : a ≤ 4 :=
by
  sorry

end problem_I_solution_problem_II_solution_l490_490238


namespace solution_exists_l490_490985

theorem solution_exists : 
  ∃ (x y z : ℝ), 
    (2 * x - 3 * y + z = -4) ∧ 
    (5 * x - 2 * y - 3 * z = 7) ∧ 
    (x + y - 4 * z = -6) :=
begin
  sorry
end

end solution_exists_l490_490985


namespace hardest_working_person_hours_difference_l490_490005

theorem hardest_working_person_hours_difference 
  (r1 r2 r3 r4 r5 : ℝ) (total_hours : ℝ)
  (hr1 : r1 = 2.5) (hr2 : r2 = 3.5) (hr3 : r3 = 4.5) 
  (hr4 : r4 = 5.5) (hr5 : r5 = 6.5) (htotal : total_hours = 550) :
  let total_ratio := r1 + r2 + r3 + r4 + r5 in
  let hours_per_unit := total_hours / total_ratio in
  let hardest_working_hours := r5 * hours_per_unit in
  let least_working_hours := r1 * hours_per_unit in
  let difference := hardest_working_hours - least_working_hours in
  difference ≈ 97.7778 :=
begin
  sorry
end

end hardest_working_person_hours_difference_l490_490005


namespace product_of_chords_l490_490953

open Complex

theorem product_of_chords 
  (r : ℝ)
  (n : ℕ)
  (A B : ℂ)
  (P : ℕ → ℂ)
  (h_radius : r = 3)
  (h_division : n = 8)
  (h_AB : A = 3 ∧ B = -3)
  (h_P : ∀ k, k < n → P k = r * exp (2 * π * I * (k : ℂ) / (2 * n))) :
  (∏ k in finset.range n, dist A (P k)) * (∏ k in finset.range n, dist B (P k)) = 63700992 := 
sorry

end product_of_chords_l490_490953


namespace difference_max_min_trig_l490_490360

theorem difference_max_min_trig:
  let f (x : ℝ) := 2 * sin (π * x / 6 - π / 3)
  let max_val := 2
  let min_val := -sqrt 3
  (0 ≤ x ∧ x ≤ 9) → 
  (max_val - min_val = 2 + sqrt 3) :=
by
  sorry

end difference_max_min_trig_l490_490360


namespace correct_if_statement_l490_490430

-- Definitions based on the conditions
def hasEndIf (stmt : String) : Prop := stmt.ends_with "END IF"
def canOmitElse (stmt : String) : Prop := ¬stmt.contains "ELSE"

-- The main statement we want to prove
theorem correct_if_statement (stmt : String) : 
  (hasEndIf stmt) ∧ (canOmitElse stmt) → stmt = "C" :=
by
  sorry

end correct_if_statement_l490_490430


namespace smallest_integer_no_inverse_mod_77_66_l490_490426

theorem smallest_integer_no_inverse_mod_77_66 :
  ∃ a : ℕ, 0 < a ∧ a = 11 ∧ gcd a 77 > 1 ∧ gcd a 66 > 1 :=
by
  sorry

end smallest_integer_no_inverse_mod_77_66_l490_490426


namespace sufficient_drivers_and_completion_time_l490_490926

noncomputable def one_way_trip_minutes : ℕ := 2 * 60 + 40
noncomputable def round_trip_minutes : ℕ := 2 * one_way_trip_minutes
noncomputable def rest_period_minutes : ℕ := 60
noncomputable def twelve_forty_pm : ℕ := 12 * 60 + 40 -- in minutes from midnight
noncomputable def one_forty_pm : ℕ := twelve_forty_pm + rest_period_minutes
noncomputable def thirteen_five_pm : ℕ := 13 * 60 + 5 -- 1:05 PM
noncomputable def sixteen_ten_pm : ℕ := 16 * 60 + 10 -- 4:10 PM
noncomputable def sixteen_pm : ℕ := 16 * 60 -- 4:00 PM
noncomputable def seventeen_thirty_pm : ℕ := 17 * 60 + 30 -- 5:30 PM
noncomputable def twenty_one_thirty_pm : ℕ := sixteen_ten_pm + round_trip_minutes -- 9:30 PM (21:30)

theorem sufficient_drivers_and_completion_time :
  4 = 4 ∧ twenty_one_thirty_pm = 21 * 60 + 30 := by
  sorry 

end sufficient_drivers_and_completion_time_l490_490926


namespace bus_network_possible_l490_490662

-- Define the setup for a bus network with routes and stops
variable (BusStop : Type) (Route : Type) [Fintype BusStop] [Fintype Route]

-- Assume a relation indicating which routes connect bus stops
variable (connects : Route → BusStop → BusStop → Prop)

-- Assume 1310 routes and sufficient bus stops
axiom num_routes : Fintype.card Route = 1310

-- Define the conditions of connectivity
def connected_after_one_closure (bs : BusStop) : Prop :=
  ∀ r : Route, (∀ b1 b2 : BusStop, b1 ≠ b2 → connects r b1 b2 → 
    ∃ r' : Route, r ≠ r' ∧ connects r' b1 b2)

def disconnected_after_two_closures (bs : BusStop) : Prop :=
  ∃ r1 r2 : Route, ∃ b1 b2 : BusStop, r1 ≠ r2 ∧ b1 ≠ b2 ∧
  ¬(∃ r' : Route, r' ≠ r1 ∧ r' ≠ r2 ∧ (connects r' b1 b2 ∨ connects r' b2 b1))

-- The main theorem to prove
theorem bus_network_possible (bs : BusStop) :
  (∀ bs : BusStop, connected_after_one_closure bs) ∧ (∀ bs : BusStop, disconnected_after_two_closures bs) := 
sorry

end bus_network_possible_l490_490662


namespace g_half_equals_four_l490_490590

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a^(2 - x) - 3 / 4

def A : ℝ × ℝ :=
  (2, 1 / 4)

noncomputable def g (α : ℝ) (x : ℝ) : ℝ :=
  x^α

theorem g_half_equals_four (a : ℝ) (α : ℝ)
  (ha : a > 0) (ha' : a ≠ 1) (hA1 : f a 2 = 1/4) (hA2 : g α 2 = 1/4) :
  g α (1 / 2) = 4 :=
by
  sorry

end g_half_equals_four_l490_490590


namespace roger_expenses_fraction_l490_490344

theorem roger_expenses_fraction {B t s n : ℝ} (h1 : t = 0.25 * (B - s))
  (h2 : s = 0.10 * (B - t)) (h3 : n = 5) :
  (t + s + n) / B = 0.41 :=
sorry

end roger_expenses_fraction_l490_490344


namespace sum_of_roots_l490_490994

theorem sum_of_roots (x y : ℝ) (h : ∀ z, z^2 + 2023 * z - 2024 = 0 → z = x ∨ z = y) : x + y = -2023 := 
by
  sorry

end sum_of_roots_l490_490994


namespace correlate_height_weight_l490_490879

-- Define the problems as types
def heightWeightCorrelated : Prop := true
def distanceTimeConstantSpeed : Prop := true
def heightVisionCorrelated : Prop := false
def volumeEdgeLengthCorrelated : Prop := true

-- Define the equivalence for the problem
def correlated : Prop := heightWeightCorrelated

-- Now state that correlated == heightWeightCorrelated
theorem correlate_height_weight : correlated = heightWeightCorrelated :=
by sorry

end correlate_height_weight_l490_490879


namespace integral_solution_l490_490067

noncomputable def definite_integral : ℝ :=
  ∫ x in (-2 : ℝ)..(0 : ℝ), (x + 2)^2 * (Real.cos (3 * x))

theorem integral_solution :
  definite_integral = (12 - 2 * Real.sin 6) / 27 :=
sorry

end integral_solution_l490_490067


namespace cupcakes_per_day_needed_l490_490127

-- Define the given conditions
def total_goal_cupcakes : ℕ := 96
def cupcakes_to_bonnie : ℕ := 24
def days : ℕ := 2

-- State the problem as a Lean theorem
theorem cupcakes_per_day_needed (goal : ℕ) (bonnie : ℕ) (days : ℕ) : ℕ :=
  (goal + bonnie) / days

-- Verify the conditions as true
example : cupcakes_per_day_needed total_goal_cupcakes cupcakes_to_bonnie days = 60 :=
by
  -- The calculation can be checked against the example conditions
  calc
    cupcakes_per_day_needed 96 24 2 = (96 + 24) / 2 : rfl
    ... = 120 / 2 : by norm_num
    ... = 60 : by norm_num

end cupcakes_per_day_needed_l490_490127


namespace binary_to_decimal_l490_490153

-- Define the binary number 10011_2
def binary_10011 : ℕ := bit0 (bit1 (bit1 (bit0 (bit1 0))))

-- Define the expected decimal value
def decimal_19 : ℕ := 19

-- State the theorem to convert binary 10011 to decimal
theorem binary_to_decimal :
  binary_10011 = decimal_19 :=
sorry

end binary_to_decimal_l490_490153


namespace probability_fourth_six_equals_2187_div_982_final_sum_of_p_q_equals_3169_l490_490952

def biased_die_probability := 3 / 4
def other_faces_probability := 1 / 20
def fair_die_six_probability := 1 / 6
def fair_die_other_faces_probability := 1 / 6
def probability_of_first_three_sixes_fair := (1/6) ^ 3
def probability_of_first_three_sixes_biased := (3/4) ^ 3

theorem probability_fourth_six_equals_2187_div_982 :
  let initial_probability := 1 / 2 in
  let total_probability_three_sixes := initial_probability * ((1 / 216) + (27 / 64)) in
  let probability_fair_given_three_sixes := (1 / 2 * (1 / 216)) / total_probability_three_sixes in
  let probability_biased_given_three_sixes := (1 / 2 * (27 / 64)) / total_probability_three_sixes in
  let final_probability_fourth_six := (probability_fair_given_three_sixes * (1 / 6)) + (probability_biased_given_three_sixes * (3 / 4)) in
  final_probability_fourth_six = 2187 / 982 := sorry

theorem final_sum_of_p_q_equals_3169 :
  let p := 2187
  let q := 982
  p + q = 3169 := by norm_num

end probability_fourth_six_equals_2187_div_982_final_sum_of_p_q_equals_3169_l490_490952


namespace product_sum_roots_reciprocals_l490_490534

theorem product_sum_roots_reciprocals {a b : ℝ} (h1 : ∀ x : ℝ, x^3 + a * x^2 + b * x + 1 = 0 → 
  ∃ x1 x2 x3 : ℝ, x1 + x2 + x3 = -a ∧ x1 * x2 + x1 * x3 + x2 * x3 = b ∧ x1 * x2 * x3 = -1)
  : a * b :=
by
  obtain ⟨x1, x2, x3, ⟨h_sum_roots, h_sum_prod_roots, h_prod_roots⟩⟩ := h1 0 
  have h_sum_reciprocal := (x1 * x2 * x3)⁻¹ * (x1 * x2 + x1 * x3 + x2 * x3)
  have h_sum_recip_list := -b
  have h_sum_roots_list := -a
  have h_prod := h_sum_roots_list * h_sum_recip_list
  exact h_prod

end product_sum_roots_reciprocals_l490_490534


namespace probability_of_random_event_l490_490383

def random_event (A : Type) : Prop := A = True

def probability (P : Prop → ℝ) (A : Prop) : Prop :=
  0 ≤ P(A) ∧ P(A) ≤ 1

theorem probability_of_random_event (A : Prop) (P : Prop → ℝ) (h : random_event A) :
  probability P A :=
sorry

end probability_of_random_event_l490_490383


namespace gcd_90_250_l490_490503

theorem gcd_90_250 : Nat.gcd 90 250 = 10 := by
  sorry

end gcd_90_250_l490_490503


namespace value_of_m_minus_n_l490_490618

theorem value_of_m_minus_n (m n : ℝ) (h : (-3)^2 + m * (-3) + 3 * n = 0) : m - n = 3 :=
sorry

end value_of_m_minus_n_l490_490618


namespace Miranda_can_stuff_pillows_l490_490322

theorem Miranda_can_stuff_pillows:
  let pounds_per_pillow := 2 in
  let goose_feathers_per_pound := 300 in
  let duck_feathers_per_pound := 500 in
  let total_goose_feathers := 3600 in
  let total_duck_feathers := 4000 in
  let goose_feathers_in_pounds := total_goose_feathers / goose_feathers_per_pound in
  let duck_feathers_in_pounds := total_duck_feathers / duck_feathers_per_pound in
  let total_feathers_in_pounds := goose_feathers_in_pounds + duck_feathers_in_pounds in
  let pillows_stuffed := total_feathers_in_pounds / pounds_per_pillow in
  pillows_stuffed = 10 := by
  sorry

end Miranda_can_stuff_pillows_l490_490322


namespace std_dev_decreases_l490_490271

variable (n : ℕ) (μ s s₁ : ℝ)
variable (A₀ A A₁ B₀ B B₁ : ℝ)

-- Conditions from the problem:
-- Number of students
def num_students := 50
-- Initial average score
def initial_avg := 70
-- Recorded scores and their corrections
def A₀ := 50  -- initial incorrect recorded score for Student A
def A := 80   -- actual score for Student A
def B₀ := 100 -- initial incorrect recorded score for Student B
def B := 70   -- actual score for Student B

theorem std_dev_decreases (h_n : n = num_students)
  (h_μ : μ = initial_avg)
  (h_s : s > 0)
  (h_A₀ : A₀ = 50)
  (h_A : A_ = 80)
  (h_B₀ : B₀ = 100)
  (h_B : B = 70)
  (h_std_dev_corr : s₁ < s) :
  s > s₁ :=
sorry

end std_dev_decreases_l490_490271


namespace betty_red_beads_l490_490948

theorem betty_red_beads (r b : ℕ) (h_ratio : r / b = 3 / 2) (h_blue_beads : b = 20) : r = 30 :=
by
  sorry

end betty_red_beads_l490_490948


namespace quadrilateral_parallelogram_l490_490338

theorem quadrilateral_parallelogram 
  (A B C D F : Type) 
  [h1 : ∀ (α : Type), ∠ABC = α ∧ ∠CDA = α]
  [h2 : ∀ (G : Type), midpoint BD AC F] 
  : parallelogram ABCD :=
sorry

end quadrilateral_parallelogram_l490_490338


namespace question_1_question_2_l490_490237

noncomputable def f (x a : ℝ) := x * real.exp (x - 1) - a * x + 1

theorem question_1 :
  (∀ x, deriv (λ x, f x a) x = (x + 1) * real.exp (x - 1) - a) →
  ((2 + 1) * real.exp (2 - 1) - a = 3 * real.exp 1 - 2) →
  a = 2 ∧ (∀ x y, y = f 2 2 → (3 * real.exp 1 - 2) * (x - 2) + (2 * real.exp 1 - 3) = y ) :=
by
  sorry

theorem question_2 :
  (∀ x, f x 2 ≥ 0) :=
by
  sorry

end question_1_question_2_l490_490237


namespace frog_climb_time_l490_490909

-- Definitions related to the problem
def well_depth : ℕ := 12
def climb_per_cycle : ℕ := 3
def slip_per_cycle : ℕ := 1
def effective_climb_per_cycle : ℕ := climb_per_cycle - slip_per_cycle

-- Time taken for each activity
def time_to_climb : ℕ := 10 -- given as t
def time_to_slip : ℕ := time_to_climb / 3
def total_time_per_cycle : ℕ := time_to_climb + time_to_slip

-- Condition specifying the observed frog position at a certain time
def observed_time : ℕ := 17 -- minutes since 8:00
def observed_position : ℕ := 9 -- meters climbed since it's 3 meters from the top of the well (well_depth - 3)

-- The main theorem stating the total time taken to climb to the top of the well
theorem frog_climb_time : 
  ∃ (k : ℕ), k * effective_climb_per_cycle + climb_per_cycle = well_depth ∧ k * total_time_per_cycle + time_to_climb = 22 := 
sorry

end frog_climb_time_l490_490909


namespace fruit_salad_problem_l490_490118

theorem fruit_salad_problem (Alaya Angel Betty Charlie : ℕ) 
  (h1 : Alaya = 200) 
  (h2 : Angel = 2 * Alaya) 
  (h3 : Betty = 3 * Angel) 
  (h4 : Charlie = Betty - 50) : 
  Alaya + Angel + Betty + Charlie = 2950 ∧ 
  (Alaya + Angel + Betty + Charlie) / 4 = 737.5 :=
by
  sorry

end fruit_salad_problem_l490_490118


namespace triangle_area_solution_l490_490269

noncomputable def triangle_area_problem 
  (a b c : ℝ) (A B C : ℝ) (h1 : A = 3 * C)
  (h2 : c = 6)
  (h3 : (2 * a - c) * Real.cos B - b * Real.cos C = 0)
  : ℝ := (1 / 2) * a * c * Real.sin B

theorem triangle_area_solution 
  (a b c : ℝ) (A B C : ℝ) (h1 : A = 3 * C)
  (h2 : c = 6)
  (h3 : (2 * a - c) * Real.cos B - b * Real.cos C = 0)
  (ha : a = 12)
  (hb : b = 6 * Real.sin (π / 3))
  (hA : A = π / 2)
  (hB : B = π / 3)
  (hC : C = π / 6) 
  : triangle_area_problem a b c A B C h1 h2 h3 = 18 * Real.sqrt 3 := by
  sorry

end triangle_area_solution_l490_490269


namespace shifted_parabola_eq_l490_490364

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := -(x^2)

-- Define the transformation for shifting left 2 units
def shift_left (x : ℝ) : ℝ := x + 2

-- Define the transformation for shifting down 3 units
def shift_down (y : ℝ) : ℝ := y - 3

-- Define the new parabola equation after shifting
def new_parabola (x : ℝ) : ℝ := shift_down (original_parabola (shift_left x))

-- The theorem to be proven
theorem shifted_parabola_eq : new_parabola x = -(x + 2)^2 - 3 := by
  sorry

end shifted_parabola_eq_l490_490364


namespace g_is_odd_l490_490293

noncomputable def g (x : ℝ) : ℝ := (7^x - 1) / (7^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intros x
  sorry

end g_is_odd_l490_490293


namespace problem_solution_l490_490124

-- Define the sub-expressions
def expr_base := 4 - 9
def expr_exp := expr_base ^ 3
def expr_mul := 4 * expr_exp
def expr_add := 5 + expr_mul

-- State the theorem
theorem problem_solution : expr_add = -495 := by
  -- Proof to be filled in by the user
  sorry

end problem_solution_l490_490124


namespace symmetric_curve_equation_l490_490363

theorem symmetric_curve_equation (y x : ℝ) :
  (y^2 = 4 * x) → (y^2 = 16 - 4 * x) :=
sorry

end symmetric_curve_equation_l490_490363


namespace sum_of_solutions_l490_490799

theorem sum_of_solutions (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) : 
  -b / a = 9 :=
by
  -- The proof is omitted here (hence the 'sorry')
  sorry

end sum_of_solutions_l490_490799


namespace sum_of_roots_quadratic_eq_l490_490841

theorem sum_of_roots_quadratic_eq :
  (∑ x in Finset.filter (λ x, x^2 = 9 * x - 20) (Finset.range 100), x) = 9 :=
begin
  sorry
end

end sum_of_roots_quadratic_eq_l490_490841


namespace sum_of_solutions_eq_9_l490_490823

theorem sum_of_solutions_eq_9 :
  let roots := {x : ℝ | x^2 = 9 * x - 20}
  in ∑ x in roots, x = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490823


namespace february1_day_of_week_l490_490255

theorem february1_day_of_week :
  ∀ (d : ℕ), d = 4 → ((11 - d) % 7) = 3 → April_first_is : Prop :=
by sorry

end february1_day_of_week_l490_490255


namespace votes_ratio_l490_490866

theorem votes_ratio (V : ℝ) 
  (counted_fraction : ℝ := 2/9) 
  (favor_fraction : ℝ := 3/4) 
  (against_fraction_remaining : ℝ := 0.7857142857142856) :
  let counted := counted_fraction * V
  let favor_counted := favor_fraction * counted
  let remaining := V - counted
  let against_remaining := against_fraction_remaining * remaining
  let against_counted := (1 - favor_fraction) * counted
  let total_against := against_counted + against_remaining
  let total_favor := favor_counted
  (total_against / total_favor) = 4 :=
by
  sorry

end votes_ratio_l490_490866


namespace count_quadrilaterals_equals_l490_490604

-- Defining the required conditions
def is_convex_cyclic_quadrilateral (a b c d : ℕ) : Prop :=
  a + b + c + d = 36 ∧ a ≤ 18 ∧ b ≤ 18 ∧ c ≤ 18 ∧ d ≤ 18 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ b ≠ d ∧ a ≠ d ∧ (a + b > c ∧ b + c > a ∧ c + a > b ∧ a + c > d)

noncomputable def count_convex_cyclic_quadrilaterals : ℕ :=
  sorry  -- This will be replaced with the actual count derived from the solution.

-- The final theorem statement
theorem count_quadrilaterals_equals : count_convex_cyclic_quadrilaterals = (here provide the exact calculated value from the steps) :=
  sorry  -- Proof to be filled in by solution.


end count_quadrilaterals_equals_l490_490604


namespace sum_of_roots_quadratic_eq_l490_490843

theorem sum_of_roots_quadratic_eq :
  (∑ x in Finset.filter (λ x, x^2 = 9 * x - 20) (Finset.range 100), x) = 9 :=
begin
  sorry
end

end sum_of_roots_quadratic_eq_l490_490843


namespace equilateral_triangle_M_properties_l490_490537

-- Define the points involved
variables (A B C M P Q R : ℝ)
-- Define distances from M to the sides as given by perpendiculars
variables (d_AP d_BQ d_CR d_PB d_QC d_RA : ℝ)

-- Equilateral triangle assumption and perpendiculars from M to sides
def equilateral_triangle (A B C : ℝ) : Prop := sorry
def perpendicular_from_point (M P R : ℝ) (line : ℝ) : Prop := sorry

-- Problem statement encapsulating the given conditions and what needs to be proved:
theorem equilateral_triangle_M_properties
  (h_triangle: equilateral_triangle A B C)
  (h_perp_AP: perpendicular_from_point M P A B)
  (h_perp_BQ: perpendicular_from_point M Q B C)
  (h_perp_CR: perpendicular_from_point M R C A) :
  (d_AP^2 + d_BQ^2 + d_CR^2 = d_PB^2 + d_QC^2 + d_RA^2) ∧ 
  (d_AP + d_BQ + d_CR = d_PB + d_QC + d_RA) := sorry

end equilateral_triangle_M_properties_l490_490537


namespace number_of_ways_to_choose_six_with_consecutives_l490_490194

theorem number_of_ways_to_choose_six_with_consecutives :
  ∀ (s : Finset ℕ), (s.card = 6) → (∀ i ∈ s, i ∈ (Finset.range 50 \ {0})) →
  ∃ (t u : Finset ℕ), t ∪ u = s ∧ (∀ i j ∈ t, abs (i - j) = 1) ∧ (t.card ≥ 2) ∧
  (nat.choose 49 6 - nat.choose 44 6) :=
by
  sorry

end number_of_ways_to_choose_six_with_consecutives_l490_490194


namespace volume_tetrahedron_PXYZ_l490_490771

noncomputable def volume_of_tetrahedron_PXYZ (x y z : ℝ) : ℝ :=
  (1 / 6) * x * y * z

theorem volume_tetrahedron_PXYZ :
  ∃ (x y z : ℝ), (x^2 + y^2 = 49) ∧ (y^2 + z^2 = 64) ∧ (z^2 + x^2 = 81) ∧
  volume_of_tetrahedron_PXYZ (Real.sqrt x) (Real.sqrt y) (Real.sqrt z) = 4 * Real.sqrt 11 := 
by {
  sorry
}

end volume_tetrahedron_PXYZ_l490_490771


namespace polar_coordinates_point_M_parametric_equation_line_AM_l490_490217

-- Definitions based on conditions
def semicircle_point (θ : ℝ) : Prop := 0 ≤ θ ∧ θ ≤ π

def point_P (θ : ℝ) : Prop := semicircle_point θ ∧ (x = cos θ ∧ y = sin θ)

def point_A : (ℝ × ℝ) := (1, 0)

def point_O : (ℝ × ℝ) := (0, 0)

def ray_OP (θ : ℝ) : Prop := ∃ k : ℝ, k > 0 ∧ (x = k * cos θ ∧ y = k * sin θ)

-- Noncomputable because of trigonometric functions
noncomputable def point_M (θ : ℝ) : (ℝ × ℝ) := 
  let r := π / 3 in 
  (r * cos θ, r * sin θ)

-- Proof that answers the first question
theorem polar_coordinates_point_M : 
  (M : (ℝ × ℝ)) (hθ : semicircle_point θ) := 
  (M = (π / 3, π / 3)) :=
sorry

-- Definition for parametric equation of line AM based on given coordinates
noncomputable def parametric_eq_line_AM (t : ℝ) : (ℝ × ℝ) := 
  let M_x := π / 6 in
  let M_y := (sqrt 3 * π) / 6 in 
  let A_x := (point_A.fst : ℝ) in
  let A_y := (point_A.snd : ℝ) in 
  (A_x + (M_x - A_x) * t, M_y * t)

-- Proof that answers the second question
theorem parametric_equation_line_AM : 
  (t : ℝ) (M : (ℝ × ℝ)) (M_x := π / 6) (M_y := (sqrt 3 * π) / 6) :=
  (parametric_eq_line_AM t = 
    (1 + ((π / 6 - 1) * t), (sqrt 3 * π / 6) * t)) :=
sorry

end polar_coordinates_point_M_parametric_equation_line_AM_l490_490217


namespace landscape_length_l490_490017

theorem landscape_length (b length : ℕ) (A_playground : ℕ) (h1 : length = 4 * b) (h2 : A_playground = 1200) (h3 : A_playground = (1 / 3 : ℚ) * (length * b)) :
  length = 120 :=
by
  sorry

end landscape_length_l490_490017


namespace quadrilateral_AYXZ_is_parallelogram_l490_490334

variables {A B C X Y Z : Type} [EuclideanGeometry A B C X Y Z]

-- Given conditions of similar triangles
axiom similar_YBA : ∀ (A B Y : point), similar (triangle Y B A)
axiom similar_ZAC : ∀ (A C Z : point), similar (triangle Z A C)
axiom similar_XBC : ∀ (B C X : point), similar (triangle X B C)

-- Note: The specific orientations and positions depend on similarity relations

-- Main theorem: Prove that quadrilateral AYXZ is a parallelogram
theorem quadrilateral_AYXZ_is_parallelogram 
    (hYZ : similar_YBA A B Y)
    (hXZ : similar_ZAC A C Z) 
    (hXY : similar_XBC B C X) : parallelogram A Y X Z := 
    sorry

end quadrilateral_AYXZ_is_parallelogram_l490_490334


namespace weights_difference_l490_490400

-- Definitions based on conditions
def A : ℕ := 36
def ratio_part : ℕ := A / 4
def B : ℕ := 5 * ratio_part
def C : ℕ := 6 * ratio_part

-- Theorem to prove
theorem weights_difference :
  (A + C) - B = 45 := by
  sorry

end weights_difference_l490_490400


namespace sample_capacity_l490_490934

theorem sample_capacity (f : ℕ) (r : ℚ) (n : ℕ) (h₁ : f = 40) (h₂ : r = 0.125) (h₃ : r * n = f) : n = 320 :=
sorry

end sample_capacity_l490_490934


namespace count_good_numbers_l490_490267

noncomputable def is_good_number (k : ℝ) : Prop :=
  let f (x : ℝ) := (x^2 - 1) * (k * x^2 - 6 * x - 8)
  let roots := { x | f x = 0 }
  (roots.to_finset : finset ℝ).card = 3

theorem count_good_numbers :
  { k : ℝ | is_good_number k }.to_finset.card = 4 :=
sorry

end count_good_numbers_l490_490267


namespace cube_corner_max_sum_l490_490749

theorem cube_corner_max_sum (a b c d e f : ℕ) (h1 : a + f = 9) (h2 : b + e = 9) (h3 : c + d = 9) : 
  max ((a + max (b + c) (e + d)) (f + max (b + c) (e + d))) ((b + max (a + d) (f + c)) (e + max (a + d) (f + c))) = 16 := 
sorry

end cube_corner_max_sum_l490_490749


namespace complex_frac_pure_imaginary_l490_490624

theorem complex_frac_pure_imaginary (a : ℝ) (i : ℂ) (hi : i = complex.I) :
  let z := (a - i) / (1 - i)
  in (z.re = 0 ∧ z.im ≠ 0) → a = -1 :=
by
  sorry

end complex_frac_pure_imaginary_l490_490624


namespace son_l490_490100

variable (S M : ℕ)
variable h1 : M = S + 26
variable h2 : M + 2 = 2 * (S + 2)

theorem son's_age_is_24 : S = 24 :=
by
  sorry

end son_l490_490100


namespace sin_double_theta_l490_490256

theorem sin_double_theta (θ : ℝ) (h : exp (2 * complex.I * θ) = (2 + complex.I * real.sqrt 5) / 3) : real.sin (2 * θ) = real.sqrt 3 / 3 :=
sorry

end sin_double_theta_l490_490256


namespace sum_of_roots_l490_490997

theorem sum_of_roots : 
  ∀ x1 x2 : ℝ, 
  (x1^2 + 2023*x1 = 2024 ∧ x2^2 + 2023*x2 = 2024) → 
  x1 + x2 = -2023 := 
by 
  sorry

end sum_of_roots_l490_490997


namespace equation_of_common_chord_equation_of_perpendicular_bisector_distance_between_centers_l490_490754

-- Define the equations of the circles
def circle1 : ℝ → ℝ → Prop := λ x y, x^2 + y^2 - 2 * x = 0
def circle2 : ℝ → ℝ → Prop := λ x y, x^2 + y^2 + 4 * x - 6 * y = 0

-- The centers of the circles
def center1 : ℝ × ℝ := (1, 0)
def center2 : ℝ × ℝ := (-2, 3)

-- Equations to prove
theorem equation_of_common_chord : ∀ x y : ℝ, circle1 x y ∧ circle2 x y → x - y = 0 := by
  sorry

theorem equation_of_perpendicular_bisector : ∀ x y : ℝ, (x = 1 ∧ y = 0) → (x = -2 ∧ y = 3) → x + y - 1 = 0 := by
  sorry

theorem distance_between_centers : Real.sqrt((1 + 2)^2 + (0 - 3)^2) = 3 * Real.sqrt 2 := by
  sorry

end equation_of_common_chord_equation_of_perpendicular_bisector_distance_between_centers_l490_490754


namespace sum_of_solutions_l490_490800

theorem sum_of_solutions (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20 ∧ ∀ x, a * x^2 + b * x + c = 0) : 
  -b / a = 9 :=
by
  -- The proof is omitted here (hence the 'sorry')
  sorry

end sum_of_solutions_l490_490800


namespace find_SC_l490_490432

variables (A B R : ℝ)

-- Angles involved by the edges and faces of the pyramid.
variables (alpha beta gamma : ℝ)

-- Required condition between angles.
axiom alpha_beta_gamma_condition : 1 / sin alpha + 1 / sin beta - 1 / sin gamma = 1

-- Definition of |SC| in terms of given angle A and B at the base and radius R of the circumscribed circle.
def |SC| : ℝ := 2 * R * sqrt (cos (A + B) * cos (A - B))

theorem find_SC : 
  ∃ (d : ℝ), 1 / sin alpha + 1 / sin beta - 1 / sin gamma = 1 → d = 2 * R * sqrt (cos (A + B) * cos (A - B)) :=
sorry

end find_SC_l490_490432


namespace find_geometric_sequence_formula_and_lambda_range_l490_490198

noncomputable def geometric_sequence_formula (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  (S 3 = 3 / 2) ∧ (S 6 = 21 / 16) ∧ (∀ n, S n = a 1 * (1 - (-1 / 2) ^ n) / (1 - (-1 / 2))) ∧
  (∀ n, a n = 2 * (-1 / 2) ^ (n - 1))

noncomputable def lambda_range (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (λ : ℝ) :=
  geometric_sequence_formula S a ∧
  (∀ n, b n = λ * a n - n ^ 2) ∧
  (∀ n : ℕ, n > 0 → b n > b (n + 1)) ↔ (-1 < λ ∧ λ < 10 / 3)


theorem find_geometric_sequence_formula_and_lambda_range :
  ∃ S a b λ, geometric_sequence_formula S a ∧ lambda_range S a b λ :=
begin
  -- proof here
  sorry
end

end find_geometric_sequence_formula_and_lambda_range_l490_490198


namespace sine_has_property_T_log_does_not_have_property_T_exp_does_not_have_property_T_cubic_does_not_have_property_T_l490_490549

def has_property_T (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f' x₁ * f' x₂ = -1

def log_function : ℝ → ℝ := λ x, log x
def sine_function : ℝ → ℝ := λ x, sin x
def exp_function : ℝ → ℝ := λ x, exp x
def cubic_function : ℝ → ℝ := λ x, x ^ 3

theorem sine_has_property_T :
  has_property_T sine_function := sorry

theorem log_does_not_have_property_T :
  ¬ has_property_T log_function := sorry

theorem exp_does_not_have_property_T :
  ¬ has_property_T exp_function := sorry

theorem cubic_does_not_have_property_T :
  ¬ has_property_T cubic_function := sorry

end sine_has_property_T_log_does_not_have_property_T_exp_does_not_have_property_T_cubic_does_not_have_property_T_l490_490549


namespace sum_of_solutions_l490_490832

theorem sum_of_solutions (x : ℝ) : 
  (∀ x : ℝ, x^2 = 9*x - 20 → x = 4 ∨ x = 5) → (4 + 5 = 9) :=
by
  intros h
  calc 4 + 5 = 9 : by norm_num
  sorry

end sum_of_solutions_l490_490832


namespace symmetry_of_translated_function_l490_490018

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) := Real.sin (ω * x + φ)

theorem symmetry_of_translated_function {ω φ : ℝ}
  (h1 : 0 < ω)
  (h2 : abs φ < Real.pi / 2)
  (h3 : Real.sin (f (x - π / 3) (2 * ω) φ) = -Real.sin (f (x - π / 3) (2 * ω) (-φ))) :
  ∃ c : ℝ, c = 5 * Real.pi / 12 ∧ ∀ x, f(x) = f(2 * c - x) := by
  sorry

end symmetry_of_translated_function_l490_490018


namespace maximize_area_of_quadrilateral_l490_490108

theorem maximize_area_of_quadrilateral (k : ℝ) (h0 : 0 < k) (h1 : k < 1) 
    (hE : ∀ E : ℝ, E = 2 * k) (hF : ∀ F : ℝ, F = 2 * k) :
    k = 1/2 ∧ (2 * (1 - k) ^ 2) = 1/2 := 
by 
  sorry

end maximize_area_of_quadrilateral_l490_490108


namespace compound_interest_time_l490_490987

theorem compound_interest_time (P r CI : ℝ) (n : ℕ) (A : ℝ) :
  P = 16000 ∧ r = 0.15 ∧ CI = 6218 ∧ n = 1 ∧ A = P + CI →
  t = 2 :=
by
  sorry

end compound_interest_time_l490_490987


namespace maximum_area_triangle_correct_l490_490209

def point (α : Type) := prod α α

def circle (α : Type) := point α × α

def triangle (α : Type) := point α × point α × point α

def is_on_circle {α : Type} [linear_ordered_field α] (p : point α) (c : circle α) : Prop :=
  let (center, radius) := c in
  let (x, y) := p in
  (x - center.1)^2 + (y - center.2)^2 = radius^2

def right_angle {α : Type} [linear_ordered_field α] (a b c : point α) : Prop :=
  let ⟨ax, ay⟩ := a in
  let ⟨bx, by⟩ := b in
  let ⟨cx, cy⟩ := c in
  let dot_product (u v w z : α) := u*v + w*z in
  (dot_product (bx - ax) (cx - ax) (by - ay) (cy - ay)) = 0

noncomputable def maximum_area_triangle {α : Type} [linear_ordered_field α]
  (A B C : point α)
  (O : circle α)
  (hA : A = (0, 3))
  (hB : is_on_circle B O)
  (hC : is_on_circle C O)
  (hAngle : right_angle A B C) : α :=
  sorry

-- Specifications
noncomputable def maximum_area {α : Type} [linear_ordered_field α] : α :=
  (25 + 3 * real.sqrt 41) / 2

-- Proof statement
theorem maximum_area_triangle_correct :
  ∀ (α : Type) [linear_ordered_field α],
  ∃ (A B C : point α) (O : circle α),
  maximum_area_triangle A B C O ((0, 3) : point α)
  (is_on_circle B O)
  (is_on_circle C O)
  (right_angle (0, 3) B C) = maximum_area :=
by
  intros
  existsi ((0, 3) : point α)
  existsi B
  existsi C
  existsi O
  sorry

end maximum_area_triangle_correct_l490_490209


namespace platform_length_is_correct_l490_490938

noncomputable def first_platform_length (time1 time2 train_length platform2_length speed1 speed2 length1: ℕ): ℕ :=
  let v := speed1 = speed2
  if v then length1 else −1

theorem platform_length_is_correct :
  ∀ (train_length time1 time2 platform2_length : ℕ),
    (train_length = 70) →
    (time1 = 15) →
    (time2 = 20) →
    (platform2_length = 250) →
    (20 * (train_length + 250)) = (15 * (train_length + 70)) →
    (3400 = 20 * (first_platform_length time1 time2 train_length platform2_length
                    ((train_length + 250) / 20) ((train_length + 70) / 15)
                    (170))) :=
begin
  -- proof to be filled in
  sorry
end

end platform_length_is_correct_l490_490938


namespace triangle_area_l490_490757

theorem triangle_area :
  ∀ (k : ℝ), ∃ (area : ℝ), 
  (∃ (r : ℝ) (a b c : ℝ), 
      r = 2 * Real.sqrt 3 ∧
      a / b = 3 / 5 ∧ a / c = 3 / 7 ∧ b / c = 5 / 7 ∧
      (∃ (A B C : ℝ),
          A = 3 * k ∧ B = 5 * k ∧ C = 7 * k ∧
          area = (1/2) * a * b * Real.sin (2 * Real.pi / 3))) →
  area = (135 * Real.sqrt 3 / 49) :=
sorry

end triangle_area_l490_490757


namespace propositions_validity_l490_490956

theorem propositions_validity :
  (∀ x y : ℝ, (x * y = 1 → x = 1/y) ↔ (x = 1/y → x * y = 1)) ∧
  (¬∀ (T1 T2 : Triangle), (T1 ~ T2) → (T1.perimeter = T2.perimeter)) ∧
  (∀ b : ℝ, ((b ≤ -1 → ∃ x : ℝ, x^2 - 2*b*x + b^2 + b = 0) ↔ ¬(∃ x : ℝ, x^2 - 2*b*x + b^2 + b = 0) → b > -1)) ∧
  (∀ (A B : Set α), (A ∪ B = B → A ⊇ B) ↔ (A ⊇ B → A ∪ B = B)) :=
by
  sorry

end propositions_validity_l490_490956


namespace conic_sections_of_equation_l490_490966

noncomputable def is_parabola (s : Set (ℝ × ℝ)) : Prop :=
∃ a b c : ℝ, ∀ x y : ℝ, (x, y) ∈ s ↔ y ≠ 0 ∧ y = a * x^3 + b * x + c

theorem conic_sections_of_equation :
  let eq := { p : ℝ × ℝ | p.2^6 - 9 * p.1^6 = 3 * p.2^3 - 1 }
  (is_parabola eq1) → (is_parabola eq2) → (eq = eq1 ∪ eq2) :=
by sorry

end conic_sections_of_equation_l490_490966


namespace compare_fraction_product_l490_490136

theorem compare_fraction_product :
  (∏ i in Finset.range 461, (100 + 2*i : ℝ) / (101 + 2*i)) < (5/16 : ℝ) := 
sorry

end compare_fraction_product_l490_490136


namespace percentage_of_students_play_sports_l490_490402

def total_students : ℕ := 400
def soccer_percentage : ℝ := 0.125
def soccer_players : ℕ := 26

theorem percentage_of_students_play_sports : 
  ∃ P : ℝ, (soccer_percentage * P = soccer_players) → (P / total_students * 100 = 52) :=
by
  sorry

end percentage_of_students_play_sports_l490_490402


namespace unique_function_satisfying_conditions_l490_490172

def satisfies_conditions (f : ℕ+ → ℕ+) : Prop :=
  (∀ n : ℕ+, f (n!) = (f n)!) ∧ (∀ n m : ℕ+, (n - m) ∣ (f n - f m))

theorem unique_function_satisfying_conditions (f : ℕ+ → ℕ+) (hf : satisfies_conditions f) : 
  ∀ n : ℕ+, f n = n :=
by 
  sorry

end unique_function_satisfying_conditions_l490_490172


namespace total_distance_walked_l490_490891

noncomputable def desk_to_fountain_distance : ℕ := 30
noncomputable def number_of_trips : ℕ := 4

theorem total_distance_walked :
  2 * desk_to_fountain_distance * number_of_trips = 240 :=
by
  sorry

end total_distance_walked_l490_490891


namespace false_statements_count_is_3_l490_490328

-- Define the statements
def statement1_false : Prop := ¬ (1 ≠ 1)     -- Not exactly one statement is false
def statement2_false : Prop := ¬ (2 ≠ 2)     -- Not exactly two statements are false
def statement3_false : Prop := ¬ (3 ≠ 3)     -- Not exactly three statements are false
def statement4_false : Prop := ¬ (4 ≠ 4)     -- Not exactly four statements are false
def statement5_false : Prop := ¬ (5 ≠ 5)     -- Not all statements are false

-- Prove that the number of false statements is 3
theorem false_statements_count_is_3 :
  (statement1_false → statement2_false →
  statement3_false → statement4_false →
  statement5_false → (3 = 3)) := by
  sorry

end false_statements_count_is_3_l490_490328


namespace rope_length_loss_l490_490410

theorem rope_length_loss
  (stories_needed : ℕ)
  (feet_per_story : ℕ)
  (pieces_of_rope : ℕ)
  (feet_per_rope : ℕ)
  (total_feet_needed : ℕ)
  (total_feet_bought : ℕ)
  (percentage_lost : ℕ) :
  
  stories_needed = 6 →
  feet_per_story = 10 →
  pieces_of_rope = 4 →
  feet_per_rope = 20 →
  total_feet_needed = stories_needed * feet_per_story →
  total_feet_bought = pieces_of_rope * feet_per_rope →
  total_feet_needed <= total_feet_bought →
  percentage_lost = ((total_feet_bought - total_feet_needed) * 100) / total_feet_bought →
  percentage_lost = 25 :=
by
  intros h_stories h_feet_story h_pieces h_feet_rope h_total_needed h_total_bought h_needed_bought h_percentage
  sorry

end rope_length_loss_l490_490410


namespace no_other_distinct_prime_products_l490_490053

theorem no_other_distinct_prime_products :
  ∀ (q1 q2 q3 : Nat), 
  Prime q1 ∧ Prime q2 ∧ Prime q3 ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 ∧ q1 * q2 * q3 ≠ 17 * 11 * 23 → 
  q1 + q2 + q3 ≠ 51 :=
by
  intros q1 q2 q3 h
  sorry

end no_other_distinct_prime_products_l490_490053


namespace sum_of_solutions_eq_9_l490_490827

theorem sum_of_solutions_eq_9 :
  let roots := {x : ℝ | x^2 = 9 * x - 20}
  in ∑ x in roots, x = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490827


namespace no_zero_k_le_4_l490_490532

def u (n : ℕ) : ℕ := n^4 + 2 * n^2

def Δ1 (un : ℕ → ℕ) (n : ℕ) : ℕ := un (n + 1) - un n

def Δk (k : ℕ) (un : ℕ → ℕ) (n : ℕ) : ℕ :=
  if k = 1 then Δ1 un n else Δ1 (Δk (k - 1)) n

theorem no_zero_k_le_4 (n : ℕ) : ¬ ∃ k ≤ 4, ∀ n, Δk k u n = 0 :=
by {
  sorry
}

end no_zero_k_le_4_l490_490532


namespace math_problem_proof_l490_490483

-- Define the fractions involved
def frac1 : ℚ := -49
def frac2 : ℚ := 4 / 7
def frac3 : ℚ := -8 / 7

-- The original expression
def original_expr : ℚ :=
  frac1 * frac2 - frac2 / frac3

-- Declare the theorem to be proved
theorem math_problem_proof : original_expr = -27.5 :=
by
  sorry

end math_problem_proof_l490_490483


namespace total_time_on_road_l490_490313

theorem total_time_on_road (v1 v2 t1 : ℝ) (h1 : v1 = 58) (h2 : t1 = 1.4) (h3 : v2 = 62) :
  let d := v1 * t1,
      t2 := d / v2,
      total_time := t1 + t2
  in total_time = 2.71 :=
by
  sorry

end total_time_on_road_l490_490313


namespace greatest_integer_jean_thinks_of_l490_490664

theorem greatest_integer_jean_thinks_of :
  ∃ n : ℕ, n < 150 ∧ (∃ a : ℤ, n + 2 = 9 * a) ∧ (∃ b : ℤ, n + 3 = 11 * b) ∧ n = 142 :=
by
  sorry

end greatest_integer_jean_thinks_of_l490_490664


namespace isosceles_trapezoid_exists_l490_490516

noncomputable theory

def isosceles_trapezoid (x y z u r s t : ℤ) :=
  x = 8 * r * t ∧
  y = 8 * s * t ∧
  z = r ^ 2 + s ^ 2 - 2 * r * s + 4 * t ^ 2 ∧
  u = r ^ 2 + s ^ 2 - 2 * r * s - 4 * t ^ 2 ∧
  x ^ 2 + y ^ 2 + 4 * u ^ 2 = 4 * z ^ 2 + 2 * x * y

theorem isosceles_trapezoid_exists : 
  ∃ (x y z u r s t : ℤ), isosceles_trapezoid x y z u r s t :=
sorry

end isosceles_trapezoid_exists_l490_490516


namespace sum_fn_geq_zero_l490_490228

noncomputable def fn (n : ℕ) (x : ℝ) : ℝ :=
  (n * x^2 - x) / (x^2 + 1)

theorem sum_fn_geq_zero (n : ℕ) (x : ℝ) (x_i : ℕ → ℝ) (h_n : 0 < n) (h_xi_pos : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < x_i i) 
  (h_sum_xi : (Finset.range n).sum (λ i, x_i i) = 1) : 
  (Finset.range n).sum (λ i, fn n (x_i i)) ≥ 0 :=
sorry

end sum_fn_geq_zero_l490_490228


namespace no_valid_sequence_of_integers_from_1_to_2004_l490_490484

theorem no_valid_sequence_of_integers_from_1_to_2004 :
  ¬ ∃ (a : ℕ → ℕ), 
    (∀ i, 1 ≤ a i ∧ a i ≤ 2004) ∧ 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    (∀ k, 1 ≤ k ∧ k + 9 ≤ 2004 → 
      (a k + a (k + 1) + a (k + 2) + a (k + 3) + a (k + 4) + a (k + 5) + 
       a (k + 6) + a (k + 7) + a (k + 8) + a (k + 9)) % 10 = 0) :=
  sorry

end no_valid_sequence_of_integers_from_1_to_2004_l490_490484


namespace solution_l490_490227

variable {A B C : ℝ} -- Represent angles in radians

noncomputable def problem : Prop :=
  -- Define conditions
  (cos (A - B) * cos (B - C) * cos (C - A) = 1) ∧ -- Condition for ①
  (sin A = cos B) ∧ -- Condition for ②
  (cos A * cos B * cos C < 0) ∧ -- Condition for ③
  (sin (2 * A) = sin (2 * B)) -- Condition for ④

theorem solution :
  -- Verify the correct statements
  problem → 
  ((
    (cos (A - B) * cos (B - C) * cos (C - A) = 1) → (A = B ∧ B = C ∧ A = C) ∧
    (cos A * cos B * cos C < 0) → (∃! x, (x = A ∨ x = B ∨ x = C) ∧ cos x < 0)
  ) ∧
  (¬((sin A = cos B) → (A = π / 2 ∨ B = π / 2 ∨ C = π / 2)) ∧
  ¬((sin (2 * A) = sin (2 * B)) → (A = B ∨ A + B = π / 2)))
:=
by sorry -- Proof description omitted

end solution_l490_490227


namespace four_drivers_suffice_l490_490920

theorem four_drivers_suffice
  (one_way_trip_time : ℕ := 160) -- in minutes
  (round_trip_time : ℕ := 320) -- in minutes
  (rest_time : ℕ := 60) -- in minutes
  (time_A_returns : ℕ := 760) -- 12:40 PM in minutes from midnight
  (time_A_next_start : ℕ := 820) -- 1:40 PM in minutes from midnight
  (time_D_departs : ℕ := 785) -- 1:05 PM in minutes from midnight
  (time_A_fifth_depart : ℕ := 970) -- 4:10 PM in minutes from midnight
  (time_B_returns : ℕ := 960) -- 4:00 PM in minutes from midnight
  (time_B_sixth_depart : ℕ := 1050) -- 5:30 PM in minutes from midnight
  (time_A_fifth_complete: ℕ := 1290) -- 9:30 PM in minutes from midnight
  : 4_drivers_sufficient : ℕ :=
    if time_A_fifth_complete = 1290 then 1 else 0
-- The theorem states that if the calculated trip completion time is 9:30 PM, then 4 drivers are sufficient.
  sorry

end four_drivers_suffice_l490_490920


namespace simple_interest_rate_l490_490783

theorem simple_interest_rate (P SI T : ℝ) (hP : P = 15000) (hSI : SI = 6000) (hT : T = 8) :
  ∃ R : ℝ, (SI = P * R * T / 100) ∧ R = 5 :=
by
  use 5
  field_simp [hP, hSI, hT]
  sorry

end simple_interest_rate_l490_490783


namespace find_minimum_m_l490_490354

theorem find_minimum_m (m : ℕ) (h1 : 1350 + 36 * m < 2136) (h2 : 1500 + 45 * m ≥ 2365) :
  m = 20 :=
by
  sorry

end find_minimum_m_l490_490354


namespace difference_between_circle_areas_l490_490888

theorem difference_between_circle_areas 
  (C1 : ℝ) (C2 : ℝ) 
  (hC1 : C1 = 268) 
  (hC2 : C2 = 380) : 
  | (π * ((C2 / (2 * π))^2) - π * ((C1 / (2 * π))^2)) - 5778.33 | < 0.01 :=
by
  sorry

end difference_between_circle_areas_l490_490888


namespace problem_f_2009_plus_f_2010_l490_490572

theorem problem_f_2009_plus_f_2010 (f : ℝ → ℝ) 
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (2 * x + 1) = f (2 * (x + 5 / 2) + 1))
  (h_f1 : f 1 = 5) :
  f 2009 + f 2010 = 0 :=
sorry

end problem_f_2009_plus_f_2010_l490_490572


namespace preparation_start_month_l490_490728

variable (ExamMonth : ℕ)
def start_month (ExamMonth : ℕ) : ℕ :=
  (ExamMonth - 5) % 12

theorem preparation_start_month :
  ∀ (ExamMonth : ℕ), start_month ExamMonth = (ExamMonth - 5) % 12 :=
by
  sorry

end preparation_start_month_l490_490728


namespace measure_angle_pbq_is_45_l490_490719

-- Definitions of geometric entities and their properties
variables (A B C D P Q : Type) [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited P] [inhabited Q]
variables (AC : A -> C -> Type)
variables (square : A -> B -> C -> D -> Prop)
variables (on_diagonal : P -> AC A C -> Prop)
variables (distance : A -> B -> ℝ)
variables (measure_angle : P -> B -> Q -> ℝ)

-- Conditions for the problem
axiom ab_eq_1 : distance A B = 1
axiom ap_eq_1 : distance A P = distance A B
axiom cq_eq_1 : distance C Q = distance A B
axiom square_abcd : square A B C D
axiom p_on_ac : on_diagonal P (AC A C)
axiom q_on_ac : on_diagonal Q (AC A C)

-- Proof problem
theorem measure_angle_pbq_is_45 : measure_angle P B Q = 45 :=
by sorry

end measure_angle_pbq_is_45_l490_490719


namespace four_digit_even_numbers_distinct_l490_490605

theorem four_digit_even_numbers_distinct (digits : Finset ℕ) (h : digits = {1, 2, 3, 4, 5}) :
  (∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 2 = 0 ∧ (∀ i j, i ≠ j → n.digit i ≠ n.digit j) ∧ ∀ d ∈ digits, d ∈ n.digit) →
  n = 48 :=
by by sorry

end four_digit_even_numbers_distinct_l490_490605


namespace correct_calculation_l490_490056

theorem correct_calculation (a : ℝ) : a^3 / a^2 = a := by
  sorry

end correct_calculation_l490_490056


namespace correct_statements_l490_490472

def angles_terminal_on_y_axis (α : ℝ) (k : ℤ) : Prop :=
  α = k * π + π / 2

def symmetry_center (x y : ℝ) : Prop :=
  y = 2 * cos (x - π / 4)

def tan_increasing_in_first_quadrant : Prop :=
  ∀ x, 0 < x → x < π / 2 → monotone (tan) x

def range_of_f (a b: ℝ) (x : ℝ) : Prop :=
  a > 0 ∧ x ∈ [π/4, 3*π/4] →
  -3 ≤ 2 * a * sin (2 * x + π / 6) - 2 * a + b ∧ 2 * a * sin (2 * x + π / 6) - 2 * a + b ≤ sqrt 3 - 1

theorem correct_statements (α : ℝ) (k : ℤ) (a b x : ℝ) :
  ¬angles_terminal_on_y_axis α k ∧ symmetry_center (3 * π / 4) 0 ∧ 
  ¬tan_increasing_in_first_quadrant ∧ range_of_f 1 1 :=
  sorry

end correct_statements_l490_490472


namespace sum_of_solutions_l490_490836

theorem sum_of_solutions (x : ℝ) : 
  (∀ x : ℝ, x^2 = 9*x - 20 → x = 4 ∨ x = 5) → (4 + 5 = 9) :=
by
  intros h
  calc 4 + 5 = 9 : by norm_num
  sorry

end sum_of_solutions_l490_490836


namespace product_distance_geq_fractional_factorial_l490_490722

def distance_to_nearest_integer (a : ℝ) : ℝ :=
  min (a - a.floor) (a.ceil - a)

theorem product_distance_geq_fractional_factorial (n : ℕ) (a : ℝ) :
  (∏ i in Finset.range (n + 1), abs (a - i)) ≥ distance_to_nearest_integer a * (n.factorial / 2^n) :=
sorry

end product_distance_geq_fractional_factorial_l490_490722


namespace problem_1_problem_2_problem_3_problem_4_l490_490479

theorem problem_1 : (2 - (-4) + 8 / (-2) + (-3) = -1) :=
by
  sorry

theorem problem_2 : (25 * (3 / 4) - (-25) * (1 / 2) + 25 * (- 1 / 4) = 25) :=
by
  sorry

theorem problem_3 : ((1 / 2 + 5 / 6 - 7 / 12) * (-24) = -18) :=
by
  sorry

theorem problem_4 : (-3^2 / (-2)^2 * | -4 / 3 | * 6 + (-2)^3 = -80) :=
by
  sorry

end problem_1_problem_2_problem_3_problem_4_l490_490479


namespace number_of_three_digit_numbers_l490_490403

theorem number_of_three_digit_numbers (cards : Set ℕ) (includes_six_nine_interchange : ∀ x, x ∈ cards → (x = 6 ∨ x = 9) → (6 ∈ cards ∧ 9 ∈ cards)) :
  (cards = {0, 1, 2, 4, 6, 9}) → ∃ n : ℕ, n = 40 ∧
  n = (∀ combination : Finset ℕ, combination.card = 3 → combination.sum % 3 = 0 → 
  (combination.card * combination.card.pred * combination.card.succ)) :=
by
  sorry

end number_of_three_digit_numbers_l490_490403


namespace projection_of_difference_l490_490570

variables (a b : ℝ^3)
variables (hab : ∥a∥ = 1 ∧ ∥b∥ = 1) (angle_ab : real.angle a b = real.angle.pi * 2 / 3)

theorem projection_of_difference :
  let proj := (a - b).dot b / b.dot b in
  proj • b = -3/2 • b :=
by 
  sorry

end projection_of_difference_l490_490570


namespace true_discount_calculation_l490_490356

structure FinancialContext where
  BG : ℝ   -- Banker's Gain
  R : ℝ    -- Rate of interest per annum (percentage)
  T : ℝ    -- Time in years

def BankersDiscount (FV R T : ℝ) : ℝ := (FV * R * T) / 100
def TrueDiscount (PV R T : ℝ) : ℝ := (PV * R * T) / 100
def PresentValue (FV TD : ℝ) : ℝ := FV - TD

theorem true_discount_calculation (ctx : FinancialContext) (FV : ℝ) (TD : ℝ) :
  ctx.BG = BankersDiscount FV ctx.R ctx.T - TrueDiscount (PresentValue FV TD) ctx.R ctx.T →
  TD = (ctx.BG * 100) / (ctx.R * ctx.T) → 
  ctx.BG = 6.6 →
  ctx.R = 12 →
  ctx.T = 1 →
  TD = 55 := 
by
  sorry

end true_discount_calculation_l490_490356


namespace part1_l490_490072

theorem part1 (a b c : ℚ) (h1 : a^2 = 9) (h2 : |b| = 4) (h3 : c^3 = 27) (h4 : a * b < 0) (h5 : b * c > 0) : 
  a * b - b * c + c * a = -33 := by
  sorry

end part1_l490_490072


namespace solve_for_y_l490_490000

theorem solve_for_y (y : ℝ) (h : arctan (2 / y) + arctan (1 / y^2) = π / 4) : y = 3 :=
  sorry

end solve_for_y_l490_490000


namespace math_problem_solution_l490_490046

theorem math_problem_solution : 8 / 4 - 3 - 9 + 3 * 9 = 17 := 
by 
  sorry

end math_problem_solution_l490_490046


namespace max_watched_hours_l490_490316

-- Define the duration of one episode in minutes
def episode_duration : ℕ := 30

-- Define the number of weekdays Max watched the show
def weekdays_watched : ℕ := 4

-- Define the total minutes Max watched
def total_minutes_watched : ℕ := episode_duration * weekdays_watched

-- Define the conversion factor from minutes to hours
def minutes_to_hours_factor : ℕ := 60

-- Define the total hours watched
def total_hours_watched : ℕ := total_minutes_watched / minutes_to_hours_factor

-- Proof statement
theorem max_watched_hours : total_hours_watched = 2 :=
by
  sorry

end max_watched_hours_l490_490316


namespace minyoung_gave_nine_notebooks_l490_490320

theorem minyoung_gave_nine_notebooks (original left given : ℕ) (h1 : original = 17) (h2 : left = 8) (h3 : given = original - left) : given = 9 :=
by
  rw [h1, h2] at h3
  exact h3

end minyoung_gave_nine_notebooks_l490_490320


namespace coeff_x3_expansion_l490_490746

theorem coeff_x3_expansion : (coeff (x^3) ((1 - x) * (1 + x)^6)) = 5 := by
  sorry

end coeff_x3_expansion_l490_490746


namespace quadrant_and_terminal_angle_l490_490541

def alpha : ℝ := -1910 

noncomputable def normalize_angle (α : ℝ) : ℝ := 
  let β := α % 360
  if β < 0 then β + 360 else β

noncomputable def in_quadrant_3 (β : ℝ) : Prop :=
  180 ≤ β ∧ β < 270

noncomputable def equivalent_theta (α : ℝ) (θ : ℝ) : Prop :=
  (α % 360 = θ % 360) ∧ (-720 ≤ θ ∧ θ < 0)

theorem quadrant_and_terminal_angle :
  in_quadrant_3 (normalize_angle alpha) ∧ 
  (equivalent_theta alpha (-110) ∨ equivalent_theta alpha (-470)) :=
by 
  sorry

end quadrant_and_terminal_angle_l490_490541


namespace domain_of_rational_function_l490_490988

theorem domain_of_rational_function :
  let f (x : ℝ) := (x^3 - 3 * x^2 + 5 * x - 2) / (x^3 - 5 * x^2 + 8 * x - 4)
  SetOf (λ x : ℝ, x^3 - 5 * x^2 + 8 * x - 4 ≠ 0) = 
    {x : ℝ | x < 1} ∪ {x : ℝ | 1 < x ∧ x < 2} ∪ {x : ℝ | 2 < x ∧ x < 4} ∪ {x : ℝ | 4 < x} := by
sorry

end domain_of_rational_function_l490_490988


namespace compliment_A_B_l490_490703

open Set

def A := {0, 2, 4, 6, 8, 10}
def B := {4, 8}

theorem compliment_A_B :
  A \ B = {0, 2, 6, 10} := by
  sorry

end compliment_A_B_l490_490703


namespace factor_81_sub_27x3_l490_490975

theorem factor_81_sub_27x3 (x : ℝ) : 81 - 27 * x^3 = 3 * (3 - x) * (81 + 27 * x + 9 * x^2) :=
sorry

end factor_81_sub_27x3_l490_490975


namespace disk_max_areas_l490_490904

-- Conditions Definition
def disk_divided (n : ℕ) : ℕ :=
  let radii := 3 * n
  let secant_lines := 2
  let total_areas := 9 * n
  total_areas

theorem disk_max_areas (n : ℕ) : disk_divided n = 9 * n :=
by
  sorry

end disk_max_areas_l490_490904


namespace part_a_part_b_part_c_l490_490955

-- Definitions for Part (a)
variable (α β γ A B C : ℝ)
variable (angle_xSz : α = ∠ x S z)
variable (angle_xSy : β = ∠ x S y)
variable (angle_ySz : γ = ∠ y S z)
variable (dihedral_y : A = ∠ {S y})
variable (dihedral_z : B = ∠ {S z})
variable (dihedral_x : C = ∠ {S x})

-- Proof for Part (a)
theorem part_a : (sin α / sin A) = (sin β / sin B) = (sin γ / sin C) := sorry

-- Definition for Part (b)
variable (angle_sum : α + β = 180)
variable (angle_dihedral_sum : A + B = 180)

-- Proof for Part (b)
theorem part_b : (α + β = 180 ↔ ∠ A + ∠ B = 180) := sorry

-- Definitions for Part (c)
variable (right_trihedral : α = 90 ∧ β = 90 ∧ γ = 90)
variable (point_O : ∃ O ∈ Sz, SO = a)
variable (points_M_N : ∀ M, M ∈ x ∧ N ∈ y)

-- Proof for Part (c)
theorem part_c : ∠ SOM + ∠ SON + ∠ MON = 270 ∧ 
                 (locus_incenter : incenter OSMN = circle O a) := sorry

end part_a_part_b_part_c_l490_490955


namespace sum_of_solutions_eq_9_l490_490808

theorem sum_of_solutions_eq_9 (a b c : ℝ) (h : a = 1 ∧ b = -9 ∧ c = 20) :
  let (sum_roots : ℝ) := -b / a in 
  sum_roots = 9 :=
by
  sorry

end sum_of_solutions_eq_9_l490_490808


namespace area_under_curve_correct_l490_490685

def g (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 6 then x^2
else if h : 6 < x ∧ x ≤ 10 then 3*x - 9
else 0

theorem area_under_curve_correct :
  ∫ x in 0..6, x^2 + ∫ x in 6..10, 3*x - 9 = 132 :=
begin
  sorry
end

end area_under_curve_correct_l490_490685


namespace sum_of_solutions_l490_490831

theorem sum_of_solutions (x : ℝ) : 
  (∀ x : ℝ, x^2 = 9*x - 20 → x = 4 ∨ x = 5) → (4 + 5 = 9) :=
by
  intros h
  calc 4 + 5 = 9 : by norm_num
  sorry

end sum_of_solutions_l490_490831


namespace profit_eq_simplified_max_profit_x_l490_490912

-- Define the conditions
variable (a : ℝ) (h1 : 0 < a) (h2 : a ≥ 0)   

-- Define the sales volume equation
def salesVolume (x : ℝ) : ℝ := 5 - 2 / x

-- Define the production cost function
def productionCost (t : ℝ) : ℝ := 10 + 2 * t

-- Define the sales price function
def salesPrice (t : ℝ) : ℝ := 4 + 20 / t

-- Define the profit function
def profit (x : ℝ) : ℝ := 
  let t := salesVolume x in
  (2 * productionCost t / t) * t - productionCost t - x

-- Simplified profit function based on the problem statement
def simplifiedProfit (x : ℝ) : ℝ := 80 - (36 / x + x)

-- Proof that the simplified profit function corresponds to the calculated profit
theorem profit_eq_simplified : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ a → profit x = simplifiedProfit x := 
by 
  sorry

-- The value of x that maximizes the profit y under the given constraints
theorem max_profit_x (x : ℝ) : 0 ≤ x ∧ x ≤ a → 
  (a ≥ 6 ∧ x = 6) ∨ (a < 6 ∧ x = a) :=
by
  sorry

end profit_eq_simplified_max_profit_x_l490_490912


namespace sum_of_solutions_l490_490854

theorem sum_of_solutions : 
  (∑ x in {x : ℝ | x^2 = 9*x - 20}, x) = 9 := 
sorry

end sum_of_solutions_l490_490854


namespace tess_distance_graph_correct_l490_490003

noncomputable def triangular_block_distance_graph (A B C : Type) [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] 
  (AB BC CA : ℝ) : Prop :=
  ∀ (tess : ℝ), 
    tess = 0 ∨ tess = AB ∨ tess = BC ∨ tess = CA ->
    match tess with
    | 0 => true -- Starting at A, distance is 0
    | AB => true -- Tess reaches B, distance peaks
    | BC => true -- Tess moves from B to C, distance may fluctuate
    | CA => true -- Tess returns to A, distance back to 0
    | _ => false -- Any other state is invalid

-- The theorem statement 
theorem tess_distance_graph_correct (A B C : Type) [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] 
  (AB BC CA : ℝ) :
triangular_block_distance_graph A B C AB BC CA :=
by sorry

end tess_distance_graph_correct_l490_490003


namespace probability_exactly_one_solves_problem_l490_490898

-- Define the context in which A and B solve the problem with given probabilities.
variables (p1 p2 : ℝ)

-- Define the constraint that the probabilities are between 0 and 1
axiom prob_A_nonneg : 0 ≤ p1
axiom prob_A_le_one : p1 ≤ 1
axiom prob_B_nonneg : 0 ≤ p2
axiom prob_B_le_one : p2 ≤ 1

-- Define the context that A and B solve the problem independently.
axiom A_and_B_independent : true

-- The theorem statement to prove the desired probability of exactly one solving the problem.
theorem probability_exactly_one_solves_problem : (p1 * (1 - p2) + p2 * (1 - p1)) =  p1 * (1 - p2) + p2 * (1 - p1) :=
by
  sorry

end probability_exactly_one_solves_problem_l490_490898


namespace solve_equation1_solve_equation2_l490_490348

-- Lean 4 statements for the given problems:
theorem solve_equation1 (x : ℝ) (h : x ≠ 0) : (2 / x = 3 / (x + 2)) ↔ (x = 4) := by
  sorry

theorem solve_equation2 (x : ℝ) (h : x ≠ 2) : ¬(5 / (x - 2) + 1 = (x - 7) / (2 - x)) := by
  sorry

end solve_equation1_solve_equation2_l490_490348


namespace magnitude_of_angle_B_area_of_triangle_l490_490270

namespace TriangleProblem

noncomputable def find_angle_B (A : ℝ) (B : ℝ) (C : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) :=
  C + (cos A - (real.sqrt 3) * sin A) * cos B = 0 ∧ 0 < B ∧ B < real.pi
  
noncomputable def find_area (A : ℝ) (B : ℝ) (C : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) :=
  b = real.sqrt 3 ∧ c = 1 ∧ B = real.pi / 3

theorem magnitude_of_angle_B (A : ℝ) (B : ℝ) (C : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) :
  find_angle_B A B C a b c → B = real.pi / 3 := 
  sorry

theorem area_of_triangle (A : ℝ) (B : ℝ) (C : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) :
  find_area A B C a b c → 1 / 2 * a * c * sin B = real.sqrt 3 / 2 :=
  sorry

end TriangleProblem

end magnitude_of_angle_B_area_of_triangle_l490_490270
