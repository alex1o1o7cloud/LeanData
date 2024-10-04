import Mathlib
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Combinatorics
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Module.LinearMap
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Quadratics
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Combinatorics
import Mathlib.Combinatorics.Partition
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Fact
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Basic
import Mathlib.LinearAlgebra.BilinearForm
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.Prime.basic
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Linarith
import Mathlib.Tactics
import Mathlib.Topology.Basic
import data.real.basic
import mathlib

namespace parabola_equation_l460_460953

theorem parabola_equation (p : ℝ) (h : p > 0) (A : ℝ × ℝ) :
  let F := (p / 2, 0)
  let M := (2,2)
  ∃ (xA yA : ℝ), A = (xA, yA) ∧ M = ((xA + p / 2)/2, (yA + 0)/2) ∧ yA^2 = 2 * p * xA ∧ yA = 4 ∧ xA = 4 - p / 2 → yA^2 = 8 * xA :=
by
  intro F M A h
  use sorry

end parabola_equation_l460_460953


namespace locus_H_segment_AB_locus_H_interior_triangle_l460_460933

open EuclideanGeometry

-- Define the conditions
variables {O A B M P Q H A' B' : Point}
variable {α : Real}
variable {locus_H : Set Point}

-- Assume the necessary geometric conditions
axiom angle_AOB_lt_90 (h : Angle A O B = α) : α < 90
axiom M_in_triangle (h : InTriangle M O A B) : M ≠ O
axiom feet_perpendicular (h₁ : FootPerpendicular M O A P) (h₂ : FootPerpendicular M O B Q)
axiom orthocenter (h : Orthocenter H O P Q)

-- (a) locus of points H when M belongs to segment AB
theorem locus_H_segment_AB (M_in_segment_AB : OnSegment M A B) : locus_H = Segment A' B' := sorry

-- (b) locus of points H when M belongs to the interior of triangle OAB
theorem locus_H_interior_triangle (M_in_interior : InteriorPoint M O A B) : locus_H = InteriorTriangle O A' B' := sorry

end locus_H_segment_AB_locus_H_interior_triangle_l460_460933


namespace kim_shoe_selection_l460_460684

theorem kim_shoe_selection : 
  ∃ n : ℕ, n > 0 ∧ 
  (∃ C : ℕ → ℕ → ℕ, 
    C 8 n > 0 ∧ 
    4 * C 6 (n - 2) = 0.14285714285714285 * C 8 n)
:= 
begin
  use 2,
  split,
  -- 2 > 0
  linarith,
  use λ a b, nat.choose a b,
  split,
  -- Combination function applied to inputs should be greater than zero
  show nat.choose 8 2 > 0,
  norm_num,
  have h : 4 * nat.choose 6 0 = 0.14285714285714285 * nat.choose 8 2,
  { 
    -- Prove the equality condition
    exact sorry 
  },
  exact h,
end

end kim_shoe_selection_l460_460684


namespace area_ABN_72_11_l460_460868

variables {A B C P Q N : Point}
variables (area_ABC : ℝ)
variables (r1 r2 : ℝ)
variables (hC : Line)
variables (hAC : Line)
variables (hBQ : Line)
variables (hAP : Line)
variables (area_ABN : ℝ)

--Definitions
def Triangle (A B C : Point) : Prop := sorry
def Point_on_Line (P : Point) (l : Line) : Prop := sorry
def Intersect_at_Point (l1 l2 : Line) (P : Point) : Prop := sorry
def Area (ABC : Triangle) : ℝ := sorry

-- Conditions
hypothesis1 : Triangle A B C
hypothesis2 : Point_on_Line P (Line.mk B C) ∧ (Segment.mk B C).length / (Segment.mk P C).length = 3
hypothesis3 : Point_on_Line Q (Line.mk A C) ∧ (Segment.mk A C).length / (Segment.mk Q C).length = 4
hypothesis4 : Intersect_at_Point (Line.mk B Q) (Line.mk A P) N
hypothesis5 : Area (Triangle.mk A B C) = 12

-- Question: Prove area of ABN
theorem area_ABN_72_11 :
  area_ABN =  Area (Triangle.mk A B N) :=
sorry

end area_ABN_72_11_l460_460868


namespace cube_root_of_prime_product_l460_460368

theorem cube_root_of_prime_product : (∛(2^9 * 5^3 * 7^3) = 280) :=
by
  sorry

end cube_root_of_prime_product_l460_460368


namespace math_problem_l460_460154

theorem math_problem :
  2 * sin (60 * π / 180) + abs (-5) - (π - sqrt 2)^0 = sqrt 3 + 4 :=
by
  have h1 : sin (60 * π / 180) = sqrt 3 / 2 := sorry
  have h2 : abs (-5) = 5 := sorry
  have h3 : (π - sqrt 2)^0 = 1 := rfl
  rw [h1, h2, h3]
  norm_num
  sorry

end math_problem_l460_460154


namespace quadratic_poly_correct_l460_460185

-- Definitions based on conditions
def poly (x : ℂ) := 2 * x^2 - 20 * x + 58
def has_real_coeff : Prop := ∀ x : ℂ, poly x ∈ ℝ
def root (z : ℂ) : Prop := poly z = 0

-- The proof of polynomial equality
theorem quadratic_poly_correct :
  ∃ (p : ℂ → ℂ), (∀ x : ℂ, p x = poly x) ∧
  (has_real_coeff) ∧
  (root (5 + 2*complex.i)) ∧
  (p.coeff 2 = 2) :=
sorry

end quadratic_poly_correct_l460_460185


namespace work_completion_time_l460_460441

theorem work_completion_time (A B : Prop) :
  (∃ t₁ t₂ : ℕ, t₁ = 18 ∧ t₂ = 9 ∧ ((1 / t₁) + (1 / t₁) = 1 / t₂)) → t₂ = 9 :=
begin
  intro h,
  rcases h with ⟨t₁, t₂, a1, a2, a3⟩,
  rw [←a3],
  have h_inv: ((1 : ℝ) / 18) + 1 / 18 = 1 / 9, by norm_num,
  rw [←a1, h_inv],
  exact a2,
end

end work_completion_time_l460_460441


namespace area_of_inscribed_triangle_l460_460857

noncomputable def triangle_area_inscribed (arc1 arc2 arc3 : ℝ) (r : ℝ) : ℝ :=
  let θ1 := (arc1 / (2 * π * r)) * (2 * π) in
  let θ2 := (arc2 / (2 * π * r)) * (2 * π) in
  let θ3 := (arc3 / (2 * π * r)) * (2 * π) in
  1 / 2 * r^2 * (Real.sin θ1 + Real.sin θ2 + Real.sin θ3)

theorem area_of_inscribed_triangle {arc1 arc2 arc3 : ℝ}
  (h1 : arc1 = 5) (h2 : arc2 = 6) (h3 : arc3 = 7) :
  triangle_area_inscribed arc1 arc2 arc3 (9 / π) = 101.2488 / π^2 := by
  sorry

end area_of_inscribed_triangle_l460_460857


namespace variance_equivalence_l460_460809

def original_data : List ℝ := [80, 82, 74, 86, 79]
def modified_data : List ℝ := original_data.map (λ x => x - 80)

def variance (data : List ℝ) : ℝ :=
  let mean := (data.sum) / data.length
  (data.map (λ x => (x - mean) ^ 2)).sum / data.length

def s1_squared : ℝ := variance original_data
def s2_squared : ℝ := variance modified_data

theorem variance_equivalence : s1_squared = s2_squared := by
  sorry

end variance_equivalence_l460_460809


namespace log_eq_two_implication_l460_460537

theorem log_eq_two_implication (x : ℝ) : log 8 (x - 8) = 2 → x = 72 := by
  sorry

end log_eq_two_implication_l460_460537


namespace a_5_value_l460_460205

def sum_of_first_n_terms (n : ℕ) : ℕ := 2 * n^2 + 3 * n - 1

theorem a_5_value :
  let a_5 := sum_of_first_n_terms 5 - sum_of_first_n_terms 4 in
  a_5 = 21
:= by
  let S := sum_of_first_n_terms
  let a_5 := S 5 - S 4
  sorry

end a_5_value_l460_460205


namespace equation_solution_unique_l460_460180

theorem equation_solution_unique (m a b : ℕ) (hm : 1 < m) (ha : 1 < a) (hb : 1 < b) :
  ((m + 1) * a = m * b + 1) ↔ m = 2 :=
sorry

end equation_solution_unique_l460_460180


namespace shaded_area_of_square_with_circles_l460_460843

theorem shaded_area_of_square_with_circles :
  let side_length_square := 12
  let radius_quarter_circle := 6
  let radius_center_circle := 3
  let area_square := side_length_square * side_length_square
  let area_quarter_circles := 4 * (1 / 4) * Real.pi * (radius_quarter_circle ^ 2)
  let area_center_circle := Real.pi * (radius_center_circle ^ 2)
  area_square - area_quarter_circles - area_center_circle = 144 - 45 * Real.pi :=
by
  sorry

end shaded_area_of_square_with_circles_l460_460843


namespace ratio_areas_equal_one_l460_460291

noncomputable def right_triangle_PQR : Type :=
  {P Q R : Type} [MetricSpace P] [MetricSpace Q] [MetricSpace R] -- assuming they are points in some type of metric space
  (PQ QR PR : ℝ) (hpq : PQ = 8) (hqr : QR = 15) (hangle : ∠Q = 90)
  (S T Y : Type)
  (midpoint_S : S = midpoint P Q) (midpoint_T : T = midpoint P R) (intersection_Y : Y = intersection (line Q S) (line R T))
  (area_PS : real) (area_QR : real)

theorem ratio_areas_equal_one (right_triangle_PQR : right_triangle_PQR) :
  area (quadrilateral P S Y T) / area (triangle Q Y R) = 1 := by
  sorry

end ratio_areas_equal_one_l460_460291


namespace john_spending_l460_460869

theorem john_spending : 
  ∃ X : ℝ, (1/2) * X + (1/3) * X + (1/10) * X + 5 = X ∧ X = 75 :=
by 
  use 75
  split
  sorry
  sorry

end john_spending_l460_460869


namespace f_at_pi_over_4_f_interval_max_min_l460_460967

-- Define the function f(x)
def f (x : Real) : Real :=
  2 * Real.sqrt 2 * Real.cos x * Real.sin (x + Real.pi / 4) - 1

-- Prove that f(π/4) = 1
theorem f_at_pi_over_4 : f (Real.pi / 4) = 1 := sorry

-- Prove the maximum and minimum of f(x) on [0, π/2]
theorem f_interval_max_min :
  let max_val := Real.sqrt 2
  let min_val := -1
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≤ max_val) ∧
  (∃ x, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = max_val) ∧
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ min_val) ∧
  (∃ x, x ∈ Set.Icc 0 (Real.pi / 2) ∧ f x = min_val) := sorry

end f_at_pi_over_4_f_interval_max_min_l460_460967


namespace area_of_triangle_side_length_b_l460_460715

-- Define the conditions
variables {a b c : ℝ} {A B C : ℝ}
variables {S1 S2 S3 : ℝ}

-- Given conditions
def conditions_part1 : Prop :=
  -- ∆ABC with sides a, b, c
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A + B + C = π ∧
  -- Areas of equilateral triangles
  S1 = (sqrt (3) / 4) * (a ^ 2) ∧
  S2 = (sqrt (3) / 4) * (b ^ 2) ∧
  S3 = (sqrt (3) / 4) * (c ^ 2) ∧
  -- Given equation involving areas
  S1 - S2 + S3 = sqrt (3) / 2 ∧
  -- sin B
  sin B = 1 / 3

-- Given condition for Part (2)
def conditions_part2 : Prop :=
  conditions_part1 ∧
  -- sin A * sin C
  sin A * sin C = sqrt (2) / 3

-- Part 1: Prove the area of ∆ABC
theorem area_of_triangle (h : conditions_part1) : 
  let ac := (3 * sqrt (2)) / 4 in
  let area := (1 / 2) * ac * (1 / 3) in 
  area = sqrt (2) / 8 :=
sorry

-- Part 2: Prove b = 1 / 2
theorem side_length_b (h : conditions_part2) : b = 1 / 2 :=
sorry

end area_of_triangle_side_length_b_l460_460715


namespace rational_function_sum_l460_460607

noncomputable def r (x : ℝ) : ℝ := x

noncomputable def s (x : ℝ) : ℝ := - (1 / 2) * (x^2 - 3 * x)

theorem rational_function_sum :
  ∀ (x : ℝ), (r 2 = 2) ∧ (s 1 = 1) ∧ (∀ x, x ≠ 3 → s x / (x - 3)) ∧ (∀ x, x ≠ 0 → r x / s x ≠ 0) → 
  r x + s x = - (1 / 2) * x^2 + (5 / 2) * x :=
by
  intro x h,
  sorry

end rational_function_sum_l460_460607


namespace circumradius_of_triangle_l460_460663

theorem circumradius_of_triangle (a b S : ℝ) (A : a = 2) (B : b = 3) (Area : S = 3 * Real.sqrt 15 / 4)
  (median_cond : ∃ c m, m = (a^2 + b^2 - c^2) / (2*a*b) ∧ m < c / 2) :
  ∃ R, R = 8 / Real.sqrt 15 :=
by
  sorry

end circumradius_of_triangle_l460_460663


namespace reciprocal_of_neg_two_l460_460782

theorem reciprocal_of_neg_two : ∀ x : ℝ, x = -2 → (1 / x) = -1 / 2 :=
by
  intro x h
  rw [h]
  norm_num

end reciprocal_of_neg_two_l460_460782


namespace solve_Q1_l460_460166

noncomputable def Q1 (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y + y * f x) = f x + f y + x * f y

theorem solve_Q1 :
  ∀ f : ℝ → ℝ, Q1 f → f = (id : ℝ → ℝ) :=
  by sorry

end solve_Q1_l460_460166


namespace initial_speed_of_car_l460_460456

-- Define the constants and the problem conditions
def initial_speed (v : ℝ) : ℝ :=
  ∀ v0 : ℝ, v0 = v * Real.sqrt 2

-- Main theorem stating that if the speed at the midpoint is 100 km/h, then the initial speed is 141.4 km/h (100 * sqrt(2) approx).
theorem initial_speed_of_car :
  initial_speed 100 = 100 * Real.sqrt 2 :=
by
  sorry

end initial_speed_of_car_l460_460456


namespace range_lambda_l460_460608

variable {λ : ℝ}

/-- Define the sequence a_n satisfying the given recurrence relation. -/
def seq_a : ℕ → ℝ
| 0       := 1
| (n + 1) := seq_a n / (seq_a n + 2)

/-- Define the sequence b_n with given initial condition and recurrence relation. -/
def seq_b : ℕ → ℝ
| 0       := -3 / 2 * λ
| (n + 1) := (n - 2 * λ) * (1 / seq_a n + 1)

/-- The sequence b_n is monotonically increasing (∀ n, seq_b (n + 1) > seq_b n). -/
def monotone_cond (b : ℕ → ℝ) : Prop := ∀ n, b (n + 1) > b n

/-- The main theorem statement: proving the range of λ. -/
theorem range_lambda : (∀ n, seq_b n > seq_b (n - 1)) → λ < 4 / 5 := 
sorry

end range_lambda_l460_460608


namespace proof_median_proof_avg_shifted_proof_var_scaled_l460_460569

variable {a : ℕ → ℝ} (h_strict_inc : ∀ i j, i < j → a i < a j)

def median (n : ℕ) :=
  if odd n then a (n / 2) else (a (n / 2 - 1) + a (n / 2)) / 2

def average (n : ℕ) := (∑ i in finset.range n, a i) / n

noncomputable def variance (n : ℕ) := 
  (∑ i in finset.range n, (a i - average n) ^ 2) / n

variable (n : ℕ) (h_n : n = 2023) 

noncomputable def new_data1 : ℕ → ℝ := λ i, a i + 2
noncomputable def new_data2 : ℕ → ℝ := λ i, 2 * a i + 1

theorem proof_median :
  median a 2023 = a 1012 :=
sorry

theorem proof_avg_shifted :
  average (new_data1 a) 2023 = average a 2023 + 2 :=
sorry

theorem proof_var_scaled :
  variance (new_data2 a) 2023 = 4 * variance a 2023 :=
sorry

end proof_median_proof_avg_shifted_proof_var_scaled_l460_460569


namespace correct_option_is_D_l460_460137

def f_A (x : ℝ) : ℝ := 1 / x
def f_B (x : ℝ) : ℝ := 2 ^ x
def f_C (x : ℝ) : ℝ := Real.log (x^2 + 1)
def f_D (x : ℝ) : ℝ := Real.log (x + 1)

theorem correct_option_is_D :
  (range f_A ≠ set.univ) ∧
  (range f_B ≠ set.univ) ∧
  (range f_C ≠ set.univ) ∧
  (range f_D = set.univ) :=
by
  sorry

end correct_option_is_D_l460_460137


namespace additional_time_due_to_leak_is_six_l460_460106

open Real

noncomputable def filling_time_with_leak (R L : ℝ) : ℝ := 1 / (R - L)
noncomputable def filling_time_without_leak (R : ℝ) : ℝ := 1 / R
noncomputable def additional_filling_time (R L : ℝ) : ℝ :=
  filling_time_with_leak R L - filling_time_without_leak R

theorem additional_time_due_to_leak_is_six :
  additional_filling_time 0.25 (3 / 20) = 6 := by
  sorry

end additional_time_due_to_leak_is_six_l460_460106


namespace garden_wall_additional_courses_l460_460313

theorem garden_wall_additional_courses (initial_courses additional_courses : ℕ) (bricks_per_course total_bricks bricks_removed : ℕ) 
  (h1 : bricks_per_course = 400) 
  (h2 : initial_courses = 3) 
  (h3 : bricks_removed = bricks_per_course / 2) 
  (h4 : total_bricks = 1800) 
  (h5 : total_bricks = initial_courses * bricks_per_course + additional_courses * bricks_per_course - bricks_removed) : 
  additional_courses = 2 :=
by
  sorry

end garden_wall_additional_courses_l460_460313


namespace greatest_value_of_squared_sum_l460_460319

-- Given conditions
variables (a b c d : ℝ)
hypothesis h1 : a + b = 20
hypothesis h2 : ab + c + d = 90
hypothesis h3 : ad + bc = 210
hypothesis h4 : cd = 125

-- Statement to be proved
theorem greatest_value_of_squared_sum :
  a^2 + b^2 + c^2 + d^2 = 1450 :=
sorry

end greatest_value_of_squared_sum_l460_460319


namespace least_roots_in_interval_l460_460467

def f : ℝ → ℝ := sorry -- Define f, fulfilling given conditions

theorem least_roots_in_interval :
  (∀ x : ℝ, f (2 + x) = f (2 - x)) →
  (∀ x : ℝ, f (7 + x) = f (7 - x)) →
  (f 0 = 0) →
  ∃ n : ℕ, n = 401 ∧ ∀ x : ℝ, -1000 ≤ x ∧ x ≤ 1000 → (f x = 0 ↔ (x ∈ {x | x % 10 = 0} ∨ x % 10 = 4)) :=
by
  intros h1 h2 h3
  use 401
  split
  . exact rfl
  sorry

end least_roots_in_interval_l460_460467


namespace part1_costs_part2_comparison_at_200_part3_cost_at_300_l460_460665

-- Variables and parameters
variables (x : ℕ) (hx : x > 50)

-- Definitions based on the conditions
def priceSoccerBall := 80
def priceSolidBall := 20
def storeAdiscountSoccerBalls := 50
def storeAdiscountSolidBalls := 50
def storeBdiscount := 0.90

-- Costs calculations for each store as functions
def storeA_cost (x : ℕ) : ℕ := 50 * priceSoccerBall + (x - storeAdiscountSolidBalls) * priceSolidBall
def storeB_cost (x : ℕ) : ℕ := ((50 * priceSoccerBall + x * priceSolidBall) * storeBdiscount)

-- Proofs of the necessary cost conditions
theorem part1_costs (hx : x > 50) : storeA_cost x = 20 * x + 3000 ∧ storeB_cost x = 3600 + 18 * x := 
sorry

theorem part2_comparison_at_200 : (storeA_cost 200 < storeB_cost 200) :=
sorry

theorem part3_cost_at_300 : 
  let mixed_strategy_cost := 4000 + ((300 - storeAdiscountSolidBalls) * priceSolidBall * storeBdiscount) 
  in mixed_strategy_cost = 8500 :=
sorry

end part1_costs_part2_comparison_at_200_part3_cost_at_300_l460_460665


namespace quadratic_inequality_l460_460999

theorem quadratic_inequality (a : ℝ) :
  (∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by sorry

end quadratic_inequality_l460_460999


namespace term3079_l460_460028

def digitCubesSum (n : Nat) : Nat :=
  (toDigits n).map (fun d => d^3).sum

noncomputable def sequence (n : Nat) : Nat :=
  Nat.rec 3079 (fun _ acc => digitCubesSum acc) n

theorem term3079 : sequence 3079 = 1459 :=
  sorry

end term3079_l460_460028


namespace rook_tour_even_cells_l460_460483

theorem rook_tour_even_cells 
  (m n : ℕ) 
  (even_total : (m * n) % 2 = 0) : 
  ∃ (r : rook), (r.tour_all_squares = true) ∧ (r.return_to_start = true) :=
sorry

end rook_tour_even_cells_l460_460483


namespace correct_statements_count_l460_460553

-- Definitions
def proper_fraction (x : ℚ) : Prop := (0 < x) ∧ (x < 1)
def improper_fraction (x : ℚ) : Prop := (x ≥ 1)

-- Statements as conditions
def statement1 (a b : ℚ) : Prop := proper_fraction a ∧ proper_fraction b → proper_fraction (a + b)
def statement2 (a b : ℚ) : Prop := proper_fraction a ∧ proper_fraction b → proper_fraction (a * b)
def statement3 (a b : ℚ) : Prop := proper_fraction a ∧ improper_fraction b → improper_fraction (a + b)
def statement4 (a b : ℚ) : Prop := proper_fraction a ∧ improper_fraction b → improper_fraction (a * b)

-- The main theorem stating the correct answer
theorem correct_statements_count : 
  (¬ (∀ a b, statement1 a b)) ∧ 
  (∀ a b, statement2 a b) ∧ 
  (∀ a b, statement3 a b) ∧ 
  (¬ (∀ a b, statement4 a b)) → 
  (2 = 2)
:= by sorry

end correct_statements_count_l460_460553


namespace typist_original_salary_l460_460861

theorem typist_original_salary (S : ℝ) (h : (1.12 * 0.93 * 1.15 * 0.90 * S = 5204.21)) : S = 5504.00 :=
sorry

end typist_original_salary_l460_460861


namespace cube_root_of_prime_product_l460_460370

theorem cube_root_of_prime_product : (∛(2^9 * 5^3 * 7^3) = 280) :=
by
  sorry

end cube_root_of_prime_product_l460_460370


namespace rain_probability_correct_l460_460060

noncomputable def probability_rain_at_most_3_days : ℝ :=
  let p := 1 / 5 in
  let q := 4 / 5 in
  let binomial (n k : ℕ) := (Nat.choose n k : ℝ) * p^k * q^(n-k) in
  binomial 30 0 + binomial 30 1 + binomial 30 2 + binomial 30 3

theorem rain_probability_correct :
  abs (probability_rain_at_most_3_days - 0.855) < 0.001 :=
sorry

end rain_probability_correct_l460_460060


namespace coefficient_x3y3_in_expansion_l460_460151

theorem coefficient_x3y3_in_expansion :
  let a := 2
  let b := -1
  let binomial := (a*x + b*y) ^ 5
  let expansion := (x + y) * binomial
  coefficient x^3 y^3 expansion = 40 :=
by
  sorry

end coefficient_x3y3_in_expansion_l460_460151


namespace binary_operation_correct_l460_460511

theorem binary_operation_correct :
  let b1 := 0b11011
  let b2 := 0b1011
  let b3 := 0b11100
  let b4 := 0b10101
  let b5 := 0b1001
  b1 + b2 - b3 + b4 - b5 = 0b11110 := by
  sorry

end binary_operation_correct_l460_460511


namespace focal_length_of_ellipse_l460_460025

theorem focal_length_of_ellipse (k : ℝ) (h : 1 > -k) (e : ℝ) (he : e = 1/2) : 
  ∃ c : ℝ, 2 * c = 1 ∧ c^2 = 1 / 4 ∧ (x y : ℝ, x^2 - y^2 / k = 1 → x^2 / 1 - y^2 / -k = 1) :=
by {
  let a : ℝ := 1,
  let b : ℝ := real.sqrt (-k),
  let c : ℝ := real.sqrt (1 + k),
  have ha : a^2 = 1 := by linarith,
  have hb : b^2 = -k := by linarith,
  have hc : c^2 = 1 + k := by linarith,
  have h_eccentricity : e^2 = (c^2) / (a^2) := by linarith,
  have h_k : 1 + k = 1 / 4 := by linarith,
  have h_c : c^2 = 1 / 4 := by linarith,
  exact ⟨c, by linarith, h_c, by simp [x y]⟩
}

end focal_length_of_ellipse_l460_460025


namespace max_area_of_trapezoid_l460_460130

-- Definitions based on the problem conditions
def radius : ℝ := 13
def distance_from_center : ℝ := 5
def half_diagonal_length : ℝ := Real.sqrt (radius^2 - distance_from_center^2)
def diagonal_length : ℝ := 2 * half_diagonal_length

theorem max_area_of_trapezoid :
  (radius = 13) ∧ (distance_from_center = 5) →
  half_diagonal_length = Real.sqrt (13^2 - 5^2) →
  diagonal_length = 2 * Real.sqrt (13^2 - 5^2) →
  (1/2 * diagonal_length * diagonal_length * Real.sin (Real.pi / 2)) = 288 :=
by {
  intros,
  sorry
}

end max_area_of_trapezoid_l460_460130


namespace number_of_valid_b_is_one_l460_460520

noncomputable def number_of_valid_b : ℕ := 
  let valid_bs : List ℕ := List.filter (λ b, 
    b^2 > 24 ∧ b^2 < 40 ∧ 2 * (Int.abs b) + 24 < 128) (List.range' 5 2) -- integers [5, 6]
  List.length (valid_bs)

theorem number_of_valid_b_is_one : number_of_valid_b = 1 := by
  -- Proof goes here
  sorry

end number_of_valid_b_is_one_l460_460520


namespace cube_root_of_product_is_integer_l460_460363

theorem cube_root_of_product_is_integer :
  (∛(2^9 * 5^3 * 7^3) = 280) :=
sorry

end cube_root_of_product_is_integer_l460_460363


namespace meaningful_expression_range_l460_460994

theorem meaningful_expression_range (a : ℝ) : (a + 1 ≥ 0) ∧ (a ≠ 2) ↔ (a ≥ -1) ∧ (a ≠ 2) :=
by
  sorry

end meaningful_expression_range_l460_460994


namespace win_game_A_win_game_C_l460_460824

-- Define the probabilities for heads and tails
def prob_heads : ℚ := 3 / 4
def prob_tails : ℚ := 1 / 4

-- Define the probability of winning Game A
def prob_win_game_A : ℚ := (prob_heads ^ 3) + (prob_tails ^ 3)

-- Define the probability of winning Game C
def prob_win_game_C : ℚ := (prob_heads ^ 4) + (prob_tails ^ 4)

-- State the theorem for Game A
theorem win_game_A : prob_win_game_A = 7 / 16 :=
by 
  -- Lean will check this proof
  sorry

-- State the theorem for Game C
theorem win_game_C : prob_win_game_C = 41 / 128 :=
by 
  -- Lean will check this proof
  sorry

end win_game_A_win_game_C_l460_460824


namespace bullet_speed_difference_l460_460430

def speed_horse : ℕ := 20  -- feet per second
def speed_bullet : ℕ := 400  -- feet per second

def speed_forward : ℕ := speed_bullet + speed_horse
def speed_backward : ℕ := speed_bullet - speed_horse

theorem bullet_speed_difference : speed_forward - speed_backward = 40 :=
by
  sorry

end bullet_speed_difference_l460_460430


namespace unique_3_letter_sequences_correct_l460_460633

noncomputable def count_unique_3_letter_sequences : ℕ :=
  let letters := "AUGUSTIN LOUIS CAUCHY".to_list,
      frequencies := letters.foldr (λ c m, m.insert c (m.find_d c + 1)) ∅,
      valid_two_different := (3 * 11 * 3),
      valid_three_different := (220 * 6) in
  1 + valid_two_different + valid_three_different

theorem unique_3_letter_sequences_correct :
  count_unique_3_letter_sequences = 1486 := 
    by 
    sorry

end unique_3_letter_sequences_correct_l460_460633


namespace remaining_sessions_l460_460478

theorem remaining_sessions (total_sessions : ℕ) (p1_sessions : ℕ) (p2_sessions_more : ℕ) (remaining_sessions : ℕ) :
  total_sessions = 25 →
  p1_sessions = 6 →
  p2_sessions_more = 5 →
  remaining_sessions = total_sessions - (p1_sessions + (p1_sessions + p2_sessions_more)) →
  remaining_sessions = 8 :=
by
  intros
  sorry

end remaining_sessions_l460_460478


namespace solve_sudoku_l460_460504

open Matrix

-- Defining the Sudoku-like grid
def is_valid_sudoku (grid : Matrix (Fin 6) (Fin 6) (Fin 6)) : Prop :=
  (∀ i : Fin 6, (Finset.univ : Finset (Fin 6)).image (λ j, grid i j) = Finset.univ) ∧
  (∀ j : Fin 6, (Finset.univ : Finset (Fin 6)).image (λ i, grid i j) = Finset.univ) ∧
  (∀ i j : Fin 6, (Fin.os_square i) = (Fin.os_square j) → 
    (Finset.univ : Finset (Fin 6)).image (λ k, grid i k) = Finset.univ)

-- The functions Fin.os_square represents the mapping to the 2x3 regions.
def Fin.os_square (i : Fin 6) : Fin 3 × Fin 2 :=
  match i with
  | ⟨k, h⟩ => (⟨(k / 2), sorry⟩, ⟨(k % 2), sorry⟩)

-- Defining the solution requirement of the bottom row's four-digit number
def four_digit_number (grid : Matrix (Fin 6) (Fin 6) (Fin 6)) : ℕ :=
  (grid 5 0).val * 1000 + (grid 5 1).val * 100 + (grid 5 2).val * 10 + (grid 5 3).val

-- Declare the theorem to prove the four-digit number
theorem solve_sudoku (grid : Matrix (Fin 6) (Fin 6) (Fin 6)) :
  is_valid_sudoku grid → four_digit_number grid = 2413 :=
by
  intros,
  -- Skipping the proof details with sorry
  sorry

end solve_sudoku_l460_460504


namespace maximum_drawn_matches_l460_460894

/-- Given 11 football teams where each plays one match against each other and each team scores
    from 1 to 10 goals in sequential matches, the maximum number of drawn matches is 50. -/
theorem maximum_drawn_matches :
  let num_teams := 11 in
  let total_matches := (num_teams * (num_teams - 1)) / 2 in
  total_matches = 55 →
  ∃ max_draws, max_draws = 50 :=
by
  sorry

end maximum_drawn_matches_l460_460894


namespace estate_division_l460_460448

theorem estate_division {estate : ℕ} (h_estate : estate = 210) :
  let son_share := 120,
      daughter_share := 30,
      mother_share := 60 in
  (son_share + daughter_share + mother_share = estate) ∧
  (son_share = (4 * estate) / 7) ∧
  (daughter_share = estate / 7) ∧
  (mother_share = (2 * estate) / 7) :=
by
  sorry

end estate_division_l460_460448


namespace sum_even_odd_digits_l460_460321

-- Define E(n) and O(n)
def sum_of_digits (pred : ℕ → Prop) (n : ℕ) : ℕ :=
  Nat.digits 10 n |>.filter pred |>.sum

def E (n : ℕ) : ℕ := sum_of_digits (λ d, d % 2 = 0) n
def O (n : ℕ) : ℕ := sum_of_digits (λ d, d % 2 = 1) n

-- The theorem we need to prove
theorem sum_even_odd_digits : (∑ n in Finset.range 201, E n + O n) = 3150 :=
by 
  sorry

end sum_even_odd_digits_l460_460321


namespace domain_of_f_l460_460386

noncomputable def f (x : ℝ) : ℝ := real.sqrt (real.log (5 - x^2))

theorem domain_of_f :
  ∀ x, -2 ≤ x ∧ x ≤ 2 ↔ ∃ x : ℝ, real.sqrt (real.log (5 - x^2)) = real.sqrt (real.log (5 - x^2)) :=
begin
  sorry
end

end domain_of_f_l460_460386


namespace mean_temperature_l460_460017

def temperatures : List Int := [-8, -3, -3, -6, 2, 4, 1]

theorem mean_temperature :
  (temperatures.sum / temperatures.length : Int) = -2 := by
  sorry

end mean_temperature_l460_460017


namespace part_a_part_b_finite_part_b_max_l460_460192

/- Definitions -/
def eq_A2 : set ℤ :=
  {x | x = ⌊↑x / 2⌋}

def eq_A3 : set ℤ :=
  {x | x = ⌊↑x / 2⌋ + ⌊↑x / 3⌋}

/- Proving Part (a) -/
theorem part_a :
  ∀ x : ℤ,
    x ∈ eq_A2 ∨ x ∈ eq_A3 ↔ x ∈ {-7, -5, -4, -3, -2, -1, 0} :=
sorry

/- Definitions and max for Part (b) -/
def eq_An (n : ℕ) (h : 2 ≤ n) : set ℤ :=
  {x | x = ∑ k in finset.range (n - 1), (⌊↑x / k.succ.succ⌋)}

/- Proof of finiteness -/
theorem part_b_finite :
  ∀ n : ℕ, ∃ x : ℤ, x ∈ eq_An n (nat.le_add_left 2 n) →
    ∃ M, ∀ m, n ≤ m → m ∉ eq_An m.succ (nat.le_add_left 2 m.succ) :=
sorry

theorem part_b_max :
  ∃ x ∈ (⋃ n ≥ 2, eq_An n (nat.le_add_left 2 n)), x = 23 :=
sorry

end part_a_part_b_finite_part_b_max_l460_460192


namespace xy_identity_l460_460987

theorem xy_identity (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 24) : (x^2 + y^2) * (x + y) = 803 := by
  sorry

end xy_identity_l460_460987


namespace linear_regression_proof_l460_460103

theorem linear_regression_proof :
  let n := 8
  let Sx := 52
  let Sy := 228
  let Sxx := 478
  let Sxy := 1849
  let x_bar := Sx / n
  let y_bar := Sy / n
  let b := (Sxy - n * x_bar * y_bar) / (Sxx - n * x_bar ^ 2)
  let a := y_bar - b * x_bar
  a = 11.47 ∧ b = 2.62 ∧ (∀ x, a + b * x = 11.47 + 2.62 * x) :=
  by {
    let n := 8
    let Sx := 52
    let Sy := 228
    let Sxx := 478
    let Sxy := 1849 
    let x_bar := Sx / n 
    let y_bar := Sy / n 
    let b := (Sxy - n * x_bar * y_bar) / (Sxx - n * x_bar ^ 2) 
    let a := y_bar - b * x_bar 
    -- Prove that a = 11.47 and b = 2.62
    have ha : a = 11.47 := sorry
    have hb : b = 2.62 := sorry
    have h_eq : ∀ x, a + b * x = 11.47 + 2.62 * x := by {
      intro x
      rw [ha, hb]
    }
    exact ⟨ha, hb, h_eq⟩
  }

end linear_regression_proof_l460_460103


namespace filling_methods_count_l460_460534

noncomputable def count_filling_methods : ℕ :=
  { g : Fin 3 × Fin 3 → Fin 3 | ∀ i j, i ≠ j → g (i, j) ≠ g (j, i) ∧ 
                                        ∀ i, finset.univ.image (λ j, g (i, j)) = finset.univ ∧
                                        ∀ j, finset.univ.image (λ i, g (i, j)) = finset.univ }.to_finset.card

theorem filling_methods_count : count_filling_methods = 12 := 
  by 
    sorry

end filling_methods_count_l460_460534


namespace remainder_5_pow_100_div_18_l460_460453

theorem remainder_5_pow_100_div_18 : (5 ^ 100) % 18 = 13 := 
  sorry

end remainder_5_pow_100_div_18_l460_460453


namespace alpha_sum_l460_460404

noncomputable def P (x : Complex) : Complex :=
  (1 + x + x^2 + ... + x^17)^2 - x^17

theorem alpha_sum : 
  let αs := [α_1, α_2, α_3, α_4, α_5] in
  ∑ αs = 159 / 323 :=
sorry

end alpha_sum_l460_460404


namespace solve_for_x_l460_460408

noncomputable def sum_of_factors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem solve_for_x (x : ℕ) (hx1 : sum_of_factors x = 18) (hx2 : 2 ∣ x) : x = 10 :=
by sorry

end solve_for_x_l460_460408


namespace absolute_value_equation_solution_l460_460789

theorem absolute_value_equation_solution (x : ℝ) (h : x ≠ 1) :
  (| x / (x - 1) | = x / (x - 1)) ↔ (x ≤ 0 ∨ 1 < x) :=
by
  sorry

end absolute_value_equation_solution_l460_460789


namespace solve_inequality_solve_range_of_a_l460_460236

-- Problem 1:
theorem solve_inequality (x : ℝ) : (|x - 1| < |2 * x - 1| - 1) ↔ (x < -1 ∨ x > 1) :=
sorry

-- Problem 2:
theorem solve_range_of_a (a x : ℝ) (h : x ∈ Ioo (-2) 1) :
  (|x - 1| > |2 * x - a - 1| - |x - a|) ↔ (a ≤ -2) :=
sorry

end solve_inequality_solve_range_of_a_l460_460236


namespace hyperbola_standard_equation_1_equilateral_hyperbola_standard_equation_l460_460168

noncomputable def focal_distance := 16
noncomputable def eccentricity_hyperbola1 := (4 / 3)

theorem hyperbola_standard_equation_1 :
  (let c : ℝ := focal_distance / 2 in
   let e : ℝ := eccentricity_hyperbola1 in
   let a : ℝ := c / e in
   let b_squared : ℝ := c^2 - a^2 in
   a = 6 ∧ b_squared = 28 →
   ∀ x y : ℝ, (y^2 / a^2) - (x^2 / b_squared) = 1) :=
sorry

noncomputable def focus_hyperbola2 := (-6, 0)

theorem equilateral_hyperbola_standard_equation :
  (let c : ℝ := -focus_hyperbola2.1 in
   let a_squared : ℝ := c^2 / 2 in
   a_squared = 18 →
   ∀ x y : ℝ, (x^2 / a_squared) - (y^2 / a_squared) = 1) :=
sorry

end hyperbola_standard_equation_1_equilateral_hyperbola_standard_equation_l460_460168


namespace total_number_of_balls_l460_460253

theorem total_number_of_balls (boxes : ℕ) (balls_per_box : ℕ) (hboxes : boxes = 3) (hballs_per_box : balls_per_box = 5) :
  boxes * balls_per_box = 15 :=
by
  rw [hboxes, hballs_per_box]
  exact Nat.mul_comm 3 5

end total_number_of_balls_l460_460253


namespace equal_total_time_l460_460825

variable (a s : ℝ) -- Distances of the asphalt and sand sections
variable (v_aP v_sP v_aV v_sV : ℝ) -- Speeds of Petya and Vasya on the sections

-- Define that Petya and Vasya meet at the midpoint of both sections
def meet_at_midpoints : Prop :=
  (a / 2 / v_aP) + (s / 2 / v_sP) = (a / 2 / v_aV) + (s / 2 / v_sV)

-- Define the total time for Petya to ride the entire path
def total_time_P := a / v_aP + s / v_sP

-- Define the total time for Vasya to ride the entire path
def total_time_V := a / v_aV + s / v_sV

-- State that if Petya and Vasya meet at the midpoints, then they both took the same total time
theorem equal_total_time (h : meet_at_midpoints a s v_aP v_sP v_aV v_sV) : total_time_P a s v_aP v_sP = total_time_V a s v_aV v_sV :=
sorry

end equal_total_time_l460_460825


namespace sine_range_pi_six_to_pi_half_l460_460412

theorem sine_range_pi_six_to_pi_half : 
  ∀ x : ℝ, (π / 6 ≤ x ∧ x ≤ π / 2) → (1 / 2 ≤ sin x ∧ sin x ≤ 1) :=
by
  intros x hx
  sorry

end sine_range_pi_six_to_pi_half_l460_460412


namespace remaining_sessions_l460_460479

theorem remaining_sessions (total_sessions : ℕ) (p1_sessions : ℕ) (p2_sessions_more : ℕ) (remaining_sessions : ℕ) :
  total_sessions = 25 →
  p1_sessions = 6 →
  p2_sessions_more = 5 →
  remaining_sessions = total_sessions - (p1_sessions + (p1_sessions + p2_sessions_more)) →
  remaining_sessions = 8 :=
by
  intros
  sorry

end remaining_sessions_l460_460479


namespace maximum_value_of_m_l460_460216

theorem maximum_value_of_m (x y : ℝ) (hx : x > 1 / 2) (hy : y > 1) : 
    (4 * x^2 / (y - 1) + y^2 / (2 * x - 1)) ≥ 8 := 
sorry

end maximum_value_of_m_l460_460216


namespace number_of_true_propositions_is_2_l460_460230

-- Define all propositions
def proposition_1 : Prop := ¬(if (1:Real) = Real.sin (π/4) then Real.tan (π/4) = 1 else true)
def proposition_2 : Prop := (∀ x : ℝ, Real.sin x ≤ 1) → ∃ x₀ : ℝ, Real.sin x₀ > 1
def proposition_3 : Prop := ∀ φ : ℝ, (φ = π/2 + k * π) → is_even (λ x, Real.sin (2*x + φ))
def proposition_4 : Prop :=
  let p := ∃ x₀ : ℝ, Real.sin x₀ + Real.cos x₀ = 3/2 in
  let q := ∀ α β : ℝ, Real.sin α > Real.sin β → α > β in
  ¬ p ∧ q

-- Define the proof problem
theorem number_of_true_propositions_is_2 : 
  ([proposition_1, proposition_2, proposition_3, proposition_4].count (λ p, p)) = 2 :=
sorry

end number_of_true_propositions_is_2_l460_460230


namespace smallest_power_of_17_not_palindrome_l460_460548

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in
  s = s.reverse

theorem smallest_power_of_17_not_palindrome : ∃ k : ℕ, 17^k = 71405 ∧ ∀ m : ℕ, (m < k ∧ 17^m > 0 ∧ ¬ is_palindrome (17^m)) → false :=
by
  sorry

end smallest_power_of_17_not_palindrome_l460_460548


namespace exp_base_lt_imp_cube_l460_460578

theorem exp_base_lt_imp_cube (a x y : ℝ) (h_a : 0 < a) (h_a1 : a < 1) (h_exp : a^x > a^y) : x^3 < y^3 :=
by
  sorry

end exp_base_lt_imp_cube_l460_460578


namespace derivative_of_y_l460_460541

noncomputable def sqrt (x : ℝ) := x^(1/2)
noncomputable def th (x : ℝ) := Mathlib.tanh x
noncomputable def arctg (x : ℝ) := Mathlib.atan x

noncomputable def y (x : ℝ) : ℝ := 
  (1 / 2) * Mathlib.log ((1 + sqrt (th x)) / (1 - sqrt (th x))) 
  - arctg (sqrt (th x))

theorem derivative_of_y (x : ℝ) : Mathlib.deriv (λ x, y x) = sqrt (th x) :=
by
  sorry

end derivative_of_y_l460_460541


namespace find_total_amount_l460_460639

-- Definitions according to the conditions
def is_proportion (a b c : ℚ) (p q r : ℚ) : Prop :=
  (a * q = b * p) ∧ (a * r = c * p) ∧ (b * r = c * q)

def total_amount (second_part : ℚ) (prop_total : ℚ) : ℚ :=
  second_part / (1/3) * prop_total

-- Main statement to be proved
theorem find_total_amount (second_part : ℚ) (p1 p2 p3 : ℚ)
  (h : is_proportion p1 p2 p3 (1/2 : ℚ) (1/3 : ℚ) (3/4 : ℚ))
  : second_part = 164.6315789473684 → total_amount second_part (19/12 : ℚ) = 65.16 :=
by
  sorry

end find_total_amount_l460_460639


namespace analyze_sine_function_l460_460244

theorem analyze_sine_function (A ω φ : ℝ) (hA : A > 0) (hω : ω > 0) (hφ : |φ| < Real.pi / 2)
  (h_highest_point : (2 : ℝ) ∈ { x | A * Real.sin (ω * x + φ) = √2 })
  (h_intersect_xaxis : (6 : ℝ) ∈ { x | A * Real.sin (ω * x + φ) = 0 }) :
  A = √2 ∧ ω = Real.pi / 8 ∧ φ = Real.pi / 4 :=
sorry

end analyze_sine_function_l460_460244


namespace compare_numbers_l460_460423

theorem compare_numbers :
  2^27 < 10^9 ∧ 10^9 < 5^13 :=
by {
  sorry
}

end compare_numbers_l460_460423


namespace factorize_expression_l460_460530

-- Lean 4 statement for the proof problem
theorem factorize_expression (a b : ℝ) : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l460_460530


namespace projection_solution_l460_460070
open Real

def vector_a := (-3: ℝ, 2: ℝ)
def vector_b := (1: ℝ, 4: ℝ)
def dir_vector := (4: ℝ, 2: ℝ)
def projection_p := (-7 / 5 : ℝ, 14 / 5 : ℝ)

theorem projection_solution : 
  ∃ v : ℝ × ℝ, 
  ∀ p : ℝ × ℝ,
  ((vector_a.fst + v.fst * (vector_b.fst - vector_a.fst), vector_a.snd + v.snd * (vector_b.snd - vector_a.snd)) = p)
  ∧ (p.fst * dir_vector.fst + p.snd * dir_vector.snd = 0) →
  p = projection_p :=
by
  sorry

end projection_solution_l460_460070


namespace number_of_valid_committees_l460_460146

def physics_male_professors := 3
def physics_female_professors := 3
def chemistry_male_professors := 2
def chemistry_female_professors := 2
def biology_male_professors := 2
def biology_female_professors := 3

def total_male_professors := physics_male_professors + chemistry_male_professors + biology_male_professors
def total_female_professors := physics_female_professors + chemistry_female_professors + biology_female_professors

def total_professors := 6
def total_men := 3
def total_women := 3

theorem number_of_valid_committees : 
  -- There are 864 possible committees that can be formed under these conditions
  (choose physics_male_professors 1 * choose physics_female_professors 1 *
   choose chemistry_male_professors 1 * choose chemistry_female_professors 1 *
   choose biology_male_professors 1 * choose biology_female_professors 1) +
  (choose physics_male_professors 2 * choose physics_female_professors 1 *
   (choose chemistry_male_professors 1 * choose chemistry_female_professors 1) *
   (choose biology_male_professors 1 * choose biology_female_professors 1)) *
  3 = 864 := 
sorry

end number_of_valid_committees_l460_460146


namespace valid_subsets_count_l460_460621

open Finset

noncomputable def count_valid_subsets (n : ℕ) : ℕ :=
  ∑ k in range 10, (choose (n - k - 1) k)

theorem valid_subsets_count :
  count_valid_subsets 20 = 17699 :=
by
  unfold count_valid_subsets
  rw [sum]
  sorry -- Steps of the proof would go here

end valid_subsets_count_l460_460621


namespace cube_root_110592_l460_460559

theorem cube_root_110592 :
  (∃ x : ℕ, x^3 = 110592) ∧ 
  10^3 = 1000 ∧ 11^3 = 1331 ∧ 12^3 = 1728 ∧ 13^3 = 2197 ∧ 14^3 = 2744 ∧ 
  15^3 = 3375 ∧ 20^3 = 8000 ∧ 21^3 = 9261 ∧ 22^3 = 10648 ∧ 23^3 = 12167 ∧ 
  24^3 = 13824 ∧ 25^3 = 15625 → 48^3 = 110592 :=
by
  sorry

end cube_root_110592_l460_460559


namespace minimum_groups_l460_460465

theorem minimum_groups (total_players : ℕ) (max_per_group : ℕ)
  (h_total : total_players = 30)
  (h_max : max_per_group = 12) :
  ∃ x y, y ∣ total_players ∧ y ≤ max_per_group ∧ total_players / y = x ∧ x = 3 :=
by {
  sorry
}

end minimum_groups_l460_460465


namespace mean_and_variance_of_y_l460_460932

variable (x : Fin 10 → ℝ)
variable (a : ℝ) (h : a ≠ 0)

def sample_mean (s : Fin 10 → ℝ) : ℝ :=
  (∑ i, s i) / 10

def variance (s : Fin 10 → ℝ) (mean : ℝ) : ℝ :=
  (∑ i, (s i - mean) ^ 2) / 10

theorem mean_and_variance_of_y :
  sample_mean x = 2 →
  variance x 2 = 5 →
  sample_mean (fun i => x i + a) = 2 + a ∧ variance (fun i => x i + a) (2 + a) = 5 :=
by
  intros hmean hvariance
  sorry

end mean_and_variance_of_y_l460_460932


namespace ratio_of_pieces_l460_460092

-- Definitions from the conditions
def total_length : ℝ := 28
def shorter_piece_length : ℝ := 8.000028571387755

-- Derived definition
def longer_piece_length : ℝ := total_length - shorter_piece_length

-- Statement to prove the ratio
theorem ratio_of_pieces : 
  (shorter_piece_length / longer_piece_length) = 0.400000571428571 :=
by
  -- Use sorry to skip the proof
  sorry

end ratio_of_pieces_l460_460092


namespace option_a_option_b_option_c_option_d_l460_460279

noncomputable def f (x : ℝ) := x^2

noncomputable def g (x : ℝ) := 1 / x

noncomputable def h (x : ℝ) := 2 * Real.exp(1) * Real.log x

noncomputable def m (x : ℝ) := f x - g x

theorem option_a : ∀ x ∈ set.Ioo (-1 / Real.cbrt 2) 0, 2 * x + 1 / (x^2) > 0 :=
by sorry

theorem option_b : ∃ (k b : ℝ), b = -4 ∧ ∀ x < 0, f x ≥ k * x + b ∧ g x ≤ k * x + b :=
by sorry

theorem option_c : ¬ ∃ (k : ℝ), (-4 ≤ k) ∧ (k ≤ 1) ∧ ∀ x < 0, f x ≥ k * x + b ∧ g x ≤ k * x + b :=
by sorry

theorem option_d : ∃! (k b : ℝ), k = 2 * Real.sqrt (Real.exp 1) ∧ b = -Real.exp 1 ∧
  ∀ x > 0, f x ≥ k * x + b ∧ h x ≤ k * x + b :=
by sorry

-- Test if all constants are defined correctly
#eval Real.exp 1
#eval Real.sqrt (Real.exp 1)

end option_a_option_b_option_c_option_d_l460_460279


namespace cube_root_of_product_is_280_l460_460361

theorem cube_root_of_product_is_280 : (∛(2^9 * 5^3 * 7^3) = 280) := 
by 
sorry

end cube_root_of_product_is_280_l460_460361


namespace prime_factors_and_divisors_6440_l460_460910

theorem prime_factors_and_divisors_6440 :
  ∃ (a b c d : ℕ), 6440 = 2^a * 5^b * 7^c * 23^d ∧ a = 3 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧
  (a + 1) * (b + 1) * (c + 1) * (d + 1) = 32 :=
by 
  sorry

end prime_factors_and_divisors_6440_l460_460910


namespace ratio_lambda_mu_l460_460678

variable (A B C D : Type) [inner_product_space ℝ A] {P Q: A}
variables {AB AC AD CB CA CD : ℝ}
variable (λ μ : ℝ)

-- Hypotheses
def is_right_triangle (ABC : Type) [euclidean_geometry_triangle ABC] 
    (P Q : A): Prop :=
    ∃ (A B C : A), angle A C B = π / 2

-- Given angle B = 30 degrees
axiom angle_B_30 : angle B A C = π / 6

-- Point D is the intersection point of the angle bisector of ∠BAC and BC
axiom D_bisector : 
    ∃ D : A, angle A D B = angle A D C

-- Given vector AD
axiom vector_AD : P = λ • Q + μ • (A - Q)

-- Convert vector form to scalars
axiom real_coefficients : λ = (1 / 3) ∧ μ = (2 / 3)

-- Prove the ratio λ / μ
theorem ratio_lambda_mu : λ / μ = 1 / 2 :=
    sorry

end ratio_lambda_mu_l460_460678


namespace find_b_l460_460756

-- Given conditions
def varies_inversely (a b : ℝ) := ∃ K : ℝ, K = a * b
def constant_a (a : ℝ) := a = 1500
def constant_b (b : ℝ) := b = 0.25

-- The theorem to prove
theorem find_b (a : ℝ) (b : ℝ) (h_inv: varies_inversely a b)
  (h_a: constant_a a) (h_b: constant_b b): b = 0.125 := 
sorry

end find_b_l460_460756


namespace incorrect_result_l460_460113

-- Define the conditions
def correct_result : ℕ := 555681
def multiplier : ℕ := 987
def n : ℕ := 555681 / 987 -- Computed as per the given correct result and multiplier

-- Define the question to be proved
theorem incorrect_result :
  let incorrect_result := 995681 in
  let altered_digits (d : ℕ) := if d = 9 then true else false in
  n * multiplier = correct_result ∧
  (incorrect_result / 100000 % 10 = 9 ∧ incorrect_result / 10 % 1000000 = correct_result / 10 % 1000000) :=
by sorry

end incorrect_result_l460_460113


namespace root_in_interval_l460_460643

def f (x : ℝ) : ℝ := x^2 + x^(2/3) - 4

theorem root_in_interval (a : ℤ) (h : ∃ m : ℝ, f(m) = 0 ∧ m ∈ (a : ℝ, (a + 1 : ℤ) : ℝ) ) :
  a = 1 ∨ a = -2 :=
sorry

end root_in_interval_l460_460643


namespace find_m_value_l460_460917

theorem find_m_value : ∃ m : ℤ, (2^4 - 3 = 3^3 + m) ∧ m = -14 := by
  use -14
  have h1 : 2^4 = 16 := by norm_num
  have h2 : 3^3 = 27 := by norm_num
  calc
    2^4 - 3 = 16 - 3 := by rw h1
    ... = 13 := by norm_num
    _ = 27 - 14 := by rw h2
    ... = 3^3 + (-14) := by rw h2
  exact ⟨rfl, rfl⟩

end find_m_value_l460_460917


namespace max_expr_l460_460399

-- Definitions for the problem
def interval := set.Icc (-7.5) 7.5
variables (a b c d : ℝ)
#check a ∈ interval
#check b ∈ interval
#check c ∈ interval
#check d ∈ interval

-- The expression to be maximized
def expr (a b c d : ℝ) := a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

-- The theorem statement proving the maximum value is 240
theorem max_expr : 
  a ∈ interval ∧ b ∈ interval ∧ c ∈ interval ∧ d ∈ interval → expr a b c d ≤ 240  := 
  by 
  sorry

end max_expr_l460_460399


namespace proof_B_correct_value_at_largest_negative_integer_l460_460820

-- Define the given polynomial expressions and conditions
def A (x : ℝ) : ℝ := 3 * x ^ 2 - x + 1
def calculatedResult (x : ℝ) : ℝ := 2 * x ^ 2 - 3 * x - 2
def B (x : ℝ) : ℝ := (calculatedResult x) - (A x)

-- Problem 1: Prove B(x) is correctly expressed given the conditions
theorem proof_B_correct (x : ℝ) : B x = -x^2 - 2x - 3 := by
  sorry

-- Problem 2: Calculate the value of the expression when x = -1
def largestNegativeInteger : ℝ := -1

theorem value_at_largest_negative_integer :
  calculatedResult largestNegativeInteger = 3 := by
  sorry

end proof_B_correct_value_at_largest_negative_integer_l460_460820


namespace height_of_triangular_prism_l460_460788

theorem height_of_triangular_prism
  (a : ℝ) -- Side length of the base triangle ABC
  (M N: ℝ × ℝ) -- Midpoints of A₁B₁ and AA₁ respectively
  (projection_BM : ℝ) -- The projection of BM onto C₁N
  (h_proji_BM : projection_BM = a / (2 * sqrt 5)) -- Given projection of BM
  (h_N_mid : N = (a / 2, 0)) -- N being the midpoint of AA₁, assume coordinate system where A₁ is at origin
  (h_M_mid : M = (a / 2, a / (2 * sqrt 3))) -- M being midpoint of A₁B₁, similar assumptions
  : ∃ x : ℝ, x = (2 * a * sqrt 14) / 7 := 
begin
  sorry
end

end height_of_triangular_prism_l460_460788


namespace audit_sampling_is_systematic_l460_460420

def is_systematic_sampling (population_size : Nat) (step : Nat) (initial_index : Nat) : Prop :=
  ∃ (k : Nat), ∀ (n : Nat), n ≠ 0 → initial_index + step * (n - 1) ≤ population_size

theorem audit_sampling_is_systematic :
  ∀ (population_size : Nat) (random_index : Nat),
  population_size = 50 * 50 →  -- This represents the total number of invoices (50% of a larger population segment)
  random_index < 50 →         -- Randomly selected index from the first 50 invoices
  is_systematic_sampling population_size 50 random_index := 
by
  intros
  sorry

end audit_sampling_is_systematic_l460_460420


namespace store_loss_l460_460486

noncomputable def original_price1 (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  selling_price / (1 + profit_percent)

noncomputable def original_price2 (selling_price : ℝ) (loss_percent : ℝ) : ℝ :=
  selling_price / (1 - loss_percent)

theorem store_loss (selling_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : 
  selling_price = 90 ∧ profit_percent = 0.2 ∧ loss_percent = 0.2 →
  (original_price1 selling_price profit_percent + original_price2 selling_price loss_percent - 2 * selling_price) = - 7.5 :=
by
  intro h
  cases h with h1 h_rest
  cases h_rest with h2 h3
  rw [h1, h2, h3]
  sorry

end store_loss_l460_460486


namespace dessert_probability_l460_460084

noncomputable def P (e : Prop) : ℝ := sorry

variables (D C : Prop)

theorem dessert_probability 
  (P_D : P D = 0.6)
  (P_D_and_not_C : P (D ∧ ¬C) = 0.12) :
  P (¬ D) = 0.4 :=
by
  -- Proof is skipped using sorry, as instructed.
  sorry

end dessert_probability_l460_460084


namespace remainder_of_functions_mod_1000_l460_460691

noncomputable def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def number_of_functions := 
  let N := 8 * ∑ k in (Finset.range 7).filter (λ k, k > 0), (Nat.choose 7 k) * (k ^ (7 - k)) 
  in N

theorem remainder_of_functions_mod_1000 : number_of_functions % 1000 = 576 :=
by 
  let N := number_of_functions
  have h1 : N = 50576 := sorry
  have h2 : 50576 % 1000 = 576 := rfl
  rw [h1, h2]

end remainder_of_functions_mod_1000_l460_460691


namespace repeating_decimal_sum_l460_460039

theorem repeating_decimal_sum :
  (∃ (c d : ℕ), (5 / 13 : ℚ) = c / 10 + d / 100 + (c / 1000 + d / 10000 + c / 100000 + d / 1000000 + ...)) →
  c + d = 11 :=
by
 sorry

end repeating_decimal_sum_l460_460039


namespace find_snail_square_l460_460341

def is_snail (x : ℕ) : Prop :=
  let digits := (to_digits x).to_multiset
  ∃ a b c : ℕ, (a + 2 = b + 1 ∧ b + 1 = c)
    ∧ (digits = (to_digits a).to_multiset ∪ (to_digits b).to_multiset ∪ (to_digits c).to_multiset)

theorem find_snail_square : ∃ n : ℕ, 1000 ≤ n^2 ∧ n^2 < 10000 ∧ is_snail (n^2) ∧ ∃ m, n^2 = m^2 :=
by
  sorry

end find_snail_square_l460_460341


namespace proof_f_neg_a_l460_460600

def f (x : ℝ) : ℝ := (x^2 + x + 1) / (x^2 + 1)

theorem proof_f_neg_a (a : ℝ) (h : f a = 2 / 3) : f (-a) = 4 / 3 :=
by {
  -- proof will go here
  sorry
}

end proof_f_neg_a_l460_460600


namespace smallest_square_value_l460_460328

theorem smallest_square_value (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h₁ : ∃ r : ℕ, 15 * a + 16 * b = r^2) (h₂ : ∃ s : ℕ, 16 * a - 15 * b = s^2) :
  ∃ (m : ℕ), m = 481^2 ∧ (15 * a + 16 * b = m ∨ 16 * a - 15 * b = m) :=
  sorry

end smallest_square_value_l460_460328


namespace T_2022_eq_2022_pow_4_not_exists_prime_n_T_eq_p_pow_2022_l460_460914

def T (n : ℕ) : ℕ := 
  ∏ d in (finset.range (n+1)).filter (λ x, n % x = 0), d 

theorem T_2022_eq_2022_pow_4 :
  T 2022 = 2022 ^ 4 :=
sorry

theorem not_exists_prime_n_T_eq_p_pow_2022 :
  ¬ ∃ (p : ℕ) (n : ℕ), (nat.prime p) ∧ (T n = p ^ 2022) :=
sorry

end T_2022_eq_2022_pow_4_not_exists_prime_n_T_eq_p_pow_2022_l460_460914


namespace f_2015_value_l460_460220

noncomputable def f : ℝ → ℝ := sorry -- Define f with appropriate conditions

theorem f_2015_value :
  (∀ x, f x = -f (-x)) ∧
  (∀ x, f (x + 4) = f x) ∧
  (∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2) →
  f 2015 = -2 :=
by
  sorry -- Proof to be provided

end f_2015_value_l460_460220


namespace shells_in_afternoon_l460_460345

-- Conditions: Lino picked up 292 shells in the morning and 616 shells in total.
def shells_in_morning : ℕ := 292
def total_shells : ℕ := 616

-- Theorem: The number of shells Lino picked up in the afternoon is 324.
theorem shells_in_afternoon : (total_shells - shells_in_morning) = 324 := 
by sorry

end shells_in_afternoon_l460_460345


namespace tan_theta_even_function_range_theta_monotonic_l460_460243

variable (θ : ℝ)

-- Conditions
def f (x : ℝ) (θ : ℝ) : ℝ := x^2 + 4 * (Real.sin (θ + Real.pi / 3)) * x - 2
def domain_theta := 0 ≤ θ ∧ θ < 2 * Real.pi

-- Question 1: Prove tan θ = -√3 given f(x) is an even function
theorem tan_theta_even_function (h_even : ∀ x, f x θ = f (-x) θ) (h_domain : domain_theta θ) : Real.tan θ = -Real.sqrt 3 := 
sorry

-- Question 2: Prove the range of θ for f(x) monotonically increasing over [-√3, 1]
theorem range_theta_monotonic (h_mono : ∀ x y, -Real.sqrt 3 ≤ x → x ≤ y → y ≤ 1 → f x θ = f y θ → x = y) (h_domain : domain_theta θ) : 
(0 ≤ θ ∧ θ ≤ Real.pi / 3) ∨ (5 * Real.pi / 6 ≤ θ ∧ θ ≤ 3 * Real.pi / 2) :=
sorry

end tan_theta_even_function_range_theta_monotonic_l460_460243


namespace sqrt_sum_sin_six_eq_neg_two_cos_three_l460_460153

theorem sqrt_sum_sin_six_eq_neg_two_cos_three :
  sqrt(1 + sin 6) + sqrt(1 - sin 6) = -2 * cos 3 :=
by
  -- The detailed proof steps are omitted
  sorry

end sqrt_sum_sin_six_eq_neg_two_cos_three_l460_460153


namespace probability_correct_l460_460496

def fair_coin_labeled := {15, 25}
def six_sided_die := {1, 2, 3, 4, 5, 6}
def alex_age : ℕ := 18

noncomputable def probability_sum_equals_age : ℚ :=
  let coin_prob := 1 / 2
  let die_prob := 1 / 6
  in if 15 + 3 = alex_age then coin_prob * die_prob else 0

theorem probability_correct : probability_sum_equals_age = 1 / 12 :=
by
  sorry

end probability_correct_l460_460496


namespace multiples_3_or_5_not_6_l460_460629

theorem multiples_3_or_5_not_6 (n : ℕ) (hn : n ≤ 200) :
  card ({m | m ∣ n ∧ m ≤ 200 ∧ ((m % 3 = 0 ∨ m % 5 = 0) ∧ ¬ (m % 6 = 0))}) = 73 := sorry

end multiples_3_or_5_not_6_l460_460629


namespace jen_shooting_game_times_l460_460373

theorem jen_shooting_game_times (x : ℕ) (h1 : 5 * x + 9 = 19) : x = 2 := by
  sorry

end jen_shooting_game_times_l460_460373


namespace line_in_plane_skew_to_l_l460_460117

noncomputable def intersects_and_not_perpendicular {α : Type*} (l : Line α) (α : Plane α) :=
  intersects l α ∧ ¬ perpendicular l α

theorem line_in_plane_skew_to_l {α : Type*} (l : Line α) (α : Plane α) :
  intersects_and_not_perpendicular l α → ∃ (lines_in_plane : set (Line α)), ∀ k ∈ lines_in_plane, skew k l :=
sorry

end line_in_plane_skew_to_l_l460_460117


namespace solution_set_l460_460233

def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem solution_set :
  {x | f x ≥ x^2 - 8 * x + 15} = {2} ∪ {x | x > 6} :=
by
  sorry

end solution_set_l460_460233


namespace find_b_and_lambda_l460_460575

open Real

noncomputable def circle₀ := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }

def A := (-2 : ℝ, 0 : ℝ)

-- Distance between two points in R²
def dist (p q : ℝ × ℝ) : ℝ :=
  ((p.1 - q.1)^2 + (p.2 - q.2)^2).sqrt

theorem find_b_and_lambda (b λ : ℝ) :
  b ≠ -2 →
  (∀ p ∈ circle₀, dist p (b, 0) = λ * dist p A) →
  b = -1/2 ∧ λ = 1/2 :=
begin
  intros h1 h2,
  sorry
end

end find_b_and_lambda_l460_460575


namespace max_n_divisible_groups_l460_460889

theorem max_n_divisible_groups (n : ℕ) 
  (hn : n = 14)
  (A B : Finset ℕ) 
  (hA : A ∩ B = ∅)
  (hAB : A ∪ B = Finset.range (n + 1) \ 0)
  (h_sum_not_square_A : ∀ (x y ∈ A), x ≠ y → ¬∃ m : ℕ, m^2 = x + y)
  (h_sum_not_square_B : ∀ (x y ∈ B), x ≠ y → ¬∃ m : ℕ, m^2 = x + y) : 
  n = 14 :=
by
  sorry

end max_n_divisible_groups_l460_460889


namespace probability_of_color_change_l460_460845

def traffic_light_cycle := 90
def green_duration := 45
def yellow_duration := 5
def red_duration := 40
def green_to_yellow := green_duration
def yellow_to_red := green_duration + yellow_duration
def red_to_green := traffic_light_cycle
def observation_interval := 4
def valid_intervals := [green_to_yellow - observation_interval + 1, green_to_yellow, 
                        yellow_to_red - observation_interval + 1, yellow_to_red, 
                        red_to_green - observation_interval + 1, red_to_green]
def total_valid_intervals := valid_intervals.length * observation_interval

theorem probability_of_color_change : 
  (total_valid_intervals : ℚ) / traffic_light_cycle = 2 / 15 := 
by
  sorry

end probability_of_color_change_l460_460845


namespace football_outcomes_l460_460787

theorem football_outcomes : 
  ∃ (W D L : ℕ), (3 * W + D = 19) ∧ (W + D + L = 14) ∧ 
  ((W = 3 ∧ D = 10 ∧ L = 1) ∨ 
   (W = 4 ∧ D = 7 ∧ L = 3) ∨ 
   (W = 5 ∧ D = 4 ∧ L = 5) ∨ 
   (W = 6 ∧ D = 1 ∧ L = 7)) ∧
  (∀ W' D' L' : ℕ, (3 * W' + D' = 19) → (W' + D' + L' = 14) → 
    (W' = 3 ∧ D' = 10 ∧ L' = 1) ∨ 
    (W' = 4 ∧ D' = 7 ∧ L' = 3) ∨ 
    (W' = 5 ∧ D' = 4 ∧ L' = 5) ∨ 
    (W' = 6 ∧ D' = 1 ∧ L' = 7)) := 
sorry

end football_outcomes_l460_460787


namespace csc_product_l460_460808

theorem csc_product (m n : ℕ) (h1 : m > 1) (h2 : n > 1) :
  (∏ k in Finset.range 60, (Real.csc (3 * (k + 1) - 1 : ℝ))^2) = (m ^ n) → m + n = 121 :=
by
  -- We will provide the proof in another step.
  sorry

end csc_product_l460_460808


namespace end_behavior_of_f_l460_460162

noncomputable def f (x : ℝ) : ℝ := -3 * x^3 + 4 * x^2 + 1

theorem end_behavior_of_f :
  (tendsto f at_top at_bot) ∧ (tendsto f at_bot at_top) :=
by
  -- Proof omitted
  sorry

end end_behavior_of_f_l460_460162


namespace arithmetic_geometric_mean_l460_460358

theorem arithmetic_geometric_mean (a b : ℝ) 
  (h1 : (a + b) / 2 = 20) 
  (h2 : Real.sqrt (a * b) = Real.sqrt 135) : 
  a^2 + b^2 = 1330 :=
by
  sorry

end arithmetic_geometric_mean_l460_460358


namespace accelerations_correct_l460_460422

def mass1 := 1 -- kg
def mass2 := 10 -- kg
def mu1 := 0.3 -- coefficient of friction for m1
def mu2 := 0.1 -- coefficient of friction for m2
def force := 20 -- Newton
def g := 10 -- m/s^2

def friction_force (mu : ℝ) (mass : ℝ) : ℝ :=
  mu * mass * g

def a1 (T : ℝ) (F_tr1 : ℝ) : ℝ :=
  (T - F_tr1) / mass1

def a2 (T : ℝ) (F_tr2 : ℝ) : ℝ :=
  0

def a_center_of_mass (a1 : ℝ) (a2 : ℝ) : ℝ :=
  (mass1 * a1 + mass2 * a2) / (mass1 + mass2)

theorem accelerations_correct :
  let T := force / 2 in
  let F_tr1 := friction_force mu1 mass1 in
  let F_tr2 := friction_force mu2 mass2 in
  a1 T F_tr1 = 7 ∧
  a2 T F_tr2 = 0 ∧
  a_center_of_mass (a1 T F_tr1) (a2 T F_tr2) = 7 / 11 := by
  sorry

end accelerations_correct_l460_460422


namespace prove_sonika_initial_deposit_l460_460015

-- Define the conditions
def sonika_initial_deposit_problem
    (P R : ℝ) -- Principal amount P and interest rate R%
    (T : ℝ)  -- Time period T (years)
    (A1 A2 : ℝ) -- Amounts after 3 years at original and increased interest rates
    (original_rate_interest : A1 = P * (1 + 3 * R / 100))
    (increased_rate_interest : A2 = P * (1 + 3 * (R + 1) / 100)) : Prop :=
  A1 = 9200 ∧ A2 = 9440 → P = 8000

-- Formal statement of the proof problem
theorem prove_sonika_initial_deposit :
  ∃ (P R : ℝ), ∃ (T : ℝ), 
  T = 3 → 
  sonika_initial_deposit_problem P R T 9200 9440 := 
by
  sorry

end prove_sonika_initial_deposit_l460_460015


namespace window_division_l460_460010

theorem window_division :
  ∃ (regions : list (set (ℝ × ℝ))),
    (∀ region ∈ regions, measurable_set region ∧ measure_theory.measure_space.measure region = 0.125) ∧
    (⋃₀ regions = set.univ ∩ {p | p.1 ≤ 1 ∧ p.2 ≤ 1}) ∧
    #regions = 8 := 
sorry

end window_division_l460_460010


namespace total_blocks_needed_l460_460097

theorem total_blocks_needed (length height : ℕ) (block_height : ℕ) (block1_length block2_length : ℕ)
                            (height_blocks : height = 8) (length_blocks : length = 102)
                            (block_height_cond : block_height = 1)
                            (block_lengths : block1_length = 2 ∧ block2_length = 1)
                            (staggered_cond : True) (even_ends : True) :
  ∃ total_blocks, total_blocks = 416 := 
  sorry

end total_blocks_needed_l460_460097


namespace abc_sum_eq_13_l460_460726

-- Define the polynomial and its roots
def polynomial : Polynomial ℝ := Polynomial.C 11 * Polynomial.X^2 - Polynomial.C 6 * Polynomial.X + Polynomial.C 1

-- Define the sequence s_k
def s (k : ℕ) : ℝ :=
  if k = 0 then 3
  else if k = 1 then 6
  else if k = 2 then 11
  else s k = a * s (k - 1) + b * s (k - 2) + c * s (k - 3)

-- State the theorem
theorem abc_sum_eq_13 :
  let a := 6
  let b := -11
  let c := 18
  a + b + c = 13 :=
by
  sorry

end abc_sum_eq_13_l460_460726


namespace value_of_Y_l460_460286

/- Define the conditions given in the problem -/
def first_row_arithmetic_seq (a1 d1 : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d1
def fourth_row_arithmetic_seq (a4 d4 : ℕ) (n : ℕ) : ℕ := a4 + (n - 1) * d4

/- Constants given by the problem -/
def a1 : ℕ := 3
def fourth_term_first_row : ℕ := 27
def a4 : ℕ := 6
def fourth_term_fourth_row : ℕ := 66

/- Calculating common differences for first and fourth rows -/
def d1 : ℕ := (fourth_term_first_row - a1) / 3
def d4 : ℕ := (fourth_term_fourth_row - a4) / 3

/- Note that we are given that Y is at position (2, 2)
   Express Y in definition forms -/
def Y_row := first_row_arithmetic_seq (a1 + d1) d4 2
def Y_column := fourth_row_arithmetic_seq (a4 + d4) d1 2

/- Problem statement in Lean 4 -/
theorem value_of_Y : Y_row = 35 ∧ Y_column = 35 := by
  sorry

end value_of_Y_l460_460286


namespace cost_per_coffee_before_l460_460304

def cost_of_coffee_machine := 200
def discount := 20
def daily_cost_of_making_coffee := 3
def coffees_per_day := 2
def days_for_machine_to_pay_for_itself := 36

/-- Prove that the cost per coffee before James started making his own coffee is $4,
    given the conditions provided. -/
theorem cost_per_coffee_before (cost_of_coffee_machine discount daily_cost_of_making_coffee
                              coffees_per_day days_for_machine_to_pay_for_itself : ℕ) :
  let total_cost_after_discount := cost_of_coffee_machine - discount
      total_making_cost := daily_cost_of_making_coffee * days_for_machine_to_pay_for_itself
      total_spent := total_cost_after_discount + total_making_cost
      total_coffees := days_for_machine_to_pay_for_itself * coffees_per_day
      cost_per_coffee := total_spent / total_coffees
  in cost_per_coffee = 4 :=
by
  sorry

end cost_per_coffee_before_l460_460304


namespace union_of_A_and_B_intersection_of_A_and_complementB_l460_460613

section

variables (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)
local notation "ℝ" => Real

def setA : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def setB : Set ℝ := {x | x^2 - 3x - 4 > 0}
def complementB : Set ℝ := {x | ¬(x^2 - 3x - 4 > 0)}

theorem union_of_A_and_B : 
  setA U ⊆ ℝ →
  setB U ⊆ ℝ →
  (setA U ∪ setB U) = {x | x ≤ 3 ∨ x > 4} :=
by
  sorry

theorem intersection_of_A_and_complementB : 
  setA U ⊆ ℝ →
  setB U ⊆ ℝ →
  (setA U ∩ complementB U) = {x | -1 ≤ x ∧ x ≤ 3} :=
by 
  sorry

end

end union_of_A_and_B_intersection_of_A_and_complementB_l460_460613


namespace small_boxes_in_large_box_l460_460116

theorem small_boxes_in_large_box (chocolates_per_box : ℕ) (total_chocolates : ℕ) 
    (h1 : chocolates_per_box = 20) (h2 : total_chocolates = 300) :
    total_chocolates / chocolates_per_box = 15 :=
by
  rw [h1, h2]
  norm_num
  sorry

end small_boxes_in_large_box_l460_460116


namespace intersection_sets_l460_460612

theorem intersection_sets :
  let A := { x : ℝ | x^2 - 1 ≥ 0 }
  let B := { x : ℝ | 1 ≤ x ∧ x < 3 }
  A ∩ B = { x : ℝ | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_sets_l460_460612


namespace ellipse_equation_and_line_inclination_range_l460_460574

theorem ellipse_equation_and_line_inclination_range :
  ∀ (l : ℝ → ℝ → Prop),
  ∃ (a b : ℝ), -- these will correspond to the semi-major and semi-minor axes
    a > b ∧ b > 0 ∧
    (l = (λ x y, y = F3 x a b)) ∧ -- Define the equation of the line
    ( ∀ (k m : ℝ), 
      (k ≠ 0 ∧ 2 * (-(1/2):ℝ) = -((2 * k * m) / (k^2 + 9)) ∧
      ∀ (A B : ℝ × ℝ),
        (k^2 - m^2 + 9 > 0 ∧ k^4 + 6 * k^2 - 27 > 0) → 
        θ ∈ range_of_angle_inclination k) :=
sorry

end ellipse_equation_and_line_inclination_range_l460_460574


namespace spending_difference_l460_460519

-- Define the cost of the candy bar
def candy_bar_cost : ℕ := 6

-- Define the cost of the chocolate
def chocolate_cost : ℕ := 3

-- Prove the difference between candy_bar_cost and chocolate_cost
theorem spending_difference : candy_bar_cost - chocolate_cost = 3 :=
by
    sorry

end spending_difference_l460_460519


namespace cone_lateral_surface_area_l460_460566

theorem cone_lateral_surface_area (r l : ℝ) (h_r : r = 2) (h_l : l = 6) : 
  let L := Real.pi * r * l in
  L = 12 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l460_460566


namespace solution_set_of_inequality_l460_460337

variable {f : ℝ → ℝ}

noncomputable def F (x : ℝ) : ℝ := x^2 * f x

theorem solution_set_of_inequality
  (h_diff : ∀ x < 0, DifferentiableAt ℝ f x) 
  (h_cond : ∀ x < 0, 2 * f x + x * (deriv f x) > x^2) :
  ∀ x, ((x + 2016)^2 * f (x + 2016) - 9 * f (-3) < 0) ↔ (-2019 < x ∧ x < -2016) :=
by
  sorry

end solution_set_of_inequality_l460_460337


namespace choosing_top_cases_l460_460048

def original_tops : Nat := 2
def bought_tops : Nat := 4
def total_tops : Nat := original_tops + bought_tops

theorem choosing_top_cases : total_tops = 6 := by
  sorry

end choosing_top_cases_l460_460048


namespace jacob_younger_than_michael_l460_460681

-- Definitions based on the conditions.
def jacob_current_age : ℕ := 9
def michael_current_age : ℕ := 2 * (jacob_current_age + 3) - 3

-- Theorem to prove that Jacob is 12 years younger than Michael.
theorem jacob_younger_than_michael : michael_current_age - jacob_current_age = 12 :=
by
  -- Placeholder for proof
  sorry

end jacob_younger_than_michael_l460_460681


namespace fraction_identity_l460_460649

theorem fraction_identity
  (x w y z : ℝ)
  (hxw_pos : x * w > 0)
  (hyz_pos : y * z > 0)
  (hxw_inv_sum : 1 / x + 1 / w = 20)
  (hyz_inv_sum : 1 / y + 1 / z = 25)
  (hxw_inv : 1 / (x * w) = 6)
  (hyz_inv : 1 / (y * z) = 8) :
  (x + y) / (z + w) = 155 / 7 :=
by
  -- proof omitted
  sorry

end fraction_identity_l460_460649


namespace max_servings_l460_460480

-- Definitions based on the conditions
def servings_recipe := 3
def bananas_per_serving := 2 / servings_recipe
def strawberries_per_serving := 1 / servings_recipe
def yogurt_per_serving := 2 / servings_recipe

def emily_bananas := 4
def emily_strawberries := 3
def emily_yogurt := 6

-- Prove that Emily can make at most 6 servings while keeping the proportions the same
theorem max_servings :
  min (emily_bananas / bananas_per_serving) 
      (min (emily_strawberries / strawberries_per_serving) 
           (emily_yogurt / yogurt_per_serving)) = 6 := sorry

end max_servings_l460_460480


namespace find_value_of_expression_l460_460959

theorem find_value_of_expression (a x y : ℝ)
  (h1 : x = (a + 3)^2)
  (h2 : x = (2a - 15)^2)
  (h3 : real.cbrt (x + y - 2) = 4) :
  x - 2 * y + 2 = 17 :=
by sorry

end find_value_of_expression_l460_460959


namespace vasya_time_from_home_to_school_l460_460672

variable (d v : ℝ)
constant t_walking : ℝ := d / v -- Time spent walking

theorem vasya_time_from_home_to_school (h : (d / v) = (d / (2 * v) + (1 / 12) + d / (4 * v))) :
  t_walking = 1 / 3 := 
sorry

end vasya_time_from_home_to_school_l460_460672


namespace max_error_in_area_estimation_l460_460109

noncomputable def diameter : ℝ := 6.4
noncomputable def max_error_diameter : ℝ := 0.05
noncomputable def area (x : ℝ) : ℝ := (1 / 4) * π * x^2
noncomputable def max_error_area (x : ℝ) (dx : ℝ) : ℝ :=
  ((1 / 2) * π * x) * dx

theorem max_error_in_area_estimation :
  let dS := max_error_area diameter max_error_diameter in
  abs (dS - 0.5024) < 0.001 :=
by
  sorry

end max_error_in_area_estimation_l460_460109


namespace range_x_minus_y_compare_polynomials_l460_460818

-- Proof Problem 1: Range of x - y
theorem range_x_minus_y (x y : ℝ) (hx : -1 < x ∧ x < 4) (hy : 2 < y ∧ y < 3) : 
  -4 < x - y ∧ x - y < 2 := 
  sorry

-- Proof Problem 2: Comparison of polynomials
theorem compare_polynomials (x : ℝ) : 
  (x - 1) * (x^2 + x + 1) < (x + 1) * (x^2 - x + 1) := 
  sorry

end range_x_minus_y_compare_polynomials_l460_460818


namespace integer_points_covered_l460_460755

theorem integer_points_covered (a b : ℝ) (h : b - a = 2020) :
  ∃ n : ℕ, n = 2020 ∨ n = 2021 :=
by 
  let ⟨m, hm⟩ := Classical.em (∃ m : ℤ, (a : ℝ) = m) -- a is an integer point
  if hm then
    let n := 2021
    exists_n : ℕ 
    use n
    left
    refl
  else
    let n := 2020
    exists_n : ℕ 
    use n
    right
    refl
  sorry

end integer_points_covered_l460_460755


namespace min_fraction_sum_l460_460242

theorem min_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 2) :
  ( ∃ c, (∀ a b, 
    a > 0 →
    b > 0 →
    a + 2 * b = 2 →
    c = 2 / (a + 1) + 1 / b ) ∧ c = 8/3) :=
begin
  sorry
end

end min_fraction_sum_l460_460242


namespace cost_price_of_article_l460_460080

-- Definitions based on the conditions
def sellingPrice : ℝ := 800
def profitPercentage : ℝ := 25

-- Statement to prove the cost price
theorem cost_price_of_article :
  ∃ cp : ℝ, profitPercentage = ((sellingPrice - cp) / cp) * 100 ∧ cp = 640 :=
by
  sorry

end cost_price_of_article_l460_460080


namespace cyclic_sum_inequality_l460_460731

open Real

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  1 / (1 + a + b) + 1 / (1 + b + c) + 1 / (1 + c + a) ≤ 1 := 
  sorry

end cyclic_sum_inequality_l460_460731


namespace candy_unclaimed_fraction_is_l460_460865

theorem candy_unclaimed_fraction_is (total_candy : ℕ)
  (ratio_Al ratio_Bert ratio_Carl ratio_Dana : ℕ)
  (total_ratio : ratio_Al + ratio_Bert + ratio_Carl + ratio_Dana = 10)
  (total_candy' : total_candy > 0) :
  let Al_candy := (ratio_Al / total_ratio : ℚ) * total_candy,
      Bert_candy := (ratio_Bert / total_ratio : ℚ) * (total_candy - Al_candy),
      Carl_candy := (ratio_Carl / total_ratio : ℚ) * (total_candy - Al_candy - Bert_candy),
      Dana_candy := (ratio_Dana / total_ratio : ℚ) * (total_candy - Al_candy - Bert_candy - Carl_candy)
  in
  (total_candy - (Al_candy + Bert_candy + Carl_candy + Dana_candy)) / total_candy = 46 / 125 :=
by
  sorry

end candy_unclaimed_fraction_is_l460_460865


namespace minimum_cans_needed_l460_460096

theorem minimum_cans_needed (h : ∀ c, c * 10 ≥ 120) : ∃ c, c = 12 :=
by
  sorry

end minimum_cans_needed_l460_460096


namespace slope_perpendicular_is_neg5_l460_460065

-- Define the points (3, 5) and (-2, 4)
def P1 := (3, 5 : ℝ × ℝ)
def P2 := (-2, 4 : ℝ × ℝ)

-- Function to calculate the slope of the line passing through two points
def slope (A B : ℝ × ℝ) : ℝ := (B.2 - A.2) / (B.1 - A.1)

-- Define the slope of the line passing through P1 and P2
def slope_P1_P2 : ℝ := slope P1 P2

-- Define the slope of the line perpendicular to the slope of the line through P1 and P2
def slope_perpendicular : ℝ := -1 / slope_P1_P2

-- Problem statement: Prove that slope_perpendicular is -5
theorem slope_perpendicular_is_neg5 : slope_perpendicular = -5 := 
sorry

end slope_perpendicular_is_neg5_l460_460065


namespace diagonals_from_vertex_of_regular_polygon_l460_460661

-- Definitions for the conditions in part a)
def exterior_angle (n : ℕ) : ℚ := 360 / n

-- Proof problem statement
theorem diagonals_from_vertex_of_regular_polygon
  (n : ℕ)
  (h1 : exterior_angle n = 36)
  : n - 3 = 7 :=
by sorry

end diagonals_from_vertex_of_regular_polygon_l460_460661


namespace solution_to_absolute_value_equation_l460_460042

theorem solution_to_absolute_value_equation (x : ℝ) : 
    abs x - 2 - abs (-1) = 2 ↔ x = 5 ∨ x = -5 :=
by
  sorry

end solution_to_absolute_value_equation_l460_460042


namespace a_36_eq_131_l460_460324

theorem a_36_eq_131 : ∃ (a : ℕ) (n : ℕ → ℕ), 
  (∀ r s t : ℤ, 0 ≤ t ∧ t < s ∧ s < r → 
  n (2^r + 2^s + 2^t) ∧ 1 ≤ a ∧ n 7 = 1 ∧ n 11 = 2 ∧ n 13 = 3 ∧ n 14 = 4) → 
  n 131 = 36 :=
sorry

end a_36_eq_131_l460_460324


namespace color_n_edges_implies_monochromatic_triangle_l460_460159

-- Definitions based on the conditions:
def points : Type := fin 9 -- Representing the 9 points in space.
def edge (p1 p2 : points) := true -- Every pair of points is connected by an edge.

def color := {c : fin 3 // c.val = 0 ∨ c.val = 1 ∨ c.val = 2} -- Colors: 0 for uncolored, 1 for blue, 2 for red.

noncomputable def edge_color : points → points → color
  | p1 p2 := sorry -- Edge coloring function, definition omitted.

-- The theorem statement based on the question and the correct answer:
theorem color_n_edges_implies_monochromatic_triangle (n : ℕ) 
  (h : ∀ p1 p2, edge_color p1 p2 ∈ ({1, 2} : set fin 3) → ∃ p3 p4, edge_color p3 p4 = edge_color p4 p1 = edge_color p1 p4) 
  : n = 33 :=
sorry

end color_n_edges_implies_monochromatic_triangle_l460_460159


namespace number_of_multiples_of_15_in_range_l460_460256

-- Define a predicate to check if a number is a multiple of 15
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

-- Define a predicate to check if a number is within the range [20, 205]
def in_range (n : ℕ) : Prop :=
  20 ≤ n ∧ n ≤ 205

-- The main theorem statement
theorem number_of_multiples_of_15_in_range : 
  finset.card (finset.filter (λ n, is_multiple_of_15 n ∧ in_range n) (finset.range 206)) = 12 :=
by
  sorry

end number_of_multiples_of_15_in_range_l460_460256


namespace function_form_l460_460175

noncomputable def f : ℕ → ℕ := sorry

theorem function_form (c d a : ℕ) (h1 : c > 1) (h2 : a - c > 1)
  (hf : ∀ n : ℕ, f n + f (n + 1) = f (n + 2) + f (n + 3) - 168) :
  (∀ n : ℕ, f (2 * n) = c + n * d) ∧ (∀ n : ℕ, f (2 * n + 1) = (168 - d) * n + a - c) :=
sorry

end function_form_l460_460175


namespace circle_area_l460_460105

theorem circle_area (r : ℝ) (h : (2 * r)^2 = 8 * (2 * real.pi * r)) : real.pi * r^2 = 16 * real.pi^3 := by
  sorry

end circle_area_l460_460105


namespace sin_ratio_l460_460284

variable (a b c A B C : ℝ)

-- Triangle sides opposite to angles A, B, C respectively
axiom opposite_sides : a = side_opposite A
axiom opposite_sides_2 : b = side_opposite B
axiom opposite_sides_3 : c = side_opposite C

-- Given condition
axiom given_condition : c * (a * cos B - b * cos A) = b^2

-- The theorem we need to prove
theorem sin_ratio : given_condition → (sin A / sin B = sqrt 2) := sorry

end sin_ratio_l460_460284


namespace significant_difference_in_hygiene_risk_indicator_estimation_l460_460473

-- Define the data given in the table
def hygiene_habits : Type := 
{
  not_good_enough_case  : ℕ
  good_enough_case      : ℕ
  not_good_enough_control: ℕ
  good_enough_control   : ℕ
}

def survey_data : hygiene_habits :=
{
  not_good_enough_case  := 40,
  good_enough_case      := 60,
  not_good_enough_control:= 10,
  good_enough_control   := 90,
}

-- Define values required for Chi-Square calculation
def n : ℕ := 200
def a : ℕ := survey_data.not_good_enough_case
def b : ℕ := survey_data.good_enough_case
def c : ℕ := survey_data.not_good_enough_control
def d : ℕ := survey_data.good_enough_control 

-- Calculate and define Chi-Square statistic
def K_squared : ℝ := 
  (n * (a * d - b * c)^2 : ℝ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Critical value at 99% confidence
def K_critical : ℝ := 6.635

-- Define probabilities for risk level calculation
def P_A_B : ℝ := 2/5
def P_A_not_B : ℝ := 1/10
def P_not_A_B : ℝ := 3/5
def P_not_A_not_B : ℝ := 9/10

-- Define R from given probabilities
noncomputable def R : ℝ := 
  (P_A_B / P_not_A_B) * (P_not_A_not_B / P_A_not_B)

-- Lean theorem statement
theorem significant_difference_in_hygiene :
  K_squared > K_critical :=
sorry

theorem risk_indicator_estimation :
  R = 6 :=
sorry

end significant_difference_in_hygiene_risk_indicator_estimation_l460_460473


namespace cube_root_of_product_is_integer_l460_460365

theorem cube_root_of_product_is_integer :
  (∛(2^9 * 5^3 * 7^3) = 280) :=
sorry

end cube_root_of_product_is_integer_l460_460365


namespace geometric_sequence_third_term_approx_l460_460027

noncomputable def geometric_sequence_third_term : ℝ :=
  let a := 1000 in
  let sixth_term := 125 in
  let r := (1 / 8) ^ (1 / 5 : ℝ) in
  a * r ^ 2

theorem geometric_sequence_third_term_approx : abs (geometric_sequence_third_term - 301) < 1 :=
  by 
  -- We leave the proof as sorry since it is beyond the scope of this task
  sorry

end geometric_sequence_third_term_approx_l460_460027


namespace equivalent_angle_l460_460019

theorem equivalent_angle (theta : ℤ) (k : ℤ) : 
  (∃ k : ℤ, (-525 + k * 360 = 195)) :=
by
  sorry

end equivalent_angle_l460_460019


namespace train_crossing_time_l460_460054

-- Define our conditions and question statement in Lean
theorem train_crossing_time 
  (length : ℝ := 100) -- the length of each train
  (v_fast : ℝ := 60.00000000000001) -- speed of the faster train
  (v_slow : ℝ := v_fast / 2) -- speed of the slower train
  : ((2 * length) / (v_fast + v_slow) ≈ 2.2222222222222223) := 
by
  -- proof steps would go here, we're leaving it as sorry to denote the theorem to be proven
  sorry

end train_crossing_time_l460_460054


namespace cost_to_consume_desired_calories_l460_460754

-- conditions
def calories_per_chip : ℕ := 10
def chips_per_bag : ℕ := 24
def cost_per_bag : ℕ := 2
def desired_calories : ℕ := 480

-- proof statement
theorem cost_to_consume_desired_calories :
  let total_calories_per_bag := chips_per_bag * calories_per_chip in
  let bags_needed := desired_calories / total_calories_per_bag in
  let total_cost := bags_needed * cost_per_bag in
  total_cost = 4 :=
by
  sorry

end cost_to_consume_desired_calories_l460_460754


namespace polygon_intersections_l460_460000

theorem polygon_intersections :
  ∀ (P₄ P₆ P₈ : Type) [regular_polygon P₄ 4] [regular_polygon P₆ 6] [regular_polygon P₈ 8]
  (shared_vertex : ∀ (P Q : Type) [regular_polygon P n] [regular_polygon Q m], 
    at_most_one_shared_vertex P Q)
  (no_common_intersections : ∀ (P Q R : Type) [regular_polygon P n] [regular_polygon Q m] [regular_polygon R k], 
    no_three_sides_intersect P Q R),
  total_intersections P₄ P₆ P₈ shared_vertex no_common_intersections = 164 := 
sorry

end polygon_intersections_l460_460000


namespace workman_problem_l460_460115

theorem workman_problem
    (total_work : ℝ)
    (B_rate : ℝ)
    (A_rate : ℝ)
    (days_together : ℝ)
    (W : total_work = 8 * (A_rate + B_rate))
    (A_2B : A_rate = 2 * B_rate) :
    total_work = 24 * B_rate :=
by
  sorry

end workman_problem_l460_460115


namespace positive_integer_solutions_l460_460773

theorem positive_integer_solutions (x y z t : ℕ) (h : x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0) (h_eq : x + y + z + t = 10) : ∃ n : ℕ, n = 84 :=
by
  let x' = x - 1
  let y' = y - 1
  let z' = z - 1
  let t' = t - 1
  have h' : x' + y' + z' + t' = 6 := sorry
  -- Use stars and bars method to show the number of solutions is 84
  have h_sol : ∃ n : ℕ, n = 84 := sorry
  exact h_sol

end positive_integer_solutions_l460_460773


namespace mowers_kvass_l460_460743

theorem mowers_kvass (x : ℕ) (k : ℕ) (h₁ : k = 6) (h₂ : 8 * k = 48) : x = 16 :=
by
  -- Conditions
  have h₃ : 8 * k = 48,
    from by rw [h₁]; norm_num,
  -- Question
  have h₄ : x * 3 = 48,
    from sorry, -- The exact proof steps are not included as per instruction
  -- Conclusion
  have h₅ : x = 48 / 3,
    from Nat.div_eq_of_eq_mul_right (Nat.succ_pos 2) h₄,
  rw Nat.div_eq_of_eq_mul_right (Nat.succ_pos 2) h₄,
  norm_num at h₅,
  exact h₅

end mowers_kvass_l460_460743


namespace tan_F_value_l460_460301

theorem tan_F_value (D E F : ℝ) (h1 : Real.cot D * Real.cot F = 1 / 3)
  (h2 : Real.cot E * Real.cot F = 1 / 8) (h3 : D + E + F = 180) : 
  Real.tan F = 12 + Real.sqrt 136 := 
by
  sorry

end tan_F_value_l460_460301


namespace range_of_a_l460_460605

noncomputable def has_solution_in_interval (a : ℝ) : Prop := 
  ∃ x ∈ Icc (-1 : ℝ) 1, a * x ^ 2 + a * x - 2 = 0

noncomputable def no_real_x (a : ℝ) : Prop := 
  ¬ ∃ x : ℝ, x ^ 2 + 2 * a * x + 2 * a ≤ 0

theorem range_of_a (a : ℝ) : 
  (¬(has_solution_in_interval a ∨ no_real_x a) ↔ (-8 < a ∧ a < 0) ∨ (0 < a ∧ a < 1)) :=
  sorry

end range_of_a_l460_460605


namespace cube_root_of_product_is_integer_l460_460366

theorem cube_root_of_product_is_integer :
  (∛(2^9 * 5^3 * 7^3) = 280) :=
sorry

end cube_root_of_product_is_integer_l460_460366


namespace isaac_sleep_time_l460_460898

theorem isaac_sleep_time :
  ∀ (wakes_up : ℕ) (sleeps_hours : ℕ), 
  wakes_up = 7 ∧ sleeps_hours = 8 → 
  ∃ (goes_to_sleep : ℕ), goes_to_sleep = 23 :=
by
  intros wakes_up sleeps_hours h
  cases h with hwakeup hsleep
  use 23
  sorry

end isaac_sleep_time_l460_460898


namespace find_y_l460_460189

def v := λ (y : ℝ), ![2, y]
def w := ![5, -1]
def proj_w_v (y : ℝ) := (inner (v y) w / inner w w) • w

theorem find_y (y : ℝ) (h : proj_w_v y = ![3, -0.6]) : y = -5.6 :=
by
  have h1 : inner (v y) w = 10 - y := by sorry
  have h2 : inner w w = 26 := by sorry
  have proj_formula : proj_w_v y = (10 - y) / 26 • w := by sorry
  rw [proj_formula] at h
  have eq1 : (10 - y) / 26 * 5 = 3 := by sorry
  have eq2 : y = -5.6 := by sorry
  exact eq2

end find_y_l460_460189


namespace vertical_asymptote_l460_460918

theorem vertical_asymptote (k : ℝ) :
  (∀ x : ℝ, f x = (x^2 + 2 * x + k) / (x^2 - 3 * x + 2)) ∧ 
  (∃ x, (x^2 - 3 * x + 2) = 0) →
  (k = -3 ∨ k = -8) :=
sorry

end vertical_asymptote_l460_460918


namespace number_of_possible_measures_l460_460394

theorem number_of_possible_measures (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
sorry

end number_of_possible_measures_l460_460394


namespace students_at_start_of_year_l460_460871

variable (S : ℕ)

def initial_students := S
def students_left := 6
def students_new := 42
def end_year_students := 47

theorem students_at_start_of_year :
  initial_students + (students_new - students_left) = end_year_students → initial_students = 11 :=
by
  sorry

end students_at_start_of_year_l460_460871


namespace root_of_quadratic_eq_when_C_is_3_l460_460913

-- Define the quadratic equation and the roots we are trying to prove
def quadratic_eq (C : ℝ) (x : ℝ) := 3 * x^2 - 6 * x + C = 0

-- Set the constant C to 3
def C : ℝ := 3

-- State the theorem that proves the root of the equation when C=3 is x=1
theorem root_of_quadratic_eq_when_C_is_3 : quadratic_eq C 1 :=
by
  -- Skip the detailed proof
  sorry

end root_of_quadratic_eq_when_C_is_3_l460_460913


namespace five_equal_size_right_triangles_l460_460121

-- Definitions to represent the problem
structure Point2D :=
(x : ℝ)
(y : ℝ)

structure Triangle :=
(A : Point2D)
(B : Point2D)
(C : Point2D)

structure Pentagon :=
(A : Point2D)
(B : Point2D)
(C : Point2D)
(D : Point2D)
(E : Point2D)
(O : Point2D)

-- Check if all triangles are right triangles.
def isRightTriangle (T : Triangle) : Prop :=
  ∃ (A B C : ℝ), T.A = ⟨A, 0⟩ ∧ T.B = ⟨0, B⟩ ∧ T.C = ⟨0, 0⟩ ∧ 
  (A ≠ 0 ∧ B ≠ 0)

def pentagonWithInnerTrianglesRight (P : Pentagon) : Prop :=
  ∀ (i j : ℕ) (hi : (0 ≤ i ∧ i < j) ∧ (j < 5)),
  let vertex := [P.A, P.B, P.C, P.D, P.E] in
  let triangles := [
    Triangle.mk P.A P.B P.O,
    Triangle.mk P.B P.C P.O,
    Triangle.mk P.C P.D P.O,
    Triangle.mk P.D P.E P.O,
    Triangle.mk P.E P.A P.O
  ] in
  isRightTriangle (triangles !! i) ∧ isRightTriangle (triangles !! j)

theorem five_equal_size_right_triangles (P : Pentagon) :
  pentagonWithInnerTrianglesRight P :=
by
  sorry

end five_equal_size_right_triangles_l460_460121


namespace maximal_chord_implication_l460_460728

theorem maximal_chord_implication (ABC : Type)
  [triangle ABC]
  (A B C X X' F G D E H : ABC)
  (circumcircle : Circle ABC)
  (perp_bisector_BC : Line ABC)
  (k : Circle ABC)
  (cond_X_on_BC : X ∈ LineSegment B C)
  (cond_X'_inter : second_intersection (line_through A X) circumcircle = X')
  (cond_maximal_length : ∀ Y, length (line_segment_from Y (second_intersection (line_through A Y) circumcircle)) ≤ length (line_segment_from X X'))
  (median_A : Line A F)
  (angle_bisector_A : Line A G) :
  lies_between (line_through A X) median_A angle_bisector_A :=
sorry

end maximal_chord_implication_l460_460728


namespace trig_expression_identity_l460_460157

theorem trig_expression_identity (θ : ℝ) :
  (tan θ)^2 + (sin θ)^2 / ((tan θ)^2 * (sin θ)^2) = (cot θ)^2 * (1 + (cos θ)^2) :=
by
  sorry

end trig_expression_identity_l460_460157


namespace propositions_correct_l460_460597

theorem propositions_correct :
  (∀ (m n : Line) (α β : Plane), m ⊥ α ∧ n ⊆ β → (α ⊥ β → m ∥ n) ∨ (α ⊥ β → ∃ (p : Plane), m ⊥ p ∧ p ⊥ n))
  ∧ (∀ (x : ℝ), x ∈ Ioi 0 → ¬ (log 2 x < log 3 x))
  ∧ (∀ (a b m : ℝ), (a < b → am^2 < bm^2) = false)
  ∧ (∀ (x : ℝ), 3 * sin (2 * x + π/3 - π/12) = 3 * sin (2 * x)) := sorry

end propositions_correct_l460_460597


namespace sum_of_solutions_l460_460426

theorem sum_of_solutions (a b c : ℝ) (h_eq : ∀ x : ℝ, x^2 - (8 * x) + 15 = a * x^2 + b * x + c) (ha : a = 1) (hb : b = -8) (hc : c = 15) :
  ∑ x in {x : ℝ | (a * x^2 + b * x + c) = 0}, x = 8 :=
by
  -- It is a proof placeholder
  sorry

end sum_of_solutions_l460_460426


namespace eat_5_pounds_together_time_l460_460348

-- Eating rates based on the conditions
def rate_fat : ℝ := 1 / 20
def rate_thin : ℝ := 1 / 30
def rate_average : ℝ := 1 / 24
def combined_rate : ℝ := rate_fat + rate_thin + rate_average

-- Time to eat 5 pounds with the combined rate
theorem eat_5_pounds_together_time : combined_rate * 40 = 5 := by
  sorry

end eat_5_pounds_together_time_l460_460348


namespace multiples_of_3_or_5_but_not_6_l460_460626

theorem multiples_of_3_or_5_but_not_6 (n : ℕ) (h1 : n ≤ 200) :
  (multiples3 : ℕ) (multiples5 : ℕ) (multiples15 : ℕ) (multiples6 : ℕ)
    (h1 : multiples3 = (200 - 3) / 3 + 1)
    (h2 : multiples5 = (200 - 5) / 5 + 1)
    (h3 : multiples15 = (200 - 15) / 15 + 1)
    (h4 : multiples6 = (200 - 6) / 6 + 1) : 
    (multiples3 + multiples5 - multiples15 - multiples6) = 60 :=
begin
  sorry,
end

end multiples_of_3_or_5_but_not_6_l460_460626


namespace math_problem_l460_460068

theorem math_problem (a : ℝ) (h : a = 1/3) : (3 * a⁻¹ + 2 / 3 * a⁻¹) / a = 33 := by
  sorry

end math_problem_l460_460068


namespace cube_difference_div_l460_460067

theorem cube_difference_div (a b : ℕ) (h_a : a = 64) (h_b : b = 27) : 
  (a^3 - b^3) / (a - b) = 6553 := by
  sorry

end cube_difference_div_l460_460067


namespace common_sum_of_magic_square_is_zero_l460_460518

theorem common_sum_of_magic_square_is_zero :
  ∃ (matrix : ℕ → ℕ → ℤ), 
  (∀ i, ∃ j, matrix i j ∈ (finset.range 9).map (λ n, -4 + n)) ∧
  (∀ i j, 0 ≤ matrix i j ∧ matrix i j ≤ 4) ∧
  (∀ i, (matrix i 0 + matrix i 1 + matrix i 2) = (matrix 0 0 + matrix 0 1 + matrix 0 2)) ∧
  (∀ j, (matrix 0 j + matrix 1 j + matrix 2 j) = (matrix 0 0 + matrix 0 1 + matrix 0 2)) ∧
  ((matrix 0 0 + matrix 1 1 + matrix 2 2) = (matrix 0 0 + matrix 0 1 + matrix 0 2)) ∧
  ((matrix 0 2 + matrix 1 1 + matrix 2 0) = (matrix 0 0 + matrix 0 1 + matrix 0 2)) ∧ 
  (matrix 0 0 + matrix 0 1 + matrix 0 2 = 0) :=
sorry

end common_sum_of_magic_square_is_zero_l460_460518


namespace angle_NCB_l460_460302

theorem angle_NCB (A B C N : Type) 
  [Triangle ABC : A ∠ B ∠ C ∧ ∠ABC = 40 ∧ ∠ACB = 20 ∧ N ∈ 𝜄ABC]
  (h1 : A B C N ∠NBC = 30)
  (h2 : A B N ∠NAB = 20) : 
  ∠NCB = 10 :=
sorry

end angle_NCB_l460_460302


namespace P_transformation_final_coords_l460_460403

theorem P_transformation_final_coords (a b : ℝ) :
  let Q := ((2 + - (b - 6)), (6 - (a - 2))) in
  let R := (-Q.2, -Q.1) in
  R = (-5, 2) → b - a = 15 :=
by
  intros Q R h
  sorry

end P_transformation_final_coords_l460_460403


namespace total_preparation_time_l460_460991

theorem total_preparation_time
    (minutes_per_game : ℕ)
    (number_of_games : ℕ)
    (h1 : minutes_per_game = 10)
    (h2 : number_of_games = 15) :
    minutes_per_game * number_of_games = 150 :=
by
  -- Lean 4 proof goes here
  sorry

end total_preparation_time_l460_460991


namespace all_positive_l460_460863

theorem all_positive (a1 a2 a3 a4 a5 a6 a7 : ℝ)
  (h1 : a1 + a2 + a3 + a4 > a5 + a6 + a7)
  (h2 : a1 + a2 + a3 + a5 > a4 + a6 + a7)
  (h3 : a1 + a2 + a3 + a6 > a4 + a5 + a7)
  (h4 : a1 + a2 + a3 + a7 > a4 + a5 + a6)
  (h5 : a1 + a2 + a4 + a5 > a3 + a6 + a7)
  (h6 : a1 + a2 + a4 + a6 > a3 + a5 + a7)
  (h7 : a1 + a2 + a4 + a7 > a3 + a5 + a6)
  (h8 : a1 + a2 + a5 + a6 > a3 + a4 + a7)
  (h9 : a1 + a2 + a5 + a7 > a3 + a4 + a6)
  (h10 : a1 + a2 + a6 + a7 > a3 + a4 + a5)
  (h11 : a1 + a3 + a4 + a5 > a2 + a6 + a7)
  (h12 : a1 + a3 + a4 + a6 > a2 + a5 + a7)
  (h13 : a1 + a3 + a4 + a7 > a2 + a5 + a6)
  (h14 : a1 + a3 + a5 + a6 > a2 + a4 + a7)
  (h15 : a1 + a3 + a5 + a7 > a2 + a4 + a6)
  (h16 : a1 + a3 + a6 + a7 > a2 + a4 + a5)
  (h17 : a1 + a4 + a5 + a6 > a2 + a3 + a7)
  (h18 : a1 + a4 + a5 + a7 > a2 + a3 + a6)
  (h19 : a1 + a4 + a6 + a7 > a2 + a3 + a5)
  (h20 : a1 + a5 + a6 + a7 > a2 + a3 + a4)
  (h21 : a2 + a3 + a4 + a5 > a1 + a6 + a7)
  (h22 : a2 + a3 + a4 + a6 > a1 + a5 + a7)
  (h23 : a2 + a3 + a4 + a7 > a1 + a5 + a6)
  (h24 : a2 + a3 + a5 + a6 > a1 + a4 + a7)
  (h25 : a2 + a3 + a5 + a7 > a1 + a4 + a6)
  (h26 : a2 + a3 + a6 + a7 > a1 + a4 + a5)
  (h27 : a2 + a4 + a5 + a6 > a1 + a3 + a7)
  (h28 : a2 + a4 + a5 + a7 > a1 + a3 + a6)
  (h29 : a2 + a4 + a6 + a7 > a1 + a3 + a5)
  (h30 : a2 + a5 + a6 + a7 > a1 + a3 + a4)
  (h31 : a3 + a4 + a5 + a6 > a1 + a2 + a7)
  (h32 : a3 + a4 + a5 + a7 > a1 + a2 + a6)
  (h33 : a3 + a4 + a6 + a7 > a1 + a2 + a5)
  (h34 : a3 + a5 + a6 + a7 > a1 + a2 + a4)
  (h35 : a4 + a5 + a6 + a7 > a1 + a2 + a3)
: a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0 ∧ a5 > 0 ∧ a6 > 0 ∧ a7 > 0 := 
sorry

end all_positive_l460_460863


namespace largest_number_of_positive_consecutive_integers_l460_460058

theorem largest_number_of_positive_consecutive_integers (n a : ℕ) (h1 : a > 0) (h2 : n > 0) (h3 : (n * (2 * a + n - 1)) / 2 = 45) : 
  n = 9 := 
sorry

end largest_number_of_positive_consecutive_integers_l460_460058


namespace ellipse_foci_coordinates_l460_460593

theorem ellipse_foci_coordinates :
  ∃ x y : Real, (3 * x^2 + 4 * y^2 = 12) ∧ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0)) :=
by
  sorry

end ellipse_foci_coordinates_l460_460593


namespace ellipse_foci_coordinates_l460_460382

theorem ellipse_foci_coordinates :
  ∀ x y : ℝ,
  25 * x^2 + 16 * y^2 = 1 →
  (x, y) = (0, 3/20) ∨ (x, y) = (0, -3/20) :=
by
  intro x y h
  sorry

end ellipse_foci_coordinates_l460_460382


namespace volume_of_right_square_prism_l460_460124

theorem volume_of_right_square_prism (length width : ℕ) (H1 : length = 12) (H2 : width = 8) :
    ∃ V, (V = 72 ∨ V = 48) :=
by
  sorry

end volume_of_right_square_prism_l460_460124


namespace polygon_even_perimeter_l460_460485

theorem polygon_even_perimeter (P : ℕ) (edges : List (ℕ × ℕ)) :
  (∀ (edge ∈ edges), edge.fst = 1 ∨ edge.snd = 1) →
  (∃ (n : ℕ), n > 1 ∧ P = 2 * n) :=
by
  intros h
  sorry

end polygon_even_perimeter_l460_460485


namespace polar_to_cartesian_conversion_l460_460881

noncomputable def polarToCartesian (ρ θ : ℝ) : ℝ × ℝ :=
  let x := ρ * Real.cos θ
  let y := ρ * Real.sin θ
  (x, y)

theorem polar_to_cartesian_conversion :
  polarToCartesian 4 (Real.pi / 3) = (2, 2 * Real.sqrt 3) :=
by
  sorry

end polar_to_cartesian_conversion_l460_460881


namespace geometric_sequence_value_l460_460635

variable {a : ℕ → ℝ}

-- Conditions
axiom geo_seq (n : ℕ) : a (n + 1) = a n * r for some r > 0
axiom positive_seq (n : ℕ) : a n > 0
axiom condition1 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25

-- Theorem statement
theorem geometric_sequence_value : a 3 + a 5 = 5 :=
by
  sorry

end geometric_sequence_value_l460_460635


namespace grid_bottom_right_not_divisible_l460_460351
open Nat

theorem grid_bottom_right_not_divisible (n : ℕ) (h : n = 2012) :
  (forall (i j : ℕ), (i = 0 ∨ j = 0) → (grid i j = 1)) ∧
  (forall (i j : ℕ), (i ≠ 0 ∧ j ≠ 0 ∧ (i + j = 4012)) → (grid i j = 0)) ∧
  (forall (i j : ℕ), (i ≠ 0 ∧ j ≠ 0 ∧ (i + j ≠ 4012)) → (grid i j = grid (i - 1) j + grid i (j - 1))) →
  ¬ (grid 2011 2011) % 2011 = 0 := 
sorry

end grid_bottom_right_not_divisible_l460_460351


namespace Kira_morning_songs_l460_460685

variable (x : ℕ)

theorem Kira_morning_songs (h1 : 5 * (x + 15 + 3) = 140) : x = 10 := by
  have : 5 * (x + 18) = 140 := h1
  have : 5 * x + 90 = 140 := by
    rw [mul_add, mul_one] at this
    exact this
  have : 5 * x = 50 := by
    linarith
  have : x = 10 := by
    apply eq_of_mul_eq_mul_left
    norm_num
    exact this
  exact this

end Kira_morning_songs_l460_460685


namespace num_possible_measures_of_A_l460_460397

-- Given conditions
variables (A B : ℕ)
variables (k : ℕ) (hk : k ≥ 1)
variables (hab : A + B = 180)
variables (ha : A = k * B)

-- The proof statement
theorem num_possible_measures_of_A : 
  ∃ (n : ℕ), n = 17 ∧ ∀ k, (k + 1) ∣ 180 ∧ k ≥ 1 → n = 17 := 
begin
  sorry
end

end num_possible_measures_of_A_l460_460397


namespace compute_four_at_seven_l460_460883

def operation (a b : ℤ) : ℤ :=
  5 * a - 2 * b

theorem compute_four_at_seven : operation 4 7 = 6 :=
by
  sorry

end compute_four_at_seven_l460_460883


namespace person_a_time_walked_l460_460750

variables (v_A v_B : real) (x : real)
hypothesis h1 : (v_A / v_B = 3 / 2)
hypothesis h2 : (v_A * (x - 5) = v_B * x)

theorem person_a_time_walked : x = 10 :=
by sorry

end person_a_time_walked_l460_460750


namespace inlet_pipe_rate_l460_460139

theorem inlet_pipe_rate (capacity : ℕ) (t_empty : ℕ) (t_with_inlet : ℕ) (R_out : ℕ) :
  capacity = 6400 →
  t_empty = 10 →
  t_with_inlet = 16 →
  R_out = capacity / t_empty →
  (R_out - (capacity / t_with_inlet)) / 60 = 4 :=
by
  intros h1 h2 h3 h4 
  sorry

end inlet_pipe_rate_l460_460139


namespace sum_of_factors_eq_18_l460_460409

theorem sum_of_factors_eq_18 (x : ℤ) (h1 : ∑ d in (finset.filter (λ d, x % d = 0) (finset.range (x + 1))), d = 18) (h2 : 2 ∣ x) : x = 10 :=
sorry

end sum_of_factors_eq_18_l460_460409


namespace remainder_17_pow_63_div_7_l460_460063

theorem remainder_17_pow_63_div_7 :
  (17^63) % 7 = 6 :=
by
  have h: 17 % 7 = 3 := rfl
  sorry

end remainder_17_pow_63_div_7_l460_460063


namespace stickers_given_l460_460074

def total_stickers : ℕ := 100
def andrew_ratio : ℚ := 1 / 5
def bill_ratio : ℚ := 3 / 10

theorem stickers_given (zander_collection : ℕ)
                       (andrew_received : ℚ)
                       (bill_received : ℚ)
                       (total_given : ℚ):
  zander_collection = total_stickers →
  andrew_received = andrew_ratio →
  bill_received = bill_ratio →
  total_given = (andrew_received * zander_collection) + (bill_received * (zander_collection - (andrew_received * zander_collection))) →
  total_given = 44 :=
by
  intros hz har hbr htg
  sorry

end stickers_given_l460_460074


namespace bookstore_loss_l460_460454

def book_prices_purchase := [60, 75, 90]
def book_prices_selling := [63, 80, 100]
def sales_tax_rate (price: ℤ) : ℚ :=
  if price ≤ 50 then 0.06
  else if price ≤ 75 then 0.07
  else 0.08

def total_purchase_price := (60 + 75 + 90 : ℤ)
def total_selling_price := (63 + 80 + 100 : ℤ)

def total_profit := total_selling_price - total_purchase_price

def total_sales_tax := 
  (63 * 0.07) + (80 * 0.08) + (100 * 0.08)

def net_profit := total_profit - total_sales_tax

def loss := total_sales_tax - total_profit

def loss_percentage := (loss / total_purchase_price : ℚ) * 100

theorem bookstore_loss : loss_percentage = 0.36  :=
by
  sorry

end bookstore_loss_l460_460454


namespace measure_of_angle_PMN_l460_460297

theorem measure_of_angle_PMN
  (P Q R M N : Type)
  [geometry.linear_space P Q R] 
  (PR_eq_RQ : PR = RQ)
  (PM_eq_PN : PM = PN)
  (triangle_PMN_is_isosceles : is_isosceles_triangle PM N)
  (angle_PQR_eq_60 : ∠PQR = 60) :
  ∠PMN = 60 :=
begin
  sorry
end

end measure_of_angle_PMN_l460_460297


namespace tim_used_to_run_days_l460_460051

def hours_per_day := 2
def total_hours_per_week := 10
def added_days := 2

theorem tim_used_to_run_days (runs_per_day : ℕ) (total_weekly_runs : ℕ) (additional_runs : ℕ) : 
  runs_per_day = hours_per_day →
  total_weekly_runs = total_hours_per_week →
  additional_runs = added_days →
  (total_weekly_runs / runs_per_day) - additional_runs = 3 :=
by
  intros h1 h2 h3
  sorry

end tim_used_to_run_days_l460_460051


namespace problem_condition_implies_statement_l460_460333

variable {a b c : ℝ}

theorem problem_condition_implies_statement :
  a^3 + a * b + a * c < 0 → b^5 - 4 * a * c > 0 :=
by
  intros h
  sorry

end problem_condition_implies_statement_l460_460333


namespace problem_1_problem_2_problem_3_problem_4_l460_460132

-- Define the function and conditions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

def symm_point (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop := ∀ x, f (2 * p.1 - x) = 2 * p.2 - f (x)

def axis_symmetry (f : ℝ → ℝ) (axis : ℝ) : Prop := ∀ x, f (2 * axis - x) = f (x)

-- Statements
theorem problem_1 (k : ℤ) : is_odd_function (λ x, sin (k * π - x)) :=
sorry

theorem problem_2 : ¬ symm_point (λ x, tan (2 * x + π / 6)) (π / 12, 0) :=
sorry

theorem problem_3 : axis_symmetry (λ x, cos (2 * x + π / 3)) (-2 * π / 3) :=
sorry

theorem problem_4 (x : ℝ) (h : tan (π - x) = 2) : cos (x) ^ 2 = 1 / 5 :=
sorry

end problem_1_problem_2_problem_3_problem_4_l460_460132


namespace remainder_of_N_div_1000_l460_460701

-- Defining the set A
def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}.to_finset

-- Defining the condition for N
def N : ℕ :=
  let k_choices := λ k, (7.choose k) * k^(7 - k)
  8 * Finset.sum (Finset.range 7) (λ k, k_choices (k + 1))

-- Theorem that must be proven
theorem remainder_of_N_div_1000 : N % 1000 = 992 := by
  sorry

end remainder_of_N_div_1000_l460_460701


namespace determine_values_and_triangle_l460_460924

theorem determine_values_and_triangle
  (a b c : ℝ)
  (h : |a - real.sqrt 7| + real.sqrt (b - 5) + (c - 4 * real.sqrt 2) ^ 2 = 0) :
  a = real.sqrt 7 ∧ b = 5 ∧ c = 4 * real.sqrt 2 ∧ 
  (a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2) :=
by {
  sorry
}

end determine_values_and_triangle_l460_460924


namespace output_is_25_l460_460148

-- Define the algorithm step
def step (i S : ℕ) : ℕ × ℕ :=
  if i < 10 then
    let i' := i + 2
    let S' := 2 * i' + 3
    (i', S')
  else
    (i, S)

-- Recursive function to iterate the algorithm until termination
def iterate (i S : ℕ) : ℕ :=
  if i < 10 then
    let (i', S') := step i S
    iterate i' S'
  else
    S

-- Initial conditions
def initial_i : ℕ := 1
def initial_S : ℕ := 0

-- The final output S
def final_S : ℕ := iterate initial_i initial_S

-- The statement to prove
theorem output_is_25 : final_S = 25 :=
  by
    sorry

end output_is_25_l460_460148


namespace multiples_3_or_5_not_6_l460_460630

theorem multiples_3_or_5_not_6 (n : ℕ) (hn : n ≤ 200) :
  card ({m | m ∣ n ∧ m ≤ 200 ∧ ((m % 3 = 0 ∨ m % 5 = 0) ∧ ¬ (m % 6 = 0))}) = 73 := sorry

end multiples_3_or_5_not_6_l460_460630


namespace sum_of_valid_n_l460_460066

theorem sum_of_valid_n :
  (∀ n : ℕ, (nat.choose 30 15 + nat.choose 30 n = nat.choose 31 16) → (n = 14 ∨ n = 16)) →
  14 + 16 = 30 :=
by
  intros h
  sorry

end sum_of_valid_n_l460_460066


namespace meaningful_expression_range_l460_460996

theorem meaningful_expression_range (a : ℝ) :
  (∃ (x : ℝ), x = (sqrt (a + 1)) / (a - 2)) ↔ a ≥ -1 ∧ a ≠ 2 := 
begin
  sorry
end

end meaningful_expression_range_l460_460996


namespace find_number_l460_460812

theorem find_number (N : ℝ) (h : (5/4 : ℝ) * N = (4/5 : ℝ) * N + 27) : N = 60 :=
by
  sorry

end find_number_l460_460812


namespace find_radius_of_C4_l460_460502

open Real

theorem find_radius_of_C4 (R1 R2 R3 R4 : ℝ) : 
  R1 = 360 → 
  R2 = 360 → 
  R3 = 90 → 
  ∃ R4, R4 = 40 :=
by intros h1 h2 h3
   use 40
   sorry

end find_radius_of_C4_l460_460502


namespace construction_output_daily_team_B_max_cost_l460_460052

-- Definitions for the problem conditions
variables (x y a : ℝ)

-- Daily output equations
def daily_output_condition_1 : Prop := 10 * x + 15 * y = 600
def daily_output_condition_2 : Prop := x + y = 50

-- Cost constraints
def team_A_daily_cost : ℝ := 0.6
def total_project_cost : ℝ := 12
def project_length : ℝ := 600
def work_together_days : ℕ := 10
def remaining_work := project_length - work_together_days * (x + y)
def team_B_work_time := remaining_work / y

-- Prove desired outputs
theorem construction_output_daily (h1 : daily_output_condition_1)
                                  (h2 : daily_output_condition_2) :
                                  x = 30 ∧ y = 20 :=
begin
  sorry
end

theorem team_B_max_cost (h1 : daily_output_condition_1)
                        (h2 : daily_output_condition_2)
                        (h3 : work_together_days * (team_A_daily_cost + a * y) + team_B_work_time * a ≤ total_project_cost) :
                        a ≤ 0.4 :=
begin
  sorry
end

end construction_output_daily_team_B_max_cost_l460_460052


namespace intersection_ab_correct_l460_460195

noncomputable def set_A : Set ℝ := { x : ℝ | x > 1/3 }
def set_B : Set ℝ := { x : ℝ | ∃ y : ℝ, x^2 + y^2 = 4 ∧ y ≥ -2 ∧ y ≤ 2 }
def intersection_AB : Set ℝ := { x : ℝ | 1/3 < x ∧ x ≤ 2 }

theorem intersection_ab_correct : set_A ∩ set_B = intersection_AB := 
by 
  -- proof omitted
  sorry

end intersection_ab_correct_l460_460195


namespace value_of_polynomial_root_l460_460265

theorem value_of_polynomial_root (m : ℝ) (h : 2 * m^2 - 3 * m - 1 = 0) : 4 * m^2 - 6 * m + 2021 = 2023 :=
by 
  have h₁ : 2 * m^2 - 3 * m = 1 := 
    calc
      2 * m^2 - 3 * m = 2 * m^2 - 3 * m - (-1) : by ring
      ... = 1 : by rw h
  calc
    4 * m^2 - 6 * m + 2021 = 2 * (2 * m^2 - 3 * m) + 2021 : by ring
    ... = 2 * 1 + 2021 : by rw [h₁]
    ... = 2023 : by norm_num

end value_of_polynomial_root_l460_460265


namespace line_through_point_area_T_l460_460471

variable (a T : ℝ)

def equation_of_line (x y : ℝ) : Prop := 2 * T * x - a^2 * y + 2 * a * T = 0

theorem line_through_point_area_T :
  ∃ (x y : ℝ), equation_of_line a T x y ∧ x = -a ∧ y = (2 * T) / a :=
by
  sorry

end line_through_point_area_T_l460_460471


namespace remainder_is_576_l460_460705

-- Definitions based on the conditions
def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def num_functions (f : ℕ → ℕ) (A : Set ℕ) : ℕ :=
  if ∃ c ∈ A, ∀ x ∈ A, f (f x) = c then 1 else 0

-- Define the total number of such functions
def total_functions (A : Set ℕ) : ℕ :=
  8 * (∑ k in ({1, 2, 3, 4, 5, 6, 7} : Set ℕ), Nat.choose 7 k * (k ^ (7 - k)))

-- The main theorem to prove
theorem remainder_is_576 : (total_functions A % 1000) = 576 :=
  by
    sorry

end remainder_is_576_l460_460705


namespace plane_intersects_unit_cubes_l460_460836

def unitCubeCount (side_length : ℕ) : ℕ :=
  side_length ^ 3

def intersectionCount (num_unitCubes : ℕ) (side_length : ℕ) : ℕ :=
  if side_length = 4 then 32 else 0 -- intersection count only applies for side_length = 4

theorem plane_intersects_unit_cubes
  (side_length : ℕ)
  (num_unitCubes : ℕ)
  (cubeArrangement : num_unitCubes = unitCubeCount side_length)
  (planeCondition : True) -- the plane is perpendicular to the diagonal and bisects it
  : intersectionCount num_unitCubes side_length = 32 := by
  sorry

end plane_intersects_unit_cubes_l460_460836


namespace find_base_r_l460_460746

noncomputable def x: ℕ := 9999

theorem find_base_r (r: ℕ) (hr_even: Even r) (hr_gt_9: r > 9) 
    (h_palindrome: ∃ a b c d: ℕ, b + c = 24 ∧ 
                   ((81 * ((r^6 * (r^6 + 2 * r^5 + 3 * r^4 + 4 * r^3 + 3 * r^2 + 2 * r + 1 + r^2)) = 
                     a * r^7 + b * r^6 + c * r^5 + d * r^4 + d * r^3 + c * r^2 + b * r + a)))):
    r = 26 :=
by
  sorry

end find_base_r_l460_460746


namespace area_enclosed_by_line_and_curve_l460_460294

theorem area_enclosed_by_line_and_curve :
  let line := { p : ℝ × ℝ | p.1 - p.2 = 0 }
  let curve := { p : ℝ × ℝ | p.2 = p.1^2 - 2 * p.1 }
  ∃ (S : ℝ), S = 9 / 2 ∧
    S = ∫ x in 0..3, (x - (x^2 - 2*x)) :=
by
  sorry

end area_enclosed_by_line_and_curve_l460_460294


namespace dihedral_angle_cuboid_l460_460140

theorem dihedral_angle_cuboid
  (A B C D A1 B1 C1 D1 : Point)
  (cube : Cube A B C D A1 B1 C1 D1)
  (dihedral_angle : ∀ {P Q R S : Point}, PlaneAngle (Plane P Q) (Plane R S) → ℝ)
  (B_A1C_D_angle : PlaneAngle (Plane B A1) (Plane C D)):
  dihedral_angle B_A1C_D_angle = 120 :=
by
  sorry

end dihedral_angle_cuboid_l460_460140


namespace solve_equation_l460_460041

theorem solve_equation : ∀ x : ℝ, (x^2 - 1) / (x + 1) = 0 → x = 1 :=
begin
  assume x,
  assume h : (x^2 - 1) / (x + 1) = 0,
  have h1 : x ≠ -1, sorry, -- Denominator should not be zero
  have h2 : x^2 - 1 = 0, sorry, -- Numerator should be zero
  have h3 : x = 1 ∨ x = -1, sorry, -- Solving the quadratic equation
  have h4 : x ≠ -1, from h1, -- Excluding extraneous root
  exact or.resolve_right h3 h4,
end

end solve_equation_l460_460041


namespace computation_correct_l460_460513

noncomputable def compute_expression (π : Real) (h : π < 9) : Real :=
  |(Real.sqrt (|π - |π - 9| - 3) - 1)|

theorem computation_correct (π : Real) (h : π < 9) : compute_expression π h = 1 - Real.sqrt (6 - 2 * π) :=
by
  sorry

end computation_correct_l460_460513


namespace normal_prob_l460_460225

noncomputable def normal_distribution := sorry
noncomputable def prob : ℝ := sorry

theorem normal_prob (σ : ℝ) (h1 : normal_distribution = N(2, σ^2))
  (h2 : prob (λ ξ, ξ < 4) = 0.8) : prob (λ ξ, 0 < ξ ∧ ξ < 2) = 0.3 :=
sorry

end normal_prob_l460_460225


namespace find_a_for_local_min_l460_460645

theorem find_a_for_local_min (a : ℝ) : 
  (∃ f : ℝ → ℝ, f = λ x => x^3 - a * x^2 + 4 * x - 8 ∧ 
   ∀ f' : ℝ → ℝ, deriv f = f' ∧ f'(2) = 0 ∧ (∀ g : ℝ → ℝ, deriv f' = g ∧ g(2) > 0)) → a = 4 :=
by
  sorry

end find_a_for_local_min_l460_460645


namespace num_terms_before_negative_30_l460_460880

theorem num_terms_before_negative_30 :
  let a := 95
  let d := -5
  let n : ℕ := 26
  (100 - 5 * n = -30) →
  n - 1 = 25 :=
by
  intros
  sorry

end num_terms_before_negative_30_l460_460880


namespace max_area_of_triangle_l460_460939

noncomputable def ellipse_eq (y x : ℝ) : Prop := y^2 / 2 + x^2 = 1

noncomputable def max_triangle_area (m : ℝ) : ℝ := sqrt 2 * sqrt (m * (1 - m)^3)

theorem max_area_of_triangle :
  (∀ y x : ℝ, ellipse_eq y x) →
  (∀ k ≠ 0, ∃ P Q : ℝ × ℝ, line_slope PQ k ∧ intersect_ellipse PQ) →
  (∀ m : ℝ, 0 < m ∧ m < 1/2) →
  (∀ m : ℝ, max_triangle_area m ≤ sqrt 2 * sqrt 27 / 256) :=
sorry

end max_area_of_triangle_l460_460939


namespace percentage_decrease_feb_to_mar_l460_460081

-- Given conditions
variables (F J M : ℝ)
def jan_feb_relation : Prop := J = 0.9 * F
def jan_mar_relation : Prop := M = 0.72 * F

-- The theorem to prove
theorem percentage_decrease_feb_to_mar (h1 : jan_feb_relation F J) (h2 : jan_mar_relation M)
: ((F - M) / F) * 100 = 28 :=
sorry

end percentage_decrease_feb_to_mar_l460_460081


namespace sector_area_l460_460586

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (r : ℝ) (area : ℝ) : 
  arc_length = π / 3 ∧ central_angle = π / 6 → arc_length = central_angle * r → area = 1 / 2 * central_angle * r^2 → area = π / 3 :=
by
  sorry

end sector_area_l460_460586


namespace number_of_valid_partitions_l460_460710

open Set

def set_of_15 : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

def valid_partition (X Y : Set ℕ) : Prop :=
  (X ∪ Y = set_of_15) ∧ (X ∩ Y = ∅) ∧ 
  (X.Nonempty ∧ Y.Nonempty) ∧ 
  (|X| ∉ X) ∧ (|Y| ∉ Y)

theorem number_of_valid_partitions : 
  ∃ M, M = 2 ^ 14 ∧ M = (Set.toFinset (Set.powerset set_of_15)).card := 
sorry

end number_of_valid_partitions_l460_460710


namespace complex_number_identity_l460_460380

theorem complex_number_identity : (1 + Complex.i)^2 * (1 - Complex.i) = 2 - 2 * Complex.i :=
by sorry -- Proof will be inserted here, though we are not required to provide it

end complex_number_identity_l460_460380


namespace cos_sq_alpha_cos_sq_beta_range_l460_460989

theorem cos_sq_alpha_cos_sq_beta_range
  (α β : ℝ)
  (h : 3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 - 2 * Real.sin α = 0) :
  (Real.cos α)^2 + (Real.cos β)^2 ∈ Set.Icc (14 / 9) 2 :=
sorry

end cos_sq_alpha_cos_sq_beta_range_l460_460989


namespace problem1_problem2_problem3_problem4_l460_460916

noncomputable def f1 (f : ℝ → ℝ) : Prop :=
∀ x, f (f x - 1) = x + 1

noncomputable def f2 (f : ℝ → ℝ) : Prop :=
∀ x y, f (2 * x + f (3 * y)) = f x + y ^ 5

noncomputable def f3 (f : ℝ → ℝ) : Prop :=
∀ x y, f (f x - f y) = x * f (x - y)

noncomputable def f4 (f : ℝ → ℝ) : Prop :=
∀ x y, f (f x + f y) - f (f x - f y) = 2 * x + 3 * y

-- Prove that given f1, f is injective and surjective
theorem problem1 (f : ℝ → ℝ) (h : f1 f) : function.injective f ∧ function.surjective f := sorry

-- Prove that given f2, there is no function f that is both injective and surjective
theorem problem2 (f : ℝ → ℝ) (h : f2 f) : ¬ (function.injective f ∧ function.surjective f) := sorry

-- Prove that given f3, f is neither injective nor surjective
theorem problem3 (f : ℝ → ℝ) (h : f3 f) : ¬ function.injective f ∧ ¬ function.surjective f := sorry

-- Prove that given f4, f is injective and surjective
theorem problem4 (f : ℝ → ℝ) (h : f4 f) : function.injective f ∧ function.surjective f := sorry

end problem1_problem2_problem3_problem4_l460_460916


namespace intersect_lines_l460_460775

theorem intersect_lines (k : ℝ) :
  (∃ y : ℝ, 3 * 5 - y = k ∧ -5 - y = -10) → k = 10 :=
by
  sorry

end intersect_lines_l460_460775


namespace incorrect_set_expressions_l460_460135

theorem incorrect_set_expressions :
    let expr1 := ¬ ({0} ∈ {1, 2, 3})
    let expr2 := (∅ ⊆ {0})
    let expr3 := ({0, 1, 2} ⊆ {1, 2, 0})
    let expr4 := ¬ (0 ∈ ∅)
    let expr5 := (0 ∩ ∅ = (∅ : set ℕ))
    (expr1 ∧ expr2 ∧ expr3 ∧ expr4 ∧ expr5) →
    (count (λ expr, expr = false) [expr1, expr2, expr3, expr4, expr5] = 3) :=
by
  unfold expr1 expr2 expr3 expr4 expr5
  intros
  count ([¬ ({0} ∈ {1, 2, 3}), ∅ ⊆ {0}, {0, 1, 2} ⊆ {1, 2, 0}, ¬0 ∈ ∅, 0 ∩ ∅ = (∅ : set ℕ)]) (λ expr, expr = false) = 3
  sorry

end incorrect_set_expressions_l460_460135


namespace polynomial_derivative_inequality_l460_460730

/-- 
For any polynomial p(x) with all real and distinct roots, 
prove that (p'(x))^2 ≥ p(x) p''(x). 
-/
theorem polynomial_derivative_inequality 
  (p : Polynomial ℝ) (h1 : ∀ x : ℝ, Polynomial.eval x p = 0 → Polynomial.eval_derivative p x ≠ 0) :
  ∀ x : ℝ, (Polynomial.eval_derivative p x)^2 ≥ Polynomial.eval p x * Polynomial.eval (Polynomial.derivative (Polynomial.derivative p)) x :=
begin
  sorry
end

end polynomial_derivative_inequality_l460_460730


namespace inverse_variation_y_at_x_l460_460637

variable (k x y : ℝ)

theorem inverse_variation_y_at_x :
  (∀ x y k, y = k / x → y = 6 → x = 3 → k = 18) → 
  k = 18 →
  x = 12 →
  y = 18 / 12 →
  y = 3 / 2 := by
  intros h1 h2 h3 h4
  sorry

end inverse_variation_y_at_x_l460_460637


namespace maximize_fraction_l460_460344

theorem maximize_fraction (A B C D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_digits : A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9)
  (h_nonneg : 0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ 0 ≤ D)
  (h_integer : (A + B) % (C + D) = 0) : A + B = 17 :=
sorry

end maximize_fraction_l460_460344


namespace ratio_luminosity_of_altair_to_vega_l460_460290

theorem ratio_luminosity_of_altair_to_vega (m1 m2 : ℝ) (E1 E2 : ℝ)
  (h1 : m1 - m2 = (1/2) * (Real.log (E2^5)) - (1/2) * (Real.log (E1^5)))
  (h2 : m1 = 0)
  (h3 : m2 = 0.75) :
  E2 / E1 = 10^(-3/10) := 
sorry

end ratio_luminosity_of_altair_to_vega_l460_460290


namespace ratio_hector_sandra_l460_460656

noncomputable def ratio_of_baskets : ℕ :=
  let alex_baskets : ℕ := 8
  let sandra_baskets : ℕ := 3 * alex_baskets
  let total_baskets : ℕ := 80
  let hector_baskets : ℕ := total_baskets - alex_baskets - sandra_baskets
  hector_baskets / sandra_baskets

theorem ratio_hector_sandra : ratio_of_baskets = 2 :=
by
  unfold ratio_of_baskets
  rw [Nat.div_eq_of_eq_mul_right (by decide : 24 ≠ 0)]
  calc
    (80 - 8 - 3 * 8) / 3 * 8 = 2 : by decide

end ratio_hector_sandra_l460_460656


namespace largest_number_of_positive_consecutive_integers_l460_460057

theorem largest_number_of_positive_consecutive_integers (n a : ℕ) (h1 : a > 0) (h2 : n > 0) (h3 : (n * (2 * a + n - 1)) / 2 = 45) : 
  n = 9 := 
sorry

end largest_number_of_positive_consecutive_integers_l460_460057


namespace count_pos_int_with_at_most_three_diff_digits_below_3000_l460_460622

theorem count_pos_int_with_at_most_three_diff_digits_below_3000 : 
  ∃ (n : ℕ), (∀ x : ℕ, 0 < x ∧ x < 3000 → at_most_three_distinct_digits x) ∧ n = 891 :=
sorry

-- Definition to check if a number has at most three different digits
def at_most_three_distinct_digits (n : ℕ) : Prop :=
  (finset.image (λ (d : ℕ), (d, finset.card (finset.image (λ i, i) (finset.of_multiset (multiset.of_list (nat.digits 10 n)))))) {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}).card ≤ 3

end count_pos_int_with_at_most_three_diff_digits_below_3000_l460_460622


namespace set_intersection_and_difference_union_range_of_a_l460_460975

theorem set_intersection_and_difference_union (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) :
  U = Set.univ ∧ A = {x | -1 ≤ x ∧ x < 3} ∧ B = {x | x ≥ 2} →
  (A ∩ B = {x | 2 ≤ x ∧ x < 3}) ∧ (U \ A ∪ B = {x | x < -1 ∨ x ≥ 2}) :=
  by
  intros h
  cases h with hU hRest
  cases hRest with hA hB
  { sorry }

theorem range_of_a (a : ℝ) (B : Set ℝ) (C : Set ℝ) :
  B = {x | x ≥ 2} ∧ C = {x | x > -(a / 2)} ∧ (B ∪ C = C) → a > 4 :=
  by
  intros h
  cases h with hB hRest
  cases hRest with hC hBC
  { sorry }

end set_intersection_and_difference_union_range_of_a_l460_460975


namespace find_trig_value_l460_460252

-- Defining the vectors
def a (α : ℝ) : ℝ × ℝ := (Real.cos α, -2)
def b (α : ℝ) : ℝ × ℝ := (Real.sin α, 1)

-- Parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

theorem find_trig_value (α : ℝ) 
  (h : parallel (a α) (b α)) : 2 * Real.sin α * Real.cos α = -4 / 5 := by
  sorry

end find_trig_value_l460_460252


namespace maximum_value_MN_over_AB_l460_460949

noncomputable def hyperbola : Type := sorry
noncomputable def right_focus (H : hyperbola) : Type := sorry
noncomputable def right_directrix (H : hyperbola) : Type := sorry
noncomputable def right_branch_points (H : hyperbola) := {p : ℝ × ℝ // p.1^2 - p.2^2 = 1}
noncomputable def perpendicular (A B F : ℝ × ℝ) : Prop := sorry
noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def projection_on_directrix (M : ℝ × ℝ) (l : Type) : ℝ × ℝ := sorry

theorem maximum_value_MN_over_AB 
  (H : hyperbola)
  (F : right_focus H)
  (l : right_directrix H)
  (A B : right_branch_points H)
  (h_perp : perpendicular A.val B.val F)
  (M := midpoint A.val B.val)
  (N := projection_on_directrix M l)
  : ∃ C : ℝ, C = 1/2 ∧ ∀ A B : right_branch_points H, h_perp → (|MN| / |AB|) ≤ C
:= sorry

end maximum_value_MN_over_AB_l460_460949


namespace card_probability_not_passing_through_l460_460414

theorem card_probability_not_passing_through {a : ℤ} :
  (a = -3 ∨ a = -2 ∨ a = -1 ∨ a = 1 ∨ a = 2) →
  (∑ b in {-3, -2, -1, 1, 2}.to_finset, if b = a then 1 else 0) = 1 →
  (∑ b in {-3, -1, 2}.to_finset, if (1 - b^2 + 2 - b ≠ 0) then 1 else 0) / 5 = 3 / 5 :=
sorry

end card_probability_not_passing_through_l460_460414


namespace otimes_2_5_l460_460720

def otimes (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem otimes_2_5 : otimes 2 5 = 23 :=
by
  sorry

end otimes_2_5_l460_460720


namespace sufficient_condition_l460_460560

theorem sufficient_condition (A B : Set α) (h : A ⊆ B) (x : α) : x ∈ A → x ∈ B :=
  by
    intro h1
    apply h
    exact h1

end sufficient_condition_l460_460560


namespace max_min_f_when_a1_bneg1_exists_a_b_in_rationals_l460_460925

section Problem1
variables {x : ℝ}

def f (a b : ℝ) (x : ℝ) := -2 * a * sin(2 * x + π / 6) + 2 * a + b

theorem max_min_f_when_a1_bneg1 : 
  f 1 (-1) x ≤ 3 ∧ f 1 (-1) x ≥ -1 :=
sorry
end Problem1

section Problem2
variables {x : ℝ}
variables {a b : ℝ}

def f (a b : ℝ) (x : ℝ) := -2 * a * sin(2 * x + π / 6) + 2 * a + b

theorem exists_a_b_in_rationals : ∃ (a b : ℚ), 
  (∀ x ∈ set.Icc (π / 4) (3 * π / 4), (f a b x) ∈ set.Icc (-3) (real.sqrt 3 - 1)) :=
sorry
end Problem2

end max_min_f_when_a1_bneg1_exists_a_b_in_rationals_l460_460925


namespace intersection_A_B_l460_460272

def A : Set ℤ := { -2, -1, 0, 1, 2 }
def B : Set ℤ := { x : ℤ | x < 1 }

theorem intersection_A_B : A ∩ B = { -2, -1, 0 } :=
by sorry

end intersection_A_B_l460_460272


namespace max_value_g_l460_460165

noncomputable def g : ℕ → ℕ
| n := if n < 12 then n + 12 else g (n - 6)

theorem max_value_g : ∃ M, ∀ n, g n ≤ M ∧ (∃ n', g n' = M) :=
begin
  use 23,
  sorry -- Placeholder for the actual proof.
end

end max_value_g_l460_460165


namespace tony_doctors_appointment_l460_460796

-- Define the conditions
variables (d_groceries d_haircut d_halfway d_total d_appointment : ℕ)
hypotheses (h1 : d_groceries = 10)
           (h2 : d_haircut = 15)
           (h3 : d_halfway = 15)
           (h4 : d_total = 2 * d_halfway)

-- Lean statement to prove the required distance
theorem tony_doctors_appointment :
  d_appointment = d_total - (d_groceries + d_haircut) :=
sorry


end tony_doctors_appointment_l460_460796


namespace fish_tagged_initially_l460_460652

theorem fish_tagged_initially (N T : ℕ) (hN : N = 1500) 
  (h_ratio : 2 / 50 = (T:ℕ) / N) : T = 60 :=
by
  -- The proof is omitted
  sorry

end fish_tagged_initially_l460_460652


namespace solve_system_l460_460764

theorem solve_system (x y z w : ℝ) :
  x - y + z - w = 2 ∧
  x^2 - y^2 + z^2 - w^2 = 6 ∧
  x^3 - y^3 + z^3 - w^3 = 20 ∧
  x^4 - y^4 + z^4 - w^4 = 60 ↔
  (x = 1 ∧ y = 2 ∧ z = 3 ∧ w = 0) ∨
  (x = 1 ∧ y = 0 ∧ z = 3 ∧ w = 2) ∨
  (x = 3 ∧ y = 2 ∧ z = 1 ∧ w = 0) ∨
  (x = 3 ∧ y = 0 ∧ z = 1 ∧ w = 2) :=
sorry

end solve_system_l460_460764


namespace solve_inequality_eq_l460_460014

theorem solve_inequality_eq (x : ℝ) :
  (sqrt (x^3 + x - 90) + 7) * (abs (x^3 - 10 * x^2 + 31 * x - 28)) ≤ 0 ↔ x = 3 + real.sqrt 2 :=
begin
  sorry
end

end solve_inequality_eq_l460_460014


namespace triangle_ABC_properties_l460_460650

theorem triangle_ABC_properties
  (a b c : ℝ)
  (A B C : ℝ)
  (area_ABC : Real.sqrt 15 * 3 = 1/2 * b * c * Real.sin A)
  (cos_A : Real.cos A = -1/4)
  (b_minus_c : b - c = 2) :
  (a = 8 ∧ Real.sin C = Real.sqrt 15 / 8) ∧
  (Real.cos (2 * A + Real.pi / 6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16) := by
  sorry

end triangle_ABC_properties_l460_460650


namespace handshakes_at_gathering_l460_460011

def total_handshakes (num_couples : ℕ) (exceptions : ℕ) : ℕ :=
  let num_people := 2 * num_couples
  let handshakes_per_person := num_people - exceptions - 1
  num_people * handshakes_per_person / 2

theorem handshakes_at_gathering : total_handshakes 6 2 = 54 := by
  sorry

end handshakes_at_gathering_l460_460011


namespace sin_cos_inequality_l460_460354

open Real

theorem sin_cos_inequality 
  (x : ℝ) (hx : 0 < x ∧ x < π / 2) 
  (m n : ℕ) (hmn : n > m)
  : 2 * abs (sin x ^ n - cos x ^ n) ≤ 3 * abs (sin x ^ m - cos x ^ m) :=
sorry

end sin_cos_inequality_l460_460354


namespace solve_eq_l460_460375

-- Defining the condition
def eq_condition (x : ℝ) : Prop := (x - 3) ^ 2 = x ^ 2 - 9

-- The statement we need to prove
theorem solve_eq (x : ℝ) (h : eq_condition x) : x = 3 :=
by
  sorry

end solve_eq_l460_460375


namespace number_of_supermarkets_in_us_42_l460_460086

noncomputable def number_of_supermarkets_in_us (total_supermarkets : ℕ) (diff_us_canada : ℕ) (us_supermarkets : ℕ) (canada_supermarkets : ℕ) : Prop :=
  total_supermarkets = us_supermarkets + canada_supermarkets ∧
  us_supermarkets = canada_supermarkets + diff_us_canada ∧
  total_supermarkets = 70 ∧
  diff_us_canada = 14

theorem number_of_supermarkets_in_us_42 :
  ∃ us_supermarkets canada_supermarkets : ℕ,
    number_of_supermarkets_in_us 70 14 us_supermarkets canada_supermarkets ∧
    us_supermarkets = 42 :=
begin
  sorry
end

end number_of_supermarkets_in_us_42_l460_460086


namespace remainder_of_functions_mod_1000_l460_460693

noncomputable def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def number_of_functions := 
  let N := 8 * ∑ k in (Finset.range 7).filter (λ k, k > 0), (Nat.choose 7 k) * (k ^ (7 - k)) 
  in N

theorem remainder_of_functions_mod_1000 : number_of_functions % 1000 = 576 :=
by 
  let N := number_of_functions
  have h1 : N = 50576 := sorry
  have h2 : 50576 % 1000 = 576 := rfl
  rw [h1, h2]

end remainder_of_functions_mod_1000_l460_460693


namespace closest_fraction_l460_460289

theorem closest_fraction (h : 24 / 150 ≈ 0.16) :
    (closest_to (24 / 150) [1/5, 1/6, 1/7, 1/8, 1/9] = 1/6) :=
sorry

end closest_fraction_l460_460289


namespace bus_probability_l460_460142

/-- Probability that the first bus is red, and exactly 4 blue buses (out of 6 non-red) in the lineup of 7 buses /-
theorem bus_probability (total_buses : ℕ) (red_buses : ℕ) (blue_buses : ℕ) (yellow_buses : ℕ) : 
  red_buses = 5 ∧ blue_buses = 6 ∧ yellow_buses = 5 ∧ total_buses = red_buses + blue_buses + yellow_buses → 
  ((red_buses / total_buses) * ((Nat.choose blue_buses 4) * (Nat.choose yellow_buses 2) * (Nat.choose (red_buses - 1) 0) / (Nat.choose (total_buses - 1) 6))) = 75 / 8080 :=
by
  sorry

end bus_probability_l460_460142


namespace ratio_of_speeds_l460_460079

theorem ratio_of_speeds (L V : ℝ) (R : ℝ) (h1 : L > 0) (h2 : V > 0) (h3 : R ≠ 0)
  (h4 : (1.48 * L) / (R * V) = (1.40 * L) / V) : R = 37 / 35 :=
by
  -- Proof would be inserted here
  sorry

end ratio_of_speeds_l460_460079


namespace remainder_is_400_l460_460698

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def num_functions (f : ℕ → ℕ) : ℕ :=
  if ∃ c ∈ A, ∀ x ∈ A, f(f(x)) = c then 1 else 0

def N : ℕ :=
  8 * (7 + 672 + 1701 + 1792 + 875 + 252 + 1)

def remainder : ℕ :=
  N % 1000

theorem remainder_is_400 : remainder = 400 :=
  sorry

end remainder_is_400_l460_460698


namespace range_of_k_l460_460338

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x
noncomputable def g (x : ℝ) : ℝ := x / (Real.exp x)

theorem range_of_k (k : ℝ) (hk : 0 < k) :
  (∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 * f(x2) * k ≤ (k + 1) * g(x1)) →
  k ≥ 1 / (2 * Real.exp 1 - 1) :=
sorry

end range_of_k_l460_460338


namespace find_irrationals_l460_460499

noncomputable def is_irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), x = q

theorem find_irrationals (a b c d e f g h i j k l : ℝ) (h_a : a = π / 2)
 (h_b : b = 22 / 7) (h_c : c = 0.1414) (h_d : d = (9 : ℝ)^(1/3)) (h_e : e = (1/2)^(1/2))
 (h_f : f = -5/2) (h_g : ∃ (n : ℕ) (f : ℕ → ℝ), (∀ (m n : ℕ), m ≠ n → f m ≠ f n) ∧
   g = (λ n, if n = 0 then 0.1 else if n % 2 = 1 then 0 else 0.1 / (10^n))),
 (h_h : h = -((1/16)^(1/2))) (h_i : i = 0) (h_j : j = 1 - (2^(1/2))) (h_k : k = (5^(1/2))/2)
 (h_l : l = abs ((4^(1/2)) - 1)) :
 {x : ℝ | is_irrational x ∧ x ∈ {a, b, c, d, e, f, g, h, i, j, k, l}} =
  {a, d, e, g, j, k, l} := sorry

end find_irrationals_l460_460499


namespace projection_of_a_onto_b_l460_460615

def proj_vector (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot := a.1 * b.1 + a.2 * b.2
  let mag_b_sq := b.1 * b.1 + b.2 * b.2
  (dot / mag_b_sq) * b.1, (dot / mag_b_sq) * b.2

theorem projection_of_a_onto_b :
  proj_vector (2, 4) (-1, 2) = (-6/5, 12/5) := by
  sorry

end projection_of_a_onto_b_l460_460615


namespace total_cookies_sold_l460_460371

-- Definitions based on conditions
def packs_sold_first_neigh_Robyn := 15
def packs_sold_first_neigh_Lucy := 12

def packs_sold_second_neigh_Robyn := 23
def packs_sold_second_neigh_Lucy := 15

def packs_sold_third_neigh_Robyn := 17
def packs_sold_third_neigh_Lucy := 16

def packs_sold_first_park_total := 25
def packs_sold_first_park_Lucy, packs_sold_first_park_Robyn : ℕ :=
  let L := packs_sold_first_park_total / 3 in
  L, 2 * L + 1

def packs_sold_second_park_total := 35
def packs_sold_second_park_Robyn, packs_sold_second_park_Lucy : ℕ :=
  let R := (packs_sold_second_park_total - 5) / 2 in
  R, R + 5

-- The final proof statement
theorem total_cookies_sold :
  let total_Robyn := 
    packs_sold_first_neigh_Robyn + 
    packs_sold_second_neigh_Robyn + 
    packs_sold_third_neigh_Robyn + 
    packs_sold_first_park_Robyn + 
    packs_sold_second_park_Robyn in
  let total_Lucy := 
    packs_sold_first_neigh_Lucy + 
    packs_sold_second_neigh_Lucy + 
    packs_sold_third_neigh_Lucy + 
    packs_sold_first_park_Lucy + 
    packs_sold_second_park_Lucy in
  total_Robyn + total_Lucy = 158 := by
  sorry

end total_cookies_sold_l460_460371


namespace rounding_range_l460_460780

theorem rounding_range (a : ℝ) (h : a.round = 0.270) : 0.2695 ≤ a ∧ a < 0.2705 :=
sorry

end rounding_range_l460_460780


namespace terminal_side_in_first_quadrant_l460_460263

noncomputable def theta := -5

def in_first_quadrant (θ : ℝ) : Prop :=
  by sorry

theorem terminal_side_in_first_quadrant : in_first_quadrant theta := 
  by sorry

end terminal_side_in_first_quadrant_l460_460263


namespace geometric_sequence_sum_5_is_75_l460_460657

noncomputable def geometric_sequence_sum_5 (a r : ℝ) : ℝ :=
  a * (1 + r + r^2 + r^3 + r^4)

theorem geometric_sequence_sum_5_is_75 (a r : ℝ)
  (h1 : a * (1 + r + r^2) = 13)
  (h2 : a * (1 - r^7) / (1 - r) = 183) :
  geometric_sequence_sum_5 a r = 75 :=
sorry

end geometric_sequence_sum_5_is_75_l460_460657


namespace jerry_current_average_l460_460682

-- Definitions for Jerry's first 3 tests average and conditions
variable (A : ℝ)

-- Condition details
def total_score_of_first_3_tests := 3 * A
def new_desired_average := A + 2
def total_score_needed := (A + 2) * 4
def score_on_fourth_test := 93

theorem jerry_current_average :
  (total_score_needed A = total_score_of_first_3_tests A + score_on_fourth_test) → A = 85 :=
by
  sorry

end jerry_current_average_l460_460682


namespace general_term_formula_sum_of_first_n_bn_l460_460572

-- Define the conditions
def a (n : ℕ) : ℤ := a1 + (n - 1) * d
def b (n : ℕ) : ℤ := 2 ^ (a n)

-- Given conditions
axiom a5_eq_9 : a 5 = 9
axiom a2_plus_a6_eq_14 : a 2 + a 6 = 14

-- Goals
theorem general_term_formula :
  ∃ (a1 d : ℤ), (a 5 = 9) ∧ (a 2 + a 6 = 14) ∧ 
    ∀ n : ℕ, a n = a1 + (n - 1) * d := sorry

theorem sum_of_first_n_bn (n : ℕ) :
  ∃ a1 d : ℤ, (a 5 = 9) ∧ (a 2 + a 6 = 14) → 
    ∑ i in finset.range n, b i = (2 * 4^n / 3) - (2 / 3) := sorry

end general_term_formula_sum_of_first_n_bn_l460_460572


namespace find_sum_lent_l460_460811

variable (P : ℝ)

/-- Given that the annual interest rate is 4%, and the interest earned in 8 years
amounts to Rs 340 less than the sum lent, prove that the sum lent is Rs 500. -/
theorem find_sum_lent
  (h1 : ∀ I, I = P - 340 → I = (P * 4 * 8) / 100) : 
  P = 500 :=
by
  sorry

end find_sum_lent_l460_460811


namespace pencil_cost_is_correct_l460_460739

def classes : Nat := 6
def folders_per_class : Nat := 1
def pencils_per_class : Nat := 3
def erasers_per_pencils : Nat := 6
def folder_cost : Nat := 6
def eraser_cost : Nat := 1
def paint_cost : Nat := 5
def total_spent : Nat := 80

theorem pencil_cost_is_correct (c: Nat) : c = 2 := by
  -- Definitions of the cost calculations:
  let folders_needed := classes * folders_per_class
  let folders_cost := folders_needed * folder_cost

  let pencils_needed := classes * pencils_per_class
  let erasers_needed := pencils_needed / erasers_per_pencils
  let erasers_cost := erasers_needed * eraser_cost

  let total_other_costs := folders_cost + erasers_cost + paint_cost
  let remaining_amount := total_spent - total_other_costs

  let pencil_cost := remaining_amount / pencils_needed

  -- Assert the cost per pencil is the correct answer:
  have : c = pencil_cost := sorry
  
  -- Correct answer based on the problem conditions
  have : pencil_cost = 2 := by
    -- Preliminary calculations are included to assert the result.
    sorry
  
  show c = 2, from sorry

end pencil_cost_is_correct_l460_460739


namespace locus_of_projections_is_correct_l460_460131

variable (A B O : Point)
variable (line : Line)
variable (proj : Point → Line → Point)

-- Definition of projection to capture the right angle condition
def is_projection (P : Point) (line : Line) (P' : Point) : Prop :=
  angle O P' P = π / 2

-- Definition of projection of segment AB onto lines passing through O
def projections_of_segment_on_lines (A B O : Point) : Set Point :=
  { P' | ∃ (P : Point) (line : Line), (P = A ∨ P = B) ∧ line.through(O) ∧ is_projection P line P' }

def locus_of_projections := 
  { P | P ∈ Circle O A.radius ∨ P ∈ Circle O B.radius } ∖ 
  { P | P ∉ Circle O A.radius ∧ P ∉ Circle O B.radius }

theorem locus_of_projections_is_correct (A B O : Point) :
  (projections_of_segment_on_lines A B O = locus_of_projections A B O) :=
sorry

end locus_of_projections_is_correct_l460_460131


namespace union_M_N_equals_0_1_5_l460_460335

def M : Set ℝ := { x | x^2 - 6 * x + 5 = 0 }
def N : Set ℝ := { x | x^2 - 5 * x = 0 }

theorem union_M_N_equals_0_1_5 : M ∪ N = {0, 1, 5} := by
  sorry

end union_M_N_equals_0_1_5_l460_460335


namespace total_turnover_seven_days_monthly_growth_rate_l460_460891

open BigOperators

-- Part 1: Total Turnover for Seven Days
theorem total_turnover_seven_days (turnover_six_days : ℝ) (percentage_seventh_day : ℝ) : 
  turnover_six_days = 450 → percentage_seventh_day = 0.12 → 
  (turnover_six_days + (turnover_six_days * percentage_seventh_day)) = 504 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

-- Part 2: Monthly Growth Rate in August and September
theorem monthly_growth_rate (turnover_july : ℝ) (final_turnover_seven_days : ℝ) (growth_rate : ℝ) :
  turnover_july = 350 → final_turnover_seven_days = 504 → 
  ((1 + growth_rate)^2 = final_turnover_seven_days / turnover_july) → growth_rate = 0.2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  field_simp at h3
  sorry

end total_turnover_seven_days_monthly_growth_rate_l460_460891


namespace find_x_l460_460325

variables (a b x : ℝ)
variables (pos_a : a > 0) (pos_b : b > 0) (pos_x : x > 0)

theorem find_x : ((2 * a) ^ (2 * b) = (a^2) ^ b * x ^ b) → (x = 4) := by
  sorry

end find_x_l460_460325


namespace relation_between_p_and_q_l460_460641

theorem relation_between_p_and_q (p q : ℝ) (α : ℝ) 
  (h1 : α + 2 * α = -p) 
  (h2 : α * (2 * α) = q) : 
  2 * p^2 = 9 * q := 
by 
  -- simplifying the provided conditions
  sorry

end relation_between_p_and_q_l460_460641


namespace train_length_l460_460129

theorem train_length (V L : ℝ) (h1 : L = V * 18) (h2 : L + 550 = V * 51) : L = 300 := sorry

end train_length_l460_460129


namespace find_snail_square_l460_460340

def is_snail (x : ℕ) : Prop :=
  let digits := (to_digits x).to_multiset
  ∃ a b c : ℕ, (a + 2 = b + 1 ∧ b + 1 = c)
    ∧ (digits = (to_digits a).to_multiset ∪ (to_digits b).to_multiset ∪ (to_digits c).to_multiset)

theorem find_snail_square : ∃ n : ℕ, 1000 ≤ n^2 ∧ n^2 < 10000 ∧ is_snail (n^2) ∧ ∃ m, n^2 = m^2 :=
by
  sorry

end find_snail_square_l460_460340


namespace sum_of_ages_ten_years_ago_l460_460466

-- Definitions of the current ages and the relationship
variables (F S : ℕ)

-- Conditions
def father_age := 64
def son_age := 16
def age_relation := father_age = 4 * son_age

-- Theorem to prove the sum of their ages ten years ago
theorem sum_of_ages_ten_years_ago : age_relation → (father_age - 10) + (son_age - 10) = 60 :=
by
  assume h : age_relation
  sorry

end sum_of_ages_ten_years_ago_l460_460466


namespace solution_set_f_l460_460235

noncomputable def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem solution_set_f (x : ℝ)  : (f(x) ≥ x^2 - 8x + 15) ↔ (5 - real.sqrt 3 ≤ x ∧ x ≤ 6) :=
sorry

end solution_set_f_l460_460235


namespace volume_of_lemon_juice_in_glass_l460_460834

noncomputable def pi : ℝ := Real.pi

theorem volume_of_lemon_juice_in_glass :
  let height := 8 / 2                  -- height of lemonade
  let radius := 3 / 2                  -- radius of the cylindrical glass
  let volume_lemonade := pi * radius^2 * height             -- volume of the lemonade
  let ratio_lemon_juice := 1 / 6                             -- ratio of lemon juice in lemonade
  let volume_lemon_juice := volume_lemonade * ratio_lemon_juice -- volume of lemon juice
  volume_lemon_juice = 3 * pi / 2      -- simplified expression of the volume
  → volume_lemon_juice ≈ 4.71 := sorry

end volume_of_lemon_juice_in_glass_l460_460834


namespace maria_anna_ages_l460_460406

theorem maria_anna_ages : 
  ∃ (x y : ℝ), x + y = 44 ∧ x = 2 * (y - (- (1/2) * x + (3/2) * ((2/3) * y))) ∧ x = 27.5 ∧ y = 16.5 := by 
  sorry

end maria_anna_ages_l460_460406


namespace total_votes_l460_460082

theorem total_votes (V : ℝ) (h1 : 0.70 * V = V - 240) (h2 : 0.30 * V = 240) : V = 800 :=
by
  sorry

end total_votes_l460_460082


namespace closest_and_farthest_correct_l460_460928

def hour_deg (t : Nat) : Float := 180 + (t / 2)
def minute_deg (t : Nat) : Float := t * 6
def sep (t : Nat) : Float :=
  Float.abs (hour_deg t - minute_deg t)

def times : List Nat := [30, 31, 32, 33, 34, 35]

def closest_time (ts : List Nat) : Nat :=
  ts.argmin sep

def farthest_time (ts : List Nat) : Nat :=
  ts.argmax sep

theorem closest_and_farthest_correct :
  closest_time times = 33 ∧ farthest_time times = 30 :=
by
  sorry

end closest_and_farthest_correct_l460_460928


namespace monotonic_intervals_max_min_values_on_interval_l460_460241

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem monotonic_intervals :
  (∀ x > -2, 0 < (x + 2) * Real.exp x) ∧ (∀ x < -2, (x + 2) * Real.exp x < 0) :=
by
  sorry

theorem max_min_values_on_interval :
  let a := -4
  let b := 0
  let f_a := (-4 + 1) * Real.exp (-4)
  let f_b := (0 + 1) * Real.exp 0
  let f_c := (-2 + 1) * Real.exp (-2)
  (f b = 1) ∧ (f_c = -1 / Real.exp 2) ∧ (f_a < f_b) ∧ (f_a < f_c) ∧ (f_c < f_b) :=
by
  sorry

end monotonic_intervals_max_min_values_on_interval_l460_460241


namespace fruit_basket_count_l460_460260

theorem fruit_basket_count (apples oranges : ℕ) (h_apples : apples = 4) (h_oranges : oranges = 12):
  4 + 12 > 0 ∧ apples >= 0 ∧ 12 >= 2 →
  (card (set_of (λ b : ℕ × ℕ, b.fst + b.snd > 0 ∧ b.snd ≥ 2 ∧ b.fst ≤ 4 ∧ b.snd ≤ 12)) = 55) :=
by
  intros _ _
  sorry

end fruit_basket_count_l460_460260


namespace triangle_ratio_l460_460651

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ)
  (hA : A = 2 * Real.pi / 3)
  (h_a : a = Real.sqrt 3 * c)
  (h_angle_sum : A + B + C = Real.pi)
  (h_law_of_sines : a / Real.sin A = c / Real.sin C) :
  b / c = 1 :=
sorry

end triangle_ratio_l460_460651


namespace probability_correct_l460_460093

-- Define the conditions of the problem
def total_white_balls : ℕ := 6
def total_black_balls : ℕ := 5
def total_balls : ℕ := total_white_balls + total_black_balls
def total_ways_draw_two_balls : ℕ := Nat.choose total_balls 2
def ways_choose_one_white_ball : ℕ := Nat.choose total_white_balls 1
def ways_choose_one_black_ball : ℕ := Nat.choose total_black_balls 1
def total_successful_outcomes : ℕ := ways_choose_one_white_ball * ways_choose_one_black_ball

-- Define the probability calculation
def probability_drawing_one_white_one_black : ℚ := total_successful_outcomes / total_ways_draw_two_balls

-- State the theorem
theorem probability_correct :
  probability_drawing_one_white_one_black = 6 / 11 :=
by
  sorry

end probability_correct_l460_460093


namespace fish_price_eq_shrimp_price_l460_460842

-- Conditions
variable (x : ℝ) -- regular price for a full pound of fish
variable (h1 : 0.6 * (x / 4) = 1.50) -- quarter-pound fish price after 60% discount
variable (shrimp_price : ℝ) -- price per pound of shrimp
variable (h2 : shrimp_price = 10) -- given shrimp price

-- Proof Statement
theorem fish_price_eq_shrimp_price (h1 : 0.6 * (x / 4) = 1.50) (h2 : shrimp_price = 10) :
  x = 10 ∧ x = shrimp_price :=
by
  sorry

end fish_price_eq_shrimp_price_l460_460842


namespace sin_alpha_eq_three_fifths_l460_460227

theorem sin_alpha_eq_three_fifths :
  let α := Angle.pi - asin ((3 : ℤ) / (5 : ℤ)) in
  let x := -4 in
  let y := 3 in
  let r := Real.sqrt (x^2 + y^2) in
  sin α = y / r :=
by
  sorry

end sin_alpha_eq_three_fifths_l460_460227


namespace American_carmakers_produce_l460_460497

theorem American_carmakers_produce :
  let first := 1000000
  let second := first + 500000
  let third := first + second
  let fourth := 325000
  let fifth := 325000
  let total := first + second + third + fourth + fifth
  total = 5650000 :=
by
  let first := 1000000
  let second := first + 500000
  let third := first + second
  let fourth := 325000
  let fifth := 325000
  let total := first + second + third + fourth + fifth
  show total = 5650000
  sorry

end American_carmakers_produce_l460_460497


namespace profit_of_150_cents_requires_120_oranges_l460_460112

def cost_price_per_orange := 15 / 4  -- cost price per orange in cents
def selling_price_per_orange := 30 / 6  -- selling price per orange in cents
def profit_per_orange := selling_price_per_orange - cost_price_per_orange  -- profit per orange in cents
def required_oranges_to_make_profit := 150 / profit_per_orange  -- number of oranges to get 150 cents of profit

theorem profit_of_150_cents_requires_120_oranges :
  required_oranges_to_make_profit = 120 :=
by
  -- the actual proof will follow here
  sorry

end profit_of_150_cents_requires_120_oranges_l460_460112


namespace weight_difference_twenty_on_one_pan_l460_460046

theorem weight_difference_twenty_on_one_pan :
  ∃ (l r : Finset (Fin 40)), l.card = 10 ∧ r.card = 10 ∧
  (∀ w ∈ l, w.val % 2 = 0) ∧ (∀ w ∈ r, w.val % 2 = 1) ∧
  ((Finset.sum l (λ x, (x : ℕ) + 1)) = (Finset.sum r (λ x, (x : ℕ) + 1))) ∧
  ∃ (a b ∈ l ∪ r), abs (a.val - b.val) = 20 :=
by
  sorry

end weight_difference_twenty_on_one_pan_l460_460046


namespace sum_of_possible_values_of_x_l460_460463

theorem sum_of_possible_values_of_x (x : ℝ) (h : (x - 3) * (x + 4) = 3 * real.pi * (x - 2)^2) :
  x + (12 * real.pi + 1) / (3 * real.pi - 1) = (12 * real.pi + 1) / (3 * real.pi - 1) :=
by
  sorry

end sum_of_possible_values_of_x_l460_460463


namespace reality_show_duration_l460_460347

variable (x : ℕ)

theorem reality_show_duration :
  (5 * x + 10 = 150) → (x = 28) :=
by
  intro h
  sorry

end reality_show_duration_l460_460347


namespace problem_statement_l460_460551

theorem problem_statement (p x : ℝ) (h : 0 ≤ p ∧ p ≤ 4) :
  (x^2 + p*x > 4*x + p - 3) ↔ (x > 3 ∨ x < -1) := by
sorry

end problem_statement_l460_460551


namespace arithmetic_mean_pq_is_10_l460_460021

variables {p q r : ℝ}

theorem arithmetic_mean_pq_is_10 
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 27)
  (h3 : r - p = 34) 
  : (p + q) / 2 = 10 :=
by 
  exact h1

end arithmetic_mean_pq_is_10_l460_460021


namespace tangent_length_general_tangent_length_isosceles_tangent_length_opposite_directions_l460_460247

theorem tangent_length_general
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let c := real.sqrt (a^2 + b^2) in
  (c * (a + b)^2) / (2 * a * b) = (c * (a + b)^2) / (2 * a * b) :=
by sorry

theorem tangent_length_isosceles
  (a : ℝ) (ha : 0 < a) :
  let c := real.sqrt (a^2 + a^2) in
  (c * (a + a)^2) / (2 * a * a) = 2 * a * real.sqrt 2 :=
by sorry

theorem tangent_length_opposite_directions
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let c := real.sqrt (a^2 + b^2) in
  (c * (a - b)^2) / (2 * a * b) = (c * (a - b)^2) / (2 * a * b) :=
by sorry

end tangent_length_general_tangent_length_isosceles_tangent_length_opposite_directions_l460_460247


namespace problem_statement_l460_460138

-- Define the functions
def f (x : ℝ) : ℝ := Real.exp x
def g (x : ℝ) : ℝ := Real.tan x
def h (x : ℝ) : ℝ := x^3 - x
def i (x : ℝ) : ℝ := Real.log ((2 + x) / (2 - x))

-- Prove that i(x) is an odd function and increasing in its domain (-2, 2)
theorem problem_statement : 
  (∀ x : ℝ, -2 < x ∧ x < 2 → i (-x) = -i x) ∧ 
  (∀ x y : ℝ, -2 < x ∧ x < 2 ∧ -2 < y ∧ y < 2 ∧ x < y → i x < i y) := by
  sorry

end problem_statement_l460_460138


namespace cos_alpha_given_tan_alpha_and_quadrant_l460_460561

theorem cos_alpha_given_tan_alpha_and_quadrant 
  (α : ℝ) 
  (h1 : Real.tan α = -1/3)
  (h2 : π/2 < α ∧ α < π) : 
  Real.cos α = -3*Real.sqrt 10 / 10 :=
by
  sorry

end cos_alpha_given_tan_alpha_and_quadrant_l460_460561


namespace train_length_is_299_98_l460_460488

noncomputable def length_of_train 
    (v_train : ℝ)       -- speed of the train in kmph
    (v_man : ℝ)         -- speed of the man in kmph
    (t_pass : ℝ)        -- time it takes for the train to pass the man in seconds
    (convert_factor : ℝ) -- conversion factor from kmph to m/s
    : ℝ := 
    let v_rel := v_train - v_man in -- relative speed in kmph
    let v_rel_mps := v_rel * convert_factor in -- convert to m/s
    v_rel_mps * t_pass -- length of the train in meters

theorem train_length_is_299_98 
    (v_train : ℝ) 
    (v_man : ℝ) 
    (t_pass : ℝ)
    (convert_factor : ℝ) :
    v_train = 68 → 
    v_man = 8 →
    t_pass = 5.999520038396929 →
    convert_factor = 1000 / 3600 →
    length_of_train v_train v_man t_pass convert_factor = 299.97600191984645 :=
by
  intros
  sorry

end train_length_is_299_98_l460_460488


namespace evaluate_expression_l460_460525

theorem evaluate_expression : -(16 / 4 * 7 - 50 + 5 * 7) = -13 :=
by
  sorry

end evaluate_expression_l460_460525


namespace sum_of_factors_eq_18_l460_460410

theorem sum_of_factors_eq_18 (x : ℤ) (h1 : ∑ d in (finset.filter (λ d, x % d = 0) (finset.range (x + 1))), d = 18) (h2 : 2 ∣ x) : x = 10 :=
sorry

end sum_of_factors_eq_18_l460_460410


namespace remainder_17_pow_63_div_7_l460_460064

theorem remainder_17_pow_63_div_7 :
  (17^63) % 7 = 6 :=
by
  have h: 17 % 7 = 3 := rfl
  sorry

end remainder_17_pow_63_div_7_l460_460064


namespace willie_stickers_l460_460440

def num_stickers_start_with (given_away remaining initial : Nat) : Prop :=
  remaining + given_away = initial

theorem willie_stickers : ∃ initial, num_stickers_start_with 7 29 initial ∧ initial = 36 :=
by {
  use 36,
  split,
  { -- prove that remaining + given_away = initial
    show 29 + 7 = 36,
    exact rfl,
  },
  { -- prove that initial = 36
    show 36 = 36,
    exact rfl,
  }
}

end willie_stickers_l460_460440


namespace ripe_mangoes_remaining_l460_460821

theorem ripe_mangoes_remaining
  (initial_mangoes : ℕ)
  (ripe_fraction : ℚ)
  (consume_fraction : ℚ)
  (initial_total : initial_mangoes = 400)
  (ripe_ratio : ripe_fraction = 3 / 5)
  (consume_ratio : consume_fraction = 60 / 100) :
  (initial_mangoes * ripe_fraction - initial_mangoes * ripe_fraction * consume_fraction) = 96 :=
by
  sorry

end ripe_mangoes_remaining_l460_460821


namespace reciprocal_of_neg_two_l460_460786

theorem reciprocal_of_neg_two :
  (∃ x : ℝ, x = -2 ∧ 1 / x = -1 / 2) :=
by
  use -2
  split
  · rfl
  · norm_num

end reciprocal_of_neg_two_l460_460786


namespace isosceles_triangle_leg_length_l460_460402

theorem isosceles_triangle_leg_length (P : ℝ) (a b c : ℝ) (h1 : P = 70) (h2 : a/b = 1/3) (h3 : a = b) :
  a = 14 ∨ a = 30 := by
s

end isosceles_triangle_leg_length_l460_460402


namespace even_periodic_function_with_pi_divby_2_period_l460_460136

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem even_periodic_function_with_pi_divby_2_period :
  (is_even_function (λ x, Real.sin (Real.pi / 2 - 4 * x)) ∧ period (λ x, Real.sin (Real.pi / 2 - 4 * x)) (Real.pi / 2)) :=
by
  sorry

end even_periodic_function_with_pi_divby_2_period_l460_460136


namespace cube_root_of_product_is_280_l460_460359

theorem cube_root_of_product_is_280 : (∛(2^9 * 5^3 * 7^3) = 280) := 
by 
sorry

end cube_root_of_product_is_280_l460_460359


namespace Vasya_arrives_first_l460_460798

theorem Vasya_arrives_first :
  ∀ (distance_to_school : ℝ) (n : ℝ), 
    (distance_to_school > 0) →
    (n > 0) →
    (∀ (d_vasya d_petya : ℝ),
      d_vasya = n →
      d_petya = 1.25 * n * 0.75 →
      d_petya < d_vasya) →
    ∃ (t_vasya t_petya : ℝ), 
      t_vasya < t_petya :=
by
  sorry

end Vasya_arrives_first_l460_460798


namespace exists_root_f_between_0_and_1_l460_460521

noncomputable def f (x : ℝ) : ℝ := 4 - 4 * x - Real.exp x

theorem exists_root_f_between_0_and_1 :
  ∃ x ∈ Set.Ioo 0 1, f x = 0 :=
sorry

end exists_root_f_between_0_and_1_l460_460521


namespace complement_setP_in_U_l460_460610

def setU : Set ℝ := {x | -1 < x ∧ x < 3}
def setP : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem complement_setP_in_U : (setU \ setP) = {x | 2 < x ∧ x < 3} :=
by
  sorry

end complement_setP_in_U_l460_460610


namespace even_factors_count_l460_460255

theorem even_factors_count (n : ℕ) (hn : n = 2^2 * 3^1 * 7^2 * 5^1) : 
  (∃ k, k = 24 ∧ k = (2 * 2 * 3 * 2)) :=
by
  use 24
  split
  . refl
  . sorry

end even_factors_count_l460_460255


namespace intersection_point_l460_460523

variable (x y : ℚ)

theorem intersection_point :
  (8 * x - 5 * y = 40) ∧ (10 * x + 2 * y = 14) → 
  (x = 25 / 11) ∧ (y = 48 / 11) :=
by
  sorry

end intersection_point_l460_460523


namespace psychologist_charge_difference_l460_460828

variables (F A : ℝ)

theorem psychologist_charge_difference
  (h1 : F + 4 * A = 375)
  (h2 : F + A = 174) :
  (F - A) = 40 :=
by sorry

end psychologist_charge_difference_l460_460828


namespace sum_arithmetic_sequence_l460_460226

variables {a_n : ℕ → ℝ}
variables {S_n : ℕ → ℝ}
variables {n : ℕ}

-- Conditions
axiom h1 : a_n 5 = 5
axiom h2 : S_n 8 = 36
axiom h3 : ∀ n, S_n (n + 1) = S_n n + a_n (n + 1)

-- The goal to prove
theorem sum_arithmetic_sequence :
  ∑ k in finset.range n, (1 / (a_n k * a_n (k + 1))) = (n - 1) / (n + 1) :=
sorry

end sum_arithmetic_sequence_l460_460226


namespace possible_clock_angles_l460_460143

theorem possible_clock_angles (α : ℝ) : 
  (∀ (t : ℝ), (t = 60) → 
    let angle_hour_hand := (60 * (t / 60) / 12 : ℝ) in 
    let angle_minute_hand := (60 * (t / 60) : ℝ) in 
    (abs(α - angle_hour_hand) = α ∨ abs(α - angle_minute_hand) = α)) →
   (α ∈ {15, 165}) :=
by
  sorry

end possible_clock_angles_l460_460143


namespace two_om_2om5_l460_460400

def om (a b : ℕ) : ℕ := a^b - b^a

theorem two_om_2om5 : om 2 (om 2 5) = 79 := by
  sorry

end two_om_2om5_l460_460400


namespace solve_for_g2_l460_460389

-- Let g : ℝ → ℝ be a function satisfying the given condition
variable (g : ℝ → ℝ)

-- The given condition
def condition (x : ℝ) : Prop :=
  g (2 ^ x) + x * g (2 ^ (-x)) = 2

-- The main theorem we aim to prove
theorem solve_for_g2 (h : ∀ x, condition g x) : g 2 = 0 :=
by
  sorry

end solve_for_g2_l460_460389


namespace card_trick_l460_460261

/-- A magician is able to determine the fifth card from a 52-card deck using a prearranged 
    communication system between the magician and the assistant, thus no supernatural 
    abilities are required. -/
theorem card_trick (deck : Finset ℕ) (h_deck : deck.card = 52) (chosen_cards : Finset ℕ)
  (h_chosen : chosen_cards.card = 5) (shown_cards : Finset ℕ) (h_shown : shown_cards.card = 4)
  (fifth_card : ℕ) (h_fifth_card : fifth_card ∈ chosen_cards \ shown_cards) :
  ∃ (prearranged_system : (Finset ℕ) → (Finset ℕ) → ℕ),
    ∀ (remaining : Finset ℕ), remaining.card = 1 → 
    prearranged_system shown_cards remaining = fifth_card := 
sorry

end card_trick_l460_460261


namespace angle_between_a_b_is_pi_over_2_l460_460251

noncomputable def a : ℝ × ℝ × ℝ := (0, 2, 1)
noncomputable def b : ℝ × ℝ × ℝ := (-1, 1, -2)

theorem angle_between_a_b_is_pi_over_2
    (a := (0 : ℝ, 2, 1))
    (b := (-1 : ℝ, 1, -2)) : 
    real.angle a b = real.pi / 2 := 
sorry

end angle_between_a_b_is_pi_over_2_l460_460251


namespace trig_identity_simplify_l460_460009

theorem trig_identity_simplify :
  sin (50 * Real.pi / 180) * (1 + (Real.sqrt 3) * tan (10 * Real.pi / 180)) = 1 := 
sorry

end trig_identity_simplify_l460_460009


namespace equal_circles_parallel_segments_l460_460896

theorem equal_circles_parallel_segments
  (S S1 S2 : Type) [IsCircle S] [IsCircle S1] [IsCircle S2]
  (A1 A2 C B1 B2 : Point)
  (internal_touch : InternalTouch S S1 A1 ∧ InternalTouch S S2 A2)
  (circumference_point : OnCircumference S C)
  (intersect1 : IntersectAtSegment S1 (Segment C A1) (Segment B1 B2))
  (intersect2 : IntersectAtSegment S2 (Segment C A2) (Segment B1 B2))
  (equal_circles : EqualCircles S1 S2) :
  Parallel (Segment A1 A2) (Segment B1 B2) :=
sorry

end equal_circles_parallel_segments_l460_460896


namespace transfer_balls_l460_460150

theorem transfer_balls (X Y q p b : ℕ) (h : p + b = q) :
  b = q - p :=
by
  sorry

end transfer_balls_l460_460150


namespace roots_in_interval_l460_460038

theorem roots_in_interval (a b : ℝ) (hb : b > 0) (h_discriminant : a^2 - 4 * b > 0)
  (h_root_interval : ∃ r1 r2 : ℝ, r1 + r2 = -a ∧ r1 * r2 = b ∧ ((-1 ≤ r1 ∧ r1 ≤ 1 ∧ (r2 < -1 ∨ 1 < r2)) ∨ (-1 ≤ r2 ∧ r2 ≤ 1 ∧ (r1 < -1 ∨ 1 < r1)))) : 
  ∃ r : ℝ, (r + a) * r + b = 0 ∧ -b < r ∧ r < b :=
by
  sorry

end roots_in_interval_l460_460038


namespace proof_a_range_l460_460246

noncomputable def a_range : Prop :=
  ∀ (x y z : ℝ), (x + y + z = 1) → (|a - 2| ≤ x^2 + 2 * y^2 + 3 * z^2) → 
  (16/11 ≤ a ∧ a ≤ 28/11)

theorem proof_a_range : a_range :=
by
  sorry

end proof_a_range_l460_460246


namespace rhombus_area_l460_460903

theorem rhombus_area (R1 R2 : ℝ) (hR1 : R1 = 15) (hR2 : R2 = 30) :
  let x := (PR : ℝ) / 2
  let y := (QS : ℝ) / 2 
  (sqrt (x^2 + y^2)) ^ 4 / 4 = 225 :=
by {
  sorry
}

end rhombus_area_l460_460903


namespace polynomial_square_b_eq_25_over_64_l460_460035

noncomputable def b : ℚ :=
  let p : ℚ := 3 / 2 in
  let q : ℚ := -5 / 8 in
  q ^ 2

theorem polynomial_square_b_eq_25_over_64 (a b : ℚ)
  (h : ∃ Q : Polynomial ℚ, Polynomial ^ 2 = x^4 + 3 * x^3 + x^2 + a * x + b) :
  b = 25 / 64 :=
sorry

end polynomial_square_b_eq_25_over_64_l460_460035


namespace tree_height_at_2_years_l460_460855

theorem tree_height_at_2_years (h₅ : ℕ) (h_four : ℕ) (h_three : ℕ) (h_two : ℕ) (h₅_value : h₅ = 243)
  (h_four_value : h_four = h₅ / 3) (h_three_value : h_three = h_four / 3) (h_two_value : h_two = h_three / 3) :
  h_two = 9 := by
  sorry

end tree_height_at_2_years_l460_460855


namespace integral_straight_line_integral_parabola_integral_broken_line_l460_460872

noncomputable def integrand (z : ℂ) : ℂ := 1 + complex.I - 2 * complex.conj z

noncomputable def integral_along_straight_line : ℂ :=
  ∫ z in segment (0 : ℂ) (1 + complex.I), integrand z

noncomputable def path_parabola (t : ℝ) : ℂ := t + complex.I * t^2 

noncomputable def param_integral_along_parabola : ℂ :=
  ∫ t in (0 : ℝ)..(1 : ℝ), integrand (path_parabola t) * complex.I * (2 * t)

noncomputable def integral_along_parabola : ℂ :=
  ∫ z in path_parabola, integrand z

noncomputable def path_broken_line_1 (t : ℝ) : ℂ := t

noncomputable def path_broken_line_2 (t : ℝ) : ℂ := 1 + complex.I * t 

noncomputable def integral_along_broken_line : ℂ := 
  ∫ t in (0 : ℝ)..(1 : ℝ), integrand (path_broken_line_1 t) +
  ∫ t in (0 : ℝ)..(1 : ℝ), integrand (path_broken_line_2 t)
  
theorem integral_straight_line : integral_along_straight_line = 2 * complex.I - 2 := by
  sorry

theorem integral_parabola : integral_along_parabola = -2 + 4/3 * complex.I := by
  sorry

theorem integral_broken_line : integral_along_broken_line = -2 := by
  sorry

end integral_straight_line_integral_parabola_integral_broken_line_l460_460872


namespace cannot_transform_triplet_l460_460207

theorem cannot_transform_triplet :
  let transform (x y : ℝ) : ℝ × ℝ := ( (x - y) / Real.sqrt 2, (x + y) / Real.sqrt 2)
  ∀ (a b c : ℝ), (a, b, c) = (2, Real.sqrt 2, 1 / Real.sqrt 2) →
  ¬∃ (a' b' c' : ℝ), (a', b', c') = (1, Real.sqrt 2, 1 + Real.sqrt 2) ∧
    (∃ (steps : list (ℝ × ℝ → ℝ × ℝ)) (init : ℝ × ℝ × ℝ),
      init = (a, b, c) ∧ list.foldl (λ t f, (f t.1 t.2)), (a, b)) steps = (a', b', c')
:= by
  -- Definitions based on given conditions
  let initial_triplet : ℝ × ℝ × ℝ := (2, Real.sqrt 2, 1 / Real.sqrt 2)
  let final_triplet : ℝ × ℝ × ℝ := (1, Real.sqrt 2, 1 + Real.sqrt 2)

  -- Proof outline using sum of squares invariant
  have initial_sum_squares : initial_triplet.1^2 + initial_triplet.2^2 + initial_triplet.3^2 = 6.5 := sorry
  have final_sum_squares : final_triplet.1^2 + final_triplet.2^2 + final_triplet.3^2 = 5 + 2 * Real.sqrt 2 := sorry

  -- Comparing sums of squares
  have sum_squares_different : initial_sum_squares ≠ final_sum_squares := sorry
  contradiction

end cannot_transform_triplet_l460_460207


namespace problem_equivalent_l460_460716

open Real

noncomputable def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

noncomputable def inradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let A := sqrt (s * (s - a) * (s - b) * (s - c))
  A / s

def distance_between_centers (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem problem_equivalent :
  ∀ (XY XZ YZ : ℝ),
  XY = 80 → XZ = 150 → YZ = 170 →
  is_right_triangle XY XZ YZ →
  let r1 := inradius XY XZ YZ
  let r2 := (r1 * 120) / 150
  let r3 := (r1 * 50) / 80
  distance_between_centers (24, 144) (68.75, 18.75) = sqrt (10 * 1769.4125) :=
by
  intros XY XZ YZ XY_eq XZ_eq YZ_eq h_right_triangle;
  let r1 := inradius XY XZ YZ
  let r2 := (r1 * 120) / 150
  let r3 := (r1 * 50) / 80
  exact sorry

end problem_equivalent_l460_460716


namespace number_of_pairs_l460_460711

open Finset

theorem number_of_pairs (P : Finset ℕ) (hP : P = ({1, 2, 3, 4, 5, 6} : Finset ℕ)) :
  ∃ (A B : Finset ℕ), A ⊆ P ∧ B ⊆ P ∧ A.nonempty ∧ B.nonempty ∧ (A.max' (by simp [hP, P.nonempty])) < (B.min' (by simp [hP, P.nonempty])) ∧
  P.pair_count (λ A B, (A.max' (by simp [hP, A.nonempty])) < (B.min' (by simp [hP, B.nonempty]))) = 129 :=
by
  sorry

end number_of_pairs_l460_460711


namespace expected_people_with_condition_l460_460749

noncomputable def proportion_of_condition := 1 / 3
def total_population := 450

theorem expected_people_with_condition :
  (proportion_of_condition * total_population) = 150 := by
  sorry

end expected_people_with_condition_l460_460749


namespace find_range_a_l460_460231

noncomputable def f (a x : ℝ) : ℝ := x^2 + (a^2 - 1) * x + (a - 2)

theorem find_range_a (a : ℝ) (h : ∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ x > 1 ∧ y < 1 ) :
  -2 < a ∧ a < 1 := sorry

end find_range_a_l460_460231


namespace jane_uses_40_ribbons_l460_460309

theorem jane_uses_40_ribbons :
  (∀ dresses1 dresses2 ribbons_per_dress, 
  dresses1 = 2 * 7 ∧ 
  dresses2 = 3 * 2 → 
  ribbons_per_dress = 2 → 
  (dresses1 + dresses2) * ribbons_per_dress = 40)
:= 
by 
  sorry

end jane_uses_40_ribbons_l460_460309


namespace coin_probability_l460_460443

def fair_coin_prob_tail_2_or_3_in_3_flips : ℝ :=
let p := 0.5 in
let n := 3 in
let P := (λ k, (nat.choose n k) * (p^k) * ((1 - p)^(n - k))) in
P 2 + P 3

theorem coin_probability : fair_coin_prob_tail_2_or_3_in_3_flips = 0.5 := by
  sorry

end coin_probability_l460_460443


namespace large_pile_toys_l460_460283

theorem large_pile_toys (x y : ℕ) (h1 : x + y = 120) (h2 : y = 2 * x) : y = 80 := by
  sorry

end large_pile_toys_l460_460283


namespace zero_point_in_interval_l460_460567

noncomputable def f (g : ℝ → ℝ) : ℝ → ℝ :=
  λ x, (x^2 - 3*x + 2) * g x + 3 * x - 4

theorem zero_point_in_interval (g : ℝ → ℝ) (h_continuous : Continuous g) :
  f g 1 = -1 ∧ f g 2 = 2 → ∃ x ∈ Ioo 1 2, f g x = 0 :=
by
  intro h
  sorry

end zero_point_in_interval_l460_460567


namespace ellipse_eqn_area_of_triangle_PQ_equation_of_line_l460_460594

-- Global assumptions for the proofs
variables (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_ab : a > b)

def ellipse := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1
def major_axis_length := 2 * sqrt 2
def eccentricity := sqrt 2 / 2

theorem ellipse_eqn (h1 : major_axis_length = 2 * a)
  (h2 : 1 - b^2 / a^2 = eccentricity^2) :
  (a = sqrt 2) ∧ (b = 1) :=
by sorry

theorem area_of_triangle_PQ (h1 : a = sqrt 2) (h2 : b = 1) (slope_l : ℝ) (h_slope : slope_l = 1)
  (focus_x : ℝ) (focus_y : ℝ) (h_focus : (focus_x, focus_y) = (1, 0))
  (P Q : ℝ × ℝ) (h_line : ∀ x y : ℝ, y = slope_l * (x - focus_x)) :
  let y1 := -1, y2 := 1 / 3 in (1 / 2) * abs (y1 - y2) = 2 / 3 :=
by sorry

theorem equation_of_line (h1 : a = sqrt 2) (h2 : b = 1)
  (k : ℝ) (h_k_square : k^2 = 2)
  (l_eq : ∀ x y : ℝ, y = k * (x - 1)) :
  k = sqrt 2 ∨ k = - sqrt 2 :=
by sorry

end ellipse_eqn_area_of_triangle_PQ_equation_of_line_l460_460594


namespace three_non_coplanar_lines_determine_three_planes_l460_460571

-- Defining the components based on the conditions in (a)
-- The definition of a non-coplanar set of lines is inherent in how they are described.
-- Three lines are sufficient, hence using lists or sets to keep track of these lines.

variables {Point : Type} [Nonempty Point]
variables (Line : Type) [Nonempty Line]

-- Definitions of collinearity and planes
class NonCoplanar (lines : list Line) : Prop :=
  (from_distinct_sets : ∀ l1 l2 l3 : Line, 
    l1 ∈ lines → l2 ∈ lines → l3 ∈ lines → 
    (l1 ≠ l2 ∧ l2 ≠ l3 ∧ l3 ≠ l1) ∧
    ¬(∃ p1 p2 p3 : Point, -- Points to define non-coplanarity
      p1 ≠ p2 ∧ p2 ≠ p3 ∧ 
      (∃ l1, p1 ∈ l1 ∧ p2 ∈ l1) ∧
      (∃ l2, p2 ∈ l2 ∧ p3 ∈ l2) ∧
      (∃ l3, p3 ∈ l3 ∧ p1 ∈ l3)))

constant planes_determined_by_lines : list Line → ℕ

-- Theorem statement
theorem three_non_coplanar_lines_determine_three_planes 
    (p : Point) 
    (l1 l2 l3 : Line) 
    (lines := [l1, l2, l3])
    [NonCoplanar lines]
    (through_point : ∀ l ∈ lines, p ∈ l) : 
    planes_determined_by_lines [l1, l2, l3] = 3 :=
sorry

end three_non_coplanar_lines_determine_three_planes_l460_460571


namespace magicSquareProof_l460_460671

noncomputable def magicSquareCondition (a b c d e : ℕ) : Prop :=
  (16 + d + 13 = 11 + b + e) ∧
  (16 + d + 13 = a + 20 + c) ∧
  (16 + b + c = 11 + b + e) ∧
  ((16 + b + c = (a + 20 + c))

theorem magicSquareProof (a b c d e : ℕ) (h : magicSquareCondition a b c d e) : 
  d + e = 7 :=
  sorry

end magicSquareProof_l460_460671


namespace maximize_profit_l460_460827

def total_orders := 100
def max_days := 160
def time_per_A := 5 / 4 -- days
def time_per_B := 5 / 3 -- days
def profit_per_A := 0.5 -- (10,000 RMB)
def profit_per_B := 0.8 -- (10,000 RMB)

theorem maximize_profit : 
  ∃ (x : ℝ) (y : ℝ), 
    (time_per_A * x + time_per_B * (total_orders - x) ≤ max_days) ∧ 
    (y = -0.3 * x + 80) ∧ 
    (x = 16) ∧ 
    (y = 75.2) :=
by 
  sorry

end maximize_profit_l460_460827


namespace no_such_alpha_exists_l460_460890

theorem no_such_alpha_exists :
  ¬ ∃ (α : ℝ) (a : ℕ → ℝ), (0 < α ∧ α < 1) ∧ (∀ n : ℕ, n > 0 → (1 + a(n+1) ≤ a(n) + (α / n) * a(n))) ∧ (∀ n, 0 < a(n)) :=
by sorry

end no_such_alpha_exists_l460_460890


namespace remainder_17_pow_63_mod_7_l460_460061

theorem remainder_17_pow_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l460_460061


namespace exists_z0_abs_f_ge_abs_C0_abs_Cn_l460_460198

theorem exists_z0_abs_f_ge_abs_C0_abs_Cn :
  ∀ (f : ℂ → ℂ) (n : ℕ) (C : Fin (n + 1) → ℂ),
  (f = (λ z, ∑ i : Fin (n + 1), C i * z^i)) →
  ∃ (z0 : ℂ), |z0| ≤ 1 ∧ |f z0| ≥ |C 0| + |C n| :=
by
  sorry

end exists_z0_abs_f_ge_abs_C0_abs_Cn_l460_460198


namespace game_last_at_most_moves_l460_460047

theorem game_last_at_most_moves
  (n : Nat)
  (positions : Fin n → Fin (n + 1))
  (cards : Fin n → Fin (n + 1))
  (move : (k l : Fin n) → (h1 : k < l) → (h2 : k < cards k) → (positions l = cards k) → Fin n)
  : True :=
sorry

end game_last_at_most_moves_l460_460047


namespace diamonds_in_F6_l460_460879

def F (n : Nat) : Nat :=
  if n = 1 then 1
  else if n = 2 then 9
  else if n = 3 then 16 -- Explicit value from condition
  else F (n - 1) + 4 * 2^(n - 1)

theorem diamonds_in_F6 : F 6 = 249 :=
by
  -- The proof is omitted for clarity
  -- Only the statement is needed 
  sorry

end diamonds_in_F6_l460_460879


namespace cartesian_equation_curve_general_equation_line_max_area_quadrilateral_l460_460942

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

theorem cartesian_equation_curve (θ : ℝ) :
    let (x, y) := polar_to_cartesian (6 * cos θ) θ
    in x^2 + y^2 = 6 * x := sorry

theorem general_equation_line (t θ : ℝ) :
    ∀ x y, (x = 4 + t * cos θ ∧ y = -1 + t * sin θ) ↔ y + 1 = tan θ * (x - 4) := sorry

theorem max_area_quadrilateral :
    let A := (x, y) where (x = 4 + t * cos θ ∧ y = -1 + t * sin θ) := sorry
    let C := (x, y) where (x = 4 + t' * cos θ ∧ y = -1 + t' * sin θ) := sorry
    let B : (ℝ × ℝ) := sorry
    let D : (ℝ × ℝ) := sorry
    let area := (|A y - B y| + |C y - D y|) * |B x - D x| / 2
    in area ≤ 16 := sorry

end cartesian_equation_curve_general_equation_line_max_area_quadrilateral_l460_460942


namespace a4_range_l460_460339

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

def sequence_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + d * (n * (n - 1)) / 2

theorem a4_range (a₁ d : ℝ) (hmono : 0 < d ∧ d ≤ 1) (hS4 : sequence_sum a₁ d 4 ≥ 10) (hS5 : sequence_sum a₁ d 5 ≤ 15) : 
  5 / 2 < arithmetic_sequence a₁ d 4 ∧ arithmetic_sequence a₁ d 4 ≤ 4 :=
by
  -- skipping proof steps here
  sorry

end a4_range_l460_460339


namespace partition_exists_l460_460353
open Function

noncomputable def Participant := ℕ
noncomputable def Country := ℕ
noncomputable def countries : Finset Country := Finset.range 100
noncomputable def participants (c : Country) : Finset Participant := Finset.range 2

variable (knows : Participant → Participant → Prop)
variable (from_country : Participant → Country)

noncomputable def valid_partition (A B : Finset Participant) : Prop :=
  (∀ p ∈ A, ∀ q ∈ A, from_country p ≠ from_country q ∧ ¬ knows p q) ∧ 
  (∀ p ∈ B, ∀ q ∈ B, from_country p ≠ from_country q ∧ ¬ knows p q)

theorem partition_exists : 
  ∃ A B : Finset Participant, A ∪ B = (Finset.univ : Finset Participant) ∧ A ∩ B = ∅ ∧ valid_partition knows from_country A B :=
sorry

end partition_exists_l460_460353


namespace cos_identity_l460_460944

theorem cos_identity (α : ℝ) (h : Real.cos (π / 6 + α) = sqrt 3 / 3) : 
  Real.cos (5 * π / 6 - α) = - (sqrt 3 / 3) :=
by
  sorry

end cos_identity_l460_460944


namespace sin_inequality_solution_l460_460177

theorem sin_inequality_solution (x : ℝ) (hx1 : 0 < x) (hx2 : x < π) :
  (8 / (3 * Real.sin x - Real.sin (3 * x)) + 3 * (Real.sin x)^2 ≤ 5) ↔ (x = π / 2) :=
begin
  sorry
end

end sin_inequality_solution_l460_460177


namespace average_gas_mileage_round_trip_l460_460831

def distance_to_friend := 150 -- miles
def distance_return := 150 -- miles
def motorcycle_mileage := 50 -- miles per gallon
def scooter_mileage := 30 -- miles per gallon

theorem average_gas_mileage_round_trip : 
  (distance_to_friend + distance_return) / 
  ((distance_to_friend / motorcycle_mileage) + (distance_return / scooter_mileage)) = 37.5 :=
by
  sorry

end average_gas_mileage_round_trip_l460_460831


namespace selling_price_per_chicken_l460_460686

noncomputable theory

-- Definitions from the problem conditions
def w_feed_bag : ℝ := 20 -- weight per bag of feed in pounds
def c_feed_bag : ℝ := 2  -- cost per bag of feed in dollars
def f_chicken : ℝ := 2  -- feed required per chicken in pounds
def N_chickens : ℕ := 50 -- number of chickens
def P_profit : ℝ := 65  -- profit from selling chickens in dollars

-- The statement to prove
theorem selling_price_per_chicken : 
  ∃ p_chicken : ℝ, 
  p_chicken * N_chickens = P_profit + 
  ((f_chicken * N_chickens / w_feed_bag) * c_feed_bag) :=
sorry

end selling_price_per_chicken_l460_460686


namespace bullet_speed_difference_l460_460431

def speed_horse : ℕ := 20  -- feet per second
def speed_bullet : ℕ := 400  -- feet per second

def speed_forward : ℕ := speed_bullet + speed_horse
def speed_backward : ℕ := speed_bullet - speed_horse

theorem bullet_speed_difference : speed_forward - speed_backward = 40 :=
by
  sorry

end bullet_speed_difference_l460_460431


namespace solution_set_for_a1_find_a_if_min_value_is_4_l460_460599

noncomputable def f (a x : ℝ) : ℝ := |2 * x - 1| + |a * x - 5|

theorem solution_set_for_a1 : 
  { x : ℝ | f 1 x ≥ 9 } = { x : ℝ | x ≤ -1 ∨ x > 5 } :=
sorry

theorem find_a_if_min_value_is_4 :
  ∃ a : ℝ, (0 < a ∧ a < 5) ∧ (∀ x : ℝ, f a x ≥ 4) ∧ (∃ x : ℝ, f a x = 4) ∧ a = 2 :=
sorry

end solution_set_for_a1_find_a_if_min_value_is_4_l460_460599


namespace eccentricity_eq_ratio_n_m_l460_460938

-- Definitions based on the conditions given in (a)
def is_ellipse (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def is_vertex (x y a b : ℝ) : Prop :=
  x = 0 ∧ (y = b ∨ y = -b)

def is_focus (x y c : ℝ) : Prop :=
  y = 0 ∧ (x = -c ∨ x = c)

def is_parallel (p1 p2 q1 q2 : ℝ) : Prop :=
  (p2 - p1) * (q2 - q1) = 0

def is_on_circumcircle (x y a b c : ℝ) : Prop :=
  let h := (x - c/2)^2 + y^2
  (h = (c/2 + c)^2)

def conditions (a b c : ℝ) : Prop :=
  ∃ x y, 
  is_ellipse a b x y ∧ 
  is_vertex x y a b ∧ 
  is_focus x y c ∧ 
  (∃ ex ey bx by, 
    (ex = 3 * c ∧ ey = 0) ∧ 
    is_parallel (-c) b (c) by ∧ 
    (bx = ex / 2 ∧ by = ey / 2) ∧ 
      (bx^2 / a^2 + by^2 / b^2 = 1))

-- Lean Theorem Statements based on correct answers in (b)
theorem eccentricity_eq (a b c : ℝ) (h : conditions a b c) :
  (c / a) = (√3 / 3) :=
sorry

theorem ratio_n_m (m n a b c : ℝ) (h : conditions a b c) :
  ∀ {hx hy}, is_on_circumcircle hx hy a b c → (n / m) = (2 * √2 / 5) :=
sorry

end eccentricity_eq_ratio_n_m_l460_460938


namespace distance_between_points_is_sqrt_42_l460_460542

-- Define the points in four-dimensional space
structure Point4D := (x y z w : ℝ)

def point1 : Point4D := ⟨3, 2, 5, -1⟩
def point2 : Point4D := ⟨7, 6, 2, 0⟩

-- Define the distance function in four-dimensional space
def distance_4d (p1 p2 : Point4D) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2 + (p2.w - p1.w)^2)

-- Theorem stating the distance between point1 and point2
theorem distance_between_points_is_sqrt_42 :
  distance_4d point1 point2 = real.sqrt 42 :=
by
  sorry

end distance_between_points_is_sqrt_42_l460_460542


namespace lattice_points_count_l460_460468

-- Defining the conditions
def is_lattice_point (p : ℝ × ℝ) : Prop :=
  p.1 ∈ Int ∧ p.2 ∈ Int

def y_eq_abs_x (p : ℝ × ℝ) : Prop :=
  p.2 = abs p.1

def y_eq_neg_x_sq_plus_8 (p : ℝ × ℝ) : Prop :=
  p.2 = -p.1^2 + 8

-- Problem statement in Lean 4
theorem lattice_points_count :
  ∃ (S : Finset (ℝ × ℝ)), (∀ p ∈ S, is_lattice_point p) ∧
    (∀ p ∈ S, y_eq_abs_x p ∨ y_eq_neg_x_sq_plus_8 p) ∧
    S.card = 29 :=
by
  sorry

end lattice_points_count_l460_460468


namespace find_number_in_parentheses_l460_460188

theorem find_number_in_parentheses :
  ∃ x : ℝ, 3 + 2 * (x - 3) = 24.16 ∧ x = 13.58 :=
by
  sorry

end find_number_in_parentheses_l460_460188


namespace tetrahedron_labeling_count_l460_460516

theorem tetrahedron_labeling_count : 
  ∃ (labelings : Finset (Vector (Fin 6) Bool)),
    (∀ labeling ∈ labelings, 
      ∀ face ∈ faces, 
        (sum (face.map (λ edge, if labeling[edge] then 1 else 0)) = 2)) ∧ 
    labelings.card = 2 := 
sorry

end tetrahedron_labeling_count_l460_460516


namespace gcd_poly_eq_one_l460_460219

noncomputable def b : ℤ := 7769 * (2 * k + 1) -- where k is some integer, making b an odd multiple of 7769
def poly1 (b : ℤ) := 4 * b^2 + 81 * b + 144
def poly2 (b : ℤ) := 2 * b + 7

theorem gcd_poly_eq_one (b : ℤ) (k : ℤ) (h : b = 7769 * (2 * k + 1)) : 
    Int.gcd (poly1 b) (poly2 b) = 1 := 
sorry

end gcd_poly_eq_one_l460_460219


namespace length_of_each_brick_l460_460459

theorem length_of_each_brick
  (x : ℕ) 
  (wall_length : ℕ := 900) 
  (wall_height : ℕ := 600) 
  (wall_thickness : ℕ := 22.5) 
  (brick_width : ℕ := 11.25) 
  (brick_height : ℕ := 6) 
  (number_of_bricks : ℕ := 7200)
  (volume_wall : ℕ := wall_length * wall_height * wall_thickness)
  (volume_brick := x * brick_width * brick_height)
  (H : volume_wall = volume_brick * number_of_bricks) :
  x = 25 :=
by
  sorry

end length_of_each_brick_l460_460459


namespace B_investment_time_l460_460493

theorem B_investment_time (annual_gain A_share : ℝ) (x : ℝ) (m : ℝ) (C_period : ℝ) :
  annual_gain = 21000 ∧ A_share = 7000 ∧ C_period = 4 →
  (12 * x) / (12 * x + 2 * x * (12 - m) + 3 * x * C_period) = 1 / 3 →
  m = 6 :=
begin
  sorry,
end

end B_investment_time_l460_460493


namespace range_of_a_l460_460644

noncomputable def f (a x : ℝ) := a * exp (2 * x) + (a - 2) * exp x - x

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ (a ∈ set.Ioo 0 1) :=
by
  sorry

end range_of_a_l460_460644


namespace triangle_area_202_2192_pi_squared_l460_460859

noncomputable def triangle_area (a b c : ℝ) : ℝ := 
  let r := (a + b + c) / (2 * Real.pi)
  let theta := 20.0 * Real.pi / 180.0  -- converting 20 degrees to radians
  let angle1 := 5 * theta
  let angle2 := 6 * theta
  let angle3 := 7 * theta
  (1 / 2) * r * r * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem triangle_area_202_2192_pi_squared (a b c : ℝ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 7) : 
  triangle_area a b c = 202.2192 / (Real.pi * Real.pi) := 
by {
  sorry
}

end triangle_area_202_2192_pi_squared_l460_460859


namespace geometric_sequence_formula_l460_460202

noncomputable def a_n (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q^n

theorem geometric_sequence_formula
  (a_1 q : ℝ)
  (h_pos : ∀ n : ℕ, a_n a_1 q n > 0)
  (h_4_eq : a_n a_1 q 4 = (a_n a_1 q 2)^2)
  (h_2_4_sum : a_n a_1 q 2 + a_n a_1 q 4 = 5 / 16) :
  ∀ n : ℕ, a_n a_1 q n = ((1 : ℝ) / 2) ^ n :=
sorry

end geometric_sequence_formula_l460_460202


namespace recess_breaks_l460_460801

theorem recess_breaks (total_outside_time : ℕ) (lunch_break : ℕ) (extra_recess : ℕ) (recess_duration : ℕ) 
  (h1 : total_outside_time = 80)
  (h2 : lunch_break = 30)
  (h3 : extra_recess = 20)
  (h4 : recess_duration = 15) : 
  (total_outside_time - (lunch_break + extra_recess)) / recess_duration = 2 := 
by {
  -- proof starts here
  sorry
}

end recess_breaks_l460_460801


namespace minimum_attempts_to_open_safe_l460_460049

-- Define types and conditions
def isValidSequence (code : List Bool) (n : Nat) : Prop := code.length = n ∧ ∀ d ∈ code, d = true ∨ d = false

-- The Lean 4 equivalent proof statement
theorem minimum_attempts_to_open_safe (n : Nat) (secret_code : List Bool)
  (h1 : isValidSequence secret_code n)
  (h2 : secret_code ≠ List.replicate n false) :
  ∃ (attempts : List (List Bool)), attempts.length = n ∧ ∀ attempt ∈ attempts, isValidSequence attempt n ∧ (attempt = secret_code ∨ (∃ idx, h1.fst idx = secret_code idx)) :=
sorry

end minimum_attempts_to_open_safe_l460_460049


namespace order_of_numbers_l460_460802

theorem order_of_numbers :
  2^30 < 10^10 ∧ 10^10 < 5^15 :=
by
  sorry

end order_of_numbers_l460_460802


namespace increasing_condition_max_min_values_l460_460968

noncomputable def f (a x : ℝ) : ℝ := (1 - x) / (a * x) + real.log x

theorem increasing_condition (a : ℝ) (ha : 1 ≤ a) :
  ∀ x ∈ set.Ici (1 : ℝ), 0 ≤ (a * x - 1) / (a * x^2) :=
begin
  sorry
end

theorem max_min_values (a : ℝ) (ha : a = 1) :
  ∃ (x y : ℝ), x ∈ set.Icc (1 / real.exp 1) (real.exp 1) ∧ y ∈ set.Icc (1 / real.exp 1) (real.exp 1) ∧
  f 1 x = real.exp 1 - 2 ∧ f 1 y = 0 :=
begin
  sorry
end

end increasing_condition_max_min_values_l460_460968


namespace count_correct_propositions_l460_460965

-- Definitions for the conditions
def prop_one : Prop :=
  ∀ x : ℝ, 0 < x → (has_deriv_at (λ y, y ^ (1/2)) _ x → 0 < (deriv (λ y, y ^ (1/2)) x)) ∧
            (has_deriv_at (λ y, y ^ 3) _ x → 0 < (deriv (λ y, y ^ 3) x))

def prop_two (m n : ℝ) : Prop :=
  log_base m 3 < log_base n 3 ∧ log_base n 3 < 0 → 0 < n ∧ n < m ∧ m < 1

def prop_three (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f(x)) →
  (graph_shift (λ y, f (y - 1)).symmetry_point = (1,0))

def prop_four (f : ℝ → ℝ) : Prop :=
  (f = (λ x, 3^x - 2*x - 3)) →
  (∃ x1 x2: ℝ, f(x1) = 0 ∧ f(x2) = 0 ∧ x1 ≠ x2)

-- Main theorem statement
theorem count_correct_propositions :
  prop_one ∧ prop_two 0.5 0.2 ∧ prop_three (λ x, x^3) ∧ prop_four (λ x, 3^x - 2*x - 3) → 
  (3 = 3) :=
by sorry

end count_correct_propositions_l460_460965


namespace relationship_among_abc_l460_460196

noncomputable def a : ℝ := (0.3) ^ 0.4
noncomputable def b : ℝ := (0.6) ^ 0.4
noncomputable def c : ℝ := Real.logBase 0.3 2

theorem relationship_among_abc : b > a ∧ a > c := 
by
  sorry

end relationship_among_abc_l460_460196


namespace shaded_region_area_l460_460481

theorem shaded_region_area :
  let s := 8 in
  let r := 4 in
  let hex_area := 6 * (sqrt 3 / 4 * s^2) in
  let sector_area := 6 * (1 / 6 * π * r^2) in
  hex_area - sector_area = 96 * sqrt 3 - 16 * π :=
by
  let s := 8
  let r := 4
  let hex_area := 6 * (sqrt 3 / 4 * s^2)
  let sector_area := 6 * (1 / 6 * π * r^2)
  have hex_area_calc: hex_area = 96 * sqrt 3 := sorry
  have sector_area_calc: sector_area = 16 * π := sorry
  calc
    hex_area - sector_area
      = 96 * sqrt 3 - 16 * π : by
        rw [hex_area_calc, sector_area_calc]
        sorry

end shaded_region_area_l460_460481


namespace problem_discriminator_l460_460971

noncomputable def f (x : ℝ) : ℝ := abs (cos (2 * x)) + cos (abs x)

theorem problem_discriminator : 
¬ (∀ x y : ℝ, (x ∈ Icc (3 * π / 4) (3 * π / 2) ∧ y ∈ Icc (3 * π / 4) (3 * π / 2) ∧ x < y → f x ≤ f y))
∧ ¬ (∃ n : ℤ, ∀ x : ℝ, f (x + π * n) = f x)
∧ ¬ (∀ x : ℝ, x ∈ Icc (π / 4) (3 * π / 4) → f x ∈ Icc (-sqrt 2 / 2) (9 / 8)) 
∧ (∀ x : ℝ, f (-x) = f x) := by
  sorry

end problem_discriminator_l460_460971


namespace non_crossing_paths_count_l460_460503

-- Definitions
def alpha_paths : set string := {"ACB", "AEB"}
def beta_paths : set string := {"ADB", "AEB", "AFCB"}

-- Problem Statement
theorem non_crossing_paths_count : 
  -- Given the conditions for alpha and beta paths
  (alpha_paths = {"ACB", "AEB"}) ∧ 
  (beta_paths = {"ADB", "AEB", "AFCB"}) →
  -- Proof goal: there are 38 non-crossing paths
  (count_non_crossing_paths alpha_paths beta_paths) = 38 :=
sorry

-- Assume or define count_non_crossing_paths to calculate paths based on given sets
def count_non_crossing_paths (alpha_paths beta_paths : set string) : ℕ :=
  -- Definition logic according to given conditions or use a placeholder
  -- Here we assume it correctly calculates the number of non-crossing paths
  38

end non_crossing_paths_count_l460_460503


namespace happy_numbers_l460_460419

theorem happy_numbers (n : ℕ) (h1 : n < 1000) 
(h2 : 7 ∣ n^2) (h3 : 8 ∣ n^2) (h4 : 9 ∣ n^2) (h5 : 10 ∣ n^2) : 
n = 420 ∨ n = 840 :=
sorry

end happy_numbers_l460_460419


namespace otimes_2_5_l460_460722

def otimes (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem otimes_2_5 : otimes 2 5 = 23 :=
by
  sorry

end otimes_2_5_l460_460722


namespace num_possible_values_for_n_l460_460392

open Real

noncomputable def count_possible_values_for_n : ℕ :=
  let log2 := log 2
  let log2_9 := log 9 / log2
  let log2_50 := log 50 / log2
  let range_n := ((6 : ℕ), 450)
  let count := range_n.2 - range_n.1 + 1
  count

theorem num_possible_values_for_n :
  count_possible_values_for_n = 445 :=
by
  sorry

end num_possible_values_for_n_l460_460392


namespace abs_sum_eq_two_l460_460984

theorem abs_sum_eq_two (a b c : ℤ) (h : (a - b) ^ 10 + (a - c) ^ 10 = 1) : 
  abs (a - b) + abs (b - c) + abs (c - a) = 2 := 
sorry

end abs_sum_eq_two_l460_460984


namespace num_circles_num_circles_through_origin_num_circles_on_line_l460_460595

-- Define the given finite set of numbers
def nums : List ℕ := [0, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the circle equation and conditions
def circle_eq (a b r x y : ℝ) := (x - a)^2 + (y - b)^2 = r^2

-- Given conditions on a, b, r
def valid_elements (a b r : ℝ) :=
  a ∈ nums ∧ b ∈ nums ∧ r ∈ nums ∧ a ≠ b ∧ b ≠ r ∧ r ≠ a ∧ r > 0

-- Number of different circles that can be formed
theorem num_circles : ∃ n : ℝ, n = 448 :=
by sorry

-- Number of circles that pass through the origin
theorem num_circles_through_origin : ∃ n : ℝ, n = 4 :=
by sorry

-- Number of circles with centers on the line x + y - 10 = 0
theorem num_circles_on_line : ∃ n : ℝ, n = 38 :=
by sorry

end num_circles_num_circles_through_origin_num_circles_on_line_l460_460595


namespace length_of_KC_l460_460931

variables (A B C D K: Point)
variables (AD BC KC: ℝ)

-- Given the properties of the rectangle and the given conditions
axiom rectangle_ABCD : rectangle A B C D
axiom AD_eq_60 : AD = 60
axiom K_on_BC : on_line_segment K B C
axiom line_divides_areas : divides_area_ratio (line A K) (rectangle A B C D) (1 / 5)

-- Prove that KC = 48 given the conditions
theorem length_of_KC : KC = 48 :=
sorry

end length_of_KC_l460_460931


namespace num_elements_in_C_eq_7_l460_460248

def setA : Set ℤ := {x | (-3 < x) ∧ (x ≤ 2)}
def setB : Set ℕ := {x | (-2 < x) ∧ (x < 3)}

/-- 
  Proof problem: The number of elements in the set 
  \( C = \{ m \mid m = xy, x \in A, y \in B \} \) 
  is 7.
--/
theorem num_elements_in_C_eq_7 : 
  (Finset.card (Finset.ofSet {m : ℤ | ∃ x ∈ setA, ∃ y ∈ setB, m = x * y})) = 7 :=
  sorry

end num_elements_in_C_eq_7_l460_460248


namespace spring_sales_l460_460119

-- Given conditions
variable (T : ℕ) (winter_sales : ℕ) (summer_sales : ℕ) (fall_sales : ℕ)

-- Assumptions based on the problem conditions
axiom winter_sales_eq : winter_sales = 3 * 10^6
axiom summer_sales_eq : summer_sales = 4 * 10^6
axiom fall_sales_eq : fall_sales = 5 * 10^6
axiom winter_percentage : 20 * T / 100 = winter_sales

-- The statement to prove
theorem spring_sales (x : ℕ) :
  (20 * T / 100 = 3 * 10^6) ∧ (4 * 10^6 = summer_sales) ∧ (5 * 10^6 = fall_sales) ∧ (T = 15 * 10^6) → x = 3 * 10^6 :=
begin
  -- The steps of the proof would follow here
  sorry
end

end spring_sales_l460_460119


namespace cube_root_of_product_is_280_l460_460360

theorem cube_root_of_product_is_280 : (∛(2^9 * 5^3 * 7^3) = 280) := 
by 
sorry

end cube_root_of_product_is_280_l460_460360


namespace find_z_l460_460638

theorem find_z (x y z : ℚ) (hx : x = 11) (hy : y = -8) (h : 2 * x - 3 * z = 5 * y) :
  z = 62 / 3 :=
by
  sorry

end find_z_l460_460638


namespace factor_expression_l460_460877

theorem factor_expression (x : ℝ) :
  (16 * x ^ 7 + 36 * x ^ 4 - 9) - (4 * x ^ 7 - 6 * x ^ 4 - 9) = 6 * x ^ 4 * (2 * x ^ 3 + 7) :=
by
  sorry

end factor_expression_l460_460877


namespace evaluate_expression_l460_460524

theorem evaluate_expression : -(16 / 4 * 7 - 50 + 5 * 7) = -13 :=
by
  sorry

end evaluate_expression_l460_460524


namespace graph_symmetric_not_minimum_positive_period_pi_monochromatic_decreasing_not_minimum_value_minus_2_l460_460970

-- Define the function f
def f (x : ℝ) : ℝ := 2 * sin x * |cos x| + sqrt 3 * cos (2 * x)

-- Symmetry proof
theorem graph_symmetric (x : ℝ) : f (π - x) = f x := 
by
  sorry

-- Minimum positive period proof
theorem not_minimum_positive_period_pi : ¬(∀ x : ℝ, f (x + π) = f x) := 
by
  sorry

-- Monotonically decreasing proof
theorem monochromatic_decreasing : 
  ∀ (x : ℝ), x ∈ set.Icc (13 * π / 6) (5 * π / 2) → f x > f (x + 0.01) := 
by
  sorry

-- Minimum value proof
theorem not_minimum_value_minus_2 : ¬(∃ x : ℝ, x ∈ set.Icc 0 π ∧ f x = -2) := 
by
  sorry

end graph_symmetric_not_minimum_positive_period_pi_monochromatic_decreasing_not_minimum_value_minus_2_l460_460970


namespace Bryce_grapes_l460_460618

theorem Bryce_grapes : 
  ∃ x : ℝ, (∀ y : ℝ, y = (1/3) * x → y = x - 7) → x = 21 / 2 :=
by
  sorry

end Bryce_grapes_l460_460618


namespace sin_3x_solution_l460_460763

theorem sin_3x_solution (n : ℝ) :
  (∃ x : ℝ, sin (3 * x) = (n^2 - 5 * n + 3) * sin x ∧ sin x ≠ 0) ↔ (1 ≤ n ∧ n ≤ 4 ∨ n = (5 + Real.sqrt 17) / 2 ∨ n = (5 - Real.sqrt 17) / 2) :=
by sorry

end sin_3x_solution_l460_460763


namespace money_left_is_18000_l460_460472

def salary : ℕ := 180_000
def spent_on_food : ℕ := (1 / 5) * salary
def spent_on_house_rent : ℕ := (1 / 10) * salary
def spent_on_clothes : ℕ := (3 / 5) * salary
def total_spent : ℕ := spent_on_food + spent_on_house_rent + spent_on_clothes
def money_left : ℕ := salary - total_spent

theorem money_left_is_18000 : money_left = 18_000 := by
  sorry

end money_left_is_18000_l460_460472


namespace geometric_seq_common_ratio_l460_460299

theorem geometric_seq_common_ratio 
  (a : ℕ → ℝ) -- a_n is the sequence
  (S : ℕ → ℝ) -- S_n is the partial sum of the sequence
  (h1 : a 3 = 2 * S 2 + 1) -- condition a_3 = 2S_2 + 1
  (h2 : a 4 = 2 * S 3 + 1) -- condition a_4 = 2S_3 + 1
  (h3 : S 2 = a 1 / (1 / q) * (1 - q^3) / (1 - q)) -- sum of first 2 terms
  (h4 : S 3 = a 1 / (1 / q) * (1 - q^4) / (1 - q)) -- sum of first 3 terms
  : q = 3 := -- conclusion
by sorry

end geometric_seq_common_ratio_l460_460299


namespace range_of_m_l460_460201

noncomputable def f (x m : ℝ) : ℝ :=
  x^2 - 2 * m * x + m + 2

theorem range_of_m
  (m : ℝ)
  (h1 : ∃ a b : ℝ, f a m = 0 ∧ f b m = 0 ∧ a ≠ b)
  (h2 : ∀ x : ℝ, x ≥ 1 → 2*x - 2*m ≥ 0) :
  m < -1 :=
sorry

end range_of_m_l460_460201


namespace missing_weight_is_223_l460_460416

-- Mathematical problem encoded in Lean
theorem missing_weight_is_223
    (weights : Fin 10 → ℕ)
    (h_distinct : Function.Injective weights)
    (h_diff : weights 9 = weights 0 + 9)
    (remaining_sum : ℕ)
    (h_sum : remaining_sum = 2022)
    (missing_weight : ℕ)
    (h_missing : ∑ i in Finset.Ico 0 10, weights i - missing_weight = remaining_sum) :
  missing_weight = 223 := sorry

end missing_weight_is_223_l460_460416


namespace distance_between_planes_l460_460183

theorem distance_between_planes :
  ∀ (x y z : ℝ),
  (2*x - 4*y + 2*z = 10) →
  (x - 2*y + z = 5) →
  let d := (|2*5 + (-4)*0 + 2*0 - 10|) / (real.sqrt (2^2 + (-4)^2 + 2^2))
  d = 0 :=
begin
  intros x y z h1 h2,
  sorry
end

end distance_between_planes_l460_460183


namespace point_O_is_center_l460_460839

-- Definitions based on the conditions
def regular_decagon (c : Geometry.Circle) (p : Fin 10 → Geometry.Point) :=
  ∀ i : Fin 10, Geometry.PointOnCircle (p i) c ∧
  ∀ i j : Fin 10, (i ≠ j) → Geometry.Segment (p i) (p j) = Geometry.Segment (p ((i + 1) % 10)) (p ((j + 1) % 10))

def adjacent_verts (p : Fin 10 → Geometry.Point) (i : Fin 10) :=
  p i = p (i + 1 % 10)

def angle_OAB (O A B : Geometry.Point) :=
  Geometry.Angle (Geometry.Segment O A) (Geometry.Segment O B) = 72

-- The theorem we need to prove
theorem point_O_is_center (c : Geometry.Circle) (O A B : Geometry.Point) (p : Fin 10 → Geometry.Point)
  (decagon : regular_decagon c p)
  (adj_verts : adjacent_verts p 0)
  (angle_condition : angle_OAB O A B) :
  Geometry.PointIsCenter O c :=
sorry

end point_O_is_center_l460_460839


namespace max_students_distribute_pens_pencils_l460_460085

noncomputable def gcd_example : ℕ :=
  Nat.gcd 1340 1280

theorem max_students_distribute_pens_pencils : gcd_example = 20 :=
sorry

end max_students_distribute_pens_pencils_l460_460085


namespace option_d_correct_l460_460428

theorem option_d_correct (x : ℝ) : 2^x + 2^(-x) ≥ 2 :=
by
  -- sorry is a placeholder for the proof.
  sorry

end option_d_correct_l460_460428


namespace simplify_polynomial_l460_460169

theorem simplify_polynomial (x : ℝ) (A B C D : ℝ) :
  (y = (x^3 + 12 * x^2 + 47 * x + 60) / (x + 3)) →
  (y = A * x^2 + B * x + C) →
  x ≠ D →
  A = 1 ∧ B = 9 ∧ C = 20 ∧ D = -3 :=
by
  sorry

end simplify_polynomial_l460_460169


namespace non_adjacent_placements_l460_460199

theorem non_adjacent_placements (n : ℕ) : 
  let total_ways := n^2 * (n^2 - 1)
  let adjacent_ways := 2 * n^2 - 2 * n
  (total_ways - adjacent_ways) = n^4 - 3 * n^2 + 2 * n :=
by
  -- Proof is sorted out
  sorry

end non_adjacent_placements_l460_460199


namespace required_tiles_to_cover_floor_l460_460484

/- This problem statement defines the room and tile dimensions, calculates the required tiles to
   cover the room including a wastage percentage, and proves that the minimum required
   is 933 tiles. -/

def room_length_m : ℕ := 8
def room_length_cm : ℕ := 88
def room_breadth_m : ℕ := 4
def room_breadth_cm : ℕ := 62
def tile_side_cm : ℕ := 22
def wastage_percentage : ℕ := 10

def total_room_length_cm : ℕ := room_length_m * 100 + room_length_cm
def total_room_breadth_cm : ℕ := room_breadth_m * 100 + room_breadth_cm
def room_area_cm2 : ℕ := total_room_length_cm * total_room_breadth_cm
def tile_area_cm2 : ℕ := tile_side_cm * tile_side_cm
def base_tiles_required : ℕ := (room_area_cm2 + tile_area_cm2 - 1) // tile_area_cm2  -- equivalent to ceil(room_area / tile_area)
def additional_tiles : ℕ := (base_tiles_required * wastage_percentage + 99) // 100 -- equivalent to ceil(0.10 * base_tiles_required)
def total_tiles_required : ℕ := base_tiles_required + additional_tiles

theorem required_tiles_to_cover_floor : total_tiles_required = 933 := by
  -- Here, the theorem states that the computed total number of tiles required is 933
  sorry

end required_tiles_to_cover_floor_l460_460484


namespace roots_poly_eq_l460_460053

theorem roots_poly_eq (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : d = 0) (root1_eq : 64 * a + 16 * b + 4 * c = 0) (root2_eq : -27 * a + 9 * b - 3 * c = 0) :
  (b + c) / a = -13 :=
by {
  sorry
}

end roots_poly_eq_l460_460053


namespace tangent_line_equation_l460_460026

noncomputable def curve (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1
noncomputable def tangent_slope (x : ℝ) : ℝ := 3 * x^2 - 6 * x

theorem tangent_line_equation : 
  ∀ (x y : ℝ), 
  curve 1 = -1 → 
  tangent_slope 1 = -3 → 
  (∀ c : ℝ, y = -3 * x + c  → (x = 1 ∧ y = -1)  → c = 2) → 
  y = -3 * x + 2 := 
begin
  intros x y h_curve h_slope h_c,
  exact eq.symm (h_c _ rfl ⟨rfl, h_curve⟩)
end

end tangent_line_equation_l460_460026


namespace fraction_lost_down_sewer_l460_460501

-- Definitions of the conditions derived from the problem
def initial_marbles := 100
def street_loss_percent := 60 / 100
def sewer_loss := 40 - 20
def remaining_marbles_after_street := initial_marbles - (initial_marbles * street_loss_percent)
def marbles_left := 20

-- The theorem statement proving the fraction of remaining marbles lost down the sewer
theorem fraction_lost_down_sewer :
  (sewer_loss / remaining_marbles_after_street) = 1 / 2 :=
by
  sorry

end fraction_lost_down_sewer_l460_460501


namespace remainder_is_576_l460_460703

-- Definitions based on the conditions
def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def num_functions (f : ℕ → ℕ) (A : Set ℕ) : ℕ :=
  if ∃ c ∈ A, ∀ x ∈ A, f (f x) = c then 1 else 0

-- Define the total number of such functions
def total_functions (A : Set ℕ) : ℕ :=
  8 * (∑ k in ({1, 2, 3, 4, 5, 6, 7} : Set ℕ), Nat.choose 7 k * (k ^ (7 - k)))

-- The main theorem to prove
theorem remainder_is_576 : (total_functions A % 1000) = 576 :=
  by
    sorry

end remainder_is_576_l460_460703


namespace heartbeats_during_activity_l460_460766

def total_heartbeats (walk_time jog_time : ℕ) (walk_rate jog_rate : ℕ) : ℕ :=
  (walk_time * walk_rate) + (jog_time * jog_rate)

theorem heartbeats_during_activity :
  let walk_time := 30 in 
  let walk_rate := 90 in
  let jog_distance := 10 in
  let jog_pace := 6 in
  let jog_rate := 120 in
  let jog_time := jog_distance * jog_pace in
  total_heartbeats walk_time jog_time walk_rate jog_rate = 9900 := by
  sorry

end heartbeats_during_activity_l460_460766


namespace factorize_expression_l460_460529

-- Lean 4 statement for the proof problem
theorem factorize_expression (a b : ℝ) : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l460_460529


namespace robin_initial_gum_l460_460002

theorem robin_initial_gum (x : ℕ) (h1 : x + 26 = 44) : x = 18 := 
by 
  sorry

end robin_initial_gum_l460_460002


namespace cube_root_of_product_is_280_l460_460362

theorem cube_root_of_product_is_280 : (∛(2^9 * 5^3 * 7^3) = 280) := 
by 
sorry

end cube_root_of_product_is_280_l460_460362


namespace cubes_identity_l460_460983

theorem cubes_identity (a b c : ℝ) (h₁ : a + b + c = 15) (h₂ : ab + ac + bc = 40) : 
    a^3 + b^3 + c^3 - 3 * a * b * c = 1575 :=
by 
  sorry

end cubes_identity_l460_460983


namespace triangle_inequality_l460_460206

theorem triangle_inequality
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (sum_ABC : A + B + C = π)
  (h4: S_Triangle_ABC = (sqrt 3 / 12) * a^2)
  (h5: a = sqrt (b^2 + c^2 - 2 * b * c * cos A))
  (h6: a = sqrt (b^2 + c^2 - 2 * b * c * cos A)) :
  2 ≤ m ∧ m ≤ 4 := sorry

end triangle_inequality_l460_460206


namespace tangent_line_eq_at_point_a_eq_2_monotonicity_of_f_range_of_a_for_inequality_l460_460239

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x - 1 - a * real.log x

-- Proof Problem (I)
theorem tangent_line_eq_at_point_a_eq_2 : ∀ x : ℝ, f x 2 = x + y - 1 := sorry

-- Proof Problem (II)
theorem monotonicity_of_f : 
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → 1 - a / x ≥ 0 ↔ f x a is monotone) → 
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → (f x a is decreasing on (0, a) ∧ increasing on (a, +∞))) := sorry

-- Proof Problem (III)
theorem range_of_a_for_inequality : 
  ∀ a : ℝ, 
  (a < 0 ∧ ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 ≤ 1) ∧ (0 < x2 ∧ x2 ≤ 1) → 
   |f x1 a - f x2 a| ≤ 4 * |1 / x1 - 1 / x2|) 
   → -3 ≤ a ∧ a < 0 
:= sorry

end tangent_line_eq_at_point_a_eq_2_monotonicity_of_f_range_of_a_for_inequality_l460_460239


namespace sum_of_solutions_eq_16_l460_460186

theorem sum_of_solutions_eq_16 : (∑ x in {x : ℝ | |x^2 - 16*x + 60| = 10}.to_finset, x) = 16 :=
by {
  sorry
}

end sum_of_solutions_eq_16_l460_460186


namespace number_of_true_propositions_l460_460774

-- Definitions reflecting the given conditions
def is_regular_polygon (P : polygon) := (∀ s, P.has_side_length s) ∧ (∀ a, P.has_angle a)
def is_centrally_symmetric (P : polygon) := P.invariant_under_rotation 180
def regular_hexagon_radius_eq_side (P : polygon) := is_regular_polygon P ∧ P.sides = 6 → P.circumscribed_circle_radius = P.side_length
def regular_n_gon_axes_of_symmetry (P : polygon) (n : ℕ) := is_regular_polygon P ∧ P.sides = n → P.axes_of_symmetry = n

-- Propositions as per problem conditions
def prop1 := ∀ P : polygon, P.has_equal_sides → is_regular_polygon P
def prop2 := ∀ P : polygon, is_regular_polygon P → is_centrally_symmetric P
def prop3 := ∃ P : polygon, is_regular_polygon P ∧ P.sides = 6 ∧ regular_hexagon_radius_eq_side P
def prop4 := ∀ P : polygon, is_regular_polygon P → regular_n_gon_axes_of_symmetry P P.sides

-- Statement to check the number of true propositions
theorem number_of_true_propositions : (prop1 = false) ∧ (prop2 = false) ∧ (prop3 = true) ∧ (prop4 = true) → 2 = 2 :=
by
  sorry

end number_of_true_propositions_l460_460774


namespace number_of_correct_propositions_is_four_l460_460452

def parallel (x y : Type) := sorry -- parallelism relationship definition
def perpendicular (x y : Type) := sorry -- perpendicular relationship definition
def not_in (line : Type) (plane : Type) := sorry -- line not in plane definition

variables (m n : Type)
variables (α β : Type)

axiom line_not_in_planes : not_in m α ∧ not_in m β ∧ not_in n α ∧ not_in n β

axiom proposition_1 : parallel m n → parallel n α → parallel m α
axiom proposition_2 : parallel m β → parallel α β → parallel m α
axiom proposition_3 : perpendicular m n → perpendicular n α → parallel m α
axiom proposition_4 : perpendicular m β → perpendicular α β → parallel m α

theorem number_of_correct_propositions_is_four :
  (proposition_1 m n α β) ∧
  (proposition_2 m α β) ∧
  (proposition_3 m n α) ∧
  (proposition_4 m α β) :=
by 
  sorry

end number_of_correct_propositions_is_four_l460_460452


namespace area_of_right_triangle_l460_460449

-- Definition of the given conditions
variables (A B C H : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space H]
variables (AC AB AH HB : ℝ)

-- Conditions
def is_right_triangle (A B C : Type) : Prop := sorry
def hypotenuse_eq (AB : ℝ) : Prop := sorry
def leg_AC_eq (AC : ℝ) : Prop := (AC = 15)
def altitude_divides (AH HB : ℝ) : Prop := (HB = 16)

-- The statement to prove
theorem area_of_right_triangle (A B C H : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space H]
  (AC AB AH HB : ℝ) (h1: is_right_triangle A B C) (h2: hypotenuse_eq AB) (h3: leg_AC_eq AC) (h4: altitude_divides AH HB) :
  (AC * AB / 2 = 150) :=
sorry

end area_of_right_triangle_l460_460449


namespace inverse_undefined_at_one_l460_460986

noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x - 5)

theorem inverse_undefined_at_one : ∀ (x : ℝ), (x = 1) → ¬∃ y : ℝ, f y = x :=
by
  sorry

end inverse_undefined_at_one_l460_460986


namespace max_value_of_function_l460_460032

theorem max_value_of_function : 
  ∃ x : ℝ, (0 < x ∧ x < (real.pi / 2)) ∧ (f x = -real.sqrt 3) :=
sorry

def f (x : ℝ) : ℝ :=
  real.tan x - (2 / abs (real.cos x))

end max_value_of_function_l460_460032


namespace toy_cost_each_l460_460385

noncomputable def toyCost (x : ℝ) : ℝ := (3/2) * x

theorem toy_cost_each (x : ℝ) :
  2 * toyCost x = 36 -> x = 12 := 
by
  intros h
  calc x = 36 / 3 : by sorry
     ... = 12    : by sorry

end toy_cost_each_l460_460385


namespace green_more_than_blue_l460_460526

-- Define the conditions
variables (B Y G n : ℕ)
def ratio_condition := 3 * n = B ∧ 7 * n = Y ∧ 8 * n = G
def total_disks_condition := B + Y + G = 72

-- State the theorem
theorem green_more_than_blue (B Y G n : ℕ) 
  (h_ratio : ratio_condition B Y G n) 
  (h_total : total_disks_condition B Y G) 
  : G - B = 20 := 
sorry

end green_more_than_blue_l460_460526


namespace number_of_classes_l460_460460

variable (s : ℕ) (h_s : s > 0)
-- Define the conditions
def student_books_year : ℕ := 4 * 12
def total_books_read : ℕ := 48
def class_books_year (s : ℕ) : ℕ := s * student_books_year
def total_classes (c s : ℕ) (h_s : s > 0) : ℕ := 1

-- Define the main theorem
theorem number_of_classes (h : total_books_read = 48) (h_s : s > 0)
  (h1 : c * class_books_year s = 48) : c = 1 := by
  sorry

end number_of_classes_l460_460460


namespace range_of_m_l460_460930

open Real

noncomputable def complex_modulus_log_condition (m : ℝ) : Prop :=
  Complex.abs (Complex.log (m : ℂ) / Complex.log 2 + Complex.I * 4) ≤ 5

theorem range_of_m (m : ℝ) (h : complex_modulus_log_condition m) : 
  (1 / 8 : ℝ) ≤ m ∧ m ≤ (8 : ℝ) :=
sorry

end range_of_m_l460_460930


namespace complex_number_quadrant_l460_460295

def real_part (z : ℂ) : ℝ := z.re
def imaginary_part (z : ℂ) : ℝ := z.im

theorem complex_number_quadrant (z : ℂ) (hz : z = -1 + complex.i) :
  real_part z < 0 ∧ imaginary_part z > 0 :=
by
  have h1 : real_part z = -1 := by rw [hz]; exact rfl
  have h2 : imaginary_part z = 1 := by rw [hz]; exact rfl
  rw [h1, h2]
  exact And.intro (by linarith) (by linarith)

end complex_number_quadrant_l460_460295


namespace chip_cost_l460_460751

theorem chip_cost 
  (calories_per_chip : ℕ)
  (chips_per_bag : ℕ)
  (cost_per_bag : ℕ)
  (desired_calories : ℕ)
  (h1 : calories_per_chip = 10)
  (h2 : chips_per_bag = 24)
  (h3 : cost_per_bag = 2)
  (h4 : desired_calories = 480) : 
  cost_per_bag * (desired_calories / (calories_per_chip * chips_per_bag)) = 4 := 
by 
  sorry

end chip_cost_l460_460751


namespace jane_uses_40_ribbons_l460_460308

theorem jane_uses_40_ribbons :
  (∀ dresses1 dresses2 ribbons_per_dress, 
  dresses1 = 2 * 7 ∧ 
  dresses2 = 3 * 2 → 
  ribbons_per_dress = 2 → 
  (dresses1 + dresses2) * ribbons_per_dress = 40)
:= 
by 
  sorry

end jane_uses_40_ribbons_l460_460308


namespace max_black_cells_l460_460317

theorem max_black_cells {n : ℕ} (Q : matrix (fin (2*n+1)) (fin (2*n+1)) bool) 
  (h : ∀ i j, black_cells (Q ⟨i, by linarith⟩ ⟨j, by linarith⟩ + 
    Q ⟨i+1, by linarith⟩ ⟨j, by linarith⟩ + Q ⟨i, by linarith⟩ ⟨j+1, by linarith⟩ 
    + Q ⟨i+1, by linarith⟩ ⟨j+1, by linarith⟩ ≤ 2) : 
  black_cells_of_Q 2*n+1 ≤ (2*n+1) * (n+1) :=
sorry

-- Definitions for readability
-- Define black_cells to count black cells in a sub-matrix
def black_cells : matrix (fin 2) (fin 2) bool → ℕ :=
  -- Placeholder; assumes a function that counts the black cells in a 2x2 matrix
  sorry

-- Define black_cells_of_Q to count black cells in the full (2n+1) × (2n+1) matrix Q
def black_cells_of_Q (Q : matrix (fin (2*n+1)) (fin (2*n+1)) bool) : ℕ :=
  -- Placeholder; assumes a function that counts black cells in the whole matrix Q
  sorry

end max_black_cells_l460_460317


namespace product_equals_permutation_l460_460733

-- Given definitions and conditions
def product_of_consecutive (n : ℕ) (hn : n < 55) : ℕ :=
  (55 - n) * (56 - n) * (57 - n) * (58 - n) * (59 - n) *
  (60 - n) * (61 - n) * (62 - n) * (63 - n) * (64 - n) *
  (65 - n) * (66 - n) * (67 - n) * (68 - n) * (69 - n)

def permutation (k n : ℕ) : ℕ := k.perm n

-- Theorem statement
theorem product_equals_permutation (n : ℕ) (hn : n < 55) :
  product_of_consecutive n hn = permutation (69 - n) 15 :=
sorry

end product_equals_permutation_l460_460733


namespace profit_percentage_is_correct_l460_460833

noncomputable def sellingPrice : ℝ := 850
noncomputable def profit : ℝ := 230
noncomputable def costPrice : ℝ := sellingPrice - profit

noncomputable def profitPercentage : ℝ :=
  (profit / costPrice) * 100

theorem profit_percentage_is_correct :
  profitPercentage = 37.10 :=
by
  sorry

end profit_percentage_is_correct_l460_460833


namespace bullet_speed_difference_l460_460437

theorem bullet_speed_difference (speed_horse speed_bullet : ℕ) 
    (h_horse : speed_horse = 20) (h_bullet : speed_bullet = 400) :
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    speed_same_direction - speed_opposite_direction = 40 :=
    by
    -- Define the speeds in terms of the given conditions.
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    -- State the equality to prove.
    show speed_same_direction - speed_opposite_direction = 40;
    -- Proof (skipped here).
    -- sorry is used to denote where the formal proof steps would go.
    sorry

end bullet_speed_difference_l460_460437


namespace magnitude_of_z_l460_460735

noncomputable def z : ℂ := (Complex.I - 1)
def complex_condition (z : ℂ) : Prop := (Complex.I / (1 + Complex.I)) * z = 1
def magnitude (z : ℂ) : ℝ := Complex.abs z

theorem magnitude_of_z (z : ℂ) (h : complex_condition z) : magnitude z = Real.sqrt 2 :=
sorry

end magnitude_of_z_l460_460735


namespace t_range_exists_f_monotonic_intervals_l460_460237

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp (-x)
noncomputable def f' (x : ℝ) : ℝ := -x * Real.exp (-x)
noncomputable def φ (x t : ℝ) : ℝ := x * f x + t * f' x + Real.exp (-x)

theorem t_range_exists (t : ℝ) : ∃ (x_1 x_2 : ℝ), x_1 ∈ Icc 0 1 ∧ x_2 ∈ Icc 0 1 ∧ 2 * φ x_1 t < φ x_2 t
    ↔ t ∈ Iio (3 - 2 * Real.exp 1) ∪ Ioi (3 - Real.exp 1 / 2) := 
  sorry

theorem f_monotonic_intervals : 
  let intervals := ((Iio 0), (Ioi 0)) in
  ∀ x : ℝ, (x ∈ intervals.1 → MonotoneOn f intervals.1) ∧ (x ∈ intervals.2 → MonotoneOn f intervals.2) :=
  sorry

end t_range_exists_f_monotonic_intervals_l460_460237


namespace g_inv_undefined_at_x_one_l460_460636

def g (x : ℝ) := (x + 2) / (x - 5)

noncomputable def g_inv (x : ℝ) := (-2 - 5 * x) / (1 - x)

theorem g_inv_undefined_at_x_one  
: ∀ x : ℝ, g_inv x = g_inv 1 -> false := 
by
  intro x
  assume h,
  have : 1 - x = 0,
    from (calc
      1 - x = 0 : by sorry)
  contradiction

end g_inv_undefined_at_x_one_l460_460636


namespace LogarithmOfExpression_l460_460776

theorem LogarithmOfExpression :
  log (sqrt 2) (16 * (4 ^ (1/3)) * (32 ^ (1/5))) = 34 / 3 :=
by
  sorry

end LogarithmOfExpression_l460_460776


namespace find_angle_B_l460_460285

theorem find_angle_B 
  (a b : ℝ) (A : ℝ) (B : ℝ)
  (hA : A = π / 4)
  (ha : a = 2)
  (hb : b = sqrt 2)
  (hLawSines : a / real.sin A = b / real.sin B) : 
  B = π / 6 :=
by
  sorry

end find_angle_B_l460_460285


namespace part_I_part_II_l460_460588

-- Definitions of f and its domain
def f (x : ℝ) : ℝ := log (2 / (x + 1) - 1)
def A : Set ℝ := {x | -1 < x ∧ x < 1}

-- Problem part I
theorem part_I : f (1 / 2013) + f (-1 / 2013) = 0 :=
by
  sorry

-- Definitions of B and the condition
def B (a : ℝ) : Set ℝ := {x | 1 - a^2 - 2 * a * x - x^2 ≥ 0 }

-- Problem part II
theorem part_II (a : ℝ) : 
  (a ≥ 2 → A ∩ B a = ∅) ∧ ¬(∀ a, (A ∩ B a = ∅) → a ≥ 2) :=
by
  sorry

end part_I_part_II_l460_460588


namespace smallest_n_for_divisibility_by_ten_million_l460_460336

theorem smallest_n_for_divisibility_by_ten_million 
  (a₁ a₂ : ℝ) 
  (a₁_eq : a₁ = 5 / 6) 
  (a₂_eq : a₂ = 30) 
  (n : ℕ) 
  (T : ℕ → ℝ) 
  (T_def : ∀ (k : ℕ), T k = a₁ * (36 ^ (k - 1))) :
  (∃ n, T n = T 9 ∧ (∃ m : ℤ, T n = m * 10^7)) := 
sorry

end smallest_n_for_divisibility_by_ten_million_l460_460336


namespace solve_base_r_l460_460747

theorem solve_base_r (r : ℕ) (hr : Even r) (x : ℕ) (hx : x = 9999) 
                     (palindrome_condition : ∃ (a b c d : ℕ), 
                      b + c = 24 ∧ 
                      ∀ (r_repr : List ℕ), 
                      r_repr.length = 8 ∧
                      r_repr = [a, b, c, d, d, c, b, a] ∧ 
                      ∃ x_squared_repr, x^2 = x_squared_repr) : r = 26 :=
by
  sorry

end solve_base_r_l460_460747


namespace quadratic_inequality_solution_l460_460555

open Real

def quadratic (x : ℝ) : ℝ := x^2 - 42 * x + 390

theorem quadratic_inequality_solution :
  { x : ℝ | quadratic x ≤ 0 } = set.Icc 13 30 :=
by
  sorry

end quadratic_inequality_solution_l460_460555


namespace ratio_of_radii_l460_460655

open Real

theorem ratio_of_radii (a b : ℝ) (h : π * b^2 - π * a^2 = 5 * π * a^2) : a / b = 1 / sqrt 6 :=
by
  sorry

end ratio_of_radii_l460_460655


namespace remainder_of_N_div_1000_l460_460699

-- Defining the set A
def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}.to_finset

-- Defining the condition for N
def N : ℕ :=
  let k_choices := λ k, (7.choose k) * k^(7 - k)
  8 * Finset.sum (Finset.range 7) (λ k, k_choices (k + 1))

-- Theorem that must be proven
theorem remainder_of_N_div_1000 : N % 1000 = 992 := by
  sorry

end remainder_of_N_div_1000_l460_460699


namespace math_problem_l460_460990

theorem math_problem (x : Real) (hx : x + sqrt (x^2 - 9) + 1 / (x - sqrt (x^2 - 9)) = 100) :
  x^3 + sqrt (x^6 - 1) + 1 / (x^3 + sqrt (x^6 - 1)) = 31507.361338 :=
by 
  sorry

end math_problem_l460_460990


namespace polar_curve_is_parabola_l460_460024

theorem polar_curve_is_parabola (ρ θ : ℝ) (h : 3 * ρ * Real.sin θ ^ 2 + Real.cos θ = 0) : ∃ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ 3 * y ^ 2 + x = 0 :=
by
  sorry

end polar_curve_is_parabola_l460_460024


namespace unit_digit_sum_neq_2_l460_460355

theorem unit_digit_sum_neq_2 (n : ℕ) : ¬ (nat.digit_sum (n * (n + 1) / 2) 10 = 2) :=
sorry

end unit_digit_sum_neq_2_l460_460355


namespace find_min_distance_l460_460318

-- Define the set of lattice points
def S : set (ℤ × ℤ) := {p | |p.1| + |p.2| ≤ 10}

-- Define the sequence of 2013 points
structure Sequence (n : ℕ) :=
  (points : fin n → ℤ × ℤ)

noncomputable def min_distance (seq : Sequence 2013) : ℝ :=
  ∑ i in finset.range (2013 - 1), real.sqrt ((seq.points i.succ.1.1 - seq.points i.1.1)^2 + (seq.points i.succ.2.1 - seq.points i.2.1)^2)

theorem find_min_distance : 
  ∃ (a b c : ℕ), a + b * real.sqrt c = min_distance ⟨P_1, P_2, ..., P_2013⟩ (S) ∧ ¬ ∃ (p : ℕ), p^2 ∣ c ∧ a + b + c = 222
:= 
sorry

end find_min_distance_l460_460318


namespace find_a_l460_460160

theorem find_a (a : ℤ) : 0 ≤ a ∧ a ≤ 18 ∧ (235935623 - a ≡ 0 [MOD 9]) → a = 4 :=
by
  assume h : 0 ≤ a ∧ a ≤ 18 ∧ (235935623 - a ≡ 0 [MOD 9])
  -- h contains the conditions 0 ≤ a, a ≤ 18, and 235935623 ≡ a [MOD 9]
  sorry

end find_a_l460_460160


namespace dot_product_of_PF1_PF2_l460_460204

variable {x y : ℝ}

-- Conditions
def hyperbola (m : ℝ) (x y : ℝ) : Prop := x^2 - (y^2 / m) = 1

def eccentricity (m : ℝ) : Prop := (Real.sqrt (1 + m)) = 2

def point_on_hyperbola (m : ℝ) (P : ℝ × ℝ) : Prop := hyperbola m P.1 P.2

def sum_of_distances (PF1 PF2 : ℝ) : Prop := PF1 + PF2 = 10

-- Theorem to prove
theorem dot_product_of_PF1_PF2 (m : ℝ) (P : ℝ × ℝ) (PF1 PF2 : ℝ) :
  m > 0 →
  eccentricity m →
  point_on_hyperbola m P →
  sum_of_distances PF1 PF2 →
  (PF1 - PF2 = 2) →
  |PF1| = 6 →
  |PF2| = 4 →
  |F1F2| = 4 →
  ∃ cos_Angle : ℝ, cos_Angle = 3 / 4 →
  ∃ dot_product : ℝ, dot_product = PF1 * PF2 * cos_Angle :=
begin
  sorry
end

end dot_product_of_PF1_PF2_l460_460204


namespace intersection_A_B_l460_460580

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def B : Set ℝ := {-3, -2, -1, 0, 1, 2}

-- Define the intersection we need to prove
def A_cap_B_target : Set ℝ := {-2, -1, 0, 1}

-- Prove the intersection of A and B equals the target set
theorem intersection_A_B :
  A ∩ B = A_cap_B_target := 
sorry

end intersection_A_B_l460_460580


namespace minimum_positive_period_f_l460_460777

noncomputable def minimum_positive_period (f : ℝ → ℝ) : ℝ := sorry

def f (x : ℝ) : ℝ := cos (2 * x) - 2 * sqrt 3 * sin x * cos x

theorem minimum_positive_period_f : minimum_positive_period f = π := 
by 
  sorry

end minimum_positive_period_f_l460_460777


namespace find_number_l460_460790

theorem find_number (x : ℕ) (h : x + 1015 = 3016) : x = 2001 :=
sorry

end find_number_l460_460790


namespace num_possible_measures_of_A_l460_460396

-- Given conditions
variables (A B : ℕ)
variables (k : ℕ) (hk : k ≥ 1)
variables (hab : A + B = 180)
variables (ha : A = k * B)

-- The proof statement
theorem num_possible_measures_of_A : 
  ∃ (n : ℕ), n = 17 ∧ ∀ k, (k + 1) ∣ 180 ∧ k ≥ 1 → n = 17 := 
begin
  sorry
end

end num_possible_measures_of_A_l460_460396


namespace range_of_a_l460_460598

-- Definition of the function
def f (x : ℝ) : ℝ := 2 * x^3 - 4 * x + 2 * (Real.exp x - Real.exp (-x))

-- Main theorem statement
theorem range_of_a (a : ℝ) : f (5 * a - 2) + f (3 * a^2) ≤ 0 → -2 ≤ a ∧ a ≤ 1 / 3 :=
by
  sorry

end range_of_a_l460_460598


namespace probability_all_same_color_l460_460981

theorem probability_all_same_color :
  let p_purple := 6 / 30,
      p_green := 8 / 30,
      p_blue := 10 / 30,
      p_silver := 6 / 30,
      prob_purple := p_purple^3,
      prob_green := p_green^3,
      prob_blue := p_blue^3,
      prob_silver := p_silver^3,
      total_probability := prob_purple + prob_green + prob_blue + prob_silver
  in total_probability = 2 / 25 :=
by
  sorry

end probability_all_same_color_l460_460981


namespace larger_pile_toys_l460_460281

-- Define the conditions
def total_toys (small_pile large_pile : ℕ) : Prop := small_pile + large_pile = 120
def larger_pile (small_pile large_pile : ℕ) : Prop := large_pile = 2 * small_pile

-- Define the proof problem
theorem larger_pile_toys (small_pile large_pile : ℕ) (h1 : total_toys small_pile large_pile) (h2 : larger_pile small_pile large_pile) : 
  large_pile = 80 := by
  sorry

end larger_pile_toys_l460_460281


namespace problem_l460_460922

theorem problem (a b : ℝ) (h : a > b) : a / 3 > b / 3 :=
sorry

end problem_l460_460922


namespace numberOfBoys_playground_boys_count_l460_460795

-- Definitions and conditions
def numberOfGirls : ℕ := 28
def totalNumberOfChildren : ℕ := 63

-- Theorem statement
theorem numberOfBoys (numberOfGirls : ℕ) (totalNumberOfChildren : ℕ) : ℕ :=
  totalNumberOfChildren - numberOfGirls

-- Proof statement
theorem playground_boys_count (numberOfGirls : ℕ) (totalNumberOfChildren : ℕ) (boysOnPlayground : ℕ) : 
  numberOfGirls = 28 → 
  totalNumberOfChildren = 63 → 
  boysOnPlayground = totalNumberOfChildren - numberOfGirls →
  boysOnPlayground = 35 :=
by
  intros
  -- since no proof is required, we use sorry here
  exact sorry

end numberOfBoys_playground_boys_count_l460_460795


namespace solve_system_of_equations_l460_460765

theorem solve_system_of_equations :
  ∃ (x y : ℝ), x * y * (x + y) = 30 ∧ x^3 + y^3 = 35 ∧ ((x = 3 ∧ y = 2) ∨ (x = 2 ∧ y = 3)) :=
sorry

end solve_system_of_equations_l460_460765


namespace Sally_next_birthday_age_l460_460862

variables (a m s d : ℝ)

def Adam_older_than_Mary := a = 1.3 * m
def Mary_younger_than_Sally := m = 0.75 * s
def Sally_younger_than_Danielle := s = 0.8 * d
def Sum_ages := a + m + s + d = 60

theorem Sally_next_birthday_age (a m s d : ℝ) 
  (H1 : Adam_older_than_Mary a m)
  (H2 : Mary_younger_than_Sally m s)
  (H3 : Sally_younger_than_Danielle s d)
  (H4 : Sum_ages a m s d) : 
  s + 1 = 16 := 
by sorry

end Sally_next_birthday_age_l460_460862


namespace P_F1_F2_min_max_line_slope_range_l460_460573

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

noncomputable def P_F1_F2_dot (x y : ℝ) : ℝ :=
  x^2 + y^2 - 3

theorem P_F1_F2_min_max :
  ∀ (x y : ℝ), ellipse_eq x y →
  (-2 ≤ P_F1_F2_dot x y) ∧ (P_F1_F2_dot x y ≤ 1) :=
by sorry

noncomputable def line_eq (k x : ℝ) : ℝ := k * x + 2

theorem line_slope_range :
  ∀ k : ℝ, 
  (∀ x y: ℝ, ellipse_eq x y → line_eq k x = y → cos_angle_pos) →
  (-2 < k ∧ k < -sqrt 3 / 2) ∨ (sqrt 3 / 2 < k ∧ k < 2) :=
by sorry

end P_F1_F2_min_max_line_slope_range_l460_460573


namespace probability_point_in_sphere_l460_460477

noncomputable def probability_of_point_in_sphere : ℝ := 
  let volume_cube : ℝ := 4^3
  let volume_sphere : ℝ := (4/3) * π * (2^3)
  volume_sphere / volume_cube

theorem probability_point_in_sphere : 
  probability_of_point_in_sphere = π / 6 :=
by sorry

end probability_point_in_sphere_l460_460477


namespace adding_2_to_odd_integer_can_be_prime_l460_460170

def is_odd (n : ℤ) : Prop := n % 2 ≠ 0
def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m ∣ n → m = 1 ∨ m = n

theorem adding_2_to_odd_integer_can_be_prime :
  ∃ n : ℤ, is_odd n ∧ is_prime (n + 2) :=
by
  sorry

end adding_2_to_odd_integer_can_be_prime_l460_460170


namespace right_triangle_excircle_incircle_l460_460356

theorem right_triangle_excircle_incircle (a b c r r_a : ℝ) (h : a^2 + b^2 = c^2) :
  (r = (a + b - c) / 2) → (r_a = (b + c - a) / 2) → r_a = 2 * r :=
by
  intros hr hra
  sorry

end right_triangle_excircle_incircle_l460_460356


namespace find_a_l460_460222

noncomputable def tangent_line_slope (a : ℝ) : ℝ :=
  a * real.exp (a * 0)

theorem find_a
  (a : ℝ)
  (htangent : tangent_line_slope a = 2)
  (hperpendicular : true) -- This is a placeholder, as the perpendicular condition is implicitly checked by the slope comparison
  : a = 2 :=
by
  sorry

end find_a_l460_460222


namespace weight_of_second_square_l460_460487

noncomputable def weight_of_square (side_length : ℝ) (density : ℝ) : ℝ :=
  side_length^2 * density

theorem weight_of_second_square :
  let s1 := 4
  let m1 := 20
  let s2 := 7
  let density := m1 / (s1 ^ 2)
  ∃ (m2 : ℝ), m2 = 61.25 :=
by
  have s1 := 4
  have m1 := 20
  have s2 := 7
  let density := m1 / (s1 ^ 2)
  have m2 := weight_of_square s2 density
  use m2
  sorry

end weight_of_second_square_l460_460487


namespace tangent_lines_through_P_l460_460962

noncomputable def curve_eq (x : ℝ) : ℝ := 1/3 * x^3 + 4/3

theorem tangent_lines_through_P (x y : ℝ) :
  ((4 * x - y - 4 = 0 ∨ y = x + 2) ∧ (curve_eq 2 = 4)) :=
by
  sorry

end tangent_lines_through_P_l460_460962


namespace arithmetic_mean_first_49_positives_starting_from_5_l460_460769

theorem arithmetic_mean_first_49_positives_starting_from_5 :
  let seq := fun (n : ℕ) => n + 4 in
  (1 / 49) * (∑ i in Finset.range(49), seq (i + 1)) = 29 :=
by
  -- Definitions
  let seq := fun (n : ℕ) => n + 4
  -- Conditions incorporated in the definition
  have sum_49 : (∑ i in Finset.range(49), seq (i + 1)) = 1421 := sorry
  -- Using the computed sum to prove the mean
  show (1 / 49) * 1421 = 29
  -- Convert the multiplication into division and calculate
  calc
    (1 / 49) * 1421 = 1421 / 49 : by rw [div_eq_mul_one_div]
    ... = 29           : by norm_num

end arithmetic_mean_first_49_positives_starting_from_5_l460_460769


namespace transportation_cost_eqn_l460_460864

/-- Given conditions on transportation donations and costs, 
prove the functional relationship for the transportation cost, 
and derive the range of x such that the cost does not exceed 16.2 million yuan. -/
theorem transportation_cost_eqn (x : ℕ) (h₁ : 3 ≤ x) (h₂ : x ≤ 27) :
  let y := -0.2 * x + 21.3 in
  (y ≤ 16.2) ↔ (25.5 ≤ x) :=
by
  sorry

end transportation_cost_eqn_l460_460864


namespace geometric_series_common_ratio_l460_460884

theorem geometric_series_common_ratio :
  -- Define the terms in the geometric series
  let a1 := 7 / 8
  let a2 := -14 / 32
  let a3 := 56 / 256
  -- Define the ratio
  let r := a2 / a1 in
  a2 / a1 = -1 / 2 ∧ a3 / a2 = -1 / 2 :=
by
  sorry

end geometric_series_common_ratio_l460_460884


namespace minimum_cuts_chain_l460_460462

theorem minimum_cuts_chain (n : ℕ) (h : n = 60) : 
  ∃ cuts : list ℕ, cuts.length = 3 ∧ 
  (∀ m : ℕ, 1 ≤ m ∧ m ≤ 60 → ∃ pieces : list ℕ, 
    (∀ p ∈ pieces, p ∈ cuts ∨ p = 1) ∧ 
    pieces.sum = m) := by
  sorry

end minimum_cuts_chain_l460_460462


namespace g_at_2_l460_460388

-- Assuming g is a function from ℝ to ℝ such that it satisfies the given condition.
def g : ℝ → ℝ := sorry

-- Condition of the problem
axiom g_condition : ∀ x : ℝ, g (2 ^ x) + x * g (2 ^ (-x)) = 2

-- The statement we want to prove
theorem g_at_2 : g (2) = 0 :=
by
  sorry

end g_at_2_l460_460388


namespace round_robin_matches_l460_460873

-- Define the number of players in the tournament
def numPlayers : ℕ := 10

-- Define a function to calculate the number of matches in a round-robin tournament
def calculateMatches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

-- Theorem statement to prove that the number of matches in a 10-person round-robin chess tournament is 45
theorem round_robin_matches : calculateMatches numPlayers = 45 := by
  sorry

end round_robin_matches_l460_460873


namespace k_value_for_even_function_l460_460275

theorem k_value_for_even_function (k : ℝ) (f : ℝ → ℝ) (h1 : f = λ x, k * x^2 + (k - 1) * x + 3)
  (hf_even : ∀ x : ℝ, f (-x) = f x) : k = 1 :=
by
  sorry

end k_value_for_even_function_l460_460275


namespace average_monthly_production_correct_l460_460840

noncomputable def month_production (jan_prod : ℝ) (increments : List ℝ) : List ℝ :=
  List.scanl (λ prod incr, prod * (1 + incr / 100)) jan_prod increments

def year_production (jan_prod : ℝ) (increments : List ℝ) : ℝ :=
  (month_production jan_prod increments).sum / 12

theorem average_monthly_production_correct :
  (year_production 1000 
      [5, 7, 10, 4, 8, 5, 7, 6, 12, 10, 8]) 
  = 1445.084204 := 
sorry

end average_monthly_production_correct_l460_460840


namespace find_k_l460_460037

-- Given: The polynomial x^2 - 3k * x * y - 3y^2 + 6 * x * y - 8
-- We want to prove the value of k such that the polynomial does not contain the term "xy".

theorem find_k (k : ℝ) : 
  (∀ x y : ℝ, (x^2 - 3 * k * x * y - 3 * y^2 + 6 * x * y - 8) = x^2 - 3 * y^2 - 8) → 
  k = 2 := 
by
  intro h
  have h_coeff := h 1 1
  -- We should observe that the polynomial should not contain the xy term
  sorry

end find_k_l460_460037


namespace third_side_triangle_l460_460856

theorem third_side_triangle (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) : c ≠ 7 :=
by
  have h3 : a + b > c, from sorry
  have h4 : a + c > b, from sorry
  have h5 : b + c > a, from sorry
  have h6 : c < 7, from sorry
  have h7 : c > 1, from sorry
  sorry

end third_side_triangle_l460_460856


namespace cube_root_of_prime_product_l460_460369

theorem cube_root_of_prime_product : (∛(2^9 * 5^3 * 7^3) = 280) :=
by
  sorry

end cube_root_of_prime_product_l460_460369


namespace max_minute_hands_l460_460810

theorem max_minute_hands (m n : ℕ) (h1 : m * n = 27) : m + n ≤ 28 :=
by sorry

end max_minute_hands_l460_460810


namespace remainder_17_pow_63_mod_7_l460_460062

theorem remainder_17_pow_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end remainder_17_pow_63_mod_7_l460_460062


namespace first_term_of_geometric_series_l460_460500

theorem first_term_of_geometric_series (r : ℚ) (S : ℚ) (a : ℚ) (h1 : r = 1/5) (h2 : S = 100) (h3 : S = a / (1 - r)) : a = 80 := 
by
  sorry

end first_term_of_geometric_series_l460_460500


namespace min_cubes_to_build_box_l460_460077

theorem min_cubes_to_build_box :
  ∀ (length width height volume_cube : ℕ), 
  length = 7 → 
  width = 18 → 
  height = 3 → 
  volume_cube = 9 → 
  length * width * height / volume_cube = 42 :=
by
  intros length width height volume_cube hlen hwn hht hvol
  rw [hlen, hwn, hht, hvol]
  sorry

end min_cubes_to_build_box_l460_460077


namespace find_x_l460_460266

theorem find_x (x : ℤ) (h : (2 * x + 7) / 5 = 22) : x = 103 / 2 :=
by
  sorry

end find_x_l460_460266


namespace char_nums_and_eigenfunctions_l460_460182

noncomputable def integral_eq (λ : ℝ) (ϕ : ℝ → ℝ) : Prop :=
  ∀ x, ϕ x - λ * ∫ t in 0..π, (cos^2 x * cos(2 * t) + cos(3 * x) * cos^3 t) * ϕ t = 0

theorem char_nums_and_eigenfunctions :
  ∃ (λ1 λ2 : ℝ) (ϕ1 ϕ2 : ℝ → ℝ),
    λ1 = 4 / π ∧ λ2 = 8 / π ∧
    integral_eq λ1 (fun x => cos^2 x) ∧
    integral_eq λ2 (fun x => cos (3 * x)) :=
by
  sorry

end char_nums_and_eigenfunctions_l460_460182


namespace find_ab_l460_460813

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) :
  a * b = 10 :=
by
  sorry

end find_ab_l460_460813


namespace members_not_in_A_or_B_l460_460792

theorem members_not_in_A_or_B :
  ∀ (U A B : Finset ℕ),
    U.card = 193 →
    A.card = 110 →
    B.card = 49 →
    (A ∩ B).card = 25 →
    (U.card - (A.card + B.card - (A ∩ B).card)) = 59 :=
by
  intros U A B hU hA hB hAB
  have h := hU - (hA + hB - hAB)
  show h = 59
  sorry

end members_not_in_A_or_B_l460_460792


namespace restore_temperature_time_l460_460003

theorem restore_temperature_time :
  let rate_increase := 8 -- degrees per hour
  let duration_increase := 3 -- hours
  let rate_decrease := 4 -- degrees per hour
  let total_increase := rate_increase * duration_increase
  let time := total_increase / rate_decrease
  time = 6 := 
by
  sorry

end restore_temperature_time_l460_460003


namespace tree_height_at_2_years_l460_460850

theorem tree_height_at_2_years (h : ℕ → ℕ) 
  (h_growth : ∀ n, h (n + 1) = 3 * h n) 
  (h_5 : h 5 = 243) : 
  h 2 = 9 := 
sorry

end tree_height_at_2_years_l460_460850


namespace trig_inequality_l460_460547

theorem trig_inequality (a : ℝ): (∀ x : ℝ, sin x ^ 6 + cos x ^ 6 + 2 * a * sin x * cos x ≥ 0) →
  (-1 / 4 ≤ a) ∧ (a ≤ 1 / 4) :=
begin
  intro h,
  sorry
end

end trig_inequality_l460_460547


namespace log_of_y_pow_x_eq_neg4_l460_460819

theorem log_of_y_pow_x_eq_neg4 (x y : ℝ) (h : |x - 8 * y| + (4 * y - 1) ^ 2 = 0) : 
  Real.logb 2 (y ^ x) = -4 :=
sorry

end log_of_y_pow_x_eq_neg4_l460_460819


namespace inscribed_sphere_touches_centroids_of_regular_tetrahedron_l460_460506

-- Given condition
def inscribed_circle_touches_midpoints (T : Type) (circ : T → Prop) (pts_midpoints : T → Prop) : Prop := 
  ∀ (triangle : T), circ(triangle) → pts_midpoints(triangle)

-- Definition of an equilateral triangle
def equilateral_triangle (T : Type) (triangle : T) : Prop := sorry

-- Definition of a regular tetrahedron
def regular_tetrahedron (T : Type) (tetrahedron : T) : Prop := sorry

-- An inscribed sphere touches exactly at centroids
def inscribed_sphere_touches_centroids (T : Type) (sphere : T → Prop) (pts_centroids : T → Prop) : Prop := 
  ∀ (tetrahedron : T), sphere(tetrahedron) → pts_centroids(tetrahedron)

-- Theorem to prove
theorem inscribed_sphere_touches_centroids_of_regular_tetrahedron
  (T : Type)
  (circ : T → Prop)
  (sphere : T → Prop)
  (pts_midpoints : T → Prop)
  (pts_centroids : T → Prop)
  (eq_triangle : T)
  (reg_tetrahedron : T) :
  inscribed_circle_touches_midpoints T circ pts_midpoints →
  equilateral_triangle T eq_triangle →
  regular_tetrahedron T reg_tetrahedron →
  inscribed_sphere_touches_centroids T sphere pts_centroids :=
sorry

end inscribed_sphere_touches_centroids_of_regular_tetrahedron_l460_460506


namespace large_pile_toys_l460_460282

theorem large_pile_toys (x y : ℕ) (h1 : x + y = 120) (h2 : y = 2 * x) : y = 80 := by
  sorry

end large_pile_toys_l460_460282


namespace problem_proof_l460_460376

noncomputable def arithmetic_sequences (a b : ℕ → ℤ) (S T : ℕ → ℤ) :=
  ∀ n, S n = (n * (2 * a 0 + (n - 1) * (a 1 - a 0))) / 2 ∧
         T n = (n * (2 * b 0 + (n - 1) * (b 1 - b 0))) / 2

theorem problem_proof 
  (a b : ℕ → ℤ) 
  (S T : ℕ → ℤ)
  (h_seq : arithmetic_sequences a b S T)
  (h_relation : ∀ n, S n / T n = (7 * n : ℤ) / (n + 3)) :
  (a 5) / (b 5) = 21 / 4 :=
by 
  sorry

end problem_proof_l460_460376


namespace PQ_meets_q_at_R_l460_460737

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def P : ℝ × ℝ := (12, 9)
def Q : ℝ × ℝ := (4, 6)
def R : ℝ × ℝ := midpoint P Q

theorem PQ_meets_q_at_R : 3 * R.1 + 2 * R.2 = 39 := by
  sorry

end PQ_meets_q_at_R_l460_460737


namespace find_ordered_pair_l460_460184

theorem find_ordered_pair :
  ∃ x y : ℚ, 
  (x + 2 * y = (7 - x) + (7 - 2 * y)) ∧
  (3 * x - 2 * y = (x + 2) - (2 * y + 2)) ∧
  x = 0 ∧ 
  y = 7 / 2 :=
by
  sorry

end find_ordered_pair_l460_460184


namespace find_custom_operator_result_l460_460717

def custom_operator (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem find_custom_operator_result :
  custom_operator 2 5 = 23 :=
by
  sorry

end find_custom_operator_result_l460_460717


namespace Vasya_can_win_l460_460817

-- We need this library to avoid any import issues and provide necessary functionality for rational numbers

theorem Vasya_can_win :
  let a := (1 : ℚ) / 2009
  let b := (1 : ℚ) / 2008
  (∃ x : ℚ, a + x = 1) ∨ (∃ x : ℚ, b + x = 1) := sorry

end Vasya_can_win_l460_460817


namespace max_abs_value_sub_2i_max_l460_460268

noncomputable def max_abs_value_sub_2i (z : ℂ) (hz : complex.abs z = 1) : ℝ :=
  complex.abs (z - (2 * complex.I))

theorem max_abs_value_sub_2i_max (z : ℂ) (hz : complex.abs z = 1) :
  ∃ z : ℂ, complex.abs z = 1 ∧ max_abs_value_sub_2i z hz = 3 :=
sorry

end max_abs_value_sub_2i_max_l460_460268


namespace cos_double_angle_l460_460927

open Real

theorem cos_double_angle (α : ℝ) (h0 : 0 < α ∧ α < π) (h1 : sin α + cos α = 1 / 2) : cos (2 * α) = -sqrt 7 / 4 :=
by
  sorry

end cos_double_angle_l460_460927


namespace inequality_range_l460_460647

theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, x ∈ set.Icc (-2 : ℝ) (-1 : ℝ) → x^2 + x + a > 0) → a > 0 :=
begin
  sorry
end

end inequality_range_l460_460647


namespace no_integer_solutions_l460_460909

theorem no_integer_solutions (x y : ℤ) : x^3 + 4 * x^2 + x ≠ 18 * y^3 + 18 * y^2 + 6 * y + 3 := 
by 
  sorry

end no_integer_solutions_l460_460909


namespace sum_even_integers_602_to_700_l460_460447

theorem sum_even_integers_602_to_700 :
  let sum_first_50_pos_even := 2550,
      first_even := 602,
      last_even := 700,
      common_diff := 2,
      num_terms := (last_even - first_even) / common_diff + 1
  in
  2 * sum_first_50_pos_even = 32550 :=
by
  let sum_terms := num_terms / 2 * (first_even + last_even)
  have h : sum_terms = 32550 := sorry
  exact h

end sum_even_integers_602_to_700_l460_460447


namespace selling_price_correct_l460_460126

namespace Shopkeeper

def costPrice : ℝ := 1500
def profitPercentage : ℝ := 20
def expectedSellingPrice : ℝ := 1800

theorem selling_price_correct
  (cp : ℝ := costPrice)
  (pp : ℝ := profitPercentage) :
  cp * (1 + pp / 100) = expectedSellingPrice :=
by
  sorry

end Shopkeeper

end selling_price_correct_l460_460126


namespace exists_square_no_visible_points_l460_460190

-- Define visibility from the origin
def visible_from_origin (x y : ℤ) : Prop :=
  Int.gcd x y = 1

-- Main theorem statement
theorem exists_square_no_visible_points (n : ℕ) (hn : 0 < n) :
  ∃ (a b : ℤ), 
    (∀ (x y : ℤ), a ≤ x ∧ x ≤ a + n ∧ b ≤ y ∧ y ≤ b + n ∧ (x ≠ 0 ∨ y ≠ 0) → ¬visible_from_origin x y) :=
sorry

end exists_square_no_visible_points_l460_460190


namespace _l460_460714

variables (A B C P Q : Type)
           [Point A] [Point B] [Point C] [Point P] [Point Q]

-- Angle BAC is a right angle
axiom right_angle_BAC : angle_at A B C = 90

-- Midpoints of AB and BC are P and Q respectively
axiom midpoint_P : P = midpoint A B
axiom midpoint_Q : Q = midpoint B C

-- Given lengths
axiom AP_length : dist A P = 23
axiom CQ_length : dist C Q = 25

-- Defining distances AB and BC
def AB := 2 * (dist A P)
def BC := 2 * (dist C Q)

-- Use the Pythagorean theorem to determine AC
def AC := sqrt (AB ^ 2 + BC ^ 2)

lemma find_AC : AC = 68 :=
by 
  rw [AB, BC],
  norm_num,
  sorry

end _l460_460714


namespace polyhedron_volume_l460_460298

noncomputable def isosceles_right_triangle : Type := sorry
noncomputable def square (side_length : ℝ) : Type := sorry
noncomputable def equilateral_triangle (side_length : ℝ) : Type := sorry

def A : isosceles_right_triangle := sorry
def E : isosceles_right_triangle := sorry
def F : isosceles_right_triangle := sorry
def B : square 1 := sorry
def C : square 1 := sorry
def D : square 1 := sorry
def G : equilateral_triangle (Real.sqrt 2) := sorry

theorem polyhedron_volume : volume (fold_polyhedron A E F B C D G) = 5 / 6 :=
by
  sorry

end polyhedron_volume_l460_460298


namespace max_S_partition_l460_460581

theorem max_S_partition (S : ℝ) :
  (∀ (x : ℝ), x ∈ Icc 0 1 → ∑ x_i ≤ S → ∃ (A B : ℝ), A + B = S ∧ A ≤ 8 ∧ B ≤ 4) ↔ S = 11.2 := 
sorry

end max_S_partition_l460_460581


namespace bullet_speed_difference_l460_460432

def speed_horse : ℕ := 20  -- feet per second
def speed_bullet : ℕ := 400  -- feet per second

def speed_forward : ℕ := speed_bullet + speed_horse
def speed_backward : ℕ := speed_bullet - speed_horse

theorem bullet_speed_difference : speed_forward - speed_backward = 40 :=
by
  sorry

end bullet_speed_difference_l460_460432


namespace reciprocal_of_neg_two_l460_460785

theorem reciprocal_of_neg_two :
  (∃ x : ℝ, x = -2 ∧ 1 / x = -1 / 2) :=
by
  use -2
  split
  · rfl
  · norm_num

end reciprocal_of_neg_two_l460_460785


namespace derivative_of_f_at_a_l460_460729

noncomputable theory

def f (x : ℝ) : ℝ := 
  let f₀ := x
  let f₁ := cos f₀
  let f₂ := cos f₁
  let f₃ := cos f₂
  let f₄ := cos f₃
  let f₅ := cos f₄
  let f₆ := cos f₅
  let f₇ := cos f₆
  let f₈ := cos f₇
  f₈

theorem derivative_of_f_at_a (a : ℝ) (h : a = cos a) : 
  deriv f a = a^8 - 4*a^6 + 6*a^4 - 4*a^2 + 1 :=
  sorry

end derivative_of_f_at_a_l460_460729


namespace find_parameters_l460_460393

theorem find_parameters (s m : ℝ) :
  (∃ t : ℝ, (∃ (x y : ℝ), y = 2 * x - 8 ∧ (∀ t, (x, y) = (s + 6 * t, 5 + m * t))) ∧ x = 12) →
  (s, m) = (13 / 2 : ℝ, 11 : ℝ) :=
by
  sorry

end find_parameters_l460_460393


namespace zander_stickers_l460_460075

theorem zander_stickers (total_stickers andrew_ratio bill_ratio : ℕ) (initial_stickers: total_stickers = 100) (andrew_fraction : andrew_ratio = 1 / 5) (bill_fraction : bill_ratio = 3 / 10) :
  let andrew_give_away := total_stickers * andrew_ratio
  let remaining_stickers := total_stickers - andrew_give_away
  let bill_give_away := remaining_stickers * bill_ratio
  let total_given_away := andrew_give_away + bill_give_away
  total_given_away = 44 :=
by
  sorry

end zander_stickers_l460_460075


namespace smallest_possible_AAB_l460_460489

-- Definitions of the digits A and B
def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

-- Definition of the condition AB equals 1/7 of AAB
def condition (A B : ℕ) : Prop := 10 * A + B = (1 / 7) * (110 * A + B)

theorem smallest_possible_AAB (A B : ℕ) : is_valid_digit A ∧ is_valid_digit B ∧ condition A B → 110 * A + B = 664 := sorry

end smallest_possible_AAB_l460_460489


namespace john_speed_l460_460314

def johns_speed (race_distance_miles next_fastest_guy_time_min won_by_min : ℕ) : ℕ :=
    let john_time_min := next_fastest_guy_time_min - won_by_min
    let john_time_hr := john_time_min / 60
    race_distance_miles / john_time_hr

theorem john_speed (race_distance_miles next_fastest_guy_time_min won_by_min : ℕ)
    (h1 : race_distance_miles = 5) (h2 : next_fastest_guy_time_min = 23) (h3 : won_by_min = 3) : 
    johns_speed race_distance_miles next_fastest_guy_time_min won_by_min = 15 := 
by
    sorry

end john_speed_l460_460314


namespace coloring_impossible_l460_460876

theorem coloring_impossible :
  ¬(∃ (color : Fin 1990 → Fin 1990 → bool),
      (∀ i : Fin 1990, (∑ j, if (color i j) then 1 else 0) = 995) ∧
      (∀ j : Fin 1990, (∑ i, if (color i j) then 1 else 0) = 995) ∧
      (∀ i j : Fin 1990, color i j ≠ color (Fin 1990 - 1 - i) (Fin 1990 - 1 - j))) :=
sorry

end coloring_impossible_l460_460876


namespace rent_percentage_l460_460688

-- Define Elaine's earnings last year
def E : ℝ := sorry

-- Define last year's rent expenditure
def rentLastYear : ℝ := 0.20 * E

-- Define this year's earnings
def earningsThisYear : ℝ := 1.35 * E

-- Define this year's rent expenditure
def rentThisYear : ℝ := 0.30 * earningsThisYear

-- Prove the required percentage
theorem rent_percentage : ((rentThisYear / rentLastYear) * 100) = 202.5 := by
  sorry

end rent_percentage_l460_460688


namespace simplify_expression_evaluate_l460_460374

theorem simplify_expression_evaluate : 
  let x := 1
  let y := 2
  (2 * x - y) * (y + 2 * x) - (2 * y + x) * (2 * y - x) = -15 :=
by
  sorry

end simplify_expression_evaluate_l460_460374


namespace triangle_altitude_l460_460814

theorem triangle_altitude (base side : ℝ) (h : ℝ) : 
  side = 6 → base = 6 → 
  (base * h) / 2 = side ^ 2 → 
  h = 12 :=
by
  intros
  sorry

end triangle_altitude_l460_460814


namespace unique_subset_splits_count_l460_460249

def is_split_of (A A1 A2 : Set ℕ) : Prop :=
  A1 ∪ A2 = A

def is_unique_split_pair (A A1 A2 : Set ℕ) : Prop :=
  ∀ (B1 B2 : Set ℕ), is_split_of A B1 B2 → (A1 = B1 ∧ A2 = B2) ∨ (A1 = B2 ∧ A2 = B1)

theorem unique_subset_splits_count (A : Set ℕ) (h : A = {1, 2, 3}) : 
  ∃ n, n = 14 ∧ n = (Set.toFinset (Set.powerset A)).card / 2 :=
by
  sorry

end unique_subset_splits_count_l460_460249


namespace prove_incorrect_description_l460_460133

-- Definitions of the conditions as stated in mathematical equivalents
def separates_organelles_differential_centrifugation (O : Type) : Prop := 
  ∀ o : O, differential_centrifugation o

def isolates_microorganisms_congo_red_urea : Prop := 
  ¬ (medium_contains_congo_red_isolates_urea_decomposing_microorganisms)

def separates_dna_na_cl_solubility : Prop := 
  ∀ d : DNA, separates_by_solubility_in_na_cl d

def separates_proteins_gel_chromatography : Prop := 
  ∀ p : Protein, gel_chromatography_separates_by_relative_mass p

-- Definition that captures the incorrect description in the problem statement
def incorrect_description : Prop :=
  separates_organelles_differential_centrifugation Organelles ∧
  isolates_microorganisms_congo_red_urea ∧
  separates_dna_na_cl_solubility DNA ∧
  separates_proteins_gel_chromatography Proteins ∧
  incorrect_answer = "B"

-- Lean statement to prove the incorrect description
theorem prove_incorrect_description : 
  separates_organelles_differential_centrifugation Organelles →
  isolates_microorganisms_congo_red_urea →
  separates_dna_na_cl_solubility DNA →
  separates_proteins_gel_chromatography Proteins →
  incorrect_description :=
by sorry

end prove_incorrect_description_l460_460133


namespace remainder_of_functions_mod_1000_l460_460694

noncomputable def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def number_of_functions := 
  let N := 8 * ∑ k in (Finset.range 7).filter (λ k, k > 0), (Nat.choose 7 k) * (k ^ (7 - k)) 
  in N

theorem remainder_of_functions_mod_1000 : number_of_functions % 1000 = 576 :=
by 
  let N := number_of_functions
  have h1 : N = 50576 := sorry
  have h2 : 50576 % 1000 = 576 := rfl
  rw [h1, h2]

end remainder_of_functions_mod_1000_l460_460694


namespace six_digit_even_count_correct_greater_than_102345_count_correct_l460_460592

-- Definitions:
-- A function to count permutations (without repetitions) based on conditions for six-digit even numbers
def even_count (digits : Finset ℕ) : ℕ :=
  let last_is_zero := (digits.erase 0).card.perm (digits.erase 0).card
  let last_is_two_or_four := (Finset.singleton 2 ∪ Finset.singleton 4).card.perm 1 * (digits.erase 2 ∪ digits.erase 4).card.perm (digits.erase 2 ∪ digits.erase 4).card
  last_is_zero + last_is_two_or_four

-- Function to count natural numbers greater than 102345 without repeating any digit from given digits
def natural_count (digits : Finset ℕ) : ℕ :=
  let total_permutations := digits.card.perm digits.card
  total_permutations - 1

-- Prove that the counts are as expected

theorem six_digit_even_count_correct (digits : Finset ℕ) : 
  digits = {0, 1, 2, 3, 4, 5} → even_count digits = 312 :=
by sorry

theorem greater_than_102345_count_correct (digits : Finset ℕ) : 
  digits = {0, 1, 2, 3, 4, 5} → natural_count digits = 599 :=
by sorry

end six_digit_even_count_correct_greater_than_102345_count_correct_l460_460592


namespace closest_integer_odd_probability_l460_460123

theorem closest_integer_odd_probability :
  let interval := Set.Icc (-15.5 : ℝ) 15.5 in
  let odd_integers := {n : ℤ | n % 2 ≠ 0 ∧ -15.5 ≤ (n : ℝ) + 0.5 ∧ (n : ℝ) - 0.5 ≤ 15.5} in
  let odd_intervals_length := card odd_integers in
  let total_interval_length := (31 : ℝ) in
  (odd_intervals_length / total_interval_length) = (16 / 31) :=
  sorry

end closest_integer_odd_probability_l460_460123


namespace fib_ratio_bound_l460_460018

def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n+1) + fib n

theorem fib_ratio_bound {a b n : ℕ} (h1: b > 0) (h2: fib (n-1) > 0)
  (h3: (fib n) * b > (fib (n-1)) * a)
  (h4: (fib (n+1)) * b < (fib n) * a) :
  b ≥ fib (n+1) :=
sorry

end fib_ratio_bound_l460_460018


namespace triangle_area_202_2192_pi_squared_l460_460860

noncomputable def triangle_area (a b c : ℝ) : ℝ := 
  let r := (a + b + c) / (2 * Real.pi)
  let theta := 20.0 * Real.pi / 180.0  -- converting 20 degrees to radians
  let angle1 := 5 * theta
  let angle2 := 6 * theta
  let angle3 := 7 * theta
  (1 / 2) * r * r * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem triangle_area_202_2192_pi_squared (a b c : ℝ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 7) : 
  triangle_area a b c = 202.2192 / (Real.pi * Real.pi) := 
by {
  sorry
}

end triangle_area_202_2192_pi_squared_l460_460860


namespace range_of_m_l460_460992

def inFirstOrThirdQuadrant(z : ℂ) : Prop :=
  (z.re > 0 ∧ z.im < 0) ∨ (z.re < 0 ∧ z.im > 0)

theorem range_of_m (m : ℝ) :
  inFirstOrThirdQuadrant ((m + 1) - (m - 3) * complex.I) ↔ (-1 < m ∧ m < 3) := by
  sorry

end range_of_m_l460_460992


namespace donuts_left_in_box_l460_460149

def initial_donuts := 50

def after_bill_eats := initial_donuts - 2

def after_secretary_takes := after_bill_eats - 4

def after_manager_takes := (after_secretary_takes - (after_secretary_takes / 10 : ℝ).floor.to_nat)

def after_coworkers_eat := (after_manager_takes - (after_manager_takes / 3 : ℝ).floor.to_nat)

def final_donuts := (after_coworkers_eat - (after_coworkers_eat / 2 : ℝ).floor.to_nat)

theorem donuts_left_in_box : final_donuts = 14 := by
  sorry

end donuts_left_in_box_l460_460149


namespace complex_number_quadrant_l460_460381

noncomputable def z : ℂ := (2 + complex.i) * complex.i

theorem complex_number_quadrant :
  z = -1 + 2 * complex.i →
  ∃ q ∈ {1, 2, 3, 4}, q = 2 := 
sorry

end complex_number_quadrant_l460_460381


namespace arithmetic_progression_25th_term_l460_460538

theorem arithmetic_progression_25th_term (a1 d : ℤ) (n : ℕ) (h_a1 : a1 = 5) (h_d : d = 7) (h_n : n = 25) :
  a1 + (n - 1) * d = 173 :=
by
  sorry

end arithmetic_progression_25th_term_l460_460538


namespace circumcenter_triangle_l460_460616

/-- Given:
* O is the circumcenter of triangle ABC
* R is the radius of the circumcircle
* lines AO, BO, and CO intersect the opposite sides at points D, E, and F respectively.
Prove: 
  1/AD + 1/BE + 1/CF = 2/R
-/
theorem circumcenter_triangle
  (A B C O D E F : Type)
  (R : ℝ)
  (h_circumcenter : ∀ (P : Type), P = O → false)
  (h_Radius : ∀ (X : Type), X = R → false)
  (h_AD : ∀ (Y : Type), Y = D → false)
  (h_BE : ∀ (Z : Type), Z = E → false)
  (h_CF : ∀ (W : Type), W = F → false):
  1 / (AD.cast ℝ) + 1 / (BE.cast ℝ) + 1 / (CF.cast ℝ) = 2 / R := 
by sorry

end circumcenter_triangle_l460_460616


namespace omega_real_iff_m_values_omega_in_fourth_quadrant_iff_m_range_l460_460961

variable (m : ℝ)

def omega : ℂ := (m^2 - 2*m - 3 : ℝ) + (m^2 - m - 12 : ℝ) * complex.I

theorem omega_real_iff_m_values :
  ((m^2 - m - 12 = 0) ↔ (m = 4 ∨ m = -3)) :=
by sorry

theorem omega_in_fourth_quadrant_iff_m_range :
  ((m^2 - 2*m - 3 > 0) ∧ (m^2 - m - 12 < 0)) ↔ (3 < m ∧ m < 4) :=
by sorry

end omega_real_iff_m_values_omega_in_fourth_quadrant_iff_m_range_l460_460961


namespace tangent_lines_through_point_chord_midpoint_l460_460565

-- Define the circle equation as a condition
def circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 5 = 0

-- 1. Tangent Lines Problem
theorem tangent_lines_through_point (x y : ℝ) (hx : x = 5) (hy : y = 1) :
    (∃ k : ℝ, ∀ x y, k * x + y - 1 - 5 * k = 0) ∨ (4 * x + 3 * y - 23 = 0 ∨ x = 5) :=
sorry

-- 2. Chord Midpoint Problem
theorem chord_midpoint (P : ℝ × ℝ) (hx : P.1 = 3) (hy : P.2 = 1) :
    ∃ A B : ℝ × ℝ,
    (circle A.1 A.2 ∧ circle B.1 B.2 ∧ (A.1 + B.1) / 2 = P.1 ∧ (A.2 + B.2) / 2 = P.2) ∧ ∀ x y, x + y - 4 = 0 :=
sorry

end tangent_lines_through_point_chord_midpoint_l460_460565


namespace jill_speed_downhill_l460_460312

theorem jill_speed_downhill 
  (up_speed : ℕ) (total_time : ℕ) (hill_distance : ℕ) 
  (up_time : ℕ) (down_time : ℕ) (down_speed : ℕ) 
  (h1 : up_speed = 9)
  (h2 : total_time = 175)
  (h3 : hill_distance = 900)
  (h4 : up_time = hill_distance / up_speed)
  (h5 : down_time = total_time - up_time)
  (h6 : down_speed = hill_distance / down_time) :
  down_speed = 12 := 
  by
    sorry

end jill_speed_downhill_l460_460312


namespace total_chocolate_bars_in_large_box_l460_460444

def large_box_contains_18_small_boxes : ℕ := 18
def small_box_contains_28_chocolate_bars : ℕ := 28

theorem total_chocolate_bars_in_large_box :
  (large_box_contains_18_small_boxes * small_box_contains_28_chocolate_bars) = 504 := 
by
  sorry

end total_chocolate_bars_in_large_box_l460_460444


namespace bullet_speed_difference_l460_460435

def bullet_speed_in_same_direction (v_h v_b : ℝ) : ℝ :=
  v_b + v_h

def bullet_speed_in_opposite_direction (v_h v_b : ℝ) : ℝ :=
  v_b - v_h

theorem bullet_speed_difference (v_h v_b : ℝ) (h_h : v_h = 20) (h_b : v_b = 400) :
  bullet_speed_in_same_direction v_h v_b - bullet_speed_in_opposite_direction v_h v_b = 40 :=
by
  rw [h_h, h_b]
  sorry

end bullet_speed_difference_l460_460435


namespace find_base_r_l460_460745

noncomputable def x: ℕ := 9999

theorem find_base_r (r: ℕ) (hr_even: Even r) (hr_gt_9: r > 9) 
    (h_palindrome: ∃ a b c d: ℕ, b + c = 24 ∧ 
                   ((81 * ((r^6 * (r^6 + 2 * r^5 + 3 * r^4 + 4 * r^3 + 3 * r^2 + 2 * r + 1 + r^2)) = 
                     a * r^7 + b * r^6 + c * r^5 + d * r^4 + d * r^3 + c * r^2 + b * r + a)))):
    r = 26 :=
by
  sorry

end find_base_r_l460_460745


namespace sum_series_value_l460_460510

def sum_series : ℝ :=
  ∑ a in Finset.Ico 1 (a + 1),
  ∑ b in Finset.Ico (a + 1) (b + 1),
  ∑ c in Finset.Ico (b + 1) (c + 1), 
  1 / (3^a * 4^b * 6^c)

theorem sum_series_value : sum_series = 1 / 27041 := 
  sorry

end sum_series_value_l460_460510


namespace corrected_mean_after_adjusting_errors_l460_460779

theorem corrected_mean_after_adjusting_errors :
  let n := 50,
      original_mean := 36,
      incorrect_observations := (23, 55),
      correct_observations := (34, 45) in
  let original_sum := original_mean * n in
  let adjusted_sum := original_sum - fst incorrect_observations - snd incorrect_observations + fst correct_observations + snd correct_observations in
  let new_mean := adjusted_sum / n in
  new_mean = 36.02 :=
by
  let n := 50
  let original_mean := 36
  let incorrect_observations := (23, 55)
  let correct_observations := (34, 45)
  let original_sum := original_mean * n
  let adjusted_sum := original_sum - fst incorrect_observations - snd incorrect_observations + fst correct_observations + snd correct_observations
  let new_mean := adjusted_sum / n
  have h : new_mean = 36.02 := sorry
  exact h

end corrected_mean_after_adjusting_errors_l460_460779


namespace parabola_vertex_l460_460391

theorem parabola_vertex (a b c : ℤ) :
  (∀ x : ℤ, (x - 2)^2 + c = y) ∧                   -- Vertex form condition
  (∃ y : ℤ, (2, 5) = y) ∧                         -- Vertex point
  (∃ y : ℤ, (1, 2) = y) ∧                         -- Point (1, 2)
  (∃ y : ℤ, (3, 2) = y)                           -- Point (3, 2)
  → a = -3 :=
sorry

end parabola_vertex_l460_460391


namespace jelly_bean_pile_weight_l460_460495

theorem jelly_bean_pile_weight :
  ∀ (initial_weight eating_weight num_piles : ℕ), 
    initial_weight = 72 → 
    eating_weight = 12 → 
    num_piles = 5 → 
    (initial_weight - eating_weight) / num_piles = 12 := 
by
  intros initial_weight eating_weight num_piles h_initial h_eating h_piles
  rw [h_initial, h_eating, h_piles]
  norm_num
  sorry

end jelly_bean_pile_weight_l460_460495


namespace intersection_points_count_l460_460885

theorem intersection_points_count :
  {p : ℝ × ℝ // p.1^2 + p.2^2 = 16} ∩ {p : ℝ × ℝ // p.1^2 = 4} = 
  {((2 : ℝ), 2 * real.sqrt 3), ((2 : ℝ), -2 * real.sqrt 3), ((-2 : ℝ), 2 * real.sqrt 3), ((-2 : ℝ), -2 * real.sqrt 3)} :=
sorry

end intersection_points_count_l460_460885


namespace circle_area_ratio_l460_460464

-- Variables for lengths and conditions
def AB : ℝ := sorry -- length of AB
def P : ℝ := AB / 42 -- point P such that AP/AB = 1/42
def AP : ℝ := AB / 42
def PB : ℝ := AB - AP

def SP : ℝ := (AB * Real.sqrt 41) / 42 -- Calculated from the problem
def ST : ℝ := (2 * SP) -- Calculated from the problem

-- Definition of circle areas
def A (d : ℝ) : ℝ := Real.pi * (d / 2) ^ 2

-- Area of circles a, b, s, t
def A_a : ℝ := A AP
def A_b : ℝ := A PB
def A_s : ℝ := A SP
def A_t : ℝ := A ST

-- Statement to be proved
theorem circle_area_ratio :
  (A_s + A_t) / (A_a + A_b) = 205 / 1681 :=
by {
  sorry
}

end circle_area_ratio_l460_460464


namespace compute_difference_l460_460988

def bin_op (x y : ℤ) : ℤ := x^2 * y - 3 * x

theorem compute_difference :
  (bin_op 5 3) - (bin_op 3 5) = 24 := by
  sorry

end compute_difference_l460_460988


namespace snail_number_is_square_l460_460342

def isSnailNumber (n : Nat) : Prop :=
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  digits ∈ [[8, 9, 1, 0], [8, 9, 0, 1], [9, 8, 1, 0], [9, 8, 0, 1],
            [8, 1, 9, 0], [8, 0, 9, 1], [1, 8, 9, 0], [1, 0, 9, 8],
            [0, 8, 9, 1], [0, 1, 9, 8], [9, 1, 0, 8], [9, 0, 1, 8]]

theorem snail_number_is_square : ∃ n : Nat, isSnailNumber n ∧ n = 33^2 :=
by 
  use 1089
  unfold isSnailNumber
  -- Manually expand the unfolding for demonstration
  have digits : 1089 / 1000 = 1 → 1089 / 100 % 10 = 0 → 1089 / 10 % 10 = 8 → 1089 % 10 = 9
  exact sorry

end snail_number_is_square_l460_460342


namespace max_sqrt_expression_l460_460583

theorem max_sqrt_expression (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + 2 * b + 3 * c = 6) :
  sqrt (a + 1) + sqrt (2 * b + 1) + sqrt (3 * c + 1) ≤ 3 * sqrt 3 := by
sorry

end max_sqrt_expression_l460_460583


namespace probability_not_all_colors_same_is_8_over_9_l460_460094

-- Define the basic elements: Yellow, Red, and White balls
inductive Ball
| yellow : Ball
| red : Ball
| white : Ball

-- Define the conditions: drawing a ball 3 times with replacement
def outcomes : List (Ball × Ball × Ball) :=
  List.product (List.product [Ball.yellow, Ball.red, Ball.white] [Ball.yellow, Ball.red, Ball.white])
               [Ball.yellow, Ball.red, Ball.white]

-- Prove the probability of the event "not all colors are the same" is 8/9
theorem probability_not_all_colors_same_is_8_over_9 : 
  (outcomes.count (λ (o : Ball × Ball × Ball), o.fst ≠ o.snd ∨ o.snd ≠ o.snd.snd ∨ o.fst ≠ o.snd.snd)).toRat / outcomes.length.toRat = 8 / 9 :=
by
  sorry

end probability_not_all_colors_same_is_8_over_9_l460_460094


namespace a_value_is_one_l460_460016

theorem a_value_is_one (a b c d : ℕ)
  (h1 : b = 2 * a ^ 2)
  (h2 : c = 2 * b ^ 2)
  (h3 : d = 2 * c ^ 2)
  (h4 : concatDigits a (concatDigits b c) = d)
  (h5 : a > 0) :
  a = 1 :=
sorry

end a_value_is_one_l460_460016


namespace range_of_a_l460_460229

theorem range_of_a (theta : ℝ) (a : ℝ) (h1 : π / 4 < theta) (h2 : theta < π / 2) :
    (x : ℝ) (h_eq : x^2 + 4 * x * Real.sin theta + a * Real.tan theta = 0) (roots_eq : (x - x1)^2 = 0)
    (x : ℝ) : 0 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l460_460229


namespace intersection_of_curve_and_line_l460_460585

noncomputable def f : ℝ → ℝ := λ x, x^3 + 2 * x + 1
noncomputable def l : ℝ → ℝ := λ x, 3 * x + 1

theorem intersection_of_curve_and_line :
  ∃ A B C : ℝ × ℝ, 
      A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
      (A.2 = f A.1) ∧ (B.2 = f B.1) ∧ (C.2 = f C.1) ∧ 
      (A.2 = l A.1) ∧ (B.2 = l B.1) ∧ (C.2 = l C.1) ∧
      (real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = real.sqrt 10) ∧
      (real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = real.sqrt 10) :=
by sorry

end intersection_of_curve_and_line_l460_460585


namespace incorrect_conclusion_l460_460955

noncomputable def quadratic (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 2) * x + 2

theorem incorrect_conclusion (m : ℝ) (hx : m - 2 = 0) :
  ¬(∀ x : ℝ, quadratic m x = 2 ↔ x = 2) :=
by
  sorry

end incorrect_conclusion_l460_460955


namespace sum_a_1_4_7_seq_l460_460974

theorem sum_a_1_4_7_seq (n : ℕ) : 
  ∑ i in range (n + 1), 2 * 3^(3 * i) = (27^(n + 1) - 1) / 13 :=
sorry

end sum_a_1_4_7_seq_l460_460974


namespace largest_square_in_rectangle_outside_triangles_l460_460303

-- Definitions for the given conditions
def Rectangle := {width : ℝ, height : ℝ}
def Triangle := {side_length : ℝ}
def Square := {side_length : ℝ}

-- Main statement of the proof problem
theorem largest_square_in_rectangle_outside_triangles :
  ∀ (r : Rectangle), r.width = 20 ∧ r.height = 15 →
  ∀ (t1 t2 : Triangle), t1.side_length = t2.side_length →
  ∃ s : Square, s.side_length = 5 :=
by
  intro r hr
  intro t1 t2 ht
  use ⟨5⟩
  sorry

end largest_square_in_rectangle_outside_triangles_l460_460303


namespace bn_general_term_sn_sum_l460_460582

-- Given conditions
variable {b : ℕ → ℕ}
variable {a : ℕ → ℕ}
variable (hn_inc : ∀ n, b n < b (n+1))
variable (hb3_hb8 : b 3 + b 8 = 26)
variable (hb5_hb6 : b 5 * b 6 = 168)
variable (h_relation : ∀ n, (∑ i in Finset.range (n+1), 2^(i+1) * a (i + 1) = 2^(b n)))

-- Sequence 1 general term proof
theorem bn_general_term : ∀ (n : ℕ), b n = 2 * n + 2 :=
sorry

-- Sequence 2 sum proof
theorem sn_sum : ∀ (n : ℕ), (∑ i in Finset.range (n+1), a (i + 1)) = 3 * 2^(n+1) - 4 :=
sorry

end bn_general_term_sn_sum_l460_460582


namespace cyclic_quadrilateral_tangents_cyclic_l460_460445

   theorem cyclic_quadrilateral_tangents_cyclic (A B C D O : Type) [circle A B C D]
     (h1 : A, B, C, D lie on circumcircle O)
     (h2 : diagonals of ABCD are perpendicular):
     quadrilateral formed by tangents at A, B, C, D is cyclic := sorry
   
end cyclic_quadrilateral_tangents_cyclic_l460_460445


namespace evaluate_operation_l460_460552

def operation (x : ℝ) : ℝ := 9 - x

theorem evaluate_operation : operation (operation 15) = 15 :=
by
  -- Proof would go here
  sorry

end evaluate_operation_l460_460552


namespace total_amount_is_47_69_l460_460741

noncomputable def Mell_order_cost : ℝ :=
  2 * 4 + 7

noncomputable def friend_order_cost : ℝ :=
  2 * 4 + 7 + 3

noncomputable def total_cost_before_discount : ℝ :=
  Mell_order_cost + 2 * friend_order_cost

noncomputable def discount : ℝ :=
  0.15 * total_cost_before_discount

noncomputable def total_after_discount : ℝ :=
  total_cost_before_discount - discount

noncomputable def sales_tax : ℝ :=
  0.10 * total_after_discount

noncomputable def total_to_pay : ℝ :=
  total_after_discount + sales_tax

theorem total_amount_is_47_69 : total_to_pay = 47.69 :=
by
  sorry

end total_amount_is_47_69_l460_460741


namespace triangle_side_relation_l460_460212

variable (A B C O : Type) [Point A] [Point B] [Point C] [Point O]
variable (α : Type) [Angle α]
variable (triABC : Triangle A B C)
variable (interior_point : IsInteriorPoint O triABC)
variable (equal_angles : ∀ {P Q R S : Point} (h : P ≠ Q ∧ Q ≠ R ∧ S ≠ O),
  Angle (P O Q) = α ∧ Angle (Q O R) = α ∧ Angle (R O S) = α ∧ Angle (S O P) = α)

theorem triangle_side_relation (P Q R S : Point) (h : P ≠ Q ∧ Q ≠ R ∧ S ≠ O)
  (triABC : Triangle A B C) (interior_point : IsInteriorPoint O triABC) 
  (equal_angles : ∀ {P Q R S : Point} (h : P ≠ Q ∧ Q ≠ R ∧ S ≠ O),
    Angle (P O Q) = α ∧ Angle (Q O R) = α ∧ Angle (R O S) = α ∧ Angle (S O P) = α) :
  BC^2 = AB * AC :=
sorry

end triangle_side_relation_l460_460212


namespace evaluate_expression_x_eq_3_l460_460874

theorem evaluate_expression_x_eq_3 : (3^5 - 5 * 3 + 7 * 3^3) = 417 := by
  sorry

end evaluate_expression_x_eq_3_l460_460874


namespace train_crossing_time_l460_460846

theorem train_crossing_time :
  ∀ (length speed : ℚ),
    length = 240 ∧ speed = 54 →
    (length / (speed * (1000 / 3600))) = 16 :=
by 
  intros length speed h,
  cases h with h_length h_speed,
  rw [h_length, h_speed],
  norm_num,
  sorry

end train_crossing_time_l460_460846


namespace smallest_possible_AAB_l460_460490

-- Definitions of the digits A and B
def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

-- Definition of the condition AB equals 1/7 of AAB
def condition (A B : ℕ) : Prop := 10 * A + B = (1 / 7) * (110 * A + B)

theorem smallest_possible_AAB (A B : ℕ) : is_valid_digit A ∧ is_valid_digit B ∧ condition A B → 110 * A + B = 664 := sorry

end smallest_possible_AAB_l460_460490


namespace option_A_is_incorrect_option_B_is_correct_option_C_is_correct_l460_460102

def P : ℕ → ℚ 
| 1 := 1 / 4
| n := if P (n - 1) = 1 / 3 then 1 / 3 else 1 / 2

def P2 := (1 / 4) * (1 / 3) + (3 / 4) * (1 / 2)

theorem option_A_is_incorrect : P 2 = 11 / 24 := 
by {
  sorry
}

noncomputable def seq_Pn_minus_3_7 (n : ℕ) : ℚ := 
  P n - 3 / 7

theorem option_B_is_correct : ∀ n : ℕ, seq_Pn_minus_3_7 n = (-1 / 6) * seq_Pn_minus_3_7 (n - 1) := 
by {
  sorry
}

theorem option_C_is_correct : ∀ n : ℕ, P n ≤ 11 / 24 := 
by {
  sorry
}

end option_A_is_incorrect_option_B_is_correct_option_C_is_correct_l460_460102


namespace punitive_damages_multiple_l460_460311

theorem punitive_damages_multiple :
  (let salary_loss := 30 * 50000,
       medical_bills := 200000,
       total_damages := salary_loss + medical_bills,
       requested_total := total_damages * (1 + x),
       actual_received := 0.80 * requested_total
   in actual_received = 5440000 → x = 3) :=
sorry

end punitive_damages_multiple_l460_460311


namespace distinct_arith_prog_triangles_l460_460257

theorem distinct_arith_prog_triangles (n : ℕ) (h10 : n % 10 = 0) : 
  (3 * n = 180 → ∃ d : ℕ, ∀ a b c, a = n - d ∧ b = n ∧ c = n + d 
  →  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d < 60) :=
by
  sorry

end distinct_arith_prog_triangles_l460_460257


namespace triangle_formation_probability_l460_460125
open Classical

noncomputable def probability_triangle_formation : ℝ :=
  ∫ x in 0..1, ((x / (2 - x)) / (2 - x)) - ((1 - x) / (2 - x)) dx

theorem triangle_formation_probability :
  probability_triangle_formation = 2 * Real.log 2 - 1 :=
begin
  sorry
end

end triangle_formation_probability_l460_460125


namespace problem_correct_answer_l460_460429

-- Define the quadratic equations
def eq_A : Polynomial ℝ := Polynomial.C 1 + Polynomial.X ^ 2
def eq_B : Polynomial ℝ := Polynomial.C 1 + Polynomial.C (-2) * Polynomial.X + Polynomial.X ^ 2
def eq_C : Polynomial ℝ := Polynomial.C 1 + Polynomial.C 1 * Polynomial.X + Polynomial.X ^ 2
def eq_D : Polynomial ℝ := Polynomial.C (-2) + Polynomial.X ^ 2

open Polynomial

-- Define the condition for two distinct real roots using the discriminant
def has_two_distinct_real_roots (p : Polynomial ℝ) : Prop :=
  let a := p.coeff 2
  let b := p.coeff 1
  let c := p.coeff 0
  (b ^ 2 - 4 * a * c) > 0

-- The problem statement to be proved
theorem problem_correct_answer :
  has_two_distinct_real_roots eq_A = False ∧
  has_two_distinct_real_roots eq_B = False ∧
  has_two_distinct_real_roots eq_C = False ∧
  has_two_distinct_real_roots eq_D = True :=
sorry

end problem_correct_answer_l460_460429


namespace length_BC_l460_460973

-- Definition of lengths as provided in the problem statement
def AD : ℝ := 50
def CD : ℝ := 25
def AC : ℝ := 20

-- Hypothesis: Triangle ABD is a right triangle at D
def right_triangle_ABD : Prop :=
  AD^2 = AC^2 + CD^2

-- Calculating length AB using Pythagorean theorem in Triangle ABD
def AB : ℝ := real.sqrt (AD^2 - (CD + AC)^2)

-- Proof statement for length of segment BC in the right triangle ABC
theorem length_BC : real.sqrt (AB^2 + AC^2) = 5 * real.sqrt 35 := by
  sorry

end length_BC_l460_460973


namespace range_of_x_l460_460191

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 + (a - 4) * x + 4 - 2 * a

theorem range_of_x (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  ∀ x : ℝ, (f x a > 0) ↔ (x < 1 ∨ x > 3) :=
by
  intro x
  sorry

end range_of_x_l460_460191


namespace range_of_m_l460_460276

noncomputable def f (m x : ℝ) : ℝ := m * Real.sin(x + Real.pi / 4) - Real.sqrt 2 * Real.sin x

theorem range_of_m (m : ℝ) (hx : 0 < x ∧ x < 7 * Real.pi / 6) :
  (∃ a b : ℝ, ∃ x₁ x₂, a ≤ f m x₁ ∧ f m x₁ ≤ b ∧ a ≤ f m x₂ ∧ f m x₂ ≤ b) ↔ 
  2 < m ∧ m < 3 + Real.sqrt 3 := sorry

end range_of_m_l460_460276


namespace reciprocals_sum_equal_l460_460725

variable {p q r : ℝ}
variable (h : Polynomial.root_set (Polynomial.C 1 * Polynomial.X ^ 3 - 2 * Polynomial.C 1 * Polynomial.X ^ 2 + Polynomial.C 1 * Polynomial.X - Polynomial.C 1) {p, q, r})

theorem reciprocals_sum_equal :
  (\frac{1}{p + 2} + \frac{1}{q + 2} + \frac{1}{r + 2} = \frac{20}{19}) :=
by
  sorry

end reciprocals_sum_equal_l460_460725


namespace number_of_valid_n_digit_numbers_l460_460979

theorem number_of_valid_n_digit_numbers (n : ℕ) (h : n > 0) :
  (∃ (digits : Fin n → ℕ), (∀ i, 1 ≤ digits i ∧ digits i ≤ 9) ∧ (Finset.univ.sum digits = 9 * n - 8)) ↔
  finset.card ((finset.range (9 + 1)).product (finset.range (9 + 1))^n) = nat.choose (n + 7) (n - 1) :=
sorry

end number_of_valid_n_digit_numbers_l460_460979


namespace trade_in_value_of_old_phone_l460_460156

-- Define the given conditions
def cost_of_iphone : ℕ := 800
def earnings_per_week : ℕ := 80
def weeks_worked : ℕ := 7

-- Define the total earnings from babysitting
def total_earnings : ℕ := earnings_per_week * weeks_worked

-- Define the final proof statement
theorem trade_in_value_of_old_phone : cost_of_iphone - total_earnings = 240 :=
by
  unfold cost_of_iphone
  unfold total_earnings
  -- Substitute in the values
  have h1 : 800 - (80 * 7) = 240 := sorry
  exact h1

end trade_in_value_of_old_phone_l460_460156


namespace expression_evaluation_l460_460187

noncomputable def x : ℝ := (Real.sqrt 1.21) ^ 3
noncomputable def y : ℝ := (Real.sqrt 0.81) ^ 2
noncomputable def a : ℝ := 4 * Real.sqrt 0.81
noncomputable def b : ℝ := 2 * Real.sqrt 0.49
noncomputable def c : ℝ := 3 * Real.sqrt 1.21
noncomputable def d : ℝ := 2 * Real.sqrt 0.49
noncomputable def e : ℝ := (Real.sqrt 0.81) ^ 4

theorem expression_evaluation : ((x / Real.sqrt y) - (Real.sqrt a / b^2) + ((Real.sqrt c / Real.sqrt d) / (3 * e))) = 1.291343 := by 
  sorry

end expression_evaluation_l460_460187


namespace find_chemistry_marks_l460_460164

theorem find_chemistry_marks
  (english_marks : ℕ) (math_marks : ℕ) (physics_marks : ℕ) (biology_marks : ℕ) (average_marks : ℕ) (chemistry_marks : ℕ) :
  english_marks = 86 → math_marks = 89 → physics_marks = 82 → biology_marks = 81 → average_marks = 85 →
  chemistry_marks = 425 - (english_marks + math_marks + physics_marks + biology_marks) →
  chemistry_marks = 87 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h6
  have total_marks := 425 - (86 + 89 + 82 + 81)
  norm_num at total_marks
  exact h6

end find_chemistry_marks_l460_460164


namespace part_a_part_b_l460_460090

noncomputable def exists_2012_points_on_unit_circle_with_rational_distances : Prop :=
  ∃ (S : Finset ℂ) (h_card : S.card = 2012), (∀ (x y : ℂ) (hx : x ∈ S) (hy : y ∈ S), abs x = 1 ∧ abs y = 1 ∧ (x ≠ y → ∃ (q : ℚ), dist x y = q))

noncomputable def exists_infinite_points_on_unit_circle_with_rational_distances : Prop :=
  ∃ (S : Set ℂ) (h_inf : S.infinite), (∀ (x y : ℂ) (hx : x ∈ S) (hy : y ∈ S), abs x = 1 ∧ abs y = 1 ∧ (x ≠ y → ∃ (q : ℚ), dist x y = q))

-- Parts (a) and (b) defined as Lean propositions
theorem part_a : exists_2012_points_on_unit_circle_with_rational_distances :=
sorry

theorem part_b : exists_infinite_points_on_unit_circle_with_rational_distances :=
sorry

end part_a_part_b_l460_460090


namespace ellipse_standard_eq_line_eq_min_lambda_l460_460937

-- Definition of the ellipse with given parameters and conditions
def ellipse_standard_eq_of_conditions (a b : ℝ) (h1 : a > b) (h2 : b > 0) (P : ℝ × ℝ) (hP : P = (sqrt(6), -1)) (h_area : ∃ F1 F2 : ℝ × ℝ, area_triangle P F1 F2 = 2) : Prop :=
  ∃ a b, (a = 2 * sqrt(2)) ∧ (b = 2) ∧ (a ^ 2 - b ^ 2 = 4) ∧ (1 / a ^ 2 + 1 / b ^ 2 = 1)

-- Equation of ellipse C
theorem ellipse_standard_eq : ellipse_standard_eq_of_conditions :=
by {
  -- Proof goes here
  sorry
}

-- Definition of the line intersecting a circle and ellipse with given conditions
def line_eq_min_lambda (a b : ℝ) (h1 : a > b) (h2 : b > 0) (P : ℝ × ℝ) (hP : P = (sqrt(6), -1))
  (h_area : ∃ F1 F2 : ℝ × ℝ, area_triangle P F1 F2 = 2) (λ : ℝ) : Prop :=
  line_eq_min_lambda (sqrt(2)) C λ = y = x

-- Intersection points and minimum lambda condition
theorem line_eq_min_lambda : line_eq_min_lambda :=
by {
  -- Proof goes here
  sorry
}

end ellipse_standard_eq_line_eq_min_lambda_l460_460937


namespace count_valid_a_in_system_number_of_valid_a_l460_460193

theorem count_valid_a_in_system (a : ℤ) (ha : |a| ≤ 2005) :
  (∃ x y : ℤ, x^2 = y + a ∧ y^2 = x + a) ↔ a ∈ (SetOf (λ k, ∃ k : ℤ, a = (k^2 - 1) / 4 ∧ k % 2 ≡ 1 [MOD 2])) :=
sorry

theorem number_of_valid_a : 
  (∃ n : ℕ, ∀ a : ℤ, |a| ≤ 2005 -> 
  (∃ x y : ℤ, x^2 = y + a ∧ y^2 = x + a) -> n = 90 ) :=
sorry

end count_valid_a_in_system_number_of_valid_a_l460_460193


namespace circle_diameter_of_circumscribed_square_l460_460844

theorem circle_diameter_of_circumscribed_square (r : ℝ) (s : ℝ) (h1 : s = 2 * r) (h2 : 4 * s = π * r^2) : 2 * r = 16 / π := by
  sorry

end circle_diameter_of_circumscribed_square_l460_460844


namespace dirichlet_function_incorrect_statement_l460_460921

def D (x : ℝ) : ℝ := if x ∈ ℚ then 1 else 0

theorem dirichlet_function_incorrect_statement :
  ¬ ∃ x : ℝ, irrational x ∧ D (x + 1) = D x + 1 := by 
  sorry

end dirichlet_function_incorrect_statement_l460_460921


namespace geometric_sequence_properties_l460_460670

noncomputable def a_1 (a_n : ℕ → ℝ) : ℝ := a_n 1

noncomputable def a_n (n : ℕ) : ℝ := (a_1 id) * (3 ^ (n - 1))

theorem geometric_sequence_properties (a_n : ℕ → ℝ) 
  (h1 : a_n 2 - a_n 1 = 2)
  (h2 : 2 * a_n 2 = (3 * a_n 1 + a_n 3) / 2) :
  a_n 1 = 1 ∧ (a_n 2 / a_n 1 = 3) ∧ (∀ n, sum_of_first_n_terms n a_n = (3^n - 1) / 2) :=
by
  sorry

end geometric_sequence_properties_l460_460670


namespace time_addition_sum_l460_460680

theorem time_addition_sum :
  let initial_hours := 15
  let initial_minutes := 0
  let initial_seconds := 0
  let added_hours := 317
  let added_minutes := 58
  let added_seconds := 33
  let total_initial_seconds := initial_hours * 3600 + initial_minutes * 60 + initial_seconds
  let total_added_seconds := added_hours * 3600 + added_minutes * 60 + added_seconds
  let total_seconds := total_initial_seconds + total_added_seconds
  let final_seconds := total_seconds % 43200 -- 43200 seconds in 12 hours
  let final_hours := (final_seconds / 3600) % 12
  let final_minutes := (final_seconds % 3600) / 60
  let final_secs := final_seconds % 60
  final_hours + final_minutes + final_secs = 99 :=
by {
  let initial_hours := 15 in
  let initial_minutes := 0 in
  let initial_seconds := 0 in
  let added_hours := 317 in
  let added_minutes := 58 in
  let added_seconds := 33 in
  let total_initial_seconds := initial_hours * 3600 + initial_minutes * 60 + initial_seconds in
  let total_added_seconds := added_hours * 3600 + added_minutes * 60 + added_seconds in
  let total_seconds := total_initial_seconds + total_added_seconds in
  let final_seconds := total_seconds % 43200 in
  let final_hours := (final_seconds / 3600) % 12 in
  let final_minutes := (final_seconds % 3600) / 60 in
  let final_secs := final_seconds % 60 in
  let sum := final_hours + final_minutes + final_secs in
  show final_hours + final_minutes + final_secs = 99, from sorry
}

end time_addition_sum_l460_460680


namespace chairs_per_row_l460_460807

noncomputable def initial_water_in_gallons : ℕ := 3
noncomputable def dixie_cup_capacity_in_ounces : ℕ := 6
noncomputable def number_of_rows : ℕ := 5
noncomputable def water_left_in_cooler_in_ounces : ℕ := 84

theorem chairs_per_row :
  let initial_water_in_ounces := initial_water_in_gallons * 128 in
  let water_used := initial_water_in_ounces - water_left_in_cooler_in_ounces in
  let total_cups_filled := water_used / dixie_cup_capacity_in_ounces in
  total_cups_filled / number_of_rows = 10 := 
by
  sorry

end chairs_per_row_l460_460807


namespace solution_set_f_l460_460234

noncomputable def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem solution_set_f (x : ℝ)  : (f(x) ≥ x^2 - 8x + 15) ↔ (5 - real.sqrt 3 ≤ x ∧ x ≤ 6) :=
sorry

end solution_set_f_l460_460234


namespace cos_identity_l460_460945

theorem cos_identity (α : ℝ) (h : Real.cos (π / 6 + α) = sqrt 3 / 3) : 
  Real.cos (5 * π / 6 - α) = - (sqrt 3 / 3) :=
by
  sorry

end cos_identity_l460_460945


namespace quadratic_root_value_l460_460218

theorem quadratic_root_value
  (a : ℝ) 
  (h : a^2 + 3 * a - 1010 = 0) :
  2 * a^2 + 6 * a + 4 = 2024 :=
by
  sorry

end quadratic_root_value_l460_460218


namespace Apollonian_circle_l460_460658

theorem Apollonian_circle
  (a k : ℝ) (A B M : ℝ × ℝ)
  (hA : A = (-a, 0))
  (hB : B = (a, 0))
  (hM : ∃ (x y : ℝ), M = (x, y) ∧ (dist (x + a, y) / dist (x - a, y) = k)) :
  ∃ (C : ℝ × ℝ) (r : ℝ),
    C = (a * (1 + k^2) / (1 - k^2), 0) ∧
    r = 2 * a * k / |1 - k^2| ∧
    ∀ (x y : ℝ), M = (x, y) → (x + a * (1 + k^2) / (1 - k^2))^2 + y^2 = (2 * a * k / |1 - k^2|)^2 :=
by
  -- the proof will go here
  sorry

end Apollonian_circle_l460_460658


namespace smallest_number_satisfying_conditions_l460_460034

theorem smallest_number_satisfying_conditions :
  ∀ n : ℕ,
    (∃ a b c d e : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
      n = a^4 + b^4 + c^4 + d^4 + e^4) ∧
    (∃ m : ℕ, n = m + (m+1) + (m+2) + (m+3) + (m+4) + (m+5)) →
    n ≥ 2019 ∧ (n = 2019 → (a, b, c, d, e) = (1, 2, 3, 5, 6) ∧ (ℕ.exists (λ m, n = m+(m+1)+(m+2)+(m+3)+(m+4)+(m+5)) = 334)) :=
by
  sorry

end smallest_number_satisfying_conditions_l460_460034


namespace planted_fraction_correct_l460_460174

noncomputable def field_planted_fraction (leg1 leg2 : ℕ) (square_distance : ℕ) : ℚ :=
  let hypotenuse := Real.sqrt (leg1^2 + leg2^2)
  let total_area := (leg1 * leg2) / 2
  let square_side := square_distance
  let square_area := square_side^2
  let planted_area := total_area - square_area
  planted_area / total_area

theorem planted_fraction_correct :
  field_planted_fraction 5 12 4 = 367 / 375 :=
by
  sorry

end planted_fraction_correct_l460_460174


namespace region_area_0_le_y_lt_1_l460_460904

noncomputable def fractional_part (y : ℝ) : ℝ :=
  y - floor y

noncomputable def region_area : ℝ :=
  ∑ k in finset.range 50, 0.02 * (k + 2)

theorem region_area_0_le_y_lt_1 :
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ 50 * fractional_part y ≥ floor x + floor y) →
  region_area = 25.48 :=
by
  sorry

end region_area_0_le_y_lt_1_l460_460904


namespace work_completed_in_5_days_l460_460442

-- Define the rates of work for A, B, and C
def rateA : ℚ := 1 / 15
def rateB : ℚ := 1 / 14
def rateC : ℚ := 1 / 16

-- Summing their rates to get the combined rate
def combined_rate : ℚ := rateA + rateB + rateC

-- This is the statement we need to prove, i.e., the time required for A, B, and C to finish the work together is 5 days.
theorem work_completed_in_5_days (hA : rateA = 1 / 15) (hB : rateB = 1 / 14) (hC : rateC = 1 / 16) :
  (1 / combined_rate) = 5 :=
by
  sorry

end work_completed_in_5_days_l460_460442


namespace remainder_is_576_l460_460704

-- Definitions based on the conditions
def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def num_functions (f : ℕ → ℕ) (A : Set ℕ) : ℕ :=
  if ∃ c ∈ A, ∀ x ∈ A, f (f x) = c then 1 else 0

-- Define the total number of such functions
def total_functions (A : Set ℕ) : ℕ :=
  8 * (∑ k in ({1, 2, 3, 4, 5, 6, 7} : Set ℕ), Nat.choose 7 k * (k ^ (7 - k)))

-- The main theorem to prove
theorem remainder_is_576 : (total_functions A % 1000) = 576 :=
  by
    sorry

end remainder_is_576_l460_460704


namespace intersecting_circles_l460_460591

theorem intersecting_circles (m c : ℝ)
  (h1 : ∃ (x1 y1 x2 y2 : ℝ), x1 = 1 ∧ y1 = 3 ∧ x2 = m ∧ y2 = 1 ∧ x1 ≠ x2 ∧ y1 ≠ y2)
  (h2 : ∀ (x y : ℝ), (x - y + (c / 2) = 0) → (x = 1 ∨ y = 3)) :
  m + c = 3 :=
sorry

end intersecting_circles_l460_460591


namespace part1_part2_part3_l460_460604

-- Define the necessary constants and functions as per conditions
variable (a : ℝ) (f : ℝ → ℝ)
variable (hpos : a > 0) (hfa : f a = 1)

-- Conditions based on the problem statement
variable (hodd : ∀ x, f (-x) = -f x)
variable (hfe : ∀ x1 x2, f (x1 - x2) = (f x1 * f x2 + 1) / (f x2 - f x1))

-- 1. Prove that f(2a) = 0
theorem part1  : f (2 * a) = 0 := sorry

-- 2. Prove that there exists a constant T > 0 such that f(x + T) = f(x)
theorem part2 : ∃ T > 0, ∀ x, f (x + 4 * a) = f x := sorry

-- 3. Prove f(x) is decreasing on (0, 4a) given x ∈ (0, 2a) implies f(x) > 0
theorem part3 (hx_correct : ∀ x, 0 < x ∧ x < 2 * a → 0 < f x) :
  ∀ x1 x2, 0 < x2 ∧ x2  < x1 ∧ x1 < 4 * a → f x2 > f x1 := sorry

end part1_part2_part3_l460_460604


namespace even_number_of_odd_handshakers_l460_460171

theorem even_number_of_odd_handshakers (V : Type) [Fintype V] (G : SimpleGraph V) :
  (∃ (n : ℕ), Fintype.card { v : V | G.degree v % 2 = 1 } = 2 * n) :=
by
  sorry

end even_number_of_odd_handshakers_l460_460171


namespace initial_observations_l460_460023

theorem initial_observations (n : ℕ) (S : ℕ) (new_obs : ℕ) :
  (S = 12 * n) → (new_obs = 5) → (S + new_obs = 11 * (n + 1)) → n = 6 :=
by
  intro h1 h2 h3
  sorry

end initial_observations_l460_460023


namespace minimum_value_l460_460329

open Real

-- Given the conditions
variables (a b c k : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k)

-- The theorem
theorem minimum_value (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < k) : 
  ∃ x, x = (3 : ℝ) / k ∧ ∀ y, y = (a / (k * b) + b / (k * c) + c / (k * a)) → y ≥ x :=
sorry

end minimum_value_l460_460329


namespace larger_pile_toys_l460_460280

-- Define the conditions
def total_toys (small_pile large_pile : ℕ) : Prop := small_pile + large_pile = 120
def larger_pile (small_pile large_pile : ℕ) : Prop := large_pile = 2 * small_pile

-- Define the proof problem
theorem larger_pile_toys (small_pile large_pile : ℕ) (h1 : total_toys small_pile large_pile) (h2 : larger_pile small_pile large_pile) : 
  large_pile = 80 := by
  sorry

end larger_pile_toys_l460_460280


namespace recurring03_m_mult_recurring8_l460_460901

noncomputable def recurring03_fraction : ℚ :=
let x := 0.03.represented_by_recurring 100 in
(100 * x - x / 99)

noncomputable def recurring8_fraction : ℚ :=
let y := 0.8.represented_by_recurring 10 in
(10 * y - y / 9)

theorem recurring03_m_mult_recurring8 : recurring03_fraction * recurring8_fraction = 8 / 297 := 
by
  sorry

end recurring03_m_mult_recurring8_l460_460901


namespace num_ordered_pairs_xy_64_l460_460258

theorem num_ordered_pairs_xy_64 : 
  {p : ℕ × ℕ // 0 < p.1 ∧ 0 < p.2 ∧ p.1 * p.2 = 64}.card = 7 :=
sorry

end num_ordered_pairs_xy_64_l460_460258


namespace volume_tetrahedron_ABCD_l460_460674

noncomputable def volume_of_tetrahedron (AB CD d angle : ℝ) : ℝ :=
  (1 / 3) * (1 / 2) * (d * sin angle * CD) * AB

theorem volume_tetrahedron_ABCD :
  volume_of_tetrahedron 1 (sqrt 3) 2 (π / 3) = 1 / 2 := by
  sorry

end volume_tetrahedron_ABCD_l460_460674


namespace fixed_point_exists_l460_460029

theorem fixed_point_exists :
  ∃ (x y : ℝ), (∀ k : ℝ, (2 * k + 1) * x + (k - 1) * y + (7 - k) = 0) ∧ 
  x = -2 ∧ y = 5 :=
by
  use -2, 5
  split
  { intro k
    linarith }
  split
  { rfl }
  { rfl }

end fixed_point_exists_l460_460029


namespace exceeded_by_600_l460_460005

noncomputable def ken_collected : ℕ := 600
noncomputable def mary_collected (ken : ℕ) : ℕ := 5 * ken
noncomputable def scott_collected (mary : ℕ) : ℕ := mary / 3
noncomputable def total_collected (ken mary scott : ℕ) : ℕ := ken + mary + scott
noncomputable def goal : ℕ := 4000
noncomputable def exceeded_goal (total goal : ℕ) : ℕ := total - goal

theorem exceeded_by_600 : exceeded_goal (total_collected ken_collected (mary_collected ken_collected) (scott_collected (mary_collected ken_collected))) goal = 600 := by
  sorry

end exceeded_by_600_l460_460005


namespace bullet_speed_difference_l460_460438

theorem bullet_speed_difference (speed_horse speed_bullet : ℕ) 
    (h_horse : speed_horse = 20) (h_bullet : speed_bullet = 400) :
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    speed_same_direction - speed_opposite_direction = 40 :=
    by
    -- Define the speeds in terms of the given conditions.
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    -- State the equality to prove.
    show speed_same_direction - speed_opposite_direction = 40;
    -- Proof (skipped here).
    -- sorry is used to denote where the formal proof steps would go.
    sorry

end bullet_speed_difference_l460_460438


namespace num_of_fractions_is_4_l460_460134

def is_fraction (e : ℚ) : Prop := 
  ∃ (a b : ℤ), b ≠ 0 ∧ e = (a : ℚ) / (b : ℚ)

def expr1 := -3 * x
def expr2 := (x + y) / (x - y)
def expr3 := (x * y - y) / 3
def expr4 := -(3 : ℚ) / 10
def expr5 := 2 / (5 + y)
def expr6 := x / (4 * x * y)

theorem num_of_fractions_is_4 (x y : ℚ) : 
  (is_fraction expr2 ∧ is_fraction expr3 ∧ is_fraction expr4 ∧ is_fraction expr5 ∧ is_fraction expr6) 
  → ¬ is_fraction expr1 
  → (4 = 4) := 
by
  intro h,
  sorry

end num_of_fractions_is_4_l460_460134


namespace circumference_tank_C_eq_8_sqrt_5_l460_460377

theorem circumference_tank_C_eq_8_sqrt_5 :
  ∀ (r_C r_B : ℝ),
    (2 * π * r_B = 10) /\
    (π * r_C^2 * 10 = 0.8 * π * r_B^2 * 8) →
    (2 * π * r_C = 8 * Real.sqrt 5) :=
by
  intros r_C r_B hc
  cases hc with h_circ_b h_vol
  sorry

end circumference_tank_C_eq_8_sqrt_5_l460_460377


namespace solve_equation_l460_460178

theorem solve_equation (x : ℝ) : 
  ( (2 ^ (3 * x) + 3 ^ (3 * x)) / (2 ^ (2 * x) * 3 ^ x + 2 ^ x * 3 ^ (2 * x)) = 11 / 6 ) ↔ 
  (x = - real.log (6) / real.log (2)) ∨ (x = real.log (1 / 6) / real.log (2 / 3)) :=
by 
  sorry

end solve_equation_l460_460178


namespace centroid_value_l460_460797

-- Define the points P, Q, R
def P : ℝ × ℝ := (4, 3)
def Q : ℝ × ℝ := (-1, 6)
def R : ℝ × ℝ := (7, -2)

-- Define the coordinates of the centroid S
noncomputable def S : ℝ × ℝ := 
  ( (4 + (-1) + 7) / 3, (3 + 6 + (-2)) / 3 )

-- Statement to prove
theorem centroid_value : 
  let x := (4 + (-1) + 7) / 3
  let y := (3 + 6 + (-2)) / 3
  8 * x + 3 * y = 101 / 3 :=
by
  let x := (4 + (-1) + 7) / 3
  let y := (3 + 6 + (-2)) / 3
  have h: 8 * x + 3 * y = 101 / 3 := sorry
  exact h

end centroid_value_l460_460797


namespace factorize_expression_l460_460532

variable (a b : ℝ)

theorem factorize_expression : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l460_460532


namespace digit_5_equals_digit_2_in_1_to_688_l460_460098

-- Define a function to count the occurrences of a specific digit in a given range
def count_digit_occurrences (digit : ℕ) (start : ℕ) (end : ℕ) : ℕ :=
  ((start to end).flatMap (λ n, n.digits 10)).count (λ d, d = digit)

theorem digit_5_equals_digit_2_in_1_to_688 :
  count_digit_occurrences 5 1 688 = count_digit_occurrences 2 1 688 :=
by
  sorry

end digit_5_equals_digit_2_in_1_to_688_l460_460098


namespace curves_intersection_angle_and_area_l460_460147

noncomputable def intersection_angle : ℝ := 70 + 53/60 + 37/3600  -- in degrees
def intersection_area : ℝ := 19.06  -- area in given units

theorem curves_intersection_angle_and_area :
  (∃ x y : ℝ, x^2 + y^2 = 16 ∧ y^2 = 6 * x ∧
  ∃ α : ℝ, α = intersection_angle) ∧
  (∃ A : ℝ, A = intersection_area) :=
by
  split;
  { sorry }

end curves_intersection_angle_and_area_l460_460147


namespace probability_all_operating_probability_shutdown_l460_460738

-- Define the events and their probabilities
def P_A : ℝ := 0.9
def P_B : ℝ := 0.8
def P_C : ℝ := 0.85

-- Prove that the probability of all three machines operating without supervision is 0.612
theorem probability_all_operating : P_A * P_B * P_C = 0.612 := 
by sorry

-- Prove that the probability of a shutdown is 0.059
theorem probability_shutdown :
    P_A * (1 - P_B) * (1 - P_C) +
    (1 - P_A) * P_B * (1 - P_C) +
    (1 - P_A) * (1 - P_B) * P_C +
    (1 - P_A) * (1 - P_B) * (1 - P_C) = 0.059 :=
by sorry

end probability_all_operating_probability_shutdown_l460_460738


namespace factorize_expression_l460_460531

variable (a b : ℝ)

theorem factorize_expression : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l460_460531


namespace cookies_thrown_on_floor_l460_460866

theorem cookies_thrown_on_floor (initialAlice : ℕ) (initialBob : ℕ) (moreAlice : ℕ) (moreBob : ℕ) 
  (finalCookies : ℕ) (totalInitial : initialAlice + initialBob = 81) (totalMore : moreAlice + moreBob = 41) 
  (totalFinal : initialAlice + initialBob + moreAlice + moreBob = 122) : initialAlice + initialBob + moreAlice + moreBob - finalCookies = 29 :=
by
  -- assumption from conditions
  have totalInitialCookies : initialAlice + initialBob = 81 := totalInitial
  have totalMoreCookies : moreAlice + moreBob = 41 := totalMore
  have totalAfterBakingMore := totalInitialCookies + totalMoreCookies
  have finalTotalCookies := 122
  apply totalAfterBakingMore
  exact totalFinal

  -- declaration for cookies thrown on floor
  have thrownOnFloor := totalAfterBakingMore - finalCookies
  have result := thrownOnFloor = 29
  exact result

end cookies_thrown_on_floor_l460_460866


namespace max_sum_of_products_l460_460977

-- Define the four variables
variables {f g h j : ℕ}

-- Define the condition that f, g, h, j must be distinct and each is one of {3, 4, 5, 6}
def valid_vars (f g h j : ℕ) : Prop :=
  {f, g, h, j} = {3, 4, 5, 6}

-- Define the main statement to prove
theorem max_sum_of_products :
  ∀ (f g h j : ℕ), valid_vars f g h j → 
    (fg + gh + hj + fj) ≤ 81 :=
by
  sorry

end max_sum_of_products_l460_460977


namespace highest_number_written_on_papers_l460_460278

theorem highest_number_written_on_papers :
  ∃ n : ℕ, (∃ p : (1:ℚ) / p = 0.010416666666666666) ∧ (p = 1 / n) ∧ n = 96 :=
by sorry

end highest_number_written_on_papers_l460_460278


namespace value_of_d_l460_460640

theorem value_of_d (d : ℝ) (h : ∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) : d = 5 :=
by
  sorry

end value_of_d_l460_460640


namespace seismic_activity_mismatch_percentage_l460_460653

theorem seismic_activity_mismatch_percentage
  (total_days : ℕ)
  (quiet_days_percentage : ℝ)
  (prediction_accuracy : ℝ)
  (predicted_quiet_days_percentage : ℝ)
  (quiet_prediction_correctness : ℝ)
  (active_days_percentage : ℝ)
  (incorrect_quiet_predictions : ℝ) :
  quiet_days_percentage = 0.8 →
  predicted_quiet_days_percentage = 0.64 →
  quiet_prediction_correctness = 0.7 →
  active_days_percentage = 0.2 →
  incorrect_quiet_predictions = predicted_quiet_days_percentage - (quiet_prediction_correctness * quiet_days_percentage) →
  (incorrect_quiet_predictions / active_days_percentage) * 100 = 40 := by
  sorry

end seismic_activity_mismatch_percentage_l460_460653


namespace cab_to_bus_ratio_l460_460316

noncomputable def train_distance : ℤ := 300
noncomputable def bus_distance : ℤ := train_distance / 2
noncomputable def total_distance : ℤ := 500
noncomputable def cab_distance : ℤ := total_distance - (train_distance + bus_distance)
noncomputable def ratio : ℚ := cab_distance / bus_distance

theorem cab_to_bus_ratio :
  ratio = 1 / 3 := by
  sorry

end cab_to_bus_ratio_l460_460316


namespace problem_l460_460709

def LCM_range (a b : ℕ) : ℕ :=
  Nat.lcm_list (List.range' a (b - a + 1))

def M := LCM_range 10 30

def LCM_with_list (n : ℕ) (l : List ℕ) : ℕ :=
  l.foldl Nat.lcm n

def N := LCM_with_list M [32, 33, 34, 35, 36, 37, 38, 39, 40]

theorem problem : N / M = 74 := by
  sorry

end problem_l460_460709


namespace angle_at_intersection_l460_460539

noncomputable def angle_between_curves (y₁ y₂ : ℝ → ℝ) (x₀ y₀ : ℝ) : ℝ :=
  let k₁ := deriv y₁ x₀
  let k₂ := deriv y₂ x₀
  let tan_phi := abs ((k₂ - k₁) / (1 + k₁ * k₂))
  arctan tan_phi

theorem angle_at_intersection :
  let y₁ : ℝ → ℝ := λ x => 2 * x^2
  let y₂ : ℝ → ℝ := λ x => x^3 + 2 * x^2 - 1
  let x₀ : ℝ := 1
  let y₀ : ℝ := 2
  angle_between_curves y₁ y₂ x₀ y₀ = arctan (3 / 29) :=
by sorry

end angle_at_intersection_l460_460539


namespace Grandmother_is_75_l460_460509

-- Variables for the ages of Cara, her mom, and her grandmother
variable (Cara_age Mom_age Grandmother_age : ℕ)

-- Conditions given in the problem
axiom Cara_is_40 : Cara_age = 40
axiom Cara_younger_than_Mom : Cara_age + 20 = Mom_age
axiom Mom_younger_than_Grandmother : Mom_age + 15 = Grandmother_age

-- Statement we need to prove
theorem Grandmother_is_75 : Grandmother_age = 75 :=
by 
  -- Using the given conditions
  rw [Cara_is_40] at *
  rw [Cara_younger_than_Mom, Mom_younger_than_Grandmother] at *
  sorry

end Grandmother_is_75_l460_460509


namespace monotonic_decreasing_f_symmetry_center_h_l460_460197

variable (a x : ℝ)

noncomputable def f (x a : ℝ) := x / (x - a)
noncomputable def h (x : ℝ) := (x / (x - 1)) + x^3 - 3 * x^2

-- Statement 1: Prove that f is monotonically decreasing on (1, +∞) if and only if 0 < a ≤ 1
theorem monotonic_decreasing_f : (0 < a ∧ a ≤ 1) ↔ ∀ x1 x2, 1 < x1 → x1 < x2 → f x1 a > f x2 a :=
sorry

-- Statement 2: Given a = 1, prove that the center of symmetry for h(x) is (1, -1)
theorem symmetry_center_h : h (-(x - 1)) + h (x - 1) = 2 ∧ (1, -1) =
  let m := 1; let n := -1 in (m, n) :=
sorry

end monotonic_decreasing_f_symmetry_center_h_l460_460197


namespace solve_for_g2_l460_460390

-- Let g : ℝ → ℝ be a function satisfying the given condition
variable (g : ℝ → ℝ)

-- The given condition
def condition (x : ℝ) : Prop :=
  g (2 ^ x) + x * g (2 ^ (-x)) = 2

-- The main theorem we aim to prove
theorem solve_for_g2 (h : ∀ x, condition g x) : g 2 = 0 :=
by
  sorry

end solve_for_g2_l460_460390


namespace find_a_l460_460947

def complex (w : Type*) := 
  { re: w, im: w } 

def purely_imaginary (z : complex ℝ) : Prop := 
  z.re = 0 ∧ z.im ≠ 0 

theorem find_a (
  a : ℝ) (z : complex ℝ)
 (hz : z = {
    re := (3 * a - 8) / 25,
    im := (4 * a + 6) / 25 })
 (h_imaginary : purely_imaginary z) :
  a = 8 / 3 :=
begin 
  have h1 : (3 * a - 8) / 25 = 0 := sorry,
  have ha : a = 8 / 3 := sorry,
end

end find_a_l460_460947


namespace num_distinct_four_digit_numbers_1335_l460_460619

theorem num_distinct_four_digit_numbers_1335 : 
  let digits := multiset.of_list [1, 3, 3, 5] in
  ((digits.permutations.map (λ l, l.foldl (λ acc d, acc * 10 + d) 0)).to_finset).card = 12 := 
by
  sorry

end num_distinct_four_digit_numbers_1335_l460_460619


namespace meaningful_expression_range_l460_460995

theorem meaningful_expression_range (a : ℝ) :
  (∃ (x : ℝ), x = (sqrt (a + 1)) / (a - 2)) ↔ a ≥ -1 ∧ a ≠ 2 := 
begin
  sorry
end

end meaningful_expression_range_l460_460995


namespace smallest_AAB_value_l460_460492

theorem smallest_AAB_value :
  ∃ (A B : ℕ), 
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  A ≠ B ∧ 
  110 * A + B = 7 * (10 * A + B) ∧ 
  (∀ (A' B' : ℕ), 
    A' ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    B' ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    A' ≠ B' ∧ 
    110 * A' + B' = 7 * (10 * A' + B') → 
    110 * A + B ≤ 110 * A' + B') :=
by
  sorry

end smallest_AAB_value_l460_460492


namespace compare_trip_times_l460_460155

variable (u : ℝ) -- speed on the first trip

-- Time taken for the first trip
def t1 := 60 / u

-- Time taken for the third trip
def t3 := 90 / u

theorem compare_trip_times : t3 = 1.5 * t1 := by
  unfold t1
  unfold t3
  sorry

end compare_trip_times_l460_460155


namespace verify_expression_l460_460549

theorem verify_expression (z : ℝ) (h_z : z = 1.00) :
  ((√1.21) / (√0.81) + (√z) / (√0.49)) = 2.650793650793651 :=
by
  rw h_z
  have h1 : (√1.21) = 1.1, by sorry
  have h2 : (√0.81) = 0.9, by sorry
  have h3 : (√1.0) = 1, by sorry
  have h4 : (√0.49) = 0.7, by sorry
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end verify_expression_l460_460549


namespace goose_flock_count_l460_460835

theorem goose_flock_count (n : ℕ) : ∃ G : ℕ, G = 2^n - 1 :=
by
  use 2^n - 1
  reflexivity
  sorry

end goose_flock_count_l460_460835


namespace find_common_ratio_find_sum_of_first_n_terms_l460_460208

open Real

-- Given conditions
variable {a : ℕ → ℝ}
variable {q : ℝ}
variable (hq : q > 1)
variable (h1 : ∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1))
variable (h2 : a 5 ^ 2 = a 10)

-- Correct answer statements (equivalently reformulated questions)
theorem find_common_ratio : q = 2 := sorry

theorem find_sum_of_first_n_terms {n : ℕ} :
  ∑ i in finset.range n, (a i / 3 ^ i) = 2 * (1 - (2 / 3) ^ n) := sorry

end find_common_ratio_find_sum_of_first_n_terms_l460_460208


namespace probability_divisible_by_5_is_9_over_25_l460_460556

def three_digit_numbers (digits : Finset ℕ) : Finset (ℕ × ℕ × ℕ) :=
  digits.product (digits \ digits.image singleton).product (digits \ digits.image singleton \ ({0} : Finset ℕ))

def divisible_by_5 (numbers : Finset (ℕ × ℕ × ℕ)) : Finset (ℕ × ℕ × ℕ) :=
  numbers.filter (λ n, n.2.2 == 5 ∨ n.2.2 == 0)

theorem probability_divisible_by_5_is_9_over_25 :
  let digits := {0,1,2,3,4,5},
      total_numbers := (three_digit_numbers digits).filter (λ n, n.1 != 0),
      favorable_numbers := (divisible_by_5 total_numbers),
      p := favorable_numbers.card.to_real / total_numbers.card.to_real
  in p = (9:ℚ) / 25 :=
by {
  let digits := {0,1,2,3,4,5},
  let total_numbers := (three_digit_numbers digits).filter (λ n, n.1 != 0),
  let favorable_numbers := (divisible_by_5 total_numbers),
  let p := favorable_numbers.card.to_real / total_numbers.card.to_real,
  sorry
}

end probability_divisible_by_5_is_9_over_25_l460_460556


namespace domino_double_probability_l460_460110

theorem domino_double_probability :
  let dominoes := finset.pair (finset.range 7) (finset.range 7)
  let double_dominoes := finset.filter (λ (p : ℕ × ℕ), p.1 = p.2) dominoes
  (finset.card double_dominoes : ℚ) / (finset.card dominoes : ℚ) = 1 / 7 :=
by sorry

end domino_double_probability_l460_460110


namespace tree_height_at_2_years_l460_460854

theorem tree_height_at_2_years (h₅ : ℕ) (h_four : ℕ) (h_three : ℕ) (h_two : ℕ) (h₅_value : h₅ = 243)
  (h_four_value : h_four = h₅ / 3) (h_three_value : h_three = h_four / 3) (h_two_value : h_two = h_three / 3) :
  h_two = 9 := by
  sorry

end tree_height_at_2_years_l460_460854


namespace line_passes_through_fixed_point_minimum_area_and_line_eq_l460_460964

noncomputable def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  (2 + m) * x + (1 - 2 * m) * y + 4 - 3 * m = 0

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), line_equation m (-1) (-2) :=
by
  sorry

theorem minimum_area_and_line_eq :
  ∃ (m : ℝ), line_equation m (-1) (-2) ∧ ∀ (k : ℝ), k < 0 → 
  let OA := |2/k - 1|
      OB := |k - 2| in
  let AOB_area := (1/2) * |OA * OB| in
  AOB_area = 4 ∧ line_equation (-2) x y
by
  sorry

end line_passes_through_fixed_point_minimum_area_and_line_eq_l460_460964


namespace sheetrock_width_l460_460120

theorem sheetrock_width (l A w : ℕ) (h_length : l = 6) (h_area : A = 30) (h_formula : A = l * w) : w = 5 :=
by
  -- Placeholder for the proof
  sorry

end sheetrock_width_l460_460120


namespace bullet_speed_difference_l460_460434

def bullet_speed_in_same_direction (v_h v_b : ℝ) : ℝ :=
  v_b + v_h

def bullet_speed_in_opposite_direction (v_h v_b : ℝ) : ℝ :=
  v_b - v_h

theorem bullet_speed_difference (v_h v_b : ℝ) (h_h : v_h = 20) (h_b : v_b = 400) :
  bullet_speed_in_same_direction v_h v_b - bullet_speed_in_opposite_direction v_h v_b = 40 :=
by
  rw [h_h, h_b]
  sorry

end bullet_speed_difference_l460_460434


namespace scientific_notation_0_000000022_l460_460666

theorem scientific_notation_0_000000022 :
    (0.000000022 : ℝ) = 2.2 * 10^(-8) :=
by
  sorry

end scientific_notation_0_000000022_l460_460666


namespace cost_to_consume_desired_calories_l460_460753

-- conditions
def calories_per_chip : ℕ := 10
def chips_per_bag : ℕ := 24
def cost_per_bag : ℕ := 2
def desired_calories : ℕ := 480

-- proof statement
theorem cost_to_consume_desired_calories :
  let total_calories_per_bag := chips_per_bag * calories_per_chip in
  let bags_needed := desired_calories / total_calories_per_bag in
  let total_cost := bags_needed * cost_per_bag in
  total_cost = 4 :=
by
  sorry

end cost_to_consume_desired_calories_l460_460753


namespace ellipse_area_problem_l460_460211

-- Define the given conditions in Lean
variables {a b e c x₁ x₂ y₁ y₂ : ℝ}
variables (O : Point ℝ) (A B : Point ℝ)

def ellipse_eqn (a b : ℝ) : Prop := (a > b) ∧ (b > 0)
def eccentricity (a b : ℝ) (e : ℝ) : Prop := e = (Real.sqrt 3) / 2 ∧ a^2 - b^2 = (e * a)^2
def point_on_ellipse (x y a b : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
def perpendicular_bisector_passing_through (A B : Point ℝ) : Prop := (0, 0.5) lies on the perpendicular bisector of segment AB 

-- Rewrite the math proof problem
theorem ellipse_area_problem
  (h1 : ellipse_eqn a b)
  (h2 : eccentricity a b e)
  (h3 : point_on_ellipse 1 (Real.sqrt 3 / 2) a b)
  (h4 : perpendicular_bisector_passing_through A B O) : 
  ((a = 2) ∧ (b = 1) ∧ ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≤ 1) := sorry

end ellipse_area_problem_l460_460211


namespace a_eq_neg_one_tenth_l460_460882

def E (a b c : ℝ) : ℝ := a * b^2 + c

theorem a_eq_neg_one_tenth : ∀ a : ℝ, E(a, 4, 5) = E(a, 6, 7) → a = -1 / 10 :=
by
  intro a h
  simp [E] at h
  sorry

end a_eq_neg_one_tenth_l460_460882


namespace complex_number_imaginary_l460_460273

theorem complex_number_imaginary (x : ℝ) 
  (h1 : x^2 - 2*x - 3 = 0)
  (h2 : x + 1 ≠ 0) : x = 3 := sorry

end complex_number_imaginary_l460_460273


namespace tangent_circles_locus_l460_460771

theorem tangent_circles_locus :
  ∃ (a b : ℝ), ∀ (C1_center : ℝ × ℝ) (C2_center : ℝ × ℝ) (C1_radius : ℝ) (C2_radius : ℝ),
    C1_center = (0, 0) ∧ C2_center = (2, 0) ∧ C1_radius = 1 ∧ C2_radius = 3 ∧
    (∀ (r : ℝ), (a - 0)^2 + (b - 0)^2 = (r + C1_radius)^2 ∧ (a - 2)^2 + (b - 0)^2 = (C2_radius - r)^2) →
    84 * a^2 + 100 * b^2 - 64 * a - 64 = 0 := sorry

end tangent_circles_locus_l460_460771


namespace organize_sports_activities_l460_460288

theorem organize_sports_activities (n : Nat) :
  ∃ (members : Fin (3 * n + 1) → Type) (tennis chess table_tennis : Fin (3 * n + 1) → Fin (3 * n + 1) → Prop),
    (∀ i j, i ≠ j → (tennis i j ∨ chess i j ∨ table_tennis i j)) ∧
    (∀ i, (Finset.univ.filter (tennis i)).card = n) ∧
    (∀ i, (Finset.univ.filter (chess i)).card = n) ∧
    (∀ i, (Finset.univ.filter (table_tennis i)).card = n) :=
sorry

end organize_sports_activities_l460_460288


namespace sum_of_coeffs_of_nonzero_y_powers_in_expansion_l460_460596

theorem sum_of_coeffs_of_nonzero_y_powers_in_expansion : 
  let expr := (4 * x + 3 * y + 2) * (2 * x + 5 * y + 6) in 
  let expanded_expr := expand expr in 
  let coeffs_of_y_terms := [termCoeff t | t in terms expanded_expr, hasNonzeroPowerOfY t] in
  sum coeffs_of_y_terms = 69 := 
by 
  -- Here we would provide proof steps to verify the sum of the coefficients of terms
  -- containing a non-zero power of y is 69, but we omit it with a sorry.
  sorry

end sum_of_coeffs_of_nonzero_y_powers_in_expansion_l460_460596


namespace area_of_set_A_l460_460379

-- Definitions coming from the conditions
def set_A (x y : ℝ) (t : ℝ) : Prop := (x - t)^2 + y^2 ≤ (1 - t / 2)^2 ∧ abs t ≤ 2

-- Theorem stating the given problem with the correct answer
theorem area_of_set_A :
  (∃ (x y : ℝ), set_A x y t) →
  ∃ (A_area : ℝ), A_area = 4 * real.sqrt 3 + 8 * real.pi / 3 :=
by
  sorry

end area_of_set_A_l460_460379


namespace number_of_groups_of_bananas_l460_460793

theorem number_of_groups_of_bananas (total_bananas : ℕ) (bananas_per_group : ℕ) (H_total_bananas : total_bananas = 290) (H_bananas_per_group : bananas_per_group = 145) :
    (total_bananas / bananas_per_group) = 2 :=
by {
  sorry
}

end number_of_groups_of_bananas_l460_460793


namespace minimize_expression_min_value_achieved_at_7_l460_460805

theorem minimize_expression (x : ℝ) : 
  (x^2 - 14 * x + 45) ≥ -4 :=
begin
  sorry
end

theorem min_value_achieved_at_7 (x : ℝ) :
  ∃ x, (x = 7) ∧ (x^2 - 14 * x + 45 = -4) :=
begin
  use 7,
  split,
  {
    sorry,
  },
  {
    sorry,
  }
end

end minimize_expression_min_value_achieved_at_7_l460_460805


namespace correct_relationship_l460_460326

-- Definitions
def Set (α : Type) := α → Prop

def cube : Set RightPrism := sorry -- Placeholder: Define the set of all cubes
def right_prism : Set RightPrism := sorry -- Placeholder: Define the set of all right prisms
def cuboid : Set RightPrism := sorry -- Placeholder: Define the set of all cuboids
def right_rectangular_prism : Set RightPrism := sorry -- Placeholder: Define the set of all right rectangular prisms

lemma cube_subset_right_prism : cube ⊆ right_prism := sorry -- Placeholder: Prove cube is a subset of right prism
lemma right_prism_subset_cuboid : right_prism ⊆ cuboid := sorry -- Placeholder: Prove right prism is a subset of cuboid
lemma cuboid_subset_right_rectangular_prism : cuboid ⊆ right_rectangular_prism := sorry -- Placeholder: Prove cuboid is a subset of right rectangular prism

theorem correct_relationship :
  cube ⊆ right_prism ∧ right_prism ⊆ cuboid ∧ cuboid ⊆ right_rectangular_prism :=
by
  exact ⟨cube_subset_right_prism, right_prism_subset_cuboid, cuboid_subset_right_rectangular_prism⟩

end correct_relationship_l460_460326


namespace false_implies_exists_nonpositive_l460_460724

variable (f : ℝ → ℝ)

theorem false_implies_exists_nonpositive (h : ¬ ∀ x > 0, f x > 0) : ∃ x > 0, f x ≤ 0 :=
by sorry

end false_implies_exists_nonpositive_l460_460724


namespace arith_seq_fraction_l460_460935

theorem arith_seq_fraction (a : ℕ → ℝ) (d : ℝ) (h1 : ∀ n, a (n + 1) - a n = d)
  (h2 : d ≠ 0) (h3 : a 3 = 2 * a 1) :
  (a 1 + a 3) / (a 2 + a 4) = 3 / 4 :=
sorry

end arith_seq_fraction_l460_460935


namespace solution_l460_460708

noncomputable def rightRectPrismVolume : ℝ := 2 * 3 * 5

noncomputable def surfaceArea (a b c : ℝ) : ℝ := 2 * (a * b + a * c + b * c)

noncomputable def quarterCylinderArea (r : ℝ) (lengths : List ℝ) : ℝ := π * r^2 * lengths.sum

noncomputable def oneEighthSphereVolume (r : ℝ) : ℝ := 8 * (1/8) * (4/3 * π * r^3)

noncomputable def solveProblem :=
let a := (4 * π) / 3
let b := 10 * π
let c := 62
let d := rightRectPrismVolume
(b * c) / (a * d) = 15.5

theorem solution : solveProblem := by sorry

end solution_l460_460708


namespace p_implies_q_and_not_converse_l460_460213

def p (a : ℝ) := a ≤ 1
def q (a : ℝ) := abs a ≤ 1

theorem p_implies_q_and_not_converse (a : ℝ) : (p a → q a) ∧ ¬(q a → p a) :=
by
  repeat { sorry }

end p_implies_q_and_not_converse_l460_460213


namespace functional_eq_1996_l460_460736

def f (x : ℝ) : ℝ := sorry

theorem functional_eq_1996 (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - (f x) * (f y) + (f y)^2)) :
    ∀ x : ℝ, f (1996 * x) = 1996 * f x := 
sorry

end functional_eq_1996_l460_460736


namespace baseball_team_opponent_total_score_l460_460095

theorem baseball_team_opponent_total_score : 
  ∀ (team_scores : list ℕ), 
  team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] →
  (∃ (lost_scores won_scores : list ℕ),
    lost_scores = [2, 4, 6, 8, 10, 12] ∧
    won_scores = [1, 2, 3, 4, 5, 6] ∧
    list.sum lost_scores + list.sum won_scores = 63) :=
by
  intro team_scores h_team_scores
  use [2, 4, 6, 8, 10, 12]
  use [1, 2, 3, 4, 5, 6]
  constructor
  . exact rfl
  constructor
  . exact rfl
  . calc
    list.sum [2, 4, 6, 8, 10, 12] + list.sum [1, 2, 3, 4, 5, 6]
    = 42 + 21 : by {refl}
    = 63      : by {refl}

end baseball_team_opponent_total_score_l460_460095


namespace equation_has_unique_solution_l460_460179

theorem equation_has_unique_solution :
  ∀ z : ℝ, (z + 2)^4 + (2 - z)^4 = 258 ↔ z = 1 :=
begin
  intro z,
  sorry
end

end equation_has_unique_solution_l460_460179


namespace find_principal_l460_460144

theorem find_principal 
  (SI : ℝ) (R : ℝ) (T : ℝ)
  (hSI : SI = 950)
  (hR : R = 2.3529411764705883)
  (hT : T = 5) : 
  let P := (SI * 100) / (R * T)
  in P = 8076.923076923077 :=
by
  sorry

end find_principal_l460_460144


namespace derivative_problem_1_derivative_problem_2_l460_460906

section
variable {x : ℝ}

theorem derivative_problem_1 : deriv (λ x : ℝ, x * (1 + 2 / x + 2 / x^2)) x = 1 - 2 / x^2 :=
sorry

theorem derivative_problem_2 : deriv (λ x : ℝ, x^4 - 3 * x^2 - 5 * x + 6) x = 4 * x^3 - 6 * x - 5 :=
sorry
end

end derivative_problem_1_derivative_problem_2_l460_460906


namespace log_base_a_b_eq_pi_l460_460829

variables {a b : ℝ}

-- Conditions
def radius := log (a ^ 3)
def circumference := log (b ^ 6)

-- Theorem to prove
theorem log_base_a_b_eq_pi (h₁ : radius = log (a ^ 3)) (h₂ : circumference = log (b ^ 6)) :
  log a b = real.pi :=
sorry

end log_base_a_b_eq_pi_l460_460829


namespace domain_of_function_l460_460772

def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 9)

theorem domain_of_function :
  { x : ℝ | x^2 - 9 ≥ 0 } = { x : ℝ | x ∈ Set.Iic (-3) ∪ Set.Ici 3 } := by
  sorry

end domain_of_function_l460_460772


namespace sum_of_geometric_sequence_l460_460978

variables {n : ℕ} {a : ℕ → ℝ}
def a1 : ℝ := 1
def an := λ n : ℕ, a n

def S (n : ℕ) := ∑ i in finset.range (n + 1), (a i)

theorem sum_of_geometric_sequence (h : ∀ k, a (k + 1) = (1/5) * a k) :
    S n = (5 / 4) * (1 - (1 / 5) ^ (n + 1)) :=
sorry

end sum_of_geometric_sequence_l460_460978


namespace tree_height_at_2_years_l460_460851

theorem tree_height_at_2_years (h : ℕ → ℕ) 
  (h_growth : ∀ n, h (n + 1) = 3 * h n) 
  (h_5 : h 5 = 243) : 
  h 2 = 9 := 
sorry

end tree_height_at_2_years_l460_460851


namespace complex_division_l460_460584

theorem complex_division (i : ℂ) (h_i : i * i = -1) : (3 - 4 * i) / i = 4 - 3 * i :=
by
  sorry

end complex_division_l460_460584


namespace definite_integral_computation_l460_460512

-- Define the integrand function
def f (x : ℝ) : ℝ := 2 * x - 1 / (x ^ 2)

-- Statement of the problem in Lean
theorem definite_integral_computation :
  ∫ x in 1..3, f x = 22 / 3 :=
sorry

end definite_integral_computation_l460_460512


namespace joan_books_correct_l460_460004

def sam_books : ℕ := 110
def total_books : ℕ := 212

def joan_books : ℕ := total_books - sam_books

theorem joan_books_correct : joan_books = 102 := by
  sorry

end joan_books_correct_l460_460004


namespace no_halfing_matrix_exists_l460_460908

def halves_second_column_matrix (N : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  ∀ (A : Matrix (Fin 2) (Fin 2) ℚ),
    N.mul A = (Matrix.of ![![A 0 0, A 0 1 / 2], ![A 1 0, A 1 1 / 2]])

theorem no_halfing_matrix_exists : 
  ¬ ∃ (N : Matrix (Fin 2) (Fin 2) ℚ), halves_second_column_matrix N :=
by
  sorry

end no_halfing_matrix_exists_l460_460908


namespace prove_area_of_square_l460_460867

noncomputable def area_of_square (s : ℝ) (a : ℝ) (h1 : 3 * s = 4 * a) (h2 : s^2 * real.sqrt 3 / 4 = 9) : ℝ :=
  (3 * s / 4) ^ 2

theorem prove_area_of_square (s a : ℝ) (h1 : 3 * s = 4 * a) (h2 : s^2 * real.sqrt 3 / 4 = 9) :
  area_of_square s a h1 h2 = 27 * real.sqrt 3 / 4 :=
  sorry

end prove_area_of_square_l460_460867


namespace ordered_pairs_count_l460_460554

theorem ordered_pairs_count : 
  (∀ (b c : ℕ), b > 0 ∧ b ≤ 6 ∧ c > 0 ∧ c ≤ 6 ∧ b^2 - 4 * c < 0 ∧ c^2 - 4 * b < 0 → 
  ((b = 1 ∧ (c = 2 ∨ c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 2 ∧ (c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 3 ∧ (c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6)) ∨ 
  (b = 4 ∧ (c = 5 ∨ c = 6)))) ∧
  (∃ (n : ℕ), n = 15) := sorry

end ordered_pairs_count_l460_460554


namespace no_contradiction_logarithm_base_b_l460_460734

noncomputable def log_base_b (b : ℝ) (x : ℝ) : ℝ :=
  if h : x > 0 ∧ b > 1 then log x / log b else 0

theorem no_contradiction_logarithm_base_b (b : ℝ) :
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y = log_base_b b x) ∧
  (∀ y : ℝ, ∃ x : ℝ, x > 0 ∧ y = log_base_b b x) →
  ¬ (∃ c1 c2 : Prop, c1 ∧ c2 ∧ ¬ (c1 ↔ c2)) :=
by
  intros h
  sorry

end no_contradiction_logarithm_base_b_l460_460734


namespace sqrt_40000_eq_200_l460_460761

theorem sqrt_40000_eq_200 : Real.sqrt 40000 = 200 := 
sorry

end sqrt_40000_eq_200_l460_460761


namespace remainder_is_576_l460_460706

-- Definitions based on the conditions
def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def num_functions (f : ℕ → ℕ) (A : Set ℕ) : ℕ :=
  if ∃ c ∈ A, ∀ x ∈ A, f (f x) = c then 1 else 0

-- Define the total number of such functions
def total_functions (A : Set ℕ) : ℕ :=
  8 * (∑ k in ({1, 2, 3, 4, 5, 6, 7} : Set ℕ), Nat.choose 7 k * (k ^ (7 - k)))

-- The main theorem to prove
theorem remainder_is_576 : (total_functions A % 1000) = 576 :=
  by
    sorry

end remainder_is_576_l460_460706


namespace count_multiples_3_or_5_not_6_up_to_200_l460_460623

theorem count_multiples_3_or_5_not_6_up_to_200 : 
  (Finset.card (Finset.filter (λ n, n ≤ 200 ∧ ((n % 3 = 0 ∨ n % 5 = 0) ∧ n % 6 ≠ 0)) (Finset.range 201))) = 60 := 
by 
  sorry

end count_multiples_3_or_5_not_6_up_to_200_l460_460623


namespace _l460_460217

noncomputable theorem max_sum_abs_diff {a : Fin 100 → ℕ} 
  (h : ∀ i, 1 ≤ a i ∧ a i ≤ 99 ∧ ∀ i j, i ≠ j → a i ≠ a j) : 
  (Finset.univ.sum (λ i, |a i - ↑i.succ|) = 4900) :=
sorry

end _l460_460217


namespace sum_of_last_two_digits_l460_460152

-- Definitions based on given conditions
def six_power_twenty_five := 6^25
def fourteen_power_twenty_five := 14^25
def expression := six_power_twenty_five + fourteen_power_twenty_five
def modulo := 100

-- The statement we need to prove
theorem sum_of_last_two_digits : expression % modulo = 0 := by
  sorry

end sum_of_last_two_digits_l460_460152


namespace min_distance_from_P_to_line_l460_460300

noncomputable def min_distance_to_line : ℝ :=
2 * Real.sqrt 2 - 0.5 * Real.sqrt 10

theorem min_distance_from_P_to_line :
  ∀ (m : ℝ) (x y : ℝ),
  (x + m * y = 0) →
  (m * x - y - m + 3 = 0) →
  let P := (x, y) in
  ∃ d, d = min_distance_to_line ∧ 
  d = Real.abs (x + y - 8) / Real.sqrt (1^2 + 1^2) - 0.5 * Real.sqrt 10 :=
begin
  intros m x y,
  intros h1 h2,
  use min_distance_to_line,
  split,
  { refl },
  { sorry }
end

end min_distance_from_P_to_line_l460_460300


namespace area_not_covered_by_small_squares_l460_460128

def large_square_side_length : ℕ := 10
def small_square_side_length : ℕ := 4
def large_square_area : ℕ := large_square_side_length ^ 2
def small_square_area : ℕ := small_square_side_length ^ 2
def uncovered_area : ℕ := large_square_area - small_square_area

theorem area_not_covered_by_small_squares :
  uncovered_area = 84 := by
  sorry

end area_not_covered_by_small_squares_l460_460128


namespace cosine_of_angle_l460_460958

variable (x y r : ℝ) (θ : ℝ)
variable (h : (x, y) = (3, -4))

def cosine := x / r

theorem cosine_of_angle : x = 3 → y = -4 → r = Real.sqrt (x^2 + y^2) → cosine θ = 3 / 5 :=
by
  intros hx hy hr
  rw [hx, hy] at hr
  rw [hx, hr]
  sorry

end cosine_of_angle_l460_460958


namespace ribbons_jane_uses_l460_460307

-- Given conditions
def dresses_sewn_first_period (dresses_per_day : ℕ) (days : ℕ) : ℕ :=
  dresses_per_day * days

def dresses_sewn_second_period (dresses_per_day : ℕ) (days : ℕ) : ℕ :=
  dresses_per_day * days

def total_dresses_sewn (dresses_first_period : ℕ) (dresses_second_period : ℕ) : ℕ :=
  dresses_first_period + dresses_second_period

def total_ribbons_used (total_dresses : ℕ) (ribbons_per_dress : ℕ) : ℕ :=
  total_dresses * ribbons_per_dress

-- Theorem to prove
theorem ribbons_jane_uses :
  total_ribbons_used (total_dresses_sewn (dresses_sewn_first_period 2 7) (dresses_sewn_second_period 3 2)) 2 = 40 :=
  sorry

end ribbons_jane_uses_l460_460307


namespace num_perfect_squares_mul_36_lt_10pow8_l460_460259

theorem num_perfect_squares_mul_36_lt_10pow8 : 
  ∃(n : ℕ), n = 1666 ∧ 
  ∀ (N : ℕ), (1 ≤ N) → (N^2 < 10^8) → (N^2 % 36 = 0) → 
  (N ≤ 9996 ∧ N % 6 = 0) :=
by
  sorry

end num_perfect_squares_mul_36_lt_10pow8_l460_460259


namespace find_xyz_l460_460943

theorem find_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 17) 
  (h3 : x^3 + y^3 + z^3 = 27) : 
  x * y * z = 32 / 3 :=
  sorry

end find_xyz_l460_460943


namespace area_of_octagon_l460_460127

theorem area_of_octagon (a b : ℝ) (hsquare : a ^ 2 = 16)
  (hperimeter : 4 * a = 8 * b) :
  2 * (1 + Real.sqrt 2) * b ^ 2 = 8 + 8 * Real.sqrt 2 :=
by
  sorry

end area_of_octagon_l460_460127


namespace triangle_circle_product_l460_460679

theorem triangle_circle_product
  (ABC : Triangle)
  (I : Point)
  (P Q : Point)
  (R r : ℝ) :
  (isIncircle I ABC) →
  (isCircumradius R ABC) →
  (isInradius r I) →
  (isChordThroughIncenter PQ ABC I) →
  (segmentProduct I P Q = 2 * R * r) :=
by
  sorry

end triangle_circle_product_l460_460679


namespace tree_height_at_2_years_l460_460852

theorem tree_height_at_2_years (h : ℕ → ℕ) 
  (h_growth : ∀ n, h (n + 1) = 3 * h n) 
  (h_5 : h 5 = 243) : 
  h 2 = 9 := 
sorry

end tree_height_at_2_years_l460_460852


namespace vasya_repainted_cells_correct_l460_460287

theorem vasya_repainted_cells_correct
  (grid : Vector (Vector bool 5) 5)
  (painted_black : ∀ i j, grid[i][j] = true → grid[i+1][j] + grid[i][j+1] + grid[i+1][j+1] ≤ 2)
  (vasya_repainted : ∀ (i₁ j₁ i₂ j₂ : Fin 5), i₁ ≠ i₂ → j₁ ≠ j₂ → grid[i₁][j₁] = false)
  (res : ∀ (i : Fin 5), (Vector (Fin 5) 5)) : 
  res = [⟨0, 3, 2, 4, 1⟩] :=
begin
  have h1 : res[0] = 1, by sorry,
  have h2 : res[1] = 4, by sorry,
  have h3 : res[2] = 2, by sorry,
  have h4 : res[3] = 3, by sorry,
  have h5 : res[4] = 5, by sorry,
  exact ⟨h1, h2, h3, h4, h5⟩,
end

end vasya_repainted_cells_correct_l460_460287


namespace find_m_l460_460215

-- Define points and vectors
structure Point := (x : ℝ) (y : ℝ)
def O : Point := ⟨0, 0⟩
def A : Point := ⟨-1, 3⟩
def B : Point := ⟨2, -4⟩

-- Define vectors as differences between points
def vector (P Q : Point) : Point := ⟨Q.x - P.x, Q.y - P.y⟩

-- Define vector OA and AB
def OA : Point := vector O A
def AB : Point := vector A B

-- Define the point P to be on the y-axis
axiom y_axis (P : Point) : P.x = 0

-- Define the operation OP
def OP (m : ℝ) : Point := ⟨2 * OA.x + m * AB.x, 2 * OA.y + m * AB.y⟩

-- The theorem we want to prove
theorem find_m (P : Point) (m : ℝ) (hP : y_axis P)
  (hOP : OP m = P) : m = 2 / 3 := by
  sorry

end find_m_l460_460215


namespace sqrt_5_transform_1_sqrt_5_transform_2_sqrt_10_transform_1_sqrt_10_transform_2_sqrt_6_transform_sqrt_7_transform_sqrt_11_transform_sqrt_13_transform_sqrt_14_transform_sqrt_15_transform_sqrt_16_transform_sqrt_26_transform_l460_460450

theorem sqrt_5_transform_1 : ∀ (x y : ℕ) (z : ℤ),
  x = 7 ∧ y = 3 ∧ z = -4 → sqrt 5 = (7:ℚ) / 3 * sqrt (1 - (4/49 : ℚ)) := sorry

theorem sqrt_5_transform_2 : ∀ (x y : ℕ) (z : ℤ),
  x = 15 ∧ y = 7 ∧ z = 4 → sqrt 5 = (15:ℚ) / 7 * sqrt (1 + (4/49 : ℚ)) := sorry

theorem sqrt_10_transform_1 : ∀ (x y : ℕ) (z : ℤ),
  x = 22 ∧ y = 7 ∧ z = -6 → sqrt 10 = (22:ℚ) / 7 * sqrt (1 - (6/484 : ℚ)) := sorry

theorem sqrt_10_transform_2 : ∀ (x y : ℕ) (z : ℤ),
  x = 19 ∧ y = 6 ∧ z = -1 → sqrt 10 = (19:ℚ) / 6 * sqrt (1 - (1/361 : ℚ)) := sorry

theorem sqrt_6_transform : ∀ (x y : ℕ) (z : ℤ),
  x = 5 ∧ y = 2 ∧ z = -1 → sqrt 6 = (5:ℚ) / 2 * sqrt (1 - (1/25 : ℚ)) := sorry

theorem sqrt_7_transform : ∀ (x y : ℕ) (z : ℤ),
  x = 8 ∧ y = 3 ∧ z = -1 → sqrt 7 = (8:ℚ) / 3 * sqrt (1 - (1/64 : ℚ)) := sorry

theorem sqrt_11_transform : ∀ (x y : ℕ) (z : ℤ),
  x = 10 ∧ y = 3 ∧ z = -1 → sqrt 11 = (10:ℚ) / 3 * sqrt (1 - (1/100 : ℚ)) := sorry

theorem sqrt_13_transform : ∀ (x y : ℕ) (z : ℤ),
  x = 18 ∧ y = 5 ∧ z = -1 → sqrt 13 = (18:ℚ) / 5 * sqrt (1 - (1/324 : ℚ)) := sorry

theorem sqrt_14_transform : ∀ (x y : ℕ) (z : ℤ),
  x = 15 ∧ y = 4 ∧ z = -1 → sqrt 14 = (15:ℚ) / 4 * sqrt (1 - (1/225 : ℚ)) := sorry

theorem sqrt_15_transform : ∀ (x y : ℕ) (z : ℤ),
  x = 31 ∧ y = 8 ∧ z = -1 → sqrt 15 = (31:ℚ) / 8 * sqrt (1 - (1/961 : ℚ)) := sorry

theorem sqrt_16_transform : ∀ (x y : ℕ) (z : ℤ),
  x = 13 ∧ y = 3 ∧ z = -25 → sqrt 16 = (13:ℚ) / 3 * sqrt (1 - (25/169 : ℚ)) := sorry

theorem sqrt_26_transform : ∀ (x y : ℕ) (z : ℤ),
  x = 5 ∧ y = 1 ∧ z = 1 → sqrt 26 = 5 * sqrt (1 + (1/25 : ℚ)) := sorry

end sqrt_5_transform_1_sqrt_5_transform_2_sqrt_10_transform_1_sqrt_10_transform_2_sqrt_6_transform_sqrt_7_transform_sqrt_11_transform_sqrt_13_transform_sqrt_14_transform_sqrt_15_transform_sqrt_16_transform_sqrt_26_transform_l460_460450


namespace equation_of_line_l460_460952

noncomputable def midpoint (A B P : Point) : Prop :=
  (A.x + B.x) / 2 = P.x ∧ (A.y + B.y) / 2 = P.y

noncomputable def on_ellipse (A B : Point) : Prop :=
  A.x^2 / 4 + A.y^2 / 2 = 1 ∧ B.x^2 / 4 + B.y^2 / 2 = 1

noncomputable def line_equation (A B : Point) : String :=
  "x + 2y - 3 = 0"

theorem equation_of_line {P A B : Point} (Hmidpoint : midpoint A B P)
  (Hoverellipse : on_ellipse A B) :
  line_equation A B = "x + 2y - 3 = 0" :=
  sorry

end equation_of_line_l460_460952


namespace identify_value_of_expression_l460_460579

theorem identify_value_of_expression (x y z : ℝ)
  (h1 : y / (x - y) = x / (y + z))
  (h2 : z^2 = x * (y + z) - y * (x - y)) :
  (y^2 + z^2 - x^2) / (2 * y * z) = 1 / 2 := 
sorry

end identify_value_of_expression_l460_460579


namespace area_of_triangle_l460_460799

-- Definitions for the conditions
def intersection_point := (8 : ℝ, 10 : ℝ)
def sum_y_intercepts := 10

-- Proving the area of triangle APQ given the conditions
theorem area_of_triangle
  (y_intercept1 y_intercept2 : ℝ)
  (b_sum : y_intercept1 + y_intercept2 = sum_y_intercepts)
  (perpendicular : y_intercept1 ≠ y_intercept2)
  : let P := (0 : ℝ, y_intercept1),
        Q := (0 : ℝ, y_intercept2),
        A := intersection_point
    in ∃ (area : ℝ), area = 40 :=
by
  sorry

end area_of_triangle_l460_460799


namespace max_value_f_on_interval_l460_460563

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (x^2 - 4) * (x - a)

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ :=
  3 * x^2 - 2 * a * x - 4

theorem max_value_f_on_interval :
  f' (-1) (1 / 2) = 0 →
  ∃ max_f, max_f = 42 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 4, f x (1 / 2) ≤ max_f :=
by
  sorry

end max_value_f_on_interval_l460_460563


namespace cartesian_to_polar_coordinates_l460_460768

noncomputable def cartToPolar (x y : ℝ) : ℝ × ℝ :=
let r := real.sqrt (x^2 + y^2) in
let θ := real.arccos (x / r) in
if y ≥ 0 then (r, θ) else (r, -θ)

theorem cartesian_to_polar_coordinates
  (x y : ℝ)
  (h_x : x = -1)
  (h_y : y = real.sqrt 3) :
  cartToPolar x y = (2, 2 * real.pi / 3) :=
by 
  sorry

end cartesian_to_polar_coordinates_l460_460768


namespace AB_perpendicular_to_plane_alpha_l460_460590

-- Definitions based on the conditions in a)
def n : ℝ × ℝ × ℝ := (2, -2, 4)
def AB : ℝ × ℝ × ℝ := (-1, 1, -2)

-- Statement of the problem as a Lean 4 theorem
theorem AB_perpendicular_to_plane_alpha (n : ℝ × ℝ × ℝ) (AB : ℝ × ℝ × ℝ)
    (h1 : n = (2, -2, 4))
    (h2 : AB = (-1, 1, -2)) :
    (AB = (-(1 / 2) * n)) → (AB.1 * n.1 + AB.2 * n.2 + AB.3 * n.3 = 0) :=
by
  intro h
  sorry

end AB_perpendicular_to_plane_alpha_l460_460590


namespace prime_implies_n_eq_3k_l460_460357

theorem prime_implies_n_eq_3k (n : ℕ) (p : ℕ) (k : ℕ) (h_pos : k > 0)
  (h_prime : Prime p) (h_eq : p = 1 + 2^n + 4^n) :
  ∃ k : ℕ, k > 0 ∧ n = 3^k :=
by
  sorry

end prime_implies_n_eq_3k_l460_460357


namespace a_n_is_perfect_square_l460_460609

def seqs : ℕ → ℕ × ℕ 
| 0       := (1, 0)
| (n + 1) := let (a_n, b_n) := seqs n in (7 * a_n + 6 * b_n - 3, 8 * a_n + 7 * b_n - 4)

def a (n : ℕ) : ℕ := (seqs n).1

theorem a_n_is_perfect_square (n : ℕ) : ∃ k : ℕ, a n = k * k :=
sorry

end a_n_is_perfect_square_l460_460609


namespace ellipse_equation_and_triangle_area_l460_460210

theorem ellipse_equation_and_triangle_area (a b : Real) (P : Real × Real) (c : Real)
  (ellipse_eq : ∀ (x y : Real), (x, y) ∈ P → x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1)
  (c_value : c = Real.sqrt 2 * b)
  (ab_ineq : a > b ∧ b > 0) :
  (a = 2 ∧ b = Real.sqrt (4 / 3)) ∧
  let M := (-2, 0) in let N := (1, 1) in
  let PM := Real.sqrt 2 in let PN := 2 * Real.sqrt 2 in
  let area := 1 / 2 * PM * PN in
  area = 2 := 
begin
  sorry
end

end ellipse_equation_and_triangle_area_l460_460210


namespace stack_map_A_front_view_l460_460878

def column1 : List ℕ := [3, 1]
def column2 : List ℕ := [2, 2, 1]
def column3 : List ℕ := [1, 4, 2]
def column4 : List ℕ := [5]

def tallest (l : List ℕ) : ℕ :=
  l.foldl max 0

theorem stack_map_A_front_view :
  [tallest column1, tallest column2, tallest column3, tallest column4] = [3, 2, 4, 5] := by
  sorry

end stack_map_A_front_view_l460_460878


namespace number_of_incorrect_statements_is_four_l460_460161

theorem number_of_incorrect_statements_is_four :
  (∀ (h1: Prop), h1 = "Using modern horse regression to predict prehistoric horse height is inaccurate" →
  (h1 = "Incorrect")) →
  (∀ (h2: Prop), h2 = "Using Phalaenopsis orchids regression to predict Cymbidium orchids germination rate is flawed" →
  (h2 = "Incorrect")) →
  (∀ (h3: Prop), h3 = "Using advertising cost regression to predict sales volume with other market conditions is unreliable" →
  (h3 = "Incorrect")) →
  (∀ (h4: Prop), h4 = "Using female college students' regression to predict weight of a 13-year-old boy is inappropriate" →
  (h4 = "Incorrect")) →
  (4 = 4) := sorry

end number_of_incorrect_statements_is_four_l460_460161


namespace range_of_f_l460_460929

noncomputable def f (x : ℝ) : ℝ :=
  Real.tan x + Real.cot x + 1 / Real.sin x - 1 / Real.cos x

theorem range_of_f :
  (∀ x : ℝ, 0 < x ∧ x < Real.pi / 2 → 1 < f x) ∧
  (∀ y : ℝ, 1 < y → ∃ x : ℝ, 0 < x ∧ x < Real.pi / 2 ∧ f x = y) :=
by
  sorry

end range_of_f_l460_460929


namespace find_f1_plus_g1_l460_460564

variable (f g : ℝ → ℝ)

-- Conditions
def even_function (h : ℝ → ℝ) := ∀ x : ℝ, h x = h (-x)
def odd_function (h : ℝ → ℝ) := ∀ x : ℝ, h x = -h (-x)
def function_relation := ∀ x : ℝ, f x - g x = x^3 + x^2 + 1

-- Mathematically equivalent proof problem
theorem find_f1_plus_g1
  (hf_even : even_function f)
  (hg_odd : odd_function g)
  (h_relation : function_relation f g) :
  f 1 + g 1 = 1 := by
  sorry

end find_f1_plus_g1_l460_460564


namespace travel_time_difference_proof_l460_460767

def telegraph_road_length : ℝ := 162
def detours : list ℝ := [5.2, 2.7, 3.8, 4.4]
def pardee_road_length_m : ℝ := 12000
def road_work_increase : ℝ := 2.5
def travel_speed : ℝ := 80

noncomputable def total_telegraph_road_length : ℝ :=
  telegraph_road_length + detours.sum

noncomputable def pardee_road_length_km : ℝ :=
  pardee_road_length_m / 1000

noncomputable def total_pardee_road_length : ℝ :=
  pardee_road_length_km + road_work_increase

noncomputable def travel_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed * 60 -- convert hours to minutes

noncomputable def travel_time_difference : ℝ :=
  travel_time total_telegraph_road_length travel_speed -
  travel_time total_pardee_road_length travel_speed

theorem travel_time_difference_proof :
  travel_time_difference = 122.7 :=
by
  sorry

end travel_time_difference_proof_l460_460767


namespace part1_part2a_part2b_part2c_l460_460915

-- Definition of fixed points and stable points
def fixedPoints (f : ℝ → ℝ) : set ℝ := {x | f x = x}
def stablePoints (f : ℝ → ℝ) : set ℝ := {x | f (f x) = x}

-- Problem (1) setup
variables (a b c : ℝ) (h_a : a ≠ 0) (h_A : fixedPoints (λ x, a * x^2 + b * x + c) = ∅)

-- Statement for problem (1)
theorem part1 : stablePoints (λ x, a * x^2 + b * x + c) = ∅ :=
sorry

-- Problem (2) setup
def f2 (x : ℝ) : ℝ := 3 * x + 4

-- Statement for problem (2)
theorem part2a : fixedPoints f2 = {-2} :=
sorry

theorem part2b : stablePoints f2 = {-2} :=
sorry

-- Counterexample function
def f_counter (x : ℕ) : ℕ
| 1 := 1
| 2 := 3
| 3 := 2
| _ := x -- Define as identity function on other values

-- Statement for counterexample
theorem part2c : 
  fixedPoints (λ x, f_counter x) = {1} ∧ 
  stablePoints (λ x, f_counter x) = {1, 2, 3} ∧ 
  fixedPoints (λ x, f_counter x) ≠ stablePoints (λ x, f_counter x) :=
sorry

end part1_part2a_part2b_part2c_l460_460915


namespace angle_EKP_is_21_l460_460664

variables {D E F P Q K : Type}
variables [affine_space D K] [affine_space E K] [affine_space F K]
variables (DEF : triangle D E F)

def acute (t : triangle D E F) : Prop :=
  ∀ (a b c : angle), sum_angles t a b c = 180 ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ a < 90 ∧ b < 90 ∧ c < 90

-- Angle DEF is 58 degrees
def angle_DEF (t : triangle D E F) : Prop :=
  ∃ (a : angle), measure_angle (angle DEF) = 58

-- Angle DFE is 69 degrees
def angle_DFE (t : triangle D E F) : Prop :=
  ∃ (a : angle), measure_angle (angle DFE) = 69

-- Altitudes intersect at K
def altitudes_intersect (t : triangle D E F) : Prop :=
  ∃ (D E F P Q K : Type), altitude D P ∧ altitude E Q ∧ intersect P Q K

-- Define the proof problem
theorem angle_EKP_is_21
  (h1 : acute DEF)
  (h2 : angle_DEF DEF)
  (h3 : angle_DFE DEF)
  (h4 : altitudes_intersect DEF) :
  ∃ angle_EKP : angle, measure_angle angle_EKP = 21 := sorry

end angle_EKP_is_21_l460_460664


namespace acute_triangle_side_range_l460_460934

theorem acute_triangle_side_range (a : ℝ) :
  (triangle_abc : Triangle ℝ)
  (triangle_acuteness : triangle_abc.acute)
  (side_ab : triangle_abc.side₁.length = 3)
  (side_bc : triangle_abc.side₂.length = 4)
  (side_ca : triangle_abc.side₃.length = a)
  : sqrt 7 < a ∧ a < 5 :=
sorry

end acute_triangle_side_range_l460_460934


namespace solve_base_r_l460_460748

theorem solve_base_r (r : ℕ) (hr : Even r) (x : ℕ) (hx : x = 9999) 
                     (palindrome_condition : ∃ (a b c d : ℕ), 
                      b + c = 24 ∧ 
                      ∀ (r_repr : List ℕ), 
                      r_repr.length = 8 ∧
                      r_repr = [a, b, c, d, d, c, b, a] ∧ 
                      ∃ x_squared_repr, x^2 = x_squared_repr) : r = 26 :=
by
  sorry

end solve_base_r_l460_460748


namespace min_sum_xi_l460_460544

def fibonacci : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

theorem min_sum_xi (n : ℕ) (x : ℕ → ℝ) (hx0 : x 0 = 1) (nonneg_x : ∀ i : ℕ, 0 ≤ x i)
  (cond : ∀ i, x i ≤ x (i + 1) + x (i + 2)) :
  x 0 + ∑ i in finset.range (n + 1), x (i + 1) = (fibonacci (n + 2) - 1) / fibonacci n :=
sorry

end min_sum_xi_l460_460544


namespace tangent_line_through_fixed_point_l460_460606

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * x^2 + 1

theorem tangent_line_through_fixed_point :
  ∀ (x1 x2 : ℝ), tangent_point f x1 a → tangent_point f x2 a →
  passes_through (line_through_points (x1, f x1) (x2, f x2)) (0, 2) :=
by
  sorry

-- Definitions of tangent_point and passes_through (which would be mathematically detailed, 
-- but for this problem statement, are just intended to show how they would integrate):
def tangent_point (f : ℝ → ℝ) (x a : ℝ) : Prop := 
  f'(x) = (f x - f a) / (x - a)

def passes_through (l : ℝ → ℝ) (p : ℝ × ℝ) : Prop := 
  l p.1 = p.2

def line_through_points (p1 p2 : ℝ × ℝ) : ℝ → ℝ := 
  λ x, p1.2 + (p2.2 - p1.2) / (p2.1 - p1.1) * (x - p1.1)

end tangent_line_through_fixed_point_l460_460606


namespace marching_band_ratio_l460_460141

theorem marching_band_ratio 
  (total_students : ℕ) 
  (alto_saxophone_players : ℕ) 
  (saxophone_ratio : ∀ x, x = 3 * alto_saxophone_players) 
  (brass_instrument_ratio : ∀ y, y = 5 * (saxophone_ratio (alto_saxophone_players * 3)))
  (marching_band_ratio : ∀ z, z = 2 * (brass_instrument_ratio (alto_saxophone_players * 3 * 5)))
  (total_students_condition : total_students = 600)
  (alto_saxophone_players_condition : alto_saxophone_players = 4)
  : (marching_band_ratio (2 * (brass_instrument_ratio (alto_saxophone_players * 3 * 5)))) / total_students = 1/5 :=
by 
  sorry

end marching_band_ratio_l460_460141


namespace ron_picks_books_two_times_per_year_l460_460372

-- Definitions based on conditions:
def total_members : ℕ := 3 * 2 + 5 + 2  -- Three couples, five single people, Ron, and his wife
def weeks_per_year : ℕ := 52
def holiday_extensions : ℕ := 4
def guest_weeks_per_month : ℕ := 1
def months_per_year : ℕ := 12

-- Prove that Ron gets to pick a new book exactly 2 times a year:
theorem ron_picks_books_two_times_per_year :
  let meeting_weeks := weeks_per_year - holiday_extensions in
  let guest_weeks := guest_weeks_per_month * months_per_year in
  let available_weeks := meeting_weeks - guest_weeks in
  let picks_per_member := available_weeks / total_members in
  picks_per_member = 2 :=
by
  let meeting_weeks := weeks_per_year - holiday_extensions
  let guest_weeks := guest_weeks_per_month * months_per_year
  let available_weeks := meeting_weeks - guest_weeks
  let picks_per_member := available_weeks / total_members
  have : picks_per_member = 2 := sorry
  exact this

end ron_picks_books_two_times_per_year_l460_460372


namespace recurring03_m_mult_recurring8_l460_460902

noncomputable def recurring03_fraction : ℚ :=
let x := 0.03.represented_by_recurring 100 in
(100 * x - x / 99)

noncomputable def recurring8_fraction : ℚ :=
let y := 0.8.represented_by_recurring 10 in
(10 * y - y / 9)

theorem recurring03_m_mult_recurring8 : recurring03_fraction * recurring8_fraction = 8 / 297 := 
by
  sorry

end recurring03_m_mult_recurring8_l460_460902


namespace diff_set_cardinality_l460_460620

-- Define the set S
def S : Set ℤ := {1, 2, 3, 4, 5, 7}

-- Define the function to compute the set of differences
def diff_set (A : Set ℤ) : Set ℤ :=
  {d | ∃ a b ∈ A, a ≠ b ∧ d = |a - b|}

-- Prove that the cardinality of the set of differences is 6
theorem diff_set_cardinality : (diff_set S).card = 6 := by
  sorry

end diff_set_cardinality_l460_460620


namespace available_codes_count_l460_460742

def is_valid_code (code : Nat) : Bool :=
  let d1 := code / 100
  let d2 := (code / 10) % 10
  let d3 := code % 10
  (code < 1000) &&
  (d1 != d2 || d2 != d3 || d1 != d3) && -- Condition 1
  (code != 112 && code != 211 && code != 121) && -- Conditions 2 and 3
  not ((d1 = d2 && d3 != d1) || (d2 = d3 && d1 != d2) || (d1 = d3 && d2 != d1)) -- Condition 4

theorem available_codes_count : (Finset.filter is_valid_code (Finset.range 1000)).card = 979 :=
by
  sorry

end available_codes_count_l460_460742


namespace polynomial_properties_l460_460176

theorem polynomial_properties :
  ∃ P : Polynomial ℝ,
    P.degree = 5 ∧
    leading_coeff P = 200 ∧
    P.coeff 1 = 2 ∧
    P.sum_coeffs = 4 ∧
    P.eval (-1) = 0 ∧
    P.eval 2 = 6 ∧
    P.eval 3 = 8 ∧
    P = 200 * X^5 + 1000 * X^4 - 2000 * X^3 + 2000 * X^2 - 998 * X - 1198 :=
  by
    sorry

end polynomial_properties_l460_460176


namespace distinct_real_roots_l460_460963

-- Define the quadratic equation 2x^2 - k = 0
def quadratic_eq (a b c : ℝ) : (x : ℝ) → ℝ := λ x, a * x^2 + b * x + c

-- Discriminant formula for quadratic equations
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the conditions for having two distinct real roots
theorem distinct_real_roots (k : ℝ) (h : k > 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq 2 0 (-k) x1 = 0 ∧ quadratic_eq 2 0 (-k) x2 = 0) :=
begin
  let Δ := discriminant 2 0 (-k),
  have Δ_positive : Δ > 0,
  {
    simp [discriminant],
    linarith,
  },

  -- Purely as a conclusion that the discriminant being positive implies existence of distinct roots
  sorry
end

-- Example value for k = 1
example : distinct_real_roots 1 (by linarith) :=
by apply distinct_real_roots; linarith

end distinct_real_roots_l460_460963


namespace number_of_valid_functions_l460_460545

def f : Fin 7 → ℤ

def satisfies_conditions (f : Fin 7 → ℤ) : Prop :=
  f 0 = 0 ∧
  f 6 = 12 ∧
  ∀ x y : Fin 7, |x - y| ≤ |f x - f y| ∧ |f x - f y| ≤ 3 * |x - y|

theorem number_of_valid_functions : 
  {f : (Fin 7 → ℤ) // satisfies_conditions f}.to_finset.card = 185 := 
sorry

end number_of_valid_functions_l460_460545


namespace x_n_gt_sqrt2_T_n_lt_bound_l460_460603

noncomputable def f (x : ℝ) : ℝ := x^2 - 2

def x_n (n : ℕ) : ℝ :=
  match n with
  | 0 => 3
  | n+1 => (2 + (x_n n)^2) / (2 * x_n n)

def b_n (n : ℕ) : ℝ := x_n n - Real.sqrt 2

def T_n (n : ℕ) : ℝ := (Finset.range n).sum (λ i, b_n (i + 1))

theorem x_n_gt_sqrt2 (n : ℕ) : n > 0 → x_n n > Real.sqrt 2 := 
sorry

theorem T_n_lt_bound (n : ℕ) : T_n n < 2 * (3 - Real.sqrt 2) :=
sorry

end x_n_gt_sqrt2_T_n_lt_bound_l460_460603


namespace bullet_speed_difference_l460_460436

theorem bullet_speed_difference (speed_horse speed_bullet : ℕ) 
    (h_horse : speed_horse = 20) (h_bullet : speed_bullet = 400) :
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    speed_same_direction - speed_opposite_direction = 40 :=
    by
    -- Define the speeds in terms of the given conditions.
    let speed_same_direction := speed_bullet + speed_horse;
    let speed_opposite_direction := speed_bullet - speed_horse;
    -- State the equality to prove.
    show speed_same_direction - speed_opposite_direction = 40;
    -- Proof (skipped here).
    -- sorry is used to denote where the formal proof steps would go.
    sorry

end bullet_speed_difference_l460_460436


namespace exists_difference_of_three_l460_460689

theorem exists_difference_of_three (A : Finset ℕ) (hA_card : A.card = 16) (hA_range : ∀ x ∈ A, 1 ≤ x ∧ x ≤ 106) 
  (h_diff : ∀ {x y : ℕ}, x ∈ A → y ∈ A → x ≠ y → |x - y| ∉ {6, 9, 12, 15, 18, 21}) :
  ∃ a b ∈ A, a ≠ b ∧ |a - b| = 3 :=
by
  sorry

end exists_difference_of_three_l460_460689


namespace gloria_money_left_calculation_l460_460617

noncomputable def cabin_cost := 129000
noncomputable def cash := 150
noncomputable def cypress_trees := 20
noncomputable def pine_trees := 600
noncomputable def maple_trees := 24
noncomputable def cypress_price := 100
noncomputable def pine_price := 200
noncomputable def maple_price := 300
noncomputable def sales_tax_rate := 0.05
noncomputable def down_payment_rate := 0.10

theorem gloria_money_left_calculation :
  let revenue :=
        cypress_trees * cypress_price +
        pine_trees * pine_price +
        maple_trees * maple_price
  let total_sales_tax := revenue * sales_tax_rate
  let net_amount := revenue - total_sales_tax
  let total_amount_with_cash := net_amount + cash
  let down_payment := cabin_cost * down_payment_rate
  let money_left := total_amount_with_cash - down_payment
  money_left = 109990 := 
by {
  let revenue := cypress_trees * cypress_price +
                 pine_trees * pine_price +
                 maple_trees * maple_price
  let total_sales_tax := revenue * sales_tax_rate
  let net_amount := revenue - total_sales_tax
  let total_amount_with_cash := net_amount + cash
  let down_payment := cabin_cost * down_payment_rate
  let money_left := total_amount_with_cash - down_payment
  sorry
}

end gloria_money_left_calculation_l460_460617


namespace number_of_four_digit_numbers_is_180_l460_460759

def select_odd_even_count (s : set ℕ) : Prop :=
    ∃ (odd_even_numbers : list ℕ), 
    odd_even_numbers ⊆ s ∧ odd_even_numbers.length = 4 ∧ 
    (countp (λ x, x % 2 = 1) odd_even_numbers = 2) ∧                   -- Two odd numbers
    (countp (λ x, x % 2 = 0) odd_even_numbers = 2) ∧                   -- Two even numbers
    (nodup odd_even_numbers)                                          -- No repeating digits

def count_valid_four_digit_numbers : ℕ := 180                         -- The correct answer is 180

theorem number_of_four_digit_numbers_is_180 : 
    select_odd_even_count {0, 1, 2, 3, 4, 5} → count_valid_four_digit_numbers = 180 :=
by sorry

end number_of_four_digit_numbers_is_180_l460_460759


namespace bushes_needed_to_surround_patio_l460_460271

noncomputable def number_of_bushes (r : ℝ) (d : ℝ) : ℕ :=
  let circumference := 2 * real.pi * r
  in nat.ceil (circumference / d)

theorem bushes_needed_to_surround_patio
  (r : ℝ) (d : ℝ)
  (hr : r = 15)
  (hd : d = 2) :
  number_of_bushes r d = 47 := 
by
  rw [hr, hd]
  simp
  -- use high-precision value of pi here
  sorry

end bushes_needed_to_surround_patio_l460_460271


namespace water_level_rise_l460_460415

theorem water_level_rise (large_side medium_side small_side : ℝ) 
  (medium_rise small_rise : ℝ) :
  large_side = 6 ∧ medium_side = 3 ∧ small_side = 2 ∧ 
  medium_rise = 0.006 ∧ small_rise = 0.004 →
  let V_medium := medium_side^2 * medium_rise,
      V_small := small_side^2 * small_rise,
      V_total := V_medium + V_small,
      large_area := large_side^2,
      h_rise := V_total / large_area in
  h_rise * 1000 = 35 / 18 := 
by
  sorry

end water_level_rise_l460_460415


namespace triangle_is_right_l460_460040

theorem triangle_is_right {x : ℝ} (hx : 0 < x) : 
  let a := 3 * x
      b := 4 * x
      c := 5 * x
  in a^2 + b^2 = c^2 := 
by
  let a := 3 * x
  let b := 4 * x
  let c := 5 * x
  calc
    a^2 + b^2 = (3 * x)^2 + (4 * x)^2  : by sorry
          ... = 9 * x^2 + 16 * x^2     : by sorry
          ... = 25 * x^2              : by sorry
          ... = (5 * x)^2             : by sorry
          ... = c^2                   : by sorry

end triangle_is_right_l460_460040


namespace num_of_elements_l460_460022

-- Lean statement to define and prove the problem condition
theorem num_of_elements (n S : ℕ) (h1 : (S + 26) / n = 5) (h2 : (S + 36) / n = 6) : n = 10 := by
  sorry

end num_of_elements_l460_460022


namespace cube_root_of_product_is_integer_l460_460364

theorem cube_root_of_product_is_integer :
  (∛(2^9 * 5^3 * 7^3) = 280) :=
sorry

end cube_root_of_product_is_integer_l460_460364


namespace min_moves_queens_switch_places_l460_460352

-- Assume a type representing the board positions
inductive Position where
| first_rank | last_rank 

-- Assume a type representing the queens
inductive Queen where
| black | white

-- Function to count minimum moves for switching places
def min_moves_to_switch_places : ℕ :=
  sorry

theorem min_moves_queens_switch_places :
  min_moves_to_switch_places = 23 :=
  sorry

end min_moves_queens_switch_places_l460_460352


namespace problem_a_problem_b_l460_460089

noncomputable section

-- Definition and Conditions for Part (a)
def PartA (f : ℝ × ℝ → ℝ) (g : ℝ → ℝ) (X Y : ℝ) (R θ : ℝ) :=
  (∀ x y, f(x, y) = g(x^2 + y^2)) ∧
  (X = R * Real.cos θ ∧ Y = R * Real.sin θ)

-- Statement for Part (a)
theorem problem_a {f g : ℝ × ℝ → ℝ} {X Y R θ : ℝ} (h : PartA f g X Y R θ) :
  Independent R θ ∧ UniformlyDistributed θ (0, 2 * Real.pi) := sorry

-- Definition and Conditions for Part (b)
def PartB (f : ℝ × ℝ → ℝ) (g : ℝ → ℝ) (α : ℝ) (X Y Xα Yα : ℝ) :=
  (∀ x y, f(x, y) = g(x^2 + y^2)) ∧
  (Xα = X * Real.cos α + Y * Real.sin α ∧ Yα = -X * Real.sin α + Y * Real.cos α) ∧
  (∀ α ∈ Set.Icc 0 (2 * Real.pi), (Xα, Yα) =d= (X, Y))

-- Statement for Part (b)
theorem problem_b {f g : ℝ × ℝ → ℝ} {α X Y Xα Yα : ℝ} (h : PartB f g α X Y Xα Yα) :
  InvariantUnderRotation f (X, Y) ↔ (∀ x y, f(x, y) = g(x^2 + y^2)) := sorry

end problem_a_problem_b_l460_460089


namespace g_at_2_l460_460387

-- Assuming g is a function from ℝ to ℝ such that it satisfies the given condition.
def g : ℝ → ℝ := sorry

-- Condition of the problem
axiom g_condition : ∀ x : ℝ, g (2 ^ x) + x * g (2 ^ (-x)) = 2

-- The statement we want to prove
theorem g_at_2 : g (2) = 0 :=
by
  sorry

end g_at_2_l460_460387


namespace line_equation_l460_460031

theorem line_equation {k b : ℝ} 
  (h1 : (∀ x : ℝ, k * x + b = -4 * x + 2023 → k = -4))
  (h2 : b = -5) :
  ∀ x : ℝ, k * x + b = -4 * x - 5 := by
sorry

end line_equation_l460_460031


namespace remainder_of_N_div_1000_l460_460700

-- Defining the set A
def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}.to_finset

-- Defining the condition for N
def N : ℕ :=
  let k_choices := λ k, (7.choose k) * k^(7 - k)
  8 * Finset.sum (Finset.range 7) (λ k, k_choices (k + 1))

-- Theorem that must be proven
theorem remainder_of_N_div_1000 : N % 1000 = 992 := by
  sorry

end remainder_of_N_div_1000_l460_460700


namespace number_of_students_suggested_mashed_potatoes_l460_460012

theorem number_of_students_suggested_mashed_potatoes 
    (students_suggested_bacon : ℕ := 374) 
    (students_suggested_tomatoes : ℕ := 128) 
    (total_students_participated : ℕ := 826) : 
    (total_students_participated - (students_suggested_bacon + students_suggested_tomatoes)) = 324 :=
by sorry

end number_of_students_suggested_mashed_potatoes_l460_460012


namespace number_of_homologous_functions_l460_460088

-- Given a function f(x) = x^2
def f : ℝ → ℝ := λ x, x^2

-- Definition of the range set
def range_set : Set ℝ := {1, 9}

-- Definition of the possible values in the domain set
def domain_values : Set ℝ := {-1, 1, -3, 3}

-- Function to check homologous functions
def homologous_functions (domain : Set ℝ) : Bool :=
  ∃ f, domain ⊆ domain_values ∧ Set.image f domain = range_set

-- Proving the number of homologous functions with different domains
theorem number_of_homologous_functions : 
  (∃ f, homologous_functions domain_values) → True :=
by
  sorry

end number_of_homologous_functions_l460_460088


namespace horizontal_asymptote_of_rational_function_l460_460888

noncomputable def horizontal_asymptote (f : ℝ → ℝ) := 
  lim atTop (fun x => f x)

theorem horizontal_asymptote_of_rational_function :
  horizontal_asymptote (λ x, (8 * x^2 - 4) / (4 * x^2 + 2 * x - 1)) = 2 :=
sorry

end horizontal_asymptote_of_rational_function_l460_460888


namespace bottles_in_case_l460_460457

def first_group := 14
def second_group := 16
def third_group := 12
def fourth_group := (first_group + second_group + third_group) / 2
def total_groups := 4
def camp_days := 3
def cases_purchased := 13
def more_bottles_needed := 255
def bottles_per_day_consumed_by_each_child := 3
      
-- Calculate the total number of children
def total_children := first_group + second_group + third_group + fourth_group

-- Calculate total bottle consumption
def consumption_per_child := camp_days * bottles_per_day_consumed_by_each_child
def total_bottle_consumption := total_children * consumption_per_child

-- Proof condition
def purchased_bottles (cases_bought num_bottles_per_case : ℕ) := cases_bought * num_bottles_per_case
def remaining_bottles := total_bottle_consumption - purchased_bottles cases_purchased 24
def total_bottles_needed := purchased_bottles cases_purchased 24 + more_bottles_needed

theorem bottles_in_case : ∀ (X : ℕ), total_bottle_consumption = (cases_purchased * X) + more_bottles_needed → X = 24 :=
by {
  intros,
  sorry
}

end bottles_in_case_l460_460457


namespace value_of_h_l460_460982

theorem value_of_h (h : ℤ) : (-1)^3 + h * (-1) - 20 = 0 → h = -21 :=
by
  intro h_cond
  sorry

end value_of_h_l460_460982


namespace neg_q_is_true_l460_460948

variable (p q : Prop)

theorem neg_q_is_true (hp : p) (hq : ¬ q) : ¬ q :=
by
  exact hq

end neg_q_is_true_l460_460948


namespace remainder_is_400_l460_460695

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def num_functions (f : ℕ → ℕ) : ℕ :=
  if ∃ c ∈ A, ∀ x ∈ A, f(f(x)) = c then 1 else 0

def N : ℕ :=
  8 * (7 + 672 + 1701 + 1792 + 875 + 252 + 1)

def remainder : ℕ :=
  N % 1000

theorem remainder_is_400 : remainder = 400 :=
  sorry

end remainder_is_400_l460_460695


namespace initial_volume_solution_l460_460461

variable (V : ℝ)

theorem initial_volume_solution
  (h1 : 0.35 * V + 1.8 = 0.50 * (V + 1.8)) :
  V = 6 :=
by
  sorry

end initial_volume_solution_l460_460461


namespace symmetric_cross_is_A_l460_460200

-- Define the original cross shape and its reflection properties
def original_cross : Type := sorry -- Type defining the cross shape
def diagonal_reflection (s : original_cross) : original_cross := sorry -- Function for diagonal reflection

-- Options
def option_A : original_cross := sorry -- Define cross shape for option A (reflected)
def option_B : original_cross := sorry -- Define cross shape for option B (rotated)
def option_C : original_cross := sorry -- Define cross shape for option C (swapped top right and bottom left)
def option_D : original_cross := sorry -- Define inverted cross shape (D)
def option_E : original_cross := sorry -- Define cross shape with no changes (E)

-- The theorem stating the correct option
theorem symmetric_cross_is_A : 
  diagonal_reflection original_cross = option_A ∧ 
  diagonal_reflection original_cross ≠ option_B ∧ 
  diagonal_reflection original_cross ≠ option_C ∧ 
  diagonal_reflection original_cross ≠ option_D ∧ 
  diagonal_reflection original_cross ≠ option_E :=
by 
  sorry

end symmetric_cross_is_A_l460_460200


namespace solve_for_a_l460_460384

noncomputable def parabola_directrix_distance (a : ℝ) (h : a > 0) : Prop :=
  let directrix := -1 / (4 * a) in
  abs(2 - directrix) = 4

theorem solve_for_a (a : ℝ) (h : a > 0) :
  parabola_directrix_distance a h → a = 1/8 :=
by
  intro h_dist
  sorry

end solve_for_a_l460_460384


namespace range_of_a_l460_460951

theorem range_of_a 
  (f : ℝ → ℝ)
  (h_even : ∀ x, -5 ≤ x ∧ x ≤ 5 → f x = f (-x))
  (h_decreasing : ∀ a b, 0 ≤ a ∧ a < b ∧ b ≤ 5 → f b < f a)
  (h_inequality : ∀ a, f (2 * a + 3) < f a) :
  ∀ a, -5 ≤ a ∧ a ≤ 5 → a ∈ (Set.Icc (-4) (-3) ∪ Set.Ioc (-1) 1) := 
by
  sorry

end range_of_a_l460_460951


namespace max_sum_at_n_8_l460_460209

variable {a : Nat → Int} -- Assuming a_n: ℕ → ℤ (a sequence from natural numbers to integers)
variable {S : Nat → Int} -- Assuming S_n: ℕ → ℤ (sum of first n terms)
variable {a1 d : Int}    -- a1: first term, d: common difference

def is_arithmetic_sequence (a : Nat → Int) (a1 d : Int) : Prop :=
  ∀ n, a (n + 1) = a n + d

def S_n (a : Nat → Int) (n : Nat) : Int :=
  ∑ i in Finset.range n, a (i + 1)

def maximizes_S_n (a : Nat → Int) (n : Nat) : Prop :=
  ∀ k : Nat, S_n a n ≥ S_n a k

def conditions (a : Nat → Int) (S : Nat → Int) (a1 d : Int) : Prop :=
  is_arithmetic_sequence a a1 d ∧ S 16 > 0 ∧ S 17 < 0

theorem max_sum_at_n_8 (a : Nat → Int) (S : Nat → Int) (a1 d : Int) :
  conditions a S a1 d → maximizes_S_n a 8 :=
by
  intros h
  sorry

end max_sum_at_n_8_l460_460209


namespace given_condition_implies_result_l460_460985

theorem given_condition_implies_result (a : ℝ) (h : a ^ 2 + 2 * a = 1) : 2 * a ^ 2 + 4 * a + 1 = 3 :=
sorry

end given_condition_implies_result_l460_460985


namespace james_additional_votes_needed_l460_460659

theorem james_additional_votes_needed 
  (total_votes : ℕ) 
  (james_percent : ℝ) 
  (win_percent : ℝ)
  (H1 : total_votes = 5000) 
  (H2 : james_percent = 0.5) 
  (H3 : win_percent = 60): 
  let james_votes := (james_percent / 100) * total_votes,
      required_votes := (win_percent / 100) * total_votes,
      votes_needed_to_win := required_votes + 1,
      additional_votes := votes_needed_to_win - james_votes in
  additional_votes = 2976 :=
begin
  sorry
end

end james_additional_votes_needed_l460_460659


namespace expected_energy_24x24_grid_48_toggles_l460_460310

-- Statement of the problem
theorem expected_energy_24x24_grid_48_toggles :
  let n := 24
  let total_toggles := 48
  let initial_state := λ (i j : Fin n), false

  -- Each light uses 1 kiloJoule of energy per minute while on
  let energy_per_light := 1
  let expected_energy := 9408

  -- Compute the expected energy
  let energy_expended := sorry -- Compute the energy expended based on toggles

  -- Prove that the expected value of the total amount of energy is 9408 kJ
  energy_expended = expected_energy :=
begin
  -- Use the provided conditions and the formula for expected energy
  sorry
end

end expected_energy_24x24_grid_48_toggles_l460_460310


namespace polynomial_division_correct_l460_460911

noncomputable def dividend : Polynomial ℤ := 8 * X^3 - 4 * X^2 + 6 * X - 3
noncomputable def divisor : Polynomial ℤ := X - 1
noncomputable def quotient : Polynomial ℤ := 8 * X^2 + 4 * X + 10

theorem polynomial_division_correct :
  (dividend / divisor).quotient = quotient :=
sorry

end polynomial_division_correct_l460_460911


namespace problem_solution_l460_460036

def otimes (a b : ℚ) : ℚ := (a ^ 3) / (b ^ 2)

theorem problem_solution :
  (otimes (otimes 2 3) 4) - (otimes 2 (otimes 3 4)) = (-2016) / 729 := by
  sorry

end problem_solution_l460_460036


namespace transformation_correct_l460_460998

def original_function (x : ℝ) : ℝ := 1/2 * real.sin x

def transformed_function (x : ℝ) : ℝ := 1/2 * real.sin (2 * x + real.pi / 2) + 1

theorem transformation_correct :
  (∀ x : ℝ, transformed_function x = original_function ((x - real.pi / 2) / 2) + 1) :=
by
  sorry

end transformation_correct_l460_460998


namespace singer_arrangement_l460_460069

theorem singer_arrangement (singers : Fin 6 → Prop) (A B C : Fin 6) :
  (∀ i, singers i) → 
  (∀ i j k, i < j → j < k → (singers i = A ∧ singers j = B ∧ singers k = C) ∨ 
    (singers k = A ∧ singers j = B ∧ singers i = C) ∨ 
    (singers i = B ∧ singers j = C ∧ singers k = A) ∨ 
    (singers k = B ∧ singers j = C ∧ singers i = A)) →
  (∃ n : Nat, n = 4 * factorial 6 / factorial 3 ∧ n = 480) :=
by 
  sorry

end singer_arrangement_l460_460069


namespace lines_through_point_l460_460837

theorem lines_through_point (P : ℝ × ℝ) (hP : P = (2, 3)) :
  ∃ (l1 l2 : ℝ → ℝ) (k1 k2 : ℝ), (l1 = λ x, k1 * x + 3 - 2 * k1) ∧ (l2 = λ x, k2 * x + 3 - 2 * k2) ∧ 
  ((k1 = -1 ∨ k2 = -1) ∧ (P = (2, 3))) ∧ l1 ≠ l2 :=
sorry

end lines_through_point_l460_460837


namespace geom_seq_decreasing_l460_460203

variable {a : ℕ → ℝ}
variable {a₁ q : ℝ}

theorem geom_seq_decreasing (h : ∀ n, a n = a₁ * q^n) (h₀ : a₁ * (q - 1) < 0) (h₁ : q > 0) :
  ∀ n, a (n + 1) < a n := 
sorry

end geom_seq_decreasing_l460_460203


namespace calculate_total_people_l460_460145

-- Definitions given in the problem
def cost_per_adult_meal := 3
def num_kids := 7
def total_cost := 15

-- The target property to prove
theorem calculate_total_people : 
  (total_cost / cost_per_adult_meal) + num_kids = 12 := 
by 
  sorry

end calculate_total_people_l460_460145


namespace reciprocal_of_neg_two_l460_460783

theorem reciprocal_of_neg_two : ∀ x : ℝ, x = -2 → (1 / x) = -1 / 2 :=
by
  intro x h
  rw [h]
  norm_num

end reciprocal_of_neg_two_l460_460783


namespace largest_common_divisor_gcd_correct_l460_460522

theorem largest_common_divisor (d : ℕ) (h1 : d ∣ 420) (h2 : d ∣ 385) : d ≤ 35 :=
begin
  sorry
end

noncomputable def largest_common_divisor_is_35 : ℕ :=
  Nat.gcd 420 385

theorem gcd_correct : largest_common_divisor_is_35 = 35 :=
begin
  sorry
end

end largest_common_divisor_gcd_correct_l460_460522


namespace converse_not_true_l460_460327

-- Definitions and assumptions
variables {O : Point} {a b c : Line} {α β : Plane}

-- The problem statement to be proved
theorem converse_not_true
  (hab : a ∩ b = O)
  (a_in_α : a ⊂ α)
  (b_in_α : b ⊂ α)
  (b_perp_β : b ⊥ β) :
  ¬(α ⊥ β → b ⊥ β) :=
sorry

end converse_not_true_l460_460327


namespace determine_symmetry_l460_460956

def quadratic_function_is_symmetric_about_y (y : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x : ℝ, y (-x) = y x

theorem determine_symmetry (m : ℝ) :
  quadratic_function_is_symmetric_about_y (λ x, m * x^2 + (m - 2) * x + 2) m ↔ m = 2 :=
by
  sorry

end determine_symmetry_l460_460956


namespace smallest_difference_of_factors_l460_460262

theorem smallest_difference_of_factors (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 2268) : 
  (a = 42 ∧ b = 54) ∨ (a = 54 ∧ b = 42) := sorry

end smallest_difference_of_factors_l460_460262


namespace arithmetic_sequence_fourth_term_l460_460907

theorem arithmetic_sequence_fourth_term (x : ℝ) (h : 0 < x) :
  let a := log 3 (2 * x),
      b := log 3 (3 * x),
      c := log 3 (4 * x + 2)
  in a + c = 2 * b → x = 4 → log 3 (x * 18 * 3 / 2) = 3 :=
by
  intros a b c h_seq h_x
  have := sorry

end arithmetic_sequence_fourth_term_l460_460907


namespace proof_problem_l460_460602

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem proof_problem {a b c : ℝ} :
  quadratic_function a b c 2 = 0.35 →
  quadratic_function a b c 4 = 0.35 →
  quadratic_function a b c 5 = 3 →
  (a + b + c) * ((-b + real.sqrt (b^2 - 4 * a * c)) / (2 * a) + (-b - real.sqrt (b^2 - 4 * a * c)) / (2 * a)) = 18 :=
by
  intros h1 h2 h3
  sorry

end proof_problem_l460_460602


namespace integer_pairs_satisfy_equation_l460_460167

theorem integer_pairs_satisfy_equation :
  ∀ (a b : ℤ), b + 1 ≠ 0 → b + 2 ≠ 0 → a + b + 1 ≠ 0 →
    ( (a + 2)/(b + 1) + (a + 1)/(b + 2) = 1 + 6/(a + b + 1) ↔ 
      (a = 1 ∧ b = 0) ∨ (∃ t : ℤ, t ≠ 0 ∧ t ≠ -1 ∧ a = -3 - t ∧ b = t) ) :=
by
  intros a b h1 h2 h3
  sorry

end integer_pairs_satisfy_equation_l460_460167


namespace smallest_positive_period_f_zeros_of_f_max_min_values_f_in_interval_l460_460238

noncomputable def f (x : ℝ) : ℝ := 4 * sin x * cos (x - (π / 3)) - sqrt 3

theorem smallest_positive_period_f :
  ∃ (T : ℝ), T = π ∧ ∀ x : ℝ, f (x + T) = f x :=
sorry

theorem zeros_of_f :
  ∃ (k : ℤ), ∀ x : ℝ, f x = 0 ↔ ∃ k : ℤ, x = (π / 6) + k * (π / 2) :=
sorry

theorem max_min_values_f_in_interval :
  ∃ (max min : ℝ), 
    (∃ x ∈ (set.Icc (π / 24) (3 * π / 4)), f x = max) ∧ 
    (∃ y ∈ (set.Icc (π / 24) (3 * π / 4)), f y = min) ∧ 
    max = 2 ∧ min = - sqrt 2 :=
sorry

end smallest_positive_period_f_zeros_of_f_max_min_values_f_in_interval_l460_460238


namespace triangle_inequality_for_beads_l460_460072

noncomputable def there_exist_three_consecutive_beads_that_form_triangle : Prop :=
  ∃ (a : Fin 1734 → ℕ), 
  (∀ n, 290 ≤ a n ∧ a n ≤ 2023 ∧ ∀ m1 m2, m1 ≠ m2 → a m1 ≠ a m2) ∧
  ∃ i : Fin 1734, let x := a i, y := a (i + 1), z := a (i + 2) in
  x + y > z ∧ x + z > y ∧ y + z > x

theorem triangle_inequality_for_beads : there_exist_three_consecutive_beads_that_form_triangle :=
by
  sorry

end triangle_inequality_for_beads_l460_460072


namespace determine_symmetry_l460_460957

def quadratic_function_is_symmetric_about_y (y : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x : ℝ, y (-x) = y x

theorem determine_symmetry (m : ℝ) :
  quadratic_function_is_symmetric_about_y (λ x, m * x^2 + (m - 2) * x + 2) m ↔ m = 2 :=
by
  sorry

end determine_symmetry_l460_460957


namespace equilateral_triangle_ab_l460_460778

-- Define the points and the condition of equilateral triangle
def is_equilateral_triangle (a b : ℝ) : Prop :=
  let z1 := (0 : ℂ)
  let z2 := (a + 19 * complex.I : ℂ)
  let z3 := (b + 61 * complex.I : ℂ)
  ∃ r : ℂ, abs (z2 - z1) = abs (z3 - z1) ∧ abs (z3 - z2) = abs (z2 - z1)

-- Main theorem stating the problem
theorem equilateral_triangle_ab (a b : ℝ) (h : is_equilateral_triangle a b) : 
  a * b = 7760 / 9 :=
  sorry

end equilateral_triangle_ab_l460_460778


namespace pentagon_area_l460_460476

theorem pentagon_area
  (a b c d e : ℝ)
  (r s : ℝ)
  (habcd: set {14.0, 21.0, 22.0, 28.0, 35.0} = {a, b, c, d, e})
  (hpythagorean: r^2 + s^2 = e^2)
  (hr: r = b - d)
  (hs: s = c - a) :
  b * c - (1 / 2) * r * s = 759.5 :=
sorry

end pentagon_area_l460_460476


namespace shelby_drove_in_windy_conditions_for_20_minutes_l460_460007

def shelby_windy_drive_time (total_distance : ℝ) (total_time : ℝ) (speed_non_windy : ℝ) (speed_windy : ℝ) : ℝ :=
  let x := (12*total_distance - 8*total_time) / 3 in
  if x < 0 ∨ x > total_time then 0 else x

theorem shelby_drove_in_windy_conditions_for_20_minutes :
  shelby_windy_drive_time 25 45 (40/60) (25/60) = 20 := by
  sorry

end shelby_drove_in_windy_conditions_for_20_minutes_l460_460007


namespace kamals_weighted_average_l460_460315

def marks : List (String × ℕ) := [("English", 96), ("Mathematics", 65), ("Physics", 82), ("Chemistry", 67), ("Biology", 85)]
def weights : List (String × ℝ) := [("English", 0.25), ("Mathematics", 0.20), ("Physics", 0.30), ("Chemistry", 0.15), ("Biology", 0.10)]

def weighted_average (marks : List (String × ℕ)) (weights : List (String × ℝ)) : ℝ :=
  (List.sum (List.map (λ (subj_mark : String × ℕ) =>
    let (subject, mark) := subj_mark
    let weight := (weights.find (λ (subj_weight : String × ℝ) => subj_weight.1 = subject)).getD 0.0
    mark * weight) marks)) / 1 -- Total weight is 1

theorem kamals_weighted_average :
  weighted_average marks weights = 80.15 := by
  sorry

end kamals_weighted_average_l460_460315


namespace color_3x3_grid_l460_460514

theorem color_3x3_grid :
  let grid := { (i, j) : Fin 3 × Fin 3 // true } in
  let no_shared_edge (cells : Finset (Fin 3 × Fin 3)) := ∀ x y ∈ cells, x ≠ y → 
    (|x.1 - y.1|, |x.2 - y.2|) ≠ (1, 0) ∧ (|x.1 - y.1|, |x.2 - y.2|) ≠ (0, 1) in
  Finset.card {cells : Finset (Fin 3 × Fin 3) // cells.card = 3 ∧ no_shared_edge cells} = 22 :=
begin
  sorry -- Proof to be implemented
end

end color_3x3_grid_l460_460514


namespace valid_paths_total_l460_460254

-- Define the points as an inductive type
inductive Point
| A | B | C | D | E | F | G

open Point

-- Define a valid path (list of points) that does not revisit any points and goes from A to B
def isValidPath (path : List Point) : Prop :=
  path.head? = some A ∧ path.getLast? = some B ∧ List.nodup path

-- Total number of valid paths (to prove it equals 13)
def numberOfValidPaths : Nat :=
  13

-- To prove:
theorem valid_paths_total : ∃ (paths : List (List Point)), 
  (∀ p ∈ paths, isValidPath p) ∧ paths.length = numberOfValidPaths := by
  sorry

end valid_paths_total_l460_460254


namespace compare_a_b_l460_460264

variable (m : ℝ)
variable (a : ℝ)
variable (b : ℝ)

noncomputable def sqrt_real := λ x : ℝ, Real.sqrt x

theorem compare_a_b (hm : m > 1)
  (ha : a = sqrt_real m - sqrt_real (m - 1))
  (hb : b = sqrt_real (m + 1) - sqrt_real m) :
  a > b :=
sorry

end compare_a_b_l460_460264


namespace buddy_thursday_cards_l460_460350

-- Definitions from the given conditions
def monday_cards : ℕ := 30
def tuesday_cards : ℕ := monday_cards / 2
def wednesday_cards : ℕ := tuesday_cards + 12
def thursday_extra_cards : ℕ := tuesday_cards / 3
def thursday_cards : ℕ := wednesday_cards + thursday_extra_cards

-- Theorem to prove the total number of baseball cards on Thursday
theorem buddy_thursday_cards : thursday_cards = 32 :=
by
  -- Proof steps would go here, but we just provide the result for now
  sorry

end buddy_thursday_cards_l460_460350


namespace prop_necessity_sufficiency_l460_460043

theorem prop_necessity_sufficiency (p q : Prop) (hpq : ¬(p ∧ q)) : (¬(p ∨ q) → (p = False ∧ q = False)) ∧ ¬(¬(p ∨ q) ← (p = False ∧ q = False)) :=
by
  sorry

end prop_necessity_sufficiency_l460_460043


namespace multiply_repeating_decimals_l460_460900

noncomputable def repeating_decimal_03 : ℚ := 1 / 33
noncomputable def repeating_decimal_8 : ℚ := 8 / 9

theorem multiply_repeating_decimals : repeating_decimal_03 * repeating_decimal_8 = 8 / 297 := by 
  sorry

end multiply_repeating_decimals_l460_460900


namespace monotonic_intervals_max_min_values_on_interval_l460_460240

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem monotonic_intervals :
  (∀ x > -2, 0 < (x + 2) * Real.exp x) ∧ (∀ x < -2, (x + 2) * Real.exp x < 0) :=
by
  sorry

theorem max_min_values_on_interval :
  let a := -4
  let b := 0
  let f_a := (-4 + 1) * Real.exp (-4)
  let f_b := (0 + 1) * Real.exp 0
  let f_c := (-2 + 1) * Real.exp (-2)
  (f b = 1) ∧ (f_c = -1 / Real.exp 2) ∧ (f_a < f_b) ∧ (f_a < f_c) ∧ (f_c < f_b) :=
by
  sorry

end monotonic_intervals_max_min_values_on_interval_l460_460240


namespace largest_consecutive_sum_is_nine_l460_460056

-- Define the conditions: a sequence of positive consecutive integers summing to 45
def is_consecutive_sum (n k : ℕ) : Prop :=
  (k > 0) ∧ (n > 0) ∧ ((k * (2 * n + k - 1)) = 90)

-- The theorem statement proving k = 9 is the largest
theorem largest_consecutive_sum_is_nine :
  ∃ n k : ℕ, is_consecutive_sum n k ∧ ∀ k', is_consecutive_sum n k' → k' ≤ k :=
sorry

end largest_consecutive_sum_is_nine_l460_460056


namespace find_r_for_two_roots_greater_than_neg_one_l460_460181

theorem find_r_for_two_roots_greater_than_neg_one :
  ∀ r : ℝ, (3.5 < r ∧ r < 4.5) ↔
  (let a := r - 4,
       b := -2 * (r - 3),
       c := r,
       discriminant := b^2 - 4 * a * c in
   discriminant > 0 ∧ 
   let vertex := -b / (2 * a) in
   vertex > -1 ∧ 
   a * (-1)^2 + b * (-1) + c > 0) :=
sorry

end find_r_for_two_roots_greater_than_neg_one_l460_460181


namespace construct_triangle_l460_460411

-- Definitions of altitudes and median
variables {α : Type*} [MetricSpace α] [NormedGroup α] [NormedSpace ℝ α]

-- Assume permutation symmetry of vertices and conditions in triangle
noncomputable def triangle_construction (A B C : α) (ma mb sa : ℝ) : Prop :=
  is_altitude A B C ma ∧
  is_altitude B A C mb ∧
  is_median A B C sa

-- The main theorem stating feasibility and correctness of construction
theorem construct_triangle {A B C : α} {ma mb sa : ℝ}
  (h1: is_altitude A B C ma) 
  (h2: is_altitude B A C mb) 
  (h3: is_median A B C sa):
  ∃ A' B' C', triangle_construction A' B' C' ma mb sa :=
sorry

end construct_triangle_l460_460411


namespace triangle_shape_and_area_l460_460946

theorem triangle_shape_and_area (a b c : ℝ) (h : a^2 + b^2 + c^2 + 50 = 6a + 8b + 10c) :
  (a = 3 ∧ b = 4 ∧ c = 5) ∧ △right_ang_abc : Prop :=
begin
  unfold △right_ang_abc,
  sorry
end

end triangle_shape_and_area_l460_460946


namespace area_of_inscribed_triangle_l460_460858

noncomputable def triangle_area_inscribed (arc1 arc2 arc3 : ℝ) (r : ℝ) : ℝ :=
  let θ1 := (arc1 / (2 * π * r)) * (2 * π) in
  let θ2 := (arc2 / (2 * π * r)) * (2 * π) in
  let θ3 := (arc3 / (2 * π * r)) * (2 * π) in
  1 / 2 * r^2 * (Real.sin θ1 + Real.sin θ2 + Real.sin θ3)

theorem area_of_inscribed_triangle {arc1 arc2 arc3 : ℝ}
  (h1 : arc1 = 5) (h2 : arc2 = 6) (h3 : arc3 = 7) :
  triangle_area_inscribed arc1 arc2 arc3 (9 / π) = 101.2488 / π^2 := by
  sorry

end area_of_inscribed_triangle_l460_460858


namespace train_has_117_cars_l460_460001

-- Definitions for conditions
def cars_counted_first_15_seconds : ℕ := 9
def time_first_15_seconds : ℕ := 15
def total_time_minutes : ℕ := 3
def additional_seconds : ℕ := 15
def total_time_seconds : ℕ := total_time_minutes * 60 + additional_seconds
def cars_per_second : ℝ := cars_counted_first_15_seconds / (time_first_15_seconds : ℝ)

-- Theorem to prove
theorem train_has_117_cars : cars_per_second * (total_time_seconds : ℝ) = 117 := by
  sorry

end train_has_117_cars_l460_460001


namespace min_pairs_of_friends_l460_460474

variables (n : ℕ) (invites_per_person : ℕ) (pairs_of_friends : ℕ)
variable (people_invited : ℕ)

theorem min_pairs_of_friends :
  n = 2000 ∧ invites_per_person = 1000 ∧ 
  (∀ i j, i ≠ j → a_ij = (invited_by i j + invited_by j i ∈ {0, 1, 2}) ) → 
  pairs_of_friends ≥ 1000 :=
begin
  sorry
end

end min_pairs_of_friends_l460_460474


namespace alternating_series_sum_l460_460897

theorem alternating_series_sum : 
  (∑ k in Finset.range 2022, (-1)^(k+1)) = 0 := 
by
  sorry

end alternating_series_sum_l460_460897


namespace sum_of_possible_values_l460_460950

theorem sum_of_possible_values (m : ℤ) (h : 0 < 5 * m ∧ 5 * m < 40) : (∑ k in finset.Icc 1 7, k) = 28 :=
by
  sorry

end sum_of_possible_values_l460_460950


namespace range_g_l460_460332

noncomputable theory

open Real

def g (x y z : ℝ) : ℝ := x / (2 * x + y) + y / (2 * y + z) + z / (2 * z + x)

theorem range_g (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 0 < g x y z ∧ g x y z ≤ 1 :=
sorry

end range_g_l460_460332


namespace find_AB_distance_l460_460293

noncomputable def curve_C1_param_eq (α : ℝ) : (ℝ × ℝ) :=
  (√7 * cos α, 2 + √7 * sin α)

def curve_C1_gen_eq (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 7

def curve_C2_gen_eq (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

def curve_C2_polar_eq (ρ θ : ℝ) : Prop :=
  (ρ * cos θ - 1)^2 + (ρ * sin θ)^2 = 1

lemma curve_C2_polar_simplified : ∀ θ, curve_C2_polar_eq (2 * cos θ) θ :=
  by sorry

theorem find_AB_distance :
  let ray_theta := π / 6
  let ρ1_solution := 3
  let ρ2_solution := sqrt 3
  | ρ1_solution - ρ2_solution | = 3 - sqrt 3 :=
  by sorry

end find_AB_distance_l460_460293


namespace expected_value_of_winnings_eq_3_point_5_l460_460111

noncomputable def expected_winnings : ℝ :=
  ∑ i in finset.range 8, (8 - i) * (1 / 8 : ℝ)

theorem expected_value_of_winnings_eq_3_point_5 :
  expected_winnings = 3.5 :=
sorry

end expected_value_of_winnings_eq_3_point_5_l460_460111


namespace remainder_of_N_div_1000_l460_460702

-- Defining the set A
def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}.to_finset

-- Defining the condition for N
def N : ℕ :=
  let k_choices := λ k, (7.choose k) * k^(7 - k)
  8 * Finset.sum (Finset.range 7) (λ k, k_choices (k + 1))

-- Theorem that must be proven
theorem remainder_of_N_div_1000 : N % 1000 = 992 := by
  sorry

end remainder_of_N_div_1000_l460_460702


namespace stripe_length_is_correct_l460_460101

-- Definitions based on conditions
def circumference : ℝ := 18
def height : ℝ := 8

-- Theorem statement
theorem stripe_length_is_correct : real.sqrt (circumference ^ 2 + height ^ 2) = real.sqrt 388 := by
  sorry

end stripe_length_is_correct_l460_460101


namespace prime_factorization_binomial_l460_460760

-- Definitions from the problem conditions
def binomial (m n : ℕ) : ℕ := m.choose n

noncomputable def legendre (p k : ℕ) : ℕ :=
∑ i in (range (1 + k)).filter (λ i, 0 < i ∧ p ^ i ≤ k), k / (p ^ i)

-- The theorem to be proved
theorem prime_factorization_binomial (m n p : ℕ) (hp: prime p) :
  legendre p m - legendre p n - legendre p (m - n) ≤ m := sorry

end prime_factorization_binomial_l460_460760


namespace transport_capacity_l460_460794

-- Declare x and y as the amount of goods large and small trucks can transport respectively
variables (x y : ℝ)

-- Given conditions
def condition1 : Prop := 2 * x + 3 * y = 15.5
def condition2 : Prop := 5 * x + 6 * y = 35

-- The goal to prove
def goal : Prop := 3 * x + 5 * y = 24.5

-- Main theorem stating that given the conditions, the goal follows
theorem transport_capacity (h1 : condition1 x y) (h2 : condition2 x y) : goal x y :=
by sorry

end transport_capacity_l460_460794


namespace solution_set_l460_460232

def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem solution_set :
  {x | f x ≥ x^2 - 8 * x + 15} = {2} ∪ {x | x > 6} :=
by
  sorry

end solution_set_l460_460232


namespace range_of_a_l460_460601

noncomputable def g (a x : ℝ) : ℝ := a - x^3
noncomputable def h (x : ℝ) : ℝ := 3 * Real.log x
noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x - x^3

theorem range_of_a (e : ℝ) (a : ℝ) (e_pos : 0 < e) (He : e = Real.exp 1) (x₁ x₂ : ℝ) (hx₁ : x₁ = 1 / e) (hx₂ : x₂ = e)
    (H : ∀ x, x ∈ Set.Icc (1 / e) e → g a x = -h x) : 
    1 ≤ a ∧ a ≤ e^3 - 3 :=
by
  have hxy : ∃ x ∈ Set.Icc (1 / e) e, g a x = -h x, from sorry
  sorry

end range_of_a_l460_460601


namespace radius_of_each_shot_l460_460632

theorem radius_of_each_shot (r_original : ℝ) (n_shots : ℕ) (V : ℝ → ℝ) : 
  r_original = 6 ∧ n_shots = 216 ∧ 
  (∀ r, V r = (4 / 3) * Real.pi * r^3) →
  ∃ r_shot : ℝ, V(r_shot) * n_shots = V(r_original) ∧ r_shot = 1 :=
by 
  intros h
  sorry

end radius_of_each_shot_l460_460632


namespace perimeter_quarter_circular_arcs_l460_460838

theorem perimeter_quarter_circular_arcs (s : ℝ) (h : s = 4 / π) : 
  let arc_perimeter := 2 * π * (s / 4) in 
  let total_perimeter := 4 * arc_perimeter / (2 * π) in
  total_perimeter = 8 :=
by
  sorry

end perimeter_quarter_circular_arcs_l460_460838


namespace total_distinct_plants_l460_460919

/-- Conditions provided -/
variable (A B C D : Set ℕ)

/-- Number of plants in each bed -/
variable (hA : card A = 550)
variable (hB : card B = 500)
variable (hC : card C = 400)
variable (hD : card D = 350)

/-- Number of plants in intersections -/
variable (hAB : card (A ∩ B) = 60)
variable (hAC : card (A ∩ C) = 110)
variable (hAD : card (A ∩ D) = 70)
variable (hABC : card (A ∩ B ∩ C) = 30)
variable (hBC : card (B ∩ C) = 0)
variable (hBD : card (B ∩ D) = 0)
variable (hCD : card (C ∩ D) = 0)
variable (hABCD : card (A ∩ B ∩ C ∩ D) = 0)

/-- Prove that the total distinct number of plants is 1590 --/
theorem total_distinct_plants : card (A ∪ B ∪ C ∪ D) = 1590 :=
by
  sorry

end total_distinct_plants_l460_460919


namespace probability_of_A_l460_460505

def event_A (x y : ℝ) : Prop := x / 2 + y ≥ 1

noncomputable def probability_event_A : ℝ :=
  ∫ x in 0..1, ∫ y in 0..1, if event_A x y then 1 else 0

theorem probability_of_A : probability_event_A = 0.25 :=
by
  -- proof omitted
  sorry

end probability_of_A_l460_460505


namespace largest_consecutive_sum_is_nine_l460_460055

-- Define the conditions: a sequence of positive consecutive integers summing to 45
def is_consecutive_sum (n k : ℕ) : Prop :=
  (k > 0) ∧ (n > 0) ∧ ((k * (2 * n + k - 1)) = 90)

-- The theorem statement proving k = 9 is the largest
theorem largest_consecutive_sum_is_nine :
  ∃ n k : ℕ, is_consecutive_sum n k ∧ ∀ k', is_consecutive_sum n k' → k' ≤ k :=
sorry

end largest_consecutive_sum_is_nine_l460_460055


namespace question1_question2_l460_460557

variable (α : ℝ)

-- Conditions
def condition1 : Prop := 0 < α ∧ α < π / 2
def condition2 : Prop := sin α = 3 / 5

-- Question 1: 
def expression1 : ℝ := (2 * sin(α)^2 + sin(2 * α)) / cos(2 * α)
def value1 : ℝ := 24 / 7

-- Question 2: 
def expression2 : ℝ := tan(α + 5 * π / 4)
def value2 : ℝ := 7

theorem question1 (h1 : condition1 α) (h2 : condition2 α) : expression1 α = value1 := 
sorry

theorem question2 (h1 : condition1 α) (h2 : condition2 α) : expression2 α = value2 := 
sorry

end question1_question2_l460_460557


namespace eighty_ray_not_fifty_ray_count_l460_460712

noncomputable def unit_square := set.Icc (0 : ℝ) 1 × set.Icc (0 : ℝ) 1

def n_ray_partitional (n : ℕ) (p : ℝ × ℝ) : Prop :=
  -- Assuming specific definitions for n-ray partitional points
  ∃ (rays : fin n → ℝ × ℝ), 
    (∀ (i : fin n), 
      let θ := i.val / n in 
      rays i = (p.1 + cos (θ * 2 * π), p.2 + sin (θ * 2 * π))) ∧
      (∀ (i : fin n), set.finite (points on each ray) ∧ all triangles have equal area )

theorem eighty_ray_not_fifty_ray_count : 
  ∃ c : ℕ, 
    (c = (finset.filter (λ (p : ℝ × ℝ), n_ray_partitional 80 p ∧ ¬ n_ray_partitional 50 p) 
             {p | p ∈ unit_square}).card ∧ c = 6319) :=
sorry

end eighty_ray_not_fifty_ray_count_l460_460712


namespace store_a_full_price_l460_460800

theorem store_a_full_price
  (P : ℝ) -- Full price of the smartphone at store A
  (discount_a : P * (1 - 0.08) = 0.92 * P)
  (price_b : 130)
  (discount_b : price_b * (1 - 0.10) = 117)
  (price_difference : 0.92 * P = 117 - 2) :
  P = 125 :=
by
  sorry

end store_a_full_price_l460_460800


namespace find_custom_operator_result_l460_460718

def custom_operator (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem find_custom_operator_result :
  custom_operator 2 5 = 23 :=
by
  sorry

end find_custom_operator_result_l460_460718


namespace tree_height_at_2_years_l460_460848

theorem tree_height_at_2_years (h : ℕ → ℚ) (h5 : h 5 = 243) 
  (h_rec : ∀ n, h (n - 1) = h n / 3) : h 2 = 9 :=
  sorry

end tree_height_at_2_years_l460_460848


namespace arithmetic_progression_probability_l460_460417

def is_arithmetic_progression (a b c : ℕ) (d : ℕ) : Prop :=
  b - a = d ∧ c - b = d

noncomputable def probability_arithmetic_progression_diff_two : ℚ :=
  have total_outcomes : ℚ := 6 * 6 * 6
  have favorable_outcomes : ℚ := 12
  favorable_outcomes / total_outcomes

theorem arithmetic_progression_probability (d : ℕ) (h : d = 2) :
  probability_arithmetic_progression_diff_two = 1 / 18 :=
by 
  sorry

end arithmetic_progression_probability_l460_460417


namespace number_of_possible_measures_l460_460395

theorem number_of_possible_measures (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
sorry

end number_of_possible_measures_l460_460395


namespace find_positive_A_l460_460713

theorem find_positive_A (A : ℕ) : (A^2 + 7^2 = 130) → A = 9 :=
by
  intro h
  sorry

end find_positive_A_l460_460713


namespace range_of_m_for_monotonicity_l460_460277

def function_is_monotonic_on (f : ℝ → ℝ) (I : set ℝ) : Prop := 
  ∀ x y, x ∈ I → y ∈ I → x < y → f x ≤ f y ∨ f y ≤ f x

def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem range_of_m_for_monotonicity :
  {m : ℝ | function_is_monotonic_on (quadratic 1 (-2*m) 3) (set.Icc 2 3)} = 
  set.Iic 2 ∪ set.Ici 3 :=
sorry

end range_of_m_for_monotonicity_l460_460277


namespace bullet_speed_difference_l460_460433

def bullet_speed_in_same_direction (v_h v_b : ℝ) : ℝ :=
  v_b + v_h

def bullet_speed_in_opposite_direction (v_h v_b : ℝ) : ℝ :=
  v_b - v_h

theorem bullet_speed_difference (v_h v_b : ℝ) (h_h : v_h = 20) (h_b : v_b = 400) :
  bullet_speed_in_same_direction v_h v_b - bullet_speed_in_opposite_direction v_h v_b = 40 :=
by
  rw [h_h, h_b]
  sorry

end bullet_speed_difference_l460_460433


namespace system_of_equations_m_value_l460_460194

theorem system_of_equations_m_value {x y m : ℝ} 
  (h1 : 2 * x + y = 4)
  (h2 : x + 2 * y = m)
  (h3 : x + y = 1) : m = -1 := 
sorry

end system_of_equations_m_value_l460_460194


namespace rationalize_expression_l460_460173

theorem rationalize_expression :
  (2 * Real.sqrt 3) / (Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5) = 
  (Real.sqrt 6 + 3 - Real.sqrt 15) / 2 :=
sorry

end rationalize_expression_l460_460173


namespace solution_of_equation_l460_460536

noncomputable def integer_part (x : ℝ) : ℤ := Int.floor x
noncomputable def fractional_part (x : ℝ) : ℝ := x - integer_part x

theorem solution_of_equation (k : ℤ) (h : -1 ≤ k ∧ k ≤ 5) :
  ∃ x : ℝ, 4 * ↑(integer_part x) = 25 * fractional_part x - 4.5 ∧
           x = k + (8 * ↑k + 9) / 50 := 
sorry

end solution_of_equation_l460_460536


namespace perpendicular_line_to_plane_implies_perpendicular_lines_l460_460118

variable {α : Type*} [Plane α] {Point : Type*} [Point α] {Line : Type*} [Line α]

def is_perpendicular_to_plane (l : Line) (α : Plane) : Prop :=
  ∀ m : Line, is_within_plane m α → ⟪l, m⟫ = ⟪0⟫

def is_within_plane (m : Line) (α : Plane) : Prop :=
  -- Definition for a line being within a plane.
  sorry

def is_perpendicular_to_line (l m : Line) : Prop :=
  ⟪l, m⟫ = ⟪0⟫

theorem perpendicular_line_to_plane_implies_perpendicular_lines
  {l m : Line} {α : Plane} (hl : is_perpendicular_to_plane l α) (hm : is_within_plane m α) :
  is_perpendicular_to_line l m :=
by
  sorry

end perpendicular_line_to_plane_implies_perpendicular_lines_l460_460118


namespace angle_equality_l460_460320

variable {B C A M N P Q : Type*}
variable [AffineSpace B C A M N P Q]

def parallel_lines (l m : AffineSubspace ℝ (EuclideanSpace n)) : Prop :=
  l.direction = m.direction

def on_segment (p a b : AffinePoint ℝ (EuclideanSpace n)) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ a + t • (b - a) = p

def line_parallel_to_base (M N A B C : Type*) [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C]
  (MN BC : AffineSubspace ℝ (EuclideanSpace n))
  (hM : on_segment M A B)
  (hN : on_segment N A C) : Prop :=
  parallel_lines MN BC

theorem angle_equality (ABC : Triangle ℝ)
  (M : EuclideanSpace ℝ n) (N : EuclideanSpace ℝ n)
  (MN BC : AffineSubspace ℝ (EuclideanSpace n))
  (h_parallel : line_parallel_to_base M N ABC.A ABC.B ABC.C MN BC)
  (h_M_on_segment : on_segment M ABC.A ABC.B)
  (h_N_on_segment : on_segment N ABC.A ABC.C)
  (P : EuclideanSpace ℝ n) (h_P_inter : P ∈ (AffineSpan ℝ [ABC.B, N]) ∩ (AffineSpan ℝ [ABC.C, M]))
  (Q : EuclideanSpace ℝ n) (h_Q_second_int : Q ∈ second_intersection_point (circumcircle ABC.B M P) (circumcircle ABC.C N P)) :
  angle ABC.B ABC.A P = angle ABC.C ABC.A Q :=
begin
  sorry
end

end angle_equality_l460_460320


namespace bouquet_combinations_l460_460830

theorem bouquet_combinations :
  ∃ n : ℕ, (∀ r c t : ℕ, 4 * r + 3 * c + 2 * t = 60 → true) ∧ n = 13 :=
sorry

end bouquet_combinations_l460_460830


namespace unique_solution_l460_460334

-- Define the distinct elements a1, ..., an represented as finset (finite set)
variable {n : ℕ} (a : fin n → ℝ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)

-- Define b0, ..., bn-1 as a vector of real numbers
variable (b : fin n → ℝ)

-- Define the system of linear equations
def system (x : fin n → ℝ) :=
  ∀ k : fin n, (∑ i in finset.univ, (a i) ^ k * x i) = b k

-- The theorem states that the system has exactly one solution
theorem unique_solution : ∃! x : fin n → ℝ, system a b x :=
sorry

end unique_solution_l460_460334


namespace tree_height_at_2_years_l460_460847

theorem tree_height_at_2_years (h : ℕ → ℚ) (h5 : h 5 = 243) 
  (h_rec : ∀ n, h (n - 1) = h n / 3) : h 2 = 9 :=
  sorry

end tree_height_at_2_years_l460_460847


namespace factorize_expression_l460_460528

-- Lean 4 statement for the proof problem
theorem factorize_expression (a b : ℝ) : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l460_460528


namespace algorithm_comparable_to_euclidean_l460_460383

-- Define the conditions
def ancient_mathematics_world_leading : Prop := 
  True -- Placeholder representing the historical condition

def song_yuan_algorithm : Prop :=
  True -- Placeholder representing the algorithmic condition

-- The main theorem representing the problem statement
theorem algorithm_comparable_to_euclidean :
  ancient_mathematics_world_leading → song_yuan_algorithm → 
  True :=  -- Placeholder representing that the algorithm is the method of successive subtraction
by 
  intro h1 h2 
  sorry

end algorithm_comparable_to_euclidean_l460_460383


namespace rectangle_to_square_l460_460401

theorem rectangle_to_square (length width : ℕ) (h1 : 2 * (length + width) = 40) (h2 : length - 8 = width + 2) :
  width + 2 = 7 :=
by {
  -- Proof goes here
  sorry
}

end rectangle_to_square_l460_460401


namespace smallest_AAB_value_l460_460491

theorem smallest_AAB_value :
  ∃ (A B : ℕ), 
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
  A ≠ B ∧ 
  110 * A + B = 7 * (10 * A + B) ∧ 
  (∀ (A' B' : ℕ), 
    A' ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    B' ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
    A' ≠ B' ∧ 
    110 * A' + B' = 7 * (10 * A' + B') → 
    110 * A + B ≤ 110 * A' + B') :=
by
  sorry

end smallest_AAB_value_l460_460491


namespace ball_bounces_to_point_five_l460_460823

theorem ball_bounces_to_point_five :
  ∃ n : ℕ, (n = 5) ∧ ∀ k : ℕ, k < n → 15 * (1 / 2)^k ≥ 0.5 
  ∧ 15 * (1 / 2)^n < 0.5 :=
begin
  sorry 
end

end ball_bounces_to_point_five_l460_460823


namespace roots_matrix_det_zero_l460_460331

noncomputable def roots (r s t : ℝ) := {d e f g : ℝ // Polynomial.eval d (Polynomial.C 1 * Polynomial.X^4 + Polynomial.C r * Polynomial.X^2 + Polynomial.C s * Polynomial.X + Polynomial.C t) = 0 ∧
                                                      Polynomial.eval e (Polynomial.C 1 * Polynomial.X^4 + Polynomial.C r * Polynomial.X^2 + Polynomial.C s * Polynomial.X + Polynomial.C t) = 0 ∧
                                                      Polynomial.eval f (Polynomial.C 1 * Polynomial.X^4 + Polynomial.C r * Polynomial.X^2 + Polynomial.C s * Polynomial.X + Polynomial.C t) = 0 ∧
                                                      Polynomial.eval g (Polynomial.C 1 * Polynomial.X^4 + Polynomial.C r * Polynomial.X^2 + Polynomial.C s * Polynomial.X + Polynomial.C t) = 0}

theorem roots_matrix_det_zero (r s t d e f g : ℝ) (hr : roots r s t d e f g) :
  Matrix.det ![
    ![d, e, f, g],
    ![e, f, g, d],
    ![f, g, d, e],
    ![g, d, e, f]
  ] = 0 :=
by sorry

end roots_matrix_det_zero_l460_460331


namespace angle_GDA_is_135_l460_460668

-- Definitions for the geometric entities and conditions mentioned
structure Triangle :=
  (angle_A : ℝ) (angle_B : ℝ) (angle_C : ℝ)

structure Square :=
  (angle : ℝ := 90)

def BCD : Triangle :=
  { angle_A := 45, angle_B := 45, angle_C := 90 }

def ABCD : Square :=
  {}

def DEFG : Square :=
  {}

-- The proof problem stated in Lean 4
theorem angle_GDA_is_135 :
  ∃ θ : ℝ, θ = 135 ∧ 
  (∀ (BCD : Triangle), BCD.angle_C = 90 ∧ BCD.angle_A = 45 ∧ BCD.angle_B = 45) ∧ 
  (∀ (Square : Square), Square.angle = 90) → 
  θ = 135 :=
by
  sorry

end angle_GDA_is_135_l460_460668


namespace bags_of_cookies_l460_460071

theorem bags_of_cookies (bags : ℕ) (cookies_total candies_total : ℕ) 
    (h1 : bags = 14) (h2 : cookies_total = 28) (h3 : candies_total = 86) :
    bags = 14 :=
by
  exact h1

end bags_of_cookies_l460_460071


namespace solve_abs_eq_l460_460762

theorem solve_abs_eq (x : ℝ) : |x - 3| = 5 - 2x → x = 2 ∨ x = 8 / 3 :=
by
  sorry

end solve_abs_eq_l460_460762


namespace class_average_score_l460_460654

theorem class_average_score :
  ∃ (average : ℚ),
    (let students_total := 40 in
     let marks_95 := 5 * 95 in
     let marks_0 := 3 * 0 in
     let marks_65 := 6 * 65 in
     let marks_80 := 8 * 80 in
     let remaining_students := students_total - (5 + 3 + 6 + 8) in
     let marks_remaining := remaining_students * 45 in
     let total_marks := marks_95 + marks_0 + marks_65 + marks_80 + marks_remaining in
     2000 ≤ total_marks ∧ total_marks ≤ 2400 ∧
     average = total_marks / students_total) :=
  ∃ (average : ℚ), average = 2315 / 40 := by
  sorry

end class_average_score_l460_460654


namespace B_work_days_l460_460100

theorem B_work_days (B : ℝ) :
  let A_rate := 1 / 20
  let combined_days := 12.727272727272728
  let combined_rate := 1 / combined_days
  let B_rate := 1 / B
  A_rate + B_rate = combined_rate → 
  B ≈ 34.90909090909091 :=
sorry

end B_work_days_l460_460100


namespace conic_section_is_parabola_l460_460886

def isParabola (equation : String) : Prop := 
  equation = "|y - 3| = sqrt((x + 4)^2 + (y - 1)^2)"

theorem conic_section_is_parabola : isParabola "|y - 3| = sqrt((x + 4)^2 + (y - 1)^2)" :=
  by
  sorry

end conic_section_is_parabola_l460_460886


namespace reciprocal_of_neg_two_l460_460781

theorem reciprocal_of_neg_two : ∀ x : ℝ, x = -2 → (1 / x) = -1 / 2 :=
by
  intro x h
  rw [h]
  norm_num

end reciprocal_of_neg_two_l460_460781


namespace tan_A_eq_zero_l460_460677

theorem tan_A_eq_zero 
    (A B C : Type) [Real] 
    (AC AB : ℝ)
    (sqrt18 : AC = Real.sqrt 18)
    (sqrt2 : AB = 3 * Real.sqrt 2)
    (is_right_triangle_at_B : ∀ a b: ℝ, triangle.a B C a b := 90) :
    Real.tan (A) = 0 := 
by 
  sorry

end tan_A_eq_zero_l460_460677


namespace Adrianna_second_store_visit_gum_l460_460494

theorem Adrianna_second_store_visit_gum (x : ℕ) 
  (initial_gum : ℕ := 10) 
  (additional_gum : ℕ := 3)
  (total_friends : ℕ := 15)
  (final_gum : ℕ := initial_gum + additional_gum + x) :
  final_gum = total_friends → x = 2 :=
by
  intro h
  have : 13 + x = 15 := h
  have : x = 15 - 13 := Nat.eq_sub_of_add_eq this
  show x = 2, from this

end Adrianna_second_store_visit_gum_l460_460494


namespace log10_calculation_l460_460804

theorem log10_calculation :
  [Real.log 10 (10 * (Real.log 10 1000))]^2 = (Real.log 10 3 + 1)^2 := by
  sorry

end log10_calculation_l460_460804


namespace midpoint_hexagon_area_l460_460033

-- Define the vertices of the hexagon ABCDEF
variables {A B C D E F : Type*}

-- Define the existence of convex hexagon ABCDEF
def is_convex_hexagon (A B C D E F : Type*) : Prop :=
sorry -- Placeholder for the actual definition

-- Define the midpoints of the diagonals
def midpoint (X Y : Type*) : Type* :=
sorry -- Placeholder for the actual definition

def A1 := midpoint A C
def B1 := midpoint B D
def C1 := midpoint C E
def D1 := midpoint D F
def E1 := midpoint E A
def F1 := midpoint F B

-- Define the convexity of the hexagon formed by midpoints
def is_convex_hexagon_midpoints (A1 B1 C1 D1 E1 F1 : Type*) : Prop :=
sorry -- Placeholder for the actual definition

-- Define the function to calculate the area
def area (hexagon : list Type*) : ℝ :=
sorry -- Placeholder for the actual definition

-- Prove that the area of the hexagon formed by the midpoints is one-fourth the area of the original hexagon
theorem midpoint_hexagon_area {A B C D E F : Type*}
  (hABCDEF : is_convex_hexagon A B C D E F) :
  area [A1, B1, C1, D1, E1, F1] = (1 / 4) * area [A, B, C, D, E, F] :=
sorry


end midpoint_hexagon_area_l460_460033


namespace chip_cost_l460_460752

theorem chip_cost 
  (calories_per_chip : ℕ)
  (chips_per_bag : ℕ)
  (cost_per_bag : ℕ)
  (desired_calories : ℕ)
  (h1 : calories_per_chip = 10)
  (h2 : chips_per_bag = 24)
  (h3 : cost_per_bag = 2)
  (h4 : desired_calories = 480) : 
  cost_per_bag * (desired_calories / (calories_per_chip * chips_per_bag)) = 4 := 
by 
  sorry

end chip_cost_l460_460752


namespace min_m_for_tan_l460_460087

theorem min_m_for_tan (m : ℝ) :
  (∀ x ∈ set.Icc (-real.pi / 4) (real.pi / 3), real.tan x ≤ m) ↔ m ≥ real.sqrt 3 :=
sorry

end min_m_for_tan_l460_460087


namespace probability_two_girls_together_l460_460045

def numberOfWaysToArrangeStudents : ℕ := 120 -- A_5^5

def numberOfWaysWithTwoGirlsTogether : ℕ := 72 -- A_3^2 * A_2^2 * A_3^2

theorem probability_two_girls_together (boys girls : ℕ) (totalWays arrangedWays : ℕ) 
  (h_boys : boys = 2) (h_girls : girls = 3) (h_totalWays : totalWays = 120) (h_arrangedWays : arrangedWays = 72) :
  arrangedWays / totalWays = 3 / 5 :=
by
  rw [h_totalWays, h_arrangedWays]
  norm_num
  sorry


end probability_two_girls_together_l460_460045


namespace quadrilateral_angle_difference_l460_460020

theorem quadrilateral_angle_difference (h_ratio : ∀ (a b c d : ℕ), a = 3 * d ∧ b = 4 * d ∧ c = 5 * d ∧ d = 6 * d) 
  (h_sum : ∀ (a b c d : ℕ), a + b + c + d = 360) : 
  ∃ (x : ℕ), 6 * x - 3 * x = 60 := 
by 
  sorry

end quadrilateral_angle_difference_l460_460020


namespace zoomtopia_social_distancing_l460_460732

theorem zoomtopia_social_distancing (n : ℕ) (hpos : n > 0) :
  ∃ (S : set (ℝ × ℝ)), S.card = 3 * n^2 + 3 * n + 1 ∧
  (∀ p q ∈ S, p ≠ q → dist p q ≥ 1) ∧
  ∀ x : ℝ × ℝ, x ∈ S → (exists i : ℤ, rotation_by_angle x (i * (π/3)) = x ∧ dist x (0, 0) ≤ 6 * n) :=
sorry

end zoomtopia_social_distancing_l460_460732


namespace packaging_combinations_l460_460099

theorem packaging_combinations :
  let wraps := 10
  let ribbons := 4
  let cards := 5
  let stickers := 6
  wraps * ribbons * cards * stickers = 1200 :=
by
  rfl

end packaging_combinations_l460_460099


namespace max_elephants_non_attacking_l460_460378

-- Define the condition of the board size and elephant movement
def board_size : ℕ := 10

def moves_diagonally (x y : ℕ × ℕ) : Prop :=
  (abs (x.1 - y.1) = 1 ∧ abs (x.2 - y.2) = 1) ∨
  (abs (x.1 - y.1) = 2 ∧ abs (x.2 - y.2) = 2)

-- Define a function to count non-attacking elephants
def max_non_attacking_elephants (n : ℕ) : ℕ :=
  if n = board_size then 40 else 0

-- Statement to prove: maximum number of non-attacking elephants on a 10x10 board is 40
theorem max_elephants_non_attacking :
  max_non_attacking_elephants board_size = 40 :=
sorry

end max_elephants_non_attacking_l460_460378


namespace pure_imaginary_number_l460_460642

theorem pure_imaginary_number (m : ℝ) (h_real : m^2 - 5 * m + 6 = 0) (h_imag : m^2 - 3 * m ≠ 0) : m = 2 :=
sorry

end pure_imaginary_number_l460_460642


namespace correct_conclusions_l460_460214

variables (α β γ : Plane) (l m : Line)

-- Conditions
axiom perp_1 : α.perp γ
axiom inter_1 : γ.inter α = m
axiom inter_2 : γ.inter β = l
axiom perp_2 : l.perp m

-- Conclusions to prove
theorem correct_conclusions :
  (l.perp α ∧ α.perp β) :=
sorry

end correct_conclusions_l460_460214


namespace caleb_spent_l460_460875

-- Definitions of conditions
def total_burgers := 50
def double_burger_cost := 1.5
def single_burger_cost := 1.0
def num_double_burgers := 29
def num_single_burgers := total_burgers - num_double_burgers

-- Theorem statement
theorem caleb_spent : (num_double_burgers * double_burger_cost + num_single_burgers * single_burger_cost) = 64.50 :=
by
  sorry

end caleb_spent_l460_460875


namespace chord_length_l460_460470

-- Define the point and the slope angle
def point : ℝ × ℝ := (1, 0)
def slope_angle : ℝ := 30

-- Define the circle equation
def circle_center : ℝ × ℝ := (2, 0)
def circle_radius : ℝ := 1
def circle_eq (x y : ℝ) : Prop := (x - circle_center.1)^2 + y^2 = circle_radius^2

-- Define the line equation derived from point and slope_angle
def line_eq (x y : ℝ) : Prop := y = (Real.tan (slope_angle * Real.pi / 180)) * (x - point.1)

-- Prove that the chord length AB is equal to √3
theorem chord_length : ∃ A B : ℝ × ℝ, line_eq A.1 A.2 ∧ circle_eq A.1 A.2 ∧ 
                                     line_eq B.1 B.2 ∧ circle_eq B.1 B.2 ∧
                                     Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 :=
by easy

end chord_length_l460_460470


namespace first_consecutive_odd_number_l460_460044

theorem first_consecutive_odd_number :
  ∃ k : Int, 2 * k - 1 + 2 * k + 1 + 2 * k + 3 = 2 * k - 1 + 128 ∧ 2 * k - 1 = 61 :=
by
  sorry

end first_consecutive_odd_number_l460_460044


namespace mixture_total_weight_l460_460458

-- State the conditions and the target to prove

variables (total_parts almonds_per_part parts_almonds parts_walnuts total_weight : ℕ)

-- Given conditions
def mixture_conditions :=
  parts_almonds = 5 ∧
  parts_walnuts = 2 ∧
  (parts_almonds + parts_walnuts = total_parts) ∧
  almonds_per_part * parts_almonds = 150

theorem mixture_total_weight (h : mixture_conditions) : total_weight = 210 := by
  sorry

end mixture_total_weight_l460_460458


namespace ratio_of_squares_l460_460727

theorem ratio_of_squares (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum_zero : x + 2 * y + 3 * z = 0) :
    (x^2 + y^2 + z^2) / (x * y + y * z + z * x) = -4 := by
  sorry

end ratio_of_squares_l460_460727


namespace remainder_is_400_l460_460697

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def num_functions (f : ℕ → ℕ) : ℕ :=
  if ∃ c ∈ A, ∀ x ∈ A, f(f(x)) = c then 1 else 0

def N : ℕ :=
  8 * (7 + 672 + 1701 + 1792 + 875 + 252 + 1)

def remainder : ℕ :=
  N % 1000

theorem remainder_is_400 : remainder = 400 :=
  sorry

end remainder_is_400_l460_460697


namespace find_positive_integer_n_l460_460546

theorem find_positive_integer_n (n : ℕ) (hn : 0 < n)
    (h : cos (Real.pi / (2 * n)) - sin (Real.pi / (2 * n)) = Real.sqrt n / 2) : n = 4 :=
by
  sorry

end find_positive_integer_n_l460_460546


namespace probability_sum_is_five_l460_460806

theorem probability_sum_is_five (m n : ℕ) (h_m : 1 ≤ m ∧ m ≤ 6) (h_n : 1 ≤ n ∧ n ≤ 6)
  (h_total_outcomes : ∃(total_outcomes : ℕ), total_outcomes = 36)
  (h_favorable_outcomes : ∃(favorable_outcomes : ℕ), favorable_outcomes = 4) :
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 9 :=
sorry

end probability_sum_is_five_l460_460806


namespace principal_cup_probability_l460_460292

noncomputable def P (event : Prop) : ℚ := sorry

theorem principal_cup_probability
  (P_A : P(A) = 3 / 4)
  (P_not_A_and_not_C : P(¬A ∧ ¬C) = 1 / 12)
  (P_B_and_C : P(B ∧ C) = 1 / 4) :
  P(B) = 3 / 8 ∧ P(C) = 2 / 3 ∧
  P(A ∧ B ∧ ¬C) + P(A ∧ ¬B ∧ C) + P(¬A ∧ B ∧ C) = 15 / 32 := by
  sorry

end principal_cup_probability_l460_460292


namespace point_M_coordinates_l460_460221

open Real

theorem point_M_coordinates (θ : ℝ) (h_tan : tan θ = -4 / 3) (h_theta : π / 2 < θ ∧ θ < π) :
  let x := 5 * cos θ
  let y := 5 * sin θ
  (x, y) = (-3, 4) := 
by 
  sorry

end point_M_coordinates_l460_460221


namespace customer_pays_correct_amount_l460_460114

def wholesale_price : ℝ := 4
def markup : ℝ := 0.25
def discount : ℝ := 0.05

def retail_price : ℝ := wholesale_price * (1 + markup)
def discount_amount : ℝ := retail_price * discount
def customer_price : ℝ := retail_price - discount_amount

theorem customer_pays_correct_amount : customer_price = 4.75 := by
  -- proof steps would go here, but we are skipping them as instructed
  sorry

end customer_pays_correct_amount_l460_460114


namespace area_difference_l460_460634

theorem area_difference:
  let r1 := 25 in
  let d2 := 15 in
  let r2 := d2 / 2 in
  let A1 := Real.pi * r1^2 in
  let A2 := Real.pi * r2^2 in
  A1 - A2 = 568.75 * Real.pi := by
  sorry

end area_difference_l460_460634


namespace snail_number_is_square_l460_460343

def isSnailNumber (n : Nat) : Prop :=
  let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
  digits ∈ [[8, 9, 1, 0], [8, 9, 0, 1], [9, 8, 1, 0], [9, 8, 0, 1],
            [8, 1, 9, 0], [8, 0, 9, 1], [1, 8, 9, 0], [1, 0, 9, 8],
            [0, 8, 9, 1], [0, 1, 9, 8], [9, 1, 0, 8], [9, 0, 1, 8]]

theorem snail_number_is_square : ∃ n : Nat, isSnailNumber n ∧ n = 33^2 :=
by 
  use 1089
  unfold isSnailNumber
  -- Manually expand the unfolding for demonstration
  have digits : 1089 / 1000 = 1 → 1089 / 100 % 10 = 0 → 1089 / 10 % 10 = 8 → 1089 % 10 = 9
  exact sorry

end snail_number_is_square_l460_460343


namespace rise_in_water_level_after_cube_submersion_l460_460108

theorem rise_in_water_level_after_cube_submersion
  (edge_cube : ℝ) (diameter_base_vessel : ℝ) (initial_height_water : ℝ)
  (h : ℝ) (π : ℝ)
  (cube_submersion : edge_cube = 16)
  (diam_base_vessel : diameter_base_vessel = 20)
  (init_height_water : initial_height_water = 10)
  (pi_value : π = real.pi)
  (volume_cube : (edge_cube ^ 3) = 4096)
  (volume_displaced_water : 100 * π * h = 4096) :
  h = 13.04 :=
by sorry

end rise_in_water_level_after_cube_submersion_l460_460108


namespace discount_percentage_l460_460841

-- Define our conditions as hypotheses
variables (P : ℕ) -- Original Price per Item
variables (h_half_price : ∀ x, (1/2) * x = 0.5 * x)
variables (h_coupon : ∀ x, (0.8) * x = 0.8 * x)
variables (h_bogo : ∀ x, x = x) -- Buy one get one free implicitly means paying for one
variables (h_same_price : P = P) -- Second item's original price is same as the first

-- Statement of the theorem
theorem discount_percentage (P : ℕ) : 
  (0.4 * P / (2 * P)) * 100 = 20 :=
by 
  -- Placeholder for the proof
  sorry

end discount_percentage_l460_460841


namespace maximum_k_l460_460543

theorem maximum_k 
  (a b c k : ℝ) 
  (h1: a > 0) 
  (h2: b > 0) 
  (h3: c > 0)
  (h4 : a^2 + b^2 + c^2 = 2 * (a*b + b*c + c*a))
  (h5 : ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → 
        (1 / (k * a * b + c^2) + 1 / (k * b * c + a^2) + 1 / (k * c * a + b^2) ≥ (k + 3) / (a^2 + b^2 + c^2))) : 
  k ≤ 2 :=
begin
  sorry
end

end maximum_k_l460_460543


namespace S2016_value_l460_460669

theorem S2016_value (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h1 : a 1 = -2016)
  (h2 : ∀ n, S (n+1) = S n + a (n+1))
  (h3 : ∀ n, a (n+1) = a n + d)
  (h4 : (S 2015) / 2015 - (S 2012) / 2012 = 3) : S 2016 = -2016 := 
sorry

end S2016_value_l460_460669


namespace cats_in_shelter_l460_460446

theorem cats_in_shelter (C D: ℕ) (h1 : 15 * D = 7 * C) 
                        (h2 : 15 * (D + 12) = 11 * C) :
    C = 45 := by
  sorry

end cats_in_shelter_l460_460446


namespace find_n_from_remainders_l460_460816

theorem find_n_from_remainders (a n : ℕ) (h1 : a^2 % n = 8) (h2 : a^3 % n = 25) : n = 113 := 
by 
  -- proof needed here
  sorry

end find_n_from_remainders_l460_460816


namespace willie_stickers_l460_460439

def num_stickers_start_with (given_away remaining initial : Nat) : Prop :=
  remaining + given_away = initial

theorem willie_stickers : ∃ initial, num_stickers_start_with 7 29 initial ∧ initial = 36 :=
by {
  use 36,
  split,
  { -- prove that remaining + given_away = initial
    show 29 + 7 = 36,
    exact rfl,
  },
  { -- prove that initial = 36
    show 36 = 36,
    exact rfl,
  }
}

end willie_stickers_l460_460439


namespace b_equals_neg8_l460_460648

noncomputable def b_value (b : ℝ) (z : ℂ) (i : ℂ) : Prop :=
  i * i = -1 ∧ (∀ a : ℝ, z = a * i → (2 - i) * z = 4 - b * i) → b = -8

theorem b_equals_neg8 {b : ℝ} {z : ℂ} (i : ℂ) (h_im : i * i = -1) 
  (h_eq : ∀ a : ℝ, z = a * i → (2 - i) * z = 4 - b * i) : b = -8 :=
begin
  apply b_value,
  split,
  exact h_im,
  exact h_eq,
  sorry
end

end b_equals_neg8_l460_460648


namespace find_f_l460_460223

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- Given condition
axiom h : ∀ x : ℝ, f (Real.sqrt (2 * x - 1)) = 8 * x^2 - 2 * x - 1

-- Theorem statement
theorem find_f : f = (λ x : ℝ, 2 * x^4 + 3 * x^2) :=
by
  funext
  intro x
  sorry

end find_f_l460_460223


namespace negation_of_universal_statement_l460_460398

theorem negation_of_universal_statement :
  (¬ (∀ x : ℝ, x > Real.sin x)) ↔ (∃ x : ℝ, x ≤ Real.sin x) :=
by sorry

end negation_of_universal_statement_l460_460398


namespace Maria_disk_count_to_make_profit_l460_460346

def cost_per_disk : ℝ := 6 / 5
def selling_per_disk : ℝ := 7 / 4
def profit_per_disk : ℝ := selling_per_disk - cost_per_disk
def desired_profit : ℝ := 120
def disk_count : ℕ := 219

theorem Maria_disk_count_to_make_profit :
  ∀ (disks_sold : ℕ), 
  disks_sold = disk_count →
  disks_sold * profit_per_disk = desired_profit :=
by
  sorry

end Maria_disk_count_to_make_profit_l460_460346


namespace cricket_total_minutes_l460_460006

theorem cricket_total_minutes (sean_daily_minutes : ℕ) (sean_days : ℕ) (indira_minutes : ℕ) : 
  sean_daily_minutes = 50 → 
  sean_days = 14 → 
  indira_minutes = 812 → 
  (sean_daily_minutes * sean_days + indira_minutes) = 1512 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end cricket_total_minutes_l460_460006


namespace part1_part2_l460_460611

-- Define set A
def A : Set ℝ := {x | 3 < x ∧ x < 6}

-- Define set B
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Define set complement in ℝ
def CR (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

-- First part of the problem
theorem part1 :
  (A ∩ B = {x | 3 < x ∧ x < 6}) ∧
  (CR A ∪ CR B = {x | x ≤ 3 ∨ x ≥ 6}) :=
sorry

-- Define set C depending on a
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2 * a - 1}

-- Second part of the problem
theorem part2 (a : ℝ) (h : B ∪ C a = B) :
  a ≤ 1 ∨ (2 ≤ a ∧ a ≤ 5) :=
sorry

end part1_part2_l460_460611


namespace johns_total_money_l460_460683

-- Defining the given conditions
def initial_amount : ℕ := 5
def amount_spent : ℕ := 2
def allowance : ℕ := 26

-- Constructing the proof statement
theorem johns_total_money : initial_amount - amount_spent + allowance = 29 :=
by
  sorry

end johns_total_money_l460_460683


namespace a_plus_b_l460_460723

theorem a_plus_b (a b : ℝ) (f g : ℝ → ℝ) (h_g : ∀ x, g(x) = 3*x - 4) (h_f : ∀ x, f(x) = a*x + b) 
  (h_gf : ∀ x, g(f(x)) = 4*x + 3) : a + b = 11/3 := 
by 
  -- Here we would need to prove our statement, but we skip it with sorry.
  sorry

end a_plus_b_l460_460723


namespace part1_part2_l460_460966

section

variable (a : ℝ)
def f (x : ℝ) := a * Real.log x + x^2

theorem part1 (h : a = -2) : ∀ x ∈ Ioi (1:ℝ), (f (-2) x)' > 0 :=
by
  intro x hx
  sorry

theorem part2 (h : 1 ≤ x ∧ x ≤ Real.exp 1) : 
  if a ≥ -2 then 
    ∃ min_x : ℝ, min_x = 1 ∧ f a min_x = 1
  else if -2 * Real.exp (2) < a then 
    ∃ min_x : ℝ, min_x = Real.sqrt (-a / 2) ∧ f a min_x = (a / 2) * Real.log (-a / 2) - (a / 2)
  else 
    ∃ min_x : ℝ, min_x = Real.exp 1 ∧ f a min_x = a + Real.exp (2) :=
by
  sorry

end

end part1_part2_l460_460966


namespace conditional_probability_l460_460405

variable {Ω : Type*} {P : MeasureTheory.ProbabilityMeasure Ω}
variable (A B : MeasureTheory.MeasurableSet Ω)

-- Given conditions
axiom prob_A : P A = 0.8
axiom prob_B : P B = 0.4
axiom prob_A_and_B : P (A ∩ B) = 0.4

-- To prove the conditional probability
theorem conditional_probability : MeasureTheory.condProb P B A = 0.5 :=
by
  have h : P (A ∩ B) / P A = 0.5,
  from sorry,
  exact MeasureTheory.condProb_eq h

end conditional_probability_l460_460405


namespace remainder_of_functions_mod_1000_l460_460692

noncomputable def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def number_of_functions := 
  let N := 8 * ∑ k in (Finset.range 7).filter (λ k, k > 0), (Nat.choose 7 k) * (k ^ (7 - k)) 
  in N

theorem remainder_of_functions_mod_1000 : number_of_functions % 1000 = 576 :=
by 
  let N := number_of_functions
  have h1 : N = 50576 := sorry
  have h2 : 50576 % 1000 = 576 := rfl
  rw [h1, h2]

end remainder_of_functions_mod_1000_l460_460692


namespace max_abs_sum_value_l460_460267

noncomputable def max_abs_sum (x y : ℝ) : ℝ := |x| + |y|

theorem max_abs_sum_value (x y : ℝ) (h : x^2 + y^2 = 4) : max_abs_sum x y ≤ 2 * Real.sqrt 2 :=
by {
  sorry
}

end max_abs_sum_value_l460_460267


namespace distance_metric_l460_460475

noncomputable def d (x y : ℝ) : ℝ :=
  (|x - y|) / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2))

theorem distance_metric (x y z : ℝ) :
  (d x x = 0) ∧
  (d x y = d y x) ∧
  (d x y + d y z ≥ d x z) := by
  sorry

end distance_metric_l460_460475


namespace max_grids_covered_by_exactly_one_2x2_square_l460_460941

theorem max_grids_covered_by_exactly_one_2x2_square (n : ℕ) (hn : Odd n) : 
  ∃ k, k = n * (n - 1) ∧ 
  ∀ arrangement, max_grids_with_exactly_one_2x2 (arrangement n) ≤ k := 
sorry

end max_grids_covered_by_exactly_one_2x2_square_l460_460941


namespace triangle_BME_area_l460_460676

variables (A B C M D E : Type*)
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace D] [MetricSpace E]
variables (AB AC BC BD DC : ℝ)

noncomputable def area_triangle (a b c: Type*) [MetricSpace a] [MetricSpace b] [MetricSpace c] : ℝ := sorry 

-- Given conditions in a)
def conditions (A B C M D E: Type*)
[MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace D] [MetricSpace E]
(AB AC BC BD DC : ℝ) :=
angle A B C = 120 ∧
is_midpoint M A B ∧
on_line_segment D B C ∧
ratio BD DC = 3/2 ∧
is_midpoint E A C ∧
area_triangle A B C = 36

-- Proof statement
theorem triangle_BME_area :
conditions A B C M D E AB AC BC BD DC →
area_triangle B M E = 9 :=
begin
  sorry
end

end triangle_BME_area_l460_460676


namespace children_with_both_colors_l460_460826

def percentage_children_with_both_colors (F : ℕ) (h_even : F % 2 = 0) : ℕ := 
  let C := F / 2 in
  let blue_percentage := 60 in
  let red_percentage := 65 in
  25

theorem children_with_both_colors (F : ℕ) (h_even : F % 2 = 0) :
  percentage_children_with_both_colors F h_even = 25 :=
by
  sorry

end children_with_both_colors_l460_460826


namespace taxi_ride_cost_is_five_dollars_l460_460980

def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def miles_traveled : ℝ := 10.0
def total_cost : ℝ := base_fare + (cost_per_mile * miles_traveled)

theorem taxi_ride_cost_is_five_dollars : total_cost = 5.00 :=
by
  -- proof omitted
  sorry

end taxi_ride_cost_is_five_dollars_l460_460980


namespace circle_diameter_and_circumference_l460_460424

theorem circle_diameter_and_circumference (A : ℝ) (hA : A = 225 * π) : 
  ∃ r d C, r = 15 ∧ d = 2 * r ∧ C = 2 * π * r ∧ d = 30 ∧ C = 30 * π :=
by
  sorry

end circle_diameter_and_circumference_l460_460424


namespace initial_students_count_l460_460770

theorem initial_students_count 
  (average_initial_weight : ℕ → ℝ)
  (n : ℕ)
  (W : ℕ → ℝ)
  (average_new_weight : ℝ)
  (new_student_weight : ℝ)
  (h1 : average_initial_weight n = 28)
  (h2 : average_new_weight = 27.5)
  (h3 : new_student_weight = 13)
  (h4 : W n = n * 28)
  (h5 : (W n + 13) / (n + 1) = 27.5) : n = 29 := 
begin
  sorry
end

end initial_students_count_l460_460770


namespace angle_terminal_side_equiv_l460_460673

def angle_equiv_terminal_side (θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * k * Real.pi

theorem angle_terminal_side_equiv : angle_equiv_terminal_side (-Real.pi / 3) (5 * Real.pi / 3) :=
by
  sorry

end angle_terminal_side_equiv_l460_460673


namespace parabola_locus_intersection_l460_460558

noncomputable def locus_of_intersection 
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < b) 
  (l m : ℝ → ℝ) 
  (h3 : ∀ y, l(y) * m(y) = (k1 * y - k1 * a) * (-k1 * y - k1 * b)) 
  (h4 : (k1 * y - y - k1 * a) * (-k1 * y - y - k1 * b) = 0)
  : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (locus_intersection k a b = 2 * x - (a + b))

theorem parabola_locus_intersection 
  (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < b) 
  (P : ℝ × ℝ)
  (hP : ∃ k : ℝ, k ≠ 0 ∧ ((P.1 = (2 * x - (a + b)) / 2) ∧ P.2 ≠ 0))
  : 2 * P.1 = a + b := sorry

end parabola_locus_intersection_l460_460558


namespace utilities_percentage_of_rent_l460_460305

-- Defining the conditions
def rent_per_week : ℝ := 1200
def weekly_expenses : ℝ := 3440
def employee_payment_per_hour : ℝ := 12.50
def hours_per_day : ℕ := 16
def days_per_week : ℕ := 5
def employees_per_shift : ℕ := 2

-- Statement of the proof problem
theorem utilities_percentage_of_rent : 
  let total_hours_per_week := hours_per_day * days_per_week in
  let total_employee_hours_per_week := total_hours_per_week * employees_per_shift in
  let total_weekly_payroll := total_employee_hours_per_week * employee_payment_per_hour in
  let rent_and_utilities_expense := weekly_expenses - total_weekly_payroll in
  let utilities_cost := rent_and_utilities_expense - rent_per_week in
  (utilities_cost / rent_per_week) * 100 = 20 :=
sorry

end utilities_percentage_of_rent_l460_460305


namespace multiples_of_3_or_5_but_not_6_l460_460627

theorem multiples_of_3_or_5_but_not_6 (n : ℕ) (h1 : n ≤ 200) :
  (multiples3 : ℕ) (multiples5 : ℕ) (multiples15 : ℕ) (multiples6 : ℕ)
    (h1 : multiples3 = (200 - 3) / 3 + 1)
    (h2 : multiples5 = (200 - 5) / 5 + 1)
    (h3 : multiples15 = (200 - 15) / 15 + 1)
    (h4 : multiples6 = (200 - 6) / 6 + 1) : 
    (multiples3 + multiples5 - multiples15 - multiples6) = 60 :=
begin
  sorry,
end

end multiples_of_3_or_5_but_not_6_l460_460627


namespace intersection_complement_P_Q_l460_460976

def P (x : ℝ) : Prop := x - 1 ≤ 0
def Q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

def complement_P (x : ℝ) : Prop := ¬ P x

theorem intersection_complement_P_Q :
  {x : ℝ | complement_P x} ∩ {x : ℝ | Q x} = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end intersection_complement_P_Q_l460_460976


namespace factorize_expression_l460_460533

variable (a b : ℝ)

theorem factorize_expression : ab^2 - a = a * (b + 1) * (b - 1) :=
sorry

end factorize_expression_l460_460533


namespace alpha_beta_sum_l460_460269

theorem alpha_beta_sum (α β : ℝ) 
  (hαβ1 : 0 < α ∧ α < π / 2) 
  (hαβ2 : 0 < β ∧ β < π / 2) 
  (h1 : sin (α / 2 - β) = -1 / 2) 
  (h2 : cos (α - β / 2) = sqrt 3 / 2) : 
  α + β = 2 * π / 3 :=
by sorry

end alpha_beta_sum_l460_460269


namespace boys_in_biology_class_l460_460870

theorem boys_in_biology_class : 
  ∃ (B : ℕ), 
  (∃ (G : ℕ), G = 3 * B) ∧ 
  (∀ x, P = 8 * B → P = 5 * x ∧ 2 * x + 3 * x = P) ∧ 
  (let E := 270 / 2 in E = 135 ∧ E = 135) ∧ 
  (4 * B = P / 2) ∧ 
  (4 * B + 8 * B + 270 = 1000) ↔ B = 60 :=
by
  sorry

end boys_in_biology_class_l460_460870


namespace relation_between_a_b_c_l460_460562

noncomputable def a : ℝ := real.sqrt 0.5
noncomputable def b : ℝ := real.sqrt 0.3
noncomputable def c : ℝ := real.log 2 / real.log 0.3

theorem relation_between_a_b_c : a > b ∧ b > c :=
by
  sorry

end relation_between_a_b_c_l460_460562


namespace smallest_possible_w_l460_460270

theorem smallest_possible_w 
  (h1 : 936 = 2^3 * 3 * 13)
  (h2 : 2^5 = 32)
  (h3 : 3^3 = 27)
  (h4 : 14^2 = 196) :
  ∃ w : ℕ, (w > 0) ∧ (936 * w) % 32 = 0 ∧ (936 * w) % 27 = 0 ∧ (936 * w) % 196 = 0 ∧ w = 1764 :=
sorry

end smallest_possible_w_l460_460270


namespace sector_area_correct_l460_460587

-- Definitions based on the problem conditions
def central_angle_deg := 72
def central_angle_rad := (72 : ℝ) * (Real.pi / 180)
def radius := 5

-- Definition of the area of the sector
def sector_area (θ r : ℝ) : ℝ := 0.5 * θ * r^2

-- The proof problem statement
theorem sector_area_correct :
  sector_area central_angle_rad radius = 5 * Real.pi :=
by {
  sorry
}

end sector_area_correct_l460_460587


namespace maclaurin_series_binomial_l460_460527

theorem maclaurin_series_binomial (m : ℝ) (x : ℝ) (h : |x| < 1) :
  (∑ n in (Finset.range (n + 1)), (m * (m - 1) * ... * (m - n + 1) / n!) * x^n) = (1 + x)^m := 
sorry

end maclaurin_series_binomial_l460_460527


namespace ellipse_eccentricity_l460_460936

theorem ellipse_eccentricity (a : ℝ) (h : a > 0) 
  (ell_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / 5 = 1 ↔ x^2 / a^2 + y^2 / 5 = 1)
  (ecc_eq : (eccentricity : ℝ) = 2 / 3) : 
  a = 3 := 
sorry

end ellipse_eccentricity_l460_460936


namespace otimes_2_5_l460_460721

def otimes (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem otimes_2_5 : otimes 2 5 = 23 :=
by
  sorry

end otimes_2_5_l460_460721


namespace sufficient_conditions_for_x_squared_lt_one_l460_460498

variable (x : ℝ)

theorem sufficient_conditions_for_x_squared_lt_one :
  (∀ x, (0 < x ∧ x < 1) → (x^2 < 1)) ∧
  (∀ x, (-1 < x ∧ x < 0) → (x^2 < 1)) ∧
  (∀ x, (-1 < x ∧ x < 1) → (x^2 < 1)) :=
by
  sorry

end sufficient_conditions_for_x_squared_lt_one_l460_460498


namespace coefficient_x3_expansion_of_a_x_l460_460274

theorem coefficient_x3_expansion_of_a_x (a : ℚ) :
  let f := (a + x) * (1 + x) ^ 4,
      a₀ := (polynomial.coeff f 0),
      a₁ := (polynomial.coeff f 1),
      a₂ := (polynomial.coeff f 2),
      a₃ := (polynomial.coeff f 3),
      a₄ := (polynomial.coeff f 4),
      a₅ := (polynomial.coeff f 5),
      sum_odd := a₁ + a₃ + a₅,
      f_at_1 := (a + 1) * (1 + 1) ^ 4,
      f_at_neg1 := (a - 1) * (1 - 1) ^ 4
  in sum_odd = 32 → polynomial.coeff f 3 = 18 :=
by
  sorry

end coefficient_x3_expansion_of_a_x_l460_460274


namespace div_25_by_4_l460_460427

theorem div_25_by_4 :
  ∃ A B : ℕ, 25 = 4 * A + B ∧ 0 ≤ B ∧ B < 4 ∧ A = 6 :=
begin
  use 6,
  use 1,
  split,
  { norm_num },
  split,
  { norm_num },
  { norm_num }
end

end div_25_by_4_l460_460427


namespace adam_ran_35_miles_l460_460687

theorem adam_ran_35_miles (katie_miles : ℕ) (h_katie : katie_miles = 10) 
  (h_adam_more : ∀ (adam_miles : ℕ), adam_miles = katie_miles + 25) :
  ∃ (adam_miles : ℕ), adam_miles = 35 :=
by
  use 35
  sorry

end adam_ran_35_miles_l460_460687


namespace solve_system_of_equations_l460_460013

variable (a b c : Real)

def K : Real := a * b * c + a^2 * c + c^2 * b + b^2 * a

theorem solve_system_of_equations 
    (h₁ : (a + b) * (a - b) * (b + c) * (b - c) * (c + a) * (c - a) ≠ 0)
    (h₂ : K a b c ≠ 0) :
    ∃ (x y z : Real), 
    x = b^2 - c^2 ∧
    y = c^2 - a^2 ∧
    z = a^2 - b^2 ∧
    (x / (b + c) + y / (c - a) = a + b) ∧
    (y / (c + a) + z / (a - b) = b + c) ∧
    (z / (a + b) + x / (b - c) = c + a) :=
by
  sorry

end solve_system_of_equations_l460_460013


namespace common_point_on_x_axis_roots_relation_l460_460245

-- First Proof Problem 
theorem common_point_on_x_axis (a : ℝ) (h : a ≠ 0) (h_common : ∃ x, f x = g x ∧ g x = 0 ) :
  a = -1 := 
by sorry

-- Second Proof Problem
theorem roots_relation (a p q x : ℝ) 
  (ha : a ≠ 0) 
  (hpq : 0 < p ∧ p < q ∧ q < (1 / a))
  (hroots : f x - g x = a * (x - p) * (x - q))
  (hx : x ∈ set.Ioo 0 p) :
  g x < f x ∧ f x < p - a := 
by sorry

-- Definitions of f and g
def f (x a : ℝ) : ℝ := a * x^2 + a * x
def g (x a : ℝ) : ℝ := x - a

end common_point_on_x_axis_roots_relation_l460_460245


namespace megan_roles_other_than_lead_l460_460740

def total_projects : ℕ := 800

def theater_percentage : ℚ := 50 / 100
def films_percentage : ℚ := 30 / 100
def television_percentage : ℚ := 20 / 100

def theater_lead_percentage : ℚ := 55 / 100
def theater_support_percentage : ℚ := 30 / 100
def theater_ensemble_percentage : ℚ := 10 / 100
def theater_cameo_percentage : ℚ := 5 / 100

def films_lead_percentage : ℚ := 70 / 100
def films_support_percentage : ℚ := 20 / 100
def films_minor_percentage : ℚ := 7 / 100
def films_cameo_percentage : ℚ := 3 / 100

def television_lead_percentage : ℚ := 60 / 100
def television_support_percentage : ℚ := 25 / 100
def television_recurring_percentage : ℚ := 10 / 100
def television_guest_percentage : ℚ := 5 / 100

theorem megan_roles_other_than_lead :
  let theater_projects := total_projects * theater_percentage
  let films_projects := total_projects * films_percentage
  let television_projects := total_projects * television_percentage

  let theater_other_roles := (theater_projects * theater_support_percentage) + 
                             (theater_projects * theater_ensemble_percentage) + 
                             (theater_projects * theater_cameo_percentage)

  let films_other_roles := (films_projects * films_support_percentage) + 
                           (films_projects * films_minor_percentage) + 
                           (films_projects * films_cameo_percentage)

  let television_other_roles := (television_projects * television_support_percentage) + 
                                (television_projects * television_recurring_percentage) + 
                                (television_projects * television_guest_percentage)
  
  theater_other_roles + films_other_roles + television_other_roles = 316 :=
by
  sorry

end megan_roles_other_than_lead_l460_460740


namespace equilateral_triangle_side_length_l460_460940

-- Define the conditions of the problem
def parabola (x y : ℝ) : Prop := y^2 = (2 * (real.sqrt 3)) * x

def is_equilateral_triangle (a b c : (ℝ × ℝ)) : Prop :=
  let d1 := (a.1 - b.1)^2 + (a.2 - b.2)^2
  let d2 := (b.1 - c.1)^2 + (b.2 - c.2)^2
  let d3 := (c.1 - a.1)^2 + (c.2 - a.2)^2
  d1 = d2 ∧ d2 = d3

-- Define the main theorem
theorem equilateral_triangle_side_length :
  ∃ (a b c : (ℝ × ℝ)), a = (0, 0) ∧
  parabola b.1 b.2 ∧ parabola c.1 c.2 ∧
  is_equilateral_triangle a b c ∧
  real.sqrt ((b.1 - c.1)^2 + (b.2 - c.2)^2) = 12 := by
  sorry

end equilateral_triangle_side_length_l460_460940


namespace simplify_expression_correct_l460_460008

noncomputable def simplify_expression : ℝ :=
  (sqrt 300 / sqrt 75) - (sqrt 98 / sqrt 49)

theorem simplify_expression_correct : simplify_expression = 2 - sqrt 2 := 
by 
  sorry

end simplify_expression_correct_l460_460008


namespace increasing_interval_of_f_l460_460030

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 1 then x^2 else real.log(x + 2)

def is_increasing_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ (x y : ℝ), a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

theorem increasing_interval_of_f :
  is_increasing_interval f 0 1 ∧ is_increasing_interval f 1 (real.to_nnreal 1).to_pnat.pnat_abs → 
  ∃ (a b : ℝ), a = 0 ∧ b = 1 ∧ ∀ x y, a ≤ x ∧ x < y ∧ y ≤ 1 ∨ x > 1 ∧ y > 1 → ∀ a, f(a) is increasing :=
sorry

end increasing_interval_of_f_l460_460030


namespace bijective_iff_pow_of_2_l460_460515

/-
Given n lamps arranged in a circular configuration, a cool procedure that alters the state of the lamps based on signals they receive, and a function f which maps configurations to their successive states:
- Prove that f is bijective if and only if n is a power of 2.
-/
theorem bijective_iff_pow_of_2 (n : ℕ) : 
  (∃ k : ℕ, n = 2^k) ↔ function.bijective f :=
sorry

end bijective_iff_pow_of_2_l460_460515


namespace problem_1_problem_2_l460_460172
-- Import the entire Mathlib library.

-- Problem (1)
theorem problem_1 (x y : ℝ) (h1 : |x - 3 * y| < 1 / 2) (h2 : |x + 2 * y| < 1 / 6) : |x| < 3 / 10 :=
sorry

-- Problem (2)
theorem problem_2 (x y : ℝ) : x^4 + 16 * y^4 ≥ 2 * x^3 * y + 8 * x * y^3 :=
sorry

end problem_1_problem_2_l460_460172


namespace point_M_trajectory_line_equation_l460_460577

-- Definition for points H, P, Q, and M
variables {H P Q M : ℝ × ℝ}
variables {l : ℝ → ℝ}

theorem point_M_trajectory (x y y' x': ℝ) 
  (H_cond : H = (-3, 0))
  (P_cond : P = (0, y'))
  (Q_cond : Q = (x', 0))
  (M_cond : M ∈ segment P Q)
  (HP_dot_PM_zero : (H.1, H.2) • (PM.1, PM.2) = 0)
  (PM_eq_neg32_MQ : PM = -3 / 2 • MQ)
  : y = 4*x := sorry

theorem line_equation (A B C D : ℝ × ℝ) 
  (circle_N : ∀ x y, x^2 + y^2 = 2*x)
  (BC_eq2 : distance B C = 2)
  (center_N : N = (1, 0))
  (line_l : ∀ y, l y = √2 * y + √2)
  (AB_BC_CD_arith : ∀ A B C D, points_in_arith_seq A B C D) 
  : ∃ l, l = √2*x - y - √2 ∨ l = √2 * x + y - √2 := sorry

end point_M_trajectory_line_equation_l460_460577


namespace isosceles_triangle_circumradius_l460_460912

theorem isosceles_triangle_circumradius (b : ℝ) (s : ℝ) (R : ℝ) (hb : b = 6) (hs : s = 5) :
  R = 25 / 8 :=
by 
  sorry

end isosceles_triangle_circumradius_l460_460912


namespace arithmetic_sequence_common_difference_and_formula_l460_460960

noncomputable def a_n (a1 d n : ℤ) : ℤ := a1 + (n - 1) * d

theorem arithmetic_sequence_common_difference_and_formula :
  (a_n a1 d 2 = -6) → (a_n a1 d 8 = -18) → 
  d = -2 ∧ ∀ n : ℤ, a_n a1 d n = -2n - 2 := 
by 
  sorry

end arithmetic_sequence_common_difference_and_formula_l460_460960


namespace min_max_M_l460_460614

noncomputable def find_min_max_M (x y z : ℝ) (h1 : x + 3 * y + 2 * z = 3) (h2 : 3 * x + 3 * y + z = 4) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) : ℝ × ℝ :=
  sorry

theorem min_max_M :
  ∀ (x y z : ℝ), x + 3 * y + 2 * z = 3 → 3 * x + 3 * y + z = 4 → 
                 x ≥ 0 → y ≥ 0 → z ≥ 0 →
  find_min_max_M x y z (x + 3 * y + 2 * z = 3) (3 * x + 3 * y + z = 4) (x ≥ 0) (y ≥ 0) (z ≥ 0) = (-1/6, 7) :=
by
  intros x y z h1 h2 hx hy hz
  sorry

end min_max_M_l460_460614


namespace bigger_part_of_sum_54_l460_460091

theorem bigger_part_of_sum_54 (x y : ℕ) (h₁ : x + y = 54) (h₂ : 10 * x + 22 * y = 780) : x = 34 :=
sorry

end bigger_part_of_sum_54_l460_460091


namespace line_eqn_l460_460469

theorem line_eqn (a : ℝ) (h : (3, 6) ∈ setOf (λ p : ℝ × ℝ, p.1 / a + p.2 / a = 1) ∧
                  (0 < a ∧ ∃ b : ℝ, b = a)) : 
  (∀ (x y : ℝ), (x, y) ∈ setOf (λ p : ℝ × ℝ, p.1 / a + p.2 / a = 1) → (x + y = 9)) :=
sorry

end line_eqn_l460_460469


namespace fifth_term_arithmetic_seq_l460_460791

theorem fifth_term_arithmetic_seq (a d : ℤ) 
  (h10th : a + 9 * d = 23) 
  (h11th : a + 10 * d = 26) 
  : a + 4 * d = 8 :=
sorry

end fifth_term_arithmetic_seq_l460_460791


namespace sum_of_solutions_l460_460425

theorem sum_of_solutions (a b c : ℝ) (h_eq : ∀ x : ℝ, x^2 - (8 * x) + 15 = a * x^2 + b * x + c) (ha : a = 1) (hb : b = -8) (hc : c = 15) :
  ∑ x in {x : ℝ | (a * x^2 + b * x + c) = 0}, x = 8 :=
by
  -- It is a proof placeholder
  sorry

end sum_of_solutions_l460_460425


namespace smallest_positive_sum_11_l460_460892

-- Definitions based on the conditions
def a : ℕ → ℤ
def is_one_or_negone (a : ℕ → ℤ) : Prop :=
  ∀ i, a i = 1 ∨ a i = -1

-- Define the problem statement and prove that the smallest positive value of the sum is 11
theorem smallest_positive_sum_11 (a : ℕ → ℤ)
  (h1 : is_one_or_negone a)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ 99) :
  ∑ (i : ℕ) (j : ℕ) in finset.range 99, if i < j then a i * a j else 0 = 11 :=
sorry

end smallest_positive_sum_11_l460_460892


namespace log_problem_l460_460972

def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_problem :
  ∃ (x : ℝ),
    let a := log_base ((x / 2 + 1)^2) (7 * x / 2 - 17 / 4)
    let b := log_base (Real.sqrt (7 * x / 2 - 17 / 4)) ((3 * x / 2 - 6)^2)
    let c := log_base (Real.sqrt (3 * x / 2 - 6)) (x / 2 + 1)
    (a = b ∧ a = c + 1) ∨ (b = c ∧ c = a + 1) ∨ (c = a ∧ a = b + 1) ∧ x = 7 :=
by
  sorry

end log_problem_l460_460972


namespace range_of_x_l460_460589

-- Define the function and its properties
def f (x : ℝ) : ℝ := (1/3) * x^3 + 2 * sin x

theorem range_of_x (x : ℝ) (h1 : -2 < x) (h2 : x < 2) (h_deriv : ∀ x ∈ (-2, 2), deriv f x = x^2 + 2*cos(x)) (h_0 : f 0 = 0) : 
    1 < x ∧ x < 2 → f (x-1) + f (x^2 - x) > 0 :=
sorry

end range_of_x_l460_460589


namespace best_model_is_Model1_l460_460660

def model_R2_values : Type := ℕ × ℝ

def Model1 : model_R2_values := (1, 0.95)
def Model2 : model_R2_values := (2, 0.70)
def Model3 : model_R2_values := (3, 0.55)
def Model4 : model_R2_values := (4, 0.30)

def best_fitting_model (m1 m2 m3 m4 : model_R2_values) : model_R2_values :=
  if m1.2 ≥ m2.2 ∧ m1.2 ≥ m3.2 ∧ m1.2 ≥ m4.2 then
    m1
  else if m2.2 ≥ m3.2 ∧ m2.2 ≥ m4.2 then
    m2
  else if m3.2 ≥ m4.2 then
    m3
  else
    m4

theorem best_model_is_Model1 : best_fitting_model Model1 Model2 Model3 Model4 = Model1 := by
  sorry

end best_model_is_Model1_l460_460660


namespace disk_tangent_position_l460_460107

theorem disk_tangent_position
  (clock_radius : ℝ) (disk_radius : ℝ) (initial_tangent : ℝ)
  (arrow_initial_dir : ℝ) (clockwise : bool) :
  clock_radius = 30 → disk_radius = 5 → initial_tangent = 12 → arrow_initial_dir = 90 → clockwise = true →
  tangent_position = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end disk_tangent_position_l460_460107


namespace max_integer_solutions_self_centered_l460_460122

/-- 
A polynomial p(x) is called self-centered if it has integer coefficients and p(200) = 200.
We want to show that the maximum number of integer solutions k to the equation p(k) = k^4
for a self-centered polynomial p(x) is 10.
-/
theorem max_integer_solutions_self_centered (p : ℤ[X]) (h_coeff : ∀ n, p.coeff n ∈ ℤ) (h_self_centered : p.eval 200 = 200) :
  ∃ k : ℕ, ∀ (n : ℕ), (nat_degree (p - X^4) : ℤ) ≤ n → n = 10 :=
sorry

end max_integer_solutions_self_centered_l460_460122


namespace repaved_before_today_l460_460832

variable (total_repaved today_repaved : ℕ)

theorem repaved_before_today (h1 : total_repaved = 4938) (h2 : today_repaved = 805) :
  total_repaved - today_repaved = 4133 :=
by 
  -- variables are integers and we are performing a subtraction
  sorry

end repaved_before_today_l460_460832


namespace multiples_of_3_or_5_but_not_6_l460_460628

theorem multiples_of_3_or_5_but_not_6 (n : ℕ) (h1 : n ≤ 200) :
  (multiples3 : ℕ) (multiples5 : ℕ) (multiples15 : ℕ) (multiples6 : ℕ)
    (h1 : multiples3 = (200 - 3) / 3 + 1)
    (h2 : multiples5 = (200 - 5) / 5 + 1)
    (h3 : multiples15 = (200 - 15) / 15 + 1)
    (h4 : multiples6 = (200 - 6) / 6 + 1) : 
    (multiples3 + multiples5 - multiples15 - multiples6) = 60 :=
begin
  sorry,
end

end multiples_of_3_or_5_but_not_6_l460_460628


namespace purely_imaginary_implies_x_eq_neg2_l460_460228

theorem purely_imaginary_implies_x_eq_neg2 (x : ℝ) (z : ℂ) (h : z = 2 + complex.i + (1 - complex.i) * x) :
  (z.re = 0) → x = -2 := by
    sorry

end purely_imaginary_implies_x_eq_neg2_l460_460228


namespace monotonic_increase_intervals_triangle_area_l460_460969

noncomputable def f (x : ℝ) : ℝ :=
    let a := (2 * Real.cos x, Real.sqrt 3 * Real.sin (2 * x))
    let b := (Real.cos x, 1)
    a.1 * b.1 + a.2 * b.2

theorem monotonic_increase_intervals :
    (∀ k : ℤ, 
      ∃ a b : ℝ, 
      (-π / 3 + k * π ≤ a) ∧ (a ≤ π / 6 + k * π) ∧ 
      (2 * Real.sin (2 * a + π / 6) + 1 ≤ f a) ∧ 
      (f a ≤ 2 * Real.sin (2 * b + π / 6) + 1) ∧ 
      (2 * Real.sin (2 * b + π / 6) + 1 < f b)) :=
sorry

theorem triangle_area :
    (∀ A B C : ℝ, 
      let a := Real.sqrt 7
      let fA := 2
      ∃ b c : ℝ, 
      f(A) = fA ∧ a^2 = b^2 + c^2 - 2 * b * c * Real.cos A ∧ 
      f(A) = 2 -> Real.sin B = 2 * Real.sin C -> 
      (b^2 + c^2 - 3 * b * c = 7) -> 
      (b = 2 * c) -> 
      (c^2 = 7 / 3) ∧ 
      (2 * Real.sin (2 * A + π / 6) + 1 = fA) -> 
      let area := (7 * Real.sqrt 3) / 6
      (area = (a * b * Real.sin C) / 2)) :=
sorry

end monotonic_increase_intervals_triangle_area_l460_460969


namespace solve_t_l460_460707

open Real

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem solve_t : ∃ t : ℝ, 
  let A := (t - 5, -2),
      B := (-3, t + 4),
      M := midpoint A B
  in (distance M A)^2 = t^2 / 4 ∧ t = 10 :=
by
  sorry

end solve_t_l460_460707


namespace measure_angle_A_l460_460667

/-- In the right triangle ABC with ∠C = 90° and AB = AC, point D on AC such that BD bisects angle ABC, 
prove the measure of angle A is 45°. --/
theorem measure_angle_A (A B C D : Type) [inhabited A] [linear_order A] [metric_space A]
  (ABC_is_right_triangle : ∀ (A B C : Type), ∠ C = 90)
  (AB_AC_equal : ∀ (AB AC : Type), AB = AC)
  (D_on_AC : D ∈ AC)
  (BD_bisects_ABC : ∀ (BD : Type), bisects ∠ABC BD) :
  ∠A = 45 := by
  sorry

end measure_angle_A_l460_460667


namespace magnitude_of_b_l460_460250

open Real

variables (a b : EuclideanSpace ℝ (Fin 2))

-- Assuming given conditions:
-- 1. dot product a • b = 3
-- 2. norm a = 3
-- 3. angle between a and b is π/6
def given_conditions : Prop :=
  a ⬝ b = 3 ∧ ∥a∥ = 3 ∧ real.angle a b = (π / 6)

-- Proving magnitude of vector b is 2 * sqrt(3) / 3.
theorem magnitude_of_b (h : given_conditions a b) : ∥b∥ = (2 * sqrt 3) / 3 :=
  sorry

end magnitude_of_b_l460_460250


namespace ribbons_jane_uses_l460_460306

-- Given conditions
def dresses_sewn_first_period (dresses_per_day : ℕ) (days : ℕ) : ℕ :=
  dresses_per_day * days

def dresses_sewn_second_period (dresses_per_day : ℕ) (days : ℕ) : ℕ :=
  dresses_per_day * days

def total_dresses_sewn (dresses_first_period : ℕ) (dresses_second_period : ℕ) : ℕ :=
  dresses_first_period + dresses_second_period

def total_ribbons_used (total_dresses : ℕ) (ribbons_per_dress : ℕ) : ℕ :=
  total_dresses * ribbons_per_dress

-- Theorem to prove
theorem ribbons_jane_uses :
  total_ribbons_used (total_dresses_sewn (dresses_sewn_first_period 2 7) (dresses_sewn_second_period 3 2)) 2 = 40 :=
  sorry

end ribbons_jane_uses_l460_460306


namespace binomial_sum_mod_500_l460_460158

theorem binomial_sum_mod_500 :
  (∑ i in finset.range 405, nat.choose 2023 (5 * i)) % 500 = 81 := 
sorry

end binomial_sum_mod_500_l460_460158


namespace can_reduce_to_zero_sequence_b_can_reduce_to_zero_sequence_c_l460_460744

-- Sequence definitions with operations allowed 
def sequence_b := List.range 2000  -- This generates [0, 1, 2, ..., 1999]
def sequence_c := List.range 2001  -- This generates [0, 1, 2, ..., 2000]

-- Function defining the operation of replacing two numbers with their absolute difference
def replace_with_difference (a b : ℕ) : ℕ := abs (a - b)

-- The main theorem statements
theorem can_reduce_to_zero_sequence_b : ¬ (ReduceToZero sequence_b) := sorry
theorem can_reduce_to_zero_sequence_c : (ReduceToZero sequence_c) := sorry

-- Predicate for reducing entire sequence to zero given the replace_with_difference operation
def ReduceToZero (seq : List ℕ) : Prop := sorry

end can_reduce_to_zero_sequence_b_can_reduce_to_zero_sequence_c_l460_460744


namespace minimum_amount_l460_460078

-- Define the basic conditions
variables (A O S G : ℕ)
variables (candy_cost : ℝ := 0.1)

-- Hypotheses from the problem
def conditions :=
  A = 2 * O ∧
  S = 2 * G ∧
  A = 2 * S ∧
  A + O + S + G = 90 

-- The question to prove
theorem minimum_amount (h : conditions A O S G) : 
  ∃ (cost : ℝ), cost = 1.9 := 
sorry

end minimum_amount_l460_460078


namespace circle_area_isosceles_triangle_l460_460104

theorem circle_area_isosceles_triangle (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 3) (h₃ : c = 2) :
  ∃ R : ℝ, R = (81 / 32) * Real.pi :=
by sorry

end circle_area_isosceles_triangle_l460_460104


namespace find_custom_operator_result_l460_460719

def custom_operator (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem find_custom_operator_result :
  custom_operator 2 5 = 23 :=
by
  sorry

end find_custom_operator_result_l460_460719


namespace range_of_alpha_in_first_quadrant_l460_460576

noncomputable def range_of_alpha (α : ℝ) : Prop :=
  (π/4 < α ∧ α < π/2) ∨ (π < α ∧ α < 5*π/4)

theorem range_of_alpha_in_first_quadrant (α : ℝ) (h1 : 0 ≤ α) (h2 : α ≤ 2*π)
  (h3 : 0 < tan α) (h4 : sin α - cos α > 0) : range_of_alpha α :=
by
  sorry

end range_of_alpha_in_first_quadrant_l460_460576


namespace smallest_sum_of_five_consecutive_primes_divisible_by_five_l460_460550

theorem smallest_sum_of_five_consecutive_primes_divisible_by_five :
  ∃ (p1 p2 p3 p4 p5 : ℕ), (Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5) ∧
  ((p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ p4 < p5 ∧ p5 ≤ p1 + 10)) ∧
  (p1 + p2 + p3 + p4 + p5 = 119) :=
by
  sorry

end smallest_sum_of_five_consecutive_primes_divisible_by_five_l460_460550


namespace min_pieces_for_cake_division_l460_460349

-- Define the problem and conditions
def cake_problem (pieces : ℕ) : Prop :=
  ∃ division : ℕ → ℚ, 
  (∀ n, 1 ≤ n ∧ n ≤ pieces → 0 < division n ∧ division n < 1) ∧ -- Each piece is a fraction of the whole
  (∀ config : Finset ℕ, 
    (config.card = 4 ∨ config.card = 5) → 
    (∑ x in config, division x) = 1 / config.card) -- Equal share for 4 or 5 children

-- State the theorem we want to prove
theorem min_pieces_for_cake_division : ∃ pieces, cake_problem pieces ∧ pieces = 8 :=
by
  existsi 8
  -- Add the proof that 8 pieces meet the conditions for 4 or 5 children (omitted here)
  exact sorry

end min_pieces_for_cake_division_l460_460349


namespace total_legs_l460_460758

-- Define the number of octopuses
def num_octopuses : ℕ := 5

-- Define the number of legs per octopus
def legs_per_octopus : ℕ := 8

-- The total number of legs should be num_octopuses * legs_per_octopus
theorem total_legs : num_octopuses * legs_per_octopus = 40 :=
by
  -- The proof is omitted
  sorry

end total_legs_l460_460758


namespace number_of_girls_l460_460413

theorem number_of_girls (sections : ℕ) (boys_per_section : ℕ) (total_boys : ℕ) (total_sections : ℕ) (boys_sections girls : ℕ) :
  total_boys = 408 → 
  total_sections = 27 → 
  total_boys / total_sections = boys_per_section → 
  boys_sections = total_boys / boys_per_section → 
  total_sections - boys_sections = girls / boys_per_section → 
  girls = 324 :=
by sorry

end number_of_girls_l460_460413


namespace area_of_circle_II_l460_460421

noncomputable def area_of_I : ℝ := 16
noncomputable def diameter_of_I_equals_radius_of_II (radius_I : ℝ) : Prop := (2 * radius_I) = radius_II
noncomputable def radius_II (radius_I : ℝ) : ℝ := 2 * radius_I

theorem area_of_circle_II (radius_I : ℝ) (h1 : π * radius_I^2 = area_of_I) (h2 : diameter_of_I_equals_radius_of_II radius_I) : 
  π * (radius_II radius_I)^2 = 64 := 
sorry

end area_of_circle_II_l460_460421


namespace reciprocal_of_neg_two_l460_460784

theorem reciprocal_of_neg_two :
  (∃ x : ℝ, x = -2 ∧ 1 / x = -1 / 2) :=
by
  use -2
  split
  · rfl
  · norm_num

end reciprocal_of_neg_two_l460_460784


namespace sequence_exists_l460_460330

theorem sequence_exists
  {a_0 b_0 c_0 a b c : ℤ}
  (gcd1 : Int.gcd (Int.gcd a_0 b_0) c_0 = 1)
  (gcd2 : Int.gcd (Int.gcd a b) c = 1) :
  ∃ (n : ℕ) (a_seq b_seq c_seq : Fin (n + 1) → ℤ),
    a_seq 0 = a_0 ∧ b_seq 0 = b_0 ∧ c_seq 0 = c_0 ∧ 
    a_seq n = a ∧ b_seq n = b ∧ c_seq n = c ∧
    ∀ (i : Fin n), (a_seq i) * (a_seq i.succ) + (b_seq i) * (b_seq i.succ) + (c_seq i) * (c_seq i.succ) = 1 :=
sorry

end sequence_exists_l460_460330


namespace multiply_repeating_decimals_l460_460899

noncomputable def repeating_decimal_03 : ℚ := 1 / 33
noncomputable def repeating_decimal_8 : ℚ := 8 / 9

theorem multiply_repeating_decimals : repeating_decimal_03 * repeating_decimal_8 = 8 / 297 := by 
  sorry

end multiply_repeating_decimals_l460_460899


namespace minimize_distance_centers_l460_460570

-- Given a triangle ABC with points C1 and A1 on sides AB and BC respectively, 
-- We need to find a point P on the circumcircle of triangle ABC such that 
-- the distance between the centers of the circumcircles of triangles APC1 and CPA1 is minimized. 
-- The statement to prove is that P is diametrically opposite to B on the circumcircle of triangle ABC.

noncomputable def optimal_point_P (ABC : Triangle) (C1 : Point) (A1 : Point) (circumcircle_ABC : Circle) 
(A1_on_BC : on_line_segment A1 B C)
(C1_on_AB : on_line_segment C1 A B)
(P_on_circumcircle : on_circumcircle P circumcircle_ABC) : bool :=
  P = diametrically_opposite B on circumcircle_ABC

theorem minimize_distance_centers
  (ABC : Triangle) (C1 : Point) (A1 : Point) (circumcircle_ABC : Circle)
  (A1_on_BC : on_line_segment A1 B C)
  (C1_on_AB : on_line_segment C1 A B)
  (P : Point) (P_on_circumcircle : on_circumcircle P circumcircle_ABC) :
  P = diametrically_opposite B on circumcircle_ABC :=
sorry

end minimize_distance_centers_l460_460570


namespace find_Tn_l460_460568

-- Define the sequence a_n
def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else if n = 2 then 3 else 3 * 4^(n-2)

-- Define the sum of the first n terms of a_n, S_n
def S (n : ℕ) : ℕ :=
  ∑ i in Nat.range n, a (i + 1)

-- Condition a_{n+1}S_{n-1} - a_nS_n = 0 for n ≥ 2
axiom condition (n : ℕ) (hn : n ≥ 2) : a (n+1) * S (n-1) = a n * S n

-- Define b_n
def b (n : ℕ) : ℚ := 
  9 * a n / (a n + 3) / (a (n + 1) + 3)

-- Define the sum of the first n terms of b_n, T_n
def T (n : ℕ) : ℚ :=
  ∑ i in Nat.range n, b (i + 1)

-- Main theorem statement
theorem find_Tn (n : ℕ) : 
  T n = (7 / 8) - (1 / (4^(n-1) + 1)) :=
sorry

end find_Tn_l460_460568


namespace longest_side_length_of_quadrilateral_l460_460517

theorem longest_side_length_of_quadrilateral :
  let region := {p : ℝ × ℝ | (p.1 + 2 * p.2 ≤ 4) ∧ (3 * p.1 + 2 * p.2 ≥ 3) ∧ (p.1 ≥ 0) ∧ (p.2 ≥ 0)}
  in ∃ a b : ℝ × ℝ, a ∈ region ∧ b ∈ region ∧ 
     ((∀ c d : ℝ × ℝ, c ∈ region ∧ d ∈ region ∧ c ≠ d → dist c d ≤ dist a b) ∧ dist a b = Real.sqrt 5) :=
by
  sorry

end longest_side_length_of_quadrilateral_l460_460517


namespace max_chain_of_divided_equilateral_triangle_l460_460893

theorem max_chain_of_divided_equilateral_triangle (n : ℕ) (h : n > 0) : 
  ∃ k, k = n^2 - n + 1 ∧  
   ∀ (chain : List (Σ (i j : ℕ), i ≤ n ∧ j ≤ n) ) (h_chain : chain.Nodup ∧ (∀ (p q: Σ (i j : ℕ), i ≤ n ∧ j ≤ n), List.Successor chain p q → (p.1 = q.1 ∧ (p.2 = q.2 + 1 ∨ p.2 = q.2 - 1) ∨ p.2 = q.2 ∧ (p.1 = q.1 + 1 ∨ p.1 = q.1 - 1)))), list.length chain ≤ k := sorry

end max_chain_of_divided_equilateral_triangle_l460_460893


namespace find_friends_in_schools_l460_460418

-- Definitions of assumptions
def School : Type := fin 200 -- Each school has 200 students
def Student : Type := fin 600 -- Overall student pool

def friends (a b : Student) : Prop := sorry -- Friend relationship, symmetric

-- Conditions
constant E : finset Student
constant in_school (s : School) (x : Student) : Prop -- x is in school s

-- student x has at least one friend in each school
constant has_friend_in_each_school : ∀ (x : Student), ∃ s : School, ∃ f : Student, friends x f 

-- students in E have unique number of friends in any given school
constant E_property : ∀ (s : School) (x y ∈ E), ¬ in_school s x → ¬ in_school s y → friends x y → x = y

-- Problem statement
theorem find_friends_in_schools :
  ∃ (a b c : Student), ¬ in_school (0 : School) a ∧ ¬ in_school (1 : School) b ∧ ¬ in_school (2 : School) c ∧ 
  friends a b ∧ friends b c ∧ friends a c :=
sorry

end find_friends_in_schools_l460_460418


namespace incorrect_conclusion_l460_460954

noncomputable def quadratic (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 2) * x + 2

theorem incorrect_conclusion (m : ℝ) (hx : m - 2 = 0) :
  ¬(∀ x : ℝ, quadratic m x = 2 ↔ x = 2) :=
by
  sorry

end incorrect_conclusion_l460_460954


namespace find_r_l460_460535

-- Lean statement
theorem find_r (r : ℚ) (log_eq : Real.logb 81 (2 * r - 1) = -1 / 2) : r = 5 / 9 :=
by {
    sorry -- proof steps should not be included according to the requirements
}

end find_r_l460_460535


namespace problem1_problem2_problem3_l460_460508

-- Problem 1
theorem problem1 : -23 + 58 - (-5) = 40 := sorry

-- Problem 2
theorem problem2 : (5 / 8 + 1 / 6 - 3 / 4) * 24 = 1 := sorry

-- Problem 3
theorem problem3 : -3 ^ 2 - [-5 - 0.2 / (4 / 5) * (-2) ^ 2] = -3 := sorry

end problem1_problem2_problem3_l460_460508


namespace translate_to_kurdish_l460_460451

theorem translate_to_kurdish :
  (translate "ленивая лев ест мясо" = "Шере qәләп гошт дьхәшә") ∧
  (translate "здоровый бедняк берет ношу" = "Кəсибе саг' бар бəр дьгртə") ∧
  (translate "бык бедняка не понимает бедняка" = "Га кəсиб кəсиб нахунэ") :=
by sorry

noncomputable def translate (phrase : String) : String :=
sorry

end translate_to_kurdish_l460_460451


namespace find_sin_theta_l460_460323

-- Definitions of the given line and plane.
def line_direction_vector := (3 : ℝ, 4 : ℝ, 5 : ℝ)
def plane_normal_vector := (4 : ℝ, 5 : ℝ, -2 : ℝ)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
 v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
 real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

-- Define the sine of the angle θ between line and plane
def sin_theta : ℝ :=
 dot_product line_direction_vector plane_normal_vector / 
 (magnitude line_direction_vector * magnitude plane_normal_vector)

-- Statement to prove
theorem find_sin_theta : sin_theta = 11 * real.sqrt 10 / 75 :=
by
  sorry

end find_sin_theta_l460_460323


namespace count_multiples_3_or_5_not_6_up_to_200_l460_460624

theorem count_multiples_3_or_5_not_6_up_to_200 : 
  (Finset.card (Finset.filter (λ n, n ≤ 200 ∧ ((n % 3 = 0 ∨ n % 5 = 0) ∧ n % 6 ≠ 0)) (Finset.range 201))) = 60 := 
by 
  sorry

end count_multiples_3_or_5_not_6_up_to_200_l460_460624


namespace tree_height_at_2_years_l460_460849

theorem tree_height_at_2_years (h : ℕ → ℚ) (h5 : h 5 = 243) 
  (h_rec : ∀ n, h (n - 1) = h n / 3) : h 2 = 9 :=
  sorry

end tree_height_at_2_years_l460_460849


namespace find_f_prime_at_1_l460_460646

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - f'(1) * x^2 + x + 5

theorem find_f_prime_at_1 : (derivative f 1) = (2 / 3) :=
by
  sorry

end find_f_prime_at_1_l460_460646


namespace range_of_omega_of_monotonic_decreasing_l460_460997

open Set Real

noncomputable def monotonic_decreasing_interval (ω : ℝ) : Prop :=
  ∀ x ∈ Icc (-π / 8) (π / 12), 
    (deriv (λ x, (1 / 2) * sin (ω * x))) x ≤ 0

theorem range_of_omega_of_monotonic_decreasing :
  ∀ ω, monotonic_decreasing_interval ω → ω ∈ Icc (-4) 0 \ {0} :=
sorry

end range_of_omega_of_monotonic_decreasing_l460_460997


namespace remainder_is_400_l460_460696

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def num_functions (f : ℕ → ℕ) : ℕ :=
  if ∃ c ∈ A, ∀ x ∈ A, f(f(x)) = c then 1 else 0

def N : ℕ :=
  8 * (7 + 672 + 1701 + 1792 + 875 + 252 + 1)

def remainder : ℕ :=
  N % 1000

theorem remainder_is_400 : remainder = 400 :=
  sorry

end remainder_is_400_l460_460696


namespace zander_stickers_l460_460076

theorem zander_stickers (total_stickers andrew_ratio bill_ratio : ℕ) (initial_stickers: total_stickers = 100) (andrew_fraction : andrew_ratio = 1 / 5) (bill_fraction : bill_ratio = 3 / 10) :
  let andrew_give_away := total_stickers * andrew_ratio
  let remaining_stickers := total_stickers - andrew_give_away
  let bill_give_away := remaining_stickers * bill_ratio
  let total_given_away := andrew_give_away + bill_give_away
  total_given_away = 44 :=
by
  sorry

end zander_stickers_l460_460076


namespace renovation_project_cement_loads_l460_460482

theorem renovation_project_cement_loads
  (s : ℚ) (d : ℚ) (t : ℚ)
  (hs : s = 0.16666666666666666) 
  (hd : d = 0.3333333333333333)
  (ht : t = 0.6666666666666666) :
  t - (s + d) = 0.1666666666666666 := by
  sorry

end renovation_project_cement_loads_l460_460482


namespace solve_for_x_l460_460407

noncomputable def sum_of_factors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d, d ∣ n).sum

theorem solve_for_x (x : ℕ) (hx1 : sum_of_factors x = 18) (hx2 : 2 ∣ x) : x = 10 :=
by sorry

end solve_for_x_l460_460407


namespace period_cos_2x_l460_460059

theorem period_cos_2x : ∃ T : ℝ, (∀ x : ℝ, cos (2 * (x + T)) = cos (2 * x)) ∧ T = π := by
  sorry

end period_cos_2x_l460_460059


namespace problem_l460_460923

theorem problem (a b : ℝ) (h : a > b) : a / 3 > b / 3 :=
sorry

end problem_l460_460923


namespace population_is_24000_l460_460822

theorem population_is_24000 (P : ℝ) (h : 0.96 * P = 23040) : P = 24000 := sorry

end population_is_24000_l460_460822


namespace intersection_of_P_and_Q_l460_460322

def P : Set ℝ := {x | Real.log x > 0}
def Q : Set ℝ := {x | -1 < x ∧ x < 2}
def R : Set ℝ := {x | 1 < x ∧ x < 2}

theorem intersection_of_P_and_Q : P ∩ Q = R := by
  sorry

end intersection_of_P_and_Q_l460_460322


namespace octal_to_decimal_l460_460163

theorem octal_to_decimal (n_octal : ℕ) (h : n_octal = 123) : 
  let d0 := 3 * 8^0
  let d1 := 2 * 8^1
  let d2 := 1 * 8^2
  n_octal = 64 + 16 + 3 :=
by
  sorry

end octal_to_decimal_l460_460163


namespace emily_original_salary_l460_460895

def original_salary_emily (num_employees : ℕ) (original_employee_salary new_employee_salary new_salary_emily : ℕ) : ℕ :=
  new_salary_emily + (new_employee_salary - original_employee_salary) * num_employees

theorem emily_original_salary :
  original_salary_emily 10 20000 35000 850000 = 1000000 :=
by
  sorry

end emily_original_salary_l460_460895


namespace tree_height_at_2_years_l460_460853

theorem tree_height_at_2_years (h₅ : ℕ) (h_four : ℕ) (h_three : ℕ) (h_two : ℕ) (h₅_value : h₅ = 243)
  (h_four_value : h_four = h₅ / 3) (h_three_value : h_three = h_four / 3) (h_two_value : h_two = h_three / 3) :
  h_two = 9 := by
  sorry

end tree_height_at_2_years_l460_460853


namespace maximum_k_value_l460_460926

noncomputable def positive_reals := { x : ℝ // 0 < x }

theorem maximum_k_value (x y k : positive_reals) 
    (h : 4 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) : 
    k ≤ 1 :=
begin
    sorry
end

end maximum_k_value_l460_460926


namespace relationship_CD_BD_O_l460_460296

structure Point (ℝ : Type) :=
(x : ℝ)
(y : ℝ)

structure Triangle (ℝ : Type) :=
(A : Point ℝ)
(B : Point ℝ)
(C : Point ℝ)

def midpoint {ℝ : Type} [Add ℝ] [Div ℝ] (P Q : Point ℝ) : Point ℝ :=
{ x := (P.x + Q.x) / 2,
  y := (P.y + Q.y) / 2 }

def is_right_triangle {ℝ : Type} [Add ℝ] [Mul ℝ] (T : Triangle ℝ) : Prop :=
(T.A.x - T.B.x)^2 + (T.A.y - T.B.y)^2 + (T.B.x - T.C.x)^2 + (T.B.y - T.C.y)^2 = 
(T.A.x - T.C.x)^2 + (T.A.y - T.C.y)^2

noncomputable def distance {ℝ : Type} [Add ℝ] [Mul ℝ] [Pow ℝ nt] (P Q : Point ℝ) : ℝ :=
((Q.x - P.x)^2 + (Q.y - P.y)^2)^(1/2)

theorem relationship_CD_BD_O'D :
  let A := Point.mk 0 0,
      B := Point.mk 4 0,
      C := Point.mk 0 5, 
      T := Triangle.mk A B C,
      O := midpoint A B,
      O' := midpoint A B,
      D  := O
  in is_right_triangle T → distance C D = distance B D ∧ distance B D = distance O' D :=
by sorry

end relationship_CD_BD_O_l460_460296


namespace stickers_given_l460_460073

def total_stickers : ℕ := 100
def andrew_ratio : ℚ := 1 / 5
def bill_ratio : ℚ := 3 / 10

theorem stickers_given (zander_collection : ℕ)
                       (andrew_received : ℚ)
                       (bill_received : ℚ)
                       (total_given : ℚ):
  zander_collection = total_stickers →
  andrew_received = andrew_ratio →
  bill_received = bill_ratio →
  total_given = (andrew_received * zander_collection) + (bill_received * (zander_collection - (andrew_received * zander_collection))) →
  total_given = 44 :=
by
  intros hz har hbr htg
  sorry

end stickers_given_l460_460073


namespace total_surface_area_l460_460815

variables {Point} [AddGroup Point] [Module ℚ Point]

/-- Geometry setup for the problem -/
def is_trapezoid (A B C D : Point) : Prop :=
  ∃ E : Point, 
    (∃ ratio : ℚ, ratio = 2 / 5 ∧
      (B - C) = ratio • (A - D) ∧
      ∃ S : Point, 
        (∃ ratio_SE : ℚ, ratio_SE = 7 / 2 ∧
          let O := (7 / 9) • S + (2 / 9) • E in
          (lateral_area S B C D) = 8 
        )
    )
    
/-- The area function defined for pyramid lateral faces -/
def lateral_area (S B C D : Point) : ℚ := sorry

/-- The total surface area of the pyramid SABC, given the conditions -/
theorem total_surface_area (A B C D S : Point) :
  is_trapezoid A B C D →
  lateral_area S B C + lateral_area S A B + lateral_area S C D + lateral_area S A D = 126 :=
by
  intro h
  sorry

end total_surface_area_l460_460815


namespace volume_of_resulting_solid_is_9_l460_460920

-- Defining the initial cube with edge length 3
def initial_cube_edge_length : ℝ := 3

-- Defining the volume of the initial cube
def initial_cube_volume : ℝ := initial_cube_edge_length^3

-- Defining the volume of the resulting solid after some parts are cut off
def resulting_solid_volume : ℝ := 9

-- Theorem stating that given the initial conditions, the volume of the resulting solid is 9
theorem volume_of_resulting_solid_is_9 : resulting_solid_volume = 9 :=
by
  sorry

end volume_of_resulting_solid_is_9_l460_460920


namespace count_multiples_3_or_5_not_6_up_to_200_l460_460625

theorem count_multiples_3_or_5_not_6_up_to_200 : 
  (Finset.card (Finset.filter (λ n, n ≤ 200 ∧ ((n % 3 = 0 ∨ n % 5 = 0) ∧ n % 6 ≠ 0)) (Finset.range 201))) = 60 := 
by 
  sorry

end count_multiples_3_or_5_not_6_up_to_200_l460_460625


namespace number_of_lines_passing_through_integer_coordinates_on_circle_l460_460224

/-- Given the line ax + by = 2017 and the circle x^2 + y^2 = 100 intersect at points with integer coordinates, 
    prove that the total number of such lines is 72. -/
theorem number_of_lines_passing_through_integer_coordinates_on_circle :
  (∃ a b : ℤ, ∀ p : ℤ × ℤ, p ∈ {p : ℤ × ℤ | p.1^2 + p.2^2 = 100} → 
    a * p.1 + b * p.2 = 2017) → 
  {p : ℤ × ℤ | p.1^2 + p.2^2 = 100}.to_finset.card = 72 :=
sorry

end number_of_lines_passing_through_integer_coordinates_on_circle_l460_460224


namespace cube_root_of_prime_product_l460_460367

theorem cube_root_of_prime_product : (∛(2^9 * 5^3 * 7^3) = 280) :=
by
  sorry

end cube_root_of_prime_product_l460_460367


namespace multiples_3_or_5_not_6_l460_460631

theorem multiples_3_or_5_not_6 (n : ℕ) (hn : n ≤ 200) :
  card ({m | m ∣ n ∧ m ≤ 200 ∧ ((m % 3 = 0 ∨ m % 5 = 0) ∧ ¬ (m % 6 = 0))}) = 73 := sorry

end multiples_3_or_5_not_6_l460_460631


namespace find_divisor_l460_460083

theorem find_divisor (d : ℕ) (h1 : 109 % d = 1) (h2 : 109 / d = 9) : d = 12 := by
  sorry

end find_divisor_l460_460083


namespace LCM_1584_1188_l460_460803

open Nat

theorem LCM_1584_1188 :
  let a := 1584
  let b := 1188
  let a_factors : a = 2^4 * 3^3 * 11 := by sorry
  let b_factors : b = 2^2 * 3^3 * 11 := by sorry
  lcm a b = 4752 := by sorry

end LCM_1584_1188_l460_460803


namespace work_combined_days_l460_460455

theorem work_combined_days (A B C : ℝ) (hA : A = 1 / 4) (hB : B = 1 / 12) (hC : C = 1 / 6) :
  1 / (A + B + C) = 2 :=
by
  sorry

end work_combined_days_l460_460455


namespace alternate_interior_angles_equal_l460_460757

-- Defining the parallel lines and the third intersecting line
def Line : Type := sorry  -- placeholder type for a line

-- Predicate to check if lines are parallel
def parallel (l1 l2 : Line) : Prop := sorry

-- Predicate to represent a line intersecting another
def intersects (l1 l2 : Line) : Prop := sorry

-- Function to get interior alternate angles formed by the intersection
def alternate_interior_angles (l1 l2 : Line) (l3 : Line) : Prop := sorry

-- Theorem statement
theorem alternate_interior_angles_equal
  (l1 l2 l3 : Line)
  (h1 : parallel l1 l2)
  (h2 : intersects l3 l1)
  (h3 : intersects l3 l2) :
  alternate_interior_angles l1 l2 l3 :=
sorry

end alternate_interior_angles_equal_l460_460757


namespace inequality_proof_l460_460690

theorem inequality_proof {n : ℕ} (a : Fin n → ℝ) (h1 : ∀ i, 0 < a i) (h2 : (Finset.univ.sum (λ i, a i)) < 1) :
  (Finset.univ.prod a * (1 - Finset.univ.sum a)) /
  (Finset.univ.sum a * Finset.univ.prod (λ i, 1 - a i)) ≤ (1 / n^((n:ℕ)+1)) :=
sorry -- proof goes here

end inequality_proof_l460_460690


namespace tiffany_initial_lives_l460_460050

variable (x : ℝ) -- Define the variable x representing the initial number of lives

-- Define the conditions
def condition1 : Prop := x + 14.0 + 27.0 = 84.0

-- Prove the initial number of lives
theorem tiffany_initial_lives (h : condition1 x) : x = 43.0 := by
  sorry

end tiffany_initial_lives_l460_460050


namespace exp_problem_l460_460887

theorem exp_problem (a b c : ℕ) (H1 : a = 1000) (H2 : b = 1000^1000) (H3 : c = 500^1000) :
  a * b / c = 2^1001 * 500 :=
sorry

end exp_problem_l460_460887


namespace triangle_area_l460_460675

theorem triangle_area (A B C : ℝ) (AB BC CA : ℝ) (sinA sinB sinC : ℝ)
    (h1 : sinA * sinB * sinC = 1 / 1000) 
    (h2 : AB * BC * CA = 1000) : 
    (AB * BC * CA / (4 * 50)) = 5 :=
by
  -- Proof is omitted
  sorry

end triangle_area_l460_460675


namespace meaningful_expression_range_l460_460993

theorem meaningful_expression_range (a : ℝ) : (a + 1 ≥ 0) ∧ (a ≠ 2) ↔ (a ≥ -1) ∧ (a ≠ 2) :=
by
  sorry

end meaningful_expression_range_l460_460993


namespace alternating_sum_eq_neg50_l460_460507

theorem alternating_sum_eq_neg50 : 
  ((List.range 100).map (λ n, if n % 2 = 0 then n + 1 else - (n + 1))).sum = -50 := by
  sorry

end alternating_sum_eq_neg50_l460_460507


namespace triangle_area_is_correct_l460_460905

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_is_correct :
  area_of_triangle (0, 3) (4, -2) (9, 6) = 16.5 :=
by
  sorry

end triangle_area_is_correct_l460_460905


namespace coefficient_x2_expansion_l460_460540

theorem coefficient_x2_expansion :
  let expansion := (1 - 2 * x) ^ 5,
      term1 := 40 * x ^ 2,
      term2 := (-80) * x ^ 2 in
  (1 + 1 / x) * expansion = 1 + (1 / x) * (1 - 10 * x + term1 + (-80) * x ^ 3 + 80 * x ^ 4 + (-32) * x ^ 5) →
  (by simp only [term1, term2] : x^2) = -40 :=
by sorry

end coefficient_x2_expansion_l460_460540


namespace triangle_properties_l460_460662

theorem triangle_properties (PQ QR : ℝ) (Q : Prop) (cosQ : ℝ) 
  (PQ_length : PQ = 15) (cosQ_val : cosQ = 0.5) (right_angle : Q → ∃ θ, cos θ = cosQ) :
  QR = 30 ∧ (1 / 2 * PQ * QR = 225) :=
by
  have h : QR = 30 := sorry
  have area : (1 / 2 * PQ * QR = 225) := sorry
  exact ⟨h, area⟩

end triangle_properties_l460_460662
