import Data.Time
import Mathlib
import Mathlib.Algebra.BigOperators.Basic
import Mathlib.Algebra.Combinatorics
import Mathlib.Algebra.ConicSections
import Mathlib.Algebra.Field
import Mathlib.Algebra.GeomSum
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower.Identities
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Trigonometry.Triangle
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Graph
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.List.Sort
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Probability
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Init.Data.Nat.Basic
import Mathlib.Probability.Basic
import Mathlib.Tactic

namespace determine_a_range_l243_243305

theorem determine_a_range (a : ℝ) (h1 : ∀ n : ℕ, n ≥ 3 → 
  (∑ i in finset.range n, (1 : ℝ) / (n + i) + (5 / 12) * real.log (a - 1) > 1 / 5)) :
  (a > (1 + real.sqrt 5) / 2) :=
sorry

end determine_a_range_l243_243305


namespace algebraic_expression_evaluates_to_2_l243_243316

theorem algebraic_expression_evaluates_to_2 (x : ℝ) (h : x^2 + x - 5 = 0) : 
(x - 1)^2 - x * (x - 3) + (x + 2) * (x - 2) = 2 := 
by 
  sorry

end algebraic_expression_evaluates_to_2_l243_243316


namespace hypotenuse_length_l243_243085

variable (ABC : Triangle)
variable (M : Point)

def right_angled_triangle (ABC : Triangle) : Prop :=
  ABC.is_right_angled

def median_to_hypotenuse (ABC : Triangle) (M : Point) : Prop :=
  ABC.median_to M ∧ ABC.hypotenuse_contains M ∧ (dist ABC.C M) = 3

theorem hypotenuse_length (ABC : Triangle) (M : Point) 
  (h1 : right_angled_triangle ABC)
  (h2 : median_to_hypotenuse ABC M) :
  hypotenuse_length ABC = 6 :=
sorry

end hypotenuse_length_l243_243085


namespace counting_arithmetic_progressions_l243_243442

open Finset

/-- The number of ways to choose 4 numbers from the first 1000 natural numbers to form an increasing arithmetic progression is 166167. -/
theorem counting_arithmetic_progressions :
  let n := 1000 in
  let count := ∑ d in range 334, n - 3*d in
  count = 166167 :=
by
  sorry

end counting_arithmetic_progressions_l243_243442


namespace solution_p_l243_243016

noncomputable def find_p (a : ℝ) (p : ℝ) : Prop :=
  (a > 0) ∧ (p > 0) ∧ (∃ (b : ℝ), b = sqrt 3 ∧ 
  (∀ (x y : ℝ), (x = 2 ∧ y = sqrt 3 → y = (b / a) * x ∨ y = -(b / a) * x)) ∧ 
  ∃ (f1 f2 : ℝ), (f1 = sqrt (a^2 + b^2) ∧ f2 = -sqrt (a^2 + b^2)) ∧ 
  (f1 = sqrt (a^2 + b^2) → f1 = sqrt 7 ∨ f2 = sqrt 7) ∧
  (sqrt 7 - 0) * (sqrt 7 - 0) = 7)

theorem solution_p : ∃ p, find_p 2 p := sorry

end solution_p_l243_243016


namespace log2_square_eq_37_l243_243516

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_square_eq_37
  {x y : ℝ}
  (hx : x ≠ 1)
  (hy : y ≠ 1)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_log : log2 x = Real.log 8 / Real.log y)
  (h_prod : x * y = 128) :
  (log2 (x / y))^2 = 37 := by
  sorry

end log2_square_eq_37_l243_243516


namespace acute_angle_with_x_axis_l243_243851

theorem acute_angle_with_x_axis :
  let P : ℝ × ℝ := (sqrt 3, -2)
  let Q : ℝ × ℝ := (0, 1)
  let m : ℝ := (1 - (-2)) / (0 - sqrt 3)
  let θ := Real.arctan m
  θ = 60 * Real.pi / 180 :=
by
  sorry

end acute_angle_with_x_axis_l243_243851


namespace largest_side_l243_243100

-- Definitions of conditions from part (a)
def perimeter_eq (l w : ℝ) : Prop := 2 * l + 2 * w = 240
def area_eq (l w : ℝ) : Prop := l * w = 2880

-- The main proof statement
theorem largest_side (l w : ℝ) (h1 : perimeter_eq l w) (h2 : area_eq l w) : l = 72 ∨ w = 72 :=
by
  sorry

end largest_side_l243_243100


namespace factorial_product_less_than_l243_243128

theorem factorial_product_less_than {
  n : ℕ,
  k : ℕ,
  a : fin n → ℕ
} (h1 : ∀ i, 0 < a i) (h2 : (∑ i, a i) < k) :
  (∏ i, (a i)!) < k! := 
sorry

end factorial_product_less_than_l243_243128


namespace smallest_four_digit_divisible_by_53_l243_243673

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243673


namespace max_ratio_OB_OA_l243_243082

open Real

noncomputable def C1_polar := ∀ (ρ θ : ℝ), 
  (sqrt 3 * ρ * cos θ + ρ * sin θ - 4 = 0)

noncomputable def C2_polar := ∀ (θ : ℝ), 
  (rho = 2 * sin θ)

noncomputable def C3_intersect_ρ1 (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ := 
  4 / (sqrt 3 * cos α + sin α)

noncomputable def C3_intersect_ρ2 (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ := 
  2 * sin α

noncomputable def ratio_OB_OA (α : ℝ) (hα : 0 < α ∧ α < π / 2) : ℝ :=
  (2 * sin (2 * α - π / 6) + 1) / 4

theorem max_ratio_OB_OA :
  ∀ (α : ℝ) (hα : 0 < α ∧ α < π / 2),
  (ratio_OB_OA α hα = 3 / 4) ↔ (α = π / 3) :=
sorry

end max_ratio_OB_OA_l243_243082


namespace sum_cis_angles_l243_243271

theorem sum_cis_angles (r : ℝ) (h : 0 < r) :
  (complex.exp (complex.I * real.pi * 60 / 180) +
   complex.exp (complex.I * real.pi * 70 / 180) +
   complex.exp (complex.I * real.pi * 80 / 180) +
   complex.exp (complex.I * real.pi * 90 / 180) +
   complex.exp (complex.I * real.pi * 100 / 180) +
   complex.exp (complex.I * real.pi * 110 / 180) +
   complex.exp (complex.I * real.pi * 120 / 180) +
   complex.exp (complex.I * real.pi * 130 / 180) +
   complex.exp (complex.I * real.pi * 140 / 180))
   = r * complex.exp (complex.I * real.pi * 100 / 180) :=
sorry

end sum_cis_angles_l243_243271


namespace magnitude_2a_minus_b_l243_243338

open Real

-- Define vectors a and b in R^n with given magnitudes and angle between them
variables {n : Type*} [NormedGroup n] [NormedSpace ℝ n] 
variables (a b : n)
variables (θ : ℝ) (cosθ : ℝ) [fact (cosθ = cos 60)]

-- Conditions
axiom angle_condition : θ = π / 3
axiom magnitude_a : ∥a∥ = 1
axiom magnitude_b : ∥b∥ = 2
axiom dot_product_ab : ⟪a, b⟫ = ∥a∥ * ∥b∥ * cosθ

-- Question turned into a proof statement
theorem magnitude_2a_minus_b : ∥2 • a - b∥ = 2 :=
  sorry

end magnitude_2a_minus_b_l243_243338


namespace constant_term_expansion_l243_243597

theorem constant_term_expansion :
  (∃ k : ℕ, k ∈ finset.range 13 ∧ 
            (∃ (term : ℚ), term = binom 12 k * (x ^ (k / 3) * (4 / x ^ 2) ^ (12 - k))) ∧
            is_constant term) →
  term.coeff = 126720 :=
by
  sorry

end constant_term_expansion_l243_243597


namespace part_I_part_II_l243_243336

section Problem

def point_F1 : (ℝ × ℝ) := (-1, 0)
def point_F2 : (ℝ × ℝ) := (1, 0)
def circle_F1 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def ellipse_C (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

theorem part_I : 
  (∀ (M : ℝ × ℝ), ∃ P : ℝ × ℝ, 
    circle_F1 P.1 P.2 ∧ 
    (|M.1 - point_F1.1| + |M.2 - point_F1.2|) + (|M.1 - point_F2.1| + |M.2 - point_F2.2|) = 4 
    → ellipse_C M.1 M.2) := sorry

theorem part_II : 
  (∃ l : ℝ × ℝ → Prop, ∀ B1 B2 : ℝ × ℝ, 
    l B1 ∧ l B2 ∧ ellipse_C B1.1 B1.2 ∧ ellipse_C B2.1 B2.2 ∧ (∃ A1 A2 : ℝ × ℝ, parabola A1.1 A1.2 ∧ parabola A2.1 A2.2 ∧ |A1.1 - A2.1| = |A1.2 - A2.2| = 0) 
    → |A1.1 - A2.1| = 64/9) := sorry

end Problem

end part_I_part_II_l243_243336


namespace smallest_four_digit_divisible_by_53_l243_243752

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243752


namespace cos_sum_identity_l243_243572

theorem cos_sum_identity :
  (Real.cos (75 * Real.pi / 180)) ^ 2 + (Real.cos (15 * Real.pi / 180)) ^ 2 + 
  (Real.cos (75 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 5 / 4 := 
by
  sorry

end cos_sum_identity_l243_243572


namespace least_side_is_8_l243_243394

-- Define the sides of the right triangle
variables (a b : ℝ) (h : a = 8) (k : b = 15)

-- Define a predicate for the least possible length of the third side
def least_possible_third_side (c : ℝ) : Prop :=
  (c = 8) ∨ (c = 15) ∨ (c = 17)

theorem least_side_is_8 (c : ℝ) (hc : least_possible_third_side c) : c = 8 :=
by
  sorry

end least_side_is_8_l243_243394


namespace sin_double_angle_l243_243462

theorem sin_double_angle (α : ℝ) (h_sin : Real.sin α = (√3) / 2) (h_cos : Real.cos α = 1 / 2) :
   Real.sin (2 * α) = (√3) / 2 := 
by
  sorry

end sin_double_angle_l243_243462


namespace smallest_four_digit_divisible_by_53_l243_243736

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243736


namespace simplify_expression_l243_243531

theorem simplify_expression (a : ℝ) :
  (1/2) * (8 * a^2 + 4 * a) - 3 * (a - (1/3) * a^2) = 5 * a^2 - a :=
by
  sorry

end simplify_expression_l243_243531


namespace speed_in_still_water_l243_243853

-- Define all given conditions as constants
constant speed_of_current_kmh : ℝ := 3
constant distance_meters : ℝ := 90
constant time_seconds : ℝ := 17.998560115190784

-- Define a function to convert speed from km/hr to m/s
def kmh_to_mps (v_kmh : ℝ) : ℝ := v_kmh * (1000 / 3600)

-- Define the downstream speed in m/s
def downstream_speed_mps : ℝ := distance_meters / time_seconds

-- Define the man's speed in still water in m/s
def speed_in_still_water_mps : ℝ := downstream_speed_mps - kmh_to_mps(speed_of_current_kmh)

-- Define a function to convert speed from m/s to km/hr
def mps_to_kmh (v_mps : ℝ) : ℝ := v_mps * (3600 / 1000)

-- State the necessary Lean theorem
theorem speed_in_still_water (v_m_kmh : ℝ) : 
  v_m_kmh = mps_to_kmh(speed_in_still_water_mps) := 
by 
  -- automatically simplify and prove 
  unfold speed_in_still_water_mps downstream_speed_mps kmh_to_mps mps_to_kmh 
  sorry

-- Correct answer: 25 km/hr
#reduce speed_in_still_water_mps -- 25 km/hr (expected value)

end speed_in_still_water_l243_243853


namespace solution_set_inequality_l243_243026

def g (x : ℝ) : ℝ := 2016 ^ x + Real.log (sqrt (x ^ 2 + 1) + x) / Real.log 2016 - 2016 ^ (-x)
def f (x : ℝ) : ℝ := g x + 2

theorem solution_set_inequality :
  {x : ℝ | f (3 * x + 1) + f x > 4} = {x : ℝ | x > -1 / 4} :=
by
  sorry

end solution_set_inequality_l243_243026


namespace germination_probability_l243_243557

open Nat

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def probability_of_success (p : ℚ) (k : ℕ) (n : ℕ) : ℚ :=
  (binomial_coeff n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem germination_probability :
  probability_of_success 0.9 5 7 = 0.124 := by
  sorry

end germination_probability_l243_243557


namespace standard_eq_circle_l243_243300

noncomputable def circle_eq (x y : ℝ) (r : ℝ) : Prop :=
  (x - 4)^2 + (y - 4)^2 = 16 ∨ (x - 1)^2 + (y + 1)^2 = 1

theorem standard_eq_circle {x y : ℝ}
  (h1 : 5 * x - 3 * y = 8)
  (h2 : abs x = abs y) :
  ∃ r : ℝ, circle_eq x y r :=
by {
  sorry
}

end standard_eq_circle_l243_243300


namespace smallest_positive_debt_l243_243199

theorem smallest_positive_debt (c s : ℤ) : ∃ D : ℤ, D > 0 ∧ (∀ c s : ℤ, D = 400 * c + 250 * s) → D = 50 :=
begin
  -- conditions
  let cows_value := 400,
  let sheep_value := 250,

  -- proof
  sorry
end

end smallest_positive_debt_l243_243199


namespace smallest_four_digit_divisible_by_53_l243_243694

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243694


namespace pf1_pf2_eq_m_minus_a_l243_243342

-- Define the given conditions
variables (x y m n a b : ℝ)
variable (F1 F2 P : ℝ)

-- Define the ellipse and hyperbola
def ellipse := x^2 / m + y^2 / n = 1
def hyperbola := x^2 / a + y^2 / b = 1

-- Define the foci and intersection point conditions
def same_foci : Prop := F1 = F1 ∧ F2 = F2
def intersection_point : Prop := (x, y) ∈ ellipse ∧ (x, y) ∈ hyperbola

-- The final proof problem
theorem pf1_pf2_eq_m_minus_a
  (H1 : m > n) (H2 : n > 0) (H3 : a > b) (H4 : b > 0)
  (H5 : same_foci ∧ intersection_point):
  |P - F1| * |P - F2| = m - a := 
  sorry

end pf1_pf2_eq_m_minus_a_l243_243342


namespace find_initial_trees_per_row_l243_243098

noncomputable def initial_trees_per_row (x : ℕ) : Prop :=
  let initial_trees := 2 * x in
  let planted_trees := 5 * x in
  let total_trees_before_doubling := initial_trees + planted_trees in
  let total_trees_after_doubling := 2 * total_trees_before_doubling in
  total_trees_after_doubling = 56

theorem find_initial_trees_per_row : initial_trees_per_row 4 :=
by
  let x := 4
  let initial_trees := 2 * x
  let planted_trees := 5 * x
  let total_trees_before_doubling := initial_trees + planted_trees
  let total_trees_after_doubling := 2 * total_trees_before_doubling
  show total_trees_after_doubling = 56
  calc
    total_trees_after_doubling = 2 * total_trees_before_doubling : by rfl
    ... = 2 * (initial_trees + planted_trees) : by rfl
    ... = 2 * (2 * x + 5 * x) : by rfl
    ... = 2 * 7 * x : by ring
    ... = 14 * x : by ring
    ... = 14 * 4 : by rw [show x = 4 from rfl]
    ... = 56 : by norm_num

end find_initial_trees_per_row_l243_243098


namespace smallest_four_digit_multiple_of_53_l243_243709

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243709


namespace log_equivalence_l243_243372

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_equivalence (x : ℝ) (h : log_base 16 (x - 3) = 1 / 2) : log_base 256 (x + 1) = 3 / 8 :=
  sorry

end log_equivalence_l243_243372


namespace distinct_remainders_exists_l243_243124

theorem distinct_remainders_exists (n : ℕ) (hn : n % 2 = 1) :
  ∃ (a b : ℕ → ℕ), 
    (∀ i, 1 ≤ i ∧ i ≤ n → a i = 3 * i - 2) ∧
    (∀ i, 1 ≤ i ∧ i ≤ n → b i = 3 * i - 3) ∧
    (∀ i k, 1 ≤ i ∧ i ≤ n ∧ 0 < k ∧ k < n → 
      let j := ((i + k - 1) % n) + 1 in
      let a_next := a ((i % n) + 1) in
      (a i + a_next) % (3 * n) ≠ (a i + b i) % (3 * n) ∧
      (a i + a_next) % (3 * n) ≠ (b i + b j) % (3 * n) ∧
      (a i + b i) % (3 * n) ≠ (b i + b j) % (3 * n)) :=
begin
  sorry
end

end distinct_remainders_exists_l243_243124


namespace correct_statements_l243_243276

variable {R : Type*} [RealField R]

-- Define the conditions
def even_function (f : R → R) : Prop := ∀ x, f (-x) = f x

def negated_after_one (f : R → R) : Prop := ∀ x, f (x + 1) = -f x

def increasing_on_interval (f : R → R) (a b : R) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

-- Define the question as propositions
def periodic_function (f : R → R) : Prop := ∃ p, p > 0 ∧ ∀ x, f (x + p) = f x

def symmetric_about_l (f : R → R) (l : R) : Prop := ∀ x, f (-x + 2 * l) = f (x + 2 * l)

def decreasing_on_interval (f : R → R) (a b : R) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≥ f y

def f2_eq_f0 (f : R → R) : Prop := f 2 = f 0

-- The main theorem to be proven
theorem correct_statements (f : R → R) (l : R)
  (h1 : even_function f)
  (h2 : negated_after_one f)
  (h3 : increasing_on_interval f (-1) 0) :
  periodic_function f ∧ symmetric_about_l f l ∧ f2_eq_f0 f :=
begin
  sorry
end

end correct_statements_l243_243276


namespace midpoint_B_of_collinear_and_equation_l243_243337

section Vectors

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (O A B C : V) (m : ℝ)

theorem midpoint_B_of_collinear_and_equation
  (h_collinear : ∃ k1 k2 k3, k1 • (A - O) + k2 • (B - O) + k3 • (C - O) = 0)
  (h_eq : m • A - 2 • B + C = 0) :
  B = (1/2:ℝ) • (A + C) :=
by sorry

end Vectors

end midpoint_B_of_collinear_and_equation_l243_243337


namespace no_solution_l243_243518

theorem no_solution : ∀ x y z t : ℕ, 16^x + 21^y + 26^z ≠ t^2 :=
by
  intro x y z t
  sorry

end no_solution_l243_243518


namespace find_triangle_heights_l243_243436

def triangle_heights : Prop :=
  ∃ (h1 h2 : ℝ), h1 + h2 = 23.2 ∧ (14 * h1 = 15 * h2) ∧ h1 ≈ 12 ∧ h2 ≈ 11.2

theorem find_triangle_heights : triangle_heights := sorry

end find_triangle_heights_l243_243436


namespace distinct_real_numbers_f_iter_eq_neg4_l243_243122

def f (x : ℝ) : ℝ := x^2 + 2 * x

theorem distinct_real_numbers_f_iter_eq_neg4 : 
  { c : ℝ | f (f (f (f c))) = -4 }.to_finset.card = 0 := by
  sorry

end distinct_real_numbers_f_iter_eq_neg4_l243_243122


namespace a5_value_l243_243087

noncomputable def sequence (n : ℕ) : ℚ :=
  match n with
  | 1 => 5 / 2
  | 2 => 1
  | n + 3 => (2 * sequence (n + 1)) / (sequence (n + 2))
  | _ => 0

theorem a5_value : sequence 5 = 25 := by
  sorry

end a5_value_l243_243087


namespace nine_a_minus_six_b_l243_243369

-- Define the variables and conditions.
variables (a b : ℚ)

-- Assume the given conditions.
def condition1 : Prop := 3 * a + 4 * b = 0
def condition2 : Prop := a = 2 * b - 3

-- Formalize the statement to prove.
theorem nine_a_minus_six_b (h1 : condition1 a b) (h2 : condition2 a b) : 9 * a - 6 * b = -81 / 5 :=
sorry

end nine_a_minus_six_b_l243_243369


namespace number_of_people_in_first_group_l243_243538

-- Define variables representing the work done by one person in one day (W) and the number of people in the first group (P)
variable (W : ℕ) (P : ℕ)

-- Conditions from the problem
-- Some people can do 3 times a particular work in 3 days
def condition1 : Prop := P * 3 * W = 3 * W

-- It takes 6 people 3 days to do 6 times of that particular work
def condition2 : Prop := 6 * 3 * W = 6 * W

-- The statement to prove
theorem number_of_people_in_first_group 
  (h1 : condition1 W P) 
  (h2 : condition2 W) : P = 3 :=
by
  sorry

end number_of_people_in_first_group_l243_243538


namespace total_playing_time_situations_l243_243883

open Nat

theorem total_playing_time_situations :
  let div7 (n : ℕ) := ∃ k, n = 7 * k
  let div13 (n : ℕ) := ∃ k, n = 13 * k
  (∃ x1 x2 x3 x4 x5 x6 x7 : ℕ, div7 x1 ∧ div7 x2 ∧ div7 x3 ∧ div7 x4 ∧ div13 x5 ∧ div13 x6 ∧ div13 x7 ∧ x1 + x2 + x3 + x4 + x5 + x6 + x7 = 270) =
  142286 :=
sorry

end total_playing_time_situations_l243_243883


namespace ice_cream_stack_orders_l243_243142

def scoops : list (fin 5 → nat) := 
  [1, 1, 1, 1, 2] -- vanilla, chocolate, strawberry, cherry, mint, mint

theorem ice_cream_stack_orders :
  let total_scoops := 5
  let arrangement := @finset.cardinal_factors _ (λ _ _, quotient (ell_eq _)) scoops.total_scoops_eq_scoops_card_eq
  let identical_mints := multiset.card 
  (multiset.card scoops.count_2) -- Two identical mints
  in  arrangement / identical_mints = 60 := sorry

end ice_cream_stack_orders_l243_243142


namespace smallest_four_digit_multiple_of_53_l243_243713

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243713


namespace annual_profit_expression_maximum_profit_at_4e_l243_243447

noncomputable def P (x : ℝ) : ℝ :=
  if h : x < 10 then
    11 - (x + 9 / x)
  else
    10 - (real.log x + 4 * real.exp 1 / x)

theorem annual_profit_expression (x : ℝ) :
  (P x = if h : x < 10 then 11 - (x + 9 / x) else 10 - (real.log x + 4 * real.exp 1 / x)) :=
sorry

theorem maximum_profit_at_4e :
  (∀ x > 0, P x ≤ P (4 * real.exp 1)) ∧ 
  (P (4 * real.exp 1) ≈ 6.6) :=
sorry

end annual_profit_expression_maximum_profit_at_4e_l243_243447


namespace find_second_number_l243_243228

theorem find_second_number (x : ℕ) : 9548 + x = 3362 + 13500 → x = 7314 := by
  sorry

end find_second_number_l243_243228


namespace final_brown_marble_count_is_40_5_l243_243846

noncomputable def final_brown_count : ℚ :=
let total_marbles: ℚ := 18 / 0.15 in
let blue_initial: ℚ := 0.25 * total_marbles in
let brown_initial: ℚ := 0.25 * total_marbles in
let blue_to_brown: ℚ := 0.5 * blue_initial in
let brown_new: ℚ := brown_initial + blue_to_brown in
let brown_removed: ℚ := 0.10 * brown_new in
brown_new - brown_removed

theorem final_brown_marble_count_is_40_5 :
  final_brown_count = 40.5 := by
  sorry

end final_brown_marble_count_is_40_5_l243_243846


namespace probability_no_two_green_hats_next_to_each_other_l243_243822

open Nat

def choose (n k : ℕ) : ℕ := Nat.fact n / (Nat.fact k * Nat.fact (n - k))

def total_ways_to_choose (n k : ℕ) : ℕ :=
  choose n k

def event_A (n : ℕ) : ℕ := n - 2

def event_B (n k : ℕ) : ℕ := choose (n - k + 1) 2 * (k - 2)

def probability_no_two_next_to_each_other (n k : ℕ) : ℚ :=
  let total_ways := total_ways_to_choose n k
  let event_A_ways := event_A (n)
  let event_B_ways := event_B n 3
  let favorable_ways := total_ways - (event_A_ways + event_B_ways)
  favorable_ways / total_ways

-- Given the conditions of 9 children and choosing 3 to wear green hats
theorem probability_no_two_green_hats_next_to_each_other (p : probability_no_two_next_to_each_other 9 3 = 5 / 14) : Prop := by
  sorry

end probability_no_two_green_hats_next_to_each_other_l243_243822


namespace problem_statement_l243_243340

noncomputable def circle_equation (a b r : ℝ) : Prop :=
  ∃ a = 1 ∧ b = 0 ∧ (x - 1)^2 + y^2 = r^2 ∧ 
  ∃ (line : ℝ → ℝ → ℝ → ℝ → ℝ) 
  (focus : ℝ × ℝ) (parabola : ℝ) (eq := (x - 1)^2 + y^2 - 1), 
  line 3 4 x y + 2 = tangent (x y)

theorem problem_statement : 
  ∀ (x y : ℝ), (x - 1)^2 + y^2 = 1 ↔ 
  (cond : ∀ (a b r : ℝ), ∃ a = 1 ∧ b = 0, 
  focus (parabola 1 a r) = tangent (x y) :
  circle_equation a b r) := 
by
  -- Proof omitted
  sorry

end problem_statement_l243_243340


namespace polynomial_solution_l243_243477

noncomputable def satisfies_polynomial_condition (P : ℝ → ℝ) (k : ℕ) : Prop :=
  ∀ x : ℝ, P (P x) = P x ^ k

theorem polynomial_solution (k : ℕ) (P : ℝ → ℝ) (h_pos : 0 < k) :
  satisfies_polynomial_condition P k ↔ 
  (∃ c : ℝ, P = λ x, c ∧ (c ≠ 0 ∧ (k = 1 ∨ (∃ m : ℕ, k = 2*m))) ) ∨ 
  (∃ n : ℕ, n = k ∧ P = λ x, x^n) :=
by
  -- proof will be filled in
  sorry

end polynomial_solution_l243_243477


namespace time_shortened_by_opening_both_pipes_l243_243234

theorem time_shortened_by_opening_both_pipes 
  (a b p : ℝ) 
  (hp : a * p > 0) -- To ensure p > 0 and reservoir volume is positive
  (h1 : p = (a * p) / a) -- Given that pipe A alone takes p hours
  : p - (a * p) / (a + b) = (b * p) / (a + b) := 
sorry

end time_shortened_by_opening_both_pipes_l243_243234


namespace ceil_sqrt_225_eq_15_l243_243970

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := by
  sorry

end ceil_sqrt_225_eq_15_l243_243970


namespace number_of_chicks_is_8_l243_243573

-- Define the number of total chickens
def total_chickens : ℕ := 15

-- Define the number of hens
def hens : ℕ := 3

-- Define the number of roosters
def roosters : ℕ := total_chickens - hens

-- Define the number of chicks
def chicks : ℕ := roosters - 4

-- State the main theorem
theorem number_of_chicks_is_8 : chicks = 8 := 
by
  -- the solution follows from the given definitions and conditions
  sorry

end number_of_chicks_is_8_l243_243573


namespace fraction_subtraction_l243_243889

theorem fraction_subtraction :
  (15 / 45) - (1 + (2 / 9)) = - (8 / 9) :=
by
  sorry

end fraction_subtraction_l243_243889


namespace ceil_sqrt_225_l243_243938

theorem ceil_sqrt_225 : Nat.ceil (Real.sqrt 225) = 15 :=
by
  sorry

end ceil_sqrt_225_l243_243938


namespace smallest_four_digit_multiple_of_53_l243_243770

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243770


namespace inscribe_parallelogram_in_quadrilateral_l243_243321

variables {A B C D E F P Q R S : Type*} [affine_plane ℝ] 
(hAB : line A B) (hCD : line C D) 
(hE : E ∈ hAB) (hF : F ∈ hCD)
(directionBF : line B F) (directionCE : line C E)

theorem inscribe_parallelogram_in_quadrilateral 
  (hPQRS : parallelogram P Q R S)
  (hP : P ∈ ray B A) (hR : R ∈ ray C D) (hQ : Q ∈ segment B C) :
  ∃ S ∈ segment E F, 
    parallelogram P Q R S ∧
    (∃ directionBF ∥ line B F) ∧
    (∃ directionCE ∥ line C E) :=
begin
  sorry,
end

end inscribe_parallelogram_in_quadrilateral_l243_243321


namespace probability_no_adjacent_green_hats_l243_243818

-- Definitions
def total_children : ℕ := 9
def green_hats : ℕ := 3

-- Main theorem statement
theorem probability_no_adjacent_green_hats : 
  (9.choose 3) = 84 → 
  (1 - (9 + 45) / 84) = 5/14 := 
sorry

end probability_no_adjacent_green_hats_l243_243818


namespace FE_eq_2FC_l243_243137

-- Define the conditions.
theorem FE_eq_2FC {A B C D P Q E F : Type}
  (h_square : square A B C D)
  (h_P_on_BC : P ∈ segment B C)
  (h_Q_on_CD : Q ∈ segment C D)
  (h_eq_tri_APQ : equilateral A P Q)
  (h_line_PE_perp_AQ : ∃ E, line_through_P E ⊥ line_through_AQ)
  (h_E_on_AD : E ∈ segment A D)
  (h_triangles_congruent : congruent (triangle P Q F) (triangle A Q E)) :
FE = 2 * FC :=
by
  sorry

end FE_eq_2FC_l243_243137


namespace convert_volumes_correctly_l243_243248

-- Definitions for the conditions
def volume_in_cubic_feet : ℝ := 216
def cubic_feet_to_cubic_yards : ℝ := 1 / 27
def cubic_feet_to_cubic_meters : ℝ := 0.028317

-- Definitions for the expected answers
def expected_volume_in_cubic_yards : ℝ := 8
def expected_volume_in_cubic_meters : ℝ := 6.116472

-- Proposition to prove equivalency
theorem convert_volumes_correctly 
    (V_ft : ℝ := volume_in_cubic_feet)
    (conv_y : ℝ := cubic_feet_to_cubic_yards)
    (conv_m : ℝ := cubic_feet_to_cubic_meters)
    (V_yd : ℝ := V_ft * conv_y)
    (V_m : ℝ := V_ft * conv_m) :
  V_yd = expected_volume_in_cubic_yards ∧ |V_m - expected_volume_in_cubic_meters| < 1e-6 := 
by
  -- Here we would prove the statement but we skip it with sorry
  sorry

end convert_volumes_correctly_l243_243248


namespace cos420_plus_sin330_eq_zero_l243_243189

theorem cos420_plus_sin330_eq_zero :
  cos (420 * (π / 180)) + sin (330 * (π / 180)) = 0 :=
by
  have h1 : cos (420 * (π / 180)) = cos (60 * (π / 180)), from sorry,
  have h2 : sin (330 * (π / 180)) = -sin (30 * (π / 180)), from sorry,
  have h3 : cos (60 * (π / 180)) = 1 / 2, from sorry,
  have h4 : sin (30 * (π / 180)) = 1 / 2, from sorry,
  rw [h1, h2],
  rw [h3, h4],
  norm_num

end cos420_plus_sin330_eq_zero_l243_243189


namespace smallest_four_digit_divisible_by_53_l243_243722

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l243_243722


namespace find_m_plus_b_l243_243169

noncomputable theory

def reflection_translation (p1 p2 : ℝ × ℝ) (m b : ℝ) (translation : ℝ) : Prop :=
  let reflect (p : ℝ × ℝ) : ℝ × ℝ := 
    let (x, y) := p in 
    ((1 - m^2)/(1 + m^2) * x + 2 * m/(1 + m^2) * y - 2 * m * b/(1 + m^2), 
     2 * m/(1 + m^2) * x + (m^2 - 1)/(1 + m^2) * y + 2 * b/(1 + m^2))
  in 
  let translate (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 + translation) in
  translate (reflect p1) = p2

theorem find_m_plus_b : 
  ∃ m b : ℝ, reflection_translation (2, 4) (10, 14) m b 3 ∧ (m + b = 13.357) := sorry

end find_m_plus_b_l243_243169


namespace S10_eq_210_l243_243554

def floor (x : ℝ) : ℤ := Int.floor x

def sequence_sum (n : ℕ) : ℤ :=
  let start := n^2
  let end_ := (n + 1)^2 - 1
  ∑ i in Finset.range (end_ - start + 1), floor (Real.sqrt (start + i))

theorem S10_eq_210 : sequence_sum 10 = 210 := 
by 
  sorry

end S10_eq_210_l243_243554


namespace right_triangle_legs_l243_243177

theorem right_triangle_legs (a b : ℝ) (r R : ℝ) (hypotenuse : ℝ) (h_ab : a + b = 14) (h_c : hypotenuse = 10)
  (h_leg: a * b = a + b + 10) (h_Pythag : a^2 + b^2 = hypotenuse^2) 
  (h_inradius : r = 2) (h_circumradius : R = 5) : (a = 6 ∧ b = 8) ∨ (a = 8 ∧ b = 6) :=
by
  sorry

end right_triangle_legs_l243_243177


namespace smallest_four_digit_divisible_by_53_l243_243644

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l243_243644


namespace smallest_four_digit_multiple_of_53_l243_243779

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l243_243779


namespace ceil_sqrt_225_l243_243933

theorem ceil_sqrt_225 : Nat.ceil (Real.sqrt 225) = 15 :=
by
  sorry

end ceil_sqrt_225_l243_243933


namespace range_of_a_l243_243370

theorem range_of_a (a : ℝ) (A B : set ℝ) (hA : A = {x | x > a}) (hB : B = {x | x > 6}) (h : A ⊆ B) : a ≥ 6 :=
by
  sorry

end range_of_a_l243_243370


namespace growing_path_product_l243_243548

-- Define a 4x4 point set
def point := (ℕ × ℕ)
def rect_array := { p : point // 0 ≤ p.1 ∧ p.1 < 4 ∧ 0 ≤ p.2 ∧ p.2 < 4 }

-- Define the concept of a growing path
def growing_path (path : Finset point) : Prop :=
  ∀ p1 p2 p3 ∈ path, (dist p1 p2 < dist p2 p3)

-- Define the distance function
def dist (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define maximum points in a growing path and count of such paths
noncomputable def m : ℕ := 10
noncomputable def r : ℕ := 24

-- Prove the product of m and r
theorem growing_path_product : m * r = 240 :=
by 
  have mp := m 
  have rp := r 
  have h1 : m = 10 := by sorry 
  have h2 : r = 24 := by sorry
  rw [h1, h2]
  exact two_mul 12

end growing_path_product_l243_243548


namespace smallest_four_digit_divisible_by_53_l243_243700

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243700


namespace omega_value_monotonic_intervals_l243_243042

noncomputable def a (ω x : ℝ) : ℝ × ℝ := (sin (ω * x), 1)

noncomputable def b (ω x : ℝ) : ℝ × ℝ := (sqrt 3, -cos (ω * x))

noncomputable def f (ω x : ℝ) : ℝ := (a ω x).fst * (b ω x).fst + (a ω x).snd * (b ω x).snd

axiom ω_is_positive (ω : ℝ) : ω > 0

theorem omega_value (ω : ℝ) (h : ∀ x, f ω x = 2 * sin (ω * x - π / 6)) : ω = 2 :=
by
  sorry

theorem monotonic_intervals (ω: ℝ) (hω: ω = 2) :
  (∀ x ∈ Icc (0 : ℝ) (π : ℝ), 0 ≤ x ∧ x ≤ π / 3 ∨ 5 * π / 6 ≤ x ∧ x ≤ π → 
  ∀ x₁ x₂ ∈ Icc (0 : ℝ) (π : ℝ), x₁ ≤ x₂ → f 2 x₁ ≤ f 2 x₂) :=
by
  sorry

end omega_value_monotonic_intervals_l243_243042


namespace eccentricity_of_hyperbola_equation_of_hyperbola_passing_through_equation_of_line_AB_l243_243353

-- Given conditions definitions
def hyperbola_eq (a b : ℝ) (x y : ℝ) := (y^2)/(a^2) - (x^2)/(b^2) = 1
def point_on_hyperbola (a b x y : ℝ) := hyperbola_eq a b x y
def point (x y : ℝ) := (x, y)

-- Proof statements
theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (F1 F2 O M P : point ℝ ℝ)
    (cond1 : vector_eq_or (F2 O) = vector_eq_or (M P))
    (cond2 : vector_eq_or (F1 M) = λ v, λ λ (vector_eq_or (F1 P) / abs (vector_eq_or (F1 P)) + 
                                        vector_eq_or (F1 O) / abs (vector_eq_or (F1 O)))) :
    eccentricity a b = 2 :=
sorry

theorem equation_of_hyperbola_passing_through (F1 F2 O M P : point ℝ ℝ) (N : point ℝ ℝ) (a b : ℝ)
    (e : ℝ) (h1 : e = 2) (h_point : point_on_hyperbola a b (sqrt 3) 2) :
    hyperbola_eq 3 9 :=
sorry

theorem equation_of_line_AB (B1 B2 A B : point ℝ ℝ) (a b μ : ℝ)
    (hyperbola_eq_a_b : hyperbola_eq a b)
    (eq_vector_B2A_B2B : vector_eq_or (B2 A) = μ (vector_eq_or (B2 B))) :
    line_eq x = y + 3 ∨ line_eq x = - y + 3 :=
sorry

end eccentricity_of_hyperbola_equation_of_hyperbola_passing_through_equation_of_line_AB_l243_243353


namespace imaginary_part_of_complex_number_l243_243170

theorem imaginary_part_of_complex_number : 
  (let i : ℂ := complex.I in 
   let z : ℂ := (4 * i) / (1 + i) in 
   complex.imaginaryPart z = 2) :=
by
  sorry

end imaginary_part_of_complex_number_l243_243170


namespace abs_value_sum_l243_243179

noncomputable def sin_theta_in_bounds (θ : ℝ) : Prop :=
  -1 ≤ Real.sin θ ∧ Real.sin θ ≤ 1

noncomputable def x_satisfies_log_eq (θ x : ℝ) : Prop :=
  Real.log x / Real.log 3 = 1 + Real.sin θ

theorem abs_value_sum (θ x : ℝ) (h1 : x_satisfies_log_eq θ x) (h2 : sin_theta_in_bounds θ) :
  |x - 1| + |x - 9| = 8 :=
sorry

end abs_value_sum_l243_243179


namespace chess_player_win_loss_diff_l243_243235

theorem chess_player_win_loss_diff
    (n m : ℕ)
    (h1 : n + m + (40 - n - m) = 40)
    (h2 : n + 0.5 * (40 - n - m) = 25) :
    n - m = 10 :=
sorry

end chess_player_win_loss_diff_l243_243235


namespace ceil_sqrt_225_l243_243951

theorem ceil_sqrt_225 : ⌈real.sqrt 225⌉ = 15 :=
by
  have h : real.sqrt 225 = 15 := by
    sorry
  rw [h]
  exact int.ceil_eq_self.mpr rfl

end ceil_sqrt_225_l243_243951


namespace fraction_taken_out_is_one_sixth_l243_243522

-- Define the conditions
def original_cards : ℕ := 43
def cards_added_by_Sasha : ℕ := 48
def cards_left_after_Karen_took_out : ℕ := 83

-- Calculate the total number of cards initially after Sasha added hers
def total_cards_after_Sasha : ℕ := original_cards + cards_added_by_Sasha

-- Calculate the number of cards Karen took out
def cards_taken_out_by_Karen : ℕ := total_cards_after_Sasha - cards_left_after_Karen_took_out

-- Define the fraction of the cards Sasha added that Karen took out
def fraction_taken_out : ℚ := cards_taken_out_by_Karen / cards_added_by_Sasha

-- Proof statement: Fraction of the cards Sasha added that Karen took out is 1/6
theorem fraction_taken_out_is_one_sixth : fraction_taken_out = 1 / 6 :=
by
    -- Sorry is a placeholder for the proof, which is not required.
    sorry

end fraction_taken_out_is_one_sixth_l243_243522


namespace minimum_reciprocal_sum_l243_243994

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) - 2

theorem minimum_reciprocal_sum (a m n : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)
  (h₃ : f a (-1) = -1) (h₄ : m + n = 2) (h₅ : 0 < m) (h₆ : 0 < n) :
  (1 / m) + (1 / n) = 2 :=
by
  sorry

end minimum_reciprocal_sum_l243_243994


namespace smallest_four_digit_multiple_of_53_l243_243765

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243765


namespace smallest_four_digit_multiple_of_53_l243_243769

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243769


namespace find_cubic_polynomial_l243_243983

theorem find_cubic_polynomial (q : ℝ → ℝ) 
  (h1 : q 1 = -8) 
  (h2 : q 2 = -12) 
  (h3 : q 3 = -20) 
  (h4 : q 4 = -40) : 
  q = (λ x, - (4 / 3) * x^3 + 6 * x^2 - 4 * x - 2) :=
sorry

end find_cubic_polynomial_l243_243983


namespace sum_b6_b7_b8_equals_64_l243_243862

open Classical

def is_dream_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, 1 / a (n + 1) - 2 / a n = 0

noncomputable def sequence_b (b : ℕ → ℝ) : Prop :=
  is_dream_sequence (λ n, 1 / b n)

theorem sum_b6_b7_b8_equals_64 (b : ℕ → ℝ) (h : sequence_b b) (h_sum : b 1 + b 2 + b 3 = 2) : b 6 + b 7 + b 8 = 64 :=
sorry

end sum_b6_b7_b8_equals_64_l243_243862


namespace square_of_1023_l243_243899

theorem square_of_1023 : 1023^2 = 1045529 := by
  sorry

end square_of_1023_l243_243899


namespace locus_of_P_is_ellipse_l243_243346

theorem locus_of_P_is_ellipse :
  let M := (-2, 0)
  let N := (2, 0)
  ∀ A : ℝ × ℝ, (A.1 + 2)^2 + A.2^2 = 36 →
  let PA := λ P : ℝ × ℝ, real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PN := λ P : ℝ × ℝ, real.sqrt ((P.1 - N.1)^2 + (P.2 - N.2)^2)
  ∀ P : ℝ × ℝ, PA P = PN P →
  ∃ P : ℝ × ℝ, PA P + PA M = 6 → P ∈ { P : ℝ × ℝ | PA P + PA M = 6 } :=
begin
  intro M, intro N, intros A hA PA PN,
  assume P hPA hAM,
  sorry
end

end locus_of_P_is_ellipse_l243_243346


namespace smallest_four_digit_divisible_by_53_l243_243703

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243703


namespace fractional_pizza_eaten_after_six_trips_l243_243214

def pizza_eaten : ℚ := (1/3) * (1 - (2/3)^6) / (1 - 2/3)

theorem fractional_pizza_eaten_after_six_trips : pizza_eaten = 665 / 729 :=
by
  -- proof will go here
  sorry

end fractional_pizza_eaten_after_six_trips_l243_243214


namespace total_cost_of_fencing_l243_243565

-- Given the conditions:
variables {L W x : ℝ} (side_pond : ℝ)

-- The sides of a rectangular field are in the ratio 3:4.
def ratio_3_4 : Prop := (L / W) = (4 / 3)

-- The area of the field is 10800 sq. m.
def area_field : Prop := L * W = 10800

-- The sides of the pond are 1/6th of the shorter side of the field.
def side_pond_condition : Prop := side_pond = (1 / 6) * W

-- The outer fence costs $1.50 per meter.
def cost_outer_per_meter : ℝ := 1.50

-- The pond fence costs $1.00 per meter.
def cost_pond_per_meter : ℝ := 1.00

-- Prove the total cost of fencing is $690
theorem total_cost_of_fencing
  (h1 : ratio_3_4)
  (h2 : area_field)
  (h3 : side_pond_condition) :
  (2 * (L + W) * cost_outer_per_meter + 4 * side_pond * cost_pond_per_meter) = 690 := 
sorry

end total_cost_of_fencing_l243_243565


namespace value_of_k_l243_243156

-- Define the points
def point1 : ℝ × ℝ := (1, 1)
def point2 : ℝ × ℝ := (1, 7)
def point3 : ℝ × ℝ := (9, 1)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the hypotenuse (which is the diameter of the circle)
def hypotenuse : ℝ := distance point1 point3

-- Define the radius of the circle
def radius : ℝ := hypotenuse / 2

-- Define the area of the circle
def circle_area : ℝ := real.pi * radius^2

-- Define the value of k
def k : ℝ := circle_area / real.pi

-- Prove that k = 25
theorem value_of_k : k = 25 :=
by
  -- Proof goes here
  sorry

end value_of_k_l243_243156


namespace smallest_four_digit_divisible_by_53_l243_243745

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243745


namespace new_three_digit_number_l243_243060

theorem new_three_digit_number (t u : ℕ) (h1 : t < 10) (h2 : u < 10) :
  let original := 10 * t + u
  let new_number := (original * 10) + 2
  new_number = 100 * t + 10 * u + 2 :=
by
  sorry

end new_three_digit_number_l243_243060


namespace sum_of_possible_values_of_N_l243_243176

theorem sum_of_possible_values_of_N : 
  ∀ (N : ℝ), (N * (N - 4) = 12) →
  (N = -2 ∨ N = 6) → (N = -2 + 6) := 
by 
  intro N h1 h2 
  cases h2 
  {
    rw h2
  }
  {
    rw h2
  }
  rfl

end sum_of_possible_values_of_N_l243_243176


namespace smallest_four_digit_divisible_by_53_l243_243636

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l243_243636


namespace ceil_sqrt_225_eq_15_l243_243966

theorem ceil_sqrt_225_eq_15 : 
  ⌈Real.sqrt 225⌉ = 15 := 
by sorry

end ceil_sqrt_225_eq_15_l243_243966


namespace calculate_expression_l243_243268

variable (x : ℝ)

theorem calculate_expression : ((3 * x)^2) * (x^2) = 9 * (x^4) := 
sorry

end calculate_expression_l243_243268


namespace pizza_fraction_eaten_l243_243215

-- The total fractional part of the pizza eaten after six trips
theorem pizza_fraction_eaten : 
  ∑ i in (finset.range 6), (1 / 3) ^ (i + 1) = 364 / 729 :=
by
  sorry

end pizza_fraction_eaten_l243_243215


namespace smallest_number_l243_243788

noncomputable def smallest_greater_than_1_1 : ℚ :=
  let nums : List ℚ := [1.4, 9/10, 1.2, 0.5, 13/10]
  let greater_than_1_1 := nums.filter (λ x => x > 1.1)
  List.minimum greater_than_1_1

theorem smallest_number : smallest_greater_than_1_1 = 1.2 := by
  sorry

end smallest_number_l243_243788


namespace smallest_four_digit_divisible_by_53_l243_243642

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l243_243642


namespace ceil_sqrt_225_eq_15_l243_243960

theorem ceil_sqrt_225_eq_15 : ⌈ Real.sqrt 225 ⌉ = 15 :=
by
  sorry

end ceil_sqrt_225_eq_15_l243_243960


namespace find_t_l243_243988

theorem find_t (t : ℝ) (h : (1 / (t + 2) + 2 * t / (t + 2) - 3 / (t + 2) = 3)) : t = -8 := 
by 
  sorry

end find_t_l243_243988


namespace sum_of_all_valid_a_l243_243290

noncomputable def arc_sum_of_a : ℝ :=
  let eq := λ a x, (6 * real.pi * a - real.arcsin (real.sin x) + 
                    2 * real.arccos (real.cos x) - a * x) / 
                   (real.tan x ^ 2 + 4)
  let valid_a := λ a, set.univ.filter (λ x, x ≥ real.pi ∧ eq a x = 0).to_finset.card = 3
  let negative_a := λ a, a < 0
  ((({a : ℝ | negative_a a ∧ valid_a a}).to_finset.sum)) / finset.card {a : ℝ | negative_a a ∧ valid_a a }.to_finset

theorem sum_of_all_valid_a : arc_sum_of_a.round (100) = -1.6 := 
by 
    sorry

end sum_of_all_valid_a_l243_243290


namespace smallest_four_digit_divisible_by_53_l243_243670

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243670


namespace smallest_four_digit_divisible_by_53_l243_243671

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243671


namespace find_R_when_S_eq_5_l243_243484

theorem find_R_when_S_eq_5
  (g : ℚ)
  (h1 : ∀ S, R = g * S^2 - 6)
  (h2 : R = 15 ∧ S = 3) :
  R = 157 / 3 := by
    sorry

end find_R_when_S_eq_5_l243_243484


namespace percent_of_decimal_l243_243594

theorem percent_of_decimal : (3 / 8 / 100) * 240 = 0.9 :=
by
  sorry

end percent_of_decimal_l243_243594


namespace least_possible_length_of_third_side_l243_243387

theorem least_possible_length_of_third_side (a b : ℕ) (h1 : a = 8) (h2 : b = 15) : 
  ∃ c : ℕ, c = 17 ∧ a^2 + b^2 = c^2 := 
by
  use 17 
  split
  · rfl
  · rw [h1, h2]
    norm_num

end least_possible_length_of_third_side_l243_243387


namespace morleys_theorem_l243_243138

theorem morleys_theorem (A B C : Type) [Nonempty A] [Nonempty B] [Nonempty C] 
  (α β γ : A) (trisect α β γ : β * 3 + γ * 3 = 180) : 
  ∃ X Y Z : Type, is_intersection X Y Z → is_equilateral_triangle X Y Z :=
begin
  sorry
end

end morleys_theorem_l243_243138


namespace smallest_four_digit_divisible_by_53_l243_243656

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l243_243656


namespace greatest_two_digit_prime_saturated_l243_243218

-- Prime Saturated Definition
def prime_saturated (a : ℕ) : Prop :=
  ∃ (p : ℕ → Prop) (P : ℕ), (∀ n, p n → nat.prime n) ∧ (P = ∏ n in (finset.filter p (finset.range a)), n) ∧ P < (nat.sqrt a)

-- Statement of the problem
theorem greatest_two_digit_prime_saturated : ∃ (a : ℕ), 10 ≤ a ∧ a ≤ 99 ∧ prime_saturated a ∧ ∀ b, (10 ≤ b ∧ b ≤ 99 ∧ prime_saturated b) → b ≤ a :=
begin
  sorry
end

end greatest_two_digit_prime_saturated_l243_243218


namespace boatcraft_total_boats_built_l243_243264

theorem boatcraft_total_boats_built :
  let february_boats := 5
  let march_boats := february_boats * 3
  let april_boats := march_boats * 3
  let may_boats := april_boats * 3
  let total_boats := february_boats + march_boats + april_boats + may_boats
  total_boats = 200 :=
by
  let february_boats := 5
  let march_boats := february_boats * 3
  let april_boats := march_boats * 3
  let may_boats := april_boats * 3
  let total_boats := february_boats + march_boats + april_boats + may_boats
  have h1 : february_boats = 5 := rfl
  have h2 : march_boats = 15 := by simp [march_boats]
  have h3 : april_boats = 45 := by simp [april_boats]
  have h4 : may_boats = 135 := by simp [may_boats]
  have h5 : total_boats = 200 := by simp [total_boats]
  exact h5

end boatcraft_total_boats_built_l243_243264


namespace height_of_sunflower_in_feet_l243_243498

def height_of_sister_in_feet : ℕ := 4
def height_of_sister_in_inches : ℕ := 3
def additional_height_of_sunflower : ℕ := 21

theorem height_of_sunflower_in_feet 
  (h_sister_feet : height_of_sister_in_feet = 4)
  (h_sister_inches : height_of_sister_in_inches = 3)
  (h_additional : additional_height_of_sunflower = 21) :
  (4 * 12 + 3 + 21) / 12 = 6 :=
by simp [h_sister_feet, h_sister_inches, h_additional]; norm_num; sorry

end height_of_sunflower_in_feet_l243_243498


namespace percent_of_240_l243_243593

theorem percent_of_240 (h : (3 / 8 / 100 : ℝ) = 3 / 800) : 
  (3 / 800 * 240 = 0.9) :=
begin
  sorry
end

end percent_of_240_l243_243593


namespace inscribed_cube_edge_length_l243_243863

theorem inscribed_cube_edge_length (a α : ℝ) (hα : α ≠ 0) :
  ∃ x : ℝ, x = (a * Real.sin(α)) / (2 * Real.sin(Real.pi / 4 + α)) :=
by
  sorry

end inscribed_cube_edge_length_l243_243863


namespace sum_of_negative_solutions_l243_243292

theorem sum_of_negative_solutions :
  (∑ a in {a : ℝ | a < 0 ∧
             ∃ solutions : Finset ℝ, 
             (∀ x ∈ solutions, x ∈ Set.Ici π) ∧ 
             (solutions.card = 3) ∧
             (∀ x ∈ solutions, 
                (6 * Real.pi * a - Real.arcsin (Real.sin x) + 
                2 * Real.arccos (Real.cos x) - a * x) / 
                (Real.tan x ^ 2 + 4) = 0)
          }, a)
  = -1.6 := 
sorry

end sum_of_negative_solutions_l243_243292


namespace smallest_four_digit_divisible_by_53_l243_243723

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l243_243723


namespace number_of_distinct_triangles_in_cube_l243_243367

theorem number_of_distinct_triangles_in_cube : (nat.choose 8 3) = 56 := 
by 
  sorry

end number_of_distinct_triangles_in_cube_l243_243367


namespace inequality_proof_l243_243013

variable {α : Type*} [LinearOrder α] {f : α → α} {a b : α}

theorem inequality_proof (h₁ : ∀ x y, x ≤ y → f(x) ≤ f(y))
(h₂ : a + b ≤ (0 : α)) : f(a) + f(b) ≤ f(-a) + f(-b) :=
by
  sorry

end inequality_proof_l243_243013


namespace ceil_sqrt_225_eq_15_l243_243928

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := 
by 
  sorry

end ceil_sqrt_225_eq_15_l243_243928


namespace determine_scalar_k_l243_243279

variables (a b c : Vector3 ℝ)

noncomputable def k : ℝ := 2

theorem determine_scalar_k
  (h₁ : a + 2 • b + c = 0)
  (h₂ : k • (b × a) + 2 • (b × c) + (c × a) = 0) :
  k = 2 :=
begin
  sorry
end

end determine_scalar_k_l243_243279


namespace smallest_four_digit_multiple_of_53_l243_243764

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243764


namespace hiker_final_distance_l243_243244

theorem hiker_final_distance (east west north south : ℝ) :
  east = 15 → west = 3 → north = 8 → south = 2 →
  (let net_east := east - west in
   let net_north := north - south in
   real.sqrt ((net_east)^2 + (net_north)^2) = 6 * real.sqrt 5) :=
begin
  intros h1 h2 h3 h4,
  let net_east := east - west,
  let net_north := north - south,
  have h_net_east : net_east = 12 := by rw [h1, h2]; norm_num,
  have h_net_north : net_north = 6 := by rw [h3, h4]; norm_num,
  rw [h_net_east, h_net_north],
  norm_num,
  sorry
end

end hiker_final_distance_l243_243244


namespace only_linear_equation_with_two_variables_l243_243209

def is_linear_equation_with_two_variables (eqn : String) : Prop :=
  eqn = "4x-5y=5"

def equation_A := "4x-5y=5"
def equation_B := "xy-y=1"
def equation_C := "4x+5y"
def equation_D := "2/x+5/y=1/7"

theorem only_linear_equation_with_two_variables :
  is_linear_equation_with_two_variables equation_A ∧
  ¬ is_linear_equation_with_two_variables equation_B ∧
  ¬ is_linear_equation_with_two_variables equation_C ∧
  ¬ is_linear_equation_with_two_variables equation_D :=
by
  sorry

end only_linear_equation_with_two_variables_l243_243209


namespace Wendy_polished_more_large_glasses_l243_243201

theorem Wendy_polished_more_large_glasses :
  ∃ (L: ℕ), L = 110 - 50 ∧ (L - 50 = 10) :=
by
  have L := 110 - 50
  use L
  split
  { refl }
  { exact nat.sub_eq_of_eq_add (by simp) }

end Wendy_polished_more_large_glasses_l243_243201


namespace smallest_four_digit_divisible_by_53_l243_243728

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l243_243728


namespace cd_value_l243_243221

theorem cd_value (a b c d : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (ab ac bd : ℝ) 
  (h_ab : ab = 2) (h_ac : ac = 5) (h_bd : bd = 6) :
  ∃ (cd : ℝ), cd = 3 :=
by sorry

end cd_value_l243_243221


namespace range_of_m_l243_243038

theorem range_of_m (a : ℝ) (m : ℝ) (f : ℝ → ℝ) 
  (domain_cond : ∀ x : ℝ, 0 < a ∧ a ≠ 1 ∧ a^(x^2 - a * x - 2 * a^2) > 1 → (-a < x ∧ x < 2a)) 
  (fx_def : ∀ x, f x = sqrt ((1 / a)^(x^2 + 2 * m * x - m) - 1)) 
  : -1 ≤ m ∧ m ≤ 0 :=
sorry

end range_of_m_l243_243038


namespace shelter_cats_incoming_l243_243874

theorem shelter_cats_incoming (x : ℕ) (h : x + x / 2 - 3 + 5 - 1 = 19) : x = 12 :=
by
  sorry

end shelter_cats_incoming_l243_243874


namespace sum_num_den_252_l243_243807

theorem sum_num_den_252 (h : (252 : ℤ) / 100 = (63 : ℤ) / 25) : 63 + 25 = 88 :=
by
  sorry

end sum_num_den_252_l243_243807


namespace sufficient_condition_not_necessary_condition_l243_243332

variable {a b : ℝ}

theorem sufficient_condition (h : 0 < a ∧ a < b) : (1 / a > 1 / b) :=
begin
  sorry
end

theorem not_necessary_condition (h : 1 / a > 1 / b) : ¬(0 < a ∧ a < b) :=
begin
  sorry
end

end sufficient_condition_not_necessary_condition_l243_243332


namespace least_third_side_of_right_triangle_l243_243406

theorem least_third_side_of_right_triangle {a b c : ℝ} 
  (h1 : a = 8) 
  (h2 : b = 15) 
  (h3 : c = Real.sqrt (8^2 + 15^2) ∨ c = Real.sqrt (15^2 - 8^2)) : 
  c = Real.sqrt 161 :=
by {
  intros h1 h2 h3,
  rw [h1, h2] at h3,
  cases h3,
  { exfalso, preciesly contradiction occurs because sqrt (8^2 + 15^2) is not sqrt161, 
   rw [← h3],
   norm_num,},
  { exact h3},
  
}

end least_third_side_of_right_triangle_l243_243406


namespace area_quadrilateral_geq_three_times_area_triangle_l243_243508

variable (A B C D M : Type)
variable [ConvexQuadrilateral A B C D]
variable [OnSideAD M A B C D]
variable [Parallel CM AB]
variable [Parallel BM CD]

theorem area_quadrilateral_geq_three_times_area_triangle
    (S_ABCD S_BCM : ℝ) [AreaOfQuadrilateral S_ABCD A B C D] [AreaOfTriangle S_BCM B C M] :
    S_ABCD ≥ 3 * S_BCM :=
by
  sorry

end area_quadrilateral_geq_three_times_area_triangle_l243_243508


namespace weight_of_b_l243_243545

variable (A B C : ℕ)

theorem weight_of_b 
  (h1 : A + B + C = 180) 
  (h2 : A + B = 140) 
  (h3 : B + C = 100) :
  B = 60 :=
sorry

end weight_of_b_l243_243545


namespace smallest_four_digit_divisible_by_53_l243_243660

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l243_243660


namespace exists_four_elements_with_integer_geometric_mean_l243_243003

-- Definitions of primes less than or equal to 26
def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23]

-- Predicate to check if an integer has no prime divisors larger than 26
def has_only_small_prime_divisors (n : ℕ) : Prop :=
  ∀ p, p ∣ n → p ∈ primes

-- The set M of 1985 positive integers
variables (M : Finset ℕ)

-- Hypotheses
hypotheses
  (hM_card : M.card = 1985)
  (hM_divisors : ∀ m ∈ M, has_only_small_prime_divisors m)

-- The theorem statement
theorem exists_four_elements_with_integer_geometric_mean :
  ∃ a b c d ∈ M, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∃ k : ℕ, k ^ 4 = a * b * c * d) :=
sorry

end exists_four_elements_with_integer_geometric_mean_l243_243003


namespace radius_of_circumcircle_l243_243587

variables (r1 r2 : ℝ) (A B : euclidean_geometry.Point ℝ) (C : euclidean_geometry.Point ℝ) (r : ℝ)

-- Conditions of the problem
def conditions : Prop :=
  (euclidean_geometry.dist A B = 6 * real.sqrt 10) ∧
  r1 + r2 = 11 ∧
  euclidean_geometry.dist (euclidean_geometry.Point.mk 0 0 r1) 
                          (euclidean_geometry.Point.mk (6 * real.sqrt 10) 0 r2) = real.sqrt 481 ∧
  euclidean_geometry.dist C A = r + r1 ∧
  euclidean_geometry.dist C B = r + r2

-- Question (proof goal)
theorem radius_of_circumcircle (h : conditions r1 r2 A B C 9) : 
  let R := 3 * real.sqrt 10 in R = (euclidean_geometry.circumradius_of_triangle A B C) :=
by { sorry }

end radius_of_circumcircle_l243_243587


namespace function_monotonicity_l243_243165

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem function_monotonicity :
  (∀ x ∈ set.Ioo 0 Real.exp 1, 0 < (1 - Real.log x) / x^2) ∧
  (∀ x ∈ set.Ioo Real.exp 1 10, (1 - Real.log x) / x^2 < 0) :=
by
  sorry

end function_monotonicity_l243_243165


namespace num_ordered_pairs_l243_243917

theorem num_ordered_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0) :
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ (6 / m + 3 / n = 1)) → 
  (set.filter (λ p : ℕ × ℕ, let m := p.fst; let n := p.snd in 6 / m + 3 / n = 1 ∧ m > 0 ∧ n > 0) 
  (set.univ : set (ℕ × ℕ))).finite.card = 6 := 
sorry

end num_ordered_pairs_l243_243917


namespace total_squares_in_6_by_6_grid_l243_243079

theorem total_squares_in_6_by_6_grid : 
  let count_1x1 := 6 * 6
  let count_2x2 := 5 * 5
  let count_3x3 := 4 * 4
  let count_4x4 := 3 * 3
  count_1x1 + count_2x2 + count_3x3 + count_4x4 = 86 :=
by 
  let count_1x1 := 6 * 6
  let count_2x2 := 5 * 5
  let count_3x3 := 4 * 4
  let count_4x4 := 3 * 3
  have sum := count_1x1 + count_2x2 + count_3x3 + count_4x4
  exact (by sorry : sum = 86)

end total_squares_in_6_by_6_grid_l243_243079


namespace probability_of_banana_l243_243422

theorem probability_of_banana :
  let meats := ["beef", "chicken"]
  let fruits := ["apple", "pear", "banana"]
  let outcomes := [(m, f) | m ∈ meats, f ∈ fruits]
  let favorable_outcomes := [(m, f) | (m, f) ∈ outcomes, f = "banana"]
  (favorable_outcomes.length : ℚ) / outcomes.length = 1 / 3 :=
by
  let meats := ["beef", "chicken"]
  let fruits := ["apple", "pear", "banana"]
  let outcomes := [(m, f) | m ∈ meats, f ∈ fruits]
  let favorable_outcomes := [(m, f) | (m, f) ∈ outcomes, f = "banana"]
  have total_outcomes : outcomes.length = 6 := by sorry
  have favorable_count : favorable_outcomes.length = 2 := by sorry
  have probability : (2 : ℚ) / 6 = 1 / 3 := by sorry
  exact probability

end probability_of_banana_l243_243422


namespace num_even_two_digit_numbers_l243_243208

theorem num_even_two_digit_numbers :
  let digits := {0, 1, 2, 3, 4};
  {n : ℕ | ∃ a b, a ∈ digits ∧ b ∈ digits ∧ a ≠ 0 ∧ 2 * b = b ∧ n = 10 * a + b}.card = 10 :=
by 
  let digits := {0, 1, 2, 3, 4};
  have h1: {n : ℕ | ∃ a b, a ∈ digits ∧ b ∈ digits ∧ a ≠ 0 ∧ 2 * b = b ∧ n = 10 * a + b}.card = 10;
  sorry

end num_even_two_digit_numbers_l243_243208


namespace trajectory_of_A_is_ellipse_range_of_m_l243_243421

noncomputable def point := (ℝ × ℝ)

structure Triangle :=
  (A B C : point)
  (side_a side_b side_c : ℝ)
  (side_condition : side_b + side_c = 2 * side_a)
  (vertex_B : B = (-1, 0))
  (vertex_C : C = (1, 0))

def ellipse_equation (x y : ℝ) :=
  (x^2 / 4 + y^2 / 3 = 1)

def point_on_line (P : point) (k m : ℝ) :=
  P.2 = k * P.1 + m

def symmetric_about (M N : point) (l : ℝ × ℝ → Point) :=
  l ⟨M.1 + N.1 / 2, M.2 + N.2 / 2⟩ = ⟨0, -1/2⟩

theorem trajectory_of_A_is_ellipse (A B C : point) (side_a side_b side_c : ℝ)
  (cond : b + c = 2 * a) (B_pos : B = (-1, 0)) (C_pos : C = (1, 0)) :
  ∃ A : point, ellipse_equation A.1 A.2 :=
sorry

theorem range_of_m (A B C : point) (b c k m : ℝ)
  (cond : b + c = 2 * a) (intersect : ∃ M N : point, point_on_line M k m ∧ point_on_line N k m ∧ symmetric_about M N (λ(x : ℝ × ℝ) ⟨x.1, x.2⟩))
  (mid_point : symmetric_about M N (λ(x : ℝ × ℝ) ⟨x.1, x.2⟩)) :
  (k ≠ 0 → (3 / 2 < m ∧ m < 2)) ∧ (k = 0 → (m ∈ [-√3, 0) ∪ (0, √3])) :=
sorry

end trajectory_of_A_is_ellipse_range_of_m_l243_243421


namespace smallest_four_digit_divisible_by_53_l243_243638

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l243_243638


namespace no_adjacent_green_hats_l243_243830

theorem no_adjacent_green_hats (n m : ℕ) (h₀ : n = 9) (h₁ : m = 3) : 
  (((1 : ℚ) - (9/14 : ℚ)) = (5/14 : ℚ)) :=
by
  rw h₀ at *,
  rw h₁ at *,
  sorry

end no_adjacent_green_hats_l243_243830


namespace correct_conclusions_l243_243173

def point_on_parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def parabola_open_upwards (a : ℝ) : Prop := a > 0

def vertex_x_coordinate_between (a b m : ℝ) : Prop :=
  let vertex_x := -b / (2 * a) in 
  (vertex_x < 1) ∧ (vertex_x > m)

def parabola_passes_through (a b c x0 y0 : ℝ) : Prop :=
  point_on_parabola a b c x0 = y0

def no_real_roots_condition (a b c m : ℝ) : Prop :=
  ∀ x : ℝ, a * (x - m) * (x - 1) + 1 ≠ 0

theorem correct_conclusions (a b c m t y1 y2 : ℝ)
  (ha : parabola_open_upwards a)
  (hm : -2 < m ∧ m < -1)
  (habc : parabola_passes_through a b c 1 0 ∧ parabola_passes_through a b c m 0)
  (habc_eq : a + b + c = 0)
  (hy1 : point_on_parabola a b c (t - 1) = y1)
  (hy2 : point_on_parabola a b c (t + 1) = y2)
  (ht : t > 0)
  (hnr : no_real_roots_condition a b c m)
:(a > 0 ∧ b > 0 ∧ c < 0 → a * b * c < 0) ∧
 (a + b + c = 0 → 2 * a + c > 0) ∧
 (t > 0 → y1 < y2) ∧
 (no_real_roots_condition a b c m → b^2 - 4*a*c < 4*a) :=
sorry

end correct_conclusions_l243_243173


namespace average_and_variance_for_player_B_choose_player_B_l243_243232

theorem average_and_variance_for_player_B :
  let goals_B := [(7 : ℕ), 9, 7, 8, 9]
  let average_B := (7 + 9 + 7 + 8 + 9) / 5
  let variance_B := (1 / 5) * ((7 - average_B) ^ 2 + (9 - average_B) ^ 2 + (7 - average_B) ^ 2 + (8 - average_B) ^ 2 + (9 - average_B) ^ 2)
  average_B = 8 ∧ variance_B = 0.8 := by
  let goals_B := [(7 : ℕ), 9, 7, 8, 9]
  let average_B := (7 + 9 + 7 + 8 + 9) / 5
  let variance_B := (1 / 5) * ((7 - average_B) ^ 2 + (9 - average_B) ^ 2 + (7 - average_B) ^ 2 + (8 - average_B) ^ 2 + (9 - average_B) ^ 2)
  have h1 : average_B = 8 := by
    sorry
  have h2 : variance_B = 0.8 := by
    sorry
  exact ⟨h1, h2⟩

theorem choose_player_B :
  let variance_A := 3.2
  let variance_B := 0.8
  variance_B < variance_A → "Choose Player B" = "Choose Player B" := by
  let variance_A := 3.2
  let variance_B := 0.8
  intro h
  exact rfl

end average_and_variance_for_player_B_choose_player_B_l243_243232


namespace video_game_price_l243_243913

theorem video_game_price (total_games not_working_games : ℕ) (total_earnings : ℕ)
  (h1 : total_games = 10) (h2 : not_working_games = 2) (h3 : total_earnings = 32) :
  ((total_games - not_working_games) > 0) →
  (total_earnings / (total_games - not_working_games)) = 4 :=
by
  sorry

end video_game_price_l243_243913


namespace smallest_four_digit_divisible_by_53_l243_243652

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l243_243652


namespace tangent_to_x_axis_at_origin_range_of_a_l243_243033

variables (f : ℝ → ℝ) (a : ℝ)

-- Definition of the function f
def f (x : ℝ) : ℝ := exp x - a * x^2 - cos x - log (x + 1)

-- Part 1: Showing that f is tangent to x-axis at origin when a = 1
theorem tangent_to_x_axis_at_origin (h : a = 1) : f 0 = 0 ∧ deriv f 0 = 0 :=
by
  intros
  sorry

-- Part 2: Finding the range of a
theorem range_of_a (h : ∀ I ∈ {[ (-1, 0); (0, +∞) ]}, ∃! x ∈ I, f' x = 0) : (3 / 2 : ℝ) < a :=
by
  intros
  sorry

end tangent_to_x_axis_at_origin_range_of_a_l243_243033


namespace max_kings_on_12x12_board_l243_243604

-- Define a chessboard as a set of squares.
def Square := (ℕ × ℕ)
def is_adjacent (s1 s2 : Square) : Prop := abs (s1.1 - s2.1) ≤ 1 ∧ abs (s1.2 - s2.2) ≤ 1 ∧ (s1 ≠ s2)

-- Define a function that counts the maximum kings with the given condition.
noncomputable def max_kings (n : ℕ) : ℕ :=
  if h : n ≥ 12 then
    2 * (169 / 6).floor
  else
    0

-- Lean theorem to assert that the maximum number of kings
-- on a 12 × 12 board where each king attacks exactly one other king is 56.
theorem max_kings_on_12x12_board : max_kings 12 = 56 := 
by {
  rw max_kings,
  split_ifs,
  sorry
}

end max_kings_on_12x12_board_l243_243604


namespace first_expression_correct_second_expression_correct_l243_243267

noncomputable def calculate_first_expression : Prop :=
  ( (9/4)^(1/2) - (-9.6)^0 - (27/8)^(-2/3) + (3/2)^(-2) = 1/2 )

noncomputable def calculate_second_expression : Prop :=
  ( logBase 3 (sqrt 3) + log 25 / log 10 + log 4 / log 10 + 7^(log 2 / log 7) = 9/2 )

theorem first_expression_correct : calculate_first_expression := 
  by sorry

theorem second_expression_correct : calculate_second_expression := 
  by sorry

end first_expression_correct_second_expression_correct_l243_243267


namespace ceil_sqrt_225_eq_15_l243_243958

theorem ceil_sqrt_225_eq_15 : ⌈ Real.sqrt 225 ⌉ = 15 :=
by
  sorry

end ceil_sqrt_225_eq_15_l243_243958


namespace sample_size_l243_243094

theorem sample_size (total_employees : ℕ) (male_employees : ℕ) (sampled_males : ℕ) (sample_size : ℕ) 
  (h1 : total_employees = 120) (h2 : male_employees = 80) (h3 : sampled_males = 24) : 
  sample_size = 36 :=
by
  sorry

end sample_size_l243_243094


namespace car_speed_first_hour_l243_243567

theorem car_speed_first_hour (x : ℝ) (h1 : (x + 75) / 2 = 82.5) : x = 90 :=
sorry

end car_speed_first_hour_l243_243567


namespace square_of_1023_l243_243900

theorem square_of_1023 : 1023^2 = 1045529 := by
  sorry

end square_of_1023_l243_243900


namespace relation_among_abc_l243_243314

noncomputable def a := 2^12
noncomputable def b := (1/2)^(-0.8)
noncomputable def c := 2 * (Real.log 2 / Real.log 5)

theorem relation_among_abc : a > b ∧ b > c :=
by
  sorry

end relation_among_abc_l243_243314


namespace smallest_four_digit_divisible_by_53_l243_243610

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l243_243610


namespace least_third_side_length_l243_243413

theorem least_third_side_length (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℝ, (c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2)) ∧ c = Real.sqrt 161 :=
by
  sorry

end least_third_side_length_l243_243413


namespace determine_a_l243_243356

theorem determine_a (a : ℝ) (p : set ℝ) (q : set ℝ) :
  (p = {x : ℝ | a - 1 < x ∧ x < a + 1}) →
  (q = {x : ℝ | x^2 - 4 * x + 3 ≥ 0}) →
  (¬ (λ x, q x) = {x : ℝ | 1 < x ∧ x < 3}) →
  (1 ≤ a - 1) ∧ (a + 1 ≤ 3) →
  a = 2 :=
by
  intros
  sorry

end determine_a_l243_243356


namespace find_x_l243_243561

-- Given data values
def data_values (x : ℝ) : List ℝ := [x, 120, 35, 70, 150, x, 55, 80, 170]

-- Mean calculation
def mean (l : List ℝ) : ℝ := (l.sum) / l.length

-- Median calculation (for sorting and picking the middle one)
noncomputable def median (l : List ℝ) : ℝ :=
  let sorted := l.sort
  sorted.sorted_nth (sorted.length / 2).toNat

-- Mode calculation (assuming the simplest mode definition for two occurrences)
noncomputable def mode (l : List ℝ) : ℝ :=
  l.mode

theorem find_x :
  let x := 111
  data_values x.mean = x ∧
  data_values x.median = x ∧
  data_values x.mode = x
:= by
  sorry

end find_x_l243_243561


namespace probability_one_male_one_female_same_problem_l243_243192

def problem_solving_probability : ℚ := 1 / 2

theorem probability_one_male_one_female_same_problem
  (teachers : list string)
  (problems : list string)
  (chosen_problems : list (list string))
  (female_teacher : string)
  (male_teachers : list string)
  (exactly_one_male_one_female_same_problem : list (list string) → bool) :
  8 = teachers.length * problems.length ∧ 
  list.length (filter exactly_one_male_one_female_same_problem chosen_problems) = 4 →
  (list.length (filter exactly_one_male_one_female_same_problem chosen_problems)) / (list.length chosen_problems: ℚ) = problem_solving_probability :=
by
  sorry

end probability_one_male_one_female_same_problem_l243_243192


namespace probability_no_two_green_hats_next_to_each_other_l243_243824

open Nat

def choose (n k : ℕ) : ℕ := Nat.fact n / (Nat.fact k * Nat.fact (n - k))

def total_ways_to_choose (n k : ℕ) : ℕ :=
  choose n k

def event_A (n : ℕ) : ℕ := n - 2

def event_B (n k : ℕ) : ℕ := choose (n - k + 1) 2 * (k - 2)

def probability_no_two_next_to_each_other (n k : ℕ) : ℚ :=
  let total_ways := total_ways_to_choose n k
  let event_A_ways := event_A (n)
  let event_B_ways := event_B n 3
  let favorable_ways := total_ways - (event_A_ways + event_B_ways)
  favorable_ways / total_ways

-- Given the conditions of 9 children and choosing 3 to wear green hats
theorem probability_no_two_green_hats_next_to_each_other (p : probability_no_two_next_to_each_other 9 3 = 5 / 14) : Prop := by
  sorry

end probability_no_two_green_hats_next_to_each_other_l243_243824


namespace problem1_sol_problem2_sol_l243_243892

noncomputable def problem1 : Prop :=
  (- (27 / 8) ^ (-2 / 3) + (0.002) ^ (-1 / 2) - 10 * (Real.sqrt 5 - 2) ^ (-1) + (Real.sqrt 2 - Real.sqrt 3) ^ 0) = -167 / 9

noncomputable def problem2 : Prop :=
  ((1 / 2) * Real.logb 10 (32 / 49) - (4 / 3) * Real.logb 10 (Real.sqrt 8) + Real.logb 10 (Real.sqrt 245)) = 1 / 2

theorem problem1_sol : problem1 := 
  by 
  sorry

theorem problem2_sol : problem2 := 
  by 
  sorry

end problem1_sol_problem2_sol_l243_243892


namespace amplitude_of_sinusoidal_l243_243885

theorem amplitude_of_sinusoidal (a b c d : ℝ) (h : ∀ x : ℝ, y = a * sin (b * x + c) + d) 
  (h_max : ∀ x : ℝ, y ≤ 4) (h_min : ∀ x : ℝ, y ≥ -2) : a = 3 := 
by
  sorry

end amplitude_of_sinusoidal_l243_243885


namespace sum_of_x_y_l243_243102

-- Definitions for floor and fractional part
def floor (x : ℝ) : ℤ := x.to_floor
def frac (x : ℝ) : ℝ := x - (floor x)

-- Assumptions from the problem
variables {x y : ℝ}
axiom h1 : (floor x : ℝ) + frac y = 3.2
axiom h2 : frac x + (floor y : ℝ) = 4.7

-- The theorem to be proven
theorem sum_of_x_y : x + y = 7.9 :=
by sorry

end sum_of_x_y_l243_243102


namespace employee_payment_l243_243576

theorem employee_payment
  (A B C : ℝ)
  (h_total : A + B + C = 1500)
  (h_A : A = 1.5 * B)
  (h_C : C = 0.8 * B) :
  A = 682 ∧ B = 454 ∧ C = 364 := by
  sorry

end employee_payment_l243_243576


namespace smallest_four_digit_divisible_by_53_l243_243748

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243748


namespace abs_diff_of_two_nums_l243_243542

theorem abs_diff_of_two_nums (x y : ℕ) (h1 : (x + y) / 2 = 361) (h2 : Real.sqrt (x * y) = 163) : abs (x - y) = 154 :=
by
  sorry

end abs_diff_of_two_nums_l243_243542


namespace sum_zero_abs_sum_one_l243_243832

variable (n : ℕ) (a : Fin n → ℝ)

theorem sum_zero_abs_sum_one (h1 : ∑ i, a i = 0) (h2 : ∑ i, |a i| = 1) :
  |∑ i, (i : ℕ) * a i| ≤ (n - 1) / 2 :=
sorry

end sum_zero_abs_sum_one_l243_243832


namespace magnitude_implies_collinear_l243_243357

-- Define vector space and required operations
variables {V : Type*} [inner_product_space ℝ V] (a b : V)

-- Given conditions
def vectors_non_zero (a b : V) : Prop :=
a ≠ 0 ∧ b ≠ 0

def magnitude_condition (a b : V) : Prop :=
∥a + b∥ = ∥a∥ - ∥b∥

-- The statement to prove
theorem magnitude_implies_collinear (a b : V) (h1: vectors_non_zero a b) (h2: magnitude_condition a b) :
  ∃ (λ : ℝ), a = λ • b := 
by sorry

end magnitude_implies_collinear_l243_243357


namespace right_triangle_least_side_l243_243399

theorem right_triangle_least_side (a b c : ℝ) (h_rt : a^2 + b^2 = c^2) (h1 : a = 8) (h2 : b = 15) : min a b = 8 := 
by
sorry

end right_triangle_least_side_l243_243399


namespace smallest_four_digit_divisible_by_53_l243_243687

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l243_243687


namespace angle_ABC_eq_60_l243_243449

-- Definitions based on conditions
universe u
variables (A B C L D S : Type u)
variables [irrel1 : is_acute_triangle A B C] [irrel2 : is_angle_bisector B L]
variables [irrel3 : intersects_circumcircle A B L D] [irrel4 : symmetric_to C D L S]
variables [irrel5 : lies_on_line S A B] [not_endpoints : not (S = A ∨ S = B)]

-- Statement of the theorem
theorem angle_ABC_eq_60 : 
  acute_triangle A B C ∧
  angle_bisector B L ∧
  circumcircle_triangle_A_B_L_intersects_BC D ∧
  symmetric_point_respect_to_line C D L S ∧
  lies_on_line_segment S A B ∧
  not (S = A ∨ S = B) →
  angle_of_triangle A B C = 60 :=
by sorry

end angle_ABC_eq_60_l243_243449


namespace train_prob_correct_l243_243570

def probability_train_there (train_start train_end train_wait alex_start alex_end : ℝ) : ℝ :=
  let train_duration := train_end - train_start
  let alex_duration := alex_end - alex_start
  let total_area := train_duration * alex_duration
  let triangle_area := 0.5 * 15 * 15
  let rectangle_area := 45 * 15
  (triangle_area + rectangle_area) / total_area

theorem train_prob_correct :
  let train_start := 0  -- 1:00 am in minutes past reference point
  let train_end := 60 -- 2:00 am in minutes past reference point
  let train_wait := 15
  let alex_start := 0  -- 1:00 am in minutes past reference point
  let alex_end := 75 -- 2:15 am in minutes past reference point
  probability_train_there train_start train_end train_wait alex_start alex_end = 7 / 40 :=
by 
  sorry

end train_prob_correct_l243_243570


namespace basketball_price_l243_243146

variable (P : ℝ)

def coachA_cost : ℝ := 10 * P
def coachB_baseball_cost : ℝ := 14 * 2.5
def coachB_bat_cost : ℝ := 18
def coachB_total_cost : ℝ := coachB_baseball_cost + coachB_bat_cost
def coachA_excess_cost : ℝ := 237

theorem basketball_price (h : coachA_cost P = coachB_total_cost + coachA_excess_cost) : P = 29 :=
by
  sorry

end basketball_price_l243_243146


namespace polynomial_divisibility_l243_243067
noncomputable theory

def polynomial_divisible (f : ℕ → ℤ) (g : ℕ → ℤ) : Prop :=
  ∃ h : ℕ → ℤ, ∀ x : ℂ, f x = g x * h x

theorem polynomial_divisibility (k : ℤ) :
  polynomial_divisible (λ x, 3 * x^3 - 9 * x^2 + k * x - 12) (λ x, x - 3) →
  k = 4 ∧ polynomial_divisible (λ x, 3 * x^3 - 9 * x^2 + 4 * x - 12) (λ x, 3 * x^2 + 4) :=
by sorry

end polynomial_divisibility_l243_243067


namespace larger_integer_of_two_with_difference_8_and_product_168_l243_243582

theorem larger_integer_of_two_with_difference_8_and_product_168 :
  ∃ (x y : ℕ), x > y ∧ x - y = 8 ∧ x * y = 168 ∧ x = 14 :=
by
  sorry

end larger_integer_of_two_with_difference_8_and_product_168_l243_243582


namespace chimes_in_a_day_l243_243046

-- Definitions for the conditions
def strikes_in_12_hours : ℕ :=
  (1 + 12) * 12 / 2

def strikes_in_24_hours : ℕ :=
  2 * strikes_in_12_hours

def half_hour_strikes : ℕ :=
  24 * 2

def total_chimes_in_a_day : ℕ :=
  strikes_in_24_hours + half_hour_strikes

-- Statement to prove
theorem chimes_in_a_day : total_chimes_in_a_day = 204 :=
by 
  -- The proof would be placed here
  sorry

end chimes_in_a_day_l243_243046


namespace solve_for_y_l243_243919

theorem solve_for_y (y : ℚ) (h : |(4 : ℚ) * y - 6| = 0) : y = 3 / 2 :=
sorry

end solve_for_y_l243_243919


namespace simplify_expression_l243_243506

theorem simplify_expression
  (x y : ℝ)
  (h : (x + 2)^3 ≠ (y - 2)^3) :
  ( (x + 2)^3 + (y + x)^3 ) / ( (x + 2)^3 - (y - 2)^3 ) = (2 * x + y + 2) / (x - y + 4) :=
sorry

end simplify_expression_l243_243506


namespace chef_initial_items_l243_243842

def initial_apples := 21
def initial_flour := 18
def initial_sugar := 27
def initial_butter := 5
def total_items := initial_apples + initial_flour + initial_sugar + initial_butter

theorem chef_initial_items:
  (initial_apples = 21) ∧
  (initial_flour = 18) ∧
  (initial_sugar = 27) ∧
  (initial_butter = 5) ∧
  (total_items = 71) :=
  by
    split;
    try { sorry }

end chef_initial_items_l243_243842


namespace smallest_four_digit_div_by_53_l243_243620

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l243_243620


namespace area_triangle_PMN_l243_243465

theorem area_triangle_PMN (P Q R M N : Type)
  [linear_ordered_field P] [linear_ordered_field Q]
  [linear_ordered_field R] [linear_ordered_field M]
  [linear_ordered_field N] 
  (midpoint_M : M = (P + Q) / 2)
  (ratio_N : 2 * PN = PR)
  (area_PQR : 36): 
  area_PMN = 6 := 
sorry

end area_triangle_PMN_l243_243465


namespace smallest_four_digit_div_by_53_l243_243625

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l243_243625


namespace triangle_AUV_isosceles_l243_243259

variables {A B C X Y D U V : Type}  -- Define the points
variables [Geometry A B C X Y D U V] -- Declare that these points are in a geometric space

-- Hypotheses / Conditions
variables (h_cond1: X ∈ line B C ∧ Y ∈ line B C ∧ X ≠ Y ∧ X ≠ B ∧ Y ≠ C)
variables (h_cond2: BX * AC = CY * AB)

-- Circumcenters of triangles ACX and ABY
variables (O1: circumcenter A C X)
variables (O2: circumcenter A B Y)

-- Line O1O2 intersecting AB and AC at U and V respectively
variables (h_intersect: line O1 O2 ∩ line A B = {U} ∧ line O1 O2 ∩ line A C = {V})

-- The statement that the triangle AUV is isosceles
theorem triangle_AUV_isosceles: is_isosceles A U V :=
begin
  sorry -- the proof itself is not required as per the prompt
end

end triangle_AUV_isosceles_l243_243259


namespace reaches_school_early_l243_243589

theorem reaches_school_early (R : ℝ) (T : ℝ) (F : ℝ) (T' : ℝ)
    (h₁ : F = (6/5) * R)
    (h₂ : T = 24)
    (h₃ : R * T = F * T')
    : T - T' = 4 := by
  -- All the given conditions are set; fill in the below placeholder with the proof.
  sorry

end reaches_school_early_l243_243589


namespace winning_votes_correct_l243_243574

-- Define the total number of votes, V
def V : ℕ := 75000

-- Percentage of total votes received by the winner
def percentage_winner : ℝ := 0.46

-- Calculate the number of votes for the winner
def votes_winner : ℕ := (percentage_winner * V).to_nat

-- Expected number of votes for the winner
def expected_votes_winner : ℕ := 34500

-- Theorem statement that asserts the winning candidate received the correct number of votes
theorem winning_votes_correct : votes_winner = expected_votes_winner := 
by sorry

end winning_votes_correct_l243_243574


namespace David_Marks_in_Mathematics_are_85_l243_243914

theorem David_Marks_in_Mathematics_are_85
  (english_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (h1 : english_marks = 86)
  (h2 : physics_marks = 92)
  (h3 : chemistry_marks = 87)
  (h4 : biology_marks = 95)
  (h5 : average_marks = 89)
  (h6 : num_subjects = 5) : 
  (86 + 92 + 87 + 95 + 85) / 5 = 89 :=
by sorry

end David_Marks_in_Mathematics_are_85_l243_243914


namespace triangle_angle_sum_depends_on_parallel_postulate_l243_243148

-- Definitions of conditions
def triangle_angle_sum_theorem (A B C : ℝ) : Prop :=
  A + B + C = 180

def parallel_postulate : Prop :=
  ∀ (l : ℝ) (P : ℝ), ∃! (m : ℝ), m ≠ l ∧ ∀ (Q : ℝ), Q ≠ P → (Q = l ∧ Q = m)

-- Theorem statement: proving the dependence of the triangle_angle_sum_theorem on the parallel_postulate
theorem triangle_angle_sum_depends_on_parallel_postulate: 
  ∀ (A B C : ℝ), parallel_postulate → triangle_angle_sum_theorem A B C :=
sorry

end triangle_angle_sum_depends_on_parallel_postulate_l243_243148


namespace smallest_lcm_of_gcd_eq_5_is_201000_l243_243054

open Nat

theorem smallest_lcm_of_gcd_eq_5_is_201000 :
  ∃ (p q : ℕ), (1000 ≤ p ∧ p ≤ 9999) ∧ (1000 ≤ q ∧ q ≤ 9999) ∧ gcd p q = 5 ∧ lcm p q = 201_000 := 
by
  sorry

end smallest_lcm_of_gcd_eq_5_is_201000_l243_243054


namespace ceil_sqrt_225_eq_15_l243_243971

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := by
  sorry

end ceil_sqrt_225_eq_15_l243_243971


namespace total_fish_fillets_l243_243995

theorem total_fish_fillets (t1 t2 t3 : ℕ) (h1 : t1 = 189) (h2 : t2 = 131) (h3 : t3 = 180) :
  t1 + t2 + t3 = 500 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end total_fish_fillets_l243_243995


namespace value_of_f_neg_4_over_3_l243_243062

def f : ℝ → ℝ
| x => if x > 0 then -Real.cos (Real.pi * x) else f (x + 1) + 1

theorem value_of_f_neg_4_over_3 : f (-4 / 3) = 5 / 2 :=
by
  sorry

end value_of_f_neg_4_over_3_l243_243062


namespace infinite_symmetry_lines_l243_243238

-- Definition of a circle
structure Circle (radius : ℝ) :=
  (center : ℝ × ℝ)

-- Definition of a line of symmetry
def is_line_of_symmetry (C : Circle r) (line : ℝ → Prop) : Prop :=
  ∀ (p : ℝ × ℝ), line p → exists q, p = q ∧ (C.center, q) = (q, C.center) 

-- Proposition that any line passing through the center of the circle is a line of symmetry
def line_through_center_if_symmetry (C : Circle r) (line : ℝ → Prop) : Prop :=
  line (C.center) → is_line_of_symmetry C line

-- The main theorem stating that a circle has infinitely many lines of symmetry
theorem infinite_symmetry_lines (C : Circle r) : ∃ (lines : ℕ → (ℝ → Prop)), 
  ∀ n, is_line_of_symmetry C (lines n) :=
sorry

end infinite_symmetry_lines_l243_243238


namespace square_of_1023_l243_243897

def square_1023_eq_1046529 : Prop :=
  let x := 1023
  x * x = 1046529

theorem square_of_1023 : square_1023_eq_1046529 :=
by
  sorry

end square_of_1023_l243_243897


namespace systematic_sampling_probability_l243_243432

/-- Given a population of 1002 individuals, if we remove 2 randomly and then pick 50 out of the remaining 1000, then the probability of picking each individual is 50/1002. 
This is because the process involves two independent steps: not being removed initially and then being chosen in the sample of size 50. --/
theorem systematic_sampling_probability :
  let population_size := 1002
  let removal_count := 2
  let sample_size := 50
  ∀ p : ℕ, p = 50 / (1002 : ℚ) := sorry

end systematic_sampling_probability_l243_243432


namespace distinct_solutions_of_an_eq_s_l243_243180

noncomputable theory

def real_sequence (s : ℝ) : ℕ → ℝ
| 0       := s
| (n + 1) := (real_sequence n) ^ 2 - 2

theorem distinct_solutions_of_an_eq_s (s : ℝ) :
  (∃ n : ℕ, real_sequence s n = s) → ∀ m n : ℕ, m ≠ n → real_sequence s m ≠ real_sequence s n :=
by sorry

end distinct_solutions_of_an_eq_s_l243_243180


namespace max_eccentricity_ellipse_l243_243018

section ellipse

def foci_F1 : ℝ × ℝ := (-1, 0)
def foci_F2 : ℝ × ℝ := (1, 0)
def intersecting_line : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), p.1 - p.2 + 3 = 0

noncomputable def ellipse_eq (a b : ℝ) : Prop := (λ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1)

theorem max_eccentricity_ellipse :
  ∃ a b, (∀ x y, intersecting_line (x, y) → ellipse_eq a b x y) ∧
         foci_F1 = (-1, 0) ∧
         foci_F2 = (1, 0) ∧
         ellipse_eq 5 4 :=
sorry

end ellipse

end max_eccentricity_ellipse_l243_243018


namespace min_value_a_l243_243064

theorem min_value_a (a : ℝ) : 
  (∀ x ∈ Ioo 0 1, x^2 + a * x + 1 ≥ 0) → a ≥ -2 :=
by
  sorry

end min_value_a_l243_243064


namespace a2_value_l243_243086

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ ∀ n, 2 * a (n + 1) = a n + 1

theorem a2_value (a : ℕ → ℕ) (h : sequence a) : a 2 = 2 :=
by 
  sorry

end a2_value_l243_243086


namespace min_distance_l243_243330

noncomputable def curve (x : ℝ) : ℝ := (1/4) * x^2 - (1/2) * Real.log x
noncomputable def line (x : ℝ) : ℝ := (3/4) * x - 1

theorem min_distance :
  ∀ P Q : ℝ × ℝ, 
  P.2 = curve P.1 → 
  Q.2 = line Q.1 → 
  ∃ min_dist : ℝ, 
  min_dist = (2 - 2 * Real.log 2) / 5 := 
sorry

end min_distance_l243_243330


namespace max_k_value_l243_243029

def f (x : ℝ) : ℝ := x * real.log x + 3 * x - 2

def g (x : ℝ) : ℝ := (x * real.log x + 3 * x - 2) / (x - 1)

theorem max_k_value : 
  (∀ x > 1, k < g x) → k ≤ 5 :=
sorry

end max_k_value_l243_243029


namespace probability_no_adjacent_green_hats_l243_243817

-- Definitions
def total_children : ℕ := 9
def green_hats : ℕ := 3

-- Main theorem statement
theorem probability_no_adjacent_green_hats : 
  (9.choose 3) = 84 → 
  (1 - (9 + 45) / 84) = 5/14 := 
sorry

end probability_no_adjacent_green_hats_l243_243817


namespace smallest_m_exists_l243_243107

def S (z : ℂ) : Prop :=
  ∃ (x y : ℝ), (z = x + y * complex.I) ∧ (1/2 ≤ x ∧ x ≤ real.sqrt 2 / 2)

theorem smallest_m_exists :
  ∃ m : ℕ, (∀ n : ℕ, n ≥ m → (∃ z : ℂ, S(z) ∧ z^n = 1)) ∧ m = 13 := 
  sorry

end smallest_m_exists_l243_243107


namespace smallest_four_digit_divisible_by_53_l243_243661

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l243_243661


namespace smallest_four_digit_divisible_by_53_l243_243692

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243692


namespace least_side_is_8_l243_243395

-- Define the sides of the right triangle
variables (a b : ℝ) (h : a = 8) (k : b = 15)

-- Define a predicate for the least possible length of the third side
def least_possible_third_side (c : ℝ) : Prop :=
  (c = 8) ∨ (c = 15) ∨ (c = 17)

theorem least_side_is_8 (c : ℝ) (hc : least_possible_third_side c) : c = 8 :=
by
  sorry

end least_side_is_8_l243_243395


namespace product_of_two_numbers_in_ratio_l243_243581

theorem product_of_two_numbers_in_ratio (x y : ℚ) 
  (h1 : x - y = d)
  (h2 : x + y = 8 * d)
  (h3 : x * y = 15 * d) :
  x * y = 100 / 7 :=
by
  sorry

end product_of_two_numbers_in_ratio_l243_243581


namespace sean_net_profit_l243_243525

noncomputable def total_cost (num_patches : ℕ) (cost_per_patch : ℝ) : ℝ :=
  num_patches * cost_per_patch

noncomputable def total_revenue (num_patches : ℕ) (selling_price_per_patch : ℝ) : ℝ :=
  num_patches * selling_price_per_patch

noncomputable def net_profit (total_revenue : ℝ) (total_cost : ℝ) : ℝ :=
  total_revenue - total_cost

-- Variables based on conditions
def num_patches := 100
def cost_per_patch := 1.25
def selling_price_per_patch := 12.00

theorem sean_net_profit : net_profit (total_revenue num_patches selling_price_per_patch) (total_cost num_patches cost_per_patch) = 1075 :=
by
  sorry

end sean_net_profit_l243_243525


namespace area_sum_correct_l243_243071

def area_ABC_plus_twice_CDE : ℝ :=
  let AB := 1
  let AC := 1
  let α := Real.pi * 80 / 180 -- 80 degrees in radians
  let area_ABC := (1/2) * AB * AC * Real.sin α
  let area_CDE := (1/2) * (1/2) * (1/2)
  area_ABC + 2 * area_CDE

theorem area_sum_correct :
  area_ABC_plus_twice_CDE = (Real.sin (Real.pi * 80 / 180)) / 2 + 1/4 :=
by 
  sorry

end area_sum_correct_l243_243071


namespace least_possible_length_of_third_side_l243_243389

theorem least_possible_length_of_third_side (a b : ℕ) (h1 : a = 8) (h2 : b = 15) : 
  ∃ c : ℕ, c = 17 ∧ a^2 + b^2 = c^2 := 
by
  use 17 
  split
  · rfl
  · rw [h1, h2]
    norm_num

end least_possible_length_of_third_side_l243_243389


namespace limit_f_at_origin_l243_243025
-- Define the function f
def f (x y : ℝ) : ℝ :=
  if x^2 + y^2 = 0 then 0 else x * Real.sin (1 / y) + y * Real.sin (1 / x)

-- State the main theorem to be proved
theorem limit_f_at_origin :
  filter.tendsto (λ (p : ℝ × ℝ), f p.1 p.2) (filter.prod (nhds 0) (nhds 0)) (nhds 0) :=
sorry

end limit_f_at_origin_l243_243025


namespace maximum_value_OM_ON_squared_l243_243083
noncomputable theory

open Real

-- Given definitions and conditions
def M (m : ℝ) := (m, 0)
def N (n : ℝ) := (n, n)
def dist (a b : ℝ × ℝ) : ℝ := sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

theorem maximum_value_OM_ON_squared (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (hMN : dist (M m) (N n) = sqrt 2) :
  (m^2 + 2 * n^2 ≤ 4 + 2 * sqrt 2) :=
sorry

end maximum_value_OM_ON_squared_l243_243083


namespace max_value_n_l243_243541

theorem max_value_n (log2 : ℝ) (log3 : ℝ)
  (h_log2 : log2 ≈ 0.3010) (h_log3 : log3 ≈ 0.4771) :
  ∃ n : ℕ, (2 / 3) ^ n ≥ 1 / 30 ∧ n ≤ 8 :=
by
  sorry

end max_value_n_l243_243541


namespace smallest_D_D_geq_9_l243_243591

-- Definitions related to the chessboard labeling
def label := Fin 64
def chessboard := list (label × label)
def adjacent (a b : label) : Prop :=
  let a_row := (a / 8 : Int)
  let a_col := (a % 8 : Int)
  let b_row := (b / 8 : Int)
  let b_col := (b % 8 : Int)
  (abs (a_row - b_row) <= 1) ∧ (abs (a_col - b_col) <= 1) ∧ (a ≠ b)

-- Definition of D (largest difference between labels of adjacent squares)
def D (lbl : chessboard) : Nat :=
  lbl.foldr (λ (a_b : label × label) acc => max acc (abs (a_b.1 - a_b.2))) 0

-- The problem statement: finding smallest possible D
theorem smallest_D (ch : chessboard) : ∃ lbl, D lbl = 9 :=
sorry

-- The lower bound proof statement on D
theorem D_geq_9 : ∀ (ch : chessboard) (lbl : chessboard), D lbl ≥ 9 :=
sorry

end smallest_D_D_geq_9_l243_243591


namespace smallest_four_digit_divisible_by_53_l243_243640

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l243_243640


namespace smallest_four_digit_div_by_53_l243_243623

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l243_243623


namespace eval_expression_l243_243907

def f (x : ℤ) : ℤ := 3 * x^2 - 6 * x + 10

theorem eval_expression : 3 * f 2 + 2 * f (-2) = 98 := by
  sorry

end eval_expression_l243_243907


namespace max_value_S_n_l243_243181

noncomputable def a_n (n : ℕ) : ℤ := 13 - 3 * n
noncomputable def b_n (n : ℕ) : ℤ := a_n n * a_n (n + 1) * a_n (n + 2)
noncomputable def S_n (n : ℕ) : ℤ := (Finset.range n).sum b_n

theorem max_value_S_n : ∃ n : ℕ, S_n n = 310 :=
by {
  have h : S_n 2 = 308,
  -- calculate S_2
  sorry,
  have g : S_n 4 = 310,
  -- calculate S_4
  sorry,
  use 4,
  exact g,
}

end max_value_S_n_l243_243181


namespace g_pi_over_4_eq_neg_sqrt2_over_4_l243_243123

noncomputable def g (x : Real) : Real := 
  Real.sqrt (5 * (Real.sin x)^4 + 4 * (Real.cos x)^2) - 
  Real.sqrt (6 * (Real.cos x)^4 + 4 * (Real.sin x)^2)

theorem g_pi_over_4_eq_neg_sqrt2_over_4 :
  g (Real.pi / 4) = - (Real.sqrt 2) / 4 := 
sorry

end g_pi_over_4_eq_neg_sqrt2_over_4_l243_243123


namespace cosine_of_largest_angle_l243_243324

theorem cosine_of_largest_angle
  (a : ℝ)
  (h₁ : 0 < a)
  (h₂ : 2 * a > sqrt 2 * a)
  (h₃ : sqrt 2 * a > a) :
  ∃ θ : ℝ, θ = real.arccos (- (sqrt 2 / 4)) :=
by
  have h₄ : 2 * a = sqrt 2 * (sqrt 2 * a) := by ring
  sorry

end cosine_of_largest_angle_l243_243324


namespace least_third_side_of_right_triangle_l243_243381

theorem least_third_side_of_right_triangle (a b : ℕ) (ha : a = 8) (hb : b = 15) :
  ∃ c : ℝ, c = real.sqrt (b^2 - a^2) ∧ c = real.sqrt 161 :=
by {
  -- We state the conditions
  have h8 : (8 : ℝ) = a, from by {rw ha},
  have h15 : (15 : ℝ) = b, from by {rw hb},

  -- The theorem states that such a c exists
  use (real.sqrt (15^2 - 8^2)),

  -- We need to show the properties of c
  split,
  { 
    -- Showing that c is the sqrt of the difference of squares of b and a
    rw [←h15, ←h8],
    refl 
  },
  {
    -- Showing that c is sqrt(161)
    calc
       real.sqrt (15^2 - 8^2)
         = real.sqrt (225 - 64) : by norm_num
     ... = real.sqrt 161 : by norm_num
  }
}
sorry

end least_third_side_of_right_triangle_l243_243381


namespace total_savings_proof_l243_243129

def oranges_saved (Liam_oranges Claire_oranges Jake_oranges : ℕ)
  (Liam_rate Claire_rate Jake_bundle1_rate Jake_bundle2_rate : ℚ)
  (Jake_bundles_total first_half_bundles second_half_bundles : ℕ) : ℚ :=
  let Liam_total := (Liam_oranges / 2) * Liam_rate
  let Claire_total := Claire_oranges * Claire_rate
  let Jake_total := (first_half_bundles * Jake_bundle1_rate) + (second_half_bundles * Jake_bundle2_rate)
  Liam_total + Claire_total + Jake_total

theorem total_savings_proof : oranges_saved 40 30 50 2.5 1.2 3 4.5 10 5 5 = 123.50 :=
by
  unfold oranges_saved
  norm_num
  sorry

end total_savings_proof_l243_243129


namespace longer_part_length_l243_243254

-- Conditions
def total_length : ℕ := 180
def diff_length : ℕ := 32

-- Hypothesis for the shorter part of the wire
def shorter_part (x : ℕ) : Prop :=
  x + (x + diff_length) = total_length

-- The goal is to find the longer part's length
theorem longer_part_length (x : ℕ) (h : shorter_part x) : x + diff_length = 106 := by
  sorry

end longer_part_length_l243_243254


namespace product_of_digits_of_m_l243_243478

-- Define the conditions and problem
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def valid_prime_pair (d e : ℕ) : Prop :=
  is_prime d ∧ d ∈ {2, 3, 5} ∧ e = d + 10 ∧ is_prime e

def m_value : ℕ :=
  if h : valid_prime_pair 3 13 then 3 * 13 else 0  -- Since we know (3, 13) is the only valid pair

def product_of_digits (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem product_of_digits_of_m : product_of_digits m_value = 27 := 
by 
  unfold m_value 
  unfold product_of_digits 
  -- Verify the pair (3, 13) fulfills conditions
  have : valid_prime_pair 3 13 := sorry 
  -- Calculate the product of digits for the valid m_value
  simp only [this, if_pos, nat.div, nat.mod, nat.succ_add]

  sorry  -- Proof steps to complete the theorem proof

end product_of_digits_of_m_l243_243478


namespace smallest_four_digit_multiple_of_53_l243_243777

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l243_243777


namespace slope_angle_of_vertical_line_l243_243185

theorem slope_angle_of_vertical_line :
  ∀ (x : ℝ), (x = 1) → (angle_of_slope (x = 1) = π / 2) :=
by
  intros x hx
  rw hx
  have h : x = 1 := by sorry -- this is just to assert the hypothesis
  -- The actual proof goes here
  sorry -- skipping the proof

end slope_angle_of_vertical_line_l243_243185


namespace find_a_l243_243051

theorem find_a (x y a : ℤ) (hxy: x = 1 ∧ y = 2) (h_eq: x + a = 3 * y - 2) : a = 3 :=
by
  obtain ⟨hx, hy⟩ := hxy
  rw [hx, hy] at h_eq
  simp at h_eq
  assumption

end find_a_l243_243051


namespace frost_cupcakes_l243_243887

def frosting_rate_cagney := 1 / 15 -- Rate for Cagney
def frosting_rate_lacey := 1 / 25 -- Rate for Lacey
def total_time_in_seconds := 10 * 60 -- Total time of 10 minutes in seconds
def delay_time_in_seconds := 30 -- Delay for Lacey to start

def cupcakes_frosted_in_given_time : Nat :=
  let time_only_cagney := delay_time_in_seconds / 15
  let combined_rate := frosting_rate_cagney + frosting_rate_lacey
  let remaining_time := total_time_in_seconds - delay_time_in_seconds
  let cupcakes_with_both := remaining_time * combined_rate
  time_only_cagney + cupcakes_with_both.to_nat

theorem frost_cupcakes : cupcakes_frosted_in_given_time = 62 := 
  by
    -- The proof that total cupcakes frosted will be 62
    sorry

end frost_cupcakes_l243_243887


namespace num_ways_arith_prog_l243_243441

theorem num_ways_arith_prog : 
  ∑ (d : ℕ) in finset.range 334, 1000 - 3 * d = 166167 :=
by
  sorry

end num_ways_arith_prog_l243_243441


namespace Ariel_age_l243_243879

theorem Ariel_age :
  ∀ (fencing_start_year birth_year: ℕ) (fencing_years: ℕ),
    fencing_start_year = 2006 →
    birth_year = 1992 →
    fencing_years = 16 →
    (fencing_start_year + fencing_years - birth_year) = 30 :=
by
  intros fencing_start_year birth_year fencing_years h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end Ariel_age_l243_243879


namespace ancient_china_pentatonic_scale_l243_243081

theorem ancient_china_pentatonic_scale (a : ℝ) (h : a * (2/3) * (4/3) * (2/3) = 32) : a = 54 :=
by
  sorry

end ancient_china_pentatonic_scale_l243_243081


namespace area_of_figure_ABCD_l243_243583

noncomputable def radius : ℝ := 12
noncomputable def angle : ℝ := 60
noncomputable def num_sectors : ℕ := 2

theorem area_of_figure_ABCD (r : ℝ) (theta : ℝ) (n : ℕ) :
  r = radius → theta = angle → n = num_sectors → 
  let full_circle_area := π * r^2 in
  let sector_area := (theta / 360) * full_circle_area in
  let total_area := n * sector_area in
  total_area = 48 * π := 
by
  intros hr htheta hn
  have h1 : full_circle_area = π * r^2 := rfl
  have h2 : sector_area = (theta / 360) * full_circle_area := rfl
  have h3 : total_area = n * sector_area := rfl
  rw [hr, htheta, hn] at *
  sorry

end area_of_figure_ABCD_l243_243583


namespace right_triangle_least_side_l243_243402

theorem right_triangle_least_side (a b c : ℝ) (h_rt : a^2 + b^2 = c^2) (h1 : a = 8) (h2 : b = 15) : min a b = 8 := 
by
sorry

end right_triangle_least_side_l243_243402


namespace smallest_four_digit_divisible_by_53_l243_243657

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l243_243657


namespace smallest_four_digit_multiple_of_53_l243_243771

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243771


namespace ceil_sqrt_225_l243_243952

theorem ceil_sqrt_225 : ⌈real.sqrt 225⌉ = 15 :=
by
  have h : real.sqrt 225 = 15 := by
    sorry
  rw [h]
  exact int.ceil_eq_self.mpr rfl

end ceil_sqrt_225_l243_243952


namespace maximum_additional_plates_l243_243434

-- Definitions of sets for each character position.
def set_first := {'B', 'G', 'J', 'S'}
def set_second := {'E', 'U'}
def set_third := {'K', 'V', 'X'}

-- Compute the initial number of possible license plates.
def initial_plates : ℕ := set_first.size * set_second.size * set_third.size

-- Define new sets with added letters (various cases).
def set_second_with_two_added := set_second.size + 2
def set_third_with_one_added := set_third.size + 1
def set_second_with_one_added := set_second.size + 1

-- Compute the maximum possible new license plates.
def case_1_new_plates := set_first.size * set_second_with_two_added * set_third.size
def case_2_new_plates := set_first.size * set_second_with_one_added * set_third_with_one_added

-- The largest possible increase in the number of license plates.
def max_additional_plates : ℕ := max (case_1_new_plates - initial_plates) (case_2_new_plates - initial_plates)

theorem maximum_additional_plates : max_additional_plates = 24 :=
by {
  -- The proof goes here...
  sorry
}

end maximum_additional_plates_l243_243434


namespace ceil_sqrt_225_eq_15_l243_243931

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := 
by 
  sorry

end ceil_sqrt_225_eq_15_l243_243931


namespace smallest_four_digit_divisible_by_53_l243_243682

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l243_243682


namespace min_coins_to_cover_board_l243_243866

theorem min_coins_to_cover_board (n : ℕ) : 
  ∃ k : ℕ, (k = Int.ceil ((2 * n - 1 : ℤ) / 3) ∧ 
  ∀ (i j : ℕ), (i < n ∧ j < n) → (cell_has_coin(i, j, k) ∨ cell_in_row_with_coin(i, k) ∨ cell_in_column_with_coin(j, k) ∨ cell_in_positive_diagonal_with_coin(i, j, k))) :=
sorry

end min_coins_to_cover_board_l243_243866


namespace diamond_problem_l243_243112

def diamond (a b : ℝ) : ℝ := (3 * a / b) * (b / a)

theorem diamond_problem : diamond 7 (diamond 4 9) = 3 ∧ diamond (diamond 7 (diamond 4 9)) 2 = 3 :=
by
  -- Using the definition of diamond:
  have h1 : diamond 4 9 = 3 := by sorry
  have h2 : diamond 7 3 = 3 := by sorry
  have h3 : diamond 3 2 = 3 := by sorry
  exact ⟨h2, h3⟩

end diamond_problem_l243_243112


namespace range_of_x_in_f_l243_243455

def f (x : ℝ) : ℝ := real.sqrt (x + 2)

theorem range_of_x_in_f (x : ℝ) : x ≥ -2 :=
by {
  sorry
}

end range_of_x_in_f_l243_243455


namespace smallest_four_digit_divisible_by_53_l243_243679

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l243_243679


namespace smallest_four_digit_divisible_by_53_l243_243699

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243699


namespace smallest_four_digit_divisible_by_53_l243_243750

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243750


namespace trigonometric_expression_eval_l243_243021

noncomputable def α : ℝ := sorry -- α is an angle

def λ : ℝ := sorry -- λ is a real number representing a non-zero scalar

axiom λ_nonzero : λ ≠ 0

axiom point_on_terminal_side : ∃ λ: ℝ, λ ≠ 0 ∧ 
  (tan α = (4 * λ) / (-3 * λ))

-- Theorem we need to prove
theorem trigonometric_expression_eval :
  ∃ α: ℝ, (point_on_terminal_side) →
    (λ ≠ 0) →
    ((sin α + cos α) / (sin α - cos α) = 1 / 7) :=
by
  sorry

end trigonometric_expression_eval_l243_243021


namespace parallelogram_area_l243_243136

/-- Given a parallelogram with one angle of 120 degrees and sides of lengths 8 and 15,
    prove that its area is 60 * sqrt(3). -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (area : ℝ) 
  (h1 : a = 8) (h2 : b = 15) (h3 : θ = 120) : area = 60 * real.sqrt 3 := 
sorry

end parallelogram_area_l243_243136


namespace solve_for_X_l243_243532

theorem solve_for_X (X : ℝ) (h : (X ^ (5 / 4)) = 32 * (32 ^ (1 / 16))) :
  X =  16 * (2 ^ (1 / 4)) :=
sorry

end solve_for_X_l243_243532


namespace smallest_four_digit_divisible_by_53_l243_243746

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243746


namespace Xiao_Ming_max_notebooks_l243_243212

-- Definitions of the given conditions
def total_yuan : ℝ := 30
def total_books : ℕ := 30
def notebook_cost : ℝ := 4
def exercise_book_cost : ℝ := 0.4

-- Definition of the variables used in the inequality
def x (max_notebooks : ℕ) : ℝ := max_notebooks
def exercise_books (max_notebooks : ℕ) : ℝ := total_books - x max_notebooks

-- Definition of the total cost inequality
def total_cost (max_notebooks : ℕ) : ℝ :=
  x max_notebooks * notebook_cost + exercise_books max_notebooks * exercise_book_cost

theorem Xiao_Ming_max_notebooks (max_notebooks : ℕ) : total_cost max_notebooks ≤ total_yuan → max_notebooks ≤ 5 :=
by
  -- Proof goes here
  sorry

end Xiao_Ming_max_notebooks_l243_243212


namespace geom_seq_product_l243_243339

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  2 * a 3 - (a 8) ^ 2 + 2 * a 13 = 0

def geometric_seq (b : ℕ → ℤ) (a8 : ℤ) : Prop :=
  b 8 = a8

theorem geom_seq_product (a b : ℕ → ℤ) (a8 : ℤ) 
  (h1 : arithmetic_seq a)
  (h2 : geometric_seq b a8)
  (h3 : a8 = 4)
: b 4 * b 12 = 16 := sorry

end geom_seq_product_l243_243339


namespace smallest_four_digit_divisible_by_53_l243_243718

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l243_243718


namespace smallest_four_digit_divisible_by_53_l243_243754

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243754


namespace smallest_four_digit_divisible_by_53_l243_243650

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l243_243650


namespace probability_no_adjacent_green_hats_l243_243816

-- Step d): Rewrite the math proof problem in a Lean 4 statement.

theorem probability_no_adjacent_green_hats (total_children green_hats : ℕ)
  (hc : total_children = 9) (hg : green_hats = 3) :
  (∃ (p : ℚ), p = 5 / 14) :=
sorry

end probability_no_adjacent_green_hats_l243_243816


namespace no_4_7_9_digits_four_digit_count_l243_243048

theorem no_4_7_9_digits_four_digit_count : 
  (number_of_valid_digits (d: ℕ) (d ≥ 1000 ∧ d ≤ 9999) (∀ i ∈ {d / 1000, (d / 100) % 10, (d / 10) % 10, d % 10}, i ≠ 4 ∧ i ≠ 7 ∧ i ≠ 9)) = 2058 :=
  sorry

end no_4_7_9_digits_four_digit_count_l243_243048


namespace ab_value_in_triangle_l243_243090

theorem ab_value_in_triangle {a b c : ℝ} (h1 : (a + b)^2 - c^2 = 4) (h2 : cos (real.pi / 3) = 1 / 2) :
  a * b = 4 / 3 :=
by
  sorry

end ab_value_in_triangle_l243_243090


namespace all_visitors_can_buy_tickets_l243_243073

-- Define the structure of the problem
structure Visitor :=
  (balance : ℕ)
  (coins3 : ℕ)
  (coins5 : ℕ)

def initial_cashier_balance : Visitor := 
  { balance := 22, coins3 := 4, coins5 := 2 }

def entrance_ticket_cost : ℕ := 4

def conditions_satisfied (v : Visitor) : Prop :=
  let coins_value := v.coins3 * 3 + v.coins5 * 5 in
  coins_value = v.balance ∧
  v.balance = 22

theorem all_visitors_can_buy_tickets :
  ∀ queue_length : ℕ, (queue_length = 200) →
  ∀ visitors : list Visitor, (∀ v ∈ visitors, conditions_satisfied v) →
  ∃ (cashier : Visitor), conditions_satisfied cashier ∧
  (∀ i, i < queue_length → 
    let visitor := visitors.nth_le i sorry in
    let new_cashier := 
      if i % 2 = 0 then 
        { visitor with balance := visitor.balance - entrance_ticket_cost, coins3 := visitor.coins3 - 3, coins5 := visitor.coins5 }
      else 
        { visitor with balance := visitor.balance - entrance_ticket_cost, coins3 := visitor.coins3, coins5 := visitor.coins5 - 2 } in
    conditions_satisfied new_cashier)
  :=
begin
  sorry
end

end all_visitors_can_buy_tickets_l243_243073


namespace find_trapezoid_area_l243_243237

-- Given conditions as definitions
variables {AB CD BC AD : ℝ}
variables {CL DL : ℝ}
variables {ω : Type*} -- the inscribed circle
variables {TrapezoidArea : ℝ}

-- Let CL:LD = 1:4; thus CL/CD = 1/5 and LD/CD = 4/5
def ratio_CL_DL (CL DL CD : ℝ) := CL = CD / 5 ∧ DL = (4 * CD) / 5
-- Known lengths
def known_lengths (BC CD : ℝ) := BC = 9 ∧ CD = 30
-- Tangent equalities stemming from inscribed circle properties
def tangent_equalities (CL DL : ℝ) := true -- placeholder for the tangency which is CL and DL partitioning CD correctly, handled above.
-- Trapezoid area calculation
def trapezoid_area (BC AD h : ℝ) := (BC + AD) / 2 * h = 972

-- The main theorem to prove the area given the conditions
theorem find_trapezoid_area 
  (CL : ℝ)
  (DL : ℝ)
  (BC : ℝ := 9)
  (CD : ℝ := 30)
  (ω : Type*) 
  (ratio : ratio_CL_DL CL DL CD) 
  (lengths : known_lengths BC CD)
  (equalities : tangent_equalities CL DL)
  : ∃ AD h, trapezoid_area BC AD h := 
begin
  -- Proof details go here.
  sorry
end

end find_trapezoid_area_l243_243237


namespace range_of_f_is_pi_div_four_l243_243977

noncomputable def f (x : ℝ) : ℝ := 
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f_is_pi_div_four : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y = π / 4 :=
sorry

end range_of_f_is_pi_div_four_l243_243977


namespace lemonade_stand_l243_243868

theorem lemonade_stand:
  ∃ S: ℝ, 21 * 4 - (10 + S + 3) = 66 ∧ S = 5 :=
by
  use 5
  rw [←add_assoc 10 S 3, ←other side expression, ←equality simplification steps]
  sorry

end lemonade_stand_l243_243868


namespace right_triangle_least_side_l243_243403

theorem right_triangle_least_side (a b c : ℝ) (h_rt : a^2 + b^2 = c^2) (h1 : a = 8) (h2 : b = 15) : min a b = 8 := 
by
sorry

end right_triangle_least_side_l243_243403


namespace smallest_four_digit_divisible_by_53_l243_243739

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243739


namespace Maria_drove_approximately_517_miles_l243_243494

noncomputable def carRentalMaria (daily_rate per_mile_charge discount_rate insurance_rate rental_duration total_invoice : ℝ) (discount_threshold : ℕ) : ℝ :=
  let total_rental_cost := rental_duration * daily_rate
  let discount := if rental_duration ≥ discount_threshold then discount_rate * total_rental_cost else 0
  let discounted_cost := total_rental_cost - discount
  let insurance_cost := rental_duration * insurance_rate
  let cost_without_mileage := discounted_cost + insurance_cost
  let mileage_cost := total_invoice - cost_without_mileage
  mileage_cost / per_mile_charge

noncomputable def approx_equal (a b : ℝ) (epsilon : ℝ := 1) : Prop :=
  abs (a - b) < epsilon

theorem Maria_drove_approximately_517_miles :
  approx_equal (carRentalMaria 35 0.09 0.10 5 4 192.50 3) 517 :=
by
  sorry

end Maria_drove_approximately_517_miles_l243_243494


namespace sum_of_first_9_primes_l243_243219

theorem sum_of_first_9_primes : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23) = 100 := 
by
  sorry

end sum_of_first_9_primes_l243_243219


namespace problem_a_problem_b_problem_c_l243_243488

variables {A B C A1 A1' B1 B1' C1 C1' M N : Type} [AffineSpace A B C A1 A1' B1 B1' C1 C1' M N]

-- Variables describing lines passing through triangle vertices
variables {AA1 : Line A A1} {BB1 : Line B B1} {CC1 : Line C C1}
variables {AA1' : Line A A1'} {BB1' : Line B B1'} {CC1' : Line C C1'}

-- Points of interest
variables {A1_ B1_ C1_ : Point} {A1'_ B1'_ C1'_ : Point}

-- Hypothesis for part (a)
variables (h_collinear_A1_B1_C1 : Collinear [A1, B1, C1])

-- Hypothesis for part (b)
variables (h_concurrent_AA1_BB1_CC1 : Concurrent [AA1, BB1, CC1])

-- Hypothesis for part (c)
variables (h_intersect_M : IntersectsAt [AA1, BB1, CC1] M)
variables (h_intersect_N : IntersectsAt [AA1', BB1', CC1'] N)

theorem problem_a 
  (h_collinear_A1_B1_C1 : Collinear [A1, B1, C1]) : 
  Collinear [A1', B1', C1'] := 
sorry

theorem problem_b 
  (h_concurrent_AA1_BB1_CC1 : Concurrent [AA1, BB1, CC1]) : 
  Concurrent [AA1', BB1', CC1'] := 
sorry

theorem problem_c 
  (h_intersect_M : IntersectsAt [AA1, BB1, CC1] M)
  (h_intersect_N : IntersectsAt [AA1', BB1', CC1'] N) : 
  Concyclic [Projections M, Projections N] := 
sorry

end problem_a_problem_b_problem_c_l243_243488


namespace sum_of_b_l243_243001

def sequence_a (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, S n = 2 * a n - 1

def sequence_b_def (a b : ℕ → ℕ) : Prop :=
b 1 = 3 ∧ ∀ k : ℕ, k ≥ 1 → b (k + 1) = a k + b k

def sum_b_eq (b : ℕ → ℕ) (n : ℕ) : ℕ :=
∑ k in finset.range n, b (k + 1)

theorem sum_of_b (a b : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ)
  (a_seq : sequence_a a S) (b_seq : sequence_b_def a b) :
  sum_b_eq b n = 2^n + 2 * n - 1 := by
  sorry 

end sum_of_b_l243_243001


namespace smallest_four_digit_multiple_of_53_l243_243778

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l243_243778


namespace smallest_four_digit_divisible_by_53_l243_243637

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l243_243637


namespace smallest_four_digit_multiple_of_53_l243_243715

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243715


namespace addition_of_decimals_l243_243571

theorem addition_of_decimals : (0.3 + 0.03 : ℝ) = 0.33 := by
  sorry

end addition_of_decimals_l243_243571


namespace smallest_four_digit_multiple_of_53_l243_243768

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243768


namespace prime_condition_l243_243859

def is_prime (n : ℕ) : Prop := nat.prime n

theorem prime_condition (p : ℕ) (q : ℕ) (h_prime_p : is_prime p) (h_eq : p + 25 = q ^ 7) (h_prime_q : is_prime q) : p = 103 :=
sorry

end prime_condition_l243_243859


namespace product_value_l243_243103

open Real

theorem product_value (A R M L : ℝ)
  (h₁ : log 2 (A * L) + log 2 (A * M) = 5)
  (h₂ : log 2 (M * L) + log 2 (M * R) = 6)
  (h₃ : log 2 (R * A) + log 2 (R * L) = 7) :
  A * R * M * L = 64 :=
by
  sorry

end product_value_l243_243103


namespace meal_combinations_l243_243800

theorem meal_combinations (n : ℕ) (h : n = 10) : 
  let Yann_choices := n in
  let Camille_choices := n in
  Yann_choices * Camille_choices = 100 := by
{
  simp [Yann_choices, Camille_choices, h],
  sorry
}

end meal_combinations_l243_243800


namespace smallest_four_digit_divisible_by_53_l243_243729

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l243_243729


namespace infinite_divisibility_of_2n_plus_n2_by_100_l243_243140

theorem infinite_divisibility_of_2n_plus_n2_by_100 :
  ∃ᶠ n in at_top, 100 ∣ (2^n + n^2) :=
sorry

end infinite_divisibility_of_2n_plus_n2_by_100_l243_243140


namespace angle_PCA_l243_243236

theorem angle_PCA (A B C P Q K L : Point)
  (ω : Circle)
  (h_circumscribed_triangle : CircumscribedTriangle ω A B C)
  (h_tangent_at_C : TangentAt ω C (Line.through P C))
  (h_intersect_ray_BA_P : IntersectRay (Ray.through B A) (Line.through P C))
  (h_on_ray_PC_beyond_C_Q : OnRayBeyondPoint (Ray.through P C) C Q)
  (h_PC_eq_QC : dist P C = dist Q C)
  (h_intersect_segment_BQ : IntersectSegment B Q ω K)
  (h_L_on_smaller_arc_BK : OnSmallerArc ω B K L)
  (h_angle_LAK_eq_angle_CQB : ∠LAK = ∠CQB)
  (h_angle_ALQ_60 : ∠ALQ = 60°) :
  ∠PCA = 30° := 
sorry

end angle_PCA_l243_243236


namespace sum_floor_log3_l243_243266

theorem sum_floor_log3 :
  (∑ N in Finset.range 2048, Int.floor (Real.log N / Real.log 3)) = 12049 :=
sorry

end sum_floor_log3_l243_243266


namespace finished_year_eq_183_l243_243438

theorem finished_year_eq_183 (x : ℕ) (h1 : x < 200) 
  (h2 : x ^ 13 = 258145266804692077858261512663) : x = 183 :=
sorry

end finished_year_eq_183_l243_243438


namespace line_passes_through_centroid_of_triangle_l243_243482

open EuclideanGeometry

theorem line_passes_through_centroid_of_triangle
  (A B C M N G : Point)
  (hM : M ∈ Segment A B)
  (hN : N ∈ Segment A C)
  (hCond : (dist B M) / (dist M A) + (dist C N) / (dist N A) = 1)
  (hG : is_centroid G A B C) :
  collinear ℝ {M, N, G} :=
begin
  sorry
end

end line_passes_through_centroid_of_triangle_l243_243482


namespace value_of_x_l243_243084

theorem value_of_x 
  (P Q R S T U : Point)
  (angle_TQR : angle T Q R = 125)
  (straight_line_1 : collinear Px Q R)
  (straight_line_2 : collinear Q S T)
  (straight_line_3 : collinear P S U)
  (angle_SPQ : angle S P Q = 30)
  (angle_SQP : angle S Q P = angle T Q P) :
  angle T S U = 95 :=
by
  sorry

end value_of_x_l243_243084


namespace darren_fergie_same_balance_after_53_75_days_l243_243275

noncomputable def darren_balance (t : ℝ) : ℝ :=
  if t <= 10 then 200 * (1 + 0.08 * t)
  else 360 * (1 + 0.06 * (t - 10))

def fergie_balance (t : ℝ) : ℝ :=
  300 * (1 + 0.04 * t)

theorem darren_fergie_same_balance_after_53_75_days :
  ∀ t > 10, darren_balance t = fergie_balance t ↔ t = 53.75 :=
begin
  intro t,
  intro ht,
  split,
  { intro h,
    sorry },
  { intro h,
    rw h,
    sorry }
end

end darren_fergie_same_balance_after_53_75_days_l243_243275


namespace smallest_four_digit_divisible_by_53_l243_243730

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l243_243730


namespace problem_l243_243059

theorem problem (a b : ℤ) (h1 : |a - 2| = 5) (h2 : |b| = 9) (h3 : a + b < 0) :
  a - b = 16 ∨ a - b = 6 := 
sorry

end problem_l243_243059


namespace sector_area_proof_l243_243017

-- Define the conditions
def central_angle_degrees : ℝ := 60
def arc_length : ℝ := 2 * Real.pi

-- Define the radius of the sector deduced from the conditions
noncomputable def radius (central_angle_degrees arc_length : ℝ) : ℝ :=
  (arc_length * 360) / (2 * Real.pi * central_angle_degrees)

-- Define the central angle in radians
def central_angle_radians (central_angle_degrees : ℝ) : ℝ :=
  (central_angle_degrees / 360) * 2 * Real.pi

-- Define the formula for the area of the sector
noncomputable def sector_area (radius central_angle_radians : ℝ) : ℝ :=
  0.5 * radius^2 * central_angle_radians

-- The proof statement
theorem sector_area_proof :
  central_angle_degrees = 60 →
  arc_length = 2 * Real.pi →
  sector_area (radius central_angle_degrees arc_length) (central_angle_radians central_angle_degrees) = 6 * Real.pi := 
by
  intros h1 h2
  -- Proof steps would go here
  sorry

end sector_area_proof_l243_243017


namespace least_third_side_length_l243_243415

theorem least_third_side_length (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℝ, (c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2)) ∧ c = Real.sqrt 161 :=
by
  sorry

end least_third_side_length_l243_243415


namespace muffin_banana_ratio_l243_243154

variable {R : Type} [LinearOrderedField R]

-- Define the costs of muffins and bananas
variables {m b : R}

-- Susie's cost
def susie_cost (m b : R) := 4 * m + 5 * b

-- Calvin's cost for three times Susie's items
def calvin_cost_tripled (m b : R) := 12 * m + 15 * b

-- Calvin's actual cost
def calvin_cost_actual (m b : R) := 2 * m + 12 * b

theorem muffin_banana_ratio (m b : R) (h : calvin_cost_tripled m b = calvin_cost_actual m b) : m = (3 / 10) * b :=
by sorry

end muffin_banana_ratio_l243_243154


namespace sufficient_condition_perpendicular_l243_243109

variables {Plane Line : Type}
variables (l : Line) (α β : Plane)

-- Definitions for perpendicularity and parallelism
def perp (l : Line) (α : Plane) : Prop := sorry
def parallel (α β : Plane) : Prop := sorry

theorem sufficient_condition_perpendicular
  (h1 : perp l α) 
  (h2 : parallel α β) : 
  perp l β :=
sorry

end sufficient_condition_perpendicular_l243_243109


namespace remainder_of_48_305_312_div_6_l243_243056

theorem remainder_of_48_305_312_div_6 : 48_305_312 % 6 = 2 := 
by 
  -- Here we are stating the proof goal directly.
  -- The details to fill here are left as sorry to match the instructions.
  sorry

end remainder_of_48_305_312_div_6_l243_243056


namespace william_left_missouri_at_7am_l243_243210

def total_travel_time (driving_hours stops_minutes : Nat) : Nat :=
  driving_hours + stops_minutes / 60

def departure_time (arrival_time : Time) (travel_hours : Nat) : Time :=
  arrival_time - travel_hours * hour

theorem william_left_missouri_at_7am :
  ∀ (arrival_time : Time),
  arrival_time = Time.mk 20 0 →
  let stops_minutes := 25 + 10 + 25 in
  let driving_hours := 12 in
  total_travel_time driving_hours stops_minutes = 13 →
  departure_time arrival_time 13 = Time.mk 7 0 := by
  intros arrival_time hArrival hTravel
  rw [hArrival]
  have : stops_minutes = 25 + 10 + 25 := rfl
  have : driving_hours = 12 := rfl
  rw [this, this] at hTravel
  exact sorry

end william_left_missouri_at_7am_l243_243210


namespace find_angle_FYD_l243_243452

-- Step a): Conditions
variables {A B C D E F X Y : Type}  -- The points in the problem
variable [parallel : A ≠ B]  -- Ensure A and B are distinct points
variable [parallel : C ≠ D]  -- Ensure C and D are distinct points
variable [parallel_AB_CD : A ≠ C ∧ B ≠ D]  -- Ensure line segments are distinct and parallel
variable (h_parallel : ∀ x : Type, x ∈ A ∧ x ∈ B ↔ x ∈ C ∧ x ∈ D)  -- Formalization of parallel lines AB ∥ CD
variable (h_AXF : Type)  -- Angle AXF
variable (angle_AXF : 118)  -- Formalization of Angle AXF

-- Step d): Statement
theorem find_angle_FYD :
  ∀ (A B C D E F X Y : Type) (h_parallel : ∀ x, (x ∈ A ∧ x ∈ B) ↔ (x ∈ C ∧ x ∈ D)) (angle_AXF : ℝ),
  angle_AXF = 118 → 
  angle_FYD = 62 := begin
  sorry
end

end find_angle_FYD_l243_243452


namespace evaluate_f_at_7_l243_243378

theorem evaluate_f_at_7 :
  (∃ f : ℕ → ℕ, (∀ x, f (2 * x + 1) = x ^ 2 - 2 * x) ∧ f 7 = 3) :=
by 
  sorry

end evaluate_f_at_7_l243_243378


namespace smallest_four_digit_divisible_by_53_l243_243721

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l243_243721


namespace value_of_y_l243_243921

theorem value_of_y (y : ℚ) : |4 * y - 6| = 0 ↔ y = 3 / 2 :=
by
  sorry

end value_of_y_l243_243921


namespace Mia_biking_speed_l243_243284

theorem Mia_biking_speed
    (Eugene_speed : ℝ)
    (Carlos_ratio : ℝ)
    (Mia_ratio : ℝ)
    (Mia_speed : ℝ)
    (h1 : Eugene_speed = 5)
    (h2 : Carlos_ratio = 3 / 4)
    (h3 : Mia_ratio = 4 / 3)
    (h4 : Mia_speed = Mia_ratio * (Carlos_ratio * Eugene_speed)) :
    Mia_speed = 5 :=
by
  sorry

end Mia_biking_speed_l243_243284


namespace total_people_selected_l243_243579

-- Define the number of residents in each age group
def residents_21_to_35 : Nat := 840
def residents_36_to_50 : Nat := 700
def residents_51_to_65 : Nat := 560

-- Define the number of people selected from the 36 to 50 age group
def selected_36_to_50 : Nat := 100

-- Define the total number of residents
def total_residents : Nat := residents_21_to_35 + residents_36_to_50 + residents_51_to_65

-- Theorem: Prove that the total number of people selected in this survey is 300
theorem total_people_selected : (100 : ℕ) / (700 : ℕ) * (residents_21_to_35 + residents_36_to_50 + residents_51_to_65) = 300 :=
  by 
    sorry

end total_people_selected_l243_243579


namespace ceil_sqrt_225_eq_15_l243_243932

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := 
by 
  sorry

end ceil_sqrt_225_eq_15_l243_243932


namespace valid_triangle_combinations_l243_243326

def is_triangle (a b c : ℝ) : Prop := 
  a + b > c ∧ a + c > b ∧ b + c > a

def count_valid_triangles (lengths : List ℝ) : ℕ :=
  lengths.combinations 3 |>.filter (λ l => is_triangle l[0] l[1] l[2]) |>.length

theorem valid_triangle_combinations : 
  count_valid_triangles [2, 3, 4, 6] = 2 := 
by 
  sorry

end valid_triangle_combinations_l243_243326


namespace quadratic_equation_among_options_l243_243796

def is_quadratic (f : ℝ → ℝ) : Prop := 
  ∃ a b c : ℝ, a ≠ 0 ∧ f = λ x, a * x^2 + b * x + c

def option_A : ℝ → ℝ := λ x, 2 * x + 1
def option_B : ℝ → ℝ := λ x, x^2 + x - 2
def option_C (x y : ℝ) : Prop := y^2 + x = 1
def option_D : ℝ → ℝ := λ x, 1 / x + x^2

theorem quadratic_equation_among_options : 
  is_quadratic option_A → 
  is_quadratic option_C → 
  is_quadratic option_D → 
  ¬ is_quadratic option_A ∧ ¬ is_quadratic option_C ∧ ¬ is_quadratic option_D ∧ is_quadratic option_B :=
  sorry

end quadratic_equation_among_options_l243_243796


namespace smallest_four_digit_multiple_of_53_l243_243773

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243773


namespace bat_wings_area_l243_243446

-- Defining a rectangle and its properties.
structure Rectangle where
  PQ : ℝ
  QR : ℝ
  PT : ℝ
  TR : ℝ
  RQ : ℝ

-- Example rectangle from the problem
def PQRS : Rectangle := { PQ := 5, QR := 3, PT := 1, TR := 1, RQ := 1 }

-- Calculate area of "bat wings" if the rectangle is specified as in the above structure.
-- Expected result is 3.5
theorem bat_wings_area (r : Rectangle) (hPQ : r.PQ = 5) (hQR : r.QR = 3) 
    (hPT : r.PT = 1) (hTR : r.TR = 1) (hRQ : r.RQ = 1) : 
    ∃ area : ℝ, area = 3.5 :=
by
  -- Adding the proof would involve geometric calculations.
  -- Skipping the proof for now.
  sorry

end bat_wings_area_l243_243446


namespace square_of_1023_l243_243901

theorem square_of_1023 : 1023^2 = 1045529 := by
  sorry

end square_of_1023_l243_243901


namespace smallest_four_digit_divisible_by_53_l243_243743

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243743


namespace cylinders_cannot_move_in_cube_l243_243512

def height (a : ℝ) := a
def diameter (a : ℝ) := a / 2
def edge_length (a : ℝ) := a
def radius (a : ℝ) := a / 4
def placement (cylinders_placed : Bool) := cylinders_placed = true

theorem cylinders_cannot_move_in_cube (a : ℝ) (cylinders_placed_correctly : placement true) :
  ∀ (cyl1 cyl2 cyl3: ℝ × ℝ × ℝ), 
  (height a ∧ diameter a / 2 ∧ edge_length a ∧ radius a ∧ cylinders_placed_correctly) → 
  no_move(cyl1, cyl2, cyl3, edge_length a) :=
sorry

end cylinders_cannot_move_in_cube_l243_243512


namespace quadratic_even_coeff_l243_243200

theorem quadratic_even_coeff (a b c : ℤ) (h₁ : a ≠ 0) (h₂ : ∃ r s : ℚ, r * s + b * r + c = 0) : (a % 2 = 0) ∨ (b % 2 = 0) ∨ (c % 2 = 0) := by
  sorry

end quadratic_even_coeff_l243_243200


namespace seth_oranges_ratio_l243_243527

theorem seth_oranges_ratio
  (total_boxes : ℕ)
  (boxes_given_to_mother : ℕ)
  (boxes_left : ℕ)
  (boxes_given_away : ℕ)
  (initial_boxes_left : ℕ) :
  total_boxes = 9 →
  boxes_given_to_mother = 1 →
  boxes_left = 4 →
  initial_boxes_left = total_boxes - boxes_given_to_mother →
  boxes_given_away = initial_boxes_left - boxes_left →
  boxes_given_away * 2 = initial_boxes_left := 
begin
  sorry
end

end seth_oranges_ratio_l243_243527


namespace points_on_circle_l243_243993

open Real

theorem points_on_circle (t : ℝ) : 
  ∃ x y, x = cos t + 1 ∧ y = sin t + 1 ∧ (x - 1) ^ 2 + (y - 1) ^ 2 = 1 :=
by
  use cos t + 1
  use sin t + 1
  split
  . rfl
  split
  . rfl
  calc
    (cos t + 1 - 1) ^ 2 + (sin t + 1 - 1) ^ 2
      = (cos t) ^ 2 + (sin t) ^ 2 : by ring
   ... = 1 : by exact Real.sin_sq_add_cos_sq t

end points_on_circle_l243_243993


namespace Winnie_the_Pooh_honey_consumption_l243_243211

theorem Winnie_the_Pooh_honey_consumption (W0 W1 W2 W3 W4 : ℝ) (pot_empty : ℝ) 
  (h1 : W1 = W0 / 2)
  (h2 : W2 = W1 / 2)
  (h3 : W3 = W2 / 2)
  (h4 : W4 = W3 / 2)
  (h5 : W4 = 200)
  (h6 : pot_empty = 200) : 
  W0 - 200 = 3000 := by
  sorry

end Winnie_the_Pooh_honey_consumption_l243_243211


namespace distance_inequality_l243_243472

theorem distance_inequality (E : Finset (ℝ × ℝ)) (h_card : E.card ≥ 2)
  (D : ℝ) (d : ℝ)
  (hD : ∀ (p q ∈ E), p ≠ q → D ≥ dist p q)
  (hd : ∀ (p q ∈ E), p ≠ q → d ≤ dist p q) :
  D ≥ (sqrt 3 / 2) * (sqrt (E.card) - 1) * d := by
  sorry

end distance_inequality_l243_243472


namespace ferry_transport_possible_l243_243849

/-- Define the problem state and initial conditions for the transport task --/
inductive Objects
  | Wolf
  | Goat
  | Cabbage
deriving DecidableEq

def is_safe (left: List Objects) (right: List Objects) :=
  (¬((Objects.Goat ∈ left ∧ Objects.Cabbage ∈ left) ∨ (Objects.Goat ∈ right ∧ Objects.Cabbage ∈ right)) ∧
  ¬((Objects.Goat ∈ left ∧ Objects.Wolf ∈ left) ∨ (Objects.Goat ∈ right ∧ Objects.Wolf ∈ right)))

structure State :=
  (left: List Objects)
  (right: List Objects)
  (ferryman: bool) -- true if the ferryman is on the left bank, false if on the right bank

def initial_state: State :=
  { left := [Objects.Wolf, Objects.Goat, Objects.Cabbage], right := [], ferryman := true }

noncomputable def is_valid_transition (s1 s2: State): Prop :=
  if s1.ferryman then
    (s1.left = s2.left ∪ [Objects.Goat] → s1.right = s2.right ∪ [Objects.Goat] ∧ is_safe s2.left s2.right) ∨ 
    (s1.left = s2.left ∪ [Objects.Cabbage] → s1.right = s2.right ∪ [Objects.Cabbage] ∧ is_safe s2.left s2.right) ∨ 
    (s1.left = s2.left ∪ [Objects.Wolf] → s1.right = s2.right ∪ [Objects.Wolf] ∧ is_safe s2.left s2.right) ∨ 
    (s1.left = s2.left ∧ s1.right = s2.right ∧ is_safe s2.left s2.right)
  else
    (s1.left = s2.left ∧ s1.right = s2.right ∪ [Objects.Goat] ∧ is_safe s2.left s2.right) ∨
    (s1.left = s2.left ∧ s1.right = s2.right ∪ [Objects.Cabbage] ∧ is_safe s2.left s2.right) ∨
    (s1.left = s2.left ∧ s1.right = s2.right ∪ [Objects.Wolf] ∧ is_safe s2.left s2.right) ∨
    (s1.left = s2.left ∧ s1.right = s2.right ∧ is_safe s2.left s2.right)

noncomputable def is_safe_transport_sequence (seq: List State): Prop :=
  seq.head = initial_state ∧ 
  seq.last.left = [] ∧ 
  seq.last.right = [Objects.Wolf, Objects.Goat, Objects.Cabbage] ∧
  ∀ i, i < seq.length - 1 → is_valid_transition (seq.get! i) (seq.get! (i+1))

theorem ferry_transport_possible : ∃ seq: List State, is_safe_transport_sequence seq := by
  sorry

end ferry_transport_possible_l243_243849


namespace part_one_part_two_l243_243575

noncomputable def curve (α : ℝ) : ℝ × ℝ :=
  (√3 * Real.cos α, Real.sin α)

noncomputable def line (x y : ℝ) : ℝ :=
  x - y - 6

-- Point P and maximum distance problem
theorem part_one :
  ∃ (α : ℝ), ∃ (P : ℝ × ℝ), P = curve α ∧ ∀ (Q : ℝ × ℝ), Q = curve α → 
    (let d := |(√3 * Real.cos α - Real.sin α - 6) / (√2)| in d ≤ 4*√2) ∧ 
    (√2 * ((√3 * Real.cos (5*π/6) - Real.sin (5*π/6)) - 6) / 2 = 4 * √2) :=
by sorry

-- Product of distances problem
theorem part_two :
  let M := (-1, 0) in
  ∃ (t1 t2 : ℝ), 
    let line1 := (x := -1 + (√2/2) * t, y := (√2/2) * t) in
    let A, B : ℝ × ℝ := curve t1, curve t2 in
    let d1 := (M.1 - A.1)^2 + (M.2 - A.2)^2 in
    let d2 := (M.1 - B.1)^2 + (M.2 - B.2)^2 in
    d1 * d2 = 2 :=
by sorry

end part_one_part_two_l243_243575


namespace smallest_four_digit_multiple_of_53_l243_243761

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243761


namespace limping_rook_adjacent_sum_not_divisible_by_4_l243_243809

/-- Problem statement: A limping rook traversed a 10 × 10 board,
visiting each square exactly once with numbers 1 through 100
written in the order visited.
Prove that the sum of the numbers in any two adjacent cells
is not divisible by 4. -/
theorem limping_rook_adjacent_sum_not_divisible_by_4 :
  ∀ (board : Fin 10 → Fin 10 → ℕ), 
  (∀ (i j : Fin 10), 1 ≤ board i j ∧ board i j ≤ 100) →
  (∀ (i j : Fin 10), (∃ (i' : Fin 10), i = i' + 1 ∨ i = i' - 1)
                 ∨ (∃ (j' : Fin 10), j = j' + 1 ∨ j = j' - 1)) →
  ((∀ (i j : Fin 10) (k l : Fin 10),
      (i = k ∧ (j = l + 1 ∨ j = l - 1) ∨ j = l ∧ (i = k + 1 ∨ i = k - 1)) →
      (board i j + board k l) % 4 ≠ 0)) :=
by
  sorry

end limping_rook_adjacent_sum_not_divisible_by_4_l243_243809


namespace number_of_marbles_l243_243870

noncomputable def W : ℝ := 0.20 * M
noncomputable def B : ℝ := 0.30 * M
noncomputable def C : ℝ := 0.50 * M

theorem number_of_marbles (M : ℝ) (h : 0.05 * W + 0.10 * B + 0.20 * C = 14) : M = 100 :=
by
  calc M = 100 : 
  sorry

end number_of_marbles_l243_243870


namespace possible_integer_roots_l243_243247

theorem possible_integer_roots (x : ℤ) :
  x^3 + 3 * x^2 - 4 * x - 13 = 0 →
  x = 1 ∨ x = -1 ∨ x = 13 ∨ x = -13 :=
by sorry

end possible_integer_roots_l243_243247


namespace sum_general_conclusion_l243_243348

theorem sum_general_conclusion (n : ℕ) (h : 0 < n) : 
  ∑ k in finset.range n, (k+3)/((k+1)*(k+2))* (1/(2^k)) = 1 - (1/((n+1)*2^n)) := 
  sorry

end sum_general_conclusion_l243_243348


namespace intersection_of_lines_at_single_point_l243_243135

noncomputable def parallelogram (A B C D : Point) : Prop := 
  ∃ v₁ v₂ : Vector, parallelogramLaw (A, B) (B, C) (C, D) (D, A) v₁ v₂

variables (A B C D K L M : Point)
  (r : ℝ) -- ratio for the division
  (hK : segmentDivide AB K r)
  (hL : segmentDivide BC L r)
  (hM : segmentDivide CD M r)
  (b : Line) (c : Line) (d : Line)
  (h₁ : linePassingThrough B b ∧ parallelTo b KL)
  (h₂ : linePassingThrough C c ∧ parallelTo c KM)
  (h₃ : linePassingThrough D d ∧ parallelTo d ML)

theorem intersection_of_lines_at_single_point : 
  ∃ P : Point, intersectsAtSinglePoint P b c d :=
begin
  sorry
end

end intersection_of_lines_at_single_point_l243_243135


namespace equal_segments_in_bisector_triangle_l243_243004

theorem equal_segments_in_bisector_triangle
  (X Y Z P Q W : Type)
  (h_triangle : is_triangle X Y Z)
  (h_length : |XY| > |XZ|)
  (h_parallel1 : is_parallel (line_through Y parallel_to XZ))
  (h_parallel2 : is_parallel (line_through Z parallel_to XY))
  (h_point1 : meets_at_external_bisector_of_angle X Y Z P)
  (h_point2 : meets_at_external_bisector_of_angle Z X Y Q)
  (h_point3 : W_on_segment XZ)
  (h_segment : |WZ| = |XY|) :
  |PW| = |WQ| :=
sorry

end equal_segments_in_bisector_triangle_l243_243004


namespace square_three_points_distance_l243_243118

-- Define the square S with side length 1 and its interior
def S : set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)^(1/2)

-- Define the property to be proved
theorem square_three_points_distance :
  ∀ (P1 P2 P3 : ℝ × ℝ),
  P1 ∈ S → P2 ∈ S → P3 ∈ S →
  ∃ (Q1 Q2 : ℝ × ℝ), Q1 ≠ Q2 ∧ (Q1 = P1 ∨ Q1 = P2 ∨ Q1 = P3) ∧ (Q2 = P1 ∨ Q2 = P2 ∨ Q2 = P3) ∧
  distance Q1 Q2 ≤ real.sqrt 6 - real.sqrt 2 := 
by
  sorry

end square_three_points_distance_l243_243118


namespace smallest_four_digit_divisible_by_53_l243_243681

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l243_243681


namespace sum_extreme_values_l243_243117

theorem sum_extreme_values (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 9) :
  let m := 1 in   -- the smallest value
  let M := (7/3) in -- the largest value
  m + M = 10 / 3 := 
sorry

end sum_extreme_values_l243_243117


namespace isotope_symbol_l243_243341

-- Define the variables
variables {X : Type} {n m y : ℕ}

-- Define the conditions as premises
def XCl_ionic_compound (XCl : X × ℕ) : Prop := 
  ∃ (x : X) (n : ℕ), XCl = (x, n)

-- Define the main theorem statement
theorem isotope_symbol (XCl : X × ℕ) (h1 : XCl_ionic_compound XCl) (h2 : ∃ m' : ℕ, m = m') (h3 : ∃ y' : ℕ, y = y') : 
  symbol_of_isotope X = (y + n, m + y + n) :=
sorry

end isotope_symbol_l243_243341


namespace pie_chart_shows_percentage_l243_243255

-- Define the different types of graphs
inductive GraphType
| PieChart
| BarGraph
| LineGraph
| Histogram

-- Define conditions of the problem
def shows_percentage_of_whole (g : GraphType) : Prop :=
  g = GraphType.PieChart

def displays_with_rectangular_bars (g : GraphType) : Prop :=
  g = GraphType.BarGraph

def shows_trends (g : GraphType) : Prop :=
  g = GraphType.LineGraph

def shows_frequency_distribution (g : GraphType) : Prop :=
  g = GraphType.Histogram

-- We need to prove that a pie chart satisfies the condition of showing percentages of parts in a whole
theorem pie_chart_shows_percentage : shows_percentage_of_whole GraphType.PieChart :=
  by
    -- Proof is skipped
    sorry

end pie_chart_shows_percentage_l243_243255


namespace line_AB_conditions_point_M_existence_l243_243319

-- Define the geometric conditions and given points
def fixed_point_C : ℝ × ℝ := (-1, 0)

def ellipse_eq (x y : ℝ) : Prop := x^2 + 3 * y^2 = 5

-- Condition: Midpoint of AB has x-coordinate -1/2
def midpoint_x_condition (A B : ℝ × ℝ) : Prop := (A.1 + B.1) / 2 = -1 / 2

-- Line passing through C and intersecting ellipse at A and B
def line_through_C (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

-- Sub-problem I: Prove the equations of line AB
theorem line_AB_conditions : 
  ∀ (k : ℝ),
  (∀ (x y : ℝ), line_through_C k x = y → ellipse_eq x y → 
  ∃ (A B : ℝ × ℝ), midpoint_x_condition A B → 
  (line_through_C k x = k * (x + 1) ∧ 
  ((x - sqrt(3) * y + 1 = 0) ∨ (x + sqrt(3) * y + 1 = 0))) :=
sorry

-- Sub-problem II: Prove existence of point M on the x-axis
theorem point_M_existence : 
  ∃ (M : ℝ × ℝ), 
  M = (-7 / 3, 0) ∧
  (∀ (A B : ℝ × ℝ), line_through_C k x = y →
  ellipse_eq x y →
  ∃ (k : ℝ), ∀ m : ℝ,
  ((A.1 - m) * (B.1 - m) + A.2 * B.2 = const) → 
  ∃ const : ℝ, const = 4 / 9) :=
sorry

end line_AB_conditions_point_M_existence_l243_243319


namespace liquor_and_beer_cost_l243_243886

-- Define the variables and conditions
variables (p_liquor p_beer : ℕ)

-- Main theorem to prove
theorem liquor_and_beer_cost (h1 : 2 * p_liquor + 12 * p_beer = 56)
                             (h2 : p_liquor = 8 * p_beer) :
  p_liquor + p_beer = 18 :=
sorry

end liquor_and_beer_cost_l243_243886


namespace smallest_four_digit_divisible_by_53_l243_243672

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243672


namespace smallest_four_digit_divisible_by_53_l243_243674

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243674


namespace ceil_sqrt_225_l243_243953

theorem ceil_sqrt_225 : ⌈real.sqrt 225⌉ = 15 :=
by
  have h : real.sqrt 225 = 15 := by
    sorry
  rw [h]
  exact int.ceil_eq_self.mpr rfl

end ceil_sqrt_225_l243_243953


namespace value_of_f3_l243_243028

def f (x : ℝ) (a b : ℝ) : ℝ := log (x + sqrt (x^2 + 1)) + a * x^7 + b * x^3 - 4

theorem value_of_f3 (a b : ℝ) (h : f (-3) a b = 4) : f 3 a b = -12 :=
by
  sorry

end value_of_f3_l243_243028


namespace least_possible_length_of_third_side_l243_243386

theorem least_possible_length_of_third_side (a b : ℕ) (h1 : a = 8) (h2 : b = 15) : 
  ∃ c : ℕ, c = 17 ∧ a^2 + b^2 = c^2 := 
by
  use 17 
  split
  · rfl
  · rw [h1, h2]
    norm_num

end least_possible_length_of_third_side_l243_243386


namespace smallest_four_digit_multiple_of_53_l243_243766

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243766


namespace jill_tax_on_other_items_l243_243507

noncomputable def tax_on_other_items (total_spent clothing_tax_percent total_tax_percent : ℝ) : ℝ :=
  let clothing_spent := 0.5 * total_spent
  let food_spent := 0.25 * total_spent
  let other_spent := 0.25 * total_spent
  let clothing_tax := clothing_tax_percent * clothing_spent
  let total_tax := total_tax_percent * total_spent
  let tax_on_others := total_tax - clothing_tax
  (tax_on_others / other_spent) * 100

theorem jill_tax_on_other_items :
  let total_spent := 100
  let clothing_tax_percent := 0.1
  let total_tax_percent := 0.1
  tax_on_other_items total_spent clothing_tax_percent total_tax_percent = 20 := by
  sorry

end jill_tax_on_other_items_l243_243507


namespace square_1023_l243_243904

theorem square_1023 : (1023 : ℤ)^2 = 1046529 :=
by
  let a := (10 : ℤ)^3
  let b := (23 : ℤ)
  have h1 : (1023 : ℤ) = a + b := by rfl
  have h2 : (a + b)^2 = a^2 + 2 * a * b + b^2 := by ring
  have h3 : a = 1000 := by rfl
  have h4 : b = 23 := by rfl
  have h5 : a^2 = 1000000 := by norm_num
  have h6 : 2 * a * b = 46000 := by norm_num
  have h7 : b^2 = 529 := by norm_num
  calc
    (1023 : ℤ)^2 = (a + b)^2 : by rw h1
    ... = a^2 + 2 * a * b + b^2 : by rw h2
    ... = 1000000 + 46000 + 529 : by rw [h5, h6, h7]
    ... = 1046529 : by norm_num

end square_1023_l243_243904


namespace find_fractions_l243_243550

-- Define the numerators and denominators
def p1 := 75
def p2 := 70
def q1 := 34
def q2 := 51

-- Define the fractions
def frac1 := p1 / q1
def frac2 := p1 / q2

-- Define the greatest common divisor (gcd) condition
def gcd_condition := Nat.gcd p1 p2 = p1 - p2

-- Define the least common multiple (lcm) condition
def lcm_condition := Nat.lcm p1 p2 = 1050

-- Define the difference condition
def difference_condition := (frac1 - frac2) = (5 / 6)

-- Lean proof statement
theorem find_fractions :
  gcd_condition ∧ lcm_condition ∧ difference_condition :=
by
  sorry

end find_fractions_l243_243550


namespace ceil_sqrt_225_eq_15_l243_243973

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := by
  sorry

end ceil_sqrt_225_eq_15_l243_243973


namespace negation_of_existence_l243_243563

theorem negation_of_existence (P : ∃ x : ℝ, x^2 + 1 < 2 * x) :
  ¬ P ↔ ∀ x : ℝ, x^2 + 1 ≥ 2 * x :=
by
  sorry

end negation_of_existence_l243_243563


namespace find_u_l243_243423

-- Definitions for the given conditions
variables (a b u : ℝ)

-- Given conditions
def is_isosceles (A B C : ℝ) : Prop := A = B
def quadratic_has_real_roots (a b : ℝ) : Prop :=
  let discriminant := b * b - 4 * a * a in
  discriminant ≥ 0 ∧ discriminant = 2 * a^2
def quadratic_roots_difference (a b : ℝ) : Prop :=
  let discriminant := b * b - 4 * a * a in
  sqrt discriminant = sqrt 2

def cosine_rule_isosceles (a b u : ℝ) : Prop :=
  b^2 = a^2 + a^2 - 2 * a * a * cos (u * pi / 180)

-- The main theorem
theorem find_u (h1 : is_isosceles a b a)
              (h2 : is_isosceles a a b)
              (h3 : quadratic_has_real_roots a b)
              (h4 : quadratic_roots_difference a b)
              (h5 : cosine_rule_isosceles a b u) :
              u = 120 :=
begin
  sorry
end

end find_u_l243_243423


namespace quadratic_solutions_l243_243345

theorem quadratic_solutions (y : ℝ) (b c : ℝ) :
  b = -8 → c = 15 → (∃ z : ℝ, z = y^2 + 4 ∧ (z^2 + b * z + c = 0)) → y = -1 ∨ y = 1 :=
by {
  intros hb hc hz,
  sorry
}

end quadratic_solutions_l243_243345


namespace f_odd_f_correct_f_max_min_l243_243240

noncomputable def f : ℝ → ℝ :=
  λ x, if x < 0 then (1 / 4) ^ x - 8 * (1 / 2) ^ x - 1
       else if x = 0 then 0
       else -4 ^ x + 8 * 2 ^ x + 1

theorem f_odd (x : ℝ) : f x + f (-x) = 0 :=
  sorry

theorem f_correct (x : ℝ) :
  f x =
  if x < 0 then (1 / 4) ^ x - 8 * (1 / 2) ^ x - 1
  else if x = 0 then 0
  else -4 ^ x + 8 * 2 ^ x + 1 :=
  sorry

theorem f_max_min : ∃ max min : ℝ, max = 17 ∧ min = 1 ∧
  ∀ x ∈ Icc 1 3, min ≤ f x ∧ f x ≤ max :=
  sorry

end f_odd_f_correct_f_max_min_l243_243240


namespace line_intersection_l243_243852

-- Parameters for the first line
def line1_param (s : ℝ) : ℝ × ℝ := (1 - 2 * s, 4 + 3 * s)

-- Parameters for the second line
def line2_param (v : ℝ) : ℝ × ℝ := (-v, 5 + 6 * v)

-- Statement of the intersection point
theorem line_intersection :
  ∃ (s v : ℝ), line1_param s = (-1 / 9, 17 / 3) ∧ line2_param v = (-1 / 9, 17 / 3) :=
by
  -- Placeholder for the proof, which we are not providing as per instructions
  sorry

end line_intersection_l243_243852


namespace smallest_four_digit_divisible_by_53_l243_243658

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l243_243658


namespace smallest_four_digit_divisible_by_53_l243_243747

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243747


namespace product_of_divisors_36_l243_243299

theorem product_of_divisors_36 : ∏ d in (finset.divisors 36), d = 10077696 := by
  sorry

end product_of_divisors_36_l243_243299


namespace ceil_sqrt_225_l243_243947

theorem ceil_sqrt_225 : ⌈real.sqrt 225⌉ = 15 :=
by
  have h : real.sqrt 225 = 15 := by
    sorry
  rw [h]
  exact int.ceil_eq_self.mpr rfl

end ceil_sqrt_225_l243_243947


namespace find_S8_l243_243010

-- Definitions and conditions
variable (S : ℕ → ℕ) (a : ℕ → ℕ)
axiom S_recurrence : ∀ n, S (n + 1) = S n + a n + 3
axiom a_4_5_sum : a 4 + a 5 = 23

-- Goal
theorem find_S8 : S 8 = 92 :=
begin
  sorry
end

end find_S8_l243_243010


namespace Ariel_current_age_l243_243876

-- Define the conditions
def Ariel_birth_year : Nat := 1992
def Ariel_start_fencing_year : Nat := 2006
def Ariel_fencing_years : Nat := 16

-- Define the problem as a theorem
theorem Ariel_current_age :
  (Ariel_start_fencing_year - Ariel_birth_year) + Ariel_fencing_years = 30 := by
sorry

end Ariel_current_age_l243_243876


namespace Billie_has_2_caps_l243_243145

-- Conditions as definitions in Lean
def Sammy_caps : ℕ := 8
def Janine_caps : ℕ := Sammy_caps - 2
def Billie_caps : ℕ := Janine_caps / 3

-- Problem statement to prove
theorem Billie_has_2_caps : Billie_caps = 2 := by
  sorry

end Billie_has_2_caps_l243_243145


namespace bicycle_cost_multiple_l243_243233

theorem bicycle_cost_multiple :
  ∃ (m : ℕ), let H := 40 in let B := m * H in B + H = 240 ∧ m = 5 :=
by
  sorry

end bicycle_cost_multiple_l243_243233


namespace shortest_path_l243_243578

variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]
variables {a b c : A} {d : real}
variable {alpha : real}
variables {v w : real} (hv : 0 < v) (hw : 0 < w)

def equidistant (c a b : A) [metric_space A] := dist c a = dist c b

theorem shortest_path {A : Type} [metric_space A]
  (a b c : A) (v w alpha : real) (hv : 0 < v) (hw : 0 < w) (h_w_lt_v : w < v) (h_equidistant: equidistant c a b) :
  (if w > v * cos alpha then dist a b
   else if w < v * cos alpha then dist a c + dist c b
   else exists e f : A, dist a e + dist f b + dist e f = dist a b)
  := sorry

end shortest_path_l243_243578


namespace number_of_2_dollar_socks_l243_243365

theorem number_of_2_dollar_socks :
  ∃ (a b c : ℕ), (a + b + c = 15) ∧ (2 * a + 3 * b + 5 * c = 40) ∧ (a ≥ 1) ∧ (b ≥ 1) ∧ (c ≥ 1) ∧ (a = 7 ∨ a = 9 ∨ a = 11) :=
by {
  -- The details of the proof will go here, but we skip it for our requirements
  sorry
}

end number_of_2_dollar_socks_l243_243365


namespace smallest_four_digit_divisible_by_53_l243_243647

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l243_243647


namespace ceil_sqrt_225_l243_243937

theorem ceil_sqrt_225 : Nat.ceil (Real.sqrt 225) = 15 :=
by
  sorry

end ceil_sqrt_225_l243_243937


namespace trajectory_of_Q_l243_243347

-- Define the conditions
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

def C2 (x y : ℝ) : Prop := x + y = 1

def P (x y : ℝ) : Prop := C2 x y

def R (x y : ℝ) (t : ℝ) (a : ℝ) : Prop := (x, y) = (t * a, t * (1 - a)) ∧ C1 (t * a) (t * (1 - a))

def Q (x y : ℝ) (t : ℝ) (a : ℝ) (q : ℝ) : Prop :=
  q * real.sqrt (2 * a^2 - 2 * a + 1) = t^2 ∧ (x, y) = (q * a, q * (1 - a))

-- The final theorem to prove the trajectory
theorem trajectory_of_Q :
  ∀ (a : ℝ) (t q x y : ℝ),
  (P a (1 - a)) →
  (R (t * a) (t * (1 - a)) t a) →
  (Q x y t a q) →
  (x - 1/2)^2 + (y - 1/2)^2 = 1/2 :=
by sorry

end trajectory_of_Q_l243_243347


namespace max_flags_l243_243431

theorem max_flags (n : ℕ) (h1 : ∀ k, n = 9 * k) (h2 : n ≤ 200)
  (h3 : ∃ m, n = 9 * m + k ∧ k ≤ 2 ∧ k + 1 ≠ 0 ∧ k - 2 ≠ 0) : n = 198 :=
by {
  sorry
}

end max_flags_l243_243431


namespace net_profit_correct_l243_243524

-- Define the conditions
def unit_price : ℝ := 1.25
def selling_price : ℝ := 12
def num_patches : ℕ := 100

-- Define the required total cost
def total_cost : ℝ := num_patches * unit_price

-- Define the required total revenue
def total_revenue : ℝ := num_patches * selling_price

-- Define the net profit calculation
def net_profit : ℝ := total_revenue - total_cost

-- The theorem we need to prove
theorem net_profit_correct : net_profit = 1075 := by
    sorry

end net_profit_correct_l243_243524


namespace num_ways_arith_prog_l243_243440

theorem num_ways_arith_prog : 
  ∑ (d : ℕ) in finset.range 334, 1000 - 3 * d = 166167 :=
by
  sorry

end num_ways_arith_prog_l243_243440


namespace square_1023_l243_243902

theorem square_1023 : (1023 : ℤ)^2 = 1046529 :=
by
  let a := (10 : ℤ)^3
  let b := (23 : ℤ)
  have h1 : (1023 : ℤ) = a + b := by rfl
  have h2 : (a + b)^2 = a^2 + 2 * a * b + b^2 := by ring
  have h3 : a = 1000 := by rfl
  have h4 : b = 23 := by rfl
  have h5 : a^2 = 1000000 := by norm_num
  have h6 : 2 * a * b = 46000 := by norm_num
  have h7 : b^2 = 529 := by norm_num
  calc
    (1023 : ℤ)^2 = (a + b)^2 : by rw h1
    ... = a^2 + 2 * a * b + b^2 : by rw h2
    ... = 1000000 + 46000 + 529 : by rw [h5, h6, h7]
    ... = 1046529 : by norm_num

end square_1023_l243_243902


namespace find_lloyd_normal_work_hours_l243_243130

-- Define the conditions and the target statement.
def lloyd_normal_work_hours : Prop :=
  ∃ (h : ℝ),
    (∀ (r : ℝ), r = 4 → 
      let excess_hours := 10.5 - h in
      let earnings_per_extra_hour := 1.5 * r in
      let total_earnings := (h * r) + (excess_hours * earnings_per_extra_hour) in
      total_earnings = 48) → h = 7.5

-- Auxilliary declaration to invoke the theorem
theorem find_lloyd_normal_work_hours : lloyd_normal_work_hours :=
sorry

end find_lloyd_normal_work_hours_l243_243130


namespace volume_of_PABCD_is_80_l243_243197

-- Conditions and definitions
structure Trapezoid where
  A B C D P : Type
  AB CD : ℝ
  parallel_AB_CD : AB ∥ CD
  AB_length : AB = 5
  CD_length : CD = 10
  BC_length : BC = 4
  PA_height : ℝ
  PA_perpendicular_AB_AD : PA_height = 8
  PA_perpendicular_AB : ∀ A B, PA_height ⊥ AB
  PA_perpendicular_AD : ∀ A D, PA_height ⊥ AD

-- Volume of the pyramid
def volume_of_pyramid (trapezoid : Trapezoid) : ℝ :=
  (1 / 3) * (1 / 2 * (trapezoid.AB + trapezoid.CD) * trapezoid.BC_length ) * trapezoid.PA_height

-- Theorem stating the volume
theorem volume_of_PABCD_is_80 (trapezoid : Trapezoid) (h : trapezoid.AB_length ∧ trapezoid.CD_length ∧ trapezoid.BC_length ∧ trapezoid.PA_perpendicular_AB_AD) : 
  volume_of_pyramid trapezoid = 80 := 
sorry

end volume_of_PABCD_is_80_l243_243197


namespace radius_of_circumcircle_l243_243586

variables (r1 r2 : ℝ) (A B : euclidean_geometry.Point ℝ) (C : euclidean_geometry.Point ℝ) (r : ℝ)

-- Conditions of the problem
def conditions : Prop :=
  (euclidean_geometry.dist A B = 6 * real.sqrt 10) ∧
  r1 + r2 = 11 ∧
  euclidean_geometry.dist (euclidean_geometry.Point.mk 0 0 r1) 
                          (euclidean_geometry.Point.mk (6 * real.sqrt 10) 0 r2) = real.sqrt 481 ∧
  euclidean_geometry.dist C A = r + r1 ∧
  euclidean_geometry.dist C B = r + r2

-- Question (proof goal)
theorem radius_of_circumcircle (h : conditions r1 r2 A B C 9) : 
  let R := 3 * real.sqrt 10 in R = (euclidean_geometry.circumradius_of_triangle A B C) :=
by { sorry }

end radius_of_circumcircle_l243_243586


namespace smallest_four_digit_divisible_by_53_l243_243609

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l243_243609


namespace smallest_four_digit_divisible_by_53_l243_243649

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l243_243649


namespace smallest_four_digit_divisible_by_53_l243_243735

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243735


namespace jaden_toy_cars_left_l243_243095

-- Definitions for each condition
def initial_toys : ℕ := 14
def purchased_toys : ℕ := 28
def birthday_toys : ℕ := 12
def given_to_sister : ℕ := 8
def given_to_vinnie : ℕ := 3
def traded_lost : ℕ := 5
def traded_received : ℕ := 7

-- The final number of toy cars Jaden has
def final_toys : ℕ :=
  initial_toys + purchased_toys + birthday_toys - given_to_sister - given_to_vinnie + (traded_received - traded_lost)

theorem jaden_toy_cars_left : final_toys = 45 :=
by
  -- The proof will be filled in here 
  sorry

end jaden_toy_cars_left_l243_243095


namespace tangent_to_x_axis_at_origin_range_of_a_l243_243032

variables (f : ℝ → ℝ) (a : ℝ)

-- Definition of the function f
def f (x : ℝ) : ℝ := exp x - a * x^2 - cos x - log (x + 1)

-- Part 1: Showing that f is tangent to x-axis at origin when a = 1
theorem tangent_to_x_axis_at_origin (h : a = 1) : f 0 = 0 ∧ deriv f 0 = 0 :=
by
  intros
  sorry

-- Part 2: Finding the range of a
theorem range_of_a (h : ∀ I ∈ {[ (-1, 0); (0, +∞) ]}, ∃! x ∈ I, f' x = 0) : (3 / 2 : ℝ) < a :=
by
  intros
  sorry

end tangent_to_x_axis_at_origin_range_of_a_l243_243032


namespace line_through_two_points_l243_243552

theorem line_through_two_points :
  ∃ (m b : ℝ), (∀ x y : ℝ, (x, y) = (-2, 4) ∨ (x, y) = (-1, 3) → y = m * x + b) ∧ b = 2 ∧ m = -1 :=
by
  sorry

end line_through_two_points_l243_243552


namespace performance_distribution_l243_243362

theorem performance_distribution : 
  (∃ x y z : ℕ, x + y + z = 14 ∧ x ≥ 3 ∧ y ≥ 3 ∧ z ≥ 3) → 
  (finset.card {p : ℕ × ℕ × ℕ // p.1 + p.2 + p.3 = 14 ∧ p.1 ≥ 3 ∧ p.2 ≥ 3 ∧ p.3 ≥ 3} = 21) :=
by
  sorry

end performance_distribution_l243_243362


namespace solve_missing_number_l243_243533

theorem solve_missing_number (n : ℤ) (h : 121 * n = 75625) : n = 625 :=
sorry

end solve_missing_number_l243_243533


namespace candy_cost_55_cents_l243_243143

theorem candy_cost_55_cents
  (paid: ℕ) (change: ℕ) (num_coins: ℕ)
  (coin1 coin2 coin3 coin4: ℕ)
  (h1: paid = 100)
  (h2: num_coins = 4)
  (h3: coin1 = 25)
  (h4: coin2 = 10)
  (h5: coin3 = 10)
  (h6: coin4 = 0)
  (h7: change = coin1 + coin2 + coin3 + coin4) :
  paid - change = 55 :=
by
  -- The proof can be provided here.
  sorry

end candy_cost_55_cents_l243_243143


namespace value_of_r_for_n_3_l243_243486

theorem value_of_r_for_n_3 :
  ∀ (r s : ℕ), 
  (r = 4^s + 3 * s) → 
  (s = 2^3 + 2) → 
  r = 1048606 :=
by
  intros r s h1 h2
  sorry

end value_of_r_for_n_3_l243_243486


namespace smallest_four_digit_multiple_of_53_l243_243714

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243714


namespace probability_two_heads_two_tails_l243_243374

theorem probability_two_heads_two_tails :
  (prob_of_two_heads_two_tails : ℚ) =
  (combinations_of_4_choose_2 * prob_of_each_sequence : ℚ) :=
by
  let prob_of_each_sequence : ℚ := (1/2)^4
  let combinations_of_4_choose_2 : ℚ := 6 -- This comes from combinatorial calculation C(4,2) = 6
  have prob_of_two_heads_two_tails := combinations_of_4_choose_2 * prob_of_each_sequence
  show prob_of_two_heads_two_tails = (3/8 : ℚ)
  sorry

end probability_two_heads_two_tails_l243_243374


namespace smallest_four_digit_divisible_by_53_l243_243759

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243759


namespace Ariel_age_l243_243878

theorem Ariel_age :
  ∀ (fencing_start_year birth_year: ℕ) (fencing_years: ℕ),
    fencing_start_year = 2006 →
    birth_year = 1992 →
    fencing_years = 16 →
    (fencing_start_year + fencing_years - birth_year) = 30 :=
by
  intros fencing_start_year birth_year fencing_years h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end Ariel_age_l243_243878


namespace equation_of_circle_with_diameter_OC_correct_l243_243023

-- Define the given conditions
def center_of_circle_C : ℝ × ℝ := (6, 8)
def origin_O : ℝ × ℝ := (0, 0)

-- Define the equation of the circle C
def equation_of_circle_C (x y : ℝ) : Prop := (x - 6)^2 + (y - 8)^2 = 4

-- Define the midpoint of diameter OC
def midpoint_E (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the circle with diameter OC
def equation_of_circle_with_diameter_OC (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 25

-- The main theorem statement in Lean 4
theorem equation_of_circle_with_diameter_OC_correct :
  equation_of_circle_C center_of_circle_C.1 center_of_circle_C.2 →
  equation_of_circle_with_diameter_OC 3 4 :=
sorry

end equation_of_circle_with_diameter_OC_correct_l243_243023


namespace math_team_count_l243_243182

open Nat

theorem math_team_count :
  let girls := 7
  let boys := 12
  let total_team := 16
  let count_ways (n k : ℕ) := choose n k
  (count_ways girls 3) * (count_ways boys 5) * (count_ways (girls - 3 + boys - 5) 8) = 456660 :=
by
  sorry

end math_team_count_l243_243182


namespace input_is_input_l243_243256

-- Definitions based on the given conditions
def PRINT_statement : Prop := "PRINT" represents an output statement
def INPUT_statement : Prop := "INPUT" represents an input statement
def IF_statement : Prop := "IF" represents a conditional statement
def END_statement : Prop := "END" represents an end statement

-- The statement to be proved
theorem input_is_input : INPUT_statement := by
  sorry

end input_is_input_l243_243256


namespace min_c_value_l243_243515

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : ∀ x y, 2 * x + y = 2003 → y = |x - a| + |x - b| + |x - c| → x = b) :
  c = 1002 := 
sorry

end min_c_value_l243_243515


namespace sum_of_negative_solutions_l243_243291

theorem sum_of_negative_solutions :
  (∑ a in {a : ℝ | a < 0 ∧
             ∃ solutions : Finset ℝ, 
             (∀ x ∈ solutions, x ∈ Set.Ici π) ∧ 
             (solutions.card = 3) ∧
             (∀ x ∈ solutions, 
                (6 * Real.pi * a - Real.arcsin (Real.sin x) + 
                2 * Real.arccos (Real.cos x) - a * x) / 
                (Real.tan x ^ 2 + 4) = 0)
          }, a)
  = -1.6 := 
sorry

end sum_of_negative_solutions_l243_243291


namespace smallest_four_digit_divisible_by_53_l243_243688

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l243_243688


namespace prove_triangle_ASD_is_right_triangle_l243_243110

noncomputable def triangle_ASD_is_right_triangle (β : ℝ) (h1 : β < 90 ∨ β > 90 ∨ β = 90)
  (h2 : ∠ABC = β)
  (h3 : ∠AOC = 2 * β)
  (h4 : ∠OAC = ∠OCA = 90 - β)
  (h5 : ∠ADL = ∠ABL = β) : Prop :=
∃ (S : Point), is_right_triangle A S D

theorem prove_triangle_ASD_is_right_triangle (β : ℝ) (h1 : β < 90 ∨ β > 90 ∨ β = 90)
  (h2 : ∠ABC = β)
  (h3 : ∠AOC = 2 * β)
  (h4 : ∠OAC = ∠OCA = 90 - β)
  (h5 : ∠ADL = ∠ABL = β) : 
  triangle_ASD_is_right_triangle β h1 h2 h3 h4 h5 :=
sorry

end prove_triangle_ASD_is_right_triangle_l243_243110


namespace simplify_fraction_l243_243530

theorem simplify_fraction : 
  (16777216 = 16 ^ 6) → (Real.sqrt (Real.cbrt (Real.sqrt (1 / 16777216))) = 1 / 4) := 
by
  intro h
  sorry

end simplify_fraction_l243_243530


namespace problem_l243_243114

theorem problem (a b c k : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (hk : k ≠ 0)
  (h1 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (k * (b - c)^2) + b / (k * (c - a)^2) + c / (k * (a - b)^2) = 0 :=
by
  sorry

end problem_l243_243114


namespace square_side_length_l243_243246

variable (s : ℝ)
variable (k : ℝ := 6)

theorem square_side_length :
  s^2 = k * 4 * s → s = 24 :=
by
  intro h
  sorry

end square_side_length_l243_243246


namespace exists_four_letter_list_with_equal_product_except_BEHK_l243_243283

-- Assign a value to each letter of the alphabet
def letter_value : Char → ℕ
| 'A' := 1
| 'B' := 2
| 'C' := 3
| -- ... similar assignments for all letters
| 'Z' := 26
| _ := 0 -- Default case for input validation

-- Calculate the product of values for a list of characters
def product_values (l : List Char) : ℕ :=
  l.foldl (λ acc c => acc * letter_value c) 1

-- The given lists
def BEHK : List Char := ['B', 'E', 'H', 'K']
def QRST : List Char := ['Q', 'R', 'S', 'T']

-- Calculate the product of BEHK and QRST
def product_BEHK : ℕ := product_values BEHK
def product_QRST : ℕ := product_values QRST

-- Define the desired property: finding another four-letter list product equal to QRST
def another_four_letter_list : List Char := ['E', 'F', 'Q', 'S']

-- Lean statement: proving such a list exists and matches
theorem exists_four_letter_list_with_equal_product_except_BEHK :
  product_BEHK = 880 ∧ product_values another_four_letter_list = product_QRST :=
  by 
    split
    { -- Proof that product_BEHK = 880, the mathlib function calculates product of BEHK to 880
      sorry 
    } 
    { -- Proof that the product of another_four_letter_list is equal to product_QRST
      sorry 
    }

end exists_four_letter_list_with_equal_product_except_BEHK_l243_243283


namespace floor_equiv_l243_243304

theorem floor_equiv {n : ℤ} (h : n > 2) : 
  Int.floor ((n * (n + 1) : ℚ) / (4 * n - 2 : ℚ)) = Int.floor ((n + 1 : ℚ) / 4) := 
sorry

end floor_equiv_l243_243304


namespace smallest_four_digit_multiple_of_53_l243_243775

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l243_243775


namespace pencil_distribution_l243_243577

theorem pencil_distribution (n k : ℕ) (friends pencils : ℕ) 
  (each_friend_at_least_one : friends = k) 
  (total_pencils : pencils = n) :
  n = 6 ∧ k = 3 ∧ each_friend_at_least_one = 1 -> ∃ (ways : ℕ), ways = 10 :=
by
  sorry

end pencil_distribution_l243_243577


namespace smallest_four_digit_div_by_53_l243_243622

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l243_243622


namespace algebraic_expression_value_l243_243068

theorem algebraic_expression_value (p q : ℝ) 
  (h : p * 3^3 + q * 3 + 1 = 2015) : 
  p * (-3)^3 + q * (-3) + 1 = -2013 :=
by 
  sorry

end algebraic_expression_value_l243_243068


namespace smallest_four_digit_divisible_by_53_l243_243693

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243693


namespace ceil_sqrt_225_eq_15_l243_243926

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := 
by 
  sorry

end ceil_sqrt_225_eq_15_l243_243926


namespace checkerboard_exists_l243_243282

open Set

def board : Type := Σ (i j : Fin 100), Bool
def is_black_border (b : board) : Prop :=
  ∃ ((i j : Fin 100)), b = ⟨i, j, true⟩ ∧ (i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99)

def is_no_monochrome_2x2_square (b : board → Bool) : Prop :=
  ∀ (i j : Fin 99), ¬ (b ⟨i, j⟩ = b ⟨i, j.succ⟩ ∧ b ⟨i.succ, j⟩ = b ⟨i, j.succ⟩ ∧ b ⟨i.succ, j.succ⟩ = b ⟨i, j.succ⟩)

def has_checkerboard_2x2_square (b : board → Bool) : Prop :=
  ∃ (i j : Fin 99), (b ⟨i, j⟩ ≠ b ⟨i, j.succ⟩ ∧ b ⟨i.succ, j⟩ ≠ b ⟨i, j.succ⟩ ∧ b ⟨i.succ, j.succ⟩ ≠ b ⟨i, j.succ⟩)

theorem checkerboard_exists 
  (b : board → Bool)
  (border_black : ∀ b, is_black_border b → b = true)
  (no_monochrome : is_no_monochrome_2x2_square b) :
  has_checkerboard_2x2_square b :=
begin
  sorry
end

end checkerboard_exists_l243_243282


namespace area_conversion_correct_l243_243433

-- Define the legs of the right triangle
def leg1 : ℕ := 60
def leg2 : ℕ := 80

-- Define the conversion factor
def square_feet_in_square_yard : ℕ := 9

-- Calculate the area of the triangle in square feet
def area_in_square_feet : ℕ := (leg1 * leg2) / 2

-- Calculate the area of the triangle in square yards
def area_in_square_yards : ℚ := area_in_square_feet / square_feet_in_square_yard

-- The theorem stating the problem
theorem area_conversion_correct : area_in_square_yards = 266 + 2 / 3 := by
  sorry

end area_conversion_correct_l243_243433


namespace speed_with_stream_l243_243854

-- Define the given conditions
def V_m : ℝ := 7 -- Man's speed in still water (7 km/h)
def V_as : ℝ := 10 -- Man's speed against the stream (10 km/h)

-- Define the stream's speed as the difference
def V_s : ℝ := V_m - V_as

-- Define man's speed with the stream
def V_ws : ℝ := V_m + V_s

-- (Correct Answer): Prove the man's speed with the stream is 10 km/h
theorem speed_with_stream :
  V_ws = 10 := by
  -- Sorry for no proof required in this task
  sorry

end speed_with_stream_l243_243854


namespace sum_of_10_smallest_n_divisible_by_4_is_137_l243_243306

def T_n (n : ℕ) : ℕ :=
  (n * (n - 1) * (n + 1) * (3 * n + 2)) / 24

def is_divisible_by_4 (n : ℕ) : Prop :=
  T_n n % 4 = 0

theorem sum_of_10_smallest_n_divisible_by_4_is_137:
  ∑ i in (Finset.filter is_divisible_by_4 (Finset.range (28 + 1))).take 10 = 137 := by
  sorry

end sum_of_10_smallest_n_divisible_by_4_is_137_l243_243306


namespace smallest_four_digit_divisible_by_53_l243_243738

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243738


namespace total_cats_count_l243_243260

universe u

variables (Cats : Type u) 
  (Jump Fetch Spin : Cats → Prop) 
  (can_jump : ∀ x : Cats, Jump x → Prop)
  (can_fetch : ∀ x : Cats, Fetch x → Prop)
  (can_spin : ∀ x : Cats, Spin x → Prop)

variables (total_cats : ∀ x : Cats, true → Prop)
  (jump_count : (∀ x : Cats, Jump x) → Nat)
  (fetch_count : (∀ x : Cats, Fetch x) → Nat)
  (spin_count : (∀ x : Cats, Spin x) → Nat)
  (jump_fetch_count : (∀ x : Cats, Jump x ∧ Fetch x) → Nat)
  (fetch_spin_count : (∀ x : Cats, Fetch x ∧ Spin x) → Nat)
  (jump_spin_count : (∀ x : Cats, Jump x ∧ Spin x) → Nat)
  (all_three_count : (∀ x : Cats, Jump x ∧ Fetch x ∧ Spin x) → Nat)
  (none_count : (∀ x : Cats, ¬Jump x ∧ ¬Fetch x ∧ ¬Spin x) → Nat)

theorem total_cats_count : 
  jump_count can_jump = 60 →
  fetch_count can_fetch = 35 →
  spin_count can_spin = 40 →
  jump_fetch_count (λ x, Jump x ∧ Fetch x) = 25 →
  fetch_spin_count (λ x, Fetch x ∧ Spin x) = 18 →
  jump_spin_count (λ x, Jump x ∧ Spin x) = 20 →
  all_three_count (λ x, Jump x ∧ Fetch x ∧ Spin x) = 12 →
  none_count (λ x, ¬ Jump x ∧ ¬ Fetch x ∧ ¬ Spin x) = 15 →
  @total_cats _ 99 sorry

end total_cats_count_l243_243260


namespace range_of_x_in_f_l243_243456

def f (x : ℝ) : ℝ := real.sqrt (x + 2)

theorem range_of_x_in_f (x : ℝ) : x ≥ -2 :=
by {
  sorry
}

end range_of_x_in_f_l243_243456


namespace line_passes_through_center_l243_243065

theorem line_passes_through_center (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y = 0 → 3 * x + y + a = 0) → a = 1 :=
by
  sorry

end line_passes_through_center_l243_243065


namespace smallest_four_digit_multiple_of_53_l243_243760

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243760


namespace sequence_count_l243_243564

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| n + 2 => fibonacci n + fibonacci (n + 1)

def a (n : ℕ) : ℕ :=
fibonacci (n + 3) - 2

theorem sequence_count (n : ℕ) : a(n) = ∑ k in Finset.range n, 
  {i : Fin k → ℕ // (∀ r : Fin (k-1), i (r + 1) > i r) ∧ (∀ r : Fin (k-1), (i (r + 1) - i r) % 2 = 1)}.card :=
sorry

end sequence_count_l243_243564


namespace smallest_four_digit_multiple_of_53_l243_243710

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243710


namespace bananas_divisible_by_3_l243_243510

variable (B : ℕ)

theorem bananas_divisible_by_3 (h1 : ∃ O : ℕ, O = 18)
                               (h2 : ∃ P : ℕ, P = 27)
                               (h3 : ∃ N : ℕ, N = 3)
                               (h4 : ∀ O P B, (O % N = 0) ∧ (P % N = 0)) :
  B % 3 = 0 :=
sorry

end bananas_divisible_by_3_l243_243510


namespace white_bread_served_l243_243509

theorem white_bread_served (total_bread : ℝ) (wheat_bread : ℝ) (white_bread : ℝ) 
  (h1 : total_bread = 0.9) (h2 : wheat_bread = 0.5) : white_bread = 0.4 :=
by
  sorry

end white_bread_served_l243_243509


namespace problem_1_problem_2_problem_3_l243_243020

-- Proof Problem 1
theorem problem_1 (n : ℕ) (h : 0 < n) (S : ℕ → ℕ) (hS : ∀ n, S n = n^2) :
    2n - 1 = (S n) - (S (n - 1)) :=
by sorry

-- Proof Problem 2
theorem problem_2 {a : ℕ → ℕ} (f : ℕ → ℕ) (h_f1 : ∀ n, n % 2 = 1 → f n = a n) (h_f2 : ∀ n, n % 2 = 0 → f n = f (n / 2))
  (c : ℕ → ℕ) (h_c : ∀ n, c n = f (2^n + 4)) :
    ∀ n, T n = (if n = 1 then 5 else 2^n + n) :=
by sorry

-- Proof Problem 3
theorem problem_3 (m n k : ℕ) (h : m + n = 3 * k) (h_mn : m ≠ n) :
    max { λ : ℝ // ∀ m n k, m + n = 3 * k ∧ m ≠ n → m^2 + n^2 > λ * k^2 } = 9 / 2 :=
by sorry

end problem_1_problem_2_problem_3_l243_243020


namespace tan_addition_formula_l243_243053

theorem tan_addition_formula (x : ℝ) (h : Real.tan x = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 := 
by 
  sorry

end tan_addition_formula_l243_243053


namespace length_ae_l243_243549

noncomputable def point := ℝ × ℝ

-- Define points A, B, C, D
def A : point := (0, 3)
def B : point := (7, 0)
def C : point := (3, 0)
def D : point := (6, 3)

-- Line equations
def line_eq (p1 p2 : point) : ℝ × ℝ → Prop := 
  λ P => (P.1 - p1.1) * (p2.2 - p1.2) = (P.2 - p1.2) * (p2.1 - p1.1)

-- Intersection of lines
def intersection (L1 L2 : ℝ × ℝ → Prop) : point :=
  Classical.choose (exists_unique_point L1 L2)

-- Distances
def dist (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

-- Problem Statement
theorem length_ae : dist A (intersection (line_eq A B) (line_eq C D)) = Real.sqrt 19.89 := 
by
  sorry

end length_ae_l243_243549


namespace bertha_sandwiches_l243_243263

theorem bertha_sandwiches :
  let salamis := 8
  let cheeses := 7
  let sauces := 3
  let ways_to_choose_salami := salamis.choose 1
  let ways_to_choose_cheeses := cheeses.choose 2
  let ways_to_choose_sauces := sauces.choose 1
  ways_to_choose_salami * ways_to_choose_cheeses * ways_to_choose_sauces = 504 :=
by
  let salamis := 8
  let cheeses := 7
  let sauces := 3
  let ways_to_choose_salami := salamis.choose 1
  let ways_to_choose_cheeses := cheeses.choose 2
  let ways_to_choose_sauces := sauces.choose 1
  calc
    ways_to_choose_salami * ways_to_choose_cheeses * ways_to_choose_sauces
        = 8 * 21 * 3 : by simp [ways_to_choose_salami, ways_to_choose_cheeses, ways_to_choose_sauces]
    ... = 504       : by norm_num

end bertha_sandwiches_l243_243263


namespace gain_percent_calculation_l243_243804

def gain : ℝ := 0.70
def cost_price : ℝ := 70.0

theorem gain_percent_calculation : (gain / cost_price) * 100 = 1 := by
  sorry

end gain_percent_calculation_l243_243804


namespace sum_of_perimeters_l243_243803

-- Define side length of T1
def S1 : ℝ := 80

-- Define a function to get side length of nth triangle
def side_length (n : ℕ) : ℝ := S1 / 2^(n-1)

-- Define a function to get perimeter of nth triangle
def perimeter (n : ℕ) : ℝ := 3 * side_length n

-- Define a function to get the sum of perimeters of all triangles
def sum_perimeters : ℝ := ∑' n, perimeter n

theorem sum_of_perimeters :
  sum_perimeters = 480 :=
by
  sorry

end sum_of_perimeters_l243_243803


namespace solution_set_of_f_l243_243350

def f (x : ℝ) : ℝ :=
  if x ≥ -1 then 2 * x + 4 else -x + 1

theorem solution_set_of_f :
  {x : ℝ | f x < 4} = {x : ℝ | -3 < x ∧ x < 0} :=
by
  sorry

end solution_set_of_f_l243_243350


namespace smallest_four_digit_multiple_of_53_l243_243774

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l243_243774


namespace sin_double_alpha_minus_pi_over_3_l243_243315

theorem sin_double_alpha_minus_pi_over_3 (α : ℝ) (h : cos (α + π / 12) = -3 / 4) : 
  sin (2 * α - π / 3) = -1 / 8 :=
by
  sorry

end sin_double_alpha_minus_pi_over_3_l243_243315


namespace mappings_with_property_P_l243_243014

def is_property_P (f : ℂ → ℝ) : Prop :=
  ∀ (z₁ z₂ : ℂ) (λ : ℝ), f (λ • z₁ + (1 - λ) • z₂) = λ * f z₁ + (1 - λ) * f z₂

def f1 : ℂ → ℝ := λ z, z.re - z.im
def f2 : ℂ → ℝ := λ z, z.re^2 - z.im
def f3 : ℂ → ℝ := λ z, 2 * z.re + z.im

theorem mappings_with_property_P :
  is_property_P f1 ∧ ¬ is_property_P f2 ∧ is_property_P f3 :=
by
  sorry

end mappings_with_property_P_l243_243014


namespace integral_f_l243_243555

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 4 - x else real.sqrt(4 - x^2)

theorem integral_f :
  ∫ x in -2..2, f x = 10 + 2 * real.pi := by
  sorry

end integral_f_l243_243555


namespace smallest_four_digit_divisible_by_53_l243_243733

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243733


namespace binomial_third_and_fifth_term_expansion_l243_243981

noncomputable def binomial_expression (x y a : ℝ) := x + a * y

theorem binomial_third_and_fifth_term_expansion (a : ℝ) :
    let x := 3/2
    let y := 2/3
    let expr := binomial_expression x y a
    let t3 := (binom 7 2) * (x^2) * (a^5) * (y^5)
    let t5 := (binom 7 4) * (x^4) * (a^3) * (y^3)
    t3 = 6 + 2/9 ∧ t5 = 52 + 1/2 :=
sorry

end binomial_third_and_fifth_term_expansion_l243_243981


namespace smallest_four_digit_divisible_by_53_l243_243645

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l243_243645


namespace min_value_of_expression_l243_243121

theorem min_value_of_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x : ℝ, x = 6 * (12 : ℝ)^(1/6) ∧
  (∀ a b c, 0 < a ∧ 0 < b ∧ 0 < c → 
  x ≤ (a + 2 * b) / c + (2 * a + c) / b + (b + 3 * c) / a) :=
sorry

end min_value_of_expression_l243_243121


namespace square_of_1023_l243_243898

theorem square_of_1023 : 1023^2 = 1045529 := by
  sorry

end square_of_1023_l243_243898


namespace PR_result_l243_243089

noncomputable def PR_length (PQ RS QS: ℝ) (anglePQS angleQRS: ℝ) (ratioRS_PQ: ℚ): Prop :=
  PQ < RS ∧
  PQ > 0 ∧ 
  RS > 0 ∧ 
  QS = 1 ∧
  anglePQS = 30 ∧
  angleQRS = 60 ∧
  ratioRS_PQ = 7/4 ∧
  PR = 3/4

theorem PR_result (PQ RS QS: ℝ) (anglePQS angleQRS: ℝ) (ratioRS_PQ: ℚ) (h: PR_length PQ RS QS anglePQS angleQRS ratioRS_PQ): PR = 3/4 := 
sorry

end PR_result_l243_243089


namespace sum_of_all_valid_a_l243_243289

noncomputable def arc_sum_of_a : ℝ :=
  let eq := λ a x, (6 * real.pi * a - real.arcsin (real.sin x) + 
                    2 * real.arccos (real.cos x) - a * x) / 
                   (real.tan x ^ 2 + 4)
  let valid_a := λ a, set.univ.filter (λ x, x ≥ real.pi ∧ eq a x = 0).to_finset.card = 3
  let negative_a := λ a, a < 0
  ((({a : ℝ | negative_a a ∧ valid_a a}).to_finset.sum)) / finset.card {a : ℝ | negative_a a ∧ valid_a a }.to_finset

theorem sum_of_all_valid_a : arc_sum_of_a.round (100) = -1.6 := 
by 
    sorry

end sum_of_all_valid_a_l243_243289


namespace smallest_four_digit_divisible_by_53_l243_243634

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l243_243634


namespace square_of_1007_l243_243893

theorem square_of_1007 : 1007^2 = 1014049 := 
by {
  -- Break down the binomial expansion as per the solution steps
  calc 
    1007^2 = (1000 + 7)^2       : by rw [add_sq]
    ... = 1000^2 + 2*1000*7 + 7^2 : by norm_num
    ... = 1000000 + 14000 + 49   : by norm_num
    ... = 1014049               : by norm_num
}

end square_of_1007_l243_243893


namespace least_third_side_length_l243_243414

theorem least_third_side_length (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℝ, (c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2)) ∧ c = Real.sqrt 161 :=
by
  sorry

end least_third_side_length_l243_243414


namespace distance_between_trees_l243_243869

theorem distance_between_trees :
  ∀ (yard_length : ℝ) (num_trees : ℕ), yard_length = 1565 ∧ num_trees = 356 →
  let num_gaps := (num_trees - 1 : ℕ) in
  let distance := (yard_length / num_gaps) in
  distance = 4.41 :=
by
  intros yard_length num_trees h,
  let num_gaps := (num_trees - 1 : ℕ),
  let distance := (yard_length / num_gaps),
  cases h,
  sorry

end distance_between_trees_l243_243869


namespace least_third_side_of_right_triangle_l243_243408

theorem least_third_side_of_right_triangle {a b c : ℝ} 
  (h1 : a = 8) 
  (h2 : b = 15) 
  (h3 : c = Real.sqrt (8^2 + 15^2) ∨ c = Real.sqrt (15^2 - 8^2)) : 
  c = Real.sqrt 161 :=
by {
  intros h1 h2 h3,
  rw [h1, h2] at h3,
  cases h3,
  { exfalso, preciesly contradiction occurs because sqrt (8^2 + 15^2) is not sqrt161, 
   rw [← h3],
   norm_num,},
  { exact h3},
  
}

end least_third_side_of_right_triangle_l243_243408


namespace Marissa_sunflower_height_l243_243502

-- Define the necessary conditions
def sister_height_feet : ℕ := 4
def sister_height_inches : ℕ := 3
def extra_sunflower_height : ℕ := 21
def inches_per_foot : ℕ := 12

-- Calculate the total height of the sister in inches
def sister_total_height_inch : ℕ := (sister_height_feet * inches_per_foot) + sister_height_inches

-- Calculate the sunflower height in inches
def sunflower_height_inch : ℕ := sister_total_height_inch + extra_sunflower_height

-- Convert the sunflower height to feet
def sunflower_height_feet : ℕ := sunflower_height_inch / inches_per_foot

-- The theorem we want to prove
theorem Marissa_sunflower_height : sunflower_height_feet = 6 := by
  sorry

end Marissa_sunflower_height_l243_243502


namespace customers_who_bought_four_paintings_each_l243_243196

/-- Tracy's art fair conditions:
- 20 people came to look at the art
- Four customers bought two paintings each
- Twelve customers bought one painting each
- Tracy sold a total of 36 paintings

We need to prove the number of customers who bought four paintings each. -/
theorem customers_who_bought_four_paintings_each:
  let total_customers := 20
  let customers_bought_two_paintings := 4
  let customers_bought_one_painting := 12
  let total_paintings_sold := 36
  let paintings_per_customer_buying_two := 2
  let paintings_per_customer_buying_one := 1
  let paintings_per_customer_buying_four := 4
  (customers_bought_two_paintings * paintings_per_customer_buying_two +
   customers_bought_one_painting * paintings_per_customer_buying_one +
   x * paintings_per_customer_buying_four = total_paintings_sold) →
  (customers_bought_two_paintings + customers_bought_one_painting + x = total_customers) →
  x = 4 :=
by
  intro h1 h2
  sorry

end customers_who_bought_four_paintings_each_l243_243196


namespace right_triangle_least_side_l243_243401

theorem right_triangle_least_side (a b c : ℝ) (h_rt : a^2 + b^2 = c^2) (h1 : a = 8) (h2 : b = 15) : min a b = 8 := 
by
sorry

end right_triangle_least_side_l243_243401


namespace ceil_sqrt_225_eq_15_l243_243927

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := 
by 
  sorry

end ceil_sqrt_225_eq_15_l243_243927


namespace evaluate_f_at_points_l243_243910

def f (x : ℝ) : ℝ :=
  3 * x ^ 2 - 6 * x + 10

theorem evaluate_f_at_points : 3 * f 2 + 2 * f (-2) = 98 :=
by
  sorry

end evaluate_f_at_points_l243_243910


namespace inequality_holds_l243_243556

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
def even_function : Prop := ∀ x : ℝ, f x = f (-x)
def decreasing_on_pos : Prop := ∀ x y : ℝ, 0 < x → x < y → f y ≤ f x

-- Proof goal
theorem inequality_holds (h_even : even_function f) (h_decreasing : decreasing_on_pos f) : 
  f (-3/4) ≥ f (a^2 - a + 1) := 
by
  sorry

end inequality_holds_l243_243556


namespace max_magnitude_of_z_plus_i_l243_243810

theorem max_magnitude_of_z_plus_i :
  ∀ (x y : ℝ), (x, y) ∈ { p : ℝ × ℝ | (p.1 ^ 2) / 4 + p.2 ^ 2 = 1 } →
    abs (x + (y + 1) * Complex.I) ≤ 4 * Real.sqrt 3 / 3 := 
by
  sorry

end max_magnitude_of_z_plus_i_l243_243810


namespace find_probability_l243_243231

open ProbabilityTheory

noncomputable def probability_of_eleventh_draw (X : ℕ → ℕ) (p : ℕ → ℚ) : ℚ :=
  let redProb := 1/3
  let whiteProb := 2/3
  let comb := Nat.choose 10 8
  comb * (redProb ^ 9) * (whiteProb ^ 2)

theorem find_probability (X : ℕ → ℕ) (p : ℕ → ℚ) (h : ∀ n, p n = if n ≤ 9 then 1 else 0): 
  probability_of_eleventh_draw X p = Nat.choose 10 8 * (1 / 3) ^ 9 * (2 / 3) ^ 2 :=
by
  sorry

end find_probability_l243_243231


namespace problem_quadratic_radicals_l243_243019

theorem problem_quadratic_radicals (x y : ℝ) (h : 3 * y = x + 2 * y + 2) : x - y = -2 :=
sorry

end problem_quadratic_radicals_l243_243019


namespace quadratic_vertex_on_x_axis_l243_243417

theorem quadratic_vertex_on_x_axis (k : ℝ) :
  (∃ x : ℝ, (x^2 + 2 * x + k) = 0) → k = 1 :=
by
  sorry

end quadratic_vertex_on_x_axis_l243_243417


namespace smallest_four_digit_multiple_of_53_l243_243786

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l243_243786


namespace parabola_tangent_triangle_l243_243309

def is_tangent (a b : ℝ) : Prop := 
  ∃ (m : ℝ), (1 - 4 * m * b - 4 * m^2 * a = 0) ∧ (m^2 - 4 * a + 4 * b = 0)

theorem parabola_tangent_triangle (a : ℝ) (p q : ℕ) (hpq : Nat.gcd p q = 1) :
  (∃ (x y z : ℝ), 
    is_tangent a x ∧ is_tangent a y ∧ is_tangent a z ∧ 
    x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
    is_equilateral (x, y, z) ∧ 
    let s := (triangle_area_equilateral (x, y, z)) in 
    s^2 = p / q) → 
    p + q = 91 := 
by
  sorry

end parabola_tangent_triangle_l243_243309


namespace value_of_y_l243_243920

theorem value_of_y (y : ℚ) : |4 * y - 6| = 0 ↔ y = 3 / 2 :=
by
  sorry

end value_of_y_l243_243920


namespace smallest_divisible_by_6_l243_243270

-- Definitions of conditions
def digits := [1, 2, 3, 4, 5, 6]
def six_digit_permutations := list.permutations digits
def is_even (n : ℕ) := n % 2 = 0
def is_divisible_by_3 (n : ℕ) := n % 3 = 0

-- Main statement of the problem
theorem smallest_divisible_by_6 : ∃ n ∈ list.map (λ l, l.foldl (λ acc d => acc * 10 + d) 0) six_digit_permutations,
  is_even (n % 10) ∧ is_divisible_by_3 (digits.sum) ∧ n = 123456 :=
begin
  sorry
end

end smallest_divisible_by_6_l243_243270


namespace incorrect_congruent_statement_l243_243799

theorem incorrect_congruent_statement (F : Type) [Fig : F → Prop] : 
  (∀ (X Y : F), (X ∩ Y ≠ ∅ → congruent X Y)) ∧ 
  (∀ (X Y : F), symmetric_about_line X Y → congruent X Y) ∧ 
  (∀ (X Y : F), equilateral_triangle X ∧ equilateral_triangle Y ∧ side_length X = side_length Y → congruent X Y) →
  ¬(∀ (X Y : F), axial_symmetric X → congruent X Y) :=
sorry

end incorrect_congruent_statement_l243_243799


namespace geometric_sequence_sum_364_l243_243458

-- Define the geometric sequence
def geometric_seq (a1 q : ℝ) : ℕ → ℝ
| 0       => a1
| (n + 1) => (geometric_seq a1 q n) * q

-- Define the sum of the first k terms of a geometric sequence
def sum_geom_seq (a1 q : ℝ) (k : ℕ) : ℝ :=
  if q = 1 then a1 * k else a1 * (1 - q ^ k) / (1 - q)

-- Conditions
axiom a1 : ℝ := 1
axiom ak : ℝ := 243
axiom q : ℝ := 3
axiom k : ℕ := 6

-- Given the conditions, we need to prove the sum of the first k terms is 364
theorem geometric_sequence_sum_364 : sum_geom_seq a1 q k = 364 := by
  sorry

end geometric_sequence_sum_364_l243_243458


namespace smallest_four_digit_div_by_53_l243_243630

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l243_243630


namespace sequence_an_sequence_bn_sum_Tn_l243_243323

theorem sequence_an (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a n ^ 2 + a n = a (n + 1) ^ 2 - a (n + 1)) :
  ∀ n, a n = n :=
by
  sorry

theorem sequence_bn (a b : ℕ → ℕ) (h1 : ∀ n, a n = n) (h2 : ∀ n, b n = (n + 1) ^ 2 + a (n + 1) - (n ^ 2 + a n)) :
  ∀ n, b n = 2 * (n + 1) :=
by
  sorry

theorem sum_Tn (a b : ℕ → ℕ) (T : ℕ → ℚ) (h1 : ∀ n, a n = n)
               (h2 : ∀ n, b n = 2 * n)
               (h3 : ∀ n, T n = 1/2 * (Σ k in (finRange n), (1/k.to_nat - 1/(k + 1).to_nat))) :
  ∀ n, T n = n / (2 * (n + 1)) :=
by
  sorry

end sequence_an_sequence_bn_sum_Tn_l243_243323


namespace smallest_four_digit_divisible_by_53_l243_243607

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l243_243607


namespace smallest_four_digit_divisible_by_53_l243_243758

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243758


namespace cranberry_juice_cost_l243_243839

theorem cranberry_juice_cost 
  (cost_per_ounce : ℕ) (number_of_ounces : ℕ) 
  (h1 : cost_per_ounce = 7) 
  (h2 : number_of_ounces = 12) : 
  cost_per_ounce * number_of_ounces = 84 := 
by 
  sorry

end cranberry_juice_cost_l243_243839


namespace rich_walks_ratio_is_2_l243_243521

-- Define the conditions in the problem
def house_to_sidewalk : ℕ := 20
def sidewalk_to_end : ℕ := 200
def total_distance_walked : ℕ := 1980
def ratio_after_left_to_so_far (x : ℕ) : ℕ := (house_to_sidewalk + sidewalk_to_end) * x / (house_to_sidewalk + sidewalk_to_end)

-- Main theorem to prove the ratio is 2:1
theorem rich_walks_ratio_is_2 (x : ℕ) (h : 2 * ((house_to_sidewalk + sidewalk_to_end) * 2 + house_to_sidewalk + sidewalk_to_end / 2 * 3 ) = total_distance_walked) :
  ratio_after_left_to_so_far x = 2 :=
by
  sorry

end rich_walks_ratio_is_2_l243_243521


namespace evaluate_ceil_sqrt_225_l243_243942

def ceil (x : ℝ) : ℤ :=
  if h : ∃ n : ℤ, n ≤ x ∧ x < n + 1 then
    classical.some h
  else
    0

theorem evaluate_ceil_sqrt_225 : ceil (Real.sqrt 225) = 15 := 
sorry

end evaluate_ceil_sqrt_225_l243_243942


namespace smallest_four_digit_divisible_by_53_l243_243726

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l243_243726


namespace fly_travel_distance_l243_243850

theorem fly_travel_distance (radius : ℝ) (center_to_opposite : ℝ) (return_distance : ℝ) : 
    radius = 70 → center_to_opposite = 2 * radius → return_distance = 90 → 
    let second_leg := real.sqrt (center_to_opposite ^ 2 - return_distance ^ 2) in
    center_to_opposite + second_leg + return_distance = 337.1 :=
by
  assume h_radius h_center_to_opposite h_return_distance
  let second_leg := real.sqrt (center_to_opposite ^ 2 - return_distance ^ 2)
  show center_to_opposite + second_leg + return_distance = 337.1
  sorry

end fly_travel_distance_l243_243850


namespace least_side_is_8_l243_243392

-- Define the sides of the right triangle
variables (a b : ℝ) (h : a = 8) (k : b = 15)

-- Define a predicate for the least possible length of the third side
def least_possible_third_side (c : ℝ) : Prop :=
  (c = 8) ∨ (c = 15) ∨ (c = 17)

theorem least_side_is_8 (c : ℝ) (hc : least_possible_third_side c) : c = 8 :=
by
  sorry

end least_side_is_8_l243_243392


namespace min_value_sin_cos_function_l243_243562

open Real

theorem min_value_sin_cos_function :
  ∀ x : ℝ, let y := sin x ^ 4 + 2 * sin x * cos x + cos x ^ 4 in
  (∃ x : ℝ, y = -1/2) :=
by
  sorry

end min_value_sin_cos_function_l243_243562


namespace unique_7_step_knight_path_l243_243843

-- Definition of the chessboard and knight move conditions.
def is_valid_knight_move (start pos_end : (ℕ × ℕ)) : Prop :=
  let ⟨x, y⟩ := start in
  let ⟨nx, ny⟩ := pos_end in
  (ny = y + 1) ∧ ((nx = x + 1) ∨ (nx = x - 1)) ∧ (nx ≤ 7) ∧ (ny ≤ 7)

-- Definition of a 7-step path from bottom-left to top-right of the chessboard
def knight_7_step_path (start end : (ℕ × ℕ)) (path : List (ℕ × ℕ)) : Prop :=
  start = (0, 0) ∧ end = (7, 7) ∧ 
  List.length path = 7 ∧ 
  List.chain' is_valid_knight_move (start :: path) ∧ 
  List.nodup (start :: path) -- ensures no repeated rows

-- The proof statement that there is exactly one such 7-step path.
theorem unique_7_step_knight_path : ∃ (path : List (ℕ × ℕ)), knight_7_step_path (0, 0) (7, 7) path ∧ 
  ∀ path', knight_7_step_path (0, 0) (7, 7) path' → path' = path :=
sorry

end unique_7_step_knight_path_l243_243843


namespace sum_of_abs_roots_l243_243308

theorem sum_of_abs_roots (a b c : ℤ) (m : ℤ) (h_root : Polynomial.eval₂ (ringHom.id ℤ) a (Polynomial.X^3 - 2011*Polynomial.X + Polynomial.C m) = 0 ∧
    Polynomial.eval₂ (ringHom.id ℤ) b (Polynomial.X^3 - 2011*Polynomial.X + Polynomial.C m) = 0 ∧
    Polynomial.eval₂ (ringHom.id ℤ) c (Polynomial.X^3 - 2011*Polynomial.X + Polynomial.C m) = 0)  
    (h_sum : a + b + c = 0) (h_prod : a * b + b * c + c * a = -2011) :
    |a| + |b| + |c| = 98 := 
by
  sorry

end sum_of_abs_roots_l243_243308


namespace probability_two_heads_two_tails_four_coins_l243_243376

theorem probability_two_heads_two_tails_four_coins :
  let combinations := Nat.choose 4 2 in
  let probability_sequence := (1 / 2) ^ 4 in
  let favorable_probability := combinations * probability_sequence in
  favorable_probability = 3 / 8 :=
by
  sorry

end probability_two_heads_two_tails_four_coins_l243_243376


namespace smallest_four_digit_divisible_by_53_l243_243615

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l243_243615


namespace smallest_four_digit_divisible_by_53_l243_243654

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l243_243654


namespace CN_eq_2AL_l243_243311

variables {O : Type} [metric_space O] [nonempty O]
variables {A B C T T' R N L : O}
variables (circle : set O) (triangle : set O)

-- Given triangle ABC inscribed in circle O
variables (Htriangle : triangle = {A, B, C})
variables (Hcircle : circle = O)

-- Given CT is a diameter of the circle
variables (Hdiameter : T ∈ O ∧ T ∈ circle ∧ ∀ P ∈ circle, dist P O < dist O T)

-- T' is the reflection of T across line AB
variables (Hreflection : T' = reflection T AB)

-- BT' intersects circle at R
variables (Hintersection_R : R ∈ circle ∧ lines_intersect (line_segment B T') (circle))

-- Line TR intersects AC at N
variables (Hintersection_N : ∃ N ∈ (line_segment T R), N ∈ (line_segment A C))

-- OL ⊥ CT and intersects AC at L
variables (Hperpendicular : L ∈ (line_segment O L) ∧ L ∈ (line_segment A C) ∧ is_perpendicular (line_segment O L) (line_segment C T))

-- Objective: Prove CN = 2AL
theorem CN_eq_2AL (H: (CN = 2AL)) : true :=
sorry

end CN_eq_2AL_l243_243311


namespace relationship_between_exponents_l243_243479

theorem relationship_between_exponents 
  (p r : ℝ) (u v s t m n : ℝ)
  (h1 : p^u = r^s)
  (h2 : r^v = p^t)
  (h3 : m = r^s)
  (h4 : n = r^v)
  (h5 : m^2 = n^3) :
  (s / u = v / t) ∧ (2 * s = 3 * v) :=
  by
  sorry

end relationship_between_exponents_l243_243479


namespace smallest_four_digit_divisible_by_53_l243_243618

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l243_243618


namespace num_valid_n_l243_243916

theorem num_valid_n : 
  {n : ℤ | 0 ≤ n ∧ n < 30 ∧ ∃ k : ℤ, n = k^2 + 30k } = {0, 15, 24, 27} :=
sorry

end num_valid_n_l243_243916


namespace ceil_sqrt_225_eq_15_l243_243969

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := by
  sorry

end ceil_sqrt_225_eq_15_l243_243969


namespace locus_of_centroids_is_segment_l243_243865

-- Declare the conditions and theorem statement
theorem locus_of_centroids_is_segment
  (θ : ℝ) -- Given angle
  (a : ℝ) -- Constant sum of intercepts
  (secant_intercepts : ℝ → ℝ × ℝ) -- Function describing secant intercepts on the sides such that their sum is constant
  (h_sum_constant : ∀ x, (secant_intercepts x).fst + (secant_intercepts x).snd = 2 * a) :
  ∃ s : set (ℝ × ℝ), (∀ x, centroid (secant_intercepts x) ∈ s) ∧ (is_line_segment s) :=
sorry

-- Assuming we have defined a centroid function and is_line_segment predicate suitably elsewhere

end locus_of_centroids_is_segment_l243_243865


namespace smallest_four_digit_divisible_by_53_l243_243608

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l243_243608


namespace ceil_sqrt_225_l243_243936

theorem ceil_sqrt_225 : Nat.ceil (Real.sqrt 225) = 15 :=
by
  sorry

end ceil_sqrt_225_l243_243936


namespace ceil_sqrt_225_eq_15_l243_243961

theorem ceil_sqrt_225_eq_15 : 
  ⌈Real.sqrt 225⌉ = 15 := 
by sorry

end ceil_sqrt_225_eq_15_l243_243961


namespace smallest_four_digit_divisible_by_53_l243_243731

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l243_243731


namespace find_blue_weights_l243_243881

theorem find_blue_weights (B : ℕ) :
  (2 * B + 15 + 2 = 25) → B = 4 :=
by
  intro h
  sorry

end find_blue_weights_l243_243881


namespace factor_theorem_q_value_l243_243220

theorem factor_theorem_q_value (q : ℤ) (m : ℤ) :
  (∀ m, (m - 8) ∣ (m^2 - q * m - 24)) → q = 5 :=
by
  sorry

end factor_theorem_q_value_l243_243220


namespace marion_ella_score_ratio_l243_243495

theorem marion_ella_score_ratio 
  (total_items : ℕ) (incorrect_answers_ella : ℕ) (score_marion : ℕ) 
  (h1 : total_items = 40) (h2 : incorrect_answers_ella = 4) (h3 : score_marion = 24) : 
  let score_ella := total_items - incorrect_answers_ella in 
  score_marion.toRat / score_ella.toRat = (2 : ℚ) / 3 :=
by
  sorry

end marion_ella_score_ratio_l243_243495


namespace ceil_sqrt_225_l243_243950

theorem ceil_sqrt_225 : ⌈real.sqrt 225⌉ = 15 :=
by
  have h : real.sqrt 225 = 15 := by
    sorry
  rw [h]
  exact int.ceil_eq_self.mpr rfl

end ceil_sqrt_225_l243_243950


namespace expression_simplifies_to_32_l243_243149

noncomputable def simplified_expression (a : ℝ) : ℝ :=
  8 / (1 + a^8) + 4 / (1 + a^4) + 2 / (1 + a^2) + 1 / (1 + a) + 1 / (1 - a)

theorem expression_simplifies_to_32 :
  simplified_expression (2^(-1/16 : ℝ)) = 32 :=
by
  sorry

end expression_simplifies_to_32_l243_243149


namespace triangle_angles_l243_243120

-- Define the variables and conditions
variables (A B C P Q : Type)
variables (triangle_ABC : triangle A B C)
variables (BC AB AC BP CQ : ℝ)
variables (h1 : BC > AB) (h2 : BC > AC)
variables (h3 : BP = AB) (h4 : CQ = AC)
variables (angle_BAC angle_PAQ : ℝ)

-- State the required proof
theorem triangle_angles (h₁ : BC > AB) (h₂ : BC > AC) (h₃ : BP = AB) (h₄ : CQ = AC)
  (h5 : ∠ BAC = angle_BAC) (h6 : ∠ PAQ = angle_PAQ) : 
  angle_BAC + 2 * angle_PAQ = 180 :=
sorry

end triangle_angles_l243_243120


namespace value_of_m_l243_243379

-- Define the initial conditions for the problem
def is_quadratic (m : ℝ) : Prop :=
  (m^2 - 3 * m + 2 = 2) ∧ (m - 3 ≠ 0)

-- Formalize the required theorem
theorem value_of_m : ∀ m : ℝ, is_quadratic m → m = 0 :=
by 
  intros m h,
  cases h with h1 h2,
  have h3 : m * (m - 3) = 0,
  {
    rw [← sub_eq_zero, sub_eq_add_neg, add_comm],
    linarith,
  },
  cases eq_or_eq_of_mul_eq_zero h3 with h4 h5,
  { exact h4 },
  { contradiction }

end value_of_m_l243_243379


namespace prime_factorization_of_x_l243_243116

theorem prime_factorization_of_x {x y : ℕ} (h1 : 0 < x) (h2: 0 < y) (h3: 5 * x^7 = 13 * y^11) :
  let a := 13; b := 5; c := 16; d := 28 in a + b + c + d = 62 :=
by
  sorry

end prime_factorization_of_x_l243_243116


namespace number_of_intersections_of_lines_l243_243298

theorem number_of_intersections_of_lines : 
  let L1 := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 = 12}
  let L2 := {p : ℝ × ℝ | 5 * p.1 - 2 * p.2 = 10}
  let L3 := {p : ℝ × ℝ | p.1 = 3}
  let L4 := {p : ℝ × ℝ | p.2 = 1}
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ p1 ∈ L1 ∧ p1 ∈ L2 ∧ p2 ∈ L3 ∧ p2 ∈ L4 :=
by
  sorry

end number_of_intersections_of_lines_l243_243298


namespace sine_curve_tangent_inclination_angle_range_l243_243320

noncomputable def inclination_angle_range (x : ℝ) : Prop :=
  let slope := Math.cos x in
  (0 ≤ slope ∧ slope ≤ Math.cos (π / 4)) ∨ (Math.cos (3 * π / 4) ≤ slope ∧ slope < Math.cos π)

theorem sine_curve_tangent_inclination_angle_range :
  ∀ (x : ℝ), inclination_angle_range x :=
sorry

end sine_curve_tangent_inclination_angle_range_l243_243320


namespace smallest_four_digit_divisible_by_53_l243_243734

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243734


namespace smallest_four_digit_divisible_by_53_l243_243677

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l243_243677


namespace acute_external_angles_l243_243281

theorem acute_external_angles {n : ℕ} (h₁ : 3 ≤ n) (h₂ : convex_ngon n) : 
  (if n = 3 then 0 else if n = 4 then 0 else n - 3) ≤ count_acute_external_angles n ∧ 
  count_acute_external_angles n ≤ n :=
sorry

end acute_external_angles_l243_243281


namespace num_arrangements_l243_243261

theorem num_arrangements : ∃ n k : ℕ, n ≥ 3 ∧ n * (2 * k + (n - 1)) = 200 ∧ (n, k) ∈ {(5, 18), (8, 9)} :=
by {
  sorry
}

end num_arrangements_l243_243261


namespace evaluate_ceil_sqrt_225_l243_243943

def ceil (x : ℝ) : ℤ :=
  if h : ∃ n : ℤ, n ≤ x ∧ x < n + 1 then
    classical.some h
  else
    0

theorem evaluate_ceil_sqrt_225 : ceil (Real.sqrt 225) = 15 := 
sorry

end evaluate_ceil_sqrt_225_l243_243943


namespace find_cubic_polynomial_l243_243984

theorem find_cubic_polynomial (q : ℝ → ℝ) 
  (h1 : q 1 = -8) 
  (h2 : q 2 = -12) 
  (h3 : q 3 = -20) 
  (h4 : q 4 = -40) : 
  q = (λ x, - (4 / 3) * x^3 + 6 * x^2 - 4 * x - 2) :=
sorry

end find_cubic_polynomial_l243_243984


namespace apples_left_over_l243_243044

open Nat

variable (G S M E : ℕ)

def split_apples (total : ℕ) : ℕ := total / 2

def susan_apples (greg_apples : ℕ) : ℕ := 2 * greg_apples

def mark_apples (susan_apples : ℕ) : ℕ := susan_apples - 5

def emily_apples (mark_apples : ℕ) : ℕ := mark_apples + (3 / 2 : ℚ).toNat

theorem apples_left_over :
  split_apples 18 + susan_apples (split_apples 18) + mark_apples (susan_apples (split_apples 18)) + emily_apples (mark_apples (susan_apples (split_apples 18))) - 40 = 14 :=
by
  -- We leave the proof as an exercise.
  sorry

end apples_left_over_l243_243044


namespace running_time_l243_243468

theorem running_time :
  ∀ (J P : ℝ) (t : ℝ),
  J = 0.266666666667 ∧ J = 2 * P ∧ 16 = J * t + P * t → t = 40 :=
by
  intros J P t h
  cases h with hJ hPJ
  cases hPJ with hJP hdist
  sorry

end running_time_l243_243468


namespace number_of_incorrect_statements_l243_243873

theorem number_of_incorrect_statements :
  let cond1 := ¬(∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧ (∃ x : ℝ, x ≠ 1 ∧ x^2 - 3*x + 2 = 0) -- Condition for ①
  let cond2 := ¬(¬(∀ x : ℝ, cos x ≤ 1) ↔ (∃ x : ℝ, cos x > 1)) -- Condition for ②
  let cond3 := ∀ p q : Prop, (¬p → ¬q) = (q → p) -- Condition for ③ 
  cond1 ∧ cond2 ∧ cond3 → 
  ((if ¬cond1 then 1 else 0) + (if ¬cond2 then 1 else 0) + (if ¬cond3 then 1 else 0)) = 2 :=
by
  sorry

end number_of_incorrect_statements_l243_243873


namespace smallest_square_area_is_320_l243_243991

noncomputable def smallest_area_of_square : ℝ :=
  let f : ℝ → ℝ := λ x, 2 * x - 20
  let g : ℝ → ℝ := λ x, x ^ 2 + 3 * x
  let intersection_point_condition (k : ℝ) : Prop :=
    ∃ x1 x2 : ℝ, x1 + x2 = -1 ∧ x1 * x2 = k - 20
  let distance_condition (k : ℝ) : Prop :=
    80 - 20 * k = (k - 40) ^ 2 / 5
  if ∃ k : ℝ, intersection_point_condition k ∧ distance_condition k
  then if distance_condition (-12) then 320 else 0
  else if distance_condition (-8) then 320 else 0

theorem smallest_square_area_is_320 : smallest_area_of_square = 320 := sorry

end smallest_square_area_is_320_l243_243991


namespace balanced_chessboard_max_x_l243_243229

noncomputable def max_x (n : ℕ) : ℝ :=
  if n % 2 = 1 then 1 / ((n + 1) / 2 : ℝ)^2
  else 1 / ((n / 2 : ℝ) * ((n / 2 : ℝ) + 1))

theorem balanced_chessboard_max_x (n : ℕ) (M : matrix (fin n) (fin n) ℝ) (h_row_sum : ∀ i : fin n, (finset.univ.sum (λ j, M i j) = 1)) (h_col_sum : ∀ j : fin n, (finset.univ.sum (λ i, M i j) = 1)) :
  ∃ (cells : finset (fin n × fin n)), cells.card = n ∧
    (∀ (c : fin n × fin n), c ∈ cells → M c.1 c.2 ≥ max_x n) ∧
    (function.injective (λ c : cells, c.fst)) ∧
    (function.injective (λ c : cells, c.snd)) :=
sorry

end balanced_chessboard_max_x_l243_243229


namespace smallest_four_digit_divisible_by_53_l243_243648

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l243_243648


namespace surface_area_ratio_cube_to_rectangular_solid_l243_243224

theorem surface_area_ratio_cube_to_rectangular_solid (s : ℝ) (h_s_pos : s > 0) : 
  let A_cube := 6 * s^2 in
  let A_rect_solid := 2 * (2 * s) * s + 2 * (2 * s) * s + 2 * s * s in
  A_cube / A_rect_solid = 3 / 5 :=
by
  sorry

end surface_area_ratio_cube_to_rectangular_solid_l243_243224


namespace constant_term_expansion_l243_243598

theorem constant_term_expansion :
  (∃ k : ℕ, k ∈ finset.range 13 ∧ 
            (∃ (term : ℚ), term = binom 12 k * (x ^ (k / 3) * (4 / x ^ 2) ^ (12 - k))) ∧
            is_constant term) →
  term.coeff = 126720 :=
by
  sorry

end constant_term_expansion_l243_243598


namespace smallest_four_digit_divisible_by_53_l243_243720

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l243_243720


namespace parallel_centers_lines_l243_243445

variables {A B C D K : Point}
variables (ABCD : Parallelogram A B C D) (K_on_AC : OnDiagonal K A C)
variables (s1 s2 : Circle)
variables (s1_tangent_AB : Tangent s1 A B) (s1_tangent_AD : Tangent s1 A D)
variables (s2_tangent_CB : Tangent s2 C B) (s2_tangent_CD : Tangent s2 C D)
variables (s1_second_intersection_AK : SecondIntersection s1 A K)
variables (s2_second_intersection_KC : SecondIntersection s2 K C)

theorem parallel_centers_lines (Centers_parallel : ∀ K ∈ AC, Parallel Lines (Center s1) (Center s2)) :
  Parallel (Line (Center s1) (Center s2)) (Line (Center s1') (Center s2')) :=
sorry

end parallel_centers_lines_l243_243445


namespace smallest_four_digit_divisible_by_53_l243_243678

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l243_243678


namespace Margie_can_drive_125_miles_l243_243493

theorem Margie_can_drive_125_miles (miles_per_gallon : ℕ) (cost_per_gallon : ℕ) (total_money : ℕ) 
    (h1 : miles_per_gallon = 25) (h2 : cost_per_gallon = 5) (h3 : total_money = 25) : 
    ((total_money / cost_per_gallon) * miles_per_gallon) = 125 := 
by
  rw [h1, h2, h3]
  show (25 / 5 * 25) = 125 
  -- Here, 25 / 5 evaluates to 5 and 5 * 25 evaluates to 125
  rfl

end Margie_can_drive_125_miles_l243_243493


namespace probability_no_adjacent_green_hats_l243_243812

-- Step d): Rewrite the math proof problem in a Lean 4 statement.

theorem probability_no_adjacent_green_hats (total_children green_hats : ℕ)
  (hc : total_children = 9) (hg : green_hats = 3) :
  (∃ (p : ℚ), p = 5 / 14) :=
sorry

end probability_no_adjacent_green_hats_l243_243812


namespace cats_sold_l243_243882

theorem cats_sold (ratio : ℕ → ℕ → Prop) (h_ratio : ratio 2 1) (dogs_sold : ℕ) (h_dogs : dogs_sold = 8) : ∃ cats_sold : ℕ, cats_sold = 16 :=
by
  existsi (2 * dogs_sold)
  calc
   2 * dogs_sold
      = 2 * 8      : by rw [h_dogs]
  ... = 16         : by norm_num

end cats_sold_l243_243882


namespace min_value_of_a_l243_243303

theorem min_value_of_a :
  ∀ (x y : ℝ), |x| + |y| ≤ 1 → (|2 * x - 3 * y + 3 / 2| + |y - 1| + |2 * y - x - 3| ≤ 23 / 2) :=
by
  intros x y h
  sorry

end min_value_of_a_l243_243303


namespace find_number_l243_243070

theorem find_number (x : ℝ) (h : 0.6 * ((x / 1.2) - 22.5) + 10.5 = 30) : x = 66 :=
by
  sorry

end find_number_l243_243070


namespace painting_problem_l243_243922

theorem painting_problem :
  let doug_rate := 1/6
  let dave_rate := 1/8
  let combined_rate := 1/6 + 1/8
  ∀ t : ℝ, (combined_rate * (t - 1) = 1) :=
begin
  let doug_rate := (1/6 : ℝ),
  let dave_rate := (1/8 : ℝ),
  let combined_rate := doug_rate + dave_rate,
  sorry -- Proof goes here
end

end painting_problem_l243_243922


namespace petya_coin_difference_20_l243_243511

-- Definitions for the problem conditions
variables (n k : ℕ) -- n: number of 5-ruble coins Petya has, k: number of 2-ruble coins Petya has

-- Condition: Petya has 60 rubles more than Vanya
def petya_has_60_more (n k : ℕ) : Prop := (5 * n + 2 * k = 5 * k + 2 * n + 60)

-- Theorem to prove Petya has 20 more 5-ruble coins than 2-ruble coins
theorem petya_coin_difference_20 (n k : ℕ) (h : petya_has_60_more n k) : n - k = 20 :=
sorry

end petya_coin_difference_20_l243_243511


namespace general_term_b_sum_S_l243_243475

noncomputable def a : ℕ → ℚ 
| 0     := 0
| 1     := 1
| 2     := 5/3
| (n+3) := 5/3 * a (n+2) - 2/3 * a (n+1)

def b (n : ℕ) : ℚ := a (n+1) - a n

def S (n : ℕ) : ℚ := ∑ k in Ico 1 n, k * a k

theorem general_term_b (n : ℕ) :
  b n = (2/3)^n :=
sorry

theorem sum_S (n : ℕ) :
  S n = (3/2 : ℚ) * n * (n + 1) + (3 + n) * 2^(n + 1) / (3^(n - 1)) - 18 :=
sorry 

end general_term_b_sum_S_l243_243475


namespace find_angle_FYD_l243_243451

-- Step a): Conditions
variables {A B C D E F X Y : Type}  -- The points in the problem
variable [parallel : A ≠ B]  -- Ensure A and B are distinct points
variable [parallel : C ≠ D]  -- Ensure C and D are distinct points
variable [parallel_AB_CD : A ≠ C ∧ B ≠ D]  -- Ensure line segments are distinct and parallel
variable (h_parallel : ∀ x : Type, x ∈ A ∧ x ∈ B ↔ x ∈ C ∧ x ∈ D)  -- Formalization of parallel lines AB ∥ CD
variable (h_AXF : Type)  -- Angle AXF
variable (angle_AXF : 118)  -- Formalization of Angle AXF

-- Step d): Statement
theorem find_angle_FYD :
  ∀ (A B C D E F X Y : Type) (h_parallel : ∀ x, (x ∈ A ∧ x ∈ B) ↔ (x ∈ C ∧ x ∈ D)) (angle_AXF : ℝ),
  angle_AXF = 118 → 
  angle_FYD = 62 := begin
  sorry
end

end find_angle_FYD_l243_243451


namespace robert_pencils_l243_243151

-- Define the conditions as given in the problem
def pencil_price := 0.20
def tolu_pencils := 3
def melissa_pencils := 2
def total_spent := 2.00

-- Given the conditions, prove that Robert wants 5 pencils
theorem robert_pencils : 
  (total_spent - (tolu_pencils * pencil_price + melissa_pencils * pencil_price)) / pencil_price = 5 :=
by
  sorry

end robert_pencils_l243_243151


namespace smallest_four_digit_divisible_by_53_l243_243727

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l243_243727


namespace range_of_f_is_pi_div_four_l243_243978

noncomputable def f (x : ℝ) : ℝ := 
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f_is_pi_div_four : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y = π / 4 :=
sorry

end range_of_f_is_pi_div_four_l243_243978


namespace value_of_a_is_minus_one_l243_243476

-- Define the imaginary unit i
def imaginary_unit_i : Complex := Complex.I

-- Define the complex number condition
def complex_number_condition (a : ℝ) : Prop :=
  let z := (a + imaginary_unit_i) / (1 + imaginary_unit_i)
  (Complex.re z) = 0 ∧ (Complex.im z) ≠ 0

-- Prove that the value of the real number a is -1 given the condition
theorem value_of_a_is_minus_one (a : ℝ) (h : complex_number_condition a) : a = -1 :=
sorry

end value_of_a_is_minus_one_l243_243476


namespace sine_middle_angle_l243_243560

theorem sine_middle_angle (n : ℝ) (h1 : 0 < n) 
  (h2 : let a := n; 
             let b := n + 2; 
             let c := n + 4;
             let A := angle a b c in
           A * 2 = angle a b (n+4)) :
  sin (angle n (n+2) (n+4)) = (Real.sqrt 5) / 3 := 
sorry

end sine_middle_angle_l243_243560


namespace limit_final_score_probability_l243_243855

open MeasureTheory

noncomputable def probability_on_die (n : ℕ) : ℚ := 1 / 6

def final_score_probability (n : ℕ) : ℕ → ℚ
| 0 := sorry -- p_0(n), need definition but filled with placeholder for now
| k := sorry -- p_k(n) for k > 0, need recurrence but filled with placeholder

def total_probability (n : ℕ) : ℚ :=
(0 : ℕ).upto 5 0 sorry -- summing all p_k(n) from k = 0 to 5

theorem limit_final_score_probability :
  filter.at_top.lim (λ n, final_score_probability n 0) = 2 / 7 :=
sorry

end limit_final_score_probability_l243_243855


namespace circumcircle_radius_triangle_l243_243584

theorem circumcircle_radius_triangle
  (r₁ r₂ : ℝ) (h₁ : r₁ + r₂ = 11) (d₁ : ∥(0 : ℝ), 0, r₁ - 0∥ = ∥(6 : ℝ), 0, r₂ - 0∥ = √481)
  (r₃ : ℝ) (h₂ : r₃ = 9)
  (AC_BC_external : ∥(0 : ℝ), 0, r₁ - (0 : ℝ), 0, r₃∥ = ∥(6 : ℝ), 0, r₂ - (6 : ℝ), 0, r₃∥) :
  ∃ (R : ℝ), R = 3 * √10 :=
by
  sorry

end circumcircle_radius_triangle_l243_243584


namespace unique_monic_polynomial_l243_243288

theorem unique_monic_polynomial (f : ℤ[X]) (N : ℕ) :
  (monic f ∧ ∀ p : ℕ, prime p ∧ p > N ∧ 0 < f.eval (p : ℤ) → p ∣ 2 * (nat.factorial (f.eval p).to_nat) + 1) →
  f = X - C (3 : ℤ) :=
sorry

end unique_monic_polynomial_l243_243288


namespace smallest_four_digit_divisible_by_53_l243_243606

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l243_243606


namespace Otto_knives_cost_l243_243074

noncomputable def knife_sharpening_cost : ℕ → ℝ
| 1 := 6.00
| n := if n ≤ 4 then 4.50 else if n ≤ 10 then 3.75 else 3.25

def discount (knife_type : string) (base_cost : ℝ) : ℝ :=
  if knife_type = "chef" then base_cost * 0.15  
  else if knife_type = "paring" then base_cost * 0.10 
  else 0.0

def total_cost (chef_knives paring_knives bread_knives : ℕ) : ℝ :=
  let total_knives := chef_knives + paring_knives + bread_knives
  let base_cost := (List.range (total_knives + 1)).tail.map knife_sharpening_cost |> List.sum
  let chef_discount := List.range chef_knives.map (knife_sharpening_cost ∘ (λ n => n + 1)).map (discount "chef") |> List.sum
  let paring_discount := List.range paring_knives.map (λ n => if n < 1 then knife_sharpening_cost (n + 2) else knife_sharpening_cost (n + 5))
                                    |> List.map (discount "paring") |> List.sum
  base_cost - chef_discount - paring_discount

theorem Otto_knives_cost : 
  total_cost 3 4 8 = 54.35 := 
  by
    -- proof goes here
    sorry

end Otto_knives_cost_l243_243074


namespace sum_of_integers_abs_lt_2023_l243_243187

theorem sum_of_integers_abs_lt_2023 : 
  (∑ n in Finset.Icc (-2022) 2022, n) = 0 :=
by
  sorry

end sum_of_integers_abs_lt_2023_l243_243187


namespace derivative_of_function_l243_243162

theorem derivative_of_function : 
  ∀ (x : ℝ), deriv (λ (x : ℝ), x^3 - x⁻¹) x = 3 * x^2 + x⁻² :=
by 
  -- the proof will go here
  -- using sorry to indicate the proof part
  sorry

end derivative_of_function_l243_243162


namespace dot_product_result_l243_243358

variables (a b : EuclideanSpace ℝ (Fin 3))

-- Given conditions
axiom unit_vector_a : ∥a∥ = 1
axiom unit_vector_b : ∥b∥ = 1
axiom angle_120 : real.angle (a, b) = real.pi / 3

-- The theorem to be proven
theorem dot_product_result : (a + b) ⬝ a = 1 / 2 :=
by sorry

end dot_product_result_l243_243358


namespace sin_C_in_right_triangle_l243_243448

theorem sin_C_in_right_triangle (A B C : Triangle) (h : has_right_angle B) (sin_A : Real) (h1 : sin_A = 3 / 5) :
  sin C = 4 / 5 :=
sorry

end sin_C_in_right_triangle_l243_243448


namespace seven_distinct_numbers_with_reversed_digits_l243_243078

theorem seven_distinct_numbers_with_reversed_digits (x y : ℕ) :
  (∃ a b c d e f g : ℕ, 
  (10 * a + b + 18 = 10 * b + a) ∧ (10 * c + d + 18 = 10 * d + c) ∧ 
  (10 * e + f + 18 = 10 * f + e) ∧ (10 * g + y + 18 = 10 * y + g) ∧ 
  a ≠ c ∧ a ≠ e ∧ a ≠ g ∧ 
  c ≠ e ∧ c ≠ g ∧ 
  e ≠ g ∧ 
  (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧
  (1 ≤ c ∧ c ≤ 9) ∧ (1 ≤ d ∧ d ≤ 9) ∧
  (1 ≤ e ∧ e <= 9) ∧ (1 ≤ f ∧ f <= 9) ∧
  (1 ≤ g ∧ g <= 9) ∧ (1 ≤ y ∧ y <= 9)) :=
sorry

end seven_distinct_numbers_with_reversed_digits_l243_243078


namespace domain_of_h_l243_243204

def h (x : ℝ) : ℝ := (4 * x^2 + 2 * x - 3) / (x - 5)

theorem domain_of_h : {x : ℝ | x ≠ 5} = (-∞, 5) ∪ (5, ∞) :=
by
  sorry

end domain_of_h_l243_243204


namespace smallest_four_digit_multiple_of_53_l243_243711

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243711


namespace cone_surface_area_l243_243184

-- Declare the necessary definitions as per the conditions
def slant_height := 2
def unfolded_shape := "semicircle"

-- Assume the radius of the base of the cone is r, and its surface area formula
theorem cone_surface_area (r : ℝ) (l : ℝ) 
  (h_slant_height : l = 2) 
  (h_unfolded_shape : unfolded_shape = "semicircle") 
  (h_arc_length : 2 * π * r = 2 * π) 
  : π * r * (r + l) = 3 * π :=
by
  have : r = 1 := by
    simp [h_arc_length]
  rw [this, h_slant_height]
  simp
  sorry


end cone_surface_area_l243_243184


namespace smallest_four_digit_divisible_by_53_l243_243613

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l243_243613


namespace num_integers_not_satisfying_inequality_l243_243992

theorem num_integers_not_satisfying_inequality :
  ∃ s : Finset ℤ, s.card = 5 ∧ ∀ x ∈ s, ¬ (4 * x^2 + 24 * x + 35 > 15) :=
by
  let s := {x | x = -5 ∨ x = -4 ∨ x = -3 ∨ x = -2 ∨ x = -1}.to_finset
  use s
  split
  · simp [Finset.card, s, List.length, List.range_succ_eq_map, List.filter, List.countp, ← List.sigma_take_one, List.perm]
  · intro x hx
    simp only [Finset.mem_to_finset, smul_eq_mul, insert_subset]
    cases hx
    all_goals
    simp [hx]
    linarith

end num_integers_not_satisfying_inequality_l243_243992


namespace smallest_four_digit_divisible_by_53_l243_243676

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l243_243676


namespace max_distance_on_spheres_l243_243205

-- Define the centers and radii of the spheres
def O : ℝ × ℝ × ℝ := (-5, -15, 10)
def P : ℝ × ℝ × ℝ := (15, 12, -21)
def r1 : ℝ := 23
def r2 : ℝ := 91

-- Calculate the distance between centers
noncomputable def distance_between_centers : ℝ :=
  real.sqrt ((-5 - 15) ^ 2 + (-15 - 12) ^ 2 + (10 - (-21)) ^ 2)

-- State the maximum distance between any point on the two spheres
theorem max_distance_on_spheres :
  let max_distance := r1 + distance_between_centers + r2 in
  max_distance = 23 + real.sqrt 2090 + 91 :=
sorry

end max_distance_on_spheres_l243_243205


namespace length_of_AC_l243_243514

-- Definitions in Lean 4
variable (O A B C : Point) (r : ℝ)
variable {h1 : r = 8} -- This sets the radius of the circle to 8
variable {h2 : dist A B = 10} -- This sets the length of the chord AB to 10
variable {h3 : midpoint C A B} -- C is the midpoint of AB

-- The statement to be proven
theorem length_of_AC : dist A C = 8 := sorry

end length_of_AC_l243_243514


namespace intersection_M_N_l243_243039

def M : Set ℤ := { x | x^2 > 1 }
def N : Set ℤ := { -2, -1, 0, 1, 2 }

theorem intersection_M_N : (M ∩ N) = { -2, 2 } :=
sorry

end intersection_M_N_l243_243039


namespace find_ON_l243_243257

-- Definitions based on the conditions
def ellipse (x y : ℝ) : Prop := (x^2 / 25) + (y^2 / 9) = 1

def distance (A B : ℝ × ℝ) : ℝ := (real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2 ))

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

variable {M F1 N O : ℝ × ℝ}

-- Conditions in Lean terms
axiom ellipse_M_on_ellipse : ellipse M.1 M.2
axiom distance_M_F1 : distance M F1 = 2
axiom N_is_midpoint : N = midpoint M F1
axiom O_is_origin : O = (0, 0)

-- Proof target
theorem find_ON : distance O N = 4 :=
sorry

end find_ON_l243_243257


namespace problem_solution_l243_243596

theorem problem_solution :
  (30 - (3010 - 310)) + (3010 - (310 - 30)) = 60 := 
  by 
  sorry

end problem_solution_l243_243596


namespace eccentricity_hyperbola_l243_243037

-- Definitions to capture the problem conditions
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

variables {a b : ℝ}

-- Assuming A at (-a, 0) since it's the left vertex and asymptotes y = ±(b/a)x
def point_A (a : ℝ) : Prop := a > 0

def point_P (a b x1 y1 : ℝ) : Prop := 
  (a > 0) ∧ (b > 0) ∧ (y1 = (b / a) * x1)

def point_Q (a b x2 y2 : ℝ) : Prop := 
  (a > 0) ∧ (b > 0) ∧ (y2 = -(b / a) * x2)

-- Given slope condition
def line_AP_slope (x1 y1 : ℝ) : Prop := y1 = x1

-- Define trisection condition
def trisection (x1 y1 x2 y2 : ℝ) : Prop := 
  (a > 0) ∧ (b > 0) ∧ 3 * (y1 = -(b / a) * x2 - (-a))

theorem eccentricity_hyperbola (h : hyperbola a b) :
  point_A a ∧ ∃ (x1 y1 x2 y2 : ℝ), point_P a b x1 y1 ∧ point_Q a b x2 y2 ∧ line_AP_slope x1 y1 ∧ trisection x1 y1 x2 y2 → 
  let e := sqrt(1 + (b^2 / a^2)) in 
  e = (sqrt(10) / 3) :=
by sorry

end eccentricity_hyperbola_l243_243037


namespace min_distance_between_points_on_circles_is_3sqrt5_minus_5_l243_243489

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 4*y + 11 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x + 2*y + 1 = 0

theorem min_distance_between_points_on_circles_is_3sqrt5_minus_5 :
  let P Q : ℝ × ℝ := (x y : P.1 × P.2 ∧ circle1 P.1 P.2) ∧ (Q.1 × Q.2 ∧ circle2 Q.1 Q.2)) in
  (sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)) = 3 * real.sqrt 5 - 5 :=
sorry

end min_distance_between_points_on_circles_is_3sqrt5_minus_5_l243_243489


namespace maximum_value_of_x_plus_2y_l243_243008

theorem maximum_value_of_x_plus_2y (x y : ℝ) (h : x^2 - 2 * x + 4 * y = 5) : ∃ m, m = x + 2 * y ∧ m ≤ 9/2 := by
  sorry

end maximum_value_of_x_plus_2y_l243_243008


namespace smallest_four_digit_multiple_of_53_l243_243784

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l243_243784


namespace ceil_sqrt_225_eq_15_l243_243967

theorem ceil_sqrt_225_eq_15 : 
  ⌈Real.sqrt 225⌉ = 15 := 
by sorry

end ceil_sqrt_225_eq_15_l243_243967


namespace radius_of_circumscribed_circle_l243_243250

theorem radius_of_circumscribed_circle (r : ℝ) :
  let θ := Real.pi / 4
  let sector_angle := θ
  ∃ R : ℝ, (θ = Real.pi / 4) ∧ (R = r * Real.sec (Real.pi / 8)) := by {
  sorry
}

end radius_of_circumscribed_circle_l243_243250


namespace curves_and_line_l243_243461

theorem curves_and_line (α θ k : ℝ) (x y ρ : ℝ) 
  (hC1_param : x = 1 + cos α ∧ y = sin α) 
  (hC2_polar : ρ = 2 * sqrt 3 * sin θ) 
  (hline : ∀ t : ℝ, t ≠ 0 → (x = t * cos α ∧ y = t * sin α ∧ y = k * x)) :
  ∃ k max_AB,
    (∀ θ, ρ = 2 * cos θ) ∧ 
    (∀ (x y : ℝ), x^2 + y^2 - 2 * sqrt 3 * y = 0) ∧
    (max_AB = 4 ∧ k = -sqrt 3) :=
by sorry

end curves_and_line_l243_243461


namespace prove_a_eq_2_prove_distance_l243_243040

-- Define the conditions for the given lines
def l₁ (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y + 1 = 0
def l₂ (a : ℝ) (x y : ℝ) : ℝ := (a - 2) * x + y + a = 0

-- Define perpendicularity and parallelism for lines
def perp (a : ℝ) : Prop := a - 2 = 0
def paral (a : ℝ) : Prop := a - 3 * (a - 2) = 0 ∧ 3 * a - 1 ≠ 0

-- Define distance between lines (needs elaboration)
def line_distance (a : ℝ) : ℝ := |9 - 1| / Real.sqrt (3 ^ 2 + 3 ^ 2)

-- The proofs

-- Proof for part 1
theorem prove_a_eq_2 : ∀ a : ℝ, ∀ b : ℝ, (b = 0) → perp a → a = 2 :=
by
  intros a b hb hperp
  rw [perp] at hperp
  exact hperp

-- Proof for part 2
theorem prove_distance : ∀ a : ℝ, ∀ b : ℝ, (b = 3 ∧ paral a) → line_distance a = (4 * Real.sqrt 2) / 3 :=
by
  intros a b hparal
  cases hparal with hb hparal
  sorry

end prove_a_eq_2_prove_distance_l243_243040


namespace smallest_four_digit_divisible_by_53_l243_243701

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243701


namespace square_1023_l243_243905

theorem square_1023 : (1023 : ℤ)^2 = 1046529 :=
by
  let a := (10 : ℤ)^3
  let b := (23 : ℤ)
  have h1 : (1023 : ℤ) = a + b := by rfl
  have h2 : (a + b)^2 = a^2 + 2 * a * b + b^2 := by ring
  have h3 : a = 1000 := by rfl
  have h4 : b = 23 := by rfl
  have h5 : a^2 = 1000000 := by norm_num
  have h6 : 2 * a * b = 46000 := by norm_num
  have h7 : b^2 = 529 := by norm_num
  calc
    (1023 : ℤ)^2 = (a + b)^2 : by rw h1
    ... = a^2 + 2 * a * b + b^2 : by rw h2
    ... = 1000000 + 46000 + 529 : by rw [h5, h6, h7]
    ... = 1046529 : by norm_num

end square_1023_l243_243905


namespace inequality_pqr_l243_243485

theorem inequality_pqr (p q r : ℝ) (n : ℕ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (h : p * q * r = 1) :
  1 / (p^n + q^n + 1) + 1 / (q^n + r^n + 1) + 1 / (r^n + p^n + 1) ≤ 1 :=
sorry

end inequality_pqr_l243_243485


namespace coronavirus_transmission_l243_243558

theorem coronavirus_transmission:
  (∃ x: ℝ, (1 + x)^2 = 225) :=
by
  sorry

end coronavirus_transmission_l243_243558


namespace find_B_l243_243050

structure Point where
  x : Int
  y : Int

def vector_sub (p1 p2 : Point) : Point :=
  ⟨p1.x - p2.x, p1.y - p2.y⟩

def O : Point := ⟨0, 0⟩
def A : Point := ⟨-1, 2⟩
def BA : Point := ⟨3, 3⟩
def B : Point := ⟨-4, -1⟩

theorem find_B :
  vector_sub A BA = B :=
by
  sorry

end find_B_l243_243050


namespace bela_wins_game_l243_243884

theorem bela_wins_game (n : ℝ) (h : n > 5) : ∃ move_sequence : ℕ → ℝ, 
  (∀ m, 0 ≤ move_sequence m ∧ move_sequence m ≤ n) ∧
  (∀ i j, i ≠ j → abs (move_sequence i - move_sequence j) ≥ 1.5) ∧
  winning_move_sequence move_sequence :=
sorry

/-- Auxiliary definition indicating Bela's move sequence leads to a win. -/
def winning_move_sequence (move_sequence : ℕ → ℝ) : Prop :=
∀ i, 
  (¬ ∃ j, i ≠ j ∧ abs (move_sequence i - move_sequence j) < 1.5) →
  (∃ k, move_sequence k = move_sequence i + 1.5 ∨ move_sequence k = move_sequence i - 1.5)

end bela_wins_game_l243_243884


namespace smallest_four_digit_div_by_53_l243_243632

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l243_243632


namespace least_third_side_length_l243_243410

theorem least_third_side_length (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℝ, (c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2)) ∧ c = Real.sqrt 161 :=
by
  sorry

end least_third_side_length_l243_243410


namespace smallest_four_digit_divisible_by_53_l243_243680

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l243_243680


namespace least_side_is_8_l243_243397

-- Define the sides of the right triangle
variables (a b : ℝ) (h : a = 8) (k : b = 15)

-- Define a predicate for the least possible length of the third side
def least_possible_third_side (c : ℝ) : Prop :=
  (c = 8) ∨ (c = 15) ∨ (c = 17)

theorem least_side_is_8 (c : ℝ) (hc : least_possible_third_side c) : c = 8 :=
by
  sorry

end least_side_is_8_l243_243397


namespace eval_expression_l243_243908

def f (x : ℤ) : ℤ := 3 * x^2 - 6 * x + 10

theorem eval_expression : 3 * f 2 + 2 * f (-2) = 98 := by
  sorry

end eval_expression_l243_243908


namespace eddy_travel_time_l243_243924

theorem eddy_travel_time (T : ℝ) (S_e S_f : ℝ) (Freddy_time : ℝ := 4)
  (distance_AB : ℝ := 540) (distance_AC : ℝ := 300) (speed_ratio : ℝ := 2.4) :
  (distance_AB / T = 2.4 * (distance_AC / Freddy_time)) -> T = 3 :=
by
  sorry

end eddy_travel_time_l243_243924


namespace prime_power_seven_l243_243857

theorem prime_power_seven (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (eqn : p + 25 = q^7) : p = 103 := by
  sorry

end prime_power_seven_l243_243857


namespace smallest_four_digit_divisible_by_53_l243_243724

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l243_243724


namespace find_range_a_l243_243490

open Real

noncomputable def is_subset (M : set ℝ) (I : Icc 0 3) : Prop :=
  ∀ x, x ∈ M → x ∈ I

theorem find_range_a :
  ∀ a : ℝ,
  (is_subset { x : ℝ | x^2 + 2 * (1 - a) * x + 3 - a ≤ 0 } (Icc 0 3)) →
  (-1 ≤ a ∧ a ≤ (18/7)) :=
sorry

end find_range_a_l243_243490


namespace nanometers_to_meters_scientific_notation_l243_243566

theorem nanometers_to_meters_scientific_notation :
  (300 * (1e-8 : ℝ)) = 3 * (10 : ℝ) ^ (-6) :=
by
  -- Proof goes here
  sorry

end nanometers_to_meters_scientific_notation_l243_243566


namespace quadratic_passes_through_l243_243168

def quadratic_value_at_point (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_passes_through (a b c : ℝ) :
  quadratic_value_at_point a b c 1 = 5 ∧ 
  quadratic_value_at_point a b c 3 = n ∧ 
  a * (-2)^2 + b * (-2) + c = -8 ∧ 
  (-4*a + b = 0) → 
  n = 253/9 := 
sorry

end quadratic_passes_through_l243_243168


namespace sum_of_diff_squares_eq_5050_l243_243835

theorem sum_of_diff_squares_eq_5050 : 
  ∑ k in (range 50).map (λ n, (2 * (n + 1))^2 - (2 * n + 1)^2) = 5050 := 
by
  sorry

end sum_of_diff_squares_eq_5050_l243_243835


namespace minimum_value_at_a_half_a_range_if_f_max_not_exceed_3_l243_243126

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |Real.log (x + 1) / Real.log 25 - a| + 2 * a + 1

theorem minimum_value_at_a_half :
  f 4 (1 / 2) = 2 :=
sorry

theorem a_range_if_f_max_not_exceed_3 :
  (∀ x ∈ set.Icc (0 : ℝ) 24, f x a ≤ 3) → a ∈ set.Ioo 0 (2 / 3) :=
sorry

end minimum_value_at_a_half_a_range_if_f_max_not_exceed_3_l243_243126


namespace point_above_line_l243_243355

-- Define the point P with coordinates (-2, t)
variable (t : ℝ)

-- Define the line equation
def line_eq (x y : ℝ) : ℝ := 2 * x - 3 * y + 6

-- Proving that t must be greater than 2/3 for the point P to be above the line
theorem point_above_line : (line_eq (-2) t < 0) -> t > 2 / 3 :=
by
  sorry

end point_above_line_l243_243355


namespace gcd_3_pow_1007_minus_1_3_pow_1018_minus_1_l243_243601

theorem gcd_3_pow_1007_minus_1_3_pow_1018_minus_1 :
  Nat.gcd (3^1007 - 1) (3^1018 - 1) = 177146 :=
by
  -- Proof follows from the Euclidean algorithm and factoring, skipping the proof here.
  sorry

end gcd_3_pow_1007_minus_1_3_pow_1018_minus_1_l243_243601


namespace smallest_four_digit_divisible_by_53_l243_243695

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243695


namespace ceil_sqrt_225_eq_15_l243_243972

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := by
  sorry

end ceil_sqrt_225_eq_15_l243_243972


namespace smallest_four_digit_divisible_by_53_l243_243612

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l243_243612


namespace count_even_three_digit_numbers_l243_243047

theorem count_even_three_digit_numbers :
  let is_three_digit (n : ℕ) := 100 ≤ n ∧ n < 1000
      is_even (n : ℕ) := n % 2 = 0
      sum_tens_units_eq_11 (n : ℕ) := ((n / 10) % 10) + (n % 10) = 11
      is_hundreds_digit_odd (n : ℕ) := ((n / 100) % 10) % 2 = 1
  in (finset.filter (λ n, is_three_digit n ∧ is_even n ∧ sum_tens_units_eq_11 n ∧ is_hundreds_digit_odd n)
       (finset.range 1000)).card = 20 := by
  sorry

end count_even_three_digit_numbers_l243_243047


namespace evaluate_ceil_sqrt_225_l243_243945

def ceil (x : ℝ) : ℤ :=
  if h : ∃ n : ℤ, n ≤ x ∧ x < n + 1 then
    classical.some h
  else
    0

theorem evaluate_ceil_sqrt_225 : ceil (Real.sqrt 225) = 15 := 
sorry

end evaluate_ceil_sqrt_225_l243_243945


namespace max_tiles_on_floor_l243_243805

   -- Definitions corresponding to conditions
   def tile_length_1 : ℕ := 35
   def tile_width_1 : ℕ := 30
   def tile_length_2 : ℕ := 30
   def tile_width_2 : ℕ := 35
   def floor_length : ℕ := 1000
   def floor_width : ℕ := 210

   -- Conditions:
   -- 1. Tiles do not overlap.
   -- 2. Tiles are placed with edges jutting against each other on all edges.
   -- 3. A tile can be placed in any orientation so long as its edges are parallel to the edges of the floor.
   -- 4. No tile should overshoot any edge of the floor.

   theorem max_tiles_on_floor :
     let tiles_orientation_1 := (floor_length / tile_length_1) * (floor_width / tile_width_1)
     let tiles_orientation_2 := (floor_length / tile_length_2) * (floor_width / tile_width_2)
     max tiles_orientation_1 tiles_orientation_2 = 198 :=
   by {
     -- The actual proof handling is skipped, as per instructions.
     sorry
   }
   
end max_tiles_on_floor_l243_243805


namespace percentage_increase_l243_243429

-- Defining the conditions as Lean 4 statements
def propertyTax1995 := 1800
def surcharge1996 := 200
def totalTax1996 := 2108

-- Problem statement as a Lean 4 theorem
theorem percentage_increase :
  let preSurchargeTax1996 := totalTax1996 - surcharge1996 in
  let increase := preSurchargeTax1996 - propertyTax1995 in
  let percentageIncrease := (increase : Float) / propertyTax1995 * 100 in
  percentageIncrease = 6 :=
by
  sorry

end percentage_increase_l243_243429


namespace no_adjacent_green_hats_l243_243831

theorem no_adjacent_green_hats (n m : ℕ) (h₀ : n = 9) (h₁ : m = 3) : 
  (((1 : ℚ) - (9/14 : ℚ)) = (5/14 : ℚ)) :=
by
  rw h₀ at *,
  rw h₁ at *,
  sorry

end no_adjacent_green_hats_l243_243831


namespace ceil_sqrt_225_eq_15_l243_243930

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := 
by 
  sorry

end ceil_sqrt_225_eq_15_l243_243930


namespace number_of_valid_queen_arrangements_is_even_l243_243519

-- Definition of an 8x8 chessboard.
def chessboard := fin 8 × fin 8

-- Definition of a valid arrangement of queens
def is_valid_queen_arrangement (qs : set chessboard) : Prop :=
  qs.card = 8 ∧ ∀ (q1 q2 : chessboard), q1 ≠ q2 → (q1.1 = q2.1 ∨ q1.2 = q2.2 ∨ 
  abs (q1.1 - q2.1) = abs (q1.2 - q2.2)) → false

-- Reflect the arrangement across the vertical middle line
def reflect_vertically (qs : set chessboard) : set chessboard :=
  qs.image (λ ⟨x, y⟩, ⟨x, fin 7 - y⟩)

-- At the core, we need to prove that the number of valid arrangements is even:
theorem number_of_valid_queen_arrangements_is_even :
  even (card { qs : set chessboard | is_valid_queen_arrangement qs }) :=
sorry

end number_of_valid_queen_arrangements_is_even_l243_243519


namespace smallest_four_digit_multiple_of_53_l243_243763

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243763


namespace smallest_four_digit_multiple_of_53_l243_243716

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243716


namespace ceil_sqrt_225_eq_15_l243_243974

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := by
  sorry

end ceil_sqrt_225_eq_15_l243_243974


namespace cone_volume_is_320pi_l243_243183

-- Define the necessary geometric parameters of the cone
def slant_height : ℝ := 17
def height : ℝ := 15

-- Use Pythagorean theorem to define the radius
def radius : ℝ := Math.sqrt (slant_height ^ 2 - height ^ 2)

-- Calculate the volume of the cone
def volume : ℝ := (1 / 3) * π * radius ^ 2 * height

-- Statement to prove that the volume is 320π
theorem cone_volume_is_320pi : volume = 320 * π := by
  sorry

end cone_volume_is_320pi_l243_243183


namespace percent_of_decimal_l243_243595

theorem percent_of_decimal : (3 / 8 / 100) * 240 = 0.9 :=
by
  sorry

end percent_of_decimal_l243_243595


namespace proof_problem_l243_243483

-- Definitions for the lengths of the sides and distances from point P
variables {A B C P : Point}
variables {a b c a1 b1 c1 : ℝ}
variables (h_a : a = dist B C)
variables (h_b : b = dist C A)
variables (h_c : c = dist A B)
variables (h_a1 : a1 = dist P A)
variables (h_b1 : b1 = dist P B)
variables (h_c1 : c1 = dist P C)

-- The theorem to be proven
theorem proof_problem (h_abc_triangle : triangle A B C)
  (h_p_inside : inside_triangle P A B C)
  (h_a_def : a = dist B C)
  (h_b_def : b = dist C A)
  (h_c_def : c = dist A B)
  (h_a1_def : a1 = dist P A)
  (h_b1_def : b1 = dist P B)
  (h_c1_def : c1 = dist P C) :
  a * a1^2 + b * b1^2 + c * c1^2 ≥ a * b * c := sorry

end proof_problem_l243_243483


namespace height_of_sunflower_in_feet_l243_243497

def height_of_sister_in_feet : ℕ := 4
def height_of_sister_in_inches : ℕ := 3
def additional_height_of_sunflower : ℕ := 21

theorem height_of_sunflower_in_feet 
  (h_sister_feet : height_of_sister_in_feet = 4)
  (h_sister_inches : height_of_sister_in_inches = 3)
  (h_additional : additional_height_of_sunflower = 21) :
  (4 * 12 + 3 + 21) / 12 = 6 :=
by simp [h_sister_feet, h_sister_inches, h_additional]; norm_num; sorry

end height_of_sunflower_in_feet_l243_243497


namespace seven_digit_palindromes_l243_243366

def is_palindrome (l : List ℕ) : Prop :=
  l = l.reverse

theorem seven_digit_palindromes : 
  (∃ l : List ℕ, l = [1, 1, 4, 4, 4, 6, 6] ∧ 
  ∃ pl : List ℕ, pl.length = 7 ∧ is_palindrome pl ∧ 
  ∀ d, d ∈ pl → d ∈ l) →
  ∃! n, n = 12 :=
by
  sorry

end seven_digit_palindromes_l243_243366


namespace inequality_for_positive_real_numbers_l243_243139

theorem inequality_for_positive_real_numbers
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 < (a / (Real.sqrt (a^2 + b^2)) + b / (Real.sqrt (b^2 + c^2)) + c / (Real.sqrt (c^2 + a^2))) ∧ 
  (a / (Real.sqrt (a^2 + b^2)) + b / (Real.sqrt (b^2 + c^2)) + c / (Real.sqrt (c^2 + a^2))) ≤ (3 * Real.sqrt 3 / 2) := 
begin
  sorry
end

end inequality_for_positive_real_numbers_l243_243139


namespace probability_of_supporting_law_l243_243076

def prob_supports (p : ℝ) (k n : ℕ) : ℝ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_of_supporting_law : 
  prob_supports 0.6 2 5 = 0.2304 :=
sorry

end probability_of_supporting_law_l243_243076


namespace least_side_is_8_l243_243396

-- Define the sides of the right triangle
variables (a b : ℝ) (h : a = 8) (k : b = 15)

-- Define a predicate for the least possible length of the third side
def least_possible_third_side (c : ℝ) : Prop :=
  (c = 8) ∨ (c = 15) ∨ (c = 17)

theorem least_side_is_8 (c : ℝ) (hc : least_possible_third_side c) : c = 8 :=
by
  sorry

end least_side_is_8_l243_243396


namespace smallest_four_digit_div_by_53_l243_243627

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l243_243627


namespace history_book_pages_l243_243171

-- Conditions
def science_pages : ℕ := 600
def novel_pages (science: ℕ) : ℕ := science / 4
def history_pages (novel: ℕ) : ℕ := novel * 2

-- Theorem to prove
theorem history_book_pages : history_pages (novel_pages science_pages) = 300 :=
by
  sorry

end history_book_pages_l243_243171


namespace minimize_RS_l243_243105

-- Define the setup conditions for the problem
variables (A B C D M R S T O : Type)
variables [linear_ordered_field A] [metric_space A]

-- Rhombus ABCD with diagonals AC and BD
-- AC = 24 and BD = 40
def rhombus_diagonals : Prop :=
  let AC := 24
  let BD := 40
  let AO := AC / 2
  let BO := BD / 2
  let OD := BO
  AO = 12 ∧ BO = 20 ∧ OD = 20

-- M is a point on AB
variable (M_on_AB : Prop)

-- R and S are feet of perpendiculars from M to AC and BD, respectively
variable (R_S_perpendiculars : Prop)

-- Circle centered at D with radius 20 intersects BD again at T
variable (circle_intersection : Prop)

-- M, R, and S are collinear with T
variable (collinear_M_R_S_T : Prop)

-- Definitions that encapsulate the problem conditions
def problem_conditions : Prop :=
  rhombus_diagonals A B C D ∧
  M_on_AB ∧
  R_S_perpendiculars ∧
  circle_intersection ∧
  collinear_M_R_S_T

-- Define the minimum possible value of RS to be approximately 18.75
def minimum_RS : Prop :=
  let BM := 2.5
  let RS := 5 * real.sqrt 14
  abs (RS - 18.75) < 1e-2

-- The final problem statement: Prove that RS is minimized to be approximately 18.75 under given conditions
theorem minimize_RS : problem_conditions A B C D M R S T O → minimum_RS A B C D M R S T O :=
sorry

end minimize_RS_l243_243105


namespace horner_evaluation_at_2_l243_243890

noncomputable def f : ℕ → ℕ :=
  fun x => (((2 * x + 3) * x + 0) * x + 5) * x - 4

theorem horner_evaluation_at_2 : f 2 = 14 :=
  by
    sorry

end horner_evaluation_at_2_l243_243890


namespace suff_but_not_nec_condition_l243_243559

theorem suff_but_not_nec_condition (x : ℝ) : (1 < x ∧ x < π / 2) → ((x - 1) * tan x > 0) ∧ ¬((x - 1) * tan x > 0 → 1 < x ∧ x < π / 2) :=
by
  sorry

end suff_but_not_nec_condition_l243_243559


namespace ceil_sqrt_225_l243_243935

theorem ceil_sqrt_225 : Nat.ceil (Real.sqrt 225) = 15 :=
by
  sorry

end ceil_sqrt_225_l243_243935


namespace shannon_heart_stones_l243_243147

theorem shannon_heart_stones (total_stones : ℝ) (total_bracelets : ℝ) (stones_per_bracelet : ℝ) :
  total_stones = 48.0 → total_bracelets = 6 → stones_per_bracelet = total_stones / total_bracelets → stones_per_bracelet = 8 :=
by {
  intros,
  rw [← h, ← h_1],
  simp [← h_2],
  norm_num
}

end shannon_heart_stones_l243_243147


namespace systematic_sampling_first_group_number_l243_243253

-- Given conditions
def total_students := 160
def group_size := 8
def groups := total_students / group_size
def number_in_16th_group := 126

-- Theorem Statement
theorem systematic_sampling_first_group_number :
  ∃ x : ℕ, (120 + x = number_in_16th_group) ∧ x = 6 :=
by
  -- Proof can be filled here
  sorry

end systematic_sampling_first_group_number_l243_243253


namespace tangency_at_origin_range_of_a_for_extrema_l243_243034

-- Part 1: Tangency at the origin
theorem tangency_at_origin (a : ℝ) (h_a : a = 1) : (let f := λ x : ℝ, (Real.exp x - a * x^2 - Real.cos x - Real.log (x + 1)) in
  f 0 = 0 ∧ deriv f 0 = 0) := sorry

-- Part 2: Range of a for exactly one extremum in each interval
theorem range_of_a_for_extrema (a : ℝ) :
  (let f := λ x : ℝ, (Real.exp x - a * x^2 - Real.cos x - Real.log (x + 1)) in
  (∃ x ∈ Ioo (-1 : ℝ) 0, deriv f x = 0) ∧
  (∃ x ∈ Ioo 0 +∞, deriv f x = 0)) ↔ a ∈ Ioo (3 / 2 : ℝ) +∞ := sorry

end tangency_at_origin_range_of_a_for_extrema_l243_243034


namespace find_d_l243_243480

variable {x1 x2 k d : ℝ}

axiom h₁ : x1 ≠ x2
axiom h₂ : 4 * x1^2 - k * x1 = d
axiom h₃ : 4 * x2^2 - k * x2 = d
axiom h₄ : x1 + x2 = 2

theorem find_d : d = -12 := by
  sorry

end find_d_l243_243480


namespace poly_coeff_divisible_by_prime_l243_243470

variable {p : ℕ} [fact p.prime]
variable {a b c d : ℤ}
variable {x1 x2 x3 x4 : ℤ}

def Q (x : ℤ) : ℤ :=
  a * x^3 + b * x^2 + c * x + d
  
theorem poly_coeff_divisible_by_prime
  (h0 : x1 ≠ x2) (h1 : x1 ≠ x3) (h2 : x1 ≠ x4)
  (h3 : x2 ≠ x3) (h4 : x2 ≠ x4) (h5 : x3 ≠ x4)
  (hx1 : 0 ≤ x1 ∧ x1 < p) (hx2 : 0 ≤ x2 ∧ x2 < p)
  (hx3 : 0 ≤ x3 ∧ x3 < p) (hx4 : 0 ≤ x4 ∧ x4 < p)
  (hQ1 : p ∣ Q x1) (hQ2 : p ∣ Q x2) (hQ3 : p ∣ Q x3) (hQ4 : p ∣ Q x4) :
  p ∣ a ∧ p ∣ b ∧ p ∣ c ∧ p ∣ d := by
  sorry

end poly_coeff_divisible_by_prime_l243_243470


namespace range_of_independent_variable_l243_243454

theorem range_of_independent_variable (x : ℝ) (y : ℝ) (h : y = sqrt (x + 2)) : x ≥ -2 :=
sorry

end range_of_independent_variable_l243_243454


namespace prime_condition_l243_243858

def is_prime (n : ℕ) : Prop := nat.prime n

theorem prime_condition (p : ℕ) (q : ℕ) (h_prime_p : is_prime p) (h_eq : p + 25 = q ^ 7) (h_prime_q : is_prime q) : p = 103 :=
sorry

end prime_condition_l243_243858


namespace jerry_age_l243_243505

theorem jerry_age (M J : ℕ) (h1 : M = 20) (h2 : M = 2 * J - 8) : J = 14 := 
by
  sorry

end jerry_age_l243_243505


namespace marley_total_fruits_l243_243492

theorem marley_total_fruits (louis_oranges : ℕ) (louis_apples : ℕ) 
                            (samantha_oranges : ℕ) (samantha_apples : ℕ)
                            (marley_oranges : ℕ) (marley_apples : ℕ) : 
  (louis_oranges = 5) → (louis_apples = 3) → 
  (samantha_oranges = 8) → (samantha_apples = 7) → 
  (marley_oranges = 2 * louis_oranges) → (marley_apples = 3 * samantha_apples) → 
  (marley_oranges + marley_apples = 31) :=
by
  intros
  sorry

end marley_total_fruits_l243_243492


namespace distance_is_correct_l243_243252

noncomputable def distance_from_center_to_plane
  (O : Point)
  (radius : ℝ)
  (vertices : Point × Point × Point)
  (side_lengths : (ℝ × ℝ × ℝ)) :
  ℝ :=
  8.772

theorem distance_is_correct
  (O : Point)
  (radius : ℝ)
  (A B C : Point)
  (h_radius : radius = 10)
  (h_sides : side_lengths = (17, 17, 16))
  (vertices := (A, B, C)) :
  distance_from_center_to_plane O radius vertices side_lengths = 8.772 := by
  sorry

end distance_is_correct_l243_243252


namespace probability_triangle_or_hexagon_l243_243428

theorem probability_triangle_or_hexagon 
  (total_shapes : ℕ) 
  (num_triangles : ℕ) 
  (num_squares : ℕ) 
  (num_circles : ℕ) 
  (num_hexagons : ℕ)
  (htotal : total_shapes = 10)
  (htriangles : num_triangles = 3)
  (hsquares : num_squares = 4)
  (hcircles : num_circles = 2)
  (hhexagons : num_hexagons = 1):
  (num_triangles + num_hexagons) / total_shapes = 2 / 5 := 
by 
  sorry

end probability_triangle_or_hexagon_l243_243428


namespace smallest_four_digit_multiple_of_53_l243_243781

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l243_243781


namespace remainder_of_polynomial_l243_243473

noncomputable def P : ℕ → ℝ := sorry

theorem remainder_of_polynomial :
  (∀ x : ℝ, P(x) = (x - 19) * (x - 99) * Q(x) + (-x + 118)) :=
by
  have h1 : P 19 = 99 := sorry
  have h2 : P 99 = 19 := sorry
  sorry

end remainder_of_polynomial_l243_243473


namespace irrational_count_is_two_l243_243872

noncomputable def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

noncomputable def number_of_irrationals : ℤ :=
  let nums := [0, (9 : ℝ) ^ (1 / 3), -3.1415, (4 : ℝ) ^ (1 / 2), 22 / 7, 
               Real.mk (Seq.mk (fun n => if n % 4 = 0 then 4 else 3) 0)] in
  nums.count is_irrational

theorem irrational_count_is_two : number_of_irrationals = 2 := by
  sorry

end irrational_count_is_two_l243_243872


namespace min_value_of_expression_l243_243987

theorem min_value_of_expression 
  (x y : ℝ) 
  (h : 3 * |x - y| + |2 * x - 5| = x + 1) : 
  ∃ (x y : ℝ), 2 * x + y = 4 :=
by {
  sorry
}

end min_value_of_expression_l243_243987


namespace correctness_of_statements_l243_243797

theorem correctness_of_statements (A_condition B_condition C_condition D_condition : Prop) 
  (hA : ∀ x, ¬ (x > 2 → x > 3))
  (hB : ∀ x y, (x > y ↔ x^3 > y^3))
  (hC : ∀ x y, (x > y → ¬ (x^2 > y^2)) ∧ (x^2 > y^2 → ¬ (x > y)))
  (hD : ∀ x y, (x > y ↔ 2^x > 2^y)) :
  B_condition ∧ C_condition :=
by 
  exact ⟨hB, hC⟩.

end correctness_of_statements_l243_243797


namespace smallest_four_digit_div_by_53_l243_243628

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l243_243628


namespace complex_product_l243_243837

theorem complex_product (i : ℂ) (h : i^2 = -1) :
  (3 - 4 * i) * (2 + 7 * i) = 34 + 13 * i :=
sorry

end complex_product_l243_243837


namespace percent_of_240_l243_243592

theorem percent_of_240 (h : (3 / 8 / 100 : ℝ) = 3 / 800) : 
  (3 / 800 * 240 = 0.9) :=
begin
  sorry
end

end percent_of_240_l243_243592


namespace symmetric_with_origin_l243_243174

-- Define the original point P
def P : ℝ × ℝ := (2, -3)

-- Define the function for finding the symmetric point with respect to the origin
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- Prove that the symmetric point of P with respect to the origin is (-2, 3)
theorem symmetric_with_origin :
  symmetric_point P = (-2, 3) :=
by
  -- Placeholders for proof
  sorry

end symmetric_with_origin_l243_243174


namespace car_trader_percentage_increase_l243_243864

theorem car_trader_percentage_increase :
  ∀ (P : ℝ), (0 < P) →
  let CP := 0.7 * P in
  let SP := 1.1899999999999999 * P in
  let percentage_increase := ((SP - CP) / CP) * 100 in
  percentage_increase = 70 :=
by
  intro P
  intro hP
  let CP := 0.7 * P
  let SP := 1.1899999999999999 * P
  let percentage_increase := ((SP - CP) / CP) * 100
  sorry

end car_trader_percentage_increase_l243_243864


namespace triangle_areas_sum_l243_243198

noncomputable def circumradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (a * b * c) / (4 * K)

theorem triangle_areas_sum (a b c : ℝ) (h1 : AB = 2) (h2 : BC = 3) (h3 : CA = 4) 
  (h4 : circumcenter O) (h5 : gcd(a, c) = 1) (h6 : ¬∃ p: ℕ, p.prime ∧ p^2 ∣ b) :
  a = 77 ∧ b = 15 ∧ c = 60 → a + b + c = 152 :=
by
  sorry

end triangle_areas_sum_l243_243198


namespace range_of_a_if_f_increasing_l243_243061

theorem range_of_a_if_f_increasing (a : ℝ) :
  (∀ x : ℝ, 3*x^2 + 3*a ≥ 0) → (a ≥ 0) :=
sorry

end range_of_a_if_f_increasing_l243_243061


namespace T1_T2_T3_l243_243002

variables (pib : Type) (maa : Type)
variable (belongs_to : maa → pib → Prop)

-- Given postulates
axiom P1 : ∀ p : pib, ∃ M : set maa, ∀ m ∈ M, belongs_to m p
axiom P2 : ∀ p1 p2 : pib, p1 ≠ p2 → ∃! m : maa, belongs_to m p1 ∧ belongs_to m p2
axiom P3 : ∀ m : maa, ∃! p1 p2 : pib, p1 ≠ p2 ∧ belongs_to m p1 ∧ belongs_to m p2
axiom P4 : ∃! P : set pib, set.card P = 5

-- Theorems to be proved
theorem T1 : ∃! M : set maa, set.card M = 10 :=
sorry

theorem T2 : ∀ p : pib, ∃ (M : set maa), set.card M = 4 ∧ (∀ m ∈ M, belongs_to m p) :=
sorry

theorem T3 : ∀ m1 : maa, ∃! m2 : maa, (∀ p : pib, ¬(belongs_to m1 p ∧ belongs_to m2 p)) :=
sorry

end T1_T2_T3_l243_243002


namespace range_contains_pi_div_4_l243_243980

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_contains_pi_div_4 : ∃ x : ℝ, f x = (Real.pi / 4) := by
  sorry

end range_contains_pi_div_4_l243_243980


namespace probability_two_heads_two_tails_l243_243375

theorem probability_two_heads_two_tails :
  (prob_of_two_heads_two_tails : ℚ) =
  (combinations_of_4_choose_2 * prob_of_each_sequence : ℚ) :=
by
  let prob_of_each_sequence : ℚ := (1/2)^4
  let combinations_of_4_choose_2 : ℚ := 6 -- This comes from combinatorial calculation C(4,2) = 6
  have prob_of_two_heads_two_tails := combinations_of_4_choose_2 * prob_of_each_sequence
  show prob_of_two_heads_two_tails = (3/8 : ℚ)
  sorry

end probability_two_heads_two_tails_l243_243375


namespace projection_of_a_minus_b_l243_243007

variables {α : Type*} [InnerProductSpace ℝ α]

theorem projection_of_a_minus_b (a b : α)
  (ha : a ≠ 0) (hb : b ≠ 0)
  (h₁ : ∥a + b∥ = ∥a - b∥)
  (h₂ : ∥a + b∥ = 2)
  (h₃ : 2 * ∥b∥ = 2) :
  (a - b) ⬝ b = -1 :=
by
  sorry

end projection_of_a_minus_b_l243_243007


namespace sufficient_not_necessary_condition_l243_243333

variables (a b c : ℝ)

theorem sufficient_not_necessary_condition (h : a > b) (hc : c ≠ 0) : ac^2 > bc^2 :=
sorry

end sufficient_not_necessary_condition_l243_243333


namespace least_third_side_of_right_triangle_l243_243382

theorem least_third_side_of_right_triangle (a b : ℕ) (ha : a = 8) (hb : b = 15) :
  ∃ c : ℝ, c = real.sqrt (b^2 - a^2) ∧ c = real.sqrt 161 :=
by {
  -- We state the conditions
  have h8 : (8 : ℝ) = a, from by {rw ha},
  have h15 : (15 : ℝ) = b, from by {rw hb},

  -- The theorem states that such a c exists
  use (real.sqrt (15^2 - 8^2)),

  -- We need to show the properties of c
  split,
  { 
    -- Showing that c is the sqrt of the difference of squares of b and a
    rw [←h15, ←h8],
    refl 
  },
  {
    -- Showing that c is sqrt(161)
    calc
       real.sqrt (15^2 - 8^2)
         = real.sqrt (225 - 64) : by norm_num
     ... = real.sqrt 161 : by norm_num
  }
}
sorry

end least_third_side_of_right_triangle_l243_243382


namespace range_of_a_l243_243163

noncomputable def has_solutions (a : ℝ) : Prop :=
  ∀ x : ℝ, 2 * a * 9^(Real.sin x) + 4 * a * 3^(Real.sin x) + a - 8 = 0

theorem range_of_a : ∀ a : ℝ,
  (has_solutions a ↔ (8 / 31 <= a ∧ a <= 72 / 23)) := sorry

end range_of_a_l243_243163


namespace count_valid_4_digit_numbers_l243_243045

def valid_4_digit_numbers (a b c d : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  d ≥ 3 * c

theorem count_valid_4_digit_numbers : ∃ n, n = 1890 ∧ 
  n = ∑ a in Finset.range 10, ∑ b in Finset.range 10, ∑ c in Finset.range 10, ∑ d in Finset.range 10, 
    if valid_4_digit_numbers a b c d then 1 else 0 := 
by
  let n := ∑ a in Finset.range 10, ∑ b in Finset.range 10, ∑ c in Finset.range 10, ∑ d in Finset.range 10, 
             if valid_4_digit_numbers a b c d then 1 else 0
  use n
  have h : n = 1890 :=
  sorry
  exact ⟨h⟩

end count_valid_4_digit_numbers_l243_243045


namespace square_1023_l243_243903

theorem square_1023 : (1023 : ℤ)^2 = 1046529 :=
by
  let a := (10 : ℤ)^3
  let b := (23 : ℤ)
  have h1 : (1023 : ℤ) = a + b := by rfl
  have h2 : (a + b)^2 = a^2 + 2 * a * b + b^2 := by ring
  have h3 : a = 1000 := by rfl
  have h4 : b = 23 := by rfl
  have h5 : a^2 = 1000000 := by norm_num
  have h6 : 2 * a * b = 46000 := by norm_num
  have h7 : b^2 = 529 := by norm_num
  calc
    (1023 : ℤ)^2 = (a + b)^2 : by rw h1
    ... = a^2 + 2 * a * b + b^2 : by rw h2
    ... = 1000000 + 46000 + 529 : by rw [h5, h6, h7]
    ... = 1046529 : by norm_num

end square_1023_l243_243903


namespace height_of_sunflower_in_feet_l243_243496

def height_of_sister_in_feet : ℕ := 4
def height_of_sister_in_inches : ℕ := 3
def additional_height_of_sunflower : ℕ := 21

theorem height_of_sunflower_in_feet 
  (h_sister_feet : height_of_sister_in_feet = 4)
  (h_sister_inches : height_of_sister_in_inches = 3)
  (h_additional : additional_height_of_sunflower = 21) :
  (4 * 12 + 3 + 21) / 12 = 6 :=
by simp [h_sister_feet, h_sister_inches, h_additional]; norm_num; sorry

end height_of_sunflower_in_feet_l243_243496


namespace no_constant_term_in_expansion_l243_243600

theorem no_constant_term_in_expansion : 
  ∀ (x : ℂ), ¬ ∃ (k : ℕ), ∃ (c : ℂ), c * x ^ (k / 3 - 2 * (12 - k)) = 0 :=
by sorry

end no_constant_term_in_expansion_l243_243600


namespace basin_capacity_l243_243838

-- Defining the flow rate of water into the basin
def inflow_rate : ℕ := 24

-- Defining the leak rate of the basin
def leak_rate : ℕ := 4

-- Defining the time taken to fill the basin in seconds
def fill_time : ℕ := 13

-- Net rate of filling the basin
def net_rate : ℕ := inflow_rate - leak_rate

-- Volume of the basin
def basin_volume : ℕ := net_rate * fill_time

-- The goal is to prove that the volume of the basin is 260 gallons
theorem basin_capacity : basin_volume = 260 := by
  sorry

end basin_capacity_l243_243838


namespace circumcircle_tangent_OP_l243_243119

-- Defining the necessary points and lines
variables {A B C O P O_A O_B O_C L_A L_B L_C : Point}
variables {Omega : Circle}
variables {l_A l_B l_C : Line}

-- Define the conditions as hypotheses
axiom circumcenter_O : is_circumcenter O (triangle A B C)
axiom circumcircle_Omega : is_circumcircle Omega (triangle A B C)
axiom point_P_on_Omega : lies_on_circle P Omega
axiom P_not_A_B_C_antipodes : P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ ¬antipode P A Omega ∧ ¬antipode P B Omega ∧ ¬antipode P C Omega
axiom circumcenters : is_circumcenter O_A (triangle A O P) ∧ is_circumcenter O_B (triangle B O P) ∧ is_circumcenter O_C (triangle C O P)
axiom perp_lines : perpendicular l_A (line_through B C) ∧ perpendicular l_B (line_through A C) ∧ perpendicular l_C (line_through A B)
axiom passes_through_circumcenters : passes_through l_A O_A ∧ passes_through l_B O_B ∧ passes_through l_C O_C

-- The proof statement
theorem circumcircle_tangent_OP (h : circumcenter_O ∧ circumcircle_Omega ∧ point_P_on_Omega ∧ P_not_A_B_C_antipodes ∧ circumcenters ∧ perp_lines ∧ passes_through_circumcenters) :
  tangent (circumcircle (triangle l_A l_B l_C)) (line_through O P) :=
sorry

end circumcircle_tangent_OP_l243_243119


namespace seating_arrangement_l243_243301

theorem seating_arrangement (boys : Fin 5 → Nat) (girls : Fin 4 → Nat) :
    (∃ (Pboys : ∀ i, boys i ∈ {1, 3, 5, 7, 9}) (Pgirls : ∀ i, girls i ∈ {2, 4, 6, 8}),
    (∃! pboys : Permutations (Fin 5), ∀ i, pboys i = boys i) ∧
    (∃! pgirls : Permutations (Fin 4), ∀ i, pgirls i = girls i)) →
    (∏ i, Permutations (Fin 5) × ∏ i, Permutations (Fin 4) = 2880) :=
by
  sorry

end seating_arrangement_l243_243301


namespace cubic_polynomial_solution_l243_243986

noncomputable def q (x : ℝ) : ℝ := - (4 / 3) * x^3 + 6 * x^2 - (50 / 3) * x - (14 / 3)

theorem cubic_polynomial_solution :
  q 1 = -8 ∧ q 2 = -12 ∧ q 3 = -20 ∧ q 4 = -40 := by
  have h₁ : q 1 = -8 := by sorry
  have h₂ : q 2 = -12 := by sorry
  have h₃ : q 3 = -20 := by sorry
  have h₄ : q 4 = -40 := by sorry
  exact ⟨h₁, h₂, h₃, h₄⟩

end cubic_polynomial_solution_l243_243986


namespace clock_angle_at_3_40_is_2278_radians_l243_243206

-- Definition of degrees to radians conversion factor
def deg_to_rad (deg : ℝ) : ℝ := (deg * Real.pi) / 180

-- Definition of the angle at a given time
def angle_at_time (hours minutes : ℝ) (hour_hand_initial_deg minute_hand_deg_per_minute hour_hand_deg_per_minute : ℝ) : ℝ :=
  let hour_hand_angle := hour_hand_initial_deg + (minutes * hour_hand_deg_per_minute)
  let minute_hand_angle := minutes * minute_hand_deg_per_minute
  let angle_diff := abs (minute_hand_angle - hour_hand_angle)
  let smaller_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff
  deg_to_rad smaller_angle

-- Main statement
theorem clock_angle_at_3_40_is_2278_radians :
  let hours := 3
  let minutes := 40
  let hour_hand_initial_deg := 90   -- at 3:00
  let minute_hand_deg_per_minute := 6
  let hour_hand_deg_per_minute := 0.5
  abs(angle_at_time hours minutes hour_hand_initial_deg minute_hand_deg_per_minute hour_hand_deg_per_minute - 2.278) < 0.001 :=
by {
  sorry -- proof will need to demonstrate calculation
}

end clock_angle_at_3_40_is_2278_radians_l243_243206


namespace Marissa_sunflower_height_l243_243503

-- Define the necessary conditions
def sister_height_feet : ℕ := 4
def sister_height_inches : ℕ := 3
def extra_sunflower_height : ℕ := 21
def inches_per_foot : ℕ := 12

-- Calculate the total height of the sister in inches
def sister_total_height_inch : ℕ := (sister_height_feet * inches_per_foot) + sister_height_inches

-- Calculate the sunflower height in inches
def sunflower_height_inch : ℕ := sister_total_height_inch + extra_sunflower_height

-- Convert the sunflower height to feet
def sunflower_height_feet : ℕ := sunflower_height_inch / inches_per_foot

-- The theorem we want to prove
theorem Marissa_sunflower_height : sunflower_height_feet = 6 := by
  sorry

end Marissa_sunflower_height_l243_243503


namespace find_F_58_59_60_l243_243241

def F : ℤ → ℤ → ℤ → ℝ := sorry

axiom F_scaling (a b c n : ℤ) : F (n * a) (n * b) (n * c) = n * F a b c
axiom F_shift (a b c n : ℤ) : F (a + n) (b + n) (c + n) = F a b c + n
axiom F_symmetry (a b c : ℤ) : F a b c = F c b a

theorem find_F_58_59_60 : F 58 59 60 = 59 :=
sorry

end find_F_58_59_60_l243_243241


namespace log_sum_is_10_l243_243457

variable {a : ℕ → ℝ}
variable (geom_seq : ∀ n m, a (n + m) = a n * a (m + n))
variable (a_pos : ∀ n, 0 < a n)
variable (cond : a 5 * a 6 = 9)

theorem log_sum_is_10 : (Finset.range 10).sum (λ n, Real.logBase 3 (a n + 1)) = 10 :=
  sorry

end log_sum_is_10_l243_243457


namespace checkPrice_l243_243845

def totalBooks := 120
def unsoldBooks := 40
def totalAmountReceived := 280.00000000000006 (default: 280.)
def booksSold := (2/3) * totalBooks  -- 80 books sold
def pricePerBook := (totalAmountReceived / booksSold) -- 3.5000000000000007

theorem checkPrice :
  pricePerBook = 3.5000000000000007 :=
by
  -- sorry can be used to indicate that the proof is omitted
  sorry

end checkPrice_l243_243845


namespace geometric_series_problem_l243_243460

theorem geometric_series_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ)
  (h_seq : ∀ n, a n + a (n + 1) = 3 * 2^n) :
  S (k + 2) - 2 * S (k + 1) + S k = 2^(k + 1) :=
sorry

end geometric_series_problem_l243_243460


namespace table_tennis_arrangements_l243_243435

-- Define the main problem setup
def problem_setup :=
  let total_players := 10
  let main_players := 3
  let participants := 5
  let remaining_players := total_players - main_players
  let main_positions := 3
  let remaining_positions := participants - main_positions
  remaining_players = 7 ∧ remaining_positions = 2

-- Define the computation for arrangements
def arrangements (main_players: ℕ) (remaining_players: ℕ) (main_positions: ℕ) (remaining_positions: ℕ) : ℕ :=
  nat.perm main_players main_positions * nat.perm remaining_players remaining_positions

-- State the theorem to be proved
theorem table_tennis_arrangements : problem_setup → arrangements 3 7 3 2 = 252 :=
by
  intro h
  sorry

end table_tennis_arrangements_l243_243435


namespace smallest_four_digit_multiple_of_53_l243_243762

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243762


namespace symmetric_lines_l243_243416

theorem symmetric_lines (a b : ℝ) :
  (∀ x y, ax - y + 2 = 0 → x - a * y + 2 = 0)
  ∧ (∀ x y, 3x - y - b = 0 → y - 3 * x + b = 0)
  → a = 1 / 3 ∧ b = 6 :=
by simp

end symmetric_lines_l243_243416


namespace smallest_four_digit_multiple_of_53_l243_243705

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243705


namespace speed_of_goods_train_l243_243242

variable (length_train length_platform time_crossing : ℝ)
variable (speed_kmph : ℝ)

-- Conditions
def condition1 := length_train = 440
def condition2 := length_platform = 80
def condition3 := time_crossing = 26
def condition4 := speed_kmph = 72

-- Definition for total distance covered
def total_distance := length_train + length_platform

-- Definition for speed in m/s
def speed_mps := total_distance / time_crossing

-- Conversion factor
def conversion_factor := 3.6

-- Definition for speed in km/hr
def speed_in_kmph := speed_mps * conversion_factor

-- Theorem to prove
theorem speed_of_goods_train : 
  condition1 → condition2 → condition3 → (speed_in_kmph = speed_kmph) := by
  sorry

end speed_of_goods_train_l243_243242


namespace increasing_log_func_on_interval_l243_243063

theorem increasing_log_func_on_interval (a : ℝ) (h1 : ∀ x ∈ Icc 2 4, (ax^2 - x)' > 0) : a > 1 := 
sorry

end increasing_log_func_on_interval_l243_243063


namespace swimming_pool_surface_area_l243_243860

def length : ℝ := 20
def width : ℝ := 15

theorem swimming_pool_surface_area : length * width = 300 := 
by
  -- The mathematical proof would go here; we'll skip it with "sorry" per instructions.
  sorry

end swimming_pool_surface_area_l243_243860


namespace initial_amount_is_750_l243_243262

-- Define the given conditions as variables or constants
variables (rate : ℝ) (final_amount : ℝ) (time : ℝ)
-- Assume the conditions specified in the problem
axiom h1 : rate = 0.05
axiom h2 : final_amount = 900
axiom h3 : time = 4

-- Define the formula and the principal
noncomputable def principal (P : ℝ) : Prop :=
  final_amount = P + (P * rate * time)

-- The statement to be proved
theorem initial_amount_is_750 : ∃ P, principal P ∧ P = 750 :=
by {
  exact sorry
}

end initial_amount_is_750_l243_243262


namespace smallest_four_digit_divisible_by_53_l243_243641

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l243_243641


namespace converging_to_equilateral_l243_243998

noncomputable def complex_rotation (θ : ℝ) : ℂ := complex.exp (complex.I * θ)
def epsilon : ℂ := complex_rotation (real.pi / 3)  -- 60 degrees rotation

structure Triangle :=
(a b c : ℂ)

def next_triangle (T : Triangle) : Triangle :=
let a' := 0.5 * (T.a + (1 - epsilon) * T.b + epsilon * T.c) in
let b' := 0.5 * (epsilon * T.a + T.b + (1 - epsilon) * T.c) in
let c' := 0.5 * ((1 - epsilon) * T.a + epsilon * T.b + T.c) in
⟨a', b', c'⟩

theorem converging_to_equilateral (A₁ B₁ C₁ : ℂ)
    (initial_triangle : Triangle := ⟨A₁, B₁, C₁⟩) :
    ∃ (limit_triangle : Triangle), 
    (∀ n, next_triangle^[n] initial_triangle = limit_triangle) ∧
    (limit_triangle.a - limit_triangle.b = (limit_triangle.c - limit_triangle.b) * epsilon) :=
sorry

end converging_to_equilateral_l243_243998


namespace compare_fractions_l243_243795

theorem compare_fractions:
  let m := 23 ^ 1973 in 
  (23 ^ 1873 + 1) / (23 * m + 1) > (23 * m + 1) / (529 * m + 1) :=
by sorry

end compare_fractions_l243_243795


namespace cats_given_by_Mr_Sheridan_l243_243133

-- Definitions of the initial state and final state
def initial_cats : Nat := 17
def total_cats : Nat := 31

-- Proof statement that Mr. Sheridan gave her 14 cats
theorem cats_given_by_Mr_Sheridan : total_cats - initial_cats = 14 := by
  sorry

end cats_given_by_Mr_Sheridan_l243_243133


namespace probability_no_two_green_hats_next_to_each_other_l243_243826

open Nat

def choose (n k : ℕ) : ℕ := Nat.fact n / (Nat.fact k * Nat.fact (n - k))

def total_ways_to_choose (n k : ℕ) : ℕ :=
  choose n k

def event_A (n : ℕ) : ℕ := n - 2

def event_B (n k : ℕ) : ℕ := choose (n - k + 1) 2 * (k - 2)

def probability_no_two_next_to_each_other (n k : ℕ) : ℚ :=
  let total_ways := total_ways_to_choose n k
  let event_A_ways := event_A (n)
  let event_B_ways := event_B n 3
  let favorable_ways := total_ways - (event_A_ways + event_B_ways)
  favorable_ways / total_ways

-- Given the conditions of 9 children and choosing 3 to wear green hats
theorem probability_no_two_green_hats_next_to_each_other (p : probability_no_two_next_to_each_other 9 3 = 5 / 14) : Prop := by
  sorry

end probability_no_two_green_hats_next_to_each_other_l243_243826


namespace player1_cannot_guarantee_win_l243_243450

def cell := (ℕ, ℕ)
def grid := fin 7 × fin 7

-- Directions in which players can move
inductive direction : Type
| up : direction
| down : direction
| left : direction
| right : direction

-- Move to an adjacent cell
def move (d : direction) (pos : grid) : option grid :=
  match d, pos with
  | direction.up, (⟨x, hx⟩, ⟨y, hy⟩) := if x = 0 then none else some (⟨x - 1, Nat.pred_lt hx⟩, ⟨y, hy⟩)
  | direction.down, (⟨x, hx⟩, ⟨y, hy⟩) := if x = 6 then none else some (⟨x + 1, Nat.add_lt_of_lt hx⟩, ⟨y, hy⟩)
  | direction.left, (⟨x, hx⟩, ⟨y, hy⟩) := if y = 0 then none else some (⟨x, hx⟩, ⟨y - 1, Nat.pred_lt hy⟩)
  | direction.right, (⟨x, hx⟩, ⟨y, hy⟩) := if y = 6 then none else some (⟨x, hx⟩, ⟨y + 1, Nat.add_lt_of_lt hy⟩)
  end

def initial_pos : grid := (⟨3, Nat.lt_succ_self 3⟩, ⟨3, Nat.lt_succ_self 3⟩)

-- Define the game's rules and properties
inductive turn : Type
| p1 : turn
| p2 : turn

-- Check if a turn results in a valid move
def valid_move (t : turn) (prev_dir : direction) (new_dir : direction) : Prop :=
  match t with
  | turn.p1 := new_dir = prev_dir ∨ new_dir = direction.left
  | turn.p2 := new_dir = prev_dir ∨ new_dir = direction.right
  end

-- Determine the winner 
def winner (last_turn : turn) (pos : grid) : Prop :=
  (pos = (⟨0, Nat.zero_lt_succ 6⟩, ⟨0, Nat.zero_lt_succ 6⟩) ∨
   pos = (⟨0, Nat.zero_lt_succ 6⟩, ⟨6, Nat.lt_succ_self 6⟩) ∨
   pos = (⟨6, Nat.lt_succ_self 6⟩, ⟨0, Nat.zero_lt_succ 6⟩) ∨
   pos = (⟨6, Nat.lt_succ_self 6⟩, ⟨6, Nat.lt_succ_self 6⟩)) ∧ last_turn = turn.p2


theorem player1_cannot_guarantee_win : ¬∀ (prev_dir : direction) (pos : grid), (valid_move turn.p1 prev_dir direction.left ∨
  valid_move turn.p1 prev_dir prev_dir) → winner turn.p1 pos :=
sorry

end player1_cannot_guarantee_win_l243_243450


namespace construct_quadratic_with_conditions_l243_243912

noncomputable def quadratic_with_roots_and_value (a b c : ℝ) := 
  - (8 / 9) * a * a + (32 / 9) * b * b + (40 / 9)

theorem construct_quadratic_with_conditions :
  (∀ x, x = -1 ∨ x = 5 → f x = 0) →
  (f 2 = 8) →
  (f = λ x, - (8 / 9) * x * x + (32 / 9) * x + (40 / 9)) :=
by
  sorry

end construct_quadratic_with_conditions_l243_243912


namespace solve_system_of_equations_solve_algebraic_equation_l243_243150

-- Problem 1: System of Equations
theorem solve_system_of_equations (x y : ℝ) (h1 : x + 2 * y = 3) (h2 : 2 * x - y = 1) : x = 1 ∧ y = 1 :=
sorry

-- Problem 2: Algebraic Equation
theorem solve_algebraic_equation (x : ℝ) (h : 1 / (x - 1) + 2 = 5 / (1 - x)) : x = -2 :=
sorry

end solve_system_of_equations_solve_algebraic_equation_l243_243150


namespace smallest_four_digit_divisible_by_53_l243_243663

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243663


namespace solve_quadratic_sum_l243_243534

theorem solve_quadratic_sum (a b : ℂ) (h1 : 6 * a^2 + 1 = 5 * a - 16) (h2 : a = (5 / 12) ∧ b = (real.sqrt 383 / 12)) : 
  a + b^2 = (443 / 144) :=
by
  sorry

end solve_quadratic_sum_l243_243534


namespace min_students_orchestra_l243_243249

theorem min_students_orchestra (n : ℕ) 
  (h1 : n % 9 = 0)
  (h2 : n % 10 = 0)
  (h3 : n % 11 = 0) : 
  n ≥ 990 ∧ ∃ k, n = 990 * k :=
by
  sorry

end min_students_orchestra_l243_243249


namespace solve_latin_square_l243_243286

def valid_latin_square (M : Matrix (Fin 4) (Fin 4) ℕ) : Prop :=
  ∀ i, (Finset.univ.image (λ j => M i j)).card = 4 ∧
       (Finset.univ.image (λ j => M j i)).card = 4

def highlighted_region_conditions (M : Matrix (Fin 4) (Fin 4) ℕ) : Prop :=
  (M 0 2 * M 1 2 * M 1 3 = 48) ∧                 --- Region with product 48
  (M 2 0 * M 2 1 * M 2 2 = 6) ∧                  --- Region with product 6
  (abs (M 3 2 - M 3 3) = 1) ∧                   --- Region with difference 1
  (M 0 3 + M 1 0 + M 3 1 = 9)                   --- Region with sum 9

def grid := ![
  ![1, 2, 3, 4],
  ![2, 3, 4, 1],
  ![4, 1, 2, 3],
  ![3, 4, 1, 2]
]

theorem solve_latin_square : valid_latin_square grid ∧ highlighted_region_conditions grid :=
by 
  sorry

end solve_latin_square_l243_243286


namespace sum_of_squares_of_segments_sum_of_squares_of_chords_l243_243426

section circle_theorems

variables {R d : ℝ} (O M A B C D : Point)
variable [MetricSpace Point]

-- conditions
variable (radius_O : dist O A = R)
variable (perp_chords : ⟪M - A, M - B⟫ = 0 ∧ ⟪M - C, M - D⟫ = 0)
variable (distance_center : dist O M = d)

-- Part (a)
theorem sum_of_squares_of_segments
   (AM MB CM MD : ℝ)
   (chord_segments : AM + MB = dist A B ∧ CM + MD = dist C D)
   (intersect : dist A M = AM ∧ dist M B = MB ∧ dist C M = CM ∧ dist M D = MD) :
  AM^2 + MB^2 + CM^2 + MD^2 = 4 * R^2 := sorry

-- Part (b)
theorem sum_of_squares_of_chords
   (AB CD : ℝ)
   (chord_lengths : AB = dist A B ∧ CD = dist C D) :
  AB^2 + CD^2 = 8 * R^2 - 4 * d^2 := sorry

end circle_theorems

end sum_of_squares_of_segments_sum_of_squares_of_chords_l243_243426


namespace triangle_angle_A_l243_243091

theorem triangle_angle_A :
  ∀ (A : ℝ), 4 * real.pi * real.sin A - 3 * real.arccos (-1 / 2) = 0 ∧ 0 < A ∧ A < real.pi → 
  A = real.pi / 6 ∨ A = 5 * real.pi / 6 :=
by intro A; sorry

end triangle_angle_A_l243_243091


namespace ratio_triangle_BDF_to_rectangle_l243_243104

variable (x : ℝ)

-- Define the lengths of the sides of the rectangle
def AB := 3 * x
def BC := 2 * x

-- Define the points E and F
def AE := 3 * EB
def EB := x
def CF := 3 * FD
def FD := x

-- Define the area calculations
def areaRectangle : ℝ := AB * BC
def areaTriangleABF : ℝ := (1/2) * AB * BF
def areaTriangleCDF : ℝ := (1/2) * CD * DF

-- Prove the ratio of areas
theorem ratio_triangle_BDF_to_rectangle (h_AB : AB = 3 * x) (h_BC : BC = 2 * x) (h_AE : AE = 3 * EB)
  (h_EB : EB = x) (h_CF : CF = 3 * FD) (h_FD : FD = x) :
  (areaTriangleBDF / areaRectangle) = 7 / 12 :=
by
  sorry

end ratio_triangle_BDF_to_rectangle_l243_243104


namespace smallest_four_digit_divisible_by_53_l243_243751

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243751


namespace distinct_points_intersection_l243_243297

theorem distinct_points_intersection :
  ∃ (S : Finset (ℝ × ℝ)), 
    (∀ (x y : ℝ), (x + 2 * y - 7 = 0 ∨ 2 * x - y + 4 = 0) → 
                   (x - 2 * y + 3 = 0 ∨ 4 * x + 3 * y - 18 = 0) → 
                   (x, y) ∈ S) ∧ 
    S.card = 4 :=
by 
  sorry

end distinct_points_intersection_l243_243297


namespace smallest_four_digit_divisible_by_53_l243_243757

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243757


namespace range_of_independent_variable_l243_243453

theorem range_of_independent_variable (x : ℝ) (y : ℝ) (h : y = sqrt (x + 2)) : x ≥ -2 :=
sorry

end range_of_independent_variable_l243_243453


namespace garden_length_l243_243066

def PerimeterLength (P : ℕ) (length : ℕ) (breadth : ℕ) : Prop :=
  P = 2 * (length + breadth)

theorem garden_length
  (P : ℕ)
  (breadth : ℕ)
  (h1 : P = 480)
  (h2 : breadth = 100):
  ∃ length : ℕ, PerimeterLength P length breadth ∧ length = 140 :=
by
  use 140
  sorry

end garden_length_l243_243066


namespace age_of_senior_citizens_is_correct_l243_243427

def club :=
  { members: ℕ,
    average_age: ℕ,
    women: ℕ,
    men: ℕ,
    senior_citizens: ℕ,
    avg_age_women: ℕ,
    avg_age_men: ℕ }

noncomputable def total_age (c: club) : ℕ :=
  c.members * c.average_age

noncomputable def sum_ages_women (c: club) : ℕ :=
  c.women * c.avg_age_women

noncomputable def sum_ages_men (c: club) : ℕ :=
  c.men * c.avg_age_men

noncomputable def sum_ages_senior_citizens (c: club) : ℕ :=
  total_age c - sum_ages_women c - sum_ages_men c

noncomputable def avg_age_senior_citizens (c: club) : ℚ :=
  (sum_ages_senior_citizens c : ℚ) / c.senior_citizens

theorem age_of_senior_citizens_is_correct (c: club) (h1: c.members = 60) (h2: c.average_age = 30)
  (h3: c.women = 25) (h4: c.men = 20) (h5: c.senior_citizens = 15) (h6: c.avg_age_women = 28)
  (h7: c.avg_age_men = 35) : avg_age_senior_citizens c = 80 / 3 := by
  sorry

end age_of_senior_citizens_is_correct_l243_243427


namespace smallest_four_digit_divisible_by_53_l243_243614

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l243_243614


namespace find_multiplier_l243_243794

theorem find_multiplier :
  ∀ (x n : ℝ), (x = 5) → (x * n = (16 - x) + 4) → n = 3 :=
by
  intros x n hx heq
  sorry

end find_multiplier_l243_243794


namespace concyclic_points_of_folded_pyramid_l243_243590

theorem concyclic_points_of_folded_pyramid
  (S A B C D K L M N: Point)
  (inscribed_sphere: ConvexPyramid → Sphere)
  (pyramid : ConvexPyramid (face S A B D) (face S B C D))
  (polygon_formed_by_unfolding: UnfoldPolygon pyramid (edges S A) (edges S B) (edges S C) (edges S D)) :
    Concyclic K L M N :=
begin
  sorry,
end

end concyclic_points_of_folded_pyramid_l243_243590


namespace ellipse_line_intersection_and_reflection_l243_243005

/-- Given an ellipse with focal length 2√2 and the equation \frac{x²}{a²} + \frac{y²}{b²} = 1, with a > b > 0, 
and a line passing through the point (-√2, 1) with a slope k intersecting the ellipse. 
Define points A and B on the ellipse from the intersection, point P on the x-axis from 
the intersection of the line with the x-axis, point C as the reflection of A over the x-axis, 
and point Q from the intersection of line BC and the x-axis. 
We prove the range of k and the product |OP| * |OQ| is 4.
-/
theorem ellipse_line_intersection_and_reflection 
  (a b : ℝ)
  (hyp1 : a > 0)
  (hyp2 : b > 0)
  (hyp3 : a > b)
  (hyp4 : 2 * real.sqrt 2 = math.sqrt (a^2 - b^2))
  (k : ℝ) 
  (hyp5 : k^2 > 1/2)
  (P Q : ℝ × ℝ)
  (hypP : P = (2 / k, 0))
  (hypQ : Q = (2 * k, 0)) : 
  ∃ (k : ℝ), ((k < -sqrt 2 / 2) ∨ (k > sqrt 2 / 2)) ∧ |P.1| * |Q.1| = 4 :=
by {
  sorry
}

end ellipse_line_intersection_and_reflection_l243_243005


namespace chord_length_circle_l243_243157

theorem chord_length_circle {x y : ℝ} :
  (x - 1)^2 + (y - 1)^2 = 2 →
  (exists (p q : ℝ), (p-1)^2 = 1 ∧ (q-1)^2 = 1 ∧ p ≠ q ∧ abs (p - q) = 2) :=
by
  intro h
  use (2 : ℝ)
  use (0 : ℝ)
  -- Formal proof omitted
  sorry

end chord_length_circle_l243_243157


namespace least_third_side_of_right_triangle_l243_243405

theorem least_third_side_of_right_triangle {a b c : ℝ} 
  (h1 : a = 8) 
  (h2 : b = 15) 
  (h3 : c = Real.sqrt (8^2 + 15^2) ∨ c = Real.sqrt (15^2 - 8^2)) : 
  c = Real.sqrt 161 :=
by {
  intros h1 h2 h3,
  rw [h1, h2] at h3,
  cases h3,
  { exfalso, preciesly contradiction occurs because sqrt (8^2 + 15^2) is not sqrt161, 
   rw [← h3],
   norm_num,},
  { exact h3},
  
}

end least_third_side_of_right_triangle_l243_243405


namespace problem_statement_l243_243152

noncomputable def real_sequence (n : ℕ) : ℝ :=
  if n = 0 then 1
  else if n % 2 = 1 then ∑ i in finset.range n.by_2, real_sequence i * real_sequence (n - 1 - i)
  else ∑ i in finset.range (n / 2), real_sequence i * real_sequence (n - 1 - i)

def generates_r (r : ℝ) :=
  let series_sum := ∑ n in finset.range 1000, real_sequence n * r^n
  series_sum = 5 / 4

def satisfies_rational_form (r : ℝ) (a b c d : ℕ) :=
  r = (a * real.sqrt b - c) / d ∧
  ∃ (p : ℕ), prime p ∧ p^2 ∣ b ∧ p ≤ b ∧
  nat.gcd a c = 1 ∧
  nat.gcd a d = 1

theorem problem_statement (a b c d : ℕ) (r : ℝ) (h1 : generates_r r) (h2 : satisfies_rational_form r a b c d) :
  a + b + c + d = 1923 :=
sorry

end problem_statement_l243_243152


namespace betta_fish_count_l243_243096

theorem betta_fish_count 
  (total_guppies_per_day : ℕ) 
  (moray_eel_consumption : ℕ) 
  (betta_fish_consumption : ℕ) 
  (betta_fish_count : ℕ) 
  (h_total : total_guppies_per_day = 55)
  (h_eel : moray_eel_consumption = 20)
  (h_betta : betta_fish_consumption = 7) 
  (h_eq : total_guppies_per_day - moray_eel_consumption = betta_fish_consumption * betta_fish_count) : 
  betta_fish_count = 5 :=
by 
  sorry

end betta_fish_count_l243_243096


namespace number_of_zeroes_f_l243_243172

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 3

theorem number_of_zeroes_f : {x : ℝ | f x = 0}.finite ∧ {x : ℝ | f x = 0}.toFinset.card = 2 :=
by
  sorry

end number_of_zeroes_f_l243_243172


namespace quadrilateral_m_in_unit_circle_l243_243481

open Real

noncomputable def maximum_m : ℝ :=
  2

theorem quadrilateral_m_in_unit_circle
  (A B C D : ℝ × ℝ)    -- points A, B, C, D
  (h_circ : (dist A B = 1) ∧ (dist B C = 1) ∧ (dist C D = 1) ∧ (dist D A = 1)) -- All points are on the unit circle
  (h_angle : ∠BAD = (π / 6))  -- Angle BAD is 30 degrees (π/6 in radians)
  : maximum_m = 2 :=
sorry

end quadrilateral_m_in_unit_circle_l243_243481


namespace smallest_four_digit_divisible_by_53_l243_243698

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243698


namespace infinite_sum_identity_l243_243302

noncomputable def harmonic (n : ℕ) : ℝ :=
  (finset.range n).sum (λ k, 1 / (k + 1))

theorem infinite_sum_identity :
  ∑' n, 1 / (n * (n + 2) * harmonic n * harmonic (n + 2)) = 
  1 / harmonic 1 + 1 / harmonic 2 + 1 / harmonic 3 :=
sorry

end infinite_sum_identity_l243_243302


namespace stratified_sampling_numbers_l243_243425

-- Definitions of the conditions
def total_teachers : ℕ := 300
def senior_teachers : ℕ := 90
def intermediate_teachers : ℕ := 150
def junior_teachers : ℕ := 60
def sample_size : ℕ := 40

-- Hypothesis of proportions
def proportion_senior := senior_teachers / total_teachers
def proportion_intermediate := intermediate_teachers / total_teachers
def proportion_junior := junior_teachers / total_teachers

-- Expected sample counts using stratified sampling method
def expected_senior_drawn := proportion_senior * sample_size
def expected_intermediate_drawn := proportion_intermediate * sample_size
def expected_junior_drawn := proportion_junior * sample_size

-- Proof goal
theorem stratified_sampling_numbers :
  (expected_senior_drawn = 12) ∧ 
  (expected_intermediate_drawn = 20) ∧ 
  (expected_junior_drawn = 8) :=
by
  sorry

end stratified_sampling_numbers_l243_243425


namespace max_radius_for_peach_l243_243811

-- Define the cross-section of the wine glass
def wine_glass (x : ℝ) : ℝ := x^4

-- Define the parameters
variable (r : ℝ)

-- Define the inequality condition that needs to be satisfied
def inequality_condition (x : ℝ) : Prop := r - sqrt (r^2 - x^2) ≥ x^4

-- Define the interval for 'x'
def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ r

-- The main theorem to prove
theorem max_radius_for_peach : r ≤ (3 / 4) * real.cbrt 2 :=
begin
  sorry -- Skip the proof for this problem
end

end max_radius_for_peach_l243_243811


namespace sum_first_100_terms_l243_243915

def optimal_decomposition_f (n : ℕ) : ℕ :=
if n = 0 then 0
else if n = 1 then 1
else 
  let k := n / 2 in 
  if 2 * k = n then 0 -- even n
  else 2 * 3^k -- odd n

def sequence_f (n : ℕ) : ℕ := optimal_decomposition_f (3^n)

def partial_sum (n : ℕ) : ℕ := (List.range n).map sequence_f |>.sum

theorem sum_first_100_terms : partial_sum 100 = 3^50 - 1 := by
  sorry

end sum_first_100_terms_l243_243915


namespace exists_cycle_not_divisible_by_three_l243_243537

open Combinatorics

theorem exists_cycle_not_divisible_by_three (V : Type) [Finite V] 
  (G : SimpleGraph V) (h : ∀ v, 3 ≤ G.degree v) : 
  ∃ (c : List V), (G.isCycle c) ∧ (¬ (c.length % 3 = 0)) :=
sorry

end exists_cycle_not_divisible_by_three_l243_243537


namespace probability_brick_box_dim_l243_243194

theorem probability_brick_box_dim (a₁ a₂ a₃ b₁ b₂ b₃ : Finset ℕ) (h₁ : a₁ ∈ (Finset.range 501) ∧ a₂ ∈ (Finset.range 501) ∧ a₃ ∈ (Finset.range 501))
(h₂ : b₁ ∉ (Finset.range 501).erase a₁ ∧ b₂ ∉ (Finset.range 501).erase a₂ ∧ b₃ ∉ (Finset.range 501).erase a₃) :
  let x := (Finset.toList a₁ ∪ Finset.toList a₂ ∪ Finset.toList a₃ ∪ Finset.toList b₁ ∪ Finset.toList b₂ ∪ Finset.toList b₃).sort >
  ((perm (sublist_cons 20 [x₁, x₂, x₃, x₄, x₅, x₆]) (take three [x₁, x₂, x₃, x₄, x₅, x₆]) = 1/4) → 
  nat.succ ((Numerator.denom 4 (normalize 1/4)) = 5)

by {
  sorry
}

end probability_brick_box_dim_l243_243194


namespace quadratic_has_root_in_interval_l243_243158

theorem quadratic_has_root_in_interval 
  {a b c : ℝ} 
  (h1 : 2 * a + 3 * b + 6 * c = 0) :
  ∃ x ∈ Ioo 0 1, a * x^2 + b * x + c = 0 :=
sorry

end quadratic_has_root_in_interval_l243_243158


namespace least_third_side_length_l243_243412

theorem least_third_side_length (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℝ, (c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2)) ∧ c = Real.sqrt 161 :=
by
  sorry

end least_third_side_length_l243_243412


namespace minimal_n_integer_seq_l243_243272

def seq (y_1 : ℝ) (y_2 : ℝ) (y : ℕ → ℝ) (n : ℕ) : Prop :=
(y_1 = real.root 4 4) ∧
(y_2 = (real.root 4 4)^(real.root 3 4)) ∧
(∀ n > 1, y n = (y (n - 1))^(real.root 3 4))

theorem minimal_n_integer_seq :
  ∃ n : ℕ, n = 4 ∧ (∃ y : ℕ → ℝ, seq (real.root 4 4) ((real.root 4 4)^(real.root 3 4)) y n ∧ y n ∈ ℤ) :=
by sorry

end minimal_n_integer_seq_l243_243272


namespace count_a_6_pairs_l243_243540

theorem count_a_6_pairs (tanX : Real) (tanY : Real) (a m n : ℕ) (h1 : tanX = 1 / m) (h2 : tanY = a / n) (h3 : tan (Real.toDegrees (tan⁻¹ tanX) + Real.toDegrees (tan⁻¹ tanY)) = 1) (h4 : a ≤ 50) :
  ∃! (S : Set ℕ), S = {a ∈ ℕ | a ≤ 50 ∧ ∃! (m' n' : ℕ), tanX = 1 / m' ∧ tanY = a / n' ∧ tan (Real.toDegrees (tan⁻¹ (1 / m')) + Real.toDegrees (tan⁻¹ (a / n'))) = 1} ∧ S.card = 12 := sorry

end count_a_6_pairs_l243_243540


namespace smallest_four_digit_divisible_by_53_l243_243732

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243732


namespace min_value_f_l243_243296

noncomputable def f (x : ℝ) : ℝ := x^3 + 9 * x + 81 / x^4

theorem min_value_f : ∃ x > 0, f x = 21 ∧ ∀ y > 0, f y ≥ 21 := by
  sorry

end min_value_f_l243_243296


namespace sequence_properties_sum_first_n_terms_l243_243041

open Nat

noncomputable def a_n (n : ℕ) : ℕ := 2^n

noncomputable def b_n (n : ℕ) : ℕ := n * (n + 1) / 2

noncomputable def c_n (n : ℕ) : ℚ := (b_n n - a_n n) / (a_n n * b_n n)

noncomputable def S_n (n : ℕ) : ℚ := (∑ i in Finset.range n, c_n (i+1))

theorem sequence_properties (n : ℕ) (h : n > 0) :
  (a_n n = 2^n) ∧ (b_n n = n * (n + 1) / 2) :=
by sorry

theorem sum_first_n_terms (n : ℕ) (h : n > 0) :
  S_n n = (2 / (n + 1)) - (1 / 2^n) - 1 :=
by sorry

end sequence_properties_sum_first_n_terms_l243_243041


namespace find_annual_pension_l243_243861

variable (P k x a b p q : ℝ) (h1 : k * Real.sqrt (x + a) = k * Real.sqrt x + p)
                                   (h2 : k * Real.sqrt (x + b) = k * Real.sqrt x + q)

theorem find_annual_pension (h_nonzero_proportionality_constant : k ≠ 0) 
(h_year_difference : a ≠ b) : 
P = (a * q ^ 2 - b * p ^ 2) / (2 * (b * p - a * q)) := 
by
  sorry

end find_annual_pension_l243_243861


namespace toyota_honda_ratio_l243_243588

noncomputable def ratio_of_vehicles (T H : ℕ) : ℚ :=
  T / H

theorem toyota_honda_ratio 
  (T H : ℕ)
  (h : 0.6 * T + 0.4 * H = 52) :
  ratio_of_vehicles T H = 85 / 3 :=
sorry

end toyota_honda_ratio_l243_243588


namespace probability_no_two_green_hats_next_to_each_other_l243_243825

open Nat

def choose (n k : ℕ) : ℕ := Nat.fact n / (Nat.fact k * Nat.fact (n - k))

def total_ways_to_choose (n k : ℕ) : ℕ :=
  choose n k

def event_A (n : ℕ) : ℕ := n - 2

def event_B (n k : ℕ) : ℕ := choose (n - k + 1) 2 * (k - 2)

def probability_no_two_next_to_each_other (n k : ℕ) : ℚ :=
  let total_ways := total_ways_to_choose n k
  let event_A_ways := event_A (n)
  let event_B_ways := event_B n 3
  let favorable_ways := total_ways - (event_A_ways + event_B_ways)
  favorable_ways / total_ways

-- Given the conditions of 9 children and choosing 3 to wear green hats
theorem probability_no_two_green_hats_next_to_each_other (p : probability_no_two_next_to_each_other 9 3 = 5 / 14) : Prop := by
  sorry

end probability_no_two_green_hats_next_to_each_other_l243_243825


namespace length_of_common_chord_l243_243159

theorem length_of_common_chord 
    (O1_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * x = 0)
    (O2_eq : ∀ x y : ℝ, x^2 + y^2 - 4 * y = 0) :
    ∃ l : ℝ, l = 4 * (sqrt 5) / 5 := by
  sorry

end length_of_common_chord_l243_243159


namespace evaluate_ceil_sqrt_225_l243_243946

def ceil (x : ℝ) : ℤ :=
  if h : ∃ n : ℤ, n ≤ x ∧ x < n + 1 then
    classical.some h
  else
    0

theorem evaluate_ceil_sqrt_225 : ceil (Real.sqrt 225) = 15 := 
sorry

end evaluate_ceil_sqrt_225_l243_243946


namespace sin_periodicity_example_l243_243265

theorem sin_periodicity_example : sin (-1560 * real.pi / 180) = - (real.sqrt 3 / 2) :=
by
  -- Conditions
  -- 1. The periodic property of the sine function
  have h1 : ∀ θ k, sin (θ + 2 * real.pi * k) = sin θ := real.sin_add_two_pi
  -- 2. Representation of -1560 in terms of the base angle
  have h2 : -1560 * real.pi / 180 = 240 * real.pi / 180 - 4 * 2 * real.pi := by
    rw [mul_div_cancel_of_implies, real.mul_assoc, real.mul_div_assoc, real.mul_div_cancel_of_implies, real.sub_eq_add_neg, real.add_comm, real.zero_add]
  -- 3. Use sine in the third quadrant relation
  have h3 : sin (4 * 2 * real.pi + 240 * real.pi / 180) = sin (240 * real.pi / 180) := 
    by rw [h2, h1]
  -- 4. Known sine value
  have h4 : sin (4 * 2 * real.pi + 240 * real.pi / 180) = -real.sqrt 3 / 2 := 
    by rw [h3, real.sin_of_real, real.sin_of_real, real.sin_pi_div_3]
  sorry

end sin_periodicity_example_l243_243265


namespace largest_k_for_3_in_g_l243_243277

theorem largest_k_for_3_in_g (k : ℝ) :
  (∃ x : ℝ, 2*x^2 - 8*x + k = 3) ↔ k ≤ 11 :=
by
  sorry

end largest_k_for_3_in_g_l243_243277


namespace least_third_side_length_l243_243411

theorem least_third_side_length (a b : ℕ) (h_a : a = 8) (h_b : b = 15) : 
  ∃ c : ℝ, (c = Real.sqrt (a^2 + b^2) ∨ c = Real.sqrt (b^2 - a^2)) ∧ c = Real.sqrt 161 :=
by
  sorry

end least_third_side_length_l243_243411


namespace monotonically_decreasing_interval_l243_243030

noncomputable def f (x : ℝ) : ℝ := 
  2 * (sin (x + π/4))^2 - (√3) * cos (2 * x)

theorem monotonically_decreasing_interval (k : ℤ) :
  ∀ x, k * π + (5 * π / 12) ≤ x ∧ x ≤ k * π + (11 * π / 12) → deriv f x < 0 :=
by
  sorry

end monotonically_decreasing_interval_l243_243030


namespace smallest_four_digit_divisible_by_53_l243_243753

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243753


namespace least_third_side_of_right_triangle_l243_243407

theorem least_third_side_of_right_triangle {a b c : ℝ} 
  (h1 : a = 8) 
  (h2 : b = 15) 
  (h3 : c = Real.sqrt (8^2 + 15^2) ∨ c = Real.sqrt (15^2 - 8^2)) : 
  c = Real.sqrt 161 :=
by {
  intros h1 h2 h3,
  rw [h1, h2] at h3,
  cases h3,
  { exfalso, preciesly contradiction occurs because sqrt (8^2 + 15^2) is not sqrt161, 
   rw [← h3],
   norm_num,},
  { exact h3},
  
}

end least_third_side_of_right_triangle_l243_243407


namespace inequality_solution_l243_243536

theorem inequality_solution (x : ℝ) : 
  (x + 1) / (x + 4) ≥ 0 ↔ x ∈ set.Iio (-4) ∪ set.Ici (-1) := 
begin
  sorry
end

end inequality_solution_l243_243536


namespace sum_x_coords_Q3_eq_153_l243_243230

theorem sum_x_coords_Q3_eq_153 (x_coords_Q1 : Fin 51 → ℝ) 
  (h_sum_Q1 : (∑ i, x_coords_Q1 i) = 153) :
  let x_coords_Q2 := λ i, (x_coords_Q1 i + x_coords_Q1 ((i+1) % 51)) / 2 in
  let x_coords_Q3 := λ i, (x_coords_Q2 i + x_coords_Q2 ((i+1) % 51)) / 2 in
  (∑ i, x_coords_Q3 i) = 153 :=
by sorry

end sum_x_coords_Q3_eq_153_l243_243230


namespace pizza_fraction_eaten_l243_243216

-- The total fractional part of the pizza eaten after six trips
theorem pizza_fraction_eaten : 
  ∑ i in (finset.range 6), (1 / 3) ^ (i + 1) = 364 / 729 :=
by
  sorry

end pizza_fraction_eaten_l243_243216


namespace shift_sin2x_l243_243195

theorem shift_sin2x :
  ∀ (x : ℝ), (sin x * cos x + sqrt 3 * cos x ^ 2 - sqrt 3 / 2) = sin (2 * x + π / 3) :=
by
  sorry

end shift_sin2x_l243_243195


namespace two_digit_subtraction_pattern_l243_243164

theorem two_digit_subtraction_pattern (a b : ℕ) (h_a : 1 ≤ a ∧ a ≤ 9) (h_b : 0 ≤ b ∧ b ≤ 9) :
  (10 * a + b) - (10 * b + a) = 9 * (a - b) := 
by
  sorry

end two_digit_subtraction_pattern_l243_243164


namespace magic8_prob_3_out_of_7_exactly_3_positive_l243_243097

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := nat.choose n k

theorem magic8_prob_3_out_of_7_exactly_3_positive :
  (3/7 : ℚ)^(3) * (4/7 : ℚ)^(4) * binomial_coefficient 7 3 = 242112/823543 := by
  sorry

end magic8_prob_3_out_of_7_exactly_3_positive_l243_243097


namespace license_plate_count_l243_243049

def num_license_plates : Nat :=
  let letters := 26 -- choices for each of the first two letters
  let primes := 4 -- choices for prime digits
  let composites := 4 -- choices for composite digits
  letters * letters * (primes * composites * 2)

theorem license_plate_count : num_license_plates = 21632 :=
  by
  sorry

end license_plate_count_l243_243049


namespace smallest_four_digit_divisible_by_53_l243_243686

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l243_243686


namespace solve_for_y_l243_243918

theorem solve_for_y (y : ℚ) (h : |(4 : ℚ) * y - 6| = 0) : y = 3 / 2 :=
sorry

end solve_for_y_l243_243918


namespace least_third_side_of_right_triangle_l243_243409

theorem least_third_side_of_right_triangle {a b c : ℝ} 
  (h1 : a = 8) 
  (h2 : b = 15) 
  (h3 : c = Real.sqrt (8^2 + 15^2) ∨ c = Real.sqrt (15^2 - 8^2)) : 
  c = Real.sqrt 161 :=
by {
  intros h1 h2 h3,
  rw [h1, h2] at h3,
  cases h3,
  { exfalso, preciesly contradiction occurs because sqrt (8^2 + 15^2) is not sqrt161, 
   rw [← h3],
   norm_num,},
  { exact h3},
  
}

end least_third_side_of_right_triangle_l243_243409


namespace cos_angle_GAE_correct_l243_243226

noncomputable def cos_angle_GAE (a : ℝ) : ℝ :=
  let AE := a
  let GE := a * Real.sqrt 2
  let AG := a * Real.sqrt 3
  Real.cos (AE.outer_angle AG GE) = Real.sqrt 3 / 3

theorem cos_angle_GAE_correct (a : ℝ) : cos_angle_GAE a = Real.sqrt 3 / 3 := by
  sorry

end cos_angle_GAE_correct_l243_243226


namespace grocer_second_month_sale_l243_243243

theorem grocer_second_month_sale (sale_1 sale_3 sale_4 sale_5 sale_6 avg_sale n : ℕ) 
(h1 : sale_1 = 6435) 
(h3 : sale_3 = 6855) 
(h4 : sale_4 = 7230) 
(h5 : sale_5 = 6562) 
(h6 : sale_6 = 7391) 
(havg : avg_sale = 6900) 
(hn : n = 6) : 
  sale_2 = 6927 :=
by
  sorry

end grocer_second_month_sale_l243_243243


namespace probability_no_adjacent_green_hats_l243_243815

-- Step d): Rewrite the math proof problem in a Lean 4 statement.

theorem probability_no_adjacent_green_hats (total_children green_hats : ℕ)
  (hc : total_children = 9) (hg : green_hats = 3) :
  (∃ (p : ℚ), p = 5 / 14) :=
sorry

end probability_no_adjacent_green_hats_l243_243815


namespace prob_white_ball_l243_243058

noncomputable def bag_A := {white := 8, red := 4}
noncomputable def bag_B := {white := 6, red := 5}

def C (n k : ℕ) := nat.choose n k

def prob := (C 8 1 * C 5 1 + C 4 1 * C 6 1) / (C 12 1 * C 11 1)

theorem prob_white_ball (X : ℕ) (A : bag_A.white + bag_A.red = 12) (B : bag_B.white + bag_B.red = 11) :
  X = 1 ↔ prob = (C 8 1 * C 5 1 + C 4 1 * C 6 1) / (C 12 1 * C 11 1) :=
sorry

end prob_white_ball_l243_243058


namespace smallest_four_digit_divisible_by_53_l243_243666

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243666


namespace ceil_sqrt_225_eq_15_l243_243957

theorem ceil_sqrt_225_eq_15 : ⌈ Real.sqrt 225 ⌉ = 15 :=
by
  sorry

end ceil_sqrt_225_eq_15_l243_243957


namespace sqrt_16_eq_plus_minus_4_l243_243568

theorem sqrt_16_eq_plus_minus_4 : ∀ x : ℝ, (x^2 = 16) ↔ (x = 4 ∨ x = -4) :=
by sorry

end sqrt_16_eq_plus_minus_4_l243_243568


namespace smallest_four_digit_divisible_by_53_l243_243635

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l243_243635


namespace prob_heads_even_correct_l243_243875

noncomputable def prob_heads_even (n : Nat) : ℝ :=
  if n = 0 then 1
  else (2 / 3) - (1 / 3) * prob_heads_even (n - 1)

theorem prob_heads_even_correct : 
  prob_heads_even 50 = (1 / 2) * (1 + (1 / 3 ^ 50)) :=
sorry

end prob_heads_even_correct_l243_243875


namespace ceil_sqrt_225_eq_15_l243_243954

theorem ceil_sqrt_225_eq_15 : ⌈ Real.sqrt 225 ⌉ = 15 :=
by
  sorry

end ceil_sqrt_225_eq_15_l243_243954


namespace cube_skew_lines_probability_l243_243580

theorem cube_skew_lines_probability 
  (vertices : Finset ℝ) 
  (h_vertices : vertices.card = 8)
  (lines : Finset (Finset ℝ)) 
  (h_lines : ∀ v₁ v₂ ∈ vertices, v₁ ≠ v₂ → ∃ l ∈ lines, {v₁, v₂} ⊆ l) 
  (total_lines : lines.card = 28) 
  (skew_lines : Finset (Finset ℝ)) 
  (h_skew : ∀ l₁ l₂ ∈ lines, l₁ ≠ l₂ → (l₁ ∩ l₂ = ∅ ∧ ¬ parallel l₁ l₂) ↔ {l₁, l₂} ∈ skew_lines)
  (total_skew_sets : skew_lines.card = 174): 
  (29 : ℝ) / 63 = (skew_lines.card : ℝ) / (lines.card * (lines.card - 1) / 2) := 
by
  -- sorry placeholder for the proof
  sorry

end cube_skew_lines_probability_l243_243580


namespace smallest_four_digit_divisible_by_53_l243_243646

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l243_243646


namespace investment_period_l243_243178

theorem investment_period (x t : ℕ) (p_investment q_investment q_time : ℕ) (profit_ratio : ℚ):
  q_investment = 5 * x →
  p_investment = 7 * x →
  q_time = 16 →
  profit_ratio = 7 / 10 →
  7 * x * t = profit_ratio * 5 * x * q_time →
  t = 8 := sorry

end investment_period_l243_243178


namespace square_of_1023_l243_243896

def square_1023_eq_1046529 : Prop :=
  let x := 1023
  x * x = 1046529

theorem square_of_1023 : square_1023_eq_1046529 :=
by
  sorry

end square_of_1023_l243_243896


namespace smallest_four_digit_multiple_of_53_l243_243780

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l243_243780


namespace linear_coefficient_of_expanded_equation_l243_243024

theorem linear_coefficient_of_expanded_equation :
  let f : ℝ → ℝ := λ x, (2 * x + 1) * (x - 3) - (x^2 + 1)
  ∃ a b c : ℝ, f = λ x, a * x^2 + b * x + c ∧ b = -5 :=
by
  sorry

end linear_coefficient_of_expanded_equation_l243_243024


namespace f_expression_for_x_greater_than_1_l243_243166

def f (x : ℝ) : ℝ :=
  if x < 1 then x^2 + 1 else sorry

theorem f_expression_for_x_greater_than_1 :
  (∀ x : ℝ, f(x + 1) = f(-x + 1)) ∧ (∀ x : ℝ, x < 1 → f x = x^2 + 1) →
  ∀ x : ℝ, x > 1 → f x = x^2 - 4 * x + 5 :=
by
  intros h1 h2 x hx
  sorry

end f_expression_for_x_greater_than_1_l243_243166


namespace number_of_cookies_on_the_fifth_plate_l243_243188

theorem number_of_cookies_on_the_fifth_plate
  (c : ℕ → ℕ)
  (h1 : c 1 = 5)
  (h2 : c 2 = 7)
  (h3 : c 3 = 10)
  (h4 : c 4 = 14)
  (h6 : c 6 = 25)
  (h_diff : ∀ n, c (n + 1) - c n = c (n + 2) - c (n + 1) + 1) :
  c 5 = 19 :=
by
  sorry

end number_of_cookies_on_the_fifth_plate_l243_243188


namespace Marissa_sunflower_height_l243_243504

-- Define the necessary conditions
def sister_height_feet : ℕ := 4
def sister_height_inches : ℕ := 3
def extra_sunflower_height : ℕ := 21
def inches_per_foot : ℕ := 12

-- Calculate the total height of the sister in inches
def sister_total_height_inch : ℕ := (sister_height_feet * inches_per_foot) + sister_height_inches

-- Calculate the sunflower height in inches
def sunflower_height_inch : ℕ := sister_total_height_inch + extra_sunflower_height

-- Convert the sunflower height to feet
def sunflower_height_feet : ℕ := sunflower_height_inch / inches_per_foot

-- The theorem we want to prove
theorem Marissa_sunflower_height : sunflower_height_feet = 6 := by
  sorry

end Marissa_sunflower_height_l243_243504


namespace factory_produces_11250_products_l243_243848

noncomputable def total_products (refrigerators_per_hour coolers_per_hour hours_per_day days : ℕ) : ℕ :=
  (refrigerators_per_hour + coolers_per_hour) * (hours_per_day * days)

theorem factory_produces_11250_products :
  total_products 90 (90 + 70) 9 5 = 11250 := by
  sorry

end factory_produces_11250_products_l243_243848


namespace smallest_four_digit_divisible_by_53_l243_243655

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l243_243655


namespace smallest_four_digit_divisible_by_53_l243_243675

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243675


namespace no_adjacent_green_hats_l243_243827

theorem no_adjacent_green_hats (n m : ℕ) (h₀ : n = 9) (h₁ : m = 3) : 
  (((1 : ℚ) - (9/14 : ℚ)) = (5/14 : ℚ)) :=
by
  rw h₀ at *,
  rw h₁ at *,
  sorry

end no_adjacent_green_hats_l243_243827


namespace triangle_angle_measure_l243_243420

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) (h1 : b^2 = a * c) (h2 : a^2 - c^2 = a * c - b * c) :
  A = π / 3 ∧ (b * sin B) / c = sqrt 3 / 2 :=
by { sorry }

end triangle_angle_measure_l243_243420


namespace smallest_four_digit_div_by_53_l243_243633

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l243_243633


namespace inhabitants_lineup_l243_243134

-- Definitions of inhabitants and types
inductive Inhabitant
| Vegetarian
| Cannibal

-- Prime number property
def is_prime (n : ℕ) : Prop :=
  n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- All vegetarians are standing a prime number of people away from any given inhabitant
def vegetarian_prime_distance (inhabitants : List Inhabitant) (index : ℕ) : Prop :=
  ∀ i : ℕ, inhabitants.nth i = some Inhabitant.Vegetarian → is_prime (abs (index - i))

-- Main theorem statement: any number of inhabitants can be arranged to satisfy the conditions
theorem inhabitants_lineup (n : ℕ) :
  ∃ inhabitants : List Inhabitant,
  (∀ i, inhabitants.nth i = some Inhabitant.Vegetarian → vegetarian_prime_distance inhabitants i) ∧
  (∀ i, inhabitants.nth i = some Inhabitant.Cannibal → ¬ vegetarian_prime_distance inhabitants i) :=
sorry

end inhabitants_lineup_l243_243134


namespace max_min_value_of_f_at_neg_pi_over_6_monotonic_range_of_theta_l243_243036

def f (x θ : Real) : Real :=
  x^2 + 2 * x * Real.tan θ - 1

theorem max_min_value_of_f_at_neg_pi_over_6 :
  let θ := -Real.pi / 6 in
  let x_min := -1 in
  let x_opt := 1 / Real.sqrt 3 in
  let x_max := Real.sqrt 3 in
  f x_opt θ = -4 / 3 ∧
  f x_min θ = 2 / Real.sqrt 3 ∧
  f x_max θ = 0 :=
  sorry

theorem monotonic_range_of_theta :
  let I := Set.Icc (-1.0 : Real) (Real.sqrt 3) in
  let A := Set.Ioo (-Real.pi / 2) (-Real.pi / 3) in
  let B := Set.Ico (Real.pi / 4) (Real.pi / 2) in
  ∀ θ : Real, θ ∈ A ∪ B → 
    ∀ x₁ x₂ : Real, x₁ ∈ I → x₂ ∈ I → (x₁ ≤ x₂ → f x₁ θ ≤ f x₂ θ) ∨ (x₁ ≥ x₂ → f x₁ θ ≥ f x₂ θ) :=
  sorry

end max_min_value_of_f_at_neg_pi_over_6_monotonic_range_of_theta_l243_243036


namespace circumcircle_radius_triangle_l243_243585

theorem circumcircle_radius_triangle
  (r₁ r₂ : ℝ) (h₁ : r₁ + r₂ = 11) (d₁ : ∥(0 : ℝ), 0, r₁ - 0∥ = ∥(6 : ℝ), 0, r₂ - 0∥ = √481)
  (r₃ : ℝ) (h₂ : r₃ = 9)
  (AC_BC_external : ∥(0 : ℝ), 0, r₁ - (0 : ℝ), 0, r₃∥ = ∥(6 : ℝ), 0, r₂ - (6 : ℝ), 0, r₃∥) :
  ∃ (R : ℝ), R = 3 * √10 :=
by
  sorry

end circumcircle_radius_triangle_l243_243585


namespace range_contains_pi_div_4_l243_243979

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_contains_pi_div_4 : ∃ x : ℝ, f x = (Real.pi / 4) := by
  sorry

end range_contains_pi_div_4_l243_243979


namespace smallest_four_digit_multiple_of_53_l243_243706

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243706


namespace outfits_without_matching_color_l243_243801

theorem outfits_without_matching_color (red_shirts green_shirts pairs_pants green_hats red_hats : ℕ) 
  (h_red_shirts : red_shirts = 5) 
  (h_green_shirts : green_shirts = 5) 
  (h_pairs_pants : pairs_pants = 6) 
  (h_green_hats : green_hats = 8) 
  (h_red_hats : red_hats = 8) : 
  (red_shirts * pairs_pants * green_hats) + (green_shirts * pairs_pants * red_hats) = 480 := 
by 
  sorry

end outfits_without_matching_color_l243_243801


namespace least_third_side_of_right_triangle_l243_243380

theorem least_third_side_of_right_triangle (a b : ℕ) (ha : a = 8) (hb : b = 15) :
  ∃ c : ℝ, c = real.sqrt (b^2 - a^2) ∧ c = real.sqrt 161 :=
by {
  -- We state the conditions
  have h8 : (8 : ℝ) = a, from by {rw ha},
  have h15 : (15 : ℝ) = b, from by {rw hb},

  -- The theorem states that such a c exists
  use (real.sqrt (15^2 - 8^2)),

  -- We need to show the properties of c
  split,
  { 
    -- Showing that c is the sqrt of the difference of squares of b and a
    rw [←h15, ←h8],
    refl 
  },
  {
    -- Showing that c is sqrt(161)
    calc
       real.sqrt (15^2 - 8^2)
         = real.sqrt (225 - 64) : by norm_num
     ... = real.sqrt 161 : by norm_num
  }
}
sorry

end least_third_side_of_right_triangle_l243_243380


namespace greatest_possible_median_l243_243222

theorem greatest_possible_median (k m r s t : ℕ) (h_avg : (k + m + r + s + t) / 5 = 10) (h_order : k < m ∧ m < r ∧ r < s ∧ s < t) (h_t : t = 20) : r = 8 :=
by
  sorry

end greatest_possible_median_l243_243222


namespace probability_no_adjacent_green_hats_l243_243819

-- Definitions
def total_children : ℕ := 9
def green_hats : ℕ := 3

-- Main theorem statement
theorem probability_no_adjacent_green_hats : 
  (9.choose 3) = 84 → 
  (1 - (9 + 45) / 84) = 5/14 := 
sorry

end probability_no_adjacent_green_hats_l243_243819


namespace ceil_sqrt_225_eq_15_l243_243955

theorem ceil_sqrt_225_eq_15 : ⌈ Real.sqrt 225 ⌉ = 15 :=
by
  sorry

end ceil_sqrt_225_eq_15_l243_243955


namespace ball_hits_ground_time_l243_243840

theorem ball_hits_ground_time (h : ℝ → ℝ) (t : ℝ) :
  (∀ (t : ℝ), h t = -16 * t ^ 2 - 30 * t + 200) → h t = 0 → t = 2.5 :=
by
  -- Placeholder for the formal proof
  sorry

end ball_hits_ground_time_l243_243840


namespace smallest_four_digit_multiple_of_53_l243_243772

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243772


namespace circle_in_quad_radius_l243_243273

theorem circle_in_quad_radius (AB BC CD DA : ℝ) (r : ℝ) (h₁ : AB = 15) (h₂ : BC = 10) (h₃ : CD = 8) (h₄ : DA = 13) :
  r = 2 * Real.sqrt 10 := 
by {
  sorry
  }

end circle_in_quad_radius_l243_243273


namespace smallest_four_digit_divisible_by_53_l243_243611

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l243_243611


namespace smallest_four_digit_multiple_of_53_l243_243783

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l243_243783


namespace factor_x_minus_1_l243_243373

-- Definitions of the polynomials P, Q, R, and S considered as elements of polynomial ring over some field
variables (F : Type*) [Field F] (P Q R S : Polynomial F)

-- Given condition
def given_condition : Prop := 
  ∀ x : F, P (x^5) + x * Q (x^5) + x^2 * R (x^5) = (x^4 + x^3 + x^2 + x + 1) * S x

-- Proof goal: x - 1 is a factor of P(x)
theorem factor_x_minus_1 (h : given_condition F P Q R S) : 
  Polynomial.X - 1 ∣ P := 
sorry

end factor_x_minus_1_l243_243373


namespace ceil_sqrt_225_eq_15_l243_243962

theorem ceil_sqrt_225_eq_15 : 
  ⌈Real.sqrt 225⌉ = 15 := 
by sorry

end ceil_sqrt_225_eq_15_l243_243962


namespace find_tan_B_l243_243072

-- Definitions based on conditions
def a : ℝ := 3
def b : ℝ := 2
def A : ℝ := Real.pi / 6

-- Problem statement
theorem find_tan_B : 
  ∃ B : ℝ, (3 / Real.sin (Real.pi / 6) = 2 / Real.sin B ∧ 
  B < Real.pi / 2 ∧ 
  Real.tan B = Real.sqrt 2 / 4) :=
sorry

end find_tan_B_l243_243072


namespace profit_percent_l243_243790

-- Definitions for the given conditions
variables (P C : ℝ)
-- Condition given: selling at (2/3) of P results in a loss of 5%, i.e., (2/3) * P = 0.95 * C
def condition : Prop := (2 / 3) * P = 0.95 * C

-- Theorem statement: Given the condition, the profit percent when selling at price P is 42.5%
theorem profit_percent (h : condition P C) : ((P - C) / C) * 100 = 42.5 :=
sorry

end profit_percent_l243_243790


namespace time_relationship_l243_243792

variable (T x : ℝ)
variable (h : T = x + (2/6) * x)

theorem time_relationship : T = (4/3) * x := by 
sorry

end time_relationship_l243_243792


namespace smallest_four_digit_div_by_53_l243_243624

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l243_243624


namespace square_not_covered_by_circles_l243_243997

noncomputable def area_uncovered_by_circles : Real :=
  let side_length := 2
  let square_area := (side_length^2 : Real)
  let radius := 1
  let circle_area := Real.pi * radius^2
  let quarter_circle_area := circle_area / 4
  let total_circles_area := 4 * quarter_circle_area
  square_area - total_circles_area

theorem square_not_covered_by_circles :
  area_uncovered_by_circles = 4 - Real.pi := sorry

end square_not_covered_by_circles_l243_243997


namespace guo_can_pay_exactly_l243_243363

theorem guo_can_pay_exactly (
  x y z : ℕ
) (h : 10 * x + 20 * y + 50 * z = 20000) : ∃ a b c : ℕ, a + 2 * b + 5 * c = 1000 :=
sorry

end guo_can_pay_exactly_l243_243363


namespace min_tan_ABC_l243_243437

variable (A B C : ℝ)

noncomputable def sin_angle (x : ℝ) := real.sin x
noncomputable def tan_angle (x : ℝ) := real.tan x

def is_acute (A B C : ℝ) : Prop :=
  A > 0 ∧ A < π / 2 ∧ B > 0 ∧ B < π / 2 ∧ C > 0 ∧ C < π / 2

theorem min_tan_ABC 
  (h_acute : is_acute A B C)
  (h_sin : sin_angle A = 2 * sin_angle B * sin_angle C) :
  ∃t, t = tan_angle A * tan_angle B * tan_angle C ∧ t = 8 := 
sorry

end min_tan_ABC_l243_243437


namespace least_third_side_of_right_triangle_l243_243385

theorem least_third_side_of_right_triangle (a b : ℕ) (ha : a = 8) (hb : b = 15) :
  ∃ c : ℝ, c = real.sqrt (b^2 - a^2) ∧ c = real.sqrt 161 :=
by {
  -- We state the conditions
  have h8 : (8 : ℝ) = a, from by {rw ha},
  have h15 : (15 : ℝ) = b, from by {rw hb},

  -- The theorem states that such a c exists
  use (real.sqrt (15^2 - 8^2)),

  -- We need to show the properties of c
  split,
  { 
    -- Showing that c is the sqrt of the difference of squares of b and a
    rw [←h15, ←h8],
    refl 
  },
  {
    -- Showing that c is sqrt(161)
    calc
       real.sqrt (15^2 - 8^2)
         = real.sqrt (225 - 64) : by norm_num
     ... = real.sqrt 161 : by norm_num
  }
}
sorry

end least_third_side_of_right_triangle_l243_243385


namespace second_rectangle_from_left_is_R_l243_243990

structure Rectangle :=
  (name : String)
  (w : Int)
  (x : Int)
  (y : Int)
  (z : Int)

def P : Rectangle := ⟨"P", 5, 0, 7, 10⟩
def Q : Rectangle := ⟨"Q", 0, 2, 5, 8⟩
def R : Rectangle := ⟨"R", 4, 9, 3, 1⟩
def S : Rectangle := ⟨"S", 8, 6, 2, 9⟩
def T : Rectangle := ⟨"T", 10, 3, 6, 0⟩

def rectangles : List Rectangle := [P, Q, R, S, T]

def compareRectangles (r1 r2: Rectangle) : Ordering :=
  if r1.w < r2.w then .lt else if r1.w > r2.w then .gt else .eq

def sorted_rectangles : List Rectangle := rectangles.qsort compareRectangles

theorem second_rectangle_from_left_is_R :
  sorted_rectangles.nth 1 = some R := by
  sorry

end second_rectangle_from_left_is_R_l243_243990


namespace least_third_side_of_right_triangle_l243_243384

theorem least_third_side_of_right_triangle (a b : ℕ) (ha : a = 8) (hb : b = 15) :
  ∃ c : ℝ, c = real.sqrt (b^2 - a^2) ∧ c = real.sqrt 161 :=
by {
  -- We state the conditions
  have h8 : (8 : ℝ) = a, from by {rw ha},
  have h15 : (15 : ℝ) = b, from by {rw hb},

  -- The theorem states that such a c exists
  use (real.sqrt (15^2 - 8^2)),

  -- We need to show the properties of c
  split,
  { 
    -- Showing that c is the sqrt of the difference of squares of b and a
    rw [←h15, ←h8],
    refl 
  },
  {
    -- Showing that c is sqrt(161)
    calc
       real.sqrt (15^2 - 8^2)
         = real.sqrt (225 - 64) : by norm_num
     ... = real.sqrt 161 : by norm_num
  }
}
sorry

end least_third_side_of_right_triangle_l243_243384


namespace problem_statement_l243_243055

theorem problem_statement (x y : ℝ) (h1 : |x| + x - y = 16) (h2 : x - |y| + y = -8) : x + y = -8 := sorry

end problem_statement_l243_243055


namespace cube_coloring_l243_243923

-- Defining the cube and its properties
def Cube := {faces : Finset (Fin 6) // faces.card = 6}
def Face := Fin 4 -- Each face is divided into 4 equal squares
def colors := Fin 3 -- Three colors available

-- Conditions
def adjacent_squares (a b : Face) : Prop := -- Adjacent squares should share a side
  sorry -- Detailed implementation of adjacency condition

axiom three_colors (a b : Face) (h : adjacent_squares a b) : colors → colors → Prop
-- If two squares are adjacent, then they must be painted differently

noncomputable def Paint (f : Cube) : (Finset (Face × colors)) := sorry
-- The painting function which assigns colors to faces

-- Statement to prove
theorem cube_coloring (c : Cube) :
  ∃ f : (Finset (Face × colors)), Paint c f ∧
  ∀ col : colors, ∃ s : Finset Face, (∀ x ∈ s, (x, col) ∈ f) ∧ s.card = 8 :=
begin
  sorry
end

end cube_coloring_l243_243923


namespace sin_half_theta_squared_eq_l243_243344

theorem sin_half_theta_squared_eq :
  ∀ θ : ℝ, (cos θ = -3 / 5) → sin (θ / 2) ^ 2 = 4 / 5 :=
by
  intros θ hcos
  -- The statement is purely for structure, the proof would go here.
  sorry

end sin_half_theta_squared_eq_l243_243344


namespace correct_statement_B_l243_243798

def flowchart_start_points : Nat := 1
def flowchart_end_points : Bool := True  -- Represents one or multiple end points (True means multiple possible)

def program_flowchart_start_points : Nat := 1
def program_flowchart_end_points : Nat := 1

def structure_chart_start_points : Nat := 1
def structure_chart_end_points : Bool := True  -- Represents one or multiple end points (True means multiple possible)

theorem correct_statement_B :
  (program_flowchart_start_points = 1 ∧ program_flowchart_end_points = 1) :=
by 
  sorry

end correct_statement_B_l243_243798


namespace max_elements_no_pair_sum_divisible_by_seven_l243_243108

theorem max_elements_no_pair_sum_divisible_by_seven :
  ∃ (T : Finset (Fin 100)), 
  (∀ x y ∈ T, x ≠ y → ¬ (x + y) % 7 = 0) ∧ T.card = 72 :=
sorry

end max_elements_no_pair_sum_divisible_by_seven_l243_243108


namespace general_formula_a_n_prove_Tn_range_k_l243_243322

variable (a_n S_n b_n T_n : ℕ → ℕ)
variable (k : ℝ)
variable (n : ℕ)
variable (f : ℕ → ℝ)

-- Condition: Sn = 2an - 3
def condition_Sn (n : ℕ) : Prop := S_n n = 2 * a_n n - 3

-- a_n formula
def formula_a_n (n : ℕ) : ℕ := 3 * 2^(n-1)

-- First question: Prove the general formula for a_n
theorem general_formula_a_n (n : ℕ) (h₁ : condition_Sn n) (h₂ : S_n 1 = 2 * a_n 1 - 3) :
  a_n n = formula_a_n n :=
sorry

-- Define bn
def b_n (n : ℕ) : ℕ := (n-1) * a_n n

-- Sum Tn condition
def sum_Tn (n : ℕ) : Prop :=
  T_n n = 3 * (n-2) * 2^n + 6

-- Second question(i): Prove T_n
theorem prove_Tn (n : ℕ) (h : ∀ n, b_n n = (n-1) * formula_a_n n) : 
  sum_Tn n :=
sorry

-- f(n) function as used in the solution
def f (n : ℕ) : ℝ := 2 * (n-2) * (1 - 8 / (3 * 2^(n-1)))

-- Second question(ii): Find the range of k
theorem range_k (h : ∀ n, T_n n > k * formula_a_n n + 16 * n - 26) :
  k < 0 :=
sorry

end general_formula_a_n_prove_Tn_range_k_l243_243322


namespace smallest_four_digit_divisible_by_53_l243_243667

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243667


namespace smallest_four_digit_divisible_by_53_l243_243659

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l243_243659


namespace f_increasing_find_a_l243_243027

-- Definition and conditions
def f (a x : ℝ) : ℝ := (1 / a) - (1 / x)

-- Problem Statement 1: Prove that f(x) is increasing on (0, +∞)
theorem f_increasing (a : ℝ) (h_a : a > 0) : (∀ (x1 x2 : ℝ), x2 > x1 → x1 > 0 → x2 > 0 → f a x2 > f a x1) :=
by sorry

-- Problem Statement 2: Given f(1/2) = 1/2 and f(2) = 2, find a
theorem find_a : (∃ a : ℝ, 0 < a ∧ f a (1 / 2) = 1 / 2 ∧ f a 2 = 2) :=
by sorry

end f_increasing_find_a_l243_243027


namespace unsold_books_l243_243844

-- Definitions from conditions
def books_total : ℕ := 150
def books_sold : ℕ := (2 / 3) * books_total
def book_price : ℕ := 5
def total_received : ℕ := 500

-- Proof statement
theorem unsold_books :
  (books_sold * book_price = total_received) →
  (books_total - books_sold = 50) :=
by
  sorry

end unsold_books_l243_243844


namespace vectors_not_collinear_l243_243258

def a : ℝ × ℝ × ℝ := (2, -1, 4)
def b : ℝ × ℝ × ℝ := (3, -7, -6)
def c1 := (2 * a.1 - 3 * b.1, 2 * a.2 - 3 * b.2, 2 * a.3 - 3 * b.3)
def c2 := (3 * a.1 - 2 * b.1, 3 * a.2 - 2 * b.2, 3 * a.3 - 2 * b.3)

theorem vectors_not_collinear : ¬ ∃ γ : ℝ, c1 = (γ * c2.1, γ * c2.2, γ * c2.3) :=
by
  sorry

end vectors_not_collinear_l243_243258


namespace least_third_side_of_right_triangle_l243_243404

theorem least_third_side_of_right_triangle {a b c : ℝ} 
  (h1 : a = 8) 
  (h2 : b = 15) 
  (h3 : c = Real.sqrt (8^2 + 15^2) ∨ c = Real.sqrt (15^2 - 8^2)) : 
  c = Real.sqrt 161 :=
by {
  intros h1 h2 h3,
  rw [h1, h2] at h3,
  cases h3,
  { exfalso, preciesly contradiction occurs because sqrt (8^2 + 15^2) is not sqrt161, 
   rw [← h3],
   norm_num,},
  { exact h3},
  
}

end least_third_side_of_right_triangle_l243_243404


namespace average_rounds_rounded_eq_4_l243_243551

def rounds_distribution : List (Nat × Nat) := [(1, 4), (2, 3), (4, 4), (5, 2), (6, 6)]

def total_rounds : Nat := rounds_distribution.foldl (λ acc (rounds, golfers) => acc + rounds * golfers) 0

def total_golfers : Nat := rounds_distribution.foldl (λ acc (_, golfers) => acc + golfers) 0

def average_rounds : Float := total_rounds.toFloat / total_golfers.toFloat

theorem average_rounds_rounded_eq_4 : Float.round average_rounds = 4 := by
  sorry

end average_rounds_rounded_eq_4_l243_243551


namespace loss_recorded_as_negative_l243_243057

-- Define the condition that a profit of 100 yuan is recorded as +100 yuan
def recorded_profit (p : ℤ) : Prop :=
  p = 100

-- Define the condition about how a profit is recorded
axiom profit_condition : recorded_profit 100

-- Define the function for recording profit or loss
def record (x : ℤ) : ℤ :=
  if x > 0 then x
  else -x

-- Theorem: If a profit of 100 yuan is recorded as +100 yuan, then a loss of 50 yuan is recorded as -50 yuan.
theorem loss_recorded_as_negative : ∀ x: ℤ, (x < 0) → record x = -x :=
by
  intros x h
  unfold record
  simp [h]
  -- sorry indicates the proof is not provided
  sorry

end loss_recorded_as_negative_l243_243057


namespace pb_distance_l243_243836

theorem pb_distance (a b c d PA PD PC PB : ℝ)
  (hPA : PA = 5)
  (hPD : PD = 6)
  (hPC : PC = 7)
  (h1 : a^2 + b^2 = PA^2)
  (h2 : b^2 + c^2 = PC^2)
  (h3 : c^2 + d^2 = PD^2)
  (h4 : d^2 + a^2 = PB^2) :
  PB = Real.sqrt 38 := by
  sorry

end pb_distance_l243_243836


namespace actual_average_height_l243_243223

theorem actual_average_height {n : ℕ} {incorrect_avg_height incorrect_height actual_height : ℝ} 
  (h_n : n = 35)
  (h_incorrect_avg : incorrect_avg_height = 183)
  (h_incorrect_height: incorrect_height = 166)
  (h_actual_height: actual_height = 106) :
  let incorrect_total_height := incorrect_avg_height * n in
  let overestimated_amount := incorrect_height - actual_height in
  let correct_total_height := incorrect_total_height - overestimated_amount in
  let actual_avg_height := correct_total_height / n in
  actual_avg_height = 181.29 := by 
{
  sorry
}

end actual_average_height_l243_243223


namespace greatest_missed_problems_l243_243424

theorem greatest_missed_problems (total_problems : ℕ) (passing_percentage : ℝ) (missed_problems : ℕ) : 
  total_problems = 50 ∧ passing_percentage = 0.85 → missed_problems = 7 :=
by
  sorry

end greatest_missed_problems_l243_243424


namespace λ_correct_μ_correct_l243_243043

-- Part Ⅰ: Prove λ = -9
def λ_problem (a b : ℝ × ℝ) (λ : ℝ) : Prop :=
  let ab := (a.1 - 2 * b.1, a.2 - 2 * b.2)
  let lb := (2 * λ + 4, λ - 3)
  ab.1 * lb.1 + ab.2 * lb.2 = 0 → λ = -9

-- Part Ⅱ: Prove the range of μ
def μ_problem (a c : ℝ × ℝ) (μ : ℝ) : Prop :=
  let dot_product := a.1 * c.1 + a.2 * c.2
  let determinant := a.1 * c.2 - a.2 * c.1
  (dot_product > 0 ∧ determinant ≠ 0) → (μ > -2 ∧ μ ≠ 1 / 2)

-- Definitions of vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (4, -3)
def c (μ : ℝ) : ℝ × ℝ := (1, μ)

-- Assertion for λ
theorem λ_correct : λ_problem a b (-9) :=
by simp [a, b, λ_problem]; sorry

-- Assertion for μ
theorem μ_correct (μ : ℝ) : μ_problem a (c μ) μ :=
by simp [a, c, μ_problem]; sorry

end λ_correct_μ_correct_l243_243043


namespace max_travel_distance_l243_243999

theorem max_travel_distance (front_tire_lifespan : ℕ) (rear_tire_lifespan : ℕ) 
  (h₁ : front_tire_lifespan = 24000) (h₂ : rear_tire_lifespan = 36000) : 
  ∃ (D : ℕ), D = 28800 :=
begin
  sorry
end

end max_travel_distance_l243_243999


namespace ceil_sqrt_225_eq_15_l243_243963

theorem ceil_sqrt_225_eq_15 : 
  ⌈Real.sqrt 225⌉ = 15 := 
by sorry

end ceil_sqrt_225_eq_15_l243_243963


namespace residue_mod_2000_l243_243474

def T : ℤ := (List.range 2000).map (λ n, if n % 2 = 0 then n + 1 else -(n + 1)).sum

theorem residue_mod_2000 : T % 2000 = 1000 :=
by
  sorry

end residue_mod_2000_l243_243474


namespace smallest_four_digit_divisible_by_53_l243_243702

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243702


namespace smallest_four_digit_divisible_by_53_l243_243756

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243756


namespace ceil_sqrt_225_eq_15_l243_243968

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := by
  sorry

end ceil_sqrt_225_eq_15_l243_243968


namespace sean_net_profit_l243_243526

noncomputable def total_cost (num_patches : ℕ) (cost_per_patch : ℝ) : ℝ :=
  num_patches * cost_per_patch

noncomputable def total_revenue (num_patches : ℕ) (selling_price_per_patch : ℝ) : ℝ :=
  num_patches * selling_price_per_patch

noncomputable def net_profit (total_revenue : ℝ) (total_cost : ℝ) : ℝ :=
  total_revenue - total_cost

-- Variables based on conditions
def num_patches := 100
def cost_per_patch := 1.25
def selling_price_per_patch := 12.00

theorem sean_net_profit : net_profit (total_revenue num_patches selling_price_per_patch) (total_cost num_patches cost_per_patch) = 1075 :=
by
  sorry

end sean_net_profit_l243_243526


namespace number_of_servings_l243_243239

def servings (total: ℚ) (serving_size: ℚ) : ℚ := total / serving_size

theorem number_of_servings :
  let container_honey := (47 * 3 + 1) / 3 in -- 47 + 1/3 as an improper fraction
  let serving_size := (1 * 6 + 1) / 6 in -- 1 + 1/6 as an improper fraction
  let expected_servings := (40 * 21 + 12) / 21 in -- 40 + 12/21 as an improper fraction
  servings container_honey serving_size = expected_servings := by
  sorry

end number_of_servings_l243_243239


namespace max_ab_l243_243354

theorem max_ab (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (h_line : ∃ (x y : ℝ), (x^2 + y^2 - 4 * x + 2 * y + 1 = 0) ∧ (ax - by = 2)) :
  ab ≤ 0.5 := 
sorry

end max_ab_l243_243354


namespace base6_addition_l243_243287

theorem base6_addition {C D : ℕ} (h1 : C + 5 = 10) (h2 : D + 2 = 5) : C + D = 8 :=
by
  calc
    C + D = 5 + 3 : by rw [h1, h2]
         ... = 8   : by norm_num

end base6_addition_l243_243287


namespace marissas_sunflower_height_l243_243501

def height_of_marissas_sunflower (sister_height_feet : ℤ) (sister_height_inches : ℤ) (additional_inches : ℤ) : Prop :=
  (sister_height_feet = 4) →
  (sister_height_inches = 3) →
  (additional_inches = 21) →
  sister_height_feet * 12 + sister_height_inches + additional_inches = 72

-- Prove that Marissa's sunflower height in feet is 6
theorem marissas_sunflower_height :
  height_of_marissas_sunflower 4 3 21 →
  72 / 12 = 6 :=
by
  assume h,
  rw Nat.div_eq_of_eq_mul_left sorry,
  sorry

end marissas_sunflower_height_l243_243501


namespace smallest_four_digit_divisible_by_53_l243_243616

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l243_243616


namespace volume_of_solid_l243_243906

-- Define the region and rotations
def volume_partition_1 (r h : ℝ) : ℝ :=
  π * r^2 * h

def volume_partition_2 (r h : ℝ) : ℝ :=
  π * r^2 * h

def total_volume (V1 V2 : ℝ) : ℝ :=
  V1 + V2

theorem volume_of_solid :
  let r1 := 6 in
  let h1 := 1 in
  let V1 := volume_partition_1 r1 h1 in
  let r2 := 3 in
  let h2 := 2 in
  let V2 := volume_partition_2 r2 h2 in
  total_volume V1 V2 = 54 * π :=
by
  sorry

end volume_of_solid_l243_243906


namespace smallest_four_digit_multiple_of_53_l243_243708

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243708


namespace total_heads_eq_fifteen_l243_243245

-- Definitions for types of passengers and their attributes
def cats_heads : Nat := 7
def cats_legs : Nat := 7 * 4
def total_legs : Nat := 43
def captain_heads : Nat := 1
def captain_legs : Nat := 1

noncomputable def crew_heads (C : Nat) : Nat := C
noncomputable def crew_legs (C : Nat) : Nat := 2 * C

theorem total_heads_eq_fifteen : 
  ∃ (C : Nat),
    cats_legs + crew_legs C + captain_legs = total_legs ∧
    cats_heads + crew_heads C + captain_heads = 15 :=
by
  sorry

end total_heads_eq_fifteen_l243_243245


namespace derivative_f_l243_243161

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.cos x

theorem derivative_f (x : ℝ) : deriv f x = 2 * x * Real.cos x - x^2 * Real.sin x :=
by
  sorry

end derivative_f_l243_243161


namespace smallest_four_digit_divisible_by_53_l243_243651

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l243_243651


namespace Jack_can_fit_12_cans_per_row_l243_243467

theorem Jack_can_fit_12_cans_per_row
    (rows_per_shelf : ℕ)
    (shelves_per_closet : ℕ)
    (total_cans : ℕ)
    (h1 : rows_per_shelf = 4)
    (h2 : shelves_per_closet = 10)
    (h3 : total_cans = 480) :
    ∃ (cans_per_row : ℕ), cans_per_row = 12 :=
by {
  let total_cans_per_shelf := total_cans / shelves_per_closet,
  let cans_per_row := total_cans_per_shelf / rows_per_shelf,
  use cans_per_row,
  have h_total_cans_per_shelf : total_cans_per_shelf = 48 := sorry,
  have h_cans_per_row : cans_per_row = 12 := sorry,
  exact h_cans_per_row,
}

end Jack_can_fit_12_cans_per_row_l243_243467


namespace evaluate_ceil_sqrt_225_l243_243941

def ceil (x : ℝ) : ℤ :=
  if h : ∃ n : ℤ, n ≤ x ∧ x < n + 1 then
    classical.some h
  else
    0

theorem evaluate_ceil_sqrt_225 : ceil (Real.sqrt 225) = 15 := 
sorry

end evaluate_ceil_sqrt_225_l243_243941


namespace smallest_four_digit_divisible_by_53_l243_243737

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243737


namespace shaniqua_earnings_correct_l243_243528

noncomputable def calc_earnings : ℝ :=
  let haircut_tuesday := 5 * 10
  let haircut_normal := 5 * 12
  let styling_vip := (6 * 25) * (1 - 0.2)
  let styling_regular := 4 * 25
  let coloring_friday := (7 * 35) * (1 - 0.15)
  let coloring_normal := 3 * 35
  let treatment_senior := (3 * 50) * (1 - 0.1)
  let treatment_other := 4 * 50
  haircut_tuesday + haircut_normal + styling_vip + styling_regular + coloring_friday + coloring_normal + treatment_senior + treatment_other

theorem shaniqua_earnings_correct : calc_earnings = 978.25 := by
  sorry

end shaniqua_earnings_correct_l243_243528


namespace smallest_four_digit_multiple_of_53_l243_243717

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243717


namespace intersection_of_sets_l243_243009

theorem intersection_of_sets (x : ℝ) :
  (\{x ∣ 2 * x - 1 > 0\} ∩ \{x ∣ sqrt x < 2\}) = \{x ∣ 1/2 < x ∧ x < 4\} := sorry

end intersection_of_sets_l243_243009


namespace evaluate_nested_fraction_l243_243285

theorem evaluate_nested_fraction :
  (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))) = 8 / 21 := 
begin
  sorry
end

end evaluate_nested_fraction_l243_243285


namespace marbles_in_each_box_l243_243191

theorem marbles_in_each_box (total_marbles : ℕ) (num_boxes : ℕ) (h_total : total_marbles = 48) (h_boxes : num_boxes = 6) : total_marbles / num_boxes = 8 :=
by
  rw [h_total, h_boxes]
  exact Nat.div_eq_of_eq_mul (by norm_num)

end marbles_in_each_box_l243_243191


namespace smallest_four_digit_divisible_by_53_l243_243689

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l243_243689


namespace rate_per_meter_fencing_l243_243982

noncomputable def rate_per_meter (d : ℝ) (total_cost : ℝ) : ℝ :=
  let C := Real.pi * d in
  total_cost / C

theorem rate_per_meter_fencing :
  rate_per_meter 16 150.79644737231007 = 3 :=
by
  -- Here we would insert the proof
  sorry

end rate_per_meter_fencing_l243_243982


namespace line_region_intersection_range_l243_243471

theorem line_region_intersection_range (b : ℝ) :
  (∃ x y : ℝ, (x-1)^2 + y^2 ≤ 1 ∧ x + sqrt 3 * y + b = 0) ↔ -3 ≤ b ∧ b ≤ 1 :=
sorry

end line_region_intersection_range_l243_243471


namespace smallest_four_digit_multiple_of_53_l243_243782

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l243_243782


namespace mrs_doe_inheritance_l243_243469

noncomputable def calculateInheritance (totalTaxes : ℝ) : ℝ :=
  totalTaxes / 0.3625

theorem mrs_doe_inheritance (h : 0.3625 * calculateInheritance 15000 = 15000) :
  calculateInheritance 15000 = 41379 :=
by
  unfold calculateInheritance
  field_simp
  norm_cast
  sorry

end mrs_doe_inheritance_l243_243469


namespace books_still_to_read_l243_243190

-- Define the given conditions
def total_books : ℕ := 22
def books_read : ℕ := 12

-- State the theorem to be proven
theorem books_still_to_read : total_books - books_read = 10 := 
by
  -- skipping the proof
  sorry

end books_still_to_read_l243_243190


namespace relationship_among_a_b_c_l243_243313

noncomputable def a : ℝ := 0.7^6
noncomputable def b : ℝ := 6^0.7
noncomputable def c : ℝ := Real.log 6 / Real.log 0.7

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  sorry

end relationship_among_a_b_c_l243_243313


namespace quadratic_inequality_solution_l243_243280

theorem quadratic_inequality_solution :
  ∀ x : ℝ, (3 * x^2 - 5 * x - 2 < 0) ↔ (-1/3 < x ∧ x < 2) :=
by
  sorry

end quadratic_inequality_solution_l243_243280


namespace arithmetic_mean_of_int_range_l243_243202

-- Define the range of integers
def int_range := list.range' (-4) 10 -- List of integers from -4 to 5

-- Define the arithmetic mean calculation
def arithmetic_mean (l : list ℤ) : ℝ :=
(l.sum : ℝ) / (l.length : ℝ)

-- The theorem statement
theorem arithmetic_mean_of_int_range : arithmetic_mean int_range = 0.5 :=
begin
  sorry -- proof to be filled in
end

end arithmetic_mean_of_int_range_l243_243202


namespace smallest_four_digit_divisible_by_53_l243_243639

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l243_243639


namespace tangent_and_normal_l243_243996

noncomputable def curve (a t : ℝ) : ℝ × ℝ :=
  (3 * a * t / (1 + t^2), 3 * a * t^2 / (1 + t^2))

noncomputable def tangent_line (a x y : ℝ) : Prop :=
  y = -4/3 * x + 4 * a

noncomputable def normal_line (a x y : ℝ) : Prop :=
  y = 3/4 * x + 3 * a / 2

theorem tangent_and_normal (a : ℝ) :
  ∃ x₀ y₀ : ℝ,
  curve a 2 = (x₀, y₀) ∧
  tangent_line a x₀ y₀ ∧
  normal_line a x₀ y₀ :=
begin
  sorry
end

end tangent_and_normal_l243_243996


namespace min_value_of_squares_l243_243113

theorem min_value_of_squares (a b c t : ℝ) (h : a + b + c = t) : 
  a^2 + b^2 + c^2 ≥ t^2 / 3 ∧ (∃ (a' b' c' : ℝ), a' = b' ∧ b' = c' ∧ a' + b' + c' = t ∧ a'^2 + b'^2 + c'^2 = t^2 / 3) := 
by
  sorry

end min_value_of_squares_l243_243113


namespace xyz_positive_and_distinct_l243_243535

theorem xyz_positive_and_distinct (a b x y z : ℝ)
  (h₁ : x + y + z = a)
  (h₂ : x^2 + y^2 + z^2 = b^2)
  (h₃ : x * y = z^2)
  (ha_pos : a > 0)
  (hb_condition : b^2 < a^2 ∧ a^2 < 3*b^2) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z :=
by
  sorry

end xyz_positive_and_distinct_l243_243535


namespace students_spending_more_than_90_minutes_l243_243310

def sample_times : List ℕ :=
  [75, 80, 85, 65, 95, 100, 70, 55, 65, 75, 85, 110, 120, 80, 85, 80, 75, 90, 90, 95, 70, 60, 60, 75, 90, 95, 65, 75, 80, 80]

def sample_size : ℕ := 30
def total_students : ℕ := 2100

def count_students_above_90 (times : List ℕ) : ℕ :=
  times.count (λ t => t ≥ 90)

def estimated_students_above_90 (sample_count : ℕ) (total_students : ℕ) (sample_size : ℕ) : ℕ :=
  (sample_count * total_students) / sample_size

theorem students_spending_more_than_90_minutes :
  estimated_students_above_90 (count_students_above_90 sample_times) total_students sample_size = 630 :=
by
  simp [sample_times, sample_size, total_students, count_students_above_90, estimated_students_above_90]
  sorry


end students_spending_more_than_90_minutes_l243_243310


namespace exists_pos_int_n_l243_243153

def sequence_x (x : ℕ → ℝ) : Prop :=
  ∀ n, x (n + 2) = x n + (x (n + 1))^2

def sequence_y (y : ℕ → ℝ) : Prop :=
  ∀ n, y (n + 2) = y n^2 + y (n + 1)

def positive_initial_conditions (x y : ℕ → ℝ) : Prop :=
  x 1 > 1 ∧ x 2 > 1 ∧ y 1 > 1 ∧ y 2 > 1

theorem exists_pos_int_n (x y : ℕ → ℝ) (hx : sequence_x x) (hy : sequence_y y) 
  (ini : positive_initial_conditions x y) : ∃ n, x n > y n := 
sorry

end exists_pos_int_n_l243_243153


namespace ceil_sqrt_225_eq_15_l243_243956

theorem ceil_sqrt_225_eq_15 : ⌈ Real.sqrt 225 ⌉ = 15 :=
by
  sorry

end ceil_sqrt_225_eq_15_l243_243956


namespace counting_arithmetic_progressions_l243_243443

open Finset

/-- The number of ways to choose 4 numbers from the first 1000 natural numbers to form an increasing arithmetic progression is 166167. -/
theorem counting_arithmetic_progressions :
  let n := 1000 in
  let count := ∑ d in range 334, n - 3*d in
  count = 166167 :=
by
  sorry

end counting_arithmetic_progressions_l243_243443


namespace smallest_four_digit_multiple_of_53_l243_243785

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l243_243785


namespace remaining_ribbon_l243_243093

theorem remaining_ribbon 
  (total_length : ℕ)
  (p1 p2 p3 p4 p5 : ℕ)
  (pattern : List (ℕ × ℕ))
  (H_total_length : total_length = 30)
  (H_p1 : p1 = 2) (H_p2 : p2 = 4) (H_p3 : p3 = 6) 
  (H_p4 : p4 = 8) (H_p5 : p5 = 10)
  (H_pattern : pattern = [(3, p1), (2, p2), (4, p3), (1, p4), (2, p5)]) :
  let used_ribbon := 3 * p1 + 2 * p2 + 2 * p3 in
  total_length - used_ribbon = 4 :=
by
  sorry

end remaining_ribbon_l243_243093


namespace equation_of_line_through_A_parallel_to_given_line_l243_243553

theorem equation_of_line_through_A_parallel_to_given_line :
  ∃ c : ℝ, 
    (∀ x y : ℝ, 2 * x - y + c = 0 ↔ ∃ a b : ℝ, a = -1 ∧ b = 0 ∧ 2 * a - b + 1 = 0) :=
sorry

end equation_of_line_through_A_parallel_to_given_line_l243_243553


namespace ceil_sqrt_225_eq_15_l243_243965

theorem ceil_sqrt_225_eq_15 : 
  ⌈Real.sqrt 225⌉ = 15 := 
by sorry

end ceil_sqrt_225_eq_15_l243_243965


namespace prime_divides_3np_minus_3n1_l243_243115

theorem prime_divides_3np_minus_3n1 (p n : ℕ) (hp : Prime p) : p ∣ (3^(n + p) - 3^(n + 1)) :=
sorry

end prime_divides_3np_minus_3n1_l243_243115


namespace value_of_f_at_2_l243_243031

def f : ℝ → ℝ :=
λ x, if x < 0 then 2^x else if x = 0 then f (-1) + 1 else f (x - 1) + 1

theorem value_of_f_at_2 : f 2 = 7 / 2 := by
  sorry

end value_of_f_at_2_l243_243031


namespace pyramid_coloring_methods_l243_243000

theorem pyramid_coloring_methods : 
  ∀ (P A B C D : ℕ),
    (P ≠ A) ∧ (P ≠ B) ∧ (P ≠ C) ∧ (P ≠ D) ∧
    (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧
    (B ≠ C) ∧ (B ≠ D) ∧ (C ≠ D) ∧
    (P < 5) ∧ (A < 5) ∧ (B < 5) ∧ (C < 5) ∧ (D < 5) →
  ∃! (num_methods : ℕ), num_methods = 420 :=
by
  sorry

end pyramid_coloring_methods_l243_243000


namespace chelsea_victory_shots_required_l243_243075

theorem chelsea_victory_shots_required
  (L : ℕ) -- Chelsea's current score
  (lead : ℕ := 60) -- Chelsea's lead
  (points_per_shot : Fin 5 := {10, 7, 3, 1, 0}) -- possible scores for each shot
  (min_score_chelsa : ℕ := 3) -- minimum score per shot for Chelsea
  (remaining_shots : ℕ := 60) -- remaining shots
  (opponent_min_score : ℕ := 1) -- opponent's lowest possible shot score
  (opponent_remaining_optimal_score : ℕ := 10) -- opponent's assumed optimal score per shot
  -- opponent's maximum possible score after all shots
  (opponent_possible_max_score := L - lead + remaining_shots * opponent_remaining_optimal_score) :
  ∀ n : ℕ, n ≥ 52 → (L + 10 * n + 3 * (remaining_shots - n) > opponent_possible_max_score) :=
by sorry

end chelsea_victory_shots_required_l243_243075


namespace net_profit_correct_l243_243523

-- Define the conditions
def unit_price : ℝ := 1.25
def selling_price : ℝ := 12
def num_patches : ℕ := 100

-- Define the required total cost
def total_cost : ℝ := num_patches * unit_price

-- Define the required total revenue
def total_revenue : ℝ := num_patches * selling_price

-- Define the net profit calculation
def net_profit : ℝ := total_revenue - total_cost

-- The theorem we need to prove
theorem net_profit_correct : net_profit = 1075 := by
    sorry

end net_profit_correct_l243_243523


namespace point_in_first_quadrant_of_complex_number_l243_243546

open Complex

theorem point_in_first_quadrant_of_complex_number :
  let z := complex.sin (real.pi * (100 / 180)) - complex.I * complex.cos (real.pi * (100 / 180))
  let Z := (z.re, z.im)
  (0 < z.re) ∧ (0 < z.im) :=
by
  let z := complex.sin (real.pi * (100 / 180)) - complex.I * complex.cos (real.pi * (100 / 180))
  let Z := (z.re, z.im)
  sorry

end point_in_first_quadrant_of_complex_number_l243_243546


namespace smallest_four_digit_divisible_by_53_l243_243684

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l243_243684


namespace ninth_number_l243_243544

theorem ninth_number (S1 S2 Total N : ℕ)
  (h1 : S1 = 9 * 56)
  (h2 : S2 = 9 * 63)
  (h3 : Total = 17 * 59)
  (h4 : Total = S1 + S2 - N) :
  N = 68 :=
by 
  -- The proof is omitted, only the statement is needed.
  sorry

end ninth_number_l243_243544


namespace pos_int_solutions_l243_243175

-- defining the condition for a positive integer solution to the equation
def is_pos_int_solution (x y : Int) : Prop :=
  5 * x + 2 * y = 25 ∧ x > 0 ∧ y > 0

-- stating the theorem for positive integer solutions of the equation
theorem pos_int_solutions : 
  ∃ x y : Int, is_pos_int_solution x y ∧ ((x = 1 ∧ y = 10) ∨ (x = 3 ∧ y = 5)) :=
by
  sorry

end pos_int_solutions_l243_243175


namespace average_speed_uphill_l243_243867

-- Definitions based on given conditions
def speed_flat : ℝ := 20 -- miles per hour
def time_flat : ℝ := 4.5 -- hours
def distance_flat : ℝ := speed_flat * time_flat -- miles

def time_uphill : ℝ := 2.5 -- hours
def time_downhill : ℝ := 1.5 -- hours
def speed_downhill : ℝ := 24 -- miles per hour
def distance_downhill : ℝ := speed_downhill * time_downhill -- miles

def distance_walk : ℝ := 8 -- miles
def total_distance : ℝ := 164 -- miles
def distance_bike : ℝ := total_distance - distance_walk -- miles

-- The goal is to prove this statement
theorem average_speed_uphill :
  let distance_uphill := time_uphill * x in
  distance_flat + distance_uphill + distance_downhill = distance_bike →
  x = 12 :=
by
  intro h
  sorry -- Proof not required

end average_speed_uphill_l243_243867


namespace ceil_sqrt_225_eq_15_l243_243964

theorem ceil_sqrt_225_eq_15 : 
  ⌈Real.sqrt 225⌉ = 15 := 
by sorry

end ceil_sqrt_225_eq_15_l243_243964


namespace mike_picked_peaches_l243_243131

theorem mike_picked_peaches :
  let initial_peaches := 34
  let total_peaches := 86
  (total_peaches - initial_peaches) = 52 :=
by
  have initial_peaches : ℕ := 34
  have total_peaches : ℕ := 86
  calc
    total_peaches - initial_peaches
      = 86 - 34 : by
        rfl
      = 52 : by
        rfl

end mike_picked_peaches_l243_243131


namespace solve_diamond_eq_l243_243841

def diamond (a b : ℝ) : ℝ := a / b

theorem solve_diamond_eq : ∃ x : ℝ, diamond 504 (diamond 12 x) = 50 ∧ x = 25 / 21 := by
  sorry

end solve_diamond_eq_l243_243841


namespace smallest_four_digit_divisible_by_53_l243_243665

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243665


namespace lines_concur_l243_243513

-- Definition of points
noncomputable def X := 0
noncomputable def Y := 1
noncomputable def Z := 2

-- Definition of triangle vertices using complex exponentials
noncomputable def A := X + complex.exp(complex.I * (real.pi / 3))
noncomputable def B1 := Y + complex.exp(-complex.I * (real.pi / 3))
noncomputable def B := X + complex.exp(-complex.I * (real.pi / 3))
noncomputable def C1 := Y + complex.exp(complex.I * (real.pi / 3))
noncomputable def C := Z + complex.exp(complex.I * (real.pi / 3))
noncomputable def D := Z + complex.exp(-complex.I * (real.pi / 3))

-- Theorem stating the concurrence of lines AC, BD, and XY
theorem lines_concur : 
  ∃ P, ∃ t u v : ℝ, P = (A + t * (C - A)) ∧ P = (B1 + u * (D - B1)) ∧ P = (X + v * (Y - X)) :=
sorry

end lines_concur_l243_243513


namespace smallest_four_digit_divisible_by_53_l243_243690

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243690


namespace expression_value_correct_l243_243207

theorem expression_value_correct (a b : ℤ) (h1 : a = -3) (h2 : b = 2) : -a - b^3 + a * b = -11 := by
  sorry

end expression_value_correct_l243_243207


namespace ceil_sqrt_225_l243_243949

theorem ceil_sqrt_225 : ⌈real.sqrt 225⌉ = 15 :=
by
  have h : real.sqrt 225 = 15 := by
    sorry
  rw [h]
  exact int.ceil_eq_self.mpr rfl

end ceil_sqrt_225_l243_243949


namespace new_person_weight_l243_243806

-- Define the given conditions as Lean definitions
def weight_increase_per_person : ℝ := 2.5
def num_people : ℕ := 8
def replaced_person_weight : ℝ := 65

-- State the theorem using the given conditions and the correct answer
theorem new_person_weight :
  (weight_increase_per_person * num_people) + replaced_person_weight = 85 :=
sorry

end new_person_weight_l243_243806


namespace prime_power_seven_l243_243856

theorem prime_power_seven (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (eqn : p + 25 = q^7) : p = 103 := by
  sorry

end prime_power_seven_l243_243856


namespace papers_left_l243_243466

def initial_paper (total : Nat) : Prop := total = 900
def used_paper (used : Nat) : Prop := used = 156

theorem papers_left (total used left : Nat) (h1 : initial_paper total) (h2 : used_paper used) : left = total - used :=
by
  rw [h1, h2]
  sorry

example : ∃ left : Nat, initial_paper 900 ∧ used_paper 156 ∧ left = 744 :=
by
  exists 744
  split
  rfl
  split
  rfl
  show 744 = 900 - 156
  calc 744 = 900 - 156 : sorry

end papers_left_l243_243466


namespace tangency_coplanar_l243_243251

variable {Point : Type} [MetricSpace Point]

-- Variables for points representing quadrilateral vertices
variables (A B C D : Point)

-- Variables for points of tangency on each side of the quadrilateral
variables (M : Point) (on_AB : Segment A B) 
                  (N : Point) (on_BC : Segment B C) 
                  (P : Point) (on_CD : Segment C D) 
                  (Q : Point) (on_DA : Segment D A)

-- Condition that a sphere is tangent to each side of the space quadrilateral
def sphere_touches_quadrilateral (s : Sphere) : Prop :=
  s.isTangentTo (Line.segment A B on_AB M) ∧ 
  s.isTangentTo (Line.segment B C on_BC N) ∧ 
  s.isTangentTo (Line.segment C D on_CD P) ∧ 
  s.isTangentTo (Line.segment D A on_DA Q)

-- The statement we need to prove: the points M, N, P, Q are coplanar
theorem tangency_coplanar (s : Sphere)
  (hs : sphere_touches_quadrilateral A B C D M on_AB N on_BC P on_CD Q on_DA s) : 
  coplanar {M, N, P, Q} := sorry

end tangency_coplanar_l243_243251


namespace max_value_g_l243_243167

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.sin (2 * x + π / 4)

-- Define the function g(x) as the translated function f(x)
def g (x : ℝ) : ℝ := Real.sin (2 * x - 5 * π / 12)

-- State the theorem to prove the maximum value of g(x) on the given interval
theorem max_value_g : ∀ x ∈ Set.Icc (-π / 8 : ℝ) (3 * π / 8), g x ≤ sqrt 3 / 2 := 
sorry

end max_value_g_l243_243167


namespace ceil_sqrt_225_l243_243948

theorem ceil_sqrt_225 : ⌈real.sqrt 225⌉ = 15 :=
by
  have h : real.sqrt 225 = 15 := by
    sorry
  rw [h]
  exact int.ceil_eq_self.mpr rfl

end ceil_sqrt_225_l243_243948


namespace area_of_EJKG_is_18_cm2_l243_243520

def area_EJKG (EFGH_dims : (ℝ × ℝ)) (J_ratio K_ratio : ℝ) : ℝ :=
  let (x, y) := EFGH_dims
  let area_rectangle := x * y
  let area_triangle_EJF := (1 / 2) * x * (J_ratio * x)
  let area_triangle_GKH := (1 / 2) * y * (K_ratio * y)
  area_rectangle - area_triangle_EJF - area_triangle_GKH

theorem area_of_EJKG_is_18_cm2
  (EFGH_dims : (ℝ × ℝ)) (J_ratio K_ratio : ℝ) (h_dims : EFGH_dims = (12, 6))
  (h_J_ratio : J_ratio = 2/3) (h_K_ratio : K_ratio = 1/3) :
  area_EJKG EFGH_dims J_ratio K_ratio = 18 := by
  rw [h_dims, h_J_ratio, h_K_ratio]
  simp [area_EJKG]
  norm_num
  sorry

end area_of_EJKG_is_18_cm2_l243_243520


namespace smallest_four_digit_divisible_by_53_l243_243696

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243696


namespace least_possible_length_of_third_side_l243_243391

theorem least_possible_length_of_third_side (a b : ℕ) (h1 : a = 8) (h2 : b = 15) : 
  ∃ c : ℕ, c = 17 ∧ a^2 + b^2 = c^2 := 
by
  use 17 
  split
  · rfl
  · rw [h1, h2]
    norm_num

end least_possible_length_of_third_side_l243_243391


namespace smallest_four_digit_divisible_by_53_l243_243662

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243662


namespace smallest_four_digit_divisible_by_53_l243_243685

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l243_243685


namespace not_geometric_sequence_of_transformed_l243_243361

theorem not_geometric_sequence_of_transformed (a b c : ℝ) (q : ℝ) (hq : q ≠ 1) 
  (h_geometric : b = a * q ∧ c = b * q) :
  ¬ (∃ q' : ℝ, 1 - b = (1 - a) * q' ∧ 1 - c = (1 - b) * q') :=
by
  sorry

end not_geometric_sequence_of_transformed_l243_243361


namespace complementary_event_l243_243069

def car_a_selling_well : Prop := sorry
def car_b_selling_poorly : Prop := sorry

def event_A : Prop := car_a_selling_well ∧ car_b_selling_poorly
def event_complement (A : Prop) : Prop := ¬A

theorem complementary_event :
  event_complement event_A = (¬car_a_selling_well ∨ ¬car_b_selling_poorly) :=
by
  sorry

end complementary_event_l243_243069


namespace eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256_l243_243789

theorem eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256 :
  11^2 + 2 * 11 * 5 + 5^2 = 256 := by
  sorry

end eleven_squared_plus_two_times_eleven_times_five_plus_five_squared_eq_256_l243_243789


namespace square_of_1023_l243_243894

def square_1023_eq_1046529 : Prop :=
  let x := 1023
  x * x = 1046529

theorem square_of_1023 : square_1023_eq_1046529 :=
by
  sorry

end square_of_1023_l243_243894


namespace picture_area_l243_243802

-- Given dimensions of the paper
def paper_width : ℝ := 8.5
def paper_length : ℝ := 10

-- Given margins
def margin : ℝ := 1.5

-- Calculated dimensions of the picture
def picture_width := paper_width - 2 * margin
def picture_length := paper_length - 2 * margin

-- Statement to prove
theorem picture_area : picture_width * picture_length = 38.5 := by
  -- skipped the proof
  sorry

end picture_area_l243_243802


namespace slope_parallelogram_cut_l243_243911

theorem slope_parallelogram_cut (a b c d : ℝ):
    (a, b) = (30, 162) ∧ (c, d) = (12, 48) ∧
    ∃ p : ℝ, ∃ q : ℝ, (p, q) = (30, 162 - p) ∧ q = 48 + p ∧ 
    a = (48 + 12) / 12 ∧ b = 1 ∧
    ∀ m n : ℕ,  m / n = 5 / 1 ∧ nat.coprime m n → m + n = 6 :=
by
    sorry

end slope_parallelogram_cut_l243_243911


namespace quadratic_equation_solution_a_plus_b_sq_l243_243186

theorem quadratic_equation_solution_a_plus_b_sq {a b : ℝ} 
  (h_eqn : ∀ x : ℝ, 6 * x ^ 2 + 7 = 5 * x - 11 ↔ 
           x = a + b * complex.I ∨ x = a - b * complex.I) : 
  a + b^2 = 467 / 144 := 
sorry

end quadratic_equation_solution_a_plus_b_sq_l243_243186


namespace probability_no_adjacent_green_hats_l243_243813

-- Step d): Rewrite the math proof problem in a Lean 4 statement.

theorem probability_no_adjacent_green_hats (total_children green_hats : ℕ)
  (hc : total_children = 9) (hg : green_hats = 3) :
  (∃ (p : ℚ), p = 5 / 14) :=
sorry

end probability_no_adjacent_green_hats_l243_243813


namespace find_age_difference_l243_243808

variable (a b c : ℕ)

theorem find_age_difference (h : a + b = b + c + 20) : c = a - 20 :=
by
  sorry

end find_age_difference_l243_243808


namespace smallest_four_digit_divisible_by_53_l243_243697

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243697


namespace alcohol_fraction_l243_243430

theorem alcohol_fraction (ratio : ℝ) (h : ratio = 0.6666666666666666) :
  let alcohol_parts := 2
  let water_parts := 3
  let total_parts := alcohol_parts + water_parts
  let fraction_alcohol := alcohol_parts / total_parts
  fraction_alcohol = 2 / 5 :=
by
  have alcohol_parts : ℝ := 2
  have water_parts : ℝ := 3
  have total_parts : ℝ := alcohol_parts + water_parts
  have fraction_alcohol : ℝ := alcohol_parts / total_parts
  have : ratio = 2 / 3, from sorry
  have : fraction_alcohol = 2 / 5, from sorry
  exact this

end alcohol_fraction_l243_243430


namespace radius_ratio_ge_sqrt2plus1_l243_243487

theorem radius_ratio_ge_sqrt2plus1 (r R a h : ℝ) (h1 : 2 * a ≠ 0) (h2 : h ≠ 0) 
  (hr : r = a * h / (a + Real.sqrt (a ^ 2 + h ^ 2)))
  (hR : R = (2 * a ^ 2 + h ^ 2) / (2 * h)) : 
  R / r ≥ 1 + Real.sqrt 2 := 
sorry

end radius_ratio_ge_sqrt2plus1_l243_243487


namespace coefficient_x7_expansion_l243_243343

theorem coefficient_x7_expansion
  (n : ℕ)
  (h : 2^n = 243) :
  let term_coeff : ℕ := 40
  ∃ (r : ℕ), (n = 5 ∧ 15 - 4 * r = 7 ∧ binomial n r * (2^r) = term_coeff) :=
by
  sorry

end coefficient_x7_expansion_l243_243343


namespace problem_part1_problem_part2_l243_243328

variable (m t : ℝ)
variable p q s : Prop

-- Conditions
def condition_p : Prop := ∃ x : ℝ, 2 * x ^ 2 + (m - 1) * x + 1 / 2 ≤ 0
def condition_q : Prop := (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ^ 2 > b) ∧ (m ^ 2 > 2 * m + 8) ∧ (2 * m + 8 > 0)
def condition_s : Prop := ∃ x y : ℝ, (m - t) * (m - t - 1) < 0 ∧ (-4 < m ∧ m < -2) ∨ (m > 4)

theorem problem_part1 (hpq : condition_p ∧ condition_q) : -4 < m ∧ m < -2 ∨ m > 4 :=
sorry

theorem problem_part2 (hq : condition_q) (hns : q → s → false) : -4 ≤ t ∧ t ≤ -3 ∨ t ≥ 4 :=
sorry

end problem_part1_problem_part2_l243_243328


namespace probability_two_heads_two_tails_four_coins_l243_243377

theorem probability_two_heads_two_tails_four_coins :
  let combinations := Nat.choose 4 2 in
  let probability_sequence := (1 / 2) ^ 4 in
  let favorable_probability := combinations * probability_sequence in
  favorable_probability = 3 / 8 :=
by
  sorry

end probability_two_heads_two_tails_four_coins_l243_243377


namespace general_term_min_sum_Sn_l243_243111

-- (I) Prove the general term formula for the arithmetic sequence
theorem general_term (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = -10) 
  (geometric_cond : (a 2 + 10) * (a 4 + 6) = (a 3 + 8) ^ 2) : 
  ∃ n : ℕ, a n = 2 * n - 12 :=
by
  sorry

-- (II) Prove the minimum value of the sum of the first n terms
theorem min_sum_Sn (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h1 : a 1 = -10)
  (general_term : ∀ n, a n = 2 * n - 12) : 
  ∃ n, S n = n * n - 11 * n ∧ S n = -30 :=
by
  sorry

end general_term_min_sum_Sn_l243_243111


namespace age_of_last_student_l243_243543

-- Definitions for given conditions
def totalAge (n : ℕ) (avg : ℕ) : ℕ := n * avg
def lastStudentAge (T T5 T9 : ℕ) : ℕ := T - T5 - T9

-- Given conditions
def fifteen_students_avg_age : ℕ := 15
def five_students_avg_age : ℕ := 13
def nine_students_avg_age : ℕ := 16
def total_age := totalAge 15 fifteen_students_avg_age
def five_students_total_age := totalAge 5 five_students_avg_age
def nine_students_total_age := totalAge 9 nine_students_avg_age

-- The theorem to prove the last student's age is 16
theorem age_of_last_student : lastStudentAge total_age five_students_total_age nine_students_total_age = 16 :=
  by
    have T := total_age
    have T5 := five_students_total_age
    have T9 := nine_students_total_age
    calc
      lastStudentAge T T5 T9 = T - T5 - T9              : rfl
      ... = 225 - 65 - 144                             : by
        rw [←total_age, ←five_students_total_age, ←nine_students_total_age]
      ... = 16                                         : by norm_num

end age_of_last_student_l243_243543


namespace quadratic_equiv_original_correct_transformation_l243_243132

theorem quadratic_equiv_original :
  (5 + 3*Real.sqrt 2) * x^2 + (3 + Real.sqrt 2) * x - 3 = 
  (7 + 4 * Real.sqrt 3) * x^2 + (2 + Real.sqrt 3) * x - 2 :=
sorry

theorem correct_transformation :
  ∃ r : ℝ, r = (9 / 7) - (4 * Real.sqrt 2 / 7) ∧ 
  ((5 + 3 * Real.sqrt 2) * x^2 + (3 + Real.sqrt 2) * x - 3) = 0 :=
sorry

end quadratic_equiv_original_correct_transformation_l243_243132


namespace necessary_but_not_sufficient_l243_243547

theorem necessary_but_not_sufficient (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 > 0) ↔ -2 < m ∧ m < 2 → m < 2 :=
by
  sorry

end necessary_but_not_sufficient_l243_243547


namespace chocolates_remaining_on_fifth_day_l243_243364

noncomputable def remaining_chocolates : ℕ := 24
def first_day_chocolates : ℕ := 4
def second_day_chocolates : ℕ := (2 * first_day_chocolates - 3)
def third_day_chocolates : ℕ := (first_day_chocolates - 2)
def fourth_day_chocolates : ℕ := (third_day_chocolates - 1)
def total_eaten_chocolates : ℕ := first_day_chocolates + second_day_chocolates + third_day_chocolates + fourth_day_chocolates

theorem chocolates_remaining_on_fifth_day :
  remaining_chocolates - total_eaten_chocolates = 12 :=
by
  unfold remaining_chocolates
  unfold first_day_chocolates
  unfold second_day_chocolates
  unfold third_day_chocolates
  unfold fourth_day_chocolates
  unfold total_eaten_chocolates
  sorry

end chocolates_remaining_on_fifth_day_l243_243364


namespace probability_no_adjacent_green_hats_l243_243814

-- Step d): Rewrite the math proof problem in a Lean 4 statement.

theorem probability_no_adjacent_green_hats (total_children green_hats : ℕ)
  (hc : total_children = 9) (hg : green_hats = 3) :
  (∃ (p : ℚ), p = 5 / 14) :=
sorry

end probability_no_adjacent_green_hats_l243_243814


namespace marissas_sunflower_height_l243_243499

def height_of_marissas_sunflower (sister_height_feet : ℤ) (sister_height_inches : ℤ) (additional_inches : ℤ) : Prop :=
  (sister_height_feet = 4) →
  (sister_height_inches = 3) →
  (additional_inches = 21) →
  sister_height_feet * 12 + sister_height_inches + additional_inches = 72

-- Prove that Marissa's sunflower height in feet is 6
theorem marissas_sunflower_height :
  height_of_marissas_sunflower 4 3 21 →
  72 / 12 = 6 :=
by
  assume h,
  rw Nat.div_eq_of_eq_mul_left sorry,
  sorry

end marissas_sunflower_height_l243_243499


namespace median_length_range_l243_243418

/-- Define the structure of the triangle -/
structure Triangle :=
  (A B C : ℝ) -- vertices of the triangle
  (AD AE AF : ℝ) -- lengths of altitude, angle bisector, and median
  (angleA : AngleType) -- type of angle A (acute, orthogonal, obtuse)

-- Define the angle type as a custom type
inductive AngleType
| acute
| orthogonal
| obtuse

def m_range (t : Triangle) : Set ℝ :=
  match t.angleA with
  | AngleType.acute => {m : ℝ | 13 < m ∧ m < (2028 / 119)}
  | AngleType.orthogonal => {m : ℝ | m = (2028 / 119)}
  | AngleType.obtuse => {m : ℝ | (2028 / 119) < m}

-- Lean statement for proving the problem
theorem median_length_range (t : Triangle)
  (hAD : t.AD = 12)
  (hAE : t.AE = 13) : t.AF ∈ m_range t :=
by
  sorry

end median_length_range_l243_243418


namespace smallest_four_digit_divisible_by_53_l243_243643

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (53 ∣ n) ∧ n = 1007 :=
by {
  -- We state the existence of n
  use 1007,
  -- Two conditions: 1000 ≤ n < 10000
  have h₁ : 1000 ≤ 1007 := by norm_num,
  have h₂ : 1007 < 10000 := by norm_num,
  -- n is divisible by 53
  have h₃ : 53 ∣ 1007 := by norm_num,
  -- Proving the equality
  exact ⟨h₁, h₂, h₃⟩,
}

end smallest_four_digit_divisible_by_53_l243_243643


namespace cos_B_equals_half_sin_A_mul_sin_C_equals_three_fourths_l243_243419

-- Definitions for angles A, B, and C forming an arithmetic sequence and their sum being 180 degrees
variables {A B C : ℝ}

-- Definitions for side lengths a, b, and c forming a geometric sequence
variables {a b c : ℝ}

-- Question 1: Prove that cos B = 1/2 under the given conditions
theorem cos_B_equals_half 
  (h1 : 2 * B = A + C) 
  (h2 : A + B + C = 180) : 
  Real.cos B = 1 / 2 :=
sorry

-- Question 2: Prove that sin A * sin C = 3/4 under the given conditions
theorem sin_A_mul_sin_C_equals_three_fourths 
  (h1 : 2 * B = A + C) 
  (h2 : A + B + C = 180) 
  (h3 : b^2 = a * c) : 
  Real.sin A * Real.sin C = 3 / 4 :=
sorry

end cos_B_equals_half_sin_A_mul_sin_C_equals_three_fourths_l243_243419


namespace square_of_1023_l243_243895

def square_1023_eq_1046529 : Prop :=
  let x := 1023
  x * x = 1046529

theorem square_of_1023 : square_1023_eq_1046529 :=
by
  sorry

end square_of_1023_l243_243895


namespace triangle_properties_l243_243334

theorem triangle_properties
  (a b c : ℝ) (A B C : ℝ)
  (h1 : c = sqrt 7 * a)
  (h2 : b = 2 * sqrt 3)
  (h3 : sqrt 3 * c * sin A = a * cos C)
  (h4 : C = π / 6) :
  let area := 1 / 2 * a * b * sin C in
  area = sqrt 3 / 2 :=
by {
  sorry
}

end triangle_properties_l243_243334


namespace find_k_find_m_l243_243360

-- Condition definitions
def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (2, 3)

-- Proof problem statements
theorem find_k (k : ℝ) :
  (3 * a.fst - b.fst) / (a.fst + k * b.fst) = (3 * a.snd - b.snd) / (a.snd + k * b.snd) →
  k = -1 / 3 :=
sorry

theorem find_m (m : ℝ) :
  a.fst * (m * a.fst - b.fst) + a.snd * (m * a.snd - b.snd) = 0 →
  m = -4 / 5 :=
sorry

end find_k_find_m_l243_243360


namespace cricket_player_avg_runs_l243_243847

theorem cricket_player_avg_runs (A : ℝ) :
  (13 * A + 92 = 14 * (A + 5)) → A = 22 :=
by
  intro h1
  have h2 : 13 * A + 92 = 14 * A + 70 := by sorry
  have h3 : 92 - 70 = 14 * A - 13 * A := by sorry
  sorry

end cricket_player_avg_runs_l243_243847


namespace find_ratio_BC_CA_AB_l243_243312

-- Define the conditions and the statement to be proved in Lean 4
variable {A B C M K L : Type}

-- Assume the given triangle conditions and definitions
axiom triangle_ABC : Triangle ABC
axiom median_AM : isMedian AM ABC
axiom incircle_intersections : IncircleIntersectsGammaAtPoints AM ABC Γ K L
axiom segment_ratios : SegmentsEqualRatios AK KL LM 1 2 3

-- Define the theorem to find the ratio BC : CA : AB
theorem find_ratio_BC_CA_AB : ratio BC CA AB = 6 / (4 - sqrt 5) / (4 + sqrt 5) := sorry

end find_ratio_BC_CA_AB_l243_243312


namespace smallest_four_digit_divisible_by_53_l243_243725

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l243_243725


namespace find_Z_l243_243015

open Complex

-- Definitions
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem find_Z (Z : ℂ) (h1 : abs Z = 3) (h2 : is_pure_imaginary (Z + (3 * Complex.I))) : Z = 3 * Complex.I :=
by
  sorry

end find_Z_l243_243015


namespace eccentricity_of_hyperbola_l243_243352

-- Conditions
variables (a b c e : ℝ) (A B : ℝ × ℝ)
variable (h_cond1 : a > 0)
variable (h_cond2 : b > 0)
variable (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
variable (h_parabola : ∀ y, ∃ x, y^2 = 4 * c * x)
variable (h_c : c = Real.sqrt (a^2 + b^2))
variable (h_intersection : A = (c, 2 * c) ∧ B = (-c, -2 * c))
variable (h_AB : Real.dist A B = 4 * c)

-- Question (to prove)
theorem eccentricity_of_hyperbola (h_e : e = c / a) : e = Real.sqrt 2 + 1 :=
sorry

end eccentricity_of_hyperbola_l243_243352


namespace jovana_added_pounds_l243_243099

noncomputable def initial_amount : ℕ := 5
noncomputable def final_amount : ℕ := 28

theorem jovana_added_pounds : final_amount - initial_amount = 23 := by
  sorry

end jovana_added_pounds_l243_243099


namespace least_common_multiple_greater_than_2n_l243_243317

variable (n : ℕ)
variable (a : Fin n → ℕ)

noncomputable def least_common_multiple (x y : ℕ) : ℕ := x * y / Nat.gcd x y

theorem least_common_multiple_greater_than_2n
  (h1 : ∀ i, 1 ≤ a i ∧ a i ≤ 2 * n)
  (h2 : ∀ i j, i ≠ j → least_common_multiple (a i) (a j) > 2 * n) :
  a 0 > (2 * n) / 3 :=
sorry

end least_common_multiple_greater_than_2n_l243_243317


namespace smallest_four_digit_divisible_by_53_l243_243744

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243744


namespace equal_areas_of_DEF_and_PQR_l243_243092
noncomputable theory

open_locale classical

variables {A B C D E F P Q R : Type*}
  [triangle ABC : Type*]
  [angle_bisector AD : Type*]
  [angle_bisector BE : Type*]
  [angle_bisector CF : Type*]
  [perpendicular_to AD at_midpoint : Type*]
  [intersects_AC_at P : Type*]
  [perpendicular_to BE at_midpoint : Type*]
  [intersects_AB_at Q : Type*]
  [perpendicular_to CF at_midpoint : Type*]
  [intersects_CB_at R : Type*]

def triangles_equal_area (DEF PQR : Type*) :=
  triangle.area DEF = triangle.area PQR

theorem equal_areas_of_DEF_and_PQR :
  ∀ {ABC : Type*}
    {AD BE CF : Type*}
    {P Q R : Type*},
    triangle ABC →
    angle_bisector AD →
    angle_bisector BE →
    angle_bisector CF →
    perpendicular_to AD at_midpoint →
    intersects_AC_at P →
    perpendicular_to BE at_midpoint →
    intersects_AB_at Q →
    perpendicular_to CF at_midpoint →
    intersects_CB_at R →
    triangles_equal_area DEF PQR :=
by sorry

end equal_areas_of_DEF_and_PQR_l243_243092


namespace smallest_four_digit_divisible_by_53_l243_243742

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243742


namespace evaluate_f_at_points_l243_243909

def f (x : ℝ) : ℝ :=
  3 * x ^ 2 - 6 * x + 10

theorem evaluate_f_at_points : 3 * f 2 + 2 * f (-2) = 98 :=
by
  sorry

end evaluate_f_at_points_l243_243909


namespace log_problem_l243_243052

noncomputable def log_prob (x : ℝ) : Prop :=
  log (216) x = (1 / 3) * (log 5 / log 6)

theorem log_problem (x : ℝ) (h : log 16 (x - 3) = 1 / 4) : log_prob x :=
sorry

end log_problem_l243_243052


namespace geometric_sequence_solution_l243_243459

noncomputable def geometric_sum_sq (n : ℕ) (a : ℕ → ℕ) : ℕ :=
  ∑ i in finset.range n, (a (i + 1))^2

theorem geometric_sequence_solution (n : ℕ) (a : ℕ → ℕ)
  (h₁ : ∑ i in finset.range n, a (i + 1) = 2^n - 1)
  (h₂ : ∃ r a₁ : ℕ, ∀ k : ℕ, a (k + 1) = r^k * a₁) :
  geometric_sum_sq n a = (4^n - 1) / 3 := 
sorry

end geometric_sequence_solution_l243_243459


namespace sarah_class_choices_l243_243444

-- Conditions 
def total_classes : ℕ := 10
def choose_classes : ℕ := 4
def specific_classes : ℕ := 2

-- Statement
theorem sarah_class_choices : 
  ∃ (n : ℕ), n = Nat.choose (total_classes - specific_classes) 3 ∧ n = 56 :=
by 
  sorry

end sarah_class_choices_l243_243444


namespace smallest_four_digit_multiple_of_53_l243_243707

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243707


namespace smallest_four_digit_divisible_by_53_l243_243719

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end smallest_four_digit_divisible_by_53_l243_243719


namespace cos_shifted_theta_eq_sin_double_alpha_shifted_eq_l243_243022

theorem cos_shifted_theta_eq : 
  let θ : ℝ := Real.arctan (-1 / 2)
  in cos (π/2 + θ) = sqrt 5 / 5 := 
by 
  let θ : ℝ := Real.arctan (-1 / 2)
  sorry

theorem sin_double_alpha_shifted_eq (α : ℝ) :
  let θ : ℝ := Real.arctan (-1 / 2)
  in cos (α + π / 4) = sin θ → sin (2 * α + π / 4) = (7 * sqrt 2 / 10) ∨ (sin (2 * α + π / 4) = - sqrt 2 / 10) :=
by 
  let θ : ℝ := Real.arctan (-1 / 2)
  sorry

end cos_shifted_theta_eq_sin_double_alpha_shifted_eq_l243_243022


namespace number_of_true_propositions_l243_243329

-- Define propositions p and q
def p (a : ℝ) : Prop := 
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 2*a*x1 - 1 = 0) ∧ (x2^2 - 2*a*x2 - 1 = 0)

def q : Prop := 
  ∀ x : ℝ, x + 4 / x ≠ 4

-- Define the given propositions
def prop1 : Prop := p ∧ q
def prop2 : Prop := p ∨ q
def prop3 : Prop := p ∧ ¬ q
def prop4 : Prop := ¬ p ∨ ¬ q

-- Define the proof problem
def proof_problem : Prop :=
  (if prop1 then 1 else 0) + 
  (if prop2 then 1 else 0) + 
  (if prop3 then 1 else 0) + 
  (if prop4 then 1 else 0) = 3

-- The Lean statement
theorem number_of_true_propositions : proof_problem := 
by sorry

end number_of_true_propositions_l243_243329


namespace inscribed_square_relationship_l243_243880

variable (a b c : ℝ)
variable (ha hb hc : ℝ)
variable (S : ℝ)

-- Conditions
axiom triangle_inequality : a > b ∧ b > c
axiom heights_inequality : ha < hb ∧ hb < hc
axiom area_relation_a : S = 1 / 2 * a * ha
axiom area_relation_b : S = 1 / 2 * b * hb
axiom area_relation_c : S = 1 / 2 * c * hc

-- Define side lengths of inscribed squares
def x_a : ℝ := 2 * S / (a + ha)
def x_b : ℝ := 2 * S / (b + hb)
def x_c : ℝ := 2 * S / (c + hc)

-- Problem statement to prove
theorem inscribed_square_relationship :
  x_a < x_b ∧ x_b < x_c := by sorry

end inscribed_square_relationship_l243_243880


namespace necessary_but_not_sufficient_for_odd_function_l243_243127

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ :=
  Real.cos (2 * x + ϕ)

def is_odd (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = -g (x)

def phi_condition (ϕ : ℝ) : Prop :=
  ϕ = π / 2

theorem necessary_but_not_sufficient_for_odd_function :
  (is_odd (f · (π / 2))) ↔ phi_condition (π / 2) :=
sorry

end necessary_but_not_sufficient_for_odd_function_l243_243127


namespace converse_proposition_l243_243203

-- Define a proposition for vertical angles
def vertical_angles (α β : ℕ) : Prop := α = β

-- Define the converse of the vertical angle proposition
def converse_vertical_angles (α β : ℕ) : Prop := β = α

-- Prove that the converse of "Vertical angles are equal" is 
-- "Angles that are equal are vertical angles"
theorem converse_proposition (α β : ℕ) : vertical_angles α β ↔ converse_vertical_angles α β :=
by
  sorry

end converse_proposition_l243_243203


namespace perpendicular_planes_l243_243011

-- Define planes and line
variables (α β γ : Plane) (l : Line)

-- Conditions
variables (h1 : l ⟂ α) (h2 : l ∥ β)

-- Theorem statement
theorem perpendicular_planes (α β γ : Plane) (l : Line) 
  (h1 : l ⟂ α) (h2 : l ∥ β) : α ⟂ β :=
sorry

end perpendicular_planes_l243_243011


namespace molecular_weight_of_one_mole_l243_243605

noncomputable def molecular_weight (total_weight : ℝ) (moles : ℕ) : ℝ :=
total_weight / moles

theorem molecular_weight_of_one_mole (h : molecular_weight 252 6 = 42) : molecular_weight 252 6 = 42 := by
  exact h

end molecular_weight_of_one_mole_l243_243605


namespace positive_integer_n_conditions_l243_243294

theorem positive_integer_n_conditions (n : ℕ) :
  (∃ k : ℕ, k ≥ 2 ∧ ∃ (a : Fin k → ℚ), (∀ i, a i > 0) ∧ (∑ i, a i = n) ∧ (∏ i, a i = n)) ↔ n = 4 ∨ n ≥ 6 :=
by
  sorry

end positive_integer_n_conditions_l243_243294


namespace sum_of_final_numbers_l243_243569

variable {x y T : ℝ}

theorem sum_of_final_numbers (h : x + y = T) : 3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 :=
by 
  -- The place for the proof steps, which will later be filled
  sorry

end sum_of_final_numbers_l243_243569


namespace least_third_side_of_right_triangle_l243_243383

theorem least_third_side_of_right_triangle (a b : ℕ) (ha : a = 8) (hb : b = 15) :
  ∃ c : ℝ, c = real.sqrt (b^2 - a^2) ∧ c = real.sqrt 161 :=
by {
  -- We state the conditions
  have h8 : (8 : ℝ) = a, from by {rw ha},
  have h15 : (15 : ℝ) = b, from by {rw hb},

  -- The theorem states that such a c exists
  use (real.sqrt (15^2 - 8^2)),

  -- We need to show the properties of c
  split,
  { 
    -- Showing that c is the sqrt of the difference of squares of b and a
    rw [←h15, ←h8],
    refl 
  },
  {
    -- Showing that c is sqrt(161)
    calc
       real.sqrt (15^2 - 8^2)
         = real.sqrt (225 - 64) : by norm_num
     ... = real.sqrt 161 : by norm_num
  }
}
sorry

end least_third_side_of_right_triangle_l243_243383


namespace probability_two_fives_two_fair_dice_l243_243793

theorem probability_two_fives_two_fair_dice
  (faces_per_die : ℕ) (fair_dice : true)
  (total_outcomes : ℕ := faces_per_die * faces_per_die)
  (successful_outcomes : ℕ := 1) :
  faces_per_die = 6 → fair_dice → (successful_outcomes / total_outcomes : ℚ) = 1/36 := by
  intros faces_eq fair_dice
  rw [faces_eq]
  have h_outcomes: total_outcomes = 36 := by simp [total_outcomes]
  rw [h_outcomes]
  have h_success: successful_outcomes = 1 := rfl
  rw [h_success]
  norm_num
  sorry

end probability_two_fives_two_fair_dice_l243_243793


namespace find_tangent_points_l243_243160

noncomputable def f : ℝ → ℝ := λ x, x^3 + x - 2

def tangent_parallel_to_line_at_p (p : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), p = (a, b) ∧ f a = b ∧ (3 * a ^ 2 + 1 = 4)

theorem find_tangent_points :
  ∃ (p1 p2 : ℝ × ℝ),
    tangent_parallel_to_line_at_p p1 ∧ tangent_parallel_to_line_at_p p2 ∧
    ((p1 = (1, 0) ∧ p2 = (-1, -4)) ∨ (p1 = (-1, -4) ∧ p2 = (1, 0))) :=
by sorry

end find_tangent_points_l243_243160


namespace least_side_is_8_l243_243393

-- Define the sides of the right triangle
variables (a b : ℝ) (h : a = 8) (k : b = 15)

-- Define a predicate for the least possible length of the third side
def least_possible_third_side (c : ℝ) : Prop :=
  (c = 8) ∨ (c = 15) ∨ (c = 17)

theorem least_side_is_8 (c : ℝ) (hc : least_possible_third_side c) : c = 8 :=
by
  sorry

end least_side_is_8_l243_243393


namespace solve_for_x_l243_243833

theorem solve_for_x : ∃ (x : ℤ), -5 - x = 1 ∧ x = -6 := by {
  use -6,
  split,
  { 
    calc
      -5 - (-6) = -5 + 6 : by rw sub_neg_eq_add
      ... = 1 : by norm_num,
  },
  { 
    reflexivity
  }
}

end solve_for_x_l243_243833


namespace parallel_line_through_point_l243_243295

theorem parallel_line_through_point :
  ∀ {x y : ℝ}, (3 * x + 4 * y + 1 = 0) ∧ (∃ (a b : ℝ), a = 1 ∧ b = 2 ∧ (3 * a + 4 * b + x0 = 0) → (x = -11)) :=
sorry

end parallel_line_through_point_l243_243295


namespace angle_BEC_90_deg_l243_243463

-- Define the trapezoid and the relevant points and conditions.
structure Trapezoid :=
( A B C D : Point )
( AB_parallel_CD : A.y = B.y ∧ C.y = D.y ∧ A.y ≠ C.y )
( AD_eq_BD : dist A D = dist B D )

noncomputable def circumcircle (A B C : Point) : Circle := sorry

noncomputable def Point in_minor_arc (A B C D E : Point) : Prop :=
sorry

noncomputable def dist (A B : Point) : ℝ := sorry

theorem angle_BEC_90_deg :
  ∀ (A B C D E : Point)
  (trapezoid : Trapezoid A B C D)
  (circumcircle_ACD : circumcircle A C D)
  (E_in_minor_arc_CD : in_minor_arc A C D E)
  (AD_eq_DE : dist A D = dist D E),
  ∠ BEC = 90 :=
by
  sorry

end angle_BEC_90_deg_l243_243463


namespace least_possible_length_of_third_side_l243_243388

theorem least_possible_length_of_third_side (a b : ℕ) (h1 : a = 8) (h2 : b = 15) : 
  ∃ c : ℕ, c = 17 ∧ a^2 + b^2 = c^2 := 
by
  use 17 
  split
  · rfl
  · rw [h1, h2]
    norm_num

end least_possible_length_of_third_side_l243_243388


namespace simplify_sqrt_l243_243791

theorem simplify_sqrt (a : ℝ) (h : a < 2) : Real.sqrt ((a - 2)^2) = 2 - a :=
by
  sorry

end simplify_sqrt_l243_243791


namespace smallest_four_digit_div_by_53_l243_243621

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l243_243621


namespace line_passes_through_vertex_unique_a_l243_243307

theorem line_passes_through_vertex_unique_a:
  ∀ (a : ℝ), (2 * (-a / 2) + a = (-(a ^ 2) / 4) + a * (-a / 2) + a^2) ↔ a = 0 := 
by
  intro a
  split
  { intro h
    sorry }
  { intro h
    rw [h, zero_mul, zero_add]
    sorry }

end line_passes_through_vertex_unique_a_l243_243307


namespace smallest_four_digit_divisible_by_53_l243_243653

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m := by
  have exists_divisible : ∃ k : ℕ, 53 * k = 1007 := by
    use 19
    norm_num
  exact exists_divisible.sorry -- Sorry placeholder for the analytical proof part

end smallest_four_digit_divisible_by_53_l243_243653


namespace smallest_four_digit_divisible_by_53_l243_243749

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243749


namespace ceil_sqrt_225_l243_243934

theorem ceil_sqrt_225 : Nat.ceil (Real.sqrt 225) = 15 :=
by
  sorry

end ceil_sqrt_225_l243_243934


namespace probability_no_adjacent_green_hats_l243_243821

-- Definitions
def total_children : ℕ := 9
def green_hats : ℕ := 3

-- Main theorem statement
theorem probability_no_adjacent_green_hats : 
  (9.choose 3) = 84 → 
  (1 - (9 + 45) / 84) = 5/14 := 
sorry

end probability_no_adjacent_green_hats_l243_243821


namespace Ariel_current_age_l243_243877

-- Define the conditions
def Ariel_birth_year : Nat := 1992
def Ariel_start_fencing_year : Nat := 2006
def Ariel_fencing_years : Nat := 16

-- Define the problem as a theorem
theorem Ariel_current_age :
  (Ariel_start_fencing_year - Ariel_birth_year) + Ariel_fencing_years = 30 := by
sorry

end Ariel_current_age_l243_243877


namespace smallest_four_digit_divisible_by_53_l243_243668

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243668


namespace convex_polygon_center_symmetry_l243_243517

theorem convex_polygon_center_symmetry
  (P : Type) [polygon P] [convex P]
  (divided_into_parallelograms : ∃ Q : Type, (parallelogram Q) ∧ (divides P Q)) :
  has_center_of_symmetry P :=
sorry

end convex_polygon_center_symmetry_l243_243517


namespace smallest_four_digit_multiple_of_53_l243_243712

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243712


namespace smallest_four_digit_divisible_by_53_l243_243691

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243691


namespace number_pairs_square_l243_243976

theorem number_pairs_square (n : ℕ) (h: n > 1) :
  ∃ (f : ℕ → ℕ → ℕ), (∀ (i j : ℕ), i ≠ j → f i j ∈ {1, 2, ..., 2 * n} ∧ (f i j).pairwise_disjoint) → 
  ∏ (i : ℕ) in finset.range n, (∑ (j : ℕ) in finset.range n, f i j) ^ 2 := 
sorry

end number_pairs_square_l243_243976


namespace least_possible_length_of_third_side_l243_243390

theorem least_possible_length_of_third_side (a b : ℕ) (h1 : a = 8) (h2 : b = 15) : 
  ∃ c : ℕ, c = 17 ∧ a^2 + b^2 = c^2 := 
by
  use 17 
  split
  · rfl
  · rw [h1, h2]
    norm_num

end least_possible_length_of_third_side_l243_243390


namespace circles_intersect_l243_243325

-- Context and given conditions
def C1 : Prop := ∃ (x y : ℝ), x^2 + y^2 + 2*x + 3*y + 1 = 0
def C2 : Prop := ∃ (x y : ℝ), x^2 + y^2 + 4*x + 3*y + 2 = 0
def d := 1
def R := 3 / 2
def r := Real.sqrt 17 / 2

-- Prove the positional relationship is intersecting
theorem circles_intersect (hC1 : C1) (hC2 : C2) : 
  (r + R > d) ∧ (d > r - R) :=
by 
  sorry

end circles_intersect_l243_243325


namespace smallest_four_digit_div_by_53_l243_243629

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l243_243629


namespace ceil_sqrt_225_eq_15_l243_243959

theorem ceil_sqrt_225_eq_15 : ⌈ Real.sqrt 225 ⌉ = 15 :=
by
  sorry

end ceil_sqrt_225_eq_15_l243_243959


namespace smallest_enclosing_sphere_radius_l243_243925

-- Define the conditions
def sphere_radius : ℝ := 2

-- Define the sphere center coordinates in each octant
def sphere_centers : List (ℝ × ℝ × ℝ) :=
  [ (2, 2, 2), (2, 2, -2), (2, -2, 2), (2, -2, -2),
    (-2, 2, 2), (-2, 2, -2), (-2, -2, 2), (-2, -2, -2) ]

-- Define the theorem statement
theorem smallest_enclosing_sphere_radius :
  (∃ (r : ℝ), r = 2 * Real.sqrt 3 + 2) :=
by
  -- conditions and proof will go here
  sorry

end smallest_enclosing_sphere_radius_l243_243925


namespace little_john_gave_to_each_friend_l243_243491

noncomputable def little_john_total : ℝ := 10.50
noncomputable def sweets : ℝ := 2.25
noncomputable def remaining : ℝ := 3.85

theorem little_john_gave_to_each_friend :
  (little_john_total - sweets - remaining) / 2 = 2.20 :=
by
  sorry

end little_john_gave_to_each_friend_l243_243491


namespace smallest_four_digit_divisible_by_53_l243_243617

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l243_243617


namespace find_varphi_l243_243349

theorem find_varphi (ω : ℝ) (varphi : ℝ) (k : ℤ) (l : ℤ) (x : ℝ) :
  (ω > 0) →
  (0 < varphi ∧ varphi < π / 2) →
  (sin varphi = -sin (ω * (π / 2) + varphi)) →
  (∃ varphi, ∀ x, (sin (ω * x + varphi) = sin (ω * x + ω * (π / 12) + varphi))) ∧
  (∃ varphi, ∀ x, (sin (ω * x + ω * (π / 12) + varphi) = (sin (ω * (-x) + ω * (π / 12) + varphi)))) →
  varphi = π / 6 :=
by
  sorry

end find_varphi_l243_243349


namespace smallest_four_digit_multiple_of_53_l243_243787

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l243_243787


namespace problem1_problem2_l243_243227

open Function

section Problem1

-- Define f such that for all x ≠ 2, f(2/x + 2) = x + 1 implies f(x) = x / (x - 2)
theorem problem1 (f : ℝ → ℝ) :
  (∀ (x : ℝ), x ≠ 2 → f (2 / x + 2) = x + 1) →
  (∀ (x : ℝ), x ≠ 2 → f x = x / (x - 2)) :=
by
  sorry

end Problem1

section Problem2

-- Define f such that if f is linear and 3f(x + 1) - 2f(x - 1) = 2x + 17, then f(x) = 2x + 7
theorem problem2 (f : ℝ → ℝ) :
  (Linear f) →
  (∀ (x : ℝ), 3 * f (x + 1) - 2 * f (x - 1) = 2 * x + 17) →
  (∀ (x : ℝ), f x = 2 * x + 7) :=
by
  sorry

end Problem2

end problem1_problem2_l243_243227


namespace normal_vector_of_plane_l243_243359

-- Vectors AB and AC definitions
def vectorAB : ℝ × ℝ × ℝ := (0, 2, 1)
def vectorAC : ℝ × ℝ × ℝ := (-1, 1, -2)
def normal_vector : ℝ × ℝ × ℝ := (5, 1, -2)

-- Dot product definition
def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Proof that the normal vector is indeed a normal vector of the plane ABC
theorem normal_vector_of_plane :
  dot_product vectorAB normal_vector = 0 ∧ dot_product vectorAC normal_vector = 0 :=
by
  sorry

end normal_vector_of_plane_l243_243359


namespace probability_of_abs_x_leq_1_in_interval_neg2_to_4_l243_243141

theorem probability_of_abs_x_leq_1_in_interval_neg2_to_4 :
  let a := -2
  let b := 4
  let favorable_start := -1
  let favorable_end := 1
  let total_length := b - a
  let favorable_length := favorable_end - favorable_start
  probability (favorable_length / total_length) = 1 / 3 :=
by
  -- proof steps in Lean would follow here
  sorry

end probability_of_abs_x_leq_1_in_interval_neg2_to_4_l243_243141


namespace no_adjacent_green_hats_l243_243829

theorem no_adjacent_green_hats (n m : ℕ) (h₀ : n = 9) (h₁ : m = 3) : 
  (((1 : ℚ) - (9/14 : ℚ)) = (5/14 : ℚ)) :=
by
  rw h₀ at *,
  rw h₁ at *,
  sorry

end no_adjacent_green_hats_l243_243829


namespace ceil_sqrt_225_eq_15_l243_243929

theorem ceil_sqrt_225_eq_15 : Real.ceil (Real.sqrt 225) = 15 := 
by 
  sorry

end ceil_sqrt_225_eq_15_l243_243929


namespace polynomial_roots_l243_243278

theorem polynomial_roots :
  (∃ x : ℝ, x^4 - 16*x^3 + 91*x^2 - 216*x + 180 = 0) ↔ (x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 6) := 
sorry

end polynomial_roots_l243_243278


namespace gcf_lcm_example_l243_243888

def lcm (a b : ℕ) : ℕ := sorry  -- Suppose we have a definition of LCM.
def gcf (a b : ℕ) : ℕ := sorry  -- Suppose we have a definition of GCF.

theorem gcf_lcm_example : gcf (lcm 9 15) (lcm 14 25) = 5 := 
by
  -- without proof here due to instructions
  sorry

end gcf_lcm_example_l243_243888


namespace hyperbola_eccentricity_l243_243351

theorem hyperbola_eccentricity 
    (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
    (h_asymptote_tangent : ∀ {x y : ℝ}, (x^2 + (y - 2)^2 = 1) → (dist (0, 2) (a * x + b * y = 0) = 1)) :
    let c := 2 * b in
    let e := c / a in
    e = 2 * (3:ℝ).sqrt / 3 := 
by
  -- Proof goes here
  sorry

end hyperbola_eccentricity_l243_243351


namespace exist_pair_with_small_difference_l243_243529

theorem exist_pair_with_small_difference (x y z : ℝ) (hx : 0 ≤ x ∧ x < 1) (hy : 0 ≤ y ∧ y < 1) (hz : 0 ≤ z ∧ z < 1) :
  ∃ a b ∈ ({x, y, z} : set ℝ), |b - a| < 1 / 2 :=
by sorry

end exist_pair_with_small_difference_l243_243529


namespace find_positive_integral_solution_l243_243975

theorem find_positive_integral_solution :
  ∃ n : ℕ, 0 < n ∧ (∑ k in finset.range(n), (2 * k + 1)) / ∑ k in finset.range(n), (2 * (k + 1)) = (124 : ℚ) / 125 :=
by
  sorry

end find_positive_integral_solution_l243_243975


namespace exists_triangle_with_area_le_one_fourth_l243_243080

open Classical

theorem exists_triangle_with_area_le_one_fourth 
  (points : Set (EuclideanSpace ℝ (Fin 2))) 
  (h_card : points.card = 5) 
  (h_unit_area : ∀ p ∈ points, ∃ T : Set (EuclideanSpace ℝ (Fin 2)), T.card = 3 ∧ is_unit_area_triangle T) : 
  ∃ (T : Set (EuclideanSpace ℝ (Fin 2))), T ⊆ points ∧ T.card = 3 ∧ triangle_area T ≤ 1 / 4 :=
sorry

end exists_triangle_with_area_le_one_fourth_l243_243080


namespace f_is_odd_g_is_odd_F_max_min_sum_zero_F_solution_set_l243_243327

noncomputable def f (x : ℝ) := (1 - 2^x) / (1 + 2^x)
noncomputable def g (x : ℝ) := log (sqrt (x^2 + 1) - x)
noncomputable def F (x : ℝ) := f x + g x

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := sorry

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := sorry

theorem F_max_min_sum_zero : (∀ x : ℝ, x ∈ Icc (-1 : ℝ) 1 → F (x) ≤ F 0) ∧ 
                             (∀ x : ℝ, x ∈ Icc (-1 : ℝ) 1 → F (x) ≥ F (-1)) → 
                             F 1 + F (-1) = 0 := sorry

theorem F_solution_set (a : ℝ) : (F (2 * a) + F (-1 - a) < 0) → (1 < a) := sorry

end f_is_odd_g_is_odd_F_max_min_sum_zero_F_solution_set_l243_243327


namespace sequence_bounded_l243_243225

open Set

theorem sequence_bounded (a : ℕ → ℕ) (h1 : ∀ j, 1 ≤ a j ∧ a j ≤ 2015)
    (h2 : ∀ k l, 1 ≤ k → k < l → k + a k ≠ l + a l) :
    ∃ b N : ℕ, (N > 0) ∧ (∀ m n : ℕ, n > m ∧ N ≤ m → 
        abs (∑ j in (range (n + 1)).filter (λ j, m < j), (a j - b)) ≤ 1007^2) := 
by
  sorry

end sequence_bounded_l243_243225


namespace greatest_digit_sum_base6_l243_243602

theorem greatest_digit_sum_base6 (n : ℕ) (h : n < 2401) : 
  ∃ s, s ≤ 12 ∧ ∀ t, n < 2401 → sum_digits_base6 n t ≤ sum_digits_base6 n s :=
sorry

end greatest_digit_sum_base6_l243_243602


namespace smallest_four_digit_divisible_by_53_l243_243755

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007 :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243755


namespace ceil_sqrt_225_l243_243939

theorem ceil_sqrt_225 : Nat.ceil (Real.sqrt 225) = 15 :=
by
  sorry

end ceil_sqrt_225_l243_243939


namespace smallest_four_digit_divisible_by_53_l243_243683

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n ≤ 9999 ∧ (∃ k : ℤ, n = 53 * k) ∧ n = 1007 :=
by {
  existsi 1007,
  split,
  exact dec_trivial,  -- justification that 1000 ≤ 1007
  split,
  exact dec_trivial,  -- justification that 1007 ≤ 9999
  split,
  existsi 19,
  exact dec_trivial,  -- calculation such that 1007 = 53 * 19
  exact dec_trivial   -- n = 1007
}

end smallest_four_digit_divisible_by_53_l243_243683


namespace no_adjacent_green_hats_l243_243828

theorem no_adjacent_green_hats (n m : ℕ) (h₀ : n = 9) (h₁ : m = 3) : 
  (((1 : ℚ) - (9/14 : ℚ)) = (5/14 : ℚ)) :=
by
  rw h₀ at *,
  rw h₁ at *,
  sorry

end no_adjacent_green_hats_l243_243828


namespace combine_polynomials_find_value_profit_or_loss_l243_243834

-- Problem 1, Part ①
theorem combine_polynomials (a b : ℝ) : -3 * (a+b)^2 - 6 * (a+b)^2 + 8 * (a+b)^2 = -(a+b)^2 := 
sorry

-- Problem 1, Part ②
theorem find_value (a b c d : ℝ) (h1 : a - 2 * b = 5) (h2 : 2 * b - c = -7) (h3 : c - d = 12) : 
  4 * (a - c) + 4 * (2 * b - d) - 4 * (2 * b - c) = 40 := 
sorry

-- Problem 2
theorem profit_or_loss (initial_cost : ℝ) (selling_prices : ℕ → ℝ) (base_price : ℝ) 
  (h_prices : selling_prices 0 = -3) (h_prices1 : selling_prices 1 = 7) 
  (h_prices2 : selling_prices 2 = -8) (h_prices3 : selling_prices 3 = 9) 
  (h_prices4 : selling_prices 4 = -2) (h_prices5 : selling_prices 5 = 0) 
  (h_prices6 : selling_prices 6 = -1) (h_prices7 : selling_prices 7 = -6) 
  (h_initial_cost : initial_cost = 400) (h_base_price : base_price = 56) : 
  (selling_prices 0 + selling_prices 1 + selling_prices 2 + selling_prices 3 + selling_prices 4 + selling_prices 5 + 
  selling_prices 6 + selling_prices 7 + 8 * base_price) - initial_cost > 0 := 
sorry

end combine_polynomials_find_value_profit_or_loss_l243_243834


namespace inequality_proof_l243_243101

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = a * b) : 
  (a / (b^2 + 4) + b / (a^2 + 4) >= 1 / 2) := 
  sorry

end inequality_proof_l243_243101


namespace teresa_siblings_l243_243155

theorem teresa_siblings :
  ∀ (colored_pencils black_pencils pencils_kept pencils_per_sibling : ℕ),
  colored_pencils = 14 →
  black_pencils = 35 →
  pencils_kept = 10 →
  pencils_per_sibling = 13 →
  (colored_pencils + black_pencils - pencils_kept) / pencils_per_sibling = 3 :=
by {
  intros,
  have total_pencils := colored_pencils + black_pencils,
  have shared_pencils := total_pencils - pencils_kept,
  have siblings := shared_pencils / pencils_per_sibling,
  exact siblings
}

end teresa_siblings_l243_243155


namespace unique_polynomial_l243_243293

theorem unique_polynomial (P : Polynomial ℝ) (hP1 : ∀ z : ℂ, IsCoprime (P.toComplexPoly (z - 1) * P.toComplexPoly (z + 1)) 1)
  (hP2 : P.roots.Nodup) : P = X :=
sorry

end unique_polynomial_l243_243293


namespace isosceles_triangle_perimeter_l243_243006

theorem isosceles_triangle_perimeter (a b : ℝ)
  (h1 : b = 7)
  (h2 : a^2 - 8 * a + 15 = 0)
  (h3 : a * 2 > b)
  : 2 * a + b = 17 :=
by
  sorry

end isosceles_triangle_perimeter_l243_243006


namespace angle_BKM_eq_half_alpha_l243_243318

variable (A B C D K M : Point)
variable (α : Real)
variable [metric_space Point] -- Metric space for distance computations
variable [has_area Triangle] -- Area for triangle computations

-- Given conditions
axiom h1 : ray B A contains D
axiom h2 : dist B D = dist B A + dist A C
axiom h3 : ray B A contains K
axiom h4 : ray B C contains M
axiom h5 : area (triangle B D M) = area (triangle B C K)
axiom h6 : angle A B C = α

-- To prove
theorem angle_BKM_eq_half_alpha : angle B K M = α / 2 :=
sorry

end angle_BKM_eq_half_alpha_l243_243318


namespace student_result_correct_l243_243193

variable (p q r : ℝ)
variable (wp wq wr : ℝ)
variable (P p_lt_q : p < q) (q_lt_r : q < r)
variable (wp_eq : wp = 2) (wq_eq : wq = 1) (wr_eq : wr = 3)
variable (sum_eq : wp + wq + wr = 6)

theorem student_result_correct :
  let A := (2 * p + q + 3 * r) / 6
  let B := (let intermediate := (2 * p + q) / 3 in (intermediate * (wp + wq) + r * wr) / (wp + wq + wr))
  A = B :=
by sorry

end student_result_correct_l243_243193


namespace arithmetic_geometric_sequence_problem_l243_243335

-- Definitions based on the conditions
variables {a1 a2 b1 b2 b3 : ℤ}

-- Conditions for arithmetic sequence
axiom h1 : ∀ a1 a2, -1, a1, a2, 8 form an arithmetic sequence → 2 * a1 = -1 + a2
axiom h2 : ∀ a1 a2, -1, a1, a2, 8 form an arithmetic sequence → 2 * a2 = a1 + 8

-- Conditions for geometric sequence
axiom h3 : ∀ b1 b2 b3, -1, b1, b2, b3, -4 form a geometric sequence → b1^2 = -b2
axiom h4 : ∀ b1 b2 b3, -1, b1, b2, b3, -4 form a geometric sequence → b2^2 = 4

-- Theorem statement
theorem arithmetic_geometric_sequence_problem (a1 a2 b2 : ℤ) (ha1: 2 * a1 = -1 + a2) 
  (ha2: 2 * a2 = a1 + 8) (hb2: b2^2 = 4) (hb2_neg: b2 < 0) : 
  a1 * a2 / b2 = -5 :=
by {
  sorry
}

end arithmetic_geometric_sequence_problem_l243_243335


namespace right_triangle_least_side_l243_243398

theorem right_triangle_least_side (a b c : ℝ) (h_rt : a^2 + b^2 = c^2) (h1 : a = 8) (h2 : b = 15) : min a b = 8 := 
by
sorry

end right_triangle_least_side_l243_243398


namespace fractional_pizza_eaten_after_six_trips_l243_243213

def pizza_eaten : ℚ := (1/3) * (1 - (2/3)^6) / (1 - 2/3)

theorem fractional_pizza_eaten_after_six_trips : pizza_eaten = 665 / 729 :=
by
  -- proof will go here
  sorry

end fractional_pizza_eaten_after_six_trips_l243_243213


namespace probability_no_adjacent_green_hats_l243_243820

-- Definitions
def total_children : ℕ := 9
def green_hats : ℕ := 3

-- Main theorem statement
theorem probability_no_adjacent_green_hats : 
  (9.choose 3) = 84 → 
  (1 - (9 + 45) / 84) = 5/14 := 
sorry

end probability_no_adjacent_green_hats_l243_243820


namespace problem_equivalence_l243_243371

theorem problem_equivalence (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) :
  (∀ x : ℝ, (2 * x - 1)^10 = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5 + a6 * (x - 1)^6 + a7 * (x - 1)^7 + a8 * (x - 1)^8 + a9 * (x - 1)^9 + a10 * (x - 1)^10) →
  (a0 = 1 ∧ a2 = 180) := 
begin
  sorry
end

end problem_equivalence_l243_243371


namespace trig_identity_tan_solutions_l243_243125

open Real

theorem trig_identity_tan_solutions :
  ∃ α β : ℝ, (tan α) * (tan β) = -3 ∧ (tan α) + (tan β) = 3 ∧
  abs (sin (α + β) ^ 2 - 3 * sin (α + β) * cos (α + β) - 3 * cos (α + β) ^ 2) = 3 :=
by
  have: ∀ x : ℝ, x^2 - 3*x - 3 = 0 → x = (3 + sqrt 21) / 2 ∨ x = (3 - sqrt 21) / 2 := sorry
  sorry

end trig_identity_tan_solutions_l243_243125


namespace man_speed_against_current_proof_l243_243217

def man_speed_against_current (Vc Vm_with_current : ℕ) : ℕ :=
  Vm_with_current - Vc

theorem man_speed_against_current_proof :
  ∀ (Vm_with_current Vc : ℕ), Vm_with_current = 20 → Vc = 3 → man_speed_against_current Vc Vm_with_current = 14 :=
by
  intros Vm_with_current Vc hVm_with_current hVc
  rw [hVm_with_current, hVc]
  unfold man_speed_against_current
  rfl

end man_speed_against_current_proof_l243_243217


namespace hyperbola_asymptotes_l243_243106

theorem hyperbola_asymptotes (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : c = 2 * a) (h4 : b = sqrt 3 * a) :
  (∀ x : ℝ, abs ((sqrt 3) * x) ≤ b / a * x) :=
sorry

end hyperbola_asymptotes_l243_243106


namespace marissas_sunflower_height_l243_243500

def height_of_marissas_sunflower (sister_height_feet : ℤ) (sister_height_inches : ℤ) (additional_inches : ℤ) : Prop :=
  (sister_height_feet = 4) →
  (sister_height_inches = 3) →
  (additional_inches = 21) →
  sister_height_feet * 12 + sister_height_inches + additional_inches = 72

-- Prove that Marissa's sunflower height in feet is 6
theorem marissas_sunflower_height :
  height_of_marissas_sunflower 4 3 21 →
  72 / 12 = 6 :=
by
  assume h,
  rw Nat.div_eq_of_eq_mul_left sorry,
  sorry

end marissas_sunflower_height_l243_243500


namespace evaluate_ceil_sqrt_225_l243_243940

def ceil (x : ℝ) : ℤ :=
  if h : ∃ n : ℤ, n ≤ x ∧ x < n + 1 then
    classical.some h
  else
    0

theorem evaluate_ceil_sqrt_225 : ceil (Real.sqrt 225) = 15 := 
sorry

end evaluate_ceil_sqrt_225_l243_243940


namespace tangency_at_origin_range_of_a_for_extrema_l243_243035

-- Part 1: Tangency at the origin
theorem tangency_at_origin (a : ℝ) (h_a : a = 1) : (let f := λ x : ℝ, (Real.exp x - a * x^2 - Real.cos x - Real.log (x + 1)) in
  f 0 = 0 ∧ deriv f 0 = 0) := sorry

-- Part 2: Range of a for exactly one extremum in each interval
theorem range_of_a_for_extrema (a : ℝ) :
  (let f := λ x : ℝ, (Real.exp x - a * x^2 - Real.cos x - Real.log (x + 1)) in
  (∃ x ∈ Ioo (-1 : ℝ) 0, deriv f x = 0) ∧
  (∃ x ∈ Ioo 0 +∞, deriv f x = 0)) ↔ a ∈ Ioo (3 / 2 : ℝ) +∞ := sorry

end tangency_at_origin_range_of_a_for_extrema_l243_243035


namespace smallest_four_digit_multiple_of_53_l243_243767

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m : ℕ, m >= 1000 → m < 10000 → m % 53 = 0 → n ≤ m) :=
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243767


namespace tan_neg_3900_eq_sqrt3_l243_243891

theorem tan_neg_3900_eq_sqrt3 : Real.tan (-3900 * Real.pi / 180) = Real.sqrt 3 := by
  -- Definitions of trigonometric values at 60 degrees
  have h_cos : Real.cos (60 * Real.pi / 180) = 1 / 2 := sorry
  have h_sin : Real.sin (60 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Using periodicity of the tangent function
  sorry

end tan_neg_3900_eq_sqrt3_l243_243891


namespace smallest_four_digit_divisible_by_53_l243_243664

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243664


namespace smallest_four_digit_div_by_53_l243_243626

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l243_243626


namespace smallest_four_digit_multiple_of_53_l243_243704

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end smallest_four_digit_multiple_of_53_l243_243704


namespace min_bn_Sn_l243_243274

theorem min_bn_Sn : ∀ n : ℕ, ∃ n', (n' = 2) → (b_n S_n n' = -4) := 
  by 
    sorry

def a_n (n : ℕ) := n * (n + 1)

def S_n (n : ℕ) := 1 - 1 / (n + 1)

def b_n (n : ℕ) := n - 8

end min_bn_Sn_l243_243274


namespace coordinates_of_B_l243_243077

/--
Given point A with coordinates (2, -3) and line segment AB parallel to the x-axis,
and the length of AB being 4, prove that the coordinates of point B are either (-2, -3)
or (6, -3).
-/
theorem coordinates_of_B (x1 y1 : ℝ) (d : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : d = 4) (hx : 0 ≤ d) :
  ∃ x2 : ℝ, ∃ y2 : ℝ, (y2 = y1) ∧ ((x2 = x1 + d) ∨ (x2 = x1 - d)) :=
by
  sorry

end coordinates_of_B_l243_243077


namespace cubic_polynomial_solution_l243_243985

noncomputable def q (x : ℝ) : ℝ := - (4 / 3) * x^3 + 6 * x^2 - (50 / 3) * x - (14 / 3)

theorem cubic_polynomial_solution :
  q 1 = -8 ∧ q 2 = -12 ∧ q 3 = -20 ∧ q 4 = -40 := by
  have h₁ : q 1 = -8 := by sorry
  have h₂ : q 2 = -12 := by sorry
  have h₃ : q 3 = -20 := by sorry
  have h₄ : q 4 = -40 := by sorry
  exact ⟨h₁, h₂, h₃, h₄⟩

end cubic_polynomial_solution_l243_243985


namespace area_of_inscribed_square_l243_243539

theorem area_of_inscribed_square (XY YZ : ℝ) (hXY : XY = 18) (hYZ : YZ = 30) :
  ∃ (s : ℝ), s^2 = 540 :=
by
  sorry

end area_of_inscribed_square_l243_243539


namespace smallest_four_digit_div_by_53_l243_243631

theorem smallest_four_digit_div_by_53 : ∃ n : ℕ, n % 53 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999 ∧ ∀ m : ℕ, (m % 53 = 0 ∧ 1000 ≤ m ∧ m ≤ 9999) → n ≤ m ∧ n = 1007 :=
sorry

end smallest_four_digit_div_by_53_l243_243631


namespace evaluate_ceil_sqrt_225_l243_243944

def ceil (x : ℝ) : ℤ :=
  if h : ∃ n : ℤ, n ≤ x ∧ x < n + 1 then
    classical.some h
  else
    0

theorem evaluate_ceil_sqrt_225 : ceil (Real.sqrt 225) = 15 := 
sorry

end evaluate_ceil_sqrt_225_l243_243944


namespace number_of_valid_pairs_l243_243368

noncomputable def number_of_pairs : ℕ :=
  let m_range := (1, 3019)
  -- Approximate values of logarithms
  let log3 := real.log 3
  let log2 := real.log 2
  let c := log3 / log2
  let pairs := [(m, n) | m ∈ finset.range (m_range.2 + 1),
                         n ∈ finset.range 1000,  -- This is larger than needed but safer for simplicity.
                         1 ≤ m ∧ m ≤ m_range.2 ∧
                         real.rpow 3 n < real.rpow 2 m ∧
                         real.rpow 2 (m + 3) < real.rpow 3 (n + 1)]
  pairs.length

theorem number_of_valid_pairs : number_of_pairs = 2400 :=
by sorry

end number_of_valid_pairs_l243_243368


namespace smallest_four_digit_divisible_by_53_l243_243619

theorem smallest_four_digit_divisible_by_53 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 53 ≠ 0 :=
begin
  sorry
end

end smallest_four_digit_divisible_by_53_l243_243619


namespace problem_statement_l243_243012

theorem problem_statement (a b : ℤ) (c : ℤ) (x : ℤ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a = -b) 
  (h4 : c = -1) (h5 : x^2 = 4) :
  c = -1 ∧ (x = 2 ∨ x = -2) ∧ 
  ((x = 2 → x + a / b + 2 * c - (a + b) / Real.pi = -1) ∧
   (x = -2 → x + a / b + 2 * c - (a + b) / Real.pi = -5)) :=
begin
  sorry
end

end problem_statement_l243_243012


namespace smallest_four_digit_divisible_by_53_l243_243741

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243741


namespace smallest_four_digit_multiple_of_53_l243_243776

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ 53 ∣ n ∧ n = 1007 :=
by
  use 1007
  split
  · linarith
  split
  · linarith
  split
  · norm_num
  rfl
  sorry

end smallest_four_digit_multiple_of_53_l243_243776


namespace intersection_point_of_given_lines_l243_243603

theorem intersection_point_of_given_lines :
  ∃ (x y : ℚ), 2 * y = -x + 3 ∧ -y = 5 * x + 1 ∧ x = -5 / 9 ∧ y = 16 / 9 :=
by
  sorry

end intersection_point_of_given_lines_l243_243603


namespace right_triangle_least_side_l243_243400

theorem right_triangle_least_side (a b c : ℝ) (h_rt : a^2 + b^2 = c^2) (h1 : a = 8) (h2 : b = 15) : min a b = 8 := 
by
sorry

end right_triangle_least_side_l243_243400


namespace smallest_four_digit_divisible_by_53_l243_243669

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243669


namespace tan_expression_independence_l243_243989

theorem tan_expression_independence (k : ℤ) (u : ℝ) :
  (∀ x : ℝ, 
    ∃ y : ℝ,
    y = (tan (x - u) + tan x + tan (x + u)) / (tan (x - u) * tan x * tan (x + u))) ↔ 
  (u = k * π + π / 3 ∨ u = k * π - π / 3) :=
by
  sorry

end tan_expression_independence_l243_243989


namespace smallest_four_digit_divisible_by_53_l243_243740

theorem smallest_four_digit_divisible_by_53 : ∃ n, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ (∀ m, 1000 ≤ m ∧ m < n ∧ m % 53 = 0 → false) :=
by
  sorry

end smallest_four_digit_divisible_by_53_l243_243740


namespace sum_of_squares_99_in_distinct_ways_l243_243269

theorem sum_of_squares_99_in_distinct_ways : 
  ∃ a b c d e f g h i j k l : ℕ, 
    (a^2 + b^2 + c^2 + d^2 = 99) ∧ (e^2 + f^2 + g^2 + h^2 = 99) ∧ (i^2 + j^2 + k^2 + l^2 = 99) ∧ 
    (a ≠ e ∨ b ≠ f ∨ c ≠ g ∨ d ≠ h) ∧ 
    (a ≠ i ∨ b ≠ j ∨ c ≠ k ∨ d ≠ l) ∧ 
    (i ≠ e ∨ j ≠ f ∨ k ≠ g ∨ l ≠ h) 
    :=
sorry

end sum_of_squares_99_in_distinct_ways_l243_243269


namespace symmetric_coordinates_to_plane_xoz_l243_243088

def is_symmetric_to_plane_xoz (P Q : ℝ × ℝ × ℝ) : Prop :=
  P.1 = Q.1 ∧ Q.2 = -P.2 ∧ P.3 = Q.3

theorem symmetric_coordinates_to_plane_xoz :
  let P := (1, 2, 3)
  ∃ Q : ℝ × ℝ × ℝ, is_symmetric_to_plane_xoz P Q ∧ Q = (1, -2, 3) :=
begin
  sorry
end

end symmetric_coordinates_to_plane_xoz_l243_243088


namespace probability_no_two_green_hats_next_to_each_other_l243_243823

open Nat

def choose (n k : ℕ) : ℕ := Nat.fact n / (Nat.fact k * Nat.fact (n - k))

def total_ways_to_choose (n k : ℕ) : ℕ :=
  choose n k

def event_A (n : ℕ) : ℕ := n - 2

def event_B (n k : ℕ) : ℕ := choose (n - k + 1) 2 * (k - 2)

def probability_no_two_next_to_each_other (n k : ℕ) : ℚ :=
  let total_ways := total_ways_to_choose n k
  let event_A_ways := event_A (n)
  let event_B_ways := event_B n 3
  let favorable_ways := total_ways - (event_A_ways + event_B_ways)
  favorable_ways / total_ways

-- Given the conditions of 9 children and choosing 3 to wear green hats
theorem probability_no_two_green_hats_next_to_each_other (p : probability_no_two_next_to_each_other 9 3 = 5 / 14) : Prop := by
  sorry

end probability_no_two_green_hats_next_to_each_other_l243_243823


namespace no_constant_term_in_expansion_l243_243599

theorem no_constant_term_in_expansion : 
  ∀ (x : ℂ), ¬ ∃ (k : ℕ), ∃ (c : ℂ), c * x ^ (k / 3 - 2 * (12 - k)) = 0 :=
by sorry

end no_constant_term_in_expansion_l243_243599


namespace binom_19_12_l243_243331

theorem binom_19_12 :
  (nat.choose 18 11 = 31824) →
  (nat.choose 18 12 = 18564) →
  nat.choose 19 12 = 50388 :=
by 
  intros h1 h2
  exact sorry

end binom_19_12_l243_243331


namespace probability_three_blue_balls_no_replacement_l243_243439

theorem probability_three_blue_balls_no_replacement : 
  let total_balls := 14
  let blue_balls := 6 in
  (blue_balls / total_balls) * 
  ((blue_balls - 1) / (total_balls - 1)) * 
  ((blue_balls - 2) / (total_balls - 2)) = 5 / 91 :=
by {
  sorry
}

end probability_three_blue_balls_no_replacement_l243_243439


namespace triangle_CE_value_l243_243464

theorem triangle_CE_value :
  ∀ {A B C D E : Type} [has_zero A] [has_one A] [division_ring A] [vector_space ℝ A]
  (angle_BAC : real.angle) (angle_ACB : real.angle) (BC : A) (CE : A)
  (midpoint_M : A) (BD_perp_AM : Prop) (DE_equals_EB : A),
  angle_BAC = 30 ∧ angle_ACB = 60 ∧ BC = 1 ∧ midpoint_M = (B + C) / 2 ∧
  BD_perp_AM ∧ DE_equals_EB ∧ D ∈ segment A C ∧ E ∈ line_through C D →
  CE = 3 / 2 :=
by
  sorry

end triangle_CE_value_l243_243464


namespace more_non_representable_ten_digit_numbers_l243_243871

-- Define the range of ten-digit numbers
def total_ten_digit_numbers : ℕ := 9 * 10^9

-- Define the range of five-digit numbers
def total_five_digit_numbers : ℕ := 90000

-- Calculate the number of pairs of five-digit numbers
def number_of_pairs_five_digit_numbers : ℕ :=
  total_five_digit_numbers * (total_five_digit_numbers + 1)

-- Problem statement
theorem more_non_representable_ten_digit_numbers:
  number_of_pairs_five_digit_numbers < total_ten_digit_numbers :=
by
  -- Proof is non-computable and should be added here
  sorry

end more_non_representable_ten_digit_numbers_l243_243871


namespace samatha_routes_l243_243144

-- Definitions based on the given conditions
def blocks_from_house_to_southwest_corner := 4
def blocks_through_park := 1
def blocks_from_northeast_corner_to_school := 4
def blocks_from_school_to_library := 3

-- Number of ways to arrange movements
def number_of_routes_house_to_southwest : ℕ :=
  Nat.choose blocks_from_house_to_southwest_corner 1

def number_of_routes_through_park : ℕ := blocks_through_park

def number_of_routes_northeast_to_school : ℕ :=
  Nat.choose blocks_from_northeast_corner_to_school 1

def number_of_routes_school_to_library : ℕ :=
  Nat.choose blocks_from_school_to_library 1

-- Total number of different routes
def total_number_of_routes : ℕ :=
  number_of_routes_house_to_southwest *
  number_of_routes_through_park *
  number_of_routes_northeast_to_school *
  number_of_routes_school_to_library

theorem samatha_routes (n : ℕ) (h : n = 48) :
  total_number_of_routes = n :=
  by
    -- Proof is skipped
    sorry

end samatha_routes_l243_243144
