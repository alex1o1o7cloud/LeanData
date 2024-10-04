import Mathlib
import Mathlib.Algebra.ArithmeticMean
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Lcm
import Mathlib.Algebra.NumberTheory.LinearDiophantine
import Mathlib.Algebra.Order.Ceil
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Real
import Mathlib.Algebra.Trig
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Integral
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Factorial
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Lcm
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Polynomial.Degree
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.SinCos
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Interval
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Triangle
import Mathlib.GroupTheory.Permutation
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.ContinuousFunction.Basic
import Mathlib.Topology.EuclideanSpace.Basic
import Mathlib.Trigonometry.Basic

namespace price_of_gift_l501_501944

structure FinancialContributions where
  lisa_savings : ℕ
  mother_multipler : ℝ
  brother_multiplier : ℝ
  additional_amount_needed : ℕ

def lisa : FinancialContributions :=
{ lisa_savings := 1200,
  mother_multipler := 3 / 5,
  brother_multiplier := 2,
  additional_amount_needed := 400 }

theorem price_of_gift : lisa.lisa_savings + (lisa.lisa_savings * lisa.mother_multipler).to_nat + (2 * (lisa.lisa_savings * lisa.mother_multipler).to_nat) + lisa.additional_amount_needed = 3760 :=
by
  sorry

end price_of_gift_l501_501944


namespace solve_quadratic_roots_l501_501945

theorem solve_quadratic_roots (m : ℤ) (h1 : m ≠ 0) (h2 : m ≠ 1) : 
  ∃ x1 x2 : ℤ, x1 ≠ x2 ∧
  (mx^2 - (m+1)x + 1 = 0) :=
by
  have h : m = -1 := sorry
  exact ⟨-1, 1, by simp⟩

end solve_quadratic_roots_l501_501945


namespace max_reflections_l501_501631

def angle_CDA := 12
def max_angle := 90

theorem max_reflections (n : ℕ) (angle_increment : ℕ) :
  angle_increment = angle_CDA →
  ∀ k, k ∈ (finset.range n) → k * angle_increment < max_angle → n ≤ 7 :=
by
  sorry

end max_reflections_l501_501631


namespace probability_two_heads_in_three_flips_l501_501093

theorem probability_two_heads_in_three_flips :
  let flip_prob := 1 / 2
  ∧ (ways_to_get_two_heads := 3)
  → (total_flips := 3)
  → let prob_two_heads := ways_to_get_two_heads * (flip_prob ^ 2 * (1 - flip_prob))
in prob_two_heads = 3 / 8 :=
by
  sorry

end probability_two_heads_in_three_flips_l501_501093


namespace car_price_l501_501380

theorem car_price (down_payment : ℕ) (monthly_payment : ℕ) (loan_years : ℕ) 
    (h_down_payment : down_payment = 5000) 
    (h_monthly_payment : monthly_payment = 250)
    (h_loan_years : loan_years = 5) : 
    down_payment + monthly_payment * loan_years * 12 = 20000 := 
by
  rw [h_down_payment, h_monthly_payment, h_loan_years]
  norm_num
  sorry

end car_price_l501_501380


namespace triangle_division_l501_501927

theorem triangle_division (n : ℕ) (h : n = 2002) : 
  divisible_by_the_number_of_divided_regions (divide_triangle_into_congruent_segments n) 6 :=
sorry

end triangle_division_l501_501927


namespace mouse_sees_cheese_5_times_l501_501888

-- Definitions of the points and segments
def point : Type := ℝ × ℝ

def A : point := (0, 0)
def B : point := (400, 0)
def D : point := (200, 200)
def C : point := (400, 200)

-- Assume BD = DC
axiom BD_eq_DC : dist B D = dist D C

-- Definition of point E (cheese location)
def E : point := (300, 200)

-- Definitions of mirrors on AD
def W₁ : point := (50, 50)
def W₂ : point := (100, 100)
def W₃ : point := (150, 150)

-- Distances
def AB : ℝ := dist A B
def AC : ℝ := dist A C
def AD : ℝ := dist A D

-- Traversal conditions
def mouse_position (n : ℕ) : ℝ :=
  if even n then 80 * (n + 1) / 2 - 20 * n / 2
  else 80 * (n + 1) / 2 - 20 * (n - 1) / 2

-- Predicate to check whether the mouse sees the cheese
def sees_cheese (n : ℕ) : Prop :=
  let pos := mouse_position n in
  pos = 60 ∨ pos = 150 ∨ pos = 300

theorem mouse_sees_cheese_5_times :
  ∃ (times : list ℕ), list.length times = 5 ∧ ∀ t ∈ times, sees_cheese t :=
sorry

end mouse_sees_cheese_5_times_l501_501888


namespace magic_square_y_value_l501_501307

/-- In a magic square, where the sum of three entries in any row, column, or diagonal is the same value.
    Given the entries as shown below, prove that \(y = -38\).
    The entries are: 
    - \( y \) at position (1,1)
    - 23 at position (1,2)
    - 101 at position (1,3)
    - 4 at position (2,1)
    The remaining positions are denoted as \( a, b, c, d, e \).
-/
theorem magic_square_y_value :
    ∃ y a b c d e: ℤ,
        y + 4 + c = y + 23 + 101 ∧ -- Condition from first column and first row
        23 + a + d = 101 + b + 4 ∧ -- Condition from middle column and diagonal
        c + d + e = 101 + b + e ∧ -- Condition from bottom row and rightmost column
        y + 23 + 101 = 4 + a + b → -- Condition from top row
        y = -38 := 
by
    sorry

end magic_square_y_value_l501_501307


namespace vector_sum_magnitude_l501_501934

variables {point : Type} [metric_space point] [add_comm_group point] [module ℝ point]
variables (O A B : point)
variables (x1 y1 x2 y2 : ℝ)

def ellipse (x y : ℝ) := (x^2) / 4 + y^2 = 1

def symmetric_about_line (x1 y1 x2 y2 : ℝ) :=
  ∃ x0 y0 : ℝ, 4 * x0 - 2 * y0 - 3 = 0 ∧ x1 + x2 = 2 * x0 ∧ y1 + y2 = 2 * y0

def vector_sum := (λ (a b : point), a + b)
def magnitude (v : point) := ∥v∥

noncomputable def coordinates_to_point (x y : ℝ) : point := sorry

axiom O_is_origin: O = coordinates_to_point 0 0
axiom A_is_ellipse: ellipse x1 y1
axiom B_is_ellipse: ellipse x2 y2
axiom A_B_symmetric: symmetric_about_line x1 y1 x2 y2

theorem vector_sum_magnitude :
  magnitude (vector_sum (coordinates_to_point x1 y1) (coordinates_to_point x2 y2)) = real.sqrt(5) := sorry

end vector_sum_magnitude_l501_501934


namespace no_such_n_exists_l501_501525

theorem no_such_n_exists :
  ¬ ∃ n : ℕ, 0 < n ∧
  (∃ a : ℕ, 2 * n^2 + 1 = a^2) ∧
  (∃ b : ℕ, 3 * n^2 + 1 = b^2) ∧
  (∃ c : ℕ, 6 * n^2 + 1 = c^2) :=
sorry

end no_such_n_exists_l501_501525


namespace intersection_with_xz_plane_l501_501924

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def direction_vector (p1 p2 : Point3D) : Point3D :=
  Point3D.mk (p2.x - p1.x) (p2.y - p1.y) (p2.z - p1.z)

def parametric_eqn (p : Point3D) (d : Point3D) (t : ℝ) : Point3D :=
  Point3D.mk (p.x + t * d.x) (p.y + t * d.y) (p.z + t * d.z)

theorem intersection_with_xz_plane (p1 p2 : Point3D) :
  let d := direction_vector p1 p2
  let t := (p1.y / d.y)
  parametric_eqn p1 d t = Point3D.mk 4 0 9 :=
sorry

#check intersection_with_xz_plane

end intersection_with_xz_plane_l501_501924


namespace marble_distribution_l501_501167

theorem marble_distribution :
  ∃ (M1 M2 B G : ℕ), 
  M1 + M2 = 1000 ∧
  M2 = M1 - 50 ∧
  B + G = M2 ∧
  B = G + 35 ∧
  17 ∣ B ∧
  525 - B / 17 = 510 :=
by
  have h1 : M1 + (M1 - 50) = 1000 := sorry,
  have h2 : M2 = M1 - 50 := sorry,
  have h3 : B + G = 475 := sorry,
  have h4 : B = G + 35 := sorry,
  have h5 : 17 ∣ 255 := sorry,
  exact ⟨525, 475, 255, 220, h1, h2, h3, h4, h5, sorry⟩

end marble_distribution_l501_501167


namespace beef_original_weight_l501_501147

theorem beef_original_weight (final_weight : ℝ) (loss_percent : ℝ) (original_weight: ℝ) : original_weight = final_weight / (1 - loss_percent) :=
by
  -- Given conditions
  let final_weight := 500   -- Final weight after processing is 500 pounds
  let loss_percent := 0.30  -- Loss percentage is 30%
  let original_weight := final_weight / (1 - loss_percent)  -- Original weight formula
  show original_weight = final_weight / (1 - loss_percent) from rfl
  sorry

end beef_original_weight_l501_501147


namespace work_days_B_C_l501_501775

variable (A B C : ℚ)
variable (work_days : ℚ)

noncomputable def work_rate (days : ℚ) := (1 : ℚ) / days

theorem work_days_B_C :
  A + B = work_rate 8 →
  A + B + C = work_rate 6 →
  A + C = work_rate 8 →
  work_days = work_rate (1 / (B + C)) →
  work_days = 12 := 
by
  intros h1 h2 h3 h4
  rw [←h4]
  sorry

end work_days_B_C_l501_501775


namespace solve_dio_eq_best_approx_l501_501683

-- Define the conditions
def is_square_free (d : ℕ) : Prop := 
  ∀ m : ℕ, m^2 ∣ d → m = 1

def solution_equation (x y d : ℕ) : Prop :=
  x^2 - d * y^2 = 1 ∨ x^2 - d * y^2 = -1

def best_approximation (x y : ℕ) (d : ℕ) : Prop :=
  ∀ a b : ℕ, b > 0 → (|a / b - real.sqrt d| ≥ |x / y - real.sqrt d|)

-- The main theorem to prove
theorem solve_dio_eq_best_approx (d x y : ℕ) (h1 : is_square_free d) (h2 : solution_equation x y d) :
  best_approximation x y d :=
begin
  sorry
end

end solve_dio_eq_best_approx_l501_501683


namespace equal_commissions_implies_list_price_l501_501880

theorem equal_commissions_implies_list_price (x : ℝ) :
  (0.15 * (x - 15) = 0.25 * (x - 25)) → x = 40 :=
by
  intro h
  sorry

end equal_commissions_implies_list_price_l501_501880


namespace ace_first_king_second_prob_l501_501782

def cards : Type := { x : ℕ // x < 52 }

def ace (c : cards) : Prop := 
  c.1 = 0 ∨ c.1 = 1 ∨ c.1 = 2 ∨ c.1 = 3

def king (c : cards) : Prop := 
  c.1 = 4 ∨ c.1 = 5 ∨ c.1 = 6 ∨ c.1 = 7

def prob_ace_first_king_second : ℚ := 4 / 52 * 4 / 51

theorem ace_first_king_second_prob :
  prob_ace_first_king_second = 4 / 663 := by
  sorry

end ace_first_king_second_prob_l501_501782


namespace eq_proof_l501_501780

variables (EF FG GH HE : ℝ) (parallel_EF_GH : EF = 100 ∧ GH = 22 ∧ EF = 60 ∧ HE = 80)

-- Definition of the problem setup
def trapezoid_EFGH : Prop :=
  EF = 100 ∧ FG = 60 ∧ GH = 22 ∧ HE = 80 ∧ EF ≠ 0 ∧ GH ≠ 0

-- Circle properties & the problem to solve
def circle_properties : Prop :=
  ∃ (EQ QF : ℝ), (EQ + QF = 100) ∧ (∀ y, (y * (EQ^2 - 100 * EQ) = 2600 - 140 * EQ)) ∧ EQ = 40

-- Reduced fraction and coprime condition
def reduced_fraction_conditions : Prop :=
  ∃ p q : ℕ, p = 40 ∧ q = 1 ∧ (nat.gcd p q = 1) ∧ (p + q = 41)

-- The main theorem to prove
theorem eq_proof : trapezoid_EFGH EF FG GH HE parallel_EF_GH → circle_properties EF FG GH HE → reduced_fraction_conditions := 
  sorry

end eq_proof_l501_501780


namespace find_constant_l501_501281

theorem find_constant (c : ℝ) (f : ℝ → ℝ)
  (h : f x = c * x^3 + 19 * x^2 - 4 * c * x + 20)
  (hx : f (-7) = 0) :
  c = 3 :=
sorry

end find_constant_l501_501281


namespace initial_crayons_per_box_l501_501360

-- Define the initial total number of crayons in terms of x
def total_initial_crayons (x : ℕ) : ℕ := 4 * x

-- Define the crayons given to Mae
def crayons_to_Mae : ℕ := 5

-- Define the crayons given to Lea
def crayons_to_Lea : ℕ := 12

-- Define the remaining crayons
def remaining_crayons : ℕ := 15

-- Prove that the initial number of crayons per box is 8 given the conditions
theorem initial_crayons_per_box (x : ℕ) : total_initial_crayons x - crayons_to_Mae - crayons_to_Lea = remaining_crayons → x = 8 :=
by
  intros h
  sorry

end initial_crayons_per_box_l501_501360


namespace exists_x0_lt_l501_501764

theorem exists_x0_lt
  (a b c d p q : ℝ)
  (s t : ℝ) 
  (hst : t - s ≥ 2)
  (P : ℝ → ℝ := λ x, x^4 + a * x^3 + b * x^2 + c * x + d)
  (Q : ℝ → ℝ := λ x, x^2 + p * x + q) :
  (∀ x, s ≤ x ∧ x ≤ t → P x < 0 ∧ Q x < 0) →
  (∀ x, x < s ∨ x > t → P x ≥ 0 ∧ Q x ≥ 0) →
  ∃ x0 : ℝ, P x0 < Q x0 :=
begin
  sorry
end

end exists_x0_lt_l501_501764


namespace area_of_triangle_ABC_l501_501429

section TriangleArea

-- Define the vertices of the triangle A, B, and C
def A := (0 : ℝ, 2 : ℝ)
def B := (3 : ℝ, 0 : ℝ)
def C := (1 : ℝ, 6 : ℝ)

-- Function to compute the area of a triangle using Shoelace formula
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  1 / 2 * real.abs (
    A.1 * B.2 + B.1 * C.2 + C.1 * A.2
    - A.2 * B.1 - B.2 * C.1 - C.2 * A.1
  )

-- Proof statement
theorem area_of_triangle_ABC :
  triangle_area A B C = 7 := by
  sorry

end TriangleArea

end area_of_triangle_ABC_l501_501429


namespace equilateral_triangles_equal_segments_l501_501365

theorem equilateral_triangles_equal_segments
  (A B C A₁ B₁ C₁ : Point)
  (triangle_ABC : Triangle A B C)
  (eq_trian_A1BC : EquilateralTriangle A₁ B C)
  (eq_trian_AB1C : EquilateralTriangle A B₁ C)
  (eq_trian_ABC1 : EquilateralTriangle A B C₁) :
  dist A A₁ = dist B B₁ ∧ dist B B₁ = dist C C₁ := 
sorry

end equilateral_triangles_equal_segments_l501_501365


namespace frustum_lateral_surface_area_l501_501470

theorem frustum_lateral_surface_area (r1 r2 h : ℝ) (hr1 : r1 = 8) (hr2 : r2 = 4) (hh : h = 5) :
  let d := r1 - r2
  let s := Real.sqrt (h^2 + d^2)
  let A := Real.pi * s * (r1 + r2)
  A = 12 * Real.pi * Real.sqrt 41 :=
by
  -- hr1 and hr2 imply that r1 and r2 are constants, therefore d = 8 - 4 = 4
  -- h = 5 and d = 4 imply s = sqrt (5^2 + 4^2) = sqrt 41
  -- The area A is then pi * sqrt 41 * (8 + 4) = 12 * pi * sqrt 41
  sorry

end frustum_lateral_surface_area_l501_501470


namespace sequence_periodic_l501_501418

def sequence (n : ℕ) : ℤ :=
if n = 1 then 20
else if n = 2 then 17
else sequence (n - 1) - sequence (n - 2)

theorem sequence_periodic (n : ℕ) (hn : 2018 = 6 * n + 2) : sequence 2018 = 17 :=
by sorry

end sequence_periodic_l501_501418


namespace sum_reciprocals_transformed_roots_l501_501668

theorem sum_reciprocals_transformed_roots (a b c : ℝ) (h : ∀ x, (x^3 - 2 * x - 5 = 0) → (x = a) ∨ (x = b) ∨ (x = c)) : 
  (1 / (a - 2)) + (1 / (b - 2)) + (1 / (c - 2)) = 10 := 
by sorry

end sum_reciprocals_transformed_roots_l501_501668


namespace time_after_duration_is_expected_l501_501823

open Nat

def start_time : String := "2021-01-05 15:00"  -- January 5, 2021 at 3:00 PM
def duration_minutes : Nat := 5050  -- 5050 minutes

def expected_time : String := "2021-01-09 03:10"  -- January 9, 2021 at 3:10 AM

theorem time_after_duration_is_expected :
  let start_time := "2021-01-05 15:00";
  let duration := 5050;
  let expected := "2021-01-09 03:10";
  calculateTimeAfter(start_time, duration) = expected := sorry

-- hypothetical function to calculate time after a duration, unlike the proof writing, 
-- creating such functions is beyond the scope of this translation
def calculateTimeAfter (start: String) (duration: Nat) : String :=
  sorry  

end time_after_duration_is_expected_l501_501823


namespace chord_length_integer_lines_l501_501599

theorem chord_length_integer_lines :
  ∃ (n : ℕ), n = 10 ∧
    ∀ (k : ℝ), 
      let l := (λ x y : ℝ, k * x - y - 4 * k + 1 = 0) in
      let C := (λ x y : ℝ, x^2 + (y + 1)^2 = 25) in
      (∃ (length : ℝ), length ∈ ℤ ∧
        ∃ (P : ℝ × ℝ), l P.1 P.2 ∧ C P.1 P.2) := sorry

end chord_length_integer_lines_l501_501599


namespace hiking_hours_l501_501790

-- Define the given conditions
def water_needed_violet_per_hour : ℕ := 800
def water_needed_dog_per_hour : ℕ := 400
def total_water_carry_capacity_liters : ℕ := 4.8 * 1000 -- converted to ml

-- Define the statement to prove
theorem hiking_hours : (total_water_carry_capacity_liters / (water_needed_violet_per_hour + water_needed_dog_per_hour)) = 4 := by
  sorry

end hiking_hours_l501_501790


namespace problem_statement_l501_501056

noncomputable def f (x : ℝ) : ℝ := log 0.3 (5 + 4 * x - x^2)

variable (a : ℝ)

def b : ℝ := log 0.3 1

def c : ℝ := 2^(1/3)

theorem problem_statement 
  (H1 : ∀ x₁ x₂, (a - 1 < x₁ ∧ x₁ < a + 1) → 
                 (a - 1 < x₂ ∧ x₂ < a + 1) → 
                 x₁ < x₂ → f x₁ ≥ f x₂)
  (H2 : 0 ≤ a ∧ a ≤ 1) 
  : b < a ∧ a < c :=
sorry

end problem_statement_l501_501056


namespace extremum_at_two_f_monotonicity_point_in_interval_l501_501511

noncomputable def f (x : ℝ) (p : ℝ) := p * (x - x⁻¹) - 2 * log x
noncomputable def g (x : ℝ) := (2 * real.exp 1) / x

-- Condition 1: If \( f(x) \) has an extremum at x = 2
theorem extremum_at_two (p : ℝ) : 
  (∀ x : ℝ, p * (1 + x^(-2)) - (2 / x) = 0 → x = 2) → p = 4 / 5 := 
sorry

-- Condition 2: Monotonicity on the domain
theorem f_monotonicity (p : ℝ) : 
  (∀ x : ℝ, (f (x) p) ≥ (f (y) p) ∨ (f (x) p) ≤ (f (y) p)) →
  (p ≤ 0 ∨ p ≥ 1) :=
sorry

-- Condition 3: There exists at least one point \( x_0 \) in [1, e] such that f(x_0) > g(x_0)
theorem point_in_interval (p : ℝ) : 
  (∃ x₀ ∈ set.Icc 1 (real.exp 1), f x₀ p > g x₀) →
  p > 4 * real.exp 1 / (real.exp 1^2 - 1) :=
sorry

end extremum_at_two_f_monotonicity_point_in_interval_l501_501511


namespace triangle_shape_l501_501244

theorem triangle_shape (a b c : ℝ) (h : a^4 - b^4 + (b^2 * c^2 - a^2 * c^2) = 0) :
  (a = b) ∨ (a^2 + b^2 = c^2) :=
sorry

end triangle_shape_l501_501244


namespace find_a_l501_501690

noncomputable def findPositiveA (X : ℝ → ℝ) (σ : ℝ) (a : ℝ) : Prop :=
  ∃ (a : ℝ), a > 0 ∧ (∀ (X : ℝ), X = gaussianPDF(1, σ) →  P(X ≤ a^2 - 1) = P(X > a - 3)) → a = 2

theorem find_a (X : ℝ → ℝ) (σ : ℝ) : findPositiveA X σ 2 :=
sorry

end find_a_l501_501690


namespace integral_evaluation_l501_501901

open Set
open Real

noncomputable def integral_cos_div_sin_cos := 
  ∫ x in 0..(π/2), (cos x) / (1 + sin x + cos x) = (π / 4) - (1 / 2) * log 2

theorem integral_evaluation : integral_cos_div_sin_cos := by
  sorry

end integral_evaluation_l501_501901


namespace parabola_exists_l501_501966

noncomputable def parabola_conditions (a b : ℝ) : Prop :=
  (a + b = -3) ∧ (4 * a - 2 * b = 12)

noncomputable def translated_min_equals_six (m : ℝ) : Prop :=
  (m > 0) ∧ ((-1 - 2 + m)^2 - 3 = 6) ∨ ((3 - 2 - m)^2 - 3 = 6)

theorem parabola_exists (a b m : ℝ) (x y : ℝ) :
  parabola_conditions a b → y = x^2 + b * x + 1 → translated_min_equals_six m →
  (y = x^2 - 4 * x + 1) ∧ (m = 6 ∨ m = 4) := 
by 
  sorry

end parabola_exists_l501_501966


namespace proportion_AB_BC_2_plus_sqrt6_l501_501377

variable {A B C D E : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variables [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C] [InnerProductSpace ℝ D] [InnerProductSpace ℝ E]

theorem proportion_AB_BC_2_plus_sqrt6 
  (AB BC CD : ℝ)
  (h1 : AB = BC ^ 2)
  (h2 : right_angle B C)
  (h3 : right_angle C D)
  (h4 : triangle_similar ABC BCD)
  (h5 : AB > BC)
  (h6 : ∃ E, triangle_similar ABC CEB ∧ area AED = 20 * area CEB) :
  AB / BC = 2 + sqrt 6 := by
  sorry

end proportion_AB_BC_2_plus_sqrt6_l501_501377


namespace max_value_of_norms_l501_501686

open Real EuclideanSpace

variables {E : Type*} [inner_product_space ℝ E]

theorem max_value_of_norms (a b c : E) 
  (ha : ∥a∥ = 3) (hb : ∥b∥ = 4) (hc : ∥c∥ = 2) :
  ∥a - 3 • b∥^2 + ∥b - 3 • c∥^2 + ∥c - 3 • a∥^2 ≤ 253 :=
sorry

end max_value_of_norms_l501_501686


namespace find_g_at_75_l501_501735

noncomputable def g : ℝ → ℝ := sorry

-- Conditions
axiom g_property : ∀ (x y : ℝ), x > 0 → y > 0 → g (x * y) = g x / y^2
axiom g_at_50 : g 50 = 25

-- The main result to be proved
theorem find_g_at_75 : g 75 = 100 / 9 :=
by
  sorry

end find_g_at_75_l501_501735


namespace subset_range_l501_501112

open Set

-- Definitions of sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- The statement of the problem
theorem subset_range (a : ℝ) (h : A ⊆ B a) : 2 ≤ a :=
sorry -- Skipping the proof

end subset_range_l501_501112


namespace find_angle_CAB_l501_501665

variable (A B C K C1 B1 B2 C2 : Type)
variable [InCenter A B C K]
variable [Midpoint C1 A B]
variable [Midpoint B1 A C]
variable [IntersectLine AC C1 K B2]
variable [IntersectLine AB B1 K C2]
variable [AreaEquality (Triangle A B2 C2) (Triangle ABC)]

theorem find_angle_CAB : ∠ CAB = 60 :=
sorry

end find_angle_CAB_l501_501665


namespace ab_eq_10_pow_117_l501_501039

theorem ab_eq_10_pow_117
  (a b : ℝ)
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : (∃ m n : ℕ, m = sqrt(log a) ∧ n = sqrt(log b) ∧ 
        m + n + (1/2) * m^2 + (1/2) * n^2 = 108)) 
  : a * b = 10^117 :=
sorry

end ab_eq_10_pow_117_l501_501039


namespace division_by_fraction_l501_501895

theorem division_by_fraction :
  (5 / (8 / 15) : ℚ) = 75 / 8 :=
by
  sorry

end division_by_fraction_l501_501895


namespace find_pq_sum_l501_501681

theorem find_pq_sum (p q : ℝ) (h : (2 + complex.i : ℂ) is_root_of (λ x : ℂ, x^3 + p * x + q)) : p + q = 9 :=
by
  sorry

end find_pq_sum_l501_501681


namespace range_of_function_l501_501815

theorem range_of_function :
  ∀ (y : ℝ),
    (∃ x : ℝ, x ≠ -1 ∧ y = (x^2 + 4*x + 3)/(x + 1)) ↔ (y ∈ Set.Ioo (-∞) 2 ∪ Set.Ioo 2 ∞) :=
by
  sorry

end range_of_function_l501_501815


namespace max_value_of_m_l501_501968

variable (m : ℝ)

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
∀ x > 0, m * x * Real.log x - (x + m) * Real.exp ((x - m) / m) ≤ 0

theorem max_value_of_m (h1 : 0 < m) (h2 : satisfies_inequality m) : m ≤ Real.exp 2 := sorry

end max_value_of_m_l501_501968


namespace prices_correct_minimum_cost_correct_l501_501843

-- Define the prices of the mustard brands
variables (x y m : ℝ)

def brandACost : ℝ := 9 * x + 6 * y
def brandBCost : ℝ := 5 * x + 8 * y

-- Conditions for prices
axiom cost_condition1 : brandACost x y = 390
axiom cost_condition2 : brandBCost x y = 310

-- Solution for prices
def priceA : ℝ := 30
def priceB : ℝ := 20

theorem prices_correct : x = priceA ∧ y = priceB :=
sorry

-- Conditions for minimizing cost
def totalCost (m : ℝ) : ℝ := 30 * m + 20 * (30 - m)
def totalPacks : ℝ := 30

-- Constraints
def constraint1 (m : ℝ) : Prop := m ≥ 5 + (30 - m)
def constraint2 (m : ℝ) : Prop := m ≤ 2 * (30 - m)

-- Minimum cost condition
def min_cost : ℝ := 780
def optimal_m : ℝ := 18

theorem minimum_cost_correct : constraint1 optimal_m ∧ constraint2 optimal_m ∧ totalCost optimal_m = min_cost :=
sorry

end prices_correct_minimum_cost_correct_l501_501843


namespace AHYO_concyclic_l501_501335

variables {A B C O H Y Z : Point}
variables {triangle_ABC : triangle A B C}
variables {triangle_AYZ : triangle A Y Z}
variables {circle_ABC : circle A B C}
variables {circle_AYZ : circle A Y Z}

-- Condition: ABC is a scalene triangle with circumcenter O and orthocenter H.
axiom scalene_triangle_ABC : scalene triangle_ABC
axiom circumcenter_ABC : is_circumcenter O triangle_ABC
axiom orthocenter_ABC : is_orthocenter H triangle_ABC

-- Condition: AYZ is a triangle with circumcenter H and orthocenter O.
axiom circumcenter_AYZ : is_circumcenter H triangle_AYZ
axiom orthocenter_AYZ : is_orthocenter O triangle_AYZ

-- Condition: Z is on BC.
axiom Z_on_BC : Z ∈ line_through B C

-- Goal: Show that A, H, O, Y are concyclic.
theorem AHYO_concyclic : concyclic A H O Y := sorry

end AHYO_concyclic_l501_501335


namespace y_coordinate_equidistant_l501_501802

theorem y_coordinate_equidistant :
  ∃ y : ℝ, (∀ P : ℝ × ℝ, P = (0, y) → dist (3, 0) P = dist (2, 5) P) ∧ y = 2 := 
by
  sorry

end y_coordinate_equidistant_l501_501802


namespace integer_values_of_Sn_l501_501067

noncomputable def sequence_an : ℕ → ℤ → ℚ
| 1, _ := 4 / 3
| (n+1), a_n := a_n * (a_n - 1) + 1

noncomputable def sequence_Sn (n : ℕ) : ℚ :=
∑ i in finset.range n, (1 / sequence_an i 4/3)

theorem integer_values_of_Sn (n : ℕ) (S_n := sequence_Sn n) :
  ∃ (s : ℤ), S_n ∈ ({0, 1, 2} : set ℤ) :=
sorry

end integer_values_of_Sn_l501_501067


namespace parabola_vertex_sum_l501_501181

theorem parabola_vertex_sum 
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, (a * x^2 + b * x + c) = (a * (x + 3)^2 + 4))
  (h2 : (a * 49 + 4) = -2)
  : a + b + c = 100 / 49 :=
by
  sorry

end parabola_vertex_sum_l501_501181


namespace line_intersects_y_axis_at_l501_501136

theorem line_intersects_y_axis_at (x₁ y₁ x₂ y₂ : ℝ) (h₁ : x₁ = 3) (h₂ : y₁ = 27) (h₃ : x₂ = -7) (h₄ : y₂ = -3) :
  ∃ y : ℝ, (0, y) ∈ set_of (λ p : ℝ × ℝ, ∃ m b : ℝ, p.2 = m * p.1 + b ∧ 
                                      m = (y₂ - y₁) / (x₂- x₁) ∧ 
                                      b = y₁ - m * x₁) ∧ y = 18 :=
by {
  sorry
}

end line_intersects_y_axis_at_l501_501136


namespace possible_second_game_scores_count_l501_501784

theorem possible_second_game_scores_count :
  ∃ (A1 A3 B2 : ℕ),
  (A1 + A3 = 22) ∧ (B2 = 11) ∧ (A1 < 11) ∧ (A3 < 11) ∧ ((B2 - A2 = 2) ∨ (B2 >= A2 + 2)) ∧ (A1 + B1 + A2 + B2 + A3 + B3 = 62) :=
  sorry

end possible_second_game_scores_count_l501_501784


namespace smallest_n_probability_red_apple_lt_half_l501_501123

theorem smallest_n_probability_red_apple_lt_half : 
  ∃ (n : ℕ), (n < 15) ∧ let R := 9 in let T := 15 in 
               (R - n) / (T - n) < 0.5 ∧ ∀ m < n, (R - m) / (T - m) > 0.5 :=
by
  -- Completing the proof is not necessary for the statement
  sorry

end smallest_n_probability_red_apple_lt_half_l501_501123


namespace data_set_properties_l501_501146

def data_set : List ℝ := [1, 2, 4, 4, 7, 10, 14]

def median (l : List ℝ) : ℝ := 
  let sorted := List.sort l
  sorted.get! (sorted.length / 2)

def mean (l : List ℝ) : ℝ := l.sum / l.length

def range (l : List ℝ) : ℝ := l.maximum' - l.minimum'

def variance (l : List ℝ) : ℝ :=
  let m := mean l
  l.sumBy (λ x => (x - m) ^ 2) / l.length

theorem data_set_properties :
  median data_set = 4 ∧
  mean data_set = 6 ∧
  range data_set = 13 ∧
  variance data_set = 130 / 7 := by
  sorry

end data_set_properties_l501_501146


namespace floor_equation_l501_501277

theorem floor_equation (n : ℤ) (h : ⌊(n^2 : ℤ) / 4⌋ - ⌊n / 2⌋^2 = 5) : n = 11 :=
sorry

end floor_equation_l501_501277


namespace binom_19_13_l501_501577

open Nat

theorem binom_19_13 :
  ∀ (binom_1811 binom_1812 binom_2013 : ℕ)
  (h1 : binom_1811 = 31824)
  (h2 : binom_1812 = 18564)
  (h3 : binom_2013 = 77520)
  (pascal : ∀ n k, binomial (n + 1) k = binomial n (k - 1) + binomial n k),
  binomial 19 13 = 58956 :=
by
  sorry

end binom_19_13_l501_501577


namespace matrix_transformation_property_l501_501935

-- Define a 2x2 matrix
structure Matrix (m n : Nat) where
  data : Fin m → Fin n → ℤ

-- Define matrix multiplication for 2x2 matrices
def matrix_mul (A B : Matrix 2 2) : Matrix 2 2 := 
  ⟨λ i j => 
    match i, j with
    | 0, 0 => A.data 0 0 * B.data 0 0 + A.data 0 1 * B.data 1 0
    | 0, 1 => A.data 0 0 * B.data 0 1 + A.data 0 1 * B.data 1 1
    | 1, 0 => A.data 1 0 * B.data 0 0 + A.data 1 1 * B.data 1 0
    | 1, 1 => A.data 1 0 * B.data 0 1 + A.data 1 1 * B.data 1 1
    | _, _ => 0⟩

-- Define the specific matrix M
def M : Matrix 2 2 := ⟨λ i j => 
  match i, j with 
  | 0, 0 => 2 
  | 0, 1 => 0 
  | 1, 0 => 0 
  | 1, 1 => 3 
  | _, _ => 0⟩

-- The desired property
theorem matrix_transformation_property (A : Matrix 2 2) : 
  matrix_mul M A = ⟨λ i j => 
  match i, j with
  | 0, 0 => 2 * A.data 0 0
  | 0, 1 => 2 * A.data 0 1
  | 1, 0 => 3 * A.data 1 0
  | 1, 1 => 3 * A.data 1 1
  | _, _ => 0⟩ := sorry

end matrix_transformation_property_l501_501935


namespace pow_mod_remainder_l501_501090

theorem pow_mod_remainder (n : ℕ) (h : 9 ≡ 2 [MOD 7]) (h2 : 9^2 ≡ 4 [MOD 7]) (h3 : 9^3 ≡ 1 [MOD 7]) : 9^123 % 7 = 1 := by
  sorry

end pow_mod_remainder_l501_501090


namespace shortest_path_proof_l501_501143

-- Define the conditions for the problem
def point_A : ℝ × ℝ := (-3, 9)
def center_C : ℝ × ℝ := (2, 3)
def radius_C : ℝ := 1

-- Define the reflection point based on the given conditions
def reflection_C' : ℝ × ℝ := (2, -3)

-- Define the distance formula between two points in the plane
def dist (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Define the shortest path length
def shortest_path_length : ℝ :=
  dist point_A reflection_C' - radius_C

-- The theorem to prove that the shortest path length is 12
theorem shortest_path_proof : shortest_path_length = 12 :=
by
  sorry

end shortest_path_proof_l501_501143


namespace highest_power_of_2_divides_17_4_minus_15_4_l501_501538

/-- Define the function to calculate the highest power of 2 that divides a number -/
def v2 : ℕ → ℕ
| 0       := 0
| (n + 1) := (Nat.find (fun k => (n + 1) % 2^k ≠ 0)) - 1

theorem highest_power_of_2_divides_17_4_minus_15_4 :
  v2 (17^4 - 15^4) = 7 := by
  sorry

end highest_power_of_2_divides_17_4_minus_15_4_l501_501538


namespace trigonometric_identity_simplification_l501_501835

theorem trigonometric_identity_simplification (α : ℝ):
  (1 - 2 * sin α ^ 2) / (2 * tan (5 * π / 4 + α) * cos (π / 4 + α) ^ 2) - tan α + sin (π / 2 + α) - cos (α - π / 2)
  = 2 * sqrt 2 * cos (π / 4 + α) * cos (α / 2) ^ 2 / cos α :=
by
  sorry

end trigonometric_identity_simplification_l501_501835


namespace minimum_ab_l501_501579

theorem minimum_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ab = a + 4 * b + 5) : ab ≥ 25 :=
sorry

end minimum_ab_l501_501579


namespace problem_statement_l501_501517

def nabla (a b : ℕ) : ℕ := 3 + b ^ a

theorem problem_statement : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end problem_statement_l501_501517


namespace collinear_MNP_l501_501661

theorem collinear_MNP (A B C M N M1 M2 N1 N2 P : Type*)
  [InAngle A B C M] [InAngle A B C N]
  (proj_M1 : Projection M1 M AB)
  (proj_M2 : Projection M2 M AC)
  (proj_N1 : Projection N1 N AB)
  (proj_N2 : Projection N2 N AC)
  (angle_eq : ∠ M A B = ∠ N A C)
  (P_eq : P = intersection (line M1 N2) (line N1 M2)) :
  Collinear (Set.Points M N P) :=
sorry

end collinear_MNP_l501_501661


namespace remainder_of_76_pow_k_mod_7_is_6_l501_501816

theorem remainder_of_76_pow_k_mod_7_is_6 (k : ℕ) (hk : k % 2 = 1) : (76 ^ k) % 7 = 6 :=
sorry

end remainder_of_76_pow_k_mod_7_is_6_l501_501816


namespace number_of_terms_geometric_sequence_l501_501321

def geometric_sequence_property (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (a 1 + a n = 82) ∧
  (a 3 * a (n - 2) = 81) ∧
  ((finset.range n).sum (λ i, a (i + 1)) = 121)

theorem number_of_terms_geometric_sequence :
  ∀ (a : ℕ → ℝ) (n : ℕ), geometric_sequence_property a n → n = 5 :=
by
  sorry

end number_of_terms_geometric_sequence_l501_501321


namespace min_value_a4b3c2_l501_501676

noncomputable def a (x : ℝ) : ℝ := if x > 0 then x else 0
noncomputable def b (x : ℝ) : ℝ := if x > 0 then x else 0
noncomputable def c (x : ℝ) : ℝ := if x > 0 then x else 0

theorem min_value_a4b3c2 (a b c : ℝ) 
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h : 1/a + 1/b + 1/c = 9) : a^4 * b^3 * c^2 ≥ 1/1152 :=
by sorry

example : ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ (1/a + 1/b + 1/c = 9) ∧ a^4 * b^3 * c^2 = 1/1152 :=
by 
  use [1/4, 1/3, 1/2]
  split
  norm_num -- 0 < 1/4
  split
  norm_num -- 0 < 1/3
  split
  norm_num -- 0 < 1/2
  split
  norm_num -- 1/(1/4) + 1/(1/3) + 1/(1/2) = 9
  norm_num -- (1/4)^4 * (1/3)^3 * (1/2)^2 = 1/1152

end min_value_a4b3c2_l501_501676


namespace parallel_iff_perpendicular_iff_l501_501268

open Classical

-- Define vectors and operations
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

def add2b (x : ℝ) : ℝ × ℝ := (a.1 + 2 * b(x).1, a.2 + 2 * b(x).2)
def sub_b (x : ℝ) : ℝ × ℝ := (2 * a.1 - b(x).1, 2 * a.2 - b(x).2)

-- Define parallelism condition: (1+2x, 4) is parallel to (2-x, 3)
def parallel_condition (x : ℝ) : Prop :=
  3 * (a.1 + 2 * b(x).1) = 4 * (2 * a.1 - b(x).1)

-- Define perpendicularity condition: (1+2x, 4) is perpendicular to (2-x, 3)
def perpendicular_condition (x : ℝ) : Prop :=
  (a.1 + 2 * b(x).1) * (2 * a.1 - b(x).1) + a.2 * (2 * a.2 - b(x).2) = 0

-- Prove that if vectors are parallel, then x = 1/2
theorem parallel_iff (x : ℝ) : parallel_condition x ↔ x = 1 / 2 := by
  sorry

-- Prove that if vectors are perpendicular, then x = 7/2 or x = -2
theorem perpendicular_iff (x : ℝ) : perpendicular_condition x ↔ (x = 7 / 2 ∨ x = -2) := by
  sorry

end parallel_iff_perpendicular_iff_l501_501268


namespace simplified_expression_correct_l501_501385

def simplify_expression (a b : ℝ) (h1 : b ≠ 0) (h2 : a ≥ 0) (h3 : b^(2/3) - a^(1/6) * b^(1/3) + a^(1/3) ≠ 0) : ℝ :=
  (a^(1/2) + a * b^(-1)) / (a^(-1/3) - a^(-1/6) * b^(-1/3) + b^(-2/3)) - a / b^(1/3)

theorem simplified_expression_correct (a b : ℝ) (h1 : b ≠ 0) (h2 : a ≥ 0) (h3 : b^(2/3) - a^(1/6) * b^(1/3) + a^(1/3) ≠ 0) : 
  simplify_expression a b h1 h2 h3 = a^(5/6) :=
by
  sorry

end simplified_expression_correct_l501_501385


namespace four_points_on_circle_l501_501394

open EuclideanGeometry

/-- 
Prove that for a quadrilateral ABCD with perpendicular diagonals AC and BD intersecting at E, 
the four points where perpendiculars from E to the sides of ABCD meet the opposite sides 
lie on a circle, and the center of this circle lies on the line segment joining the midpoints 
of AC and BD.
-/
theorem four_points_on_circle (A B C D E P Q R S : Point) (hAB : Line A B) (hBC : Line B C) (hCD : Line C D) (hDA : Line D A)
(hAC : Line A C) (hBD : Line B D)
(hE : E = intersection_point hAC hBD)
(h_perp_AC_BD : Perpendicular hAC hBD)
(hP : Perpendicular_Foot E hAB P)
(hQ : Perpendicular_Foot E hBC Q)
(hR : Perpendicular_Foot E hCD R)
(hS : Perpendicular_Foot E hDA S)
(hPQ : Foot_Opposite_Perpendicular P Q E hBC)
(hQR : Foot_Opposite_Perpendicular Q R E hCD)
(hRS : Foot_Opposite_Perpendicular R S E hDA)
(hSP : Foot_Opposite_Perpendicular S P E hAB)
(mid_AC : Point) (mid_BD : Point)
(hM1: Midpoint mid_AC A C)
(hM2: Midpoint mid_BD B D)
(h_center: Circle_center P Q R S mid_AC mid_BD) : 
Collinear mid_AC mid_BD Circle.center :=
sorry

end four_points_on_circle_l501_501394


namespace minimum_common_difference_l501_501024

noncomputable def quadratic_discriminant (a b c : ℤ) := 
  b^2 - 4 * a * c

theorem minimum_common_difference
  (a b c : ℤ)
  (h_nonzero : ∀ x ∈ {a, b, c}, x ≠ 0)
  (h_arith_prog : ∃ d : ℤ, b = a + d ∧ c = a + 2 * d)
  (h_discriminants : (
    quadratic_discriminant a b (2 * c) > 0 ∧ 
    quadratic_discriminant a (2 * c) b > 0 ∧ 
    quadratic_discriminant b a (2 * c) > 0 ∧ 
    quadratic_discriminant b (2 * c) a > 0 ∧ 
    quadratic_discriminant (2 * c) a b > 0 ∧ 
    quadratic_discriminant (2 * c) b a > 0)):
  4 = min {d | ∃ a b c : ℤ, b = a + d ∧ c = a + 2 * d ∧ 
    (quadratic_discriminant a b (2 * c) > 0 ∧ 
     quadratic_discriminant a (2 * c) b > 0 ∧ 
     quadratic_discriminant b a (2 * c) > 0 ∧ 
     quadratic_discriminant b (2 * c) a > 0 ∧ 
     quadratic_discriminant (2 * c) a b > 0 ∧ 
     quadratic_discriminant (2 * c) b a > 0) ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0} ∧
    ∃ a b c : ℤ, 
      b = a + 4 ∧ c = a + 8 ∧ 
      a = -5 ∧ b = -1 ∧ c = 3 :=
begin
  sorry
end

end minimum_common_difference_l501_501024


namespace no_rational_roots_l501_501687

-- Define the polynomial f with coefficients a_i
noncomputable def polynomial (n : ℕ) (a : ℕ → ℤ) : ℤ[X] :=
  ∑ i in finset.range (n + 1), (a i) * X^(n - i)

-- Define the conditions: coefficients a_0 and a_n, and f(1)
def conditions (n : ℕ) (a : ℕ → ℤ) : Prop :=
  odd (a 0 ∧ odd (a n ∧ odd (polynomial n a).eval 1))

-- Prove that f(x) = 0 has no rational roots under the given conditions
theorem no_rational_roots (n : ℕ) (a : ℕ → ℤ) (h : conditions n a) :
  ¬ ∃ p q : ℤ, q ≠ 0 ∧ gcd p q = 1 ∧ (polynomial n a).eval (p / q) = 0 :=
sorry

end no_rational_roots_l501_501687


namespace intersection_P_Q_l501_501605

def P : Set ℝ := {x | Real.log x / Real.log 2 < -1}
def Q : Set ℝ := {x | abs x < 1}

theorem intersection_P_Q : P ∩ Q = {x | 0 < x ∧ x < 1 / 2} := by
  sorry

end intersection_P_Q_l501_501605


namespace complex_number_in_third_quadrant_l501_501013

theorem complex_number_in_third_quadrant (i : ℂ) (hi : i = complex.I) : 
  let z := (1 - i) / i in (z.re < 0 ∧ z.im < 0) := 
by 
  let z := (1 - i) / i
  have h1 : z = -1 - i := sorry -- Here we assume we have proven this simplification
  have h2 : z.re = -1 := by rw [h1, complex.re_add, complex.re_neg, complex.re_of_real, complex.re_of_real]
  have h3 : z.im = -1 := by rw [h1, complex.im_add, complex.im_neg, complex.im_of_real, complex.im_of_real]
  exact ⟨by linarith, by linarith⟩

end complex_number_in_third_quadrant_l501_501013


namespace probability_conditional_independence_l501_501425

theorem probability_conditional_independence :
  let cards := {1, 2, 3, 4, 5, 6}
  let event_A := λ (x y : Nat), x = 3
  let event_B := λ (x y : Nat), y = 2
  let event_C := λ (x y : Nat), x + y = 6
  let event_D := λ (x y : Nat), x + y = 7
  let probability := (· : Set (Nat × Nat)) : Float
  P(event_A|event_D) = P(event_A) :=
sorry

end probability_conditional_independence_l501_501425


namespace winning_candidate_votes_l501_501076

def total_votes := 35533
def percent_of_winning_candidate := 0.4577952755905512
def votes_of_candidate_2 := 7636
def votes_of_candidate_3 := 11628

theorem winning_candidate_votes :
  ∃ W : ℕ, W = 16270 ∧
  total_votes = votes_of_candidate_2 + votes_of_candidate_3 + W ∧
  W = percent_of_winning_candidate * total_votes :=
by
  sorry

end winning_candidate_votes_l501_501076


namespace distance_between_A_and_B_l501_501807

def point := (ℝ × ℝ)

def distance (A B : point) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

def point_A : point := (-3, 1)
def point_B : point := (4, -9)

theorem distance_between_A_and_B : distance point_A point_B = real.sqrt 149 := 
by sorry

end distance_between_A_and_B_l501_501807


namespace range_of_q_3_elements_l501_501848

def A (q : ℝ) := {x : ℝ | x^2 - 4 * x + q + 3 ≤ 0}

theorem range_of_q_3_elements (q : ℝ) :
  3 = (A q ∩ Set.Ico 0 (2 + real.sqrt (1 - q))).count.toNat →
  (-3 < q ∧ q ≤ 0) :=
by
  sorry

end range_of_q_3_elements_l501_501848


namespace min_value_f_l501_501543

noncomputable def f (a x : ℝ) : ℝ := x ^ 2 - 2 * a * x - 1

theorem min_value_f (a : ℝ) : 
  (∀ x ∈ (Set.Icc (-1 : ℝ) 1), f a x ≥ 
    if a < -1 then 2 * a 
    else if -1 ≤ a ∧ a ≤ 1 then -1 - a ^ 2 
    else -2 * a) := 
by
  sorry

end min_value_f_l501_501543


namespace remaining_integers_count_l501_501069

open Finset

theorem remaining_integers_count :
  let T := range 101 \ {x | x % 4 = 0 ∧ x ≤ 100} ∪ {x | x % 5 = 0 ∧ x % 4 ≠ 0 ∧ x ≤ 100}
  in T.card = 60 :=
by
  let T := range 101 \ {x | x % 4 = 0 ∧ x ≤ 100} ∪ {x | x % 5 = 0 ∧ x % 4 ≠ 0 ∧ x ≤ 100}
  show T.card = 60
  sorry

end remaining_integers_count_l501_501069


namespace doubled_cost_percentage_l501_501442

variable {t b : ℕ}

theorem doubled_cost_percentage (h : t * b^4) :
  let C := t * b^4 in
  let C_new := t * (2 * b)^4 in
  (C_new / C) * 100 = 1600 :=
by
  sorry

end doubled_cost_percentage_l501_501442


namespace line_passes_fixed_point_max_distance_eqn_l501_501261

-- Definition of the line equation
def line_eq (a b x y : ℝ) : Prop :=
  (2 * a + b) * x + (a + b) * y + a - b = 0

-- Point P
def point_P : ℝ × ℝ :=
  (3, 4)

-- Fixed point that the line passes through
def fixed_point : ℝ × ℝ :=
  (-2, 3)

-- Statement that the line passes through the fixed point
theorem line_passes_fixed_point (a b : ℝ) :
  line_eq a b (-2) 3 :=
sorry

-- Equation of the line when distance from point P to line is maximized
def line_max_distance (a b : ℝ) : Prop :=
  5 * 3 + 4 + 7 = 0

-- Statement that the equation of the line is as given when distance is maximized
theorem max_distance_eqn (a b : ℝ) :
  line_max_distance a b :=
sorry

end line_passes_fixed_point_max_distance_eqn_l501_501261


namespace min_value_a4b3c2_l501_501674

theorem min_value_a4b3c2 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : 1/a + 1/b + 1/c = 9) : a^4 * b^3 * c^2 ≥ 1/1152 := 
sorry

end min_value_a4b3c2_l501_501674


namespace original_number_without_10s_digit_l501_501477

theorem original_number_without_10s_digit (h : ℕ) (n : ℕ) 
  (h_eq_1 : h = 1) 
  (n_eq : n = 2 * 1000 + h * 100 + 84) 
  (div_by_6: n % 6 = 0) : n = 2184 → 284 = 284 :=
by
  sorry

end original_number_without_10s_digit_l501_501477


namespace fraction_equiv_l501_501973

theorem fraction_equiv (x b c : ℝ) (h₁ : x ≠ -c) (h₂ : x ≠ -3c) : 
  (x + 2 * b) / (x + 3 * c) = (x + b) / (x + c) ↔ b = 2 * c :=
sorry

end fraction_equiv_l501_501973


namespace integral_exp_neg_x_equals_l501_501900

noncomputable def integral_exp_neg_x : ℝ := ∫ x in 0..1, real.exp (-x)

theorem integral_exp_neg_x_equals :
  integral_exp_neg_x = 1 - real.exp (-1) :=
by 
  -- Proof omitted
  sorry

end integral_exp_neg_x_equals_l501_501900


namespace num_k_values_lcm_l501_501214

-- Define prime factorizations of given numbers
def nine_pow_nine := 3^18
def twelve_pow_twelve := 2^24 * 3^12
def eighteen_pow_eighteen := 2^18 * 3^36

-- Number of values of k making eighteen_pow_eighteen the LCM of nine_pow_nine, twelve_pow_twelve, and k
def number_of_k_values : ℕ := 
  19 -- Based on calculations from the proof

theorem num_k_values_lcm :
  ∀ (k : ℕ), eighteen_pow_eighteen = Nat.lcm (Nat.lcm nine_pow_nine twelve_pow_twelve) k → ∃ n, n = number_of_k_values :=
  sorry -- Add the proof later

end num_k_values_lcm_l501_501214


namespace solve_log_eq_l501_501548

theorem solve_log_eq (x : ℝ) (h : log 3 (1 + 2 * 3^x) = x + 1) : x = 0 := 
sorry

end solve_log_eq_l501_501548


namespace range_of_m_l501_501355

variable {m : ℝ} {x : ℝ}

def p := (m - 2) / (m - 3) ≤ 2 / 3
def q := ∀ (x : ℝ), ¬ (x ^ 2 - 4 * x + m ^ 2 ≤ 0)

theorem range_of_m (h : (p ∨ q) ∧ ¬ (p ∧ q)) : 
  m ∈ ((Set.Iio (-2)) ∪ (Set.Icc 0 2) ∪ (Set.Ici 3)) := 
begin
  sorry
end

end range_of_m_l501_501355


namespace minimum_clients_l501_501451

theorem minimum_clients :
  ∃ (n : ℕ), (∀ m, (repunit m > 10) ∧ (repunit (m * n) = repunit m * 101)) ∧ n = 101 := by
  -- repunit is a helper function to generate repunit numbers
  def repunit (k : ℕ) : ℕ := (dec_trivial : ℕ)
  sorry

end minimum_clients_l501_501451


namespace smallest_unlucky_positions_l501_501642

/-- 
In the cells of an 8 × 8 board, the numbers 1 and -1 are placed (each cell contains exactly one number).
A position of a shape □ on the board is called unlucky if the sum of the numbers in the four cells 
of the shape is not equal to 0. Prove that the smallest possible number of unlucky positions is 36.
/--
theorem smallest_unlucky_positions :
  ∀ (board : ℕ × ℕ → ℤ), 
    (∀ i j, i < 8 ∧ j < 8 → (board (i, j) = 1 ∨ board (i, j) = -1)) →
    (∃ (unlucky_positions : ℕ), unlucky_positions = 36 ∧
      ∀ (shape_pos : ℕ × ℕ), shape_pos.1 < 7 ∧ shape_pos.2 < 7 →
      let sum := board (shape_pos.1, shape_pos.2) + board (shape_pos.1 + 1, shape_pos.2) +
                 board (shape_pos.1, shape_pos.2 + 1) + board (shape_pos.1 + 1, shape_pos.2 + 1) in
      sum ≠ 0) :=
sorry

end smallest_unlucky_positions_l501_501642


namespace correct_operation_l501_501831

variable (a b : ℝ)

theorem correct_operation : (-a * b^2)^2 = a^2 * b^4 :=
  sorry

end correct_operation_l501_501831


namespace feed_can_supply_5_ducks_for_21_days_l501_501461

-- Define the variables
variables {x y : ℝ} -- x: feed consumption per duck per day, y: feed consumption per chicken per day

-- Define conditions from the problem
def condition1 := (10 * x + 15 * y) * 6
def condition2 := (12 * x + 6 * y) * 7

-- Define the relationship between x and y (derived from conditions)
theorem feed_can_supply_5_ducks_for_21_days (h : condition1 = condition2) : 
  ∃ D : ℕ, D = 5 :=
by
  sorry

end feed_can_supply_5_ducks_for_21_days_l501_501461


namespace simplify_fraction_l501_501894

-- Define factorial (or use the existing factorial definition if available in Mathlib)
def fact : ℕ → ℕ 
| 0       => 1
| (n + 1) => (n + 1) * fact n

-- Problem statement
theorem simplify_fraction :
  (5 * fact 7 + 35 * fact 6) / fact 8 = 5 / 4 := by
  sorry

end simplify_fraction_l501_501894


namespace jennie_speed_difference_l501_501327

noncomputable def average_speed_difference : ℝ :=
  let distance := 200
  let time_heavy_traffic := 5
  let construction_delay := 0.5
  let rest_stops_heavy := 0.5
  let time_no_traffic := 4
  let rest_stops_no_traffic := 1 / 3
  let actual_driving_time_heavy := time_heavy_traffic - construction_delay - rest_stops_heavy
  let actual_driving_time_no := time_no_traffic - rest_stops_no_traffic
  let average_speed_heavy := distance / actual_driving_time_heavy
  let average_speed_no := distance / actual_driving_time_no
  average_speed_no - average_speed_heavy

theorem jennie_speed_difference :
  average_speed_difference = 4.5 :=
sorry

end jennie_speed_difference_l501_501327


namespace positive_integer_solution_l501_501217

theorem positive_integer_solution (x : ℕ) (h_pos : 0 < x) :
  (x + 3) / 2 - (2 * x - 1) / 3 > 1 → x ∈ {1, 2, 3, 4} :=
sorry

end positive_integer_solution_l501_501217


namespace sum_of_triangles_l501_501054

def triangle (a b c : ℤ) : ℤ := a + b - c

theorem sum_of_triangles : triangle 1 3 4 + triangle 2 5 6 = 1 := by
  sorry

end sum_of_triangles_l501_501054


namespace repeated_digit_percentage_l501_501283

/-
  Lean statement for the math proof problem:
  Prove that the percentage \( y \) of three-digit numbers that have a repeated digit is \( 28.0 \%\).
-/

theorem repeated_digit_percentage : 
  let total_numbers := 900
  let no_repeated_digits := 9 * 9 * 8
  let repeated_digits := total_numbers - no_repeated_digits
  let y := ((repeated_digits : ℝ) / total_numbers) * 100
  (Float.round y 1) = 28.0 := 
by
  sorry

end repeated_digit_percentage_l501_501283


namespace trig_relation_l501_501008

theorem trig_relation (a b c : ℝ) 
  (h1 : a = Real.sin 2) 
  (h2 : b = Real.cos 2) 
  (h3 : c = Real.tan 2) : c < b ∧ b < a := 
by
  sorry

end trig_relation_l501_501008


namespace quadratic_has_at_most_two_solutions_l501_501040

theorem quadratic_has_at_most_two_solutions (a b c : ℝ) (h : a ≠ 0) :
  ¬(∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧
    a * x1^2 + b * x1 + c = 0 ∧ 
    a * x2^2 + b * x2 + c = 0 ∧ 
    a * x3^2 + b * x3 + c = 0) := 
by {
  sorry
}

end quadratic_has_at_most_two_solutions_l501_501040


namespace equal_areas_of_constructed_quadrilaterals_l501_501297

theorem equal_areas_of_constructed_quadrilaterals (a b c d : ℝ) (H : convex_quadrilateral a b c d):
  let P Q R S : point := divide_sides a b c d (1 : ℝ, sqrt 2 : ℝ, 1 : ℝ) in
  area (construct_quadrilateral P Q R S) = area (quadrilateral a b c d) :=
by sorry

end equal_areas_of_constructed_quadrilaterals_l501_501297


namespace zach_cookies_l501_501102

theorem zach_cookies : 
  let monday := 32
  let tuesday := monday / 2
  let wednesday := (tuesday * 3) - 4
  let thursday := (monday * 2) - 10
  let friday := wednesday - 6
  let saturday := monday + friday
  let total_baked := monday + tuesday + wednesday + thursday + friday + saturday
  let total_eaten := (2 * 6) + 8
  in (total_baked - total_eaten) = 242 := 
by
  sorry

end zach_cookies_l501_501102


namespace part1_part2_l501_501980

noncomputable theory

def f (a x : ℝ) : ℝ := |2 * x - 1| + |a * x - 5|
def condition_a (a : ℝ) : Prop := 0 < a ∧ a < 5

theorem part1 (x : ℝ) : 
  (f 1 x ≥ 9 ↔ (x ≤ -1 ∨ x > 5)) :=
sorry

theorem part2 (a : ℝ) (h: condition_a a) :
  (∀ x, f a x ≥ 4) → a = 2 :=
sorry

end part1_part2_l501_501980


namespace simplify_and_evaluate_expression_l501_501049

-- Define the conditions
def a := 2
def b := -1

-- State the theorem
theorem simplify_and_evaluate_expression : 
  ((2 * a + 3 * b) * (2 * a - 3 * b) - (2 * a - b) ^ 2 - 3 * a * b) / (-b) = -12 := by
  -- Placeholder for the proof
  sorry

end simplify_and_evaluate_expression_l501_501049


namespace grandfather_common_child_l501_501889

theorem grandfather_common_child (kids : Finset ℕ) (grandfather_relation : ∀ (x y : ℕ), x ≠ y → ∃ g : ℕ, has_grandfather g x ∧ has_grandfather g y)
  (h_num_kids : kids.card = 10) : ∃ g : ℕ, kids.filter (has_grandfather g) > 7 :=
begin
  sorry
end

end grandfather_common_child_l501_501889


namespace trisect_54_degree_angle_l501_501568

theorem trisect_54_degree_angle :
  ∃ (a1 a2 : ℝ), a1 = 18 ∧ a2 = 36 ∧ a1 + a2 + a2 = 54 :=
by sorry

end trisect_54_degree_angle_l501_501568


namespace min_students_wearing_both_l501_501302

theorem min_students_wearing_both (n : ℕ) (h_n: n = 63) (n_g n_b y : ℕ)
  (h_glasses : n_g = 3 * n / 7)
  (h_shirts : n_b = 4 * n / 9)
  (h_min_y: y = max 0 (n_g + n_b - n)) :
  y = 8 :=
by
  rw [h_n, h_glasses, h_shirts, nat.mul_div_cancel_left, nat.mul_div_cancel_left],
  -- Show that the given conditions lead to the conclusion y = 8
  sorry

end min_students_wearing_both_l501_501302


namespace least_k_divisible_by_240_l501_501622

theorem least_k_divisible_by_240 : ∃ (k : ℕ), k^2 % 240 = 0 ∧ k = 60 :=
by
  sorry

end least_k_divisible_by_240_l501_501622


namespace quadratic_value_at_5_l501_501400

-- Define the conditions provided in the problem
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Create a theorem that states that if a quadratic with given conditions has its vertex at (2, 7) and passes through (0, -7), then passing through (5, n) means n = -24.5
theorem quadratic_value_at_5 (a b c n : ℝ)
  (h1 : quadratic a b c 2 = 7)
  (h2 : quadratic a b c 0 = -7)
  (h3 : quadratic a b c 5 = n) :
  n = -24.5 :=
by
  sorry

end quadratic_value_at_5_l501_501400


namespace jaden_toy_cars_l501_501655

variable (initial_cars bought_cars birthday_cars gave_sister gave_vinnie : ℕ)

theorem jaden_toy_cars :
  let final_cars := initial_cars + bought_cars + birthday_cars - gave_sister - gave_vinnie in
  initial_cars = 14 → bought_cars = 28 → birthday_cars = 12 → gave_sister = 8 → gave_vinnie = 3 →
  final_cars = 43 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end jaden_toy_cars_l501_501655


namespace smallest_area_cross_section_of_cube_l501_501197

theorem smallest_area_cross_section_of_cube {a : ℝ} (h : a > 0) :
  let d1 := a * Real.sqrt 3,
      d2 := a * Real.sqrt 2 in
  (1 / 2) * d1 * d2 = (a^2 * Real.sqrt 6) / 2 := by
  sorry

end smallest_area_cross_section_of_cube_l501_501197


namespace hawkeye_remaining_balance_l501_501270

theorem hawkeye_remaining_balance
  (cost_per_charge : ℝ) (number_of_charges : ℕ) (initial_budget : ℝ) : 
  cost_per_charge = 3.5 → number_of_charges = 4 → initial_budget = 20 → 
  initial_budget - (number_of_charges * cost_per_charge) = 6 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end hawkeye_remaining_balance_l501_501270


namespace Mary_current_age_l501_501458

theorem Mary_current_age
  (M J : ℕ) 
  (h1 : J - 5 = (M - 5) + 7) 
  (h2 : J + 5 = 2 * (M + 5)) : 
  M = 2 :=
by
  /- We need to show that the current age of Mary (M) is 2
     given the conditions h1 and h2.-/
  sorry

end Mary_current_age_l501_501458


namespace symmetry_center_of_transformed_function_l501_501387

theorem symmetry_center_of_transformed_function :
  let g (x : ℝ) := 4 * sin (2 * x - (π / 6)) in g (π / 12) = 0 :=
by
  sorry

end symmetry_center_of_transformed_function_l501_501387


namespace decimal_place_250_of_13_over_17_is_8_l501_501822

theorem decimal_place_250_of_13_over_17_is_8 :
  let repeating_sequence := '7647058823529411'
  let n := 250
  let repetition_length := 16
  let position_in_repetition := (n % repetition_length)
  position_in_repetition = 10 → 
  repeating_sequence[10] = 8 := 
by
  sorry

end decimal_place_250_of_13_over_17_is_8_l501_501822


namespace distance_parallel_lines_distance_point_line_l501_501265

def line1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y + 1 = 0
def point : ℝ × ℝ := (0, 2)

noncomputable def distance_between_lines (A B C1 C2 : ℝ) : ℝ :=
  |C2 - C1| / Real.sqrt (A^2 + B^2)

noncomputable def distance_point_to_line (A B C x0 y0 : ℝ) : ℝ :=
  |A * x0 + B * y0 + C| / Real.sqrt (A^2 + B^2)

theorem distance_parallel_lines : distance_between_lines 2 1 (-1) 1 = (2 * Real.sqrt 5) / 5 := by
  sorry

theorem distance_point_line : distance_point_to_line 2 1 (-1) 0 2 = (Real.sqrt 5) / 5 := by
  sorry

end distance_parallel_lines_distance_point_line_l501_501265


namespace singer_arrangements_l501_501828

theorem singer_arrangements (s1 s2 : Type) [Fintype s1] [Fintype s2] 
  (h1 : Fintype.card s1 = 4) (h2 : Fintype.card s2 = 1) :
  ∃ n : ℕ, n = 18 :=
by
  sorry

end singer_arrangements_l501_501828


namespace number_of_possible_values_l501_501859

-- Define the problem conditions and goal
theorem number_of_possible_values {n : ℕ} (h_non_degenerate : 
    ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b > c ∧ a + c > b ∧ b + c > a) :
    (∃ n_values : ℕ, n_values = 203797 ∧ 
    ∀ n : ℕ, 21 ≤ n ∧ n ≤ 203817 → 
    (log 101 + log 2018 > log n) ∧ 
    (log 101 + log n > log 2018) ∧ 
    (log 2018 + log n > log 101)) := 
sorry

end number_of_possible_values_l501_501859


namespace rectangle_z_value_l501_501144

theorem rectangle_z_value (z : ℝ) 
  (h_pos : 0 < z) 
  (h_area : let length := 6 - (-2) in
            let height := z - 4 in
            length * height = 64) : 
  z = 12 := 
sorry

end rectangle_z_value_l501_501144


namespace projection_AB_onto_AC_l501_501557

variables (AB AC : ℝ × ℝ)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def projection (v w : ℝ × ℝ) : ℝ :=
  dot_product v w / magnitude w

theorem projection_AB_onto_AC :
  ∀ (AB AC : ℝ × ℝ), AB = (1, -1) → AC = (2, 0) → projection AB AC = 1 :=
by
  intros AB AC hAB hAC
  rw [hAB, hAC]
  -- automatically simplify the expression to the correct answer
  sorry

end projection_AB_onto_AC_l501_501557


namespace range_of_m_l501_501583

theorem range_of_m (x y : ℝ) (hxy : 0 < x ∧ 0 < y) (h : (2 * x - y / Real.exp 1) * Real.log (y / x) ≤ x / (m * Real.exp 1)) :
  m ∈ Ioc 0 (1 / Real.exp 1) :=
sorry

end range_of_m_l501_501583


namespace minimum_clients_repunits_l501_501448

theorem minimum_clients_repunits (n m k : ℕ) (h_n : n > 1) (h_banks : ∀ x, x > 10)
  (amt_per_client : ℕ) (h_amt_repunits : ∀ i, amt_per_client = (nat.replicate i 1).foldr (λ a b, 10 * b + a) 0 ∧ amt_per_client > 10) 
  (total_amount : ℕ) (h_total_repunits : total_amount = (nat.replicate n 1).foldr (λ a b, 10 * b + a) 0) 
  (h_eq : total_amount = amt_per_client * m) 
  (h_cond : ∃ k, k > 1 ∧ k < n ∧ total_amount = amt_per_client * (nat.replicate k 1).foldr (λ a b, 10 * b + a) 0) : 
  n = 101 :=
sorry

end minimum_clients_repunits_l501_501448


namespace max_blocks_fit_l501_501810

def block : Type :=
  { length: ℝ, width: ℝ, height: ℝ }

def box : Type :=
  { length: ℝ, width: ℝ, height: ℝ }

noncomputable def blocks_fit (blk: block) (bx: box) : ℕ :=
  let num_length := bx.length / blk.length
  let num_width := bx.width / blk.width
  let num_height := bx.height / blk.height
  (num_length * num_width * num_height).to_nat

theorem max_blocks_fit : blocks_fit { length:=20, width:=30, height:=40 } { length:=40, width:=60, height:=80} = 8 :=
    by sorry

end max_blocks_fit_l501_501810


namespace Dana_pencils_equals_combined_l501_501178

-- Definitions based on given conditions
def pencils_Jayden : ℕ := 20
def pencils_Marcus (pencils_Jayden : ℕ) : ℕ := pencils_Jayden / 2
def pencils_Dana (pencils_Jayden : ℕ) : ℕ := pencils_Jayden + 15
def pencils_Ella (pencils_Marcus : ℕ) : ℕ := 3 * pencils_Marcus - 5
def combined_pencils (pencils_Marcus : ℕ) (pencils_Ella : ℕ) : ℕ := pencils_Marcus + pencils_Ella

-- Theorem to prove:
theorem Dana_pencils_equals_combined (pencils_Jayden : ℕ := 20) : 
  pencils_Dana pencils_Jayden = combined_pencils (pencils_Marcus pencils_Jayden) (pencils_Ella (pencils_Marcus pencils_Jayden)) := by
  sorry

end Dana_pencils_equals_combined_l501_501178


namespace hiking_hours_l501_501791

-- Define the given conditions
def water_needed_violet_per_hour : ℕ := 800
def water_needed_dog_per_hour : ℕ := 400
def total_water_carry_capacity_liters : ℕ := 4.8 * 1000 -- converted to ml

-- Define the statement to prove
theorem hiking_hours : (total_water_carry_capacity_liters / (water_needed_violet_per_hour + water_needed_dog_per_hour)) = 4 := by
  sorry

end hiking_hours_l501_501791


namespace lcm_value_count_l501_501204

theorem lcm_value_count (a b : ℕ) (k : ℕ) (h1 : 9^9 = 3^18) (h2 : 12^12 = 2^24 * 3^12) 
  (h3 : 18^18 = 2^18 * 3^36) (h4 : k = 2^a * 3^b) (h5 : 18^18 = Nat.lcm (9^9) (Nat.lcm (12^12) k)) :
  ∃ n : ℕ, n = 25 :=
begin
  sorry
end

end lcm_value_count_l501_501204


namespace ending_time_is_2_30_PM_l501_501401

-- Conditions as defined
def degrees_in_12_hours := 360 
def hours := 12
def degrees_moved := 75

-- Rate at which the hour hand moves
def degrees_per_hour := degrees_in_12_hours / hours

-- Number of hours that have passed
def hours_passed := degrees_moved / degrees_per_hour

-- Initial time in hours since midnight (noon is 12:00 PM)
def initial_time := 12

-- Final time in hours since midnight
def final_time := initial_time + hours_passed

-- The corresponding time in hours and minutes form
def final_time_in_hours_and_minutes : (nat × nat) :=
  let hours := (initial_time + hours_passed).floor
  let minutes := ((initial_time + hours_passed) - hours) * 60
  (hours.toNat, minutes.toNat)

-- Stating the problem: To prove that given the conditions, the ending time is 2:30 PM
theorem ending_time_is_2_30_PM :
  final_time_in_hours_and_minutes = (14, 30) :=
by
  simp [degrees_in_12_hours, hours, degrees_moved, degrees_per_hour, hours_passed, initial_time, final_time_in_hours_and_minutes]
  sorry

end ending_time_is_2_30_PM_l501_501401


namespace sum_of_squares_xy_l501_501081

theorem sum_of_squares_xy (x y : ℝ) (h₁ : x + y = 10) (h₂ : x^3 + y^3 = 370) : x * y = 21 :=
by
  sorry

end sum_of_squares_xy_l501_501081


namespace correct_operation_l501_501830

theorem correct_operation :
  ¬(a^2 * a^3 = a^6) ∧ ¬(6 * a / (3 * a) = 2 * a) ∧ ¬(2 * a^2 + 3 * a^3 = 5 * a^5) ∧ (-a * b^2)^2 = a^2 * b^4 :=
by
  sorry

end correct_operation_l501_501830


namespace renovation_total_cost_l501_501658

theorem renovation_total_cost 
  (charge_first_pro: ℕ)
  (charge_second_pro: ℕ)
  (hours_per_day: ℕ)
  (days: ℕ)
  (materials_cost: ℕ)
  (plumbing_cost: ℕ) :
  charge_first_pro = 15 →
  charge_second_pro = 20 →
  hours_per_day = 6 →
  days = 7 →
  materials_cost = 1500 →
  plumbing_cost = 500 →
  (charge_first_pro * hours_per_day * days + 
   charge_second_pro * hours_per_day * days + 
   materials_cost + plumbing_cost) = 3470 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end renovation_total_cost_l501_501658


namespace average_cost_per_hour_is_316_l501_501774

-- Define constants for the base costs and additional hourly rates
def base_cost_compact : ℝ := 12.00
def additional_hourly_compact : ℝ := 1.50
def base_cost_sedan : ℝ := 14.00
def additional_hourly_sedan : ℝ := 1.75
def base_cost_truck : ℝ := 18.00
def additional_hourly_truck : ℝ := 2.25

-- Define durations
def duration_compact : ℝ := 9
def duration_sedan : ℝ := 5
def duration_truck : ℝ := 12

-- Define total costs calculations
def total_cost_compact := base_cost_compact + (duration_compact - 2) * additional_hourly_compact
def total_cost_sedan := base_cost_sedan + (duration_sedan - 2) * additional_hourly_sedan
def total_cost_truck := base_cost_truck + (duration_truck - 2) * additional_hourly_truck

-- Define total hours and total cost for all vehicles
def total_hours : ℝ := duration_compact + duration_sedan + duration_truck
def total_cost : ℝ := total_cost_compact + total_cost_sedan + total_cost_truck

-- Define overall average cost per hour
def overall_average_cost_per_hour : ℝ := total_cost / total_hours

-- The theorem to prove that overall average cost per hour is $3.16/hour
theorem average_cost_per_hour_is_316 : overall_average_cost_per_hour = 3.16 := by
  sorry

end average_cost_per_hour_is_316_l501_501774


namespace binom_19_13_l501_501578

open Nat

theorem binom_19_13 :
  ∀ (binom_1811 binom_1812 binom_2013 : ℕ)
  (h1 : binom_1811 = 31824)
  (h2 : binom_1812 = 18564)
  (h3 : binom_2013 = 77520)
  (pascal : ∀ n k, binomial (n + 1) k = binomial n (k - 1) + binomial n k),
  binomial 19 13 = 58956 :=
by
  sorry

end binom_19_13_l501_501578


namespace constant_polynomial_if_real_not_necessarily_constant_if_complex_l501_501452

-- Part (a)
theorem constant_polynomial_if_real 
  (P Q R : Polynomial ℝ) 
  (h : ∀ z : ℂ, P.eval z * Q.eval (conj z) = R.eval z) :
  (is_constant Q) :=
sorry

-- Part (b)
theorem not_necessarily_constant_if_complex :
  ¬ ∀ (P Q R : Polynomial ℂ), 
    (∀ z : ℂ, P.eval z * Q.eval (conj z) = R.eval z) → is_constant Q :=
sorry

end constant_polynomial_if_real_not_necessarily_constant_if_complex_l501_501452


namespace hyperbola_exists_l501_501235

-- Define the center of the hyperbola
def center (E : Type) : E := (0, 0)

-- Define the focus of the hyperbola
def focus_1 (E : Type) : E := (3, 0)

-- Define the midpoint condition
def midpoint (A B N : Type) : Prop :=
  N = (-12, -15)

-- Define the hyperbola equation
def hyperbola_equation (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the problem statement
theorem hyperbola_exists (E : Type) (A B N : E) :
  center E = (0, 0) ∧ 
  focus_1 E = (3, 0) ∧ 
  midpoint A B N → 
  (∃ a b, a^2 = 4 ∧ b^2 = 5 ∧ hyperbola_equation x y a b) := 
by
  sorry

end hyperbola_exists_l501_501235


namespace hiking_hours_l501_501796

def violet_water_per_hour : ℕ := 800 -- Violet's water need per hour in ml
def dog_water_per_hour : ℕ := 400    -- Dog's water need per hour in ml
def total_water_capacity : ℚ := 4.8  -- Total water capacity Violet can carry in L

theorem hiking_hours :
  let total_water_per_hour := (violet_water_per_hour + dog_water_per_hour) / 1000 in
  total_water_capacity / total_water_per_hour = 4 :=
by
  let total_water_per_hour := (violet_water_per_hour + dog_water_per_hour) / 1000
  have h1 : violet_water_per_hour = 800 := rfl
  have h2 : dog_water_per_hour = 400 := rfl
  have h3 : total_water_capacity = 4.8 := rfl
  have h4 : total_water_per_hour = 1.2 := by simp [violet_water_per_hour, dog_water_per_hour]
  have h5 : total_water_capacity / total_water_per_hour = 4 := by simp [total_water_capacity, total_water_per_hour]
  exact h5

end hiking_hours_l501_501796


namespace empty_shell_probability_l501_501083

def placeA_walnut_probability : ℝ := 0.40
def placeB_walnut_probability : ℝ := 0.60
def empty_shell_given_placeA : ℝ := 0.02
def empty_shell_given_placeB : ℝ := 0.04

theorem empty_shell_probability : 
  (placeA_walnut_probability * empty_shell_given_placeA) + 
  (placeB_walnut_probability * empty_shell_given_placeB) = 0.032 :=
begin
  sorry
end

end empty_shell_probability_l501_501083


namespace sheila_weekly_earnings_l501_501047

theorem sheila_weekly_earnings :
  let wage_per_hour : ℝ := 14
  let hours_per_day_mwf : ℝ := 8
  let days_mwf : ℝ := 3
  let hours_per_day_tt : ℝ := 6
  let days_tt : ℝ := 2
  (wage_per_hour * hours_per_day_mwf * days_mwf + wage_per_hour * hours_per_day_tt * days_tt = 504) := 
by
  let wage_per_hour : ℝ := 14
  let hours_per_day_mwf : ℝ := 8
  let days_mwf : ℝ := 3
  let hours_per_day_tt : ℝ := 6
  let days_tt : ℝ := 2
  show 14 * 8 * 3 + 14 * 6 * 2 = 504
  calc
    14 * 8 * 3 + 14 * 6 * 2 = 336 + 168 : by ring
                       ... = 504 : by norm_num

end sheila_weekly_earnings_l501_501047


namespace find_a_l501_501734

theorem find_a (a : ℝ) :
  let p1 := (3, -1) in
  let p2 := (-1, 4) in
  let direction_vector := (p2.1 - p1.1, p2.2 - p1.2) in
  let k := 2 / direction_vector.2 in
  let normalized_vector := (k * direction_vector.1, k * direction_vector.2) in
  normalized_vector = (a, 2) →
  a = -8 / 5 :=
by
  sorry

end find_a_l501_501734


namespace max_length_BP_squared_l501_501348

theorem max_length_BP_squared :
  let r := 12,
      A, B, C, T, P : Point,
      ω : Circle,
      O : Point,
      -- Conditions
      h1 : diameter ω ⟨A, B⟩,
      h2 : collinear [A, B, C],
      h3 : tangent_to_circle ⟨C, T⟩ ω T,
      h4 : foot_of_perpendicular A ⟨C, T⟩ P,
      h5 : length ⟨A, B⟩ = 24,
      -- Define n as the maximum length of segment BP
      n := max_length (length ⟨B, P⟩)
  -- Prove that n² = 580
  in n² = 580 :=
sorry

end max_length_BP_squared_l501_501348


namespace equations_and_velocity_of_point_M_l501_501172

-- Condition Definitions
def omega : ℝ := 10 -- Angular velocity in rad/s
def length_OA : ℝ := 90 -- Length of the crank in cm
def length_AB : ℝ := 90 -- Length of the connecting rod in cm
def length_AM : ℝ := (2/3) * length_AB -- Length AM, which is 60 cm

-- Define the position of the crank as function of time t
def theta (t : ℝ) : ℝ := omega * t

-- Coordinates of point A
def A (t : ℝ) : ℝ × ℝ := (length_OA * Real.cos (theta t), length_OA * Real.sin (theta t))

-- Coordinates of point M
def M (t : ℝ) : ℝ × ℝ := (length_AM * Real.cos (theta t), length_AM * (5/3) * Real.sin (theta t))

-- Trajectory equations
def x (t : ℝ) : ℝ := (length_AM / 3) * Real.cos (theta t)
def y (t : ℝ) : ℝ := (length_AM * 5) * Real.sin (theta t)

-- Velocity components of point M
def vx (t : ℝ) : ℝ := (-length_AM * 10) * Real.sin (theta t)
def vy (t : ℝ) : ℝ := (length_AM * (5/3) * 10) * Real.cos (theta t)

-- Resultant velocity of point M
def velocity_M (t : ℝ) : ℝ := 300 * Real.sqrt (1 + 24 * Real.cos (theta t) ^ 2)

-- The final statement that needs to be proved
theorem equations_and_velocity_of_point_M
  (t : ℝ) :
  x t = (30 : ℝ) * Real.cos (theta t) ∧
  y t = (150 : ℝ) * Real.sin (theta t) ∧
  velocity_M t = 300 * Real.sqrt (1 + 24 * Real.cos (theta t) ^ 2) := by
  sorry

end equations_and_velocity_of_point_M_l501_501172


namespace eva_triplets_divisible_by_6_l501_501928

theorem eva_triplets_divisible_by_6 :
  ∃ (a b c d e f : ℕ), 
  {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 6} ∧
  (∃ (x y z : ℕ), List.perm [a, b, c, d, e, f] [x, y, z] ∧
  x % 6 = 0 ∧ y % 6 = 0 ∧ z % 6 = 0) := by
sorry

end eva_triplets_divisible_by_6_l501_501928


namespace smallest_white_marbles_l501_501487

/-
Let n be the total number of Peter's marbles.
Half of the marbles are orange.
One fifth of the marbles are purple.
Peter has 8 silver marbles.
-/
def total_marbles (n : ℕ) : ℕ :=
  n

def orange_marbles (n : ℕ) : ℕ :=
  n / 2

def purple_marbles (n : ℕ) : ℕ :=
  n / 5

def silver_marbles : ℕ :=
  8

def white_marbles (n : ℕ) : ℕ :=
  n - (orange_marbles n + purple_marbles n + silver_marbles)

-- Prove that the smallest number of white marbles Peter could have is 1.
theorem smallest_white_marbles : ∃ n : ℕ, n % 10 = 0 ∧ white_marbles n = 1 :=
sorry

end smallest_white_marbles_l501_501487


namespace max_value_of_x_l501_501809

theorem max_value_of_x : ∃ x : ℝ, 
  ( (4*x - 16) / (3*x - 4) )^2 + ( (4*x - 16) / (3*x - 4) ) = 18 
  ∧ x = (3 * Real.sqrt 73 + 28) / (11 - Real.sqrt 73) :=
sorry

end max_value_of_x_l501_501809


namespace cubic_root_sum_log_eqn_l501_501216

theorem cubic_root_sum_log_eqn
  (u v w : ℝ) (c d : ℝ)
  (h1 : 9 * u^3 + 5 * c * u^2 + 6 * d * u + c = 0)
  (h2 : 9 * v^3 + 5 * c * v^2 + 6 * d * v + c = 0)
  (h3 : 9 * w^3 + 5 * c * w^2 + 6 * d * w + c = 0)
  (h_uvw_distinct : u ≠ v ∧ u ≠ w ∧ v ≠ w ∧ u > 0 ∧ v > 0 ∧ w > 0)
  (h_log_sum : log 2 u + log 2 v + log 2 w = 6) :
  c = -576 :=
by
  sorry

end cubic_root_sum_log_eqn_l501_501216


namespace solve_system_of_equations_l501_501719

theorem solve_system_of_equations (a b c x y z : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  (a * y + b * x = c) ∧ (c * x + a * z = b) ∧ (b * z + c * y = a) →
  (x = (b^2 + c^2 - a^2) / (2 * b * c)) ∧
  (y = (a^2 + c^2 - b^2) / (2 * a * c)) ∧
  (z = (a^2 + b^2 - c^2) / (2 * a * b)) :=
by
  sorry

end solve_system_of_equations_l501_501719


namespace op_op_1_3_2_2_1_4_3_4_1_l501_501199

-- Definition of the operation ⊕
def op⊕ (x y z : ℝ) (h : y ≠ z) := (y * z) / (y - z)

-- The theorem statement
theorem op_op_1_3_2_2_1_4_3_4_1 : 
    op⊕ (op⊕ 1 3 2 (by norm_num)) (op⊕ 2 1 4 (by norm_num)) (op⊕ 3 4 1 (by norm_num)) (by norm_num) = 4 :=
sorry

end op_op_1_3_2_2_1_4_3_4_1_l501_501199


namespace matrix_characteristic_eq_l501_501332

def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 2, 3], ![3, 1, 2], ![2, 3, 1]]

def I : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.one (Fin 3)

def Z : Matrix (Fin 3) (Fin 3) ℝ :=
  0

theorem matrix_characteristic_eq :
  ∃ p q r : ℝ, p = -3 ∧ q = -9 ∧ r = -2 ∧ (A^3 + p • A^2 + q • A + r • I = Z) :=
by {
  use [-3, -9, -2],
  simp,
  sorry
}

end matrix_characteristic_eq_l501_501332


namespace smaller_root_of_equation_l501_501750

theorem smaller_root_of_equation :
  let a := (3 : ℚ) / 5 in
  let b := (1 : ℚ) / 3 in
  ∃ x : ℚ, x = 19 / 45 ∧ (x - a) ^ 2 + 2 * (x - a) * (x - b) = 0 :=
by
  let a := (3 : ℚ) / 5
  let b := (1 : ℚ) / 3
  use (19 / 45)
  split
  · rfl
  · sorry

end smaller_root_of_equation_l501_501750


namespace batsman_average_after_12th_l501_501120

variable (score_in_12th : ℕ) (average_increase : ℕ) (innings : ℕ)

def average_before_12th (total_runs : ℕ) : ℕ := total_runs / 11

def total_runs_12th (total_runs : ℕ) : ℕ := total_runs + score_in_12th

def average_after_12th (total_runs : ℕ) : ℕ := total_runs_12th total_runs / 12

theorem batsman_average_after_12th
  (score_in_12th : ℕ := 65)
  (average_increase : ℕ := 2)
  (A : ℕ := 41) :
  average_after_12th (11 * A) = A + average_increase + average_increase := 
by
  sorry

end batsman_average_after_12th_l501_501120


namespace quadrilateral_length_CD_l501_501313

variables (AB CD BD BC p q : ℕ)
variables (p q : ℕ)

-- Given conditions
def quadrilateral (A B C D : Type) := true -- placeholder for quadrilateral

axiom angle_BAD_eq_angle_ADC : ∀ (A B C D : Type), quadrilateral A B C D → A = D -- placeholder for equal angles
axiom angle_ABD_eq_angle_BCD : ∀ (A B C D : Type), quadrilateral A B C D → B = C -- placeholder for equal angles

-- Specific lengths
axiom AB_length : AB = 10
axiom BD_length : BD = 15
axiom BC_length : BC = 9

-- Important theorem we want to prove
theorem quadrilateral_length_CD (h1 : angle_BAD_eq_angle_ADC) (h2 : angle_ABD_eq_angle_BCD) (h3 : AB = 10) (h4 : BD = 15) (h5 : BC = 9):
  let CD := 20 in
  let p := 20 in
  let q := 1 in
  (p + q) = 21 := by
  sorry

end quadrilateral_length_CD_l501_501313


namespace correct_factorization_l501_501488

-- Define the expressions involved in the options
def option_A (x a b : ℝ) : Prop := x * (a - b) = a * x - b * x
def option_B (x y : ℝ) : Prop := x^2 - 1 + y^2 = (x - 1) * (x + 1) + y^2
def option_C (x : ℝ) : Prop := x^2 - 1 = (x + 1) * (x - 1)
def option_D (x a b c : ℝ) : Prop := a * x + b * x + c = x * (a + b) + c

-- Theorem stating that option C represents true factorization
theorem correct_factorization (x : ℝ) : option_C x := by
  sorry

end correct_factorization_l501_501488


namespace no_points_with_given_distance_sum_l501_501680

variable {P : Type} [InnerProductSpace ℝ P]
variable (a b : ℝ) (F₁ F₂ : P)

-- Definition of an ellipse with given semi-major and semi-minor axes
def is_ellipse (p : P) :=
  let c := real.sqrt(a^2 - b^2)
  ∃ x y, p = (x, y) ∧ (x/a)^2 + (y/b)^2 ≤ 1

-- Definition of the specific property involving distances to the foci
def sum_of_squares_of_distances_to_foci_is_5 (p : P) :=
  let foci_distance := dist F₁ F₂
  dist p F₁ ^ 2 + dist p F₂ ^ 2 = 5

theorem no_points_with_given_distance_sum :
  ∀ p : P, is_ellipse 2 1 p → ¬sum_of_squares_of_distances_to_foci_is_5 p :=
by
  intro p h_ellipse h_sum_squares
  sorry

end no_points_with_given_distance_sum_l501_501680


namespace alternating_colors_probability_l501_501852

theorem alternating_colors_probability :
  let total_balls : ℕ := 10
  let white_balls : ℕ := 5
  let black_balls : ℕ := 5
  let successful_outcomes : ℕ := 2
  let total_outcomes : ℕ := Nat.choose total_balls white_balls
  (successful_outcomes : ℚ) / (total_outcomes : ℚ) = (1 / 126) := 
by
  let total_balls := 10
  let white_balls := 5
  let black_balls := 5
  let successful_outcomes := 2
  let total_outcomes := Nat.choose total_balls white_balls
  have h_total_outcomes : total_outcomes = 252 := sorry
  have h_probability : (successful_outcomes : ℚ) / (total_outcomes : ℚ) = (1 / 126) := sorry
  exact h_probability

end alternating_colors_probability_l501_501852


namespace pass_each_other_number_l501_501025

-- Definitions and conditions
variable (r_O r_K v_O v_K : ℝ)
variable (T : ℕ)
variable (radial_start_coincidence : Prop)

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
def angular_speed (v : ℝ) (C : ℝ) : ℝ := (v / C) * 2 * Real.pi
def relative_angular_speed (omega_O omega_K : ℝ) : ℝ := omega_O + omega_K
def time_to_meet (relative_speed : ℝ) : ℝ := 2 * Real.pi / relative_speed

-- Theorem statement
theorem pass_each_other_number :
  let C_O := circumference r_O
  let C_K := circumference r_K
  let omega_O := angular_speed v_O C_O
  let omega_K := angular_speed v_K C_K
  let omega_rel := relative_angular_speed omega_O omega_K
  let k := time_to_meet omega_rel
  (radial_start_coincidence ∧ r_O = 40 ∧ r_K = 55 ∧ v_O = 240 ∧ v_K = 320 ∧ T = 40) → 
  (⌊T / k⌋ = 75) :=
begin
  sorry
end

end pass_each_other_number_l501_501025


namespace find_x_square_l501_501427

noncomputable theory

open Real

theorem find_x_square (x : ℝ) (h1 : 0 < x) (h2 : sin (arctan x) = 1 / x) : x^2 = 1 :=
by
  sorry

end find_x_square_l501_501427


namespace num_k_values_lcm_l501_501215

-- Define prime factorizations of given numbers
def nine_pow_nine := 3^18
def twelve_pow_twelve := 2^24 * 3^12
def eighteen_pow_eighteen := 2^18 * 3^36

-- Number of values of k making eighteen_pow_eighteen the LCM of nine_pow_nine, twelve_pow_twelve, and k
def number_of_k_values : ℕ := 
  19 -- Based on calculations from the proof

theorem num_k_values_lcm :
  ∀ (k : ℕ), eighteen_pow_eighteen = Nat.lcm (Nat.lcm nine_pow_nine twelve_pow_twelve) k → ∃ n, n = number_of_k_values :=
  sorry -- Add the proof later

end num_k_values_lcm_l501_501215


namespace mrs_hilt_bees_l501_501023

theorem mrs_hilt_bees (n : ℕ) (h : 3 * n = 432) : n = 144 := by
  sorry

end mrs_hilt_bees_l501_501023


namespace find_M_l501_501492

theorem find_M :
  (∃ M : ℕ, ∀ (M_val = (50 + 7 * M) / (10 + M), M_val = 0.62) → M = 7) :=
begin
  sorry
end

end find_M_l501_501492


namespace basis_options_l501_501667

-- Define the vectors
variables {α : Type*} [add_comm_group α] [module ℝ α]
variables (a b c x y z : α)

-- Define the conditions in the problem
def condition_x : Prop := x = a + b
def condition_y : Prop := y = b + c
def condition_z : Prop := z = c + a

-- Define what it means for a set to be a basis
def is_basis (v1 v2 v3 : α) : Prop :=
  linear_independent ℝ ![v1, v2, v3] ∧ spans ℝ ![v1, v2, v3] (⊤ : submodule ℝ α)

-- The proof problem showing which sets form a basis
theorem basis_options (h_cond_x : condition_x x a b) (h_cond_y : condition_y y b c) (h_cond_z : condition_z z c a) :
  (is_basis x y z) ∧ (is_basis b c z) ∧ (is_basis x y (a + b + c)) :=
by {
  -- Provided the hypotheses and conditions, show each valid set forms a basis
  sorry
}

end basis_options_l501_501667


namespace spider_legs_solution_l501_501874

def single_spider_legs (L : ℕ) : Prop :=
  let num_spiders := L / 2 + 10 in
  let total_legs := num_spiders * L in
  total_legs = 112

theorem spider_legs_solution : ∃ L : ℕ, single_spider_legs L ∧ L = 8 :=
sorry

end spider_legs_solution_l501_501874


namespace ratio_of_distances_l501_501173

noncomputable def cubic_function (a b c : ℝ) : ℝ → ℝ :=
  λ x, x^3 + a*x^2 + b*x + c

theorem ratio_of_distances (a b c x_A x_B x_C x_D : ℝ)
  (f : ℝ → ℝ)
  (H_f : ∀ x, f x = cubic_function a b c x)
  (H_parallel: ∀ x y, x != y → f' x = f' y)
  (H_tangent_B : ∀ x, f' x = f' x_B)
  (H_tangent_C : ∀ x, f' x = f' x_C)
  : (x_A - x_B) / (x_B - x_C) = 1 ∧ (x_B - x_C) / (x_C - x_D) = 2 :=
begin
  sorry
end

end ratio_of_distances_l501_501173


namespace well_depth_calculation_l501_501484

noncomputable def depth_of_well : ℝ :=
sorry 

theorem well_depth_calculation :
  ∃ d : ℝ, (sqrt(d) / sqrt(14) + d / 1200 = 10) ∧
  (d = depth_of_well) ∧
  (depth_of_well = 1538.5 ∨ depth_of_well = 2000 ∨ depth_of_well = 1764 ∨ depth_of_well = 725.9 ∨ depth_of_well = 1642.4) :=
sorry

end well_depth_calculation_l501_501484


namespace percent_of_x_is_65_l501_501619

variable (z y x : ℝ)

theorem percent_of_x_is_65 :
  (0.45 * z = 0.39 * y) → (y = 0.75 * x) → (z / x = 0.65) := by
  sorry

end percent_of_x_is_65_l501_501619


namespace find_remainder_l501_501089

-- Definitions based on given conditions
def dividend := 167
def divisor := 18
def quotient := 9

-- Statement to prove
theorem find_remainder : dividend = (divisor * quotient) + 5 :=
by
  -- Definitions used in the problem
  unfold dividend divisor quotient
  sorry

end find_remainder_l501_501089


namespace number_of_k_values_l501_501210

theorem number_of_k_values :
  let k (a b : ℕ) := 2^a * 3^b in
  (∀ a b : ℕ, 18 ≤ a ∧ b = 36 → 
  let lcm_val := Nat.lcm (Nat.lcm (9^9) (12^12)) (k a b) in 
  lcm_val = 18^18) →
  (Finset.card (Finset.filter (λ a, 18 ≤ a ∧ a ≤ 24) (Finset.range (24 + 1))) = 7) :=
by
  -- proof skipped
  sorry

end number_of_k_values_l501_501210


namespace area_outside_squares_inside_triangle_l501_501509

noncomputable def side_length_large_square : ℝ := 6
noncomputable def side_length_small_square1 : ℝ := 2
noncomputable def side_length_small_square2 : ℝ := 3
noncomputable def area_large_square := side_length_large_square ^ 2
noncomputable def area_small_square1 := side_length_small_square1 ^ 2
noncomputable def area_small_square2 := side_length_small_square2 ^ 2
noncomputable def area_triangle_EFG := area_large_square / 2
noncomputable def total_area_small_squares := area_small_square1 + area_small_square2

theorem area_outside_squares_inside_triangle :
  (area_triangle_EFG - total_area_small_squares) = 5 :=
by
  sorry

end area_outside_squares_inside_triangle_l501_501509


namespace range_of_x_l501_501919

noncomputable def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
noncomputable def specific_function (f : ℝ → ℝ) := ∀ x : ℝ, x ≥ 0 → f x = 2^x

theorem range_of_x (f : ℝ → ℝ)  
  (hf_even : even_function f) 
  (hf_specific : specific_function f) : {x : ℝ | f (1 - 2 * x) < f 3} = {x : ℝ | -1 < x ∧ x < 2} := 
by
  sorry

end range_of_x_l501_501919


namespace find_y_l501_501006

def star (a b : ℝ) : ℝ := 2 * a * b - 3 * b - a

theorem find_y (y : ℝ) (h : star 4 y = 80) : y = 16.8 :=
by
  sorry

end find_y_l501_501006


namespace left_square_side_length_l501_501060

theorem left_square_side_length (x : ℕ) (h1 : x + (x + 17) + (x + 11) = 52) : x = 8 :=
sorry

end left_square_side_length_l501_501060


namespace recommended_apps_l501_501550

namespace RogerPhone

-- Let's define the conditions.
def optimalApps : ℕ := 50
def currentApps (R : ℕ) : ℕ := 2 * R
def appsToDelete : ℕ := 20

-- Defining the problem as a theorem.
theorem recommended_apps (R : ℕ) (h1 : 2 * R = optimalApps + appsToDelete) : R = 35 := by
  sorry

end RogerPhone

end recommended_apps_l501_501550


namespace MN_equal_l501_501987

def M : Set ℝ := {x | ∃ (m : ℤ), x = Real.sin ((2 * m - 3) * Real.pi / 6)}
def N : Set ℝ := {y | ∃ (n : ℤ), y = Real.cos (n * Real.pi / 3)}

theorem MN_equal : M = N := by
  sorry

end MN_equal_l501_501987


namespace perimeter_C_is_74_l501_501532

/-- Definitions of side lengths based on given perimeters -/
def side_length_A (p_A : ℕ) : ℕ :=
  p_A / 4

def side_length_B (p_B : ℕ) : ℕ :=
  p_B / 4

/-- Definition of side length of C in terms of side lengths of A and B -/
def side_length_C (s_A s_B : ℕ) : ℚ :=
  (s_A : ℚ) / 2 + 2 * (s_B : ℚ)

/-- Definition of perimeter in terms of side length -/
def perimeter (s : ℚ) : ℚ :=
  4 * s

/-- Theorem statement: the perimeter of square C is 74 -/
theorem perimeter_C_is_74 (p_A p_B : ℕ) (h₁ : p_A = 20) (h₂ : p_B = 32) :
  perimeter (side_length_C (side_length_A p_A) (side_length_B p_B)) = 74 := by
  sorry

end perimeter_C_is_74_l501_501532


namespace fp_minus_p_divisible_by_9_l501_501198

def f (n : ℕ) : ℕ := 2^(n - 1) * n

theorem fp_minus_p_divisible_by_9 (p : ℕ) (h: nat.digits 10 p).length = 2011) : (f(p) - p) % 9 = 0 := by
  sorry

end fp_minus_p_divisible_by_9_l501_501198


namespace black_and_gray_areas_equal_l501_501555

theorem black_and_gray_areas_equal
    (R r : ℝ)
    (h : R = 2 * r) :
    let A_large := π * R^2,
        A_small := π * r^2,
        total_area_small := 4 * A_small,
        gray_area := total_area_small / 4,
        black_area := A_large - total_area_small
    in gray_area = black_area :=
by 
  -- Proof (filled with sorry to skip the details):
  sorry

end black_and_gray_areas_equal_l501_501555


namespace triangle_angle_A_is_60_l501_501649

theorem triangle_angle_A_is_60 (a b c A B C : ℝ) 
  (h1: a = b * sin A)
  (h2: b = a * sin B)
  (h3: c = a * sin C)
  (h4: (2 * b - c) * cos A = a * cos C):
  A = 60 :=
by
  -- The proof will be provided here
  sorry

end triangle_angle_A_is_60_l501_501649


namespace option_C_is_different_l501_501489

def cause_and_effect_relationship (description: String) : Prop :=
  description = "A: Great teachers produce outstanding students" ∨
  description = "B: When the water level rises, the boat goes up" ∨
  description = "D: The higher you climb, the farther you see"

def not_cause_and_effect_relationship (description: String) : Prop :=
  description = "C: The brighter the moon, the fewer the stars"

theorem option_C_is_different :
  ∀ (description: String),
  (not_cause_and_effect_relationship description) →
  ¬ cause_and_effect_relationship description :=
by intros description h1 h2; sorry

end option_C_is_different_l501_501489


namespace inversely_proportional_solution_l501_501789

theorem inversely_proportional_solution :
  ∃ x y : ℚ, (x + y = 40) ∧ (x - y = 8) ∧ (x * y = 384) ∧ (y = 54 + 6/7 ∧ x = 7) :=
by
  use [7, 54 + 6/7]
  split; linarith
  split; linarith
  split; linarith
  linarith

end inversely_proportional_solution_l501_501789


namespace inequality_proof_l501_501372

theorem inequality_proof (α : ℝ) (n : ℕ) (x : ℕ → ℝ) 
  (hα : α ≤ 1)
  (hx_pos : ∀ i, 1 ≤ i → i ≤ n → x i > 0)
  (hx_dec : ∀ i j, 1 ≤ i → i ≤ j → j ≤ n → x i ≥ x j)
  (hx_one : ∀ i, 1 ≤ i → i ≤ n → x i ≤ 1)
  : ((1 + ∑ i in Finset.range n, x (i+1)) ^ α) ≤ 
    (1 + ∑ i in Finset.range n, (i+1)^(α-1) * (x (i+1))^α) := sorry

end inequality_proof_l501_501372


namespace find_a_solve_inequality_l501_501984

-- Definitions based on the conditions
def inequality_condition (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - 3 * x + 6 > 4 ↔ x < 1 ∨ x > 2

-- Theorem for the first part of the problem
theorem find_a (a : ℝ) (h : inequality_condition a) : a = 1 :=
  sorry

-- Theorem for the second part of the problem
theorem solve_inequality (c : ℝ) (a : ℝ) (h : a = 1) : 
  (∀ x : ℝ, (c - x) * (a * x + 2) > 0 → 
    (if c = -2 then false else if c > -2 then x ∈ set.Ioo (-2) c else x ∈ set.Ioo c (-2))) :=
  sorry

end find_a_solve_inequality_l501_501984


namespace lana_total_spending_l501_501453

theorem lana_total_spending (ticket_price : ℕ) (tickets_friends : ℕ) (tickets_extra : ℕ)
  (H1 : ticket_price = 6)
  (H2 : tickets_friends = 8)
  (H3 : tickets_extra = 2) :
  ticket_price * (tickets_friends + tickets_extra) = 60 :=
by
  sorry

end lana_total_spending_l501_501453


namespace find_P2_l501_501691

def P1 : ℕ := 64
def total_pigs : ℕ := 86

theorem find_P2 : ∃ (P2 : ℕ), P1 + P2 = total_pigs ∧ P2 = 22 :=
by 
  sorry

end find_P2_l501_501691


namespace value_of_f_at_5_l501_501592

def f : ℤ → ℝ
| x := if x ≤ 0 then 2^x else f (x - 3)

theorem value_of_f_at_5 : f 5 = 1 / 2 :=
by
  sorry

end value_of_f_at_5_l501_501592


namespace mass_percentage_iodine_neq_662_l501_501933

theorem mass_percentage_iodine_neq_662 (atomic_mass_Al : ℝ) (atomic_mass_I : ℝ) (molar_mass_AlI3 : ℝ) :
  atomic_mass_Al = 26.98 ∧ atomic_mass_I = 126.90 ∧ molar_mass_AlI3 = ((1 * atomic_mass_Al) + (3 * atomic_mass_I)) →
  (3 * atomic_mass_I / molar_mass_AlI3 * 100) ≠ 6.62 :=
by
  sorry

end mass_percentage_iodine_neq_662_l501_501933


namespace singer_arrangements_l501_501827

theorem singer_arrangements (s1 s2 : Type) [Fintype s1] [Fintype s2] 
  (h1 : Fintype.card s1 = 4) (h2 : Fintype.card s2 = 1) :
  ∃ n : ℕ, n = 18 :=
by
  sorry

end singer_arrangements_l501_501827


namespace min_value_a4b3c2_l501_501672

theorem min_value_a4b3c2 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : 1/a + 1/b + 1/c = 9) : a^4 * b^3 * c^2 ≥ 1/1152 := 
sorry

end min_value_a4b3c2_l501_501672


namespace sequence_squares_l501_501481

theorem sequence_squares (k : ℤ) (h : k = 1) :
  ∀ n : ℕ, ∃ m : ℤ, a_n = m^2
  where
    a : ℕ → ℤ
    | 0       => 1
    | (n + 1) => a n + 8*n := 
begin
  sorry
end

end sequence_squares_l501_501481


namespace area_inside_Z_outside_X_l501_501504

structure Circle :=
  (center : Real × Real)
  (radius : ℝ)

def tangent (A B : Circle) : Prop :=
  dist A.center B.center = A.radius + B.radius

theorem area_inside_Z_outside_X (X Y Z : Circle)
  (hX : X.radius = 1) 
  (hY : Y.radius = 1) 
  (hZ : Z.radius = 1)
  (tangent_XY : tangent X Y)
  (tangent_XZ : tangent X Z)
  (non_intersect_YZ : dist Z.center Y.center > Z.radius + Y.radius) :
  π - 1/2 * π = 1/2 * π := 
by
  sorry

end area_inside_Z_outside_X_l501_501504


namespace find_a_l501_501289

theorem find_a (a : ℝ) (h1 : ∀ x : ℝ, a^(2*x - 4) ≤ 2^(x^2 - 2*x)) (ha_pos : a > 0) (ha_neq1 : a ≠ 1) : a = 2 :=
sorry

end find_a_l501_501289


namespace long_diagonal_length_l501_501871

-- Define the lengths of the rhombus sides and diagonals
variables (a b : ℝ) (s : ℝ)
variable (side_length : ℝ)
variable (short_diagonal : ℝ)
variable (long_diagonal : ℝ)

-- Given conditions
def rhombus (side_length: ℝ) (short_diagonal: ℝ) : Prop :=
  side_length = 51 ∧ short_diagonal = 48

-- To prove: length longer diagonal is 90 units
theorem long_diagonal_length (side_length: ℝ) (short_diagonal: ℝ) (long_diagonal: ℝ) :
  rhombus side_length short_diagonal →
  long_diagonal = 90 :=
by
  sorry 

end long_diagonal_length_l501_501871


namespace exists_k_l501_501772

-- Define the problem conditions
variable {x0 y0 z0 : ℚ}

-- Hypothesis: Point (x0, y0, z0) has rational coordinates and does not lie on any plane x ± y ± z = n for n ∈ ℤ
def rational_not_on_plane (x0 y0 z0 : ℚ) : Prop :=
  ∀ n : ℤ, (x0 - y0 - z0 ≠ n) ∧ (x0 - y0 + z0 ≠ n) ∧ (x0 + y0 - z0 ≠ n) ∧ (x0 + y0 + z0 ≠ n)

-- Main theorem
theorem exists_k (h : rational_not_on_plane x0 y0 z0) : ∃ k : ℕ, 
  let x := k * x0, y := k * y0, z := k * z0 in
  ∃ a b c : ℤ, x = (a - b - c) ∧ y = (b - c - a) ∧ z = (c - a - b)
  ∧ ∀ m : ℤ, x ≠ m ∨ y ≠ m ∨ z ≠ m ∨ x + y + z ≠ m :=
sorry

end exists_k_l501_501772


namespace train_speed_l501_501153

def length_of_train : ℝ := 250
def length_of_bridge : ℝ := 120
def time_taken : ℝ := 20
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def speed_of_train : ℝ := total_distance / time_taken

theorem train_speed : speed_of_train = 18.5 :=
  by sorry

end train_speed_l501_501153


namespace probability_wheel_l501_501486

theorem probability_wheel (P : ℕ → ℚ) 
  (hA : P 0 = 1/4) 
  (hB : P 1 = 1/3) 
  (hC : P 2 = 1/6) 
  (hSum : P 0 + P 1 + P 2 + P 3 = 1) : 
  P 3 = 1/4 := 
by 
  -- Proof here
  sorry

end probability_wheel_l501_501486


namespace exists_odd_square_in_200x200_l501_501502

-- Define the setup for the grid
def Grid : Type := Array (Array Bool) -- True represents a black cell, False represents a white cell

-- Condition: Grid size is 200x200
def is200x200 (g : Grid) : Prop := (g.size = 200) ∧ ∀ row, row ∈ g → row.size = 200

-- Condition: There are 404 more black cells than white cells.
def blackWhiteDifference (g : Grid) : Prop := 
  let blackCells := g.foldl (λ acc row, acc + row.count id) 0
  let whiteCells := g.foldl (λ acc row, acc + row.count (λ b, ¬b)) 0
  blackCells = whiteCells + 404

-- Question: Exists a 2x2 square with an odd number of white cells
def exists2x2OddWhiteCells (g : Grid) : Prop := 
  ∃ i j, i < 199 ∧ j < 199 ∧
  let subGrid := [g[i][j], g[i][j+1], g[i+1][j], g[i+1][j+1]]
  (subGrid.count (not ∘ id)) % 2 = 1

-- Theorem: Prove that there exists a 2x2 square with an odd number of white cells.
theorem exists_odd_square_in_200x200 (g : Grid) (h1 : is200x200 g) (h2 : blackWhiteDifference g) : 
  exists2x2OddWhiteCells g := 
sorry

end exists_odd_square_in_200x200_l501_501502


namespace hiking_hours_l501_501795

theorem hiking_hours
  (violet_water_per_hour : ℕ := 800)
  (dog_water_per_hour : ℕ := 400)
  (total_water : ℕ := 4800) :
  (total_water / (violet_water_per_hour + dog_water_per_hour) = 4) :=
by
  sorry

end hiking_hours_l501_501795


namespace sum_x1_x2_eq_3_l501_501720

variable {X : Type} [Fintype X] [DecidableEq X]

-- Defining probabilities
constants (x1 x2 : ℝ) (P : X → ℝ)
axiom prob_x1 : P x1 = 2/3
axiom prob_x2 : P x2 = 1/3

-- Defining expectations and variances
axiom expec_X : (2/3) * x1 + (1/3) * x2 = 4/3
axiom var_X : (2/3) * (x1 - 4/3)^2 + (1/3) * (x2 - 4/3)^2 = 2/9

-- Condition that x1 < x2
axiom cond_x1_lt_x2 : x1 < x2

-- Theorem that states the result
theorem sum_x1_x2_eq_3 : x1 + x2 = 3 :=
sorry

end sum_x1_x2_eq_3_l501_501720


namespace problem_part1_problem_part2_l501_501007

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem problem_part1 (h : ∀ x : ℝ, f (-x) = -f x) : a = 1 :=
sorry

theorem problem_part2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
sorry

end problem_part1_problem_part2_l501_501007


namespace point_not_on_transformed_plane_l501_501014

def point_A : ℝ × ℝ × ℝ := (4, 0, -3)

def plane_eq (x y z : ℝ) : ℝ := 7 * x - y + 3 * z - 1

def scale_factor : ℝ := 3

def transformed_plane_eq (x y z : ℝ) : ℝ := 7 * x - y + 3 * z - (scale_factor * 1)

theorem point_not_on_transformed_plane :
  transformed_plane_eq 4 0 (-3) ≠ 0 :=
by
  sorry

end point_not_on_transformed_plane_l501_501014


namespace tennis_tournament_matches_l501_501308

theorem tennis_tournament_matches:
  (total_players : ℕ) (byes : ℕ) (matches : ℕ)
  (h1 : total_players = 120)
  (h2 : byes = 32)
  (h3 : matches = total_players - byes - 1)
  (h4 : (total_players - byes - 1) % 7 = 0) :
  matches = 119 :=
by
  sorry

end tennis_tournament_matches_l501_501308


namespace find_inverse_l501_501255

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 + 1) / Real.log 2

theorem find_inverse :
  (∃ x : ℝ, x ≤ 0 ∧ f x = 2) ↔ (-sqrt 3) = -sqrt 3 :=
by
  sorry

end find_inverse_l501_501255


namespace min_sum_value_l501_501551

noncomputable def σ (Y : finset ℝ) := Y.sum id

theorem min_sum_value (m n : ℕ) (hm : 0 < m) (hn : 0 < n) 
    (x : fin m → ℝ) (hx : ∀ i, 0 < x i) (hx_sorted : sorted (finset.sorted_lt (finset.univ.image x : finset ℝ)))
    (A : fin n → finset (fin m)) (hA : ∀ i, (A i).nonempty) :
    ∑ i in finset.univ, ∑ j in finset.univ, (σ ((finset.bUnion A) i ∩ (finset.bUnion A) j) / (σ ((finset.bUnion A) i) * σ ((finset.bUnion A) j))) ≥ 
    n^2 / (finset.univ.image x).sum :=
begin
  sorry
end

end min_sum_value_l501_501551


namespace min_third_side_triangle_l501_501748

theorem min_third_side_triangle (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
    (h_distinct_1 : 42 * a ≠ 72 * b) (h_distinct_2 : 42 * a ≠ c) (h_distinct_3 : 72 * b ≠ c) :
    (42 * a + 72 * b > c) ∧ (42 * a + c > 72 * b) ∧ (72 * b + c > 42 * a) → c ≥ 7 :=
sorry

end min_third_side_triangle_l501_501748


namespace ratio_of_areas_l501_501237

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 0⟩
def B : Point := ⟨0, 2⟩
def C : Point := ⟨3, 2⟩
def D : Point := ⟨3, 0⟩
def E : Point := ⟨1.5, 1⟩
def F : Point := ⟨2.25, 0⟩

def area_triangle (p1 p2 p3 : Point) : ℝ :=
  (1 / 2) * abs ((p1.x * (p2.y - p3.y)) + (p2.x * (p3.y - p1.y)) + (p3.x * (p1.y - p2.y)))

def area_ABEF : ℝ := (area_triangle A B E) + (area_triangle A E F)
def area_DFE : ℝ := area_triangle D F E

theorem ratio_of_areas :
  (area_DFE / area_ABEF) = (3 / 17) :=
sorry

end ratio_of_areas_l501_501237


namespace family_reunion_attendees_l501_501494

variable (A : ℕ) 
variable (teenagers children : ℕ) 
variable (attendees : ℕ)

-- Given conditions in the problem
def adultMales (A : ℕ) := 0.30 * A = 100
def teenagers_equation (A teenagers: ℕ) := teenagers = A / 2
def children_equation (A children: ℕ) := children = 2 * A
def attendees_equation (A teenagers children attendees: ℕ) := attendees = A + teenagers + children

theorem family_reunion_attendees (h1 : adultMales A) (h2 : teenagers_equation A teenagers) (h3 : children_equation A children) (h4 : attendees_equation A teenagers children attendees) : attendees = 1170 := by
  sorry

end family_reunion_attendees_l501_501494


namespace polynomial_count_is_five_l501_501565

noncomputable def polynomial_count : Nat :=
  let cond (n : Nat) (a : Fin n → Int) : Prop :=
    2 * (Finset.univ.sum (λ i => Int.natAbs (a i))) + n = 4
  Finset.univ.filter (λ n => Finset.univ.filter (λ a => cond (n + 1) a).card).sum

theorem polynomial_count_is_five :
  polynomial_count = 5 := by
  sorry

end polynomial_count_is_five_l501_501565


namespace solve_equation_l501_501190

theorem solve_equation : ∀ x : ℝ, x ≠ 3 → (x + 36 / (x - 3) = -9) ↔ (x = -3) :=
by
  intro x h
  have h_eq : (x + 36 / (x - 3) = -9) ↔ ((x-3)(x+9)=0) := sorry
  have h_solve : ((x-3)(x+9)=0) ↔ (x = -3) := sorry
  exact (h_eq.trans h_solve)

end solve_equation_l501_501190


namespace consecutive_even_product_form_correct_l501_501412

theorem consecutive_even_product_form_correct :
  ∃ (n : ℕ) (digits : ℕ → ℕ),
  let product := n * (n + 2) * (n + 4) in
  product = 87526608 ∧ 
  (digits 0 = 5 ∧ digits 1 = 2 ∧ digits 2 = 6 ∧ digits 3 = 6 ∧ digits 4 = 0) :=
sorry

end consecutive_even_product_form_correct_l501_501412


namespace inverse_composition_l501_501353

variable {X Y Z W : Type}
variable (p : X → Y) (q : Y → Z) (r : Z → W)
variable [invertible p] [invertible q] [invertible r]

noncomputable def f : X → W :=
  p ∘ q ∘ r

theorem inverse_composition :
  (f p q r)⁻¹ = (r⁻¹) ∘ (q⁻¹) ∘ (p⁻¹) :=
sorry

end inverse_composition_l501_501353


namespace find_missing_digit_l501_501189

-- Definitions
def divisible_by (m n : ℕ) : Prop := ∃ k : ℕ, n = k * m

def valid_number (B : ℕ) : Prop :=
  let n := 100 * B + 40 in
  divisible_by 15 n ∧ divisible_by 5 n

-- Theorem statement
theorem find_missing_digit (B : ℕ) (h1 : valid_number B) : B = 5 :=
sorry

end find_missing_digit_l501_501189


namespace find_b_l501_501282

theorem find_b (a b : ℝ) 
  (h1: a + b - 2 = 0) 
  (h2: 3a + 2b - 3 = 0) : 
  b = 3 := 
by 
  sorry

end find_b_l501_501282


namespace sum_of_powers_i_l501_501169

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the power cycle property of i
def i_pow (n : ℤ) : ℂ := i ^ n

theorem sum_of_powers_i : (∑ k in Finset.range (301), i_pow (k - 150)) = 1 :=
by
  -- We need to supply Lean with the proof here
  sorry

end sum_of_powers_i_l501_501169


namespace sin_420_deg_l501_501903

theorem sin_420_deg : 
  let sin_periodic (θ : ℝ) := sin (θ + 360) = sin θ in
  sin 420 = (√3) / 2 :=
  by
    sorry

end sin_420_deg_l501_501903


namespace trains_crossing_time_l501_501839

noncomputable def timeToCross (L1 L2 : ℕ) (v1 v2 : ℕ) : ℝ :=
  let total_distance := (L1 + L2 : ℝ)
  let relative_speed := ((v1 + v2) * 1000 / 3600 : ℝ) -- converting km/hr to m/s
  total_distance / relative_speed

theorem trains_crossing_time :
  timeToCross 140 160 60 40 = 10.8 := 
  by 
    sorry

end trains_crossing_time_l501_501839


namespace largest_prime_divisor_l501_501776

theorem largest_prime_divisor {n : ℕ} (h1 : 1000 ≤ n) (h2 : n ≤ 1050) : ∃ p : ℕ, prime p ∧ p <= int.floor (real.sqrt n) ∧ p = 31 := 
by sorry

end largest_prime_divisor_l501_501776


namespace largest_N_with_square_in_base_nine_l501_501001

theorem largest_N_with_square_in_base_nine:
  ∃ N: ℕ, (9^2 ≤ N^2 ∧ N^2 < 9^3) ∧ ∀ M: ℕ, (9^2 ≤ M^2 ∧ M^2 < 9^3) → M ≤ N ∧ N = 26 := 
sorry

end largest_N_with_square_in_base_nine_l501_501001


namespace IncorrectStatement_l501_501833

-- Define the conditions
def WaterActsAsAgent : Prop := 
  "Water can act as both an oxidizing agent and a reducing agent in chemical reactions"

def SameMassNumberDifferentProperties : Prop := 
  "^14C and ^14N have the same mass number but different chemical properties"

def MilkIsColloid : Prop := 
  "Milk is a colloid and can exhibit the Tyndall effect"

def SameElementDifferentSubstances : Prop := 
  "The same element can form different substances, e.g., oxygen (O2) and ozone (O3)"

-- Define the statement
theorem IncorrectStatement
  (c1 : WaterActsAsAgent)
  (c2 : SameMassNumberDifferentProperties)
  (c3 : MilkIsColloid)
  (c4 : SameElementDifferentSubstances) :
  ¬ "A substance composed of the same element must be a pure substance" :=
  sorry

end IncorrectStatement_l501_501833


namespace car_min_distance_from_mount_fuji_l501_501148

/-- Define the coordinates of the points based on the conditions. --/
def F : ℝ × ℝ := (0, 0) -- Mount Fuji
def A : ℝ × ℝ := (0, 60) -- Initial position of the car
def B : ℝ × ℝ := (-45, 0) -- Position of the car after one hour

/-- Define the distance function on the plane. --/
def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

/-- Define the length of the hypotenuse AB. --/
noncomputable def AB := distance A B

/-- Given the area triangles equations to find the altitude. --/
theorem car_min_distance_from_mount_fuji : 
  let triangle_area := 1 / 2 * 60 * 45
  let altitude_dist := 2 * triangle_area / AB
  altitude_dist = 36 :=
sorry

end car_min_distance_from_mount_fuji_l501_501148


namespace proof_problem_l501_501096

noncomputable theory

def num_students := 2023
def num_boys := 1012
def num_girls := 1011

def students_selection (X : ℕ) : Prop :=
  ∀ X, X ≥ 0 ∧ X ≤ 10

def is_hypergeometric (X : ℕ) : Prop :=
  students_selection X → true -- Placeholder for hypergeometric property

def expectation_X := 2023
def expectation_shift (E : ℕ) : Prop :=
  E = expectation_X - 1 → E ≠ expectation_X

def variance_X := 2
def variance_scaling (V : ℕ) : Prop :=
  V = 4 * variance_X → V = 8

def binomial_distribution (n : ℕ) (p : ℚ) : Prop := 
  true -- Placeholder for binomial property

def binomial_symmetry (P : ℚ) : Prop :=
  P ≠ P -- Expected inequality for binomial tail probabilities

theorem proof_problem :
  is_hypergeometric X ∧
  expectation_shift 2022 ∧
  variance_scaling 8 ∧
  binomial_symmetry (0.5 : ℚ) :=
by
  -- Proof omitted
  sorry

end proof_problem_l501_501096


namespace distinct_collections_of_letters_l501_501364

theorem distinct_collections_of_letters :
  let magnets := "MATHEMATICSE".toList,
      vowels := ['A', 'A', 'E', 'E', 'I'],
      consonants := ['M', 'M', 'T', 'T', 'H', 'C', 'S', 'T'] in
  ∃ collections : ℕ,
  collections = 336 :=
by
  sorry

end distinct_collections_of_letters_l501_501364


namespace zero_in_interval_l501_501767

noncomputable def f (x : ℝ) : ℝ := log x / log 2 + x - 10

theorem zero_in_interval : ∃ c ∈ set.Ioo (6 : ℝ) 8, f c = 0 :=
  sorry

end zero_in_interval_l501_501767


namespace remainder_polynomial_l501_501344

open Polynomial

noncomputable def Q : Polynomial ℚ := 
  Polynomial.X * 1 -- placeholder polynomial for Q, definition is not needed for proof statement

theorem remainder_polynomial (Q : Polynomial ℚ) (h1 : Q.eval 17 = 15) (h2 : Q.eval 13 = 8) :
  ∃ (c d : ℚ), (∀ x, Q.eval x = ((x - 17) * (x - 13) * R(x) + c * x + d) → (c = 7/4 ∧ d = -59/4)) :=
by
  sorry

end remainder_polynomial_l501_501344


namespace actual_height_of_boy_l501_501726

variable (wrong_height : ℕ) (boys : ℕ) (wrong_avg correct_avg : ℕ)
variable (x : ℕ)

-- Given conditions
def conditions 
:= boys = 35 ∧
   wrong_height = 166 ∧
   wrong_avg = 185 ∧
   correct_avg = 183

-- Question: Proving the actual height
theorem actual_height_of_boy (h : conditions boys wrong_height wrong_avg correct_avg) : 
  x = wrong_height + (boys * wrong_avg - boys * correct_avg) := 
  sorry

end actual_height_of_boy_l501_501726


namespace total_seats_value_l501_501065

noncomputable def students_per_bus : ℝ := 14.0
noncomputable def number_of_buses : ℝ := 2.0
noncomputable def total_seats : ℝ := students_per_bus * number_of_buses

theorem total_seats_value : total_seats = 28.0 :=
by
  sorry

end total_seats_value_l501_501065


namespace pair_d_are_equal_l501_501883

theorem pair_d_are_equal : -(2 ^ 3) = (-2) ^ 3 :=
by
  -- Detailed proof steps go here, but are omitted for this task.
  sorry

end pair_d_are_equal_l501_501883


namespace approx_sqrt_one_plus_x_approx_sqrt_10_l501_501842

theorem approx_sqrt_one_plus_x (x : Real) (h : abs x < 0.1) :
  abs (sqrt (1 + x) - (1 + x / 2 - x^2 / 2^3 + x^3 / 2^4 - 5 * x^4 / 2^7 + 7 * x^5 / 2^8 - 21 * x^6 / 2^10)) < ε :=
sorry

theorem approx_sqrt_10 :
  abs (sqrt 10 - (let x := 9 in
  let ε := 10^(-7) in
  let approx1 := 3 + 1/6 - 3/2 * x^2 + 3/2 * x^3 - 15/2 * x^4 + 21/8 * x^5 - 63/16 * x^6 in
  let approx2 := (10 / 3) * (1 - 1/10) in
  min approx1 approx2)) < 10^(-7) :=
sorry

end approx_sqrt_one_plus_x_approx_sqrt_10_l501_501842


namespace num_k_values_lcm_l501_501212

-- Define prime factorizations of given numbers
def nine_pow_nine := 3^18
def twelve_pow_twelve := 2^24 * 3^12
def eighteen_pow_eighteen := 2^18 * 3^36

-- Number of values of k making eighteen_pow_eighteen the LCM of nine_pow_nine, twelve_pow_twelve, and k
def number_of_k_values : ℕ := 
  19 -- Based on calculations from the proof

theorem num_k_values_lcm :
  ∀ (k : ℕ), eighteen_pow_eighteen = Nat.lcm (Nat.lcm nine_pow_nine twelve_pow_twelve) k → ∃ n, n = number_of_k_values :=
  sorry -- Add the proof later

end num_k_values_lcm_l501_501212


namespace average_temperature_is_95_l501_501756

noncomputable def tempNY := 80
noncomputable def tempMiami := tempNY + 10
noncomputable def tempSD := tempMiami + 25
noncomputable def avg_temp := (tempNY + tempMiami + tempSD) / 3

theorem average_temperature_is_95 :
  avg_temp = 95 :=
by
  sorry

end average_temperature_is_95_l501_501756


namespace gcd_of_power_of_two_plus_one_l501_501493

theorem gcd_of_power_of_two_plus_one (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m ≠ n) : 
  Nat.gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 := 
sorry

end gcd_of_power_of_two_plus_one_l501_501493


namespace range_of_k_l501_501590

noncomputable def point_satisfies_curve (a k : ℝ) : Prop :=
(-a)^2 - a * (-a) + 2 * a + k = 0

theorem range_of_k (a k : ℝ) (h : point_satisfies_curve a k) : k ≤ 1 / 2 :=
by
  sorry

end range_of_k_l501_501590


namespace positive_integers_satisfy_condition_l501_501999

noncomputable def count_positive_integers (n : ℕ) : ℕ :=
  if (140 * n) ^ 40 > n ^ 80 ∧ n ^ 80 > 3 ^ 160 then 1 else 0

theorem positive_integers_satisfy_condition :
  Nat.sum (List.map count_positive_integers (List.range (140 + 1))) = 130 :=
begin
  sorry
end

end positive_integers_satisfy_condition_l501_501999


namespace tom_fractions_l501_501778

theorem tom_fractions (packages : ℕ) (cars_per_package : ℕ) (cars_left : ℕ) (nephews : ℕ) :
  packages = 10 → 
  cars_per_package = 5 → 
  cars_left = 30 → 
  nephews = 2 → 
  ∃ fraction_given : ℚ, fraction_given = 1/5 :=
by
  intros
  sorry

end tom_fractions_l501_501778


namespace arithmetic_sequence_term_l501_501391

theorem arithmetic_sequence_term :
  ∃ n : ℕ, (2 + (n - 1) * 3 = 20 ∧ n = 7) :=
begin
  -- Proof goes here
  sorry
end

end arithmetic_sequence_term_l501_501391


namespace probability_of_same_color_balls_l501_501424

-- Definitions of the problem
def total_balls_bag_A := 8 + 4
def total_balls_bag_B := 6 + 6
def white_balls_bag_A := 8
def red_balls_bag_A := 4
def white_balls_bag_B := 6
def red_balls_bag_B := 6

def P (event: Nat -> Bool) (total: Nat) : Nat :=
  let favorable := (List.range total).filter event |>.length
  favorable / total

-- Probability of drawing a white ball from bag A
def P_A := P (λ n => n < white_balls_bag_A) total_balls_bag_A

-- Probability of drawing a red ball from bag A
def P_not_A := P (λ n => n >= white_balls_bag_A && n < total_balls_bag_A) total_balls_bag_A

-- Probability of drawing a white ball from bag B
def P_B := P (λ n => n < white_balls_bag_B) total_balls_bag_B

-- Probability of drawing a red ball from bag B
def P_not_B := P (λ n => n >= white_balls_bag_B && n < total_balls_bag_B) total_balls_bag_B

-- Independence assumption (product rule for independent events)
noncomputable def P_same_color := P_A * P_B + P_not_A * P_not_B

-- Final theorem to prove
theorem probability_of_same_color_balls :
  P_same_color = 1 / 2 := by
    sorry

end probability_of_same_color_balls_l501_501424


namespace average_decrease_l501_501632

theorem average_decrease (A : ℕ) (original_students additional_students : ℕ) (inc_total_exp new_total_exp : ℝ) :
  original_students = 100 →
  additional_students = 20 →
  inc_total_exp = 400 →
  new_total_exp = 5400 →
  let original_total_exp := new_total_exp - inc_total_exp in
  let new_total_students := original_students + additional_students in
  let original_avg_exp_per_student := original_total_exp / original_students in
  let new_avg_exp_per_student := new_total_exp / new_total_students in
  original_avg_exp_per_student - new_avg_exp_per_student = 5 :=
by
  intros h1 h2 h3 h4
  let original_total_exp := new_total_exp - inc_total_exp
  let new_total_students := original_students + additional_students
  let original_avg_exp_per_student := original_total_exp / original_students
  let new_avg_exp_per_student := new_total_exp / new_total_students
  sorry

end average_decrease_l501_501632


namespace number_of_k_values_l501_501209

theorem number_of_k_values :
  let k (a b : ℕ) := 2^a * 3^b in
  (∀ a b : ℕ, 18 ≤ a ∧ b = 36 → 
  let lcm_val := Nat.lcm (Nat.lcm (9^9) (12^12)) (k a b) in 
  lcm_val = 18^18) →
  (Finset.card (Finset.filter (λ a, 18 ≤ a ∧ a ≤ 24) (Finset.range (24 + 1))) = 7) :=
by
  -- proof skipped
  sorry

end number_of_k_values_l501_501209


namespace percentage_per_cup_l501_501610

-- Define the conditions
def pitcher_total_capacity (C : ℚ) := C

def orange_juice (C : ℚ) := (1 / 2) * C
def apple_juice (C : ℚ) := (1 / 4) * C

def total_juice (C : ℚ) := orange_juice C + apple_juice C
def juice_per_cup (C : ℚ) := total_juice C / 4

-- Define the statement to prove
theorem percentage_per_cup (C : ℚ) (h₁ : pitcher_total_capacity C)
  (h₂ : total_juice C = (3 / 4) * C) 
  (h₃ : juice_per_cup C = (3 / 4) * C / 4) :
  ((juice_per_cup C) / C) * 100 = 18.75 := by
  sorry

end percentage_per_cup_l501_501610


namespace no_distinct_sequence_exists_l501_501552

theorem no_distinct_sequence_exists (C : ℕ) (hC : C > 1) :
  ¬ ∃ (a : ℕ → ℕ),
    (∀ n m : ℕ, n ≠ m → a n ≠ a m) ∧ 
    (∀ k : ℕ, k ≥ 1 → ∃ m : ℕ, m = k → (a (k+1))^k ∣ (C^k * ∏ i in (Finset.range k), a (i + 1))) :=
sorry

end no_distinct_sequence_exists_l501_501552


namespace average_temperature_l501_501758

def temperature_NY := 80
def temperature_MIA := temperature_NY + 10
def temperature_SD := temperature_MIA + 25

theorem average_temperature :
  (temperature_NY + temperature_MIA + temperature_SD) / 3 = 95 := 
sorry

end average_temperature_l501_501758


namespace value_of_x_l501_501643

theorem value_of_x (
  BE_eq_AD : BE = AD,
  AE_eq_CD : AE = CD,
  angle_ACB_eq_80 : ∠ ACB = 80,
  angle_ABC_eq_80 : ∠ ABC = 80,
  angle_ADE_eq_30 : ∠ ADE = 30,
  angle_BAC_eq_20 : ∠ BAC = 20
) : x = 50 :=
sorry

end value_of_x_l501_501643


namespace length_ID_equals_one_l501_501878

noncomputable theory

def circle : Type := sorry -- Define the type for a circle, as it is not available in standard libraries
def Triangle (α : Type) := sorry -- Similarly, define a type for triangle

variables (C : circle) (ABC : Triangle ℝ)
variables [is_circumscribed C ABC] -- Assume ABC is inscribed in circle C

variable (A B C I D : Point) -- Define the points
variables [incenter_of_triangle I ABC] [radius_of_circle C = 1]
variables [bisectsᵢ AI (angle BAC) = 30°] [meets_again AI D]

theorem length_ID_equals_one :
  segment_length I D = 1 := by
  sorry -- Proof skipped

end length_ID_equals_one_l501_501878


namespace new_cube_edge_length_l501_501732

/-- Given three cubes with edge lengths 6 cm, 8 cm, and 10 cm,
    prove that the edge length of the new cube formed by melting them is 12 cm. -/
theorem new_cube_edge_length
  (a1 a2 a3 : ℕ) (h1 : a1 = 6) (h2 : a2 = 8) (h3 : a3 = 10) :
  (∛((a1 ^ 3 + a2 ^ 3 + a3 ^ 3) : ℝ) : ℝ) = 12 :=
by
  sorry

end new_cube_edge_length_l501_501732


namespace find_sequence_l501_501601

def sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, 3 + 2^n = ∑ i in finset.range n.succ, a i) ∧ 
  a 1 = 5 ∧
  ∀ n ≥ 2, a n = 2^(n-1)

theorem find_sequence (a : ℕ → ℝ) :
  sequence a → (∀ n, a n = if n = 1 then 5 else 2^(n-1)) :=
by
  intro h
  sorry

end find_sequence_l501_501601


namespace f_zero_g_is_odd_range_of_m_l501_501229

-- Define the function f and the conditions associated with it.
axiom f : ℝ → ℝ
axiom functional_eq : ∀ x y : ℝ, f (x + y) = f x + f y - 2
axiom f_two : f 2 = 6

-- Part 1: Prove f(0) = 2
theorem f_zero : f 0 = 2 := 
sorry

-- Define the function g based on f.
def g (x : ℝ) : ℝ := f x - 2

-- Prove g is odd
theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := 
sorry

-- Define the conditions for part 2.
axiom f_increasing_condition : ∀ x y : ℝ, x ≠ y → (f x - f y) * (x - y) > 0
axiom f_inequality : ∀ x : ℝ, (0 < x ∧ x ≤ 4) → f x + f (1 / x - m) ≥ 8

-- Prove the range of m
theorem range_of_m (m : ℝ) : m ≤ 0 :=
sorry

end f_zero_g_is_odd_range_of_m_l501_501229


namespace find_original_cost_price_l501_501440

theorem find_original_cost_price (C S C_new S_new : ℝ) (h1 : S = 1.25 * C) (h2 : C_new = 0.80 * C) (h3 : S_new = S - 16.80) (h4 : S_new = 1.04 * C_new) : C = 80 :=
by
  sorry

end find_original_cost_price_l501_501440


namespace bakery_muffins_least_boxes_l501_501119

noncomputable def least_boxes_needed (total_muffins large_box_capacity medium_box_capacity small_box_capacity : ℕ) : ℕ :=
let large_boxes := total_muffins / large_box_capacity in
let remaining_after_large := total_muffins % large_box_capacity in
let medium_boxes := remaining_after_large / medium_box_capacity in
let remaining_after_medium := remaining_after_large % medium_box_capacity in
let small_boxes := if remaining_after_medium % small_box_capacity = 0 then remaining_after_medium / small_box_capacity else (remaining_after_medium / small_box_capacity) + 1 in
large_boxes + medium_boxes + small_boxes

theorem bakery_muffins_least_boxes :
  least_boxes_needed 250 12 8 4 = 22 :=
by
  -- Proof will be filled in here
  sorry

end bakery_muffins_least_boxes_l501_501119


namespace triangle_area_from_altitudes_l501_501724

theorem triangle_area_from_altitudes :
  ∀ (a b c : ℝ), (h_a h_b h_c : ℝ),
  h_a = 12 ∧ h_b = 15 ∧ h_c = 20 ∧
  a / h_a = b / h_b ∧ a / h_a = c / h_c ∧
  a / h_a = b / h_b  ∧ a / h_a = c / h_c →
  1 / 2 * a * h_a = 150 :=
by
  intros a b c h_a h_b h_c h_eq1 h_eq2 h_eq3 h_eq4
  sorry

end triangle_area_from_altitudes_l501_501724


namespace number_modulo_conditions_l501_501861

theorem number_modulo_conditions : 
  ∃ n : ℕ, 
  (n % 10 = 9) ∧ 
  (n % 9 = 8) ∧ 
  (n % 8 = 7) ∧ 
  (n % 7 = 6) ∧ 
  (n % 6 = 5) ∧ 
  (n % 5 = 4) ∧ 
  (n % 4 = 3) ∧ 
  (n % 3 = 2) ∧ 
  (n % 2 = 1) ∧ 
  (n = 2519) :=
by
  sorry

end number_modulo_conditions_l501_501861


namespace scientific_notation_110_billion_l501_501156

theorem scientific_notation_110_billion :
  ∃ (n : ℝ) (e : ℤ), 110000000000 = n * 10 ^ e ∧ 1 ≤ n ∧ n < 10 ∧ n = 1.1 ∧ e = 11 :=
by
  sorry

end scientific_notation_110_billion_l501_501156


namespace domain_of_w_l501_501808

def w (x : ℝ) : ℝ := real.sqrt (x - 2) + real.cbrt (x + 1)

theorem domain_of_w : {x : ℝ | ∃ y : ℝ, w x = y} = Ici 2 :=
by
  sorry

end domain_of_w_l501_501808


namespace optimal_selling_price_l501_501923

-- Definitions based on the conditions
def cost_price : ℝ := 80
def base_price : ℝ := 90
def base_units_sold : ℝ := 400
def price_increase_effect_on_units : ℝ := 20

-- Definition of profit function
def profit (x : ℝ) : ℝ :=
  let new_price := base_price + x
  let units_sold := base_units_sold - price_increase_effect_on_units * x
  (new_price - cost_price) * units_sold

-- Statement to prove
theorem optimal_selling_price :
  let x_opt := 5 in      -- Calculated from the axis of symmetry of the quadratic profit function
  base_price + x_opt = 95 :=
by
  -- Axis of symmetry calculation is skipped as it comes from the solution steps
  sorry

end optimal_selling_price_l501_501923


namespace chord_divides_hexagon_l501_501471

-- Define the hexagon and its properties
structure Hexagon :=
(circle : Type)
(A B C D E F : circle)
(length_AB : Real)
(length_BC : Real)
(length_CD : Real)
(length_DE : Real)
(length_EF : Real)
(length_FA : Real)
(length_AB_eq : length_AB = 4)
(length_BC_eq : length_BC = 4)
(length_CD_eq : length_CD = 4)
(length_DE_eq : length_DE = 6)
(length_EF_eq : length_EF = 6)
(length_FA_eq : length_FA = 6)

-- Define the chord that divides the hexagon
def chord_length (hex : Hexagon) : Real :=
  let radius := 5 in
  2 * radius

theorem chord_divides_hexagon : ∀ (h : Hexagon), chord_length h = 10 := 
by
  intro h
  unfold chord_length
  sorry

end chord_divides_hexagon_l501_501471


namespace no_solution_xyz_l501_501537

theorem no_solution_xyz : ∀ (x y z : Nat), (1 ≤ x) → (x ≤ 9) → (0 ≤ y) → (y ≤ 9) → (0 ≤ z) → (z ≤ 9) →
    100 * x + 10 * y + z ≠ 10 * x * y + x * z :=
by
  intros x y z hx1 hx9 hy1 hy9 hz1 hz9
  sorry

end no_solution_xyz_l501_501537


namespace line_intersects_circle_l501_501521

theorem line_intersects_circle :
  let λ : ℝ → ℝ → Prop := λ x y, 2 * x - y + 3 = 0
  let C : ℝ → ℝ → Prop := λ x y, x^2 + (y - 1)^2 = 5
  (∃ x y, λ x y ∧ C x y) :=
by
  sorry

end line_intersects_circle_l501_501521


namespace largest_possible_a_l501_501912

theorem largest_possible_a (a b c e : ℕ) (h1 : a < 2 * b) (h2 : b < 3 * c) (h3 : c < 5 * e) (h4 : e < 100) : a ≤ 2961 :=
by
  sorry

end largest_possible_a_l501_501912


namespace count_12_step_paths_l501_501274

noncomputable def count_paths_through_B (x_A y_A x_B y_B x_C y_C : ℕ) : ℕ :=
  (nat.choose (x_B - x_A + y_B - y_A) (x_B - x_A)) * 
  (nat.choose (x_C - x_B + y_C - y_B) (x_C - x_B))

theorem count_12_step_paths :
  count_paths_through_B 0 0 5 2 7 4 = 126 :=
by
  simp
  -- This is a placeholder proof statement
  sorry

end count_12_step_paths_l501_501274


namespace fixed_point_through_PQ_l501_501232

open EuclideanGeometry

variables {A B C P M N Q : Point} 

theorem fixed_point_through_PQ
  (hP : P ∈ line_through A B)
  (hPM_parallel : parallel (line_through P M) (line_through A C))
  (hPN_parallel : parallel (line_through P N) (line_through B C))
  (hM : M ∈ line_through B C)
  (hN : N ∈ line_through A C)
  (hcp1 : cyclic (A, P, N, Q))
  (hcp2 : cyclic (B, P, M, Q)) :
  ∃ R : Point, ∀ P' Q',
  (P' ∈ line_through A B) →
  parallel (line_through P' (pt_on_line (line_through B C))) (line_through A C) →
  parallel (line_through P' (pt_on_line (line_through A C))) (line_through B C) → 
  (cyclic (A, P', (pt_on_line (line_through A C)), Q') ∧ cyclic (B, P', (pt_on_line (line_through B C)), Q') → 
  (P' ∈ line_through R Q')) :=
begin
  sorry
end

end fixed_point_through_PQ_l501_501232


namespace hyperbola_eccentricity_eq_l501_501246

theorem hyperbola_eccentricity_eq {m : ℝ} (h : 1 + m = 4) : m = 3 :=
by
  linarith

end hyperbola_eccentricity_eq_l501_501246


namespace correct_operation_l501_501829

theorem correct_operation :
  ¬(a^2 * a^3 = a^6) ∧ ¬(6 * a / (3 * a) = 2 * a) ∧ ¬(2 * a^2 + 3 * a^3 = 5 * a^5) ∧ (-a * b^2)^2 = a^2 * b^4 :=
by
  sorry

end correct_operation_l501_501829


namespace abs_eq_solution_l501_501091

theorem abs_eq_solution (x : ℚ) : |x - 2| = |x + 3| → x = -1 / 2 :=
by
  sorry

end abs_eq_solution_l501_501091


namespace volleyball_team_starters_l501_501366

/--
There are 18 players in the school's girls volleyball team, including a set of quadruplets: Beth, Barbara, Bonnie, and Brenda.
- Beth, Barbara, Bonnie, and Brenda must be in the starting lineup.
We want to determine the number of ways to choose 8 starters given this condition.
--/
theorem volleyball_team_starters :
  let total_players := 18
  let quadruplets := 4
  let remaining_positions := 8 - quadruplets
  choose (total_players - quadruplets) remaining_positions = 1001 :=
by
  sorry

end volleyball_team_starters_l501_501366


namespace cosine_angle_ST_RQ_l501_501781

variables {P Q R S T : Type*} [inner_product_space ℝ P]

open_locale real_inner_product_space

/-- 
  Given triangles PQR and PST where Q is the midpoint of segment ST,
  PQ = ST = 1, QR = 8, PR = √65, and the dot product relation 
  PQ · PS + PR · PT = 3, prove that the cosine of the angle between
  vectors ST and RQ is 5/4.
-/
theorem cosine_angle_ST_RQ
  (PQ PS PR PT ST RQ : P)
  (h_midpoint : ∥2 • Q - (S + T)∥ = 0)
  (h_PQ : ∥PQ∥ = 1)
  (h_ST : ∥ST∥ = 1)
  (h_QR : ∥QR∥ = 8)
  (h_PR : ∥PR∥ = real.sqtr 65)
  (h_dot_prod : inner_product_space.inner PQ PS + inner_product_space.inner PR PT = 3) :
  real.cos (inner_product_space.angle ST RQ) = 5 / 4 := sorry

end cosine_angle_ST_RQ_l501_501781


namespace min_value_a4b3c2_l501_501671

theorem min_value_a4b3c2 {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1/a + 1/b + 1/c = 9) :
  a ^ 4 * b ^ 3 * c ^ 2 ≥ 1 / 5184 := 
sorry

end min_value_a4b3c2_l501_501671


namespace ratio_of_volumes_l501_501414

-- Define the parameters
variable {q : ℝ} (h : 0 < q)

-- Define the radius and height based on the problem conditions
def sphere_radius := 4 * q
def cylinder_radius := 3 * q
def cylinder_height := 5 * q

-- Define the volumes based on the given formulas
def sphere_volume := (4 / 3) * Real.pi * (sphere_radius)^3
def cylinder_volume := Real.pi * (cylinder_radius)^2 * (cylinder_height)

-- The statement to be proved 
theorem ratio_of_volumes : 
  sphere_volume / cylinder_volume = 256 / 135 :=
by sorry

end ratio_of_volumes_l501_501414


namespace problem_l501_501620

theorem problem (x : ℝ) (h : 8 * x = 3) : 200 * (1 / x) = 533.33 := by
  sorry

end problem_l501_501620


namespace semi_minor_axis_of_ellipse_l501_501634

theorem semi_minor_axis_of_ellipse : 
  let center := (-4, 2)
  let focus := (-4, 0)
  let semimajor_endpoint := (-4, 5)
  let c := abs (2 - 0)
  let a := abs (5 - 2)
  semi_minor_axis center focus semimajor_endpoint = sqrt (a^2 - c^2) :=
by
  let center := (-4, 2)
  let focus := (-4, 0)
  let semimajor_endpoint := (-4, 5)
  let c := abs (2 - 0)
  let a := abs (5 - 2)
  let b := sqrt (a^2 - c^2)
  have h1 : c = 2 := rfl
  have h2 : a = 3 := rfl
  have h3 : b = sqrt (3^2 - 2^2) := rfl
  have h4 : b = sqrt 5 := rfl
  exact h4

def semi_minor_axis (center : ℝ × ℝ) (focus : ℝ × ℝ) (semimajor_endpoint : ℝ × ℝ) : ℝ :=
  let c := abs (center.snd - focus.snd)
  let a := abs (semimajor_endpoint.snd - center.snd)
  sqrt (a^2 - c^2)

#eval semi_minor_axis (-4, 2) (-4, 0) (-4, 5) -- Expected output: sqrt 5

end semi_minor_axis_of_ellipse_l501_501634


namespace imaginary_part_of_z_l501_501688

noncomputable def i : ℂ := complex.I
noncomputable def z : ℂ := (i^3) / (2 - i)

theorem imaginary_part_of_z :
  complex.im z = -2/5 := by 
sorry

end imaginary_part_of_z_l501_501688


namespace temperature_difference_l501_501062

def lowest_temp : ℝ := -15
def highest_temp : ℝ := 3

theorem temperature_difference :
  highest_temp - lowest_temp = 18 :=
by
  sorry

end temperature_difference_l501_501062


namespace lemonade_stand_sales_l501_501779

theorem lemonade_stand_sales (
  small_sales : ℕ := 11,
  medium_sales : ℕ := 24,
  large_cups : ℕ := 5,
  price_per_large_cup : ℕ := 3
) : small_sales + medium_sales + (large_cups * price_per_large_cup) = 50 :=
sorry

end lemonade_stand_sales_l501_501779


namespace find_side_length_in_triangle_l501_501298

def cosine_rule (a b c : ℝ) (C : ℝ) : Prop :=
C = real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

def valid_side_length (s : ℝ) : Prop :=
s > 0

theorem find_side_length_in_triangle
  (a b c : ℝ) (C : ℝ)
  (h1 : b = 1)
  (h2 : c = real.sqrt 3)
  (h3 : C = real.pi * 120 / 180)
  (h4 : cosine_rule a b c C)
  (h5 : valid_side_length a) :
  a = 1 :=
by
  sorry

end find_side_length_in_triangle_l501_501298


namespace divide_polygon_into_colored_triangles_l501_501457

theorem divide_polygon_into_colored_triangles :
  ∀ (points : Fin 33 → ℝ × ℝ),  -- A function to name and locate each of the 33 points on a circle
  ∃ (colors : Fin 33 → Fin 3),  -- Each side is colored with one of three colors
  ∃ (triangles : Fin 31 → (Fin 33 × Fin 33 × Fin 33)),  -- A function returning 31 triangles (a triangulation)
  (∀ i, let (v1, v2, v3) := triangles i in
         colors (Fin.mk v1.1 (by linarith [v1.1])) ≠ colors (Fin.mk v2.1 (by linarith [v2.1])) ∧
         colors (Fin.mk v2.1 (by linarith [v2.1])) ≠ colors (Fin.mk v3.1 (by linarith [v3.1])) ∧
         colors (Fin.mk v1.1 (by linarith [v1.1])) ≠ colors (Fin.mk v3.1 (by linarith [v3.1]))) := 
sorry

end divide_polygon_into_colored_triangles_l501_501457


namespace sequence_periodic_l501_501417

def sequence (n : ℕ) : ℤ :=
if n = 1 then 20
else if n = 2 then 17
else sequence (n - 1) - sequence (n - 2)

theorem sequence_periodic (n : ℕ) (hn : 2018 = 6 * n + 2) : sequence 2018 = 17 :=
by sorry

end sequence_periodic_l501_501417


namespace people_with_diploma_l501_501312

def percent_of_people_with_diploma 
  (total_population : ℕ)
  (percent_job_choice : ℚ)
  (percent_no_diploma_with_job : ℚ)
  (percent_diploma_without_job : ℚ)
  : ℚ :=
  let percent_diploma_with_job := percent_job_choice - percent_no_diploma_with_job in
  let percent_no_job := 1 - percent_job_choice in
  let percent_diploma_without_job := percent_diploma_without_job * percent_no_job in
  percent_diploma_with_job + percent_diploma_without_job

theorem people_with_diploma
  (total_population : ℕ)
  (percent_job_choice : ℚ := 0.4)
  (percent_no_diploma_with_job : ℚ := 0.1)
  (percent_diploma_without_job : ℚ := 0.15) :
  percent_of_people_with_diploma total_population percent_job_choice percent_no_diploma_with_job percent_diploma_without_job = 0.39 :=
by
  -- Leaving the proof as sorry, as per the instructions.
  sorry

end people_with_diploma_l501_501312


namespace right_triangle_area_l501_501804

theorem right_triangle_area (a b c : ℝ) (ht : a^2 + b^2 = c^2) (h1 : a = 24) (h2 : c = 26) : 
    (1/2) * a * b = 120 :=
begin
  sorry
end

end right_triangle_area_l501_501804


namespace subset_of_inter_eq_self_l501_501576

variable {α : Type*}
variables (M N : Set α)

theorem subset_of_inter_eq_self (h : M ∩ N = M) : M ⊆ N :=
sorry

end subset_of_inter_eq_self_l501_501576


namespace polygon_interior_angles_sum_l501_501752

theorem polygon_interior_angles_sum (n : ℕ) (h : (n - 2) * 180 = 720) : n = 6 := 
by sorry

end polygon_interior_angles_sum_l501_501752


namespace regression_lines_intersect_at_sample_center_l501_501777

variables {x y l1 l2 : Type}
variables (s t : ℝ)

-- Assume the average of x is s and the average of y is t
def avg_x := s
def avg_y := t
def sample_center := (s, t)

-- Let l1 and l2 be regression lines passing through the sample center point
def line1_passes_through_sample_center :=
  ∀ (p : (ℝ × ℝ)), (p = sample_center) → l1 ∋ p

def line2_passes_through_sample_center :=
  ∀ (p : (ℝ × ℝ)), (p = sample_center) → l2 ∋ p

theorem regression_lines_intersect_at_sample_center :
  line1_passes_through_sample_center s t l1 →
  line2_passes_through_sample_center s t l2 →
  ∃ p, (p = sample_center) ∧ l1 ∋ p ∧ l2 ∋ p :=
by sorry

end regression_lines_intersect_at_sample_center_l501_501777


namespace ellipse_parabola_intersection_points_on_same_circle_l501_501570

/-- Given an ellipse E: x^2/a^2 + y^2/b^2 = 1 with a > b > 0, 
and its right focus coinciding with the focus of the parabola y^2 = 4x,
we aim to find the standard equation of the ellipse E and determine specific
lines that cause intersections to lie on the same circle.
-/
theorem ellipse_parabola_intersection :
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
    (let E := λ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1 in
      let c := 1,
      ∀ x y : ℝ, 
        (y^2 = 4 * x) →
        (2 * b^2 = 3 * a ∧ c = (a^2 - b^2)^(1 / 2)) →
        (E x y → (x / 2)^2 + (y / (√3))^2 = 1)) :=
sorry

/-- Given the same ellipse as in the previous theorem,
we aim to determine specific lines that, when intersected with the ellipse, their points of intersection
lie on the same circle.
-/
theorem points_on_same_circle :
  ∃ k : ℝ, k ≠ 0 ∧
      let l := λ x y : ℝ, y = k * (x - 1) in
      let m := λ x y : ℝ, y = (-1 / k) * (x - 1) in
      ∀ (x1 x2 y1 y2 : ℝ), 
        (l x1 y1 ∧ l x2 y2 ∧ m x1 y1 ∧ m x2 y2) →
        (k = 1 ∨ k = -1) :=
sorry

end ellipse_parabola_intersection_points_on_same_circle_l501_501570


namespace find_n_l501_501280

theorem find_n : ∃ n : ℤ, (n^2 / 4).toFloor - (n / 2).toFloor^2 = 5 ∧ n = 11 :=
by
  sorry

end find_n_l501_501280


namespace slope_range_of_tangent_line_l501_501546

theorem slope_range_of_tangent_line (x : ℝ) (h : x ≠ 0) : (1 - 1/(x^2)) < 1 :=
by
  calc 
    1 - 1/(x^2) < 1 := sorry

end slope_range_of_tangent_line_l501_501546


namespace sum_of_digits_smallest_N_l501_501341

/-- Define the probability Q(N) -/
def Q (N : ℕ) : ℚ :=
  ((2 * N) / 3 + 1) / (N + 1)

/-- Main mathematical statement to be proven in Lean 4 -/

theorem sum_of_digits_smallest_N (N : ℕ) (h1 : N > 9) (h2 : N % 6 = 0) (h3 : Q N < 7 / 10) : 
  (N.digits 10).sum = 3 :=
  sorry

end sum_of_digits_smallest_N_l501_501341


namespace carnations_count_l501_501154

-- Define the conditions 
def vase_capacity : Nat := 9
def number_of_vases : Nat := 3
def number_of_roses : Nat := 23
def total_flowers : Nat := number_of_vases * vase_capacity

-- Define the number of carnations
def number_of_carnations : Nat := total_flowers - number_of_roses

-- Assertion that should be proved
theorem carnations_count : number_of_carnations = 4 := by
  sorry

end carnations_count_l501_501154


namespace solve_sqrt_problem_l501_501587

theorem solve_sqrt_problem 
  (a b c : ℕ) 
  (h1 : sqrt (2 * a - 1) = 1 ∨ sqrt (2 * a - 1) = -1)
  (h2 : sqrt (3 * a + b - 6) = 5)
  (h3 : c = Int.floor (sqrt 67)) :
  sqrt (a + 2 * b - c) = 7 := 
sorry

end solve_sqrt_problem_l501_501587


namespace f_22_value_l501_501179

noncomputable def f : ℕ+ → ℝ
| ⟨1, _⟩ := 1
| ⟨n+1, hn⟩ := if n % 2 = 0 then (1 / 2) * f ⟨n, (Nat.succ_pos n)⟩ else f ⟨n, (Nat.succ_pos n)⟩

theorem f_22_value :
  f 22 = 1 / 1024 :=
sorry -- You can fill in the proof here

end f_22_value_l501_501179


namespace binary_to_decimal_110010_l501_501514

theorem binary_to_decimal_110010 :
  let b : Nat := 110010
  1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0 = 50 :=
by {
  -- Definitions directly from given conditions
  let b := 1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0,
  show b = 50,
  sorry
}

end binary_to_decimal_110010_l501_501514


namespace hyperbola_asymptote_equation_correct_l501_501983

def hyperbola_asymptote_equation 
  (m n : ℝ) (h : m * n ≠ 0) 
  (eccentricity : ℝ) (focus_parabola : ℝ × ℝ) 
  (hyp_eq : ∃! a b : ℝ, (y^2 = 4 * x ∧ focus_parabola = (a, b)))
  (hyp_foci_eq : eccentricity = 2 ∧ sqrt (m / n) * a = 1) 
  (asymptote_eq : String) : Prop := 
∃ x_f y_f : ℝ, hyp_foci_eq → asymptote_eq = "sqrt(3) * x ± y = 0"

theorem hyperbola_asymptote_equation_correct : 
  hyperbola_asymptote_equation (1/4) (3/4)
    (by norm_num)
    2
    (1, 0)
    (hyp_eq := ∃! a b : ℝ, (y^2 = 4 * x ∧ (1, 0) = (a, b))) 
    "sqrt(3) * x ± y = 0" := sorry

end hyperbola_asymptote_equation_correct_l501_501983


namespace monotonous_numbers_count_l501_501500

theorem monotonous_numbers_count :
  let
    is_monotonous_digit (d : Nat) := d <= 8
    is_strictly_increasing (digits : List Nat) := digits.pairwise (· < ·)
    is_strictly_decreasing (digits : List Nat) := digits.pairwise (· > ·)
    is_monotonous_number (n : Nat) :=
      let digits := Nat.digits 10 n
      (digits.length = 1 ∧ digits.all is_monotonous_digit) ∨
      (is_strictly_increasing digits ∧ digits.all is_monotonous_digit) ∨
      (is_strictly_decreasing digits ∧ digits.all is_monotonous_digit)
  in
  (Finset.range 9).filter is_monotonous_number |>.card = 1013 :=
sorry

end monotonous_numbers_count_l501_501500


namespace sequence_2018_value_l501_501419

theorem sequence_2018_value :
  let x : ℕ → ℤ := λ n, if n = 1 then 20 else if n = 2 then 17 else x (n - 1) - x (n - 2) in
  x 2018 = 17 := by
  sorry

end sequence_2018_value_l501_501419


namespace find_d_l501_501946

-- Define the six-digit number as a function of d
def six_digit_num (d : ℕ) : ℕ := 3 * 100000 + 2 * 10000 + 5 * 1000 + 4 * 100 + 7 * 10 + d

-- Define the sum of digits of the six-digit number
def sum_of_digits (d : ℕ) : ℕ := 3 + 2 + 5 + 4 + 7 + d

-- The statement we want to prove
theorem find_d (d : ℕ) : sum_of_digits d % 3 = 0 ↔ d = 3 :=
by
  sorry

end find_d_l501_501946


namespace johns_money_l501_501657

variables (total_needed more_needed current_money : ℝ)

-- Conditions from a)
def John_needs_total : Prop := total_needed = 2.50
def John_needs_more : Prop := more_needed = 1.75

-- Question and correct answer from b)
def John_already_has : Prop := current_money = total_needed - more_needed

theorem johns_money : John_needs_total ∧ John_needs_more → John_already_has :=
by
  intros h,
  cases h with hn_total hn_more,
  rw [John_needs_total, John_needs_more] at *,
  sorry

end johns_money_l501_501657


namespace function_range_l501_501812

-- Define the function y = (x^2 + 4x + 3) / (x + 1)
noncomputable def f (x : ℝ) : ℝ := (x^2 + 4*x + 3) / (x + 1)

-- State the theorem regarding the range of the function
theorem function_range : set.range (λ (x : {x : ℝ // x ≠ -1}), f x) = set.Ioo (-(∞) : ℝ) 2 ∪ set.Ioo 2 (∞ : ℝ) :=
sorry

end function_range_l501_501812


namespace lollipops_given_l501_501553

theorem lollipops_given (initial_people later_people : ℕ) (total_people groups_of_five : ℕ) :
  initial_people = 45 →
  later_people = 15 →
  total_people = initial_people + later_people →
  groups_of_five = total_people / 5 →
  total_people = 60 →
  groups_of_five = 12 :=
by intros; sorry

end lollipops_given_l501_501553


namespace solve_for_x_l501_501588

theorem solve_for_x (x : ℝ) (h1 : sqrt (2 - x) / x = 0) (h2 : x ≠ 0) : x = 2 := 
  sorry

end solve_for_x_l501_501588


namespace total_fish_l501_501455

-- Conditions
def initial_fish : ℕ := 22
def given_fish : ℕ := 47

-- Question: Total fish Mrs. Sheridan has now
theorem total_fish : initial_fish + given_fish = 69 := by
  sorry

end total_fish_l501_501455


namespace sin_5pi_over_6_l501_501523

theorem sin_5pi_over_6 : Real.sin (5 * Real.pi / 6) = 1 / 2 :=
by
  -- According to the cofunction identity for sine,
  have h1 : Real.sin (5 * Real.pi / 6) = Real.sin (Real.pi - Real.pi / 6) := by
    rw [Real.sin_sub_pi]
  -- Considering the identity sin(π - x) = sin(x),
  rw [Real.sin_of_real, Real.sin_pi_div_six]
  sorry

end sin_5pi_over_6_l501_501523


namespace label_regions_of_lines_l501_501226

universe u

-- Definitions
variables {α : Type u} [LinearOrder α] [AddGroup α]

/-- Given m lines in a plane such that no two are parallel and no three are concurrent, 
it is possible to label each region with an integer in the set {-m, -m+1, ..., -1, 1, ..., m-1, m}
such that the sum of the labels on the same side of each line is zero. -/
theorem label_regions_of_lines 
  (m : ℕ) 
  (lines : Finset (Finset (α × α)))
  (h1 : ∀ l₁ l₂ ∈ lines, l₁ ≠ l₂ → NonParallel l₁ l₂)
  (h2 : ∀ l₁ l₂ l₃ ∈ lines, Concurrent l₁ l₂ l₃ → False) :
  ∃ (labels : Finset α), 
    (∀ region ∈ PartitionRegions lines, 
      labels region ∈ (Finset.range (2 * m + 1)).map (λ x, x - m)) ∧ 
    (∀ line ∈ lines, 
      sum_labels_side labels line = 0) := 
sorry

end label_regions_of_lines_l501_501226


namespace second_player_wins_at_n_eq_4_l501_501787

def player_can_win (n : ℕ) : Type :=
  ∀ plays : Π (x : ℕ), x < n → bool, -- represents the sequence of plays available 
  ∃ win : bool, -- represents whether the second player can win
    win = true ↔ n = 4

theorem second_player_wins_at_n_eq_4 :
  player_can_win 4 :=
sorry

end second_player_wins_at_n_eq_4_l501_501787


namespace ratio_of_selected_terms_l501_501200

variable (a b : ℕ → ℚ)
variable (S T : ℕ → ℚ)

-- Definition of arithmetic sequences and their sum conditions
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  n * (a 1 + a n) / 2

-- Given condition
axiom sum_ratio_condition : ∀ n, sum_first_n_terms a n / sum_first_n_terms b n = (2 * n + 1) / (3 * n + 2)

-- The math proof problem
theorem ratio_of_selected_terms :
  is_arithmetic_sequence a →
  is_arithmetic_sequence b →
  sum_first_n_terms S = λ n, sum_first_n_terms a n →
  sum_first_n_terms T = λ n, sum_first_n_terms b n →
  ((a 2 + a 5 + a 17 + a 20) / (b 8 + b 10 + b 12 + b 14)) = 43 / 65 :=
by
  intros
  sorry

end ratio_of_selected_terms_l501_501200


namespace gcd_107_exp7_plus1_107_exp7_plus107_exp3_plus1_l501_501168

theorem gcd_107_exp7_plus1_107_exp7_plus107_exp3_plus1 : 
  let x := 107
  prime x →
  Nat.gcd (x^7 + 1) (x^7 + x^3 + 1) = 1 :=
by
  intros 
  let x := 107
  sorry

end gcd_107_exp7_plus1_107_exp7_plus107_exp3_plus1_l501_501168


namespace compute_xy_l501_501079

theorem compute_xy (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 :=
sorry

end compute_xy_l501_501079


namespace simplify_factorial_expression_l501_501664

theorem simplify_factorial_expression :
  (100! * 100!) / (99! * 101!) = 100 / 101 :=
by sorry

end simplify_factorial_expression_l501_501664


namespace case_a_case_b_case_c_l501_501747

namespace SequenceTransformation

inductive Sequence : Type
| base : (ℕ → ℝ) → Sequence
| add : Sequence → Sequence → Sequence
| sub : Sequence → Sequence → Sequence
| mul : Sequence → Sequence → Sequence
| div : Sequence → Sequence → Sequence
| remove_initial : ℕ → Sequence → Sequence

open Sequence

noncomputable def is_transformable_to_n : Sequence → Prop
| base f => ∀ n, ∃ k, f (n + k) = n
| add s1 s2 => is_transformable_to_n s1 ∧ is_transformable_to_n s2
| sub s1 s2 => is_transformable_to_n s1 ∧ is_transformable_to_n s2
| mul s1 s2 => is_transformable_to_n s1 ∧ is_transformable_to_n s2
| div s1 s2 => is_transformable_to_n s1 ∧ is_transformable_to_n s2
| remove_initial k s => is_transformable_to_n s

def seq1 : Sequence := base (λ n => (n : ℝ) ^ 2)
def seq2 : Sequence := base (λ n => (n : ℝ) + real.sqrt 2)
def seq3 : Sequence := base (λ n => ((n ^ 2000 : ℕ) + 1) / n)

theorem case_a : is_transformable_to_n seq1 := sorry

theorem case_b : ¬ is_transformable_to_n seq2 := sorry

theorem case_c : is_transformable_to_n seq3 := sorry

end SequenceTransformation

end case_a_case_b_case_c_l501_501747


namespace ratio_of_max_min_value_l501_501109

theorem ratio_of_max_min_value (n : ℕ) (x : Fin n → ℤ) 
  (h1 : ∀ i : Fin n, -1 ≤ x i ∧ x i ≤ 2)
  (h2 : (∑ i : Fin n, x i) = 19)
  (h3 : (∑ i : Fin n, (x i) ^ 2) = 99) :
  let M := max ((∑ i : Fin n, (x i) ^ 3))
  let m := min ((∑ i : Fin n, (x i) ^ 3)) in
  M / m = 7 := 
sorry

end ratio_of_max_min_value_l501_501109


namespace speed_limit_of_friend_l501_501907

theorem speed_limit_of_friend (total_distance : ℕ) (christina_speed : ℕ) (christina_time_min : ℕ) (friend_time_hr : ℕ) 
(h1 : total_distance = 210)
(h2 : christina_speed = 30)
(h3 : christina_time_min = 180)
(h4 : friend_time_hr = 3)
(h5 : total_distance = (christina_speed * (christina_time_min / 60)) + (christina_speed * friend_time_hr)) :
  (total_distance - christina_speed * (christina_time_min / 60)) / friend_time_hr = 40 := 
by
  sorry

end speed_limit_of_friend_l501_501907


namespace Kelly_weight_is_M_l501_501873

variable (M : ℝ) -- Megan's weight
variable (K : ℝ) -- Kelly's weight
variable (Mike : ℝ) -- Mike's weight

-- Conditions based on the problem statement
def Kelly_less_than_Megan (M K : ℝ) : Prop := K = 0.85 * M
def Mike_greater_than_Megan (M Mike : ℝ) : Prop := Mike = M + 5
def Total_weight_exceeds_bridge (total_weight : ℝ) : Prop := total_weight = 100 + 19
def Total_weight_of_children (M K Mike total_weight : ℝ) : Prop := total_weight = M + K + Mike

theorem Kelly_weight_is_M : (M = 40) → (Total_weight_exceeds_bridge 119) → (Kelly_less_than_Megan M K) → (Mike_greater_than_Megan M Mike) → K = 34 :=
by
  -- Insert proof here
  sorry

end Kelly_weight_is_M_l501_501873


namespace find_integer_l501_501535

theorem find_integer (x : ℤ) : (x ≡ 1 [MOD 7]) ∧ (x ≡ 2 [MOD 11]) ↔ x = 57 :=
by
  sorry

end find_integer_l501_501535


namespace omega_plus_m_eq_three_l501_501259

noncomputable def f (ω x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6) + 1 / 2

theorem omega_plus_m_eq_three (ω m x0 : ℝ) (hω : 0 < ω) (hm : 0 < m) 
  (hPQ : Real.abs (x0 - (Real.pi / (2 * ω) - x0)) = Real.pi / (3 * ω)) 
  (hQR : Real.abs ((Real.pi / ω) + x0 - x0) = 2 * Real.pi / (3 * ω)) :
  ω + m = 3 :=
  sorry

end omega_plus_m_eq_three_l501_501259


namespace fraction_of_yard_occupied_l501_501479

/-
Proof Problem: Given a rectangular yard that measures 30 meters by 8 meters and contains
an isosceles trapezoid-shaped flower bed with parallel sides measuring 14 meters and 24 meters,
and a height of 6 meters, prove that the fraction of the yard occupied by the flower bed is 19/40.
-/

theorem fraction_of_yard_occupied (length_yard width_yard b1 b2 h area_trapezoid area_yard : ℝ) 
  (h_length_yard : length_yard = 30) 
  (h_width_yard : width_yard = 8) 
  (h_b1 : b1 = 14) 
  (h_b2 : b2 = 24) 
  (h_height_trapezoid : h = 6) 
  (h_area_trapezoid : area_trapezoid = (1/2) * (b1 + b2) * h) 
  (h_area_yard : area_yard = length_yard * width_yard) : 
  area_trapezoid / area_yard = 19 / 40 := 
by {
  -- Follow-up steps to prove the statement would go here
  sorry
}

end fraction_of_yard_occupied_l501_501479


namespace number_of_squares_l501_501926

-- Define the conditions and the goal
theorem number_of_squares {x : ℤ} (hx0 : 0 ≤ x) (hx6 : x ≤ 6) {y : ℤ} (hy0 : -1 ≤ y) (hy : y ≤ 3 * x) :
  ∃ (n : ℕ), n = 123 :=
by 
  sorry

end number_of_squares_l501_501926


namespace probability_alex_finds_train_l501_501763

open Real Set

def train_arrival_time : Set ℝ := Icc 60 120
def train_waiting_time : ℝ := 20
def alex_arrival_time : Set ℝ := Icc 30 150

theorem probability_alex_finds_train : 
  (measure_theory.measure.probability {t ∈ train_arrival_time | ∃ a ∈ alex_arrival_time, t ≤ a ∧ a ≤ t + train_waiting_time}) = 5 / 24 := 
sorry

end probability_alex_finds_train_l501_501763


namespace min_value_a4b3c2_l501_501675

noncomputable def a (x : ℝ) : ℝ := if x > 0 then x else 0
noncomputable def b (x : ℝ) : ℝ := if x > 0 then x else 0
noncomputable def c (x : ℝ) : ℝ := if x > 0 then x else 0

theorem min_value_a4b3c2 (a b c : ℝ) 
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h : 1/a + 1/b + 1/c = 9) : a^4 * b^3 * c^2 ≥ 1/1152 :=
by sorry

example : ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ (1/a + 1/b + 1/c = 9) ∧ a^4 * b^3 * c^2 = 1/1152 :=
by 
  use [1/4, 1/3, 1/2]
  split
  norm_num -- 0 < 1/4
  split
  norm_num -- 0 < 1/3
  split
  norm_num -- 0 < 1/2
  split
  norm_num -- 1/(1/4) + 1/(1/3) + 1/(1/2) = 9
  norm_num -- (1/4)^4 * (1/3)^3 * (1/2)^2 = 1/1152

end min_value_a4b3c2_l501_501675


namespace find_f1_l501_501134

noncomputable def f : ℝ → ℝ := sorry

lemma functional_equation (x y : ℝ) : f(x + y) = f(x) + f(y) + 7 * x * y + 4 := sorry

lemma specific_values : f(2) + f(5) = 125 := sorry

theorem find_f1 : f(1) = 4 :=
by {
  -- Assume f satisfies the conditions
  have h1 : ∀ x y, f(x + y) = f(x) + f(y) + 7 * x * y + 4 := functional_equation,
  have h2 : f(2) + f(5) = 125 := specific_values,
  -- Goal: prove f(1) = 4
  sorry
}

end find_f1_l501_501134


namespace find_depth_of_second_digging_project_l501_501125

/-
Problem:
A certain number of people can dig earth 100 m deep, 25 m long, and 30 m broad in 12 days.
They require 12 days to dig earth some depth, 20 m long, and 50 m broad.
What is the depth of the second digging project?
-/

-- Given conditions
def volume_1 (depth length breadth : ℝ) : ℝ := depth * length * breadth
def days_1 := 12
def volume_2 (depth length breadth : ℝ) : ℝ := depth * length * breadth
def days_2 := 12

-- Known values for the first digging project
def depth_1 := 100.0
def length_1 := 25.0
def breadth_1 := 30.0

-- Known values for the second digging project
def length_2 := 20.0
def breadth_2 := 50.0

-- Correct answer as depth_2
def depth_2 := 75.0

-- The proof statement
theorem find_depth_of_second_digging_project :
  volume_1 depth_1 length_1 breadth_1 / days_1 = volume_2 depth_2 length_2 breadth_2 / days_2 :=
by
  sorry

end find_depth_of_second_digging_project_l501_501125


namespace det_A_plus_B_eq_det_B_l501_501512

open Complex Matrix

variables {A B : Matrix (Fin 3) (Fin 3) ℂ}

theorem det_A_plus_B_eq_det_B (hA : A = -Aᴴ) (hB : B = Bᴴ) 
  (hroot : ∃ x0 : ℂ, ∃ m > 1, ∀ k < m, (Polynomial.eval₂ Matrix.det (A + Polynomial.X * B) x0).derivative k = 0) :
  Matrix.det (A + B) = Matrix.det B :=
sorry

end det_A_plus_B_eq_det_B_l501_501512


namespace moles_of_ammonia_combined_l501_501544

theorem moles_of_ammonia_combined (n_CO2 n_Urea n_NH3 : ℕ) (h1 : n_CO2 = 1) (h2 : n_Urea = 1) (h3 : n_Urea = n_CO2)
  (h4 : n_Urea = 2 * n_NH3): n_NH3 = 2 := 
by
  sorry

end moles_of_ammonia_combined_l501_501544


namespace parabola_and_area_ratios_l501_501230

variables (p : ℝ) (F : ℝ × ℝ) (A B O M N : ℝ × ℝ)
variables (x1 x2 y1 y2 : ℝ) (k : ℝ)
variables (C : ℝ → ℝ) (L : ℝ → ℝ)

-- Conditions
def parabola (C : ℝ → ℝ) := ∀ x : ℝ, (C x)^2 = 2 * p * x
def focus (F : ℝ × ℝ) := F = (1, 0)
def line_passing_through_focus (L : ℝ → ℝ) := ∀ x : ℝ, L x = k * (x - 1)
def points_on_parabola (A B : ℝ × ℝ) := (A.2)^2 = 2 * p * A.1 ∧ (B.2)^2 = 2 * p * B.1
def intersections_with_line_x_neg_2 (M N : ℝ × ℝ) := M.1 = -2 ∧ N.1 = -2

-- Theorem
theorem parabola_and_area_ratios {p : ℝ} (hF : F = (1, 0))
  (hLine : ∀ x : ℝ, L x = k * (x - 1))
  (hParabola : ∀ x : ℝ, (C x)^2 = 2 * p * x)
  (hPoints : (A.2)^2 = 2 * p * A.1 ∧ (B.2)^2 = 2 * p * B.1)
  (hIntersections : M.1 = -2 ∧ N.1 = -2) :
  (p = 2 → ∀ x : ℝ, C x = sqrt (4 * x)) ∧ 
  ( ∃ (AO BO MO NO : ℝ), ∃ (α β : ℝ), 
      (x1 * x2 = 1) →
      (AO / MO * BO / NO = 1 / 4) ):=
by {
  sorry
}

end parabola_and_area_ratios_l501_501230


namespace max_permutation_sum_and_count_l501_501666

theorem max_permutation_sum_and_count : 
  let permutations := (x1 x2 x3 x4 x5 x6 : ℕ) 
                      in {x1, x2, x3, x4, x5, x6} = {1, 2, 3, 4, 5, 6} 
                         ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ x4 ≠ x5 ∧ x5 ≠ x6 ∧ x6 ≠ x1
  in max (x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x6 + x6 * x1) = 79 
  ∧ count (x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x6 + x6 * x1 = 79) permutations = 12 
  implies 79 + 12 = 91 := sorry

end max_permutation_sum_and_count_l501_501666


namespace car_price_l501_501378

/-- Prove that the price of the car Quincy bought is $20,000 given the conditions. -/
theorem car_price (years : ℕ) (monthly_payment : ℕ) (down_payment : ℕ) 
  (h1 : years = 5) 
  (h2 : monthly_payment = 250) 
  (h3 : down_payment = 5000) : 
  (down_payment + (monthly_payment * (12 * years))) = 20000 :=
by
  /- We provide the proof below with sorry because we are only writing the statement as requested. -/
  sorry

end car_price_l501_501378


namespace part_a_part_b_l501_501837

-- Part (a)
theorem part_a (S : ℕ) (coins : Fin 6 → ℕ)
  (H1 : ∀ (i j : Fin 6), i ≠ j → (coins i + coins j) % 2 = 0)
  (H2 : ∀ (i j k : Fin 6), i ≠ j ∧ i ≠ k ∧ j ≠ k → (coins i + coins j + coins k) % 3 = 0) :
  S % 6 = 0 :=
sorry

-- Part (b)
theorem part_b (S : ℕ) (coins : Fin 8 → ℕ)
  (H1 : ∀ (i j : Fin 8), i ≠ j → (coins i + coins j) % 2 = 0)
  (H2 : ∀ (i j k : Fin 8), i ≠ j ∧ i ≠ k ∧ j ≠ k → (coins i + coins j + coins k) % 3 = 0)
  (H3 : ∀ (i j k l : Fin 8), ∀ H : i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l, 
        (coins i + coins j + coins k + coins l) % 4 = 0)
  (H4 : ∀ (i j k l m : Fin 8), ∀ H : i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ 
        k ≠ l ∧ k ≠ m ∧ l ≠ m, (coins i + coins j + coins k + coins l + coins m) % 5 = 0)
  (H5 : ∀ (i j k l m n : Fin 8), ∀ H : i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ i ≠ n ∧ 
        j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ j ≠ n ∧ k ≠ l ∧ k ≠ m ∧ k ≠ n ∧ l ≠ m ∧ l ≠ n ∧ m ≠ n, 
        (coins i + coins j + coins k + coins l + coins m + coins n) % 6 = 0)
  (H6 : ∀ (i j k l m n o : Fin 8), ∀ H : i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ i ≠ m ∧ i ≠ n ∧ i ≠ o ∧
        j ≠ k ∧ j ≠ l ∧ j ≠ m ∧ j ≠ n ∧ j ≠ o ∧ k ≠ l ∧ k ≠ m ∧ k ≠ n ∧ k ≠ o ∧
        l ≠ m ∧ l ≠ n ∧ l ≠ o ∧ m ≠ n ∧ m ≠ o ∧ n ≠ o, 
        (coins i + coins j + coins k + coins l + coins m + coins n + coins o) % 7 = 0) :
  false :=
sorry

end part_a_part_b_l501_501837


namespace sales_on_second_street_l501_501711

noncomputable def commission_per_system : ℕ := 25
noncomputable def total_commission : ℕ := 175
noncomputable def total_systems_sold : ℕ := total_commission / commission_per_system

def first_street_sales (S : ℕ) : ℕ := S
def second_street_sales (S : ℕ) : ℕ := 2 * S
def third_street_sales : ℕ := 0
def fourth_street_sales : ℕ := 1

def total_sales (S : ℕ) : ℕ := first_street_sales S + second_street_sales S + third_street_sales + fourth_street_sales

theorem sales_on_second_street (S : ℕ) : total_sales S = total_systems_sold → second_street_sales S = 4 := by
  sorry

end sales_on_second_street_l501_501711


namespace pablo_mother_pays_each_page_l501_501367

-- Definitions based on the conditions in the problem
def pages_per_book := 150
def number_books_read := 12
def candy_cost := 15
def money_leftover := 3
def total_money := candy_cost + money_leftover
def total_pages := number_books_read * pages_per_book
def amount_paid_per_page := total_money / total_pages

-- The theorem to be proven
theorem pablo_mother_pays_each_page
    (pages_per_book : ℝ)
    (number_books_read : ℝ)
    (candy_cost : ℝ)
    (money_leftover : ℝ)
    (total_money := candy_cost + money_leftover)
    (total_pages := number_books_read * pages_per_book)
    (amount_paid_per_page := total_money / total_pages) :
    amount_paid_per_page = 0.01 :=
by
  sorry

end pablo_mother_pays_each_page_l501_501367


namespace sequence_term_l501_501940

theorem sequence_term (a : ℕ → ℝ) (h : ∀ n : ℕ, 0 < n → (∑ i in finset.range n, a (i + 1)) / n = n) :
  a 2023 = 4045 :=
by
  sorry

end sequence_term_l501_501940


namespace work_rate_C_l501_501836

noncomputable def work_rate_A : ℚ := 1 / 6
noncomputable def work_rate_B : ℚ := 1 / 5

theorem work_rate_C :
  ∃ (x : ℚ), (1 / 6 + 1 / 5 + 1 / x = 1 / 2) ∧ (x = 7.5) :=
by {
  existsi (15 / 2 : ℚ),
  split,
  { norm_num, },
  { norm_num, }
}

end work_rate_C_l501_501836


namespace max_lateral_surface_area_cylinder_optimizes_l501_501185

noncomputable def max_lateral_surface_area_cylinder (r m : ℝ) : ℝ × ℝ :=
  let r_c := r / 2
  let h_c := m / 2
  (r_c, h_c)

theorem max_lateral_surface_area_cylinder_optimizes {r m : ℝ} (hr : 0 < r) (hm : 0 < m) :
  let (r_c, h_c) := max_lateral_surface_area_cylinder r m
  r_c = r / 2 ∧ h_c = m / 2 :=
sorry

end max_lateral_surface_area_cylinder_optimizes_l501_501185


namespace find_range_a_l501_501986

open Real

noncomputable def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x > a

noncomputable def q (a : ℝ) : Prop :=
  ∃ x0 : ℝ, x0^2 + 2*a*x0 + 2 - a = 0

noncomputable def proposition_p_or_q (a : ℝ) : Prop := p a ∨ q a

noncomputable def proposition_p_and_not_q (a : ℝ) : Prop := p a ∧ ¬q a

theorem find_range_a (a : ℝ) :
  proposition_p_or_q a ∧ ¬proposition_p_and_not_q a ↔ a ∈ Ioo (-2 : ℝ) (-1) ∪ Ici 1 :=
sorry

end find_range_a_l501_501986


namespace divisible_by_3_l501_501841

noncomputable def red_covering (square : ℕ × ℕ) (dominoes : set (ℕ × ℕ)) : bool := sorry
noncomputable def blue_covering (square : ℕ × ℕ) (dominoes : set (ℕ × ℕ)) : bool := sorry

theorem divisible_by_3 (n : ℕ) (h1 : 0 < n)
  (h2 : ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 2 * n ∧ 1 ≤ j ∧ j ≤ 2 * n → red_covering (i, j) → blue_covering (i, j) = ff)
  (h3 : ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ 2 * n ∧ 1 ≤ j ∧ j ≤ 2 * n → ∃ k : ℤ, (i, j) = (k - (i, j)) - (red_covering (i, j) - blue_covering (i, j))): (n % 3 = 0) := 
sorry

end divisible_by_3_l501_501841


namespace minimum_clients_l501_501450

theorem minimum_clients :
  ∃ (n : ℕ), (∀ m, (repunit m > 10) ∧ (repunit (m * n) = repunit m * 101)) ∧ n = 101 := by
  -- repunit is a helper function to generate repunit numbers
  def repunit (k : ℕ) : ℕ := (dec_trivial : ℕ)
  sorry

end minimum_clients_l501_501450


namespace roots_of_quadratic_eq_l501_501745

theorem roots_of_quadratic_eq : ∀ (x : ℂ), x^2 + 4 = 0 ↔ (x = 2 * complex.I ∨ x = -2 * complex.I) :=
by sorry

end roots_of_quadratic_eq_l501_501745


namespace ratio_Ryn_Nikki_l501_501330

def Joyce_movie_length (M : ℝ) : ℝ := M + 2
def Nikki_movie_length (M : ℝ) : ℝ := 3 * M
def Ryn_movie_fraction (F : ℝ) (Nikki_movie_length : ℝ) : ℝ := F * Nikki_movie_length

theorem ratio_Ryn_Nikki 
  (M : ℝ) 
  (Nikki_movie_is_30 : Nikki_movie_length M = 30) 
  (total_movie_hours_is_76 : M + Joyce_movie_length M + Nikki_movie_length M + Ryn_movie_fraction F (Nikki_movie_length M) = 76) 
  : F = 4 / 5 := 
by 
  sorry

end ratio_Ryn_Nikki_l501_501330


namespace sum_of_squares_of_sums_l501_501684

axiom roots_of_polynomial (p q r : ℝ) : p^3 - 15*p^2 + 25*p - 12 = 0 ∧ q^3 - 15*q^2 + 25*q - 12 = 0 ∧ r^3 - 15*r^2 + 25*r - 12 = 0

theorem sum_of_squares_of_sums (p q r : ℝ)
  (h_roots : p^3 - 15*p^2 + 25*p - 12 = 0 ∧ q^3 - 15*q^2 + 25*q - 12 = 0 ∧ r^3 - 15*r^2 + 25*r - 12 = 0) :
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 := 
sorry

end sum_of_squares_of_sums_l501_501684


namespace triangle_tangent_ratio_l501_501233

variable {A B C a b c : ℝ}
hypothesis h1 : a * Real.cos B - b * Real.cos A = (3 / 5) * c

theorem triangle_tangent_ratio :
  Real.tan A / Real.tan B = 4 :=
by
  sorry

end triangle_tangent_ratio_l501_501233


namespace ratio_of_c_to_d_l501_501176

theorem ratio_of_c_to_d (x y c d : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
    (h1 : 9 * x - 6 * y = c) (h2 : 15 * x - 10 * y = d) :
    c / d = -2 / 5 :=
by
  sorry

end ratio_of_c_to_d_l501_501176


namespace total_passengers_landed_l501_501331

theorem total_passengers_landed (on_time late : ℕ) (h_on_time : on_time = 14507) (h_late : late = 213) :
  on_time + late = 14720 :=
by
  sorry

end total_passengers_landed_l501_501331


namespace find_intersection_point_l501_501195

def point := (ℝ × ℝ × ℝ)

def direction_vector (p1 p2 : point) : point :=
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

def parametrize_line (p : point) (d : point) (t : ℝ) : point :=
  (p.1 + t * d.1, p.2 + t * d.2, p.3 + t * d.3)

def xz_plane_intersection (p : point) (d : point) : point :=
  let t := p.2 / -d.2 in
  parametrize_line p d t

theorem find_intersection_point :
  let p1 := (2, 3, 2)
  let p2 := (6, -1, 7)
  let d := direction_vector p1 p2
  xz_plane_intersection p1 d = (5, 0, 23/4) := 
sorry

end find_intersection_point_l501_501195


namespace fewer_heads_than_tails_l501_501088

theorem fewer_heads_than_tails (n : ℕ) (h_n : n = 10) :
  (∑ k in finset.range (n / 2), (nat.choose n k) / 2^n) = 193 / 512 :=
by
  rw h_n
  sorry

end fewer_heads_than_tails_l501_501088


namespace unique_solution_for_a_l501_501534

theorem unique_solution_for_a (a : ℝ) :
  unique (λ x : ℝ,
    2 * x^2 - 6 * a * x + 4 * a^2 - 2 * a - 2 + log (2 * x^2 + 2 * x - 6 * a * x + 4 * a^2) /
    log 2 = log (x^2 + 2 * x - 3 * a * x + 2 * a^2 + a + 1) / log 2) ↔ a = -2 :=
sorry

end unique_solution_for_a_l501_501534


namespace arithmetic_sequence_common_difference_l501_501242

/--
Given an arithmetic sequence $\{a_n\}$ and $S_n$ being the sum of the first $n$ terms, 
with $a_1=1$ and $S_3=9$, prove that the common difference $d$ is equal to $2$.
-/
theorem arithmetic_sequence_common_difference :
  ∃ (d : ℝ), (∀ (n : ℕ), aₙ = 1 + (n - 1) * d) ∧ S₃ = a₁ + (a₁ + d) + (a₁ + 2 * d) ∧ a₁ = 1 ∧ S₃ = 9 → d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l501_501242


namespace solve_equation_l501_501723

theorem solve_equation :
  ∃ (x : ℤ), (1 / (x + 6) + 1 / (x + 3) = 1 / 2 ∧ (x = 0 ∨ x = -5)) :=
by {
  existsi 0,
  split,
  { sorry },
  { left, refl },
  existsi -5,
  split,
  { sorry },
  { right, refl },
}

end solve_equation_l501_501723


namespace original_price_of_apples_l501_501160

-- Define variables and conditions
variables (P : ℝ)

-- The conditions of the problem
def price_increase_condition := 1.25 * P * 8 = 64

-- The theorem stating the original price per pound of apples
theorem original_price_of_apples (h : price_increase_condition P) : P = 6.40 :=
sorry

end original_price_of_apples_l501_501160


namespace linear_dependence_iff_l501_501182

theorem linear_dependence_iff (m : ℝ) :
  (∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
                   a • (⟨1, 2, 3⟩ : ℝ³) + b • (⟨2, m, 5⟩ : ℝ³) + c • (⟨3, 7, m⟩ : ℝ³) = 0) ↔
  (m = (5 + Real.sqrt 37) / 2) ∨ (m = (5 - Real.sqrt 37) / 2) :=
by
  sorry

end linear_dependence_iff_l501_501182


namespace term_2023_of_sequence_l501_501943

theorem term_2023_of_sequence :
  ∀ (a : ℕ → ℕ), 
    (∀ n : ℕ, n > 0 → (∀ m, m > 0 ∧ m ≤ n → a m) → (∑ i in finset.range (n + 1), a i) = n^2) →
    a 2023 = 4045 := 
by 
  sorry

end term_2023_of_sequence_l501_501943


namespace trisected_angle_ratio_l501_501648

variables {A B C F G : Type}
variables (triangle_ABC : Triangle A B C)
variables (trisection_A : Trisector A triangle_ABC F G)
variables (meet_BC : MeetsAt F G B C)

theorem trisected_angle_ratio (h1 : Trisector A triangle_ABC F G) 
    (h2 : MeetsAt F G B C) : 
    BF / GC = (AB * AF) / (AG * AC) :=
sorry

end trisected_angle_ratio_l501_501648


namespace safari_fewer_giraffes_than_snakes_l501_501713

theorem safari_fewer_giraffes_than_snakes :
  let lions_safari := 100
  let snakes_safari := lions_safari / 2
  let giraffes_safari := 40
  let lions_savanna := 2 * lions_safari
  let snakes_savanna := 3 * snakes_safari
  let giraffes_savanna := giraffes_safari + 20
  (lions_savanna + snakes_savanna + giraffes_savanna = 410) →
  giraffes_safari < snakes_safari ∧ (snakes_safari - giraffes_safari) = 10 :=
by {
  intros,
  sorry
}

end safari_fewer_giraffes_than_snakes_l501_501713


namespace prime_dividing_fibonacci_l501_501066

def sequence (L : ℕ → ℕ) : Prop := 
  L 0 = 2 ∧ L 1 = 1 ∧ ∀ n ≥ 1, L (n + 1) = L n + L (n - 1)

theorem prime_dividing_fibonacci
  (L : ℕ → ℕ)
  (p : ℕ)
  (hp : Nat.Prime p)
  (h_sequence : sequence L)
  (k : ℕ)
  (h_div : p ∣ L (2 * k) - 2) :
  p ∣ L (2 * k + 1) - 1 := sorry

end prime_dividing_fibonacci_l501_501066


namespace polynomial_solution_l501_501536

noncomputable def q (x : ℝ) : ℝ :=
  -20 / 93 * x^3 - 110 / 93 * x^2 - 372 / 93 * x - 525 / 93

theorem polynomial_solution :
  (q 1 = -11) ∧
  (q 2 = -15) ∧
  (q 3 = -25) ∧
  (q 5 = -65) :=
by
  sorry

end polynomial_solution_l501_501536


namespace arithmetic_geometric_progression_inequality_l501_501663

theorem arithmetic_geometric_progression_inequality
  {a b c d e f D g : ℝ}
  (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d)
  (e_pos : 0 < e) (f_pos : 0 < f)
  (h1 : b = a + D)
  (h2 : c = a + 2 * D)
  (h3 : e = a * g)
  (h4 : f = a * g^2)
  (h5 : d = a + 3 * D)
  (h6 : d = a * g^3) : 
  b * c ≥ e * f :=
by sorry

end arithmetic_geometric_progression_inequality_l501_501663


namespace number_of_integers_satisfying_inequality_l501_501996

theorem number_of_integers_satisfying_inequality :
  {n : ℤ | (n + 1) * (n + 2) * (n - 7) ≤ 5}.card = 12 :=
sorry

end number_of_integers_satisfying_inequality_l501_501996


namespace no_solution_in_nat_for_xx_plus_2yy_eq_zz_l501_501045

theorem no_solution_in_nat_for_xx_plus_2yy_eq_zz :
  ¬∃ (x y z : ℕ), x^x + 2 * y^y = z^z := by
  sorry

end no_solution_in_nat_for_xx_plus_2yy_eq_zz_l501_501045


namespace domain_of_f_l501_501731

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (-x^2 + 9 * x + 10)) / Real.log (x - 1)

theorem domain_of_f :
  {x : ℝ | -x^2 + 9 * x + 10 ≥ 0 ∧ x - 1 > 0 ∧ Real.log (x - 1) ≠ 0} =
  {x : ℝ | (1 < x ∧ x < 2) ∨ (2 < x ∧ x ≤ 10)} :=
by
  sorry

end domain_of_f_l501_501731


namespace smallest_natural_number_multiple_of_36_with_unique_digits_digit_count_1023457896_sum_of_digits_1023457896_l501_501938

open Nat

theorem smallest_natural_number_multiple_of_36_with_unique_digits :
  ∃ (n : ℕ), (∀ k : ℕ, (k % 36 = 0 → (∀ d : ℕ, d ∈ List.range 10 → d ∈ digit_list k) → n ≤ k)) ∧
    n = 1023457896 :=
by
  -- Placeholder for the proof
  sorry

def digit_list (n : ℕ) : list ℕ :=
  let chars := n.digits 10
  chars.to_finset.val.to_list.sort

theorem digit_count_1023457896 :
  ∀ d : ℕ, d < 10 → d ∈ digit_list 1023457896 :=
by
  -- Placeholder for the proof
  sorry

theorem sum_of_digits_1023457896 :
  finset.sum (finset.range 10) (digit_list 1023457896) = 45 :=
by
  -- Placeholder for the proof
  sorry

end smallest_natural_number_multiple_of_36_with_unique_digits_digit_count_1023457896_sum_of_digits_1023457896_l501_501938


namespace term_2011_is_18_l501_501021

def next_term (n : ℕ) : ℕ :=
  let d := n % 10
  let m := n / 10
  d * d + 2 * m

def seq : ℕ → ℕ
| 0       := 52
| (n + 1) := next_term (seq n)

theorem term_2011_is_18 : seq 2010 = 18 :=
  sorry

end term_2011_is_18_l501_501021


namespace gunther_cleaning_free_time_l501_501992

theorem gunther_cleaning_free_time :
  let vacuum := 45
  let dusting := 60
  let mopping := 30
  let bathroom := 40
  let windows := 15
  let brushing_per_cat := 5
  let cats := 4

  let free_time_hours := 4
  let free_time_minutes := 25

  let cleaning_time := vacuum + dusting + mopping + bathroom + windows + (brushing_per_cat * cats)
  let free_time_total := (free_time_hours * 60) + free_time_minutes

  free_time_total - cleaning_time = 55 :=
by
  sorry

end gunther_cleaning_free_time_l501_501992


namespace total_books_gwen_has_l501_501993

-- Definitions based on conditions in part a
def mystery_shelves : ℕ := 5
def picture_shelves : ℕ := 3
def books_per_shelf : ℕ := 4

-- Problem statement in Lean 4
theorem total_books_gwen_has : 
  mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = 32 := by
  -- This is where the proof would go, but we include sorry to skip for now
  sorry

end total_books_gwen_has_l501_501993


namespace ellipse_properties_find_line_l501_501225

noncomputable def ellipse_equation (a b : ℝ) (C : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, C (x, y) ↔ (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def point_on_ellipse (P : ℝ × ℝ) (C : ℝ × ℝ → Prop) : Prop :=
  C P

noncomputable def orthocenter_property 
  (F1 F2 P H : ℝ × ℝ) : Prop :=
  let k_F1H := (H.2 - F1.2) / (H.1 - F1.1)
  let k_PF2 := (F2.2 - P.2) / (F2.1 - P.1)
  k_F1H * k_PF2 = -1

theorem ellipse_properties :
  ∃ (a b : ℝ) (C : ℝ × ℝ → Prop),
    a > b ∧ 
    b > 0 ∧
    ellipse_equation a b C ∧
    point_on_ellipse 
      (frac(2 * sqrt 6) 3, 1) C ∧
    orthocenter_property 
      (-1, 0) 
      (1, 0)
      (frac(2 * sqrt 6) 3, 1) 
      (frac(2 * sqrt 6) 3, -frac 5 3) ∧
    (C = λ p : ℝ × ℝ, p.1^2 / 4 + p.2^2 / 3 = 1) :=
sorry

noncomputable def line_passing_through
  (F2 : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, l (x, y) ↔ y = 2 * (x - F2.1)

theorem find_line :
  ∃ (l : ℝ × ℝ → Prop),
    line_passing_through (1, 0) l ∧
    (l = λ p : ℝ × ℝ, p.2 = 2 * (p.1 - 1)) :=
sorry

end ellipse_properties_find_line_l501_501225


namespace vec_a_vec_b_magnitude_range_l501_501082

variable θ : ℝ 

def vec_a : ℝ × ℝ := (1, Real.sin θ)
def vec_b : ℝ × ℝ := (Real.cos θ, Real.sqrt 3)

theorem vec_a_vec_b_magnitude_range :
  1 ≤ Real.sqrt (((vec_a.1 - vec_b.1)^2) + ((vec_a.2 - vec_b.2)^2)) ∧
  Real.sqrt (((vec_a.1 - vec_b.1)^2) + ((vec_a.2 - vec_b.2)^2)) ≤ 3 :=
sorry

end vec_a_vec_b_magnitude_range_l501_501082


namespace max_balls_fit_in_cube_l501_501086

theorem max_balls_fit_in_cube :
    let radius := 1.5
    let side_length := 8
    let volume_cube := side_length^3
    let volume_ball := (4/3) * Real.pi * radius^3
    ⌊(volume_cube / volume_ball)⌋ = 36 :=
by
  let radius := 1.5
  let side_length := 8
  let volume_cube := side_length^3
  let volume_ball := (4/3) * Real.pi * radius^3
  have h : ⌊(volume_cube / volume_ball)⌋ = 36 := sorry
  exact h

end max_balls_fit_in_cube_l501_501086


namespace yuna_candies_l501_501438

theorem yuna_candies (initial left : ℕ) (h1 : initial = 23) (h2 : left = 7) : initial - left = 16 := 
by
  rw [h1, h2]
  rfl

end yuna_candies_l501_501438


namespace sum_of_squares_xy_l501_501080

theorem sum_of_squares_xy (x y : ℝ) (h₁ : x + y = 10) (h₂ : x^3 + y^3 = 370) : x * y = 21 :=
by
  sorry

end sum_of_squares_xy_l501_501080


namespace decimal_to_fraction_l501_501530

theorem decimal_to_fraction : (0.3 : ℚ) + (0.\overline{45} : ℚ) = 83 / 110 := sorry

end decimal_to_fraction_l501_501530


namespace initial_number_of_persons_l501_501058

variable (N : ℕ)
variable (avg_weight : ℝ)
variable (avg_weight_increase : ℝ)
variable (weight_new_person : ℝ)
variable (weight_old_person : ℝ)

theorem initial_number_of_persons :
  avg_weight_increase = 4 →
  weight_new_person = 97 →
  weight_old_person = 65 →
  N * (avg_weight + avg_weight_increase) - N * avg_weight = weight_new_person - weight_old_person →
  N = 8 :=
by
  intros h1 h2 h3 h4
  have eq1 : N * avg_weight + N * avg_weight_increase - N * avg_weight = weight_new_person - weight_old_person,
  by rw [←h1, ←h2, ←h3] at h4
  simp at eq1
  sorry

end initial_number_of_persons_l501_501058


namespace hawkeye_remaining_money_l501_501272

-- Define the conditions
def cost_per_charge : ℝ := 3.5
def number_of_charges : ℕ := 4
def budget : ℝ := 20

-- Define the theorem to prove the remaining money
theorem hawkeye_remaining_money : 
  budget - (number_of_charges * cost_per_charge) = 6 := by
  sorry

end hawkeye_remaining_money_l501_501272


namespace mixed_grains_in_batch_l501_501317

theorem mixed_grains_in_batch :
  ∀ (total_rice : ℕ) (sample_total : ℕ) (sample_mixed : ℕ),
  total_rice = 1500 → sample_total = 200 → sample_mixed = 20 →
  (total_rice * sample_mixed / sample_total) = 150 :=
by
  intros total_rice sample_total sample_mixed
  intros Htotal_rice Hsample_total Hsample_mixed
  rw [Htotal_rice, Hsample_total, Hsample_mixed]
  simp
  rw [Nat.mul_div_eq_of_eq_div Nat.gcd_eq_right_iff_eq_div]
  exact sorry

end mixed_grains_in_batch_l501_501317


namespace sequence_not_periodic_l501_501145

noncomputable def a : ℕ → ℕ
| 0 => 0
| (2 * n + 1) => 1
| (2 * (2 * n + 1) + 1) => 1
| (2 * (2 * n + 1) + 2) => 0
| (2 * (2 * n + 1) + 3) => 0
| (2 * n) => a n

theorem sequence_not_periodic : ¬∃ T > 0, ∀ n ≥ 1, a (n + T) = a n := by
  sorry

end sequence_not_periodic_l501_501145


namespace largest_circle_radius_l501_501510

noncomputable def largest_inscribed_circle_radius (AB BC CD DA : ℝ) : ℝ :=
  let s := (AB + BC + CD + DA) / 2
  let A := Real.sqrt ((s - AB) * (s - BC) * (s - CD) * (s - DA))
  A / s

theorem largest_circle_radius {AB BC CD DA : ℝ} (hAB : AB = 10) (hBC : BC = 11) (hCD : CD = 6) (hDA : DA = 13)
  : largest_inscribed_circle_radius AB BC CD DA = 3 * Real.sqrt 245 / 10 :=
by
  simp [largest_inscribed_circle_radius, hAB, hBC, hCD, hDA]
  sorry

end largest_circle_radius_l501_501510


namespace collinear_P_B_Q_exists_fixed_point_C_l501_501077

-- Define the required points and conditions
variables (Circle1 Circle2 : ℝ → ℝ → Prop) (A B : ℝ) (P Q : ℝ → ℝ → ℝ)
          (O O' C : ℝ → ℝ → ℝ)

-- Assume circles intersect at A and B
axiom intersect_at_A_and_B: Circle1 A ∧ Circle1 B ∧ Circle2 A ∧ Circle2 B

-- Assume points P and Q travel on respective circles and intersect at A at the same time
axiom P_moves_on_Circle1: ∀ t, Circle1 (P t)
axiom Q_moves_on_Circle2: ∀ t, Circle2 (Q t)
axiom P_and_Q_pass_through_A_simultaneously: ∀ t, P t = A ∧ Q t = A → t = 0

-- Constants defining the circles and point reflections 
axiom O_center_of_Circle1: ∀ x, x ∈ Circle1 → ∃ r, (x - O)^2 = r^2
axiom O'_center_of_Circle2: ∀ x, x ∈ Circle2 → ∃ r, (x - O')^2 = r^2
axiom C_is_reflection_of_A: reflection_perpendicular_bisector O O' A = C

-- Proof goals
theorem collinear_P_B_Q (t : ℝ) :
  is_collinear (P t) B (Q t) := sorry

theorem exists_fixed_point_C:
  ∃ C, ∀ t, distance (P t) C = distance (Q t) C := sorry

end collinear_P_B_Q_exists_fixed_point_C_l501_501077


namespace sum_altitudes_of_triangle_l501_501061

theorem sum_altitudes_of_triangle :
  let x_intercept : ℝ := 60 / 20,
      y_intercept : ℝ := 60 / 3,
      A : ℝ := (1/2) * x_intercept * y_intercept,
      third_altitude : ℝ := 60 / (Real.sqrt (20^2 + 3^2)) in
  x_intercept + y_intercept + third_altitude = 23 + 60 / (Real.sqrt 409) :=
by
  sorry

end sum_altitudes_of_triangle_l501_501061


namespace sets_equality_l501_501989

-- Definitions of sets A, B, and C
def A : Set (List String) := sorry -- Define A as the set of all distinct ways to parenthesize a_1 a_2 a_3 a_4 a_5 with 3 pairs of parentheses.

def B : Set (List (List String)) := sorry -- Define B as the set of all distinct ways to partition a convex hexagon into 4 triangles.

def C : Set (List (List Char)) := sorry -- Define C as the set of all distinct ways to arrange 4 black balls and 4 white balls in a row such that the number of white balls is never less than the number of black balls at any position.

-- Lean theorem statement to prove that the cardinalities of sets A, B, and C are equal.
theorem sets_equality : (A.card = B.card) ∧ (B.card = C.card) := by
  sorry

end sets_equality_l501_501989


namespace filter_probability_l501_501908

noncomputable def letter_probability : ℚ :=
  let p_river := (Nat.choose 5 3 : ℚ)⁻¹
  let p_stone := 6 / (Nat.choose 5 3 : ℚ)
  let p_flight := 6 / (Nat.choose 6 4 : ℚ)
  p_river * p_stone * p_flight

theorem filter_probability :
  letter_probability = 3 / 125 := by
  sorry

end filter_probability_l501_501908


namespace long_diagonal_length_l501_501870

-- Define the lengths of the rhombus sides and diagonals
variables (a b : ℝ) (s : ℝ)
variable (side_length : ℝ)
variable (short_diagonal : ℝ)
variable (long_diagonal : ℝ)

-- Given conditions
def rhombus (side_length: ℝ) (short_diagonal: ℝ) : Prop :=
  side_length = 51 ∧ short_diagonal = 48

-- To prove: length longer diagonal is 90 units
theorem long_diagonal_length (side_length: ℝ) (short_diagonal: ℝ) (long_diagonal: ℝ) :
  rhombus side_length short_diagonal →
  long_diagonal = 90 :=
by
  sorry 

end long_diagonal_length_l501_501870


namespace top_three_black_l501_501132

/-- A definition representing the customized deck of cards for our problem -/
def deck : Type := list card
/-- A card in the deck defined by rank and suit color -/
structure card :=
(rank : ℕ)
(color : string) -- "black", "red" or "blue"

/-- Condition: The customized deck of 52 cards as described -/
def customized_deck (d : deck) : Prop :=
  d.length = 52 ∧
  ∀ (c : card), c ∈ d → 
    (c.color = "black" ∨ c.color = "red" ∨ c.color = "blue") ∧
    ∀ (r : ℕ), 1 ≤ r ∧ r ≤ 13 → 
      4 * (c.color = "black"

/-- The main theorem to prove: Probability that the top three cards are all black -/
theorem top_three_black (d : deck) (h : customized_deck d) :
  probability (top_n_black d 3) = 40 / 1301 :=
sorry

end top_three_black_l501_501132


namespace shorter_meeting_time_in_minutes_l501_501362

theorem shorter_meeting_time_in_minutes (preparation_time_long_meeting planning_time_long_meeting paperwork_time_long_meeting preparation_time_short_meeting : ℝ) 
    (h1 : planning_time_long_meeting = 2)
    (h2 : paperwork_time_long_meeting = 7)
    (h3 : preparation_time_short_meeting = 4.5)
    (h_total_preparation_long : preparation_time_long_meeting = planning_time_long_meeting + paperwork_time_long_meeting)
    (ratio_eq : preparation_time_long_meeting / True = preparation_time_short_meeting / True) : 
    (4.5 * 60 = 270) :=
by
  have h_total_preparation : preparation_time_long_meeting = 9,
  { rw [h1, h2, h_total_preparation_long], },
  have ratio_proportion : preparation_time_long_meeting / True = 4.5 / True,
  from ratio_eq,
  have meeting_time_in_minutes : 4.5 * 60 = 270,
  { norm_num, },
  exact meeting_time_in_minutes,
sorry

end shorter_meeting_time_in_minutes_l501_501362


namespace provisions_last_more_days_l501_501299

noncomputable def provisions_last_800_men := 800 * 20
noncomputable def provisions_last_200_men := 200 * 10
noncomputable def initial_provisions := provisions_last_800_men + provisions_last_200_men

noncomputable def additional_men := 200
noncomputable def additional_rate := 1.5
noncomputable def total_men := 800 + 200
noncomputable def additional_consumption := additional_men * additional_rate
noncomputable def consumption_rate_per_day := total_men + additional_consumption

noncomputable def days_consumed := 10
noncomputable def remaining_provisions := initial_provisions - (consumption_rate_per_day * days_consumed)

noncomputable def additional_supply := 300 * 15
noncomputable def total_remaining_provisions := remaining_provisions + additional_supply

noncomputable def more_days := total_remaining_provisions / consumption_rate_per_day

theorem provisions_last_more_days : more_days = 7.31 := 
sorry

end provisions_last_more_days_l501_501299


namespace smallest_n_l501_501818

theorem smallest_n (n : ℕ) (hn_term : n = 2^a * 5^b) (hn_contains_digits : contains_digits n 9 ∧ contains_digits n 7) : n = 32768 := 
by {
  sorry
}

end smallest_n_l501_501818


namespace outfit_count_l501_501100

def red_shirts := 6
def green_shirts := 4
def blue_shirts := 5
def pairs_of_pants := 7
def red_hats := 9
def green_hats := 7
def blue_hats := 6

theorem outfit_count :
  let outfits := (red_shirts * (green_hats + blue_hats) + 
                  green_shirts * (red_hats + blue_hats) + 
                  blue_shirts * (red_hats + green_hats)) * 
                  pairs_of_pants in
  outfits = 1526 :=
by
  let outfits := (red_shirts * (green_hats + blue_hats) + 
                  green_shirts * (red_hats + blue_hats) + 
                  blue_shirts * (red_hats + green_hats)) * 
                  pairs_of_pants
  show outfits = 1526
  sorry

end outfit_count_l501_501100


namespace tan_half_is_odd_and_period_2pi_l501_501399

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

def has_period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

def tan_half (x : ℝ) : ℝ := Real.tan (x / 2)

theorem tan_half_is_odd_and_period_2pi :
  is_odd tan_half ∧ has_period tan_half (2 * Real.pi) :=
by
  sorry

end tan_half_is_odd_and_period_2pi_l501_501399


namespace perpendicular_vectors_l501_501608

variable (λ : ℝ)
def m : ℝ × ℝ := (λ + 1, 1)
def n : ℝ × ℝ := (λ + 2, 2)

theorem perpendicular_vectors (h : (m λ).1 + (n λ).1 = 0 ∧ (m λ).2 + (n λ).2 = 0) : λ = -3 :=
by
  sorry

end perpendicular_vectors_l501_501608


namespace defective_rate_worker_y_l501_501187

theorem defective_rate_worker_y (d_x d_y : ℝ) (f_y : ℝ) (total_defective_rate : ℝ) :
  d_x = 0.005 → f_y = 0.8 → total_defective_rate = 0.0074 → 
  (0.2 * d_x + f_y * d_y = total_defective_rate) → d_y = 0.008 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end defective_rate_worker_y_l501_501187


namespace average_temperature_l501_501761

theorem average_temperature (T_NY T_Miami T_SD : ℝ) (h1 : T_NY = 80) (h2 : T_Miami = T_NY + 10) (h3 : T_SD = T_Miami + 25) :
  (T_NY + T_Miami + T_SD) / 3 = 95 :=
by
  sorry

end average_temperature_l501_501761


namespace verify_general_term_formula_verify_sequence_sum_l501_501171

noncomputable def general_term_formula (a : ℕ → ℚ) (n : ℕ) : Prop :=
  a n = (n + 1) / 2

noncomputable def b (a : ℕ → ℚ) (n : ℕ) : ℕ → ℚ :=
  λ k, 1 / (k * a k)

noncomputable def sequence_sum (b : ℕ → ℚ) (n : ℕ) : ℚ :=
  ∑ k in Finset.range(n) + 1, b k

noncomputable def correct_sum (n : ℕ) : ℚ :=
  2 * n / (n + 1)

variables (a : ℕ → ℚ) (n : ℕ)
axiom condition1 : a 7 = 4
axiom condition2 : a 19 = 2 * a 9

theorem verify_general_term_formula :
  general_term_formula a n :=
sorry

theorem verify_sequence_sum :
  sequence_sum (b a) n = correct_sum n :=
sorry

end verify_general_term_formula_verify_sequence_sum_l501_501171


namespace sheets_in_total_l501_501075

theorem sheets_in_total (boxes_needed : ℕ) (sheets_per_box : ℕ) (total_sheets : ℕ) 
  (h1 : boxes_needed = 7) (h2 : sheets_per_box = 100) : total_sheets = boxes_needed * sheets_per_box := by
  sorry

end sheets_in_total_l501_501075


namespace calculate_selling_price_l501_501151

theorem calculate_selling_price (cost_price profit : ℕ) (h_cost_price : cost_price = 44) (h_profit : profit = 11) : cost_price + profit = 55 := 
by
  rw [h_cost_price, h_profit]
  rfl

end calculate_selling_price_l501_501151


namespace positive_difference_diagonal_sums_l501_501459

def init_matrix : matrix ℕ ℕ ℕ := 
  ![
    ![1, 2, 3, 4, 5],
    ![6, 7, 8, 9, 10],
    ![11, 12, 13, 14, 15],
    ![16, 17, 18, 19, 20],
    ![21, 22, 23, 24, 25]
  ]

def reversed_matrix : matrix ℕ ℕ ℕ := 
  ![
    ![1, 2, 3, 4, 5],
    ![6, 7, 8, 9, 10],
    ![15, 14, 13, 12, 11],
    ![16, 17, 18, 19, 20],
    ![21, 22, 23, 24, 25]
  ]

def main_diagonal_sum (m : matrix ℕ ℕ ℕ) : ℕ := 
  finset.sum (finset.range 5) (λ i, m i i)

def anti_diagonal_sum (m : matrix ℕ ℕ ℕ) : ℕ := 
  finset.sum (finset.range 5) (λ i, m i (4 - i))

theorem positive_difference_diagonal_sums : 
  |main_diagonal_sum reversed_matrix - anti_diagonal_sum reversed_matrix| = 0 := by
  sorry

end positive_difference_diagonal_sums_l501_501459


namespace find_coordinates_of_tangent_point_l501_501589

noncomputable def coordinates_of_tangent_point : (ℝ × ℝ) :=
  let f := λ x : ℝ, Real.exp x
  let g := λ x : ℝ, x⁻¹
  let df := λ x : ℝ, Real.exp x
  let dg := λ x : ℝ, -x⁻²
  let P := (1, 1) -- Given x > 0, x = 1, y = 1
  P

theorem find_coordinates_of_tangent_point :
  ∃ P : ℝ × ℝ, (λ (x : ℝ), Real.exp x) 0 = 1 ∧
                ∃ x, x > 0 ∧ ∃ P : ℝ × ℝ, P = (1, 1) :=
by
  sorry

end find_coordinates_of_tangent_point_l501_501589


namespace trig_expression_eval_l501_501165

theorem trig_expression_eval :
  (cos (30 * Real.pi / 180) * tan (60 * Real.pi / 180) - cos (45 * Real.pi / 180) ^ 2 + tan (45 * Real.pi / 180)) = 2 :=
by
  have h1 : cos (30 * Real.pi / 180) = (Real.sqrt 3) / 2 := sorry
  have h2 : tan (60 * Real.pi / 180) = Real.sqrt 3 := sorry
  have h3 : cos (45 * Real.pi / 180) = (Real.sqrt 2) / 2 := sorry
  have h4 : tan (45 * Real.pi / 180) = 1 := sorry
  sorry

end trig_expression_eval_l501_501165


namespace rectangle_perimeter_l501_501875

theorem rectangle_perimeter (s : ℕ) (h_perimeter : 4 * s = 160) :
  let rect_length := s,
      rect_width := s / 2,
      rect_perimeter := 2 * (rect_length + rect_width)
  in rect_perimeter = 120 :=
by
  sorry

end rectangle_perimeter_l501_501875


namespace trapezoid_area_l501_501447

theorem trapezoid_area (KN LM : ℝ) (P Q : ℝ) (h₁ : KN = 9) (h₂ : LM = 5)
  (h₃ : LP QN : ℝ) (h₄ : P = LN) (h₅ : Q = (LN + LM) / 2)
  (h₆ : P < Q) (h₇ : KP MQ : ℝ) (h₈ : ∀ x, KP x) (h₉ : ∀ y, MQ y)
  (h₁₀ : QN / LP = 5) : 
  let area := (1/2) * (KN + LM) * (√21) in
  area = 7 * √21 :=
sorry

end trapezoid_area_l501_501447


namespace martha_correct_guess_probability_l501_501357

namespace MarthaGuess

-- Definitions for the conditions
def height_guess_child_accurate : ℚ := 4 / 5
def height_guess_adult_accurate : ℚ := 5 / 6
def weight_guess_tight_clothing_accurate : ℚ := 3 / 4
def weight_guess_loose_clothing_accurate : ℚ := 7 / 10

-- Probabilities of incorrect guesses
def height_guess_child_inaccurate : ℚ := 1 - height_guess_child_accurate
def height_guess_adult_inaccurate : ℚ := 1 - height_guess_adult_accurate
def weight_guess_tight_clothing_inaccurate : ℚ := 1 - weight_guess_tight_clothing_accurate
def weight_guess_loose_clothing_inaccurate : ℚ := 1 - weight_guess_loose_clothing_accurate

-- Combined probability of guessing incorrectly for each case
def incorrect_prob_child_loose : ℚ := height_guess_child_inaccurate * weight_guess_loose_clothing_inaccurate
def incorrect_prob_adult_tight : ℚ := height_guess_adult_inaccurate * weight_guess_tight_clothing_inaccurate
def incorrect_prob_adult_loose : ℚ := height_guess_adult_inaccurate * weight_guess_loose_clothing_inaccurate

-- Total probability of incorrect guesses for all three cases
def total_incorrect_prob : ℚ := incorrect_prob_child_loose * incorrect_prob_adult_tight * incorrect_prob_adult_loose

-- Probability of at least one correct guess
def correct_prob_at_least_once : ℚ := 1 - total_incorrect_prob

-- Main theorem stating the final result
theorem martha_correct_guess_probability : correct_prob_at_least_once = 7999 / 8000 := by
  sorry

end MarthaGuess

end martha_correct_guess_probability_l501_501357


namespace factor_x4_plus_81_l501_501733

theorem factor_x4_plus_81 (x : ℝ) : x^4 + 81 = (x^2 + 6 * x + 9) * (x^2 - 6 * x + 9) :=
by 
  -- The proof is omitted.
  sorry

end factor_x4_plus_81_l501_501733


namespace triangle_sine_equality_l501_501043

theorem triangle_sine_equality {a b c : ℝ} {α β γ : ℝ} 
  (cos_rule : c^2 = a^2 + b^2 - 2 * a * b * Real.cos γ)
  (area : ∃ T : ℝ, T = (1 / 2) * a * b * Real.sin γ)
  (sin_addition_γ : Real.sin (γ + Real.pi / 6) = Real.sin γ * (Real.sqrt 3 / 2) + Real.cos γ * (1 / 2))
  (sin_addition_β : Real.sin (β + Real.pi / 6) = Real.sin β * (Real.sqrt 3 / 2) + Real.cos β * (1 / 2))
  (sin_addition_α : Real.sin (α + Real.pi / 6) = Real.sin α * (Real.sqrt 3 / 2) + Real.cos α * (1 / 2)) :
  c^2 + 2 * a * b * Real.sin (γ + Real.pi / 6) = b^2 + 2 * a * c * Real.sin (β + Real.pi / 6) ∧
  b^2 + 2 * a * c * Real.sin (β + Real.pi / 6) = a^2 + 2 * b * c * Real.sin (α + Real.pi / 6) :=
sorry

end triangle_sine_equality_l501_501043


namespace regular_tickets_sold_l501_501137

variables (S R : ℕ) (h1 : S + R = 65) (h2 : 10 * S + 15 * R = 855)

theorem regular_tickets_sold : R = 41 :=
sorry

end regular_tickets_sold_l501_501137


namespace min_balls_to_guarantee_18_l501_501122

noncomputable def min_balls_needed {red green yellow blue white black : ℕ}
    (h_red : red = 30) 
    (h_green : green = 23) 
    (h_yellow : yellow = 21) 
    (h_blue : blue = 17) 
    (h_white : white = 14) 
    (h_black : black = 12) : ℕ :=
  95

theorem min_balls_to_guarantee_18 {red green yellow blue white black : ℕ}
    (h_red : red = 30) 
    (h_green : green = 23) 
    (h_yellow : yellow = 21) 
    (h_blue : blue = 17) 
    (h_white : white = 14) 
    (h_black : black = 12) :
  min_balls_needed h_red h_green h_yellow h_blue h_white h_black = 95 :=
  by
  -- Placeholder for the actual proof
  sorry

end min_balls_to_guarantee_18_l501_501122


namespace recreation_spending_percentage_eq_180_l501_501660

variable (W : ℝ)

-- Define last week's wages and recreation spending
def last_week_recreation_spending : ℝ := 0.15 * W

-- Define this week's wages
def this_week_wages : ℝ := 0.90 * W

-- Define this week's recreation spending
def this_week_recreation_spending : ℝ := 0.30 * this_week_wages

-- The theorem to prove
theorem recreation_spending_percentage_eq_180 :
  (this_week_recreation_spending W / last_week_recreation_spending W) * 100 = 180 :=
by
  sorry

end recreation_spending_percentage_eq_180_l501_501660


namespace smallest_theta_for_unit_vectors_l501_501005

noncomputable def angle_in_degrees (θ : ℝ) : ℝ := θ * (180 / Real.pi)

theorem smallest_theta_for_unit_vectors
  (a b c : ℝ^3)
  (ha : ‖a‖ = 1)
  (hb : ‖b‖ = 1)
  (hc : ‖c‖ = 1)
  (θ : ℝ)
  (hθab : inner a b = Real.cos θ)
  (hθcaxb : inner c (a × b) = Real.cos θ)
  (hbca : inner b (c × a) = 1 / 3) :
  angle_in_degrees θ = 20.9 :=
by
  sorry

end smallest_theta_for_unit_vectors_l501_501005


namespace sweet_apples_percentage_is_75_l501_501166

noncomputable def percentage_sweet_apples 
  (price_sweet : ℝ) 
  (price_sour : ℝ) 
  (total_apples : ℕ) 
  (total_earnings : ℝ) 
  (percentage_sweet_expr : ℝ) :=
  price_sweet * percentage_sweet_expr + price_sour * (total_apples - percentage_sweet_expr) = total_earnings

theorem sweet_apples_percentage_is_75 :
  percentage_sweet_apples 0.5 0.1 100 40 75 :=
by
  unfold percentage_sweet_apples
  sorry

end sweet_apples_percentage_is_75_l501_501166


namespace sequence_term_l501_501941

theorem sequence_term (a : ℕ → ℝ) (h : ∀ n : ℕ, 0 < n → (∑ i in finset.range n, a (i + 1)) / n = n) :
  a 2023 = 4045 :=
by
  sorry

end sequence_term_l501_501941


namespace inscribed_circle_radius_l501_501128

-- Define the triangle and its properties
structure Triangle :=
  (A B C : ℝ) -- Vertices A, B, and C
  (angleACB : ℝ) -- Angle at C which is the right angle
  (hypotenuseAB : ℝ) -- The hypotenuse AB
  (area : ℝ) -- The area of the triangle

-- Given conditions
def givenTriangle : Triangle := {
  A := 0,
  B := 0,
  C := 0,
  angleACB := pi / 2,
  hypotenuseAB := 9,
  area := 36
}

-- Proving the radius of the inscribed circle
theorem inscribed_circle_radius (t : Triangle) (ht : t = givenTriangle) : 
∃ x : ℝ, x = 4.24 := 
sorry

end inscribed_circle_radius_l501_501128


namespace david_chemistry_marks_l501_501918

theorem david_chemistry_marks (marks_english marks_math marks_physics marks_biology : ℝ)
  (average_marks: ℝ) (marks_english_val: marks_english = 72) (marks_math_val: marks_math = 45)
  (marks_physics_val: marks_physics = 72) (marks_biology_val: marks_biology = 75)
  (average_marks_val: average_marks = 68.2) : 
  ∃ marks_chemistry : ℝ, (marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) / 5 = average_marks ∧ 
    marks_chemistry = 77 := 
by
  sorry

end david_chemistry_marks_l501_501918


namespace find_m_plus_n_l501_501556

variable (x n m : ℝ)

def condition : Prop := (x + 5) * (x + n) = x^2 + m * x - 5

theorem find_m_plus_n (hnm : condition x n m) : m + n = 3 := 
sorry

end find_m_plus_n_l501_501556


namespace median_of_extended_sequence_l501_501910

-- Define the problem conditions
def sequence (n : ℕ) : list ℕ := (List.range' 1 (n + 1)).bind (λ i, List.replicate i i)
def extended_sequence : list ℕ := sequence 150 ++ [1000]

-- Define the property to be proved
theorem median_of_extended_sequence : 
  (median ((extended_sequence).qsort (≤)) = 106) := 
  sorry

end median_of_extended_sequence_l501_501910


namespace cos_theta_of_unit_circle_point_l501_501251

/-- Given point P(-3/5, 4/5) on the unit circle, prove that the cosine of the angle θ, whose 
vertex is at the origin and initial side is the positive x-axis, and whose terminal side 
passes through P, is -3/5. -/
theorem cos_theta_of_unit_circle_point 
  (P : ℝ × ℝ) (hP : P = (-3 / 5, 4 / 5)) (h_unit_circle : P.1 ^ 2 + P.2 ^ 2 = 1) : 
  ∃ θ : ℝ, real.cos θ = P.1 :=
by
  use (real.angle_of_point P)
  rw [hP, real.angle_of_point]
  sorry

end cos_theta_of_unit_circle_point_l501_501251


namespace remainder_division_l501_501363

theorem remainder_division (exists_quotient : ∃ q r : ℕ, r < 5 ∧ N = 5 * 5 + r)
    (exists_quotient_prime : ∃ k : ℕ, N = 11 * k + 3) :
  ∃ r : ℕ, r = 0 ∧ N % 5 = r := 
sorry

end remainder_division_l501_501363


namespace find_value_of_m_l501_501398

def ellipse_condition (x y : ℝ) (m : ℝ) : Prop :=
  x^2 + m * y^2 = 1

theorem find_value_of_m (m : ℝ) 
  (h1 : ∀ (x y : ℝ), ellipse_condition x y m)
  (h2 : ∀ a b : ℝ, (a^2 = 1/m ∧ b^2 = 1) ∧ (a = 2 * b)) : 
  m = 1/4 :=
by
  sorry

end find_value_of_m_l501_501398


namespace angle_GAB_90_degrees_l501_501773

noncomputable theory
open_locale classical

variables {ω₁ ω₂ ω₃ : Type} [circle ω₁] [circle ω₂] [circle ω₃]
variables (D A E F B G : Point)

-- Conditions
-- 1. Three circles pass through common point D
axiom circles_pass_through_D : D ∈ ω₁ ∧ D ∈ ω₂ ∧ D ∈ ω₃

-- 2. A is the intersection of ω₁ and ω₃
axiom A_intersection : A ∈ ω₁ ∧ A ∈ ω₃

-- 3. E is the intersection of ω₃ and ω₂
axiom E_intersection : E ∈ ω₃ ∧ E ∈ ω₂

-- 4. F is the intersection of ω₂ and ω₁
axiom F_intersection : F ∈ ω₂ ∧ F ∈ ω₁

-- 5. ω₃ passes through the center B of the circle ω₂
axiom B_center_of_ω₂ : ∃ (c : Point), is_center c ω₂ ∧ B ∈ ω₃

-- 6. The line EF intersects ω₁ a second time at G
axiom line_EF_intersects_ω₁_at_G : line_through E F ∩ ω₁ = {E, G}

-- Theorem statement
theorem angle_GAB_90_degrees : ∠ G A B = 90 :=
begin
  sorry
end

end angle_GAB_90_degrees_l501_501773


namespace wendy_wait_time_l501_501084

theorem wendy_wait_time 
  (facial_products : ℕ)
  (additional_makeup_time : ℕ)
  (total_time : ℕ)
  (product_wait_time : ℝ)
  (h1 : facial_products = 5)
  (h2 : additional_makeup_time = 30)
  (h3 : total_time = 55)
  (h4 : product_wait_time = (total_time - additional_makeup_time) / (facial_products - 1)) :
  product_wait_time = 6.25 :=
by
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end wendy_wait_time_l501_501084


namespace correct_algorithm_description_l501_501437

def conditions_about_algorithms (desc : String) : Prop :=
  (desc = "A" → false) ∧
  (desc = "B" → false) ∧
  (desc = "C" → true) ∧
  (desc = "D" → false)

theorem correct_algorithm_description : ∃ desc : String, 
  conditions_about_algorithms desc :=
by
  use "C"
  unfold conditions_about_algorithms
  simp
  sorry

end correct_algorithm_description_l501_501437


namespace checker_in_central_cell_l501_501030

theorem checker_in_central_cell
  (n : ℕ) (hn : n = 25)
  (board : Fin n.succ × Fin n.succ → Prop)
  (center : Fin n.succ × Fin n.succ := (⟨13, Nat.succ_lt_succ Nat.succ_lt_succ⟩, ⟨13, Nat.succ_lt_succ Nat.succ_lt_succ⟩))
  (board_symmetric_wrt_diagonals : 
    ∀ (i j : Fin n.succ), board (i, j) ↔ board (j, i) ∧ board (i, j) ↔ board (⟨n-j, sorry⟩, ⟨n-i, sorry⟩))
  (board_filled : ∃ checker_positions : Fin n.succ → Fin n.succ × Fin n.succ, 
    (∀ i, checker_positions i ≠ ⟨13, 13⟩ ↔ ∃ j, checker_positions i = ⟨j, ⟨n - j, sorry⟩⟩)) :
  (board center) :=
by
  sorry

end checker_in_central_cell_l501_501030


namespace total_arrangements_excluding_zhang_for_shooting_event_l501_501416

theorem total_arrangements_excluding_zhang_for_shooting_event
  (students : Fin 5) 
  (events : Fin 3)
  (shooting : events ≠ 0) : 
  ∃ arrangements, arrangements = 48 := 
sorry

end total_arrangements_excluding_zhang_for_shooting_event_l501_501416


namespace sum_mod_p_l501_501956

theorem sum_mod_p (p : ℕ) [Fact p.Prime] (x y : ℤ) :
  let S := (Finset.range p).sum (λ k, x^k * y^(p - 1 - k))
  if x % p = y % p then S % p = 0 else S % p = 1 := by
  sorry

end sum_mod_p_l501_501956


namespace percent_increase_of_income_l501_501612

theorem percent_increase_of_income (original_income new_income : ℝ) 
  (h1 : original_income = 120) (h2 : new_income = 180) :
  ((new_income - original_income) / original_income) * 100 = 50 := 
by 
  rw [h1, h2]
  norm_num

end percent_increase_of_income_l501_501612


namespace Jasper_height_in_10_minutes_l501_501651

noncomputable def OmarRate : ℕ := 240 / 12
noncomputable def JasperRate : ℕ := 3 * OmarRate
noncomputable def JasperHeight (time: ℕ) : ℕ := JasperRate * time

theorem Jasper_height_in_10_minutes :
  JasperHeight 10 = 600 :=
by
  sorry

end Jasper_height_in_10_minutes_l501_501651


namespace find_b_l501_501721

def f (x : ℝ) : ℝ := x / 4 + 2
def g (x : ℝ) : ℝ := 5 - 2 * x

theorem find_b (b : ℝ) (h : f(g(b)) = 4) : b = -3 / 2 := by
  sorry

end find_b_l501_501721


namespace jason_home_distance_l501_501656

theorem jason_home_distance :
  let v1 := 60 -- speed in miles per hour
  let t1 := 0.5 -- time in hours
  let d1 := v1 * t1 -- distance covered in first part of the journey
  let v2 := 90 -- speed in miles per hour for the second part
  let t2 := 1.0 -- remaining time in hours
  let d2 := v2 * t2 -- distance covered in second part of the journey
  let total_distance := d1 + d2 -- total distance to Jason's home
  total_distance = 120 := 
by
  simp only
  sorry

end jason_home_distance_l501_501656


namespace sum_a_n_first_100_terms_eq_100_l501_501679

def f (n : ℕ) : ℤ := (-1)^(n-1) * n^2

def a (n : ℕ) := f n + f (n + 1)

theorem sum_a_n_first_100_terms_eq_100 : (Finset.range 100).sum (λ n, a (n + 1)) = 100 := 
by
  sorry

end sum_a_n_first_100_terms_eq_100_l501_501679


namespace last_digit_is_zero_last_ten_digits_are_zero_l501_501104

-- Condition: The product includes a factor of 10
def includes_factor_of_10 (n : ℕ) : Prop :=
  ∃ k, n = k * 10

-- Conclusion: The last digit of the product must be 0
theorem last_digit_is_zero (n : ℕ) (h : includes_factor_of_10 n) : 
  n % 10 = 0 :=
sorry

-- Condition: The product includes the factors \(5^{10}\) and \(2^{10}\)
def includes_10_to_the_10 (n : ℕ) : Prop :=
  ∃ k, n = k * 10^10

-- Conclusion: The last ten digits of the product must be 0000000000
theorem last_ten_digits_are_zero (n : ℕ) (h : includes_10_to_the_10 n) : 
  n % 10^10 = 0 :=
sorry

end last_digit_is_zero_last_ten_digits_are_zero_l501_501104


namespace part_a_part_b_part_c_l501_501847

variables {A B C A1 B1 : Type} [triangle : triangle A B C] [triangle : altitude A A1 B1]

-- Assume the conditions of the problem
def is_acute (α : Type) : Prop := true -- Placeholder, define acute triangle properly
def altitude (α β γ : Type) : Prop := true -- Placeholder, define altitude property properly

-- Prove the similarity of triangles AA1C and BB1C
theorem part_a (h : triangle A B C) (h_acute : is_acute A B C) (h1 : altitude A A1 B1) :
  triangle.similar (A A1 C) (B B1 C) :=
sorry

-- Prove the similarity between triangles ABC and A1B1C
theorem part_b (h : triangle A B C) (h_acute : is_acute A B C) (h1 : altitude A A1 B1) :
  triangle.similar (A B C) (A1 B1 C) :=
sorry

-- Prove the similarity ratio between A1B1C and ABC with given ∠C
theorem part_c (h : triangle A B C) (h_acute : is_acute A B C) (h1 : altitude A A1 B1) (γ : ℝ) :
  triangle.similarity_ratio (A1 B1 C) (A B C) = abs (cos γ) :=
sorry

end part_a_part_b_part_c_l501_501847


namespace boat_speed_in_still_water_l501_501850

theorem boat_speed_in_still_water
  (V_s : ℝ) (t : ℝ) (d : ℝ) (V_b : ℝ)
  (h_stream_speed : V_s = 4)
  (h_travel_time : t = 7)
  (h_distance : d = 196)
  (h_downstream_speed : d / t = V_b + V_s) :
  V_b = 24 :=
by
  sorry

end boat_speed_in_still_water_l501_501850


namespace trapezoid_area_l501_501465

-- Define a trapezoid with given sides and circumscribed circle properties.
def circumscribed_trapezoid (AB CD AD : ℝ) (AC CB BD DA : ℝ) : Prop :=
  2*AD - (AB + CD) = 0 -- AD is considered as the midpoint due to the circle properties

-- Lean definition of the problem with conditions
theorem trapezoid_area {AB CD AC : ℝ} (H1 : AB = 3/4) (H2 : AC = 1) :
  circumscribed_trapezoid AB CD AC (1-(3/4)) (1-1) AC = true →
  AB * ( AC * (AD - 2*(AB + CD)) ) = 12 / 25 :=
sorry

end trapezoid_area_l501_501465


namespace question_1_question_2_question_3_l501_501593

open Real

noncomputable def f (x : ℝ) (a b : ℝ) := x^2 + b*x - a * log x

theorem question_1 {a : ℝ} (h₁ : a ≠ 0) :
  (∀ x : ℝ, 0 < x → a ≤ 0 → (2 * x^2 - a) / x > 0) ∧
  (a > 0 → ∀ x : ℝ, 0 < x → (x < sqrt (a / 2) → (2 * x^2 - a) / x < 0) ∧ 
                 (x > sqrt (a / 2) → (2 * x^2 - a) / x > 0)) :=
by sorry

theorem question_2 {a b : ℝ} 
  (h₁ : 2 * 2 - a / 2 + b = 0) (h₂ : 1 + b = 0) : a + b = 5 :=
by sorry

theorem question_3 {a b : ℝ} 
  (hx : ∀ b : ℝ, b ∈ Icc (-2 : ℝ) (-1 : ℝ) → ∃ x : ℝ, x ∈ Ioo 1 exp 1 ∧ (x^2 + b * x - a * log x < 0)) : a > 1 :=
by sorry

end question_1_question_2_question_3_l501_501593


namespace minimum_clients_repunits_l501_501449

theorem minimum_clients_repunits (n m k : ℕ) (h_n : n > 1) (h_banks : ∀ x, x > 10)
  (amt_per_client : ℕ) (h_amt_repunits : ∀ i, amt_per_client = (nat.replicate i 1).foldr (λ a b, 10 * b + a) 0 ∧ amt_per_client > 10) 
  (total_amount : ℕ) (h_total_repunits : total_amount = (nat.replicate n 1).foldr (λ a b, 10 * b + a) 0) 
  (h_eq : total_amount = amt_per_client * m) 
  (h_cond : ∃ k, k > 1 ∧ k < n ∧ total_amount = amt_per_client * (nat.replicate k 1).foldr (λ a b, 10 * b + a) 0) : 
  n = 101 :=
sorry

end minimum_clients_repunits_l501_501449


namespace original_radius_eq_n_div_3_l501_501390

theorem original_radius_eq_n_div_3 (r n : ℝ) (h : (r + n)^2 = 4 * r^2) : r = n / 3 :=
by
  sorry

end original_radius_eq_n_div_3_l501_501390


namespace antonio_age_in_months_l501_501326

-- Definitions based on the conditions
def is_twice_as_old (isabella_age antonio_age : ℕ) : Prop :=
  isabella_age = 2 * antonio_age

def future_age (current_age months_future : ℕ) : ℕ :=
  current_age + months_future

-- Given the conditions
variables (isabella_age antonio_age : ℕ)
variables (future_age_18months target_age : ℕ)

-- Conditions
axiom condition1 : is_twice_as_old isabella_age antonio_age
axiom condition2 : future_age_18months = 18
axiom condition3 : target_age = 10 * 12

-- Assertion that we need to prove
theorem antonio_age_in_months :
  ∃ (antonio_age : ℕ), future_age isabella_age future_age_18months = target_age → antonio_age = 51 :=
by
  sorry

end antonio_age_in_months_l501_501326


namespace balls_distribution_ways_l501_501186

theorem balls_distribution_ways : 
  ∃ (ways : ℕ), ways = 15 := by
  sorry

end balls_distribution_ways_l501_501186


namespace pyramid_volume_24_l501_501376

theorem pyramid_volume_24 
  (PA PB AD AB AC BD: ℝ)
  (AB_eq_5 : AB = 5) 
  (PB_eq_13 : PB = 13)
  (AC_eq_7 : AC = 7)
  (BD_eq_4 : BD = 4)
  (PA_perp_AD : ∃ A B D, PA ⊥ AD)
  (PA_perp_AB : ∃ A B, PA ⊥ AB) :
  let PARALLELOGRAM_AREA := (1 / 2) * real.sqrt (2 * AC ^ 2 + 2 * BD ^ 2 - AB ^ 2 - (PA - AB) ^ 2)
  let HEIGHT := real.sqrt (PB ^ 2 - AB ^ 2)
  let VOLUME := (1 / 3) * PARALLELOGRAM_AREA * HEIGHT in
  VOLUME = 24 :=
by
  -- We will fill in the proof later
  sorry

end pyramid_volume_24_l501_501376


namespace balls_into_boxes_arrangement_l501_501368

theorem balls_into_boxes_arrangement : ∃ n, n = 10 ∧ 
  (∑ x in {1,2,3}, x ≥ 1 ∧ x ∈ { 
      arrangements | multiset.card arrangements = 6 ∧ arrangements.card ≤ 6 ∧ 
      multiset.card arrangements ≥ 1 ∧  arrangements.card ≥ 3 } = 1) :=
begin
  use 10,
  sorry,
end

end balls_into_boxes_arrangement_l501_501368


namespace pure_imaginary_solved_l501_501015

theorem pure_imaginary_solved {m : ℝ} (H : (m^2 + m - 2 : ℂ) = (0 : ℂ) + (m^2 - 1 : ℂ) * complex.I) : m = -2 :=
sorry

end pure_imaginary_solved_l501_501015


namespace average_temperature_is_95_l501_501755

noncomputable def tempNY := 80
noncomputable def tempMiami := tempNY + 10
noncomputable def tempSD := tempMiami + 25
noncomputable def avg_temp := (tempNY + tempMiami + tempSD) / 3

theorem average_temperature_is_95 :
  avg_temp = 95 :=
by
  sorry

end average_temperature_is_95_l501_501755


namespace book_arrangement_l501_501615

theorem book_arrangement :
  let total_books := 7
  let identical_books := 3
  let unique_books := 4
  let block_books := 2
  let reduced_books := total_books - (block_books - 1)
  (Nat.fact (reduced_books) / Nat.fact (identical_books)) * Nat.fact (block_books) = 240 :=
by
  sorry

end book_arrangement_l501_501615


namespace Mr_A_financial_outcome_l501_501358

def home_worth : ℝ := 200000
def profit_percent : ℝ := 0.15
def loss_percent : ℝ := 0.05

def selling_price := (1 + profit_percent) * home_worth
def buying_price := (1 - loss_percent) * selling_price

theorem Mr_A_financial_outcome : 
  selling_price - buying_price = 11500 :=
by
  sorry

end Mr_A_financial_outcome_l501_501358


namespace fraction_division_l501_501898

-- Define the fractions and the operation result.
def complex_fraction := 5 / (8 / 15)
def result := 75 / 8

-- State the theorem indicating that these should be equal.
theorem fraction_division :
  complex_fraction = result :=
  by
  sorry

end fraction_division_l501_501898


namespace weather_forecast_probability_l501_501411

noncomputable def binomial {n : ℕ} (p : ℝ) (k : ℕ) : ℝ :=
  (nat.choose n k : ℝ) * p^k * (1-p)^(n-k)

theorem weather_forecast_probability :
  binomial 3 0.8 2 = 0.384 :=
by sorry

end weather_forecast_probability_l501_501411


namespace similar_monochromatic_triangles_l501_501188

theorem similar_monochromatic_triangles (n : ℤ) (h1 : Odd n) (h2 : 3 ≤ n) :
  ∀ (color : ℝ × ℝ → Prop) (A B : set (ℝ × ℝ)),
  (∀ p : ℝ × ℝ, color p ∨ ¬ color p) →
  (∀ p : ℝ × ℝ, p ∈ A ↔ color p) →
  (∀ p : ℝ × ℝ, p ∈ B ↔ ¬ color p) →
  ∃ Δ₁ Δ₂ : finset (ℝ × ℝ),
    (Δ₁.card = 3 ∧ Δ₂.card = 3) ∧
    (¬ ∃ p₁ p₂ p₃ ∈ Δ₁, p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    |p₁.1 - p₂.1| / |p₁.1 - p₃.1| = n ∧ |p₁.2 - p₂.2| / |p₁.2 - p₃.2| = n) ∧
    (¬ ∃ p₁ p₂ p₃ ∈ Δ₂, p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    |p₁.1 - p₂.1| / |p₁.1 - p₃.1| = n ∧ |p₁.2 - p₂.2| / |p₁.2 - p₃.2| = n).
Proof
  sorry

end similar_monochromatic_triangles_l501_501188


namespace number_A_properties_l501_501948

theorem number_A_properties :
  let odd_numbers_from_1_to_103 := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103] in
  let A := odd_numbers_from_1_to_103.foldl (λ acc n, acc * 10 ^ (Nat.floor (Real.log10 n) + 1) + n) 0 in
  (number_of_digits : Nat) ∧ (remainder_mod_9 : Nat) :=
  by 
    let number_of_digits := 5 + 90 + 6
    have number_of_digits_correct : number_of_digits = 101 := sorry
    let sum_of_digits := 25 + 450 + 6
    let remainder_mod_9 := sum_of_digits % 9
    have remainder_mod_9_correct : remainder_mod_9 = 4 := sorry
    exact ⟨number_of_digits_correct, remainder_mod_9_correct⟩

end number_A_properties_l501_501948


namespace largest_value_frac_l501_501582

theorem largest_value_frac (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 3) :
  ∃ k, k = 1 + 2 * y / x ∧ k = -1 / 5 :=
by {
  use 1 + 2 * y / x,
  split,
  {
    sorry,
  },
  {
    obtain ⟨hx_left, hx_right⟩ := hx,
    obtain ⟨hy_left, hy_right⟩ := hy,
    have hx_neg : x < 0 := by { linarith, },
    have hy_pos : y > 0 := by { linarith, },
    have min_x : x = -5 := by { sorry, }, -- We would justify this by arguing that -5 is indeed the min within -5 ≤ x ≤ -3
    have max_y : y = 3 := by { sorry, }, -- We argue that 3 is indeed the max within 1 ≤ y ≤ 3
    rw [min_x, max_y],
    simp,
  },
}

end largest_value_frac_l501_501582


namespace union_of_M_N_l501_501988

-- Definitions of sets M and N
def M : Set ℕ := {0, 1}
def N : Set ℕ := {1, 2}

-- The theorem to prove
theorem union_of_M_N : M ∪ N = {0, 1, 2} :=
  by sorry

end union_of_M_N_l501_501988


namespace nico_min_right_turns_l501_501696

theorem nico_min_right_turns (P Q : Point) (hPQ : Q = P.shift_right) 
  : ∃ n, n = 4 ∧ Nico_reaches(P, Q, n, facing_north, can_only_turn_right) :=
sorry

end nico_min_right_turns_l501_501696


namespace albert_matured_amount_l501_501441

def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem albert_matured_amount :
  compound_interest 6500 0.065 1 2 = 7359.46 :=
by sorry

end albert_matured_amount_l501_501441


namespace mean_proportional_between_64_and_81_l501_501937

theorem mean_proportional_between_64_and_81 :
  (real.sqrt (64 * 81) = 72) :=
by {
  sorry,
}

end mean_proportional_between_64_and_81_l501_501937


namespace problem1_no_solution_problem2_solution_l501_501717

theorem problem1_no_solution (x : ℝ) 
  (h : (5*x - 4)/(x - 2) = (4*x + 10)/(3*x - 6) - 1) : false :=
by
  -- The original equation turns out to have no solution
  sorry

theorem problem2_solution (x : ℝ) 
  (h : 1 - (x - 2)/(2 + x) = 16/(x^2 - 4)) : x = 6 :=
by
  -- The equation has a solution x = 6
  sorry

end problem1_no_solution_problem2_solution_l501_501717


namespace find_length_of_AB_hyperbola_asymptote_circle_l501_501982

def hyperbola_asymptote_intersects_circle (a b : ℝ) (ha : 0 < a) (hb : b = 2 * a) : Prop :=
∀ (x y : ℝ), ( x = 2 ) → ( y = 3 ) → ∃ (A B : ℝ × ℝ),
  let line := 2 * x in 
  ((A = (x, 2 * x)) ∧ (B = (x, -2 * x))) ∧
  (dist A B = (4 * Real.sqrt 5) / 5)

theorem find_length_of_AB_hyperbola_asymptote_circle :
  hyperbola_asymptote_intersects_circle 1 2
  (by norm_num) 
  (by norm_num)
  sorry

end find_length_of_AB_hyperbola_asymptote_circle_l501_501982


namespace zytron_midway_distance_l501_501407

noncomputable def zytron_distance_from_star : ℝ :=
  let perihelion := 3
  let aphelion := 8
  let major_axis_length := perihelion + aphelion
  let semi_major_axis := major_axis_length / 2
  semi_major_axis

theorem zytron_midway_distance :
  (zytron_distance_from_star) = 5.5 := by
  let perihelion : ℝ := 3
  let aphelion : ℝ := 8
  let major_axis_length := perihelion + aphelion
  let semi_major_axis := major_axis_length / 2
  show semi_major_axis = 5.5
  calc
    semi_major_axis = (3 + 8) / 2 := by sorry -- Proof steps would go here
    ... = 11 / 2 := by sorry
    ... = 5.5 := by sorry

end zytron_midway_distance_l501_501407


namespace Eunji_higher_than_Yoojung_l501_501099

-- Define floors for Yoojung and Eunji
def Yoojung_floor: ℕ := 17
def Eunji_floor: ℕ := 25

-- Assert that Eunji lives on a higher floor than Yoojung
theorem Eunji_higher_than_Yoojung : Eunji_floor > Yoojung_floor :=
  by
    sorry

end Eunji_higher_than_Yoojung_l501_501099


namespace min_distance_origin_to_line_l501_501038

theorem min_distance_origin_to_line : 
  (∃ (x y : ℝ), 2 * x - y + 1 = 0) → 
  ∀ (O : ℝ × ℝ), O = (0, 0) → 
  ∃ (d : ℝ), d = abs (2 * 0 - 0 + 1) / sqrt (2^2 + (-1)^2) ∧ d = sqrt 5 / 5 := 
by 
  intros line_condition O_origin
  use sqrt 5 / 5
  sorry

end min_distance_origin_to_line_l501_501038


namespace investment_ratio_l501_501879

-- Definitions of all the conditions
variables (A B C profit b_share: ℝ)

-- Conditions based on the provided problem
def condition1 (n : ℝ) : Prop := A = n * B
def condition2 : Prop := B = (2 / 3) * C
def condition3 : Prop := profit = 4400
def condition4 : Prop := b_share = 800

-- The theorem we want to prove
theorem investment_ratio (n : ℝ) :
  (condition1 A B n) ∧ (condition2 B C) ∧ (condition3 profit) ∧ (condition4 b_share) → A / B = 3 :=
by
  sorry

end investment_ratio_l501_501879


namespace rowing_speed_still_water_l501_501473

theorem rowing_speed_still_water (v r : ℕ) (h1 : r = 18) (h2 : 1 / (v - r) = 3 * (1 / (v + r))) : v = 36 :=
by sorry

end rowing_speed_still_water_l501_501473


namespace ratio_difference_l501_501106

theorem ratio_difference (x : ℕ) (h : (2 * x + 4) * 7 = (3 * x + 4) * 5) : 3 * x - 2 * x = 8 := 
by sorry

end ratio_difference_l501_501106


namespace students_without_A_l501_501301

theorem students_without_A 
  (total_students : ℕ) 
  (A_in_literature : ℕ) 
  (A_in_science : ℕ) 
  (A_in_both : ℕ) 
  (h_total_students : total_students = 35)
  (h_A_in_literature : A_in_literature = 10)
  (h_A_in_science : A_in_science = 15)
  (h_A_in_both : A_in_both = 5) :
  total_students - (A_in_literature + A_in_science - A_in_both) = 15 :=
by {
  sorry
}

end students_without_A_l501_501301


namespace root_in_interval_l501_501618

def f (x : ℝ) : ℝ := -x^3 - 3 * x + 5

theorem root_in_interval : ∃ x₀ ∈ Ioo 1 2, f x₀ = 0 :=
begin
  sorry
end

end root_in_interval_l501_501618


namespace water_level_lowered_l501_501766

noncomputable def water_level_lowering_inches : ℝ :=
let volume_gallons := 4687.5 in
let gallon_to_cubic_feet := 7.48052 in
let pool_length := 50 in
let pool_width := 25 in
let cubic_feet_volume := volume_gallons / gallon_to_cubic_feet in
let pool_area := pool_length * pool_width in
let lowering_feet := cubic_feet_volume / pool_area in
lowering_feet * 12

theorem water_level_lowered : water_level_lowering_inches = 6.012 :=
by
  sorry

end water_level_lowered_l501_501766


namespace five_alpha_plus_two_beta_is_45_l501_501004

theorem five_alpha_plus_two_beta_is_45
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (tan_α : Real.tan α = 1 / 7) 
  (tan_β : Real.tan β = 3 / 79) :
  5 * α + 2 * β = π / 4 :=
by
  sorry

end five_alpha_plus_two_beta_is_45_l501_501004


namespace population_increase_rate_l501_501742

theorem population_increase_rate (persons : ℕ) (minutes : ℕ) (seconds_per_person : ℕ) 
  (h1 : persons = 240) 
  (h2 : minutes = 60) 
  (h3 : seconds_per_person = (minutes * 60) / persons) 
  : seconds_per_person = 15 :=
by 
  sorry

end population_increase_rate_l501_501742


namespace yellow_chips_are_one_l501_501444

-- Definitions based on conditions
def yellow_chip_points : ℕ := 2
def blue_chip_points : ℕ := 4
def green_chip_points : ℕ := 5

variables (Y B G : ℕ)

-- Given conditions
def point_product_condition : Prop := (yellow_chip_points^Y * blue_chip_points^B * green_chip_points^G = 16000)
def equal_blue_green : Prop := (B = G)

-- Theorem to prove the number of yellow chips
theorem yellow_chips_are_one (Y B G : ℕ) (hprod : point_product_condition Y B G) (heq : equal_blue_green B G) : Y = 1 :=
by {
    sorry -- Proof omitted
}

end yellow_chips_are_one_l501_501444


namespace sum_q_p_values_l501_501260

def p (x : ℤ) : ℤ := x^2 - 4

def q (x : ℤ) : ℤ := -abs x

theorem sum_q_p_values : 
  (q (p (-3)) + q (p (-2)) + q (p (-1)) + q (p (0)) + q (p (1)) + q (p (2)) + q (p (3))) = -20 :=
by
  sorry

end sum_q_p_values_l501_501260


namespace polynomial_symmetry_and_formula_l501_501064

noncomputable def p : ℕ → (ℝ → ℝ)
| 0 := λ x, 1
| (n + 1) := λ x, x * p n x + p n (a * x)

theorem polynomial_symmetry_and_formula (n : ℕ) (x : ℝ) :
  p n x = x ^ n * p n (1 / x) ∧
  p n x = ∑ k in finset.range (n + 1), (∏ i in finset.range k, (a ^ (n - i) - 1)) / (∏ i in finset.range k, (a ^ i - 1)) * x ^ k :=
sorry

end polynomial_symmetry_and_formula_l501_501064


namespace sam_spent_136_96_l501_501714

def glove_original : Real := 35
def glove_discount : Real := 0.20
def baseball_price : Real := 15
def bat_original : Real := 50
def bat_discount : Real := 0.10
def cleats_price : Real := 30
def cap_price : Real := 10
def tax_rate : Real := 0.07

def total_spent (glove_original : Real) (glove_discount : Real) (baseball_price : Real) (bat_original : Real) (bat_discount : Real) (cleats_price : Real) (cap_price : Real) (tax_rate : Real) : Real :=
  let glove_price := glove_original - (glove_discount * glove_original)
  let bat_price := bat_original - (bat_discount * bat_original)
  let total_before_tax := glove_price + baseball_price + bat_price + cleats_price + cap_price
  let tax_amount := total_before_tax * tax_rate
  total_before_tax + tax_amount

theorem sam_spent_136_96 :
  total_spent glove_original glove_discount baseball_price bat_original bat_discount cleats_price cap_price tax_rate = 136.96 :=
sorry

end sam_spent_136_96_l501_501714


namespace min_postage_cost_l501_501409

noncomputable def postage_cost (weight : ℕ) : ℚ :=
  if weight ≤ 100 then
    0.8 * (if weight < 20 then 1 else (weight + 19) / 20)
  else
    4 + 2 * ((weight - 100 + 99) / 100)

def total_cost (x : ℕ) : ℚ :=
  postage_cost (12 * x + 4) + postage_cost (12 * (11 - x) + 4)

theorem min_postage_cost :
  ∃ x, total_cost x = 5.6 ∧ (∀ y, total_cost y ≥ 5.6) :=
by
  sorry

end min_postage_cost_l501_501409


namespace poly_division_l501_501524

noncomputable def A := 1
noncomputable def B := 3
noncomputable def C := 2
noncomputable def D := -1

theorem poly_division :
  (∀ x : ℝ, x ≠ -1 → (x^3 + 4*x^2 + 5*x + 2) / (x+1) = x^2 + 3*x + 2) ∧
  (A + B + C + D = 5) :=
by
  sorry

end poly_division_l501_501524


namespace decimal_place_of_fraction_l501_501819

theorem decimal_place_of_fraction :
  let sequence : List ℕ := [7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7]
  (17 : ℕ) -- length of the repeating sequence 
  (n : ℕ) -- position we are interested in
  (m : ℕ) -- sequence position resulting from modulo calculation
  in (n = 250) → (n % 17 = m) → (m = 11) → (sequence.get? m = some 3) :=
by
  intros sequence len n m h_n h_mod h_m
  sorry

end decimal_place_of_fraction_l501_501819


namespace minimum_knights_l501_501768

/-!
Problem: There are 1001 people seated around a round table. Each person is either a knight (always tells the truth) or a liar (always lies). Next to each knight, there is exactly one liar, and next to each liar, there is exactly one knight. Prove that the minimum number of knights is 502.
-/

def person := Type
def is_knight (p : person) : Prop := sorry
def is_liar (p : person) : Prop := sorry

axiom round_table (persons : list person) : (∀ (p : person),
  (is_knight p → (∃! q : person, is_liar q ∧ (q = list.nth_le persons ((list.index_of p persons + 1) % 1001) sorry ∨ q = list.nth_le persons ((list.index_of p persons - 1 + 1001) % 1001) sorry))) ∧
  (is_liar p → (∃! k : person, is_knight k ∧ (k = list.nth_le persons ((list.index_of p persons + 1) % 1001) sorry ∨ k = list.nth_le persons ((list.index_of p persons - 1 + 1001) % 1001) sorry))))

theorem minimum_knights (persons : list person) (h : persons.length = 1001) : 
  (∃ (knights : list person), (∀ k ∈ knights, is_knight k) ∧ (∀ l ∉ knights, is_liar l) ∧ knights.length = 502) :=
sorry

end minimum_knights_l501_501768


namespace volume_of_prism_l501_501428

theorem volume_of_prism 
  (a b c : ℝ) 
  (h₁ : a * b = 51) 
  (h₂ : b * c = 52) 
  (h₃ : a * c = 53) 
  : (a * b * c) = 374 :=
by sorry

end volume_of_prism_l501_501428


namespace lottery_probability_l501_501402

theorem lottery_probability :
  let megaBallProb := (1 : ℚ) / 30
  let winnerBallsProb := (1 : ℚ) / Real.choose 50 5
  let bonusBallProb := (1 : ℚ) / 15
  let totalProb := megaBallProb * winnerBallsProb * bonusBallProb
  totalProb = 1 / 95673600 :=
by
  sorry

end lottery_probability_l501_501402


namespace geometric_series_sum_l501_501174

theorem geometric_series_sum :
  ∀ (a r : ℝ) (h : |r| < 1), a = 5 → r = -2 / 3 → (a / (1 - r)) = 3 := by
  intros a r h ha hr
  rw [ha, hr]
  have hv : 1 - (-2 / 3) = 5 / 3 := by linarith
  rw hv
  linarith

end geometric_series_sum_l501_501174


namespace floor_equation_l501_501278

theorem floor_equation (n : ℤ) (h : ⌊(n^2 : ℤ) / 4⌋ - ⌊n / 2⌋^2 = 5) : n = 11 :=
sorry

end floor_equation_l501_501278


namespace hawkeye_remaining_balance_l501_501269

theorem hawkeye_remaining_balance
  (cost_per_charge : ℝ) (number_of_charges : ℕ) (initial_budget : ℝ) : 
  cost_per_charge = 3.5 → number_of_charges = 4 → initial_budget = 20 → 
  initial_budget - (number_of_charges * cost_per_charge) = 6 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  norm_num
  sorry

end hawkeye_remaining_balance_l501_501269


namespace distances_product_P_to_AB_l501_501640

open Real

noncomputable def pointP := (0.5, 1 : ℝ)

noncomputable def parametric_line_l (t : ℝ) : ℝ × ℝ :=
  (0.5 + (sqrt 3 / 2) * t, 1 + 0.5 * t)

noncomputable def polar_curve_C (θ : ℝ) : ℝ :=
  sqrt 2 * cos (θ - π / 4)

theorem distances_product_P_to_AB (t1 t2 : ℝ) (A B : ℝ × ℝ) :
  (A = parametric_line_l t1) →
  (B = parametric_line_l t2) →
  ∃ C_x C_y, (C_x = pointP.1) ∧ (C_y = pointP.2) ∧ 
  (C_x = (√2 * C_y)) ∧
  ((sqrt (A.1^2 + A.2^2) = polar_curve_C (atan2 A.2 A.1)) ∧
  (sqrt (B.1^2 + B.2^2) = polar_curve_C (atan2 B.2 B.1))) →
  abs (t1 * t2) = |t1 * t2| :=
begin
  sorry
end

end distances_product_P_to_AB_l501_501640


namespace sum_first_12_terms_is_76_l501_501646

-- Define the sequence using provided recurrence relation
def sequence (a : ℕ → ℤ) : Prop :=
  ∀ (n : ℕ), a (n + 1) + (-1 : ℤ)^n * a n = 2 * n - 1

-- Problem statement: Proving the sum of the first 12 terms is 76 given the recurrence relation
theorem sum_first_12_terms_is_76 (a : ℕ → ℤ) (h : sequence a) : (∑ k in finset.range 12, a k) = 76 :=
by
  sorry

end sum_first_12_terms_is_76_l501_501646


namespace positive_difference_between_sums_l501_501329

def sum_of_integers (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_5 (n : ℕ) : ℕ :=
  let m := n % 5
  if m < 3 then n - m else n + (5 - m)

def sum_rounded (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i, round_to_nearest_5 (i + 1))

theorem positive_difference_between_sums :
  let S_Jo := sum_of_integers 200
  let S_Alex := sum_rounded 200
  S_Jo - S_Alex = 0 :=
by
  sorry

end positive_difference_between_sums_l501_501329


namespace sum_of_coefficients_l501_501915

theorem sum_of_coefficients (f : ℕ → ℕ) :
  (5 * 1 + 2)^7 = 823543 :=
by
  sorry

end sum_of_coefficients_l501_501915


namespace genevieve_thermoses_l501_501117

theorem genevieve_thermoses (g n p r : ℕ) (h1 : g = 4.5) (h2 : n = 18) (h3 : p = 6) (h4 : r = 8) : 
  (p / (g * r / n)) = 3 :=
by
  -- Convert gallons to pints
  let total_pints := g * r
  have h_total_pints : total_pints = 36 := by sorry
  -- Find pints per thermos
  let pints_per_thermos := total_pints / n
  have h_pints_per_thermos : pints_per_thermos = 2 := by sorry
  -- Calculate number of thermoses Genevieve drank
  let thermoses_drank := p / pints_per_thermos
  have h_thermoses_drank : thermoses_drank = 3 := by sorry
  exact h_thermoses_drank

end genevieve_thermoses_l501_501117


namespace RogersWifeIsAnne_l501_501219

-- Define individuals involved
inductive Person
| Harry | Peter | Louis | Roger | Elizabeth | Jeanne | Mary | Anne

open Person

def isHusband (husband wife : Person) : Prop :=
  (wife = Elizabeth ∧ husband = Louis) ∨ 
  (wife = Jeanne ∧ husband = Harry) ∨ 
  (wife = Mary ∧ husband = _) ∨ -- We don't need to specify the complete relations, only required for proof
  (wife = Anne ∧ husband = Roger)

def isDancing(couplesDancing: List (Person × Person))(p: Person): Prop :=
  p ∈ (couplesDancing.map Prod.fst ++ couplesDancing.map Prod.snd)

def isPlayingTrumpet(p : Person) : Prop := p = Peter
def isPlayingPiano(p : Person) : Prop := p = Mary

def isNotDancing : Person → Prop := λ p, p = Roger ∨ p = Anne

theorem RogersWifeIsAnne :
  assume (h1 : ∃ (elizabethsHusband : Person), 
              (elizabethsHusband ≠ Harry ∧ 
               isDancing [(Harry, jeanne), (elizabethsHusband, _)] elizabethsHusband))
        (h2 : ∀ p, isPlayingTrumpet p ∨ isPlayingPiano p → ¬ isDancing [(Harry, Jeanne), (_ , _)] p)
        (h3: ∀ p, isNotDancing p),
  isHusband Roger Anne :=
sorry

end RogersWifeIsAnne_l501_501219


namespace trig_identity_l501_501950

theorem trig_identity {α : ℝ} (h : Real.tan α = 2) : 
  (Real.sin (π + α) - Real.cos (π - α)) / 
  (Real.sin (π / 2 + α) - Real.cos (3 * π / 2 - α)) 
  = -1 / 3 := 
by 
  sorry

end trig_identity_l501_501950


namespace money_left_over_l501_501445

def initial_amount : ℕ := 120
def sandwich_fraction : ℚ := 1 / 5
def museum_ticket_fraction : ℚ := 1 / 6
def book_fraction : ℚ := 1 / 2

theorem money_left_over :
  let sandwich_cost := initial_amount * sandwich_fraction
  let museum_ticket_cost := initial_amount * museum_ticket_fraction
  let book_cost := initial_amount * book_fraction
  let total_spent := sandwich_cost + museum_ticket_cost + book_cost
  initial_amount - total_spent = 16 :=
by
  sorry

end money_left_over_l501_501445


namespace parabolas_intersect_sum_eq_zero_l501_501513

theorem parabolas_intersect_sum_eq_zero :
  let intersect_points := 
    { p : ℝ × ℝ // (p.2 = (p.1 - 2)^2) ∧ (p.1 + 3 = (p.2 + 2)^2) } in
  let xs := intersect_points.map Prod.fst in
  let ys := intersect_points.map Prod.snd in
  (xs.sum + ys.sum) = 0 :=
sorry

end parabolas_intersect_sum_eq_zero_l501_501513


namespace max_value_cos_sin_expr_l501_501540

theorem max_value_cos_sin_expr :
  ∀ θ1 θ2 θ3 θ4 θ5 θ6 θ7 : ℝ,
    (cos θ1 * sin θ2 + cos θ2 * sin θ3 + cos θ3 * sin θ4 +
     cos θ4 * sin θ5 + cos θ5 * sin θ6 + cos θ6 * sin θ7 +
     cos θ7 * sin θ1) ≤ 7 / 2 :=
begin
  sorry
end

end max_value_cos_sin_expr_l501_501540


namespace no_integer_solutions_other_than_zero_l501_501375

theorem no_integer_solutions_other_than_zero (x y z : ℤ) :
  x^2 + y^2 + z^2 = x^2 * y^2 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro h
  sorry

end no_integer_solutions_other_than_zero_l501_501375


namespace find_zero_difference_l501_501741

-- Definitions based on the given conditions.
def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def vertex_x := 5
def vertex_y := -9

def known_point_x := 6
def known_point_y := -8

-- Proof statement for the problem.
theorem find_zero_difference :
  ∃ (a b c : ℝ), 
    vertex_y = a * vertex_x^2 + b * vertex_x + c ∧
    known_point_y = a * known_point_x^2 + b * known_point_x + c ∧
    (let x₁ := 2 in let x₂ := 8 in x₁ < x₂ ∧ a ≠ 0 ∧ (x₁ ≠ x₂) ∧ x₁ = 2 ∧ x₂ = 8 ∧ x₂ - x₁ = 6) :=
sorry

end find_zero_difference_l501_501741


namespace max_value_f_inequality_g_l501_501111

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - (1 - x) * Real.log (1 - x)
noncomputable def g (x : ℝ) : ℝ := x^(1 - x) + (1 - x)^x

-- Proof for the first problem:
-- ∀ x ∈ (0, 1/2], f(x) ≤ 0 and ∃ x ∈ (0, 1/2], f(x) = 0.
theorem max_value_f : (∀ (x : ℝ), (0 < x) ∧ (x ≤ 1 / 2) → f(x) ≤ 0) ∧ (∃ (x : ℝ), (0 < x) ∧ (x ≤ 1 / 2) ∧ f(x) = 0) :=
by sorry

-- Proof for the second problem:
-- ∀ x ∈ (0,1), x^(1-x) + (1-x)^x ≤ sqrt(2).
theorem inequality_g : ∀ (x : ℝ), (0 < x) ∧ (x < 1) → g(x) ≤ Real.sqrt 2 :=
by sorry

end max_value_f_inequality_g_l501_501111


namespace ratio_of_novelists_to_poets_l501_501155

theorem ratio_of_novelists_to_poets (total_people novelists : ℕ) (h1 : total_people = 24) (h2 : novelists = 15) : (novelists = 5 ∧ (total_people - novelists) = 3) :=
by
  -- Define constants and acknowledge conditions
  let poets := total_people - novelists
  have h3 : poets = 24 - 15 := by rw [h1, h2]
  have h4 : poets = 9 := by norm_num
  have ratio : (15 / 3) = 5 ∧ (9 / 3) = 3 := by norm_num

  -- Assert the final conclusion
  show (novelists = 5) ∧ ((total_people - novelists) = 3)
  from ⟨by rw [h2]; exact ratio.1, by rw [h3, ratio.2]⟩


end ratio_of_novelists_to_poets_l501_501155


namespace solution_of_inequality_l501_501052

theorem solution_of_inequality (a : ℝ) :
  (a = 0 → ∀ x : ℝ, ax^2 - (a + 1) * x + 1 < 0 ↔ x > 1) ∧
  (a < 0 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ x > 1 ∨ x < 1/a)) ∧
  (0 < a ∧ a < 1 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ 1 < x ∧ x < 1/a)) ∧
  (a > 1 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ 1/a < x ∧ x < 1)) ∧
  (a = 1 → ∀ x : ℝ, ¬(ax^2 - (a + 1) * x + 1 < 0)) :=
by
  sorry

end solution_of_inequality_l501_501052


namespace cake_icing_volume_sum_l501_501856

-- Definitions of geometrical properties
def edge_length : ℝ := 3

def midpoint_of_edge (a b : ℝ) : ℝ := (a + b) / 2

def cube_volume (a : ℝ) : ℝ := a ^ 3

def triangle_area (base height : ℝ) : ℝ := 0.5 * base * height

def cake_volume (area height : ℝ) : ℝ := area * height

def icing_area_triangle (area : ℝ) : ℝ := area

def icing_area_rect (base height : ℝ) : ℝ := base * height

-- The complete icing area and volume calculation for the specific problem
theorem cake_icing_volume_sum : 
  let c := cake_volume (triangle_area edge_length (midpoint_of_edge edge_length 0)) edge_length in
  let s := icing_area_triangle (triangle_area edge_length (midpoint_of_edge edge_length 0))
            + icing_area_rect edge_length edge_length
            + icing_area_rect edge_length (midpoint_of_edge edge_length 0)
            + icing_area_rect edge_length edge_length in
  c + s = 24 := 
begin
  -- Skipping the proof
  sorry
end

end cake_icing_volume_sum_l501_501856


namespace sequence_formula_l501_501068

theorem sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (h_sum: ∀ n : ℕ, n ≥ 2 → S n = n^2 * a n)
  (h_a1 : a 1 = 1) : ∀ n : ℕ, n ≥ 2 → a n = 2 / (n * (n + 1)) :=
by {
  sorry
}

end sequence_formula_l501_501068


namespace func_eq_solution_l501_501346

noncomputable section
open Classical

variables (f : ℝ → ℝ) (α : ℝ)

/-- If α is a nonzero real number and f fulfills the functional equation, then:
    - For α = -1, the only solution is f(x) = x.
    - For any other α, there is no solution.
 -/
theorem func_eq_solution :
  (α ≠ 0) →
  ((α = -1 → ∀ x : ℝ, f x = x) ∧ (α ≠ -1 → ¬ ∃ f : ℝ → ℝ, 
    ∀ x y : ℝ, f(f(x+y)) = f(x+y) + f(x) * f(y) + α * x * y)) :=
by {
  sorry
}

end func_eq_solution_l501_501346


namespace rectangle_proof_l501_501636

noncomputable def ProofEquivalence (A B C D K M : Point) (h_rect : Rectangle ABCD)
  (hK_on_AC : K ∈ AC) (hK_CK_eq_BC : dist C K = dist B C)
  (hM_on_BC : M ∈ BC) (hKM_eq_CM : dist K M = dist C M) : Prop :=
  dist A K + dist B M = dist C M

theorem rectangle_proof (A B C D K M : Point) (h_rect : Rectangle ABCD)
  (hK_on_AC : K ∈ AC) (hK_CK_eq_BC : dist C K = dist B C)
  (hM_on_BC : M ∈ BC) (hKM_eq_CM : dist K M = dist C M) :
  ProofEquivalence A B C D K M h_rect hK_on_AC hK_CK_eq_BC hM_on_BC hKM_eq_CM :=
sorry

end rectangle_proof_l501_501636


namespace remaining_pages_after_a_week_l501_501072

-- Define the conditions
def total_pages : Nat := 381
def pages_read_initial : Nat := 149
def pages_per_day : Nat := 20
def days : Nat := 7

-- Define the final statement to prove
theorem remaining_pages_after_a_week :
  let pages_left_initial := total_pages - pages_read_initial
  let pages_read_week := pages_per_day * days
  let pages_remaining := pages_left_initial - pages_read_week
  pages_remaining = 92 := by
  sorry

end remaining_pages_after_a_week_l501_501072


namespace last_locker_opened_2041_l501_501150

-- Define the initial condition and the process of opening lockers
def initial_lockers (n : Nat) : List Bool := List.replicate n false

def process_trip (lockers : List Bool) (trip : Nat) : List Bool :=
  let skip := trip - 1
  lockers.mapWithIx (fun ix status =>
    if status = false ∧ (ix + 1 - trip) % (skip + 2) = 0 then !status else status)

def final_lockers (n : Nat) : List Bool :=
  let rec aux (lockers : List Bool) (trip : Nat) : List Bool :=
    let new_lockers := process_trip lockers trip
    if new_lockers.any (λ s => s = false) then aux new_lockers (trip + 1) else new_lockers
  aux (initial_lockers n) 2

def last_opened_locker (n : Nat) : Nat :=
  let lockers := final_lockers n
  lockers.indexes (fun status => status).getLast! + 1

-- The main theorem to prove
theorem last_locker_opened_2041 (n : Nat) (h : n = 2048) : last_opened_locker n = 2041 :=
  by sorry

end last_locker_opened_2041_l501_501150


namespace problem_statement_l501_501422

noncomputable def middle_of_three_consecutive (x : ℕ) : ℕ :=
  let y := x + 1
  let z := x + 2
  y

theorem problem_statement :
  ∃ x : ℕ, 
    (x + (x + 1) = 18) ∧ 
    (x + (x + 2) = 20) ∧ 
    ((x + 1) + (x + 2) = 23) ∧ 
    (middle_of_three_consecutive x = 7) :=
by
  sorry

end problem_statement_l501_501422


namespace pseudocode_output_l501_501972

theorem pseudocode_output:
  let x := 2 in
  let i := 1 in
  let s := 0 in
  let final_s := (λ (x i s: ℕ), 
                 nat.iterate (λ (p: ℕ × ℕ × ℕ), (p.1, p.2 + 1, p.3 * p.1 + 1)) 4 (x, i, s)).2.2 
  in final_s = 15 := 
by
  sorry

end pseudocode_output_l501_501972


namespace apples_before_harvesting_l501_501328

theorem apples_before_harvesting (h r c : ℤ) (current_apples : c = 725) (rotten_apples : r = 263) (new_apples : h = 419) :
  let x := c + r - h
  in x = 569 :=
by
  sorry

end apples_before_harvesting_l501_501328


namespace ab_equals_one_l501_501976

def f (x : ℝ) : ℝ := |Real.log x|

theorem ab_equals_one (a b : ℝ) (h1 : a ≠ b) (h2 : f a = f b) : a * b = 1 :=
by
  sorry

end ab_equals_one_l501_501976


namespace intersection_points_trajectory_of_P_general_trajectory_eq_l501_501113

-- Definitions of lines and circles
def C1 (t : ℝ) : ℝ × ℝ := (t, t - 1)  -- Parametric equation of line C1
def C2 (θ : ℝ) : ℝ × ℝ := (cos θ, sin θ)  -- Parametric equation of circle C2

-- Condition for α and intersection points
def α : ℝ := 0  -- Given some specific α
def intersection_point1 : ℝ × ℝ := (1, 0)
def intersection_point2 : ℝ × ℝ := (0, -1)  -- Placeholders, use specific values

-- Coordinates of point A and trajectory of point P
def point_A (α : ℝ) : ℝ × ℝ := (sin(α)^2, -cos(α) * sin(α))
def trajectory_P (α : ℝ) : ℝ × ℝ := ((cos(α)^2) + 1, 0)  -- Parametric equation, placeholder values

-- The general equation of the trajectory of P
def trajectory_eq (x y : ℝ) : Prop := (x - cos 0)^2 + y^2 = 1  -- Example general equation

-- Proving the given statements
theorem intersection_points :
  ∃ t θ, C1 t = intersection_point1 ∧ C2 θ = intersection_point1 ∨ C1 t = intersection_point2 ∧ C2 θ = intersection_point2 := 
  sorry

theorem trajectory_of_P : 
  ∀ α, trajectory_P α = (cos (2 * α) / 2 + 0.5, 0) :=
  sorry

theorem general_trajectory_eq :
  ∀ x y, trajectory_eq x y ↔ (x - 0)^2 + y^2 = 1 :=
  sorry

end intersection_points_trajectory_of_P_general_trajectory_eq_l501_501113


namespace quadrilateral_inscribed_circle_ratios_l501_501705

theorem quadrilateral_inscribed_circle_ratios
  (r : ℝ)
  (h_r_pos : 0 < r)
  (FEG_angle : ℝ)
  (EFG_angle : ℝ)
  (h_FEG_angle : FEG_angle = 40)
  (h_EFG_angle : EFG_angle = 50)
  (EG_is_diameter : ∀ P, P = 2 * r) :
  let π := Real.pi in
  let K := r^2 * Real.sqrt ((3-2*Real.sqrt 2) * (2*Real.sqrt 2 - 1)) in
  let A_circle := π * r^2 in
  let perimeter_EFGH := 2 * r * (Real.sqrt (3 - 2 * Real.sqrt 2) + Real.sqrt (2 * Real.sqrt 2 - 1)) in
  let diameter_circle := 2 * r in
  (K / A_circle = Real.sqrt ((3-2*Real.sqrt 2) * (2*Real.sqrt 2 - 1)) / π) ∧
  (perimeter_EFGH / diameter_circle = Real.sqrt (3 - 2 * Real.sqrt 2) + Real.sqrt (2 * Real.sqrt 2 - 1)) :=
by
  sorry

end quadrilateral_inscribed_circle_ratios_l501_501705


namespace count_perfect_cubes_in_range_150_to_2000_l501_501998

theorem count_perfect_cubes_in_range_150_to_2000 : 
  ∃ (n : ℤ) (S : finset ℤ), S = {n | 150 ≤ n^3 ∧ n^3 ≤ 2000}.to_finset ∧ S.card = 7 := 
by sorry

end count_perfect_cubes_in_range_150_to_2000_l501_501998


namespace max_value_of_expression_l501_501107

theorem max_value_of_expression (x y : ℝ) (h1 : |x - y| ≤ 2) (h2 : |3 * x + y| ≤ 6) : x^2 + y^2 ≤ 10 :=
sorry

end max_value_of_expression_l501_501107


namespace f_neg_one_eq_zero_f_is_even_function_range_of_x_l501_501939

variables {R : Type*} [linear_ordered_field R]

noncomputable def f : R → R := sorry
-- condition of problem statement
axiom f_mul (x1 x2 : R) (h1 : x1 ≠ 0) (h2 : x2 ≠ 0) : f (x1 * x2) = f x1 + f x2
axiom f_increasing_on_positive : ∀ x y : R, 0 < x → 0 < y → x < y → f x < f y
axiom f_less_than (x : R) : f (2 * x - 1) < f x

-- prove that f(-1) = 0
theorem f_neg_one_eq_zero : f (-1) = 0 :=
sorry

-- prove that f is an even function
theorem f_is_even_function (x : R) : f x = f (-x) :=
sorry

-- prove the range of values for x
theorem range_of_x (x : R) : 1/3 < x → x < 1 → f (2 * x - 1) < f x :=
sorry

end f_neg_one_eq_zero_f_is_even_function_range_of_x_l501_501939


namespace sqrt_inequality_xyz_l501_501352

theorem sqrt_inequality_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (hxyz : x * y * z = 1) :
  sqrt (1 + 8 * x) + sqrt (1 + 8 * y) + sqrt (1 + 8 * z) ≥ 9 :=
by
  sorry

end sqrt_inequality_xyz_l501_501352


namespace area_of_triangle_DEF_l501_501193

-- Definitions of the given conditions
def angle_D : ℝ := 45
def DF : ℝ := 4
def DE : ℝ := DF -- Because it's a 45-45-90 triangle

-- Leam statement proving the area of the triangle
theorem area_of_triangle_DEF : 
  (1 / 2) * DE * DF = 8 := by
  -- Since DE = DF = 4, the area of the triangle can be computed
  sorry

end area_of_triangle_DEF_l501_501193


namespace small_planters_needed_l501_501034

-- This states the conditions for the problem
def Oshea_seeds := 200
def large_planters := 4
def large_planter_capacity := 20
def small_planter_capacity := 4
def remaining_seeds := Oshea_seeds - (large_planters * large_planter_capacity) 

-- The target we aim to prove: the number of small planters required
theorem small_planters_needed :
  remaining_seeds / small_planter_capacity = 30 := by
  sorry

end small_planters_needed_l501_501034


namespace smallest_integer_k_condition_l501_501817

theorem smallest_integer_k_condition :
  ∃ k : ℤ, k > 1 ∧ k % 12 = 1 ∧ k % 5 = 1 ∧ k % 3 = 1 ∧ k = 61 :=
by
  sorry

end smallest_integer_k_condition_l501_501817


namespace proof_problem_1_proof_problem_2_l501_501164

noncomputable def problem1 : ℝ :=
  real.cbrt (-27) + real.sqrt ((-6)^2) + (real.sqrt 5)^2

theorem proof_problem_1 :
  problem1 = 8 := 
by
  -- Sorry is used as proof is not needed.
  sorry

noncomputable def abs_sub (a b : ℝ) : ℝ :=
  if a - b < 0 then b - a else a - b

noncomputable def problem2 : ℝ :=
  (-real.sqrt 3)^2 - real.sqrt (16) - abs_sub 1 (real.sqrt 2)

theorem proof_problem_2 :
  problem2 = -real.sqrt 2 := 
by
  -- Sorry is used as proof is not needed.
  sorry

end proof_problem_1_proof_problem_2_l501_501164


namespace find_integer_m_l501_501539

theorem find_integer_m (m : ℤ) (h1 : 5 ≤ m) (h2 : m ≤ 9) (h3 : m ≡ 5023 [MOD 6]) : m = 7 :=
by
  have h4 : 5023 ≡ 1 [MOD 6], from
    calc 5023 % 6 = 1 : by norm_num,
  have h5 : m ≡ 1 [MOD 6], from h3.trans h4.symm,
  have possible_values := [5, 6, 7, 8, 9].filter (λ k, k ≡ 1 [MOD 6]),
  have result := List.head possible_values,
  exact result


end find_integer_m_l501_501539


namespace cheenu_time_difference_l501_501503

def cheenu_bike_time_per_mile (distance_bike : ℕ) (time_bike : ℕ) : ℕ := time_bike / distance_bike
def cheenu_walk_time_per_mile (distance_walk : ℕ) (time_walk : ℕ) : ℕ := time_walk / distance_walk
def time_difference (time1 : ℕ) (time2 : ℕ) : ℕ := time2 - time1

theorem cheenu_time_difference 
  (distance_bike : ℕ) (time_bike : ℕ) 
  (distance_walk : ℕ) (time_walk : ℕ) 
  (H_bike : distance_bike = 20) (H_time_bike : time_bike = 80) 
  (H_walk : distance_walk = 8) (H_time_walk : time_walk = 160) :
  time_difference (cheenu_bike_time_per_mile distance_bike time_bike) (cheenu_walk_time_per_mile distance_walk time_walk) = 16 := 
by
  sorry

end cheenu_time_difference_l501_501503


namespace monotonic_increasing_range_l501_501597

theorem monotonic_increasing_range (a : ℝ) :
  (∀ x : ℝ, (3*x^2 + 2*x - a) ≥ 0) ↔ (a ≤ -1/3) :=
by
  sorry

end monotonic_increasing_range_l501_501597


namespace intersection_of_sets_l501_501340

theorem intersection_of_sets (M N: set ℝ):
  (M = {x | (x + 2) * (x - 2) ≤ 0}) →
  (N = {x | -1 < x ∧ x < 3}) →
  (M ∩ N = {x | -1 < x ∧ x ≤ 2}) :=
by
  intros hM hN
  simp [hM, hN]
  sorry

end intersection_of_sets_l501_501340


namespace find_acres_of_wheat_l501_501389

-- Definitions of the conditions
def total_land (x y : ℕ) : Prop := x + y = 500
def total_cost (x y : ℕ) : Prop := 42 * x + 30 * y = 18,600

-- The main theorem we want to prove
theorem find_acres_of_wheat (x y : ℕ) (h1 : total_land x y) (h2 : total_cost x y) : y = 200 :=
sorry

end find_acres_of_wheat_l501_501389


namespace ratio_jensen_finley_l501_501799

-- Define the initial conditions
def total_tickets := 400
def fraction_given := 3 / 4
def tickets_given := fraction_given * total_tickets
def finleys_tickets := 220
def jensens_tickets := tickets_given - finleys_tickets

-- Define the goal: the ratio of Jensen's tickets to Finley's tickets
def tickets_ratio := jensens_tickets / finleys_tickets

theorem ratio_jensen_finley:
  tickets_given = 300 ∧ finleys_tickets = 220 ∧ jensens_tickets = 80 ∧ tickets_ratio = 4 / 11 :=
by
  -- We state the conditions without proving them.
  have h1: tickets_given = 300 := sorry,
  have h2: finleys_tickets = 220 := sorry,
  have h3: jensens_tickets = 80 := sorry,
  have h4: tickets_ratio = 4 / 11 := sorry,
  exact ⟨h1, h2, h3, h4⟩

end ratio_jensen_finley_l501_501799


namespace find_angle_B_l501_501567

theorem find_angle_B (A B C : ℝ) (a b c : ℝ) (h1: 0 < A ∧ A < π / 2)
  (h2: 0 < B ∧ B < π / 2) (h3: 0 < C ∧ C < π / 2)
  (h4: a * real.cos C + c * real.lcos A = 2 * b * real.cos B)
  (h5: A + B + C = π) :
  B = π / 3 :=
by
  sorry

end find_angle_B_l501_501567


namespace calculate_expression_l501_501893

theorem calculate_expression : 6 * (8 + 1/3) = 50 := by
  sorry

end calculate_expression_l501_501893


namespace cosine_arithmetic_progression_l501_501191

theorem cosine_arithmetic_progression (a : ℝ) :
  (∃ x y z : ℝ, (cos x ≠ cos y ∧ cos y ≠ cos z ∧ cos z ≠ cos x) ∧
    (2 * cos y = cos x + cos z) ∧
    (2 * cos (y + a) = cos (x + a) + cos (z + a))) ↔
    ∃ k : ℤ, a = k * π :=
by
  sorry

end cosine_arithmetic_progression_l501_501191


namespace number_of_k_values_l501_501208

theorem number_of_k_values :
  let k (a b : ℕ) := 2^a * 3^b in
  (∀ a b : ℕ, 18 ≤ a ∧ b = 36 → 
  let lcm_val := Nat.lcm (Nat.lcm (9^9) (12^12)) (k a b) in 
  lcm_val = 18^18) →
  (Finset.card (Finset.filter (λ a, 18 ≤ a ∧ a ≤ 24) (Finset.range (24 + 1))) = 7) :=
by
  -- proof skipped
  sorry

end number_of_k_values_l501_501208


namespace minimum_value_F_l501_501433

noncomputable def minimum_value_condition (x y : ℝ) : Prop :=
  x^2 + y^2 + 25 = 10 * (x + y)

noncomputable def F (x y : ℝ) : ℝ :=
  6 * y + 8 * x - 9

theorem minimum_value_F :
  (∃ x y : ℝ, minimum_value_condition x y) → ∃ x y : ℝ, minimum_value_condition x y ∧ F x y = 11 :=
sorry

end minimum_value_F_l501_501433


namespace probability_point_less_than_one_l501_501564

theorem probability_point_less_than_one :
  let segment := [0, 3]
  let length_of_segment := 3
  let interval := [0, 1]
  let length_of_interval := 1
  let probability := length_of_interval / length_of_segment
  in probability = 1 / 3 :=
by
  sorry

end probability_point_less_than_one_l501_501564


namespace triangle_side_s_l501_501749

/-- The sides of a triangle have lengths 8, 13, and s where s is a whole number.
    What is the smallest possible value of s?
    We need to show that the minimum possible value of s such that 8 + s > 13,
    s < 21, and 13 + s > 8 is s = 6. -/
theorem triangle_side_s (s : ℕ) : 
  (8 + s > 13) ∧ (8 + 13 > s) ∧ (13 + s > 8) → s = 6 :=
by
  sorry

end triangle_side_s_l501_501749


namespace line_tangent_to_circle_perpendicular_l501_501245

theorem line_tangent_to_circle_perpendicular 
  (l₁ l₂ : String)
  (C : String)
  (h1 : l₂ = "4 * x - 3 * y + 1 = 0")
  (h2 : C = "x^2 + y^2 + 2 * y - 3 = 0") :
  (l₁ = "3 * x + 4 * y + 14 = 0" ∨ l₁ = "3 * x + 4 * y - 6 = 0") :=
by
  sorry

end line_tangent_to_circle_perpendicular_l501_501245


namespace infinitely_many_div_by_15_l501_501609

def sequence (v : ℕ → ℤ) : Prop :=
  v 0 = 0 ∧ v 1 = 1 ∧ ∀ n ≥ 1, v (n + 1) = 8 * v n - v (n - 1)

theorem infinitely_many_div_by_15 (v : ℕ → ℤ) (h : sequence v) : ∀ m, ∃ n ≥ m, 15 ∣ v n := 
sorry

end infinitely_many_div_by_15_l501_501609


namespace calculate_rem_and_add_l501_501498

def rem (x y : ℚ) : ℚ := x - y * (floor (x / y))

theorem calculate_rem_and_add :
  rem (5/7) (-3/4) + 1/14 = 1/28 := 
by 
  sorry

end calculate_rem_and_add_l501_501498


namespace interest_rate_is_20_l501_501092

-- Definitions based on conditions in the problem
def principal : ℝ := 500
def time : ℕ := 2
def difference : ℝ := 20

def simple_interest (P r t : ℝ) : ℝ := P * r * t / 100
def compound_interest (P r t : ℝ) : ℝ := P * (1 + r / 100)^t - P

-- The main proof problem statement
theorem interest_rate_is_20 (r : ℝ) :
  compound_interest principal r time - simple_interest principal r time = difference → 
  r = 20 :=
by
  sorry

end interest_rate_is_20_l501_501092


namespace f_is_odd_and_increasing_l501_501595

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_is_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = - f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end f_is_odd_and_increasing_l501_501595


namespace expression_simplification_l501_501844

theorem expression_simplification :
  (- (1 / 2)) ^ 2023 * 2 ^ 2024 = -2 :=
by
  sorry

end expression_simplification_l501_501844


namespace sophie_loads_per_week_l501_501053

noncomputable def loads_per_week (savings : ℝ) (cost_per_box : ℝ) (sheets_per_box : ℕ) (weeks_per_year : ℕ) : ℕ :=
  let boxes_per_year := savings / cost_per_box
  let total_sheets_per_year := boxes_per_year * (sheets_per_box : ℝ)
  let total_loads_per_year := total_sheets_per_year
  (total_loads_per_year / (weeks_per_year : ℝ)).toNat

theorem sophie_loads_per_week : loads_per_week 11 5.5 104 52 = 4 :=
by
  have h₁ : 11 / 5.5 = 2 := by norm_num
  have h₂ : 2 * 104 = 208 := by norm_num
  have h₃ : 208 / 52 = 4 := by norm_num
  rw [loads_per_week, h₁, h₂, h₃]
  norm_num

end sophie_loads_per_week_l501_501053


namespace problem1_problem2_l501_501574

-- Definitions for points and function f(x)
def P : ℝ × ℝ := (Real.sqrt 3, 1)
def Q (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def O : ℝ × ℝ := (0, 0)

def f (x : ℝ) := (sqrt 3, 1).fst * (sqrt 3 - cos x) + (sqrt 3, 1).snd * (1 - sin x)

-- Problem 1: Prove the smallest positive period of f(x)
theorem problem1 : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x := by
  sorry

-- Definitions for angles and sides
variables (A B C : ℝ)
variables (a b c BC : ℝ)

-- Problem 2: Given f(A) = 4, A is an internal angle, and BC = 3,
-- prove the maximum perimeter of triangle ABC is 3/sqrt(2) + 3
theorem problem2 (h1 : f A = 4) (h2 : 0 < A ∧ A < π) (h3 : BC = 3) : 
  b + a + c ≤ (3 / sqrt 2) + 3 := by
  sorry

end problem1_problem2_l501_501574


namespace triangle_transformation_l501_501311

theorem triangle_transformation
  (A B C P Q R : Point)
  (hABC : Triangle A B C)
  (hBPC : Triangle B P C)
  (hCQA : Triangle C Q A)
  (hARB : Triangle A R B)
  (hPBC : ∠P B C = 45)
  (hCAQ : ∠C A Q = 45)
  (hBCP : ∠B C P = 30)
  (hQCA : ∠Q C A = 30)
  (hABR : ∠A B R = 15)
  (hBAR : ∠B A R = 15)
  : ∠Q R P = 90 ∧ dist Q R = dist R P := 
by
  sorry

end triangle_transformation_l501_501311


namespace unique_polynomial_l501_501373

theorem unique_polynomial (n : ℕ) (h : 0 < n) :
  ∃! f : Polynomial ℝ, f.degree = n ∧ f.eval 0 = 1 ∧ ((x:ℝ) + 1) * (f.eval x)^2 - 1 = -( ((-x):ℝ) + 1) * (f.eval (-x))^2 + 1 :=
begin
  sorry
end

end unique_polynomial_l501_501373


namespace num_distinct_triangles_in_octahedron_l501_501275

theorem num_distinct_triangles_in_octahedron : ∃ n : ℕ, n = 48 ∧ ∀ (V : Finset (Fin 8)), 
  V.card = 3 → (∀ {a b c : Fin 8}, a ∈ V ∧ b ∈ V ∧ c ∈ V → 
  ¬((a = 0 ∧ b = 1 ∧ c = 2) ∨ (a = 3 ∧ b = 4 ∧ c = 5) ∨ (a = 6 ∧ b = 7 ∧ c = 8)
  ∨ (a = 7 ∧ b = 0 ∧ c = 1) ∨ (a = 2 ∧ b = 3 ∧ c = 4) ∨ (a = 5 ∧ b = 6 ∧ c = 7))) :=
by sorry

end num_distinct_triangles_in_octahedron_l501_501275


namespace prob_same_color_l501_501849

variable (totalBalls blackBalls whiteBalls : ℕ)
variable (firstBlack secondBlack firstWhite secondWhite sameColorProb : ℚ)

-- Conditions
def totalBalls := 15
def blackBalls := 7
def whiteBalls := 8

-- Probabilities
def firstBlack := (blackBalls : ℚ) / totalBalls
def secondBlack := (blackBalls - 1 : ℚ) / (totalBalls - 1)
def firstWhite := (whiteBalls : ℚ) / totalBalls
def secondWhite := (whiteBalls - 1 : ℚ) / (totalBalls - 1)
def sameColorProb := (firstBlack * secondBlack) + (firstWhite * secondWhite)

-- The probability of drawing two balls of the same color
theorem prob_same_color : sameColorProb = 7 / 15 := by sorry

end prob_same_color_l501_501849


namespace solve_inequality_l501_501051

open Real

noncomputable theory

def inequality (x : ℝ) : Prop :=
  (2 * x + 2) / (3 * x + 1) < (x - 3) / (x + 4)

def solution_set : Set ℝ :=
  {x : ℝ | (-sqrt 11 < x ∧ x < -1/3) ∨ (sqrt 11 < x)}

theorem solve_inequality :
  {x : ℝ | inequality x} = solution_set :=
sorry

end solve_inequality_l501_501051


namespace angle_MOC_half_BLD_l501_501300

variable (O M C L B D : Point)
variable (circle : Circle O)
variable (AB CD PQ : Chord circle)
variable (M O C : Point)

def equal_chords : Prop := 
  (AB = CD) ∧ (CD = PQ) 

def angle_half_relation (angle1 angle2 : Angle) : Prop := 
  angle1 = 1/2 * angle2

theorem angle_MOC_half_BLD
  (h_equal_chords : equal_chords AB CD PQ) :
  angle_half_relation (angle M O C) (angle B L D) :=
sorry

end angle_MOC_half_BLD_l501_501300


namespace tangent_line_through_M_l501_501526

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 20

def point_M (x y : ℝ) : Prop := x = 2 ∧ y = -4

def is_tangent_line (x y : ℝ) (L : ℝ → ℝ → Prop) : Prop :=
  L = λ x y, x - 2*y - 10 = 0

theorem tangent_line_through_M :
  (∀ (x y : ℝ), circle_eq x y -> x = 2 ∧ y = -4) →
  (∃ L : ℝ → ℝ → Prop, is_tangent_line 2 (-4) L) :=
by sorry

end tangent_line_through_M_l501_501526


namespace group_embark_total_ways_l501_501635

-- Define the number of adults and children
def n_adults : nat := 3
def n_children : nat := 2

-- Define the capacities of the boats
def capacity_P : nat := 3
def capacity_Q : nat := 2
def capacity_R : nat := 1

-- Define the constraint that any boat carrying a child must also carry an adult
def valid_distribution (A : nat) (C : nat) (capacity : nat) : Prop :=
  C = 0 ∨ A ≥ C ∧ A + C ≤ capacity

-- Statement of the problem
theorem group_embark_total_ways :
  ∃ ways : nat,
    ways = 27 ∧
    (∑(A_P C_P A_Q C_Q A_R) in 
      { (A_P, C_P, A_Q, C_Q, A_R) :
        nat × nat × nat × nat × nat |
        -- Sum of adults and children in each boat should be valid
        valid_distribution A_P C_P capacity_P ∧
        valid_distribution A_Q C_Q capacity_Q ∧
        valid_distribution A_R 0 capacity_R ∧

        -- Total number of adults and children should match given numbers
        A_P + A_Q + A_R = n_adults ∧
        C_P + C_Q = n_children
      }, 1) = ways :=
  sorry

end group_embark_total_ways_l501_501635


namespace trig_property_l501_501249

theorem trig_property (α : ℝ) (hpt : (cos α, sin α) = (4 / 5, 3 / 5)) :
  sin α = 3 / 5 ∧ cos α = 4 / 5 ∧ tan α = 3 / 4 ∧ (sin (π + α) + 2 * sin (π / 2 - α)) / (2 * cos (π - α)) = -5 / 8 :=
by
  sorry

end trig_property_l501_501249


namespace Crimson_Valley_skirts_l501_501516

theorem Crimson_Valley_skirts
  (Azure_Valley_skirts : ℕ)
  (Seafoam_Valley_skirts : ℕ)
  (Purple_Valley_skirts : ℕ)
  (Crimson_Valley_skirts : ℕ)
  (h1 : Azure_Valley_skirts = 90)
  (h2 : Seafoam_Valley_skirts = (2/3 : ℚ) * Azure_Valley_skirts)
  (h3 : Purple_Valley_skirts = (1/4 : ℚ) * Seafoam_Valley_skirts)
  (h4 : Crimson_Valley_skirts = (1/3 : ℚ) * Purple_Valley_skirts)
  : Crimson_Valley_skirts = 5 := 
sorry

end Crimson_Valley_skirts_l501_501516


namespace length_of_tank_l501_501877

variable {Length Width Depth : ℝ}
variable {CostPerSqMeter TotalCost : ℝ}
variable (L : ℝ)

-- Assumptions based on given problem constraints
axiom tank_dimensions : Width = 12 ∧ Depth = 6
axiom cost_constraints : CostPerSqMeter = 0.75 ∧ TotalCost = 558

-- Prove the length of the tank
theorem length_of_tank : (24 * L + 144) * CostPerSqMeter = TotalCost → L = 25 := by
  intro h
  have area_eq : (24 * L + 144) * 0.75 = 558 := by
    rw [←cost_constraints.1, ←cost_constraints.2]
    exact h
  sorry

end length_of_tank_l501_501877


namespace num_k_values_lcm_l501_501213

-- Define prime factorizations of given numbers
def nine_pow_nine := 3^18
def twelve_pow_twelve := 2^24 * 3^12
def eighteen_pow_eighteen := 2^18 * 3^36

-- Number of values of k making eighteen_pow_eighteen the LCM of nine_pow_nine, twelve_pow_twelve, and k
def number_of_k_values : ℕ := 
  19 -- Based on calculations from the proof

theorem num_k_values_lcm :
  ∀ (k : ℕ), eighteen_pow_eighteen = Nat.lcm (Nat.lcm nine_pow_nine twelve_pow_twelve) k → ∃ n, n = number_of_k_values :=
  sorry -- Add the proof later

end num_k_values_lcm_l501_501213


namespace sum_of_first_18_abs_terms_l501_501318

theorem sum_of_first_18_abs_terms 
  (a : ℕ → ℝ) 
  (a1_pos : 0 < a 1) 
  (a10_a11_neg : a 10 * a 11 < 0) 
  (S10_eq : ∑ i in finset.range 10, a (i + 1) = 36) 
  (S18_eq : ∑ i in finset.range 18, a (i + 1) = 12) :
  ∑ i in finset.range 18, |a (i + 1)| = 60 :=
sorry

end sum_of_first_18_abs_terms_l501_501318


namespace minimum_knights_l501_501769

/-!
Problem: There are 1001 people seated around a round table. Each person is either a knight (always tells the truth) or a liar (always lies). Next to each knight, there is exactly one liar, and next to each liar, there is exactly one knight. Prove that the minimum number of knights is 502.
-/

def person := Type
def is_knight (p : person) : Prop := sorry
def is_liar (p : person) : Prop := sorry

axiom round_table (persons : list person) : (∀ (p : person),
  (is_knight p → (∃! q : person, is_liar q ∧ (q = list.nth_le persons ((list.index_of p persons + 1) % 1001) sorry ∨ q = list.nth_le persons ((list.index_of p persons - 1 + 1001) % 1001) sorry))) ∧
  (is_liar p → (∃! k : person, is_knight k ∧ (k = list.nth_le persons ((list.index_of p persons + 1) % 1001) sorry ∨ k = list.nth_le persons ((list.index_of p persons - 1 + 1001) % 1001) sorry))))

theorem minimum_knights (persons : list person) (h : persons.length = 1001) : 
  (∃ (knights : list person), (∀ k ∈ knights, is_knight k) ∧ (∀ l ∉ knights, is_liar l) ∧ knights.length = 502) :=
sorry

end minimum_knights_l501_501769


namespace correct_exponentiation_l501_501436

variable (a : ℝ)

theorem correct_exponentiation : (a^2)^3 = a^6 := by
  sorry

end correct_exponentiation_l501_501436


namespace john_spent_at_candy_store_l501_501611

-- Definition of the conditions
def allowance : ℚ := 1.50
def arcade_spent : ℚ := (3 / 5) * allowance
def remaining_after_arcade : ℚ := allowance - arcade_spent
def toy_store_spent : ℚ := (1 / 3) * remaining_after_arcade

-- Statement and Proof of the Problem
theorem john_spent_at_candy_store : (remaining_after_arcade - toy_store_spent) = 0.40 :=
by
  -- Proof is left as an exercise
  sorry

end john_spent_at_candy_store_l501_501611


namespace sum_of_first_four_terms_geometric_sequence_l501_501291

theorem sum_of_first_four_terms_geometric_sequence (a : ℕ) (r : ℕ) (n : ℕ) :
  a = 4 ∧ r = 2 ∧ n = 4 → (finset.sum (finset.range n) (λ k, a * r ^ k) = 60) :=
by
  sorry

end sum_of_first_four_terms_geometric_sequence_l501_501291


namespace number_of_integers_satisfying_sqrt_inequality_l501_501751

theorem number_of_integers_satisfying_sqrt_inequality :
  {x : ℤ | 3 < real.sqrt x ∧ real.sqrt x < 6}.finite.to_finset.card = 26 :=
sorry

end number_of_integers_satisfying_sqrt_inequality_l501_501751


namespace inequality_proof_l501_501222

def frac (a b : ℝ) := a / b

-- Conditions
def a := (frac 3 5)^(frac 2 5)
def b := (frac 2 5)^(frac 3 5)
def c := (frac 2 5)^(frac 2 5)

-- Statement
theorem inequality_proof : b < c ∧ c < a := by
  -- Proof will be filled in here
  sorry

end inequality_proof_l501_501222


namespace pavel_sum_associative_l501_501700

def pavel_sum (x y : ℝ) : ℝ := (x + y) / (1 - x * y)

theorem pavel_sum_associative (a b c : ℝ) (hab : 1 - a * b ≠ 0) 
  (hbc : 1 - b * c ≠ 0) (ha_bc : 1 - a * ((b + c) / (1 - b * c)) ≠ 0)
  (habc : 1 - ((a + b) / (1 - a * b)) * c ≠ 0) :
  pavel_sum a (pavel_sum b c) = pavel_sum (pavel_sum a b) c :=
  by sorry

end pavel_sum_associative_l501_501700


namespace min_value_a4b3c2_l501_501669

theorem min_value_a4b3c2 {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1/a + 1/b + 1/c = 9) :
  a ^ 4 * b ^ 3 * c ^ 2 ≥ 1 / 5184 := 
sorry

end min_value_a4b3c2_l501_501669


namespace middle_income_sample_count_l501_501303

def total_households : ℕ := 600
def high_income_families : ℕ := 150
def middle_income_families : ℕ := 360
def low_income_families : ℕ := 90
def sample_size : ℕ := 80

theorem middle_income_sample_count : 
  (middle_income_families / total_households) * sample_size = 48 := 
by
  sorry

end middle_income_sample_count_l501_501303


namespace value_of_f_at_neg_2009_point_9_l501_501228

theorem value_of_f_at_neg_2009_point_9 (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f(x) = f(-x))
  (h_add : ∀ x : ℝ, f(x + 1) + f(x) = 3)
  (h_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f(x) = 2 - x) :
  f(-2009.9) = 1.9 :=
by
  sorry

end value_of_f_at_neg_2009_point_9_l501_501228


namespace quadratic_solution_l501_501175

noncomputable def s_value (s t : ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ (x1 = (-s + sqrt (s^2 - 4 * t)) / 2) ∧ (x2 = (-s - sqrt (s^2 - 4 * t)) / 2) ∧ |x1 - x2| = 2

theorem quadratic_solution (s t : ℝ) (hs : s > 0) (ht : t > 0) :
  (x^2 + s * x + t = 0) ∧ (s_value s t) → s = 2 * sqrt (t + 1) := 
by
  sorry

end quadratic_solution_l501_501175


namespace harriet_speed_l501_501098

/-- Harriet drove back from B-town to A-ville at a constant speed of 145 km/hr.
    The entire trip took 5 hours, and it took Harriet 2.9 hours to drive from A-ville to B-town.
    Prove that Harriet's speed while driving from A-ville to B-town was 105 km/hr. -/
theorem harriet_speed (v_return : ℝ) (T_total : ℝ) (t_AB : ℝ) (v_AB : ℝ) :
  v_return = 145 →
  T_total = 5 →
  t_AB = 2.9 →
  v_AB = 105 :=
by
  intros
  sorry

end harriet_speed_l501_501098


namespace rhombus_diagonal_length_l501_501869

theorem rhombus_diagonal_length (side : ℝ) (shorter_diagonal : ℝ) 
  (h1 : side = 51) (h2 : shorter_diagonal = 48) : 
  ∃ longer_diagonal : ℝ, longer_diagonal = 90 :=
by
  sorry

end rhombus_diagonal_length_l501_501869


namespace minimum_N_chips_color_change_l501_501159

theorem minimum_N_chips_color_change (board : Array (Array Color)) :
  (∀ r c, ∃ unique_color, board[r][c] = unique_color) →
  (∀ t, ∃ r c color, board[r][c] = color ∧ 
     (∃! i, board[r][i] = color ∨ ∃! j, board[j][c] = color)) →
  (∀ N, no_more_moves board) →
  minimum_possible_N board = 75 :=
  sorry -- Proof to be filled in

end minimum_N_chips_color_change_l501_501159


namespace toilet_paper_squares_per_roll_l501_501496

theorem toilet_paper_squares_per_roll
  (trips_per_day : ℕ)
  (squares_per_trip : ℕ)
  (num_rolls : ℕ)
  (supply_days : ℕ)
  (total_squares : ℕ)
  (squares_per_roll : ℕ)
  (h1 : trips_per_day = 3)
  (h2 : squares_per_trip = 5)
  (h3 : num_rolls = 1000)
  (h4 : supply_days = 20000)
  (h5 : total_squares = trips_per_day * squares_per_trip * supply_days)
  (h6 : squares_per_roll = total_squares / num_rolls) :
  squares_per_roll = 300 :=
by sorry

end toilet_paper_squares_per_roll_l501_501496


namespace smallest_k_l501_501454

noncomputable def boolean_vars (n : ℕ) := fin n → bool

def cnf7_clause (n : ℕ) (φ : fin n → bool) :=
  ∃ (i : fin 7 → fin n) (ψ : fin 7 → bool → bool),
    ∀ (j : fin 7), ψ j = id ∨ ψ j = (λ x, bnot x) ∧
    φ (i j) = true

def cnf_formula (n k : ℕ) (clauses : fin k → (fin n → bool) → bool) (φ : fin n → bool) :=
  ∀ (f : fin n → bool), (∏ i in finset.univ, clauses i φ) = 0

theorem smallest_k (n : ℕ) (h_n : n = 2017) :
  ∃ (k : ℕ) (clauses : fin k → (fin n → bool) → bool),
    (∀ i, cnf7_clause n (clauses i)) ∧
    cnf_formula n k clauses (λ _, bool.tt) ∧
    k = 2 :=
sorry

end smallest_k_l501_501454


namespace problem_solution_l501_501256

noncomputable def f : ℝ → ℝ := sorry

axiom domain_f : ∀ x, x ∈ Ioo (-1 : ℝ) (1 : ℝ) → f x < 0

axiom functional_eqn_f : ∀ x y, x ∈ Ioo (-1 : ℝ) (1 : ℝ) → y ∈ Ioo (-1 : ℝ) (1 : ℝ) → f x + f y = f ((x + y) / (1 + x * y))

axiom special_value_f : f (1 / 2) = 1

def a : ℕ → ℝ 
| 0 := 1 / 2
| (n + 1) := 2 * a n / (1 + a(n) ^ 2)

theorem problem_solution : 
(∀ x ∈ Ioo (-1 : ℝ) (1 : ℝ), f x ≠ f (-x)) ∧
(∀ x₁ x₂, x₁ ∈ Ioo (-1 : ℝ) (1 : ℝ) → x₂ ∈ Ioo (-1 : ℝ) (1 : ℝ) → x₁ < x₂ → f x₁ < f x₂) ∧
(∀ n : ℕ, f (a n) = 2 ^ n) ∧
(∀ A B, A ∈ Ioo (0 : ℝ) (π / 2) → B ∈ Ioo (0 : ℝ) (π / 2) → A + B > π / 2 → f (sin A) < f (cos B)) :=
by sorry

end problem_solution_l501_501256


namespace even_positive_factors_of_m_l501_501951

theorem even_positive_factors_of_m : 
  let m := 2^4 * 3^3 * 7 in 
  (∃ n : ℕ, (0 < n) ∧ (n ∣ m) ∧ (nat.even n)) → 
  (choose 4 4 * choose 4 4 * choose 2 2) = 32 := 
by
  sorry

end even_positive_factors_of_m_l501_501951


namespace max_min_values_in_interval_l501_501404

noncomputable def f (x : ℝ) : ℝ := 6 - 12*x + x^3

theorem max_min_values_in_interval :
  (∀ x ∈ set.Icc (-3:ℝ) (1:ℝ), f x ≤ 22) ∧
  (∃ x ∈ set.Icc (-3:ℝ) (1:ℝ), f x = 22) ∧
  (∀ x ∈ set.Icc (-3:ℝ) (1:ℝ), f (-5) ≤ f x ) ∧
  (∃ x ∈ set.Icc (-3:ℝ) (1:ℝ), f x = -5) :=
sorry

end max_min_values_in_interval_l501_501404


namespace An_nonempty_finite_l501_501519

def An (n : ℕ) : Set (ℕ × ℕ) :=
  { p : ℕ × ℕ | ∃ (k : ℕ), ∃ (a : ℕ), ∃ (b : ℕ), a = Nat.sqrt (p.1^2 + p.2 + n) ∧ b = Nat.sqrt (p.2^2 + p.1 + n) ∧ k = a + b }

theorem An_nonempty_finite (n : ℕ) (h : n ≥ 1) : Set.Nonempty (An n) ∧ Set.Finite (An n) :=
by
  sorry -- The proof goes here

end An_nonempty_finite_l501_501519


namespace album_pages_l501_501485

variable (x y : ℕ)

theorem album_pages :
  (20 * x < y) ∧
  (23 * x > y) ∧
  (21 * x + y = 500) →
  x = 12 := by
  sorry

end album_pages_l501_501485


namespace prove_ellipse_properties_l501_501571

-- Define the conditions and key parameters
def ellipse_equation (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Condition a > b > 0
def valid_ellipse_params (a b : ℝ) : Prop :=
  a > b ∧ b > 0

-- Distance between the right vertex and the right focus
def right_vertex_focus_distance (a c : ℝ) : Prop :=
  a - c = sqrt 3 - 1

-- Length of the minor axis
def minor_axis_length (b : ℝ) : Prop :=
  2 * b = 2 * sqrt 2

-- Correct answers
def correct_ellipse_equation : Prop :=
  ellipse_equation x y (sqrt 3) (sqrt 2)

def line_through_focus (x y k : ℝ) : Prop :=
  y = k * (x + 1)

def triangle_area (area : ℝ) : Prop :=
  area = 3 * sqrt 2 / 4

-- Proof problem statement
theorem prove_ellipse_properties : (forall a b : ℝ, valid_ellipse_params a b → (right_vertex_focus_distance a 1 → minor_axis_length b → correct_ellipse_equation)) ∧ (forall k : ℝ, line_through_focus x y k → triangle_area (1/2 * 4 * sqrt 3 * (k^2 + 1) / (2 + 3 * k^2) * |k| / sqrt(1 + k^2)) → (y = sqrt 2 * (x + 1) ∨ y = -sqrt 2 * (x + 1)))
  :=
by
  sorry

end prove_ellipse_properties_l501_501571


namespace dot_product_equality_l501_501266

variable {V : Type} [InnerProductSpace ℝ V] (a b : V)

axiom a_magnitude : ∥a∥ = 2
axiom b_magnitude : ∥b∥ = 3

theorem dot_product_equality :
  inner ((2 : ℝ) • a - b) ((2 : ℝ) • a + b) = 7 :=
by 
  sorry

end dot_product_equality_l501_501266


namespace ellipse_equation_l501_501959

noncomputable def b : ℝ := 3
noncomputable def c : ℝ := 3
noncomputable def a : ℝ := real.sqrt (b^2 + c^2)
noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

def is_tangent_line (x y : ℝ) (k : ℝ) : Prop := ellipse_eq x (k * x - 3)

theorem ellipse_equation :
  (∀ x y : ℝ, ellipse_eq x y ↔ (x^2 / 18) + (y^2 / 9) = 1) ∧
  ((is_tangent_line 0 (-3) (1/2)) ∨ (is_tangent_line 0 (-3) 1)) :=
by
  split
  case left =>
    sorry  -- Proof for ellipse equation
  case right =>
    sorry  -- Proof for tangent line equations

end ellipse_equation_l501_501959


namespace PQ_tangent_to_C_l501_501127

theorem PQ_tangent_to_C 
(A B C P Q O : Point) 
(h_iso : IsIsosceles A B C)
(h_circ : CenterOnBase C O B C (Base B C))
(h_tang_AB : Tangent C O A B)
(h_tang_AC : Tangent C O A C)
(h_P_on_AB : OnLine P A B)
(h_Q_on_AC : OnLine Q A C)
(h_product : PB * CQ = (BC / 2)^2) :
Tangent PQ C ∧ (Tangent PQ C → PB * CQ = (BC / 2)^2) := sorry

end PQ_tangent_to_C_l501_501127


namespace tan_sum_identity_l501_501560

theorem tan_sum_identity (α β : ℝ)
  (h1 : Real.tan (α - π / 6) = 3 / 7)
  (h2 : Real.tan (π / 6 + β) = 2 / 5) :
  Real.tan (α + β) = 1 :=
sorry

end tan_sum_identity_l501_501560


namespace intersection_equal_l501_501239

-- Define the sets M and N based on given conditions
def M : Set ℝ := {x : ℝ | x^2 - 3 * x - 28 ≤ 0}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 > 0}

-- Define the intersection of M and N
def intersection : Set ℝ := {x : ℝ | (-4 ≤ x ∧ x ≤ -2) ∨ (3 < x ∧ x ≤ 7)}

-- The statement to be proved
theorem intersection_equal : M ∩ N = intersection :=
by 
  sorry -- Skipping the proof

end intersection_equal_l501_501239


namespace value_of_x_plus_1_times_y_plus_1_l501_501227

theorem value_of_x_plus_1_times_y_plus_1 (x y : ℝ)
  (h₁ : (2 * x, 1, y - 1) is_arithmetic_seq)
  (h₂ : (y + 3, |x + 1|, |x - 1|) is_geometric_seq)
  (h₃ : -1 ≤ x ∧ x ≤ 1)
  (h₄ : |x| * (y + 3) = 4) :
  (x + 1) * (y + 1) = 4 ∨ (x + 1) * (y + 1) = 2 * (sqrt 17 - 3) := sorry

end value_of_x_plus_1_times_y_plus_1_l501_501227


namespace average_temperature_l501_501757

def temperature_NY := 80
def temperature_MIA := temperature_NY + 10
def temperature_SD := temperature_MIA + 25

theorem average_temperature :
  (temperature_NY + temperature_MIA + temperature_SD) / 3 = 95 := 
sorry

end average_temperature_l501_501757


namespace rhombus_diagonal_length_l501_501868

theorem rhombus_diagonal_length (side : ℝ) (shorter_diagonal : ℝ) 
  (h1 : side = 51) (h2 : shorter_diagonal = 48) : 
  ∃ longer_diagonal : ℝ, longer_diagonal = 90 :=
by
  sorry

end rhombus_diagonal_length_l501_501868


namespace _l501_501354

lemma no_integral_roots (n : ℕ) (a : fin (n+1) → ℤ) : 
  (odd (a 0)) → (odd (∑ i : fin (n+1), a i)) → ¬ (∃ r : ℤ, p a r = 0) :=
sorry

def p (a : fin (n+1) → ℤ) (x : ℤ) : ℤ := ∑ i : fin (n+1), a i * x ^ i

noncomputable theorem main_theorem {n : ℕ} {a : fin (n+1) → ℤ} 
  (h0 : odd (a 0)) (h1 : odd (∑ i : fin (n+1), a i)) : 
  ¬ (∃ r : ℤ, p a r = 0) :=
sorry

end _l501_501354


namespace primes_with_7_as_ones_digit_l501_501614

theorem primes_with_7_as_ones_digit : 
  (card (filter (λ n, prime n ∧ n < 50 ∧ n % 10 = 7) (List.range 50))) = 4 := 
by
  sorry

end primes_with_7_as_ones_digit_l501_501614


namespace smallest_number_of_edges_l501_501682

theorem smallest_number_of_edges (n : ℕ) (G : Type) [graph G] (complete_G : complete_graph G) :
  (∃ G' : Type, graph G' ∧ (∃ operation_series : list (four_cycle_removal G'), 
  (no_edge_removal_disconnected (apply_operations G complete_G operation_series) ∧ 
  E (apply_operations G complete_G operation_series) ≥ n))) :=
sorry

end smallest_number_of_edges_l501_501682


namespace perimeter_ratio_l501_501435

-- Define the conditions given in the problem
variables (d : ℝ) (P1 P2 s1 s2 : ℝ)

-- Relate the side lengths to the diagonals
def smaller_square_side_length := d / Real.sqrt 2
def larger_square_side_length := 2 * Real.sqrt 2 * d

-- Define the perimeters based on the side lengths
def smaller_square_perimeter := 4 * smaller_square_side_length
def larger_square_perimeter := 4 * larger_square_side_length

-- The statement to prove
theorem perimeter_ratio
    (h1 : smaller_square_perimeter = 4 * (d / Real.sqrt 2))
    (h2 : larger_square_perimeter = 4 * (2 * Real.sqrt 2 * d)) :
    (larger_square_perimeter / smaller_square_perimeter) = 8 := 
sorry

end perimeter_ratio_l501_501435


namespace smallest_positive_period_max_min_values_monotonic_increasing_interval_l501_501594

noncomputable def f (x : ℝ) : ℝ := 2 * sin (x / 4) * cos (x / 4) - 2 * sqrt 3 * (sin (x / 4))^2 + sqrt 3

theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

theorem max_min_values : ∃ a b, ∀ x, f x ≤ a ∧ f x ≥ b ∧ a = 2 ∧ b = -2 :=
sorry

theorem monotonic_increasing_interval (k : ℤ) :
  ∀ x, (-5 * Real.pi / 3 + 4 * k * Real.pi) ≤ x →
       x ≤ (Real.pi / 3 + 4 * k * Real.pi) →
       ∀ y, x ≤ y ∧ y ≤ x → f x ≤ f y :=
sorry

end smallest_positive_period_max_min_values_monotonic_increasing_interval_l501_501594


namespace floor_sqrt_12_squared_l501_501527

theorem floor_sqrt_12_squared : (Int.floor (Real.sqrt 12))^2 = 9 := by
  sorry

end floor_sqrt_12_squared_l501_501527


namespace area_of_shaded_region_eq_l501_501162

theorem area_of_shaded_region_eq :
  let line1 := fun x => (-3/5:ℚ) * x + 5
  let line2 := fun x => (-1:ℚ) * x + 6
  let intersection := (25/8:ℚ, 23/8:ℚ)
  ∀ (x y : ℚ),
    (0 ≤ x ∧ x ≤ 25/8) →
    (0 ≤ y) →
    (y = line1 x ∨ y = line2 x) →
    (0 ≤ y ∧ y ≤ max (line1 x) (line2 x))
   area_of (shaded_region line1 line2 0 (25/8)) = 625/128 := 
by
  sorry

end area_of_shaded_region_eq_l501_501162


namespace integer_solutions_of_quadratic_l501_501718

theorem integer_solutions_of_quadratic:
    {(x y : ℤ) | x^2 + x = y^4 + y^3 + y^2 + y} = {(-1, -1), (-1, 0), (0, -1), (0, 0), (5, 2), (-6, 2)} :=
by sorry

end integer_solutions_of_quadratic_l501_501718


namespace incorrect_statements_count_l501_501881

theorem incorrect_statements_count : ∀ (A B C D : Point) (θ : Angle) (l1 l2 : Line) (t : Transversal),
  (line_segment_count A B C D = 6) ∧
  (¬ obtuse_angle θ) ∧
  (consecutive_interior_angles_complementary l1 l2 t = false) →
  incorrect_statements_count = 3 :=
by sorry

end incorrect_statements_count_l501_501881


namespace root_of_equation_value_l501_501295

theorem root_of_equation_value (x m : ℝ) (h : 3 / x = m / (x - 3)) (hx : x = 6) : m = 3 / 2 :=
by
  rw [hx] at h
  have h₁ : (3 : ℝ) / 6 = m / (6 - 3) := h
  norm_num at h₁
  exact h₁.symm

end root_of_equation_value_l501_501295


namespace zeros_of_f_l501_501258

def f (x : ℝ) : ℝ := |Real.logb 2 (x-1)| - (1/3)^x

theorem zeros_of_f (x1 x2 : ℝ) (hx1_lt_x2 : x1 < x2)
  (h1 : f x1 = 0) (h2 : f x2 = 0) :
  (1 < x1 ∧ x1 < 2) ∧ (2 < x2 ∧ x2 < ∞) :=
by sorry

end zeros_of_f_l501_501258


namespace lcm_value_count_l501_501201

theorem lcm_value_count (a b : ℕ) (k : ℕ) (h1 : 9^9 = 3^18) (h2 : 12^12 = 2^24 * 3^12) 
  (h3 : 18^18 = 2^18 * 3^36) (h4 : k = 2^a * 3^b) (h5 : 18^18 = Nat.lcm (9^9) (Nat.lcm (12^12) k)) :
  ∃ n : ℕ, n = 25 :=
begin
  sorry
end

end lcm_value_count_l501_501201


namespace largest_among_given_options_l501_501882

def bin_to_dec (n : Nat) : Nat := sorry
def tern_to_dec (n : Nat) : Nat := sorry
def oct_to_dec (n : Nat) : Nat := sorry
def duo_to_dec (n : Nat) : Nat := sorry

theorem largest_among_given_options :
  let A := bin_to_dec 0b101111
  let B := tern_to_dec 1210
  let C := oct_to_dec 0o112
  let D := duo_to_dec 69
  D > A ∧ D > B ∧ D > C :=
by
  -- Definitions and assumptions:
  have A_eq : A = 47 := sorry
  have B_eq : B = 48 := sorry
  have C_eq : C = 74 := sorry
  have D_eq : D = 81 := sorry
  -- Proof:
  sorry

end largest_among_given_options_l501_501882


namespace range_of_a_l501_501293

noncomputable def abs_sub (x : ℝ) (a : ℝ) : ℝ := |x - a| - |x + 2|

theorem range_of_a (a : ℝ) : (∀ x : ℝ, abs_sub x a ≤ 3) ↔ -5 ≤ a ∧ a ≤ 1 :=
by sorry

end range_of_a_l501_501293


namespace smallest_among_three_powers_l501_501834

theorem smallest_among_three_powers :
  let a := 33^12
  let b := 63^10
  let c := 127^8
  a > 2^60 ∧ b < 2^60 ∧ c < 2^56 ∧ c < b → c < a ∧ c < b ∧ c < a :=
by 
  intros a b c h₁ h₂ h₃ h₄
  split
  exact h₃.trans h₁
  split
  exact h₃
  exact h₃.trans h₁

end smallest_among_three_powers_l501_501834


namespace largest_power_of_5_in_fact50_52_54_l501_501529

noncomputable def factorial : ℕ → ℕ 
| 0     := 1 
| (n+1) := (n+1) * factorial n 

def expr_sum : ℕ := factorial 50 + factorial 52 + factorial 54

def largest_power_of_5 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else n / 5 + largest_power_of_5 (n / 5)

theorem largest_power_of_5_in_fact50_52_54 :
  largest_power_of_5 expr_sum = 12 :=
sorry

end largest_power_of_5_in_fact50_52_54_l501_501529


namespace find_quadratic_function_l501_501250

theorem find_quadratic_function (a h k x y : ℝ) (vertex_y : ℝ) (intersect_y : ℝ)
    (hv : h = 1 ∧ k = 2)
    (hi : x = 0 ∧ y = 3) :
    (∀ x, y = a * (x - h) ^ 2 + k) → vertex_y = h ∧ intersect_y = k →
    y = x^2 - 2 * x + 3 :=
by
  sorry

end find_quadratic_function_l501_501250


namespace matrix_sequence_exists_l501_501334

variables {m n : ℕ}
variables (A B : Matrix (Fin m) (Fin n) ℕ)

def row_sum_eq (A B : Matrix (Fin m) (Fin n) ℕ) : Prop :=
  ∀ i, (Finset.univ.sum (λ j, A i j) = Finset.univ.sum (λ j, B i j))

def col_sum_eq (A B : Matrix (Fin m) (Fin n) ℕ) : Prop :=
  ∀ j, (Finset.univ.sum (λ i, A i j) = Finset.univ.sum (λ i, B i j))

theorem matrix_sequence_exists 
  (h1 : row_sum_eq A B) 
  (h2 : col_sum_eq A B) :
  ∃ (n : ℕ) (A_seq : Fin (n + 1) → Matrix (Fin m) (Fin n) ℕ), 
  (A_seq 0 = A) ∧ (A_seq n = B) ∧ 
  (∀ i, ∃ k j u v, 
    A_seq (i + 1) - A_seq i = 
      (Matrix.update (Matrix.update (Matrix.update (Matrix.update 0 u k 1) u j (-1)) v k (-1)) v j 1) ∨
      A_seq (i + 1) - A_seq i = 
      (Matrix.update (Matrix.update (Matrix.update (Matrix.update 0 u k (-1)) u j 1) v k 1) v j (-1))) :=
sorry

end matrix_sequence_exists_l501_501334


namespace cone_lateral_surface_area_l501_501967

noncomputable def lateralSurfaceArea (r l : ℝ) : ℝ := Real.pi * r * l

theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), r = 2 → l = 5 → lateralSurfaceArea r l = 10 * Real.pi :=
by 
  intros r l hr hl
  rw [hr, hl]
  unfold lateralSurfaceArea
  norm_num
  sorry

end cone_lateral_surface_area_l501_501967


namespace number_drawn_from_fourth_group_l501_501630

-- Conditions
def students : Nat := 72
def sample_size : Nat := 6
def groups : List (List Nat) :=
  [[1, 2, ... , 12], [13, 14, ... , 24], [25, 26, ... , 36], [37, 38, ... , 48], [49, 50, ... , 60], [61, 62, ... , 72]]
def second_group_num_drawn : Nat := 16

-- Theorem statement
theorem number_drawn_from_fourth_group : 
  (∃ n ∈ groups[3], n = 40) →
  (second_group_num_drawn = 16 → (∃ m ∈ groups[3], m = 40)) :=
sorry

end number_drawn_from_fourth_group_l501_501630


namespace equal_surface_area_of_intersections_l501_501342

noncomputable theory

-- Define the scene and given conditions
structure Sphere (α : Type*) [NormedGroup α] [NormedSpace ℝ α] :=
(center : α)
(radius : ℝ)

variables {α : Type*} [NormedGroup α] [NormedSpace ℝ α]

def sphere (O : α) (r : ℝ) : Sphere α :=
{ center := O,
  radius := r }

def surface_area_of_intersection (S1 S2 : Sphere α) : ℝ :=
  2 * real.pi * S1.radius * (S1.radius^2 / (2 * S2.radius))

-- Define spheres K, A, and B
variables (O P Q : α) (r R1 R2 : ℝ)
  (h1 : Sphere α) (h2 : Sphere α) (h3 : Sphere α)

-- Sphere K
def K : Sphere α := sphere O r

-- Sphere A
def A : Sphere α := sphere P (dist P O)

-- Sphere B
def B : Sphere α := sphere Q (dist Q O)

-- Prove that the surface area of parts of A and B that lie inside K are equal
theorem equal_surface_area_of_intersections :
  surface_area_of_intersection A K = surface_area_of_intersection B K :=
by
  -- Given conditions and derived formula for intersections
  sorry

end equal_surface_area_of_intersections_l501_501342


namespace height_of_triangle_l501_501325

open Real

/-- In a triangle ABC, if AB = 3, BC = sqrt(13), and AC = 4, 
then the height from B to AC is 3/2 * sqrt(3). -/
theorem height_of_triangle (A B C D : Point)
  (h1 : distance A B = 3)
  (h2 : distance B C = sqrt 13)
  (h3 : distance A C = 4) :
  height B A C = (3/2) * sqrt 3 :=
  sorry

end height_of_triangle_l501_501325


namespace total_mass_of_individuals_l501_501103

def boat_length : Float := 3.0
def boat_breadth : Float := 2.0
def initial_sink_depth : Float := 0.018
def density_of_water : Float := 1000.0
def mass_of_second_person : Float := 75.0

theorem total_mass_of_individuals :
  let V1 := boat_length * boat_breadth * initial_sink_depth
  let m1 := V1 * density_of_water
  let total_mass := m1 + mass_of_second_person
  total_mass = 183 :=
by
  sorry

end total_mass_of_individuals_l501_501103


namespace probability_no_adjacent_same_l501_501549

variable {A B C D E : Nat}

def is_valid_roll_sequence (seq : List Nat) : Prop :=
  seq.length = 5 ∧
  (∀ i, i < 4 → seq.nth i ≠ seq.nth (i + 1)) ∧
  seq.nth 4 ≠ seq.nth 0

theorem probability_no_adjacent_same :
  (∃ seq : List Nat, is_valid_roll_sequence seq ∧ ∀ x ∈ seq, 1 ≤ x ∧ x ≤ 6) →
  (∃ (p : ℚ), p = 125 / 324) :=
by
  intro h
  use (125 / 324)
  sorry

end probability_no_adjacent_same_l501_501549


namespace average_temperature_l501_501762

theorem average_temperature (T_NY T_Miami T_SD : ℝ) (h1 : T_NY = 80) (h2 : T_Miami = T_NY + 10) (h3 : T_SD = T_Miami + 25) :
  (T_NY + T_Miami + T_SD) / 3 = 95 :=
by
  sorry

end average_temperature_l501_501762


namespace car_price_l501_501381

theorem car_price (down_payment : ℕ) (monthly_payment : ℕ) (loan_years : ℕ) 
    (h_down_payment : down_payment = 5000) 
    (h_monthly_payment : monthly_payment = 250)
    (h_loan_years : loan_years = 5) : 
    down_payment + monthly_payment * loan_years * 12 = 20000 := 
by
  rw [h_down_payment, h_monthly_payment, h_loan_years]
  norm_num
  sorry

end car_price_l501_501381


namespace number_of_inverse_pairs_l501_501403

theorem number_of_inverse_pairs {a d : ℝ} (h : matrix (fin 2) (fin 2) ℝ := !![a, 4; -9, d]) : 
  (h * h = 1 : matrix (fin 2) (fin 2) ℝ) → 
  ({p : ℝ × ℝ | ∃ a d, p = (a, d) ∧ matrix.mul !![a, 4; -9, d] !![a, 4; -9, d] = 1}.to_finset.card = 2) :=
sorry

end number_of_inverse_pairs_l501_501403


namespace find_special_number_l501_501432

-- Define the function to reverse the digits of a natural number.
def reverseDigits (n : Nat) : Nat :=
  n.toDigits.reverse.foldl (λ acc d => acc * 10 + d) 0

-- Define the proof problem statement.
theorem find_special_number : ∃ n : Nat, 1 ≤ n ∧ n ≤ 10000 ∧ reverseDigits n = Nat.ceil (n / 2) ∧ n = 7993 :=
by sorry

end find_special_number_l501_501432


namespace sin_angle_BAO_eq_3_5_l501_501395

-- Definitions
variables (A B C D O : Point) (AB BC : ℝ)
variable (isRectangle : Rectangle A B C D)
variable (AB_eq : AB = 12)
variable (BC_eq : BC = 16)
variable (O_is_intersection : O = midpoint (diagonal A C) (diagonal B D))

-- The theorem statement
theorem sin_angle_BAO_eq_3_5 : sin (angle B A O) = 3 / 5 :=
by
  -- Proof implementation goes here
  -- We have to prove sin(angle B A O) = 3 / 5
  sorry

end sin_angle_BAO_eq_3_5_l501_501395


namespace complex_expression_evaluation_l501_501952

theorem complex_expression_evaluation (z : ℂ) (hz : z = 1 + complex.i) : (2 / z + z^2 = 1 + complex.i) := 
    by sorry

end complex_expression_evaluation_l501_501952


namespace spherical_to_rectangular_coords_l501_501515

theorem spherical_to_rectangular_coords :
  ∀ (ρ θ φ : ℝ), ρ = 5 → θ = π / 6 → φ = π / 3 →
  let x := ρ * sin φ * cos θ
  let y := ρ * sin φ * sin θ
  let z := ρ * cos φ
  (x, y, z) = (15 / 4, (5 * sqrt 3) / 4, 5 / 2) :=
by
  intros ρ θ φ h₁ h₂ h₃
  simp [h₁, h₂, h₃]
  sorry

end spherical_to_rectangular_coords_l501_501515


namespace same_fixed_point_l501_501180

open Finset Equiv.Perm

variables (n : ℕ)
-- Denote Sn as the group of permutations of the sequence (1, 2, ..., n)
noncomputable def S_n := equiv.perm (fin n)

-- Assume G is a subgroup of Sn
variable (G : subgroup (S_n n))

-- Assume for every non-identity element in G, there exists a unique k in {1,...,n} such that π(k) = k.
def condition (π : S_n n) : Prop :=
  ∃! k : fin n, π k = k

-- The main theorem to prove
theorem same_fixed_point : 
  (∀ π ∈ G, π ≠ 1 → condition n π) → 
  ∃ k : fin n, ∀ π ∈ G, π ≠ 1 → π k = k :=
  sorry

end same_fixed_point_l501_501180


namespace rectangle_bisectors_form_square_l501_501044

noncomputable def rectangle :=
  {A B C D : Prop // 
    (∃ (b c : Prop), b ∧ c) ∧ 
    (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ A)}

def is_not_square (A B C D : Prop) (h : rectangle) : Prop :=
  ∃ (a b : Prop), 
    a ≠ b ∧ 
    a ≠ A ∧ 
    b ≠ B

def angle_bisectors_form_square (A B C D : Prop) (h : rectangle) : Prop :=
  ∀ (P Q R S : Prop), 
    P ∈ angle_bisector A ∧ 
    Q ∈ angle_bisector B ∧ 
    R ∈ angle_bisector C ∧ 
    S ∈ angle_bisector D → 
    is_square P Q R S

theorem rectangle_bisectors_form_square
  (A B C D : Prop) 
  (h : rectangle) 
  (h_rect : is_not_square A B C D h) :
  angle_bisectors_form_square A B C D h :=
sorry

end rectangle_bisectors_form_square_l501_501044


namespace number_factorial_difference_l501_501739

theorem number_factorial_difference :
  ∃ (a : ℕ) (b : ℕ), 
  (2639 = (Nat.factorial a * Nat.factorial 13) / (Nat.factorial b * Nat.factorial 18)) ∧ 
  (a ≥ 13 ∧ b ≥ 18) ∧ 
  (a + b = 67 + 61) → 
  (|a - b| = 6) :=
sorry

end number_factorial_difference_l501_501739


namespace correct_statement_l501_501097

theorem correct_statement :
  (∀ (x : ℝ), x ≥ 0 → sqrt (x^2) = x) →
  sqrt ((-2 : ℝ)^2) = 2 :=
by
  intros h
  exact h 2 (by norm_num)

end correct_statement_l501_501097


namespace rhombus_area_l501_501704

-- Define the basic parameters
def ABCD : Type := { points : ℕ → ℝ × ℝ // quotient (setoid ℝ × ℝ) }

def is_rhombus (A B C D : ABCD) : Prop :=
  let perimeter := 80
  let diagonal_AC := 30
  ∃ (side_length : ℝ), (4 * side_length = perimeter) ∧
  -- Diagonals bisect each other in a rhombus
  (side_length = 20) ∧
  (diagonal_AC = 30)

theorem rhombus_area {A B C D : ABCD} (h : is_rhombus A B C D) :
  ∃ (area : ℝ), area = 150 * real.sqrt 7 :=
by
  sorry

end rhombus_area_l501_501704


namespace problem1_problem2_l501_501845
-- Importing the necessary library

-- Proof for the first problem
theorem problem1 (x : ℝ) (h : x - 1/x = 2) : x^2 + 1/x^2 = 6 :=
by
  sorry

-- Proof for the second problem
theorem problem2 (a : ℝ) (h : a^2 + 1/a^2 = 4) : a - 1/a = sqrt 2 ∨ a - 1/a = -sqrt 2 :=
by
  sorry

end problem1_problem2_l501_501845


namespace Oshea_needs_30_small_planters_l501_501036

theorem Oshea_needs_30_small_planters 
  (total_seeds : ℕ) 
  (large_planters : ℕ) 
  (capacity_large : ℕ) 
  (capacity_small : ℕ)
  (h1: total_seeds = 200) 
  (h2: large_planters = 4) 
  (h3: capacity_large = 20) 
  (h4: capacity_small = 4) : 
  (total_seeds - large_planters * capacity_large) / capacity_small = 30 :=
by 
  sorry

end Oshea_needs_30_small_planters_l501_501036


namespace rotations_per_block_l501_501991

/--
If Greg's bike wheels have already rotated 600 times and need to rotate 
1000 more times to reach his goal of riding at least 8 blocks,
then the number of rotations per block is 200.
-/
theorem rotations_per_block (r1 r2 n b : ℕ) (h1 : r1 = 600) (h2 : r2 = 1000) (h3 : n = 8) :
  (r1 + r2) / n = 200 := by
  sorry

end rotations_per_block_l501_501991


namespace find_a_set_l501_501627

theorem find_a_set (a : ℝ) :
  (∀ x ∈ set.Icc (a - 2) (a + 2), x^2 - 2 * x + 3 ≥ 6) →
  (x^2 - 2 * x + 3).has_inf_on (set.Icc (a-2) (a+2)) 6 →
  a = -3 ∨ a = 5 :=
by sorry

end find_a_set_l501_501627


namespace symmetric_about_y_axis_l501_501585

-- Condition: f is an odd function defined on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Given that f is odd and F is defined as specified
theorem symmetric_about_y_axis (f : ℝ → ℝ)
  (hf : odd_function f) :
  ∀ x : ℝ, |f x| + f (|x|) = |f (-x)| + f (|x|) := 
by
  sorry

end symmetric_about_y_axis_l501_501585


namespace hiking_hours_l501_501794

theorem hiking_hours
  (violet_water_per_hour : ℕ := 800)
  (dog_water_per_hour : ℕ := 400)
  (total_water : ℕ := 4800) :
  (total_water / (violet_water_per_hour + dog_water_per_hour) = 4) :=
by
  sorry

end hiking_hours_l501_501794


namespace cylinder_volume_ratio_l501_501460

theorem cylinder_volume_ratio (a b : ℕ) (h_dim : (a, b) = (9, 12)) :
  let r₁ := (a : ℝ) / (2 * Real.pi)
  let h₁ := (↑b : ℝ)
  let V₁ := (Real.pi * r₁^2 * h₁)
  let r₂ := (b : ℝ) / (2 * Real.pi)
  let h₂ := (↑a : ℝ)
  let V₂ := (Real.pi * r₂^2 * h₂)
  (if V₂ > V₁ then V₂ / V₁ else V₁ / V₂) = (16 / 3) :=
by {
  sorry
}

end cylinder_volume_ratio_l501_501460


namespace average_students_per_bus_l501_501029

-- Definitions
def total_students : ℕ := 396
def students_in_cars : ℕ := 18
def number_of_buses : ℕ := 7

-- Proof problem statement
theorem average_students_per_bus : (total_students - students_in_cars) / number_of_buses = 54 := by
  sorry

end average_students_per_bus_l501_501029


namespace crease_length_correct_l501_501863

variable (A B C : Point)
variable (a b c : ℝ)

-- Given a triangle ABC with side lengths 3, 4, and 5, where A is folded to B.
def isRightTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Hypothesis: The triangle ABC is a right triangle with the given side lengths.
hypothesis (h_triangle : isRightTriangle 3 4 5)

-- The length of the crease when point A is folded to point B.
noncomputable def length_of_crease (a b c : ℝ) : ℝ :=
  let d := (a^2 + b^2 : ℝ / b^2) in
  real.sqrt(d)

-- Prove that the length of the crease is 15/8 inches
theorem crease_length_correct (h : isRightTriangle 3 4 5) : length_of_crease 3 4 5 = 15 / 8 := by
  sorry

end crease_length_correct_l501_501863


namespace planting_ways_l501_501469

namespace FarmerField

/-- The types of crops available for planting -/
inductive Crop
| corn | wheat | soybeans | potatoes | rice

/-- A 3x3 grid represented as a 9-element set -/
def Field := Fin 3 × Fin 3

structure PlantingConfiguration :=
(sections : Field → Crop)
(no_adjacent_corn_soybeans : ∀ {i j : Field}, i ≠ j → sections i = Crop.corn → sections j ≠ Crop.soybeans)
(no_adjacent_wheat_potatoes : ∀ {i j : Field}, i ≠ j → sections i = Crop.wheat → sections j ≠ Crop.potatoes)

noncomputable def count_valid_configurations : ℕ :=
sorry

theorem planting_ways : count_valid_configurations = 2045 :=
by
  -- We skip the proof here
  sorry

end FarmerField

end planting_ways_l501_501469


namespace water_height_in_cylinder_l501_501491

noncomputable def cone_base_radius : ℝ := 15 -- cm
noncomputable def cone_height : ℝ := 15 -- cm
noncomputable def cylinder_base_radius : ℝ := 18 -- cm
noncomputable def water_loss_rate : ℝ := 0.1

theorem water_height_in_cylinder :
  let V_cone := (1 / 3) * real.pi * (cone_base_radius^2) * cone_height,
      V_remaining := (1 - water_loss_rate) * V_cone,
      h_cylinder := V_remaining / (real.pi * (cylinder_base_radius^2))
  in h_cylinder = 3.125 :=
by sorry

end water_height_in_cylinder_l501_501491


namespace decimal_place_of_fraction_l501_501820

theorem decimal_place_of_fraction :
  let sequence : List ℕ := [7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7]
  (17 : ℕ) -- length of the repeating sequence 
  (n : ℕ) -- position we are interested in
  (m : ℕ) -- sequence position resulting from modulo calculation
  in (n = 250) → (n % 17 = m) → (m = 11) → (sequence.get? m = some 3) :=
by
  intros sequence len n m h_n h_mod h_m
  sorry

end decimal_place_of_fraction_l501_501820


namespace bisection_method_accuracy_l501_501158

-- Define the condition for the bisection method
def bisectionMethod (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f a * f b < 0

-- Lean proof statement for the problem
theorem bisection_method_accuracy (f : ℝ → ℝ) (a b : ℝ) (n : ℕ)
  (h : bisectionMethod f a b) : 
  ∃ x : ℝ, |f x| < 10^(-n) ∧ a ≤ x ∧ x ≤ b :=
sorry

end bisection_method_accuracy_l501_501158


namespace hiking_hours_l501_501798

def violet_water_per_hour : ℕ := 800 -- Violet's water need per hour in ml
def dog_water_per_hour : ℕ := 400    -- Dog's water need per hour in ml
def total_water_capacity : ℚ := 4.8  -- Total water capacity Violet can carry in L

theorem hiking_hours :
  let total_water_per_hour := (violet_water_per_hour + dog_water_per_hour) / 1000 in
  total_water_capacity / total_water_per_hour = 4 :=
by
  let total_water_per_hour := (violet_water_per_hour + dog_water_per_hour) / 1000
  have h1 : violet_water_per_hour = 800 := rfl
  have h2 : dog_water_per_hour = 400 := rfl
  have h3 : total_water_capacity = 4.8 := rfl
  have h4 : total_water_per_hour = 1.2 := by simp [violet_water_per_hour, dog_water_per_hour]
  have h5 : total_water_capacity / total_water_per_hour = 4 := by simp [total_water_capacity, total_water_per_hour]
  exact h5

end hiking_hours_l501_501798


namespace anna_least_days_l501_501887

theorem anna_least_days (borrow : ℝ) (interest_rate : ℝ) (days : ℕ) :
  (borrow = 20) → (interest_rate = 0.10) → borrow + (borrow * interest_rate * days) ≥ 2 * borrow → days ≥ 10 :=
by
  intros h1 h2 h3
  sorry

end anna_least_days_l501_501887


namespace average_temperature_l501_501759

def temperature_NY := 80
def temperature_MIA := temperature_NY + 10
def temperature_SD := temperature_MIA + 25

theorem average_temperature :
  (temperature_NY + temperature_MIA + temperature_SD) / 3 = 95 := 
sorry

end average_temperature_l501_501759


namespace largest_angle_is_90_degrees_l501_501413

variables (a b c : ℝ) (r r_a r_b r_c s : ℝ) (q : ℝ)
-- Conditions
def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2
def geometric_progression (q : ℝ) (r r_a r_b r_c : ℝ) : Prop :=
  r_a = q * r ∧ r_b = q^2 * r ∧ r_c = q^3 * r
def area_relations (t r r_a r_b r_c s a b c : ℝ) : Prop :=
  t = r * s ∧ t = r_a * (s - a) ∧ t = r_b * (s - b) ∧ t = r_c * (s - c)

-- Proof Problem Statement
theorem largest_angle_is_90_degrees 
  (ha : a < b < c)
  (hs : s = semi_perimeter a b c)
  (hg : geometric_progression q r r_a r_b r_c)
  (hr : area_relations (t : ℝ) r r_a r_b r_c s a b c) :
  ∃ (γ : ℝ), γ = 90 :=
  sorry

end largest_angle_is_90_degrees_l501_501413


namespace problem_statement_l501_501518

def nabla (a b : ℕ) : ℕ := 3 + b ^ a

theorem problem_statement : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end problem_statement_l501_501518


namespace polar_coordinates_A_B_length_MN_l501_501262
open Real

def curve_C1 (ρ θ : ℝ) : Prop := ρ^2 * cos (2 * θ) = 8
def curve_C2 (ρ θ : ℝ) : Prop := θ = π / 6

def line_x (t : ℝ) : ℝ := 2 + (sqrt 3) / 2 * t
def line_y (t : ℝ) : ℝ := 1 / 2 * t

def isIntersection (ρ θ : ℝ) : Prop := curve_C1 ρ θ ∧ curve_C2 ρ θ

theorem polar_coordinates_A_B :
  ∃ (ρ₁ ρ₂ : ℝ) (θ₁ θ₂ : ℝ), 
    isIntersection ρ₁ θ₁ ∧ isIntersection ρ₂ θ₂ ∧
    ρ₁ = 4 ∧ θ₁ = π / 6 ∧ ρ₂ = -4 ∧ θ₂ = π / 6 :=
by sorry

theorem length_MN :
  let t1 := -4 * (sqrt 3),
      t2 := 2 in
  let M := (2 + (sqrt 3)/2 * t1, 1/2 * t1),
      N := (2 + (sqrt 3)/2 * t2, 1/2 * t2) in
  dist M N = 4 * sqrt 5 :=
by sorry

end polar_coordinates_A_B_length_MN_l501_501262


namespace calc_expression_l501_501902

-- Define the fractions and whole number in the problem
def frac1 : ℚ := 5/6
def frac2 : ℚ := 1 + 1/6
def whole : ℚ := 2

-- Define the expression to be proved
def expression : ℚ := (frac1) - (-whole) + (frac2)

-- The theorem to be proved
theorem calc_expression : expression = 4 :=
by { sorry }

end calc_expression_l501_501902


namespace students_arrangement_l501_501046

theorem students_arrangement :
  ({s | s ⊆ (finset.range 5) ∧ finset.card s = 4}.card *
   ((finset.card {t | t ⊆ (finset.range 5) ∧ finset.card t = 2}) *
   ((finset.card {u | u ⊆ ((finset.range 5) \ finset.image fintype.some s) ∧ finset.card u = 1}) *
   (finset.card {v | v ⊆ ((finset.range 5) \ (finset.image fintype.some s ∪ finset.image fintype.some t)) ∧ finset.card v = 1}))) = 60 :=
sorry

end students_arrangement_l501_501046


namespace sampling_methods_correct_l501_501464

def first_method_sampling : String :=
  "Simple random sampling"

def second_method_sampling : String :=
  "Systematic sampling"

theorem sampling_methods_correct :
  first_method_sampling = "Simple random sampling" ∧ second_method_sampling = "Systematic sampling" :=
by
  sorry

end sampling_methods_correct_l501_501464


namespace systematic_sampling_l501_501629

theorem systematic_sampling (num_students groups second_draw : ℕ) (h1 : num_students = 60) (h2 : groups = 5) (h3 : second_draw = 16) :
  let interval := num_students / groups in
  let fourth_group_draw := second_draw + 2 * interval in
  fourth_group_draw = 40 :=
by 
  sorry

end systematic_sampling_l501_501629


namespace true_propositions_count_l501_501884

def proposition_1 (x : ℝ) : Prop := x^2 - x + (1/4) ≥ 0
def proposition_2 : Prop := ∃ (x : ℝ), x > 0 ∧ ln x + (1 / ln x) ≤ 2
def proposition_3 (a b c : ℝ) : Prop := (a > b ↔ a * c^2 > b * c^2)
def proposition_4 (x : ℝ) : Prop := 2^x - 2^(-x) = -(2^x - 2^(-x))

def numTrueProps (p1 p2 p3 p4 : Prop) : ℕ :=
  [p1, p2, p3, p4].count (λ p => p)

theorem true_propositions_count :
  let p1 := ∀ (x : ℝ), proposition_1 x
  let p2 := proposition_2
  let p3 := ¬ ∀ (a b c : ℝ), proposition_3 a b c
  let p4 := ∀ (x : ℝ), proposition_4 x
  numTrueProps p1 p2 p3 p4 = 3 := by
  sorry

end true_propositions_count_l501_501884


namespace singer_arrangements_l501_501826

-- Let's assume the 5 singers are represented by the indices 1 through 5

theorem singer_arrangements :
  ∀ (singers : List ℕ) (no_first : ℕ) (must_last : ℕ), 
  singers = [1, 2, 3, 4, 5] →
  no_first ∈ singers →
  must_last ∈ singers →
  no_first ≠ must_last →
  ∃ (arrangements : ℕ),
    arrangements = 18 :=
by
  sorry

end singer_arrangements_l501_501826


namespace farm_needs_horse_food_per_day_l501_501161

-- Definition of conditions
def ratio_sheep_to_horses := 4 / 7
def food_per_horse := 230
def number_of_sheep := 32

-- Number of horses based on ratio
def number_of_horses := (number_of_sheep * 7) / 4

-- Proof Statement
theorem farm_needs_horse_food_per_day :
  (number_of_horses * food_per_horse) = 12880 :=
by
  -- skipping the proof steps
  sorry

end farm_needs_horse_food_per_day_l501_501161


namespace cost_of_orange_juice_l501_501431

theorem cost_of_orange_juice (O : ℝ) (H1 : ∀ (apple_juice_cost : ℝ), apple_juice_cost = 0.60 ):
  let total_bottles := 70
  let total_cost := 46.20
  let orange_juice_bottles := 42
  let apple_juice_bottles := total_bottles - orange_juice_bottles
  let equation := (orange_juice_bottles * O + apple_juice_bottles * 0.60 = total_cost)
  equation -> O = 0.70 := by
  sorry

end cost_of_orange_juice_l501_501431


namespace order_of_numbers_l501_501678

def a : ℝ := (1 / 2) ^ 0.1
def b : ℝ := (1 / 2) ^ (-0.1)
def c : ℝ := (1 / 2) ^ 0.2

theorem order_of_numbers : c < a ∧ a < b := by
  -- We provide the theorem statement and will skip the proof.
  sorry

end order_of_numbers_l501_501678


namespace correct_propositions_l501_501253

def proposition_1 (P : Type) [plane : AddCommGroup P] : Prop :=
  ∀ (p : P) (L : set P), p ∉ L → ∃ (l : set P), l ⊆ L ∧ p ∈ l

def proposition_2 (P : Type) [plane : AddCommGroup P] : Prop :=
  ∀ (p : P) (l : P), p ≠ l → ∃ (P' : set P), P' ⊆ P ∧ p ∈ P' ∧ l ⊆ P'

def proposition_3 (P : Type) [plane : AddCommGroup P] : Prop :=
  ∀ (L₁ L₂ : P) (P' : Type) [plane' : AddCommGroup P'], L₁ ⊆ P' → L₂ ⊆ P' → P = P'

def proposition_4 (P : Type) [plane : AddCommGroup P] : Prop :=
  ∀ (P₁ P₂ P₃ : P), P₁ ∩ P₃ ≠ ∅ → P₂ ∩ P₃ ≠ ∅ → (∀ (L : set P), L ⊆ P₁ ∩ P₃ → L ⊆ P₂ ∩ P₃)

theorem correct_propositions : proposition_1 P ∧ proposition_2 P := 
  sorry

end correct_propositions_l501_501253


namespace probability_of_two_white_balls_l501_501462

-- Define the total number of balls
def total_balls : ℕ := 11

-- Define the number of white balls
def white_balls : ℕ := 5

-- Define the number of ways to choose 2 out of n (combinations)
def choose (n r : ℕ) : ℕ := n.choose r

-- Define the total combinations of drawing 2 balls out of 11
def total_combinations : ℕ := choose total_balls 2

-- Define the combinations of drawing 2 white balls out of 5
def white_combinations : ℕ := choose white_balls 2

-- Define the probability of drawing 2 white balls
noncomputable def probability_white : ℚ := (white_combinations : ℚ) / (total_combinations : ℚ)

-- Now, state the theorem that states the desired result
theorem probability_of_two_white_balls : probability_white = 2 / 11 := sorry

end probability_of_two_white_balls_l501_501462


namespace arithmetic_sequence_diff_l501_501055

theorem arithmetic_sequence_diff (b : ℕ → ℚ) (h_arith : ∀ n : ℕ, b (n + 1) = b n + (b 2 - b 1))
  (h_sum1 : (Finset.range 150).sum (λ n, b (n + 1)) = 150)
  (h_sum2 : (Finset.range 150).sum (λ n, b (n + 151)) = 450) :
  b 2 - b 1 = 1 / 75 :=
by
  sorry

end arithmetic_sequence_diff_l501_501055


namespace yellow_jelly_bean_probability_l501_501135

theorem yellow_jelly_bean_probability :
  ∀ (p_red p_orange p_blue p_yellow : ℝ),
    p_red = 0.25 →
    p_orange = 0.40 →
    p_blue = 0.15 →
    (p_red + p_orange + p_blue + p_yellow = 1) →
    p_yellow = 0.20 :=
by
  intros p_red p_orange p_blue p_yellow h_red h_orange h_blue h_sum
  rw [h_red, h_orange, h_blue] at h_sum
  linarith

sorry

end yellow_jelly_bean_probability_l501_501135


namespace sum_all_possible_a1_l501_501231

-- Definitions for the sequence and conditions
def sequence_condition (k : ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop :=
  a (n + 1) = k * a n + 3 * k - 3

def valid_a1 (a1 : ℝ) : Prop :=
  a1 ∈ {-678, -78, -3, 22, 222, 2222}

def sum_of_valid_a1 (sum : ℝ) : Prop :=
  sum = -3 - 34 / 3 + 2022

-- The main theorem statement
theorem sum_all_possible_a1 (k : ℝ) (a : ℕ → ℝ) (sum : ℝ) :
  (∀ n, sequence_condition k a n ∧ valid_a1 (a 1)) → sum_of_valid_a1 sum :=
by
  sorry

end sum_all_possible_a1_l501_501231


namespace infinite_k_terms_l501_501644

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
∀ n : ℕ+, a (n + 2) = | a (n + 1) - a n |

def gcd_condition (a1 a2 : ℕ) : ℕ :=
nat.gcd a1 a2

theorem infinite_k_terms (a : ℕ → ℕ) (a1 a2 : ℕ) (k : ℕ)
  (h1 : a 1 = a1) (h2 : a 2 = a2)
  (h3 : sequence a)
  (h4 : k = gcd_condition a1 a2) :
  ∃ᶠ n in at_top, a n = k :=
sorry

end infinite_k_terms_l501_501644


namespace non_congruent_squares_count_l501_501997

-- Definition of the grid size
def grid_size : ℕ := 6

-- Function to count standard squares of size n x n on a grid_size x grid_size grid
def count_standard_squares (n : ℕ) : ℕ :=
  if n < grid_size then (grid_size - n) * (grid_size - n) else 0

-- Function to count diagonal squares formed by 1 x n rectangles
def count_diagonal_squares (n : ℕ) : ℕ :=
  if n < grid_size then (grid_size - n) * (grid_size - n) else 0

-- Total number of non-congruent squares
def total_non_congruent_squares : ℕ :=
  (0 : ℕ).sum (λ n, count_standard_squares n) +
  (0 : ℕ).sum (λ n, count_diagonal_squares n)

-- The theorem that needs to be proven
theorem non_congruent_squares_count : total_non_congruent_squares = 110 := sorry

end non_congruent_squares_count_l501_501997


namespace ab_is_10_pow_116_l501_501371

noncomputable def ab (a b : ℝ) : ℝ :=
  if 2 * (Real.sqrt (Real.log a) + Real.sqrt (Real.log b)) + Real.log (Real.sqrt a) + Real.log (Real.sqrt b) = 108
  then a * b
  else 0

theorem ab_is_10_pow_116 (a b : ℝ) 
  (hPosa : 0 < a) (hPosb : 0 < b) 
  (h : 2 * (Real.sqrt (Real.log a) + Real.sqrt (Real.log b)) + Real.log (Real.sqrt a) + Real.log (Real.sqrt b) = 108) :
  ab a b = 10^116 :=
by
  sorry

end ab_is_10_pow_116_l501_501371


namespace max_distance_point_to_line_l501_501606

def circle_C (m : ℝ) : Set (ℝ × ℝ) := { p | (p.1)^2 + (p.2)^2 - 4 * p.1 + m = 0 }
def circle_C_prime : Set (ℝ × ℝ) := { p | (p.1 - 3)^2 + (p.2 + 2*Real.sqrt 2)^2 = 4 }
def line_l : Set (ℝ × ℝ) := { p | 3 * p.1 - 4 * p.2 + 4 = 0 }

theorem max_distance_point_to_line (m : ℝ) (P : ℝ × ℝ) (P_on_C : P ∈ circle_C m):
  m = 3 →
  ((∀ Q ∈ circle_C m, dist Q line_l = 3) ∧ (∀ P ∈ circle_C_prime, dist P line_l ≠ 3)) := sorry

end max_distance_point_to_line_l501_501606


namespace intervals_of_monotonicity_a3_range_of_a_l501_501254

-- Define the function f(x) and its derivative when a = 3
def f (x : ℝ) (a : ℝ) : ℝ := 2*x - (1 / x) - a * Real.log x

def f' (x : ℝ) : ℝ := (2 * x^2 - 3 * x + 1) / (x^2)

-- The first part of the problem
theorem intervals_of_monotonicity_a3 :
  ∀ x : ℝ, 
  (x > 0 → f' x > 0 → ((x < 1/2) ∨ (x > 1))) ∧
  (x > 0 → f' x < 0 → (1/2 < x ∧ x < 1)) :=
sorry

-- Define the function g(x) and its derivative
def g (x : ℝ) (a : ℝ) : ℝ := f x a - x + 2 * a * Real.log x

def g' (x : ℝ) (a : ℝ) : ℝ := (x^2 + a * x + 1) / x^2

-- The second part of the problem
theorem range_of_a :
  ∀ a : ℝ, 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g' x1 a = 0 ∧ g' x2 a = 0) ↔ (a < -2) :=
sorry

end intervals_of_monotonicity_a3_range_of_a_l501_501254


namespace work_problem_l501_501853

theorem work_problem (hA : ∀ n : ℝ, n = 15)
  (h_work_together : ∀ n : ℝ, 3 * (1/15 + 1/n) = 0.35) :  
  1/20 = 1/20 :=
by
  sorry

end work_problem_l501_501853


namespace solve_exponential_eq_l501_501050

theorem solve_exponential_eq (x : ℝ) : 4^x - 2^x - 6 = 0 ↔ x = log 2 3 := by
  sorry

end solve_exponential_eq_l501_501050


namespace expected_num_reflections_l501_501074

noncomputable def expectedReflections
  (length : ℝ) 
  (width : ℝ) 
  (initial_position : ℝ × ℝ) 
  (travel_distance : ℝ) : ℝ :=
1 + 2 / (real.pi) *
      (real.arccos (3 / 4) + real.arccos (1 / 4) -
       real.arcsin (3 / 4))

theorem expected_num_reflections (length width : ℝ)
  (h_length : length = 3)
  (h_width : width = 1)
  (initial_position : ℝ × ℝ)
  (h_initial_position : initial_position = (1.5, 0.5))
  (travel_distance : ℝ)
  (h_travel_distance : travel_distance = 2) :
  expectedReflections length width initial_position travel_distance = 1.7594 :=
by
  sorry

end expected_num_reflections_l501_501074


namespace sum_of_x_coordinates_mod_17_l501_501456

theorem sum_of_x_coordinates_mod_17 :
  let m := 17
  let points := { p : ℤ × ℤ | ∃ (x y : ℤ), y ≡ 6 * x + 3 [ZMOD m] ∧ y ≡ 13 * x + 8 [ZMOD m] }
  let xs := { x : ℤ | ∃ y : ℤ, (x, y) ∈ points }
  (xs.sum) % m = 7 :=
by
  -- Definitions of constants and sets
  let m := 17
  let points := { p : ℤ × ℤ | ∃ (x y : ℤ), y ≡ 6 * x + 3 [ZMOD m] ∧ y ≡ 13 * x + 8 [ZMOD m] }
  let xs := { x : ℤ | ∃ y : ℤ, (x, y) ∈ points }
  
  -- Start of proof
  sorry

end sum_of_x_coordinates_mod_17_l501_501456


namespace systems_on_second_street_l501_501709

-- Definitions based on the conditions
def commission_per_system : ℕ := 25
def total_commission : ℕ := 175
def systems_on_first_street (S : ℕ) := S / 2
def systems_on_third_street : ℕ := 0
def systems_on_fourth_street : ℕ := 1

-- Question: How many security systems did Rodney sell on the second street?
theorem systems_on_second_street (S : ℕ) :
  S / 2 + S + 0 + 1 = total_commission / commission_per_system → S = 4 :=
by
  intros h
  sorry

end systems_on_second_street_l501_501709


namespace vector_length_AB_l501_501314

def A : ℝ × ℝ × ℝ := (0, 1, 2)
def B : ℝ × ℝ × ℝ := (1, 2, 3)

def vector_length (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem vector_length_AB : vector_length (B.1 - A.1, B.2 - A.2, B.3 - A.3) = Real.sqrt 3 := by sorry

end vector_length_AB_l501_501314


namespace cricket_team_throwers_l501_501031

open_locale classical

theorem cricket_team_throwers (T N : ℕ) 
    (h1 : N + T = 67) 
    (h2 : (2/3 : ℚ) * N = 57 - T) : 
    T = 37 := 
by 
    sorry

end cricket_team_throwers_l501_501031


namespace range_of_function_l501_501814

theorem range_of_function :
  ∀ (y : ℝ),
    (∃ x : ℝ, x ≠ -1 ∧ y = (x^2 + 4*x + 3)/(x + 1)) ↔ (y ∈ Set.Ioo (-∞) 2 ∪ Set.Ioo 2 ∞) :=
by
  sorry

end range_of_function_l501_501814


namespace max_abs_diff_chain_l501_501936

namespace MaximumExpression

def abs_diff_chain : (List ℕ) → ℕ
| [] => 0
| [x] => x
| (x :: y :: xs) => abs_diff_chain ((|x - y|) :: xs)

theorem max_abs_diff_chain :
  ∀ (l : List ℕ),
  (∀ x ∈ l, x ∈ Finset.range 1991 ∧ (l.nodup ∧ l.length = 1990)) →
  abs_diff_chain l ≤ 1989 :=
by
  sorry

end MaximumExpression

end max_abs_diff_chain_l501_501936


namespace triangle_area_l501_501702

theorem triangle_area (a b c R : ℝ) (A B C : ℝ) (hA : sin A = a / (2 * R)) (hB : sin B = b / (2 * R)) (hC : sin C = c / (2 * R)) :
  let S := (1 / 2) * a * b * sin C
  in S = (a * b * c) / (4 * R) :=
by
  sorry

end triangle_area_l501_501702


namespace count_integers_congruent_to_one_mod_seven_l501_501276

theorem count_integers_congruent_to_one_mod_seven (N : ℕ) (hN : N = 300) :
  ∃ n : ℕ, n = (finset.filter (λ k, k % 7 = 1) (finset.range (N + 1))).card ∧ n = 43 :=
by {
  sorry
}

end count_integers_congruent_to_one_mod_seven_l501_501276


namespace popcorn_kernels_needed_l501_501115

theorem popcorn_kernels_needed
  (h1 : 2 * 4 = 4 * 1) -- Corresponds to "2 tablespoons make 4 cups"
  (joanie : 3) -- Joanie wants 3 cups
  (mitchell : 4) -- Mitchell wants 4 cups
  (miles_davis : 6) -- Miles and Davis together want 6 cups
  (cliff : 3) -- Cliff wants 3 cups
  : 2 * (joanie + mitchell + miles_davis + cliff) / 4 = 8 :=
by sorry

end popcorn_kernels_needed_l501_501115


namespace min_distance_l501_501541

theorem min_distance (x y : ℝ) (h1 : 8 * x + 15 * y = 120) (h2 : x ≥ 0) :
  ∃ (d : ℝ), d = sqrt (x^2 + y^2) ∧ d = 120 / 17 :=
by 
  sorry

end min_distance_l501_501541


namespace polar_equation_of_ellipse_l501_501554

/-- The polar equation of an ellipse given the Cartesian equation
  (x^2/a^2) + (y^2/b^2) = 1 with specified conditions. -/
theorem polar_equation_of_ellipse 
  (a b ρ φ : ℝ)
  (h_conditions : ∀ x y, (x = ρ * cos φ) ∧ (y = ρ * sin φ) ∧ (ρ > 0))
  (h_equation : (ρ * cos φ)^2 / a^2 + (ρ * sin φ)^2 / b^2 = 1) :
    ρ^2 = b^2 / (1 - ((1 - b^2/a^2) * (cos φ)^2)) :=
by
  sorry

end polar_equation_of_ellipse_l501_501554


namespace gcd_24_36_54_l501_501737

-- Define the numbers and the gcd function
def num1 : ℕ := 24
def num2 : ℕ := 36
def num3 : ℕ := 54

-- The Lean statement to prove that the gcd of num1, num2, and num3 is 6
theorem gcd_24_36_54 : Nat.gcd (Nat.gcd num1 num2) num3 = 6 := by
  sorry

end gcd_24_36_54_l501_501737


namespace volume_of_water_displaced_l501_501467

theorem volume_of_water_displaced (r h s : ℝ) (V : ℝ) 
  (r_eq : r = 5) (h_eq : h = 12) (s_eq : s = 6) :
  V = s^3 :=
by
  have cube_volume : V = s^3 := by sorry
  show V = s^3
  exact cube_volume

end volume_of_water_displaced_l501_501467


namespace probability_of_choosing_two_yellow_apples_l501_501694

theorem probability_of_choosing_two_yellow_apples :
  let total_apples := 10
  let red_apples := 6
  let yellow_apples := 4
  let total_ways_to_choose_2 := (total_apples.choose 2)
  let ways_to_choose_2_yellow := (yellow_apples.choose 2)
  (ways_to_choose_2_yellow : ℚ) / total_ways_to_choose_2 = 2 / 15 := by
sorry

end probability_of_choosing_two_yellow_apples_l501_501694


namespace third_side_correct_length_longest_side_feasibility_l501_501501

-- Definitions for part (a)
def adjacent_side_length : ℕ := 40
def total_fencing_length : ℕ := 140

-- Define third side given the conditions
def third_side_length : ℕ :=
  total_fencing_length - (2 * adjacent_side_length)

-- Problem (a)
theorem third_side_correct_length (hl : adjacent_side_length = 40) (ht : total_fencing_length = 140) :
  third_side_length = 60 :=
sorry

-- Definitions for part (b)
def longest_side_possible1 : ℕ := 85
def longest_side_possible2 : ℕ := 65

-- Problem (b)
theorem longest_side_feasibility (hl : adjacent_side_length = 40) (ht : total_fencing_length = 140) :
  ¬ (longest_side_possible1 = 85 ∧ longest_side_possible2 = 65) :=
sorry

end third_side_correct_length_longest_side_feasibility_l501_501501


namespace abc_cubic_sum_identity_l501_501333

theorem abc_cubic_sum_identity (a b c : ℂ) 
  (M : Matrix (Fin 3) (Fin 3) ℂ)
  (h1 : M = fun i j => if i = 0 then (if j = 0 then a else if j = 1 then b else c)
                      else if i = 1 then (if j = 0 then b else if j = 1 then c else a)
                      else (if j = 0 then c else if j = 1 then a else b))
  (h2 : M ^ 3 = 1)
  (h3 : a * b * c = -1) :
  a^3 + b^3 + c^3 = 4 := sorry

end abc_cubic_sum_identity_l501_501333


namespace polygon_sides_with_diagonals_44_l501_501922

theorem polygon_sides_with_diagonals_44 (n : ℕ) (hD : 44 = n * (n - 3) / 2) : n = 11 :=
by
  sorry

end polygon_sides_with_diagonals_44_l501_501922


namespace number_of_positive_integer_solutions_l501_501446

open Finset

theorem number_of_positive_integer_solutions (n k : ℕ) (h : n = 15) (l : k = 4) : 
  ∃ c : ℕ, c = 364 ∧ fintype.card {t : fin k → ℕ | (∀ i, 0 < t i) ∧ (finset.univ.sum t) = n} = c :=
by
  sorry

end number_of_positive_integer_solutions_l501_501446


namespace exists_subset_with_infinite_multiples_l501_501070

theorem exists_subset_with_infinite_multiples (S : ℕ → ℕ → Prop) (k : ℕ) (partitions : list (set ℕ))
  (part_finite : partitions.length = k)
  (part_disjoint : ∀ i j, i ≠ j → partitions.nth i ∩ partitions.nth j = ∅)
  (part_total : (∅ : set ℕ) = ⋃ i < k, partitions.nth i) :
  ∃ i < k, ∀ n, ∀ m, m > n → m ∈ partitions.nth i :=
sorry

end exists_subset_with_infinite_multiples_l501_501070


namespace x_axis_direction_south_west_l501_501697

-- Let's define the conditions first
variables (a b c : ℝ)
variables (h_ab : a ≠ b) (h_bc : b ≠ c) (h_ca : c ≠ a)
noncomputable def y1 (x : ℝ) : ℝ := a * x + b
noncomputable def y2 (x : ℝ) : ℝ := b * x + c
noncomputable def y3 (x : ℝ) : ℝ := c * x + a

-- Using the given conditions, we aim to prove the direction of the x-axis
theorem x_axis_direction_south_west :
  (cyclic_permuted a b c) →
  -- prove the direction of the positive x-axis
  positive_x_axis_direction = southwest :=
sorry

end x_axis_direction_south_west_l501_501697


namespace cone_radius_l501_501969

theorem cone_radius (surface_area : ℝ) (semi_circle : bool) : 
  semi_circle ∧ surface_area = 24 * Real.pi → 
  let r := 2 * Real.sqrt 2 in 
  r = 2 * Real.sqrt 2 :=
by
  assume h
  let r := 2 * Real.sqrt 2
  sorry

end cone_radius_l501_501969


namespace average_temperature_l501_501760

theorem average_temperature (T_NY T_Miami T_SD : ℝ) (h1 : T_NY = 80) (h2 : T_Miami = T_NY + 10) (h3 : T_SD = T_Miami + 25) :
  (T_NY + T_Miami + T_SD) / 3 = 95 :=
by
  sorry

end average_temperature_l501_501760


namespace center_of_conic_l501_501130

-- Define the conic equation
def conic_equation (p q r α β γ : ℝ) : Prop :=
  p * α * β + q * α * γ + r * β * γ = 0

-- Define the barycentric coordinates of the center
def center_coordinates (p q r : ℝ) : ℝ × ℝ × ℝ :=
  (r * (p + q - r), q * (p + r - q), p * (r + q - p))

-- Theorem to prove that the barycentric coordinates of the center are as expected
theorem center_of_conic (p q r α β γ : ℝ) (h : conic_equation p q r α β γ) :
  center_coordinates p q r = (r * (p + q - r), q * (p + r - q), p * (r + q - p)) := 
sorry

end center_of_conic_l501_501130


namespace jaden_toy_cars_l501_501653

theorem jaden_toy_cars :
  let initial : Nat := 14
  let bought : Nat := 28
  let birthday : Nat := 12
  let to_sister : Nat := 8
  let to_friend : Nat := 3
  initial + bought + birthday - to_sister - to_friend = 43 :=
by
  let initial : Nat := 14
  let bought : Nat := 28
  let birthday : Nat := 12
  let to_sister : Nat := 8
  let to_friend : Nat := 3
  show initial + bought + birthday - to_sister - to_friend = 43
  sorry

end jaden_toy_cars_l501_501653


namespace M_less_than_new_N_l501_501142

theorem M_less_than_new_N (N M : ℝ) (hN : N > 0) (hM : M > 0):
  (M < 1.5 * N) → ((100 - (200 * M) / (3 * N)) : ℝ) :=
by
  sorry

end M_less_than_new_N_l501_501142


namespace Randy_used_blocks_l501_501707

theorem Randy_used_blocks (initial_blocks blocks_left used_blocks : ℕ) 
  (h1 : initial_blocks = 97) 
  (h2 : blocks_left = 72) 
  (h3 : used_blocks = initial_blocks - blocks_left) : 
  used_blocks = 25 :=
by
  sorry

end Randy_used_blocks_l501_501707


namespace valid_complex_numbers_count_l501_501011

noncomputable def count_valid_complex_numbers : ℕ :=
  sorry -- Placeholder; the correct quantity 'n' 

theorem valid_complex_numbers_count (g : ℂ → ℂ)
  (h1 : ∀ z, g z = z^2 + 2 * complex.I * z + 2)
  (h2 : ∀ z, complex.im z > 0)
  (h3 : ∀ z, ∃ a b : ℤ, g z = a + b * complex.I ∧ |a| ≤ 5 ∧ |b| ≤ 5) :
  ∃ n, count_valid_complex_numbers = n :=
sorry

end valid_complex_numbers_count_l501_501011


namespace sum_of_first_20_terms_c_n_l501_501221

theorem sum_of_first_20_terms_c_n (a b c : ℕ → ℕ) (d q : ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + d)
  (h3 : ∀ n, b (n + 1) = b n * q) (h4 : b 1 = 2) (h5 : a 4 = b 2) (h6 : a 8 = b 3) 
  (h7 : ∀ n, c n = b n - ite (log 4 (b n) = a k ∨ b n = 3 * a k + 1) (b n) 0) :
  ∑ i in finset.range 20, c (i + 1) = (2^41 - 2) / 3 := sorry

end sum_of_first_20_terms_c_n_l501_501221


namespace min_value_a4b3c2_l501_501677

noncomputable def a (x : ℝ) : ℝ := if x > 0 then x else 0
noncomputable def b (x : ℝ) : ℝ := if x > 0 then x else 0
noncomputable def c (x : ℝ) : ℝ := if x > 0 then x else 0

theorem min_value_a4b3c2 (a b c : ℝ) 
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h : 1/a + 1/b + 1/c = 9) : a^4 * b^3 * c^2 ≥ 1/1152 :=
by sorry

example : ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ (1/a + 1/b + 1/c = 9) ∧ a^4 * b^3 * c^2 = 1/1152 :=
by 
  use [1/4, 1/3, 1/2]
  split
  norm_num -- 0 < 1/4
  split
  norm_num -- 0 < 1/3
  split
  norm_num -- 0 < 1/2
  split
  norm_num -- 1/(1/4) + 1/(1/3) + 1/(1/2) = 9
  norm_num -- (1/4)^4 * (1/3)^3 * (1/2)^2 = 1/1152

end min_value_a4b3c2_l501_501677


namespace polygon_interior_exterior_relation_l501_501753

theorem polygon_interior_exterior_relation (n : ℕ) (h1 : (n-2) * 180 = 2 * 360) : n = 6 :=
by sorry

end polygon_interior_exterior_relation_l501_501753


namespace right_triangle_area_l501_501803

theorem right_triangle_area (a b c : ℝ) (ht : a^2 + b^2 = c^2) (h1 : a = 24) (h2 : c = 26) : 
    (1/2) * a * b = 120 :=
begin
  sorry
end

end right_triangle_area_l501_501803


namespace hexagon_labeling_l501_501382

-- Definitions
variables (A B C D E F G : ℕ)
variables (f : fin 7 → ℕ)

-- Assume each variable represents a distinct digit from 1 to 7.
def all_digits := {1, 2, 3, 4, 5, 6, 7}

-- The condition that all digits are distinct and within the range 1 to 7.
def valid_digits : Prop :=
  f = finset.univ.image digit_of_fintype.to_nat ∧ 
  finset.univ.image f.card = 7

-- Given sums
def sum_line (x y z : ℕ) := x + y + z 

-- x is the common sum
def common_sum (x : ℕ) := 
  sum_line A G C = x ∧
  sum_line B G D = x ∧
  sum_line C G E = x

-- Proposition to prove the number of valid arrangements
theorem hexagon_labeling :
  valid_digits f →
  (∃ x, common_sum f A B C D E F G x) →
  (∀ g ∈ all_digits, ∃! g, count_valid_arrangements A B C D E F G = 144) :=
sorry

end hexagon_labeling_l501_501382


namespace eccentricity_range_l501_501240

-- Definitions and assumptions based on given problem
def is_foci (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ F1 F2 : ℝ × ℝ, F1 = (-c, 0) ∧ F2 = (c, 0) ∧ c = sqrt (a^2 - b^2)

def on_ellipse (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  P = (x, y) ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def right_angle (P F1 F2 : ℝ × ℝ) : Prop :=
  ∠PF1F2 = 90

noncomputable def eccentricity (a b c : ℝ) : ℝ :=
  c / a

theorem eccentricity_range 
  (a b c : ℝ) (P : ℝ × ℝ) 
  (ha : a > 0) (hb : 0 < b) (hab : a > b) 
  (foci_cond : is_foci a b P) 
  (ellipse_cond : on_ellipse a b P) 
  (angle_cond : right_angle P (-c, 0) (c, 0)) :
  √2/2 ≤ eccentricity a b c ∧ eccentricity a b c < 1 :=
sorry

end eccentricity_range_l501_501240


namespace quadratic_has_distinct_real_roots_l501_501294

theorem quadratic_has_distinct_real_roots (m : ℝ) :
  ∃ (a b c : ℝ), a = 1 ∧ b = -2 ∧ c = m - 1 ∧ (b^2 - 4 * a * c > 0) → (m < 2) :=
by
  sorry

end quadratic_has_distinct_real_roots_l501_501294


namespace second_player_wins_for_n_11_l501_501662

theorem second_player_wins_for_n_11 (N : ℕ) (h1 : N = 11) :
  ∃ (list : List ℕ), (∀ x ∈ list, x > 0 ∧ x ≤ 25) ∧
     list.sum ≥ 200 ∧
     (∃ sublist : List ℕ, sublist.sum ≥ 200 - N ∧ sublist.sum ≤ 200 + N) :=
by
  let N := 11
  sorry

end second_player_wins_for_n_11_l501_501662


namespace tan_arithmetic_progression_l501_501316

variables {A B C : ℝ} -- Angles of the triangle
variable  [Fact (A + B + C = π)] -- Sum of angles in a triangle

-- Condition for acute-angled triangle
variables (hA : A < π / 2) (hB : B < π / 2) (hC : C < π / 2)

-- Condition for circumcenter and centroid
variable {O G : Point} -- O is circumcenter, G is centroid (point definitions assumed)
variable (hCircumcenter : is_circumcenter O A B C)
variable (hCentroid : is_centroid G A B C)
variable (hParallel : is_parallel OG AC)

noncomputable def is_arithmetic_sequence_tan : Prop :=
  2 * Real.tan B = Real.tan A + Real.tan C

theorem tan_arithmetic_progression 
  (hA : A < π / 2) (hB : B < π / 2) (hC : C < π / 2) 
  (hCircumcenter : is_circumcenter O A B C)
  (hCentroid : is_centroid G A B C)
  (hParallel : is_parallel OG AC) :
  is_arithmetic_sequence_tan A B C :=
sorry

end tan_arithmetic_progression_l501_501316


namespace num_arithmetic_sequences_l501_501958

theorem num_arithmetic_sequences (d : ℕ) (S_n : ℕ) : d = 2 → S_n = 2016 → 34 = ({ n : ℕ // n ≥ 3 ∧ ∃ a_1 : ℤ, n * (a_1 + n - 1) = S_n }.to_finset.card) :=
by
  intros h_d h_S_n
  rw h_d at *
  rw h_S_n at *
  sorry

end num_arithmetic_sequences_l501_501958


namespace decreasing_intervals_l501_501978

noncomputable def f (ω a x : ℝ) : ℝ := sin(ω * x) + a * cos(ω * x)

theorem decreasing_intervals (ω a : ℝ) (hω : ω > 0) (h1 : f ω a π = (1/2) * sqrt(a^2 + 1))
  (h2 : f ω a (3 * π) = (1/2) * sqrt(a^2 + 1))
  (h3 : f ω a (7 * π) = (1/2) * sqrt(a^2 + 1)) :
  ∀ k : ℤ, ∀ x : ℝ, (6 * k * π + 2 * π ≤ x ∧ x ≤ 6 * k * π + 5 * π) → (f ω a).deriv x < 0 := 
sorry

end decreasing_intervals_l501_501978


namespace isosceles_BMC_l501_501729

open EuclideanGeometry

variable {A B C D M O : Point ℝ}

-- Given conditions in the problem
variables (h1 : Trapezoid A B C D)
          (h2 : ∃ (O : Point ℝ), IsDiagonalIntersection h1 A C B D O)
          (h3 : ∃ (circ1 circ2 : Circle ℝ), 
                 CircumscribedTriangle circ1 A O B ∧
                 CircumscribedTriangle circ2 C O D ∧
                 IntersectsAtPoint circ1 circ2 M ∧
                 M ∈ Line.segment A D)

-- The goal to prove
theorem isosceles_BMC : BM = CM :=
by
  cases h2 with O hO,
  cases h3 with circ1 hcirc1,
  cases hcirc1 with circ2 hcirc2,
  sorry

end isosceles_BMC_l501_501729


namespace roots_polynomial_identity_l501_501041

theorem roots_polynomial_identity (a b x₁ x₂ : ℝ) 
  (h₁ : x₁^2 + b*x₁ + b^2 + a = 0) 
  (h₂ : x₂^2 + b*x₂ + b^2 + a = 0) : x₁^2 + x₁*x₂ + x₂^2 + a = 0 :=
by 
  sorry

end roots_polynomial_identity_l501_501041


namespace term_2023_of_sequence_l501_501942

theorem term_2023_of_sequence :
  ∀ (a : ℕ → ℕ), 
    (∀ n : ℕ, n > 0 → (∀ m, m > 0 ∧ m ≤ n → a m) → (∑ i in finset.range (n + 1), a i) = n^2) →
    a 2023 = 4045 := 
by 
  sorry

end term_2023_of_sequence_l501_501942


namespace probability_of_not_shorter_than_one_meter_l501_501388

noncomputable def probability_of_event_A : ℝ := 
  let length_of_rope : ℝ := 3
  let event_A_probability : ℝ := 1 / 3
  event_A_probability

theorem probability_of_not_shorter_than_one_meter (l : ℝ) (h_l : l = 3) : 
    probability_of_event_A = 1 / 3 :=
sorry

end probability_of_not_shorter_than_one_meter_l501_501388


namespace students_taking_chem_or_phys_not_both_l501_501426

def students_taking_both : ℕ := 12
def students_taking_chemistry : ℕ := 30
def students_taking_only_physics : ℕ := 18

theorem students_taking_chem_or_phys_not_both : 
  (students_taking_chemistry - students_taking_both) + students_taking_only_physics = 36 := 
by
  sorry

end students_taking_chem_or_phys_not_both_l501_501426


namespace injectivity_of_composition_l501_501337

variable {R : Type*} [LinearOrderedField R]

def injective (f : R → R) := ∀ a b, f a = f b → a = b

theorem injectivity_of_composition {f g : R → R} (h : injective (g ∘ f)) : injective f :=
by
  sorry

end injectivity_of_composition_l501_501337


namespace kite_max_area_is_correct_l501_501339

variable {A B C D : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

noncomputable def maximum_area (AB AD : ℝ) (BC CD : ℝ) (equilateral_centroids : Prop) : ℝ :=
  if AB = AD ∧ BC = 3 ∧ CD = 5 ∧ equilateral_centroids
  then 12.25 * Real.sqrt 3 + 7.5
  else 0

theorem kite_max_area_is_correct :
  ∀ (AB AD : ℝ) (BC CD : ℝ) (equilateral_centroids : Prop),
    AB = AD ∧ BC = 3 ∧ CD = 5 ∧ equilateral_centroids →
    maximum_area AB AD BC CD equilateral_centroids = 12.25 * Real.sqrt 3 + 7.5 :=
by
  intros
  sorry

end kite_max_area_is_correct_l501_501339


namespace correct_operation_l501_501832

variable (a b : ℝ)

theorem correct_operation : (-a * b^2)^2 = a^2 * b^4 :=
  sorry

end correct_operation_l501_501832


namespace sqrt_defined_iff_x_le_l501_501218

theorem sqrt_defined_iff_x_le (x : ℝ) : (∃ y : ℝ, y = sqrt (5 - 3 * x)) ↔ x ≤ 5 / 3 :=
by sorry

end sqrt_defined_iff_x_le_l501_501218


namespace area_ORQ_l501_501322

def hexagon (P Q R S T U : Type) : Prop :=
  -- Definition of an equiangular hexagon

def square (A B C D : Type) (area : ℝ) : Prop :=
  -- Definition of a square with given area

def equilateral_triangle (A B C : Type) : Prop :=
  -- Definition of an equilateral triangle

variables (P Q R S T U N M O : Type)

def UT : ℝ := 7
def QR : ℝ := 7
def NM : ℝ := 5
def MO : ℝ := 5

axiom hex_eq : hexagon P Q R S T U
axiom sq1 : square P Q N M 25
axiom sq2 : square U T S R 49
axiom tri_eq : equilateral_triangle N M O
axiom eq_sides : UT = QR

theorem area_ORQ : 
  let ORQ := 
    1 / 2 * QR * MO 
  in ORQ = 17.5 := by 
    sorry

end area_ORQ_l501_501322


namespace measure_of_angle_PMN_l501_501306

-- Given Conditions
variables {P Q R M N : Type}
variables (PR PQ : ℝ)
variables (angle_PQR angle_PRQ : ℝ)

-- Assumptions
axiom angle_PQR_eq_60 : angle_PQR = 60
axiom PR_gt_PQ : PR > PQ
axiom MP_meet_four_equal_angles : ∀ (angle1 angle2 angle3 angle4 : ℝ), 
  angle1 = angle2 ∧ angle2 = angle3 ∧ angle3 = angle4
axiom triangle_PMN_is_isosceles : ∀ (PM PN : ℝ), PM = PN

-- Proof statement
theorem measure_of_angle_PMN (PM PN : ℝ) : PM = PN → ∠PMN = 60 :=
by
  sorry

end measure_of_angle_PMN_l501_501306


namespace evidence_cocyclic_points_l501_501234

section cocyclic_points

variables {A B C D S A' D' : Type*}
variables [cocycle A B C D] [south_pole S A (triangle A B C)]
  [intersection A' B C (line S A)] [intersection D' B C (line S D)]

theorem evidence_cocyclic_points :
  cocyclic A' D' D A :=
by sorry

end cocyclic_points

end evidence_cocyclic_points_l501_501234


namespace sufficient_but_not_necessary_l501_501110

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1) → (x < -1 ∨ x > 1) ∧ ¬((x < -1 ∨ x > 1) → (x < -1)) :=
by
  sorry

end sufficient_but_not_necessary_l501_501110


namespace range_of_m_l501_501604

open Set

variable {m : ℝ}

def A : Set ℝ := { x | x^2 < 16 }
def B (m : ℝ) : Set ℝ := { x | x < m }

theorem range_of_m (h : A ∩ B m = A) : 4 ≤ m :=
by
  sorry

end range_of_m_l501_501604


namespace find_set_a_l501_501598

noncomputable def f (x : ℝ) : ℝ :=
  ∑ i in finset.range 2018, |x + (i + 1)| + ∑ i in finset.range 2017, |x - (i + 1)|

theorem find_set_a :
  {a : ℝ | f (3 * a + 1) = f (2 * a + 2)} = 
  {a | -2/3 ≤ a ∧ a ≤ -1/2} ∪ {1} :=
by
  sorry

end find_set_a_l501_501598


namespace number_of_k_values_l501_501206

theorem number_of_k_values :
  let k (a b : ℕ) := 2^a * 3^b in
  (∀ a b : ℕ, 18 ≤ a ∧ b = 36 → 
  let lcm_val := Nat.lcm (Nat.lcm (9^9) (12^12)) (k a b) in 
  lcm_val = 18^18) →
  (Finset.card (Finset.filter (λ a, 18 ≤ a ∧ a ≤ 24) (Finset.range (24 + 1))) = 7) :=
by
  -- proof skipped
  sorry

end number_of_k_values_l501_501206


namespace hexagon_area_l501_501406

theorem hexagon_area (P Q R P' Q' R' O : Type)
  (h1 : ∀ (A : Type), perpendicular_bisector (PQR A) = circumcircle_meet (A))
  (h2 : is_isosceles_triangle P Q R)
  (h3 : perimeter P Q R = 42)
  (h4 : circumcircle_radius P Q R = 10) :
  ∀ (area_PQ_RP_QR : Type), area_PQ_RP_QR = 210 :=
by sorry

end hexagon_area_l501_501406


namespace diagonals_equal_sides_l501_501905

theorem diagonals_equal_sides (n : ℕ) : ∃ n, (n ≠ 0) ∧ (n ≠ 3) ∧ (∃ k : ℕ, k = D ∧ D = n) :=
by
  have diag_eq_sides : (D = (n * (n - 3)) / 2 ∧ D = n)
  sorry

end diagonals_equal_sides_l501_501905


namespace distance_between_nest_and_ditch_in_meters_l501_501855

-- Definitions of the given conditions
def speed : ℝ := 8 -- speed in km/h
def hours : ℝ := 1.5 -- time in hours
def trips : ℕ := 15 -- number of round trips
def km_to_m (d : ℝ) : ℝ := d * 1000 -- conversion from kilometers to meters

-- Definitions based on the conditions
def total_distance_flown : ℝ := speed * hours -- total distance flown in km
def round_trip_distance : ℝ := total_distance_flown / trips -- distance per round trip in km
def one_way_distance : ℝ := round_trip_distance / 2 -- one-way distance in km

-- The theorem to prove
theorem distance_between_nest_and_ditch_in_meters : km_to_m one_way_distance = 400 := 
sorry

end distance_between_nest_and_ditch_in_meters_l501_501855


namespace probability_of_edge_endpoints_in_icosahedron_l501_501783

theorem probability_of_edge_endpoints_in_icosahedron :
  let vertices := 12
  let edges := 30
  let connections_per_vertex := 5
  (5 / (vertices - 1)) = (5 / 11) := by
  sorry

end probability_of_edge_endpoints_in_icosahedron_l501_501783


namespace polynomials_with_sum_of_abs_values_and_degree_eq_4_l501_501192

-- We define the general structure and conditions of the problem.
def polynomial_count : ℕ := 
  let count_0 := 1 -- For n = 0
  let count_1 := 6 -- For n = 1
  let count_2 := 9 -- For n = 2
  let count_3 := 1 -- For n = 3
  count_0 + count_1 + count_2 + count_3

theorem polynomials_with_sum_of_abs_values_and_degree_eq_4 : polynomial_count = 17 := 
by
  unfold polynomial_count
  -- The detailed proof steps for the count would go here
  sorry

end polynomials_with_sum_of_abs_values_and_degree_eq_4_l501_501192


namespace correctNetIsD_l501_501131

-- Define the vertex structure for a net
structure Vertex :=
  (id : Nat)

-- Define a function to check diagonal intersection at vertices with matching labels
def diagonalsIntersectAtMatchingVertices (net : List (List Vertex)) : Bool :=
  sorry

-- Define the nets A, B, C, D, and E as examples
def netA : List (List Vertex) := sorry
def netB : List (List Vertex) := sorry
def netC : List (List Vertex) := sorry
def netD : List (List Vertex) := sorry
def netE : List (List Vertex) := sorry

-- Theorem that only net D can form the cube with diagonals intersecting at matching vertices
theorem correctNetIsD 
  (nets : List (List (List Vertex))) 
  (validNet : List (List Vertex)) 
  (hA : nets = [netA, netB, netC, netD, netE]) 
  (hD : validNet = netD)
  : list.get nets 3 = validNet := 
by
  -- Proof omitted
  sorry

end correctNetIsD_l501_501131


namespace tangencyPointsCoincide_l501_501336

-- Given a triangle ABC with circumscribed circle L
variables (A B C P : Point)
variables (L : Circle)
-- Assuming internal bisector of angle A meets BC at point P
axiom isInternalBisector (A B C P : Point) : ∃ (L : Circle), isCircumscribed A B C L ∧ meetsAtIntBisectorAngle A B C P

-- Define the circles L_1 and L_2 as specified
variables (L1 L2 : Circle)
axiom isTangentToAPBP (L1 : Circle) (A B P : Point) (L : Circle) : isTangent L1 A P ∧ isTangent L1 B P ∧ isTangent L1 L
axiom isTangentToAPCP (L2 : Circle) (A C P : Point) (L : Circle) : isTangent L2 A P ∧ isTangent L2 C P ∧ isTangent L2 L

-- Prove that the tangent points of L1 and L2 with AP coincide
theorem tangencyPointsCoincide 
    (A B C P : Point) 
    (L : Circle) 
    (L1 L2 : Circle)
    (bisectorProx : isInternalBisector A B C P)
    (tangentL1 : isTangentToAPBP L1 A B P L)
    (tangentL2 : isTangentToAPCP L2 A C P L) : 
    tangentPoint L1 A P = tangentPoint L2 A P :=
sorry

end tangencyPointsCoincide_l501_501336


namespace length_of_crease_proof_l501_501865

-- Define the points in the plane
variables {A B C D: Point}

noncomputable
def length_of_crease (A B C D : Point) : ℝ :=
  if right_triangle A B C 3 4 5 ∧ midpoint D A B then
    15 / 8
  else
    0

-- The proof statement
theorem length_of_crease_proof (A B C D : Point) :
  (right_triangle A B C 3 4 5 ∧ midpoint D A B) → length_of_crease A B C D = 15 / 8 :=
by sorry

end length_of_crease_proof_l501_501865


namespace value_of_sum_l501_501616

theorem value_of_sum (x y z : ℝ) 
    (h1 : x + 2*y + 3*z = 10) 
    (h2 : 4*x + 3*y + 2*z = 15) : 
    x + y + z = 5 :=
by
    sorry

end value_of_sum_l501_501616


namespace ellipse_minor_axis_length_l501_501133

noncomputable def length_minor_axis (points : List (ℝ × ℝ)) : ℝ :=
if points = [(-3/2, 1), (0, 0), (0, 2), (3, 0), (3, 2)] then 4 * real.sqrt 3 / 3 else 0

theorem ellipse_minor_axis_length :
  let points : List (ℝ × ℝ) := [(-3/2, 1), (0, 0), (0, 2), (3, 0), (3, 2)] in
  length_minor_axis points = 4 * real.sqrt 3 / 3 := by
  sorry

end ellipse_minor_axis_length_l501_501133


namespace employees_seating_arrangements_l501_501890

theorem employees_seating_arrangements (n : ℕ) 
  (h_fixed_position : True) 
  (h_circular_arrangement : True) 
  (h_seating_arrangements : (n - 2)! = 120) : n = 7 :=
by sorry

end employees_seating_arrangements_l501_501890


namespace find_k_value_l501_501562

variables {R : Type*} [Field R] {a b x k : R}

-- Definitions for the conditions in the problem
def f (x : R) (a b : R) : R := (b * x + 1) / (2 * x + a)

-- Statement of the problem
theorem find_k_value (h_ab : a * b ≠ 2)
  (h_k : ∀ (x : R), x ≠ 0 → f x a b * f (x⁻¹) a b = k) :
  k = (1 : R) / 4 :=
by
  sorry

end find_k_value_l501_501562


namespace final_percentage_HCl_l501_501118

noncomputable def HCl_percentage_final (vol1 vol2 : ℝ) (perc1 perc2 : ℝ) : ℝ :=
  ((vol1 * perc1 + vol2 * perc2) / (vol1 + vol2)) * 100

theorem final_percentage_HCl : 
  HCl_percentage_final 60 90 0.40 0.15 = 25 := 
begin
  sorry
end

end final_percentage_HCl_l501_501118


namespace parallelogram_division_l501_501866

theorem parallelogram_division (AB AD BE : ℝ) (H1 : AB = 153) (H2 : AD = 180) (H3 : BE = 135) :
  ∃ d1 d2 : ℝ, 
    (d1 = 96 ∧ d2 = 156) ∧ 
    (d1 ∈ set.Icc 0 AD) ∧ 
    (d2 ∈ set.Icc 0 AD) ∧ 
    (d2 > d1) :=
sorry

end parallelogram_division_l501_501866


namespace incorrect_proposition_D_l501_501095

-- Definitions based on conditions
def proposition_A : Prop :=
  ∀ (P1 P2 : Plane) (l : Line), (l ∈ P1 ∧ (∀ (m : Line), m ∈ P2 → l ⊥ m)) → P1 ⊥ P2

def proposition_B : Prop :=
  ∀ (P1 P2 : Plane), (∀ (l : Line), l ∈ P1 → l ∥ P2) → P1 ∥ P2

def proposition_C : Prop :=
  ∀ (P1 P2 : Plane) (l : Line), (l ∥ P1 ∧ (∃ (P : Plane), l ∈ P ∧ P ∩ P2 = l_inter P P2)) → l ∥ l_inter P P2

def proposition_D : Prop :=
  ∀ (l1 l2 : Line) (P : Plane), (proj l1 P ⊥ proj l2 P) → l1 ⊥ l2

-- Incorrectness statement
theorem incorrect_proposition_D : ¬ proposition_D := sorry

end incorrect_proposition_D_l501_501095


namespace solution_z1_solution_z2_area_triangle_OAB_l501_501252

noncomputable def z1 : ℂ := -4 + 3 * ℂ.I
noncomputable def z2 : ℂ := -1 + 2 * ℂ.I

-- Condition 1 for z1
def condition_z1 : Prop := Complex.abs z1 = 1 + 3 * ℂ.I - z1

-- Condition 2 for z2
def condition_z2 : Prop := z2 * (1 - ℂ.I) + (3 - 2 * ℂ.I) = 4 + ℂ.I

-- Definitions of points A and B
def A : ℂ := z1
def B : ℂ := z2

-- Calculation of distances in the complex plane
def dist_OA : ℝ := Complex.abs (A)
def dist_OB : ℝ := Complex.abs (B)

-- Calculate the area of triangle OAB
def area_OAB : ℝ := 0.5 * dist_OA * dist_OB * Real.sin (Complex.arg (B / A))

-- Prove the conditions and area
theorem solution_z1 : z1 = -4 + 3 * ℂ.I := by
  sorry

theorem solution_z2 : z2 = -1 + 2 * ℂ.I := by
  sorry

theorem area_triangle_OAB : area_OAB = 5 / 2 := by
  sorry

end solution_z1_solution_z2_area_triangle_OAB_l501_501252


namespace number_of_zeros_sum_of_reciprocals_gt_4_l501_501591

noncomputable def F (x p : ℝ) : ℝ := (p / x) + Real.log (p * x)
def p : ℝ := 1 / 2

-- Statement (1): Prove that the function F(x) has 2 zeros when p = 1/2
theorem number_of_zeros :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ F x1 p = 0 ∧ F x2 p = 0 :=
sorry

-- Statement (2): Given that x1 and x2 are the zeros of F(x) when p = 1/2, prove that 1/x1 + 1/x2 > 4
theorem sum_of_reciprocals_gt_4 (x1 x2 : ℝ)
  (hx1 : F x1 p = 0) (hx2 : F x2 p = 0) : (1 / x1) + (1 / x2) > 4 :=
sorry

end number_of_zeros_sum_of_reciprocals_gt_4_l501_501591


namespace value_of_a_l501_501238

theorem value_of_a
  (M : Set ℤ := {a^2, a+1, -3})
  (N : Set ℤ := {a-3, 2a-1, a^2+1})
  (h : {−3} = M ∩ N) :
  a = -1 :=
sorry

end value_of_a_l501_501238


namespace inequality_has_three_integers_l501_501628

theorem inequality_has_three_integers (m : ℝ) :
  (∃ f : ℝ → ℝ, 
    (∀ x : ℝ, f x = x^2 - 2 * x + m) ∧ 
    set.countable (set_of (λ x, f x ≤ 0)).to_subtype.count = 3) →
  -3 < m ∧ m ≤ 0 :=
begin
  -- proof will go here
  sorry,
end

end inequality_has_three_integers_l501_501628


namespace width_of_metallic_sheet_l501_501139

-- Define the given conditions
def length_of_sheet : ℝ := 48
def side_of_square_cut : ℝ := 7
def volume_of_box : ℝ := 5236

-- Define the question as a Lean theorem
theorem width_of_metallic_sheet : ∃ (w : ℝ), w = 36 ∧
  volume_of_box = (length_of_sheet - 2 * side_of_square_cut) * (w - 2 * side_of_square_cut) * side_of_square_cut := by
  sorry

end width_of_metallic_sheet_l501_501139


namespace equality_conditions_l501_501183

theorem equality_conditions (a b c d : ℝ) :
  a + bcd = (a + b) * (a + c) * (a + d) ↔ a = 0 ∨ a^2 + a * (b + c + d) + bc + bd + cd = 1 :=
by
  sorry

end equality_conditions_l501_501183


namespace sqrt_1001_1003_plus_1_eq_1002_verify_identity_sqrt_2014_2017_plus_1_eq_2014_2017_l501_501838

-- Define the first proof problem
theorem sqrt_1001_1003_plus_1_eq_1002 : Real.sqrt (1001 * 1003 + 1) = 1002 := 
by sorry

-- Define the second proof problem to verify the identity
theorem verify_identity (n : ℤ) : (n * (n + 3) + 1)^2 = n * (n + 1) * (n + 2) * (n + 3) + 1 :=
by sorry

-- Define the third proof problem
theorem sqrt_2014_2017_plus_1_eq_2014_2017 : Real.sqrt (2014 * 2015 * 2016 * 2017 + 1) = 2014 * 2017 :=
by sorry

end sqrt_1001_1003_plus_1_eq_1002_verify_identity_sqrt_2014_2017_plus_1_eq_2014_2017_l501_501838


namespace g_of_2_l501_501736

noncomputable def g : ℝ → ℝ := sorry

axiom cond1 (x y : ℝ) : x * g y = y * g x
axiom cond2 : g 10 = 30

theorem g_of_2 : g 2 = 6 := by
  sorry

end g_of_2_l501_501736


namespace trees_falling_count_l501_501891

/-- Definition of the conditions of the problem. --/
def initial_mahogany_trees : ℕ := 50
def initial_narra_trees : ℕ := 30
def trees_on_farm_after_typhoon : ℕ := 88

/-- The mathematical proof problem statement in Lean 4:
Prove the total number of trees that fell during the typhoon (N + M) is equal to 5,
given the conditions.
--/
theorem trees_falling_count (M N : ℕ) 
  (h1 : M = N + 1)
  (h2 : (initial_mahogany_trees - M + 3 * M) + (initial_narra_trees - N + 2 * N) = trees_on_farm_after_typhoon) :
  N + M = 5 := sorry

end trees_falling_count_l501_501891


namespace log_50_between_consecutive_integers_l501_501423

theorem log_50_between_consecutive_integers :
    (∃ (m n : ℤ), m < n ∧ m < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < n ∧ m + n = 3) :=
by
  have log_10_eq_1 : Real.log 10 / Real.log 10 = 1 := by sorry
  have log_100_eq_2 : Real.log 100 / Real.log 10 = 2 := by sorry
  have log_increasing : ∀ (x y : ℝ), x < y → Real.log x / Real.log 10 < Real.log y / Real.log 10 := by sorry
  have interval : 10 < 50 ∧ 50 < 100 := by sorry
  use 1
  use 2
  sorry

end log_50_between_consecutive_integers_l501_501423


namespace find_x_l501_501463

theorem find_x :
  ∃ x : ℤ, x * 30 + (12 + 8) * 3 / 5 = 1212 ∧ x = 40 :=
begin
  use 40,
  sorry
end

end find_x_l501_501463


namespace hiking_hours_l501_501797

def violet_water_per_hour : ℕ := 800 -- Violet's water need per hour in ml
def dog_water_per_hour : ℕ := 400    -- Dog's water need per hour in ml
def total_water_capacity : ℚ := 4.8  -- Total water capacity Violet can carry in L

theorem hiking_hours :
  let total_water_per_hour := (violet_water_per_hour + dog_water_per_hour) / 1000 in
  total_water_capacity / total_water_per_hour = 4 :=
by
  let total_water_per_hour := (violet_water_per_hour + dog_water_per_hour) / 1000
  have h1 : violet_water_per_hour = 800 := rfl
  have h2 : dog_water_per_hour = 400 := rfl
  have h3 : total_water_capacity = 4.8 := rfl
  have h4 : total_water_per_hour = 1.2 := by simp [violet_water_per_hour, dog_water_per_hour]
  have h5 : total_water_capacity / total_water_per_hour = 4 := by simp [total_water_capacity, total_water_per_hour]
  exact h5

end hiking_hours_l501_501797


namespace problem_1_problem_2_l501_501558

theorem problem_1 (x : ℝ) 
  (h1 : ∥(sin x, 3 / 4 : ℝ)∥ = ∥(cos x, -1 : ℝ)∥) 
  (h2 : ∥(sin x, 3 / 4 : ℝ)∥ = ∥((k : ℝ) * (cos x, -1 : ℝ))∥)
  (h3 : k ≠ 0) : 
  sin x ^ 2 + 2 * sin x * cos x = -3 / 5 := by {
  sorry
}

theorem problem_2 (A x : ℝ) 
  (m : ℝ × ℝ := (sin x, 3 / 4))
  (n : ℝ × ℝ := (cos x, -1))
  (f : ℝ → ℝ := λ x, 2 * ((sin x + cos x) * cos x - (1 / 4))) 
  (h1 : sin A + cos A = sqrt 2)
  (h2 : (2 * A + π / 4) % (2 * π) = π / 2) : 
  f A = 5 / 2 := by {
  sorry
}

end problem_1_problem_2_l501_501558


namespace num_k_values_lcm_l501_501211

-- Define prime factorizations of given numbers
def nine_pow_nine := 3^18
def twelve_pow_twelve := 2^24 * 3^12
def eighteen_pow_eighteen := 2^18 * 3^36

-- Number of values of k making eighteen_pow_eighteen the LCM of nine_pow_nine, twelve_pow_twelve, and k
def number_of_k_values : ℕ := 
  19 -- Based on calculations from the proof

theorem num_k_values_lcm :
  ∀ (k : ℕ), eighteen_pow_eighteen = Nat.lcm (Nat.lcm nine_pow_nine twelve_pow_twelve) k → ∃ n, n = number_of_k_values :=
  sorry -- Add the proof later

end num_k_values_lcm_l501_501211


namespace minimum_marked_cells_needed_l501_501087

-- Definition of 8x8 chessboard
def Chessboard := (Fin 8) × (Fin 8)

-- Define a subset of marked cells
def is_marked (cells : set Chessboard) (pos : Chessboard) : Prop := pos ∈ cells

-- Define adjacency relation
def is_adjacent (p1 p2 : Chessboard) : Prop :=
  (abs ((p1.fst : Int) - (p2.fst : Int)) + abs ((p1.snd : Int) - (p2.snd : Int)) = 1)

-- The minimum sufficient number of marked cells
def min_marked_cells (cells : set Chessboard) : Prop :=
  ∀ (pos : Chessboard), ∃ (marked : Chessboard), is_adjacent pos marked ∧ is_marked cells marked

-- The main theorem statement
theorem minimum_marked_cells_needed (cells : set Chessboard) (h : cells.card = 20) :
  min_marked_cells cells :=
by sorry

end minimum_marked_cells_needed_l501_501087


namespace square_side_length_l501_501482

theorem square_side_length (s : ℝ) (h : s^2 = 1 / 9) : s = 1 / 3 :=
sorry

end square_side_length_l501_501482


namespace sequence_2018_value_l501_501420

theorem sequence_2018_value :
  let x : ℕ → ℤ := λ n, if n = 1 then 20 else if n = 2 then 17 else x (n - 1) - x (n - 2) in
  x 2018 = 17 := by
  sorry

end sequence_2018_value_l501_501420


namespace variance_of_data_l501_501623

theorem variance_of_data (a : ℝ) (h : (5 + 8 + a + 7 + 4) / 5 = a) : 
  let data := [5, 8, a, 7, 4] in
  a = 6 → 
  Mathlib.Statistics.variance data = 2 := 
by
  intro h' h''
  have : data = [5, 8, 6, 7, 4] := by
    rw h''
  rw this
  sorry

end variance_of_data_l501_501623


namespace find_crew_members_l501_501886

noncomputable def passengers_initial := 124
noncomputable def passengers_texas := passengers_initial - 58 + 24
noncomputable def passengers_nc := passengers_texas - 47 + 14
noncomputable def total_people_virginia := 67

theorem find_crew_members (passengers_initial passengers_texas passengers_nc total_people_virginia : ℕ) :
  passengers_initial = 124 →
  passengers_texas = passengers_initial - 58 + 24 →
  passengers_nc = passengers_texas - 47 + 14 →
  total_people_virginia = 67 →
  ∃ crew_members : ℕ, total_people_virginia = passengers_nc + crew_members ∧ crew_members = 10 :=
by
  sorry

end find_crew_members_l501_501886


namespace max_marks_l501_501149

theorem max_marks (M : ℝ) (h_pass : (0.45 * M = 230)) : M = 512 :=
by
  have frac_to_int : Int.ofNat (Nat.ceil (230 / 0.45)) = 512 :=
    by
      sorry -- This step involves calculating and working with the ceiling function
  rw [← frac_to_int],
  exact Nat.ceil_le,
  norm_num,
  exact h_pass, -- Given condition
  sorry

end max_marks_l501_501149


namespace shaded_area_l501_501319

theorem shaded_area (w h : ℝ) (hw : w = 15) (hh : h = 5) : 
  let area_grid := w * h,
      area_triangle := (1 / 2) * w * h,
      area_shaded := area_grid - area_triangle
  in area_shaded = 37.5 := 
by
  sorry

end shaded_area_l501_501319


namespace midpoint_of_BC_l501_501961

-- Set up the geometric context
variables {A B C D E N K M O : Type*}

-- Define the incircle and points of tangency as given
axiom incircle_of_triangle : ∀ {A B C : Type*}, ∃ (O : Type*), incircle O
axiom points_of_tangency : ∀ {A B C O : Type*}, 
  ∃ (D E N : Type*), tangent D B C ∧ tangent E C A ∧ tangent N A B

-- Define the intersections as per the problem statement
axiom intersection_NO_DE : ∀ {N O D E : Type*}, ∃ K : Type*, line N O ∧ ext N O D E K
axiom intersection_AK_BC : ∀ {A K B C : Type*}, ∃ M : Type*, line A K ∧ ext A K B C M

-- State the goal that M is the midpoint of BC
theorem midpoint_of_BC (h1 : ∃ O, incircle_of_triangle A B C O)
  (h2 : ∃ D E N, points_of_tangency A B C O D E N)
  (h3 : ∃ K, intersection_NO_DE N O D E K)
  (h4 : ∃ M, intersection_AK_BC A K B C M) :
  midpoint M B C :=
by sorry

end midpoint_of_BC_l501_501961


namespace odd_function_inequality_solution_l501_501561

noncomputable def f (x : ℝ) : ℝ := if x > 0 then x - 2 else -(x - 2)

theorem odd_function_inequality_solution :
  {x : ℝ | f x < 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by
  -- A placeholder for the actual proof
  sorry

end odd_function_inequality_solution_l501_501561


namespace bouquet_combinations_l501_501129

/--
Given a budget of $60, roses costing $4 each, and carnations costing $2 each, prove that 
there are 16 different combinations of roses and carnations that sum up to exactly $60.
-/
theorem bouquet_combinations : ∃ (r c : ℕ), 4 * r + 2 * c = 60 ∧ finite_combinations 16 :=
by
  sorry

end bouquet_combinations_l501_501129


namespace find_k_find_a_range_l501_501975

/-- Part 1: Prove k = -1/2 if f(x) is an even function -/
theorem find_k (k : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = log (4^(x : ℝ) + 1) / log 4 + k * x)
  (even_f : ∀ x, f x = f (-x)) : k = -1/2 :=
sorry

/-- Part 2: Prove the range of a such that f(x) and g(x) intersect at exactly one point -/
theorem find_a_range (a : ℝ) (f g : ℝ → ℝ) (h_f : ∀ x, f x = log (4^(x : ℝ) + 1) / log 4 - x)
  (h_g : ∀ x, g x = log x / log 4):
  (∃ x, f x = g (a : ℝ)) ↔ (a > 1 ∨ a = -3) :=
sorry

end find_k_find_a_range_l501_501975


namespace max_tan_alpha_l501_501243

open Real

axiom acute_angle (x : ℝ) : 0 < x ∧ x < π / 2

noncomputable def condition (α β : ℝ) (h₁ : acute_angle α) (h₂ : acute_angle β) : Prop :=
  cos (α + β) = sin α / sin β

theorem max_tan_alpha (α β : ℝ) (h₁ : acute_angle α) (h₂ : acute_angle β) (h₃ : condition α β h₁ h₂) :
  tan α ≤ sqrt 2 / 4 :=
sorry

end max_tan_alpha_l501_501243


namespace tan_20_40_sqrt3_l501_501932

noncomputable def tan (x : ℝ) := Mathlib.Tan.tan x

theorem tan_20_40_sqrt3 
  (h1 : tan 60 = Real.sqrt 3)
  (h2 : tan (20 + 40) = (tan 20 + tan 40) / (1 - tan 20 * tan 40))
  : tan 20 + tan 40 + Real.sqrt 3 * tan 20 * tan 40 = Real.sqrt 3 :=
sorry

end tan_20_40_sqrt3_l501_501932


namespace sin_ratio_eq_sqrt3_l501_501584

theorem sin_ratio_eq_sqrt3
  (A B C : Type)
  [HasSin A] [HasSin B] [HasSin C]
  (area_ABC : ℝ) (c : ℝ) (angle_B : ℝ)
  (h1 : area_ABC = 2 * Real.sqrt 3)
  (h2 : c = 2)
  (h3 : angle_B = Real.pi / 3) :
  (HasSin.sin angle_B / HasSin.sin (A.angle_of_c B C)) = Real.sqrt 3 :=
by
  sorry

end sin_ratio_eq_sqrt3_l501_501584


namespace TA_work_problem_l501_501854

theorem TA_work_problem (N : ℕ) (W : ℕ) (hours_N : ℕ) (hours_N_plus_1 : ℕ) 
  (h1 : hours_N = 5) 
  (h2 : hours_N_plus_1 = 4)    
  (h3 : N * hours_N * (W / (N * hours_N)) = W) 
  (h4 : (N + 1) * hours_N_plus_1 * (W / ((N + 1) * hours_N_plus_1)) = W) 
  : N = 4 → W = 20 → ∀ t : ℕ, t = 1 → t * W / t = 20 := 
by
  intros hN hW t ht
  rw [ht, mul_one, div_one]
  exact hW

end TA_work_problem_l501_501854


namespace money_lent_years_l501_501126

noncomputable def compound_interest_time (A P r n : ℝ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem money_lent_years :
  compound_interest_time 740 671.2018140589569 0.05 1 = 2 := by
  sorry

end money_lent_years_l501_501126


namespace sphere_radius_l501_501650

theorem sphere_radius (r : ℝ) (π : ℝ)
    (h1 : Volume = (4 / 3) * π * r^3)
    (h2 : SurfaceArea = 4 * π * r^2)
    (h3 : Volume = SurfaceArea) :
    r = 3 :=
by
  -- Here starts the proof, but we use 'sorry' to skip it as per the instructions.
  sorry

end sphere_radius_l501_501650


namespace identify_base_7_l501_501698

theorem identify_base_7 :
  ∃ b : ℕ, (b > 1) ∧ 
  (2 * b^4 + 3 * b^3 + 4 * b^2 + 5 * b^1 + 1 * b^0) +
  (1 * b^4 + 5 * b^3 + 6 * b^2 + 4 * b^1 + 2 * b^0) =
  (4 * b^4 + 2 * b^3 + 4 * b^2 + 2 * b^1 + 3 * b^0) ∧
  b = 7 :=
by
  sorry

end identify_base_7_l501_501698


namespace necessary_but_not_sufficient_condition_l501_501223

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (a / (a - 1) ≤ 0) → (0 < a ∧ a < 1) ∧ (¬ (a ≠ 1 → 0 ≤ a ∧ a < 1 → y = a^x → decreasing (λ x, y))) := 
begin
  sorry,
end

end necessary_but_not_sufficient_condition_l501_501223


namespace car_price_l501_501379

/-- Prove that the price of the car Quincy bought is $20,000 given the conditions. -/
theorem car_price (years : ℕ) (monthly_payment : ℕ) (down_payment : ℕ) 
  (h1 : years = 5) 
  (h2 : monthly_payment = 250) 
  (h3 : down_payment = 5000) : 
  (down_payment + (monthly_payment * (12 * years))) = 20000 :=
by
  /- We provide the proof below with sorry because we are only writing the statement as requested. -/
  sorry

end car_price_l501_501379


namespace find_angle_C_range_of_ab_over_c_l501_501324

variables {A B C : ℝ} (a b c : ℝ)

-- Given conditions
def triangle_identity (A B C : ℝ) : Prop :=
  sin A ^ 2 + sin B ^ 2 + sin A * sin B = sin C ^ 2

def angle_opposite_sides (a b c : ℝ) (A B C : ℝ) : Prop :=
  A = angle_of_side a ∧ B = angle_of_side b ∧ C = angle_of_side c

theorem find_angle_C (A B C : ℝ) (a b c : ℝ)
  (h : triangle_identity A B C) :
  C = 2 * π / 3 :=
sorry

theorem range_of_ab_over_c (A B C : ℝ) (a b c : ℝ)
  (h : triangle_identity A B C)
  (hC : C = 2 * π / 3) :
  1 < (a + b) / c ∧ (a + b) / c < 2 * sqrt 3 / 3 :=
sorry

end find_angle_C_range_of_ab_over_c_l501_501324


namespace sequence_tends_to_zero_l501_501716

noncomputable def p (n : ℕ) : ℝ := (1 / (4 * n * Real.sqrt 3)) * Real.exp (Real.sqrt (2 * n / 3))

theorem sequence_tends_to_zero (r : ℝ) (hr : r > 1) :
  Filter.Tendsto (λ n : ℕ, p n / (r ^ n)) Filter.atTop (nhds 0) :=
begin
  sorry
end

end sequence_tends_to_zero_l501_501716


namespace range_of_m_l501_501743

theorem range_of_m 
  (a : ℝ) 
  (m : ℝ) 
  (f : ℝ → ℝ)
  (h_symm : ∀ x : ℝ, f(x) = a + 2 * x - x^2 ∧ f(1 + x) = f(1 - x))
  (h_increasing : ∀ x : ℝ, x ≤ 4 → ∀ (g : ℝ → ℝ), (g = λ x, f(x + m)) → ∀ y1 y2 : ℝ, y1 ≤ y2 → g(y1) ≤ g(y2))
  : m ≤ -3 :=
sorry

end range_of_m_l501_501743


namespace distinct_values_z_l501_501685

theorem distinct_values_z :
  let z_values := λ (a b : ℕ), 9 * (a - b).natAbs in
  ∃ (z_set : Set ℕ), 
    (∀ (x y : ℕ), 10 ≤ x ∧ x < 100 ∧ y = (10 * (x % 10) + (x / 10)) → (z_values (x / 10) (x % 10)) ∈ z_set) ∧
    z_set = {0, 9, 18, 27, 36, 45, 54, 63, 72, 81} ∧
    z_set.card = 10 :=
by
  sorry

end distinct_values_z_l501_501685


namespace ratio_HD_HA_l501_501309

theorem ratio_HD_HA (a b c : ℝ) (h : a^2 + b^2 = c^2) (H : a = 8 ∧ b = 15 ∧ c = 17) :
  let HD := 0 in
  let HA := 8 in
  (HD / HA) = 0 :=
by
  let HD := 0
  let HA := 8
  sorry

end ratio_HD_HA_l501_501309


namespace words_per_page_l501_501851

/-- 
  Let p denote the number of words per page.
  Given conditions:
  - A book contains 154 pages.
  - Each page has the same number of words, p, and no page contains more than 120 words.
  - The total number of words in the book (154p) is congruent to 250 modulo 227.
  Prove that the number of words in each page p is congruent to 49 modulo 227.
 -/
theorem words_per_page (p : ℕ) (h1 : p ≤ 120) (h2 : 154 * p ≡ 250 [MOD 227]) : p ≡ 49 [MOD 227] :=
sorry

end words_per_page_l501_501851


namespace solution_set_of_inequality_l501_501421

theorem solution_set_of_inequality : {x : ℝ | -3 < x ∧ x < 1} = {x : ℝ | x^2 + 2 * x < 3} :=
sorry

end solution_set_of_inequality_l501_501421


namespace repetend_of_4_div_7_l501_501899

theorem repetend_of_4_div_7 : ∃ r : String, r = "571428" ∧ (∃ n : ℕ, (4 / 7 : ℚ) = rat.mk (50 + n) 10 ^ (6 + n) / (10 ^ n) ∧ n ≥ 6) :=
by
  sorry

end repetend_of_4_div_7_l501_501899


namespace classes_to_factories_l501_501480

theorem classes_to_factories : 
  ∃ (f : Fin 5 → Fin 4), Function.Surjective f ∧ (∑ x, 1 = 5) := sorry

end classes_to_factories_l501_501480


namespace first_degree_function_solution_l501_501581

theorem first_degree_function_solution
  (f : ℕ → ℕ)
  (h1 : ∀ x, f x = 2 * x + 7)
  (h2 : ∀ x, 3 * f(x + 1) - 2 * f(x - 1) = 2 * x + 17)
  : f = (λ x, 2 * x + 7) :=
sorry

end first_degree_function_solution_l501_501581


namespace range_of_a_l501_501596

open Real

def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x
def g (x : ℝ) : ℝ := exp x
def h (x : ℝ) : ℝ := log x

theorem range_of_a : 
  ∀ e : ℝ, (e > 0) → 
    (∀ x ∈ set.Icc (1/e) e, ∃ y ∈ set.Icc (1/e) e, 
      f x a = y ∧ f y a = x) → 
    (∃ a ∈ set.Icc 1 (e + 1/e), 
      ∀ x ∈ set.Icc (1/e) e, f(x) = h(x)) :=
by sorry

end range_of_a_l501_501596


namespace orchard_total_mass_l501_501315

def num_gala_trees := 20
def yield_gala_tree := 120
def num_fuji_trees := 10
def yield_fuji_tree := 180
def num_redhaven_trees := 30
def yield_redhaven_tree := 55
def num_elberta_trees := 15
def yield_elberta_tree := 75

def total_mass_gala := num_gala_trees * yield_gala_tree
def total_mass_fuji := num_fuji_trees * yield_fuji_tree
def total_mass_redhaven := num_redhaven_trees * yield_redhaven_tree
def total_mass_elberta := num_elberta_trees * yield_elberta_tree

def total_mass_fruit := total_mass_gala + total_mass_fuji + total_mass_redhaven + total_mass_elberta

theorem orchard_total_mass : total_mass_fruit = 6975 := by
  sorry

end orchard_total_mass_l501_501315


namespace expected_value_max_f_l501_501712

def coin_flip (n : ℕ) : (ℕ → ℤ) :=
  λ i => if (i % 2 = 0) then 1 else -1

def f (n : ℕ) : ℤ :=
  ∑ i in Finset.range (n + 1), coin_flip 2013 i

def max_f (n : ℕ) : ℤ :=
  Finset.max' (Finset.image f (Finset.range (n + 1))) (by simp)

theorem expected_value_max_f :
  let expected_value := (1 / 2) - (1007 * nat.choose 2013 1006 : ℚ) / (2 ^ 2012 : ℚ) in
  expected_value = (-1 / 2) + (1007 * nat.choose 2013 1006 : ℚ) / (2 ^ 2012 : ℚ) :=
begin
  sorry
end

end expected_value_max_f_l501_501712


namespace number_of_possible_multisets_roots_l501_501914

noncomputable def p (a : ℕ → ℤ) : ℤ[X] := 
  ∑ i in finset.range 9, polynomial.C (a i) * polynomial.X ^ i

noncomputable def q (a : ℕ → ℤ) : ℤ[X] := 
  ∑ i in finset.range 9, polynomial.C (a (8 - i)) * polynomial.X ^ i

theorem number_of_possible_multisets_roots 
(a : ℕ → ℤ) 
(hr : multiset (root_set (p a)).nodup) 
: 
    (multiset.card ((multiset.filter (λ x, x = 1 ∨ x = -1) hr) + 
    multiset.filter (λ x, x = complex.I ∨ x = -complex.I) hr) = 149) := sorry

end number_of_possible_multisets_roots_l501_501914


namespace angle_between_vectors_is_45_deg_l501_501267

noncomputable def vector_angle (a b : Vector ℝ) : ℝ :=
  Real.acos ((a.dot b) / (a.norm * b.norm))

theorem angle_between_vectors_is_45_deg 
  (a b : Vector ℝ) 
  (ha : ∥a∥ = Real.sqrt 2) 
  (hb : ∥b∥ = 2)
  (h_perp : (a - b).dot a = 0) : vector_angle a b = Real.pi / 4 :=
by
  sorry

end angle_between_vectors_is_45_deg_l501_501267


namespace projectile_reaches_45_feet_first_time_l501_501059

theorem projectile_reaches_45_feet_first_time :
  ∃ t : ℝ, (-20 * t^2 + 90 * t = 45) ∧ abs (t - 0.9) < 0.1 := sorry

end projectile_reaches_45_feet_first_time_l501_501059


namespace sequence_bound_l501_501602

noncomputable def sequence (a : ℕ → ℕ) : Prop := 
  (a 1 < a 2) ∧ (∀ k ≥ 3, a k = 4 * a (k-1) - 3 * a (k-2))

theorem sequence_bound (a : ℕ → ℕ) (h : sequence a) : a 45 > 3^43 :=
by
  sorry

end sequence_bound_l501_501602


namespace min_perimeter_is_correct_l501_501323

section perimeter_minimization

variable (A B P Q : ℝ × ℝ)
variable (M : ℝ × ℝ)
variable hA : A = (6, 5)
variable hB : B = (10, 2)
variable hM : ∀ (x : ℝ), M = (x, x)

/-- Minimum perimeter of the quadrilateral ABQP -/
noncomputable def minimum_perimeter : ℝ :=
  let A' := (5, 6)
  let B' := (10, -2)
  let AB_dist := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let A'B'_dist := Real.sqrt ((B'.1 - A'.1)^2 + (B'.2 - A'.2)^2)
  AB_dist + A'B'_dist

/-- The minimum value of the perimeter l of the quadrilateral ABQP is 5 + sqrt(89) -/
theorem min_perimeter_is_correct :
  minimum_perimeter A B = 5 + Real.sqrt 89 := by
  rw [hA, hB]
  unfold minimum_perimeter
  simp only [Real.sqrt_eq_rpow]
  norm_num
  sorry

end perimeter_minimization

end min_perimeter_is_correct_l501_501323


namespace triangle_internal_angles_external_angle_theorem_l501_501310

theorem triangle_internal_angles {A B C : ℝ}
 (mA : A = 64) (mB : B = 33) (mC_ext : C = 120) :
  180 - A - B = 83 :=
by
  sorry

theorem external_angle_theorem {A C D : ℝ}
 (mA : A = 64) (mC_ext : C = 120) :
  C = A + D → D = 56 :=
by
  sorry

end triangle_internal_angles_external_angle_theorem_l501_501310


namespace area_proof_l501_501955

-- Define the points and their distances
structure Point where
  x : ℝ
  y : ℝ

noncomputable def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Define the sides of the triangle
def A : Point := ⟨0, 0⟩
def B : Point := ⟨6, 0⟩
def C : Point := ⟨3, real.sqrt 45⟩ -- Solving the coordinates to satisfy given sides

-- Define the problem statement and known answer
def area_of_region : ℝ := 288 / 5

theorem area_proof :
  (∃ (P : Point),
    let dAB := distance P (⟨A.x, A.y⟩ + B) in
    let dBC := distance P (⟨B.x, B.y⟩ + C) in
    let dAC := distance P (⟨A.x, A.y⟩ + C) in
    ∀ (x y : ℝ), dAB + x = dBC + y ∧ dBC + y = dAC ∧
    (3 * x + 4 * y + 24) / 12 = area_of_region / 2) →
  true := sorry

end area_proof_l501_501955


namespace find_k_value_l501_501248

theorem find_k_value (S : ℕ → ℕ) (a : ℕ → ℕ) (k : ℤ) 
  (hS : ∀ n, S n = 5 * n^2 + k * n)
  (ha2 : a 2 = 18) :
  k = 3 := 
sorry

end find_k_value_l501_501248


namespace sum_of_cosines_correct_l501_501930

def eval_sum_of_cosines : ℂ :=
  let terms : List ℂ := List.range 41 |>.map (λ n, (complex.i ^ n) * complex.cos (30 + 90 * n))
  List.sum terms

theorem sum_of_cosines_correct :
  eval_sum_of_cosines = (21 * (complex.sqrt 3 / 2) - 20 * (complex.i) * (complex.sqrt 3 / 2)) :=
  sorry

end sum_of_cosines_correct_l501_501930


namespace income_is_10000_l501_501738

theorem income_is_10000 (x : ℝ) (h : 10 * x = 8 * x + 2000) : 10 * x = 10000 := by
  have h1 : 2 * x = 2000 := by
    linarith
  have h2 : x = 1000 := by
    linarith
  linarith

end income_is_10000_l501_501738


namespace shelves_needed_l501_501876

theorem shelves_needed (initial_books sold_books books_per_shelf remaining_books shelves_used : ℕ) :
  initial_books = 40 →
  sold_books = 20 →
  books_per_shelf = 4 →
  remaining_books = initial_books - sold_books →
  shelves_used = remaining_books / books_per_shelf →
  shelves_used = 5 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h3] at h5
  simp at h4 h5
  rw h4 h5
  sorry

end shelves_needed_l501_501876


namespace range_f_l501_501545

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctanh x

theorem range_f : Set.range f = Set.univ :=
by
  -- Proof is skipped
  sorry

end range_f_l501_501545


namespace general_term_seq_l501_501603

theorem general_term_seq 
  (a : ℕ → ℚ) 
  (h1 : a 1 = 1) 
  (h2 : a 2 = 5/3) 
  (h_rec : ∀ n, n > 0 → a (n + 2) = (5 / 3) * a (n + 1) - (2 / 3) * a n) : 
  ∀ n, a n = 2 - (3 / 2) * (2 / 3)^n :=
by
  sorry

end general_term_seq_l501_501603


namespace Oshea_needs_30_small_planters_l501_501035

theorem Oshea_needs_30_small_planters 
  (total_seeds : ℕ) 
  (large_planters : ℕ) 
  (capacity_large : ℕ) 
  (capacity_small : ℕ)
  (h1: total_seeds = 200) 
  (h2: large_planters = 4) 
  (h3: capacity_large = 20) 
  (h4: capacity_small = 4) : 
  (total_seeds - large_planters * capacity_large) / capacity_small = 30 :=
by 
  sorry

end Oshea_needs_30_small_planters_l501_501035


namespace smallest_altitude_le_3_l501_501744

theorem smallest_altitude_le_3 (a b c h_a h_b h_c : ℝ) (r : ℝ) (h_r : r = 1)
    (h_a_ge_b : a ≥ b) (h_b_ge_c : b ≥ c) 
    (area_eq1 : (a + b + c) / 2 * r = (a * h_a) / 2) 
    (area_eq2 : (a + b + c) / 2 * r = (b * h_b) / 2) 
    (area_eq3 : (a + b + c) / 2 * r = (c * h_c) / 2) : 
    min h_a (min h_b h_c) ≤ 3 := 
by
  sorry

end smallest_altitude_le_3_l501_501744


namespace deck_probability_problem_l501_501468

theorem deck_probability_problem (deck_size pairs_removed remaining_cards pairs_probability m n : ℕ) 
  (h_deck_size : deck_size = 40)
  (h_pairs_removed : pairs_removed = 4)
  (h_remaining_cards : remaining_cards = deck_size - pairs_removed)
  (h_remaining_value : remaining_cards = 36)
  (h_total_ways : remaining_cards.choose 2 = 630)
  (h_pairs_probability : pairs_probability = 50)
  (h_reduced_probability : Nat.gcd 25 315 = 1)
  (h_m_val : m = 25)
  (h_n_val : n = 315) : 
  m + n = 340 := by
  sorry

end deck_probability_problem_l501_501468


namespace computation_l501_501506

open Matrix

def initialMatrix : Matrix (Fin 2) (Fin 2) ℤ := !![1, 1; 2, 1]

def computedMatrix : Matrix (Fin 2) (Fin 2) ℤ := !![17, 12; 24, 17]

def matrixPow (m : Matrix (Fin 2) (Fin 2) ℤ) (n : ℕ) : Matrix (Fin 2) (Fin 2) ℤ :=
  if h : n = 0 then 1 else nat.recOn (nat.pred h) (pow m 1) fun _ _ in pow m (nat.succ _)

theorem computation :
  (matrixPow initialMatrix 10) = (matrixPow computedMatrix 5) := by
  sorry

end computation_l501_501506


namespace lcm_value_count_l501_501205

theorem lcm_value_count (a b : ℕ) (k : ℕ) (h1 : 9^9 = 3^18) (h2 : 12^12 = 2^24 * 3^12) 
  (h3 : 18^18 = 2^18 * 3^36) (h4 : k = 2^a * 3^b) (h5 : 18^18 = Nat.lcm (9^9) (Nat.lcm (12^12) k)) :
  ∃ n : ℕ, n = 25 :=
begin
  sorry
end

end lcm_value_count_l501_501205


namespace maria_initial_cookies_l501_501020

theorem maria_initial_cookies (X : ℕ) 
  (h1: X - 5 = 2 * (5 + 2)) 
  (h2: X ≥ 5)
  : X = 19 := 
by
  sorry

end maria_initial_cookies_l501_501020


namespace remainder_of_x4_div_x2_add_3x_sub_4_l501_501196

noncomputable def remainder_when_divided (p q : Polynomial ℚ) : Polynomial ℚ :=
  (Polynomial.divModByMonic p q).snd

theorem remainder_of_x4_div_x2_add_3x_sub_4 :
  remainder_when_divided (Polynomial.C 1 * Polynomial.X ^ 4) (Polynomial.C 1 * Polynomial.X ^ 2 + Polynomial.C 3 * Polynomial.X - Polynomial.C 4) 
  = - Polynomial.C 51 * Polynomial.X + Polynomial.C 52 :=
by
  sorry

end remainder_of_x4_div_x2_add_3x_sub_4_l501_501196


namespace smallest_positive_period_max_min_values_l501_501990

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.sin x)
noncomputable def f (x : ℝ) : ℝ := (vector_a x).1 * (vector_b x).1 + (vector_a x).2 * (vector_b x).2 - 1 / 2

-- Theorem 1: Smallest positive period of the function f(x)
theorem smallest_positive_period : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi :=
  sorry

-- Theorem 2: Maximum and minimum values of the function f(x) on [0, π/2]
theorem max_min_values : 
  ∀ x ∈ Set.Icc 0 (Real.pi / 2),
    f x ≤ 1 ∧ f x ≥ -1 / 2 ∧ (∃ (x_max : ℝ), x_max ∈ Set.Icc 0 (Real.pi / 2) ∧ f x_max = 1) ∧
    (∃ (x_min : ℝ), x_min ∈ Set.Icc 0 (Real.pi / 2) ∧ f x_min = -1 / 2) :=
  sorry

end smallest_positive_period_max_min_values_l501_501990


namespace exists_x_for_every_n_l501_501374

theorem exists_x_for_every_n (n : ℕ) (hn : 0 < n) : ∃ x : ℤ, 2^n ∣ (x^2 - 17) :=
sorry

end exists_x_for_every_n_l501_501374


namespace fraction_division_l501_501897

-- Define the fractions and the operation result.
def complex_fraction := 5 / (8 / 15)
def result := 75 / 8

-- State the theorem indicating that these should be equal.
theorem fraction_division :
  complex_fraction = result :=
  by
  sorry

end fraction_division_l501_501897


namespace polynomial_degree_l501_501140

-- Define the problem setup.
def is_root (p : Polynomial ℚ) (x : ℝ) := p.eval x = 0

noncomputable def polynomial_with_given_roots : Polynomial ℚ :=
  ∏ n in Finset.range 1100 + 1, (X - (n : ℂ) - Complex.sqrt (2 * n + 1)) *
                               (X - (n : ℂ) + Complex.sqrt (2 * n + 1))

-- State the main theorem.
theorem polynomial_degree :
  ∃ (p : Polynomial ℚ), (∀ n ∈ Finset.range 1100 + 1, is_root p (n + Real.sqrt (2 * n + 1))) ∧ p.degree = 2200 :=
sorry

end polynomial_degree_l501_501140


namespace length_pz_l501_501002

noncomputable def P : Type := sorry -- define a point P
noncomputable def Q : Type := sorry -- define a point Q
noncomputable def X : Type := sorry -- define a point X
noncomputable def Z : Type := sorry -- define a point Z
noncomputable def R : Type := sorry -- define a point R
noncomputable def S : Type := sorry -- define a point S

axiom pq_perp_xz : ∀ (P Q X Z : Type), PQ ⊥ XZ
axiom sr_perp_pz : ∀ (S R P Z : Type), SR ⊥ PZ
axiom pq_len : ∀ (P Q : Type), length PQ = 6
axiom xz_len : ∀ (X Z : Type), length XZ = 7
axiom sr_len : ∀ (S R : Type), length SR = 5

theorem length_pz (P Q X Z R S : Type) (PQ : length PQ = 6) (XZ : length XZ = 7) (SR : length SR = 5) :
  length PZ = 8.4 := by
  sorry

end length_pz_l501_501002


namespace fraction_equality_of_fraction_l501_501801

noncomputable theory

def fraction_of_fraction (a b : ℚ) : ℚ := a * b

theorem fraction_equality_of_fraction (a b x target : ℚ) (h1 : fraction_of_fraction a b = 7 / 24) (h2 : target = 1 / 8) (h3 : x * fraction_of_fraction a b = target) : x = 3 / 7 :=
by sorry

end fraction_equality_of_fraction_l501_501801


namespace tangent_OP_circumscribed_circle_APM_l501_501369

-- Definitions of points and geometric concepts.
variables {O A B C D E F M P : Type} -- Abstract points for the triangle and circumcircle.

-- Given conditions:
axiom circumscribed_circle (O : Point) (A B C : Point) : Circle
noncomputable def center_of_circumscribed_circle : Point := O

axiom perpendicular_bisector_intersects (A B C D E: Point)
  : (perpendicular_bisector A B intersects CA at D) ∧ (perpendicular_bisector A B intersects CB at E)

axiom line_intersection_BE_OD (B E O D F : Point) : (line BE intersects line OD at F)

axiom midpoint_M (A B M : Point) : (M is the midpoint of AB)

axiom circumscribed_circle_intersection_AMF_CEF (A M F C E P : Point)
  : (circumscribed_circle A M F intersects circumscribed_circle C E F at P) ∧ (P ≠ F)

-- Conclusion to prove:
theorem tangent_OP_circumscribed_circle_APM : tangency (line OP) (circumscribed_circle A P M) :=
begin
  sorry
end

end tangent_OP_circumscribed_circle_APM_l501_501369


namespace find_f_value_l501_501625

noncomputable def f : ℝ → ℝ := 
  λ x, if x ∈ set.Ico (-π / 2) 0 then Real.cos x else 
    if x < -π / 2 then 
      have hx : ∃ n : ℤ, x + π * n ∈ set.Ico (-π / 2) 0, from sorry, -- using periodicity
      -(Real.cos (x + π * (hx.some)))
    else
      have hx : ∃ n : ℤ, x - π * n ∈ set.Ico (-π / 2) 0, from sorry, -- using periodicity
      -(Real.cos (x - π * (hx.some)))

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + π) = f x

theorem find_f_value : f (-5 * π / 3) = -1 / 2 :=
by {
  sorry
}

end find_f_value_l501_501625


namespace number_of_partitions_l501_501236

theorem number_of_partitions (N k : ℕ) (hN : N = 65) (hk : k = 61) :
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ k ∧ 1 ≤ b ∧ b ≤ k ∧ 1 ≤ c ∧ c ≤ k ∧ a + b + c = N) →
  2007 = ((N - 1).choose 2 - (count_invalid_combinations N k)).natAbs := sorry


end number_of_partitions_l501_501236


namespace f1_is_class_k_f2_is_not_class_k_l501_501288

noncomputable def is_class_k_polynomial (f : ℚ[X][Y]) : Prop :=
  ∃ g u : ℚ[X], ∃ h v : ℚ[Y], g ≠ 0 ∧ u ≠ 0 ∧ h ≠ 0 ∧ v ≠ 0 ∧ f = (g * h) + (u * v)

def f1 : ℚ[X][Y] := 1 + (polynomial.C 1) * (polynomial.C 1)
def f2 : ℚ[X][Y] := 1 + (polynomial.C 1) * (polynomial.C 1) + (polynomial.C 1) * (polynomial.C 1)

theorem f1_is_class_k : is_class_k_polynomial f1 := sorry
theorem f2_is_not_class_k : ¬ is_class_k_polynomial f2 := sorry

end f1_is_class_k_f2_is_not_class_k_l501_501288


namespace selection_possible_l501_501857

variable (n : ℕ)
variable (knows : Fin n → Fin n → Prop) -- knows a b means a (from A) knows b (from B)

definition satisfies_conditions (selected : Fin n → Prop) : Prop :=
  (∀ b : Fin n, (Finset.card {a : Fin n | selected a ∧ knows a b}) % 2 = 0) ∨
  (∀ b : Fin n, (Finset.card {a : Fin n | selected a ∧ knows a b}) % 2 = 1)

theorem selection_possible (h_nonempty : ∃ a : Fin n, ∀ b : Fin n, knows a b) :
  ∃ selected : Fin n → Prop, satisfies_conditions selected :=
sorry

end selection_possible_l501_501857


namespace division_by_fraction_l501_501896

theorem division_by_fraction :
  (5 / (8 / 15) : ℚ) = 75 / 8 :=
by
  sorry

end division_by_fraction_l501_501896


namespace tan_alpha_plus_pi_div_4_l501_501607

noncomputable def tan_plus_pi_div_4 (α : ℝ) : ℝ := Real.tan (α + Real.pi / 4)

theorem tan_alpha_plus_pi_div_4 (α : ℝ) 
  (h1 : α > Real.pi / 2) 
  (h2 : α < Real.pi) 
  (h3 : (Real.cos α, Real.sin α) • (Real.cos α ^ 2, Real.sin α - 1) = 1 / 5)
  : tan_plus_pi_div_4 α = -1 / 7 := sorry

end tan_alpha_plus_pi_div_4_l501_501607


namespace common_root_proof_l501_501788

noncomputable def common_root (C D : ℚ) (p : ℚ) : ℚ :=
  if h : (∃ q r s t : ℚ, p + q + r = -C ∧ pqr = -15 ∧ p + s + t = 0 ∧ pst = -35) then
    (p^3 = 525 ∧ p = 525 ^ (1 / 3)) ∧ -- ensure arithmetic consistency
    p -- return the common root
  else 0

theorem common_root_proof (C D : ℚ) (p : ℚ) :
  ∃ q r s t : ℚ, 
  (p + q + r = -C) ∧
  (p * q * r = -15) ∧
  (p + s + t = 0) ∧
  (p * s * t = -35) →
  (p^3 = 525) ∧ (p = 525 ^ (1/3)) :=
by
  intros h
  sorry

end common_root_proof_l501_501788


namespace player_A_always_wins_l501_501786

theorem player_A_always_wins :
  ∃ (a b c : ℤ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (∃ p q : ℚ, p ≠ q ∧ a * p^2 + b * p + c = 0 ∧ a * q^2 + b * q + c = 0) :=
begin
  sorry
end

end player_A_always_wins_l501_501786


namespace pencils_at_home_l501_501037

theorem pencils_at_home (P : ℕ) (H1 : 2 - P = -13) : P = 15 := 
by {
  sorry
}

end pencils_at_home_l501_501037


namespace min_value_proof_l501_501626

open Real

theorem min_value_proof (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  (1 / m + 2 / n) ≥ 3 + 2 * sqrt 2 :=
by
  sorry

end min_value_proof_l501_501626


namespace inequality_solution_set_l501_501977

noncomputable def f : ℝ → ℝ := sorry

theorem inequality_solution_set
  (f_defined : ∀ x : ℝ, f x ∈ ℝ)
  (h0 : f 0 = 1)
  (h_derivative : ∀ x : ℝ, deriv f x < f x + 1) :
  {x : ℝ | f x + 1 < 2 * real.exp x} = {x : ℝ | x > 0} :=
by
  sorry

end inequality_solution_set_l501_501977


namespace part_a_six_by_six_grid_sum_zero_l501_501017

theorem part_a_six_by_six_grid_sum_zero (m n : ℕ) (grid : Fin m → Fin n → ℝ)
  (h_m : m = 6) (h_n : n = 6)
  (h_range : ∀ i j, -1 ≤ grid i j ∧ grid i j ≤ 1)
  (h_sum_subgrid : ∀ i j, i < m - 1 → j < n - 1 → 
    grid i j + grid i (j + 1) + grid (i + 1) j + grid (i + 1) (j + 1) = 0) :
  (∑ i, ∑ j, grid i j) = 0 :=
sorry

end part_a_six_by_six_grid_sum_zero_l501_501017


namespace A_inter_B_eq_l501_501000

open Set

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℤ := {0, 1, 2, 3, 4}

theorem A_inter_B_eq :
  A ∩ (B : Set ℝ) = ({0, 1} : Set ℝ) :=
by 
  sorry

end A_inter_B_eq_l501_501000


namespace black_percentage_correct_l501_501692

def radius_seq (n : ℕ) : ℕ := 2 * (n + 1)

def circle_area (r : ℕ) : ℝ := π * (r : ℝ) ^ 2

def black_area (n : ℕ) : ℝ :=
  finset.sum (finset.filter (λ i, i % 2 = 0) (finset.range n)) (λ i, circle_area (radius_seq i)) -
  finset.sum (finset.filter (λ i, i % 2 = 1) (finset.range ((n - 1) / 2))) (λ i, circle_area (radius_seq (i + 1)))

def total_area (r : ℕ) : ℝ :=
  circle_area r

def black_percentage (n r : ℕ) : ℝ :=
  100 * (black_area n) / (total_area r)

theorem black_percentage_correct :
  black_percentage 6 (radius_seq 5) = 42 :=
sorry

end black_percentage_correct_l501_501692


namespace business_ownership_l501_501474

variable (x : ℝ) (total_value : ℝ)
variable (fraction_sold : ℝ)
variable (sale_amount : ℝ)

-- Conditions
axiom total_value_condition : total_value = 10000
axiom fraction_sold_condition : fraction_sold = 3 / 5
axiom sale_amount_condition : sale_amount = 2000
axiom equation_condition : (fraction_sold * x * total_value = sale_amount)

theorem business_ownership : x = 1 / 3 := by 
  have hv := total_value_condition
  have hf := fraction_sold_condition
  have hs := sale_amount_condition
  have he := equation_condition
  sorry

end business_ownership_l501_501474


namespace right_triangle_area_l501_501806

theorem right_triangle_area (a b c : ℝ) (h₁ : a = 24) (h₂ : c = 26) (h₃ : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 120 :=
by
  sorry

end right_triangle_area_l501_501806


namespace cost_per_mile_l501_501892

theorem cost_per_mile 
    (round_trip_distance : ℝ)
    (num_days : ℕ)
    (total_cost : ℝ) 
    (h1 : round_trip_distance = 200 * 2)
    (h2 : num_days = 7)
    (h3 : total_cost = 7000) 
  : (total_cost / (round_trip_distance * num_days) = 2.5) :=
by
  sorry

end cost_per_mile_l501_501892


namespace minimum_possible_value_of_S_l501_501345

open Finset

def valid_set (S : Finset ℕ) : Prop :=
  (∀ x ∈ S, x ∈ (range 16 \ {0})) ∧      -- S is a subset of {1, 2, ..., 15}
  (S.card = 7) ∧                          -- S has exactly 7 elements
  (∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → ¬ (a ∣ b ∨ b ∣ a)). -- No a is multiple/factor of b

theorem minimum_possible_value_of_S :
  ∃ S : Finset ℕ, valid_set S ∧ S.min' ⟨_, by linarith⟩ = 3 :=
sorry

end minimum_possible_value_of_S_l501_501345


namespace no_relation_between_x_and_y_l501_501617

theorem no_relation_between_x_and_y (t : ℝ) (h1 : 0 < t) (h2 : t ≠ 1) :
    let x := t^(2/(t-1))
    let y := t^(t/(t-1))
    ¬ ((xy)^x = (yx)^y ∨ x^(2y) = y^(2x) ∨ (xy)^y = (yx)^x ∨ y^x = x^y) :=
by {
    let x := t^(2/(t-1))
    let y := t^(t/(t-1))
    sorry
}

end no_relation_between_x_and_y_l501_501617


namespace radius_of_smaller_molds_l501_501858

noncomputable def hemisphereVolume (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r ^ 3

theorem radius_of_smaller_molds :
  ∀ (R r : ℝ), R = 2 ∧ (64 * hemisphereVolume r) = hemisphereVolume R → r = 1 / 2 :=
by
  intros R r h
  sorry

end radius_of_smaller_molds_l501_501858


namespace Hammie_weights_l501_501994

theorem Hammie_weights :
  let weights := [67, 8, 8, 9, 10, 60, 60] in
  let median := (weights[3] + weights[4]) / 2 in
  let average := (weights.sum : ℝ) / weights.length in
  median = 35 ∧ average = 27.75 :=
by
  let weights := [67, 8, 8, 9, 10, 60, 60]
  let sorted_weights := weights.sorted
  have median_def : median = (sorted_weights[3] + sorted_weights[4])/2 := rfl
  have average_def : average = (weights.sum : ℝ) / weights.length := rfl
  have median_value: median = 35 := by linarith  -- proof that the median is 35
  have average_value: average = 27.75 := by linarith -- proof that the average is 27.75
  exact ⟨median_value, average_value⟩

end Hammie_weights_l501_501994


namespace symmetric_scanning_codes_l501_501872

open Classical

-- Define the conditions and the proof outline
theorem symmetric_scanning_codes {α : Type} [DecidableEq α] (black white : α) :
  (∃ g : Matrix (Fin 5) (Fin 5) α, 
      (∀ i j, g i j = g j i ∧ g i j = g (4 - i) j ∧ g i j = g i (4 - j)) ∧
      (∃ i j, g i j = black) ∧ 
      (∃ i j, g i j = white)) →
  (∃ n : ℕ, n = 62) :=
begin
  sorry
end

end symmetric_scanning_codes_l501_501872


namespace number_of_k_values_l501_501207

theorem number_of_k_values :
  let k (a b : ℕ) := 2^a * 3^b in
  (∀ a b : ℕ, 18 ≤ a ∧ b = 36 → 
  let lcm_val := Nat.lcm (Nat.lcm (9^9) (12^12)) (k a b) in 
  lcm_val = 18^18) →
  (Finset.card (Finset.filter (λ a, 18 ≤ a ∧ a ≤ 24) (Finset.range (24 + 1))) = 7) :=
by
  -- proof skipped
  sorry

end number_of_k_values_l501_501207


namespace g_18_sum_l501_501012

def g (n : ℕ) : ℕ := sorry

axiom g_pos_ints : ∀ n : ℕ, 0 < n → 0 < g(n)
axiom g_increasing : ∀ n : ℕ, 0 < n → g(n + 1) > g(n)
axiom g_mult_prop : ∀ m n : ℕ, 0 < m → 0 < n → g(m * n) = g(m) * g(n)
axiom g_special_case : ∀ m n : ℕ, 0 < m → 0 < n → m ≠ n → m^n = n^m → g(m) = n ∨ g(n) = m

theorem g_18_sum : g(18) = 5832 :=
sorry

end g_18_sum_l501_501012


namespace proof_problem_l501_501408

noncomputable def h : Polynomial ℝ := Polynomial.X^3 - Polynomial.X^2 - 4 * Polynomial.X + 4
noncomputable def p : Polynomial ℝ := Polynomial.X^3 + 12 * Polynomial.X^2 - 13 * Polynomial.X - 64

theorem proof_problem : 
  (∀ x : ℝ, h.eval x = 0 → p.eval (x^3) = 0) ∧ 
  (∀ a b c : ℝ, (p = Polynomial.X^3 + a * Polynomial.X^2 + b * Polynomial.X + c) → 
  ((a, b, c) = (12, -13, -64))) :=
sorry

end proof_problem_l501_501408


namespace spencer_walk_distance_l501_501659

theorem spencer_walk_distance :
  let distance_house_library := 0.3
  let distance_library_post_office := 0.1
  let total_distance := 0.8
  (total_distance - (distance_house_library + distance_library_post_office)) = 0.4 :=
by
  sorry

end spencer_walk_distance_l501_501659


namespace expected_value_of_biased_die_l501_501121

noncomputable def E (prob1to3 prob4 prob5 prob6 gain1to3 loss4 loss5 loss6 : ℝ) : ℝ :=
  prob1to3 * gain1to3 + 
  prob1to3 * gain1to3 + 
  prob1to3 * gain1to3 + 
  prob4 * loss4 + 
  prob5 * loss5 + 
  prob6 * loss6

theorem expected_value_of_biased_die :
  let prob1to3 := 1 / 3
      prob4 := 1 / 6
      prob5 := 1 / 6
      prob6 := 1 / 6
      gain1to3 := 4
      loss4 := -2
      loss5 := -5
      loss6 := -7 in
  E prob1to3 prob4 prob5 prob6 gain1to3 loss4 loss5 loss6 = 1.67 :=
by {
  -- We do not need a proof, hence we add sorry
  sorry
}

end expected_value_of_biased_die_l501_501121


namespace decimal_place_250_of_13_over_17_is_8_l501_501821

theorem decimal_place_250_of_13_over_17_is_8 :
  let repeating_sequence := '7647058823529411'
  let n := 250
  let repetition_length := 16
  let position_in_repetition := (n % repetition_length)
  position_in_repetition = 10 → 
  repeating_sequence[10] = 8 := 
by
  sorry

end decimal_place_250_of_13_over_17_is_8_l501_501821


namespace smallest_degree_q_l501_501522

-- Define the numerator polynomial
def p(x : ℝ) := 3 * x ^ 7 - 5 * x ^ 6 + 2 * x ^ 3 + 4

-- Define the condition that the rational function has a horizontal asymptote
def has_horizontal_asymptote (q : ℝ → ℝ) : Prop :=
  ∃ (d : ℕ), d ≥ 0 ∧ (∃ c : ℝ, c ≠ 0) ∧
  (∀ x : ℝ, x > 1 → (p x / q x - c / x ^ (max (degree p) (degree q))) → 0)

-- Prove that the smallest possible degree of q(x) is 7
theorem smallest_degree_q : ∃ (q : ℝ → ℝ), has_horizontal_asymptote q ∧ degree q = 7 := by
  sorry

end smallest_degree_q_l501_501522


namespace area_of_region_B_l501_501916

-- Define the region B based on the conditions
def region_B (z : ℂ) : Prop :=
  let x := z.re in
  let y := z.im in
  0 ≤ x ∧ x ≤ 80 ∧ 0 ≤ y ∧ y ≤ 80 ∧
  (40 - x) ^ 2 + y^2 ≥ 1600 ∧ 
  x^2 + (y - 40) ^ 2 ≥ 1600

-- Define the area function of a region
noncomputable def area (S : set ℂ) : ℝ := sorry

-- Define what is to be proved
theorem area_of_region_B : area {z : ℂ | region_B z} = 4000 - 800 * Real.pi :=
by sorry

end area_of_region_B_l501_501916


namespace derivative_of_y_is_correct_l501_501356

def y (x : ℝ) : ℝ := real.exp x * real.cos x

theorem derivative_of_y_is_correct :
  ∀ x : ℝ, deriv y x = real.exp x * real.cos x - real.exp x * real.sin x :=
by
  intro x
  sorry

end derivative_of_y_is_correct_l501_501356


namespace mica_kilograms_of_pasta_l501_501695

def cost_ground_beef := (1 / 4) * 8
def cost_pasta_sauce := 2 * 2
def cost_quesadilla := 6
def total_other_costs := cost_ground_beef + cost_pasta_sauce + cost_quesadilla

def total_amount_with_mica := 15
def remaining_amount := total_amount_with_mica - total_other_costs
def pasta_price_per_kg := 1.5
def kilograms_pasta := remaining_amount / pasta_price_per_kg

theorem mica_kilograms_of_pasta : kilograms_pasta = 2 :=
by
  unfold kilograms_pasta remaining_amount
  unfold total_other_costs cost_ground_beef cost_pasta_sauce cost_quesadilla
  unfold pasta_price_per_kg
  norm_num
  rfl

end mica_kilograms_of_pasta_l501_501695


namespace net_worth_changes_l501_501022

namespace FinancialTransaction

def initial_cash_A := 15000
def initial_house_value := 12000
def initial_cash_B := 13000
def first_transaction_price := 14000
def second_transaction_price := 10000

def final_cash_A := initial_cash_A + first_transaction_price - second_transaction_price
def final_cash_B := initial_cash_B - first_transaction_price + second_transaction_price

def net_worth_A := final_cash_A + initial_house_value - initial_cash_A - initial_house_value
def net_worth_B := final_cash_B - initial_cash_B

theorem net_worth_changes :
  final_cash_A = 19000 ∧
  final_cash_B = 9000 ∧
  net_worth_A = 4000 ∧
  net_worth_B = -4000 :=
by
  sorry

end FinancialTransaction

end net_worth_changes_l501_501022


namespace area_ratio_of_squares_l501_501163

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * (4 * b) = 4 * a) : (a * a) / (b * b) = 16 :=
by
  sorry

end area_ratio_of_squares_l501_501163


namespace loan_period_l501_501472

theorem loan_period (principal : ℝ) (rate_A rate_C : ℝ) (gain : ℝ) (years : ℝ) :
  principal = 3500 ∧ rate_A = 0.1 ∧ rate_C = 0.12 ∧ gain = 210 →
  (rate_C * principal * years - rate_A * principal * years) = gain →
  years = 3 :=
by
  sorry

end loan_period_l501_501472


namespace valid_three_digit_numbers_count_l501_501911

def is_three_digit_number (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9

def valid_number (a b c : ℕ) : Prop :=
  is_three_digit_number a b c ∧ ((99 * a + 9 * b) % 10 = 3)

def count_valid_numbers : ℕ :=
  ∑ a in { a | 1 ≤ a ∧ a ≤ 9}, ∑ b in { b | 0 ≤ b ∧ b ≤ 9 }, ∑ c in { c | 0 ≤ c ∧ c ≤ 9 }, if valid_number a b c then 1 else 0

theorem valid_three_digit_numbers_count : count_valid_numbers = 70 :=
  sorry

end valid_three_digit_numbers_count_l501_501911


namespace remainder_b21_div_12_l501_501009

def b_n (n : ℕ) : ℕ := 
  "concatenate all integers from 1 to n into a single number" -- This is a pseudocode for the definition

def b_21 : ℕ := b_n 21

theorem remainder_b21_div_12 : b_21 % 12 = 9 :=
  sorry

end remainder_b21_div_12_l501_501009


namespace slope_line_OM_l501_501985

theorem slope_line_OM : 
  let t := (Real.pi / 3)
  let x := 2 * Real.cos t
  let y := 4 * Real.sin t
  let M := (x, y)
  let O := (0 : ℝ, 0 : ℝ)
  (M = (1, 2 * Real.sqrt 3)) →
  (O = (0, 0)) →
  (if (O.1 ≠ M.1) then (((M.2 - O.2) / (M.1 - O.1)) = 2 * Real.sqrt 3) else False) :=
by
  intro t x y M O hM hO
  exact sorry

end slope_line_OM_l501_501985


namespace trapezoid_area_division_in_half_l501_501566

open EuclideanGeometry

-- Definitions and conditions from part (a)
variables {A B C D K L M N O P Q : Point}

-- A function to represent central symmetry with respect to a point O
def is_symm (O P Q : Point) : Prop := dist O P = dist O Q

-- Conditions
variable (H1 : IsTrapezoid A B C D)
variable (H2 : IsMidpoint O (midline A D C B))
variable (H3 : is_symm O A K)
variable (H4 : is_symm O B L)
variable (H5 : is_symm O C M)
variable (H6 : is_symm O D N)
variable (H7 : intersection KM (CD) = {P})
variable (H8 : intersection NL (AB) = {Q})

-- Statement to be proved
theorem trapezoid_area_division_in_half :
  divides_area_in_half (CM) (trapezoid A B C D) ∧
  divides_area_in_half (BL) (trapezoid A B C D) ∧
  divides_area_in_half (AP) (trapezoid A B C D) ∧
  divides_area_in_half (DQ) (trapezoid A B C D) := sorry

end trapezoid_area_division_in_half_l501_501566


namespace chessboard_all_same_color_l501_501563

/-- A chessboard is a 16x16 grid with alternating black and white colors.
An operation A changes the color of all squares in the i-th row and all squares in the j-th column.
Prove that we can change all the squares on the chessboard to the same color through a certain number of operations. -/
theorem chessboard_all_same_color :
  ∃ (ops : list (ℕ × ℕ)), 
    (∀ (i j : ℕ) (hi : 1 ≤ i ∧ i ≤ 16) (hj : 1 ≤ j ∧ j ≤ 16), 
      (ops.any (λ (p : ℕ × ℕ), 
        (p.1 = i ∨ p.2 = j)) → 
      true)) ∧ 
    (∀ (c : bool), 
      (ops.foldl (λ (b : bool) (p : ℕ × ℕ), !b) c = c)) :=
sorry

end chessboard_all_same_color_l501_501563


namespace sum_series_eq_99_div_50_l501_501508

theorem sum_series_eq_99_div_50 :
  ∑' n : ℕ, if n ≥ 2 then (n^4 + 4 * n^2 + 15 * n + 15) / (2^n * (n^4 + 9)) else 0 = 99 / 50 :=
by
  sorry

end sum_series_eq_99_div_50_l501_501508


namespace remaining_requests_after_7_days_l501_501019

-- Definitions based on the conditions
def dailyRequests : ℕ := 8
def dailyWork : ℕ := 4
def days : ℕ := 7

-- Theorem statement representing our final proof problem
theorem remaining_requests_after_7_days : 
  (dailyRequests * days - dailyWork * days) + dailyRequests * days = 84 := by
  sorry

end remaining_requests_after_7_days_l501_501019


namespace diameter_of_circle_l501_501624

def isTangentToYAxis (C : Circle) : Prop := 
  C.center.x = C.radius

def isTangentToLine (C : Circle) (l : Line) : Prop := 
  abs (l.slope * C.center.x - C.center.y + l.y_intercept) = C.radius * sqrt (1 + l.slope^2)

def passesThrough (C : Circle) (P : Point) : Prop := 
  (P.x - C.center.x)^2 + (P.y - C.center.y)^2 = C.radius^2

noncomputable def Circle := { center : Point, radius : ℝ }

theorem diameter_of_circle :
  ∃ (C : Circle), 
    isTangentToYAxis C ∧ 
    isTangentToLine C ⟨sqrt 3 / 3, 0⟩ ∧ 
    passesThrough C ⟨2, sqrt 3⟩ ∧ 
    (2 * C.radius = 2 ∨ 2 * C.radius = 14 / 3) := 
sorry

end diameter_of_circle_l501_501624


namespace man_speed_down_l501_501475

variable (d : ℝ) (v : ℝ)

theorem man_speed_down (h1 : 32 > 0) (h2 : 38.4 > 0) (h3 : d > 0) (h4 : v > 0) 
  (avg_speed : 38.4 = (2 * d) / ((d / 32) + (d / v))) : v = 48 :=
sorry

end man_speed_down_l501_501475


namespace proof_part1_proof_part2_l501_501499

-- Proof problem for the first part (1)
theorem proof_part1 (m : ℝ) : m^3 * m^6 + (-m^3)^3 = 0 := 
by
  sorry

-- Proof problem for the second part (2)
theorem proof_part2 (a : ℝ) : a * (a - 2) - 2 * a * (1 - 3 * a) = 7 * a^2 - 4 * a := 
by
  sorry

end proof_part1_proof_part2_l501_501499


namespace sales_on_second_street_l501_501710

noncomputable def commission_per_system : ℕ := 25
noncomputable def total_commission : ℕ := 175
noncomputable def total_systems_sold : ℕ := total_commission / commission_per_system

def first_street_sales (S : ℕ) : ℕ := S
def second_street_sales (S : ℕ) : ℕ := 2 * S
def third_street_sales : ℕ := 0
def fourth_street_sales : ℕ := 1

def total_sales (S : ℕ) : ℕ := first_street_sales S + second_street_sales S + third_street_sales + fourth_street_sales

theorem sales_on_second_street (S : ℕ) : total_sales S = total_systems_sold → second_street_sales S = 4 := by
  sorry

end sales_on_second_street_l501_501710


namespace range_of_x_l501_501964

noncomputable def f : ℝ → ℝ := sorry -- Define the function f

variable (f_increasing : ∀ x y, x < y → f x < f y) -- f is increasing
variable (f_at_2 : f 2 = 0) -- f(2) = 0

theorem range_of_x (x : ℝ) : f (x - 2) > 0 ↔ x > 4 :=
by
  sorry

end range_of_x_l501_501964


namespace train_crossing_time_l501_501152

/-- Define the length of the train in meters -/
def length_train : ℝ := 200

/-- Define the length of the platform in meters -/
def length_platform : ℝ := 300.04

/-- Define the speed of the train in km/h -/
def speed_train_kmph : ℝ := 72

/-- Conversion factor from km/h to m/s -/
def kmph_to_mps (v : ℝ) : ℝ := v * (1000 / 3600)

theorem train_crossing_time :
  let distance := length_train + length_platform
  let speed := kmph_to_mps speed_train_kmph
  let time := distance / speed
  time = 25.002 := 
by
  sorry

end train_crossing_time_l501_501152


namespace find_positive_integral_solution_eq_l501_501533

theorem find_positive_integral_solution_eq (n : ℕ) (h_pos : 0 < n)
  (h_numer : (∑ i in finset.range n, (2 * i + 1)) = n^2)
  (h_denom : (∑ i in finset.range n, (2 * (i + 1))) = n * (n + 1)) :
  (n / (n + 1) = 115 / 116) → n = 115 :=
by
  sorry

end find_positive_integral_solution_eq_l501_501533


namespace multiply_both_sides_l501_501824

theorem multiply_both_sides (f g : ℝ → ℝ) (x : ℝ) : 
  f x = g x → (x - 3) * f x = (x - 3) * g x :=
by
  assume h : f x = g x
  exact congr_arg ((*) (x - 3)) h

end multiply_both_sides_l501_501824


namespace sqrt_cosine_difference_l501_501965

theorem sqrt_cosine_difference (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) :
  sqrt ((1 + cos α) / (1 - cos α)) - sqrt ((1 - cos α) / (1 + cos α)) = -2 / tan α :=
by
  sorry

end sqrt_cosine_difference_l501_501965


namespace binomial_coefficient_7_5_l501_501507

open Nat

theorem binomial_coefficient_7_5 : Nat.binomial 7 5 = 21 := by
  sorry

end binomial_coefficient_7_5_l501_501507


namespace factorization_correct_l501_501531

theorem factorization_correct (x y : ℝ) : 
  x^2 + y^2 + 2*x*y - 1 = (x + y + 1) * (x + y - 1) := 
by
  sorry

end factorization_correct_l501_501531


namespace range_sum_of_h_l501_501913

noncomputable def h (x : ℝ) : ℝ := 5 / (5 + 3 * x^2)

theorem range_sum_of_h : 
  (∃ a b : ℝ, (∀ x : ℝ, 0 < h x ∧ h x ≤ 1) ∧ a = 0 ∧ b = 1 ∧ a + b = 1) :=
sorry

end range_sum_of_h_l501_501913


namespace correct_quotient_l501_501305

theorem correct_quotient (D d q N : ℕ) (hD : D = 21) (hd : d = 12) (hq : q = 70) (hN : N = d * q) :
  N = D * 40 :=
by
  rw [hD, hd, hq] at hN
  rw hN
  sorry

end correct_quotient_l501_501305


namespace necessary_without_sufficient_for_parallel_lines_l501_501572

noncomputable def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 2 = 0
noncomputable def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y - 1 = 0

theorem necessary_without_sufficient_for_parallel_lines :
  (∀ (a : ℝ), a = 2 → (∀ (x y : ℝ), line1 a x y → line2 a x y)) ∧ 
  ¬ (∀ (a : ℝ), (∀ (x y : ℝ), line1 a x y → line2 a x y) → a = 2) :=
sorry

end necessary_without_sufficient_for_parallel_lines_l501_501572


namespace triangle_area_l501_501645

open Real

theorem triangle_area :
  -- Given parametric form of line l
  (∀ t : ℝ, (x = 1 + t ∧ y = -3 + t)) →
  -- General equation of the line l
  (x - y - 4 = 0) →
  -- Polar equation of curve C
  (ρ = 6 * cos θ) →
  -- Rectangular coordinate equation of curve C
  ((x - 3) ^ 2 + y ^ 2 = 9) →
  -- Distance from (3,0) to the line l is (√2 / 2)
  (d = abs (3 - 0 - 4) / sqrt 2 = sqrt 2 / 2) →
  -- |AB| = √34
  ((|AB| / 2) ^ 2 + (d) ^ 2 = r ^ 2 → (| AB | = sqrt 34)) →
  -- Area of triangle ABC is √17 / 2
  (S = 1 / 2 * |AB| * d → S = sqrt 17 / 2).
Proof := by
   sorry

end triangle_area_l501_501645


namespace sum_f_1_to_2015_l501_501010

def f : ℝ → ℝ :=
  λ x, if -3 ≤ x ∧ x < -1 then -(x + 2)^2 else
       if -1 ≤ x ∧ x < 3 then x else f (x - 6)

theorem sum_f_1_to_2015 : (∑ i in finset.range 2015, f (1 + i)) = 336 :=
by
  sorry

end sum_f_1_to_2015_l501_501010


namespace f_36_l501_501953

variable {R : Type*} [CommRing R]
variable (f : R → R) (p q : R)

-- Conditions
axiom f_mult_add : ∀ x y, f (x * y) = f x + f y
axiom f_2 : f 2 = p
axiom f_3 : f 3 = q

-- Statement to prove
theorem f_36 : f 36 = 2 * (p + q) :=
by
  sorry

end f_36_l501_501953


namespace percentage_correct_l501_501443

variables (A B N : ℝ) 

theorem percentage_correct (hA : A = 0.75)
                          (hB : B = 0.55)
                          (hN : N = 0.20) :
                            A + B - (A ∩ B) + N = 1 → (A ∩ B) = 0.50 :=
by
  sorry

end percentage_correct_l501_501443


namespace parallelepiped_lateral_surface_area_l501_501108

noncomputable def lateral_surface_area_parallelepiped (d α β : ℝ) : ℝ :=
  2 * sqrt 2 * d^2 * sin α * tan β * sin (α + π / 4)

theorem parallelepiped_lateral_surface_area (d α β : ℝ) :
  lateral_surface_area_parallelepiped d α β = 2 * sqrt 2 * d^2 * sin α * tan β * sin (α + π / 4) :=
by
  sorry

end parallelepiped_lateral_surface_area_l501_501108


namespace range_of_m_l501_501290

theorem range_of_m (m : ℝ) (h1 : m + 3 > 0) (h2 : m - 1 < 0) : -3 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l501_501290


namespace sandy_worked_days_l501_501715

-- Definitions based on the conditions
def total_hours_worked : ℕ := 45
def hours_per_day : ℕ := 9

-- The theorem that we need to prove
theorem sandy_worked_days : total_hours_worked / hours_per_day = 5 :=
by sorry

end sandy_worked_days_l501_501715


namespace race_outcomes_l501_501722

open Fintype

-- Define the number of participants
def participants : Finset String := {"Abe", "Bobby", "Charles", "Devin", "Edwin", "Fred"}

noncomputable def count_outcomes (s : Finset String) : ℕ :=
  let charles_top3 : ℕ := 3
  let charles_not_in_top3 : Finset String := s.erase "Charles"
  let remaining_two_selected : ℕ := choose (charles_not_in_top3.card) 2
  let remaining_two_arranged : ℕ := 2.factorial
  charles_top3 * remaining_two_selected * remaining_two_arranged

theorem race_outcomes : count_outcomes participants = 60 := by
  sorry

end race_outcomes_l501_501722


namespace actual_height_of_boy_l501_501725

variable (wrong_height : ℕ) (boys : ℕ) (wrong_avg correct_avg : ℕ)
variable (x : ℕ)

-- Given conditions
def conditions 
:= boys = 35 ∧
   wrong_height = 166 ∧
   wrong_avg = 185 ∧
   correct_avg = 183

-- Question: Proving the actual height
theorem actual_height_of_boy (h : conditions boys wrong_height wrong_avg correct_avg) : 
  x = wrong_height + (boys * wrong_avg - boys * correct_avg) := 
  sorry

end actual_height_of_boy_l501_501725


namespace find_varphi_l501_501384

noncomputable def shifted_function (phi : ℝ) : (ℝ → ℝ) := 
  λ x, 3 * Real.sin (2 * (x - phi) + Real.pi / 3)

def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

theorem find_varphi (phi : ℝ) (h : 0 < phi ∧ phi < Real.pi / 2) 
  (heven : is_even (shifted_function phi)) : 
  phi = 5 * Real.pi / 12 :=
sorry

end find_varphi_l501_501384


namespace min_area_sum_l501_501947

noncomputable def f (x : ℝ) : ℝ := -x^3
noncomputable def g (x : ℝ) : ℝ := (2 / (|x^3| + x^3))

def tangent (f : ℝ → ℝ) (f' : ℝ → ℝ) (x0 : ℝ) : ℝ → ℝ := 
  λ x, f x0 + f' x0 * (x - x0)

def intersection_y (l : ℝ → ℝ) : ℝ := l 0

def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |(B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)|

theorem min_area_sum (x0 : ℝ) (h₁ : x0 ≠ 0):
  let f' := λ x, -3 * x^2
  let g' := λ x, -3 * x ^ (-4)
  let A := (x0, 0)
  let B := (0, tangent f f' x0 0)
  let C := (0, tangent g g' x0 0)
  let S1 := area_triangle (0, 0) A B
  let S2 := area_triangle (0, 0) A C
  in S1 + S2 = 8 := 
by
  sorry

end min_area_sum_l501_501947


namespace range_of_m_l501_501970

variable (P Q A : Point)
variable (m y : ℝ)
variable (onCurve : P.x = -sqrt(4 - y^2))
variable (onLine : Q.x = 6)
variable (midpoint : A.x = m ∧ A.y = 0 ∧ A = midpoint P Q)

theorem range_of_m : 
  ∃ P Q : Point, 
  P.x = -sqrt(4 - y^2) ∧ 
  Q.x = 6 ∧ 
  (A.x = m ∧ A.y = 0 ∧ A = midpoint P Q) → 
  2 ≤ m ∧ m ≤ 3 :=
sorry

end range_of_m_l501_501970


namespace school_pupils_l501_501638

theorem school_pupils (girls boys : ℕ) (hg : girls = 542) (hb : boys = 387) : girls + boys = 929 :=
by
  rw [hg, hb]
  norm_num

end school_pupils_l501_501638


namespace sum_of_solutions_l501_501925

-- Given the quadratic equation: x^2 + 3x - 20 = 7x + 8
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x - 20 = 7*x + 8

-- Prove that the sum of the solutions to this quadratic equation is 4
theorem sum_of_solutions : 
  ∀ x1 x2 : ℝ, (quadratic_equation x1) ∧ (quadratic_equation x2) → x1 + x2 = 4 :=
by
  sorry

end sum_of_solutions_l501_501925


namespace length_of_BC_l501_501263

open Real

-- Let points A, B, C, D and distances be as follows:
variables (A B C D : Point) (AD CD AC : ℝ)
-- Right triangles and distances conditions
axiom right_triangle_ABC: is_right_triangle A B C
axiom right_triangle_ABD: is_right_triangle A B D

-- Known distances
axiom AD_eq_37 : dist A D = 37
axiom CD_eq_19 : dist C D = 19
axiom AC_eq_16 : dist A C = 16

-- Define what needs to be proved: BC = 20
theorem length_of_BC : dist B C = 20 :=
  sorry

end length_of_BC_l501_501263


namespace quadratic_equation_unique_real_root_l501_501184

/-- Determine the positive value of k such that the quadratic x^2 - 6kx + 9k has exactly one real root. -/
theorem quadratic_equation_unique_real_root (k : ℝ) (hk : k > 0) : 
  (∃ x : ℝ, (x^2 - 6*k*x + 9*k = 0) ∧ (x^2 - 6*k*x + 9*k = 0).coeff 2 = 1) → k = 1 :=
by sorry

end quadratic_equation_unique_real_root_l501_501184


namespace max_value_sqrt_sum_l501_501580

theorem max_value_sqrt_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 3) :
  sqrt (3 * a + 1) + sqrt (3 * b + 1) + sqrt (3 * c + 1) ≤ 6 :=
sorry

end max_value_sqrt_sum_l501_501580


namespace first_player_winning_strategy_l501_501785

theorem first_player_winning_strategy (num_chips : ℕ) : 
  (num_chips = 110) → 
  ∃ (moves : ℕ → ℕ × ℕ), (∀ n, 1 ≤ (moves n).1 ∧ (moves n).1 ≤ 9) ∧ 
  (∀ n, (moves n).1 ≠ (moves (n-1)).1) →
  (∃ move_sequence : ℕ → ℕ, ∀ k, move_sequence k ≤ num_chips ∧ 
  ((move_sequence (k+1) < move_sequence k) ∨ (move_sequence (k+1) = 0 ∧ move_sequence k = 1)) ∧ 
  (move_sequence k > 0) ∧ (move_sequence 0 = num_chips) →
  num_chips ≡ 14 [MOD 32]) :=
by 
  sorry

end first_player_winning_strategy_l501_501785


namespace function_range_l501_501621

noncomputable def f (x : ℝ) : ℝ := x + 1 / x

theorem function_range (x : ℝ) (h : x > 0) : 
  set.range (f) = set.Ici 2 := 
sorry

end function_range_l501_501621


namespace compute_xy_l501_501078

theorem compute_xy (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 :=
sorry

end compute_xy_l501_501078


namespace find_k_l501_501547

theorem find_k 
  (h : ∀ x, 2 * x ^ 2 + 14 * x + k = 0 → x = ((-14 + Real.sqrt 10) / 4) ∨ x = ((-14 - Real.sqrt 10) / 4)) :
  k = 93 / 4 :=
sorry

end find_k_l501_501547


namespace food_price_before_tax_and_tip_l501_501466

theorem food_price_before_tax_and_tip (total_paid : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (P : ℝ) (h1 : total_paid = 198) (h2 : tax_rate = 0.10) (h3 : tip_rate = 0.20) : 
  P = 150 :=
by
  -- Given that total_paid = 198, tax_rate = 0.10, tip_rate = 0.20,
  -- we should show that the actual price of the food before tax
  -- and tip is $150.
  sorry

end food_price_before_tax_and_tip_l501_501466


namespace arithmetic_sequence_sum_l501_501641

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℝ) (d : ℝ),
  (∀ n : ℕ, a (n+1) = a n + d) →
  (d = (1 / 2)) →
  (∑ i in finset.range 50, a (2*i + 1)) = 60 →
  (∑ i in finset.range 50, a (2*i + 2)) = 85 :=
by
  intros a d hadd hdiff hsum
  sorry

end arithmetic_sequence_sum_l501_501641


namespace coefficient_x2y7_in_expansion_l501_501727

theorem coefficient_x2y7_in_expansion :
  let p := (x - 2 * y) * (x + y) ^ 8 in
  coefficient (monomial 2 7) (expand p) = -48 := sorry

end coefficient_x2y7_in_expansion_l501_501727


namespace general_term_a_general_term_b_l501_501569

def arithmetic_sequence (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) :=
∀ n, a_n n = n ∧ S_n n = (n^2 + n) / 2

def sequence_b (b_n : ℕ → ℝ) (T_n : ℕ → ℝ) :=
  (b_n 1 = 1/2) ∧
  (∀ n, b_n (n+1) = (n+1) / n * b_n n) ∧ 
  (∀ n, b_n n = n / 2) ∧ 
  (∀ n, T_n n = (n^2 + n) / 4) ∧ 
  (∀ m, m = 1 → T_n m = 1/2)

-- Arithmetic sequence {a_n}
theorem general_term_a (a : ℕ → ℕ) (S : ℕ → ℕ) (h1 : a 2 = 2) (h2 : S 5 = 15) :
  arithmetic_sequence a S := sorry

-- Sequence {b_n}
theorem general_term_b (b : ℕ → ℝ) (T : ℕ → ℝ) (h1 : b 1 = 1/2) (h2 : ∀ n, b (n+1) = (n+1) / n * b n) :
  sequence_b b T := sorry

end general_term_a_general_term_b_l501_501569


namespace parallel_DE_FG_l501_501840

open EuclideanGeometry

theorem parallel_DE_FG (ABC : Triangle) (Γ : Circle) (A B C D E F G : Point) 
  (h_acute : acute ABC) 
  (h_circumcircle : circumcircle ABC Γ) 
  (h_D_on_AB : D ∈ [AB]) 
  (h_E_on_AC : E ∈ [AC]) 
  (h_AD_AE : dist A D = dist A E)
  (h_F_perp_bisector_BD : is_perpendicular_bisector_to_arc Γ B D F)
  (h_G_perp_bisector_CE : is_perpendicular_bisector_to_arc Γ C E G) : 
  is_parallel (segment D E) (segment F G) := 
sorry

end parallel_DE_FG_l501_501840


namespace percent_problem_l501_501284

theorem percent_problem (x : ℝ) (h : 0.35 * 400 = 0.20 * x) : x = 700 :=
by sorry

end percent_problem_l501_501284


namespace smallest_prime_not_factor_of_large_n_l501_501846

open Nat

def large_n : Nat :=
  20 * 30 * 40 * 50 * 60 * 70 * 80 * 90 * 100 * 110 * 120 - 130

theorem smallest_prime_not_factor_of_large_n :
  ∃ p, Prime p ∧ p > 0 ∧ ¬ (p ∣ large_n) ∧ ∀ q, Prime q ∧ ¬ (q ∣ large_n) → p ≤ q :=
  by
    use 17
    sorry

end smallest_prime_not_factor_of_large_n_l501_501846


namespace olivia_initial_quarters_l501_501026

theorem olivia_initial_quarters : 
  ∀ (spent_quarters left_quarters initial_quarters : ℕ),
  spent_quarters = 4 → left_quarters = 7 → initial_quarters = spent_quarters + left_quarters → initial_quarters = 11 :=
by
  intros spent_quarters left_quarters initial_quarters h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end olivia_initial_quarters_l501_501026


namespace integer_solutions_to_equation_l501_501921

-- Define the problem statement in Lean 4
theorem integer_solutions_to_equation :
  ∀ (x y : ℤ), (x ≠ 0) → (y ≠ 0) → (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 19) →
      (x, y) = (38, 38) ∨ (x, y) = (380, 20) ∨ (x, y) = (-342, 18) ∨ 
      (x, y) = (20, 380) ∨ (x, y) = (18, -342) :=
by
  sorry

end integer_solutions_to_equation_l501_501921


namespace central_angle_of_unfolded_side_surface_l501_501415

theorem central_angle_of_unfolded_side_surface
  (radius : ℝ) (slant_height : ℝ) (arc_length : ℝ) (central_angle_deg : ℝ)
  (h_radius : radius = 1)
  (h_slant_height : slant_height = 3)
  (h_arc_length : arc_length = 2 * Real.pi) :
  central_angle_deg = 120 :=
by
  sorry

end central_angle_of_unfolded_side_surface_l501_501415


namespace kernel_preserves_regular_structure_l501_501703

-- Definitions needed for the problem statement
def Polyhedron (V : Type) : Type :=
{ E : V → V → Prop // symmetric E ∧ irreflexive E ∧ ∀ v, ∃ u, E v u ∧ ∀ u, E v u → u ≠ v }

def RegularPolyhedron {V : Type} (P : Polyhedron V) : Prop :=
∀ v, ∃ u, P.E v u ∧ ∃ n : ℕ, ∀ f, polygon v u f n

def Convex {V : Type} (P : Polyhedron V) : Prop :=
∀ v, ∀ u, P.E v u → ∀ w, ∃ (λ ws : Polyhedron V, ws (P))).Vertex.E)

def Kernel {V : Type} (P : Polyhedron V) : Polyhedron V :=
sorry -- assume some construction of kernel

-- Main theorem
theorem kernel_preserves_regular_structure {V : Type} (P : Polyhedron V) (hReg : RegularPolyhedron P)
  (hConvex : Convex P) : RegularPolyhedron (Kernel P) :=
sorry

end kernel_preserves_regular_structure_l501_501703


namespace olivia_baggies_l501_501027

theorem olivia_baggies : ∃ (n : ℕ), n = Nat.gcd 33 67 := 
by
  -- The Lean theorem prover knows the GCD of 33 and 67 is 1, which we can use here.
  use Nat.gcd 33 67
  -- The proof is not required, so we leave this as a placeholder with 'sorry'.
  sorry

end olivia_baggies_l501_501027


namespace count_multiples_of_12_between_15_and_165_l501_501613

theorem count_multiples_of_12_between_15_and_165 : 
  let count := (finset.filter (λ n, 15 < 12 * n ∧ 12 * n < 165) (finset.range 14)).card in
  count = 12 :=
by {
  sorry
}

end count_multiples_of_12_between_15_and_165_l501_501613


namespace find_y_l501_501286

theorem find_y (x y : ℝ) (h1 : x = 8) (h2 : x^(3 * y) = 64) : y = 2 / 3 :=
by
  -- Proof omitted
  sorry

end find_y_l501_501286


namespace magnitude_of_complex_l501_501528

theorem magnitude_of_complex : complex.abs (⟨12, -5⟩ : ℂ) = 13 := 
by
  sorry

end magnitude_of_complex_l501_501528


namespace find_c_find_perimeter_l501_501963

noncomputable def cos_A := sorry
noncomputable def sin_C := sorry

variables (a b c : ℝ) (A B C : ℝ)

-- Conditions
def given_conditions [Triangle ABC] (h1 : c * cos A = 5) (h2 : a * sin C = 4) (h3 : sin A ^ 2 + cos A ^ 2 = 1) (S : ℝ) : Prop :=
  h1 ∧ h2 ∧ h3 ∧ S = 16

-- Proving c = √41
theorem find_c (h1 : c * cos A = 5) (h2 : a * sin C = 4) (h3 : sin A ^ 2 + cos A ^ 2 = 1) : 
  c = Real.sqrt 41 :=
sorry

-- Using the area of triangle and finding the perimeter
theorem find_perimeter [Triangle ABC] (a b : ℝ) (c : ℝ) (h1 : c = Real.sqrt 41) (h2 : a * sin C = 4) (S : ℝ) (h4 : S = 16) (h5 : b = 8) : 
  perimeter ABC = 13 + c :=
sorry

end find_c_find_perimeter_l501_501963


namespace range_of_log_sqrt_sin_is_neg_infty_0_l501_501811

noncomputable def range_of_log_sqrt_sin : Set ℝ :=
  {y : ℝ | ∃ x ∈ Icc (0 : ℝ) (Real.pi / 2), y = Real.log10 (Real.sqrt (Real.sin x))}

theorem range_of_log_sqrt_sin_is_neg_infty_0 :
  range_of_log_sqrt_sin = Iic 0 := 
sorry

end range_of_log_sqrt_sin_is_neg_infty_0_l501_501811


namespace necessary_but_not_sufficient_l501_501241

noncomputable def planes_condition (α β : Plane) (m n : Line) : Prop :=
  (α ≠ β) → (m ⟂ α) → (n ⟂ β) → (∃ (h : α ∩ β), skew m n) ↔ 
  ((α ∩ β) → skew m n ∧ ¬(skew m n → α ∩ β))

-- Lean statement of the theorem
theorem necessary_but_not_sufficient (α β : Plane) (m n : Line) : Prop :=
  (α ≠ β) → (m ⟂ α) → (n ⟂ β) → 
  (∃ h₁ : intersect α β, skew m n) ↔ (∀ h₂ : intersect α β, skew m n ∧ ¬(skew m n → intersect α β)) :=
sorry

end necessary_but_not_sufficient_l501_501241


namespace prob_four_questions_to_advance_l501_501633

open ProbabilityTheory

/-- Definition of probability of answering a question correctly -/
def prob_correct : ℝ := 0.8

/-- Definition of probability of answering a question incorrectly -/
def prob_incorrect : ℝ := 1 - prob_correct

/-- Probability that the contestant correctly answers the first question -/
def p1 : ℝ := prob_correct

/-- Probability that the contestant correctly answers the second question -/
def p2 : ℝ := prob_correct

/-- Probability that the contestant incorrectly answers the third question -/
def p3 : ℝ := prob_incorrect

/-- Probability that the contestant correctly answers the fourth question -/
def p4 : ℝ := prob_correct

/-- Theorem stating the probability that the contestant answers exactly four questions before advancing -/
theorem prob_four_questions_to_advance :
  p1 * p2 * p3 * p4 = 0.128 :=
by sorry

end prob_four_questions_to_advance_l501_501633


namespace problem_1_problem_2_problem_3_l501_501586

-- The sequences a_n, b_n, c_n, and S_n
def a_n (n : ℕ) : ℕ := n

def b_n (n : ℕ) : ℕ := 2^n

def c_n (n : ℕ) : ℕ := (List.range (n + 1)).count (λ x => ∃ k, x = b_n k)

def S_n (n : ℕ) : ℕ := (List.range (n + 1)).map c_n |>.sum

-- The proof statements
theorem problem_1 (k : ℕ) (hk : k > 0) : c_n (2^k) = k := by
  sorry

theorem problem_2 (k : ℕ) (hk : k > 0) : c_n (2^(k + 1) - 1) = k := by
  sorry

theorem problem_3 (k : ℕ) (hk : k > 0) : S_n (2^(k + 1) - 1) ≠ (k - 1) * 2^(k + 1) + 2 := by
  sorry

end problem_1_problem_2_problem_3_l501_501586


namespace find_second_number_l501_501393

def average (a b c : ℕ) : ℕ := (a + b + c) / 3

theorem find_second_number :
  let x := 40,
      sum_set := 10 + 60 + 35,
      avg_set := average 10 60 35,
      avg_new_set := avg_set + 5
  in avg_new_set = 40 → average 20 60 x = 40 → x = 40 :=
by 
  intros,
  sorry

end find_second_number_l501_501393


namespace small_planters_needed_l501_501033

-- This states the conditions for the problem
def Oshea_seeds := 200
def large_planters := 4
def large_planter_capacity := 20
def small_planter_capacity := 4
def remaining_seeds := Oshea_seeds - (large_planters * large_planter_capacity) 

-- The target we aim to prove: the number of small planters required
theorem small_planters_needed :
  remaining_seeds / small_planter_capacity = 30 := by
  sorry

end small_planters_needed_l501_501033


namespace crease_length_correct_l501_501862

variable (A B C : Point)
variable (a b c : ℝ)

-- Given a triangle ABC with side lengths 3, 4, and 5, where A is folded to B.
def isRightTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Hypothesis: The triangle ABC is a right triangle with the given side lengths.
hypothesis (h_triangle : isRightTriangle 3 4 5)

-- The length of the crease when point A is folded to point B.
noncomputable def length_of_crease (a b c : ℝ) : ℝ :=
  let d := (a^2 + b^2 : ℝ / b^2) in
  real.sqrt(d)

-- Prove that the length of the crease is 15/8 inches
theorem crease_length_correct (h : isRightTriangle 3 4 5) : length_of_crease 3 4 5 = 15 / 8 := by
  sorry

end crease_length_correct_l501_501862


namespace evaluate_expression_l501_501931

theorem evaluate_expression (x : ℤ) (h : x = 5) : 
  3 * (3 * (3 * (3 * (3 * x + 2) + 2) + 2) + 2) + 2 = 1457 := 
by
  rw [h]
  sorry

end evaluate_expression_l501_501931


namespace lcm_value_count_l501_501202

theorem lcm_value_count (a b : ℕ) (k : ℕ) (h1 : 9^9 = 3^18) (h2 : 12^12 = 2^24 * 3^12) 
  (h3 : 18^18 = 2^18 * 3^36) (h4 : k = 2^a * 3^b) (h5 : 18^18 = Nat.lcm (9^9) (Nat.lcm (12^12) k)) :
  ∃ n : ℕ, n = 25 :=
begin
  sorry
end

end lcm_value_count_l501_501202


namespace sum_A_inter_Z_l501_501264

noncomputable def A : set ℝ := { x : ℝ | abs (x - 1) < 2 }
def Z : set ℤ := set.univ

theorem sum_A_inter_Z : ∑ x in (A ∩ (Z : set ℝ)), x = 3 := by
  -- Summarize and skip proof steps
  sorry

end sum_A_inter_Z_l501_501264


namespace problem1_problem2_l501_501505

-- First problem: Proving the position of 43251
theorem problem1 (digits : List Nat) (sorted_numbers: List (List Nat)) (n : Nat) 
    (h1 : digits = [1, 2, 3, 4, 5]) 
    (h2 : sorted_numbers = List.permutations digits)
    (h3 : sorted_numbers = List.sort (· < ·) sorted_numbers)
    (h4 : n = 88) 
    : List.nth sorted_numbers (n-1) = some [4, 3, 2, 5, 1] := sorry

-- Second problem: Proving the sum of digits in each position
theorem problem2 (digits : List Nat) (positions : Nat) (sum_digits : Nat) 
    (h1 : digits = [1, 2, 3, 4, 5]) 
    (h2 : positions = (digits.sum * 24)) 
    (h3 : sum_digits = 5 * positions) 
    : sum_digits = 1800 := sorry

end problem1_problem2_l501_501505


namespace third_side_length_correct_l501_501639

noncomputable def triangle_third_side_length : ℝ :=
  let a := 10
  let b := 15
  let theta := real.pi * (135 / 180) -- convert degrees to radians
  real.sqrt (a^2 + b^2 - 2 * a * b * real.cos theta)

theorem third_side_length_correct :
  triangle_third_side_length = real.sqrt (325 - 150 * real.sqrt 2) :=
sorry

end third_side_length_correct_l501_501639


namespace sebastian_weekly_salary_l501_501383

-- Define the given conditions
def missed_days := 2
def deducted_salary := 745
def days_in_week := 5

-- The value we need to prove
def expected_weekly_salary := 1862.5

-- Define daily income based on the conditions
def daily_income := deducted_salary / missed_days

-- Calculate weekly income based on daily income
def weekly_salary := daily_income * days_in_week

-- State the theorem to be proved
theorem sebastian_weekly_salary :
  weekly_salary = expected_weekly_salary :=
by
  -- Insert the proof here
  sorry

end sebastian_weekly_salary_l501_501383


namespace prob_digit_3_in_7_div_13_l501_501699

open Rat Real

-- Define the repeating block of the decimal representation
def repeating_block_7_div_13 : list ℕ := [5, 3, 8, 4, 6, 1]

-- Prove that the repeating block of 7/13 is as defined
theorem prob_digit_3_in_7_div_13 : (1 : ℚ) / 6 = 
  let freq_3 := (list.count 3 repeating_block_7_div_13)
  let total_digits := repeating_block_7_div_13.length
  freq_3 / total_digits := by
  sorry

end prob_digit_3_in_7_div_13_l501_501699


namespace cos_sum_simplification_l501_501048

theorem cos_sum_simplification :
  cos (3 * π / 13) + cos (5 * π / 13) + cos (7 * π / 13) = (real.sqrt 13 - 1) / 4 :=
by
  sorry

end cos_sum_simplification_l501_501048


namespace distance_AC_l501_501370

noncomputable def distance_between_points (A B C : ℝ) := (A ≤ B ∧ B ≤ C) ∨ (C ≤ B ∧ B ≤ A)
noncomputable def AB : ℝ := 5
noncomputable def BC : ℝ := 4

theorem distance_AC : ∃ AC : ℝ, (AC = 1 ∨ AC = 9) :=
by
  use 1
  use 9
  sorry

end distance_AC_l501_501370


namespace no_zero_in_0_2_l501_501287

theorem no_zero_in_0_2 {f : ℝ → ℝ} :
  (∀ x ∈ (set.Ioo 0 16), f x = 0 → x ∈ set.Ioo 2 4) →
  (∀ x ∈ (set.Ioo 0 8), f x = 0 → x ∈ set.Ioo 2 4) →
  (∀ x ∈ (set.Ioo 0 6), f x = 0 → x ∈ set.Ioo 2 4) →
  (∀ x ∈ (set.Ioo 2 4), f x = 0 → x ∈ set.Ioo 2 4) →
  ∀ x ∈ set.Ioo 0 2, f x ≠ 0 :=
by sorry

end no_zero_in_0_2_l501_501287


namespace correct_propositions_count_l501_501073

-- Define the propositions as Lean propositions
def prop1 : Prop :=
  (∀ x : ℝ, x^2 - x = 0 → x = 1) ↔ (∀ x : ℝ, x ≠ 1 → x^2 - x ≠ 0)

def prop2 (p q : Prop) : Prop :=
  (¬p ∨ q) = false → (p ∧ ¬q) = true

def prop3 : Prop :=
  (∀ x : ℝ, x * (x - 2) ≤ 0 ↔ log 2 x ≤ 1)

def prop4 (p : Prop) : Prop :=
  (∃ x : ℝ, 2^x < x^2) ↔ ¬ (∀ x : ℝ, 2^x ≥ x^2)

-- The main assertion that there are exactly 3 correct propositions
theorem correct_propositions_count : 
  (prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4) := by
  sorry

end correct_propositions_count_l501_501073


namespace determine_values_of_a_and_b_l501_501575

namespace MathProofProblem

variables (a b : ℤ)

theorem determine_values_of_a_and_b :
  (b + 1 = 2) ∧ (a - 1 ≠ -3) ∧ (a - 1 = -3) ∧ (b + 1 ≠ 2) ∧ (a - 1 = 2) ∧ (b + 1 = -3) →
  a = 3 ∧ b = -4 := by
  sorry

end MathProofProblem

end determine_values_of_a_and_b_l501_501575


namespace singer_arrangements_l501_501825

-- Let's assume the 5 singers are represented by the indices 1 through 5

theorem singer_arrangements :
  ∀ (singers : List ℕ) (no_first : ℕ) (must_last : ℕ), 
  singers = [1, 2, 3, 4, 5] →
  no_first ∈ singers →
  must_last ∈ singers →
  no_first ≠ must_last →
  ∃ (arrangements : ℕ),
    arrangements = 18 :=
by
  sorry

end singer_arrangements_l501_501825


namespace minimum_knights_l501_501770

-- Definitions based on the conditions
def total_people := 1001
def is_knight (person : ℕ) : Prop := sorry -- Assume definition of knight
def is_liar (person : ℕ) : Prop := sorry    -- Assume definition of liar

-- Conditions
axiom next_to_each_knight_is_liar : ∀ (p : ℕ), is_knight p → is_liar (p + 1) ∨ is_liar (p - 1)
axiom next_to_each_liar_is_knight : ∀ (p : ℕ), is_liar p → is_knight (p + 1) ∨ is_knight (p - 1)

-- Proving the minimum number of knights
theorem minimum_knights : ∃ (k : ℕ), k ≤ total_people ∧ k ≥ 502 ∧ (∀ (n : ℕ), n ≥ k → is_knight n) :=
  sorry

end minimum_knights_l501_501770


namespace number_of_digits_in_N_l501_501405

noncomputable def N : ℕ := 2^12 * 5^8

theorem number_of_digits_in_N : (Nat.digits 10 N).length = 10 := by
  sorry

end number_of_digits_in_N_l501_501405


namespace infinite_points_inside_circle_l501_501909

theorem infinite_points_inside_circle:
  ∀ c : ℝ, c = 3 → ∀ x y : ℚ, 0 < x ∧ 0 < y  ∧ x^2 + y^2 < 9 → ∃ a b : ℚ, 0 < a ∧ 0 < b ∧ a^2 + b^2 < 9 :=
sorry

end infinite_points_inside_circle_l501_501909


namespace beginner_trigonometry_probability_l501_501105

noncomputable def probability_beginner_trigonometry (C : ℝ) : ℝ :=
let total_students := 2.5 * C in
let beginner_calculus := 0.8 * C in
let beginner_trigonometry := 1.2 * C in
begin
  (beginner_trigonometry / total_students)
end

theorem beginner_trigonometry_probability (C : ℝ) (hC : C > 0) :
  (probability_beginner_trigonometry C = 0.48) :=
by
  sorry

end beginner_trigonometry_probability_l501_501105


namespace find_sum_of_squares_l501_501917

variable (x y : ℝ)

theorem find_sum_of_squares (h₁ : x * y = 8) (h₂ : x^2 * y + x * y^2 + x + y = 94) : 
  x^2 + y^2 = 7540 / 81 :=
by
  sorry

end find_sum_of_squares_l501_501917


namespace problem1_problem2_l501_501949

noncomputable def tan_alpha_eq_two (α : ℝ) : Prop := tan α = 2

theorem problem1 (α : ℝ) (h : tan_alpha_eq_two α) : (2 * sin α + 2 * cos α) / (sin α - cos α) = 8 := by
  sorry

theorem problem2 (α : ℝ) (h : tan_alpha_eq_two α) :
  (cos (π - α) * cos (π / 2 + α) * sin (α - 3 * π / 2))
  / (sin (3 * π + α) * sin (α - π) * cos (π + α)) = -1 / 2 := by
  sorry

end problem1_problem2_l501_501949


namespace minimum_distance_mn_l501_501350

-- Define the ellipse and its left focus
def ellipse : Set (ℝ × ℝ) := { p | p.1^2 / 3 + p.2^2 = 1 }
def left_focus : (ℝ × ℝ) := (-√2, 0)

-- Conditions
def line_passing_through_focus (m: ℝ) : (ℝ × ℝ) → Prop :=
  λ p, ∃ x y, (x / y = m) ∧ (x, y) = p ∧ (x / 3 + y^2 = 1)

-- Perpendicular lines meeting x-axis at M and N
def perpendicular_meets_x_axis (n: ℝ) (A B: ℝ×ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let M := (A.1 + A.2 / n, 0)
  let N := (B.1 + B.2 / n, 0)
  (M, N)

-- Define the distance |MN|
def distance_between_points (p1 p2: ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the problem statement
theorem minimum_distance_mn :
  ∀ (m: ℝ), 0 < m → 
  ∃ (A B: ℝ × ℝ), 
    line_passing_through_focus m A ∧ 
    line_passing_through_focus m B ∧ 
    let (M, N) := perpendicular_meets_x_axis m A B in
    distance_between_points M N = real.sqrt 6 :=
sorry

end minimum_distance_mn_l501_501350


namespace chocolate_chips_cover_one_fourth_area_l501_501101

/-
Yumi has a flat circular chocolate chip cookie with a radius of 3 cm.
On the top of the cookie, there are k circular chocolate chips, each with
a radius of 0.3 cm. No two chocolate chips overlap, and no chocolate chip
hangs over the edge of the cookie. For what value of k is exactly 1/4 of
the area of the top of the cookie covered with chocolate chips?
-/

def cookie_radius : ℝ := 3
def chip_radius : ℝ := 0.3
def area_ratio : ℝ := 1 / 4
def k : ℕ := 25

theorem chocolate_chips_cover_one_fourth_area :
  ∀ (k : ℕ), (0 ≤ k) →
  let area_cookie := Real.pi * cookie_radius^2 in
  let area_chip := Real.pi * chip_radius^2 in
  (k * area_chip = area_ratio * area_cookie) ↔ (k = 25) :=
by
  sorry

end chocolate_chips_cover_one_fourth_area_l501_501101


namespace hiking_hours_l501_501792

-- Define the given conditions
def water_needed_violet_per_hour : ℕ := 800
def water_needed_dog_per_hour : ℕ := 400
def total_water_carry_capacity_liters : ℕ := 4.8 * 1000 -- converted to ml

-- Define the statement to prove
theorem hiking_hours : (total_water_carry_capacity_liters / (water_needed_violet_per_hour + water_needed_dog_per_hour)) = 4 := by
  sorry

end hiking_hours_l501_501792


namespace systems_on_second_street_l501_501708

-- Definitions based on the conditions
def commission_per_system : ℕ := 25
def total_commission : ℕ := 175
def systems_on_first_street (S : ℕ) := S / 2
def systems_on_third_street : ℕ := 0
def systems_on_fourth_street : ℕ := 1

-- Question: How many security systems did Rodney sell on the second street?
theorem systems_on_second_street (S : ℕ) :
  S / 2 + S + 0 + 1 = total_commission / commission_per_system → S = 4 :=
by
  intros h
  sorry

end systems_on_second_street_l501_501708


namespace min_value_of_f_l501_501194

noncomputable def f (x : ℝ) : ℝ := x^4 + 16*x + 256 / x^6

theorem min_value_of_f : ∃ x > 0, is_minimum x (f x) := 
sorry

end min_value_of_f_l501_501194


namespace sum_of_two_digit_factors_is_162_l501_501740

-- Define the number
def num := 6545

-- Define the condition: num can be written as a product of two two-digit numbers
def are_two_digit_numbers (a b : ℕ) : Prop :=
  10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = num

-- The theorem to prove
theorem sum_of_two_digit_factors_is_162 : ∃ a b : ℕ, are_two_digit_numbers a b ∧ a + b = 162 :=
sorry

end sum_of_two_digit_factors_is_162_l501_501740


namespace function_range_l501_501813

-- Define the function y = (x^2 + 4x + 3) / (x + 1)
noncomputable def f (x : ℝ) : ℝ := (x^2 + 4*x + 3) / (x + 1)

-- State the theorem regarding the range of the function
theorem function_range : set.range (λ (x : {x : ℝ // x ≠ -1}), f x) = set.Ioo (-(∞) : ℝ) 2 ∪ set.Ioo 2 (∞ : ℝ) :=
sorry

end function_range_l501_501813


namespace average_goals_per_game_l501_501906

def Carter : ℝ := 4
def Shelby : ℝ := Carter / 2
def Judah : ℝ := 2 * Shelby - 3
def Morgan : ℝ := Judah + 1
def Alex : ℝ := Carter / 2 - 2
def Taylor : ℝ := 1 / 3

def total_average_goals : ℝ := Carter + Shelby + Judah + Morgan + Alex + Taylor

theorem average_goals_per_game
  (Carter := 4 : ℝ)
  (Shelby := Carter / 2 : ℝ)
  (Judah := 2 * Shelby - 3 : ℝ)
  (Morgan := Judah + 1 : ℝ)
  (Alex := Carter / 2 - 2 : ℝ)
  (Taylor := 1 / 3 : ℝ)
  : total_average_goals = 28 / 3 := by
  sorry

end average_goals_per_game_l501_501906


namespace trout_ratio_l501_501904

theorem trout_ratio (caleb_trouts dad_trouts : ℕ) (h_c : caleb_trouts = 2) (h_d : dad_trouts = caleb_trouts + 4) :
  dad_trouts / (Nat.gcd dad_trouts caleb_trouts) = 3 ∧ caleb_trouts / (Nat.gcd dad_trouts caleb_trouts) = 1 :=
by
  sorry

end trout_ratio_l501_501904


namespace symmetric_points_parabola_l501_501177

theorem symmetric_points_parabola (x1 x2 y1 y2 m : ℝ) (h1 : y1 = 2 * x1^2) (h2 : y2 = 2 * x2^2)
    (h3 : x1 * x2 = -3 / 4) (h_sym: (y2 - y1) / (x2 - x1) = -1)
    (h_mid: (y2 + y1) / 2 = (x2 + x1) / 2 + m) :
    m = 2 := sorry

end symmetric_points_parabola_l501_501177


namespace solve_geom_seq_l501_501962

variable {a : ℕ → ℝ} (r : ℝ) (a1 : ℝ)

-- Conditions documentation:
-- 1. {a_n} is a geometric sequence with ratio r > 0.
-- 2. a_n > 0 for all n.
-- 3. a2 * a4 + 2 * a3 * a5 + a4 * a6 = 25.

def geom_seq (a : ℕ → ℝ) (r : ℝ) := ∀ n : ℕ, a (n + 1) = a n * r

axiom pos_seq (a : ℕ → ℝ) : ∀ n, a n > 0

axiom given_eq (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ) :
  (a 1 * r) * (a1 * r^3) + 2 * (a1 * r^2) * (a1 * r^4) + (a1 * r^3) * (a1 * r^5) = 25

theorem solve_geom_seq (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ)
  [geom_seq a r] [pos_seq a] [given_eq a r a1]:
  a 2 + a 4 = 5 := sorry

end solve_geom_seq_l501_501962


namespace greatest_consecutive_sum_l501_501085

theorem greatest_consecutive_sum (S : ℤ) (hS : S = 105) : 
  ∃ N : ℤ, (∃ a : ℤ, (N * (2 * a + N - 1) = 2 * S)) ∧ 
  (∀ M : ℤ, (∃ b : ℤ, (M * (2 * b + M - 1) = 2 * S)) → M ≤ N) ∧ N = 210 := 
sorry

end greatest_consecutive_sum_l501_501085


namespace pentagon_perimeter_l501_501434

theorem pentagon_perimeter 
  (A B C D E : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]
  [has_dist A B] [has_dist B C] [has_dist C D] [has_dist D E] [has_dist A E]
  [has_dist A C] [has_dist A D]
  (AB BC CD DE EA : ℝ)
  (AB_val : AB = 2)
  (BC_val : BC = 2)
  (CD_val : CD = 1)
  (DE_val : DE = 1)
  (EA_val : EA = 3)
  (AC_val : has_dist(A, C) = real.sqrt 5)
  (AD_val : has_dist(A, D) = real.sqrt 6)
  : AB + BC + CD + DE + EA = 9 :=
by
  rw [AB_val, BC_val, CD_val, DE_val, EA_val]
  norm_num
  sorry

end pentagon_perimeter_l501_501434


namespace percent_extra_cost_l501_501439

-- Define the individual prices of the filters
def price_filter1 := 12.45
def price_filter2 := 14.05
def price_filter3 := 11.50

-- Define the number of each type of filter
def num_filter1 := 2
def num_filter2 := 2
def num_filter3 := 1

-- Define the price of the kit
def kit_price := 72.50

-- Calculate the total price if purchased individually
def individual_total_price := (num_filter1 * price_filter1) + (num_filter2 * price_filter2) + (num_filter3 * price_filter3)

-- Calculate the amount saved by purchasing the kit
def amount_saved := individual_total_price - kit_price

-- Calculate the percent saved (or extra cost in this case)
def percent_saved := (amount_saved / individual_total_price) * 100

-- The theorem we want to prove
theorem percent_extra_cost : percent_saved = -12.40 := by
  sorry

end percent_extra_cost_l501_501439


namespace average_condition_l501_501057

theorem average_condition (x y : ℚ) : 
  (\sum i in range 1 151, (i : ℚ)) + x + y = 152 * (75 * x + y) ↔ 
  x = 50 / 11399 ∧ y = (-11275) / 151 :=
by
  have h_sum : ∑ i in finset.range 151, (i.succ : ℚ) = 11325 := 
    by sorry -- We sum up the series 1, 2, ..., 150
  
  split
  {
    intro h
    -- Proof direction (→), assuming \( (\sum i in range 1 151, (i : ℚ)) + x + y = 152 * (75 * x + y) \),
    -- we derive \( x = 50 / 11399 \) and \( y = -11275 / 151 \)
    sorry
  }
  {
    intro h
    -- Proof direction (←), assuming \( x = 50 / 11399 \) and \( y = -11275 / 151 \),
    -- we deduce \( (\sum i in range 1 151, (i : ℚ)) + x + y = 152 * (75 * x + y) \)
    sorry
  }

end average_condition_l501_501057


namespace expected_value_to_log_calculation_l501_501141

theorem expected_value_to_log_calculation :
  ∃ a b : ℕ, nat.coprime a b ∧
  (∃ S : ℕ, S = ∑ (p : ℕ) in (finset.range 2010).filter nat.prime, (nat.choose 2010 p)) ∧
  a = S ∧ b = 100 * 2^2010 - 100 * S ∧
  nat.ceil (real.logb 2 (100 * a + b)) = 2017 :=
sorry

end expected_value_to_log_calculation_l501_501141


namespace length_of_crease_proof_l501_501864

-- Define the points in the plane
variables {A B C D: Point}

noncomputable
def length_of_crease (A B C D : Point) : ℝ :=
  if right_triangle A B C 3 4 5 ∧ midpoint D A B then
    15 / 8
  else
    0

-- The proof statement
theorem length_of_crease_proof (A B C D : Point) :
  (right_triangle A B C 3 4 5 ∧ midpoint D A B) → length_of_crease A B C D = 15 / 8 :=
by sorry

end length_of_crease_proof_l501_501864


namespace minimum_knights_l501_501771

-- Definitions based on the conditions
def total_people := 1001
def is_knight (person : ℕ) : Prop := sorry -- Assume definition of knight
def is_liar (person : ℕ) : Prop := sorry    -- Assume definition of liar

-- Conditions
axiom next_to_each_knight_is_liar : ∀ (p : ℕ), is_knight p → is_liar (p + 1) ∨ is_liar (p - 1)
axiom next_to_each_liar_is_knight : ∀ (p : ℕ), is_liar p → is_knight (p + 1) ∨ is_knight (p - 1)

-- Proving the minimum number of knights
theorem minimum_knights : ∃ (k : ℕ), k ≤ total_people ∧ k ≥ 502 ∧ (∀ (n : ℕ), n ≥ k → is_knight n) :=
  sorry

end minimum_knights_l501_501771


namespace geometric_series_lim_sum_l501_501520

theorem geometric_series_lim_sum :
  let a := 3
  let r := -0.5
  (|r| < 1) →
  lim (Seq.sum (λ n, a * r^n)) = 2 :=
by
  let a := 3
  let r := -0.5
  intro hr
  have h1 : a = 3 := rfl
  have h2 : r = -0.5 := rfl
  have h3 : |r| < 1 := hr
  sorry

end geometric_series_lim_sum_l501_501520


namespace remainder_of_m_div_1000_l501_501347

   -- Define the set T
   def T : Set ℕ := {n | 1 ≤ n ∧ n ≤ 12}

   -- Define the computation of m
   noncomputable def m : ℕ := (3^12 - 2 * 2^12 + 1) / 2

   -- Statement for the proof problem
   theorem remainder_of_m_div_1000 : m % 1000 = 625 := by
     sorry
   
end remainder_of_m_div_1000_l501_501347


namespace fourth_root_of_390820584961_l501_501497

theorem fourth_root_of_390820584961 : (390820584961 : ℕ)^(1/4 : ℝ) = 76 :=
by
  sorry

end fourth_root_of_390820584961_l501_501497


namespace part1_part2_l501_501257

noncomputable section
open Real

section
variables {x A a b c : ℝ}
variables {k : ℤ}

def f (x : ℝ) : ℝ := sin (2 * x - (π / 6)) + 2 * cos x ^ 2 - 1

theorem part1 (k : ℤ) : 
  ∀ x : ℝ, 
  k * π - (π / 3) ≤ x ∧ x ≤ k * π + (π / 6) → 
    ∀ x₁ x₂, 
      k * π - (π / 3) ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ k * π + (π / 6) → 
        f x₁ < f x₂ := sorry

theorem part2 {A a b c : ℝ} 
  (h_a_seq : 2 * a = b + c) 
  (h_dot : b * c * cos A = 9) 
  (h_A_fA : f A = 1 / 2) 
  : 
  a = 3 * sqrt 2 := sorry

end

end part1_part2_l501_501257


namespace sin_eq_sin_is_necessary_but_not_sufficient_for_eq_l501_501071

-- Define the conditions
def sin_periodic (A B : ℝ) (k : ℤ) : Prop :=
  B = A + 2 * k * Real.pi

-- Statement of the problem
theorem sin_eq_sin_is_necessary_but_not_sufficient_for_eq (A B : ℝ) :
  (∃ k : ℤ, B = A + 2 * k * Real.pi) → sin A = sin B :=
by sorry

end sin_eq_sin_is_necessary_but_not_sufficient_for_eq_l501_501071


namespace min_value_a4b3c2_l501_501673

theorem min_value_a4b3c2 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : 1/a + 1/b + 1/c = 9) : a^4 * b^3 * c^2 ≥ 1/1152 := 
sorry

end min_value_a4b3c2_l501_501673


namespace even_number_remains_after_998_moves_l501_501600

theorem even_number_remains_after_998_moves :
  ∀ (n : ℕ) (seq : list ℕ), 
    seq = list.range (n + 1) → -- natural numbers 1 to n
    ∃ k : ℕ, k ∈ (iterate (λ (l : list ℕ), l.tail.tail ++ [nat.abs (l.head - l.tail.head)]) 998 seq) ∧ even k :=
begin
  sorry
end

end even_number_remains_after_998_moves_l501_501600


namespace percentage_men_speaking_french_l501_501304

-- Define the initial conditions as constants
variables (E : ℝ) (percentage_men : ℝ)
variables (percentage_french_employees : ℝ)
variables (percentage_women_not_french : ℝ)

-- Assume the given conditions
axiom percentage_men_condition : percentage_men = 0.70
axiom percentage_french_employees_condition : percentage_french_employees = 0.40
axiom percentage_women_not_french_condition : percentage_women_not_french = 83.33333333333331 / 100

-- The theorem we need to prove
theorem percentage_men_speaking_french :
  let percentage_women_speaking_french := 1 - percentage_women_not_french in
  let men_count := percentage_men * E in
  let women_count := (1 - percentage_men) * E in
  let french_count := percentage_french_employees * E in
  let women_french_count := percentage_women_speaking_french * women_count in
  let men_french_count := french_count - women_french_count in
  let percentage_men_french := (men_french_count / men_count) * 100 in
  percentage_men_french = 50 := sorry

end percentage_men_speaking_french_l501_501304


namespace proof_funcA_proof_funcB_proof_funcC_proof_funcD_l501_501094

open Real

/- Definitions and conditions from the problem -/
def cond1 (x : ℝ) := x ≥ 0
def cond2 (x : ℝ) := x > 0
def funcA (x : ℝ) := x + 1 + 1 / (x + 1)
def funcB (x : ℝ) := (x + 1) / sqrt x
def funcC (x : ℝ) := x + 1 / x
def funcD (x : ℝ) := sqrt (x^2 + 2) + 1 / sqrt (x^2 + 2)

/- Statements to be proven -/
theorem proof_funcA (x : ℝ) (h : cond1 x) : funcA x ≥ 2 := 
by sorry

theorem proof_funcB (x : ℝ) (h : cond2 x) : funcB x ≥ 2 := 
by sorry

theorem proof_funcC (x : ℝ) (h : cond2 x) : ∀ y, y = funcC x -> y ≥ 2 :=
by sorry

theorem proof_funcD (x : ℝ) : ¬(∃ y, y = funcD x ∧ y = 2) :=
by sorry

end proof_funcA_proof_funcB_proof_funcC_proof_funcD_l501_501094


namespace find_a9_l501_501746

variable {a_n : ℕ → ℝ}

-- Definition of arithmetic progression
def is_arithmetic_progression (a : ℕ → ℝ) (a1 d : ℝ) := ∀ n : ℕ, a n = a1 + (n - 1) * d

-- Conditions
variables (a1 d : ℝ)
variable (h1 : a1 + (a1 + d)^2 = -3)
variable (h2 : ((a1 + a1 + 4 * d) * 5 / 2) = 10)

-- Question, needing the final statement
theorem find_a9 (a : ℕ → ℝ) (ha : is_arithmetic_progression a a1 d) : a 9 = 20 :=
by
    -- Since the theorem requires solving the statements, we use sorry to skip the proof.
    sorry

end find_a9_l501_501746


namespace order_of_terms_l501_501224

theorem order_of_terms (m n : ℝ) (hm : m < 0) (hn : -1 < n ∧ n < 0) : 
  m < m * n ^ 2 ∧ m * n ^ 2 < m * n :=
by
  intro
  sorry

end order_of_terms_l501_501224


namespace min_sum_of_dimensions_l501_501765

theorem min_sum_of_dimensions (a b c : ℕ) (h1 : a * b * c = 1645) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : 
  a + b + c ≥ 129 :=
sorry

end min_sum_of_dimensions_l501_501765


namespace necessary_but_not_sufficient_ellipse_l501_501320

def is_ellipse (m : ℝ) : Prop := 
  1 < m ∧ m < 3 ∧ m ≠ 2

theorem necessary_but_not_sufficient_ellipse (m : ℝ) :
  (1 < m ∧ m < 3) → (m ≠ 2) → is_ellipse m :=
by
  intros h₁ h₂
  have h : 1 < m ∧ m < 3 ∧ m ≠ 2 := ⟨h₁.left, h₁.right, h₂⟩
  exact h

end necessary_but_not_sufficient_ellipse_l501_501320


namespace middle_part_concert_arrivals_l501_501359

theorem middle_part_concert_arrivals :
  ∀ (total_tickets : ℕ) (before_start : ℚ) (after_first_song : ℚ) (not_attended : ℕ),
    total_tickets = 900 →
    before_start = 3 / 4 →
    after_first_song = 5 / 9 →
    not_attended = 20 →
    let before_concert := (before_start * total_tickets).to_nat in
    let remaining_after_concert := total_tickets - before_concert in
    let after_first_song_count := (after_first_song * remaining_after_concert).to_nat in
    let middle_part_concert := remaining_after_concert - after_first_song_count - not_attended in
    middle_part_concert = 80 :=
by
  intros total_tickets before_start after_first_song not_attended
  rintros rfl rfl rfl rfl
  let before_concert := (3 / 4 * 900).to_nat
  let remaining_after_concert := 900 - before_concert
  let after_first_song_count := (5 / 9 * remaining_after_concert).to_nat
  let middle_part_concert := remaining_after_concert - after_first_song_count - 20
  show middle_part_concert = 80 from
    by sorry

end middle_part_concert_arrivals_l501_501359


namespace Rachelle_GPA_Probability_correct_l501_501706

noncomputable def Rachelle_GPA_Probability : ℚ :=
  let P_A_English := 1 / 7 in
  let P_B_English := 1 / 5 in
  let P_C_English := 1 / 3 in
  let P_D_English := 1 - (P_A_English + P_B_English + P_C_English) in
  let P_A_History := 1 / 5 in
  let P_B_History := 1 / 4 in
  let P_C_History := 1 / 2 in
  let P_D_History := 1 - (P_A_History + P_B_History + P_C_History) in

  -- Ensure no D grades
  let P_NotD_English := 1 - P_D_English in
  let P_NotD_History := 1 - P_D_History in

  -- Calculate probabilities for achieving GPA >= 3.5 (Total Points ≥ 14)
  let P_A_A := P_A_English * P_A_History in
  let P_A_B := P_A_English * P_B_History in
  let P_B_A := P_B_English * P_A_History in
  let P_B_B := P_B_English * P_B_History in

  -- Total probability of Rachelle achieving GPA ≥ 3.5 excluding D grades
  let Total_Probability := P_A_A + P_A_B + P_B_A + P_B_B in

  -- Return answer
  Total_Probability

theorem Rachelle_GPA_Probability_correct : Rachelle_GPA_Probability = 27 / 175 :=
by admit -- sorry to skip the detailed proof steps.

end Rachelle_GPA_Probability_correct_l501_501706


namespace impossible_to_reach_l501_501170

-- Define the initial and final grid configurations as matrices
def initial_grid : Matrix (Fin 3) (Fin 3) ℤ :=
  !![ 0, 1, 0,
      1, 0, 1,
      0, 1, 0 ]

def final_grid : Matrix (Fin 3) (Fin 3) ℤ :=
  !![ 1, 0, 1,
      0, 1, 0,
      1, 0, 1 ]

-- Sum of elements in a 3x3 grid
def grid_sum (m : Matrix (Fin 3) (Fin 3) ℤ) : ℤ :=
  m.all (0:Fin 3) (0:Fin 3)

-- Possible changes in the grid sum via allowed moves
def is_valid_move_change (current_sum : ℤ) : Prop :=
  ∃ k : ℤ, (current_sum + k * 3) % 3 = 0

noncomputable def initial_grid_sum := grid_sum initial_grid
noncomputable def final_grid_sum := grid_sum final_grid

theorem impossible_to_reach :
  is_valid_move_change 4 → is_valid_move_change 5 → False :=
by
  intro h_initial h_final
  sorry

end impossible_to_reach_l501_501170


namespace hiking_hours_l501_501793

theorem hiking_hours
  (violet_water_per_hour : ℕ := 800)
  (dog_water_per_hour : ℕ := 400)
  (total_water : ℕ := 4800) :
  (total_water / (violet_water_per_hour + dog_water_per_hour) = 4) :=
by
  sorry

end hiking_hours_l501_501793


namespace dolphin_population_estimate_l501_501476

theorem dolphin_population_estimate 
  (tagged_jan : ℕ)  -- Number of dolphins tagged on January 1st
  (captured_jun : ℕ)  -- Number of dolphins captured on June 1st
  (tagged_jun : ℕ)  -- Number of tagged dolphins found on June 1st
  (migration_rate : ℚ)  -- Percentage of dolphins that migrated
  (new_arrival_rate : ℚ)  -- Percentage of new dolphins on June 1st
  (initial_population : ℕ)  -- Estimated initial population on January 1st
  :
  tagged_jan = 100 ∧ 
  captured_jun = 90 ∧
  tagged_jun = 4 ∧
  migration_rate = 20/100 ∧
  new_arrival_rate = 50/100 ∧
  initial_population = 1125 
  → let present_jun := captured_jun * (1 - new_arrival_rate) in 
     (tagged_jun : ℚ) / present_jun = (tagged_jan : ℚ) / initial_population := 
begin
  intros h,
  let present_jun := captured_jun * (1 - new_arrival_rate),
  sorry
end

end dolphin_population_estimate_l501_501476


namespace vincent_correct_answer_l501_501430

theorem vincent_correct_answer (y : ℕ) (h : (y - 7) / 5 = 23) : (y - 5) / 7 = 17 :=
by
  sorry

end vincent_correct_answer_l501_501430


namespace neg_five_cubed_is_neg_of_five_cubed_l501_501063

theorem neg_five_cubed_is_neg_of_five_cubed (a : ℤ) : a = -5 → -a^3 = -(5^3) :=
by
  intro ha
  rw ha
  simp
  sorry

end neg_five_cubed_is_neg_of_five_cubed_l501_501063


namespace power_function_decreasing_and_even_l501_501410

-- Definitions based on problem conditions
def is_decreasing_on (f : ℝ → ℝ) (I : set ℝ) := ∀ x ∈ I, ∀ y ∈ I, x < y → f x > f y

def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

-- Main theorem based on problem and solution
theorem power_function_decreasing_and_even (m : ℕ) :
  (is_decreasing_on (λ x : ℝ, x ^ (3 * m - 5)) (set.Ioi 0) ∧ 
   even_function (λ x : ℝ, x ^ (3 * m - 5))) → m = 1 :=
by
  sorry

end power_function_decreasing_and_even_l501_501410


namespace square_side_length_l501_501483

theorem square_side_length (area : ℝ) (h : area = 225) : ∃ s : ℝ, s * s = area ∧ s = 15 := by
  use 15
  split
  · rw h
    norm_num
  · norm_num

end square_side_length_l501_501483


namespace perpendicular_iff_even_function_l501_501573

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def is_even_function (f : ℝ → ℝ) :=
∀ x, f (-x) = f x

theorem perpendicular_iff_even_function
  (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ x, ((inner a (x•a)) + b) ^ 2 = ((inner a (-x•a)) + b) ^ 2) ↔ inner a b = 0 :=
sorry

end perpendicular_iff_even_function_l501_501573


namespace find_a_for_odd_function_l501_501979

def f (x : ℝ) (a : ℝ) := a - 1 / (2^x + 1)

theorem find_a_for_odd_function (a : ℝ) : (f 0 a = 0) → a = 1 / 2 :=
by
  intro h
  rw [f, pow_zero, add_comm, div_one, sub_eq_zero] at h
  assumption
  sorry

end find_a_for_odd_function_l501_501979


namespace common_ratio_geometric_seq_l501_501247

noncomputable def common_ratio (a b : ℕ → ℝ) (d : ℝ) : ℝ :=
  if d > 0 ∧ (∀ n, a (n + 1) = a n + d) ∧ (b 1 = a 1 ^ 2) ∧ (b 2 = (a 1 + d) ^ 2) ∧ (b 3 = (a 1 + 2 * d) ^ 2) then
    (b 2 / b 1)
  else 0

theorem common_ratio_geometric_seq (a b : ℕ → ℝ) (d : ℝ) (h1 : d > 0) (h2 : ∀ n, a (n + 1) = a n + d) 
  (h3 : b 1 = a 1 ^ 2) (h4 : b 2 = (a 1 + d) ^ 2) (h5 : b 3 = (a 1 + 2 * d) ^ 2) : 
  common_ratio a b d = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end common_ratio_geometric_seq_l501_501247


namespace find_sin_value_l501_501559

noncomputable def tan (x : ℝ) : ℝ := sin x / cos x

theorem find_sin_value (α : ℝ) 
  (h1 : tan (2 * α) = 3/4) 
  (h2 : α ∈ Ioo (-real.pi / 2) (real.pi / 2)) 
  (h3 : ∀ x : ℝ, (sin (x + α) + sin (α - x) - 2 * sin α) ≥ 0) : 
  sin (α - real.pi / 4) = -2 * real.sqrt 5 / 5 :=
sorry

end find_sin_value_l501_501559


namespace percentage_of_water_in_dried_grapes_l501_501220

def fresh_grapes_water_content := 0.90
def fresh_grapes_weight := 20 -- kg
def dried_grapes_weight := 2.5 -- kg

theorem percentage_of_water_in_dried_grapes :
  let solid_content_weight := fresh_grapes_weight * (1 - fresh_grapes_water_content) in
  let weight_of_water_in_dried_grapes := dried_grapes_weight - solid_content_weight in
  (weight_of_water_in_dried_grapes / dried_grapes_weight) * 100 = 20 :=
by
  sorry

end percentage_of_water_in_dried_grapes_l501_501220


namespace average_marks_correct_l501_501392

def average_marks_class1 : ℕ := 45
def students_class1 : ℕ := 35

def average_marks_class2 : ℕ := 65
def students_class2 : ℕ := 55

def total_marks_class1 : ℕ := students_class1 * average_marks_class1
def total_marks_class2 : ℕ := students_class2 * average_marks_class2

def total_students : ℕ := students_class1 + students_class2
def total_marks : ℕ := total_marks_class1 + total_marks_class2

def average_marks_all : ℚ := total_marks / total_students

theorem average_marks_correct : average_marks_all ≈ 57.22 := 
by {
  -- This would be the point where the proof steps would go.
  sorry
}

end average_marks_correct_l501_501392


namespace general_term_formula_l501_501003

def Sn (n : ℕ) : ℕ := n^2 - 3 * n + 3

def an : ℕ → ℕ
| 1       := 1
| (n + 1) := 2 * (n + 1) - 4

theorem general_term_formula :
  ∀ n, an n = match n with
               | 1       := 1
               | (n + 1) := 2 * (n + 1) - 4
               end :=
begin
  intro n,
  cases n,
  { -- case n = 1
    simp [an, Sn], },
  { -- case n >= 2
    simp [an, Sn],
    sorry
  }
end

end general_term_formula_l501_501003


namespace trig_identity_l501_501971

theorem trig_identity (θ : ℝ) (h : sin (π / 2 + θ) + 3 * cos (θ - π) = sin (-θ)) : sin θ * cos θ + cos θ^2 = 3 / 5 :=
by
  sorry

end trig_identity_l501_501971


namespace right_triangle_area_l501_501805

theorem right_triangle_area (a b c : ℝ) (h₁ : a = 24) (h₂ : c = 26) (h₃ : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 120 :=
by
  sorry

end right_triangle_area_l501_501805


namespace number_of_paths_l501_501351

theorem number_of_paths (n s : ℕ) (hn : 1 ≤ n) (hs : 1 ≤ s ∧ s ≤ n) :
  num_paths (0, 0) (n, n) s = (1 / s) * Nat.choose (n-1) (s-1) * Nat.choose n (s-1) :=
sorry

end number_of_paths_l501_501351


namespace find_a_l501_501292

-- Definitions of the mathematical functions and conditions
def f : ℝ → ℝ
| x => if x > 0 then Real.log x else x + ∫ t in 0..a, 3 * t^2

-- Assertion of the goal given conditions
theorem find_a (a : ℝ) (h : f(f(1)) = 27) : a = 3 :=
sorry

end find_a_l501_501292


namespace day_shift_production_times_l501_501995

theorem day_shift_production_times (total_production : ℕ) (day_shift_production : ℕ) (second_shift_production : ℕ)
  (h_total : total_production = 5500)
  (h_day_shift : day_shift_production = 4400)
  (h_second_shift : second_shift_production = total_production - day_shift_production) :
  day_shift_production = 4 * second_shift_production :=
by {
  subst h_total,
  subst h_day_shift,
  subst h_second_shift,
  sorry
}

end day_shift_production_times_l501_501995


namespace real_inequality_l501_501114

noncomputable def inequality (n : ℕ) (a b : Fin n → ℝ) :=
  (∀ i, 0 < a i) →
  (∀ i, 0 < b i) →
  (Finset.univ.sum a) * (Finset.univ.sum b) ≥ (Finset.univ.sum (λ i, a i + b i)) * (Finset.univ.sum (λ i, (a i * b i) / (a i + b i)))

theorem real_inequality (n : ℕ) (a b : Fin n → ℝ) (h_pos_a : ∀ i, 0 < a i) (h_pos_b : ∀ i, 0 < b i) :
  inequality n a b :=
sorry

end real_inequality_l501_501114


namespace cross_section_area_l501_501478

-- Define the conditions and parameters for the problem
variables (a : ℝ)

-- The proof goal: the area of the cross-section through specified conditions
theorem cross_section_area (a : ℝ) : ∃ area : ℝ, 
  area = (a^2 * Real.sqrt 29) / 8 :=
begin
  use (a^2 * Real.sqrt 29) / 8,
  sorry
end

end cross_section_area_l501_501478


namespace rightmost_three_digits_of_5_pow_1993_l501_501800

theorem rightmost_three_digits_of_5_pow_1993 : (5^1993 : ℕ) % 1000 = 125 := by
  -- We state and rewrite the conditions inferred from the problem
  have h0 : (5^0 : ℕ) % 1000 = 1 := by norm_num [Nat.pow_zero, Nat.mod_1_eq, Nat.cast_zero, Nat.cast_one]
  have h1 : (5^1 : ℕ) % 1000 = 5 := by norm_num
  have h2 : (5^2 : ℕ) % 1000 = 25 := by norm_num
  have h3 : (5^3 : ℕ) % 1000 = 125 := by norm_num
  have h4 : (5^4 : ℕ) % 1000 = 625 := by norm_num
  have h_odd : ∀ n, n > 2 ∧ n % 2 = 1 → (5^n : ℕ) % 1000 = 125 := by
    intros n hn
    suffices hn4 : (5^4 : ℕ) % 1000 = 625 by
    obtain ⟨m, rfl⟩: ∃ m, n = 2 * m + 3 := exists_eq_odd_n hn.left hn.right
    calc
      (5^(2 * m + 3) : ℕ) % 1000 = (5^3 * (5 * 5^4)^m : ℕ) % 1000 := by ring_exp
      ... = (125 * 625^m : ℕ) % 1000 := by rw [h3, hn4]
      ... = (125 * 0 : ℕ) % 1000 := by
        rw [Nat.pow_mul_mod _ _ 1000, Nat.mod_eq_zero_of_dvd 1000, Nat.mul_zero, Nat.zero_mul]
        exact nat.dvd_pow (by norm_num) m
      ... = 125 := by norm_num
  exact h_odd 1993 (by norm_num)

-- sorry in the completion of proof steps can be used if detailed proofs are required later

end rightmost_three_digits_of_5_pow_1993_l501_501800


namespace cost_to_feed_chickens_l501_501701

/-- Peter has 18 birds. Among them, 1/3 are ducks, 1/4 are parrots,
and the rest are chickens. Each type of bird requires special
feed: ducks need $3 per bird, parrots need $4 per bird, and chickens need $2 per bird.
Prove that it costs $16 to feed all the chickens. -/
theorem cost_to_feed_chickens :
  let total_birds := 18
  let ducks := total_birds / 3
  let parrots := total_birds / 4
  let chickens := total_birds - ducks - parrots
  let cost_per_chicken := 2
  in chickens * cost_per_chicken = 16 := by
  sorry

end cost_to_feed_chickens_l501_501701


namespace average_speed_l501_501124

theorem average_speed (v1 v2 t1 t2 total_time total_distance : ℝ)
  (h1 : v1 = 50)
  (h2 : t1 = 4)
  (h3 : v2 = 80)
  (h4 : t2 = 4)
  (h5 : total_time = t1 + t2)
  (h6 : total_distance = v1 * t1 + v2 * t2) :
  (total_distance / total_time = 65) :=
by
  sorry

end average_speed_l501_501124


namespace find_length_YW_l501_501637

theorem find_length_YW {d : ℝ} (h : d ≥ sqrt 7) : 
  ∃ y : ℝ, y = sqrt (d^2 - 7) :=
by
  use sqrt (d^2 - 7)
  sorry

end find_length_YW_l501_501637


namespace number_of_true_propositions_l501_501954

def line : Type := ℝ → ℝ → ℝ
def plane : Type := ℝ → ℝ

variables {l m : line} {α β : plane}

-- Conditions
def perpendicular_to_plane (l : line) (α : plane) : Prop := sorry  -- formalize this later
def line_in_plane (m : line) (β : plane) : Prop := sorry  -- formalize this later

def parallel_planes (α β : plane) : Prop := sorry  -- formalize this later
def perpendicular_planes (α β : plane) : Prop := sorry  -- formalize this later

-- Propositions
def prop1 := parallel_planes α β → perpendicular_to_plane l m
def prop2 := perpendicular_to_plane l m → parallel_planes α β
def prop3 := perpendicular_planes α β → l = m  -- assuming l parallel to m means equality here
def prop4 := l = m → perpendicular_planes α β

-- Condition mapping
def condition1 : Prop := perpendicular_to_plane l α
def condition2 : Prop := line_in_plane m β

-- Problem statement
theorem number_of_true_propositions :
  condition1 →
  condition2 →
  (prop1 ↔ false) ∧
  (prop2 ↔ true) ∧
  (prop3 ↔ false) ∧
  (prop4 ↔ false) →
  nat.succ 0 = 1 :=
by
  intros
  sorry

end number_of_true_propositions_l501_501954


namespace four_digit_numbers_count_l501_501490

theorem four_digit_numbers_count : 
  let digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  let abs_diff_condition := λ (u h: ℕ), abs (u - h) = 8
  (∃ n : ℕ, ∃ (a b c d : ℕ), 
    a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ 
    list.nodup [a, b, c, d] ∧ 
    n = 1000*a + 100*b + 10*c + d ∧ 
    abs_diff_condition d b) → 
  210 := sorry

end four_digit_numbers_count_l501_501490


namespace determinant_magnitude_eq_sqrt5_l501_501689

open Complex

theorem determinant_magnitude_eq_sqrt5 (z : ℂ) 
  (h : det ![![1, Complex.i], ![1 - 2 * Complex.i, z]] = 0) : 
  Complex.abs z = Real.sqrt 5 := 
sorry

end determinant_magnitude_eq_sqrt5_l501_501689


namespace solution_set_f_ge_1_range_of_m_l501_501981

def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Question 1: Prove the solution set of f(x) ≥ 1 is {x | x ≥ 1}
theorem solution_set_f_ge_1 (x : ℝ) : f(x) ≥ 1 ↔ x ≥ 1 := sorry

-- Question 2: Prove the range of m for which the inequality f(x) ≥ x^2 - x + m has non-empty solution set is (-∞, 5/4]
theorem range_of_m (m : ℝ) : (∃ x : ℝ, f(x) ≥ x^2 - x + m) ↔ m ≤ 5 / 4 := sorry

end solution_set_f_ge_1_range_of_m_l501_501981


namespace arithmetic_sequence_sum_bn_l501_501957

variable {n : ℕ}
variable (a : ℕ → ℕ) {b : ℕ → ℕ}

noncomputable def Sn (n : ℕ) : ℕ := n * n
noncomputable def an (n : ℕ) : ℕ := n * n - (n - 1) * (n - 1)
noncomputable def bn (n : ℕ) : ℕ := 2 ^ (an n) + (-1) ^ n * an n

theorem arithmetic_sequence : ∀ n ≥ 1, an n = 2 * n - 1 := sorry

theorem sum_bn : ∀ n, (Finset.range (2 * n)).sum bn = 2 * 4 ^ n / 3 + 2 * n - 2 / 3 := sorry

end arithmetic_sequence_sum_bn_l501_501957


namespace lcm_value_count_l501_501203

theorem lcm_value_count (a b : ℕ) (k : ℕ) (h1 : 9^9 = 3^18) (h2 : 12^12 = 2^24 * 3^12) 
  (h3 : 18^18 = 2^18 * 3^36) (h4 : k = 2^a * 3^b) (h5 : 18^18 = Nat.lcm (9^9) (Nat.lcm (12^12) k)) :
  ∃ n : ℕ, n = 25 :=
begin
  sorry
end

end lcm_value_count_l501_501203


namespace portraits_count_l501_501495

theorem portraits_count (P S : ℕ) (h1 : S = 6 * P) (h2 : P + S = 200) : P = 28 := 
by
  -- The proof will be here.
  sorry

end portraits_count_l501_501495


namespace average_temperature_is_95_l501_501754

noncomputable def tempNY := 80
noncomputable def tempMiami := tempNY + 10
noncomputable def tempSD := tempMiami + 25
noncomputable def avg_temp := (tempNY + tempMiami + tempSD) / 3

theorem average_temperature_is_95 :
  avg_temp = 95 :=
by
  sorry

end average_temperature_is_95_l501_501754


namespace jaden_toy_cars_l501_501652

theorem jaden_toy_cars :
  let initial : Nat := 14
  let bought : Nat := 28
  let birthday : Nat := 12
  let to_sister : Nat := 8
  let to_friend : Nat := 3
  initial + bought + birthday - to_sister - to_friend = 43 :=
by
  let initial : Nat := 14
  let bought : Nat := 28
  let birthday : Nat := 12
  let to_sister : Nat := 8
  let to_friend : Nat := 3
  show initial + bought + birthday - to_sister - to_friend = 43
  sorry

end jaden_toy_cars_l501_501652


namespace representation_fewer_nonzero_l501_501338

theorem representation_fewer_nonzero 
  (n k : ℕ) (u : ℕ → ℕ) (hku : ∀ i, u i ≤ 2^k) (t : ℕ)
  (a : ℕ → ℕ) (hrep : t = ∑ i in finset.range n, a i * u i)
  (hk3 : 3 ≤ k) :
  ∃ b : ℕ → ℕ, t = ∑ i in finset.range n, b i * u i ∧ (∑ i in finset.range n, if b i = 0 then 0 else 1) < 2 * k := 
sorry

end representation_fewer_nonzero_l501_501338


namespace find_n_l501_501279

theorem find_n : ∃ n : ℤ, (n^2 / 4).toFloor - (n / 2).toFloor^2 = 5 ∧ n = 11 :=
by
  sorry

end find_n_l501_501279


namespace skew_lines_definition_l501_501386

-- Definitions based on the conditions
def non_intersecting_lines (L₁ L₂ : set ℝ^3) : Prop :=
  L₁ ∩ L₂ = ∅

def lines_in_different_planes (L₁ L₂ : set ℝ^3) : Prop :=
  ∃ (P₁ P₂ : set ℝ^3), L₁ ⊆ P₁ ∧ L₂ ⊆ P₂ ∧ P₁ ≠ P₂

def line_in_plane_and_not_in_plane (L₁ L₂ : set ℝ^3) : Prop :=
  ∃ (P : set ℝ^3), (L₁ ⊆ P ∧ ¬(L₂ ⊆ P)) ∨ (L₂ ⊆ P ∧ ¬(L₁ ⊆ P))

def not_in_single_plane (L₁ L₂ : set ℝ^3) : Prop :=
  ¬∃ (P : set ℝ^3), L₁ ⊆ P ∧ L₂ ⊆ P

-- The theorem to be proven
theorem skew_lines_definition (L₁ L₂ : set ℝ^3) :
  ((non_intersecting_lines L₁ L₂) ∧ (¬lines_in_different_planes L₁ L₂) ∧
  (¬line_in_plane_and_not_in_plane L₁ L₂) → not_in_single_plane L₁ L₂) :=
by
  sorry

end skew_lines_definition_l501_501386


namespace axis_of_symmetry_l501_501397

theorem axis_of_symmetry (x : ℝ) : 
  (∃ k : ℤ, x = k * real.pi / 4 + 5 * real.pi / 24) → x = 11 * real.pi / 24 :=
by
  intro h
  rcases h with ⟨k, hk⟩
  have := calc
    k * real.pi / 4 + 5 * real.pi / 24 = 11 * real.pi / 24 : sorry
  exact this

end axis_of_symmetry_l501_501397


namespace correct_term_is_D_l501_501273

-- Definitions for each condition
def option_A := "made up"
def option_B := "showed up"
def option_C := "picked up"
def option_D := "mixed up"

-- Theorem statement: prove that the correct term to use is "mixed up" given the context.
theorem correct_term_is_D (context : String) : (context = "He introduced himself and his pet dog at the same time. So I \_\_\_\_\_\_\_ their names and called the dog’s name instead of his.") →
option_D = "mixed up" :=
by 
  intros
  -- Proof not provided
  sorry

end correct_term_is_D_l501_501273


namespace grid_rows_l501_501693

theorem grid_rows (R : ℕ) :
  let squares_per_row := 15
  let red_squares := 4 * 6
  let blue_squares := 4 * squares_per_row
  let green_squares := 66
  let total_squares := red_squares + blue_squares + green_squares 
  total_squares = squares_per_row * R →
  R = 10 :=
by
  intros
  sorry

end grid_rows_l501_501693


namespace swimming_speed_l501_501138

theorem swimming_speed (s v : ℝ) (h_s : s = 4) (h_time : 1 / (v - s) = 2 * (1 / (v + s))) : v = 12 := 
by
  sorry

end swimming_speed_l501_501138


namespace popcorn_kernels_needed_l501_501116

theorem popcorn_kernels_needed
  (h1 : 2 * 4 = 4 * 1) -- Corresponds to "2 tablespoons make 4 cups"
  (joanie : 3) -- Joanie wants 3 cups
  (mitchell : 4) -- Mitchell wants 4 cups
  (miles_davis : 6) -- Miles and Davis together want 6 cups
  (cliff : 3) -- Cliff wants 3 cups
  : 2 * (joanie + mitchell + miles_davis + cliff) / 4 = 8 :=
by sorry

end popcorn_kernels_needed_l501_501116


namespace a_b_sum_possible_values_l501_501349

theorem a_b_sum_possible_values (a b : ℝ) 
  (h1 : a^3 - 12 * a^2 + 9 * a - 18 = 0)
  (h2 : 9 * b^3 - 135 * b^2 + 450 * b - 1650 = 0) :
  a + b = 6 ∨ a + b = 14 :=
sorry

end a_b_sum_possible_values_l501_501349


namespace complex_ellipse_l501_501728

theorem complex_ellipse (w : ℂ) (h : abs w = 3) : 
  ∃ a b : ℝ, (w = a + b * complex.i) ∧ ((a^2 + b^2 = 9) → (w + 2 / w) ∈ {z : ℂ | ∀ x y : ℝ, z = x + y * complex.i → (x^2 / (121/81) + y^2 / (49/81) = 1)}) :=
begin
  sorry
end

end complex_ellipse_l501_501728


namespace symmetry_center_l501_501042

-- Definitions of the conditions
def planar_figure (F : set (ℝ × ℝ)) : Prop := true -- Placeholder definition

def symmetric_about_x_axis (F : set (ℝ × ℝ)) : Prop :=
  ∀ (a b : ℝ), (a, b) ∈ F ↔ (a, -b) ∈ F

def symmetric_about_y_axis (F : set (ℝ × ℝ)) : Prop :=
  ∀ (a b : ℝ), (a, b) ∈ F ↔ (-a, b) ∈ F

def center_of_symmetry (F : set (ℝ × ℝ)) (O : ℝ × ℝ) : Prop :=
  ∀ (a b : ℝ), (a, b) ∈ F ↔ (-a, -b) ∈ F

-- Theorem statement
theorem symmetry_center (F : set (ℝ × ℝ)) (O : ℝ × ℝ)
  (h_planar : planar_figure F)
  (h_sym_x : symmetric_about_x_axis F)
  (h_sym_y : symmetric_about_y_axis F)
  (h_intersect : O = (0, 0)) :
  center_of_symmetry F O :=
by
  sorry

end symmetry_center_l501_501042


namespace equal_surface_area_of_intersections_l501_501343

noncomputable theory

-- Define the scene and given conditions
structure Sphere (α : Type*) [NormedGroup α] [NormedSpace ℝ α] :=
(center : α)
(radius : ℝ)

variables {α : Type*} [NormedGroup α] [NormedSpace ℝ α]

def sphere (O : α) (r : ℝ) : Sphere α :=
{ center := O,
  radius := r }

def surface_area_of_intersection (S1 S2 : Sphere α) : ℝ :=
  2 * real.pi * S1.radius * (S1.radius^2 / (2 * S2.radius))

-- Define spheres K, A, and B
variables (O P Q : α) (r R1 R2 : ℝ)
  (h1 : Sphere α) (h2 : Sphere α) (h3 : Sphere α)

-- Sphere K
def K : Sphere α := sphere O r

-- Sphere A
def A : Sphere α := sphere P (dist P O)

-- Sphere B
def B : Sphere α := sphere Q (dist Q O)

-- Prove that the surface area of parts of A and B that lie inside K are equal
theorem equal_surface_area_of_intersections :
  surface_area_of_intersection A K = surface_area_of_intersection B K :=
by
  -- Given conditions and derived formula for intersections
  sorry

end equal_surface_area_of_intersections_l501_501343


namespace sum_h_k_a_b_eq_14_l501_501016

-- Defining points F1 and F2
def F1 : (ℝ × ℝ) := (0, 2)
def F2 : (ℝ × ℝ) := (6, 2)

-- Define the given ellipse equation conditions
def ellipse_eq (x y h k a b : ℝ) : Prop :=
  ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1

-- Definition of the condition that PF1 + PF2 = 10 forms an ellipse
def is_ellipse (P : ℝ × ℝ) : Prop :=
  (Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) +
   Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)) = 10

-- Prove that the sum h + k + a + b for this ellipse equals 14
theorem sum_h_k_a_b_eq_14 : 
  ∃ h k a b : ℝ, 
    ellipse_eq (3 : ℝ) (2 : ℝ) h k a b ∧ 
    h = 3 ∧ k = 2 ∧ a = 5 ∧ b = 4 ∧ 
    h + k + a + b = 14 := 
by 
  existsi 3, 
  existsi 2, 
  existsi 5, 
  existsi 4,
  split,
  sorry,
  split,
  refl,
  split,
  refl,
  split,
  refl,
  split,
  refl,
  norm_num

end sum_h_k_a_b_eq_14_l501_501016


namespace angle_IFH_l501_501396

theorem angle_IFH {F H I G M N : Type*} [metric_space F] [metric_space H] [metric_space I]
  (α : ℝ) (GH_eq_FI : dist G H = dist F I) (M_mid_FG : midpoint M F G) (N_mid_HI : midpoint N H I)
  (angle_NMH_eq : ∠N M H = α) : ∠I F H = 2 * α :=
sorry

end angle_IFH_l501_501396


namespace find_a_l501_501974

noncomputable def f (x : ℝ) (a : ℝ) := (2 / x) - 2 + 2 * a * Real.log x

theorem find_a (a : ℝ) (h : ∃ x ∈ Set.Icc (1/2 : ℝ) 2, f x a = 0) : a = 1 := by
  sorry

end find_a_l501_501974


namespace buddy_has_40_baseball_cards_on_monday_l501_501028

theorem buddy_has_40_baseball_cards_on_monday :
  ∀ (cards_monday : ℕ), 
    (cards_monday / 2 = cards_tuesday) →
    (cards_wednesday = cards_tuesday + 12) →
    (cards_thursday = cards_wednesday + cards_tuesday / 3) →
    (cards_thursday = 32) →
    cards_monday = 40 :=
begin
  sorry
end

end buddy_has_40_baseball_cards_on_monday_l501_501028


namespace norris_savings_l501_501361

def savings_sep := 29
def savings_oct := 25
def savings_nov := 31
def spending := 75

theorem norris_savings :
  (savings_sep + savings_oct + savings_nov - spending) = 10 :=
by
  have total_savings := savings_sep + savings_oct + savings_nov
  have total_savings_eq : total_savings = 85 := by norm_num
  have remaining := total_savings - spending
  have remaining_eq : remaining = 10 := by norm_num
  exact remaining_eq

end norris_savings_l501_501361


namespace sum_of_sequence_l501_501647

theorem sum_of_sequence (n : ℕ) (h_pos : 0 < n) :
  let a : ℕ → ℕ := λ n, 2^n,
      b : ℕ → ℕ := λ n, (3 * 2^n) / 2 in
  let S : ℕ → ℕ := λ n, 3 * (2^n - 1) in
  (S n = 3 * (2^n - 1)) :=
by
  sorry

end sum_of_sequence_l501_501647


namespace inequality_solution_empty_l501_501296

theorem inequality_solution_empty {a x: ℝ} : 
  (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0 → 
  (-2 < a) ∧ (a < 6 / 5) :=
sorry

end inequality_solution_empty_l501_501296


namespace simplify_expr_1_simplify_expr_2_l501_501929

theorem simplify_expr_1 : 
  (2 * 9 + 7 / 9) ^ (0.5) + 0.1 ^ (-2) + (2 * 27 + 10 / 27) ^ (-2 / 3) - 3 * (π^0) + 37 / 48 
  = 4813 / 48 :=
by sorry

theorem simplify_expr_2 : 
  (1 / 2) * log 10 (32 / 49) - (4 / 3) * log 10 (sqrt 8) + log 10 (sqrt 245) 
  = 1 / 2 :=
by sorry

end simplify_expr_1_simplify_expr_2_l501_501929


namespace exists_one_to_one_sigma_sum_l501_501018

variable {n : ℕ}

theorem exists_one_to_one_sigma_sum 
  (h_pos : 0 < n) :
  ∃ σ : fin n → fin n, function.injective σ ∧
    (∑ k, (k : ℕ) / ((k + σ k) : ℕ) ^ 2 : ℝ) < 1 / 2 :=
sorry

end exists_one_to_one_sigma_sum_l501_501018


namespace equidistant_x_coordinate_l501_501867

noncomputable def equidistant_point (x y : ℝ) : Prop :=
  (abs x = abs y) ∧ (abs x = abs ((2 * x + y - 4) / (real.sqrt 5)))

theorem equidistant_x_coordinate :
  ∃ y : ℝ, equidistant_point (4 / 3) y :=
sorry

end equidistant_x_coordinate_l501_501867


namespace hawkeye_remaining_money_l501_501271

-- Define the conditions
def cost_per_charge : ℝ := 3.5
def number_of_charges : ℕ := 4
def budget : ℝ := 20

-- Define the theorem to prove the remaining money
theorem hawkeye_remaining_money : 
  budget - (number_of_charges * cost_per_charge) = 6 := by
  sorry

end hawkeye_remaining_money_l501_501271


namespace jaden_toy_cars_l501_501654

variable (initial_cars bought_cars birthday_cars gave_sister gave_vinnie : ℕ)

theorem jaden_toy_cars :
  let final_cars := initial_cars + bought_cars + birthday_cars - gave_sister - gave_vinnie in
  initial_cars = 14 → bought_cars = 28 → birthday_cars = 12 → gave_sister = 8 → gave_vinnie = 3 →
  final_cars = 43 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end jaden_toy_cars_l501_501654


namespace find_p_of_abs_sum_roots_eq_five_l501_501920

theorem find_p_of_abs_sum_roots_eq_five (p : ℝ) : 
  (∃ x y : ℝ, x + y = -p ∧ x * y = -6 ∧ |x| + |y| = 5) → (p = 1 ∨ p = -1) := by
  sorry

end find_p_of_abs_sum_roots_eq_five_l501_501920


namespace center_to_line_distance_l501_501730

theorem center_to_line_distance :
  let center := (1 : Real, -2 : Real)
  let line := { x : Real × Real | x.1 - x.2 = 0 }
  let A := 1
  let B := -1
  let C := 0
  let (x1, y1) := center 
  Real.abs (A * x1 + B * y1 + C) / Real.sqrt (A^2 + B^2) = (3 * Real.sqrt 2) / 2 :=
by
  sorry

end center_to_line_distance_l501_501730


namespace absolute_difference_volumes_l501_501885

/-- The absolute difference in volumes of the cylindrical tubes formed by Amy and Carlos' papers. -/
theorem absolute_difference_volumes :
  let h_A := 12
  let C_A := 10
  let r_A := C_A / (2 * Real.pi)
  let V_A := Real.pi * r_A^2 * h_A
  let h_C := 8
  let C_C := 14
  let r_C := C_C / (2 * Real.pi)
  let V_C := Real.pi * r_C^2 * h_C
  abs (V_C - V_A) = 92 / Real.pi :=
by
  sorry

end absolute_difference_volumes_l501_501885


namespace min_value_eq_l501_501542

theorem min_value_eq:
  (∃ x y: ℝ, 2*x + 8*y = 3 ∧ (∀ (a b: ℝ), (2*a + 8*b = 3) → (a^2 + 4*b^2 - 2*a) ≥ -19/20)) :=
begin
  sorry
end

end min_value_eq_l501_501542


namespace log_eq_l501_501285

theorem log_eq {a b : ℝ} (h₁ : a = Real.log 256 / Real.log 4) (h₂ : b = Real.log 27 / Real.log 3) : 
  a = (4 / 3) * b :=
by
  sorry

end log_eq_l501_501285


namespace trajectory_eq_l501_501960

theorem trajectory_eq (M : ℝ × ℝ) (A : ℝ × ℝ) (l : ℝ) (hA : A = (2, 0)) (hl : l = 1 / 2) (h_ratio : ∀ x y, (M = (x, y) → sqrt ((x - 2) ^ 2 + y ^ 2) / abs (x - 1 / 2) = 2)) :
  ∀ x y, (M = (x, y) → x^2 - y^2 / 3 = 1) :=
by
  sorry

end trajectory_eq_l501_501960


namespace min_value_a4b3c2_l501_501670

theorem min_value_a4b3c2 {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 1/a + 1/b + 1/c = 9) :
  a ^ 4 * b ^ 3 * c ^ 2 ≥ 1 / 5184 := 
sorry

end min_value_a4b3c2_l501_501670


namespace overall_gain_percent_l501_501860

theorem overall_gain_percent
    (CP1 CP2 CP3 : ℕ) (SP1 SP2 SP3 : ℕ)
    (hCP1 : CP1 = 25) (hCP2 : CP2 = 35) (hCP3 : CP3 = 45)
    (hSP1 : SP1 = 31) (hSP2 : SP2 = 40) (hSP3 : SP3 = 54) :
    let CP := CP1 + CP2 + CP3,
        SP := SP1 + SP2 + SP3,
        Gain := SP - CP,
        GainPercent := (Gain / CP : ℚ) * 100 in
    GainPercent ≈ 19.05 := by 
  sorry

end overall_gain_percent_l501_501860


namespace balloons_difference_l501_501157

theorem balloons_difference :
  ∀ (JakeBalloons AllanBalloons : ℕ), 
  JakeBalloons = 11 ∧ AllanBalloons = 5 → JakeBalloons - AllanBalloons = 6 :=
by
  intros JakeBalloons AllanBalloons h
  obtain ⟨hJake, hAllan⟩ := h
  rw [hJake, hAllan]
  norm_num

end balloons_difference_l501_501157


namespace count_knights_l501_501032

noncomputable theory

def knights_and_liars (N : ℕ) : Prop :=
∃ (students : ℤ → Prop),  -- students are either knights or liars
  (∀ i : ℤ, 0 ≤ i → i < 2 * N → students i = (students i ↔ ¬ students (i + N % (2 * N)))) ∧  -- each student claims to be taller than the one opposite
  (∃ (num_knights : ℤ), num_knights = N)  -- number of knights is exactly N

theorem count_knights (N : ℕ) : (knights_and_liars N) → ∃ num_knights, num_knights = N :=
by
  sorry

end count_knights_l501_501032
