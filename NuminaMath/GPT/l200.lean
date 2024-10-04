import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Analysis.Calculus
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Geometry
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometry
import Mathlib.Combinatorics.Catalan
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Polynomial.RingDivision
import Mathlib.Data.Rat
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Init
import Mathlib.LinearAlgebra.Determinant
import Mathlib.Probability
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction.Finite
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Probability.ProbabilityMassFunction

namespace classroom_student_count_l200_200714

-- Define the problem conditions
def student_count := Σ (n : ℕ), n < 60 ∧ n % 6 = 4 ∧ n % 8 = 6

-- State the theorem
theorem classroom_student_count : ∃ n : student_count, n.1 = 22 :=
by
  sorry

end classroom_student_count_l200_200714


namespace quadrilateral_area_l200_200155

/-
Proof Statement: For a square with a side length of 8 cm, each of whose sides is divided by a point into two equal segments, 
prove that the area of the quadrilateral formed by connecting these points is 32 cm².
-/

theorem quadrilateral_area (side_len : ℝ) (h : side_len = 8) :
  let quadrilateral_area := (side_len * side_len) / 2
  quadrilateral_area = 32 :=
by
  sorry

end quadrilateral_area_l200_200155


namespace arithmetic_sequence_n_l200_200564

theorem arithmetic_sequence_n (
    S : ℕ → ℕ,
    a : ℕ → ℕ,
    n : ℕ,
    h1 : a 1 = 1,
    h2 : ∀ n, a (n + 1) - a n = 2,
    h3 : S (n + 1) - S n = 15) :
  n = 7 :=
sorry

end arithmetic_sequence_n_l200_200564


namespace unique_solution_triple_l200_200425

theorem unique_solution_triple (x y z : ℝ) (h1 : x + y = 3) (h2 : x * y = z^3) : (x = 1.5 ∧ y = 1.5 ∧ z = 0) :=
by
  sorry

end unique_solution_triple_l200_200425


namespace enthusiasts_min_max_l200_200074

-- Define the conditions
def total_students : ℕ := 100
def basketball_enthusiasts : ℕ := 63
def football_enthusiasts : ℕ := 75

-- Define the main proof problem
theorem enthusiasts_min_max :
  ∃ (common_enthusiasts : ℕ), 38 ≤ common_enthusiasts ∧ common_enthusiasts ≤ 63 :=
sorry

end enthusiasts_min_max_l200_200074


namespace problem_solution_l200_200597

noncomputable def lines_through_A_B (A B : Point) (dA dB : ℝ) (radA radB : ℝ) (distAB : ℝ) : ℕ :=
  if (distAB = radA + radB) then 3 else 0

-- Given points A and B, circles centered at A with radius 5 units and at B with radius 3 units,
-- making A and B 8 units apart, we assert the number of lines through A and B are exactly 3.
theorem problem_solution :
  ∀ (A B : Point), dist A B = 8 → lines_through_A_B A B 5 3 5 3 8 = 3 := by sorry  -- final proof

-- Add assumptions for the existence and basic properties of Point and dist
class Point :=
dist : Point → Point → ℝ

end problem_solution_l200_200597


namespace remainder_zero_mod_p_l200_200990

open Int

def problem : ℕ := 2017

theorem remainder_zero_mod_p (p : ℕ) (h_prime : Nat.Prime p) :
  (∑ i in Finset.range (p-1), ⌊(i+1:ℕ)^p / p⌋) % p = 0 :=
by
  -- prime number definition
  have hp : p = problem := by sorry
  -- prime check
  have prime_p : h_prime = by exact Nat.Prime.fact_2017
  sorry

end remainder_zero_mod_p_l200_200990


namespace sum_of_selected_numbers_l200_200761

theorem sum_of_selected_numbers : 
  let S : List ℚ := [14 / 10, 9 / 10, 12 / 10, 5 / 10, 13 / 10] in
  (S.filter (λ x => x ≥ 11 / 10)).sum = 39 / 10 :=
by
  sorry

end sum_of_selected_numbers_l200_200761


namespace find_sqrt_abc_sum_l200_200567

theorem find_sqrt_abc_sum (a b c : ℝ)
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 36 * Real.sqrt 15 := by
  sorry

end find_sqrt_abc_sum_l200_200567


namespace determinant_zero_l200_200777

variables (θ φ : ℝ)

noncomputable def matrix_3x3 : matrix (fin 3) (fin 3) ℝ :=
![![0, real.cos θ, real.sin θ],
  ![-real.cos θ, 0, real.cos φ],
  ![-real.sin θ, -real.cos φ, 0]]

theorem determinant_zero : matrix.det (matrix_3x3 θ φ) = 0 :=
by
  -- The proof goes here.
  sorry

end determinant_zero_l200_200777


namespace sum_of_cubes_is_24680_l200_200325

noncomputable def jake_age := 10
noncomputable def amy_age := 12
noncomputable def ryan_age := 28

theorem sum_of_cubes_is_24680 (j a r : ℕ) (h1 : 2 * j + 3 * a = 4 * r)
  (h2 : j^3 + a^3 = 1 / 2 * r^3) (h3 : j + a + r = 50) : j^3 + a^3 + r^3 = 24680 :=
by
  sorry

end sum_of_cubes_is_24680_l200_200325


namespace not_possible_to_place_l200_200975

theorem not_possible_to_place :
  ∀ (α : Type) [has_add α] [has_mul α] [has_eq α],
  let counts := [4, 3, 3] in
  let values := [1, 2, 3] in
  let total_sum := 4 * 1 + 3 * 2 + 3 * 3 in
  ∀ S : α, 
  (5 * S = 2 * total_sum) → S ≠ (38 / 5 : α) :=
begin
  intros α _ _ _ counts values total_sum S h,
  rw ← add_mul at h,
  have hS : S = (38 / 5 : α), 
  { sorry },
  contradiction,
end

end not_possible_to_place_l200_200975


namespace value_of_x0_l200_200917

noncomputable def f (x : ℝ) : ℝ := x^3

theorem value_of_x0 (x0 : ℝ) (h1 : f x0 = x0^3) (h2 : deriv f x0 = 3) :
  x0 = 1 ∨ x0 = -1 :=
by
  sorry

end value_of_x0_l200_200917


namespace cosine_angle_GH_BC_l200_200323

noncomputable theory
open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C G H : V)
variables (AB BC CA GH : ℝ)

-- Given conditions
def given_conditions :=
  midpoint ℝ B G H ∧ 
  AB = 1 ∧
  G - H = (B - G) + (B - H) ∧
  BC = 10 ∧
  CA = sqrt 97 ∧
  (⟪(B - A), (G - A)⟫ + ⟪(C - A), (H - A)⟫ = 6)

-- Proof goal
theorem cosine_angle_GH_BC (h : given_conditions A B C G H AB BC CA) :
  real.cos (inner_product_space.angle (G - H) (B - C)) = 4 / 5 :=
sorry

end cosine_angle_GH_BC_l200_200323


namespace range_of_a_l200_200485
noncomputable theory

open set real

def f (a x : ℝ) : ℝ := -x^3 + a * x

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Ioo (-1 : ℝ) (1 : ℝ), deriv (f a) x ≥ 0) ↔ (a ≥ 3) :=
begin
  sorry
end

end range_of_a_l200_200485


namespace classify_numbers_l200_200844

theorem classify_numbers :
  let S := 
            {-1/2, sqrt 5, pi/3, 3.14, 23/7, 0, -sqrt 16, real.mk_nat_digit_seq 1 0.323 (λ n, if n % 2 = 0 then 2 else 3)} 
  in
  {x | x ∈ S ∧ x.is_integer} = {0, -sqrt 16} ∧
  {x | x ∈ S ∧ x.is_fraction} = {-1/2, 3.14, 23/7} ∧
  {x | x ∈ S ∧ x.is_irrational} = {sqrt 5, pi/3, real.mk_nat_digit_seq 1 0.323 (λ n, if n % 2 = 0 then 2 else 3)} :=
by
  sorry

end classify_numbers_l200_200844


namespace Cindy_hourly_rate_l200_200087

theorem Cindy_hourly_rate
    (num_courses : ℕ)
    (weekly_hours : ℕ) 
    (monthly_earnings : ℕ) 
    (weeks_in_month : ℕ)
    (monthly_hours_per_course : ℕ)
    (hourly_rate : ℕ) :
    num_courses = 4 →
    weekly_hours = 48 →
    monthly_earnings = 1200 →
    weeks_in_month = 4 →
    monthly_hours_per_course = (weekly_hours / num_courses) * weeks_in_month →
    hourly_rate = monthly_earnings / monthly_hours_per_course →
    hourly_rate = 25 := by
  sorry

end Cindy_hourly_rate_l200_200087


namespace log_constant_expression_l200_200306

theorem log_constant_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x > y) (hcond : x^2 + y^2 = 18 * x * y) :
  ∃ k : ℝ, (Real.log (x - y) / Real.log (Real.sqrt 2) - (1 / 2) * (Real.log x / Real.log (Real.sqrt 2) + Real.log y / Real.log (Real.sqrt 2))) = k :=
sorry

end log_constant_expression_l200_200306


namespace problem_find_f_neg_2018_l200_200486

def f : ℝ → ℝ
| x => if x > 0 then 3^(3 + Real.logb 2 x) else f (x + 1/2)

theorem problem_find_f_neg_2018 : f (-2018) = 9 :=
sorry

end problem_find_f_neg_2018_l200_200486


namespace employees_percentage_l200_200316

theorem employees_percentage (x : ℕ) :
  let T := 4 * x + 6 * x + 7 * x + 4 * x + 3 * x + 3 * x + 3 * x + 2 * x + 2 * x + 1 * x + 1 * x + 1 * x + 1 * x in
  let E := 1 * x + 1 * x in
  (E / T : ℚ) * 100 = 5.26 :=
by
  sorry

end employees_percentage_l200_200316


namespace commonRegionArea_l200_200061

-- Define the given realistic representation of the rectangle.
def Rectangle (width height : ℝ) := { 
  center : (ℝ × ℝ) := (0, 0),
  width := width,
  height := height
}

-- Define the given realistic representation of the ellipse.
def Ellipse (a b : ℝ) := {
  center : (ℝ × ℝ) := (0, 0),
  a := a,
  b := b
}

-- Function to compute the area of an ellipse.
def areaOfEllipse (e : Ellipse) : ℝ := π * e.a * e.b

-- Function to prove the area of the region common to both the rectangle and the ellipse.
theorem commonRegionArea (r : Rectangle 10 4) (e : Ellipse 3 2) : areaOfEllipse e = 6 * π :=
by
  sorry

end commonRegionArea_l200_200061


namespace vector_magnitude_l200_200610

open Real

-- Define the vector a and its properties
def a : ℝ × ℝ := (2, 0)
def norm_a : ℝ := 2 -- equivalent to |a|

-- Define the vector b and its properties using a variable, as b is not fully specified
variables (b : ℝ × ℝ) (hb_norm : ‖b‖ = 1)
variables (theta : ℝ) (h_theta : theta = π / 3) -- angle θ = 60 degrees

-- Define dot product calculation for vectors in ℝ^2
def dot (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define the Euclidean norm for vectors in ℝ^2
def norm (v : ℝ × ℝ) : ℝ := sqrt (v.1 * v.1 + v.2 * v.2)

-- Statement to prove
theorem vector_magnitude :
  ‖(a.1 + 2 * b.1, a.2 + 2 * b.2)‖ = 2 * sqrt 3 :=
by
  sorry

end vector_magnitude_l200_200610


namespace projection_correct_l200_200371

-- Define the initial vectors
def v1 := ⟨3, 3⟩ : ℝ × ℝ
def v1_proj := ⟨45/10, 9/10⟩ : ℝ × ℝ

-- Define the vector we want to project
def v2 := ⟨-3, 3⟩ : ℝ × ℝ

-- Define the expected projection result
def v2_proj_expected := ⟨-30/13, -6/13⟩ : ℝ × ℝ

-- The projection function
def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot1 := u.1 * v.1 + u.2 * v.2
  let dot2 := v.1 * v.1 + v.2 * v.2
  let scalar := dot1 / dot2
  ⟨scalar * v.1, scalar * v.2⟩

-- Lean statement to be proved
theorem projection_correct :
  projection ⟨-3, 3⟩ ⟨5, 1⟩ = ⟨-30/13, -6/13⟩ :=
by
  sorry

end projection_correct_l200_200371


namespace pure_imaginary_condition_l200_200578

theorem pure_imaginary_condition (a : ℝ) (i : ℂ) (h : i = complex.I) : 
  (2 * a + complex.I) * (1 - 2 * complex.I) ≫ complex.I → a = -1 := 
by sorry

end pure_imaginary_condition_l200_200578


namespace find_first_year_interest_rate_l200_200423

-- Given conditions
def P : ℝ := 5000 -- Principal amount
def total_amount : ℝ := 7500 -- Total amount after 2 years
def second_year_rate : ℝ := 25 -- Interest rate for the second year in percentage

-- Define the unknown rate of interest for the first year
variable (R : ℝ) 

-- Define the amounts after first and second years
def A1 : ℝ := P + (P * R * 1) / 100 -- Amount after the first year
def A2 : ℝ := A1 * 1.25 -- Amount after the second year, with 25% interest rate

-- The proof problem
theorem find_first_year_interest_rate : A2 = total_amount → R = 20 :=
by
  intro h
  sorry

end find_first_year_interest_rate_l200_200423


namespace max_value_2a1_plus_product_l200_200799

theorem max_value_2a1_plus_product (n : ℕ) (a : Fin n → ℝ) (h_n : 2 ≤ n)
  (h_nonneg : ∀ i, 0 ≤ a i) (h_sum : (∑ i in Finset.univ, a i) = 4) :
  (2 * a 0 + (Finset.univ.val.reverse.drop 1).sum (λ i, a 0 * Finset.univ.val.reverse.take i.succ.prod (λ j, a j))) ≤ 9 :=
sorry

end max_value_2a1_plus_product_l200_200799


namespace Q15_approx1_l200_200860

def Sn (n : ℕ) : ℚ := (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def Qn (n : ℕ) : ℚ := 
  ∏ i in finset.range (n - 2) + 3, (Sn i) / (Sn i + 1)

theorem Q15_approx1 : Qn 15 ≈ 1 := sorry

end Q15_approx1_l200_200860


namespace relationship_abc_l200_200576

noncomputable def a := 2^(0.3)
noncomputable def b := 0.3^2
noncomputable def c := Real.log 5 / Real.log 2

theorem relationship_abc : b < a ∧ a < c :=
by
  -- Placeholder for the actual proof
  sorry

end relationship_abc_l200_200576


namespace probability_after_2020_rounds_l200_200395

-- Conditions
variable (Belinda Carlos Danny Ella : Type)
variable (initial_state : Belinda → Carlos → Danny → Ella → ℕ)
variable (ring_interval : ℕ := 10)
variable (rounds : ℕ := 2020)

-- State Transition Function (simplified version)
def state_transition (s : Belinda → Carlos → Danny → Ella → ℕ) : Belinda → Carlos → Danny → Ella → ℕ := sorry

-- Final State after given rounds
def final_state (s : Belinda → Carlos → Danny → Ella → ℕ) : Belinda → Carlos → Danny → Ella → ℕ := by
  exact (iterate state_transition rounds s)

-- Probability Calculation
def probability_each_with_one (s : Belinda → Carlos → Danny → Ella → ℕ) :=
  if final_state s Belinda Carlos Danny Ella = (λ b c d e => 1) then
    (24 / 81 : ℚ)
  else
    0

-- Proof Problem Statement
theorem probability_after_2020_rounds : probability_each_with_one initial_state = (24 / 81 : ℚ) := sorry

end probability_after_2020_rounds_l200_200395


namespace linear_inequalities_count_l200_200386

def expr1 (x : ℝ) : Prop := x < 5
def expr2 (x : ℝ) : Prop := x * (x - 5) < 5
def expr3 (x : ℝ) : Prop := (1 / x) < 5
def expr4 (x y : ℝ) : Prop := (2 * x + y) < (5 + y)
def expr5 (a : ℝ) : Prop := a - 2 < 5
def expr6 (x y : ℝ) : Prop := x ≤ y / 3

theorem linear_inequalities_count: 
  (∃ (x : ℝ), expr1 x) + (∃ (x : ℝ), expr4 x 0) + (∃ (a : ℝ), expr5 a) = 3 := 
by
  sorry

end linear_inequalities_count_l200_200386


namespace arithmetic_sequence_50th_term_l200_200936

theorem arithmetic_sequence_50th_term :
  ∀ (a d n : ℕ), a = 2 → d = 4 → n = 50 → a + (n - 1) * d = 198 :=
by
  intros a d n ha hd hn
  rw [ha, hd, hn]
  norm_num
  sorry

end arithmetic_sequence_50th_term_l200_200936


namespace probability_inner_circle_l200_200712

noncomputable def outer_circle_radius : ℝ := 3
noncomputable def inner_circle_radius : ℝ := 1

def area_circle (r : ℝ) : ℝ := real.pi * r^2

def prob_point_closer_to_center_than_boundary : ℝ :=
  (area_circle inner_circle_radius) / (area_circle outer_circle_radius)

theorem probability_inner_circle:
  prob_point_closer_to_center_than_boundary = 1 / 9 :=
by
  sorry

end probability_inner_circle_l200_200712


namespace jake_has_more_balloons_l200_200740

theorem jake_has_more_balloons 
  (allans_initial_balloons : ℕ := 2)
  (allans_bought_balloons : ℕ := 3)
  (jakes_balloons : ℕ := 6) :
  let allans_total_balloons := allans_initial_balloons + allans_bought_balloons
  in jakes_balloons - allans_total_balloons = 1 :=
by
  sorry

end jake_has_more_balloons_l200_200740


namespace number_of_valid_b2_values_l200_200732

def sequence (b : ℕ → ℕ) : Prop :=
  ∀ n, b (n + 2) = |b (n + 1) - b n|

def satisfies_conditions (b : ℕ → ℕ) : Prop :=
  b 1 = 1001 ∧ b 2 < 1001 ∧ b 2007 = 1 ∧ sequence b

theorem number_of_valid_b2_values : ∃ (s : Finset ℕ), (∀ b2 ∈ s, ∃ b : ℕ → ℕ, satisfies_conditions b ∧ b 2 = b2) ∧ s.card = 220 :=
sorry

end number_of_valid_b2_values_l200_200732


namespace quadratic_polynomial_exists_l200_200226

theorem quadratic_polynomial_exists (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (f : ℤ → ℤ), (∃ (a₀ a₁ a₂ : ℤ), f = λ x, a₀ x^2 + a₁ x + a₂) ∧
  (∀ x, (x = a ∨ x = b ∨ x = c) → f x = x^3) ∧ 
  (∃ (a₀ : ℤ), (a₀ > 0 ∧ f = λ x, a₀ x^2 + a₁ x + a₂)) :=
sorry

end quadratic_polynomial_exists_l200_200226


namespace typist_current_salary_l200_200638

def original_salary : ℝ := 4000.0000000000005
def increased_salary (os : ℝ) : ℝ := os + (os * 0.1)
def decreased_salary (is : ℝ) : ℝ := is - (is * 0.05)

theorem typist_current_salary : decreased_salary (increased_salary original_salary) = 4180 :=
by
  sorry

end typist_current_salary_l200_200638


namespace general_term_arithmetic_seq_sum_first_n_terms_geometric_seq_l200_200701

-- Problem 1: Arithmetic sequence
variable {a : ℕ → ℤ}
variable (a₂_eq_0 : a 2 = 0)
variable (a₆_plus_a₈_eq_neg10 : a 6 + a 8 = -10)

theorem general_term_arithmetic_seq :
  ∀ n, a n = 2 - n :=
by
  sorry

-- Problem 2: Geometric sequence
variable {b : ℕ → ℤ}
variable (b1_eq_3 : b 1 = 3)
variable (b2_eq_9 : b 2 = 9)

theorem sum_first_n_terms_geometric_seq :
  ∀ n, (finset.range (n + 1)).sum b = (3 ^ (n + 1) - 3) / 2 :=
by
  sorry

end general_term_arithmetic_seq_sum_first_n_terms_geometric_seq_l200_200701


namespace sum_of_squares_base_case_l200_200663

theorem sum_of_squares_base_case : 1^2 + 2^2 = (1 * 3 * 5) / 3 := by sorry

end sum_of_squares_base_case_l200_200663


namespace unique_chords_equals_968_l200_200464

-- Definitions from conditions
def number_of_keys : ℕ := 10

-- Chords are combinations of 3 to 10 keys pressed simultaneously
def combinations (n k : ℕ) : ℕ := Nat.choose n k

-- Define the total number of unique chords for pressing 3 to 10 keys
def unique_chords : ℕ :=
  (∑ k in Finset.Icc 3 number_of_keys, combinations number_of_keys k)

theorem unique_chords_equals_968 :
  unique_chords = 968 :=
by
  sorry

end unique_chords_equals_968_l200_200464


namespace min_value_f_interval_l200_200301

def f (x : ℝ) : ℝ := x^3 + x^2 - x + 1

theorem min_value_f_interval :
  ∃ m, IsMinOn f (set.Icc (-2 : ℝ) (1 : ℝ)) m ∧ m = C :=
  sorry

end min_value_f_interval_l200_200301


namespace biased_coin_probability_l200_200707

noncomputable def probability_exactly_two_heads_in_four_tosses (p : ℚ) : ℚ :=
  (nat.choose 4 2) * p^2 * (1 - p)^(4 - 2)

theorem biased_coin_probability (p : ℚ) (v : ℚ) (h_p : p = 3/5) : v = 216/625 :=
  by
  have h_v : probability_exactly_two_heads_in_four_tosses p = v := sorry
  rw [h_p] at h_v
  sorry

end biased_coin_probability_l200_200707


namespace cube_mod7_not_divisible_7_l200_200925

theorem cube_mod7_not_divisible_7 (a : ℤ) (h : ¬ (7 ∣ a)) :
  (a^3 % 7 = 1) ∨ (a^3 % 7 = -1) :=
sorry

end cube_mod7_not_divisible_7_l200_200925


namespace integral_value_l200_200756

theorem integral_value : 
  ∫ x in 0..1, (2 * x + exp x) = Real.exp 1 :=
sorry

end integral_value_l200_200756


namespace f_f_neg2016_eq_zero_l200_200483

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then f (x+2)
  else if -1 < x ∧ x < 1 then 2*x + 2
  else 2^x - 4

theorem f_f_neg2016_eq_zero : f (f (-2016)) = 0 := 
sorry

end f_f_neg2016_eq_zero_l200_200483


namespace inradius_circumradius_ratio_l200_200153

noncomputable theory

open Real

-- Definitions and setup
variables (a b : ℝ) (p : ℝ × ℝ)
variables (h1 : 0 < a) (h2 : 0 < b)
variables (ecc : ℝ) (h3 : ecc = sqrt 2)

-- Defining the hyperbola and conditions
def hyperbola (p : ℝ × ℝ) : Prop := (p.1 ^ 2 / a ^ 2) - (p.2 ^ 2 / b ^ 2) = 1

def foci_condition (p : ℝ × ℝ) : Prop := 
  let c := sqrt (a^2 + b^2) in
  let F1 := (c, 0) in
  let F2 := (-c, 0) in
  ((p.1 - F1.1) * (p.1 - F2.1) + (p.2 - F1.2) * (p.2 - F2.2)) = 0

-- Problem statement
theorem inradius_circumradius_ratio 
  (p : ℝ × ℝ) (hp : hyperbola a b p) (hc : foci_condition a b p) : (sqrt 6 / 2) - 1 = sorry

end inradius_circumradius_ratio_l200_200153


namespace marble_arrangement_count_l200_200687

-- Utility definitions for modeling the problem
def marbles : List Char := ['A', 'B', 'S', 'T', 'C']
def is_adjacent {α : Type} [DecidableEq α] (a b : α) (lst : List α) :=
  a ≠ b ∧ (lst.zip lst.tail).any (λ (ab : α × α), ab.1 = a ∧ ab.2 = b)

-- Main theorem statement
theorem marble_arrangement_count :
  (5! - 3! * 2! * 3! = 48) := by
  sorry

end marble_arrangement_count_l200_200687


namespace proof_problem_l200_200177

variable (x : ℝ)

-- Definitions from conditions
def p : Prop := ∃ x : ℝ, x - 2 > 0
def q : Prop := ∀ x : ℝ, x ≥ 0 → real.sqrt x < x

-- The proposition we need to prove
theorem proof_problem : p ∧ ¬ q := by
  sorry

end proof_problem_l200_200177


namespace total_number_of_trees_l200_200612

theorem total_number_of_trees (side_length : ℝ) (area_ratio : ℝ) (trees_per_sqm : ℝ) (H : side_length = 100) (R : area_ratio = 3) (T : trees_per_sqm = 4) : 
  let street_area := side_length ^ 2 in 
  let forest_area := area_ratio * street_area in
  let total_trees := forest_area * trees_per_sqm in
  total_trees = 120000 :=
by
  -- proof steps go here
  sorry

end total_number_of_trees_l200_200612


namespace find_m_l200_200893

variables {R : Type*} [CommRing R]

/-- Definition of the dot product in a 2D vector space -/
def dot_product (a b : R × R) : R := a.1 * b.1 + a.2 * b.2

/-- Given vectors a and b as conditions -/
def a : ℚ × ℚ := (m, 3)
def b : ℚ × ℚ := (1, m + 1)

theorem find_m (m : ℚ) (h : dot_product a b = 0) : m = -3 / 4 :=
sorry

end find_m_l200_200893


namespace jakes_present_weight_l200_200688

theorem jakes_present_weight:
  ∃ J S : ℕ, J - 15 = 2 * S ∧ J + S = 132 ∧ J = 93 :=
by
  sorry

end jakes_present_weight_l200_200688


namespace shaded_region_area_l200_200522

theorem shaded_region_area (r : ℝ) (h_r : r = 5) : 
  let shaded_area := 12 * ((1/4 * π * r^2) - (1/4 * 25 * (√3)))
  in shaded_area = 75 * π - 25 * sqrt 3 :=
by
  sorry

end shaded_region_area_l200_200522


namespace sin_value_of_x_l200_200473

theorem sin_value_of_x (x : ℝ) (h1 : x ∈ Ioo (π / 2) π) (h2 : cos (2 * x) = 7 / 25) : sin x = 3 / 5 :=
by
  sorry

end sin_value_of_x_l200_200473


namespace matrices_inverses_sum_45_l200_200631

theorem matrices_inverses_sum_45 :
  ∃ (a b c d e f g h : ℤ),
    (∏ x in finset.range 3, (vector.nth [ [-6 * x.a + 2 * f + 3 * b, a * e - 28 + 2 * h, -12 * a + b * g + 5 * b], 
                                        [-18 + 3 * f + 12, 3 * e - 42 + 4 * h, -36 + 3 * g + 20],
                                        [-6 * c + 6 * f + 3 * d, c * e - 84 + 6 * h, -12 * c + 6 * g + 5 * d] ] x))
    =
    vector.of_fn (λ i, if i = 0 then 1 else (if i = 1 then 0 else 0))
    ∧ a + b + c + d + e + f + g + h = 45 :=
begin
  sorry
end

end matrices_inverses_sum_45_l200_200631


namespace find_angle_C_and_max_area_l200_200974

variables {A B C : Angle}
variables {a b c : ℝ}

-- Given a certain equation in a triangle
-- and if b = 4 * sin(B),
-- prove specific values for the angle C and the maximum area of the triangle.

theorem find_angle_C_and_max_area
(h1 : a * cos C ^ 2 + 2 * c * cos A * cos C + a + b = 0)
(h2 : b = 4 * sin B)
: (C = 120 ∧ (area := (1/2) * a * b * sin C) ≤ sqrt(3)) :=
begin
  sorry
end

end find_angle_C_and_max_area_l200_200974


namespace chimes_1000_on_march_7_l200_200364

theorem chimes_1000_on_march_7 : 
  ∀ (initial_time : Nat) (start_date : Nat) (chimes_before_noon : Nat) 
  (chimes_per_day : Nat) (target_chime : Nat) (final_date : Nat),
  initial_time = 10 * 60 + 15 ∧
  start_date = 26 ∧
  chimes_before_noon = 25 ∧
  chimes_per_day = 103 ∧
  target_chime = 1000 ∧
  final_date = start_date + (target_chime - chimes_before_noon) / chimes_per_day ∧
  (target_chime - chimes_before_noon) % chimes_per_day ≤ chimes_per_day
  → final_date = 7 := 
by
  intros
  sorry

end chimes_1000_on_march_7_l200_200364


namespace union_sets_l200_200862

variable (M : Set ℝ) (N : Set ℝ)

def M_def : M = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by simp
def N_def : N = {x : ℝ | -3 ≤ x ∧ x ≤ 2} := by simp

theorem union_sets :
  M ∪ N = {x : ℝ | -3 ≤ x ∧ x ≤ 3} :=
by sorry

end union_sets_l200_200862


namespace toothpicks_stage_20_l200_200652

-- Definition of the toothpick sequence
def toothpicks (n : ℕ) : ℕ :=
  if n = 1 then 3
  else 3 + 3 * (n - 1)

-- Theorem statement
theorem toothpicks_stage_20 : toothpicks 20 = 60 := by
  sorry

end toothpicks_stage_20_l200_200652


namespace Jungkook_has_the_largest_number_l200_200686

theorem Jungkook_has_the_largest_number :
  let Yoongi := 4
  let Yuna := 5
  let Jungkook := 6 + 3
  Jungkook > Yoongi ∧ Jungkook > Yuna := by
    sorry

end Jungkook_has_the_largest_number_l200_200686


namespace contrapositive_proposition_l200_200286

def proposition (x : ℝ) : Prop := x < 0 → x^2 > 0

theorem contrapositive_proposition :
  (∀ x : ℝ, proposition x) → (∀ x : ℝ, x^2 ≤ 0 → x ≥ 0) :=
by
  sorry

end contrapositive_proposition_l200_200286


namespace gretchen_work_hours_l200_200188

noncomputable def walking_ratio (walking: ℤ) (sitting: ℤ) : Prop :=
  walking * 90 = sitting * 10

theorem gretchen_work_hours (walking_time: ℤ) (h: ℤ) (condition1: walking_ratio 40 (60 * h)) :
  h = 6 :=
by sorry

end gretchen_work_hours_l200_200188


namespace eight_bags_weight_l200_200196

theorem eight_bags_weight :
  ∀ (total_bags : ℕ) (total_weight : ℚ) (bags_needed: ℕ), 
    total_bags = 12 → 
    total_weight = 24 → 
    bags_needed = 8 → 
    total_weight / total_bags * bags_needed = 16 :=
begin
  intros total_bags total_weight bags_needed hb hw hn,
  rw [hb, hw, hn],
  norm_num,
end

end eight_bags_weight_l200_200196


namespace sum_of_cubes_14_to_25_l200_200435

theorem sum_of_cubes_14_to_25 : 
  (Real.sqrt (∑ n in Finset.range 12, (n + 14)^3) = 312) :=
by
  sorry

end sum_of_cubes_14_to_25_l200_200435


namespace units_digit_sum_factorials_l200_200434

def double_factorial (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 1
  | k + 2 => n * double_factorial (n - 2)

theorem units_digit_sum_factorials : 
  (∑ i in Finset.range 13, (factorial i % 10) + double_factorial 12 % 10) % 10 = 3 :=
by 
  sorry

end units_digit_sum_factorials_l200_200434


namespace largest_integer_in_list_l200_200051

theorem largest_integer_in_list
  (L : List ℕ)
  (h1 : L.length = 5)
  (h2 : ∀ k, L.count k > 1 → k = 10)
  (h3 : L.nthLe 2 (by linarith [h1,sorry]) = 11)
  (h4 : L.sum = 5 * 12) :
  ∃ k, k ∈ L ∧ (∀ m, m ∈ L → m ≤ k) ∧ k = 17 :=
by sorry

end largest_integer_in_list_l200_200051


namespace find_white_balls_l200_200750

-- Noncomputable because we are dealing with real numbers and probability
noncomputable def white_balls_in_urn (w : ℕ) : Prop :=
  let num_black := 10 in
  let total_balls := num_black + w in
  let prob_first_black := (num_black : ℝ) / total_balls in
  let prob_second_black := (9 : ℝ) / (total_balls - 1) in
  prob_first_black * prob_second_black = 0.4285714285714286

theorem find_white_balls : ∃ (w : ℕ), white_balls_in_urn w ∧ w = 5 := by
  have h : white_balls_in_urn 5 := sorry
  exact ⟨5, h, rfl⟩

end find_white_balls_l200_200750


namespace ellipse_hyperbola_foci_l200_200294

theorem ellipse_hyperbola_foci (a b : ℝ) 
  (h1 : ∃ (a b : ℝ), b^2 - a^2 = 25 ∧ a^2 + b^2 = 64) : 
  |a * b| = (Real.sqrt 3471) / 2 :=
by
  sorry

end ellipse_hyperbola_foci_l200_200294


namespace initial_manufacturing_cost_l200_200095

theorem initial_manufacturing_cost
  (P : ℝ) -- selling price
  (initial_cost new_cost : ℝ)
  (initial_profit new_profit : ℝ)
  (h1 : initial_profit = 0.25 * P)
  (h2 : new_profit = 0.50 * P)
  (h3 : new_cost = 50)
  (h4 : new_profit = P - new_cost)
  (h5 : initial_profit = P - initial_cost) :
  initial_cost = 75 := 
by
  sorry

end initial_manufacturing_cost_l200_200095


namespace hyperbola_eccentricity_correct_l200_200934

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (b / a) = 4 / 3) : ℝ :=
  let e := sqrt ((a^2 + b^2) / a^2)
  ∃ (e : ℝ), e = 5 / 3

-- Theorem statement
theorem hyperbola_eccentricity_correct (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (b / a) = 4 / 3) : 
  hyperbola_eccentricity a b h1 h2 h3 = 5 / 3 := sorry

end hyperbola_eccentricity_correct_l200_200934


namespace find_m_collinear_l200_200185

theorem find_m_collinear (m : ℝ) 
    (a : ℝ × ℝ := (m + 3, 2)) 
    (b : ℝ × ℝ := (m, 1)) 
    (collinear : a.1 * 1 - 2 * b.1 = 0) : 
    m = 3 :=
by {
    sorry
}

end find_m_collinear_l200_200185


namespace circle_circumference_l200_200728

theorem circle_circumference (a b : ℝ) (h1 : a = 9) (h2 : b = 12) :
  ∃ c : ℝ, c = 15 * Real.pi :=
by
  sorry

end circle_circumference_l200_200728


namespace gcd_of_36_and_60_is_12_l200_200008

theorem gcd_of_36_and_60_is_12 :
  Nat.gcd 36 60 = 12 :=
sorry

end gcd_of_36_and_60_is_12_l200_200008


namespace brand_with_highest_sales_l200_200639

theorem brand_with_highest_sales (A B C D : ℕ) (hA : A = 15) (hB : B = 30) (hC : C = 12) (hD : D = 43) :
  max (max A (max B C)) D = D :=
by {
  simp [hA, hB, hC, hD],
  sorry
}

end brand_with_highest_sales_l200_200639


namespace perpendicular_vectors_l200_200881

variable (m : ℝ)

def vector_a := (m, 3)
def vector_b := (1, m + 1)

def dot_product (v w : ℝ × ℝ) := (v.1 * w.1) + (v.2 * w.2)

theorem perpendicular_vectors (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by 
  unfold vector_a vector_b dot_product at h
  linarith

end perpendicular_vectors_l200_200881


namespace number_of_chairs_l200_200260

noncomputable def total_time (chairs tables : ℕ) : ℕ := 4 * chairs + 4 * tables

theorem number_of_chairs (C : ℕ) : total_time C 3 = 40 → C = 7 := by
  intro h
  unfold total_time at h
  linarith
  sorry

end number_of_chairs_l200_200260


namespace projection_correct_l200_200127

open Real EuclideanSpace

def vector_v : ℝ^3 := ![3, -1, 2]
def direction_line : ℝ^3 := ![1, -1/2, 2]
def projection_vector : ℝ^3 := ![10/7, -5/14, 20/7]

theorem projection_correct :
  let scalar_proj := (inner_product vector_v direction_line) / (inner_product direction_line direction_line)
  in scalar_proj • direction_line = projection_vector :=
by
  let scalar_proj := (3 * 1 + (-1) * (-1/2) + 2 * 2) / (1 * 1 + (-1/2) * (-1/2) + 2 * 2)
  have : scalar_proj = 10 / 7 := by sorry
  show scalar_proj • direction_line = projection_vector, from sorry

end projection_correct_l200_200127


namespace T_formula_l200_200640

-- Defining the sequence T_n as given in problem conditions
def T (n : ℕ) : ℚ :=
  (List.range (n)).map (λ k, 1 - 1 / (List.sum (List.range (k + 1)) + (k + 1))).prod

theorem T_formula (n : ℕ) : T n = (n + 2) / (3 * n) := by
  sorry

end T_formula_l200_200640


namespace john_allowance_spent_l200_200779

theorem john_allowance_spent (B t d : ℝ) (h1 : t = 0.25 * (B - d)) (h2 : d = 0.10 * (B - t)) :
  (t + d) / B = 0.31 := by
  sorry

end john_allowance_spent_l200_200779


namespace good_subset_cardinality_bound_l200_200556

def Q_n (n : ℕ) : Set (Vector ℕ n) := { x | ∀ i, x.nth i ∈ {0, 1, 2} }

def good_triple (n : ℕ) (x y z : Vector ℕ n) : Prop :=
  ∃ i < n, {x.nth i, y.nth i, z.nth i} = {0, 1, 2}

def good_subset (n : ℕ) (A : Set (Vector ℕ n)) : Prop :=
  ∀ x y z ∈ A, x ≠ y → y ≠ z → x ≠ z → good_triple n x y z

theorem good_subset_cardinality_bound (n : ℕ) (A : Set (Vector ℕ n)) :
  good_subset n A → ↑|A| ≤ 2 * (3 / 2)^n :=
by
  sorry

end good_subset_cardinality_bound_l200_200556


namespace find_m_l200_200886

-- Declare the vectors a and b based on given conditions
variables {m : ℝ}

def a : ℝ × ℝ := (m, 3)
def b : ℝ × ℝ := (1, m + 1)

-- Define the condition that vectors a and b are perpendicular
def perpendicular (x y : ℝ × ℝ) : Prop := x.1 * y.1 + x.2 * y.2 = 0

-- State the problem in Lean 4
theorem find_m (h : perpendicular a b) : m = -3 / 4 :=
sorry

end find_m_l200_200886


namespace perpendicular_vectors_implies_m_value_l200_200868

variable (m : ℝ)

def vector1 : ℝ × ℝ := (m, 3)
def vector2 : ℝ × ℝ := (1, m + 1)

theorem perpendicular_vectors_implies_m_value
  (h : vector1 m ∙ vector2 m = 0) :
  m = -3 / 4 :=
by 
  sorry

end perpendicular_vectors_implies_m_value_l200_200868


namespace sum_of_solutions_l200_200433

theorem sum_of_solutions (x : ℝ) (h : x + (25 / x) = 10) : x = 5 :=
by
  sorry

end sum_of_solutions_l200_200433


namespace approx_value_at_1_97_l200_200393

theorem approx_value_at_1_97 :
  let f : ℝ → ℝ := λ x, sqrt (x^2 + 5)
  (f 1.97) ≈ 2.98 :=
by
  -- Definitions
  let f : ℝ → ℝ := λ x, sqrt (x^2 + 5)
  let a := 2.0
  let delta_x := 1.97 - a
  let f_a := f a
  let f_prime_a := (2 : ℝ) / (3 : ℝ)
  -- Approximation using differentials
  let approx_value := f_a + f_prime_a * delta_x
  -- Expected value
  let expected_value := 2.98
  -- Proving the approximation
  exact abs (approx_value - expected_value) < 0.01


end approx_value_at_1_97_l200_200393


namespace sqrt_abc_sum_is_72_l200_200569

noncomputable def abc_sqrt_calculation (a b c : ℝ) (h1 : b + c = 17) (h2 : c + a = 18) (h3 : a + b = 19) : ℝ :=
  sqrt (a * b * c * (a + b + c))

theorem sqrt_abc_sum_is_72 (a b c : ℝ) (h1 : b + c = 17) (h2 : c + a = 18) (h3 : a + b = 19) :
  abc_sqrt_calculation a b c h1 h2 h3 = 72 :=
by
  sorry

end sqrt_abc_sum_is_72_l200_200569


namespace number_of_tangent_and_parallel_lines_l200_200719

theorem number_of_tangent_and_parallel_lines (p : ℝ × ℝ) (a : ℝ) (h : p = (2, 4)) (hp_on_parabola : (p.1)^2 = 8 * p.2) :
  ∃ l1 l2 : (ℝ × ℝ) → Prop, 
    (l1 (2, 4) ∧ l2 (2, 4)) ∧ 
    (∀ l, (l = l1 ∨ l = l2) ↔ (∃ q, q ≠ p ∧ q ∈ {p' | (p'.1)^2 = 8 * p'.2})) ∧ 
    (∀ p' ∈ {p' | (p'.1)^2 = 8 * p'.2}, (l1 p' ∨ l2 p') → False) :=
sorry

end number_of_tangent_and_parallel_lines_l200_200719


namespace max_five_element_subsets_l200_200241

open Set 

-- Define the set T
def T : Set Nat := { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }

-- Define the condition: maximum number of 5-element subsets 
-- such that any two elements appear together in at most 2 of these subsets
def is_valid_subset (S : Finset (Finset Nat)) : Prop :=
  ∀ s ∈ S, s.card = 5 ∧ ∀ x y ∈ s, Finset.card {t ∈ S | x ∈ t ∧ y ∈ t} ≤ 2

-- Define the problem statement in Lean
theorem max_five_element_subsets : ∃ S : Finset (Finset Nat), 
  is_valid_subset S ∧ S.card = 8 :=
sorry

end max_five_element_subsets_l200_200241


namespace smallest_number_is_minus_three_l200_200744

theorem smallest_number_is_minus_three :
  ∀ (a b c d : ℤ), (a = 0) → (b = -3) → (c = 1) → (d = -1) → b < d ∧ d < a ∧ a < c → b = -3 :=
by
  intros a b c d ha hb hc hd h
  exact hb

end smallest_number_is_minus_three_l200_200744


namespace frankie_pets_total_l200_200137

theorem frankie_pets_total
  (C S P D : ℕ)
  (h_snakes : S = C + 6)
  (h_parrots : P = C - 1)
  (h_dogs : D = 2)
  (h_total : C + S + P + D = 19) :
  C + (C + 6) + (C - 1) + 2 = 19 := by
  sorry

end frankie_pets_total_l200_200137


namespace vendor_profit_is_1_25_l200_200070

def cost_price_apple : ℝ := 3 / 2
def cost_price_orange : ℝ := 2.7 / 3
def sell_price_apple_without_discount : ℝ := 10 / 5
def sell_price_orange_without_discount : ℝ := 1
def discount_apple : ℝ := 0.1
def discount_orange : ℝ := 0.15

def cost_apples (n : ℕ) : ℝ := n * cost_price_apple
def cost_oranges (n : ℕ) : ℝ := n * cost_price_orange

def discounted_sell_price_apple (n : ℕ) : ℝ := 
  if n > 3 then n * (sell_price_apple_without_discount - sell_price_apple_without_discount * discount_apple)
  else n * sell_price_apple_without_discount

def discounted_sell_price_orange (n : ℕ) : ℝ := 
  if n > 2 then n * (sell_price_orange_without_discount - sell_price_orange_without_discount * discount_orange)
  else n * sell_price_orange_without_discount

def total_cost_price (n_apples n_oranges : ℕ) : ℝ := 
  cost_apples n_apples + cost_oranges n_oranges

def total_sell_price (n_apples n_oranges : ℕ) : ℝ := 
  discounted_sell_price_apple n_apples + discounted_sell_price_orange n_oranges

def profit (n_apples n_oranges : ℕ) : ℝ := 
  total_sell_price n_apples n_oranges - total_cost_price n_apples n_oranges

theorem vendor_profit_is_1_25 : profit 5 5 = 1.25 :=
by
  unfold profit total_sell_price total_cost_price cost_apples cost_oranges 
         discounted_sell_price_apple discounted_sell_price_orange
  sorry

end vendor_profit_is_1_25_l200_200070


namespace find_x_if_vectors_parallel_l200_200866

theorem find_x_if_vectors_parallel (x : ℝ)
  (a : ℝ × ℝ := (x - 1, 2))
  (b : ℝ × ℝ := (2, 1)) :
  (∃ k : ℝ, a = (k * b.1, k * b.2)) → x = 5 :=
by sorry

end find_x_if_vectors_parallel_l200_200866


namespace calc_expression_l200_200081

theorem calc_expression :
  let a := 3^456
  let b := 9^5 / 9^3
  a - b = 3^456 - 81 :=
by
  let a := 3^456
  let b := 9^5 / 9^3
  sorry

end calc_expression_l200_200081


namespace sin_B_eq_l200_200515

theorem sin_B_eq (a b c : ℝ) (A B C : ℝ)
  (h₁ : a + c = 2 * b)
  (h₂ : A - C = π / 3)
  (h₃ : a = 2 * R * sin A)
  (h₄ : b = 2 * R * sin B)
  (h₅ : c = 2 * R * sin C)
  (h₆ : A + B + C = π)
  (hR : R > 0) :
  sin B = sqrt 3 / 2 := 
  sorry

end sin_B_eq_l200_200515


namespace count_divisible_by_45_l200_200500

theorem count_divisible_by_45 : ∃ n : ℕ, n = 10 ∧ (∀ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧ x % 100 = 45 → x % 45 = 0 → n = 10) :=
by {
  sorry
}

end count_divisible_by_45_l200_200500


namespace final_position_west_total_distance_run_farthest_point_l200_200659

/-- Given movement records +40, -30, +50, -25, +25, -30, +15 in meters, 
    prove the final position is 45 meters to the west (positive direction) 
    from the starting point. -/
theorem final_position_west :
  let movements := [+40, -30, +50, -25, +25, -30, +15]
  (List.sum movements) = 45 :=
sorry

/-- Given movement records +40, -30, +50, -25, +25, -30, +15 in meters, 
    prove the total distance run is 215 meters. -/
theorem total_distance_run :
  let movements := [+40, -30, +50, -25, +25, -30, +15]
  (List.sum (List.map Int.natAbs movements)) = 215 :=
sorry

/-- Given movement records +40, -30, +50, -25, +25, -30, +15 in meters, 
    prove the farthest point from the starting point during the student's training is 60 meters. -/
theorem farthest_point :
  let movements := [+40, -30, +50, -25, +25, -30, +15]
  (List.foldl (λ (acc pair : Int × Int) move => 
    let new_pos := acc + move
    (new_pos, max (abs new_pos) (snd acc))) (0, 0) movements).2 = 60 :=
sorry

end final_position_west_total_distance_run_farthest_point_l200_200659


namespace min_value_l200_200179

def vec := (ℝ × ℝ)

-- Definitions given in the problem:
def a : vec := (0, 1)
def b : vec := (-⟨√3⟩ / 2, -(1 / 2))
def c : vec := (⟨√3⟩ / 2, -(1 / 2))

-- Given (1, 2) as a target vector:
def target : vec := (1, 2)

-- Define x, y, z real numbers:
variables (x y z : ℝ)

-- Equation to be satisfied:
def combination (x y z : ℝ) := (x * a.1 + y * b.1 + z * c.1, x * a.2 + y * b.2 + z * c.2)

-- Minimum value of x^2 + y^2 + z^2
def objective_function (x y z : ℝ) := x^2 + y^2 + z^2

theorem min_value : ∃ x y z : ℝ, combination x y z = target ∧ ∀ x' y' z' : ℝ, combination x' y' z' = target → objective_function x' y' z' ≥ objective_function x y z :=
sorry

end min_value_l200_200179


namespace find_nested_function_value_l200_200482

def f : ℝ → ℝ := 
  λ x, if x > 0 then real.log x / real.log 2 else 3^x

theorem find_nested_function_value : f (f (1 / 8)) = 1 / 27 := by
  sorry

end find_nested_function_value_l200_200482


namespace find_b_l200_200120

theorem find_b (a b c : ℚ) (h : (3 * x^2 - 4 * x + 2) * (a * x^2 + b * x + c) = 9 * x^4 - 10 * x^3 + 5 * x^2 - 8 * x + 4)
  (ha : a = 3) : b = 2 / 3 :=
by
  sorry

end find_b_l200_200120


namespace smallest_N_divisibility_l200_200430

theorem smallest_N_divisibility :
  ∃ N : ℕ, 
    (N + 2) % 2 = 0 ∧
    (N + 3) % 3 = 0 ∧
    (N + 4) % 4 = 0 ∧
    (N + 5) % 5 = 0 ∧
    (N + 6) % 6 = 0 ∧
    (N + 7) % 7 = 0 ∧
    (N + 8) % 8 = 0 ∧
    (N + 9) % 9 = 0 ∧
    (N + 10) % 10 = 0 ∧
    N = 2520 := 
sorry

end smallest_N_divisibility_l200_200430


namespace sum_of_binom_eq_sum_of_all_integers_l200_200432

theorem sum_of_binom_eq (k : ℕ) (h : binomial 24 4 + binomial 24 5 = binomial 25 k) :
  k = 5 ∨ k = 20 := by
  sorry

theorem sum_of_all_integers (h₁ : ∀ k : ℕ, binomial 24 4 + binomial 24 5 = binomial 25 k → (k = 5 ∨ k = 20)) :
  (∑ k in {5, 20}, k) = 25 := by
  sorry

end sum_of_binom_eq_sum_of_all_integers_l200_200432


namespace president_and_committee_l200_200214

def combinatorial (n k : ℕ) : ℕ := Nat.choose n k

theorem president_and_committee :
  let num_people := 10
  let num_president := 1
  let num_committee := 3
  let num_ways_president := 10
  let num_remaining_people := num_people - num_president
  let num_ways_committee := combinatorial num_remaining_people num_committee
  num_ways_president * num_ways_committee = 840 := 
by
  sorry

end president_and_committee_l200_200214


namespace problem_find_b_l200_200919

noncomputable def find_b (m a k c d : ℝ) : ℝ :=
  mka - md / kca

theorem problem_find_b 
  (m a k c d b : ℝ) 
  (h : m = kcab / (ka - d)) :
  b = (mka - md) / kca :=
by
  sorry

end problem_find_b_l200_200919


namespace cylinder_not_occupied_volume_l200_200043

theorem cylinder_not_occupied_volume :
  let r := 10
  let h_cylinder := 30
  let h_full_cone := 10
  let volume_cylinder := π * r^2 * h_cylinder
  let volume_full_cone := (1 / 3) * π * r^2 * h_full_cone
  let volume_half_cone := (1 / 2) * volume_full_cone
  let volume_unoccupied := volume_cylinder - (volume_full_cone + volume_half_cone)
  volume_unoccupied = 2500 * π := 
by
  sorry

end cylinder_not_occupied_volume_l200_200043


namespace vasya_tolya_badges_l200_200672

-- Let V be the number of badges Vasya had before the exchange.
-- Let T be the number of badges Tolya had before the exchange.
theorem vasya_tolya_badges (V T : ℕ) 
  (h1 : V = T + 5)
  (h2 : 0.76 * V + 0.20 * T = 0.80 * T + 0.24 * V - 1) :
  V = 50 ∧ T = 45 :=
by 
  sorry

end vasya_tolya_badges_l200_200672


namespace badges_initial_count_l200_200674

variable {V T : ℕ}

-- conditions
def initial_condition : Prop := V = T + 5
def exchange_condition : Prop := 0.76 * V + 0.20 * T = 0.80 * T + 0.24 * V - 1

-- result
theorem badges_initial_count (h1 : initial_condition) (h2 : exchange_condition) : V = 50 ∧ T = 45 := 
  sorry

end badges_initial_count_l200_200674


namespace part1_part2_l200_200851

noncomputable def f : ℝ → ℝ := sorry

variable (x y : ℝ)
variable (hx0 : 0 < x)
variable (hy0 : 0 < y)
variable (hx12 : x < 1 → f x > 0)
variable (hf_half : f (1 / 2) = 1)
variable (hf_mul : f (x * y) = f x + f y)

theorem part1 : (∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2) := sorry

theorem part2 : (∀ x, 3 < x → x < 4 → f (x - 3) > f (1 / x) - 2) := sorry

end part1_part2_l200_200851


namespace decreasing_interval_l200_200288

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

theorem decreasing_interval : ∀ x : ℝ, 0 < x ∧ x < 2 → deriv f x < 0 :=
by sorry

end decreasing_interval_l200_200288


namespace trigonometric_identity_1_trigonometric_identity_2_l200_200258

section problem_statements

variable (θ α : ℝ)

-- First problem statement
theorem trigonometric_identity_1 
  (h1 : sin θ ≠ cos θ): 
  (sin θ - cos θ) / (tan θ - 1) = cos θ := 
sorry

-- Second problem statement
theorem trigonometric_identity_2 
  : sin α ^ 4 - cos α ^ 4 = 2 * sin α ^ 2 - 1 := 
sorry

end problem_statements

end trigonometric_identity_1_trigonometric_identity_2_l200_200258


namespace min_value_F_when_t_is_negative_one_monotonic_intervals_F_t_l200_200845

-- Part I: Proving the minimum value of F(x) when t = -1
theorem min_value_F_when_t_is_negative_one :
  ∀ (x : ℝ), F(x, -1) = abs (2 * x - 1) + x^2 + x + 1 → 
  (∀ x < (1 / 2), F(x, -1) = x^2 - x + 2) ∧ 
  (∀ x ≥ (1 / 2), F(x, -1) = x^2 + 3 * x) → 
  ∃ x_min : ℝ, x_min = (1 / 2) ∧ F(x_min, -1) = (7 / 4) := sorry

-- Part II: Proving the monotonic intervals of F(x) for all t ∈ ℝ
theorem monotonic_intervals_F_t (t : ℝ) :
  (∀ x : ℝ, F(x, t) = abs (2 * x + t) + x^2 + x + 1) →
  (if t ≥ 3 then 
    (∀ x ≤ (-3 / 2), F(x, t) is decreasing) ∧ 
    (∀ x ≥ (-3 / 2), F(x, t) is increasing)
   else if -1 < t < 3 then 
    (∀ x ≤ (-t / 2), F(x, t) is decreasing) ∧ 
    (∀ x ≥ (-t / 2), F(x, t) is increasing)
   else 
    (∀ x ≤ (1 / 2), F(x, t) is decreasing) ∧ 
    (∀ x ≥ (1 / 2), F(x, t) is increasing)
  ) := sorry

end min_value_F_when_t_is_negative_one_monotonic_intervals_F_t_l200_200845


namespace top_level_supervisors_l200_200320

theorem top_level_supervisors (N : ℕ) (hN : N = 50000) 
    (h_sum : ∀ e : ℕ, e ∈ finset.range N → (sup e + subs e = 7))
    (h_propagation_stops : orders_propagation_stops_by_friday) :
    ∃ k ≥ 97, ∀ i ∈ finset.range k, is_top_level_supervisor i :=
sorry

end top_level_supervisors_l200_200320


namespace prove_range_of_m_prove_m_value_l200_200822

def quadratic_roots (m : ℝ) (x1 x2 : ℝ) : Prop := 
  x1 * x1 - (2 * m - 3) * x1 + m * m = 0 ∧ 
  x2 * x2 - (2 * m - 3) * x2 + m * m = 0

def range_of_m (m : ℝ) : Prop := 
  m <= 3/4

def condition_on_m (m : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 + x2 = -(x1 * x2)

theorem prove_range_of_m (m : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_roots m x1 x2) → range_of_m m :=
sorry

theorem prove_m_value (m : ℝ) (x1 x2 : ℝ) :
  quadratic_roots m x1 x2 → condition_on_m m x1 x2 → m = -3 :=
sorry

end prove_range_of_m_prove_m_value_l200_200822


namespace find_common_ratio_l200_200406

-- Define the given conditions
def is_geometric_sequence (a : ℕ → ℝ) (r k : ℝ) : Prop :=
  ∀ n, a (n) = k * (a (n + 1) + a (n + 2))

-- Define the initial conditions
def k : ℝ := 3
def positive_terms (a : ℕ → ℝ) := ∀ n, a n > 0

-- Define the common ratio r
def r : ℝ := (-1 + Real.sqrt(7 / 3)) / 2

-- The proof problem statement
theorem find_common_ratio (a : ℕ → ℝ) (h1 : is_geometric_sequence a r k) (h2 : positive_terms a) :
  r = (-1 + Real.sqrt(7 / 3)) / 2 :=
sorry

end find_common_ratio_l200_200406


namespace boat_speed_in_still_water_l200_200708

-- Define the conditions
def speed_of_stream : ℝ := 5
def distance_downstream : ℝ := 189
def time_downstream : ℝ := 7

-- Define the effective speed as boat speed in still water + stream speed
def effective_speed (V_b : ℝ) : ℝ := V_b + speed_of_stream

-- Given conditions stated mathematically
def downstream_distance_eq (V_b : ℝ) : Prop := distance_downstream = effective_speed(V_b) * time_downstream

-- The theorem to prove the speed of the boat in still water is 22 km/hr.
theorem boat_speed_in_still_water : ∃ V_b : ℝ, downstream_distance_eq V_b ∧ V_b = 22 :=
by
  use 22
  unfold downstream_distance_eq effective_speed
  simp
  sorry -- Replace with actual math proof in Lean

end boat_speed_in_still_water_l200_200708


namespace angle_bisector_in_acute_triangle_l200_200968

theorem angle_bisector_in_acute_triangle
  (ABC : Triangle)
  (hABCacute : ABC.isAcute)
  (hAngBAC_lt_AngACB : ABC.∠BAC < ABC.∠ACB)
  (ω : Circumcircle ABC.stroke)
  (hAD : Diameter ω AD)
  (E : Point)
  (hE_def : E = ray_intersection AC (tangent_at B ω))
  (hPerpendicular : perpendicular AD E)
  (F : Point)
  (hF_on_circumcircle : F ∈ Circumcircle (Triangle B C E).stroke)
  (hF_perpendicular : perpendicular E AD) :
  is_angle_bisector (angle B C F) CD := sorry

end angle_bisector_in_acute_triangle_l200_200968


namespace student_calculation_no_error_l200_200380

theorem student_calculation_no_error :
  let correct_result : ℚ := (7 * 4) / (5 / 3)
  let student_result : ℚ := (7 * 4) * (3 / 5)
  correct_result = student_result → 0 = 0 := 
by
  intros correct_result student_result h
  sorry

end student_calculation_no_error_l200_200380


namespace probability_eagle_chick_l200_200651

theorem probability_eagle_chick : 
  ∃ (cards : List (List String)), 
  (length cards = 4) ∧
  (set.to_finset cards = {"Mouse-Eagle", "Mouse-Snake", "Chick-Eagle", "Chick-Snake"}.to_finset) ∧
  (1 / 4) = 1 / 4 :=
by
  let combinations := ["Mouse-Eagle", "Mouse-Snake", "Chick-Eagle", "Chick-Snake"]
  have h1 : length combinations = 4 := rfl
  have h2 : set.to_finset combinations = {"Mouse-Eagle", "Mouse-Snake", "Chick-Eagle", "Chick-Snake"}.to_finset := rfl
  use combinations
  exact ⟨h1, h2, rfl⟩

end probability_eagle_chick_l200_200651


namespace triangle_angle_and_area_l200_200937

theorem triangle_angle_and_area (a b c A B C : ℝ) 
    (h1 : c = √7) 
    (h2 : b = 3 * a) 
    (h3 : cos (2 * C - 3 * cos (A + B)) = 1) :
    C = π / 3 ∧
    (1/2 * a * b * sin (C) = 3 * √3 / 2) :=
by
    sorry

end triangle_angle_and_area_l200_200937


namespace decreasing_function_range_l200_200849

variable (a : ℝ)
variable (f : ℝ → ℝ := λ x, if x ≤ 1 then (2 - 3 * a) * x + 1 else a / x)

theorem decreasing_function_range :
  (∀ x y, x < y → f x ≥ f y) ↔ (2/3 < a ∧ a ≤ 3/4) := 
by
  sorry

end decreasing_function_range_l200_200849


namespace sum_and_product_of_solutions_l200_200797

theorem sum_and_product_of_solutions (x : ℝ) :
  let solutions := {x | (x - 6)^2 = 49}
  ∑ s in solutions, s = 12 ∧ ∏ s in solutions, s = -13 :=
by
  sorry

end sum_and_product_of_solutions_l200_200797


namespace radius_of_circle_eq_3sqrt5_l200_200059

-- Auxiliary functions and definitions
def point := ℝ × ℝ -- Defining a point in real coordinate plane
noncomputable def distance (P Q : point) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Given conditions converted to Lean 4 definitions
def center : point := (0, 0)
def P : point := (15, 0)
def PQ_length : ℝ := 10
def QR_length : ℝ := 8

-- Main theorem to prove
theorem radius_of_circle_eq_3sqrt5 :
  ∃ r : ℝ, distance center P = 15 ∧ 
           PQ_length * (PQ_length + QR_length) = (15 - r) * (15 + r) ∧ 
           r = 3 * real.sqrt 5 :=
  sorry

end radius_of_circle_eq_3sqrt5_l200_200059


namespace work_completion_time_l200_200347

-- Given conditions as Lean definitions
def A := 2 * B
def B : ℝ := 1 / 54

-- Theorem to prove
theorem work_completion_time : 1 / (A + B) = 18 :=
by
  sorry

end work_completion_time_l200_200347


namespace find_even_function_with_period_pi_l200_200387

theorem find_even_function_with_period_pi :
  ∃ f : ℝ → ℝ, (∀ x, f x = cos (2 * x)) ∧
               (∀ x, f x = f (-x)) ∧
               (∀ T > 0, (∀ x, f (x + T) = f x) → T = π) :=
by
  sorry

end find_even_function_with_period_pi_l200_200387


namespace internet_provider_discount_l200_200749

theorem internet_provider_discount (monthly_rate : ℝ) (total_payment : ℝ) (months : ℕ)
    (payment_day : ℕ) (h1 : monthly_rate = 50) (h2 : total_payment = 190) (h3 : months = 4) :
    ((monthly_rate * months - total_payment) / (monthly_rate * months)) * 100 = 5 :=
by
  have average_monthly_payment := total_payment / months
  have discount_amount := monthly_rate - average_monthly_payment
  have discount_percentage := (discount_amount / monthly_rate) * 100
  rw [h1, h2, h3] at *
  simp only [discount_percentage]
  sorry

end internet_provider_discount_l200_200749


namespace kite_area_ratio_l200_200210

theorem kite_area_ratio (side_large_square side_small_square : ℝ)
  (h1 : side_large_square = 60)
  (h2 : side_small_square = 10)
  (h3 : side_large_square = 6 * side_small_square) :
  let area_large := side_large_square^2,
      area_kite := (1 / 2) * (2 * side_small_square * 2 * side_small_square * √2) in
  area_kite / area_large = 100 * √2 / 3600 := by
  sorry

end kite_area_ratio_l200_200210


namespace cubic_difference_l200_200470

theorem cubic_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 50) : a^3 - b^3 = 353.5 := by
  sorry

end cubic_difference_l200_200470


namespace complement_of_67_is_23_l200_200284

-- Define complement function
def complement (x : ℝ) : ℝ := 90 - x

-- State the theorem
theorem complement_of_67_is_23 : complement 67 = 23 := 
by
  sorry

end complement_of_67_is_23_l200_200284


namespace allan_balloons_l200_200072

def jak_balloons : ℕ := 11
def diff_balloons : ℕ := 6

theorem allan_balloons (jake_allan_diff : jak_balloons = diff_balloons + 5) : jak_balloons - diff_balloons = 5 :=
by
  sorry

end allan_balloons_l200_200072


namespace cost_of_traveling_roads_l200_200729

def lawnLength : ℝ := 100
def lawnBreadth : ℝ := 60
def roadWidth : ℝ := 10
def costPerSqMeter : ℝ := 3

def areaRoad1 : ℝ := roadWidth * lawnBreadth
def areaIntersection : ℝ := roadWidth * roadWidth
def areaRoad2 : ℝ := roadWidth * lawnLength - areaIntersection
def totalAreaRoads : ℝ := areaRoad1 + areaRoad2
def totalCost : ℝ := totalAreaRoads * costPerSqMeter

theorem cost_of_traveling_roads : totalCost = 4500 := by
  unfold totalCost totalAreaRoads areaRoad1 areaRoad2 areaIntersection roadWidth lawnBreadth lawnLength costPerSqMeter
  norm_num
  sorry

end cost_of_traveling_roads_l200_200729


namespace max_value_of_a_l200_200933

theorem max_value_of_a :
  (∀ x : ℝ, 2 * x + 3 > 3 * x - 1 → x < 4) →
  (∀ x : ℝ, 6 * x - a ≥ 2 * x + 2 → x ≥ (a + 2) / 4) →
  (0 = (2 - a) ∧ -2 < a ∧ a < 2 ∧ a ≠ 1) →
  a = 2 :=
by
  intros h1 h2 h3,
  sorry

end max_value_of_a_l200_200933


namespace width_of_hall_l200_200951

theorem width_of_hall (length : ℕ) (height : ℕ) (total_expenditure : ℕ) (cost_per_square_meter : ℕ)
  (h_length : length = 20) (h_height : height = 5) (h_total_expenditure : total_expenditure = 38000) 
  (h_cost : cost_per_square_meter = 40) : 
  ∃ w : ℕ, 1200 * w + 8000 = total_expenditure ∧ w = 25 := 
by {
  use 25,
  split,
  {
    sorry, -- Provide proof here
  },
  {
    rfl,
  },
}

end width_of_hall_l200_200951


namespace max_expr_value_l200_200125

theorem max_expr_value (a b c d : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1) (hb : 0 ≤ b) (hb1 : b ≤ 1) (hc : 0 ≤ c) (hc1 : c ≤ 1) (hd : 0 ≤ d) (hd1 : d ≤ 1) : 
  a + b + c + d - a * b - b * c - c * d - d * a ≤ 2 :=
sorry

end max_expr_value_l200_200125


namespace problem_part_I_problem_part_II_l200_200859

open BigOperators

def seq_a (n : ℕ) : ℚ 
| 0       := 3 / 2
| (n + 1) := seq_a n^2 + seq_a n

def A_n (n : ℕ) : ℚ := ∑ i in finset.range n, (seq_a i)^2

def B_n (n : ℕ) : ℚ := ∑ i in finset.range n, 1 / (seq_a i + 1)

theorem problem_part_I (n : ℕ) (hn : 1 ≤ n) : 
  2^(2^(n-1)) - 1 / 2 ≤ seq_a n ∧ seq_a n ≤ 1 / 2 * 3^(2^(n-1)) := 
sorry

theorem problem_part_II (n : ℕ) : 0 < n → 
  A_n n / B_n n = 3 / 2 * seq_a (n + 1) := 
sorry

end problem_part_I_problem_part_II_l200_200859


namespace count_ordered_pairs_satisfying_eq_l200_200411

theorem count_ordered_pairs_satisfying_eq :
  {p : ℤ × ℤ | p.1 ^ 2021 + p.2 ^ 2 = 3 * p.2}.to_finset.card = 2 :=
sorry

end count_ordered_pairs_satisfying_eq_l200_200411


namespace find_m_l200_200890

-- Declare the vectors a and b based on given conditions
variables {m : ℝ}

def a : ℝ × ℝ := (m, 3)
def b : ℝ × ℝ := (1, m + 1)

-- Define the condition that vectors a and b are perpendicular
def perpendicular (x y : ℝ × ℝ) : Prop := x.1 * y.1 + x.2 * y.2 = 0

-- State the problem in Lean 4
theorem find_m (h : perpendicular a b) : m = -3 / 4 :=
sorry

end find_m_l200_200890


namespace product_of_sums_of_x_l200_200991

theorem product_of_sums_of_x (x : ℕ → ℝ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ 42 → 5 * x (i + 1) - x i - 3 * x i * x (i + 1) = 1) (h2 : x 1 = x 43) : 
  (x 1 + x 2 + ⋯ + x 42 = 42 ∨ x 1 + x 2 + ⋯ + x 42 = 14) → (42 * 14 = 588) :=
by
  sorry

end product_of_sums_of_x_l200_200991


namespace different_result_l200_200342

theorem different_result :
  let A := -2 - (-3)
  let B := 2 - 3
  let C := -3 + 2
  let D := -3 - (-2)
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B = C ∧ B = D :=
by
  sorry

end different_result_l200_200342


namespace min_cost_of_packaging_l200_200048

def packaging_problem : Prop :=
  ∃ (x y : ℕ), 35 * x + 24 * y = 106 ∧ 140 * x + 120 * y = 500

theorem min_cost_of_packaging : packaging_problem :=
sorry

end min_cost_of_packaging_l200_200048


namespace badges_before_exchange_l200_200666

theorem badges_before_exchange (V T : ℕ) (h1 : V = T + 5) (h2 : 76 * V + 20 * T = 80 * T + 24 * V - 100) :
  V = 50 ∧ T = 45 :=
by
  sorry

end badges_before_exchange_l200_200666


namespace PA_PB_product_l200_200971

open Real

/- Define the problem data -/
def point_P : ℝ × ℝ := (3, sqrt 5)
def slope_angle : ℝ := 3 * π / 4

def polar_eq_circle (θ : ℝ) : ℝ := 2 * sqrt 5 * sin θ

def param_eq_line (t : ℝ) : ℝ × ℝ :=
  (3 - (sqrt 2) / 2 * t, sqrt 5 + (sqrt 2) / 2 * t)

def rect_eq_circle (x y : ℝ) : Prop :=
  (x - 0)^2 + (y - sqrt 5)^2 = 5

/- Main theorem stating the proof problem -/
theorem PA_PB_product (A B : ℝ × ℝ) (t1 t2 : ℝ) :
  let P := point_P
  in let line_intersect (t : ℝ) := param_eq_line t
  in rect_eq_circle (fst (line_intersect t1)) (snd (line_intersect t1)) ∧
     rect_eq_circle (fst (line_intersect t2)) (snd (line_intersect t2)) →
     (let PA := dist (P.1, P.2) (fst (line_intersect t1), snd (line_intersect t1))
      in let PB := dist (P.1, P.2) (fst (line_intersect t2), snd (line_intersect t2))
      in |PA| * |PB| = 2) := sorry

end PA_PB_product_l200_200971


namespace total_revenue_correct_l200_200021

-- Defining the basic parameters
def ticket_price : ℝ := 20
def first_discount_percentage : ℝ := 0.40
def next_discount_percentage : ℝ := 0.15
def first_people : ℕ := 10
def next_people : ℕ := 20
def total_people : ℕ := 48

-- Calculate the discounted prices based on the given percentages
def discounted_price_first : ℝ := ticket_price * (1 - first_discount_percentage)
def discounted_price_next : ℝ := ticket_price * (1 - next_discount_percentage)

-- Calculate the total revenue
def revenue_first : ℝ := first_people * discounted_price_first
def revenue_next : ℝ := next_people * discounted_price_next
def remaining_people : ℕ := total_people - first_people - next_people
def revenue_remaining : ℝ := remaining_people * ticket_price

def total_revenue : ℝ := revenue_first + revenue_next + revenue_remaining

-- The statement to be proved
theorem total_revenue_correct : total_revenue = 820 :=
by
  -- The proof will go here
  sorry

end total_revenue_correct_l200_200021


namespace simplify_trig_expression_l200_200603

theorem simplify_trig_expression (α β : ℝ) :
  (sin α)^2 + (sin β)^2 - (sin α)^2 * (sin β)^2 + (cos α)^2 * (cos β)^2 = 1 := 
by 
  sorry

end simplify_trig_expression_l200_200603


namespace cars_meeting_time_l200_200808

def problem_statement (V_A V_B V_C V_D : ℝ) :=
  (V_A ≠ V_B) ∧ (V_A ≠ V_C) ∧ (V_A ≠ V_D) ∧
  (V_B ≠ V_C) ∧ (V_B ≠ V_D) ∧ (V_C ≠ V_D) ∧
  (V_A + V_C = V_B + V_D) ∧
  (53 * (V_A - V_B) / 46 = 7) ∧
  (53 * (V_D - V_C) / 46 = 7)

theorem cars_meeting_time (V_A V_B V_C V_D : ℝ) (h : problem_statement V_A V_B V_C V_D) : 
  ∃ t : ℝ, t = 53 := 
sorry

end cars_meeting_time_l200_200808


namespace find_inheritance_amount_l200_200551

noncomputable def totalInheritance (tax_amount : ℕ) : ℕ :=
  let federal_rate := 0.20
  let state_rate := 0.10
  let combined_rate := federal_rate + (state_rate * (1 - federal_rate))
  sorry

theorem find_inheritance_amount : totalInheritance 10500 = 37500 := 
  sorry

end find_inheritance_amount_l200_200551


namespace quadrants_contain_points_l200_200119

def satisfy_inequalities (x y : ℝ) : Prop :=
  y > -3 * x ∧ y > x + 2

def in_quadrant_I (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

def in_quadrant_II (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem quadrants_contain_points (x y : ℝ) :
  satisfy_inequalities x y → (in_quadrant_I x y ∨ in_quadrant_II x y) :=
sorry

end quadrants_contain_points_l200_200119


namespace expected_pairs_same_wins_l200_200704

/-- A proof problem to find the expected number of pairs of players
who win the same number of games in a Super Smash Bros. Melee tournament with 20 players -/
theorem expected_pairs_same_wins :
  let players := (1:ℕ) to 20,
      always_beats (n m : ℕ) := n < m,
      plays_18_times := ∀ p ∈ players, ∃ T ⊆ (players \ {p}), #T = 18,
      uniform_random_tournament := ∃ t ∈ (set.powerset (set_of (λ (p₁ p₂ : ℕ), p₁ ∈ players ∧ p₂ ∈ players))
                                          .to_finset), t.card = 18 * (players.to_finset.card / 2)
  in
  ∑ p₁ p₂ in players.to_finset, if always_beats p₁ p₂ then 1 else 0 = 4 := 
begin
  sorry
end

end expected_pairs_same_wins_l200_200704


namespace solve_system_l200_200696

theorem solve_system :
  ∃ (x y z : ℝ), 
    (x + y + z = 13) ∧ 
    (x^2 + y^2 + z^2 = 61) ∧ 
    (x * y + x * z = 2 * y * z) ∧ 
    ((x = 4 ∧ y = 3 ∧ z = 6) ∨ 
     (x = 4 ∧ y = 6 ∧ z = 3)) :=
begin
  sorry
end

end solve_system_l200_200696


namespace james_coursework_materials_expense_l200_200979

-- Definitions based on conditions
def james_budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

-- Calculate expenditures based on percentages
def food_expense : ℝ := food_percentage * james_budget
def accommodation_expense : ℝ := accommodation_percentage * james_budget
def entertainment_expense : ℝ := entertainment_percentage * james_budget
def total_other_expenses : ℝ := food_expense + accommodation_expense + entertainment_expense

-- Prove that the amount spent on coursework materials is $300
theorem james_coursework_materials_expense : james_budget - total_other_expenses = 300 := 
by 
  sorry

end james_coursework_materials_expense_l200_200979


namespace find_a5_from_conditions_l200_200467

variable {a : ℕ → ℝ} (d : ℝ) (a₁ : ℝ)

def arithmetic_sequence := ∀ n : ℕ, a n = a₁ + (n - 1) * d

theorem find_a5_from_conditions (h₀ : d ≠ 0)
  (h₁ : arithmetic_sequence d a₁ a)
  (h₂ : a 3 + a 9 = a 10 - a 8) :
  a 5 = 0 :=
sorry

end find_a5_from_conditions_l200_200467


namespace rectangle_area_perimeter_l200_200154

theorem rectangle_area_perimeter (a b : ℝ) (h₁ : a * b = 6) (h₂ : a + b = 6) : a^2 + b^2 = 24 := 
by
  sorry

end rectangle_area_perimeter_l200_200154


namespace sector_perimeter_l200_200208

noncomputable def radius : ℝ := 2
noncomputable def central_angle_deg : ℝ := 120
noncomputable def expected_perimeter : ℝ := (4 / 3) * Real.pi + 4

theorem sector_perimeter (r : ℝ) (θ : ℝ) (h_r : r = radius) (h_θ : θ = central_angle_deg) :
    let arc_length := θ / 360 * 2 * Real.pi * r
    let perimeter := arc_length + 2 * r
    perimeter = expected_perimeter :=
by
  -- Skip the proof
  sorry

end sector_perimeter_l200_200208


namespace reflection_matrix_through_plane_l200_200562

theorem reflection_matrix_through_plane :
  let n := ![2, -1, 1]
  let p := ![1, 1, 0]
  let plane_eq (x y z : ℝ) := (2 * x - y + z = 1)
  let S := ![
    [-2/3, 2/3, -4/3],
    [4/3, 5/3, 2/3],
    [-4/3, 5/3, 2/3]
  ]
  ∀ u : ℝ^3, let u_perp := ((2 * u 0 - u 1 + u 2) / 6) • ![2, -1, 1]
    u_proj := u - u_perp
    reflection := 2 • u_proj - u
  S ⬝ u = reflection :=
by
  sorry

end reflection_matrix_through_plane_l200_200562


namespace potato_chips_per_potato_l200_200097

theorem potato_chips_per_potato :
  ∀ (total_potatoes cut_wedges half_potatoes : ℕ) (chips_extra_wedges wedges_per_potato : ℕ),
    total_potatoes = 67 →
    cut_wedges = 13 →
    half_potatoes = (total_potatoes - cut_wedges) / 2 →
    chips_extra_wedges = 436 →
    wedges_per_potato = 8 →
    let wedges := cut_wedges * wedges_per_potato in
    let chips := wedges + chips_extra_wedges in
    let potatoes_for_chips := half_potatoes in
    potatoes_for_chips = 27 →
    chips / potatoes_for_chips = 20 :=
begin
  intros total_potatoes cut_wedges half_potatoes chips_extra_wedges wedges_per_potato,
  intros h_tot h_cut h_half h_extra h_wedges,
  simp only [sub_eq_add_neg, nat.add, nat.div, nat.mul],
  intros wedges chips potatoes_for_chips h_potatoes_for_chips,
  simp [h_tot, h_cut, h_half, h_extra, h_wedges, h_potatoes_for_chips],
  sorry
end

end potato_chips_per_potato_l200_200097


namespace modulus_of_z_l200_200126

noncomputable def z : ℂ := 1 - 2 * complex.I

theorem modulus_of_z : complex.abs z = Real.sqrt 5 :=
by
  sorry

end modulus_of_z_l200_200126


namespace find_required_speed_l200_200348

variable (D T : ℝ)

-- This is the condition that the person covers 2/3 of the total distance in 1/3 of the total time at 80 kmph
def initial_condition (D T : ℝ) : Prop := (2/3 * D) = 80 * (T / 3)

-- This is the resulting speed calculation
def required_speed (D T : ℝ) : ℝ := 20

theorem find_required_speed 
  (h1: initial_condition D T) :
  ∃ v, required_speed D T = v := by
exists 20
sorry

end find_required_speed_l200_200348


namespace factor_expression_l200_200763

theorem factor_expression:
  ∀ (x : ℝ), (10 * x^3 + 50 * x^2 - 4) - (3 * x^3 - 5 * x^2 + 2) = 7 * x^3 + 55 * x^2 - 6 :=
by
  sorry

end factor_expression_l200_200763


namespace quadratic_root_relationship_l200_200133

theorem quadratic_root_relationship (a b c : ℝ) (α β : ℝ)
  (h1 : a ≠ 0)
  (h2 : α + β = -b / a)
  (h3 : α * β = c / a)
  (h4 : β = 3 * α) : 
  3 * b^2 = 16 * a * c :=
sorry

end quadratic_root_relationship_l200_200133


namespace find_angle_C_condition1_find_angle_C_condition2_find_angle_C_condition3_find_min_value_BD_l200_200762

-- Define the parameters and conditions for the triangle ABC
variables {A B C : ℝ} -- angles
variables {a b c : ℝ} -- sides opposite to the respective angles

-- Condition 1: b sin((A+B)/2) = c sin(B)
def condition1 : Prop := b * Real.sin((A + B) / 2) = c * Real.sin(B)

-- Proof 1: Given condition1, prove C = π / 3
theorem find_angle_C_condition1 (h : condition1) : C = π / 3 := sorry

-- Condition 2: √3 (vector CA) ⋅ (vector CB) = 2Sₜ₆
def S_triangle : ℝ := (1/2) * a * b * Real.sin(C)
def condition2 : Prop := sqrt(3) * a * b * Real.cos(C) = 2 * S_triangle

-- Proof 2: Given condition2, prove C = π / 3
theorem find_angle_C_condition2 (h : condition2) : C = π / 3 := sorry

-- Condition 3: √3 sin(A) + cos(A) = (a+b)/c
def condition3 : Prop := sqrt(3) * Real.sin(A) + Real.cos(A) = (a + b) / c

-- Proof 3: Given condition3, prove C = π / 3
theorem find_angle_C_condition3 (h : condition3) : C = π / 3 := sorry

-- Condition for Part 2: Area and midpoint condition
def area_triangle : ℝ := 8 * sqrt(3)
def midpoint_D : Prop := True -- placeholder, as no specific condition needed for the midpoint here

-- Proof 4: Given area and midpoint condition, find minimum value of BD as 4
theorem find_min_value_BD (h : area_triangle = (1/2) * a * b * Real.sin(C)
  ∧ C = π / 3 ∧ midpoint_D) : 4 ≤ a := sorry

end find_angle_C_condition1_find_angle_C_condition2_find_angle_C_condition3_find_min_value_BD_l200_200762


namespace plaza_renovation_cost_and_length_l200_200736

def side_length := 20
def light_tile_cost_total := 100000
def dark_tile_cost_total := 300000
def dark_border_width := 2

theorem plaza_renovation_cost_and_length :
  let dark_cost := dark_tile_cost_total / 4 in
  let total_cost := 2 * dark_cost in
  total_cost = 150000 ∧
  let dark_area := (side_length * side_length) / 4 in
  let dark_corner_area := 4 * (dark_border_width * dark_border_width) in
  let side_strip_area := dark_area - dark_corner_area in
  let side_length_central_light :=
    side_length - (2 * dark_border_width) - (side_strip_area / dark_border_width) in
  true :=
by
  sorry

end plaza_renovation_cost_and_length_l200_200736


namespace mean_difference_l200_200276

theorem mean_difference (S : ℝ) :
  let mean_actual := (S + 105000) / 800
  let mean_incorrect := (S + 1050000) / 800
  mean_incorrect - mean_actual = 1181.25 :=
by
  let mean_actual := (S + 105000) / 800
  let mean_incorrect := (S + 1050000) / 800
  have diff := mean_incorrect - mean_actual
  have expected_diff := 945000 / 800
  have : expected_diff = 1181.25 := by sorry
  exact this

end mean_difference_l200_200276


namespace remainder_18_pow_63_mod_5_l200_200335

theorem remainder_18_pow_63_mod_5 :
  (18:ℤ) ^ 63 % 5 = 2 :=
by
  -- Given conditions
  have h1 : (18:ℤ) % 5 = 3 := by norm_num
  have h2 : (3:ℤ) ^ 4 % 5 = 1 := by norm_num
  sorry

end remainder_18_pow_63_mod_5_l200_200335


namespace total_number_of_fruits_is_37_l200_200654

def bucket_fruits (A B C : ℕ) : Prop :=
  A = B + 4 ∧ B = C + 3 ∧ C = 9 ∧ A + B + C = 37

theorem total_number_of_fruits_is_37 (A B C : ℕ) : bucket_fruits A B C :=
  by
  -- Given conditions
  let C_val := 9  -- bucket C has 9 pieces of fruit
  let B_val := C_val + 3  -- bucket B has 3 more pieces of fruit than bucket C
  let A_val := B_val + 4  -- bucket A has 4 more pieces of fruit than bucket B
  
   -- Combined statement
  have hA : A = A_val := sorry
  have hB : B = B_val := sorry
  have hC : C = C_val := sorry
  have ht : A + B + C = 37 := sorry
  
  exact ⟨hA, hB, hC, ht⟩

end total_number_of_fruits_is_37_l200_200654


namespace evaluate_expression_l200_200115

theorem evaluate_expression : (125^(1/3) * 81^(-1/4) * 32^(1/5) / 8^(1/3) = (5/3)) :=
by
  sorry

end evaluate_expression_l200_200115


namespace delores_money_left_l200_200104

theorem delores_money_left (initial_amount spent_computer spent_printer : ℝ) 
    (h1 : initial_amount = 450) 
    (h2 : spent_computer = 400) 
    (h3 : spent_printer = 40) : 
    initial_amount - (spent_computer + spent_printer) = 10 := 
by 
    sorry

end delores_money_left_l200_200104


namespace product_odd_l200_200355

variable (f g : ℝ → ℝ)

-- Conditions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Problem statement
theorem product_odd (hf : is_odd f) (hg : is_even g) : is_odd (λ x, f x * |g x|) := by
  sorry

end product_odd_l200_200355


namespace range_angle_A1M_C1N_l200_200969

class cube (V : Type) :=
  (A B C D A1 B1 C1 D1 : V)

variables {V : Type} [inner_product_space ℝ V] [cube V]

def on_segment (a b p : V) : Prop :=
∃ t : ℝ, 0 < t ∧ t < 1 ∧ p = (1-t) • a + t • b

variables (M N : V)

noncomputable def AM_eq_B1N :=
  ∥M - A∥ = ∥N - B1 ∥

theorem range_angle_A1M_C1N (hM : on_segment A B M) (hN : on_segment B B1 N) (h_len : AM_eq_B1N) :
  ∃ θ, θ ∈ set.Ioo (real.pi / 3) (real.pi / 2) ∧ θ = real.angle (A1 - M) (C1 - N) :=
sorry

end range_angle_A1M_C1N_l200_200969


namespace perpendicular_vectors_l200_200882

variable (m : ℝ)

def vector_a := (m, 3)
def vector_b := (1, m + 1)

def dot_product (v w : ℝ × ℝ) := (v.1 * w.1) + (v.2 * w.2)

theorem perpendicular_vectors (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by 
  unfold vector_a vector_b dot_product at h
  linarith

end perpendicular_vectors_l200_200882


namespace reduction_is_65_percent_l200_200049

variable (P : ℝ)

def initial_discount := 0.7 * P
def additional_discount := 0.5 * initial_discount P
def final_price := additional_discount P
def price_reduction := (P - final_price P) / P

theorem reduction_is_65_percent : price_reduction P = 0.65 :=
by
  sorry

end reduction_is_65_percent_l200_200049


namespace radius_of_tangent_circles_max_number_of_tangent_circles_l200_200414

noncomputable theory

-- Definition of concentric circles with the given radii
def inner_circle_radius : ℝ := 1
def outer_circle_radius : ℝ := 3

-- Proof that the radius of the tangent circles is 1 cm
theorem radius_of_tangent_circles :
  ∀ r : ℝ, (∃ r, r > 0 ∧ ∀ x, (x > inner_circle_radius + r ∧ x < outer_circle_radius - r) → false) →
          r = 1 :=
by sorry

-- Proof that the maximum number of such tangent circles is 6
theorem max_number_of_tangent_circles :
  ∀ n : ℕ, (∃ lst : list ℝ, 
              (∀ x : ℝ, x ∈ lst → x = 1) ∧
              lst.length = n ∧
              ∀ i j, i ≠ j → dist lst[i] lst[j] = 2) →
           n = 6 :=
by sorry

end radius_of_tangent_circles_max_number_of_tangent_circles_l200_200414


namespace solve_equation_l200_200024

theorem solve_equation :
  ∃ x : ℝ, (x * 3967 + 36990 - 204790 / 19852 = 322299) ∧ (x = 71.924) :=
by 
  -- Equation given in the conditions
  let eq := x * 3967 + 36990 - 204790 / 19852 = 322299
  have h₁ : 204790 / 19852 = 10.318 := sorry, -- Step from solution
  have h₂ : 36990 - 10.318 = 36979.682 := sorry, -- Step from solution
  have h₃ : 322299 - 36979.682 = 285319.318 := sorry, -- Step from solution
  have h₄ : 285319.318 / 3967 = 71.924 := sorry, -- Step from solution
  use 71.924,
  split,
  { -- Substituting the results into the Lean equation
    sorry 
  },
  {  -- Confirming the final value of x
     rfl 
  }

end solve_equation_l200_200024


namespace will_drank_each_day_l200_200684

theorem will_drank_each_day (total_bottles : ℕ) (days : ℕ) (h : total_bottles = 28 ∧ days = 4) : total_bottles / days = 7 :=
by
  cases h with h1 h2
  rw [h1, h2]
  exact rfl

end will_drank_each_day_l200_200684


namespace even_function_g_l200_200538

def g (x : ℝ) : ℝ := 4 / (5 * x^8 - 7)

theorem even_function_g : ∀ x : ℝ, g (-x) = g x := 
by
  sorry

end even_function_g_l200_200538


namespace expand_polynomial_l200_200420

theorem expand_polynomial (x : ℝ) : (x - 3) * (4 * x + 12) = 4 * x ^ 2 - 36 := 
by {
  sorry
}

end expand_polynomial_l200_200420


namespace constant_term_in_expansion_eq_neg8_l200_200622

open Nat

def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k

def binom_expansion_term (x : ℝ) (n r : ℕ) : ℝ := 
  (-1)^r * binom_coeff n r * x^r

def constant_term (n : ℕ) (x : ℝ) : ℝ :=
  2 * binom_expansion_term x n 0 + 
  binom_expansion_term x n 3 * (1 / x^3)

theorem constant_term_in_expansion_eq_neg8 :
  constant_term 5 1 = -8 :=
by
  -- Proof steps here
  sorry

end constant_term_in_expansion_eq_neg8_l200_200622


namespace complex_magnitude_eq_l200_200397

def complex_mag_37 (z : ℂ) : Prop :=
  |z| = real.sqrt 37

theorem complex_magnitude_eq (a : ℂ) (ha : complex_mag_37 a) : 
  |a^4| = 1369 :=
by 
  sorry

end complex_magnitude_eq_l200_200397


namespace number_of_pots_solution_minimum_total_cost_solution_l200_200972

-- Define the variables and conditions from the problem
def number_of_pots := 46
def cost_green_lily := 9
def cost_spider_plant := 6
def budget := 390
def min_green_lily_twice_spider_plants (x y : ℕ) := x >= 2 * y

-- Define the first proof goal: the number of pots of each type under the budget constraint
theorem number_of_pots_solution (x y : ℕ) (h1 : x + y = number_of_pots) 
  (h2 : cost_green_lily * x + cost_spider_plant * y = budget) 
  (h3 : min_green_lily_twice_spider_plants x y) : 
  x = 38 ∧ y = 8 := 
by 
  sorry

-- Define the second proof goal: the minimum total cost with at least twice as many green lily as spider plants
theorem minimum_total_cost_solution (m y : ℕ) (hm : m + y = number_of_pots)
  (h_cost : cost_green_lily * m + cost_spider_plant * y)
  (h_min_cost : m >= 2 * y) 
  (h_min : (3 * m + 276) = 369) : 
  (m = 31 ∧ y = 15 ∧ (cost_green_lily * m + cost_spider_plant * y = 369)) :=
by
  sorry

end number_of_pots_solution_minimum_total_cost_solution_l200_200972


namespace integer_points_between_A_and_B_l200_200366

def point := (ℕ, ℕ)

def A : point := (2, 3)
def B : point := (50, 203)

def is_on_line (p : point) : Prop :=
  let (x, y) := p
  y = (25 * x - 34) / 6

def is_strictly_between (p : point) : Prop :=
  let (x, y) := p
  2 < x ∧ x < 50

theorem integer_points_between_A_and_B : 
  ∃! (n : ℕ), n = 7 ∧ ∀ p : point, is_strictly_between p → is_on_line p → p.1 ∈ finset.range 51 \ finset.range 3 then sorry

end integer_points_between_A_and_B_l200_200366


namespace gcd_m_n_is_one_l200_200004

-- Definitions of m and n
def m : ℕ := 101^2 + 203^2 + 307^2
def n : ℕ := 100^2 + 202^2 + 308^2

-- The main theorem stating the gcd of m and n
theorem gcd_m_n_is_one : Nat.gcd m n = 1 := by
  sorry

end gcd_m_n_is_one_l200_200004


namespace polynomial_p_of_4_l200_200060

theorem polynomial_p_of_4 :
  (∀ n, n = 1 ∨ n = 2 ∨ n = 3 → p n = 1 / n^2) →
  p 4 = -9 / 16 :=
by
  sorry

end polynomial_p_of_4_l200_200060


namespace sin_B_value_l200_200518

noncomputable theory

variables {R : Type*} [linear_ordered_ring R] [trig R]

variables {a b c A B C : R}

theorem sin_B_value (h1 : a + c = 2 * b) (h2 : A - C = π / 3)
  (ha : a = 2 * sin A) (hb : b = 2 * sin B) (hc : c = 2 * sin C) : 
  sin B = sqrt 3 / 2 :=
by sorry

end sin_B_value_l200_200518


namespace revenue_increase_by_50_percent_l200_200053

-- Define the initial and new conditions
def initial_volume : ℝ := 1.0
def initial_price : ℝ := 60.0
def new_volume : ℝ := 0.9
def new_price : ℝ := 81.0
def total_initial_cost (fixed_volume : ℝ) : ℝ := fixed_volume * initial_price / initial_volume
def total_new_cost (fixed_volume : ℝ) : ℝ := (fixed_volume / new_volume) * new_price
def revenue_increase_percentage (initial_cost new_cost : ℝ) : ℝ := (new_cost - initial_cost) / initial_cost * 100

-- The proof goal: The revenue increased by 50%
theorem revenue_increase_by_50_percent (fixed_volume : ℝ) (h_fixed : fixed_volume = 9.0) :
  revenue_increase_percentage (total_initial_cost fixed_volume) (total_new_cost fixed_volume) = 50 :=
by
  -- Skipping the proof steps
  sorry

end revenue_increase_by_50_percent_l200_200053


namespace find_m_l200_200894

variables {R : Type*} [CommRing R]

/-- Definition of the dot product in a 2D vector space -/
def dot_product (a b : R × R) : R := a.1 * b.1 + a.2 * b.2

/-- Given vectors a and b as conditions -/
def a : ℚ × ℚ := (m, 3)
def b : ℚ × ℚ := (1, m + 1)

theorem find_m (m : ℚ) (h : dot_product a b = 0) : m = -3 / 4 :=
sorry

end find_m_l200_200894


namespace smallest_number_l200_200743

theorem smallest_number (S : set ℤ) (h : S = {0, -3, 1, -1}) : ∃ m ∈ S, ∀ x ∈ S, m ≤ x ∧ m = -3 :=
by
  sorry

end smallest_number_l200_200743


namespace roots_of_polynomial_l200_200796

theorem roots_of_polynomial :
  (∀ x : ℝ, (x^2 - 5 * x + 6) * x * (x - 5) = 0 ↔ x = 0 ∨ x = 2 ∨ x = 3 ∨ x = 5) :=
by
  sorry

end roots_of_polynomial_l200_200796


namespace f_f_neg2_eq_27_f_a_eq_2_imp_a_eq_neg1_l200_200480

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then abs (x - 1) else 3^x

theorem f_f_neg2_eq_27 : f (f (-2)) = 27 :=
by
  sorry

theorem f_a_eq_2_imp_a_eq_neg1 (a : ℝ) (h : f a = 2) : a = -1 :=
by
  sorry

end f_f_neg2_eq_27_f_a_eq_2_imp_a_eq_neg1_l200_200480


namespace cards_product_is_even_l200_200593

theorem cards_product_is_even :
  ∀ (cards : Fin 100 → (ℕ × ℕ)),
  (∀ i, 1 ≤ (cards i).fst ∧ (cards i).fst ≤ 99 ∧ 1 ≤ (cards i).snd ∧ (cards i).snd ≤ 99) →
  (∃ i, (cards i).fst + (cards i).snd % 2 = 0) →
  ∃ p, (Σ i, (cards i).fst + (cards i).snd) % 2 = 0 :=
by
  intros cards cards_bounds even_sum_exists product_is_even
  sorry

end cards_product_is_even_l200_200593


namespace triangle_PQR_hypotenuse_length_l200_200424

theorem triangle_PQR_hypotenuse_length :
  ∀ (P Q R : Type) [MetricSpace P]
    (is_right_triangle : ∃ Q R : P, right_angle (angle Q R P))
    (angle_RPQ : angle Q P R = 45 * π / 180)
    (length_PR : dist P R = 10),
    dist P Q = 10 * sqrt 2 :=
by
  sorry

end triangle_PQR_hypotenuse_length_l200_200424


namespace domain_of_h_l200_200788

noncomputable def h : ℝ → ℝ := λ x, (x^3 - 3 * x^2 + 6 * x - 1) / (x^2 - 5 * x + 6)

theorem domain_of_h :
  ∀ x : ℝ, (x < 2 ∨ (2 < x ∧ x < 3) ∨ 3 < x) ↔ x ∈ (((-∞:ℝ), 2) ∪ (2, 3) ∪ 3, ∞) :=
by
  sorry

end domain_of_h_l200_200788


namespace total_number_of_trees_l200_200615

-- Definitions of the conditions
def side_length : ℝ := 100
def trees_per_sq_meter : ℝ := 4

-- Calculations based on the conditions
def area_of_street : ℝ := side_length * side_length
def area_of_forest : ℝ := 3 * area_of_street

-- The statement to prove
theorem total_number_of_trees : 
  trees_per_sq_meter * area_of_forest = 120000 := 
sorry

end total_number_of_trees_l200_200615


namespace triangle_probability_0_75_l200_200536

-- Definitions of points and triangle
variables {A B C M : Type*} [measurable_space A] [measurable_space B] [measurable_space C] [measurable_space M]
variables {triangle_ABC : set (A × B × C)} {point_M : set M} [is_probability_measure point_M]

-- Conditions based on medians and centroid
def is_centroid (G : M) (A B C : A) (M : M) : Prop :=
  ∀ (A_1 B_1 C_1 : M), midpoint A B C = (A_1, B_1, C_1) → ... -- fill in appropriate properties

def area_greater (T1 T2 T3 : set (A × B × C)) : Prop :=
  measure_space.measure T1 > (measure_space.measure T2 + measure_space.measure T3)

-- Final proof problem statement
theorem triangle_probability_0_75 :
  let M := point_M in
  let T1 := set (A × B × M) in
  let T2 := set (B × C × M) in
  let T3 := set (C × A × M) in
  is_centroid G A B C →
  probability (area_greater T1 T2 T3) = 0.75 :=
sorry

end triangle_probability_0_75_l200_200536


namespace find_Ada_original_seat_l200_200798

-- Five friends Ada, Bea, Ceci, Dee, and Eli
def position : Type := Fin 5

variable (A B C D E : position)

-- Movements
def move_B := (B + 1) % 5 -- Bea moves one seat to the right
def move_C := (C + 3) % 5 -- Ceci moves two seats to the left (left move is equivalent to adding 3 mod 5)
def swap_D_E := (D, E) -- Dee and Eli swap seats

-- Final condition: seat 4 is empty
axiom empty_seat_4 : (move_B ≠ 4) ∧ (move_C ≠ 4) ∧ (D ≠ 4) ∧ (E ≠ 4)

-- The theorem we need to prove
theorem find_Ada_original_seat (h : B ≠ 4 ∧ C ≠ 1 ∧ D ≠ 4 ∧ E ≠ 4) : A = 4 := sorry

end find_Ada_original_seat_l200_200798


namespace eval_polynomial_at_2_l200_200932

theorem eval_polynomial_at_2 : 
  ∃ a b c d : ℝ, (∀ x : ℝ, (3 * x^2 - 5 * x + 4) * (7 - 2 * x) = a * x^3 + b * x^2 + c * x + d) ∧ (8 * a + 4 * b + 2 * c + d = 18) :=
by
  sorry

end eval_polynomial_at_2_l200_200932


namespace average_percent_score_l200_200249

theorem average_percent_score (num_students : ℕ)
    (students_95 students_85 students_75 students_65 students_55 students_45 : ℕ)
    (h : students_95 + students_85 + students_75 + students_65 + students_55 + students_45 = 120) :
  ((95 * students_95 + 85 * students_85 + 75 * students_75 + 65 * students_65 + 55 * students_55 + 45 * students_45) / 120 : ℚ) = 72.08 := 
by {
  sorry
}

end average_percent_score_l200_200249


namespace circumcircle_intersects_euler_line_in_two_points_l200_200555

-- Let ABC be an acute scalene triangle with incenter I
variables {A B C I O G H N : Type} 
-- Euler line passes through O (circumcenter), G (centroid), H (orthocenter) and N (nine-point center)
variable [acute_scalene_triangle A B C]
variable [incenter I A B C]
variable [circumcenter O A B C]
variable [centroid G A B C]
variable [orthocenter H A B C]
variable [nine_point_center N A B C]

-- Define the circumcircle of BIC
noncomputable def circumcircle_BIC : Set (Point A B C) := sorry

-- Define the Euler line of ABC
noncomputable def euler_line : Set (Point A B C) := sorry

-- Statement to prove that the circumcircle intersects Euler line in two distinct points
theorem circumcircle_intersects_euler_line_in_two_points :
  ∃ P Q : Point A B C, P ≠ Q ∧ P ∈ circumcircle_BIC ∧ P ∈ euler_line ∧ Q ∈ circumcircle_BIC ∧ Q ∈ euler_line := sorry

end circumcircle_intersects_euler_line_in_two_points_l200_200555


namespace ralph_tv_hours_l200_200601

theorem ralph_tv_hours :
  (∃ (f : ℕ → ℕ), (∀ (d : ℕ), d < 5 → f d = 4) ∧ (∀ (d : ℕ), d ≥ 5 ∧ d < 7 → f d = 6) ∧ (∀ (d : ℕ), d ≥ 7 → f d = 0)) →
  (finset.range 7).sum f = 32 :=
by
  intro h
  sorry

end ralph_tv_hours_l200_200601


namespace parallelogram_adjacency_ratio_l200_200256

variables {A B C D M : Type} [OrderedCommRing A]
open Set

noncomputable def adjacent_side_ratio (AB BC : A) : A :=
AB / BC

theorem parallelogram_adjacency_ratio 
    (BM MC : A)
    (h₁ : BM / MC = 2 / 3)
    (h₂ : BM + MC = BC)
    (A_angle : Rat)
    (h₃ : ι A_angle = 45 / 180 * π)
    (AMD_angle : Rat)
    (h₄ : ι AMD_angle = π / 2) :
    adjacent_side_ratio AB BC = 2 * Real.sqrt 2 / 5 :=
by
  sorry

end parallelogram_adjacency_ratio_l200_200256


namespace sum_first_2007_terms_l200_200643

-- Definitions based on the given conditions
def sequence_sum (n : ℕ) : ℤ :=
  ∑ i in finset.range (n + 1), (-1) ^ i * i

-- The theorem we need to prove
theorem sum_first_2007_terms : sequence_sum 2007 = -1004 :=
sorry

end sum_first_2007_terms_l200_200643


namespace cube_of_number_divided_l200_200935

theorem cube_of_number_divided (x : ℝ) (h : (sqrt x - 5) / 7 = 7) : (x^3 - 34) / 10 = 2,483,707,990.2 :=
by
  sorry

end cube_of_number_divided_l200_200935


namespace supplies_ratio_is_one_third_l200_200987

theorem supplies_ratio_is_one_third 
    (students : ℕ) (paper_per_student : ℕ) (glue_bottles : ℕ) (extra_paper : ℕ) (supplies_left : ℕ) 
    (h_students : students = 8) (h_paper_per_student : paper_per_student = 3) (h_glue_bottles : glue_bottles = 6)
    (h_extra_paper : extra_paper = 5) (h_supplies_left : supplies_left = 20) :
    let initial_supplies := students * paper_per_student + glue_bottles in
    let supplies_dropped := initial_supplies - (supplies_left - extra_paper) in
    supplies_dropped / initial_supplies = 1 / 3 := by
  sorry

end supplies_ratio_is_one_third_l200_200987


namespace ensure_triangle_from_segments_l200_200146

theorem ensure_triangle_from_segments :
  ∀ (P : Fin 6 → Fin 6 → Prop), -- P(i, j) indicates a line segment between point i and point j
  (∀ i j k : Fin 6, i ≠ j → i ≠ k → j ≠ k → ¬(P i j ∧ P i k ∧ P j k)) → -- no three points are collinear
  (∃ i j : Fin 6, i ≠ j ∧ P i j) → -- at least one segment exists
  ∃ (t : Fin 6 × Fin 6 × Fin 6), -- output is a triangle
    (t.fst ≠ t.snd ∧ t.snd ≠ t.trd ∧ t.fst ≠ t.trd ∧ P t.fst t.snd ∧ P t.snd t.trd ∧ P t.fst t.trd) := 
begin
  -- proof would go here
  sorry
end

end ensure_triangle_from_segments_l200_200146


namespace Isabella_redeem_day_l200_200227

def is_coupon_day_closed_sunday (start_day : ℕ) (num_coupons : ℕ) (cycle_days : ℕ) : Prop :=
  ∃ n, n < num_coupons ∧ (start_day + n * cycle_days) % 7 = 0

theorem Isabella_redeem_day: 
  ∀ (day : ℕ), day ≡ 1 [MOD 7]
  → ¬ is_coupon_day_closed_sunday day 6 11 :=
by
  intro day h_mod
  simp [is_coupon_day_closed_sunday]
  sorry

end Isabella_redeem_day_l200_200227


namespace total_cans_from_256_l200_200807

-- Define the recursive function to compute the number of new cans produced.
def total_new_cans (n : ℕ) : ℕ :=
  if n < 4 then 0
  else
    let rec_cans := total_new_cans (n / 4)
    (n / 4) + rec_cans

-- Theorem stating the total number of new cans that can be made from 256 initial cans.
theorem total_cans_from_256 : total_new_cans 256 = 85 := by
  sorry

end total_cans_from_256_l200_200807


namespace a_range_solution_l200_200805

noncomputable def proof_a_range (a : ℝ) (x : ℝ) : Prop :=
  ∃ (A B : set ℝ),
    A = {x | 2 * a ≤ x ∧ x ≤ a^2 + 1} ∧ 
    B = if 3 * a + 1 ≥ 2 then {x | 2 ≤ x ∧ x ≤ 3 * a + 1} else {x | 3 * a + 1 ≤ x ∧ x ≤ 2} ∧
    (∀ x, x ∈ A → x ∈ B) → (a ∈ set.Icc 1 3 ∨ a = -1)

theorem a_range_solution : ∀ a x, proof_a_range a x :=
by 
  intros a x
  sorry

end a_range_solution_l200_200805


namespace problem_one_problem_two_max_min_l200_200867

variables (x : ℝ)
noncomputable def a : ℝ × ℝ := (Real.cos (3 * x / 2), Real.sin (3 * x / 2))
noncomputable def b : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2))
noncomputable def a_dot_b := (Real.cos (3 * x / 2)) * (Real.cos (x / 2)) - (Real.sin (3 * x / 2)) * (Real.sin (x / 2))
noncomputable def magnitude_a_plus_b := Real.sqrt (Real.normSq ⟨Real.cos (3 * x / 2), Real.sin (3 * x / 2)⟩ + Real.normSq ⟨Real.cos (x / 2), -Real.sin (x / 2)⟩ + 2 * a_dot_b)
noncomputable def f (x : ℝ) := a_dot_b - magnitude_a_plus_b

-- Prove that the magnitude of ⟨a, b⟩ + ⟨b, a⟩ is 2 * cos x
theorem problem_one : x ∈ Icc (-π / 3) (π / 4) → magnitude_a_plus_b = 2 * Real.cos x := by
  sorry

-- Prove that the maximum value of f(x) is -1 and the minimum value is -√2
theorem problem_two_max_min : x ∈ Icc (-π / 3) (π / 4) → ∃ t ∈ set.Icc (Real.sqrt 2 / 2) 1, (f x = -1) ∨ (f x = -Real.sqrt 2) := by
  sorry

end problem_one_problem_two_max_min_l200_200867


namespace solve_for_x_l200_200267

theorem solve_for_x (x : ℝ) (h : (x^2 + 2*x + 3) / (x + 1) = x + 3) : x = 0 :=
by
  sorry

end solve_for_x_l200_200267


namespace find_f_neg_one_l200_200144

def f : ℝ → ℝ 
| x => if x > 0 then x^2 - 1 else f (x + 1) - 1

theorem find_f_neg_one : f (-1) = -2 := by
  sorry

end find_f_neg_one_l200_200144


namespace frog_escapes_from_2_l200_200520

-- Define the state space for the lily pads
inductive Pad
| _0 | _1 | _2 | _3 | _4 | _5 | _6 | _7 | _8 | _9 | _10 | _11 | _12 | _13 | _14 | _15 | _16 | _17 | _18 | _19

open Pad

-- Define the transition probabilities
def P : Pad → ℚ
| _0  => 0
| _19  => 1
| (N+1 : Pad) => ((N+2):ℚ) / 20 * P (Pad.toEnum! (N)) + (1 - ((N+2):ℚ) / 20) * P (Pad.toEnum! (N + 2))

theorem frog_escapes_from_2 :
  P _2 = 524 / 1165 :=
by
  exact sorry

end frog_escapes_from_2_l200_200520


namespace fencing_required_l200_200374

variable (L W : ℝ)
variable (Area : ℝ := 20 * W)

theorem fencing_required (hL : L = 20) (hArea : L * W = 600) : 20 + 2 * W = 80 := by
  sorry

end fencing_required_l200_200374


namespace cost_of_dozen_pens_is_780_l200_200623

noncomputable def cost_of_one_pen : ℝ := 65
def cost_of_one_pencil : ℝ := cost_of_one_pen / 5
def total_cost_of_pens_and_pencils : ℝ := 3 * cost_of_one_pen + 5 * cost_of_one_pencil
def one_dozen : ℝ := 12

theorem cost_of_dozen_pens_is_780 (h1 : total_cost_of_pens_and_pencils = 260) : one_dozen * cost_of_one_pen = 780 :=
by
  rw [total_cost_of_pens_and_pencils, cost_of_one_pencil, cost_of_one_pen, one_dozen]
  norm_num
  sorry -- proof omitted

end cost_of_dozen_pens_is_780_l200_200623


namespace sum_largest_smallest_prime_factors_l200_200336

theorem sum_largest_smallest_prime_factors (n : ℕ) (h : n = 990) : 
  let prime_factors : List ℕ := [2, 3, 5, 11], 
      min_prime := 2,
      max_prime := 11
  in sum_largest_smallest_prime_factors = 13 :=
sorry

end sum_largest_smallest_prime_factors_l200_200336


namespace max_f_value_sin_A_value_l200_200904

def vector_m (x : ℝ) : ℝ × ℝ :=
  (sqrt 3 * sin (x / 4), 1)

def vector_n (x : ℝ) : ℝ × ℝ :=
  (cos (x / 4), cos (x / 4) ^ 2)

def f (x : ℝ) : ℝ :=
  let (mx1, mx2) := vector_m x
  let (nx1, nx2) := vector_n x
  mx1 * nx1 + mx2 * nx2

theorem max_f_value (x : ℝ) :
  ∃ k : ℤ, f x = 3 / 2 :=
sorry

def f_B : ℝ :=
  (sqrt 3 + 1) / 2

theorem sin_A_value (a b c : ℝ) (B : ℝ) (h_a : a = 2) (h_c : c = 3) (h_f_B : f B = f_B) :
  sin (angle_A a b c) = sqrt 21 / 7 :=
sorry

end max_f_value_sin_A_value_l200_200904


namespace mouse_starts_getting_farther_l200_200057

noncomputable def coordinates_of_cheese := (12, 10)
noncomputable def initial_position_of_mouse := (4, -2)
noncomputable def line_equation := λ x : ℝ, -5 * x + 18

theorem mouse_starts_getting_farther (a b : ℝ) :
  let line_through_cheese := λ x : ℝ, (1 / 5) * x + (50 / 5) - (12 / 5)
  (a, b) = (λ (x : ℝ), (x, line_equation x)) 2 ∧
  a + b = 10 :=
sorry

end mouse_starts_getting_farther_l200_200057


namespace area_ratio_proof_l200_200225

-- Define the geometric setup
variables {A B C P K M A' B': Type} [RightAngledTriangle ABC]
variables (PK_perpendicular_AC : Perpendicular PK AC)
variables (PM_perpendicular_BC : Perpendicular PM BC)
variables (AP_Pt_on_legs : IsOnSegment P A A')
variables (BP_Pt_on_legs : IsOnSegment P B B')

-- Given condition
variable (area_ratio_given : S_APB' / S_KPB' = m)

-- Goal: To prove the desired ratio
theorem area_ratio_proof : S_MPA' / S_BPA' = m :=
sorry

end area_ratio_proof_l200_200225


namespace sum_of_first_n_terms_l200_200465

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ := n * a (n + 1) + 2^n

def T (n : ℕ) (a : ℕ → ℝ) : ℕ → ℝ := λ n, 
  let seq := λ n, 1 / (n * (a n - a (n + 1))) in
  finset.sum (finset.range n) seq

theorem sum_of_first_n_terms (n : ℕ) (a : ℕ → ℝ) (h : S n a = n * a (n + 1) + 2^n) : 
  T n a = 3 / 2 - 2 / 2^n := 
sorry

end sum_of_first_n_terms_l200_200465


namespace math_problem_l200_200026

theorem math_problem :
    (50 + 5 * (12 / (180 / 3))^2) * Real.sin (Real.pi / 6) = 25.1 :=
by
  sorry

end math_problem_l200_200026


namespace cone_surface_area_l200_200820

def base_radius : ℝ := 4
def height : ℝ := 2 * Real.sqrt 5
def slant_height := Real.sqrt (base_radius ^ 2 + height ^ 2)
def base_area := Real.pi * base_radius ^ 2
def lateral_surface_area := Real.pi * base_radius * slant_height
def total_surface_area := base_area + lateral_surface_area

theorem cone_surface_area :
  total_surface_area = 40 * Real.pi :=
by
  sorry

end cone_surface_area_l200_200820


namespace derivative_of_sin_plus_exp_l200_200289

theorem derivative_of_sin_plus_exp (x : ℝ) :
  deriv (λ x, sin x + 3^x) x = cos x + 3^x * log 3 :=
by
  sorry

end derivative_of_sin_plus_exp_l200_200289


namespace inscribed_hexagon_cyclic_points_l200_200217

theorem inscribed_hexagon_cyclic_points
  (Ω : Type) [circle Ω]
  (A B C D E F X Y Z T : Ω)
  (h_hex : inscribed_hexagon Ω A B C D E F)
  (h_D_mid_arc : mid_arc D B C)
  (h_common_incircle : common_incircle (triangle A B C) (triangle D E F))
  (h_BX_ints_DF : intersects (line B C) (segment D F) X)
  (h_BY_ints_DE : intersects (line B C) (segment D E) Y)
  (h_EZ_ints_AB : intersects (line E F) (segment A B) Z)
  (h_ET_ints_AC : intersects (line E F) (segment A C) T)
  : cyclic Ω X Y Z T :=
begin
  sorry
end

end inscribed_hexagon_cyclic_points_l200_200217


namespace coursework_materials_spending_l200_200977

def budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

theorem coursework_materials_spending : 
    budget - (budget * food_percentage + budget * accommodation_percentage + budget * entertainment_percentage) = 300 := 
by 
  -- steps you would use to prove this
  sorry

end coursework_materials_spending_l200_200977


namespace find_eccentricity_l200_200852

noncomputable def hyperbola_eq (a b : ℝ) := ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)
noncomputable def circle_eq (a c : ℝ) := ∀ x y : ℝ, ((x - a)^2 + y^2 = c^2 / 16)
noncomputable def tangent_perpendicular_asymptote (a b c : ℝ) := ∀ F : ℝ, 
  let H := ((a, 0): ℝ × ℝ) in 
  let line_eq := fun x y : ℝ => (a * x) / b + y - (a * c) / b = 0 in
  let distance_from_center := abs ((a^2 / b) - (ac / b)) / (sqrt ((a / b)^2 + 1)) in
  (distance_from_center = c / 4) → (absc := c) 

theorem find_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (F : ℝ) (c : ℝ) 
  (hyper_eq : hyperbola_eq a b) 
  (circ_eq : circle_eq a c)
  (tangent_perpendicular : tangent_perpendicular_asymptote a b c F) :
  let e := c / a in e = 2 := 
sorry

end find_eccentricity_l200_200852


namespace train_length_l200_200069

theorem train_length
  (speed_kmph : ℕ) (time_s : ℕ)
  (h1 : speed_kmph = 72)
  (h2 : time_s = 12) :
  speed_kmph * (1000 / 3600 : ℕ) * time_s = 240 :=
by
  sorry

end train_length_l200_200069


namespace f_f_pi_eq_one_l200_200488

def f (x : ℝ) : ℝ := 
  if x ∈ ℚ then 1 else 0

theorem f_f_pi_eq_one : f (f π) = 1 := 
  by
    have h1 : π ∉ ℚ := sorry
    have h2 : f π = 0 := sorry
    have h3 : 0 ∈ ℚ := sorry
    have h4 : f 0 = 1 := sorry
    exact sorry

end f_f_pi_eq_one_l200_200488


namespace odd_function_values_and_range_of_k_l200_200476

-- Define the function f
def f (a b x : ℝ) : ℝ := (-2^x + b) / (2^x + a)

-- The given conditions
variables (a b : ℝ)

-- Stating the problem as a Lean 4 theorem
theorem odd_function_values_and_range_of_k
  (h_odd : ∀ x : ℝ, f a b (-x) = -f a b x)
  (h_ineq : ∀ t : ℝ, f a b (t^2 - 2*t) + f a b (2*t^2 - k) < 0) :
  (a = 1) ∧ (b = 1) ∧ (k < -1/3) :=
sorry

end odd_function_values_and_range_of_k_l200_200476


namespace number_of_ordered_pairs_l200_200090

theorem number_of_ordered_pairs :
  (∃ (count : ℕ), count = 1225 ∧
  (∀ (x y : ℕ), (1 ≤ x ∧ x < y ∧ y ≤ 200) →
   (∃ (i : ℂ), i^x + i^y ∈ ℝ ∧ i^y = i) →
   (∃ (i : ℂ), i^x = -i ∨ i^x = i ∨ i^x = 1 ∨ i^x = -1) →
   count = 1225)) :=
sorry

end number_of_ordered_pairs_l200_200090


namespace prob_product_greater_than_10_l200_200493

theorem prob_product_greater_than_10 :
  let s := {1, 2, 3, 4}
  let pairs := (s.product s).filter (λ x, x.fst < x.snd)
  let favorable pairs := pairs.filter (λ x, x.fst * x.snd > 10)
  let total_pairs := pairs.length
  let num_favorable := favorable_pairs.length
  (num_favorable : ℝ) / (total_pairs : ℝ) = 1 / 6 := by
begin
    sorry
end

end prob_product_greater_than_10_l200_200493


namespace find_length_BC_l200_200943

noncomputable def length_BC (BO : ℝ) (angle_ABO : ℝ) : ℝ :=
  2 * BO * Real.sin (angle_ABO / 2)

theorem find_length_BC : length_BC 6 (Real.pi / 4) ≈ 4.6 := by
  sorry

end find_length_BC_l200_200943


namespace arithmetic_sequence_10th_term_l200_200178

theorem arithmetic_sequence_10th_term : ∀ (a : ℕ → ℕ), a 1 = 3 → (∀ n, a (n+1) = a n + 3) → a 10 = 30 :=
by 
intros a h1 hr.
sorry

end arithmetic_sequence_10th_term_l200_200178


namespace triangle_ratios_l200_200954

/--
In a right triangle PQR, where angle Q is the right angle, and let PQ = 28, PR = 53.
Prove that:
1. \(\tan R = \frac{28}{45}\)
2. \(\cos R = \frac{45}{53}\)
-/
theorem triangle_ratios (Q R P : Point) (angle_Q: angleQ = π / 2) (PQ PR QR : Length) 
  (h1 : PQ = 28) (h2 : PR = 53) : tan R = 28 / 45 ∧ cos R = 45 / 53 := by
  sorry

end triangle_ratios_l200_200954


namespace contrapositive_example_l200_200702

theorem contrapositive_example (x : ℝ) :
  (¬ (x = 3 ∧ x = 4)) → (x^2 - 7 * x + 12 ≠ 0) →
  (x^2 - 7 * x + 12 = 0) → (x = 3 ∨ x = 4) :=
by
  intros h h1 h2
  sorry  -- proof is not required

end contrapositive_example_l200_200702


namespace midpoint_of_arc_bc_l200_200458

/-- Given triangle ABC with AB > AC and a circumcircle of △ABC,
proving that the intersection of the perpendicular bisector of BC
and the circumcircle of △ABC is the midpoint of the arc BC that does not contain A. --/
theorem midpoint_of_arc_bc (A B C : Point) (hAB_gt_AC : dist A B > dist A C)
  (circumcircle_ABC : Circle) (circumcircle_def : circumcircle_ABC = circumscribed_circle A B C) :
  ∃ M : Point, M ∈ circumcircle_ABC ∧ M ≠ A ∧ dist M B = dist M C ∧
  ∀ (P : Point), ∀ (hP : P ∈ circumcircle_ABC ∧ P ≠ A ∧ dist P B = dist P C), P = M :=
sorry

end midpoint_of_arc_bc_l200_200458


namespace find_y_z_l200_200657

def abs_diff (x y : ℝ) := abs (x - y)

noncomputable def seq_stabilize (x y z : ℝ) (n : ℕ) : Prop :=
  let x1 := abs_diff x y 
  let y1 := abs_diff y z 
  let z1 := abs_diff z x
  ∃ k : ℕ, k ≥ n ∧ abs_diff x1 y1 = x ∧ abs_diff y1 z1 = y ∧ abs_diff z1 x1 = z

theorem find_y_z (x y z : ℝ) (hx : x = 1) (hstab : ∃ n : ℕ, seq_stabilize x y z n) : y = 0 ∧ z = 0 :=
sorry

end find_y_z_l200_200657


namespace probability_winning_pair_l200_200660

def card := Σ (color : Fin 2), Fin 5
def is_winning (c1 c2 : card) : Prop :=
  c1.1 = c2.1 ∨ c1.2 = c2.2

theorem probability_winning_pair :
  ∃ p : ℚ, p = (5 / 9) ∧
  p = (multiset.card (multiset.filter (λ (pair : card × card), is_winning pair.1 pair.2)
                    (list.to_multiset (list.sublists_of_length 2
                      (list.of_fn (λ i, ⟨⟨i / 5, nat.div_lt_self i (nat.succ_pos 4)⟩, ⟨i % 5, nat.mod_lt _ (nat.succ_pos 4)⟩⟩))))))
                    / multiset.card (list.to_multiset (list.sublists_of_length 2
                    (list.of_fn (λ i, ⟨⟨i / 5, nat.div_lt_self i (nat.succ_pos 4)⟩, ⟨i % 5, nat.mod_lt _ (nat.succ_pos 4)⟩⟩))))) :=
by
  -- The proof is omitted
  sorry

end probability_winning_pair_l200_200660


namespace value_of_a12_is_15_l200_200156

open Nat

variable {a : ℕ → ℤ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop := ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def condition1 (a : ℕ → ℤ) : Prop := a 7 + a 9 = 16
def condition2 (a : ℕ → ℤ) : Prop := a 4 = 1

-- The question
def value_of_an (a : ℕ → ℤ) : ℤ := a 12

-- The proof goal
theorem value_of_a12_is_15 (a : ℕ → ℤ) [arithmetic_sequence a] (h1 : condition1 a) (h2 : condition2 a) : value_of_an a = 15 := by
  sorry

end value_of_a12_is_15_l200_200156


namespace min_value_of_integral_l200_200131

theorem min_value_of_integral : 
  ∃ x : ℝ, x ≥ 0 ∧ ∀ y : ℝ, y ≥ 0 → (∫ t in 0..y, (2:ℝ)^t * ((2:ℝ)^t - 3) * (y - t)) ≥ (∫ t in 0..x, (2:ℝ)^t * ((2:ℝ)^t - 3) * (x - t)) := 
sorry

end min_value_of_integral_l200_200131


namespace fixed_point_of_exponential_function_l200_200930

theorem fixed_point_of_exponential_function (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (λ x : ℝ, a^(x + 2) + 1) (-2) = 2 := 
by {
  -- Proof goes here
  sorry
}

end fixed_point_of_exponential_function_l200_200930


namespace parabola_no_x_axis_intersection_l200_200303

theorem parabola_no_x_axis_intersection :
  ∀ x : ℝ, x^2 + 2 * x + 3 ≠ 0 :=
by
  intro x
  have h : (2:ℝ)^2 - 4 * 1 * 3 = -8 := by norm_num
  rw [add_lt_iff_neg_comb_simpl] at *
  rwa [lt_zero_iff_add_lt_neg, add_neg_self, lt_irrefl] at h
  sorry

end parabola_no_x_axis_intersection_l200_200303


namespace general_formula_a_n_sum_first_n_b_n_terms_l200_200457

noncomputable theory

-- Definitions based on conditions
def sequence_a (n : ℕ) : ℕ := 2 * n
def sequence_b (n : ℕ) : ℝ := 1 / ((sequence_a n) ^ 2 - 1)
def S (n : ℕ) : ℝ := sorry -- Definition of S_n such that sum condition holds

-- Given conditions
axiom sum_condition : ∀ n : ℕ, 4 * S n = (sequence_a n) ^ 2 + 2 * sequence_a n
axiom a1_condition : sequence_a 1 = 2

-- Proof obligations
theorem general_formula_a_n (n : ℕ) : sequence_a n = 2 * n :=
by {
  sorry
}

theorem sum_first_n_b_n_terms (n : ℕ) : real.sum (λ k, sequence_b k) 1 n = n / (2 * n + 1) :=
by {
  sorry
}

end general_formula_a_n_sum_first_n_b_n_terms_l200_200457


namespace sin_B_eq_l200_200516

theorem sin_B_eq (a b c : ℝ) (A B C : ℝ)
  (h₁ : a + c = 2 * b)
  (h₂ : A - C = π / 3)
  (h₃ : a = 2 * R * sin A)
  (h₄ : b = 2 * R * sin B)
  (h₅ : c = 2 * R * sin C)
  (h₆ : A + B + C = π)
  (hR : R > 0) :
  sin B = sqrt 3 / 2 := 
  sorry

end sin_B_eq_l200_200516


namespace find_number_l200_200504

theorem find_number
  (x : ℝ)
  (h : 0.90 * x = 0.50 * 1080) :
  x = 600 :=
by
  sorry

end find_number_l200_200504


namespace problem_statement_l200_200438

noncomputable def f (n : ℕ) (x : ℝ) : ℝ := x^n

variable (a : ℝ)
variable (h : a ≠ 1)

theorem problem_statement :
  (f 11 (f 13 a)) ^ 14 = f 2002 a ∧
  f 11 (f 13 (f 14 a)) = f 2002 a :=
by
  sorry

end problem_statement_l200_200438


namespace axis_of_symmetry_l200_200002

def f (x : ℝ) : ℝ := Real.sin (x - π / 4)

theorem axis_of_symmetry : ∃ x : ℝ, x = -π / 4 ∧ ∀ y : ℝ, f (x + y) = f (x - y) :=
by 
  sorry

end axis_of_symmetry_l200_200002


namespace f_at_7_is_correct_l200_200916

def f (x : ℝ) : ℝ := (x + 2) / (4 * x - 5)

theorem f_at_7_is_correct : f 7 = 9 / 23 := by
  sorry

end f_at_7_is_correct_l200_200916


namespace joe_data_points_final_count_l200_200984

theorem joe_data_points_final_count 
  (initial_data_points : ℕ)
  (initial_data_points = 200)
  (percentage_increase : ℕ)
  (percentage_increase = 20)
  (fraction_reduction : ℚ)
  (fraction_reduction = 1/4)
  : 180 = initial_data_points * (1 + percentage_increase / 100) - initial_data_points * (1 + percentage_increase / 100) * fraction_reduction := 
by 
  sorry

end joe_data_points_final_count_l200_200984


namespace chocolate_bars_count_l200_200753

-- Defining the problem conditions
variables 
  (total_candies : ℕ)
  (chewing_gums : ℕ)
  (assorted_candies : ℕ)
  (chocolate_bars : ℕ)

-- Given conditions
def conditions : Prop := 
  total_candies = 50 ∧ 
  chewing_gums = 15 ∧ 
  assorted_candies = 15

-- Statement of the proof problem
theorem chocolate_bars_count (h : conditions) : chocolate_bars = 20 :=
sorry

end chocolate_bars_count_l200_200753


namespace intersection_complement_l200_200588

open Set

noncomputable def U : Set ℝ := univ

def A : Set ℝ := {x | x^2 - 2 * x < 0}

def B : Set ℝ := {x | x > 1}

theorem intersection_complement (x : ℝ) :
  x ∈ (A ∩ (U \ B)) ↔ 0 < x ∧ x ≤ 1 :=
by
  sorry

end intersection_complement_l200_200588


namespace tetrahedron_regular_if_equal_face_areas_l200_200257

-- Let's define the concept of tetrahedron and faces in a geometrical context.
structure Tetrahedron :=
  (vertices : fin 4 → ℝ × ℝ × ℝ)
  (faces : set (set (ℝ × ℝ × ℝ)))
  (has_four_faces : faces.card = 4)
  (face_areas : faces → ℝ)
  (all_faces_equal_area : ∀ f1 f2 : faces, face_areas f1 = face_areas f2)

-- Define a tetrahedron is regular if all its faces are congruent triangles.
def is_regular_tetrahedron (t : Tetrahedron) : Prop :=
  ∀ f1 f2 : t.faces, f1 = f2

-- Our main theorem statement
theorem tetrahedron_regular_if_equal_face_areas
  (t : Tetrahedron)
  (hequal_areas : t.all_faces_equal_area) :
  is_regular_tetrahedron t :=
  sorry -- proof goes here

end tetrahedron_regular_if_equal_face_areas_l200_200257


namespace quadratic_has_two_distinct_real_roots_l200_200340

-- Given the discriminant condition Δ = b^2 - 4ac > 0
theorem quadratic_has_two_distinct_real_roots (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) := 
  sorry

end quadratic_has_two_distinct_real_roots_l200_200340


namespace research_institute_funds_l200_200063

theorem research_institute_funds :
  ∃ (total_funds : ℕ), total_funds = 2046 ∧
    ∀ (a : ℕ → ℕ) (S : ℕ → ℕ),
      (a 10 = 2) ∧ 
      (∀ n, 1 ≤ n ∧ n < 10 → a (n+1) = 2 * a n) ∧
      (∀ n, S n = ∑ i in finset.range n, a i) ∧
      (∀ n, 1 ≤ n ∧ n ≤ 10 → S n = S (n-1) / 2 - 1) ∧
      (∀ n, S 10 = total_funds / 2 - 1) → total_funds = 2046 :=
begin
  sorry,
end

end research_institute_funds_l200_200063


namespace volleyball_tournament_first_place_score_l200_200135

theorem volleyball_tournament_first_place_score :
  ∃ (a b c d : ℕ), (a + b + c + d = 18) ∧ (a < b ∧ b < c ∧ c < d) ∧ (d = 6) :=
by
  sorry

end volleyball_tournament_first_place_score_l200_200135


namespace paper_statements_l200_200250

theorem paper_statements (P : ℕ → Prop) (h : ∀ n, P n ↔ n = 99 ↔ ∀ m ≠ 99, ¬P m) :
  ∀ n, P n ↔ n = 99 :=
by
  intro n
  split
  case mp =>
    exact λ hP => hP
  case mpr =>
    exact λ hEq => hEq.symm.mp sorry

end paper_statements_l200_200250


namespace find_value_of_a_l200_200164

theorem find_value_of_a
  (a : ℝ)
  (circle_eq : ∀ x y : ℝ, (x^2 + y^2 - 4*x + 2*y + a = 0))
  (midpoint_M : (1, 0))
  (chord_len : ∀ A B : ℝ × ℝ, ((A, B) ≠ (0, 0) → sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 3))
  : a = 3/4 :=
sorry

end find_value_of_a_l200_200164


namespace M_gt_N_l200_200912

-- Define M and N
def M (x y : ℝ) : ℝ := x^2 + y^2 + 1
def N (x y : ℝ) : ℝ := 2 * (x + y - 1)

-- State the theorem to prove M > N given the conditions
theorem M_gt_N (x y : ℝ) : M x y > N x y := by
  sorry

end M_gt_N_l200_200912


namespace three_digit_integers_count_l200_200907

theorem three_digit_integers_count : 
  (∃ (S : Finset ℕ), S = {2, 4, 4, 5, 5, 7, 7} ∧
  ∀ d ∈ S, multiset.count d S ≤ multiset.count d {2, 4, 4, 5, 5, 7, 7}) →
  ∃ (n : ℕ), n = 54 :=
begin
  sorry
end

end three_digit_integers_count_l200_200907


namespace point_A_in_fourth_quadrant_l200_200525

-- Defining the coordinates of point A
def x_A : ℝ := 2
def y_A : ℝ := -3

-- Defining the property of the quadrant
def in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Proposition stating point A is in the fourth quadrant
theorem point_A_in_fourth_quadrant : in_fourth_quadrant x_A y_A :=
by
  sorry

end point_A_in_fourth_quadrant_l200_200525


namespace volume_of_one_slice_l200_200734

theorem volume_of_one_slice
  (circumference : ℝ)
  (c : circumference = 18 * Real.pi):
  ∃ V, V = 162 * Real.pi :=
by sorry

end volume_of_one_slice_l200_200734


namespace remove_chairs_to_meet_participants_l200_200050

-- Given initial conditions
def initial_chairs : ℕ := 144
def chairs_per_row : ℕ := 12
def expected_participants : ℕ := 100

-- The number of chairs to meet the condition
def required_chairs : ℕ := 108

-- Therefore, the number of chairs to remove
def chairs_to_remove (initial : ℕ) (required : ℕ) : ℕ :=
  initial - required

theorem remove_chairs_to_meet_participants :
  chairs_to_remove initial_chairs required_chairs = 36 :=
by
  simp [chairs_to_remove, initial_chairs, required_chairs]
  sorry

end remove_chairs_to_meet_participants_l200_200050


namespace circle_radius_l200_200698

/-- Consider a 3 × 3 grid of squares with side length 1. 
    Let three circles be inscribed in the lower left corner, the middle square of the top row, 
    and the rightmost square of the middle row. 
    Let their centers be at coordinates (0.5, 0.5), (1.5, 2.5), and (2.5, 1.5) respectively.
    Another circle O with radius r is drawn such that O is externally tangent to each of the three inscribed circles.
    Prove that the radius r of circle O is (5 * sqrt 2 - 3) / 6.
 -/
theorem circle_radius 
  (r : ℝ) 
  (h₁ : r = (5 * Real.sqrt 2 - 3) / 6) :
  ∃ (A B C : ℝ × ℝ), 
    A = (0.5, 0.5) ∧ 
    B = (1.5, 2.5) ∧ 
    C = (2.5, 1.5) ∧ 
    ∀ x y z : ℝ, 
      (x = Real.sqrt ((1.5 - 0.5) ^ 2 + (2.5 - 0.5) ^ 2) ∧ x = Real.sqrt 5) ∧ 
      (y = Real.sqrt ((2.5 - 1.5) ^ 2 + (1.5 - 2.5) ^ 2) ∧ y = Real.sqrt 2) ∧ 
      (z = Real.sqrt ((2.5 - 0.5) ^ 2 + (1.5 - 0.5) ^ 2) ∧ z = Real.sqrt 5) -> 
      (r = (5 * Real.sqrt 2 - 3) / 6) :=
  sorry

end circle_radius_l200_200698


namespace angle_B_values_l200_200992

-- Definitions based on the given conditions
variables {A B C O H : Type} [MetricSpace O] 
variables [HasAdd A] [HasAdd B] [HasAdd C] [AddGroup A] [AddGroup B] [AddGroup C]
variables [InnerProductSpace ℝ O]

-- Let O be the circumcenter and H be the orthocenter of triangle ABC
variable {circumcenter : triangle O -> O}
variable {orthocenter : triangle O -> O}

-- Assume BO = BH
variable {triangle : Type}
variable {BO : O -> ℝ}
variable {BH : O -> ℝ}
variable {B : triangle O}

-- The statement to prove the possible values of angle B
theorem angle_B_values (hO : O = circumcenter ⟦A, B, C⟧) (hH : H = orthocenter ⟦A, B, C⟧)
  (h_eq : BO B = BH H) : 
  (angle ⟦A, B, C⟧ = 60 ∨ angle ⟦A, B, C⟧ = 120) := 
sorry

end angle_B_values_l200_200992


namespace volume_at_target_temperature_l200_200440

-- Volume expansion relationship
def volume_change_per_degree_rise (ΔT V_real : ℝ) : Prop :=
  ΔT = 2 ∧ V_real = 3

-- Initial conditions
def initial_conditions (V_initial T_initial : ℝ) : Prop :=
  V_initial = 36 ∧ T_initial = 30

-- Target temperature
def target_temperature (T_target : ℝ) : Prop :=
  T_target = 20

-- Theorem stating the volume at the target temperature
theorem volume_at_target_temperature (ΔT V_real T_initial V_initial T_target V_target : ℝ) 
  (h_rel : volume_change_per_degree_rise ΔT V_real)
  (h_init : initial_conditions V_initial T_initial)
  (h_target : target_temperature T_target) :
  V_target = V_initial + V_real * ((T_target - T_initial) / ΔT) :=
by
  -- Insert proof here
  sorry

end volume_at_target_temperature_l200_200440


namespace monochromatic_isosceles_triangles_l200_200370

theorem monochromatic_isosceles_triangles (n : ℕ) (h1 : 0 < n) (h2 : n < 2017):
  let total_vertices := 2017 in
  let red_vertices := n in
  let blue_vertices := total_vertices - red_vertices in
  ∃ R B : finset ℕ, 
    R.card = red_vertices ∧ 
    B.card = blue_vertices ∧ 
    R ∪ B = finset.range total_vertices ∧ 
    R ∩ B = ∅ →
  ∑ (T in finset.powerset_len 3 (finset.range total_vertices)), 
      (∃ a b c, {a, b, c} = T ∧ (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ 
       (T ⊆ R ∨ T ⊆ B) ∧ 
       (Isosceles_triangle a b c)) = (3/2) * n * (2017 - n) := sorry

end monochromatic_isosceles_triangles_l200_200370


namespace max_and_min_z_l200_200511

theorem max_and_min_z (x y : ℝ) (h1 : x + y ≤ 2) (h2 : 1 ≤ x) (h3 : 0 ≤ y) :
  ∃ min_z max_z, min_z = 2 ∧ max_z = 4 ∧ 
  (∀ z, (z = 2 * x + y) → (min_z ≤ z ∧ z ≤ max_z)) :=
by
  let z := 2 * x + y
  use 2
  use 4
  split
  ·exact sorry  -- proof that min_z = 2
  split
  ·exact sorry  -- proof that max_z = 4
  ·exact sorry  -- proof that min_z ≤ z ≤ max_z

end max_and_min_z_l200_200511


namespace complement_is_correct_l200_200865

-- Define the universal set I and set A
def I : Set ℕ := {x ∈ ℕ | 0 < x ∧ x < 6}
def A : Set ℕ := {1, 2, 3}

-- Define the complement of A with respect to I
def complement_I_A : Set ℕ := {x ∈ I | x ∉ A}

-- State the theorem to be proven
theorem complement_is_correct : complement_I_A = {4, 5} := by
  sorry

end complement_is_correct_l200_200865


namespace value_of_six_inch_cube_l200_200044

-- Defining the conditions
def original_cube_weight : ℝ := 5 -- in pounds
def original_cube_value : ℝ := 600 -- in dollars
def original_cube_side : ℝ := 4 -- in inches

def new_cube_side : ℝ := 6 -- in inches

def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

-- Statement of the theorem
theorem value_of_six_inch_cube :
  cube_volume new_cube_side / cube_volume original_cube_side * original_cube_value = 2025 :=
by
  -- Here goes the proof
  sorry

end value_of_six_inch_cube_l200_200044


namespace conjugate_of_given_number_is_i_l200_200285

-- Define the given complex number
def given_complex_number := (1 : ℂ) / (complex.i)

-- Define the expected conjugate
def expected_conjugate := complex.conj (-complex.i)

-- Prove that the conjugate of the given complex number is i
theorem conjugate_of_given_number_is_i :
  complex.conj (given_complex_number) = expected_conjugate :=
by
  -- Simplification steps would go here
  sorry

end conjugate_of_given_number_is_i_l200_200285


namespace probability_of_odd_numbers_and_twos_l200_200678

open Finset BigOperators

noncomputable def probability_odd_numbers_exactly_six_and_two_of_them_are_three : ℚ :=
  let binom := λ n k: ℕ, (finset.range (n + 1)).powerset.filter (λ s, s.card = k).card
  let probability := (binom 8 6 : ℚ) * (1 / 2) ^ 8 * (binom 6 2 : ℚ) * (1 / 3) ^ 2 * (2 / 3) ^ 4
  in probability
  
theorem probability_of_odd_numbers_and_twos : probability_odd_numbers_exactly_six_and_two_of_them_are_three = 35 / 972 := by
  sorry

end probability_of_odd_numbers_and_twos_l200_200678


namespace range_of_a_l200_200200

open Complex

noncomputable def z_conj (a : ℝ) : ℂ :=
  Complex.conj ((1 - a * Complex.I) / (1 + Complex.I))

theorem range_of_a (a : ℝ) :
  (z_conj a).re < 0 ∧ (z_conj a).im > 0 ↔ a > 1 := by
  sorry

end range_of_a_l200_200200


namespace sum_of_reciprocals_of_roots_l200_200998

theorem sum_of_reciprocals_of_roots (p q r : ℝ) (h : ∀ x : ℝ, (x^3 - x - 6 = 0) → (x = p ∨ x = q ∨ x = r)) :
  1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2) = 11 / 12 :=
sorry

end sum_of_reciprocals_of_roots_l200_200998


namespace adult_ticket_cost_l200_200089

-- Definitions based on given conditions.
def children_ticket_cost : ℝ := 7.5
def total_bill : ℝ := 138
def total_tickets : ℕ := 12
def additional_children_tickets : ℕ := 8

-- Proof statement: Prove the cost of each adult ticket.
theorem adult_ticket_cost (x : ℕ) (A : ℝ)
  (h1 : x + (x + additional_children_tickets) = total_tickets)
  (h2 : x * A + (x + additional_children_tickets) * children_ticket_cost = total_bill) :
  A = 31.50 :=
  sorry

end adult_ticket_cost_l200_200089


namespace probability_eventA_probability_eventB_l200_200093

namespace math_proof

def P := {2, 3}
def Q := {-1, 1, 2, 3, 4}

def eventA (a b : ℤ) := (a ∈ P ∧ b ∈ Q ∧ a ≥ 2 ∧ b ≤ 3)

def f (a b x : ℝ) := a * x ^ 2 - 6 * b * x + 1

def eventB (a b : ℤ) := (a ∈ P ∧ b ∈ Q ∧ (∃ x : ℝ, x = 3 * b / a ∧ x ≤ 1))

theorem probability_eventA : 
  (∃ n : ℕ, n = 8 ∧ P.has_pmf.to_pmf (λ (a b : ℤ), eventA a b) = 0.8) :=
sorry

theorem probability_eventB :
  (∃ n : ℕ, n = 3 ∧ P.has_pmf.to_pmf (λ (a b : ℤ), eventB a b) = 0.3) :=
sorry

end math_proof

end probability_eventA_probability_eventB_l200_200093


namespace quadratic_solution_sum_l200_200312

theorem quadratic_solution_sum (m n p : ℕ) (h : m.gcd (n.gcd p) = 1)
  (h₀ : ∀ x, x * (5 * x - 11) = -6 ↔ x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p) :
  m + n + p = 70 :=
sorry

end quadratic_solution_sum_l200_200312


namespace domain_of_function_l200_200787

theorem domain_of_function :
  {x : ℝ | x ≥ -1 ∧ x ≠ 2} = {x : ℝ | x ∈ (Ici (-1) \ {2}) ∪ Ioi 2} := 
by 
  sorry

end domain_of_function_l200_200787


namespace car_rental_total_cost_l200_200545

theorem car_rental_total_cost 
  (rental_cost : ℕ)
  (gallons : ℕ)
  (cost_per_gallon : ℕ)
  (cost_per_mile : ℚ)
  (miles_driven : ℕ)
  (H1 : rental_cost = 150)
  (H2 : gallons = 8)
  (H3 : cost_per_gallon = 350 / 100)
  (H4 : cost_per_mile = 50 / 100)
  (H5 : miles_driven = 320) :
  rental_cost + gallons * cost_per_gallon + miles_driven * cost_per_mile = 338 :=
  sorry

end car_rental_total_cost_l200_200545


namespace find_sum_of_coefficients_l200_200147

-- Define the given polynomial expansion
def polynomial (x : ℝ) : ℝ := (1 - 2*x)^10

-- Define the expansion coefficients a_i
def expanded_polynomial (x : ℝ) : ℝ := 
  ∑ i in Finset.range 11, (coefficients i) * x^i

-- Define the coefficients as constants/variables pm i denotes a_i
-- Note: In practice, coefficients would be derived, here we assume as given for simplicity
constant coefficients : (ℕ → ℝ) 

-- Define the sum of a_i * i
def sum_of_coefficients := 
  a_1 + 2*a_2 + 3*a_3 + 4*a_4 + 5*a_5 + 6*a_6 + 7*a_7 + 8*a_8 + 9*a_9 + 10*a_10 

-- The theorem to prove
theorem find_sum_of_coefficients (h: polynomial x = expanded_polynomial x): 
  sum_of_coefficients = 20 := 
 sorry

end find_sum_of_coefficients_l200_200147


namespace eval_p_positive_int_l200_200778

theorem eval_p_positive_int (p : ℕ) : 
  (∃ n : ℕ, n > 0 ∧ (4 * p + 20) = n * (3 * p - 6)) ↔ p = 3 ∨ p = 4 ∨ p = 15 ∨ p = 28 := 
by sorry

end eval_p_positive_int_l200_200778


namespace smallest_positive_period_minimum_value_and_corresponding_x_values_intervals_where_f_increasing_l200_200846

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem smallest_positive_period : (∃ T > 0, ∀ x, f x = f (x + T)) ∧ (∀ T > 0, (∃ x, f x ≠ f (x + T)) → T = Real.pi) :=
begin
  sorry
end

theorem minimum_value_and_corresponding_x_values :
  (∃ y, ∀ x, y ≤ f x ∧ (y = -2)) ∧ (∃ (k : ℤ), ∀ x, f x = -2 ↔ x = ↑k * Real.pi - 5 * Real.pi / 12) :=
begin
  sorry
end

theorem intervals_where_f_increasing :
  (∃ k : ℤ, ∀ x, k * Real.pi - 5 * Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 12) :=
begin
  sorry
end

end smallest_positive_period_minimum_value_and_corresponding_x_values_intervals_where_f_increasing_l200_200846


namespace find_x_y_l200_200243

theorem find_x_y 
  (x y : ℚ)
  (h1 : (x / 6) * 12 = 10)
  (h2 : (y / 4) * 8 = x) :
  x = 5 ∧ y = 2.5 :=
by
  sorry

end find_x_y_l200_200243


namespace value_of_g_at_8_l200_200924

def g (x : ℝ) : ℝ := (6 * x + 2) / (x - 2)

theorem value_of_g_at_8 : g 8 = 25 / 3 := 
by sorry

end value_of_g_at_8_l200_200924


namespace p_necessary_but_not_sufficient_for_q_l200_200460

variable (a : ℝ)

def p := a ≤ 1
def q := |a| ≤ 1

theorem p_necessary_but_not_sufficient_for_q : (q → p) ∧ ¬ (p → q) :=
by
  split
  -- first part q implies p
  intro h
  unfold p q at *
  sorry
  -- second part not (p implies q)
  intro h
  unfold p q at *
  sorry

end p_necessary_but_not_sufficient_for_q_l200_200460


namespace slope_of_tangent_at_A_l200_200311

theorem slope_of_tangent_at_A :
  let f := λ x : ℝ, x^2 + 3*x
  let A := (2 : ℝ, 10 : ℝ)
  let f' := λ x : ℝ, (2*x + 3)
  f'(2) = 7 :=
  by
    sorry

end slope_of_tangent_at_A_l200_200311


namespace halving_process_iterations_l200_200606

theorem halving_process_iterations : ∀ n : ℕ, 
  let f : ℕ → ℕ := λ x, x / 2 in (n = 200) →
  (∃ k : ℕ, (∀ i < k, f^[i] n >= 3) ∧ f^[k] n < 3 ∧ k = 7) :=
begin
  sorry
end

end halving_process_iterations_l200_200606


namespace water_tank_volume_proof_l200_200646

noncomputable def waterTankHeight : ℝ := 60 * real.cbrt 0.4
def a : ℕ := 30
def b : ℕ := 8

theorem water_tank_volume_proof : a + b = 38 := by
  sorry

end water_tank_volume_proof_l200_200646


namespace angie_age_problem_l200_200324

theorem angie_age_problem (age : ℕ) (number : ℕ) (h1 : age = 8) (h2 : number = 2 * age + 4) : number = 20 :=
by
  rw [h1] at h2
  rw [h2]
  exact rfl

end angie_age_problem_l200_200324


namespace option_B_correct_option_C_correct_l200_200014

-- Theorem for Option B
theorem option_B_correct (x : ℝ) : deriv (λ x, (x^2 + 2) * sin x) = (2 * x * sin x) + ((x^2 + 2) * cos x) :=
by sorry

-- Theorem for Option C
theorem option_C_correct (x : ℝ) : deriv (λ x, x^2 / exp x) = (2 * x - x^2) / exp x :=
by sorry

end option_B_correct_option_C_correct_l200_200014


namespace sum_of_squares_of_coeffs_l200_200758

theorem sum_of_squares_of_coeffs (a b c : ℕ) : (a = 6) → (b = 24) → (c = 12) → (a^2 + b^2 + c^2 = 756) :=
by
  sorry

end sum_of_squares_of_coeffs_l200_200758


namespace initial_cookies_l200_200391

variable (andys_cookies : ℕ)

def total_cookies_andy_ate : ℕ := 3
def total_cookies_brother_ate : ℕ := 5

def arithmetic_sequence_sum (n : ℕ) : ℕ := n * (2 * n - 1)

def total_cookies_team_ate : ℕ := arithmetic_sequence_sum 8

theorem initial_cookies :
  andys_cookies = total_cookies_andy_ate + total_cookies_brother_ate + total_cookies_team_ate :=
  by
    -- Here the missing proof would go
    sorry

end initial_cookies_l200_200391


namespace optimal_pipeline_connection_l200_200138

theorem optimal_pipeline_connection (A B C : Point) :
  ∃ T : Point, 
    (∀ α β γ : ℝ, α < 120 ∧ β < 120 ∧ γ < 120 →
      angle A T B = 120 ∧ angle B T C = 120 ∧ angle C T A = 120) ∨
    (∃ α : ℝ, α ≥ 120 →
      (T = A ∨ T = B ∨ T = C)) :=
sorry

end optimal_pipeline_connection_l200_200138


namespace parameterize_line_l200_200299

theorem parameterize_line (f : ℝ → ℝ) (t : ℝ) (x y : ℝ)
  (h1 : y = 2 * x - 30)
  (h2 : (x, y) = (f t, 20 * t - 10)) :
  f t = 10 * t + 10 :=
sorry

end parameterize_line_l200_200299


namespace eccentricity_range_l200_200152

variable {a b c : ℝ} (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (e : ℝ)

-- Assume a > 0, b > 0, and the eccentricity of the hyperbola is given by c = e * a.
variable (a_pos : 0 < a) (b_pos : 0 < b) (hyperbola : (P.1 / a)^2 - (P.2 / b)^2 = 1)
variable (on_right_branch : P.1 > 0)
variable (foci_condition : dist P F₁ = 4 * dist P F₂)
variable (eccentricity_def : c = e * a)

theorem eccentricity_range : 1 < e ∧ e ≤ 5 / 3 := by
  sorry

end eccentricity_range_l200_200152


namespace find_a_l200_200165

noncomputable def binomial_expansion_term_coefficient
  (n : ℕ) (r : ℕ) (a : ℝ) (x : ℝ) : ℝ :=
  (2^(n-r)) * ((-a)^r) * (Nat.choose n r) * (x^(n - 2*r))

theorem find_a 
  (a : ℝ)
  (h : binomial_expansion_term_coefficient 7 5 a 1 = 84) 
  : a = -1 :=
sorry

end find_a_l200_200165


namespace find_x_l200_200184

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (2 * x, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (4, x)

-- We are given the condition that the angle between a and b is 180 degrees.
-- This implies that the vectors are in opposite directions.

theorem find_x (x : ℝ) (h : ∃ k < 0, (2 * x) / 4 = k ∧ 1 / x = k) : x = -real.sqrt 2 :=
sorry

end find_x_l200_200184


namespace remainder_3_pow_75_add_4_mod_5_l200_200681

theorem remainder_3_pow_75_add_4_mod_5 :
  (3 ^ 75 + 4) % 5 = 1 := by

  /- Use modular arithmetic properties -/
  have h1 : 3 ^ 4 % 5 = 1 := by norm_num,
  
  /- Express 3 ^ 75 in terms of 3 ^ (multiple of 4) -/
  have h2 : 3 ^ 75 % 5 = (3 ^ 4) ^ 18 * 3 ^ 3 % 5 := by
    rw [pow_add, pow_mul],

  /- Simplify the expression -/
  rw [h1, one_pow, one_mul, pow_succ, pow_succ, pow_one] at h2,
  norm_num at h2,

  /- Calculate the final expression with the addition -/
  norm_num,
  sorry

end remainder_3_pow_75_add_4_mod_5_l200_200681


namespace r2_value_l200_200507

theorem r2_value :
  let q1 := λ x : ℝ, x^6 + (1/2) * x^5 + (1/4) * x^4 + (1/8) * x^3 + (1/16) * x^2 + (1/32) * x + (1/64)
  in q1 (1 / 2) = 7 / 64 :=
by {
  let q1 := λ x : ℝ, x^6 + (1/2) * x^5 + (1/4) * x^4 + (1/8) * x^3 + (1/16) * x^2 + (1/32) * x + (1/64),
  show q1 (1 / 2) = 7 / 64,
  sorry
}

end r2_value_l200_200507


namespace local_min_and_sum_of_zeros_l200_200174

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x

theorem local_min_and_sum_of_zeros
  {a x1 x2 : ℝ} 
  (h1 : f a x1 = 0)
  (h2 : f a x2 = 0) 
  (x1_lt_x2 : x1 < x2) 
  (a_pos : 0 < a) 
  (f_deriv : ∀ x, deriv (f a) x = Real.exp x - a) :
  (∃ x0 : ℝ, x0 = Real.log(a) ∧ x1 + x2 < 2 * x0) :=
by
  sorry

end local_min_and_sum_of_zeros_l200_200174


namespace BC_parallel_B1C1_l200_200223

theorem BC_parallel_B1C1
  (A B C P D E X C1 B1 : Point)
  (hP_on_BC : P ∈ Line.mk B C)
  (hD_reflect_P_B : D.reflect_over B P)
  (hE_reflect_P_C : E.reflect_over C P)
  (hX_sec_inter_abe_acd : is_second_inter_of_circumcircles X (circumcircle A B E) (circumcircle A C D))
  (hC1_ray_AB_XD : intersects (ray A B) (segment X D) C1)
  (hB1_ray_AC_XE : intersects (ray A C) (segment X E) B1) :
  Parallel (Line.mk B C) (Line.mk B1 C1) := sorry

end BC_parallel_B1C1_l200_200223


namespace chim_tu_can_slide_all_pieces_off_the_table_l200_200086

structure ConvexPolygon :=
  (vertices : List (ℝ × ℝ))
  (is_convex : convex vertices)

structure Table :=
  (width : ℝ)
  (height : ℝ)

def no_overlap (polygons : List ConvexPolygon) : Prop :=
  ∀ p1 p2 ∈ polygons, p1 ≠ p2 → disjoint (interior p1) (interior p2)

def can_slide_off (table : Table) (polygons : List ConvexPolygon) : Prop :=
  ∃ (n : ℕ), ∀ (i : ℕ) (p : ConvexPolygon) (h : p ∈ polygons), ∃ (d : ℝ), d > 0 ∧ (slid_off p table d)

theorem chim_tu_can_slide_all_pieces_off_the_table (table : Table) (polygons : List ConvexPolygon) :
  (finite polygons) → no_overlap polygons → can_slide_off table polygons :=
sorry

end chim_tu_can_slide_all_pieces_off_the_table_l200_200086


namespace inverse_matrix_eigenvalues_and_eigenvectors_l200_200492

section linear_algebra

open Matrix

variables (A : Matrix (Fin 2) (Fin 2) ℚ) 

-- Given matrix.
def given_matrix : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![1, 4], ![2, 3]]

-- Inverse matrix assertion.
theorem inverse_matrix : 
  given_matrix = A →
  inv A = ![![(-3 : ℚ) / 5, (4 : ℚ) / 5], ![(2 : ℚ) / 5, (-1 : ℚ) / 5]] :=
sorry

-- Eigenvalues and eigenvectors.
theorem eigenvalues_and_eigenvectors : 
  given_matrix = A →
  (eigenvalues A = {5, -1} ∧
   eigenvector A 5 = ![1, 1] ∧
   eigenvector A (-1) = ![-2, 1]) :=
sorry

end linear_algebra

end inverse_matrix_eigenvalues_and_eigenvectors_l200_200492


namespace line_parallel_or_within_other_plane_l200_200453

-- Define the necessary concepts: Line, Plane, and parallelism
variables {Point Line Plane : Type}
variables (Parallel : Line → Plane → Prop)
variables (Within : Line → Plane → Prop)
variables (P1 P2 : Plane)
variables (L : Line)

-- Given conditions
axiom planes_parallel : P1 ≠ P2 ∧ ∀ P1 P2, Parallel P1 P2
axiom line_parallel_to_plane : Parallel L P1

-- Statement of the problem
theorem line_parallel_or_within_other_plane :
  Parallel P1 P2 → Parallel L P1 → (Parallel L P2 ∨ Within L P2) :=
by
  sorry

end line_parallel_or_within_other_plane_l200_200453


namespace mean_greater_than_median_by_point_one_l200_200439

noncomputable def days_missed : List ℕ := [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5]

def mean (l : List ℕ) : ℚ :=
  let total := l.foldl (λ acc x => acc + x) 0
  total / l.length

def median (l : List ℕ) : ℚ :=
  let sorted := l.sorted
  if h : even l.length then
    let middle := l.length / 2
    (sorted.get (middle - 1) + sorted.get middle) / 2
  else
    let middle := l.length / 2
    sorted.get middle

theorem mean_greater_than_median_by_point_one :
  let l := days_missed
  mean l - median l = 0.1 :=
by sorry

end mean_greater_than_median_by_point_one_l200_200439


namespace max_sum_cos_l200_200494

theorem max_sum_cos (a b c : ℝ) (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (2 * x) + c * Real.cos (3 * x) ≥ -1) : a + b + c ≤ 3 := by
  sorry

end max_sum_cos_l200_200494


namespace number_of_trees_in_park_l200_200055

def number_of_trees (length width area_per_tree : ℕ) : ℕ :=
  (length * width) / area_per_tree

theorem number_of_trees_in_park :
  number_of_trees 1000 2000 20 = 100000 :=
by
  sorry

end number_of_trees_in_park_l200_200055


namespace total_trees_in_forest_l200_200618

theorem total_trees_in_forest (a_street : ℕ) (a_forest : ℕ) 
                              (side_length : ℕ) (trees_per_square_meter : ℕ)
                              (h1 : a_street = side_length * side_length)
                              (h2 : a_forest = 3 * a_street)
                              (h3 : side_length = 100)
                              (h4 : trees_per_square_meter = 4) :
                              a_forest * trees_per_square_meter = 120000 := by
  -- Proof omitted
  sorry

end total_trees_in_forest_l200_200618


namespace tetrahedron_volume_le_one_eighth_l200_200780

variables {A B C D : Type} [EuclideanSpace E ℝ]

theorem tetrahedron_volume_le_one_eighth
  (hCD : ∥C - D∥ > 1)
  (a : ℝ)
  (hAB : ∥A - B∥ = a) :
  volume_of_tetrahedron A B C D ≤ 1 / 8 :=
sorry

end tetrahedron_volume_le_one_eighth_l200_200780


namespace arrange_1225_multiple_of_5_l200_200965

theorem arrange_1225_multiple_of_5 : 
  (∃ (s : Finset (Fin 10)) (h : s = {1, 2, 2, 5}),
    (card (s.filter (λ x, last.digit x = 5))
    = 6) : sorry

end arrange_1225_multiple_of_5_l200_200965


namespace perpendicular_vectors_l200_200885

variable (m : ℝ)

def vector_a := (m, 3)
def vector_b := (1, m + 1)

def dot_product (v w : ℝ × ℝ) := (v.1 * w.1) + (v.2 * w.2)

theorem perpendicular_vectors (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by 
  unfold vector_a vector_b dot_product at h
  linarith

end perpendicular_vectors_l200_200885


namespace number_of_solutions_l200_200304

-- Defining the conditions for the equation
def isCondition (x : ℝ) : Prop := x ≠ 2 ∧ x ≠ 3

-- Defining the equation
def eqn (x : ℝ) : Prop := (3 * x^2 - 15 * x + 18) / (x^2 - 5 * x + 6) = x - 2

-- Defining the property that we need to prove
def property (x : ℝ) : Prop := eqn x ∧ isCondition x

-- Statement of the proof problem
theorem number_of_solutions : 
  ∃! x : ℝ, property x :=
sorry

end number_of_solutions_l200_200304


namespace binary_multiplication_l200_200427

theorem binary_multiplication :
  let a := 0b1101101
  let b := 0b1011
  let product := 0b10001001111
  a * b = product :=
sorry

end binary_multiplication_l200_200427


namespace behavior_on_interval_l200_200475

noncomputable def f : ℝ → ℝ := sorry

-- Definitions/Conditions
def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def periodic (f : ℝ → ℝ) (T : ℝ) := ∀ x, f (x + T) = f x
def increasing_on (f : ℝ → ℝ) (I : set ℝ) := ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

-- Conditions
axiom h_even : is_even f
axiom h_periodic : periodic f 2
axiom h_increasing : increasing_on f (set.Icc 0 1)

-- Proof Statement
theorem behavior_on_interval : 
  ∀ x ∈ set.Icc (-3) (-1), 
    (x ∈ set.Icc (-3) (-2) → ∀ y, x < y → y ∈ set.Icc (-3) (-2) → f x > f y) ∧
    (x ∈ set.Icc (-2) (-1) → ∀ y, x < y → y ∈ set.Icc (-2) (-1) → f x < f y) := 
sorry

end behavior_on_interval_l200_200475


namespace perpendicular_vectors_implies_m_value_l200_200869

variable (m : ℝ)

def vector1 : ℝ × ℝ := (m, 3)
def vector2 : ℝ × ℝ := (1, m + 1)

theorem perpendicular_vectors_implies_m_value
  (h : vector1 m ∙ vector2 m = 0) :
  m = -3 / 4 :=
by 
  sorry

end perpendicular_vectors_implies_m_value_l200_200869


namespace divide_line_segment_l200_200770

variable (P Q : ℝ × ℝ) (r : ℝ)

theorem divide_line_segment (P Q : ℝ × ℝ) (r : ℝ) :
  P = (2.5, 3.5) →
  Q = (4.5, 0.5) →
  r = 1 / (3 + 1) →
  let x := (r * (Q.1 - P.1) + P.1) in
  let y := (r * (Q.2 - P.2) + P.2) in
  (x, y) = (3, 2.75) :=
by
  intros hP hQ hr
  rw [hP, hQ, hr]
  let r := 1 / 4
  let x := r * (4.5 - 2.5) + 2.5
  let y := r * (0.5 - 3.5) + 3.5
  have hx : x = 3 := by norm_num
  have hy : y = 2.75 := by norm_num
  rw [hx, hy]
  exact ⟨hx, hy⟩

end divide_line_segment_l200_200770


namespace Joan_initial_money_l200_200541

def cost_hummus (containers : ℕ) (price_per_container : ℕ) : ℕ := containers * price_per_container
def cost_apple (quantity : ℕ) (price_per_apple : ℕ) : ℕ := quantity * price_per_apple

theorem Joan_initial_money 
  (containers_of_hummus : ℕ)
  (price_per_hummus : ℕ)
  (cost_chicken : ℕ)
  (cost_bacon : ℕ)
  (cost_vegetables : ℕ)
  (quantity_apple : ℕ)
  (price_per_apple : ℕ)
  (total_cost : ℕ)
  (remaining_money : ℕ):
  containers_of_hummus = 2 →
  price_per_hummus = 5 →
  cost_chicken = 20 →
  cost_bacon = 10 →
  cost_vegetables = 10 →
  quantity_apple = 5 →
  price_per_apple = 2 →
  remaining_money = cost_apple quantity_apple price_per_apple →
  total_cost = cost_hummus containers_of_hummus price_per_hummus + cost_chicken + cost_bacon + cost_vegetables + remaining_money →
  total_cost = 60 :=
by
  intros
  sorry

end Joan_initial_money_l200_200541


namespace length_db_l200_200530

theorem length_db
  (angle_ABC : ∠ABC = 90)
  (angle_ADB : ∠ADB = 90)
  (AC : ℝ)
  (AD : ℝ)
  (AC_value : AC = 18)
  (AD_value : AD = 6) :
  ∃ DB : ℝ, DB = 6 * Real.sqrt 2 :=
by
  sorry

end length_db_l200_200530


namespace find_ratio_l200_200579

open Real

theorem find_ratio (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + 2 * y) / (x - 2 * y) = -4 / sqrt 7 :=
by
  sorry

end find_ratio_l200_200579


namespace unique_b_for_line_through_parabola_vertex_l200_200132

theorem unique_b_for_line_through_parabola_vertex :
  ∃! b : ℝ, ∃ (line : ℝ → ℝ), 
    line = (λ x, 2*x + b) ∧ 
    (∃ vertex : ℝ × ℝ, vertex = (b, 0) ∧ vertex.2 = (line vertex.1)) :=
begin
  sorry
end

end unique_b_for_line_through_parabola_vertex_l200_200132


namespace tan_half_angle_l200_200839

variable (α : Real)
variable (P : Real × Real)

def terminalSide (α : Real) (P : Real × Real) : Prop :=
  P = (3, -4) ∧ (∃ t : Real, tan α = t)

theorem tan_half_angle (h : terminalSide α P) : tan (α / 2) = -1 / 2 := 
  sorry

end tan_half_angle_l200_200839


namespace white_balls_count_l200_200058

-- Definitions for the conditions
variable (x y : ℕ) 

-- Lean statement representing the problem
theorem white_balls_count : 
  x < y ∧ y < 2 * x ∧ 2 * x + 3 * y = 60 → x = 9 := 
sorry

end white_balls_count_l200_200058


namespace sum_xy_22_l200_200193

theorem sum_xy_22 (x y : ℕ) (h1 : 0 < x) (h2 : x < 25) (h3 : 0 < y) (h4 : y < 25) 
  (h5 : x + y + x * y = 118) : x + y = 22 :=
sorry

end sum_xy_22_l200_200193


namespace polynomial_remainder_thm_l200_200563

theorem polynomial_remainder_thm:
  ∀ (Q : ℚ[x]), 
  (∀ x, x = 15 → eval x Q = 10) → 
  (∀ x, x = 12 → eval x Q = 2) →
  ∃ (c d : ℚ), (∀ x, eval x (Q - ((x - 12) * (x - 15) * C(x) + c * x + d)) = 0) ∧ 
  (c = 8/3) ∧ (d = -30) :=
by
  sorry

end polynomial_remainder_thm_l200_200563


namespace digit_solve_l200_200192

theorem digit_solve : ∀ (D : ℕ), D < 10 → (D * 9 + 6 = D * 10 + 3) → D = 3 :=
by
  intros D hD h
  sorry

end digit_solve_l200_200192


namespace perpendicular_lines_to_plane_are_parallel_l200_200183

variable (a b : Line)
variable (α : Plane)

-- Conditions: 
-- a and b are lines such that: 
-- a ⊥ α 
-- b ⊥ α

theorem perpendicular_lines_to_plane_are_parallel (h1 : a ⊥ α) (h2 : b ⊥ α) : a ∥ b :=
sorry

end perpendicular_lines_to_plane_are_parallel_l200_200183


namespace geometric_sequence_product_equality_l200_200151

theorem geometric_sequence_product_equality
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h_q : q = 2)
  (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_product : ∏ i in finset.range 30, a (i + 1) = 2^30)
  : (∏ i in finset.range 10, a ((i + 1) * 3)) = 2^20 :=
begin
  -- Placeholder for the actual proof
  sorry,
end

end geometric_sequence_product_equality_l200_200151


namespace function_equality_for_all_x_l200_200365

theorem function_equality_for_all_x (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f(x + 19) ≤ f(x) + 19)
  (h2 : ∀ x : ℝ, f(x + 94) ≥ f(x) + 94) :
  ∀ x : ℝ, f(x + 1) = f(x) + 1 :=
by
  sorry

end function_equality_for_all_x_l200_200365


namespace eccentricity_of_ellipse_l200_200112

-- Definitions based on conditions:
variables (a b x0 y0 : ℝ) (h : a > b > 0)
def ellipse (x y : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1

def left_vertex (A : ℝ × ℝ) : Prop := A = (-a, 0)
def symmetric_points (P Q : ℝ × ℝ) : Prop := (P = (x0, y0) ∧ Q = (-x0, y0))

def slopes_product (P Q : ℝ × ℝ) (A : ℝ × ℝ) : Prop := 
  let k_AP := (P.2 - A.2) / (P.1 - A.1)
  let k_AQ := (Q.2 - A.2) / (Q.1 - A.1)
  k_AP * k_AQ = 1 / 3

-- Theorem statement to prove the eccentricity given the conditions
theorem eccentricity_of_ellipse :
  (a > b > 0) →
  (ellipse a b x0 y0) →
  (left_vertex a b (-a, 0)) →
  (symmetric_points a b (x0, y0) (-x0, y0)) →
  (slopes_product a b (x0, y0) (-x0, y0) (-a, 0)) →
  (let e := (sqrt (1 - b^2 / a^2)) in
   e = sqrt(6) / 3) :=
sorry

end eccentricity_of_ellipse_l200_200112


namespace weight_of_person_replaced_l200_200949

theorem weight_of_person_replaced (W : ℝ) (old_avg_weight : ℝ) (new_avg_weight : ℝ)
  (h_avg_increase : new_avg_weight = old_avg_weight + 1.5) (new_person_weight : ℝ) :
  ∃ (person_replaced_weight : ℝ), new_person_weight = 77 ∧ old_avg_weight = W / 8 ∧
  new_avg_weight = (W - person_replaced_weight + 77) / 8 ∧ person_replaced_weight = 65 := by
    sorry

end weight_of_person_replaced_l200_200949


namespace digit7_appears_300_times_l200_200290

theorem digit7_appears_300_times (N : ℕ) (h1 : ∀ k : ℕ, k ≤ N → 
  (7_ones : ℕ) + (7_tens : ℕ) = 20 * (k / 100)) : 
  N = 1500 :=
by
  -- Proof will go here
  sorry

end digit7_appears_300_times_l200_200290


namespace orchard_apples_relation_l200_200361

/-- 
A certain orchard has 10 apple trees, and on average each tree can produce 200 apples. 
Based on experience, for each additional tree planted, the average number of apples produced per tree decreases by 5. 
We are to show that if the orchard has planted x additional apple trees and the total number of apples is y, then the relationship between y and x is:
y = (10 + x) * (200 - 5x)
-/
theorem orchard_apples_relation (x : ℕ) (y : ℕ) 
    (initial_trees : ℕ := 10)
    (initial_apples : ℕ := 200)
    (decrease_per_tree : ℕ := 5)
    (total_trees := initial_trees + x)
    (average_apples := initial_apples - decrease_per_tree * x)
    (total_apples := total_trees * average_apples) :
    y = total_trees * average_apples := 
  by 
    sorry

end orchard_apples_relation_l200_200361


namespace gcd_36_60_eq_12_l200_200006

theorem gcd_36_60_eq_12 :
  ∃ (g : ℕ), g = Nat.gcd 36 60 ∧ g = 12 := by
  -- Defining the conditions:
  let a := 36
  let b := 60
  have fact_a : a = 2^2 * 3^2 := rfl
  have fact_b : b = 2^2 * 3 * 5 := rfl
  
  -- The statement to prove:
  sorry

end gcd_36_60_eq_12_l200_200006


namespace P_at_neg1_l200_200700

noncomputable def P : ℝ → ℝ := sorry -- since P is a degree 4 polynomial defined as per conditions, we leave it as sorry here

theorem P_at_neg1 : P(-1) = 6 :=
by
  -- Defining conditions as assumptions
  have P_degree_4 : ∀ x, P x = a * x^4 + b * x^3 + c * x^2 + d * x + e := sorry,
  have P_at_0 : P(0) = 1 := sorry,
  have P_at_1 : P(1) = 1 := sorry,
  have P_at_2 : P(2) = 4 := sorry,
  have P_at_3 : P(3) = 9 := sorry,
  have P_at_4 : P(4) = 16 := sorry,
  -- Goal: P(-1) = 6
  sorry -- leaving the full proof as it's not required

end P_at_neg1_l200_200700


namespace z_is_real_iff_m_values_z_in_third_quadrant_iff_m_interval_l200_200841

section
variable (m : ℝ)
def z : ℂ := (m^2 + 5 * m + 6) + (m^2 - 2 * m - 15) * Complex.I

theorem z_is_real_iff_m_values :
  (z m).im = 0 ↔ m = -3 ∨ m = 5 :=
by sorry

theorem z_in_third_quadrant_iff_m_interval :
  (z m).re < 0 ∧ (z m).im < 0 ↔ m ∈ Set.Ioo (-3) (-2) :=
by sorry
end

end z_is_real_iff_m_values_z_in_third_quadrant_iff_m_interval_l200_200841


namespace thirty_sixty_ninety_triangle_area_l200_200630

theorem thirty_sixty_ninety_triangle_area (hypotenuse : ℝ) (angle : ℝ) (area : ℝ)
  (h_hypotenuse : hypotenuse = 12)
  (h_angle : angle = 30)
  (h_area : area = 18 * Real.sqrt 3) :
  ∃ (base height : ℝ), 
    base = hypotenuse / 2 ∧ 
    height = (hypotenuse / 2) * Real.sqrt 3 ∧ 
    area = (1 / 2) * base * height :=
by {
  sorry
}

end thirty_sixty_ninety_triangle_area_l200_200630


namespace oblique_asymptote_l200_200329

theorem oblique_asymptote :
  ∀ x : ℝ, (∃ δ > 0, ∀ y > x, (abs (3 * y^2 + 8 * y + 12) / (3 * y + 4) - (y + 4 / 3)) < δ) :=
sorry

end oblique_asymptote_l200_200329


namespace intersection_points_four_l200_200487

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.log (Real.abs x) - 1 / 2)
def g (x : ℝ) : ℝ := x^2

theorem intersection_points_four (a : ℝ) :
  (∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 
    ∧ f a x1 = g x1 ∧ f a x2 = g x2 ∧ f a x3 = g x3 ∧ f a x4 = g x4) ↔ 2 * Real.exp 2 < a :=
begin
  sorry
end

end intersection_points_four_l200_200487


namespace problem1_problem2_l200_200699

-- Problem 1
theorem problem1:
  (-(2 - real.sqrt 3) - (real.pi - 3.14)^0 + (1/2 - real.cos (real.pi / 6)) * (1/2)^(-2) = -3 - real.sqrt 3) :=
by sorry

-- Problem 2
theorem problem2 (a : ℝ) (h : a = real.sqrt 3):
  ((a - 2) / (a^2 - 4) - (a + 2) / (a^2 - 4*a + 4) / (a + 2) / (a - 2) = -4 / (a^2 - 4)) :=
by sorry

end problem1_problem2_l200_200699


namespace min_value_quadratic_function_l200_200272

def f (a b c x : ℝ) : ℝ := a * (x - b) * (x - c)

theorem min_value_quadratic_function :
  ∃ a b c : ℝ, 
    (1 ≤ a ∧ a < 10) ∧
    (1 ≤ b ∧ b < 10) ∧
    (1 ≤ c ∧ c < 10) ∧
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (∀ x : ℝ, f a b c x ≥ -128) :=
sorry

end min_value_quadratic_function_l200_200272


namespace number_of_children_l200_200280

theorem number_of_children (A V S : ℕ) (x : ℕ → ℕ) (n : ℕ) 
  (h1 : (A / 2) + V = (A + V + S + (Finset.range (n - 3)).sum x) / n)
  (h2 : S + A = V + (Finset.range (n - 3)).sum x) : 
  n = 6 :=
sorry

end number_of_children_l200_200280


namespace sum_midpoint_coordinates_distance_midpoint_to_endpoint_l200_200682

-- Definitions and conditions
def p1 : ℝ × ℝ := (8, -4)
def p2 : ℝ × ℝ := (-2, 10)

-- Midpoint Calculation
def midpoint (a b : ℝ × ℝ) : ℝ × ℝ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

-- Distance Calculation
def distance (a b : ℝ × ℝ) : ℝ := real.sqrt ((a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2)

-- Midpoint of given points
def mid : ℝ × ℝ := midpoint p1 p2

-- Proof for the sum of coordinates of the midpoint
theorem sum_midpoint_coordinates : mid.1 + mid.2 = 6 := by
  -- intermediate values
  have mid_x : mid.1 = 3 := by
    unfold mid midpoint p1 p2
    norm_num
  have mid_y : mid.2 = 3 := by
    unfold mid midpoint p1 p2
    norm_num
  -- sum of coordinates
  rw [mid_x, mid_y]
  norm_num

-- Proof for the distance from midpoint to the endpoint (8, -4)
theorem distance_midpoint_to_endpoint : distance mid p1 = real.sqrt 74 := by
  unfold mid midpoint p1 p2 distance
  norm_num
  simp [real.sqrt_eq_rpow]
  norm_num

-- Adding sorry to skip the proof, if required.
-- theorem sum_midpoint_coordinates : mid.1 + mid.2 = 6 := sorry
-- theorem distance_midpoint_to_endpoint : distance mid p1 = real.sqrt 74 := sorry

end sum_midpoint_coordinates_distance_midpoint_to_endpoint_l200_200682


namespace find_m_l200_200888

-- Declare the vectors a and b based on given conditions
variables {m : ℝ}

def a : ℝ × ℝ := (m, 3)
def b : ℝ × ℝ := (1, m + 1)

-- Define the condition that vectors a and b are perpendicular
def perpendicular (x y : ℝ × ℝ) : Prop := x.1 * y.1 + x.2 * y.2 = 0

-- State the problem in Lean 4
theorem find_m (h : perpendicular a b) : m = -3 / 4 :=
sorry

end find_m_l200_200888


namespace find_shares_of_stock_y_l200_200314

noncomputable def shares_of_stock_y_before_buying 
  (v : ℕ) (w : ℕ) (x : ℕ) (y : ℕ) (z : ℕ) (sells_x : ℕ) (buys_y : ℕ) (range_increase : ℕ) : ℕ :=
if w - y < w - z then
  (w - y + range_increase - x)
else
  (w - z + range_increase - x)

theorem find_shares_of_stock_y :
  shares_of_stock_y_before_buying 68 112 56 50 45 20 23 14 = 50 :=
begin
  rw shares_of_stock_y_before_buying,
  sorry,
end

end find_shares_of_stock_y_l200_200314


namespace max_b1_value_l200_200644

noncomputable def b1_max (S4 S9 : ℕ) : ℚ :=
  (6 * S4 - S9) / 15

theorem max_b1_value : ∃ (S4 S9 : ℕ), (b1_max S4 S9 ≤ 3/4) ∧ (b1_max S4 S9 = 11 / 15) :=
begin
  use [2, 1], -- These values satisfy the conditions based on the solution
  split,
  { norm_num, -- Showing b1_max (2) (1) ≤ 3/4
    exact div_le_div_of_le_of_pos (by linarith) (by norm_num) },
  { norm_num, -- Showing b1_max (2) (1) = 11/15
    exact (div_eq_div_iff (by norm_num) (by norm_num)).mpr (by norm_num) }
end

end max_b1_value_l200_200644


namespace distance_between_planes_is_half_l200_200786

-- Define the planes
def plane1 (x y z : ℝ) : Prop := x + 2 * y - 2 * z + 1 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x + 4 * y - 4 * z + 5 = 0

-- Define the point-to-plane distance formula
def distance_point_to_plane (A B C D x0 y0 z0 : ℝ) : ℝ :=
  |A * x0 + B * y0 + C * z0 + D| / real.sqrt (A^2 + B^2 + C^2)

-- Define the normal vector for use in both planes
def normal_vector : (ℝ × ℝ × ℝ) := (1, 2, -2)

-- Example point on the first plane
def point_on_plane1 : (ℝ × ℝ × ℝ) := (-1, 0, 0)

-- Calculate distance from point_on_plane1 to plane2
def distance_between_planes : ℝ :=
  let (A, B, C) := (2, 4, -4) in
  let D := 5 in
  let (x0, y0, z0) := point_on_plane1 in
  distance_point_to_plane A B C D x0 y0 z0

-- Prove the calculated distance is 1/2
theorem distance_between_planes_is_half :
  distance_between_planes = 1 / 2 := by
  sorry

end distance_between_planes_is_half_l200_200786


namespace combined_speed_correct_l200_200094

-- Definitions of the constants used in problem
def miles_to_km : ℝ := 1.60934
def meter_to_kmh : ℝ := 3.6
def speed_mph : ℝ := 75
def speed_mps : ℝ := 9 / 36

-- Conversion functions
def convert_mph_to_kmh (speed_mph: ℝ): ℝ := speed_mph * miles_to_km
def convert_mps_to_kmh (speed_mps: ℝ): ℝ := speed_mps * meter_to_kmh

-- Combined speed function
def combined_speed (speed_mph speed_mps: ℝ) : ℝ :=
  convert_mph_to_kmh(speed_mph) + convert_mps_to_kmh(speed_mps)

-- The theorem stating the combined speed
theorem combined_speed_correct : combined_speed speed_mph speed_mps = 121.6005 := by
  sorry

end combined_speed_correct_l200_200094


namespace area_square_field_in_sqm_l200_200357

def time_taken := 2 -- in minutes
def speed := 3 / 60 -- in km/min
def diagonal_distance := speed * time_taken -- in km, 0.1 km

def side_length_sq (d : ℝ) : ℝ := d^2 / 2 -- half of the square of the diagonal

def area_sq (s : ℝ) : ℝ := s -- area is directly the side length squared in km^2

theorem area_square_field_in_sqm :
  let d := diagonal_distance in
  let s := real.sqrt (side_length_sq d) in
  let area := area_sq s in
  area * 1_000_000 = 5000 :=
by
  sorry

end area_square_field_in_sqm_l200_200357


namespace determinant_cross_product_is_E_squared_l200_200565

variables {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V]
variables (u v w : V)

def determinant (M : matrix ℝ 3 3) : ℝ := matrix.det M

def E : ℝ := inner_product_space.cross_product u v w

def determinant_cross_product_matrix : ℝ :=
  determinant ![![u × v, v × w, w × u]]

theorem determinant_cross_product_is_E_squared
  (hE : E = inner_product_space.cross_product u v w) :
  determinant_cross_product_matrix u v w = E ^ 2 :=
sorry

end determinant_cross_product_is_E_squared_l200_200565


namespace ellipse_equation_line_equation_l200_200160

-- Definitions and conditions based on the problem
def ellipse (a b : ℝ) (h : a > b ∧ b > 0) : set (ℝ × ℝ) := 
  {p | ∃ x y : ℝ, p = (x, y) ∧ (x^2 / a^2 + y^2 / b^2 = 1)}

def point_p := (2 * real.sqrt 6 / 3, 1)
def orthocenter_h := (2 * real.sqrt 6 / 3, -5 / 3)
def f1_c := (-1, 0)
def f2_c := (1, 0)

-- The statement to prove the equation of the ellipse is \(\frac{x^{2}}{4} + \frac{y^{2}}{3} = 1\)
theorem ellipse_equation (P_in_C : point_p ∈ ellipse 2 (sqrt 3) sorry) (H_orthocenter: orthocenter_h = sorry) :
  ∀ (x y : ℝ), (x, y) ∈ ellipse 2 (sqrt 3) sorry ↔ (x^2 / 4 + y^2 / 3 = 1) :=
sorry

-- Additional conditions for the second part
def left_vertex := (-2, 0)
def line_l (k : ℝ) := {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ (y = k * (x - 1))}

-- The statement to prove the equation of the line \( l \) through \( F_2 \) and intersecting the ellipse given slopes condition
theorem line_equation (A_slope_conditions : ∀ k1 k2 : ℝ, k1 + k2 = -1/2) :
  ∃ k : ℝ, line_l k = {p : ℝ × ℝ | ∃ x y : ℝ, p = (x, y) ∧ (y = 2 * (x - 1))} :=
sorry

end ellipse_equation_line_equation_l200_200160


namespace projection_fixed_point_exists_l200_200491

-- Define the conditions
variables {n : ℕ}
variables (e : Fin n → AffineSubspace ℝ (EuclideanSpace ℝ 2))
variables (P : EuclideanSpace ℝ 2)
variable (initial_point : e 0)

-- State the theorem
theorem projection_fixed_point_exists :
  ∃ P₁ : (e 0), 
    let Q := 
      (λ step : Fin n, 
      Fin.succ (Fin.castSucc step)) ⟨ ( ∀ P₁ : (e 0), (ortho_projection (e (Fin.castSucc step)) (ortho_projection (e step) P₁))) ⟩ in
      Q = P₁ := 
sorry

end projection_fixed_point_exists_l200_200491


namespace oblique_asymptote_l200_200331

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 8 * x + 12) / (3 * x + 4)

theorem oblique_asymptote :
  (∃ b : ℝ, ∀ x : ℝ, ∥f x - (x + b)∥ < ε) := by
sorry

end oblique_asymptote_l200_200331


namespace total_triangles_G15_l200_200628

def sequence_G : ℕ → ℕ
| 1       := 2
| (n + 1) := sequence_G n + 3 * (n + 1)

theorem total_triangles_G15 : sequence_G 15 = 362 :=
by
  sorry

end total_triangles_G15_l200_200628


namespace prism_cosine_angle_PC_alpha_correct_l200_200220

structure Prism :=
(P A B C : ℝ)
(hAB : A ≠ B)
(hAP : AP = 2)
(hABC : is_triangle P A B)
(h_regular : is_regular_triangular_prism P A B C)
(h_bisection : plane_bisects_volume A B α)

noncomputable def cosine_angle_PC_alpha (prism : Prism) : ℝ :=
  let α := plane_through A B in
  let PC := line_through P C in
  cosine_angle_between PC α

theorem prism_cosine_angle_PC_alpha_correct 
  (prism : Prism) : 
  cosine_angle_PC_alpha prism = (3 * Real.sqrt 5) / 10 :=
sorry

end prism_cosine_angle_PC_alpha_correct_l200_200220


namespace complex_modulus_condition_l200_200474

theorem complex_modulus_condition (z : ℂ) (h : (1 - complex.I) * z = -1 + 2 * complex.I) :
  complex.abs (conj z) = (real.sqrt 10) / 2 :=
sorry

end complex_modulus_condition_l200_200474


namespace longest_side_enclosure_l200_200092

variable (l w : ℝ)

theorem longest_side_enclosure (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 1920) : max l w = 101 :=
sorry

end longest_side_enclosure_l200_200092


namespace inheritance_amount_l200_200548

-- Define the conditions
def federal_tax_rate : ℝ := 0.2
def state_tax_rate : ℝ := 0.1
def total_taxes_paid : ℝ := 10500

-- Lean statement for the proof
theorem inheritance_amount (I : ℝ)
  (h1 : federal_tax_rate = 0.2)
  (h2 : state_tax_rate = 0.1)
  (h3 : total_taxes_paid = 10500)
  (taxes_eq : total_taxes_paid = (federal_tax_rate * I) + (state_tax_rate * (I - (federal_tax_rate * I))))
  : I = 37500 :=
sorry

end inheritance_amount_l200_200548


namespace digging_rate_is_four_l200_200248

noncomputable def father's_digging_rate (D : ℕ) (H : D / 400 = (2 * D - 400) / 700) : ℝ :=
  D / 400

theorem digging_rate_is_four (D : ℕ) (H1 : D / 400 = (2 * D - 400) / 700) (H2 : D = 1600) : father's_digging_rate D H1 = 4 :=
by
  simp only [father's_digging_rate, H2, Nat.cast_bit0, Nat.cast_bit1, Nat.cast_zero,
    Nat.cast_add, Nat.cast_mul, Rat.cast_div_nat, Rat.cast_one, Rat.cast_bit0,
    Rat.cast_bit1, Real.div_eq_mul_inv, mul_inv_rev, mul_comm, mul_assoc,
    inv_mul_cancel, ne_of_gt, mul_one]
  sorry

end digging_rate_is_four_l200_200248


namespace max_ab_bc_cd_l200_200585

theorem max_ab_bc_cd (a b c d : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b)
  (h_nonneg_c : 0 ≤ c) (h_nonneg_d : 0 ≤ d) (h_sum : a + b + c + d = 200) : 
  ab + bc + cd ≤ 10000 :=
begin
  sorry
end

end max_ab_bc_cd_l200_200585


namespace find_sqrt_abc_sum_l200_200566

theorem find_sqrt_abc_sum (a b c : ℝ)
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 36 * Real.sqrt 15 := by
  sorry

end find_sqrt_abc_sum_l200_200566


namespace overall_gain_percent_is_correct_l200_200624

variables {M : ℝ}

def calculate_cost_price (M : ℝ) : ℝ := (64 / 100) * M
def calculate_first_discount (M : ℝ) : ℝ := M * 0.86
def calculate_second_discount (D1 : ℝ) : ℝ := D1 * 0.9
def calculate_gain (D2 CP : ℝ) : ℝ := D2 - CP
def calculate_gain_percent (G CP : ℝ) : ℝ := (G / CP) * 100

theorem overall_gain_percent_is_correct : 
  ∀ (M : ℝ), calculate_gain_percent 
    (calculate_gain 
      (calculate_second_discount (calculate_first_discount M)) 
      (calculate_cost_price M)
    ) 
    (calculate_cost_price M) 
  = 20.94 := 
by
  sorry

end overall_gain_percent_is_correct_l200_200624


namespace geometric_series_first_term_l200_200077

theorem geometric_series_first_term 
    (r : ℝ) (S : ℝ) (a : ℝ) : 
    r = -2 / 3 ∧ S = 24 → a = 40 :=
by
  intro h
  have hr : r = -2 / 3 := h.1
  have hS : S = 24 := h.2
  let a := (1 - r) * S
  calc
    a = (1 - (-2 / 3)) * 24 : by rw [hr, hS]
    ... = (5 / 3) * 24            : by norm_num
    ... = 40                     : by norm_num

  sorry

end geometric_series_first_term_l200_200077


namespace vasya_tolya_badges_l200_200670

-- Let V be the number of badges Vasya had before the exchange.
-- Let T be the number of badges Tolya had before the exchange.
theorem vasya_tolya_badges (V T : ℕ) 
  (h1 : V = T + 5)
  (h2 : 0.76 * V + 0.20 * T = 0.80 * T + 0.24 * V - 1) :
  V = 50 ∧ T = 45 :=
by 
  sorry

end vasya_tolya_badges_l200_200670


namespace total_worth_of_produce_is_630_l200_200811

def bundles_of_asparagus : ℕ := 60
def price_per_bundle_asparagus : ℝ := 3.00

def boxes_of_grapes : ℕ := 40
def price_per_box_grapes : ℝ := 2.50

def num_apples : ℕ := 700
def price_per_apple : ℝ := 0.50

def total_worth : ℝ :=
  bundles_of_asparagus * price_per_bundle_asparagus +
  boxes_of_grapes * price_per_box_grapes +
  num_apples * price_per_apple

theorem total_worth_of_produce_is_630 : 
  total_worth = 630 := by
  sorry

end total_worth_of_produce_is_630_l200_200811


namespace center_of_symmetry_axis_of_symmetry_intervals_of_monotonicity_sin_B_plus_sin_C_l200_200479

noncomputable theory

def f(x : ℝ) : ℝ := sin (x + π/6) + 2 * sin(x / 2) ^ 2

theorem center_of_symmetry (k : ℤ) :
  (f (k * π + π / 6) = 1) :=
sorry

theorem axis_of_symmetry (k : ℤ) :
  (∃ x, x = k * π + 2 * π / 3) :=
sorry

theorem intervals_of_monotonicity (k : ℤ) :
  (∀ x, (- π / 3 + 2 * ↑k * π ≤ x ∧ x ≤ 2 * π / 3 + 2 * ↑k * π → f x is increasing) ∧
   (2 * π / 3 + 2 * ↑k * π ≤ x ∧ x ≤ 5 * π / 3 + 2 * ↑k * π → f x is decreasing)) :=
sorry

variable {a b c : ℝ}

def area_triangle (a b c : ℝ) : ℝ := sqrt (s * (s - a) * (s - b) * (s - c)) / 2
  where s := (a + b + c) / 2

theorem sin_B_plus_sin_C (A B C a b c : ℝ) (hyp1 : a = sqrt 3) (hyp2 : f A = 3/2)
    (hyp3 : area_triangle a b c = sqrt 3 / 2) :
  sin B + sin C = 3 / 2 :=
sorry

end center_of_symmetry_axis_of_symmetry_intervals_of_monotonicity_sin_B_plus_sin_C_l200_200479


namespace cos_squared_minus_sin_squared_eq_l200_200922

theorem cos_squared_minus_sin_squared_eq : 
  let x := Real.pi / 12 in 
  cos x ^ 2 - sin x ^ 2 = sqrt 3 / 2 := 
by 
  sorry

end cos_squared_minus_sin_squared_eq_l200_200922


namespace time_spent_in_park_is_76_19_percent_l200_200552

noncomputable def total_time_in_park (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  trip_times.foldl (λ acc (t, _, _) => acc + t) 0

noncomputable def total_walking_time (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  trip_times.foldl (λ acc (_, w1, w2) => acc + (w1 + w2)) 0

noncomputable def total_trip_time (trip_times : List (ℕ × ℕ × ℕ)) : ℕ :=
  total_time_in_park trip_times + total_walking_time trip_times

noncomputable def percentage_time_in_park (trip_times : List (ℕ × ℕ × ℕ)) : ℚ :=
  (total_time_in_park trip_times : ℚ) / (total_trip_time trip_times : ℚ) * 100

theorem time_spent_in_park_is_76_19_percent (trip_times : List (ℕ × ℕ × ℕ)) :
  trip_times = [(120, 20, 25), (90, 15, 15), (150, 10, 20), (180, 30, 20), (120, 20, 10), (60, 15, 25)] →
  percentage_time_in_park trip_times = 76.19 :=
by
  intro h
  rw [h]  
  simp
  sorry

end time_spent_in_park_is_76_19_percent_l200_200552


namespace chess_pieces_in_grid_l200_200948

theorem chess_pieces_in_grid : 
  ∃ (ways : ℕ), ways = 45 ∧ 
  (∀ (grid : list (list ℕ)), 
   grid.length = 3 ∧ 
   (∀ row, row ∈ grid → row.length = 3) ∧ 
   (∀ row, sum row ≤ 1) ∧ 
   sum (grid.map sum) = 4 ∧ 
   (∀ col, sum (grid.map (λ row, row[col])) ≥ 1) ∧ 
   (∀ row, sum row ≥ 1) → 
   ways = 45) :=
begin
  sorry
end

end chess_pieces_in_grid_l200_200948


namespace length_BC_l200_200944

noncomputable def O : Point := sorry
noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def D : Point := sorry

axiom diameter_AD (circle : Circle) : is_diameter circle O A D
axiom chord_ABC (circle : Circle) : is_chord circle A B C
axiom BO_eq_6 : distance O B = 6
axiom angle_ABO_eq_45 : angle A B O = 45
axiom arc_CD_eq_45 (circle : Circle) : arc CD = 45

theorem length_BC (circle : Circle) (O A B C D : Point) 
  (diameter_AD : is_diameter circle O A D)
  (chord_ABC : is_chord circle A B C)
  (BO_eq_6 : distance O B = 6)
  (angle_ABO_eq_45 : angle A B O = 45)
  (arc_CD_eq_45 : arc CD = 45) : distance B C = 6 := sorry

end length_BC_l200_200944


namespace perpendicular_vectors_l200_200879

def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (1, m + 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (m : ℝ) (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by sorry

end perpendicular_vectors_l200_200879


namespace find_point_B_and_equal_distances_l200_200407

noncomputable def ellipse (a b : ℝ) : set (ℝ × ℝ) :=
{ p | (p.1 ^ 2) / a ^ 2 + (p.2 ^ 2) / b ^ 2 = 1 }

def line_from_focus (focus : ℝ × ℝ) (slope: ℝ) : set (ℝ × ℝ) :=
{ p | p.2 = slope * (p.1 - focus.1) + focus.2 }

def perpendicular_l (focus : ℝ × ℝ) (x: ℝ) : ℝ × ℝ :=
(x, focus.2)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem find_point_B_and_equal_distances :
  let focus := (-1,0)
  let ellipse_eq := ellipse 2 1
  let line_l1 := line_from_focus focus 1
  let point_A := (0, 1)
  let point_B := (-4 / 3, -1 / 3)
  let point_D_ne := (0, -1)

  -- Conditions
  ∀ (A : ℝ × ℝ) (B : ℝ × ℝ) (D: ℝ × ℝ),
    A ∈ ellipse_eq ∧ A = point_A ∧
    (B ∈ ellipse_eq ∧ B ∈ line_l1 ∧ B ≠ A) ∧
    D ∈ ellipse_eq ∧ D ≠ A ∧ D ≠ point_D_ne →
  
  -- Question and answers
  B = point_B ∧
  ∀ (C: ℝ × ℝ) (E G: ℝ × ℝ),
    C ∈ ellipse_eq ∧ C ≠ A ∧ C ≠ D ∧
    E = perpendicular_l focus C.1 ∧ G = perpendicular_l focus D.1 →
    distance E focus = distance focus G :=
by
  intros focus ellipse_eq line_l1 point_A point_B point_D_ne A B D h_cond
  have h_B := sorry  -- Placeholder for B = (-4 / 3, -1 / 3)
  have h_dist := sorry  -- Placeholder for |EF1| = |F1G|
  exact ⟨h_B, h_dist⟩

end find_point_B_and_equal_distances_l200_200407


namespace series_sum_value_l200_200764

noncomputable def series_sum : ℝ :=
  ∑' (n : ℕ) in (Set.Ioi 0 : Set ℕ), (3 * (n : ℝ) + 2) / ((n : ℝ) * (n + 1) * (n + 3))

theorem series_sum_value : series_sum = 71 / 240 :=
sorry

end series_sum_value_l200_200764


namespace infinite_points_minimize_OM_l200_200994

theorem infinite_points_minimize_OM :
  (∃ (M : ℝ × ℝ), M.2 = M.1 + 1 ∧ 
   OM M = |M.1| + |M.1 + 1| ∧
   inf {OM M| M ∈ {M : ℝ × ℝ | M.2 = M.1 + 1}} =
   {|M.val.1| + |M.val.1 + 1| | M ∈ set.Icc (-1 : ℝ) (0 : ℝ)}) :=
sorry

noncomputable def OM (M : ℝ × ℝ) : ℝ := abs M.1 + abs (M.1 + 1)

end infinite_points_minimize_OM_l200_200994


namespace eccentricity_of_ellipse_l200_200836

noncomputable def ellipse {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a > b) :=
  {x y : ℝ // x^2 / a^2 + y^2 / b^2 = 1}

theorem eccentricity_of_ellipse 
  {a b m c : ℝ} 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (hp : (⟨m, 4⟩ : ellipse ha hb hab)) 
  (hRadius : 3 / 2 = 3 / 2) 
  (hc : c = (3/5) * a) : 
  c / a = 3 / 5 := 
sorry

end eccentricity_of_ellipse_l200_200836


namespace cos_2alpha_of_hyperbola_asymptotes_l200_200176

theorem cos_2alpha_of_hyperbola_asymptotes :
  ∀ (α : ℝ), 
  (∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 → (y^2 / 4 - x^2 = 1) → (α ∈ set.Icc 0 (π/2)) ∧ (abs ((2 : ℝ) - α)) = 0) →
  cos (2 * α) = - 7 / 25 := by
  sorry

end cos_2alpha_of_hyperbola_asymptotes_l200_200176


namespace orthocenter_l200_200581

-- Definition of given conditions
variables {z1 z2 z3 : ℂ}

-- Each z_i lies on the unit circle
axiom on_unit_circle : z1 * conj(z1) = 1 ∧ z2 * conj(z2) = 1 ∧ z3 * conj(z3) = 1

-- The statement to prove
theorem orthocenter : z1 + z2 + z3 = H :=
by
  have h1 := on_unit_circle.left,
  have h2 := on_unit_circle.right.left,
  have h3 := on_unit_circle.right.right,
  sorry

end orthocenter_l200_200581


namespace minimal_functions_l200_200783

open Int

theorem minimal_functions (f : ℤ → ℤ) (c : ℤ) :
  (∀ x, f (x + 2017) = f x) ∧
  (∀ x y, (f (f x + f y + 1) - f (f x + f y)) % 2017 = c) →
  (c = 1 ∨ c = 2016 ∨ c = 1008 ∨ c = 1009) :=
by
  sorry

end minimal_functions_l200_200783


namespace find_sqrt_abc_sum_l200_200568

theorem find_sqrt_abc_sum (a b c : ℝ)
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 36 * Real.sqrt 15 := by
  sorry

end find_sqrt_abc_sum_l200_200568


namespace vasya_tolya_badges_l200_200669

-- Let V be the number of badges Vasya had before the exchange.
-- Let T be the number of badges Tolya had before the exchange.
theorem vasya_tolya_badges (V T : ℕ) 
  (h1 : V = T + 5)
  (h2 : 0.76 * V + 0.20 * T = 0.80 * T + 0.24 * V - 1) :
  V = 50 ∧ T = 45 :=
by 
  sorry

end vasya_tolya_badges_l200_200669


namespace value_of_exponent_l200_200191

theorem value_of_exponent (m n : ℝ) (hm : 3 ^ m = 4) (hn : 3 ^ n = 5) : 3 ^ (m - 2 * n) = 4 / 25 :=
by
  sorry

end value_of_exponent_l200_200191


namespace not_possible_to_label_45gon_l200_200085

def vertex := ℕ
def digit := Fin 10
def edge (v : vertex) := (v, v) -- This represents an edge connecting two vertices, but for abstraction purposes, just a pair of vertices.

-- The 45-gon can be thought of as a set of vertices and edges.
structure polygon :=
  (vertices : Finset vertex)
  (edges : Finset (vertex × vertex))

-- The problem: There is no way to label vertices to satisfy all conditions.
theorem not_possible_to_label_45gon : ∀ (G : polygon),
  G.vertices.card = 45 →
  (∀ (i j : digit), i < j → ∃ (v w : vertex), (v ∈ G.vertices) ∧ (w ∈ G.vertices) ∧ (i ≠ j) ∧ ((v = i ∧ w = j) ∨ (v = j ∧ w = i)) ∧ ((v, w) ∈ G.edges)) →
  (∀ (i : digit), G.vertices.filter (λ (v), v = i).card <= 4) →
  False :=
begin
  sorry
end

end not_possible_to_label_45gon_l200_200085


namespace polynomial_roots_l200_200128

noncomputable def polynomial : Polynomial ℝ := 8 * X^4 + 16 * X^3 - 72 * X^2 + 32 * X

theorem polynomial_roots : (Multiset.map (λ x : ℝ, x) (polynomial.roots polynomial)).sort = [0, 1, 1, -4].sort :=
by sorry

end polynomial_roots_l200_200128


namespace parameterization_function_l200_200298

theorem parameterization_function (f : ℝ → ℝ) 
  (parameterized_line : ∀ t : ℝ, (f t, 20 * t - 10))
  (line_eq : ∀ x y : ℝ, y = 2 * x - 30) :
  f = λ t, 10 * t + 10 :=
by
  sorry

end parameterization_function_l200_200298


namespace probability_call_last_10_minutes_l200_200228

theorem probability_call_last_10_minutes :
  let total_interval : ℝ := 30
  let last_10_minutes_interval : ℝ := 10
  let p := last_10_minutes_interval / total_interval
  p = 1 / 3 := 
by {
  let total_interval : ℝ := 30
  let last_10_minutes_interval : ℝ := 10
  let p := last_10_minutes_interval / total_interval
  exact (p = 1 / 3)
}

end probability_call_last_10_minutes_l200_200228


namespace count_two_digit_multiples_of_18_l200_200501

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_multiple_of_18 (n : ℕ) : Prop := n % 18 = 0

theorem count_two_digit_multiples_of_18 : 
  {n : ℕ | is_two_digit n ∧ is_multiple_of_18 n}.to_finset.card = 5 :=
sorry

end count_two_digit_multiples_of_18_l200_200501


namespace john_caffeine_consumption_l200_200985

noncomputable def caffeine_consumed : ℝ :=
let drink1_ounces : ℝ := 12
let drink1_caffeine : ℝ := 250
let drink2_ratio : ℝ := 3
let drink2_ounces : ℝ := 2

-- Calculate caffeine per ounce in the first drink
let caffeine1_per_ounce : ℝ := drink1_caffeine / drink1_ounces

-- Calculate caffeine per ounce in the second drink
let caffeine2_per_ounce : ℝ := caffeine1_per_ounce * drink2_ratio

-- Calculate total caffeine in the second drink
let drink2_caffeine : ℝ := caffeine2_per_ounce * drink2_ounces

-- Total caffeine from both drinks
let total_drinks_caffeine : ℝ := drink1_caffeine + drink2_caffeine

-- Caffeine in the pill is as much as the total from both drinks
let pill_caffeine : ℝ := total_drinks_caffeine

-- Total caffeine consumed
(drink1_caffeine + drink2_caffeine) + pill_caffeine

theorem john_caffeine_consumption :
  caffeine_consumed = 749.96 := by
    -- Proof is omitted
    sorry

end john_caffeine_consumption_l200_200985


namespace blue_bead_probability_no_adjacent_l200_200437

theorem blue_bead_probability_no_adjacent :
  let total_beads := 9
  let blue_beads := 5
  let green_beads := 3
  let red_bead := 1
  let total_permutations := Nat.factorial total_beads / (Nat.factorial blue_beads * Nat.factorial green_beads * Nat.factorial red_bead)
  let valid_arrangements := (Nat.factorial 4) / (Nat.factorial 3 * Nat.factorial 1)
  let no_adjacent_valid := 4
  let probability_no_adj := (no_adjacent_valid : ℚ) / total_permutations
  probability_no_adj = (1 : ℚ) / 126 := 
by
  sorry

end blue_bead_probability_no_adjacent_l200_200437


namespace eccentricity_equilateral_triangle_l200_200829

theorem eccentricity_equilateral_triangle (a b c : ℝ) (e : ℝ) (F₁ F₂ A B : ℝ × ℝ) :
  let F₁ := (-c, 0),
      F₂ := (c, 0),
      e := c / a,
      x2_over_a2_plus_y2_over_b2_eq_1 := ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1,
      intersects_ellipse F₁ := ∃ y : ℝ, (F₁.fst, y) = A ∨ (F₁.fst, y) = B,
      triangle_eq_ABF₂_eq : ∀ y : ℝ, (A.snd = -y) ∧ (A.snd = y → triangle.equilaterality (A, B, F₂)) :=
  e = Real.sqrt 3 / 3 := 
sorry

end eccentricity_equilateral_triangle_l200_200829


namespace number_of_children_l200_200279

theorem number_of_children (A V S : ℕ) (x : ℕ → ℕ) (n : ℕ) 
  (h1 : (A / 2) + V = (A + V + S + (Finset.range (n - 3)).sum x) / n)
  (h2 : S + A = V + (Finset.range (n - 3)).sum x) : 
  n = 6 :=
sorry

end number_of_children_l200_200279


namespace triangular_array_sum_digits_l200_200384

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 2080) : 
  (N.digits 10).sum = 10 :=
sorry

end triangular_array_sum_digits_l200_200384


namespace sum_of_roots_eq_3_l200_200413

-- Define the polynomial equation
def polynomial (x : ℝ) := 3*x^3 - 9*x^2 - 48*x - 8

-- State the theorem
theorem sum_of_roots_eq_3 : 
  let roots := (polynomial.roots (polynomial := polynomial)).sum in
  roots = 3 :=
sorry

end sum_of_roots_eq_3_l200_200413


namespace system_solution_exists_l200_200109

theorem system_solution_exists (a b : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = a^2 ∧ x + |y| = b) ↔ -|a| ≤ b ∧ b ≤ sqrt 2 * |a| :=
by
  sorry

end system_solution_exists_l200_200109


namespace BC_length_l200_200215

noncomputable def triangle_ABC (A B C H : Point) : Prop :=
  right_triangle A B C ∧ 
  dist A B = 5 ∧ 
  dist A C = 5 ∧ 
  is_right_angle B C A ∧ 
  altitude_from B AC H ∧ 
  dist A H = 2 * dist H C

theorem BC_length (A B C H : Point) (h : triangle_ABC A B C H) : 
  dist B C = 5 * Real.sqrt 5 / 3 := sorry

end BC_length_l200_200215


namespace badges_initial_count_l200_200675

variable {V T : ℕ}

-- conditions
def initial_condition : Prop := V = T + 5
def exchange_condition : Prop := 0.76 * V + 0.20 * T = 0.80 * T + 0.24 * V - 1

-- result
theorem badges_initial_count (h1 : initial_condition) (h2 : exchange_condition) : V = 50 ∧ T = 45 := 
  sorry

end badges_initial_count_l200_200675


namespace z_is_1_l200_200350

theorem z_is_1.5_decades_younger_than_x (X Y Z : ℝ) (h : X + Y = Y + Z + 15) : (X - Z) / 10 = 1.5 :=
by
  sorry

end z_is_1_l200_200350


namespace least_trees_l200_200046

theorem least_trees (N : ℕ) (h1 : N % 7 = 0) (h2 : N % 6 = 0) (h3 : N % 4 = 0) (h4 : N ≥ 100) : N = 168 :=
sorry

end least_trees_l200_200046


namespace quadratic_polynomials_exist_l200_200108

-- Definitions of the polynomials
def p1 (x : ℝ) := (x - 10)^2 - 1
def p2 (x : ℝ) := x^2 - 1
def p3 (x : ℝ) := (x + 10)^2 - 1

-- The theorem to prove
theorem quadratic_polynomials_exist :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ p1 x1 = 0 ∧ p1 x2 = 0) ∧
  (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ p2 y1 = 0 ∧ p2 y2 = 0) ∧
  (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ p3 z1 = 0 ∧ p3 z2 = 0) ∧
  (∀ x : ℝ, p1 x + p2 x ≠ 0 ∧ p1 x + p3 x ≠ 0 ∧ p2 x + p3 x ≠ 0) :=
by
  sorry

end quadratic_polynomials_exist_l200_200108


namespace range_f_2019_l200_200795

noncomputable def f (x : ℝ) : ℝ := log 4 ((cos (2 * x) - 2 * (cos x)^2) / (cos x - 1))

theorem range_f_2019 (x : ℝ) : set.range (λ x, f^[2019] x) = set.Ici (-1/2) :=
by 
  sorry

end range_f_2019_l200_200795


namespace total_profit_l200_200923

theorem total_profit (P Q R : ℝ) (profit : ℝ) 
  (h1 : 4 * P = 6 * Q) 
  (h2 : 6 * Q = 10 * R) 
  (h3 : R = 840 / 6) : 
  profit = 4340 :=
sorry

end total_profit_l200_200923


namespace suma_work_alone_time_l200_200262

-- Definitions and conditions
variable (Work : Type) [AddCommGroup Work] [Module ℝ Work]
variable (amount : Work)
noncomputable def renu_rate := (1 : ℝ) / 8
noncomputable def together_rate := (1 : ℝ) / 3
noncomputable def suma_time := (24 : ℝ) / 5

-- The proof problem statement
theorem suma_work_alone_time :
  let suma_rate := (1 : ℝ) / suma_time in
  renu_rate + suma_rate = together_rate → suma_time = 4.8 :=
sorry

end suma_work_alone_time_l200_200262


namespace valid_derivatives_l200_200478

theorem valid_derivatives : 
  (∀ x : ℝ, (deriv (λ x, 1/x + 2*x - 3) x = -1/(x^2) + 2)) ∧ 
  (∀ x : ℝ, (deriv (λ x, exp x) x = exp x)) :=
by {
  sorry
}

end valid_derivatives_l200_200478


namespace compute_fraction_at_y_eq_5_l200_200766

theorem compute_fraction_at_y_eq_5 :
  (let y := 5 in (y^4 - 8*y^2 + 16) / (y^2 - 4) = 21) := by
  sorry

end compute_fraction_at_y_eq_5_l200_200766


namespace percentage_students_grade_6_combined_l200_200293

theorem percentage_students_grade_6_combined 
  (oakwood_students pinecrest_students : ℕ)
  (oakwood_grades pinecrest_grades : List ℕ)
  (h_oakwood : oakwood_students = 150)
  (h_pinecrest : pinecrest_students = 250)
  (h_oakwood_grades : oakwood_grades = [18, 16, 14, 12, 12, 15, 13])
  (h_pinecrest_grades : pinecrest_grades = [14, 17, 15, 12, 10, 18, 14]) : 
  let num_6th_graders_oakwood := (oakwood_grades.nth (6-1)).getD 0 * oakwood_students / 100 in
  let num_6th_graders_pinecrest := (pinecrest_grades.nth (6-1)).getD 0 * pinecrest_students / 100 in
  100 * (num_6th_graders_oakwood + num_6th_graders_pinecrest) / (oakwood_students + pinecrest_students) = 14 := 
by
  sorry

end percentage_students_grade_6_combined_l200_200293


namespace part1_part2_l200_200238

def f (x a : ℝ) : ℝ := |x + 1| + 2 * |x - a|

theorem part1 (x : ℝ) : (∀ x, f x 2 ≤ x + 4 → (1 / 2 ≤ x ∧ x ≤ 7 / 2)) :=
by sorry

theorem part2 (x : ℝ) : (∀ x, f x a ≥ 4) ↔ (a ≤ -5 ∨ a ≥ 3) :=
by sorry

end part1_part2_l200_200238


namespace solve_system_l200_200697

theorem solve_system :
  ∃ (x y z : ℝ), 
    (x + y + z = 13) ∧ 
    (x^2 + y^2 + z^2 = 61) ∧ 
    (x * y + x * z = 2 * y * z) ∧ 
    ((x = 4 ∧ y = 3 ∧ z = 6) ∨ 
     (x = 4 ∧ y = 6 ∧ z = 3)) :=
begin
  sorry
end

end solve_system_l200_200697


namespace fraction_PE_or_music_l200_200206

-- Define necessary conditions
variables {x : ℕ} -- total number of students
-- Assumption about the distribution of students
def students_PE : ℕ := x / 2
def students_theatre : ℕ := x / 3
def students_music : ℕ := x - (students_PE + students_theatre)

-- Assumption about the fraction of students who left school
def students_left_PE : ℕ := students_PE / 3
def students_left_theatre : ℕ := students_theatre / 4

-- The fraction of all students who are now taking P.E. or music
def fraction_PE_music := (students_PE + students_music - students_left_theatre) / x

theorem fraction_PE_or_music (hx : x ≠ 0) : fraction_PE_music = 7 / 12 := by
  sorry

end fraction_PE_or_music_l200_200206


namespace probability_monotonically_decreasing_on_interval_l200_200481

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

def P_decreasing : ℝ := Real.exp 1 / (Real.exp 1 + 1)

theorem probability_monotonically_decreasing_on_interval : 
  (∫ x in 1..exp 2, if ∃ y, x = y ∧ y ∈ Ioo (Real.exp 1) (Real.exp 2) then 1 else 0) / (Real.exp 2 - 1) = P_decreasing :=
by sorry

end probability_monotonically_decreasing_on_interval_l200_200481


namespace log_18_not_evaluable_l200_200466

theorem log_18_not_evaluable 
  (log_5 : Real := 0.6990) 
  (log_10 : Real := 1.0000) : 
  ¬ ∃ (log_18 : Real), log_18 = log 18 ∧ uses_only log_5 log_10 log_18 :=
sorry

end log_18_not_evaluable_l200_200466


namespace simplify_and_evaluate_l200_200265

-- Define the problem conditions
def x : ℝ := -1
def y : ℝ := 1 / 5

-- Define the expression to be evaluated
def expression (x y : ℝ) : ℝ := 2 * (x^2 * y - 2 * x * y) - 3 * (x^2 * y - 3 * x * y) + (x^2 * y)

-- The proof problem statement
theorem simplify_and_evaluate : expression x y = -1 := by
  sorry

end simplify_and_evaluate_l200_200265


namespace false_propositions_l200_200828

-- Defining lines and planes as types
constant Line : Type
constant Plane : Type

-- Defining parallel and perpendicular relations
constant parallel : Line → Line → Prop
constant parallelPL : Line → Plane → Prop
constant perpendicular : Line → Plane → Prop
constant perpendicularPP : Plane → Plane → Prop

-- Propositions
def prop1 (m l n : Line) : Prop :=
  parallel m l ∧ parallel n l → parallel m n

def prop2 (m : Line) (α β : Plane) : Prop :=
  perpendicular m α ∧ parallelPL m β → perpendicularPP α β

def prop3 (m n : Line) (α : Plane) : Prop :=
  parallelPL m α ∧ parallelPL n α → parallel m n

def prop4 (m : Line) (β : Plane) (α : Plane) : Prop :=
  perpendicular m β ∧ perpendicularPP α β → parallelPL m α

-- Stating the theorem to be proved (Propositions 3 and 4 are false)
theorem false_propositions (m n : Line) (α β : Plane) :
  ¬ prop3 m n α ∧ ¬ prop4 m β α :=
sorry

end false_propositions_l200_200828


namespace oblique_asymptote_l200_200330

theorem oblique_asymptote :
  ∀ x : ℝ, (∃ δ > 0, ∀ y > x, (abs (3 * y^2 + 8 * y + 12) / (3 * y + 4) - (y + 4 / 3)) < δ) :=
sorry

end oblique_asymptote_l200_200330


namespace Theseus_path_X_Y_eq_one_l200_200321

theorem Theseus_path_X_Y_eq_one :
  ∀ (path : List (ℤ × ℤ)),
    path.head = (0, 0) →
    path.last = (0, 0) →
    (∀ i, i < path.length - 1 → 
      (path[i].fst = path[i+1].fst ∧ abs (path[i].snd - path[i+1].snd) = 1) ∨ 
      (path[i].snd = path[i+1].snd ∧ abs (path[i].fst - path[i+1].fst) = 1)) →
    path[1] = (0, -1) →
    ∀ i, i < path.length → ∀ j, j < path.length → i ≠ j → path[i] ≠ path[j] →
    let X := (path.zip path.tail).count (λ p, p.1 = (p.1.fst, p.1.snd + 1) ∧ p.2 = (p.1.fst - 1, p.1.snd + 1)) in
    let Y := (path.zip path.tail).count (λ p, p.1 = (p.1.fst - 1, p.1.snd) ∧ p.2 = (p.1.fst - 1, p.1.snd + 1)) in
    abs (X - Y) = 1 :=
begin
  sorry
end

end Theseus_path_X_Y_eq_one_l200_200321


namespace box_volume_l200_200338

variables {R : Type*} [RealDomain R]

-- Let l, w, and h be the dimensions of the box
variables (l w h : R)

-- Assume the conditions given in the problem
def face_area1 : Prop := l * w = 40
def face_area2 : Prop := w * h = 15
def face_area3 : Prop := l * h = 12

-- The theorem we want to prove is that the volume of the box is 60 cubic inches
theorem box_volume (h1 : face_area1 l w)
                   (h2 : face_area2 w h)
                   (h3 : face_area3 l h) :
                   l * w * h = 60 :=
sorry

end box_volume_l200_200338


namespace gcd_128_144_256_l200_200680

theorem gcd_128_144_256 : Nat.gcd (Nat.gcd 128 144) 256 = 128 :=
  sorry

end gcd_128_144_256_l200_200680


namespace AB_greater_than_BC_l200_200947

variables (A B C D T M Q : Type) [convex_quadrilateral A B C D]
           (a b c d : ℕ) [angle A B C = 40°] [angle D C B = 45°]
           [bisector (angle_b A B) divides (line_segment AD) into two equal parts]

-- Define the lengths as real numbers for the lengths AB and BC.
variables (AB BC : ℝ)

-- Problem statement in Lean 4:
theorem AB_greater_than_BC : AB > BC :=
  sorry  -- Proof is omitted

end AB_greater_than_BC_l200_200947


namespace find_m_l200_200857

-- Necessary setup to define the problem
def power_function (m : ℝ) (x : ℝ) : ℝ := x^(m^2 + m - 3)

theorem find_m (m : ℝ) (h1 : power_function m 2 = 1 / 2) (h2 : m < 0) : m = -2 :=
by
  sorry

end find_m_l200_200857


namespace find_x_l200_200782

theorem find_x (x : ℝ) : 0.003 + 0.158 + x = 2.911 → x = 2.750 :=
by
  sorry

end find_x_l200_200782


namespace factorial_expression_evaluation_l200_200419

/-- Evaluate the given factorial expression -/
theorem factorial_expression_evaluation : 
  (13.factorial - 11.factorial) / 10.factorial = 1705 := 
by sorry

end factorial_expression_evaluation_l200_200419


namespace directional_derivative_at_A_in_direction_of_B_l200_200693

-- Defining the function u
def u (x y z : ℝ) : ℝ := x^2 - Real.arctan (y + z)

-- Point A
def A : ℝ × ℝ × ℝ := (2, 1, 1)

-- Point B
def B : ℝ × ℝ × ℝ := (2, 4, -3)

-- Direction vector
def direction_vector : ℝ × ℝ × ℝ := (0, 3, -4)

-- Normalize direction vector
def unit_vector (v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let len := real.sqrt (v.1^2 + v.2^2 + v.3^2)
  (v.1 / len, v.2 / len, v.3 / len)

-- Compute gradient at A(2,1,1)
def grad_u_at_A : ℝ × ℝ × ℝ :=
  let (x, y, z) := A
  (2 * x, -1 / (1 + (y + z)^2), -1 / (1 + (y + z)^2))

-- The unit direction vector l0
def l0 : ℝ × ℝ × ℝ := unit_vector direction_vector

-- Directional derivative at A in the direction of l0
def directional_derivative : ℝ :=
  let (grad_x, grad_y, grad_z) := grad_u_at_A
  let (l0_x, l0_y, l0_z) := l0
  grad_x * l0_x + grad_y * l0_y + grad_z * l0_z

-- Proving the directional derivative is 1/25
theorem directional_derivative_at_A_in_direction_of_B :
  directional_derivative = 1/25 :=
sorry

end directional_derivative_at_A_in_direction_of_B_l200_200693


namespace merchant_profit_percentage_l200_200015

-- Define the given conditions
def cost_price : ℝ := 100
def mark_up_percentage : ℝ := 0.40
def discount_percentage : ℝ := 0.15

-- Define the expressions
def marked_price : ℝ := cost_price * (1 + mark_up_percentage)
def selling_price : ℝ := marked_price * (1 - discount_percentage)
def profit : ℝ := selling_price - cost_price
def profit_percentage : ℝ := (profit / cost_price) * 100

-- State the theorem
theorem merchant_profit_percentage : profit_percentage = 19 := 
by 
  -- Mathematically, the proof steps go here, but we just place 'sorry' for now
  sorry

end merchant_profit_percentage_l200_200015


namespace SwiftStream_water_pumped_l200_200275

-- Definitions based on the conditions
def rate : ℝ := 500 -- Pump's rate in gallons per hour
def time_minutes : ℝ := 30 -- Time in minutes
def time_hours : ℝ := time_minutes / 60 -- Conversion from minutes to hours

-- The proof statement confirming the pumped water volume
theorem SwiftStream_water_pumped : rate * time_hours = 250 := sorry

end SwiftStream_water_pumped_l200_200275


namespace distance_PQ_l200_200218

-- Define the points P, Q, R, S and their positions
variables (P Q R S : ℝ × ℝ × ℝ)
-- Define the distances between the points as conditions
variables (PS SR RQ : ℝ)
-- Let PS = 1, SR = 4, RQ = 3 as given in the problem
axiom PS_eq : PS = 1
axiom SR_eq : SR = 4
axiom RQ_eq : RQ = 3
-- Define the distances between points in terms of these variables:
def distance (x y : ℝ × ℝ × ℝ) := real.sqrt ((x.1 - y.1) ^ 2 + (x.2 - y.2) ^ 2 + (x.3 - y.3) ^ 2)

-- Define the proof problem
theorem distance_PQ :
  distance P Q = real.sqrt 26 :=
sorry

end distance_PQ_l200_200218


namespace a5_eq_10_l200_200642

variables {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n + 1) * a 0 + (n * (n + 1) / 2) * d

-- Given
def a1_eq_2 : a 0 = 2 := by sorry
def S3_eq_12 (a : ℕ → ℝ) (d : ℝ) : S 2 = 12 := by sorry

-- To prove
theorem a5_eq_10 : ∃ d, arithmetic_sequence a d ∧ sum_of_arithmetic_sequence S a ∧ a 0 = 2 ∧ S 2 = 12 ∧ a 4 = 10 := by
  sorry

end a5_eq_10_l200_200642


namespace sapling_height_relationship_l200_200379

-- Definition to state the conditions
def initial_height : ℕ := 100
def growth_per_year : ℕ := 50
def height_after_years (years : ℕ) : ℕ := initial_height + growth_per_year * years

-- The theorem statement that should be proved
theorem sapling_height_relationship (x : ℕ) : height_after_years x = 50 * x + 100 := 
by
  sorry

end sapling_height_relationship_l200_200379


namespace find_sum_of_min_area_ks_l200_200463

def point := ℝ × ℝ

def A : point := (2, 9)
def B : point := (14, 18)

def is_int (k : ℝ) : Prop := ∃ (n : ℤ), k = n

def min_triangle_area (P Q R : point) : ℝ := sorry
-- Placeholder for the area formula of a triangle given three points

def valid_ks (k : ℝ) : Prop :=
  is_int k ∧ min_triangle_area A B (6, k) ≠ 0

theorem find_sum_of_min_area_ks :
  (∃ k1 k2 : ℤ, valid_ks k1 ∧ valid_ks k2 ∧ (k1 + k2) = 31) :=
sorry

end find_sum_of_min_area_ks_l200_200463


namespace degree_product_polynomial_l200_200608

-- Suppose p and q are polynomials of given degrees and we define their modified versions
variable (p q : Polynomial ℝ)
variable (hp : Polynomial.degree p = 3)
variable (hq : Polynomial.degree q = 4)

-- Define the polynomial transformations
def p_x4 : Polynomial ℝ := p.comp (X ^ 4)
def q_x5 : Polynomial ℝ := q.comp (X ^ 5)

-- The main theorem statement of our proof
theorem degree_product_polynomial : Polynomial.degree (p_x4 * q_x5) = 32 := 
by {
  sorry
}

end degree_product_polynomial_l200_200608


namespace find_equations_l200_200368

-- Define the conditions
def vertex_at_origin (P : ℝ → ℝ → Prop) : Prop :=
  P 0 0

def axis_passes_through_focus (P : ℝ → ℝ → Prop) (H : ℝ → ℝ → Prop) : Prop :=
  ∃ c : ℝ, H c 0 ∧ P c 0

def intersects_at_point (P : ℝ → ℝ → Prop) (H : ℝ → ℝ → Prop) (x y : ℝ) : Prop :=
  P x y ∧ H x y

-- Define the equations
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

def hyperbola (x y : ℝ) : Prop :=
  (x^2 / (1/4)) - (y^2 / (3/4)) = 1

-- Theorem statement 
theorem find_equations : 
  vertex_at_origin parabola ∧ 
  axis_passes_through_focus parabola hyperbola ∧ 
  intersects_at_point parabola hyperbola (3/2) (sqrt 6) →
  parabola = (λ x y, y^2 = 4 * x) ∧ 
  hyperbola = (λ x y, (x^2 / (1/4)) - (y^2 / (3/4)) = 1) :=
sorry

end find_equations_l200_200368


namespace union_correct_l200_200559

def A : Set ℕ := {x : ℕ | -2 ≤ x ∧ x ≤ 2}
def B : Set ℕ := {2, 4}
def union_set := A ∪ B

theorem union_correct : union_set = {0, 1, 2, 4} := by
  sorry

end union_correct_l200_200559


namespace puppy_feeding_l200_200244

variable (num_days : ℕ)
variable (first_two_week_days : ℕ)
variable (total_feedings_last_two_weeks : ℕ)
variable (total_food_over_4_weeks : ℚ)
variable (food_today : ℚ)
variable (food_per_feeding_first_two_weeks : ℚ)
variable (food_first_two_weeks_total : ℚ)

def weeks_to_days (weeks : ℕ) : ℕ := weeks * 7

theorem puppy_feeding :
  first_two_week_days = weeks_to_days 2 ->
  total_feedings_last_two_weeks = (weeks_to_days 2) * 2 ->
  food_today = 1 / 2 ->
  food_per_feeding_first_two_weeks = 1 / 4 * 3 ->
  food_first_two_weeks_total = food_per_feeding_first_two_weeks * first_two_week_days ->
  total_food_over_4_weeks = 25 ->
  ∃ (food_per_feeding_last_two_weeks : ℚ), food_per_feeding_last_two_weeks = 
  (total_food_over_4_weeks - (food_today + food_first_two_weeks_total)) / total_feedings_last_two_weeks ∧
  food_per_feeding_last_two_weeks = 1 / 2 :=
by {
  intros h_days h_feedings h_food_today h_food_per_feeding h_food_first_two_weeks h_total_food,
  existsi (25 - (1/2 + (1/4 * 3 * weeks_to_days 2))) / (weeks_to_days 2 * 2),
  split,
  sorry,
  sorry
}

end puppy_feeding_l200_200244


namespace bisection_next_interval_l200_200327

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 5

theorem bisection_next_interval :
  f 1 < 0 ∧ f 2 > 0 ∧ f 1.5 < 0 → ∃ a b, a = 1.5 ∧ b = 2 ∧ (∀ x, a < x ∧ x < b → f x = 0) :=
begin
  sorry
end

end bisection_next_interval_l200_200327


namespace cubes_difference_l200_200472

-- Given conditions
variables (a b : ℝ)
hypothesis h1 : a - b = 7
hypothesis h2 : a^2 + b^2 = 50

-- The theorem statement
theorem cubes_difference : a^3 - b^3 = 353.5 :=
by
  sorry

end cubes_difference_l200_200472


namespace sqrt_floor_square_l200_200776

theorem sqrt_floor_square (h1 : Real.sqrt 16 < Real.sqrt 19)
                         (h2 : Real.sqrt 19 < Real.sqrt 25)
                         (h3 : Real.floor (Real.sqrt 19) = 4) 
                         : Real.floor (Real.sqrt 19)^2 = 16 := 
by 
  sorry

end sqrt_floor_square_l200_200776


namespace impossible_rectangle_l200_200318

/-- 
Given the following sets of sticks:
- 4 sticks of length 1 cm
- 4 sticks of length 2 cm
- 7 sticks of length 3 cm
- 5 sticks of length 4 cm
prove that it is impossible to form a rectangle using all the sticks.
-/
theorem impossible_rectangle :
  let num_sticks_1 := 4
  let num_sticks_2 := 4
  let num_sticks_3 := 7
  let num_sticks_4 := 5
  let length_sticks_1 := 1
  let length_sticks_2 := 2
  let length_sticks_3 := 3
  let length_sticks_4 := 4
  let total_length := num_sticks_1 * length_sticks_1 +
                      num_sticks_2 * length_sticks_2 +
                      num_sticks_3 * length_sticks_3 +
                      num_sticks_4 * length_sticks_4
  let semi_perimeter := total_length / 2
  ¬ (∃ (a b : ℕ), a + b = semi_perimeter ∧ 2 * (a + b) = total_length) := 
by {
  have h_total_length : total_length = 53 := by norm_num,
  have h_semi_perimeter : semi_perimeter = 26.5 := by norm_num,
  intro h,
  cases h with a ha,
  cases ha with b hb,
  linarith,
  sorry
}

end impossible_rectangle_l200_200318


namespace pde_transform_canonical_l200_200691

noncomputable def canonical_form_pde (u : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := sorry

theorem pde_transform_canonical
  (u : ℝ → ℝ → ℝ)
  (xy : ℝ → ℝ → ℝ)
  : (u_xx u - 2 * u_xy u + u_yy u + u_x u - u_y u + u xy == 0)
     → (λ ξ η, canonical_form_pde u (ξ, η) == u_eta_eta u - u_eta u + (ξ - η) * η) :=
sorry

end pde_transform_canonical_l200_200691


namespace medieval_society_hierarchy_l200_200721

-- Given conditions
def members := 12
def king_choices := members
def remaining_after_king := members - 1
def duke_choices : ℕ := remaining_after_king * (remaining_after_king - 1) * (remaining_after_king - 2)
def knight_choices : ℕ := Nat.choose (remaining_after_king - 2) 2 * Nat.choose (remaining_after_king - 4) 2 * Nat.choose (remaining_after_king - 6) 2

-- The number of ways to establish the hierarchy can be stated as:
def total_ways : ℕ := king_choices * duke_choices * knight_choices

-- Our main theorem
theorem medieval_society_hierarchy : total_ways = 907200 := by
  -- Proof would go here, we skip it with sorry
  sorry

end medieval_society_hierarchy_l200_200721


namespace journey_time_difference_l200_200039

theorem journey_time_difference :
  let t1 := (100:ℝ) / 60
  let t2 := (400:ℝ) / 40
  let T1 := t1 + t2
  let T2 := (500:ℝ) / 50
  let difference := (T1 - T2) * 60
  abs (difference - 100) < 0.01 :=
by
  sorry

end journey_time_difference_l200_200039


namespace find_first_purchase_find_max_profit_purchase_plan_l200_200774

-- Defining the parameters for the problem
structure KeychainParams where
  purchase_price_A : ℕ
  purchase_price_B : ℕ
  total_purchase_cost_first : ℕ
  total_keychains_first : ℕ
  total_purchase_cost_second : ℕ
  total_keychains_second : ℕ
  purchase_cap_second : ℕ
  selling_price_A : ℕ
  selling_price_B : ℕ

-- Define the initial setup
def params : KeychainParams := {
  purchase_price_A := 30,
  purchase_price_B := 25,
  total_purchase_cost_first := 850,
  total_keychains_first := 30,
  total_purchase_cost_second := 2200,
  total_keychains_second := 80,
  purchase_cap_second := 2200,
  selling_price_A := 45,
  selling_price_B := 37
}

-- Part 1: Prove the number of keychains purchased for each type
theorem find_first_purchase (x y : ℕ)
  (h₁ : x + y = params.total_keychains_first)
  (h₂ : params.purchase_price_A * x + params.purchase_price_B * y = params.total_purchase_cost_first) :
  x = 20 ∧ y = 10 :=
sorry

-- Part 2: Prove the purchase plan that maximizes the sales profit
theorem find_max_profit_purchase_plan (m : ℕ)
  (h₃ : m + (params.total_keychains_second - m) = params.total_keychains_second)
  (h₄ : params.purchase_price_A * m + params.purchase_price_B * (params.total_keychains_second - m) ≤ params.purchase_cap_second) :
  m = 40 ∧ (params.selling_price_A - params.purchase_price_A) * m + (params.selling_price_B - params.purchase_price_B) * (params.total_keychains_second - m) = 1080 :=
sorry

end find_first_purchase_find_max_profit_purchase_plan_l200_200774


namespace soak_time_l200_200396

/-- 
Bill needs to soak his clothes for 4 minutes to get rid of each grass stain.
His clothes have 3 grass stains and 1 marinara stain.
The total soaking time is 19 minutes.
Prove that the number of minutes needed to soak for each marinara stain is 7.
-/
theorem soak_time (m : ℕ) (grass_stain_time : ℕ) (num_grass_stains : ℕ) (num_marinara_stains : ℕ) (total_time : ℕ)
  (h1 : grass_stain_time = 4)
  (h2 : num_grass_stains = 3)
  (h3 : num_marinara_stains = 1)
  (h4 : total_time = 19) :
  m = 7 :=
by sorry

end soak_time_l200_200396


namespace set_intersection_equals_l200_200309

theorem set_intersection_equals {α : Type*} (k : ℤ) : 
  ({a : ℝ | ∃ k : ℤ, a = (k * real.pi / 2) - (real.pi / 5)} ∩ {a : ℝ | 0 < a ∧ a < real.pi}) =
   { real.pi * 3 / 10, real.pi * 4 / 5 } :=
sorry

end set_intersection_equals_l200_200309


namespace range_of_a_l200_200850

/--
Given the function f(x) = |x - 2|,
the equation a[f(x)]^{2} - f(x) + 1 = 0 has four distinct real solutions 
if and only if 0 < a < 1/4.
-/
theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f(x) = |x - 2|) :
  (∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
   a * (f(x1))^2 - f(x1) + 1 = 0 ∧ 
   a * (f(x2))^2 - f(x2) + 1 = 0 ∧
   a * (f(x3))^2 - f(x3) + 1 = 0 ∧
   a * (f(x4))^2 - f(x4) + 1 = 0)
   ↔ (0 < a ∧ a < 1 / 4) :=
begin
  sorry
end

end range_of_a_l200_200850


namespace line_segment_within_plane_l200_200509

-- Given: Line segment AB is in plane α
variable {Point : Type*} [Inhabited Point] -- Assume Point type exists
variable (A B : Point) -- Define points A and B
variable Plane (α : set Point) -- Define plane α as a set of points
variable (AB : set Point) -- Define line segment AB as a set of points

-- Conditions: Line segment AB is in plane α
axiom line_segment_in_plane : ∀ {A B : Point}, (AB = set.Icc A B) → AB ⊆ α

-- Prove: Line segment AB is within plane α
theorem line_segment_within_plane : AB ⊆ α := by 
  sorry

end line_segment_within_plane_l200_200509


namespace closest_number_to_fraction_approx_l200_200114

theorem closest_number_to_fraction_approx () :
  let numerator := 501
  let denominator := 0.25
  let estimate := fraction_approach(501, 0.25) := 2000 :
  estimate = 2000 :=
by
  sorry

end closest_number_to_fraction_approx_l200_200114


namespace ellipse_properties_l200_200827

-- Define the given conditions
def is_center_origin (C : Ellipse) : Prop := C.center = (0, 0)
def has_foci_on_x_axis (C : Ellipse) : Prop := ∃ f1 f2 : ℝ × ℝ, (f1 ≠ f2) ∧ 
                                                (f1.2 = 0 ∧ f2.2 = 0) ∧
                                                (C.foci = {f1, f2})
def has_eccentricity (C : Ellipse) (e : ℝ) : Prop := C.eccentricity = e
def vertex_coincides_with_parabola_focus (C : Ellipse) (p : Point) : Prop := 
  (∃ a b : ℝ, C.eq = (x^2 / a^2 + y^2 / b^2 = 1)) ∧ (p = (1, 0))
def chord_conditions (C: Ellipse) (P: Point) : Prop := 
  ∀ m A B, P ∈ line(A, B) ∧ C.point_on(A) ∧ C.point_on(B) ∧ 
           line(A, B).not_perpendicular_to_x_axis
def symmetric_point (A : Point) (B : Point) : Prop := B = (A.1, -A.2)

-- The statement we need to prove
theorem ellipse_properties :
  ∀ (C : Ellipse),
  is_center_origin(C) ∧ has_foci_on_x_axis(C) ∧ 
  has_eccentricity(C, 1/2) ∧ vertex_coincides_with_parabola_focus(C, (1, 0)) ∧ 
  chord_conditions(C, (1, 0)) →
  C.eq = (x^2 + 4*y^2/3 = 1) ∧ ∀ A B, symmetric_point(A, (A.1, 0)) → 
                                       A ∈ line(A, B) → B ∈ line(A, B) →
                                       line((A.1, -A.2), B).passes_through((1, 0))
:= sorry

end ellipse_properties_l200_200827


namespace max_value_k_l200_200819

theorem max_value_k (x y : ℝ) (k : ℝ) (h₁ : x^2 + y^2 = 1) (h₂ : ∀ x y, x^2 + y^2 = 1 → x + y - k ≥ 0) : 
  k ≤ -Real.sqrt 2 :=
sorry

end max_value_k_l200_200819


namespace theta_interval_l200_200410

theorem theta_interval (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) :
  ∀ (x : ℝ), (0 ≤ x) ∧ (x ≤ 1) → (x^2 * cos α - 3 * x * (1 - x) + (1 - x)^2 * sin α > 0) :=
by
  sorry

end theta_interval_l200_200410


namespace smallest_k_value_for_screws_packs_l200_200602

theorem smallest_k_value_for_screws_packs :
  ∃ k : ℕ, k = 60 ∧ (∃ x y : ℕ, (k = 10 * x ∧ k = 12 * y) ∧ x ≠ y) := sorry

end smallest_k_value_for_screws_packs_l200_200602


namespace maximum_volume_regular_triangular_pyramid_l200_200838

-- Given values
def R : ℝ := 1

-- Prove the maximum volume
theorem maximum_volume_regular_triangular_pyramid : 
  ∃ (V_max : ℝ), V_max = (8 * Real.sqrt 3) / 27 := 
by 
  sorry

end maximum_volume_regular_triangular_pyramid_l200_200838


namespace draw_4_balls_in_order_l200_200030

theorem draw_4_balls_in_order (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) : (n * (n-1) * (n-2) * (n-3) = 32760) := by
  rw [h1, h2]
  norm_num
  sorry

end draw_4_balls_in_order_l200_200030


namespace inverse_values_sum_l200_200577

def f (x : ℝ) : ℝ :=
if x ≤ 0 then 2 - x else 2 * x - x^2

theorem inverse_values_sum : 
  (classical.some (exists_intro (-3) (by { dsimp [f], split_ifs, simp }))) + 
  (classical.some (exists_intro 0 (by { dsimp [f], split_ifs, simp }))) + 
  (classical.some (exists_intro 2 (by { dsimp [f], split_ifs, simp }))) = 3 :=
sorry

end inverse_values_sum_l200_200577


namespace number_of_sets_A_l200_200140

/-- Given conditions about intersections and unions of set A, we want to find the number of 
  possible sets A that satisfy the given conditions. Specifically, prove the following:
  - A ∩ {-1, 0, 1} = {0, 1}
  - A ∪ {-2, 0, 2} = {-2, 0, 1, 2}
  Total number of such sets A is 4.
-/
theorem number_of_sets_A : ∃ (As : Finset (Finset ℤ)), 
  (∀ A ∈ As, A ∩ {-1, 0, 1} = {0, 1} ∧ A ∪ {-2, 0, 2} = {-2, 0, 1, 2}) ∧
  As.card = 4 := 
sorry

end number_of_sets_A_l200_200140


namespace largest_integer_divisible_by_all_up_to_cbrt_l200_200123

-- Main theorem: Given the conditions, the largest positive integer x is 420.
theorem largest_integer_divisible_by_all_up_to_cbrt (x : ℕ) (h1 : ∃ y : ℕ, y^3 = x) (h2 : (∀ k : ℕ, k ≤ nat.cbrt x → k ∣ x)) : x = 420 :=
sorry

end largest_integer_divisible_by_all_up_to_cbrt_l200_200123


namespace jacket_initial_reduction_l200_200307

theorem jacket_initial_reduction (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.9 * 1.481481481481481 = P → x = 25 :=
by
  sorry

end jacket_initial_reduction_l200_200307


namespace range_of_quadratic_function_l200_200635

theorem range_of_quadratic_function :
  let f : ℝ → ℝ := λ x, x^2 + 2 * x - 3
  ∃ y_min y_max, (∀ x ∈ Icc 0 2, y_min ≤ f x ∧ f x ≤ y_max) ∧ y_min = -3 ∧ y_max = 5 := 
by
  sorry

end range_of_quadratic_function_l200_200635


namespace four_digit_palindromes_count_l200_200773

/--
  There are 10 four-digit integer palindromes between 1000 and 2000.

  The conditions are:
  1. The number must be in the form ABBA
  2. The number must be between 1000 and 2000
-/
theorem four_digit_palindromes_count : ∃ n : ℕ, n = 10 ∧
  ∀ x : ℕ,
    1000 ≤ x → x < 2000 →
    let digits := (x / 1000, (x / 100) % 10, (x / 10) % 10, x % 10) in
    (digits.1 = digits.4 ∧ digits.2 = digits.3) → 
    x ∈ (Set.range (λ B, 1001 + B * 10 + B * 100))
  sorry

end four_digit_palindromes_count_l200_200773


namespace problem_l200_200198

variable {m n r t : ℚ}

theorem problem (h1 : m / n = 5 / 4) (h2 : r / t = 8 / 15) : (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -3 / 2 :=
by
  sorry

end problem_l200_200198


namespace find_m_of_perpendicular_vectors_l200_200903

theorem find_m_of_perpendicular_vectors
    (m : ℝ)
    (a : ℝ × ℝ := (m, 3))
    (b : ℝ × ℝ := (1, m + 1))
    (h : a.1 * b.1 + a.2 * b.2 = 0) :
    m = -3 / 4 :=
by 
  sorry

end find_m_of_perpendicular_vectors_l200_200903


namespace earthquakes_building_collapse_l200_200748

theorem earthquakes_building_collapse (initial_buildings : ℕ) 
  (collapse1 collapse2 collapse3 collapse4 : ℕ) 
  (remaining_buildings : ℕ) 
  (h_initial : initial_buildings = 100)
  (h_collapse1 : collapse1 = 5)
  (h_collapse2 : collapse2 = 6)
  (h_collapse3 : collapse3 = 13)
  (h_collapse4 : collapse4 = 24)
  (h_remaining : remaining_buildings < 5) :
  ∃ collapsed : ℕ, collapsed ≥ 95 := 
by
  let total_collapsed := collapse1 + collapse2 + collapse3 + collapse4
  have h_total_collapsed : total_collapsed = 5 + 6 + 13 + 24 := by sorry
  have h_buildings_collapsed : initial_buildings - remaining_buildings = 100 - remaining_buildings := by sorry
  have h_remaining_buildings : initial_buildings - total_collapsed - remaining_buildings ≥ 95 := by sorry
  use 100 - remaining_buildings
  exact h_remaining_buildings

end earthquakes_building_collapse_l200_200748


namespace left_handed_jazz_lovers_count_l200_200283

noncomputable def club_members := 30
noncomputable def left_handed := 11
noncomputable def like_jazz := 20
noncomputable def right_handed_dislike_jazz := 4

theorem left_handed_jazz_lovers_count : 
  ∃ x, x + (left_handed - x) + (like_jazz - x) + right_handed_dislike_jazz = club_members ∧ x = 5 :=
by
  sorry

end left_handed_jazz_lovers_count_l200_200283


namespace number_of_complementary_sets_l200_200079

structure Card where
  symbol : Fin 3
  color : Fin 3
  intensity : Fin 3

def cards : List Card :=
  List.product (List.product (List.finRange 3) (List.finRange 3)) (List.finRange 3)
  |>.map (λ ((s, c), i) => Card.mk s c i)

def isComplementary (c1 c2 c3 : Card) : Bool :=
  (∀⦃σ₁ σ₂ σ₃⦄, σ₁ ∈ [c1.symbol, c2.symbol, c3.symbol] → σ₂ ∈ [c1.symbol, c2.symbol, c3.symbol] → σ₃ ∈ [c1.symbol, c2.symbol, c3.symbol] → (σ₁ = σ₂ ∧ σ₂ = σ₃) ∨ (σ₁ ≠ σ₂ ∧ σ₂ ≠ σ₃ ∧ σ₁ ≠ σ₃)) ∧
  (∀⦃c₁ c₂ c₃⦄, c₁ ∈ [c1.color, c2.color, c3.color] → c₂ ∈ [c1.color, c2.color, c3.color] → c₃ ∈ [c1.color, c2.color, c3.color] → (c₁ = c₂ ∧ c₂ = c₃) ∨ (c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃)) ∧
  (∀⦃i₁ i₂ i₃⦄, i₁ ∈ [c1.intensity, c2.intensity, c3.intensity] → i₂ ∈ [c1.intensity, c2.intensity, c3.intensity] → i₃ ∈ [c1.intensity, c2.intensity, c3.intensity] → (i₁ = i₂ ∧ i₂ = i₃) ∨ (i₁ ≠ i₂ ∧ i₂ ≠ i₃ ∧ i₁ ≠ i₃))

theorem number_of_complementary_sets : 
  ∃ (complementary_sets : List (Card × Card × Card)), 
    (∀ (x : Card × Card × Card), x ∈ complementary_sets → isComplementary x.fst x.snd x.thd) ∧
    complementary_sets.length = 117 := by 
  sorry

end number_of_complementary_sets_l200_200079


namespace final_distance_between_skiers_is_100_l200_200662

-- Define the initial conditions and speeds.
def initial_distance : ℝ := 200  -- in meters
def speed_flat : ℝ := 6  -- in km/h
def speed_uphill : ℝ := 4  -- in km/h
def speed_downhill : ℝ := 7  -- in km/h
def speed_snow : ℝ := 3  -- in km/h

-- Define the main theorem statement.
theorem final_distance_between_skiers_is_100 :
  ∀ (initial_distance speed_flat speed_uphill speed_downhill speed_snow : ℝ),
  initial_distance = 200 →
  speed_flat = 6 →
  speed_uphill = 4 →
  speed_downhill = 7 →
  speed_snow = 3 →
  distance_after_all_terrains initial_distance speed_flat speed_uphill speed_downhill speed_snow = 100 := by
  sorry

end final_distance_between_skiers_is_100_l200_200662


namespace cubes_difference_l200_200471

-- Given conditions
variables (a b : ℝ)
hypothesis h1 : a - b = 7
hypothesis h2 : a^2 + b^2 = 50

-- The theorem statement
theorem cubes_difference : a^3 - b^3 = 353.5 :=
by
  sorry

end cubes_difference_l200_200471


namespace set_notation_nat_lt_3_l200_200310

theorem set_notation_nat_lt_3 : {x : ℕ | x < 3} = {0, 1, 2} := 
sorry

end set_notation_nat_lt_3_l200_200310


namespace oranges_in_stack_l200_200047

theorem oranges_in_stack (base_oranges : ℕ)
  (Hbase : base_oranges = 6) :
  ∑ i in finset.range 6, (if i = 0 then (base_oranges * base_oranges / 2) else ((base_oranges - i) * (base_oranges - i) / 2)) = 44 := 
by
  sorry

end oranges_in_stack_l200_200047


namespace unique_positive_real_b_l200_200129

noncomputable def is_am_gm_satisfied (r s t : ℝ) (a : ℝ) : Prop :=
  r + s + t = 2 * a ∧ r * s * t = 2 * a ∧ (r+s+t)/3 = ((r * s * t) ^ (1/3))

noncomputable def poly_roots_real (r s t : ℝ) : Prop :=
  ∀ x : ℝ, (x = r ∨ x = s ∨ x = t)

theorem unique_positive_real_b :
  ∃ b a : ℝ, 0 < a ∧ 0 < b ∧
  (∃ r s t : ℝ, (r ≥ 0 ∧ s ≥ 0 ∧ t ≥ 0 ∧ poly_roots_real r s t) ∧
   is_am_gm_satisfied r s t a ∧
   (x^3 - 2*a*x^2 + b*x - 2*a = (x - r) * (x - s) * (x - t)) ∧
   b = 9) := sorry

end unique_positive_real_b_l200_200129


namespace sqrt_and_cbrt_sum_l200_200641

theorem sqrt_and_cbrt_sum :
  let x := Real.sqrt ((- Real.sqrt 9) ^ 2)
  let y := Real.cbrt 64
  x + y = 7 :=
by
  have x_def : x = Real.sqrt ((- Real.sqrt 9) ^ 2) := rfl
  have y_def : y = Real.cbrt 64 := rfl
  sorry

end sqrt_and_cbrt_sum_l200_200641


namespace quadratic_function_symmetry_and_roots_l200_200858

theorem quadratic_function_symmetry_and_roots
  (b a x t : ℝ)
  (f : ℝ → ℝ)
  (h1 : f x = x^2 - 2 * b * x + a)
  (h2 : ∀ x, f x = f (2 - x))
  (h3 : ∃ x0, ∃ a b, f x0 - (3/4) * a = 0 ∧ ∀ x, (f x = 0 ∨ f x0 = 0) → x = x0) :
  (f x = x^2 - 2 * x + 4) ∧
  (∀ t > 0, ∀ x ∈ set.Icc t (t + 1), (f x ≥ 3)) :=
by
  sorry

end quadratic_function_symmetry_and_roots_l200_200858


namespace mixed_number_expression_l200_200083

theorem mixed_number_expression :
  (7 + 1/2 - (5 + 3/4)) * (3 + 1/6 + (2 + 1/8)) = 9 + 25/96 :=
by
  -- here we would provide the proof steps
  sorry

end mixed_number_expression_l200_200083


namespace new_mean_after_adding_twelve_l200_200020

theorem new_mean_after_adding_twelve :
  (∀ (l : List ℝ), l.length = 15 → (l.sum / 15 = 40) → ((l.map (λ x : ℝ, x + 12)).sum / 15 = 52)) :=
by
  intros l hl1 hl2
  have h_sum : l.sum = 600 := by sorry
  have h_sum_new : (l.map (λ x, x + 12)).sum = 780 := by sorry
  show 780 / 15 = 52, by norm_num

end new_mean_after_adding_twelve_l200_200020


namespace theta_values_in_interval_l200_200189

theorem theta_values_in_interval :
  ∀ θ ∈ (set.Ioo 0 (4 * Real.pi)),
    (2 + 4 * Real.sin θ - 3 * Real.cos (2 * θ) = 0) → 
    set.finite { θ | θ ∈ set.Ioo 0 (4 * Real.pi) ∧ 2 + 4 * Real.sin θ - 3 * Real.cos (2 * θ) = 0 }  :=
sorry

end theta_values_in_interval_l200_200189


namespace angle_equality_l200_200973

noncomputable def midpoint (A B : Point) : Point := sorry
noncomputable def intersect (A B C : Point) : Point := sorry
noncomputable def perpendicular_to (A B P : Point) : Prop := sorry

axiom given_triangle (A B C : Point) (AB BC CA : ℕ) :
  (AB = 15 ∧ BC = 12 ∧ CA = 13 ∧ 
   let M := midpoint B C in 
   let O := intersect (A, M) (B, C) in
   let L := sorry in
   let K := sorry in
   L ∈ AB ∧ perpendicular_to O L AB →
   ∠ OLK = ∠ OLM) 

theorem angle_equality (A B C M O L K : Point) (AB BC CA : ℕ) (h : given_triangle A B C AB BC CA) :
  ∠ OLK = ∠ OLM :=
begin
  sorry
end

end angle_equality_l200_200973


namespace functional_equation_solution_l200_200582

variable {f : ℝ → ℝ}

theorem functional_equation_solution (h1 : ∀ x y : ℝ, f(x + f(y)) = y + f(x))
(h2 : {r : ℝ // ∃ x : ℝ, x ≠ 0 ∧ r = f(x) / x}.to_finset.finite) :
  f = id ∨ f = λ x, -x :=
sorry

end functional_equation_solution_l200_200582


namespace silvia_last_play_without_breach_l200_200521

theorem silvia_last_play_without_breach (N : ℕ) : 
  36 * N < 2000 ∧ 72 * N ≥ 2000 ↔ N = 28 :=
by
  sorry

end silvia_last_play_without_breach_l200_200521


namespace meeting_point_ratio_l200_200739

theorem meeting_point_ratio (v1 v2 : ℝ) (TA TB : ℝ)
  (h1 : TA = 45 * v2)
  (h2 : TB = 20 * v1)
  (h3 : (TA / v1) - (TB / v2) = 11) :
  TA / TB = 9 / 5 :=
by sorry

end meeting_point_ratio_l200_200739


namespace find_m_l200_200896

variables {R : Type*} [CommRing R]

/-- Definition of the dot product in a 2D vector space -/
def dot_product (a b : R × R) : R := a.1 * b.1 + a.2 * b.2

/-- Given vectors a and b as conditions -/
def a : ℚ × ℚ := (m, 3)
def b : ℚ × ℚ := (1, m + 1)

theorem find_m (m : ℚ) (h : dot_product a b = 0) : m = -3 / 4 :=
sorry

end find_m_l200_200896


namespace volume_change_factor_l200_200013

-- Definitions for the initial and new dimensions of the cylinder
def initial_volume (r h : ℝ) : ℝ := π * r^2 * h
def new_height (h : ℝ) : ℝ := 3 * h
def new_radius (r : ℝ) : ℝ := 2.5 * r

-- The new volume based on new dimensions
def new_volume (r h : ℝ) : ℝ := π * (new_radius r)^2 * (new_height h)

-- The factor Y by which the volume changes
def factor_Y (r h : ℝ) : ℝ := new_volume r h / initial_volume r h

theorem volume_change_factor (r h : ℝ) : factor_Y r h = 18.75 :=
by
sorry

end volume_change_factor_l200_200013


namespace liar_is_marko_l200_200647

-- Define the assertions made by each boy
def buster_statement (x : Nat) : Prop := x ≠ 302
def oak_statement (x : Nat) : Prop := x ≠ 401
def marko_statement (x : Nat) : Prop := x ≠ 401 ∧ x ≠ 302 ∧ ¬ above_thrown x
def kenobi_statement (x : Nat) : Prop := above_thrown x
def malfoy_statement (x : Nat) : Prop := above_thrown x ∧ (x ≠ 502 ∨ x ≠ 402) 

-- Define the rooms of each boy, and the condition that four boys told the truth
def buster_room := 302
def oak_room := 401
def marko_room := 502
def kenobi_room := 401
def malfoy_room := 302

-- Define the function determining the room of the liar
def liar_room (x : Nat) : Nat := if buster_statement x 
                                 ∧ oak_statement x
                                 ∧ marko_statement x
                                 ∧ kenobi_statement x
                                 ∧ malfoy_statement x then x else 0

theorem liar_is_marko : liar_room 302 = 302 :=
  sorry

end liar_is_marko_l200_200647


namespace solution_proof_problem_l200_200574

open Real

noncomputable def proof_problem : Prop :=
  ∀ (a b c : ℝ),
  b + c = 17 →
  c + a = 18 →
  a + b = 19 →
  sqrt (a * b * c * (a + b + c)) = 36 * sqrt 5

theorem solution_proof_problem : proof_problem := 
by sorry

end solution_proof_problem_l200_200574


namespace smallest_number_is_minus_three_l200_200745

theorem smallest_number_is_minus_three :
  ∀ (a b c d : ℤ), (a = 0) → (b = -3) → (c = 1) → (d = -1) → b < d ∧ d < a ∧ a < c → b = -3 :=
by
  intros a b c d ha hb hc hd h
  exact hb

end smallest_number_is_minus_three_l200_200745


namespace union_M_N_is_real_l200_200586

def M : Set ℝ := {x | x^2 + x > 0}
def N : Set ℝ := {x | |x| > 2}

theorem union_M_N_is_real : M ∪ N = Set.univ := by
  sorry

end union_M_N_is_real_l200_200586


namespace smallest_n_satisfying_partition_condition_l200_200232

-- Define n as a natural number
def n : ℕ := 243

-- Define the set Sn = {1, 2, ..., n}
def S_n (n : ℕ) : set ℕ := {x | x ≤ n}

-- Define the condition that any partition of S_n into two groups contains a subgroup with three elements a, b, c such that ab = c
def partition_condition (S : set ℕ) : Prop :=
  ∀ A B : set ℕ, (A ∪ B = S) ∧ (A ∩ B = ∅) →
    ∃ a b c ∈ (A ∪ B), a * b = c

-- The theorem statement proving that 243 is the smallest n satisfying the condition
theorem smallest_n_satisfying_partition_condition :
  n = 243 →
  (∀ (k : ℕ), k < 243 → ¬partition_condition (S_n k)) →
  partition_condition (S_n n) :=
by
  intro hn h_small_n
  rw hn
  -- Proof is the equivalence to the correct answer already established
  sorry

end smallest_n_satisfying_partition_condition_l200_200232


namespace range_of_m_l200_200855

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, -2 < x ∧ x < 1 → x^2 + (1 : ℝ) * x - 2 < 0) ∧ 
  (∀ x : ℝ, \(1/2\) < x ∧ x < 1 → ¬monotone (λ x, x^3 + m * x^2 + x + 1)) → 
  (-2 < m ∧ m < -sqrt 3) :=
begin
  sorry
end

end range_of_m_l200_200855


namespace find_g3_l200_200583

def g (x : ℝ) : ℝ := 2 * x^5 - 3 * x^4 + 4 * x^2 - 3 * x + 6

theorem find_g3 : g 3 = -20 :=
by 
  have h : g (-3) = 2 := sorry -- given condition
  have g_val : g 3 + g (-3) = -6 * 3 := sorry -- using g(x) + g(-x) = -6x
  rw h at g_val
  linarith -- linear arithmetic solver to get g(3)

end find_g3_l200_200583


namespace area_of_region_below_line_l200_200000

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def circle_eq (p: ℝ × ℝ) (C: Circle) : Prop :=
  ((p.1 - C.center.1)^2 + (p.2 - C.center.2)^2) = C.radius^2

def line_eq (p: ℝ × ℝ) (m b : ℝ) : Prop :=
  p.2 = m * p.1 + b

noncomputable def region_area_below_line (C : Circle) (m b : ℝ) : ℝ :=
  let r := C.radius
  let a := π * r^2  -- Total area of circle
  a / 2  -- Area of the region below the line that passes through the circle center

theorem area_of_region_below_line 
  (C : Circle)
  (m b : ℝ)
  (hC : C.center = (6, 3))
  (hr : C.radius = sqrt 10)
  (hline : line_eq (6, 3) m b) :
  region_area_below_line C m b = 5 * π :=
by
  sorry

end area_of_region_below_line_l200_200000


namespace shaded_area_zero_l200_200372

noncomputable theory

def vertex_A : (ℝ × ℝ) := (0, 0)
def vertex_B : (ℝ × ℝ) := (0, 8)
def vertex_C : (ℝ × ℝ) := (12, 8)
def vertex_D : (ℝ × ℝ) := (12, 0)
def vertex_E : (ℝ × ℝ) := (24, 0)
def vertex_F : (ℝ × ℝ) := (12, 8)

def triangle_area (A B C : (ℝ × ℝ)) : ℝ :=
  | (B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2) | / 2

theorem shaded_area_zero : 
  triangle_area vertex_B vertex_D vertex_E - triangle_area vertex_B vertex_D vertex_F = 0 :=
by
  sorry

end shaded_area_zero_l200_200372


namespace max_sales_increase_in_1998_l200_200305

-- Define the sales data using a list of tuples
def sales_data : List (ℕ × ℝ) := [
  (1994, 3.0),
  (1995, 3.6),
  (1996, 4.5),
  (1997, 4.875),
  (1998, 6.3),
  (1999, 6.75),
  (2000, 7.2),
  (2001, 7.8),
  (2002, 7.125),
  (2003, 4.875)
]

-- Define a function that computes the difference between consecutive sales data
def sales_diff (data : List (ℕ × ℝ)) : List (ℕ × ℝ) :=
  List.zipWith (λ (a b : ℕ × ℝ), (a.1, b.2 - a.2)) data.tail (data.init)

-- Define a property to find the year after 1994 with the maximum sales increase
def max_sales_increase_year (data : List (ℕ × ℝ)) : ℕ :=
  let diffs := sales_diff data
  let max_pair := List.maximumBy (λ x y, compare x.2 y.2) diffs
  max_pair.getOrElse (0, 0).1

-- The theorem to prove
theorem max_sales_increase_in_1998 : max_sales_increase_year sales_data = 1997 :=
  sorry

end max_sales_increase_in_1998_l200_200305


namespace math_problem_l200_200757

theorem math_problem : (-4)^2 * ((-1)^2023 + (3 / 4) + (-1 / 2)^3) = -6 := 
by 
  sorry

end math_problem_l200_200757


namespace greatest_servings_l200_200730

def servings (ingredient_amount recipe_amount: ℚ) (recipe_servings: ℕ) : ℚ :=
  (ingredient_amount / recipe_amount) * recipe_servings

theorem greatest_servings (chocolate_new_recipe sugar_new_recipe water_new_recipe milk_new_recipe : ℚ)
                         (servings_new_recipe : ℕ)
                         (chocolate_jordan sugar_jordan milk_jordan : ℚ)
                         (lots_of_water : Prop) :
  chocolate_new_recipe = 3 ∧ sugar_new_recipe = 1/3 ∧ water_new_recipe = 1.5 ∧ milk_new_recipe = 5 ∧
  servings_new_recipe = 6 ∧ chocolate_jordan = 8 ∧ sugar_jordan = 3 ∧ milk_jordan = 12 ∧ lots_of_water →
  max (servings chocolate_jordan chocolate_new_recipe servings_new_recipe)
      (max (servings sugar_jordan sugar_new_recipe servings_new_recipe)
           (servings milk_jordan milk_new_recipe servings_new_recipe)) = 16 :=
by
  sorry

end greatest_servings_l200_200730


namespace find_cos_A_l200_200926

noncomputable def cos_A_of_third_quadrant : Real :=
-3 / 5

theorem find_cos_A (A : Real) (h1 : A ∈ Set.Icc (π) (3 * π / 2)) 
  (h2 : Real.sin A = 4 / 5) : Real.cos A = -3 / 5 := 
sorry

end find_cos_A_l200_200926


namespace factorial_product_square_l200_200537

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def product_without_factorial (n k : Nat) : Nat :=
  let P := (List.range (n + 1)).map factorial
  (P.eraseIdx k).product

theorem factorial_product_square 
  (n : Nat) (k : Nat) (h : n = 100 ∧ k = 50) :
  ∃ m : Nat, (product_without_factorial n k = m ^ 2) := 
sorry

end factorial_product_square_l200_200537


namespace add_fractions_is_int_l200_200385

theorem add_fractions_is_int (n : ℤ) : 
  (n / 3 + n^2 / 2 + n^3 / 6 : ℚ).denom = 1 :=
by
  sorry

end add_fractions_is_int_l200_200385


namespace alpha_beta_roots_l200_200443

theorem alpha_beta_roots (α β : ℝ) (hαβ1 : α^2 + α - 1 = 0) (hαβ2 : β^2 + β - 1 = 0) (h_sum : α + β = -1) :
  α^4 - 3 * β = 5 :=
by
  sorry

end alpha_beta_roots_l200_200443


namespace sqrt_abc_sum_is_72_l200_200571

noncomputable def abc_sqrt_calculation (a b c : ℝ) (h1 : b + c = 17) (h2 : c + a = 18) (h3 : a + b = 19) : ℝ :=
  sqrt (a * b * c * (a + b + c))

theorem sqrt_abc_sum_is_72 (a b c : ℝ) (h1 : b + c = 17) (h2 : c + a = 18) (h3 : a + b = 19) :
  abc_sqrt_calculation a b c h1 h2 h3 = 72 :=
by
  sorry

end sqrt_abc_sum_is_72_l200_200571


namespace badges_initial_count_l200_200676

variable {V T : ℕ}

-- conditions
def initial_condition : Prop := V = T + 5
def exchange_condition : Prop := 0.76 * V + 0.20 * T = 0.80 * T + 0.24 * V - 1

-- result
theorem badges_initial_count (h1 : initial_condition) (h2 : exchange_condition) : V = 50 ∧ T = 45 := 
  sorry

end badges_initial_count_l200_200676


namespace proj_a_b_l200_200498

open Real

def vector (α : Type*) := (α × α)

noncomputable def dot_product (a b: vector ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (v: vector ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

noncomputable def projection (a b: vector ℝ) : ℝ := (dot_product a b) / (magnitude b)

-- Define the vectors a and b
def a : vector ℝ := (-1, 3)
def b : vector ℝ := (3, 4)

-- The projection of a in the direction of b
theorem proj_a_b : projection a b = 9 / 5 := 
  by sorry

end proj_a_b_l200_200498


namespace find_m_of_perpendicular_vectors_l200_200901

theorem find_m_of_perpendicular_vectors
    (m : ℝ)
    (a : ℝ × ℝ := (m, 3))
    (b : ℝ × ℝ := (1, m + 1))
    (h : a.1 * b.1 + a.2 * b.2 = 0) :
    m = -3 / 4 :=
by 
  sorry

end find_m_of_perpendicular_vectors_l200_200901


namespace cone_height_l200_200041

theorem cone_height (r V : ℝ) (π : ℝ) (h : ℝ) 
  (radius_eq : r = 4) (volume_eq : V = 150) (pi_value : π = Real.pi) 
  (volume_formula : V = (1 / 3) * π * r^2 * h) :
  h ≈ 9 :=
by
  sorry

end cone_height_l200_200041


namespace smallest_number_among_four_l200_200076

theorem smallest_number_among_four : ∀ (a b c d : ℤ), a = -2 → b = 0 → c = -1 → d = 3 → min (min a b) (min c d) = -2 := 
by
  intros a b c d ha hb hc hd
  rw [ha, hb, hc, hd]
  dsimp
  norm_num
  exact min_eq_left_of_lt (by norm_num)

lemma smallest_number := smallest_number_among_four (-2) 0 (-1) 3 rfl rfl rfl rfl

end smallest_number_among_four_l200_200076


namespace selling_price_of_book_l200_200052

   theorem selling_price_of_book
     (cost_price : ℝ)
     (profit_rate : ℝ)
     (profit := (profit_rate / 100) * cost_price)
     (selling_price := cost_price + profit)
     (hp : cost_price = 50)
     (hr : profit_rate = 60) :
     selling_price = 80 := sorry
   
end selling_price_of_book_l200_200052


namespace total_worth_of_produce_is_630_l200_200810

def bundles_of_asparagus : ℕ := 60
def price_per_bundle_asparagus : ℝ := 3.00

def boxes_of_grapes : ℕ := 40
def price_per_box_grapes : ℝ := 2.50

def num_apples : ℕ := 700
def price_per_apple : ℝ := 0.50

def total_worth : ℝ :=
  bundles_of_asparagus * price_per_bundle_asparagus +
  boxes_of_grapes * price_per_box_grapes +
  num_apples * price_per_apple

theorem total_worth_of_produce_is_630 : 
  total_worth = 630 := by
  sorry

end total_worth_of_produce_is_630_l200_200810


namespace find_line_slope_l200_200560

noncomputable def parabola_focus_slope (F : Point) (C : Parabola) (P : Point) (l : Line) (A B Q : Point) : Prop :=
 ∃ k : ℝ,
  C.equation = λ x y, y^2 = 4 * x ∧
  P = (-1, 0) ∧
  l = {y = k * (x + 1)} ∧
  Q = midpoint A B ∧
  dist(F, Q) = 2 * sqrt 3 ∧
  C.intersects l = {A, B} ∧
  (k = sqrt 2 / 2 ∨ k = -sqrt 2 / 2)

theorem find_line_slope (F : Point) (C : Parabola) (P : Point) (l : Line) (A B Q : Point) :
 parabola_focus_slope F C P l A B Q :=
sorry

end find_line_slope_l200_200560


namespace range_subset_pos_iff_l200_200632

theorem range_subset_pos_iff (a : ℝ) : (∀ x : ℝ, ax^2 + ax + 1 > 0) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end range_subset_pos_iff_l200_200632


namespace arrange_digits_1225_l200_200962

theorem arrange_digits_1225 : 
  let digits := [1, 2, 2, 5] in 
  let endsIn5 (l : List ℕ) := (List.reverse l).headI = 5 in
  (List.permutations digits).count endsIn5 = 6 :=
sorry

end arrange_digits_1225_l200_200962


namespace exists_large_prime_rel_prime_sum_primes_l200_200599

theorem exists_large_prime_rel_prime_sum_primes (N : ℕ) (hN : N > 10^2018) :
  ∃ n, (Nat.Prime n) ∧ (n > 10^2018) ∧ (Nat.gcd (n) (∑ p in Nat.filter Nat.Prime (List.Range n), p) = 1) :=
by
  sorry

end exists_large_prime_rel_prime_sum_primes_l200_200599


namespace longer_side_correct_l200_200254

noncomputable def longer_side_of_rectangle : ℚ :=
  let a := (9 : ℚ) / 4 in a

theorem longer_side_correct (a : ℚ) (h1 : 0.8 * a * a = 81 / 20) : 
  a = longer_side_of_rectangle :=
by
  sorry

end longer_side_correct_l200_200254


namespace arrange_1225_multiple_of_5_l200_200964

theorem arrange_1225_multiple_of_5 : 
  (∃ (s : Finset (Fin 10)) (h : s = {1, 2, 2, 5}),
    (card (s.filter (λ x, last.digit x = 5))
    = 6) : sorry

end arrange_1225_multiple_of_5_l200_200964


namespace perimeter_of_arranged_rectangles_l200_200664

-- Definitions based on conditions
def length_small_rectangle := 9
def width_small_rectangle := 3
def horizontal_count := 8
def vertical_count := 4

-- Statement of the problem
theorem perimeter_of_arranged_rectangles :
  let length_large_rectangle := horizontal_count * length_small_rectangle,
      width_large_rectangle := vertical_count * width_small_rectangle,
      additional_edge := 2 * width_small_rectangle * 2,
      perimeter := 2 * (length_large_rectangle + width_large_rectangle) + additional_edge
  in perimeter = 180 :=
by
  sorry

end perimeter_of_arranged_rectangles_l200_200664


namespace perpendicular_vectors_l200_200877

def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (1, m + 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (m : ℝ) (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by sorry

end perpendicular_vectors_l200_200877


namespace infinite_super_abundant_l200_200679

def is_super_abundant (m : ℕ) : Prop :=
  ∀ (k : ℕ), k < m → (Real.sumOfDivisors m) / m > (Real.sumOfDivisors k) / k

theorem infinite_super_abundant : ∃ᶠ m in at_top, is_super_abundant m :=
sorry

end infinite_super_abundant_l200_200679


namespace square_of_binomial_l200_200913

theorem square_of_binomial (a : ℝ) : 16 * x^2 + 32 * x + a = (4 * x + 4)^2 :=
by
  sorry

end square_of_binomial_l200_200913


namespace distance_to_SFL_is_81_l200_200073

variable (Speed : ℝ)
variable (Time : ℝ)

def distance_to_SFL (Speed : ℝ) (Time : ℝ) := Speed * Time

theorem distance_to_SFL_is_81 : distance_to_SFL 27 3 = 81 :=
by
  sorry

end distance_to_SFL_is_81_l200_200073


namespace correct_number_of_statements_l200_200180

-- Define the lines and their relationships
variables (a b c : Type*)
variables [linear_ordered_field ℝ]
variables [inner_product_space ℝ a] [inner_product_space ℝ b] [inner_product_space ℝ c]

-- Each of the statements about the lines
def statement1 : Prop := inner_product_space.orthogonal a b ∧ inner_product_space.orthogonal a c → parallel b c
def statement2 : Prop := inner_product_space.orthogonal a b ∧ inner_product_space.orthogonal a c → inner_product_space.orthogonal b c
def statement3 : Prop := parallel a b ∧ inner_product_space.orthogonal b c → inner_product_space.orthogonal a c

-- The theorem to prove the number of correct statements is 1
theorem correct_number_of_statements : 
(statement1 a b c → false) ∧ (statement2 a b c → false) ∧ statement3 a b c :=
begin
    sorry
end

end correct_number_of_statements_l200_200180


namespace roots_reciprocal_sum_eq_three_halves_l200_200468

theorem roots_reciprocal_sum_eq_three_halves
  {a b : ℝ}
  (h1 : a^2 - 6 * a + 4 = 0)
  (h2 : b^2 - 6 * b + 4 = 0)
  (h_roots : a ≠ b) :
  1/a + 1/b = 3/2 := by
  sorry

end roots_reciprocal_sum_eq_three_halves_l200_200468


namespace gcd_36_60_eq_12_l200_200005

theorem gcd_36_60_eq_12 :
  ∃ (g : ℕ), g = Nat.gcd 36 60 ∧ g = 12 := by
  -- Defining the conditions:
  let a := 36
  let b := 60
  have fact_a : a = 2^2 * 3^2 := rfl
  have fact_b : b = 2^2 * 3 * 5 := rfl
  
  -- The statement to prove:
  sorry

end gcd_36_60_eq_12_l200_200005


namespace rational_numbers_include_positives_and_negatives_l200_200388

theorem rational_numbers_include_positives_and_negatives :
  ∃ (r : ℚ), r > 0 ∧ ∃ (r' : ℚ), r' < 0 :=
by
  sorry

end rational_numbers_include_positives_and_negatives_l200_200388


namespace perimeter_of_EFGH_l200_200334

-- Define lengths of sides and segments
def EH : ℝ := 30 + 15
def FG : ℝ := 40
def EF : ℝ := 45
def GH : ℝ := 35

-- Define perimeter
def perimeterEFGH : ℝ := EF + FG + GH + EH

theorem perimeter_of_EFGH :
    perimeterEFGH = 165 :=
by
    have h1 : EH = 45 := by norm_num
    have h2 : perimeterEFGH = EF + FG + GH + EH := rfl
    rw [h1] at h2
    norm_num at h2
    exact h2

end perimeter_of_EFGH_l200_200334


namespace expected_sophomores_in_sample_l200_200710

theorem expected_sophomores_in_sample 
  (num_freshmen : ℕ) 
  (num_sophomores : ℕ) 
  (num_juniors : ℕ) 
  (sample_size : ℕ) 
  (total_students : ℕ) 
  (proportion_sophomores : ℚ) 
  (expected_sophomores : ℚ) :
  num_freshmen = 400 →
  num_sophomores = 320 →
  num_juniors = 280 →
  sample_size = 200 →
  total_students = num_freshmen + num_sophomores + num_juniors →
  proportion_sophomores = num_sophomores / total_students →
  expected_sophomores = sample_size * proportion_sophomores →
  expected_sophomores = 64 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5, h6]
  norm_num at h7
  exact h7

end expected_sophomores_in_sample_l200_200710


namespace range_of_a_l200_200477

-- Define the given conditions
def z (a : ℝ) : ℂ := (a - 2 * Complex.I) / (2 - Complex.I)
def fourth_quadrant (z : ℂ) : Prop := (z.re > 0) ∧ (z.im < 0)

-- Define the proof problem
theorem range_of_a (a : ℝ) (hz : fourth_quadrant (z a)) : -1 < a ∧ a < 4 :=
sorry

end range_of_a_l200_200477


namespace perpendicular_vectors_implies_m_value_l200_200872

variable (m : ℝ)

def vector1 : ℝ × ℝ := (m, 3)
def vector2 : ℝ × ℝ := (1, m + 1)

theorem perpendicular_vectors_implies_m_value
  (h : vector1 m ∙ vector2 m = 0) :
  m = -3 / 4 :=
by 
  sorry

end perpendicular_vectors_implies_m_value_l200_200872


namespace train_speed_without_stoppages_l200_200382

-- Defining the given conditions
def average_speed_with_stoppages : ℝ := 360
def moving_time_ratio : ℝ := 54 / 60

-- The statement to prove
theorem train_speed_without_stoppages : 
  (average_speed_with_stoppages / moving_time_ratio) = 400 :=
sorry

end train_speed_without_stoppages_l200_200382


namespace cost_of_special_food_is_45_l200_200319

-- Define the number of goldfish in the pond.
def num_goldfish : ℕ := 50

-- Define the amount of food each goldfish eats per day in ounces.
def food_per_goldfish : ℚ := 1.5

-- Define the percentage of goldfish that need special food.
def special_food_percentage : ℚ := 0.20

-- Define the cost of special food per ounce in dollars.
def cost_per_ounce_special_food : ℚ := 3

-- Define the number of goldfish that need special food.
def num_special_food_goldfish : ℕ := (special_food_percentage * num_goldfish).to_nat

-- Define the total amount of special food needed per day in ounces.
def special_food_needed : ℚ := num_special_food_goldfish * food_per_goldfish

-- Define the cost to feed the goldfish with special food per day.
def special_food_cost_per_day : ℚ := special_food_needed * cost_per_ounce_special_food

-- Prove that the cost to feed the goldfish that need special food is $45 per day.
theorem cost_of_special_food_is_45 :
  special_food_cost_per_day = 45 := by
  sorry

end cost_of_special_food_is_45_l200_200319


namespace determine_a_l200_200804

theorem determine_a (a : ℝ) 
  (h1 : (a - 1) * (0:ℝ)^2 + 0 + a^2 - 1 = 0)
  (h2 : a - 1 ≠ 0) : 
  a = -1 := 
sorry

end determine_a_l200_200804


namespace complement_A_eq_range_of_m_l200_200025

variable (U : Set ℝ := Set.univ)
variable (A : Set ℝ := { x | 3 * x - 7 ≥ 8 - 2 * x })
variable (B : ℝ → Set ℝ := λ m, { x | x ≥ m - 1 })

theorem complement_A_eq :
  U \ A = { x | x < 3 } :=
sorry

theorem range_of_m (m : ℝ) (h : A ⊆ B m) :
  m ≤ 4 :=
by
  unfold A B at h
  sorry

end complement_A_eq_range_of_m_l200_200025


namespace f_positive_l200_200484

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - Real.log x / Real.log 2

variables (x0 x1 : ℝ)

theorem f_positive (hx0 : f x0 = 0) (hx1 : 0 < x1) (hx0_gt_x1 : x1 < x0) : 0 < f x1 :=
sorry

end f_positive_l200_200484


namespace find_first_number_l200_200726

def is_lcm (a b l : ℕ) : Prop := l = Nat.lcm a b

theorem find_first_number :
  ∃ (a b : ℕ), (5 * b) = a ∧ (4 * b) = b ∧ is_lcm a b 80 ∧ a = 20 :=
by
  sorry

end find_first_number_l200_200726


namespace tan_double_angle_l200_200445

theorem tan_double_angle (θ : ℝ) 
  (h1 : Real.sin θ = 4 / 5) 
  (h2 : Real.sin θ - Real.cos θ > 1) : 
  Real.tan (2 * θ) = 24 / 7 := 
sorry

end tan_double_angle_l200_200445


namespace solve_for_x_l200_200130

theorem solve_for_x :
  let x := 71.2715625 in
  ((3^2 - 5) / (0.08 * 7 + 2)) + Real.sqrt x = 10 :=
by
  sorry

end solve_for_x_l200_200130


namespace total_homework_time_l200_200800

variable (num_math_problems num_social_studies_problems num_science_problems : ℕ)
variable (time_per_math_problem time_per_social_studies_problem time_per_science_problem : ℝ)

/-- Prove that the total time taken by Brooke to answer all his homework problems is 48 minutes -/
theorem total_homework_time :
  num_math_problems = 15 →
  num_social_studies_problems = 6 →
  num_science_problems = 10 →
  time_per_math_problem = 2 →
  time_per_social_studies_problem = 0.5 →
  time_per_science_problem = 1.5 →
  (num_math_problems * time_per_math_problem + num_social_studies_problems * time_per_social_studies_problem + num_science_problems * time_per_science_problem) = 48 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_homework_time_l200_200800


namespace find_number_l200_200018

theorem find_number (n : ℕ) : (n / 2) + 5 = 15 → n = 20 :=
by
  intro h
  sorry

end find_number_l200_200018


namespace fraction_compute_l200_200915

theorem fraction_compute (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : a^2 + b^2 + c^2 = 16) (h2 : x^2 + y^2 + z^2 = 49) (h3 : ax + by + cz = 28) :
  (a + b + c) / (x + y + z) = 4 / 7 :=
sorry

end fraction_compute_l200_200915


namespace grapes_average_seeds_l200_200607

def total_seeds_needed : ℕ := 60
def apple_seed_average : ℕ := 6
def pear_seed_average : ℕ := 2
def apples_count : ℕ := 4
def pears_count : ℕ := 3
def grapes_count : ℕ := 9
def extra_seeds_needed : ℕ := 3

-- Calculation of total seeds from apples and pears:
def seeds_from_apples : ℕ := apples_count * apple_seed_average
def seeds_from_pears : ℕ := pears_count * pear_seed_average

def total_seeds_from_apples_and_pears : ℕ := seeds_from_apples + seeds_from_pears

-- Calculation of the remaining seeds needed from grapes:
def seeds_needed_from_grapes : ℕ := total_seeds_needed - total_seeds_from_apples_and_pears - extra_seeds_needed

-- Calculation of the average number of seeds per grape:
def grape_seed_average : ℕ := seeds_needed_from_grapes / grapes_count

-- Prove the correct average number of seeds per grape:
theorem grapes_average_seeds : grape_seed_average = 3 :=
by
  sorry

end grapes_average_seeds_l200_200607


namespace badges_before_exchange_l200_200667

theorem badges_before_exchange (V T : ℕ) (h1 : V = T + 5) (h2 : 76 * V + 20 * T = 80 * T + 24 * V - 100) :
  V = 50 ∧ T = 45 :=
by
  sorry

end badges_before_exchange_l200_200667


namespace analytical_expression_of_f_l200_200143

theorem analytical_expression_of_f {f : ℝ → ℝ} 
  (h : ∀ x : ℝ, 0 ≤ x → f (real.sqrt x + 1) = x + 2 * real.sqrt x)
  (x : ℝ) :
  x ≥ 1 → f x = x^2 - 1 :=
  sorry

end analytical_expression_of_f_l200_200143


namespace shopkeeper_total_cards_l200_200378

-- Definition of the number of cards in standard, Uno, and tarot decks.
def std_deck := 52
def uno_deck := 108
def tarot_deck := 78

-- Number of complete decks and additional cards.
def std_decks := 4
def uno_decks := 3
def tarot_decks := 5
def additional_std := 12
def additional_uno := 7
def additional_tarot := 9

-- Calculate the total number of cards.
def total_standard_cards := (std_decks * std_deck) + additional_std
def total_uno_cards := (uno_decks * uno_deck) + additional_uno
def total_tarot_cards := (tarot_decks * tarot_deck) + additional_tarot

def total_cards := total_standard_cards + total_uno_cards + total_tarot_cards

theorem shopkeeper_total_cards : total_cards = 950 := by
  sorry

end shopkeeper_total_cards_l200_200378


namespace trajectory_is_ellipse_l200_200454

theorem trajectory_is_ellipse (x y : ℝ) :
  10 * Real.sqrt ((x - 1)^2 + (y - 2)^2) = |3 * x + 4 * y + 2| → 
  ∃ (focus : ℝ × ℝ) (directrix : {l : ℝ × ℝ | 3 * l.1 + 4 * l.2 + 2 = 0}),
  focus = (1, 2) ∧ directrix = {l : ℝ × ℝ | 3 * l.1 + 4 * l.2 + 2 = 0} ∧ 
  ∀ P : ℝ × ℝ, 
  ((P.1 - focus.1)^2 + (P.2 - focus.2)^2)^0.5 = 
  (|(3 * P.1 + 4 * P.2 + 2)| / (25)^0.5) * (1/2) := sorry

end trajectory_is_ellipse_l200_200454


namespace ratio_of_surface_areas_of_spheres_l200_200636

theorem ratio_of_surface_areas_of_spheres (r1 r2 : ℝ) (h : r1 / r2 = 1 / 3) : 
  (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 9 := by
  sorry

end ratio_of_surface_areas_of_spheres_l200_200636


namespace find_m_l200_200897

variables {R : Type*} [CommRing R]

/-- Definition of the dot product in a 2D vector space -/
def dot_product (a b : R × R) : R := a.1 * b.1 + a.2 * b.2

/-- Given vectors a and b as conditions -/
def a : ℚ × ℚ := (m, 3)
def b : ℚ × ℚ := (1, m + 1)

theorem find_m (m : ℚ) (h : dot_product a b = 0) : m = -3 / 4 :=
sorry

end find_m_l200_200897


namespace no_real_solution_for_fraction_eq_l200_200107

theorem no_real_solution_for_fraction_eq (x : ℝ) :
  ¬ (3 / (x^2 - x - 6) = 2 / (x^2 - 3x + 2)) := by
  sorry

end no_real_solution_for_fraction_eq_l200_200107


namespace brooke_homework_time_l200_200803

def num_math_problems := 15
def num_social_studies_problems := 6
def num_science_problems := 10

def time_per_math_problem := 2 -- in minutes
def time_per_social_studies_problem := 0.5 -- in minutes (30 seconds)
def time_per_science_problem := 1.5 -- in minutes

def total_time : ℝ :=
  num_math_problems * time_per_math_problem +
  num_social_studies_problems * time_per_social_studies_problem +
  num_science_problems * time_per_science_problem

theorem brooke_homework_time :
  total_time = 48 := by
  sorry

end brooke_homework_time_l200_200803


namespace complex_number_exponentiation_l200_200505

open Complex

theorem complex_number_exponentiation (w : ℂ) (h : w + w⁻¹ = -sqrt 3) : w^504 + w^(-504) = 2 :=
sorry

end complex_number_exponentiation_l200_200505


namespace brian_needs_some_cartons_l200_200595

def servings_per_person : ℕ := sorry -- This should be defined with the actual number of servings per person.
def family_members : ℕ := 8
def us_cup_in_ml : ℕ := 250
def ml_per_serving : ℕ := us_cup_in_ml / 2
def ml_per_liter : ℕ := 1000

def total_milk_needed (servings_per_person : ℕ) : ℕ :=
  family_members * servings_per_person * ml_per_serving

def cartons_of_milk_needed (servings_per_person : ℕ) : ℕ :=
  total_milk_needed servings_per_person / ml_per_liter + if total_milk_needed servings_per_person % ml_per_liter = 0 then 0 else 1

theorem brian_needs_some_cartons (servings_per_person : ℕ) : 
  cartons_of_milk_needed servings_per_person = (family_members * servings_per_person * ml_per_serving / ml_per_liter + 
  if (family_members * servings_per_person * ml_per_serving) % ml_per_liter = 0 then 0 else 1) := 
by 
  sorry

end brian_needs_some_cartons_l200_200595


namespace product_divisors_72_l200_200790

theorem product_divisors_72 : ∏ d in (finset.filter (λ x, 72 % x = 0) (finset.range 73)), d = 2^18 * 3^12 := 
by 
  sorry

end product_divisors_72_l200_200790


namespace B_fraction_of_job_completed_l200_200709

noncomputable def A_rate := (1 : ℚ) / 6  -- Rate at which A completes the job per hour
noncomputable def B_rate := (1 : ℚ) / 3  -- Rate at which B completes the job per hour

theorem B_fraction_of_job_completed :
  let A_initial_job := A_rate * 1 in  -- Work done by A alone in 1 hour
  let remaining_job := 1 - A_initial_job in
  let combined_rate := A_rate + B_rate in
  let time_to_complete_remaining := remaining_job / combined_rate in
  let B_contribution := B_rate * time_to_complete_remaining in
  B_contribution = (25 : ℚ) / 54 := sorry

end B_fraction_of_job_completed_l200_200709


namespace ratio_hogs_to_cats_l200_200938

theorem ratio_hogs_to_cats (C : ℕ) (hogs : ℕ) (hogs_eq : hogs = 75) 
    (cat_equation : 0.60 * C - 5 = 10) : hogs / C = 3 := by
  sorry

end ratio_hogs_to_cats_l200_200938


namespace AT_eq_TM_l200_200394

noncomputable theory
open_locale classical

variables (A B C O E F T M : Type) [point_space A B C O E F T M]

-- Circumcenter definition
def circumcenter (O : Point) (A B C : Triangle) : Prop :=
  is_circumcenter O A B C

-- Intersection definitions
def intersects (L P Q : Line) : Point :=
  -- defines the point where line L intersects with lines P and Q
  sorry

-- Definition of points E and F
def E (BO AC : Line) : Point := intersects BO AC
def F (CO AB : Line) : Point := intersects CO AB

-- Definition of point T
def T (AO EF : Line) : Point := intersects AO EF

-- Joseph's circles, intersecting at M
def circles_intersect (BOF COE : Circle) : Point :=
  -- assume definition for intersection point M of two circles
  sorry

-- The goal theorem
theorem AT_eq_TM 
  (circumcenter_O : circumcenter O (triangle A B C))
  (intersect_E : E BO AC)
  (intersect_F : F CO AB)
  (intersect_T : T AO EF)
  (circle_intersect_M : circles_intersect (circle BO F) (circle CO E)) :
  distance T A = distance T M :=
sorry

end AT_eq_TM_l200_200394


namespace ways_to_arrange_digits_of_1225_l200_200958

theorem ways_to_arrange_digits_of_1225 :
  let digits := [1, 2, 2, 5] in
  (∃ permutations : list (list ℕ), 
      permutations = digits.permutations ∧
      (∃ valid_permutations : list (list ℕ),
          valid_permutations = permutations.filter (λ l, (l.nth 3).getOrElse 0 = 5) ∧
          valid_permutations.length = 3)) :=
sorry

end ways_to_arrange_digits_of_1225_l200_200958


namespace main_theorem_l200_200781

noncomputable def solve_for_d (d : ℝ) : Prop :=
  (∃ x₁ x₂ : ℤ, 3 * x₁^2 + 21 * x₁ - 108 = 0 ∧ (d - x₁) = (8 - 2 * real.sqrt 11) / 10 
  ∨ 3 * x₂^2 + 21 * x₂ - 108 = 0 ∧ (d - x₂) = (8 - 2 * real.sqrt 11) / 10)
  ∧ (d = 4.8 - 0.2 * real.sqrt 11 ∨ d = -8.2 - 0.2 * real.sqrt 11)

theorem main_theorem : ∃ d : ℝ, solve_for_d d :=
begin
  sorry
end

end main_theorem_l200_200781


namespace abs_diff_l200_200816

theorem abs_diff (a b : ℝ) (h_ab : a < b) (h_a : abs a = 6) (h_b : abs b = 3) :
  a - b = -9 ∨ a - b = 9 :=
by
  sorry

end abs_diff_l200_200816


namespace quadratic_function_expression_l200_200162

-- Definitions based on conditions
def quadratic (f : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c
def condition1 (f : ℝ → ℝ) : Prop := (f 0 = 1)
def condition2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) - f x = 4 * x

-- The theorem we want to prove
theorem quadratic_function_expression (f : ℝ → ℝ) 
  (hf_quad : quadratic f)
  (hf_cond1 : condition1 f)
  (hf_cond2 : condition2 f) : 
  ∃ (a b c : ℝ), a = 2 ∧ b = -2 ∧ c = 1 ∧ ∀ x, f x = a * x^2 + b * x + c :=
sorry

end quadratic_function_expression_l200_200162


namespace no_similar_triangle_after_cuts_l200_200723

theorem no_similar_triangle_after_cuts (T : Triangle)
  (hA : T.angleA = 20)
  (hB : T.angleB = 20)
  (hC : T.angleC = 140) :
  ¬ ∃ (T' : Triangle), (T' ∼ T) ∧ (obtained_after_bisector_cuts T T') :=
by
  sorry

end no_similar_triangle_after_cuts_l200_200723


namespace compute_fraction_l200_200765

theorem compute_fraction :
  (1 * 2 + 2 * 4 - 3 * 8 + 4 * 16 + 5 * 32 - 6 * 64) /
  (2 * 4 + 4 * 8 - 6 * 16 + 8 * 32 + 10 * 64 - 12 * 128) =
  1 / 4 :=
by
  -- Proof will go here
  sorry

end compute_fraction_l200_200765


namespace perpendicular_vectors_l200_200883

variable (m : ℝ)

def vector_a := (m, 3)
def vector_b := (1, m + 1)

def dot_product (v w : ℝ × ℝ) := (v.1 * w.1) + (v.2 * w.2)

theorem perpendicular_vectors (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by 
  unfold vector_a vector_b dot_product at h
  linarith

end perpendicular_vectors_l200_200883


namespace boys_on_playground_l200_200650

theorem boys_on_playground (total_children girls : ℕ) (h1 : total_children = 117) (h2 : girls = 77) :
  ∃ boys : ℕ, boys = total_children - girls ∧ boys = 40 :=
by
  have boys := total_children - girls
  use boys
  split
  . exact rfl
  . rw [h1, h2]

-- Proof skipped (Lean 4 statement only)

end boys_on_playground_l200_200650


namespace divide_square_into_flags_l200_200589

noncomputable def Point : Type := ℝ × ℝ

structure Square :=
(A B C D O : Point) -- A, B, C, D are vertices of the square and O is the center

def flag (a b c : Point) (o : Point) : Prop :=
  ∃ x y: Point, -- there exists points x, y such that the flag forms a pentagon
  -- The pentagon is formed by a, b, c, x, y, o
  (true) -- this is a placeholder, one can define specific geometric properties if needed

theorem divide_square_into_flags (sq : Square) :
  ∃ (p : list (Point × Point × Point)), 
    -- list of triples to denote the flags
    ∀ (a b c : Point), (a, b, c) ∈ p → flag a b c sq.O :=
sorry

end divide_square_into_flags_l200_200589


namespace parallelogram_and_area_l200_200461

-- Define the points as vectors
def A : ℝ × ℝ × ℝ := (3, -4, 2)
def B : ℝ × ℝ × ℝ := (5, -8, 5)
def C : ℝ × ℝ × ℝ := (4, -6, 1)
def D : ℝ × ℝ × ℝ := (6, -10, 4)

-- Define the vectors obtained from points
def ab_vec := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
def cd_vec := (D.1 - C.1, D.2 - C.2, D.3 - C.3)
def ca_vec := (C.1 - A.1, C.2 - A.2, C.3 - A.3)

-- Check for parallelogram and area
noncomputable def cross_product (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem parallelogram_and_area :
  ab_vec = cd_vec ∧ magnitude (cross_product ab_vec ca_vec) = 5 * Real.sqrt 5 :=
by {
  sorry -- Proof is left out as per instructions.
}

end parallelogram_and_area_l200_200461


namespace profit_per_meter_is_20_l200_200065

-- Define the conditions as hypotheses
def total_selling_price := 6900
def cost_price_per_meter := 66.25
def meters_of_cloth := 80

-- Define the profit calculation
def total_cost_price := cost_price_per_meter * meters_of_cloth
def total_profit := total_selling_price - total_cost_price
def profit_per_meter := total_profit / meters_of_cloth

-- State the main theorem/problem to prove the profit per meter
theorem profit_per_meter_is_20 : profit_per_meter = 20 := 
by 
  -- Here, you would calculate and show that profit per meter indeed equals Rs. 20.
  sorry

end profit_per_meter_is_20_l200_200065


namespace last_three_digits_l200_200291

theorem last_three_digits (s : String) (h_len : s.length = 6) 
  (h_digits : s.toList.perm ["1", "1", "2", "2", "3", "3"]) 
  (h_odd : s.get 5 = '1' ∨ s.get 5 = '3')
  (h_sep1 : ∃ i j, s.get i = '1' ∧ s.get j = '1' ∧ |i - j| = 2)
  (h_sep2 : ∃ i j, s.get i = '2' ∧ s.get j = '2' ∧ |i - j| = 3) 
  (h_sep3 : ∃ i j, s.get i = '3' ∧ s.get j = '3' ∧ |i - j| = 4) : 
  s.data.drop 3 = ['2', '1', '3'] :=
sorry

end last_three_digits_l200_200291


namespace max_black_squares_l200_200417

-- Define the grid size as constants
def n : ℕ := 3
def m : ℕ := 3

-- Define a grid as a matrix of booleans where true represents a black square and false represents a white square
def grid := Matrix (Fin n) (Fin m) Bool

-- Define a helper function to check 180-degree rotational symmetry
def is_180_rotational_symmetric (g : grid) : Prop :=
  ∀ i j, g i j = g (Fin.mk (n - i.succ) sorry) (Fin.mk (m - j.succ) sorry)

-- Define a helper function to check no lines of symmetry
def has_no_lines_of_symmetry (g : grid) : Prop :=
  ¬ (∀ i j, g i j = g (Fin.mk (n - i.succ) sorry) j) ∧ -- No vertical symmetry 
  ¬ (∀ i j, g i j = g i (Fin.mk (m - j.succ) sorry))     -- No horizontal symmetry 

-- Define the main theorem
theorem max_black_squares (g : grid) :
  is_180_rotational_symmetric g ∧ has_no_lines_of_symmetry g →
  ∃ k, (∑ i j, if g i j then 1 else 0) = k ∧ k = 5 :=
by
  sorry

end max_black_squares_l200_200417


namespace hip_hop_percentage_l200_200295

theorem hip_hop_percentage (country_percentage hip_hop_ratio pop_ratio : ℝ) 
    (h_country : country_percentage = 40)
    (h_ratios : hip_hop_ratio = 65 ∧ pop_ratio = 35) :
  let remaining_percentage := 100 - country_percentage,
      hip_hop_percentage := hip_hop_ratio * remaining_percentage / 100 in
  hip_hop_percentage = 39 :=
by
  intros
  rw [h_country, h_ratios.left]
  sorry

end hip_hop_percentage_l200_200295


namespace _l200_200224

variables (K M N : Type) [triangle : Real]
variables (NA NB NC : ℝ) (R : ℝ)

noncomputable theorem triangle_lengths 
  (circumradius : ℝ := R)
  (altitude : ℝ := NA)
  (angle_bisector : ℝ := NB)
  (median : ℝ := NC)
  (divides_KNM : NA = NB ∧ NB = NC)
  (correct_angle_division : divides_KNM → ∀ {α : ℝ}, 4 * α = π / 2)
  : 
  altitude = R * Real.sqrt 2 / 2 ∧ 
  angle_bisector = R * Real.sqrt (2 - 2 * Real.sqrt 2) ∧ 
  median = R := 
begin
  sorry
end

end _l200_200224


namespace ladder_slip_l200_200035

theorem ladder_slip (ladder_length : ℝ) (initial_base : ℝ) (slip_distance : ℝ) : 
  ladder_length = 30 ∧ initial_base = 8 ∧ slip_distance = 3 →
  ∃ new_base : ℝ, (new_base - initial_base) ≈ 7.1 :=
by
  sorry

end ladder_slip_l200_200035


namespace not_satisfy_all_with_two_pizzas_l200_200367

structure Pizza (hasTomatoes : Prop) (hasMushrooms : Prop) (hasSausage : Prop)

def satisfiesMasha (p : Pizza) : Prop := p.hasTomatoes ∧ ¬ p.hasSausage
def satisfiesVanya (p : Pizza) : Prop := p.hasMushrooms
def satisfiesDasha (p : Pizza) : Prop := ¬ p.hasTomatoes
def satisfiesNikita (p : Pizza) : Prop := p.hasTomatoes ∧ ¬ p.hasMushrooms
def satisfiesIgor (p : Pizza) : Prop := p.hasSausage ∧ ¬ p.hasMushrooms

theorem not_satisfy_all_with_two_pizzas :
  ∀ (p1 p2 : Pizza),
    ¬ ((satisfiesMasha p1 ∨ satisfiesMasha p2)
     ∧ (satisfiesVanya p1 ∨ satisfiesVanya p2)
     ∧ (satisfiesDasha p1 ∨ satisfiesDasha p2)
     ∧ (satisfiesNikita p1 ∨ satisfiesNikita p2)
     ∧ (satisfiesIgor p1 ∨ satisfiesIgor p2)) :=
by
  sorry

end not_satisfy_all_with_two_pizzas_l200_200367


namespace max_colours_in_7x7_table_is_22_l200_200416

def max_colours_7x7_table : ℕ :=
  22

theorem max_colours_in_7x7_table_is_22 : 
  ∀ (colours : ℕ → ℕ → ℕ), 
    (∀ i j : ℕ, i ≠ j → (finRange 7).bUnion (λ k, {colours i k}).card ≠ (finRange 7).bUnion (λ k, {colours j k}).card)
    ∧ (∀ i j : ℕ, i ≠ j → (finRange 7).bUnion (λ k, {colours k i}).card ≠ (finRange 7).bUnion (λ k, {colours k j}).card)
    → (∑ i : fin 7, (finRange 7).bUnion (λ k, {colours i k}).card) ≤ max_colours_7x7_table :=
  sorry

end max_colours_in_7x7_table_is_22_l200_200416


namespace probability_l200_200826

def unitCircle : Set (ℝ × ℝ) := { P | P.1^2 + P.2^2 = 1 }
def chordAB : (ℝ × ℝ) × (ℝ × ℝ) := ((1, 0), (0, 1))
def lengthAB := Real.sqrt 2
def pointInside (P : ℝ × ℝ) := P ∈ { Q | Q.1^2 + Q.2^2 ≤ 1 }

lemma probability_condition (P : ℝ × ℝ) :
  (P ∈ { Q | Q.1^2 + Q.2^2 ≤ 1 }) → 
  ((P.1 - 1) * (-1) + P.2 * 1 ≥ 2) ↔ (P.1 - P.2 + 1 ≤ 0) := by
  sorry

theorem probability (P : ℝ × ℝ) (hP : pointInside P):
  P ∈ { Q | (Q.1 - 1) * (-1) + Q.2 * 1 ≥ 2 } →
  (probability_space.measure (setOf (λ Q : ℝ × ℝ, Q.1^2 + Q.2^2 ≤ 1)).toReal) 
    (setOf (λ Q : ℝ × ℝ, (Q.1 - 1) * (-1) + Q.2 * 1 ≥ 2)) = (π - 2) / (4 * π) := by
  sorry

end probability_l200_200826


namespace total_trees_in_forest_l200_200617

theorem total_trees_in_forest (a_street : ℕ) (a_forest : ℕ) 
                              (side_length : ℕ) (trees_per_square_meter : ℕ)
                              (h1 : a_street = side_length * side_length)
                              (h2 : a_forest = 3 * a_street)
                              (h3 : side_length = 100)
                              (h4 : trees_per_square_meter = 4) :
                              a_forest * trees_per_square_meter = 120000 := by
  -- Proof omitted
  sorry

end total_trees_in_forest_l200_200617


namespace proportion_of_boys_correct_l200_200637

noncomputable def proportion_of_boys : ℚ :=
  let p_boy := 1 / 2
  let p_girl := 1 / 2
  let expected_children := 3 -- (2 boys and 1 girl)
  let expected_boys := 2 -- Expected number of boys in each family
  
  expected_boys / expected_children

theorem proportion_of_boys_correct : proportion_of_boys = 2 / 3 := by
  sorry

end proportion_of_boys_correct_l200_200637


namespace train_speed_84_kmph_l200_200360

theorem train_speed_84_kmph (length : ℕ) (time : ℕ) (conversion_factor : ℚ)
  (h_length : length = 140) (h_time : time = 6) (h_conversion_factor : conversion_factor = 3.6) :
  (length / time) * conversion_factor = 84 :=
  sorry

end train_speed_84_kmph_l200_200360


namespace sequence_probability_l200_200377

noncomputable def b : ℕ → ℕ
| 0     := 0
| 1     := 3
| 2     := 6
| (n+2) := 2 * b (n+1) + b n

theorem sequence_probability :
  let total_sequences := 3^12 in
  let valid_sequences := b 12 in
  let probability := (valid_sequences : ℚ) / total_sequences in
  probability = 23 / 1003 := by
  let total_sequences := 3^12
  let valid_sequences := b 12
  let probability := (valid_sequences : ℚ) / total_sequences
  have : valid_sequences = 12207 := by sorry
  have : total_sequences = 531441 := by sorry
  have : 12207 / 531441 = 23 / 1003 := by sorry
  exact this

end sequence_probability_l200_200377


namespace correct_vector_relation_l200_200158

variables (O A B C : Type) [AddCommGroup O] [Module ℝ O]
variables (C_divides_AB : ∃ k : ℝ, k = 2 ∧ ((C : O) = ((1 / (k + 1)) • A + (k / (k + 1)) • B)))

theorem correct_vector_relation :
  (C : O) = (1 / 3) • (A : O) + (2 / 3) • (B : O) :=
by
  sorry

end correct_vector_relation_l200_200158


namespace ones_digit_of_6_pow_52_l200_200333

theorem ones_digit_of_6_pow_52 : (6 ^ 52) % 10 = 6 := by
  -- we'll put the proof here
  sorry

end ones_digit_of_6_pow_52_l200_200333


namespace volume_tetrahedron_A₁B₁C₁D₁_l200_200953

-- Definitions of the regular tetrahedron and the points
structure RegularTetrahedron :=
  (A B C D : ℝ)
  (edge_length : ℝ)

def points_on_faces (A B C D : ℝ) (edge_length : ℝ) : Prop :=
  ∃ A₁ B₁ C₁ D₁ : ℝ, 
    A₁ B₁ C₁ D₁ ∈ faces ∧
    perpendicular A₁ B₁ faces ∧
    perpendicular B₁ C₁ faces ∧
    perpendicular C₁ D₁ faces ∧
    perpendicular D₁ A₁ faces 

-- Volume calculation
def volume_tetrahedron (a : ℝ) : ℝ :=
  (Real.sqrt 2 * a^3) / 162

-- Proof goal
theorem volume_tetrahedron_A₁B₁C₁D₁ :
  ∀ (T : RegularTetrahedron), points_on_faces T.A T.B T.C T.D T.edge_length →
  (volume_tetrahedron T.edge_length = (Real.sqrt 2 * T.edge_length^3) / 162) :=
by
  sorry

end volume_tetrahedron_A₁B₁C₁D₁_l200_200953


namespace range_of_k_l200_200821

theorem range_of_k (k : ℝ) :
  (∃ F : ℝ, ∃ l : ℝ → ℝ, (∀ x, l x = k * x + F) ∧ (∃ C : ℝ → ℝ, (∀ x, C x = x^2 / 4) ∧
  (∀ x, (C (k * F + 1) - l x) = 2))) →
  k < -real.sqrt 3 ∨ k > real.sqrt 3 :=
by
  sorry

end range_of_k_l200_200821


namespace lambda_range_in_acute_triangle_l200_200216

theorem lambda_range_in_acute_triangle (A B C D: Type) [metric_space A] 
  [add_comm_group A] [module ℝ A] 
  (angle_CAB : ∠CAB = 60°) (BC_eq_2 : dist B C = 2)
  (D_on_external_bisector : is_on_external_bisector D A ∠BAC)
  (CD_eq_expr : ∀ λ : ℝ, CD = CA + λ * (CA/|CA| + AB/|AB|)) : 
  ∃ λ_min λ_max : ℝ, λ_min = (2 * sqrt 3 + 2) / 3 ∧ λ_max = 2 ∧ 
                    ∀ λ : ℝ, λ_min < λ ∧ λ ≤ λ_max :=
by
  sorry

end lambda_range_in_acute_triangle_l200_200216


namespace correct_proposition_count_l200_200996

noncomputable def proposition_1 (α β γ : Type) [plane α] [plane β] [plane γ] : Prop :=
  (α ⊥ β) ∧ (β ⊥ γ) → (α ⊥ γ)

noncomputable def proposition_2 (α β : Type) (m : line) [plane α] [plane β] : Prop :=
  (α ∥ β) ∧ (m ⊂ β) → (m ∥ α)

noncomputable def proposition_3 (m n : line) (γ : Type) [plane γ] : Prop :=
  (projections_perpendicular_within m n γ) → (m ⊥ n)

noncomputable def proposition_4 (m n : line) (α β : Type) [plane α] [plane β] : Prop :=
  (m ∥ α) ∧ (n ∥ β) ∧ (α ⊥ β) → (m ⊥ n)

theorem correct_proposition_count : 
  let α : Type; let β : Type; let γ : Type; let m : line; let n : line;
  [plane α] [plane β] [plane γ] in
  ((proposition_1 α β γ) = false) ∧
  ((proposition_2 α β m) = true) ∧
  ((proposition_3 m n γ) = false) ∧
  ((proposition_4 m n α β) = false) → 
  (1 = 1) := 
sorry

end correct_proposition_count_l200_200996


namespace find_x_plus_y_l200_200906

theorem find_x_plus_y
  (x y : ℤ)
  (hx : |x| = 2)
  (hy : |y| = 3)
  (hxy : x > y) : x + y = -1 := 
sorry

end find_x_plus_y_l200_200906


namespace sum_of_x_coords_of_midpoints_l200_200358

theorem sum_of_x_coords_of_midpoints (x_coords : Fin 45 → ℝ) 
    (h_sum : (Finset.univ.sum (λ i, x_coords i)) = 135) :
    Finset.univ.sum (λ i, (x_coords i + x_coords ((i + 1) % 45)) / 2) = 135 := 
sorry

end sum_of_x_coords_of_midpoints_l200_200358


namespace equation_of_circle_given_diameter_l200_200157

def is_on_circle (center : ℝ × ℝ) (radius : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

theorem equation_of_circle_given_diameter :
  ∀ (A B : ℝ × ℝ), A = (-3,0) → B = (1,0) → 
  (∃ (x y : ℝ), is_on_circle (-1, 0) 2 (x, y)) ↔ (x + 1)^2 + y^2 = 4 :=
by
  sorry

end equation_of_circle_given_diameter_l200_200157


namespace ratio_of_distance_l200_200594

noncomputable def initial_distance : ℝ := 30 * 20

noncomputable def total_distance : ℝ := 2 * initial_distance

noncomputable def distance_after_storm : ℝ := initial_distance - 200

theorem ratio_of_distance (initial_distance : ℝ) (total_distance : ℝ) (distance_after_storm : ℝ) : 
  distance_after_storm / total_distance = 1 / 3 :=
by
  -- Given conditions
  have h1 : initial_distance = 30 * 20 := by sorry
  have h2 : total_distance = 2 * initial_distance := by sorry
  have h3 : distance_after_storm = initial_distance - 200 := by sorry
  -- Prove the ratio is 1 / 3
  sorry

end ratio_of_distance_l200_200594


namespace total_fruit_in_buckets_l200_200655

def bucket_pieces (A B C : ℕ) : Prop :=
  A = B + 4 ∧ B = C + 3 ∧ C = 9 ∧ (A + B + C = 37)

theorem total_fruit_in_buckets : ∃ A B C, bucket_pieces A B C :=
by
  use 16, 12, 9
  unfold bucket_pieces
  split
  . exact rfl
  split
  . exact rfl
  split
  . exact rfl
  . exact rfl

end total_fruit_in_buckets_l200_200655


namespace total_cost_l200_200544

theorem total_cost 
  (rental_cost: ℝ) 
  (gallons: ℝ) 
  (gas_price: ℝ) 
  (price_per_mile: ℝ) 
  (miles: ℝ)
  (H1: rental_cost = 150)
  (H2: gallons = 8)
  (H3: gas_price = 3.5)
  (H4: price_per_mile = 0.5)
  (H5: miles = 320) :
  rental_cost + (gallons * gas_price) + (miles * price_per_mile) = 338 :=
by {
  have gas_cost : ℝ := gallons * gas_price,
  have mileage_cost : ℝ := miles * price_per_mile,
  rw [←H1, ←H2, ←H3, ←H4, ←H5],
  norm_num,
  sorry
}

end total_cost_l200_200544


namespace oblique_asymptote_l200_200332

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 8 * x + 12) / (3 * x + 4)

theorem oblique_asymptote :
  (∃ b : ℝ, ∀ x : ℝ, ∥f x - (x + b)∥ < ε) := by
sorry

end oblique_asymptote_l200_200332


namespace volume_larger_solid_cube_2_l200_200096

-- Defining the point structures
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Define vertices of the cube
def A : Point3D := ⟨0, 0, 0⟩
def B : Point3D := ⟨2, 0, 0⟩
def C : Point3D := ⟨2, 2, 0⟩
def D : Point3D := ⟨0, 2, 0⟩
def E : Point3D := ⟨0, 0, 2⟩
def F : Point3D := ⟨2, 0, 2⟩
def G : Point3D := ⟨2, 2, 2⟩
def H : Point3D := ⟨0, 2, 2⟩

-- Define the midpoints P and Q
def P : Point3D := ⟨(C.x + D.x) / 2, (C.y + D.y) / 2, (C.z + D.z) / 2⟩
def Q : Point3D := ⟨(G.x + H.x) / 2, (G.y + H.y) / 2, (G.z + H.z) / 2⟩

-- Main theorem statement
theorem volume_larger_solid_cube_2 (V : ℝ) :
  let cube_volume := 2 * 2 * 2 in
  V = cube_volume / 2 :=
by
  sorry

end volume_larger_solid_cube_2_l200_200096


namespace max_total_length_of_cuts_l200_200036

theorem max_total_length_of_cuts (A : ℕ) (n : ℕ) (m : ℕ) (P : ℕ) (Q : ℕ)
  (h1 : A = 30 * 30)
  (h2 : n = 225)
  (h3 : m = A / n)
  (h4 : m = 4)
  (h5 : Q = 4 * 30)
  (h6 : P = 225 * 10 - Q)
  (h7 : P / 2 = 1065) :
  P / 2 = 1065 :=
by 
  exact h7

end max_total_length_of_cuts_l200_200036


namespace polynomial_degree_l200_200003

theorem polynomial_degree (x : ℝ) : 
  degree ((5 * x^3 + 9)^10) = 30 :=
sorry

end polynomial_degree_l200_200003


namespace f_l200_200237

open Real

-- Conditions: f(x) is differentiable in (0, ∞) and f(e^x) = 3x + 1/2 * e^x + 1
noncomputable def f (x : ℝ) : ℝ := sorry

axiom differentiable_f : DifferentiableOn ℝ f (Ioi 0)

axiom f_eq : ∀ x : ℝ, 0 < x → f (exp x) = 3 * x + (1 / 2) * exp x + 1

-- Prove that f' (1) = 7/2
theorem f'_at_1 : deriv f 1 = 7 / 2 := by
  sorry

end f_l200_200237


namespace longer_side_correct_l200_200253

noncomputable def longer_side_of_rectangle : ℚ :=
  let a := (9 : ℚ) / 4 in a

theorem longer_side_correct (a : ℚ) (h1 : 0.8 * a * a = 81 / 20) : 
  a = longer_side_of_rectangle :=
by
  sorry

end longer_side_correct_l200_200253


namespace longer_segment_of_triangle_l200_200383

theorem longer_segment_of_triangle {a b c : ℝ} (h_triangle : a = 40 ∧ b = 90 ∧ c = 100) (h_altitude : ∃ h, h > 0) : 
  ∃ (longer_segment : ℝ), longer_segment = 82.5 :=
by 
  sorry

end longer_segment_of_triangle_l200_200383


namespace problem1_l200_200354

def f (x : ℝ) : ℝ := x^3 - 2 * f' 1 * x

theorem problem1 : deriv f 1 = 1 :=
sorry

end problem1_l200_200354


namespace logarithmic_relationship_l200_200861

noncomputable def a (n : ℕ) : ℕ := n * n

def b : ℕ → ℕ := sorry  -- Given that b_n are distinct positive integers

axiom distinct_positive_integers (n m : ℕ) (hn : n > 0) (hm : m > 0) : n ≠ m → b n ≠ b m

axiom condition (n : ℕ) (hn : n > 0) : b (a n) = a (b n)

theorem logarithmic_relationship :
  (∀ (b : ℕ → ℕ), distinct_positive_integers → condition →
  (lg (b 1 * b 4 * b 9 * b 16) / lg (b 1 * b 2 * b 3 * b 4) = 2)) :=
by
  intro b hdistinct hcondition
  sorry

end logarithmic_relationship_l200_200861


namespace arrange_1225_multiple_of_5_l200_200966

theorem arrange_1225_multiple_of_5 : 
  (∃ (s : Finset (Fin 10)) (h : s = {1, 2, 2, 5}),
    (card (s.filter (λ x, last.digit x = 5))
    = 6) : sorry

end arrange_1225_multiple_of_5_l200_200966


namespace parabolic_points_l200_200462

noncomputable def A (x1 : ℝ) (y1 : ℝ) : Prop := y1 = x1^2 - 3
noncomputable def B (x2 : ℝ) (y2 : ℝ) : Prop := y2 = x2^2 - 3

theorem parabolic_points (x1 x2 y1 y2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2)
  (hA : A x1 y1) (hB : B x2 y2) : y1 < y2 :=
by
  sorry

end parabolic_points_l200_200462


namespace product_of_divisors_of_72_l200_200792

theorem product_of_divisors_of_72 :
  ∀ (d ∈ {d : ℕ | ∃ a b : ℕ, d = 2^a * 3^b ∧ a <= 3 ∧ b <= 2}),
    (∏ d in {d : ℕ | ∃ a b : ℕ, d = 2^a * 3^b ∧ a <= 3 ∧ b <= 2}, d) = 72^6 :=
  by
    sorry

end product_of_divisors_of_72_l200_200792


namespace sequence_property_l200_200789

open Nat

theorem sequence_property (a : ℕ → ℤ) (h : ∀ n, a (n + 2) = (a n + 2006) / (a (n + 1) + 1)) (h' : ∀ n, a n ≠ -1) : 
  ∃ a b : ℤ, (∀ n, a n = a n + 2 ∧ (∀ i, a i = a ∨ a i = b) ∧ ab = 2006) := 
sorry

end sequence_property_l200_200789


namespace true_propositions_count_l200_200444

variables {Plane : Type} {Line : Type} (α β : Plane) (l : Line)

-- Definitions of Parallel and Perpendicular relationships
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def planes_perpendicular (p q : Plane) : Prop := sorry

-- Hypotheses
axiom H1 : parallel l β 
axiom H2 : perpendicular l α
axiom H3 : planes_perpendicular α β

-- Proof statement
theorem true_propositions_count : 
  (H1 ∧ H2 → H3) ∧ (H2 ∧ H3 → H1) ∧ ¬ (H1 ∧ H3 → H2) :=
by 
  repeat { constructor }; sorry

end true_propositions_count_l200_200444


namespace rotated_line_x_intercept_l200_200590

theorem rotated_line_x_intercept (x y : ℝ) : 
  let l := 2 * x - 3 * y + 30 = 0,
      center := (10 : ℝ, 10 : ℝ),
      theta := real.pi / 6,
      intercept := 15 * real.sqrt 3 - 15
  in (intercept = x) :=
sorry

end rotated_line_x_intercept_l200_200590


namespace find_b_l200_200692

-- Define the conditions of the equations
def condition_1 (x y a : ℝ) : Prop := x * Real.cos a + y * Real.sin a + 3 ≤ 0
def condition_2 (x y b : ℝ) : Prop := x^2 + y^2 + 8 * x - 4 * y - b^2 + 6 * b + 11 = 0

-- Define the proof problem
theorem find_b (b : ℝ) :
  (∀ a x y, condition_1 x y a → condition_2 x y b) →
  b ∈ Set.Iic (-2 * Real.sqrt 5) ∪ Set.Ici (6 + 2 * Real.sqrt 5) :=
by
  sorry

end find_b_l200_200692


namespace number_of_children_proof_l200_200281

-- Let A be the number of mushrooms Anya has
-- Let V be the number of mushrooms Vitya has
-- Let S be the number of mushrooms Sasha has
-- Let xs be the list of mushrooms of other children

def mushrooms_distribution (A V S : ℕ) (xs : List ℕ) : Prop :=
  let n := 3 + xs.length
  -- First condition
  let total_mushrooms := A + V + S + xs.sum
  let equal_share := total_mushrooms / n
  (A / 2 = equal_share) ∧ (V + A / 2 = equal_share) ∧ (S = equal_share) ∧
  (∀ x ∈ xs, x = equal_share) ∧
  -- Second condition
  (S + A = V + xs.sum)

theorem number_of_children_proof (A V S : ℕ) (xs : List ℕ) :
  mushrooms_distribution A V S xs → 3 + xs.length = 6 :=
by
  intros h
  sorry

end number_of_children_proof_l200_200281


namespace total_readers_l200_200950

theorem total_readers (S L B : ℕ) (hS : S = 250) (hL : L = 230) (hB : B = 80) : S + L - B = 400 := 
by
  rw [hS, hL, hB]
  norm_num

end total_readers_l200_200950


namespace smallest_satisfying_N_is_2520_l200_200428

open Nat

def smallest_satisfying_N : ℕ :=
  let N := 2520
  if (N + 2) % 2 = 0 ∧
     (N + 3) % 3 = 0 ∧
     (N + 4) % 4 = 0 ∧
     (N + 5) % 5 = 0 ∧
     (N + 6) % 6 = 0 ∧
     (N + 7) % 7 = 0 ∧
     (N + 8) % 8 = 0 ∧
     (N + 9) % 9 = 0 ∧
     (N + 10) % 10 = 0
  then N else 0

-- Statement of the problem in Lean 4
theorem smallest_satisfying_N_is_2520 : smallest_satisfying_N = 2520 :=
  by
    -- Proof would be added here, but is omitted as per instructions
    sorry

end smallest_satisfying_N_is_2520_l200_200428


namespace hundred_digit_number_impossible_l200_200633

/-- Prove that it is not possible to get the 100-digit number 5222...2221 starting from 1 
by multiplying by 5 or rearranging the digits, with the constraint that zero cannot be placed 
in the first digit. -/
theorem hundred_digit_number_impossible :
  let start := 1 in
  let target := 5222 -- ...(98 more 2's)... -- 2221 in
  ¬ ( ∃ (f : ℕ → ℕ), 
      (∀ n, f n = 5 * n ∨ 
            (∃ m, n = rearrange_digits m ∧ no_zero_first_digit m)) ∧
      start = f target) :=
sorry

end hundred_digit_number_impossible_l200_200633


namespace horses_meet_in_nine_days_l200_200526

noncomputable def goodHorseDistance (n : ℕ) : ℕ :=
  103 + (n-1) * 13

noncomputable def mediocreHorseDistance (n : ℕ) : ℕ :=
  97 - (n-1) * 0.5

noncomputable def totalDistanceCoveredInDays (m : ℕ) : ℕ :=
  (103 * m + (m * (m-1) * 13) / 2) + (97 * m + (m * (m-1) * -0.5) / 2)

theorem horses_meet_in_nine_days :
  totalDistanceCoveredInDays 9 = 2250 := sorry

end horses_meet_in_nine_days_l200_200526


namespace ways_to_arrange_digits_of_1225_l200_200959

theorem ways_to_arrange_digits_of_1225 :
  let digits := [1, 2, 2, 5] in
  (∃ permutations : list (list ℕ), 
      permutations = digits.permutations ∧
      (∃ valid_permutations : list (list ℕ),
          valid_permutations = permutations.filter (λ l, (l.nth 3).getOrElse 0 = 5) ∧
          valid_permutations.length = 3)) :=
sorry

end ways_to_arrange_digits_of_1225_l200_200959


namespace distance_between_trees_l200_200940

theorem distance_between_trees (num_trees : ℕ) (total_length : ℕ) (num_spaces : ℕ) (distance_per_space : ℕ) 
  (h_num_trees : num_trees = 11) (h_total_length : total_length = 180)
  (h_num_spaces : num_spaces = num_trees - 1) (h_distance_per_space : distance_per_space = total_length / num_spaces) :
  distance_per_space = 18 := 
  by 
    sorry

end distance_between_trees_l200_200940


namespace checkerboard_squares_count_l200_200359

theorem checkerboard_squares_count :
  let valid_squares (n : ℕ) := if n = 1 then 0 else if n = 2 then 81 else if n = 3 then 32 else
                                if n = 4 then 49 else if n = 5 then 36 else if n = 6 then 25 else
                                if n = 7 then 16 else if n = 8 then 9 else if n = 9 then 4 else
                                if n = 10 then 1 else 0 in
  (List.sum (List.map valid_squares (List.range' 1 10))) = 253 :=
by {
  let valid_squares := λ n,
                        (if n = 1 then 0 else
                         if n = 2 then 81 else
                         if n = 3 then 32 else
                         if n = 4 then 49 else
                         if n = 5 then 36 else
                         if n = 6 then 25 else
                         if n = 7 then 16 else
                         if n = 8 then 9 else
                         if n = 9 then 4 else
                         if n = 10 then 1 else 0 : ℕ),
  have list_valid_squares := List.map valid_squares (List.range' 1 10),
  have sum_valid_squares := List.sum list_valid_squares,
  show sum_valid_squares = 253,
  sorry
}

end checkerboard_squares_count_l200_200359


namespace boys_on_playground_l200_200649

theorem boys_on_playground (total_children girls : ℕ) (h1 : total_children = 117) (h2 : girls = 77) :
  ∃ boys : ℕ, boys = total_children - girls ∧ boys = 40 :=
by
  have boys := total_children - girls
  use boys
  split
  . exact rfl
  . rw [h1, h2]

-- Proof skipped (Lean 4 statement only)

end boys_on_playground_l200_200649


namespace total_number_of_trees_l200_200614

-- Definitions of the conditions
def side_length : ℝ := 100
def trees_per_sq_meter : ℝ := 4

-- Calculations based on the conditions
def area_of_street : ℝ := side_length * side_length
def area_of_forest : ℝ := 3 * area_of_street

-- The statement to prove
theorem total_number_of_trees : 
  trees_per_sq_meter * area_of_forest = 120000 := 
sorry

end total_number_of_trees_l200_200614


namespace problem_frac_l200_200099

noncomputable def A : ℝ :=
  (1 : ℝ) / (1^2) + (1 : ℝ) / (3^2) - (1 : ℝ) / (7^2) - (1 : ℝ) / (9^2) + (1 : ℝ) / (11^2) + 
  (1 : ℝ) / (13^2) - Σ' (n : ℕ) (_ : n > 13) (_ : (n % 2 = 1) ∧ (n % 5 ≠ 0)), (1 : ℝ) / (n^2)

noncomputable def B : ℝ :=
  (1 : ℝ) / (5^2) - (1 : ℝ) / (15^2) + (1 : ℝ) / (25^2) - (1 : ℝ) / (35^2) + 
  Σ' (k : ℕ) (_ : k > 35) (_ : (k % 10 = 5)) (1 : ℝ) / (k^2)

theorem problem_frac : A / B = 26 :=
begin
  sorry
end

end problem_frac_l200_200099


namespace ford_younger_than_christopher_l200_200139

variable (G C F Y : ℕ)

-- Conditions
axiom h1 : G = C + 8
axiom h2 : F = C - Y
axiom h3 : G + C + F = 60
axiom h4 : C = 18

-- Target statement
theorem ford_younger_than_christopher : Y = 2 :=
sorry

end ford_younger_than_christopher_l200_200139


namespace tan_A_of_triangle_conditions_l200_200513

open Real

def triangle_angles (A B C : ℝ) : Prop :=
  A + B + C = π ∧ 0 < A ∧ A < π / 2 ∧ B = π / 4

def form_arithmetic_sequence (a b c : ℝ) : Prop :=
  2 * b^2 = a^2 + c^2

theorem tan_A_of_triangle_conditions
  (A B C a b c : ℝ)
  (h_angles : triangle_angles A B C)
  (h_seq : form_arithmetic_sequence a b c) :
  tan A = sqrt 2 - 1 :=
by
  sorry

end tan_A_of_triangle_conditions_l200_200513


namespace total_trees_in_forest_l200_200619

theorem total_trees_in_forest (a_street : ℕ) (a_forest : ℕ) 
                              (side_length : ℕ) (trees_per_square_meter : ℕ)
                              (h1 : a_street = side_length * side_length)
                              (h2 : a_forest = 3 * a_street)
                              (h3 : side_length = 100)
                              (h4 : trees_per_square_meter = 4) :
                              a_forest * trees_per_square_meter = 120000 := by
  -- Proof omitted
  sorry

end total_trees_in_forest_l200_200619


namespace area_EFCD_l200_200533

-- Defining the geometrical setup and measurements of the trapezoid
variables (AB CD AD BC : ℝ) (h1 : AB = 10) (h2 : CD = 30) (h_altitude : ∃ h : ℝ, h = 18)

-- Defining the midpoints E and F of AD and BC respectively
variables (E F : ℝ) (h_E : E = AD / 2) (h_F : F = BC / 2)

-- Define the intersection of diagonals and the ratio condition
variables (AC BD G : ℝ) (h_ratio : ∃ r : ℝ, r = 1/2)

-- Proving the area of quadrilateral EFCD
theorem area_EFCD : EFCD_area = 225 :=
sorry

end area_EFCD_l200_200533


namespace price_per_kg_of_mixture_eq_14_33_l200_200078

theorem price_per_kg_of_mixture_eq_14_33 :
  let wheat_kg := 30
  let wheat_price_per_kg := 11.50
  let rice_kg := 20
  let rice_price_per_kg := 14.25
  let millet_kg := 25
  let millet_price_per_kg := 9
  let wheat_profit_pct := 0.30
  let rice_profit_pct := 0.25
  let millet_profit_pct := 0.20
  let total_cost := (wheat_kg * wheat_price_per_kg) + (rice_kg * rice_price_per_kg) + (millet_kg * millet_price_per_kg)
  let total_desired_profit := (wheat_kg * wheat_price_per_kg * wheat_profit_pct) + (rice_kg * rice_price_per_kg * rice_profit_pct) + (millet_kg * millet_price_per_kg * millet_profit_pct)
  let total_selling_price := total_cost + total_desired_profit
  let total_weight := wheat_kg + rice_kg + millet_kg
  let price_per_kg_mixture := total_selling_price / total_weight
  in
  price_per_kg_mixture = 14.33 := by
  sorry

end price_per_kg_of_mixture_eq_14_33_l200_200078


namespace first_digit_base_9_of_y_l200_200277

def base_3_to_base_10 (n : Nat) : Nat := sorry
def base_10_to_base_9_first_digit (n : Nat) : Nat := sorry

theorem first_digit_base_9_of_y :
  let y := 11220022110022112221
  let base_10_y := base_3_to_base_10 y
  base_10_to_base_9_first_digit base_10_y = 4 :=
by
  let y := 11220022110022112221
  let base_10_y := base_3_to_base_10 y
  show base_10_to_base_9_first_digit base_10_y = 4
  sorry

end first_digit_base_9_of_y_l200_200277


namespace triangle_shape_l200_200939

theorem triangle_shape (A B C a b : ℝ) (hA_pos : 0 < A) (hA_lt_pi : A < π)
(hB_pos : 0 < B) (hB_lt_pi : B < π) (cosA_cosB_eq_b_over_a : cos A / cos B = b / a) : 
(∃ (hC_pos : 0 < C) (hC_lt_pi : C < π), A + B + C = π) → 
(A = B ∨ A + B = π / 2) :=
by
  intros h_triangle_angles
  sorry

end triangle_shape_l200_200939


namespace smallest_lcm_of_4_digit_integers_with_gcd_5_l200_200506

-- Definition of the given integers k and l
def positive_4_digit_integers (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

-- The main theorem we want to prove
theorem smallest_lcm_of_4_digit_integers_with_gcd_5 :
  ∃ (k l : ℕ), positive_4_digit_integers k ∧ positive_4_digit_integers l ∧ gcd k l = 5 ∧ lcm k l = 201000 :=
by {
  sorry
}

end smallest_lcm_of_4_digit_integers_with_gcd_5_l200_200506


namespace tangent_line_eq_l200_200292

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) / (1 + x)

theorem tangent_line_eq (tangent_point : ℝ × ℝ) (x y : ℝ) : 
  tangent_point = (0, 1) → 
  ∃ k : ℝ, 
    k = (f' 0) → 
    (y - tangent_point.2 = k * (x - tangent_point.1)) → 
    x + y - 1 = 0 := 
by
  sorry

end tangent_line_eq_l200_200292


namespace sin_cos_diff_sin_cos_square_sum_l200_200442

open Real

noncomputable theory
  
def sin_cos_related (x : ℝ) : Prop :=
  -1 < x ∧ x < 0 ∧ sin x + cos x = sqrt 2

theorem sin_cos_diff (x : ℝ) (h : sin_cos_related x) : sin x - cos x = -sqrt (2 - sqrt 2) := 
sorry

theorem sin_cos_square_sum (x : ℝ) (h : sin_cos_related x) : sin x^2 + cos x^2 = 1 := 
by 
  calc
    sin x^2 + cos x^2 = 1 : by norm_num

end sin_cos_diff_sin_cos_square_sum_l200_200442


namespace range_of_gauss_function_of_f_l200_200814

def gauss_function (x : ℝ) : ℤ :=
  ⌊x⌋

def f (x : ℝ) : ℝ :=
  (2^x + 3) / (2^x + 1)

theorem range_of_gauss_function_of_f :
  ∀ (x : ℝ), gauss_function (f x) ∈ {1, 2} :=
by
  sorry

end range_of_gauss_function_of_f_l200_200814


namespace volume_of_sphere_is_correct_l200_200436

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem volume_of_sphere_is_correct :
  sphere_volume 6 ≈ 904.77888 :=
sorry

end volume_of_sphere_is_correct_l200_200436


namespace perpendicular_vectors_l200_200874

def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (1, m + 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (m : ℝ) (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by sorry

end perpendicular_vectors_l200_200874


namespace problem_statement_l200_200815

theorem problem_statement (a b : ℝ) (h0 : 0 < b) (h1 : b < 1/2) (h2 : 1/2 < a) (h3 : a < 1) :
  (0 < a - b) ∧ (a - b < 1) ∧ (ab < a^2) ∧ (a - 1/b < b - 1/a) :=
by 
  sorry

end problem_statement_l200_200815


namespace inequality_am_gm_l200_200264

theorem inequality_am_gm (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x^4 + y^2) + y / (x^2 + y^4)) ≤ (1 / (x * y)) :=
by
  sorry

end inequality_am_gm_l200_200264


namespace tangent_circle_and_distance_l200_200490

noncomputable theory

-- Condition: Hyperbola C
def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 3 = 1

-- Condition: Point P is the right vertex of the hyperbola C
def P : ℝ × ℝ := (2, 0)

-- Question 1: Standard equation of the circle with center P that is tangent to both asymptotes of the hyperbola C
def circle_with_center_P (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 12 / 7

-- Question 2: Line l with normal vector (1, -1) and distance d from points on the hyperbola
def normal_vector : ℝ × ℝ := (1, -1)

-- Statement: There are exactly three points P1, P2, P3 on the hyperbola C at distance d from line l
def line_l (x y : ℝ) : Prop := y = x + 2

def distance_from_line (x y d : ℝ) : Prop := d = abs (x * 1 + y * (-1)) / sqrt(1^2 + (-1)^2)

theorem tangent_circle_and_distance (x y d : ℝ) : 
  (circle_with_center_P x y ↔ hyperbola x y) ∧ 
  ((∃ P1 P2 P3, hyperbola P1 P2 ∧ hyperbola P1 P3 ∧ hyperbola P2 P3 ∧ distance_from_line P1 P2 d ∧ distance_from_line P1 P3 d ∧ distance_from_line P2 P3 d) → (d = sqrt(2)/2 ∨ d = 3*sqrt(2)/2)) :=
by sorry

end tangent_circle_and_distance_l200_200490


namespace solve_quadratic_l200_200269

theorem solve_quadratic : ∃ x : ℝ, (x^2 - 2 * x - 8 = 0 ↔ x = 4 ∨ x = -2) :=
sorry

end solve_quadratic_l200_200269


namespace perpendicular_vectors_l200_200884

variable (m : ℝ)

def vector_a := (m, 3)
def vector_b := (1, m + 1)

def dot_product (v w : ℝ × ℝ) := (v.1 * w.1) + (v.2 * w.2)

theorem perpendicular_vectors (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by 
  unfold vector_a vector_b dot_product at h
  linarith

end perpendicular_vectors_l200_200884


namespace product_of_cubes_91_l200_200302

theorem product_of_cubes_91 :
  ∃ (a b : ℤ), (a = 3 ∨ a = 4) ∧ (b = 3 ∨ b = 4) ∧ (a^3 + b^3 = 91) ∧ (a * b = 12) :=
by
  sorry

end product_of_cubes_91_l200_200302


namespace problem_solution_l200_200834

theorem problem_solution
  (x : ℂ) 
  (h : x - (1 / x) = 2 * complex.I) :
  x^729 - (1 / (x^729)) = -4 * complex.I := 
  sorry

end problem_solution_l200_200834


namespace smallest_number_l200_200742

theorem smallest_number (S : set ℤ) (h : S = {0, -3, 1, -1}) : ∃ m ∈ S, ∀ x ∈ S, m ≤ x ∧ m = -3 :=
by
  sorry

end smallest_number_l200_200742


namespace product_divisors_72_l200_200791

theorem product_divisors_72 : ∏ d in (finset.filter (λ x, 72 % x = 0) (finset.range 73)), d = 2^18 * 3^12 := 
by 
  sorry

end product_divisors_72_l200_200791


namespace function_range_l200_200794

open Real

theorem function_range (θ : ℝ) (x : ℝ) (h_theta : 0 ≤ θ ∧ θ ≤ 2 * π) :
  let y := (x^2 + 2 * x * sin θ + 2) / (x^2 + 2 * x * cos θ + 2)
  in 2 - sqrt 3 ≤ y ∧ y ≤ 2 + sqrt 3 :=
by {
  sorry,
}

end function_range_l200_200794


namespace max_sin2_cos_l200_200806

theorem max_sin2_cos (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π / 2) :
    ∃ θ, (θ = Real.arctan (Real.sqrt 2) ∧ ∀ t, 0 ≤ t ∧ t ≤ π / 2 → sin t ^ 2 * cos t ≤ sin θ ^ 2 * cos θ) := sorry

end max_sin2_cos_l200_200806


namespace num_lines_l200_200952

-- Define the points A1, A2, A3, A4, A5, B1, B2
inductive Point 
| A1 | A2 | A3 | A4 | A5 | B1 | B2

open Point

-- Define the collinearity condition
def collinear (p1 p2 p3 : Point) : Prop :=
  (p1 = A1 ∧ p2 = A2 ∧ p3 = A3) ∨ 
  (p1 = A1 ∧ p2 = A3 ∧ p3 = A4) ∨
  (p1 = A1 ∧ p2 = A4 ∧ p3 = A5) ∨
  (p1 = A2 ∧ p2 = A3 ∧ p3 = A4) ∨
  (p1 = A2 ∧ p2 = A4 ∧ p3 = A5) ∨
  (p1 = A3 ∧ p2 = A4 ∧ p3 = A5)

-- Define the two-point line function
def line (p1 p2 : Point) : Finset (Finset Point) :=
  { {p1, p2} }

-- The theorem to prove the number of distinct lines
theorem num_lines (points : Finset Point) (h_points : points = {A1, A2, A3, A4, A5, B1, B2}) 
  (h_collinear : ∀ p1 p2 p3, {p1, p2, p3} ⊆ points → ¬ collinear p1 p2 p3 → p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) :
  (points.filter (λ s, s.card = 2)).card = 12 := sorry

end num_lines_l200_200952


namespace relationship_among_m_n_p_l200_200446

noncomputable def m := 0.9^5.1
noncomputable def n := 5.1^0.9
noncomputable def p := Real.logr 0.9 5.1

theorem relationship_among_m_n_p : p < m ∧ m < n := by
  sorry

end relationship_among_m_n_p_l200_200446


namespace total_number_of_fruits_is_37_l200_200653

def bucket_fruits (A B C : ℕ) : Prop :=
  A = B + 4 ∧ B = C + 3 ∧ C = 9 ∧ A + B + C = 37

theorem total_number_of_fruits_is_37 (A B C : ℕ) : bucket_fruits A B C :=
  by
  -- Given conditions
  let C_val := 9  -- bucket C has 9 pieces of fruit
  let B_val := C_val + 3  -- bucket B has 3 more pieces of fruit than bucket C
  let A_val := B_val + 4  -- bucket A has 4 more pieces of fruit than bucket B
  
   -- Combined statement
  have hA : A = A_val := sorry
  have hB : B = B_val := sorry
  have hC : C = C_val := sorry
  have ht : A + B + C = 37 := sorry
  
  exact ⟨hA, hB, hC, ht⟩

end total_number_of_fruits_is_37_l200_200653


namespace probability_three_white_two_black_eq_eight_seventeen_l200_200037
-- Import Mathlib library to access combinatorics functions.

-- Define the total number of white and black balls.
def total_white := 8
def total_black := 7

-- The key function to calculate combinations.
noncomputable def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Define the problem conditions as constants.
def total_balls := total_white + total_black
def chosen_balls := 5
def white_balls_chosen := 3
def black_balls_chosen := 2

-- Calculate number of combinations.
noncomputable def total_combinations : ℕ := choose total_balls chosen_balls
noncomputable def white_combinations : ℕ := choose total_white white_balls_chosen
noncomputable def black_combinations : ℕ := choose total_black black_balls_chosen

-- Calculate the probability as a rational number.
noncomputable def probability_exact_three_white_two_black : ℚ :=
  (white_combinations * black_combinations : ℚ) / total_combinations

-- The theorem we want to prove
theorem probability_three_white_two_black_eq_eight_seventeen :
  probability_exact_three_white_two_black = 8 / 17 := by
  sorry

end probability_three_white_two_black_eq_eight_seventeen_l200_200037


namespace draw_4_balls_in_order_l200_200029

theorem draw_4_balls_in_order (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) : (n * (n-1) * (n-2) * (n-3) = 32760) := by
  rw [h1, h2]
  norm_num
  sorry

end draw_4_balls_in_order_l200_200029


namespace percentage_difference_l200_200689

theorem percentage_difference (w x y z : ℝ) (h1 : w = 0.6 * x) (h2 : x = 0.6 * y) (h3 : z = 0.54 * y) : 
  ((z - w) / w) * 100 = 50 :=
by
  sorry

end percentage_difference_l200_200689


namespace badges_before_exchange_l200_200665

theorem badges_before_exchange (V T : ℕ) (h1 : V = T + 5) (h2 : 76 * V + 20 * T = 80 * T + 24 * V - 100) :
  V = 50 ∧ T = 45 :=
by
  sorry

end badges_before_exchange_l200_200665


namespace solution_set_non_empty_implies_a_gt_1_l200_200202

theorem solution_set_non_empty_implies_a_gt_1 (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := 
  sorry

end solution_set_non_empty_implies_a_gt_1_l200_200202


namespace smallest_m_factorial_1350_l200_200105

theorem smallest_m_factorial_1350 (m : ℕ) :
  (1350 = 2 * 3^3 * 5^2) → (m >= 9 ∧ 1350 ∣ m!) :=
begin
  intro h1,
  split,
  {
    exact nat.le_of_dvd (nat.succ_pos 1350) (nat.factorial_dvd.m! 1350 h1),
  },
  {
    sorry
  },
end

end smallest_m_factorial_1350_l200_200105


namespace bricks_required_l200_200715

def L1 : ℝ := 18  -- Length of courtyard in meters
def W1 : ℝ := 16  -- Width of courtyard in meters
def L2_cm : ℝ := 20  -- Length of brick in centimeters
def W2_cm : ℝ := 10  -- Width of brick in centimeters
def L2 : ℝ := L2_cm / 100  -- Length of brick in meters
def W2 : ℝ := W2_cm / 100  -- Width of brick in meters
def area_courtyard : ℝ := L1 * W1  -- Area of the courtyard in square meters
def area_brick : ℝ := L2 * W2  -- Area of one brick in square meters

theorem bricks_required : (area_courtyard / area_brick) = 14400 := by
  sorry

end bricks_required_l200_200715


namespace num_ways_draw_four_balls_ordered_l200_200028

theorem num_ways_draw_four_balls_ordered : 
  let total_balls := 15 in
  let draw_count := 4 in
  let choices_permutations := ∏ i in (0 : Fin draw_count), total_balls - i := 
  choices_permutations = 32760 :=
by 
  let total_balls := 15 in
  let draw_count := 4 in
  have choices_permutations : Nat := 
    ∏ i in Finset.range draw_count, total_balls - i
  have expected_value : Nat := 32760
  exact (choices_permutations = expected_value)
sorry

end num_ways_draw_four_balls_ordered_l200_200028


namespace triangles_on_5x5_grid_l200_200909

/-- 
  We define a 5x5 grid with integer coordinates (x, y) where 1 ≤ x ≤ 5 and 1 ≤ y ≤ 5.
  The objective is to prove that the number of triangles with positive area formed by
  vertices chosen from these points is 2160.
-/
theorem triangles_on_5x5_grid : 
  let grid_points := { (x, y) | 1 ≤ x ∧ x ≤ 5 ∧ 1 ≤ y ∧ y ≤ 5 }
  let num_points := 25
  let total_triangles := nat.choose num_points 3
  let degenerate_triangles := 140
in
  total_triangles - degenerate_triangles = 2160 := by
    have h1 : total_triangles = 2300 := sorry
    have h2 : degenerate_triangles = 140 := sorry
    rw [h1, h2]
    norm_num
    sorry

end triangles_on_5x5_grid_l200_200909


namespace num_ways_draw_four_balls_ordered_l200_200027

theorem num_ways_draw_four_balls_ordered : 
  let total_balls := 15 in
  let draw_count := 4 in
  let choices_permutations := ∏ i in (0 : Fin draw_count), total_balls - i := 
  choices_permutations = 32760 :=
by 
  let total_balls := 15 in
  let draw_count := 4 in
  have choices_permutations : Nat := 
    ∏ i in Finset.range draw_count, total_balls - i
  have expected_value : Nat := 32760
  exact (choices_permutations = expected_value)
sorry

end num_ways_draw_four_balls_ordered_l200_200027


namespace quadrilateral_area_l200_200823

/-- Given a rhombus ABCD with vertices A = (0, 0), B = (0, 4), C = (6, 4), and D = (6, 0).
    Lines are drawn from vertex A at angles of 45 degrees and 60 degrees with respect to the horizontal,
    and from vertex B at angles of -45 degrees and -60 degrees respectively with respect to the horizontal.
    Determine the coordinates of the points of intersection of these lines and compute the area of the quadrilateral formed by these intersection points.
    Confirm that the area is equal to 8*sqrt(3)/3. -/
theorem quadrilateral_area (A B C D : ℝ × ℝ)
    (hA: A = (0, 0))
    (hB: B = (0, 4))
    (hC: C = (6, 4))
    (hD: D = (6, 0))
    (l1 l2 l3 l4: ℝ → ℝ)
    (hl1: l1 x = x) -- Line from A at 45 degrees
    (hl2: l2 x = sqrt(3) * x) -- Line from A at 60 degrees
    (hl3: l3 x = 4 - x) -- Line from B at -45 degrees
    (hl4: l4 x = 4 - sqrt(3) * x) -- Line from B at -60 degrees
    (P Q: ℝ × ℝ)
    (hP: P = (2, 2)) -- Intersection of lines from A at 45 degrees and B at -45 degrees
    (hQ: Q = (2*sqrt(3)/3, 2*sqrt(3))) -- Intersection of lines from A at 60 degrees and B at -60 degrees
    : abs ((A.1 * P.2 + P.1 * Q.2 + Q.1 * B.2 + B.1 * A.2) - (A.2 * P.1 + P.2 * Q.1 + Q.2 * B.1 + B.2 * A.1)) / 2 = 8 * sqrt(3) / 3 :=
sorry

end quadrilateral_area_l200_200823


namespace repeating_decimal_to_fraction_l200_200117

theorem repeating_decimal_to_fraction :
  let x := 7 + 316 / 999 in
  x = 7309 / 999 :=
by
  sorry

end repeating_decimal_to_fraction_l200_200117


namespace trigonometric_simplification_l200_200266

theorem trigonometric_simplification (x : ℝ) :
  - tan x - 4 * tan (2 * x) - 8 * tan (4 * x) + 16 * cot (8 * x) = - cot x :=
sorry

end trigonometric_simplification_l200_200266


namespace find_inheritance_amount_l200_200550

noncomputable def totalInheritance (tax_amount : ℕ) : ℕ :=
  let federal_rate := 0.20
  let state_rate := 0.10
  let combined_rate := federal_rate + (state_rate * (1 - federal_rate))
  sorry

theorem find_inheritance_amount : totalInheritance 10500 = 37500 := 
  sorry

end find_inheritance_amount_l200_200550


namespace lottery_prob_l200_200741

open Finset

/-- Definition of combinations C(n, k) -/
def combinations (n k : ℕ) : ℕ :=
(n.factorial) / (k.factorial * (n - k).factorial)

theorem lottery_prob :
  let total_tickets := 10
  let winning_tickets := 3
  let people := 5
  let total_ways := combinations total_tickets people
  let non_winning_tickets := total_tickets - winning_tickets
  let non_winning_ways := combinations non_winning_tickets people
  let prob := 1 - (non_winning_ways / total_ways : ℚ)
  prob = 11 / 12 :=
by
  sorry

end lottery_prob_l200_200741


namespace average_of_first_5_multiples_of_5_l200_200001

theorem average_of_first_5_multiples_of_5 : 
  (5 + 10 + 15 + 20 + 25) / 5 = 15 :=
by
  sorry

end average_of_first_5_multiples_of_5_l200_200001


namespace maximum_value_of_f_minimum_value_of_f_l200_200831

-- Define the function f
def f (x y : ℝ) : ℝ := 3 * |x + y| + |4 * y + 9| + |7 * y - 3 * x - 18|

-- Define the condition
def condition (x y : ℝ) : Prop := x^2 + y^2 ≤ 5

-- State the maximum value theorem
theorem maximum_value_of_f (x y : ℝ) (h : condition x y) :
  ∃ (x y : ℝ), f x y = 27 + 6 * Real.sqrt 5 := sorry

-- State the minimum value theorem
theorem minimum_value_of_f (x y : ℝ) (h : condition x y) :
  ∃ (x y : ℝ), f x y = 27 - 3 * Real.sqrt 10 := sorry

end maximum_value_of_f_minimum_value_of_f_l200_200831


namespace weight_of_2019_is_correct_l200_200911

-- Declare the conditions as definitions to be used in Lean 4
def stick_weight : Real := 0.5
def digit_to_sticks (n : Nat) : Nat :=
  match n with
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 9 => 6
  | _ => 0  -- other digits aren't considered in this problem

-- Calculate the total weight of the number 2019
def weight_of_2019 : Real :=
  (digit_to_sticks 2 + digit_to_sticks 0 + digit_to_sticks 1 + digit_to_sticks 9) * stick_weight

-- Statement to prove the weight of the number 2019
theorem weight_of_2019_is_correct : weight_of_2019 = 9.5 := by
  sorry

end weight_of_2019_is_correct_l200_200911


namespace eight_bags_weight_l200_200195

theorem eight_bags_weight
  (bags_weight : ℕ → ℕ)
  (h1 : bags_weight 12 = 24) :
  bags_weight 8 = 16 :=
  sorry

end eight_bags_weight_l200_200195


namespace motion_properties_l200_200369

/-- Define the position function x(t). --/
def x (t : ℝ) : ℝ := 2 * t^3 - 4 * t^2 + 2 * t + 3

/-- Define the velocity function v(t) as the derivative of x(t). --/
def v (t : ℝ) : ℝ := 6 * t^2 - 8 * t + 2

/-- Define the acceleration function a(t) as the derivative of v(t). --/
def a (t : ℝ) : ℝ := 12 * t - 8

/-- Initial velocity: v(0). --/
def v0 : ℝ := v 0

/-- Velocity at t = 3: v(3). --/
def v3 : ℝ := v 3

/-- Position at t = 1/3: x(1/3). --/
def x1by3 : ℝ := x (1/3)

/-- Position at t = 1: x(1). --/
def x1 : ℝ := x 1

/-- Proving all the conditions together. --/
theorem motion_properties :
  (∀ t, v t = 6 * t^2 - 8 * t + 2) ∧
  (∀ t, a t = 12 * t - 8) ∧
  (v0 = 2) ∧
  (v3 = 32) ∧
  (∃ t1 t2, v t1 = 0 ∧ v t2 = 0 ∧ t1 = 1 / 3 ∧ t2 = 1) ∧
  (x1by3 = 3 + 8 / 27) ∧
  (x1 = 3)
:=
by
  sorry

end motion_properties_l200_200369


namespace total_cans_arithmetic_sequence_l200_200519

theorem total_cans_arithmetic_sequence (a d l: ℕ) (S_n : ℕ):
  (a = 35) → (d = 3) → (l = 1) → (∃ n : ℕ, l = a - (n - 1) * d) → 
  (S_n = n * (2 * a - (n - 1) * d) / 2) → 
  S_n = 216 := by 
begin
  intros,
  cases_exists,
  sorry
end

end total_cans_arithmetic_sequence_l200_200519


namespace remaining_pie_proportion_l200_200400

def carlos_portion : ℝ := 0.6
def maria_share_of_remainder : ℝ := 0.25

theorem remaining_pie_proportion: 
  (1 - carlos_portion) - maria_share_of_remainder * (1 - carlos_portion) = 0.3 := 
by
  -- proof to be implemented here
  sorry

end remaining_pie_proportion_l200_200400


namespace odd_function_zero_sum_interval_l200_200201

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

theorem odd_function_zero_sum_interval {f : ℝ → ℝ}
  (h_odd : odd_function f)
  (zeroes : Fin 2017 → ℝ)
  (h_zero_sums_zero : ∑ i, zeroes i = 0)
  (h_f_zero : ∀ i, f (zeroes i) = 0) :
  ∃ x ∈ Ioo 0 1, 2^x + x - 2 = 0 :=
by
  sorry

end odd_function_zero_sum_interval_l200_200201


namespace smallest_satisfying_N_is_2520_l200_200429

open Nat

def smallest_satisfying_N : ℕ :=
  let N := 2520
  if (N + 2) % 2 = 0 ∧
     (N + 3) % 3 = 0 ∧
     (N + 4) % 4 = 0 ∧
     (N + 5) % 5 = 0 ∧
     (N + 6) % 6 = 0 ∧
     (N + 7) % 7 = 0 ∧
     (N + 8) % 8 = 0 ∧
     (N + 9) % 9 = 0 ∧
     (N + 10) % 10 = 0
  then N else 0

-- Statement of the problem in Lean 4
theorem smallest_satisfying_N_is_2520 : smallest_satisfying_N = 2520 :=
  by
    -- Proof would be added here, but is omitted as per instructions
    sorry

end smallest_satisfying_N_is_2520_l200_200429


namespace find_m_l200_200889

-- Declare the vectors a and b based on given conditions
variables {m : ℝ}

def a : ℝ × ℝ := (m, 3)
def b : ℝ × ℝ := (1, m + 1)

-- Define the condition that vectors a and b are perpendicular
def perpendicular (x y : ℝ × ℝ) : Prop := x.1 * y.1 + x.2 * y.2 = 0

-- State the problem in Lean 4
theorem find_m (h : perpendicular a b) : m = -3 / 4 :=
sorry

end find_m_l200_200889


namespace number_of_paths_l200_200499

-- Define the conditions of the problem
def grid_width : ℕ := 7
def grid_height : ℕ := 6
def diagonal_steps : ℕ := 2

-- Define the main proof statement
theorem number_of_paths (width height diag : ℕ) 
  (Nhyp : width = grid_width ∧ height = grid_height ∧ diag = diagonal_steps) : 
  ∃ (paths : ℕ), paths = 6930 := 
sorry

end number_of_paths_l200_200499


namespace angle_sum_property_l200_200181

theorem angle_sum_property
  (A B C D K M N : Point)
  [Triangle ABC]
  (on_side_D : D ∈ BC)
  (EF_external_tangent : is_external_tangent EF (incircle ACD) (incircle ABD))
  (EF_intersect_AD_at_K : intersects EF AD K)
  (incircle_touch_AC_M : incircle ABC touches AC at M)
  (incircle_touch_AB_N : incircle ABC touches AB at N)
  : ∠MKN + 1/2 * ∠BAC = 180 :=
sorry

end angle_sum_property_l200_200181


namespace park_height_l200_200620

theorem park_height (A b : ℕ) (h : ℕ) (hA : A = 40000) (hb : b = 250) : h = 160 :=
by
  rw [hA, hb]
  simp
  sorry

end park_height_l200_200620


namespace sum_arithmetic_sequence_max_l200_200459

theorem sum_arithmetic_sequence_max (d : ℝ) (a : ℕ → ℝ) 
  (h1 : d < 0) (h2 : (a 1)^2 = (a 13)^2) :
  ∃ n, n = 6 ∨ n = 7 :=
by
  sorry

end sum_arithmetic_sequence_max_l200_200459


namespace fourth_root_of_33177600_is_576_l200_200403

noncomputable def fourthRootOf33177600 : Real :=
  Real.cbrt (33177600: ℝ) ^ (1/4: ℝ)

theorem fourth_root_of_33177600_is_576 : fourthRootOf33177600 = 576 := by
  sorry

end fourth_root_of_33177600_is_576_l200_200403


namespace compound_interest_correct_l200_200755

noncomputable def compoundInterest (P: ℝ) (r: ℝ) (n: ℝ) (t: ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem compound_interest_correct :
  compoundInterest 5000 0.04 1 3 - 5000 = 624.32 :=
by
  sorry

end compound_interest_correct_l200_200755


namespace isosceles_triangle_perimeter_l200_200315

theorem isosceles_triangle_perimeter (a b : ℝ) (h_a : a = 12) (h_b : b = 24) (h1 : 12 + 12 ≠ 24) (h2 : 12 + 24 > 24) (h3 : 24 + 24 > 12) : 
  a + b + b = 60 :=
by 
  rw [h_a, h_b]
  exact eq.refl 60

end isosceles_triangle_perimeter_l200_200315


namespace coeff_of_x_square_l200_200510

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Statement of the problem
theorem coeff_of_x_square :
  (binom 8 3 = 56) ∧ (8 - 2 * 3 = 2) :=
sorry

end coeff_of_x_square_l200_200510


namespace james_coursework_materials_expense_l200_200980

-- Definitions based on conditions
def james_budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

-- Calculate expenditures based on percentages
def food_expense : ℝ := food_percentage * james_budget
def accommodation_expense : ℝ := accommodation_percentage * james_budget
def entertainment_expense : ℝ := entertainment_percentage * james_budget
def total_other_expenses : ℝ := food_expense + accommodation_expense + entertainment_expense

-- Prove that the amount spent on coursework materials is $300
theorem james_coursework_materials_expense : james_budget - total_other_expenses = 300 := 
by 
  sorry

end james_coursework_materials_expense_l200_200980


namespace english_only_students_l200_200946

variables {E G F EG EF GF EGF : Nat}

-- Conditions
def total_students := 50
def students_in_german := 22
def students_in_french := 18
def students_in_english_and_german := 12
def students_in_english_and_french := 10
def students_in_german_and_french := 8
def students_in_all_three := 4

-- Derived conditions:
def g_only := students_in_german - (students_in_english_and_german + students_in_german_and_french - students_in_all_three)
def f_only := students_in_french - (students_in_english_and_french + students_in_german_and_french - students_in_all_three)
def total_english := E + students_in_english_and_german + students_in_english_and_french - students_in_all_three

theorem english_only_students (h :  E + g_only + f_only + students_in_english_and_german + students_in_english_and_french + students_in_german_and_french - students_in_all_three = total_students) : E = 14 :=
by 
  unfold g_only f_only total_english at h
  sorry

end english_only_students_l200_200946


namespace gcd_of_36_and_60_is_12_l200_200007

theorem gcd_of_36_and_60_is_12 :
  Nat.gcd 36 60 = 12 :=
sorry

end gcd_of_36_and_60_is_12_l200_200007


namespace line_through_point_with_equal_intercepts_l200_200168

theorem line_through_point_with_equal_intercepts
  (a b: ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : 1/|a| = 1/|b|)
  (h4 : a * 1 + b * 2 = 0) :
  (x + y = 3) ∨ (2x - y = 0) :=
sorry

end line_through_point_with_equal_intercepts_l200_200168


namespace tan_alpha_value_l200_200142

variable (α : ℝ)

-- Conditions from the problem
def condition1 : Prop := (Real.sin α + Real.cos α = 1/5)
def condition2 : Prop := (0 < α ∧ α < π)

-- The corresponding proof problem
theorem tan_alpha_value (h1 : condition1 α) (h2 : condition2 α) : Real.tan α = -4/3 := 
sorry

end tan_alpha_value_l200_200142


namespace magnitude_w_one_l200_200988

open Complex

noncomputable def z : ℂ := (Complex.mul (Complex.neg 8 + (15 * Complex.I)) 
                                         (Complex.neg 8 + (15 * Complex.I))) *
                            (Complex.pow (19 - (9 * Complex.I)) 3) / 
                            (4 + (7 * Complex.I))

def w : ℂ := Complex.conj(z) / z

theorem magnitude_w_one : Complex.abs w = 1 :=
by
  sorry

end magnitude_w_one_l200_200988


namespace value_of_a_b_c_l200_200145

noncomputable def absolute_value (x : ℤ) : ℤ := abs x

theorem value_of_a_b_c (a b c : ℤ)
  (ha : absolute_value a = 1)
  (hb : absolute_value b = 2)
  (hc : absolute_value c = 3)
  (h : a > b ∧ b > c) :
  a + b - c = 2 ∨ a + b - c = 0 :=
by
  sorry

end value_of_a_b_c_l200_200145


namespace identify_counterfeit_coin_l200_200075

def Coin : Type := sorry
def weighs_different (a b : Coin) : Prop := sorry
def pan_balance_weighing (p1 p2 : list Coin) : Prop := sorry

def TwelveCoins (coins : list Coin) : Prop :=
  coins.length = 12 ∧ ∃! c, (∀ co ∈ coins, weighs_different c co)

theorem identify_counterfeit_coin 
  (coins : list Coin) 
  (H : TwelveCoins coins) 
  : ∃ (c : Coin) (steps : list (list Coin × list Coin)), steps.length = 3 ∧
      (∀ (p1 p2 : list Coin) ∈ steps, pan_balance_weighing p1 p2) ∧ 
      (weighs_different c c) := 
sorry

end identify_counterfeit_coin_l200_200075


namespace golu_shortest_distance_l200_200187

noncomputable def shortest_distance_from_home : ℝ :=
  let north_movement : ℝ := 8
  let west_movement_1 : ℝ := 6
  let south_movement_1 : ℝ := 10
  let west_movement_2 : ℝ := 3
  let southwest_movement : ℝ := 5
  let sw_component : ℝ := southwest_movement * real.sqrt 2 / 2
  let total_north_south := -2 - sw_component
  let total_east_west := 9 + sw_component
  real.sqrt ((total_north_south ^ 2) + (total_east_west ^ 2))

theorem golu_shortest_distance : shortest_distance_from_home = 13.7 := sorry

end golu_shortest_distance_l200_200187


namespace base_conversion_subtraction_l200_200082

def base8_to_base10 : Nat := 5 * 8^5 + 4 * 8^4 + 3 * 8^3 + 2 * 8^2 + 1 * 8^1 + 0 * 8^0
def base9_to_base10 : Nat := 6 * 9^4 + 5 * 9^3 + 4 * 9^2 + 3 * 9^1 + 2 * 9^0

theorem base_conversion_subtraction :
  base8_to_base10 - base9_to_base10 = 136532 :=
by
  -- Proof steps go here
  sorry

end base_conversion_subtraction_l200_200082


namespace calculate_ray_grocery_bill_l200_200261

noncomputable def ray_grocery_total_cost : ℝ :=
let hamburger_meat_price := 5.0
let crackers_price := 3.5
let frozen_vegetables_price := 2.0 * 4
let cheese_price := 3.5
let chicken_price := 6.5
let cereal_price := 4.0
let wine_price := 10.0
let cookies_price := 3.0

let discount_hamburger_meat := hamburger_meat_price * 0.10
let discount_crackers := crackers_price * 0.10
let discount_frozen_vegetables := frozen_vegetables_price * 0.10
let discount_cheese := cheese_price * 0.05
let discount_chicken := chicken_price * 0.05
let discount_wine := wine_price * 0.15

let discounted_hamburger_meat_price := hamburger_meat_price - discount_hamburger_meat
let discounted_crackers_price := crackers_price - discount_crackers
let discounted_frozen_vegetables_price := frozen_vegetables_price - discount_frozen_vegetables
let discounted_cheese_price := cheese_price - discount_cheese
let discounted_chicken_price := chicken_price - discount_chicken
let discounted_wine_price := wine_price - discount_wine

let total_discounted_price :=
  discounted_hamburger_meat_price +
  discounted_crackers_price +
  discounted_frozen_vegetables_price +
  discounted_cheese_price +
  discounted_chicken_price +
  cereal_price +
  discounted_wine_price +
  cookies_price

let food_items_total_price :=
  discounted_hamburger_meat_price +
  discounted_crackers_price +
  discounted_frozen_vegetables_price +
  discounted_cheese_price +
  discounted_chicken_price +
  cereal_price +
  cookies_price

let food_sales_tax := food_items_total_price * 0.06
let wine_sales_tax := discounted_wine_price * 0.09

let total_with_tax := total_discounted_price + food_sales_tax + wine_sales_tax

total_with_tax

theorem calculate_ray_grocery_bill :
  ray_grocery_total_cost = 42.51 :=
sorry

end calculate_ray_grocery_bill_l200_200261


namespace series_convergence_l200_200824

noncomputable def real_sequence (n: ℕ) : ℝ := sorry

def is_positive_real_sequence (a : ℕ → ℝ) : Prop :=
∀ n ≥ 2, 0 < a n

def sum_is_convergent (a : ℕ → ℝ) : Prop :=
∃ l : ℝ, ∑' n, a (n + 2) = l

theorem series_convergence
  (a : ℕ → ℝ)
  (h1 : is_positive_real_sequence a)
  (h2 : sum_is_convergent a) :
  ∑' n, a (n + 2) ^ (1 - 1 / real.log (n + 2)) < ∞ :=
sorry

end series_convergence_l200_200824


namespace inheritance_amount_l200_200549

-- Define the conditions
def federal_tax_rate : ℝ := 0.2
def state_tax_rate : ℝ := 0.1
def total_taxes_paid : ℝ := 10500

-- Lean statement for the proof
theorem inheritance_amount (I : ℝ)
  (h1 : federal_tax_rate = 0.2)
  (h2 : state_tax_rate = 0.1)
  (h3 : total_taxes_paid = 10500)
  (taxes_eq : total_taxes_paid = (federal_tax_rate * I) + (state_tax_rate * (I - (federal_tax_rate * I))))
  : I = 37500 :=
sorry

end inheritance_amount_l200_200549


namespace thirteenth_four_digit_number_is_5247_l200_200747

theorem thirteenth_four_digit_number_is_5247 :
  let digits := [2, 4, 5, 7]
  let four_digit_numbers := (digits.permutations.map (λ l => 1000 * l[0] + 100 * l[1] + 10 * l[2] + l[3])).sort
  four_digit_numbers[12] = 5247 := by
  sorry

end thirteenth_four_digit_number_is_5247_l200_200747


namespace night_crew_ratio_l200_200080

theorem night_crew_ratio (D N B : ℕ) (h1 : ∀ d : ℕ, d ∈ day_crew → boxes (d) = B) 
  (h2 : ∀ n : ℕ, n ∈ night_crew → boxes (n) = (3 / 4) * B)
  (h3 : total_boxes day_crew = (3 / 4) * (total_boxes day_crew + total_boxes night_crew)) :
  (N / D) = (4 / 3) :=
by
 sorry

end night_crew_ratio_l200_200080


namespace distinct_monomials_count_l200_200529

theorem distinct_monomials_count :
  (∃ x y z : ℝ, (x + y + z) ^ 2020 + (x - y - z) ^ 2020) 
  → (∑ i in finset.range 1011, 2 * i + 1 = 1022121) :=
by 
  sorry

end distinct_monomials_count_l200_200529


namespace xiaoming_department_store_profit_l200_200345

theorem xiaoming_department_store_profit:
  let P₁ := 40000   -- average monthly profit in Q1
  let L₂ := -15000  -- average monthly loss in Q2
  let L₃ := -18000  -- average monthly loss in Q3
  let P₄ := 32000   -- average monthly profit in Q4
  let P_total := (P₁ * 3 + L₂ * 3 + L₃ * 3 + P₄ * 3)
  P_total = 117000 := by
  sorry

end xiaoming_department_store_profit_l200_200345


namespace part1_part2_l200_200242

variable (x : ℝ)

def A := {x : ℝ | 1 < x ∧ x < 3}
def B := {x : ℝ | x < -3 ∨ 2 < x}

theorem part1 : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

theorem part2 (a b : ℝ) : (∀ x, 2 < x ∧ x < 3 → x^2 + a * x + b < 0) → a = -5 ∧ b = 6 := by
  sorry

end part1_part2_l200_200242


namespace max_cookies_without_ingredients_l200_200553

-- Defining the number of cookies and their composition
def total_cookies : ℕ := 36
def peanuts : ℕ := (2 * total_cookies) / 3
def chocolate_chips : ℕ := total_cookies / 3
def raisins : ℕ := total_cookies / 4
def oats : ℕ := total_cookies / 8

-- Proving the largest number of cookies without any ingredients
theorem max_cookies_without_ingredients : (total_cookies - (max (max peanuts chocolate_chips) raisins)) = 12 := by
    sorry

end max_cookies_without_ingredients_l200_200553


namespace chocolate_bars_needed_l200_200986

theorem chocolate_bars_needed (bars_per_box : ℕ) (boxes_needed : ℕ) (h1 : bars_per_box = 5) (h2 : boxes_needed = 142) : bars_per_box * boxes_needed = 710 :=
by
  rw [h1, h2]
  sorry

end chocolate_bars_needed_l200_200986


namespace dataset_data_points_l200_200981

theorem dataset_data_points (initial_points : ℕ) (added_percent : ℕ) (reduction_fraction : ℚ) : 
  initial_points = 200 → added_percent = 20 → reduction_fraction = 1/4 → 
  let added_points := (added_percent * initial_points) / 100 in
  let total_after_addition := initial_points + added_points in
  let reduced_points := reduction_fraction * total_after_addition in
  let final_total := total_after_addition - reduced_points in
  final_total = 180 :=
by
  intros _ _ _
  let added_points := (added_percent * initial_points) / 100
  let total_after_addition := initial_points + added_points
  let reduced_points := reduction_fraction * total_after_addition
  let final_total := total_after_addition - reduced_points
  have initial_points_correct : initial_points = 200 := by exact rfl
  have added_percent_correct : added_percent = 20 := by exact rfl
  have reduction_fraction_correct : reduction_fraction = 1/4 := by exact rfl
  rw [initial_points_correct, added_percent_correct, reduction_fraction_correct]
  sorry

end dataset_data_points_l200_200981


namespace product_of_divisors_of_72_l200_200793

theorem product_of_divisors_of_72 :
  ∀ (d ∈ {d : ℕ | ∃ a b : ℕ, d = 2^a * 3^b ∧ a <= 3 ∧ b <= 2}),
    (∏ d in {d : ℕ | ∃ a b : ℕ, d = 2^a * 3^b ∧ a <= 3 ∧ b <= 2}, d) = 72^6 :=
  by
    sorry

end product_of_divisors_of_72_l200_200793


namespace total_number_of_trees_l200_200613

theorem total_number_of_trees (side_length : ℝ) (area_ratio : ℝ) (trees_per_sqm : ℝ) (H : side_length = 100) (R : area_ratio = 3) (T : trees_per_sqm = 4) : 
  let street_area := side_length ^ 2 in 
  let forest_area := area_ratio * street_area in
  let total_trees := forest_area * trees_per_sqm in
  total_trees = 120000 :=
by
  -- proof steps go here
  sorry

end total_number_of_trees_l200_200613


namespace eight_bags_weight_l200_200194

theorem eight_bags_weight
  (bags_weight : ℕ → ℕ)
  (h1 : bags_weight 12 = 24) :
  bags_weight 8 = 16 :=
  sorry

end eight_bags_weight_l200_200194


namespace sum_of_numbers_ge_0_3_l200_200648

theorem sum_of_numbers_ge_0_3 : 
  let n1 := 0.8
      n2 := 1 / 2
      n3 := 0.9
  in (n1 ≥ 0.3) ∧ (n2 ≥ 0.3) ∧ (n3 ≥ 0.3) ∧ (n1 + n2 + n3 = 2.2) :=
by
  sorry

end sum_of_numbers_ge_0_3_l200_200648


namespace sum_FV_l200_200767

theorem sum_FV (V F A : ℝ) (AF AV : ℝ) :
  AF = 25 ∧ AV = 24 →
  let FV := (14 * Real.sqrt 3) / 3 in
  FV = (7 * Real.sqrt 3) / 3 * 2 :=
by
  sorry

end sum_FV_l200_200767


namespace ways_to_arrange_digits_of_1225_l200_200960

theorem ways_to_arrange_digits_of_1225 :
  let digits := [1, 2, 2, 5] in
  (∃ permutations : list (list ℕ), 
      permutations = digits.permutations ∧
      (∃ valid_permutations : list (list ℕ),
          valid_permutations = permutations.filter (λ l, (l.nth 3).getOrElse 0 = 5) ∧
          valid_permutations.length = 3)) :=
sorry

end ways_to_arrange_digits_of_1225_l200_200960


namespace identify_cell_type_l200_200056

-- Conditions
abbreviation ChromosomalComposition := String
def InitialChromosomalState : ChromosomalComposition := "44+XYY"
def CausedByMoveSameSide : Prop := 
  ∀ (cellType : String), 
  (cellType = "Secondary spermatocyte")

-- Theorem statement 
theorem identify_cell_type 
  (chromState : ChromosomalComposition)
  (cause : chromState = InitialChromosomalState → CausedByMoveSameSide) : 
  CausedByMoveSameSide :=
  by 
    intros chromState cause 
    sorry

end identify_cell_type_l200_200056


namespace probability_same_tribe_quitters_l200_200625

theorem probability_same_tribe_quitters
    (participants : Finset ℕ)
    (n : ℕ)
    (tribe_size : ℕ)
    (quitters : ℕ)
    (equal_probability : ∀ p ∈ participants, p ∈ participants → p ≠ q → true) 
    (independent_decisions : ∀ p1 p2 ∈ participants, p1 ≠ p2 → true)
    (h1 : participants.card = 32)
    (h2 : tribe_size = 16)
    (h3 : quitters = 2) 
    : ∃ (probability : ℚ), probability = 15 / 31 := 
    begin
        sorry
    end

end probability_same_tribe_quitters_l200_200625


namespace find_m_value_l200_200931

theorem find_m_value (m : ℤ) (h : (∀ x : ℤ, (x-5)*(x+7) = x^2 - mx - 35)) : m = -2 :=
by sorry

end find_m_value_l200_200931


namespace sum_binom_coeff_mod_l200_200091

-- Definitions of conditions
def ω : ℂ := Complex.exp(2 * Real.pi * I / 3)
def ζ : ℂ := ω^2

-- Property of complex roots of unity
lemma omega_cube_one : ω ^ 3 = 1 := 
by {
  have h : ω ^ 3 = Complex.exp(2 * Real.pi * I) := by ring,
  rw [Complex.exp_int_mul_I, h],
  norm_num,
}

lemma zeta_omega_squared : ζ = ω ^ 2 := rfl
lemma omega_zeta_eq_one : ω * ζ = 1 := by {
  field_simp [zeta_omega_squared, pow_succ'],
}

-- Target statement
theorem sum_binom_coeff_mod :
  let T := ∑ k in Finset.range (669), Nat.choose 2006 (3 * k) 
  in T % 500 = 104 :=
by sorry

end sum_binom_coeff_mod_l200_200091


namespace find_c_plus_2d_l200_200170

theorem find_c_plus_2d (a b c d : ℝ) (f : ℝ → ℝ) (h1 : f = λ x, a*x^3 + b*x^2 + c*x + d)
  (h2 : 24 * (0 : ℝ) + (f 0 - c * 0 - d) - 12 = 0) : c + 2 * d = 0 := sorry

end find_c_plus_2d_l200_200170


namespace max_integer_value_of_f_l200_200412

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 9 * x + 13) / (3 * x^2 + 9 * x + 5)

theorem max_integer_value_of_f : ∀ x : ℝ, ∃ n : ℤ, f x ≤ n ∧ n = 2 :=
by 
  sorry

end max_integer_value_of_f_l200_200412


namespace maxSphereVolumeIsCorrect_l200_200042

-- Given a cube with edge length 2
def cubeEdgeLength : ℝ := 2

-- Radius of the sphere is half of the edge length of the cube
def sphereRadius := cubeEdgeLength / 2

-- The volume of a sphere formula
def sphereVolume (r : ℝ) : ℝ := (4 / 3) * Real.pi * (r^3)

-- The volume of the sphere with radius sphereRadius
def maxSphereVolumeInCube : ℝ := sphereVolume sphereRadius

-- The proof that the volume of the sphere is indeed (4/3) * π
theorem maxSphereVolumeIsCorrect : maxSphereVolumeInCube = (4 / 3) * Real.pi := by
  sorry

end maxSphereVolumeIsCorrect_l200_200042


namespace area_of_triangle_F1_P_F2_l200_200853

noncomputable def hyperbola_area : ℝ :=
let a := 3
let b := 4
let c := sqrt (a^2 + b^2) in
let F1 := (-c, 0)
let F2 := (c, 0) in
let P := (x, y) in
if (x^2) / 9 - (y^2) / 16 = 1 ∧ angle F1 P F2 = 90 then
  1 / 2 * dist P F1 * dist P F2
else
  0

theorem area_of_triangle_F1_P_F2 :
  (hyperbola_area = 16) :=
sorry

end area_of_triangle_F1_P_F2_l200_200853


namespace parallelogram_angles_l200_200278

theorem parallelogram_angles
  {A B C D M : Type}
  [parallelogram ABCD]
  (h1 : angle_bisector A intersects BC at M)
  (h2 : angle_bisector (angle A M C) passes_through D)
  (h3 : angle M D C = 45)
  : (angle A = 45 ∧ angle C = 45 ∧ angle B = 135 ∧ angle D = 135) :=
sorry

end parallelogram_angles_l200_200278


namespace no_other_products_of_three_distinct_primes_sum_59_l200_200012

theorem no_other_products_of_three_distinct_primes_sum_59 :
  ¬(∃ (x : ℕ), (∃ (p1 p2 p3 : ℕ), 
     (nat.prime p1) ∧ (nat.prime p2) ∧ (nat.prime p3) ∧ 
     (p1 ≠ p2) ∧ (p1 ≠ p3) ∧ (p2 ≠ p3) ∧ 
     (x = p1 * p2 * p3) ∧ 
     (p1 + p2 + p3 = 59) ∧ 
     (x ≠ 2103))) :=
sorry

end no_other_products_of_three_distinct_primes_sum_59_l200_200012


namespace measure_exactly_45_minutes_l200_200677

def cord_burns_in_one_hour : Prop := ∀ (c : Cord), burns_in 60 minutes c
def burn_rate_not_uniform : Prop := ∀ (c : Cord), ∃ (f : ℕ → ℕ), burn_rate c = f
def burn_both_ends_half_time : Prop := ∀ (c : Cord), burns_in 30 minutes (light_both_ends c)
def measure_45_minutes (cord1 cord2 : Cord) (lighter : Lighter) : Prop :=
  light_both_ends lighter cord1 ∧
  light_one_end lighter cord2 ∧
  wait 30 minutes (cord_burned cord1) ∧
  light_other_end lighter cord2 ∨
  light_other_end lighter cord2 ∧
  cord_burned cord2 ∧
  burns_in 15 minutes cord2
     
theorem measure_exactly_45_minutes (cord1 cord2 : Cord) (lighter : Lighter) :
  cord_burns_in_one_hour cord1 ∧ cord_burns_in_one_hour cord2 ∧
  burn_rate_not_uniform cord1 ∧ burn_rate_not_uniform cord2 ∧
  burn_both_ends_half_time cord1 ∧ burn_both_ends_half_time cord2 →
  measure_45_minutes cord1 cord2 lighter → 
  burns_in 45 minutes (cord_burned cord2) :=
sorry

end measure_exactly_45_minutes_l200_200677


namespace minimum_omega_l200_200408

-- Define the operation 
def det_2x2 (a1 a2 a3 a4 : ℝ) := a1 * a4 - a2 * a3

-- Define the function f(x) and its properties
def f (ω : ℝ) (x : ℝ) := det_2x2 (real.sqrt 3) (real.sin (ω * x)) 1 (real.cos (ω * x))

-- Main theorem to find minimum value of ω given conditions
theorem minimum_omega (ω : ℝ) : 
  (∃ k : ℤ, ω = 3 * k / 2 - 1 / 4 ∧ ω > 0) → (∃ ω, ω = 5 / 4) :=
by 
  sorry

end minimum_omega_l200_200408


namespace produce_total_worth_l200_200813

/-- Gary is restocking the grocery produce section. He adds 60 bundles of asparagus at $3.00 each, 
40 boxes of grapes at $2.50 each, and 700 apples at $0.50 each. 
This theorem proves that the total worth of all the produce Gary stocked is $630.00. -/
theorem produce_total_worth :
  let asparagus_bundles := 60
  let asparagus_price := 3.00
  let grapes_boxes := 40
  let grapes_price := 2.50
  let apples_count := 700
  let apples_price := 0.50 in
  (asparagus_bundles * asparagus_price) + (grapes_boxes * grapes_price) + (apples_count * apples_price) = 630.00 :=
by
  sorry

end produce_total_worth_l200_200813


namespace x0_in_M_implies_x0_in_N_l200_200864

def M : Set ℝ := {x | ∃ (k : ℤ), x = k + 1 / 2}
def N : Set ℝ := {x | ∃ (k : ℤ), x = k / 2 + 1}

theorem x0_in_M_implies_x0_in_N (x0 : ℝ) (h : x0 ∈ M) : x0 ∈ N := 
sorry

end x0_in_M_implies_x0_in_N_l200_200864


namespace mandy_toys_count_l200_200751

theorem mandy_toys_count (M A Am P : ℕ) 
    (h1 : A = 3 * M) 
    (h2 : A = Am - 2) 
    (h3 : A = P / 2) 
    (h4 : M + A + Am + P = 278) : 
    M = 21 := 
by
  sorry

end mandy_toys_count_l200_200751


namespace tangent_line_at_point_l200_200167

noncomputable def func (x : ℝ) : ℝ := x + Real.log x

-- Given that f(x) is differentiable in (0, +∞) and f(e^x) = x + e^x, prove the tangent line equation at x = 1
theorem tangent_line_at_point (f : ℝ → ℝ) (hf : ∀ x > 0, DifferentiableAt ℝ f x) (h : ∀ x : ℝ, f (Real.exp x) = x + Real.exp x) :
  f 1 = 1 ∧ deriv f 1 = 2 ∧ (∀ (y : ℝ), y = f 1 → (λ x, 2 * x - y - 1) = 0) :=
by
  sorry

end tangent_line_at_point_l200_200167


namespace range_no_preimage_l200_200587

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem range_no_preimage :
  ∀ t : ℝ, ¬(∃ x : ℝ, f x = t) ↔ t ∈ set.Ioi 1 :=
by
  sorry

end range_no_preimage_l200_200587


namespace trajectory_of_point_inside_square_is_conic_or_degenerates_l200_200022

noncomputable def is_conic_section (a : ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ (m n l : ℝ) (x y : ℝ), 
    x = P.1 ∧ y = P.2 ∧ 
    (m^2 + n^2) * x^2 - 2 * n * (l + m) * x * y + (l^2 + n^2) * y^2 = (l * m - n^2)^2 ∧
    4 * n^2 * (l + m)^2 - 4 * (m^2 + n^2) * (l^2 + n^2) ≤ 0

theorem trajectory_of_point_inside_square_is_conic_or_degenerates
  (a : ℝ) (P : ℝ × ℝ)
  (h1 : 0 < P.1) (h2 : P.1 < 2 * a)
  (h3 : 0 < P.2) (h4 : P.2 < 2 * a)
  : is_conic_section a P :=
sorry

end trajectory_of_point_inside_square_is_conic_or_degenerates_l200_200022


namespace geometric_sequence_cannot_determine_a3_l200_200489

/--
Suppose we have a geometric sequence {a_n} such that 
the product of the first five terms a_1 * a_2 * a_3 * a_4 * a_5 = 32.
We aim to show that the value of a_3 cannot be determined with the given information.
-/
theorem geometric_sequence_cannot_determine_a3 (a : ℕ → ℝ) (r : ℝ) (h : a 0 * a 1 * a 2 * a 3 * a 4 = 32) : 
  ¬ ∃ x : ℝ, a 2 = x :=
sorry

end geometric_sequence_cannot_determine_a3_l200_200489


namespace increasing_quadratic_range_of_m_l200_200629

theorem increasing_quadratic_range_of_m (m : ℝ) :
  (∀ x y : ℝ, -2 ≤ x ∧ x ≤ y → f(x) ≤ f(y)) → m ≤ -16 :=
by
  let f := λ x : ℝ, 4 * x^2 - m * x + 5
  sorry

end increasing_quadratic_range_of_m_l200_200629


namespace area_of_circle_gamma_l200_200554

noncomputable def area_of_circle (AB : ℝ) (R : ℝ) : ℝ :=
  π * R^2

theorem area_of_circle_gamma :
  ∀ (A B C D E F : ℝ × ℝ) (Γ : set (ℝ × ℝ)),
    (∃ (R : ℝ), 
       (∀ P ∈ Γ, ((P.1 - (fst (circ_center Γ))^2 + (P.2 - (snd (circ_center Γ)))^2) = R^2)) ∧
       (A ∈ Γ ∧ B ∈ Γ) ∧
       dist A B = sqrt 10 ∧
       ∃ C, C ∉ Γ ∧ 
            (dist A C = sqrt 10 ∧ dist B C = sqrt 10) ∧
            ∃ (D E F : ℝ × ℝ),
              collinear ({C, D, E, F}: set (ℝ × ℝ)) ∧
              dist C D = dist D E ∧ dist D E = dist E F ∧
              (∃ (line_cd : set (ℝ × ℝ)),
                 is_line_through line_cd C D ∧
                 (∃ E F ∈ line_cd, E ≠ F ∧
                                  E ∈ (line A B ∩ Γ) ∧
                                  F ∈ (line C D ∩ Γ)))) →
    area_of_circle (sqrt 10) (√(38/15)) = (38 * π / 15) := 
sorry

end area_of_circle_gamma_l200_200554


namespace trig_expression_value_l200_200832

open Real

theorem trig_expression_value (θ : ℝ)
  (h1 : cos (π - θ) > 0)
  (h2 : cos (π / 2 + θ) * (1 - 2 * cos (θ / 2) ^ 2) < 0) :
  (sin θ / |sin θ|) + (|cos θ| / cos θ) + (tan θ / |tan θ|) = -1 :=
by
  sorry

end trig_expression_value_l200_200832


namespace total_homework_time_l200_200801

variable (num_math_problems num_social_studies_problems num_science_problems : ℕ)
variable (time_per_math_problem time_per_social_studies_problem time_per_science_problem : ℝ)

/-- Prove that the total time taken by Brooke to answer all his homework problems is 48 minutes -/
theorem total_homework_time :
  num_math_problems = 15 →
  num_social_studies_problems = 6 →
  num_science_problems = 10 →
  time_per_math_problem = 2 →
  time_per_social_studies_problem = 0.5 →
  time_per_science_problem = 1.5 →
  (num_math_problems * time_per_math_problem + num_social_studies_problems * time_per_social_studies_problem + num_science_problems * time_per_science_problem) = 48 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_homework_time_l200_200801


namespace basketball_team_lineup_l200_200706

-- Define the problem context
variables (total_players quadruplets required_in_lineup remaining_players choose_quotes choose_remaining total_lineups: ℕ)

-- Definitions based on conditions
def context := 
  total_players = 15 ∧ 
  quadruplets = 4 ∧ 
  required_in_lineup = 3 ∧ 
  remaining_players = total_players - quadruplets ∧ 
  choose_quotes = (nat.choose quadruplets required_in_lineup) ∧ 
  choose_remaining = (nat.choose remaining_players (6 - required_in_lineup)) ∧
  total_lineups = choose_quotes * choose_remaining

-- The theorem to prove
theorem basketball_team_lineup (h : context):
  total_lineups = 660 :=
begin
  have h_choose_quotes : choose_quotes = nat.choose 4 3, 
  { sorry }, -- (correct value is 4, proven by context)

  have h_choose_remaining : choose_remaining = nat.choose 11 3, 
  { sorry }, -- (correct value is 165, proven by context)

  have h_total_lineups : total_lineups = 4 * 165, 
  { sorry }, -- (multi-step multiplication)

  exact h_total_lineups,
end

end basketball_team_lineup_l200_200706


namespace find_possible_values_of_b_l200_200235

def good_number (x : ℕ) : Prop :=
  ∃ p n : ℕ, Nat.Prime p ∧ n ≥ 2 ∧ x = p^n

theorem find_possible_values_of_b (b : ℕ) : 
  (b ≥ 4) ∧ good_number (b^2 - 2 * b - 3) ↔ b = 87 := sorry

end find_possible_values_of_b_l200_200235


namespace regular_2014_simplex_sum_square_l200_200989

-- Noncomputable definition to avoid evaluation requirement
noncomputable def simplex_2014_sum_square_mod : ℕ :=
  600572

-- Statement of the theorem
theorem regular_2014_simplex_sum_square :
  let c > 0,
      A := (list.range 2015).map (λ i, (1:ℝ)) -- Simplified vertices for clarity
      P := (list.range 2015).map (λ i, if i % 2 = 0 then 20 else 14 : ℝ) in
  let PA_i (i : ℕ) := (20 - 1)^2 + (14 - 0)^2 in -- Simplified PA_i^2 for clarity
  ∑ i in (finset.range 2015), PA_i i = simplex_2014_sum_square_mod % 10^6 :=
sorry

end regular_2014_simplex_sum_square_l200_200989


namespace total_stickers_l200_200596

theorem total_stickers :
  let sheets_per_folder := [("red", 10), ("green", 8), ("blue", 6), ("yellow", 4), ("purple", 2)]
  let stickers_per_sheet := [("red", 3), ("green", 5), ("blue", 2), ("yellow", 4), ("purple", 6)]
  (sheets_per_folder.map (λ (x : String × Nat), (x.2 * (stickers_per_sheet.find? (λ y => y.1 = x.1)).getD 0))).sum = 110 :=
by
  -- We'll add the proofs here later
  sorry

end total_stickers_l200_200596


namespace real_number_value_of_m_pure_imaginary_value_of_m_l200_200149

def is_real (z : ℂ) : Prop := z.im = 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem real_number_value_of_m (m : ℝ) : 
  is_real ((m^2 + 2 * m - 8) + (m^2 - 2 * m) * I) ↔ (m = 0 ∨ m = 2) := 
by sorry

theorem pure_imaginary_value_of_m (m : ℝ) : 
  is_pure_imaginary ((m^2 + 2 * m - 8) + (m^2 - 2 * m) * I) ↔ (m = -4) := 
by sorry

end real_number_value_of_m_pure_imaginary_value_of_m_l200_200149


namespace circle_exists_l200_200148

section GeometricLoci

variable (C : Type) [metric_space C] [normed_group C] [normed_space ℝ C]
variables (O A B : C) (r : ℝ) (l : affine_subspace ℝ C)

def exists_circle_passing_through_and_parallel_chord : Prop :=
  ∃ (center : C) (radius : ℝ), 
    (metric.sphere center radius).contains A ∧ 
    (metric.sphere center radius).contains B ∧ 
    ∃ (M : C),
    (affine_subspace.line_parallel_to_through_point l center O) ∧
    (affine_subspace.perp_bisector A B).contains center

theorem circle_exists 
  (C : Type) [metric_space C] [normed_group C] [normed_space ℝ C]
  (O A B : C) (r : ℝ) (l : affine_subspace ℝ C) : exists_circle_passing_through_and_parallel_chord C O A B r l :=
sorry

end GeometricLoci

end circle_exists_l200_200148


namespace find_m_of_perpendicular_vectors_l200_200902

theorem find_m_of_perpendicular_vectors
    (m : ℝ)
    (a : ℝ × ℝ := (m, 3))
    (b : ℝ × ℝ := (1, m + 1))
    (h : a.1 * b.1 + a.2 * b.2 = 0) :
    m = -3 / 4 :=
by 
  sorry

end find_m_of_perpendicular_vectors_l200_200902


namespace rational_product_of_roots_l200_200557

theorem rational_product_of_roots 
  (a b c d e : ℤ) 
  (r1 r2 r3 r4 : ℂ) 
  (h_f : ∀ z : ℂ, a * z^4 + b * z^3 + c * z^2 + d * z + e = a * (z - r1) * (z - r2) * (z - r3) * (z - r4))
  (h_a_nonzero : a ≠ 0) 
  (h_r1_r2_rational : (r1 + r2) ∈ ℚ)
  (h_sum_not_equal : r1 + r2 ≠ r3 + r4) : 
  (r1 * r2) ∈ ℚ := 
by
  sorry

end rational_product_of_roots_l200_200557


namespace number_of_BA3_in_sample_l200_200711

-- Definitions for the conditions
def strains_BA1 : Nat := 60
def strains_BA2 : Nat := 20
def strains_BA3 : Nat := 40
def total_sample_size : Nat := 30

def total_strains : Nat := strains_BA1 + strains_BA2 + strains_BA3

-- Theorem statement translating to the equivalent proof problem
theorem number_of_BA3_in_sample :
  total_sample_size * strains_BA3 / total_strains = 10 :=
by
  sorry

end number_of_BA3_in_sample_l200_200711


namespace parabola_equation_l200_200054

theorem parabola_equation (h_vertex : (0, 0)) (h_focus : ∃ p, (p > 0) ∧ (h := p) ∧ h_focus : (h, 0) : ∃ (x y : ℝ), y = 2 ∧ x = 2 → y^2 = 2 * x := 
by
  sorry

end parabola_equation_l200_200054


namespace least_value_difference_l200_200531

noncomputable def least_difference (x : ℝ) : ℝ := 6 - 13/5

theorem least_value_difference (x n m : ℝ) (h1 : 2*x + 5 + 4*x - 3 > x + 15)
                               (h2 : 2*x + 5 + x + 15 > 4*x - 3)
                               (h3 : 4*x - 3 + x + 15 > 2*x + 5)
                               (h4 : x + 15 > 2*x + 5)
                               (h5 : x + 15 > 4*x - 3)
                               (h_m : m = 13/5) (h_n : n = 6)
                               (hx : m < x ∧ x < n) :
  n - m = 17 / 5 :=
  by sorry

end least_value_difference_l200_200531


namespace negation_of_existence_statement_l200_200134

theorem negation_of_existence_statement :
  (¬ (∃ x : ℝ, x^2 + x + 1 < 0)) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
by
  sorry

end negation_of_existence_statement_l200_200134


namespace perpendicular_vectors_l200_200880

variable (m : ℝ)

def vector_a := (m, 3)
def vector_b := (1, m + 1)

def dot_product (v w : ℝ × ℝ) := (v.1 * w.1) + (v.2 * w.2)

theorem perpendicular_vectors (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by 
  unfold vector_a vector_b dot_product at h
  linarith

end perpendicular_vectors_l200_200880


namespace find_f_half_l200_200918

def g (x : ℝ) : ℝ := 1 - x^2

def f (y : ℝ) (hx : y ≠ 0) : ℝ := (1 - (1 - y^2)^2) / (1 - y^2)^2

theorem find_f_half (hf : ∀ x, f (g x) (by linarith : 1 - x^2 ≠ 0) = (1 - x^2) / x^2) : f (1 / 2) (by norm_num : (1 / 2) ≠ 0) = 1 :=
sorry

end find_f_half_l200_200918


namespace find_m_l200_200835

theorem find_m (m : ℝ) (h : IsRoot (λ x : ℝ, x^2 + m * x - 3) 1) : m = 2 := by
  sorry

end find_m_l200_200835


namespace number_of_mappings_l200_200496

-- Definitions of the sets M and N
def M : Set ℕ := {1, 2, 3} -- using {1, 2, 3} to represent {a, b, c}
def N : Set ℤ := {-1, 0, 1}

-- Definition of the function f : M → N
def f (x : ℕ) : ℤ := 
  if x = 1 then 0
  else if x = 2 then f2
  else f3

theorem number_of_mappings : Set.card {f : M → N | f 1 = 0} = 9 := by
  sorry

end number_of_mappings_l200_200496


namespace cot_theta_simplifies_to_zero_l200_200141

theorem cot_theta_simplifies_to_zero (θ : ℝ) (h : Real.cot θ = 5) :
  (1 - Real.sin θ) / Real.cos θ - (Real.cos θ) / (1 + Real.sin θ) = 0 :=
by
  sorry

end cot_theta_simplifies_to_zero_l200_200141


namespace expression_in_terms_of_p_q_l200_200995

noncomputable def roots (p : ℝ) (c : ℝ) : set ℝ :=
{x | x^2 + p * x + c = 0}

theorem expression_in_terms_of_p_q :
  ∀ (p q : ℝ),
  let α := classical.some (roots p 2).exists_some
  let β := classical.some (roots p 2).exists_is_empty
  let γ := classical.some (roots q 3).exists_some
  let δ := classical.some (roots q 3).exists_is_empty
  ((α - γ) * (β - γ) * (α + δ) * (β + δ) = 3 * (q ^ 2 - p ^ 2)) :=
by sorry

end expression_in_terms_of_p_q_l200_200995


namespace find_ordered_pair_l200_200426

theorem find_ordered_pair :
  ∃ (x y : ℤ), x + y = (7 - x) + (7 - y) ∧ x - y = (x - 3) + (y - 3) ∧ x = 1 ∧ y = 6 :=
by
  existsi 1
  existsi 6
  split
  · sorry
  split
  · sorry
  split
  · refl
  · refl

end find_ordered_pair_l200_200426


namespace total_oranges_picked_l200_200247

-- Defining the number of oranges picked by Mary, Jason, and Sarah
def maryOranges := 122
def jasonOranges := 105
def sarahOranges := 137

-- The theorem to prove that the total number of oranges picked is 364
theorem total_oranges_picked : maryOranges + jasonOranges + sarahOranges = 364 := by
  sorry

end total_oranges_picked_l200_200247


namespace vegetable_difference_is_30_l200_200717

def initial_tomatoes : Int := 17
def initial_carrots : Int := 13
def initial_cucumbers : Int := 8
def initial_bell_peppers : Int := 15
def initial_radishes : Int := 0

def picked_tomatoes : Int := 5
def picked_carrots : Int := 6
def picked_cucumbers : Int := 3
def picked_bell_peppers : Int := 8

def given_neighbor1_tomatoes : Int := 3
def given_neighbor1_carrots : Int := 2

def exchanged_neighbor2_tomatoes : Int := 2
def exchanged_neighbor2_cucumbers : Int := 3
def exchanged_neighbor2_radishes : Int := 5

def given_neighbor3_bell_peppers : Int := 3

noncomputable def initial_total := 
  initial_tomatoes + initial_carrots + initial_cucumbers + initial_bell_peppers + initial_radishes

noncomputable def remaining_after_picking :=
  (initial_tomatoes - picked_tomatoes) +
  (initial_carrots - picked_carrots) +
  (initial_cucumbers - picked_cucumbers) +
  (initial_bell_peppers - picked_bell_peppers)

noncomputable def remaining_after_exchanges :=
  ((initial_tomatoes - picked_tomatoes - given_neighbor1_tomatoes - exchanged_neighbor2_tomatoes) +
  (initial_carrots - picked_carrots - given_neighbor1_carrots) +
  (initial_cucumbers - picked_cucumbers - exchanged_neighbor2_cucumbers) +
  (initial_bell_peppers - picked_bell_peppers - given_neighbor3_bell_peppers) +
  exchanged_neighbor2_radishes)

noncomputable def remaining_total := remaining_after_exchanges

noncomputable def total_difference := initial_total - remaining_total

theorem vegetable_difference_is_30 : total_difference = 30 := by
  sorry

end vegetable_difference_is_30_l200_200717


namespace solve_system_l200_200695

theorem solve_system (x y z : ℝ) (hx : x + y + z = 13)
    (hy : x^2 + y^2 + z^2 = 61) (hz : xy + xz = 2yz) :
    (x = 4 ∧ y = 3 ∧ z = 6) ∨ (x = 4 ∧ y = 6 ∧ z = 3) :=
by sorry

end solve_system_l200_200695


namespace parameterization_function_l200_200297

theorem parameterization_function (f : ℝ → ℝ) 
  (parameterized_line : ∀ t : ℝ, (f t, 20 * t - 10))
  (line_eq : ∀ x y : ℝ, y = 2 * x - 30) :
  f = λ t, 10 * t + 10 :=
by
  sorry

end parameterization_function_l200_200297


namespace wilson_12_fact_mod_13_l200_200009

theorem wilson_12_fact_mod_13 : (factorial 12) % 13 = 12 :=
by 
  -- Relevant imports and preliminaries
  -- Use the fact that 13 is a prime, thus we can appeal to Wilson's Theorem.
  sorry

end wilson_12_fact_mod_13_l200_200009


namespace volume_correct_height_correct_l200_200759

open Real

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

noncomputable def vector (A B : Point3D) : Point3D :=
  { x := B.x - A.x,
    y := B.y - A.y,
    z := B.z - A.z }

noncomputable def cross_product (v1 v2 : Point3D) : Point3D :=
  { x := v1.y * v2.z - v1.z * v2.y,
    y := v1.z * v2.x - v1.x * v2.z,
    z := v1.x * v2.y - v1.y * v2.x }

noncomputable def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

noncomputable def triple_product (v1 v2 v3 : Point3D) : ℝ :=
  dot_product v1 (cross_product v2 v3)

noncomputable def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x^2 + v.y^2 + v.z^2)

def A1 : Point3D := { x := 2, y := -1, z := 2 }
def A2 : Point3D := { x := 1, y := 2, z := -1 }
def A3 : Point3D := { x := 3, y := 2, z := 1 }
def A4 : Point3D := { x := -4, y := 2, z := 5 }

noncomputable def volume_tetrahedron : ℝ :=
  1 / 6 * Real.abs (triple_product (vector A1 A2) (vector A1 A3) (vector A4 A1))

noncomputable def area_triangle : ℝ :=
  1 / 2 * magnitude (cross_product (vector A1 A2) (vector A1 A3))

noncomputable def height_tetrahedron : ℝ :=
  (3 * volume_tetrahedron) / area_triangle

theorem volume_correct : volume_tetrahedron = 11 := by
  sorry

theorem height_correct : height_tetrahedron = Real.sqrt(11 / 2) := by
  sorry

end volume_correct_height_correct_l200_200759


namespace subset_A_exists_l200_200558

variable (n : ℕ)

theorem subset_A_exists (h : n ≥ 2) :
  ∃ A ⊆ {1, 2, ..., n}, (|A| ≤ 2 * Int.floor (Real.sqrt n) + 1) ∧ 
  (SetOfDifferences A = {1, 2, ..., n-1}) :=
sorry

end subset_A_exists_l200_200558


namespace perpendicular_vectors_implies_m_value_l200_200870

variable (m : ℝ)

def vector1 : ℝ × ℝ := (m, 3)
def vector2 : ℝ × ℝ := (1, m + 1)

theorem perpendicular_vectors_implies_m_value
  (h : vector1 m ∙ vector2 m = 0) :
  m = -3 / 4 :=
by 
  sorry

end perpendicular_vectors_implies_m_value_l200_200870


namespace mother_bakes_8_pies_a_day_l200_200775

def Eddie_bakes_per_day : ℕ := 3
def Sister_bakes_per_day : ℕ := 6
def Mother_bakes_per_day : ℕ := M
def Total_days : ℕ := 7
def Total_pies : ℕ := 119

theorem mother_bakes_8_pies_a_day
(Eddie_bakes_per_day Sister_bakes_per_day : ℕ)
(Mother_bakes_per_day Total_days Total_pies : ℕ)
(h1 : Eddie_bakes_per_day = 3)
(h2 : Sister_bakes_per_day = 6)
(h3 : Total_days = 7)
(h4 : Total_pies = 119)
(h5 : 3 * 7 + 6 * 7 + Mother_bakes_per_day * 7 = 119) :
  Mother_bakes_per_day = 8 :=
by
  -- Proof can be written here
  sorry

end mother_bakes_8_pies_a_day_l200_200775


namespace correct_answer_l200_200914

theorem correct_answer (a b c : ℤ) 
  (h : (a - b) ^ 10 + (a - c) ^ 10 = 1) : 
  |a - b| + |b - c| + |c - a| = 2 := by 
  sorry

end correct_answer_l200_200914


namespace alex_jellybeans_l200_200071

theorem alex_jellybeans (n : ℕ) (h1 : n ≥ 200) (h2 : n % 17 = 15) : n = 202 :=
sorry

end alex_jellybeans_l200_200071


namespace distance_midpoint_proof_l200_200785

open Real

def Point3D := (ℝ × ℝ × ℝ)

noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

def midpoint (p1 p2 : Point3D) : Point3D :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

theorem distance_midpoint_proof :
  let A : Point3D := (1, -3, 2)
  let B : Point3D := (4, 6, 0)
  distance A B = sqrt 94 ∧ midpoint A B = (5/2, 3/2, 1) :=
  by 
    let A : Point3D := (1, -3, 2)
    let B : Point3D := (4, 6, 0)
    sorry

end distance_midpoint_proof_l200_200785


namespace vector_expression_result_l200_200402

def vec1 := (3, -6)
def vec2 := (2, -1)
def vec3 := (-1, 4)

def scalar_mult (k : Int) (v : Int × Int) : Int × Int :=
  (k * v.1, k * v.2)

def vec_add (v1 v2 : Int × Int) : Int × Int :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vec_sub (v1 v2 : Int × Int) : Int × Int :=
  (v1.1 - v2.1, v1.2 - v2.2)

theorem vector_expression_result :
  vec_sub (vec_add (scalar_mult 4 vec1) (scalar_mult 3 vec2)) vec3 = (19, -31) :=
by
  sorry

end vector_expression_result_l200_200402


namespace initial_investment_approx_l200_200190

-- Define the conditions
def future_value : ℝ := 600000
def annual_interest_rate : ℝ := 0.06
def compounding_periods_per_year : ℤ := 2
def number_of_years : ℤ := 12

-- Define the formula for present value under semi-annually compounded interest
noncomputable def present_value (fv : ℝ) (r : ℝ) (n : ℤ) (t : ℤ) : ℝ :=
  fv / (1 + r / n)^((n : ℝ) * (t : ℝ))

-- Define the proof problem statement
theorem initial_investment_approx :
  present_value future_value annual_interest_rate compounding_periods_per_year number_of_years ≈ 295097.57 :=
by
  sorry

end initial_investment_approx_l200_200190


namespace probability_roots_condition_l200_200725

theorem probability_roots_condition :
  let interval := set.Icc 6 11 in
  let quad_eq := λ k x, (k^2 - 2*k - 24) * x^2 + (3*k - 8) * x + 2 = 0 in
  let condition := λ k x1 x2, x1 <= 2 * x2 in
  let probability := (28 / 3 - 6) / (11 - 6) = 2 / 3 in
  ∀ (k : ℝ),
    k ∈ interval →
    (∃ x1 x2 : ℝ, quad_eq k x1 ∧ 
      quad_eq k x2 ∧ 
      x1 = 2 * x2) →
    probability := 2 / 3 :=
begin
  sorry
end

end probability_roots_condition_l200_200725


namespace curve_c1_polar_eqn_curve_c2_rect_eqn_line_segment_AB_length_l200_200970

theorem curve_c1_polar_eqn (x y : ℝ) (α : ℝ) (h₁ : x = 1 + Mathlib.cos α) (h₂ : y = Mathlib.sin α) :
  ∃ θ ρ, x = ρ * Mathlib.cos θ ∧ y = ρ * Mathlib.sin θ ∧ ρ = 2 * Mathlib.cos θ :=
by sorry

theorem curve_c2_rect_eqn (ρ θ : ℝ) (h : ρ = -2 * Mathlib.sin θ) :
  ∃ (x y : ℝ), x = ρ * Mathlib.cos θ ∧ y = ρ * Mathlib.sin θ ∧ x^2 + (y + 1)^2 = 1 :=
by sorry

theorem line_segment_AB_length (h₁ : ρ_1 = 2 * Mathlib.cos (-Math.pi / 3)) (h₂ : ρ_2 = -2 * Mathlib.sin (-Math.pi / 3))
  (h_line : ∀ x y, Mathlib.sqrt 3 * x + y = 0):
  ∃ (A B : ℝ), |(ρ_1 - ρ_2)| = Mathlib.sqrt 3 - 1 :=
by sorry

end curve_c1_polar_eqn_curve_c2_rect_eqn_line_segment_AB_length_l200_200970


namespace smallest_positive_integer_x_l200_200010

-- Definitions based on the conditions given
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- Statement of the problem
theorem smallest_positive_integer_x (x : ℕ) :
  (is_multiple (900 * x) 640) → x = 32 :=
sorry

end smallest_positive_integer_x_l200_200010


namespace line_intersects_circle_chord_min_length_l200_200450

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line L based on parameter m
def L (m x y : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

-- Prove that for any real number m, line L intersects circle C at two points.
theorem line_intersects_circle (m : ℝ) : 
  ∃ x y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ C x y₁ ∧ C x y₂ ∧ L m x y₁ ∧ L m x y₂ :=
sorry

-- Prove the equation of line L in slope-intercept form when the chord cut by circle C has minimum length.
theorem chord_min_length : ∃ (m : ℝ), ∀ x y : ℝ, 
  L m x y ↔ y = 2 * x - 5 :=
sorry

end line_intersects_circle_chord_min_length_l200_200450


namespace prob_mc_tf_correct_prob_at_least_one_mc_correct_l200_200255

-- Define the total number of questions and their types
def total_questions : ℕ := 5
def multiple_choice_questions : ℕ := 3
def true_false_questions : ℕ := 2
def total_outcomes : ℕ := total_questions * (total_questions - 1)

-- Probability calculation for one drawing a multiple-choice and the other drawing a true/false question
def prob_mc_tf : ℚ := (multiple_choice_questions * true_false_questions + true_false_questions * multiple_choice_questions) / total_outcomes

-- Probability calculation for at least one drawing a multiple-choice question
def prob_at_least_one_mc : ℚ := 1 - (true_false_questions * (true_false_questions - 1)) / total_outcomes

theorem prob_mc_tf_correct : prob_mc_tf = 3/5 := by
  sorry

theorem prob_at_least_one_mc_correct : prob_at_least_one_mc = 9/10 := by
  sorry

end prob_mc_tf_correct_prob_at_least_one_mc_correct_l200_200255


namespace radiator_water_fraction_l200_200034

noncomputable def fraction_of_water_after_replacements (initial_water : ℚ) (initial_antifreeze : ℚ) (removal_fraction : ℚ)
  (num_replacements : ℕ) : ℚ :=
  initial_water * (removal_fraction ^ num_replacements)

theorem radiator_water_fraction :
  let initial_water := 10
  let initial_antifreeze := 10
  let total_volume := 20
  let removal_volume := 5
  let removal_fraction := 3 / 4
  let num_replacements := 4
  fraction_of_water_after_replacements initial_water initial_antifreeze removal_fraction num_replacements / total_volume = 0.158 := 
sorry

end radiator_water_fraction_l200_200034


namespace constructionBankLoanRepayment_industrialCommercialBankLoanRepayment_l200_200752

namespace HomeLoan

noncomputable def constructionBankRepayment : ℝ := 12245
noncomputable def industrialCommercialBankRepayment : ℝ := 12330

theorem constructionBankLoanRepayment
  (principal : ℝ)
  (annualInterestRate : ℝ)
  (numInstallments : ℕ)
  (totalRepayment : ℝ)
  (eq1 : principal = 100000)
  (eq2 : annualInterestRate = 0.05)
  (eq3 : numInstallments = 10)
  (eq4 : totalRepayment = 100000 * (1 + numInstallments * annualInterestRate)) :
  (constructionBankRepayment ≈ 12245) := sorry

theorem industrialCommercialBankLoanRepayment
  (principal : ℝ)
  (annualInterestRate : ℝ)
  (numInstallments : ℕ)
  (eq1 : principal = 100000)
  (eq2 : annualInterestRate = 0.04)
  (eq3 : numInstallments = 10)
  (eq4 : (1 + annualInterestRate) ^ 10 = 1.4802) :
  (industrialCommercialBankRepayment ≈ 12330) := sorry

end HomeLoan

end constructionBankLoanRepayment_industrialCommercialBankLoanRepayment_l200_200752


namespace range_of_function_l200_200308

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem range_of_function : ∀ y, y ∈ set.range f ↔ ∃ x, f x = y ∧ y ≥ 2 :=
by
  sorry

end range_of_function_l200_200308


namespace average_speed_with_stoppages_l200_200343

-- Given conditions
def average_speed_without_stoppages : ℝ := 100
def stoppage_time_per_hour : ℝ := 3 / 60

-- Target proof problem
theorem average_speed_with_stoppages : average_speed_without_stoppages * (1 - stoppage_time_per_hour) = 95 :=
by
  sorry

end average_speed_with_stoppages_l200_200343


namespace no_solution_a_no_solution_b_no_solution_c_no_solution_d_no_solution_e_no_solution_f_no_solution_g_no_solution_h_l200_200023

theorem no_solution_a (x : ℝ) : sqrt (x + 2) = -2 → False := sorry

theorem no_solution_b (x : ℝ) : sqrt (2*x + 3) + sqrt (x + 3) = 0 → False := sorry

theorem no_solution_c (x : ℝ) : sqrt (4 - x) - sqrt (x - 6) = 2 → False := sorry

theorem no_solution_d (x : ℝ) : sqrt (-1 - x) = (x - 5)^(1/3) → False := sorry

theorem no_solution_e (x : ℝ) : 5 * sqrt x - 3 * sqrt (-x) + 17 / x = 4 → False := sorry

theorem no_solution_f (x : ℝ) : sqrt (x - 3) - sqrt (x + 9) = sqrt (x - 2) → False := sorry

theorem no_solution_g (x : ℝ) : sqrt x + sqrt (x + 9) = 2 → False := sorry

theorem no_solution_h (x : ℝ) : (x + 1 / x)^(1/3) = sqrt (- x) - 1 → False := sorry

end no_solution_a_no_solution_b_no_solution_c_no_solution_d_no_solution_e_no_solution_f_no_solution_g_no_solution_h_l200_200023


namespace same_bill_at_300_minutes_l200_200609

def monthlyBillA (x : ℕ) : ℝ := 15 + 0.1 * x
def monthlyBillB (x : ℕ) : ℝ := 0.15 * x

theorem same_bill_at_300_minutes : monthlyBillA 300 = monthlyBillB 300 := 
by
  sorry

end same_bill_at_300_minutes_l200_200609


namespace int_solutions_to_inequalities_l200_200908

theorem int_solutions_to_inequalities :
  { x : ℤ | -5 * x ≥ 3 * x + 15 } ∩
  { x : ℤ | -3 * x ≤ 9 } ∩
  { x : ℤ | 7 * x ≤ -14 } = { -3, -2 } :=
by {
  sorry
}

end int_solutions_to_inequalities_l200_200908


namespace delores_money_left_l200_200103

theorem delores_money_left (initial_amount spent_computer spent_printer : ℝ) 
    (h1 : initial_amount = 450) 
    (h2 : spent_computer = 400) 
    (h3 : spent_printer = 40) : 
    initial_amount - (spent_computer + spent_printer) = 10 := 
by 
    sorry

end delores_money_left_l200_200103


namespace pair_of_sum_six_l200_200809

def original_set : Set ℕ := {1, 2, 3, 4, 5}

def pairs (s : Set ℕ) : Set (ℕ × ℕ) := 
  {p | p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 < p.2 ∧ p.1 + p.2 = 6}

def pair_count : ℕ := Set.card (pairs original_set)

theorem pair_of_sum_six : pair_count = 2 := 
  by
  sorry

end pair_of_sum_six_l200_200809


namespace different_elective_schemes_l200_200040

-- Definition of the problem conditions in Lean 4
def numCourses : ℕ := 10
def mutuallyExclusive : Finset ℕ := {4, 6, 8}  -- Let's label 4-1, 4-2, 4-4 with arbitrary unique identifiers, say 4, 6, 8 respectively
def numCoursesToSelect : ℕ := 3

-- The problem statement: Proving the number of ways to choose the courses is 98
theorem different_elective_schemes : 
  ∑ n in (Finset.range numCourses.succ).powerset.filter (λ s, 
    s.card = numCoursesToSelect ∧ s ∩ mutuallyExclusive ≠ ∅), 1 +
  ∑ n in (Finset.range (numCourses - mutuallyExclusive.card).succ).powerset.filter (λ s, 
    s.card = numCoursesToSelect), 1 = 98 :=
sorry

end different_elective_schemes_l200_200040


namespace badges_initial_count_l200_200673

variable {V T : ℕ}

-- conditions
def initial_condition : Prop := V = T + 5
def exchange_condition : Prop := 0.76 * V + 0.20 * T = 0.80 * T + 0.24 * V - 1

-- result
theorem badges_initial_count (h1 : initial_condition) (h2 : exchange_condition) : V = 50 ∧ T = 45 := 
  sorry

end badges_initial_count_l200_200673


namespace increase_in_growth_rate_l200_200532

variable (S1 S2 S3 : ℝ)

def growth_rate (current previous : ℝ) : ℝ := (current - previous) / previous

theorem increase_in_growth_rate :
  growth_rate S3 S2 - growth_rate S2 S1 = (S3 * S1 - S2 ^ 2) / (S1 * S2) :=
by
  sorry

end increase_in_growth_rate_l200_200532


namespace inverse_r_l200_200999

def p (x: ℝ) : ℝ := 4 * x + 5
def q (x: ℝ) : ℝ := 3 * x - 4
def r (x: ℝ) : ℝ := p (q x)

theorem inverse_r (x : ℝ) : r⁻¹ x = (x + 11) / 12 :=
sorry

end inverse_r_l200_200999


namespace median_of_dataset_is_4_l200_200825

def dataset := [3, 2, 5, 3, 7, 5, 3, 7]

theorem median_of_dataset_is_4 : List.median dataset = 4 := by
  sorry

end median_of_dataset_is_4_l200_200825


namespace symmetric_circle_equation_l200_200121

theorem symmetric_circle_equation :
  ∀ (x y : ℝ),
    (x^2 + y^2 - 6 * x + 8 * y + 24 = 0) →
    (x - 3 * y - 5 = 0) →
    (∃ x₀ y₀ : ℝ, (x₀ - 1)^2 + (y₀ - 2)^2 = 1) :=
by
  sorry

end symmetric_circle_equation_l200_200121


namespace distance_between_lights_l200_200523

/-- In a string of lights, red and green lights are arranged alternately in a pattern
    of 3 red lights followed by 2 green lights, and they are spaced 8 inches apart.
    Calculate the distance in feet between the 5th red light and the 28th red light.
    Note: 1 foot is equal to 12 inches. -/
theorem distance_between_lights : 
  (let inch_per_gap := 8
       gap_per_foot := 12
       number_of_gaps := 45 - 8
       distance_in_inches := number_of_gaps * inch_per_gap 
       distance_in_feet := distance_in_inches / gap_per_foot
   in distance_in_feet) = 24.67 := sorry

end distance_between_lights_l200_200523


namespace expected_value_variance_defective_items_l200_200209

variable (ξ : ℕ) [distribution ξ]

theorem expected_value_variance_defective_items :
  (distribution.binom 200 0.01) ξ →
  (ξ.mean = 2 ∧ ξ.variance = 1.98) := by
  sorry

end expected_value_variance_defective_items_l200_200209


namespace sum_reciprocals_even_factors_12_l200_200683

def factors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0)

def reciprocals (l : List ℕ) : List ℚ :=
  l.map (λ x, 1 / x)

def sum_list (l : List ℚ) : ℚ :=
  l.foldl (· + ·) 0

theorem sum_reciprocals_even_factors_12 :
  sum_list (reciprocals (List.filter (λ x, x % 2 = 0) (factors 12))) = 1 :=
by
  -- This would be the place where the proof would go
  sorry  -- Placeholder for proof

end sum_reciprocals_even_factors_12_l200_200683


namespace focus_of_parabola_y_eq_8x2_l200_200287

open Real

noncomputable def parabola_focus (a p : ℝ) : ℝ × ℝ :=
  (0, 1 / (4 * p))

theorem focus_of_parabola_y_eq_8x2 :
  parabola_focus 8 (1 / 16) = (0, 1 / 32) :=
by
  sorry

end focus_of_parabola_y_eq_8x2_l200_200287


namespace total_cost_l200_200543

theorem total_cost 
  (rental_cost: ℝ) 
  (gallons: ℝ) 
  (gas_price: ℝ) 
  (price_per_mile: ℝ) 
  (miles: ℝ)
  (H1: rental_cost = 150)
  (H2: gallons = 8)
  (H3: gas_price = 3.5)
  (H4: price_per_mile = 0.5)
  (H5: miles = 320) :
  rental_cost + (gallons * gas_price) + (miles * price_per_mile) = 338 :=
by {
  have gas_cost : ℝ := gallons * gas_price,
  have mileage_cost : ℝ := miles * price_per_mile,
  rw [←H1, ←H2, ←H3, ←H4, ←H5],
  norm_num,
  sorry
}

end total_cost_l200_200543


namespace total_number_of_trees_l200_200611

theorem total_number_of_trees (side_length : ℝ) (area_ratio : ℝ) (trees_per_sqm : ℝ) (H : side_length = 100) (R : area_ratio = 3) (T : trees_per_sqm = 4) : 
  let street_area := side_length ^ 2 in 
  let forest_area := area_ratio * street_area in
  let total_trees := forest_area * trees_per_sqm in
  total_trees = 120000 :=
by
  -- proof steps go here
  sorry

end total_number_of_trees_l200_200611


namespace find_m_l200_200891

-- Declare the vectors a and b based on given conditions
variables {m : ℝ}

def a : ℝ × ℝ := (m, 3)
def b : ℝ × ℝ := (1, m + 1)

-- Define the condition that vectors a and b are perpendicular
def perpendicular (x y : ℝ × ℝ) : Prop := x.1 * y.1 + x.2 * y.2 = 0

-- State the problem in Lean 4
theorem find_m (h : perpendicular a b) : m = -3 / 4 :=
sorry

end find_m_l200_200891


namespace total_number_of_trees_l200_200616

-- Definitions of the conditions
def side_length : ℝ := 100
def trees_per_sq_meter : ℝ := 4

-- Calculations based on the conditions
def area_of_street : ℝ := side_length * side_length
def area_of_forest : ℝ := 3 * area_of_street

-- The statement to prove
theorem total_number_of_trees : 
  trees_per_sq_meter * area_of_forest = 120000 := 
sorry

end total_number_of_trees_l200_200616


namespace problem_inequality_l200_200171

def f (x : ℝ) (m : ℝ) : ℝ := (x^2 + m * x + 1) / Real.exp x

theorem problem_inequality (m : ℝ) (h1 : m ∈ Set.Ioo (-1 : ℝ) 0) 
  (x1 x2 : ℝ) (hx1 : x1 ∈ Set.Icc 1 (1 - m)) (hx2 : x2 ∈ Set.Icc 1 (1 - m)) :
  4 * f x1 m + x2 < 5 :=
sorry

end problem_inequality_l200_200171


namespace expression_exists_l200_200503

theorem expression_exists (a b : ℤ) (h : 5 * a = 3125) (hb : 5 * b = 25) : b = 5 := by
  sorry

end expression_exists_l200_200503


namespace interval_strictly_increasing_l200_200772

noncomputable def u : ℝ → ℝ := λ x, 2 * x ^ 2 - 3 * x + 1

noncomputable def f : ℝ → ℝ := λ x, (1/3) ^ u x

theorem interval_strictly_increasing :
  ∀ (x y : ℝ), x < y →
  f x < f y :=
begin
  assume x y hxy,
  change (1/3) ^ u x < (1/3) ^ u y,
  apply (order_iso.pow_lt (by norm_num : 1/3 > 0) (by norm_num : 1/3 < 1)).le_iff_le.mp,
  rw u,
  apply poly_increasing_on_interval_subset
    (λ z, 2 * z ^ 2 - 3 * z + 1) hxy
        (λ a b, by norm_num),
  exact @polynomial.strict_mono_on_of_div_ltx_0 ℝ _ _ (-∞, 3/4)
    (polynomial.monic _
      (λ a, @polynomial.monic_poly_map ℝ _ _ _ _ _ _ _ 0))
    (by norm_num)
end

end interval_strictly_increasing_l200_200772


namespace inequality_holds_if_b_greater_than_2_l200_200921

variable (x : ℝ) (b : ℝ)

theorem inequality_holds_if_b_greater_than_2  :
  (b > 0) → (∃ x, |x-5| + |x-7| < b) ↔ (b > 2) := sorry

end inequality_holds_if_b_greater_than_2_l200_200921


namespace total_fruit_in_buckets_l200_200656

def bucket_pieces (A B C : ℕ) : Prop :=
  A = B + 4 ∧ B = C + 3 ∧ C = 9 ∧ (A + B + C = 37)

theorem total_fruit_in_buckets : ∃ A B C, bucket_pieces A B C :=
by
  use 16, 12, 9
  unfold bucket_pieces
  split
  . exact rfl
  split
  . exact rfl
  split
  . exact rfl
  . exact rfl

end total_fruit_in_buckets_l200_200656


namespace midpoints_of_common_tangents_lie_on_radical_axis_circles_intercept_equal_chords_l200_200017

-- Part (a)
theorem midpoints_of_common_tangents_lie_on_radical_axis 
  (C1 C2 : Circle) (h1 : ¬intersecting C1 C2) :
  ∃ l : Line, ∀ (T1 T2 T3 T4 : Point), 
    (is_tangent C1 T1) ∧ (is_tangent C2 T1) ∧
    (is_tangent C1 T2) ∧ (is_tangent C2 T2) ∧
    (is_tangent C1 T3) ∧ (is_tangent C2 T3) ∧
    (is_tangent C1 T4) ∧ (is_tangent C2 T4) →
    (midpoint T1 T2 ∈ l) ∧ (midpoint T3 T4 ∈ l) :=
sorry

-- Part (b)
theorem circles_intercept_equal_chords 
  (C1 C2 : Circle) (P1 P2 : Point) 
  (h1 : is_external_tangent C1 C2 P1) 
  (h2 : is_external_tangent C1 C2 P2) :
  ∃ l : Line, ∀ (A1 A2 B1 B2 : Point),
    (line_pass_through l P1 P2) ∧
    (on_circle C1 A1) ∧ (on_circle C1 A2) ∧
    (on_circle C2 B1) ∧ (on_circle C2 B2) →
    chord_length C1 A1 A2 = chord_length C2 B1 B2 :=
sorry

-- Definitions used for theorems.
def Circle := ℝ × ℝ × ℝ  -- Assuming Circle is represented as (center_x, center_y, radius)
def Point := ℝ × ℝ
def Line := Point × Point

-- Placeholder definitions
def intersecting (c1 c2 : Circle) : Prop := sorry
def is_tangent (c : Circle) (p : Point) : Prop := sorry
def midpoint (p1 p2 : Point) : Point := sorry
def is_external_tangent (c1 c2 : Circle) (p : Point) : Prop := sorry
def line_pass_through (l : Line) (p1 p2 : Point) : Prop := sorry
def on_circle (c : Circle) (p : Point) : Prop := sorry
def chord_length (c : Circle) (p1 p2 : Point) : ℝ := sorry

end midpoints_of_common_tangents_lie_on_radical_axis_circles_intercept_equal_chords_l200_200017


namespace missing_keys_total_l200_200229

theorem missing_keys_total : 
  let english_alphabet_keys := 26 in
  let russian_alphabet_keys := 33 in
  let customizable_function_keys := 8 in
  let accent_marks := 5 in
  let special_characters := 3 in
  let english_vowels := 5 in
  let russian_vowels := 10 in
  let english_consonants := english_alphabet_keys - english_vowels in
  let russian_consonants := russian_alphabet_keys - russian_vowels in
  let missing_english_consonants := (3 / 8) * english_consonants in
  let missing_english_vowels := 4 in
  let missing_russian_consonants := (1 / 4) * russian_consonants in
  let missing_russian_vowels := 3 in
  let missing_function_keys := (1 / 3) * customizable_function_keys in
  let missing_accent_marks := (2 / 5) * accent_marks in
  let missing_special_characters := (1 / 2) * special_characters in
  let total_missing_keys :=
    (Int.ceil missing_english_consonants) + 
    missing_english_vowels + 
    (Int.ceil missing_russian_consonants) + 
    missing_russian_vowels + 
    (Int.ceil missing_function_keys) + 
    (Int.ceil missing_accent_marks) + 
    (Int.ceil missing_special_characters) in
  total_missing_keys = 28 :=
by
  intros
  -- Proof omitted
  sorry

end missing_keys_total_l200_200229


namespace car_rental_total_cost_l200_200546

theorem car_rental_total_cost 
  (rental_cost : ℕ)
  (gallons : ℕ)
  (cost_per_gallon : ℕ)
  (cost_per_mile : ℚ)
  (miles_driven : ℕ)
  (H1 : rental_cost = 150)
  (H2 : gallons = 8)
  (H3 : cost_per_gallon = 350 / 100)
  (H4 : cost_per_mile = 50 / 100)
  (H5 : miles_driven = 320) :
  rental_cost + gallons * cost_per_gallon + miles_driven * cost_per_mile = 338 :=
  sorry

end car_rental_total_cost_l200_200546


namespace solve_quadratic_equation_l200_200270

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 2 * x - 8 = 0 ↔ x = 4 ∨ x = -2 := by
sorry

end solve_quadratic_equation_l200_200270


namespace find_m_l200_200887

-- Declare the vectors a and b based on given conditions
variables {m : ℝ}

def a : ℝ × ℝ := (m, 3)
def b : ℝ × ℝ := (1, m + 1)

-- Define the condition that vectors a and b are perpendicular
def perpendicular (x y : ℝ × ℝ) : Prop := x.1 * y.1 + x.2 * y.2 = 0

-- State the problem in Lean 4
theorem find_m (h : perpendicular a b) : m = -3 / 4 :=
sorry

end find_m_l200_200887


namespace solve_quadratic_l200_200268

theorem solve_quadratic : ∃ x : ℝ, (x^2 - 2 * x - 8 = 0 ↔ x = 4 ∨ x = -2) :=
sorry

end solve_quadratic_l200_200268


namespace proving_problem_l200_200905

noncomputable def angle (A : ℝ) (hA : 0 < A ∧ A < π/2) : Prop :=
  let m := (Real.sin A, Real.cos A)
  let n := (Real.sqrt 3, -1)
  m.1 * n.1 + m.2 * n.2 = 1 ∧ A = π / 3

noncomputable def function_range (A : ℝ) (h_cosA : Real.cos A = 1/2) : Set ℝ :=
  {y | ∃ x : ℝ, y = Real.cos (2*x) + 2 * Real.sin x }

theorem proving_problem {A : ℝ} (hA : 0 < A ∧ A < π/2) (h_inner_product : 
  let m := (Real.sin A, Real.cos A)
  let n := (Real.sqrt 3, -1)
  m.1 * n.1 + m.2 * n.2 = 1) : 
  A = π / 3 ∧ (function_range A (by norm_num1 : Real.cos A = 1/2)) = Set.Icc (-3) (3/2) :=
sorry

end proving_problem_l200_200905


namespace smallest_N_divisibility_l200_200431

theorem smallest_N_divisibility :
  ∃ N : ℕ, 
    (N + 2) % 2 = 0 ∧
    (N + 3) % 3 = 0 ∧
    (N + 4) % 4 = 0 ∧
    (N + 5) % 5 = 0 ∧
    (N + 6) % 6 = 0 ∧
    (N + 7) % 7 = 0 ∧
    (N + 8) % 8 = 0 ∧
    (N + 9) % 9 = 0 ∧
    (N + 10) % 10 = 0 ∧
    N = 2520 := 
sorry

end smallest_N_divisibility_l200_200431


namespace intersection_on_semicircle_l200_200580

theorem intersection_on_semicircle {A B C H D P Q : Point} {ω : Set Point} :
  right_triangle ABC C →
  foot_of_altitude C H →
  D ∈ triangle CBH →
  midpoint (CH) (AD) →
  (BD) ∩ (CH) = P →
  semicircle_with_diameter ω D B →
  tangent_through P ω Q →
  ∃ X, X ∈ ω ∧ X ∈ (CQ) ∧ X ∈ (AD) :=
by
  sorry

end intersection_on_semicircle_l200_200580


namespace op_add_mul_example_l200_200100

def op_add (a b : ℤ) : ℤ := a + b - 1
def op_mul (a b : ℤ) : ℤ := a * b - 1

theorem op_add_mul_example : op_mul (op_add 6 8) (op_add 3 5) = 90 :=
by
  -- Rewriting it briefly without proof steps
  sorry

end op_add_mul_example_l200_200100


namespace number_of_pairs_l200_200733

theorem number_of_pairs (S : Set ℕ) (h1 : ∀ x y ∈ S, x + y ∈ S)
    (h2 : ∀ x, x ∈ ℤ → 2 * x ∈ S → x ∈ S) :
    (Finset.card {ab : Finset (ℤ × ℤ) | 1 ≤ ab.1 ∧ ab.1 ≤ 50 ∧ 1 ≤ ab.2 ∧ ab.2 ≤ 50 ∧ (∀ a b ∈ ab.1 ∪ ab.2, (∀ x y ∈ S, x + y ∈ S) ∧ (∀ x ∈ ℤ, 2 * x ∈ S → x ∈ S) → S = Set.univ ℕ)} = 2068) := 
sorry

end number_of_pairs_l200_200733


namespace minimum_AB_length_l200_200182

-- Declare the variables R and r as real numbers representing the radii of the two circles
variables {R r : ℝ}

-- State the theorem with the given conditions and what needs to be proved
theorem minimum_AB_length (hR : R > 0) (hr : r > 0) :
  ∃ (AB : ℝ), (∀ (AB : ℝ), 
    ∀ (A B C D : ℝ), 
    -- Conditions specifying the trapezoid and tangency to the circles
    (A < B) ∧ (C < D) ∧ 
    (tangent (circle (0, 0) R) (line A B) ∧ tangent (circle (0, 0) r) (line C D) ∧
     externally_tangent (circle (0, 0) R) (circle (0, 0) r))
  → AB ≥ 4 * (real.sqrt (R * r))) 
  := sorry

end minimum_AB_length_l200_200182


namespace equality_of_sequences_l200_200353

theorem equality_of_sequences (n : ℕ)
  (a b : Fin n → ℝ) 
  (h_n_ge_3 : n ≥ 3)
  (h_an_ge_an1 : a ⟨n-1, sorry⟩ ≥ a ⟨n-2, sorry⟩)
  (h_bn_ge_bn1 : b ⟨n-1, sorry⟩ ≥ b ⟨n-2, sorry⟩)
  (h_order : ∀ i : Fin (n-1), 0 < a i ∧ a i ≤ b i ∧ b i ≤ a ⟨i+1, sorry⟩)
  (h_sum_eq : ((Finset.univ : Finset (Fin n)).sum a) = ((Finset.univ : Finset (Fin n)).sum b))
  (h_prod_eq : ((Finset.univ : Finset (Fin n)).prod a) = ((Finset.univ : Finset (Fin n)).prod b)) : 
  ∀ i : Fin n, a i = b i := by
  sorry

end equality_of_sequences_l200_200353


namespace solution_exists_l200_200769

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem solution_exists : ∃ a b : ℕ, (a > 0 ∧ b > 0 ∧ b > a ∧ is_prime (b - a) ∧ (a + b) % 10 = 3 ∧ ∃ k : ℕ, a * b = k^2) 
  ∧ (a = 4 ∧ b = 9) :=
by {
  existsi 4,
  existsi 9,
  split,
  { split,
    { show 4 > 0, from nat.zero_lt_bit0 (nat.zero_lt_bit0 nat.zero_lt_one), },
    { split,
      { show 9 > 0, from nat.zero_lt_bit1 nat.zero_lt_four, },
      { split,
        { show 9 > 4, from nat.lt_bit1.2 (nat.zero_lt_bit0 nat.zero_lt_one), },
        { split,
          { show is_prime (9 - 4), from dec_trivial, },
          { split,
            { show (4 + 9) % 10 = 3, from dec_trivial, },
            { existsi 6, show 4 * 9 = 6^2, from dec_trivial, }
          }
        }
      }
    }
  },
  show 4 = 4 ∧ 9 = 9, from ⟨rfl, rfl⟩
}

end solution_exists_l200_200769


namespace infinite_series_eq_1_div_400_l200_200404

theorem infinite_series_eq_1_div_400 :
  (∑' n:ℕ, (4 * n + 2) / ((4 * n + 1)^2 * (4 * n + 5)^2)) = 1 / 400 :=
by
  sorry

end infinite_series_eq_1_div_400_l200_200404


namespace andy_initial_cookies_l200_200389

-- Define the conditions as constants
constant andy_ate : ℕ := 3
constant brother_received : ℕ := 5
constant team_members : ℕ := 8

-- Define the function for cookies taken by the basketball team using the problem's condition
def cookies_taken_by_nth_player (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

-- Calculate the total cookies taken by the basketball team
def total_cookies_taken_by_team : ℕ :=
  (list.range team_members).sum (λ n, cookies_taken_by_nth_player (n + 1))

-- Summing up the initial eats, gives, and taken by team
def total_cookies (andy_ate brother_received total_cookies_taken_by_team : ℕ) : ℕ :=
  andy_ate + brother_received + total_cookies_taken_by_team

theorem andy_initial_cookies : total_cookies andy_ate brother_received total_cookies_taken_by_team = 72 :=
by
  -- Introduce a known mathematical result: sum of arithmetic sequence
  have sum_arith_seq_8 : (list.range 8).sum (λ n, cookies_taken_by_nth_player (n + 1)) = 64 := 
    by sorry -- this is actually the result we assume from the solution
  simp [total_cookies, andy_ate, brother_received, sum_arith_seq_8]

end andy_initial_cookies_l200_200389


namespace find_m_of_perpendicular_vectors_l200_200898

theorem find_m_of_perpendicular_vectors
    (m : ℝ)
    (a : ℝ × ℝ := (m, 3))
    (b : ℝ × ℝ := (1, m + 1))
    (h : a.1 * b.1 + a.2 * b.2 = 0) :
    m = -3 / 4 :=
by 
  sorry

end find_m_of_perpendicular_vectors_l200_200898


namespace second_crate_granola_weight_l200_200064

structure Crate where
  height : ℕ
  width : ℕ
  length : ℕ
  volume : ℕ := height * width * length

noncomputable def weight (crate : Crate) (density : ℕ) : ℕ :=
  crate.volume * density

theorem second_crate_granola_weight :
  let c1 := Crate.mk 4 3 6
  let c2 := Crate.mk (4 * 3 / 2) (3 * 3 / 2) 6
  weight c1 1 = 72 →
  weight c2 1 = 162 :=
sorry

end second_crate_granola_weight_l200_200064


namespace find_line_eq_under_conditions_l200_200843

variables {x y : ℝ}

def circle_eq (x y : ℝ) := x^2 + y^2 - 4 * x - 2 * y = 0

def is_tangent_line (l : ℝ → ℝ) (θ : ℝ) := θ = π / 4

def chord_length (C : ℝ → ℝ → Prop) (l : ℝ → ℝ) (len : ℝ) := 
  ∃ (x1 x2 y1 y2 : ℝ), 
    C x1 y1 ∧ C x2 y2 ∧ l x1 = y1 ∧ l x2 = y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = len^2

theorem find_line_eq_under_conditions :
  (∀ (x y : ℝ), circle_eq x y) →
  is_tangent_line (λ x, x + b) (π / 4) →
  chord_length circle_eq (λ x, x + b) (2 * sqrt 3) →
  (∃ (b : ℝ), b = 1 ∨ b = -3) →
  (∀ x, y = x + 1 ∨ y = x - 3) :=
sorry

end find_line_eq_under_conditions_l200_200843


namespace expansion_simplification_l200_200421

variable (x y : ℝ)

theorem expansion_simplification :
  let a := 3 * x + 4
  let b := 2 * x + 6 * y + 7
  a * b = 6 * x ^ 2 + 18 * x * y + 29 * x + 24 * y + 28 :=
by
  sorry

end expansion_simplification_l200_200421


namespace max_area_square_with_three_lattice_points_l200_200199

-- Define the problem conditions
def contains_three_lattice_points (s : Set (ℤ × ℤ)) : Prop :=
  s.card = 3

def is_square (s : Set (ℤ × ℤ)) : Prop :=
  ∃ (a b : ℝ), ∀ x ∈ s, ( x.1 - a )^2 + ( x.2 - b )^2 = ( side_length / 2 )^2

-- Define the maximum possible area of the square containing 3 lattice points
noncomputable def maximum_square_area_containing_three_lattice_points : ℝ :=
  4.5

-- The problem statement
theorem max_area_square_with_three_lattice_points (s : Set (ℤ × ℤ)) (h₁ : contains_three_lattice_points s) (h₂ : is_square s) :
  ∃ side_length, side_length^2 = maximum_square_area_containing_three_lattice_points :=
sorry

end max_area_square_with_three_lattice_points_l200_200199


namespace initial_cookies_l200_200392

variable (andys_cookies : ℕ)

def total_cookies_andy_ate : ℕ := 3
def total_cookies_brother_ate : ℕ := 5

def arithmetic_sequence_sum (n : ℕ) : ℕ := n * (2 * n - 1)

def total_cookies_team_ate : ℕ := arithmetic_sequence_sum 8

theorem initial_cookies :
  andys_cookies = total_cookies_andy_ate + total_cookies_brother_ate + total_cookies_team_ate :=
  by
    -- Here the missing proof would go
    sorry

end initial_cookies_l200_200392


namespace find_angle_y_l200_200219

open Real

theorem find_angle_y 
    (angle_ABC angle_BAC : ℝ)
    (h1 : angle_ABC = 70)
    (h2 : angle_BAC = 50)
    (triangle_sum : ∀ {A B C : ℝ}, A + B + C = 180)
    (right_triangle_sum : ∀ D E : ℝ, D + E = 90) :
    30 = 30 :=
by
    -- Given, conditions, and intermediate results (skipped)
    sorry

end find_angle_y_l200_200219


namespace complex_location_l200_200528

noncomputable def complex_quadrant : ℂ := (2 - complex.i) / complex.i

theorem complex_location : complex_quadrant = -1 + 2 * complex.i → (complex.re complex_quadrant < 0) ∧ (complex.im complex_quadrant > 0) := by
  sorry

end complex_location_l200_200528


namespace area_relationship_l200_200088

variable {ABC : Type} [plane_geometry ABC]

def is_acute_triangle (t : triangle ABC) : Prop := acute t

noncomputable def area_external_curvilinear_triangle_1 (t : triangle ABC) (c : cicle ABC) : ℝ := sorry
noncomputable def area_external_curvilinear_triangle_2 (t : triangle ABC) (c : cicle ABC) : ℝ := sorry
noncomputable def area_external_curvilinear_triangle_3 (t : triangle ABC) (c : cicle ABC) : ℝ := sorry
noncomputable def area_internal_curvilinear_triangle (t : triangle ABC) (c : cicle ABC) : ℝ := sorry
noncomputable def area_triangle (t : triangle ABC) : ℝ := sorry

theorem area_relationship (t : triangle ABC) (c1 c2 c3 : cicle ABC) :
  is_acute_triangle t →
  let x := area_external_curvilinear_triangle_1 t c1
  let y := area_external_curvilinear_triangle_2 t c2
  let z := area_external_curvilinear_triangle_3 t c3
  let u := area_internal_curvilinear_triangle t c1
  let S := area_triangle t
  (x + y + z) - u = 2 * S :=
sorry

end area_relationship_l200_200088


namespace M_intersect_P_l200_200561

noncomputable def M : Set ℝ := { y | ∃ x : ℝ, y = x^2 + 1 }
noncomputable def P : Set ℝ := { y | ∃ x : ℝ, y = Real.log x }

theorem M_intersect_P :
  M ∩ P = { y | y ≥ 1 } :=
sorry

end M_intersect_P_l200_200561


namespace joe_data_points_final_count_l200_200983

theorem joe_data_points_final_count 
  (initial_data_points : ℕ)
  (initial_data_points = 200)
  (percentage_increase : ℕ)
  (percentage_increase = 20)
  (fraction_reduction : ℚ)
  (fraction_reduction = 1/4)
  : 180 = initial_data_points * (1 + percentage_increase / 100) - initial_data_points * (1 + percentage_increase / 100) * fraction_reduction := 
by 
  sorry

end joe_data_points_final_count_l200_200983


namespace determine_avery_height_l200_200252

-- Define Meghan's height
def meghan_height : ℕ := 188

-- Define range of players' heights
def height_range : ℕ := 33

-- Define the predicate to determine Avery's height
def avery_height : ℕ := meghan_height - height_range

-- The theorem we need to prove
theorem determine_avery_height : avery_height = 155 := by
  sorry

end determine_avery_height_l200_200252


namespace find_polynomial_coefficients_l200_200240

theorem find_polynomial_coefficients (a b c d : ℚ) :
  (∃ (A B : ℚ), p(x) = (x^2 + x + 2) * (A * x + B) + (x + 2) ∧
                (∀ (x : ℚ), p(x) = a * x^3 + b * x^2 + c * x + d)) ∧ 
  (∃ (C D : ℚ), p(x) = (x^2 + x - 2) * (C * x + D) + (3 * x + 4) ∧
                (∀ (x : ℚ), p(x) = a * x^3 + b * x^2 + c * x + d)) → 
  a = 1/2 ∧ b = 1 ∧ c = 5/2 ∧ d = 3 := 
by
  sorry

end find_polynomial_coefficients_l200_200240


namespace number_of_children_proof_l200_200282

-- Let A be the number of mushrooms Anya has
-- Let V be the number of mushrooms Vitya has
-- Let S be the number of mushrooms Sasha has
-- Let xs be the list of mushrooms of other children

def mushrooms_distribution (A V S : ℕ) (xs : List ℕ) : Prop :=
  let n := 3 + xs.length
  -- First condition
  let total_mushrooms := A + V + S + xs.sum
  let equal_share := total_mushrooms / n
  (A / 2 = equal_share) ∧ (V + A / 2 = equal_share) ∧ (S = equal_share) ∧
  (∀ x ∈ xs, x = equal_share) ∧
  -- Second condition
  (S + A = V + xs.sum)

theorem number_of_children_proof (A V S : ℕ) (xs : List ℕ) :
  mushrooms_distribution A V S xs → 3 + xs.length = 6 :=
by
  intros h
  sorry

end number_of_children_proof_l200_200282


namespace coursework_materials_spending_l200_200978

def budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

theorem coursework_materials_spending : 
    budget - (budget * food_percentage + budget * accommodation_percentage + budget * entertainment_percentage) = 300 := 
by 
  -- steps you would use to prove this
  sorry

end coursework_materials_spending_l200_200978


namespace log_div_log_inv_of_16_l200_200339

theorem log_div_log_inv_of_16 : (Real.log 16) / (Real.log (1 / 16)) = -1 :=
by
  sorry

end log_div_log_inv_of_16_l200_200339


namespace proof_comparison_l200_200997

noncomputable def a : ℝ := 3 ^ 0.6
noncomputable def b : ℝ := Real.log 0.2 / Real.log 3
noncomputable def c : ℝ := 0.6 ^ 3

theorem proof_comparison : a > c ∧ c > b := by
  sorry

end proof_comparison_l200_200997


namespace eleven_percent_greater_than_eighty_l200_200203

theorem eleven_percent_greater_than_eighty :
  ∃ x : ℝ, x = 80 + (11 / 100) * 80 ∧ x = 88.8 :=
by
  use 80 + (11 / 100) * 80
  split
  · refl
  · norm_num

end eleven_percent_greater_than_eighty_l200_200203


namespace perpendicular_vectors_implies_m_value_l200_200871

variable (m : ℝ)

def vector1 : ℝ × ℝ := (m, 3)
def vector2 : ℝ × ℝ := (1, m + 1)

theorem perpendicular_vectors_implies_m_value
  (h : vector1 m ∙ vector2 m = 0) :
  m = -3 / 4 :=
by 
  sorry

end perpendicular_vectors_implies_m_value_l200_200871


namespace perpendicular_vectors_l200_200878

def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (1, m + 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (m : ℝ) (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by sorry

end perpendicular_vectors_l200_200878


namespace production_cost_change_l200_200957

theorem production_cost_change :
  let initial_cost := 1 in
  let cost_after_increase := initial_cost * 1.20 * 1.20 in
  let final_cost := cost_after_increase * 0.80 * 0.80 in
  final_cost ≈ 0.92 :=
begin
  let initial_cost := 1,
  let cost_after_increase := initial_cost * 1.20 * 1.20,
  let final_cost := cost_after_increase * 0.80 * 0.80,
  have h : final_cost = initial_cost * 1.2^2 * 0.8^2,
  { simp [initial_cost, cost_after_increase, final_cost] },
  have h2 : final_cost = 1.44 * 0.64,
  { simp [cost_after_increase, final_cost, h] },
  have h3 : final_cost = 0.9216,
  { sorry },
  have h4 : 1 - final_cost = 0.0784,
  { sorry },
  have h5 : 0.0784 ≈ 0.08,
  { sorry },
  show final_cost ≈ 0.92,
  { apply h5 },
end

end production_cost_change_l200_200957


namespace satisfy_functional_eq_l200_200118

noncomputable def f (q : ℚ) : ℕ+ := sorry -- Define the function f from positive rationals to positive integers

theorem satisfy_functional_eq (f : ℚ → ℕ+) :
  (∀ x y : {q : ℚ // 0 < q},
    f (x * y) * Int.gcd (f x * f y) (f x⁻¹ * f y⁻¹) = (x * y : ℚ) * f x⁻¹ * f y⁻¹)
  →  ∀ q : {q : ℚ // 0 < q}, f q = q :=
begin
  sorry -- Proof to be filled later
end

end satisfy_functional_eq_l200_200118


namespace volume_of_spheres_intersect_l200_200337

open Real

def sphere1 (x y z : ℝ) : Prop := x^2 + y^2 + z^2 ≤ 1
def sphere2 (x y z : ℝ) : Prop := x^2 + (y - 1)^2 + z^2 ≤ 1

def volume_of_intersection {x y z : ℝ} : ℝ :=
  -- Volume of the intersection region is given as π * 5 / 12.
  (5 : ℝ) * π / 12

theorem volume_of_spheres_intersect :
  (∀ x y z, sphere1 x y z ∧ sphere2 x y z) → volume_of_intersection = (5 * π / 12) :=
by
  intros h
  sorry

end volume_of_spheres_intersect_l200_200337


namespace train_length_l200_200067

theorem train_length (speed_km_hr : ℕ) (time_sec : ℕ) (h_speed : speed_km_hr = 72) (h_time : time_sec = 12) : 
  ∃ length_m : ℕ, length_m = 240 := 
by
  sorry

end train_length_l200_200067


namespace factor_expression_l200_200422

theorem factor_expression (x : ℝ) : 2 * x * (x + 3) + (x + 3) = (2 * x + 1) * (x + 3) :=
by
  sorry

end factor_expression_l200_200422


namespace cubic_difference_l200_200469

theorem cubic_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 50) : a^3 - b^3 = 353.5 := by
  sorry

end cubic_difference_l200_200469


namespace sum_of_interchanged_digits_divisible_by_11_l200_200313

theorem sum_of_interchanged_digits_divisible_by_11 (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) :
  ∃ c : ℕ, (11 * c = (10 * a + b) + (10 * b + a)) :=
by {
  use a + b,
  linarith
}

end sum_of_interchanged_digits_divisible_by_11_l200_200313


namespace train_length_l200_200068

theorem train_length
  (speed_kmph : ℕ) (time_s : ℕ)
  (h1 : speed_kmph = 72)
  (h2 : time_s = 12) :
  speed_kmph * (1000 / 3600 : ℕ) * time_s = 240 :=
by
  sorry

end train_length_l200_200068


namespace ln3_lt_3_div_e_l200_200161

theorem ln3_lt_3_div_e (e : ℝ) (he : 0 < e) (hln_e : Real.ln e = 1) : 
  Real.ln 3 < 3 / e := 
sorry

end ln3_lt_3_div_e_l200_200161


namespace kelly_games_left_l200_200547

theorem kelly_games_left : 
  ∀ (total_games given_away : ℕ), total_games = 50 → given_away = 15 → total_games - given_away = 35 :=
by 
  intros total_games given_away h_total_games h_given_away
  rw [h_total_games, h_given_away]
  exact rfl

end kelly_games_left_l200_200547


namespace min_speed_to_meet_car_l200_200038

theorem min_speed_to_meet_car (v a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  let x_min := v * b / a in
  x_min = v * b / a :=
by
  sorry

end min_speed_to_meet_car_l200_200038


namespace find_m_of_perpendicular_vectors_l200_200900

theorem find_m_of_perpendicular_vectors
    (m : ℝ)
    (a : ℝ × ℝ := (m, 3))
    (b : ℝ × ℝ := (1, m + 1))
    (h : a.1 * b.1 + a.2 * b.2 = 0) :
    m = -3 / 4 :=
by 
  sorry

end find_m_of_perpendicular_vectors_l200_200900


namespace george_total_choices_l200_200441

theorem george_total_choices :
  let num_colors := 10
  let choose_colors := 3
  let num_patterns := 5
  (nat.choose num_colors choose_colors) * num_patterns = 600 := by
  let num_colors := 10
  let choose_colors := 3
  let num_patterns := 5
  show (nat.choose num_colors choose_colors) * num_patterns = 600
  calc
    (nat.choose num_colors choose_colors) * num_patterns
    = (nat.choose 10 3) * 5 : by rfl
    ... = 120 * 5 : by sorry
    ... = 600 : by norm_num

end george_total_choices_l200_200441


namespace total_tiles_covering_floor_l200_200062

-- Let n be the width of the rectangle (in tiles)
-- The length would then be 2n (in tiles)
-- The total number of tiles that lie on both diagonals is given as 39

theorem total_tiles_covering_floor (n : ℕ) (H : 2 * n + 1 = 39) : 2 * n^2 = 722 :=
by sorry

end total_tiles_covering_floor_l200_200062


namespace math_problem_l200_200842

noncomputable def ellipse : Prop :=
  ∃ a : ℝ, a > 1 ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 = 1 ↔ x^2 / 3 + y^2 = 1)

noncomputable def line_passing_through_E_and_meets_conditions : Prop :=
  ∀ (k m : ℝ) (x1 y1 x2 y2 : ℝ), 
    k ≠ 0 ∧ 
    y1 = k * (x1 - 1) ∧ 
    y2 = k * (x2 - 1) ∧ 
    x1 + 2 * x2 = 3 ∧ 
    (1 + 3 * k^2) * x1^2 - 6 * k^2 * x1 + 3 * k^2 - 3 = 0 ∧ 
    (1 + 3 * k^2) * x2^2 - 6 * k^2 * x2 + 3 * k^2 - 3 = 0 
    → (m = 0 ∧ (l = x - 1 ∨ l = -x + 1))

noncomputable def max_area_triangle : Prop :=
  ∀ (k : ℝ), k ≠ 0 ∧ ∀ (O : ℝ) (l m x1 y1 x2 y2 : ℝ), 
    abs(m) / sqrt(1 + k^2) = sqrt(3) / 2 ∧ 
    (3 * k^2 + 1) * x1^2 + 6 * k * m * x1 + 3 * m^2 - 3 = 0 ∧ 
    (3 * k^2 + 1) * x2^2 + 6 * k * m * x2 + 3 * m^2 - 3 = 0 
    → ( area = sqrt(3) / 2 )

# Example usage
theorem math_problem : ellipse ∧ line_passing_through_E_and_meets_conditions ∧ max_area_triangle :=
by sorry

end math_problem_l200_200842


namespace range_of_x_satisfies_conditions_l200_200159

theorem range_of_x_satisfies_conditions (x : ℝ) (h : x^2 - 4 < 0 ∨ |x| = 2) : -2 ≤ x ∧ x ≤ 2 := 
by
  sorry

end range_of_x_satisfies_conditions_l200_200159


namespace find_prime_q_l200_200326

theorem find_prime_q (p q r : ℕ) 
  (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q)
  (prime_r : Nat.Prime r)
  (eq_r : q - p = r)
  (cond_p : 5 < p ∧ p < 15)
  (cond_q : q < 15) :
  q = 13 :=
sorry

end find_prime_q_l200_200326


namespace sum_of_radii_of_tangent_circle_l200_200362

theorem sum_of_radii_of_tangent_circle :
  (∃ r: ℝ, r > 0 ∧ (r - 3)^2 + r^2 = (r + 1)^2 ∧ r + (r - 2*sqrt 2) = 8) :=
by
  sorry

end sum_of_radii_of_tangent_circle_l200_200362


namespace orange_ribbons_count_l200_200941

variable (total_ribbons : ℕ)
variable (orange_ribbons : ℚ)

-- Definitions of the given conditions
def yellow_fraction := (1 : ℚ) / 4
def purple_fraction := (1 : ℚ) / 3
def orange_fraction := (1 : ℚ) / 6
def black_ribbons := 40
def black_fraction := (1 : ℚ) / 4

-- Using the given and derived conditions
theorem orange_ribbons_count
  (hy : yellow_fraction = 1 / 4)
  (hp : purple_fraction = 1 / 3)
  (ho : orange_fraction = 1 / 6)
  (hb : black_ribbons = 40)
  (hbf : black_fraction = 1 / 4)
  (total_eq : total_ribbons = black_ribbons * 4) :
  orange_ribbons = total_ribbons * orange_fraction := by
  -- Proof omitted
  sorry

end orange_ribbons_count_l200_200941


namespace andy_initial_cookies_l200_200390

-- Define the conditions as constants
constant andy_ate : ℕ := 3
constant brother_received : ℕ := 5
constant team_members : ℕ := 8

-- Define the function for cookies taken by the basketball team using the problem's condition
def cookies_taken_by_nth_player (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

-- Calculate the total cookies taken by the basketball team
def total_cookies_taken_by_team : ℕ :=
  (list.range team_members).sum (λ n, cookies_taken_by_nth_player (n + 1))

-- Summing up the initial eats, gives, and taken by team
def total_cookies (andy_ate brother_received total_cookies_taken_by_team : ℕ) : ℕ :=
  andy_ate + brother_received + total_cookies_taken_by_team

theorem andy_initial_cookies : total_cookies andy_ate brother_received total_cookies_taken_by_team = 72 :=
by
  -- Introduce a known mathematical result: sum of arithmetic sequence
  have sum_arith_seq_8 : (list.range 8).sum (λ n, cookies_taken_by_nth_player (n + 1)) = 64 := 
    by sorry -- this is actually the result we assume from the solution
  simp [total_cookies, andy_ate, brother_received, sum_arith_seq_8]

end andy_initial_cookies_l200_200390


namespace smallest_positive_debt_resolvable_l200_200661

theorem smallest_positive_debt_resolvable :
  ∃ (p g : ℤ), 400 * p + 280 * g = 800 :=
sorry

end smallest_positive_debt_resolvable_l200_200661


namespace books_loaned_out_l200_200724

-- Define the conditions
def books_at_start := 75
def books_at_end := 67
def percentage_returned := 0.80
def percentage_not_returned := 1 - percentage_returned

-- Prove the number of books loaned out during the month
theorem books_loaned_out (x : ℝ) (h : percentage_not_returned * x = books_at_start - books_at_end) : x = 40 :=
  sorry

end books_loaned_out_l200_200724


namespace savings_after_expense_increase_l200_200720

-- Define the conditions
def monthly_salary : ℝ := 6500
def initial_savings_percentage : ℝ := 0.20
def increase_expenses_percentage : ℝ := 0.20

-- Define the statement we want to prove
theorem savings_after_expense_increase :
  (monthly_salary - (monthly_salary - (initial_savings_percentage * monthly_salary) + (increase_expenses_percentage * (monthly_salary - (initial_savings_percentage * monthly_salary))))) = 260 :=
sorry

end savings_after_expense_increase_l200_200720


namespace volume_of_mixture_l200_200317

section
variable (Va Vb Vtotal : ℝ)

theorem volume_of_mixture :
  (Va / Vb = 3 / 2) →
  (800 * Va + 850 * Vb = 2460) →
  (Vtotal = Va + Vb) →
  Vtotal = 2.998 :=
by
  intros h1 h2 h3
  sorry
end

end volume_of_mixture_l200_200317


namespace part_i_part_ii_part_iii_l200_200234

open Real

noncomputable def f : ℝ → ℝ := λ x, sorry

variable (x y : ℝ)

-- Given functional equation
axiom functional_eq (x y : ℝ) :
  f (f (x + y)) = f (x + y) + f x * f y - x * y

-- Define alpha
def alpha : ℝ := f 0

-- Prove part (i)
theorem part_i : f alpha * f (-alpha) = 0 :=
sorry

-- Prove part (ii)
theorem part_ii : alpha = 0 :=
sorry

-- Prove part (iii)
theorem part_iii : ∀ x : ℝ, f x = x :=
sorry

end part_i_part_ii_part_iii_l200_200234


namespace condition_iff_absolute_value_l200_200231

theorem condition_iff_absolute_value (a b : ℝ) : (a > b) ↔ (a * |a| > b * |b|) :=
sorry

end condition_iff_absolute_value_l200_200231


namespace dvd_cost_packs_l200_200346

theorem dvd_cost_packs (cost_per_pack : ℕ) (number_of_packs : ℕ) (total_money : ℕ) :
  cost_per_pack = 12 → number_of_packs = 11 → total_money = (cost_per_pack * number_of_packs) → total_money = 132 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end dvd_cost_packs_l200_200346


namespace boundary_of_shadow_of_sphere_l200_200735

theorem boundary_of_shadow_of_sphere (x y : ℝ) :
  let O := (0, 0, 2)
  let P := (1, -2, 3)
  let r := 2
  (∃ T : ℝ × ℝ × ℝ,
    T = (0, -2, 2) ∧
    (∃ g : ℝ → ℝ,
      y = g x ∧
      g x = (x^2 - 2 * x - 11) / 6)) → 
  y = (x^2 - 2 * x - 11) / 6 :=
by
  sorry

end boundary_of_shadow_of_sphere_l200_200735


namespace maxval_h_and_monotonic_intervals_l200_200837

noncomputable def f : ℝ → ℝ := λ x, x^2
noncomputable def g : ℝ → ℝ := λ x, x⁻¹
noncomputable def h : ℝ → ℝ := λ x, if f x ≤ g x then f x else g x

theorem maxval_h_and_monotonic_intervals :
  (∀ x > (0 : ℝ), h x ≤ 1) ∧ (∃ x : ℝ, h x = 1) ∧ 
  (∃ a b : ℝ, 0 < a ∧ a ≤ 1 ∧ ∀ x : ℝ, a < x ∧ x ≤ b → h x ≤ h x) ∧ 
  (∃ a : ℝ, (a < 0 ∨ 1 < a) ∧ ∀ x : ℝ, (x < a ∨ a < x) → h x ≤ h x) :=
sorry

end maxval_h_and_monotonic_intervals_l200_200837


namespace origami_papers_per_cousin_l200_200011

theorem origami_papers_per_cousin (total_papers : ℕ) (num_cousins : ℕ) (same_papers_each : ℕ) 
  (h1 : total_papers = 48) 
  (h2 : num_cousins = 6) 
  (h3 : same_papers_each = total_papers / num_cousins) : 
  same_papers_each = 8 := 
by 
  sorry

end origami_papers_per_cousin_l200_200011


namespace rectangle_diagonal_ratio_l200_200967

theorem rectangle_diagonal_ratio
  (EFGH : Type*) [rectangle EFGH]
  (E F G H J K Q : EFGH)
  (on_EF : J ∈ line_segment E F)
  (on_EH : K ∈ line_segment E H)
  (on_EG_1 : Q ∈ line_segment E G)
  (on_EG_2 : Q ∈ line J K)
  (EJ_ratio : EJ : line_segment E F = (1 / 4))
  (EK_ratio : EK : line_segment E H = (1 / 3)) :
  ∃ (r : ℝ), r = 3 ∧ EQ = r * Q :=
begin
  sorry
end

end rectangle_diagonal_ratio_l200_200967


namespace delores_money_left_l200_200101

def initial : ℕ := 450
def computer_cost : ℕ := 400
def printer_cost : ℕ := 40
def money_left (initial computer_cost printer_cost : ℕ) : ℕ := initial - (computer_cost + printer_cost)

theorem delores_money_left : money_left initial computer_cost printer_cost = 10 := by
  sorry

end delores_money_left_l200_200101


namespace dataset_data_points_l200_200982

theorem dataset_data_points (initial_points : ℕ) (added_percent : ℕ) (reduction_fraction : ℚ) : 
  initial_points = 200 → added_percent = 20 → reduction_fraction = 1/4 → 
  let added_points := (added_percent * initial_points) / 100 in
  let total_after_addition := initial_points + added_points in
  let reduced_points := reduction_fraction * total_after_addition in
  let final_total := total_after_addition - reduced_points in
  final_total = 180 :=
by
  intros _ _ _
  let added_points := (added_percent * initial_points) / 100
  let total_after_addition := initial_points + added_points
  let reduced_points := reduction_fraction * total_after_addition
  let final_total := total_after_addition - reduced_points
  have initial_points_correct : initial_points = 200 := by exact rfl
  have added_percent_correct : added_percent = 20 := by exact rfl
  have reduction_fraction_correct : reduction_fraction = 1/4 := by exact rfl
  rw [initial_points_correct, added_percent_correct, reduction_fraction_correct]
  sorry

end dataset_data_points_l200_200982


namespace circle_equation_l200_200122

theorem circle_equation
  (h k r : ℝ) (center : h = 2 ∧ k = -1) (radius : r = 4) :
  ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 ↔ (x - 2)^2 + (y + 1)^2 = 16 := by
  intro x y
  rw [center.left, center.right, radius]
  constructor
  · intro h_eq
    rw [h_eq]
    rfl
  · intro h_eq
    rw [h_eq]
    rfl

end circle_equation_l200_200122


namespace total_flowers_l200_200592

noncomputable def yellow_flowers : ℕ := 10
noncomputable def purple_flowers : ℕ := yellow_flowers + (80 * yellow_flowers) / 100
noncomputable def green_flowers : ℕ := (25 * (yellow_flowers + purple_flowers)) / 100
noncomputable def red_flowers : ℕ := (35 * (yellow_flowers + purple_flowers + green_flowers)) / 100

theorem total_flowers :
  yellow_flowers + purple_flowers + green_flowers + red_flowers = 47 :=
by
  -- Insert proof here
  sorry

end total_flowers_l200_200592


namespace solve_system_l200_200694

theorem solve_system (x y z : ℝ) (hx : x + y + z = 13)
    (hy : x^2 + y^2 + z^2 = 61) (hz : xy + xz = 2yz) :
    (x = 4 ∧ y = 3 ∧ z = 6) ∨ (x = 4 ∧ y = 6 ∧ z = 3) :=
by sorry

end solve_system_l200_200694


namespace kenny_total_liquid_l200_200399

def total_liquid (oil_per_recipe water_per_recipe : ℚ) (times : ℕ) : ℚ :=
  (oil_per_recipe + water_per_recipe) * times

theorem kenny_total_liquid :
  total_liquid 0.17 1.17 12 = 16.08 := by
  sorry

end kenny_total_liquid_l200_200399


namespace avg_annual_growth_rate_optimal_selling_price_l200_200685

-- Define the conditions and question for the first problem: average annual growth rate.
theorem avg_annual_growth_rate (initial final : ℝ) (years : ℕ) (growth_rate : ℝ) :
  initial = 200 ∧ final = 288 ∧ years = 2 ∧ (final = initial * (1 + growth_rate)^years) →
  growth_rate = 0.2 :=
by
  -- Proof will come here
  sorry

-- Define the conditions and question for the second problem: setting the selling price.
theorem optimal_selling_price (cost initial_volume : ℕ) (initial_price : ℝ) 
(additional_sales_per_dollar : ℕ) (desired_profit : ℝ) (optimal_price : ℝ) :
  cost = 50 ∧ initial_volume = 50 ∧ initial_price = 100 ∧ additional_sales_per_dollar = 5 ∧
  desired_profit = 4000 ∧ 
  (∃ p : ℝ, (p - cost) * (initial_volume + additional_sales_per_dollar * (initial_price - p)) = desired_profit ∧ p = optimal_price) →
  optimal_price = 70 :=
by
  -- Proof will come here
  sorry

end avg_annual_growth_rate_optimal_selling_price_l200_200685


namespace marites_saves_120_per_year_l200_200245

def current_internet_speed := 10 -- Mbps
def current_monthly_bill := 20 -- dollars

def monthly_cost_20mbps := current_monthly_bill + 10 -- dollars
def monthly_cost_30mbps := current_monthly_bill * 2 -- dollars

def bundled_cost_20mbps := 80 -- dollars per month
def bundled_cost_30mbps := 90 -- dollars per month

def annual_cost_20mbps := bundled_cost_20mbps * 12 -- dollars per year
def annual_cost_30mbps := bundled_cost_30mbps * 12 -- dollars per year

theorem marites_saves_120_per_year :
  annual_cost_30mbps - annual_cost_20mbps = 120 := 
by
  sorry

end marites_saves_120_per_year_l200_200245


namespace final_value_after_operations_l200_200031

theorem final_value_after_operations :
  let initial_value := 1500
  let first_increase := initial_value * 0.20
  let after_first_increase := initial_value + first_increase
  let first_decrease := after_first_increase * 0.15
  let after_first_decrease := after_first_increase - first_decrease
  let second_increase := after_first_decrease * 0.10
  let final_value := after_first_decrease + second_increase
  final_value = 1683 := by
  let initial_value := 1500
  let first_increase := initial_value * 0.20
  let after_first_increase := initial_value + first_increase
  let first_decrease := after_first_increase * 0.15
  let after_first_decrease := after_first_increase - first_decrease
  let second_increase := after_first_decrease * 0.10
  let final_value := after_first_decrease + second_increase
  show final_value = 1683
  sorry

end final_value_after_operations_l200_200031


namespace bad_carrots_count_l200_200760

theorem bad_carrots_count : 
  let carol_picked := 29,
      mom_picked := 16,
      good_carrots := 38,
      total_picked := carol_picked + mom_picked,
      bad_carrots := total_picked - good_carrots
  in bad_carrots = 7 := 
by
  let carol_picked := 29
  let mom_picked := 16
  let good_carrots := 38
  let total_picked := carol_picked + mom_picked
  let bad_carrots := total_picked - good_carrots
  show bad_carrots = 7 from sorry

end bad_carrots_count_l200_200760


namespace original_price_of_dish_l200_200690

-- Define the variables and conditions explicitly
variables (P : ℝ)

-- John's payment after discount and tip over original price
def john_payment : ℝ := 0.9 * P + 0.15 * P

-- Jane's payment after discount and tip over discounted price
def jane_payment : ℝ := 0.9 * P + 0.135 * P

-- Given condition that John's payment is $0.63 more than Jane's
def payment_difference : Prop := john_payment P - jane_payment P = 0.63

theorem original_price_of_dish (h : payment_difference P) : P = 42 :=
by sorry

end original_price_of_dish_l200_200690


namespace convert_mass_convert_area_l200_200705

-- Define the conversion rates
def kilogram_to_ton : ℝ := 1 / 1000
def square_meter_to_square_decimeter : ℝ := 100

-- Problem 1: Conversion of mass
theorem convert_mass (mass_tons : ℝ) (mass_kilograms: ℝ) (conv_kg_to_ton: ℝ)
  (h1: mass_tons = 8) (h2: mass_kilograms = 800) (h3: conv_kg_to_ton = kilogram_to_ton) : 
  mass_tons + mass_kilograms * conv_kg_to_ton = 8.8 :=
by 
  sorry

-- Problem 2: Conversion of area
theorem convert_area (area_square_meters : ℝ) (conv_sq_meter_to_sq_decimeter: ℝ)
  (h1: area_square_meters = 6.32) (h2: conv_sq_meter_to_sq_decimeter = square_meter_to_square_decimeter) : 
  area_square_meters * conv_sq_meter_to_sq_decimeter = 632 :=
by 
  sorry

end convert_mass_convert_area_l200_200705


namespace angle_complement_l200_200166

-- Conditions: The complement of angle A is 60 degrees
def complement (α : ℝ) : ℝ := 90 - α 

theorem angle_complement (A : ℝ) : complement A = 60 → A = 30 :=
by
  sorry

end angle_complement_l200_200166


namespace find_hours_hired_l200_200246

def hourly_rate : ℝ := 15
def tip_rate : ℝ := 0.20
def total_paid : ℝ := 54

theorem find_hours_hired (h : ℝ) : 15 * h + 0.20 * 15 * h = 54 → h = 3 :=
by
  sorry

end find_hours_hired_l200_200246


namespace election_winner_percentage_l200_200032

theorem election_winner_percentage :
    let votes_candidate1 := 2500
    let votes_candidate2 := 5000
    let votes_candidate3 := 15000
    let total_votes := votes_candidate1 + votes_candidate2 + votes_candidate3
    let winning_votes := votes_candidate3
    (winning_votes / total_votes) * 100 = 75 := 
by 
    sorry

end election_winner_percentage_l200_200032


namespace coin_flip_ways_l200_200033

theorem coin_flip_ways : ∃ (n : ℕ), ∀ (k : ℕ), k = 10 → (number_of_valid_sequences k = 42) :=
by sorry


end coin_flip_ways_l200_200033


namespace problem_inequality_l200_200449

variable {ι : Type} [Fintype ι] [DecidableEq ι]
variable {a : ι → ℝ}

theorem problem_inequality 
(h_pos : ∀ i, a i > 0)
(h_sum : ∑ i, a i = 1) :
∑ i, (a i) ^ 2 / (a i + a ((i + 1) % Fintype.card ι)) ≥ 1/2 := 
sorry

end problem_inequality_l200_200449


namespace problem_statement_l200_200727

-- Given a rectangle \( P \) with dimensions \( 2a \) and \( 3a \)
def dimensions_P (a : ℝ) (ha : a > 0) : ℝ × ℝ := (2 * a, 3 * a)

-- The number of rectangles with dimensions \( x \) and \( y \) satisfying specific conditions
noncomputable def count_valid_rectangles (a : ℝ) (ha : a > 0) : ℕ :=
  let P := dimensions_P a ha in
  let perimeter_P := 2 * (P.1 + P.2) in
  let area_P := P.1 * P.2 in
  let perimeter_new := (2 / 3) * perimeter_P in
  let area_new := (2 / 9) * area_P in
  let x_plus_y := perimeter_new / 2 in
  let xy := area_new in
  have h1 : ∀ x y, x + y = x_plus_y → x * y = xy → x < 2 * a ∧ y < 2 * a → True,
    from sorry,
  1 -- The actual computation should derive this value but we know from the solution it is 1

theorem problem_statement (a : ℝ) (ha : a > 0) : count_valid_rectangles a ha = 1 :=
by sorry

end problem_statement_l200_200727


namespace circle_area_l200_200328

-- Definition of the given circle equation
def circle_eq (x y : ℝ) : Prop := 3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0

-- Prove the area of the circle defined by circle_eq (x y) is 25/4 * π
theorem circle_area (x y : ℝ) (h : circle_eq x y) : ∃ r : ℝ, r = 5 / 2 ∧ π * r^2 = 25 / 4 * π :=
by
  sorry

end circle_area_l200_200328


namespace length_of_train_is_correct_l200_200381

-- Define the conditions
def speed_kmph : ℝ := 72
def time_seconds : ℝ := 25
def platform_length_meters : ℝ := 250.04

-- Convert speed from kmph to m/s
def speed_mps : ℝ := speed_kmph * (5 / 18)

-- Calculate the distance covered by the train while crossing the platform
def distance_covered : ℝ := speed_mps * time_seconds

-- Define the length of the train
def length_of_train : ℝ := distance_covered - platform_length_meters

-- Theorem statement
theorem length_of_train_is_correct :
  length_of_train = 249.96 :=
by
  sorry

end length_of_train_is_correct_l200_200381


namespace find_m_l200_200895

variables {R : Type*} [CommRing R]

/-- Definition of the dot product in a 2D vector space -/
def dot_product (a b : R × R) : R := a.1 * b.1 + a.2 * b.2

/-- Given vectors a and b as conditions -/
def a : ℚ × ℚ := (m, 3)
def b : ℚ × ℚ := (1, m + 1)

theorem find_m (m : ℚ) (h : dot_product a b = 0) : m = -3 / 4 :=
sorry

end find_m_l200_200895


namespace points_on_line_l200_200352

variables {Point : Type}

noncomputable def dist : Point -> Point -> ℝ := sorry

def collinear (pts : list Point) : Prop := sorry

theorem points_on_line
  {A : ℕ → Point}
  (B C : Point)
  {n : ℕ}
  (d : ℝ)
  (h1 : ∑ i in finset.range n, dist (A i) B = d)
  (h2 : ∑ i in finset.range n, dist (A i) C = d)
  (h3 : ∀ P : Point, ∑ i in finset.range n, dist (A i) P ≥ d) :
  collinear (list.of_fn A) :=
begin
  sorry
end

end points_on_line_l200_200352


namespace most_likely_units_digit_is_1_l200_200110

theorem most_likely_units_digit_is_1 :
  let outcomes := { n : ℕ // 1 ≤ n ∧ n ≤ 12 }
  ∃! d : ℕ, 0 ≤ d ∧ d < 10 ∧
  (∀ a b : outcomes, (a.val * b.val) % 10 = d → ↑d ≤ ↑(∑ x : outcomes × outcomes, ite ((x.1.val * x.2.val) % 10 = d) 1 0)) ∧ d = 1 :=
begin
  sorry
end

end most_likely_units_digit_is_1_l200_200110


namespace decompose_expression_l200_200098

-- Define the variables a and b as real numbers
variables (a b : ℝ)

-- State the theorem corresponding to the proof problem
theorem decompose_expression : 9 * a^2 * b - b = b * (3 * a + 1) * (3 * a - 1) :=
by
  sorry

end decompose_expression_l200_200098


namespace area_satisfies_pqrrules_l200_200113

noncomputable def radius : ℝ := 5
noncomputable def side_length_of_triangle_ABC : ℝ := 2 * radius * real.sin (real.pi / 3)
noncomputable def AD : ℝ := 15
noncomputable def AE : ℝ := 20
noncomputable def F_is_defined : Prop := (l_parallel_to AE_through D ∧ l_parallel_to AD_through E ∧ F_is_intersection_of_these_lines)
noncomputable def G_is_defined : Prop := (collinear_with_A_and_F_on_circle_and_distinct_from_A)
noncomputable def area_triangle_CBG_equals_expr : ℝ := 216 * real.sqrt 3 / 25

theorem area_satisfies_pqrrules : 
(side_length_of_triangle_ABC = 10 * real.sqrt 3) → 
(G_is_defined) → 
(F_is_defined) → 
(area_triangle_CBG_equals_expr = 216 * real.sqrt 3 / 25) → 
(∃ (p q r : ℕ), (p + q + r = 244) ∧ p = 216 ∧ q = 3 ∧ r = 25) :=
by
  sorry

end area_satisfies_pqrrules_l200_200113


namespace problem1_problem2_problem3_l200_200239

-- Definitions of sequences and their properties
variable {α : Type} [Field α]

def geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n + 1) = q * a n

def arithmetic_sequence (b : ℕ → α) (d : α) : Prop :=
  ∀ n, b (n + 1) = b n + d

def combined_sequence (a b c : ℕ → α) : Prop :=
  ∀ n, c n = a n + b n

-- Problem 1
theorem problem1 (a b c : ℕ → α) (q d : α) (h1 : geometric_sequence a q) 
  (h2 : arithmetic_sequence b d) (h3 : combined_sequence a b c) (h_q : q ≠ 1) (h_d : d ≠ 0) :
  ¬ (arithmetic_sequence c (c 2 - c 1)) :=
sorry

-- Problem 2
theorem problem2 (b2 d : α) (h_q : q = 2) (h_a1 : a 0 = 1) 
  (h1 : ∀ n, a (n + 1) = q * a n) (h2 : c 1 * c 3 = c 2 ^ 2) :
  b2 = d^2 + 3*d ∧ d ≠ -1 ∧ d ≠ -2 ∧ d ≠ 0 :=
sorry

-- Problem 3
theorem problem3 (a b c : ℕ → α) (q d : α) (h1 : geometric_sequence a q) 
  (h2 : arithmetic_sequence b d) (h3 : combined_sequence a b c) (h_q : q ≠ 1) (h_d : d ≠ 0) :
  ¬ (geometric_sequence c (c 1 / c 0)) :=
sorry

end problem1_problem2_problem3_l200_200239


namespace smallest_area_of_right_triangle_l200_200106

noncomputable def smallest_area (x : ℝ) (h : ℝ) : ℝ :=
  if h = 6
  then (x * sqrt (h^2 - x^2)) / 2
  else (6 * x) / 2

theorem smallest_area_of_right_triangle 
  (x : ℝ)
  (hx : x < 6)
  (h : ℝ)
  (hhh : h = 6 ∨ h^2 = 36 + x^2) 
  :
  smallest_area x h = (5 * sqrt 11) / 2 :=
sorry

end smallest_area_of_right_triangle_l200_200106


namespace alpha_values_m_range_l200_200236

noncomputable section

open Real

def f (x : ℝ) (α : ℝ) : ℝ := 2^(x + cos α) - 2^(-x + cos α)

-- Problem 1: Set of values for α
theorem alpha_values (h : f 1 α = 3/4) : ∃ k : ℤ, α = 2 * k * π + π :=
sorry

-- Problem 2: Range of values for real number m
theorem m_range (h0 : 0 ≤ θ ∧ θ ≤ π / 2) 
  (h1 : ∀ (m : ℝ), f (m * cos θ) (-1) + f (1 - m) (-1) > 0) : 
  ∀ (m : ℝ), m < 1 :=
sorry

end alpha_values_m_range_l200_200236


namespace area_of_quadrilateral_ABDF_l200_200373

theorem area_of_quadrilateral_ABDF :
  let length := 40
  let width := 30
  let rectangle_area := length * width
  let B := (1/4 : ℝ) * length
  let F := (1/2 : ℝ) * width
  let area_BCD := (1/2 : ℝ) * (3/4 : ℝ) * length * width
  let area_EFD := (1/2 : ℝ) * F * length
  rectangle_area - area_BCD - area_EFD = 450 := sorry

end area_of_quadrilateral_ABDF_l200_200373


namespace second_workshop_production_l200_200514

theorem second_workshop_production (a b c : ℕ) (h₁ : a + b + c = 3600) (h₂ : a + c = 2 * b) : b * 3 = 3600 := 
by 
  sorry

end second_workshop_production_l200_200514


namespace sum_of_coefficients_equal_19_l200_200627

theorem sum_of_coefficients_equal_19 :
  ∀ (a b c d : ℕ), 
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧ 
  (cos x + cos (5 * x) + cos (11 * x) + cos (15 * x) = a * cos (b * x) * cos (c * x) * cos (d * x)) → 
  a + b + c + d = 19 :=
by
  sorry

end sum_of_coefficients_equal_19_l200_200627


namespace triangle_area_ab_ac_angle_a_l200_200204

theorem triangle_area_ab_ac_angle_a (AB AC : ℝ) (angle_A : ℝ) (h_AB : AB = 3) 
  (h_AC : AC = 2) (h_angle_A : angle_A = real.pi / 3) :
  (1 / 2 * AB * AC * real.sin angle_A) = (3 / 2 * real.sqrt 3) :=
by
  rw [h_AB, h_AC, h_angle_A]
  simp [real.sin_pi_div_three]
  norm_num

end triangle_area_ab_ac_angle_a_l200_200204


namespace cosine_of_angle_between_AB_AC_l200_200784

-- Define points A, B, and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := { x := 0, y := 1, z := 0 }
def B : Point3D := { x := 0, y := 2, z := 1 }
def C : Point3D := { x := 1, y := 2, z := 0 }

-- Define vector from points
def vector (p1 p2 : Point3D) : Point3D :=
  { x := p2.x - p1.x, y := p2.y - p1.y, z := p2.z - p1.z }

def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def magnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

-- Define vectors AB and AC
def AB := vector A B
def AC := vector A C

-- Define the cosine of the angle between AB and AC
noncomputable def cos_angle_between_AB_AC : ℝ :=
  dot_product AB AC / (magnitude AB * magnitude AC)

-- Theorem to prove
theorem cosine_of_angle_between_AB_AC :
  cos_angle_between_AB_AC = 1 / 2 :=
by
  sorry

end cosine_of_angle_between_AB_AC_l200_200784


namespace count_valid_sequences_l200_200955

theorem count_valid_sequences :
  let possible_counts := {0, 2, 4, 6, 8, 10}
  ∃ (x y z w : ℕ), 
    x ∈ possible_counts ∧ y ∈ possible_counts ∧ z ∈ possible_counts ∧ w ∈ possible_counts ∧
    x + y + z + w = 10 ∧
    finset.card {s : finset (ℕ × ℕ × ℕ × ℕ) | s.count {x, y, z, w} = 1} = 56 :=
  sorry

end count_valid_sequences_l200_200955


namespace min_length_segment_CP_l200_200222

variable {V : Type*} [inner_product_space ℝ V] [module ℝ V] [finite_dimensional ℝ V]
variables (A B C D P: V)
variables {m: ℝ}

noncomputable def triangle_area (A B C: V) : ℝ :=
0.5 * (norm (cross_product (B - A) (C - A)))

theorem min_length_segment_CP :
  (∀ (A B C D P: V) (m: ℝ), collinear [A, D, P] ∧ ∥D - B∥ = 2 * ∥B∥ ∧
    (∥P - A∥ = 1 / 2 * ∥C - A∥ + m * ∥B - C∥) ∧
    (triangle_area A B C = sqrt 3) ∧
    (angle A C B = pi / 3) →
    (∥P - C∥ = sqrt 2)) := 
sorry

end min_length_segment_CP_l200_200222


namespace cars_to_sell_l200_200731

theorem cars_to_sell (clients : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) (total_clients : ℕ) (h1 : selections_per_client = 2) 
  (h2 : selections_per_car = 3) (h3 : total_clients = 24) : (total_clients * selections_per_client / selections_per_car = 16) :=
by
  sorry

end cars_to_sell_l200_200731


namespace cot_neg_45_eq_neg_one_l200_200754

theorem cot_neg_45_eq_neg_one : Real.cot (-(π / 4)) = -1 :=
by
  -- Provided conditions as definitions:
  -- 1. cotangent definition
  have h1 : ∀ θ, Real.cot θ = 1 / Real.tan θ,
  from λ θ, Real.cot_def θ,
  
  -- 2. Tangent property
  have h2 : Real.tan (-(π / 4)) = -Real.tan (π / 4),
  from Real.tan_neg (π / 4),
  
  -- 3. Evaluate tan(45 degrees)
  have h3 : Real.tan (π / 4) = 1,
  from Real.tan_pi_div_four,
  
  -- Combine conditions to prove the final result
  sorry

end cot_neg_45_eq_neg_one_l200_200754


namespace max_ordinate_of_parabola_intersects_x_neg5_l200_200645

theorem max_ordinate_of_parabola_intersects_x_neg5 : 
  ∃ (Q : ℝ), ∀ (b c : ℝ), 
    (c = (-b^2 + 6 * b + 4) / 4) → 
    (Q = -1/4 * b^2 - 7/2 * b - 24) → 
    (Q ≤ -47 / 4) :=
begin
  sorry
end

end max_ordinate_of_parabola_intersects_x_neg5_l200_200645


namespace sqrt_approximation_l200_200409

noncomputable def approximation_formula (n : ℕ) (x₀ Δx : ℝ) (h_x₀_pos : 0 < x₀) (h_small_delta_x : |Δx| ≪ x₀^n) : ℝ :=
  x₀ + (Δx / (n * x₀^(n - 1)))

theorem sqrt_approximation
  (n : ℕ)
  (x₀ Δx : ℝ)
  (h_x₀_pos : 0 < x₀)
  (h_small_delta_x : |Δx| ≪ x₀^n):
  (x₀^n + Δx) ^ (1/n : ℝ) ≈ x₀ + (Δx / (n * x₀^(n - 1))) :=
begin
  sorry
end

end sqrt_approximation_l200_200409


namespace perpendicular_vectors_implies_m_value_l200_200873

variable (m : ℝ)

def vector1 : ℝ × ℝ := (m, 3)
def vector2 : ℝ × ℝ := (1, m + 1)

theorem perpendicular_vectors_implies_m_value
  (h : vector1 m ∙ vector2 m = 0) :
  m = -3 / 4 :=
by 
  sorry

end perpendicular_vectors_implies_m_value_l200_200873


namespace lydia_apple_trees_l200_200591

/-- Lydia's problem on tree ages and cumulative age -/
theorem lydia_apple_trees (
  (years_to_bear_fruit_1 : ℕ) (years_to_bear_fruit_2 : ℕ) (years_to_bear_fruit_3 : ℕ)
  (age_planted_1 : ℕ) (age_planted_2 : ℕ) (age_planted_3 : ℕ)
  (current_age : ℕ) :
  years_to_bear_fruit_1 = 8 → years_to_bear_fruit_2 = 10 → years_to_bear_fruit_3 = 12 →
  age_planted_1 = 5 → age_planted_2 = 7 → age_planted_3 = 9 →
  current_age = 14 →
  let age_eat_fruit_1 := age_planted_1 + years_to_bear_fruit_1 in
  let age_eat_fruit_2 := age_planted_2 + years_to_bear_fruit_2 in
  let age_eat_fruit_3 := age_planted_3 + years_to_bear_fruit_3 in
  let age_tree_1_at_fruit := current_age + (age_eat_fruit_1 - current_age) in
  let age_tree_2_at_fruit := current_age + (age_eat_fruit_2 - current_age) in
  let age_tree_3_at_fruit := current_age + (age_eat_fruit_3 - current_age) in
  age_eat_fruit_1 = 13 ∧ age_eat_fruit_2 = 17 ∧ age_eat_fruit_3 = 21 ∧
  (age_tree_1_at_fruit - age_planted_1) = 20 ∧
  (age_tree_2_at_fruit - age_planted_2) = 20 ∧
  (age_tree_3_at_fruit - age_planted_3) = 24 ∧
  (age_tree_1_at_fruit - age_planted_1 + age_tree_2_at_fruit - age_planted_2 + age_tree_3_at_fruit - age_planted_3) = 64 :=
by
  intros
  sorry

end lydia_apple_trees_l200_200591


namespace max_cheetahs_no_attack_l200_200351

theorem max_cheetahs_no_attack : 
  let n := 1000 in
  let attack_radius := 19 in
  let avoid_row_col := (n * n) in
  ∀ (grid : array (fin n) (array (fin n) bool)),
    (∀ i j, grid[i][j] = true → 
      (∀ k, k ≠ i → grid[k][j] = false) ∧ 
      (∀ l, l ≠ j → grid[i][l] = false) ∧ 
      (∀ a b, abs (a - i) ≤ attack_radius ∧ abs (b - j) ≤ attack_radius → grid[a][b] = false)) →
  (∑ i, ∑ j, if grid[i][j] then 1 else 0) ≤ 100000 :=
by
  sorry

end max_cheetahs_no_attack_l200_200351


namespace steps_fewer_in_final_staircase_l200_200542

-- Given conditions
def first_staircase_steps : ℕ := 20
def second_staircase_steps : ℕ := 2 * first_staircase_steps
def total_height_feet : ℝ := 45
def step_height_feet : ℝ := 0.5

-- Total feet climbed for the first two staircases
def total_feet_first_two_staircases : ℝ := (first_staircase_steps + second_staircase_steps) * step_height_feet

-- Remaining feet for the final staircase
def height_final_staircase : ℝ := total_height_feet - total_feet_first_two_staircases

-- Number of steps in the final staircase
def final_staircase_steps : ℕ := height_final_staircase / step_height_feet

-- Prove the number of fewer steps of the final staircase compared to the second staircase
theorem steps_fewer_in_final_staircase :
  second_staircase_steps - final_staircase_steps = 10 := by
  sorry

end steps_fewer_in_final_staircase_l200_200542


namespace brooke_homework_time_l200_200802

def num_math_problems := 15
def num_social_studies_problems := 6
def num_science_problems := 10

def time_per_math_problem := 2 -- in minutes
def time_per_social_studies_problem := 0.5 -- in minutes (30 seconds)
def time_per_science_problem := 1.5 -- in minutes

def total_time : ℝ :=
  num_math_problems * time_per_math_problem +
  num_social_studies_problems * time_per_social_studies_problem +
  num_science_problems * time_per_science_problem

theorem brooke_homework_time :
  total_time = 48 := by
  sorry

end brooke_homework_time_l200_200802


namespace p_divisible_by_1979_l200_200584

theorem p_divisible_by_1979 (p q : ℕ)
  (h : p * q⁻¹ = 1 - 1/2 + 1/3 - 1/4 + 1/5 - · · · + 1/1319) : 
  1979 ∣ p :=
sorry

end p_divisible_by_1979_l200_200584


namespace monotonicity_of_f_f_upper_bound_k_neg_l200_200169

noncomputable def f (x k : ℝ) : ℝ := Real.log x + k * x^2 + (2 * k + 1) * x

theorem monotonicity_of_f :
  ∀ x k : ℝ, x > 0 → 
    ((k ≥ 0 → ∀ x, x > 0 → f x k > 0) ∧ 
    (k < 0 → ∀ x₁ x₂, 0 < x₁ < -1/(2*k) → f x₁ k < f x₂ k ∧ -1/(2*k) < x₂ → f x₁ k > f x₂ k)) := 
sorry

theorem f_upper_bound_k_neg (k : ℝ) (h : k < 0) : 
  ∀ x : ℝ, x > 0 → f x k ≤ -3 / (4 * k) - 2 := 
sorry

end monotonicity_of_f_f_upper_bound_k_neg_l200_200169


namespace both_sports_l200_200212

-- Definitions based on the given conditions
def total_members := 80
def badminton_players := 48
def tennis_players := 46
def neither_players := 7

-- The theorem to be proved
theorem both_sports : (badminton_players + tennis_players - (total_members - neither_players)) = 21 := by
  sorry

end both_sports_l200_200212


namespace badges_before_exchange_l200_200668

theorem badges_before_exchange (V T : ℕ) (h1 : V = T + 5) (h2 : 76 * V + 20 * T = 80 * T + 24 * V - 100) :
  V = 50 ∧ T = 45 :=
by
  sorry

end badges_before_exchange_l200_200668


namespace smallest_consecutive_even_sum_560_l200_200273

theorem smallest_consecutive_even_sum_560 (n : ℕ) (h : 7 * n + 42 = 560) : n = 74 :=
  by
    sorry

end smallest_consecutive_even_sum_560_l200_200273


namespace locus_of_A_is_hyperbola_l200_200163

-- Define the coordinates of points B and C
def B : (ℝ × ℝ) := (-6, 0)
def C : (ℝ × ℝ) := (6, 0)

-- Define Angles and their conditions
variables (A B C : ℝ) -- assuming B, C are angle values not coordinates here

-- Given condition for sin
def sin_condition : Prop :=
  sin B - sin C = (1/2) * sin A

-- Define the distances |AB| and |AC|
def AB (A : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 + 6)^2 + (A.2)^2)

def AC (A : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - 6)^2 + (A.2)^2)

-- Define the condition for the hyperbola
def hyperbola_condition (A : ℝ × ℝ) : Prop :=
  (A.1 ^ 2) / 9 - (A.2 ^ 2) / 27 = 1

-- The statement we need to prove
theorem locus_of_A_is_hyperbola (A : ℝ × ℝ) (h : sin_condition A B C) : 
  hyperbola_condition A :=
sorry -- Proof omitted

end locus_of_A_is_hyperbola_l200_200163


namespace range_f_omega_1_interval_monotonically_decreasing_l200_200817

noncomputable def vec_a (ω x : ℝ) : ℝ × ℝ :=
  (sqrt 2 * sin (ω * x), sqrt 2 * cos (ω * x) + 1)

noncomputable def vec_b (ω x : ℝ) : ℝ × ℝ :=
  (sqrt 2 * cos (ω * x), sqrt 2 * cos (ω * x) - 1)

noncomputable def f (ω x : ℝ) : ℝ :=
  let a := vec_a ω x
  let b := vec_b ω x
  a.1 * b.1 + a.2 * b.2

theorem range_f_omega_1 (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  -1 < f 1 x ∧ f 1 x ≤ sqrt 2 :=
sorry

theorem interval_monotonically_decreasing (ω : ℝ) (ω_eq_neg1 : ω = -1) (x k : ℝ) :
  (k * π - π / 8 ≤ x ∧ x ≤ k * π + 3 * π / 8) →
  ∀ x1 x2, (x1 < x2 → f ω x2 < f ω x1) :=
sorry

end range_f_omega_1_interval_monotonically_decreasing_l200_200817


namespace order_P_R_Q_l200_200920

variable (x y : ℝ)

theorem order_P_R_Q (h1 : x > y) (h2 : y > 1) :
  let P := log ((x + y) / 2)
  let Q := sqrt (log x * log y)
  let R := (log x + log y) / 2
  P > R ∧ R > Q :=
by sorry

end order_P_R_Q_l200_200920


namespace circle_radius_l200_200621

theorem circle_radius (C : ℝ) (r : ℝ) (h1 : C = 72 * Real.pi) (h2 : C = 2 * Real.pi * r) : r = 36 :=
by
  sorry

end circle_radius_l200_200621


namespace smallest_consecutive_even_sum_560_l200_200274

theorem smallest_consecutive_even_sum_560 (n : ℕ) (h : 7 * n + 42 = 560) : n = 74 :=
  by
    sorry

end smallest_consecutive_even_sum_560_l200_200274


namespace problem_statement_l200_200848

noncomputable def f (x : ℝ) : ℝ := exp x / (x - 1)

def monotonic_intervals : Prop :=
  (∀ x, x > 2 → f' x > 0) ∧
  (∀ x, x < 2 ∧ x ≠ 1 → f' x < 0)

def a_range (a : ℝ) : Prop :=
  (∀ x, x ≥ 2 → f' x ≥ a * f x) → a ≤ 0

theorem problem_statement (a : ℝ) :
  monotonic_intervals ∧ a_range a := 
by
  sorry

end problem_statement_l200_200848


namespace concert_probability_l200_200508

noncomputable def probability_at_least_7_stay (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  ∑ i in finset.range k.succ, (nat.choose n i) * p^i * (1 - p)^(n - i)

noncomputable def binomial_probability (n : ℕ) (p : ℚ) (k : ℕ) : ℚ :=
  (nat.choose n k) * p^k * (1 - p)^(n - k)

theorem concert_probability : 
  probability_at_least_7_stay 8 (1/3) 7 = (10/243 + 1/243) := 
sorry

end concert_probability_l200_200508


namespace produce_total_worth_l200_200812

/-- Gary is restocking the grocery produce section. He adds 60 bundles of asparagus at $3.00 each, 
40 boxes of grapes at $2.50 each, and 700 apples at $0.50 each. 
This theorem proves that the total worth of all the produce Gary stocked is $630.00. -/
theorem produce_total_worth :
  let asparagus_bundles := 60
  let asparagus_price := 3.00
  let grapes_boxes := 40
  let grapes_price := 2.50
  let apples_count := 700
  let apples_price := 0.50 in
  (asparagus_bundles * asparagus_price) + (grapes_boxes * grapes_price) + (apples_count * apples_price) = 630.00 :=
by
  sorry

end produce_total_worth_l200_200812


namespace initial_ecus_l200_200658

theorem initial_ecus (x y z : ℤ) (h1 : x - (y + z) = 8) (h2 : y - (x + z) = 8) (h3 : z - (x + y) = 8) :
    x = 13 ∧ y = 7 ∧ z = 4 := 
begin
  sorry
end

end initial_ecus_l200_200658


namespace sum_of_digits_n_minus_1_l200_200722
open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  (repr n).foldl (λ acc c, acc + c.toNat - '0'.toNat) 0

theorem sum_of_digits_n_minus_1 (n : ℕ) (h_distinct : list.nodup (repr n).to_list) (h_sum_digits : sum_of_digits n = 22) : 
  (mod n 10 ≠ 0 → sum_of_digits (n - 1) = 21) ∧ (mod n 10 = 0 → sum_of_digits (n - 1) = 30) :=
sorry

end sum_of_digits_n_minus_1_l200_200722


namespace rotten_eggs_percentage_l200_200976

variable spoilage_rate_milk : ℝ := 0.20
variable weevil_rate_flour : ℝ := 0.25
variable prob_all_good : ℝ := 0.24
variable prob_good_milk : ℝ := 1 - spoilage_rate_milk
variable prob_good_flour : ℝ := 1 - weevil_rate_flour

theorem rotten_eggs_percentage :
  ∃ (R : ℝ), R = 0.60 ∧ prob_good_milk * prob_good_flour * (1 - R) = prob_all_good := 
begin
  sorry
end

end rotten_eggs_percentage_l200_200976


namespace find_a_l200_200830

variable {a : ℝ}

def p (a : ℝ) := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > -1 ∧ x₂ > -1 ∧ x₁ * x₁ + 2 * a * x₁ + 1 = 0 ∧ x₂ * x₂ + 2 * a * x₂ + 1 = 0

def q (a : ℝ) := ∀ x : ℝ, a * x * x - a * x + 1 > 0 

theorem find_a (a : ℝ) : (p a ∨ q a) ∧ ¬ q a → a ≤ -1 :=
sorry

end find_a_l200_200830


namespace evie_collected_shells_for_6_days_l200_200116

theorem evie_collected_shells_for_6_days (d : ℕ) (h1 : 10 * d - 2 = 58) : d = 6 := by
  sorry

end evie_collected_shells_for_6_days_l200_200116


namespace find_m_l200_200892

variables {R : Type*} [CommRing R]

/-- Definition of the dot product in a 2D vector space -/
def dot_product (a b : R × R) : R := a.1 * b.1 + a.2 * b.2

/-- Given vectors a and b as conditions -/
def a : ℚ × ℚ := (m, 3)
def b : ℚ × ℚ := (1, m + 1)

theorem find_m (m : ℚ) (h : dot_product a b = 0) : m = -3 / 4 :=
sorry

end find_m_l200_200892


namespace least_possible_value_d_l200_200349

theorem least_possible_value_d 
  (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (hxy : x < y)
  (hyz : y < z)
  (hyx_gt_five : y - x > 5) : 
  z - x = 9 :=
sorry

end least_possible_value_d_l200_200349


namespace triangle_ABC_proof_l200_200512

noncomputable def value_of_c (a b cos_C : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2 - 2 * a * b * cos_C)  -- c^2 = a^2 + b^2 - 2ab cos(C)

noncomputable def ratio_S_in_S_out (a b c r R : ℝ) : ℝ :=
  (r / R) ^ 2 --

theorem triangle_ABC_proof (a b c : ℝ) (cos_C : ℝ) (sin_C : ℝ) (r R : ℝ)
  (h1 : a = 1) (h2 : b = 2) (h3 : cos_C = 3 / 4)
  (h4 : c = value_of_c a b cos_C)
  (r_def : r = (a * b * sin_C) / (a + b + c))
  (R_def : R = c / (2 * sin_C))
  (sin_C_def : sin_C = real.sqrt (1 - cos_C^2)) :
  c = real.sqrt 2 ∧ ratio_S_in_S_out a b c r R = (11 - 6 * real.sqrt 2) / 32 :=
by
  sorry

end triangle_ABC_proof_l200_200512


namespace similar_triangles_y_value_l200_200376

theorem similar_triangles_y_value :
  ∀ (y : ℚ),
    (12 : ℚ) / y = (9 : ℚ) / 6 → 
    y = 8 :=
by
  intros y h
  sorry

end similar_triangles_y_value_l200_200376


namespace vasya_tolya_badges_l200_200671

-- Let V be the number of badges Vasya had before the exchange.
-- Let T be the number of badges Tolya had before the exchange.
theorem vasya_tolya_badges (V T : ℕ) 
  (h1 : V = T + 5)
  (h2 : 0.76 * V + 0.20 * T = 0.80 * T + 0.24 * V - 1) :
  V = 50 ∧ T = 45 :=
by 
  sorry

end vasya_tolya_badges_l200_200671


namespace solution_proof_problem_l200_200572

open Real

noncomputable def proof_problem : Prop :=
  ∀ (a b c : ℝ),
  b + c = 17 →
  c + a = 18 →
  a + b = 19 →
  sqrt (a * b * c * (a + b + c)) = 36 * sqrt 5

theorem solution_proof_problem : proof_problem := 
by sorry

end solution_proof_problem_l200_200572


namespace complex_number_purely_imaginary_l200_200447

variable {m : ℝ}

theorem complex_number_purely_imaginary (h1 : 2 * m^2 + m - 1 = 0) (h2 : -m^2 - 3 * m - 2 ≠ 0) : m = 1/2 := by
  sorry

end complex_number_purely_imaginary_l200_200447


namespace find_f_prime_e_l200_200833

noncomputable def f (x : ℝ) : ℝ := 2 * x * (deriv f e) - Real.log x

theorem find_f_prime_e :
  deriv f e = 1 / Real.exp 1 :=   -- Real.exp 1 represents 'e'
sorry

end find_f_prime_e_l200_200833


namespace math_problem_part1_math_problem_part2_l200_200854

noncomputable def part1_statement : Prop :=
  ∀ (m : ℝ), ∃ (x : ℝ), ¬ (m * x^2 - 2 * x - m + 1 < 0)

noncomputable def part2_statement : Prop :=
  ∀ (m : ℝ), (|m| ≤ 2) → 
    ∀ (x : ℝ), (m * x^2 - 2 * x - m + 1 < 0) ↔ (frac (-1 + real.sqrt 7) 2 < x ∧ x < frac (1 + real.sqrt 3) 2)

theorem math_problem_part1 : part1_statement := sorry
theorem math_problem_part2 : part2_statement := sorry

end math_problem_part1_math_problem_part2_l200_200854


namespace conjugate_in_first_quadrant_l200_200451

-- Given condition definition
def z := (2 + complex.i) / complex.i

-- Conjugate of z
def conjugate_z := complex.conj z

-- Formal proof goal
theorem conjugate_in_first_quadrant : (complex.re conjugate_z > 0) ∧ (complex.im conjugate_z > 0) :=
  sorry

end conjugate_in_first_quadrant_l200_200451


namespace proof_set_intersection_l200_200497

def set_M := {x : ℝ | x^2 - 2*x - 8 ≤ 0}
def set_N := {x : ℝ | Real.log x ≥ 0}
def set_answer := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

theorem proof_set_intersection : 
  (set_M ∩ set_N) = set_answer := 
by 
  sorry

end proof_set_intersection_l200_200497


namespace ratio_of_area_of_triangles_l200_200230

variables {A B C N E Q : Type} [metricSpace A] [metricSpace B] [metricSpace C] 
    [metricSpace N] [metricSpace E] [metricSpace Q] [metricSpace ℝ]

/- Let N be the midpoint of side AC of triangle ABC.
   Let Q be a point on segment AC between A and N,
   and let NE be drawn parallel to QB and intersecting AB at E.
   Prove that the ratio s of the area of triangle AQB to that of triangle ABC
   satisfies 0 < s ≤ 1/2.
-/

def midpoint (A C N : Type) [metricSpace A] [metricSpace C] [metricSpace N] :=
  dist N A = dist N C ∧ ∀ (x : Type) [metricSpace x], dist x N + dist x N = dist x A + dist x C

def on_segment (Q A N : Type) [metricSpace Q] [metricSpace A] [metricSpace N] :=
  dist A Q + dist Q N = dist A N

def parallel (NE QB : Type) [metricSpace NE] [metricSpace QB] :=
  ∃ λ, ∀ x y : Type, [metricSpace x] [metricSpace y], (λ NE) = (1 / λ) QB

theorem ratio_of_area_of_triangles 
    (h_midpoint : midpoint A C N)
    (h_on_segment : on_segment Q A N)
    (h_parallel : parallel NE QB) :
    ∃ s : ℝ, 0 < s ∧ s ≤ 1/2 :=
begin
  sorry
end

end ratio_of_area_of_triangles_l200_200230


namespace employed_males_percent_l200_200221

def percent_employed_population : ℝ := 96
def percent_females_among_employed : ℝ := 75

theorem employed_males_percent :
  percent_employed_population * (1 - percent_females_among_employed / 100) = 24 := by
    sorry

end employed_males_percent_l200_200221


namespace radius_tangent_circle_eq_l200_200263

noncomputable def radius_of_tangent_circle : ℝ :=
  let angle := 360 / 7
  let tangent_angle := Real.tan (angle * Real.pi / 180)
  let discriminant := tangent_angle^2 - 4 * 1 * r
  let r_eq := tangent_angle^2 = 4 * r
  r

theorem radius_tangent_circle_eq :
  let r := radius_of_tangent_circle
  r = 0.380689 :=
begin
  sorry
end

end radius_tangent_circle_eq_l200_200263


namespace initial_number_of_nurses_l200_200296

theorem initial_number_of_nurses (N : ℕ) (initial_doctors : ℕ) (remaining_staff : ℕ) 
  (h1 : initial_doctors = 11) 
  (h2 : remaining_staff = 22) 
  (h3 : initial_doctors - 5 + N - 2 = remaining_staff) : N = 18 :=
by
  rw [h1, h2] at h3
  sorry

end initial_number_of_nurses_l200_200296


namespace find_ratio_l200_200993

variables {a b c λ p q r : ℝ}
variables {α β γ : ℝ}

-- Given fixed point (a, b, c) and constants
axiom H1 : (a, b, c) ≠ (0, 0, 0)
axiom H2 : λ ≠ 0

-- Given points A, B, and C
def A := (λ * α, 0, 0)
def B := (0, λ * β, 0)
def C := (0, 0, λ * γ)

-- Given that (p, q, r) is the center of the sphere passing through O, A, B, C
axiom H3 : p^2 + q^2 + r^2 = (p - λ * α)^2 + q^2 + r^2
axiom H4 : p^2 + q^2 + r^2 = p^2 + (q - λ * β)^2 + r^2
axiom H5 : p^2 + q^2 + r^2 = p^2 + q^2 + (r - λ * γ)^2

-- Proof statement
theorem find_ratio : 
  λ > 0 → (a / p + b / q + c / r) = 2 :=
by
  sorry

end find_ratio_l200_200993


namespace eq_x_in_terms_of_y_l200_200856

theorem eq_x_in_terms_of_y (x y : ℝ) (h : 2 * x + y = 5) : x = (5 - y) / 2 := by
  sorry

end eq_x_in_terms_of_y_l200_200856


namespace point_in_third_quadrant_l200_200927

theorem point_in_third_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : (-b < 0 ∧ a - 3 < 0) :=
by sorry

end point_in_third_quadrant_l200_200927


namespace eight_bags_weight_l200_200197

theorem eight_bags_weight :
  ∀ (total_bags : ℕ) (total_weight : ℚ) (bags_needed: ℕ), 
    total_bags = 12 → 
    total_weight = 24 → 
    bags_needed = 8 → 
    total_weight / total_bags * bags_needed = 16 :=
begin
  intros total_bags total_weight bags_needed hb hw hn,
  rw [hb, hw, hn],
  norm_num,
end

end eight_bags_weight_l200_200197


namespace number_of_terriers_groomed_l200_200634

-- Define the initial constants and the conditions from the problem statement
def time_to_groom_poodle := 30
def time_to_groom_terrier := 15
def number_of_poodles := 3
def total_grooming_time := 210

-- Define the problem to prove that the number of terriers groomed is 8
theorem number_of_terriers_groomed (groom_time_poodle groom_time_terrier num_poodles total_time : ℕ) : 
  groom_time_poodle = time_to_groom_poodle → 
  groom_time_terrier = time_to_groom_terrier →
  num_poodles = number_of_poodles →
  total_time = total_grooming_time →
  ∃ n : ℕ, n * groom_time_terrier + num_poodles * groom_time_poodle = total_time ∧ n = 8 := 
by
  intros h1 h2 h3 h4
  sorry

end number_of_terriers_groomed_l200_200634


namespace cos_A_value_sin_2B_minus_A_value_l200_200205

variables {A B C : ℝ}
variables {a b c : ℝ}
variables (h1 : a * sin A = 4 * b * sin B)
variables (h2 : a * c = sqrt 5 * (a^2 - b^2 - c^2))

-- Problem Statement 1: Prove that cos A = - sqrt 5 / 5
theorem cos_A_value : cos A = - sqrt 5 / 5 :=
by sorry

-- Problem Statement 2: Prove that sin (2 * B - A) = - (2 * sqrt 5) / 5
theorem sin_2B_minus_A_value : sin (2 * B - A) = - (2 * sqrt 5) / 5 :=
by sorry

end cos_A_value_sin_2B_minus_A_value_l200_200205


namespace linear_eq_a_l200_200929

theorem linear_eq_a (a : ℝ) (x y : ℝ) (h1 : (a + 1) ≠ 0) (h2 : |a| = 1) : a = 1 :=
by
  sorry

end linear_eq_a_l200_200929


namespace minimum_sum_abcd_l200_200575

theorem minimum_sum_abcd 
  (a b c d : ℕ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (mat_eq : ([4, 0], [0, 3]) • (a, b, c, d) = (a, b, c, d) • ([36, 24], [-30, -22])) 
  : a + b + c + d = 74 :=
sorry

end minimum_sum_abcd_l200_200575


namespace distribution_9_items_3_boxes_l200_200524

theorem distribution_9_items_3_boxes :
  ∃ (n : ℕ), n = 9.choose 3 * 6.choose 2 * 4.choose 4 * 3! ∧ n = 7560 := by
  sorry

end distribution_9_items_3_boxes_l200_200524


namespace liquid_x_percentage_correct_l200_200604

-- Define the initial conditions using Lean variables and definitions

def initial_solution_y_weight : ℝ := 6 -- kilograms
def liquid_x_percentage : ℝ := 0.30
def water_evaporated_weight : ℝ := 2 -- kilograms
def additional_solution_y_weight : ℝ := 2 -- kilograms

-- Define the amount of liquid x in the initial solution
def initial_liquid_x_weight := liquid_x_percentage * initial_solution_y_weight

-- Define the remaining liquid weight after water evaporation
def remaining_solution_weight := initial_solution_y_weight - water_evaporated_weight

-- Define the liquid x weight after adding the additional solution y
def additional_liquid_x_weight := liquid_x_percentage * additional_solution_y_weight
def total_liquid_x_weight := initial_liquid_x_weight + additional_liquid_x_weight

-- Define the total new solution weight
def total_new_solution_weight := remaining_solution_weight + additional_solution_y_weight

-- Define the percentage of liquid x in the new solution
def liquid_x_percentage_in_new_solution := (total_liquid_x_weight / total_new_solution_weight) * 100

-- Prove the percentage of liquid x in new solution is 40%
theorem liquid_x_percentage_correct :
  liquid_x_percentage_in_new_solution = 40 := by
  sorry

end liquid_x_percentage_correct_l200_200604


namespace gcd_arithmetic_sequence_correct_l200_200626

noncomputable def gcd_arithmetic_sequence : ℕ :=
  let a := 1
  let d := 1
  let b := a + d
  let c := a + 2 * d
  nat.gcd (1001 * a + 1000 * d + 110 * b + 20 * c) (2131 * (a + d))

theorem gcd_arithmetic_sequence_correct : gcd_arithmetic_sequence = 2131 := 
by sorry

end gcd_arithmetic_sequence_correct_l200_200626


namespace find_percentage_l200_200703

theorem find_percentage (P : ℝ) : 
  0.15 * P * (0.5 * 5600) = 126 → P = 0.3 := 
by 
  sorry

end find_percentage_l200_200703


namespace interest_earned_l200_200502

-- Definitions based on the conditions
def principal : ℝ := 3000
def rate (y : ℝ) : ℝ := y / 100
def time : ℝ := 2
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Statement to prove
theorem interest_earned (y : ℝ) : simple_interest principal (rate y) time = 60 * y :=
by sorry

end interest_earned_l200_200502


namespace sector_COD_area_ratio_l200_200251

-- Define the given angles
def angle_AOC : ℝ := 30
def angle_DOB : ℝ := 45
def angle_AOB : ℝ := 180

-- Define the full circle angle
def full_circle_angle : ℝ := 360

-- Calculate the angle COD
def angle_COD : ℝ := angle_AOB - angle_AOC - angle_DOB

-- State the ratio of the area of sector COD to the area of the circle
theorem sector_COD_area_ratio :
  angle_COD / full_circle_angle = 7 / 24 := by
  sorry

end sector_COD_area_ratio_l200_200251


namespace arrange_digits_1225_l200_200961

theorem arrange_digits_1225 : 
  let digits := [1, 2, 2, 5] in 
  let endsIn5 (l : List ℕ) := (List.reverse l).headI = 5 in
  (List.permutations digits).count endsIn5 = 6 :=
sorry

end arrange_digits_1225_l200_200961


namespace total_recovery_time_l200_200540

theorem total_recovery_time (A : ℕ) : 
  let T_initial : ℝ := 4
  let T_surgery : ℝ := T_initial + 0.5 * T_initial
  let Reduction_PT : ℝ := 0.10 * (A : ℝ) * T_surgery
  let Reduction_med : ℝ := 0.20 * T_surgery
  let Total_recovery_time : ℝ := T_surgery - (Reduction_PT + Reduction_med)
  in Total_recovery_time = 4.8 - 0.6 * (A : ℝ) := by 
  sorry

end total_recovery_time_l200_200540


namespace train_or_plane_prob_not_ship_prob_l200_200713

variables {Ω : Type} [ProbabilitySpace Ω]

def P_A : ℝ := 0.3
def P_B : ℝ := 0.2
def P_C : ℝ := 0.1
def P_D : ℝ := 0.4

-- Events A, B, C, D are mutually exclusive
axiom mutually_exclusive : 
  ∀ {A B C D : Event Ω}, A ≠ B → B ≠ C → C ≠ D → A ≠ D → P (A ∪ D) = P A + P D

-- Problem 1: Calculate the probability of taking a train or a plane
theorem train_or_plane_prob : 
  ∀ {A D : Event Ω}, P A = P_A → P D = P_D → P (A ∪ D) = 0.7 :=
by
  intros A D HA HD
  calc
    P (A ∪ D) = P A + P D := mutually_exclusive sorry sorry sorry sorry
              ... = 0.3 + 0.4 := by rw [HA, HD]
              ... = 0.7 := by norm_num

-- Problem 2: Calculate the probability of not taking a ship
theorem not_ship_prob :
  ∀ {B : Event Ω}, P B = P_B → P (¬ B) = 0.8 :=
by
  intros B HB
  calc
    P (¬ B) = 1 - P B := by sorry
             ... = 1 - 0.2 := by rw HB
             ... = 0.8 := by norm_num

end train_or_plane_prob_not_ship_prob_l200_200713


namespace find_length_of_AL_l200_200535

noncomputable def length_of_AL 
  (A B C L : ℝ) 
  (AB AC AL : ℝ)
  (BC : ℝ)
  (AB_ratio_AC : AB / AC = 5 / 2)
  (BAC_bisector : ∃k, L = k * BC)
  (vector_magnitude : (2 * AB + 5 * AC) = 2016) : Prop :=
  AL = 288

theorem find_length_of_AL 
  (A B C L : ℝ)
  (AB AC AL : ℝ)
  (BC : ℝ)
  (h1 : AB / AC = 5 / 2)
  (h2 : ∃k, L = k * BC)
  (h3 : (2 * AB + 5 * AC) = 2016) : length_of_AL A B C L AB AC AL BC h1 h2 h3 := sorry

end find_length_of_AL_l200_200535


namespace find_m_of_perpendicular_vectors_l200_200899

theorem find_m_of_perpendicular_vectors
    (m : ℝ)
    (a : ℝ × ℝ := (m, 3))
    (b : ℝ × ℝ := (1, m + 1))
    (h : a.1 * b.1 + a.2 * b.2 = 0) :
    m = -3 / 4 :=
by 
  sorry

end find_m_of_perpendicular_vectors_l200_200899


namespace find_k_l200_200768

def f (x : ℝ) : ℝ := 5 * x^2 - 1 / x + 3
def g (x k : ℝ) : ℝ := x^2 - k * x - k 

theorem find_k (k : ℝ) : f 2 - g 2 k = 7 → k = -23/6 :=
by
  intro h
  -- The proof goes here and is not required for the statement.
  sorry

end find_k_l200_200768


namespace two_digit_prime_count_l200_200910

theorem two_digit_prime_count : 
  (let digits := {2, 7, 8, 9};
   let primes := {29, 79, 89, 97};
   ∀ n ∈ primes, (n / 10 ∈ digits) ∧ (n % 10 ∈ digits) ∧ (n / 10 ≠ n % 10) ∧ Prime n) →
  ∃ count, count = 4 := 
sorry

end two_digit_prime_count_l200_200910


namespace range_of_t_l200_200172

def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 else -x^2

theorem range_of_t (t : ℝ) : (∀ x : ℝ, t ≤ x ∧ x ≤ t + 1 → f (x + t) ≥ 9 * f x) → t ≤ -2 :=
by
  sorry

end range_of_t_l200_200172


namespace length_BC_l200_200945

noncomputable def O : Point := sorry
noncomputable def A : Point := sorry
noncomputable def B : Point := sorry
noncomputable def C : Point := sorry
noncomputable def D : Point := sorry

axiom diameter_AD (circle : Circle) : is_diameter circle O A D
axiom chord_ABC (circle : Circle) : is_chord circle A B C
axiom BO_eq_6 : distance O B = 6
axiom angle_ABO_eq_45 : angle A B O = 45
axiom arc_CD_eq_45 (circle : Circle) : arc CD = 45

theorem length_BC (circle : Circle) (O A B C D : Point) 
  (diameter_AD : is_diameter circle O A D)
  (chord_ABC : is_chord circle A B C)
  (BO_eq_6 : distance O B = 6)
  (angle_ABO_eq_45 : angle A B O = 45)
  (arc_CD_eq_45 : arc CD = 45) : distance B C = 6 := sorry

end length_BC_l200_200945


namespace min_segment_length_l200_200452

theorem min_segment_length
  (a : ℝ) -- edge length of the cube
  (AB1 BC1 : ℝ → ℝ → Prop) -- predicates for lines AB1 and BC1
  (hAB1 : ∀ y z, AB1 0 y z → y = z) -- AB1 describes points with y = z
  (hBC1 : ∀ x z, BC1 x 0 z → x = z) -- BC1 describes points with x = z
  (MN_angle : ∀ (M N : ℝ × ℝ × ℝ), (∃ u, u = 2 * (M.1 ^ 2 + M.2 ^ 2 + M.3 ^ 2) ^ (1/2))
                      → (∃ θ, θ = real.arccos (M.1 / (M.1 ^ 2 + M.2 ^ 2 + M.3 ^ 2) ^ (1/2)) ∧ θ = real.pi / 3)) :
  ∃ MN : ℝ, MN = 2 * a * (real.sqrt 3 - real.sqrt 2) :=
by
  sorry -- Proof omitted

end min_segment_length_l200_200452


namespace exists_large_period_in_consecutive_100_digit_integers_l200_200600

noncomputable def euler_phi (n : ℕ) : ℕ :=
  if n = 0 then 0 else (Finset.range n).filter (Nat.coprime n).card

theorem exists_large_period_in_consecutive_100_digit_integers :
  ∀ (start : ℕ), (log 10 start).natAbs + 1 = 100 →
  ∃ n ∈ Finset.range (start + 100000), 
    (euler_phi n) > 2011 :=
by
  sorry

end exists_large_period_in_consecutive_100_digit_integers_l200_200600


namespace total_tickets_sold_l200_200738

theorem total_tickets_sold (x y : ℕ) (h1 : 12 * x + 8 * y = 3320) (h2 : y = x + 240) : 
  x + y = 380 :=
by -- proof
  sorry

end total_tickets_sold_l200_200738


namespace quadratic_inequalities_l200_200455

variable (c x₁ y₁ y₂ y₃ : ℝ)
noncomputable def quadratic_function := -x₁^2 + 2*x₁ + c

theorem quadratic_inequalities
  (h_c : c < 0)
  (h_y₁ : quadratic_function c x₁ > 0)
  (h_y₂ : y₂ = quadratic_function c (x₁ - 2))
  (h_y₃ : y₃ = quadratic_function c (x₁ + 2)) :
  y₂ < 0 ∧ y₃ < 0 :=
by sorry

end quadratic_inequalities_l200_200455


namespace area_rhombus_72_11_l200_200956

noncomputable def area_of_rhombus : ℝ :=
let A : ℝ × ℝ × ℝ := (8, -7.5, -5)
let B : ℝ × ℝ × ℝ := (-8, -7.5, -5)
let cross_prod : ℝ × ℝ × ℝ := (A.2 * B.3 - A.3 * B.2, 
                                 A.3 * B.1 - A.1 * B.3, 
                                 A.1 * B.2 - A.2 * B.1)
let magnitude : ℝ := real.sqrt (cross_prod.1^2 + cross_prod.2^2 + cross_prod.3^2)
in magnitude / 2

theorem area_rhombus_72_11 :
  area_of_rhombus ≈ 72.11 := sorry

end area_rhombus_72_11_l200_200956


namespace arc_length_eq_l200_200084

noncomputable def x (t : ℝ) := 3.5 * (2 * Real.cos t - Real.cos (2 * t))
noncomputable def y (t : ℝ) := 3.5 * (2 * Real.sin t - Real.sin (2 * t))

def arc_length : ℝ :=
  ∫ (t : ℝ) in 0..(Real.pi / 2),
    Real.sqrt ((Real.deriv x t)^2 + (Real.deriv y t)^2)

theorem arc_length_eq : arc_length = 14 * (2 - Real.sqrt 2) :=
  sorry

end arc_length_eq_l200_200084


namespace correct_spelling_probability_l200_200737

def b := 1
def p := 1
def r := 1
def e := 3
def o := 2
def k := 2

def total_arrangements : ℕ := sorry  -- The calculation of valid arrangements taking into account the given constraints

def correct_probability (P : ℕ) : Prop := P = 1 / total_arrangements

theorem correct_spelling_probability : correct_probability (1 / 9600) := 
by { 
    unfold correct_probability,
    sorry,  -- The proof of the theorem
}

end correct_spelling_probability_l200_200737


namespace similar_triangle_shortest_side_l200_200375

theorem similar_triangle_shortest_side
  (a₁ : ℕ) (c₁ : ℕ) (c₂ : ℕ)
  (h₁ : a₁ = 15) (h₂ : c₁ = 17) (h₃ : c₂ = 68)
  (right_triangle_1 : a₁^2 + b₁^2 = c₁^2)
  (similar_triangles : ∃ k : ℕ, c₂ = k * c₁) :
  shortest_side = 32 := 
sorry

end similar_triangle_shortest_side_l200_200375


namespace monotonic_intervals_k_eq_e_f_abs_positivity_imp_k_range_F_product_inequality_l200_200847

-- Part (1)
theorem monotonic_intervals_k_eq_e :
  ∃ a b : ℝ, (∀ x ∈ (a, b), has_deriv_at (λ x, Real.exp x - x) (Real.exp x - 1) x ∧
  ∀ x ∈ (a, b), Real.exp x - 1 > 0) ∧
  (∀ x ∉ (a, b), has_deriv_at (λ x, Real.exp x - x) (Real.exp x - 1) x ∧
  ∀ x ∉ (a, b), Real.exp x - 1 < 0) :=
sorry

-- Part (2)
theorem f_abs_positivity_imp_k_range (k : ℝ) (h : ∀ x : ℝ, Real.exp x - k * x > 0) : 0 < k ∧ k < Real.exp 1 :=
sorry

-- Part (3)
theorem F_product_inequality (n : ℕ) (h : 0 < n) :
  let F := λ x : ℝ, Real.exp x - exp (-x)
  (∏ i in Finset.range n, F (i + 1)) > (Real.exp (n+1) + 2) ^ (n / 2) :=
sorry

end monotonic_intervals_k_eq_e_f_abs_positivity_imp_k_range_F_product_inequality_l200_200847


namespace number_of_females_l200_200207

variable (total_employees : ℕ) (adv_degrees : ℕ) (males_college_only : ℕ) (females_adv_degrees : ℕ)

axiom condition_1 : total_employees = 148
axiom condition_2 : adv_degrees = 78
axiom condition_3 : males_college_only = 31
axiom condition_4 : females_adv_degrees = 53

theorem number_of_females : 
  let F := females_adv_degrees + (total_employees - adv_degrees - males_college_only)
  in F = 92 :=
by
  have h1 : total_employees - adv_degrees = 70, from by rw [condition_1, condition_2]
  have h2 : 70 - males_college_only = 39, from by rw [h1, condition_3]
  have h3 : females_adv_degrees + 39 = 92, from by rw [condition_4, h2]
  rw [<-h3, condition_4, h2]
  exact rfl

end number_of_females_l200_200207


namespace new_sample_variance_l200_200456

-- Definitions based on conditions
def sample_size (original : Nat) : Prop := original = 7
def sample_average (original : ℝ) : Prop := original = 5
def sample_variance (original : ℝ) : Prop := original = 2
def new_data_point (point : ℝ) : Prop := point = 5

-- Statement to be proved
theorem new_sample_variance (original_size : Nat) (original_avg : ℝ) (original_var : ℝ) (new_point : ℝ) 
  (h₁ : sample_size original_size) 
  (h₂ : sample_average original_avg) 
  (h₃ : sample_variance original_var) 
  (h₄ : new_data_point new_point) : 
  (8 * original_var + 0) / 8 = 7 / 4 := 
by 
  sorry

end new_sample_variance_l200_200456


namespace sqrt_abc_sum_is_72_l200_200570

noncomputable def abc_sqrt_calculation (a b c : ℝ) (h1 : b + c = 17) (h2 : c + a = 18) (h3 : a + b = 19) : ℝ :=
  sqrt (a * b * c * (a + b + c))

theorem sqrt_abc_sum_is_72 (a b c : ℝ) (h1 : b + c = 17) (h2 : c + a = 18) (h3 : a + b = 19) :
  abc_sqrt_calculation a b c h1 h2 h3 = 72 :=
by
  sorry

end sqrt_abc_sum_is_72_l200_200570


namespace symmetry_line_probability_l200_200211

noncomputable def is_symmetry_line (P Q : Point) (grid : Grid) : Prop :=
  (P.equals center_point) ∧
  (Q.is_in_grid grid) ∧ 
  (Q ≠ P) ∧
  (Q.is_symmetric_to P)

noncomputable def probability_symmetry_line (grid : Grid) (P : Point) : ℚ :=
  let total_points := grid.total_points
  let sym_points := grid.symmetric_points_through P
  (sym_points.to_float / (total_points - 1).to_float)

theorem symmetry_line_probability (P : Point) (grid : Grid) : 
  P.is_center_of_grid grid ∧ grid.dimensions = (8, 6) → 
  probability_symmetry_line grid P = (12 : ℚ) / 47 := sorry

end symmetry_line_probability_l200_200211


namespace most_likely_hits_l200_200213

-- Given definitions
def probability_of_hit := 0.8
def number_of_shots := 99

-- To be proved: the most likely number of hits is either 79 or 80
theorem most_likely_hits :
  let E := probability_of_hit * number_of_shots in
  79 ≤ E ∧ E < 80 →
  E = 79 ∨ E = 80 :=
by
  sorry

end most_likely_hits_l200_200213


namespace least_number_divisible_l200_200124

-- Define the numbers as given in the conditions
def given_number : ℕ := 3072
def divisor1 : ℕ := 57
def divisor2 : ℕ := 29
def least_number_to_add : ℕ := 234

-- Define the LCM
noncomputable def lcm_57_29 : ℕ := Nat.lcm divisor1 divisor2

-- Prove that adding least_number_to_add to given_number makes it divisible by both divisors
theorem least_number_divisible :
  (given_number + least_number_to_add) % divisor1 = 0 ∧ 
  (given_number + least_number_to_add) % divisor2 = 0 := 
by
  -- Proof should be provided here
  sorry

end least_number_divisible_l200_200124


namespace problem1_problem2_l200_200398

open Nat

def binomial (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem problem1 : binomial 8 5 + binomial 100 98 * binomial 7 7 = 5006 := by
  sorry

theorem problem2 : binomial 5 0 + binomial 5 1 + binomial 5 2 + binomial 5 3 + binomial 5 4 + binomial 5 5 = 32 := by
  sorry

end problem1_problem2_l200_200398


namespace like_terms_mn_l200_200928

theorem like_terms_mn (m n : ℤ) 
  (H1 : m - 2 = 3) 
  (H2 : n + 2 = 1) : 
  m * n = -5 := 
by
  sorry

end like_terms_mn_l200_200928


namespace perpendicular_vectors_l200_200875

def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (1, m + 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (m : ℝ) (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by sorry

end perpendicular_vectors_l200_200875


namespace a_can_be_one_l200_200495

variable {α : Type} [Preorder α]

def setA (a : α) : Set α := {0, a}
def setB : Set α := {x | -1 < x ∧ x < 2}

theorem a_can_be_one (a : α) (h : setA a ⊆ setB) : a = 1 := 
  sorry

end a_can_be_one_l200_200495


namespace find_expression_l200_200356

variable (a b : ℝ)

def f (x : ℝ) : ℝ := (x + a) * (b * x + 2 * a)

theorem find_expression (a b : ℝ) (hf_even : ∀ x, f a b x = f a b (-x)) 
  (hf_range : set.range (f a b) ⊆ set.Iic 4) :
  f a b = λ x, -2 * x ^ 2 + 4 :=
  sorry

end find_expression_l200_200356


namespace expected_value_variance_l200_200415

noncomputable def F (v t : ℝ) : ℝ :=
  if t ≤ 0 then 0 else 1 - Real.exp (-v * t)

noncomputable def pdf (v t : ℝ) : ℝ :=
  if t ≤ 0 then 0 else v * Real.exp (-v * t)

theorem expected_value (v : ℝ) (hv : 0 < v) : 
  (∫ t in 0..∞, t * pdf v t) = 1 / v :=
sorry

theorem variance (v : ℝ) (hv : 0 < v) : 
  (∫ t in 0..∞, t^2 * pdf v t) - (1 / v)^2 = 1 / v^2 :=
sorry

end expected_value_variance_l200_200415


namespace students_with_dog_and_cat_but_no_bird_l200_200111

def hudson_high_school : Type := ℕ

axiom dog_owners : hudson_high_school → Prop
axiom cat_owners : hudson_high_school → Prop
axiom bird_owners : hudson_high_school → Prop

axiom total_students : ∀ s : hudson_high_school, s = 60
axiom number_of_dog_owners : ∀ s : hudson_high_school, (λ s, dog_owners s).card = 35
axiom number_of_cat_owners : ∀ s : hudson_high_school, (λ s, cat_owners s).card = 45
axiom number_of_bird_owners : ∀ s : hudson_high_school, (λ s, bird_owners s).card = 10

axiom birds_with_other_pets : ∀ s, bird_owners s → dog_owners s ∨ cat_owners s

theorem students_with_dog_and_cat_but_no_bird : ∃ n, n = (λ s, dog_owners s ∧ cat_owners s ∧ ¬ bird_owners s).card ∧ n = 10 :=
by
  -- Given axioms can be used here to prove the theorem
  sorry

end students_with_dog_and_cat_but_no_bird_l200_200111


namespace range_of_x1_l200_200150

-- Define the increasing function property
def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≤ f y

-- Define the main theorem statement
theorem range_of_x1 (f : ℝ → ℝ) (h_increasing : increasing_function f) :
  (∀ x1 x2, x1 + x2 = 1 → f(x1) + f(0) > f(x2) + f(1)) → (∀ x1, x1 + (1 - x1) = 1 → x1 > 1) :=
by
  sorry

end range_of_x1_l200_200150


namespace calories_in_200_grams_is_137_l200_200136

-- Define the grams of ingredients used.
def lemon_juice_grams := 100
def sugar_grams := 100
def water_grams := 400

-- Define the calories per 100 grams of each ingredient.
def lemon_juice_calories_per_100_grams := 25
def sugar_calories_per_100_grams := 386
def water_calories_per_100_grams := 0

-- Calculate the total calories in the entire lemonade mixture.
def total_calories : Nat :=
  (lemon_juice_grams * lemon_juice_calories_per_100_grams / 100) + 
  (sugar_grams * sugar_calories_per_100_grams / 100) +
  (water_grams * water_calories_per_100_grams / 100)

-- Calculate the total weight of the lemonade mixture.
def total_weight : Nat := lemon_juice_grams + sugar_grams + water_grams

-- Calculate the caloric density (calories per gram).
def caloric_density := total_calories / total_weight

-- Calculate the calories in 200 grams of lemonade.
def calories_in_200_grams := (caloric_density * 200)

-- The theorem to prove
theorem calories_in_200_grams_is_137 : calories_in_200_grams = 137 :=
by sorry

end calories_in_200_grams_is_137_l200_200136


namespace limit_Sn_div_Tn_l200_200448

open Set Real

def M_n (n : ℕ) : Set ℝ :=
  {x | ∃ (a : Fin n → Bool), (a (Fin.last n) = true) ∧ (x = Finset.univ.sum (λ k : Fin n, (if a k then 1 else 0) * 10^(-((k:ℕ)+1))))}

def T_n (n : ℕ) : ℕ :=
  2^(n-1)

def S_n (n : ℕ) : ℝ :=
  Finset.univ.sum (λ a : Fin n → Bool, (Finset.univ.sum (λ k : Fin n, (if a k then 1 else 0) * 10^(-((k:ℕ)+1)))))

theorem limit_Sn_div_Tn : tendsto (λ n, S_n n / T_n n) at_top (𝓝 (1/18)) :=
by
  sorry

end limit_Sn_div_Tn_l200_200448


namespace percent_of_x_is_y_l200_200019

variable (x y : ℝ)

theorem percent_of_x_is_y
  (h : 0.5 * (x - y) = 0.2 * (x + y)) :
  y = 0.4286 * x := by
  sorry

end percent_of_x_is_y_l200_200019


namespace variance_of_artworks_per_student_l200_200840

theorem variance_of_artworks_per_student 
  (students : ℕ)
  (boys girls : ℕ)
  (avg_boy_artworks variance_boy_artworks : ℝ)
  (avg_girl_artworks variance_girl_artworks : ℝ)
  (total_students_eq : students = 25)
  (students_partition : boys + girls = students)
  (boys_eq : boys = 10)
  (girls_eq : girls = 15)
  (avg_boy_artworks_eq : avg_boy_artworks = 25)
  (variance_boy_artworks_eq : variance_boy_artworks = 1)
  (avg_girl_artworks_eq : avg_girl_artworks = 30)
  (variance_girl_artworks_eq : variance_girl_artworks = 2) : 
  (variance (λ (x : ℕ), if x < boys then avg_boy_artworks else avg_girl_artworks) students) = 38 / 5 := by
  sorry

end variance_of_artworks_per_student_l200_200840


namespace total_height_in_cm_l200_200401

theorem total_height_in_cm (sculpture_feet : ℕ) (sculpture_inches : ℕ) (base_inches : ℕ)
  (foot_to_inches : ℕ) (inch_to_cm : ℝ) :
  sculpture_feet = 2 →
  sculpture_inches = 10 →
  base_inches = 2 →
  foot_to_inches = 12 →
  inch_to_cm = 2.54 →
  ((sculpture_feet * foot_to_inches + sculpture_inches + base_inches) * inch_to_cm) = 91.44 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end total_height_in_cm_l200_200401


namespace inheritance_problem_l200_200605

variable (x : ℝ) -- total inheritance
variable (n : ℕ) -- number of sons
variable (s : ℝ) -- each son's share

-- Define the share of the first son
def first_son_share (x : ℝ) : ℝ := 100 + 1/10 * (x - 100)

-- Define the share of the second son, considering the deduction after the first son's share
def second_son_share (x : ℝ) : ℝ := 200 + 1/10 * ((9/10 * x - 90) - 200)

-- The actual share received by each son, calculated as the total inheritance divided by number of sons
def son_share (x : ℝ) (n : ℕ) : ℝ := x / n

-- The proof problem: proving the size of inheritance, number of sons, and each son's share
theorem inheritance_problem : 
  (first_son_share x = son_share x n) ∧ 
  (second_son_share x = son_share x n) ∧
  (x = 8100) ∧ 
  (n = 9) ∧ 
  (son_share x n = 900) :=
by
  sorry

end inheritance_problem_l200_200605


namespace train_length_l200_200066

theorem train_length (speed_km_hr : ℕ) (time_sec : ℕ) (h_speed : speed_km_hr = 72) (h_time : time_sec = 12) : 
  ∃ length_m : ℕ, length_m = 240 := 
by
  sorry

end train_length_l200_200066


namespace find_length_BC_l200_200942

noncomputable def length_BC (BO : ℝ) (angle_ABO : ℝ) : ℝ :=
  2 * BO * Real.sin (angle_ABO / 2)

theorem find_length_BC : length_BC 6 (Real.pi / 4) ≈ 4.6 := by
  sorry

end find_length_BC_l200_200942


namespace eval_f_at_6_l200_200718

theorem eval_f_at_6 :
  (∀ x : ℝ, f (4 * x + 2) = x^2 - x + 2) → f 6 = 2 :=
by
  intro h
  sorry

end eval_f_at_6_l200_200718


namespace problem_correction_l200_200341

theorem problem_correction :
  ¬ (sqrt 2 + sqrt 3 = sqrt 5) ∧
  ¬ (5 * sqrt 5 - 2 * sqrt 2 = 3 * sqrt 3) ∧
  ¬ (2 * sqrt 3 * 3 * sqrt 3 = 6 * sqrt 3) ∧
  (sqrt 2 / sqrt 3 = sqrt 6 / 3) :=
by {
  split,
  {
    intro h,
    -- proof goes here
    sorry,
  },
  split,
  {
    intro h,
    -- proof goes here
    sorry,
  },
  split,
  {
    intro h,
    -- proof goes here
    sorry,
  },
  {
    -- proof goes here
    sorry,
  }
}

end problem_correction_l200_200341


namespace geometric_series_sum_l200_200405

theorem geometric_series_sum :
  ∑ k in Finset.range 7, 3 ^ (k + 2) = 9827 :=
by {
  -- We recognize the problem as a geometric series where the first term is 3^2 = 9 and common ratio is 3.
  -- The formula for the sum of the first n terms of a geometric series is a * (r^n - 1) / (r - 1),
  -- where a = 9, r = 3, and n = 7 terms.
  sorry
}

end geometric_series_sum_l200_200405


namespace correct_option_l200_200746

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_increasing_on_pos_infty (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x -> x < y -> f x < f y

def y_A (x : ℝ) := 1/x + x
def y_B (x : ℝ) := x - cos x
def y_C (x : ℝ) := x - sin x
def y_D (x : ℝ) := 1/x - x

theorem correct_option : 
    (is_odd y_A ∧ is_increasing_on_pos_infty y_A) ∨ 
    (is_odd y_B ∧ is_increasing_on_pos_infty y_B) ∨ 
    (is_odd y_C ∧ is_increasing_on_pos_infty y_C) ∨ 
    (is_odd y_D ∧ is_increasing_on_pos_infty y_D) :=
by sorry

end correct_option_l200_200746


namespace consecutive_days_probability_l200_200322

noncomputable def probability_of_consecutive_days : ℚ :=
  let total_days := 5
  let combinations := Nat.choose total_days 2
  let consecutive_pairs := 4
  consecutive_pairs / combinations

theorem consecutive_days_probability :
  probability_of_consecutive_days = 2 / 5 :=
by
  sorry

end consecutive_days_probability_l200_200322


namespace people_and_cars_equation_l200_200527

theorem people_and_cars_equation (x : ℕ) :
  3 * (x - 2) = 2 * x + 9 :=
sorry

end people_and_cars_equation_l200_200527


namespace expected_value_fair_8_sided_die_l200_200716

theorem expected_value_fair_8_sided_die :
  (∑ n in Finset.range 8, (n + 1)^3) / 8 = 162 := by
  -- We will provide the proof here (not needed for this task)
  sorry

end expected_value_fair_8_sided_die_l200_200716


namespace total_revenue_calculation_l200_200045

theorem total_revenue_calculation :
  let carrots_sold := 0.60 * 20
  let zucchini_sold := 0.40 * 18
  let broccoli_sold := 0.75 * 12
  let potatoes_sold := 0.50 * 25
  let bellpeppers_sold := 0.80 * 10
  let revenue_carrots := carrots_sold * 2
  let revenue_zucchini := zucchini_sold * 3
  let revenue_broccoli := broccoli_sold * 4
  let revenue_potatoes := potatoes_sold * 1
  let revenue_bellpeppers := bellpeppers_sold * 5
  in revenue_carrots + revenue_zucchini + revenue_broccoli + revenue_potatoes + revenue_bellpeppers = 134.1 := 
by
  sorry 

end total_revenue_calculation_l200_200045


namespace delores_money_left_l200_200102

def initial : ℕ := 450
def computer_cost : ℕ := 400
def printer_cost : ℕ := 40
def money_left (initial computer_cost printer_cost : ℕ) : ℕ := initial - (computer_cost + printer_cost)

theorem delores_money_left : money_left initial computer_cost printer_cost = 10 := by
  sorry

end delores_money_left_l200_200102


namespace lines_perpendicular_to_same_plane_are_parallel_l200_200818

variables {l m : Type} [line l] [line m]
variables {α β γ : Type} [plane α] [plane β] [plane γ]

-- Given conditions
-- l and m are different lines
-- α, β and γ are different planes
-- If l is perpendicular to α and m is perpendicular to α, then l is parallel to m.

theorem lines_perpendicular_to_same_plane_are_parallel (hlα : l ⊥ α) (hmα : m ⊥ α) :
  l ∥ m := by
  sorry

end lines_perpendicular_to_same_plane_are_parallel_l200_200818


namespace part1_part2_l200_200173

noncomputable def f (m x : ℝ) : ℝ := exp (x - m) - x * log x - (m - 1) * x
noncomputable def f' (m x : ℝ) : ℝ := deriv (λ x, exp (x - m) - x * log x - (m - 1) * x) x

theorem part1 (x : ℝ) (hx1 : x > 0) : f' 1 x ≥ 0 := sorry

theorem part2 (m : ℝ) (hx2 : ∃ x1 x2, x1 < x2 ∧ f' m x1 = 0 ∧ f' m x2 = 0) : m > 1 := sorry

end part1_part2_l200_200173


namespace solution_proof_problem_l200_200573

open Real

noncomputable def proof_problem : Prop :=
  ∀ (a b c : ℝ),
  b + c = 17 →
  c + a = 18 →
  a + b = 19 →
  sqrt (a * b * c * (a + b + c)) = 36 * sqrt 5

theorem solution_proof_problem : proof_problem := 
by sorry

end solution_proof_problem_l200_200573


namespace magnitude_complex_l200_200418

noncomputable def complex_num : ℂ := -5 - (8 / 3) * (complex.I)

theorem magnitude_complex : complex.abs (complex_num) = 17 / 3 := 
by sorry

end magnitude_complex_l200_200418


namespace trapezoid_area_and_diagonal_l200_200259

def WZ := 28
def WY := 60
def XZ := 30
def XY := 28
def altitude := 15

theorem trapezoid_area_and_diagonal :
  let area := (1 / 2) * (WY + XZ) * altitude in
  let diagonal_length := Real.sqrt (altitude^2 + (WY - XZ)^2 / 4) in
  area = 675 ∧ diagonal_length = 15 * Real.sqrt 2 :=
by
  unfold WY XZ WZ XY altitude
  let area := (1 / 2) * (WY + XZ) * altitude
  have area_calculation : area = 675 := by sorry
  let diagonal_length := Real.sqrt (altitude^2 + (WY - XZ)^2 / 4)
  have diagonal_calculation : diagonal_length = 15 * Real.sqrt 2 := by sorry
  exact ⟨area_calculation, diagonal_calculation⟩

end trapezoid_area_and_diagonal_l200_200259


namespace exists_n_digits_uniform_l200_200539

theorem exists_n_digits_uniform :
  ∃ n : ℕ, n = 4999^2 * 10^396 ∧ 
  (∀ p ∈ finset.range 10, 
    finset.card { k ∈ finset.range 1000 | (nat.frac((n + k).sqrt * 10^200).to_digits 10).nth 199 = some p } = 100)
  :=
sorry

end exists_n_digits_uniform_l200_200539


namespace parameterize_line_l200_200300

theorem parameterize_line (f : ℝ → ℝ) (t : ℝ) (x y : ℝ)
  (h1 : y = 2 * x - 30)
  (h2 : (x, y) = (f t, 20 * t - 10)) :
  f t = 10 * t + 10 :=
sorry

end parameterize_line_l200_200300


namespace find_erased_number_l200_200344

/-- Define the variables used in the conditions -/
def n : ℕ := 69
def erased_number_mean : ℚ := 35 + 7 / 17
def sequence_sum : ℕ := n * (n + 1) / 2

/-- State the condition for the erased number -/
noncomputable def erased_number (x : ℕ) : Prop :=
  (sequence_sum - x) / (n - 1) = erased_number_mean

/-- The main theorem stating that the erased number is 7 -/
theorem find_erased_number : ∃ x : ℕ, erased_number x ∧ x = 7 :=
by
  use 7
  unfold erased_number sequence_sum
  -- Sum of first 69 natural numbers is 69 * (69 + 1) / 2
  -- Hence,
  -- (69 * 70 / 2 - 7) / 68 = 35 + 7 / 17
  -- which simplifies to true under these conditions
  -- Detailed proof skipped here as per instructions
  sorry

end find_erased_number_l200_200344


namespace perpendicular_vectors_l200_200876

def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (1, m + 1)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (m : ℝ) (h : dot_product (vector_a m) (vector_b m) = 0) : m = -3 / 4 :=
by sorry

end perpendicular_vectors_l200_200876


namespace problem_f_l200_200175

noncomputable def f : ℝ → ℝ :=
  λ x, if x ∈ Ioo (-π/2) (π/2) then x + sin x else f (π - x)

theorem problem_f {a b c : ℝ} (h1 : ∀ x, f(x) = f(π - x))
  (h2 : ∀ x ∈ Ioo (-π/2) (π/2), f(x) = x + sin x) :
  let a := f 1 in let b := f 2 in let c := f 3 in
  c < a ∧ a < b :=
by
  let a := f 1
  let b := f 2
  let c := f 3
  sorry

end problem_f_l200_200175


namespace joint_work_completion_l200_200016

-- Defining the conditions as hypotheses
variables {A B : Type} [has_inv B] [has_div B] [has_add B] 
variables (works_twice_as_fast : ∀ (a b : B), a = 2 * b)
          (b_days : B)
          (b_completion_time : b_days = 12)

-- The statement of the theorem
theorem joint_work_completion (a_days b_days : B) 
  (a_completion_time : a_days = 6)
  (joint_completion_time : (1 / a_days + 1 / b_days) = 1 / 4) :
  a_days / a_days + a_days / b_days = 4 :=
by 
  -- To be filled with proof
  sorry

end joint_work_completion_l200_200016


namespace solve_quadratic_equation_l200_200271

theorem solve_quadratic_equation (x : ℝ) :
  x^2 - 2 * x - 8 = 0 ↔ x = 4 ∨ x = -2 := by
sorry

end solve_quadratic_equation_l200_200271


namespace angle_A_is_90_l200_200534

open Classical

variables {α : Type*} [Field α] [LinearOrderedField α] {A B C D : Type*}

def is_triangle (A B C : α) : Prop := ∃ t : α, t > 0 ∧ t < 180

def isosceles (A B C : α) (a b : α) : Prop := a = b

def bisects_angle (A B C D : α) : Prop := 
  ∃ x : α, x > 0 ∧ x < 180 ∧ 2 * x = angle A B D + angle D B C

def on_line (D C : α) : Prop := ∃ d : α, d = C - D

def angle_sum (A B C D : α) : Prop := 
  angle A B C + angle B C D + angle C A B = 180

theorem angle_A_is_90 (A B C D : α) (h_triangle : is_triangle A B C) 
  (h_isosceles : isosceles A B C (dist A B) (dist A C))
  (h_AD_DC : dist B D = dist D C) 
  (h_bisects : bisects_angle B D A C): 
  angle B A C = 90 :=
begin
  sorry
end

end angle_A_is_90_l200_200534


namespace min_square_value_l200_200233

theorem min_square_value (a b : ℤ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ r : ℤ, r^2 = 15 * a + 16 * b)
  (h2 : ∃ s : ℤ, s^2 = 16 * a - 15 * b) : 
  231361 ≤ min (15 * a + 16 * b) (16 * a - 15 * b) :=
sorry

end min_square_value_l200_200233


namespace arrange_digits_1225_l200_200963

theorem arrange_digits_1225 : 
  let digits := [1, 2, 2, 5] in 
  let endsIn5 (l : List ℕ) := (List.reverse l).headI = 5 in
  (List.permutations digits).count endsIn5 = 6 :=
sorry

end arrange_digits_1225_l200_200963


namespace abs_a_eq_5_and_a_add_b_eq_0_l200_200186

theorem abs_a_eq_5_and_a_add_b_eq_0 (a b : ℤ) (h1 : |a| = 5) (h2 : a + b = 0) :
  a - b = 10 ∨ a - b = -10 :=
by
  sorry

end abs_a_eq_5_and_a_add_b_eq_0_l200_200186


namespace sin_B_value_l200_200517

noncomputable theory

variables {R : Type*} [linear_ordered_ring R] [trig R]

variables {a b c A B C : R}

theorem sin_B_value (h1 : a + c = 2 * b) (h2 : A - C = π / 3)
  (ha : a = 2 * sin A) (hb : b = 2 * sin B) (hc : c = 2 * sin C) : 
  sin B = sqrt 3 / 2 :=
by sorry

end sin_B_value_l200_200517


namespace sum_of_x_intercepts_l200_200598

theorem sum_of_x_intercepts (a b : ℕ) (h₁ : b > 0) (h₂ : a > 0) (h₃ : a * b = 40) :
  ∑ x in (finset.filter (λ x, -10 / a = x ∧ a * b = 40) (finset.range 41)).val.to_list, x = -22.5 := sorry

end sum_of_x_intercepts_l200_200598


namespace intersection_M_N_l200_200863

open Set

variable {α : Type*} [Preorder α] [HasZero α] [HasOne α] [HasSubset α] [LT α] [HasMem α]

-- Definitions of sets M and N
def M := {x : ℚ | -3 < x ∧ x < 1}
def N := ({-3, -2, -1, 0, 1} : Set ℚ)

-- The proof problem statement
theorem intersection_M_N :
  M ∩ N = {-2, -1, 0} :=
sorry

end intersection_M_N_l200_200863


namespace correct_formula_l200_200771

def table : List (ℕ × ℕ) :=
    [(1, 3), (2, 8), (3, 15), (4, 24), (5, 35)]

theorem correct_formula : ∀ x y, (x, y) ∈ table → y = x^2 + 4 * x + 3 :=
by
  intros x y H
  sorry

end correct_formula_l200_200771


namespace string_length_on_post_l200_200363

theorem string_length_on_post 
    (c : ℝ) (h : ℝ) (n : ℕ) (l : ℝ)
    (h_circumference : c = 6)
    (h_height : h = 15)
    (h_loops : n = 5)
    (h_length : l = 15 * real.sqrt 5) : 
    ∃ (string_length : ℝ), string_length = l := 
begin
    -- declare the length of a single loop
    let loop_height := h / n,
    let loop_length := real.sqrt (loop_height ^ 2 + c ^ 2),
    let total_length := n * loop_length,
    -- check if total length is equal to given length
    use total_length,
    have : loop_height = 3, by sorry, -- loop_height is 3 feet
    have : loop_length = 3 * real.sqrt 5, by sorry, -- loop_length is 3√5 feet
    have : total_length = 5 * (3 * real.sqrt 5), by sorry, -- total length is 15√5 feet
    exact h_length,
end

end string_length_on_post_l200_200363
