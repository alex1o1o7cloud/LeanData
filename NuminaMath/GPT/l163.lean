import Mathlib
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Algebra.Real
import Mathlib.Algebra.Ring.Basic
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Int.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Combinatorics
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Angle
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Circles
import Mathlib.Geometry.Euclidean.Circumcircle
import Mathlib.Geometry.Euclidean.NinePointCircle
import Mathlib.Geometry.Line
import Mathlib.NumberTheory.Basic
import Mathlib.NumberTheory.ModularArithmetic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Algebra.Affine
import Mathlib.Topology.ContinuousFunction.Basic
import Mathlib.Topology.EuclideanTriangle
import Mathlib.Topology.Instances.Real
import Real.Basic

namespace find_possible_value_l163_163491

variable (a b : ℝ)
variable (h : (a * b) / (a^2 + b^2) = 1 / 4)

theorem find_possible_value : abs (a^2 - b^2) / (a^2 + b^2) = sqrt 3 / 2 := 
by sorry

end find_possible_value_l163_163491


namespace students_doing_at_least_one_hour_of_homework_l163_163979

theorem students_doing_at_least_one_hour_of_homework (total_angle : ℝ) (less_than_one_hour_angle : ℝ) 
  (h1 : total_angle = 360) (h2 : less_than_one_hour_angle = 90) :
  let less_than_one_hour_fraction := less_than_one_hour_angle / total_angle
  let less_than_one_hour_percentage := less_than_one_hour_fraction * 100
  let at_least_one_hour_percentage := 100 - less_than_one_hour_percentage
  at_least_one_hour_percentage = 75 :=
by
  let less_than_one_hour_fraction := less_than_one_hour_angle / total_angle
  let less_than_one_hour_percentage := less_than_one_hour_fraction * 100
  let at_least_one_hour_percentage := 100 - less_than_one_hour_percentage
  sorry

end students_doing_at_least_one_hour_of_homework_l163_163979


namespace winning_strategy_l163_163260

theorem winning_strategy (n : ℕ) (h : n > 2) :
  (∃ strategy : (Π turn, bool), true) ↔ n % 3 = 1 :=
by {
  sorry
}

end winning_strategy_l163_163260


namespace problem1_solution_problem2_solution_l163_163499

-- Given the function f(x) = |-2x + 4| - |x + 6|
def f (x : ℝ) : ℝ := |-(2 * x) + 4| - |x + 6|

-- First part: Proving the solution set of f(x) ≥ 0 is (-∞, -2/3] ∪ [10, +∞)
theorem problem1_solution (x : ℝ) : f(x) ≥ 0 ↔ x ∈ Iic (-2 / 3) ∨ x ∈ Ici 10 := sorry

-- Second part: Proving there exists a real number x such that f(x) > a + |x - 2| if and only if a < 8
theorem problem2_solution (a : ℝ) : (∃ x : ℝ, f(x) > a + |x - 2|) ↔ a < 8 := sorry

end problem1_solution_problem2_solution_l163_163499


namespace ratio_of_areas_equilateral_triangles_l163_163469

noncomputable def side_length (ABC : Triangle) : ℝ := sorry

noncomputable def extend_side (s : ℝ) : ℝ := 3 * s

theorem ratio_of_areas_equilateral_triangles (ABC A' B' C' : Triangle) (h₁ : equilateral ABC)
  (h₂ : equilateral A') (h₃ : equilateral B') (h₄ : equilateral C') (s : ℝ) :
  area A' / area ABC = 9 :=
sorry

end ratio_of_areas_equilateral_triangles_l163_163469


namespace ceilings_left_to_paint_l163_163511

theorem ceilings_left_to_paint
    (floors : ℕ)
    (rooms_per_floor : ℕ)
    (ceilings_painted_this_week : ℕ)
    (hallways_per_floor : ℕ)
    (hallway_ceilings_per_hallway : ℕ)
    (ceilings_painted_ratio : ℚ)
    : floors = 4
    → rooms_per_floor = 7
    → ceilings_painted_this_week = 12
    → hallways_per_floor = 1
    → hallway_ceilings_per_hallway = 1
    → ceilings_painted_ratio = 1 / 4
    → (floors * rooms_per_floor + floors * hallways_per_floor * hallway_ceilings_per_hallway 
        - ceilings_painted_this_week 
        - (ceilings_painted_ratio * ceilings_painted_this_week + floors * hallway_ceilings_per_hallway) = 13) :=
by
  intros
  sorry

end ceilings_left_to_paint_l163_163511


namespace option_d_is_correct_l163_163250

theorem option_d_is_correct (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b :=
by
  sorry

end option_d_is_correct_l163_163250


namespace part1_part2_l163_163474

namespace ArithmeticSeq

variable (n : ℕ)

def a (n : ℕ) := -2 * n + 15
def S (n : ℕ) := n * (a 1 + a n) / 2
def T (n : ℕ) := 
  if n ≤ 7 then 
    -n^2 + 14 * n
  else 
    n^2 - 14 * n + 98

theorem part1 (a2 : a 2 = 11) (S10 : S 10 = 40) :
  ∀ n, a n = -2 * n + 15 := 
sorry

theorem part2 (a2 : a 2 = 11) (S10 : S 10 = 40) :
  ∀ n, T n = 
    if n ≤ 7 then 
      -n^2 + 14 * n
    else 
      n^2 - 14 * n + 98 := 
sorry

end ArithmeticSeq

end part1_part2_l163_163474


namespace coeff_x3_in_expansion_l163_163870

theorem coeff_x3_in_expansion :
  let expr := (x + 2/x)^5 in 
  ∀ x : ℝ, -- assuming x is a real number for this context
  (coeff expr 3) = 10 :=
by 
sorry

end coeff_x3_in_expansion_l163_163870


namespace inequality_proof_l163_163131

variable (a b c : ℝ)

noncomputable def specific_condition (a b c : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ (1 / a + 1 / b + 1 / c = 1)

theorem inequality_proof (h : specific_condition a b c) :
  (a^a * b * c + b^b * c * a + c^c * a * b) ≥ 27 * (b * c + c * a + a * b) := 
by {
  sorry
}

end inequality_proof_l163_163131


namespace problem_a_problem_b_problem_c_l163_163285

-- Definition to check if a polynomial is a sum of squares
def is_sum_of_squares (f : Polynomial ℝ) : Prop :=
  ∃ p : list (Polynomial ℝ), f = p.sum (λ p, p ^ 2)

-- Define the specific polynomials for the problem
def polynomial_a (a : ℝ) : Polynomial ℝ := 
  Polynomial.C a + Polynomial.x * Polynomial.C 4 + Polynomial.x^2

def polynomial_b (a : ℝ) : Polynomial ℝ := 
  Polynomial.C a + Polynomial.x * Polynomial.C (4 - 2 * a) + Polynomial.x^2 * Polynomial.C (a - 7) + Polynomial.x^3 * Polynomial.C 2 + Polynomial.x^4

-- Problems to be proved
theorem problem_a (a : ℝ) : a ≥ 4 ↔ is_sum_of_squares (polynomial_a a) := sorry

theorem problem_b (a : ℝ) : a ≥ 4 ↔ is_sum_of_squares (polynomial_b a) := sorry

theorem problem_c (f : Polynomial ℝ) : is_sum_of_squares f → 
  ∃ u v : Polynomial ℝ, f = u ^ 2 + v ^ 2 := sorry

end problem_a_problem_b_problem_c_l163_163285


namespace solve_inequality_l163_163543

theorem solve_inequality (x : ℝ) (h : x < 4) : (x - 2) / (x - 4) ≥ 3 := sorry

end solve_inequality_l163_163543


namespace remainder_of_power_l163_163242

theorem remainder_of_power :
  (4^215) % 9 = 7 := by
sorry

end remainder_of_power_l163_163242


namespace anne_carries_total_weight_l163_163300

-- Definitions for the conditions
def weight_female_cat : ℕ := 2
def weight_male_cat : ℕ := 2 * weight_female_cat

-- Problem statement
theorem anne_carries_total_weight : weight_female_cat + weight_male_cat = 6 :=
by
  sorry

end anne_carries_total_weight_l163_163300


namespace product_of_powers_l163_163577

theorem product_of_powers (n : ℕ) : 
  (∏ k in (Finset.range (n + 1)), (2^(2^k) + 1)) = 4^(2^n) - 1 :=
sorry

end product_of_powers_l163_163577


namespace count_valid_4_digit_numbers_l163_163661

-- Definitions based on conditions from part (a)
def valid_first_two_digits (d1 d2 : ℕ) : Prop :=
  d1 = 1 ∨ d1 = 4 ∨ d1 = 5 ∧ d2 = 1 ∨ d2 = 4 ∨ d2 = 5

def valid_third_digit (d3 : ℕ) : Prop :=
  d3 = 5 ∨ d3 = 7 ∨ d3 = 8 ∨ d3 = 9

def valid_fourth_digit (d3 d4 : ℕ) : Prop :=
  (d4 = 5 ∨ d4 = 7 ∨ d4 = 8 ∨ d4 = 9) ∧ d4 ≠ d3

-- Proposition based on the question and correct answer from part (b)
theorem count_valid_4_digit_numbers : 
  (∑ d1 in {1, 4, 5}.to_finset, ∑ d2 in {1, 4, 5}.to_finset, ∑ d3 in {5, 7, 8, 9}.to_finset, ∑ d4 in {5, 7, 8, 9}.to_finset.filter (λ d4, d4 ≠ d3), 1) = 108 := sorry

end count_valid_4_digit_numbers_l163_163661


namespace P_at_6_l163_163145

noncomputable def P (x : ℕ) : ℚ := (720 * x) / (x^2 - 1)

theorem P_at_6 : P 6 = 48 :=
by
  -- Definitions and conditions derived from the problem.
  -- Establishing given condition and deriving P(6) value.
  sorry

end P_at_6_l163_163145


namespace points_on_quadratic_function_l163_163769

theorem points_on_quadratic_function (m y1 y2 y3 : ℝ) :
  y1 = 4 - 8 - m →
  y2 = 9 - 12 - m →
  y3 = 1 + 4 - m →
  y3 > y2 ∧ y2 > y1 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  linarith

end points_on_quadratic_function_l163_163769


namespace constant_term_expansion_eq_61_l163_163729

theorem constant_term_expansion_eq_61 (x : ℝ) :
  let n := 6 in
  (∀ r : ℕ, r ≠ 3 → binomial n r * 2^r * x^r < binomial n 3 * 2^3 * x^3) →
  constant_term ((1 + 1 / (x^2)) * (1 + 2 * x)^n) = 61 :=
by
  simp [constant_term]
  sorry

end constant_term_expansion_eq_61_l163_163729


namespace tom_subtracts_value_l163_163905

theorem tom_subtracts_value :
  let a := 50
  let b := 1
  let square_50 := a^2
  let square_49 := (a - b)^2
  ∃ k, square_50 - k = square_49 ∧ k = 99 :=
by
  let a := 50
  let b := 1
  let square_50 := a^2
  let square_49 := (a - b)^2
  use square_50 - square_49
  split
  { exact rfl }
  {
    have h1 : square_49 = 49^2 := by rfl
    have h2 : square_50 = 2500 := by rfl
    have h3 : 49^2 = 2401 := by rfl
    rw [h2, h3]
    exact rfl
  }

end tom_subtracts_value_l163_163905


namespace angle_A_eq_pi_over_3_area_of_triangle_ABC_l163_163135

variables (A B C a b c d : ℝ)

-- Given internal angles A, B, C of triangle ABC, with opposite side lengths a, b, c respectively
-- Vectors m and n are defined as follows
def m := (real.cos A, real.cos C)
def n := (c - 2 * b, a)

-- m is perpendicular to n
axiom perp_m_n : (c - 2 * b) * real.cos A + a * real.cos C = 0

-- Given B = π / 3
axiom B_is_pi_over_3 : B = real.pi / 3

-- Given length of the median AM is d
axiom AM_length : (1 / 2) * a * a + (1 / 2) * b * b + (1 / 2) * c * c - (1 / 4) * a * a = d * d

-- Proving the magnitude of angle A is π / 3
theorem angle_A_eq_pi_over_3 : A = real.pi / 3 :=
by
  sorry

-- Proving the area of triangle ABC is (3√3 / 4)d^2
theorem area_of_triangle_ABC : (1 / 2) * b * c * real.sin A = (3 * real.sqrt 3 / 4) * d * d :=
by
  sorry

end angle_A_eq_pi_over_3_area_of_triangle_ABC_l163_163135


namespace line_perpendicular_to_two_planes_parallel_l163_163360

-- Declare lines and planes
variables {Line Plane : Type}

-- Define the perpendicular and parallel relationships
variables (perpendicular : Line → Plane → Prop)
variables (parallel : Plane → Plane → Prop)

-- Given conditions
variables (m n : Line) (α β : Plane)
-- The known conditions are:
-- 1. m is perpendicular to α
-- 2. m is perpendicular to β
-- We want to prove:
-- 3. α is parallel to β

theorem line_perpendicular_to_two_planes_parallel (h1 : perpendicular m α) (h2 : perpendicular m β) : parallel α β :=
sorry

end line_perpendicular_to_two_planes_parallel_l163_163360


namespace total_spent_is_140_l163_163327

-- Define the original prices and discounts
def original_price_shoes : ℕ := 50
def original_price_dress : ℕ := 100
def discount_shoes : ℕ := 40
def discount_dress : ℕ := 20

-- Define the number of items purchased
def number_of_shoes : ℕ := 2
def number_of_dresses : ℕ := 1

-- Define the calculation of discounted prices
def discounted_price_shoes (original_price : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  (original_price * quantity * (100 - discount)) / 100

def discounted_price_dress (original_price : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  (original_price * quantity * (100 - discount)) / 100

-- Define the total cost calculation
def total_cost : ℕ :=
  discounted_price_shoes original_price_shoes discount_shoes number_of_shoes +
  discounted_price_dress original_price_dress discount_dress number_of_dresses

-- The theorem to prove
theorem total_spent_is_140 : total_cost = 140 := by
  sorry

end total_spent_is_140_l163_163327


namespace minimum_value_of_function_l163_163212

-- Define the function y = 2x + 1/(x - 1) with the constraint x > 1
noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / (x - 1)

-- Prove that the minimum value of the function for x > 1 is 2√2 + 2
theorem minimum_value_of_function : 
  ∃ x : ℝ, x > 1 ∧ ∀ y : ℝ, (y = f x) → y ≥ 2 * Real.sqrt 2 + 2 := 
  sorry

end minimum_value_of_function_l163_163212


namespace find_OH_squared_l163_163133

noncomputable def OH_squared (a b c R : ℝ) : ℝ :=
  9 * R^2 - (2 * a^2 + b^2 + c^2)

theorem find_OH_squared 
  (a b c R : ℝ) 
  (h1 : R = 8) 
  (h2 : 2 * a^2 + b^2 + c^2 = 50) 
  : OH_squared a b c R = 526 :=
by 
  subst h1 
  rw [OH_squared, h2]
  norm_num
  sorry

end find_OH_squared_l163_163133


namespace triangle_similarity_MNC_ABC_l163_163014

-- Define the points and assumptions.
variables {A B C H M N : Type}
variables [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint H] [IsPoint M] [IsPoint N]

-- Define properties for the points in the acute triangle.
axiom acute_triangle_ABC : IsAcuteTriangle A B C
axiom height_CH : IsHeight C H A B
axiom perp_HM : IsPerpendicular H M B C
axiom perp_HN : IsPerpendicular H N A C

-- Define the similarity relation
theorem triangle_similarity_MNC_ABC :
  SimilarTriangles (Triangle.mk M N C) (Triangle.mk A B C) :=
sorry

end triangle_similarity_MNC_ABC_l163_163014


namespace algebra_inequality_l163_163365

theorem algebra_inequality
  (x y z : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z)
  (h_cond : x * y + y * z + z * x ≤ 1) :
  (x + 1 / x) * (y + 1 / y) * (z + 1 / z) ≥ 8 * (x + y) * (y + z) * (z + x) :=
by
  sorry

end algebra_inequality_l163_163365


namespace sum_of_solutions_l163_163816

theorem sum_of_solutions
  (T : ℝ)
  (cond : ∀ x : ℝ, x > 0 → x ^ 3 ^ sqrt 3 = (sqrt 3) ^ 3 ^ x) :
  2 ≤ T ∧ T < 4 :=
sorry

end sum_of_solutions_l163_163816


namespace math_problem_l163_163735

noncomputable def f (ω x : ℝ) : ℝ :=
  6 * (Real.cos (ω * x / 2))^2 + (Real.sqrt 3) * Real.sin (ω * x) - 3

theorem math_problem (ω : ℝ) (x₀ : ℝ) 
  (h₀ : ω > 0)
  (h₁ : 6 * (Real.cos (ω * x₀ / 2))^2 + (Real.sqrt 3) * Real.sin (ω * x₀) - 3 = (6 : ℝ) * Real.sqrt 3 / 5)
  (h₂ : x₀ > 2/3 ∧ x₀ < 14/3)
  (h₃ : (8 : ℝ) = 4 * 2) :
  (ω = Real.pi / 4) ∧
  (∀ k : ℤ, ∀ x : ℝ, x ∈ [-10/3 + 8 * k, 2/3 + 8 * k] → 
    6 * (Real.cos (ω * x / 2))^2 + (Real.sqrt 3) * Real.sin (ω * x) - 3 = 2 * Real.sqrt 3 * Real.sin (π * x / 4 + π / 3)) ∧
  (6 * (Real.cos (ω * (x₀ + 1) / 2))^2 + (Real.sqrt 3) * Real.sin (ω * (x₀ + 1)) - 3 = -Real.sqrt 6 / 5) :=
sorry

end math_problem_l163_163735


namespace problem_1_problem_2_iff_problem_3_l163_163927

-- Problem I
theorem problem_1 (x : ℝ) (hx : x > 1) : 2 * Real.log x < x - 1 / x := by
  sorry

-- Problem II
def inequality_holds_for_all_t (a : ℝ) : Prop :=
  ∀ t : ℝ, t > 0 → (1 + a / t) * Real.log (1 + t) > a

theorem problem_2_iff (a : ℝ) : inequality_holds_for_all_t a ↔ a ∈ Ioo 0 2 := by
  sorry

-- Problem III
theorem problem_3 : (9 / 10 : ℝ) ^ 19 < 1 / Real.exp 2 := by
  sorry

end problem_1_problem_2_iff_problem_3_l163_163927


namespace volume_of_tetrahedron_ABCD_l163_163523

theorem volume_of_tetrahedron_ABCD
  (PQ RS : ℝ)
  (PR QS : ℝ)
  (PS QR : ℝ)
  (A B C D P Q R S : ℝ)
  (PQRΔ PQSΔ PRSΔ QRSΔ : Type) :
  PQ = 7 → RS = 7 →
  PR = 8 → QS = 8 →
  PS = 9 → QR = 9 →
  inscribed_circle_center PQRΔ P SQ A →
  inscribed_circle_center PQSΔ P S B →
  inscribed_circle_center PRSΔ P Q C →
  inscribed_circle_center QRSΔ Q R D →
  abs (volume_of_tetrahedron A B C D - 1.84) < 0.01 :=
by
  sorry

end volume_of_tetrahedron_ABCD_l163_163523


namespace negation_of_p_l163_163496

theorem negation_of_p :
  ¬ (∃ (x : ℝ), x > 0 ∧ sin x > 2^x - 1) ↔ ∀ (x : ℝ), x > 0 → sin x ≤ 2^x - 1 :=
sorry

end negation_of_p_l163_163496


namespace sum_of_marked_angles_of_star_l163_163168

theorem sum_of_marked_angles_of_star 
  (α β γ δ ε : ℝ)
  (h : α + β + γ + δ + ε = 180) :
  ∑ angles = 180 := 
sorry

end sum_of_marked_angles_of_star_l163_163168


namespace sum_of_digits_l163_163515

theorem sum_of_digits (d : ℕ) (exchange_rate : ℕ → ℕ) (spent_pesos remaining_pesos : ℕ)
  (h1 : exchange_rate d = 1.5 * d)
  (h2 : spent_pesos = 72)
  (h3 : remaining_pesos = d)
  (h4 : 1.5 * d - spent_pesos = remaining_pesos) :
  d.digits.sum = 9 :=
sorry

end sum_of_digits_l163_163515


namespace irrational_alpha_condition_l163_163702

noncomputable def is_floor (r : ℝ) (n : ℕ) : Prop := n = Int.floor r

def A (x : ℝ) : Set ℕ := {n | ∃ (k : ℕ), is_floor (k * x) n}

theorem irrational_alpha_condition (α : ℝ) (h1 : α > 2) (h2 : α ∉ ℚ) :
  ∀ β : ℝ, β > 0 ∧ (A(α).subset A(β)) → (β / α).denom = 1 := by
  sorry

end irrational_alpha_condition_l163_163702


namespace cistern_water_height_l163_163936

theorem cistern_water_height (h : ℝ) (L : ℝ) (W : ℝ) (A_wet : ℝ)
  (hC : L = 6) (wC : W = 4) (A_wet_C : A_wet = 49) :
  20 * h = 25 :=
by
  have : 24 + 12 * h + 8 * h = A_wet,
    sorry

end cistern_water_height_l163_163936


namespace cut_out_seconds_l163_163235

def original_length_minutes : ℝ := 0.5
def original_length_seconds : ℝ := 60 * original_length_minutes
def reduction_percentage : ℝ := 30 / 100
def length_to_cut_seconds : ℝ := reduction_percentage * original_length_seconds

theorem cut_out_seconds : length_to_cut_seconds = 9 := by
  sorry

end cut_out_seconds_l163_163235


namespace integral_sin6_cos2_correct_l163_163668

noncomputable def integral_sin6_cos2 : ℝ :=
∫ (x : ℝ) in 0..(2 * Real.pi), (Real.sin (x / 4))^6 * (Real.cos (x / 4))^2

theorem integral_sin6_cos2_correct : integral_sin6_cos2 = (5 * Real.pi) / 64 :=
by
  sorry

end integral_sin6_cos2_correct_l163_163668


namespace count_positive_integers_in_square_range_l163_163005

theorem count_positive_integers_in_square_range :
  {x : ℕ // 121 ≤ x^2 ∧ x^2 ≤ 289}.card = 7 :=
sorry

end count_positive_integers_in_square_range_l163_163005


namespace Lyle_can_buy_for_his_friends_l163_163508

theorem Lyle_can_buy_for_his_friends
  (cost_sandwich : ℝ) (cost_juice : ℝ) (total_money : ℝ)
  (h1 : cost_sandwich = 0.30)
  (h2 : cost_juice = 0.20)
  (h3 : total_money = 2.50) :
  (total_money / (cost_sandwich + cost_juice)).toNat - 1 = 4 :=
by
  sorry

end Lyle_can_buy_for_his_friends_l163_163508


namespace sum_of_x_values_l163_163490

-- Define the quadratic function g
def g (x : ℝ) : ℝ := x^2 - 6 * x + 8

-- The theorem to prove that the sum of all distinct x values satisfying g(g(g(x))) = 2 is 12
theorem sum_of_x_values (x : ℝ) (hx : g (g (g x)) = 2) : 
  ∑ x in {a | g (g (g a)) = 2}.to_finset, x = 12 := by
  sorry

end sum_of_x_values_l163_163490


namespace holidays_taken_per_year_l163_163969

theorem holidays_taken_per_year (holidays_per_month : ℕ) (months_per_year : ℕ) (h1 : holidays_per_month = 4) (h2 : months_per_year = 12) : holidays_per_month * months_per_year = 48 :=
by
  rw [h1, h2]
  exact mul_comm 4 12
  exact Nat.mul_comm 4 12
  sorry  -- Skip the complete proof for now

-- You can also use a noncomputable def if needed

end holidays_taken_per_year_l163_163969


namespace cauchy_schwarz_am_gm_inequality_l163_163264

-- Part 1
theorem cauchy_schwarz 
  (a b c x y z : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) : 
  (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) ≥ (ax + by + cz)^2 := 
  sorry

-- Part 2
theorem am_gm_inequality 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h_sum : a + b + c = 1) : 
  sqrt a + sqrt (2 * b) + sqrt (3 * c) ≤ sqrt 6 := 
  sorry

end cauchy_schwarz_am_gm_inequality_l163_163264


namespace cross_product_and_perpendicular_l163_163687

noncomputable def v1 : ℝ × ℝ × ℝ := (4, 3, -5)
noncomputable def v2 : ℝ × ℝ × ℝ := (2, -1, 4)
noncomputable def cross_prod : ℝ × ℝ × ℝ := (7, -26, -10)

theorem cross_product_and_perpendicular :
  let cross_v := (v1.2.1 * v2.2.2 - v1.2.2 * v2.2.1, v1.2.2 * v2.1 - v1.1 * v2.2.2, v1.1 * v2.2.1 - v1.2.1 * v2.1) in
  cross_v = cross_prod ∧ 
  (v1.1 * cross_v.1 + v1.2.1 * cross_v.2 + v1.2.2 * cross_v.2.1 = 0) ∧
  (v2.1 * cross_v.1 + v2.2.1 * cross_v.2 + v2.2.2 * cross_v.2.1 = 0) :=
by
  sorry

end cross_product_and_perpendicular_l163_163687


namespace part_I_part_II_l163_163025

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {T : ℕ → ℝ}

-- Definitions from conditions
axiom a1 : a 1 = 1
axiom seq_cond : ∀ n, n ≥ 2 → S n * S n = a n * (S n - 1 / 2)
noncomputable def b (n : ℕ) : ℝ := (S n) / (2 * n + 1)

-- Proof statements
theorem part_I (n : ℕ) (h : n ≥ 2) : S n = 1 / (2 * n - 1) := sorry

theorem part_II (n : ℕ) (h : n ≥ 1) : (Finset.range n).sum (λ k, b (k + 1)) = n / (2 * n + 1) := sorry

end part_I_part_II_l163_163025


namespace decrease_iff_inequality_l163_163470

variable {S : Type} [Fintype S]
variable (f : Set S → ℝ)

def monotone_decreasing (f : Set S → ℝ) : Prop :=
  ∀ X Y : Set S, X ⊆ Y → f X ≥ f Y

theorem decrease_iff_inequality (h_monotone : monotone_decreasing f) :
  (∀ X Y, f (X ∪ Y) + f (X ∩ Y) ≤ f X + f Y) ↔
  (∀ a ∈ S, ∀ X ⊆ S \ {a}, f (X ∪ {a}) - f X ≤ f ((X ∪ {a}) ∪ X) - f X) := by
sorry

end decrease_iff_inequality_l163_163470


namespace problem_solution_set_l163_163412

variable {a b c : ℝ}

theorem problem_solution_set (h_condition : ∀ x, 1 ≤ x → x ≤ 2 → a * x^2 - b * x + c ≥ 0) : 
  { x : ℝ | c * x^2 + b * x + a ≤ 0 } = { x : ℝ | x ≤ -1 } ∪ { x | -1/2 ≤ x } :=
by 
  sorry

end problem_solution_set_l163_163412


namespace triangle_tangency_perimeter_l163_163240

def triangle_perimeter (a b c : ℝ) (s : ℝ) (t : ℝ) (u : ℝ) : ℝ :=
  s + t + u

theorem triangle_tangency_perimeter (a b c : ℝ) (D E F : ℝ) (s : ℝ) (t : ℝ) (u : ℝ)
  (h1 : a = 5) (h2 : b = 7) (h3 : c = 8) 
  (h4 : s + t + u = 3) : triangle_perimeter a b c s t u = 3 :=
by
  sorry

end triangle_tangency_perimeter_l163_163240


namespace trailing_zeroes_500_fact_l163_163678

-- Define a function to count multiples of a given number in a range
def countMultiples (n m : Nat) : Nat :=
  m / n

-- Define a function to count trailing zeroes in the factorial
def trailingZeroesFactorial (n : Nat) : Nat :=
  countMultiples 5 n + countMultiples (5^2) n + countMultiples (5^3) n + countMultiples (5^4) n

theorem trailing_zeroes_500_fact : trailingZeroesFactorial 500 = 124 :=
by
  sorry

end trailing_zeroes_500_fact_l163_163678


namespace parabola_solution_l163_163109

noncomputable def parabola_question (p : ℝ) (hp : p > 0) : ℝ :=
  if h : (∃ F : ℝ × ℝ, ∃ M : ℝ × ℝ, y^2 = 2 * p * x ∧ 
  (let r := 3 in 
    (|OF| = p / 2) ∧ 
    (⅟2)|M - F| * |F| ∧ 
    (3 = p / 2 + p / 4))) 
  then 4 
  else 0

theorem parabola_solution (p : ℝ) (hp : p > 0) (h_circle_area : 9π) :
  parabola_question p hp = 4 := 
sorry

end parabola_solution_l163_163109


namespace johns_grandpa_money_l163_163464

theorem johns_grandpa_money :
  ∃ G : ℝ, (G + 3 * G = 120) ∧ (G = 30) := 
by
  sorry

end johns_grandpa_money_l163_163464


namespace a_minus_b_7_l163_163199

theorem a_minus_b_7 (a b : ℤ) : (2 * y + a) * (y + b) = 2 * y^2 - 5 * y - 12 → a - b = 7 :=
by
  sorry

end a_minus_b_7_l163_163199


namespace smaller_root_of_quadratic_eq_l163_163860

theorem smaller_root_of_quadratic_eq : 
  ∃ x : ℝ, (x - 2/3) * (x - 5/6) + (x - 2/3)^2 - 1 = 0 ∧ 
            ∀ y : ℝ, (y - 2/3) * (y - 5/6) + (y - 2/3)^2 - 1 = 0 → x ≤ y → x = -1/12 :=
begin
  sorry
end

end smaller_root_of_quadratic_eq_l163_163860


namespace probability_at_least_twice_average_profit_twice_probability_select_once_visited_twice_l163_163934

-- Assumptions and givens
def total_members : ℕ := 100
def members_at_least_twice : ℕ := 40
def cost_per_service : ℝ := 150
def charge_ratios : list ℝ := [1, 0.95, 0.90, 0.85, 0.80]

-- The total possible ways to select 2 customers from 6
def total_possible_ways : ℕ := (6.choose 2)

-- The number of favorable ways to select exactly 1 visitor from 4 twice visitors and 2 thrice visitors
def favorable_ways : ℕ := 4 * 2

-- Helper function to calculate the charge for a visit
def charge_per_visit (n : ℕ) : ℝ :=
  200 * (nth charge_ratios (n-1)).get_or_else 0

-- Step d): Lean 4 statements
theorem probability_at_least_twice : (40:ℝ) / 100 = 0.4 := by
  sorry

theorem average_profit_twice : (charge_per_visit 1 - cost_per_service + charge_per_visit 2 - cost_per_service) / 2 = 45 := by
  sorry

theorem probability_select_once_visited_twice :
  (favorable_ways:ℝ) / total_possible_ways = (8:ℝ) / 15 := by
  sorry

end probability_at_least_twice_average_profit_twice_probability_select_once_visited_twice_l163_163934


namespace range_of_p_l163_163409

variable (p : ℝ)

def A : set ℝ := { x | -2 ≤ x ∧ x ≤ 5 }
def B (p : ℝ) : set ℝ := { x | p + 1 ≤ x ∧ x ≤ 2 * p - 1 }

theorem range_of_p (h : A ∩ B p = B p) : p ≤ 3 :=
sorry

end range_of_p_l163_163409


namespace sufficient_condition_perpendicular_planes_l163_163029

variables {Plane Line : Type}
variables (m : Line) (α β : Plane)

-- Define parallel and perpendicular relationships
def parallel (x y : Plane) : Prop := ∀ (l : Line), l ⊂ x → (∃ k : Line, k ⊂ y ∧ l ∥ k)
def perpendicular (x y : Plane) : Prop := ∃ l (k : Line), l ⊂ x ∧ k ⊂ y ∧ l ⟂ k

-- Given conditions
variables (h1 : ∀ l : Line, l ⊂ m → parallel l α)
variables (h2 : parallel m α)
variables (h3 : perpendicular m β)

-- Proof statement
theorem sufficient_condition_perpendicular_planes : perpendicular α β :=
sorry

end sufficient_condition_perpendicular_planes_l163_163029


namespace min_lines_3x3_min_lines_4x4_l163_163610

-- Problem (a): Minimum number of lines for a 3x3 board
theorem min_lines_3x3 : ∃ (n : ℕ), min_lines 3 3 n ∧ n = 2 :=
sorry

-- Problem (b): Minimum number of lines for a 4x4 board
theorem min_lines_4x4 : ∃ (n : ℕ), min_lines 4 4 n ∧ n = 3 :=
sorry

end min_lines_3x3_min_lines_4x4_l163_163610


namespace pyramid_value_l163_163637

theorem pyramid_value (a b c d e f : ℕ) (h_b : b = 6) (h_d : d = 20) (h_prod1 : d = b * (20 / b)) (h_prod2 : e = (20 / b) * c) (h_prod3 : f = c * (72 / c)) : a = b * c → a = 54 :=
by 
  -- Assuming the proof would assert the calculations done in the solution.
  sorry

end pyramid_value_l163_163637


namespace berries_ratio_l163_163186

theorem berries_ratio (total_berries : ℕ) (stacy_berries : ℕ) (ratio_stacy_steve : ℕ)
  (h_total : total_berries = 1100) (h_stacy : stacy_berries = 800)
  (h_ratio : stacy_berries = 4 * ratio_stacy_steve) :
  ratio_stacy_steve / (total_berries - stacy_berries - ratio_stacy_steve) = 2 :=
by {
  sorry
}

end berries_ratio_l163_163186


namespace cost_of_new_shoes_l163_163935

theorem cost_of_new_shoes :
  ∃ P : ℝ, P = 32 ∧ (P / 2 = 14.50 + 0.10344827586206897 * 14.50) :=
sorry

end cost_of_new_shoes_l163_163935


namespace sum_zeros_g_interval_l163_163820

theorem sum_zeros_g_interval :
  ∃ (s : ℝ), s = -5 ∧ ∀ f : ℝ → ℝ,
  (∀ x, f (-x) = - f x) ∧
  (∀ x, f (x + 2) = - f x) ∧
  (∀ x, 0 ≤ x → x ≤ 1 → f x = x / 2) →
  let g := λ x, f x + 1 / 2 in
  let zeros := {x | -10 ≤ x ∧ x ≤ 10 ∧ g x = 0} in
  s = (∑ z in zeros, z) := sorry

end sum_zeros_g_interval_l163_163820


namespace count_pos_integers_three_digits_l163_163072

/-- The number of positive integers less than 50,000 having at most three distinct digits equals 7862. -/
theorem count_pos_integers_three_digits : 
  ∃ n : ℕ, n < 50000 ∧ (∀ d1 d2 d3 d4 d5 : ℕ, d1 ≠ d2 ∨ d1 ≠ d3 ∨ d1 ≠ d4 ∨ d1 ≠ d5 ∨ d2 ≠ d3 ∨ d2 ≠ d4 ∨ d2 ≠ d5 ∨ d3 ≠ d4 ∨ d3 ≠ d5 ∨ d4 ≠ d5) ∧ n = 7862 :=
sorry

end count_pos_integers_three_digits_l163_163072


namespace find_angle_A_l163_163776

open Real

theorem find_angle_A (a b : ℝ) (B A : ℝ) 
  (ha : a = sqrt 2) 
  (hb : b = 2) 
  (hB : sin B + cos B = sqrt 2) :
  A = π / 6 := 
  sorry

end find_angle_A_l163_163776


namespace place_circle_in_rectangle_l163_163439

theorem place_circle_in_rectangle :
  ∃ (c : ℝ × ℝ), let rRect : set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 20 ∧ 0 ≤ p.2 ∧ p.2 ≤ 25},
                      unit_squares : set (set (ℝ × ℝ)) := {s | ∃ (x y : ℕ), 0 ≤ x ∧ x < 20 ∧ 0 ≤ y ∧ y < 25 ∧ 
                                                         s = {(px, py) | x ≤ px ∧ px < x + 1 ∧ y ≤ py ∧ py < y + 1}},
                      circle : set (ℝ × ℝ) := {p | (p.1 - c.1)^2 + (p.2 - c.2)^2 < 1/4} in
                      (∀ sq ∈ unit_squares, ¬(sq ∩ circle ≠ ∅)) ∧
                      ∃ x y, c = (x, y) ∧ circle ⊆ rRect :=
sorry

end place_circle_in_rectangle_l163_163439


namespace red_pairs_l163_163303

theorem red_pairs (total_students green_students red_students total_pairs green_pairs : ℕ) 
  (h1 : total_students = green_students + red_students)
  (h2 : green_students = 67)
  (h3 : red_students = 89)
  (h4 : total_pairs = 78)
  (h5 : green_pairs = 25)
  (h6 : 2 * green_pairs ≤ green_students ∧ 2 * green_pairs ≤ red_students ∧ 2 * green_pairs ≤ 2 * total_pairs) :
  ∃ red_pairs : ℕ, red_pairs = 36 := by
    sorry

end red_pairs_l163_163303


namespace Cody_book_series_total_count_l163_163667

theorem Cody_book_series_total_count :
  ∀ (weeks: ℕ) (books_first_week: ℕ) (books_second_week: ℕ) (books_per_week_after: ℕ),
    weeks = 7 ∧ books_first_week = 6 ∧ books_second_week = 3 ∧ books_per_week_after = 9 →
    (books_first_week + books_second_week + (weeks - 2) * books_per_week_after) = 54 :=
by
  sorry

end Cody_book_series_total_count_l163_163667


namespace ratio_of_segments_l163_163380

theorem ratio_of_segments (a b : ℕ) (ha : a = 200) (hb : b = 40) : a / b = 5 :=
by sorry

end ratio_of_segments_l163_163380


namespace ratio_of_areas_triangle_DEF_to_rectangle_ABCD_l163_163450

theorem ratio_of_areas_triangle_DEF_to_rectangle_ABCD (x : ℝ) :
  let AD := x,
      AB := 3 * x,
      area_ABCD := 3 * x^2,
      DE := x / Real.sqrt 2,
      EG := x / (2 * Real.sqrt 2),
      area_DEF := (1/2) * x * (x / (2 * Real.sqrt 2)) in
  (area_DEF / area_ABCD) = 1 / (12 * Real.sqrt 2) :=
by
sorry

end ratio_of_areas_triangle_DEF_to_rectangle_ABCD_l163_163450


namespace permutations_with_repetition_l163_163419

def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

theorem permutations_with_repetition (digits : multiset ℕ) (unique_digits : list ℕ) (counts : list ℕ) (h_counts : counts = [3, 1, 1]) :
  (factorial (multiset.card digits)) / (counts.product.map factorial).prod = 20 :=
by {
  have h1: factorial 5 = 120 := rfl, -- 5! = 120
  have h2: factorial 3 = 6 := rfl, -- 3! = 6
  have h3: factorial 1 = 1 := rfl, -- 1! = 1
  have h4: (counts.product.map factorial).prod = 6 :=
    by simp [counts, h_counts, h2, h3]; norm_num,
  simp [multiset.card, digits, h1, h4]; norm_num,
  sorry
}

end permutations_with_repetition_l163_163419


namespace necessary_but_not_sufficient_l163_163075

theorem necessary_but_not_sufficient (a : ℝ) : 
  (a > 2 → a > 5) ∧ (a > 5 → a > 2) :=
by
  split
  -- sufficient
  . intro ha
    sorry
  -- necessary
  . intro hb
    sorry

end necessary_but_not_sufficient_l163_163075


namespace janet_spends_more_on_piano_l163_163805

-- Condition definitions
def clarinet_hourly_rate : ℝ := 40
def clarinet_hours_per_week : ℝ := 3
def piano_hourly_rate : ℝ := 28
def piano_hours_per_week : ℝ := 5
def weeks_per_year : ℝ := 52

-- Calculations based on conditions
def weekly_cost_clarinet : ℝ := clarinet_hourly_rate * clarinet_hours_per_week
def weekly_cost_piano : ℝ := piano_hourly_rate * piano_hours_per_week
def weekly_difference : ℝ := weekly_cost_piano - weekly_cost_clarinet
def yearly_difference : ℝ := weekly_difference * weeks_per_year

theorem janet_spends_more_on_piano : yearly_difference = 1040 := by
  sorry 

end janet_spends_more_on_piano_l163_163805


namespace count_divisible_by_7_l163_163973

def isDivisibleBy7 (n : ℕ) : Prop :=
  (2^n - n^2) % 7 = 0

def countValidNs (limit : ℕ) : ℕ :=
  (List.range limit).filter isDivisibleBy7 |>.length

theorem count_divisible_by_7 :
  countValidNs 10000 = 2857 := by
  sorry

end count_divisible_by_7_l163_163973


namespace degree_of_resulting_polynomial_l163_163309

-- Define the three polynomials
def p1 := (λ x : ℝ, x^4)
def p2 := (λ x : ℝ, x + 1/x)
def p3 := (λ x : ℝ, 2 + 3/x + 4/x^2)

-- Define the polynomial formed by multiplying p1, p2, p3
def resulting_polynomial (x : ℝ) := p1 x * p2 x * p3 x

-- Statement asserting the degree of the resulting polynomial
theorem degree_of_resulting_polynomial (x : ℝ) : 
  (∃ n : ℕ, n = 5 ∧ ∀ c : ℝ, resulting_polynomial c = x^5 + 3 * c^4 + 6 * c^3 + 3 * c^2 + 4 * c) :=
sorry

end degree_of_resulting_polynomial_l163_163309


namespace sum_of_integers_between_3_and_12_l163_163600

theorem sum_of_integers_between_3_and_12 : 
  (∑ i in (Finset.filter (λ x, 3 < x ∧ x < 12) (Finset.range 12)), i) = 60 := 
by
  sorry

end sum_of_integers_between_3_and_12_l163_163600


namespace gen_formula_arithmetic_seq_sum_of_abs_arithmetic_seq_l163_163479

-- Definition of an arithmetic sequence and its sum
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + d * (n - 1)

def arithmetic_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- The given conditions
variable (a₂_value : ℤ := 11)
variable (S₁₀_value : ℤ := 40)

-- The general formula for the sequence given the conditions
theorem gen_formula_arithmetic_seq :
  ∃ a₁ d, (a₁ + d = a₂_value) ∧
          (10 * a₁ + (10 * 9 / 2) * d = S₁₀_value) ∧
          (∀ n, arithmetic_seq a₁ d n = -2 * n + 15) :=
sorry

-- The sum of absolute values of the first n terms of the sequence
def abs_arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ :=
  abs (arithmetic_seq a₁ d n)

def abs_arithmetic_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, abs_arithmetic_seq a₁ d (i + 1)

theorem sum_of_abs_arithmetic_seq (n : ℕ) :
  let T_n : ℤ :=
    if n ≤ 7 then -n^2 + 14 * n
    else n^2 - 14 * n + 98 in
  ∃ a₁ d, (a₁ + d = a₂_value) ∧
           (10 * a₁ + (10 * 9 / 2) * d = S₁₀_value) ∧
           (∀ n, arithmetic_seq a₁ d n = -2 * n + 15) ∧
           (abs_arithmetic_sum a₁ d n = T_n) :=
sorry

end gen_formula_arithmetic_seq_sum_of_abs_arithmetic_seq_l163_163479


namespace symmetric_line_correct_l163_163881

-- The original line
def original_line (x y : ℝ) : Prop := 2 * x + 3 * y - 6 = 0

-- The symmetric point function
def symmetric_point (px py qx qy : ℝ) : (ℝ × ℝ) :=
  (2 * px - qx, 2 * py - qy)

-- The point of reflection
def reflection_point : ℝ × ℝ := (1, -1)

-- The line to test for the correct answer
def tested_line (x y : ℝ) : Prop := 2 * x + 3 * y + 8 = 0

theorem symmetric_line_correct :
  ∃ (line : ℝ → ℝ → Prop), 
    line = tested_line ∧ 
    (∀ (x y : ℝ), original_line x y → 
       let (sx, sy) := symmetric_point 1 (-1) x y in
       line sx sy) :=
sorry

end symmetric_line_correct_l163_163881


namespace height_of_prism_l163_163104

noncomputable def prism_height (a : ℝ) (α : ℝ) : ℝ :=
  a * (Real.sin (Real.pi / 3 - α / 2) * Real.sin (Real.pi / 3 + α / 2)).sqrt / Real.sin (α / 2)

theorem height_of_prism (a α : ℝ) :
  ∃ h, h = prism_height a α :=
by
  use prism_height a α
  -- sorry can be used as follows until we provide the actual proof details
  sorry

end height_of_prism_l163_163104


namespace b_investment_is_63000_l163_163257

open Real

variable a b c : ℝ
variable total_profit c_profit : ℝ
variable X : ℝ

-- Conditions
def a_investment := 45000
def c_investment := 72000
def total_profit_amount := 60000
def c_profit_amount := 24000

-- Auxiliary condition that expresses the ratio condition
def investment_ratio_condition := (c_investment / (a_investment + X + c_investment) = 2 / 5)

theorem b_investment_is_63000 :
  investment_ratio_condition → X = 63000 :=
by
  sorry

end b_investment_is_63000_l163_163257


namespace find_full_price_l163_163153

-- Defining the conditions
variables (P : ℝ) 
-- The condition that 20% of the laptop's total cost is $240.
def condition : Prop := 0.2 * P = 240

-- The proof goal is to show that the full price P is $1200 given the condition
theorem find_full_price (h : condition P) : P = 1200 :=
sorry

end find_full_price_l163_163153


namespace y_coordinate_equidistant_l163_163236

-- Define the points C and D
structure Point where
  x : ℤ
  y : ℤ

def C : Point := { x := -3, y := -1 }
def D : Point := { x := 4, y := 7 }

-- Define the distance function between two points on a plane
@[simp]
def distance (P1 P2 : Point) : ℝ :=
  real.sqrt ((P1.x - P2.x)^2 + (P1.y - P2.y)^2)

-- Define the point on the y-axis
def y_axis_point (y : ℝ) : Point := { x := 0, y := y }

-- The theorem stating the y-coordinate for the equidistant point
theorem y_coordinate_equidistant : ∃ y : ℝ, distance C (y_axis_point y) = distance D (y_axis_point y) ∧ y = 55 / 16 :=
by
  -- declaration part initiates the theorem
  sorry

end y_coordinate_equidistant_l163_163236


namespace graph_symmetric_about_7_l163_163878

variable (f : ℝ → ℝ)

theorem graph_symmetric_about_7 (h : ∀ x : ℝ, f(x + 5) = f(9 - x)) : ∀ x : ℝ, f(x) = f(14 - x) :=
by
  sorry

end graph_symmetric_about_7_l163_163878


namespace value_of_x_div_y_l163_163494

theorem value_of_x_div_y (x y : ℝ) (h1 : 3 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 6) (h3 : ∃ t : ℤ, x = t * y) : 
  ∃ t : ℤ, x = t * y ∧ t = -2 := 
sorry

end value_of_x_div_y_l163_163494


namespace triangle_area_given_conditions_l163_163123

theorem triangle_area_given_conditions
  (b : ℝ) (B C : ℝ)
  (hb : b = 2)
  (hB : B = π/6)
  (hC : C = π/3) :
  let A := π - B - C in
  let c := (b * real.sin C) / real.sin B in
  (1/2) * b * c * real.sin A = 2 * real.sqrt 3 :=
by
  sorry

end triangle_area_given_conditions_l163_163123


namespace simplify_expression_l163_163847

theorem simplify_expression : 
  ( (Real.cbrt 125 - Real.sqrt (25 / 2))^2 = (75 - 50 * Real.sqrt 2) / 2 ) :=
by
  sorry

end simplify_expression_l163_163847


namespace trapezoid_centroid_l163_163230

variables {A B C D E F M N K O : Point} -- Variables for points
variables (AB CD BC : Line) -- Variables for lines/trapezoid sides
variables (BE EF FC : Segment) -- Variables for segments
variables [IsTrapezoid AB CD BC] -- AB and CD are parallel, completing definition of trapezoid
variables [H1 : NonParallelPoint BE EF FC BC] -- BE, EF, FC divide BC into equal parts
variables [H2 : IntersectionPoint AE DF K] -- K is intersection of AE and DF
variables [H3 : ParallelLineThrough K AB CD O] -- line through K parallel to AB and CD intersects MN at O
variables [MidPoints MN AB CD] -- M, N are midpoints of AB and CD

theorem trapezoid_centroid :
  Centroid O ABCD := 
sorry -- Proof not required, so we insert sorry

end trapezoid_centroid_l163_163230


namespace y_value_of_first_two_vertices_l163_163440

-- Definition of the problem conditions
def rect_coords_area (y : ℝ) : Prop :=
  let length := 1 - (-7) in -- Horizontal side
  let area := 56 in
  area = length * (y + 6) -- Using the area formula of a rectangle

-- The proof problem statement
theorem y_value_of_first_two_vertices : ∃ (y : ℝ), rect_coords_area y ∧ y = 1 :=
by
  use 1 -- The expected y-value
  unfold rect_coords_area
  norm_num
  ring
  sorry -- Complete the steps

end y_value_of_first_two_vertices_l163_163440


namespace cody_tickets_l163_163658

theorem cody_tickets (initial_tickets spent_tickets won_tickets : ℕ) (h_initial : initial_tickets = 49) (h_spent : spent_tickets = 25) (h_won : won_tickets = 6) : initial_tickets - spent_tickets + won_tickets = 30 := 
by 
  sorry

end cody_tickets_l163_163658


namespace binom_100_100_eq_one_l163_163992

/-- Definition of binomial coefficient. -/
def binomial (n k : ℕ) : ℕ :=
  nat.choose n k

/-- A theorem using the property of binomial coefficients. -/
theorem binom_100_100_eq_one : binomial 100 100 = 1 :=
by
  -- The following proof is omitted.
  -- The main goal is to ensure the statement is correct and compilable.
  sorry

end binom_100_100_eq_one_l163_163992


namespace coefficient_of_term_containing_x_l163_163193

-- Define the binomial coefficient for convenience
def binom (n k : Nat) : Nat :=
  Nat.choose n k

-- Define the general term in the expansion of (sqrt(x) - 1/sqrt(x))^6
noncomputable def term (r : Nat) (x : ℝ) : ℝ :=
  (pow (-1 : ℝ) r) * (binom 6 r) * (pow x (3 - r))

-- Define the expression (1-x)(sqrt(x) - 1/sqrt(x))^6 and extract the relevant term's coefficient
noncomputable def coefficient (x : ℝ) : ℝ :=
  let term1 := term 2 x
  let term2 := term 3 x
  term1 + term2

theorem coefficient_of_term_containing_x (x : ℝ) : coefficient x = 35 := 
by
  sorry

end coefficient_of_term_containing_x_l163_163193


namespace bob_smallest_number_l163_163293

theorem bob_smallest_number (b : ℕ) : 
  (∀ p : ℕ, prime p → p ∣ 72 → p ∣ b) → 
  (∀ c : ℕ, (∀ p : ℕ, prime p → p ∣ 72 → p ∣ c) → b ≤ c) → 
  b = 6 := 
by sorry

end bob_smallest_number_l163_163293


namespace lizard_eyes_fewer_than_spots_and_wrinkles_l163_163129

noncomputable def lizard_problem : Nat :=
  let eyes_jan := 3
  let wrinkles_jan := 3 * eyes_jan
  let spots_jan := 7 * (wrinkles_jan ^ 2)
  let eyes_cousin := 3
  let wrinkles_cousin := 2 * eyes_cousin
  let spots_cousin := 5 * (wrinkles_cousin ^ 2)
  let total_eyes := eyes_jan + eyes_cousin
  let total_wrinkles := wrinkles_jan + wrinkles_cousin
  let total_spots := spots_jan + spots_cousin
  (total_spots + total_wrinkles) - total_eyes

theorem lizard_eyes_fewer_than_spots_and_wrinkles :
  lizard_problem = 756 :=
by
  sorry

end lizard_eyes_fewer_than_spots_and_wrinkles_l163_163129


namespace perimeter_of_square_l163_163645

theorem perimeter_of_square (a : ℤ) (h : a * a = 36) : 4 * a = 24 := 
by
  sorry

end perimeter_of_square_l163_163645


namespace concyclic_MNPQ_l163_163105

-- Define the conditions for the triangle and points
variables (A B C : Point)
variables (M N P Q : Point)

-- Assume ABC is an acute-angled triangle
axiom acute_triangle (ABC : Triangle) : acute ABC

-- Define the altitudes and intersection points with circles
axiom M_condition : M ∈ (Circle (diameter (AB)) ∩ AltitudeFrom C ABC)
axiom N_condition : N ∈ (Circle (diameter (AC)) ∩ AltitudeFrom C ABC)
axiom P_condition : P ∈ (Circle (diameter (AB)) ∩ AltitudeFrom B ABC)
axiom Q_condition : Q ∈ (Circle (diameter (AC)) ∩ AltitudeFrom B ABC)

-- State the problem as a Lean theorem
theorem concyclic_MNPQ (ABC : Triangle) (M N P Q : Point) : 
  acute_triangle ABC →
  M_condition ABC M →
  N_condition ABC N →
  P_condition ABC P →
  Q_condition ABC Q →
  concyclic {M, N, P, Q} :=
sorry

end concyclic_MNPQ_l163_163105


namespace hypotenuse_length_l163_163639

theorem hypotenuse_length
  (a b c : ℝ)
  (h1 : a + b + c = 40)
  (h2 : (1 / 2) * a * b = 24)
  (h3 : a^2 + b^2 = c^2) :
  c = 18.8 :=
by sorry

end hypotenuse_length_l163_163639


namespace find_a_and_monotonicity_find_t_l163_163041

def f (x : ℝ) (a : ℝ) : ℝ := 2^x - a * 2^(-x)

theorem find_a_and_monotonicity :
  (∀ x : ℝ, f(-x) a = -f(x) a) → (a = 1 ∧ (∀ x y : ℝ, x < y → f(x) 1 < f(y) 1)) :=
sorry

theorem find_t (h : ∀ x : ℝ, f(x - t) 1 + f(x^2 - t^2) 1 ≥ 0) : t = -1 / 2 :=
sorry

end find_a_and_monotonicity_find_t_l163_163041


namespace sum_of_solutions_of_quadratic_l163_163353

theorem sum_of_solutions_of_quadratic :
  ∀ a b c x₁ x₂ : ℝ, a ≠ 0 →
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ (x = x₁ ∨ x = x₂)) →
  (∃ s : ℝ, s = x₁ + x₂ ∧ -b / a = s) :=
by
  sorry

end sum_of_solutions_of_quadratic_l163_163353


namespace range_of_a_inequality_l163_163400

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := real.log x + 1 / (x ^ 2) - a / x

theorem range_of_a (a : ℝ) (h : ∀ x > 0, f x a ≥ 0) : a ≤ 1 := sorry

theorem inequality (n : ℕ) : (∑ i in finset.range n, (i+1) / ((i+2)^2)) ≤ real.log (n + 1) := sorry

end range_of_a_inequality_l163_163400


namespace positive_real_x_l163_163681

theorem positive_real_x (A C : ℝ) (x : ℝ) (hA : 0 ≤ A) (hC : 0 ≤ C) (hx : 0 < x) :
  (sqrt (2 + A * C + 2 * C * x) + sqrt (A * C - 2 + 2 * A * x) = sqrt (2 * (A + C) * x + 2 * A * C)) ↔
  (x = 4) := sorry

end positive_real_x_l163_163681


namespace decimal_to_base_8_l163_163673

-- Define a function to convert a decimal number to its base 8 equivalent
def to_base_8 (n : ℕ) : ℕ := 
  let rec convert (n : ℕ) (acc : ℕ) (pow : ℕ) : ℕ :=
    if n = 0 then acc 
    else convert (n / 8) (acc + (n % 8) * pow) (pow * 10)
  in convert n 0 1

-- The theorem stating the conversion result
theorem decimal_to_base_8 : to_base_8 2357 = 4445 := by
  sorry

end decimal_to_base_8_l163_163673


namespace solve_for_x_l163_163541

theorem solve_for_x : ∀ x : ℝ, 3^(2 * x) = Real.sqrt 27 → x = 3 / 4 :=
by
  intro x h
  sorry

end solve_for_x_l163_163541


namespace travel_time_reduction_no_effect_l163_163579

-- Definitions and conditions
def cities : ℕ := 11
def capital : string := "Capital"
def time_to_capital : ℕ := 7
def time_cyclic_routes : ℕ := 3
def initial_transfer_time : ℕ := 2
def reduced_transfer_time : ℚ := 1.5

-- The maximal travel time between two cities A and B
def optimized_travel_time (x : ℚ) : ℚ :=
  if x = initial_transfer_time then 14 + x 
  else if x = reduced_transfer_time then 15.5 
  else 21

-- The proof problem in Lean 4 statement
theorem travel_time_reduction_no_effect :
  optimized_travel_time reduced_transfer_time = 15.5 :=
  sorry

end travel_time_reduction_no_effect_l163_163579


namespace count_squares_below_line_l163_163663

noncomputable def within_first_quadrant (x y : ℕ) : Prop :=
 x ≥ 0 ∧ y ≥ 0

noncomputable def below_line (x y : ℕ) : Prop :=
 (13 * x + 52 * y) < 676

theorem count_squares_below_line :
  (∑ x in finset.range 53, ∑ y in finset.range 14, if below_line x y then 1 else 0) = 306 :=
by
  sorry

end count_squares_below_line_l163_163663


namespace base10_to_base4_of_255_l163_163596

theorem base10_to_base4_of_255 :
  (255 : ℕ) = 3 * 4^3 + 3 * 4^2 + 3 * 4^1 + 3 * 4^0 :=
by
  sorry

end base10_to_base4_of_255_l163_163596


namespace Joe_total_time_correct_l163_163463

theorem Joe_total_time_correct :
  ∀ (distance : ℝ) (walk_rate : ℝ) (bike_rate : ℝ) (walk_time bike_time : ℝ),
    (walk_time = 9) →
    (bike_rate = 5 * walk_rate) →
    (walk_rate * walk_time = distance / 3) →
    (bike_rate * bike_time = 2 * distance / 3) →
    (walk_time + bike_time = 12.6) := 
by
  intros distance walk_rate bike_rate walk_time bike_time
  intro walk_time_cond
  intro bike_rate_cond
  intro walk_distance_cond
  intro bike_distance_cond
  sorry

end Joe_total_time_correct_l163_163463


namespace men_with_all_characteristics_at_least_15_percent_l163_163101

theorem men_with_all_characteristics_at_least_15_percent 
  (P_b P_h P_t P_w : ℝ)
  (hb : P_b = 70)
  (hh : P_h = 70)
  (ht : P_t = 85)
  (hw : P_w = 90) :
  ∃ P : ℝ, P = P_b + P_h + P_t + P_w - 100 * (4 - 1) ∧ P ≥ 15 :=
by {
  have h_sum : P_b + P_h + P_t + P_w = 70 + 70 + 85 + 90, by linarith [hb, hh, ht, hw],
  let P := P_b + P_h + P_t + P_w - 100 * (4 - 1),
  use P,
  split,
  { simp [P, hb, hh, ht, hw], linarith [h_sum] },
  { simp [P, hb, hh, ht, hw], linarith }
}

end men_with_all_characteristics_at_least_15_percent_l163_163101


namespace area_of_circle_outside_triangle_l163_163468

theorem area_of_circle_outside_triangle 
  (A B C X Y : Point)
  (O : Circle)
  (r : ℝ)
  (h_triangle : right_triangle A B C)
  (h_ABC : ∠BAC = 90)
  (h_tangent_AB : tangent_to_circle AB O)
  (h_tangent_AC : tangent_to_circle AC O)
  (h_XY_horizontal : horizontal_line X Y)
  (h_BC_hypotenuse : hypotenuse B C A)
  (h_AB : AB = 8)
  (h_AC : AC = 6) :
  area_circle_outside_triangle O A B C = 8 * π - 16 :=
sorry

end area_of_circle_outside_triangle_l163_163468


namespace line_triangle_intersection_l163_163418

theorem line_triangle_intersection (L : set ℝ²) (T : Finset ℝ²) (in_plane : L ⊆ plane_of T) :
  ∃ n, n ∈ {0, 1, 2, ∞} ∧ cardinality {p ∈ L | p ∈ boundary_of T} = n := by
  sorry

end line_triangle_intersection_l163_163418


namespace remainder_of_large_number_div_13_l163_163427

theorem remainder_of_large_number_div_13 :
  let a := 2614303940317
  in a % 13 = 4 :=
by
  sorry

end remainder_of_large_number_div_13_l163_163427


namespace range_of_f_l163_163886

noncomputable def f (x : ℝ) : ℝ := (1/3) ^ (x^2 - 2*x)

theorem range_of_f : Set.Ioo 0 3 ∪ {3} = { y | ∃ x, f x = y } :=
by
  sorry

end range_of_f_l163_163886


namespace volume_of_tetrahedron_ABCD_l163_163524

theorem volume_of_tetrahedron_ABCD
  (PQ RS : ℝ)
  (PR QS : ℝ)
  (PS QR : ℝ)
  (A B C D P Q R S : ℝ)
  (PQRΔ PQSΔ PRSΔ QRSΔ : Type) :
  PQ = 7 → RS = 7 →
  PR = 8 → QS = 8 →
  PS = 9 → QR = 9 →
  inscribed_circle_center PQRΔ P SQ A →
  inscribed_circle_center PQSΔ P S B →
  inscribed_circle_center PRSΔ P Q C →
  inscribed_circle_center QRSΔ Q R D →
  abs (volume_of_tetrahedron A B C D - 1.84) < 0.01 :=
by
  sorry

end volume_of_tetrahedron_ABCD_l163_163524


namespace smallest_positive_period_of_f_l163_163561

def f (x : ℝ) : ℝ := cos (2 * x) - 2 * sqrt 3 * sin x * cos x

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T' ≥ T) :=
sorry

end smallest_positive_period_of_f_l163_163561


namespace number_of_valid_integers_l163_163149

def greatest_prime_factor (n : ℕ) : ℕ := sorry -- Definition of greatest_prime_factor

theorem number_of_valid_integers : 
  ∀ (n : ℕ), n > 1 ∧ n ≤ 100 ∧ greatest_prime_factor(n) = nat.sqrt(n) ∧ greatest_prime_factor(2 * n + 1) = nat.sqrt(n) → false :=
by
  sorry

end number_of_valid_integers_l163_163149


namespace fertilizer_half_field_l163_163940

-- Definitions of the conditions
def field_area : ℝ := 7200
def total_fertilizer : ℝ := 1200
def half_field_area : ℤ := 1 / 2

-- Statement of the proof problem
theorem fertilizer_half_field : 
  let fertilizer_per_square_yard := total_fertilizer / field_area in
  let half_area := field_area * half_field_area in
  let fertilizer_half_field := fertilizer_per_square_yard * half_area in
  fertilizer_half_field = 600 :=
by
  sorry

end fertilizer_half_field_l163_163940


namespace people_sitting_arrangement_l163_163108

def num_ways_to_sit (total_chairs available_chairs num_people : ℕ) : ℕ :=
  if available_chairs >= num_people then 
    (Finset.range num_people).card.factorial
  else 
    0

theorem people_sitting_arrangement :
  num_ways_to_sit 7 5 5 = 120 :=
by
  unfold num_ways_to_sit
  simp
  exact finset.card_fin 5 !factorial
  sorry

end people_sitting_arrangement_l163_163108


namespace average_grams_per_cookie_is_approximately_l163_163991

-- Define the conditions
def num_cookies := 48
def chocolate_chips_grams := 108
def mms_grams := chocolate_chips_grams / 3
def white_chocolate_chips_ounces := mms_grams / 2
def raisins_grams := 2 * mms_grams
def ounce_to_grams := 28.35

-- Convert white chocolate chips to grams
def white_chocolate_chips_grams := white_chocolate_chips_ounces * ounce_to_grams

-- Total grams of all ingredients
def total_grams_all_ingredients := chocolate_chips_grams + mms_grams + white_chocolate_chips_grams + raisins_grams

-- Average grams per cookie
def average_grams_per_cookie := total_grams_all_ingredients / num_cookies

theorem average_grams_per_cookie_is_approximately :
  average_grams_per_cookie ≈ 15.13 :=
sorry

end average_grams_per_cookie_is_approximately_l163_163991


namespace compare_exponents_l163_163016

theorem compare_exponents :
  let a := (3 / 2) ^ 0.1
  let b := (3 / 2) ^ 0.2
  let c := (3 / 2) ^ 0.08
  c < a ∧ a < b := by
  sorry

end compare_exponents_l163_163016


namespace median_qr_less_than_half_l163_163466

theorem median_qr_less_than_half 
  (p : ℕ) [hp_prime : Fact (Nat.Prime p)]
  (mean_qr_lt_half : (∑ x in filter (quadratic_residue p) (range p) \ {0}, x) / 
    (filter (quadratic_residue p) (range p) \ {0}).card < p / 2) :
  ((filter (quadratic_residue p) (range p) \ {0}).nth ((filter (quadratic_residue p) (range p) \ {0}).card / 2)).getOrElse 0 < p / 2 :=
by
  sorry

end median_qr_less_than_half_l163_163466


namespace shopkeeper_profit_percentage_l163_163642

theorem shopkeeper_profit_percentage 
  (cost_price : ℝ := 100) 
  (loss_due_to_theft_percent : ℝ := 30) 
  (overall_loss_percent : ℝ := 23) 
  (remaining_goods_value : ℝ := 70) 
  (overall_loss_value : ℝ := 23) 
  (selling_price : ℝ := 77) 
  (profit_percentage : ℝ) 
  (h1 : remaining_goods_value = cost_price * (1 - loss_due_to_theft_percent / 100)) 
  (h2 : overall_loss_value = cost_price * (overall_loss_percent / 100)) 
  (h3 : selling_price = cost_price - overall_loss_value) 
  (h4 : remaining_goods_value + remaining_goods_value * profit_percentage / 100 = selling_price) :
  profit_percentage = 10 := 
by 
  sorry

end shopkeeper_profit_percentage_l163_163642


namespace trigonometric_ratio_sum_l163_163489

open Real

theorem trigonometric_ratio_sum (x y : ℝ) 
  (h₁ : sin x / sin y = 2) 
  (h₂ : cos x / cos y = 1 / 3) :
  sin (2 * x) / sin (2 * y) + cos (2 * x) / cos (2 * y) = 41 / 57 := 
by
  sorry

end trigonometric_ratio_sum_l163_163489


namespace correct_parentheses_structure_count_l163_163452

/-
  A correct parentheses structure is a sequence with n pairs of parentheses such that
  in any prefix of the sequence, the number of right parentheses does not exceed the number of
  left parentheses.
  A Dyck path of 2n steps is a piecewise linear path that connects the points (0,0) and (0,2n),
  has segment vectors (1,1) and (1,-1), and lies entirely in the upper half-plane x ≥ 0.
-/
def correct_parentheses_structure (n : ℕ) : Type := 
  { s : list (ℤ × ℤ) // s.length = 2 * n ∧ ∀ k < s.length, let (x1, y1) := s.nth_le k (by linarith) in x1 >= 0 }

def isValidBracketStructure : list char → ℕ → Prop
| [], 0 := true
| '(':xs, n := isValidBracketStructure xs (n + 1)
| ')':xs, n := n > 0 ∧ isValidBracketStructure xs (n - 1)
| _, _ := false

noncomputable def catalan : ℕ → ℕ
| 0 := 1
| n := ∑ k in finset.range n, catalan k * catalan (n - k - 1)

theorem correct_parentheses_structure_count (n : ℕ) :
  ∃ count : ℕ, count = catalan n ∧
  (∃ s, correct_parentheses_structure n s ∧ isValidBracketStructure s.to_list n) :=
begin
  sorry
end

end correct_parentheses_structure_count_l163_163452


namespace sequence_sum_l163_163746

def sequence (n : ℕ) : ℕ → ℕ
| 0 := 1
| 1 := 1
| (k + 2) := (-1)^(k + 1) * (3 * sequence k + 1)

def sum_upto (f : ℕ → ℤ) (n : ℕ) : ℤ :=
(list.range (n + 1)).map f |>.sum

theorem sequence_sum : 
  let a := sequence in 
  let S := sum_upto in 
  S a 15 = (7 * ((3 ^ 8 - 1) / 16) - 6) :=
by
  sorry

end sequence_sum_l163_163746


namespace lyle_friends_sandwich_and_juice_l163_163505

theorem lyle_friends_sandwich_and_juice : 
  ∀ (sandwich_cost juice_cost lyle_money : ℝ),
    sandwich_cost = 0.30 → 
    juice_cost = 0.20 → 
    lyle_money = 2.50 → 
    (⌊lyle_money / (sandwich_cost + juice_cost)⌋.toNat - 1) = 4 :=
by
  intros sandwich_cost juice_cost lyle_money hc_sandwich hc_juice hc_money
  have cost_one_set := sandwich_cost + juice_cost
  have number_of_sets := lyle_money / cost_one_set
  have friends := (number_of_sets.toNat - 1)
  have friends_count := 4
  sorry

end lyle_friends_sandwich_and_juice_l163_163505


namespace determine_parameters_inequality_solution_l163_163744

variables {a b c x : ℝ}

theorem determine_parameters (h : ∀ x : ℝ, ax^2 - 3x + 6 > 4 -> x < 1 ∨ x > b)
  : a = 1 ∧ b = 2 :=
sorry

theorem inequality_solution (h_eq : a = 1) (h_b : b = 2)
  : (c > 2 → ∀ x : ℝ, (2 < x ∧ x < c → x^2 - (2 + c)x + 2c < 0)) ∧ 
    (c < 2 → ∀ x : ℝ, (c < x ∧ x < 2 → x^2 - (2 + c)x + 2c < 0)) ∧
    (c = 2 → ∀ x : ℝ, ¬ (x^2 - (2 + c)x + 2c < 0)) :=
sorry

end determine_parameters_inequality_solution_l163_163744


namespace angle_ABC_and_BCD_l163_163449

variables (A B C D : Type*) 
  [parallelogram A B C D]
  [angle_ABD : angle A B D = 2 * angle B C D]

theorem angle_ABC_and_BCD (x : ℝ) (h1 : x = angle B C D) (h2 : angle A B D = 2 * x) :
  (angle B C D = 60) ∧ (angle A B C = 120) :=
by
  sorry

end angle_ABC_and_BCD_l163_163449


namespace set_intersection_identity_l163_163060

noncomputable def U := Set.univ
def A : Set ℝ := {x | x^2 - 2 * x < 0}
def B : Set ℝ := {x | x ≥ 1 }
def complement_U_B : Set ℝ := U \ B
def intersection_A_complement_U_B : Set ℝ := A ∩ complement_U_B

theorem set_intersection_identity : intersection_A_complement_U_B = {x | 0 < x ∧ x < 1} := 
by {
  sorry
}

end set_intersection_identity_l163_163060


namespace find_value_of_theta_l163_163035

noncomputable def internal_angle_triangle (θ p : ℝ) : Prop :=
  (∃ θ, 0 < θ ∧ θ < π) ∧ (4 * (sin θ)^2 + p * (sin θ) - 2 = 0) ∧ (4 * (cos θ)^2 + p * (cos θ) - 2 = 0)

theorem find_value_of_theta (θ p : ℝ) (h : internal_angle_triangle θ p) : θ = 3 * Real.pi / 4 :=
  sorry

end find_value_of_theta_l163_163035


namespace prove_m_greater_than_0_l163_163879

-- Define the conditions
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

def domain_R (f : ℝ → ℝ) : Prop :=
  ∀ x, true -- implies domain is ℝ

def condition_ineq (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2, x1 ≠ x2 → (f (x1) - f (x2)) / (x1 - x2) > 1

-- Define the final statement
theorem prove_m_greater_than_0
  (f : ℝ → ℝ) 
  (h1 : is_odd f) 
  (h2 : domain_R f)
  (h3 : condition_ineq f)
  (m : ℝ) 
  (h4 : f (m) > m) : m > 0 :=
by
  sorry

end prove_m_greater_than_0_l163_163879


namespace alfreds_gain_percent_l163_163971

-- Define constants for the purchase price, repair cost, and selling price
def purchase_price : ℝ := 4700
def repair_cost : ℝ := 1000
def selling_price : ℝ := 5800

-- Define a function to compute the total cost
def total_cost : ℝ := purchase_price + repair_cost

-- Define a function to compute the gain
def gain : ℝ := selling_price - total_cost

-- Define a function to compute the gain percent
def gain_percent : ℝ := (gain / total_cost) * 100

-- The main theorem stating the gain percent is approximately 1.75
theorem alfreds_gain_percent : gain_percent ≈ 1.75 :=
by
  -- Proof is not required for this task
  sorry

end alfreds_gain_percent_l163_163971


namespace max_S_value_l163_163019

theorem max_S_value (θ : ℝ) (h1 : π / 4 < θ) (h2 : θ < π / 2) : 
  ∃ M : ℝ, (∀ θ, (π / 4 < θ → θ < π / 2 → S θ ≤ M)) ∧ M = (sqrt 5 - 1) / 2 :=
by
  sorry

def S (θ : ℝ) : ℝ := sin (2 * θ) - cos θ ^ 2

end max_S_value_l163_163019


namespace max_value_of_expression_l163_163689

theorem max_value_of_expression (x y z : ℤ) 
  (h1 : x * y + x + y = 20) 
  (h2 : y * z + y + z = 6) 
  (h3 : x * z + x + z = 2) : 
  x^2 + y^2 + z^2 ≤ 84 :=
sorry

end max_value_of_expression_l163_163689


namespace color_preference_l163_163157

-- Define the conditions
def total_students := 50
def girls := 30
def boys := 20

def girls_pref_pink := girls / 3
def girls_pref_purple := 2 * girls / 5
def girls_pref_blue := girls - girls_pref_pink - girls_pref_purple

def boys_pref_red := 2 * boys / 5
def boys_pref_green := 3 * boys / 10
def boys_pref_orange := boys - boys_pref_red - boys_pref_green

-- Proof statement
theorem color_preference :
  girls_pref_pink = 10 ∧
  girls_pref_purple = 12 ∧
  girls_pref_blue = 8 ∧
  boys_pref_red = 8 ∧
  boys_pref_green = 6 ∧
  boys_pref_orange = 6 :=
by
  sorry

end color_preference_l163_163157


namespace choose_four_socks_l163_163176

theorem choose_four_socks : ∃ n : ℕ, n = @fintype.card {c : fin 7 // true}
  ∧ (finset.filter (λ (s : finset (fin 7)), s.card = 4) (finset.powerset (finset.univ : finset (fin 7)))).card = 35 :=
by sorry

end choose_four_socks_l163_163176


namespace jacob_age_proof_l163_163445

-- Definitions based on given conditions
def rehana_current_age : ℕ := 25
def rehana_age_in_five_years : ℕ := rehana_current_age + 5
def phoebe_age_in_five_years : ℕ := rehana_age_in_five_years / 3
def phoebe_current_age : ℕ := phoebe_age_in_five_years - 5
def jacob_current_age : ℕ := 3 * phoebe_current_age / 5

-- Statement of the problem
theorem jacob_age_proof :
  jacob_current_age = 3 :=
by 
  -- Skipping the proof for now
  sorry

end jacob_age_proof_l163_163445


namespace base_b_equivalence_l163_163082

theorem base_b_equivalence (b : ℕ) (h : (2 * b + 4) ^ 2 = 5 * b ^ 2 + 5 * b + 4) : b = 12 :=
sorry

end base_b_equivalence_l163_163082


namespace average_weight_increase_l163_163867

theorem average_weight_increase (A : ℝ) :
  let initial_weight := 8 * A
  let new_weight := initial_weight - 65 + 89
  let new_average := new_weight / 8
  let increase := new_average - A
  increase = (89 - 65) / 8 := 
by 
  sorry

end average_weight_increase_l163_163867


namespace distinct_positive_real_roots_probability_l163_163862

/-- Define the set from which a, b, and c are chosen -/
def int_set : Finset ℤ := Finset.range 21 ∪ Finset.range 21.map (equiv.neg ℤ).toFun

/-- Define the function to test for distinct positive real roots -/
def has_distinct_positive_real_roots (a b c : ℤ) : Prop :=
  let discriminant := b * b - 4 * a * c in
  discriminant > 0 ∧
  ((-b + Int.sqrt discriminant) > 0 ∧ (-b - Int.sqrt discriminant) > 0)

noncomputable def probability_no_distinct_positive_real_roots : ℚ :=
  (Finset.card (Finset.univ.filter (λ a b c, ¬has_distinct_positive_real_roots a b c))) / (21 * 21 * 21 : ℚ)

/-- Theorem stating the required probability -/
theorem distinct_positive_real_roots_probability :
  probability_no_distinct_positive_real_roots = 8825 / 9261 :=
sorry

end distinct_positive_real_roots_probability_l163_163862


namespace triangle_acute_angle_sufficient_condition_triangle_acute_angle_not_necessary_condition_l163_163124

theorem triangle_acute_angle_sufficient_condition
  (a b c : ℝ) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  a ≤ (b + c) / 2 → b^2 + c^2 > a^2 :=
sorry

theorem triangle_acute_angle_not_necessary_condition
  (a b c : ℝ) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  b^2 + c^2 > a^2 → ¬ (a ≤ (b + c) / 2) :=
sorry

end triangle_acute_angle_sufficient_condition_triangle_acute_angle_not_necessary_condition_l163_163124


namespace burritos_needed_l163_163800

-- Defining the conditions as constants
constant cheese_per_burrito : ℕ := 4
constant cheese_per_taco : ℕ := 9
constant total_cheese_needed : ℕ := 37

-- Define the number of burritos 'B' to be found
variable (B : ℕ)

-- The theorem stating the problem and proof structure
theorem burritos_needed (h : 4 * B + 9 = 37) : B = 7 :=
by
  sorry

end burritos_needed_l163_163800


namespace remaining_problems_to_grade_l163_163924

variable (worksheets_total graded_worksheets problems_per_worksheet : ℕ)
variable (h1 : worksheets_total = 16)
variable (h2 : graded_worksheets = 8)
variable (h3 : problems_per_worksheet = 4)

theorem remaining_problems_to_grade : 
    worksheets_total - graded_worksheets = 8 →
    8 * problems_per_worksheet = 32 :=
by
  intros h
  have h4 : worksheets_total - graded_worksheets = 8 := by exact h
  rw [h4]
  have h5 : 8 * problems_per_worksheet = 32 := by rw [h3]; norm_num
  exact h5

#check remaining_problems_to_grade

end remaining_problems_to_grade_l163_163924


namespace second_trail_length_l163_163462

theorem second_trail_length 
  (speed_first_trail : ℕ) (distance_first_trail : ℕ)
  (speed_second_trail : ℕ) (break_time : ℕ) 
  (time_difference : ℕ) 
  (h1 : distance_first_trail = 20)
  (h2 : speed_first_trail = 5)
  (h3 : speed_second_trail = 3)
  (h4 : break_time = 1)
  (h5 : time_difference = 1) : 
  let time_first_trail := distance_first_trail / speed_first_trail in 
  ∀ (x : ℕ), ((x / 2 / speed_second_trail + break_time + x / 2 / speed_second_trail) = time_first_trail + time_difference) → x = 12 :=
by
  intros x h
  unfold time_first_trail at h
  rw [h1, h2] at h
  have : x / 3 + 1 = 5 := by sorry
  exact this

end second_trail_length_l163_163462


namespace imaginary_part_of_fraction_l163_163495

-- Define the imaginary unit
def i : ℂ := ⟨0, 1⟩ 

-- State the given problem
theorem imaginary_part_of_fraction :
  (im (frac (4 - 3 * i) i) = -4) :=
by
  sorry

end imaginary_part_of_fraction_l163_163495


namespace center_of_circle_in_second_quadrant_l163_163775

theorem center_of_circle_in_second_quadrant (a b : ℝ) 
  (h1 : a < 0) 
  (h2 : b > 0) : 
  ∃ (q : ℕ), q = 2 := 
by 
  sorry

end center_of_circle_in_second_quadrant_l163_163775


namespace integer_solutions_l163_163348

theorem integer_solutions (x : ℤ) (h1 : x > 1) :
  8.58 * log 4 x + log 2 (sqrt x - 1) < log 2 (log (sqrt 5) 5) ↔ x = 2 ∨ x = 3 := by
  sorry

end integer_solutions_l163_163348


namespace fourth_vertex_of_square_l163_163453

noncomputable def z1 : ℂ := (3 + 1 * complex.I) / (1 - 1 * complex.I)
def z2 : ℂ := -2 + complex.I
def z3 : ℂ := 0

theorem fourth_vertex_of_square :
  ∃ (z4 : ℂ), z4 = -1 + 3 * complex.I ∧
  ∃ (z1 z2 z3 : ℂ), z1 = (3 + 1 * complex.I) / (1 - 1 * complex.I) ∧ z2 = -2 + complex.I ∧ z3 = 0 :=
begin
  use -1 + 3 * complex.I,
  split,
  { -- show that the fourth vertex is -1 + 3i
    refl,
  },
  use z1, use z2, use z3,
  split, 
  { refl }, split,  
  { refl },   
  { refl }
end

end fourth_vertex_of_square_l163_163453


namespace magical_stack_l163_163551

theorem magical_stack (n : ℕ) (h1 : 3 * n = 303) :
  let A := list.range' (2 * n + 1) n,
      B := list.range' (n + 1) n,
      C := list.range' 1 n,
      stack := C.zip_with (λx y => x :: y :: []) B.flatten.zip_with (λx y => x :: y :: []) A.flatten.flatten
  in stack.nth (100) = some 101 → n = 101 :=
by
  sorry

end magical_stack_l163_163551


namespace law_of_cosines_l163_163265

open Real

-- Define the vertices of the triangle
variables (A B C : Point)

-- Define the sides of the triangle
def AB := dist A B
def AC := dist A C
def BC := dist B C

-- Define the angle at A
def angleBAC : ℝ := angle B A C

-- Define the cosine of the angle BAC
def cosBAC := cos angleBAC

-- Al-Kashi's theorem (law of cosines)
theorem law_of_cosines (A B C : Point) :
  BC ^ 2 = AB ^ 2 + AC ^ 2 - 2 * AB * AC * cosBAC := sorry

end law_of_cosines_l163_163265


namespace probability_at_least_7_heads_in_9_tosses_l163_163629

theorem probability_at_least_7_heads_in_9_tosses :
  (let n_flips := 9;
       total_outcomes := 2 ^ n_flips;
       successful_outcomes := Nat.choose n_flips 7 + Nat.choose n_flips 8 + Nat.choose n_flips 9) in
  successful_outcomes / total_outcomes = 23 / 256 := 
by
  sorry

end probability_at_least_7_heads_in_9_tosses_l163_163629


namespace min_shift_value_even_function_l163_163391

noncomputable def determinant (a₁ a₂ a₃ a₄ : ℂ) : ℂ :=
  a₁ * a₄ - a₂ * a₃

noncomputable def f (x : ℝ) : ℂ :=
  determinant (complex.sqrt 3) (complex.sin x) 1 (complex.cos x)

noncomputable def shifted_f (x n : ℝ) : ℂ :=
  f(x + n)

theorem min_shift_value_even_function:
  ∃ k : ℤ, ∃ n : ℝ, n > 0 ∧ shifted_f(x, n) = shifted_f(-x, n) ∧ n = (k * π - π / 6) := 
begin
  use 1,
  use (5 * π / 6),
  sorry
end

end min_shift_value_even_function_l163_163391


namespace red_before_green_probability_l163_163289

open Classical

noncomputable def probability_red_before_green (total_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ) : ℚ :=
  let total_arrangements := (Nat.choose (total_chips - 1) green_chips)
  let favorable_arrangements := Nat.choose (total_chips - red_chips - 1) (green_chips - 1)
  favorable_arrangements / total_arrangements

theorem red_before_green_probability :
  probability_red_before_green 8 4 3 = 3 / 7 :=
sorry

end red_before_green_probability_l163_163289


namespace find_length_of_iron_bar_l163_163298

variables (A V_ball m n : ℕ) (L : ℕ)

-- Cross-sectional area of iron bar (48 cm²)
def cross_sectional_area (x : ℕ) := A = 48

-- Volume of each iron ball (8 cm³)
def volume_of_ball (x : ℕ) := V_ball = 8

-- Number of iron balls (720)
def number_of_balls (x : ℕ) := m = 720

-- Number of iron bars (10)
def number_of_bars (x : ℕ) := n = 10

-- Volume equality condition
def volume_equality (x : ℕ) := n * A * L = m * V_ball

theorem find_length_of_iron_bar (x : ℕ) :
  cross_sectional_area x →
  volume_of_ball x →
  number_of_balls x →
  number_of_bars x →
  volume_equality x → 
  L = 12 :=
by
  intro hA hVb hm hn hveq
  sorry

end find_length_of_iron_bar_l163_163298


namespace total_amount_paid_l163_163861

def p1 := 20
def p2 := p1 + 2
def p3 := p2 + 3
def p4 := p3 + 4

theorem total_amount_paid : p1 + p2 + p3 + p4 = 96 :=
by
  sorry

end total_amount_paid_l163_163861


namespace incorrect_observation_value_l163_163211

-- Definitions stemming from the given conditions
def initial_mean : ℝ := 100
def corrected_mean : ℝ := 99.075
def number_of_observations : ℕ := 40
def correct_observation_value : ℝ := 50

-- Lean theorem statement to prove the incorrect observation value
theorem incorrect_observation_value (initial_mean corrected_mean correct_observation_value : ℝ) (number_of_observations : ℕ) :
  (initial_mean * number_of_observations - corrected_mean * number_of_observations + correct_observation_value) = 87 := 
sorry

end incorrect_observation_value_l163_163211


namespace bank_policies_for_retirees_favorable_l163_163209

-- Define the problem statement
theorem bank_policies_for_retirees_favorable :
  (∃ banks : Type,
    (∃ retirees: Type,
      (∃ policies : banks → Prop,
        (∀ bank, (policies bank → ∃ retiree: retirees,
          (let conscientious := True
          in let stable_income := True 
          in let attract_funds := True 
          in let save_preference := True 
          in let regular_income := True 
          in let long_term_deposit := True
          in conscientious ∧ stable_income ∧ attract_funds ∧ save_preference ∧ regular_income ∧ long_term_deposit))))) :=
begin
  sorry
end

end bank_policies_for_retirees_favorable_l163_163209


namespace geom_seq_general_formula_sum_first_n_terms_formula_l163_163370

namespace GeometricArithmeticSequences

def geom_seq_general (a_n : ℕ → ℝ) (n : ℕ) : Prop :=
  a_n 1 = 1 ∧ (2 * a_n 3 = a_n 2) → a_n n = 1 / (2 ^ (n - 1))

def sum_first_n_terms (a_n b_n : ℕ → ℝ) (S_n T_n : ℕ → ℝ) (n : ℕ) : Prop :=
  b_n 1 = 2 ∧ S_n 3 = b_n 2 + 6 → 
  T_n n = 6 - (n + 3) / (2 ^ (n - 1))

theorem geom_seq_general_formula :
  ∀ a_n : ℕ → ℝ, ∀ n : ℕ, geom_seq_general a_n n :=
by sorry

theorem sum_first_n_terms_formula :
  ∀ a_n b_n : ℕ → ℝ, ∀ S_n T_n : ℕ → ℝ, ∀ n : ℕ, sum_first_n_terms a_n b_n S_n T_n n :=
by sorry

end GeometricArithmeticSequences

end geom_seq_general_formula_sum_first_n_terms_formula_l163_163370


namespace max_correct_answers_l163_163099

theorem max_correct_answers :
  ∀ (a b c : ℕ), a + b + c = 60 ∧ 4 * a - c = 112 → a ≤ 34 :=
by
  sorry

end max_correct_answers_l163_163099


namespace certain_number_k_l163_163767

theorem certain_number_k :
  ∃ k : ℤ, ∀ (p q : ℕ), p > 1 ∧ q > 1 ∧ p + q = 36 ∧ 17 * (p + 1) = k * (q + 1) → k = 2 :=
begin
  sorry,
end

end certain_number_k_l163_163767


namespace points_comparison_l163_163055

def quadratic_function (m x : ℝ) : ℝ :=
  (x + m - 3) * (x - m) + 3

def point_on_graph (m x y : ℝ) : Prop :=
  y = quadratic_function m x

theorem points_comparison (m x1 x2 y1 y2 : ℝ)
  (h1 : point_on_graph m x1 y1)
  (h2 : point_on_graph m x2 y2)
  (hx : x1 < x2)
  (h_sum : x1 + x2 < 3) :
  y1 > y2 := 
  sorry

end points_comparison_l163_163055


namespace rook_captures_knight_l163_163968

-- Define the chessboard and initial positions for the rook and knight
structure Position := (row : ℕ) (col : ℕ)

-- Define the move function for the Rook
def rook_move (r : Position) (n : Position) : Position := 
  if r.row = 2 then 
    Position.mk 2 n.col -- moves to the middle row or stays
  else 
    Position.mk r.row r.col -- does not move if not the first move

-- Define the alternating move strategy
def alternate_move (r n : Position) : Position := 
  if n.row = 3 then 
    Position.mk 2 n.col -- diagonal move if knight is at row 3
  else if n.row = 1 then 
    Position.mk 2 n.col -- diagonal move if knight is at row 1
  else 
    Position.mk r.row r.col -- position remains if knight is in unreachable position

-- Main theorem to state chase strategy
theorem rook_captures_knight (B : ∀ (r : Position), r.row ∈ {1, 2, 3}) 
  (rook_moves_first : rook_move (Position.mk 2 1) (Position.mk 3 2)) : 
  ∃ (move_strategy : Position → Position → Position), 
  ∀ (r n : Position), move_strategy r n = alternate_move r n :=
by
  -- strategy to keep the rook and knight moving alternately
  existsi alternate_move
  sorry

end rook_captures_knight_l163_163968


namespace exists_zero_in_interval_l163_163742

noncomputable def f (x : ℝ) : ℝ := (3 / x) - log x / log 2

theorem exists_zero_in_interval : ∃ x ∈ Ioo 2 3, f x = 0 := by
  sorry

end exists_zero_in_interval_l163_163742


namespace bank_policies_for_retirees_favorable_l163_163208

-- Define the problem statement
theorem bank_policies_for_retirees_favorable :
  (∃ banks : Type,
    (∃ retirees: Type,
      (∃ policies : banks → Prop,
        (∀ bank, (policies bank → ∃ retiree: retirees,
          (let conscientious := True
          in let stable_income := True 
          in let attract_funds := True 
          in let save_preference := True 
          in let regular_income := True 
          in let long_term_deposit := True
          in conscientious ∧ stable_income ∧ attract_funds ∧ save_preference ∧ regular_income ∧ long_term_deposit))))) :=
begin
  sorry
end

end bank_policies_for_retirees_favorable_l163_163208


namespace functional_inequality_l163_163492

variables {ℝ : Type*} [linear_ordered_field ℝ]

def rstar := {x : ℝ // x ≠ 0}

noncomputable def af (x : ℝ) : ℝ := sorry  -- Define af accordingly

theorem functional_inequality (a : ℝ) (f : rstar → ℝ) :
  (∀ (x y z : rstar), af (x.val / y.val) + af (x.val / z.val) - (f x) * (f ⟨(y.val + z.val) / 2, sorry⟩) ≥ a^2) →
  (∀ x : rstar, f x = a) :=
sorry

end functional_inequality_l163_163492


namespace cannot_cover_98_l163_163011

theorem cannot_cover_98 :
  ∀ n : ℕ, 
    let l_shaped_covers := 3
    let unit_square_covers := 1
    let chessboard_size := n^2
    let conditions := chessboard_size % 3 = 0 ∨ chessboard_size % 3 = 1
    n = 98 ↔ ¬conditions :=
begin
  sorry
end

end cannot_cover_98_l163_163011


namespace count_points_l163_163026

def ellipse : Prop := ∀ (x y : ℝ), (x^2 / 4) + (y^2 / 2) = 1

def foci (f1 f2 : ℝ × ℝ) : Prop := 
  f1 = (-√2, 0) ∧ f2 = (√2, 0)

def right_angle_triangle (P : ℝ × ℝ) (f1 f2 : ℝ × ℝ) : Prop := 
  ∃ (A B C : ℝ × ℝ), (A = f1 ∧ B = P ∧ C = f2) ∧
  (∃ (θ : ℝ), θ = π / 2 ∧ 
    ((B.1 - A.1)*(C.1 - A.1) + (B.2 - A.2)*(C.2 - A.2) = 0 ∨
     (B.1 - C.1)*(A.1 - C.1) + (B.2 - C.2)*(A.2 - C.2) = 0 ∨
     (A.1 - B.1)*(C.1 - B.1) + (A.2 - B.2)*(C.2 - B.2) = 0))

theorem count_points (P : ℝ × ℝ) : 
  ellipse P.1 P.2 →
  foci (-√2, 0) (√2, 0) →
  right_angle_triangle P (-√2, 0) (√2, 0) →
  (finset.filter (λ P, ellipse P.1 P.2 ∧ right_angle_triangle P (-√2, 0) (√2, 0)) 
  (finset.univ : finset (ℝ × ℝ))).card = 6 :=
by {
  sorry
}

end count_points_l163_163026


namespace part1_part2_l163_163476

namespace ArithmeticSeq

variable (n : ℕ)

def a (n : ℕ) := -2 * n + 15
def S (n : ℕ) := n * (a 1 + a n) / 2
def T (n : ℕ) := 
  if n ≤ 7 then 
    -n^2 + 14 * n
  else 
    n^2 - 14 * n + 98

theorem part1 (a2 : a 2 = 11) (S10 : S 10 = 40) :
  ∀ n, a n = -2 * n + 15 := 
sorry

theorem part2 (a2 : a 2 = 11) (S10 : S 10 = 40) :
  ∀ n, T n = 
    if n ≤ 7 then 
      -n^2 + 14 * n
    else 
      n^2 - 14 * n + 98 := 
sorry

end ArithmeticSeq

end part1_part2_l163_163476


namespace max_pawns_non_attacking_l163_163238

-- Define the conditions under which the pawns are placed
def white_attacks (x y : Nat) (board : Nat → Nat → Bool) : Bool :=
  if x + 1 < 9 then (board (x + 1) (y - 1)) || (board (x + 1) (y + 1)) else false

def black_attacks (x y : Nat) (board : Nat → Nat → Bool) : Bool :=
  if x - 1 >= 0 then (board (x - 1) (y - 1)) || (board (x - 1) (y + 1)) else false

-- Define the main theorem that needs to be proven
theorem max_pawns_non_attacking : 
  (∃ (board : Nat → Nat → Bool), 
    (∀ x y, 
      (board x y → 
        ¬white_attacks x y board ∧ ¬black_attacks x y board)) 
    → (∑ i in range 9, ∑ j in range 9, if board i j then 1 else 0) = 56) := 
by sorry

end max_pawns_non_attacking_l163_163238


namespace sampling_proof_l163_163783

axiom num_classes : ℕ
axiom students_per_class : ℕ
axiom chosen_student_number : ℕ
axiom systematic_sampling : Prop

noncomputable def sampling_method (n_classes : ℕ) (students_per_cls : ℕ) (chosen_stud_num : ℕ) : Prop :=
  ∀ (c : ℕ), c < n_classes → chosen_stud_num = 40 → systematic_sampling

theorem sampling_proof (n_classes : ℕ) (students_per_cls : ℕ) (chosen_stud_num : ℕ) :
  n_classes = 12 → students_per_cls = 50 → chosen_stud_num = 40 → sampling_method n_classes students_per_cls chosen_stud_num :=
by
  intros h1 h2 h3
  unfold sampling_method
  assume c hc hchosen
  sorry

end sampling_proof_l163_163783


namespace simplest_quadratic_radical_l163_163088

theorem simplest_quadratic_radical (a : ℝ) (h : sqrt (3 * a - 4) = sqrt 2) : a = 2 :=
sorry

end simplest_quadratic_radical_l163_163088


namespace projection_of_u_onto_w_is_correct_l163_163817

open Matrix

theorem projection_of_u_onto_w_is_correct : 
  let v := ⟨[1, 2, 3], by simp⟩ : Vector 3 ℝ
  let u := ⟨[2, 1, -1], by simp⟩ : Vector 3 ℝ
  let proj_v_w := ⟨[2, 4, 6], by simp⟩ : Vector 3 ℝ
  ∃ (w : Vector 3 ℝ),
    (proj_v_w = ((v ⬝ w) / (w ⬝ w)) • w) ∧ 
    ∃ proj_u_w, proj_u_w = ((u ⬝ w) / (w ⬝ w)) • w ∧ 
                 proj_u_w = ⟨[1/14, 1/7, 3/14], by simp⟩ :=
by
  sorry

end projection_of_u_onto_w_is_correct_l163_163817


namespace find_special_number_l163_163344

-- Definitions derived from conditions
def is_three_digit_number (x : ℕ) := x ≥ 100 ∧ x < 1000
def digits_do_not_repeat (x : ℕ) : Prop :=
  let a := x / 100 in
  let b := (x % 100) / 10 in
  let c := x % 10 in
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def reverse_digits (x : ℕ) : ℕ :=
  let a := x / 100 in
  let b := (x % 100) / 10 in
  let c := x % 10 in
  100 * c + 10 * b + a

def same_digits (x y : ℕ) : Prop :=
  let a1 := x / 100 in
  let b1 := (x % 100) / 10 in
  let c1 := x % 10 in
  let a2 := y / 100 in
  let b2 := (y % 100) / 10 in
  let c2 := y % 10 in
  a1 ≠ b1 ∧ b1 ≠ c1 ∧ a1 ≠ c1 ∧
  (a1 = a2 ∨ a1 = b2 ∨ a1 = c2) ∧
  (b1 = a2 ∨ b1 = b2 ∨ b1 = c2) ∧
  (c1 = a2 ∨ c1 = b2 ∨ c1 = c2)

-- Lean 4 statement to prove
theorem find_special_number (x : ℕ) (hx : is_three_digit_number x) (hd : digits_do_not_repeat x) :
  let diff := x - reverse_digits x in
  is_three_digit_number diff ∧ same_digits x diff → x = 954 ∨ x = 459 :=
by 
  sorry

end find_special_number_l163_163344


namespace paper_clips_in_two_cases_l163_163254

-- Defining the problem statement in Lean 4
theorem paper_clips_in_two_cases (c b : ℕ) :
  2 * (c * b * 300) = 2 * c * b * 300 :=
by
  sorry

end paper_clips_in_two_cases_l163_163254


namespace train_speed_is_60_kmph_l163_163963

noncomputable def speed_of_train_in_kmph (length_meters time_seconds : ℝ) : ℝ :=
  (length_meters / time_seconds) * 3.6

theorem train_speed_is_60_kmph (length_meters time_seconds : ℝ) :
  length_meters = 50 → time_seconds = 3 → speed_of_train_in_kmph length_meters time_seconds = 60 :=
by
  intros h_length h_time
  simp [speed_of_train_in_kmph, h_length, h_time]
  norm_num
  sorry

end train_speed_is_60_kmph_l163_163963


namespace range_of_a_zeros_of_F_inequality_sqrt_e_l163_163743

-- Definitions of the functions f, g, and h based on the given conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.log (x + 1))
def g (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * x^3 - a * x
def h (x : ℝ) : ℝ := Real.exp x - 1

-- Part (I): Prove the range of a for which f(x) ≤ h(x) for all x ≥ 0
theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ 0 → f a x ≤ h x) ↔ a ≤ 1 := sorry

-- Part (II): Investigate the number of zeros of F(x) = h(x) - g(x) for x < 0
def F (a : ℝ) (x : ℝ) : ℝ := h x - g a x

theorem zeros_of_F (a : ℝ) : 
  (a ≤ -1 → ∀ x : ℝ, x < 0 → F a x ≠ 0) ∧ 
  (a > -1 → ∃ x : ℝ, x < 0 ∧ F a x = 0) := sorry

-- Part (III): Prove the inequality involving the 10th root of e
theorem inequality_sqrt_e : (1095 / 1000) < Real.exp (1 / 10) ∧ Real.exp (1 / 10) < (3000 / 2699) := sorry

end range_of_a_zeros_of_F_inequality_sqrt_e_l163_163743


namespace find_parallel_line_l163_163346

def point_A := (-1 : ℝ, 1 : ℝ)
def line1 (x y : ℝ) := x + 3 * y + 4 = 0

theorem find_parallel_line (x y : ℝ) : 
  (∃ m : ℝ, (x + 3 * y + m = 0) ∧ (x = -1 ∧ y = 1)) → (x + 3 * y - 2 = 0) := 
by
  sorry

end find_parallel_line_l163_163346


namespace jacob_age_proof_l163_163446

-- Definitions based on given conditions
def rehana_current_age : ℕ := 25
def rehana_age_in_five_years : ℕ := rehana_current_age + 5
def phoebe_age_in_five_years : ℕ := rehana_age_in_five_years / 3
def phoebe_current_age : ℕ := phoebe_age_in_five_years - 5
def jacob_current_age : ℕ := 3 * phoebe_current_age / 5

-- Statement of the problem
theorem jacob_age_proof :
  jacob_current_age = 3 :=
by 
  -- Skipping the proof for now
  sorry

end jacob_age_proof_l163_163446


namespace number_of_correct_relations_l163_163295

-- Declare the sets a and b used in the conditions
def a : Type := sorry
def b : Type := sorry

-- Condition translations into Lean
def P1 : Prop := ({a, b} ⊆ {a,b})
def P2 : Prop := ({a, b} = {b, a})
def P3 : Prop := (∅ = {∅})
def P4 : Prop := (0 ∈ {0})
def P5 : Prop := (∅ ∈ {∅})
def P6 : Prop := (∅ ⊆ {∅})

-- Proof problem in Lean 4 statement: prove the question is equal to the answer given the conditions
theorem number_of_correct_relations : (P1 ∧ P2 ∧ ¬P3 ∧ P4 ∧ P5 ∧ P6) ↔ (5 = 5) :=
by sorry

end number_of_correct_relations_l163_163295


namespace frog_jumps_l163_163142

theorem frog_jumps (p q : ℕ) (hpq_coprime : Nat.coprime p q) (h_frog : ∃ (n m : ℕ), n * p = m * q):
  ∀ (d : ℕ), d < p + q → ∃ (a b : ℤ), (a - b) * p - (a - b) * q = d := 
by
  sorry

end frog_jumps_l163_163142


namespace train_speed_l163_163967

theorem train_speed (length_of_train time_to_cross : ℕ) (h_length : length_of_train = 50) (h_time : time_to_cross = 3) : 
  (length_of_train / time_to_cross : ℝ) * 3.6 = 60 := by
  sorry

end train_speed_l163_163967


namespace f_is_odd_and_periodic_l163_163829

local notation "ℝ" => Real

noncomputable def f (x : ℝ) : ℝ := cos (2 * x - π / 2)

theorem f_is_odd_and_periodic :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ p = π) :=
sorry

end f_is_odd_and_periodic_l163_163829


namespace count_quadrilaterals_with_circumcenter_l163_163699

namespace Quadrilaterals

-- Definitions for the types of quadrilaterals
def isSquare (q : Type) : Prop := sorry
def isCyclicQuadrilateralNotRectangle (q : Type) : Prop := sorry
def isKiteNotRhombus (q : Type) : Prop := sorry
def isParallelogramNotRectangleNorRhombus (q : Type) : Prop := sorry
def isIsoscelesTrapezoidNotParallelogram (q : Type) : Prop := sorry

-- Definitions for the conditions provided
def hasCircumcenter (q : Type) : Prop := sorry

-- Problem statement: Proving the number of quadrilaterals
theorem count_quadrilaterals_with_circumcenter :
  let quadrilaterals := [isSquare, isCyclicQuadrilateralNotRectangle, isKiteNotRhombus, isParallelogramNotRectangleNorRhombus, isIsoscelesTrapezoidNotParallelogram] in
  (finset.filter (fun q => hasCircumcenter q) quadrilaterals.to_finset).card = 3 :=
sorry

end Quadrilaterals

end count_quadrilaterals_with_circumcenter_l163_163699


namespace root_shrinking_method_l163_163244

theorem root_shrinking_method {a p x α β : ℝ} (h₁ : a ≠ 0) (h₂ : p ≠ 0) (h₃ : a * α^2 + b * α + c = 0) (h₄ : a * β^2 + b * β + c = 0) :
  let x1 := α / p,
      x2 := β / p
  in
  (p^2 * a * x1^2 + p * b * x1 + c = 0) ∧ (p^2 * a * x2^2 + p * b * x2 + c = 0) :=
by
  sorry

end root_shrinking_method_l163_163244


namespace prop_B_contrapositive_correct_l163_163603

/-
Proposition B: The contrapositive of the proposition 
"If x^2 < 1, then -1 < x < 1" is 
"If x ≥ 1 or x ≤ -1, then x^2 ≥ 1".
-/
theorem prop_B_contrapositive_correct :
  (∀ (x : ℝ), x^2 < 1 → -1 < x ∧ x < 1) ↔ (∀ (x : ℝ), (x ≥ 1 ∨ x ≤ -1) → x^2 ≥ 1) :=
sorry

end prop_B_contrapositive_correct_l163_163603


namespace polynomial_integer_values_l163_163821

/-- Given a natural number n and a polynomial P of degree n such that P(k^2) is an integer for all k in the range [0, n], prove that P(k^2) is an integer for all integers k. -/
theorem polynomial_integer_values (n : ℕ) (P : Real[X]) (h_deg : P.degree = n) 
  (h_int : ∀ k ∈ Finset.range (n+1), (P.eval (k : ℝ) ∈ Int)) :
  ∀ k : ℤ, (P.eval (k^2 : ℝ) ∈ Int) :=
sorry

end polynomial_integer_values_l163_163821


namespace probability_of_three_primes_out_of_six_l163_163308

-- Define the conditions
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

-- Given six 12-sided fair dice
def total_dice : ℕ := 6
def sides : ℕ := 12

-- Probability of rolling a prime number on one die
def prime_probability : ℚ := 5 / 12

-- Probability of rolling a non-prime number on one die
def non_prime_probability : ℚ := 7 / 12

-- Number of ways to choose 3 dice from 6 to show a prime number
def combination (n k : ℕ) : ℕ := n.choose k
def choose_3_out_of_6 : ℕ := combination total_dice 3

-- Combined probability for exactly 3 primes and 3 non-primes
def combined_probability : ℚ :=
  (prime_probability ^ 3) * (non_prime_probability ^ 3)

-- Total probability
def total_probability : ℚ :=
  choose_3_out_of_6 * combined_probability

-- Main theorem statement
theorem probability_of_three_primes_out_of_six :
  total_probability = 857500 / 5177712 :=
by
  sorry

end probability_of_three_primes_out_of_six_l163_163308


namespace net_square_removal_l163_163164

theorem net_square_removal (net : array (fin 10) (fin 2 × fin 2)) (parallelepiped_size : fin 2 × fin 2) : 
  ∃ (positions : fin 10 → Prop), ∀ pos, positions pos ↔ valid_net_after_removal net parallelepiped_size pos :=
by
  sorry

end net_square_removal_l163_163164


namespace triangle_area_sqrt_sum_l163_163228

noncomputable def equilateral_triangle_area_with_circles (R : ℝ) : ℝ :=
  let d := 2 * R in
  let s := 2 * d in
  (sqrt 3 / 4) * s^2

theorem triangle_area_sqrt_sum (R : ℝ) (hR : R = 32) :
  ∃ (a b : ℕ), (equilateral_triangle_area_with_circles R = sqrt a + sqrt b) ∧ (a + b = 12291) :=
by
  sorry

end triangle_area_sqrt_sum_l163_163228


namespace prove_complex_problem_l163_163015

noncomputable def complex_problem_statement : Prop :=
  let A : ℝ × ℝ := (-2, 2)
  let B : ℝ × ℝ := (- (5 / 2) * Real.sqrt 3, 2)
  let F : ℝ × ℝ := (-3, 0) in
  (B.fst ^ 2 / 25 + B.snd ^ 2 / 16 = 1) ∧
  (∀ B₀ : ℝ × ℝ, (B₀.fst ^ 2 / 25 + B₀.snd ^ 2 / 16 = 1) →
  (|((B₀.fst - A.fst) ^ 2 + (B₀.snd - A.snd) ^ 2).sqrt + (5 / 3) * ((B₀.fst - F.fst) ^ 2 + (B₀.snd - F.snd) ^ 2).sqrt| ≥ 
   ((B.fst - A.fst) ^ 2 + (B.snd - A.snd) ^ 2).sqrt + (5 / 3) * ((B.fst - F.fst) ^ 2 + (B.snd - F.snd) ^ 2).sqrt))

theorem prove_complex_problem : complex_problem_statement :=
  by
    sorry

end prove_complex_problem_l163_163015


namespace pens_distribution_l163_163012

theorem pens_distribution (friends : ℕ) (pens : ℕ) (at_least_one : ℕ) 
  (h1 : friends = 4) (h2 : pens = 10) (h3 : at_least_one = 1) 
  (h4 : ∀ f : ℕ, f < friends → at_least_one ≤ f) :
  ∃ ways : ℕ, ways = 142 := 
sorry

end pens_distribution_l163_163012


namespace arithmetic_sequence_general_term_and_sum_l163_163110

theorem arithmetic_sequence_general_term_and_sum :
  (∃ (a₁ d : ℤ), a₁ + d = 14 ∧ a₁ + 4 * d = 5 ∧ ∀ n : ℤ, a_n = a₁ + (n - 1) * d ∧ (∀ N : ℤ, N ≥ 1 → S_N = N * ((2 * a₁ + (N - 1) * d) / 2) ∧ N = 10 → S_N = 35)) :=
sorry

end arithmetic_sequence_general_term_and_sum_l163_163110


namespace polygonal_line_divisible_by_4_l163_163273

theorem polygonal_line_divisible_by_4 
  (closed_polygonal_line : ℕ → ℕ × ℕ)
  (H V : ℕ)
  (odd_lengths : ∀ i, (closed_polygonal_line i).1 % 2 = 1 ∨ (closed_polygonal_line i).2 % 2 = 1)
  (sum_horizontal_even : ∑ i in range H, (closed_polygonal_line i).1 = ∑ i in range H, (closed_polygonal_line i).2)
  (sum_vertical_even : ∑ i in range V, (closed_polygonal_line i).1 = ∑ i in range V, (closed_polygonal_line i).2) :
  (H + V) % 4 = 0 :=
sorry

end polygonal_line_divisible_by_4_l163_163273


namespace find_y_l163_163187

theorem find_y (y : ℝ) (h : sqrt (2 + sqrt (3 * y - 4)) = sqrt 9) : y = 53 / 3 :=
by
  sorry

end find_y_l163_163187


namespace digit_packages_for_room_numbers_l163_163563

theorem digit_packages_for_room_numbers : 
  ∀ (rooms1 rooms2 : Nat → Prop),
  (∀ n, 300 ≤ n ∧ n ≤ 350 → rooms1 n) →
  (∀ n, 400 ≤ n ∧ n ≤ 450 → rooms2 n) →
  (∀ n, rooms1 n ∨ rooms2 n → n.digits (10) ⊆ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) →
  ∃ packages : ℕ, packages = 51 :=
begin
  sorry
end

end digit_packages_for_room_numbers_l163_163563


namespace log_sawing_time_l163_163632

theorem log_sawing_time :
  ∀ (length section_length sawing_time : ℕ), 
    length = 10 → 
    section_length = 1 → 
    sawing_time = 3 → 
    (length / section_length - 1) * sawing_time = 27 :=
by
  intros length section_length sawing_time h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end log_sawing_time_l163_163632


namespace message_forwarding_time_l163_163429

theorem message_forwarding_time :
  ∃ n : ℕ, (∀ m : ℕ, (∀ p : ℕ, (∀ q : ℕ, 1 + (2 * (2 ^ n)) - 1 = 2047)) ∧ n = 10) :=
sorry

end message_forwarding_time_l163_163429


namespace probability_relative_frequency_l163_163885

noncomputable def event_probability_dev (n : ℕ) (p : ℝ) (ε : ℝ) : ℝ :=
2 * Real.cdf (ε * Real.sqrt (n / (p * (1 - p))))

theorem probability_relative_frequency (n : ℕ) (p : ℝ) (ε : ℝ)
  (h₁ : n = 625) (h₂ : p = 0.8) (h₃ : ε = 0.04) :
  event_probability_dev n p ε = 0.9876 :=
by
  subst h₁
  subst h₂
  subst h₃
  sorry

end probability_relative_frequency_l163_163885


namespace height_of_water_a_height_of_water_b_height_of_water_c_l163_163276

noncomputable def edge_length : ℝ := 10  -- Edge length of the cube in cm.
noncomputable def angle_deg : ℝ := 20   -- Angle in degrees.

noncomputable def volume_a : ℝ := 100  -- Volume in cm^3 for case a)
noncomputable def height_a : ℝ := 2.53  -- Height in cm for case a)

noncomputable def volume_b : ℝ := 450  -- Volume in cm^3 for case b)
noncomputable def height_b : ℝ := 5.94  -- Height in cm for case b)

noncomputable def volume_c : ℝ := 900  -- Volume in cm^3 for case c)
noncomputable def height_c : ℝ := 10.29  -- Height in cm for case c)

theorem height_of_water_a :
  ∀ (edge_length angle_deg volume_a : ℝ), volume_a = 100 → height_a = 2.53 := by 
  sorry

theorem height_of_water_b :
  ∀ (edge_length angle_deg volume_b : ℝ), volume_b = 450 → height_b = 5.94 := by 
  sorry

theorem height_of_water_c :
  ∀ (edge_length angle_deg volume_c : ℝ), volume_c = 900 → height_c = 10.29 := by 
  sorry

end height_of_water_a_height_of_water_b_height_of_water_c_l163_163276


namespace ellipse_standard_eq_max_area_triangle_line_eq_l163_163377

-- First problem: standard equation of the ellipse C
theorem ellipse_standard_eq (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : 2 * b = 2) (h4 : sqrt 2 / 2 = sqrt (a^2 - b^2) / a) :
  ∀ x y, (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 2 + y^2 = 1) := by
  sorry

-- Second problem: equation of the line l when maximizing the area of triangle AOB
theorem max_area_triangle_line_eq (a b k m : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : 2 * b = 2)
                                   (h4 : sqrt 2 / 2 = sqrt (a^2 - b^2) / a) (h5 : k ≠ 0)
                                   (h6 : 2 * k^2 + 1 = 2 * m) (h7 : 2 * k^2 + 1 > m^2) :
  ∀ x y, (x^2 / 2 + y^2 = 1) → (y = k * x + m) ∨ (y = -k * x + m) :=
by
  sorry

end ellipse_standard_eq_max_area_triangle_line_eq_l163_163377


namespace Lyle_friends_sandwich_juice_l163_163500

/-- 
Lyle wants to buy himself and his friends a sandwich and a pack of juice. 
A sandwich costs $0.30 while a pack of juice costs $0.20. Given Lyle has $2.50, 
prove that he can buy sandwiches and juice for 4 of his friends.
-/
theorem Lyle_friends_sandwich_juice :
  let sandwich_cost := 0.30
  let juice_cost := 0.20
  let total_money := 2.50
  let total_cost_one_set := sandwich_cost + juice_cost
  let total_sets := total_money / total_cost_one_set
  total_sets - 1 = 4 :=
by
  sorry

end Lyle_friends_sandwich_juice_l163_163500


namespace solve_trig_eq_l163_163544

-- Definitions and conditions from part a)
def trig_eq (x : Real) : Prop :=
  (\cos (2 * x) - 2 * \cos (4 * x))^2 = 9 + (\cos (5 * x))^2

-- Statement to prove from part c)
theorem solve_trig_eq (x : Real) (k : Int) : trig_eq x ↔ ∃ k : ℤ, x = (π / 2) + k * π := by
  sorry

end solve_trig_eq_l163_163544


namespace part_one_part_two_range_tan_theta_l163_163393

noncomputable def f (x a : ℝ) : ℝ := -x^3 + a * x^2 + 1

theorem part_one (a : ℝ) : 
  (∀ x ∈ (Ioo 0 (2/3 : ℝ)), deriv (λ x, f x a) x > 0) ∧ 
  (∀ x ∈ (Ico (2/3 : ℝ) real.top), deriv (λ x, f x a) x < 0) → 
  a = 1 :=
sorry

theorem part_two_range_tan_theta {a : ℝ} (ha : a ∈ Ioo (3 / 2) real.top) :
  ∀ x ∈ Icc 0 1, 
  if a ≤ 3 then 
    0 ≤ (fun x : ℝ => -3 * x^2 + 2 * a * x) x ∧ (fun x : ℝ => -3 * x^2 + 2 * a * x) x ≤ a^2 / 3 
  else 
    0 ≤ (fun x : ℝ => -3 * x^2 + 2 * a * x) x ∧ (fun x : ℝ => -3 * x^2 + 2 * a * x) x ≤ 2 * a - 3 :=
sorry

end part_one_part_two_range_tan_theta_l163_163393


namespace divisibility_condition_l163_163338

theorem divisibility_condition (M C D U A q1 q2 q3 r1 r2 r3 : ℕ)
  (h1 : 10 = A * q1 + r1)
  (h2 : 10 * r1 = A * q2 + r2)
  (h3 : 10 * r2 = A * q3 + r3) :
  (U + D * r1 + C * r2 + M * r3) % A = 0 ↔ (1000 * M + 100 * C + 10 * D + U) % A = 0 :=
sorry

end divisibility_condition_l163_163338


namespace M_inter_N_eq_l163_163410

open Set

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {3, 4, 5, 6}

theorem M_inter_N_eq : M ∩ N = {3, 4} := 
by 
  sorry

end M_inter_N_eq_l163_163410


namespace ellipse_equation_of_arithmetic_sequence_focus_distance_line_equation_max_area_triangle_OAB_l163_163040

-- Problem 1
theorem ellipse_equation_of_arithmetic_sequence_focus_distance
  (a b : ℝ) (h : a > b > 0)
  (c := sqrt 3)
  (h_sum : 3 * a = 6)
  (h_eq : a = (a - c) + sqrt 3) :
  ∃ b, ∀ x y : ℝ, (x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1) ↔ (x ^ 2 / 4 + y ^ 2 = 1) :=
sorry

-- Problem 2
theorem line_equation_max_area_triangle_OAB
  (a b : ℝ)
  (h_eq : ∀ x y : ℝ, (x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1) ↔ (x ^ 2 / 4 + y ^ 2 = 1))
  (F : ℝ × ℝ) (λ : ℝ)
  (h_intersect : ∀ A B : ℝ × ℝ, A.1 ^ 2 / 4 + A.2 ^ 2 = 1 ∧ B.1 ^ 2 / 4 + B.2 ^ 2 = 1 ∧ 2 * A + B = λ * F)
  (max_area : ∀ t : ℝ, t = 2 / sqrt 5 -> area_of_triangle (0, 0) (t, t) (t, -t) = 1) :
  ∃ m t : ℝ, (line_eq m t = (λ y, (m * y + t)) ∧ (line_eq m t) (2 / 3) = ( (1/3) , t) ) :=
sorry

end ellipse_equation_of_arithmetic_sequence_focus_distance_line_equation_max_area_triangle_OAB_l163_163040


namespace ticket_distribution_ways_l163_163584

-- Define the set of tickets and people
def tickets := {1, 2, 3, 4, 5}
def people := {A, B, C, D}

-- Define the conditions for ticket distribution
def valid_distribution (dist : (people → finset ℕ)) : Prop :=
  ∀ p ∈ people, 1 ≤ dist p ∧ dist p ≤ 2 ∧ (∀ x y, {x,y} ⊆ dist p → abs (x - y) = 1)

-- Define the main theorem
theorem ticket_distribution_ways :
  (finset.univ.filter valid_distribution).card = 96 :=
sorry

end ticket_distribution_ways_l163_163584


namespace Part_1_Part_2_l163_163471

noncomputable def S (n : ℕ) : ℤ := (40 : ℤ) -- Given S₁₀ = 40
def a (n : ℕ) : ℤ := if n = 2 then 11 else 0 -- Given a₂ = 11
def a_n (n : ℕ) : ℤ := if n = 2 then a n else 15 - 2 * (n - 1) -- General form for a_n

theorem Part_1 {a1 d : ℤ} (h₁ : a_n 2 = 11) (h₂ : ∑ i in range 10, a_n (i + 1) = 40) :
  ∀ n, a_n n = 15 - 2 * (n-1) := by
  sorry

def absSum1 (n : ℕ) : ℤ := -n^2 + 14 * n
def absSum2 (n : ℕ) : ℤ := n^2 - 14 * n + 98

theorem Part_2 {a_n : ℕ → ℤ} (h₁ : ∀ n, a_n n = 15 - 2 * (n-1)) :
  (∀  n, 1 ≤ n ∧ n ≤ 7 → ∑ i in range n, |a_n (i + 1)| = -n^2 + 14 * n) ∧
  (∀ n, n ≥ 8 → ∑ i in range n, |a_n (i + 1)| = n^2 - 14 * n + 98) := by
  sorry

end Part_1_Part_2_l163_163471


namespace cloth_sold_l163_163646

theorem cloth_sold (x : ℕ) (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) : x = 85 :=
by
  assume h1 : total_selling_price = 8925,
  assume h2 : profit_per_meter = 15,
  assume h3 : cost_price_per_meter = 90,
  have total_selling_price_per_meter := cost_price_per_meter + profit_per_meter,
  have equation := total_selling_price_per_meter * x = total_selling_price,
  rw [h1, h2, h3] at *,
  sorry

end cloth_sold_l163_163646


namespace triangle_perimeter_l163_163291

theorem triangle_perimeter
  (x : ℝ) 
  (h : x^2 - 6 * x + 8 = 0)
  (a b c : ℝ)
  (ha : a = 2)
  (hb : b = 4)
  (hc : c = x)
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 10 := 
sorry

end triangle_perimeter_l163_163291


namespace product_mn_eq_neg24_l163_163064

-- Given vectors a and b
variables (m n : ℝ)
def a : ℝ × ℝ × ℝ := (m, -6, 2)
def b : ℝ × ℝ × ℝ := (4, n, 1)

-- Given that a is parallel to b
def parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b = (k * a.1, k * a.2, k * a.3)

-- Proof statement: if a and b are parallel, then m * n = -24
theorem product_mn_eq_neg24 (hmn : parallel (m, -6, 2) (4, n, 1)) : m * n = -24 :=
sorry

end product_mn_eq_neg24_l163_163064


namespace radius_of_smaller_base_l163_163869

theorem radius_of_smaller_base (C1 C2 : ℝ) (r : ℝ) (l : ℝ) (A : ℝ) 
    (h1 : C2 = 3 * C1) 
    (h2 : l = 3) 
    (h3 : A = 84 * Real.pi) 
    (h4 : C1 = 2 * Real.pi * r) 
    (h5 : C2 = 2 * Real.pi * (3 * r)) :
    r = 7 := 
by
  -- proof steps here
  sorry

end radius_of_smaller_base_l163_163869


namespace minimum_spending_l163_163948

theorem minimum_spending (wholesale1 wholesale2 wholesale3 : ℕ) (markup1 markup2 markup3 : ℕ)
  (disc1_week1 disc2_week1 disc3_week1 : ℕ)
  (disc1_week2 disc2_week2 disc3_week2 : ℕ)
  (retail1 retail2 retail3 final1 final2 final3 total : ℕ) :
  wholesale1 = 200 ∧ wholesale2 = 250 ∧ wholesale3 = 300 ∧
  markup1 = 20 ∧ markup2 = 25 ∧ markup3 = 30 ∧
  disc1_week1 = 20 ∧ disc2_week1 = 15 ∧ disc3_week1 = 10 ∧
  disc1_week2 = 25 ∧ disc2_week2 = 20 ∧ disc3_week2 = 15 ∧
  retail1 = wholesale1 + (markup1 * wholesale1) / 100 ∧
  retail2 = wholesale2 + (markup2 * wholesale2) / 100 ∧
  retail3 = wholesale3 + (markup3 * wholesale3) / 100 ∧
  final1 = retail1 - (disc1_week2 * retail1) / 100 ∧
  final2 = retail2 - (disc2_week2 * retail2) / 100 ∧
  final3 = retail3 - (disc3_week2 * retail3) / 100 ∧
  total = final1 + final2 + final3
  → total = 76150 :=
begin
  sorry
end

end minimum_spending_l163_163948


namespace seq_arithmetic_l163_163497

noncomputable def f (x n : ℝ) : ℝ := (x - 1)^2 + n

def a_n (n : ℝ) : ℝ := n
def b_n (n : ℝ) : ℝ := n + 4
def c_n (n : ℝ) : ℝ := (b_n n)^2 - (a_n n) * (b_n n)

theorem seq_arithmetic (n : ℕ) (hn : 0 < n) :
  ∃ d, d ≠ 0 ∧ ∀ n, c_n (↑n : ℝ) = c_n (↑n + 1 : ℝ) - d := 
sorry

end seq_arithmetic_l163_163497


namespace problem1_problem2_l163_163183

theorem problem1 (x : ℚ) (h : x ≠ -4) : (3 - x) / (x + 4) = 1 / 2 → x = 2 / 3 :=
by
  sorry

theorem problem2 (x : ℚ) (h : x ≠ 1) : x / (x - 1) - 2 * x / (3 * (x - 1)) = 1 → x = 3 / 2 :=
by
  sorry

end problem1_problem2_l163_163183


namespace problem_proof_l163_163980

-- Define the variables
variable (x_sides y_sides : ℕ)

-- Conditions given
axiom h1 : (180 * (x_sides - 2)) + (180 * (y_sides - 2)) = 1440
axiom h2 : x_sides / y_sides = 1 / 3

-- The conjecture to be proven
theorem problem_proof :
  let x_sides := 3,
  let y_sides := 9
  in (180 * x_sides + 360 * y_sides = 2160
      ∧ x_sides = x_sides
      ∧ y_sides = 3 * x_sides
      ∧ x_sides = 3
      ∧ y_sides = 9
      ∧ (x_sides + y_sides) = 12
      ∧ ∑ i in range(2), (if i = 0 then 360 else 360) = 720
      ∧ ∑ i in range(2 * x_sides/9), 180 * i = 1080
      ∧ y_sides = 9
      ∧ x_sides = 3
      ∧ y_sides * (y_sides - 3) / 2 = 27) := sorry

end problem_proof_l163_163980


namespace num_six_digit_asc_digits_l163_163759

theorem num_six_digit_asc_digits : 
  ∃ n : ℕ, n = (Nat.choose 9 3) ∧ n = 84 := 
by
  sorry

end num_six_digit_asc_digits_l163_163759


namespace cylindrical_container_capacity_l163_163618

theorem cylindrical_container_capacity :
  ∀ (C H : ℕ) (pi : ℝ) (bushel_volume : ℝ),
  C = 54 → H = 18 → pi = 3 → bushel_volume = 1.62 →
  let r := C / (2 * pi),
  let volume_cubic_feet := pi * r^2 * H,
  let bushels := volume_cubic_feet / bushel_volume,
  bushels ≈ 2700 :=
by
  intros C H pi bushel_volume hC hH hpi hbushel_volume r volume_cubic_feet bushels,
  rw [hC, hH, hpi, hbushel_volume],
  let r := 54 / (2 * 3),
  simp at r, 
  let volume_cubic_feet := 3 * r^2 * 18,
  have hr : r = 9, by norm_num,
  rw hr at *,
  have hvolume : volume_cubic_feet = 4374, by norm_num,
  let bushels := 4374 / 1.62,
  have hbushels : bushels ≈ 2700, by norm_num,
  exact hbushels,

end cylindrical_container_capacity_l163_163618


namespace probability_two_different_colors_l163_163779

-- Define the setup
def initial_bag : List (String × ℕ) := [("blue", 6), ("red", 5), ("yellow", 4)]

-- Define the condition for drawing a chip and adding a blue chip if the drawn chip is blue
def draw_and_add (bag : List (String × ℕ)) : String -> List (String × ℕ) :=
  fun color =>
    if color = "blue" then 
      [("blue", (bag.lookup "blue").getD 0 + 1), ("red", bag.lookup "red").getD 0, ("yellow", bag.lookup "yellow").getD 0]
    else 
      bag

-- Define the probability calculation function (stub for illustration, not actual implementation)
def probability_of_different_colors (bag : List (String × ℕ)) : ℚ := sorry

-- Statement of the problem rewritten in Lean 4
theorem probability_two_different_colors :
  probability_of_different_colors initial_bag = 593 / 900 := sorry

end probability_two_different_colors_l163_163779


namespace slope_range_l163_163120

open Real

theorem slope_range (k : ℝ) :
  (∃ b : ℝ, 
    ∃ x1 x2 x3 : ℝ,
      (x1 + x2 + x3 = 0) ∧
      (x1 ≥ 0) ∧ (x2 ≥ 0) ∧ (x3 < 0) ∧
      ((kx1 + b) = ((x1 + 1) / (|x1| + 1))) ∧
      ((kx2 + b) = ((x2 + 1) / (|x2| + 1))) ∧
      ((kx3 + b) = ((x3 + 1) / (|x3| + 1)))) →
  (0 < k ∧ k < (2 / 9)) :=
sorry

end slope_range_l163_163120


namespace problem_bc_correctness_l163_163126

theorem problem_bc_correctness :
  (∀ {A B C a b c : ℝ},
    (a / b / c = 2 / 3 / 4) →
    ∃ k : ℝ, a = 2 * k ∧ b = 3 * k ∧ c = 4 * k ∧
      (let cos_C := (a^2 + b^2 - c^2) / (2 * a * b) in cos_C < 0) ∧ C > 90) ∧
  (∀ {A B : ℝ},
    (sin A > sin B) → A > B) :=
by
  sorry

end problem_bc_correctness_l163_163126


namespace L1_L2_sums_circumscribe_sum_K1_K2_not_parallel_equally_directed_falsify_circumscription_l163_163530

-- Definitions
variable {K1 K2 : Type} [convex_curve K1] [convex_curve K2]
variable {L1 L2 : Type} [polygon L1] [polygon L2]

-- Assumptions of the problem
variable (L1_circumscribes_K1 : circumscribes L1 K1)
variable (L2_circumscribes_K2 : circumscribes L2 K2)
variable (L1_L2_parallel_equally_directed : parallel_and_equally_directed L1 L2)

-- Main Proof Statement
theorem L1_L2_sums_circumscribe_sum_K1_K2 :
  circumscribes (L1 + L2) (K1 + K2) :=
by
  -- Skipping the actual proof
  sorry

-- Statement about non-pairwise parallel and equally directed case
theorem not_parallel_equally_directed_falsify_circumscription :
  ¬(parallel_and_equally_directed L1 L2) →
  ¬circumscribes (L1 + L2) (K1 + K2) :=
by
  -- Skipping the actual proof
  sorry

end L1_L2_sums_circumscribe_sum_K1_K2_not_parallel_equally_directed_falsify_circumscription_l163_163530


namespace match_roots_l163_163196

noncomputable def original_eq (x : ℝ) : ℝ := 
  (3 * x^2) / (x - 2) - (3 * x + 8) / 4 + (9 - 9 * x) / (x - 2) + 2
          
def transformed_eq (x : ℝ) : ℝ := 9 * x^2 - 26 * x + 4
          
theorem match_roots :
  (∀ x : ℝ, original_eq x = 0 → transformed_eq x = 0) ∧ 
  (transformed_eq 3 = 0 ∧ transformed_eq (1 / 3) = 0) :=
by
  -- We state what needs to be proven
  have root_3 : transformed_eq 3 = 0 := sorry,
  have root_1_3 : transformed_eq (1 / 3) = 0 := sorry,
  have roots_equivalence : ∀ x : ℝ, original_eq x = 0 → transformed_eq x = 0 := sorry,
  exact ⟨roots_equivalence, ⟨root_3, root_1_3⟩⟩

end match_roots_l163_163196


namespace find_f_2012_l163_163379

noncomputable def f : ℤ → ℤ := sorry

axiom even_function : ∀ x : ℤ, f (-x) = f x
axiom f_1 : f 1 = 1
axiom f_2011_ne_1 : f 2011 ≠ 1
axiom max_property : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b)

theorem find_f_2012 : f 2012 = 1 := sorry

end find_f_2012_l163_163379


namespace triangle_base_length_l163_163864

theorem triangle_base_length (height : ℝ) (area : ℝ) (base : ℝ) 
  (h_height : height = 6) (h_area : area = 9) 
  (h_formula : area = (1/2) * base * height) : 
  base = 3 :=
by
  sorry

end triangle_base_length_l163_163864


namespace kenneth_payment_per_oz_l163_163159

open_locale classical -- Use classical logic

noncomputable def nicholas_paid_more (K : ℝ) : Prop :=
  let kenneth_total := 700 * K in
  let nicholas_total := 6 * 700 * K in
  nicholas_total = kenneth_total + 140000 

theorem kenneth_payment_per_oz (K : ℝ) 
  (h1 : nicholas_paid_more K) : K = 40 :=
by
  rw [nicholas_paid_more, let_def, let_def] at h1
  sorry

end kenneth_payment_per_oz_l163_163159


namespace part1_part2_1_part2_2_l163_163090

-- Definitions for Part 1.
def A (x : ℤ) := (x - 1) / (x - 4)
def B (x : ℤ) := (x - 7) / (x - 4)
def perfect_value_2 := 2

-- Part 1 proof statement.
theorem part1 (x : ℤ) : A(x) + B(x) = perfect_value_2 := by
  sorry

-- Definitions for Part 2.
def C (x : ℤ) := (3 * x - 4) / (x - 2)
def D (E : ℤ) (x : ℤ) := E / (x^2 - 4)
def perfect_value_3 := 3

-- Part 2 proof statement for E.
theorem part2_1 (x : ℤ) (E : ℤ) (hE : D(E, x) = perfect_value_3 - C(x)) : E = -2 * (x + 2) := by
  sorry

-- Part 2 proof statement for x.
theorem part2_2 (x : ℤ) (E : ℤ) (hE : D(E, x) = perfect_value_3 - C(x)) (hx : x = 1) : x = 1 := by
  sorry

end part1_part2_1_part2_2_l163_163090


namespace no_member_divisible_by_2003_l163_163572

noncomputable def a : ℕ → ℤ 
| 0     := 1
| (n+1) := a n * 2001 + b n
  
noncomputable def b : ℕ → ℤ 
| 0     := 4
| (n+1) := b n * 2001 + a n

theorem no_member_divisible_by_2003 :
  ∀ n : ℕ, a n % 2003 ≠ 0 ∧ b n % 2003 ≠ 0 :=
sorry

end no_member_divisible_by_2003_l163_163572


namespace calculate_expression_l163_163660

theorem calculate_expression : (3 / 4 - 1 / 8) ^ 5 = 3125 / 32768 :=
by
  sorry

end calculate_expression_l163_163660


namespace smallest_positive_period_maximum_area_triangle_l163_163401

open Real

noncomputable def f (w x : ℝ) : ℝ :=
  sin (w * x) ^ 2 - sin (w * x - π / 6) ^ 2

theorem smallest_positive_period (w : ℝ) (hw : 1/2 < w ∧ w < 1) :
  (∃ T > 0, ∀ x, f w (x + T) = f w x ∧ T = 6 * π / 5) :=
sorry

variables (A : ℝ) (a b c : ℝ) (w : ℝ)
hypotheses (h_a : a = 1) (h_func : f w (3/5 * A) = 1 / 4) (h_A : A = π / 3)
          (h_w : w = 5 / 6) (h_cos : cos A = 1 / 2) (h_bc : b ^ 2 + c ^ 2 = b * c + 1)
include h_a h_func h_A h_w h_cos h_bc

theorem maximum_area_triangle :
  (∃ S, ∀ b c, S = 1 / 2 * b * c * sin (A) ∧ S ≤ sqrt 3 / 4) :=
sorry

end smallest_positive_period_maximum_area_triangle_l163_163401


namespace find_sum_inverse_a_b_l163_163823

def a_seq : ℕ → ℝ
| 0       := -3
| (n + 1) := a_seq n + b_seq n + 2 * real.sqrt ((a_seq n)^2 + (b_seq n)^2)

def b_seq : ℕ → ℝ
| 0       := 2
| (n + 1) := a_seq n + b_seq n - 2 * real.sqrt ((a_seq n)^2 + (b_seq n)^2)

theorem find_sum_inverse_a_b :
  (1 / a_seq 2013) + (1 / b_seq 2013) = -1 / 6 :=
by
  sorry

end find_sum_inverse_a_b_l163_163823


namespace gift_bag_combinations_l163_163950

theorem gift_bag_combinations (giftBags tissuePapers tags : ℕ) (h1 : giftBags = 10) (h2 : tissuePapers = 4) (h3 : tags = 5) : 
  giftBags * tissuePapers * tags = 200 := 
by 
  sorry

end gift_bag_combinations_l163_163950


namespace total_hours_worked_l163_163659

theorem total_hours_worked :
  (∃ (hours_per_day : ℕ) (days : ℕ), hours_per_day = 3 ∧ days = 6) →
  (∃ (total_hours : ℕ), total_hours = 18) :=
by
  intros
  sorry

end total_hours_worked_l163_163659


namespace jim_pages_now_l163_163811

theorem jim_pages_now (rate : ℕ) (total_pages : ℕ) (increase : ℕ) (hours_less : ℕ) :
  (rate = 40) →
  (total_pages = 600) →
  (increase = 150) →
  (hours_less = 4) →
  let hours_before := total_pages / rate in
  let new_hours := hours_before - hours_less in
  let new_rate := rate * increase / 100 in
  let pages_now := new_hours * new_rate in
  pages_now = 660 :=
by
  intros h_rate h_total_pages h_increase h_hours_less,
  let hours_before := total_pages / rate,
  let new_hours := hours_before - hours_less,
  let new_rate := rate * increase / 100,
  let pages_now := new_hours * new_rate,
  have : pages_now = 11 * 60, from calc
    pages_now = (total_pages / rate - hours_less) * (rate * increase / 100) : by
      simp [hours_before, new_hours, new_rate]
    ... = 11 * 60 : by
      simp [h_rate, h_total_pages, h_increase, h_hours_less],
  have : 11 * 60 = 660, by norm_num,
  rw [this, this]
  sorry

end jim_pages_now_l163_163811


namespace cubic_roots_identity_l163_163481

noncomputable def roots_of_cubic (a b c : ℝ) : Prop :=
  (5 * a^3 - 2019 * a + 4029 = 0) ∧ 
  (5 * b^3 - 2019 * b + 4029 = 0) ∧ 
  (5 * c^3 - 2019 * c + 4029 = 0)

theorem cubic_roots_identity (a b c : ℝ) (h_roots : roots_of_cubic a b c) : 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 12087 / 5 :=
by 
  -- proof steps
  sorry

end cubic_roots_identity_l163_163481


namespace coins_sum_equal_l163_163902

theorem coins_sum_equal (n t : ℕ) (first_sheet second_sheet : List ℕ)
  (h_initial : ∀ x ∈ first_sheet, x ≤ n)
  (h_final  : ∀ y ∈ second_sheet, y ≤ t)
  (h_process : ∀ k, (k ∈ first_sheet ∨ k ∈ second_sheet))
  (h_gold_end : all_gold n t) :
  first_sheet.sum = second_sheet.sum := 
  sorry

def all_gold (n t: ℕ) := ∀ m, m = n → t = m + n

end coins_sum_equal_l163_163902


namespace initial_coins_l163_163941

theorem initial_coins (x : ℕ) (ht : ∀ (k : ℕ), 1 ≤ k ∧ k ≤ 15 → (k * fact 14 * x) % (15 ^ 14) = 0) : x = 7862400 :=
sorry

end initial_coins_l163_163941


namespace alternatingSum_100_alternatingSum_even_alternatingSum_odd_l163_163985

open Nat

-- Definition of the series up to a natural number n
def alternatingSum (n : Nat) : Int :=
  ∑ i in Finset.range (n + 1), (-1) ^ (i + 1) * i

-- Proof statement for the first part: 1 - 2 + 3 - 4 + ... + 99 - 100 = -50
theorem alternatingSum_100 : alternatingSum 100 = -50 :=
  sorry

-- Proof statements for the second part
theorem alternatingSum_even (n : Nat) (h : Even n) : alternatingSum n = - (n / 2) :=
  sorry

theorem alternatingSum_odd (n : Nat) (h : Odd n) : alternatingSum n = (n + 1) / 2 :=
  sorry

end alternatingSum_100_alternatingSum_even_alternatingSum_odd_l163_163985


namespace sum_of_common_ratios_is_five_l163_163487

theorem sum_of_common_ratios_is_five {k p r : ℝ} 
  (h1 : p ≠ r)                       -- different common ratios
  (h2 : k ≠ 0)                       -- non-zero k
  (a2 : ℝ := k * p)                  -- term a2
  (a3 : ℝ := k * p^2)                -- term a3
  (b2 : ℝ := k * r)                  -- term b2
  (b3 : ℝ := k * r^2)                -- term b3
  (h3 : a3 - b3 = 5 * (a2 - b2))     -- given condition
  : p + r = 5 := 
by
  sorry

end sum_of_common_ratios_is_five_l163_163487


namespace triangles_formed_l163_163839

-- Define the combinatorial function for binomial coefficients.
def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k)) else 0

-- Given conditions
def points_on_first_line := 6
def points_on_second_line := 8

-- Number of triangles calculation
def total_triangles :=
  binom points_on_first_line 2 * binom points_on_second_line 1 +
  binom points_on_first_line 1 * binom points_on_second_line 2

-- The final theorem to prove
theorem triangles_formed : total_triangles = 288 :=
by
  sorry

end triangles_formed_l163_163839


namespace coordinates_wrt_origin_l163_163874

theorem coordinates_wrt_origin (x y : ℝ) (h : (x, y) = (1, -5)) : (x, y) = (1, -5) :=
by
  simp [h]
  sorry

end coordinates_wrt_origin_l163_163874


namespace angle_measure_l163_163152

noncomputable def angle_ACB (A B : ℝ × ℝ) (C : Type) [topological_space C] [metric_space C] [normed_group C] [normed_space ℝ C] [finite_dimensional ℝ C] : ℝ :=
  if A.1 = 0 ∧ A.2 = 100 ∧ B.1 = 30 ∧ B.2 = -90 then 180 else 0

theorem angle_measure (A B C : Type) [topological_space C] [metric_space C] [normed_group C] [normed_space ℝ C] [finite_dimensional ℝ C]
  (hA : A = (0, 100)) (hB : B = (30, -90)): angle_ACB A B C = 180 :=
by
  simp [angle_ACB, hA, hB]
  sorry

end angle_measure_l163_163152


namespace Part_1_Part_2_l163_163473

noncomputable def S (n : ℕ) : ℤ := (40 : ℤ) -- Given S₁₀ = 40
def a (n : ℕ) : ℤ := if n = 2 then 11 else 0 -- Given a₂ = 11
def a_n (n : ℕ) : ℤ := if n = 2 then a n else 15 - 2 * (n - 1) -- General form for a_n

theorem Part_1 {a1 d : ℤ} (h₁ : a_n 2 = 11) (h₂ : ∑ i in range 10, a_n (i + 1) = 40) :
  ∀ n, a_n n = 15 - 2 * (n-1) := by
  sorry

def absSum1 (n : ℕ) : ℤ := -n^2 + 14 * n
def absSum2 (n : ℕ) : ℤ := n^2 - 14 * n + 98

theorem Part_2 {a_n : ℕ → ℤ} (h₁ : ∀ n, a_n n = 15 - 2 * (n-1)) :
  (∀  n, 1 ≤ n ∧ n ≤ 7 → ∑ i in range n, |a_n (i + 1)| = -n^2 + 14 * n) ∧
  (∀ n, n ≥ 8 → ∑ i in range n, |a_n (i + 1)| = n^2 - 14 * n + 98) := by
  sorry

end Part_1_Part_2_l163_163473


namespace probability_of_non_defective_is_seven_ninetyninths_l163_163100

-- Define the number of total pencils, defective pencils, and the number of pencils selected
def total_pencils : ℕ := 12
def defective_pencils : ℕ := 4
def selected_pencils : ℕ := 5

-- Define the number of ways to choose k elements from n elements (the combination function)
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the total number of ways to choose 5 pencils out of 12
def total_ways : ℕ := combination total_pencils selected_pencils

-- Calculate the number of non-defective pencils
def non_defective_pencils : ℕ := total_pencils - defective_pencils

-- Calculate the number of ways to choose 5 non-defective pencils out of 8
def non_defective_ways : ℕ := combination non_defective_pencils selected_pencils

-- Calculate the probability that all 5 chosen pencils are non-defective
def probability_non_defective : ℚ :=
  non_defective_ways / total_ways

-- Prove that this probability equals 7/99
theorem probability_of_non_defective_is_seven_ninetyninths :
  probability_non_defective = 7 / 99 :=
by
  -- The proof is left as an exercise
  sorry

end probability_of_non_defective_is_seven_ninetyninths_l163_163100


namespace num_teachers_l163_163903

-- This statement involves defining the given conditions and stating the theorem to be proved.
theorem num_teachers (parents students total_people : ℕ) (h_parents : parents = 73) (h_students : students = 724) (h_total : total_people = 1541) :
  total_people - (parents + students) = 744 :=
by
  -- Including sorry to skip the proof, as required.
  sorry

end num_teachers_l163_163903


namespace unique_geometric_sequence_l163_163749

theorem unique_geometric_sequence (a : ℝ) (q : ℝ) (a_n b_n : ℕ → ℝ) 
    (h1 : a > 0) 
    (h2 : a_n 1 = a) 
    (h3 : b_n 1 - a_n 1 = 1) 
    (h4 : b_n 2 - a_n 2 = 2) 
    (h5 : b_n 3 - a_n 3 = 3) 
    (h6 : ∀ n, a_n (n + 1) = a_n n * q) 
    (h7 : ∀ n, b_n (n + 1) = b_n n * q) : 
    a = 1 / 3 := sorry

end unique_geometric_sequence_l163_163749


namespace side_length_of_square_l163_163977

-- Define the areas of the triangles AOR, BOP, and CRQ
def S1 := 1
def S2 := 3
def S3 := 1

-- Prove that the side length of the square OPQR is 2
theorem side_length_of_square (side_length : ℝ) : 
  S1 = 1 ∧ S2 = 3 ∧ S3 = 1 → side_length = 2 :=
by
  intros h
  sorry

end side_length_of_square_l163_163977


namespace dima_cipher_is_good_l163_163333

def dima_cipher_unique_decoding (alphabet_size : ℕ) (max_letter_size : ℕ) (max_word_size : ℕ) : Prop :=
  ∀ cipher : list (string × string), 
    (∀ (ltr : string), string.length (cipher ltr) ≤ max_letter_size) →
    ∀ (encoded_word : string),
      encoded_word.length ≤ max_word_size →
      ∃ unique_decoding : list string, 
        -- Definition of unique decoding here, assuming you have some helper functions defined
       (∀ (decoding1 decoding2 : list string),
        decode(cipher, encoded_word) = decoding1 → decode(cipher, encoded_word) = decoding2 → decoding1 = decoding2)

theorem dima_cipher_is_good : 
  ∀ (alphabet_size : ℕ) (max_letter_size : ℕ) (max_word_size : ℕ),
  alphabet_size = 33 → max_letter_size = 10 → max_word_size = 10000 →
  dima_cipher_unique_decoding alphabet_size max_letter_size max_word_size :=
by 
  intros alphabet_size max_letter_size max_word_size h1 h2 h3 
  rw [h1, h2, h3]
  sorry

end dima_cipher_is_good_l163_163333


namespace bees_moving_in_correct_directions_l163_163907

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

def beeA_position (n : ℕ) : ℝ × ℝ × ℝ :=
  let cycle := n / 3
  let remainder := n % 3
  match remainder with
  | 0 => (cycle : ℝ, cycle : ℝ, cycle : ℝ)
  | 1 => ((cycle : ℝ) + 1, cycle : ℝ, cycle : ℝ)
  | 2 => ((cycle : ℝ) + 1, (cycle : ℝ) + 1, cycle : ℝ)

def beeB_position (n : ℕ) : ℝ × ℝ × ℝ :=
  let cycle := n / 2
  let remainder := n % 2
  match remainder with
  | 0 => (-(cycle : ℝ), -(cycle : ℝ), 0)
  | 1 => (-(cycle : ℝ) - 1, -(cycle : ℝ), 0)

lemma bees_10_feet_apart_at_8 : ∃ (n : ℕ), distance (beeA_position n) (beeB_position n) = 10 :=
by
  -- Note: Proof will go here
  sorry

lemma bees_directions_at_8 : (beeA_position 8).1 > (beeA_position 7).1 ∧ (beeA_position 8).2 = (beeA_position 7).2 ∧
                             (beeB_position 8).1 < (beeB_position 7).1 ∧ (beeB_position 8).2 = (beeB_position 7).2 :=
by
  -- Note: Proof will go here
  sorry

theorem bees_moving_in_correct_directions : ∃ (m n : ℕ), distance (beeA_position m) (beeB_position n) = 10 ∧
                                                    (beeA_position m).1 > (beeA_position (m-1)).1 ∧
                                                    (beeB_position n).1 < (beeB_position (n-1)).1 :=
by
  use 8, 8
  split
  · exact bees_10_feet_apart_at_8
  · exact bees_directions_at_8

end bees_moving_in_correct_directions_l163_163907


namespace charlie_morgan_can_volume_ratio_l163_163990

theorem charlie_morgan_can_volume_ratio :
  let charlie_diameter := 4
      charlie_height := 16
      charlie_usable_factor := 0.75
      morgan_diameter := 16
      morgan_height := 4
      π := Real.pi
      charlie_radius := charlie_diameter / 2
      morgan_radius := morgan_diameter / 2
      charlie_usable_volume := π * (charlie_radius ^ 2) * (charlie_height * charlie_usable_factor)
      morgan_volume := π * (morgan_radius ^ 2) * morgan_height
  in charlie_usable_volume / morgan_volume = 3 / 16 := by
    sorry

end charlie_morgan_can_volume_ratio_l163_163990


namespace evaluate_expression_l163_163684

theorem evaluate_expression (x y : ℤ) (hx : x = 3) (hy : y = 2) : 4 * x^y - 5 * y^x = -4 := by
  sorry

end evaluate_expression_l163_163684


namespace vector_magnitude_range_l163_163750

variable {V : Type} [InnerProductSpace ℝ V]

theorem vector_magnitude_range (a b : V) (h1 : ∥b∥ = 2) (h2 : ∥a∥ = 2 * ∥b - a∥) : 
  ∥a∥ ∈ Set.Icc (4 / 3) 4 :=
by
  sorry

end vector_magnitude_range_l163_163750


namespace plane_equation_l163_163793

-- Point A in 3D Cartesian coordinate system
def A : ℝ × ℝ × ℝ := (1, 0, 2)

-- Point B in 3D Cartesian coordinate system
def B : ℝ × ℝ × ℝ := (1, 1, -1)

-- Direction vector from A to B
def AB : ℝ × ℝ × ℝ := (0, 1, -3)

-- Any point M on the plane
def M (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

-- Vector from A to M
def AM (M : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (M.1 - A.1, M.2 - A.2, M.3 - A.3)

-- Dot product of two vectors
def dot (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Statement to be proved: The plane equation is y - 3z + 6 = 0
theorem plane_equation (x y z : ℝ) (M : ℝ × ℝ × ℝ := (x, y, z)) :
  dot AB (AM M) = 0 ↔ (y - 3 * z + 6 = 0) :=
sorry

end plane_equation_l163_163793


namespace total_chocolate_bar_count_l163_163648

def large_box_count : ℕ := 150
def small_box_count_per_large_box : ℕ := 45
def chocolate_bar_count_per_small_box : ℕ := 35

theorem total_chocolate_bar_count :
  large_box_count * small_box_count_per_large_box * chocolate_bar_count_per_small_box = 236250 :=
by
  sorry

end total_chocolate_bar_count_l163_163648


namespace solvability_of_triangle_with_given_conditions_l163_163323

-- Define the context of the problem
variables {A B C M : Type}
variables (b c : ℝ) (ω : ℝ)
variable [linear_ordered_field ℝ]

-- Assume conditions
def midpoint (B C M : Type) : Prop := dist B M = dist C M
def acute_angle (ω : ℝ) : Prop := 0 < ω ∧ ω < π / 2

-- Main statement
theorem solvability_of_triangle_with_given_conditions (hMidpoint : midpoint B C M) 
    (hAC : dist A C = b) (hAB : dist A B = c) (hAngleAMB : ∠ A M B = ω) (hAcute : acute_angle ω):
  b * tan(ω / 2) ≤ c ∧ c < b :=
sorry

end solvability_of_triangle_with_given_conditions_l163_163323


namespace arithmetic_sqrt_of_9_l163_163191

theorem arithmetic_sqrt_of_9 : ∃ y : ℝ, y ^ 2 = 9 ∧ y ≥ 0 ∧ y = 3 := by
  sorry

end arithmetic_sqrt_of_9_l163_163191


namespace even_derivative_implies_b_zero_l163_163498

noncomputable def f (a b c : ℝ) : ℝ → ℝ := λ x, a * x^3 + b * x^2 + c * x + 2

def f_prime (a b c : ℝ) : ℝ → ℝ := λ x, 3 * a * x^2 + 2 * b * x + c

theorem even_derivative_implies_b_zero (a b c : ℝ) (h : ∀ x, f_prime(a, b, c) x = f_prime(a, b, c) (-x)) : b = 0 :=
by
  sorry

end even_derivative_implies_b_zero_l163_163498


namespace eating_nuts_time_l163_163158

def rate_ms_quick (time: ℝ) (amount: ℝ) := amount / time
def rate_ms_slow (time: ℝ) (amount: ℝ) := amount / time

def combined_rate (r1 r2: ℝ) := r1 + r2

theorem eating_nuts_time:
  let rate_quick := rate_ms_quick 15 0.5 in
  let rate_slow := rate_ms_slow 20 0.5 in
  let combined := combined_rate rate_quick rate_slow in
  let total_amount := 2 in
  (total_amount / combined) = 240 / 7 :=
by
  sorry

end eating_nuts_time_l163_163158


namespace coefficient_of_fourth_term_l163_163871

open BigOperators

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (finset.range (k)).prod (λ i, (n - i) / (k - i))

theorem coefficient_of_fourth_term :
  binomial_coefficient 6 3 * (1 / 2) ^ 3 * 2 ^ 3 = 20 :=
by
  sorry

end coefficient_of_fourth_term_l163_163871


namespace find_second_pay_cut_l163_163534

-- Definitions for conditions
def first_pay_cut_percentage : ℝ := 5 / 100
def third_pay_cut_percentage : ℝ := 15 / 100
def overall_decrease_percentage : ℝ := 27.325 / 100

theorem find_second_pay_cut (x : ℝ) 
  (h : (0.85 * (0.95 - 0.0095 * x) = 0.72675)) : 
  x = 10 :=
begin
  -- The proof is omitted here as we are only required to state the theorem.
  sorry
end

end find_second_pay_cut_l163_163534


namespace solutions_eq_ten_l163_163002

theorem solutions_eq_ten (n : ℕ) : 4x + 6y + 2z = n ∧ x > 0 ∧ y > 0 ∧ z > 0 → n = 32 ∨ n = 33 := 
by
  sorry

end solutions_eq_ten_l163_163002


namespace ones_digit_73_pow_355_l163_163611

theorem ones_digit_73_pow_355 : 
  (73 ^ 355) % 10 = 7 := 
by
  -- Introduce the conditions as local assumptions
  have h1 : 73 % 10 = 3 := rfl
  have h2 : (355 % 4 = 3) := by simp
  -- Find the repeating cycle for the ones digits of powers of 3
  have hcycle : ∀ n, (3 ^ n) % 10 ∈ [3, 9, 7, 1] := by
    intro n
    cases n % 4
    · exact rfl
    · exact rfl
    · exact rfl
    · exact rfl
  sorry -- The detailed proof steps are omitted

end ones_digit_73_pow_355_l163_163611


namespace Ivan_cannot_cut_off_all_heads_l163_163461

-- Defining the number of initial heads
def initial_heads : ℤ := 100

-- Effect of the first sword: Removes 21 heads
def first_sword_effect : ℤ := 21

-- Effect of the second sword: Removes 4 heads and adds 2006 heads
def second_sword_effect : ℤ := 2006 - 4

-- Proving Ivan cannot reduce the number of heads to zero
theorem Ivan_cannot_cut_off_all_heads :
  (∀ n : ℤ, n % 7 = initial_heads % 7 → n ≠ 0) :=
by
  sorry

end Ivan_cannot_cut_off_all_heads_l163_163461


namespace max_rounds_cleared_pass_first_three_rounds_l163_163778

-- Definitions for the conditions given in the problem
def fair_six_sided_die := {1, 2, 3, 4, 5, 6}

noncomputable def roll_sum(n : ℕ) : ℕ := sorry -- function to return sum of rolling n times

def clears_round (n : ℕ) : Prop := roll_sum(n) > 2^n

-- Problem 1:
theorem max_rounds_cleared : ∀ n : ℕ, clears_round n → n ≤ 4 :=
sorry

-- Problem 2:
noncomputable def pass_probability(n : ℕ) : ℝ := sorry -- function to calculate the probability of passing n rounds

theorem pass_first_three_rounds : pass_probability 3 = 100 / 243 :=
sorry

end max_rounds_cleared_pass_first_three_rounds_l163_163778


namespace tangent_circles_count_l163_163132

theorem tangent_circles_count
  (C1 C2 : Circle) (r1 r3 : ℝ)
  (h₀ : r1 = 1) 
  (h₁ : radius C1 = r1)
  (h₂ : radius C2 = r1)
  (h₃ : tangent C1 C2)
  (h₄ : radius C1 = radius C2) :
  ∃ n : ℕ, n = 6 ∧ ∀ C3 : Circle, radius C3 = r3 ∧ tangent C3 C1 ∧ tangent C3 C2 → count C3 = n := 
sorry

end tangent_circles_count_l163_163132


namespace find_integer_n_l163_163347

theorem find_integer_n :
  ∃ n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -4000 [MOD 10] ∧ n = 0 :=
by 
sorry

end find_integer_n_l163_163347


namespace chessboard_placements_l163_163203

theorem chessboard_placements (n : ℕ) (h : n > 2) :
  ∃ placements : ℕ, 
  (∀ x y : ℕ, 1 ≤ x ∧ x ≤ n^2 ∧ 1 ≤ y ∧ y ≤ n^2 ∧ (x - y).nat_abs ≤ n + 1) 
  ∧ placements = 32 :=
by
  sorry

end chessboard_placements_l163_163203


namespace no_equalization_possible_l163_163976

theorem no_equalization_possible : 
  ∀ x : Fin 6 → ℕ, 
    (∀ a b c : Fin 6, (x a < x b ∨ x b < x c ∨ x c < x a)) → 
    (∀ a b : Fin 6, a ≠ b → (∃ k : ℕ, (x 1 + x 4) = 5 + k ∧ (x 2 + x 5) = 7 + k ∧ (x 3 + x 6) = 9 + k) → 
      (∃ a b c : Fin 6, (x a := x a + 1 ∧ x b := x b + 1 ∧ x c := x c + 1 ∨ 
                         x a := x a - 1 ∧ x b := x b - 1 ∧ x c := x c - 1) 
                           ∧ (x a = x b ∧ x b = x c)) → 
  false :=
sorry

end no_equalization_possible_l163_163976


namespace sum_of_ratios_bounds_l163_163917

noncomputable def conditions (n : ℕ) [Fact (n ≥ 3)] (a : ℕ → ℕ) :=
  ∀ i, (a ((i - 1) % n) + a ((i + 1) % n)) % a i = 0

noncomputable def sum_of_ratios (n : ℕ) [Fact (n ≥ 3)] (a : ℕ → ℕ) : ℝ :=
  ∑ i in Finset.range n, (a ((i - 1) % n) + a ((i + 1) % n)) / a i

theorem sum_of_ratios_bounds (n : ℕ) [Fact (n ≥ 3)] (a : ℕ → ℕ) (h : conditions n a) :
  2 * n ≤ sum_of_ratios n a ∧ sum_of_ratios n a < 3 * n :=
sorry

end sum_of_ratios_bounds_l163_163917


namespace maximum_volume_of_tetrahedron_l163_163388

-- Definitions based on conditions
def surface_area_of_sphere : ℝ := (289 * Real.pi) / 16
def side_length_of_equilateral_triangle : ℝ := Real.sqrt 3

-- Translate the problem statement into a Lean theorem
theorem maximum_volume_of_tetrahedron
  (P A B C : ℝ×ℝ×ℝ)
  (h_sphere_vertices : ∀ X ∈ {P, A, B, C}, ∃ R : ℝ, X.1^2 + X.2^2 + X.3^2 = R^2)
  (h_surface_area : 4 * Real.pi * (R : ℝ)^2 = surface_area_of_sphere)
  (h_base_triangle : (dist A B = side_length_of_equilateral_triangle) ∧
                    (dist B C = side_length_of_equilateral_triangle) ∧
                    (dist C A = side_length_of_equilateral_triangle)) :
  ∃ V : ℝ, V = Real.sqrt 3 :=
sorry

end maximum_volume_of_tetrahedron_l163_163388


namespace inverse_proportional_example_l163_163034

variable (x y : ℝ)

def inverse_proportional (x y : ℝ) := y = 8 / (x - 1)

theorem inverse_proportional_example
  (h1 : y = 4)
  (h2 : x = 3) :
  inverse_proportional x y :=
by
  sorry

end inverse_proportional_example_l163_163034


namespace sum_of_three_numbers_l163_163601

theorem sum_of_three_numbers (x y z : ℝ) (h₁ : x + y = 29) (h₂ : y + z = 46) (h₃ : z + x = 53) : x + y + z = 64 :=
by
  sorry

end sum_of_three_numbers_l163_163601


namespace hyperbola_equation_chord_length_l163_163054

noncomputable def length_real_axis := 2
noncomputable def eccentricity := Real.sqrt 3
noncomputable def a := 1
noncomputable def b := Real.sqrt 2
noncomputable def hyperbola_eq (x y : ℝ) := x^2 - y^2 / 2 = 1

theorem hyperbola_equation : 
  (∀ x y : ℝ, hyperbola_eq x y ↔ x^2 - (y^2 / 2) = 1) :=
by
  intros x y
  sorry

theorem chord_length (m : ℝ) : 
  ∀ x1 x2 y1 y2 : ℝ, y1 = x1 + m → y2 = x2 + m →
    x1^2 - y1^2 / 2 = 1 → x2^2 - y2^2 / 2 = 1 →
    Real.sqrt (2 * ((x1 + x2)^2 - 4 * x1 * x2)) = 4 * Real.sqrt 2 →
    m = 1 ∨ m = -1 :=
by
  intros x1 x2 y1 y2 h1 h2 h3 h4 h5
  sorry

end hyperbola_equation_chord_length_l163_163054


namespace validate_proposition_l163_163651

-- Declare the four propositions as conditions
variable {Line Plane : Type}

-- Proposition ①: Two lines parallel to the same plane are parallel to each other
def proposition1 (l1 l2 : Line) (p : Plane) : Prop :=
  (∀ {p : Plane}, l1 ∥ p → l2 ∥ p) → (l1 ∥ l2)

-- Proposition ②: Two planes parallel to the same line are parallel to each other
def proposition2 (p1 p2 : Plane) (l : Line) : Prop :=
  (∀ {l : Line}, p1 ∥ l → p2 ∥ l) → (p1 ∥ p2)

-- Proposition ③: Two lines perpendicular to the same plane are parallel to each other
def proposition3 (l1 l2 : Line) (p : Plane) : Prop :=
  (∀ {p : Plane}, l1 ⊥ p → l2 ⊥ p) → (l1 ∥ l2)

-- Proposition ④: Two planes perpendicular to the same line are parallel to each other
def proposition4 (p1 p2 : Plane) (l : Line) : Prop :=
  (∀ {l : Line}, p1 ⊥ l → p2 ⊥ l) → (p1 ∥ p2)

-- Defining the main theorem to validate the third proposition
theorem validate_proposition : proposition3
    := by {
  sorry
}

end validate_proposition_l163_163651


namespace angle_between_hands_230_pm_l163_163302

def hour_hand_position (hour minute : ℕ) : ℕ := hour % 12 * 5 + minute / 12
def minute_hand_position (minute : ℕ) : ℕ := minute
def divisions_to_angle (divisions : ℕ) : ℕ := divisions * 30

theorem angle_between_hands_230_pm :
    hour_hand_position 2 30 = 2 * 5 + 30 / 12 ∧
    minute_hand_position 30 = 30 ∧
    divisions_to_angle (minute_hand_position 30 / 5 - hour_hand_position 2 30 / 5) = 105 :=
by {
    sorry
}

end angle_between_hands_230_pm_l163_163302


namespace total_dolphins_l163_163094

theorem total_dolphins (initial_dolphins : ℕ) (triple_of_initial : ℕ) (final_dolphins : ℕ) 
    (h1 : initial_dolphins = 65) (h2 : triple_of_initial = 3 * initial_dolphins) (h3 : final_dolphins = initial_dolphins + triple_of_initial) : 
    final_dolphins = 260 :=
by
  -- Proof goes here
  sorry

end total_dolphins_l163_163094


namespace interval_of_increase_l163_163688

noncomputable def f (x : ℝ) : ℝ := Real.logb 10 (-x^2 + 4*x + 12)

theorem interval_of_increase :
  (∀ x ∈ Ioo (-2 : ℝ) 2, deriv f x > 0) ∧ (∀ x ∈ Ioc 2 6, deriv f x < 0) ∧ (∃ x ∈ Icc (-2 : ℝ) 6, deriv f x = 0) :=
sorry

end interval_of_increase_l163_163688


namespace triangle_cos_A_l163_163437

noncomputable def cos_of_angle_A (A B C : ℝ) (sinA sinB sinC : ℝ) (h: sinA / sinB = Real.sqrt 2 ∧ sinB / sinC = 1 ∧ sinC / sinA = 2) : 
  Real :=
  (sinA, sinB, sinC, h).2

theorem triangle_cos_A (A B C : ℝ) (sinA sinB sinC : ℝ) (h: sinA / sinB = Real.sqrt 2 ∧ sinB / sinC = 1 ∧ sinC / sinA = 2) : 
  cos_of_angle_A A B C sinA sinB sinC h = 3 / 4 :=
sorry

end triangle_cos_A_l163_163437


namespace number_of_outfits_l163_163252

theorem number_of_outfits 
  (shirts : ℕ) (pants : ℕ) (shoes : ℕ)
  (h_shirts : shirts = 5)
  (h_pants : pants = 3)
  (h_shoes : shoes = 2) :
  shirts * pants * shoes = 30 :=
by
  rw [h_shirts, h_pants, h_shoes]
  norm_num
  sorry

end number_of_outfits_l163_163252


namespace solve_for_s_l163_163328

def F (a b c : ℕ) : ℕ := a * b^(c + 1)

theorem solve_for_s : ∃ s : ℕ, F(s, s, 2) = 1296 ∧ s = 6 :=
by
  -- Translation of the key problem into Lean
  -- Defining F(s, s, 2) and asserting that s = 6 is indeed the solution
  use 6
  simp [F]
  norm_num
  sorry -- skipped proof

end solve_for_s_l163_163328


namespace circle_equation_l163_163385

-- Given conditions
variables (x y : ℝ)
def circle_center_on_line (a : ℝ) : Prop := a = x ∧ x = y

def tangent_to_lines (a : ℝ) : Prop :=
  (abs (a + a) / real.sqrt 2 = abs (a + a + 4) / real.sqrt 2)

-- The theorem to prove
theorem circle_equation :
  ∃ a : ℝ, circle_center_on_line a ∧ tangent_to_lines a ∧
  (∀ x y : ℝ, (x + 1)^2 + (y + 1)^2 = 2) :=
sorry

end circle_equation_l163_163385


namespace exists_nonzero_constants_to_zero_poly_l163_163709

theorem exists_nonzero_constants_to_zero_poly 
  {R : Type*} [CommRing R] {r : ℕ} (p : Fin r → R[X]) (n : Fin r → ℕ)
  (Hdeg : ∑ i, n i < (r - 1) / 2) 
  (Hdeg_p : ∀ i, nat_degree (p i) = n i) :
  ∃ (α : Fin r → R), (∀ i, α i ≠ 0) ∧ (∑ i, α i * p i = 0) :=
sorry

end exists_nonzero_constants_to_zero_poly_l163_163709


namespace sequence_third_term_is_thirteen_l163_163217

def seq (n : ℕ) : ℕ :=
  Nat.recOn n
    1
    (λ n an, 3 * an + 1)

theorem sequence_third_term_is_thirteen : seq 3 = 13 :=
  by
  sorry

end sequence_third_term_is_thirteen_l163_163217


namespace max_edges_in_graph_l163_163292

theorem max_edges_in_graph (n : ℕ) (h : n > 2) 
  (G : Type) [graph G] (A B : G) (h_distinct : A ≠ B) 
  (h_non_adj : ¬ adjacent A B) 
  (start_on_A : vertex A)
  (start_on_B : vertex B)
  (adithya_strategy : ∃ f : strategy G, winning_strategy A B f)
  : ∃ k, k = (nat.choose (n-1) 2 + 1) ∧ edges G ≤ k := 
sorry

end max_edges_in_graph_l163_163292


namespace liquid_level_ratio_is_4_1_l163_163234

 -- Define the conditions stated in the problem
def cone_radius_1 := 4 -- cm
def cone_radius_2 := 8 -- cm
def marble_radius_1 := 2 -- cm
def marble_radius_2 := 1 -- cm

 -- Given heights
noncomputable def h1 := 4 * h2

 -- Volumes for the cones initially containing the same volume of liquid
def initial_volume_cone1 (h1 : ℝ) : ℝ := (1 / 3 : ℝ) * real.pi * (cone_radius_1 ^ 2) * h1
def initial_volume_cone2 (h2 : ℝ) : ℝ := (1 / 3 : ℝ) * real.pi * (cone_radius_2 ^ 2) * h2

lemma equal_initial_volumes (h1 h2 : ℝ) (h1_eq : h1 = 4 * h2) :
  initial_volume_cone1 h1 = initial_volume_cone2 h2 :=
by
simp [initial_volume_cone1, initial_volume_cone2, h1_eq]

 -- Volumes of the marbles
def volume_marble_1 : ℝ := (4 / 3 : ℝ) * real.pi * (marble_radius_1 ^ 3)
def volume_marble_2 : ℝ := (4 / 3 : ℝ) * real.pi * (marble_radius_2 ^ 3)

 -- Scaling factors x and y
noncomputable def x (h1 : ℝ) : ℝ := real.cbrt(1 + (2 / h1))
noncomputable def y (h2 : ℝ) : ℝ := real.cbrt(1 + (1 / (16 * h2)))

 -- Changes in liquid levels
noncomputable def rise_ratio (h1 h2 : ℝ) (h1_eq : h1 = 4 * h2) : ℝ :=
  (h1 * (x h1 - 1)) / (h2 * (y h2 - 1))

theorem liquid_level_ratio_is_4_1 (h2 : ℝ) (h1 : ℝ) (h1_eq : h1 = 4 * h2) :
  rise_ratio h1 h2 h1_eq = 4 :=
by sorry

end liquid_level_ratio_is_4_1_l163_163234


namespace train_speed_in_km_per_hr_l163_163953

-- Definitions from the problem conditions
def length_of_train : ℝ := 50
def time_to_cross_pole : ℝ := 3

-- Conversion factor from the problem 
def meter_per_sec_to_km_per_hr : ℝ := 3.6

-- Lean theorem statement based on problem conditions and solution
theorem train_speed_in_km_per_hr : 
  (length_of_train / time_to_cross_pole) * meter_per_sec_to_km_per_hr = 60 := by
  sorry

end train_speed_in_km_per_hr_l163_163953


namespace correct_output_l163_163915

-- Given conditions
variable {A B : Nat}

def B_value := B = 3
def C_expr (x : Nat) : Nat := 2 * x + 1
def D_expr (x : Nat) : Nat := 4 * x

-- Proposition we want to prove: Option D is the correct output
theorem correct_output : D_expr = (λ x, 4 * x) :=
by
  -- Proof is omitted
  sorry

end correct_output_l163_163915


namespace part_I_part_II_part_III_l163_163891

noncomputable def sequence_a (n : ℕ) : ℕ := sorry
noncomputable def sequence_S (n : ℕ) : ℕ := sorry

theorem part_I {a : ℕ → ℕ} (S : ℕ → ℕ) (h1: a 1 = 1) (h2: ∀ n, S (n + 1) = 4 * a n + 2) (bn : ℕ → ℕ) :
    (bn = λ n, a (n + 1) - 2 * a n) → (bn = (λ n, 2 * (bn n - 1))) :=
begin
  sorry
end

theorem part_II {a : ℕ → ℕ} (h1: a 1 = 1) (h2: ∀ n, S (n + 1) = 4 * a n + 2) (cn : ℕ → ℕ) :
    (cn = λ n, a(n/2^n)) → (∀ n, cn n - cn (n - 1) = 3 / 4) :=
begin
  sorry
end

theorem part_III {a : ℕ → ℕ} (S : ℕ → ℕ) (h1: a 1 = 1) (h2: ∀ n, S (n + 1) = 4 * a n + 2) :
    (∀ n, a n = (3 * n - 1) / 4 * (2^n)) ∧ (∀ n, S n = 1/4 [8 + (3 n - 4) * 2^(n+1)]) :=
begin
  sorry
end

end part_I_part_II_part_III_l163_163891


namespace amelia_jet_bars_l163_163650

theorem amelia_jet_bars
    (required : ℕ) (sold_monday : ℕ) (sold_tuesday_less : ℕ) (total_sold : ℕ) (remaining : ℕ) :
    required = 90 →
    sold_monday = 45 →
    sold_tuesday_less = 16 →
    total_sold = sold_monday + (sold_monday - sold_tuesday_less) →
    remaining = required - total_sold →
    remaining = 16 :=
by
  intros
  sorry

end amelia_jet_bars_l163_163650


namespace boxes_of_apples_l163_163222

theorem boxes_of_apples (apples_per_crate crates_delivered rotten_apples apples_per_box : ℕ) 
       (h1 : apples_per_crate = 42) 
       (h2 : crates_delivered = 12) 
       (h3 : rotten_apples = 4) 
       (h4 : apples_per_box = 10) : 
       crates_delivered * apples_per_crate - rotten_apples = 500 ∧
       (crates_delivered * apples_per_crate - rotten_apples) / apples_per_box = 50 := by
  sorry

end boxes_of_apples_l163_163222


namespace volume_of_pyramid_l163_163643

theorem volume_of_pyramid (r : ℝ) (A B C D O H : ℝ) 
  (h1 : A = B) (h2 : B = C) (h3 : C = A)
  (sphere_touches_midpoints : ∀ (X : ℝ), X ∈ {A, B, C} → (d(D, X) = r ∧ d(O, X) = r))
  (H_midpoint_OD : d(O, H) = d(H, D))
  (H_on_base : H ∈ {A, B, C}) :
  volume_pyramid A B C D = (r^3 * sqrt 6) / 4 := by
  sorry

end volume_of_pyramid_l163_163643


namespace regain_original_wage_l163_163084

-- Conditions: original wage W, after a 20% cut, the new wage is 0.8W
variables (W : ℝ) 

-- Let's state the problem and the condition
theorem regain_original_wage (h : W > 0) :
  let new_wage := (4 / 5) * W in
  (new_wage * (1 + 25 / 100) = W) :=
by
  let new_wage := (4 / 5) * W
  have : new_wage > 0 := by linarith
  show (new_wage * 1.25 = W)
  sorry

end regain_original_wage_l163_163084


namespace solve_system_l163_163545

theorem solve_system:
  ∀ x y z : ℝ,
  4 * x * y * z = (x + y) * (x * y + 2) →
  4 * x * y * z = (x + z) * (x * z + 2) →
  4 * x * y * z = (y + z) * (y * z + 2) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = sqrt 2 ∧ y = sqrt 2 ∧ z = sqrt 2) ∨ 
  (x = -sqrt 2 ∧ y = -sqrt 2 ∧ z = -sqrt 2) := 
by
  intros x y z h1 h2 h3
  sorry

end solve_system_l163_163545


namespace rectangles_with_same_parity_sides_l163_163923

def colors : Type := {red, yellow, blue, green}

axiom square_colors (u : unit) : {a: colors // a ≠ red ∧ a ≠ yellow ∧ a ≠ blue ∧ a ≠ green}

noncomputable def can_form_rectangle (a b : ℕ) : Prop :=
  (∃ T, has_uniform_color_sides T ∧ are_all_sides_different_colors T ∧ side_lengths T = (a, b))

theorem rectangles_with_same_parity_sides {a b : ℕ} :
  (can_form_rectangle a b) ↔ (a % 2 = b % 2) :=
begin
  sorry
end

end rectangles_with_same_parity_sides_l163_163923


namespace weight_of_A_l163_163868

theorem weight_of_A
  (W_A W_B W_C W_D W_E : ℕ)
  (H_A H_B H_C H_D : ℕ)
  (Age_A Age_B Age_C Age_D : ℕ)
  (hw1 : (W_A + W_B + W_C) / 3 = 84)
  (hh1 : (H_A + H_B + H_C) / 3 = 170)
  (ha1 : (Age_A + Age_B + Age_C) / 3 = 30)
  (hw2 : (W_A + W_B + W_C + W_D) / 4 = 80)
  (hh2 : (H_A + H_B + H_C + H_D) / 4 = 172)
  (ha2 : (Age_A + Age_B + Age_C + Age_D) / 4 = 28)
  (hw3 : (W_B + W_C + W_D + W_E) / 4 = 79)
  (hh3 : (H_B + H_C + H_D + H_E) / 4 = 173)
  (ha3 : (Age_B + Age_C + Age_D + (Age_A - 3)) / 4 = 27)
  (hw4 : W_E = W_D + 7)
  : W_A = 79 := 
sorry

end weight_of_A_l163_163868


namespace alice_prevents_divisible_by_3_l163_163972

theorem alice_prevents_divisible_by_3 :
  ∀ (digits : Fin 2018 → ℤ), 
  (∀ i : Fin 2017, (digits i) % 3 ≠ (digits (i + 1)) % 3) →
  Alice_turn (first_and_every_other_turns), 
  Bob_turn (last_turn) →
  Alice_prevents_divisible_by_3 digits :=
sorry

end alice_prevents_divisible_by_3_l163_163972


namespace solution_correctness_l163_163343

theorem solution_correctness (x : ℝ) (h : x ≥ 2) :
  (sqrt (x + 5 - 6 * sqrt (x - 2)) + sqrt (x + 9 - 8 * sqrt (x - 2)) = 3) ↔ (x = 6 ∨ x = 27) :=
by
  sorry

end solution_correctness_l163_163343


namespace avg_bc_eq_28_l163_163550

variable (A B C : ℝ)

-- Conditions
def avg_abc_eq_30 : Prop := (A + B + C) / 3 = 30
def avg_ab_eq_25 : Prop := (A + B) / 2 = 25
def b_eq_16 : Prop := B = 16

-- The Proved Statement
theorem avg_bc_eq_28 (h1 : avg_abc_eq_30 A B C) (h2 : avg_ab_eq_25 A B) (h3 : b_eq_16 B) : (B + C) / 2 = 28 := 
by
  sorry

end avg_bc_eq_28_l163_163550


namespace sufficient_but_not_necessary_condition_for_q_l163_163383

variable (p q r : Prop)

theorem sufficient_but_not_necessary_condition_for_q (hp : p → r) (hq1 : r → q) (hq2 : ¬(q → r)) : 
  (p → q) ∧ ¬(q → p) :=
by
  sorry

end sufficient_but_not_necessary_condition_for_q_l163_163383


namespace complement_intersection_l163_163061

open Set

variable (U M N : Set ℕ)

-- Conditions
def U := {0, 1, 2, 3, 4}
def M := {1, 2, 3}
def N := {0, 3, 4}

-- Statement to be proven
theorem complement_intersection :
  (U \ M) ∩ N = {0, 4} :=
sorry

end complement_intersection_l163_163061


namespace scaled_shifted_area_l163_163670

noncomputable def area_under_curve (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, f x

theorem scaled_shifted_area (g : ℝ → ℝ) (a b : ℝ)
  (h : area_under_curve g a b = 15) :
  area_under_curve (λ x, 2 * g (x + 3) - 1) a b = 30 :=
by
  sorry

end scaled_shifted_area_l163_163670


namespace school_boys_count_l163_163574

theorem school_boys_count (B G : ℕ) (h1 : B + G = 1150) (h2 : G = (B / 1150) * 100) : B = 1058 := 
by 
  sorry

end school_boys_count_l163_163574


namespace inequality_with_conditions_l163_163815

variable {a b c : ℝ}

theorem inequality_with_conditions (h : a * b + b * c + c * a = 1) :
  (|a - b| / |1 + c^2|) + (|b - c| / |1 + a^2|) ≥ (|c - a| / |1 + b^2|) :=
by
  sorry

end inequality_with_conditions_l163_163815


namespace right_angled_triangle_count_l163_163655

variable {Solid : Type} [GeometricSolid Solid]

def num_right_angled_triangles (solid : Solid) : Nat := 3 -- Assuming we know from the problem there are 3 right-angled triangles.

theorem right_angled_triangle_count (solid : Solid) : num_right_angled_triangles solid = 3 :=
by
  sorry

end right_angled_triangle_count_l163_163655


namespace taylor_third_degree_at_3_of_sqrt_1_plus_x_l163_163605

noncomputable def sqrtTaylorThirdDegreeCenter3 (x : ℝ) : ℝ :=
  2 + (1/4) * (x - 3) - (1/64) * (x - 3)^2 + (1/512) * (x - 3)^3

theorem taylor_third_degree_at_3_of_sqrt_1_plus_x :
  ∀ x : ℝ, 
    sqrtTaylorThirdDegreeCenter3 x = 2 + (1/4) * (x - 3) - (1/64) * (x - 3)^2 + (1/512) * (x - 3)^3 :=
begin
  sorry
end

end taylor_third_degree_at_3_of_sqrt_1_plus_x_l163_163605


namespace distance_between_intersection_points_is_sqrt6_l163_163408

def polar_to_rect_line (ρ θ : ℝ) : ℝ := ρ * cos θ - ρ * sin θ + 1
def parametric_curve_x (θ : ℝ) : ℝ := sqrt 2 * cos θ
def parametric_curve_y (θ : ℝ) : ℝ := sqrt 2 * sin θ

def line_l (x y : ℝ) : Prop := x - y + 1 = 0
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 2

def intersection_points (x1 y1 x2 y2 : ℝ) : Prop :=
  line_l x1 y1 ∧ curve_C x1 y1 ∧ line_l x2 y2 ∧ curve_C x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2)

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem distance_between_intersection_points_is_sqrt6 :
  ∀ (x1 y1 x2 y2 : ℝ), intersection_points x1 y1 x2 y2 → distance x1 y1 x2 y2 = sqrt 6 :=
by
  sorry

end distance_between_intersection_points_is_sqrt6_l163_163408


namespace seventeenth_permutation_is_6951_l163_163215

def is_valid_permutation (n : ℕ) : Prop :=
  ∃ (digits : List ℕ) (h : digits.nodup), 
    digits ~ [1, 5, 6, 9] ∧ 
    digits.to_nat = n 

theorem seventeenth_permutation_is_6951 :
  sorted (filter is_valid_permutation (range ((10^4) - 10^3))).nth (17 - 1) = 6951 :=
by 
  sorry

end seventeenth_permutation_is_6951_l163_163215


namespace sum_of_solutions_is_24_l163_163434

theorem sum_of_solutions_is_24 (a : ℝ) (x1 x2 : ℝ) 
    (h1 : abs (x1 - a) = 100) (h2 : abs (x2 - a) = 100)
    (sum_eq : x1 + x2 = 24) : a = 12 :=
sorry

end sum_of_solutions_is_24_l163_163434


namespace sqrt_power_form_l163_163170

noncomputable def sqrt_diff : ℕ → ℚ
| n := (real.sqrt 2 - 1) ^ n

theorem sqrt_power_form (n : ℕ) (n_pos : 0 < n) :
  ∃ m : ℕ, sqrt_diff n = real.sqrt (m) - real.sqrt (m - 1) := by
  sorry

end sqrt_power_form_l163_163170


namespace triangle_ABC_BC_AD_l163_163798

noncomputable def find_BC_AD (A B C D : Point) (AB AC : ℝ) (cos_ABC : ℝ) : ℝ × ℝ :=
  if (AB = 5) ∧ (AC = 3) ∧ (cos_ABC = 13/14) then (7, 15/8) else (0, 0)

theorem triangle_ABC_BC_AD :
  ∀ (A B C D : Point) (AB AC : ℝ) (cos_ABC : ℝ),
    in_triangle_ABC A B C D →
    is_angle_bisector A B C D →
    AB = 5 →
    AC = 3 →
    cos_ABC = 13/14 →
    find_BC_AD A B C D AB AC cos_ABC = (7, 15/8) := 
by
  intros A B C D AB AC cos_ABC hABC hAngleBisector hAB hAC hCosABC
  have h1 : AB = 5 := hAB
  have h2 : AC = 3 := hAC
  have h3 : cos_ABC = 13/14 := hCosABC
  simp [find_BC_AD, h1, h2, h3]
  sorry

end triangle_ABC_BC_AD_l163_163798


namespace sum_abs_first_10_terms_l163_163376

noncomputable def general_term_formula (d : ℝ) (h_d : d > 0) (a_1 a_2 a_3 : ℝ) 
(h_sum : a_1 + a_2 + a_3 = -3) (h_prod : a_1 * a_2 * a_3 = 8) : ℝ → ℝ :=
  λ n, if n >= 3 then 3 * n - 7 else 0

theorem sum_abs_first_10_terms 
  (d : ℝ) (h_d : d > 0) (a_1 a_2 a_3 : ℝ) 
  (h_sum : a_1 + a_2 + a_3 = -3) (h_prod : a_1 * a_2 * a_3 = 8) :
  (Finset.range 10).sum (λ n, |general_term_formula d h_d a_1 a_2 a_3 h_sum h_prod (n + 1)|) = 105 :=
sorry

end sum_abs_first_10_terms_l163_163376


namespace max_integer_inequality_l163_163237

theorem max_integer_inequality (a b c: ℝ) (h₀: 0 < a) (h₁: 0 < b) (h₂: 0 < c) :
  (a^2 / (b / 29 + c / 31) + b^2 / (c / 29 + a / 31) + c^2 / (a / 29 + b / 31)) ≥ 14 * (a + b + c) :=
sorry

end max_integer_inequality_l163_163237


namespace find_m_l163_163415

def vector := (ℝ × ℝ)

def dot_product (u v : vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def perpendicular (u v : vector) : Prop :=
  dot_product u v = 0

theorem find_m :
  let a : vector := (4, 2)
  let b : vector := (-2, m)
  let c : vector := (a.1 + b.1, a.2 + b.2)
  perpendicular a c → m = -6 := 
by 
    intro H
    unfold perpendicular dot_product at H
    unfold c at H
    -- (4, 2) dot (2, 2 + m) = 0 ⟹ 12 + 2m = 0 ⟹ m = -6 
    have : 4 * 2 + 2 * (2 + m) = 12 + 2 * m := by reflexivity
    rw [H, this] at H
    sorry -- proof steps to show 12 + 2m = 0 implies m = -6

end find_m_l163_163415


namespace expansion_dissimilar_terms_count_l163_163117

def number_of_dissimilar_terms (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

theorem expansion_dissimilar_terms_count :
  number_of_dissimilar_terms 7 4 = 120 := by
  sorry

end expansion_dissimilar_terms_count_l163_163117


namespace problem_statement_l163_163884

def are_coplanar (P Q R : ℝ × ℝ × ℝ) : Prop :=
  ∃ (a b c d : ℝ), a * P.1 + b * P.2 + c * P.3 + d = 0 ∧
                   a * Q.1 + b * Q.2 + c * Q.3 + d = 0 ∧
                   a * R.1 + b * R.2 + c * R.3 + d = 0

noncomputable def sphere_center_to_triangle_distance 
  (S P Q R : ℝ × ℝ × ℝ) (radius : ℝ) (PQ QR RP : ℝ) : ℝ :=
  let s := (PQ + QR + RP) / 2 in
  let K := real.sqrt (s * (s - PQ) * (s - QR) * (s - RP)) in
  let circumradius := (PQ * QR * RP) / (4 * K) in
  real.sqrt (radius ^ 2 - circumradius ^ 2)

noncomputable def sphere_distance_result (x y z : ℕ) : Prop :=
  x = 20 ∧ y = 111 ∧ z = 9

theorem problem_statement 
  (S P Q R : ℝ × ℝ × ℝ) 
  (radius PQ QR RP : ℝ) 
  (h_eq1 : radius = 24) 
  (h_eq2 : PQ = 12) 
  (h_eq3 : QR = 15) 
  (h_eq4 : RP = 18)
  (h_coplanar : are_coplanar P Q R) :
  sphere_center_to_triangle_distance S P Q R radius PQ QR RP = 
  20 * real.sqrt 111 / 9 ∧ 
  sphere_distance_result (20 + 111 + 9) :=
by
  sorry

end problem_statement_l163_163884


namespace Sam_drinks_l163_163683

theorem Sam_drinks (juice_don : ℚ) (fraction_sam : ℚ) 
  (h1 : juice_don = 3 / 7) (h2 : fraction_sam = 4 / 5) : 
  (fraction_sam * juice_don = 12 / 35) :=
by
  sorry

end Sam_drinks_l163_163683


namespace acute_angle_sine_l163_163089

theorem acute_angle_sine (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : Real.sin α = 0.58) : (π / 6) < α ∧ α < (π / 4) :=
by
  sorry

end acute_angle_sine_l163_163089


namespace number_of_selections_l163_163716

theorem number_of_selections (n : ℕ) (h : n > 1) :
  ∃ count : ℕ, count = n * 2^(n - 1) + 1 :=
begin
  use n * 2^(n - 1) + 1,
  sorry
end

end number_of_selections_l163_163716


namespace train_speed_l163_163966

theorem train_speed (length_of_train time_to_cross : ℕ) (h_length : length_of_train = 50) (h_time : time_to_cross = 3) : 
  (length_of_train / time_to_cross : ℝ) * 3.6 = 60 := by
  sorry

end train_speed_l163_163966


namespace janet_spending_difference_l163_163807

-- Definitions for the conditions
def clarinet_hourly_rate : ℝ := 40
def clarinet_hours_per_week : ℝ := 3
def piano_hourly_rate : ℝ := 28
def piano_hours_per_week : ℝ := 5
def weeks_per_year : ℕ := 52

-- The theorem to be proven
theorem janet_spending_difference :
  (piano_hourly_rate * piano_hours_per_week * weeks_per_year - clarinet_hourly_rate * clarinet_hours_per_week * weeks_per_year) = 1040 :=
by
  sorry

end janet_spending_difference_l163_163807


namespace part1_part2_l163_163475

namespace ArithmeticSeq

variable (n : ℕ)

def a (n : ℕ) := -2 * n + 15
def S (n : ℕ) := n * (a 1 + a n) / 2
def T (n : ℕ) := 
  if n ≤ 7 then 
    -n^2 + 14 * n
  else 
    n^2 - 14 * n + 98

theorem part1 (a2 : a 2 = 11) (S10 : S 10 = 40) :
  ∀ n, a n = -2 * n + 15 := 
sorry

theorem part2 (a2 : a 2 = 11) (S10 : S 10 = 40) :
  ∀ n, T n = 
    if n ≤ 7 then 
      -n^2 + 14 * n
    else 
      n^2 - 14 * n + 98 := 
sorry

end ArithmeticSeq

end part1_part2_l163_163475


namespace bank_policy_advantageous_for_retirees_l163_163207

theorem bank_policy_advantageous_for_retirees
  (special_programs : Prop)
  (higher_deposit_rates : Prop)
  (lower_credit_rates : Prop)
  (reliable_loan_payers : Prop)
  (stable_income : Prop)
  (family_interest : Prop)
  (savings_tendency : Prop)
  (regular_income : Prop)
  (long_term_deposits : Prop) :
  reliable_loan_payers ∧ stable_income ∧ family_interest ∧ savings_tendency ∧ regular_income ∧ long_term_deposits → 
  special_programs ∧ higher_deposit_rates ∧ lower_credit_rates :=
sorry

end bank_policy_advantageous_for_retirees_l163_163207


namespace non_intersecting_paths_l163_163068

theorem non_intersecting_paths (n : ℕ) :
  let paths : ℕ := (nat.choose (2 * n - 2) (n - 1)) ^ 2 - (nat.choose (2 * n - 2) (n - 2)) ^ 2
  in 
  paths ≥ 0 := sorry

end non_intersecting_paths_l163_163068


namespace probability_red_then_blue_l163_163599

-- Definitions from the problem conditions
def totalMarbles := 4 + 3 + 6
def redMarbles := 4
def blueMarbles := 3
def yellowMarbles := 6
def remainingMarblesAfterRed := totalMarbles - 1
def probabilityDrawRed := redMarbles / totalMarbles.toRat
def probabilityDrawBlueAfterRed := blueMarbles / remainingMarblesAfterRed.toRat

-- The statement to prove
theorem probability_red_then_blue : (probabilityDrawRed * probabilityDrawBlueAfterRed) = (1 / 13).toRat := by
  sorry

end probability_red_then_blue_l163_163599


namespace coupon_difference_l163_163290

noncomputable def coupon_problem (P : ℝ) (hP : P > 150) : ℝ :=
let q := P - 150 in
if h1 : 30 + 0.2 * q ≥ 40 then
  if h2 : 30 + 0.2 * q ≥ 0.3 * q then
    b - a
    -- Calculate a and b based on the inequalities
    let a := 150 + 50 in
    let b := 150 + 300 in
    b - a
else sorry
else sorry

theorem coupon_difference : coupon_problem = 250 :=
sorry

end coupon_difference_l163_163290


namespace quadrilateral_midpoints_concurrent_l163_163842

structure Quadrilateral :=
(A B C D : Point)

def midpoint (A B : Point) : Point := sorry

structure ProblemData :=
(K L M N E F : Point) -- the midpoints of the sides and diagonals

theorem quadrilateral_midpoints_concurrent
  (ABCD : Quadrilateral)
  (data : ProblemData)
  (hK : data.K = midpoint ABCD.A ABCD.B)
  (hL : data.L = midpoint ABCD.B ABCD.C)
  (hM : data.M = midpoint ABCD.C ABCD.D)
  (hN : data.N = midpoint ABCD.D ABCD.A)
  (hE : data.E = midpoint ABCD.A ABCD.C)
  (hF : data.F = midpoint ABCD.B ABCD.D) :
  ∃ X : Point, 
  (midpoint data.K data.M = X) ∧ (midpoint data.L data.N = X) ∧ (midpoint data.E data.F = X) := sorry

end quadrilateral_midpoints_concurrent_l163_163842


namespace limit_of_given_sequence_l163_163863
noncomputable def a_sequence (n : ℕ) : ℕ :=
  if n = 0 then 0 else (2 * n + 2)^2

theorem limit_of_given_sequence (h : ∀ n, 0 < n → ∑ i in finset.range n, real.sqrt (a_sequence (i + 1)) = n^2 + 3 * n) :
  filter.tendsto (λ n : ℕ, (1 : ℝ) / n^2 * ∑ i in finset.range n, (a_sequence (i + 1) / (i + 2))) filter.at_top (nhds 2) :=
sorry

end limit_of_given_sequence_l163_163863


namespace garage_sale_items_l163_163656

theorem garage_sale_items (h : 34 = 13 + n + 1 + 14 - 14) : n = 22 := by
  sorry

end garage_sale_items_l163_163656


namespace escalator_steps_l163_163784

theorem escalator_steps (boy_speed girl_speed : ℕ) (boy_steps girl_steps : ℕ)
    (h1 : boy_speed = 2 * girl_speed)
    (h2 : boy_steps = 27) (h3 : girl_steps = 18) : 
    let x := (boy_steps - girl_steps * boy_speed / girl_speed) / (1 - boy_speed / girl_speed)
    in boy_steps + x = 54 :=
by
  sorry

end escalator_steps_l163_163784


namespace probability_AC_less_12_l163_163299

noncomputable def probability_AC_lt_12_cm (A B C : ℝ) (β : ℝ) : ℝ :=
  if h : β ∈ set.Ioo 0 π then
    7 / 18
  else
    0

theorem probability_AC_less_12
  (A B C : ℝ)
  (h_move_AB : dist A B = 10)
  (h_move_BC : dist B C = 7)
  (β : ℝ)
  (h_beta_range: β ∈ set.Ioo 0 π) 
  : probability_AC_lt_12_cm A B C β = 7 / 18 := 
sorry

end probability_AC_less_12_l163_163299


namespace sum_of_new_numbers_l163_163573

theorem sum_of_new_numbers (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ) :
  (n = 100) →
  (∑ i in Finset.range n, a i = 1990) →
  (∀ i : ℕ, i < n → b i = a i + (-1) ^ (i + 1) * (i + 1)) →
  (∑ i in Finset.range n, b i = 2040) :=
by
  intros h_n h_sum h_operations
  sorry

end sum_of_new_numbers_l163_163573


namespace inequality_solution_function_min_value_l163_163151

theorem inequality_solution (a : ℕ) (h₁ : abs ((3 / 2 : ℚ) - 2) < a) (h₂ : abs ((1 / 2 : ℚ) - 2) ≥ a) : a = 1 := 
by
  -- proof omitted
  sorry

theorem function_min_value (a : ℕ) (h₁ : abs ((3 / 2 : ℚ) - 2) < a) (h₂ : abs ((1 / 2 : ℚ) - 2) ≥ a)
  (h₃ : a = 1) : ∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → (abs (x + a) + abs (x - 2)) = 3 :=
by
  -- proof omitted
  use 0
  -- proof omitted
  sorry

end inequality_solution_function_min_value_l163_163151


namespace min_sets_for_prime_diff_l163_163386

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def satisfies_condition (n : ℕ) (sets : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → is_prime (|i - j|) → sets i ≠ sets j

theorem min_sets_for_prime_diff : ∃ (n : ℕ), (∀ sets : ℕ → ℕ, satisfies_condition n sets) ∧ ∀ m : ℕ, m < n → ¬ (∀ sets : ℕ → ℕ, satisfies_condition m sets) :=
sorry

end min_sets_for_prime_diff_l163_163386


namespace no_guaranteed_strategy_18_attempts_l163_163922

theorem no_guaranteed_strategy_18_attempts :
  ∀ (strategy : ℕ → ℕ), ∃ (alex_number : ℕ), alex_number ≥ 10 ∧ alex_number ≤ 99 ∧
  (∀ (attempts ≤ 18), sorry) :=
sorry

end no_guaranteed_strategy_18_attempts_l163_163922


namespace simplified_expression_value_l163_163538

theorem simplified_expression_value (x : ℝ) (h : x = -2) :
  (x - 2)^2 - 4 * x * (x - 1) + (2 * x + 1) * (2 * x - 1) = 7 := 
  by
    -- We are given x = -2
    simp [h]
    -- sorry added to skip the actual solution in Lean
    sorry

end simplified_expression_value_l163_163538


namespace largest_k_divisible_by_3_l163_163134

-- Definitions for the problem conditions.
def odd_product (n : ℕ) : ℕ := ∏ i in (Finset.range n).filter (λ x, 2*x + 1 < 2*n), (2 * i + 1)

-- Lean statement to prove the largest exponent k such that P is divisible by 3^k
theorem largest_k_divisible_by_3 :
  ∃ k, (3 ^ k ∣ odd_product 100) ∧ ∀ m, (m > k → ¬ (3 ^ m ∣ odd_product 100)) ∧ k = 49 := by
  sorry

end largest_k_divisible_by_3_l163_163134


namespace relationship_between_y_l163_163771

theorem relationship_between_y (y1 y2 y3 m : ℝ) :
  y1 = 2^2 - 4 * 2 - m ∧ y2 = 3^2 - 4 * 3 - m ∧ y3 = (-1)^2 - 4 * (-1) - m → y3 > y2 ∧ y2 > y1 :=
by
  intro h
  cases h with h1 h23
  cases h23 with h2 h3
  calc
    y3 = 5 - m : h3
    ... > -3 - m : by sorry
    ... = y2 : h2.symm
    ... > -4 - m : by sorry
    ... = y1 : h1.symm

end relationship_between_y_l163_163771


namespace find_other_root_l163_163262

theorem find_other_root (m : ℝ) (h : 1^2 + m * 1 - 5 = 0) : ∃ b : ℝ, (1 = b ∨ ∃ a : ℝ, a = -5 ∧ (1 + a = -m ∧ 1 * a = -(5 * 1))) :=
by {
    use -5,
    split,
    { simp, },
    { 
        use 1,
        split,
        { refl, },
        { 
            split,
            { ring, },
            { ring, },
        },
    },
    sorry
}

end find_other_root_l163_163262


namespace imaginary_part_of_1_minus_i_squared_l163_163486

def complex_imaginary_part (z : ℂ) : ℝ := z.im

theorem imaginary_part_of_1_minus_i_squared : complex_imaginary_part ((1 - complex.I)^2) = -2 := by
  sorry

end imaginary_part_of_1_minus_i_squared_l163_163486


namespace range_of_f_l163_163669

noncomputable def f (x k : ℝ) : ℝ := x ^ k + 3

theorem range_of_f (k : ℝ) (hk : 0 < k) :
  set.range (λ x : ℝ, f x k) ∩ set.Ici 2 = set.Ici (2 ^ k + 3) :=
by
  sorry

end range_of_f_l163_163669


namespace count_pos_integers_three_digits_l163_163071

/-- The number of positive integers less than 50,000 having at most three distinct digits equals 7862. -/
theorem count_pos_integers_three_digits : 
  ∃ n : ℕ, n < 50000 ∧ (∀ d1 d2 d3 d4 d5 : ℕ, d1 ≠ d2 ∨ d1 ≠ d3 ∨ d1 ≠ d4 ∨ d1 ≠ d5 ∨ d2 ≠ d3 ∨ d2 ≠ d4 ∨ d2 ≠ d5 ∨ d3 ≠ d4 ∨ d3 ≠ d5 ∨ d4 ≠ d5) ∧ n = 7862 :=
sorry

end count_pos_integers_three_digits_l163_163071


namespace decreasing_function_positive_l163_163725

variable {f : ℝ → ℝ}

axiom decreasing (h : ℝ → ℝ) : ∀ x1 x2, x1 < x2 → h x1 > h x2

theorem decreasing_function_positive (h_decreasing: ∀ x1 x2: ℝ, x1 < x2 → f x1 > f x2)
    (h_condition: ∀ x: ℝ, f x / (deriv^[2] f x) + x < 1) :
  ∀ x : ℝ, f x > 0 := 
by
  sorry

end decreasing_function_positive_l163_163725


namespace total_value_of_coins_l163_163703

theorem total_value_of_coins :
  let quarters := 5 * (40 * 0.25),
      dimes := 4 * (50 * 0.10),
      nickels := 3 * (40 * 0.05),
      pennies := 2 * (50 * 0.01)
    in quarters + dimes + nickels + pennies = 77 := by
  sorry

end total_value_of_coins_l163_163703


namespace sum_of_naturals_not_square_l163_163654

theorem sum_of_naturals_not_square :
  let S := (Finset.range 1976).sum (λ n => n + 1)
  S % 9 ≠ 0 ∧ S % 9 ≠ 1 ∧ S % 9 ≠ 4 ∧ S % 9 ≠ 7 :=
by
  let S := (Finset.range 1976).sum (λ n => n + 1)
  have h1 : S = 1976 * (1976 + 1) / 2 :=
    sorry -- by a standard formula for the sum of the first n natural numbers
  have h2 : S % 9 = 6 :=
    sorry -- by computing the sum modulo 9
  show 6 ≠ 0 ∧ 6 ≠ 1 ∧ 6 ≠ 4 ∧ 6 ≠ 7

end sum_of_naturals_not_square_l163_163654


namespace simplify_expression_l163_163180

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1/3 : ℝ)

theorem simplify_expression :
  (cube_root 512) * (cube_root 343) = 56 := by
  -- conditions
  let h1 : 512 = 2^9 := by rfl
  let h2 : 343 = 7^3 := by rfl
  -- goal
  sorry

end simplify_expression_l163_163180


namespace fraction_ratio_l163_163421

variable (M Q P N R : ℝ)

theorem fraction_ratio (h1 : M = 0.40 * Q)
                       (h2 : Q = 0.25 * P)
                       (h3 : N = 0.40 * R)
                       (h4 : R = 0.75 * P) :
  M / N = 1 / 3 := 
by
  -- proof steps can be provided here
  sorry

end fraction_ratio_l163_163421


namespace janet_spends_more_on_piano_l163_163804

-- Condition definitions
def clarinet_hourly_rate : ℝ := 40
def clarinet_hours_per_week : ℝ := 3
def piano_hourly_rate : ℝ := 28
def piano_hours_per_week : ℝ := 5
def weeks_per_year : ℝ := 52

-- Calculations based on conditions
def weekly_cost_clarinet : ℝ := clarinet_hourly_rate * clarinet_hours_per_week
def weekly_cost_piano : ℝ := piano_hourly_rate * piano_hours_per_week
def weekly_difference : ℝ := weekly_cost_piano - weekly_cost_clarinet
def yearly_difference : ℝ := weekly_difference * weeks_per_year

theorem janet_spends_more_on_piano : yearly_difference = 1040 := by
  sorry 

end janet_spends_more_on_piano_l163_163804


namespace doctors_assignment_l163_163336

theorem doctors_assignment :
  ∃ (assignments : Finset (Fin 3 → Finset (Fin 5))),
    (∀ h ∈ assignments, (∀ i, ∃ j ∈ h i, True) ∧
      ¬(∃ i j, (A ∈ h i ∧ B ∈ h j ∨ A ∈ h j ∧ B ∈ h i)) ∧
      ¬(∃ i j, (C ∈ h i ∧ D ∈ h j ∨ C ∈ h j ∧ D ∈ h i))) ∧
    assignments.card = 84 :=
sorry

end doctors_assignment_l163_163336


namespace edge_length_increase_correct_l163_163556

variable (L : ℝ) (L' : ℝ)
variables (s_increase : 0.331)
variables (percent_increase : ℝ)

-- Definition of the original and increased surface area
def original_surface_area := 6 * L^2
def new_surface_area := 6 * L'^2
def surface_area_increase := new_surface_area = original_surface_area * (1 + s_increase)

-- Expressing L' in terms of L and the surface area increase
def new_edge_length := L' = L * Real.sqrt (1 + s_increase)

-- Definition of the percentage increase in edge length
def edge_length_increase := percent_increase = ((Real.sqrt (1 + s_increase) - 1) * 100)

-- Hypothesis that the surface area increases by 33.1%
axiom h1 : surface_area_increase s_increase L L'

-- The statement we want to prove
theorem edge_length_increase_correct : 
  edge_length_increase s_increase 15.3 :=
by
  sorry

end edge_length_increase_correct_l163_163556


namespace probability_of_event_l163_163844

statement : Prop :=
  let interval := set.Icc (-1 : ℝ) 1
  ∃ event : set ℝ, (∀ x ∈ interval, (2 * x - 1 < 0) ↔ x < (1 / 2)) ∧
                   real.volume (interval ∩ {x | x < (1 / 2)}) / real.volume interval = 3 / 4

theorem probability_of_event : statement := by
  sorry

end probability_of_event_l163_163844


namespace triangles_in_figure_l163_163760

-- Definitions for the figure
def number_of_triangles : ℕ :=
  -- The number of triangles in a figure composed of a rectangle with three vertical lines and two horizontal lines
  50

-- The theorem we want to prove
theorem triangles_in_figure : number_of_triangles = 50 :=
by
  sorry

end triangles_in_figure_l163_163760


namespace crabapple_sequences_l163_163513

theorem crabapple_sequences (n : ℕ) : 
  n = 11 → (∏ i in (finset.range 5), (11 - i)) = 55440 :=
by
  intro h
  rw h
  sorry

end crabapple_sequences_l163_163513


namespace inequality_for_pos_a_b_c_d_l163_163701

theorem inequality_for_pos_a_b_c_d
  (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b) * (b + c) * (c + d) * (d + a) * (1 + (abcd ^ (1/4)))^4
  ≥ 16 * abcd * (1 + a) * (1 + b) * (1 + c) * (1 + d) :=
by
  sorry

end inequality_for_pos_a_b_c_d_l163_163701


namespace smallest_positive_period_l163_163559

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) - 2 * real.sqrt 3 * sin x * cos x

theorem smallest_positive_period : ∀ x : ℝ, f (x + π) = f x := by
  sorry

end smallest_positive_period_l163_163559


namespace find_angle_B_l163_163456

-- Definitions based on conditions
-- In triangle ABC, the sides opposite angles A, B, and C are a, b, and c respectively.
variable (a b c : ℝ)
variable (angle_A : ℝ)
variable (angle_B : ℝ)
variable (angle_C : ℝ)

-- Given conditions
def given_conditions := 
  angle_A = 2 * Real.pi / 3 ∧ a = 2 ∧ b = 2 * Real.sqrt 3 / 3

-- Law of Sines
def law_of_sines : Prop :=
  a / Real.sin angle_A = b / Real.sin angle_B

-- The goal is to prove that if the given conditions hold, then angle_B = π / 6
theorem find_angle_B (h₀ : given_conditions a b) (h₁ : law_of_sines a b angle_A angle_B) : 
  angle_B = Real.pi / 6 :=
sorry

end find_angle_B_l163_163456


namespace part1_zero_of_f_a_neg1_part2_range_of_a_l163_163394

noncomputable def f (a x : ℝ) := a * x^2 + 2 * x - 2 - a

theorem part1_zero_of_f_a_neg1 : 
  f (-1) 1 = 0 :=
by 
  sorry

theorem part2_range_of_a (a : ℝ) :
  a ≤ 0 →
  (∃ x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a x = 0) ∧ (∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x = 0 → x = 1) ↔ 
  (-1 ≤ a ∧ a ≤ 0) ∨ (a ≤ -2) :=
by 
  sorry

end part1_zero_of_f_a_neg1_part2_range_of_a_l163_163394


namespace sum_of_numbers_l163_163875

theorem sum_of_numbers (a b : ℝ) 
  (h1 : a^2 - b^2 = 6) 
  (h2 : (a - 2)^2 - (b - 2)^2 = 18): 
  a + b = -2 := 
by 
  sorry

end sum_of_numbers_l163_163875


namespace exists_least_n_l163_163138

noncomputable def b : ℕ → ℕ
| 7       := 7
| (n + 1) := ite (n + 1 > 7) (50 * b n + 2 * (n + 1)) (b (n + 1))

theorem exists_least_n :
  ∃ (n : ℕ), n > 7 ∧ (b n) % 55 = 0 ∧ ∀ m, m > 7 ∧ (b m) % 55 = 0 → m ≥ n :=
begin
  have H1 : b 7 = 7, from rfl,
  have H2 : ∀ n > 7, b n = 50 * b (n - 1) + 2 * n, from λ n hn, sorry,
  use 54,
  split,
  { exact nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.succ_le_succ (nat.le_refl 7)))))) },
  split,
  { show (b 54) % 55 = 0, from sorry },
  { assume m hm1 hm2, show m ≥ 54, from sorry }
end

end exists_least_n_l163_163138


namespace inequality_solution_l163_163150

noncomputable def solution_set (f : ℝ → ℝ) (f' : ℝ → ℝ) : Set ℝ :=
{ x | (x + 2014) ^ 2 * f (x + 2014) - 4 * f (-2) > 0 }

theorem inequality_solution (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h_diff : ∀ x : ℝ, x < 0 → has_deriv_at f (f' x) x)
  (h_ineq : ∀ x : ℝ, x < 0 → 2 * f x + x * f' x > x ^ 2) :
  solution_set f f' = {x | x < -2016} :=
by
  sorry

end inequality_solution_l163_163150


namespace final_shirt_price_l163_163998

theorem final_shirt_price :
  let cost_price := 20
  let profit_rate := 0.30
  let discount_rate := 0.50
  let profit := cost_price * profit_rate
  let regular_selling_price := cost_price + profit
  let final_price := regular_selling_price * discount_rate
  in final_price = 13 :=
by
  sorry

end final_shirt_price_l163_163998


namespace probability_transform_in_U_l163_163570

open Complex

def is_in_U (z : ℂ) : Prop :=
  let (x, y) := (z.re, z.im)
  -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2

theorem probability_transform_in_U (w : ℂ) (hw : is_in_U w) :
  ∃ c : ℝ, c = 1 → is_in_U ((1/2 : ℝ) + (1/2 : ℝ) * I * w) :=
sorry

end probability_transform_in_U_l163_163570


namespace transport_problem_part1_transport_problem_part2_l163_163619

noncomputable def teamA_transport (x : ℝ) : ℝ := (5 / 3) * x

theorem transport_problem_part1 : 
  ∀ (x : ℝ), 
  (4000 / teamA_transport x + 2 = 3000 / x) → 
  teamA_transport x = 500 :=
begin
  sorry
end

noncomputable def updated_transportA (orig_transportA m : ℝ) : ℝ := orig_transportA + m
noncomputable def updated_transportB (orig_transportB m : ℝ) : ℝ := orig_transportB * (1 + m / 300)

theorem transport_problem_part2 :
  ∀ (x m : ℝ),
  (m = 50) → 
  (updated_transportB x m * 9 * 50 = 157500) :=
begin
  sorry
end

end transport_problem_part1_transport_problem_part2_l163_163619


namespace remainder_of_8_pow_2050_mod_100_l163_163241

theorem remainder_of_8_pow_2050_mod_100 :
  (8 ^ 2050) % 100 = 24 := 
by
  have h1: 8 % 100 = 8 := rfl
  have h2: (8 ^ 2) % 100 = 64 := by norm_num
  have h3: (8 ^ 3) % 100 = 12 := by norm_num
  have h4: (8 ^ 4) % 100 = 96 := by norm_num
  have h5: (8 ^ 5) % 100 = 68 := by norm_num
  have h6: (8 ^ 6) % 100 = 44 := by norm_num
  have h7: (8 ^ 7) % 100 = 52 := by norm_num
  have h8: (8 ^ 8) % 100 = 16 := by norm_num
  have h9: (8 ^ 9) % 100 = 28 := by norm_num
  have h10: (8 ^ 10) % 100 = 24 := by norm_num
  have h20: (8 ^ 20) % 100 = 76 := by norm_num
  have h40: (8 ^ 40) % 100 = 76 := by norm_num
  
  -- Given periodicity and expressing 2050 in terms of periodicity
  have h_periodicity : (8 ^ 2050) % 100 = (8 ^ (20 * 102 + 10)) % 100 := by
    rw [pow_add, pow_mul]
    rw [(8 ^ 2010 % 100), (8 ^ 10) % 100]
    sorry

end remainder_of_8_pow_2050_mod_100_l163_163241


namespace calculate_B_calculate_two_A_minus_B_two_A_minus_B_independent_of_c_calculate_two_A_minus_B_specific_l163_163707

noncomputable def A (a b c : ℝ) := 3 * a^2 * b - 2 * a * b^2 + abc
noncomputable def C (a b c : ℝ) := 4 * a^2 * b - 3 * a * b^2 + 4 * abc
noncomputable def B (a b c : ℝ) := C a b c - 2 * A a b c
noncomputable def two_A_minus_B (a b c : ℝ) := 2 * A a b c - B a b c

theorem calculate_B (a b c : ℝ) : 
  B a b c = -2 * a^2 * b + a * b^2 + 2 * abc := sorry

theorem calculate_two_A_minus_B (a b c : ℝ) : 
  two_A_minus_B a b c = 8 * a^2 * b - 5 * a * b^2 := sorry

theorem two_A_minus_B_independent_of_c (a b c : ℝ) : 
  (two_A_minus_B a b c = two_A_minus_B a b 0) := by
  sorry

theorem calculate_two_A_minus_B_specific (c : ℝ) :
  two_A_minus_B (1 / 8) (1 / 5) c = 0 := sorry

end calculate_B_calculate_two_A_minus_B_two_A_minus_B_independent_of_c_calculate_two_A_minus_B_specific_l163_163707


namespace quadratic_inverse_roots_sum_l163_163372

theorem quadratic_inverse_roots_sum :
  (∀ (x : ℝ), x^2 - 2 * x - 5 = 0 → (1 / x + 1 / (x / 2 - 5)) = -2 / 5) :=
by
  intros x hx
  have h := hx,
  sorry

end quadratic_inverse_roots_sum_l163_163372


namespace triangle_position_after_five_rolls_l163_163219

theorem triangle_position_after_five_rolls :
  ∀ (square : Type) (start_pos : square) (triangle_pos : square → square),
  let octagon_angle := ((8 - 2) * 180) / 8
  let square_angle := 90
  let total_rotation := 4 * (360 - (octagon_angle + square_angle))
  let effective_rotation := total_rotation % 360
  (triangle_pos start_pos) = start_pos →
  effective_rotation = 180 →
  triangle_pos (rotate 180 start_pos) = top :=
by
  sorry

end triangle_position_after_five_rolls_l163_163219


namespace f_16_explicit_l163_163364

/-- Define the function f(x) -/
def f (x : ℝ) : ℝ := (1 + x) / (2 - x)

/-- Recursive definition of the fn series -/
def f_seq : ℕ → (ℝ → ℝ)
| 0     := f
| (n+1) := f ∘ f_seq n

/- Prove the explicit formula for f16(x) -/
theorem f_16_explicit (x : ℝ) : f_seq 15 x = (x - 1) / x :=
sorry

end f_16_explicit_l163_163364


namespace proof_statements_l163_163009

variable {a : ℝ}
def line1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, a * x + 2 * y + 3 * a = 0
def line2 (a : ℝ) : ℝ → ℝ → Prop := λ x y, 3 * x + (a-1) * y + (3 - a) = 0
def is_parallel (a1 b1 a2 b2 : ℝ) : Prop := a1 * b2 = a2 * b1
def is_perpendicular (a1 b1 a2 b2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0
def passes_through (l : ℝ → ℝ → Prop) (x y : ℝ) : Prop := l x y
def distance (x1 y1 x2 y2 : ℝ) : ℝ := sqrt ((x1 - x2) ^ 2 + (y1 - y2) ^ 2)

theorem proof_statements :
  (is_parallel a 2 3 (a - 1) ↔ a = 3 ∨ a = -2) ∧
  (a = 2/5 → is_perpendicular a 2 3 (a-1)) ∧
  ¬ passes_through (line1 a) 3 0 ∧
  distance 1 3 (line1 a) = 5 := 
by sorry

end proof_statements_l163_163009


namespace correct_operation_l163_163248

theorem correct_operation (a b : ℤ) : -3 * (a - b) = -3 * a + 3 * b := 
sorry

end correct_operation_l163_163248


namespace lap_time_improvement_l163_163337

theorem lap_time_improvement:
  let initial_lap_time := (30 : ℚ) / 15
  let current_lap_time := (33 : ℚ) / 18
  initial_lap_time - current_lap_time = 1 / 6 :=
by
  let initial_lap_time := (30 : ℚ) / 15
  let current_lap_time := (33 : ℚ) / 18
  calc
    initial_lap_time - current_lap_time
        = 2 - 11 / 6 : by rw [initial_lap_time, current_lap_time]
    ... = 1 / 6 : sorry

end lap_time_improvement_l163_163337


namespace points_on_quadratic_function_l163_163768

theorem points_on_quadratic_function (m y1 y2 y3 : ℝ) :
  y1 = 4 - 8 - m →
  y2 = 9 - 12 - m →
  y3 = 1 + 4 - m →
  y3 > y2 ∧ y2 > y1 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  linarith

end points_on_quadratic_function_l163_163768


namespace KLMN_is_parallelogram_l163_163704

-- Define the hypotheses and state the theorem to prove

-- The context of the problem involves points, circles, and intersection points.
-- We define the points and circles, and state the required properties.

noncomputable theory

variables (P K L M N : Point)
variables (k1 k2 k3 k4 : Circle)
variable (r : ℝ)

-- Each circle has the same radius
axiom CirclesEqualRadius : (∀ c ∈ {k1, k2, k3, k4}, radius c = r)

-- Each circle passes through a common point P
axiom CirclesPassThroughP : (∀ c ∈ {k1, k2, k3, k4}, c ∋ P)

-- Define the intersection points other than P
axiom IntersectionPoints : (k1 ≠ k2 ∧ k2 ≠ k3 ∧ k3 ≠ k4 ∧ k4 ≠ k1) →
                           ∃! (K, L, M, N : Point), 
                              K ≠ P ∧ L ≠ P ∧ M ≠ P ∧ N ≠ P ∧
                              K ∈ k1 ∧ K ∈ k2 ∧
                              L ∈ k2 ∧ L ∈ k3 ∧
                              M ∈ k3 ∧ M ∈ k4 ∧
                              N ∈ k4 ∧ N ∈ k1

-- State the problem as a theorem to prove that KLMN is a parallelogram
theorem KLMN_is_parallelogram :
  (k1 ∋ P) ∧
  (k2 ∋ P) ∧
  (k3 ∋ P) ∧
  (k4 ∋ P) ∧
  CirclesEqualRadius k1 k2 k3 k4 r ∧
  IntersectionPoints k1 k2 k3 k4 K L M N →
  IsParallelogram K L M N := 
sorry

end KLMN_is_parallelogram_l163_163704


namespace rearrangements_count_l163_163319

theorem rearrangements_count (n : ℕ) (h : n = 2018) :
  ∃ count, (∀ a : vector ℕ n, (∀ k, 1 ≤ k ∧ k ≤ n → a.nth k > k → a k > 0)) → count = 2^2018 - 2019 :=
by sorry

end rearrangements_count_l163_163319


namespace avg_and_exp_val_l163_163447

noncomputable def x : Fin 10 → ℝ
| ⟨0, _⟩ => 38
| ⟨1, _⟩ => 41
| ⟨2, _⟩ => 44
| ⟨3, _⟩ => 51
| ⟨4, _⟩ => 54
| ⟨5, _⟩ => 56
| ⟨6, _⟩ => 58
| ⟨7, _⟩ => 64
| ⟨8, _⟩ => 74
| ⟨9, _⟩ => 80
| _ => 0 -- This case should never happen

def average (x : Fin 10 → ℝ) : ℝ :=
  (Finset.univ.sum x) / 10

def variance (x : Fin 10 → ℝ) (mean : ℝ) : ℝ :=
  (Finset.univ.sum (λ i => (x i - mean) ^ 2)) / 10

theorem avg_and_exp_val:
  let mean := average x in
  mean = 56 ∧
  let var := variance x mean in
  var = 169 ∧ 
  let mu := mean in
  let sigma := Real.sqrt var in
  let p := 0.9545 in
  ∃ Y : ℕ → ℝ, 
    (binomial 100 p).expected_value Y = 95.45 :=
by 
  sorry

end avg_and_exp_val_l163_163447


namespace average_speed_trip_l163_163608

theorem average_speed_trip (d1 d2 : ℝ) (s1 s2 : ℝ)
  (h1 : d1 = 8) (h2 : d2 = 11) (h3 : s1 = 11) (h4 : s2 = 8) :
  let total_distance := d1 + d2 in
  let t1 := d1 / s1 in
  let t2 := d2 / s2 in
  let total_time := t1 + t2 in
  let average_speed := total_distance / total_time in
  average_speed ≈ 8.99 :=
by {
  sorry
}

end average_speed_trip_l163_163608


namespace modulus_of_z_l163_163662

-- Define the complex number z and the imaginary unit i
def z : ℂ := 1 + complex.I

-- State the theorem for the modulus of z
theorem modulus_of_z : complex.abs z = Real.sqrt 2 := by
  -- Proof will be provided here
  sorry

end modulus_of_z_l163_163662


namespace two_digit_sum_condition_l163_163345

theorem two_digit_sum_condition (x y : ℕ) (hx : 1 ≤ x) (hx9 : x ≤ 9) (hy : 0 ≤ y) (hy9 : y ≤ 9)
    (h : (x + 1) + (y + 2) - 10 = 2 * (x + y)) :
    (x = 6 ∧ y = 8) ∨ (x = 5 ∧ y = 9) :=
sorry

end two_digit_sum_condition_l163_163345


namespace domain_f_l163_163680

noncomputable def f (x : ℝ) : ℝ := 
  (3 * x^2) / (real.sqrt (1 - x)) + real.log (3 * x + 1)

theorem domain_f : set_of (λ x : ℝ, (1 - x ≥ 0) ∧ (3 * x + 1 > 0)) = Ioo (-1/3 : ℝ) 1 := by
  sorry

end domain_f_l163_163680


namespace strawberries_harvest_l163_163155

theorem strawberries_harvest 
  (length width : ℕ) (plants_per_sqft strawberries_per_plant : ℕ)
  (h_length : length = 10) (h_width : width = 7) 
  (h_plants_per_sqft : plants_per_sqft = 5) 
  (h_strawberries_per_plant : strawberries_per_plant = 12) : 
  length * width * plants_per_sqft * strawberries_per_plant = 4200 :=
by
  rw [h_length, h_width, h_plants_per_sqft, h_strawberries_per_plant]
  norm_num

end strawberries_harvest_l163_163155


namespace find_a1_b1_l163_163641

noncomputable def complex_sequence (n : ℕ) : ℂ → ℂ
| (a, b) := (sqrt 3 * a - 2 * b, 2 * a + sqrt 3 * b)

def recurrence_relation (n : ℕ) : ℂ → Prop :=
∀ n, complex_sequence n = (complex_sequence (n-1))

theorem find_a1_b1 (a : ℤ) (b : ℤ) :
  (∃ a₁ b₁,
    complex_sequence 49 (1, 3) = (1, 3) ∧
    complex_sequence 0 = (a₁, b₁)) ↔
  a₁ + b₁ = (-(1/2) - 3*sqrt(3)/2) / (sqrt(13)^49) := sorry

end find_a1_b1_l163_163641


namespace sequence_not_infinite_l163_163715

def sequence_rule (a : ℕ) : ℕ :=
  if a % 10 <= 5 then a / 10
  else 9 * a

theorem sequence_not_infinite (a₀ : ℕ) :
  ¬ ∃ (a : ℕ → ℕ), a 0 = a₀ ∧ (∀ n : ℕ, a (n + 1) = sequence_rule (a n)) ∧ ∀ n m : ℕ, n < m → a n > a m :=
begin
  sorry
end

end sequence_not_infinite_l163_163715


namespace solve_triangle_problem_l163_163458
noncomputable def triangle_problem : Prop :=
  ∃ (A B C : ℝ) (a b c : ℝ) (cosC sinA sinC : ℝ),
    c = √2 ∧
    cosC = 3 / 4 ∧
    2 * c * sinA = b * sinC ∧
    b = 2 ∧
    sinA = √14 / 8 ∧
    sin (2 * A + π / 3) = (5 * √7 + 9 * √3) / 32

theorem solve_triangle_problem : triangle_problem :=
  sorry

end solve_triangle_problem_l163_163458


namespace arithmetic_sequence_general_term_sum_of_first_n_terms_l163_163039

noncomputable def a_n (n : ℕ) : ℕ :=
2 * n - 1

def b_n (n : ℕ) : ℤ :=
(-1)^(n-1) * 4 * n / (a_n n * a_n (n + 1))

def T_n (n : ℕ) : ℚ :=
if even n then
  2 * n / (2 * n + 1)
else
  (2 * n + 2) / (2 * n + 1)

theorem arithmetic_sequence_general_term (a_n : ℕ → ℕ) (d : ℕ) (h_d : d = 2)
  (a1 : ℕ) (S : ℕ → ℕ) (h_S1 : S 1 = a1) (h_S2 : S 2 = 4 + 2 * a1)
  (h_S4 : S 4 = 10 + 4 * a1) (h_geometric : (S 2) ^ 2 = S 1 * S 4) :
  a_n = 2 * (λ (n : ℕ), n) - 1 :=
sorry

theorem sum_of_first_n_terms (T_n : ℕ → ℚ) (b : ℕ → ℤ) :
  (∀ n, b n = (-1)^(n-1) * 4 * n / (a_n n * a_n (n + 1))) →
  (T_n = λ n,
  if even n then
    2 * n / (2 * n + 1)
  else
    (2 * n + 2) / (2 * n + 1)) :=
sorry

end arithmetic_sequence_general_term_sum_of_first_n_terms_l163_163039


namespace sum_positive_real_solutions_l163_163000

theorem sum_positive_real_solutions :
  ∃ (s : ℝ), (∀ x : ℝ, (0 < x) → 2 * cos(2 * x) * (cos(2 * x) - cos(1007 * π^2 / x)) = cos(4 * x) - 1) ∧ s = 1080 * π :=
by sorry

end sum_positive_real_solutions_l163_163000


namespace find_a_b_range_f_l163_163739

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x ^ 3 - a * x + b

theorem find_a_b (a b : ℝ) : (f 2 a b = -8) ∧ (derivative (λ (x : ℝ), f x a b) 2 = 0) → a = 12 ∧ b = 8 :=
by {
  sorry
}

noncomputable def f_specific (x : ℝ) : ℝ := x ^ 3 - 12 * x + 8

theorem range_f : (Set.image f_specific (Set.Icc (-3) (3))) = Set.Icc (-8) (24) :=
by {
  sorry
}

end find_a_b_range_f_l163_163739


namespace value_at_x_eq_1_intersection_point_on_graph_l163_163830

-- Definitions for linear functions and generating function
def linear_func₁ (a₁ b₁ x : ℝ) : ℝ := a₁ * x + b₁
def linear_func₂ (a₂ b₂ x : ℝ) : ℝ := a₂ * x + b₂

-- Generating function given the condition m+n=1
def generating_func (a₁ b₁ a₂ b₂ m n x : ℝ) [h : m + n = 1] : ℝ := m * linear_func₁ a₁ b₁ x + n * linear_func₂ a₂ b₂ x

-- 1. Prove the value of the generating function at x = 1
theorem value_at_x_eq_1 (m n : ℝ) [h : m + n = 1] : 
  generating_func 1 1 2 0 m n 1 = 2 :=
sorry

-- 2. Determine if the intersection point P lies on the graph of the generating function
theorem intersection_point_on_graph (a₁ b₁ a₂ b₂ m n a b : ℝ) [h₁ : a₁ * a + b₁ = b] [h₂ : a₂ * a + b₂ = b] [h : m + n = 1] :
  generating_func a₁ b₁ a₂ b₂ m n a = b :=
sorry

end value_at_x_eq_1_intersection_point_on_graph_l163_163830


namespace proof_inequality_for_line_segments_l163_163590

variables {R : Type*} [linear_ordered_field R] {ABC : R → R → R → triangle R}

theorem proof_inequality_for_line_segments
  (f : ℝ → ℝ)
  (h_property : ∀ (D E M : ℝ), M = (D + E) / 2 → f (d ABC D) + f (d ABC E) ≤ 2 * f (d ABC M)) :
  ∀ (P Q : ℝ) (N : ℝ), N ∈ interior (segment P Q) →
    |QN| * f (d ABC P) + |PN| * f (d ABC Q) ≤ |PQ| * f (d ABC N) :=
sorry

end proof_inequality_for_line_segments_l163_163590


namespace distance_AD_is_40_l163_163521

variable {Point : Type}

-- Given conditions
variable A B C D : Point
variable (distance : Point → Point → ℝ)
variable (angle : Point → Point → Point → ℝ)
variable (distAC : distance A C = 15 * Real.sqrt 2)
variable (angleBAC : angle B A C = 30)
variable (distDC : distance D C = 30)
variable (east : ∀ {P Q R : Point}, east_of P Q → north_of Q R → angle P Q R = 90)
variable (east_of : Point → Point → Prop)
variable (north_of : Point → Point → Prop)
variable (is_east : east_of A B)
variable (is_north1 : north_of B C)
variable (is_north2 : north_of C D)

-- Proof problem
theorem distance_AD_is_40 : distance A D = 40 := 
sorry

end distance_AD_is_40_l163_163521


namespace mike_total_points_l163_163512

theorem mike_total_points (games: ℕ) (points_per_game: ℕ) (total_points: ℕ) 
  (h1: games = 6) 
  (h2: points_per_game = 4) 
  (h3: total_points = games * points_per_game) : 
  total_points = 24 :=
by
  rw [h1, h2]
  rw h3
  rfl

end mike_total_points_l163_163512


namespace domain_of_g_is_0_to_1_exclusive_l163_163047

theorem domain_of_g_is_0_to_1_exclusive (f : ℝ → ℝ) (h_dom : ∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f y = x) :
  ∀ x, 0 ≤ x ∧ x < 1 ↔ ∃ y, f(2 * y) ≠ (x - 1) :=
begin
  sorry
end

end domain_of_g_is_0_to_1_exclusive_l163_163047


namespace shirt_price_after_discount_l163_163996

/-- Given a shirt with an initial cost price of $20 and a profit margin of 30%, 
    and a sale discount of 50%, prove that the final sale price of the shirt is $13. -/
theorem shirt_price_after_discount
  (cost_price : ℝ)
  (profit_margin : ℝ)
  (discount : ℝ)
  (selling_price : ℝ)
  (final_price : ℝ)
  (h_cost : cost_price = 20)
  (h_profit_margin : profit_margin = 0.30)
  (h_discount : discount = 0.50)
  (h_selling_price : selling_price = cost_price + profit_margin * cost_price)
  (h_final_price : final_price = selling_price - discount * selling_price) :
  final_price = 13 := 
  sorry

end shirt_price_after_discount_l163_163996


namespace hallie_total_profit_l163_163065

-- Definitions of the conditions
def art_contest_prize : ℕ := 150
def sale_price_per_painting : ℕ := 50
def number_of_paintings_sold : ℕ := 3
def cost_of_art_supplies : ℕ := 40
def exhibition_fee : ℕ := 20

-- The main theorem to prove
theorem hallie_total_profit : 
  let total_earnings := art_contest_prize + number_of_paintings_sold * sale_price_per_painting in
  let total_expenses := cost_of_art_supplies + exhibition_fee in
  total_earnings - total_expenses = 240 :=
by
  sorry

end hallie_total_profit_l163_163065


namespace prisoner_hat_color_l163_163586

-- Define the possible colors of the hats
inductive Color
| red
| green

open Color

-- We introduce the conditions as definitions
def total_red_hats := 2
def total_green_hats := 3

structure Prisoner :=
  (can_see : List Color) -- The list of hats the prisoner can see

noncomputable def hats_choice (P1 : Color) (P2 : Color) (P3 : Color) (remaining : List Color) : Prop :=
  (P1 :: P2 :: P3 :: remaining).length = 5 ∧
  (P1 :: P2 :: P3 :: remaining).count red = total_red_hats ∧
  (P1 :: P2 :: P3 :: remaining).count green = total_green_hats

-- Defining the condition of silence
def is_silent (P : Prisoner) : Prop :=
  ∀ color, ¬(P.can_see.count red = 2) → (P.can_see.count color < 3 - 1)

-- The Lean 4 statement that represents the proof problem
theorem prisoner_hat_color :
  ∀ (P1 P2 P3 : Color) (remaining : List Color),
  hats_choice P1 P2 P3 remaining →
  is_silent { can_see := [P1, P2] } →
  is_silent { can_see := [P1, P2, P3] } →
  P1 = green :=
by sorry

end prisoner_hat_color_l163_163586


namespace altitudes_of_triangle_l163_163517

-- Define the points and conditions in Lean
variables (A B C A1 B1 C1 : Type*) [EuclideanGeometry A B C A1 B1 C1]

-- Assumptions
variable (ABC : Triangle A B C)
variable (hA1 : A1 ∈ segment B C)
variable (hB1 : B1 ∈ segment A C)
variable (hC1 : C1 ∈ segment A B)
variable (hA1A : is_angle_bisector A1 A (angle B C))
variable (hB1B : is_angle_bisector B1 B (angle A C))
variable (hC1C : is_angle_bisector C1 C (angle A B))

-- The proof goal
theorem altitudes_of_triangle :
  is_altitude (A A1) ∧ is_altitude (B B1) ∧ is_altitude (C C1) :=
by
  -- The proof will be filled here
  sorry

end altitudes_of_triangle_l163_163517


namespace intersection_in_first_quadrant_l163_163062

theorem intersection_in_first_quadrant (a : ℝ) :
  (∃ x y : ℝ, ax - y + 2 = 0 ∧ x + y - a = 0 ∧ x > 0 ∧ y > 0) ↔ a > 2 := 
by
  sorry

end intersection_in_first_quadrant_l163_163062


namespace largest_quotient_in_set_l163_163597

def largestQuotientSet : Set Int := {-30, -6, 0, 3, 5, 15}

theorem largest_quotient_in_set : ∃ (a b : Int), a ∈ largestQuotientSet ∧ b ∈ largestQuotientSet ∧ b ≠ 0 ∧ (a / b : Int) = 5 := 
begin
  sorry
end

end largest_quotient_in_set_l163_163597


namespace algebra_inequality_l163_163366

theorem algebra_inequality
  (x y z : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z)
  (h_cond : x * y + y * z + z * x ≤ 1) :
  (x + 1 / x) * (y + 1 / y) * (z + 1 / z) ≥ 8 * (x + y) * (y + z) * (z + x) :=
by
  sorry

end algebra_inequality_l163_163366


namespace oregano_basil_ratio_l163_163981

-- Define the number of oregano plants and total number of plants in Betty's herb garden
variable (O : Nat)
variable (TotalPlants BasilPlants : Nat)

-- Conditions
axiom total_plants : TotalPlants = 17
axiom basil_plants : BasilPlants = 5
axiom oregano_plus_basil : O + BasilPlants = TotalPlants

-- Theorem to prove the ratio
theorem oregano_basil_ratio (h1 : TotalPlants = 17) (h2 : BasilPlants = 5) (h3 : O + BasilPlants = TotalPlants) :
  O / BasilPlants = 12 / 5 :=
by
  rw [h1, h2] at h3
  -- From the given condition h3: O + 5 = 17, we derive O = 12
  have hO : O = 12 :=
    by linarith
  rw hO
  norm_num

  -- skipping further detailed proof steps with 'sorry'
  sorry

end oregano_basil_ratio_l163_163981


namespace n_square_of_odd_integer_l163_163826

theorem n_square_of_odd_integer (n : ℕ) (h : ∑ d in divisors n, d = 2 * n + 1) : ∃ (k : ℕ), k.odd ∧ n = k^2 := 
sorry

end n_square_of_odd_integer_l163_163826


namespace abc_is_cube_of_integer_l163_163818

theorem abc_is_cube_of_integer (a b c : ℤ) (h : (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a = 3) : ∃ k : ℤ, abc = k^3 := 
by
  sorry

end abc_is_cube_of_integer_l163_163818


namespace dig_pit_in_days_l163_163442

def work_capacity_younger_old (work_middle : ℝ) : ℝ := 0.75 * work_middle
def work_capacity_boy (work_middle : ℝ) : ℝ := (2/3) * work_middle
def total_work_units (n_middle : ℕ) (days : ℕ) (work_middle : ℝ) : ℝ := n_middle * work_middle * days
def total_team_capacity (young men middle_aged men old men boys young middle_aged_boys old_boys : ℕ)
  (work_middle : ℝ) : ℝ :=
  (young men + old men) * work_capacity_younger_old work_middle
  + middle_aged_men * work_middle
  + (young boys + old boys) * work_capacity_boy work_middle
  + middle_aged_boys * work_capacity_boy work_middle

theorem dig_pit_in_days
  (n_middle : ℕ := 8)
  (days_to_dig : ℕ := 20)
  (work_middle : ℝ := 1)
  (young men : ℕ := 2)
  (middle_aged_me: ℕ := 2)
  (old_me: ℕ := 2)
  (young boys : ℕ := 5)
  (middle_aged boys: ℕ := 5)
  (old boys : ℕ := 5) :
  let total_work := total_work_units n_middle days_to_dig work_middle,
      team_capacity := total_team_capacity young men_middle_aged men old_me young boys_middle_aged_boys old_bows work_middle in
  (total_work / team_capacity) ≈ 12 := sorry

end dig_pit_in_days_l163_163442


namespace hilton_final_marbles_l163_163757

theorem hilton_final_marbles :
  let initial_marbles := 26
  let found_marbles := 6
  let lost_marbles := 10
  let gift_multiplication_factor := 2
  let marbles_after_find_and_lose := initial_marbles + found_marbles - lost_marbles
  let gift_marbles := gift_multiplication_factor * lost_marbles
  let final_marbles := marbles_after_find_and_lose + gift_marbles
  final_marbles = 42 :=
by
  -- Proof to be filled
  sorry

end hilton_final_marbles_l163_163757


namespace erdos_ginzburg_ziv_2047_l163_163529

open Finset

theorem erdos_ginzburg_ziv_2047 (s : Finset ℕ) (h : s.card = 2047) : 
  ∃ t ⊆ s, t.card = 1024 ∧ (t.sum id) % 1024 = 0 :=
sorry

end erdos_ginzburg_ziv_2047_l163_163529


namespace sine_cosine_fraction_l163_163424

theorem sine_cosine_fraction (θ : ℝ) (h : Real.tan θ = 2) : 
    (Real.sin θ * Real.cos θ) / (1 + Real.sin θ ^ 2) = 2 / 9 := 
by 
  sorry

end sine_cosine_fraction_l163_163424


namespace circle_equation_of_tangent_parabola_line_l163_163772

theorem circle_equation_of_tangent_parabola_line :
  let center := (0, -1)
  let radius : ℝ := abs (4 * center.2 - 1) / (sqrt ((3 : ℝ)^2 + (4 : ℝ)^2))
  (∀ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ x^2 + (y + 1)^2 = 1) := by
  sorry

end circle_equation_of_tangent_parabola_line_l163_163772


namespace radius_of_circle_is_61_2_inches_l163_163930

noncomputable def large_square_side : ℝ := 144
noncomputable def total_area : ℝ := large_square_side * large_square_side
noncomputable def l_region_fraction : ℝ := 5 / 18
noncomputable def l_region_area : ℝ := 4 * l_region_fraction * total_area
noncomputable def center_square_area : ℝ := total_area - l_region_area
noncomputable def center_square_side : ℝ := Real.sqrt center_square_area
noncomputable def circle_radius : ℝ := center_square_side / 2

theorem radius_of_circle_is_61_2_inches : circle_radius = 61.2 := 
sorry

end radius_of_circle_is_61_2_inches_l163_163930


namespace angle_bisector_theorem_l163_163256

theorem angle_bisector_theorem (A B C D : Type) [n : triangle A B C]
    (BD_internal : bisector_angle A B C D) :
    AD / DC = AB / BC := by
    sorry

end angle_bisector_theorem_l163_163256


namespace isosceles_triangle_aacute_l163_163719

theorem isosceles_triangle_aacute (a b c : ℝ) (h1 : a = b) (h2 : a + b + c = 180) (h3 : c = 108)
  : ∃ x y z : ℝ, x + y + z = 180 ∧ x < 90 ∧ y < 90 ∧ z < 90 ∧ x > 0 ∧ y > 0 ∧ z > 0 :=
by {
  sorry
}

end isosceles_triangle_aacute_l163_163719


namespace limit_C_n_l163_163994

theorem limit_C_n (C : ℕ → ℝ) 
  (h : ∀ n, C n = n*(n-1) / 2) : 
  tendsto (λ n, C n / (2 * n^2 + n)) at_top (𝓝 (1/4)) :=
by {
  -- Definition of C_n using hypothesis
  simp [h],  
  -- Substitution step and limit evaluation would be done here
  sorry 
}

end limit_C_n_l163_163994


namespace tangent_line_at_e_l163_163197

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x

theorem tangent_line_at_e :
  let a := Real.exp 1 in -- a = e
  let slope := (deriv f) a in
  let point := (a, 0) in
  let tangent_line := λ x, slope * (x - a) + 0 in
  tangent_line = λ x, x - a :=
by
  let a := Real.exp 1
  let slope := (deriv f) a
  let point := (a, 0)
  let tangent_line := λ x, slope * (x - a) + 0
  sorry -- Proof not required

end tangent_line_at_e_l163_163197


namespace remaining_pencils_total_l163_163809

-- Definitions corresponding to the conditions:
def J : ℝ := 300
def J_d : ℝ := 0.30 * J
def J_r : ℝ := J - J_d

def V : ℝ := 2 * J
def V_d : ℝ := 125
def V_r : ℝ := V - V_d

def S : ℝ := 450
def S_d : ℝ := 0.60 * S
def S_r : ℝ := S - S_d

-- Proving the remaining pencils add up to the required amount:
theorem remaining_pencils_total : J_r + V_r + S_r = 865 := by
  sorry

end remaining_pencils_total_l163_163809


namespace probability_one_head_in_three_tosses_l163_163569

/--
The probability of getting exactly one head in three independent tosses of a fair coin is 3/8.
--/
theorem probability_one_head_in_three_tosses : 
  let p := (1 / 2 : ℚ)
  in (3 choose 1) * p * (p ^ 2) = 3 / 8 :=
by
  sorry

end probability_one_head_in_three_tosses_l163_163569


namespace rowing_speed_l163_163172

theorem rowing_speed :
  ∀ (initial_width final_width increase_per_10m : ℝ) (time_seconds : ℝ)
  (yards_to_meters : ℝ → ℝ) (width_increase_in_yards : ℝ) (distance_10m_segments : ℝ) 
  (total_distance : ℝ),
  initial_width = 50 →
  final_width = 80 →
  increase_per_10m = 2 →
  time_seconds = 30 →
  yards_to_meters 1 = 0.9144 →
  width_increase_in_yards = (final_width - initial_width) →
  width_increase_in_yards * (yards_to_meters 1) = 27.432 →
  distance_10m_segments = (width_increase_in_yards * (yards_to_meters 1)) / 10 →
  total_distance = distance_10m_segments * 10 →
  (total_distance / time_seconds) = 0.9144 :=
by
  intros initial_width final_width increase_per_10m time_seconds yards_to_meters 
        width_increase_in_yards distance_10m_segments total_distance
  sorry

end rowing_speed_l163_163172


namespace more_divisible_by_7_than_11_l163_163245

open Nat

theorem more_divisible_by_7_than_11 :
  let N := 10000
  let count_7_not_11 := (N / 7) - (N / 77)
  let count_11_not_7 := (N / 11) - (N / 77)
  count_7_not_11 > count_11_not_7 := 
  by
    let N := 10000
    let count_7_not_11 := (N / 7) - (N / 77)
    let count_11_not_7 := (N / 11) - (N / 77)
    sorry

end more_divisible_by_7_than_11_l163_163245


namespace shaded_area_sum_l163_163904

-- Given conditions as definitions
def radius_A := sqrt 3 - 1
def radius_B := 3 - sqrt 3
def radius_C := 1 + sqrt 3

-- Each circle is externally tangent to the other two
def is_tangent (r1 r2 : ℝ) : Prop := 
  ∃ d, d = r1 + r2

-- Definition of the shaded area in specific form
def shaded_area (a b c : ℚ) : ℝ := a * sqrt 3 + b * π + c * π * sqrt 3

-- The main theorem statement
theorem shaded_area_sum (a b c : ℚ) :
  is_tangent radius_A radius_B ∧
  is_tangent radius_A radius_C ∧
  is_tangent radius_B radius_C ∧
  (∃ a b c : ℚ, shaded_area a b c = 2 * sqrt 3 - π + (π * sqrt 3) / 2)
  → a + b + c = 3/2 :=
sorry

end shaded_area_sum_l163_163904


namespace choose_president_and_committee_l163_163107

theorem choose_president_and_committee (n : ℕ) (k : ℕ) (hn : n = 10) (hk : k = 3) : 
  (finset.card (finset.univ : finset (fin (n - 1))).choose k) * n = 840 :=
by
  rw [hn, hk]
  have h1 : finset.card (finset.univ : finset (fin 9)).choose 3 = nat.choose 9 3 := rfl
  rw [h1, nat.choose]
  exact dec_trivial

end choose_president_and_committee_l163_163107


namespace monotonically_increasing_interval_l163_163773

def f (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem monotonically_increasing_interval : 
  ∀ x y : ℝ, x ∈ Ioo (-1) y → y ∈ Ioo (-1) y → (f x ≤ f y) := by
  sorry

end monotonically_increasing_interval_l163_163773


namespace symmetry_axis_l163_163740

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * sin (2 * x - π / 3)

-- Define the property we want to prove: x = -π/12 is an axis of symmetry
-- This means: for any x, f(a - x) = f(a + x) where a is -π/12
theorem symmetry_axis : ∀ x: ℝ, f (-π / 12 - x) = f (-π / 12 + x) := 
by
  sorry

end symmetry_axis_l163_163740


namespace geometric_series_sum_first_six_terms_l163_163310

theorem geometric_series_sum_first_six_terms :
  let a := (3 : ℚ)
  let r := (1 / 3 : ℚ)
  let n := (6 : ℕ)
  (a * (1 - r ^ n) / (1 - r)) = (364 / 81 : ℚ) :=
by
  let a := 3
  let r := (1 / 3 : ℚ)
  let n := 6
  have h1 : 1 - r ^ n = 1 - (1 / 3) ^ n := sorry
  have h2 : 1 - r = 1 - (1 / 3) := sorry
  rw [h1, h2]
  sorry

end geometric_series_sum_first_six_terms_l163_163310


namespace find_b_l163_163083

-- Definitions
def gcd (a b : ℕ) : ℕ := Nat.gcd a b

-- Given conditions
variables (a b : ℕ)
variables (h1 : gcd 16 20 * gcd 18 b = 2)

-- Proof statement
theorem find_b : b = 1 := sorry

end find_b_l163_163083


namespace mediant_fraction_of_6_11_and_5_9_minimized_is_31_l163_163147

theorem mediant_fraction_of_6_11_and_5_9_minimized_is_31 
  (p q : ℕ) (h_pos : 0 < p ∧ 0 < q)
  (h_bounds : (6 : ℝ) / 11 < p / q ∧ p / q < 5 / 9)
  (h_min_q : ∀ r s : ℕ, (6 : ℝ) / 11 < r / s ∧ r / s < 5 / 9 → s ≥ q) :
  p + q = 31 :=
sorry

end mediant_fraction_of_6_11_and_5_9_minimized_is_31_l163_163147


namespace slope_of_bisector_is_minus_two_l163_163794

-- Definitions of points used in the conditions
structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := 0, y := 2 }
def Q : Point := { x := 12, y := 8 }
def M : Point := { x := 4, y := 4 }

-- Function to determine if a line passing through a point is equidistant from two points
def is_equidistant_from (L M : Point) (P Q : Point) : Prop :=
  let dL_P := real.sqrt ((L.x - P.x)^2 + (L.y - P.y)^2)
  let dL_Q := real.sqrt ((L.x - Q.x)^2 + (L.y - Q.y)^2)
  dL_P = dL_Q

-- Function definition for the slope of a line given two points
def slope (A B : Point) : ℝ :=
  (B.y - A.y) / (B.x - A.x)

noncomputable def is_perpendicular_to (L1 L2 : Point) :=
  slope L1 L2 = -2

-- Theorem stating the slope of the line that is equidistant from P and Q and passes through the point (4, 4)
theorem slope_of_bisector_is_minus_two :
  ∀ L, is_equidistant_from L M P Q → slope L M = -2 :=
sorry

end slope_of_bisector_is_minus_two_l163_163794


namespace clean_room_to_homework_ratio_l163_163808

-- Define the conditions
def timeHomework : ℕ := 30
def timeWalkDog : ℕ := timeHomework + 5
def timeTrash : ℕ := timeHomework / 6
def totalTimeAvailable : ℕ := 120
def remainingTime : ℕ := 35

-- Definition to calculate total time spent on other tasks
def totalTimeOnOtherTasks : ℕ := timeHomework + timeWalkDog + timeTrash

-- Definition to calculate the time to clean the room
def timeCleanRoom : ℕ := totalTimeAvailable - remainingTime - totalTimeOnOtherTasks

-- The theorem to prove the ratio
theorem clean_room_to_homework_ratio : (timeCleanRoom : ℚ) / (timeHomework : ℚ) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end clean_room_to_homework_ratio_l163_163808


namespace find_a_b_extreme_values_l163_163053

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x - 1

theorem find_a_b (a b : ℝ) :
  (∃ a b, f a b (-1) = 4 ∧ f'(-1) = 0 ∧ f a b = x^3 - 3x^2 - 9x - 1) :=
by 
    have f_der : f' (a b x : ℝ) = 3 * x^2 + 2 * a * x + b := sorry
    sorry

theorem extreme_values :
  let f := λ x : ℝ, x^3 - 3x^2 - 9x - 1 in
  (∀ x ∈ set.Icc 0 4, f x ≤ -1) ∧ (∀ x ∈ set.Icc 0 4, f x ≥ -28) := 
by 
    sorry

end find_a_b_extreme_values_l163_163053


namespace coordinate_system_problem_l163_163455

theorem coordinate_system_problem :
  let parametric_eq_of_l (t : ℝ) := (⟨t * Real.cos (2 * Real.pi / 3), 4 + t * Real.sin (2 * Real.pi / 3)⟩ : ℝ × ℝ)
  let polar_eq_of_C (θ : ℝ) := (4 * Real.cos θ, 4 * Real.sin θ)
  let ordinary_eq_of_l := (√3) * x + y - 4 = 0
  let rectangular_eq_of_C := x^2 + y^2 = 16
  let angle_AOB := Real.angle (some_point_A) (0, 0) (some_point_B)
  in
  (∀ t : ℝ, ∃ x y : ℝ, parametric_eq_of_l t = (x, y)) ∧
  (∀ θ : ℝ, ∃ x y : ℝ, polar_eq_of_C θ = (x, y)) ∧
  (∀ x y : ℝ, ordinary_eq_of_l x y → rectangular_eq_of_C x y → (Real.cos (0.5 * angle_AOB)) = 1/2) →
  angle_AOB = 2 * Real.pi / 3 :=
by 
  sorry

end coordinate_system_problem_l163_163455


namespace intersection_M_N_l163_163411

def M : Set ℝ := {x | -3 < x ∧ x < 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def intersection : Set ℝ := {x | x = -1 ∨ x = 0 ∨ x = 1}

theorem intersection_M_N : M ∩ N = intersection := 
by {
  sorry
}

end intersection_M_N_l163_163411


namespace sanya_towels_count_l163_163536

-- Defining the conditions based on the problem
def towels_per_hour := 7
def hours_per_day := 2
def days_needed := 7

-- The main statement to prove
theorem sanya_towels_count : 
  (towels_per_hour * hours_per_day * days_needed = 98) :=
by
  sorry

end sanya_towels_count_l163_163536


namespace laura_running_speed_l163_163465

theorem laura_running_speed (x : ℝ) (hx : 3 * x + 1 > 0) : 
    (30 / (3 * x + 1)) + (10 / x) = 31 / 12 → x = 7.57 := 
by 
  sorry

end laura_running_speed_l163_163465


namespace squares_circles_intersections_l163_163315

noncomputable def number_of_intersections (p1 p2 : (ℤ × ℤ)) (square_side : ℚ) (circle_radius : ℚ) : ℕ :=
sorry -- function definition placeholder

theorem squares_circles_intersections :
  let p1 := (0, 0)
  let p2 := (1009, 437)
  let square_side := (1 : ℚ) / 4
  let circle_radius := (1 : ℚ) / 8
  (number_of_intersections p1 p2 square_side circle_radius) = 526 := by
  sorry

end squares_circles_intersections_l163_163315


namespace arc_length_of_circle_l163_163781

theorem arc_length_of_circle (r : ℝ) (α : ℝ) (h_r : r = 3) (h_α : α = π / 7) : 
  arc_length r α = 3 * π / 7 :=
by
  -- This is where the proof should go
  sorry

def arc_length (r : ℝ) (α : ℝ) : ℝ := α * r

end arc_length_of_circle_l163_163781


namespace probability_all_girls_l163_163274

noncomputable def.choose_comb {n k : Nat} : ℚ :=
  (Nat.choose n k : ℚ)

theorem probability_all_girls (total_members girls boys k : ℕ)
  (h_members : total_members = 15)
  (h_girls : girls = 7)
  (h_boys : boys = 8)
  (h_k : k = 3) :
  (choose_comb girls k / choose_comb total_members k) = (1 / 13) :=
by
  -- Define the conditions
  rw [h_members, h_girls, h_boys, h_k]
  -- Sorry to skip the proof
  sorry

end probability_all_girls_l163_163274


namespace solve_for_y_l163_163853
-- Import the necessary Lean 4 libraries

-- Define the problem conditions and the proof statement
theorem solve_for_y (y : ℤ) (h : 3^(y - 2) = 9^(y + 2)) : y = -6 :=
begin
  -- The statement requires us to show y = -6 given the initial condition
  sorry
end

end solve_for_y_l163_163853


namespace carlson_fraction_l163_163832

-- Define variables
variables (n m k p T : ℝ)

theorem carlson_fraction (h1 : k = 0.6 * n)
                         (h2 : p = 2.5 * m)
                         (h3 : T = n * m + k * p) :
                         k * p / T = 3 / 5 := by
  -- Omitted proof
  sorry

end carlson_fraction_l163_163832


namespace max_a_decreasing_interval_l163_163078

noncomputable def f (x : ℝ) : ℝ := Real.cos x - Real.sin x

theorem max_a_decreasing_interval :
  ∀ a : ℝ, (∀ x : ℝ, -a ≤ x ∧ x ≤ a → f' x < 0) → a ≤ 3 * Real.pi / 4 :=
by
  sorry

end max_a_decreasing_interval_l163_163078


namespace committee_selection_l163_163836

theorem committee_selection :
  let total_members := 30
  let dept_size := 10
  let remaining_members := 27
  let choose (n k : ℕ) := nat.factorial n / (nat.factorial k * nat.factorial (n - k))
  let ways_to_choose :=
    10 * 10 * 10 * choose remaining_members 2
  ways_to_choose = 351000 := by
  sorry

end committee_selection_l163_163836


namespace Jim_new_total_pages_per_week_l163_163813

-- Given conditions
constant initialReadingSpeed : ℕ := 40
constant initialTotalPages : ℕ := 600
constant increasedSpeedFactor : ℕ := 150
constant hoursLessPerWeek : ℕ := 4

-- Prove that Jim reads 660 pages per week now.
theorem Jim_new_total_pages_per_week : 
  let newReadingSpeed := (initialReadingSpeed * increasedSpeedFactor) / 100
  let initialHoursPerWeek := initialTotalPages / initialReadingSpeed
  let newHoursPerWeek := initialHoursPerWeek - hoursLessPerWeek
  newReadingSpeed * newHoursPerWeek = 660 := 
by {
  sorry
}

end Jim_new_total_pages_per_week_l163_163813


namespace boxes_of_apples_l163_163224

theorem boxes_of_apples (n_crates apples_per_crate rotten_apples apples_per_box : ℕ) 
  (h1 : n_crates = 12) 
  (h2 : apples_per_crate = 42)
  (h3: rotten_apples = 4) 
  (h4 : apples_per_box = 10) : 
  (n_crates * apples_per_crate - rotten_apples) / apples_per_box = 50 :=
by
  sorry

end boxes_of_apples_l163_163224


namespace increase_80_by_50_percent_l163_163627

theorem increase_80_by_50_percent :
  let initial_number : ℕ := 80
  let increase_percentage : ℝ := 0.5
  initial_number + (initial_number * increase_percentage) = 120 :=
by
  sorry

end increase_80_by_50_percent_l163_163627


namespace number_of_circles_gt_bound_l163_163822

-- Define the conditions
variable {n : ℕ}
variable (P : Fin (2 * n + 3) → EuclideanGeometry.Point ℝ)

-- Assume conditions
def conditions (P : Fin (2 * n + 3) → EuclideanGeometry.Point ℝ) : Prop :=
  (∀ (i j k : Fin (2 * n + 3)), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    ¬ EuclideanGeometry.collinear ℝ {P i, P j, P k}) ∧
  (∀ (i j k l : Fin (2 * n + 3)), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
    ¬ EuclideanGeometry.concyclic ℝ {P i, P j, P k, P l})

-- Define the problem statement
theorem number_of_circles_gt_bound
  (hP : conditions P) :
  let K := {C : Set (EuclideanGeometry.Circle ℝ) //
    ∃ (i j k : Fin (2 * n + 3)), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    C = EuclideanGeometry.circumcircle ℝ {P i, P j, P k} ∧
    (∃ (S : Finset (Fin (2 * n + 3))), S.card = n ∧ S ⊆ Finset.filter
      (λ x, EuclideanGeometry.inside ℝ C (P x))
      (Finset.univ.erase i).erase j.erase k)}.card in
  K > (Real.pi⁻¹ : ℝ) * Nat.choose (2 * n + 3) 2 :=
sorry

end number_of_circles_gt_bound_l163_163822


namespace fred_socks_probability_l163_163705

noncomputable def socks_probability : ℚ := 27 / 66

theorem fred_socks_probability :
  let socks := ["red", "red", "red", "blue", "blue", "blue",
                "green", "green", "green", "yellow", "yellow", "yellow"] in
  let draw_5_socks := { s : multiset string // s.card = 5 ∧ s ⊆ socks.to_multiset } in
  let count_desired_outcomes := (num : ℕ) in
  ( ∃ (s : draw_5_socks), s.val.count "red" = 2 ∧ s.val.count "blue" = 1 ∧ s.val.count "green" = 1 ∧ s.val.count "yellow" = 1 ) ∨
  ( ∃ (s : draw_5_socks), s.val.count "blue" = 2 ∧ s.val.count "red" = 1 ∧ s.val.count "green" = 1 ∧ s.val.count "yellow" = 1 ) ∨
  ( ∃ (s : draw_5_socks), s.val.count "green" = 2 ∧ s.val.count "red" = 1 ∧ s.val.count "blue" = 1 ∧ s.val.count "yellow" = 1 ) ∨
  ( ∃ (s : draw_5_socks), s.val.count "yellow" = 2 ∧ s.val.count "red" = 1 ∧ s.val.count "blue" = 1 ∧ s.val.count "green" = 1 ) →
  count_desired_outcomes / 792 = socks_probability :=
sorry

end fred_socks_probability_l163_163705


namespace sin_is_odd_l163_163127

-- Define the function f
def f (x : ℝ) : ℝ := Real.sin x

-- Define the condition for f(x) to be even
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Define the condition for f(x) to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = -f (-x)

theorem sin_is_odd : is_odd f :=
by
  -- The proof is omitted and left as an exercise
  sorry


end sin_is_odd_l163_163127


namespace height_of_Brixton_l163_163801

theorem height_of_Brixton
  (I Z B Zr : ℕ)
  (h1 : I = Z + 4)
  (h2 : Z = B - 8)
  (h3 : Zr = B)
  (h4 : (I + Z + B + Zr) / 4 = 61) :
  B = 64 := by
  sorry

end height_of_Brixton_l163_163801


namespace speed_of_man_in_still_water_l163_163633

variable (v_m v_s : ℝ)

theorem speed_of_man_in_still_water
  (h1 : (v_m + v_s) * 4 = 24)
  (h2 : (v_m - v_s) * 5 = 20) :
  v_m = 5 := 
sorry

end speed_of_man_in_still_water_l163_163633


namespace part_I_part_II_l163_163028

-- Definitions
variables {A B C : ℝ} {a b c : ℝ} {area : ℝ}

-- Part (I):
theorem part_I (hB : B = π / 3) (ha : a = sqrt 2) (hb : b = sqrt 3) : A = π / 4 :=
sorry

-- Part (II):
theorem part_II (hB : B = π / 3) (ha : a = sqrt 2) (harea : area = 3 * sqrt 3 / 2) :
  b = sqrt 14 :=
sorry

end part_I_part_II_l163_163028


namespace harry_annual_pet_feeding_cost_l163_163066

def monthly_cost_snake := 10
def monthly_cost_iguana := 5
def monthly_cost_gecko := 15
def num_snakes := 4
def num_iguanas := 2
def num_geckos := 3
def months_in_year := 12

theorem harry_annual_pet_feeding_cost :
  (num_snakes * monthly_cost_snake + 
   num_iguanas * monthly_cost_iguana + 
   num_geckos * monthly_cost_gecko) * 
   months_in_year = 1140 := 
sorry

end harry_annual_pet_feeding_cost_l163_163066


namespace arctan_quadratic_roots_l163_163073

theorem arctan_quadratic_roots :
  (∃! x₁, -real.pi / 2 < real.arctan x₁ ∧ real.arctan x₁ = x₁^2 - 1.6) ∧
  (∃! x₂, -real.pi / 2 < real.arctan x₂ ∧ real.arctan x₂ = x₂^2 - 1.6) :=
sorry

end arctan_quadratic_roots_l163_163073


namespace selection_schemes_soccer_basketball_calligraphy_l163_163887

open Finset

-- Define the activities
inductive Activity
| Music : Activity
| Soccer : Activity
| Basketball : Activity
| Calligraphy : Activity
| Speaking : Activity
| Painting : Activity
| Handicraft : Activity
| Technology : Activity

open Activity

-- Define the schedule as sets of activities for each day of the week
def schedule : ℕ → Finset Activity
| 1 := {Music, Soccer, Basketball, Calligraphy}
| 2 := {Speaking, Soccer, Calligraphy, Painting}
| 3 := {Handicraft, Soccer, Technology, Basketball}
| 4 := {Speaking, Soccer, Basketball, Calligraphy}
| 5 := {Basketball, Soccer, Calligraphy, Technology}
| _ := ∅

-- Define the main statement
theorem selection_schemes_soccer_basketball_calligraphy :
  (∃ s : Finset ℕ, s.card = 3 ∧
     ∀ x ∈ s, ∀ y ∈ s, x ≠ y →
       (schedule x).card ≥ 1 ∧
        Soccer ∈ schedule x ∧
        Basketball ∈ schedule y ∧
        Calligraphy ∈ schedule y ∧ 
          x ≠ y ∧ _) :=
  sorry

end selection_schemes_soccer_basketball_calligraphy_l163_163887


namespace ellipses_intersect_at_two_points_l163_163700

variable (T A B C Q P : Point)
noncomputable
def E (P Q R : Point) : Ellipse := sorry

axiom collinear_points (h : ∀ P Q R : Point, Collinear P Q R) : 
  Collinear T A B ∧ Collinear T A C 

axiom ray_intersects_ellipse_E_ab (h : ∀ P Q R : Point, Ray_intersects P Q R) :
  Ray A B T → (Ray A B Q ∧ Ray A C P)

theorem ellipses_intersect_at_two_points : 
  Ellipse (E B P A) ∧ Ellipse (E C Q A) → 
  ∃ X Y : Point, X ≠ Y ∧ (OnEllipse X (E B P A) ∧ OnEllipse X (E C Q A)) ∧ 
  (OnEllipse Y (E B P A) ∧ OnEllipse Y (E C Q A)) := 
  sorry

end ellipses_intersect_at_two_points_l163_163700


namespace train_speed_l163_163964

theorem train_speed (length_of_train time_to_cross : ℕ) (h_length : length_of_train = 50) (h_time : time_to_cross = 3) : 
  (length_of_train / time_to_cross : ℝ) * 3.6 = 60 := by
  sorry

end train_speed_l163_163964


namespace problem_statement_l163_163724

variable {a b : ℝ}

-- Define the conditions
def cond1 : Prop := a > 0
def cond2 : Prop := b > 0
def cond3 : Prop := a + b = 4

-- Prove the main statements under these conditions
theorem problem_statement (h1 : cond1) (h2 : cond2) (h3 : cond3) :
  a + b = 4 ∧ ∃ a b, cond1 ∧ cond2 ∧ cond3 ∧ (∀ a b, (cond1 ∧ cond2 ∧ cond3) → (by calc
    let min_val := \frac{1}{4}a^2 + \frac{1}{9}b^2
    min_val ≥ \frac{16}{13})) :=
by
  sorry

end problem_statement_l163_163724


namespace smallest_positive_value_of_n_l163_163111

-- Let {a_n} be an arithmetic sequence
variables {a : ℕ → ℝ} {d : ℝ}

-- Condition 1: In the arithmetic sequence {a_n}, a_11 / a_10 < -1
def condition1 (a : ℕ → ℝ) : Prop := a 11 / a 10 < -1

-- Definition of the sum of the first n terms of an arithmetic sequence
def S (a : ℕ → ℝ) (n : ℕ) : ℝ := 
  n * (a 1 + a n - (n - 1) * d) / 2

-- The question asks for the smallest positive value of n where S_n is maximized
theorem smallest_positive_value_of_n (a : ℕ → ℝ) (d : ℝ) 
  (hcond1 : condition1 a) : 
  ∃ n : ℕ, S a n > 0 ∧ ∀ m : ℕ, m < n → S a m < S a n :=
begin
  -- We are interested in the smallest positive value of n
  use 20,
  split,
  { sorry },  -- Prove S a 20 > 0
  { intros m hm,
    sorry }  -- Prove ∀ m < 20, S a m < S a 20
end

end smallest_positive_value_of_n_l163_163111


namespace proof_problem_l163_163708

noncomputable def a : ℝ := 1 / 2
noncomputable def b : ℝ := real.sqrt 2
noncomputable def c : ℝ := real.log 2 / real.log 3

theorem proof_problem : b > c ∧ c > a := by
  -- Proof will be here
  sorry

end proof_problem_l163_163708


namespace find_f_prime_one_l163_163049

/-- Given the function f satisfying f(x) = 2 * x * f'(1) + x^2, prove that f'(1) = -2. -/
theorem find_f_prime_one {f : ℝ → ℝ} (h₀ : ∀ x, f x = 2 * x * (f 1 - 2) + x ^ 2)
  (h₁ : ∀ x, f' x = 2 * (f 1 - 2) + 2 * x) :
  (f' 1 = -2) :=
sorry

end find_f_prime_one_l163_163049


namespace sqrt_expression_simplification_l163_163313

theorem sqrt_expression_simplification :
  (Real.sqrt 72 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 12 - |2 - Real.sqrt 6|) = 2 :=
by
  sorry

end sqrt_expression_simplification_l163_163313


namespace max_height_l163_163636

def height (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 50

theorem max_height : ∃ t : ℝ, height t = 130 :=
sorry

end max_height_l163_163636


namespace min_packs_to_115_l163_163539

def packs := [6, 10, 12, 25] -- Define the possible pack sizes
def target := 115 -- Define the target number of cans
def min_packs_required := 6 -- Define the minimum number of packs needed

theorem min_packs_to_115 :
  ∃ n : ℕ, 
  (∀ counts : list ℕ, counts.length = packs.length → 
    (counts.zip packs).map (λ t, t.1 * t.2) |>.sum = target → 
    counts.sum = n) →
  n = min_packs_required :=
  sorry

end min_packs_to_115_l163_163539


namespace divisor_of_subtracted_number_l163_163349

theorem divisor_of_subtracted_number (n : ℕ) (m : ℕ) (h : n = 5264 - 11) : Nat.gcd n 5264 = 5253 :=
by
  sorry

end divisor_of_subtracted_number_l163_163349


namespace parallel_lines_l163_163480

open Set

-- Define the essential elements: points and circles.
variable (A B P P' Q Q' : Point)
variable (Γ1 Γ2 : Circle)

-- Define the conditions
def intersecting_circles (Γ1 Γ2 : Circle) (A B : Point) : Prop :=
  A ∈ Γ1 ∧ A ∈ Γ2 ∧ B ∈ Γ1 ∧ B ∈ Γ2

def line_through (d : Line) (X Y : Point) : Prop :=
  X ∈ d ∧ Y ∈ d

def line_intersects_circle (d : Line) (Γ : Circle) (X : Point) : Prop :=
  X ∈ d ∧ X ∈ Γ

-- Given conditions
axiom h1 : intersecting_circles Γ1 Γ2 A B
axiom h2 : ∃(d : Line), line_through d A ∧ line_intersects_circle d Γ1 P ∧ line_intersects_circle d Γ2 Q
axiom h3 : ∃(d' : Line), line_through d' B ∧ line_intersects_circle d' Γ1 P' ∧ line_intersects_circle d' Γ2 Q'

-- The theorem to be proven
theorem parallel_lines : Parallel (LineThrough P P') (LineThrough Q Q') :=
by
  sorry

end parallel_lines_l163_163480


namespace chad_dog_food_packages_l163_163666

theorem chad_dog_food_packages (d : ℕ) :
  let cat_food_packages := 6 in
  let cans_per_cat_package := 9 in
  let cans_per_dog_package := 3 in
  let extra_cans_cat := 48 in
  (cat_food_packages * cans_per_cat_package) = 
  (d * cans_per_dog_package) + extra_cans_cat → 
  d = 2 :=
by
  sorry

end chad_dog_food_packages_l163_163666


namespace solve_for_x_l163_163621

theorem solve_for_x (x : ℕ) 
  (h : 225 + 2 * 15 * 4 + 16 = x) : x = 361 := 
by 
  sorry

end solve_for_x_l163_163621


namespace value_of_b_is_one_l163_163482

open Complex

theorem value_of_b_is_one (a b : ℝ) (h : (1 + I) / (1 - I) = a + b * I) : b = 1 := 
by
  sorry

end value_of_b_is_one_l163_163482


namespace percent_of_x_is_z_l163_163612

variable {x y z : ℝ}

theorem percent_of_x_is_z 
  (h1 : 0.45 * z = 0.72 * y) 
  (h2 : y = 0.75 * x) : 
  z / x = 1.2 := 
sorry

end percent_of_x_is_z_l163_163612


namespace integral_f_l163_163314

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then x ^ 2
  else if 1 < x ∧ x ≤ 2 then 2 - x
  else 0

theorem integral_f : ∫ x in (0 : ℝ)..2, f x = 5 / 6 :=
by 
  sorry

end integral_f_l163_163314


namespace supremum_of_expression_l163_163698

theorem supremum_of_expression (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) : 
  ∃ M : ℝ, (∀ x, - (1 / (2 * a) + 2 / b) ≤ M) ∧ 
            (∀ N : ℝ, (∀ x, - (1 / (2 * a) + 2 / b) ≤ N) → M ≤ N) ∧ 
            M = -9 / 2 :=
begin
  sorry
end

end supremum_of_expression_l163_163698


namespace intersection_M_N_l163_163059

variable (x : ℝ)

def M := {x : ℝ | -2 < x ∧ x < 2}
def N := {x : ℝ | x > 1}

theorem intersection_M_N :
  {x | x ∈ M ∧ x ∈ N} = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end intersection_M_N_l163_163059


namespace train_b_leaves_after_train_a_l163_163232

noncomputable def time_difference := 2

theorem train_b_leaves_after_train_a 
  (speedA speedB distance t : ℝ) 
  (h1 : speedA = 30)
  (h2 : speedB = 38)
  (h3 : distance = 285)
  (h4 : distance = speedB * t)
  : time_difference = (distance - speedA * t) / speedA := 
by 
  sorry

end train_b_leaves_after_train_a_l163_163232


namespace product_sets_not_identical_l163_163594

theorem product_sets_not_identical :
  let nums := (list.range' 104 100).to_finset in
  let table := (matrix 10 10 ℕ) in
  ∀ (tbl : table) (rows : fin 10 → ℕ) (cols : fin 10 → ℕ),
  (∀ i, rows i = (finset.univ.product (finset.range 10)).to_set.sum (λ j, tbl i j) ∧ tbl i j ∈ nums) ∧
  (∀ j, cols j = (finset.univ.product (finset.range 10)).to_set.sum (λ i, tbl i j) ∧ tbl i j ∈ nums) →
  (¬ ∃ r c, (rows r = cols c)) := 
by
  sorry

end product_sets_not_identical_l163_163594


namespace problem_condition_l163_163369

variable {f : ℝ → ℝ}
variable {a b : ℝ}

noncomputable def fx_condition (f : ℝ → ℝ) :=
  ∀ x : ℝ, f x + x * (deriv f x) < 0

theorem problem_condition {f : ℝ → ℝ} {a b : ℝ} (h1 : fx_condition f) (h2 : a < b) :
  a * f a > b * f b :=
sorry

end problem_condition_l163_163369


namespace EF_vector_eq_angle_EMF_eq_2pi_div_3_l163_163119

open Real
open Affine

variable {V : Type _} [InnerProductSpace ℝ V]

variables (a b : V)
variables (A B C D E F M : V)
variables [OrderedAddCommGroup (Module ℝ V)]
variables (M_eq_AF_inter_DE : affine_inter AF DE M)
variables (b_norm : ∥b∥ = (3 / 2) * ∥a∥)

-- Conditions given

axiom parallelogram_ABCD : parallelogram A B C D
axiom E_is_midpoint_AB : midpoint A B E
axiom F_divides_BC : divides_segment_closer_to_B C B F 3
axiom angle_DAB_eq_pi_div_3 : ∠D A B = π / 3
axiom AB_vector : vector AB = a
axiom AD_vector : vector AD = b

-- Questions to prove

theorem EF_vector_eq : vector EF = (1 / 2) * a + (1 / 3) * b :=
sorry

theorem angle_EMF_eq_2pi_div_3 : ∠E M F = 2 * π / 3 :=
sorry

end EF_vector_eq_angle_EMF_eq_2pi_div_3_l163_163119


namespace prove_sum_opposite_sqrt_l163_163892

noncomputable def sum_opposite_sqrt (a b : ℝ) (h₁ : a = 1 - Real.sqrt 2) (h₂ : b = Real.sqrt (Real.sqrt 81)) : Prop :=
  (-(a) + b = 2 + Real.sqrt 2) ∨ (-(a) + b = Real.sqrt 2 - 4)

theorem prove_sum_opposite_sqrt :
  ∃ a b, a = 1 - Real.sqrt 2 ∧ b = Real.sqrt (Real.sqrt 81) ∧ sum_opposite_sqrt a b (rfl) (rfl) :=
begin
  use [1 - Real.sqrt 2, Real.sqrt (Real.sqrt 81)],
  split, { refl },
  split, { refl },
  sorry

end prove_sum_opposite_sqrt_l163_163892


namespace maximum_marks_l163_163613

theorem maximum_marks (passing_percentage : ℝ) (score : ℝ) (shortfall : ℝ) (total_marks : ℝ) : 
  passing_percentage = 30 → 
  score = 212 → 
  shortfall = 16 → 
  total_marks = (score + shortfall) * 100 / passing_percentage → 
  total_marks = 760 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  assumption

end maximum_marks_l163_163613


namespace equal_regions_by_median_l163_163013

theorem equal_regions_by_median {A B C D O : Type*}
  (h_triangle : triangle A B C)
  (h_median : median A B C D)
  (h_centroid : centroid A B C O)
  : regions_created_on_each_side_AD_are_equal A B C D O :=
sorry

end equal_regions_by_median_l163_163013


namespace tim_total_spent_l163_163765

def lunch_cost : ℝ := 50.50
def tip_rate : ℝ := 0.15
def total_spent : ℝ := 58.08

theorem tim_total_spent : lunch_cost + (tip_rate * lunch_cost).round = total_spent := 
by
  sorry

end tim_total_spent_l163_163765


namespace OE_passes_through_centroid_of_triangle_FGH_l163_163144

-- Definitions of points in the tetrahedron
variables {A B C D O E F G H Q : Type*}
variables [RegularTetrahedron A B C D] [Center O A B C D]
variables [OnFace E A B C] [Projection F E A B D] [Projection G E B C D] [Projection H E C D A]

-- Main theorem statement
theorem OE_passes_through_centroid_of_triangle_FGH 
  (hO : Center O A B C D) 
  (hE : OnFace E A B C) 
  (hF : Projection F E A B D)
  (hG : Projection G E B C D)
  (hH : Projection H E C D A) :
  passes_through OE (Centroid Q F G H) :=
sorry

end OE_passes_through_centroid_of_triangle_FGH_l163_163144


namespace train_speed_l163_163958

-- Define the conditions
def train_length : ℝ := 50 -- Length of the train in meters
def crossing_time : ℝ := 3 -- Time to cross the pole in seconds

-- Define the speed in meters per second and convert it to km/hr
noncomputable def speed_mps : ℝ := train_length / crossing_time
noncomputable def speed_kmph : ℝ := speed_mps * 3.6 -- Conversion factor

-- Theorem statement: Prove that the calculated speed in km/hr is 60 km/hr
theorem train_speed : speed_kmph = 60 := by
  sorry

end train_speed_l163_163958


namespace shortest_track_length_l163_163156

open Nat

def Melanie_track_length := 8
def Martin_track_length := 20

theorem shortest_track_length :
  Nat.lcm Melanie_track_length Martin_track_length = 40 :=
by
  sorry

end shortest_track_length_l163_163156


namespace ellipse_equation_l163_163717

-- Defining the conditions
variable (a b : ℝ)
variable (ha : a > b)
variable (hb : b > 0)
variable (c : ℝ := 3)
variable (E : set (ℝ × ℝ) := { p | (p.1^2 / a^2) + (p.2^2 / b^2) = 1 })
variable (F : ℝ × ℝ := (3, 0))
variable (mid_AB : ℝ × ℝ := (1, -1))

-- The statement to prove
theorem ellipse_equation :
  ∃ (a b : ℝ), 
    a > b ∧ b > 0 ∧ 
    (∀ p ∈ E, (p.1 - F.1)^2 + (p.2 - F.2)^2 = c^2) ∧ 
    (∃ (A B : ℝ × ℝ), 
       A ∈ E ∧ B ∈ E ∧ 
       (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = -1) 
  → (E = { p | (p.1^2 / 18) + (p.2^2 / 9) = 1 }) :=
sorry

end ellipse_equation_l163_163717


namespace train_speed_is_60_kmph_l163_163962

noncomputable def speed_of_train_in_kmph (length_meters time_seconds : ℝ) : ℝ :=
  (length_meters / time_seconds) * 3.6

theorem train_speed_is_60_kmph (length_meters time_seconds : ℝ) :
  length_meters = 50 → time_seconds = 3 → speed_of_train_in_kmph length_meters time_seconds = 60 :=
by
  intros h_length h_time
  simp [speed_of_train_in_kmph, h_length, h_time]
  norm_num
  sorry

end train_speed_is_60_kmph_l163_163962


namespace chess_tournament_matches_l163_163899

theorem chess_tournament_matches (n : ℕ) (b : ℕ) (h1 : n = 128) (h2 : b = 32) :
  let players_first_round := n - b in
  let matches_first_round := players_first_round / 2 in
  let players_after_first_round := matches_first_round + b in
  let matches_second_round := players_after_first_round / 2 in
  let matches_third_round := matches_second_round / 2 in
  let matches_fourth_round := matches_third_round / 2 in
  let matches_fifth_round := matches_fourth_round / 2 in
  let players_sixth_round := matches_fifth_round - 1 + 1 * 1 in
  let matches_sixth_round := players_sixth_round / 2 in
  let matches_final_round := 1 in
  matches_first_round + matches_second_round + matches_third_round + matches_fourth_round + matches_fifth_round + matches_sixth_round + matches_final_round = 127 :=
sorry

end chess_tournament_matches_l163_163899


namespace probability_P_eq_1_result_l163_163578

-- Define the set of vertices
def V := {2 * complex.i, -2 * complex.i, 1 + complex.i, -1 + complex.i, 1 - complex.i, -1 - complex.i, real.sqrt 2 + real.sqrt 2 * complex.i, -real.sqrt 2 - real.sqrt 2 * complex.i}

-- Define the random selection and product computation
noncomputable def random_selection_P : ℕ := 
  let selections := λ j : fin 16, (choose_one_from V)
  ∏ j in fin 16, selections j

-- Define the probability computation
noncomputable def probability_P_eq_1 : ℚ := 
  (count_valid_configurations V) / (8 ^ 16)

-- State the final theorem
theorem probability_P_eq_1_result : ∃ a b p : ℕ, probability_P_eq_1 = (a / p^b) ∧ nat.prime p ∧ ¬ p ∣ a ∧ a + b + p = 17 := 
by {
  use 1, 
  use 14, 
  use 2, 
  split,
  {
    exact by norm_num, 
  },
  {
    split,
    {
      exact nat.prime_two,
    },
    {
      split,
      {
        exact dec_trivial,
      },
      {
        exact dec_trivial,
      }
    }
  }
}

end probability_P_eq_1_result_l163_163578


namespace distance_sum_correct_l163_163438

noncomputable def pointA : (ℝ × ℝ) := (20, 0)
noncomputable def pointB : (ℝ × ℝ) := (0, 0)
noncomputable def pointD : (ℝ × ℝ) := (8, 6)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def AD: ℝ := distance pointA pointD
noncomputable def BD: ℝ := distance pointB pointD

theorem distance_sum_correct : AD + BD = 23.4 := 
by
  sorry

end distance_sum_correct_l163_163438


namespace interval_monotonic_increase_range_of_b_plus_c_l163_163416

-- Define the vectors and functions
def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, -1)
def vector_b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -1 / 2)
def f (x : ℝ) : ℝ := (vector_a x + vector_b x) ⬝ vector_a x - 1

-- Importing trigonometric and inequality properties
open Real

-- Proving the interval of monotonic increase for f(x)
theorem interval_monotonic_increase (k : ℤ) : 
  ∀ x,
  -π/6 + k * π ≤ x ∧ x ≤ π/3 + k * π →
  f (2 * x - π / 6) - 1 > 0 := 
sorry
  
-- Proving the range of values for b+c
theorem range_of_b_plus_c (A : ℝ) (b c : ℝ) (h1 : f (A / 2) = 3 / 2) (h2 : a = 2) : 
  2 < b + c ∧ b + c ≤ 4 := 
sorry

end interval_monotonic_increase_range_of_b_plus_c_l163_163416


namespace cylinder_h_over_r_equals_one_l163_163020

theorem cylinder_h_over_r_equals_one
  (A : ℝ) (r h : ℝ)
  (h_surface_area : A = 2 * π * r^2 + 2 * π * r * h)
  (V : ℝ := π * r^2 * h)
  (max_V : ∀ r' h', (A = 2 * π * r'^2 + 2 * π * r' * h') → (π * r'^2 * h' ≤ V) → (r' = r ∧ h' = h)) :
  h / r = 1 := by
sorry

end cylinder_h_over_r_equals_one_l163_163020


namespace contradiction_assumption_l163_163873

-- Define the numbers x, y, z
variables (x y z : ℝ)

-- Define the assumption that all three numbers are non-positive
def all_non_positive (x y z : ℝ) : Prop := x ≤ 0 ∧ y ≤ 0 ∧ z ≤ 0

-- State the proposition to prove using the method of contradiction
theorem contradiction_assumption (h : all_non_positive x y z) : ¬ (x > 0 ∨ y > 0 ∨ z > 0) :=
by
  sorry

end contradiction_assumption_l163_163873


namespace calculate_plot_size_in_acres_l163_163567

theorem calculate_plot_size_in_acres :
  let bottom_edge_cm : ℝ := 15
  let top_edge_cm : ℝ := 10
  let height_cm : ℝ := 10
  let cm_to_miles : ℝ := 3
  let miles_to_acres : ℝ := 640
  let trapezoid_area_cm2 := (bottom_edge_cm + top_edge_cm) * height_cm / 2
  let trapezoid_area_miles2 := trapezoid_area_cm2 * (cm_to_miles ^ 2)
  (trapezoid_area_miles2 * miles_to_acres) = 720000 :=
by
  sorry

end calculate_plot_size_in_acres_l163_163567


namespace product_of_roots_is_one_l163_163568

theorem product_of_roots_is_one
  (Q : Polynomial ℝ)
  (h1 : Q.Monic)
  (h2 : Q.degree = 4)
  (h3 : ∃ φ : ℝ, 0 < φ ∧ φ < π / 6 ∧ 0 = Q.eval (Complex.exp (Complex.I * φ)) ∧ 0 = Q.eval (Complex.I * Complex.exp (Complex.I * φ)))
  (h4 : Q.eval 0 = 2 * (2 * Real.sin φ * 2 * Real.cos φ))
  : Q.root_product = 1 := 
sorry

end product_of_roots_is_one_l163_163568


namespace triangle_interior_angle_proof_method_l163_163909

open Real

theorem triangle_interior_angle_proof_method :
  ∀ (A B C : ℝ), 
  (A + B + C = 180) → 
  (A > 60 ∧ B > 60 ∧ C > 60) → 
  (exists X : ℝ, X ≤ 60 ∧ (A = X ∨ B = X ∨ C = X)) ∧ 
  (A + B + C ≠ 180 → false) := 
by 
  intros A B C h_sum h_angles 
  have h_contr : A + B + C > 180 := by linarith
  have h_false : A + B + C ≠ 180 := by linarith h_sum h_contr
  exact ⟨h_false, false.elim (h_false h_sum)⟩

end triangle_interior_angle_proof_method_l163_163909


namespace yellow_tiled_area_is_correct_l163_163251

noncomputable def length : ℝ := 3.6
noncomputable def width : ℝ := 2.5 * length
noncomputable def total_area : ℝ := length * width
noncomputable def yellow_tiled_area : ℝ := total_area / 2

theorem yellow_tiled_area_is_correct (length_eq : length = 3.6)
    (width_eq : width = 2.5 * length)
    (total_area_eq : total_area = length * width)
    (yellow_area_eq : yellow_tiled_area = total_area / 2) :
    yellow_tiled_area = 16.2 := 
by sorry

end yellow_tiled_area_is_correct_l163_163251


namespace jim_pages_now_l163_163810

theorem jim_pages_now (rate : ℕ) (total_pages : ℕ) (increase : ℕ) (hours_less : ℕ) :
  (rate = 40) →
  (total_pages = 600) →
  (increase = 150) →
  (hours_less = 4) →
  let hours_before := total_pages / rate in
  let new_hours := hours_before - hours_less in
  let new_rate := rate * increase / 100 in
  let pages_now := new_hours * new_rate in
  pages_now = 660 :=
by
  intros h_rate h_total_pages h_increase h_hours_less,
  let hours_before := total_pages / rate,
  let new_hours := hours_before - hours_less,
  let new_rate := rate * increase / 100,
  let pages_now := new_hours * new_rate,
  have : pages_now = 11 * 60, from calc
    pages_now = (total_pages / rate - hours_less) * (rate * increase / 100) : by
      simp [hours_before, new_hours, new_rate]
    ... = 11 * 60 : by
      simp [h_rate, h_total_pages, h_increase, h_hours_less],
  have : 11 * 60 = 660, by norm_num,
  rw [this, this]
  sorry

end jim_pages_now_l163_163810


namespace train_speed_in_km_per_hr_l163_163955

-- Definitions from the problem conditions
def length_of_train : ℝ := 50
def time_to_cross_pole : ℝ := 3

-- Conversion factor from the problem 
def meter_per_sec_to_km_per_hr : ℝ := 3.6

-- Lean theorem statement based on problem conditions and solution
theorem train_speed_in_km_per_hr : 
  (length_of_train / time_to_cross_pole) * meter_per_sec_to_km_per_hr = 60 := by
  sorry

end train_speed_in_km_per_hr_l163_163955


namespace lengths_of_broken_lines_eq_l163_163799

noncomputable theory

open EuclideanGeometry

variable {R : Type*} [Real.R R]

/--
Given:
1. Inside the acute angle X O Y, points M and N are chosen such that \(\angle X O N = \angle Y O M\).
2. A point Q is chosen on segment O X such that \(\angle N Q O = \angle M Q X\).
3. A point P is chosen on segment O Y such that \(\angle N P O = \angle M P Y\).

Prove that:
The lengths of the broken lines M P N and M Q N are equal.
-/
theorem lengths_of_broken_lines_eq 
  {O X Y M N Q P : EuclideanGeometry.Point R}
  (h1 : ∠X O N = ∠Y O M)
  (h2 : Q ∈ OpenSegment O X)
  (h3 : ∠N Q O = ∠M Q X)
  (h4 : P ∈ OpenSegment O Y)
  (h5 : ∠N P O = ∠M P Y) :
  EuclideanGeometry.length (segment M P) + EuclideanGeometry.length (segment P N) =
  EuclideanGeometry.length (segment M Q) + EuclideanGeometry.length (segment Q N) :=
sorry

end lengths_of_broken_lines_eq_l163_163799


namespace linear_function_increasing_and_composition_eq_implies_values_monotonic_gx_implies_m_range_l163_163714

-- Defining the first part of the problem
theorem linear_function_increasing_and_composition_eq_implies_values
  (a b : ℝ)
  (H1 : ∀ x y : ℝ, x < y → a * x + b < a * y + b)
  (H2 : ∀ x : ℝ, a * (a * x + b) + b = 16 * x + 5) :
  a = 4 ∧ b = 1 :=
by
  sorry

-- Defining the second part of the problem
theorem monotonic_gx_implies_m_range (m : ℝ)
  (H3 : ∀ x1 x2 : ℝ, 1 ≤ x1 → x1 < x2 → (x2 + m) * (4 * x2 + 1) > (x1 + m) * (4 * x1 + 1)) :
  -9 / 4 ≤ m :=
by
  sorry

end linear_function_increasing_and_composition_eq_implies_values_monotonic_gx_implies_m_range_l163_163714


namespace smallest_period_of_f_find_A_b_area_l163_163753

open Real

namespace TriangleProof

def vector_m (x : ℝ) : ℝ × ℝ := (cos x, -1)
def vector_n (x : ℝ) : ℝ × ℝ := (sqrt 3 * sin x, -1 / 2)
def f (x : ℝ) : ℝ := (vector_m x + vector_n x) • (vector_m x)

noncomputable def period : ℝ := π

theorem smallest_period_of_f :
  ∀ x : ℝ, f (x + period) = f x :=
sorry

variables {a c A : ℝ} (b : ℝ) (area : ℝ)
variables (ha : a = 1) (hc : c = sqrt 3) (hA : A = π / 6)

theorem find_A_b_area (A : ℝ) (ha : a = 1) (hc : c = sqrt 3) (hA_max : f A = 3) :
  A = π / 6 ∧ b = 2 ∧ (area = (sqrt 3) / 2 ∨ area = (sqrt 3) / 4) :=
sorry

end TriangleProof

end smallest_period_of_f_find_A_b_area_l163_163753


namespace hexagon_divisible_by_three_l163_163947

-- Define the condition of the hexagon being divided into N parallelograms of equal area
def regular_hexagon_divided_into_equal_parallelograms (N : Nat) : Prop :=
  ∃ area : ℝ, ∀ (i : Fin N), parallelogram_of_area_i = area

-- Define the main theorem that N is divisible by 3
theorem hexagon_divisible_by_three (N : Nat) (h : regular_hexagon_divided_into_equal_parallelograms N) :
  N % 3 = 0 :=
begin
  sorry
end

end hexagon_divisible_by_three_l163_163947


namespace subsets_of_C_l163_163916

def card_set : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

def A : Set ℕ := {2, 4, 6, 8, 10}
def B : Set ℕ := {3, 6, 9}
def C : Set ℕ := card_set \ (A ∪ B)

theorem subsets_of_C : 
  (℘ C = { ∅, {1}, {5}, {7}, {1,5}, {1,7}, {5,7}, {1,5,7} }) := sorry

end subsets_of_C_l163_163916


namespace gen_formula_arithmetic_seq_sum_of_abs_arithmetic_seq_l163_163477

-- Definition of an arithmetic sequence and its sum
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + d * (n - 1)

def arithmetic_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- The given conditions
variable (a₂_value : ℤ := 11)
variable (S₁₀_value : ℤ := 40)

-- The general formula for the sequence given the conditions
theorem gen_formula_arithmetic_seq :
  ∃ a₁ d, (a₁ + d = a₂_value) ∧
          (10 * a₁ + (10 * 9 / 2) * d = S₁₀_value) ∧
          (∀ n, arithmetic_seq a₁ d n = -2 * n + 15) :=
sorry

-- The sum of absolute values of the first n terms of the sequence
def abs_arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ :=
  abs (arithmetic_seq a₁ d n)

def abs_arithmetic_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, abs_arithmetic_seq a₁ d (i + 1)

theorem sum_of_abs_arithmetic_seq (n : ℕ) :
  let T_n : ℤ :=
    if n ≤ 7 then -n^2 + 14 * n
    else n^2 - 14 * n + 98 in
  ∃ a₁ d, (a₁ + d = a₂_value) ∧
           (10 * a₁ + (10 * 9 / 2) * d = S₁₀_value) ∧
           (∀ n, arithmetic_seq a₁ d n = -2 * n + 15) ∧
           (abs_arithmetic_sum a₁ d n = T_n) :=
sorry

end gen_formula_arithmetic_seq_sum_of_abs_arithmetic_seq_l163_163477


namespace internet_service_cost_l163_163294

-- Define the conditions and question
def daily_cost (payment : ℝ) (days : ℕ) : ℝ := payment / days

-- The proof problem: Prove that the daily cost equals $0.28
theorem internet_service_cost : 
  ∀ (payment : ℝ) (days : ℕ), payment = 7 → days = 25 → daily_cost payment days = 7 / 25 :=
by
  intros payment days h_payment h_days
  rw [h_payment, h_days]
  simp
  sorry

end internet_service_cost_l163_163294


namespace f_g_2_plus_g_f_2_eq_4_l163_163819

def f (x : ℚ) : ℚ := (2 * x^2 + 4 * x + 7) / (x^2 - 2 * x + 5)
def g (x : ℚ) : ℚ := x - 2

theorem f_g_2_plus_g_f_2_eq_4 : f(g(2)) + g(f(2)) = 4 := by
  sorry

end f_g_2_plus_g_f_2_eq_4_l163_163819


namespace solve_for_y_l163_163851

theorem solve_for_y : ∃ y : ℤ, 3^(y - 2) = 9^(y + 2) ∧ y = -6 :=
by
  use -6
  sorry

end solve_for_y_l163_163851


namespace unique_solution_for_exponential_eq_l163_163341

theorem unique_solution_for_exponential_eq (a y : ℕ) (h_a : a ≥ 1) (h_y : y ≥ 1) :
  3^(2*a-1) + 3^a + 1 = 7^y ↔ (a = 1 ∧ y = 1) := by
  sorry

end unique_solution_for_exponential_eq_l163_163341


namespace twenty_five_percent_less_than_80_is_one_fourth_more_than_what_number_l163_163906

theorem twenty_five_percent_less_than_80_is_one_fourth_more_than_what_number : 
  ∃ x : ℕ, (80 - 0.25 * 80) = (x + 0.25 * x) ∧ x = 48 :=
by
  sorry

end twenty_five_percent_less_than_80_is_one_fourth_more_than_what_number_l163_163906


namespace equation_of_hyperbola_l163_163404

-- Definitions for the hyperbola and parabola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2 / a^2) - (y^2 / b^2) = 1

def parabola (k : ℝ) (x y : ℝ) : Prop := 
  x^2 = 4 * k * y

-- Problem conditions
variables (a b c : ℝ) (x y : ℝ)
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom hyperbola_eq : ∀ x y, hyperbola a b x y
axiom parabola_eq : ∀ x y, parabola (√6) x y
axiom B_focus : ∃ y, y = √6 ∧ ∀ x, parabola (√6) x y
axiom intersection_condition : ∀ m n, (m - 0 = 2 * (c - m)) ∧ (n - √6 = 2 * (0 - n)) → 
                                 (∃ m n, (m = 2 * c / 3) ∧ (n = √6 / 3))

-- Main statement
theorem equation_of_hyperbola : hyperbola 2 (√6) x y := 
by {
  sorry  -- Proof is omitted
}

end equation_of_hyperbola_l163_163404


namespace minimum_distance_between_curve_and_line_l163_163038

noncomputable def minimum_distance : ℝ :=
  let f := λ x : ℝ, x * Real.exp (-x)
  let g := λ x : ℝ, x + 3
  let dist := λ x1 y1 A B C: ℝ, (Real.abs (A * x1 + B * y1 + C) / Real.sqrt (A * A + B * B))
  Classinhat:
    dist 0 0 1 (-1) (-3) = (3 * Real.sqrt 2 / 2)

theorem minimum_distance_between_curve_and_line :
  ∃ P Q : ℝ × ℝ,
    (P.2 = P.1 * Real.exp (-P.1)) ∧
    (Q.2 = Q.1 + 3) ∧     
    Classinhat .dist P.1 P.2 1 (-1) (-3) = 3 * Real.sqrt 2 / 2 :=
begin
  sorry
end

end minimum_distance_between_curve_and_line_l163_163038


namespace not_perfect_square_l163_163888

noncomputable def sequence (n : ℕ) : ℕ
| 0 => 1
| 1 => 1
| (k + 2) => sequence k * sequence (k + 1) + 1

theorem not_perfect_square {n : ℕ} (h : n ≥ 2) : ¬ ∃ m : ℕ, sequence n = m * m :=
by
  sorry

end not_perfect_square_l163_163888


namespace mixture_solution_l163_163615

theorem mixture_solution (x y : ℝ) :
  (0.30 * x + 0.40 * y = 32) →
  (x + y = 100) →
  (x = 80) :=
by
  intros h₁ h₂
  sorry

end mixture_solution_l163_163615


namespace sequence_sum_square_l163_163160

theorem sequence_sum_square (n : ℕ) (hn : n > 0) : 
  (sum (range n) + sum (range (n-1)) + 1 = n * n) := 
  sorry

end sequence_sum_square_l163_163160


namespace solve_for_y_l163_163854

theorem solve_for_y (y : ℤ) (h : 3^(y-2) = 9^(y-1)) : y = 0 :=
by
  sorry

end solve_for_y_l163_163854


namespace barry_sotter_magic_trick_l163_163516

theorem barry_sotter_magic_trick (x : ℝ) (n : ℕ) 
  (h : n = 53) :
  x * ((4 / 3)^2 * (5 / 4) * ∏ k in (finset.range (n - 2)).map finset.succ, (k + 2) / (k + 1)) = 50 * x :=
by
  sorry

end barry_sotter_magic_trick_l163_163516


namespace minimum_value_of_f_l163_163711

noncomputable def min_f (f : ℝ → ℝ) (a b : ℝ) : ℝ := 
  Inf (set.range (λ x, f x))

def f (x : ℝ) : ℝ := Real.cos x + x * Real.sin x

theorem minimum_value_of_f : min_f f 0 (Real.pi / 2) = 1 := by
  sorry

end minimum_value_of_f_l163_163711


namespace bottle_caps_shared_l163_163509

theorem bottle_caps_shared (initial_bottle_caps : ℕ) (remaining_bottle_caps : ℕ) : 
  initial_bottle_caps = 51 → remaining_bottle_caps = 15 → initial_bottle_caps - remaining_bottle_caps = 36 :=
by
  intros h1 h2
  rw h1
  rw h2
  simp

end bottle_caps_shared_l163_163509


namespace areas_equal_l163_163114

-- Define the basic structure and properties of a cyclic trapezoid
structure CyclicTrapezoid (α : Type) [EuclideanPlane α] :=
  (A B C D L K : α)
  (AB BC CD DA : ℝ)
  (diagonal_AC diagonal_BD : ℝ)
  (is_cyclic : isCyclicQuadrilateral A B C D)
  (is_trapezoid : isTrapezoid A B C D)
  (extend_BD_to_L : dist D L = DA)
  (measure_AC_inward_to_K : dist C K = DA)

-- The theorem to prove
theorem areas_equal
  {α : Type} [EuclideanPlane α]
  (trap : CyclicTrapezoid α) :
  let a := trap.AB in
  let c := trap.CD in
  let ak := dist trap.A trap.K in
  let bl := dist trap.B trap.L in
  a * c = ak * bl :=
sorry

end areas_equal_l163_163114


namespace train_speed_is_60_kmph_l163_163960

noncomputable def speed_of_train_in_kmph (length_meters time_seconds : ℝ) : ℝ :=
  (length_meters / time_seconds) * 3.6

theorem train_speed_is_60_kmph (length_meters time_seconds : ℝ) :
  length_meters = 50 → time_seconds = 3 → speed_of_train_in_kmph length_meters time_seconds = 60 :=
by
  intros h_length h_time
  simp [speed_of_train_in_kmph, h_length, h_time]
  norm_num
  sorry

end train_speed_is_60_kmph_l163_163960


namespace sum_red_leq_sum_green_l163_163528

theorem sum_red_leq_sum_green (n : ℕ) (red green : fin n → ℕ) (h1 : ∀ i, 1 ≤ red i ∧ red i ≤ n)
  (h2 : ∀ i, 1 ≤ green i ∧ green i ≤ n)
  (h3 : ∀ i j, i ≠ j → red i ≠ red j)
  (h4 : ∀ i j, i ≠ j → green i ≠ green j)
  (h5 : ∀ i, contains (exit_point (red i)) grid_edges)
  (h6 : ∀ i, contains (exit_point (green i)) grid_edges) :
  ∑ i in finset.range n, red i ≤ ∑ i in finset.range n, green i := sorry

end sum_red_leq_sum_green_l163_163528


namespace integer_count_l163_163067

theorem integer_count (n : ℕ) :
  (n ≥ 2 ∧ (∃ (z : ℕ → ℂ), (∀ i, |z i| = 1) ∧ (∑ i in finset.range n, z i = 0)) ∧ (¬ nat.prime n → ¬ nat.composite n)) → n = 4 :=
sorry

end integer_count_l163_163067


namespace neg_p_sufficient_not_necessary_for_neg_q_l163_163721

variables (x : ℝ)

def p : Prop := |x + 1| > 2
def q : Prop := x > 2

theorem neg_p_sufficient_not_necessary_for_neg_q : (¬p -> ¬q) ∧ ¬(¬q -> ¬p) :=
by
  sorry

end neg_p_sufficient_not_necessary_for_neg_q_l163_163721


namespace find_conjugate_of_z_l163_163044

variable {ℂ : Type*} [Field ℂ] [IsComplex ℂ]
open Complex

theorem find_conjugate_of_z (z : ℂ) (h : (sqrt 2 + I) * z = 3 * I) : conj z = sqrt 2 - I :=
sorry

end find_conjugate_of_z_l163_163044


namespace paths_from_A_to_D_l163_163697

theorem paths_from_A_to_D : 
  let paths_A_B := 2 in
  let paths_B_C := 2 in
  let paths_C_D := 2 in
  let direct_path_A_D := 1 in
  paths_A_B * paths_B_C * paths_C_D + direct_path_A_D = 9 :=
by
  sorry

end paths_from_A_to_D_l163_163697


namespace probability_sum_seventeen_dodecahedral_dice_l163_163202

theorem probability_sum_seventeen_dodecahedral_dice : 
  let outcomes := (finset.fin_range 12).product (finset.fin_range 12) in
  let favorable_pairs := {(5, 12), (6, 11), (7, 10), (8, 9), (9, 8), (10, 7), (11, 6), (12, 5)}.to_finset in
  (favorable_pairs.card : ℚ) / outcomes.card = 1 / 18 :=
by
  let outcomes := (finset.fin_range 12 ⊗ finset.fin_range 12)
  let favorable_pairs := [(5, 12), (6, 11), (7, 10), (8, 9), (9, 8), (10, 7), (11, 6), (12, 5)].to_finset in
  have h1 : outcomes.card = 144 := by simp
  have h2 : favorable_pairs.card = 8 := by finset.card
  calc
    (8 : ℚ) / 144 = 1 / 18 : by norm_num

end probability_sum_seventeen_dodecahedral_dice_l163_163202


namespace total_geese_l163_163277

/-- Definition of the number of geese that remain flying after each lake, 
    based on the given conditions. -/
def geese_after_lake (G : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then G else 2^(n : ℕ) - 1

/-- Main theorem stating the total number of geese in the flock. -/
theorem total_geese (n : ℕ) : ∃ (G : ℕ), geese_after_lake G n = 2^n - 1 :=
by
  sorry

end total_geese_l163_163277


namespace train_speed_l163_163965

theorem train_speed (length_of_train time_to_cross : ℕ) (h_length : length_of_train = 50) (h_time : time_to_cross = 3) : 
  (length_of_train / time_to_cross : ℝ) * 3.6 = 60 := by
  sorry

end train_speed_l163_163965


namespace election_winner_margin_l163_163786

theorem election_winner_margin (T : ℕ) (V : ℕ) (L : ℕ) 
  (h1 : V = 720) 
  (h2 : (60 : ℚ) / 100 * T = V) 
  (h3 : L = (40 : ℚ) / 100 * T) : V - L = 240 :=
by
  have hT : T = 1200 := by
    sorry
  have hL : L = 480 := by
    sorry
  rw [h1, hL]
  rfl

end election_winner_margin_l163_163786


namespace max_value_fraction_l163_163125

variable {A B C : ℝ} {a b c : ℝ}

theorem max_value_fraction (h : sin A = sin B * sin C) :
  ∃ M, ∀ x y z, (sin x = sin B * sin C) → ((sin y / sin z) + (sin z / sin y)) ≤ M :=
begin
  sorry,
end

end max_value_fraction_l163_163125


namespace A1_lies_on_nine_point_circle_l163_163460

open EuclideanGeometry

noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def reflection (P O : Point) : Point := sorry
noncomputable def foot_of_altitude (A B C : Point) : Point := sorry
noncomputable def lies_on_circle (P O R : Point) : Prop := sorry

theorem A1_lies_on_nine_point_circle (A B C : Point) :
  let H := orthocenter A B C
  let O := circumcenter A B C
  let H' := reflection H O
  let A1 := foot_of_altitude A B C
  let nine_point_circle := ninePointCircle A B C
  H' ∈ line_through B C →
  lies_on_circle A1 nine_point_circle :=
by
  intros H H' O A1 nine_point_circle hH'_on_BC
  sorry

end A1_lies_on_nine_point_circle_l163_163460


namespace center_of_circle_bisector_l163_163908

open Locale Classical
noncomputable theory

-- Define the geometric objects and their properties
variables (O A B X Y : Point) -- Points as variables
variables (angle_bisector : Line) -- A line representing the angle bisector
variables (angle O X Y : Angle) -- Angle OXY
variables (circle_center : Point) -- Center of the circle
variables (circle_radius : ℝ) -- Radius of the circle
variables (circle : Circle) -- Circle through A and B

-- Conditions
axiom A_in_XOY : is_in_angle O X Y A -- Point A is inside angle XOY
axiom B_in_XOY : is_in_angle O X Y B -- Point B is inside angle XOY
axiom passes_through : passes_through circle A ∧ passes_through circle B -- Circle passes through A and B
axiom equal_interceptions : intercepts_equal_segments circle XOY -- Circle intercepts equal segments on OX and OY

-- Statement
theorem center_of_circle_bisector :
  is_bisector line angle_bisector angle ∧
  is_perpendicular_bisector line (segment A B) = circle_center →
  passes_through (Circle.mk circle_center circle_radius) A ∧ passes_through (Circle.mk circle_center circle_radius) B :=
sorry -- Proof omitted

end center_of_circle_bisector_l163_163908


namespace all_conditions_true_l163_163392

variables (α : ℝ)

-- Define each trigonometric identity condition
def condition1 : Prop := sin (π + α) = - sin α
def condition2 : Prop := cos (π / 2 + α) = - sin α
def condition3 : Prop := tan (π - α) = - tan α

-- The theorem to state all conditions are true
theorem all_conditions_true : condition1 α ∧ condition2 α ∧ condition3 α :=
by
  -- Proof can be filled in here.
  sorry

end all_conditions_true_l163_163392


namespace passenger_route_optimization_l163_163581

theorem passenger_route_optimization (n : ℕ) (h_n : n = 11) 
  (capital_travel_time : ℕ) (h_capital_travel_time : capital_travel_time = 7)
  (cyclic_road_time : ℕ) (h_cyclic_road_time : cyclic_road_time = 3)
  (initial_transfer_time final_transfer_time : ℝ) 
  (h_initial_transfer_time : initial_transfer_time = 2) 
  (h_final_transfer_time : final_transfer_time = 1.5)
  (t_initial : ℝ) (h_t_initial : t_initial = min (14 + initial_transfer_time) (15 + 4 * initial_transfer_time)) :
  min (14 + final_transfer_time) (15 + 4 * final_transfer_time) ≤ t_initial := 
by
  rw [h_initial_transfer_time, h_final_transfer_time, h_t_initial]
  norm_num
  -- Calculation shows that the minimal time considering the final transfer time is 15.5,
  -- and this is already less than the minimal time when initial transfer time is used,
  -- showing that the passenger will not find a faster route.
  sorry

end passenger_route_optimization_l163_163581


namespace domain_of_function_l163_163876

noncomputable def domain_f : Set ℝ := {x : ℝ | x > 2}

theorem domain_of_function : ∀ x, (x ∈ domain_f ↔ (∃ z, f(x) = 1/z) ) :=
by
  sorry

end domain_of_function_l163_163876


namespace committee_selection_l163_163951

/-- There are exactly 20 ways to select a three-person team for the welcoming committee. -/
axiom welcoming_committee_ways : 20

/-- From those selected for the welcoming committee, a two-person finance committee must be selected.
  The student council has 15 members in total. Prove that the number of different ways to select
  the four-person planning committee and the two-person finance committee is 4095.
-/
theorem committee_selection (h1 : nat.choose 15 3 = 20) : 
  (nat.choose 15 4) * (nat.choose 3 2) = 4095 :=
by
  have h2 : nat.choose 15 4 = 1365 := by sorry
  have h3 : nat.choose 3 2 = 3 := by sorry
  have h4 : 1365 * 3 = 4095 := by sorry
  rw [h2, h3]
  exact h4

end committee_selection_l163_163951


namespace braden_total_amount_l163_163982

variable initial_amount : ℕ
variable condition1 : 0 < initial_amount  -- Initial amount should be positive
def total_after_bets (initial_amount : ℕ) : ℕ :=
  initial_amount + 4 * initial_amount

theorem braden_total_amount (h : initial_amount = 400) : total_after_bets initial_amount = 2000 :=
by
  unfold total_after_bets
  rw [h]
  norm_num
  sorry

end braden_total_amount_l163_163982


namespace coeff_sum_exclude_constant_is_671_l163_163872

noncomputable def coefficient_sum_excluding_constant (x : ℝ) : ℝ :=
  (x^-1 - 2 * x^2)^9

theorem coeff_sum_exclude_constant_is_671 : coefficient_sum_excluding_constant 1 - (-672) = 671 := by
  sorry

end coeff_sum_exclude_constant_is_671_l163_163872


namespace sum_of_first_20_terms_l163_163390

theorem sum_of_first_20_terms (a : ℕ → ℤ) (h_rec : ∀ n, a (n + 1) = a n - 9) 
  (h_cond : a 3 + a 18 = 9) : ∑ i in finset.range 20, a i = 90 :=
by
  sorry

end sum_of_first_20_terms_l163_163390


namespace pascal_sum_difference_l163_163537

def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then nat.choose n k else 0

theorem pascal_sum_difference :
  (∑ i in finset.range (3006 + 1), (binomial 3006 i) / (binomial 3007 i)) -
  (∑ i in finset.range (3005 + 1), (binomial 3005 i) / (binomial 3006 i)) = 1 / 2 :=
by
  sorry

end pascal_sum_difference_l163_163537


namespace boxes_of_apples_l163_163223

theorem boxes_of_apples (apples_per_crate crates_delivered rotten_apples apples_per_box : ℕ) 
       (h1 : apples_per_crate = 42) 
       (h2 : crates_delivered = 12) 
       (h3 : rotten_apples = 4) 
       (h4 : apples_per_box = 10) : 
       crates_delivered * apples_per_crate - rotten_apples = 500 ∧
       (crates_delivered * apples_per_crate - rotten_apples) / apples_per_box = 50 := by
  sorry

end boxes_of_apples_l163_163223


namespace soup_problem_l163_163933

def cans_needed_for_children (children : ℕ) (children_per_can : ℕ) : ℕ :=
  children / children_per_can

def remaining_cans (initial_cans used_cans : ℕ) : ℕ :=
  initial_cans - used_cans

def half_cans (cans : ℕ) : ℕ :=
  cans / 2

def adults_fed (cans : ℕ) (adults_per_can : ℕ) : ℕ :=
  cans * adults_per_can

theorem soup_problem
  (initial_cans : ℕ)
  (children_fed : ℕ)
  (children_per_can : ℕ)
  (adults_per_can : ℕ)
  (reserved_fraction : ℕ)
  (hreserved : reserved_fraction = 2)
  (hintial : initial_cans = 8)
  (hchildren : children_fed = 24)
  (hchildren_per_can : children_per_can = 6)
  (hadults_per_can : adults_per_can = 4) :
  adults_fed (half_cans (remaining_cans initial_cans (cans_needed_for_children children_fed children_per_can))) adults_per_can = 8 :=
by
  sorry

end soup_problem_l163_163933


namespace range_of_slope_l163_163889

def f (x : ℝ) : ℝ :=
  x + Real.cos x - Real.sqrt 3 * Real.sin x

theorem range_of_slope :
  ∃ k_min k_max : ℝ, (∀ x : ℝ, k_min ≤ (1 - Real.sin x - Real.sqrt 3 * Real.cos x) ∧ (1 - Real.sin x - Real.sqrt 3 * Real.cos x) ≤ k_max) ∧
  k_min = -1 ∧ k_max = 3 :=
by
  sorry

end range_of_slope_l163_163889


namespace find_number_l163_163625

theorem find_number (x : ℝ) (h : 0.40 * x - 11 = 23) : x = 85 :=
sorry

end find_number_l163_163625


namespace petya_friends_count_l163_163840

theorem petya_friends_count 
  (classmates : Finset Person)
  (h_classmates_size : classmates.card = 28)
  (h_unique_friend_pairs : ∀ (p1 p2 : Person), p1 ≠ p2 → (number_of_friends p1) ≠ (number_of_friends p2)) :
  (number_of_friends Petya) = 14 :=
sorry

end petya_friends_count_l163_163840


namespace triangle_area_ratio_l163_163795

noncomputable def triangle_ratio (ABC DEF : ℝ) :=
  let AB := ABC
  let AC := ABC
  let BC := ABC
  let BD := BC / 3 * 2
  let DC := BC / 3
  -- Additional assumptions and bisector properties implicitly follow
  DEF / ABC = sqrt 3 / 16

theorem triangle_area_ratio : ∀ (AB AC BC BD DC DEF ABC : ℝ),
  AB = AC ∧
  BD = 2 * DC ∧
  BD + DC = BC ∧
  DEF / ABC = sqrt 3 / 16

end triangle_area_ratio_l163_163795


namespace union_AB_eq_interval_l163_163032

open Set

def A : Set ℝ := { x | 2 ≤ x ∧ x < 4 }
def B : Set ℝ := { x | x ≥ 3 }

theorem union_AB_eq_interval : A ∪ B = { x | 2 ≤ x } := by
  sorry

end union_AB_eq_interval_l163_163032


namespace solve_for_y_l163_163850

theorem solve_for_y : ∃ y : ℤ, 3^(y - 2) = 9^(y + 2) ∧ y = -6 :=
by
  use -6
  sorry

end solve_for_y_l163_163850


namespace fraction_of_boxes_loaded_by_day_crew_l163_163657

theorem fraction_of_boxes_loaded_by_day_crew :
  let M := (3/5) * D
  let N := (3/4) * D
  let WM := (5/6) * W
  let WN := (3/4) * W
  let total_morning := M * WM
  let total_day := D * W
  let total_night := N * WN
  let total_boxes := total_morning + total_day + total_night
  (total_day / total_boxes) = (16 / 33) :=
by
  -- Definitions to be used in calculations as provided by conditions
  let M := (3/5) * D
  let N := (3/4) * D
  let WM := (5/6) * W
  let WN := (3/4) * W
  let total_morning := M * WM
  let total_day := D * W
  let total_night := N * WN
  let total_boxes := total_morning + total_day + total_night
  -- Assertion of the problem translated mathematically
  have : (total_day / total_boxes) = (16 / 33), sorry
  exact this

end fraction_of_boxes_loaded_by_day_crew_l163_163657


namespace factorization_a_minus_b_l163_163200

-- Define the problem in Lean 4
theorem factorization_a_minus_b (a b : ℤ) : 
  (∀ y : ℝ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b))
  → a - b = 7 := 
by 
  sorry

end factorization_a_minus_b_l163_163200


namespace rectangular_coordinates_transformation_l163_163284

-- Define the conditions
variables (x y z r θ φ : ℝ)

-- Assume initial rectangular coordinates and relations with spherical coordinates
def initial_rectangular_coordinates : Prop := 
  x = -3 ∧ y = 2 ∧ z = 5

def spherical_relation : Prop := 
  x = r * real.sin φ * real.cos θ ∧ 
  y = r * real.sin φ * real.sin θ ∧ 
  z = r * real.cos φ

-- Define the transformed coordinates
def transformed_coordinates (r θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (r * real.cos φ * real.cos θ, r * real.cos φ * real.sin θ, -r * real.sin φ)

theorem rectangular_coordinates_transformation : 
  initial_rectangular_coordinates x y z ∧ spherical_relation x y z r θ φ → 
  transformed_coordinates r θ (φ - real.pi / 2) = (-3, 2, -5) := 
by 
  sorry

end rectangular_coordinates_transformation_l163_163284


namespace smallest_possible_sum_of_primes_l163_163693

theorem smallest_possible_sum_of_primes : 
  ∃ (primes : list ℕ), 
    primes.length = 5 ∧ 
    (∀ p ∈ primes, prime p) ∧ 
    (count_digits 3 primes = 2) ∧ 
    (count_digits 7 primes = 2) ∧ 
    (count_digits 8 primes = 2) ∧ 
    ∀ d ∈ [1, 2, 4, 5, 6, 9], count_digits d primes = 1 ∧ 
    list.sum primes = 2063 :=
sorry

-- Helper function to count occurrences of a digit in the list of primes
def count_digits (d : ℕ) (primes : list ℕ) : ℕ :=
  (primes.join.to_string.filter (λ c, c = (d.to_char))).length

-- Helper function to convert a digit to its character representation
def ℕ.to_char (n : ℕ) : char :=
  char.of_nat (n + 48)

end smallest_possible_sum_of_primes_l163_163693


namespace circle_area_irrational_of_rational_radius_l163_163431

theorem circle_area_irrational_of_rational_radius (r : ℚ) : ¬ ∃ A : ℚ, A = π * (r:ℝ) * (r:ℝ) :=
by sorry

end circle_area_irrational_of_rational_radius_l163_163431


namespace integer_solutions_a2_integer_solutions_a20_l163_163420

theorem integer_solutions_a2 (x y : ℤ) : (|y - x| + |3 * x - 2 * y| ≤ 2) ↔ (13) := sorry

theorem integer_solutions_a20 (x y : ℤ) : (|y - x| + |3 * x - 2 * y| ≤ 20) ↔ (841) := sorry

end integer_solutions_a2_integer_solutions_a20_l163_163420


namespace sum_of_form_l163_163987

theorem sum_of_form :
  ∀ (n : ℕ),
  (∑ k in finset.range (n + 1), 2 * k^2 + 1) = (2 * n^3 + 4 * n^2 + 3 * n) / 3 :=
by
  sorry

end sum_of_form_l163_163987


namespace tan_A_is_1_l163_163797

-- Define the internal angles and given equation
variables {A B C : ℝ}
axiom angle_sum : A + B + C = π
axiom given_eq : (sqrt 3 * cos A + sin A) / (sqrt 3 * sin A - cos A) = tan (-7 * π / 12)

-- The target to prove: tan A = 1
theorem tan_A_is_1 (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) : tan A = 1 := 
by
  sorry

end tan_A_is_1_l163_163797


namespace tangents_sum_l163_163318

-- Define basic properties of the circle and tangents
variables {O A B C T1 T2 T3 : Point}
variables {omega : Circle}

-- Given conditions as Lean definitions
def circle_omega : omega.radius = 6 := sorry
def point_A_outside_circle : dist O A = 15 := sorry
def tangents_thru_A : tangent A T1 omega ∧ tangent A T2 omega := sorry
def line_BC_tangent_omega : tangent B C omega := sorry
def distance_BC : dist B C = 8 := sorry

-- Define the main result, AB + AC = 16
theorem tangents_sum : dist A B + dist A C = 16 :=
by {
  -- Sorry for placeholder, real proof needed
  sorry,
}

end tangents_sum_l163_163318


namespace storks_minus_birds_l163_163928

/-- Define the initial values --/
def s : ℕ := 6         -- Number of storks
def b1 : ℕ := 2        -- Initial number of birds
def b2 : ℕ := 3        -- Number of additional birds

/-- Calculate the total number of birds --/
def b : ℕ := b1 + b2   -- Total number of birds

/-- Prove the number of storks minus the number of birds --/
theorem storks_minus_birds : s - b = 1 :=
by sorry

end storks_minus_birds_l163_163928


namespace part1_calc_l163_163858

theorem part1_calc : sqrt 25 + abs (1 - sqrt 3) + sqrt 27 = 4 + 4 * sqrt 3 := 
by 
  sorry

end part1_calc_l163_163858


namespace werewolf_eats_per_week_l163_163647
-- First, we import the necessary libraries

-- We define the conditions using Lean definitions

-- The vampire drains 3 people a week
def vampire_drains_per_week : Nat := 3

-- The total population of the village
def village_population : Nat := 72

-- The number of weeks both can live off the population
def weeks : Nat := 9

-- Prove the number of people the werewolf eats per week (W) given the conditions
theorem werewolf_eats_per_week :
  ∃ W : Nat, vampire_drains_per_week * weeks + weeks * W = village_population ∧ W = 5 :=
by
  sorry

end werewolf_eats_per_week_l163_163647


namespace angle_relation_l163_163361

theorem angle_relation
  (n : ℕ) (h : n ≥ 2)
  (A B C : Type)
  [linear_order B]
  [linear_order C]
  [inner_product_space A B]
  [B.abs := ∠ B]
  [C.abs := ∠ C]
  (h1 : ∠B < 180) (h2 : ∠C < 180) (h3 : AB = n * AC):
  ∠ C > n * ∠ B :=
sorry

end angle_relation_l163_163361


namespace all_props_hold_l163_163403

-- Define the three functions
def f1 (x : ℝ) : ℝ := -x^2 + 2 * x

def f2 (x : ℝ) : ℝ := Real.cos (π / 2 - π * x / 2)

def f3 (x : ℝ) : ℝ := Real.sqrt (Real.abs (x - 1))

-- Define the proposition r: f(1/2) > 1/2
def prop_r (f : ℝ → ℝ) : Prop := f (1/2) > 1/2

-- Define the proposition s: The graph of f(x) is symmetric about the line x = 1
def symmetric_about_x1 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(1 - x) = f(1 + x)

-- State the theorem
theorem all_props_hold :
  (prop_r f1 ∧ prop_r f2 ∧ prop_r f3) ∧
  (symmetric_about_x1 f1 ∧ symmetric_about_x1 f2 ∧ symmetric_about_x1 f3) :=
by
  -- Proof to be provided
  sorry

end all_props_hold_l163_163403


namespace carousel_problem_l163_163931

theorem carousel_problem (n : ℕ) : 
  (∃ (f : Fin n → Fin n), 
    (∀ i, f (f i) = i) ∧ 
    (∀ i j, i ≠ j → f i ≠ f j) ∧ 
    (∀ i, f i < n)) ↔ 
  (Even n) := 
sorry

end carousel_problem_l163_163931


namespace hyperbola_foci_difference_l163_163033

noncomputable def foci_difference (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : ℝ :=
  -- Let F1 and F2 be the left and right foci of the hyperbola
  -- x^2/a^2 - y^2/b^2 = 1 with the given conditions.
  -- Assume a line perpendicular to the asymptote passes through F2
  -- and intersects the hyperbola at point P.

  -- We are to prove that 
  -- |PF1|^2 - |PF2|^2 = 4a^2
  let PF1_sq := a^2 + c^2 - 2 * a * c * (-a / c) in -- from cosine rule
  let PF2_sq := b^2 in                            -- from given condition
  PF1_sq - PF2_sq                                 -- required difference

theorem hyperbola_foci_difference (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ PF1 PF2 P : ℝ × ℝ,
  -- Assuming specific definitions based on given conditions
  let F1 : ℝ × ℝ := (-sqrt (a^2 + b^2), 0) in
  let F2 : ℝ × ℝ := (sqrt (a^2 + b^2), 0) in
  F2 = (sqrt (a^2 + b^2), 0) → 
  ∀ (P : ℝ × ℝ),
  P = (a, b) →   -- example points (P is the intersection with the line)
  let |PF1|^2 := (a + sqrt (a^2 + b^2))^2 + b^2 in
  let |PF2|^2 := (a - sqrt (a^2 + b^2))^2 + b^2 in
  |PF1|^2 - |PF2|^2 = 4 * a^2 := 
by
  sorry

end hyperbola_foci_difference_l163_163033


namespace min_b_minus_2c_over_a_l163_163414

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (h1 : a ≤ b + c ∧ b + c ≤ 3 * a)
variable (h2 : 3 * b^2 ≤ a * (a + c) ∧ a * (a + c) ≤ 5 * b^2)

theorem min_b_minus_2c_over_a : (∃ u : ℝ, (u = (b - 2 * c) / a) ∧ (∀ v : ℝ, (v = (b - 2 * c) / a) → u ≤ v)) :=
  sorry

end min_b_minus_2c_over_a_l163_163414


namespace sum_of_cubes_mod_9_l163_163692

theorem sum_of_cubes_mod_9 : 
  (∑ i in Finset.range 151, i^3) % 9 = 6 :=
by
  sorry

end sum_of_cubes_mod_9_l163_163692


namespace vector_CD_l163_163098

-- Definitions based on conditions in the problem
variables (A B C D : Type) [AddCommGroup A] [Module ℝ A]
variables (CB CA : A) (BD DA : A)
variables (h1 : BD = (1 / 2) • DA)
variables (h2 : CB = CB) (h3 : CA = CA)
variables (h4 : BD = BD)

-- The proof statement
theorem vector_CD (CB CA : A) (BD : A) (h1 : BD = (1 / 2) • DA) : (2 / 3) • CB + (1 / 3) • CA =
sorry

end vector_CD_l163_163098


namespace number_of_child_workers_l163_163270

-- Define the conditions
def number_of_male_workers : ℕ := 20
def number_of_female_workers : ℕ := 15
def wage_per_male : ℕ := 35
def wage_per_female : ℕ := 20
def wage_per_child : ℕ := 8
def average_wage : ℕ := 26

-- Define the proof goal
theorem number_of_child_workers (C : ℕ) : 
  ((number_of_male_workers * wage_per_male +
    number_of_female_workers * wage_per_female +
    C * wage_per_child) /
   (number_of_male_workers + number_of_female_workers + C) = average_wage) → 
  C = 5 :=
by 
  sorry

end number_of_child_workers_l163_163270


namespace sum_sequence_l163_163375

noncomputable def a : ℕ → ℕ 
| 1        := 1
| (2 * n)  := n - a n
| (2 * n + 1) := a n + 1

theorem sum_sequence (s : ℕ → ℕ) (h1 : ∀ n, s (2 * n) = n - s n) 
  (h2 : ∀ n, s (2 * n + 1) = s n + 1) : (∑ i in range 100, s (i + 1)) = 1306 :=
by {
  have h : s 1 = 1, from rfl,
  let a := s,
  -- The steps of the precise mathematical arguments leading to the conclusion go here
  sorry
}

end sum_sequence_l163_163375


namespace weight_of_new_person_l163_163258

variable (avg_increase : ℝ) (n_persons : ℕ) (weight_replaced : ℝ)

theorem weight_of_new_person (h1 : avg_increase = 3.5) (h2 : n_persons = 8) (h3 : weight_replaced = 65) :
  let total_weight_increase := n_persons * avg_increase
  let weight_new := weight_replaced + total_weight_increase
  weight_new = 93 := by
  sorry

end weight_of_new_person_l163_163258


namespace second_player_win_probability_l163_163837

theorem second_player_win_probability (s1 s2 : ℕ) (h1 : 2 ≤ s1 ∧ s1 ≤ 5) (h2 : 2 ≤ s2 ∧ s2 ≤ 5) :
  exists (s2_strategy : ℕ), (2 ≤ s2_strategy ∧ s2_strategy ≤ 5) ∧ winning_probability(s2_strategy) > 1 / 2 :=
by
  sorry

end second_player_win_probability_l163_163837


namespace upper_limit_of_multiples_of_10_l163_163912

theorem upper_limit_of_multiples_of_10 (n : ℕ) (hn : 10 * n = 100) (havg : (10 * n + 10) / (n + 1) = 55) : 10 * n = 100 :=
by
  sorry

end upper_limit_of_multiples_of_10_l163_163912


namespace sequence_periodic_l163_163640

theorem sequence_periodic (a : ℕ → ℕ) (h : ∀ n > 2, a (n + 1) = (a n ^ n + a (n - 1)) % 10) :
  ∃ n₀, ∀ k, a (n₀ + k) = a (n₀ + k + 4) :=
by {
  sorry
}

end sequence_periodic_l163_163640


namespace problem_1_simplify_problem_2_evaluate_l163_163986

-- Define variables and functions for the first problem
variable (α : ℝ)

-- Define the expression for the first problem
def expr1 := (sin (α + real.pi))^2 * cos (real.pi + α) * (cot (-α - 2 * real.pi)) / 
              (tan (real.pi + α) * (cos (-α - real.pi))^3)

-- State and prove the first theorem
theorem problem_1_simplify : expr1 α = -1 := by
  -- proof omitted
  sorry

-- Definitions and expressions for the second problem
def expr2 := (0.064 : ℝ) ^ (-1 / 3 : ℝ) + (-2 : ℝ) ^ (-3 * (4 / 3) : ℝ) + 
             16 ^ (-0.75 : ℝ) - log (sqrt 0.1) / log 10 - 
             (log 9 / log 2) * (log 2 / log 3)

-- State and prove the second theorem
theorem problem_2_evaluate : expr2 = 19 / 16 := by
  -- proof omitted
  sorry

end problem_1_simplify_problem_2_evaluate_l163_163986


namespace sequence_props_l163_163792

def sequence (n : ℕ) : ℕ → ℚ
| 1 := 1
| k + 2 := (sequence 1 + (List.range' 1 k).map (λ i, (1:ℚ)/(i+1) * sequence (i+1))).sum

theorem sequence_props :
  sequence 10 = 5 ∧
  (sequence 2 / sequence 4 = sequence 4 / sequence 8) ∧
  (∑ i in List.range n, sequence (i+1)) = (n^2 + n + 2) / 4 :=
by
  sorry

end sequence_props_l163_163792


namespace trajectory_of_P_l163_163023

theorem trajectory_of_P
  (P A B : ℝ → ℝ → Prop)
  (points_tangents : ∀ {h : ℝ}, P (h, 0) → P (0, h) → A (h, 0) → B (0, h))
  (circle : ∀ {x y : ℝ}, x^2 + y^2 = 1)
  (angle_right : ∀ a b c : ℝ, ∠ (a, b) (c, b) = 90) :
  ∃ x y : ℝ, (P x y) ∧ (x^2 + y^2 = 2) :=
by
  sorry 

end trajectory_of_P_l163_163023


namespace hilton_final_marbles_l163_163756

theorem hilton_final_marbles :
  let initial_marbles := 26
  let found_marbles := 6
  let lost_marbles := 10
  let gift_multiplication_factor := 2
  let marbles_after_find_and_lose := initial_marbles + found_marbles - lost_marbles
  let gift_marbles := gift_multiplication_factor * lost_marbles
  let final_marbles := marbles_after_find_and_lose + gift_marbles
  final_marbles = 42 :=
by
  -- Proof to be filled
  sorry

end hilton_final_marbles_l163_163756


namespace symmetric_line_x_axis_l163_163877

theorem symmetric_line_x_axis (x y : ℝ) : (2 * x - 3 * y + 2 = 0) → (2 * x + 3 * -y + 2 = 0) :=
begin
  sorry
end

end symmetric_line_x_axis_l163_163877


namespace max_terms_of_sequence_l163_163103

theorem max_terms_of_sequence
  (a : ℕ → ℝ)
  (h_cond1 : ∀ k, (a k + a (k + 1) + a (k + 2) + a (k + 3) + a (k + 4) + a (k + 5) + a (k + 6)) < 0)
  (h_cond2 : ∀ k, (a k + a (k + 1) + a (k + 2) + a (k + 3) + a (k + 4) + a (k + 5) + a (k + 6) + a (k + 7) + a (k + 8) + a (k + 9) + a (k + 10)) > 0) :
  ∃ (n : ℕ), n ≤ 16 ∧ ∀ m > 16, ¬ (∃ (b : ℕ → ℝ), (∀ k, (b k + b (k + 1) + b (k + 2) + b (k + 3) + b (k + 4) + b (k + 5) + b (k + 6)) < 0) ∧ (∀ k, (b k + b (k + 1) + b (k + 2) + b (k + 3) + b (k + 4) + b (k + 5) + b (k + 6) + b (k + 7) + b (k + 8) + b (k + 9) + b (k + 10)) > 0) ∧ (m = b.length))
| sorry

end max_terms_of_sequence_l163_163103


namespace problem_statement_l163_163399

section
variables (a x λ x₁ x₂ : ℝ)

def f (x a : ℝ) : ℝ := (1/2) * x^2 - (2*a + 2) * x + (2*a + 1) * ln x
def g (x a λ : ℝ) : ℝ := f x a - λ / x

noncomputable def h (x λ : ℝ) : ℝ := x^3 - 6*x^2 + 5*x + λ

theorem problem_statement 
  (a_in : a ∈ set.Icc (1/2 : ℝ) 2) 
  (x1_in : x₁ ∈ set.Icc 1 2) 
  (x2_in : x₂ ∈ set.Icc 1 2) 
  (x1_ne_x2 : x₁ ≠ x₂) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → g' x a λ ≥ 0) →
  (h 2 λ ≥ 0) → 
  |f x₁ a - f x₂ a| < λ * |(1 / x₁) - (1 / x₂) :=
sorry

end problem_statement_l163_163399


namespace lcm_gcd_pairs_l163_163259

theorem lcm_gcd_pairs (a b : ℕ) :
  (lcm a b + gcd a b = (a * b) / 5) ↔
  (a = 10 ∧ b = 10) ∨ (a = 6 ∧ b = 30) ∨ (a = 30 ∧ b = 6) :=
sorry

end lcm_gcd_pairs_l163_163259


namespace machine_A_produces_40_percent_l163_163939

theorem machine_A_produces_40_percent (p : ℝ) : 
  (0 < p ∧ p < 1 ∧
  (0.0156 = p * 0.009 + (1 - p) * 0.02)) → 
  p = 0.4 :=
by 
  intro h
  sorry

end machine_A_produces_40_percent_l163_163939


namespace solve_for_x_l163_163398

noncomputable def f : ℝ → ℝ
| x := if x ≤ 1 then 3^x else -x

theorem solve_for_x : ∃ x : ℝ, f x = 2 ↔ x = Real.log 2 / Real.log 3 := by
  sorry

end solve_for_x_l163_163398


namespace find_b_value_l163_163305

theorem find_b_value (a c d : ℝ) : (b : ℝ) → (y = a * cos (b * x + c ) + d) ∧ (2 * π / b = 3 * π / 2) → (b = 4 / 3) :=
by 
  intros b h_value h_period,
  sorry

end find_b_value_l163_163305


namespace choose_three_objects_l163_163547

theorem choose_three_objects (n objects : ℕ) (choose : ℕ → ℕ → ℕ)
  (adjacent_pairs : ℕ) (diametrically_opposite_pairs : ℕ) (adjacent_triplets : ℕ) :
  n = 28 →
  objects = 3 →
  choose 28 3 - (28 * (28 - 4)) - (14 * (28 - 6)) - 28 = 2268 :=
by
  intros n_eq objects_eq
  have choose_eq : choose 28 3 = 3276 := sorry,
  have adj_cases : 28 * 24 = 672 := sorry,
  have diam_cases : 14 * 22 = 308 := sorry,
  have triplet_cases : 28 = 28 := rfl,
  have combined : 3276 - 672 - 308 - 28 = 2268 := sorry,
  exact combined

end choose_three_objects_l163_163547


namespace triangle_is_isosceles_right_l163_163777

theorem triangle_is_isosceles_right (a b c : ℝ) (A B C : ℝ) (h1 : b = a * Real.sin C) (h2 : c = a * Real.cos B) : 
  A = π / 2 ∧ b = c := 
sorry

end triangle_is_isosceles_right_l163_163777


namespace min_translation_value_l163_163880

theorem min_translation_value (m : ℝ) (h : m > 0) : 
  (∀ x, f (x + m) = f (-(x + m))) → m = 3 * (Real.pi / 8) :=
by
  have f : ℝ → ℝ := λ x, cos (2 * x) + sin (2 * x)
  sorry

end min_translation_value_l163_163880


namespace find_t_range_l163_163063

variable {t : ℝ}

def a : ℝ × ℝ := (-2, -1)
def b : ℝ × ℝ := (t, 1)

-- Condition for dot product to be less than zero for obtuse angle
def dot_product_is_obtuse (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 < 0

-- Condition for vectors not being antiparallel
def not_antiparallel (a b : ℝ × ℝ) : Prop :=
  ¬(∃ λ : ℝ, λ < 0 ∧ a = (λ * b.1, λ * b.2))

theorem find_t_range (h1 : a = (-2, -1)) (h2 : b = (t, 1)) (h3 : dot_product_is_obtuse a b) (h4 : not_antiparallel a b) :
  t ∈ ((-1 / 2 : ℝ), 2) ∪ (2, +∞ : Set ℝ) :=
by
  sorry

end find_t_range_l163_163063


namespace max_profit_l163_163937

def maximizeProfit (x y : ℝ) : ℝ := 0.4 * x + 0.6 * y

theorem max_profit :
  ∃ (x y : ℝ), x + y = 600000 ∧ x ≥ 2/3 * y ∧ x ≥ 50000 ∧ y ≥ 50000 ∧ maximizeProfit x y = 312000 :=
by
  use [240000, 360000]
  split
  { sorry }

end max_profit_l163_163937


namespace volume_of_soil_extracted_l163_163787

-- Definition of the conditions
def Length : ℝ := 20
def Width : ℝ := 10
def Depth : ℝ := 8

-- Statement of the proof problem
theorem volume_of_soil_extracted : Length * Width * Depth = 1600 := by
  -- Proof skipped
  sorry

end volume_of_soil_extracted_l163_163787


namespace problem1_problem2_l163_163738

theorem problem1 (m : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = m - |x - 2|) 
  (h2 : ∀ x, f (x + 2) ≥ 0 → -1 ≤ x ∧ x ≤ 1) : 
  m = 1 := 
sorry

theorem problem2 (a b c : ℝ) 
  (h : 1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) : 
  a + 2 * b + 3 * c ≥ 9 := 
sorry

end problem1_problem2_l163_163738


namespace rectangle_area_increase_l163_163549

   theorem rectangle_area_increase (l w : ℝ) : 
     let original_area := l * w
     let new_length := 1.3 * l
     let new_width := 1.2 * w
     let new_area := new_length * new_width
     let percentage_increase := 100 * (new_area / original_area - 1)
     in percentage_increase = 56 :=
   by
     sorry
   
end rectangle_area_increase_l163_163549


namespace find_min_max_l163_163710

noncomputable def f (x y : ℝ) : ℝ := Real.sin x + Real.sin y - Real.sin (x + y)

theorem find_min_max :
  (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x + y ≤ 2 * Real.pi → 
    (0 ≤ f x y ∧ f x y ≤ 3 * Real.sqrt 3 / 2)) :=
sorry

end find_min_max_l163_163710


namespace problem_a_problem_b_l163_163255

open_locale real

section ProofProblem

variables {α : Type*} [metric_space α] [normed_group α] [normed_space ℝ α]

-- Geometry problem definitions:
variable {P A A_1 B C R : α}
variable {d : ℝ}
variable (circ : ∀ {X Y Z : α}, dist P X = R ∧ dist P Y = R ∧ dist P Z = R)
variable (perp_P_A1: ∀ {X : α}, dist P A_1 = dist X A_1)
variable (perp_P_B1: ∀ {X : α}, dist P B_1 = dist X B_1)
variable (dist_PA1_d : ∀ {X}, dist A_1 B_1 = d)

-- Question (a) theorem statement in Lean:
theorem problem_a (h1: ∀ {X}, dist P A = 2 * R * real.sin ℝ(d))
  (h2: ∀ {X}, dist P A_1 = d / real.sin ℝ(d)) :
  dist P A * dist P A_1 = 2 * R * d :=
sorry

-- Assuming the angle α definitions
variable (α : angle α) (cos_α_P : cos α = dist P (α: ℝ) / (2 * R))

-- Question (b) theorem statement in Lean:
theorem problem_b :
  cos α = (dist P A) / (2 * R) :=
sorry

end ProofProblem

end problem_a_problem_b_l163_163255


namespace prob_first_diamond_second_ace_or_face_l163_163706

theorem prob_first_diamond_second_ace_or_face :
  let deck_size := 52
  let first_card_diamonds := 13 / deck_size
  let prob_ace_after_diamond := 4 / (deck_size - 1)
  let prob_face_after_diamond := 12 / (deck_size - 1)
  first_card_diamonds * (prob_ace_after_diamond + prob_face_after_diamond) = 68 / 867 :=
by
  let deck_size := 52
  let first_card_diamonds := 13 / deck_size
  let prob_ace_after_diamond := 4 / (deck_size - 1)
  let prob_face_after_diamond := 12 / (deck_size - 1)
  sorry

end prob_first_diamond_second_ace_or_face_l163_163706


namespace ratio_is_one_half_l163_163803

noncomputable def ratio_dresses_with_pockets (D : ℕ) (total_pockets : ℕ) (pockets_two : ℕ) (pockets_three : ℕ) :=
  ∃ (P : ℕ), D = 24 ∧ total_pockets = 32 ∧
  (P / 3) * 2 + (2 * P / 3) * 3 = total_pockets ∧ 
  P / D = 1 / 2

theorem ratio_is_one_half :
  ratio_dresses_with_pockets 24 32 2 3 :=
by 
  sorry

end ratio_is_one_half_l163_163803


namespace value_of_f_at_3_l163_163136

variable {a b c : ℝ}
def f (x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x

theorem value_of_f_at_3 (h : f (-3) = 7) : f (3) = -7 :=
by
  sorry

end value_of_f_at_3_l163_163136


namespace sum_of_max_min_values_l163_163430

variable {n : ℕ}

-- Definition of the product of the first n terms of the sequence.
def T (n : ℕ) : ℝ := 1 - (2 / 15) * n

-- Definition of the sequence a_n.
def a (n : ℕ) : ℝ := 
if h : n > 0 then (T n) / (T (n - 1)) else T 1

theorem sum_of_max_min_values : 
  let max_a : ℝ := 3 in
  let min_a : ℝ := -1 in
  max_a + min_a = 2 :=
by
  sorry

end sum_of_max_min_values_l163_163430


namespace total_dolphins_correct_l163_163093

-- Define the initial number of dolphins
def initialDolphins : Nat := 65

-- Define the multiplier for the dolphins joining from the river
def joiningMultiplier : Nat := 3

-- Define the total number of dolphins after joining
def totalDolphins : Nat := initialDolphins + (joiningMultiplier * initialDolphins)

-- Prove that the total number of dolphins is 260
theorem total_dolphins_correct : totalDolphins = 260 := by
  sorry

end total_dolphins_correct_l163_163093


namespace infinite_coprime_pairs_l163_163841

theorem infinite_coprime_pairs (m : ℕ) (h : 0 < m) :
  ∀ N : ℕ, ∃ x y : ℤ, nat.gcd x y = 1 ∧ x ∣ (y^2 + m) ∧ y ∣ (x^2 + m) ∧ N < x ∧ N < y :=
sorry

end infinite_coprime_pairs_l163_163841


namespace limit_ln4_minus_1_l163_163177

noncomputable def limit_sequence (n : ℕ) : ℝ :=
  (1 / n : ℝ) * ∑ i in Finset.range (n + 1) \ {0}, 
    (Real.floor (2 * n / i) - 2 * Real.floor (n / i))

theorem limit_ln4_minus_1 :
  tendsto limit_sequence at_top (𝓝 (Real.log 4 - 1)) :=
by sorry

end limit_ln4_minus_1_l163_163177


namespace volume_of_tetrahedron_abcd_l163_163526

theorem volume_of_tetrahedron_abcd
  (P Q R S A B C D : Type)
  [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited S]
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (h1 : distance P Q = 7)
  (h2 : distance R S = 7)
  (h3 : distance P R = 8)
  (h4 : distance Q S = 8)
  (h5 : distance P S = 9)
  (h6 : distance Q R = 9)
  (inscribed_center : ∀ (X Y Z : Type), (distance X Y) * (distance Y Z) * (distance Z X) ≠ 0 → Type)
  (inscribed_center_PQS : inscribed_center P Q S = A)
  (inscribed_center_PRS : inscribed_center P R S = B)
  (inscribed_center_QRS : inscribed_center Q R S = C)
  (inscribed_center_PQR : inscribed_center P Q R = D) :
  volume A B C D = (5 * Real.sqrt 11) / 9 := sorry

end volume_of_tetrahedron_abcd_l163_163526


namespace concurrency_BD_FX_ME_l163_163894

variables {P : Type} [euclidean_geometry P]

-- Define points and collinear function
variables (B C F A D E M X : P)
variables (lBD lFX lME : set P)

noncomputable
def is_right_triangle (B C F : P) : Prop :=
∃ triangle : triangle P, triangle.is_right_triangle_at B C F

noncomputable
def is_midpoint (M : P) (C F : P) : Prop :=
M = midpoint C F

noncomputable
def is_bisector (AC : line P) (angle_dab : angle) (D B A : P) : Prop :=
AC.is_angle_bisector (angle_dab D B A)

noncomputable
def are_concurrent (l1 l2 l3 : set P) : Prop :=
∃ P : P, P ∈ l1 ∧ P ∈ l2 ∧ P ∈ l3

-- Given conditions
axiom triangle_properties :
  is_right_triangle B C F ∧ (∃ fa fb : P, A ∈ line.of_points F C ∧ F ∈ segment A C ∧ dist F A = dist F B)
  ∧ (DA = DC ∧ is_bisector (line.of_points A C) (angle_bisector (D A B)) D A)
  ∧ (EA = ED ∧ is_bisector (line.of_points A D) (angle_bisector (E A C)) E A)
  ∧ (is_midpoint M C F)
  ∧ (is_parallelogram A M X E)

-- Proof statement
theorem concurrency_BD_FX_ME :
  are_concurrent (line.of_points B D) (line.of_points F X) (line.of_points M E) :=
sorry

end concurrency_BD_FX_ME_l163_163894


namespace bruce_total_payment_l163_163918

theorem bruce_total_payment :
  let grapes_kg := 8
  let grapes_rate := 70
  let mangoes_kg := 9
  let mangoes_rate := 55
  let cost_grapes := grapes_kg * grapes_rate
  let cost_mangoes := mangoes_kg * mangoes_rate
  let total_payment := cost_grapes + cost_mangoes
  total_payment = 1055 := by
    let grapes_kg := 8
    let grapes_rate := 70
    let mangoes_kg := 9
    let mangoes_rate := 55
    let cost_grapes := grapes_kg * grapes_rate
    let cost_mangoes := mangoes_kg * mangoes_rate
    let total_payment := cost_grapes + cost_mangoes
    have h1 : cost_grapes = 560 := rfl
    have h2 : cost_mangoes = 495 := rfl
    have h3 : total_payment = 1055 := rfl
    exact h3

end bruce_total_payment_l163_163918


namespace total_pictures_on_wall_l163_163514

theorem total_pictures_on_wall (oil_paintings watercolor_paintings : ℕ) (h1 : oil_paintings = 9) (h2 : watercolor_paintings = 7) :
  oil_paintings + watercolor_paintings = 16 := 
by
  sorry

end total_pictures_on_wall_l163_163514


namespace points_comparison_l163_163056

def quadratic_function (m x : ℝ) : ℝ :=
  (x + m - 3) * (x - m) + 3

def point_on_graph (m x y : ℝ) : Prop :=
  y = quadratic_function m x

theorem points_comparison (m x1 x2 y1 y2 : ℝ)
  (h1 : point_on_graph m x1 y1)
  (h2 : point_on_graph m x2 y2)
  (hx : x1 < x2)
  (h_sum : x1 + x2 < 3) :
  y1 > y2 := 
  sorry

end points_comparison_l163_163056


namespace minimum_ratio_cone_cylinder_l163_163911

theorem minimum_ratio_cone_cylinder (r : ℝ) (h : ℝ) (a : ℝ) :
  (h = 4 * r) →
  (a^2 = r^2 * h^2 / (h - 2 * r)) →
  (∀ h > 0, ∃ V_cone V_cylinder, 
    V_cone = (1/3) * π * a^2 * h ∧ 
    V_cylinder = π * r^2 * (2 * r) ∧ 
    V_cone / V_cylinder = (4 / 3)) := 
sorry

end minimum_ratio_cone_cylinder_l163_163911


namespace kolya_triangles_l163_163970

theorem kolya_triangles (ABC : Triangle) (fold_line : Line) (cut_line : Line) :
  (∀ part : Triangle, part ∈ (cut (fold ABC fold_line) cut_line) → equal_non_isosceles part) →
  (exists three_parts : Finset Triangle, (three_parts.card = 3) ∧ (∀ tri ∈ three_parts, is_non_isosceles tri ∧ equal tri three_parts)) :=
sorry

end kolya_triangles_l163_163970


namespace cosine_of_multiple_angles_rational_l163_163077

theorem cosine_of_multiple_angles_rational (θ : ℚ) (k : ℕ) (h_cos_kθ : ∃ r : ℚ, r = ℚ.cos (k * θ)) :
  ∃ n : ℕ, n > k ∧ ∃ r1 r2 : ℚ, r1 = ℚ.cos ((n-1) * θ) ∧ r2 = ℚ.cos (n * θ) := 
by
  -- skipping the proof
  sorry

end cosine_of_multiple_angles_rational_l163_163077


namespace simplify_expression_1_simplify_expression_2_l163_163624

-- Problem 1
theorem simplify_expression_1 (a b : ℤ) : a + 2 * b + 3 * a - 2 * b = 4 * a :=
by
  sorry

-- Problem 2
theorem simplify_expression_2 (m n : ℤ) (h_m : m = 2) (h_n : n = 1) :
  (2 * m ^ 2 - 3 * m * n + 8) - (5 * m * n - 4 * m ^ 2 + 8) = 8 :=
by
  sorry

end simplify_expression_1_simplify_expression_2_l163_163624


namespace number_of_tagged_fish_l163_163780

-- Define variables and conditions
variables (N T : ℕ)
variable h1 : 10 / 50 = 2 / 10
variable h2 : N = 250

-- Statement of the problem
theorem number_of_tagged_fish (h1 : 10 / 50 = 2 / 10) (h2 : N = 250) : 
  (T : ℕ) = 50 :=
sorry 

end number_of_tagged_fish_l163_163780


namespace circles_condition_l163_163592

noncomputable def circles_intersect_at (p1 p2 : ℝ × ℝ) (m c : ℝ) : Prop :=
  p1 = (1, 3) ∧ p2 = (m, 1) ∧ (∃ (x y : ℝ), (x - y + c / 2 = 0) ∧ 
    (p1.1 - x)^2 + (p1.2 - y)^2 = (p2.1 - x)^2 + (p2.2 - y)^2)

theorem circles_condition (m c : ℝ) (h : circles_intersect_at (1, 3) (m, 1) m c) : m + c = 3 :=
sorry

end circles_condition_l163_163592


namespace domain_of_function_l163_163195

theorem domain_of_function : 
  (∃ f : ℝ → ℝ, (∀ x, x > 1 → f x = 1 / real.sqrt (real.logb 2 (2*x - 1))) ∧ (∀ x, x ≤ 1 → f x = 0)) :=
sorry

end domain_of_function_l163_163195


namespace square_O1M1O2M2_l163_163162

-- Define structures of points and triangle
structure Point := (x : ℝ) (y : ℝ)
structure Triangle := (A B C : Point)
structure Square := (A B C D : Point) (center : Point)

-- Define functions to get midpoints of segments
def midpoint (P Q : Point) : Point := 
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

-- Define the problem in Lean 4
theorem square_O1M1O2M2 (A B C D L K : Point) (O1 O2 : Point)
  (M1 M2 : Point) (sq1 : Square) (sq2 : Square) (T : Triangle) : 
  sq1 = Square.mk T.A T.B D sq1.center O1 ∧
  sq2 = Square.mk T.B T.C K sq2.center O2 ∧
  M1 = midpoint D L ∧
  M2 = midpoint T.A T.C →
  -- Conclusion to prove
  let O1M1O2M2 := (O1, M1, O2, M2) in
  -- We need to prove this forms a square
  ((O1.x - M1.x)^2 + (O1.y - M1.y)^2 = (M2.x - O2.x)^2 + (M2.y - O2.y)^2) ∧
  ((O1.x - O2.x)^2 + (O1.y - O2.y)^2 = (M1.x - M2.x)^2 + (M1.y - M2.y)^2) ∧
  ((O1.x - M2.x)^2 + (O1.y - M2.y)^2 = (M1.x - O2.x)^2 + (M1.y - O2.y)^2) ∧
  ((O1.x - M1.x) * (M2.x - O2.x) + (O1.y - M1.y) * (M2.y - O2.y) = 0) :=
sorry

end square_O1M1O2M2_l163_163162


namespace parametric_curve_general_line_max_PA_min_PA_l163_163045

noncomputable theory

-- Given conditions
def curve (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 9) = 1
def line_parametric (t : ℝ) : ℝ × ℝ := (2 + t, 2 - 2 * t)

-- Statements to prove
theorem parametric_curve (θ : ℝ) : curve (2 * Real.cos θ) (3 * Real.sin θ) :=
sorry

theorem general_line (x y : ℝ) (t : ℝ) (h : (x, y) = line_parametric t) : 2 * x + y - 6 = 0 :=
sorry

theorem max_PA (θ α : ℝ) (h : ∀ t, (4 * Real.cos θ + 3 * Real.sin θ - 6)/Real.sin (30 * (Real.pi / 180)) = 5 * Real.sin (θ + α)) :
∃ θ, 5 * Real.sin (θ + α) = -6 ∧ ∃ PA_max, PA_max = (22 * Real.sqrt 5) / 5 := 
sorry

theorem min_PA (θ α : ℝ) (h : ∀ t, (4 * Real.cos θ + 3 * Real.sin θ - 6)/Real.sin (30 * (Real.pi / 180)) = 5 * Real.sin (θ + α)) :
∃ θ, 5 * Real.sin (θ + α) = 1 ∧ ∃ PA_min, PA_min = (2 * Real.sqrt 5) / 5 := 
sorry

end parametric_curve_general_line_max_PA_min_PA_l163_163045


namespace equation_of_line_perpendicular_and_passing_point_l163_163282

theorem equation_of_line_perpendicular_and_passing_point :
  ∃ (a b c : ℝ), a = 3 ∧ b = 2 ∧ c = -1 ∧
  (∀ (x y : ℝ), (2 * x - 3 * y + 4 = 0 → y = (2 / 3) * x + 4 / 3) →
  (∀ (x1 y1 : ℝ), x1 = -1 ∧ y1 = 2 →
  (a * x1 + b * y1 + c = 0) ∧
  (∀ (x y : ℝ), (-3 / 2) * (x + 1) + 2 = y) →
  (a * x + b * y + c = 0))) :=
sorry

end equation_of_line_perpendicular_and_passing_point_l163_163282


namespace section_B_students_l163_163583

theorem section_B_students (x : ℕ) 
  (h1 : 50 * 50 = 2500) 
  (h2 : 70 * x > 0)
  (h3 : 58.89 * (50 + x) = 2500 + 70 * x) :
  x = 40 := 
by 
  -- Using known values and conditions to derive the equation
  calc
    -- Apply the average weight equations
    2500 + 70 * 40 = 2944.5 : by sorry,
  -- Solve the resultant system and show the expected outcome
  40 = 40 : by sorry

end section_B_students_l163_163583


namespace Mary_earnings_l163_163833

theorem Mary_earnings :
  ∀ (regular_hours : ℕ) (regular_rate : ℕ) (overtime1_hours : ℕ) (overtime1_rate : ℕ) (overtime2_hours : ℕ) (overtime2_rate : ℕ)
    (overtime3_hours : ℕ) (overtime3_rate : ℕ) (overtime_remaining_hours : ℕ) (overtime_remaining_rate : ℕ),
  regular_hours = 20 ∧ regular_rate = 8 ∧
  overtime1_hours = 10 ∧ overtime1_rate = (8 + (8 * 25 / 100)) ∧
  overtime2_hours = 10 ∧ overtime2_rate = (8 + (8 * 50 / 100)) ∧
  overtime3_hours = 10 ∧ overtime3_rate = (8 + (8 * 75 / 100)) ∧
  overtime_remaining_hours = 20 ∧ overtime_remaining_rate = (8 + (8 * 100 / 100))
  → let total_earnings := (regular_hours * regular_rate) +
                          (overtime1_hours * overtime1_rate) +
                          (overtime2_hours * overtime2_rate) +
                          (overtime3_hours * overtime3_rate) +
                          (overtime_remaining_hours * overtime_remaining_rate)
  in total_earnings = 840 := by
  intros regular_hours regular_rate overtime1_hours overtime1_rate overtime2_hours overtime2_rate 
          overtime3_hours overtime3_rate overtime_remaining_hours overtime_remaining_rate h,
  cases h with h1 h2,
  cases h2 with h3 h4,
  cases h4 with h5 h6,
  cases h6 with h7 h8,
  cases h8 with h9 h10,
  cases h10 with h11 h12,
  cases h12 with h13 h14,
  cases h14 with h15 h16,
  cases h16 with h17 h18,
  cases h18 with h19 h20,
  sorry

end Mary_earnings_l163_163833


namespace five_digit_numbers_without_adjacent_1_and_5_count_l163_163664

theorem five_digit_numbers_without_adjacent_1_and_5_count : 
  let digits := {1, 2, 3, 4, 5} in
  let adjacent_5_positions := (λ (lst : List ℕ), ∃ (i: ℕ), i < 4 ∧ ((lst[i] = 5 ∧ (lst[i+1] = 1 ∨ lst[i+1] = 2)) ∨ ((i > 0) ∧ (lst[i-1] = 5 ∧ (lst[i] = 1 ∨ lst[i] = 2))))) in
  (∃ lst : List ℕ, lst.length = 5 ∧ lst.nodup ∧ lst.perm digits ∧ ¬ adjacent_5_positions lst) → 
   (List.permutations digits).count (λ lst, ¬ adjacent_5_positions lst) = 36
:= 
by 
-- proof omitted
sorry

end five_digit_numbers_without_adjacent_1_and_5_count_l163_163664


namespace number_of_digits_of_smallest_n_is_9_l163_163141

noncomputable def smallest_n : ℕ :=
  (2^6) * (3^6) * (5^6)

theorem number_of_digits_of_smallest_n_is_9 :
  let n := smallest_n in
  n = (2^6) * (3^6) * (5^6) ∧
  (n % 30 = 0) ∧
  ∃ k1 : ℕ, n^2 = (k1 : ℕ)^3 ∧
  ∃ k2 : ℕ, n^3 = (k2 : ℕ)^2 → 
  (string.length (toString n) = 9) :=
sorry

end number_of_digits_of_smallest_n_is_9_l163_163141


namespace solve_for_q_l163_163540

theorem solve_for_q : 
  let n : ℤ := 63
  let m : ℤ := 14
  ∀ (q : ℤ),
  (7 : ℤ) / 9 = n / 81 ∧
  (7 : ℤ) / 9 = (m + n) / 99 ∧
  (7 : ℤ) / 9 = (q - m) / 135 → 
  q = 119 :=
by
  sorry

end solve_for_q_l163_163540


namespace evaluate_f_log2_3_eq_24_l163_163736

def f : ℝ → ℝ :=
  λ x, if x < 4 then f (x + 1) else 2^x

theorem evaluate_f_log2_3_eq_24 : f (2 + Real.log 3 / Real.log 2) = 24 :=
by
  sorry

end evaluate_f_log2_3_eq_24_l163_163736


namespace problem_1_problem_2_l163_163397

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem problem_1 (h₁ : ∀ x, x > 0 → x ≠ 1 → f x = x / Real.log x) :
  (∀ x, 1 < x ∧ x < Real.exp 1 → (Real.log x - 1) / (Real.log x * Real.log x) > 0) ∧
  (∀ x, x > Real.exp 1 → (Real.log x - 1) / (Real.log x * Real.log x) > 0) :=
sorry

theorem problem_2 (h₁ : f x₁ = 1) (h₂ : f x₂ = 1) (h₃ : x₁ ≠ x₂) (h₄ : x₁ > 0) (h₅ : x₂ > 0):
  x₁ + x₂ > 2 * Real.exp 1 :=
sorry

end problem_1_problem_2_l163_163397


namespace total_amount_paid_correct_l163_163288

def initial_value_porcelain := 8500
def initial_value_crystal := 1500
def discounts_porcelain := [0.25, 0.15, 0.05]
def discounts_crystal := [0.30, 0.10, 0.05]

def apply_discount (initial : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (fun acc d => acc * (1 - d)) initial

theorem total_amount_paid_correct :
  apply_discount initial_value_porcelain discounts_porcelain +
  apply_discount initial_value_crystal discounts_crystal 
  = 6045.56 :=
sorry

end total_amount_paid_correct_l163_163288


namespace collinearity_preservation_l163_163910

noncomputable def permutation_of_points (f : Point → Point) : Prop :=
∀ (A B C : Point), 
  (∃ (circ : Circle), circ.contains A ∧ circ.contains B ∧ circ.contains C) → 
  (∃ (circ' : Circle), circ'.contains (f A) ∧ circ'.contains (f B) ∧ circ'.contains (f C))

theorem collinearity_preservation 
  (f : Point → Point)
  (h : permutation_of_points f) :
  ∀ (A B C : Point), collinear A B C ↔ collinear (f A) (f B) (f C) := 
by sorry

end collinearity_preservation_l163_163910


namespace total_collisions_l163_163354

theorem total_collisions (n m : ℕ) (h₁ : n = 5) (h₂ : m = 5) : (n * m) = 25 :=
by
  rw [h₁, h₂]
  norm_num

end total_collisions_l163_163354


namespace index_50th_bn_gt_0_l163_163676

def b (n : ℕ) : ℝ := ∑ k in Finset.range n, Real.cos (k + 1)

theorem index_50th_bn_gt_0 : ∃ n : ℕ, (∑ k in Finset.range n, Real.cos (k + 1)) > 0 ∧ ∀ k, ((∑ k in Finset.range k, Real.cos (k + 1)) > 0 → k < n) ∧ ∑ k in Finset.range (n - 1), Real.cos (k + 1) ≤ 0 :=
begin
  sorry -- Placeholder for the actual proof
end

end index_50th_bn_gt_0_l163_163676


namespace range_of_fx_l163_163720

theorem range_of_fx {f : ℝ → ℝ}
  (h1 : ∀ x, f (-x) = -f x)  -- f(x) is odd
  (h2 : ∀ x y, x < y → x < 0 → y < 0 → f x < f y) -- f(x) is increasing on (-∞, 0)
  (h3 : f (-1) = 0) :
  {x : ℝ | f x > 0} = set.Ioo (-1) 0 ∪ set.Ioi 1 := 
sorry

end range_of_fx_l163_163720


namespace margie_change_l163_163154

def cost_of_banana_cents : ℕ := 30
def cost_of_orange_cents : ℕ := 60
def num_bananas : ℕ := 4
def num_oranges : ℕ := 2
def amount_paid_dollars : ℝ := 10.0

noncomputable def cost_of_banana_dollars := (cost_of_banana_cents : ℝ) / 100
noncomputable def cost_of_orange_dollars := (cost_of_orange_cents : ℝ) / 100

noncomputable def total_cost := 
  (num_bananas * cost_of_banana_dollars) + (num_oranges * cost_of_orange_dollars)

noncomputable def change_received := amount_paid_dollars - total_cost

theorem margie_change : change_received = 7.60 := 
by sorry

end margie_change_l163_163154


namespace intersection_of_A_and_B_l163_163031

def A : Set ℝ := { x | x^2 - x > 0 }
def B : Set ℝ := { x | Real.log x / Real.log 2 < 2 }

theorem intersection_of_A_and_B : A ∩ B = { x | 1 < x ∧ x < 4 } :=
by sorry

end intersection_of_A_and_B_l163_163031


namespace carol_places_second_in_spelling_l163_163226

def distinct_pos_integers (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A > 0 ∧ B > 0 ∧ C > 0

def total_score (Alice Betty Carol : ℕ) : Prop :=
  Alice + Betty + Carol = 39

def exams_sum_to_13 (A B C : ℕ) : Prop :=
  A + B + C = 13

def satisfies_scores (A B C : ℕ) (Alice Betty Carol : list ℕ) : Prop :=
  ∀ scores, scores = [A, B, C] ∨ scores = [A, C, B] ∨ scores = [B, A, C] ∨ scores = [B, C, A] ∨
    scores = [C, A, B] ∨ scores = [C, B, A] →
  Alice = sum (Alice scores) ∧ Betty = sum (Betty scores) ∧ Carol = sum (Carol scores)

theorem carol_places_second_in_spelling (A B C Alice_score Betty_score Carol_score : ℕ)
  (Alice_score_eq : Alice_score = 20)
  (Betty_score_eq : Betty_score = 10)
  (Carol_score_eq : Carol_score = 9)
  (distinct_scores : distinct_pos_integers A B C)
  (sum_of_scores : total_score Alice_score Betty_score Carol_score)
  (sum_of_exams : exams_sum_to_13 A B C)
  (score_conditions : satisfies_scores A B C [Alice_score, Betty_score, Carol_score])
  (Betty_first_in_arithmetic : ∀ exam_exam, exam_exam = "arithmetic" → 
    (Betty_score > Alice_score ∧ Betty_score > Carol_score)) :
  placed_second_in_subject "spelling" Carol_score :=
sorry

end carol_places_second_in_spelling_l163_163226


namespace bank_policy_advantageous_for_retirees_l163_163206

theorem bank_policy_advantageous_for_retirees
  (special_programs : Prop)
  (higher_deposit_rates : Prop)
  (lower_credit_rates : Prop)
  (reliable_loan_payers : Prop)
  (stable_income : Prop)
  (family_interest : Prop)
  (savings_tendency : Prop)
  (regular_income : Prop)
  (long_term_deposits : Prop) :
  reliable_loan_payers ∧ stable_income ∧ family_interest ∧ savings_tendency ∧ regular_income ∧ long_term_deposits → 
  special_programs ∧ higher_deposit_rates ∧ lower_credit_rates :=
sorry

end bank_policy_advantageous_for_retirees_l163_163206


namespace angle_C_is_90_degrees_l163_163036

variables {A B C I : Type*}
variables [Incenter I A B C]

-- Condition related to vectors from incenter to vertices
axiom incenter_eq : 3 • (IA - origin) + 4 • (IB - origin) + 5 • (IC - origin) = (0 : vector_space)

def angle_C := 90

theorem angle_C_is_90_degrees :
  ∀ {triangle ABC : Type*}, incenter_eq → angle_C = 90 := 
sorry

end angle_C_is_90_degrees_l163_163036


namespace cube_edges_not_in_same_plane_as_diagonal_l163_163102

theorem cube_edges_not_in_same_plane_as_diagonal (E : Fin 12) :
  let intersecting_edges := { ("AB", "BC", "CD", "DA", "A1C1", "B1D1") : Set String }
  let total_edges := 12
  let non_intersecting_edges := total_edges - intersecting_edges.size
  non_intersecting_edges = 6 :=
by
  -- Definitions for the edges
  let all_edges := { "AB", "BC", "CD", "DA", "A1C1", "B1D1", "BB1", "A1D1", "A1B1", "B1C1", "C1D1", "DD1" }
  let intersecting_edges := { "AB", "BC", "CD", "DA", "A1C1", "B1D1" }
  let non_intersecting_edges := 12 - 6
  
  have h1 : all_edges.size = 12 := by sorry
  have h2 : intersecting_edges.size = 6 := by sorry
  
  -- Proof that non intersecting edges count is 6
  have h3 : non_intersecting_edges = 6 := by
    calc non_intersecting_edges
      = total_edges - intersecting_edges.size : by rw [non_intersecting_edges, total_edges, intersecting_edges.size]
      = 12 - 6 : by rw [total_edges, h2]
      = 6 : by norm_num
  exact h3

end cube_edges_not_in_same_plane_as_diagonal_l163_163102


namespace product_inequality_l163_163727

theorem product_inequality (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 < x i) :
  let s := ∑ i, x i in
  (∏ i, (1 + x i)) ≤ (∑ k in Finset.range (n + 1), (s ^ k) / (k.factorial)) :=
by
  let s := ∑ i, x i
  sorry

end product_inequality_l163_163727


namespace max_rectangles_in_triangle_l163_163178

theorem max_rectangles_in_triangle (T : Triangle) : 
  let rects := {R : Rectangle | is_inscribed_in R T ∧ has_largest_area R T}
  ∃ (n : ℕ), (n = 1 ∨ n = 3) ∧ (n = card rects) := 
by 
  sorry

end max_rectangles_in_triangle_l163_163178


namespace measure_of_angle_C_l163_163435

-- Given conditions
variables {a b c : ℝ} (A B C : ℝ)

-- Condition based on the given problem
def given_condition : Prop := a^2 + b^2 - c^2 = Real.sqrt 2 * a * b

-- The proof problem statement
theorem measure_of_angle_C (h : given_condition A B C) : C = Real.pi / 4 :=
sorry

end measure_of_angle_C_l163_163435


namespace max_min_diff_l163_163148

theorem max_min_diff (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (λ f : ℝ, ∀ x y, f x y = |x - y| / (|x| + |y|)) → 
  (let m := 0 in let M := 1 in M - m = 1) :=
by sorry

end max_min_diff_l163_163148


namespace production_in_three_minutes_l163_163535

noncomputable def production_rate_per_machine (total_bottles : ℕ) (num_machines : ℕ) : ℕ :=
  total_bottles / num_machines

noncomputable def production_per_minute (machines : ℕ) (rate_per_machine : ℕ) : ℕ :=
  machines * rate_per_machine

noncomputable def total_production (production_per_minute : ℕ) (minutes : ℕ) : ℕ :=
  production_per_minute * minutes

theorem production_in_three_minutes :
  ∀ (total_bottles : ℕ) (num_machines : ℕ) (machines : ℕ) (minutes : ℕ),
  total_bottles = 16 → num_machines = 4 → machines = 8 → minutes = 3 →
  total_production (production_per_minute machines (production_rate_per_machine total_bottles num_machines)) minutes = 96 :=
by
  intros total_bottles num_machines machines minutes h_total_bottles h_num_machines h_machines h_minutes
  sorry

end production_in_three_minutes_l163_163535


namespace f_neg_one_f_eq_half_l163_163396

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then 2^(-x) else Real.log x / Real.log 2

theorem f_neg_one : f (-1) = 2 := by
  sorry

theorem f_eq_half (x : ℝ) : f x = 1 / 2 ↔ x = Real.sqrt 2 := by
  sorry

end f_neg_one_f_eq_half_l163_163396


namespace rowing_ratio_l163_163945

-- Define the speeds
def Vm : ℝ := 3.3
def Vc : ℝ := 1.1

-- Effective speeds upstream and downstream
def Vu : ℝ := Vm - Vc
def Vd : ℝ := Vm + Vc

-- Define the distance D (arbitrary positive value)
variable {D : ℝ} (hD : D > 0)

-- Calculate the times to row upstream and downstream
def Tu := D / Vu
def Td := D / Vd

-- Prove that the ratio of the time to row upstream to the time to row downstream is 2
theorem rowing_ratio : (Tu / Td) = 2 := by
  sorry

end rowing_ratio_l163_163945


namespace find_ab_find_k_l163_163737

variable (a b k : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem find_ab (h₀ : a ≠ 0) (h₁ : f a b (-1) = 0) (h₂ : ∀ x : ℝ, f a b x ≥ 0) :
  a = 1 ∧ b = 2 :=
sorry

def g (k : ℝ) (x : ℝ) : ℝ := x^2 + (2 - k) * x + 1

theorem find_k (ha : a = 1) (hb : b = 2) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → (g k x ≤ g k (x + 1))) ↔ k ∈ set.Iic (-2) ∪ set.Ici 6 :=
sorry

end find_ab_find_k_l163_163737


namespace point_relationship_l163_163058

variables {m x1 x2 y1 y2 : ℝ}

def quadratic_function (x : ℝ) (m : ℝ) : ℝ :=
  (x + m - 3)*(x - m) + 3

theorem point_relationship 
  (hx1_lt_x2 : x1 < x2)
  (hA : y1 = quadratic_function x1 m)
  (hB : y2 = quadratic_function x2 m)
  (h_sum_lt : x1 + x2 < 3) :
  y1 > y2 :=
sorry

end point_relationship_l163_163058


namespace thirtieth_number_base_6_is_50_l163_163448

theorem thirtieth_number_base_6_is_50 :
  let thirtieth_number_base_6 := 30
  (to_base_six : ℕ → ℕ) := 
    nat.div thirtieth_number_base_6 6 = 5 ∧
    nat.mod thirtieth_number_base_6 6 = 0 ∧
    nat.div 5 6 = 0 ∧
    nat.mod 5 6 = 5
  in to_base_six thirtieth_number_base_6 = 50 :=
by
  sorry

end thirtieth_number_base_6_is_50_l163_163448


namespace hyperbola_eccentricity_l163_163405

noncomputable def hyperbola_properties (x y b : ℝ) (b_nat : b ∈ ℕ) : Prop :=
  (x^2 / 4) - (y^2 / b^2) = 1 ∧ b ≠ 0

theorem hyperbola_eccentricity (b : ℕ) (P : ℝ × ℝ) (e : ℝ) :
  let F1 := (P.1 - 2, P.2) in
  let F2 := (P.1 + 2, P.2) in
  let OP := (P.1, P.2) in
  hyperbola_properties P.1 P.2 b → 
  | OP.1 + OP.2 | < 5 →
  let PF1 := (P.1 - F1.1)^2 + (P.2 - F1.2)^2 in
  let PF2 := (P.1 - F2.1)^2 + (P.2 - F2.2)^2 in
  PF1 * PF2 = (4 * √(b^2 + 4))^2 →
  (PF1 - PF2) = 4 →
  e = √(5) / 2 :=
sorry

end hyperbola_eccentricity_l163_163405


namespace find_analytical_expression_of_f_find_range_of_t_l163_163017

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + b * x + c

theorem find_analytical_expression_of_f :
  (∀ x, (0 < x ∧ x < 1) → f x < 0) ↔ (f = λ x, 2 * x^2 - 2 * x + c) :=
begin
  sorry
end

theorem find_range_of_t (t : ℝ) :
  (∀ x ∈ Icc (-1 : ℝ) 1, f x + t ≤ 2) ↔ (t ≤ -2) :=
begin
  sorry
end

end find_analytical_expression_of_f_find_range_of_t_l163_163017


namespace find_other_number_l163_163920

/--
Given two numbers A and B, where:
    * The reciprocal of the HCF of A and B is \( \frac{1}{13} \).
    * The reciprocal of the LCM of A and B is \( \frac{1}{312} \).
    * A = 24
Prove that B = 169.
-/
theorem find_other_number 
  (A B : ℕ) 
  (h1 : A = 24)
  (h2 : (Nat.gcd A B) = 13)
  (h3 : (Nat.lcm A B) = 312) : 
  B = 169 := 
by 
  sorry

end find_other_number_l163_163920


namespace isosceles_triangle_incenter_ratio_l163_163027

theorem isosceles_triangle_incenter_ratio (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_isosceles : ∃ (A B C : Type) [metric_space A] [metric_space B] [metric_space C],
    dist B A = b ∧ dist C B = b ∧ dist A C = a ∧ ∃ (O : Type) [metric_space O], 
    is_incenter O A B C) : 
  ∃ (f : Type) [linear_ordered_field f], (f = (O).divides_angle_bisector A B C) = (a + b) / b :=
sorry

end isosceles_triangle_incenter_ratio_l163_163027


namespace probability_C_l163_163267

noncomputable def P : Type := ℝ

def P_A : P := 3 / 8
def P_B : P := 1 / 4
def P_D : P := 1 / 8

def total_prob : P := 1

theorem probability_C :
  ∃ P_C : P, P_C = 1 - P_A - P_B - P_D ∧ P_C = 1 / 4 :=
by
  use 1 - P_A - P_B - P_D
  split
  { calc
      1 - P_A - P_B - P_D = 1 - 3 / 8 - 1 / 4 - 1 / 8 : sorry
       ... = 1 - 3 / 8 - 2 / 8 - 1 / 8 : sorry
       ... = 1 - 6 / 8 : sorry
       ... = 2 / 8 : sorry
       ... = 1 / 4 : sorry }

end probability_C_l163_163267


namespace additional_cards_l163_163949

theorem additional_cards (total_cards : ℕ) (complete_decks : ℕ) (cards_per_deck : ℕ) (num_decks : ℕ) 
  (h1 : total_cards = 160) (h2 : num_decks = 3) (h3 : cards_per_deck = 52) :
  total_cards - (num_decks * cards_per_deck) = 4 :=
by
  sorry

end additional_cards_l163_163949


namespace probability_blackboard_empty_k_l163_163307

-- Define the conditions for the problem
def Ben_blackboard_empty_probability (n : ℕ) : ℚ :=
  if h : n = 2013 then (2 * (2013 / 3) + 1) / 2^(2013 / 3 * 2) else 0 / 1

-- Define the theorem that Ben's blackboard is empty after 2013 flips, and determine k
theorem probability_blackboard_empty_k :
  ∃ (u v k : ℕ), Ben_blackboard_empty_probability 2013 = (2 * u + 1) / (2^k * (2 * v + 1)) ∧ k = 1336 :=
by sorry

end probability_blackboard_empty_k_l163_163307


namespace problem1_problem2_problem3_l163_163263

-- Problem 1
theorem problem1 : let i := complex.i in
  i + i^2 + i^3 + i^4 = 0 := sorry

-- Problem 2
theorem problem2 : (nat.choose 100 2 + nat.choose 100 97) / nat.fact 101 = 1 / 6 := sorry

-- Problem 3
theorem problem3 : (finset.range 11).sum (λ n, if n ≥ 3 then nat.choose n 3 else 0) = 330 := sorry

end problem1_problem2_problem3_l163_163263


namespace degrees_of_freedom_uniform_distribution_l163_163604

theorem degrees_of_freedom_uniform_distribution
    (s : ℕ) -- Number of sample intervals
    (r : ℕ) -- Number of parameters estimated from the sample
    (h1 : r = 2) -- Uniform distribution assumption that r = 2
    (h2 : ∀ s r, k = s - 1 - r) -- General formula for degrees of freedom
: k = s - 3 :=
by
  -- Due to uniform distribution assumption
  have h3 : r = 2 := h1,
  -- Apply the general formula for degrees of freedom
  have h4 := h2 s r,
  -- Substitute r = 2 into the general formula
  sorry

end degrees_of_freedom_uniform_distribution_l163_163604


namespace solution_in_quadrant_I_l163_163331

theorem solution_in_quadrant_I (k : ℝ) :
  (∃ x y : ℝ, 
    3 * x - 4 * y = 6 ∧ 
    k * x + 2 * y = 8 ∧ 
    0 < x ∧ 
    0 < y) ↔ 
  -3 / 2 < k ∧ k < 4 :=
sorry

end solution_in_quadrant_I_l163_163331


namespace proof_problem_l163_163741

section proof_problem

variable {α : Type} [LinearOrder α] [AddGroup α] [OrderedAddCommGroup α] {f : α → α}

def odd_function (f : α → α) : Prop :=
  ∀ x, f (-x) = -f(x)

def periodic_function (f : α → α) (T : α) : Prop :=
  ∀ x, f (x + T) = f x

def monotonically_decreasing_function (f : α → α) : Prop :=
  ∀ x y, x < y → f x > f y

def has_real_root (f : α → α) : Prop :=
  ∃ x, f x = x

def has_real_root_composed (f : α → α) : Prop :=
  ∃ x, f (f x) = x

theorem proof_problem :
  (odd_function f → odd_function (f ∘ f))
  ∧ (∃ T, periodic_function f T ∧ periodic_function (f ∘ f) T)
  ∧ (monotonically_decreasing_function f → ∀ x y, x < y → f (f x) < f (f y))
  ∧ ¬(has_real_root_composed f → has_real_root f) :=
by sorry

end proof_problem

end proof_problem_l163_163741


namespace crocodile_length_in_meters_l163_163204

-- Definitions based on conditions
def ken_to_cm : ℕ := 180
def shaku_to_cm : ℕ := 30
def ken_to_shaku : ℕ := 6
def cm_to_m : ℕ := 100

-- Lengths given in the problem expressed in ken
def head_to_tail_in_ken (L : ℚ) : Prop := 3 * L = 10
def tail_to_head_in_ken (L : ℚ) : Prop := L = (3 + (2 / ken_to_shaku : ℚ))

-- Final length conversion to meters
def length_in_m (L : ℚ) : ℚ := L * ken_to_cm / cm_to_m

-- The length of the crocodile in meters
theorem crocodile_length_in_meters (L : ℚ) : head_to_tail_in_ken L → tail_to_head_in_ken L → length_in_m L = 6 :=
by
  intros _ _
  sorry

end crocodile_length_in_meters_l163_163204


namespace fraction_of_fractions_eq_l163_163595

theorem fraction_of_fractions_eq :
  ∃ x : ℚ, x * (5 / 9) * (1 / 2) = 11111111111111112 / 100000000000000000 :=
∃ x : ℚ, x = 4 / 10 := sorry

end fraction_of_fractions_eq_l163_163595


namespace complex_in_fourth_quadrant_l163_163113

theorem complex_in_fourth_quadrant (m : ℝ) (z : ℂ) (h : z = (m + complex.I) / (1 + complex.I)) : 
  z.re > 0 ∧ z.im < 0 → m > 1 :=
by
  sorry

end complex_in_fourth_quadrant_l163_163113


namespace find_equation_of_ellipse_find_ratio_BM_BN_l163_163286
-- Import necessary libraries

-- Define parameters and conditions
variables (a b x1 y1 x2 y2 x3 y3 lambda : ℝ)
variables (h1 : a > b ∧ b > 0)
variables (h2 : 2 * a * b = 2 * real.sqrt 2)
variables (h3 : a^2 + b^2 = 3)
variables (h4 : (x1, y1) ∈ set_of (p : ℝ × ℝ, (p.1^2) / 2 + p.2^2 = 1))
variables (h5 : (x2, y2) ∈ set_of (p : ℝ × ℝ, (p.1^2) / 2 + p.2^2 = 1))
variables (h6 : y1 * y2 / (x1 * x2) = -1 / 2)
variables (h7 : x3 = (2 / (3 * lambda)) * x1 + ((lambda - 1) / lambda) * x2)
variables (h8 : y3 = (2 / (3 * lambda)) * y1 + ((lambda - 1) / lambda) * y2)
variables (h9 : (x3, y3) ∈ set_of (p : ℝ × ℝ, (p.1^2) / 2 + p.2^2 = 1))

-- Proof that the equation of the ellipse C is x^2/2 + y^2 = 1
theorem find_equation_of_ellipse : (a = real.sqrt 2 ∧ b = 1) -> (∀ x y, (x^2)/2 + y^2 = 1) :=
sorry

-- Proof that the value of BM/BN is 13/18
theorem find_ratio_BM_BN : λ = 13 / 18 :=
begin
    sorry
end

end find_equation_of_ellipse_find_ratio_BM_BN_l163_163286


namespace units_digit_expression_l163_163332

lemma units_digit_2_pow_2023 : (2 ^ 2023) % 10 = 8 := sorry

lemma units_digit_5_pow_2024 : (5 ^ 2024) % 10 = 5 := sorry

lemma units_digit_11_pow_2025 : (11 ^ 2025) % 10 = 1 := sorry

theorem units_digit_expression : ((2 ^ 2023) * (5 ^ 2024) * (11 ^ 2025)) % 10 = 0 :=
by 
  have h1 := units_digit_2_pow_2023
  have h2 := units_digit_5_pow_2024
  have h3 := units_digit_11_pow_2025
  sorry

end units_digit_expression_l163_163332


namespace sum_binom_P_eq_l163_163130

/-- Let P_k(x) = 1 + x + x^2 + ... + x^(k-1). We aim to prove that 
  ∑ k in finset.range n, (binom n k) * P_k(x) = 2^(n-1) * P_n ((x + 1) / 2) 
  for every real number x and every positive integer n. -/

def P : ℕ → ℝ → ℝ
| 0, _ => 0
| (k + 1), x => (1 - x^(k+1)) / (1 - x)

theorem sum_binom_P_eq
  (x : ℝ) (n : ℕ) :
  ∑ k in finset.range n, nat.choose n k * P k x = 2^(n - 1) * P n ((x + 1) / 2) :=
sorry

end sum_binom_P_eq_l163_163130


namespace students_can_enter_disinfection_effective_l163_163649

def concentration (t : ℝ) : ℝ :=
if h : 0 ≤ t ∧ t ≤ 10 then 0.1 * t else (0.5)^(0.1 * t - 1)

theorem students_can_enter (t : ℝ) : 
  (concentration t ≤ 0.25) → t ≥ 30 :=
begin
  sorry
end

theorem disinfection_effective :
  (∃ t1 t2 : ℝ, 0 ≤ t1 ∧ t1 ≤ t2 ∧ (concentration t2 < 0.5) ∧ (t2 - t1 ≥ 8)) :=
begin
  sorry
end

end students_can_enter_disinfection_effective_l163_163649


namespace quadratic_inequality_solution_l163_163184

theorem quadratic_inequality_solution :
  (∀ x : ℝ, x ∈ Set.Ioo ((1 - Real.sqrt 2) / 3) ((1 + Real.sqrt 2) / 3) → -9 * x^2 + 6 * x + 1 < 0) ∧
  (∀ x : ℝ, -9 * x^2 + 6 * x + 1 < 0 → x ∈ Set.Ioo ((1 - Real.sqrt 2) / 3) ((1 + Real.sqrt 2) / 3)) :=
by
  sorry

end quadratic_inequality_solution_l163_163184


namespace percentage_second_correct_l163_163764

-- Definitions for the conditions provided in the problem
def percentage_first_correct := 0.63
def percentage_neither_correct := 0.20
def percentage_both_correct := 0.33

-- The Lean statement representing the mathematical proof problem
theorem percentage_second_correct :
  (0.63 + percentage_second - 0.33 = 0.80) → percentage_second = 0.50 :=
begin
  let percentage_second := (0.80 + 0.33 - 0.63 : ℝ),
  sorry,
end

end percentage_second_correct_l163_163764


namespace log_inequality_implies_fractional_bases_l163_163076

theorem log_inequality_implies_fractional_bases {a b : ℝ}
  (h₁ : log a 2 < log b 2) (h₂ : log b 2 < 0) :
  0 < b ∧ b < a ∧ a < 1 :=
sorry

end log_inequality_implies_fractional_bases_l163_163076


namespace digit_in_decimal_expansion_l163_163679

def decimal_repeating_sequence : ℕ → ℕ
| 1 := 2 | 2 := 4 | 3 := 1 | 4 := 3
| 5 := 7 | 6 := 9 | 7 := 3 | 8 := 1
| 9 := 0 | 10 := 3 | 11 := 4 | 12 := 4
| 13 := 8 | 14 := 2 | 15 := 7 | 16 := 5
| 17 := 8 | 18 := 6 | 19 := 2 | 20 := 0
| 21 := 6 | 22 := 8 | 23 := 9 | 24 := 6
| 25 := 5 | 26 := 5 | 27 := 1 | 28 := 7

theorem digit_in_decimal_expansion :
  (decimal_repeating_sequence (789 % 28)) = 6 :=
by
  sorry

end digit_in_decimal_expansion_l163_163679


namespace midpoint_perpendicular_bisector_l163_163166

theorem midpoint_perpendicular_bisector
  {A B C P : Point}
  {M S1 S2 : Point} [AcuteTriangle A B C]
  (hM_midpoint : Midpoint M A B)
  (hP_on_AB : OnSegment P A B)
  (hS1_circumcenter : Circumcenter S1 A P C)
  (hS2_circumcenter : Circumcenter S2 B P C) :
  ∃ N : Point, Midpoint N S1 S2 ∧ OnPerpendicularBisector N M C :=
sorry

end midpoint_perpendicular_bisector_l163_163166


namespace positive_integer_equality_l163_163527

theorem positive_integer_equality (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
    (h1 : fract (Real.sqrt (↑(x^2) + 2 * ↑y)) > 2 / 3) 
    (h2 : fract (Real.sqrt (↑(y^2) + 2 * ↑x)) > 2 / 3) : 
    x = y :=
by {
  -- proof goes here 
  sorry
}

end positive_integer_equality_l163_163527


namespace point_relationship_l163_163057

variables {m x1 x2 y1 y2 : ℝ}

def quadratic_function (x : ℝ) (m : ℝ) : ℝ :=
  (x + m - 3)*(x - m) + 3

theorem point_relationship 
  (hx1_lt_x2 : x1 < x2)
  (hA : y1 = quadratic_function x1 m)
  (hB : y2 = quadratic_function x2 m)
  (h_sum_lt : x1 + x2 < 3) :
  y1 > y2 :=
sorry

end point_relationship_l163_163057


namespace arithmetic_seq_property_l163_163785

variable {α : Type} [AddGroup α] [AddCommGroup α] {a : ℕ → α}

theorem arithmetic_seq_property (a_4 a_5 a_6 : α) (h1 : a_5 = 2) (h2 : a_4 + a_6 = 2 * a_5) :
  a_4 - a_5 + a_6 = 2 :=
by
  sorry

end arithmetic_seq_property_l163_163785


namespace train_speed_l163_163959

-- Define the conditions
def train_length : ℝ := 50 -- Length of the train in meters
def crossing_time : ℝ := 3 -- Time to cross the pole in seconds

-- Define the speed in meters per second and convert it to km/hr
noncomputable def speed_mps : ℝ := train_length / crossing_time
noncomputable def speed_kmph : ℝ := speed_mps * 3.6 -- Conversion factor

-- Theorem statement: Prove that the calculated speed in km/hr is 60 km/hr
theorem train_speed : speed_kmph = 60 := by
  sorry

end train_speed_l163_163959


namespace sum_of_inv_squares_of_distance_to_one_l163_163988

theorem sum_of_inv_squares_of_distance_to_one (z : ℂ) (hz : z^8 = 1) : 
  ∑ z in {z : ℂ | z^8 = 1}, 1 / |1 - z|^2 = 37 / 2 := 
by sorry

end sum_of_inv_squares_of_distance_to_one_l163_163988


namespace three_digit_number_difference_l163_163441

theorem three_digit_number_difference
  (x y z : ℕ)
  (h1 : y = z + 2) :
  let N := 100 * y + 10 * x + z
      M := 100 * z + 10 * x + y
  in N - M = 198 := 
by
  sorry

end three_digit_number_difference_l163_163441


namespace subcommitteesWithAtLeastOneTeacher_l163_163214

namespace Subcommittee

def totalCombinations : ℕ := 12.choose 5
def nonTeacherCombinations : ℕ := 7.choose 5

theorem subcommitteesWithAtLeastOneTeacher : totalCombinations - nonTeacherCombinations = 771 :=
by
  sorry

end Subcommittee

end subcommitteesWithAtLeastOneTeacher_l163_163214


namespace relationship_between_y_l163_163770

theorem relationship_between_y (y1 y2 y3 m : ℝ) :
  y1 = 2^2 - 4 * 2 - m ∧ y2 = 3^2 - 4 * 3 - m ∧ y3 = (-1)^2 - 4 * (-1) - m → y3 > y2 ∧ y2 > y1 :=
by
  intro h
  cases h with h1 h23
  cases h23 with h2 h3
  calc
    y3 = 5 - m : h3
    ... > -3 - m : by sorry
    ... = y2 : h2.symm
    ... > -4 - m : by sorry
    ... = y1 : h1.symm

end relationship_between_y_l163_163770


namespace Part_1_Part_2_l163_163472

noncomputable def S (n : ℕ) : ℤ := (40 : ℤ) -- Given S₁₀ = 40
def a (n : ℕ) : ℤ := if n = 2 then 11 else 0 -- Given a₂ = 11
def a_n (n : ℕ) : ℤ := if n = 2 then a n else 15 - 2 * (n - 1) -- General form for a_n

theorem Part_1 {a1 d : ℤ} (h₁ : a_n 2 = 11) (h₂ : ∑ i in range 10, a_n (i + 1) = 40) :
  ∀ n, a_n n = 15 - 2 * (n-1) := by
  sorry

def absSum1 (n : ℕ) : ℤ := -n^2 + 14 * n
def absSum2 (n : ℕ) : ℤ := n^2 - 14 * n + 98

theorem Part_2 {a_n : ℕ → ℤ} (h₁ : ∀ n, a_n n = 15 - 2 * (n-1)) :
  (∀  n, 1 ≤ n ∧ n ≤ 7 → ∑ i in range n, |a_n (i + 1)| = -n^2 + 14 * n) ∧
  (∀ n, n ≥ 8 → ∑ i in range n, |a_n (i + 1)| = n^2 - 14 * n + 98) := by
  sorry

end Part_1_Part_2_l163_163472


namespace minimum_wire_length_cube_l163_163239

-- Definitions based on conditions
def num_edges : ℕ := 12
def edge_length : ℕ := 10
def wire_can_pass_twice : Prop := True
def wire_can_bend : Prop := True
def wire_cannot_be_broken : Prop := True

-- Problem statement
theorem minimum_wire_length_cube :
  wire_can_pass_twice →
  wire_can_bend →
  wire_cannot_be_broken →
  num_edges = 12 →
  edge_length = 10 →
  ∃ l, l = 150 :=
by
  intros
  use 150
  sorry

end minimum_wire_length_cube_l163_163239


namespace mixture_replacement_l163_163280

theorem mixture_replacement
  (A B : ℕ)
  (hA : A = 48)
  (h_ratio1 : A / B = 4)
  (x : ℕ)
  (h_ratio2 : A / (B + x) = 2 / 3) :
  x = 60 :=
by
  sorry

end mixture_replacement_l163_163280


namespace cos_pi_plus_alpha_l163_163723

theorem cos_pi_plus_alpha (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : Real.cos (π + α) = - 1 / 3 :=
by
  sorry

end cos_pi_plus_alpha_l163_163723


namespace shirt_price_after_discount_l163_163995

/-- Given a shirt with an initial cost price of $20 and a profit margin of 30%, 
    and a sale discount of 50%, prove that the final sale price of the shirt is $13. -/
theorem shirt_price_after_discount
  (cost_price : ℝ)
  (profit_margin : ℝ)
  (discount : ℝ)
  (selling_price : ℝ)
  (final_price : ℝ)
  (h_cost : cost_price = 20)
  (h_profit_margin : profit_margin = 0.30)
  (h_discount : discount = 0.50)
  (h_selling_price : selling_price = cost_price + profit_margin * cost_price)
  (h_final_price : final_price = selling_price - discount * selling_price) :
  final_price = 13 := 
  sorry

end shirt_price_after_discount_l163_163995


namespace part_I_part_II_l163_163402

/-- Part I: Given the function f, find f(π/4). -/
theorem part_I (ω : ℝ) (hω : 0 < ω) :
  let f (x : ℝ) := sin (ω * x) ^ 2 + 2 * sqrt 3 * sin (ω * x) * cos (ω * x) - cos (ω * x) ^ 2 in
  f (π / 4) = 1 := 
  sorry

/-- Part II: Given the function g and a symmetry center, find the monotonically increasing intervals. -/
theorem part_II :
  let g (x : ℝ) := 2 * sin (4 * x + π / 3) in
  (∀ k : ℤ, 
    (∃ l : ℝ, l ∈ Ioo (k * π / 2 - 5 * π / 24) (k * π / 2 + π / 24)) ∧
    ∀ x ∈ Ioo (k * π / 2 - 5 * π / 24) (k * π / 2 + π / 24), 
      0 < deriv (g x)) :=
  sorry

end part_I_part_II_l163_163402


namespace total_ounces_of_coffee_l163_163665

/-
Defining the given conditions
-/
def num_packages_10_oz : Nat := 5
def num_packages_5_oz : Nat := num_packages_10_oz + 2
def ounces_per_10_oz_pkg : Nat := 10
def ounces_per_5_oz_pkg : Nat := 5

/-
Statement to prove the total ounces of coffee
-/
theorem total_ounces_of_coffee :
  (num_packages_10_oz * ounces_per_10_oz_pkg + num_packages_5_oz * ounces_per_5_oz_pkg) = 85 := by
  sorry

end total_ounces_of_coffee_l163_163665


namespace number_of_standing_demons_l163_163790

variable (N : ℕ)
variable (initial_knocked_down : ℕ)
variable (initial_standing : ℕ)
variable (current_knocked_down : ℕ)
variable (current_standing : ℕ)

axiom initial_condition : initial_knocked_down = (3 * initial_standing) / 2
axiom condition_after_changes : current_knocked_down = initial_knocked_down + 2
axiom condition_after_changes_2 : current_standing = initial_standing - 10
axiom final_condition : current_standing = (5 * current_knocked_down) / 4

theorem number_of_standing_demons : current_standing = 35 :=
sorry

end number_of_standing_demons_l163_163790


namespace smallest_real_constant_α_l163_163694

noncomputable def α : ℝ := 16 * Real.sqrt 2 / 9

theorem smallest_real_constant_α (n : ℕ) (y : ℕ → ℝ)
  (h0 : y 0 = 0) (h1 : ∀ k < n, y k < y (k+1)) :
  α * ∑ k in Finset.range n, (k+1)^(3/2 : ℝ) / Real.sqrt ((y (k+1))^2 - (y k)^2) ≥
    ∑ k in Finset.range n, (k^2 + 3 * k + 3 : ℝ) / (y (k+1)) :=
sorry

end smallest_real_constant_α_l163_163694


namespace rectangle_perimeter_l163_163205

theorem rectangle_perimeter : 
  ∀ (length width : ℝ), 
  length = 0.54 ∧ length = width + 0.08 → 
  2 * (length + width) = 2 :=
by
  intros length width h,
  let ⟨h1, h2⟩ := h,
  rw [h1, h2],
  sorry

end rectangle_perimeter_l163_163205


namespace multiplicative_magic_square_l163_163890

theorem multiplicative_magic_square :
  ∃ (b c d f h : ℕ), (∀ g ∈ {1, 2, 3, 5, 6, 10, 15, 25, 30, 50, 75, 150}, g * (150 / g) = 150) ∧
  (30 * b * c = 450) ∧
  (d * 5 * f = 450) ∧
  (g * h * 3 = 450) ∧
  ((1 * 2 * 3 * 5 * 6 * 10 * 15 * 25 * 30 * 50 * 75 * 150) = 113906250000) :=
by
  use sorry

end multiplicative_magic_square_l163_163890


namespace max_value_of_expression_l163_163137

noncomputable def max_function_value (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) : ℝ :=
  max 
    (set.image (λ (x : ℝ × ℝ × ℝ), 
       let ⟨a, b, c⟩ := x in
       (a^2 * b^2) / (a + b) + (a^2 * c^2) / (a + c) + (b^2 * c^2) / (b + c)) 
      {x | let ⟨a, b, c⟩ := x in 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b^2 + c^2 = 1})

theorem max_value_of_expression 
  (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  max_function_value a b c h1 h2 h3 h4 = 1 / 6 := by
  sorry

end max_value_of_expression_l163_163137


namespace find_p_in_terms_of_a_b_l163_163671

variables {α : Type*} [Field α]

theorem find_p_in_terms_of_a_b (a b p : α) (α β : α) (h1 : α^2 - a * α + b = 0) 
    (h2 : β^2 - a * β + b = 0) (h3 : (α^2 + β^2)^2 - p * (α^2 + β^2) + β * (α^2 + β^2) = 0) : 
    p = a^2 - b :=
by
  -- sorry to skip the proof.
  sorry

end find_p_in_terms_of_a_b_l163_163671


namespace num_possible_chords_bisected_by_AB_l163_163229

-- Definitions of the problem's given conditions
variables {P : Type} [euclidean_space P] 
variables {C : circle P} {O M A B N K : P}

-- Given conditions
variable (hM : M ∈ C)
variable (hO : O = center C)
variable (hAB : is_chord C A B)

-- Objective: Prove the number of possible chords MN through M bisected by AB is 0, 1, or 2
theorem num_possible_chords_bisected_by_AB :
  ∃ (n : ℕ), n = 0 ∨ n = 1 ∨ n = 2 :=
sorry

end num_possible_chords_bisected_by_AB_l163_163229


namespace sum_of_sequence_l163_163087

theorem sum_of_sequence :
  (∑ k in Finset.range 10, (λ n, (-1)^n * (3 * n - 2)) (k + 1)) = 15 :=
by
  sorry

end sum_of_sequence_l163_163087


namespace difference_between_positive_integers_with_conditions_l163_163575

noncomputable def difference_of_two_integers (a b : ℕ) : ℕ := abs (a - b)

theorem difference_between_positive_integers_with_conditions (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b + a * b = 2005) (h_perfect_square : ∃ x : ℕ, x * x = a ∨ x * x = b) :
  difference_of_two_integers a b = 1001 ∨ difference_of_two_integers a b = 101 :=
sorry

end difference_between_positive_integers_with_conditions_l163_163575


namespace triangle_area_not_twice_parallelogram_l163_163865

theorem triangle_area_not_twice_parallelogram (b h : ℝ) :
  (1 / 2) * b * h ≠ 2 * b * h :=
sorry

end triangle_area_not_twice_parallelogram_l163_163865


namespace last_black_probability_l163_163628

/-- Given a box containing 3 black balls and 4 white balls,
    balls are drawn randomly one at a time without replacement 
    until all balls of one color have been drawn. 
    The probability that the last ball drawn is a black ball is 3/7. -/
theorem last_black_probability (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (without_replacement : Prop) (until_all_of_one_color : Prop) : 
  total_balls = 7 ∧ black_balls = 3 ∧ white_balls = 4 → 
  (∃ probability, probability = 3 / 7) :=
begin
  intros h,
  cases h,
  exact ⟨3/7, rfl⟩,
end

end last_black_probability_l163_163628


namespace smallest_integer_solution_system_of_inequalities_solution_l163_163925

-- Define the conditions and problem
variable (x : ℝ)

-- Part 1: Prove smallest integer solution for 5x + 15 > x - 1
theorem smallest_integer_solution :
  5 * x + 15 > x - 1 → x = -3 := sorry

-- Part 2: Prove solution set for system of inequalities
theorem system_of_inequalities_solution :
  (-3 * (x - 2) ≥ 4 - x) ∧ ((1 + 4 * x) / 3 > x - 1) → (-4 < x ∧ x ≤ 1) := sorry

end smallest_integer_solution_system_of_inequalities_solution_l163_163925


namespace max_distance_implies_m_l163_163975

noncomputable theory

-- Definitions: 
def ellipse (a b : ℝ) (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def distance (x1 y1 x2 y2 : ℝ) := real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Assumptions:
variables {P Q : ℝ × ℝ} {m : ℝ}
variables (x y : ℝ)
variable (m_pos : 0 < m)

-- Assumptions translated to definitions in Lean 4
def is_on_ellipse : Prop := ellipse 2 (real.sqrt 3) x y

-- Requirement:
def max_distance_PQ (m : ℝ) : ℝ := distance x y 0 (-m)

-- Final theorem statement:
theorem max_distance_implies_m:
  ∀ (P : ℝ × ℝ), is_on_ellipse P.1 P.2 → max_distance_PQ P.1 P.2 m = real.sqrt 5 → m = 1 / 2 :=
sorry

end max_distance_implies_m_l163_163975


namespace tan_alpha_add_pi_over_4_l163_163037

theorem tan_alpha_add_pi_over_4 (α : ℝ) (P : ℝ × ℝ) (hP : P = (-1, 2)) :
  let tan_α := -2
  tan (α + π / 4) = -1 / 3 :=
by
  -- since P = (-1, 2), we know tan α = -2
  have tan_α_def : tan_α = -2, from rfl
  -- applying the tangent addition formula
  let t := (1 + tan_α) / (1 - tan_α)
  have : t = (1 + (-2)) / (1 - (-2)), from rfl
  rw [this]
  have : t = -1 / 3, -- simplification
  sorry
  exact this

end tan_alpha_add_pi_over_4_l163_163037


namespace P_lt_Q_l163_163359

noncomputable def P (a : ℝ) : ℝ := real.sqrt a + real.sqrt (a + 5)
noncomputable def Q (a : ℝ) : ℝ := real.sqrt (a + 2) + real.sqrt (a + 3)

theorem P_lt_Q (a : ℝ) (h : 0 ≤ a) : P a < Q a := 
by 
  -- Proof details are skipped
  sorry

end P_lt_Q_l163_163359


namespace calculate_volume_and_diagonal_calculate_volume_and_surface_rotation_calculate_radius_given_volume_l163_163566

noncomputable def volume_of_parallelepiped (R : ℝ) : ℝ := R^3 * Real.sqrt 6

noncomputable def diagonal_A_C_prime (R: ℝ) : ℝ := R * Real.sqrt 6

noncomputable def volume_of_rotation (R: ℝ) : ℝ := R^3 * Real.sqrt 12

theorem calculate_volume_and_diagonal (R : ℝ) : 
  volume_of_parallelepiped R = R^3 * Real.sqrt 6 ∧ 
  diagonal_A_C_prime R = R * Real.sqrt 6 :=
by sorry

theorem calculate_volume_and_surface_rotation (R : ℝ) :
  volume_of_rotation R = R^3 * Real.sqrt 12 :=
by sorry

theorem calculate_radius_given_volume (V : ℝ) (h : V = 0.034786) : 
  ∃ R : ℝ, V = volume_of_parallelepiped R :=
by sorry

end calculate_volume_and_diagonal_calculate_volume_and_surface_rotation_calculate_radius_given_volume_l163_163566


namespace periodic_functions_exist_l163_163467

theorem periodic_functions_exist (p1 p2 : ℝ) (h1 : p1 > 0) (h2 : p2 > 0) :
    ∃ (f1 f2 : ℝ → ℝ), (∀ x, f1 (x + p1) = f1 x) ∧ (∀ x, f2 (x + p2) = f2 x) ∧ ∃ T > 0, ∀ x, (f1 - f2) (x + T) = (f1 - f2) x :=
sorry

end periodic_functions_exist_l163_163467


namespace ab_equals_two_l163_163306

theorem ab_equals_two (a b : ℝ)
  (root_condition : (∃ x : ℝ, ∀ x ∈ { -Real.pi / 4, Real.pi / 4 }, a * Real.tan (b * x) = 0))
  (point_condition : a * Real.tan (b * Real.pi / 8) = 1) :
  a * b = 2 :=
sorry

end ab_equals_two_l163_163306


namespace hilton_final_marbles_l163_163755

def initial_marbles : ℕ := 26
def found_marbles : ℕ := 6
def lost_marbles : ℕ := 10
def given_marbles := 2 * lost_marbles

theorem hilton_final_marbles (initial_marbles : ℕ) (found_marbles : ℕ) (lost_marbles : ℕ)
  (given_marbles : ℕ) : 
  initial_marbles = 26 →
  found_marbles = 6 →
  lost_marbles = 10 →
  given_marbles = 2 * lost_marbles →
  (initial_marbles + found_marbles - lost_marbles + given_marbles) = 42 :=
by
  intros,
  sorry

end hilton_final_marbles_l163_163755


namespace angle_BHC_correct_l163_163317

open_locale real sojun

noncomputable def angle_BHC (A B C D E F G H : Point) (α : ℝ) : Prop :=
  let arc_mAC := α
  let AD_perpendicular_BC := AD ⊥ BC
  let E_midpoint_AC := midpoint E A C
  let F_midpoint_CD := midpoint F C D
  let G_intersection_AD_BE := G ∈ (line AD).intersection (line BE)
  let H_intersection_AF_BC := H ∈ (line AF).intersection (line BC)
  ∠B H C = 90° - α / 2

theorem angle_BHC_correct {A B C D E F G H : Point} {α : ℝ} 
  (AD_perp_BC : AD ⊥ BC)
  (E_mid_ARC_AC : midpoint E A C)
  (F_mid_ARC_CD : midpoint F C D)
  (G_int_AD_BE : G ∈ (line AD).intersection (line BE))
  (H_int_AF_BC : H ∈ (line AF).intersection (line BC))
  (mAC : ∠ arc A C = α) :
  angle_BHC A B C D E F G H α :=
begin
  sorry
end

end angle_BHC_correct_l163_163317


namespace problem_solution_l163_163097

open Real TrigonometricConstants

namespace TriangleProblem

theorem problem_solution
  (A B C : ℝ) 
  (sin_A sin_B sin_C : ℝ)
  (h1 : sin_A = sin A)
  (h2 : sin_B = sin B)
  (h3 : sin_C = sin C)
  (condition : (sin_A + sin_B + sin_C) * (sin_B + sin_C - sin_A) = 3 * sin_B * sin_C)
  (triangle_constraint : A + B + C = π) 
  (A_pos : 0 < A) (A_lt_pi : A < π)
  : (A = π / 3) ∧ 
    (∃ B C, 0 < B ∧ B < 2 * π / 3 ∧ 
    (A + B + C = π) ∧ 
    (sqrt(3) * sin B - cos C) ≤ 1) :=
by
  sorry

end TriangleProblem

end problem_solution_l163_163097


namespace average_speed_correct_l163_163616

-- Define the distances and time durations as given in the problem
def distance_first_hour : ℝ := 90
def distance_second_hour : ℝ := 80
def time_first_hour : ℝ := 1
def time_second_hour : ℝ := 1

-- The total distance is the sum of the distances traveled in both hours
def total_distance : ℝ := distance_first_hour + distance_second_hour

-- The total time is the sum of the times taken in both hours
def total_time : ℝ := time_first_hour + time_second_hour

-- The average speed is the total distance divided by the total time
def average_speed : ℝ := total_distance / total_time

-- Prove that the average speed is 85 km/h
theorem average_speed_correct :
  average_speed = 85 := by
  calc
    average_speed = (distance_first_hour + distance_second_hour) / (time_first_hour + time_second_hour) : rfl
                ... = 170 / 2 : by sorry
                ... = 85 : by sorry

end average_speed_correct_l163_163616


namespace total_inscribed_circle_area_l163_163591

theorem total_inscribed_circle_area (a b c : ℝ) (K : ℝ)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_heron : K = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
  let r := (2 * K) / (a + b + c) in
  let total_area := (Real.pi * r ^ 2 * (a ^ 2 + b ^ 2 + c ^ 2) / (a + b + c) ^ 2) in
  total_area = 
    (a^2 + b^2 + c^2) * Real.pi * (r^2 / (a + b + c)^2) :=
by sorry

end total_inscribed_circle_area_l163_163591


namespace final_price_hat_final_price_tie_l163_163287

theorem final_price_hat (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) 
    (h_initial : initial_price = 20) 
    (h_first : first_discount = 0.25) 
    (h_second : second_discount = 0.20) : 
    initial_price * (1 - first_discount) * (1 - second_discount) = 12 := 
by 
  rw [h_initial, h_first, h_second]
  norm_num

theorem final_price_tie (initial_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) 
    (t_initial : initial_price = 15) 
    (t_first : first_discount = 0.10) 
    (t_second : second_discount = 0.30) : 
    initial_price * (1 - first_discount) * (1 - second_discount) = 9.45 := 
by 
  rw [t_initial, t_first, t_second]
  norm_num

end final_price_hat_final_price_tie_l163_163287


namespace smallest_n_digits_l163_163140

theorem smallest_n_digits :
  ∃ n : ℕ, (n % 30 = 0) 
          ∧ (∃ k1 : ℕ, n^2 = k1^5) 
          ∧ (∃ k2 : ℕ, n^3 = k2^4) 
          ∧ (Int.log10 n + 1).toNat = 21 :=
sorry

end smallest_n_digits_l163_163140


namespace grid_8_8_eq_3432_l163_163974

-- Definitions for conditions
def grid : ℕ → ℕ → ℕ
| 1, _ => 1
| _, 1 => 1
| i, j => ∑ k in Finset.range (i-1), ∑ m in Finset.range (j-1), grid k.succ m.succ

-- Theorem statement
theorem grid_8_8_eq_3432 : grid 8 8 = 3432 :=
by
  -- Placeholder for proof
  sorry

end grid_8_8_eq_3432_l163_163974


namespace range_of_m_plus_n_l163_163018

noncomputable def f (m n x : ℝ) : ℝ := m * 2^x + x^2 + n * x

theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x : ℝ, f m n x = 0 ∧ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
by
  sorry

end range_of_m_plus_n_l163_163018


namespace passenger_route_optimization_l163_163582

theorem passenger_route_optimization (n : ℕ) (h_n : n = 11) 
  (capital_travel_time : ℕ) (h_capital_travel_time : capital_travel_time = 7)
  (cyclic_road_time : ℕ) (h_cyclic_road_time : cyclic_road_time = 3)
  (initial_transfer_time final_transfer_time : ℝ) 
  (h_initial_transfer_time : initial_transfer_time = 2) 
  (h_final_transfer_time : final_transfer_time = 1.5)
  (t_initial : ℝ) (h_t_initial : t_initial = min (14 + initial_transfer_time) (15 + 4 * initial_transfer_time)) :
  min (14 + final_transfer_time) (15 + 4 * final_transfer_time) ≤ t_initial := 
by
  rw [h_initial_transfer_time, h_final_transfer_time, h_t_initial]
  norm_num
  -- Calculation shows that the minimal time considering the final transfer time is 15.5,
  -- and this is already less than the minimal time when initial transfer time is used,
  -- showing that the passenger will not find a faster route.
  sorry

end passenger_route_optimization_l163_163582


namespace orthocenter_of_triangle_l163_163283

variables (A B C D P Q R : Type) [AffGeom : AffineGeometry ℝ A]

open AffGeom

-- Definitions based on the stated conditions
def isRhombus (ABCD : set A) : Prop :=
  isParallelogram ABCD ∧ ∀ x ∈ ABCD, ∀ y ∈ ABCD, dist x y = dist A B

def isParallelogram_APQR_on_AC (AC : set A) (APQR : set A) : Prop :=
  isParallelogram APQR ∧ A ∈ APQR ∧ P ∈ APQR ∧ Q ∈ APQR ∧ R ∈ APQR ∧ (A ∈ AC ∧ C ∈ AC)

def side_equal (APQR : set A) (AB : ℝ) : Prop :=
  dist A P = AB

-- The main theorem to be proved
theorem orthocenter_of_triangle (ABCD APQR : set A) (AB : ℝ) (B_in_ABC : B ∈ ABCD) :
  isRhombus ABCD →
  isParallelogram_APQR_on_AC AC APQR →
  side_equal APQR AB →
  isOrthocenter B D P Q :=
sorry

end orthocenter_of_triangle_l163_163283


namespace police_officers_on_duty_l163_163519

theorem police_officers_on_duty
  (F : ℕ) (hF : F = 1000)
  (D : ℕ)
  (h1 : 0.25 * F = D / 2) :
  D = 500 :=
by
  sorry

end police_officers_on_duty_l163_163519


namespace horizontal_asymptote_of_f_l163_163685

-- Define the function
def f (x : ℝ) : ℝ := (15 * x^5 + 7 * x^3 + 4 * x^2 + 6 * x + 5) / (3 * x^5 + 2 * x^3 + 7 * x^2 + 4 * x + 2)

-- State the theorem
theorem horizontal_asymptote_of_f :
  tendsto (λ x : ℝ, (15 * x^5 + 7 * x^3 + 4 * x^2 + 6 * x + 5) / (3 * x^5 + 2 * x^3 + 7 * x^2 + 4 * x + 2)) at_top (nhds 5) :=
sorry

end horizontal_asymptote_of_f_l163_163685


namespace fencing_required_l163_163609

variable (L W F : ℝ)
variable (area : ℝ)

-- Conditions
def L_def : L = 20 := by sorry
def area_def : area = 680 := by sorry
def area_formula : area = L * W := by sorry

-- Theorem to prove
theorem fencing_required : F = L + 2 * W := by
  have L_eq : L = 20 := L_def
  have area_eq : area = 680 := area_def
  have area_formula_eq : area = L * W := area_formula
  have W_eq : W = 680 / 20 := by
    calc
      W = 680 / 20 : by sorry
  show F = 20 + 2 * 34 := by sorry
  sorry

end fencing_required_l163_163609


namespace isosceles_triangle_BC_x_value_l163_163106

theorem isosceles_triangle_BC_x_value
  (A B C M : Type)
  (AB AC BC d h x : ℝ)
  (AB_eq_AC : AB = AC)
  (BM : BM = x)
  (BC_length : BC = h)
  (AC_length : AC = d)
  (BM_MA_eq_AC : x + sqrt ((h - x)^2 + d^2) = d)
  : x = h^2 / (2 * (h - d)) :=
sorry

end isosceles_triangle_BC_x_value_l163_163106


namespace polynomial_function_form_l163_163342

noncomputable theory

open Complex Polynomial

/-- A theorem stating that a polynomial with real coefficients,
     satisfying P(z^2) = P(z) * P(z - 1) for all z in ℂ, must be of the form
     (z^2 + z + 1)^k for some positive integer k, excluding constant polynomials. -/
theorem polynomial_function_form
  (P : Polynomial ℝ) (h : ∀ z : ℂ, eval z^2 P = (eval z P) * (eval (z - 1) P))
  (h_non_const : ¬ is_constant P) :
  ∃ k : ℕ, k > 0 ∧ ∀ z : ℂ, eval z P = (eval z (X^2 + X + 1))^k :=
sorry

end polynomial_function_form_l163_163342


namespace coordinates_of_foci_l163_163554

-- Given conditions
def equation_of_hyperbola : Prop := ∃ (x y : ℝ), (x^2 / 4) - (y^2 / 5) = 1

-- The mathematical goal translated into a theorem
theorem coordinates_of_foci (x y : ℝ) (a b c : ℝ) (ha : a^2 = 4) (hb : b^2 = 5) (hc : c^2 = a^2 + b^2) :
  equation_of_hyperbola →
  ((x = 3 ∨ x = -3) ∧ y = 0) :=
sorry

end coordinates_of_foci_l163_163554


namespace option_d_is_correct_l163_163249

theorem option_d_is_correct (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b :=
by
  sorry

end option_d_is_correct_l163_163249


namespace lyle_friends_sandwich_and_juice_l163_163503

theorem lyle_friends_sandwich_and_juice : 
  ∀ (sandwich_cost juice_cost lyle_money : ℝ),
    sandwich_cost = 0.30 → 
    juice_cost = 0.20 → 
    lyle_money = 2.50 → 
    (⌊lyle_money / (sandwich_cost + juice_cost)⌋.toNat - 1) = 4 :=
by
  intros sandwich_cost juice_cost lyle_money hc_sandwich hc_juice hc_money
  have cost_one_set := sandwich_cost + juice_cost
  have number_of_sets := lyle_money / cost_one_set
  have friends := (number_of_sets.toNat - 1)
  have friends_count := 4
  sorry

end lyle_friends_sandwich_and_juice_l163_163503


namespace factorial_division_l163_163243

-- Definitions of factorial used in Lean according to math problem statement.
open Nat

-- Statement of the proof problem in Lean 4.
theorem factorial_division : (12! - 11!) / 10! = 121 := by
  sorry

end factorial_division_l163_163243


namespace smallest_positive_period_of_f_l163_163562

def f (x : ℝ) : ℝ := cos (2 * x) - 2 * sqrt 3 * sin x * cos x

theorem smallest_positive_period_of_f :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', T' > 0 → (∀ x, f (x + T') = f x) → T' ≥ T) :=
sorry

end smallest_positive_period_of_f_l163_163562


namespace area_of_quadrilateral_l163_163796

theorem area_of_quadrilateral (X Y Z M N Q : Type*) [plane_geometry : PlaneGeometry]
  (XY_med : is_median X Y Z M) (YN_med : is_median Y Z X N)
  (intersect_at_Q : medians_intersect_at XY_med YN_med Q)
  (QN_len : length Q N = 2) (QM_len : length Q M = 3)
  (MN_len : length M N = 3.5) :
  area_of MYNZ = 27 := 
sorry

end area_of_quadrilateral_l163_163796


namespace sum_of_two_dice_greater_than_nine_l163_163593

theorem sum_of_two_dice_greater_than_nine : 
  let outcomes := {(i, j) | i ∈ Finset.range 1 7, j ∈ Finset.range 1 7}
  let favorable := outcomes.filter (λ (p : ℕ × ℕ), p.1 + p.2 > 9)
  let probability := (favorable.card : ℚ) / (outcomes.card : ℚ)
  probability = 1 / 6 :=
by
  -- Lean code for the proof goes here
  sorry

end sum_of_two_dice_greater_than_nine_l163_163593


namespace shape_with_congruent_views_is_sphere_l163_163433

def is_congruent_views (shape : Type) : Prop :=
  ∀ (front_view left_view top_view : shape), 
  (front_view = left_view) ∧ (left_view = top_view) ∧ (front_view = top_view)

noncomputable def is_sphere (shape : Type) : Prop := 
  ∀ (s : shape), true -- Placeholder definition for a sphere, as recognizing a sphere is outside Lean's scope

theorem shape_with_congruent_views_is_sphere (shape : Type) :
  is_congruent_views shape → is_sphere shape :=
by
  intro h
  sorry

end shape_with_congruent_views_is_sphere_l163_163433


namespace minimum_value_of_f_l163_163382

noncomputable def f (x a : ℝ) : ℝ := Real.log (x^2 + 2*x + a) / Real.log 2

theorem minimum_value_of_f (a : ℝ) (h: a > 1) (f_max : f 3 a = 5) :
  ∃ x ∈ set.Icc (-3 : ℝ) 3, f x a = 4 :=
begin
  sorry
end

end minimum_value_of_f_l163_163382


namespace min_value_log2_interval_l163_163565

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem min_value_log2_interval : ∀ x ∈ set.Icc 1 2, log2 x ≥ 0 ∧ ∃ y ∈ set.Icc 1 2, log2 y = 0 := by
  sorry

end min_value_log2_interval_l163_163565


namespace even_function_on_neg_interval_l163_163548

theorem even_function_on_neg_interval
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_incr : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 3 → f x₁ ≤ f x₂)
  (h_min : ∀ x : ℝ, 1 ≤ x → x ≤ 3 → 0 ≤ f x) :
  (∀ x : ℝ, -3 ≤ x → x ≤ -1 → 0 ≤ f x) ∧ (∀ x₁ x₂ : ℝ, -3 ≤ x₁ → x₁ < x₂ → x₂ ≤ -1 → f x₁ ≥ f x₂) :=
sorry

end even_function_on_neg_interval_l163_163548


namespace remainder_division_by_24_l163_163188

theorem remainder_division_by_24 (y : ℤ)
  (h1 : 2 + y ≡ 2^3 [ZMOD 2^3])
  (h2 : 4 + y ≡ 2^3 [ZMOD 4^3])
  (h3 : 6 + y ≡ 2^3 [ZMOD 6^3]) :
  y ≡ 6 [ZMOD 24] := by
  sorry

end remainder_division_by_24_l163_163188


namespace sin_cos_difference_l163_163728

theorem sin_cos_difference 
  (α : ℝ)
  (h1 : π / 2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : sin α + cos α = 1 / 5) : 
  sin α - cos α = 7 / 5 :=
sorry

end sin_cos_difference_l163_163728


namespace probability_vertex_A_east_l163_163122

noncomputable def prob_vertex_A_east_of_others (A B C : Point) (angle_A_deg : ℝ) (random_placement : Triangle → Probability) : Probability :=
  random_placement ⟨A, B, C⟩

theorem probability_vertex_A_east (A B C : Point) (angle_A : angle A B C) (random_placement : Triangle → Probability) (h_angleA : angle_A_deg = 40) :
  prob_vertex_A_east_of_others A B C angle_A_deg random_placement = 1 / 3 :=
sorry

end probability_vertex_A_east_l163_163122


namespace possible_values_of_x_l163_163079

theorem possible_values_of_x (x : ℤ) (h : x ∈ {1, 2, x^2}) : x = 0 ∨ x = 2 :=
sorry

end possible_values_of_x_l163_163079


namespace total_monkeys_is_correct_l163_163989

-- Define the parameters
variables (m n : ℕ)

-- Define the conditions as separate definitions
def monkeys_on_n_bicycles : ℕ := 3 * n
def monkeys_on_remaining_bicycles : ℕ := 5 * (m - n)

-- Define the total number of monkeys
def total_monkeys : ℕ := monkeys_on_n_bicycles n + monkeys_on_remaining_bicycles m n

-- State the theorem
theorem total_monkeys_is_correct : total_monkeys m n = 5 * m - 2 * n :=
by
  sorry

end total_monkeys_is_correct_l163_163989


namespace xiao_li_profit_l163_163459

noncomputable def original_price_per_share : ℝ := 21 / 1.05
noncomputable def closing_price_first_day : ℝ := original_price_per_share * 0.94
noncomputable def selling_price_second_day : ℝ := closing_price_first_day * 1.10
noncomputable def total_profit : ℝ := (selling_price_second_day - 21) * 5000

theorem xiao_li_profit :
  total_profit = 600 := sorry

end xiao_li_profit_l163_163459


namespace sum_of_smallest_and_largest_prime_between_1_and_50_l163_163533

theorem sum_of_smallest_and_largest_prime_between_1_and_50 :
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] in
  (primes.head + primes.relaxed_last) = 49 :=
by
  let primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
  show primes.head + primes.relaxed_last = 49
  sorry

end sum_of_smallest_and_largest_prime_between_1_and_50_l163_163533


namespace sum_possible_remainders_l163_163555

theorem sum_possible_remainders :
  let m (n : ℕ) := 2222 * n + 6420 in
  (0 ≤ n ∧ n ≤ 1) →
  let remainder (n : ℕ) := (2222 * n + 6420) % 31 in
  remainder 0 + remainder 1 = 32 :=
by
  intros n h m remainder
  have h₀ : remainder 0 = 24 := sorry
  have h₁ : remainder 1 = 8 := sorry
  rw [h₀, h₁]
  exact add_comm 24 8

end sum_possible_remainders_l163_163555


namespace relationship_K2_l163_163552

theorem relationship_K2 (X Y : Type) (K2 : Type) (h : K2.reflects_strength_of_association X Y) :
  (closer_relationship X Y → larger_value_of K2) := sorry

end relationship_K2_l163_163552


namespace correct_operation_l163_163247

theorem correct_operation (a b : ℤ) : -3 * (a - b) = -3 * a + 3 * b := 
sorry

end correct_operation_l163_163247


namespace maximum_of_f_attain_maximum_of_f_l163_163210

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 - 4

theorem maximum_of_f : ∀ x : ℝ, f x ≤ 0 :=
sorry

theorem attain_maximum_of_f : ∃ x : ℝ, f x = 0 :=
sorry

end maximum_of_f_attain_maximum_of_f_l163_163210


namespace min_faces_of_polyhedron_l163_163564

-- Define a 3-dimensional polyhedron.
def is_polyhedron (n : ℕ) : Prop :=
  ∃ (faces : set (finset ℝ)), faces.size = n ∧ ∀ f ∈ faces, f.card = 3

-- The theorem we want to prove.
theorem min_faces_of_polyhedron : ∃ n, is_polyhedron n → n = 4 :=
sorry

end min_faces_of_polyhedron_l163_163564


namespace train_speed_in_km_per_hr_l163_163954

-- Definitions from the problem conditions
def length_of_train : ℝ := 50
def time_to_cross_pole : ℝ := 3

-- Conversion factor from the problem 
def meter_per_sec_to_km_per_hr : ℝ := 3.6

-- Lean theorem statement based on problem conditions and solution
theorem train_speed_in_km_per_hr : 
  (length_of_train / time_to_cross_pole) * meter_per_sec_to_km_per_hr = 60 := by
  sorry

end train_speed_in_km_per_hr_l163_163954


namespace sixth_number_is_129_l163_163866

theorem sixth_number_is_129
    (A : Fin 11 → ℝ)
    (h1 : (∑ i, A i) / 11 = 60)
    (h2 : (∑ i in Finset.range 6, A i) / 6 = 78)
    (h3 : (∑ i in Finset.Ico 5 11, A i) / 6 = 75) :
    A 5 = 129 := 
sorry

end sixth_number_is_129_l163_163866


namespace percent_defective_units_shipped_l163_163789

theorem percent_defective_units_shipped (h1 : 8 / 100 * 4 / 100 = 32 / 10000) :
  (32 / 10000) * 100 = 0.32 := 
sorry

end percent_defective_units_shipped_l163_163789


namespace no_base_131_cubed_l163_163758

open Nat

theorem no_base_131_cubed (n : ℕ) (k : ℕ) : 
  (4 ≤ n ∧ n ≤ 12) ∧ (1 * n^2 + 3 * n + 1 = k^3) → False := by
  sorry

end no_base_131_cubed_l163_163758


namespace collinear_intersections_of_similar_quadrilaterals_l163_163674

open EuclideanGeometry

-- Definitions/Prerequisites
variables {P P1 P2 : Point}
variables {A B C D A1 B1 C1 D1 A2 B2 C2 D2 : Point}
variables {k1 k2 : ℝ}

-- Conditions
def similar_quadrilateral (A B C D A1 B1 C1 D1 : Point) : Prop :=
  -- Definitions to ensure that the quadrilaterals are similar.
  similar (triangle A B C) (triangle A1 B1 C1) ∧
  similar (triangle B C D) (triangle B1 C1 D1)

def collinear4 (A A1 B2 B : Point) : Prop :=
  collinear A A1 B2 ∧ collinear A1 B2 B

def intersect_diagonals_at (A C B D P : Point) : Prop :=
  exists P, between A P C ∧ between B P D

def collinear_points (P P1 P2 : Point) := collinear P P1 P2

-- Theorem Statement
theorem collinear_intersections_of_similar_quadrilaterals
  (h1 : similar_quadrilateral A B C D A1 B1 C1 D1)
  (h2 : similar_quadrilateral A B C D A2 B2 C2 D2)
  (h3 : collinear4 A A1 B2 B)
  (h4 : collinear4 B B1 C2 C)
  (h5 : collinear4 C C1 D2 D)
  (h6 : collinear4 D D1 A2 A)
  (h7 : intersect_diagonals_at A C B D P)
  (h8 : intersect_diagonals_at A1 C1 B1 D1 P1)
  (h9 : intersect_diagonals_at A2 C2 B2 D2 P2) :
  collinear_points P P1 P2 :=
sorry

end collinear_intersections_of_similar_quadrilaterals_l163_163674


namespace f_one_equals_two_l163_163558

-- Definitions
def monotonically_increasing (f : ℕ → ℕ) : Prop :=
  ∀ {x y : ℕ}, x < y → f(x) < f(y)

def f_nat (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < n → f(n) > 0

-- The main theorem
theorem f_one_equals_two (f : ℕ → ℕ) 
  (mono_inc : monotonically_increasing f)
  (f_nat_inst : f_nat f)
  (f_n_condition : ∀ n : ℕ, 0 < n → f(f(n)) = 3 * n) : 
  f 1 = 2 :=
by
  sorry

end f_one_equals_two_l163_163558


namespace train_speed_l163_163957

-- Define the conditions
def train_length : ℝ := 50 -- Length of the train in meters
def crossing_time : ℝ := 3 -- Time to cross the pole in seconds

-- Define the speed in meters per second and convert it to km/hr
noncomputable def speed_mps : ℝ := train_length / crossing_time
noncomputable def speed_kmph : ℝ := speed_mps * 3.6 -- Conversion factor

-- Theorem statement: Prove that the calculated speed in km/hr is 60 km/hr
theorem train_speed : speed_kmph = 60 := by
  sorry

end train_speed_l163_163957


namespace angle_ABQ_eq_angle_BAS_l163_163896

-- Variables and points on the triangle and its circumcircle
variables (A B C D E P Q R S : Point)
variables (hABCacute : AcuteTriangle A B C)
variables (hTangentA : TangentToCircumcircle A (Circumcircle A B C))
variables (hTangentB : TangentToCircumcircle B (Circumcircle A B C))
variables (hIntersectionC : LineThroughTangent D A C ∧ LineThroughTangent E B C)
variables (hIntersectionP : LineThroughPoint AE B P)
variables (hIntersectionR : LineThroughPoint BD C R)
variables (hPointOnAP : Q ∈ LineThroughPoint AP)
variables (hMidpointS : Midpoint S B R)

-- Define the angles
def angle_ABQ := Angle A B Q
def angle_BAS := Angle B A S
  
theorem angle_ABQ_eq_angle_BAS : angle_ABQ = angle_BAS := sorry

end angle_ABQ_eq_angle_BAS_l163_163896


namespace bicycle_count_l163_163585

theorem bicycle_count (T : ℕ) (B : ℕ) (h1 : T = 14) (h2 : 2 * B + 3 * T = 90) : B = 24 :=
by {
  sorry
}

end bicycle_count_l163_163585


namespace sum_of_first_5_terms_l163_163091

-- Define an arithmetic sequence with third term a_3 = 3
variable {a : ℕ → ℝ}

-- Define the condition
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition: a_3 = 3
def third_term_is_3 : Prop :=
  a 3 = 3

-- We need to prove that the sum of the first 5 terms is 15.
theorem sum_of_first_5_terms (h_seq : is_arithmetic_sequence a) (h_a3 : third_term_is_3) :
  (a 1 + a 2 + a 3 + a 4 + a 5) = 15 :=
sorry

end sum_of_first_5_terms_l163_163091


namespace distance_AB_l163_163046

def C1_parametric (t : ℝ) : ℝ × ℝ :=
  (-2 + cos t, 1 + sin t)

def C2_parametric (θ : ℝ) : ℝ × ℝ :=
  (4 * cos θ, 3 * sin θ)

def line_l (s : ℝ) : ℝ × ℝ :=
  (-4 + (sqrt 2 / 2) * s, (sqrt 2 / 2) * s)

theorem distance_AB : 
  ∃ (s1 s2 : ℝ), 
  (let AB := line_l s1, C1 := C1_parametric s2 in
      s1 + s2 = 3 * sqrt 2 ∧ s1 * s2 = 4 ∧ 
      (abs (s1 - s2) = sqrt 2)) :=
sorry

end distance_AB_l163_163046


namespace total_dolphins_correct_l163_163092

-- Define the initial number of dolphins
def initialDolphins : Nat := 65

-- Define the multiplier for the dolphins joining from the river
def joiningMultiplier : Nat := 3

-- Define the total number of dolphins after joining
def totalDolphins : Nat := initialDolphins + (joiningMultiplier * initialDolphins)

-- Prove that the total number of dolphins is 260
theorem total_dolphins_correct : totalDolphins = 260 := by
  sorry

end total_dolphins_correct_l163_163092


namespace verify_prediction_l163_163231

noncomputable def year_nums : List ℕ := [1, 2, 3, 4, 5]
noncomputable def years : List ℕ := [2018, 2019, 2020, 2021, 2022]
noncomputable def stations : List ℕ := [37, 104, 147, 186, 226]

noncomputable def x_mean : ℝ := 3
noncomputable def y_mean : ℝ := 140
noncomputable def sum_y : ℝ := 700
noncomputable def sum_xy : ℝ := 2560
noncomputable def sqrt_sum_square_y_diff : ℝ := 146.51
noncomputable def sqrt_10 : ℝ := 3.16
noncomputable def b_hat : ℝ := 46
noncomputable def a_hat : ℝ := 2
noncomputable def predict_x : ℕ := 9
noncomputable def predicted_y : ℝ := 416

theorem verify_prediction :
  (∑ i in [0..4], stations[i]) = 700 →
  (∑ i in [0..4], year_nums[i] * stations[i]) = 2560 →
  (sqrt ((∑ i in [0..4], (stations[i] - y_mean) ^ 2))) ≈ 146.51 →
  (sqrt 10) ≈ 3.16 →
  (b_hat = 46) →
  (a_hat = 2) →
  (y_mean - (b_hat * x_mean - a_hat) = 0) → 
  (∑ i in [0..4], (year_nums[i] - x_mean) * (stations[i] - y_mean)) = 460 →
  (46 * predict_x + 2 = predicted_y) :=
by {
  all_goals { sorry }
}

end verify_prediction_l163_163231


namespace dan_money_left_l163_163999

theorem dan_money_left (original_amount spent_amount remaining_amount : ℝ) 
  (h1 : original_amount = 4) 
  (h2 : spent_amount = 3)
  (h3 : remaining_amount = original_amount - spent_amount) : 
  remaining_amount = 1 := 
by {
  rw [h1, h2] at h3,
  exact h3,
}

end dan_money_left_l163_163999


namespace max_value_sqrt_2x_1_sqrt_2y_1_sqrt_2z_1_l163_163389

variable (x y z : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (hz : z > 0)
variable (hxyz : x + y + z = 3)

theorem max_value_sqrt_2x_1_sqrt_2y_1_sqrt_2z_1 :
  (sqrt (2 * x + 1) + sqrt (2 * y + 1) + sqrt (2 * z + 1)) ≤ 3 * sqrt 3 :=
sorry

end max_value_sqrt_2x_1_sqrt_2y_1_sqrt_2z_1_l163_163389


namespace gcd_47_pow6_plus_1_l163_163993

theorem gcd_47_pow6_plus_1 (h_prime : Prime 47) : 
  Nat.gcd (47^6 + 1) (47^6 + 47^3 + 1) = 1 := 
by 
  sorry

end gcd_47_pow6_plus_1_l163_163993


namespace problem_1_problem_2_problem_3_l163_163121

-- Define the sequence a_n and its recurrence relation
def a (n : ℕ) : ℕ :=
  Nat.recOn n 1 (λ n an, 2 * an + 2 ^ n)

-- Define the sequence b_n
def b (n : ℕ) : ℕ :=
  match n with
  | 0     => a 1
  | k + 1 => a (k + 1) / 2 ^ k

-- Define the sequence S_n for the sum of the first n terms of a_n
def S (n : ℕ) : ℕ :=
  (List.range n).sum (λ k, a (k + 1))

-- Problem 1: Prove that b_{n+1} - b_n = 1
theorem problem_1 (n : ℕ) : b (n + 1) - b n = 1 := 
  sorry

-- Problem 2: Prove that the general term formula for a_n is a_n = n * 2^(n-1)
theorem problem_2 (n : ℕ) : a n = n * 2 ^ (n - 1) :=
  sorry

-- Problem 3: Prove that S_n = (n-1) * 2^n + 1
theorem problem_3 (n : ℕ) : S n = (n-1) * 2^n + 1 :=
  sorry

end problem_1_problem_2_problem_3_l163_163121


namespace Papi_Calot_Plants_l163_163163

theorem Papi_Calot_Plants :
  let potato_initial := 4 * 22 + 4 * 25 in
  let potato_total := potato_initial + 18 in
  let carrot_initial :=
    let a1 := 30 in
    let d := 5 in
    let n := 12 in
    (n * (2 * a1 + (n - 1) * d)) / 2 in
  let carrot_total := carrot_initial + 24 in
  let onion_initial :=
    4 * 15 + 4 * 20 + 4 * 25 in
  let onion_total := onion_initial + 12 in
  potato_total = 206 ∧ carrot_total = 714 ∧ onion_total = 252 :=
by
  sorry

end Papi_Calot_Plants_l163_163163


namespace count_integers_with_at_most_three_digits_l163_163070

/-- The number of positive integers less than 50,000 with at most three different digits is 7503. -/
theorem count_integers_with_at_most_three_digits : 
  (finset.filter (λ n : ℕ, n < 50000 ∧ (finset.card (finset.image (λ d : ℕ, (n / 10^d) % 10) (finset.range 5)) ≤ 3)) (finset.range 50000)).card = 7503 := 
sorry

end count_integers_with_at_most_three_digits_l163_163070


namespace total_dolphins_l163_163095

theorem total_dolphins (initial_dolphins : ℕ) (triple_of_initial : ℕ) (final_dolphins : ℕ) 
    (h1 : initial_dolphins = 65) (h2 : triple_of_initial = 3 * initial_dolphins) (h3 : final_dolphins = initial_dolphins + triple_of_initial) : 
    final_dolphins = 260 :=
by
  -- Proof goes here
  sorry

end total_dolphins_l163_163095


namespace dan_spent_on_candy_bars_l163_163325

def dan_money : ℝ := 2
def candy_bar_cost : ℝ := 3
def candy_bars_bought : ℕ := 2

theorem dan_spent_on_candy_bars : (candy_bar_cost * candy_bars_bought = 6) ∧ (6 > dan_money) :=
by
  simp only [candy_bar_cost, candy_bars_bought, dan_money]
  exact and.intro rfl (by linarith)

end dan_spent_on_candy_bars_l163_163325


namespace problem_statement_l163_163713

noncomputable def f (a x : ℝ) : ℝ := 
  (a / (a^2 - 1)) * (a^x - a^(-x))

theorem problem_statement (a : ℝ) (x : ℝ) (m : ℝ) :
  (0 < a ∧ x ∈ ℝ) → 
  (∀ x : ℝ, f a (-x) = -f a x) ∧ 
  (∀ x y : ℝ, x < y → f a x < f a y) ∧
  ((1 - m) ∈ (1 - sqrt 2, 1) → (1 - m^2) ∈ (1 - sqrt 2, 1) → f a (1 - m) + f a (1 - m^2) < 0 → m ∈ (1, sqrt 2)) :=
by intros; sorry

end problem_statement_l163_163713


namespace calculate_roots_l163_163311

noncomputable def cube_root (x : ℝ) := x^(1/3 : ℝ)
noncomputable def square_root (x : ℝ) := x^(1/2 : ℝ)

theorem calculate_roots : cube_root (-8) + square_root 9 = 1 :=
by
  sorry

end calculate_roots_l163_163311


namespace prob_of_selecting_blue_ball_l163_163672

noncomputable def prob_select_ball :=
  let prob_X := 1 / 3
  let prob_Y := 1 / 3
  let prob_Z := 1 / 3
  let prob_blue_X := 7 / 10
  let prob_blue_Y := 1 / 2
  let prob_blue_Z := 2 / 5
  prob_X * prob_blue_X + prob_Y * prob_blue_Y + prob_Z * prob_blue_Z

theorem prob_of_selecting_blue_ball :
  prob_select_ball = 8 / 15 :=
by
  -- Provide the proof here
  sorry

end prob_of_selecting_blue_ball_l163_163672


namespace Jim_new_total_pages_per_week_l163_163812

-- Given conditions
constant initialReadingSpeed : ℕ := 40
constant initialTotalPages : ℕ := 600
constant increasedSpeedFactor : ℕ := 150
constant hoursLessPerWeek : ℕ := 4

-- Prove that Jim reads 660 pages per week now.
theorem Jim_new_total_pages_per_week : 
  let newReadingSpeed := (initialReadingSpeed * increasedSpeedFactor) / 100
  let initialHoursPerWeek := initialTotalPages / initialReadingSpeed
  let newHoursPerWeek := initialHoursPerWeek - hoursLessPerWeek
  newReadingSpeed * newHoursPerWeek = 660 := 
by {
  sorry
}

end Jim_new_total_pages_per_week_l163_163812


namespace sandy_gain_percent_is_10_l163_163614

def total_cost (purchase_price repair_costs : ℕ) := purchase_price + repair_costs

def gain (selling_price total_cost : ℕ) := selling_price - total_cost

def gain_percent (gain total_cost : ℕ) := (gain / total_cost : ℚ) * 100

theorem sandy_gain_percent_is_10 
  (purchase_price : ℕ := 900)
  (repair_costs : ℕ := 300)
  (selling_price : ℕ := 1320) :
  gain_percent (gain selling_price (total_cost purchase_price repair_costs)) 
               (total_cost purchase_price repair_costs) = 10 := 
by
  simp [total_cost, gain, gain_percent]
  sorry

end sandy_gain_percent_is_10_l163_163614


namespace difference_mean_median_l163_163838

/-- Given the distribution of scores among students, with
   5% scoring 60 points,
   35% scoring 75 points,
   30% scoring 82 points,
   15% scoring 88 points,
   and the remaining 15% scoring 92 points,
   prove that the difference between the mean and the median score is 1.15. --/
theorem difference_mean_median :
  let perc60 := 0.05 in
  let perc75 := 0.35 in
  let perc82 := 0.30 in
  let perc88 := 0.15 in
  let perc92 := 1 - (perc60 + perc75 + perc82 + perc88) in
  let total_students := 20 in
  let students60 := total_students * perc60 in
  let students75 := total_students * perc75 in
  let students82 := total_students * perc82 in
  let students88 := total_students * perc88 in
  let students92 := total_students * perc92 in
  let median := 82 in
  let mean := (60 * students60 + 75 * students75 + 82 * students82 + 88 * students88 + 92 * students92) / total_students in
  abs (mean - median) = 1.15 := sorry

end difference_mean_median_l163_163838


namespace boxes_of_apples_l163_163225

theorem boxes_of_apples (n_crates apples_per_crate rotten_apples apples_per_box : ℕ) 
  (h1 : n_crates = 12) 
  (h2 : apples_per_crate = 42)
  (h3: rotten_apples = 4) 
  (h4 : apples_per_box = 10) : 
  (n_crates * apples_per_crate - rotten_apples) / apples_per_box = 50 :=
by
  sorry

end boxes_of_apples_l163_163225


namespace solve_equation_l163_163182

def intPart (x : ℝ) : ℤ := x.toInt
def fracPart (x : ℝ) : ℝ := x - (intPart x)

theorem solve_equation (x : ℝ) (h1 : intPart x = 2) (h2 : fracPart x = 1 / 4) :
  2 * (intPart x : ℝ) * fracPart x = x^2 - (3 / 2) * x - 11 / 16 :=
by
  sorry

end solve_equation_l163_163182


namespace distance_from_origin_l163_163748

def z1 : ℂ := complex.I
def z2 : ℂ := 1 + complex.I

theorem distance_from_origin :
  complex.abs (z1 * z2) = Real.sqrt 2 := 
sorry

end distance_from_origin_l163_163748


namespace rational_root_even_coefficient_l163_163493

theorem rational_root_even_coefficient (a b c : ℤ) (h : a ≠ 0)
  (r : ℚ) (hr : a * r.den^2 * int.ofNat r.num^2 + 
                  b * r.den * int.ofNat r.num + c * r.den^2 = 0) :
  even a ∨ even b ∨ even c :=
sorry

end rational_root_even_coefficient_l163_163493


namespace a_minus_b_7_l163_163198

theorem a_minus_b_7 (a b : ℤ) : (2 * y + a) * (y + b) = 2 * y^2 - 5 * y - 12 → a - b = 7 :=
by
  sorry

end a_minus_b_7_l163_163198


namespace solve_inequality_system_l163_163859

theorem solve_inequality_system
  (x : ℝ)
  (h1 : 3 * (x - 1) < 5 * x + 11)
  (h2 : 2 * x > (9 - x) / 4) :
  x > 1 :=
sorry

end solve_inequality_system_l163_163859


namespace math_problem_l163_163913

noncomputable def a : ℝ := Real.sqrt 6.5
noncomputable def b : ℝ := Real.rpow 2 (1/3)
def c : ℝ := 9.5 - 2^2
def d : ℝ := 7.2
def e : ℝ := 8.7 - 0.3
def f : ℝ := 4.3 + 1
def g : ℝ := 5.3
def h : ℝ := 1 / (3 + 4)

theorem math_problem :
  a * b + c * d + e * 2 * f - g^2 + h ≈ 103.903776014 :=
by sorry

end math_problem_l163_163913


namespace necessary_and_sufficient_condition_l163_163362

-- Definitions of the conditions
def line1 : Prop := ∃ (a : ℝ), ∃ (y : ℝ), a * y = x - 1
def line2 (a : ℝ) : Prop := ∃ (y : ℝ), x + a * y = 2

-- Propositions p and q
def p : Prop := ∀ a : ℝ, line1 -> line2 a
def q : Prop := ∀ a : ℝ, a = -1

-- Statement to be proved
theorem necessary_and_sufficient_condition : (p ↔ q) := 
by
  sorry

end necessary_and_sufficient_condition_l163_163362


namespace intersection_complement_l163_163747

open Set

-- Definitions
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Theorem
theorem intersection_complement :
  A ∩ (U \ B) = {1} :=
sorry

end intersection_complement_l163_163747


namespace moles_of_AgCl_formed_l163_163690

-- Declare the variables representing the moles of HCl and AgNO3
variables (HCl AgNO3 AgCl : ℕ)
-- Define the condition for the chemical reaction in terms of moles
def chemical_reaction := HCl = 3 ∧ AgNO3 = 3 ∧ (HCl = AgNO3)

-- State the theorem to prove the number of moles of AgCl formed
theorem moles_of_AgCl_formed (HCl AgNO3 : ℕ) (h : chemical_reaction HCl AgNO3 AgCl) : AgCl = 3 :=
sorry

end moles_of_AgCl_formed_l163_163690


namespace cubes_with_red_face_l163_163929

theorem cubes_with_red_face :
  let totalCubes := 10 * 10 * 10
  let innerCubes := (10 - 2) * (10 - 2) * (10 - 2)
  let redFaceCubes := totalCubes - innerCubes
  redFaceCubes = 488 :=
by
  let totalCubes := 10 * 10 * 10
  let innerCubes := (10 - 2) * (10 - 2) * (10 - 2)
  let redFaceCubes := totalCubes - innerCubes
  sorry

end cubes_with_red_face_l163_163929


namespace angle_AEB_eq_angle_FDE_l163_163436

-- Given definitions
variable {A B C P Q E D M N F : Type}
variable (triangle_ABC : triangle) 
variable (E_is_midpoint_BC : E = midpoint(B, C))
variable (P_lies_on_AB : P ∈ segment(A, B))
variable (Q_lies_on_AC : Q ∈ segment(A, C))
variable (circumcircle_GammaB : circle)
variable (circumcircle_GammaC : circle)
variable (GammaB_is_circumcircle_BPE : circumcircle_GammaB = circumcircle(⟨B, P, E⟩))
variable (GammaC_is_circumcircle_CQE : circumcircle_GammaC = circumcircle(⟨C, Q, E⟩))
variable (D_is_second_intersection : second_intersection(circumcircle_GammaB, circumcircle_GammaC) = D)
variable (M_is_intersection_AE_GammaB : M = intersection(AE, circumcircle_GammaB))
variable (N_is_intersection_AE_GammaC : N = intersection(AE, circumcircle_GammaC))
variable (F_is_midpoint_MN : F = midpoint(M, N))

-- Proof statement
theorem angle_AEB_eq_angle_FDE :
  angle(A, E, B) = angle(F, D, E) := by
  sorry

end angle_AEB_eq_angle_FDE_l163_163436


namespace proof_y_card_eq_18_l163_163096

open Set

-- Definitions based on the problem conditions
variables {α : Type*} (x y : Set α)

-- Conditions provided in the problem
def cond1 := (x : Finset ℤ).card = 16
def cond2 := (x ∩ y : Finset ℤ).card = 6
def cond3 := ((x \ y) ∪ (y \ x) : Finset ℤ).card = 22

-- Goal: Prove the size of set y
theorem proof_y_card_eq_18 (x y : Finset ℤ) (h1 : (x.card = 16))
  (h2 : ((x ∩ y).card = 6)) (h3 : (((x \ y) ∪ (y \ x)).card = 22)) :
  y.card = 18 :=
sorry

end proof_y_card_eq_18_l163_163096


namespace decimal_equivalent_of_quarter_cubed_l163_163921

theorem decimal_equivalent_of_quarter_cubed :
    (1 / 4 : ℝ) ^ 3 = 0.015625 := 
by
    sorry

end decimal_equivalent_of_quarter_cubed_l163_163921


namespace Ann_age_is_46_l163_163652

theorem Ann_age_is_46
  (a b : ℕ) 
  (h1 : a + b = 72)
  (h2 : b = (a / 3) + 2 * (a - b)) : a = 46 :=
by
  sorry

end Ann_age_is_46_l163_163652


namespace box_made_by_Bellini_l163_163518

-- Let B and C represent Bellini and Cellini respectively.
-- Let S be a predicate indicating that a box is made by one of Bellini's sons.

def made_by_Bellini : Prop := 
  ∀ made_by_son_of_Bellini : Prop, ¬ made_by_son_of_Bellini → made_by_Bellini

theorem box_made_by_Bellini 
  (inscription : ¬ ∃ son_of_Bellini : Prop, son_of_Bellini ∧ made_by_son_of_Bellini)
  (made_by_Bellini : Prop)
  (made_by_Cellini : Prop) 
  (made_by_son_of_Bellini : Prop) 
  (made_by_son_of_Cellini : Prop) :
  made_by_Bellini :=
by
  sorry

end box_made_by_Bellini_l163_163518


namespace count_of_valid_gnollish_sentences_l163_163116

noncomputable def count_valid_gnollish_sentences : ℕ :=
  let words := {"splargh", "glumph", "amr", "blarg"}
  let all_sentences := Finset.pi (Finset.univ : Finset (Fin 3)) (λ _, words)
  let invalid_s1 := all_sentences.filter (λ s, s 0 = "splargh" ∧ s 1 = "glumph")
  let invalid_s2 := all_sentences.filter (λ s, s 1 = "splargh" ∧ s 2 = "glumph")
  let invalid_a1 := all_sentences.filter (λ s, s 0 = "amr" ∧ (s 1 = "glumph" ∨ s 1 = "blarg"))
  let invalid_a2 := all_sentences.filter (λ s, s 1 = "amr" ∧ (s 2 = "glumph" ∨ s 2 = "blarg"))

  all_sentences.card - (invalid_s1.card + invalid_s2.card + invalid_a1.card + invalid_a2.card)

theorem count_of_valid_gnollish_sentences : count_valid_gnollish_sentences = 40 := by
  sorry

end count_of_valid_gnollish_sentences_l163_163116


namespace add_P1_P2_factorize_add_P1_P3_factorize_add_P2_P3_factorize_l163_163413

def P1 := (1 / 2 : ℚ) * x^2 + 2 * x - 1
def P2 := (1 / 2 : ℚ) * x^2 + 4 * x + 1
def P3 := (1 / 2 : ℚ) * x^2 - 2 * x

theorem add_P1_P2_factorize (x : ℚ) : P1 + P2 = x * (x + 6) :=
by
  sorry

theorem add_P1_P3_factorize (x : ℚ) : P1 + P3 = (x + 1) * (x - 1) :=
by
  sorry

theorem add_P2_P3_factorize (x : ℚ) : P2 + P3 = (x + 1) ^ 2 :=
by
  sorry

end add_P1_P2_factorize_add_P1_P3_factorize_add_P2_P3_factorize_l163_163413


namespace range_of_function_x_l163_163216

theorem range_of_function_x :
  ∀ x : ℝ, (1 - x ≥ 0 ∧ x ≠ 0) ↔ (x ≤ 1 ∧ x ≠ 0) :=
by
  intro x
  have h1 : 1 - x ≥ 0 ↔ x ≤ 1 := by
    split
    {
      intro h
      linarith
    }
    {
      intro h
      linarith
    }
  split
  {
    intro h
    cases h
    constructor
    {
      rw h1
      assumption
    }
    {
      assumption
    }
  }
  {
    intro h
    cases h
    constructor
    {
      rw h1
      assumption
    }
    {
      assumption
    }
  }
  sorry

end range_of_function_x_l163_163216


namespace min_moves_to_guess_triple_l163_163814

theorem min_moves_to_guess_triple :
  ∀ (x y z : ℤ), 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 →
  ∃ (B: (ℤ × ℤ × ℤ) → ℤ), ∀ (A_guess : ℤ × ℤ × ℤ),
  let resp := λ (a b c : ℤ), |x + y - a - b| + |y + z - b - c| + |z + x - c - a| in
  -- B can determine (x, y, z) in at most 3 moves
  (∀ B1 B2 B3 : ℤ × ℤ × ℤ,
    B1 ≠ B2 → B2 ≠ B3 →
    exists_unique (x, y, z) : ℤ × ℤ × ℤ,
    B (B1) = resp x y z B1 ∧ B (B2) = resp x y z B2 ∧ B (B3) = resp x y z B3 ) :=
begin
  -- Here is where the actual proof would go
  sorry
end

end min_moves_to_guess_triple_l163_163814


namespace solve_for_y_l163_163542

theorem solve_for_y : ∃ y : ℝ, y = -2 ∧ y^2 + 6 * y + 8 = -(y + 2) * (y + 6) :=
by
  use -2
  sorry

end solve_for_y_l163_163542


namespace find_dividend_l163_163782

theorem find_dividend (dividend divisor quotient : ℕ) 
  (h_sum : dividend + divisor + quotient = 103)
  (h_quotient : quotient = 3)
  (h_divisor : divisor = dividend / quotient) : 
  dividend = 75 :=
by
  rw [h_quotient, h_divisor] at h_sum
  sorry

end find_dividend_l163_163782


namespace determine_a_l163_163406

theorem determine_a (a : ℝ) : 
  (∀ x, sqrt(x + a) ≥ x) ∧ (intervalLength (solutionInterval a) = 4 * |a|) → 
  a = (1 - sqrt(5)) / 8 := 
by 
  sorry

end determine_a_l163_163406


namespace inequality_holds_l163_163733

variable (a : ℝ)

def f (x : ℝ) : ℝ := -a * log (x + 1) + (a + 1) / (x + 1) - a - 1

theorem inequality_holds (h : ∀ (n : ℕ), 1 ≤ n → (1 + 1/(n:ℝ))^(n - a) > real.exp 1) : a ≤ -1/2 :=
by 
    sorry

end inequality_holds_l163_163733


namespace molecular_weight_of_moles_l163_163598

-- Approximate atomic weights
def atomic_weight_N := 14.01
def atomic_weight_O := 16.00

-- Molecular weight of N2O3
def molecular_weight_N2O3 := (2 * atomic_weight_N) + (3 * atomic_weight_O)

-- Given the total molecular weight of some moles of N2O3
def total_molecular_weight : ℝ := 228

-- We aim to prove that the total molecular weight of some moles of N2O3 equals 228 g
theorem molecular_weight_of_moles (h: molecular_weight_N2O3 ≠ 0) :
  total_molecular_weight = 228 := by
  sorry

end molecular_weight_of_moles_l163_163598


namespace parabola_point_exists_l163_163003

theorem parabola_point_exists :
  ∃ (x y : ℝ), y^2 = 12 * x ∧ (real.sqrt ((x - 3)^2 + y^2)) = 8 ∧ x = 5 :=
sorry

end parabola_point_exists_l163_163003


namespace problem_statement_l163_163146

def f (x : ℝ) : ℝ := (4 * x ^ 3 + 2 * x ^ 2 + 3 * x + 7) / (x ^ 2 + x + 5)
def g (x : ℝ) : ℝ := x + 2

theorem problem_statement : f (g (-1)) + g (f (-1)) = 30 / 7 := 
by sorry

end problem_statement_l163_163146


namespace cereal_boxes_l163_163221

variable (x : ℕ)

theorem cereal_boxes:
  (let
    b1 := x,
    b2 := x / 2,
    b3 := x / 2 + 5
  in
    b1 + b2 + b3 = 33) → x = 14 :=
by
  intros h
  let b1 := x
  let b2 := x / 2
  let b3 := x / 2 + 5
  have h1 : b1 + b2 + b3 = 33 := h
  sorry

end cereal_boxes_l163_163221


namespace abs_difference_of_counts_l163_163007

def tau (n : ℕ) : ℕ := if n = 0 then 0 else (Finset.range (n + 1)).filter (λ d, n % d = 0).card

def S (n : ℕ) : ℕ := (Finset.range (n + 1)).sum tau

theorem abs_difference_of_counts (a b : ℕ) :
  (a = (Finset.filter (λ n, S n % 2 = 1) (Finset.range 1001)).card) ∧
  (b = (Finset.filter (λ n, S n % 2 = 0) (Finset.range 1001)).card) →
  |a - b| = 7 := 
by 
  sorry

end abs_difference_of_counts_l163_163007


namespace find_B_coords_l163_163030

-- Define point A and vector a
def A : (ℝ × ℝ) := (1, -3)
def a : (ℝ × ℝ) := (3, 4)

-- Assume B is at coordinates (m, n) and AB = 2a
def B : (ℝ × ℝ) := (7, 5)
def AB : (ℝ × ℝ) := (B.1 - A.1, B.2 - A.2)

-- Prove point B has the correct coordinates
theorem find_B_coords : AB = (2 * a.1, 2 * a.2) → B = (7, 5) :=
by
  intro h
  sorry

end find_B_coords_l163_163030


namespace Jacob_age_is_3_l163_163444

def Phoebe_age : ℕ := sorry
def Rehana_age : ℕ := 25
def Jacob_age (P : ℕ) : ℕ := 3 * P / 5

theorem Jacob_age_is_3 (P : ℕ) (h1 : Rehana_age + 5 = 3 * (P + 5)) (h2 : Rehana_age = 25) (h3 : Jacob_age P = 3) : Jacob_age P = 3 := by {
  sorry
}

end Jacob_age_is_3_l163_163444


namespace Lyle_can_buy_for_his_friends_l163_163507

theorem Lyle_can_buy_for_his_friends
  (cost_sandwich : ℝ) (cost_juice : ℝ) (total_money : ℝ)
  (h1 : cost_sandwich = 0.30)
  (h2 : cost_juice = 0.20)
  (h3 : total_money = 2.50) :
  (total_money / (cost_sandwich + cost_juice)).toNat - 1 = 4 :=
by
  sorry

end Lyle_can_buy_for_his_friends_l163_163507


namespace part1_part2_l163_163734

noncomputable def f (x a k : ℝ) : ℝ :=
  2 * Real.log x + a / x - 2 * Real.log a - k * x / a

theorem part1 (a : ℝ) (ha : 0 < a) (x : ℝ) (hx : 0 < x) : f x a 0 > 0 :=
  sorry

theorem part2 (a : ℝ) (ha : 0 < a) (k : ℝ) (x : ℝ) (hx : 0 < x) (h : f x a k ≥ 0) :
  k ≤ 0 ∧ (∃ y : ℝ, ∀ z : ℝ, f z a k ≥ f y a k) ∧ ∃ c : ℝ, ∀ b : ℝ, 0 < b → f c b k = f c a k :=
  sorry

end part1_part2_l163_163734


namespace problem_statement_max_AB_l163_163378

noncomputable def ellipse_eq (x y : ℝ) : Prop := 
  ∃ (a b : ℝ) (h : a > b ∧ b > 0), 
  a = 2 ∧ b = 1 ∧ (x^2 / a^2 + y^2 / b^2 = 1 ∧ a^2 - b^2 = (3/4) * a^2)

noncomputable def tangency_condition (k t R : ℝ) : Prop := 
  1 < R ∧ R < 2 ∧ R = |t| / sqrt (1 + k^2) ∧ t^2 = R^2 * (1 + k^2)

noncomputable def common_point (k x t : ℝ) : Prop :=
  x ≠ 0 ∧ ((1 + 4*k^2)*x^2 + 8*k*t*x + 4*t^2 - 4 = 0)

theorem problem_statement (k t R : ℝ) (h_tang : tangency_condition k t R)
  (h_common : ∀ x, common_point k x t → false) : k^2 = (R^2 - 1) / (4 - R^2) := 
by
  sorry

theorem max_AB (R : ℝ) (hR : 1 < R ∧ R < 2) (hk2 : ∃ k, k^2 = (R^2 - 1) / (4 - R^2))
  (x₀ y₀ : ℝ) (hxy : ellipse_eq x₀ y₀) : |OB|^2 ≤ 1
:= by 
  sorry

end problem_statement_max_AB_l163_163378


namespace ratio_of_areas_l163_163457

theorem ratio_of_areas
  (PQ QR RP : ℝ)
  (PQ_pos : 0 < PQ)
  (QR_pos : 0 < QR)
  (RP_pos : 0 < RP)
  (s t u : ℝ)
  (s_pos : 0 < s)
  (t_pos : 0 < t)
  (u_pos : 0 < u)
  (h1 : s + t + u = 3 / 4)
  (h2 : s^2 + t^2 + u^2 = 1 / 2)
  : (1 - (s * (1 - u) + t * (1 - s) + u * (1 - t))) = 7 / 32 := by
  sorry

end ratio_of_areas_l163_163457


namespace double_root_values_l163_163695

theorem double_root_values (c : ℝ) :
  (∃ a : ℝ, (a^5 - 5 * a + c = 0) ∧ (5 * a^4 - 5 = 0)) ↔ (c = 4 ∨ c = -4) :=
by
  sorry

end double_root_values_l163_163695


namespace min_omega_l163_163051

-- Given the conditions
def f (ω x : ℝ) : ℝ := Real.sin(2 * ω * x)
def condition (x ω : ℝ) : Prop := 
f ω x = f ω (x - Real.pi / 4)

-- Prove that the minimum value of ω for which the condition holds is 4
theorem min_omega (ω : ℝ) (h : 0 < ω) : (∀ x, condition x ω) → ω = 4 := 
sorry

end min_omega_l163_163051


namespace min_dist_tangent_to_circle_l163_163334

/-!
# Tangent to a Circle Problem
-/ 

open Real
open Function
open Set

variables {P M : EuclideanSpace ℝ (Fin 2)} {O : EuclideanSpace ℝ (Fin 2)} 

theorem min_dist_tangent_to_circle (h : dist (P, O) = dist (P, M)) (circle_eq : ∀ (P : EuclideanSpace ℝ (Fin 2)), (∥P + ![1, -2]∥ ^ 2 = 1)) : 
  dist (P, O) = 2 * sqrt 5 / 5 :=
by
  -- Proof skipped
  sorry

end min_dist_tangent_to_circle_l163_163334


namespace find_integer_l163_163297

-- Definition of the given conditions
def conditions (x : ℤ) (r : ℤ) : Prop :=
  (0 ≤ r ∧ r < 7) ∧ ((x - 77) * 8 = 259 + r)

-- Statement of the theorem to be proved
theorem find_integer : ∃ x : ℤ, ∃ r : ℤ, conditions x r ∧ (x = 110) :=
by
  sorry

end find_integer_l163_163297


namespace jimmy_eats_7_cookies_l163_163675

def cookies_and_calories (c: ℕ) : Prop :=
  50 * c + 150 = 500

theorem jimmy_eats_7_cookies : cookies_and_calories 7 :=
by {
  -- This would be where the proof steps go, but we replace it with:
  sorry
}

end jimmy_eats_7_cookies_l163_163675


namespace car_purchase_amount_l163_163843

theorem car_purchase_amount 
  (X : ℝ)
  (repair_cost : ℝ := 12000)
  (selling_price : ℝ := 80000)
  (profit_percent : ℝ := 0.4035)
  (total_cost : ℝ := X + repair_cost)
  (profit : ℝ := selling_price - total_cost) :
  profit = profit_percent * X ↔ X ≈ 48425.44 := 
by 
  sorry

end car_purchase_amount_l163_163843


namespace probability_within_sphere_correct_l163_163946

noncomputable def probability_within_sphere : ℝ :=
  let cube_volume := (2 : ℝ) * (2 : ℝ) * (2 : ℝ)
  let sphere_volume := (4 * Real.pi / 3) * (0.5) ^ 3
  sphere_volume / cube_volume

theorem probability_within_sphere_correct (x y z : ℝ) 
  (hx1 : -1 ≤ x) (hx2 : x ≤ 1) 
  (hy1 : -1 ≤ y) (hy2 : y ≤ 1) 
  (hz1 : -1 ≤ z) (hz2 : z ≤ 1) 
  (hx_sq : x^2 ≤ 0.5) 
  (hxyz : x^2 + y^2 + z^2 ≤ 0.25) : 
  probability_within_sphere = Real.pi / 48 :=
by
  sorry

end probability_within_sphere_correct_l163_163946


namespace g_neg_2_value_l163_163732

def f (x : ℝ) : ℝ :=
  if x >= 0 then 
    real.log (x^2 + 1) / real.log 3 -- log base 3
  else 
    g x + 3*x

axiom h_odd : ∀ x : ℝ, f (-x) = -f x

theorem g_neg_2_value : g (-2) = 6 - (real.log (5) / real.log 3) :=
begin
  sorry
end

end g_neg_2_value_l163_163732


namespace purchasing_plans_count_l163_163882

theorem purchasing_plans_count :
  ∃ (x y : ℕ), (4 * y + 6 * x = 40)  ∧ (y ≥ 0) ∧ (x ≥ 0) ∧ (∃! (x y : ℕ), (4 * y + 6 * x = 40)  ∧ (y ≥ 0) ∧ (x ≥ 0)) := sorry

end purchasing_plans_count_l163_163882


namespace tournament_min_participants_l163_163898

theorem tournament_min_participants (a : ℕ → ℕ) (h₁ : ∀ k : ℕ, a (k + 1) < a k) (h₂ : ∀ k : ℕ, a k > 0) :
  ∃ n : ℕ, n = 13 ∧ (∀ (x : ℕ → ℕ), (∀ i, i < n → x i ∈ {a j | j ∈ finset.range n}) →
  (∃ y z, y ≠ z ∧ (∃ j, x y > x z ∧ j < 12 ∧ speed_up j x y = speed_up j x z))) :=
begin
  sorry
end

end tournament_min_participants_l163_163898


namespace remainder_T2015_mod_12_eq_8_l163_163355

-- Define sequences of length n consisting of the letters A and B,
-- with no more than two A's in a row and no more than two B's in a row
def T : ℕ → ℕ :=
  sorry  -- Definition for T(n) must follow the given rules

-- Theorem to prove that T(2015) modulo 12 equals 8
theorem remainder_T2015_mod_12_eq_8 :
  (T 2015) % 12 = 8 :=
  sorry

end remainder_T2015_mod_12_eq_8_l163_163355


namespace rulers_added_initially_46_finally_71_l163_163901

theorem rulers_added_initially_46_finally_71 : 
  ∀ (initial final added : ℕ), initial = 46 → final = 71 → added = final - initial → added = 25 :=
by
  intros initial final added h_initial h_final h_added
  rw [h_initial, h_final] at h_added
  exact h_added

end rulers_added_initially_46_finally_71_l163_163901


namespace range_of_t_l163_163485

theorem range_of_t {a b t : ℝ} (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = 2 * a * x ^ 2 + 2 * b * x)
  (h_exists : ∃ x0, x0 ∈ Ioo 0 t ∧ ∀ (a b : ℝ), (a ≠ 0 ∧ b ≠ 0) → f x0 = a + b) :
  t ∈ Ioi (1 : ℝ) := sorry

end range_of_t_l163_163485


namespace final_probability_l163_163081

def total_cards := 52
def kings := 4
def aces := 4
def chosen_cards := 3

namespace probability

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

def prob_three_kings : ℚ :=
  (4 / 52) * (3 / 51) * (2 / 50)

def prob_exactly_two_aces : ℚ :=
  (choose 4 2 * choose 48 1) / choose 52 3

def prob_exactly_three_aces : ℚ :=
  (choose 4 3) / choose 52 3

def prob_at_least_two_aces : ℚ :=
  prob_exactly_two_aces + prob_exactly_three_aces

def prob_three_kings_or_two_aces : ℚ :=
  prob_three_kings + prob_at_least_two_aces

theorem final_probability :
  prob_three_kings_or_two_aces = 6 / 425 :=
by
  sorry

end probability

end final_probability_l163_163081


namespace subcommittees_count_l163_163357

theorem subcommittees_count (n : ℕ) (h : n = 7) : ∃ k, k = 6 ∧ (finset.card {x : finset (fin n) | (0 ∈ x ∧ finset.card x = 2)} = k) :=
by {
  use 6,
  split,
  { refl, },
  { sorry, }
}

end subcommittees_count_l163_163357


namespace range_of_m_l163_163722

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, 2^x - m + 1 > 0
def proposition_q (m : ℝ) : Prop := 5 - 2*m > 1

theorem range_of_m (m : ℝ) (hp : proposition_p m) (hq : proposition_q m) : m ≤ 1 :=
sorry

end range_of_m_l163_163722


namespace initial_men_in_hostel_l163_163278

theorem initial_men_in_hostel (x : ℕ) (h1 : 36 * x = 45 * (x - 50)) : x = 250 := 
  sorry

end initial_men_in_hostel_l163_163278


namespace closest_vector_t_value_l163_163001

open Real EuclideanSpace

def v (t : ℝ) : EuclideanSpace ℝ (Fin 4) :=
  ![3 + 8 * t, -2 + 4 * t, 1 - 2 * t, -4 + 6 * t]

def a : EuclideanSpace ℝ (Fin 4) :=
  ![5, 3, 7, 2]

def direction_vector : EuclideanSpace ℝ (Fin 4) :=
  ![8, 4, -2, 6]

theorem closest_vector_t_value : ∃ t : ℝ, (t = 1 / 2) :=
  ∃ t : ℝ, (dot_product (v t - a) direction_vector = 0) ∧ (t = 1 / 2)

end closest_vector_t_value_l163_163001


namespace find_coeff_b_find_coeffs_ac_f_increasing_find_m_plus_n_range_of_k_l163_163395

-- Define the given conditions
def f (a b c x : ℝ) := a * x ^ 3 + b * x ^ 2 + c * x

-- Prove that b = 0
theorem find_coeff_b (a c : ℝ) : 
  (∀ x : ℝ, f a 0 c (-x) = -f a 0 c x) → 
  f a 0 c 1 = 3 → 
  f a 0 c 2 = 12 → 
  b = 0 := 
sorry

-- Prove the values of a and c
theorem find_coeffs_ac (a c : ℝ) :
  (f a 0 c 1 = 3) ∧ (f a 0 c 2 = 12) → 
  a = 1 ∧ c = 2 := 
sorry

-- Define the function f after coefficients are known
noncomputable def f_fixed (x : ℝ) := x ^ 3 + 2 * x

-- Prove that f(x) is an increasing function
theorem f_increasing : 
  (∀ x : ℝ, derivative f_fixed x > 0) → 
  ∀ x y : ℝ, x < y → f_fixed x < f_fixed y := 
sorry

-- Prove that m + n = 2
theorem find_m_plus_n (m n : ℝ) : 
  (m ^ 3 - 3 * m ^ 2 + 5 * m = 5) ∧ 
  (n ^ 3 - 3 * n ^ 2 + 5 * n = 1) → 
  m + n = 2 := 
sorry

-- Define the function g
def g (k x : ℝ) := x ^ 2 + k * x + 2 * k - 4

-- Prove the range of k
theorem range_of_k (k : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 1 → f_fixed (x ^ 2 - 4) + f_fixed (k * x + 2 * k) < 0) → 
  k ≤ 1 := 
sorry

end find_coeff_b_find_coeffs_ac_f_increasing_find_m_plus_n_range_of_k_l163_163395


namespace compute_abc_l163_163484

theorem compute_abc (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h_sum : a + b + c = 30) (h_frac : (1 : ℚ) / a + 1 / b + 1 / c + 450 / (a * b * c) = 1) : a * b * c = 1920 :=
by sorry

end compute_abc_l163_163484


namespace num_solutions_cosine_eq_l163_163330

theorem num_solutions_cosine_eq :
  ∃ S : Finset ℝ, S.card = 5 ∧ ∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧
    3 * Real.cos(x)^4 - 6 * Real.cos(x)^3 + 4 * Real.cos(x)^2 - 1 = 0 :=
by
  sorry

end num_solutions_cosine_eq_l163_163330


namespace find_lambda_l163_163008

theorem find_lambda
  (ω : ℂ) (λ : ℝ) (hω : abs ω = 3)
  (h_eq_triangle : ∃ λ, ∀ z, z ∈ {ω, ω^2, λ * ω} → z ∈ set_of (z ≠ z + 1)) -- this represents the equilateral triangle condition
  (hλ_pos : λ > 0) :
  λ = (1 + Real.sqrt 33) / 2 :=
sorry

end find_lambda_l163_163008


namespace balls_into_boxes_l163_163165

def number_of_ways_to_place_balls_in_boxes : Nat :=
  36

theorem balls_into_boxes :
  ∃ (f : Fin 4 → Fin 3), 
    (∀ b : Fin 3, ∃ i : Fin 4, f i = b) ∧
    (∑ b : Fin 3, (Finset.univ.filter (λ i, f i = b)).card = 4) :=
  sorry

end balls_into_boxes_l163_163165


namespace game_is_fair_l163_163233

variable (outcomes : List (Bool × Bool)) 
variable (win_A : (Bool × Bool) → Bool)
variable (win_B : (Bool × Bool) → Bool)

-- Define the possible outcomes from flipping two coins
def possible_outcomes : List (Bool × Bool) := [
  (tt, tt), -- (Heads, Heads)
  (ff, ff), -- (Tails, Tails)
  (tt, ff), -- (Heads, Tails)
  (ff, tt)  -- (Tails, Heads)
]

-- Define winning conditions for player A
def wins_A (outcome : (Bool × Bool)) : Bool :=
  outcome.1 = outcome.2

-- Define winning conditions for player B
def wins_B (outcome : (Bool × Bool)) : Bool :=
  outcome.1 ≠ outcome.2

-- Define the main theorem to prove the game is fair
theorem game_is_fair : 
  ( (possible_outcomes.filter wins_A).length = (possible_outcomes.filter wins_B).length ) :=
by 
  sorry

end game_is_fair_l163_163233


namespace last_letter_of_95th_permutation_of_ABCDE_is_B_l163_163189

theorem last_letter_of_95th_permutation_of_ABCDE_is_B :
  let perms := list.perm.permutations ['A', 'B', 'C', 'D', 'E']
  (list.nth (perms.qsort (λ ls1 ls2, (ls1 < ls2)) 94)).get_last sorry :=
  sorry

end last_letter_of_95th_permutation_of_ABCDE_is_B_l163_163189


namespace collinearity_equiv_l163_163301

variables (P A B C D E F M : Point)
variables (O : Circle)
variables (PA PBC : Line)

-- Conditions
variables (tangent_cond : Tangent_to_circle PA O A)
variables (secant_cond : Secant_to_circle PBC O)
variables (chords_intersect : Intersects_at AD BC E)
variables (angle_cond : ∠ F B D = ∠ F E D)

-- Required Theorem Statement
theorem collinearity_equiv (collinear_PFD : Collinear P F D) : Collinear M B D ↔ Collinear P F D :=
sorry

end collinearity_equiv_l163_163301


namespace min_airlines_needed_l163_163900

theorem min_airlines_needed 
  (towns : Finset ℕ) 
  (h_towns : towns.card = 21)
  (flights : Π (a : Finset ℕ), a.card = 5 → Finset (Finset ℕ))
  (h_flight : ∀ {a : Finset ℕ} (ha : a.card = 5), (flights a ha).card = 10):
  ∃ (n : ℕ), n = 21 :=
sorry

end min_airlines_needed_l163_163900


namespace bottle_caps_shared_l163_163510

theorem bottle_caps_shared (initial_bottle_caps : ℕ) (remaining_bottle_caps : ℕ) : 
  initial_bottle_caps = 51 → remaining_bottle_caps = 15 → initial_bottle_caps - remaining_bottle_caps = 36 :=
by
  intros h1 h2
  rw h1
  rw h2
  simp

end bottle_caps_shared_l163_163510


namespace Lyle_friends_sandwich_juice_l163_163502

/-- 
Lyle wants to buy himself and his friends a sandwich and a pack of juice. 
A sandwich costs $0.30 while a pack of juice costs $0.20. Given Lyle has $2.50, 
prove that he can buy sandwiches and juice for 4 of his friends.
-/
theorem Lyle_friends_sandwich_juice :
  let sandwich_cost := 0.30
  let juice_cost := 0.20
  let total_money := 2.50
  let total_cost_one_set := sandwich_cost + juice_cost
  let total_sets := total_money / total_cost_one_set
  total_sets - 1 = 4 :=
by
  sorry

end Lyle_friends_sandwich_juice_l163_163502


namespace inverse_variation_with_constant_l163_163426

theorem inverse_variation_with_constant
  (k : ℝ)
  (x y : ℝ)
  (h1 : y = (3 * k) / x)
  (h2 : x = 4)
  (h3 : y = 8) :
  (y = (3 * (32 / 3)) / -16) := by
sorry

end inverse_variation_with_constant_l163_163426


namespace opposite_of_fraction_reciprocal_of_fraction_absolute_value_of_fraction_l163_163691

def improper_fraction : ℚ := -4/3

theorem opposite_of_fraction : -improper_fraction = 4/3 :=
by sorry

theorem reciprocal_of_fraction : (improper_fraction⁻¹) = -3/4 :=
by sorry

theorem absolute_value_of_fraction : |improper_fraction| = 4/3 :=
by sorry

end opposite_of_fraction_reciprocal_of_fraction_absolute_value_of_fraction_l163_163691


namespace hot_dogs_left_over_l163_163763

theorem hot_dogs_left_over : 25197629 % 6 = 5 := 
sorry

end hot_dogs_left_over_l163_163763


namespace pq_square_eq_169_div_4_l163_163488

-- Defining the quadratic equation and the condition on solutions p and q.
def quadratic_eq (x : ℚ) : Prop := 2 * x^2 + 7 * x - 15 = 0

-- Defining the specific solutions p and q.
def p : ℚ := 3 / 2
def q : ℚ := -5

-- The main theorem stating that (p - q)^2 = 169 / 4 given the conditions.
theorem pq_square_eq_169_div_4 (hp : quadratic_eq p) (hq : quadratic_eq q) : (p - q) ^ 2 = 169 / 4 :=
by
  -- Proof is omitted using sorry
  sorry

end pq_square_eq_169_div_4_l163_163488


namespace program_list_arrangements_count_l163_163653

-- Definitions and conditions
def solo_programs := 5
def chorus_programs := 3
def total_programs := solo_programs + chorus_programs

-- The theorem to prove
theorem program_list_arrangements_count :
    let spaces := solo_programs + 1 in
    let solo_arrangements := Nat.factorial solo_programs in
    let chorus_arrangements := (spaces - 1) * (spaces - 2) * (spaces - 3) in
    solo_arrangements * chorus_arrangements = 7200 :=
by
  sorry

end program_list_arrangements_count_l163_163653


namespace smallest_positive_period_l163_163560

noncomputable def f (x : ℝ) : ℝ := cos (2 * x) - 2 * real.sqrt 3 * sin x * cos x

theorem smallest_positive_period : ∀ x : ℝ, f (x + π) = f x := by
  sorry

end smallest_positive_period_l163_163560


namespace no_one_collects_everyone_collects_A_and_D_not_B_l163_163275

-- Define the types of stamps as a datatype
inductive StampType
| pre1900British : StampType
| post1900British : StampType
| foreign : StampType
| foreignSpecial : StampType
| britishSpecial : StampType

-- Define the collectors and their stamp collections
structure Collector where
  collects : StampType → Prop

-- Definitions of the collectors' stamp collections
def A : Collector := {
  collects := λ s, match s with
  | StampType.pre1900British => true
  | StampType.foreign => true
  | _ => false
}

def B : Collector := {
  collects := λ s, match s with
  | StampType.post1900British => true
  | StampType.foreignSpecial => true
  | _ => false
}

def C : Collector := {
  collects := λ s, match s with
  | StampType.foreignSpecial => true
  | StampType.britishSpecial => true
  | _ => false
}

def D : Collector := {
  collects := λ s, match s with
  | StampType.post1900British => true
  | StampType.britishSpecial => true
  | _ => false
}

-- Lean theorem statements for the three parts of the problem
theorem no_one_collects :
  ¬ (A.collects StampType.post1900British ∨
     B.collects StampType.post1900British ∨
     C.collects StampType.post1900British ∨
     D.collects StampType.post1900British) := sorry

theorem everyone_collects (s : StampType) :
  A.collects s ∧ B.collects s ∧ C.collects s ∧ D.collects s → false := sorry

theorem A_and_D_not_B :
  (A.collects StampType.pre1900British ∧ D.collects StampType.pre1900British ∧ ¬ B.collects StampType.pre1900British) ∨
  (A.collects StampType.foreign ∧ D.collects StampType.foreign ∧ ¬ B.collects StampType.foreignSpecial) := sorry

end no_one_collects_everyone_collects_A_and_D_not_B_l163_163275


namespace proof_g_l163_163677

variable (x : ℤ)
def g (x : ℤ) := -7 * x ^ 4 - 5 * x ^ 3 + 6 * x ^ 2 - 9

theorem proof_g:
  7 * x ^ 4 - 4 * x ^ 2 + 2 + g x = -5 * x ^ 3 + 2 * x ^ 2 - 7 :=
by
  sorry

end proof_g_l163_163677


namespace infinitely_many_solutions_b_value_l163_163682

theorem infinitely_many_solutions_b_value :
  ∀ (x : ℝ) (b : ℝ), (5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := 
by
  intro x b
  sorry

end infinitely_many_solutions_b_value_l163_163682


namespace unique_real_solution_in_interval_l163_163622

noncomputable def equation := λ x : ℝ, 2 * Real.log x + x - 4

theorem unique_real_solution_in_interval :
  ∃! x ∈ set.Ioo 1 Real.exp 1, equation x = 0 :=
by
  sorry

end unique_real_solution_in_interval_l163_163622


namespace simplify_fraction_l163_163848

theorem simplify_fraction (x : ℝ) (h : x = real.sqrt 3 + 1) : 
  (1 - (1 / (x + 2))) / ((x^2 - 1) / (x + 2)) = real.sqrt 3 / 3 :=
  sorry

end simplify_fraction_l163_163848


namespace solution_x_l163_163623

noncomputable def find_x (x : ℝ) : Prop :=
  (Real.log (x^4))^2 = (Real.log x)^6

theorem solution_x (x : ℝ) : find_x x ↔ (x = 1 ∨ x = Real.exp 2 ∨ x = Real.exp (-2)) :=
sorry

end solution_x_l163_163623


namespace AO_perpendicular_BE_l163_163454

-- Define the required angles and conditions in Lean
variables {A B C D E O : Type}
variables (hConvex : ConvexPentagon A B C D E)
variables (hAngleB : ∠ABC = 90°) (hAngleE : ∠EAD = 90°)
variables (hEqualAngles : ∠BAC = ∠EAD)
variables (hIntersect : IntersectsDiagonals A B C D E O)

-- Main theorem statement
theorem AO_perpendicular_BE (hConvex : ConvexPentagon A B C D E)
  (hAngleB : ∠ABC = 90°) (hAngleE : ∠EAD = 90°)
  (hEqualAngles : ∠BAC = ∠EAD) (hIntersect : IntersectsDiagonals A B C D E O) :
  Perpendicular (Line A O) (Line B E) :=
by
  sorry

end AO_perpendicular_BE_l163_163454


namespace find_theta_l163_163423

noncomputable def theta : ℝ := 25

theorem find_theta (θ : ℝ) (h1 : 0 < θ) (h2 : θ < 90) :
  (√3 * real.sin (20 * real.pi / 180) = real.cos (θ * real.pi / 180) - real.sin (θ * real.pi / 180)) ↔ θ = theta :=
by
  sorry

end find_theta_l163_163423


namespace contrapositive_l163_163553

/-- Let m be a real number and consider the quadratic equation x^2 + x - m = 0. -/
variable (m : ℝ)

/-- The hypothesis that m > 0 -/
def hyp (m : ℝ) : Prop := m > 0

/-- The conclusion that the equation x^2 + x - m has real roots -/
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, x^2 + x - m = 0

/-- The contrapositive of the proposition "If m > 0, then the equation x^2 + x - m = 0 has real roots" -/
theorem contrapositive (m : ℝ) : ¬ has_real_roots m → m ≤ 0 :=
by
  sorry

end contrapositive_l163_163553


namespace salt_concentration_solution_l163_163588

theorem salt_concentration_solution
  (x y z : ℕ)
  (h1 : x + y + z = 30)
  (h2 : 2 * x + 3 * y = 35)
  (h3 : 3 * y + 2 * z = 45) :
  x = 10 ∧ y = 5 ∧ z = 15 := by
  sorry

end salt_concentration_solution_l163_163588


namespace farmer_feed_full_price_l163_163630

theorem farmer_feed_full_price
  (total_spent : ℕ)
  (chicken_feed_discount_percent : ℕ)
  (chicken_feed_percent : ℕ)
  (goat_feed_percent : ℕ)
  (total_spent_val : total_spent = 35)
  (chicken_feed_discount_percent_val : chicken_feed_discount_percent = 50)
  (chicken_feed_percent_val : chicken_feed_percent = 40)
  (goat_feed_percent_val : goat_feed_percent = 60) :
  (total_spent * chicken_feed_percent / 100 * 2) + (total_spent * goat_feed_percent / 100) = 49 := 
by
  -- Placeholder for proof.
  sorry

end farmer_feed_full_price_l163_163630


namespace solve_equation_l163_163895

theorem solve_equation (x : ℝ) :
  ((x - 2)^2 - 4 = 0) ↔ (x = 4 ∨ x = 0) :=
by
  sorry

end solve_equation_l163_163895


namespace discount_percentage_correct_l163_163173

def original_price : ℝ := 10
def price_paid : ℝ := 9

def discount_amount : ℝ := original_price - price_paid
def discount_percentage : ℝ := (discount_amount / original_price) * 100

theorem discount_percentage_correct : discount_percentage = 10 :=
by
  unfold discount_percentage discount_amount original_price price_paid
  norm_num

end discount_percentage_correct_l163_163173


namespace range_of_a_l163_163745

theorem range_of_a (a : ℝ) (h : ∃ α β : ℝ, (α + β = -(a^2 - 1)) ∧ (α * β = a - 2) ∧ (1 < α ∧ β < 1) ∨ (α < 1 ∧ 1 < β)) :
  -2 < a ∧ a < 1 :=
sorry

end range_of_a_l163_163745


namespace total_spent_is_140_l163_163326

-- Define the original prices and discounts
def original_price_shoes : ℕ := 50
def original_price_dress : ℕ := 100
def discount_shoes : ℕ := 40
def discount_dress : ℕ := 20

-- Define the number of items purchased
def number_of_shoes : ℕ := 2
def number_of_dresses : ℕ := 1

-- Define the calculation of discounted prices
def discounted_price_shoes (original_price : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  (original_price * quantity * (100 - discount)) / 100

def discounted_price_dress (original_price : ℕ) (discount : ℕ) (quantity : ℕ) : ℕ :=
  (original_price * quantity * (100 - discount)) / 100

-- Define the total cost calculation
def total_cost : ℕ :=
  discounted_price_shoes original_price_shoes discount_shoes number_of_shoes +
  discounted_price_dress original_price_dress discount_dress number_of_dresses

-- The theorem to prove
theorem total_spent_is_140 : total_cost = 140 := by
  sorry

end total_spent_is_140_l163_163326


namespace factorization_a_minus_b_l163_163201

-- Define the problem in Lean 4
theorem factorization_a_minus_b (a b : ℤ) : 
  (∀ y : ℝ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b))
  → a - b = 7 := 
by 
  sorry

end factorization_a_minus_b_l163_163201


namespace cube_pass_through_another_cube_l163_163169

-- Define conditions for the problem
variable (a : ℝ) (cube : Type) [has_edge_length cube a] [has_space_diagonal cube (a * Real.sqrt 3)] [has_projection_onto_plane cube regular_hexagon]

-- Define the property of the hexagon as mentioned in the conditions
def regular_hexagon (side_length : ℝ) :=
  side_length = a * Real.sqrt 6 / 3 ∧ has_inradius regular_hexagon (a * Real.sqrt 2 / 2)

-- Formalize the statement
theorem cube_pass_through_another_cube (a : ℝ) (cube_small cube_big : cube) :
  (∃ hole : regular_hexagon, hole ⊆ cube_big) → (cube_small can_pass_through hole) :=
  sorry

end cube_pass_through_another_cube_l163_163169


namespace measure_angle_ADF_l163_163115

noncomputable def circle_O (O A B E : Point) (r : ℝ) : Prop :=
  let C := sorry, -- C as an arbitrary point for now
  let line_BE := line_through B E,
  let line_CA := tangent_line_through O A,
  let F : Point := intersection (bisector_through (angle A C B)) (line_through A E),
  let D : Point := intersection (bisector_through (angle A C B)) (line_through A B) in
  ∠A D F = 67.5°

theorem measure_angle_ADF (O A B E : Point) (r : ℝ) (H1 : circle O A B E r)
  (H2 : C ∈ extension_line B E) (H3 : tangent_at A O C A)
  (H4 : bisects D C (∠A C B)) (H5 : D ∈ line_through A B) (H6 : F ∈ line_through A E) : 
  ∠A D F = 67.5° :=
sorry

end measure_angle_ADF_l163_163115


namespace profit_percentage_is_twenty_percent_l163_163607

def selling_price : ℕ := 900
def profit : ℕ := 150
def cost_price : ℕ := selling_price - profit
def profit_percentage : ℕ := (profit * 100) / cost_price

theorem profit_percentage_is_twenty_percent : profit_percentage = 20 := by
  sorry

end profit_percentage_is_twenty_percent_l163_163607


namespace dartboard_probability_odd_score_l163_163634

def dartboard :=
{ radius_inner : ℝ := 4,
  radius_outer : ℝ := 8,
  values_inner : list ℕ := [3, 5, 5],
  values_outer : list ℕ := [4, 3, 3] }

def area (radius : ℝ) : ℝ := real.pi * radius^2

def prob_of_point_value (radius_inner radius_outer : ℝ) (values_inner values_outer : list ℕ) (value : ℕ) : ℝ :=
  let area_inner := area radius_inner
  let area_outer := area radius_outer - area_inner
  let regions_inner := values_inner.filter (λ v, v = value)
  let regions_outer := values_outer.filter (λ v, v = value)
  let area_value_inner := area_inner * (regions_inner.length / values_inner.length)
  let area_value_outer := area_outer * (regions_outer.length / values_outer.length)
  let total_area := area radius_outer
  (area_value_inner + area_value_outer) / total_area

def prob_odd_score (dartboard : { radius_inner : ℝ, radius_outer : ℝ, values_inner : list ℕ, values_outer : list ℕ }) : ℝ :=
  let prob_odd := prob_of_point_value dartboard.radius_inner dartboard.radius_outer dartboard.values_inner dartboard.values_outer 5 -- 5 is the only odd value in both inner and outer
  let prob_even := prob_of_point_value dartboard.radius_inner dartboard.radius_outer dartboard.values_inner dartboard.values_outer 3 -- 3 and 4 are the only even values in both inner and outer
  let prob_odd_then_even := prob_odd * prob_even
  2 * prob_odd_then_even

theorem dartboard_probability_odd_score : prob_odd_score dartboard = 4/9 :=
sorry

end dartboard_probability_odd_score_l163_163634


namespace inequality_proof_l163_163367

theorem inequality_proof
  (x y z : ℝ) (hxpos : 0 < x) (hypos : 0 < y) (hzpos : 0 < z)
  (hineq : x * y + y * z + z * x ≤ 1) :
  (x + 1 / x) * (y + 1 / y) * (z + 1 / z) ≥ 8 * (x + y) * (y + z) * (z + x) :=
sorry

end inequality_proof_l163_163367


namespace smallest_integer_sum_consecutive_l163_163218

theorem smallest_integer_sum_consecutive
  (l m n a : ℤ)
  (h1 : a = 9 * l + 36)
  (h2 : a = 10 * m + 45)
  (h3 : a = 11 * n + 55)
  : a = 495 :=
sorry

end smallest_integer_sum_consecutive_l163_163218


namespace arithmetic_progression_term_l163_163356

theorem arithmetic_progression_term (n r : ℕ) (S : ℕ → ℕ)
  (hS : ∀ n, S n = 3n + 5n^2) : S r - S (r - 1) = 10r - 2 :=
by
  intros n r S hS
  sorry

end arithmetic_progression_term_l163_163356


namespace shared_friends_count_l163_163128

theorem shared_friends_count (james_friends : ℕ) (total_combined : ℕ) (john_factor : ℕ) 
  (h1 : james_friends = 75) 
  (h2 : john_factor = 3) 
  (h3 : total_combined = 275) : 
  james_friends + (john_factor * james_friends) - total_combined = 25 := 
by
  sorry

end shared_friends_count_l163_163128


namespace calculate_value_l163_163984

theorem calculate_value : (Int.floor (| -5.7 |) + Int.abs (Int.floor (-5.7)) + Int.ceil (- | 5.7 |)) = 6 := by
  sorry

end calculate_value_l163_163984


namespace y_minus_x_l163_163324

def binary_representation (n : ℕ) : string := "10010011"

def count_bits (s : string) (bit : char) : ℕ :=
  s.to_list.count (λ c, c = bit)

noncomputable def x : ℕ := count_bits (binary_representation 147) '0'
noncomputable def y : ℕ := count_bits (binary_representation 147) '1'

theorem y_minus_x :
  y - x = 0 :=
by
  sorry

end y_minus_x_l163_163324


namespace number_of_arrangements_including_qu_l163_163175

theorem number_of_arrangements_including_qu : 
  ∃ n : ℕ, n = 480 ∧ n = 
  -- Combination: Choose 3 letters from the remaining 6 letters
  (nat.choose 6 3) * 
  -- Permutation: Arrange the chosen letters plus the "qu" block
  (nat.factorial 4) := 
begin
  use 480,
  split,
  { refl },
  { sorry }  -- proof details are omitted
end

end number_of_arrangements_including_qu_l163_163175


namespace intersection_of_A_and_B_l163_163828

def A := {(x, y) : ℝ × ℝ | y = x}
def B := {(x, y) : ℝ × ℝ | y = x^2}
def IntersectionSet := ({(0 : ℝ, 0 : ℝ)} : Set (ℝ × ℝ)) ∪ ({(1 : ℝ, 1 : ℝ)} : Set (ℝ × ℝ))

theorem intersection_of_A_and_B : A ∩ B = IntersectionSet := by
  sorry

end intersection_of_A_and_B_l163_163828


namespace trigonometric_identity_l163_163220

theorem trigonometric_identity :
  cos (43 * real.pi / 180) * cos (77 * real.pi / 180) - sin (43 * real.pi / 180) * sin (77 * real.pi / 180) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l163_163220


namespace ratio_of_perimeter_to_inradius_l163_163617

theorem ratio_of_perimeter_to_inradius (BC AD E C D : Point) (α : ℝ) (r P : ℝ)
  (h1 : inscribed_trapezoid ABCD BC AD)
  (h2 : on_arc E C D)
  (h3 : ∠ CED = 120)
  (h4 : ∠ ABE - ∠ BAE = α)
  : P / r = 2 * (Real.cot (30 - α / 4) + Real.cot (30 + α / 4) + Real.sqrt 3) :=
sorry

end ratio_of_perimeter_to_inradius_l163_163617


namespace point_P_in_Quadrant_II_l163_163384

noncomputable def α : ℝ := (5 * Real.pi) / 8

theorem point_P_in_Quadrant_II : (Real.sin α > 0) ∧ (Real.tan α < 0) := sorry

end point_P_in_Quadrant_II_l163_163384


namespace evaluate_g_l163_163139

theorem evaluate_g (x : ℝ) : (λ x : ℝ, 3 * x + 1) (x^2 + 2*x + 2) = 3*x^2 + 6*x + 7 := 
by
  -- Proof skipped
  sorry

end evaluate_g_l163_163139


namespace opposite_of_sqrt3_plus_a_l163_163080

theorem opposite_of_sqrt3_plus_a 
  (a b : ℝ) 
  (h : |a - 3b| + real.sqrt (b + 1) = 0) : 
  -(real.sqrt 3 + a) = 3 - real.sqrt 3 :=
sorry

end opposite_of_sqrt3_plus_a_l163_163080


namespace finiteness_of_algorithm_implies_option_C_l163_163557

-- Definitions for options A, B, C, and D
def option_A : Prop := ∀ (algorithm : Type), ∃ output : Type, True
def option_B : Prop := ∀ (algorithm : Type), ∀ step : algorithm, executable step
def option_C : Prop := ∀ (algorithm : Type), finite_steps algorithm
def option_D : Prop := ¬ (option_A ∨ option_B ∨ option_C)

-- The definition of finiteness of an algorithm
def finiteness (algorithm : Type) : Prop := finite_steps algorithm

-- The theorem we are proving
theorem finiteness_of_algorithm_implies_option_C :
  ∀ (algorithm : Type), finiteness algorithm → option_C :=
  by
    sorry -- Proof not needed as per instructions

end finiteness_of_algorithm_implies_option_C_l163_163557


namespace general_formulas_largest_n_l163_163043

namespace sequence_problem

def a_sequence (n : ℕ) : ℕ := 2^n

def b_sequence (n : ℕ) : ℕ := n

noncomputable def sum_first_n_terms {a_seq : ℕ → ℕ} : ℕ → ℕ
| 0       => 0
| (n + 1) => sum_first_n_terms n + a_seq (n + 1)

def S_n (n : ℕ) : ℕ := sum_first_n_terms a_sequence n

def condition1 (n : ℕ) : Prop := S_n n + 2 = 2 * a_sequence n

theorem general_formulas (n : ℕ) (h : n > 0) : 
  (∀ n > 0, condition1 n) →
  (∀ n, a_sequence n = 2^n) ∧ (∀ n, b_sequence n = n) := 
begin
  intros h_condition,
  split;
  sorry
end

noncomputable def c_sequence (n : ℕ) : ℕ := a_sequence n * b_sequence n

noncomputable def T_n (n : ℕ) : ℕ :=
  ∑ i in finset.range (n + 1), c_sequence i

theorem largest_n (h : ∀ n, condition1 n) : 
  (∀ n, a_sequence n = 2^n) → 
  (∀ n, b_sequence n = n) → 
  ∀ n, (T_n n < 2022) → n ≤ 7 := 
begin
  intros h_a_seq h_b_seq,
  intros n h_T_n,
  sorry
end

end sequence_problem

end general_formulas_largest_n_l163_163043


namespace angle_at_1_10_l163_163329

-- Define the angular displacement per minute for minute hand and hour hand
def minute_hand_angle_per_minute : ℝ := 6
def hour_hand_angle_per_minute : ℝ := 0.5

-- Define the initial position of the hour hand at 1:00
def hour_hand_initial_angle : ℝ := 30

-- Define the time in minutes passed
def time_passed : ℝ := 10

-- Define the angular positions of the hour and minute hands at 1:10
def minute_hand_angle : ℝ := minute_hand_angle_per_minute * time_passed
def hour_hand_angle : ℝ := hour_hand_initial_angle + hour_hand_angle_per_minute * time_passed

-- Define the angle between the hour and minute hands at 1:10
def angle_between_hands : ℝ := minute_hand_angle - hour_hand_angle

-- The theorem statement
theorem angle_at_1_10 : angle_between_hands = 25 :=
by
  -- Implementing the statement for the proof
  unfold angle_between_hands
  unfold minute_hand_angle
  unfold hour_hand_angle
  rw [minute_hand_angle_per_minute, hour_hand_angle_per_minute, hour_hand_initial_angle, time_passed]
  norm_num
  sorry

end angle_at_1_10_l163_163329


namespace geometric_sequence_l163_163893

theorem geometric_sequence (a : ℝ) (h1 : a > 0)
  (h2 : ∃ r : ℝ, 210 * r = a ∧ a * r = 63 / 40) :
  a = 18.1875 :=
by
  sorry

end geometric_sequence_l163_163893


namespace house_transaction_l163_163279

variable (initial_value : ℝ) (loss_rate : ℝ) (gain_rate : ℝ) (final_loss : ℝ)

theorem house_transaction
  (h_initial : initial_value = 12000)
  (h_loss : loss_rate = 0.15)
  (h_gain : gain_rate = 0.15)
  (h_final_loss : final_loss = 270) :
  let selling_price := initial_value * (1 - loss_rate)
  let buying_price := selling_price * (1 + gain_rate)
  (initial_value - buying_price) = final_loss :=
by
  simp only [h_initial, h_loss, h_gain, h_final_loss]
  sorry

end house_transaction_l163_163279


namespace final_solutions_l163_163339

noncomputable def functional_equation_solutions (f : ℝ → ℝ) :=
∀ x y : ℝ, f (f x^2 + f y) = x * f x + y

theorem final_solutions : 
  ∀ f : ℝ → ℝ, 
  (functional_equation_solutions f) → (∀ x : ℝ, f x = x ∨ f x = -x) :=
begin
  intros f h,
  sorry
end

end final_solutions_l163_163339


namespace grid1_not_transformable_to_all_zeros_grid2_not_transformable_to_all_zeros_grid3_transformable_to_all_zeros_l163_163320

-- Define the grids
def grid1 : matrix (fin 4) (fin 4) bool :=
  ![![false, true, true, false],
    ![true, true, false, true],
    ![false, false, true, true],
    ![false, false, true, true]]

def grid2 : matrix (fin 4) (fin 4) bool :=
  ![![false, true, false, false],
    ![true, true, false, true],
    ![false, false, false, true],
    ![true, false, true, true]]

def grid3 : matrix (fin 4) (fin 4) bool :=
  ![![false, false, false, false],
    ![true, true, false, false],
    ![false, true, false, true],
    ![true, false, false, true]]

-- Define the theorem statements
theorem grid1_not_transformable_to_all_zeros : ¬ (exists operations, apply_operations grid1 operations = (0 : matrix (fin 4) (fin 4) bool)) :=
sorry

theorem grid2_not_transformable_to_all_zeros : ¬ (exists operations, apply_operations grid2 operations = (0 : matrix (fin 4) (fin 4) bool)) :=
sorry

theorem grid3_transformable_to_all_zeros : (exists operations, apply_operations grid3 operations = (0 : matrix (fin 4) (fin 4) bool)) :=
sorry

end grid1_not_transformable_to_all_zeros_grid2_not_transformable_to_all_zeros_grid3_transformable_to_all_zeros_l163_163320


namespace perp_condition_norm_condition_l163_163752

open Real

variables (α : ℝ) (k : ℤ)
def m := (cos α, 1 - sin α) : ℝ × ℝ
def n := (-cos α, sin α) : ℝ × ℝ

-- Problem 1
theorem perp_condition (h : (m α) • (n α) = 0) : 
  ∃ k : ℤ, α = 2 * k * π + π / 2 :=
  sorry

-- Problem 2
theorem norm_condition (h : (∥(m α).fst - (n α).fst, (m α).snd - (n α).snd∥ = √3)) : 
  cos (2 * α) = 1 / 2 :=
  sorry

end perp_condition_norm_condition_l163_163752


namespace derivative_at_zero_l163_163762

def f (x : ℝ) : ℝ := (x + 1)^4

theorem derivative_at_zero : deriv f 0 = 4 :=
by
  sorry

end derivative_at_zero_l163_163762


namespace quadratic_always_real_roots_rhombus_area_when_m_minus_7_l163_163371

-- Define the quadratic equation
def quadratic_eq (m x : ℝ) : ℝ := 2 * x^2 + (m - 2) * x - m

-- Statement 1: For any real number m, the quadratic equation always has real roots.
theorem quadratic_always_real_roots (m : ℝ) : ∃ x1 x2 : ℝ, quadratic_eq m x1 = 0 ∧ quadratic_eq m x2 = 0 :=
by {
  -- Proof omitted
  sorry
}

-- Statement 2: When m = -7, the area of the rhombus whose diagonals are the roots of the quadratic equation is 7/4.
theorem rhombus_area_when_m_minus_7 : (∃ x1 x2 : ℝ, quadratic_eq (-7) x1 = 0 ∧ quadratic_eq (-7) x2 = 0 ∧ (1 / 2) * x1 * x2 = 7 / 4) :=
by {
  -- Proof omitted
  sorry
}

end quadratic_always_real_roots_rhombus_area_when_m_minus_7_l163_163371


namespace cylinder_area_ratio_l163_163086

noncomputable def ratio_lateral_to_total_area (r h : ℝ) : ℝ :=
  let Slateral := 2 * π * r * h
  let Stotal := Slateral + 2 * π * r^2
  Slateral / Stotal

theorem cylinder_area_ratio (r h : ℝ) (hratio : 2 * r = h / (2 * π * r)) :
  ratio_lateral_to_total_area r h = (2 * sqrt π) / (2 * sqrt π + 1) :=
by
  sorry

end cylinder_area_ratio_l163_163086


namespace max_value_diagonal_square_area_ratio_l163_163638

noncomputable def max_ratio_diagonal_area (T : Triangle) : ℝ := 
let d := shortest_diagonal_of_inscribed_rectangle T in
d^2 / (area T)

theorem max_value_diagonal_square_area_ratio :
  ∃ (T : Triangle), max_ratio_diagonal_area T = (4 * Real.sqrt 3 / 7) := 
sorry

end max_value_diagonal_square_area_ratio_l163_163638


namespace correct_eccentricity_l163_163718

noncomputable def eccentricity_of_ellipse (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) 
  (h₃ : ∃ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ (∃ c : ℝ, x^2 + y^2 = c^2 ∧ 2c = a))
  (h₄ : ∃ F1 F2 P O : point3dℝ, OP.norm = (F1F2.norm / 2) ∧ PF1.norm * PF2.norm = a^2) : ℝ :=
  let e := (b / a) in
  e

theorem correct_eccentricity (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) 
  (h₃ : ∃ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ (∃ c : ℝ, x^2 + y^2 = c^2 ∧ 2c = a))
  (h₄ : ∃ F1 F2 P O : point3dℝ, OP.norm = (F1F2.norm / 2) ∧ PF1.norm * PF2.norm = a^2) :
  eccentricity_of_ellipse a b h₁ h₂ h₃ h₄ = (Real.sqrt 2 / 2) :=
sorry

end correct_eccentricity_l163_163718


namespace correct_derivatives_l163_163914

def derivative_log2x_correct (x : ℝ) (hx : x > 0) : Prop :=
  deriv (λ x, log 2 x) x = 1 / (x * (log 2).logbase 10)

def derivative_x2_cosx_correct (x : ℝ) : Prop :=
  deriv (λ x, x^2 * cos x) x = 2 * x * cos x - x^2 * sin x

theorem correct_derivatives :
  ∀ (x : ℝ), (x > 0 → derivative_log2x_correct x x) ∧ derivative_x2_cosx_correct x :=
by
  intros x
  constructor
  intro hx
  unfold derivative_log2x_correct
  sorry
  unfold derivative_x2_cosx_correct
  sorry


end correct_derivatives_l163_163914


namespace largest_angle_of_line_with_plane_l163_163428

theorem largest_angle_of_line_with_plane (θ : ℝ) : θ = 72 → ∃ φ : ℝ, φ = 90 :=
by
  assume h : θ = 72,
  use 90,
  sorry

end largest_angle_of_line_with_plane_l163_163428


namespace divide_quadrilateral_area_l163_163589

-- Definitions of vertices and quadrilateral
structure Point :=
(x : ℝ)
(y : ℝ)

structure Quadrilateral :=
(A B C D : Point)
(convex : ∀ P : Point, is_convex {A, B, C, D})

-- Areas of triangular regions
noncomputable def triangle_area (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y))

-- Main theorem
theorem divide_quadrilateral_area (A B C D : Point) (h_convex : ∀ P : Point, is_convex {A, B, C, D}) :
  ∃ l : line, passes_through l A ∧ divides_area_equally l (Quadrilateral.mk A B C D h_convex) :=
sorry

end divide_quadrilateral_area_l163_163589


namespace solution_set_inequality_l163_163021

noncomputable def f : ℝ → ℝ := sorry

def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 1 = 1) ∧ ∀ x, deriv f x < 1 

theorem solution_set_inequality : satisfies_conditions f →  
  { x : ℝ | f((log 10) ^2 x) < (log 10) ^2 x } = { x : ℝ | x > 10 } ∪ { x : ℝ | 0 < x ∧ x < 1 / 10 } :=
by
  sorry

end solution_set_inequality_l163_163021


namespace find_x_l163_163085

-- Definitions for LCM in Lean
noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem find_x :
  let x := 9 in
  (lcm (lcm 12 16) (lcm x 24) = 144) :=
by
  let x := 9
  show lcm (lcm 12 16) (lcm x 24) = 144 from sorry

end find_x_l163_163085


namespace count_integers_with_at_most_three_digits_l163_163069

/-- The number of positive integers less than 50,000 with at most three different digits is 7503. -/
theorem count_integers_with_at_most_three_digits : 
  (finset.filter (λ n : ℕ, n < 50000 ∧ (finset.card (finset.image (λ d : ℕ, (n / 10^d) % 10) (finset.range 5)) ≤ 3)) (finset.range 50000)).card = 7503 := 
sorry

end count_integers_with_at_most_three_digits_l163_163069


namespace digits_sum_divisible_l163_163171

theorem digits_sum_divisible :
  let S := ∑ p in (Nat.digits 9).permutations, Nat.of_digits 10 p
  in S % 999999999 = 0 :=
by
  sorry

end digits_sum_divisible_l163_163171


namespace remainder_76_pow_77_mod_7_l163_163626

theorem remainder_76_pow_77_mod_7 : (76 ^ 77) % 7 = 6 := 
by 
  sorry 

end remainder_76_pow_77_mod_7_l163_163626


namespace number_of_handshakes_l163_163849

theorem number_of_handshakes (n : ℕ) (h : n = 12) (m : ℕ) (hm1 : m = 6) 
(persons : Fin n → Fin n → Prop)
(spouses : Fin m → Fin 2 → Fin n)
(exclusions : Fin n) :
  (∑ i, if exclusions = i then 7 else 10) / 2 = 57 := 
sorry

end number_of_handshakes_l163_163849


namespace probability_x_gt_9y_in_rectangle_l163_163520

theorem probability_x_gt_9y_in_rectangle :
  let a := 1007
  let b := 1008
  let area_triangle := (a * a / 18 : ℚ)
  let area_rectangle := (a * b : ℚ)
  area_triangle / area_rectangle = (1 : ℚ) / 18 :=
by
  sorry

end probability_x_gt_9y_in_rectangle_l163_163520


namespace Jacob_age_is_3_l163_163443

def Phoebe_age : ℕ := sorry
def Rehana_age : ℕ := 25
def Jacob_age (P : ℕ) : ℕ := 3 * P / 5

theorem Jacob_age_is_3 (P : ℕ) (h1 : Rehana_age + 5 = 3 * (P + 5)) (h2 : Rehana_age = 25) (h3 : Jacob_age P = 3) : Jacob_age P = 3 := by {
  sorry
}

end Jacob_age_is_3_l163_163443


namespace numberOfInnerSquares_l163_163074

-- Define the region bounded by the given lines
def inRegion (x y : ℤ) : Prop :=
  y ≤ 2 * x ∧ y ≥ -1 ∧ x ≤ 5

-- Define the property of being an inner square in the region
def isSquare (x y size : ℤ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i ≤ size ∧ 0 ≤ j ∧ j ≤ size → inRegion (x + i) (y + j)

-- Count the number of inner squares of a given size that exist in the region
def countSquaresOfSize (size : ℤ) : ℤ :=
  ∑ x in Finset.range (5 - size + 1), ∑ y in Finset.range (11 - size),
    if isSquare x y size then 1 else 0

-- The total number of inner squares
def totalInnerSquares : ℤ :=
  ∑ size in Finset.range 5, countSquaresOfSize size

-- The theorem statement asserting the total number of inner squares
theorem numberOfInnerSquares : totalInnerSquares = 43 := by
  sorry

end numberOfInnerSquares_l163_163074


namespace number_of_diagonals_of_regular_polygon_l163_163387

theorem number_of_diagonals_of_regular_polygon (exterior_angle : ℕ → ℕ) (n : ℕ) :
  (∀ p, exterior_angle p = 60 → (360 / 60 = n) → (number_of_diagonals n = 9)) :=
by
  sorry

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

end number_of_diagonals_of_regular_polygon_l163_163387


namespace problem1_solution_problem2_solution_l163_163857

theorem problem1_solution (x : ℝ) : x^2 - x - 6 > 0 ↔ x < -2 ∨ x > 3 := sorry

theorem problem2_solution (x : ℝ) : -2*x^2 + x + 1 < 0 ↔ x < -1/2 ∨ x > 1 := sorry

end problem1_solution_problem2_solution_l163_163857


namespace paper_flowers_per_hour_l163_163335

theorem paper_flowers_per_hour
  (time_eq : ∀ (x : ℤ), (120 : ℤ) / (x - 20) = (160 : ℤ) / x) :
  ∃ x : ℤ, x = 80 := by
  use 80
  sorry

end paper_flowers_per_hour_l163_163335


namespace congruent_triangle_colored_sides_l163_163024

theorem congruent_triangle_colored_sides (plane : ℝ × ℝ → ℕ)
  (colors : fin 1992) (H1 : ∀ c : fin 1992, ∃ x y : ℝ × ℝ, plane x = c ∧ plane y = c):
  ∀ (T : triangle ℝ), ∃ (T' : triangle ℝ), T ≅ T' ∧ ∀ (side : segment ℝ), side ∈ T'.sides →
  ∃ p1 p2 : ℝ × ℝ, p1 ≠ p2 ∧ p1 ∈ side ∧ p2 ∈ side ∧ plane p1 = plane p2 :=
begin
  sorry
end

end congruent_triangle_colored_sides_l163_163024


namespace shorter_piece_length_l163_163635

theorem shorter_piece_length (L : ℝ) (hL : L = 60) (x y : ℝ) (hy : y = 2 * x) (hxy : x + y = L) : x = 20 :=
by
  have h1 : x + 2 * x = 60 := by
    rw [hy, hxy, hL]
  have h2 : 3 * x = 60 := by
    linarith
  have h3 : x = 20 := by
    linarith
  exact h3

end shorter_piece_length_l163_163635


namespace math_problem_l163_163631

-- Define the lines
def line1 (x y : ℝ) := 3 * x + y - 1 = 0
def line2 (x y : ℝ) := x - 5 * y - 11 = 0
def perp_line (x y : ℝ) := x + 4 * y = 0

-- Define the intersection point of line1 and line2
def intersection_point := (1 : ℝ, -2 : ℝ)

-- Define the equation of line l
def line_l (x y : ℝ) := 4 * x - y - 6 = 0

-- Define the circle and its properties
def circle_center : (ℝ × ℝ) := (0, 11)
def circle_radius : ℝ := 5

-- Define the distance from the center of the circle to line l
def distance_to_line : ℝ := Real.abs (-11 - 6) / Real.sqrt (4^2 + 1^2)

-- Define the length of the chord
def chord_length : ℝ := 2 * Real.sqrt (circle_radius^2 - distance_to_line^2)

-- The theorem stating the required proof
theorem math_problem :
  line_l intersection_point.1 intersection_point.2 ∧
  chord_length = 4 * Real.sqrt 2 := by
  sorry

end math_problem_l163_163631


namespace squirrels_caught_in_one_hour_l163_163983

theorem squirrels_caught_in_one_hour :
  (S : ℕ) (R : ℕ) (cals_squirrel : ℕ) (cals_rabbit : ℕ) (extra_cals_per_hour : ℕ) (catch_time : ℕ) :
  cals_squirrel = 300 → cals_rabbit = 800 → extra_cals_per_hour = 200 → R = 2 →
  300 * S = 800 * R + 200 → S = 6 :=
by
  intros S R cals_squirrel cals_rabbit extra_cals_per_hour catch_time hcs hcr hcp hR hCal
  sorry

end squirrels_caught_in_one_hour_l163_163983


namespace number_placing_possible_iff_even_l163_163112

theorem number_placing_possible_iff_even (n : ℕ) (h : n > 1) :
  (∃ f : Fin (n^2) → Fin n × Fin n,
  (∀ i j : Fin (n^2), (f i).fst = (f j).fst → (f i).snd ≠ (f j).snd) ∧
  (∀ i j : Fin (n^2), (f i).snd = (f j).snd → (f i).fst ≠ (f j).fst) ∧
  (∀ i, ∃ j : Fin (n^2), abs ((f i).fst - (f j).fst) + abs ((f i).snd - (f j).snd) = 1)) ↔
  Even n :=
sorry

end number_placing_possible_iff_even_l163_163112


namespace fleas_can_meet_in_triangle_l163_163897

noncomputable def flea_meet (n : ℕ) : Prop :=
∃ (initial_config : Fin n → Fin n.succ → Fin n.succ → Bool),
  ∀ (k : ℕ), ∃ (m : ℕ) (config : Fin n → Fin n.succ → Fin n.succ → Bool), 
    k ≥ m ∧ 
    (∀ (i j : Fin n.succ), 
      (initial_config i j True → config i j False) ∧ 
      (initial_config i j False → config i j True)) ∧ 
    ∃ (p x y : Fin n.succ), config p x y True

theorem fleas_can_meet_in_triangle (n : ℕ) : flea_meet n :=
sorry

end fleas_can_meet_in_triangle_l163_163897


namespace minimum_value_of_expression_l163_163483

theorem minimum_value_of_expression 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : 3 * a + 4 * b + 2 * c = 3) : 
  (1 / (2 * a + b) + 1 / (a + 3 * c) + 1 / (4 * b + c)) = 1.5 :=
sorry

end minimum_value_of_expression_l163_163483


namespace final_shirt_price_l163_163997

theorem final_shirt_price :
  let cost_price := 20
  let profit_rate := 0.30
  let discount_rate := 0.50
  let profit := cost_price * profit_rate
  let regular_selling_price := cost_price + profit
  let final_price := regular_selling_price * discount_rate
  in final_price = 13 :=
by
  sorry

end final_shirt_price_l163_163997


namespace perimeter_equality_l163_163253

noncomputable def prove_perimeter_sum (BE EC AE : ℝ) (h_perpendicular : true)
  (m n p q : ℝ) (h_m : true) (h_n : true) (h_p : true) (h_q : true) : ℝ :=
  sqrt (m * n) + sqrt (p + q)

theorem perimeter_equality : prove_perimeter_sum 3 2 6
  true -- Placeholder for perpendicularity condition
  5 5 3 13
  true true true true -- Placeholders for derived values of m, n, p, q
  = 9 :=
by
  sorry

end perimeter_equality_l163_163253


namespace f_seq_2018_l163_163358

noncomputable def f (x : ℝ) : ℝ := x / (1 + x)

def f_seq : ℕ → (ℝ → ℝ)
| 0     := f
| (n+1) := f ∘ f_seq n

theorem f_seq_2018 (x : ℝ) (hx : x ≥ 0) : f_seq 2017 x = x / (1 + 2018 * x) :=
by
  sorry

-- f_seq 2017 instead of f_seq 2018 because Lean indexes from 0, 
-- thus f_seq 2017 represents the 2018th function in the sequence.

end f_seq_2018_l163_163358


namespace find_t_l163_163788

theorem find_t (t : ℝ) : 
  let OA := (-1, t)
  let OB := (2, 2)
  let AB := ((2 - (-1)), (2 - t))
  angle OB AB O = 90 →
  t = 5 :=
by
  intros
  sorry

end find_t_l163_163788


namespace problem_solution_l163_163602

-- Define the possible options
def option_A := ∃ (seed : bool), true -- Simplification for the sake of illustration
def option_B := ∃ (die : ℕ), die = 1 -- Simplification assuming biased die
def option_C := ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 4
def option_D := ∃ (die1 die2 : ℕ), 1 ≤ die1 ∧ die1 ≤ 6 ∧ 1 ≤ die2 ∧ die2 ≤ 6 ∧ die1 + die2 = 5

-- Classical probability model condition
def classical_probability_model (event : Prop) :=
  ∃ (outcomes : ℕ → Prop), finite (set_of outcomes) ∧ (∀ (o : ℕ), (outcomes o) → (1 / (nat.card { x | outcomes x})) = 1) ∧ event = ∃ (o : ℕ), outcomes o

-- The theorem that we need to prove
theorem problem_solution :
  classical_probability_model option_D :=
sorry

end problem_solution_l163_163602


namespace calculate_roots_l163_163312

noncomputable def cube_root (x : ℝ) := x^(1/3 : ℝ)
noncomputable def square_root (x : ℝ) := x^(1/2 : ℝ)

theorem calculate_roots : cube_root (-8) + square_root 9 = 1 :=
by
  sorry

end calculate_roots_l163_163312


namespace shorter_piece_length_l163_163606

-- Definitions corresponding to the conditions:
def total_length : ℝ := 70
def ratio : ℝ := 2 / 3

-- Lean 4 statement that the question equals the answer, given the conditions:
theorem shorter_piece_length : ∃ x : ℝ, x + ratio * x = total_length ∧ x = 42 :=
by
  sorry

end shorter_piece_length_l163_163606


namespace number_of_babies_in_quintuplets_is_500_l163_163978

theorem number_of_babies_in_quintuplets_is_500 :
  ∀ (t q p : ℕ), t = 2 * q ∧ p = q / 2 ∧ 3 * t + 4 * q + 5 * p = 2500 →  5 * p = 500 :=
by {
  intros t q p h,
  sorry
}

end number_of_babies_in_quintuplets_is_500_l163_163978


namespace probability_of_negative_card_l163_163696

theorem probability_of_negative_card :
  let cards := [-2, 1, 0, -1, real.sqrt 2] in
  let total_cards := list.length cards in
  let negative_cards := list.countp (λ x : ℝ, x < 0) cards in
  (negative_cards : ℚ) / (total_cards : ℚ) = 2 / 5 :=
by 
  sorry

end probability_of_negative_card_l163_163696


namespace intersection_A_complement_B_l163_163381

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x - 3 > 0}
def comR (S : Set ℝ) : Set ℝ := {x | ¬ (x ∈ S)}

theorem intersection_A_complement_B : A ∩ (comR B) = {x | -1 < x ∧ x ≤ 3} := 
by
  sorry

end intersection_A_complement_B_l163_163381


namespace symmetric_point_m_eq_one_l163_163576

theorem symmetric_point_m_eq_one (m : ℝ) (A B : ℝ × ℝ) 
  (hA : A = (-3, 2 * m - 1))
  (hB : B = (-3, -1))
  (symmetric : A.1 = B.1 ∧ A.2 = -B.2) : 
  m = 1 :=
by
  sorry

end symmetric_point_m_eq_one_l163_163576


namespace train_speed_in_km_per_hr_l163_163952

-- Definitions from the problem conditions
def length_of_train : ℝ := 50
def time_to_cross_pole : ℝ := 3

-- Conversion factor from the problem 
def meter_per_sec_to_km_per_hr : ℝ := 3.6

-- Lean theorem statement based on problem conditions and solution
theorem train_speed_in_km_per_hr : 
  (length_of_train / time_to_cross_pole) * meter_per_sec_to_km_per_hr = 60 := by
  sorry

end train_speed_in_km_per_hr_l163_163952


namespace proof_partA_proof_partB_l163_163161

noncomputable def partA (A B C M L P : Point) (BC_Length BM MC Area_ABP Area_LPMC : ℝ) 
(BM_MC_ratio : BM / MC = 2 / 7) (angle_BPA : ∠ BPA = 90): Prop :=
  Area_ABP / Area_LPMC = 11 / 70

noncomputable def partB (A B C M L P T : Point) (MC_Length MT TC cos_gamma : ℝ)
(MT_TC_ratio : MT / TC = 1 / 6) (LT_perp_BC : ∠ LTB = 90) : Prop :=
  cos_gamma = arccos (sqrt 11 / (2 * sqrt 3))

-- Theorem statements
theorem proof_partA (A B C M L P : Point) (BC_Length BM MC Area_ABP Area_LPMC : ℝ) 
(BM_MC_ratio : BM / MC = 2 / 7) (angle_BPA : ∠ BPA = 90) : partA A B C M L P BC_Length BM MC Area_ABP Area_LPMC BM_MC_ratio angle_BPA :=
  sorry

theorem proof_partB (A B C M L P T : Point) (MC_Length MT TC cos_gamma : ℝ) 
(MT_TC_ratio : MT / TC = 1 / 6) (LT_perp_BC : ∠ LTB = 90) : partB A B C M L P T MC_Length MT TC cos_gamma MT_TC_ratio LT_perp_BC :=
  sorry

end proof_partA_proof_partB_l163_163161


namespace evaluate_f_at_points_l163_163050

noncomputable def f : ℝ → ℝ := λ x, if x < 1 then 1 + Math.logBase 2 (2 - x) else 2^(x - 1)

theorem evaluate_f_at_points :
  f (-2) + f (Math.logBase 2 12) = 9 := by
  sorry

end evaluate_f_at_points_l163_163050


namespace log_difference_condition_l163_163422

theorem log_difference_condition (x y a : ℝ) (h : log 10 x - log 10 y = a) : 
  log 10 ((x / 2)^3) - log 10 ((y / 2)^3) = 3 * a :=
by
  sorry

end log_difference_condition_l163_163422


namespace angle_between_vectors_is_60_degrees_l163_163751

noncomputable def find_angle_between_vectors
  (a b : EuclideanSpace ℝ n)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 2)
  (hab : inner a b = 1) : Real.Angle :=
let θ := Real.arccos (1 / ((∥a∥) * (∥b∥))) in
if θ = 60 then θ else sorry

-- Proof outline in Lean: 
-- Prove that the calculated angle from the given conditions is 60 degrees.
theorem angle_between_vectors_is_60_degrees
  (a b : EuclideanSpace ℝ n)
  (ha : ∥a∥ = 1)
  (hb : ∥b∥ = 2)
  (hab : inner a b = 1) :
  find_angle_between_vectors a b ha hb hab = 60 :=
  by sorry

end angle_between_vectors_is_60_degrees_l163_163751


namespace new_complex_number_l163_163350

open Complex

theorem new_complex_number :
  (Complex.mk 2 (-2) : ℂ) = 
    (Complex.mk 
      (Im (2 * I - Real.sqrt 5)) 
      (Re (Real.sqrt 5 * I + 2 * I ^ 2))) := 
by
  sorry

end new_complex_number_l163_163350


namespace square_field_side_length_l163_163269

variable (t : ℝ) (s : ℝ) (L : ℝ)

-- Conditions given in the problem
axiom (H1 : t = 72) -- time in seconds
axiom (H2 : s = 12 * 1000 / 3600) -- speed in m/s

-- The question is to evaluate the length of each side of the square field
theorem square_field_side_length : (4 * L = s * t) → L = 60 :=
by
  intros H
  sorry

end square_field_side_length_l163_163269


namespace scarlet_initial_savings_l163_163174

variable (saved earrings necklace left : ℕ)

def scarlet_saving_conditions : Prop :=
  earrings = 23 ∧ necklace = 48 ∧ left = 9

theorem scarlet_initial_savings (h : scarlet_saving_conditions saved earrings necklace left) :
  saved = earrings + necklace + left := by
  sorry

end scarlet_initial_savings_l163_163174


namespace correct_param_eq_l163_163730

-- Given line equation definition
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y + 1 = 0

-- Option B parametric equations definition
def param_eq (t : ℝ) : ℝ × ℝ := (1 - 4 * t, -1 + 3 * t)

theorem correct_param_eq : ∀ t : ℝ, line_eq (1 - 4 * t) (-1 + 3 * t) :=
by
  intro t
  dsimp [line_eq, param_eq]
  rw [mul_sub, mul_add, mul_one, mul_one, add_sub_assoc, sub_add_eq_sub_sub, sub_sub_eq_add_add]
  ring
  sorry

end correct_param_eq_l163_163730


namespace find_b_l163_163143

theorem find_b (b : ℝ) (p q : ℝ → ℝ)
  (hp : ∀ x, p(x) = 2 * x - 7)
  (hq : ∀ x, q(x) = 3 * x - b)
  (h : p(q(4)) = 7) :
  b = 5 :=
by
  sorry

end find_b_l163_163143


namespace probability_of_drawing_three_white_marbles_l163_163281

noncomputable def probability_of_three_white_marbles : ℚ :=
  let total_marbles := 5 + 7 + 15
  let prob_first_white := 15 / total_marbles
  let prob_second_white := 14 / (total_marbles - 1)
  let prob_third_white := 13 / (total_marbles - 2)
  prob_first_white * prob_second_white * prob_third_white

theorem probability_of_drawing_three_white_marbles :
  probability_of_three_white_marbles = 2 / 13 := 
by 
  sorry

end probability_of_drawing_three_white_marbles_l163_163281


namespace points_between_A_and_B_are_8_l163_163943

/-- Given points A (2, 3) and B (50, 203), prove that the number of points with integer coordinates on the line
    passing through A and B, and strictly between A and B, is 8. -/
theorem points_between_A_and_B_are_8 (A B : (ℝ × ℝ)) (hA : A = (2, 3)) (hB : B = (50, 203)) :
    ∃ (n : ℕ), n = 8 ∧ ∀ (P : ℤ × ℤ), P.1 > 2 ∧ P.1 < 50 ∧ (∃ m : ℚ, m * P.1 + b = P.2) → P = (m * x + b) → n = 8 := sorry

end points_between_A_and_B_are_8_l163_163943


namespace train_speed_l163_163956

-- Define the conditions
def train_length : ℝ := 50 -- Length of the train in meters
def crossing_time : ℝ := 3 -- Time to cross the pole in seconds

-- Define the speed in meters per second and convert it to km/hr
noncomputable def speed_mps : ℝ := train_length / crossing_time
noncomputable def speed_kmph : ℝ := speed_mps * 3.6 -- Conversion factor

-- Theorem statement: Prove that the calculated speed in km/hr is 60 km/hr
theorem train_speed : speed_kmph = 60 := by
  sorry

end train_speed_l163_163956


namespace profit_percentage_is_correct_l163_163296

noncomputable def CP : ℝ := 47.50
noncomputable def SP : ℝ := 74.21875
noncomputable def MP : ℝ := SP / 0.8
noncomputable def Profit : ℝ := SP - CP
noncomputable def ProfitPercentage : ℝ := (Profit / CP) * 100

theorem profit_percentage_is_correct : ProfitPercentage = 56.25 := by
  -- Proof steps to be filled in
  sorry

end profit_percentage_is_correct_l163_163296


namespace value_of_m_l163_163363

theorem value_of_m 
  (m : ℤ) 
  (h : ∀ x : ℤ, x^2 - 2 * (m + 1) * x + 16 = (x - 4)^2) : 
  m = 3 := 
sorry

end value_of_m_l163_163363


namespace simplify_trig_identity_l163_163181

theorem simplify_trig_identity (A : ℝ) (h1 : cot A = cos A / sin A)  
  (h2 : sec A = 1 / cos A) (h3 : tan A = sin A / cos A) (h4 : csc A = 1 / sin A) :
  (1 - cot A + sec A) * (1 + tan A - csc A) = 2 :=
  by
  sorry

end simplify_trig_identity_l163_163181


namespace line_slope_intercept_product_l163_163321

theorem line_slope_intercept_product :
  ∃ m b : ℝ, (∀ x y : ℝ, y = m * x + b ∧ ((0, -4) ∧ (4, 4) : set ℝ × set ℝ)) → m * b = -8 :=
begin
  sorry
end

end line_slope_intercept_product_l163_163321


namespace geometric_seq_product_l163_163022

theorem geometric_seq_product (a : ℕ → ℝ) (h : a 1005 * a 1007 = 4) :
    ∏ i in finset.range 2011.succ, a i = 2 ^ 2011 ∨ ∏ i in finset.range 2011.succ, a i = - (2 ^ 2011) :=
sorry

end geometric_seq_product_l163_163022


namespace derivative_at_0_l163_163304

def f (x : ℝ) : ℝ :=
if x ≠ 0 then 1 - Real.cos (x * Real.sin (1 / x)) else 0

theorem derivative_at_0 : (Real.deriv f 0) = 0 :=
by {
  -- Proof steps go here
  sorry
}

end derivative_at_0_l163_163304


namespace percentage_of_failed_candidates_l163_163919

noncomputable def total_candidates : ℕ := 2000
noncomputable def number_of_girls : ℕ := 900
noncomputable def number_of_boys : ℕ := total_candidates - number_of_girls
noncomputable def boys_passed_percentage : ℝ := 0.30
noncomputable def girls_passed_percentage : ℝ := 0.32

theorem percentage_of_failed_candidates :
  let boys_passed := boys_passed_percentage * number_of_boys
      girls_passed := girls_passed_percentage * number_of_girls
      total_passed := boys_passed + girls_passed
      total_failed := total_candidates - total_passed
  in (total_failed / total_candidates) * 100 = 69.1 :=
by
  sorry

end percentage_of_failed_candidates_l163_163919


namespace thickness_in_scientific_notation_l163_163834

-- Define the thickness of the bubble in millimeters
def bubble_thickness_mm : ℝ := 0.000309

-- Define the scientific notation expression
def scientific_notation_val : ℝ := 3.09 * 10^(-4)

-- Prove that the thickness of the bubble is equal to the scientific notation value
theorem thickness_in_scientific_notation :
  bubble_thickness_mm = scientific_notation_val :=
by sorry

end thickness_in_scientific_notation_l163_163834


namespace polynomial_integer_root_l163_163340

theorem polynomial_integer_root (b : ℤ) :
  (∃ x : ℤ, x^3 + 5 * x^2 + b * x + 9 = 0) ↔ b = -127 ∨ b = -74 ∨ b = -27 ∨ b = -24 ∨ b = -15 ∨ b = -13 :=
by
  sorry

end polynomial_integer_root_l163_163340


namespace find_ab_l163_163407

theorem find_ab (a b : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 2 →
  (3 * x - 2 < a + 1 ∧ 6 - 2 * x < b + 2)) →
  a = 3 ∧ b = 6 :=
by
  sorry

end find_ab_l163_163407


namespace inverse_trig_identity_l163_163926

theorem inverse_trig_identity :
    arctan (Real.sqrt 3 / 3) + arcsin (- (1 / 2)) + arccos 1 = 0 :=
by
    sorry

end inverse_trig_identity_l163_163926


namespace infinitely_many_sets_of_positive_integers_l163_163532

theorem infinitely_many_sets_of_positive_integers (k : ℕ) (h : k > 2015) : 
  let a := k
  let b := k + 1 
  let c := k^2 + k + 1
  in a > 2015 ∧ b > 2015 ∧ c > 2015 ∧
     a ∣ (b * c - 1) ∧
     b ∣ (a * c + 1) ∧
     c ∣ (a * b + 1) :=
sorry

end infinitely_many_sets_of_positive_integers_l163_163532


namespace n_cubed_minus_9n_plus_27_not_div_by_81_l163_163846

theorem n_cubed_minus_9n_plus_27_not_div_by_81 (n : ℤ) : ¬ 81 ∣ (n^3 - 9 * n + 27) :=
sorry

end n_cubed_minus_9n_plus_27_not_div_by_81_l163_163846


namespace cartesian_line_eq_range_of_m_l163_163451

noncomputable def polar_to_cartesian (rho theta m : ℝ) : Prop :=
  rho * sin theta + rho * cos theta - sqrt 2 * m = 0

def circle_eq (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem cartesian_line_eq (m : ℝ) : ∀ (rho theta : ℝ), polar_to_cartesian rho theta m → ∃ x y, (x + y - sqrt 2 * m = 0) :=
sorry

theorem range_of_m : ∀ m x y, circle_eq x y → x + y - sqrt 2 * m = 0 →
  (abs (2 - sqrt 2 * m) / sqrt 2 <= 3 * sqrt 2 / 2) → - sqrt 2 / 2 ≤ m ∧ m ≤ 5 * sqrt 2 / 2 :=
sorry

end cartesian_line_eq_range_of_m_l163_163451


namespace angle_KLD_is_right_angle_l163_163522

theorem angle_KLD_is_right_angle
  (A B C D K L : Point)
  (h_square : is_square A B C D)
  (h_midpoint : midpoint K A B)
  (h_ratio : divides_diagonal L A C (3/4)) :
  ∠ K L D = 90 := 
sorry

end angle_KLD_is_right_angle_l163_163522


namespace total_birds_in_tree_l163_163227

def kg_to_g (kg : ℕ) : ℕ := kg * 1000
def g_to_hummingbirds (grams : ℕ) : ℕ := grams / 60
def oz_to_lb (oz : ℕ) : ℕ := oz / 16

def initial_sparrows : ℕ := 7
def initial_robins : ℕ := 5
def initial_blue_jays : ℕ := 2
def additional_sparrows : ℕ := 12
def additional_robins : ℕ := 4
def additional_blue_jays : ℕ := 5
def hummingbirds_kg : ℕ := 0.3 * 1000 -- Converting to grams directly
def hummingbird_weight_g : ℕ := 60
def cardinal_oz : ℕ := 80
def cardinal_pound : ℕ := 1

def total_sparrows : ℕ := initial_sparrows + additional_sparrows
def total_robins : ℕ := initial_robins + additional_robins
def total_blue_jays : ℕ := initial_blue_jays + additional_blue_jays
def total_hummingbirds : ℕ := g_to_hummingbirds hummingbirds_kg
def total_cardinals : ℕ := oz_to_lb cardinal_oz

def total_birds : ℕ :=
  total_sparrows + total_robins + total_blue_jays + total_hummingbirds + total_cardinals

theorem total_birds_in_tree : total_birds = 45 :=
by
  sorry

end total_birds_in_tree_l163_163227


namespace correct_number_of_propositions_l163_163042

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, a (n + 1) = r * a n

noncomputable def proposition_p (a : ℕ → ℝ) : Prop :=
(a 1 < a 2 ∧ a 2 < a 3) → ∀ n m : ℕ, n < m → a n < a m

noncomputable def converse_of_p (a : ℕ → ℝ) : Prop :=
(∀ n m : ℕ, n < m → a n < a m) → (a 1 < a 2 ∧ a 2 < a 3)

noncomputable def negation_of_p (a : ℕ → ℝ) : Prop :=
¬(a 1 < a 2 ∧ a 2 < a 3) → ¬∀ n m : ℕ, n < m → a n < a m

noncomputable def contrapositive_of_p (a : ℕ → ℝ) : Prop :=
¬(∀ n m : ℕ, n < m → a n < a m) → ¬(a 1 < a 2 ∧ a 2 < a 3)

noncomputable def number_of_correct_propositions (a : ℕ → ℝ) : ℕ :=
if (geometric_sequence a ∧ proposition_p a ∧ converse_of_p a ∧ negation_of_p a ∧ contrapositive_of_p a)
then 4 else 0

theorem correct_number_of_propositions (a : ℕ → ℝ) :
  geometric_sequence a → number_of_correct_propositions a = 4 :=
by
  intros h
  sorry

end correct_number_of_propositions_l163_163042


namespace part1_part2_part3_l163_163374

-- Define the sequence a_n
def a : ℕ → ℝ
| 0 := 1
| (n + 1) := a n / (1 + a n)

-- The first sequence which is {1 / a n} is an arithmetic sequence with difference 1
theorem part1 : ∀ n : ℕ, (∀ k m : ℕ,  a (n + (m + 1)) ≠ 0 → 1 / a (n + 1) - 1 / a n = 1) :=
by sorry

-- The general term for a_n is 1/n
theorem part2 : ∀ n : ℕ, a n = 1 / (n + 1) :=
by sorry

-- Given the inequality, prove n = 4 or n = 5
theorem part3 (n : ℕ) (h1 : 2 / 3 < ∑ i in Finset.range (n - 1), a i * a (i + 1)) (h2 : ∑ i in Finset.range (n - 1), a i * a (i + 1) < 5 / 6) : n = 4 ∨ n = 5 :=
by sorry

end part1_part2_part3_l163_163374


namespace doritos_piles_proof_l163_163835

open Real

def total_bags_of_chips : ℕ := 1200
def doritos_fraction : ℝ := 3/7
def number_of_piles : ℕ := 9

def doritos_bags_per_pile (total_bags: ℕ) (fraction: ℝ) (piles: ℕ) : ℕ :=
  let total_doritos := (fraction * (total_bags : ℝ)).floor
  in (total_doritos / (piles : ℝ)).floor

theorem doritos_piles_proof : doritos_bags_per_pile total_bags_of_chips doritos_fraction number_of_piles = 57 :=
by
  sorry

end doritos_piles_proof_l163_163835


namespace root_of_function_l163_163774

-- Define the conditions from the problem
variable (f : ℝ → ℝ)
variable (y : ℝ)
variable (P : ℝ × ℝ := (0, 2))

-- The main statement of the problem as a Lean theorem
theorem root_of_function 
  (hf_inv : ∀ x : ℝ, f (f⁻¹ x) = x)
  (hP : P = (0, 2))
  (hy : y = -f⁻¹(0)) :
  f 2 = 0 :=
sorry

end root_of_function_l163_163774


namespace hilton_final_marbles_l163_163754

def initial_marbles : ℕ := 26
def found_marbles : ℕ := 6
def lost_marbles : ℕ := 10
def given_marbles := 2 * lost_marbles

theorem hilton_final_marbles (initial_marbles : ℕ) (found_marbles : ℕ) (lost_marbles : ℕ)
  (given_marbles : ℕ) : 
  initial_marbles = 26 →
  found_marbles = 6 →
  lost_marbles = 10 →
  given_marbles = 2 * lost_marbles →
  (initial_marbles + found_marbles - lost_marbles + given_marbles) = 42 :=
by
  intros,
  sorry

end hilton_final_marbles_l163_163754


namespace solve_for_y_l163_163852
-- Import the necessary Lean 4 libraries

-- Define the problem conditions and the proof statement
theorem solve_for_y (y : ℤ) (h : 3^(y - 2) = 9^(y + 2)) : y = -6 :=
begin
  -- The statement requires us to show y = -6 given the initial condition
  sorry
end

end solve_for_y_l163_163852


namespace ellipse_focal_length_l163_163048

theorem ellipse_focal_length {m : ℝ} : 
  (m > 2 ∧ 4 ≤ 10 - m ∧ 4 ≤ m - 2) → 
  (10 - m - (m - 2) = 4) ∨ (m - 2 - (10 - m) = 4) :=
by
  sorry

end ellipse_focal_length_l163_163048


namespace inequality_proof_l163_163368

theorem inequality_proof
  (x y z : ℝ) (hxpos : 0 < x) (hypos : 0 < y) (hzpos : 0 < z)
  (hineq : x * y + y * z + z * x ≤ 1) :
  (x + 1 / x) * (y + 1 / y) * (z + 1 / z) ≥ 8 * (x + y) * (y + z) * (z + x) :=
sorry

end inequality_proof_l163_163368


namespace Lyle_friends_sandwich_juice_l163_163501

/-- 
Lyle wants to buy himself and his friends a sandwich and a pack of juice. 
A sandwich costs $0.30 while a pack of juice costs $0.20. Given Lyle has $2.50, 
prove that he can buy sandwiches and juice for 4 of his friends.
-/
theorem Lyle_friends_sandwich_juice :
  let sandwich_cost := 0.30
  let juice_cost := 0.20
  let total_money := 2.50
  let total_cost_one_set := sandwich_cost + juice_cost
  let total_sets := total_money / total_cost_one_set
  total_sets - 1 = 4 :=
by
  sorry

end Lyle_friends_sandwich_juice_l163_163501


namespace boys_less_than_two_fifths_total_l163_163261

theorem boys_less_than_two_fifths_total
  (n b g n1 n2 b1 b2 : ℕ)
  (h_total: n = b + g)
  (h_first_trip: b1 < 2 * n1 / 5)
  (h_second_trip: b2 < 2 * n2 / 5)
  (h_participation: b ≤ b1 + b2)
  (h_total_participants: n ≤ n1 + n2) :
  b < 2 * n / 5 := 
sorry

end boys_less_than_two_fifths_total_l163_163261


namespace x_represents_uphill_length_l163_163587

-- Defining the conditions given in the problem
def uphill_speed : ℝ := 3
def flat_speed  : ℝ := 4
def downhill_speed : ℝ := 5
def time_AB : ℝ := 36 / 60
def time_BA : ℝ := 24 / 60

-- System of equations
def equation1 (x y : ℝ) : Prop := x / uphill_speed + y / flat_speed = time_AB
def equation2 (x y : ℝ) : Prop := x / downhill_speed + y / flat_speed = time_BA

-- The proof statement: Prove that x represents the length of the uphill section from location A to location B
theorem x_represents_uphill_length (x y : ℝ) (h1 : equation1 x y) (h2 : equation2 x y) : 
  ∃ (u : ℝ), u = x := sorry

end x_represents_uphill_length_l163_163587


namespace quadratic_max_m_l163_163373

theorem quadratic_max_m (m : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → (m * x^2 - 2 * m * x + 2) ≤ 4) ∧ 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 2 ∧ (m * x^2 - 2 * m * x + 2) = 4) ∧ 
  m ≠ 0 → 
  (m = 2 / 3 ∨ m = -2) := 
by
  sorry

end quadratic_max_m_l163_163373


namespace largest_inscribed_pentagon_is_regular_l163_163531

theorem largest_inscribed_pentagon_is_regular :
  ∀ (A_1 A_2 A_3 A_4 A_5 : ℝ × ℝ) (O : ℝ × ℝ), 
    dist O A_1 = 1 ∧ dist O A_2 = 1 ∧ dist O A_3 = 1 ∧ dist O A_4 = 1 ∧ dist O A_5 = 1 →
    (∃ r : ℝ, r > 0 ∧ ∀ (B_1 B_2 B_3 B_4 B_5 : ℝ × ℝ), 
       dist O B_1 = r ∧ dist O B_2 = r ∧ dist O B_3 = r ∧ dist O B_4 = r ∧ dist O B_5 = r →
          is_regular_pentagon B_1 B_2 B_3 B_4 B_5) :=
begin
  sorry
end

noncomputable def is_regular_pentagon (A_1 A_2 A_3 A_4 A_5 : ℝ × ℝ) : Prop :=
  let P := [A_1, A_2, A_3, A_4, A_5] in
  (∀ i : ℤ, 0 ≤ i ∧ i < int.of_nat 5 → dist (P.nth_le ( (i:int).to_nat % 5) sorry) (P.nth_le ( (i + 1:int).to_nat % 5) sorry) = dist (P.nth_le (0) sorry) (P.nth_le (1) sorry)) ∧
  (∀ i : ℤ, 0 ≤ i ∧ i < int.of_nat 5 → ∃ θ : ℝ, ∠ (P.nth_le ( (i:int).to_nat % 5) sorry) (P.nth_le (0) sorry) (P.nth_le ( (i + 1:int):.to_nat % 5) sorry) = θ ∧ θ = (2 * (Real.pi) / int.of_nat 5))


end largest_inscribed_pentagon_is_regular_l163_163531


namespace Lucas_cell_phone_bill_l163_163271

theorem Lucas_cell_phone_bill :
  ∀ (base_cost texts_cost_per talk_cost_1 talk_cost_2 num_texts talk_hours),
    base_cost = 25 →
    texts_cost_per = 0.10 →
    talk_cost_1 = 0.15 →
    talk_cost_2 = 0.20 →
    num_texts = 150 →
    talk_hours = 31.5 →
    let texts_cost := num_texts * texts_cost_per,
        overage_hours := max (talk_hours - 30) 0,
        overage_minutes := overage_hours * 60,
        overage_minutes_1 := min overage_minutes 60,
        overage_minutes_2 := max (overage_minutes - 60) 0,
        cost_1 := overage_minutes_1 * talk_cost_1,
        cost_2 := overage_minutes_2 * talk_cost_2,
        total_cost := base_cost + texts_cost + cost_1 + cost_2
    in total_cost = 55 :=
by 
  intros base_cost texts_cost_per talk_cost_1 talk_cost_2 num_texts talk_hours
  intro h_base h_texts_per h_talk1 h_talk2 h_num_texts h_talk_hours
  let texts_cost := num_texts * texts_cost_per
  let overage_hours := max (talk_hours - 30) 0
  let overage_minutes := overage_hours * 60
  let overage_minutes_1 := min overage_minutes 60
  let overage_minutes_2 := max (overage_minutes - 60) 0
  let cost_1 := overage_minutes_1 * talk_cost_1
  let cost_2 := overage_minutes_2 * talk_cost_2
  let total_cost := base_cost + texts_cost + cost_1 + cost_2
  have : total_cost = 55 := sorry
  exact this

end Lucas_cell_phone_bill_l163_163271


namespace train_speed_is_60_kmph_l163_163961

noncomputable def speed_of_train_in_kmph (length_meters time_seconds : ℝ) : ℝ :=
  (length_meters / time_seconds) * 3.6

theorem train_speed_is_60_kmph (length_meters time_seconds : ℝ) :
  length_meters = 50 → time_seconds = 3 → speed_of_train_in_kmph length_meters time_seconds = 60 :=
by
  intros h_length h_time
  simp [speed_of_train_in_kmph, h_length, h_time]
  norm_num
  sorry

end train_speed_is_60_kmph_l163_163961


namespace decahedron_consecutive_number_probability_l163_163883

-- Definitions based on problem conditions
def faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def are_consecutive (a b : ℕ) : Prop := (a = 10 ∧ b = 1) ∨ (b = 10 ∧ a = 1) ∨ (a + 1 = b) ∨ (b + 1 = a)

def adjacent_faces : List (ℕ × ℕ) := [-- list of all adjacent face pairs on a regular decahedron
  -- this list should contain all pairs (i, j) where the faces i and j share an edge
  -- omitted for simplicity, actual implementation would list all pairs
]

-- Problem statement: Prove the required probability
theorem decahedron_consecutive_number_probability :
  ∃ (m n : ℕ), m = 1 ∧ n = 90 ∧ RelPrime m n ∧
  no_consecutive_adjacent_placement_probability faces adjacent_faces are_consecutive = (m / n) :=
by
  sorry

end decahedron_consecutive_number_probability_l163_163883


namespace chef_initial_eggs_l163_163192

-- Define the conditions
def eggs_in_fridge := 10
def eggs_per_cake := 5
def cakes_made := 10

-- Prove that the number of initial eggs is 60
theorem chef_initial_eggs : (eggs_per_cake * cakes_made + eggs_in_fridge) = 60 :=
by
  sorry

end chef_initial_eggs_l163_163192


namespace boat_upstream_distance_l163_163268

theorem boat_upstream_distance
  (V_s : ℝ) (distance_downstream : ℝ) (time_downstream : ℝ) (time_upstream : ℝ)
  (h_stream_speed : V_s = 3.75)
  (h_distance_downstream : distance_downstream = 100)
  (h_time_downstream : time_downstream = 8)
  (h_time_upstream : time_upstream = 15) :
  let V_b := (distance_downstream / time_downstream) - V_s in
  (V_b - V_s) * time_upstream = 75 :=
by
  sorry

end boat_upstream_distance_l163_163268


namespace gen_formula_arithmetic_seq_sum_of_abs_arithmetic_seq_l163_163478

-- Definition of an arithmetic sequence and its sum
def arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ :=
  a₁ + d * (n - 1)

def arithmetic_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

-- The given conditions
variable (a₂_value : ℤ := 11)
variable (S₁₀_value : ℤ := 40)

-- The general formula for the sequence given the conditions
theorem gen_formula_arithmetic_seq :
  ∃ a₁ d, (a₁ + d = a₂_value) ∧
          (10 * a₁ + (10 * 9 / 2) * d = S₁₀_value) ∧
          (∀ n, arithmetic_seq a₁ d n = -2 * n + 15) :=
sorry

-- The sum of absolute values of the first n terms of the sequence
def abs_arithmetic_seq (a₁ d : ℤ) (n : ℕ) : ℤ :=
  abs (arithmetic_seq a₁ d n)

def abs_arithmetic_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  ∑ i in Finset.range n, abs_arithmetic_seq a₁ d (i + 1)

theorem sum_of_abs_arithmetic_seq (n : ℕ) :
  let T_n : ℤ :=
    if n ≤ 7 then -n^2 + 14 * n
    else n^2 - 14 * n + 98 in
  ∃ a₁ d, (a₁ + d = a₂_value) ∧
           (10 * a₁ + (10 * 9 / 2) * d = S₁₀_value) ∧
           (∀ n, arithmetic_seq a₁ d n = -2 * n + 15) ∧
           (abs_arithmetic_sum a₁ d n = T_n) :=
sorry

end gen_formula_arithmetic_seq_sum_of_abs_arithmetic_seq_l163_163478


namespace not_necessarily_best_for_fly_l163_163644

noncomputable def cube : Type := sorry

structure Point (num : Type) :=
  (x : num)
  (y : num)
  (z : num)

def is_midpoint_of_edge (p : Point ℝ) (a b : Point ℝ) : Prop := sorry
def symmetric_opposite_to_center (p : Point ℝ) (cube_length : ℝ) : Point ℝ := sorry
def shortest_path_on_cube (a b : Point ℝ) (cube : cube) : ℝ := sorry

theorem not_necessarily_best_for_fly (spider fly : Point ℝ) (length : ℝ) 
  (h_len : length = 1)
  (h_spider : is_midpoint_of_edge spider (Point.mk 0 0 0) (Point.mk 1 0 0)) 
  (h_fly_surface : fly.z <= 1) 
  (h_opposite : fly = symmetric_opposite_to_center spider length) : 
  ∃ (other_fly : Point ℝ), shortest_path_on_cube spider fly cube < shortest_path_on_cube spider other_fly cube :=
sorry

end not_necessarily_best_for_fly_l163_163644


namespace units_digit_smallest_n_l163_163825

theorem units_digit_smallest_n (n : ℕ) (h1 : 7 * n ≥ 10^2015) (h2 : 7 * (n - 1) < 10^2015) : (n % 10) = 6 :=
sorry

end units_digit_smallest_n_l163_163825


namespace janet_spending_difference_l163_163806

-- Definitions for the conditions
def clarinet_hourly_rate : ℝ := 40
def clarinet_hours_per_week : ℝ := 3
def piano_hourly_rate : ℝ := 28
def piano_hours_per_week : ℝ := 5
def weeks_per_year : ℕ := 52

-- The theorem to be proven
theorem janet_spending_difference :
  (piano_hourly_rate * piano_hours_per_week * weeks_per_year - clarinet_hourly_rate * clarinet_hours_per_week * weeks_per_year) = 1040 :=
by
  sorry

end janet_spending_difference_l163_163806


namespace num_elements_A_l163_163827

def A : Set ℝ := { x | (x - 1) * (x - 2) ^ 2 = 0 }

theorem num_elements_A : Fintype.card ↥A = 2 := 
sorry

end num_elements_A_l163_163827


namespace prove_solutions_l163_163856

noncomputable def solve_equation (x : ℝ) : Prop :=
  sqrt (sqrt (1 - cos (15 * x) ^ 7 * cos (9 * x) ^ 2)) = sin (9 * x)

theorem prove_solutions (x : ℝ) (n s : ℤ) (h : sin (9 * x) ≥ 0) :
  solve_equation x →
  (∃ n : ℤ, x = π / 18 + 2 * π * n / 9) ∨ (∃ s : ℤ, x = 2 * π * s / 3) :=
begin
  sorry
end

end prove_solutions_l163_163856


namespace Lyle_can_buy_for_his_friends_l163_163506

theorem Lyle_can_buy_for_his_friends
  (cost_sandwich : ℝ) (cost_juice : ℝ) (total_money : ℝ)
  (h1 : cost_sandwich = 0.30)
  (h2 : cost_juice = 0.20)
  (h3 : total_money = 2.50) :
  (total_money / (cost_sandwich + cost_juice)).toNat - 1 = 4 :=
by
  sorry

end Lyle_can_buy_for_his_friends_l163_163506


namespace solve_system_equations_l163_163185

theorem solve_system_equations (x y : ℝ) :
  (x^2 + y^2 ≠ 0) →
  (6 / (x^2 + y^2) + x^2 * y^2 = 10) →
  (x^4 + y^4 + 7 * x^2 * y^2 = 81) →
  (x = sqrt 3 ∧ y = sqrt 3) ∨ (x = sqrt 3 ∧ y = -sqrt 3) ∨ (x = -sqrt 3 ∧ y = sqrt 3) ∨ (x = -sqrt 3 ∧ y = -sqrt 3) :=
sorry

end solve_system_equations_l163_163185


namespace meet_point_distance_l163_163802

noncomputable def jack_time_up (k : ℝ) : ℝ := 3 / 12 + 0.25 + 3 / 12
noncomputable def jill_time_up (k : ℝ) : ℝ := 6 / 15

noncomputable def jack_position (t : ℝ) : ℝ :=
  if t <= 0.25 + 3 / 12
  then 12 * t
  else 6 -  18 * (t - (0.25 + 3 / 12))

noncomputable def jill_position (t : ℝ) : ℝ := 15 * t

theorem meet_point_distance (k jack_time_up jill_time_up : ℝ) (h1 : ht = jack_time_up) (h2 : hz = jill_time_up):
  (6 - jack_position t = jill_position t ) =
  (6 - 6 * (15 * ((3/4 + (13/22 * (6 - 3/22.


end meet_point_distance_l163_163802


namespace sum_of_c_with_8_solutions_l163_163322

noncomputable def g (x : ℝ) : ℝ :=
  (x - 6) * (x - 4) * (x - 2) * x * (x + 2) * (x + 4) / 360 - 5

theorem sum_of_c_with_8_solutions : 
  ∑ c in {c : ℤ | ∃! x, g x = (c : ℝ)} (finite_of_finitely_many _), (c : ℝ) = -9 :=
sorry

end sum_of_c_with_8_solutions_l163_163322


namespace volume_of_tetrahedron_abcd_l163_163525

theorem volume_of_tetrahedron_abcd
  (P Q R S A B C D : Type)
  [Inhabited P] [Inhabited Q] [Inhabited R] [Inhabited S]
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (h1 : distance P Q = 7)
  (h2 : distance R S = 7)
  (h3 : distance P R = 8)
  (h4 : distance Q S = 8)
  (h5 : distance P S = 9)
  (h6 : distance Q R = 9)
  (inscribed_center : ∀ (X Y Z : Type), (distance X Y) * (distance Y Z) * (distance Z X) ≠ 0 → Type)
  (inscribed_center_PQS : inscribed_center P Q S = A)
  (inscribed_center_PRS : inscribed_center P R S = B)
  (inscribed_center_QRS : inscribed_center Q R S = C)
  (inscribed_center_PQR : inscribed_center P Q R = D) :
  volume A B C D = (5 * Real.sqrt 11) / 9 := sorry

end volume_of_tetrahedron_abcd_l163_163525


namespace solve_system_l163_163546

noncomputable def solution_exists : Prop :=
  ∃ (x y : ℝ), x + 3 * y + 14 ≤ 0 ∧ x^4 + 2 * x^2 * y^2 + y^4 + 64 - 20 * x^2 - 20 * y^2 = 8 * x * y

theorem solve_system : solution_exists → ∃ (x y : ℝ), x = -2 ∧ y = -4 :=
by
  intro h
  cases' h with x h
  cases' h with y h
  use [-2, -4]
  constructor
  . sorry
  . sorry

end solve_system_l163_163546


namespace selling_price_of_article_l163_163944

theorem selling_price_of_article (cost_price gain_percent : ℝ) (h1 : cost_price = 100) (h2 : gain_percent = 30) : 
  cost_price + (gain_percent / 100) * cost_price = 130 := 
by 
  sorry

end selling_price_of_article_l163_163944


namespace greatest_possible_overlap_l163_163266

variable {P Q : Type} -- Assume P and Q represent predicates for airlines offering specific services.

def percent_equip_wi (p : P) : Prop := 0.5 -- 50% of companies equip with wireless internet
def percent_offer_snacks (p : P) : Prop := 0.7 -- 70% of companies offer free snacks

theorem greatest_possible_overlap (p : P) :
  percent_equip_wi p ∧ percent_offer_snacks p → (p = 0.5) := sorry

end greatest_possible_overlap_l163_163266


namespace profit_formula_l163_163272

noncomputable def W (x : ℝ) : ℝ :=
if (0 ≤ x ∧ x ≤ 2) then 5 * (x * x + 3)
else if (2 < x ∧ x ≤ 5) then 50 * x / (1 + x)
else 0

noncomputable def profit (W : ℝ → ℝ) (x : ℝ) : ℝ :=
if (0 ≤ x ∧ x ≤ 2) then 15 * W(x) - 30 * x
else if (2 < x ∧ x ≤ 5) then 15 * W(x) - 30 * x
else 0

theorem profit_formula (x : ℝ) (h₁ : 0 ≤ x ∧ x ≤ 2)
  (h₂ : 2 < x ∧ x ≤ 5) :
  (profit W x) =
  (if (0 ≤ x ∧ x ≤ 2) then 75 * x * x - 30 * x + 225
  else if (2 < x ∧ x ≤ 5) then 750 * x / (1 + x) - 30 * x
  else 0) :=
by { sorry }

end profit_formula_l163_163272


namespace expansion_coefficient_l163_163118

theorem expansion_coefficient (x : ℝ) (h : x ≠ 0): 
  (∃ r : ℕ, (7 - (3 / 2 : ℝ) * r = 1) ∧ Nat.choose 7 r = 35) := 
  sorry

end expansion_coefficient_l163_163118


namespace lyle_friends_sandwich_and_juice_l163_163504

theorem lyle_friends_sandwich_and_juice : 
  ∀ (sandwich_cost juice_cost lyle_money : ℝ),
    sandwich_cost = 0.30 → 
    juice_cost = 0.20 → 
    lyle_money = 2.50 → 
    (⌊lyle_money / (sandwich_cost + juice_cost)⌋.toNat - 1) = 4 :=
by
  intros sandwich_cost juice_cost lyle_money hc_sandwich hc_juice hc_money
  have cost_one_set := sandwich_cost + juice_cost
  have number_of_sets := lyle_money / cost_one_set
  have friends := (number_of_sets.toNat - 1)
  have friends_count := 4
  sorry

end lyle_friends_sandwich_and_juice_l163_163504


namespace arithmetic_seq_a7_a8_a9_l163_163731

variable {α : Type*} [linear_ordered_field α]

def arithmetic_seq_sum (a₁ d : α) (n : ℕ) : α :=
  n / 2 * (2 * a₁ + (n - 1) * d)

theorem arithmetic_seq_a7_a8_a9 (a₁ d : α) (n : ℕ) (S S₆ : α) (hS : S = 9) (hS₆ : S₆ = 36) :
  S₆ = arithmetic_seq_sum a₁ d 6 ∧ S = arithmetic_seq_sum a₁ d n →
  3 * (a₁ + 6 * d) + 3 * (a₁ + 7 * d) + 3 * (a₁ + 8 * d) = 45 :=
sorry

end arithmetic_seq_a7_a8_a9_l163_163731


namespace find_number_l163_163010

theorem find_number (some_number : ℤ)
    (h : some_number ” 3 = 4)
    (symbol_def : ∀ m n : ℤ, m ” n = n^2 - m) :
    some_number = 5 :=
by
  sorry

end find_number_l163_163010


namespace smallest_sum_A_b_l163_163761

theorem smallest_sum_A_b (A b : ℕ) (hA : A ∈ finset.range 4) (hb : b > 5)
    (h_eq : 21 * A = 3 * b + 3) : A + b = 7 :=
sorry

end smallest_sum_A_b_l163_163761


namespace find_n_l163_163791

-- Define the arithmetic sequence {a_n} and the sum S_5 of the first 5 terms of the sequence {log_2 a_n}
noncomputable def a (n : ℕ) : ℕ := 2 * n
noncomputable def log2 (x : ℕ) : ℝ := Real.logb 2 (x : ℝ)
noncomputable def S_5 : ℝ := (List.sum (List.map (fun n => log2 (a n)) (List.range 5))) + 5

-- State the main theorem
theorem find_n (n : ℤ) (hn : S_5 ∈ Set.Ico n (n+1)) : n = 11 := 
by sorry

end find_n_l163_163791


namespace number_of_subsets_is_four_l163_163831

-- Define the set M
def M : Set ℕ := {0, 1}

-- State the theorem
theorem number_of_subsets_is_four : (Finset.powerset M.to_finset).card = 4 :=
sorry

end number_of_subsets_is_four_l163_163831


namespace surfaces_proportional_to_volumes_l163_163179

noncomputable def surface_area_dodecahedron (R : ℝ) : ℝ :=
  2 * R^2 * real.sqrt (10 * (5 - real.sqrt 5))

noncomputable def surface_area_icosahedron (R : ℝ) : ℝ :=
  2 * R^2 * (5 * real.sqrt 3 - real.sqrt 15)

noncomputable def volume_dodecahedron (R : ℝ) : ℝ :=
  (2 / 9) * R^3 * real.sqrt (30 * (3 + real.sqrt 5))

noncomputable def volume_icosahedron (R : ℝ) : ℝ :=
  (2 / 3) * R^3 * (10 + 2 * real.sqrt 5)

theorem surfaces_proportional_to_volumes (R : ℝ) (hR : R > 0) :
  (surface_area_dodecahedron R / surface_area_icosahedron R) =
  (volume_dodecahedron R / volume_icosahedron R) :=
by
  sorry

end surfaces_proportional_to_volumes_l163_163179


namespace sum_of_coordinates_D_l163_163167

theorem sum_of_coordinates_D (M C D : ℝ × ℝ)
  (h1 : M = (5, 5))
  (h2 : C = (10, 10))
  (h3 : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 + D.2 = 0 := 
sorry

end sum_of_coordinates_D_l163_163167


namespace find_y_value_l163_163425

-- Define the problem statement in Lean 4
theorem find_y_value (y : ℝ) (h : y = Real.sqrt (2 - Real.sqrt (2 + Real.sqrt (2 - Real.sqrt (2 + ...)))) :
  y = (1 + Real.sqrt 5) / 2 :=
sorry

end find_y_value_l163_163425


namespace min_max_xy_yz_zx_minus_3xyz_l163_163726

theorem min_max_xy_yz_zx_minus_3xyz (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  (min_value : ℝ) ∧ (max_value : ℝ) :=
  begin
    have h_f := λ x y z, x * y + y * z + z * x - 3 * x * y * z,
    use (0 : ℝ),
    use (1 / 4 : ℝ),
    split,
    { -- Proof that 0 is the minimum value
      sorry },
    { -- Proof that 1/4 is the maximum value
      sorry }
  end

end min_max_xy_yz_zx_minus_3xyz_l163_163726


namespace travel_time_reduction_no_effect_l163_163580

-- Definitions and conditions
def cities : ℕ := 11
def capital : string := "Capital"
def time_to_capital : ℕ := 7
def time_cyclic_routes : ℕ := 3
def initial_transfer_time : ℕ := 2
def reduced_transfer_time : ℚ := 1.5

-- The maximal travel time between two cities A and B
def optimized_travel_time (x : ℚ) : ℚ :=
  if x = initial_transfer_time then 14 + x 
  else if x = reduced_transfer_time then 15.5 
  else 21

-- The proof problem in Lean 4 statement
theorem travel_time_reduction_no_effect :
  optimized_travel_time reduced_transfer_time = 15.5 :=
  sorry

end travel_time_reduction_no_effect_l163_163580


namespace overflow_fraction_of_hemispherical_bowl_l163_163942

theorem overflow_fraction_of_hemispherical_bowl (R : ℝ) (h₀ : R > 0) :
  let V_hemisphere := (2 / 3) * Real.pi * R^3,
      V_cap := (5 / 24) * Real.pi * R^3,
      V_overflow := V_hemisphere - V_cap in
  V_overflow / V_hemisphere = 11 / 16 :=
by
  -- Here we will define V_hemisphere, V_cap, V_overflow
  -- and show that V_overflow / V_hemisphere = 11 / 16
  sorry

end overflow_fraction_of_hemispherical_bowl_l163_163942


namespace arbitrarily_large_ratios_l163_163824

open Nat

theorem arbitrarily_large_ratios (a : ℕ → ℕ) (h_distinct: ∀ m n, m ≠ n → a m ≠ a n)
  (h_no_100_ones: ∀ n, ¬ (∃ k, a n / 10^k % 10^100 = 10^100 - 1)):
  ∀ M : ℕ, ∃ n : ℕ, a n / n ≥ M :=
by
  sorry

end arbitrarily_large_ratios_l163_163824


namespace Gary_books_ratio_l163_163417

def Harry_books : ℕ := 50
def Flora_books : ℕ := 2 * Harry_books
def Total_books : ℕ := 175
def Gary_books (R : ℚ) : ℚ := R * Harry_books

theorem Gary_books_ratio :
  Harry_books + Flora_books + Gary_books (1/2) = Total_books →
  (∃ R : ℚ, Gary_books R = Gary_books (1/2)) :=
begin
  intros h,
  use 1/2,
  calc 
    Gary_books (1/2) = (1/2) * Harry_books : by refl
    ... = 25 : by norm_num,
  sorry
end

end Gary_books_ratio_l163_163417


namespace total_money_made_l163_163932

structure Building :=
(floors : Nat)
(rooms_per_floor : Nat)

def cleaning_time_per_room : Nat := 8

structure CleaningRates :=
(first_4_hours_rate : Int)
(next_4_hours_rate : Int)
(unpaid_break_hours : Nat)

def supply_cost : Int := 1200

def total_earnings (b : Building) (c : CleaningRates) : Int :=
  let rooms := b.floors * b.rooms_per_floor
  let earnings_per_room := (4 * c.first_4_hours_rate + 4 * c.next_4_hours_rate)
  rooms * earnings_per_room - supply_cost

theorem total_money_made (b : Building) (c : CleaningRates) : 
  b.floors = 12 →
  b.rooms_per_floor = 25 →
  cleaning_time_per_room = 8 →
  c.first_4_hours_rate = 20 →
  c.next_4_hours_rate = 25 →
  c.unpaid_break_hours = 1 →
  total_earnings b c = 52800 := 
by
  intros
  sorry

end total_money_made_l163_163932


namespace sin_condition_l163_163190

theorem sin_condition (α : ℝ) :
  |sin α| = -sin α → ∃ k : ℤ, α ∈ Set.Icc (2 * (k : ℝ) * Real.pi - Real.pi) (2 * (k : ℝ) * Real.pi) :=
by
  sorry

end sin_condition_l163_163190


namespace number_of_possible_committees_l163_163194

theorem number_of_possible_committees 
  (num_departments : ℕ)
  (males_per_department : ℕ)
  (females_per_department : ℕ)
  (nonbinary_per_department : ℕ)
  (total_committees : ℕ)
  (gender_count_per_committee : ℕ)
  (dept_count_per_committee : ℕ)
  (total_members_per_committee : ℕ)
  (case1 : ℕ)
  (case2 : ℕ)
  (correct_answer : ℕ) :
  num_departments = 3 →
  males_per_department = 3 →
  females_per_department = 3 →
  nonbinary_per_department = 3 →
  gender_count_per_committee = 3 →
  dept_count_per_committee = 3 →
  total_members_per_committee = 9 →
  case1 = 27^3 →
  case2 = 3! →
  correct_answer = 19683 + 6 →
  total_committees = correct_answer :=
by
  intros
  have h1 : case1 = 27^3 := by assumption
  have h2 : case2 = 3! := by assumption
  have h3 : correct_answer = 19683 + 6 := by assumption
  show total_committees = correct_answer, from sorry

end number_of_possible_committees_l163_163194


namespace product_and_sum_of_roots_l163_163351

theorem product_and_sum_of_roots :
  let a := 24
  let b := 60
  let c := -600
  (c / a = -25) ∧ (-b / a = -2.5) := 
by
  sorry

end product_and_sum_of_roots_l163_163351


namespace problem_1_problem_2_l163_163620

-- Problem 1
theorem problem_1 : 
  ∛(-27) - Real.sqrt ((-2:ℝ)^2) + abs (1 - Real.sqrt 2) = -6 + Real.sqrt 2 :=
by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : (x - 1)^3 = -8) : x = -1 :=
by
  sorry

end problem_1_problem_2_l163_163620


namespace sin_alpha_l163_163432

noncomputable def cos60 : ℝ := 1 / 2
noncomputable def sin45 : ℝ := real.sqrt 2 / 2

def P : ℝ × ℝ := (-2 * cos60, -real.sqrt 2 * sin45)

def r : ℝ := real.sqrt (P.1^2 + P.2^2)

theorem sin_alpha (α : ℝ) (h₁ : ∃ α, (cos α, sin α) = (P.1 / r, P.2 / r))
  : sin α = - real.sqrt 2 / 2 := by
  sorry

end sin_alpha_l163_163432


namespace find_number_l163_163845

theorem find_number (n : ℤ) (h : 7 * n = 3 * n + 12) : n = 3 :=
sorry

end find_number_l163_163845


namespace negation_of_proposition_l163_163213

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^2 + 3*x + 2 < 0) ↔ ∃ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
by
  sorry

end negation_of_proposition_l163_163213


namespace three_digit_number_count_l163_163316

noncomputable def total_three_digit_numbers : ℕ :=
  let chosen_from_024 := {0, 2, 4}
  let chosen_from_135 := {1, 3, 5}
  let all_possible_digits := chosen_from_024 ∪ chosen_from_135
  let total_combinations := 
    (card chosen_from_024) * (card chosen_from_135).choose 2 * (card all_possible_digits - 3).factorial
  total_combinations

theorem three_digit_number_count : total_three_digit_numbers = 48 :=
by
  sorry

end three_digit_number_count_l163_163316


namespace smallest_a_for_nonprime_l163_163352

theorem smallest_a_for_nonprime (a : ℕ) : (∀ x : ℤ, ∃ d : ℤ, d ∣ (x^4 + a^4) ∧ d ≠ 1 ∧ d ≠ (x^4 + a^4)) ↔ a = 3 := by
  sorry

end smallest_a_for_nonprime_l163_163352


namespace perfect_squares_count_l163_163006

theorem perfect_squares_count :
  let N := 100
  let is_perfect_square (x : Nat) := ∃ m : Nat, m * m = x
  let fulfill_condition (n : Nat) := n^3 + 5 * n^2
  in 
  Nat.card { n // n > 0 ∧ n ≤ N ∧ is_perfect_square (fulfill_condition n)} = 8 :=
by sorry

end perfect_squares_count_l163_163006


namespace trig_equation_solution_l163_163855

theorem trig_equation_solution
  (ϕ : ℝ)
  (x y : ℝ)
  (hx : x = Math.sin ϕ)
  (hy : y = Math.cos ϕ)
  (hxy : x^2 + y^2 = 1) :
  3 * x * y + 4 * x + 3 * y^2 - y - 4 = 0 := by
  sorry

end trig_equation_solution_l163_163855


namespace correct_operation_B_l163_163246

theorem correct_operation_B (x : ℝ) : 
  x - 2 * x = -x :=
sorry

end correct_operation_B_l163_163246


namespace two_f_of_x_l163_163766

noncomputable def f (x : ℝ) : ℝ := 3 / (3 + x)

theorem two_f_of_x (x : ℝ) (h : x > 0) : 2 * f x = 18 / (9 + x) :=
  sorry

end two_f_of_x_l163_163766


namespace expected_flips_is_N_plus_one_div_two_l163_163938

def expected_flips_to_second_A (N : ℕ) : ℚ :=
  (N + 1) / 2

theorem expected_flips_is_N_plus_one_div_two 
  (N : ℕ) (hN : N ≥ 3) :
  ∀ (deck : list ℕ) (hA : deck.count 1 = 3), -- Assume 1 represents the 'A' card
  (randomly_shuffled deck).cards_flipped_to_second_A.expected_value = expected_flips_to_second_A N := 
sorry

end expected_flips_is_N_plus_one_div_two_l163_163938


namespace possible_values_of_n_l163_163686

theorem possible_values_of_n (E M n : ℕ) (h1 : M + 3 = n * (E - 3)) (h2 : E + n = 3 * (M - n)) : 
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7 :=
sorry

end possible_values_of_n_l163_163686


namespace sum_first_80_terms_l163_163571

variable {a : ℕ → ℝ}

-- Condition
axiom sequence_condition : ∀ n : ℕ, a (n + 1) + (-1)^n * a n = 2 * n - 1

-- Theorem statement
theorem sum_first_80_terms : (finset.range 80).sum (λ n, a (n + 1)) = 3240 :=
sorry

end sum_first_80_terms_l163_163571


namespace inequality_transformation_range_of_a_l163_163052

-- Define the given function f(x) = |x + 2|
def f (x : ℝ) : ℝ := abs (x + 2)

-- State the inequality transformation problem
theorem inequality_transformation (x : ℝ) :  (2 * abs (x + 2) < 4 - abs (x - 1)) ↔ (-7 / 3 < x ∧ x < -1) :=
by sorry

-- State the implication problem involving m, n, and a
theorem range_of_a (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (hmn : m + n = 1) (a : ℝ) :
  (∀ x : ℝ, abs (x - a) - f x ≤ 1 / m + 1 / n) → (-6 ≤ a ∧ a ≤ 2) :=
by sorry

end inequality_transformation_range_of_a_l163_163052


namespace given_x_gt_0_l163_163712

theorem given_x_gt_0 (x : ℕ → ℝ) (h_pos : ∀ i, x i > 0) (h_n : ∃ n, n ≥ 5) :
  ∃ n, (∑ i in finset.range n, x i * x (i + 1) / (x i ^ 2 + x (i + 1) ^ 2 + 2 * x (i + 2) * x (i + 3))) ≤ (n - 1) / 2 :=
sorry

end given_x_gt_0_l163_163712


namespace expression_never_zero_l163_163004

theorem expression_never_zero (a : ℚ) : |a| + 1 ≠ 0 := 
by {
  intro h,
  have h_abs : |a| ≥ 0 := abs_nonneg a,
  linarith,
}

end expression_never_zero_l163_163004
