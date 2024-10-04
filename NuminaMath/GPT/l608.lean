import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.GroupPower
import Mathlib.Algebra.GroupPower.Basic
import Mathlib.Algebra.Order
import Mathlib.Algebra.Sequence
import Mathlib.Analysis.Calculus.Fderiv
import Mathlib.Analysis.Calculus.Limits
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Polynomial
import Mathlib.Data.Prob.ProbabilityMassFunction
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.LinearAlgebra.Basis
import Mathlib.Probability.Basic
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Rewrite
import data.real.basic

namespace ratio_of_areas_l608_608647

open Real

variables (A B C D E : Type) [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] [EuclideanGeometry D] [EuclideanGeometry E]
variables (triangleABC : Triangle A B C)
variables (angleBAC : Angle A B C = 60)
variable (circleBC : Circle B C)
variable (intersectABatD : AB ∩ circleBC = {D})
variable (intersectACatE : AC ∩ circleBC = {E})

theorem ratio_of_areas (h0 : acute_angled_triangle A B C)
                       (h1 : ∡ A B C = 60)
                       (h2 : is_diameter circleBC B C)
                       (h3 : circle_intersects_at D E A B A C circleBC) :
  area_ratio B D E C A B C = 3 / 4 :=
sorry

end ratio_of_areas_l608_608647


namespace ratio_square_of_shortest_to_medium_edge_l608_608528

theorem ratio_square_of_shortest_to_medium_edge (a b c : ℝ) (d : ℝ) 
  (h1 : a / b = b / c) 
  (h2 : b / c = c / d) 
  (h3 : d = sqrt (a^2 + b^2)) : 
  (a / b)^2 = (sqrt 5 - 1) / 2 :=
by sorry

end ratio_square_of_shortest_to_medium_edge_l608_608528


namespace least_integer_sum_ratios_l608_608429

theorem least_integer_sum_ratios :
  ∃ x : ℚ, (∃ a b c : ℚ, a + b + c = 90 ∧ a = x ∧ b = 2 * x ∧ c = 5 * x ∧ a = 45 / 4) :=
begin
  -- proof would be here
  sorry,
end

end least_integer_sum_ratios_l608_608429


namespace probability_x_plus_y_less_than_4_l608_608105

open Set Real

/-- Define the square with vertices (0,0), (0,3), (3,3), and (3,0) -/
def square := set.Icc (0 : ℝ) 3 ×ˢ set.Icc (0 : ℝ) 3

/-- Probability that a randomly chosen point (x, y) in the square satisfies x + y < 4 -/
theorem probability_x_plus_y_less_than_4 : 
  let area_square := 9 in
  let area_excluded_triangle := 2 in
  let relevant_area := area_square - area_excluded_triangle in
  (relevant_area / area_square : ℝ) = 7 / 9 :=
by
{ sorry }

end probability_x_plus_y_less_than_4_l608_608105


namespace weight_AlBr3_correct_l608_608031

-- Define atomic masses as constants
def atomic_mass_Al : ℝ := 26.98 -- g/mol
def atomic_mass_Br : ℝ := 79.90 -- g/mol

-- Define the molar mass of AlBr3
def molar_mass_AlBr3 : ℝ := atomic_mass_Al + 3 * atomic_mass_Br

-- Define the weight calculation
def weight_of_6_moles_AlBr3 : ℝ := 6 * molar_mass_AlBr3

-- Theorem stating the weight of 6 moles of AlBr3 is 1600.08 grams
theorem weight_AlBr3_correct : weight_of_6_moles_AlBr3 = 1600.08 := 
by
2   sorry

end weight_AlBr3_correct_l608_608031


namespace johns_profit_l608_608346

theorem johns_profit (bought_pencils : ℕ) (bought_cost : ℕ)
                     (sold_pencils : ℕ) (sold_price : ℕ) (desired_profit : ℝ) :
  bought_pencils = 5 → bought_cost = 6 → 
  sold_pencils = 4 → sold_price = 7 → 
  desired_profit = 80 →
  ∃ n : ℕ, n = 146 ∧ (sold_price / sold_pencils : ℝ) * n - (bought_cost / bought_pencils : ℝ) * n = desired_profit :=
by
  intros h1 h2 h3 h4 h5
  use 146
  split
  . rfl
  . sorry

end johns_profit_l608_608346


namespace no_arith_seq_coefficients_F_inequality_l608_608899

-- Part 1
theorem no_arith_seq_coefficients (n : ℕ) (h : 0 < n) :
  let a₁_x := 1
  let a₂_x := n / 3
  let a₃_x := n * (n - 1) / 18
  (a₂_x - a₁_x) ≠ (a₃_x - a₂_x) := 
by {
  let a₁_x := 1,
  let a₂_x := n / 3,
  let a₃_x := n * (n - 1) / 18,
  sorry
}

-- Part 2
theorem F_inequality (n : ℕ) (x₁ x₂ : ℝ) (h₁ : 0 ≤ x₁) (h₂ : x₁ ≤ 3) (h₃ : 0 ≤ x₂) (h₄ : x₂ ≤ 3) :
  let F_x (x : ℝ) := ∑ k in Finset.range (n + 1), (k + 1) * (Nat.choose n k) * ((1 / 3) * x) ^ k
  |F_x x₁ - F_x x₂| < 2 ^ (n - 1) * (n + 2) :=
by {
  let F_x := fun x : ℝ => ∑ k in Finset.range (n + 1), (k + 1) * (Nat.choose n k) * ((1 / 3) * x) ^ k,
  sorry
}

end no_arith_seq_coefficients_F_inequality_l608_608899


namespace trials_satisfy_inequality_l608_608726

noncomputable def number_of_trials (p : ℝ) (epsilon : ℝ) (confidence : ℝ) : ℕ :=
  ⌈1 / (confidence * epsilon^2 / (p * (1 - p)))⌉₊

theorem trials_satisfy_inequality (p : ℝ) (epsilon : ℝ) (confidence : ℝ) (n : ℕ) :
  p = 0.8 ∧ epsilon = 0.1 ∧ confidence = 0.03 → n >= 534 :=
by
  sorry

end trials_satisfy_inequality_l608_608726


namespace max_area_quadrilateral_l608_608589

theorem max_area_quadrilateral (AB BC CD DA : ℝ) (hAB : AB = 2) (hBC : BC = 4) (hCD : CD = 5) (hDA : DA = 3) : 
  let ABCD_is_convex := (convex_quadrilateral ABCD) in
  maximum_area ABCD = 2 * sqrt 30 :=
by
  sorry

end max_area_quadrilateral_l608_608589


namespace binomial_coefficient_term_x4_in_expansion_l608_608249

theorem binomial_coefficient_term_x4_in_expansion :
  ∀ (x : ℝ), (∃ r : ℕ, 10 - 3 * r = 4 ∧ binomial 5 r = 10) :=
by
  assume x,
  use 2,
  split,
  sorry

end binomial_coefficient_term_x4_in_expansion_l608_608249


namespace probability_of_x_plus_y_lt_4_l608_608093

open Classical -- To handle probability and random selection
open Set -- For geometric notions
open Filter -- For measure and integration

noncomputable def probability_condition (x y : ℝ) (h : 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) : Prop := x + y < 4

theorem probability_of_x_plus_y_lt_4 :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) →
                pr (λ (xy : ℝ × ℝ), probability_condition xy.1 xy.2 (and.intro (and.left xy.2) (and.right (and.left xy.2)))) = 7/9 :=
by sorry

end probability_of_x_plus_y_lt_4_l608_608093


namespace vova_gave_pavlik_three_nuts_l608_608019

variable {V P k : ℕ}
variable (h1 : V > P)
variable (h2 : V - P = 2 * P)
variable (h3 : k ≤ 5)
variable (h4 : ∃ m : ℕ, V - k = 3 * m)

theorem vova_gave_pavlik_three_nuts (h1 : V > P) (h2 : V - P = 2 * P) (h3 : k ≤ 5) (h4 : ∃ m : ℕ, V - k = 3 * m) : k = 3 := by
  sorry

end vova_gave_pavlik_three_nuts_l608_608019


namespace no_solution_consecutive_squares_sum_l608_608770

theorem no_solution_consecutive_squares_sum (x : ℤ) :
  let y := 1005 in
  let sum_of_squares := x^2 + (x + 1)^2 + (x + 2)^2 + (x + 3)^2 + (x + 4)^2 + y^2 in
  sum_of_squares ≠ 6 :=
by
  let y := 1005
  let sum_of_squares := x^2 + (x + 1)^2 + (x + 2)^2 + (x + 3)^2 + (x + 4)^2 + y^2
  have calc₁ : y^2 = 1005^2 := rfl
  have calc₂ : 1005^2 = 1010025 := rfl
  have calc₃ : sum_of_squares ≥ (y^2) := by sorry -- by nonnegative squares of integers
  have calc₄ : sum_of_squares ≥ 1010025 := calc₃ ▸ calc₂ ▸ calc₁ ▸ rfl
  show 1010025 ≠ 6, by sorry

end no_solution_consecutive_squares_sum_l608_608770


namespace functional_eq_log_l608_608572

theorem functional_eq_log {f : ℝ → ℝ} (h₁ : f 4 = 2) 
                           (h₂ : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → f (x1 * x2) = f x1 + f x2) : 
                           (∀ x : ℝ, 0 < x → f x = Real.log x / Real.log 2) := 
by
  sorry

end functional_eq_log_l608_608572


namespace height_of_right_triangle_on_parabola_l608_608326

noncomputable def height_from_hypotenuse
  (A B C : Point)
  (h_triangle : isRightAngledTriangle A B C)
  (h_parabola : onParabola A ∧ onParabola B ∧ onParabola C)
  (h_parallel : hypotenuse_parallel_to_y_axis A B) : ℝ :=
  sorry

-- Define a type for Point
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for isRightAngledTriangle 
def isRightAngledTriangle (A B C : Point) : Prop :=
  sorry

-- Define a type for onParabola
def onParabola (P : Point) : Prop :=
  P.y^2 = 4 * P.x

-- Define a type for hypotenuse_parallel_to_y_axis
def hypotenuse_parallel_to_y_axis (A B : Point) : Prop :=
  -- Assuming A and B form the hypotenuse AB
  A.x = B.x

theorem height_of_right_triangle_on_parabola 
  (A B C : Point)
  (h_triangle : isRightAngledTriangle A B C)
  (h_parabola : onParabola A ∧ onParabola B ∧ onParabola C)
  (h_parallel : hypotenuse_parallel_to_y_axis A B) :
  height_from_hypotenuse A B C h_triangle h_parabola h_parallel = 4 :=
  sorry

end height_of_right_triangle_on_parabola_l608_608326


namespace probability_not_collected_3_cards_in_4_purchases_distribution_X_expectation_X_l608_608066

-- Definitions for the problem conditions
def num_colors : ℕ := 3
def full_purchases (n : ℕ) := fin n → fin num_colors

/-- 
  Part 1:
  Prove the probability that a customer, after making full purchases 4 times, 
  still has not collected 3 cards of the same color is 2/3. 
--/
theorem probability_not_collected_3_cards_in_4_purchases :
  let scenario1 := 3 * combinatorial.choose 4 2 * 2
  let scenario2 := combinatorial.choose 3 2 * combinatorial.choose 4 2
  (scenario1 + scenario2) / (num_colors ^ 4) = 2 / 3 := by sorry

/-- 
  Part 2:
  Let X be the number of times a customer makes full purchases before collecting exactly 3 cards of the same color. 
  Prove the distribution and expectation of X.
--/

-- Defining the probabilities
def P_X (n : ℕ) : ℚ :=
  match n with
  | 3 := 1 / 9
  | 4 := 2 / 9
  | 5 := 8 / 27
  | 6 := 20 / 81
  | 7 := 10 / 81
  | _ := 0

-- Expected value of X
def E_X : ℚ :=
  sum (λ n, n * P_X n) (finset.range 8) -- since X ranges from 3 to 7

-- Proving the distribution
theorem distribution_X :
  (∀ n, n ≥ 3 → P_X n = 1 / 9 ∨ P_X n = 2 / 9 ∨ P_X n = 8 / 27 ∨ P_X n = 20 / 81 ∨ P_X n = 10 / 81) := by sorry

-- Proving the expectation
theorem expectation_X :
  E_X = 409 / 81 := by sorry

end probability_not_collected_3_cards_in_4_purchases_distribution_X_expectation_X_l608_608066


namespace min_value_proof_l608_608969

noncomputable def minimum_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (3/2) * a + b = 1) : ℝ :=
  (3 / a) + (2 / b)

theorem min_value_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (3/2) * a + b = 1) : minimum_value a b h1 h2 h3 = 25 / 2 :=
sorry

end min_value_proof_l608_608969


namespace find_RU_l608_608802

-- Define lengths of sides
variables (PQ QR RP : ℝ)
-- Define the points on the plane
variables (P Q R S T U : Type) [point P] [point Q] [point R] [point S] [point T] [point U]
-- Define the angle bisector and circumcircle intersection points
variables (angle_bisector_PQR : Π P Q R S T : Type, point P → point Q → point R → point S → point T)
variables (circumcircle : Π P Q R : Type, circle P Q R)

-- Conditions based on problem statement
axiom side_lengths : PQ = 13 ∧ QR = 30 ∧ RP = 26
axiom bisector_intersect : angle_bisector_PQR P Q R = (S, T)
axiom circumcircle_intersects : (circumcircle P T S).intersect_line PQ = (P, U)

-- Theorem to prove given above conditions
theorem find_RU : PQ = 13 ∧ QR = 30 ∧ RP = 26 → 
                  angle_bisector_PQR P Q R = (S, T) →
                  (circumcircle P T S).intersect_line PQ = (P, U) →
                  distance R U = 34 :=
by
  sorry

end find_RU_l608_608802


namespace antonov_packs_l608_608514

theorem antonov_packs (total_candies packs_given pieces_per_pack remaining_pieces packs_remaining : ℕ) 
    (h1 : total_candies = 60) 
    (h2 : packs_given = 1) 
    (h3 : pieces_per_pack = 20) 
    (h4 : remaining_pieces = total_candies - (packs_given * pieces_per_pack)) 
    (h5 : packs_remaining = remaining_pieces / pieces_per_pack) : 
    packs_remaining = 2 := 
by
  rw [h1, h3] at h4
  rw [Nat.mul_comm, Nat.sub_eq_iff_eq_add, Nat.sub_sub] at h4
  rw [Nat.mul_comm, Nat.div_eq_iff_eq_mul, Nat.mul_comm] at h5
  exact h5
sorry

end antonov_packs_l608_608514


namespace probability_x_plus_y_less_than_4_l608_608106

open Set Real

/-- Define the square with vertices (0,0), (0,3), (3,3), and (3,0) -/
def square := set.Icc (0 : ℝ) 3 ×ˢ set.Icc (0 : ℝ) 3

/-- Probability that a randomly chosen point (x, y) in the square satisfies x + y < 4 -/
theorem probability_x_plus_y_less_than_4 : 
  let area_square := 9 in
  let area_excluded_triangle := 2 in
  let relevant_area := area_square - area_excluded_triangle in
  (relevant_area / area_square : ℝ) = 7 / 9 :=
by
{ sorry }

end probability_x_plus_y_less_than_4_l608_608106


namespace average_income_of_other_40_customers_l608_608138

theorem average_income_of_other_40_customers
    (avg_income_50 : ℝ)
    (num_50 : ℕ)
    (avg_income_10 : ℝ)
    (num_10 : ℕ)
    (total_num : ℕ)
    (remaining_num : ℕ)
    (total_income_50 : ℝ)
    (total_income_10 : ℝ)
    (total_income_40 : ℝ)
    (avg_income_40 : ℝ) 
    (hyp_avg_income_50 : avg_income_50 = 45000)
    (hyp_num_50 : num_50 = 50)
    (hyp_avg_income_10 : avg_income_10 = 55000)
    (hyp_num_10 : num_10 = 10)
    (hyp_total_num : total_num = 50)
    (hyp_remaining_num : remaining_num = 40)
    (hyp_total_income_50 : total_income_50 = 2250000)
    (hyp_total_income_10 : total_income_10 = 550000)
    (hyp_total_income_40 : total_income_40 = 1700000)
    (hyp_avg_income_40 : avg_income_40 = total_income_40 / remaining_num) :
  avg_income_40 = 42500 :=
  by
    sorry

end average_income_of_other_40_customers_l608_608138


namespace sum_of_squares_of_ages_l608_608624

theorem sum_of_squares_of_ages {a b c : ℕ} (h1 : 5 * a + b = 3 * c) (h2 : 3 * c^2 = 2 * a^2 + b^2) 
  (relatively_prime : Nat.gcd (Nat.gcd a b) c = 1) : 
  a^2 + b^2 + c^2 = 374 :=
by
  sorry

end sum_of_squares_of_ages_l608_608624


namespace triangle_AEF_area_l608_608516

-- Definitions for given conditions
def area_rect (area : ℝ) (length width : ℝ) : Prop := length * width = area

def area_triangle (base height : ℝ) : ℝ := 1 / 2 * base * height

-- Problem statement
theorem triangle_AEF_area (length width : ℝ) (BE DF : ℝ) :
  area_rect 56 length width →
  BE = 3 →
  DF = 2 →
  area_triangle BE DF = 3 →
  area_triangle length width / 2 - 3 = 25 :=
by
  intros h1 h2 h3 h4
  sorry

end triangle_AEF_area_l608_608516


namespace smallest_M_for_inequality_l608_608227

open Real

theorem smallest_M_for_inequality :
  ∀ a b c : ℝ,
    let M := (9 * sqrt 2) / 32 in
    abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)) ≤
    M * (a^2 + b^2 + c^2)^2 :=
by
  intros,
  let M := (9 * sqrt 2) / 32,
  sorry

end smallest_M_for_inequality_l608_608227


namespace point_in_second_quadrant_l608_608990

theorem point_in_second_quadrant 
  {A B C : ℝ} (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) (hC : 0 < C ∧ C < π / 2) 
  (sum_angles : A + B + C = π) : 
  let P := (Real.cos B - Real.sin A, Real.sin B - Real.cos A) in 
  P.1 < 0 ∧ P.2 > 0 := 
by 
  sorry

end point_in_second_quadrant_l608_608990


namespace solve_exponential_eq_l608_608392

theorem solve_exponential_eq (x : ℝ) : 4^(9^x) = 9^(4^x) → x = 0 :=
by
  intro h
  -- Proof goes here
  sorry

end solve_exponential_eq_l608_608392


namespace remainder_when_divided_by_8_l608_608814

theorem remainder_when_divided_by_8 (x k : ℤ) (h : x = 63 * k + 27) : x % 8 = 3 :=
sorry

end remainder_when_divided_by_8_l608_608814


namespace range_of_m_l608_608410

def f (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) m, 1 ≤ f x ∧ f x ≤ 10) ↔ 2 ≤ m ∧ m ≤ 5 := 
by
  sorry

end range_of_m_l608_608410


namespace geom_seq_common_ratio_range_l608_608409

def f (x : ℝ) : ℤ :=
  if ∃ (m : ℤ), | x - m | < 1 / 2 then
    classical.some (classical.some_spec (exists.intro m (classical.some_spec (abs_lt.mp (classical.some_spec ⟨m, sorry⟩)))))
  else 0 -- This is a placeholder as Lean requires each branch of if-then-else to have the same type.

theorem geom_seq_common_ratio_range {a : ℕ → ℝ} {q : ℝ}
  (h₀ : a 1 = 1)
  (h₁ : ∀ n, a (n + 1) = q * a n)
  (h₂ : f (a 2) + f (a 3) = 2) :
  (sqrt 2) / 2 < q ∧ q < (sqrt 3) / 2 :=
sorry

end geom_seq_common_ratio_range_l608_608409


namespace probability_no_3_same_color_after_4_purchases_expectation_of_X_l608_608064

-- Probability that after 4 purchases, no 3 cards of the same color are collected
theorem probability_no_3_same_color_after_4_purchases :
  let num_colors := 3 in
  let probability := (num_colors.choose 1 * 4.choose 2 * 2.choose 1 + 
                      num_colors.choose 2 * (4.choose 2) * (4.choose 2) / num_colors^4) in
  probability = 2 / 3 :=
by sorry

-- Expectation E[X], number of times purchases made before collecting 3 cards of the same color
theorem expectation_of_X :
  let X_dist (n : Nat) : Rational :=
    match n with
    | 3 => 1 / 9
    | 4 => 2 / 9
    | 5 => 8 / 27
    | 6 => 20 / 81
    | 7 => 10 / 81
    | _ => 0 in
  let E_X := 3 * (1 / 9) + 4 * (2 / 9) + 5 * (8 / 27) + 6 * (20 / 81) + 7 * (10 / 81) in
  E_X = 409 / 81 :=
by sorry

end probability_no_3_same_color_after_4_purchases_expectation_of_X_l608_608064


namespace fourth_term_expansion_eq_l608_608256

noncomputable def a : ℝ :=
  ∫ x in 1..2, (3 * x^2 - 2 * x)

theorem fourth_term_expansion_eq : 
  let term := (a * x^2 - 1/x)^6 in 
  (∃ x : ℝ, term.coeff x^3 = -1280) :=
by
  let T_r_plus_1 (r : ℕ) : ℝ := nat.choose 6 r * (a * x^2)^(6 - r) * (-1 / x)^r
  have T_4 : T_r_plus_1 3 = -1280 * x^3 := sorry
  use T_r_plus_1 3
  exact T_4

end fourth_term_expansion_eq_l608_608256


namespace total_amount_l608_608062

theorem total_amount (A B C T : ℝ)
  (h1 : A = 1 / 4 * (B + C))
  (h2 : B = 3 / 5 * (A + C))
  (h3 : A = 20) :
  T = A + B + C → T = 100 := by
  sorry

end total_amount_l608_608062


namespace probability_three_heads_one_tail_is_one_fourth_l608_608993

-- Defining the probability function based on conditions
def prob_head : ℝ := 1 / 2
def prob_tail : ℝ := 1 / 2

-- Count possible cases with exactly three heads and one tail in four coin tosses
def count_three_heads_one_tail : ℕ := 4

-- Calculate the probability of each sequence of 4 coin tosses
def individual_probability : ℝ := (1 / 2) ^ 4

-- The total probability of getting exactly three heads and one tail
def total_probability : ℝ := count_three_heads_one_tail * individual_probability

-- Theorem to prove
theorem probability_three_heads_one_tail_is_one_fourth :
  total_probability = 1 / 4 :=
by
  -- Skipping the actual proof
  sorry

end probability_three_heads_one_tail_is_one_fourth_l608_608993


namespace area_of_rectangle_is_8sqrt3_l608_608462

open EuclideanGeometry

-- Define the rectangle and points as per conditions
variables {A B C D P Q R : EuclideanPlane.Point}
variables {a b c : ℝ}

-- Rectangle property definition
def is_rectangle (A B C D : EuclideanPlane.Point) : Prop :=
  Euclidean.collinear A B D ∧ Euclidean.collinear B C D ∧ 
  Euclidean.collinear A D C
  
-- Definitions according to the conditions
axiom h1 : is_rectangle A B C D
axiom h2 : Euclidean.midpoint A B P
axiom h3 : Euclidean.midpoint B C Q
axiom h4 : Euclidean.intersect AQ CP R
axiom h5 : Euclidean.distance A C = 6
axiom h6 : Euclidean.angle A R C = 150

-- Lean Theorem statement
theorem area_of_rectangle_is_8sqrt3 (A B C D P Q R : EuclideanPlane.Point) 
  (h1 : is_rectangle A B C D)
  (h2: Euclidean.midpoint A B P)
  (h3: Euclidean.midpoint B C Q)
  (h4: Euclidean.intersect AQ CP R)
  (h5: Euclidean.distance A C = 6)
  (h6: Euclidean.angle A R C = 150)
  : Euclidean.area A B C D = 8 * sqrt 3 := 
sorry

end area_of_rectangle_is_8sqrt3_l608_608462


namespace triangle_AC_square_eq_AD_times_AB_l608_608657

open Triangle

variable {A B C D : Point}
variables (α β γ : ℝ)

-- Conditions: 
def is_triangle (ABC : Triangle) : Prop := (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ A)

def bisects_angle (AD : Line) (ABC : Triangle) : Prop :=
  AD ∈ interior_angle_bisector A ∧ D ∈ AB

def circumcenter_is_incenter (ABC : Triangle) (BCD : Triangle) (O : Point) : Prop :=
  circumcenter ABC = incenter BCD

-- Prove that AC^2 = AD · AB
theorem triangle_AC_square_eq_AD_times_AB 
  (ABC : Triangle)
  (condition1 : is_triangle ABC)
  (AD : Line)
  (condition2 : bisects_angle AD ABC)
  (condition3 : ∃ (O : Point), circumcenter_is_incenter ABC (Triangle B C D) O) :
  AC^2 = AD · AB :=
sorry

end triangle_AC_square_eq_AD_times_AB_l608_608657


namespace solve_system_l608_608724

-- Define the given system of differential equations
def system (d t : ℝ) (d x : ℝ) (d y : ℝ) (y : ℝ) (x : ℝ) (t : ℝ) : Prop :=
  d t / (4 * y - 5 * x) = d x / (5 * t - 3 * y) ∧
  d x / (5 * t - 3 * y) = d y / (3 * x - 4 * t)

-- Define the first solution integral
def first_integral (t x y C1 : ℝ) : Prop :=
  3 * t + 4 * x + 5 * y = C1

-- Define the second solution integral
def second_integral (t x y C2 : ℝ) : Prop :=
  t^2 + x^2 + y^2 = C2

-- Prove that given the system of differential equations,
-- the solutions integrals hold.
theorem solve_system (d t d x d y y x t C1 C2 : ℝ) :
  system d t d x d y y x t → first_integral t x y C1 ∧ second_integral t x y C2 :=
by
  intro h
  -- Sorry to skip the proof steps
  sorry

end solve_system_l608_608724


namespace city_council_vote_l608_608635

theorem city_council_vote :
  ∀ (x y x' y' m : ℕ),
    x + y = 350 →
    y > x →
    y - x = m →
    x' - y' = 2 * m →
    x' + y' = 350 →
    x' = (10 * y) / 9 →
    x' - x = 66 :=
by
  intros x y x' y' m h1 h2 h3 h4 h5 h6
  -- proof goes here
  sorry

end city_council_vote_l608_608635


namespace find_x0_for_derivative_l608_608620

theorem find_x0_for_derivative :
  ∀ (f : ℝ → ℝ) (f' : ℝ → ℝ) (x0 : ℝ), f = (λ x, x^3) → f' x0 = 3 → x0 = 1 ∨ x0 = -1 :=
by
  sorry

end find_x0_for_derivative_l608_608620


namespace total_salmon_l608_608540

def male_salmon : Nat := 712261
def female_salmon : Nat := 259378

theorem total_salmon :
  male_salmon + female_salmon = 971639 := by
  sorry

end total_salmon_l608_608540


namespace value_of_d_l608_608228

theorem value_of_d (d y : ℤ) (h₁ : y = 2) (h₂ : 5 * y^2 - 8 * y + 55 = d) : d = 59 := by
  sorry

end value_of_d_l608_608228


namespace count_permutations_conditions_l608_608351

open Nat

theorem count_permutations_conditions : 
  let L := (List.range 14).map (· + 1)
  in let cond1 := ∀ i j, 1 ≤ i → i < j → j ≤ 7 → L.nth i > L.nth j
  in let cond2 := ∀ i j, 7 < i → i < j → j ≤ 14 → L.nth i < L.nth j
  in (L.toFinset.card.choose 6) = 1716 := 
by {
  sorry
}

end count_permutations_conditions_l608_608351


namespace probability_of_x_plus_y_lt_4_l608_608097

open Classical -- To handle probability and random selection
open Set -- For geometric notions
open Filter -- For measure and integration

noncomputable def probability_condition (x y : ℝ) (h : 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) : Prop := x + y < 4

theorem probability_of_x_plus_y_lt_4 :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) →
                pr (λ (xy : ℝ × ℝ), probability_condition xy.1 xy.2 (and.intro (and.left xy.2) (and.right (and.left xy.2)))) = 7/9 :=
by sorry

end probability_of_x_plus_y_lt_4_l608_608097


namespace exams_left_to_grade_wednesday_l608_608703

-- Define the total number of exams
def total_exams : ℕ := 120

-- Define the percentage of exams graded on Monday
def percent_monday : ℝ := 0.60

-- Define the percentage of exams graded on Tuesday
def percent_tuesday : ℝ := 0.75

-- Calculate exams graded on Monday
def exams_graded_monday : ℕ := (percent_monday * total_exams).toInt

-- Calculate remaining exams after Monday
def remaining_exams_after_monday : ℕ := total_exams - exams_graded_monday

-- Calculate exams graded on Tuesday
def exams_graded_tuesday : ℕ := (percent_tuesday * remaining_exams_after_monday).toInt

-- Calculate remaining exams after Tuesday
def remaining_exams_after_tuesday : ℕ := remaining_exams_after_monday - exams_graded_tuesday

-- Theorem for the number of exams left to grade on Wednesday
theorem exams_left_to_grade_wednesday : remaining_exams_after_tuesday = 12 := by
  sorry

end exams_left_to_grade_wednesday_l608_608703


namespace circumcenter_lies_on_ak_l608_608837

noncomputable def triangle_circumcenter_lies_on_ak
  {α β γ : ℝ}
  (A B C L H K O : Type*)
  [triangle A B C]
  [angle_bisector A L]
  [height_from B H]
  [circumcircle_of_triangle A (B ∧ L) K]
  [circumcenter_of_triangle A B C O]
  : Prop :=
  lies_on_line O (line_through A K)

-- We'll add the assumptions as hypotheses to the lemma
theorem circumcenter_lies_on_ak 
  {α β γ : ℝ} {A B C L H K O : Type*}
  [triangle A B C]
  [angle_bisector A L]
  [height_from B H]
  [circumcircle_of_triangle A (B ∧ L) K]
  [circumcenter_of_triangle A B C O]
  : lies_on_line O (line_through A K) :=
sorry

end circumcenter_lies_on_ak_l608_608837


namespace true_proposition_l608_608274

-- Define the assumptions
def p : Prop := ∀ x : ℝ, 2^x + 1/2^x > 2
def q : Prop := ∃ x : ℝ, 0 ≤ x ∧ x ≤ (Real.pi / 2) ∧ (Real.sin x + Real.cos x = 1 / 2)

-- State the theorem to be proved
theorem true_proposition : ¬ p ∧ ¬ q := 
by
  sorry

end true_proposition_l608_608274


namespace ellipse_foci_distance_l608_608548

noncomputable def distance_between_foci (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance :
  ∃ (a b : ℝ), (a = 6) ∧ (b = 3) ∧ distance_between_foci a b = 6 * Real.sqrt 3 :=
by
  sorry

end ellipse_foci_distance_l608_608548


namespace average_speed_72_l608_608058

theorem average_speed_72 (s : ℝ) (h1 : s > 0) :
  let dist1 := s / 2,
      dist2 := s / 6,
      dist3 := s / 3,
      speed1 := 60,
      speed2 := 120,
      speed3 := 80,
      t1 := dist1 / speed1,
      t2 := dist2 / speed2,
      t3 := dist3 / speed3,
      t_total := t1 + t2 + t3,
      avg_speed := s / t_total
  in avg_speed = 72 :=
by
  intros,
  let dist1 := s / 2,
  let dist2 := s / 6,
  let dist3 := s / 3,
  let speed1 := 60,
  let speed2 := 120,
  let speed3 := 80,
  let t1 := dist1 / speed1,
  let t2 := dist2 / speed2,
  let t3 := dist3 / speed3,
  let t_total := t1 + t2 + t3,
  let avg_speed := s / t_total,
  have ht1 : t1 = s / 120 := by sorry,
  have ht2 : t2 = s / 720 := by sorry,
  have ht3 : t3 = s / 240 := by sorry,
  have ht_total : t_total = s / 72 := by sorry,
  have h_avg_speed : avg_speed = 72 := by sorry,
  exact h_avg_speed

end average_speed_72_l608_608058


namespace library_books_difference_l608_608433

theorem library_books_difference :
  let books_old_town := 750
  let books_riverview := 1240
  let books_downtown := 1800
  let books_eastside := 1620
  books_downtown - books_old_town = 1050 :=
by
  sorry

end library_books_difference_l608_608433


namespace intersection_P_Q_l608_608975

-- Define set P
def P : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

-- Define set Q (using real numbers, but we will be interested in natural number intersections)
def Q : Set ℝ := {x | x^2 + x - 6 ≤ 0}

-- The intersection of P with Q in the natural numbers should be {1, 2}
theorem intersection_P_Q :
  {x : ℕ | x ∈ P ∧ (x : ℝ) ∈ Q} = {1, 2} :=
by
  sorry

end intersection_P_Q_l608_608975


namespace proof_problem_l608_608876

noncomputable def orthocenter (A B C : Point) : Point := sorry
def angleA (A B C : Point) : Real := sorry
def angleB (A B C : Point) : Real := sorry
def angleC (A B C : Point) : Real := sorry
def distSq (A B : Point) : Real := sorry
def tan (x : Real) : Real := Real.tan x

theorem proof_problem (A B C P: Point) (H : Point) (hH : H = orthocenter A B C) :
  (distSq A P - distSq A H) * tan (angleA A B C) +
  (distSq B P - distSq B H) * tan (angleB A B C) +
  (distSq C P - distSq C H) * tan (angleC A B C) =
  distSq P H * tan (angleA A B C) * tan (angleB A B C) * tan (angleC A B C) :=
sorry

end proof_problem_l608_608876


namespace sum_of_digits_of_N_l608_608500

theorem sum_of_digits_of_N :
  (∃ N, N * (N + 1) / 2 = 3003) → ∑ d in (77.digits 10), d = 14 :=
by
  sorry

end sum_of_digits_of_N_l608_608500


namespace triangle_height_from_area_l608_608399

theorem triangle_height_from_area {A b h : ℝ} (hA : A = 36) (hb : b = 8) 
    (formula : A = 1 / 2 * b * h) : h = 9 := 
by
  sorry

end triangle_height_from_area_l608_608399


namespace rational_xy_exists_l608_608893

theorem rational_xy_exists (λ : ℝ) :
  ∃ (x y n : ℚ), x = (n + 1 / n)^n ∧ y = (n + 1 / n)^(n + 1) ∧ λ^y = y^x :=
by
  sorry

end rational_xy_exists_l608_608893


namespace distance_A_B_l608_608944

-- Define points A and B
def A : ℝ × ℝ × ℝ := (-1, 1, 1)
def B : ℝ × ℝ × ℝ := (0, 1, 1)

-- Define the Euclidean distance formula in 3D
def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

-- Formulate the theorem statement
theorem distance_A_B : distance A B = 1 := by
  sorry

end distance_A_B_l608_608944


namespace find_line_equation_l608_608549

theorem find_line_equation (P : ℝ × ℝ) (hx : ∀ (x y : ℝ), P = (1, 2)) 
  (h_intercepts : ∀ (m : ℝ), (m, 0) ∨ (0, m) ∈ ℝ):
  (∀ (a b : ℝ), a = 1 ∧ b = 2 → x + y - 3 = 0 ∨ 2x - y = 0) :=
sorry

end find_line_equation_l608_608549


namespace problem_l608_608967

noncomputable def f : ℝ → ℝ :=
| x if x < 3 => 3^(x - 2) - 5
| x => -log2(x + 1)

theorem problem (m : ℝ) (h : f m = -6) : f (m - 61) = -4 := by
  sorry

end problem_l608_608967


namespace minimum_noninteger_weights_l608_608424

theorem minimum_noninteger_weights (n : ℕ) (ws : fin n → ℝ) 
    (h_non_int : ∀ i, ws i ∉ set.Ioo (-∞ : ℝ) (∞) ∩ set.Icc ⌊ws i⌋ ⌈ws i⌉ )
    (h_balance : ∀ (m : ℕ), 1 ≤ m → m ≤ 40 → ∃ s : finset (fin n), ∑ i in s, ws i = (m : ℝ)) 
    : 7 ≤ n :=
sorry

end minimum_noninteger_weights_l608_608424


namespace max_plus_min_eq_zero_l608_608466

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

def maxValue : ℝ := sup (f '' (set.univ : set ℝ))
def minValue : ℝ := inf (f '' (set.univ : set ℝ))

theorem max_plus_min_eq_zero : maxValue + minValue = 0 := 
sorry

end max_plus_min_eq_zero_l608_608466


namespace antonov_packs_l608_608515

theorem antonov_packs (total_candies packs_given pieces_per_pack remaining_pieces packs_remaining : ℕ) 
    (h1 : total_candies = 60) 
    (h2 : packs_given = 1) 
    (h3 : pieces_per_pack = 20) 
    (h4 : remaining_pieces = total_candies - (packs_given * pieces_per_pack)) 
    (h5 : packs_remaining = remaining_pieces / pieces_per_pack) : 
    packs_remaining = 2 := 
by
  rw [h1, h3] at h4
  rw [Nat.mul_comm, Nat.sub_eq_iff_eq_add, Nat.sub_sub] at h4
  rw [Nat.mul_comm, Nat.div_eq_iff_eq_mul, Nat.mul_comm] at h5
  exact h5
sorry

end antonov_packs_l608_608515


namespace minimum_value_of_f_l608_608931

noncomputable def f (x : ℝ) : ℝ :=
  |2 * x - 1| + |3 * x - 2| + |4 * x - 3| + |5 * x - 4|

theorem minimum_value_of_f : ∃ x : ℝ, ∀ y : ℝ, f(x) ≤ f(y) ∧ f(x) = 1 := 
by 
  sorry

end minimum_value_of_f_l608_608931


namespace red_lights_l608_608786

theorem red_lights (total_lights yellow_lights blue_lights red_lights : ℕ)
  (h1 : total_lights = 95)
  (h2 : yellow_lights = 37)
  (h3 : blue_lights = 32)
  (h4 : red_lights = total_lights - (yellow_lights + blue_lights)) :
  red_lights = 26 := by
  sorry

end red_lights_l608_608786


namespace equivalent_fraction_l608_608880

theorem equivalent_fraction :
  (6 + 6 + 6 + 6) / ((-2) * (-2) * (-2) * (-2)) = (4 * 6) / ((-2)^4) :=
by 
  sorry

end equivalent_fraction_l608_608880


namespace sin_value_l608_608565

theorem sin_value(
  (α : ℝ)
  (h: cos (α - π / 6) = 1 / 3)
) : sin (2 * α + π / 6) = -7 / 9 :=
sorry

end sin_value_l608_608565


namespace combinations_of_coins_l608_608614

noncomputable def count_combinations (target : ℕ) : ℕ :=
  (30 - 0*0) -- As it just returns 45 combinations

theorem combinations_of_coins : count_combinations 30 = 45 :=
  sorry

end combinations_of_coins_l608_608614


namespace ratio_is_one_half_l608_608783

namespace CupRice

-- Define the grains of rice in one cup
def grains_in_one_cup : ℕ := 480

-- Define the grains of rice in the portion of the cup
def grains_in_portion : ℕ := 8 * 3 * 10

-- Define the ratio of the portion of the cup to the whole cup
def portion_to_cup_ratio := grains_in_portion / grains_in_one_cup

-- Prove that the ratio of the portion of the cup to the whole cup is 1:2
theorem ratio_is_one_half : portion_to_cup_ratio = 1 / 2 := by
  -- Proof goes here, but we skip it as required
  sorry
end CupRice

end ratio_is_one_half_l608_608783


namespace coeff_x2_expansion_l608_608196

theorem coeff_x2_expansion :
  let p1 := 3 * X^3 + 5 * X^2 - 4 * X + 1
  let p2 := 2 * X^2 - 9 * X + 3
  coefficient (expansion p1 p2) X^2 = 51 :=
by
  let p1 : Polynomial Int := 3 * X^3 + 5 * X^2 - 4 * X + 1
  let p2 : Polynomial Int := 2 * X^2 - 9 * X + 3
  sorry

end coeff_x2_expansion_l608_608196


namespace median_circumradius_altitude_inequality_l608_608685

variable (h R m_a m_b m_c : ℝ)

-- Define the condition for the lengths of the medians and other related parameters
-- m_a, m_b, m_c are medians, R is the circumradius, h is the greatest altitude

theorem median_circumradius_altitude_inequality :
  m_a + m_b + m_c ≤ 3 * R + h :=
sorry

end median_circumradius_altitude_inequality_l608_608685


namespace find_number_l608_608220

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 58) : x = 145 := by
  sorry

end find_number_l608_608220


namespace solve_for_x_l608_608038

theorem solve_for_x (x : ℕ) (h : x + 1 = 2) : x = 1 :=
sorry

end solve_for_x_l608_608038


namespace intersecting_triangle_area_of_tetrahedron_l608_608131

-- Given definitions
def regular_tetrahedron (s : ℝ) := s > 0
def side_length := 2
def intersecting_triangle_area {s : ℝ} (h₁: s > 0) (h₂: s = 2) := 
  ∃ A : ℝ, A = (s * (s * (sqrt 3) / 4)) * (1 / (4 * sqrt 3))
  ∧ A = sqrt 3 / 4

-- The theorem to prove
theorem intersecting_triangle_area_of_tetrahedron : 
  intersecting_triangle_area (regular_tetrahedron side_length) (side_length = 2) :=
begin
  sorry
end

end intersecting_triangle_area_of_tetrahedron_l608_608131


namespace probability_of_x_plus_y_lt_4_l608_608094

open Classical -- To handle probability and random selection
open Set -- For geometric notions
open Filter -- For measure and integration

noncomputable def probability_condition (x y : ℝ) (h : 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) : Prop := x + y < 4

theorem probability_of_x_plus_y_lt_4 :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) →
                pr (λ (xy : ℝ × ℝ), probability_condition xy.1 xy.2 (and.intro (and.left xy.2) (and.right (and.left xy.2)))) = 7/9 :=
by sorry

end probability_of_x_plus_y_lt_4_l608_608094


namespace jonathan_daily_calories_l608_608667

theorem jonathan_daily_calories (C : ℕ) (daily_burn weekly_deficit extra_calories total_burn : ℕ) 
  (h1 : daily_burn = 3000) 
  (h2 : weekly_deficit = 2500) 
  (h3 : extra_calories = 1000) 
  (h4 : total_burn = 7 * daily_burn) 
  (h5 : total_burn - weekly_deficit = 7 * C + extra_calories) :
  C = 2500 :=
by 
  sorry

end jonathan_daily_calories_l608_608667


namespace distance_CD_l608_608530

noncomputable def ellipse_eq (x y : ℝ) : Prop :=
  16 * (x + 2)^2 + 4 * y^2 = 64

def major_axis_distance : ℝ := 4
def minor_axis_distance : ℝ := 2

theorem distance_CD : ∃ (d : ℝ), 16 * (x + 2)^2 + 4 * y^2 = 64 → d = 2 * Real.sqrt 5 :=
by
  sorry

end distance_CD_l608_608530


namespace find_tangent_angle_l608_608725

-- Define the problem conditions
variables (O1 O2 M : EuclideanSpace ℝ (fin 3))
variable (r1 : ℝ)
variable (r2 : ℝ)
variable (d : ℝ)
variable (θ : ℝ)

-- Specify the conditions
-- O1 and O2 are the centers of the spheres with radii 3 and 1 respectively
def radius_cond : Prop := r1 = 3 ∧ r2 = 1

-- The spheres touch each other
def touch_cond : Prop := dist O1 O2 = r1 + r2

-- M is 3 units away from O2
def M_cond : Prop := dist M O2 = 3

-- One of the tangent lines makes a 45 degree angle with line O1O2
def tangent_cond : Prop := θ = 45

-- The angle between the two tangent lines
def tangent_angle (M O1 O2 : EuclideanSpace ℝ (fin 3)) (r1 r2 d θ : ℝ) : ℝ := 
  2 * real.arctan (1 / 3)

-- The theorem we want to prove
theorem find_tangent_angle : 
  radius_cond r1 r2 ∧ touch_cond O1 O2 r1 r2 ∧ M_cond M O2 d ∧ tangent_cond θ →
  tangent_angle M O1 O2 r1 r2 d θ = 2 * real.arctan (1 / 3) :=
sorry

end find_tangent_angle_l608_608725


namespace ap_sum_conditions_l608_608713

theorem ap_sum_conditions (P : ℝ × ℝ) (O₁ O₂ : ℝ × ℝ) (a b : ℝ) : 
  -- Conditions
  (P ≠ (6, 6)) → -- since AP > CP
  O₁ = (6, 0) → 
  O₂ = (6, 12) →
  (dist O₁ P ≠ 0) → (dist O₂ P ≠ 0) → -- O₁ and O₂ are distinct points from P
  ∠O₁ P O₂ = 120 → 
  a = 72 → b = 24 → 
  -- Question/Proof
  ( √a + √b = AP → a + b = 96) :=
by
  intros
  sorry

end ap_sum_conditions_l608_608713


namespace plane_ticket_price_l608_608863

-- Define the given conditions
def luggage_weight := 30
def free_weight := 20
def overweight_charge_percentage := 0.015
def luggage_ticket_cost := 180

-- State the main theorem to be proven
theorem plane_ticket_price (x : ℝ) : 
  (luggage_weight - free_weight) * overweight_charge_percentage * x = luggage_ticket_cost →
  x = 1200 :=
by
  sorry

end plane_ticket_price_l608_608863


namespace car_rental_max_revenue_l608_608852

theorem car_rental_max_revenue
  -- Conditions
  (total_cars : ℕ)
  (initial_fee : ℝ)
  (increment : ℝ)
  (maintenance_rented : ℝ)
  (maintenance_unrented : ℝ)
  (rented_cars : ℕ → ℕ := λ n, total_cars - n)
  (rental_fee_per_car : ℕ → ℝ := λ n, initial_fee + increment * n)
  (revenue : ℕ → ℝ := λ n, rented_cars n * rental_fee_per_car n)
  (maintenance_cost : ℕ → ℝ := λ n, maintenance_rented * rented_cars n + maintenance_unrented * n)
  (profit : ℕ → ℝ := λ n, revenue n - maintenance_cost n)
  (optimal_n : ℕ := 21)
  
  -- Assertions to prove
  (h1 : total_cars = 100)
  (h2 : initial_fee = 3000)
  (h3 : increment = 50)
  (h4 : maintenance_rented = 150)
  (h5 : maintenance_unrented = 50)
  
  : rental_fee_per_car optimal_n = 4050 ∧ profit optimal_n = 307050 := by
  sorry

end car_rental_max_revenue_l608_608852


namespace find_polynomial_l608_608545

-- Define the polynomial P satisfying the given condition
def P (x : ℝ) : ℝ 

-- Axiom stating the polynomial condition for all real x
axiom polynomial_condition : ∀ x : ℝ, (P (x^2)) * (P (x^3)) = (P x) ^ 5

-- Theorem stating the solution
theorem find_polynomial (P : ℝ → ℝ) (polynomial_condition: ∀ x : ℝ, (P (x^2)) * (P (x^3)) = (P x) ^ 5) :
  ∃ n : ℕ, ∀ x : ℝ, P x = x^n :=
sorry  -- Proof is omitted

end find_polynomial_l608_608545


namespace cubic_sum_identity_l608_608621

theorem cubic_sum_identity (x y z : ℝ) (h1 : x + y + z = 15) (h2 : xy + yz + zx = 34) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 1845 :=
by
  sorry

end cubic_sum_identity_l608_608621


namespace problem1_problem2_problem3_l608_608354

theorem problem1 (x y x1 x2 : ℝ) :
  (y = (2 * real.sqrt 5 / 5) * x1 ∧ y = -(2 * real.sqrt 5 / 5) * x2)
  ∧ (x1 + x2 = x ∧ x1 - x2 = (real.sqrt 5 / 2) * y)
  ∧ (real.sqrt ((5 / 4) * y^2 + (4 / 5) * x^2) = real.sqrt 20)
  → (x^2 / 25 + y^2 / 16 = 1) := sorry

theorem problem2 (s t x y λ : ℝ) :
  (real.sqrt (s^2 / 25 + t^2 / 16) = 1)
  ∧ (real.sqrt ((λ^2 * s^2) / 25 + ((λ * t - 16*λ + 16)^2) / 16) = 1)
  → (λ ≠ 0 ∧ λ ≠ 1
  ∧ (3 / 5 ≤ λ ∧ λ ≤ 5 / 3)) := sorry

theorem problem3 (b k : ℝ) :
  (real.sqrt (25 * k^2 + 16) > 0)
  → (let y0 := (-9 * b) / (25 * k^2 + 16) in
    (y0^2 < 81 / (25 * k^2 + 16))
    ∧ (-(9 / 4) < y0 ∧ y0 < 9 / 4)) := sorry

end problem1_problem2_problem3_l608_608354


namespace sin_beta_value_l608_608622

-- We need noncomputable theory because trigonometric functions are involved.
noncomputable theory

-- Assuming two acute angles α and β
variables {α β : ℝ}
-- Hypotheses of the problem
def angle_conditions (α β : ℝ) : Prop :=
  (0 < α ∧ α < π/2) ∧ (0 < β ∧ β < π/2) ∧
  cos α = 4/5 ∧ cos (α + β) = 5/13

-- Lean theorem statement to prove the value of sin β
theorem sin_beta_value (h : angle_conditions α β) : sin β = 33 / 65 :=
by
  sorry

end sin_beta_value_l608_608622


namespace present_worth_calculation_l608_608453

-- Define the conditions
def banker's_gain : ℝ := 24
def rate_of_interest : ℝ := 10 / 100
def time_years : ℝ := 2

-- Present worth calculation
def present_worth (bg roi ty : ℝ) : ℝ := bg / (roi * ty)

-- The proof problem
theorem present_worth_calculation
    (bg : ℝ := banker's_gain)
    (roi : ℝ := rate_of_interest)
    (ty : ℝ := time_years):
    present_worth bg roi ty = 120 :=
by
  -- Calculate the present worth
  have h1 : present_worth bg roi ty = bg / (roi * ty) := rfl
  have h2 : roi * ty = 0.2 := by norm_num
  have h3 : present_worth bg roi ty = 24 / 0.2 := by rw [h1, h2]
  have h4 : 24 / 0.2 = 120 := by norm_num
  rw [h3, h4]
  rfl

end present_worth_calculation_l608_608453


namespace complex_range_l608_608594

theorem complex_range (z : ℂ) (hz : abs z = 1) : 
  (abs (z^2 + complex.i * z^2 + 1) ∈ set.Icc (real.sqrt 2 - 1) (real.sqrt 2 + 1)) :=
sorry

end complex_range_l608_608594


namespace AC_gt_4_l608_608246

-- Define the points and segments
variables (A B C M N : Type)
variables [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C] [LinearOrderedField M] [LinearOrderedField N]
variables (AM BN MN AC : A)

-- Given conditions
axiom hM : M = A + B -- M lies on AB
axiom hN : N = B + C -- N lies on BC
axiom h_parallel : MN || AC -- MN parallel to AC
axiom h_BN : BN = 1
axiom h_MN : MN = 2
axiom h_AM : AM = 3

-- Goal
theorem AC_gt_4 :
  AC > 4 := sorry

end AC_gt_4_l608_608246


namespace antonella_toonies_l608_608156

theorem antonella_toonies (L T : ℕ) (h1 : L + T = 10) (h2 : L + 2 * T = 14) : T = 4 :=
by
  sorry

end antonella_toonies_l608_608156


namespace find_a_plus_b_l608_608205

def eight_points_circle_radius (r : ℝ) (a b : ℤ) : Prop :=
  ∃ (points : Fin 8 → ℝ × ℝ) (radius : ℝ) (s : ℝ), 
    s = 2 ∧ 
    radius = 1 ∧ 
    (∀ i, points i ∈ metric.sphere (0, 0) r) ∧ 
    (∀ i, metric.dist (points i) (points ((i + 1) % 8)) = s) ∧ 
    r ^ 2 = (a : ℝ) + (b : ℝ) * real.sqrt 2

theorem find_a_plus_b (r : ℝ) (a b : ℤ) : 
  eight_points_circle_radius r a b → a + b = 6 :=
by
  sorry

end find_a_plus_b_l608_608205


namespace Josanna_min_avg_score_l608_608668

theorem Josanna_min_avg_score (scores : List ℕ) (cur_avg target_avg : ℚ)
  (next_test_bonus : ℚ) (additional_avg_points : ℚ) : ℚ :=
  let cur_avg := (92 + 81 + 75 + 65 + 88) / 5
  let target_avg := cur_avg + 6
  let needed_total := target_avg * 7
  let additional_points := 401 + 5
  let needed_sum := needed_total - additional_points
  needed_sum / 2

noncomputable def min_avg_score : ℚ :=
  Josanna_min_avg_score [92, 81, 75, 65, 88] 80.2 86.2 5 6

example : min_avg_score = 99 :=
by
  sorry

end Josanna_min_avg_score_l608_608668


namespace find_n_l608_608908

theorem find_n (n : ℤ) (h1 : 0 ≤ n ∧ n ≤ 10) (h2 : n ≡ 123456 [MOD 11]) : n = 3 :=
sorry

end find_n_l608_608908


namespace econ_not_feasible_l608_608232

theorem econ_not_feasible (x y p q: ℕ) (h_xy : 26 * x + 29 * y = 687) (h_pq : 27 * p + 31 * q = 687) : p + q ≥ x + y := by
  sorry

end econ_not_feasible_l608_608232


namespace polynomial_cube_identity_l608_608213

theorem polynomial_cube_identity
  (a₀ a₁ a₂ a₃ : ℝ)
  (h : (⟨sqrt 3, sorry⟩ * x - 1) ^ 3 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3) :
  (a₀ + a₂) ^ 2 - (a₁ + a₃) ^ 2 = -8 :=
sorry

end polynomial_cube_identity_l608_608213


namespace center_of_circumcircle_lies_on_AK_l608_608839

variable {α β γ : Real} -- Angles in triangle ABC
variable (A B C L H K O : Point) -- Points in the configuration
variable (circumcircle_ABC : TriangularCircumcircle A B C) -- Circumcircle of triangle ABC

-- Definitions based on the given conditions
variable (is_angle_bisector : angle_bisector A B C L)
variable (is_height : height_from_point_to_line B A L H)
variable (intersects_circle_at_K : intersects_circumcircle A B L K circumcircle_ABC)
variable (is_circumcenter : circumcenter A B C O circumcircle_ABC)

theorem center_of_circumcircle_lies_on_AK
  (h_angle_bisector : is_angle_bisector)
  (h_height : is_height)
  (h_intersects_circle_at_K : intersects_circle_at_K)
  (h_circumcenter : is_circumcenter) 
    : lies_on_line O A K := 
sorry -- Proof is omitted

end center_of_circumcircle_lies_on_AK_l608_608839


namespace prime_pairs_perfect_square_l608_608890

theorem prime_pairs_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  ∃ k : ℕ, p^(q-1) + q^(p-1) = k^2 ↔ (p = 2 ∧ q = 2) :=
by
  sorry

end prime_pairs_perfect_square_l608_608890


namespace observation_corrected_correctly_l608_608750

def initialMean (n : ℕ) (mean : ℝ) (sum : ℝ) : Prop := sum = n * mean

def correctedMean (n : ℕ) (mean : ℝ) (sum : ℝ) : Prop := sum = n * mean

def correctObservation (initialSum incorrectObservation correctedSum : ℝ) (correctObservation : ℝ) : Prop :=
  correctedSum = initialSum - incorrectObservation + correctObservation

theorem observation_corrected_correctly :
  ∀ (n : ℕ) (initialMeanValue correctedMeanValue initialSum incorrectObservation correctedSum correctObservation : ℝ),
    initialMean n initialMeanValue initialSum →
    correctedMean n correctedMeanValue correctedSum →
    correctObservation initialSum incorrectObservation correctedSum correctObservation →
    (initialMeanValue = 36) →
    (correctedMeanValue = 36.5) →
    (n = 50) →
    (incorrectObservation = 23) →
    correctObservation = 48 :=
by
  intros n initialMeanValue correctedMeanValue initialSum incorrectObservation correctedSum correctObservation
  intros initialMean_h correctedMean_h correctObservation_h
  intros initialMeanValue_eq correctedMeanValue_eq n_eq incorrectObservation_eq
  rw [initialMeanValue_eq, correctedMeanValue_eq, n_eq, incorrectObservation_eq, initialMean, correctedMean, correctObservation] at *
  sorry

end observation_corrected_correctly_l608_608750


namespace moses_more_than_esther_l608_608800

noncomputable theory

def total_amount : ℝ := 50
def moses_share_percentage : ℝ := 0.40
def moses_share : ℝ := moses_share_percentage * total_amount
def remainder : ℝ := total_amount - moses_share
def esther_share : ℝ := remainder / 2

theorem moses_more_than_esther : moses_share - esther_share = 5 :=
by
  -- Proof goes here
  sorry

end moses_more_than_esther_l608_608800


namespace all_real_roots_in_interval_l608_608688

noncomputable def P (x : ℝ) : ℝ := sorry -- Polynomial function P(x)

variables (a b : ℝ)
variables (n : ℕ)

-- Conditions
axiom P_a_neg : P a < 0
axiom P_deriv_a_leq : ∀ (k : ℕ) (hk : k ≤ n), (-1)^k * derivative^[k] P a ≤ 0
axiom P_b_pos : P b > 0
axiom P_deriv_b_geq : ∀ (k : ℕ) (hk : k ≤ n), derivative^[k] P b ≥ 0
axiom a_lt_b : a < b

theorem all_real_roots_in_interval : ∀ x : ℝ, P x = 0 → a < x ∧ x < b :=
by
  sorry

end all_real_roots_in_interval_l608_608688


namespace probability_x_plus_y_lt_4_l608_608117

open Set

-- Define the square and the line
def square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 }
def line_lt_4 : Set (ℝ × ℝ) := { p | p.1 + p.2 < 4 }

-- The probability to prove
theorem probability_x_plus_y_lt_4 : 
  (volume (square ∩ line_lt_4) / volume square) = 7 / 9 := 
sorry

end probability_x_plus_y_lt_4_l608_608117


namespace inequality_holds_for_a_l608_608922

theorem inequality_holds_for_a (a : ℝ) : 
  (a > -6 ∧ a < 6) → ∀ x : ℝ, (a * x) / (x^2 + 4) < 1.5 := 
by
  intros h x
  have hx : x^2 + 4 > 0 := by linarith [pow_two_nonneg x, show (0 : ℝ) < 4]
  sorry

end inequality_holds_for_a_l608_608922


namespace pages_lost_l608_608002

theorem pages_lost (stickers_per_page : ℕ) (initial_pages : ℕ) (remaining_stickers : ℕ) (h_stickers_per_page : stickers_per_page = 20) (h_initial_pages : initial_pages = 12) (h_remaining_stickers : remaining_stickers = 220) : 
  initial_pages - (remaining_stickers / stickers_per_page) = 1 :=
by
  rw [h_stickers_per_page, h_initial_pages, h_remaining_stickers]
  norm_num

end pages_lost_l608_608002


namespace equation_B_no_solution_l608_608816

theorem equation_B_no_solution : ¬ ∃ x : ℝ, |-2 * x| + 6 = 0 :=
by
  sorry

end equation_B_no_solution_l608_608816


namespace ratio_of_areas_l608_608788

structure Circle :=
  (diameter: ℝ)

def radius (c: Circle) : ℝ := c.diameter / 2
def area (c: Circle) : ℝ := Real.pi * (radius c) ^ 2

def smaller_circle : Circle := ⟨2⟩
def larger_circle : Circle := ⟨6⟩

def yellow_area : ℝ := area smaller_circle
def large_area : ℝ := area larger_circle
def green_area : ℝ := large_area - yellow_area

theorem ratio_of_areas : green_area / yellow_area = 8 := by
  sorry

end ratio_of_areas_l608_608788


namespace michael_reaches_eric_l608_608897

-- Definitions of the problem's conditions
def distance_between_towns : ℝ := 30
def eric_relative_speed : ℝ := 4
def decrease_rate : ℝ := 2
def initial_biking_time : ℝ := 4
def delay_time : ℝ := 6
def michael_speed : ℝ := 2 / 5
def eric_speed : ℝ := 4 * michael_speed
def remaining_distance_after_initial_time : ℝ := distance_between_towns - decrease_rate * initial_biking_time
def michael_remaining_time : ℝ := remaining_distance_after_initial_time / michael_speed

-- Total time calculation
def total_time : ℝ := initial_biking_time + delay_time + michael_remaining_time

theorem michael_reaches_eric (T : ℝ) : T = total_time -> T = 65 :=
by
  sorry

end michael_reaches_eric_l608_608897


namespace circle_radius_sqrt2_l608_608553

theorem circle_radius_sqrt2 (θ : ℝ) : 
  sqrt ((2 * sin (π / 4) * cos θ)^2 + (2 * sin (π / 4) * sin θ)^2) = sqrt 2 :=
by
  -- Proof skipped
  sorry

end circle_radius_sqrt2_l608_608553


namespace reflect_transformations_l608_608531

theorem reflect_transformations :
  let R := (0, -5)
  let reflect_x := λ (p : ℝ × ℝ), (p.1, -p.2)
  let reflect_y := λ (p : ℝ × ℝ), (-p.1, p.2)
  let reflect_y_eq_x := λ (p : ℝ × ℝ), (p.2, p.1)
  reflect_y_eq_x (reflect_y (reflect_x R)) = (5, 0) :=
by
  sorry

end reflect_transformations_l608_608531


namespace cos2_alpha_plus_2sin2_alpha_l608_608255

theorem cos2_alpha_plus_2sin2_alpha {α : ℝ} (h : Real.tan α = 3 / 4) : 
    Real.cos α ^ 2 + 2 * Real.sin (2 * α) = 64 / 25 := 
by 
  sorry

end cos2_alpha_plus_2sin2_alpha_l608_608255


namespace option_d_correct_l608_608035

variable (a b : ℝ)

theorem option_d_correct : (-a^3)^4 = a^(12) := by sorry

end option_d_correct_l608_608035


namespace range_of_a_l608_608999

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ x^2 + (a - 1) * x + 1 < 0) ↔ (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l608_608999


namespace valid_propositions_l608_608696

variables {α β : Type*} [Plane α] [Plane β]
variables {m n : Line}

-- Definitions of conditions
def prop_1 (m_perp_n : m ⥊ n) (m_perp_α : m ⥊ α) (n_para_β : n ∥ β) : Prop :=
  α ⥊ β

def prop_2 (m_perp_α : m ⥊ α) (n_para_α : n ∥ α) : Prop :=
  m ⥊ n

def prop_3 (a_para_b : α ∥ β) (m_subset_α : m ∈ α) : Prop :=
  m ∥ β

def prop_4 (m_para_n : m ∥ n) (a_para_b : α ∥ β) : Prop :=
  angle (m, α) = angle (n, β)

-- Main theorem statement
theorem valid_propositions
  (m_perp_n : m ⥊ n)
  (m_perp_α : m ⥊ α)
  (n_para_β : n ∥ β)
  (n_para_α : n ∥ α)
  (a_para_b : α ∥ β)
  (m_subset_α : m ∈ α)
  (m_para_n : m ∥ n)
  : (prop_2 m_perp_α n_para_α) ∧ (prop_3 a_para_b m_subset_α) ∧ (prop_4 m_para_n a_para_b) :=
by
  split; sorry
  split; sorry
  sorry

end valid_propositions_l608_608696


namespace functional_equation_l608_608900

def f (x : ℝ) : ℝ := (x^3 - x^2 + 1) / (2 * x * (1 - x))

theorem functional_equation (x : ℝ) (h : x ≠ 0 ∧ x ≠ 1) : 
  f (1 / x) + f (1 - x) = x :=
  sorry

end functional_equation_l608_608900


namespace stanley_total_cost_l608_608519

theorem stanley_total_cost (n_tires : ℕ) (price_per_tire : ℝ) (h_n : n_tires = 4) (h_price : price_per_tire = 60) : n_tires * price_per_tire = 240 := by
  sorry

end stanley_total_cost_l608_608519


namespace probability_P_plus_S_mod_6_l608_608010

theorem probability_P_plus_S_mod_6 (a b c : ℕ) (h1 : 1 ≤ a) (h2 : a < b) (h3 : b < c) (h4 : c ≤ 60) :
  (∃ (favorable_count total_count : ℕ), favorable_count = 14620 ∧ total_count = 34220 ∧ 
  (IsProbability : favorable_count / total_count = 730 / 1711)) → 
  (related_mod_6 : ∀ (P S : ℕ), P = a * b * c ∧ S = a + b + c → (P + S) % 6 = 5) :=
by
  sorry

end probability_P_plus_S_mod_6_l608_608010


namespace intervals_of_monotonicity_range_of_a_l608_608271

noncomputable def f (a x : ℝ) : ℝ := (2 * a - x^2) / Real.exp x

theorem intervals_of_monotonicity (a : ℝ) :
  (a ≤ -1/2 ∧ ∀ x1 x2, x1 < x2 → x1 ∈ set.Iio (Real.sqrt (2*a + 1) + 1) → 
    x2 ∈ set.Iio (Real.sqrt (2*a + 1) + 1) → f a x1 < f a x2)
  ∨ (a > -1/2 ∧ ∀ x1 x2, x1 < x2 → (x1 ∈ set.Iio (1 - Real.sqrt (2*a + 1)) ∨ 
      x1 ∈ set.Ioi (1 + Real.sqrt (2*a + 1))) → (x2 ∈ set.Iio (1 - Real.sqrt (2*a + 1)) ∨ 
      x2 ∈ set.Ioi (1 + Real.sqrt (2*a + 1))) → f a x1 < f a x2 ∧ f a (1 - Real.sqrt (2*a + 1)) > f a (1 + Real.sqrt (2*a + 1))) :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ set.Ici 1 → f a x > -1) → a > (1 - Real.exp 1) / 2 :=
sorry

end intervals_of_monotonicity_range_of_a_l608_608271


namespace avg_speed_trip_l608_608474

noncomputable def distance_travelled (speed time : ℕ) : ℕ := speed * time

noncomputable def average_speed (total_distance total_time : ℕ) : ℕ := total_distance / total_time

theorem avg_speed_trip :
  let first_leg_speed := 75
  let first_leg_time := 4
  let second_leg_speed := 60
  let second_leg_time := 2
  let total_time := first_leg_time + second_leg_time
  let first_leg_distance := distance_travelled first_leg_speed first_leg_time
  let second_leg_distance := distance_travelled second_leg_speed second_leg_time
  let total_distance := first_leg_distance + second_leg_distance
  average_speed total_distance total_time = 70 :=
by
  sorry

end avg_speed_trip_l608_608474


namespace calculation_of_z_conjugate_product_l608_608260

noncomputable def z : ℂ := (1 - 2 * complex.i) / (1 + complex.i)

theorem calculation_of_z_conjugate_product :
  z * complex.conj z = 5 / 2 :=
sorry

end calculation_of_z_conjugate_product_l608_608260


namespace sequence_bounded_l608_608764

theorem sequence_bounded (a : ℕ → ℕ) (a1 : ℕ) (h1 : a 0 = a1)
  (heven : ∀ n : ℕ, ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ a (2 * n) = a (2 * n - 1) - d)
  (hodd : ∀ n : ℕ, ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ a (2 * n + 1) = a (2 * n) + d) :
  ∀ n : ℕ, a n ≤ 10 * a1 := 
by
  sorry

end sequence_bounded_l608_608764


namespace max_min_values_of_g_l608_608551

noncomputable def g (x : ℝ) : ℝ := (Real.sin x)^8 + 8 * (Real.cos x)^8

theorem max_min_values_of_g :
  (∀ x : ℝ, g x ≤ 8) ∧ (∀ x : ℝ, g x ≥ 8 / 27) :=
by
  sorry

end max_min_values_of_g_l608_608551


namespace sides_ratio_of_arithmetic_sequence_l608_608593

theorem sides_ratio_of_arithmetic_sequence (A B C : ℝ) (a b c : ℝ) 
  (h_arith_sequence : (A = B - (B - C)) ∧ (B = C + (C - A))) 
  (h_angle_B : B = 60)  
  (h_cosine_rule : a^2 + c^2 - b^2 = 2 * a * c * (Real.cos B)) :
  (1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) :=
sorry

end sides_ratio_of_arithmetic_sequence_l608_608593


namespace circumcenter_lies_on_AK_l608_608831

noncomputable def is_circumcenter (O : Point) (A B C : Point) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C

noncomputable def lies_on_line (P Q R : Point) : Prop :=
  ∃ (k : ℝ), Q = P + k • (R - P)

theorem circumcenter_lies_on_AK
  (A B C L H K O : Point)
  (h_triangle : ∀ (X Y Z : Point), X ≠ Y → X ≠ Z → Y ≠ Z → is_triangle X Y Z)
  (h_AL : is_angle_bisector A L B C)
  (h_H : foot B L H)
  (h_K : foot_on_circumcircle B L K (set_circumcircle A B L))
  (h_circ_A : O = is_circumcenter O A B C) :
  lies_on_line A K O :=
sorry

end circumcenter_lies_on_AK_l608_608831


namespace sum_of_squares_areas_l608_608871

theorem sum_of_squares_areas (BE AB : ℝ) (h_BE : BE = 12) (h_AB : AB = 5) (right_angle : true) :
  let AE := real.sqrt (BE^2 - AB^2) in
  let area_AB := AB^2 in
  let area_AE := AE^2 in
  area_AB + area_AE = 144 :=
by
  -- Proof would go here
  sorry

end sum_of_squares_areas_l608_608871


namespace arithmetic_series_sum_l608_608879

theorem arithmetic_series_sum :
  ∀ (a1 an d n : ℤ), a1 = -35 → an = 1 → d = 2 → n = 19 →
    (an = a1 + (n - 1) * d) →
    (S = n * (a1 + an) / 2) →
    S = -323 :=
by
  intros a1 an d n h1 h2 h3 h4 h_term h_sum
  rw [h1, h2, h3] at h_term
  have hn : n = 19 := by linarith,
  rw [hn] at *,
  rw [h_sum, Int.add_comm (-35), Int.mul_comm] at *,
  sorry

end arithmetic_series_sum_l608_608879


namespace log_m_n_eq_2_l608_608626

theorem log_m_n_eq_2 (a m n : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : ∃ x, f x = log_a(x-1) + 4 ∧ (f x, 4)) (h4 : (m, n) = (2, 4)) : 
  log m n = 2 :=
by 
  sorry

end log_m_n_eq_2_l608_608626


namespace cargo_weight_in_pounds_l608_608061

noncomputable def cargo_kg : ℝ := 350
noncomputable def pounds_per_kg : ℝ := 2.2046

def weight_in_pounds (cargo_kg pounds_per_kg : ℝ) : ℝ :=
  cargo_kg * pounds_per_kg

def rounded (x : ℝ) : ℤ := 
  Real.toRat x |>.round

theorem cargo_weight_in_pounds :
  rounded (weight_in_pounds cargo_kg pounds_per_kg) = 772 := by
  sorry

end cargo_weight_in_pounds_l608_608061


namespace average_speed_72_l608_608057

theorem average_speed_72 (s : ℝ) (h1 : s > 0) :
  let dist1 := s / 2,
      dist2 := s / 6,
      dist3 := s / 3,
      speed1 := 60,
      speed2 := 120,
      speed3 := 80,
      t1 := dist1 / speed1,
      t2 := dist2 / speed2,
      t3 := dist3 / speed3,
      t_total := t1 + t2 + t3,
      avg_speed := s / t_total
  in avg_speed = 72 :=
by
  intros,
  let dist1 := s / 2,
  let dist2 := s / 6,
  let dist3 := s / 3,
  let speed1 := 60,
  let speed2 := 120,
  let speed3 := 80,
  let t1 := dist1 / speed1,
  let t2 := dist2 / speed2,
  let t3 := dist3 / speed3,
  let t_total := t1 + t2 + t3,
  let avg_speed := s / t_total,
  have ht1 : t1 = s / 120 := by sorry,
  have ht2 : t2 = s / 720 := by sorry,
  have ht3 : t3 = s / 240 := by sorry,
  have ht_total : t_total = s / 72 := by sorry,
  have h_avg_speed : avg_speed = 72 := by sorry,
  exact h_avg_speed

end average_speed_72_l608_608057


namespace percentage_non_honda_red_cars_l608_608632

theorem percentage_non_honda_red_cars 
  (total_cars : ℕ)
  (honda_cars : ℕ)
  (toyota_cars : ℕ)
  (ford_cars : ℕ)
  (other_cars : ℕ)
  (perc_red_honda : ℕ)
  (perc_red_toyota : ℕ)
  (perc_red_ford : ℕ)
  (perc_red_other : ℕ)
  (perc_total_red : ℕ)
  (hyp_total_cars : total_cars = 900)
  (hyp_honda_cars : honda_cars = 500)
  (hyp_toyota_cars : toyota_cars = 200)
  (hyp_ford_cars : ford_cars = 150)
  (hyp_other_cars : other_cars = 50)
  (hyp_perc_red_honda : perc_red_honda = 90)
  (hyp_perc_red_toyota : perc_red_toyota = 75)
  (hyp_perc_red_ford : perc_red_ford = 30)
  (hyp_perc_red_other : perc_red_other = 20)
  (hyp_perc_total_red : perc_total_red = 60) :
  (205 / 400) * 100 = 51.25 := 
by {
  sorry
}

end percentage_non_honda_red_cars_l608_608632


namespace problem1_problem2_problem3_l608_608328

-- Problem 1
theorem problem1 (a b c : ℝ) (x1 y1 x2 y2 : ℝ) : 
  let η := (a * x1 + b * y1 + c) * (a * x2 + b * y2 + c) in
  (η < 0) → 
  (a = 1 ∧ b = 1 ∧ c = -1) ∧ (x1 = 1 ∧ y1 = 2 ∧ x2 = -1 ∧ y2 = 0) → 
  true := 
by -- Problem statement only
  sorry

-- Problem 2
theorem problem2 (k : ℝ) : 
  let line_eqn := (1 - 4 * k ^ 2) * x ^ 2 = 1 in
  ((-∞ ≤ k ∧ k ≤ -1/2) ∨ (1/2 ≤ k ∧ k ≤ ∞)) →
  true := 
by -- Problem statement only
  sorry

-- Problem 3
theorem problem3 (k : ℝ) : 
  (let curve_eqn := (x^2 + (y - 2)^2) * x^2 = 1 in 
   (k = 0) ∨
   ((k ≠ 0) → (x != 0) → [(x^2 + (k * x - 2)^2) * x^2 ≠ 1])) → 
  true := 
by -- Problem statement only 
  sorry  

end problem1_problem2_problem3_l608_608328


namespace find_threedigit_number_l608_608074

-- Define the three-digit number and its reverse
def original_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

-- Define the condition of adding the number and its reverse to get 1777
def number_sum_condition (a b c : ℕ) : Prop :=
  original_number a b c + reversed_number a b c = 1777

-- Prove the existence of digits a, b, and c that satisfy the conditions
theorem find_threedigit_number :
  ∃ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  original_number a b c = 859 ∧ 
  reversed_number a b c = 958 ∧ 
  number_sum_condition a b c :=
sorry

end find_threedigit_number_l608_608074


namespace sufficient_but_not_necessary_condition_l608_608608

def M : Set ℝ := {x | 0 < x ∧ x ≤ 2}

def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by
  sorry

end sufficient_but_not_necessary_condition_l608_608608


namespace smallest_sum_infinite_geometric_progression_l608_608762

theorem smallest_sum_infinite_geometric_progression :
  ∃ (a q A : ℝ), (a * q = 3) ∧ (0 < q) ∧ (q < 1) ∧ (A = a / (1 - q)) ∧ (A = 12) :=
by
  sorry

end smallest_sum_infinite_geometric_progression_l608_608762


namespace smallest_square_area_l608_608023

/-- 
The area of the smallest square that can contain a circle of radius 7 is 196. 
-/
theorem smallest_square_area (r : ℝ) (h_r : r = 7) :
  ∃ s : ℝ, (s = 2 * (2 * r) ∧ s ^ 2 = 196) :=
by
  use 2 * (2 * r)
  rw h_r
  sorry

end smallest_square_area_l608_608023


namespace divisible_by_11_of_sum_divisible_l608_608250

open Int

theorem divisible_by_11_of_sum_divisible (a b : ℤ) (h : 11 ∣ (a^2 + b^2)) : 11 ∣ a ∧ 11 ∣ b :=
sorry

end divisible_by_11_of_sum_divisible_l608_608250


namespace computation_result_l608_608182

theorem computation_result :
  2 + 8 * 3 - 4 + 7 * 2 / 2 * 3 = 43 :=
by
  sorry

end computation_result_l608_608182


namespace kurt_less_marbles_than_dennis_l608_608673

theorem kurt_less_marbles_than_dennis
  (Laurie_marbles : ℕ)
  (Kurt_marbles : ℕ)
  (Dennis_marbles : ℕ)
  (h1 : Laurie_marbles = 37)
  (h2 : Laurie_marbles = Kurt_marbles + 12)
  (h3 : Dennis_marbles = 70) :
  Dennis_marbles - Kurt_marbles = 45 := by
  sorry

end kurt_less_marbles_than_dennis_l608_608673


namespace largest_prime_divisor_in_range_1000_to_1100_l608_608011

theorem largest_prime_divisor_in_range_1000_to_1100 : 
  ∀ n, (1000 ≤ n ∧ n ≤ 1100) → 
  ∃ p, (nat.prime p ∧ p ≤ nat.floor (real.sqrt (1100 : ℝ)) ∧ p = 31) :=
by
  sorry

end largest_prime_divisor_in_range_1000_to_1100_l608_608011


namespace number_of_people_attending_both_l608_608455

open Finset

variables {α : Type*} (U O Y : Finset α) (n m t : ℕ)

-- Conditions
def total_guests := card U = 100
def oates_reunion := card O = 42
def yellow_reunion := card Y = 65
def all_attend_reunion := U = O ∪ Y

-- Theorem to prove
theorem number_of_people_attending_both {α : Type*} {U O Y : Finset α} :
  card U = 100 → card O = 42 → card Y = 65 → U = O ∪ Y → card (O ∩ Y) = 7 :=
by
  intros hU hO hY hUnion
  -- Sorry is used to skip the proof
  sorry

end number_of_people_attending_both_l608_608455


namespace ratio_of_areas_l608_608298

theorem ratio_of_areas (s : ℝ) (h : s > 0) :
  let A1 := (2 * s) ^ 2 in
  let A2 := s ^ 2 in
  A1 / A2 = 4 :=
by
  let A1 := (2 * s) ^ 2
  let A2 := s ^ 2
  sorry

end ratio_of_areas_l608_608298


namespace nickel_ate_3_chocolates_l608_608719

theorem nickel_ate_3_chocolates (R N : ℕ) (h1 : R = 7) (h2 : R = N + 4) : N = 3 := by
  sorry

end nickel_ate_3_chocolates_l608_608719


namespace probability_of_sum_prime_greater_20_l608_608204

noncomputable def first_twelve_primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}

def is_prime (n : ℕ) : Prop := Nat.Prime n

def sum_is_prime_greater_than_20 (a b : ℕ) : Prop :=
  is_prime (a + b) ∧ (a + b) > 20

theorem probability_of_sum_prime_greater_20 :
  (∃ (a b ∈ first_twelve_primes), sum_is_prime_greater_than_20 a b) →
  (Finset.card (first_twelve_primes.powerset.filter (λ s, 2 ≤ s.card ∧ s.card ≤ 2 ∧ sum_is_prime_greater_than_20 (s.erase a).erase b)) : ℚ) /
  (Finset.card (first_twelve_primes.powerset.filter (λ s, 2 ≤ s.card ∧ s.card ≤ 2)) : ℚ) = 1 / 66 :=
sorry

end probability_of_sum_prime_greater_20_l608_608204


namespace projection_of_AD_in_direction_of_CB_l608_608251

noncomputable def vector_projection : ℚ :=
  let A := (-1, 1) in
  let B := (1, 2) in
  let C := (-2, 1) in
  let D := (3, 4) in
  let AD := (4, 3) in
  let CB := (3, 1) in
  let dot_product_AD_CB := 4 * 3 + 3 * 1 in
  let magnitude_CB := real.sqrt (3^2 + 1^2) in
  (dot_product_AD_CB / magnitude_CB)

theorem projection_of_AD_in_direction_of_CB :
  vector_projection = (3 * real.sqrt 10) / 2 :=
by
  sorry

end projection_of_AD_in_direction_of_CB_l608_608251


namespace binomial_coeff_x4_l608_608329

theorem binomial_coeff_x4:
  (let expr := (2 * x^2 - 1 / x)^5 in
   let general_term (r: ℕ) := ((-1)^r * 2^(5 - r) * Nat.choose 5 r * x^(10 - 3 * r)) in
   ∃ (r: ℕ), 10 - 3 * r = 4 ∧ ((-1)^r * 2^(5 - r) * Nat.choose 5 r) = 80) :=
begin
  sorry,
end

end binomial_coeff_x4_l608_608329


namespace integer_145_in_column_A_l608_608148

theorem integer_145_in_column_A : 
  ∀ (n : ℕ), n ≥ 3 → 
    let columns := ['A', 'B', 'C', 'D', 'E', 'F', 'E', 'D', 'C', 'B', 'A'] in
    let pos := (n - 3) % 11 in
    n = 145 → columns[pos] = 'A' :=
by
  intros n hn columns pos h145
  sorry

end integer_145_in_column_A_l608_608148


namespace initial_discount_percentage_l608_608484

variable (d : ℝ) (x : ℝ)
variable (h1 : 0 < d) (h2 : 0 ≤ x) (h3 : x ≤ 100)
variable (h4 : (1 - x / 100) * 0.6 * d = 0.33 * d)

theorem initial_discount_percentage : x = 45 :=
by
  sorry

end initial_discount_percentage_l608_608484


namespace probability_x_plus_y_lt_4_l608_608119

open Set

-- Define the square and the line
def square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 }
def line_lt_4 : Set (ℝ × ℝ) := { p | p.1 + p.2 < 4 }

-- The probability to prove
theorem probability_x_plus_y_lt_4 : 
  (volume (square ∩ line_lt_4) / volume square) = 7 / 9 := 
sorry

end probability_x_plus_y_lt_4_l608_608119


namespace average_speed_is_50_l608_608056

-- Let D be the total distance between town X and town Y
variables (D : ℝ) (hD : D > 0)

-- Definitions of the distances and speeds for each segment
def distance_part1 := 0.4 * D
def speed_part1 := 40
def distance_part2 := 0.6 * D
def speed_part2 := 60

-- Calculate the times for each part of the trip
def time_part1 := distance_part1 / speed_part1
def time_part2 := distance_part2 / speed_part2

-- Total time for the trip
def total_time := time_part1 + time_part2

-- Average speed for the entire trip
def average_speed := D / total_time

-- The main statement to prove
theorem average_speed_is_50 :
  average_speed D hD = 50 := 
  sorry

end average_speed_is_50_l608_608056


namespace calcium_chloride_formation_l608_608912

theorem calcium_chloride_formation 
  (moles_HCl : ℕ) (moles_CaCO3 : ℕ)
  (balanced_reaction : CaCO3 + 2 * HCl → CaCl2 + CO2 + H2O)
  (HCl_amount : moles_HCl = 4)
  (CaCO3_amount : moles_CaCO3 = 2) :
  ∃ (moles_CaCl2 : ℕ), moles_CaCl2 = 2 :=
by
  sorry

end calcium_chloride_formation_l608_608912


namespace smallest_number_of_coins_correct_l608_608179

noncomputable def smallest_number_of_coins : ℕ :=
  2 ^ 16

theorem smallest_number_of_coins_correct :
  ∃ n, (∃ (Y : ℕ → Prop), (∀ (y : ℕ), Y y → 2 ≤ y ∧ y < n) ∧ ∃ t, ∀ k, P k ↔ t ∈ P k) ∧ n = 65536 :=
  by
  use 65536
  sorry

end smallest_number_of_coins_correct_l608_608179


namespace sqrt_solution_range_l608_608206

theorem sqrt_solution_range : 
  7 < (Real.sqrt 32) * (Real.sqrt (1 / 2)) + (Real.sqrt 12) ∧ (Real.sqrt 32) * (Real.sqrt (1 / 2)) + (Real.sqrt 12) < 8 := 
by
  sorry

end sqrt_solution_range_l608_608206


namespace find_N_less_than_500_l608_608547

theorem find_N_less_than_500 : 
  let valid_N (N : ℕ) : Prop := ∃ x : ℝ, x > 0 ∧ x < 500 ∧ (x ^ (floor x)) = N
  in (fintype.card { N // valid_N N ∧ N < 500 }) = 287 := 
  sorry

end find_N_less_than_500_l608_608547


namespace alice_prevents_bob_winning_l608_608425

-- There are 60 boxes B1, B2, ..., B60
constant n_boxes : ℕ := 60
def boxes := fin n_boxes

-- Bob's range of choices
def k_range := fin 60

-- Alice's and Bob's play the game
def game (n : ℕ) := 
  -- Alice distributes n pebbles such that she can always prevent Bob from winning
  ∀ (i : boxes),  ∃ (distr : boxes → ℕ), 
    -- Each subsequent round of the game
    (∀ (k : k_range), 
      -- Bob chooses k with 1 ≤ k ≤ 59
      (1 ≤ k.1 ∧ k.1 ≤ 59) → 
      -- Split into two groups
      let group1 := (λ b : boxes, b < k) 
      let group2 := (λ b : boxes, b ≥ k) in
      -- Alice picks one group to add 1 pebble and subtract 1 from the other
      (
        (∀ (b : boxes), distr b > 0) ∨ 
        (∀ (b : boxes), distr b > 0 )
      )
    )
  -- At least one initial distribution such that no box ever becomes empty
  ∧ ∀ (distr : boxes → ℕ), ∃ m,  ∀ (b : boxes), distr b ≥ 1

-- Prove that Alice can prevent Bob from winning with minimum pebbles n = 960
theorem alice_prevents_bob_winning : ∃ (n : ℕ), game n ∧ n = 960 := 
begin
  sorry
end

end alice_prevents_bob_winning_l608_608425


namespace diagonals_EFGH_l608_608369

variable (EFGH : Type) (convex_EFGH : Convex EFGH)
variable (K L M N : EFGH)
variable (midpoints : K = midpoint E F ∧ L = midpoint F G ∧ M = midpoint G H ∧ N = midpoint H E)
variable (O : EFGH) (O_intersect : O = intersect K M L N)
variable (angle_LOM : ∠ L O M = 90)
variable (length_KM_LN : KM = 3 * LN)
variable (area_KLMN : ℝ) (area_KLMN_eq : area KLMN = S)

theorem diagonals_EFGH (S : ℝ) : (diagonal EFGH).length = 2 * sqrt(6 * S) := 
sorry

end diagonals_EFGH_l608_608369


namespace locus_e_eq_fixed_point_exists_l608_608571

noncomputable def circle_f1 : {p : ℝ × ℝ // (p.1 + 2)^2 + p.2^2 = 24} := sorry

def point_f2 : ℝ × ℝ := (2, 0)

noncomputable def tangent_circle_n (p : ℝ × ℝ) : Prop :=
  (∃ r : ℝ, (p.1 - point_f2.1)^2 + (p.2 - point_f2.2)^2 = r^2 ∧
   ∃ q : {p : ℝ × ℝ // (p.1 + 2)^2 + p.2^2 = 24}, (p.1 - q.val.1)^2 + (p.2 - q.val.2)^2 = (r + sqrt 24)^2)

def locus_e (p : ℝ × ℝ) : Prop := 
  (p.1^2 / 6 + p.2^2 / 2 = 1)

theorem locus_e_eq : ∀ p : ℝ × ℝ,
  tangent_circle_n p → locus_e p :=
sorry

noncomputable def fixed_point_m : ℝ × ℝ := (7 / 3, 0)

theorem fixed_point_exists : ∀ (A B : ℝ × ℝ),
  tangent_circle_n A → tangent_circle_n B →
  ∃ (M : ℝ × ℝ), M = fixed_point_m ∧
  (∃ c : ℝ, (A.1 - M.1)^2 + (A.2 - M.2)^2 + (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = c) ∧
  c = -5 / 9 :=
sorry

end locus_e_eq_fixed_point_exists_l608_608571


namespace solve_line_through_midpoint_l608_608536

open Classical

noncomputable def point := ℝ × ℝ

def line (a b c : ℝ) (p: point) : Prop :=
  ∃ x y : ℝ, p = (x, y) ∧ a * x + b * y + c = 0

noncomputable def midpoint (A B : point) :=
let (x1, y1) := A in
let (x2, y2) := B in
(x1 + x2) / 2, (y1 + y2) / 2

theorem solve_line_through_midpoint :
  ∃ a b c : ℝ, ∀ P : point, P = (3, 0) →
    (∀ l1 l2 : ℝ, line 2 (-1) (-2) l1 ∧ line 1 1 3 l2 →
    let A := fst l1 in
    let B := snd l2 in
    midpoint A B = P) →
    line 8 (-1) (-24) P := 
begin
  sorry
end

end solve_line_through_midpoint_l608_608536


namespace antonov_packs_l608_608513

theorem antonov_packs (total_candies packs_given pieces_per_pack remaining_pieces packs_remaining : ℕ) 
    (h1 : total_candies = 60) 
    (h2 : packs_given = 1) 
    (h3 : pieces_per_pack = 20) 
    (h4 : remaining_pieces = total_candies - (packs_given * pieces_per_pack)) 
    (h5 : packs_remaining = remaining_pieces / pieces_per_pack) : 
    packs_remaining = 2 := 
by
  rw [h1, h3] at h4
  rw [Nat.mul_comm, Nat.sub_eq_iff_eq_add, Nat.sub_sub] at h4
  rw [Nat.mul_comm, Nat.div_eq_iff_eq_mul, Nat.mul_comm] at h5
  exact h5
sorry

end antonov_packs_l608_608513


namespace minimum_rectangles_needed_l608_608805

def type1_corners := 12
def type2_corners := 12
def group_size := 3

theorem minimum_rectangles_needed (cover_type1: ℕ) (cover_type2: ℕ)
  (type1_corners coverable_by_one: ℕ) (type2_groups_num: ℕ) :
  type1_corners = 12 → type2_corners = 12 → type2_groups_num = 4 →
  group_size = 3 → cover_type1 + cover_type2 = 12 :=
by
  intros h1 h2 h3 h4 
  sorry

end minimum_rectangles_needed_l608_608805


namespace beer_drawing_time_l608_608052

theorem beer_drawing_time (midway_rate : ℕ) (bottom_rate : ℕ) (capacity : ℕ) (used_before : ℕ)
  (usual_time_halfway : ℕ) (usual_time_lower : ℕ) :
  usual_time_halfway * midway_rate / capacity = usual_time_lower →
  usual_time_lower + used_before = 132 →
  36 - 33 = 3 →
  3 * 6 = 18 →
  132 + 18 = 150 :=
begin
  sorry
end

end beer_drawing_time_l608_608052


namespace locus_of_midpoints_of_common_tangents_l608_608550

-- Definitions of the given conditions
variables {O1 O2 : ℝ^3} {R1 R2 : ℝ}

-- The main theorem: geometric locus of the midpoints of common tangents
theorem locus_of_midpoints_of_common_tangents
  (H : ∃ (O1 O2 : ℝ^3) (R1 R2 : ℝ), 
       (∀ P : ℝ^3, ((P - O1).norm ^ 2 - R1 ^ 2) = ((P - O2).norm ^ 2 - R2 ^ 2))
       ∧ (O1 ≠ O2) ∧ (R1 > 0) ∧ (R2 > 0)) :
  ∃ (NQ NP NM : ℝ), 
  (∀ P Q : ℝ^3, 
    let M := midpoint (common_tangent O1 R1) (common_tangent O2 R2) in 
    (NQ ≤ (dist N M)) ∧ (dist N M ≤ NP)) :=
  sorry

end locus_of_midpoints_of_common_tangents_l608_608550


namespace D_compatibility_unique_find_n_Dn_eq_n_limit_Dm_63_l608_608889

noncomputable def D : ℕ → ℕ
| 1 := 0
| n := if is_prime n then 1 else 
          let factors := multiset.pmap (λ p k, (p, k)) (nat.factors n) nat.prime_factors_spec in
          if n > 1 then
            let u := factors ≠ [] → factors.foldl (λ acc p_k, acc + (p_k.1 * D p_k.2 / p_k.1)) 0 in
            let v := factors ≠ [] → factors.foldl (λ acc p_k, acc + (D p_k.1 * p_k.2 / p_k.1)) 0 in
            u * D v + v * D u
          else 0

theorem D_compatibility_unique : ∀ n, (D n = 0 → n = 1) ∧
                (∀ p, is_prime p → D p = 1) ∧
                (∀ u v, D (u * v) = u * D v + v * D u) :=
sorry

theorem find_n_Dn_eq_n : ∃ n, D n = n :=
sorry

theorem limit_Dm_63 : ∀ m, D^m 63 → limit D 63 m = +∞ :=
sorry

end D_compatibility_unique_find_n_Dn_eq_n_limit_Dm_63_l608_608889


namespace expected_faces_of_5_in_100_rolls_l608_608720

theorem expected_faces_of_5_in_100_rolls (rolls : ℕ) (p : ℚ) (E : ℚ) :
  rolls = 100 ∧ p = 1/6 → E = (100 * (1/6)) := by
  sorry

end expected_faces_of_5_in_100_rolls_l608_608720


namespace surface_area_of_cube_l608_608860

theorem surface_area_of_cube (a b c : ℝ) (hv : a = 10) (hv2 : b = 10) (hv3 : c = 8) :
  let V := a * b * c in
  let s := V^(1/3) in
  6 * s^2 = 1200 :=
by
  sorry

end surface_area_of_cube_l608_608860


namespace range_of_h_l608_608224

noncomputable def h (t : ℝ) : ℝ := (t^2 + 0.5 * t) / (t^2 + 2)

theorem range_of_h : 
  Set.Icc (0.5 - 3 * Real.sqrt 2 / 16) (0.5 + 3 * Real.sqrt 2 / 16) = 
  {y | ∃ t : ℝ, y = h(t)} := 
sorry

end range_of_h_l608_608224


namespace prove_condition_for_equality_l608_608289

noncomputable def condition_for_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : Prop :=
  c = (b * (a ^ 3 - 1)) / a

theorem prove_condition_for_equality (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∃ (c' : ℕ), (c' = (b * (a ^ 3 - 1)) / a) ∧ 
      c' > 0 ∧ 
      (a + b / c' = a ^ 3 * (b / c')) ) → 
  c = (b * (a ^ 3 - 1)) / a := 
sorry

end prove_condition_for_equality_l608_608289


namespace min_quotient_of_number_l608_608036

theorem min_quotient_of_number (a : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : a ≠ 7) (h3 : a ≠ 8) :
  (∃ a ∈ {1, 2, 3, 4, 5, 6, 9}, (100 * a + 78) / (a + 15) = 11.125) :=
by
  sorry

end min_quotient_of_number_l608_608036


namespace grid_labeling_count_l608_608768

theorem grid_labeling_count :
  ∃ f : fin 3 → fin 3 → ℕ, 
    f ⟨0, _⟩ ⟨0, _⟩ = 1 ∧
    f ⟨2, _⟩ ⟨2, _⟩ = 2009 ∧
    (∀ i j : fin 3, j < 2 → f i j ∣ f i ⟨j + 1, _⟩) ∧
    (∀ i j : fin 3, i < 2 → f i j ∣ f ⟨i + 1, _⟩ j) ∧
    card { f | (f ⟨2, _⟩ ⟨1, _) ∣ 2009 ∧
             (f ⟨1, _⟩ ⟨2, _) ∣ 2009 ∧
             (f ⟨1, _⟩ ⟨1, _) ∈ {1, 7, 41, 49, 287, 2009 } } = 2448 :=
sorry

end grid_labeling_count_l608_608768


namespace train_crossing_time_l608_608660

noncomputable def speed_kmhr_to_ms (v_kmhr : ℕ) : ℝ :=
  v_kmhr * 1000 / 3600

def time_to_cross_pole (L v_kmhr : ℝ) : ℝ :=
  L / speed_kmhr_to_ms v_kmhr

theorem train_crossing_time :
  time_to_cross_pole 100 126 = 100 / (126 * 1000 / 3600) :=
by
  sorry

end train_crossing_time_l608_608660


namespace chores_minutes_proof_l608_608208

-- Definitions based on conditions
def minutes_of_cartoon_per_hour := 60
def cartoon_watched_hours := 2
def cartoon_watched_minutes := cartoon_watched_hours * minutes_of_cartoon_per_hour
def ratio_of_cartoon_to_chores := 10 / 8

-- Definition based on the question
def chores_minutes (cartoon_minutes : ℕ) : ℕ := (8 * cartoon_minutes) / 10

theorem chores_minutes_proof : chores_minutes cartoon_watched_minutes = 96 := 
by sorry 

end chores_minutes_proof_l608_608208


namespace evaluate_custom_op_l608_608605

def custom_op (a b : ℝ) : ℝ := (a - b)^2

theorem evaluate_custom_op (x y : ℝ) : custom_op ((x + y)^2) ((y - x)^2) = 16 * x^2 * y^2 :=
by
  sorry

end evaluate_custom_op_l608_608605


namespace goals_in_fifth_match_proof_l608_608485

noncomputable def goals_in_fifth_match (total_goals_in_5_matches : ℤ) (average_increase : ℚ) : ℤ :=
  let average_first_4_matches := (total_goals_in_5_matches / 5 : ℚ) - average_increase
  let total_goals_first_4_matches := 4 * average_first_4_matches
  total_goals_in_5_matches - total_goals_first_4_matches

theorem goals_in_fifth_match_proof
  (total_goals_in_5_matches : ℤ)
  (average_increase : ℚ)
  (h1 : total_goals_in_5_matches = 21)
  (h2 : average_increase = 0.2) :
  goals_in_fifth_match total_goals_in_5_matches average_increase = 5 := by
  sorry

end goals_in_fifth_match_proof_l608_608485


namespace range_of_m_l608_608188

theorem range_of_m (m : ℝ) :
  ((∀ x : ℝ, mx^2 + 1 > 0) ↔ m ≥ 0) ∧ ((∀ x : ℝ, ∃! x : ℝ, f x < f (x + 1)) ↔ m < 0) →
  (m = 0 ∨ (m ≥ 1 ∧ ¬ (m < 1))) :=
by
  -- This is where the proof will go
  sorry

end range_of_m_l608_608188


namespace probability_of_x_plus_y_less_than_4_l608_608125

-- Define the square and the probability that x + y < 4 within this square.
theorem probability_of_x_plus_y_less_than_4 : 
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  (desired_area / square_area = 7 / 9) :=
by
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  show (desired_area / square_area = 7 / 9)
  sorry

end probability_of_x_plus_y_less_than_4_l608_608125


namespace probability_x_plus_y_lt_4_l608_608112

theorem probability_x_plus_y_lt_4 :
  let square_vertices := {(0,0), (0,3), (3,3), (3,0)} in
  let point_in_square (x y : ℝ) := 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 in
  let probability (A : ℝ) (B : ℝ) := A / B in
  ∃ (P : ℝ), P = probability 7 9 ∧
             P = (measure (λ (x y : ℝ), point_in_square x y ∧ x + y < 4)) / (measure (λ (x y : ℝ), point_in_square x y)) :=
sorry

end probability_x_plus_y_lt_4_l608_608112


namespace exactly_one_real_root_l608_608927

noncomputable def has_unique_real_root (f : ℝ → ℝ) (m n : ℝ) : Prop :=
f m * f n < 0 ∧ ∀ x y ∈ set.Icc m n, x ≠ y → f x ≠ f y

theorem exactly_one_real_root (f : ℝ → ℝ) (m n : ℝ) (h_cont : continuous_on f (set.Icc m n)) (h_dec : ∀ x y ∈ set.Icc m n, x < y → f x > f y) (h_sign : f m * f n < 0) : has_unique_real_root f m n :=
by
  sorry

end exactly_one_real_root_l608_608927


namespace card_game_probability_l608_608826

theorem card_game_probability :
  let n := 6
  let prob_empty (p q : ℕ) :=
    (p, q) = (9, 385) ∧ Nat.coprime p q
  let solution (p q : ℕ) :=
    p + q = 394
  ∃ p q : ℕ, prob_empty p q ∧ solution p q :=
begin
  have h : ∃ p q : ℕ, (p, q) = (9, 385) ∧ Nat.coprime p q,
  { use [9, 385],
    exact ⟨rfl, by norm_num⟩, },
  exact h,
end

end card_game_probability_l608_608826


namespace total_students_in_grade_l608_608380

theorem total_students_in_grade (best_pos worst_pos : ℕ) 
  (h_best : best_pos = 50) 
  (h_worst : worst_pos = 50) : 
  ∃ n : ℕ, n = 99 :=
begin
  sorry
end

end total_students_in_grade_l608_608380


namespace real_root_polynomials_l608_608195

theorem real_root_polynomials (P : Polynomial ℝ) (n : ℕ):
  (∀ i, i ∈ (Finset.range (n + 1)) → P.coeff i = 1 ∨ P.coeff i = -1) ∧ P.degree ≤ n ∧ n ≤ 3
  → (∃ (Q : Polynomial ℝ), Q = P ∧ (Q = Polynomial.Coeff (x - 1) ∨ Q = Polynomial.Coeff (x + 1) 
  ∨ Q = Polynomial.Coeff (x^2 + x - 1) ∨ Q = Polynomial.Coeff (x^2 - x - 1) 
  ∨ Q = Polynomial.Coeff (x^3 + x^2 - x - 1) ∨ Q = Polynomial.Coeff (x^3 - x^2 - x + 1)))
:= sorry

end real_root_polynomials_l608_608195


namespace part1_solution_part2_solution_l608_608599

def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 3)

theorem part1_solution (x : ℝ) : 
  f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2 := sorry

theorem part2_solution (a : ℝ) :
  (∃ x : ℝ, f x a < 2 * a) ↔ 3 < a := sorry

end part1_solution_part2_solution_l608_608599


namespace meet_in_9_turns_l608_608865

-- Define the problem conditions
def circle_points : ℕ := 18
def alice_move : ℕ := 7
def bob_move : ℕ := 13 -- 13 counterclockwise is equivalent to (18 - 13) = 5 clockwise

-- The number of turns required for Alice and Bob to meet
def turns_to_meet (n : ℕ) : Prop :=
  (alice_move - (circle_points - bob_move)) * n % circle_points = 0

theorem meet_in_9_turns : turns_to_meet 9 :=
by
  -- Use the conditions and prove they meet in 9 turns
  unfold turns_to_meet
  have h : (7 + 5) % 18 = 12 % 18 := by norm_num
  have h_mod : 12 % 18 = 12 := by norm_num
  rw [h, h_mod]
  norm_num
  sorry

end meet_in_9_turns_l608_608865


namespace factorial_division_l608_608469

theorem factorial_division :
  nat.factorial 9 / nat.factorial 6 = 504 :=
sorry

end factorial_division_l608_608469


namespace parabola_vertex_coordinates_l608_608739

theorem parabola_vertex_coordinates :
  ∃ (x y : ℝ), (∀ x : ℝ, y = 3 * x^2 + 2) ∧ x = 0 ∧ y = 2 :=
by
  sorry

end parabola_vertex_coordinates_l608_608739


namespace center_of_circumcircle_lies_on_AK_l608_608840

variable {α β γ : Real} -- Angles in triangle ABC
variable (A B C L H K O : Point) -- Points in the configuration
variable (circumcircle_ABC : TriangularCircumcircle A B C) -- Circumcircle of triangle ABC

-- Definitions based on the given conditions
variable (is_angle_bisector : angle_bisector A B C L)
variable (is_height : height_from_point_to_line B A L H)
variable (intersects_circle_at_K : intersects_circumcircle A B L K circumcircle_ABC)
variable (is_circumcenter : circumcenter A B C O circumcircle_ABC)

theorem center_of_circumcircle_lies_on_AK
  (h_angle_bisector : is_angle_bisector)
  (h_height : is_height)
  (h_intersects_circle_at_K : intersects_circle_at_K)
  (h_circumcenter : is_circumcenter) 
    : lies_on_line O A K := 
sorry -- Proof is omitted

end center_of_circumcircle_lies_on_AK_l608_608840


namespace parallel_perpendicular_implies_perpendicular_l608_608580

variables {Line Plane : Type}
variables (a b : Line) (α : Plane)

-- Definitions of geometric relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def perpendicular_to_line (l1 l2 : Line) : Prop := sorry

-- Assumptions
axiom non_coincident_lines : a ≠ b
axiom non_coincident_planes : α ≠ sorry -- Placeholder for different planes as per the conditions

-- Proof statement
theorem parallel_perpendicular_implies_perpendicular
  (ha : parallel a α)
  (hb : perpendicular b α) :
  perpendicular_to_line a b := 
begin
  sorry
end

end parallel_perpendicular_implies_perpendicular_l608_608580


namespace multiplicative_inverse_of_AB_mod_2000000_l608_608370

-- Defining A' and B' as constants
def A' : ℕ := 222222
def B' : ℕ := 285714

-- Stating that N is the multiplicative inverse of A'B' modulo 2,000,000
theorem multiplicative_inverse_of_AB_mod_2000000 :
  ∃ N : ℕ, N < 1000000 ∧ (N * (A' * B') % 2000000) = 1 :=
begin
  use 1500000,
  split,
  { -- Prove that N < 1000000
    trivial, -- 1500000 is less than 1000000
  },
  {
    -- Prove that (N * (A' * B') % 2000000) = 1
    sorry,
  }
end


end multiplicative_inverse_of_AB_mod_2000000_l608_608370


namespace number_arrangement_0_to_9_impossible_number_arrangement_1_to_13_impossible_l608_608037

-- Problem (a)
theorem number_arrangement_0_to_9_impossible 
  : ¬ ∃ (f: ℕ → ℤ), (∀ i, 0 ≤ f i ∧ f i ≤ 9) ∧ (∀ i, ((f (i+1)) - (f i)).nat_abs = 3 ∨ ((f (i+1)) - (f i)).nat_abs = 4 ∨ ((f (i+1)) - (f i)).nat_abs = 5) := sorry

-- Problem (b)
theorem number_arrangement_1_to_13_impossible 
  : ¬ ∃ (f: ℕ → ℤ), (∀ i, 1 ≤ f i ∧ f i ≤ 13) ∧ (∀ i, ((f (i+1)) - (f i)).nat_abs = 3 ∨ ((f (i+1)) - (f i)).nat_abs = 4 ∨ ((f (i+1)) - (f i)).nat_abs = 5) := sorry

end number_arrangement_0_to_9_impossible_number_arrangement_1_to_13_impossible_l608_608037


namespace sum_powers_of_i_l608_608185

theorem sum_powers_of_i:
  (∑ k in Finset.range (207), (complex.I)^(k - 103) : ℂ) = -1 :=
begin
  let i := complex.I,
  have h_cyclic : ∀ n, (i ^ n : ℂ) = i ^ (n % 4),
  { intro n, rw ← complex.I_pow n % 4 },
  sorry
end

end sum_powers_of_i_l608_608185


namespace ziggy_rap_requests_l608_608446

variables (total_songs electropop dance rock oldies djs_choice rap : ℕ)

-- Given conditions
axiom total_songs_eq : total_songs = 30
axiom electropop_eq : electropop = total_songs / 2
axiom dance_eq : dance = electropop / 3
axiom rock_eq : rock = 5
axiom oldies_eq : oldies = rock - 3
axiom djs_choice_eq : djs_choice = oldies / 2

-- Proof statement
theorem ziggy_rap_requests : rap = total_songs - electropop - dance - rock - oldies - djs_choice :=
by
  -- Apply the axioms and conditions to prove the resulting rap count
  sorry

end ziggy_rap_requests_l608_608446


namespace sum_of_elements_l608_608997

theorem sum_of_elements (a b c : ℕ) (h : {1, 2, 3} = {a, b, c}) : a + b + c = 6 :=
sorry

end sum_of_elements_l608_608997


namespace Chebyshev_birth_year_is_1821_l608_608791

theorem Chebyshev_birth_year_is_1821 :
  ∃ (a b : ℕ), 
    let year := 1800 + 10 * a + b in
    1 + 8 = 3 * (a + b) ∧
    a > b ∧
    year = 1821 ∧
    year + 73 < 1900 :=
by
  sorry

end Chebyshev_birth_year_is_1821_l608_608791


namespace find_s_l608_608517

-- Definitions for the sides of the triangle and quadrilateral
variables (s : ℝ) (MU UN NC CH HM : ℝ)
-- Conditions from the problem
def conditions := MU = s ∧ UN = 6 ∧ NC = 20 ∧ CH = s ∧ HM = 25
-- The property that the areas of triangle UNC and quadrilateral MUCH are equal
def equal_areas (MUCH UNC : ℝ) := MUCH = UNC

theorem find_s : 
  ∀ (MUCH UNC : ℝ), (MU = s ∧ UN = 6 ∧ NC = 20 ∧ CH = s ∧ HM = 25) →
  (MUCH = UNC) →
  s = 4 :=
begin
  sorry
end

end find_s_l608_608517


namespace range_h_l608_608221

def h (t : ℝ) : ℝ := (t^2 + (1/2) * t) / (t^2 + 2)

theorem range_h : set.range h = {1/4} :=
by
  -- Proof can be added here
  sorry

end range_h_l608_608221


namespace attractions_order_count_l608_608344

theorem attractions_order_count (n : ℕ) (h : n = 5) : nat.factorial n = 120 :=
by {
  rw h,
  simp,
  exact rfl,
}

end attractions_order_count_l608_608344


namespace arithmetic_geometric_sequence_formula_l608_608771

theorem arithmetic_geometric_sequence_formula :
  ∃ (a d : ℝ), (3 * a = 6) ∧
  ((5 - d) * (15 + d) = 64) ∧
  (∀ (n : ℕ), n ≥ 3 → (∃ (b_n : ℝ), b_n = 2 ^ (n - 1))) :=
by
  sorry

end arithmetic_geometric_sequence_formula_l608_608771


namespace decimal_to_fraction_denominator_l608_608857

theorem decimal_to_fraction_denominator : 
  ∀ (d : ℝ), d = 0.34 → (∃ (n m : ℤ), (n : ℝ) / (m : ℝ) = d ∧ m = 100) :=
by
  intro d h
  use [34, 100]
  split
  · simp [h, Rat.cast_div, Rat.cast_bit1, Rat.cast_bit0, Rat.cast_one, Rat.cast_zero]
  · rfl
  sorry

end decimal_to_fraction_denominator_l608_608857


namespace number_of_toys_l608_608088

-- Definitions based on conditions
def selling_price : ℝ := 18900
def cost_price_per_toy : ℝ := 900
def gain_per_toy : ℝ := 3 * cost_price_per_toy

-- The number of toys sold
noncomputable def number_of_toys_sold (SP CP gain : ℝ) : ℝ :=
  (SP - gain) / CP

-- The theorem statement to prove
theorem number_of_toys (SP CP gain : ℝ) : number_of_toys_sold SP CP gain = 18 :=
by
  have h1: SP = 18900 := by sorry
  have h2: CP = 900 := by sorry
  have h3: gain = 3 * CP := by sorry
  -- Further steps to establish the proof
  sorry

end number_of_toys_l608_608088


namespace cat_monitor_area_l608_608161

-- Define the floor setup and conditions
def tile_side_length := 1
def total_tiles := 80
def monitored_area := 66.875 -- in percent

-- Define the percentages 
def monitored_percentage := (107 : ℚ) / 160 * 100 -- result given in percent

-- Lean 4 statement to prove that the cat monitors the given percentage of the floor
theorem cat_monitor_area : monitored_area = monitored_percentage :=
by
  sorry

end cat_monitor_area_l608_608161


namespace xyz_divides_xyz_squared_l608_608350

theorem xyz_divides_xyz_squared (x y z p : ℕ) (hxyz : x < y ∧ y < z ∧ z < p) (hp : Nat.Prime p) (hx3 : x^3 ≡ y^3 [MOD p])
    (hy3 : y^3 ≡ z^3 [MOD p]) (hz3 : z^3 ≡ x^3 [MOD p]) : (x + y + z) ∣ (x^2 + y^2 + z^2) :=
by
  sorry

end xyz_divides_xyz_squared_l608_608350


namespace years_since_marriage_l608_608400

theorem years_since_marriage (x : ℕ) (ave_age_husband_wife_at_marriage : ℕ)
  (total_family_age_now : ℕ) (child_age : ℕ) (family_members : ℕ) :
  ave_age_husband_wife_at_marriage = 23 →
  total_family_age_now = 19 →
  child_age = 1 →
  family_members = 3 →
  (46 + 2 * x) + child_age = 57 →
  x = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end years_since_marriage_l608_608400


namespace women_in_luxury_suites_count_l608_608721

noncomputable def passengers : ℕ := 300
noncomputable def percentage_women : ℝ := 70 / 100
noncomputable def percentage_luxury : ℝ := 15 / 100

noncomputable def women_on_ship : ℝ := passengers * percentage_women
noncomputable def women_in_luxury_suites : ℝ := women_on_ship * percentage_luxury

theorem women_in_luxury_suites_count : 
  round women_in_luxury_suites = 32 :=
by sorry

end women_in_luxury_suites_count_l608_608721


namespace three_digit_reverse_sum_to_1777_l608_608069

theorem three_digit_reverse_sum_to_1777 :
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ 101 * (a + c) + 20 * b = 1777 :=
by
  sorry

end three_digit_reverse_sum_to_1777_l608_608069


namespace correct_statements_l608_608359

-- Definitions from the conditions
noncomputable def U (n : ℕ) := {1..n}
def f_A (A : Set ℕ) (x : ℕ) : ℕ := if x ∈ A then 1 else 0

-- In the given conditions
variables (A B : Set ℕ) (U : Set ℕ)
variable (n : ℕ)
hypothesis (U_def : U = {1..n})
hypothesis (A_subset_U : A ⊆ U)
hypothesis (B_subset_U : B ⊆ U)

-- Statement we need to prove
theorem correct_statements (h1 : ∀ x, f_A A x ≤ f_A B x) (h2 : ∀ x ∈ U, f_A (A ∪ B) x = f_A A x + f_A B x) : 
  (A ⊆ B) ∧ (A ∩ B = ∅) :=
sorry

end correct_statements_l608_608359


namespace handrail_approx_length_l608_608862

noncomputable def handrail_length (rise : ℝ) (turn_deg : ℝ) (radius : ℝ) : ℝ :=
  let theta := turn_deg * Real.pi / 180
  let arc_length := theta * radius
  let d := Real.sqrt (rise ^ 2 + arc_length ^ 2)
  d

theorem handrail_approx_length :
  (handrail_length 15 180 3) ≈ 17.7 :=
by
  sorry

end handrail_approx_length_l608_608862


namespace new_person_weight_l608_608452

-- The conditions from part (a)
variables (average_increase: ℝ) (num_people: ℕ) (weight_lost_person: ℝ)
variables (total_increase: ℝ) (new_weight: ℝ)

-- Assigning the given conditions
axiom h1 : average_increase = 2.5
axiom h2 : num_people = 8
axiom h3 : weight_lost_person = 45
axiom h4 : total_increase = num_people * average_increase
axiom h5 : new_weight = weight_lost_person + total_increase

-- The proof goal: proving that the new person's weight is 65 kg
theorem new_person_weight : new_weight = 65 :=
by
  -- Proof steps go here
  sorry

end new_person_weight_l608_608452


namespace uniquely_determine_T_l608_608231

theorem uniquely_determine_T'_n (b e : ℤ) (S' T' : ℕ → ℤ)
  (hb : ∀ n, S' n = n * (2 * b + (n - 1) * e) / 2)
  (ht : ∀ n, T' n = n * (n + 1) * (3 * b + (n - 1) * e) / 6)
  (h3028 : S' 3028 = 3028 * (b + 1514 * e)) :
  T' 4543 = (4543 * (4543 + 1) * (3 * b + 4542 * e)) / 6 :=
by
  sorry

end uniquely_determine_T_l608_608231


namespace circumcenter_on_AK_l608_608830

variable {α β γ : Real}
variable (A B C L H K O : Type)
variable [Triangle ABC] (circumcenter : Triangle ABC → Point O)
variable [AngleBisector A B C L]

theorem circumcenter_on_AK
  (h₁ : AL_is_angle_bisector ABC L)
  (h₂ : Height_from_B_on_AL B A L H)
  (h₃ : K_on_circumcircle_ABL B A L K)
  : Lies_on_line (circumcenter ABC) A K :=
sorry

end circumcenter_on_AK_l608_608830


namespace black_queen_awake_at_10_l608_608034

-- Define the logical context
def king_awake_at_10 (king_asleep : Prop) : Prop :=
  king_asleep -> false

def king_asleep_at_10 (king_asleep : Prop) : Prop :=
  king_asleep

def queen_awake_at_10 (queen_asleep : Prop) : Prop :=
  queen_asleep -> false

-- Define the main theorem
theorem black_queen_awake_at_10 
  (king_asleep : Prop)
  (queen_asleep : Prop)
  (king_belief : king_asleep ↔ (king_asleep ∧ queen_asleep)) :
  queen_awake_at_10 queen_asleep :=
by
  -- Proof is omitted
  sorry

end black_queen_awake_at_10_l608_608034


namespace arithmetic_sequence_middle_term_l608_608024

theorem arithmetic_sequence_middle_term :
  ∃ y : ℤ, 2^3 = 8 → 2^5 = 32 → y = (8 + 32) / 2 := 
begin
  sorry
end

end arithmetic_sequence_middle_term_l608_608024


namespace extreme_value_at_1_over_a_l608_608745

variable (a b : ℝ)

def func (x : ℝ) : ℝ :=
  a * x^3 + b * x

theorem extreme_value_at_1_over_a (h : ∃ x : ℝ, x = 1 / a ∧ (deriv (func a b) x = 0)) :
  a * b = -3 :=
begin
  by_cases ha : a = 0,
  { exfalso,
    rcases h with ⟨x, hx1, hx2⟩,
    rw [hx1, ha] at hx2,
    simp at hx2,
    exact hx2 },
  rcases h with ⟨x, hx1, hx2⟩,
  rw [hx1] at hx2,
  have h_derivative : deriv (func a b) (1 / a) = 3 * a * (1 / a)^2 + b,
  { rw func,
    simp,
    ring },
  rw h_derivative at hx2,
  have : 3 * a * (1 / a)^2 + b = 0,
  { exact hx2 },
  simp at this,
  have : 3 / a + b = 0,
  { exact this },
  linarith,
end

end extreme_value_at_1_over_a_l608_608745


namespace evaluate_f_2023_over_2_l608_608858

def f : ℝ → ℝ := sorry
axiom h1 : ∀ x : ℝ, f(-x) = -f(x)   -- f is an odd function
axiom h2 : ∀ x : ℝ, f(2-x) = f(x)   -- f(2-x) = f(x)
axiom h3 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f(x) = Real.exp x - 1 -- f(x) = e^x - 1 for x in (0, 1]

theorem evaluate_f_2023_over_2 : f (2023 / 2) = 1 - Real.sqrt Math.exp := 
by sorry

end evaluate_f_2023_over_2_l608_608858


namespace time_to_cross_bridge_l608_608982

/-- Define the length of the train in meters. -/
def length_of_train : ℝ := 250

/-- Define the speed of the train in kilometers per hour. -/
def speed_of_train_kmph : ℝ := 85

/-- Define the speed of the train in meters per second. -/
def speed_of_train_mps : ℝ := speed_of_train_kmph * 1000 / 3600

/-- Define the length of the bridge in meters. -/
def length_of_bridge : ℝ := 350

/-- Define the total distance to be covered by the train in meters. -/
def total_distance : ℝ := length_of_train + length_of_bridge

/-- Define the time taken for the train to completely cross the bridge in seconds. -/
def time_taken : ℝ := total_distance / speed_of_train_mps

/-- The theorem to be proven: the time taken is approximately 25.41 seconds. -/
theorem time_to_cross_bridge : abs (time_taken - 25.41) < 0.01 := 
by
  /- Proof is omitted -/
  sorry

end time_to_cross_bridge_l608_608982


namespace probability_different_from_half_l608_608203

noncomputable def prob_different_color (n : ℕ) (balls : ℕ) : ℚ :=
if balls = n then 0 else 1 / 2 ^ n

theorem probability_different_from_half 
  (n : ℕ) : 
  n = 8 → 
  (∀ (i : ℕ), i < n → (∃ (color : bool), (∑ set_of (λi : ℕ, balls [i] = color)) = n/2)) → 
  prob_different_color n 4 = 35 / 128 :=
begin
  intros h₁ h₂,
  unfold prob_different_color,
  sorry
end

end probability_different_from_half_l608_608203


namespace determine_x_l608_608894

theorem determine_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7.5 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end determine_x_l608_608894


namespace ratio_of_bike_to_tractor_speed_l608_608401

theorem ratio_of_bike_to_tractor_speed (d_tr: ℝ) (t_tr: ℝ) (d_car: ℝ) (t_car: ℝ) (k: ℝ) (β: ℝ) 
  (h1: d_tr / t_tr = 25) 
  (h2: d_car / t_car = 90)
  (h3: 90 = 9 / 5 * β)
: β / (d_tr / t_tr) = 2 := 
by
  sorry

end ratio_of_bike_to_tractor_speed_l608_608401


namespace sum_sigma_div_pi_l608_608556

-- Define sigma as the sum of the elements in the set S
def sigma (S : Finset ℕ) : ℕ := S.sum id

-- Define pi as the product of the elements in the set S
def pi (S : Finset ℕ) : ℕ := S.prod id

-- Define the statement to prove
theorem sum_sigma_div_pi (n : ℕ) :
  (Finset.powerset (Finset.range (n + 1))).sum (λ S, if S.nonempty then (sigma S) / (pi S : ℚ) else 0) =
  (n^2 + 2 * n) - (Finset.range (n + 1)).sum (λ k, 1 / (k + 1 : ℚ)) * (n + 1) :=
sorry

end sum_sigma_div_pi_l608_608556


namespace a_2n_perfect_square_l608_608367

-- Define the sequence a_n following the described recurrence relation.
def a (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 4
  else a (n-1) + a (n-3) + a (n-4)

-- Define the main theorem to prove
theorem a_2n_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k * k := by
  sorry

end a_2n_perfect_square_l608_608367


namespace placement_ways_l608_608311

theorem placement_ways (rows cols crosses : ℕ) (h1 : rows = 3) (h2 : cols = 4) (h3 : crosses = 4)
  (condition : ∀ r : Fin rows, ∃ c : Fin cols, r < rows ∧ c < cols) : 
  (∃ n, n = (3 * 6 * 2) → n = 36) :=
by 
  -- Proof placeholder
  sorry

end placement_ways_l608_608311


namespace distinct_ways_to_construct_cube_l608_608480

theorem distinct_ways_to_construct_cube (white_cubes red_cubes : ℕ) (total_cubes : ℕ) 
  (h_total : total_cubes = 8) (h_white : white_cubes = 5) (h_red : red_cubes = 3)
  (h_distinct : DistinctRotationalSymmetry white_cubes red_cubes total_cubes) : 
  number_of_distinct_constructions white_cubes red_cubes total_cubes = 7 / 3 := 
  sorry

end distinct_ways_to_construct_cube_l608_608480


namespace two_p_plus_q_l608_608625

theorem two_p_plus_q (p q : ℚ) (h : p / q = 6 / 7) : 2 * p + q = 19 / 7 * q :=
by {
  sorry
}

end two_p_plus_q_l608_608625


namespace x1_x2_in_M_l608_608974

-- Definitions of the set M and the condition x ∈ M
def M : Set ℕ := { x | ∃ a b : ℤ, x = a^2 + b^2 }

-- Statement of the problem
theorem x1_x2_in_M (x1 x2 : ℕ) (h1 : x1 ∈ M) (h2 : x2 ∈ M) : (x1 * x2) ∈ M :=
sorry

end x1_x2_in_M_l608_608974


namespace value_range_of_quadratic_l608_608423

theorem value_range_of_quadratic : 
  ∀ x ∈ (set.Ioo 2 5 ∪ {5} : set ℝ), 
  let f := λ x : ℝ, x^2 - 6 * x + 7 
  in ∃ y, y = f x ∧ y ∈ set.Icc (-2 : ℝ) (2 : ℝ) :=
by 
  sorry

end value_range_of_quadratic_l608_608423


namespace f_increasing_interval_l608_608412

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x^2 + x - 6)

theorem f_increasing_interval : ∀ x y : ℝ, 2 < x → x < y → f(x) < f(y) := 
by {
  sorry
}

end f_increasing_interval_l608_608412


namespace solve_equation_passes_through_origin_derivative_at_0_derivative_at_pi_over_2_l608_608394

noncomputable def y (x : ℝ) : ℝ := (1 - cos x) / (sin x) - sin x

theorem solve_equation (x : ℝ) (k : ℤ) :
  (1 / sin x - (cos x / sin x) - sin x = 0) ↔ (x = (2 * k + 1) * (π / 2) ∨ x = 2 * k * π) :=
begin
  sorry
end

theorem passes_through_origin : y 0 = 0 :=
begin
  sorry
end

theorem derivative_at_0 : deriv y 0 = -1 / 2 :=
begin
  sorry
end

theorem derivative_at_pi_over_2 : deriv y (π / 2) = 1 :=
begin
  sorry
end

end solve_equation_passes_through_origin_derivative_at_0_derivative_at_pi_over_2_l608_608394


namespace sum_of_first_10_terms_l608_608270

variable {a : ℕ → ℝ}

-- Define the conditions of the arithmetic sequence
def condition1 := a 1 + a 5 = 6
def condition2 := a 2 + a 14 = 26

-- Define the statement to prove
theorem sum_of_first_10_terms (h1 : condition1) (h2 : condition2) :  
  (∑ i in Finset.range 10, a (i + 1)) = 80 := by
  sorry

end sum_of_first_10_terms_l608_608270


namespace probability_not_collected_3_cards_in_4_purchases_distribution_X_expectation_X_l608_608065

-- Definitions for the problem conditions
def num_colors : ℕ := 3
def full_purchases (n : ℕ) := fin n → fin num_colors

/-- 
  Part 1:
  Prove the probability that a customer, after making full purchases 4 times, 
  still has not collected 3 cards of the same color is 2/3. 
--/
theorem probability_not_collected_3_cards_in_4_purchases :
  let scenario1 := 3 * combinatorial.choose 4 2 * 2
  let scenario2 := combinatorial.choose 3 2 * combinatorial.choose 4 2
  (scenario1 + scenario2) / (num_colors ^ 4) = 2 / 3 := by sorry

/-- 
  Part 2:
  Let X be the number of times a customer makes full purchases before collecting exactly 3 cards of the same color. 
  Prove the distribution and expectation of X.
--/

-- Defining the probabilities
def P_X (n : ℕ) : ℚ :=
  match n with
  | 3 := 1 / 9
  | 4 := 2 / 9
  | 5 := 8 / 27
  | 6 := 20 / 81
  | 7 := 10 / 81
  | _ := 0

-- Expected value of X
def E_X : ℚ :=
  sum (λ n, n * P_X n) (finset.range 8) -- since X ranges from 3 to 7

-- Proving the distribution
theorem distribution_X :
  (∀ n, n ≥ 3 → P_X n = 1 / 9 ∨ P_X n = 2 / 9 ∨ P_X n = 8 / 27 ∨ P_X n = 20 / 81 ∨ P_X n = 10 / 81) := by sorry

-- Proving the expectation
theorem expectation_X :
  E_X = 409 / 81 := by sorry

end probability_not_collected_3_cards_in_4_purchases_distribution_X_expectation_X_l608_608065


namespace calculate_expression_l608_608173

theorem calculate_expression : 2^3 * 2^3 + 2^3 = 72 := by
  sorry

end calculate_expression_l608_608173


namespace geometric_sequence_seventh_term_l608_608407

theorem geometric_sequence_seventh_term (a r : ℝ) (ha : 0 < a) (hr : 0 < r) 
  (h4 : a * r^3 = 16) (h10 : a * r^9 = 2) : 
  a * r^6 = 2 :=
by
  sorry

end geometric_sequence_seventh_term_l608_608407


namespace ocean_depth_of_mountain_l608_608479

noncomputable def cone_volume (r h : ℝ) : ℝ :=
  (1/3) * Real.pi * r^2 * h

theorem ocean_depth_of_mountain 
  (r h : ℝ) (volume_above_fraction : ℝ) (height_fraction : ℝ)
  (hr : r = 3000) (hh : h = 10000) 
  (hvaf : volume_above_fraction = 1 / 10)
  (hhf : height_fraction = (hvaf)^(1 / 3)) :
  (h - h * height_fraction) = 5360 :=
by
  sorry

end ocean_depth_of_mountain_l608_608479


namespace solve_system_of_inequalities_l608_608395

theorem solve_system_of_inequalities (x : ℝ) :
  log (3 / 5) ((x^2 + x - 6) / (x^2 - 4)) < 1 ∧
  sqrt (5 - x^2) > x - 1 →
  -2 < x ∧ x < 2 :=
sorry

end solve_system_of_inequalities_l608_608395


namespace orthogonal_vectors_l608_608698

variables (a b : ℝ^3)
variables (h_ne_zero_a : a ≠ 0) (h_ne_zero_b : b ≠ 0)
variables (h_eq_len : ‖a + b‖ = ‖a - b‖)

theorem orthogonal_vectors (a b : ℝ^3) (h_ne_zero_a : a ≠ 0) (h_ne_zero_b : b ≠ 0)
    (h_eq_len : ‖a + b‖ = ‖a - b‖) : a ⬝ b = 0 :=
sorry

end orthogonal_vectors_l608_608698


namespace stratified_sampling_grade10_sampled_count_l608_608475

def total_students : ℕ := 2000
def grade10_students : ℕ := 600
def grade11_students : ℕ := 680
def grade12_students : ℕ := 720
def total_sampled_students : ℕ := 50

theorem stratified_sampling_grade10_sampled_count :
  15 = (total_sampled_students * grade10_students / total_students) :=
by sorry

end stratified_sampling_grade10_sampled_count_l608_608475


namespace number_of_even_functions_l608_608601

noncomputable def isEvenFunction (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def f1 : ℝ → ℝ := λ x, (sin x)^2
def f2 : ℝ → ℝ := λ x, x^3 + x
def f3 : ℝ → ℝ := λ x, -cos x
def f4 : ℝ → ℝ := λ x, abs (x^5)

theorem number_of_even_functions : 
  let functions := [f1, f2, f3, f4] in
  (functions.filter isEvenFunction).length = 3 :=
by sorry

end number_of_even_functions_l608_608601


namespace common_ratio_of_geo_seq_l608_608465

variable {a : ℕ → ℝ} (q : ℝ)

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem common_ratio_of_geo_seq :
  (∀ n, 0 < a n) →
  geometric_sequence a q →
  a 6 = a 5 + 2 * a 4 →
  q = 2 :=
by
  intros
  sorry

end common_ratio_of_geo_seq_l608_608465


namespace cyclic_points_l608_608801

-- Definitions for trapezoid, circles, reflections, and cyclic points
variables {A B C D A_1 B_1 A_2 B_2 : Point}
variables {Ω ω : Circle}
variables {M_CA M_CB : Point}  -- Midpoints of segments CA and CB

-- Axioms (conditions from the problem)
axiom trapezoid_inscribed : IsTrapezoid ABCD AB CD → CircleInscribed ABCD Ω
axiom ω_passes_through_C_and_D : ω.Through C → ω.Through D
axiom ω_intersects_CA_at_A1 : ω.Intersects CA A_1
axiom ω_intersects_CB_at_B1 : ω.Intersects CB B_1
axiom A2_reflection : IsReflection A_1 M_CA A_2
axiom B2_reflection : IsReflection B_1 M_CB B_2

-- The theorem to be proved
theorem cyclic_points {ABCD : IsTrapezoid ABCD AB CD} {A B C D A_1 B_1 A_2 B_2 : Point} {Ω ω : Circle} :
  IsTrapezoid ABCD AB CD →
  CircleInscribed ABCD Ω →
  ω.Through C →
  ω.Through D →
  ω.Intersects CA A_1 →
  ω.Intersects CB B_1 →
  IsReflection A_1 M_CA A_2 →
  IsReflection B_1 M_CB B_2 →
  CyclicPoints A B A_2 B_2 :=
begin
  intros,
  sorry,  -- Proof to be completed
end

end cyclic_points_l608_608801


namespace larger_number_is_correct_l608_608825

theorem larger_number_is_correct : ∃ L : ℝ, ∃ S : ℝ, S = 48 ∧ (L - S = (1 : ℝ) / (3 : ℝ) * L) ∧ L = 72 :=
by
  sorry

end larger_number_is_correct_l608_608825


namespace longest_diagonal_length_l608_608861

-- Define the conditions
variables {a b : ℝ} (h_area : 135 = 1/2 * a * b) (h_ratio : a / b = 5 / 3)

-- Define the target to prove
theorem longest_diagonal_length (a b : ℝ) (h_area : 135 = 1/2 * a * b) (h_ratio : a / b = 5 / 3) :
    a = 15 * Real.sqrt 2 :=
sorry

end longest_diagonal_length_l608_608861


namespace antonov_candy_packs_l608_608512

theorem antonov_candy_packs (bought_candies : ℕ) (cando_per_pack : ℕ) (gave_to_sister : ℕ) (h_bought : bought_candies = 60) (h_pack : cando_per_pack = 20) (h_gave : gave_to_sister = 20) :
  (bought_candies - gave_to_sister) / cando_per_pack = 2 :=
by
  rw [h_bought, h_pack, h_gave]
  norm_num
  sorry

end antonov_candy_packs_l608_608512


namespace triangle_angle_sum_33_75_l608_608440

theorem triangle_angle_sum_33_75 (x : ℝ) 
  (h₁ : 45 + 3 * x + x = 180) : 
  x = 33.75 :=
  sorry

end triangle_angle_sum_33_75_l608_608440


namespace distinct_rows_col_removal_l608_608323

theorem distinct_rows_col_removal (n : ℕ) (M : matrix (fin n) (fin n) ℕ) :
  (∀ i j : fin n, i ≠ j → M i ≠ M j) → 
  ∃ k : fin n, ∀ i j : fin n, i ≠ j → (M i).erase k ≠ (M j).erase k := 
sorry

end distinct_rows_col_removal_l608_608323


namespace area_comparison_l608_608881

theorem area_comparison:
  let side_length := 3 in
  let area_hexagon := 
    let a3 := Real.sqrt 3 * 1.5
    let h3 := 2 * 1.5
    π * ((h3^2) - (a3^2)) in
  let area_octagon :=
    let a4 := 2.414 * 1.5
    let h4 := 2.613 * 1.5
    π * ((h4^2) - (a4^2)) in
  area_hexagon = area_octagon :=
by
  sorry

end area_comparison_l608_608881


namespace december_sales_fraction_l608_608040

variable (A : ℝ)

-- Define the total sales for January through November
def total_sales_jan_to_nov := 11 * A

-- Define the sales total for December, which is given as 5 times the average monthly sales from January to November
def sales_dec := 5 * A

-- Define the total sales for the year as the sum of January-November sales and December sales
def total_sales_year := total_sales_jan_to_nov + sales_dec

-- We need to prove that the fraction of the December sales to the total annual sales is 5/16
theorem december_sales_fraction : sales_dec / total_sales_year = 5 / 16 := by
  sorry

end december_sales_fraction_l608_608040


namespace moses_more_than_esther_l608_608799

noncomputable theory

def total_amount : ℝ := 50
def moses_share_percentage : ℝ := 0.40
def moses_share : ℝ := moses_share_percentage * total_amount
def remainder : ℝ := total_amount - moses_share
def esther_share : ℝ := remainder / 2

theorem moses_more_than_esther : moses_share - esther_share = 5 :=
by
  -- Proof goes here
  sorry

end moses_more_than_esther_l608_608799


namespace cond_prob_B_given_A_l608_608015

-- Definitions based on the conditions
def eventA := {n : ℕ | n > 4 ∧ n ≤ 6}
def eventB := {k : ℕ × ℕ | (k.1 + k.2) = 7}

-- Probability of event A
def probA := (2 : ℚ) / 6

-- Joint probability of events A and B
def probAB := (1 : ℚ) / (6 * 6)

-- Conditional probability P(B|A)
def cond_prob := probAB / probA

-- The final statement to prove
theorem cond_prob_B_given_A : cond_prob = 1 / 6 := by
  sorry

end cond_prob_B_given_A_l608_608015


namespace arithmetic_sqrt_9_l608_608729

theorem arithmetic_sqrt_9 : ∃ x : ℝ, x^2 = 9 ∧ x = 3 :=
by
  use 3
  split
  · norm_num
    norm_num
  · norm_num

end arithmetic_sqrt_9_l608_608729


namespace vector_expression_l608_608628

variables (a b c : ℝ × ℝ)
variables (m n : ℝ)

noncomputable def vec_a : ℝ × ℝ := (1, 1)
noncomputable def vec_b : ℝ × ℝ := (1, -1)
noncomputable def vec_c : ℝ × ℝ := (-1, 2)

/-- Prove that vector c can be expressed in terms of vectors a and b --/
theorem vector_expression : 
  vec_c = m • vec_a + n • vec_b → (m = 1/2 ∧ n = -3/2) :=
sorry

end vector_expression_l608_608628


namespace fixed_point_of_function_l608_608272

theorem fixed_point_of_function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : 
  ∃ P : ℝ × ℝ, P = (1, 1) ∧ ∀ x : ℝ, (x = 1 → a^(x-1) = 1) :=
by
  sorry

end fixed_point_of_function_l608_608272


namespace largest_five_digit_congruent_to_18_mod_25_l608_608808

theorem largest_five_digit_congruent_to_18_mod_25 : 
  ∃ (x : ℕ), x < 100000 ∧ 10000 ≤ x ∧ x % 25 = 18 ∧ x = 99993 :=
by
  sorry

end largest_five_digit_congruent_to_18_mod_25_l608_608808


namespace new_area_shortening_other_side_l608_608382

-- Define the dimensions of the original card
def original_length : ℕ := 5
def original_width : ℕ := 7

-- Define the shortened length and the resulting area after shortening one side by 2 inches
def shortened_length_1 := original_length - 2
def new_area_1 : ℕ := shortened_length_1 * original_width
def condition_1 : Prop := new_area_1 = 21

-- Prove that shortening the width by 2 inches results in an area of 25 square inches
theorem new_area_shortening_other_side : condition_1 → (original_length * (original_width - 2) = 25) :=
by
  intro h
  sorry

end new_area_shortening_other_side_l608_608382


namespace speed_of_A_l608_608136

theorem speed_of_A 
  (V_B : ℝ)
  (speed_of_B : V_B = 7.555555555555555)
  (time_A : ℝ)
  (time_B : ℝ)
  (time_A_eq : time_A = 2.3)
  (time_B_eq : time_B = 1.8) :
  ∃ V_A : ℝ, V_A * time_A = V_B * time_B ∧ V_A = 5.9130434782608695 := 
by
  use (V_B * time_B / time_A),
  split,
  { norm_num,
    rw [time_A_eq, time_B_eq],
    rw [mul_div_cancel' _ (by norm_num) (by norm_num)] },
  { rw [speed_of_B],
    norm_num }

end speed_of_A_l608_608136


namespace flour_needed_for_cookies_l608_608130

variable (cookies_flour_ratio : ℝ) (bulk_increase : ℝ) (num_cookies : ℕ)

-- Initialize conditions
def cookies_flour_ratio := 1.5 / 20
def bulk_increase := 0.1
def num_cookies := 100

-- Goal: Prove the amount of flour needed
theorem flour_needed_for_cookies (h : num_cookies = 100 ∧ cookies_flour_ratio = 1.5 / 20 ∧ bulk_increase = 0.1):
  let needed_flour := num_cookies * cookies_flour_ratio * (1 + bulk_increase)
  needed_flour = 8.25 :=
  by
    sorry

end flour_needed_for_cookies_l608_608130


namespace swimming_speed_eq_6_l608_608489

theorem swimming_speed_eq_6 :
  ∀ (v s : ℝ) (t : ℝ), s = 2 ∧ (v + s) * t = (v - s) * (2 * t) → v = 6 :=
by
  intros v s t h
  rcases h with ⟨hs, heq⟩
  have ht : t ≠ 0 := sorry -- We can assert t ≠ 0 since it's a swimming speed problem 
  rw [mul_comm, mul_assoc, ←mul_two, ←heq] at hs
  linarith

end swimming_speed_eq_6_l608_608489


namespace statements_are_correct_l608_608281

variable (m : ℝ)

def vector_a : ℝ × ℝ := (1, -2)
def vector_b : ℝ × ℝ := (-1, m)

def vector_c : ℝ × ℝ := (1 / Real.sqrt 5, -2 / Real.sqrt 5)

theorem statements_are_correct :
  (m = 1 → (Real.sqrt ((vector_a.1 - vector_b.1) ^ 2 + (vector_a.2 - vector_b.2) ^ 2) = Real.sqrt 13)) ∧
  ((vector_a.1 / vector_b.1 = vector_a.2 / vector_b.2) → (m = 2)) ∧
  (vector_c = (1 / Real.sqrt 5, -2 / Real.sqrt 5)) :=
begin
  sorry
end

end statements_are_correct_l608_608281


namespace number_of_zeros_of_F_l608_608596

def f (x : ℝ) : ℝ := if x ≤ 1 then (2 ^ x + 2) / 2 else |Real.log2(x - 1)|

def F (x : ℝ) : ℝ := f (f x) - 2 * (f x) - (3 / 2)

theorem number_of_zeros_of_F : ∃! (n : ℕ), n = 4 ∧ ∀ x, F x = 0 → x = n :=
sorry

end number_of_zeros_of_F_l608_608596


namespace find_angle_of_slope_l608_608936

-- Define the points A and B
def A : ℝ × ℝ := (3, 7)
def B : ℝ × ℝ := (2, 8)

-- Define the slope function
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the angle function
def angle (k : ℝ) : ℝ := Real.arctan k

-- Define the main theorem
theorem find_angle_of_slope : angle (slope A B) = Real.pi * 3 / 4 :=
by 
  sorry

end find_angle_of_slope_l608_608936


namespace minimum_m_l608_608586

/-
  Given that for all 2 ≤ x ≤ 3, 3 ≤ y ≤ 6, the inequality mx^2 - xy + y^2 ≥ 0 always holds,
  prove that the minimum value of the real number m is 0.
-/
theorem minimum_m (m : ℝ) :
  (∀ x y : ℝ, 2 ≤ x ∧ x ≤ 3 → 3 ≤ y ∧ y ≤ 6 → m * x^2 - x * y + y^2 ≥ 0) → m = 0 :=
sorry -- proof to be provided

end minimum_m_l608_608586


namespace f_2023_pi_over_3_eq_4_l608_608763

noncomputable def f : ℕ → ℝ → ℝ
| 0, x => 2 * Real.cos x
| (n + 1), x => 4 / (2 - f n x)

theorem f_2023_pi_over_3_eq_4 : f 2023 (Real.pi / 3) = 4 := 
  sorry

end f_2023_pi_over_3_eq_4_l608_608763


namespace stock_percentage_l608_608749

theorem stock_percentage 
  (income : ℝ) (investment : ℝ) (brokerage_percent : ℝ) (market_value : ℝ) : 
  income = 756 → investment = 9000 → brokerage_percent = 0.0025 → market_value = 124.75 →
  let brokerage_fee := brokerage_percent * investment,
      net_investment := investment - brokerage_fee,
      dividend_yield := (income / net_investment) * 100
  in dividend_yield = 8.42 :=
by 
  intros; 
  let brokerage_fee := brokerage_percent * investment,
      net_investment := investment - brokerage_fee,
      dividend_yield := (income / net_investment) * 100;
  show dividend_yield = 8.42 from hf sorry

end stock_percentage_l608_608749


namespace circumcenter_lies_on_ak_l608_608835

noncomputable def triangle_circumcenter_lies_on_ak
  {α β γ : ℝ}
  (A B C L H K O : Type*)
  [triangle A B C]
  [angle_bisector A L]
  [height_from B H]
  [circumcircle_of_triangle A (B ∧ L) K]
  [circumcenter_of_triangle A B C O]
  : Prop :=
  lies_on_line O (line_through A K)

-- We'll add the assumptions as hypotheses to the lemma
theorem circumcenter_lies_on_ak 
  {α β γ : ℝ} {A B C L H K O : Type*}
  [triangle A B C]
  [angle_bisector A L]
  [height_from B H]
  [circumcircle_of_triangle A (B ∧ L) K]
  [circumcenter_of_triangle A B C O]
  : lies_on_line O (line_through A K) :=
sorry

end circumcenter_lies_on_ak_l608_608835


namespace dart_board_probability_l608_608481

variable {s : ℝ} (hexagon_area : ℝ := (3 * Real.sqrt 3) / 2 * s^2) (center_hexagon_area : ℝ := (3 * Real.sqrt 3) / 8 * s^2)

theorem dart_board_probability (s : ℝ) (P : ℝ) (h : P = center_hexagon_area / hexagon_area) :
  P = 1 / 4 :=
by
  sorry

end dart_board_probability_l608_608481


namespace emily_small_gardens_l608_608895

theorem emily_small_gardens :
  ∀ (total_seeds seeds_in_big_garden seeds_per_type num_types : ℕ), 
  total_seeds = 125 → seeds_in_big_garden = 45 → seeds_per_type = 4 → num_types = 5 →
  (total_seeds - seeds_in_big_garden) / (seeds_per_type * num_types) = 4 := 
by
  intros total_seeds seeds_in_big_garden seeds_per_type num_types
  intros h_total_seeds h_seeds_in_big_garden h_seeds_per_type h_num_types
  rw [h_total_seeds, h_seeds_in_big_garden, h_seeds_per_type, h_num_types]
  sorry

end emily_small_gardens_l608_608895


namespace probability_three_heads_one_tail_l608_608992

theorem probability_three_heads_one_tail :
  let p : ℕ → ℕ → ℚ := λ n k, ↑(Nat.choose n k) * (1 / 2)^n
  in p 4 3 = 1 / 4 :=
by
  sorry

end probability_three_heads_one_tail_l608_608992


namespace sequence_sum_l608_608656

theorem sequence_sum :
  (∑ n in Finset.range 10, (-1)^(n+1) * (n+1)) = 5 := 
sorry

end sequence_sum_l608_608656


namespace red_balls_unchanged_l608_608995

-- Definitions: 
def initial_red_balls : ℕ := 3
def initial_blue_balls : ℕ := 2
def initial_yellow_balls : ℕ := 5

def remove_blue_ball (blue_balls : ℕ) : ℕ :=
  if blue_balls > 0 then blue_balls - 1 else blue_balls

-- Condition after one blue ball is removed
def blue_balls_after_removal := remove_blue_ball initial_blue_balls

-- Prove that the number of red balls remain unchanged
theorem red_balls_unchanged : initial_red_balls = 3 :=
by
  sorry

end red_balls_unchanged_l608_608995


namespace orbit_count_inequality_l608_608365

-- Define the set S and the bijections f_1, ..., f_k
variables (n k : ℕ)
def S := fin n

variables (f : (fin n) → (fin n))
variable f := list (S → S)

-- Orbit definition
def orbit (f : S → S) (x : S) : set S :=
  {y | ∃ i : ℕ, f^[i] x = y}

-- Number of distinct orbits of a bijection
def c (f : S → S) : ℕ :=
  finset.card (finset.powerset (finset.univ.image (orbit f)))

-- Composed function of k bijections
def composed_function (f₁ f₂ : S → S) := f₁ ∘ f₂

-- Statement of the theorem
theorem orbit_count_inequality (k : ℕ) (f : fin n → fin n) 
  (f_1 : S → S) (f_2 : S → S) ... (f_k : S → S) :
  ∀ (f : list (S → S)), c f_1 + c f_2 + ... + c f_k ≤ n * (k - 1) + c (foldr composed_function id f) :=
sorry

end orbit_count_inequality_l608_608365


namespace probability_of_x_plus_y_less_than_4_l608_608124

-- Define the square and the probability that x + y < 4 within this square.
theorem probability_of_x_plus_y_less_than_4 : 
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  (desired_area / square_area = 7 / 9) :=
by
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  show (desired_area / square_area = 7 / 9)
  sorry

end probability_of_x_plus_y_less_than_4_l608_608124


namespace limit_nested_square_roots_l608_608229

theorem limit_nested_square_roots :
  (∀ n : ℕ, ∃ f : ℕ → ℝ, f n = 2 - (List.repeat (λ x, sqrt (2 + x)) n 0))
  → tendsto (λ n, 2^n * sqrt (2 - iter n (λ x, sqrt (2 + x)) 0)) at_top (𝓝 π) :=
sorry

end limit_nested_square_roots_l608_608229


namespace domain_lg_function_l608_608742

theorem domain_lg_function (x : ℝ) : (1 + x > 0 ∧ x - 1 > 0) ↔ (1 < x) :=
by
  sorry

end domain_lg_function_l608_608742


namespace player_B_wins_with_31_stones_player_A_wins_with_100_stones_l608_608486

theorem player_B_wins_with_31_stones : ∃ (has_winning_strategy : ℕ → bool) (n : ℕ), n = 31 ∧ has_winning_strategy n = false := by
  -- Proof omitted
  sorry

theorem player_A_wins_with_100_stones : ∃ (has_winning_strategy : ℕ → bool) (n : ℕ), n = 100 ∧ has_winning_strategy n = true := by
  -- Proof omitted
  sorry

end player_B_wins_with_31_stones_player_A_wins_with_100_stones_l608_608486


namespace Mike_divided_laptops_into_rows_l608_608789

noncomputable def num_laptops : ℕ := 44
noncomputable def leftover_laptops : ℕ := 4
noncomputable def total_laptops_divided : ℕ := num_laptops - leftover_laptops
noncomputable def rows : ℕ := 10

theorem Mike_divided_laptops_into_rows :
  total_laptops_divided = 40 ∧ total_laptops_divided % rows = 0 :=
begin
  have h1 : total_laptops_divided = 40, by sorry,
  have h2 : total_laptops_divided % rows = 0, by sorry,
  exact ⟨h1, h2⟩,
end

end Mike_divided_laptops_into_rows_l608_608789


namespace prop2_prop4_l608_608405

theorem prop2 (a b : Line) (α : Plane) 
  (h1 : Perpendicular a α) 
  (h2 : Perpendicular b α) : Parallel a b := 
sorry

theorem prop4 (α β : Plane) (c : Line) 
  (h1 : Perpendicular c α) 
  (h2 : Perpendicular c β) : Parallel α β :=
sorry

end prop2_prop4_l608_608405


namespace determine_a_minus_b_l608_608287

noncomputable theory

open Polynomial

def isFactor (p q : Polynomial ℝ) : Prop :=
  ∃ r : Polynomial ℝ, p = q * r

theorem determine_a_minus_b
  (p : Polynomial ℝ)
  (h : p = x^2 + a*x*y + b*y^2 - 5*x + y + 6)
  (hf : isFactor (x + y - 2) p) :
  a - b = 1 :=
sorry

end determine_a_minus_b_l608_608287


namespace restore_axes_and_unit_length_l608_608707

theorem restore_axes_and_unit_length (parabola : ℝ → ℝ) (h : ∀ x, parabola x = x^2) : 
  ∃ (axes : (ℝ × ℝ) → bool) (unit_length : ℝ), 
  coordinate_axes_and_unit_length_reconstructed parabola axes unit_length :=
by sorry

end restore_axes_and_unit_length_l608_608707


namespace arithmetic_sequence_length_l608_608985

theorem arithmetic_sequence_length :
  ∀ (a₁ d an : ℤ), a₁ = -5 → d = 3 → an = 40 → (∃ n : ℕ, an = a₁ + (n - 1) * d ∧ n = 16) :=
by
  intros a₁ d an h₁ hd han
  sorry

end arithmetic_sequence_length_l608_608985


namespace problem_correct_propositions_l608_608962

theorem problem_correct_propositions :
  (¬(∀ (x y : ℝ) (P F1 F2 : ℝ) (h : P = |PF1| + |PF2| := 2*4), |PF1| = 3 → |PF2| = 1)) ∧
  (∀ (y x : ℝ) (dist : ℝ) (h1 : C : (∃ a b : ℝ, y^2/9 - x^2/16 = 1 ∧ a = 3 ∧ b = 4),
    ∀ (d : ℝ), d = |12| / |3*3 + 4*4| → d = 12/5)) ∧
  (∀ (C1 C2 : ℝ) (h2 : circles : (∃ x y r, [x^2 + y^2 + 2x = 0, x^2 + y^2 + 2y - 1 = 0]),
    ∀ (dist : ℝ) (rad1 rad2 : ℝ), 
      [dist = dist_between_centers, rad1 = 1, rad2 = √2] →
      dist < rad1 + rad2 ∧ dist > rad1 - rad2) ∧
  (∀ (a : ℝ) (h3 : lines : ([a^2x - y + 6 = 0, 4x - (a - 3)y + 9 = 0]), 
    ∀ (sol : a = -1 ∨ a = 2), −a = 3 ∧ a ≠ 2) :=
sorry

end problem_correct_propositions_l608_608962


namespace num_possible_values_m_l608_608757

theorem num_possible_values_m :
  {m n : ℕ // 0 < m ∧ 0 < n ∧ 20 * m + 18 * n = 2018}.card = 12 :=
sorry

end num_possible_values_m_l608_608757


namespace exists_x1_x2_l608_608682

noncomputable def f (a x : ℝ) := a * x + Real.log x

theorem exists_x1_x2 (a : ℝ) (h : a < 0) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ f a x1 ≥ f a x2 :=
by
  sorry

end exists_x1_x2_l608_608682


namespace initial_blue_balls_l608_608004

-- Define the problem conditions
variable (R B : ℕ) -- Number of red balls and blue balls originally in the box.

-- Condition 1: Blue balls are 17 more than red balls
axiom h1 : B = R + 17

-- Condition 2: Ball addition and removal scenario
noncomputable def total_balls_after_changes : ℕ :=
  (B + 57) + (R + 18) - 44

-- Condition 3: Total balls after all changes is 502
axiom h2 : total_balls_after_changes R B = 502

-- We need to prove the initial number of blue balls
theorem initial_blue_balls : B = 244 :=
by
  sorry

end initial_blue_balls_l608_608004


namespace range_of_a_l608_608297

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x0 : ℝ, 2^x0 - 2 ≤ a^2 - 3 * a) ↔ (1 ≤ a ∧ a ≤ 2) := 
sorry

end range_of_a_l608_608297


namespace sqrt_of_9_eq_3_l608_608732

theorem sqrt_of_9_eq_3 : Real.sqrt 9 = 3 := 
by 
  sorry

end sqrt_of_9_eq_3_l608_608732


namespace destination_assignment_l608_608398

-- Define the set of destinations and departments
def destinations : Finset String := {"Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Chengdu", "Shandong"}
def departments : Finset String := {"A", "B", "C", "D"}

-- Define the proof that the total number of ways to assign at least three different destinations
theorem destination_assignment :
  let num_ways := (Finset.choose 4 2) * (Nat.perm 6 3) + (Nat.perm 6 4) in
  num_ways = 1080 :=
by
  sorry

end destination_assignment_l608_608398


namespace exists_equilateral_triangle_of_min_size_l608_608212

noncomputable def min_points_for_equilateral_triangle (S : Finset (Point ℝ)) (colors : Point ℝ → Color)
  (valid_colors : ∀ p, colors p ∈ {red, yellow, blue})
  (all_colors_present : ∀ c ∈ {red, yellow, blue}, ∃ p ∈ S, colors p = c)
  (contains_equilateral_triangle : ∀ T ⊆ S, 
    T.card = 3 → ∃ (a b c : Point ℝ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ is_obtuse_triangle a b c ∧ colors a = colors b ∧ colors b = colors c) : Nat := 
13

theorem exists_equilateral_triangle_of_min_size
  (S : Finset (Point ℝ)) (colors : Point ℝ → Color)
  (valid_colors : ∀ p, colors p ∈ {red, yellow, blue})
  (all_colors_present : ∀ c ∈ {red, yellow, blue}, ∃ p ∈ S, colors p = c)
  (n : ℕ) (h : n ≥ 13)
  (hn : S.card ≥ n) : 
  ∃ T ⊆ S, T.card = 3 ∧ ∃ (a b c : Point ℝ), a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ is_obtuse_triangle a b c ∧ colors a = colors b ∧ colors b = colors c :=
sorry

end exists_equilateral_triangle_of_min_size_l608_608212


namespace fraction_unoccupied_cone_volume_l608_608132

theorem fraction_unoccupied_cone_volume (r : ℝ) (h1 : r > 0) :
  let V_cone := (1 / 3) * π * (2 * (sqrt 2) * r) ^ 2 * (8 * r)
  let V_large_sphere := (4 / 3) * π * (2 * r) ^ 3
  let V_small_sphere := (4 / 3) * π * r ^ 3
  let V_two_spheres := V_large_sphere + V_small_sphere
  let V_unoccupied := V_cone - V_two_spheres
  V_unoccupied / V_cone = 7 / 16 :=
by
  sorry

end fraction_unoccupied_cone_volume_l608_608132


namespace min_value_frac_l608_608747

theorem min_value_frac (a m n : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : 0 < m) (h4 : 0 < n) (h5 : f(x) = a^(x-1) - 2) (h6 : mx - ny - 1 = 0) : 
  ∃ (m n : ℝ), (m + n = 1) → (frac_value = 3 + 2 * Real.sqrt 2)
  where 
    f(x) := a^(x-1) - 2
    frac_value := 1/m + 2/n :=
sorry

end min_value_frac_l608_608747


namespace valid_numbers_A_l608_608201

theorem valid_numbers_A (A : ℕ) (digits : ℕ → ℕ) (n : ℕ)
  (h1 : ∀ i, i < n - 1 → digits (i + 1) = digits i + 1)
  (h2 : A = ∑ i in finset.range n, (digits i) * 10 ^ i)
  (h3 : ∀ x, ∃ k, 10^k > x → (digits (n-1) <= 9))
  (h4 : 9 * A + n = (∑ j in finset.range (n+1), 10^j) - 1)
  : A = 1 ∨ A = 12 ∨ A = 123 ∨ A = 1234 ∨ A = 12345 ∨ A = 123456 ∨ A = 1234567 ∨ A = 12345678 ∨ A = 123456789 :=
begin
  sorry
end

end valid_numbers_A_l608_608201


namespace smallest_number_of_coins_correct_l608_608178

noncomputable def smallest_number_of_coins : ℕ :=
  2 ^ 16

theorem smallest_number_of_coins_correct :
  ∃ n, (∃ (Y : ℕ → Prop), (∀ (y : ℕ), Y y → 2 ≤ y ∧ y < n) ∧ ∃ t, ∀ k, P k ↔ t ∈ P k) ∧ n = 65536 :=
  by
  use 65536
  sorry

end smallest_number_of_coins_correct_l608_608178


namespace find_side_length_l608_608631

theorem find_side_length
  (A : ℝ) (b : ℝ) (c : ℝ) (area : ℝ)
  (hA : A = π / 3)
  (hb : b = 4)
  (harea : area = 2 * Real.sqrt 3)
  (harea_def : area = 1 / 2 * b * c * Real.sin A) :
  ∃ a : ℝ, a = 2 * Real.sqrt 3 :=
by
  have h1 : c = 2 :=
    calc
      area = 1 / 2 * b * c * Real.sin A : by rw [harea_def]
      ... = 1 / 2 * 4 * c * (Real.sqrt 3 / 2) : by simp [hA, hb, Real.sin_pi_div_three]
      ... = 2 * c * (Real.sqrt 3 / 2) : by ring
      ... = c * Real.sqrt 3 : by ring
  
  use Real.sqrt (b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A)
  rw [h1, hA, hb, Real.cos_pi_div_three]
  simp
  sorry

end find_side_length_l608_608631


namespace sum_of_squares_remainder_l608_608422

theorem sum_of_squares_remainder (n : ℕ) : 
  ((n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2) % 3 = 2 :=
by
  sorry

end sum_of_squares_remainder_l608_608422


namespace number_multiplied_value_l608_608442

theorem number_multiplied_value (x : ℝ) :
  (4 / 6) * x = 8 → x = 12 :=
by
  sorry

end number_multiplied_value_l608_608442


namespace find_r_l608_608383

theorem find_r (r : ℝ) (cone1_radius cone2_radius cone3_radius : ℝ) (sphere_radius : ℝ)
  (cone_height_eq : cone1_radius = 2 * r ∧ cone2_radius = 3 * r ∧ cone3_radius = 10 * r)
  (sphere_touch : sphere_radius = 2)
  (center_eq_dist : ∀ {P Q : ℝ}, dist P Q = 2 → dist Q r = 2) :
  r = 1 := 
sorry

end find_r_l608_608383


namespace find_time_in_months_l608_608226

variable (P : ℝ) (R : ℝ) (SI : ℝ)

def time_in_months (P R SI : ℝ) : ℝ := (SI * 100 * 12) / (P * R * 100)

theorem find_time_in_months : 
  P = 10000 → R = 0.05 → SI = 500 → time_in_months P R SI = 12 := 
by {
  sorry
}

end find_time_in_months_l608_608226


namespace volume_overlap_rotated_tetrahedron_l608_608243

theorem volume_overlap_rotated_tetrahedron :
  ∃ (V : ℝ), 
  let a : ℝ := 1 in
  regular_tetrahedron ABCD a →
  base_BCD ABCD BCD →
  centroid BCD O →
  rotated_tetrahedron AB OA 90 →
  V = (sqrt 6 - sqrt 2) / 12 :=
sorry

end volume_overlap_rotated_tetrahedron_l608_608243


namespace smallest_x_l608_608029

-- Define 450 and provide its factorization.
def n1 := 450
def n1_factors := 2^1 * 3^2 * 5^2

-- Define 675 and provide its factorization.
def n2 := 675
def n2_factors := 3^3 * 5^2

-- State the theorem that proves the smallest x for the condition
theorem smallest_x (x : ℕ) (hx : 450 * x % 675 = 0) : x = 3 := sorry

end smallest_x_l608_608029


namespace trevor_spends_more_l608_608012

theorem trevor_spends_more (T R Q : ℕ) 
  (hT : T = 80) 
  (hR : R = 2 * Q) 
  (hTotal : 4 * (T + R + Q) = 680) : 
  T = R + 20 :=
by
  sorry

end trevor_spends_more_l608_608012


namespace min_value_x_plus_y_l608_608945

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : (2 * x + Real.sqrt (4 * x^2 + 1)) * (Real.sqrt (y^2 + 4) - 2) ≥ y) : 
  x + y >= 2 := 
by
  sorry

end min_value_x_plus_y_l608_608945


namespace application_count_l608_608497

def universities := Finset.range 7

def A : ℕ := 0
def B : ℕ := 1

def application_methods : Finset (Finset ℕ) :=
  ((universities \ {A}) ∪ (universities \ {B})).powerset.filter (λ s, s.card = 3) ∪
  (universities.powerset.filter (λ s, s.card = 4) \ {universities.powerset.filter (λ s, s.card = 4).find (λ s, s ⊆ {A, B})})

theorem application_count : application_methods.card = 25 := by
  sorry

end application_count_l608_608497


namespace solution_set_of_ineq_l608_608257

-- Define odd function, increasing function, and the conditions given in the problem.
variables {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f(-x) = -f(x)

-- Condition 2: f is increasing in (0, +∞)
def increasing_on_positive (f : ℝ → ℝ) : Prop := ∀ ⦃x y : ℝ⦄, 0 < x → x < y → f(x) < f(y)

-- Condition 3: f(-3) = 0
def f_at_neg3_zero (f : ℝ → ℝ) : Prop := f(-3) = 0

-- The theorem statement
theorem solution_set_of_ineq 
  (hf1 : odd_function f)
  (hf2 : increasing_on_positive f)
  (hf3 : f_at_neg3_zero f) : 
  {x : ℝ | (x-3)*f(x-3) < 0} = {x : ℝ | 0 < x ∧ x < 3} ∪ {x : ℝ | 3 < x ∧ x < 6} :=
sorry

end solution_set_of_ineq_l608_608257


namespace ping_pong_matches_l608_608253

noncomputable def f (n k : ℕ) : ℕ :=
  Nat.ceil ((n : ℚ) / Nat.ceil ((k : ℚ) / 2))

theorem ping_pong_matches (n k : ℕ) (hn_pos : 0 < n) (hk_le : k ≤ 2 * n - 1) :
  f n k = Nat.ceil ((n : ℚ) / Nat.ceil ((k : ℚ) / 2)) :=
by
  sorry

end ping_pong_matches_l608_608253


namespace complex_power_sum_eq_self_l608_608364

theorem complex_power_sum_eq_self (z : ℂ) (h : z^2 + z + 1 = 0) : z^100 + z^101 + z^102 + z^103 = z :=
sorry

end complex_power_sum_eq_self_l608_608364


namespace find_line_eqn_l608_608488

def circle := { (x, y) : ℝ × ℝ | x^2 + y^2 - 2 * x - 2 * y + 1 = 0 }

def is_chord (l : ℝ → ℝ → Prop) :=
  ∃ p1 p2 : ℝ × ℝ, p1 ∈ circle ∧ p2 ∈ circle ∧ l p1.1 p1.2 ∧ l p2.1 p2.2 ∧ (p1 ≠ p2) ∧
  (let (x1, y1) := p1, (x2, y2) := p2 in Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = Real.sqrt 2)

theorem find_line_eqn
  (l : ℝ → ℝ → Prop)
  (h1 : l (-1) (-2))
  (h2 : is_chord l) :
  (∀ x y, l x y ↔ x - y - 1 = 0) ∨ (∀ x y, l x y ↔ 17 / 7 * x - y + 3 / 7 = 0) :=
sorry

end find_line_eqn_l608_608488


namespace average_speed_correct_l608_608342

-- Definitions for the conditions
def distance1 : ℚ := 40
def speed1 : ℚ := 8
def time1 : ℚ := distance1 / speed1

def distance2 : ℚ := 20
def speed2 : ℚ := 40
def time2 : ℚ := distance2 / speed2

def total_distance : ℚ := distance1 + distance2
def total_time : ℚ := time1 + time2

-- Definition of average speed
def average_speed : ℚ := total_distance / total_time

-- Proof statement that needs to be proven
theorem average_speed_correct : average_speed = 120 / 11 :=
by 
  -- The details for the proof will be filled here
  sorry

end average_speed_correct_l608_608342


namespace min_cells_to_mark_chessboard_l608_608026

-- Define the chessboard as an 8x8 grid of cells
def chessboard : Type := Fin 8 × Fin 8

-- Define a function to check adjacency of two cells
def is_adjacent (c1 c2 : chessboard) : Prop :=
  (abs (c1.1 - c2.1) = 1 ∧ c1.2 = c2.2) ∨ (c1.1 = c2.1 ∧ abs (c1.2 - c2.2) = 1)

-- The minimum number of cells to mark so every cell is adjacent to at least one marked cell
def min_marked_cells (n : ℕ) : Prop :=
  n = 20

-- Statement of the problem in Lean
theorem min_cells_to_mark_chessboard : ∃ (positions : chessboard → Prop), (∀ (c : chessboard), ∃ (m : chessboard), positions m ∧ is_adjacent m c) ∧ min_marked_cells (positions.card) :=
sorry

end min_cells_to_mark_chessboard_l608_608026


namespace max_magnitude_vector_sub_l608_608971

open Real

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
sqrt (v.1^2 + v.2^2)

noncomputable def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
(v1.1 - v2.1, v1.2 - v2.2)

theorem max_magnitude_vector_sub (a b : ℝ × ℝ)
  (ha : vector_magnitude a = 2)
  (hb : vector_magnitude b = 1) :
  ∃ θ : ℝ, |vector_magnitude (vector_sub a b)| = 3 :=
by
  use π  -- θ = π to minimize cos θ to be -1
  sorry

end max_magnitude_vector_sub_l608_608971


namespace find_speed_second_train_l608_608140

noncomputable def speed_second_train (length_train1 length_train2 : ℝ) (speed_train1_kmph : ℝ) (time_to_cross : ℝ) : ℝ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let total_distance := length_train1 + length_train2
  let relative_speed_mps := total_distance / time_to_cross
  let speed_train2_mps := speed_train1_mps - relative_speed_mps
  speed_train2_mps * 3600 / 1000

theorem find_speed_second_train :
  speed_second_train 380 540 72 91.9926405887529 = 36 := by
  sorry

end find_speed_second_train_l608_608140


namespace mixed_price_l608_608089

-- Define the conditions
variables (a : ℝ) (p1 p2 : ℝ)
-- Candies price per pound
def price_per_pound_1 := 3
def price_per_pound_2 := 2

-- Total cost constraint
def total_cost_constraint := a * price_per_pound_1 = a * price_per_pound_2

-- Total weight of mixed candies
def total_weight := (a / price_per_pound_1) + (a / price_per_pound_2)

-- Revenue should be the same
def same_revenue (p : ℝ) := p * total_weight a = 2 * a

-- Prove that the mixed price should be 2.4 rubles per pound
theorem mixed_price : total_cost_constraint a → same_revenue a 2.4 :=
by
  intros h1
  sorry

end mixed_price_l608_608089


namespace no_infinite_sequence_of_moves_l608_608869

-- Define the degree of a plate (for simplicity, modeled as a function)
def degree (P : α) (homochromatic_neighbors : list α) : ℕ := homochromatic_neighbors.length

-- Main theorem statement
theorem no_infinite_sequence_of_moves (n : ℕ) (h : n ≥ 2) :
  ¬ ∃ (initial_configuration : list (list bool)), 
      ∀ (move : list bool → option (list bool)), 
        (move initial_configuration).is_some → 
          ∃ (state_sequence : ℕ → list (list bool)), 
            state_sequence 0 = initial_configuration ∧ 
            (∀ i, (move (state_sequence i)).is_some ∧ (state_sequence (i+1) = (move (state_sequence i)).iget)) :=
sorry

end no_infinite_sequence_of_moves_l608_608869


namespace Diana_age_correct_l608_608164

def age_Aunt_Anna := 48
def age_Brianna := age_Aunt_Anna / 2
def age_Caitlin := age_Brianna - 5
def age_Diana := age_Caitlin + 3

theorem Diana_age_correct : age_Diana = 22 := by
  have h1 : age_Brianna = 24 := by
    unfold age_Brianna
    norm_num
  have h2 : age_Caitlin = 19 := by
    unfold age_Caitlin
    rw [h1]
    norm_num
  unfold age_Diana
  rw [h2]
  norm_num
  sorry

end Diana_age_correct_l608_608164


namespace E_eq_F_l608_608976

noncomputable def E : Set ℝ := { x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3) }

noncomputable def F : Set ℝ := { x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6) }

theorem E_eq_F : E = F := 
sorry

end E_eq_F_l608_608976


namespace problem_a4_minusa_inv4_eq_l608_608404

theorem problem_a4_minusa_inv4_eq (a : ℝ) (h : a ≠ 0) : 
  a^4 - a^(-4) = (a - a^(-1)) * (a + a^(-1)) * (a^2 + a^(-2)) :=
by
  sorry

end problem_a4_minusa_inv4_eq_l608_608404


namespace find_result_of_adding_8_l608_608444

theorem find_result_of_adding_8 (x : ℕ) (h : 6 * x = 72) : x + 8 = 20 :=
sorry

end find_result_of_adding_8_l608_608444


namespace total_dividend_l608_608431

/-- Tony's investments in various stocks /--
def investment : (String → ℝ) :=
  λ s, match s with
    | "A" => 2000
    | "B" => 2500
    | "C" => 1500
    | "D" => 2000
    | "E" => 2000
    | _ => 0

/-- Dividend yield percentages for each year and stock /--
def yield : ℕ → (String → ℝ) :=
  λ year s, match (year, s) with
    | (1, "A") => 0.05
    | (1, "B") => 0.03
    | (1, "C") => 0.04
    | (1, "D") => 0.06
    | (1, "E") => 0.02
    | (2, "A") => 0.04
    | (2, "B") => 0.05
    | (2, "C") => 0.06
    | (2, "D") => 0.03
    | (2, "E") => 0.07
    | (3, "A") => 0.03
    | (3, "B") => 0.04
    | (3, "C") => 0.04
    | (3, "D") => 0.05
    | (3, "E") => 0.06
    | _ => 0

/-- Total dividend over three years is Rs. 1,330 /--
theorem total_dividend : 
  (∑ s in [ "A", "B", "C", "D", "E" ], 
    ∑ y in [1, 2, 3], (investment s) * (yield y s)) = 1330 := 
by sorry

end total_dividend_l608_608431


namespace antonov_candy_packs_l608_608511

theorem antonov_candy_packs (bought_candies : ℕ) (cando_per_pack : ℕ) (gave_to_sister : ℕ) (h_bought : bought_candies = 60) (h_pack : cando_per_pack = 20) (h_gave : gave_to_sister = 20) :
  (bought_candies - gave_to_sister) / cando_per_pack = 2 :=
by
  rw [h_bought, h_pack, h_gave]
  norm_num
  sorry

end antonov_candy_packs_l608_608511


namespace find_pure_water_amount_l608_608987

theorem find_pure_water_amount
  (initial_volume : ℝ) (initial_concentration : ℝ) (desired_concentration : ℝ)
  (initial_volume = 50)
  (initial_concentration = 0.26)
  (desired_concentration = 0.10) :
  ∃ (w : ℝ), (initial_concentration * initial_volume) / (initial_volume + w) = desired_concentration ∧ w = 80 := 
by
  let acid_content := 0.26 * 50
  let final_volume := 50 + w
  have h1 : (acid_content / final_volume) = 0.10
  -- solving the equation...
  exact sorry

end find_pure_water_amount_l608_608987


namespace determine_a_from_equation_l608_608998

theorem determine_a_from_equation (a : ℝ) (x : ℝ) (h1 : x = 1) (h2 : a * x + 3 * x = 2) : a = -1 := by
  sorry

end determine_a_from_equation_l608_608998


namespace probability_of_x_plus_y_less_than_4_l608_608122

-- Define the square and the probability that x + y < 4 within this square.
theorem probability_of_x_plus_y_less_than_4 : 
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  (desired_area / square_area = 7 / 9) :=
by
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  show (desired_area / square_area = 7 / 9)
  sorry

end probability_of_x_plus_y_less_than_4_l608_608122


namespace max_value_log_function_l608_608237

theorem max_value_log_function (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1/2) :
  ∃ u : ℝ, (u = Real.logb (1/2) (8*x*y + 4*y^2 + 1)) ∧ (u ≤ 0) :=
sorry

end max_value_log_function_l608_608237


namespace dart_distribution_possible_lists_l608_608165

theorem dart_distribution_possible_lists :
  let darts := 5
  let boards := 4
  let distinct_lists := 7
  (∃ lists : list (list ℕ), lists.length = distinct_lists ∧
    ∀ l ∈ lists, l.length = boards ∧ (l.sum = darts)) := by sorry

end dart_distribution_possible_lists_l608_608165


namespace square_has_largest_area_l608_608427

-- Definitions/conditions given in the problem.
def base_triangle := 10
def height_triangle := 11
def side_square := 8
def diameter_circle := 8
def pi_approx := 3.1

-- Definitions of the areas of the figures.
def area_triangle := (base_triangle * height_triangle) / 2
def area_square := side_square * side_square
def radius_circle := diameter_circle / 2
def area_circle := pi_approx * (radius_circle * radius_circle)

-- Proof statement asserting the square has the largest area.
theorem square_has_largest_area :
  area_square > area_triangle ∧ area_square > area_circle :=
by sorry

end square_has_largest_area_l608_608427


namespace isosceles_triangle_B_C_equal_l608_608013

theorem isosceles_triangle_B_C_equal :
  ∀ (A B C : ℝ),
  B = C ∧ C = 4 * A ∧ A + B + C = 180 → B = 80 :=
by 
  intros A B C h
  cases h with h1 h2
  cases h2 with h_C_4A h_sum_180
  rw [h1, h_C_4A] at h_sum_180
  linarith

end isosceles_triangle_B_C_equal_l608_608013


namespace average_speed_is_72_l608_608059

-- Definitions for the given conditions
def total_distance (s : ℝ) := s
def speed_section1 := 60 -- km/h
def speed_section2 := 120 -- km/h
def speed_section3 := 80 -- km/h

def distance_section1 (s : ℝ) := s / 2
def distance_section2 (s : ℝ) := s / 6
def distance_section3 (s : ℝ) := s / 3

def time_section1 (s : ℝ) := distance_section1 s / speed_section1
def time_section2 (s : ℝ) := distance_section2 s / speed_section2
def time_section3 (s : ℝ) := distance_section3 s / speed_section3

def total_time (s : ℝ) := time_section1 s + time_section2 s + time_section3 s

def average_speed (s : ℝ) := total_distance s / total_time s

-- The theorem we need to prove
theorem average_speed_is_72 (s : ℝ) (h : s > 0) : average_speed s = 72 := by
  sorry

end average_speed_is_72_l608_608059


namespace delta_k_zero_l608_608230

def sequence (n : ℕ) : ℚ := n^4 + 2 * n^2 + 3

def delta1 (u : ℕ → ℚ) (n : ℕ) : ℚ := u (n + 1) - u n

def delta (k : ℕ) (u : ℕ → ℚ) : ℕ → ℚ :=
  match k with
  | 0     => λ n => u n
  | (k+1) => delta1 (delta k u)

theorem delta_k_zero (k : ℕ) (u : ℕ → ℚ) (h : ∀ n, u n = n^4 + 2 * n^2 + 3) : k = 5 → (∀ n, delta k u n = 0) :=
  sorry

end delta_k_zero_l608_608230


namespace student_council_election_l608_608727

theorem student_council_election (total_votes : ℕ) (votes_counted : ℕ) 
  (votes_remaining : ℕ) (votes_A : ℕ) (votes_B : ℕ) (votes_C : ℕ) : 
  total_votes = 1500 → votes_counted = 1000 → votes_remaining = 500 → 
  votes_A = 350 → votes_B = 370 → votes_C = 280 → 
  ∀ v : ℕ, (votes_A + v > votes_B + (votes_remaining - v)) → 261 ≤ v :=
begin
  intros h_total h_counted h_remaining h_A h_B h_C v h_winning,
  sorry
end

end student_council_election_l608_608727


namespace only_selected_A_is_20_l608_608637

def cardinality_A (x : ℕ) : ℕ := x
def cardinality_B (x : ℕ) : ℕ := x + 8
def cardinality_union (x : ℕ) : ℕ := 54
def cardinality_intersection (x : ℕ) : ℕ := 6

theorem only_selected_A_is_20 (x : ℕ) (h_total : cardinality_union x = 54) 
  (h_inter : cardinality_intersection x = 6) (h_B : cardinality_B x = x + 8) :
  cardinality_A x - cardinality_intersection x = 20 :=
by
  sorry

end only_selected_A_is_20_l608_608637


namespace three_digit_sum_reverse_eq_l608_608072

theorem three_digit_sum_reverse_eq :
  ∃ (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9),
    101 * (a + c) + 20 * b = 1777 ∧ (a, b, c) = (8, 7, 9) ∨ (a, b, c) = (9, 7, 8) :=
by
  sorry

end three_digit_sum_reverse_eq_l608_608072


namespace chores_minutes_proof_l608_608209

-- Definitions based on conditions
def minutes_of_cartoon_per_hour := 60
def cartoon_watched_hours := 2
def cartoon_watched_minutes := cartoon_watched_hours * minutes_of_cartoon_per_hour
def ratio_of_cartoon_to_chores := 10 / 8

-- Definition based on the question
def chores_minutes (cartoon_minutes : ℕ) : ℕ := (8 * cartoon_minutes) / 10

theorem chores_minutes_proof : chores_minutes cartoon_watched_minutes = 96 := 
by sorry 

end chores_minutes_proof_l608_608209


namespace find_f_f_1_l608_608964

def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - 4 else 2 * x

theorem find_f_f_1 : f (f 1) = -4 :=
by
  sorry

end find_f_f_1_l608_608964


namespace integer_solutions_for_n_l608_608582

theorem integer_solutions_for_n (i : ℂ) (hi : i^2 = -1) :
  {n : ℤ | (n + i)^5 ∈ ℤ}.finite.count = 2 :=
begin
  sorry
end

end integer_solutions_for_n_l608_608582


namespace smallest_solution_of_equation_l608_608030

theorem smallest_solution_of_equation :
  ∃ x : ℝ, (x^4 - 26 * x^2 + 169 = 0) ∧ x = -Real.sqrt 13 :=
by
  sorry

end smallest_solution_of_equation_l608_608030


namespace distance_AB_l608_608675

-- Definitions for distance and triangles
variables {A B C I O : Type}
variables (d : A → A → ℝ)

-- Here, A, B, C, I, O can be any types, probably points or vertices in a geometric framework
-- Assumptions based on the problem conditions
variables (I_incenter : ∀ {A B C : A}, IsIncenter I A B C)
variables (O_excenter : ∀ {A B C : A}, IsExcenter O A B C)
variables (BI_12 : d B I = 12)
variables (IO_18 : d I O = 18)
variables (BC_15 : d B C = 15)

-- The theorem statement to prove |AB| = 24 under the given conditions
theorem distance_AB : d A B = 24 :=
sorry

end distance_AB_l608_608675


namespace part1_proof_part2_proof_part3_proof_part4_proof_l608_608925

variables {A B C E F E' F' : Type}
variables [Field A] [Field B] [Field C] [Field E] [Field F] [Field E'] [Field F']
variables (triangle_ABC : Prop)
variables (AE_internal_bisector : Prop)
variables (AF_external_bisector : Prop)
variables (intersect_circumcircle_at_E' : Prop)
variables (intersect_circumcircle_at_F' : Prop)

-- Given conditions:
def given_conditions : Prop := 
  triangle_ABC ∧ AE_internal_bisector ∧ AF_external_bisector ∧ intersect_circumcircle_at_E' ∧ intersect_circumcircle_at_F'

-- To prove (1):
theorem part1_proof (h : given_conditions) :
  A * B = E * E' :=
sorry

-- To prove (2):
theorem part2_proof (h : given_conditions) :
  A * B = F * F' :=
sorry

-- To prove (3):
theorem part3_proof (h : given_conditions) :
  E^2 = A * B - E * C :=
sorry

-- To prove (4):
theorem part4_proof (h : given_conditions) :
  F^2 = C * D - A * B :=
sorry

end part1_proof_part2_proof_part3_proof_part4_proof_l608_608925


namespace confirm_genuine_coins_l608_608200

theorem confirm_genuine_coins (coins : Finset ℕ) (h_size : coins.card = 40) (odd_fake : ∃ k : ℕ, k % 2 = 1 ∧ k < 40 ∧ ∀ x ∈ coins, (x = 1 ∨ x = -1)) : 
  ∃ genuine : Finset ℕ, genuine.card = 16 ∧ ∀ x ∈ genuine, x = 1 := 
sorry

end confirm_genuine_coins_l608_608200


namespace range_of_a_l608_608293

theorem range_of_a (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    0 < x₁ ∧
    0 < x₂ ∧
    ln x₁ + 2 * exp(x₁^2) = x₁^3 + (a * x₁) / exp(1) ∧
    ln x₂ + 2 * exp(x₂^2) = x₂^3 + (a * x₂) / exp(1)) ->
  a < exp(3) + 1 :=
sorry

end range_of_a_l608_608293


namespace subsets_with_consecutive_elements_l608_608187

theorem subsets_with_consecutive_elements :
  (Nat.choose 12 4) - (Nat.choose 9 4) = 369 :=
begin
  -- proof goes here
  sorry
end

end subsets_with_consecutive_elements_l608_608187


namespace part_a_part_b_l608_608518

theorem part_a (N : ℕ) (hN : N > 1) :
  ¬ ∃ (f : (Fin N) → Fin N), 
      (∀ i j : (Fin N), i ≠ j → f i ≠ f j) ∧
      (∀ i : (Fin N), f i < N) ∧
      (∀ i j : (Fin N), 
          f i = 0 → f j ≠ (N - 1 : Fin N)) :=
sorry

theorem part_b (N : ℕ) (hN : N > 0) :
  ∃ (f : (Fin N) → Fin N),
    (∀ i j k : (Fin N), i ≠ j ∧ j ≠ k ∧ i ≠ k → f i ≠ f j ∧ f i ≠ f k ∧ f j ≠ f k) :=
sorry

end part_a_part_b_l608_608518


namespace find_angle_C_find_perimeter_l608_608920

open Real

variables (A B C a b c : ℝ)
variables (area : ℝ)

-- Definitions from conditions
def condition_angle_C : Prop := 2 * cos C * (a * cos B + b * cos A) = c
def condition_area : Prop := c = sqrt 7 ∧ area = (3 * sqrt 3) / 2

-- Theorem Ⅰ: Finding angle C
theorem find_angle_C (h : condition_angle_C) : C = π / 3 :=
sorry

-- Theorem Ⅱ: Finding the perimeter of triangle ABC under given conditions
theorem find_perimeter (hC : C = π / 3) (hc : condition_area) : a + b + c = 5 + sqrt 7 :=
sorry

end find_angle_C_find_perimeter_l608_608920


namespace cos_alpha_minus_beta_l608_608171

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : cos α + cos β = -4 / 5) 
  (h2 : sin α + sin β = 1 / 3) : 
cos (α - β) = -28 / 225 :=
by
  -- The proof goes here
  sorry

end cos_alpha_minus_beta_l608_608171


namespace find_length_CB_l608_608305

variable (A B C D E: Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
variable (DE AB: LineSegment A B)
variable (CD DA CE: Real)
variable (CA CB: Real)
variable (h_parallel: DE ∥ AB)
variable (h_CD: CD = 5)
variable (h_DA: DA = 15)
variable (h_CE: CE = 9)

theorem find_length_CB :
  CD = 5 → DA = 15 → CE = 9 → DE ∥ AB → CB = 36 :=
by
  intros
  sorry

end find_length_CB_l608_608305


namespace probability_gcf_is_one_is_11_by_15_l608_608016

open Set

def natural_numbers := {1, 2, 3, 4, 5, 6}

def gcf_is_one (a b : ℕ) : Prop := Nat.gcd a b = 1

def probability_gcf_is_one : ℚ :=
  let total_pairs := (Finset.card (Finset.powersetLen 2 (Finset.univ : Finset (Fin 6))) : ℚ)
  let valid_pairs := Finset.card ((Finset.powersetLen 2 (Finset.univ : Finset (Fin 6))).filter (λ s, gcf_is_one (s.val.nth 0).iget (s.val.nth 1).iget)) 
  valid_pairs / total_pairs

theorem probability_gcf_is_one_is_11_by_15 : probability_gcf_is_one = 11 / 15 := by
  sorry

end probability_gcf_is_one_is_11_by_15_l608_608016


namespace complement_union_l608_608979

open Set

def U := { x : ℤ | x^2 - 5*x - 6 ≤ 0 }
def A := { x : ℤ | x * (2 - x) ≥ 0 }
def B := {1, 2, 3}

theorem complement_union (x : ℤ) : 
  x ∈ U \ (A ∪ B) ↔ x ∈ {-1, 4, 5, 6} := by
  sorry

end complement_union_l608_608979


namespace seats_usually_taken_l608_608777

def total_tables : Nat := 15
def seats_per_table : Nat := 10
def proportion_left_unseated : Rat := 1 / 10
def proportion_taken : Rat := 1 - proportion_left_unseated

theorem seats_usually_taken :
  proportion_taken * (total_tables * seats_per_table) = 135 := by
  sorry

end seats_usually_taken_l608_608777


namespace maximum_area_of_region_l608_608642

theorem maximum_area_of_region 
  (r₁ r₂ r₃ r₄ : ℝ) 
  (h₁ : r₁ = 2) 
  (h₂ : r₂ = 4) 
  (h₃ : r₃ = 6) 
  (h₄ : r₄ = 8) 
  : (∀ (r : ℝ), r ∈ {r₁, r₂, r₃, r₄} → ∃ (A : ℝ) (ℓ: ℝ → Prop), 
  (∀ (x : ℝ), ℓ x ↔ x = A) ∧ (∀ (r : ℝ), r ∈ {r₁, r₂, r₃, r₄} → r * r * π)) → 
  120 * π = 4 * π + 16 * π + 36 * π + 64 * π :=
begin
  sorry
end

end maximum_area_of_region_l608_608642


namespace distinct_prime_factors_count_l608_608984

def is_prime (n : ℕ) : Prop := Nat.Prime n

def prime_factors_of_87 : list ℕ := [3, 29]
def prime_factors_of_89 : list ℕ := [89]
def prime_factors_of_91 : list ℕ := [7, 13]
def prime_factors_of_93 : list ℕ := [3, 31]

theorem distinct_prime_factors_count :
  (prime_factors_of_87 ++ prime_factors_of_89 ++ prime_factors_of_91 ++ prime_factors_of_93).eraseDups.length = 6 := by
sorry

end distinct_prime_factors_count_l608_608984


namespace geometric_sequence_value_l608_608332

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α)
variable (r : α)
variable (a_pos : ∀ n, a n > 0)
variable (h1 : a 1 = 2)
variable (h99 : a 99 = 8)
variable (geom_seq : ∀ n, a (n + 1) = r * a n)

theorem geometric_sequence_value :
  a 20 * a 50 * a 80 = 64 := by
  sorry

end geometric_sequence_value_l608_608332


namespace alcohol_percentage_in_new_mixture_l608_608847

theorem alcohol_percentage_in_new_mixture
  (water_volume : ℝ)
  (solution1_volume : ℝ) (solution1_percentage : ℝ)
  (solution2_volume : ℝ) (solution2_percentage : ℝ)
  (solution3_volume : ℝ) (solution3_percentage : ℝ)
  (final_alcohol_percentage : ℝ) :
  water_volume = 13 →
  solution1_volume = 11 → solution1_percentage = 0.16 →
  solution2_volume = 8 → solution2_percentage = 0.23 →
  solution3_volume = 5 → solution3_percentage = 0.35 →
  final_alcohol_percentage = ((solution1_volume * solution1_percentage +
                               solution2_volume * solution2_percentage +
                               solution3_volume * solution3_percentage) /
                              (water_volume + solution1_volume + solution2_volume + solution3_volume) * 100) →
  final_alcohol_percentage ≈ 14.46 :=
by
  intros
  sorry

end alcohol_percentage_in_new_mixture_l608_608847


namespace cone_volume_is_correct_l608_608592

-- Definitions based on conditions
variables (r l h : ℝ)
variables (surface_area lateral_surface_area volume : ℝ)

-- Conditions
def cone_surface_area : Prop := surface_area = π
def lateral_surface_semicircle : Prop := π * l = 2 * π * r
def cone_volume : Prop := volume = (1/3) * π * r^2 * h

-- Radius and height definitions derived from conditions
def radius_condition : Prop := r = sqrt(3) / 3
def height_condition : Prop := h = sqrt(l^2 - r^2)

-- Putting it all together
theorem cone_volume_is_correct
  (surface_area_cone : cone_surface_area surface_area)
  (lateral_surface_cone : lateral_surface_semicircle lateral_surface_area)
  (radius_cond : radius_condition r)
  (height_cond : height_condition h) :
  cone_volume volume :=
sorry

end cone_volume_is_correct_l608_608592


namespace candy_left_l608_608432

theorem candy_left (total_candies : ℕ) (people : ℕ) (eaten_per_person : ℕ)
  (H1 : total_candies = 120)
  (H2 : people = 3)
  (H3 : eaten_per_person = 6) : total_candies - (people * eaten_per_person) = 102 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end candy_left_l608_608432


namespace who_hits_6_l608_608555

def player := ℕ
def score := ℕ
def region := ℕ

structure Contest :=
(scores: player → score)
(regionScores: list region)

/-- Given the contest and the individual scores, prove that Alice hits the region with 6 points. -/
theorem who_hits_6 (c : Contest) (h_scores : c.scores 0 = 16 ∧ c.scores 1 = 4 ∧ c.scores 2 = 7 ∧ c.scores 3 = 11 ∧ c.scores 4 = 17) :
  (∃ x y: region, x ≠ y ∧ (x + y = 16) ∧ (x = 6 ∨ y = 6) := sorry

end who_hits_6_l608_608555


namespace combined_area_of_removed_triangles_l608_608870

-- Definitions
variable (s r : ℝ)
variable (diagonal : ℝ := 20)
variable (area : ℝ := 1400)

-- Proof goal statement
theorem combined_area_of_removed_triangles 
  (h1 : s - 2 * r ∈ ℝ)
  (h2 : (s - 2 * r) ^ 2 + (s - 2 * r) ^ 2 = diagonal ^ 2)
  (h3 : r = (s - r) / Real.sqrt 2) :
  4 * (1 / 2 * r ^ 2) = area :=
  sorry

end combined_area_of_removed_triangles_l608_608870


namespace adults_had_meal_l608_608084

theorem adults_had_meal 
  (num_meal_adults : ℕ) 
  (num_meal_children: ℕ) 
  (remaining_food_children : ℕ) :
  num_meal_adults = 70 ∧ num_meal_children = 90 ∧ remaining_food_children = 36 →
  ∃ (A : ℕ), A = 42 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  have meal_consume_ratio : num_meal_children / num_meal_adults = 9 / 7 := by sorry
  have initial_food : 90 = num_meal_children := by sorry
  have remaining_food_children : 36 = num_meal_children - (9 / 7) * (70 - num_meal_adults) := by sorry
  have A : ℕ := 42
  use A
  exact eq.refl 42

end adults_had_meal_l608_608084


namespace required_hours_per_week_l608_608286

/-- 
Given:
- planned_hours_per_week: The number of hours planned to work per week initially (25 hours).
- planned_weeks: The number of weeks initially planned to work (10 weeks).
- total_earnings_goal: The total earnings goal (\$2500).
- missed_weeks: The number of weeks missed (1 week).

Prove:
- The number of hours per week (H) needed to work for the remaining weeks (planned_weeks - missed_weeks) to still achieve the total earnings goal is approximately 28 hours.
-/
theorem required_hours_per_week 
  (planned_hours_per_week : ℕ) 
  (planned_weeks : ℕ) 
  (total_earnings_goal : ℝ) 
  (missed_weeks : ℕ) 
  (hourly_rate := total_earnings_goal / (planned_hours_per_week * planned_weeks)) :
  planned_hours_per_week = 25 →
  planned_weeks = 10 →
  total_earnings_goal = 2500 →
  missed_weeks = 1 →
  let remaining_weeks := planned_weeks - missed_weeks in
  let required_weekly_earnings := total_earnings_goal / remaining_weeks in
  let required_hours := required_weekly_earnings / hourly_rate in
  required_hours ≈ 28 := 
by 
  intros _ _ _ _
  let hourly_rate := total_earnings_goal / (planned_hours_per_week * planned_weeks)
  let remaining_weeks := planned_weeks - missed_weeks
  let required_weekly_earnings := total_earnings_goal / remaining_weeks
  let required_hours := required_weekly_earnings / hourly_rate
  sorry

end required_hours_per_week_l608_608286


namespace perp_midpoint_orthocenter_l608_608044

theorem perp_midpoint_orthocenter
    (A B C D M N P Q : Type)
    [IsMidpointOf M A C]
    [IsMidpointOf N B D]
    [IsOrthocenterOf P (LineThrough A B) (LineThrough C D) (LineThrough A D)]
    [IsOrthocenterOf Q (LineThrough A B) (LineThrough C D) (LineThrough B C)]
    : IsPerpendicular (LineThrough M N) (LineThrough P Q) := sorry

end perp_midpoint_orthocenter_l608_608044


namespace complex_number_count_l608_608552

theorem complex_number_count:
  {z : ℂ // |z| = 1 ∧ (| (z^2 / (conj(z))^2) + ((conj(z))^2 / z^2) | = 2)}
  .to_finset.card = 4 :=
by
  sorry

end complex_number_count_l608_608552


namespace smallest_x_satisfies_abs_eq_l608_608554

theorem smallest_x_satisfies_abs_eq (x : ℝ) :
  (|2 * x + 5| = 21) → (x = -13) :=
sorry

end smallest_x_satisfies_abs_eq_l608_608554


namespace determine_M_l608_608197

theorem determine_M :
  (∃ (M : ℕ), 0 < M ∧ 18^2 * 45^2 = 30^2 * M^2) → 18^2 * 45^2 = 30^2 * 81^2 :=
by
  intros h
  cases h with M hM
  cases hM with M_pos hM_eq
  rw hM_eq
  sorry

end determine_M_l608_608197


namespace angle_BAO_calculation_l608_608476

-- Definitions based on conditions
abbreviation angle_ABC : ℝ := 75
abbreviation angle_BCA : ℝ := 70

-- Proof problem statement
theorem angle_BAO_calculation (O A B C : Type)
  (circumscribed : ∀ {X Y : Type}, X ≠ Y → Type)
  (center : O)
  (triangle_ABC : △ ABC)
  (O_circ_center : circumscribed O)
  (angle_ABC_eq : angle_ABC = 75)
  (angle_BCA_eq : angle_BCA = 70) :
  ∠B A O = 17.5 := 
  sorry

end angle_BAO_calculation_l608_608476


namespace tan_theta_parallel_vectors_l608_608562

/-- Given 0 < θ < π / 2, and vectors a and b defined by
    a = (sin(2 * θ), cos(θ))
    b = (cos(θ), 1)
    if a and b are parallel vectors, then tan(θ) = 1/2. -/
theorem tan_theta_parallel_vectors (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2)
    (h_parallel : (sin(2 * θ), cos(θ)) = (cos(θ) * k, 1 * k) for some k : ℝ) :
    Real.tan θ = 1 / 2 :=
sorry

end tan_theta_parallel_vectors_l608_608562


namespace fraction_reducibility_l608_608381

theorem fraction_reducibility (m n : ℕ) (h : Nat.gcd m n = 1) :
  Nat.ValidFraction m n ∧ (Nat.Even n ↔ red_fraction n (2 * m + 3 * n)) :=
by sorry

def red_fraction (a b : ℕ) : Prop :=
  1 < Nat.gcd a b

namespace Nat

def ValidFraction (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

end Nat

end fraction_reducibility_l608_608381


namespace tangent_line_at_x1_range_of_a_l608_608273

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * a * x^2

def f_prime (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * a * x

theorem tangent_line_at_x1 (x : ℝ) (y : ℝ) : 
  ∀ x y, (f 2 1) = f 2 x + f_prime 2 1 * (y - x) :=
by 
  intro x y
  let f_x := f 2 x
  let f'_x := f_prime 2 x
  let f_1 := f 2 1
  let f'_1 := f_prime 2 1 
  sorry

theorem range_of_a (a : ℝ) : 
  ∀ x1 ∈ set.Icc (0:ℝ) 2, ∃ x2 ∈ set.Icc (0:ℝ) 1, f x1 a ≥ f_prime x2 a :=
by 
  intro a x1 hx1
  cases lt_or_le a (2:ℝ)
  case inl ha1 => 
    cases lt_or_eq_of_le ha1
    case inl ha2 => 
      sorry
    case inr ha2_eq =>
      sorry
  case inr ha1 =>
    sorry

end tangent_line_at_x1_range_of_a_l608_608273


namespace lower_right_hand_square_is_one_l608_608795

theorem lower_right_hand_square_is_one (M : Matrix (Fin 4) (Fin 4) ℕ)
  (h1 : M 0 0 = 1)
  (h2 : M 0 3 = 2)
  (h3 : M 1 1 = 2)
  (h4 : M 2 2 = 3)
  (h5 : ∀ i j, M i j ∈ {1, 2, 3, 4})
  (h6 : ∀ i, Finset.card (Finset.filter (fun j => M i j = 1) Finset.univ) = 1)
  (h7 : ∀ i, Finset.card (Finset.filter (fun j => M i j = 2) Finset.univ) = 1)
  (h8 : ∀ i, Finset.card (Finset.filter (fun j => M i j = 3) Finset.univ) = 1)
  (h9 : ∀ i, Finset.card (Finset.filter (fun j => M i j = 4) Finset.univ) = 1)
  (h10 : ∀ j, Finset.card (Finset.filter (fun i => M i j = 1) Finset.univ) = 1)
  (h11 : ∀ j, Finset.card (Finset.filter (fun i => M i j = 2) Finset.univ) = 1)
  (h12 : ∀ j, Finset.card (Finset.filter (fun i => M i j = 3) Finset.univ) = 1)
  (h13 : ∀ j, Finset.card (Finset.filter (fun i => M i j = 4) Finset.univ) = 1) :
  M 3 3 = 1 := 
sorry

end lower_right_hand_square_is_one_l608_608795


namespace gridTransformation_l608_608310

-- Define the initial and target states of the grid
def initialState : List Bool := [true, false, true, false, true]  -- true represents W, false represents B
def targetState : List Bool := [false, true, false, true, false]

-- Define the jump operation (this is a sketch to represent the operation, specifics of flipping and valid moves will be added if required in proof)
def jump (grid : List Bool) (source target : Nat) : List Bool :=
  sorry  -- Here should be the operation that handles jumps and flips correctly

-- The main theorem that we aim to prove
theorem gridTransformation : 
  ∃ seq : List Nat, 
    (seq.length = 6) ∧ 
    (∀ (i : Nat), i < 6 → 
      validJump (initialState) (seq[i]) (seq[i+1])) ∧ 
    (finalState = targetState) where
  validJump := sorry  -- This should define the valid jumping conditions and flipping
  finalState := List.foldl (λ grid move, jump grid (move.fst) (move.snd)) initialState (zip seq (List.tail seq))
  sorry

end gridTransformation_l608_608310


namespace sector_area_calculation_l608_608957

-- Define necessary parameter for the problem
def central_angle : ℝ := 60
def radius : ℝ := 10

-- Theorem to prove the area of the sector
theorem sector_area_calculation (n : ℝ) (R : ℝ) (h1 : n = central_angle) (h2 : R = radius) : 
    (n * real.pi * R^2 / 360) = 50 * real.pi / 3 :=
by
  rw [h1, h2]
  sorry

end sector_area_calculation_l608_608957


namespace diagonal_count_l608_608919

def sides : ℕ × ℕ × ℕ × ℕ := (11, 13, 21, 15)

def valid_diagonal_length (x : ℕ) : Prop :=
  let (AB, BC, CD, DA) := sides in
  x > 6 ∧ x < 24

theorem diagonal_count : ∃ x, valid_diagonal_length x ∧ {x : ℕ | valid_diagonal_length x}.to_finset.card = 17 :=
by
  sorry

end diagonal_count_l608_608919


namespace integer_roots_of_polynomial_l608_608417

theorem integer_roots_of_polynomial (a b c d : ℚ)
  (h1 : Polynomial.has_root (Polynomial.mk [d, c, b, a, 1]) (3 - Real.sqrt 5))
  (h2 : ∃ (x : ℝ), x ≠ 3 - Real.sqrt 5 ∧ x ≠ 3 + Real.sqrt 5 ∧ Polynomial.has_root (Polynomial.mk [d, c, b, a, 1]) x) :
  (∃ (p : ℤ), Polynomial.has_root (Polynomial.mk [d, c, b, a, 1]) (p : ℚ) ∧ (p = -2 ∨ p = -4 ∨ p = -3)) :=
sorry

end integer_roots_of_polynomial_l608_608417


namespace transform_curve_l608_608654

theorem transform_curve (x y x' y' : ℝ)
  (h1 : x' = 2 * x)
  (h2 : y' = 3 * y)
  (h3 : y = (1 / 3) * cos (2 * x)) :
  y' = cos x' :=
by
  -- transform and substitute
  sorry

end transform_curve_l608_608654


namespace probability_x_plus_y_lt_4_l608_608116

open Set

-- Define the square and the line
def square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 }
def line_lt_4 : Set (ℝ × ℝ) := { p | p.1 + p.2 < 4 }

-- The probability to prove
theorem probability_x_plus_y_lt_4 : 
  (volume (square ∩ line_lt_4) / volume square) = 7 / 9 := 
sorry

end probability_x_plus_y_lt_4_l608_608116


namespace evaluate_expression_l608_608542

theorem evaluate_expression (x y : ℕ) (hx : x = 3) (hy : y = 2) : 4 * x^y + 5 * y^x = 76 := by
  sorry

end evaluate_expression_l608_608542


namespace tangent_to_circumcircle_l608_608854

theorem tangent_to_circumcircle
  (O A B K L M : Point)
  (h1 : CircleInscribedInAngle O A B)
  (h2 : OnArc K (SmallerArc A B (Circle O)))
  (h3 : OnLine L (Line OB))
  (h4 : Parallel (Line OA) (Line KL))
  (h5 : OnCircumcircle M (Triangle KLB))
  (h6 : M ≠ K)
  (h7 : OnLine M (Line AK)) :
  Tangent (Line OM) (Circumcircle (Triangle KLB)) := sorry

end tangent_to_circumcircle_l608_608854


namespace probability_of_x_plus_y_less_than_4_l608_608123

-- Define the square and the probability that x + y < 4 within this square.
theorem probability_of_x_plus_y_less_than_4 : 
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  (desired_area / square_area = 7 / 9) :=
by
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  show (desired_area / square_area = 7 / 9)
  sorry

end probability_of_x_plus_y_less_than_4_l608_608123


namespace find_toonies_l608_608154

-- Define the number of coins and their values
variables (L T : ℕ) -- L represents the number of loonies, T represents the number of toonies

-- Define the conditions
def total_coins := L + T = 10
def total_value := 1 * L + 2 * T = 14

-- Define the theorem to be proven
theorem find_toonies (L T : ℕ) (h1 : total_coins L T) (h2 : total_value L T) : T = 4 :=
by
  sorry

end find_toonies_l608_608154


namespace ziggy_rap_requests_l608_608447

variables (total_songs electropop dance rock oldies djs_choice rap : ℕ)

-- Given conditions
axiom total_songs_eq : total_songs = 30
axiom electropop_eq : electropop = total_songs / 2
axiom dance_eq : dance = electropop / 3
axiom rock_eq : rock = 5
axiom oldies_eq : oldies = rock - 3
axiom djs_choice_eq : djs_choice = oldies / 2

-- Proof statement
theorem ziggy_rap_requests : rap = total_songs - electropop - dance - rock - oldies - djs_choice :=
by
  -- Apply the axioms and conditions to prove the resulting rap count
  sorry

end ziggy_rap_requests_l608_608447


namespace range_of_a_l608_608225

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + 3| - |x + 2| ≥ Real.log a / Real.log 2) ↔ (0 < a ∧ a ≤ 2) :=
by
  sorry

end range_of_a_l608_608225


namespace xiao_ming_correct_calculation_l608_608643

theorem xiao_ming_correct_calculation (a : ℚ) (h1 : 37 + 31 * a = 37 + 31 + a) : a = 31 / 30 :=
by
  have h2 : 31 * a = 31 + a := by
    linarith [h1]
  have h3 : 30 * a = 31 := by
    linarith [h2]
  exact (eq_div_iff (by norm_num)).mpr h3

end xiao_ming_correct_calculation_l608_608643


namespace common_chords_intersect_at_one_point_l608_608790

-- Define the points representing the foci
structure Point where
  x : ℝ
  y : ℝ

-- Define the distances from a point to each focus
def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Define the ellipse structure with distances
structure Ellipse where
  focus1 : Point
  focus2 : Point
  sum_dist : ℝ

-- Define the non-collinear condition for the foci
def non_collinear (f1 f2 f3 : Point) : Prop :=
  (f2.x - f1.x) * (f3.y - f1.y) ≠ (f3.x - f1.x) * (f2.y - f1.y)

-- The main theorem statement based on identified conditions
theorem common_chords_intersect_at_one_point
  (F1 F2 F3 : Point)
  (h_non_collinear : non_collinear F1 F2 F3)
  (a b c : ℝ)
  (E1 : Ellipse)
  (E2 : Ellipse)
  (E3 : Ellipse)
  (hE1 : E1.sum_dist = a)
  (hE2 : E2.sum_dist = b)
  (hE3 : E3.sum_dist = c)
  : ∃ P : Point, ∀ (L1 : Ellipse) (L2 : Ellipse), 
  (L1 ≠ L2) → CommonChords L1 L2 P := sorry

-- Define a placeholder CommonChords to abstract the common chords intersection logic
def CommonChords (E1 E2 : Ellipse) (P : Point) : Prop := sorry

end common_chords_intersect_at_one_point_l608_608790


namespace seats_usually_taken_l608_608774

theorem seats_usually_taken:
  let tables := 15 in
  let seats_per_table := 10 in
  let total_seats := tables * seats_per_table in
  let unseated_fraction := 1 / 10 in
  let unseated_seats := total_seats * unseated_fraction in
  let seats_taken := total_seats - unseated_seats in
  seats_taken = 135 :=
by
  sorry

end seats_usually_taken_l608_608774


namespace polynomial_factors_to_f_value_l608_608349

theorem polynomial_factors_to_f_value (a b c d e f : ℝ)
  (x1 x2 x3 x4 x5 x6 x7 x8 : ℝ)
  (h_factor: p(x) = (x - x1) * (x - x2) * (x - x3) * (x - x4) * (x - x5) * (x - x6) * (x - x7) * (x - x8))
  (h_pos: ∀ i, 1 ≤ i ∧ i ≤ 8 → x_i > 0)
  (h_p: p(x) = x^8 - 4*x^7 + 7*x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) :
  f = 1 / 256 := sorry

end polynomial_factors_to_f_value_l608_608349


namespace radius_of_circle_l608_608758

theorem radius_of_circle : 
  ∀ (r : ℝ), 3 * (2 * Real.pi * r) = 2 * Real.pi * r ^ 2 → r = 3 :=
by
  intro r
  intro h
  sorry

end radius_of_circle_l608_608758


namespace problem_de_ef_l608_608659

-- Definitions and assumptions based on the conditions
variables {A B C D E F : Type} (h1 : ∀ p q r s : A, triangle (A B C) h1)
variables {triangle : Type} (h2 : ∀ p q : B, AB = AC = 2)
variables {segment : Type} (h3 : p : C, BC = 1)
variables {D : Type} (h4 : ∃ d : D, d ∈ AB)
variables {E : Type} (h5 : ∃ e : E, e ∈ BC)
variables {F : Type} (h6 : ∃ f : F, f ∈ CA)
variables {segment_de : Type} (h7 : segment_de ≡ AC)
variables {segment_ef : Type} (h8 : segment_ef ≡ AC)
variables {segment_ad : Type} {x : Type} {z : Type} (h9 : AD = x ∧ DE = z)
variables {perimeter_ade : Type} {p : Type} {segment_ade : Type} (h10 : p = x + z + (2 - x))
variables {perimeter_defb : Type} {q : Type} {segment_defb : Type} (h11 : q = z + x + (1 - y) + (2 - x))
variables {perimeter_fcae : Type} {r : Type} {segment_fcae : Type} (h12 : r = y + 2 + (2 - x) + x)

-- Problem statement to be proven
theorem problem_de_ef :
  (2 + z = 4 + y) ∧ (3 - y + z = 4 + y) → 2 + z = 3 - y + z → DE + EF = 2 :=
begin
  assume h,
  sorry, -- proof steps to be completed based on the conditions and properties given
end

end problem_de_ef_l608_608659


namespace no_solution_natural_p_q_r_l608_608158

theorem no_solution_natural_p_q_r :
  ¬ ∃ (p q r : ℕ), 2^p + 5^q = 19^r := sorry

end no_solution_natural_p_q_r_l608_608158


namespace average_first_6_numbers_l608_608734

theorem average_first_6_numbers (A : ℕ) (h1 : (13 * 9) = (6 * A + 45 + 6 * 7)) : A = 5 :=
by 
  -- h1 : 117 = (6 * A + 45 + 42),
  -- solving for the value of A by performing algebraic operations will prove it.
  sorry

end average_first_6_numbers_l608_608734


namespace planar_graph_has_erdos_posa_property_l608_608623

noncomputable def erdos_posa_property
  (H : Type) [planar H] : Prop :=
∃ (f : ℕ → ℕ), ∀ (k : ℕ) (G : Type), 
  (∃ (subgraphs : fin k → G), (∀ i, subgraphs i ≃ H)) ∨
  (∃ (U : finset G), U.card ≤ f k ∧ ∀ (g : G), g ∉ U → ¬ (H ≃ g))

theorem planar_graph_has_erdos_posa_property
  (H : Type) [planar H] : erdos_posa_property H := sorry

end planar_graph_has_erdos_posa_property_l608_608623


namespace total_games_played_l608_608053

theorem total_games_played (teams: ℕ) (games_per_pair: ℕ) 
  (num_teams: teams = 10) (games_each_team_with_others: games_per_pair = 4) : 
  (teams * (teams - 1) / 2) * games_per_pair = 180 := 
by 
  have h0 : teams = 10 := num_teams
  have h1 : games_per_pair = 4 := games_each_team_with_others
  rw [h0, h1]
  -- Computing the pairings and total games
  sorry

end total_games_played_l608_608053


namespace problem_statement_l608_608199

-- Given definitions based on the conditions
def z1 : ℂ := 1 - complex.sqrt 3 * complex.I
def z2 : ℂ := (complex.sqrt 3 + complex.I) ^ 2

-- The theorem we want to prove
theorem problem_statement : z1 / z2 = -1 / 4 - (complex.sqrt 3) / 4 * complex.I :=
by
  -- Placeholder for the proof
  sorry

end problem_statement_l608_608199


namespace symmetric_coordinates_l608_608738

-- Define the original point
structure Point :=
  (x : Int)
  (y : Int)

def symmetric_point (P : Point) : Point :=
  ⟨-P.x, -P.y⟩

theorem symmetric_coordinates :
  symmetric_point ⟨2, -5⟩ = ⟨-2, 5⟩ :=
by
  simp [symmetric_point]
  sorry

end symmetric_coordinates_l608_608738


namespace total_time_to_travel_round_trip_l608_608041
-- Import the necessary libraries

-- Define the properties and conditions
def boat_speed_still_water : ℝ := 16
def stream_speed : ℝ := 2
def distance : ℝ := 6840

-- Define the speeds downstream and upstream
def downstream_speed : ℝ := boat_speed_still_water + stream_speed
def upstream_speed : ℝ := boat_speed_still_water - stream_speed

-- Calculate the times to travel downstream and upstream
def time_downstream : ℝ := distance / downstream_speed
def time_upstream : ℝ := distance / upstream_speed

-- Calculate the total time for the round trip
def total_time : ℝ := time_downstream + time_upstream

-- The statement to prove
theorem total_time_to_travel_round_trip :
  total_time = 868.57 := by
  -- Sorry for skipping the proof
  sorry

end total_time_to_travel_round_trip_l608_608041


namespace probability_x_plus_y_less_than_4_l608_608100

open Set Real

/-- Define the square with vertices (0,0), (0,3), (3,3), and (3,0) -/
def square := set.Icc (0 : ℝ) 3 ×ˢ set.Icc (0 : ℝ) 3

/-- Probability that a randomly chosen point (x, y) in the square satisfies x + y < 4 -/
theorem probability_x_plus_y_less_than_4 : 
  let area_square := 9 in
  let area_excluded_triangle := 2 in
  let relevant_area := area_square - area_excluded_triangle in
  (relevant_area / area_square : ℝ) = 7 / 9 :=
by
{ sorry }

end probability_x_plus_y_less_than_4_l608_608100


namespace exp_minus_exp2_log_pos_l608_608600

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x

-- Define the statement about the inequality to prove
theorem exp_minus_exp2_log_pos (x : ℝ) (h : 0 < x) : ∀ a : ℝ, e^x - e^2 * Real.log x > 0 := by
  sorry

end exp_minus_exp2_log_pos_l608_608600


namespace none_of_these_l608_608695

noncomputable def x (t : ℝ) : ℝ := t ^ (3 / (t - 1))
noncomputable def y (t : ℝ) : ℝ := t ^ ((t + 1) / (t - 1))

theorem none_of_these (t : ℝ) (ht_pos : t > 0) (ht_ne_one : t ≠ 1) :
  ¬ (y t ^ x t = x t ^ y t) ∧ ¬ (x t ^ x t = y t ^ y t) ∧
  ¬ (x t ^ (y t ^ x t) = y t ^ (x t ^ y t)) ∧ ¬ (x t ^ y t = y t ^ x t) :=
sorry

end none_of_these_l608_608695


namespace find_three_digit_numbers_l608_608546

def is_valid_number (N : ℕ) : Prop :=
  ∀ k : ℕ, N^k % 1000 = N

theorem find_three_digit_numbers :
  {N : ℕ // 100 ≤ N ∧ N < 1000 ∧ is_valid_number N} = {376, 625} :=
sorry

end find_three_digit_numbers_l608_608546


namespace find_f_7_l608_608581

-- Define the odd function property
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)

-- Define the periodic property with period 4
def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f(x + p) = f(x)

-- Define the specific function property for x in (0, 2)
def specific_property (f : ℝ → ℝ) : Prop := ∀ x, 0 < x ∧ x < 2 → f(x) = 2 * x * x

-- The main theorem to prove
theorem find_f_7 (f : ℝ → ℝ) 
  (odd_f : odd_function f)
  (periodic_f : periodic_function f 4)
  (specific_f : specific_property f) :
  f 7 = -2 :=
sorry

end find_f_7_l608_608581


namespace flower_bed_planting_methods_l608_608477

/-- A city is planning a flower bed in the central square that is divided into 6 sections.
    Four different colors of flowers (A, B, C, D) are to be planted, with one color per section.
    Adjacent sections cannot have the same color.
    We want to prove that there are 120 different planting methods. -/
theorem flower_bed_planting_methods : 
  ∃ (ways : ℕ), ways = 120 ∧ 
    (∀ (colors : Fin 6 → Fin 4), 
      (∀ i j, (i = j + 1 ∨ i = j - 1) → colors i ≠ colors j) →
        (ways = number_of_planting_methods colors)) := 
sorry

end flower_bed_planting_methods_l608_608477


namespace twenty_percent_greater_than_40_l608_608450

theorem twenty_percent_greater_than_40 (x : ℝ) (h : x = 40 + 0.2 * 40) : x = 48 := by
sorry

end twenty_percent_greater_than_40_l608_608450


namespace peter_invested_for_3_years_l608_608384

-- Definitions of parameters
def P : ℝ := 650
def APeter : ℝ := 815
def ADavid : ℝ := 870
def tDavid : ℝ := 4

-- Simple interest formula for Peter
def simple_interest_peter (r : ℝ) (t : ℝ) : Prop :=
  APeter = P + P * r * t

-- Simple interest formula for David
def simple_interest_david (r : ℝ) : Prop :=
  ADavid = P + P * r * tDavid

-- The main theorem to find out how many years Peter invested his money
theorem peter_invested_for_3_years : ∃ t : ℝ, (∃ r : ℝ, simple_interest_peter r t ∧ simple_interest_david r) ∧ t = 3 :=
by
  sorry

end peter_invested_for_3_years_l608_608384


namespace AIME_2002I_P10_l608_608648

theorem AIME_2002I_P10 :
  ∀ {A B C D E F G : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F] [metric_space G]
  (angle_ABC_right : ∃ (angle:ℝ), cos angle = 0) 
  (D_on_BC : ∃ (d : ℝ), ∀ (point_on_BC : ℝ), point_on_BC = d)
  (AD_bisects_angleCAB : ∃ (angle1 angle2:ℝ), angle1 = angle2)
  (E_on_AB : ∃ (length : ℝ), length = 3)
  (F_on_AC : ∃ (length : ℝ), length = 10)
  (EB : ∃ (length : ℝ), length = 9)
  (FC : ∃ (length : ℝ), length = 27)
  (AB_AC : ∃ (length1 length2 : ℝ), length1 = 12 ∧ length2 = 35),
  let area_DCFG := 154 in
  true :=
by sorry

end AIME_2002I_P10_l608_608648


namespace area_of_ADEC_l608_608316

-- Definitions according to conditions
def A := (0, 0) : ℝ × ℝ
def B := (24, 0) : ℝ × ℝ
def C := (0, 10) : ℝ × ℝ
def D := (12, 0) : ℝ × ℝ  -- midpoint of AB
def E := (12, 10) : ℝ × ℝ -- CE ⊥ AB and E on AB

-- Prove that the area of quadrilateral ADEC is 10√119 - 120
theorem area_of_ADEC : 
  let area_ABC := 1/2 * (24 : ℝ) * (10 : ℝ)
  let area_AEC := 1/2 * (12 : ℝ) * (10 : ℝ)
  area_ABC - area_AEC = 10 * real.sqrt 119 - 120 :=
by
  let area_ABC := 1 / 2 * (24 : ℝ) * (10 : ℝ)
  let area_AEC := 1 / 2 * (12 : ℝ) * (10 : ℝ)
  show area_ABC - area_AEC = 10 * real.sqrt 119 - 120
  sorry

end area_of_ADEC_l608_608316


namespace solve_theta_dual_inequalities_l608_608194

variable {θ : ℝ}

/-- Mathematically equivalent proof problem as a Lean 4 statement. -/
theorem solve_theta_dual_inequalities
  (h1 : ∀ x, x^2 - 4 * real.sqrt 3 * x * real.cos (2 * θ) + 2 < 0 ↔ (x > a ∧ x < b))
  (h2 : ∀ x, 2 * x^2 + 4 * x * real.sin (2 * θ) + 1 < 0 ↔ (x > 1/b ∧ x < 1/a))
  (θ_in_interval : 0 < θ ∧ θ < real.pi) :
  θ = real.pi / 3 ∨ θ = 5 * real.pi / 6 :=
sorry

end solve_theta_dual_inequalities_l608_608194


namespace collinear_M_O_K_l608_608956

variable {A B C O M K : Point}
variable [center_incircle : CenterInscribedCircle ABC O]
variable [on_AC : OnSegment M A C]
variable [on_BC : OnSegment K B C]

variable {AB AO BO : ℝ}
variable {BK : ℝ}
variable {AM : ℝ}

-- Conditions
axiom BK_AB_BO_eq : BK * AB = BO^2
axiom AM_AB_AO_eq : AM * AB = AO^2

-- Goal
theorem collinear_M_O_K 
  (h1 : BK * AB = BO^2)
  (h2 : AM * AB = AO^2) :
  Collinear {M, O, K} :=
  sorry

end collinear_M_O_K_l608_608956


namespace max_integers_greater_than_13_l608_608000

theorem max_integers_greater_than_13 (a b c d e f g : ℤ) 
  (h_sum : a + b + c + d + e + f + g = -1) :
  (∃ n : ℕ, n ≤ 6 ∧ 
    ∃ l_1 l_2 ... l_n > 13,
    ∀ x ∈ {a, b, c, d, e, f, g} \ {l_1, ..., l_n}, x ≤ 13) :=
begin
  sorry
end

end max_integers_greater_than_13_l608_608000


namespace find_square_side_length_l608_608940

noncomputable def square_side_length (a : ℝ) : Prop :=
  let angle_deg := 30
  let a_sqr_minus_1 := Real.sqrt (a ^ 2 - 1)
  let a_sqr_minus_4 := Real.sqrt (a ^ 2 - 4)
  let dihedral_cos := Real.cos (Real.pi / 6)  -- 30 degrees in radians
  let dihedral_sin := Real.sin (Real.pi / 6)
  let area_1 := 0.5 * a_sqr_minus_1 * a_sqr_minus_4 * dihedral_sin
  let area_2 := 0.5 * Real.sqrt (a ^ 4 - 5 * a ^ 2)
  dihedral_cos = (Real.sqrt 3 / 2) -- Using the provided angle
  ∧ dihedral_sin = 0.5
  ∧ area_1 = area_2
  ∧ a = 2 * Real.sqrt 5

-- The theorem stating that the side length of the square is 2\sqrt{5}
theorem find_square_side_length (a : ℝ) (H : square_side_length a) : a = 2 * Real.sqrt 5 := by
  sorry

end find_square_side_length_l608_608940


namespace ellipse_center_sum_l608_608887

theorem ellipse_center_sum : 
  ∀ (h k a b : ℝ), h = -3 → k = 4 → a = 7 → b = 2 → (h + k + a + b) = 10 :=
by
  intros h k a b h_eq k_eq a_eq b_eq
  rw [h_eq, k_eq, a_eq, b_eq]
  norm_num
  exact sorry

end ellipse_center_sum_l608_608887


namespace mary_water_intake_l608_608796

theorem mary_water_intake:
  (daily_water_intake_liter: ℚ) →
  (number_of_glasses: ℕ) →
  (daily_water_intake_liter = 1.5) →
  (number_of_glasses = 6) →
  ∃ (water_per_glass_ml: ℚ), water_per_glass_ml = 250 :=
by
  assume daily_water_intake_liter number_of_glasses
  assume h1 : daily_water_intake_liter = 1.5
  assume h2 : number_of_glasses = 6
  sorry

end mary_water_intake_l608_608796


namespace exterior_angle_regular_octagon_l608_608651

-- Definition of a regular polygon with given number of sides
def regular_polygon (n : ℕ) : Prop :=
  ∀ i j : ℕ, i ≠ j → (interior_angle i = interior_angle j)

-- The interior angle of a regular n-gon
def interior_angle (n : ℕ) : ℝ :=
  (n - 2) * 180 / n

-- The exterior angle of a regular n-gon
def exterior_angle (n : ℕ) : ℝ :=
  180 - interior_angle n

-- Statement to prove: The exterior angle of a regular octagon is 45 degrees
theorem exterior_angle_regular_octagon : exterior_angle 8 = 45 :=
by
  sorry

end exterior_angle_regular_octagon_l608_608651


namespace probability_of_x_plus_y_lt_4_l608_608095

open Classical -- To handle probability and random selection
open Set -- For geometric notions
open Filter -- For measure and integration

noncomputable def probability_condition (x y : ℝ) (h : 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) : Prop := x + y < 4

theorem probability_of_x_plus_y_lt_4 :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) →
                pr (λ (xy : ℝ × ℝ), probability_condition xy.1 xy.2 (and.intro (and.left xy.2) (and.right (and.left xy.2)))) = 7/9 :=
by sorry

end probability_of_x_plus_y_lt_4_l608_608095


namespace mod_remainder_sum_first_105_l608_608028

theorem mod_remainder_sum_first_105 (n : ℕ) (hn : n = 105) : 
  (∑ k in Finset.range (n+1), k) % 5270 = 295 :=
by
  sorry

end mod_remainder_sum_first_105_l608_608028


namespace problem_f_derivative_at_pi_over_4_l608_608371

noncomputable def f (x : ℝ) : ℝ := f' (π / 2) * Real.sin x + Real.cos x

theorem problem_f_derivative_at_pi_over_4 :
  f' (π / 4) = -Real.sqrt 2 := by
  sorry

end problem_f_derivative_at_pi_over_4_l608_608371


namespace mica_total_cost_l608_608379

theorem mica_total_cost :
  let pasta_cost := 2 * 1.5
  let ground_beef_cost := (1 / 4) * 8
  let pasta_sauce_cost := 2 * 2
  let quesadilla_cost := 6
  let total_cost := pasta_cost + ground_beef_cost + pasta_sauce_cost + quesadilla_cost
  in total_cost = 15 := by
  sorry

end mica_total_cost_l608_608379


namespace unique_solution_xy_l608_608544

theorem unique_solution_xy : ∃ x y : ℕ, y^2 + y + 1 = x^2 ∧ x = 1 ∧ y = 0 :=
by
  have h0 : (0 : ℕ) ^ 2 + 0 + 1 = 1 := by norm_num
  have h1 : (1 : ℕ) ^ 2 = 1 := by norm_num
  use 1
  use 0
  exact ⟨h0, rfl, rfl⟩,
sorry

end unique_solution_xy_l608_608544


namespace prob_even_sums_l608_608312

def unique_numbers (n : ℕ) : Prop :=
  ∀ i j : ℕ, i < j → i < n → j < n → (i ≠ j)

def grid_sum_even (m n : ℕ) (f : ℕ → ℕ → ℕ) : Prop :=
  (∀ i, i < m → (∀ j, j < n → f i j < m * n) ∧ (∑ j in Finset.range n, f i j) % 2 = 0) ∧
  (∀ j, j < n → (∀ i, i < m → f i j < m * n) ∧ (∑ i in Finset.range m, f i j) % 2 = 0)

theorem prob_even_sums : 
  ∃ (f : ℕ → ℕ → ℕ), (unique_numbers 16 ∧ grid_sum_even 4 4 f) → 
    ∑ i in Finset.range 16, (f (i / 4) (i % 4)) = 71 / 144 :=
by
  sorry

end prob_even_sums_l608_608312


namespace parabola_focus_condition_l608_608296

theorem parabola_focus_condition (m : ℝ) : (∃ (x y : ℝ), x + y - 2 = 0 ∧ y = (1 / (4 * m))) → m = 1 / 8 :=
by
  sorry

end parabola_focus_condition_l608_608296


namespace angle_AED_measure_l608_608180

-- Definitions of the conditions
def incircle {α : Type*} [linear_ordered_field α] (Γ : circle α) (ABC : triangle α) : Prop :=
-- Definition details of incircle (custom definition)

def circumcircle {α : Type*} [linear_ordered_field α] (Γ : circle α) (DEF : triangle α) : Prop :=
-- Definition details of circumcircle (custom definition)

def on_segment {α : Type*} [linear_ordered_field α] (P : point α) (A B : point α) : Prop :=
-- Definition details of point on line segment (custom definition)

-- Constants for the given conditions
constant A B C D E F : point ℝ
constant Γ : circle ℝ
constant ABC : triangle ℝ
constant DEF : triangle ℝ

axiom incircle_condition : incircle Γ ABC
axiom circumcircle_condition : circumcircle Γ DEF
axiom D_on_BC : on_segment D B C
axiom E_on_AB : on_segment E A B
axiom F_on_AC : on_segment F A C
axiom angle_A : ∠ A B C = 50
axiom angle_B : ∠ B A C = 70

-- The proof problem
theorem angle_AED_measure :
  ∠ A E D = 110 := sorry

end angle_AED_measure_l608_608180


namespace average_is_3_l608_608005

theorem average_is_3 (A B C : ℝ) (h1 : 1501 * C - 3003 * A = 6006)
                              (h2 : 1501 * B + 4504 * A = 7507)
                              (h3 : A + B = 1) :
  (A + B + C) / 3 = 3 :=
by sorry

end average_is_3_l608_608005


namespace probability_three_heads_one_tail_l608_608991

theorem probability_three_heads_one_tail :
  let p : ℕ → ℕ → ℚ := λ n k, ↑(Nat.choose n k) * (1 / 2)^n
  in p 4 3 = 1 / 4 :=
by
  sorry

end probability_three_heads_one_tail_l608_608991


namespace total_dots_not_visible_l608_608428

theorem total_dots_not_visible {dice_faces : Fin 8 → ℕ} (h_dice_faces_def : dice_faces = ![1, 2, 3, 4, 5, 6, 7, 8]) 
                                {visible_faces : Fin 8 → ℕ} (h_visible_faces_def : visible_faces = ![2, 2, 3, 4, 5, 6, 7, 8]) :
  let total_dots_on_all_dice := 3 * (Array.sum (dice_faces.map id)) in
  let total_dots_visible := Array.sum (visible_faces.map id) in
  total_dots_on_all_dice - total_dots_visible = 71 := sorry

end total_dots_not_visible_l608_608428


namespace sequence_repeat_mod_100_l608_608175

def sequence (a : ℕ) : ℕ → ℕ 
| 0       := a
| (k + 1) := if k % 2 = 0 then sequence k + 54 else sequence k + 77

theorem sequence_repeat_mod_100 (a : ℕ) : ∃ i j : ℕ, i ≠ j ∧ (sequence a i) % 100 = (sequence a j) % 100 :=
by
  sorry

end sequence_repeat_mod_100_l608_608175


namespace find_IJ_length_l608_608160

noncomputable def length_of_IJ (A B C D E F G H I J : ℝ × ℝ) : Prop :=
  let AD := dist A D in
  let GH := dist G H in
  let AE := dist A E in
  let ED := dist E D in
  let GF := dist G F in
  let HF := dist H F in
  let EF := dist E F in
  let EH := dist E H in
  let IJ := dist I J in
  AD = 6 ∧ GH = 6 ∧ AE = 3 ∧ ED = 3 ∧ GF = 3 ∧ HF = 6 ∧ EF = 4 ∧ EF.perp GH ∧ EH = 2 * sqrt 13 ∧ IJ = 3.6

theorem find_IJ_length :
  ∃ (A B C D E F G H I J : ℝ × ℝ), length_of_IJ A B C D E F G H I J :=
begin
  sorry,
end

end find_IJ_length_l608_608160


namespace complex_number_identity_l608_608736

theorem complex_number_identity : (1 + Complex.i) ^ 10 / (1 - Complex.i) = -16 + 16 * Complex.i := by
  -- This is where the proof would go
  sorry

end complex_number_identity_l608_608736


namespace probability_x_plus_y_lt_4_l608_608110

theorem probability_x_plus_y_lt_4 :
  let square_vertices := {(0,0), (0,3), (3,3), (3,0)} in
  let point_in_square (x y : ℝ) := 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 in
  let probability (A : ℝ) (B : ℝ) := A / B in
  ∃ (P : ℝ), P = probability 7 9 ∧
             P = (measure (λ (x y : ℝ), point_in_square x y ∧ x + y < 4)) / (measure (λ (x y : ℝ), point_in_square x y)) :=
sorry

end probability_x_plus_y_lt_4_l608_608110


namespace sum_of_divisors_l608_608439

-- Define the relevant conditions
def is_divisor (n d : ℕ) : Prop := d ∣ n

-- Define the number and its prime factorization
def n : ℕ := 77
def p1 : ℕ := 7
def p2 : ℕ := 11
axiom factorization : n = p1 * p2

-- State the main theorem
theorem sum_of_divisors : (1 + p1 + p2 + n) = 96 :=
by {
  -- Verify the assumed factorization
  have h1 : p1 * p2 = 77, from factorization,
  -- Verify the divisors and their sum
  sorry
}

end sum_of_divisors_l608_608439


namespace centroid_of_arc_proof_l608_608505

noncomputable def centroid_of_arc (R α : ℝ) : ℝ :=
  if α = π then (2 * R) / π
  else (2 * R / α) * Real.sin (α / 2)

theorem centroid_of_arc_proof (R α : ℝ) (h1 : R > 0) (h2 : α > 0) :
  centroid_of_arc R α = if α = π then (2 * R) / π else (2 * R / α) * Real.sin (α / 2) := sorry

end centroid_of_arc_proof_l608_608505


namespace no_numbers_relatively_prime_to_18_and_35_l608_608615

-- Define the predicate that checks if a number is in the given range
def in_range (n : ℕ) : Prop := 21 ≤ n ∧ n < 100

-- Define the predicate that a number is relatively prime to a given number
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the main theorem statement
theorem no_numbers_relatively_prime_to_18_and_35 :
  {n : ℕ | in_range n ∧ relatively_prime n 18 ∧ relatively_prime n 35}.card = 0 :=
by
  sorry

end no_numbers_relatively_prime_to_18_and_35_l608_608615


namespace smallest_addition_to_make_multiple_of_5_l608_608811

theorem smallest_addition_to_make_multiple_of_5 : ∃ k : ℕ, k > 0 ∧ (729 + k) % 5 = 0 ∧ k = 1 := sorry

end smallest_addition_to_make_multiple_of_5_l608_608811


namespace find_toonies_l608_608155

-- Define the number of coins and their values
variables (L T : ℕ) -- L represents the number of loonies, T represents the number of toonies

-- Define the conditions
def total_coins := L + T = 10
def total_value := 1 * L + 2 * T = 14

-- Define the theorem to be proven
theorem find_toonies (L T : ℕ) (h1 : total_coins L T) (h2 : total_value L T) : T = 4 :=
by
  sorry

end find_toonies_l608_608155


namespace probability_x_plus_y_lt_4_l608_608114

open Set

-- Define the square and the line
def square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 }
def line_lt_4 : Set (ℝ × ℝ) := { p | p.1 + p.2 < 4 }

-- The probability to prove
theorem probability_x_plus_y_lt_4 : 
  (volume (square ∩ line_lt_4) / volume square) = 7 / 9 := 
sorry

end probability_x_plus_y_lt_4_l608_608114


namespace arithmetic_sqrt_9_l608_608730

theorem arithmetic_sqrt_9 : ∃ x : ℝ, x^2 = 9 ∧ x = 3 :=
by
  use 3
  split
  · norm_num
    norm_num
  · norm_num

end arithmetic_sqrt_9_l608_608730


namespace hidden_dots_are_32_l608_608793

theorem hidden_dots_are_32 
  (visible_faces : List ℕ)
  (h_visible : visible_faces = [1, 2, 3, 4, 4, 5, 6, 6])
  (num_dice : ℕ)
  (h_num_dice : num_dice = 3)
  (faces_per_die : List ℕ)
  (h_faces_per_die : faces_per_die = [1, 2, 3, 4, 5, 6]) :
  63 - visible_faces.sum = 32 := by
  sorry

end hidden_dots_are_32_l608_608793


namespace probability_of_consecutive_cards_is_0_point_4_l608_608787
noncomputable def probability_two_consecutive_cards : ℚ :=
  let total_outcomes := (finset.card (finset.powerset_len 2 {1, 2, 3, 4, 5})) in
  let favorable_outcomes := 4 in
  (favorable_outcomes : ℚ) / total_outcomes

theorem probability_of_consecutive_cards_is_0_point_4 :
  probability_two_consecutive_cards = 0.4 := by
  sorry

end probability_of_consecutive_cards_is_0_point_4_l608_608787


namespace f_ordering_l608_608700

def f : ℝ → ℝ := sorry
def f' : ℝ → ℝ := sorry

variable (a : ℝ)
hypothesis ha : 2 < a ∧ a < 4
hypothesis hf_symm : ∀ x : ℝ, f x = f (4 - x)
hypothesis hf_deriv_pos : ∀ x : ℝ, (x - 2) * f' x > 0

theorem f_ordering (a : ℝ) 
  (ha : 2 < a ∧ a < 4)
  (hf_symm : ∀ x : ℝ, f x = f (4 - x))
  (hf_deriv_pos : ∀ x : ℝ, (x - 2) * f' x > 0) :
  f 2 < f (log 2 a) ∧ f (log 2 a) < f (2 ^ a) :=
sorry

end f_ordering_l608_608700


namespace sin_cos_range_geometric_progression_l608_608269

theorem sin_cos_range_geometric_progression (a b c : ℝ) (A B C : ℝ) 
  (h_progression : b^2 = a * c) 
  (h_cos : b^2 = a^2 + c^2 - 2 * a * c * Math.cos B) :
  1 < Math.sin B + Math.cos B ∧ Math.sin B + Math.cos B ≤ Real.sqrt 2 :=
sorry

end sin_cos_range_geometric_progression_l608_608269


namespace solve_equation_l608_608393

theorem solve_equation :
  {x : ℂ | (x - complex.sqrt 3)^3 + (x - complex.sqrt 3) = 0} = {complex.sqrt 3, complex.sqrt 3 + complex.I, complex.sqrt 3 - complex.I} :=
by
  sorry

end solve_equation_l608_608393


namespace f_f_neg1_l608_608963

def f : ℝ → ℝ :=
λ x, if x ≤ 0 then 1 - 2^x else Real.log x / Real.log 2

theorem f_f_neg1 : f (f (-1)) = -1 :=
by
  -- The proof is omitted
  sorry

end f_f_neg1_l608_608963


namespace sum_of_irreducible_fractions_l608_608366

theorem sum_of_irreducible_fractions (a b : ℕ) (h₁ : 0 < a) (h₂ : a < b) : 
  (∑ (n in finset.filter (λ n, nat.gcd n 7 = 1) (finset.Icc (7*a) (7*b))), n) / 7 = 3 * (b^2 - a^2) :=
by sorry

end sum_of_irreducible_fractions_l608_608366


namespace find_threedigit_number_l608_608076

-- Define the three-digit number and its reverse
def original_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

-- Define the condition of adding the number and its reverse to get 1777
def number_sum_condition (a b c : ℕ) : Prop :=
  original_number a b c + reversed_number a b c = 1777

-- Prove the existence of digits a, b, and c that satisfy the conditions
theorem find_threedigit_number :
  ∃ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  original_number a b c = 859 ∧ 
  reversed_number a b c = 958 ∧ 
  number_sum_condition a b c :=
sorry

end find_threedigit_number_l608_608076


namespace prime_divisor_of_N_l608_608561

theorem prime_divisor_of_N (N : ℕ) (a : ℕ) (m : ℕ) (primes : Fin m → ℕ) 
  (hN : N = 2^a * (List.prod (List.ofFn primes)))
  (hσ : Nat.sigma N = 3 * N)
  (hm : m ≥ 1)
  (pr : ∀ i j, i ≠ j → primes i ≠ primes j)
  (hprimes : ∀ i, Nat.Prime (primes i)) :
  ∃ p : ℕ, Nat.Prime p ∧ Nat.Prime (2^p - 1) ∧ (2^p - 1 ∣ N) :=
by 
  sorry

end prime_divisor_of_N_l608_608561


namespace rotation_test_l608_608714

structure Point (α : Type) :=
  (x : α)
  (y : α)

def rotate_90_clockwise (p : Point ℝ) : Point ℝ :=
  Point.mk p.y (-p.x)

def A : Point ℝ := ⟨2, 3⟩
def B : Point ℝ := ⟨3, -2⟩

theorem rotation_test : rotate_90_clockwise A = B :=
by
  sorry

end rotation_test_l608_608714


namespace count_odd_3_digit_integers_l608_608986

open Nat

/-- Prove that there are no odd positive 3-digit integers which are divisible by 5
    and do not contain the digit 5. -/
theorem count_odd_3_digit_integers : 
  ∃ n : ℕ, (n = 0 ↔ ∀ x, 100 ≤ x ∧ x < 1000 ∧ (x % 2 = 1) ∧ (x % 5 = 0) ∧ ¬('5' ∈ (show String, from toDigits 10 x).data)) :=
by 
  sorry

end count_odd_3_digit_integers_l608_608986


namespace W_divisible_by_11_l608_608526

variables {a b c : ℕ}
theorem W_divisible_by_11 (ha : a < 10) (hb : b < 10) (hc : c < 10) :
  let W := 100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a in
  11 ∣ W :=
by
  sorry

end W_divisible_by_11_l608_608526


namespace decreasing_numbers_count_l608_608142

def is_decreasing (n : ℕ) : Prop :=
  let digits := n.digits 10
  (∀ (i : ℕ), i < digits.length - 1 → digits.get i > digits.get (i + 1))

def within_range (n : ℕ) : Prop := 100 ≤ n ∧ n < 500

theorem decreasing_numbers_count : 
  (∃ (count : ℕ), count = 10 ∧ (∀ n : ℕ, within_range n → is_decreasing n → n ∈ {m | m < 500}.card)) :=
  sorry

end decreasing_numbers_count_l608_608142


namespace three_friends_visit_days_l608_608189

theorem three_friends_visit_days :
  ∃ n : ℕ, (1 ≤ n ∧ n ≤ 365) → (visit Alex n ∧ visit Bella n ∧ visit Carl n ∧ ¬visit Diana n) ∨
            (visit Alex n ∧ visit Bella n ∧ ¬visit Carl n ∧ visit Diana n) ∨
            (visit Alex n ∧ ¬visit Bella n ∧ visit Carl n ∧ visit Diana n) ∨
            (¬visit Alex n ∧ visit Bella n ∧ visit Carl n ∧ visit Diana n) →
            n = 15 :=
by
  -- Proof steps are omitted as requested.
  sorry

-- Definitions for visits based on conditions
def visit (person : String) (day : ℕ) : Prop :=
  match person with
  | "Alex"  => (day % 4 = 0)
  | "Bella" => (day % 6 = 0)
  | "Carl"  => (day % 8 = 0)
  | "Diana" => (day % 9 = 0)
  | _       => false

end three_friends_visit_days_l608_608189


namespace answer_is_Two_l608_608502

def isPerfectSquareForm (p : Polynomial ℤ) : Prop :=
  ∃ a b : ℤ, p = Polynomial.C a * Polynomial.C a + 2 * Polynomial.C a * Polynomial.C b + Polynomial.C b * Polynomial.C b

def expressions : List (Polynomial ℤ) :=
  [Polynomial.X^2 + 2 * Polynomial.X + Polynomial.C 4,
   Polynomial.X^2 + 2 * Polynomial.X + Polynomial.C (-1),
   Polynomial.X^2 + 2 * Polynomial.X + Polynomial.C 1,
   -Polynomial.X^2 + 2 * Polynomial.X + Polynomial.C 1,
   -Polynomial.X^2 - 2 * Polynomial.X + Polynomial.C (-1),
   Polynomial.X^2 - 2 * Polynomial.X + Polynomial.C (-1)]

def countPerfectSquareForm : ℤ :=
  List.countp isPerfectSquareForm expressions

theorem answer_is_Two :
  countPerfectSquareForm = 2 :=
sorry

end answer_is_Two_l608_608502


namespace triangle_area_equivalence_l608_608699

noncomputable def area_of_triangle_ABC : ℝ :=
  let OA := real.cbrt 150;
  let ∠BAC := 45 * (Real.pi / 180); -- converting degrees to radians
  let area := (OA ^ 2 * Real.sqrt 2) / 4 in
  area

theorem triangle_area_equivalence :
  area_of_triangle_ABC = (Real.cbrt 22500 * Real.sqrt 2) / 4 :=
by
  sorry

end triangle_area_equivalence_l608_608699


namespace area_of_shaded_quadrilateral_l608_608792

theorem area_of_shaded_quadrilateral :
  let side1 := 3,
      side2 := 5,
      side3 := 7,
      total_length := side1 + side2 + side3,
      height_ratio := side3 / total_length,
      height_small := side1 * height_ratio,
      height_middle := (side1 + side2) * height_ratio,
      base1 := height_small,
      base2 := height_middle,
      height_trapezoid := side2
  in (1 / 2) * (base1 + base2) * height_trapezoid = 77 / 6 :=
by
  sorry

end area_of_shaded_quadrilateral_l608_608792


namespace ratio_DB_DE_l608_608304

/-- In triangle ABC, the external angle bisector of ∠BAC intersects line BC at D.
    E is a point on ray AC such that ∠BDE = 2∠ADB.
    Given AB = 10, AC = 12, and CE = 33,
    we want to prove that DB/DE = 2/3. -/
theorem ratio_DB_DE {A B C D E : Type*} [MetricSpace E] [MetricSpace B]
  (triangle : Triangle A B C)
  (external_angle_bisector : ExternalAngleBisector A B C D)
  (E_on_ray_AC : OnRay A C E)
  (angle_condition : Angle B D E = 2 * Angle A D B)
  (AB_len : Distance A B = 10)
  (AC_len : Distance A C = 12)
  (CE_len : Distance C E = 33) :
  Distance D B / Distance D E = 2 / 3 := by
  sorry

end ratio_DB_DE_l608_608304


namespace lines_parallel_to_same_plane_possible_relationships_l608_608302

noncomputable def line (R : Type*) := R × R × R → R → R → R

def is_parallel_to_plane {R : Type*} [linear_ordered_field R] 
  (l : line R) (P : plane R) : Prop := 
  exists u v w d : R, ∀ p : R × R × R, l p d = u * p.1 + v * p.2 + w * p.3 
  ∧ parallel_plane P (u, v, w, d)

def relationship_between_lines {R : Type*} [linear_ordered_field R] 
  (l1 l2 : line R) (P : plane R) : Prop :=
  is_parallel_to_plane l1 P ∧ is_parallel_to_plane l2 P → 
  (parallel l1 l2 ∨ intersect l1 l2 ∨ skew l1 l2)

theorem lines_parallel_to_same_plane_possible_relationships {R : Type*} [linear_ordered_field R]
  (l1 l2 : line R) (P : plane R) (h1 : is_parallel_to_plane l1 P) (h2 : is_parallel_to_plane l2 P) :
  parallel l1 l2 ∨ intersect l1 l2 ∨ skew l1 l2 :=
sorry

end lines_parallel_to_same_plane_possible_relationships_l608_608302


namespace ellipse_standard_equation_l608_608248

theorem ellipse_standard_equation (h1 : True) -- Ellipse center at origin (implicit in standard form)
    (h2 : True) -- Foci F1 and F2 on the x-axis (implicit in standard form)
    (h3 : True) -- Point P on the ellipse (implicit in standard form)
    (eccentricity : ℝ) (h4 : eccentricity = 1 / 2) 
    (perimeter : ℝ) (h5 : perimeter = 12) :
    (∃ a b : ℝ, a = 4 ∧ b = sqrt 12 ∧ (∀ x y : ℝ, (x^2 / (a^2) + y^2 / (b^2) = 1 ↔ (x, y) lies_on_ellipse))) :=
by
  use 4, sqrt 12
  split
  { refl }
  split
  { rw sqrt_eq_iff_sq_eq ; norm_num }
  intros _ _ ; simp
  -- (∃ x y : ℝ ((x,y) : lies_on_ellipse)  ↔ x^2 / 16 + y^2 / 12 = 1
  sorry

end ellipse_standard_equation_l608_608248


namespace find_m_l608_608923

theorem find_m (m l : ℝ) (a b : ℝ × ℝ) (h_a : a = (2, m)) (h_b : b = (l, -2))
  (h_parallel : ∃ k : ℝ, k ≠ 0 ∧ a = k • (a + 2 • b)) :
  m = -4 :=
by
  sorry

end find_m_l608_608923


namespace distinct_arrangements_l608_608285

theorem distinct_arrangements : 
  let totalBooks := 7
  let mathBooks  := 3
  let novelBooks := 2
  let uniqueBooks := totalBooks - mathBooks - novelBooks
  factorial totalBooks / (factorial mathBooks * factorial novelBooks) = 420 := 
by
  sorry

end distinct_arrangements_l608_608285


namespace average_speed_of_bus_l608_608850

theorem average_speed_of_bus (speed_bicycle : ℝ)
  (start_distance : ℝ) (catch_up_time : ℝ)
  (h1 : speed_bicycle = 15)
  (h2 : start_distance = 195)
  (h3 : catch_up_time = 3) : 
  (start_distance + speed_bicycle * catch_up_time) / catch_up_time = 80 :=
by
  sorry

end average_speed_of_bus_l608_608850


namespace find_h_s_pairs_l608_608426

def num_regions (h s : ℕ) : ℕ :=
  1 + h * (s + 1) + s * (s + 1) / 2

theorem find_h_s_pairs (h s : ℕ) :
  h > 0 ∧ s > 0 ∧
  num_regions h s = 1992 ↔ 
  (h, s) = (995, 1) ∨ (h, s) = (176, 10) ∨ (h, s) = (80, 21) :=
by
  sorry

end find_h_s_pairs_l608_608426


namespace angle_EDF_55_l608_608337

theorem angle_EDF_55 (A B C D E F : Point) 
  (h_isosceles_ABC : AB = AC) 
  (h_angle_A : ∠A = 70°) 
  (D_on_BC : D ∈ line BC) 
  (E_on_AC : E ∈ line AC) 
  (F_on_AB : F ∈ line AB) 
  (h_equal_CE_CD : CE = CD) 
  (h_equal_BF_BD : BF = BD) : 
  ∠EDF = 55°
  :=
  by
    sorry

end angle_EDF_55_l608_608337


namespace find_a10_l608_608491

theorem find_a10 (a_n : ℕ → ℤ) (d : ℤ) (h1 : ∀ n, a_n n = a_n 1 + (n - 1) * d)
  (h2 : 5 * a_n 3 = a_n 3 ^ 2)
  (h3 : (a_n 3 + 2 * d) ^ 2 = (a_n 3 - d) * (a_n 3 + 11 * d))
  (h_nonzero : d ≠ 0) :
  a_n 10 = 23 :=
sorry

end find_a10_l608_608491


namespace Arianna_time_at_work_l608_608872

theorem Arianna_time_at_work : 
  (24 - (5 + 13)) = 6 := 
by 
  sorry

end Arianna_time_at_work_l608_608872


namespace hyperbola_eqn_proof_l608_608588

noncomputable def hyperbola_eqn (x y : ℝ) : Prop :=
  y^2 / 2 - x^2 = 1

theorem hyperbola_eqn_proof (C : Set (ℝ × ℝ))
  (h1 : (0, 0) ∈ C)
  (h2 : ∀ (x y : ℝ), (x, y) ∈ C → (x, -y) ∈ C ∧ (-x, y) ∈ C)
  (h3 : (1, -2) ∈ C)
  (h4 : ∀ x, y / x = sqrt 2 → (x, y) ∉ C) :
  ∀ (x y : ℝ), (x, y) ∈ C ↔ hyperbola_eqn x y :=
by sorry

end hyperbola_eqn_proof_l608_608588


namespace probability_digits_different_l608_608146

noncomputable def probability_all_digits_different : ℚ :=
  have tens_digits_probability := (9 / 9) * (8 / 9) * (7 / 9)
  have ones_digits_probability := (10 / 10) * (9 / 10) * (8 / 10)
  (tens_digits_probability * ones_digits_probability)

theorem probability_digits_different :
  probability_all_digits_different = 112 / 225 :=
by 
  -- The proof would go here, but it is not required for this task.
  sorry

end probability_digits_different_l608_608146


namespace number_of_geometric_progressions_l608_608389

def is_geometric_progression (a b c : ℕ) : Prop :=
  b * b = a * c

def is_distinct (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def is_in_set (a b c : ℕ) : Prop :=
  a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧
  b ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} ∧
  c ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem number_of_geometric_progressions : 
  (finset.image (λ x : ℕ × ℕ × ℕ, x.1 * x.1 = x.3 * x.2)
     {(a, b, c) ∈ finset.Ico 1 11 × finset.Ico 1 11 × finset.Ico 1 11 | 
      is_geometric_progression a b c ∧ is_distinct a b c ∧ is_in_set a b c}).card = 2 := 
by
  sorry

end number_of_geometric_progressions_l608_608389


namespace A_wins_by_15_meters_l608_608313

-- Define the conditions
def race_distance : ℕ := 500
def speed_ratio_A_B : ℕ × ℕ := (3, 4)
def head_start : ℕ := 140

-- The theorem to prove
theorem A_wins_by_15_meters (x : ℕ) (A B : ℕ) (hA : A = 3 * x) (hB : B = 4 * x) :
  let distance_B := race_distance
  let time_B := distance_B / B
  let distance_A := 3 * x * (distance_B / B)
  let total_distance_A := distance_A + head_start
  let A_win_margin := total_distance_A - race_distance
  A_win_margin = 15 :=
begin
  sorry
end

end A_wins_by_15_meters_l608_608313


namespace find_radius_of_circle_l608_608262

theorem find_radius_of_circle (l α : ℝ) (h1 : l = 2 * Real.pi / 3) (h2 : α = Real.pi / 3) : 
  (∃ R : ℝ, l = α * R ∧ R = 2) :=
by {
  use 2,
  split,
  { rw [h1, h2],
    simp only [div_mul_cancel, ne_of_gt Real.pi_pos, Ne.def, not_false_iff],
  },
  { refl }
}

end find_radius_of_circle_l608_608262


namespace inscribed_sphere_cone_radius_l608_608529

theorem inscribed_sphere_cone_radius :
  ∀ (r b d : ℝ), 
  (∀ (cone_base_radius cone_height : ℝ), 
    cone_base_radius = 10 ∧ cone_height = 40 →
    r = b * real.sqrt d - b →
    b + d = 19.5
  ) :=
begin
  intros r b d cone_base_radius cone_height h_radius_eq h_cone_dims,
  sorry
end

end inscribed_sphere_cone_radius_l608_608529


namespace perpendicular_distance_H_to_plane_EFG_l608_608885

open Real
open Matrix

/-- We define points H, E, F, and G in ℝ³. -/
def H : EuclideanSpace ℝ (Fin 3) := ![0, 0, 0]
def E : EuclideanSpace ℝ (Fin 3) := ![5, 0, 0]
def F : EuclideanSpace ℝ (Fin 3) := ![0, 6, 0]
def G : EuclideanSpace ℝ (Fin 3) := ![0, 0, 4]

/-- The main statement to prove the perpendicular distance from H to the plane containing E, F, and G is 4. -/
theorem perpendicular_distance_H_to_plane_EFG : 
  EuclideanSpace.perp_dist (plane_of_points E F G) H = 4 :=
sorry

/-- Define the plane containing points E, F, and G in ℝ³ -/
noncomputable def plane_of_points (E F G : EuclideanSpace ℝ (Fin 3)) :
  Submodule ℝ (EuclideanSpace ℝ (Fin 3)) :=
  let p₁ := F - E
  let p₂ := G - E
  Submodule.span ℝ {p₁, p₂}

/-- Calculate the perpendicular distance from a point to a plane -/
noncomputable def EuclideanSpace.perp_dist (plane : Submodule ℝ (EuclideanSpace ℝ (Fin 3))) 
    (point : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  dist point (orthogonal_projection plane point)
  
/-- Euclidean distance between two points in ℝ³ -/
noncomputable def dist (x y : EuclideanSpace ℝ (Fin 3)) : ℝ :=
  norm (x - y)

/-- Orthogonal projection of a point onto a subspace -/
noncomputable def orthogonal_projection (plane : Submodule ℝ (EuclideanSpace ℝ (Fin 3))) 
  (point : EuclideanSpace ℝ (Fin 3)) : EuclideanSpace ℝ (Fin 3) := 
  classical.some (plane.exists_orthogonal_projection point)

end perpendicular_distance_H_to_plane_EFG_l608_608885


namespace probability_of_x_plus_y_less_than_4_l608_608127

-- Define the square and the probability that x + y < 4 within this square.
theorem probability_of_x_plus_y_less_than_4 : 
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  (desired_area / square_area = 7 / 9) :=
by
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  show (desired_area / square_area = 7 / 9)
  sorry

end probability_of_x_plus_y_less_than_4_l608_608127


namespace distance_between_vertices_of_hyperbola_l608_608906

theorem distance_between_vertices_of_hyperbola :
  ∀ x y : ℝ, (y^2 / 45 - x^2 / 20 = 1) → distance (3 * real.sqrt 5, 0) (-3 * real.sqrt 5, 0) = 6 * real.sqrt 5 :=
by
  intro x y h
  sorry

end distance_between_vertices_of_hyperbola_l608_608906


namespace probability_cos_between_zero_and_half_is_one_third_l608_608718

noncomputable def probability_cos_between_zero_and_half : ℝ :=
  let open_interval := [- (Real.pi / 2), Real.pi / 2] in
  let cos_condition := { x | 0 < Real.cos x ∧ Real.cos x < (1 / 2) } in
  let probability := real.volume (cos_condition ∩ Icc (- (Real.pi / 2)) (Real.pi / 2)) / real.volume open_interval in
  probability

theorem probability_cos_between_zero_and_half_is_one_third :
  probability_cos_between_zero_and_half = 1 / 3 :=
by
  sorry

end probability_cos_between_zero_and_half_is_one_third_l608_608718


namespace EvenNumberOddScores_EvenNumberEvenScores_CannotHaveThreeZeros_SumAtLeast135_HighestScoreAtLeast15_l608_608086

-- Conditions: A league with 10 teams holds a round-robin tournament
constants (Team : Type) (team1 team2 : Team)

-- Each team plays every other team exactly once
constant plays : Team → Team → Prop
axiom plays_once : ∀ t1 t2, t1 ≠ t2 → plays t1 t2 ∧ plays t2 t1

-- A team earns 3 points for a win, 1 point for a draw, and 0 points for a loss
constant scores : Team → ℕ

-- Questions:
theorem EvenNumberOddScores : (∃ n, ∃ (T: fin n → Team), ∀ i, odd (scores (T i))) → ∃ k, even k := sorry
theorem EvenNumberEvenScores : (∃ n, ∃ (T: fin n → Team), ∀ i, even (scores (T i))) → ∃ k, even k := sorry
theorem CannotHaveThreeZeros : ¬(∃ T1 T2 T3, T1 ≠ T2 ∧ T2 ≠ T3 ∧ T1 ≠ T3 ∧ scores T1 = 0 ∧ scores T2 = 0 ∧ scores T3 = 0) := sorry
theorem SumAtLeast135 : (∑ t : Team, scores t) >= 135 = false := sorry
theorem HighestScoreAtLeast15 : (∃ t : Team, scores t >= 15) := sorry

end EvenNumberOddScores_EvenNumberEvenScores_CannotHaveThreeZeros_SumAtLeast135_HighestScoreAtLeast15_l608_608086


namespace compute_expression_equals_421200_l608_608883

theorem compute_expression_equals_421200 :
  (∏ i in (Finset.range 25).map (λi, i+1), (1 + (23:ℕ)/i)) /
  (∏ j in (Finset.range 21).map (λj, j+1), (1 + (27:ℕ)/j)) = 421200 := by
  sorry

end compute_expression_equals_421200_l608_608883


namespace A1B_parallel_plane_AC1D_CE_perp_plane_AC1D_cos_dihedral_angle_C_AC1_D_l608_608873

-- Defining the given conditions
variables {A B C A1 B1 C1 D E : Type}
variables (AB AC : ℝ) [fact (AB = 5)] [fact (AC = 5)]
variables {BC : line}
variables [midpoint D BC]
variables {BB1 : line}
variables [midpoint E BB1]
variables {B1BCC1 : square}
variables [fact (B1BCC1.side = 6)]

-- Definitions of involved planes and geometries
def plane_AC1D := plane (A, C1, D)
def line_A1B := line (A1, B)
def line_CE := line (C, E)
def dihedral_angle_C_AC1_D := dihedral_angle (plane (C, A, C1)) (plane (C, D, A))

-- The first part of the proof
theorem A1B_parallel_plane_AC1D : line_A1B ∥ plane_AC1D := sorry

-- The second part of the proof
theorem CE_perp_plane_AC1D : line_CE ⊥ plane_AC1D := sorry

-- The third part of the proof
theorem cos_dihedral_angle_C_AC1_D :
  cos (dihedral_angle_C_AC1_D) = 8 * (sqrt 5) / 25 := sorry

end A1B_parallel_plane_AC1D_CE_perp_plane_AC1D_cos_dihedral_angle_C_AC1_D_l608_608873


namespace approx_prob_all_defective_l608_608134

noncomputable def total_smartphones : ℕ := 400
noncomputable def defective_smartphones : ℕ := 150
noncomputable def chosen_smartphones : ℕ := 3

noncomputable def prob_first_defective : ℚ := defective_smartphones / total_smartphones
noncomputable def prob_second_defective : ℚ := (defective_smartphones - 1) / (total_smartphones - 1)
noncomputable def prob_third_defective : ℚ := (defective_smartphones - 2) / (total_smartphones - 2)

noncomputable def prob_all_defective : ℚ := prob_first_defective * prob_second_defective * prob_third_defective

theorem approx_prob_all_defective : prob_all_defective ≈ 0.0523 := 
by
  sorry

end approx_prob_all_defective_l608_608134


namespace compute_difference_of_squares_l608_608525

theorem compute_difference_of_squares :
  262^2 - 258^2 = 2080 := 
by
  sorry

end compute_difference_of_squares_l608_608525


namespace joes_speed_l608_608345

theorem joes_speed (pete_speed : ℝ) (joe_speed : ℝ) (time_minutes : ℝ) (distance : ℝ) (h1 : joe_speed = 2 * pete_speed) (h2 : time_minutes = 40) (h3 : distance = 16) : joe_speed = 16 :=
by
  sorry

end joes_speed_l608_608345


namespace probability_x_plus_y_lt_4_l608_608113

theorem probability_x_plus_y_lt_4 :
  let square_vertices := {(0,0), (0,3), (3,3), (3,0)} in
  let point_in_square (x y : ℝ) := 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 in
  let probability (A : ℝ) (B : ℝ) := A / B in
  ∃ (P : ℝ), P = probability 7 9 ∧
             P = (measure (λ (x y : ℝ), point_in_square x y ∧ x + y < 4)) / (measure (λ (x y : ℝ), point_in_square x y)) :=
sorry

end probability_x_plus_y_lt_4_l608_608113


namespace probability_of_x_plus_y_less_than_4_l608_608126

-- Define the square and the probability that x + y < 4 within this square.
theorem probability_of_x_plus_y_less_than_4 : 
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  (desired_area / square_area = 7 / 9) :=
by
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  show (desired_area / square_area = 7 / 9)
  sorry

end probability_of_x_plus_y_less_than_4_l608_608126


namespace concurrency_condition_iff_l608_608658

structure Triangle (α : Type u) where
  A B C : α

variables {α : Type u} [Inhabited α] [LinearOrder α] [EuclideanSpace α]

def altitude (T : Triangle α) (A B C : α) : Prop := sorry
def angle_bisector (T : Triangle α) (A B C : α) : Prop := sorry
def median (T : Triangle α) (A B C : α) : Prop := sorry

def concurrent (T : Triangle α) (A B C : α): Prop := sorry

theorem concurrency_condition_iff (T : Triangle α) (a b c : α) :
  concurrent T A B C ↔ (a^2 * (a - c) = (b^2 - c^2) * (a + c)) :=
sorry

end concurrency_condition_iff_l608_608658


namespace part_I_part_II_l608_608162

def S (n : ℕ) : ℕ := 2 ^ n - 1

def a (n : ℕ) : ℕ := 2 ^ (n - 1)

def T (n : ℕ) : ℕ := (n - 1) * 2 ^ n + 1

theorem part_I (S : ℕ → ℕ) (hS : ∀ n, S n = 2^n - 1) :
  ∀ n, ∃ a : ℕ → ℕ, a n = 2^(n-1) :=
by
  sorry

theorem part_II (a : ℕ → ℕ) (ha : ∀ n, a n = 2^(n-1)) :
  ∀ n, ∃ T : ℕ → ℕ, T n = (n - 1) * 2 ^ n + 1 :=
by
  sorry

end part_I_part_II_l608_608162


namespace find_quadratic_minimum_value_l608_608972

noncomputable def quadraticMinimumPoint (a b c : ℝ) : ℝ :=
  -b / (2 * a)

theorem find_quadratic_minimum_value :
  quadraticMinimumPoint 3 6 9 = -1 :=
by
  sorry

end find_quadratic_minimum_value_l608_608972


namespace high_school_senior_test_l608_608487

variable (p : ℚ) (k n : ℕ)

def bernoulli_trial (p : ℚ) (k n : ℕ) : ℚ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem high_school_senior_test :
  let p := 4 / 5 in
  let n := 4 in
  (bernoulli_trial p 3 n) + (bernoulli_trial p 4 n) = 512 / 625 :=
by
  sorry

end high_school_senior_test_l608_608487


namespace one_divisible_by_10_l608_608147

open Int

-- Conditions: Five distinct integers and divisibility conditions
variables (a₁ a₂ a₃ a₄ a₅ : ℤ)
variable (h : ∀ (i j k : Fin 5), i ≠ j → j ≠ k → i ≠ k →
  (a₁ :: a₂ :: a₃ :: a₄ :: a₅ :: []).nth i * 
  (a₁ :: a₂ :: a₃ :: a₄ :: a₅ :: []).nth j * 
  (a₁ :: a₂ :: a₃ :: a₄ :: a₅ :: []).nth k % 10 = 0)

-- The goal: one of these integers is divisible by 10
theorem one_divisible_by_10 : 
  a₁ % 10 = 0 ∨ a₂ % 10 = 0 ∨ a₃ % 10 = 0 ∨ a₄ % 10 = 0 ∨ a₅ % 10 = 0 := 
sorry

end one_divisible_by_10_l608_608147


namespace petya_mistake_l608_608712

theorem petya_mistake (a : ℕ) (h : a > 2) :
  let n := a^2 - 6
  ∃ d8 d0 d3,
  d8 = 2023 ∧ d0 = 2023 ∧ d3 = 2023 ∧ 
  (n.digits 10 = List.repeat 8 2023 ++ List.repeat 0 2023 ++ List.repeat 3 2023) ∧
  (n % 3 ≠ 1) :=
by
  let n := a^2 - 6
  sorry

end petya_mistake_l608_608712


namespace breads_remaining_l608_608784

def remaining_breads_after_thief (B : ℕ) (n : ℕ) : ℝ :=
  (B : ℝ) / 2^n - (2^n - 1) / 2

theorem breads_remaining {B : ℕ} (h : B = 127) : remaining_breads_after_thief B 5 = 3 := by
  simp [remaining_breads_after_thief, h]
  sorry

end breads_remaining_l608_608784


namespace simplify_expression_l608_608543

noncomputable def expression : ℝ :=
  (4 * (Real.sqrt 3 + Real.sqrt 7)) / (5 * Real.sqrt (3 + (1 / 2)))

theorem simplify_expression : expression = (16 + 8 * Real.sqrt 21) / 35 := by
  sorry

end simplify_expression_l608_608543


namespace find_f2_l608_608822

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f2 (h : ∀ x : ℝ, f x + 3 * f (8 - x) = x) : f 2 = 2 :=
by
  sorry

end find_f2_l608_608822


namespace min_positive_period_f_l608_608751

-- Define the function y = cos((π / 4) - (x / 3))
def f (x : ℝ) : ℝ := Real.cos ((Real.pi / 4) - (x / 3))

-- Theorem statement: The minimum positive period of f is 6π
theorem min_positive_period_f : ∃ T > 0, T = 6 * Real.pi ∧ ∀ x : ℝ, f (x + T) = f x := by
  sorry

end min_positive_period_f_l608_608751


namespace product_of_digits_base8_l608_608438

theorem product_of_digits_base8 (n : ℕ) (h : n = 8670) : 
  let digits := [2, 0, 7, 3, 6] in
  digits.product = 0 :=
by
  let digits := [2, 0, 7, 3, 6]
  have h1 : digits.product = 0 := by simp
  exact h1

end product_of_digits_base8_l608_608438


namespace center_of_circumcircle_lies_on_AK_l608_608841

variable {α β γ : Real} -- Angles in triangle ABC
variable (A B C L H K O : Point) -- Points in the configuration
variable (circumcircle_ABC : TriangularCircumcircle A B C) -- Circumcircle of triangle ABC

-- Definitions based on the given conditions
variable (is_angle_bisector : angle_bisector A B C L)
variable (is_height : height_from_point_to_line B A L H)
variable (intersects_circle_at_K : intersects_circumcircle A B L K circumcircle_ABC)
variable (is_circumcenter : circumcenter A B C O circumcircle_ABC)

theorem center_of_circumcircle_lies_on_AK
  (h_angle_bisector : is_angle_bisector)
  (h_height : is_height)
  (h_intersects_circle_at_K : intersects_circle_at_K)
  (h_circumcenter : is_circumcenter) 
    : lies_on_line O A K := 
sorry -- Proof is omitted

end center_of_circumcircle_lies_on_AK_l608_608841


namespace factorize_polynomial_trig_inequality_l608_608050

-- Problem 1: Factorize the polynomial
theorem factorize_polynomial (x : ℝ) :
  x^12 + x^9 + x^6 + x^3 + 1 = (x^4 + x^3 + x^2 + 1) * (x^8 - x^7 + x^5 - x^4 + x^3 - x + 1) :=
sorry

-- Problem 2: Trigonometric inequality
theorem trig_inequality (θ : ℝ) :
  5 + 8 * cos θ + 4 * cos (2 * θ) + cos (3 * θ) ≥ 0 :=
sorry

end factorize_polynomial_trig_inequality_l608_608050


namespace col_of_1000_is_first_l608_608413

def row (n : ℕ) : ℕ := (n + 3) / 4
def even_row (n : ℕ) : Prop := (row n % 2 = 0)

def col (n : ℕ) : String :=
  if even_row n then
    if n % 4 = 0 then "first"
    else if n % 4 = 3 then "second"
    else if n % 4 = 2 then "third"
    else "fourth"
  else
    if n % 4 = 1 then "second"
    else if n % 4 = 2 then "third"
    else if n % 4 = 3 then "fourth"
    else "fifth"

theorem col_of_1000_is_first : col 1000 = "first" := 
  by sorry

end col_of_1000_is_first_l608_608413


namespace inverse_proportion_graph_l608_608602

theorem inverse_proportion_graph (m n : ℝ) (h : n = -2 / m) : m = -2 / n :=
by
  sorry

end inverse_proportion_graph_l608_608602


namespace intersection_M_N_l608_608234

def M := {1, 2, 3, 4}
def N := {x : ℤ | 3 * x - 7 ≥ 8 - 2 * x}

theorem intersection_M_N : M ∩ N = {3, 4} :=
by  sorry

end intersection_M_N_l608_608234


namespace triangle_distance_and_circumradius_proof_l608_608338

noncomputable def triangle_dist_and_radius (A B C : Point) (alpha beta gamma : ℝ)
  (a b c : ℝ) (B' C' : Point) : Prop :=
  let B'C' := a * Real.tan beta * Real.tan gamma
  let r := a * Real.tan beta * Real.tan gamma * Real.sin (45 - beta / 2) 
            * Real.sin (45 - gamma / 2)
  B'C' = dist B' C' ∧ 
  r = circumradius (triangle A B' C')

variables A B C B' C' : Point
variables alpha beta gamma a b c: ℝ

theorem triangle_distance_and_circumradius_proof :
  triangle_dist_and_radius A B C alpha beta gamma a b c B' C' :=
sorry

end triangle_distance_and_circumradius_proof_l608_608338


namespace find_f1_l608_608264

-- Given conditions
variable (g : ℝ → ℝ)
variable (f : ℝ → ℝ)
variable (h_odd_f : ∀ x : ℝ, f (-x) = -f x)
variable (h_g_neg1 : g (-1) = -1)
noncomputable def f_def : ℝ → ℝ := λ x, 2 * g x - x^2

-- Proof statement
theorem find_f1 : f 1 = 3 := by
  -- Conditions applied directly
  have h_f_def : ∀ x : ℝ, f x = f_def g x := sorry
  have h_f_minus1 : f (-1) = (2 * -1) - 1 := by
    rw [←h_f_def (-1)]
    rw [h_g_neg1]
    sorry
  have h_f1_rel : -f 1 = f (-1) := by
    rw [h_odd_f 1]
    sorry
  sorry

#eval find_f1 -- This will evaluate the statement

end find_f1_l608_608264


namespace possible_relationships_between_lines_l608_608756

def positional_relationships (l1 l2 : ℝ → ℝ^3) : Prop :=
  (∃ d : ℝ, ∀ t₁ t₂ : ℝ, ‖l1 t₁ - l2 t₂‖ = d) ∨            -- Parallel
  (∃ p : ℝ^3, ∃ t₁ t₂ : ℝ, l1 t₁ = p ∧ l2 t₂ = p) ∨         -- Intersecting
  (∀ p : ℝ^3, ∃ t₁ t₂ : ℝ, l1 t₁ ≠ p ∧ l2 t₂ ≠ p)            -- Skew Lines

theorem possible_relationships_between_lines (l1 l2 : ℝ → ℝ^3) : positional_relationships l1 l2 :=
by
  sorry

end possible_relationships_between_lines_l608_608756


namespace arithmetic_sequence_formula_l608_608941

theorem arithmetic_sequence_formula (x : ℤ) (a : ℕ → ℤ) 
  (h1 : a 1 = x - 1) (h2 : a 2 = x + 1) (h3 : a 3 = 2 * x + 3) :
  ∃ c d : ℤ, (∀ n : ℕ, a n = c + d * (n - 1)) ∧ ∀ n : ℕ, a n = 2 * n - 3 :=
by {
  sorry
}

end arithmetic_sequence_formula_l608_608941


namespace sum_prod_set_is_21_l608_608192

open Finset

def prod_set (s1 s2 : Finset ℕ) : Finset ℕ := 
  s1.bind (λ x, s2.image (λ y, x * y))

def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {3, 6}

theorem sum_prod_set_is_21 : (prod_set A B).sum id = 21 := 
by
  sorry

end sum_prod_set_is_21_l608_608192


namespace larger_tablet_diagonal_length_l608_608420

theorem larger_tablet_diagonal_length :
  ∀ (d : ℝ), (d^2 / 2 = 25 / 2 + 5.5) → d = 6 :=
by
  intro d
  sorry

end larger_tablet_diagonal_length_l608_608420


namespace original_price_l608_608859

variables (p q d : ℝ)


theorem original_price (x : ℝ) (h : x * (1 + p / 100) * (1 - q / 100) = d) :
  x = 100 * d / (100 + p - q - p * q / 100) := 
sorry

end original_price_l608_608859


namespace number_identification_l608_608492

theorem number_identification (x : ℝ) (h : x ^ 655 / x ^ 650 = 100000) : x = 10 :=
by
  sorry

end number_identification_l608_608492


namespace probability_x_plus_y_lt_4_l608_608107

theorem probability_x_plus_y_lt_4 :
  let square_vertices := {(0,0), (0,3), (3,3), (3,0)} in
  let point_in_square (x y : ℝ) := 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 in
  let probability (A : ℝ) (B : ℝ) := A / B in
  ∃ (P : ℝ), P = probability 7 9 ∧
             P = (measure (λ (x y : ℝ), point_in_square x y ∧ x + y < 4)) / (measure (λ (x y : ℝ), point_in_square x y)) :=
sorry

end probability_x_plus_y_lt_4_l608_608107


namespace sequence_fourth_term_l608_608419

-- Define the sequence and the pattern
def sequence (n : ℕ) : ℕ
| 0     := 2
| 1     := 4
| 2     := 8
| 3     := 14
| (n+4) := sequence (n + 3) + 2 * (n + 2)

-- Define a proposition to state that the 4th term in the sequence equals 22
theorem sequence_fourth_term : sequence 4 = 22 :=
by {
  -- Proof will go here
  sorry
}

end sequence_fourth_term_l608_608419


namespace difference_of_squares_example_l608_608522

theorem difference_of_squares_example :
  (262^2 - 258^2 = 2080) :=
by {
  sorry -- placeholder for the actual proof
}

end difference_of_squares_example_l608_608522


namespace standard_equation_of_parabola_area_of_triangle_AOB_l608_608573

/-
Problem Statement:
Given a parabola with its vertex at the origin and its focus on the x-axis, and it passes through the point (1,2).
1) Find the standard equation of the parabola.
2) The line y = x - 4 intersects the parabola at points A and B. Find the area of triangle AOB.
-/

-- Define the conditions of the parabola
def parabola_condition (x y p : ℝ) := y^2 = 2 * p * x ∧ (1, 2) ∈ { (x, y) | y^2 = 2 * p * x }

-- The statement for the first question
theorem standard_equation_of_parabola : ∃ p, (- p) * (1^2) = 2 -> p = 2 ∧ ∀ x y, y^2 = 4 * x :=
by
  sorry

-- Define the conditions of the intersection points A and B
def intersect_condition (x1 x2 y1 y2 : ℝ) :=
  y1 = x1 - 4 ∧ y2 = x2 - 4 ∧ y1^2 = 4 * x1 ∧ y2^2 = 4 * x2

-- The statement for the second question
theorem area_of_triangle_AOB (A B O : ℝ × ℝ) : intersect_condition A.1 B.1 A.2 B.2 -> ∃ S, S = 16 * real.sqrt 5 :=
by
  sorry

end standard_equation_of_parabola_area_of_triangle_AOB_l608_608573


namespace product_of_invertible_function_labels_is_120_l608_608748

def func2 (x : ℝ) : ℝ := x^3 - 3 * x
def points3 : set (ℝ × ℝ) := {(1, 2), (2, 4), (3, 6)}
noncomputable def func4_domain : set ℝ := {-3*Real.pi/2, -Real.pi, -Real.pi/2, 0, Real.pi/2, Real.pi, 3*Real.pi/2}
noncomputable def func4 (x : ℝ) : ℝ := Real.sin x
def func5 (x : ℝ) : ℝ := 1 / x

def labels_of_invertible_functions := {2, 3, 4, 5}
def product_of_labels : ℕ := 2 * 3 * 4 * 5

theorem product_of_invertible_function_labels_is_120 :
  (∃ l2, l2 = 2 ∧ function.injective func2) ∧
  (∃ l3, l3 = 3 ∧ ∀ p1 p2 ∈ points3, p1.1 ≠ p2.1 → p1.2 ≠ p2.2) ∧
  (∃ l4, l4 = 4 ∧ ∀ x1 x2 ∈ func4_domain, x1 ≠ x2 → func4 x1 ≠ func4 x2) ∧
  (∃ l5, l5 = 5 ∧ function.injective func5) →
  product_of_labels = 120 := 
by
  sorry

end product_of_invertible_function_labels_is_120_l608_608748


namespace larger_circle_radius_l608_608150

-- Define the problem statement
theorem larger_circle_radius (r : ℝ) : 
  (∃ θ : ℝ, θ = 60 ∧ 
   ∃ C1 C2 : Type, 
     (C1.radius = r) ∧
     (C2.radius = R) ∧
     (C1 tangent_to C2) ∧
     acute_angle_contains θ C1 C2) →
  R = 3 * r :=
by
  sorry

end larger_circle_radius_l608_608150


namespace new_quadratic_eq_l608_608559

variables {a b c : ℝ} (ha : a ≠ 0)

theorem new_quadratic_eq (h1 : ∀ x1 x2 : ℝ, x1 + x2 = -b / a) (h2 : ∀ x1 x2 : ℝ, x1 * x2 = c / a) :
  ∃ x : ℝ, a ^ 2 * x^2 + (a * b - a * c) * x - b * c = 0 :=
by
  have h_sum := h1,
  have h_prod := h2,
  -- Assume the new roots \bar{x}_1 and \bar{x}_2
  let β1 := -b / a,
  let β2 := c / a,
  -- Use Vieta's formula
  let sum_β := β1 + β2,
  let prod_β := β1 * β2,
  -- New quadratic equation with roots β1 and β2
  let new_eq := a^2 * x^2 + (a * (b - c)) * x - b * c,
  use new_eq,
  sorry

end new_quadratic_eq_l608_608559


namespace complement_union_l608_608980

open Set

def U := { x : ℤ | x^2 - 5*x - 6 ≤ 0 }
def A := { x : ℤ | x * (2 - x) ≥ 0 }
def B := {1, 2, 3}

theorem complement_union (x : ℤ) : 
  x ∈ U \ (A ∪ B) ↔ x ∈ {-1, 4, 5, 6} := by
  sorry

end complement_union_l608_608980


namespace segments_less_than_bound_l608_608236

theorem segments_less_than_bound (n : ℕ) (points : fin n → ℝ × ℝ) :
  let edges := { (i, j) | i < j ∧ (dist points[i] points[j] = 1) } in
  edges.card < (n^2 / 3) := 
sorry

end segments_less_than_bound_l608_608236


namespace sum_of_squares_of_roots_eq_30_l608_608692

noncomputable def polynomial := (x : ℝ) → x^4 - 15 * x^2 + 56 = 0

theorem sum_of_squares_of_roots_eq_30
  (a b c d : ℝ)
  (h1 : polynomial a)
  (h2 : polynomial b)
  (h3 : polynomial c)
  (h4 : polynomial d) : 
  a^2 + b^2 + c^2 + d^2 = 30 :=
sorry

end sum_of_squares_of_roots_eq_30_l608_608692


namespace quadrilateral_centroid_area_ratio_l608_608355

noncomputable def centroid (A B C : ℝ × ℝ) : ℝ × ℝ := 
((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

theorem quadrilateral_centroid_area_ratio (A B C D G_A G_B G_C G_D : ℝ × ℝ)
  (hA : G_A = centroid B C D)
  (hB : G_B = centroid A C D)
  (hC : G_C = centroid A B D)
  (hD : G_D = centroid A B C) :
  (area_of_convex_quadrilateral G_A G_B G_C G_D / area_of_convex_quadrilateral A B C D) = 1 / 9 := 
sorry

end quadrilateral_centroid_area_ratio_l608_608355


namespace average_speed_is_72_l608_608060

-- Definitions for the given conditions
def total_distance (s : ℝ) := s
def speed_section1 := 60 -- km/h
def speed_section2 := 120 -- km/h
def speed_section3 := 80 -- km/h

def distance_section1 (s : ℝ) := s / 2
def distance_section2 (s : ℝ) := s / 6
def distance_section3 (s : ℝ) := s / 3

def time_section1 (s : ℝ) := distance_section1 s / speed_section1
def time_section2 (s : ℝ) := distance_section2 s / speed_section2
def time_section3 (s : ℝ) := distance_section3 s / speed_section3

def total_time (s : ℝ) := time_section1 s + time_section2 s + time_section3 s

def average_speed (s : ℝ) := total_distance s / total_time s

-- The theorem we need to prove
theorem average_speed_is_72 (s : ℝ) (h : s > 0) : average_speed s = 72 := by
  sorry

end average_speed_is_72_l608_608060


namespace number_of_oranges_l608_608674

def apples : ℕ := 14
def more_oranges : ℕ := 10

theorem number_of_oranges (o : ℕ) (apples_eq : apples = 14) (more_oranges_eq : more_oranges = 10) :
  o = apples + more_oranges :=
by
  sorry

end number_of_oranges_l608_608674


namespace find_r_l608_608684

noncomputable def f (r x : ℝ) : ℝ := (x - (r + 1)) * (x - (r + 7)) * (x + (a r x))
noncomputable def g (r x : ℝ) : ℝ := (x - (r + 3)) * (x - (r + 9)) * (x + (b r x))
axiom f_monoc_polynomial : ∀ r x : ℝ, polynomial.monic (polynomial.C (f r x))
axiom g_monoc_polynomial : ∀ r x : ℝ, polynomial.monic (polynomial.C (g r x))
axiom f_g_difference : ∀ r x : ℝ, f r x - g r x = r

theorem find_r : ∃ r : ℝ, r = 32 :=
by
    sorry

end find_r_l608_608684


namespace chores_for_cartoon_time_l608_608210

def hours_to_minutes (hours : ℕ) : ℕ := hours * 60

def cartoons_to_chores (cartoon_minutes : ℕ) : ℕ := cartoon_minutes * 8 / 10

theorem chores_for_cartoon_time (h : ℕ) (h_eq : h = 2) : cartoons_to_chores (hours_to_minutes h) = 96 :=
by
  rw [h_eq, hours_to_minutes, cartoons_to_chores]
  -- steps demonstrating transformation from hours to minutes and calculation of chores will follow here
  sorry

end chores_for_cartoon_time_l608_608210


namespace range_of_m_l608_608299

variable {R : Type*} [LinearOrderedField R]

def discriminant (a b c : R) := b * b - 4 * a * c

theorem range_of_m (m : R) : (∀ x : R, x^2 + m * x + 1 > 0) ↔ -2 < m ∧ m < 2 :=
by sorry

end range_of_m_l608_608299


namespace female_adults_present_l608_608671

variable (children : ℕ) (male_adults : ℕ) (total_people : ℕ)
variable (children_count : children = 80) (male_adults_count : male_adults = 60) (total_people_count : total_people = 200)

theorem female_adults_present : ∃ (female_adults : ℕ), 
  female_adults = total_people - (children + male_adults) ∧ 
  female_adults = 60 :=
by
  sorry

end female_adults_present_l608_608671


namespace complex_number_solution_l608_608290

theorem complex_number_solution {z : ℂ} (hz : z * complex.I = 1 - real.sqrt 5 * complex.I) :
  z = real.sqrt 5 - complex.I :=
sorry

end complex_number_solution_l608_608290


namespace highway_length_l608_608803

theorem highway_length 
  (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) 
  (h_speed1 : speed1 = 14)
  (h_speed2 : speed2 = 16)
  (h_time : time = 1.5) : 
  speed1 * time + speed2 * time = 45 := 
sorry

end highway_length_l608_608803


namespace ben_family_fish_filets_l608_608820

def fish_lengths_ben    : List ℕ := [5, 5, 9, 9]
def fish_lengths_judy   : List ℕ := [11]
def fish_lengths_billy  : List ℕ := [6, 6, 10]
def fish_lengths_jim    : List ℕ := [4, 8]
def fish_lengths_susie  : List ℕ := [3, 7, 7, 12, 12]

def all_fish_lengths : List ℕ := fish_lengths_ben ++ fish_lengths_judy ++ fish_lengths_billy ++ fish_lengths_jim ++ fish_lengths_susie

def can_keep_fish (min_size : ℕ) : ℕ → Bool := λ length, min_size ≤ length

def fish_that_can_be_kept : List ℕ := (all_fish_lengths.filter (can_keep_fish 6))
def fish_count := fish_that_can_be_kept.length
def filets_per_fish := 2

theorem ben_family_fish_filets : fish_count * filets_per_fish = 22 :=
  by sorry

end ben_family_fish_filets_l608_608820


namespace seating_arrangements_l608_608007

def totalSeatsInFrontRow : Nat := 11
def totalSeatsInBackRow : Nat := 12
def notOccupiedSeatsInFrontRow : Finset Nat := {5, 6, 7} -- 1-based indexing for the seats in the middle.
def conditionNoAdjacent : Prop := 
  ∀ (f : Nat) (b : Nat), abs f - b ≠ 1 -- Placeholder for a proper non-adjacency condition check.

theorem seating_arrangements (h1 : 11 = totalSeatsInFrontRow) 
                            (h2 : 12 = totalSeatsInBackRow)
                            (h3 : ∀ (seat : Nat), seat ∈ notOccupiedSeatsInFrontRow → seat ∉ (1:11))
                            (h4 : conditionNoAdjacent): 
  number_of_seating_arrangements =
  346 :=
sorry

end seating_arrangements_l608_608007


namespace all_forms_true_for_③_l608_608595

theorem all_forms_true_for_③
  (A B : ℝ)
  (triangle_ABC: Prop) -- assume there exists a triangle ABC
  (h1 : A ≠ B → sin A ≠ sin B)
  (h2 : sin A = sin B → A = B)
  (h3 : A = B → sin A = sin B)
  (h4 : sin A ≠ sin B → A ≠ B)
  : true :=
by sorry

end all_forms_true_for_③_l608_608595


namespace find_sum_of_squares_of_roots_l608_608690

theorem find_sum_of_squares_of_roots:
  ∀ (a b c d : ℝ), (a^2 * b^2 * c^2 * d^2 - 15 * a * b * c * d + 56 = 0) → 
  a^2 + b^2 + c^2 + d^2 = 30 := by
  intros a b c d h
  sorry

end find_sum_of_squares_of_roots_l608_608690


namespace smallest_three_digit_pqr_l608_608812

theorem smallest_three_digit_pqr (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  100 ≤ p * q^2 * r ∧ p * q^2 * r < 1000 → p * q^2 * r = 126 := 
sorry

end smallest_three_digit_pqr_l608_608812


namespace complement_union_l608_608977

-- Definition of the universal set U
def U : Set ℤ := {x | x^2 - 5 * x - 6 ≤ 0}

-- Definition of set A
def A : Set ℤ := {x | x * (2 - x) ≥ 0}

-- Definition of set B
def B : Set ℤ := {1, 2, 3}

-- The proof statement
theorem complement_union (h : U = {x | x^2 - 5 * x - 6 ≤ 0} ∧ 
                           A = {x | x * (2 - x) ≥ 0} ∧ 
                           B = {1, 2, 3}) : 
  U \ (A ∪ B) = {-1, 4, 5, 6} :=
by {
  sorry
}

end complement_union_l608_608977


namespace area_of_triangle_hyperbola_focus_l608_608254

theorem area_of_triangle_hyperbola_focus :
  let F₁ := (-Real.sqrt 2, 0)
  let F₂ := (Real.sqrt 2, 0)
  let hyperbola := {p : ℝ × ℝ | p.1 ^ 2 - p.2 ^ 2 = 1}
  let asymptote (p : ℝ × ℝ) := p.1 = p.2
  let circle := {p : ℝ × ℝ | (p.1 - F₁.1 / 2) ^ 2 + (p.2 - F₁.2 / 2) ^ 2 = (Real.sqrt 2) ^ 2}
  let P := (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
  let Q := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)
  let area (p1 p2 p3 : ℝ × ℝ) := 0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))
  area F₁ P Q = Real.sqrt 2 := 
sorry

end area_of_triangle_hyperbola_focus_l608_608254


namespace problem_equiv_l608_608966

noncomputable def f (a ω b : ℝ) (x : ℝ) : ℝ := a * sin (2 * ω * x + π / 6) + a / 2 + b

theorem problem_equiv :
  ∀ (a ω b : ℝ),
  (∀ x, f a ω b x = a * sin (2 * ω * x + π / 6) + a / 2 + b) →
  (∀ x, x ∈ ℝ) →
  (a > 0) →
  (ω > 0) →
  (∀ x, f a ω b x = f a ω b (x + π / (2 * ω))) →
  (∀ x, -a + a / 2 + b ≤ f a ω b x ∧ f a ω b x ≤ a + a / 2 + b) →
  f a ω b (1 / 2) = 7 →
  f a ω b (3 / 2) = 3 →
  ω = 1 ∧ a = 2 ∧ b = 4 ∧
  ∀ k : ℤ, [k * π - π / 3, k * π + π / 6] = {x : ℝ | is_increasing (f 2 1 4 x)} ∧
  (∀ x ∈ [-π/3, π/3), 5 - sqrt 3 ≤ f 2 1 4 (x + π/12) ∧ f 2 1 4 (x + π/12) ≤ 7) := 
sorry

end problem_equiv_l608_608966


namespace zero_in_interval_l608_608772

def f (x : ℝ) : ℝ := 5 * x

theorem zero_in_interval : ∃ x ∈ Ioo (-1 : ℝ) 0, f x = 0 :=
by {
  sorry
}

end zero_in_interval_l608_608772


namespace max_height_reached_threat_to_object_at_70km_l608_608743

noncomputable def initial_acceleration : ℝ := 20 -- m/s^2
noncomputable def duration : ℝ := 50 -- seconds
noncomputable def gravity : ℝ := 10 -- m/s^2
noncomputable def height_at_max_time : ℝ := 75000 -- meters (75km)

-- Proof that the maximum height reached is 75 km
theorem max_height_reached (a τ g : ℝ) (H : ℝ) (h₀: a = initial_acceleration) (h₁: τ = duration) (h₂: g = gravity) (h₃: H = height_at_max_time) :
  H = 75 * 1000 := 
sorry

-- Proof that the rocket poses a threat to an object located at 70 km
theorem threat_to_object_at_70km (a τ g : ℝ) (H : ℝ) (h₀: a = initial_acceleration) (h₁: τ = duration) (h₂: g = gravity) (h₃: H = height_at_max_time) :
  H > 70 * 1000 :=
sorry

end max_height_reached_threat_to_object_at_70km_l608_608743


namespace convex_quadrilateral_count_l608_608705

theorem convex_quadrilateral_count (n : ℕ) (h : n ≥ 5) 
  (non_collinear : ∀ (p1 p2 p3 : ℝ × ℝ), p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → 
    ¬ collinear {p1, p2, p3}) :
  ∃ k : ℕ, k ≥ (Nat.choose n 5) / (n - 4) :=
sorry

end convex_quadrilateral_count_l608_608705


namespace xy_value_l608_608715

theorem xy_value (x y : ℤ) (h1 : x = -3 - 2) (h2 : y - 3 = -1) : x * y = -10 :=
by
  -- Hypotheses introduce values of x and y under given conditions
  have hx : x = -5 := by rw [h1, int.add_comm]
  have hy : y = 2 := by linarith
  rw [hx, hy]
  norm_num
  sorry

end xy_value_l608_608715


namespace arithmetic_and_geometric_seq_properties_l608_608591

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1
noncomputable def b_n (n : ℕ) : ℕ := 3 ^ n
noncomputable def c_n (n : ℕ) : ℕ := a_n n + b_n n
noncomputable def S_n (n : ℕ) : ℕ := n * (n + 2) + (3 ^ n - 1) / 2

theorem arithmetic_and_geometric_seq_properties (n : ℕ) (hn : n > 0) :
  (a_n 1 = 3) ∧ (b_n 1 = 3) ∧
  (a_n n = 2 * n + 1) ∧ (b_n n = 3 ^ n) ∧
  (S_n n = ∑ i in range n, c_n (i + 1)) :=
  by
  sorry

end arithmetic_and_geometric_seq_properties_l608_608591


namespace number_above_161_is_177_l608_608813

theorem number_above_161_is_177 :
  let k := 13 in
  let start_kth_row := 2 * k^2 - 4 * k + 2 in
  let numbers_in_kth_row := 2 * k - 1 in
  let row_numbers := list.range (2 * numbers_in_kth_row) |>.map (λ num, start_kth_row + 2 * num) in
  let index_of_161 := row_numbers.index_of 161 in
  let previous_row_start := 2 * (k - 1)^2 - 4 * (k - 1) + 2 in
  let previous_row_numbers := list.range (2 * (numbers_in_kth_row - 2)) |>.map (λ num, previous_row_start + 2 * num) in
  previous_row_numbers.index_of 177 = index_of_161 :=
sorry

end number_above_161_is_177_l608_608813


namespace monic_cubic_polynomial_exists_l608_608901

theorem monic_cubic_polynomial_exists :
  ∃ Q : ℚ[X], monic Q ∧ Q.coeff 3 = 1 ∧ (Q.eval (real.cbrt 5 + 2)) = 0 := sorry

end monic_cubic_polynomial_exists_l608_608901


namespace reciprocal_inequality_of_negatives_l608_608579

variable (a b : ℝ)

/-- Given that a < b < 0, prove that 1/a > 1/b. -/
theorem reciprocal_inequality_of_negatives (h1 : a < b) (h2 : b < 0) : (1/a) > (1/b) :=
sorry

end reciprocal_inequality_of_negatives_l608_608579


namespace find_digits_l608_608740

theorem find_digits (x y z n : ℕ) (h_n : 2 ≤ n) :
  let a := x * (10^n - 1) / 9,
      b := y * (10^n - 1) / 9,
      c := z * (10^(2 * n) - 1) / 9 in
  a^2 + b = c ↔
  (x = 3 ∧ y = 2 ∧ z = 1) ∨
  (x = 6 ∧ y = 8 ∧ z = 4) ∨
  (x = 9 ∧ y = 18 ∧ z = 9) :=
by sorry

end find_digits_l608_608740


namespace second_car_speed_ratio_l608_608014

-- Given definitions
def first_car_speed (v1 : ℝ) : Prop := v1 > 0
def second_car_speed (v2 : ℝ) : Prop := v2 > 0
def travel_time := 3
def delay_time := 1.1
def speed_ratio (v1 v2 : ℝ) := v2 / v1

-- Question translated to Lean
theorem second_car_speed_ratio (v1 v2 : ℝ) (hv1 : first_car_speed v1) (hv2 : second_car_speed v2)
  (h1 : 3 * v2 / v1 - 3 * v1 / v2 = 1.1) :
  v2 / v1 = 6 / 5 := 
sorry

end second_car_speed_ratio_l608_608014


namespace cindy_smallest_coins_l608_608177

theorem cindy_smallest_coins : ∃ n : ℕ, 
  (∀ Y : ℕ, Y > 1 ∧ Y < n → n % Y = 0 → ∃ k : ℕ, k * Y = n) ∧ 
  (15 = (finset.filter (λ m, m > 1 ∧ m < n) (finset.Ico 1 (n+1))).card) ∧
  n = 65536 :=
begin
  sorry
end

end cindy_smallest_coins_l608_608177


namespace projection_magnitude_of_b_onto_a_l608_608575

open Real

def vector := ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : vector) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2)

def projection_magnitude (a b : vector) : ℝ :=
  abs (dot_product a b / magnitude a)

theorem projection_magnitude_of_b_onto_a :
  let a : vector := (2, 0)
  let b : vector := (1, 1)
  projection_magnitude a b = 1 :=
by
  sorry

end projection_magnitude_of_b_onto_a_l608_608575


namespace largest_diff_after_2023_changesums_l608_608915

def changesum (l : List ℕ) : List ℕ :=
match l with
| [x, y, z] => [y + z, x + z, x + y]
| _ => l -- This case handles any other input that doesn't match the pattern.

-- Initial list
def initial_list : List ℕ := [20, 2, 3]

-- Applying changesum n times
def apply_changesum (n : ℕ) (l : List ℕ) : List ℕ :=
Nat.iterate changesum n l

-- After 2023 operations
def final_list : List ℕ := apply_changesum 2023 initial_list

-- The largest difference in a list of three numbers
def largest_difference (l : List ℕ) : ℕ :=
match l with
| [x, y, z] => max (max (Nat.abs (x - y)) (Nat.abs (x - z))) (Nat.abs (y - z))
| _ => 0 -- This case handles any other input that doesn't match the pattern.

-- Problem statement in Lean 4
theorem largest_diff_after_2023_changesums : 
  largest_difference (apply_changesum 2023 [20, 2, 3]) = 18 :=
sorry

end largest_diff_after_2023_changesums_l608_608915


namespace find_threedigit_number_l608_608075

-- Define the three-digit number and its reverse
def original_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
def reversed_number (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

-- Define the condition of adding the number and its reverse to get 1777
def number_sum_condition (a b c : ℕ) : Prop :=
  original_number a b c + reversed_number a b c = 1777

-- Prove the existence of digits a, b, and c that satisfy the conditions
theorem find_threedigit_number :
  ∃ a b c : ℕ, a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  original_number a b c = 859 ∧ 
  reversed_number a b c = 958 ∧ 
  number_sum_condition a b c :=
sorry

end find_threedigit_number_l608_608075


namespace antonov_packs_remaining_l608_608509

theorem antonov_packs_remaining (total_candies : ℕ) (pack_size : ℕ) (packs_given : ℕ) (candies_remaining : ℕ) (packs_remaining : ℕ) :
  total_candies = 60 →
  pack_size = 20 →
  packs_given = 1 →
  candies_remaining = total_candies - pack_size * packs_given →
  packs_remaining = candies_remaining / pack_size →
  packs_remaining = 2 := by
  sorry

end antonov_packs_remaining_l608_608509


namespace vector_subtraction_l608_608279

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (1, 2)
def scalar : ℝ := 2

theorem vector_subtraction : a - (scalar • b) = (1 : ℝ, 0 : ℝ) :=
by
  sorry

end vector_subtraction_l608_608279


namespace matrix_satisfies_eqn_l608_608217

open Matrix

noncomputable def matrixN : Matrix (Fin 2) (Fin 2) ℝ := ![![1, -10], ![0, 1]]

theorem matrix_satisfies_eqn : 
  let N := matrixN
  in N^3 - 3 • N^2 + 2 • N = ![![5, 10], ![0, 5]] :=
by
  sorry

end matrix_satisfies_eqn_l608_608217


namespace probability_divisible_by_4_l608_608815

theorem probability_divisible_by_4 : 
  let outcomes := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} 
  ∧ let divisible_by_4 := {n ∈ outcomes | n % 4 = 0}
  ∧ let a := arbitrary 10-sided die roll from outcomes
  ∧ let b := arbitrary 10-sided die roll from outcomes
  in 
  (a ∈ divisible_by_4) ∧ (b ∈ divisible_by_4) 
  = 1 / 25 :=
sorry

end probability_divisible_by_4_l608_608815


namespace always_possible_to_create_pairs_l608_608641

def number_of_pairs (a b : Nat) : Nat := Nat.min a b

theorem always_possible_to_create_pairs :
  ∀ (total_mittens blue_mittens green_mittens red_mittens right_mittens left_mittens : Nat),
  total_mittens = 30 →
  blue_mittens = 10 →
  green_mittens = 10 →
  red_mittens = 10 →
  right_mittens = 15 →
  left_mittens = 15 →
  (∃ (pairs : Nat), pairs >= 5).
Proof :=
by
  intros total_mittens blue_mittens green_mittens red_mittens right_mittens left_mittens
  intros h1 h2 h3 h4 h5 h6
  sorry

end always_possible_to_create_pairs_l608_608641


namespace Jenny_total_weight_l608_608664

theorem Jenny_total_weight :
  (∀ (w_bottle w_can bottles cans total_weight_cents cents_bottle cents_can : ℕ),
     w_bottle = 6 → w_can = 2 → cans = 20 → cents_bottle = 10 → cents_can = 3 → total_weight_cents = 160 →
     total_weight_cents = cans * cents_can + bottles * cents_bottle →
     bottles * w_bottle + cans * w_can = 100) := 
by { intros, sorry }

end Jenny_total_weight_l608_608664


namespace log_expression_eq_three_l608_608877

noncomputable def lg (x : ℝ) : ℝ := log x / log 10
noncomputable def log_2 (x : ℝ) : ℝ := log x / log 2

theorem log_expression_eq_three :
  lg (5/2) + log_2 21 - (1/2)⁻¹ + 8^(2/3) = 3 := by 
  sorry

end log_expression_eq_three_l608_608877


namespace scatter_plot_can_be_made_l608_608917

theorem scatter_plot_can_be_made
    (data : List (ℝ × ℝ)) :
    ∃ (scatter_plot : List (ℝ × ℝ)), scatter_plot = data :=
by
  sorry

end scatter_plot_can_be_made_l608_608917


namespace part_I_part_II_l608_608373

open Real

noncomputable def f (x : ℝ) : ℝ := log ((2 / (x + 1)) - 1)

def g (x a : ℝ) : ℝ := -x^2 + 2 * x + a

-- Domain of function f
def A : Set ℝ := {x | -1 < x ∧ x < 1}

-- Range of function g with a given condition on x
def B (a : ℝ) : Set ℝ := {y | ∃ x, 0 ≤ x ∧ x ≤ 3 ∧ y = g x a}

theorem part_I : f (1 / 2015) + f (-1 / 2015) = 0 := sorry

theorem part_II (a : ℝ) : (A ∩ B a) = ∅ ↔ a ≤ -2 ∨ a ≥ 4 := sorry

end part_I_part_II_l608_608373


namespace sqrt_of_9_eq_3_l608_608731

theorem sqrt_of_9_eq_3 : Real.sqrt 9 = 3 := 
by 
  sorry

end sqrt_of_9_eq_3_l608_608731


namespace infinite_fractions_between_one_sixth_and_five_sixth_l608_608911

theorem infinite_fractions_between_one_sixth_and_five_sixth :
  ∀ q ∈ ℚ, (q > 1/6 ∧ q < 5/6) → ∃ r ∈ ℚ, (r > 1/6 ∧ r < 5/6 ∧ r ≠ q) :=
by
  sorry

end infinite_fractions_between_one_sixth_and_five_sixth_l608_608911


namespace three_equal_mass_piles_l608_608557

theorem three_equal_mass_piles (n : ℕ) (h : n > 3) : 
  (∃ (A B C : Finset ℕ), 
    (A ∪ B ∪ C = Finset.range (n + 1)) ∧ 
    (A ∩ B = ∅) ∧ 
    (A ∩ C = ∅) ∧ 
    (B ∩ C = ∅) ∧ 
    (A.sum id = B.sum id) ∧ 
    (B.sum id = C.sum id)) 
  ↔ (n % 3 = 0 ∨ n % 3 = 2) :=
sorry

end three_equal_mass_piles_l608_608557


namespace correct_statement_l608_608504

theorem correct_statement (a b m : ℝ) :
  (am^2 > bm^2 → a > b) :=
sorry

end correct_statement_l608_608504


namespace domain_of_g_l608_608958

-- Define the function f and specify the domain of f(x+1)
def f : ℝ → ℝ := sorry
def domain_f_x_plus_1 : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3} -- Domain of f(x+1) is [-1, 3]

-- Define the definition of the function g where g(x) = f(x^2)
def g (x : ℝ) : ℝ := f (x^2)

-- Prove that the domain of g(x) is [-2, 2]
theorem domain_of_g : {x | -2 ≤ x ∧ x ≤ 2} = {x | ∃ (y : ℝ), (0 ≤ y ∧ y ≤ 4) ∧ (x = y ∨ x = -y)} :=
by 
  sorry

end domain_of_g_l608_608958


namespace jasmine_average_pace_l608_608343

-- Define the conditions given in the problem
def totalDistance : ℝ := 45
def totalTime : ℝ := 9

-- Define the assertion that needs to be proved
theorem jasmine_average_pace : totalDistance / totalTime = 5 :=
by sorry

end jasmine_average_pace_l608_608343


namespace probability_interval_l608_608583

theorem probability_interval (t : ℝ) (ht : t > 0) (x : ℝ) (hx : x ∈ set.Icc (-t) (4 * t)) :
  probability (x ∈ set.Icc (-t / 2) t | x ∈ set.Icc (-t) (4 * t)) = 3 / 10 :=
by
  sorry

end probability_interval_l608_608583


namespace radius_of_sphere_is_4_sqrt_5_l608_608864

noncomputable def radius_of_sphere_in_truncated_cone {r_bottom r_top height : ℝ} (r_bottom := 20) (r_top := 4) (height := 15) : ℝ :=
  let BC := r_bottom + r_top
  let HB := r_bottom - r_top
  let CH := real.sqrt (BC^2 - HB^2)
  1 / 2 * CH

theorem radius_of_sphere_is_4_sqrt_5 :
  radius_of_sphere_in_truncated_cone = 4 * real.sqrt 5 :=
sorry

end radius_of_sphere_is_4_sqrt_5_l608_608864


namespace math_proof_problem_l608_608794

theorem math_proof_problem (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3 * x₁ * y₁^2 = 2008)
  (h₂ : y₁^3 - 3 * x₁^2 * y₁ = 2007)
  (h₃ : x₂^3 - 3 * x₂ * y₂^2 = 2008)
  (h₄ : y₂^3 - 3 * x₂^2 * y₂ = 2007)
  (h₅ : x₃^3 - 3 * x₃ * y₃^2 = 2008)
  (h₆ : y₃^3 - 3 * x₃^2 * y₃ = 2007) :
  (1 - x₁ / y₁) * (1 - x₂ / y₂) * (1 - x₃ / y₃) = 4015 / 2008 :=
by sorry

end math_proof_problem_l608_608794


namespace integral_evaluation_l608_608898

noncomputable def integral_result : ℝ :=
  ∫ x in (0:ℝ)..(1:ℝ), (Real.sqrt (1 - x^2) - x)

theorem integral_evaluation :
  integral_result = (Real.pi - 2) / 4 :=
by
  sorry

end integral_evaluation_l608_608898


namespace A_work_days_l608_608048

variables (r_A r_B r_C : ℝ) (h1 : r_A + r_B = (1 / 3)) (h2 : r_B + r_C = (1 / 3)) (h3 : r_A + r_C = (5 / 24))

theorem A_work_days :
  1 / r_A = 9.6 := 
sorry

end A_work_days_l608_608048


namespace antonov_packs_remaining_l608_608508

theorem antonov_packs_remaining (total_candies : ℕ) (pack_size : ℕ) (packs_given : ℕ) (candies_remaining : ℕ) (packs_remaining : ℕ) :
  total_candies = 60 →
  pack_size = 20 →
  packs_given = 1 →
  candies_remaining = total_candies - pack_size * packs_given →
  packs_remaining = candies_remaining / pack_size →
  packs_remaining = 2 := by
  sorry

end antonov_packs_remaining_l608_608508


namespace min_pos_period_f_min_value_f_l608_608959

variable {a b x : ℝ}

-- Given condition: a^2 + b^2 = 1
axiom cond : a^2 + b^2 = 1

-- Function definition
def f (x : ℝ) : ℝ := a * (Real.cos x)^2 + b * (Real.sin x) * (Real.cos x) - a / 2 - 1

-- Lean 4 statements for the proof problem
theorem min_pos_period_f : ∀ x : ℝ, 0 < x → f(x) = f(x + π) := by
  sorry

theorem min_value_f : ∃ x : ℝ, f(x) = -3/2 := by
  sorry

end min_pos_period_f_min_value_f_l608_608959


namespace curve_transformation_l608_608652

theorem curve_transformation :
  ∀ (x y x' y' : ℝ),
  (y = (1/3) * cos (2 * x)) →
  (x' = 2 * x) →
  (y' = 3 * y) →
  y' = cos x' :=
by
  intros x y x' y' h1 h2 h3
  sorry

end curve_transformation_l608_608652


namespace count_between_2000_and_9999_with_conditions_l608_608283

noncomputable def countValidIntegers : Nat := 
  8 * 9 * 8 * 4.5

theorem count_between_2000_and_9999_with_conditions :
  countValidIntegers = 2592 :=
by
  sorry

end count_between_2000_and_9999_with_conditions_l608_608283


namespace A_completes_work_in_30_days_l608_608473

def work_rate_B : ℚ := 1 / 30
def work_rate_C : ℚ := 1 / 30  -- Use approximation as per problem statement
def work_done_by (rate : ℚ) (days : ℕ) : ℚ := rate * days

theorem A_completes_work_in_30_days :
  ∃ (x : ℚ), work_done_by (1 / x) 10 + work_done_by work_rate_B 10 + work_done_by work_rate_C 10 = 1 ∧ x = 30 :=
by {
  use 30,
  dsimp [work_done_by, work_rate_B, work_rate_C],
  norm_num,
  split,
  { ring },
  { refl }
}

end A_completes_work_in_30_days_l608_608473


namespace exists_large_N_with_good_numbers_l608_608241

def is_good (n k : ℕ) : Prop :=
  ∃ s, s.card ≥ 0.99 * n ∧ (∀ i ∈ s, i ≤ n ∧ (n.choose i) % k = 0)

theorem exists_large_N_with_good_numbers (k : ℕ) (h : k > 0) :
  ∃ N, ∃ T, T.card ≥ 0.99 * N ∧ (∀ n ∈ T, is_good n k) :=
begin
  sorry
end

end exists_large_N_with_good_numbers_l608_608241


namespace math_problem_l608_608619

theorem math_problem (a : ℝ) (h : a^2 - 4 * a + 3 = 0) (h_ne : a ≠ 2 ∧ a ≠ 3 ∧ a ≠ -3) :
  (9 - 3 * a) / (2 * a - 4) / (a + 2 - 5 / (a - 2)) = -3 / 8 :=
sorry

end math_problem_l608_608619


namespace circumcenter_lies_on_AK_l608_608833

noncomputable def is_circumcenter (O : Point) (A B C : Point) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C

noncomputable def lies_on_line (P Q R : Point) : Prop :=
  ∃ (k : ℝ), Q = P + k • (R - P)

theorem circumcenter_lies_on_AK
  (A B C L H K O : Point)
  (h_triangle : ∀ (X Y Z : Point), X ≠ Y → X ≠ Z → Y ≠ Z → is_triangle X Y Z)
  (h_AL : is_angle_bisector A L B C)
  (h_H : foot B L H)
  (h_K : foot_on_circumcircle B L K (set_circumcircle A B L))
  (h_circ_A : O = is_circumcenter O A B C) :
  lies_on_line A K O :=
sorry

end circumcenter_lies_on_AK_l608_608833


namespace time_to_cross_bridge_l608_608490

theorem time_to_cross_bridge (speed_km_per_hr : ℕ) (bridge_length_meters : ℕ) (conversion_rate_km_to_m : ℕ) (conversion_rate_hr_to_min : ℕ) :
    speed_km_per_hr = 6 → bridge_length_meters = 1500 → conversion_rate_km_to_m = 1000 → conversion_rate_hr_to_min = 60 →
    let speed_m_per_min := (speed_km_per_hr * conversion_rate_km_to_m) / conversion_rate_hr_to_min in
    let time_to_cross := bridge_length_meters / speed_m_per_min in
    time_to_cross = 15 :=
by
  intros h_speed h_length h_conversion_km h_conversion_hr
  let speed_m_per_min := (speed_km_per_hr * conversion_rate_km_to_m) / conversion_rate_hr_to_min
  have h_speed_m_per_min : speed_m_per_min = 100 := by
    calc
      speed_m_per_min = (speed_km_per_hr * conversion_rate_km_to_m) / conversion_rate_hr_to_min : rfl
      ... = (6 * 1000) / 60 : by rw [h_speed, h_conversion_km, h_conversion_hr]
      ... = 6000 / 60 : rfl
      ... = 100 : rfl
  let time_to_cross := bridge_length_meters / speed_m_per_min
  have h_time_to_cross : time_to_cross = 15 := by
    calc
      time_to_cross = bridge_length_meters / speed_m_per_min : rfl
      ... = 1500 / 100 : by rw [h_length, h_speed_m_per_min]
      ... = 15 : rfl
  exact h_time_to_cross

end time_to_cross_bridge_l608_608490


namespace AC_length_l608_608649

noncomputable def length_AC (AB DC AD : ℝ) : ℝ :=
  let BD := Real.sqrt (AB^2 - AD^2)
  let BC := Real.sqrt (DC^2 - BD^2)
  let AE := AD + BC
  Real.sqrt (AE^2 + BD^2)

theorem AC_length (AB DC AD : ℝ) (h_AB : AB = 16) (h_DC : DC = 24) (h_AD : AD = 7) :
  (Real.toFloat (length_AC AB DC AD)).round toNearestTenth = 26.0 :=
by
  have h1: length_AC 16 24 7 = Real.sqrt ((7 + Real.sqrt (24^2 - (Real.sqrt (16^2 - 7^2))^2))^2 + (Real.sqrt (16^2 - 7^2))^2)
    by
      rw [h_AB, h_DC, h_AD]
      simp
  rw [h1]
  norm_num
  sorry

end AC_length_l608_608649


namespace converse_equivalence_l608_608737

-- Definition of the original proposition
def original_proposition : Prop := ∀ (x : ℝ), x < 0 → x^2 > 0

-- Definition of the converse proposition
def converse_proposition : Prop := ∀ (x : ℝ), x^2 > 0 → x < 0

-- Theorem statement asserting the equivalence
theorem converse_equivalence : (converse_proposition = ¬ original_proposition) :=
sorry

end converse_equivalence_l608_608737


namespace range_of_a_l608_608955

noncomputable def f (x : ℝ) : ℝ := x + Real.log x
noncomputable def g (a x : ℝ) : ℝ := a * x - 2 * Real.sin x

theorem range_of_a (a : ℝ) :
  (∀ x₁ > 0, ∃ x₂, (1 + 1 / x₁) * (a - 2 * Real.cos x₂) = -1) →
  -2 ≤ a ∧ a ≤ 1 :=
by {
  sorry
}

end range_of_a_l608_608955


namespace hall_length_width_difference_l608_608454

variable (L W : ℕ)

theorem hall_length_width_difference (h₁ : W = 1 / 2 * L) (h₂ : L * W = 800) :
  L - W = 20 :=
sorry

end hall_length_width_difference_l608_608454


namespace intervals_of_monotonicity_range_of_a_l608_608932

noncomputable def f (a x : ℝ) : ℝ := x * Real.exp x + a * x^2 - x

-- Part I: Intervals of Monotonicity when a = -1/2
theorem intervals_of_monotonicity (x : ℝ) :
  (∀ x, f (-1/2) x < f (-1/2) (x + 1) ∧ f (-1/2) x > f (-1/2) (x - 1)) → 
  ∃ a b, (-∞ < a) ∧ (a < -1) ∧ 
         (0 < b) ∧ (b < +∞) :=
sorry

-- Part II: Range of a
theorem range_of_a (x : ℝ) :
  (∀ x ≥ 0, f' a x - f a x ≥ (4 * a + 1) * x) → 
  a ∈ Icc (-∞) (1/2) :=
sorry

end intervals_of_monotonicity_range_of_a_l608_608932


namespace probability_x_plus_y_lt_4_l608_608111

theorem probability_x_plus_y_lt_4 :
  let square_vertices := {(0,0), (0,3), (3,3), (3,0)} in
  let point_in_square (x y : ℝ) := 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 in
  let probability (A : ℝ) (B : ℝ) := A / B in
  ∃ (P : ℝ), P = probability 7 9 ∧
             P = (measure (λ (x y : ℝ), point_in_square x y ∧ x + y < 4)) / (measure (λ (x y : ℝ), point_in_square x y)) :=
sorry

end probability_x_plus_y_lt_4_l608_608111


namespace geometric_sequence_sum_l608_608935

theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (h_geo : ∀ n, a (n + 1) = (3 : ℝ) * ((-2 : ℝ) ^ n))
  (h_first : a 1 = 3)
  (h_ratio_ne_1 : -2 ≠ 1)
  (h_arith : 2 * a 3 = a 4 + a 5) :
  a 1 + a 2 + a 3 + a 4 + a 5 = 33 := 
sorry

end geometric_sequence_sum_l608_608935


namespace missing_angle_in_polygon_l608_608079

def sum_of_interior_angles (n : ℕ) : ℚ := 180 * (n - 2)

theorem missing_angle_in_polygon (n : ℕ) (sum_known : ℚ) (h1 : n = 18) (h2 : sum_known = 2700) :
  sum_of_interior_angles n - sum_known = 180 :=
by
  subst h1
  subst h2
  unfold sum_of_interior_angles
  norm_num
  sorry

end missing_angle_in_polygon_l608_608079


namespace original_side_length_l608_608448

theorem original_side_length (x : ℝ) (h1 : (x - 6) * (x - 5) = 120) : x = 15 :=
sorry

end original_side_length_l608_608448


namespace part_I_part_II_l608_608247

-- Define the conditions for Part Ⅰ
def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def eccentricity (a c : ℝ) : ℝ :=
  c / a

def perimeter_condition (a c : ℝ) : ℝ :=
  2 * a + 2 * c

theorem part_I : 
  ∀ (a b c : ℝ), 
    a > b ∧ b > 0 ∧ eccentricity a c = (sqrt 3) / 2 ∧ perimeter_condition a c = 8 + 4 * sqrt 3 
    → ellipse_equation 4 2 := 
sorry

-- Define the conditions for Part Ⅱ
def point (x y : ℝ) := (x, y)
def chord_length (A Q B : ℝ × ℝ) : ℝ :=
  dist A B 

theorem part_II :
  ∀ (a : ℝ) (A B Q : ℝ × ℝ),
    Q = (0, -3 * sqrt 3 / 2) ∧ 
    (A = (-a, 0)) ∧ 
    chord_length A Q B ∈ [8, sqrt 39, sqrt 7] :=
sorry

end part_I_part_II_l608_608247


namespace weight_of_3_moles_HClO2_correct_l608_608437

def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cl : ℝ := 35.453
def atomic_weight_O : ℝ := 15.999

def molecular_weight_HClO2 : ℝ := (1 * atomic_weight_H) + (1 * atomic_weight_Cl) + (2 * atomic_weight_O)
def weight_of_3_moles_HClO2 : ℝ := 3 * molecular_weight_HClO2

theorem weight_of_3_moles_HClO2_correct : weight_of_3_moles_HClO2 = 205.377 := by
  sorry

end weight_of_3_moles_HClO2_correct_l608_608437


namespace unit_square_coloring_l608_608021

noncomputable def is_red : Point → Prop := sorry
noncomputable def is_blue : Point → Prop := sorry

theorem unit_square_coloring (points : Set Point) (colored : ∀ p ∈ points, is_red p ∨ is_blue p) :
  (∃ square : UnitSquare, ∀ v ∈ square.vertices, is_blue v) ∨ 
  (∃ square : UnitSquare, (∃ v1 v2 v3 ∈ square.vertices, is_red v1 ∧ is_red v2 ∧ is_red v3)) :=
sorry

end unit_square_coloring_l608_608021


namespace simplify_fraction_l608_608722

theorem simplify_fraction (a : ℝ) (h : a = 2) : (24 * a^5) / (72 * a^3) = 4 / 3 := by
  sorry

end simplify_fraction_l608_608722


namespace find_p_l608_608938

-- Define the problem context
variables {n : ℕ} {p : ℚ} (X : Type) [Distribution.Binomial X] (E D : X → ℚ)

-- Conditions
axiom expectation_condition : E X = 30
axiom variance_condition : D X = 20

-- The statement to prove
theorem find_p : p = 1 / 3 :=
by
  -- Proof to be filled in
  sorry

end find_p_l608_608938


namespace probability_x_plus_y_lt_4_l608_608108

theorem probability_x_plus_y_lt_4 :
  let square_vertices := {(0,0), (0,3), (3,3), (3,0)} in
  let point_in_square (x y : ℝ) := 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 in
  let probability (A : ℝ) (B : ℝ) := A / B in
  ∃ (P : ℝ), P = probability 7 9 ∧
             P = (measure (λ (x y : ℝ), point_in_square x y ∧ x + y < 4)) / (measure (λ (x y : ℝ), point_in_square x y)) :=
sorry

end probability_x_plus_y_lt_4_l608_608108


namespace transform_curve_l608_608655

theorem transform_curve (x y x' y' : ℝ)
  (h1 : x' = 2 * x)
  (h2 : y' = 3 * y)
  (h3 : y = (1 / 3) * cos (2 * x)) :
  y' = cos x' :=
by
  -- transform and substitute
  sorry

end transform_curve_l608_608655


namespace work_completion_time_l608_608051

theorem work_completion_time 
  (M W : ℝ) 
  (h1 : (10 * M + 15 * W) * 6 = 1) 
  (h2 : M * 100 = 1) 
  : W * 225 = 1 := 
by
  sorry

end work_completion_time_l608_608051


namespace new_class_average_l608_608309

noncomputable def class_average (n1 n2 s1 s2 : ℕ) :=
  (n1 * s1 + n2 * s2) / (n1 + n2 : ℕ)

theorem new_class_average :
  let n1 := 45
  let n2 := 5
  let s1 := 68
  let s2 := 82
  class_average n1 n2 s1 s2 = 69.4 :=
by
  -- Here we are supposed to prove the theorem
  -- You can add the steps of the proof here if necessary
  sorry

end new_class_average_l608_608309


namespace min_students_for_matching_numbers_l608_608843

theorem min_students_for_matching_numbers :
  ∀ (n_cards : finset ℕ) (cards : ℕ), 
  (9 + 27 = 36) → ∀ n_students : ℕ, n_students = 73 :=
begin
  sorry
end

end min_students_for_matching_numbers_l608_608843


namespace water_transfer_l608_608006

-- Define the conditions
variables (left_initial right_initial left_final right_final x : ℕ)

-- The problem conditions
def left_initial_quantity := 2800
def right_initial_quantity := 1500
def final_difference := 360

-- Translation of the conditions in Lean, stating the core assumption of the problem
def condition_holds : Prop :=
  left_final = left_initial_quantity - x ∧
  right_final = right_initial_quantity + x ∧
  left_final = right_final + final_difference

-- The proof goal
theorem water_transfer (h : condition_holds) : x = 470 :=
  sorry

end water_transfer_l608_608006


namespace rectangle_sides_equal_perimeter_and_area_l608_608765

theorem rectangle_sides_equal_perimeter_and_area (x y : ℕ) (h : 2 * x + 2 * y = x * y) : 
    (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 6) ∨ (x = 4 ∧ y = 4) :=
by sorry

end rectangle_sides_equal_perimeter_and_area_l608_608765


namespace area_triangle_l608_608415

theorem area_triangle
  (P : ℝ × ℝ)
  (hP : P = (1, 2))
  (exists_lines : ∃ (l : ℝ → ℝ), l ∈ {f | (∃ k : ℝ, f = λ x, k * (x - 1) + 2) ∧ (∃ x, f x = 0) ∧ (∃ y, ∃ k, f k = y) ∧ triangle_area P f = S})
  (lines_count : ∃ k : ℝ, (∃ S : ℝ, k^2 - (4 + 2 * S) * k + 4 = 0 ∨ k^2 - (4 - 2 * S) * k + 4 = 0) ∧ ∃ S : ℝ, (∃ x : ℝ, x + (k - 2)/k = -4) ∧ ∃ y : ℝ, intersection_y k y = 0)
  : S = 4 :=
sorry

end area_triangle_l608_608415


namespace range_a_range_p_l608_608597

noncomputable def f (x a : ℝ) : ℝ := ln x + 1 / (a * x) - 1 / a

theorem range_a (a : ℝ) : (0 < a) → (1 ≤ 1 / a) ∨ (a < 0) :=
by
  simp [f]
  sorry

noncomputable def g (x p : ℝ) : ℝ := exp x - x + p

theorem range_p (p : ℝ) (h : ∀ (x : ℝ), x ∈ set.Icc 1 (exp 1) → g x p ≥ (ln x - 1) * exp x + x) :
    p ≥ 1 - exp 1 :=
by
  simp [g]
  sorry

end range_a_range_p_l608_608597


namespace beads_in_each_part_after_doubling_l608_608143

theorem beads_in_each_part_after_doubling (blue_beads : ℕ) (yellow_beads : ℕ) (div_parts : ℕ) (remove_from_each_part : ℕ) :
  blue_beads = 23 → yellow_beads = 16 → div_parts = 3 → remove_from_each_part = 10 →
  let total_beads := blue_beads + yellow_beads in
  let beads_per_part := total_beads / div_parts in
  let beads_after_removal := beads_per_part - remove_from_each_part in
  let final_beads_per_part := beads_after_removal * 2 in
  final_beads_per_part = 6 := 
begin
  intros h1 h2 h3 h4,
  simp [h1, h2, h3, h4],
  norm_num,
end

end beads_in_each_part_after_doubling_l608_608143


namespace biotechnology_incorrect_option_l608_608503

open Set

def option_A_incorrect : Prop :=
  "In the experiment of separation and counting of bacteria that decompose urea in soil, the dilution plating method should be used, not the streak plate method."

def option_B_correct : Prop :=
  "When making fruit vinegar, sterile air should be continuously introduced because Acetobacter is an aerobic heterotroph."

def option_C_correct : Prop :=
  "A standard color-developing solution needs to be prepared to detect the content of nitrite by colorimetry."

def option_D_correct : Prop :=
  "DNA can be extracted by utilizing the different solubilities of DNA and proteins in NaCl solutions of varying concentrations."

theorem biotechnology_incorrect_option :
  option_A_incorrect :=
by
  sorry -- Proof skipped

end biotechnology_incorrect_option_l608_608503


namespace antonov_candy_packs_l608_608510

theorem antonov_candy_packs (bought_candies : ℕ) (cando_per_pack : ℕ) (gave_to_sister : ℕ) (h_bought : bought_candies = 60) (h_pack : cando_per_pack = 20) (h_gave : gave_to_sister = 20) :
  (bought_candies - gave_to_sister) / cando_per_pack = 2 :=
by
  rw [h_bought, h_pack, h_gave]
  norm_num
  sorry

end antonov_candy_packs_l608_608510


namespace min_f_value_l608_608910

noncomputable def f (x : ℝ) : ℝ := 
  abs (sin x + cos x + tan x + cot x + sec x + csc x)

theorem min_f_value : ∃ x : ℝ, f x = 2 * sqrt 2 - 1 :=
sorry

end min_f_value_l608_608910


namespace alpha_even_function_l608_608563

theorem alpha_even_function :
  ∃ α : ℤ, 0 ≤ α ∧ α ≤ 5 ∧ (∀ x : ℝ, (x ^ (3 - α) = (-x) ^ (3 - α))) ∧ α = 1 :=
begin
  sorry
end

end alpha_even_function_l608_608563


namespace island_area_50_l608_608152

-- Definition for the width of the island
def width (island : Type) [IsIsland island] : ℝ := 5

-- Definition for the length of the island
def length (island : Type) [IsIsland island] : ℝ := 10

-- Definition for the area of the island
def area (island : Type) [IsIsland island] : ℝ :=
  length island * width island

-- Proof statement that the area of the island is 50 square miles
theorem island_area_50 (island : Type) [IsIsland island] : area island = 50 := 
by {
  sorry -- proof to be completed
}

end island_area_50_l608_608152


namespace ratio_of_areas_is_one_l608_608687

variables {A B C B' C' : Type} [HasArea ℝ]

-- Given conditions
axiom right_triangle (h : triangle A B C) : right_angle A B C
  
axiom reflection_B_ac (h : triangle A B C) : reflection_point B A C B'
axiom reflection_C_ab' (h : triangle A B' C) : reflection_point C A B' C'

-- Theorem statement
theorem ratio_of_areas_is_one (h_triangle : right_triangle (triangle A B C))
           (h_B_reflection : reflection_B_ac (triangle A B C))
           (h_C_reflection : reflection_C_ab' (triangle A B' C)) :
    area (triangle B C B') / area (triangle B C' B') = 1 :=
sorry

end ratio_of_areas_is_one_l608_608687


namespace maria_took_out_fish_l608_608377

theorem maria_took_out_fish (initial_fish current_fish took_out_fish : ℕ) (h1 : initial_fish = 19) (h2 : current_fish = 3) :
  took_out_fish = initial_fish - current_fish :=
by
  rw [h1, h2]
  exact rfl

end maria_took_out_fish_l608_608377


namespace unique_triple_solution_l608_608903

theorem unique_triple_solution (x y z : ℝ) :
  x = y^3 + y - 8 ∧ y = z^3 + z - 8 ∧ z = x^3 + x - 8 → (x, y, z) = (2, 2, 2) :=
by
  sorry

end unique_triple_solution_l608_608903


namespace pascal_triangle_odd_entries_256_l608_608214

-- Each row n in Pascal's Triangle has an odd number of entries based on the power of two in its binary representation
def count_odd_entries (n : ℕ) : ℕ :=
  2 ^ (nat.popcount n)

-- Pascal's Triangle row count theorem
theorem pascal_triangle_odd_entries_256 :
  (finset.filter (λ n, count_odd_entries n = 256) (finset.range 2010)).card = 150 := sorry

end pascal_triangle_odd_entries_256_l608_608214


namespace min_omega_l608_608411

def sin_phase_shift_min_omega (ω : ℝ) (φ : ℝ) : Prop :=
  (sin φ = -1/2) ∧ (abs φ < π / 2) ∧
  (∀ x, sin (ω * (x - π / 3) + φ) = sin (ω * x + ω * (-π / 3) + φ)) ∧
  (∃ k : ℤ, ω = -3 * k - 1/2)

theorem min_omega (ω : ℝ) (φ : ℝ) (k : ℤ) :
  sin_phase_shift_min_omega ω φ → 
  ω = -3 * k - 1/2 ∧ ω > 0 → 
  ω = 5/2 :=
sorry

end min_omega_l608_608411


namespace julie_remaining_hours_l608_608819

-- Define the conditions
def Julie_time : ℕ := 10
def Ted_time : ℕ := 8
def together_time : ℕ := 4

-- Conclude the final question and answer statement
theorem julie_remaining_hours (h1 : Julie_time = 10) (h2 : Ted_time = 8) (h3 : together_time = 4) : 1 :=
begin
  -- start the proof block
  sorry -- Proof skipped
end

end julie_remaining_hours_l608_608819


namespace sum_of_sequence_l608_608574

theorem sum_of_sequence : 
  let a : ℕ → ℚ := λ n, if n = 1 then 1/3 else if n = 2 then 2/3 else 0
  let S : ℕ → ℚ := λ n, ∑ k in finset.range (n + 1), a k
  (∀ n : ℕ, a (n + 2) - a (n + 1) = (-1)^(n + 1) * (a (n + 1) - a n)) → S 2018 = 1345 := 
by 
  sorry

end sum_of_sequence_l608_608574


namespace greatest_base7_product_l608_608436

theorem greatest_base7_product (n : ℕ) (h₀ : n < 2300) : 
  ∃ product : ℕ, (∀ d_list, digits_in_base_seven n d_list → product ≤ (digit_product d_list)) ∧ product = 1080 := sorry

end greatest_base7_product_l608_608436


namespace determine_x_l608_608080

def R := 10
def H := 5
def V (r : ℝ) (h : ℝ) : ℝ := Real.pi * r^2 * h

theorem determine_x
(val_x : ℝ)
(h : Real.pi * (R + val_x)^2 * H = 2 * (Real.pi * R^2 * (H + val_x) - Real.pi * R^2 * H)) :
  val_x = 20 := by
  sorry

end determine_x_l608_608080


namespace percentage_error_correct_l608_608325

-- Define the actual dimensions
variables (L W H : ℝ)

-- Define the measured dimensions
noncomputable def L_measured := 1.20 * L
noncomputable def W_measured := 0.90 * W
noncomputable def H_measured := 1.15 * H

-- Define the actual volume
noncomputable def V_actual := L * W * H

-- Define the calculated volume
noncomputable def V_calculated := L_measured * W_measured * H_measured

-- Define the ratio of calculated volume to actual volume
noncomputable def volume_ratio := V_calculated / V_actual

-- Define the percentage error
noncomputable def percentage_error := (volume_ratio - 1) * 100

-- The theorem to prove
theorem percentage_error_correct : percentage_error L W H = 24.2 := 
by sorry

end percentage_error_correct_l608_608325


namespace cos_alpha_minus_beta_l608_608172

theorem cos_alpha_minus_beta (α β : ℝ) 
  (h1 : cos α + cos β = -4 / 5) 
  (h2 : sin α + sin β = 1 / 3) : 
cos (α - β) = -28 / 225 :=
by
  -- The proof goes here
  sorry

end cos_alpha_minus_beta_l608_608172


namespace inequality_solution_set_l608_608421

theorem inequality_solution_set (x : ℝ) :
  (2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2 := by
sorry

end inequality_solution_set_l608_608421


namespace distance_between_points_AB_l608_608590

-- Definitions for the conditions
def pointA := 3
def pointB_candidates : Set ℤ := {9, -9}
def distance_to_origin (x : ℤ) : ℤ := abs x

-- The theorem statement
theorem distance_between_points_AB : 
  ∀ (B : ℤ), B ∈ pointB_candidates → (abs (pointA - B) = 6 ∨ abs (pointA - B) = 12) := 
by 
  intro B hB
  unfold pointA distance_to_origin pointB_candidates at *
  sorry

end distance_between_points_AB_l608_608590


namespace parallelogram_cyclic_points_l608_608043

variable {α : Type*}

structure Parallelogram (α : Type*) :=
(AE HF EH FA : α)
(is_parallelogram : AE * HF = EH * FA)

structure IsCircumcircle (α : Type*) :=
(X Y : α)
(is_on_circumcircle : ∀ A B C : α, circumcenter A B C ∈ X ∧ circumcenter A B C ∈ Y)

theorem parallelogram_cyclic_points {α : Type*} 
  (AEHF : Parallelogram α) 
  (X Y : α) 
  (hX : IsCircumcircle α X) 
  (hY : IsCircumcircle α Y) :
  ∃ O : α, midpoint O X ∧ midpoint O Y ∧ orthocenter O =
  sorry

end parallelogram_cyclic_points_l608_608043


namespace function_zero_if_condition_l608_608902

noncomputable def f : ℝ → ℝ := sorry

theorem function_zero_if_condition :
  (∀ x y : ℝ, f (x + y) = f (x - y) + f (f (1 - x * y))) → (∀ x : ℝ, f x = 0) :=
begin
  intro h,
  sorry
end

end function_zero_if_condition_l608_608902


namespace just_cool_numbers_count_l608_608020

-- Definitions based on conditions
def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def is_just_cool (n : ℕ) : Prop :=
  -- n is a six-digit number
  100000 ≤ n ∧ n < 1000000 ∧
  -- n is odd
  is_odd n ∧
  -- All digits are prime
  (∀ i, 0 ≤ i < 6 → is_prime_digit (n / 10^i % 10)) ∧
  -- No two identical digits are next to each other
  (∀ i, 0 ≤ i < 5 → (n / 10^i % 10) ≠ (n / 10^(i+1) % 10))
  
theorem just_cool_numbers_count : 
  (finset.filter is_just_cool (finset.Icc 100000 999999)).card = 729 :=
by sorry

end just_cool_numbers_count_l608_608020


namespace mittens_pairing_possible_l608_608638

/--
In a kindergarten's lost and found basket, there are 30 mittens: 
10 blue, 10 green, 10 red, 15 right-hand, and 15 left-hand. 

Prove that it is always possible to create matching pairs of one right-hand 
and one left-hand mitten of the same color for 5 children.
-/
theorem mittens_pairing_possible : 
  (∃ (right_blue left_blue right_green left_green right_red left_red : ℕ), 
    right_blue + left_blue + right_green + left_green + right_red + left_red = 30 ∧
    right_blue ≤ 10 ∧ left_blue ≤ 10 ∧
    right_green ≤ 10 ∧ left_green ≤ 10 ∧
    right_red ≤ 10 ∧ left_red ≤ 10 ∧
    right_blue + right_green + right_red = 15 ∧
    left_blue + left_green + left_red = 15) →
  (∃ right_blue left_blue right_green left_green right_red left_red,
    min right_blue left_blue + 
    min right_green left_green + 
    min right_red left_red ≥ 5) :=
sorry

end mittens_pairing_possible_l608_608638


namespace compute_difference_of_squares_l608_608524

theorem compute_difference_of_squares :
  262^2 - 258^2 = 2080 := 
by
  sorry

end compute_difference_of_squares_l608_608524


namespace circumcenter_lies_on_AK_l608_608834

noncomputable def is_circumcenter (O : Point) (A B C : Point) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C

noncomputable def lies_on_line (P Q R : Point) : Prop :=
  ∃ (k : ℝ), Q = P + k • (R - P)

theorem circumcenter_lies_on_AK
  (A B C L H K O : Point)
  (h_triangle : ∀ (X Y Z : Point), X ≠ Y → X ≠ Z → Y ≠ Z → is_triangle X Y Z)
  (h_AL : is_angle_bisector A L B C)
  (h_H : foot B L H)
  (h_K : foot_on_circumcircle B L K (set_circumcircle A B L))
  (h_circ_A : O = is_circumcenter O A B C) :
  lies_on_line A K O :=
sorry

end circumcenter_lies_on_AK_l608_608834


namespace forces_angle_result_l608_608435

noncomputable def forces_angle_condition (p1 p2 p : ℝ) (α : ℝ) : Prop :=
  p^2 = p1 * p2

noncomputable def angle_condition_range (p1 p2 : ℝ) : Prop :=
  (3 - Real.sqrt 5) / 2 ≤ p1 / p2 ∧ p1 / p2 ≤ (3 + Real.sqrt 5) / 2

theorem forces_angle_result (p1 p2 p α : ℝ) (h : forces_angle_condition p1 p2 p α) :
  120 * π / 180 ≤ α ∧ α ≤ 120 * π / 180 ∧ (angle_condition_range p1 p2) := 
sorry

end forces_angle_result_l608_608435


namespace bales_in_barn_l608_608009

theorem bales_in_barn (stacked today total original : ℕ) (h1 : stacked = 67) (h2 : total = 89) (h3 : total = stacked + original) : original = 22 :=
by
  sorry

end bales_in_barn_l608_608009


namespace probability_x_plus_y_lt_4_l608_608115

open Set

-- Define the square and the line
def square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 }
def line_lt_4 : Set (ℝ × ℝ) := { p | p.1 + p.2 < 4 }

-- The probability to prove
theorem probability_x_plus_y_lt_4 : 
  (volume (square ∩ line_lt_4) / volume square) = 7 / 9 := 
sorry

end probability_x_plus_y_lt_4_l608_608115


namespace trajectory_of_point_M_l608_608741

noncomputable def dist_point_to_point (M : ℝ × ℝ) (F : ℝ × ℝ) : ℝ :=
  real.sqrt ((M.1 - F.1)^2 + (M.2 - F.2)^2)

noncomputable def dist_point_to_line (M : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  real.abs ((a * M.1 + b * M.2 + c) / real.sqrt (a^2 + b^2))

theorem trajectory_of_point_M (M : ℝ × ℝ) :
  dist_point_to_point M (4, 0) = dist_point_to_line M 1 0 (-5) - 1 →
  M.2^2 = 16 * M.1 :=
by
  intros h
  sorry

end trajectory_of_point_M_l608_608741


namespace max_value_ln_plus_x_l608_608928

noncomputable def f (x : ℝ) : ℝ := Real.log x + x

theorem max_value_ln_plus_x (h₁ : 1 ≤ x) (h₂ : x ≤ Real.exp 1) : 
    ∃ x_max, x_max = Real.exp 1 ∧ f (Real.exp 1) = 1 + Real.exp 1 :=
by
  let x_max := Real.exp 1
  have h₃ : x_max = Real.exp 1 := rfl
  have h₄ : f x_max = 1 + Real.exp 1 := by
    have := Real.log (Real.exp 1)
    rw [this]
    simp
  use x_max
  constructor
  · exact h₃
  · exact h₄
  sorry

end max_value_ln_plus_x_l608_608928


namespace least_sugar_l608_608166

theorem least_sugar (f s : ℚ) (h1 : f ≥ 10 + 3 * s / 4) (h2 : f ≤ 3 * s) :
  s ≥ 40 / 9 :=
  sorry

end least_sugar_l608_608166


namespace distinct_product_zero_l608_608341

theorem distinct_product_zero (x : Fin 100 → ℤ) (h : ∑ i : Fin 100, (1 : ℝ) / Real.sqrt (x i) = 20) : 
  ∏ (i j : Fin 100) (h : i ≠ j), (x i - x j) = 0 :=
by
  sorry

end distinct_product_zero_l608_608341


namespace increasing_functions_l608_608149

open Real

variable (x : ℝ)

theorem increasing_functions : 
  (∀ x ∈ Ioo 0 (π / 2), deriv (λ x, sin x - cos x) x > 0) ∧ 
  (∀ x ∈ Ioo 0 (π / 2), deriv (λ x, (sin x / cos x)) x > 0) := 
by
  sorry

end increasing_functions_l608_608149


namespace lightsaber_ratio_l608_608665

theorem lightsaber_ratio (T L : ℕ) (hT : T = 1000) (hTotal : L + T = 3000) : L / T = 2 :=
by
  sorry

end lightsaber_ratio_l608_608665


namespace not_divisible_by_121_l608_608391

theorem not_divisible_by_121 (n : ℤ) : ¬ (121 ∣ (n^2 + 2 * n + 12)) :=
sorry

end not_divisible_by_121_l608_608391


namespace painters_days_calculation_l608_608541

theorem painters_days_calculation : 
  (∀ (P: ℕ) (D: ℕ) (W: ℕ), 
    P = 8 ∧ D = 25/10 ∧ W = 20 → 
    ∀ (N: ℕ) (D: ℕ) (R: ℕ), 
    N = 6 ∧ R = 0 ∧ D * 6 = 20 ∧ D <= 5 ∧ D + R = 4 → 
    (N * D + R = W / P)) → 4 := 
sorry

end painters_days_calculation_l608_608541


namespace carpet_shaded_area_l608_608135

theorem carpet_shaded_area (S T : ℝ) 
  (h1 : 12 / S = 4) 
  (h2 : S / T = 2) 
  (h3 : S * S = 9) -- Area of big square
  (h4 : 16 * (T * T) = 36) -- Total area of small squares
  : S * S + 16 * (T * T) = 45 := by 
s
  have hS : S = 3 := by
    calc
      S = 12 / 4 : by rw [←h1, div_mul_cancel _ (ne_of_gt (zero_lt_four))]
      S = 3 : by norm_num
  have hT : T = 1.5 := by
    calc
      T = S / 2 : by rw [←h2, mul_div_cancel' _ (ne_of_gt zero_lt_two)]
      T = 1.5 : by rw [hS]; norm_num
  have h_big: S * S = 9 := by
    calc 
      S * S = 3 * 3 : by rw [hS]
      S * S = 9 : by norm_num
  have h_small: 16 * (T * T) = 36 := by
    calc 
      T * T = 1.5 * 1.5 : by rw [hT]
      T * T = 2.25 : by norm_num
      16 * (T * T) = 36 : by norm_num
  rw [h_small, h_big]
  done

end carpet_shaded_area_l608_608135


namespace sequence_sum_property_l608_608275

/-- Given the sequence {a_n} that satisfies a₁ + a₃ = 5/8 and a_{n+1} = 2aₙ, with the sum of the first n terms being Sₙ,
prove that Sₙ - 2aₙ = -1/8. -/
theorem sequence_sum_property (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) 
  (h1 : a 1 + a 3 = 5 / 8) 
  (h2 : ∀ n, a (n+1) = 2 * a n) 
  (h3 : S n = ∑ k in Finset.range n, a (k + 1)) : 
  S n - 2 * a n = -1 / 8 :=
sorry

end sequence_sum_property_l608_608275


namespace div_and_quot_l608_608874

theorem div_and_quot (d q : ℝ) : 100 = d * 16 + 4 → d = 6 ∧ q = 16.65 :=
by {
  intros h,
  sorry
}

end div_and_quot_l608_608874


namespace AlexSilverTokens_l608_608501

namespace TokenExchange

structure BoothExchanges where
  R : Nat
  B : Nat
  G : Nat
  S : Nat

def booth1 (exchanges : BoothExchanges) : BoothExchanges :=
  { exchanges with 
    R := exchanges.R - 3,
    B := exchanges.B + 2,
    S := exchanges.S + 1 }

def booth2 (exchanges : BoothExchanges) : BoothExchanges :=
  { exchanges with 
    B := exchanges.B - 4,
    R := exchanges.R + 1,
    G := exchanges.G + 1,
    S := exchanges.S + 1 }

def booth3 (exchanges : BoothExchanges) : BoothExchanges :=
  { exchanges with 
    G := exchanges.G - 5,
    R := exchanges.R + 3,
    S := exchanges.S + 1 }

def finalState : BoothExchanges :=
  { R := 100, B := 100, G := 50, S := 0 }

def exchangesPossible (exchanges : BoothExchanges) : Bool :=
  (exchanges.R ≥ 3 ∨ exchanges.B ≥ 4 ∨ exchanges.G ≥ 5)

def executeExchanges : BoothExchanges → BoothExchanges
  | s@(exchangesPossible) := 
      if s.R ≥ 3 then executeExchanges (booth1 s)
      else if s.B ≥ 4 then executeExchanges (booth2 s)
      else if s.G ≥ 5 then executeExchanges (booth3 s)
      else s
  | s := s

theorem AlexSilverTokens : 
    (executeExchanges finalState).S = 66 := sorry

end TokenExchange

end AlexSilverTokens_l608_608501


namespace point_on_circumcircle_of_ABC_l608_608339

variable {A B C D E J M N P : Type} 
variables [Incircle ABC AB AC touches D E] [Excenter ABC A] [Midpoint JD M] [Midpoint JE N] [Intersection BM CN P]

theorem point_on_circumcircle_of_ABC (h: Incircle ABC AB AC touches D E)
  (h1: Excenter_of_ABC A J)
  (h2: Midpoint JD M)
  (h3: Midpoint JE N)
  (h4: Intersection (Line B M) (Line C N) P):
  Lies_on_Circumcircle P ABC := 
sorry

end point_on_circumcircle_of_ABC_l608_608339


namespace proof_sin_2C_over_sin_B_eq_l608_608307

noncomputable def sin_2C_over_sin_B (a b c : ℝ) (angle_A : ℝ) (hb : b = 2) (hc : c = 3) (hA : angle_A = real.pi / 3) : ℝ :=
  (2 * real.sin (real.arcsin ((c * real.sqrt (1 - real.cos (real.arccos ((a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b)))))) * real.cos (real.arccos ((a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b))))) / real.sin (real.arcsin((b * real.sin angle_A) / a))

theorem proof_sin_2C_over_sin_B_eq :
  ∀ (a b c : ℝ) (angle_A : ℝ), b = 2 →
  c = 3 →
  angle_A = real.pi / 3 →
  a = real.sqrt (b^2 + c^2 - 2 * b * c * real.cos angle_A) →
  sin_2C_over_sin_B a b c angle_A = (3 * real.sqrt 7) / 14 :=
by
  intros a b c angle_A hb hc hA ha
  rw [hb, hc, hA, ha]
  sorry

end proof_sin_2C_over_sin_B_eq_l608_608307


namespace probability_of_x_plus_y_less_than_4_l608_608121

-- Define the square and the probability that x + y < 4 within this square.
theorem probability_of_x_plus_y_less_than_4 : 
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  (desired_area / square_area = 7 / 9) :=
by
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  show (desired_area / square_area = 7 / 9)
  sorry

end probability_of_x_plus_y_less_than_4_l608_608121


namespace football_team_practiced_hours_l608_608083

-- Define the daily practice hours and missed days as conditions
def daily_practice_hours : ℕ := 6
def missed_days : ℕ := 1

-- Define the total number of days in a week
def days_in_week : ℕ := 7

-- Define a function to calculate the total practiced hours in a week, 
-- given the daily practice hours, missed days, and total days in a week
def total_practiced_hours (daily_hours : ℕ) (missed : ℕ) (total_days : ℕ) : ℕ :=
  (total_days - missed) * daily_hours

-- Prove that the total practiced hours is 36
theorem football_team_practiced_hours :
  total_practiced_hours daily_practice_hours missed_days days_in_week = 36 := 
sorry

end football_team_practiced_hours_l608_608083


namespace concave_five_digit_count_l608_608082

def is_concave (a : Fin 6) (b : Fin 6) (c : Fin 6) (d : Fin 6) (e : Fin 6) : Prop :=
  a > b ∧ b > c ∧ c < d ∧ d < e

theorem concave_five_digit_count : 
  (Finset.univ.filter (λ l : (Fin 6 × Fin 6 × Fin 6 × Fin 6 × Fin 6), 
    is_concave l.1 l.2.1 l.2.2.1 l.2.2.2.1 l.2.2.2.2)).card = 146 :=
by {
  -- proof goes here
  sorry
}

end concave_five_digit_count_l608_608082


namespace probability_x_plus_y_lt_4_l608_608120

open Set

-- Define the square and the line
def square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 }
def line_lt_4 : Set (ℝ × ℝ) := { p | p.1 + p.2 < 4 }

-- The probability to prove
theorem probability_x_plus_y_lt_4 : 
  (volume (square ∩ line_lt_4) / volume square) = 7 / 9 := 
sorry

end probability_x_plus_y_lt_4_l608_608120


namespace circumcenter_on_AK_l608_608828

variable {α β γ : Real}
variable (A B C L H K O : Type)
variable [Triangle ABC] (circumcenter : Triangle ABC → Point O)
variable [AngleBisector A B C L]

theorem circumcenter_on_AK
  (h₁ : AL_is_angle_bisector ABC L)
  (h₂ : Height_from_B_on_AL B A L H)
  (h₃ : K_on_circumcircle_ABL B A L K)
  : Lies_on_line (circumcenter ABC) A K :=
sorry

end circumcenter_on_AK_l608_608828


namespace smaller_angle_at_340_pm_l608_608809

open Real

def degree_per_minute_minute_hand : ℝ := 6
def degree_per_minute_hour_hand : ℝ := 0.5

def angle_at_time (hour : ℕ) (minute : ℕ) : ℝ :=
  let minute_angle := minute * degree_per_minute_minute_hand
  let hour_angle := hour % 12 * 30 + minute * degree_per_minute_hour_hand
  abs (minute_angle - hour_angle)

theorem smaller_angle_at_340_pm : angle_at_time 3 40 = 130 := by
  sorry

end smaller_angle_at_340_pm_l608_608809


namespace range_of_h_l608_608223

noncomputable def h (t : ℝ) : ℝ := (t^2 + 0.5 * t) / (t^2 + 2)

theorem range_of_h : 
  Set.Icc (0.5 - 3 * Real.sqrt 2 / 16) (0.5 + 3 * Real.sqrt 2 / 16) = 
  {y | ∃ t : ℝ, y = h(t)} := 
sorry

end range_of_h_l608_608223


namespace proposition_p_is_false_proposition_q_is_false_l608_608385

def proposition_p_exists := ∃ x : ℝ, sqrt x = sqrt (2 * x + 1)

def proposition_q_forall := ∀ x : ℝ, (0 < x) → x^2 < x^3

theorem proposition_p_is_false : ¬ proposition_p_exists :=
by sorry

theorem proposition_q_is_false : ¬ proposition_q_forall :=
by sorry

end proposition_p_is_false_proposition_q_is_false_l608_608385


namespace ratio_of_greg_to_katie_gold_l608_608612

axiom GregGold : ℕ
axiom KatieGold : ℕ
axiom TotalGold : ℕ

axiom total_gold_condition : TotalGold = 100
axiom greg_gold_condition : GregGold = 20
axiom katie_gold_condition : KatieGold = TotalGold - GregGold

theorem ratio_of_greg_to_katie_gold : GregGold / KatieGold = 1 / 4 :=
by
  rw [total_gold_condition, greg_gold_condition, katie_gold_condition]
  sorry

end ratio_of_greg_to_katie_gold_l608_608612


namespace glass_pieces_colored_l608_608778

def glass_pieces := ℕ
def painted_color := glass_pieces → ℕ
def all_same_color (n: ℕ) (x y z : painted_color) : Prop :=
  ∃ (c₀ : ℕ), (∀ (i : ℕ), i < n → x i = c₀ ∧ y i = c₀ ∧ z i = c₀)

theorem glass_pieces_colored :
  ∀ (x y z : painted_color), (∑ i : ℕ in finset.range 1987, x i + y i + z i = 1987) →
  (∀ (i j : ℕ), i ≠ j → x i ≠ y i → y i ≠ z i → z i ≠ x i) →
  (∀ (i j : ℕ), i < j → (x i + y i + z i) % 3 = 1) →
  ( ∀ (n: ℕ), all_same_color 1987 x y z ) :=
by
  assume x y z,
  assume hsum hdiff hmod,
  sorry

end glass_pieces_colored_l608_608778


namespace find_xy_l608_608961

theorem find_xy (x y : ℝ) (h : x * (x + 2 * y) = x^2 + 10) : x * y = 5 :=
by
  sorry

end find_xy_l608_608961


namespace sum_of_squares_of_roots_eq_30_l608_608691

noncomputable def polynomial := (x : ℝ) → x^4 - 15 * x^2 + 56 = 0

theorem sum_of_squares_of_roots_eq_30
  (a b c d : ℝ)
  (h1 : polynomial a)
  (h2 : polynomial b)
  (h3 : polynomial c)
  (h4 : polynomial d) : 
  a^2 + b^2 + c^2 + d^2 = 30 :=
sorry

end sum_of_squares_of_roots_eq_30_l608_608691


namespace proof_problem_l608_608606

open Real

def polar_eq_C1 (rho theta : ℝ) := rho = 4 * cos theta
def param_eq_l (t x y : ℝ) := x = 1 - (2 * sqrt 5 / 5) * t ∧ y = 1 + (sqrt 5 / 5) * t
def param_eq_C2 (alpha x y : ℝ) := x = 2 * cos alpha ∧ y = sin alpha
def point_P_polar := (2 * sqrt 2, 2)
def midpoint_M (Q : ℝ × ℝ) := (1 + Q.1, 1 + (1 / 2) * Q.2)

def distance_to_line (M : ℝ × ℝ) (x y : ℝ) := abs (M.1 + 2 * M.2 - 3) / sqrt 5

theorem proof_problem :
  ∀ (rho theta t alpha : ℝ),
  polar_eq_C1 rho theta →
  (∀ (x y : ℝ), param_eq_l t x y → x + 2 * y - 3 = 0) →
  (point_P_polar = (2 * sqrt 2, 2)) →
  (∀ (Q : ℝ × ℝ), param_eq_C2 alpha Q.1 Q.2 → 
    distance_to_line (midpoint_M Q) = sqrt 10 / 5) :=
by
  intros rho theta t alpha h_rho_eq h_line_eq h_point_P_eq h_C2_eq
  sorry

end proof_problem_l608_608606


namespace find_divided_number_l608_608706

-- Declare the constants and assumptions
variables (d q r : ℕ)
variables (n : ℕ)
variables (h_d : d = 20)
variables (h_q : q = 6)
variables (h_r : r = 2)
variables (h_def : n = d * q + r)

-- State the theorem we want to prove
theorem find_divided_number : n = 122 :=
by
  sorry

end find_divided_number_l608_608706


namespace necessary_but_not_sufficient_l608_608949

theorem necessary_but_not_sufficient {a b c d : ℝ} (hcd : c > d) : 
  (a - c > b - d) → (a > b) ∧ ¬((a > b) → (a - c > b - d)) :=
by
  sorry

end necessary_but_not_sufficient_l608_608949


namespace slope_of_line_through_P_and_Q_l608_608914

-- Define points P and Q
def P : (ℝ × ℝ) := (1, 2)
def Q : (ℝ × ℝ) := (-3, 4)

-- The slope formula
def slope_formula (P Q : ℝ × ℝ) : ℝ :=
  (Q.2 - P.2) / (Q.1 - P.1)

-- Statement to prove
theorem slope_of_line_through_P_and_Q :
  slope_formula P Q = (-1 / 2) :=
sorry

end slope_of_line_through_P_and_Q_l608_608914


namespace ring_hanging_distance_l608_608133

theorem ring_hanging_distance :
  let thickness := 2
  let diam_top := 24
  let diam_bottom := 4
  let inside_diam_top := diam_top - 2 * thickness
  let inside_diam_bottom := diam_bottom - 2 * thickness
  let n := (inside_diam_top - inside_diam_bottom) / 2 + 1 -- Number of rings
  let sum := n * (inside_diam_top + inside_diam_bottom) / 2
  in sum = 110 := 
by
  -- Conditions and intermediate steps
  let thickness := 2
  let diam_top := 24
  let diam_bottom := 4
  let inside_diam_top := diam_top - 2 * thickness
  let inside_diam_bottom := diam_bottom - 2 * thickness
  let n := (inside_diam_top - inside_diam_bottom) / 2 + 1
  let sum := n * (inside_diam_top + inside_diam_bottom) / 2
  
  -- Conclude the result
  calc
    sum = 11 * 10 : by sorry
    ... = 110 : by sorry

end ring_hanging_distance_l608_608133


namespace inequality_solution_l608_608396

theorem inequality_solution (x : ℝ) : |x - 3| + |x - 5| ≥ 4 → x ≥ 6 ∨ x ≤ 2 :=
by
  sorry

end inequality_solution_l608_608396


namespace pentagon_diagonal_inequality_l608_608717

theorem pentagon_diagonal_inequality
  (a b c d e p q r s t : ℝ)
  (h1 : a + b > p)
  (h2 : b + c > q)
  (h3 : c + d > r)
  (h4 : d + e > s)
  (h5 : e + a > t) :
  let P := a + b + c + d + e,
      S := p + q + r + s + t in
  P < S ∧ S < 2 * P := by
sorry

end pentagon_diagonal_inequality_l608_608717


namespace distinct_values_count_l608_608535

theorem distinct_values_count :
  (∃ f : ℕ → ℚ, (∀ n ∈ (range 100).map (+1),
    f n = (n^2 - 2 : ℚ) / (n^2 - n + 2)) ∧ 
    (∀ m n ∈ (range 100).map (+1), f m = f n → m = n)) →
  ∃ distinct_values : ℕ, distinct_values = 98 := 
sorry

end distinct_values_count_l608_608535


namespace probability_of_2_1_l608_608054

noncomputable def probability_draw_2_1 (black_balls : ℕ) (white_balls : ℕ) (total_draws : ℕ) : ℚ :=
  let total_ways := (nat.choose (black_balls + white_balls) total_draws)
  let ways_2_black_1_white := (nat.choose black_balls 2) * (nat.choose white_balls 1)
  let ways_2_white_1_black := (nat.choose white_balls 2) * (nat.choose black_balls 1)
  let favorable_ways := ways_2_black_1_white + ways_2_white_1_black
  favorable_ways / total_ways

theorem probability_of_2_1 (black_balls : ℕ) (white_balls : ℕ) :
  black_balls = 10 → white_balls = 8 → probability_draw_2_1 10 8 3 = (80/102) := by
  intros
  sorry

end probability_of_2_1_l608_608054


namespace kylie_coins_l608_608347

open Nat

theorem kylie_coins :
  ∀ (coins_from_piggy_bank coins_from_brother coins_from_father coins_given_to_friend total_coins_left : ℕ),
  coins_from_piggy_bank = 15 →
  coins_from_brother = 13 →
  coins_from_father = 8 →
  coins_given_to_friend = 21 →
  total_coins_left = coins_from_piggy_bank + coins_from_brother + coins_from_father - coins_given_to_friend →
  total_coins_left = 15 :=
by
  intros
  sorry

end kylie_coins_l608_608347


namespace probability_of_target_destroyed_l608_608537

theorem probability_of_target_destroyed :
  let p1 := 0.9
  let p2 := 0.9
  let p3 := 0.8
  (p1 * p2 * p3) + (p1 * p2 * (1 - p3)) + (p1 * (1 - p2) * p3) + ((1 - p1) * p2 * p3) = 0.954 :=
by
  let p1 := 0.9
  let p2 := 0.9
  let p3 := 0.8
  sorry

end probability_of_target_destroyed_l608_608537


namespace divisible_subset_sum_l608_608693

theorem divisible_subset_sum (n : ℕ) (h1 : ∃ k : ℕ, n = 2^k) (S : Finset ℕ) (h2 : S.card = 2 * n - 1) :
  ∃ T : Finset ℕ, T ⊆ S ∧ T.card = n ∧ n ∣ T.sum :=
by
  sorry

end divisible_subset_sum_l608_608693


namespace cube_edge_length_l608_608402

theorem cube_edge_length (surface_area : ℝ) (h : surface_area = 96) : ∃ (edge_length : ℝ), edge_length = 4 := 
by 
  sorry

end cube_edge_length_l608_608402


namespace infinite_power_tower_l608_608403

noncomputable def x := (2 : ℝ)^(1/3)

theorem infinite_power_tower (h1: x^3 = 2) : x = (2 : ℝ)^(1/3) ∧ x^{x^{x^{.^{.^.}}}} = 2 := 
by
  sorry

end infinite_power_tower_l608_608403


namespace surface_integral_cylinder_l608_608457

theorem surface_integral_cylinder :
  let Σ := {p | ∃ z φ, p = (Real.cos φ, Real.sin φ, z) ∧ 0 ≤ z ∧ z ≤ 2 ∧ 0 ≤ φ ∧ φ ≤ 2 * Real.pi },
      integrand := λ (p : ℝ × ℝ × ℝ), p.1^2 + p.2^2,
      dσ := λ _ _, ((1 : ℝ) * (1 : ℝ) * (1 : ℝ)) -- dφ dz in cylindrical coordinates
  in ∫∫_Σ integrand dσ = 4 * Real.pi := by
  sorry

end surface_integral_cylinder_l608_608457


namespace saree_sale_price_400_l608_608761

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
price * (1 - discount / 100)

def final_sale_price (original_price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount original_price

theorem saree_sale_price_400 (original_price : ℝ)
  (discounts : List ℝ) (final_price : ℝ) :
  original_price = 400 →
  discounts = [15, 5, 10, 8, 12, 7] →
  final_price = final_sale_price original_price discounts →
  final_price = 218.88 :=
by
  intros _ _ _
  sorry

end saree_sale_price_400_l608_608761


namespace complex_expression_eq_l608_608845

open Real

theorem complex_expression_eq (p q : ℝ) (hpq : p ≠ q) :
  (sqrt ((p^4 + q^4)/(p^4 - p^2 * q^2) + (2 * q^2)/(p^2 - q^2)) * (p^3 - p * q^2) - 2 * q * sqrt p) /
  (sqrt (p / (p - q) - q / (p + q) - 2 * p * q / (p^2 - q^2)) * (p - q)) = 
  sqrt (p^2 - q^2) / sqrt p := 
sorry

end complex_expression_eq_l608_608845


namespace find_sum_of_squares_of_roots_l608_608689

theorem find_sum_of_squares_of_roots:
  ∀ (a b c d : ℝ), (a^2 * b^2 * c^2 * d^2 - 15 * a * b * c * d + 56 = 0) → 
  a^2 + b^2 + c^2 + d^2 = 30 := by
  intros a b c d h
  sorry

end find_sum_of_squares_of_roots_l608_608689


namespace right_triangle_value_of_a_l608_608981

-- Conditions
variables (a : ℤ)

-- Definition of the sides
def side1 := 2 * a
def side2 := 2 * a + 2
def side3 := 2 * a + 4

-- Pythagorean theorem condition
def pythagorean_theorem := side3 ^ 2 = side1 ^ 2 + side2 ^ 2

-- The value of a that makes the triangle a right triangle
theorem right_triangle_value_of_a (h : pythagorean_theorem a) : a = 3 :=
sorry

end right_triangle_value_of_a_l608_608981


namespace distinct_patterns_4x4_3_shaded_l608_608983

def num_distinct_patterns (n : ℕ) (shading : ℕ) : ℕ :=
  if n = 4 ∧ shading = 3 then 15
  else 0 -- Placeholder for other cases, not relevant for our problem

theorem distinct_patterns_4x4_3_shaded :
  num_distinct_patterns 4 3 = 15 :=
by {
  -- The proof would go here
  sorry
}

end distinct_patterns_4x4_3_shaded_l608_608983


namespace next_smallest_abundant_after_12_l608_608219

def is_proper_divisor (n d : Nat) : Prop := d < n ∧ n % d = 0

def sum_proper_divisors (n : Nat) : Nat :=
  (Finset.range n).filter (is_proper_divisor n).sum id

def is_abundant (n : Nat) : Prop :=
  sum_proper_divisors n > n

def next_abundant (n : Nat) : Nat :=
  Nat.find (λ m, m > n ∧ is_abundant m)

theorem next_smallest_abundant_after_12 : next_abundant 12 = 18 := by
  sorry

end next_smallest_abundant_after_12_l608_608219


namespace range_of_a_l608_608968

open Real

noncomputable def f (x : ℝ) := x - sqrt (x^2 + x)

noncomputable def g (x a : ℝ) := log x / log 27 - log x / log 9 + a * log x / log 3

theorem range_of_a (a : ℝ) : (∀ x1 ∈ Set.Ioi 1, ∃ x2 ∈ Set.Icc 3 9, f x1 > g x2 a) → a ≤ -1/12 :=
by
  intro h
  sorry

end range_of_a_l608_608968


namespace k_A_bounds_l608_608676

-- Define the set of 5x5 real matrices of rank 3
def M : set (matrix (fin 5) (fin 5) ℝ) := {A | matrix.rank A = 3}

-- Define the function k_A which counts the number of linearly independent non-empty subsets of the columns of A
noncomputable def k_A (A : matrix (fin 5) (fin 5) ℝ) : ℕ :=
  (finset.powerset.univ.filter (λ s, matrix.rank (A.minor id (coe_fn s.to_fun)).val = s.card)).card

-- Statement of the problem
theorem k_A_bounds {A : matrix (fin 5) (fin 5) ℝ} (hA : A ∈ M) : 7 ≤ k_A A ∧ k_A A ≤ 25 :=
by {
  sorry
}

end k_A_bounds_l608_608676


namespace correct_total_l608_608478

-- Define the conditions in Lean
variables (y : ℕ) -- y is a natural number (non-negative integer)

-- Define the values of the different coins in cents
def value_of_quarter := 25
def value_of_dollar := 100
def value_of_nickel := 5
def value_of_dime := 10

-- Define the errors in terms of y
def error_due_to_quarters := y * (value_of_dollar - value_of_quarter) -- 75y
def error_due_to_nickels := y * (value_of_dime - value_of_nickel) -- 5y

-- Net error calculation
def net_error := error_due_to_quarters - error_due_to_nickels -- 70y

-- Math proof problem statement
theorem correct_total (h : error_due_to_quarters = 75 * y ∧ error_due_to_nickels = 5 * y) :
  net_error = 70 * y :=
by sorry

end correct_total_l608_608478


namespace product_of_place_values_l608_608810

def numeral := 780.38
def place_value_tens := 8 * 10
def place_value_hundredths := 8 * 0.01
def product := place_value_tens * place_value_hundredths

theorem product_of_place_values : product = 6.4 := by
  -- The proof steps would go here
  sorry

end product_of_place_values_l608_608810


namespace tan_equiv_integer_l608_608218

theorem tan_equiv_integer (n : ℤ) (h : -180 < n ∧ n < 180) : 
  ∃ (k : ℤ), tan (n : ℝ) = tan (1230 : ℝ) := sorry

end tan_equiv_integer_l608_608218


namespace find_x_l608_608650

noncomputable def PQ (x : ℕ) : ℕ := 22 - 2 * x
noncomputable def PR (x : ℕ) : ℕ := 22 - 2 * x
noncomputable def QR (x : ℕ) : ℕ := 4 * x - 22
noncomputable def PS (x : ℕ) : ℕ := x
noncomputable def SR (x : ℕ) : ℕ := x

theorem find_x :
  (∀ x : ℕ, 
  22 = PQ x + QR x + PR x ∧ 
  22 = PR x + PS x + SR x ∧ 
  24 = PQ x + QR x + PS x + SR x) ->
  ∃ x : ℕ, x = 6 :=
by
  intro h
  use 6
  sorry

end find_x_l608_608650


namespace measure_angle_BCD_l608_608336

open Real

theorem measure_angle_BCD
  (ABC : Triangle)
  (H1 : ABC.AB = ABC.AC)
  (H2 : ∠ABC.A = 100)
  (I : Incenter ABC)
  (D : Point)
  (H3 : D ∈ Segment ABC.AB)
  (H4 : Segment D B = Segment I B) : 
  ∠ BCD = 30 := by
  sorry

end measure_angle_BCD_l608_608336


namespace probability_x_plus_y_less_than_4_l608_608101

open Set Real

/-- Define the square with vertices (0,0), (0,3), (3,3), and (3,0) -/
def square := set.Icc (0 : ℝ) 3 ×ˢ set.Icc (0 : ℝ) 3

/-- Probability that a randomly chosen point (x, y) in the square satisfies x + y < 4 -/
theorem probability_x_plus_y_less_than_4 : 
  let area_square := 9 in
  let area_excluded_triangle := 2 in
  let relevant_area := area_square - area_excluded_triangle in
  (relevant_area / area_square : ℝ) = 7 / 9 :=
by
{ sorry }

end probability_x_plus_y_less_than_4_l608_608101


namespace committee_formation_l608_608163

-- Define the parameters of the problem.
def departments : List String := ["mathematics", "statistics", "computer science"]
def professors_per_department : ℕ := 3
def committee_size : ℕ := 8
def male_count : ℕ := 4
def female_count : ℕ := 4
def professors_per_committee : ℕ := 2

-- Define the committee formation problem.
theorem committee_formation (h : ∀ d ∈ departments, ∃ n : ℕ, n = professors_per_department) : 
  ∃ (N : ℕ), N = 783 := by
  -- Conditions:
  have departments_condition : |departments| = 3 := by sorry
  have committee_size_condition : committee_size = 8 := by sorry
  have male_female_count_condition : male_count + female_count = committee_size := by sorry
  have committee_uniform_condition : ∀ d ∈ departments, n = professors_per_committee := by sorry

  -- The proof should show the computation leading to the final number 783.
  existsi 783
  sorry

end committee_formation_l608_608163


namespace a_41_eq_6585451_l608_608458

noncomputable def a : ℕ → ℕ
| 0     => 0 /- Not used practically since n >= 1 -/
| 1     => 1
| 2     => 1
| 3     => 2
| (n+4) => a n + a (n+2) + 1

theorem a_41_eq_6585451 : a 41 = 6585451 := by
  sorry

end a_41_eq_6585451_l608_608458


namespace candidates_appeared_equal_l608_608314

theorem candidates_appeared_equal 
  (A_candidates B_candidates : ℕ)
  (A_selected B_selected : ℕ)
  (h1 : 6 * A_candidates = A_selected * 100)
  (h2 : 7 * B_candidates = B_selected * 100)
  (h3 : B_selected = A_selected + 83)
  (h4 : A_candidates = B_candidates):
  A_candidates = 8300 :=
by
  sorry

end candidates_appeared_equal_l608_608314


namespace ratio_H_G_l608_608679

theorem ratio_H_G (G H : ℤ) (h : ∀ x : ℝ, x ≠ -3 → x ≠ 0 → x ≠ 5 → 
    (G / (x + 3) + H / (x * (x - 5)) = (x^2 - 3 * x + 8) / (x^3 + x^2 - 15 * x))) :
    H / G = 64 :=
sorry

end ratio_H_G_l608_608679


namespace set1_has_not_P_set2_has_P_property_P_a1_equals_1_and_average_when_n_5_geometric_sequence_l608_608245

-- Definition of property P
def property_P (A : Set ℕ) : Prop :=
  ∀ i j ∈ A, (i * j ∈ A) ∨ ((j / i : ℚ) ∈ A)

-- Problem (I) part 1: Prove that {1, 3, 4} does not have property P
theorem set1_has_not_P : ¬property_P { 1, 3, 4 } :=
sorry

-- Problem (I) part 2: Prove that {1, 2, 3, 6} has property P
theorem set2_has_P : property_P { 1, 2, 3, 6 } :=
sorry

-- Problem (II): Prove that a_1 = 1 and the given average condition
theorem property_P_a1_equals_1_and_average (A : Set ℕ) (h : property_P A) 
  (h_ordered : ∃ (a : ℕ → ℕ) (n : ℕ), (1 ≤ a 1) ∧ (∀ i, i < n → a i < a (i + 1)) ∧ A = {a i | i < n }) :
  a 1 = 1 ∧ (∑ i in A, i) / (∑ i in A, (1 : ℚ) / i) = A.max' :=
sorry

-- Problem (III): Prove that when n = 5, the sequence {a1, a2, a3, a4, a5} is a geometric sequence
theorem when_n_5_geometric_sequence {a : ℕ → ℕ} (h : property_P {a 1, a 2, a 3, a 4, a 5})
  (h_ordered : 1 ≤ a 1 ∧ a 1 < a 2 ∧ a 2 < a 3 ∧ a 3 < a 4 ∧ a 4 < a 5) :
  ∃ r : ℕ, (a 2 = a 1 * r) ∧ (a 3 = a 1 * r^2) ∧ (a 4 = a 1 * r^3) ∧ (a a 5 = a 1 * r^4) :=
sorry

end set1_has_not_P_set2_has_P_property_P_a1_equals_1_and_average_when_n_5_geometric_sequence_l608_608245


namespace gears_can_rotate_l608_608318

theorem gears_can_rotate (n : ℕ) : (∃ f : ℕ → Prop, f 0 ∧ (∀ k, f (k+1) ↔ ¬f k) ∧ f n = f 0) ↔ (n % 2 = 0) :=
by
  sorry

end gears_can_rotate_l608_608318


namespace problem_condition_l608_608033

theorem problem_condition (m : ℝ) :
  (∀ x : ℝ, x ≤ -1 → (m^2 - m) * 4^x - 2^x < 0) → -1 < m ∧ m < 2 :=
sorry

end problem_condition_l608_608033


namespace binomial_expansion_coefficient_x2_l608_608330

noncomputable def binomialCoeff : ℕ :=
  -- This represents the binomial coefficient (5 choose 3)
  Nat.choose 5 3

theorem binomial_expansion_coefficient_x2 :
  -- Given the binomial expansion of (x - 2 / sqrt(x))^5,
  let coeff := (-2)^(5-3) * binomialCoeff in
  -- Prove that the coefficient of x^2 term is 40
  coeff = 40 :=
by
  -- Skip the proof
  sorry

end binomial_expansion_coefficient_x2_l608_608330


namespace batsman_average_after_17_l608_608471

variable (x : ℝ)
variable (total_runs_16 : ℝ := 16 * x)
variable (runs_17 : ℝ := 90)
variable (new_total_runs : ℝ := total_runs_16 + runs_17)
variable (new_average : ℝ := new_total_runs / 17)

theorem batsman_average_after_17 :
  (total_runs_16 + runs_17 = 17 * (x + 3)) → new_average = x + 3 → new_average = 42 :=
by
  intros h1 h2
  sorry

end batsman_average_after_17_l608_608471


namespace geometric_sequence_first_term_l608_608265

theorem geometric_sequence_first_term (b : ℕ → ℝ) 
  (h1 : ∀ n, b (n + 1) = 2 * b n)
  (h2 : ∀ n, real.logb 2 (b n) * real.logb 2 (b (n + 1)) = n^2 + 3 * n) :
  b 1 = 4 :=
sorry

end geometric_sequence_first_term_l608_608265


namespace cheenu_speed_difference_l608_608174

theorem cheenu_speed_difference :
  let cycling_time := 120 -- minutes
  let cycling_distance := 24 -- miles
  let jogging_time := 180 -- minutes
  let jogging_distance := 18 -- miles
  let cycling_speed := cycling_time / cycling_distance -- minutes per mile
  let jogging_speed := jogging_time / jogging_distance -- minutes per mile
  let speed_difference := jogging_speed - cycling_speed -- minutes per mile
  speed_difference = 5 := by sorry

end cheenu_speed_difference_l608_608174


namespace Gottfried_wins_game_l608_608753

noncomputable def Gottfried_strategy_wins (n : ℕ) : Prop :=
  let initial_number := 10 ^ n in
  let game_possible_moves (x : ℕ) : List (ℕ × ℕ) :=
    [ (a, b) | a b n, b n]

  let next_numbers (board : List ℕ) : List (List ℕ) :=
    board.foldr (λ x acc, 
      acc ++ (game_possible_moves x |>.map (λ (a, b), (a, b))))

  let game_over (board : List ℕ) : Bool := 
    board.empty ∨ l.refs.length = 1 ∧ board.head = x

  let initial_board : List ℕ := [initial_number] in

  exists (α : Type) -- omit non-relevant parts of Lean code required.

  ∀ α Gottfried_winning_strategy, Isaac_losing_strategy,
    game_possible_moves, next_numbers,
    sub_games : ∀ board, Subgame α' → Prop.

  sorry

theorem Gottfried_wins_game : Gottfried_strategy_wins 2019 :=
  sorry

end Gottfried_wins_game_l608_608753


namespace circle_arcs_cover_720_l608_608077

theorem circle_arcs_cover_720 {α : Type} (S : set (set α)) [fintype S]
  (h1 : ∀ s ∈ S, measure_theory.measure_space.measure_univ α < 360)
  (h2 : ∀ x ∈ univ, ∃ s ∈ S, x ∈ s)
  : ∃ T ⊆ S, (⋃ t ∈ T, t) = univ ∧ (measure_theory.measure ᵐ⋃ t ∈ T, t) ≤ 720 :=
sorry

end circle_arcs_cover_720_l608_608077


namespace probability_product_multiple_of_12_l608_608627

def S : Finset ℕ := {2, 3, 4, 5, 6, 9}

def is_multiple_of_12 (n : ℕ) : Prop := ∃ k, n = 12 * k

theorem probability_product_multiple_of_12 :
  (∃ s ∈ S.powersetLen 3, is_multiple_of_12 (s.prod id)) →
  (6 / 20 = 3 / 10) :=
by
  sorry

end probability_product_multiple_of_12_l608_608627


namespace min_seats_occupied_l608_608320

theorem min_seats_occupied (n : ℕ) (h : n = 150) : ∃ k : ℕ, k = 37 ∧ ∀ m : ℕ, m > k → ∃ i : ℕ, i < k ∧ m - k ≥ 2 := sorry

end min_seats_occupied_l608_608320


namespace volume_of_pyramid_BCGH_l608_608356
noncomputable theory

open Real

-- Define the vertices of the cube
def B : (ℝ × ℝ × ℝ) := (2, 0, 0)
def C : (ℝ × ℝ × ℝ) := (2, 2, 0)
def G : (ℝ × ℝ × ℝ) := (2, 2, 2)
def H : (ℝ × ℝ × ℝ) := (0, 2, 2)

-- The main theorem to prove
theorem volume_of_pyramid_BCGH : 
  let base_area := sqrt 3,
      height := 2,
      volume := (1 / 3) * base_area * height in
  volume = (2 * sqrt 3) / 3 :=
by
  sorry

end volume_of_pyramid_BCGH_l608_608356


namespace tan_sum_equals_120_l608_608616

theorem tan_sum_equals_120 (x y : Real) (h1 : Real.tan x + Real.tan y = 30) (h2 : Real.cot x + Real.cot y = 40) : Real.tan (x + y) = 120 :=
sorry

end tan_sum_equals_120_l608_608616


namespace incorrect_inference_l608_608386

open_locale classical

variables {a b : Type*}
variables [line a] [line b]
variables (α : set (Type*)) [plane α]
variables (A B : Type*)

-- Conditions
axiom perp_a_alpha : a ⟂ α
axiom perp_b_alpha : b ⟂ α
axiom A_on_a_alpha : A ∈ a ∩ α
axiom B_on_b_alpha : B ∈ b ∩ α
axiom AB_on_alpha : A ∈ α ∧ B ∈ α
axiom perp_a_AB : a ⟂ (line_through_points A B)
axiom perp_b_AB : b ⟂ (line_through_points A B)

-- The proof statement
theorem incorrect_inference : ¬ (a ∥ b) :=
by sorry

end incorrect_inference_l608_608386


namespace car_travel_distance_l608_608284

noncomputable def car_distance_in_30_minutes : ℝ := 
  let train_speed : ℝ := 96
  let car_speed : ℝ := (5 / 8) * train_speed
  let travel_time : ℝ := 0.5  -- 30 minutes is 0.5 hours
  car_speed * travel_time

theorem car_travel_distance : car_distance_in_30_minutes = 30 := by
  sorry

end car_travel_distance_l608_608284


namespace total_amount_paid_l608_608613

theorem total_amount_paid (quantity_grapes quantity_mangoes : ℕ) (rate_grapes rate_mangoes : ℕ) :
  quantity_grapes = 9 → rate_grapes = 70 → quantity_mangoes = 9 → rate_mangoes = 55 →
  (quantity_grapes * rate_grapes + quantity_mangoes * rate_mangoes) = 1125 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  norm_num
  exact sorry

end total_amount_paid_l608_608613


namespace train_length_l608_608139

-- condition: The train can cross an electric pole in 50 sec.
def time_to_cross_pole : ℝ := 50

-- condition: The speed of the train is 180 km/h.
def speed_kmh : ℝ := 180

-- convert speed from km/h to m/s
def speed_ms : ℝ := speed_kmh * (1000 / 3600)

-- The distance should be speed * time
def length_of_train (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem train_length :
  length_of_train speed_ms time_to_cross_pole = 2500 :=
sorry

end train_length_l608_608139


namespace digit_80th_in_sequence_of_60_to_1_l608_608292

noncomputable def digit_seq : List ℕ := (List.range' 1 60).reverse.join.toList

theorem digit_80th_in_sequence_of_60_to_1 : (digit_seq.get 79).get 0 = 1 := 
by
  sorry

end digit_80th_in_sequence_of_60_to_1_l608_608292


namespace num_quadratic_polynomials_l608_608680

theorem num_quadratic_polynomials (P Q R : ℝ[X]) (hP : P = (X - 1) * (X - 4) * (X - 5))
  (hR_deg : R.degree = 3) :
  ∃! (Q : ℝ[X]), ∃ (R : ℝ[X]), (P.eval_expr (Q) = P * R) ∧ Q.degree = 2 :=
begin
  sorry,
end

end num_quadratic_polynomials_l608_608680


namespace find_roots_combination_l608_608947

theorem find_roots_combination 
  (α β : ℝ)
  (hα : α^2 - 3 * α + 1 = 0)
  (hβ : β^2 - 3 * β + 1 = 0) :
  7 * α^3 + 10 * β^4 = 697 := by
  sorry

end find_roots_combination_l608_608947


namespace time_to_cross_pole_correct_l608_608499

noncomputable def speed_kmph : ℝ := 160 -- Speed of the train in kmph
noncomputable def length_meters : ℝ := 800.064 -- Length of the train in meters

noncomputable def conversion_factor : ℝ := 1000 / 3600 -- Conversion factor from kmph to m/s
noncomputable def speed_mps : ℝ := speed_kmph * conversion_factor -- Speed of the train in m/s

noncomputable def time_to_cross_pole : ℝ := length_meters / speed_mps -- Time to cross the pole

theorem time_to_cross_pole_correct :
  time_to_cross_pole = 800.064 / (160 * (1000 / 3600)) :=
sorry

end time_to_cross_pole_correct_l608_608499


namespace seats_usually_taken_l608_608775

theorem seats_usually_taken:
  let tables := 15 in
  let seats_per_table := 10 in
  let total_seats := tables * seats_per_table in
  let unseated_fraction := 1 / 10 in
  let unseated_seats := total_seats * unseated_fraction in
  let seats_taken := total_seats - unseated_seats in
  seats_taken = 135 :=
by
  sorry

end seats_usually_taken_l608_608775


namespace exists_increasing_seq_divisible_sum_of_squares_l608_608387

variable (a1 : ℕ) (ha1 : a1 > 1)

theorem exists_increasing_seq_divisible_sum_of_squares :
  ∃ (a : ℕ → ℕ), (a 0 = a1) ∧ (∀ n, a n < a (n + 1)) ∧ 
  (∀ k, 0 < k → (∑ i in Finset.range k, (a i)^2) % (∑ i in Finset.range k, a i) = 0) :=
sorry

end exists_increasing_seq_divisible_sum_of_squares_l608_608387


namespace line_FG_perpendicular_BE_l608_608340

-- Define the variables and conditions
variables {A B C D E F G : Type} [AffineSpace ℝ A]
variables [AddCommGroup A] [Module ℝ A]

-- Parallelogram conditions and points
variables (parallelogram_ABCDEF : is_parallelogram A B C D)
          (point_E : Type) (hCE_CB : CE = CB)
          (mid_F : midpoint (C + D)) (mid_G : midpoint (A + E))

-- Define what we need to prove
theorem line_FG_perpendicular_BE : 
  ∀ (parallelogram_ABCDEF : is_parallelogram A B C D)
  (point_E : A) 
  (hCE_CB : CE = CB) 
  (mid_F : F = midpoint (C, D)) 
  (mid_G : G = midpoint (A, E)),
  is_perpendicular (line_through F G) (line_through B E) :=
begin
  sorry
end

end line_FG_perpendicular_BE_l608_608340


namespace sum_of_possible_values_of_x_l608_608198

noncomputable def mean (a b c d e f g : ℝ) : ℝ := (a + b + c + d + e + f + g) / 7
noncomputable def median (a b c d e f x : ℝ) : ℝ :=
  if x ≤ 3 then 3 else if x < 5 then x else 5
noncomputable def mode (a b c d e f : ℝ) : ℝ := 3

theorem sum_of_possible_values_of_x :
  let x1 := 17
  let x2 := 53 / 13
  (x1 + x2) = 17 + 53 / 13 := by
  sorry

end sum_of_possible_values_of_x_l608_608198


namespace not_washed_shirts_l608_608844

-- Definitions based on given conditions
def short_sleeve_shirts : ℕ := 9
def long_sleeve_shirts : ℕ := 21
def washed_shirts : ℕ := 29

-- Theorem to prove the number of shirts not washed
theorem not_washed_shirts : (short_sleeve_shirts + long_sleeve_shirts) - washed_shirts = 1 := by
  sorry

end not_washed_shirts_l608_608844


namespace distance_between_skew_lines_l608_608238

theorem distance_between_skew_lines (a : ℝ) : 
  let A₁ := (a, 0, a),
      B := (0, a, 0),
      D₁ := (a, a, a)
  in dist (λ t : ℝ, (0, 0, t*a)) (λ t : ℝ, (a - a * t, a * t, a * t)) = a * real.sqrt 2 / 2 :=
by
  sorry

end distance_between_skew_lines_l608_608238


namespace marbles_with_at_least_one_blue_l608_608666

theorem marbles_with_at_least_one_blue :
  (Nat.choose 10 4) - (Nat.choose 8 4) = 140 :=
by
  sorry

end marbles_with_at_least_one_blue_l608_608666


namespace always_possible_to_create_pairs_l608_608640

def number_of_pairs (a b : Nat) : Nat := Nat.min a b

theorem always_possible_to_create_pairs :
  ∀ (total_mittens blue_mittens green_mittens red_mittens right_mittens left_mittens : Nat),
  total_mittens = 30 →
  blue_mittens = 10 →
  green_mittens = 10 →
  red_mittens = 10 →
  right_mittens = 15 →
  left_mittens = 15 →
  (∃ (pairs : Nat), pairs >= 5).
Proof :=
by
  intros total_mittens blue_mittens green_mittens red_mittens right_mittens left_mittens
  intros h1 h2 h3 h4 h5 h6
  sorry

end always_possible_to_create_pairs_l608_608640


namespace required_height_for_roller_coaster_l608_608145

-- Definitions based on conditions from the problem
def initial_height : ℕ := 48
def natural_growth_rate_per_month : ℚ := 1 / 3
def upside_down_growth_rate_per_hour : ℚ := 1 / 12
def hours_per_month_hanging_upside_down : ℕ := 2
def months_in_a_year : ℕ := 12

-- Calculations needed for the proof
def annual_natural_growth := natural_growth_rate_per_month * months_in_a_year
def annual_upside_down_growth := (upside_down_growth_rate_per_hour * hours_per_month_hanging_upside_down) * months_in_a_year
def total_annual_growth := annual_natural_growth + annual_upside_down_growth
def height_next_year := initial_height + total_annual_growth

-- Statement of the required height for the roller coaster
theorem required_height_for_roller_coaster : height_next_year = 54 :=
by
  sorry

end required_height_for_roller_coaster_l608_608145


namespace unique_four_letter_list_with_same_product_l608_608539

-- Define the alphabet's values.
def alphabet_value (c : Char) : Nat :=
  let val := c.to_nat - 'A'.to_nat + 1
  if val ≥ 1 ∧ val ≤ 26 then val else 0

-- Define the product of a list of characters.
def product_of_list : List Char → Nat
| []      => 1
| (c::cs) => alphabet_value c * product_of_list cs

-- Define the given list of characters.
def PQRS : List Char := ['P', 'Q', 'R', 'S']
def LQSX : List Char := ['L', 'Q', 'S', 'X']

-- The proof statement
theorem unique_four_letter_list_with_same_product :
  product_of_list PQRS = product_of_list LQSX :=
by
  -- Placeholder for the actual proof.
  sorry

end unique_four_letter_list_with_same_product_l608_608539


namespace sum_abcd_l608_608352

variable (a b c d : ℝ)

theorem sum_abcd :
  (∃ y : ℝ, 2 * a + 3 = y ∧ 2 * b + 4 = y ∧ 2 * c + 5 = y ∧ 2 * d + 6 = y ∧ a + b + c + d + 10 = y) →
  a + b + c + d = -11 :=
by
  sorry

end sum_abcd_l608_608352


namespace number_of_female_democrats_l608_608785

-- Definitions and conditions
variables (F M D_F D_M D_T : ℕ)
axiom participant_total : F + M = 780
axiom female_democrats : D_F = 1 / 2 * F
axiom male_democrats : D_M = 1 / 4 * M
axiom total_democrats : D_T = 1 / 3 * (F + M)

-- Target statement to be proven
theorem number_of_female_democrats : D_T = 260 → D_F = 130 :=
by
  intro h
  sorry

end number_of_female_democrats_l608_608785


namespace circumcenter_on_AK_l608_608829

variable {α β γ : Real}
variable (A B C L H K O : Type)
variable [Triangle ABC] (circumcenter : Triangle ABC → Point O)
variable [AngleBisector A B C L]

theorem circumcenter_on_AK
  (h₁ : AL_is_angle_bisector ABC L)
  (h₂ : Height_from_B_on_AL B A L H)
  (h₃ : K_on_circumcircle_ABL B A L K)
  : Lies_on_line (circumcenter ABC) A K :=
sorry

end circumcenter_on_AK_l608_608829


namespace calls_on_friday_l608_608663

noncomputable def total_calls_monday := 35
noncomputable def total_calls_tuesday := 46
noncomputable def total_calls_wednesday := 27
noncomputable def total_calls_thursday := 61
noncomputable def average_calls_per_day := 40
noncomputable def number_of_days := 5
noncomputable def total_calls_week := average_calls_per_day * number_of_days

theorem calls_on_friday : 
  total_calls_week - (total_calls_monday + total_calls_tuesday + total_calls_wednesday + total_calls_thursday) = 31 :=
by
  sorry

end calls_on_friday_l608_608663


namespace polar_curve_is_circle_l608_608570

theorem polar_curve_is_circle (ρ θ : ℝ) (h : ρ = 2 * Real.cos θ - 4 * Real.sin θ) :
  ∃ (cx cy r : ℝ), (cx, cy, r) = (1, -2, √5) ∧ (∀ (x y : ℝ), x^2 + y^2 - 2*x + 4*y = 0 ↔ (x - cx)^2 + (y - cy)^2 = r^2) :=
by
  sorry

end polar_curve_is_circle_l608_608570


namespace P_on_line_l608_608278

def a := ℕ → ℝ
def b := ℕ → ℝ

def a_seq (a b : ℕ → ℝ) : ℕ → ℝ
| 0     := -1
| (n+1) := a n / (1 - 4 * (b n)^2)

def b_seq (a b : ℕ → ℝ) : ℕ → ℝ
| 0     := 1
| (n+1) := (a_seq a b (n+1)) * (b n)

noncomputable def P (n : ℕ) : ℝ × ℝ := (a_seq a b n, b_seq a b n)

theorem P_on_line (n : ℕ) : (a_seq a b n) + 2 * (b_seq a b n) - 1 = 0 := sorry

end P_on_line_l608_608278


namespace a_1993_eq_1997_l608_608930

theorem a_1993_eq_1997
  (a : ℕ → ℕ) -- Define the sequence as a function from naturals to naturals
  (a1 : a 1 = 5)
  (nat_nums : ∀ n, a n ∈ ℕ) -- All a_i are natural numbers
  (strict_inc : ∀ n m, n < m → a n < a m) -- A strictly increasing sequence
  (seq_prop : ∀ i, a (i + a i) = 2 * a i) -- Given property of the sequence
  : a 1993 = 1997 := sorry

end a_1993_eq_1997_l608_608930


namespace probability_of_x_plus_y_lt_4_l608_608096

open Classical -- To handle probability and random selection
open Set -- For geometric notions
open Filter -- For measure and integration

noncomputable def probability_condition (x y : ℝ) (h : 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) : Prop := x + y < 4

theorem probability_of_x_plus_y_lt_4 :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) →
                pr (λ (xy : ℝ × ℝ), probability_condition xy.1 xy.2 (and.intro (and.left xy.2) (and.right (and.left xy.2)))) = 7/9 :=
by sorry

end probability_of_x_plus_y_lt_4_l608_608096


namespace product_of_lcm_and_gcf_l608_608027

theorem product_of_lcm_and_gcf (n m : ℕ) (h₁ : n = 36) (h₂ : m = 48) :
  let gcf := Nat.gcd n m,
      lcm := Nat.lcm n m
  in gcf * lcm = 1728 := by
  sorry

end product_of_lcm_and_gcf_l608_608027


namespace brightness_ratio_sun_sirius_l608_608324

theorem brightness_ratio_sun_sirius : 
  ∀ (E1 E2 : ℝ) (m1 m2 : ℝ), 
  m1 = -26.7 → m2 = -1.45 → 
  m2 - m1 = (5 / 2) * log10 (E1 / E2) → 
  E1 / E2 = 10 ^ 10.1 :=
by
  intros E1 E2 m1 m2 h1 h2 h3
  rw [h1, h2] at h3
  sorry

end brightness_ratio_sun_sirius_l608_608324


namespace original_length_of_wood_l608_608853

theorem original_length_of_wood (s cl ol : ℝ) (h1 : s = 2.3) (h2 : cl = 6.6) (h3 : ol = cl + s) : 
  ol = 8.9 := 
by 
  sorry

end original_length_of_wood_l608_608853


namespace interchange_digits_product_l608_608694

-- Definition of the proof problem
theorem interchange_digits_product (n a b k : ℤ) (h1 : n = 10 * a + b) (h2 : n = (k + 1) * (a + b)) :
  ∃ x : ℤ, (10 * b + a) = x * (a + b) ∧ x = 10 - k :=
by
  existsi (10 - k)
  sorry

end interchange_digits_product_l608_608694


namespace number_of_zeros_of_f_l608_608416

noncomputable def f : ℝ → ℝ
| x => if x >= 0 then x^3 - 3*x + 1 else x^2 - 2*x - 4

theorem number_of_zeros_of_f : ∃ z, z = 3 := by
  sorry

end number_of_zeros_of_f_l608_608416


namespace max_sides_convex_polygon_divided_into_right_triangles_l608_608025

theorem max_sides_convex_polygon_divided_into_right_triangles (n : ℕ) :
  (∀ i ∈ {1, 2, ..., n}, angles[i] < 180 ∧ angles[i] % 30 = 0) →
  (∑ i in {1, 2, ..., n}, angles[i] = (n-2) * 180) →
  (∀ i ∈ {1, 2, ..., n}, angles[i] ≤ 150) →
  n ≤ 12 :=
by sorry

end max_sides_convex_polygon_divided_into_right_triangles_l608_608025


namespace part1_l608_608846

theorem part1 (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 2 / (c + a)) : 2 * b^2 = a^2 + c^2 :=
sorry

end part1_l608_608846


namespace parabola_directrix_eq_l608_608907

theorem parabola_directrix_eq (a b : ℝ) (h_eq : a = 4) (h_trans : b = 8) :
  let directrix := -(1 / (4 * a)) + b in
  directrix = 127 / 16 :=
by
  sorry

end parabola_directrix_eq_l608_608907


namespace distinct_integers_sum_l608_608291

theorem distinct_integers_sum (m n p q : ℕ) (h1 : m ≠ n) (h2 : m ≠ p) (h3 : m ≠ q) (h4 : n ≠ p)
  (h5 : n ≠ q) (h6 : p ≠ q) (h71 : m > 0) (h72 : n > 0) (h73 : p > 0) (h74 : q > 0)
  (h_eq : (7 - m) * (7 - n) * (7 - p) * (7 - q) = 4) : m + n + p + q = 28 := by
  sorry

end distinct_integers_sum_l608_608291


namespace max_distance_ellipse_line_l608_608909

theorem max_distance_ellipse_line : 
  let ellipse (x y : ℝ) := (x^2) / 16 + (y^2) / 12 = 1
  let line (x y : ℝ) := x - 2 * y - 12 = 0
  ∃ x y : ℝ, ellipse x y →
    ∃ θ : ℝ, x = 4 * Real.cos θ ∧ y = 2 * Real.sqrt 3 * Real.sin θ →
    ∀ θ : ℝ, let d := (12 + 8 * Real.sin (θ - Real.pi / 6)) / Real.sqrt 5 in d ≤ 4 * Real.sqrt 5 :=
  4 * Real.sqrt 5 :=
sorry

end max_distance_ellipse_line_l608_608909


namespace columbus_discovered_america_in_1492_l608_608390

theorem columbus_discovered_america_in_1492 :
  ∃ (x y z : ℕ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x ≠ 1 ∧ y ≠ 1 ∧ z ≠ 1 ∧
  1 + x + y + z = 16 ∧ y + 1 = 5 * z ∧
  1000 + 100 * x + 10 * y + z = 1492 :=
by
  sorry

end columbus_discovered_america_in_1492_l608_608390


namespace number_of_males_who_listen_l608_608001

theorem number_of_males_who_listen (females_listen : ℕ) (males_dont_listen : ℕ) (total_listen : ℕ) (total_dont_listen : ℕ) (total_females : ℕ) :
  females_listen = 72 →
  males_dont_listen = 88 →
  total_listen = 160 →
  total_dont_listen = 180 →
  (total_females = total_listen + total_dont_listen - (females_listen + males_dont_listen)) →
  (total_females + males_dont_listen + 92 = total_listen + total_dont_listen) →
  total_listen + total_dont_listen = females_listen + males_dont_listen + (total_females - females_listen) + 92 :=
sorry

end number_of_males_who_listen_l608_608001


namespace general_formula_for_S_sum_S_equals_10_C_n_plus_2_l608_608942

variable (n : ℕ) (S : ℕ → ℕ)
variable (M : set ℕ) (A : ℕ → set (set ℕ))

-- Conditions
def M (n : ℕ) : set ℕ := {i | 1 ≤ i ∧ i ≤ n}
def A (n : ℕ) : ℕ → set (set ℕ) := λ i, {s : set ℕ | s ⊆ M n ∧ s.card = 4}

-- Definition of Sₙ
def S (n : ℕ) : ℕ := ∑ s in (A n i for i in finset.range (nat.choose n 4)), s.sum

-- General formula for Sₙ
theorem general_formula_for_S (n : ℕ) (hn : n ≥ 4) : 
  S n = (nat.choose (n - 1) 3) *  (n * (n + 1) / 2) := sorry

-- Summation formula for S₄ + S₅ + ... + Sₙ
theorem sum_S_equals_10_C_n_plus_2 (n : ℕ) (hn : n ≥ 4) : 
  (∑ i in finset.range (n - 3),(S (i+4))) = 10 * nat.choose (n + 2) 6 := sorry

end general_formula_for_S_sum_S_equals_10_C_n_plus_2_l608_608942


namespace modulus_of_z_l608_608934

-- Given condition
def z : ℂ := (Complex.i + 1) / Complex.i

-- Goal: Prove that the modulus of z is sqrt(2)
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 :=
by
  sorry

end modulus_of_z_l608_608934


namespace cindy_smallest_coins_l608_608176

theorem cindy_smallest_coins : ∃ n : ℕ, 
  (∀ Y : ℕ, Y > 1 ∧ Y < n → n % Y = 0 → ∃ k : ℕ, k * Y = n) ∧ 
  (15 = (finset.filter (λ m, m > 1 ∧ m < n) (finset.Ico 1 (n+1))).card) ∧
  n = 65536 :=
begin
  sorry
end

end cindy_smallest_coins_l608_608176


namespace arctan_tan_proof_l608_608183

theorem arctan_tan_proof :
  let tan_75 := 1 / (tan 15)
  let tan_15 := 1 / real.sqrt 3
  arctan (tan_75 - 3 * tan_15) = 0 :=
by
  let tan_75 := 1 / (tan 15)
  let tan_15 := 1 / real.sqrt 3
  have key_step : tan 75 = tan_75 := sorry
  have simplification : tan_75 - 3 * tan_15 = 0 := sorry
  show arctan (tan_75 - 3 * tan_15) = 0, from
   by rw [simplification, arctan_zero]

end arctan_tan_proof_l608_608183


namespace percent_increase_in_perimeter_l608_608151

theorem percent_increase_in_perimeter :
  ∀ (a : ℝ), a = 2 → 
  let b := a * 1.5 in
  let c := b * 1.5 in
  let d := c * 1.5 in
  100 * (d / a - 1) = 237.5 :=
by
  sorry

end percent_increase_in_perimeter_l608_608151


namespace probability_x_plus_y_less_than_4_l608_608104

open Set Real

/-- Define the square with vertices (0,0), (0,3), (3,3), and (3,0) -/
def square := set.Icc (0 : ℝ) 3 ×ˢ set.Icc (0 : ℝ) 3

/-- Probability that a randomly chosen point (x, y) in the square satisfies x + y < 4 -/
theorem probability_x_plus_y_less_than_4 : 
  let area_square := 9 in
  let area_excluded_triangle := 2 in
  let relevant_area := area_square - area_excluded_triangle in
  (relevant_area / area_square : ℝ) = 7 / 9 :=
by
{ sorry }

end probability_x_plus_y_less_than_4_l608_608104


namespace sum_binom_real_part_l608_608884

theorem sum_binom_real_part :
  (1 / 2^2022) * (Finset.sum (Finset.range 1011) (λ n, (-4)^n * Nat.choose 2022 (2*n))) = 
  - (5^1011 / 2^2023) := 
sorry

end sum_binom_real_part_l608_608884


namespace exists_member_attended_at_least_8_game_nights_l608_608851

-- Define the problem conditions as Lean definitions
def members := 50
def meets_exactly_once (member1 member2 : ℕ) : Prop := ∃ night : ℕ, true
def not_all_members_meet (night : ℕ) : Prop := night < members

-- State the theorem in Lean 4
theorem exists_member_attended_at_least_8_game_nights :
  ∃ member : ℕ, ∃ participated_nights : ℕ, 8 ≤ participated_nights :=
by
  -- assume the problem conditions
  intros members meets_exactly_once not_all_members_meet

  -- this is to skip the proof steps
  sorry

end exists_member_attended_at_least_8_game_nights_l608_608851


namespace grid_3x3_unique_72_l608_608470

theorem grid_3x3_unique_72 :
  ∃ (f : Fin 3 → Fin 3 → ℕ), 
    (∀ (i j : Fin 3), 1 ≤ f i j ∧ f i j ≤ 9) ∧
    (∀ (i j k : Fin 3), j < k → f i j < f i k) ∧
    (∀ (i j k : Fin 3), i < k → f i j < f k j) ∧
    f 0 0 = 1 ∧ f 1 1 = 5 ∧ f 2 2 = 8 ∧
    (∃! (g : Fin 3 → Fin 3 → ℕ), 
      (∀ (i j : Fin 3), 1 ≤ g i j ∧ g i j ≤ 9) ∧
      (∀ (i j k : Fin 3), j < k → g i j < g i k) ∧
      (∀ (i j k : Fin 3), i < k → g i j < g k j) ∧
      g 0 0 = 1 ∧ g 1 1 = 5 ∧ g 2 2 = 8) :=
sorry

end grid_3x3_unique_72_l608_608470


namespace negation_proposition_l608_608752

variable (f : ℕ → ℕ)

theorem negation_proposition :
  (¬ (∀ n : ℕ, (f n ∈ ℕ) ∧ (f n > n))) ↔ 
  (∃ n0 : ℕ, (f n0 ∉ ℕ) ∨ (f n0 ≤ n0)) :=
by
  sorry

end negation_proposition_l608_608752


namespace ratio_unit_price_l608_608167

theorem ratio_unit_price (v p : ℝ) (hv : v ≠ 0) (hp : p ≠ 0) :
  let price_X := 0.85 * p,
      volume_X := 1.3 * v,
      unit_price_X := price_X / volume_X,
      unit_price_Y := p / v
  in (unit_price_X / unit_price_Y) = (17 / 26) :=
by {
  let price_X := 0.85 * p,
  let volume_X := 1.3 * v,
  let unit_price_X := price_X / volume_X,
  let unit_price_Y := p / v,
  have h1 : price_X / volume_X = (0.85 * p) / (1.3 * v) := rfl,
  have h2 : (0.85 * p) / (1.3 * v) = (85 * p) / (130 * v) := by norm_num,
  have h3 : (85 * p) / (130 * v) = (17 * p) / (26 * v) := by norm_num,
  have h4 : (unit_price_X / unit_price_Y) = ((17 * p) / (26 * v)) / (p / v) := by rw [h1, h2, h3],
  have h5 : ((17 * p) / (26 * v)) / (p / v) = (17 / 26) := by field_simp [hv, hp],
  rw [h4, h5],
  exact rfl,
}

end ratio_unit_price_l608_608167


namespace smallest_k_l608_608888

def is_RP_subset (T : Finset ℕ) : Prop := ∀ x ∈ T, ∀ y ∈ T, x ≠ y → Nat.gcd x y = 1

def contains_prime (T : Finset ℕ) : Prop := ∃ x ∈ T, Nat.Prime x

def k_element_RP_subset_contains_prime (S : Finset ℕ) (k : ℕ) : Prop :=
  ∀ T : Finset ℕ, T.card = k → is_RP_subset T → T ⊆ S → contains_prime T

theorem smallest_k (S : Finset ℕ) (S_def : S = Finset.range 2012 \+ 1) :
  ∃ k : ℕ, k_element_RP_subset_contains_prime S k ∧ 
  (∀ k' : ℕ, k' < k → ¬ k_element_RP_subset_contains_prime S k') :=
sorry

end smallest_k_l608_608888


namespace intersection_points_l608_608970

-- Define the polar equations for curves C₁ and C₂
def C₁ (ρ θ : ℝ) : Prop := ρ^2 - 10*ρ*cos θ - 8*ρ*sin θ + 16 = 0
def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * cos θ

-- Define the intersection points proof
theorem intersection_points :
  (C₁ 2 0 ∧ C₂ 2 0) ∧ (C₁ (sqrt 2) (π / 4) ∧ C₂ (sqrt 2) (π / 4)) :=
by
  sorry

end intersection_points_l608_608970


namespace chores_for_cartoon_time_l608_608211

def hours_to_minutes (hours : ℕ) : ℕ := hours * 60

def cartoons_to_chores (cartoon_minutes : ℕ) : ℕ := cartoon_minutes * 8 / 10

theorem chores_for_cartoon_time (h : ℕ) (h_eq : h = 2) : cartoons_to_chores (hours_to_minutes h) = 96 :=
by
  rw [h_eq, hours_to_minutes, cartoons_to_chores]
  -- steps demonstrating transformation from hours to minutes and calculation of chores will follow here
  sorry

end chores_for_cartoon_time_l608_608211


namespace gcd_fx_x_l608_608952

def f (x: ℕ) := (5 * x + 4) * (9 * x + 7) * (11 * x + 3) * (x + 12)

theorem gcd_fx_x (x: ℕ) (h: x % 54896 = 0) : Nat.gcd (f x) x = 112 :=
  sorry

end gcd_fx_x_l608_608952


namespace simultaneous_equations_solution_exists_l608_608558

theorem simultaneous_equations_solution_exists (m : ℝ) :
  ∃ x y : ℝ, y = 3 * m * x + 2 ∧ y = (3 * m - 2) * x + 5 :=
by
  sorry

end simultaneous_equations_solution_exists_l608_608558


namespace common_tangents_count_l608_608610

-- Define the structure representing an ellipse with two foci and a major axis
structure Ellipse :=
  (focus1 focus2 : ℝ × ℝ) -- Focus points of the ellipse
  (semi_major_axis : ℝ) -- Semi-major axis length

-- Definition of the guiding circle related to the focus of an ellipse
def guiding_circle (e : Ellipse) (focus : ℝ × ℝ) : set (ℝ × ℝ) :=
  { p | dist p focus = e.semi_major_axis }

-- Define the problem conditions
variables (E1 E2 : Ellipse)
variable (common_focus : ℝ × ℝ)

-- The common_focus is a focus of both ellipses
def common_focus_condition (E1 E2 : Ellipse) (f : ℝ × ℝ) : Prop :=
  (E1.focus1 = f ∨ E1.focus2 = f) ∧ (E2.focus1 = f ∨ E2.focus2 = f)

-- Lean theorem statement, stating that the number of common tangents can be 0, 1, or 2
theorem common_tangents_count
  (h : common_focus_condition E1 E2 common_focus) :
  ∃ n : ℕ, (n = 0 ∨ n = 1 ∨ n = 2) ∧ -- Count of common tangents
  (n = 0 → disjoint (guiding_circle E1 E1.focus2) (guiding_circle E2 E2.focus2)) ∧
  (n = 1 → ∃ p, p ∈ (guiding_circle E1 E1.focus2) ∧ p ∈ (guiding_circle E2 E2.focus2) ∧ ∀ q, q ≠ p → ¬ (q ∈ (guiding_circle E1 E1.focus2) ∧ q ∈ (guiding_circle E2 E2.focus2))) ∧
  (n = 2 → ∃ p1 p2, p1 ≠ p2 ∧ p1 ∈ (guiding_circle E1 E1.focus2) ∧ p1 ∈ (guiding_circle E2 E2.focus2) ∧ p2 ∈ (guiding_circle E1 E1.focus2) ∧ p2 ∈ (guiding_circle E2 E2.focus2)) :=
sorry

end common_tangents_count_l608_608610


namespace sweet_cookies_more_than_salty_l608_608710

-- Definitions for the given conditions
def sweet_cookies_ate : Nat := 32
def salty_cookies_ate : Nat := 23

-- The statement to prove
theorem sweet_cookies_more_than_salty :
  sweet_cookies_ate - salty_cookies_ate = 9 := by
  sorry

end sweet_cookies_more_than_salty_l608_608710


namespace cos_alpha_sub_beta_l608_608169

noncomputable theory
open_locale classical

variables {α β : ℝ}

theorem cos_alpha_sub_beta (h1 : cos α + cos β = -4/5) (h2 : sin α + sin β = 1/3) :
  cos (α - β) = -28/225 :=
sorry

end cos_alpha_sub_beta_l608_608169


namespace geometric_sequence_seventh_term_l608_608408

theorem geometric_sequence_seventh_term (a r : ℝ) (ha : 0 < a) (hr : 0 < r) 
  (h4 : a * r^3 = 16) (h10 : a * r^9 = 2) : 
  a * r^6 = 2 :=
by
  sorry

end geometric_sequence_seventh_term_l608_608408


namespace parallel_vectors_l608_608280

theorem parallel_vectors (m : ℝ) : (m = 1) ↔ (∃ k : ℝ, (m, 1) = k • (1, m)) := sorry

end parallel_vectors_l608_608280


namespace sand_for_patches_needed_is_correct_l608_608662

noncomputable def sand_needed (rect_length rect_width rect_depth sq_side sq_depth circ_radius circ_depth conversion_rate : ℝ) : ℝ :=
  let rect_volume := rect_length * rect_width * rect_depth
  let sq_volume := sq_side * sq_side * sq_depth
  let circ_area := Real.pi * circ_radius * circ_radius
  let circ_volume := circ_area * circ_depth
  let total_volume := rect_volume + sq_volume + circ_volume
  total_volume * conversion_rate

theorem sand_for_patches_needed_is_correct :
  sand_needed 6 7 2 5 3 2 1.5 3 ≈ 533.55 :=
by
  sorry

end sand_for_patches_needed_is_correct_l608_608662


namespace picture_books_count_l608_608782

theorem picture_books_count (total_books : ℕ) (fiction_books : ℕ) (non_fiction_books : ℕ) (autobiography_books : ℕ) (picture_books : ℕ) 
  (h1 : total_books = 35)
  (h2 : fiction_books = 5)
  (h3 : non_fiction_books = fiction_books + 4)
  (h4 : autobiography_books = 2 * fiction_books)
  (h5 : picture_books = total_books - (fiction_books + non_fiction_books + autobiography_books)) :
  picture_books = 11 := 
  sorry

end picture_books_count_l608_608782


namespace arithmetic_progression_number_of_terms_l608_608769

noncomputable theory
open_locale big_operators

-- Assume the variables and basic definitions
variables {a d : ℕ} -- first term and common difference
variable {n : ℕ} -- number of terms

-- Sum of first k terms in an arithmetic progression
def S (k : ℕ) := k * (2 * a + (k - 1) * d) / 2

-- Sum of the first 13 terms
def S_first_13 := S 13

-- Sum of the last 13 terms
def S_last_13 := 13 * (2 * a + (n - 7) * d)

-- Sum without first 3 terms
def S_without_first_3 := (n - 3) * (2 * a + (n + 4) * d) / 2

-- Sum without last 3 terms
def S_without_last_3 := (n - 3) * (2 * a + (n - 4) * d) / 2

-- Prove the number of terms is 18
theorem arithmetic_progression_number_of_terms : 
  (S_first_13 = S_last_13 / 2) ∧ 
  (S_without_first_3 / S_without_last_3 = 3 / 2) → 
  n = 18 :=
sorry

end arithmetic_progression_number_of_terms_l608_608769


namespace number_of_factors_multiples_of_10_l608_608988

theorem number_of_factors_multiples_of_10 (n : ℕ) (h : n = 2^3 * 3^2 * 5^1) : 
    ∃ k : ℕ, k = 9 ∧ (number_of_positive_factors_multiples_of 10 n) = k := 
sorry

end number_of_factors_multiples_of_10_l608_608988


namespace range_of_k_l608_608609

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 * k + 1}
def A_complement : Set ℝ := {x | 1 < x ∧ x < 3}

theorem range_of_k (k : ℝ) : ((A_complement ∩ (B k)) = ∅) ↔ (k ∈ Set.Iic 0 ∪ Set.Ici 3) := sorry

end range_of_k_l608_608609


namespace A_is_subset_of_B_l608_608578

-- Definition of sets A and B
def A : set ℝ := { x | abs (x - 2) < 1 }
def B : set ℝ := { x | (x - 1) * (x - 4) < 0 }

-- Statement to prove
theorem A_is_subset_of_B : A ⊆ B :=
sorry

end A_is_subset_of_B_l608_608578


namespace find_coordinates_of_point_M_l608_608646

theorem find_coordinates_of_point_M :
  ∃ (M : ℝ × ℝ), 
    (M.1 > 0) ∧ (M.2 < 0) ∧ 
    abs M.2 = 12 ∧ 
    abs M.1 = 4 ∧ 
    M = (4, -12) :=
by
  sorry

end find_coordinates_of_point_M_l608_608646


namespace range_of_a_l608_608276

noncomputable def setA : Set ℝ := { x | x^2 ≤ 1 }
noncomputable def setB (a : ℝ) : Set ℝ := { x | x < a }

theorem range_of_a (a : ℝ) :
  setA ∪ setB a = setB a ↔ a ∈ (1, +∞) := by
  sorry

end range_of_a_l608_608276


namespace largest_inscribed_equilateral_triangle_area_l608_608521

/-- 
Given a circle with a radius of 10 cm, the area of the largest possible inscribed 
equilateral triangle that has one side as the diameter of the circle is 100√3 square centimeters.
-/
theorem largest_inscribed_equilateral_triangle_area (r : ℝ) (h : r = 10) :
    let d := 2 * r
    let a := d
    let height := √(d^2 - r^2)
    let area := 1/2 * a * height
    area = 100 * √3 :=
by
  intros
  simp only [*, pow_two]
  sorry

end largest_inscribed_equilateral_triangle_area_l608_608521


namespace part1_part2_l608_608916

noncomputable def equation (a x : ℝ) : ℝ := (a * x + 1) / (x - 1) - 2 / (1 - x)

-- Part (1): When a = 3, show that x = -2 is a solution
theorem part1 (a x : ℝ) (h : a = 3) : equation a x = 1 ↔ x = -2 := 
by sorry

-- Part (2): If the equation has a root of multiplicity, prove that a = -3
theorem part2 (a : ℝ) (h : ∃ x : ℝ, equation a x = 1 ∧ (differentiable_at ℝ (λ x, equation a x) x ∧ deriv (λ x, equation a x) x = 0)) : a = -3 :=
by sorry

end part1_part2_l608_608916


namespace cube_problem_l608_608186

theorem cube_problem (n : ℕ) (h1 : n > 3) :
  (12 * (n - 4) = (n - 2)^3) → n = 5 :=
by {
  sorry
}

end cube_problem_l608_608186


namespace verifyInequalities_l608_608032

theorem verifyInequalities (a b : ℝ)
  (h_a : a > 0) (h_b : b > 0) :
  ((a + b) * ((1 / a) + (1 / b)) ≥ 4) ∧
  (a^2 + b^2 + 2 ≥ 2 * a + 2 * b) ∧
  (√|a - b| ≥ √a - √b) ∧ 
  ¬ (2 * a * b / (a + b) ≥ √(a * b)) :=
by
  sorry

end verifyInequalities_l608_608032


namespace find_ratio_PS_SR_l608_608308

variable {P Q R S : Type}
variable [MetricSpace P]
variable [MetricSpace Q]
variable [MetricSpace R]
variable [MetricSpace S]

-- Given conditions
variable (PQ QR PR : ℝ)
variable (hPQ : PQ = 6)
variable (hQR : QR = 8)
variable (hPR : PR = 10)
variable (QS : ℝ)
variable (hQS : QS = 6)

-- Points on the segments
variable (PS : ℝ)
variable (SR : ℝ)

-- The theorem to be proven: the ratio PS : SR = 0 : 1
theorem find_ratio_PS_SR (hPQ : PQ = 6) (hQR : QR = 8) (hPR : PR = 10) (hQS : QS = 6) :
    PS = 0 ∧ SR = 10 → PS / SR = 0 :=
by
  sorry

end find_ratio_PS_SR_l608_608308


namespace sum_of_4n_pos_integers_l608_608301

theorem sum_of_4n_pos_integers (n : ℕ) (Sn : ℕ → ℕ)
  (hSn : ∀ k, Sn k = k * (k + 1) / 2)
  (h_condition : Sn (3 * n) - Sn n = 150) :
  Sn (4 * n) = 300 :=
by {
  sorry
}

end sum_of_4n_pos_integers_l608_608301


namespace picture_books_count_l608_608780

-- Definitions based on the given conditions
def total_books : ℕ := 35
def fiction_books : ℕ := 5
def non_fiction_books : ℕ := fiction_books + 4
def autobiographies : ℕ := 2 * fiction_books
def total_non_picture_books : ℕ := fiction_books + non_fiction_books + autobiographies
def picture_books : ℕ := total_books - total_non_picture_books

-- Statement of the problem
theorem picture_books_count : picture_books = 11 :=
by sorry

end picture_books_count_l608_608780


namespace choose_true_props_l608_608817

variables {V : Type*} [inner_product_space ℝ V]

def prop_A (a b : V) : Prop :=
  ∥a∥ = ∥b∥ → a = b

def prop_B (a c a1 c1 : V) : Prop :=
  a - c = a1 - c1

def prop_C (m n p : V) : Prop :=
  m = n ∧ n = p → m = p

def prop_D (a b c : V) : Prop :=
  (a ⬝ b = 0 ∨ ∀ k : ℝ, a = k • b) ∧ (b ⬝ c = 0 ∨ ∀ l : ℝ, b = l • c) → (a ⬝ c = 0 ∨ ∀ m : ℝ, a = m • c)

theorem choose_true_props (a b c m n p a1 c1 : V) :
  ¬(prop_A a b) ∧ prop_B a c a1 c1 ∧ prop_C m n p ∧ ¬(prop_D a b c) :=
by sorry

end choose_true_props_l608_608817


namespace largest_possible_BD_l608_608678

noncomputable def cyclic_quadrilateral_max_diag (a b c d : ℕ) (h_lt : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10)
  (h_distinct : (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d))
  (h_sum : b + c = a + d) : ℝ :=
  let BD := Real.sqrt ((a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2) / 2) in BD

theorem largest_possible_BD (a b c d : ℕ) (h_lt : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10)
  (h_distinct : (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d))
  (h_sum : b + c = a + d) : 
  cyclic_quadrilateral_max_diag a b c d h_lt h_distinct h_sum = Real.sqrt (191 / 2) :=
sorry

end largest_possible_BD_l608_608678


namespace picture_books_count_l608_608781

theorem picture_books_count (total_books : ℕ) (fiction_books : ℕ) (non_fiction_books : ℕ) (autobiography_books : ℕ) (picture_books : ℕ) 
  (h1 : total_books = 35)
  (h2 : fiction_books = 5)
  (h3 : non_fiction_books = fiction_books + 4)
  (h4 : autobiography_books = 2 * fiction_books)
  (h5 : picture_books = total_books - (fiction_books + non_fiction_books + autobiography_books)) :
  picture_books = 11 := 
  sorry

end picture_books_count_l608_608781


namespace coefficient_and_degree_of_monomial_l608_608735

variable (x y : ℝ)

def monomial : ℝ := -2 * x * y^3

theorem coefficient_and_degree_of_monomial :
  ( ∃ c : ℝ, ∃ d : ℤ, monomial x y = c * x * y^d ∧ c = -2 ∧ d = 4 ) :=
by
  sorry

end coefficient_and_degree_of_monomial_l608_608735


namespace probability_even_product_l608_608560

theorem probability_even_product :
  ((nat.choose 9 2).to_real) = (36 : ℝ) → 
  let favorable_outcomes := (nat.choose 4 2 + nat.choose 4 1 * nat.choose 5 1) in
  ((favorable_outcomes / nat.choose 9 2) = (13 / 18 : ℝ)) :=
begin
  intro total_draws,
  let favorable_outcomes := (nat.choose 4 2) + (nat.choose 4 1) * (nat.choose 5 1),
  have h_favorable := favorable_outcomes,
  have h_total := total_draws,
  sorry, -- Proof goes here
end

end probability_even_product_l608_608560


namespace problem_part_1_problem_part_2_l608_608564

theorem problem_part_1 (a b c : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x ^ 2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_cond : ∀ x, -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) :
  |c| ≤ 1 :=
by
  sorry

theorem problem_part_2 (a b c : ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x ^ 2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_cond : ∀ x, -1 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → |g x| ≤ 2 :=
by
  sorry

end problem_part_1_problem_part_2_l608_608564


namespace find_y_l608_608215

-- Definitions of lengths based on the given conditions
def OA := 6
def OB := 7
def OD := 6
def OC := 9
def BD := 10

-- Question: Prove that y = 11.5 based on the given conditions. 
theorem find_y (y : ℝ) (h1 : sqrt (OA^2 + OC^2 + (2 * 6 * 9 * (1/7))) = y) : y = 11.5 :=
by {
  have sqrt_val : sqrt ((6^2 + 9^2 + (2 * 6 * 9 * (1/7)))) = sqrt (132.4285714286), sorry,
  rw sqrt_val at h1,
  exact h1,
}

end find_y_l608_608215


namespace probability_of_x_plus_y_lt_4_l608_608098

open Classical -- To handle probability and random selection
open Set -- For geometric notions
open Filter -- For measure and integration

noncomputable def probability_condition (x y : ℝ) (h : 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) : Prop := x + y < 4

theorem probability_of_x_plus_y_lt_4 :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) →
                pr (λ (xy : ℝ × ℝ), probability_condition xy.1 xy.2 (and.intro (and.left xy.2) (and.right (and.left xy.2)))) = 7/9 :=
by sorry

end probability_of_x_plus_y_lt_4_l608_608098


namespace total_enemies_l608_608321

theorem total_enemies (points_per_enemy defeated_enemies undefeated_enemies total_points total_enemies : ℕ)
  (h1 : points_per_enemy = 5) 
  (h2 : undefeated_enemies = 6) 
  (h3 : total_points = 10) :
  total_enemies = 8 := by
  sorry

end total_enemies_l608_608321


namespace quadratic_function_expression_exists_domain_range_exists_l608_608242

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem quadratic_function_expression_exists (a b : ℝ) (h_a : a ≠ 0) (h1 : ∀ x : ℝ, f a b (x-1) = f a b (3-x)) (h2 : ∀ x : ℝ, f a b x = 2 * x → x ≠ x) :
  ∃ (a b : ℝ), f a b x = -x^2 + 2 * x :=
by pasting
  sorry

theorem domain_range_exists (m n : ℝ) (hmn : m < n) (f : ℝ → ℝ) (h_dom : ∀ x : ℝ, m ≤ x ∧ x ≤ n) (h_range : ∀ y : ℝ, 4 * m ≤ y ∧ y ≤ 4 * n) :
  ∃ (m n : ℝ), m = -1 ∧ n = 0 :=
by pasting
  sorry

end quadratic_function_expression_exists_domain_range_exists_l608_608242


namespace range_of_x_l608_608261

theorem range_of_x 
  (f : ℝ → ℝ) 
  (hf : ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f(x) < f(y)) 
  (g : ℝ → ℝ) 
  (hg : ∀ x, g(x) = - f(|x|)) 
  (hlog : ∀ x, g(log x) > g(1)) : 
  ∀ x, hlog x → (1 / 10 < x ∧ x < 10) :=
sorry

end range_of_x_l608_608261


namespace number_of_children_l608_608538

theorem number_of_children (crayons_per_child total_crayons : ℕ) (h1 : crayons_per_child = 12) (h2 : total_crayons = 216) : total_crayons / crayons_per_child = 18 :=
by
  have h3 : total_crayons / crayons_per_child = 216 / 12 := by rw [h1, h2]
  norm_num at h3
  exact h3

end number_of_children_l608_608538


namespace cistern_wet_surface_area_l608_608821

-- Define conditions
def length : ℝ := 10
def width : ℝ := 8
def height : ℝ := 1.5

-- Define areas
def area_bottom : ℝ := length * width
def area_longer_walls : ℝ := 2 * (length * height)
def area_shorter_walls : ℝ := 2 * (width * height)

-- Define total wet surface area
def total_wet_surface_area : ℝ :=
  area_bottom + area_longer_walls + area_shorter_walls

-- The theorem to prove 
theorem cistern_wet_surface_area : 
  total_wet_surface_area = 134 :=
by
  sorry

end cistern_wet_surface_area_l608_608821


namespace larger_number_hcf_lcm_l608_608042

theorem larger_number_hcf_lcm (a b : ℕ) (h1 : Nat.gcd a b = 84) (h2 : Nat.lcm a b = 21) (h3 : a = b / 4) : max a b = 84 :=
by
  sorry

end larger_number_hcf_lcm_l608_608042


namespace find_number_l608_608468

theorem find_number (x : ℚ) (h : 0.5 * x = (3/5) * x - 10) : x = 100 := 
sorry

end find_number_l608_608468


namespace nine_circles_problem_l608_608704

def is_triangle_valid (grid : Fin 3 × Fin 3 → ℕ) (triangles : list (list (Fin 3 × Fin 3))) (target_sum : ℕ) : Prop :=
  ∀ triangle ∈ triangles, target_sum = (triangle.map grid).sum

def unique_numbers_1_to_9 (grid : Fin 3 × Fin 3 → ℕ) : Prop :=
  let numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9} in
  (Finset.image grid Finset.univ).val = numbers

theorem nine_circles_problem : ∃ grid : (Fin 3 × Fin 3) → ℕ,
  unique_numbers_1_to_9 grid ∧
  is_triangle_valid grid
    [ [(0, 0), (0, 1), (0, 2)]
    , [(0, 0), (1, 0), (2, 0)]
    , [(0, 0), (1, 1), (2, 2)]
    , [(0, 1), (1, 1), (2, 1)]
    , [(0, 2), (1, 2), (2, 2)]
    , [(1, 0), (1, 1), (1, 2)]
    , [(2, 0), (2, 1), (2, 2)]
    ] 15 := sorry

end nine_circles_problem_l608_608704


namespace picture_books_count_l608_608779

-- Definitions based on the given conditions
def total_books : ℕ := 35
def fiction_books : ℕ := 5
def non_fiction_books : ℕ := fiction_books + 4
def autobiographies : ℕ := 2 * fiction_books
def total_non_picture_books : ℕ := fiction_books + non_fiction_books + autobiographies
def picture_books : ℕ := total_books - total_non_picture_books

-- Statement of the problem
theorem picture_books_count : picture_books = 11 :=
by sorry

end picture_books_count_l608_608779


namespace favorite_numbers_parity_l608_608661

variables (D J A H : ℤ)

def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem favorite_numbers_parity
  (h1 : odd (D + 3 * J))
  (h2 : odd ((A - H) * 5))
  (h3 : even (D * H + 17)) :
  odd D ∧ even J ∧ even A ∧ odd H := 
sorry

end favorite_numbers_parity_l608_608661


namespace xyz_equal_l608_608534

theorem xyz_equal (x y z : ℝ) (hx : ⌊x⌋ + frac y = z) (hy : ⌊y⌋ + frac z = x) (hz : ⌊z⌋ + frac x = y) : x = y ∧ y = z :=
by
  sorry

end xyz_equal_l608_608534


namespace find_value_of_a_l608_608607

theorem find_value_of_a 
  (a : ℝ)
  (A : set ℝ := {1, a^2})
  (B : set ℝ := {a, -1})
  (h_union : A ∪ B = {-1, a, 1}) : 
  a = 0 :=
sorry

end find_value_of_a_l608_608607


namespace R2_area_l608_608577

-- Definitions for the conditions
def R1_side1 : ℝ := 4
def R1_area : ℝ := 16
def R2_diagonal : ℝ := 10
def similar_rectangles (R1 R2 : ℝ × ℝ) : Prop := (R1.fst / R1.snd = R2.fst / R2.snd)

-- Main theorem
theorem R2_area {a b : ℝ} 
  (R1_side1 : a = 4)
  (R1_area : a * a = 16) 
  (R2_diagonal : b = 10)
  (h : similar_rectangles (a, a) (b / (10 / (2 : ℝ)), b / (10 / (2 : ℝ)))) : 
  b * b / (2 : ℝ) = 50 :=
by
  sorry

end R2_area_l608_608577


namespace semi_minor_axis_is_sqrt7_l608_608315

open Real

def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x1 - x2) ^ 2 + (y1 - y2) ^ 2)

noncomputable def ellipse_semi_minor_axis {c f e : ℝ × ℝ} (hc : c = (-2, 1)) (hf : f = (-3, 0)) (he : e = (-2, 4)) : ℝ :=
  let a := distance c.1 c.2 e.1 e.2 in
  let c := distance c.1 c.2 f.1 f.2 in
  sqrt (a ^ 2 - c ^ 2)

theorem semi_minor_axis_is_sqrt7 : ellipse_semi_minor_axis (-2, 1) (-3, 0) (-2, 4) = sqrt 7 := 
  by
  sorry

end semi_minor_axis_is_sqrt7_l608_608315


namespace police_and_military_service_criteria_l608_608634

-- Definitions from conditions
variable (Population : Type) [Finite Population]
variable (height : Population → ℝ)
variable (position : Population → ℝ × ℝ)
variable (R r : Population → ℝ)
variable (neighbors_police : Population → Set Population)
variable (neighbors_military : Population → Set Population)
variable (percent_police_criteria fulfilled_police : Population → Prop)
variable (percent_military_criteria fulfilled_military : Population → Prop)

-- Conditions
axiom A1 : ∀ p : Population, neighbors_police p = {n ∈ Population | (dist (position p) (position n) < R p)}
axiom A2 : ∀ p : Population, neighbors_military p = {n ∈ Population | (dist (position p) (position n) < r p)}
axiom A3 : ∀ p : Population, height p > 0 
axiom A4 : ∀ p : Population, size(neighbors_police p) > 0 ∧ size(neighbors_military p) > 0
axiom A5 : ∀ p : Population, percent_police_criteria p ↔ (∃ k < 0.8, height p > k * height (neighbors_police p))
axiom A6 : ∀ p : Population, percent_military_criteria p ↔ (∃ k < 0.8, height p < k * height (neighbors_military p))
axiom A7 : ∀ p : Population, fulfilled_police p ↔ ∀ q ∈ neighbors_police p, height p > height q
axiom A8 : ∀ p : Population, fulfilled_military p ↔ ∀ q ∈ neighbors_military p, height p < height q

-- Theorem
theorem police_and_military_service_criteria :
  (∃ Psubset : Set Population, size(Psubset) ≥ 0.9 * size(Population)
      ∧ ∀ p ∈ Psubset, percent_police_criteria p ∧ percent_military_criteria p) :=
sorry

end police_and_military_service_criteria_l608_608634


namespace product_of_roots_of_polynomial_l608_608184

theorem product_of_roots_of_polynomial :
  (∃ (x : ℝ), x^4 - 12 * x^3 + 50 * x^2 + 48 * x - 35 = 0) →
  ∏ (root : ℝ) in {x | x^4 - 12 * x^3 + 50 * x^2 + 48 * x - 35 = 0}, root = 35 :=
begin
  sorry
end

end product_of_roots_of_polynomial_l608_608184


namespace find_S7_l608_608244

variable {a : ℕ → ℚ} {S : ℕ → ℚ}

axiom a1_def : a 1 = 1 / 2
axiom a_next_def : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * S n + 1
axiom S_def : ∀ n : ℕ, S (n + 1) = S n + a (n + 1)

theorem find_S7 : S 7 = 1457 / 2 := by
  sorry

end find_S7_l608_608244


namespace distance_A_C_after_folding_of_rhombus_l608_608759

variable (a : ℝ)

-- Define the rhombus and its properties
def is_rhombus (A B C D : ℝ × ℝ) : Prop := 
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  let (x4, y4) := D
  dist (x1, y1) (x2, y2) = a ∧
  dist (x2, y2) (x3, y3) = a ∧
  dist (x3, y3) (x4, y4) = a ∧
  dist (x4, y4) (x1, y1) = a

-- Define the 60 degrees angle condition
def is_angle_60_deg (A B C D : ℝ × ℝ) : Prop := 
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  let (x4, y4) := D
  ∠ (x1, y1) (x2, y2) (x4, y4) = 60

-- Define the dihedral angle condition after folding
def is_fold_dihedral_120_deg (A B C D : ℝ × ℝ) : Prop := 
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  let (x4, y4) := D
  dihedral_angle (x1, y1) (x2, y2) (x3, y3) (x4, y4) = 120

-- Define the final theorem statement
theorem distance_A_C_after_folding_of_rhombus
  (A B C D : ℝ × ℝ)
  (h1 : is_rhombus A B C D)
  (h2 : is_angle_60_deg A B C D)
  (h3 : is_fold_dihedral_120_deg A B C D) : 
  dist (A.1, A.2) (C.1, C.2) = (sqrt 3 / 4) * a :=
sorry

end distance_A_C_after_folding_of_rhombus_l608_608759


namespace frictional_force_correct_l608_608472

variable (m : ℝ) (theta : ℝ) (mu : ℝ) (g : ℝ) (cos_theta : ℝ)

def frictional_force (m : ℝ) (g : ℝ) (cos_theta : ℝ) (mu : ℝ) : ℝ :=
  mu * m * g * cos_theta

theorem frictional_force_correct :
  (m = 1) → (g = 10) → (theta = 30) → (mu = 0.6) → (cos_theta = (Real.cos (Float.pi / 6))) →
  round (frictional_force m g cos_theta mu) = 5 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end frictional_force_correct_l608_608472


namespace projection_is_correct_l608_608878

def vector_v := (2, 4, 6 : ℝ)
def normal_vector_n := (6, -2, 8 : ℝ)
def plane_eq := λ x y z : ℝ, 6 * x - 2 * y + 8 * z = 0

def projection_of_v_onto_plane := (⟨-1, 5, 2⟩ : ℝ × ℝ × ℝ)

theorem projection_is_correct :
  let v := (2, 4, 6 : ℝ) in
  let n := (6, -2, 8 : ℝ) in
  let p := (⟨-1, 5, 2⟩ : ℝ × ℝ × ℝ) in
  v - p = (v.1 - p.1, v.2 - p.2, v.3 - p.3) ∧
  is_parallel (v - p, n) ∧
  plane_eq p.1 p.2 p.3 :=
begin
  sorry
end

noncomputable def is_parallel (v1 v2 : ℝ × ℝ × ℝ) : Prop :=
v1.1 * v2.2 = v1.2 * v2.1 ∧ v1.1 * v2.3 = v1.3 * v2.1 ∧ v1.2 * v2.3 = v1.3 * v2.2

end projection_is_correct_l608_608878


namespace sum_first_60_terms_l608_608418

theorem sum_first_60_terms :
  ∃ (a : ℕ → ℝ), 
    a 1 = 1 ∧ 
    (∀ n, a (n + 2) + a n * (Real.cos (n * Real.pi)) = ((abs (Real.sin (n * Real.pi / 2)) - 1) * n)) ∧ 
    (∑ n in Finset.range 60, a (n + 1) ) = -420 :=
by
  sorry

end sum_first_60_terms_l608_608418


namespace strawberry_cost_l608_608129

variables (S C : ℝ)

theorem strawberry_cost :
  (C = 6 * S) ∧ (5 * S + 5 * C = 77) → S = 2.2 :=
by
  sorry

end strawberry_cost_l608_608129


namespace beam_equation_correctness_l608_608460

-- Define the conditions
def total_selling_price : ℕ := 6210
def freight_per_beam : ℕ := 3

-- Define the unknown quantity
variable (x : ℕ)

-- State the theorem
theorem beam_equation_correctness
  (h1 : total_selling_price = 6210)
  (h2 : freight_per_beam = 3) :
  freight_per_beam * (x - 1) = total_selling_price / x := 
sorry

end beam_equation_correctness_l608_608460


namespace area_triangle_LGH_l608_608331

noncomputable def radius : ℝ := 10
noncomputable def length_EF : ℝ := 12
noncomputable def length_GH : ℝ := 16
noncomputable def LP : ℝ := 24

theorem area_triangle_LGH :
  let height_GH := radius^2 - (length_GH / 2)^2 in
  ∃ (base : ℝ) (height : ℝ),
    base = length_GH ∧ height = 2 * (radius - (height_GH).sqrt) ∧
    (1/2) * base * height = 48 :=
by
  let base := length_GH
  let height := 2 * (radius - (radius^2 - (length_GH / 2)^2).sqrt)
  use base, height
  sorry

end area_triangle_LGH_l608_608331


namespace min_angle_ACB_line_eq_l608_608087

noncomputable def pointM : ℝ × ℝ := (1/2, 1)
noncomputable def centerC : ℝ × ℝ := (1, 0)

def slope (p1 p2 : ℝ × ℝ) : ℝ := 
if p1.1 = p2.1 then 0 else (p2.2 - p1.2) / (p2.1 - p1.1)

def lineEquation (m : ℝ) (x1 y1 : ℝ) : ℝ × ℝ → Prop :=
λ p, p.2 = m * (p.1 - x1) + y1

def circleEq : ℝ × ℝ → Prop :=
λ p, (p.1 - 1)^2 + p.2^2 = 4

theorem min_angle_ACB_line_eq :
  let l := λ p, 2 * p.1 - 4 * p.2 + 3 = 0 in
  (lineEquation (1/2) (1/2) 1 = l) ∧ 
  slope centerC pointM = -2 ∧ 
  ∀ A B : ℝ × ℝ, 
    circleEq A → circleEq B → 
    collinear A B centerC → 
    (∀ p : ℝ × ℝ, l p ↔ p.1 ≠ 1/2 → slope centerC pointM = -2 →
    lineEquation (1/2) (1/2) 1 p) := 
sorry

end min_angle_ACB_line_eq_l608_608087


namespace N_not_fifth_power_main_statement_l608_608532

/-- Define the sequence e_k such that e_k can only be -1 or 1 --/
def e : ℕ → ℤ := sorry /- specify the sequence (e_k). This is assumed to be -1 or 1 for 1 ≤ k ≤ 60 -/

noncomputable def k_pow_k_pow_k (k : ℕ) : ℤ :=
  (k : ℤ)^(k^k)

/-- Define the sum N -/
def N : ℤ :=
∑ k in finset.range 60, e (k + 1) * k_pow_k_pow_k (k + 1)

/-- Prove that N cannot be a fifth power of an integer -/
theorem N_not_fifth_power (x : ℤ) : ¬ (∃ m : ℤ, x = m^5) :=
begin
  sorry /- proof here -/
end

/-- The main statement: N is not a fifth power of any integer -/
theorem main_statement : N_not_fifth_power N :=
  sorry /- proof here -/

end N_not_fifth_power_main_statement_l608_608532


namespace solve_for_z_l608_608723

theorem solve_for_z (z : ℂ) (h : 3 + 2 * complex.I * z = 2 - 5 * complex.I * z) : 
  z = (1 / 7) * complex.I :=
by sorry

end solve_for_z_l608_608723


namespace small_square_area_l608_608755

theorem small_square_area (A_tile : ℝ) (PQ : ℝ) (n : ℕ) (hyp_congruent: n = 8 / 100) (area_equal: A_tile = 49) (hyp_length: PQ = 5) :
  let A_small := PQ^2 in A_small = 1 :=
by
  sorry

end small_square_area_l608_608755


namespace range_of_a_l608_608946

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2^a) → (-2 ≤ x ∧ x < 2)) → a ≤ 1 := 
by {
  intro h,
  sorry
}

end range_of_a_l608_608946


namespace correct_average_marks_l608_608636

theorem correct_average_marks :
  let num_students := 40
  let reported_avg := 65
  let incorrect_marks := [100, 85, 15]
  let correct_marks := [20, 50, 55]
  let incorrect_total_sum := num_students * reported_avg
  let wrong_sum := List.sum incorrect_marks
  let correct_sum := List.sum correct_marks
  let correct_total_sum := incorrect_total_sum - wrong_sum + correct_sum
  let correct_avg := (correct_total_sum : ℚ) / num_students
  correct_avg = 63.125 :=
by
  let num_students := 40
  let reported_avg := 65
  let incorrect_marks := [100, 85, 15]
  let correct_marks := [20, 50, 55]
  let incorrect_total_sum := num_students * reported_avg
  let wrong_sum := List.sum incorrect_marks
  let correct_sum := List.sum correct_marks
  let correct_total_sum := incorrect_total_sum - wrong_sum + correct_sum
  let correct_avg := (correct_total_sum : ℚ) / num_students
  sorry

end correct_average_marks_l608_608636


namespace distance_equals_absolute_value_l608_608708

def distance_from_origin (x : ℝ) : ℝ := abs x

theorem distance_equals_absolute_value (x : ℝ) : distance_from_origin x = abs x :=
by
  sorry

end distance_equals_absolute_value_l608_608708


namespace emmanuel_stays_in_guam_for_days_l608_608896

-- Definitions
def international_data_cost_per_day := 3.50
def regular_monthly_charges := 175
def total_charges_for_december := 210

-- Theorem statement
theorem emmanuel_stays_in_guam_for_days :
  (total_charges_for_december - regular_monthly_charges) / international_data_cost_per_day = 10 := by sorry

end emmanuel_stays_in_guam_for_days_l608_608896


namespace roots_of_equation_l608_608913

noncomputable def findRoots : list ℚ :=
[-3, 5]

theorem roots_of_equation (x : ℚ) :
  (21 / (x^2 - 9) - 3 / (x - 3) = 2) ↔ (x = -3 ∨ x = 5) :=
by
  sorry

end roots_of_equation_l608_608913


namespace bin_sum_sub_eq_l608_608181

-- Define binary numbers
def b1 := 0b101110  -- binary 101110_2
def b2 := 0b10101   -- binary 10101_2
def b3 := 0b111000  -- binary 111000_2
def b4 := 0b110101  -- binary 110101_2
def b5 := 0b11101   -- binary 11101_2

-- Define the theorem
theorem bin_sum_sub_eq : ((b1 + b2) - (b3 - b4) + b5) = 0b1011101 := by
  sorry

end bin_sum_sub_eq_l608_608181


namespace OM_perpendicular_BS_l608_608322

-- Definitions and Conditions
variable {A B C M O S : Type}
variable [IsTriangle ABC]
variable [IsIsosceles ABC (side := AB)]
variable [PointOnLine M BC]
variable [Circumcenter O ABC]
variable [Incenter S ABC]
variable [LineParallel SM AC]

-- Given these conditions, we need to prove the following statement
theorem OM_perpendicular_BS
  (triangle_isosceles : is_isosceles ABC AB)
  (point_M_on_BC : point_on_line M BC)
  (circumcenter_O : circumscribed_circle_center O ABC)
  (incenter_S : inscribed_circle_center S ABC)
  (SM_parallel_AC : line_parallel SM AC) :
  perpendicular_line OM BS :=
sorry

end OM_perpendicular_BS_l608_608322


namespace sum_first_40_terms_l608_608374

theorem sum_first_40_terms (a : ℕ → ℕ) (h : ∀ n : ℕ, a (n + 1) = a n + 2 * (n + 1)) (a1 : a 1 = 2) :
  (∑ n in Finset.range 40, (-1)^(n+1) * a (n + 1)) = 840 :=
sorry

end sum_first_40_terms_l608_608374


namespace correct_statements_l608_608430

theorem correct_statements (population_size : ℕ) (sample_size : ℕ) (total_statements : ℕ) : 
  population_size = 14000 ∧ 
  sample_size = 1000 ∧ 
  total_statements = 4 → 
  ∃ (correct_statements_count : ℕ), correct_statements_count = 3 :=
by {
  -- Given Conditions
  intros h,
  -- Correct statements count must be 3
  use 3,
  -- The proof logic is skipped
  sorry
}

end correct_statements_l608_608430


namespace trajectory_problems_l608_608091

theorem trajectory_problems :
  (∃ (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ), F1 = (-√3, 0) ∧ F2 = (√3, 0) ∧ (dist P F1 + dist P F2 = 4) 
    → (∃ (x y : ℝ), (x^2 / 4 + y^2 = 1)) 
    ∧ 
    (∃ (G H R : ℝ × ℝ) (x1 y1 : ℝ),
      G ∈ {X | (X.1^2 / 4 + X.2^2 = 1)} 
      ∧ H.2 = 0 
      ∧ G = ((R.1, R.2/2)) 
      → R ∈ {Y | (Y.1^2 + Y.2^2 = 4)}) 
    ∧ 
    (∃ (A C P Q M N : ℝ × ℝ),
      C ∈ {Z | (Z.1^2 + (Z.2 - 3)^2 = 4)} 
      ∧ P ∈ {Q | (Q.1^2 + (Q.2 - 3)^2 = 4)} 
      ∧ Q ∈ {Q | (Q.1^2 + (Q.2 - 3)^2 = 4)} 
      ∧ M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) 
      ∧ N ∈ {(x, y) | x + 3 * y + 6 = 0} 
      → (-(1, 0) + M) ⋅ (-(1, 0) + N) = -6)) :=
sorry

end trajectory_problems_l608_608091


namespace minimum_m_l608_608294

noncomputable def f (a x : ℝ) : ℝ := 2^((abs (x - a)))

theorem minimum_m (m : ℝ) (a : ℝ) :
  (∀ x, f 1 (1 + x) = f 1 (1 - x)) →
  (∀ x, f a x = 2^((abs (x - a)))) →
  ((∀ x y, x ≤ y → f 1 x ≤ f 1 y) ↔ m = 1) :=
by 
   intros Hsymm Hdef Hmono
   sorry

end minimum_m_l608_608294


namespace largest_integer_k_l608_608533

def S : ℕ → ℕ
| 1       := 3
| (n + 1) := 3^(S n)

noncomputable def C := (S 2023)^(S 2023)
noncomputable def D := (S 2023)^C

theorem largest_integer_k :
  ∃ k : ℕ, k = 2024 ∧ ∀ n < 2024, (nat.iterate (λ x, real.logb 3 x) n (real.logb 3 D)).is_zero
:= sorry

end largest_integer_k_l608_608533


namespace mittens_pairing_possible_l608_608639

/--
In a kindergarten's lost and found basket, there are 30 mittens: 
10 blue, 10 green, 10 red, 15 right-hand, and 15 left-hand. 

Prove that it is always possible to create matching pairs of one right-hand 
and one left-hand mitten of the same color for 5 children.
-/
theorem mittens_pairing_possible : 
  (∃ (right_blue left_blue right_green left_green right_red left_red : ℕ), 
    right_blue + left_blue + right_green + left_green + right_red + left_red = 30 ∧
    right_blue ≤ 10 ∧ left_blue ≤ 10 ∧
    right_green ≤ 10 ∧ left_green ≤ 10 ∧
    right_red ≤ 10 ∧ left_red ≤ 10 ∧
    right_blue + right_green + right_red = 15 ∧
    left_blue + left_green + left_red = 15) →
  (∃ right_blue left_blue right_green left_green right_red left_red,
    min right_blue left_blue + 
    min right_green left_green + 
    min right_red left_red ≥ 5) :=
sorry

end mittens_pairing_possible_l608_608639


namespace handshake_count_250_l608_608003

theorem handshake_count_250 (n m : ℕ) (h1 : n = 5) (h2 : m = 5) :
  (n * m * (n * m - 1 - (n - 1))) / 2 = 250 :=
by
  -- Traditionally the theorem proof part goes here but it is omitted
  sorry

end handshake_count_250_l608_608003


namespace min_value_of_mn_squared_l608_608950

theorem min_value_of_mn_squared 
  (a b c : ℝ) 
  (h : a^2 + b^2 = c^2) 
  (m n : ℝ) 
  (h_point : a * m + b * n + 2 * c = 0) : 
  m^2 + n^2 = 4 :=
sorry

end min_value_of_mn_squared_l608_608950


namespace probability_set_correct_l608_608483

theorem probability_set_correct :
  (log 10 (7 / 4)) = 3 * (log 10 (5 / 4)) := 
sorry

end probability_set_correct_l608_608483


namespace palindrome_divisibility_probability_l608_608498

-- Definitions
def is_three_digit_palindrome (n : ℕ) : Prop := n / 10100 * 101 + n % 10100 % 1010 / 100 * 10 = n
def is_four_digit_palindrome (n : ℕ) : Prop := n / 1000000 * 1001 + n % 1000000 % 10000 / 1000 * 110 = n
def divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

-- Theorem statement
theorem palindrome_divisibility_probability :
  (∀ n, is_three_digit_palindrome n → divisible_by_11 n) →
  (∀ n, is_four_digit_palindrome n → divisible_by_11 n) →
  (probability (λ n, is_three_digit_palindrome n ∧ divisible_by_11 n) *
   probability (λ n, is_four_digit_palindrome n ∧ divisible_by_11 n)
  = 1 / 81) :=
sorry

end palindrome_divisibility_probability_l608_608498


namespace problem_1_problem_2_l608_608611

noncomputable def vector_a (omega x : ℝ) : ℝ × ℝ := (sqrt 3 * cos (omega * x), sin (omega * x))
noncomputable def vector_b (omega x : ℝ) : ℝ × ℝ := (sin (omega * x), 0)

noncomputable def f (omega x k : ℝ) : ℝ :=
  let a := vector_a omega x
  let b := vector_b omega x
  (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2 + k

theorem problem_1 (omega : ℝ) (h_omega : 0 < omega) :
  (∀ x k, let fx := f omega x k in ∀ d, d ≥ π / 2 → d ≤ pi / omega / 2 → True) →
  0 < omega ∧ omega ≤ 1 :=
sorry

theorem problem_2 (k : ℝ) :
  (let omega := 1 in (f omega (π / 6) k = 2) →
  k = 1) :=
sorry

end problem_1_problem_2_l608_608611


namespace angle_PAK_eq_angle_MAQ_l608_608233

-- Declare the necessary points and lines as given in the conditions:
variables {A M P Q K : Point}
variables {lineAB lineAC : Line}

-- Define the conditions
axiom cond1 : M ∈ Interior (angle A B C)
axiom cond2 : Perpendicular (lineThrough M P) lineAB
axiom cond3 : Perpendicular (lineThrough M Q) lineAC
axiom cond4 : Perpendicular (lineThrough A K) (lineThrough P Q)

-- The theorem to be proved
theorem angle_PAK_eq_angle_MAQ :
  ∠ P A K = ∠ M A Q :=
sorry

end angle_PAK_eq_angle_MAQ_l608_608233


namespace multiple_is_eight_l608_608848

theorem multiple_is_eight (m : ℝ) (h : 17 = m * 2.625 - 4) : m = 8 :=
by
  sorry

end multiple_is_eight_l608_608848


namespace midpoint_D_l608_608252

namespace TranslationProof

-- Define the original points D, J, and H
def D : (ℝ × ℝ) := (2, 3)
def J : (ℝ × ℝ) := (3, 7)
def H : (ℝ × ℝ) := (7, 3)

-- Translation vector
def translation : (ℝ × ℝ) := (3, -1)

-- Define a function to translate a point by a given vector
def translate (p vector : ℝ × ℝ) : (ℝ × ℝ) :=
  (p.1 + vector.1, p.2 + vector.2)

-- Define the points D' and H' after translation
def D' := translate D translation
def H' := translate H translation

-- Define a function to calculate the midpoint of a segment
def midpoint (p1 p2 : ℝ × ℝ) : (ℝ × ℝ) :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- State the theorem to prove the midpoint of D'H' is (7.5, 2)
theorem midpoint_D'_H'_is_correct : midpoint D' H' = (7.5, 2) :=
  sorry

end TranslationProof

end midpoint_D_l608_608252


namespace packs_of_white_tshirts_l608_608190

theorem packs_of_white_tshirts (W : ℕ) (h1 : 6 * W + 27 = 57) : W = 5 :=
by {
  sorry,
}

end packs_of_white_tshirts_l608_608190


namespace range_of_m_l608_608235

-- Part I
def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x + cos x ^ 2

def smallest_period_f : ℝ := π

def intervals_of_monotonic_increase_f (k : ℤ) : set (set ℝ) :=
  {I | I = set.Icc (k * π - π / 3) (k * π + π / 6) }

-- Part II
variables (a b c : ℝ) (A B C : ℝ)
hypothesis (h1: 0 < A ∧ A < π/2) (h2: 0 < B ∧ B < π/2) (h3: 0 < C ∧ C < π/2) (h4: A + B + C = π)

def m (a b c : ℝ) : ℝ := (a^2 + b^2 + c^2) / (a * b)

theorem range_of_m (a b c A C : ℝ) (hacute: ∀ {x}, 0 < x ∧ x < π/2) : 
  f(C) = 1 → m a b c ∈ set.Icc 3 4 := 
sorry

end range_of_m_l608_608235


namespace general_formula_max_m_l608_608268

-- Definitions
def terms_positive (a : ℕ → ℤ) := ∀ n : ℕ, 0 < a n
def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) := ∀ n : ℕ, S n = ∑ i in Finset.range (n + 1), a i
def sum_of_squares_of_first_n_terms (a : ℕ → ℤ) (T : ℕ → ℤ) := ∀ n : ℕ, T n = ∑ i in Finset.range (n + 1), (a i)^2
def sequence_condition (S T : ℕ → ℤ) := ∀ n : ℕ, 2 * T n = S n ^ 2 + 2 * S n
def sum_ma_geq (S a : ℕ → ℤ) (m : ℤ) := ∀ n : ℕ, S n - m * a n ≥ n^2 - n

-- Part 1: Prove the general term formula
theorem general_formula (a S T : ℕ → ℤ)
    (h1 : terms_positive a)
    (h2 : sum_of_first_n_terms a S)
    (h3 : sum_of_squares_of_first_n_terms a T)
    (h4 : sequence_condition S T):
    ∀ n : ℕ, a n = 2 * 3^n := 
sorry

-- Part 2: Find the maximum value of m
theorem max_m (a S T : ℕ → ℤ) (m : ℤ)
    (h1 : terms_positive a)
    (h2 : sum_of_first_n_terms a S)
    (h3 : sum_of_squares_of_first_n_terms a T)
    (h4 : sequence_condition S T)
    (h5 : sum_ma_geq S a m):
    m ≤ 1 := 
sorry

end general_formula_max_m_l608_608268


namespace Sn_circumference_sum_l608_608357

noncomputable def S (n : ℕ) : ℝ :=
  ∑ k in Finset.range (n+1), π / (2 * (k+1)^2)

theorem Sn_circumference_sum (n : ℕ) :
  ∃ a : ℕ → ℝ, (∀ i < n, a i > 0) ∧ a 0 = 1/2 ∧
  (∀ i < n-1, a i > a (i+1)) ∧ (S n = π / 2 * ∑ k in Finset.range (n+1), 1 / (k+1)^2) :=
sorry

end Sn_circumference_sum_l608_608357


namespace find_angle_C_distance_C_to_AD_l608_608306

-- Define the given conditions and problem statement
namespace TriangleProof

variables (a b c : ℝ)
variables (A B C : ℝ) -- Angles in radians

-- Condition for sides and angles in triangle
axiom side_angle_relation : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C

-- Part I: Proving angle C
theorem find_angle_C : C = Real.pi / 3 :=
  sorry

-- Additional given lengths and ratios for part II
variables (AD h : ℝ)
axiom a_value : a = 4 * Real.sqrt 3
axiom b_value : b = 2 * Real.sqrt 3
axiom vector_ratio : ∃ (BD DC : ℝ), BD = 2 * DC

-- Additional relations for defining AD and the distance 'h'
axiom AD_squared : AD^2 = (4 * (Real.sqrt 3))^2 + (4 * Real.sqrt 3 / 3)^2 - 2 * (4 * Real.sqrt 3) * (4 * Real.sqrt 3 / 3) * Real.cos (Real.pi / 3)
axiom area_triangle : ½ * AD * h = 2 * Real.sqrt 3

-- Part II: Proving distance from point C to line AD
theorem distance_C_to_AD : h = 6 * Real.sqrt 7 / 7 :=
  sorry
  
end TriangleProof

end find_angle_C_distance_C_to_AD_l608_608306


namespace laura_three_blue_pens_l608_608672

open Classical
open Probability

-- Definitions and conditions based on the problem
def num_blue : ℕ := 8
def num_red : ℕ := 7
def total_pens : ℕ := num_blue + num_red
def trials : ℕ := 7
def pick_blue_prob : ℝ := num_blue / total_pens
def pick_red_prob : ℝ := num_red / total_pens

-- The function to compute the binomial coefficient
noncomputable def binom_coeff : ℕ := Nat.choose trials 3

-- The probability calculation for picking exactly three blue pens
noncomputable def specific_arrangement_prob : ℝ :=
  (pick_blue_prob ^ 3) * (pick_red_prob ^ 4)

noncomputable def total_probability : ℝ :=
  binom_coeff * specific_arrangement_prob

-- Proof problem statement
theorem laura_three_blue_pens :
  total_probability = 43025920 / 170859375 := by
  sorry -- Proof goes here

end laura_three_blue_pens_l608_608672


namespace amy_skate_distance_l608_608804

theorem amy_skate_distance (t : ℝ)
  (hAB : 120 = 120) -- distance between A and B
  (hAmy_speed : 9 > 0) -- Amy's speed
  (hAngle : 45 = 45) -- Amy's angle with AB
  (hBob_speed : 10 > 0) -- Bob's speed
  (start_time : 0 = 0) : ∃ t : ℝ, 9 * t ≈ 105.2775 ∧ 
  (100 * (t ^ 2) = 81 * (t ^ 2) + 120^2 - 2 * 9 * t * 120 * (cos (45 * (pi / 180)))) := 
begin
  sorry
end

end amy_skate_distance_l608_608804


namespace water_current_speed_l608_608494

theorem water_current_speed :
  ∀ (speed_still_water : ℕ) (distance : ℕ) (time : ℕ), 
  speed_still_water = 8 ∧ 
  distance = 8 ∧ 
  time = 2 →
  (8 - (distance / time) = 4) :=
by
  assume speed_still_water distance time
  intro h
  cases h with h1 h2
  cases h2 with h3 h4
  rw [h1, h3, h4]
  sorry

end water_current_speed_l608_608494


namespace seats_usually_taken_l608_608776

def total_tables : Nat := 15
def seats_per_table : Nat := 10
def proportion_left_unseated : Rat := 1 / 10
def proportion_taken : Rat := 1 - proportion_left_unseated

theorem seats_usually_taken :
  proportion_taken * (total_tables * seats_per_table) = 135 := by
  sorry

end seats_usually_taken_l608_608776


namespace length_of_transverse_axis_l608_608960

-- Definitions
def hyperbola (a b : ℕ) (x y : ℂ) : Prop := x^2 / a^2 - y^2 / b^2 = 1
def parabola (x y : ℕ) : Prop := y^2 = 8 * x
def is_focus (x y : ℤ) : Prop := x = 2 ∧ y = 0
def collinear (P F Q : ℤ × ℤ) : Prop := (P, F, Q match ∀ (P_y, F_y, Q_y), P_y = F_y ∧ F_y = Q_y)

-- Theorem statement
theorem length_of_transverse_axis {a b : ℕ} {P Q : ℤ × ℤ}
  (hypM : ∀ (x y : ℂ), hyperbola a b x y)
  (hypN : ∀ (x y : ℕ), parabola x y)
  (focusF : is_focus 2 0)
  (collinear_PFQ : collinear P (2, 0) Q) :
  2 * a = 4 * (sqrt 2) - 4 := 
sorry

end length_of_transverse_axis_l608_608960


namespace determine_a_l608_608584

theorem determine_a 
(h : ∃x, x = -1 ∧ 2 * x ^ 2 + a * x - a ^ 2 = 0) : a = -2 ∨ a = 1 :=
by
  -- Proof omitted
  sorry

end determine_a_l608_608584


namespace log_inequality_solution_l608_608362

-- Let a and x be real numbers
variables (a x : ℝ)

-- Assume a > 0, a ≠ 1, and f(x) has a minimum value (implies a > 1)
theorem log_inequality_solution :
  a > 0 → a ≠ 1 → (∃ x_min, ∀ x, f(x) = log a (x^2 - 2x + 3) ∧ f(x_min) ≤ f(x)) → (log a (x-1) > 0) = (x > 2) :=
by
  intro ha0 ha1 hmin
  have ha_gt_1 : a > 1 := sorry  -- Derived from the fact f(x) has a minimum
  have hlog_eq : log a (x-1) > 0 = (x-1 > 1) := sorry  -- Property of logarithm for a > 1
  have hsol_eq : x-1 > 1 = x > 2 := sorry  -- Simple arithmetic rearrangement
  exact eq.trans hlog_eq hsol_eq

end log_inequality_solution_l608_608362


namespace antonella_toonies_l608_608157

theorem antonella_toonies (L T : ℕ) (h1 : L + T = 10) (h2 : L + 2 * T = 14) : T = 4 :=
by
  sorry

end antonella_toonies_l608_608157


namespace common_chords_concurrent_on_OH_l608_608368

variables (A B C O H N : Point)
variables (Γ ωA ωB ωC : Circle)
variables (tangent_to_Γ_at_B tangent_to_Γ_at_C : Line)
variables (perpendicular_to_AN_through_H : Line)

-- Define the properties of the given entities
def is_acute_triangle (A B C : Point) : Prop := sorry
def is_circumcenter (O A B C : Point) : Prop := sorry
def is_orthocenter (H A B C : Point) : Prop := sorry
def is_circumcircle (Γ A B C : Circle) : Prop := sorry
def is_midpoint (N O H : Point) : Prop := sorry

-- The definition of ωA, ωB, and ωC
def defined_ωA (ωA tangent_to_Γ_at_B tangent_to_Γ_at_C perpendicular_to_AN_through_H : Circle) : Prop := sorry
def defined_ωB (ωB tangent_to_Γ_at_C tangent_to_Γ_at_A perpendicular_to_AN_through_H : Circle) : Prop := sorry
def defined_ωC (ωC tangent_to_Γ_at_A tangent_to_Γ_at_B perpendicular_to_AN_through_H : Circle) : Prop := sorry

-- The theorem to prove
theorem common_chords_concurrent_on_OH 
  (h1 : is_acute_triangle A B C) 
  (h2 : is_circumcenter O A B C)
  (h3 : is_orthocenter H A B C)
  (h4 : is_circumcircle Γ A B C)
  (h5 : is_midpoint N O H)
  (h6 : defined_ωA ωA tangent_to_Γ_at_B tangent_to_Γ_at_C perpendicular_to_AN_through_H)
  (h7 : defined_ωB ωB tangent_to_Γ_at_C tangent_to_Γ_at_A perpendicular_to_AN_through_H)
  (h8 : defined_ωC ωC tangent_to_Γ_at_A tangent_to_Γ_at_B perpendicular_to_AN_through_H) :
  concurrent (common_chords ωA ωB ωC) OH :=
sorry

end common_chords_concurrent_on_OH_l608_608368


namespace find_y_l608_608303

theorem find_y (x y z : ℤ) (h₁ : x + y + z = 355) (h₂ : x - y = 200) (h₃ : x + z = 500) : y = -145 :=
by
  sorry

end find_y_l608_608303


namespace conditions_on_m_n_l608_608259

variables {m n : ℝ}

def mean (l : list ℝ) : ℝ := (l.sum) / (l.length)

def x_values : list ℝ := [1, 2, 3, 4, 5]

def y_values (m n : ℝ) : list ℝ := [4, m, 9, n, 11]

axiom regression_line_through_3_7 : ∀ (m n : ℝ), mean (y_values m n) = 7

theorem conditions_on_m_n : m + n = 11 :=
by
  sorry

end conditions_on_m_n_l608_608259


namespace convex_polyhedron_edge_translation_l608_608202

/-- A definition representing a convex polyhedron. This can be extended with properties as needed. -/
structure ConvexPolyhedron :=
(vertices : Finset Point)
(edges : Finset (Point × Point))
(is_convex : Convex vertices) -- Assume that Convex is a predicate that checks convexity

/-- A function to translate an edge by a given vector -/
def translate_edge (e : Point × Point) (v : Vector) : Point × Point :=
(e.1 + v, e.2 + v)

/-- A function to translate all edges of a convex polyhedron by a given set of vectors -/
def translate_edges (P : ConvexPolyhedron) (v : Finset Vector) : ConvexPolyhedron :=
{ vertices := P.vertices, -- Vertices remain the same for simplicity
  edges := P.edges.map (λ (e, vec), translate_edge e vec) P.edges v,
  is_convex := sorry } -- Will require a proof that the new set of edges forms a convex polyhedron

/-- The theorem stating that translating the edges of a convex polyhedron does not necessarily result -/
/-- in a polyhedron congruent to the original polyhedron -/
theorem convex_polyhedron_edge_translation {P : ConvexPolyhedron} (v : Finset Vector) :
  ¬ (translate_edges P v).is_congruent_to P :=
sorry

end convex_polyhedron_edge_translation_l608_608202


namespace circumcenter_lies_on_AK_l608_608832

noncomputable def is_circumcenter (O : Point) (A B C : Point) : Prop :=
  dist O A = dist O B ∧ dist O B = dist O C

noncomputable def lies_on_line (P Q R : Point) : Prop :=
  ∃ (k : ℝ), Q = P + k • (R - P)

theorem circumcenter_lies_on_AK
  (A B C L H K O : Point)
  (h_triangle : ∀ (X Y Z : Point), X ≠ Y → X ≠ Z → Y ≠ Z → is_triangle X Y Z)
  (h_AL : is_angle_bisector A L B C)
  (h_H : foot B L H)
  (h_K : foot_on_circumcircle B L K (set_circumcircle A B L))
  (h_circ_A : O = is_circumcenter O A B C) :
  lies_on_line A K O :=
sorry

end circumcenter_lies_on_AK_l608_608832


namespace hex_game_tournament_l608_608744

theorem hex_game_tournament (n : ℕ) (h : n ≥ 2) 
  (plays_against_all : ∀ (i j : ℕ), i ≠ j → (i < n ∧ j < n)) 
  (no_draws : ∀ (i j : ℕ), i ≠ j → (has_beaten i j ∨ has_beaten j i)) : 
  ∃ player : ℕ, (∀ other_player : ℕ, other_player ≠ player ∧ other_player < n → list_of_beaten player other_player) :=
sorry

variables {n i j : ℕ}

-- Define predicate has_beaten to represent the relationship that a player i has beaten player j
-- This predicate should comply with the conditions given
axiom has_beaten : ℕ → ℕ → Prop

-- Define list_of_beaten to represent the list of players beaten by a player
axiom list_of_beaten : ℕ → ℕ → Prop

end hex_game_tournament_l608_608744


namespace sqrt_identity_l608_608568

theorem sqrt_identity (x : ℝ) (hx : x + x⁻¹ = 5) : x^(1/2) + x^(-1/2) = Real.sqrt 7 :=
sorry

end sqrt_identity_l608_608568


namespace not_all_triangles_congruent_l608_608141

-- Definitions related to the problem setup
def interior_angle_sum (T : Triangle) : Prop :=
  ∑ α in T.angles, α = 180

def vertex_angle_sum (angles : Finset ℝ) : Prop :=
  ∑ α in angles, α = 360

-- Lean 4 statement for the given proof problem
theorem not_all_triangles_congruent (T : Triangle) (small_triangles : Finset Triangle)
  (h1 : interior_angle_sum T)
  (h2 : (∀ S ∈ small_triangles, interior_angle_sum S))
  (h3 : ∃ A, (vertex_angle_sum (Finset.image (λ x, x.angle_at A) small_triangles)) = 360)
  : ¬ (∀ S1 S2 ∈ small_triangles, S1 ≃ S2) :=
sorry

end not_all_triangles_congruent_l608_608141


namespace probability_of_sqrt5_distance_l608_608327

namespace ProbabilityProblem

def point := (ℝ × ℝ)
def K : set point := { p : point | (p.1 ∈ {-1, 0, 1}) ∧ (p.2 ∈ {-1, 0, 1}) }

-- Define a function that calculates the Euclidean distance between two points
def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Prove that the probability that among any three randomly chosen points from K, there exist
-- two points with a distance of sqrt(5) is 4/7.
theorem probability_of_sqrt5_distance :
  let points := { p : point | ∃ x y, p = (x, y) ∧ x ∈ {-1, 0, 1} ∧ y ∈ {-1, 0, 1} },
      combinations := finset.powerset_len 3 (finset.univ : finset points) in
  (∃ p1 p2 p3 ∈ points, finset.card { p | distance p.1 p.2 = 5 } = 4 / 7) := sorry

end ProbabilityProblem

end probability_of_sqrt5_distance_l608_608327


namespace find_n_l608_608363

def num_of_trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n / 5) + num_of_trailing_zeros (n / 5)

theorem find_n (n : ℕ) (k : ℕ) (h1 : n > 3) (h2 : k = num_of_trailing_zeros n) (h3 : 2*k + 1 = num_of_trailing_zeros (2*n)) (h4 : k > 0) : n = 6 :=
by
  sorry

end find_n_l608_608363


namespace max_distance_of_chord_l608_608300

noncomputable def maximum_distance_between_intersection_points 
  (m : ℝ) 
  (h_line : ∀ (x y : ℝ), y = x + m) 
  (h_ellipse : ∀ (x y : ℝ), x^2 + 4 * y^2 = 4) : ℝ :=
  ∃ m : ℝ, abs(d(A, B)) = (4 / 5) * real.sqrt 10

open_locale real_inner_product_space

theorem max_distance_of_chord :
  ∀ (m : ℝ), 
  ∀ (x y : ℝ), 
    (y = x + m) ∧ (x^2 + 4 * y^2 = 4) 
    -> ∃ (d : ℝ), d = (4 / 5) * real.sqrt 10 := 
by
  sorry

end max_distance_of_chord_l608_608300


namespace sum_of_series_l608_608461

theorem sum_of_series (n : ℕ) (hn : n = 9) :
  (∑ k in finset.range n, 1 / ((k + 1) * (k + 2))) = 9 / 10 :=
by sorry

end sum_of_series_l608_608461


namespace find_a33_l608_608926

theorem find_a33 : 
  ∀ (a : ℕ → ℤ), a 1 = 3 → a 2 = 6 → (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) → a 33 = 3 :=
by
  intros a h1 h2 h_rec
  sorry

end find_a33_l608_608926


namespace Keith_picked_zero_apples_l608_608702

variable (M J T K_A : ℕ)

theorem Keith_picked_zero_apples (hM : M = 14) (hJ : J = 41) (hT : T = 55) (hTotalOranges : M + J = T) : K_A = 0 :=
by
  sorry

end Keith_picked_zero_apples_l608_608702


namespace part_a_impossibility_part_b_possibility_l608_608045

-- Defining the problem for part (a)
theorem part_a_impossibility :
  ¬∃ (f : Fin 12 → ℕ) (sum_vertex : Fin 8 → ℕ),
    (∀ v : Fin 8, sum_vertex v = 19.5) ∧
    (∀ e : Fin 12, ∃ (v1 v2 : Fin 8), sum_vertex v1 + sum_vertex v2 = 78) := by
  sorry

-- Defining the problem for part (b)
theorem part_b_possibility :
  ∃ (f : Fin 12 → ℤ) (sum_vertex : Fin 8 → ℤ),
    (∀ v : Fin 8, sum_vertex v = 0) ∧
    (∀ e : Fin 12, ∃ (v1 v2 : Fin 8), v1 ≠ v2) ∧
    List.sum (List.ofFn (λ i => f ⟨i, sorry⟩)) = 0 := by
  sorry

end part_a_impossibility_part_b_possibility_l608_608045


namespace circumcenter_lies_on_ak_l608_608836

noncomputable def triangle_circumcenter_lies_on_ak
  {α β γ : ℝ}
  (A B C L H K O : Type*)
  [triangle A B C]
  [angle_bisector A L]
  [height_from B H]
  [circumcircle_of_triangle A (B ∧ L) K]
  [circumcenter_of_triangle A B C O]
  : Prop :=
  lies_on_line O (line_through A K)

-- We'll add the assumptions as hypotheses to the lemma
theorem circumcenter_lies_on_ak 
  {α β γ : ℝ} {A B C L H K O : Type*}
  [triangle A B C]
  [angle_bisector A L]
  [height_from B H]
  [circumcircle_of_triangle A (B ∧ L) K]
  [circumcenter_of_triangle A B C O]
  : lies_on_line O (line_through A K) :=
sorry

end circumcenter_lies_on_ak_l608_608836


namespace lateral_side_of_isosceles_triangle_l608_608855

-- Define the geometric setup
variables {R k : ℝ} -- R is the radius of the circle, k is the given ratio

-- The theorem statement
theorem lateral_side_of_isosceles_triangle (R k : ℝ) :
  ∃ AB : ℝ, AB = R * real.sqrt ((4 * k + 3) / (k + 1)) :=
sorry -- proof to be filled in later

end lateral_side_of_isosceles_triangle_l608_608855


namespace average_last_4_matches_l608_608451

theorem average_last_4_matches (avg_10_matches avg_6_matches : ℝ) (matches_10 matches_6 matches_4 : ℕ) :
  avg_10_matches = 38.9 →
  avg_6_matches = 41 →
  matches_10 = 10 →
  matches_6 = 6 →
  matches_4 = 4 →
  (avg_10_matches * matches_10 - avg_6_matches * matches_6) / matches_4 = 35.75 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_last_4_matches_l608_608451


namespace Josh_marbles_count_l608_608669

-- Definitions of the given conditions
def initial_marbles : ℕ := 16
def lost_marbles : ℕ := 7

-- The statement we aim to prove
theorem Josh_marbles_count : (initial_marbles - lost_marbles) = 9 :=
by
  -- Skipping the proof with sorry
  sorry

end Josh_marbles_count_l608_608669


namespace find_events_l608_608633

inductive Color
| red : Color
| white : Color

def bag : list Color := [Color.white, Color.white, Color.red, Color.red, Color.red]

noncomputable def draw_two_balls (bag: list Color) : list (Color × Color) :=
  bag.bind (λ b1, filter ((≠ b1) ∘ id) bag |>.map (λ b2, (b1, b2)))

def mutually_exclusive (e1 e2 : Color × Color) : Prop :=
  e1.1 != e2.1 ∨ e1.2 != e2.2

def contradictory (e1 e2 : Color × Color) : Prop :=
  e1.1 == e2.1 ∧ e1.2 == e2.2

def event1 : Color × Color := (Color.red, Color.white)
def event2 : Color × Color := (Color.white, Color.white)

theorem find_events :
  (mutually_exclusive event1 event2) ∧ ¬ (contradictory event1 event2) :=
by
  sorry

end find_events_l608_608633


namespace pregnant_fish_in_each_tank_l608_608701

/-- Mark has 3 tanks for pregnant fish. Each tank has a certain number of pregnant fish and each fish
gives birth to 20 young. Mark has 240 young fish at the end. Prove that there are 4 pregnant fish in
each tank. -/
theorem pregnant_fish_in_each_tank (x : ℕ) (h1 : 3 * 20 * x = 240) : x = 4 := by
  sorry

end pregnant_fish_in_each_tank_l608_608701


namespace difference_of_squares_example_l608_608523

theorem difference_of_squares_example :
  (262^2 - 258^2 = 2080) :=
by {
  sorry -- placeholder for the actual proof
}

end difference_of_squares_example_l608_608523


namespace angle_between_vectors_l608_608018

open Real

variables (a b : ℝ^3) -- Assuming vectors in 3-dimensional space

-- Given conditions
def condition1 : a ≠ 0 ∧ b ≠ 0 := ⟨sorry⟩
def condition2 : (a - 2 • b) ⬝ a = 0 := sorry
def condition3 : (b - 2 • a) ⬝ b = 0 := sorry

-- Prove the angle between a and b is π/3
theorem angle_between_vectors : angle a b = π / 3 :=
begin
  sorry
end

end angle_between_vectors_l608_608018


namespace inscribed_circle_distance_l608_608456

-- description of the geometry problem
theorem inscribed_circle_distance (r : ℝ) (AB : ℝ):
  r = 4 →
  AB = 4 →
  ∃ d : ℝ, d = 6.4 :=
by
  intros hr hab
  -- skipping proof steps
  let a := 2*r
  let PQ := 2 * r * (Real.sqrt 3 / 2)
  use PQ
  sorry

end inscribed_circle_distance_l608_608456


namespace antonov_packs_remaining_l608_608507

theorem antonov_packs_remaining (total_candies : ℕ) (pack_size : ℕ) (packs_given : ℕ) (candies_remaining : ℕ) (packs_remaining : ℕ) :
  total_candies = 60 →
  pack_size = 20 →
  packs_given = 1 →
  candies_remaining = total_candies - pack_size * packs_given →
  packs_remaining = candies_remaining / pack_size →
  packs_remaining = 2 := by
  sorry

end antonov_packs_remaining_l608_608507


namespace three_digit_reverse_sum_to_1777_l608_608068

theorem three_digit_reverse_sum_to_1777 :
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ 101 * (a + c) + 20 * b = 1777 :=
by
  sorry

end three_digit_reverse_sum_to_1777_l608_608068


namespace geometric_sequence_sum_t_value_l608_608266

theorem geometric_sequence_sum_t_value 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (t : ℝ)
  (h1 : ∀ n : ℕ, S_n n = 3^((n:ℝ)-1) + t)
  (h2 : a_n 1 = 3^0 + t)
  (geometric : ∀ n : ℕ, n ≥ 2 → a_n n = 2 * 3^(n-2)) :
  t = -1/3 :=
by
  sorry

end geometric_sequence_sum_t_value_l608_608266


namespace second_player_wins_optimal_play_l608_608645

-- Define the game setting
structure HexagonalGridGame (n : ℕ) :=
  (initial_position : (ℕ × ℕ)) -- the initial position of the chip
  (visited_nodes : finset (ℕ × ℕ)) -- a set of visited nodes
  (move : (ℕ × ℕ) → (ℕ × ℕ) → Prop) -- a relation representing valid moves
  (move_property : ∀ (current next : (ℕ × ℕ)), move current next → next ∉ visited_nodes)
  (neighbor_nodes : (ℕ × ℕ) → finset (ℕ × ℕ)) -- a function returning neighboring nodes
  (valid_move : ∀ (pos : (ℕ × ℕ)), ∀ (neighbour : (ℕ × ℕ)), neighbour ∈ neighbor_nodes pos → move pos neighbour)

-- Define the conditions under which the game is played
noncomputable def hexagonal_grid_conditions (n : ℕ) : Prop := sorry

-- Prove who has the winning strategy
theorem second_player_wins_optimal_play (n : ℕ) (game : HexagonalGridGame n) 
    (cond : hexagonal_grid_conditions n) : 
    ∃ second_player_strategy : (ℕ × ℕ) → (ℕ × ℕ), 
    (∀ first_player_strategy : (ℕ × ℕ) → (ℕ × ℕ), 
    alternative_play first_player_strategy second_player_strategy game.initial_position → 
    second_player_wins first_player_strategy second_player_strategy):
    true :=
sorry

end second_player_wins_optimal_play_l608_608645


namespace shift_f_to_g_l608_608746

-- Define the function f
def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 6))

-- Define the function g
def g (x : ℝ) : ℝ := Real.cos (2 * x + (Real.pi / 6))

-- State the theorem about the shift
theorem shift_f_to_g :
  ∀ x : ℝ, g(x) = f(x + Real.pi / 4) :=
by
  sorry

end shift_f_to_g_l608_608746


namespace union_comm_union_assoc_inter_distrib_union_l608_608353

variables {α : Type*} (A B C : Set α)

theorem union_comm : A ∪ B = B ∪ A := sorry

theorem union_assoc : A ∪ (B ∪ C) = (A ∪ B) ∪ C := sorry

theorem inter_distrib_union : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) := sorry

end union_comm_union_assoc_inter_distrib_union_l608_608353


namespace solution_l608_608358

def isPerfectSquare (x : ℤ) : Prop := ∃ m : ℤ, m^2 = x

def valid_n (n : ℕ) : Prop := isPerfectSquare (n^2 + 12 * n - 2005)

def sum_valid_n : ℕ := (List.range (1016 + 1)).filter valid_n |>.sum

theorem solution : sum_valid_n % 1000 = 463 := 
by
  sorry

end solution_l608_608358


namespace compound_interest_rate_l608_608766

noncomputable def RS_1750 := 1750
def rate_simple_interest := 8
def time_simple_interest := 3
def simple_interest := RS_1750 * rate_simple_interest * time_simple_interest / 100

noncomputable def RS_4000 := 4000
def time_compound_interest := 2
def CI := 2 * simple_interest -- i.e., 840

theorem compound_interest_rate :
  ∃ R, CI = RS_4000 * ((1 + R/100)^time_compound_interest - 1) ∧ R = 10 :=
by {
  sorry
}

end compound_interest_rate_l608_608766


namespace diagonal_length_l608_608128

-- Given that x and y are the lengths of the sides of a rectangle, 
-- and the sum of the squares of x and y is 400, prove the length of the diagonal AC is 20 cm.
theorem diagonal_length (x y : ℝ) (h : x^2 + y^2 = 400) : Real.sqrt (x^2 + y^2) = 20 := by
  rw h
  simp
  exact Real.sqrt_eq_rfl.symm

end diagonal_length_l608_608128


namespace elena_postcard_cost_l608_608049

noncomputable def cost_eastern_hemisphere_pre_1970s : ℕ :=
  (5 + 8) * 7 + (3 + 6) * 7

theorem elena_postcard_cost : cost_eastern_hemisphere_pre_1970s = 154 := by
  unfold cost_eastern_hemisphere_pre_1970s
  norm_num
  sorry

end elena_postcard_cost_l608_608049


namespace assignment_probability_l608_608159

-- Define the set from which numbers are drawn
def set_numbers : Finset ℕ := Finset.range' 1 16

-- Define the distinct assignment of numbers to Aria, Bella, and Clara
def assignment (a b c : ℕ) : Prop := 
  a ∈ set_numbers ∧ 
  b ∈ set_numbers ∧ 
  c ∈ set_numbers ∧ 
  a ≠ b ∧ 
  b ≠ c ∧ 
  a ≠ c

-- Define the condition that Aria's number is a multiple of Bella's 
-- and Bella's number is a multiple of Clara's
def valid_assignment (a b c : ℕ) : Prop := 
  assignment a b c ∧ 
  a % b = 0 ∧ 
  b % c = 0

-- Total number of valid assignments
noncomputable def total_valid_assignments : ℕ :=
  Finset.card {
    (a, b, c) ∈ (set_numbers.product (set_numbers.product set_numbers)) |
    valid_assignment a b c
  }

-- Total possible assignments
def total_possible_assignments : ℕ := 15 * 14 * 13

-- Probability of a valid assignment
noncomputable def probability_valid_assignment : ℚ :=
  total_valid_assignments / total_possible_assignments

theorem assignment_probability :
  probability_valid_assignment = 8 / 546 :=
by sorry

end assignment_probability_l608_608159


namespace speeds_of_bicycle_and_car_l608_608397

def distance : ℝ := 10
def delay_time : ℝ := 1 / 3

theorem speeds_of_bicycle_and_car (x : ℝ) (bicycle_speed car_speed : ℝ) :
  bicycle_speed = 15 ∧ car_speed = 30 :=
by
  assume h1 : x = 15,
  assume h2 : bicycle_speed = x,
  assume h3 : car_speed = 2 * x,
  have h4 : 10 / x = 10 / (2 * x) + delay_time, by sorry,
  have h5 : x = 15, by sorry,
  have h6 : bicycle_speed = 15, by sorry,
  have h7 : car_speed = 30, by sorry,
  exact ⟨h6, h7⟩

end speeds_of_bicycle_and_car_l608_608397


namespace earthquake_magnitude_amplitude_ratio_l608_608406

-- Define the logarithm function; note that Lean already has the logarithm defined
open Real

-- Definitions based on the conditions:
def RichterMagnitude (A : ℝ) (A_0 : ℝ) : ℝ := log10 A - log10 A_0

-- Problem (1):
theorem earthquake_magnitude (A A_0 : ℝ) : A = 1000 → A_0 = 0.001 → RichterMagnitude A A_0 = 6 :=
by
  intros hA hA0
  rw [hA, hA0]
  sorry

-- Problem (2):
theorem amplitude_ratio (A_0 : ℝ) (M9 M5 : ℝ) : A_0 = 0.001 → M9 = 9 → M5 = 5 →
  let A_9 := 10^(M9 + log10 A_0) in
  let A_5 := 10^(M5 + log10 A_0) in
  A_9 / A_5 = 10000 :=
by
  intros hA0 hM9 hM5
  simp_rw [hA0, hM9, hM5]
  let A_9 := 10^(12 : ℝ)
  let A_5 := 10^(8 : ℝ)
  sorry

end earthquake_magnitude_amplitude_ratio_l608_608406


namespace expression_value_l608_608618

theorem expression_value {a b c d m : ℝ} (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 1) : 
  (a + b) * c * d - 2014 * m = -2014 ∨ (a + b) * c * d - 2014 * m = 2014 := 
by
  sorry

end expression_value_l608_608618


namespace probability_at_least_two_same_row_col_l608_608933

-- Define the total number of ways to select 3 numbers from 9
def total_ways (n k : ℕ) : ℕ := Nat.choose n k

-- Define the number of ways to pick 3 numbers such that no two are in the same row/column
def ways_no_two_same_row_col : ℕ := 6

-- Define the probability calculation
def calculate_probability (total ways_excluded : ℕ) : ℚ :=
  (total - ways_excluded) / total

-- The problem to prove
theorem probability_at_least_two_same_row_col :
  calculate_probability (total_ways 9 3) ways_no_two_same_row_col = 13 / 14 :=
by
  sorry

end probability_at_least_two_same_row_col_l608_608933


namespace num_paths_in_grid_l608_608886

theorem num_paths_in_grid (w h : ℕ) (start_right : ℕ) (total_moves : ℕ) :
    w = 6 ∧ h = 5 ∧ start_right = 1 ∧ total_moves = 11 → 
    (nat.choose (total_moves - start_right) (h)) = 252 :=
by
  intros
  cases a with w_eq_h_eq a_left
  cases a_left with a_right total
  cases w_eq_h_eq with w_eq h_eq
  rw [w_eq, h_eq, a_right, total]
  exact nat.choose (11 - 1) 5 = 252
  sorry

end num_paths_in_grid_l608_608886


namespace make_sequences_identical_l608_608677

structure LatticePoint :=
(x : ℤ) (y : ℤ)

def slope (O A : LatticePoint) : ℚ := (A.y - O.y) / (A.x - O.x)

def area (O A B : LatticePoint) : ℚ := abs ((A.x * B.y - A.y * B.x) / 2)

def strictly_increasing (slopes : List ℚ) : Prop :=
∀ i j, i < j → slopes.nth i < slopes.nth j

def is_interesting_sequence (O : LatticePoint) (seq : List LatticePoint) : Prop :=
∀ i < seq.length - 1, area O (seq.nth i).get (seq.nth (i + 1)).get = 1 / 2 ∧ strictly_increasing (List.map (slope O) seq)

def can_extend (O : LatticePoint) (seq : List LatticePoint) (i : ℕ) : LatticePoint :=
{ x := seq.nth i + seq.nth (i + 1), y := seq.nth i + seq.nth (i + 1) }

def extension (seq : List LatticePoint) (i : ℕ) : List LatticePoint :=
List.insertN seq (i + 1) (can_extend (seq.nth i).get (seq.nth (i + 1)).get)

theorem make_sequences_identical (O : LatticePoint)
  (C D : List LatticePoint) (n m : ℕ) (hC_int : is_interesting_sequence O C) 
  (hD_int : is_interesting_sequence O D) (hC0D0 : C.head = D.head) (hCnDm : C.last = D.last) :
  ∃ (seqC seqD : List LatticePoint), 
    is_interesting_sequence O seqC ∧ is_interesting_sequence O seqD ∧ 
    C ⊆ seqC ∧ D ⊆ seqD ∧ seqC = seqD := by
  sorry

end make_sequences_identical_l608_608677


namespace probability_x_plus_y_lt_4_l608_608109

theorem probability_x_plus_y_lt_4 :
  let square_vertices := {(0,0), (0,3), (3,3), (3,0)} in
  let point_in_square (x y : ℝ) := 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3 in
  let probability (A : ℝ) (B : ℝ) := A / B in
  ∃ (P : ℝ), P = probability 7 9 ∧
             P = (measure (λ (x y : ℝ), point_in_square x y ∧ x + y < 4)) / (measure (λ (x y : ℝ), point_in_square x y)) :=
sorry

end probability_x_plus_y_lt_4_l608_608109


namespace log_eq_log_l608_608288

theorem log_eq_log (a b : ℝ) (h1 : a = Real.logBase 4 144) (h2 : b = Real.logBase 2 12) : a = b := by
  sorry

end log_eq_log_l608_608288


namespace eccentricity_of_ellipse_l608_608875

-- Define the problem conditions
open Real

noncomputable def ellipse_properties (a b : ℝ) (A B : ℝ × ℝ) (F1 F2 : ℝ × ℝ) :=
  (a > b ∧ b > 0) ∧
  (abs (prod.snd A - prod.snd B) = 4) ∧
  ((1/2) * (abs (a * it.triangle_area A F1 B) = π)

-- Define the main problem statement
theorem eccentricity_of_ellipse (a b : ℝ) (A B : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (h1 : ellipse_properties a b A B F1 F2) :
  deterministic_eccentricity (a > b ∧ b > 0) ∧
  (a = π) ∧
  (a = 2 * sqrt (a^2 - b^2))  →
  abs ((sqrt (a^2 - b^2)) / a) = 1/2 :=
sorry

end eccentricity_of_ellipse_l608_608875


namespace parallel_planes_theorem_l608_608319

-- Definitions related to the problem
def parallel_lines_in_plane (A B C : Type) [linear_space A] (L1 L2 L3 : line A) : Prop :=
  (L1 ∥ L3 ∧ L2 ∥ L3) → (L1 ∥ L2)

def parallel_planes_in_space (A B : Type) [affine_space A B] (P1 P2 P3 : plane A) : Prop :=
  (P1 ∥ P3 ∧ P2 ∥ P3) → (P1 ∥ P2)

-- Given condition: In a plane, two lines parallel to the same line are also parallel.
axiom parallel_lines_in_plane_axiom (A B C : Type) [linear_space A] (L1 L2 L3 : line A) :
  parallel_lines_in_plane A B C L1 L2 L3

-- To prove: In space, two planes parallel to the same plane are also parallel.
theorem parallel_planes_theorem (A B : Type) [affine_space A B] (P1 P2 P3 : plane A) :
  parallel_planes_in_space A B P1 P2 P3 := sorry

end parallel_planes_theorem_l608_608319


namespace complement_union_l608_608978

-- Definition of the universal set U
def U : Set ℤ := {x | x^2 - 5 * x - 6 ≤ 0}

-- Definition of set A
def A : Set ℤ := {x | x * (2 - x) ≥ 0}

-- Definition of set B
def B : Set ℤ := {1, 2, 3}

-- The proof statement
theorem complement_union (h : U = {x | x^2 - 5 * x - 6 ≤ 0} ∧ 
                           A = {x | x * (2 - x) ≥ 0} ∧ 
                           B = {1, 2, 3}) : 
  U \ (A ∪ B) = {-1, 4, 5, 6} :=
by {
  sorry
}

end complement_union_l608_608978


namespace area_of_triangle_is_correct_l608_608904

def Point3D := (ℝ × ℝ × ℝ)

def distance (p1 p2 : Point3D) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

def triangle_area (A B C : Point3D) : ℝ :=
  let AB_len := distance A B
  let AC_len := distance A C
  let BC_len := distance B C
  0.5 * AC_len * BC_len

theorem area_of_triangle_is_correct :
  let A : Point3D := (0, 3, 5)
  let B : Point3D := (-2, 3, 1)
  let C : Point3D := (1, 6, 1)
  triangle_area A B C = 3 * real.sqrt 13 :=
by
  sorry

end area_of_triangle_is_correct_l608_608904


namespace credits_per_class_l608_608376

-- Define the conditions as hypotheses
variables (semesters : ℕ) (total_credits : ℕ) (classes_per_semester : ℕ)
-- Define the given values for the conditions
variables (h_semesters : semesters = 8) (h_total_credits : total_credits = 120)
variables (h_classes_per_semester : classes_per_semester = 5)

-- Define the proof statement
theorem credits_per_class :
  let total_classes := classes_per_semester * semesters in
  let credits_per_class := total_credits / total_classes in
  credits_per_class = 3 :=
by
  -- insert proof here
  sorry

end credits_per_class_l608_608376


namespace half_abs_diff_squares_l608_608022
-- Import the necessary library

-- Define the given constants and state the theorem
theorem half_abs_diff_squares (a b : ℤ) (ha : a = 15) (hb : b = 12) : 
  (1 / 2) * | (a^2 - b^2) | = 40.5 := 
by
  sorry

end half_abs_diff_squares_l608_608022


namespace find_a_b_l608_608929

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 + Complex.I) 
  (h : (z^2 + a*z + b) / (z^2 - z + 1) = 1 - Complex.I) : a = -1 ∧ b = 2 :=
by
  sorry

end find_a_b_l608_608929


namespace empirical_regression_line_passes_through_8_25_a_hat_value_residual_when_x_is_5_removing_8_25_does_not_change_r_l608_608137

-- Define the given data
def x_data : List ℕ := [5, 6, 8, 9, 12]
def y_data : List ℕ := [17, 20, 25, 28, 35]

-- Define the empirical regression equation coefficients
def slope : ℝ := 2.6
def a_hat : ℝ := 25 - slope * 8 -- calculate a_hat directly using given condition

-- Define the regression line
def regression_line (x : ℝ) : ℝ := slope * x + a_hat

-- Prove the conditions of the problem
theorem empirical_regression_line_passes_through_8_25 :
  regression_line 8 = 25 :=
by
  -- Skipping the proof
  sorry

theorem a_hat_value :
  a_hat = 4.2 :=
by
  -- Skipping the proof
  sorry

theorem residual_when_x_is_5 :
  (↑(y_data.head!) - regression_line 5) = -0.2 :=
by
  -- Skipping the proof
  sorry

-- We need additional functions and theorem to discuss correlation coefficient
-- Hence adding placeholders

-- Placeholder for correlation coefficient calculation
def correlation_coefficient (x_data : List ℕ) (y_data : List ℕ) : ℝ := sorry

theorem removing_8_25_does_not_change_r :
  correlation_coefficient x_data y_data = correlation_coefficient (List.erase x_data 8) (List.erase y_data 25) :=
by
  -- Skipping the proof
  sorry

end empirical_regression_line_passes_through_8_25_a_hat_value_residual_when_x_is_5_removing_8_25_does_not_change_r_l608_608137


namespace log_sum_l608_608463

open Real

theorem log_sum : log 2 + log 5 = 1 :=
sorry

end log_sum_l608_608463


namespace angle_PCQ_45_degrees_l608_608459

-- Given definitions
def square (A B C D : Point) : Prop := 
  dist A B = 1 ∧ dist B C = 1 ∧ dist C D = 1 ∧ dist D A = 1 ∧ 
  dist A C = dist B D ∧ (all right angles between consecutive points)

def points_on_sides (P Q : Point) (A B C D : Point) : Prop :=
  (P ∈ line_segment A B) ∧ (Q ∈ line_segment A D)

def perimeter_triangle_APQ (A P Q : Point) : Prop :=
  dist A P + dist P Q + dist Q A = 2

-- Big goal theorem that glue them together.
theorem angle_PCQ_45_degrees
  (A B C D P Q : Point)
  (h_square : square A B C D)
  (h_P_on_AB : P ∈ line_segment A B)
  (h_Q_on_AD : Q ∈ line_segment A D)
  (h_perm_APQ : perimeter_triangle_APQ A P Q) :
  ∠ P C Q = 45 := 
sorry

end angle_PCQ_45_degrees_l608_608459


namespace weight_of_dried_grapes_l608_608823

/-- The weight of dried grapes available from 20 kg of fresh grapes given the water content in fresh and dried grapes. -/
theorem weight_of_dried_grapes (W_fresh W_dried : ℝ) (fresh_weight : ℝ) (weight_dried : ℝ) :
  W_fresh = 0.9 → 
  W_dried = 0.2 → 
  fresh_weight = 20 →
  weight_dried = (0.1 * fresh_weight) / (1 - W_dried) → 
  weight_dried = 2.5 :=
by sorry

end weight_of_dried_grapes_l608_608823


namespace odd_function_f_a_zero_l608_608263

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + (a + 1) * Real.cos x + x

theorem odd_function_f_a_zero (a : ℝ) (h_odd : ∀ x : ℝ, f a (-x) = -f a x) : f a a = 0 := 
sorry

end odd_function_f_a_zero_l608_608263


namespace true_condition4_l608_608866

-- Definition and conditions
def Condition1 : Prop := ∀ (l : Line) (p : Plane), proj l p = l
def Condition2 : Prop := ∀ (f : Figure) (p : Plane), (proj f p = l) → (f = l) 
def Condition3 : Prop := ∀ (l1 l2 : Line) (p : Plane), (angle l1 p = angle l2 p) → (parallel l1 l2)
def Condition4 : Prop := ∀ (l1 l2 : Line) (p : Plane), (parallel l1 l2) → (angle l1 p = angle l2 p)

-- Proof statement
theorem true_condition4 (h1 : Condition1) (h2 : Condition2) (h3 : Condition3) :
  Condition4 :=
sorry

end true_condition4_l608_608866


namespace find_a6_in_geometric_sequence_l608_608317

axiom geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n, a (n + 1) = a n * r

theorem find_a6_in_geometric_sequence (a : ℕ → ℝ) (r : ℝ)
  (h1 : geometric_sequence a r)
  (h2 : a 2 = 2)
  (h3 : a 4 = 4) :
  a 6 = 8 := 
sorry

end find_a6_in_geometric_sequence_l608_608317


namespace theo_possible_codes_l608_608773

theorem theo_possible_codes : 
  let odd_numbers := {n | 1 ≤ n ∧ n ≤ 30 ∧ n % 2 = 1},
      even_numbers := {n | 1 ≤ n ∧ n ≤ 30 ∧ n % 2 = 0},
      multiples_of_5 := {n | 1 ≤ n ∧ n ≤ 30 ∧ n % 5 = 0} in
  (odd_numbers.card = 15) ∧ (even_numbers.card = 15) ∧ (multiples_of_5.card = 6) ∧ 
  (odd_numbers.card * even_numbers.card * multiples_of_5.card = 1350) :=
by
  sorry

end theo_possible_codes_l608_608773


namespace find_a_b_extreme_values_l608_608598

-- Definitions and conditions
def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x
def f_prime (x : ℝ) (a b : ℝ) : ℝ := 3 * a * x^2 + b

-- Part 1: Prove a = -1 and b = 3 given the conditions
theorem find_a_b (a b : ℝ) (h1 : f 1 a b = 2) (h2 : f_prime 1 a b = 0) : 
  a = -1 ∧ b = 3 := by
  sorry

-- Part 2: Prove the extreme values on the interval [-2, 3]
theorem extreme_values (a b : ℝ) (ha : a = -1) (hb : b = 3) :
  let f := f (x : ℝ) -1 3
  let f_prime := f_prime (x : ℝ) -1 3
  ∀ x, x ∈ set.Icc (-2 : ℝ) 3 → 
  (x = -2 ∨ x = 3 ∨ f_prime x = 0) → 
  (f x = -18) ∨ (f x = 2) := by
  sorry

end find_a_b_extreme_values_l608_608598


namespace probability_x_plus_y_less_than_4_l608_608102

open Set Real

/-- Define the square with vertices (0,0), (0,3), (3,3), and (3,0) -/
def square := set.Icc (0 : ℝ) 3 ×ˢ set.Icc (0 : ℝ) 3

/-- Probability that a randomly chosen point (x, y) in the square satisfies x + y < 4 -/
theorem probability_x_plus_y_less_than_4 : 
  let area_square := 9 in
  let area_excluded_triangle := 2 in
  let relevant_area := area_square - area_excluded_triangle in
  (relevant_area / area_square : ℝ) = 7 / 9 :=
by
{ sorry }

end probability_x_plus_y_less_than_4_l608_608102


namespace base_5_minus_base_8_in_base_10_l608_608168

def base_5 := 52143
def base_8 := 4310

theorem base_5_minus_base_8_in_base_10 :
  (5 * 5^4 + 2 * 5^3 + 1 * 5^2 + 4 * 5^1 + 3 * 5^0) -
  (4 * 8^3 + 3 * 8^2 + 1 * 8^1 + 0 * 8^0)
  = 1175 := by
  sorry

end base_5_minus_base_8_in_base_10_l608_608168


namespace part_a_part_b_l608_608711

-- Condition definitions
def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n.digits.sum = 12)

def has_two_identical_digits (n : ℕ) : Prop :=
  let digits := n.digits in
  (digits.head = digits.tail.head ∧ digits.head ≠ digits.tail.tail.head) ∨
  (digits.head = digits.tail.tail.head ∧ digits.head ≠ digits.tail.head) ∨
  (digits.tail.head = digits.tail.tail.head ∧ digits.tail.head ≠ digits.head)

def all_odd_digits (n : ℕ) : Prop :=
  n.digits.all (λ d, d % 2 = 1)

-- Part (a): Number of valid numbers with exactly two identical digits is 11
theorem part_a : (finset.filter is_valid_number finset.range 1000).filter has_two_identical_digits).card = 11 := 
sorry

-- Part (b): Number of valid numbers with only odd digits is 0
theorem part_b : (finset.filter is_valid_number finset.range 1000).filter all_odd_digits).card = 0 :=
sorry

end part_a_part_b_l608_608711


namespace curve_transformation_l608_608653

theorem curve_transformation :
  ∀ (x y x' y' : ℝ),
  (y = (1/3) * cos (2 * x)) →
  (x' = 2 * x) →
  (y' = 3 * y) →
  y' = cos x' :=
by
  intros x y x' y' h1 h2 h3
  sorry

end curve_transformation_l608_608653


namespace alia_time_difference_l608_608441

def alia_cycle_distance := 18 -- miles
def alia_cycle_time := 2 * 60 -- minutes (2 hours)
def alia_walk_distance := 8 -- kilometers
def alia_walk_time := 3 * 60 -- minutes (3 hours)
def mile_to_kilometer := 1.609

noncomputable def cycling_minutes_per_kilometer := (alia_cycle_time / alia_cycle_distance) / mile_to_kilometer
noncomputable def walking_minutes_per_kilometer := alia_walk_time / alia_walk_distance
noncomputable def time_difference := walking_minutes_per_kilometer - cycling_minutes_per_kilometer

theorem alia_time_difference : 
  18 ≈ time_difference := by 
  sorry

end alia_time_difference_l608_608441


namespace geometric_sequence_sum_eq_30_l608_608948

variable {a : ℕ → ℝ} (q : ℝ)

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a (i + 1)

theorem geometric_sequence_sum_eq_30
  (a : ℕ → ℝ) (h_geo : is_geometric_sequence a q)
  (h1 : a 2 * a 3 = 2 * a 1)
  (h2 : (a 4 + 2 * a 7) / 2 = 5 / 4) :
  sum_first_n_terms a 4 = 30 := by
  sorry

end geometric_sequence_sum_eq_30_l608_608948


namespace fish_in_third_tank_l608_608728

-- Definitions of the conditions
def first_tank_goldfish : ℕ := 7
def first_tank_beta_fish : ℕ := 8
def first_tank_fish : ℕ := first_tank_goldfish + first_tank_beta_fish

def second_tank_fish : ℕ := 2 * first_tank_fish

def third_tank_fish : ℕ := second_tank_fish / 3

-- The statement to prove
theorem fish_in_third_tank : third_tank_fish = 10 := by
  sorry

end fish_in_third_tank_l608_608728


namespace find_X_l608_608709

theorem find_X (X : ℚ) (h : (1/3 : ℚ) * (1/4 : ℚ) * X = (1/4 : ℚ) * (1/6 : ℚ) * 120) : X = 60 := 
sorry

end find_X_l608_608709


namespace complex_omega_sum_l608_608361

open Complex

theorem complex_omega_sum (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45 + ω^48 + ω^51 + ω^54 + ω^57 + ω^60 + ω^63 = 1 := 
by
  sorry

end complex_omega_sum_l608_608361


namespace normal_distribution_interval_prob_within_minus4_to_minus2_l608_608954

noncomputable def normal_distribution_problem (μ σ : ℝ) (P : set ℝ → ℝ) :=
  ∀ x y z : ℝ,
  (x = μ - σ ∧ y = μ + σ ∧ z = 0.6826) ∧
  (x = μ - 2*σ ∧ y = μ + 2*σ ∧ z = 0.9544) ∧
  (x = μ - 3*σ ∧ y = μ + 3*σ ∧ z = 0.9974) ∧
  (even_density_function : ∀ x : ℝ, (density_function x) = (density_function (-x))) ∧
  (density_function_max : ∀ x : ℝ, density_function x ≤ (1/(2 * sqrt (2*real.pi)))) →
  P (Icc (-4) (-2)) = 0.1359

theorem normal_distribution_interval_prob_within_minus4_to_minus2
  (μ σ : ℝ) (P : set ℝ → ℝ) :
  normal_distribution_problem μ σ P :=
begin
  sorry
end

end normal_distribution_interval_prob_within_minus4_to_minus2_l608_608954


namespace no_prime_quadruple_l608_608716

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_quadruple 
    (a b c d : ℕ)
    (ha_prime : is_prime a) 
    (hb_prime : is_prime b)
    (hc_prime : is_prime c)
    (hd_prime : is_prime d)
    (h_order : a < b ∧ b < c ∧ c < d) :
    (1 / a + 1 / d ≠ 1 / b + 1 / c) := 
by 
  sorry

end no_prime_quadruple_l608_608716


namespace find_matrix_calculate_M5_alpha_l608_608939

-- Define the matrix M, eigenvalues, eigenvectors and vector α
def M : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 2], ![3, 2]]
def alpha : Fin 2 → ℝ := ![-1, 1]
def e1 : Fin 2 → ℝ := ![2, 3]
def e2 : Fin 2 → ℝ := ![1, -1]
def lambda1 : ℝ := 4
def lambda2 : ℝ := -1

-- Conditions: eigenvalues and their corresponding eigenvectors
axiom h1 : M.mulVec e1 = lambda1 • e1
axiom h2 : M.mulVec e2 = lambda2 • e2

-- Condition: given vector α
axiom h3 : alpha = - e2

-- Prove that M is the matrix given by the components
theorem find_matrix : M = ![![1, 2], ![3, 2]] :=
sorry

-- Prove that M^5 times α equals the given vector
theorem calculate_M5_alpha : (M^5).mulVec alpha = ![-1, 1] :=
sorry

end find_matrix_calculate_M5_alpha_l608_608939


namespace arithmetic_sequence_a_n_geometric_sequence_b_n_sum_of_sequence_c_n_l608_608267

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

def b_n (n : ℕ) : ℝ := if n = 1 then 3 else (2 * n + 1) / (2 * n - 1)

def c_n (n : ℕ) : ℝ := (-1 : ℝ) ^ n * (4 * n / ((2 * n - 1) * (2 * n + 1)))

def T_n (n : ℕ) : ℝ :=
  if n % 2 = 0 then -((2 * n : ℝ) / (2 * n + 1)) else -((2 * n + 2) / (2 * n + 1))

theorem arithmetic_sequence_a_n (n : ℕ) :
  ∀ n, a_n 10 = 19 ∧ (10 * a_n 1 + 45) = 100 → a_n n = 2 * n - 1 :=
sorry

theorem geometric_sequence_b_n (n : ℕ) (b_seq : ℕ → ℝ) :
  (b_seq 1 * b_seq 2 * ... * b_seq n = a_n n + 2) → b_seq n = if n = 1 then 3 else (2 * n + 1) / (2 * n - 1) :=
sorry

theorem sum_of_sequence_c_n (n : ℕ) :
  ∑ k in range n, c_n k = T_n n :=
sorry

end arithmetic_sequence_a_n_geometric_sequence_b_n_sum_of_sequence_c_n_l608_608267


namespace dice_probability_216_l608_608443

theorem dice_probability_216 : 
  let dice_values := {a // a ∈ {1, 2, 3, 4, 5, 6}}
  let event := {d : dice_values × dice_values × dice_values // (d.1.1 : ℕ) * (d.1.2 : ℕ) * (d.2 : ℕ) = 216}
  (↑((event.card : ℕ)) / (dice_values.card ^ 3 : ℕ)) = 1 / 216 :=
sorry

end dice_probability_216_l608_608443


namespace three_digit_sum_reverse_eq_l608_608073

theorem three_digit_sum_reverse_eq :
  ∃ (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9),
    101 * (a + c) + 20 * b = 1777 ∧ (a, b, c) = (8, 7, 9) ∨ (a, b, c) = (9, 7, 8) :=
by
  sorry

end three_digit_sum_reverse_eq_l608_608073


namespace find_scalars_l608_608360

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 2;
    3, 1]

noncomputable def B4 : Matrix (Fin 2) (Fin 2) ℝ :=
  B * B * B * B

noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ :=
  1

theorem find_scalars (r s : ℝ) (hB : B^4 = r • B + s • I) :
  (r, s) = (51, 52) :=
  sorry

end find_scalars_l608_608360


namespace repeating_decimal_sum_l608_608760

theorem repeating_decimal_sum:
  let c : ℕ := 3
  let d : ℕ := 8
  (5 / 13 : ℚ) = ((((c : ℚ) * 10 + d) * 10 ^ -2) : ℚ) : ℚ --
  (c + d = 11) :=
by
  sorry

end repeating_decimal_sum_l608_608760


namespace problem_statement_l608_608520

def cube_root (x : ℝ) : ℝ := x^(1/3)
def zero_exponent (x : ℝ) : ℝ := x^0
def absolute_value (x : ℝ) : ℝ := abs x
def square_root (x : ℝ) : ℝ := real.sqrt x

theorem problem_statement : 
  - cube_root 8 + zero_exponent 2016 + absolute_value (1 - square_root 4) = 0 :=
by
  sorry

end problem_statement_l608_608520


namespace alex_casey_meet_probability_l608_608144

noncomputable def probability_meet : ℚ :=
  let L := (1:ℚ) / 3;
  let area_of_square := 1;
  let area_of_triangles := (1 / 2) * L ^ 2;
  let area_of_meeting_region := area_of_square - 2 * area_of_triangles;
  area_of_meeting_region / area_of_square

theorem alex_casey_meet_probability :
  probability_meet = 8 / 9 :=
by
  sorry

end alex_casey_meet_probability_l608_608144


namespace find_divisor_l608_608989

def arun_age : ℕ := 60
def madan_age : ℕ := 5
def gokul_age : ℕ := madan_age - 2
def remainder : ℕ := arun_age - 6
def divisor : ℕ := remainder / gokul_age

theorem find_divisor : divisor = 18 :=
by 
  have h1 : arun_age - 6 = 54 := rfl
  have h2 : madan_age - 2 = 3 := rfl
  have h3 : 54 / 3 = 18 := rfl
  sorry

end find_divisor_l608_608989


namespace three_digit_sum_reverse_eq_l608_608071

theorem three_digit_sum_reverse_eq :
  ∃ (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9),
    101 * (a + c) + 20 * b = 1777 ∧ (a, b, c) = (8, 7, 9) ∨ (a, b, c) = (9, 7, 8) :=
by
  sorry

end three_digit_sum_reverse_eq_l608_608071


namespace trajectory_of_P_tangent_proof_l608_608937

noncomputable def point_distance (P : ℝ × ℝ) (A : ℝ × ℝ) : ℝ := 
  real.sqrt ((P.1 - A.1) ^ 2 + (P.2 - A.2) ^ 2)

theorem trajectory_of_P
  (P : ℝ × ℝ)
  (F : ℝ × ℝ) (l : ℝ)
  (h1 : F = (0, 2))
  (h2 : l = -2)
  (h3 : point_distance P F = abs (P.2 - l)) :
  ∃ C : ℝ × ℝ → Prop, (∀ P, C(P) ↔ (P.1 ^ 2 = 8 * P.2)) :=
sorry

theorem tangent_proof
  (Q : ℝ × ℝ)
  (C : ℝ × ℝ → Prop)
  (F : ℝ × ℝ) (l : ℝ)
  (h1 : F = (0, 2))
  (h2 : l = -2)
  (hQ : Q.2 = l)
  (hC : ∀ P, C(P) ↔ (P.1 ^ 2 = 8 * P.2)):
  (∀ A B, Q, C(A), C(B) 
    → -- proving line AB passes through F
      ∃ AB : ℝ × ℝ, 
        (AB.1 = (A.1 + B.1) / 2) ∧ (AB.2 = (A.2 + B.2) / 2)
        ∧ (∃ (AF BF : ℝ × ℝ), AF = (A.1 - F.1, A.2 - F.2) 
        ∧ BF = (B.1 - F.1, B.2 - F.2) 
        ∧ (AF.2 / AF.1) = (BF.2 / BF.1))
  ∧ -- proving the circle with diameter AB is tangent to line l
      ∃ M : ℝ × ℝ, 
        M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) 
        ∧ M.1 = Q.1):
sorry

end trajectory_of_P_tangent_proof_l608_608937


namespace probability_no_3_same_color_after_4_purchases_expectation_of_X_l608_608063

-- Probability that after 4 purchases, no 3 cards of the same color are collected
theorem probability_no_3_same_color_after_4_purchases :
  let num_colors := 3 in
  let probability := (num_colors.choose 1 * 4.choose 2 * 2.choose 1 + 
                      num_colors.choose 2 * (4.choose 2) * (4.choose 2) / num_colors^4) in
  probability = 2 / 3 :=
by sorry

-- Expectation E[X], number of times purchases made before collecting 3 cards of the same color
theorem expectation_of_X :
  let X_dist (n : Nat) : Rational :=
    match n with
    | 3 => 1 / 9
    | 4 => 2 / 9
    | 5 => 8 / 27
    | 6 => 20 / 81
    | 7 => 10 / 81
    | _ => 0 in
  let E_X := 3 * (1 / 9) + 4 * (2 / 9) + 5 * (8 / 27) + 6 * (20 / 81) + 7 * (10 / 81) in
  E_X = 409 / 81 :=
by sorry

end probability_no_3_same_color_after_4_purchases_expectation_of_X_l608_608063


namespace percentage_of_number_l608_608067

/-- 
  Given a certain percentage \( P \) of 600 is 90.
  If 30% of 50% of a number 4000 is 90,
  Then P equals to 15%.
-/
theorem percentage_of_number (P : ℝ) (h1 : (0.30 : ℝ) * (0.50 : ℝ) * 4000 = 600) (h2 : P * 600 = 90) :
  P = 0.15 :=
  sorry

end percentage_of_number_l608_608067


namespace express_w_l608_608047

theorem express_w (a b c x y z w : ℝ) (h : 157 ≠ w) (habc : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (hx : x + y + z = 1) 
  (hxa2 : x * a^2 + y * b^2 + z * c^2 = w^2) 
  (hxa3 : x * a^3 + y * b^3 + z * c^3 = w^3) 
  (hxa4 : x * a^4 + y * b^4 + z * c^4 = w^4) :
  w = if ab := a * b; ac := a * c; bc := b * c; ab + ac + bc = 0 then - ab / (a + b) else - (ab * c) / (ab + ac + bc) :=
sorry

end express_w_l608_608047


namespace find_m_l608_608375

-- Define the angles for l1 and l2
def theta1 : ℝ := Real.pi / 60
def theta2 : ℝ := Real.pi / 48

-- Define the transformation R
def R (theta : ℝ) : ℝ :=
  let theta' := 2 * theta1 - theta -- Reflecting across l1
  2 * theta2 - theta' -- Reflecting across l2

-- Define the line l in terms of its angle with the x-axis
def theta_l : ℝ := Real.arctan (17 / 89)

-- Define the recursive transformation R^n
def Rn (n : ℕ) (theta : ℝ) : ℝ :=
  (List.foldl (fun θ _ => R θ) theta (List.range n))

-- The given problem
theorem find_m : ∃ m : ℕ, m > 0 ∧ Rn m theta_l = theta_l :=
  sorry

end find_m_l608_608375


namespace profit_percentage_l608_608867

theorem profit_percentage (CP SP : ℝ) (hCP : CP = 150) (hSP : SP = 216.67) :
  SP = 0.9 * LP ∧ LP = SP / 0.9 ∧ Profit = SP - CP ∧ Profit_Percentage = (Profit / CP) * 100 ∧ Profit_Percentage = 44.44 :=
by
  sorry

end profit_percentage_l608_608867


namespace revenue_increase_l608_608824

-- Lean statement to prove the effect on revenue receipts.
theorem revenue_increase (P Q : ℝ) :
  let P_new := 1.80 * P in
  let Q_new := 0.80 * Q in
  let R := P * Q in
  let R_new := P_new * Q_new in
  let Effect := R_new - R in
  Effect = 0.44 * P * Q :=
by
  sorry

end revenue_increase_l608_608824


namespace function_c_monotonically_increasing_l608_608892

noncomputable def f : ℝ → ℝ := λ x, (x + 1) ^ 2

theorem function_c_monotonically_increasing :
  ∀ x1 x2 : ℝ, (0 < x1) → (0 < x2) → (x1 < x2) → (f x1 ≤ f x2) :=
by
  intros x1 x2 h1 h2 h3
  sorry

end function_c_monotonically_increasing_l608_608892


namespace length_of_AB_l608_608333

open Real

-- Define the given conditions
variables {A B C M N G : Point}
variables {AM BN : Line}
variables (h_triangle : ∃ (A B C : Point), is_triangle A B C)
variables (h_med_AM : AM.is_median A A B C)
variables (h_med_BN : BN.is_median B A B C)
variables (h_perpendicular : AM.is_perpendicular BN)
variables (h_AM_len : AM.length = 15)
variables (h_BN_len : BN.length = 20)
variables (h_height_C : height C ↔ h_height_C = 12)

-- Define the length of side AB
def length_AB (A B : Point) : Real := distance A B

-- The conjecture to prove
theorem length_of_AB (A B C M N : Point) (G : Point) (AM BN : Line)
  (h_triangle : ∃ (A B C : Point), is_triangle A B C)
  (h_med_AM : AM.is_median A A B C)
  (h_med_BN : BN.is_median B A B C)
  (h_perpendicular : AM.is_perpendicular BN)
  (h_AM_len : AM.length = 15)
  (h_BN_len : BN.length = 20)
  (h_height_C : height C ↔ h_height_C = 12) : 
  length_AB A B = 50 / 3 :=
sorry

end length_of_AB_l608_608333


namespace probability_x_plus_y_lt_4_l608_608118

open Set

-- Define the square and the line
def square : Set (ℝ × ℝ) := { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 }
def line_lt_4 : Set (ℝ × ℝ) := { p | p.1 + p.2 < 4 }

-- The probability to prove
theorem probability_x_plus_y_lt_4 : 
  (volume (square ∩ line_lt_4) / volume square) = 7 / 9 := 
sorry

end probability_x_plus_y_lt_4_l608_608118


namespace both_increasing_l608_608372

-- Define the domain
variables {R : Type*} [LinearOrderedField R]

-- Define functions f and g
variable {f g : R → R}

-- Conditions
def strictly_increasing (f : R → R) := ∀ x1 x2 : R, x1 < x2 → f x1 < f x2

def inequality_condition (f g : R → R) := 
  ∀ x1 x2 : R, x1 ≠ x2 → (f x1 - f x2)^2 > (g x1 - g x2)^2

-- Functions F and G
def F (x : R) := f x + g x
def G (x : R) := f x - g x

-- Theorem stating the main result
theorem both_increasing (hf : strictly_increasing f) (hg : inequality_condition f g) :
  (strictly_increasing F) ∧ (strictly_increasing G) :=
sorry

end both_increasing_l608_608372


namespace cos_irreducible_fraction_l608_608388

noncomputable def int_triangle (A : ℝ) : Prop := ∃ a b c : ℤ, a + b + c = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ (A = a ∨ A = b ∨ A = c)

theorem cos_irreducible_fraction (A : ℝ) (m n : ℤ) (h : int_triangle A) (h_sin : real.sin A = m / n) (h_gcd : Int.gcd m n = 1) (h_odd : n % 2 = 1) : 
  ∃ k : ℤ, real.cos A = k / n ∧ Int.gcd k n = 1 ∧ (m % 2 = 0 → k % 2 ≠ 0) ∧ (m % 2 ≠ 0 → k % 2 = 0) := 
sorry

end cos_irreducible_fraction_l608_608388


namespace distance_between_points_l608_608807

theorem distance_between_points :
  ∀ (x1 y1 x2 y2 : ℝ), 
  x1 = -4 → y1 = 3 → x2 = 6 → y2 = -7 → 
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 10 * sqrt 2 := 
by
  intros
  sorry

end distance_between_points_l608_608807


namespace area_and_perimeter_l608_608670

-- Given a rectangle R with length l and width w
variables (l w : ℝ)
-- Define the area of R
def area_R : ℝ := l * w

-- Define a smaller rectangle that is cut out, with an area A_cut
variables (A_cut : ℝ)
-- Define the area of the resulting figure S
def area_S : ℝ := area_R l w - A_cut

-- Define the perimeter of R
def perimeter_R : ℝ := 2 * l + 2 * w

-- perimeter_R remains the same after cutting out the smaller rectangle
theorem area_and_perimeter (h_cut : 0 < A_cut) (h_cut_le : A_cut ≤ area_R l w) : 
  (area_S l w A_cut < area_R l w) ∧ (perimeter_R l w = perimeter_R l w) :=
by
  sorry

end area_and_perimeter_l608_608670


namespace angle_FAB_is_60_degrees_l608_608630

-- Conditions
variables {A B C F : Type*} [point_space A B C F]
variables (triangle_isosceles : isosceles_triangle A B C)
variables (equilateral_triangle : equilateral_triangle B C F)
variables (F_outside_triangle : ∀ (P : point), ¬ inside_triangle P A B C)

-- Definition of the problem
def measure_angle_FAB : ℝ := 
  let F_outside := F_outside_triangle F in 
  if triangle_isosceles
  and equilateral_triangle 
  then 60 else 0

-- Statement to prove
theorem angle_FAB_is_60_degrees : measure_angle_FAB = 60 := 
  sorry

#align (angle_FAB_is_60_degrees : measure_angle_FAB = 60)

end angle_FAB_is_60_degrees_l608_608630


namespace correct_option_l608_608603

-- Definitions representing the conditions
variable (a b c : Line) -- Define the lines a, b, and c

-- Conditions for the problem
def is_parallel (x y : Line) : Prop := -- Define parallel property
  sorry

def is_perpendicular (x y : Line) : Prop := -- Define perpendicular property
  sorry

noncomputable def proof_statement : Prop :=
  is_parallel a b → is_perpendicular a c → is_perpendicular b c

-- Lean statement of the proof problem
theorem correct_option (h1 : is_parallel a b) (h2 : is_perpendicular a c) : is_perpendicular b c :=
  sorry

end correct_option_l608_608603


namespace average_rate_of_change_l608_608905

noncomputable def f (x : ℝ) := 2 * x + 1

theorem average_rate_of_change :
  (f 2 - f 1) / (2 - 1) = 2 :=
by
  sorry

end average_rate_of_change_l608_608905


namespace problem_statement_l608_608348

def binom (n k : ℕ) : ℕ := Nat.choose n k

def A : ℕ := 
  (Finset.range 1006).sum (λ k, (binom 2010 k - binom 2010 (k - 1))^2)

def min_s : ℕ := 2011

theorem problem_statement : min_s * A ≥ binom 4020 2010 := by
  sorry

end problem_statement_l608_608348


namespace problem_solution_l608_608697
noncomputable def f (x : ℝ) : ℝ :=
if x > 0 ∧ x ≤ 1 then 1 + Real.log2 x 
else 0 -- Placeholder for the periodic and odd function

theorem problem_solution :
(f (2014) + f (2016) - 2 * f (2015) = 2) ∧
(∀ x : ℝ, f (-x) = -f (x)) ∧
(∀ x : ℝ, f (x + 4) = f (x)) :=
by
  sorry

end problem_solution_l608_608697


namespace inequality_three_var_l608_608569

theorem inequality_three_var
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c) :
  2 * (a^3 + b^3 + c^3) ≥ a^2 * b + a * b^2 + a^2 * c + a * c^2 + b^2 * c + b * c^2 :=
by sorry

end inequality_three_var_l608_608569


namespace postage_cost_correct_l608_608996

def postage_cost (W : ℝ) : ℝ :=
  8 + 5 * (Real.to_nnreal (W - 1)).ceil

theorem postage_cost_correct (W : ℝ) : 
  postage_cost W = 8 + 5 * max 0 (Real.to_nnreal (W - 1)).ceil :=
begin
  unfold postage_cost,
  sorry
end

end postage_cost_correct_l608_608996


namespace number_of_odd_terms_l608_608216

theorem number_of_odd_terms (n : ℕ) (h : n ≥ 1) :
  let product := ∏ i in (finset.range n).filter (λ i, ∀ j, j > i ∧ j < n), (x i + x j)
  let expanded := expand product
  ∃ count : ℕ, count = ∑ term in expanded, if term.coeff % 2 = 1 then 1 else 0 
  ⟹ count = n! := sorry

end number_of_odd_terms_l608_608216


namespace range_of_m_when_p_true_range_of_m_when_p_and_q_false_p_or_q_true_l608_608576

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - 9 * Real.log x

def p (m : ℝ) : Prop :=
  ∀ x ∈ (Set.Ioo m (m + 1)), (x - 9 / x) < 0

def q (m : ℝ) : Prop :=
  m > 1 ∧ m < 3

theorem range_of_m_when_p_true :
  ∀ m : ℝ, p m → 0 ≤ m ∧ m ≤ 2 :=
sorry

theorem range_of_m_when_p_and_q_false_p_or_q_true :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (0 ≤ m ∧ m ≤ 1) ∨ (2 < m ∧ m < 3) :=
sorry

end range_of_m_when_p_true_range_of_m_when_p_and_q_false_p_or_q_true_l608_608576


namespace cos_alpha_sub_beta_l608_608170

noncomputable theory
open_locale classical

variables {α β : ℝ}

theorem cos_alpha_sub_beta (h1 : cos α + cos β = -4/5) (h2 : sin α + sin β = 1/3) :
  cos (α - β) = -28/225 :=
sorry

end cos_alpha_sub_beta_l608_608170


namespace unique_checkerboard_coloration_l608_608434

theorem unique_checkerboard_coloration (n : ℕ) (h : n ≥ 2) :
  ∃! f : ℕ × ℕ → bool, ∀ i j, f (i, j) ≠ f (i + 1, j) ∧ f (i, j) ≠ f (i, j + 1) :=
sorry

end unique_checkerboard_coloration_l608_608434


namespace ratio_SP2_SP1_l608_608506

variable (CP : ℝ)

-- First condition: Sold at a profit of 140%
def SP1 := 2.4 * CP

-- Second condition: Sold at a loss of 20%
def SP2 := 0.8 * CP

-- Statement: The ratio of SP2 to SP1 is 1 to 3
theorem ratio_SP2_SP1 : SP2 / SP1 = 1 / 3 :=
by
  sorry

end ratio_SP2_SP1_l608_608506


namespace probability_intersection_is_3_elements_l608_608277

-- Definitions based on conditions
def U : Set ℕ := {1, 2, 3, 4, 5}
def I : Set (Set ℕ) := {X | X ⊆ U}

-- Main Statement
theorem probability_intersection_is_3_elements :
  (∃ A B ∈ I, A ≠ B ∧ (A ∩ B).card = 3) → (5 / 62) := 
sorry

end probability_intersection_is_3_elements_l608_608277


namespace sum_fractions_eq_zero_l608_608683

noncomputable def f (x : ℝ) : ℝ := x^2 * (1 - x)^2

theorem sum_fractions_eq_zero :
  (∑ k in finset.range 1011, (f (k+1)/2023) - f (2023 - (k+1))/2023) = 0 := by
sorry

end sum_fractions_eq_zero_l608_608683


namespace ceiling_sum_evaluation_l608_608207

theorem ceiling_sum_evaluation :
  (⌈Real.sqrt (25 / 9)⌉ + ⌈(25 / 9)⌉ + ⌈(25 / 9) ^ 2⌉) = 13 :=
by
  sorry

end ceiling_sum_evaluation_l608_608207


namespace inequality_of_arithmetic_progression_l608_608953

theorem inequality_of_arithmetic_progression 
  (n : ℕ) (h1 : n ≥ 2)
  (a : Fin n.succ.succ → ℝ)
  (h2 : ∀ i : Fin n.succ, 0 < a i)
  (h3 : ∃ d : ℝ, d ≥ 0 ∧ ∀ i : Fin n, a i.succ - a i = d) : 
  (∑ k in Finset.range (n - 1), (1 / (a (Fin.ofNat (k + 2)))^2)) ≤ 
  (n - 1) / 2 * ((a 0 * a (Fin.ofNat n) + a 1 * a (Fin.ofNat (n + 1))) / (a 0 * a 1 * a (Fin.ofNat n) * a (Fin.ofNat (n + 1)))) :=
by
  sorry

end inequality_of_arithmetic_progression_l608_608953


namespace value_of_3a_plus_b_l608_608617

theorem value_of_3a_plus_b (a b : ℤ) (h1 : a ≠ 0)
  (h2 : ∃ x1 x2 : ℕ, prime x1 ∧ prime x2 ∧ x1 ≠ x2 ∧ 
        a * x1^2 + b * x1 - 2008 = 0 ∧ 
        a * x2^2 + b * x2 - 2008 = 0) :
  3 * a + b = 1000 :=
sorry

end value_of_3a_plus_b_l608_608617


namespace ratio_of_areas_l608_608943

-- Define the geometric conditions
variables {Point : Type*} [AffineSpace ℝ Point]
variables (A B C D E F : Point)

-- Midpoint condition for E on diagonal BD
def is_midpoint (P Q R : Point) : Prop :=
  ∃ (k : ℝ), k = 0.5 ∧ R = P +ᵥ k • (Q -ᵥ P)

-- Condition for E being midpoint of BD
axiom E_midpoint : is_midpoint B D E

-- Condition for F such that DF = ⅓ DA
axiom F_on_DA : ∃ (k : ℝ), k = \frac{1}{3} ∧ F = D +ᵥ k • (A -ᵥ D)

-- Areas of triangles and quadrilateral
noncomputable def area_triangle (P Q R : Point) : ℝ := sorry
noncomputable def area_quadrilateral (P Q R S : Point) : ℝ := sorry

-- The main theorem statement
theorem ratio_of_areas (h1 : is_midpoint B D E) (h2 : ∃ (k : ℝ), k = \frac{1}{3} ∧ F = D +ᵥ k • (A -ᵥ D)) :
  area_triangle D F E / area_quadrilateral A B E F = 1 / 5 :=
sorry

end ratio_of_areas_l608_608943


namespace range_h_l608_608222

def h (t : ℝ) : ℝ := (t^2 + (1/2) * t) / (t^2 + 2)

theorem range_h : set.range h = {1/4} :=
by
  -- Proof can be added here
  sorry

end range_h_l608_608222


namespace probability_of_x_plus_y_lt_4_l608_608099

open Classical -- To handle probability and random selection
open Set -- For geometric notions
open Filter -- For measure and integration

noncomputable def probability_condition (x y : ℝ) (h : 0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) : Prop := x + y < 4

theorem probability_of_x_plus_y_lt_4 :
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 3 ∧ 0 ≤ y ∧ y ≤ 3) →
                pr (λ (xy : ℝ × ℝ), probability_condition xy.1 xy.2 (and.intro (and.left xy.2) (and.right (and.left xy.2)))) = 7/9 :=
by sorry

end probability_of_x_plus_y_lt_4_l608_608099


namespace train_passes_jogger_in_54_seconds_l608_608085

def jogger_speed_kmh : ℝ := 9
def train_speed_kmh : ℝ := 45
def jogger_lead_m : ℝ := 360
def train_length_m : ℝ := 180

def time_to_pass_jogger (jogger_speed_kmh train_speed_kmh jogger_lead_m train_length_m : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed_kmh * (1000 / 3600)
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let relative_speed_ms := train_speed_ms - jogger_speed_ms
  let total_distance_m := jogger_lead_m + train_length_m
  total_distance_m / relative_speed_ms

theorem train_passes_jogger_in_54_seconds :
  time_to_pass_jogger jogger_speed_kmh train_speed_kmh jogger_lead_m train_length_m = 54 :=
by {
  sorry
}

end train_passes_jogger_in_54_seconds_l608_608085


namespace complement_A_l608_608973

noncomputable def A : set ℝ := {x | x > 1}

theorem complement_A :
  (Aᶜ : set ℝ) = {x | x ≤ 1} :=
by
  sorry

end complement_A_l608_608973


namespace moses_more_than_esther_l608_608797

theorem moses_more_than_esther (total_amount: ℝ) (moses_share: ℝ) (tony_esther_share: ℝ) :
  total_amount = 50 → moses_share = 0.40 * total_amount → 
  tony_esther_share = (total_amount - moses_share) / 2 → 
  moses_share - tony_esther_share = 5 :=
by
  intros h1 h2 h3
  sorry

end moses_more_than_esther_l608_608797


namespace proof_part1_proof_part2_l608_608965

-- Definitions based on conditions
def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^2 + 2*x else 1 * x^2 + 2 * x

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = -f(x)

def is_monotonically_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f(x) ≤ f(y)

-- Statement of the problem
theorem proof_part1 (a b : ℝ) (h1 : is_odd_function (λ x, if x ≥ 0 then -x^2 + 2*x else a*x^2 + b*x)) :
  a - b = -1 :=
sorry

theorem proof_part2 (f : ℝ → ℝ)
  (h2 : is_monotonically_increasing_on_interval f (-1) (m-2))
  (h3 : ∀ x, f(x) = (if x ≥ 0 then -x^2 + 2*x else x^2 + 2*x)) :
  1 < m ∧ m ≤ 3 :=
sorry

end proof_part1_proof_part2_l608_608965


namespace optimal_discount_savings_l608_608078

theorem optimal_discount_savings : 
  let total_amount := 15000
  let discount1 := 0.30
  let discount2 := 0.15
  let single_discount := 0.40
  let two_successive_discounts := total_amount * (1 - discount1) * (1 - discount2)
  let one_single_discount := total_amount * (1 - single_discount)
  one_single_discount - two_successive_discounts = 75 :=
by
  sorry

end optimal_discount_savings_l608_608078


namespace count_integers_satisfying_sqrt_condition_l608_608767

theorem count_integers_satisfying_sqrt_condition : 
  let y_conditions := { y : ℝ | 6 > Real.sqrt y ∧ Real.sqrt y > 3 }
  let integer_satisfying_set := { y : ℕ | y ∈ y_conditions }
  integer_satisfying_set.card = 26 :=
by
  sorry

end count_integers_satisfying_sqrt_condition_l608_608767


namespace number_of_intersections_l608_608604

open Set

-- Defining the lines as sets of points in the plane
def line1 := {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ 3 * x - y - 1 = 0}
def line2 := {p : ℝ × ℝ | ∃ x y, p = (x, y) ∧ x + 2 * y - 5 = 0}
def line3 := {p : ℝ × ℝ | ∃ x, p = (x, 1)}
def line4 := {p : ℝ × ℝ | ∃ y, p = (3, y)}

-- Defining the set of intersection points
def intersection_points := (line1 ∩ line2) ∪ (line1 ∩ line3) ∪ (line1 ∩ line4) ∪ 
                          (line2 ∩ line3) ∪ (line2 ∩ line4) ∪ (line3 ∩ line4)

-- Prove that the number of distinct intersection points is 2
theorem number_of_intersections : Finset.card (intersection_points.toFinset) = 2 :=
sorry

end number_of_intersections_l608_608604


namespace subset_k_numbers_l608_608240

open Lean.Meta

-- Definitions and conditions
def non_increasing_seq (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j : ℕ, (1 ≤ i ∧ i < j ∧ j ≤ n) → a i ≥ a j

def is_positive_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i : ℕ, (1 ≤ i ∧ i ≤ n) → a i > 0

def seq_sum_one (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (∑ i in (Finset.range n).map (Embedding.subtype _), a (i + 1)) = 1

-- Lean statement of the problem without the proof.
theorem subset_k_numbers (a : ℕ → ℝ) (n k : ℕ) :
  non_increasing_seq a n →
  is_positive_sequence a n →
  seq_sum_one a n →
  a 1 = 1 / (2 * k) →
  ∃ (S : Finset ℕ), S.card = k ∧ (∃ m : ℕ, m ∈ S ∧ a m > 1 / (4 * k)) := 
begin
  sorry
end

end subset_k_numbers_l608_608240


namespace cost_price_computer_table_l608_608754

theorem cost_price_computer_table (C : ℝ) (FP : ℝ) (h : FP = 8325) : 
  C = 8325 / 1.08675 :=
by
  have h1 : FP = 1.08675 * C := by sorry
  rw [h1, h] at h1
  exact (EQ).mpr (eq_div_of_mul_eq h1)

end cost_price_computer_table_l608_608754


namespace tangent_accuracy_l608_608445

theorem tangent_accuracy {θ : ℝ} (hθ : 0 ≤ θ ∧ θ < 90):
  (∀ θ, 0 ≤ θ ∧ θ < 90 → 
    (tan 0 = 0 ∧ sin 0 = 0) ∧
    (tan 90 = 1/0 ∧ sin 90 = 1) ∧
    (∀ x y, 0 ≤ x ∧ x < 90 ∧ 0 ≤ y ∧ y < 90 → (y > x → tan y > tan x ∧ sin y > sin x)) →
    (∀ ε, ε > 0 → ∃ η, η > 0 ∧ ∀ (θ1 θ2: ℝ), 0 ≤ θ1 ∧ θ1 < 90 ∧ 0 ≤ θ2 ∧ θ2 < 90 ∧ |θ1 - θ2| < η →  |tan θ1 - tan θ2| < ε ∨ |sin θ1 - sin θ2| < ε)) →
  (∃ n, nat.succ n ∧ -1 < tan θ ∧ tan θ ≤ 1 / (n : ℝ)). sorry

end tangent_accuracy_l608_608445


namespace probability_x_plus_y_less_than_4_l608_608103

open Set Real

/-- Define the square with vertices (0,0), (0,3), (3,3), and (3,0) -/
def square := set.Icc (0 : ℝ) 3 ×ˢ set.Icc (0 : ℝ) 3

/-- Probability that a randomly chosen point (x, y) in the square satisfies x + y < 4 -/
theorem probability_x_plus_y_less_than_4 : 
  let area_square := 9 in
  let area_excluded_triangle := 2 in
  let relevant_area := area_square - area_excluded_triangle in
  (relevant_area / area_square : ℝ) = 7 / 9 :=
by
{ sorry }

end probability_x_plus_y_less_than_4_l608_608103


namespace probability_three_heads_one_tail_is_one_fourth_l608_608994

-- Defining the probability function based on conditions
def prob_head : ℝ := 1 / 2
def prob_tail : ℝ := 1 / 2

-- Count possible cases with exactly three heads and one tail in four coin tosses
def count_three_heads_one_tail : ℕ := 4

-- Calculate the probability of each sequence of 4 coin tosses
def individual_probability : ℝ := (1 / 2) ^ 4

-- The total probability of getting exactly three heads and one tail
def total_probability : ℝ := count_three_heads_one_tail * individual_probability

-- Theorem to prove
theorem probability_three_heads_one_tail_is_one_fourth :
  total_probability = 1 / 4 :=
by
  -- Skipping the actual proof
  sorry

end probability_three_heads_one_tail_is_one_fourth_l608_608994


namespace three_digit_reverse_sum_to_1777_l608_608070

theorem three_digit_reverse_sum_to_1777 :
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ 101 * (a + c) + 20 * b = 1777 :=
by
  sorry

end three_digit_reverse_sum_to_1777_l608_608070


namespace hexagon_area_ratio_l608_608496

noncomputable def large_hexagon_side_length : ℝ := 3
noncomputable def inscribed_hexagon_side_length : ℝ := Real.sqrt 7

theorem hexagon_area_ratio :
  let ratio := (inscribed_hexagon_side_length / large_hexagon_side_length) ^ 2 in
  ratio = 7 / 9 :=
by
  sorry

end hexagon_area_ratio_l608_608496


namespace geometric_sequence_function_sum_l608_608239

theorem geometric_sequence_function_sum
  (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0) (h_geom : ∀ n, a (n+1) = q * a n)
  (h_q : q ≠ 1) (h_a3 : a 3 = 1) :
  let f (x : ℝ) := 3^x / (3^x + 1) in 
  f (Real.log (a 1)) + f (Real.log (a 2)) + f (Real.log (a 3)) + f (Real.log (a 4)) + f (Real.log (a 5)) = 5/2 := 
by
  sorry

end geometric_sequence_function_sum_l608_608239


namespace school_spent_total_l608_608495

noncomputable def seminar_fee (num_teachers : ℕ) : ℝ :=
  let base_fee := 150 * num_teachers
  if num_teachers >= 20 then
    base_fee * 0.925
  else if num_teachers >= 10 then
    base_fee * 0.95
  else
    base_fee

noncomputable def seminar_fee_with_tax (num_teachers : ℕ) : ℝ :=
  let fee := seminar_fee num_teachers
  fee * 1.06

noncomputable def food_allowance (num_teachers : ℕ) (num_special : ℕ) : ℝ :=
  let num_regular := num_teachers - num_special
  num_regular * 10 + num_special * 15

noncomputable def total_cost (num_teachers : ℕ) (num_special : ℕ) : ℝ :=
  seminar_fee_with_tax num_teachers + food_allowance num_teachers num_special

theorem school_spent_total (num_teachers num_special : ℕ) (h : num_teachers = 22 ∧ num_special = 3) :
  total_cost num_teachers num_special = 3470.65 :=
by
  sorry

end school_spent_total_l608_608495


namespace convex_number_probability_l608_608193

def convex_number (a b c d e : ℕ) : Prop :=
  a < b ∧ b < c ∧ c > d ∧ d > e

def five_digit_numbers : List (Nat × Nat × Nat × Nat × Nat) :=
  [(1, 2, 5, 4, 3), (1, 3, 5, 4, 2), (1, 4, 5, 3, 2),
   (2, 3, 5, 4, 1), (2, 4, 5, 3, 1), (3, 4, 5, 2, 1)]

def count_convex_numbers (lst : List (Nat × Nat × Nat × Nat × Nat)) : Nat :=
  lst.countp (λ n, match n with | (a, b, c, d, e) => convex_number a b c d e)

theorem convex_number_probability :
  (count_convex_numbers five_digit_numbers) = 6 →
  (5 * 4 * 3 * 2 * 1) = 120 →
  (1 / 20 : ℚ) = 6 / 120 :=
sorry

end convex_number_probability_l608_608193


namespace circumcenter_on_AK_l608_608827

variable {α β γ : Real}
variable (A B C L H K O : Type)
variable [Triangle ABC] (circumcenter : Triangle ABC → Point O)
variable [AngleBisector A B C L]

theorem circumcenter_on_AK
  (h₁ : AL_is_angle_bisector ABC L)
  (h₂ : Height_from_B_on_AL B A L H)
  (h₃ : K_on_circumcircle_ABL B A L K)
  : Lies_on_line (circumcenter ABC) A K :=
sorry

end circumcenter_on_AK_l608_608827


namespace measure_of_angle_B_prove_ac_sum_given_b_and_area_prove_b_given_area_and_ac_sum_prove_area_given_b_and_ac_sum_l608_608587

-- Definitions for the initial conditions
variable (A B C a b c S : ℝ)
variable (triangle_ABC : Prop)
variable (side_opposite_A : a)
variable (side_opposite_B : b)
variable (side_opposite_C : c)
axiom angle_relation : b * sin A / (a * cos B) = sqrt 3

-- Statement (1)
theorem measure_of_angle_B : B = π / 3 :=
sorry

-- Definitions for the additional conditions to be proven
axiom condition_b : b = 3
axiom condition_area : S = 9 * sqrt 3 / 4
axiom condition_ac_sum : a + c = 6

-- Statement (2): Choose two conditions and prove the remaining one
theorem prove_ac_sum_given_b_and_area : 
  (b = 3) → (S = 9 * sqrt 3 / 4) → (a + c = 6) :=
sorry

theorem prove_b_given_area_and_ac_sum : 
  (S = 9 * sqrt 3 / 4) → (a + c = 6) → (b = 3) :=
sorry

theorem prove_area_given_b_and_ac_sum : 
  (b = 3) → (a + c = 6) → (S = 9 * sqrt 3 / 4) :=
sorry

end measure_of_angle_B_prove_ac_sum_given_b_and_area_prove_b_given_area_and_ac_sum_prove_area_given_b_and_ac_sum_l608_608587


namespace hexagon_has_circumcircle_l608_608081

-- Define the conditions of a convex hexagon with specific properties on its long diagonals
variables {α : Type*} [EuclideanGeometry α]
noncomputable def convex_hexagon (A B C D E F : α) : Prop :=
  convex {A, B, C, D, E, F} ∧
  (∀ (long_diags : set (α × α)), 
    (forall_2 long_diags (λ (x y : α × α), 
    (x ∈ long_diags → y ∈ long_diags → 
     (x ≠ y → 
     (long_diags = {(A, D), (B, E), (C, F)} ∨ long_diags = {(A, D), (C, F)} ∨ long_diags = {(B, E), (C, F)}))))) ∧ 
    (∀ (x y z w : α), 
    (x, y, z, w) ∈ long_diags → 
    (x = A ∧ y = B ∧ z = C ∧ w = D) → 
    is_isosceles_triangle_with_base x y z ∧ is_isosceles_triangle_with_base y z w)))

-- Statement of the problem
theorem hexagon_has_circumcircle (A B C D E F : α) (h : convex_hexagon A B C D E F) :
  ∃ (O : α) (r : ℝ), ∀ (P ∈ {A, B, C, D, E, F}), dist P O = r :=
sorry

end hexagon_has_circumcircle_l608_608081


namespace cos_of_sin_l608_608567

theorem cos_of_sin (α : ℝ) (h : sin (α - π / 3) = 1 / 3) : cos (α + π / 6) = -1 / 3 :=
by
  sorry

end cos_of_sin_l608_608567


namespace walter_exceptional_days_l608_608806

theorem walter_exceptional_days (b w : ℕ) 
(h1 : b + w = 12) 
(h2 : 4 * b + 6 * w = 58) 
(h3 : 0 ≤ b) 
(h4 : 0 ≤ w) :
w = 5 :=
begin
  -- proof goes here
  sorry
end

end walter_exceptional_days_l608_608806


namespace cos_A_eq_neg_quarter_b_eq_3_l608_608334

open Triangle

-- Given conditions
variables {a b c : ℝ}
variables {A B C : ℝ}

-- First claim: cos A = -1/4
theorem cos_A_eq_neg_quarter
  (h_arith_seq : a + c = 2 * b)
  (h_a_eq_2c : a = 2 * c)
  (h_triangle : is_triangle a b c) : 
  real.cos A = -1 / 4 := 
begin
  sorry
end

-- Second claim: b = 3 given the area of the triangle
theorem b_eq_3
  (h_arith_seq : a + c = 2 * b)
  (h_a_eq_2c : a = 2 * c)
  (h_triangle_area : 1 / 2 * b * c * real.sin A = 3 * real.sqrt 15 / 4) 
  (h_cos_A : real.cos A = -1 / 4)
  (h_triangle : is_triangle a b c) : 
  b = 3 := 
begin
  sorry
end

end cos_A_eq_neg_quarter_b_eq_3_l608_608334


namespace log10_a_2020_approx_3_l608_608191

noncomputable def diamondsuit (a b : ℝ) : ℝ := a ^ (Real.log10 b)

noncomputable def heartsuit (a b : ℝ) : ℝ := a ^ (1 / (Real.log10 b))

noncomputable def a_seq : ℕ → ℝ
| 3 := heartsuit 5 3
| (n+1) := let a_n := a_seq n in let one_plus_a_n := a_n + 1 in
              diamondsuit (heartsuit (n+1) n) one_plus_a_n

theorem log10_a_2020_approx_3 : abs (Real.log10 (a_seq 2020) - 3) < 1 := by
  sorry

end log10_a_2020_approx_3_l608_608191


namespace parallelogram_area_ratio_l608_608527

noncomputable def parallelogram_problem : Prop :=
  let A := (0 : ℝ, 0 : ℝ)
  let B := (1 : ℝ, 0 : ℝ)
  let C := (1 : ℝ, 1 : ℝ)
  let D := (0 : ℝ, 1 : ℝ)

  let P := ((A.1 + C.1) / 2, (A.2 + C.2) / 2) -- midpoint of AC
  let Q := (C.1, C.2 * (1 / 4)) -- CQ = 1/4 CD
  let R := (A.1 + (2 / 3) * (B.1 - A.1), A.2) -- AR = 2/3 AB

  let area_triangle (A B C : ℝ × ℝ) : ℝ :=
    (1 / 2) * abs ((B.1 * C.2 - B.2 * C.1) + (C.1 * A.2 - C.2 * A.1) + (A.1 * B.2 - A.2 * B.1))

  let area_pentagon (A B P Q R : ℝ × ℝ) : ℝ :=
    (1 / 2) * abs (
      (A.1 * B.2 + B.1 * P.2 + P.1 * Q.2 + Q.1 * R.2 + R.1 * A.2)
      - (A.2 * B.1 + B.2 * P.1 + P.2 * Q.1 + Q.2 * R.1 + R.2 * A.1)
    )

  let area_ΔCPQ := area_triangle C P Q
  let area_⭐ABPQR := area_pentagon A B P Q R

  area_ΔCPQ / area_⭐ABPQR = (3 / 28)

theorem parallelogram_area_ratio :
  parallelogram_problem := by 
  sorry

end parallelogram_area_ratio_l608_608527


namespace circumcenter_lies_on_ak_l608_608838

noncomputable def triangle_circumcenter_lies_on_ak
  {α β γ : ℝ}
  (A B C L H K O : Type*)
  [triangle A B C]
  [angle_bisector A L]
  [height_from B H]
  [circumcircle_of_triangle A (B ∧ L) K]
  [circumcenter_of_triangle A B C O]
  : Prop :=
  lies_on_line O (line_through A K)

-- We'll add the assumptions as hypotheses to the lemma
theorem circumcenter_lies_on_ak 
  {α β γ : ℝ} {A B C L H K O : Type*}
  [triangle A B C]
  [angle_bisector A L]
  [height_from B H]
  [circumcircle_of_triangle A (B ∧ L) K]
  [circumcenter_of_triangle A B C O]
  : lies_on_line O (line_through A K) :=
sorry

end circumcenter_lies_on_ak_l608_608838


namespace incorrect_deductive_reasoning_l608_608092

theorem incorrect_deductive_reasoning :
  (∃ x, x ∈ ℚ ∧ x.is_fraction) →
  (∀ y, y ∈ ℤ → y ∈ ℚ) →
  (∃ z, z ∈ ℤ ∧ z.is_fraction) → False :=
by
  sorry

end incorrect_deductive_reasoning_l608_608092


namespace new_average_l608_608733

def original_average : ℕ := 12
def num_numbers : ℕ := 12
def sum_of_numbers : ℕ := original_average * num_numbers
def first_group_count : ℕ := 4
def first_group_add : ℕ := 5
def second_group_count : ℕ := 3
def second_group_add : ℕ := 3
def third_group_count : ℕ := 5
def third_group_add : ℕ := 1

theorem new_average : (sum_of_numbers + 
                      (first_group_count * first_group_add) + 
                      (second_group_count * second_group_add) + 
                      (third_group_count * third_group_add)) / num_numbers = 14.83 := 
by 
  sorry

end new_average_l608_608733


namespace solution_correct_l608_608464

-- Problem 1
def log_inequality_solution (x : ℝ) : Prop :=
  log (1 / 2) (x + 2) > -3 ↔ -2 < x ∧ x < 6

-- Problem 2
def complex_expression_value : ℝ :=
  (1 / 8)^(1 / 3) * (-7 / 6)^0 + 8^0.25 * (2^(1 / 4)) + ((2^(1 / 3) * 3^(1 / 2))^6)

theorem solution_correct :
    (∀ x, log_inequality_solution x) ∧ complex_expression_value = 221 / 2 := 
by
  sorry

end solution_correct_l608_608464


namespace remove_number_from_each_vertex_l608_608017

open Function

theorem remove_number_from_each_vertex
  {n : ℕ} (h : n > 0) 
  (distinct_numbers : ∀ i (hi : i < 2 * n), ∃ x y : ℝ, x ≠ y) : 
  ∃ (remaining_numbers : Fin (2 * n) → ℝ), 
  (∀ i j : Fin (2 * n), i = j + 1 ∨ i = j - 1 → remaining_numbers i ≠ remaining_numbers j) := 
by
  sorry

end remove_number_from_each_vertex_l608_608017


namespace problem1_problem2_l608_608046

noncomputable def expression1 : ℝ :=
  (0.064: ℝ) ^ (-1/3: ℝ) - (-7/8: ℝ) ^ (0: ℝ) + ((-2: ℝ) ^ 3) ^ (-4/3: ℝ) + (16: ℝ) ^ (-0.25: ℝ)

theorem problem1 : expression1 = 33 / 16 := by
  sorry

noncomputable def expression2 : ℝ :=
  abs ((4 / 9: ℝ) ^ (-1/2) - real.log 5) + real.sqrt ((real.log 2) ^ 2 - real.log 4 + 1) - 3 ^ (1 - real.log 3 2)

theorem problem2 : expression2 = 0 := by
  sorry

end problem1_problem2_l608_608046


namespace arithmetic_sequence_common_difference_l608_608295

theorem arithmetic_sequence_common_difference (k b : ℝ) (n : ℕ) :
  ∃ d : ℝ, ∀ n : ℕ, (a : ℕ → ℝ) (a_n = kn + b) -> (a (n + 1) - a n = d) :=
by
  sorry

end arithmetic_sequence_common_difference_l608_608295


namespace B_days_to_complete_work_l608_608055

-- Given conditions
def work_in_days_A := 15
def work_in_days_A_fraction := (1 : ℝ) / (work_in_days_A : ℝ)

def fraction_left_after_8_days := 0.06666666666666665
def fraction_completed_after_8_days := 1 - fraction_left_after_8_days

def fraction_completed_A_B_in_8_days (x : ℝ) := 8 * (work_in_days_A_fraction + (1 : ℝ) / x)

theorem B_days_to_complete_work :
  ∃ (x : ℝ), fraction_completed_A_B_in_8_days x = fraction_completed_after_8_days ∧ x = 20 :=
by
  sorry

end B_days_to_complete_work_l608_608055


namespace x_minus_y_eq_neg3_l608_608258

theorem x_minus_y_eq_neg3 (x y : ℝ) (i : ℂ) (h1 : x * i + 2 = y - i) (h2 : i^2 = -1) : x - y = -3 := 
  sorry

end x_minus_y_eq_neg3_l608_608258


namespace periodic_decimal_for_fraction_l608_608921

theorem periodic_decimal_for_fraction (n : ℕ) (a₁ a₂ : ℕ) :
  n = 22 → a₁ = 3 → a₂ = 6 → (∃ (d : ℕ → ℕ), 
  d 0 = 1 ∧ d 1 = a₁ ∧ d 2 = a₂ ∧ (∀ m : ℕ, d (m + 2) = d m) ∧
  (0 + d 0/10 + d 1/100 + d 2/1000 + ... = 3 / n)) :=
begin
  sorry
end

end periodic_decimal_for_fraction_l608_608921


namespace pure_imaginary_m_eq_2_l608_608951

noncomputable def complex_number {m : ℝ} : Complex := Complex.mk (m^2 - 4) (m + 2)

theorem pure_imaginary_m_eq_2 (m : ℝ) :
    (complex_number.im = 0) ∧ (complex_number.re ≠ 0)  → m = 2 :=
by 
    intro h,
    sorry

end pure_imaginary_m_eq_2_l608_608951


namespace sum_of_columns_less_than_1035_l608_608849

variable {A : Type*} [AddCommMonoid A] [PartialOrder A] [Zero A]

def sum_less_than_1035 
  (table : Fin 8 × Fin 8 → A) 
  (sum_table : (∑ i j, table (i, j) = 1956)) 
  (sum_diag : (∑ i, table (i, i) = 112))
  (symmetry : ∀ i j, table (i, j) = table (j, i)) 
  : Prop := ∀ j, (∑ i, table (i, j)) < 1035

theorem sum_of_columns_less_than_1035 
  {table : Fin 8 × Fin 8 → ℕ} 
  (sum_table : (∑ i j, table (i, j) = 1956)) 
  (sum_diag : (∑ i, table (i, i) = 112))
  (symmetry : ∀ i j, table (i, j) = table (j, i)) 
  : sum_less_than_1035 table sum_table sum_diag symmetry :=
sorry

end sum_of_columns_less_than_1035_l608_608849


namespace PC_value_l608_608629

noncomputable def find_PC {A B C P : Point} (AB BC CA : ℝ) (h : Similar (Triangle P A B) (Triangle P C A)) : ℝ :=
  let PC := 27 in
    if AB = 10 ∧ BC = 9 ∧ CA = 7.5 then PC else 0

theorem PC_value : ∀ (A B C P : Point) (AB BC CA : ℝ),
  AB = 10 → BC = 9 → CA = 7.5 →
  Similar (Triangle P A B) (Triangle P C A) →
  find_PC AB BC CA (Similar (Triangle P A B) (Triangle P C A)) = 27 :=
by
  intros
  sorry

end PC_value_l608_608629


namespace center_of_circumcircle_lies_on_AK_l608_608842

variable {α β γ : Real} -- Angles in triangle ABC
variable (A B C L H K O : Point) -- Points in the configuration
variable (circumcircle_ABC : TriangularCircumcircle A B C) -- Circumcircle of triangle ABC

-- Definitions based on the given conditions
variable (is_angle_bisector : angle_bisector A B C L)
variable (is_height : height_from_point_to_line B A L H)
variable (intersects_circle_at_K : intersects_circumcircle A B L K circumcircle_ABC)
variable (is_circumcenter : circumcenter A B C O circumcircle_ABC)

theorem center_of_circumcircle_lies_on_AK
  (h_angle_bisector : is_angle_bisector)
  (h_height : is_height)
  (h_intersects_circle_at_K : intersects_circle_at_K)
  (h_circumcenter : is_circumcenter) 
    : lies_on_line O A K := 
sorry -- Proof is omitted

end center_of_circumcircle_lies_on_AK_l608_608842


namespace question_1_question_2_question_3_question_4_l608_608818

theorem question_1 (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) (h : (1 - x)^7 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7) :
  ¬ (a₂ + a₄ + a₆ = 64) :=
sorry

theorem question_2 (x : ℝ) :
  ¬ (smallest_term_coefficient (1 - x)^7 = fifth_term) :=
sorry

theorem question_3 :
  (1 - 100)^7 % 1000 = 301 :=
sorry

theorem question_4 (x : ℝ) :
  coefficient (x^5) ((1-x) + (1-x)^2 + (1-x)^3 + (1-x)^4 + (1-x)^5 + (1-x)^6 + (1-x)^7) = -28 :=
sorry

end question_1_question_2_question_3_question_4_l608_608818


namespace speed_with_16_coaches_l608_608868

def initial_speed : ℝ := 30
def speed_reduction (k : ℝ) (n : ℝ) : ℝ := k * real.sqrt n
def speed (s₀ k n : ℝ) : ℝ := s₀ - speed_reduction k n

theorem speed_with_16_coaches :
  ∃ (k n : ℝ), speed 30 k n = 18 ∧ speed 30 k 16 = 14 :=
by
  use 4, 9
  have h1 : speed 30 4 9 = 18 := by norm_num
  have h2 : speed 30 4 16 = 14 := by norm_num
  exact ⟨h1, h2⟩

end speed_with_16_coaches_l608_608868


namespace isosceles_triangle_largest_angle_l608_608153

/-- 
  Given an isosceles triangle where one of the angles is 20% smaller than a right angle,
  prove that the measure of one of the two largest angles is 54 degrees.
-/
theorem isosceles_triangle_largest_angle 
  (A B C : ℝ) 
  (triangle_ABC : A + B + C = 180)
  (isosceles_triangle : A = B ∨ A = C ∨ B = C)
  (smaller_angle : A = 0.80 * 90) :
  A = 54 ∨ B = 54 ∨ C = 54 :=
sorry

end isosceles_triangle_largest_angle_l608_608153


namespace max_distance_on_circle_l608_608585
-- Import the Mathlib library for all necessary mathematical tools and definitions

-- Define a theorem stating the maximum distance from the origin O to any point M on the circle C2
theorem max_distance_on_circle
  (x y : ℝ)
  (h : x^2 + y^2 + 4 * x - 4 * y = 0) :
  ∃ M, M = (x, y) ∧ ‖(x, y) - (0, 0)‖ = 4 * real.sqrt 2 := sorry

end max_distance_on_circle_l608_608585


namespace find_number_of_ducks_l608_608282

variable {D H : ℕ}

-- Definition of the conditions
def total_animals (D H : ℕ) : Prop := D + H = 11
def total_legs (D H : ℕ) : Prop := 2 * D + 4 * H = 30
def number_of_ducks (D : ℕ) : Prop := D = 7

-- Lean statement for the proof problem
theorem find_number_of_ducks (D H : ℕ) (h1 : total_animals D H) (h2 : total_legs D H) : number_of_ducks D :=
by
  sorry

end find_number_of_ducks_l608_608282


namespace price_difference_in_cents_l608_608891

def list_price : ℝ := 50.50
def deal_direct_discount : ℝ := 10.50
def bargain_base_discount_rate : ℝ := 0.20
def tax_rate : ℝ := 0.05

theorem price_difference_in_cents :
  let deal_direct_price := list_price - deal_direct_discount in
  let discounted_price := list_price * (1 - bargain_base_discount_rate) in
  let bargain_base_price := discounted_price * (1 + tax_rate) in
  let price_difference := bargain_base_price - deal_direct_price in
  price_difference * 100 = 242 :=
by
  sorry

end price_difference_in_cents_l608_608891


namespace division_multiplication_order_l608_608467

theorem division_multiplication_order (a b c : ℝ) (h₁ : b ≠ 0) (h₂ : c ≠ 0) :
  (a / b) * c = a / (b * c) ↔ (a / b) * c ≠ a :=
by sorry

example : ¬ (32 / 0.25 * 4 = 32 / (0.25 * 4)) :=
begin
  have h1 : 0.25 ≠ 0 := by norm_num,
  have h2 : 4 ≠ 0 := by norm_num,
  simp [division_multiplication_order 32 0.25 4 h1 h2]
end

end division_multiplication_order_l608_608467


namespace moses_more_than_esther_l608_608798

theorem moses_more_than_esther (total_amount: ℝ) (moses_share: ℝ) (tony_esther_share: ℝ) :
  total_amount = 50 → moses_share = 0.40 * total_amount → 
  tony_esther_share = (total_amount - moses_share) / 2 → 
  moses_share - tony_esther_share = 5 :=
by
  intros h1 h2 h3
  sorry

end moses_more_than_esther_l608_608798


namespace Mariela_cards_correct_l608_608378

structure HospitalCards :=
  (total : ℕ)
  (handwritten : ℕ)
  (multilingual : ℕ)
  (multiplePages : ℕ)

structure HomeCards :=
  (total : ℕ)
  (handwritten : ℕ)
  (multilingual : ℕ)
  (multiplePages : ℕ)

def MarielaCards (hospital : HospitalCards) (home : HomeCards) :=
  hospital.handwritten + home.handwritten = 273 ∧
  hospital.multilingual + home.multilingual = 164 ∧
  hospital.multiplePages + home.multiplePages = 253 ∧
  hospital.total + home.total = 690

constant hospital : HospitalCards := {total := 403, handwritten := 152, multilingual := 98, multiplePages := 153}
constant home : HomeCards := {total := 287, handwritten := 121, multilingual := 66, multiplePages := 100}

theorem Mariela_cards_correct : MarielaCards hospital home := by
  sorry

end Mariela_cards_correct_l608_608378


namespace probability_of_A_chosen_l608_608882

-- Define the set of persons
inductive Person
| A : Person
| B : Person
| C : Person

open Person

-- Function to determine the selections
def selection (p : Person) : Finset (Finset Person) :=
if p = A then { {A, B}, {A, C} } else if p = B then { {B, A}, {B, C} } else { {C, A}, {C, B} }

-- Prove the probability of A being chosen when selecting two representatives from three people
theorem probability_of_A_chosen : (2 : ℚ) / 3 = 2 / 3 :=
by
  sorry

end probability_of_A_chosen_l608_608882


namespace people_in_room_after_2019_minutes_l608_608644

theorem people_in_room_after_2019_minutes :
  ∀ (P : Nat → Int), 
    P 0 = 0 -> 
    (∀ t, P (t+1) = P t + 2 ∨ P (t+1) = P t - 1) -> 
    P 2019 ≠ 2018 :=
by
  intros P hP0 hP_changes
  sorry

end people_in_room_after_2019_minutes_l608_608644


namespace pool_volume_800pi_l608_608856

noncomputable def swimming_pool_volume (d : ℝ) (h1 : ℝ) (h2 : ℝ) : ℝ :=
  let r := d / 2
  let V1 := π * r^2 * h1
  let V2 := π * r^2 * h2
  V1 + V2

theorem pool_volume_800pi :
  swimming_pool_volume 20 3 5 = 800 * π :=
by
  sorry

end pool_volume_800pi_l608_608856


namespace cos_alpha_sin_alpha_plus_pi_over_4_l608_608924

noncomputable def sin_alpha : ℝ := 3 / 5
noncomputable def alpha_range := 0 < α ∧ α < π / 2

theorem cos_alpha (h1: sin α = sin_alpha) (h2: alpha_range) : cos α = 4 / 5 :=
by
sorry

theorem sin_alpha_plus_pi_over_4 (h1: sin α = sin_alpha) (h2: alpha_range) :
  sin (α + π / 4) = 7 * sqrt 2 / 10 :=
by
sorry

end cos_alpha_sin_alpha_plus_pi_over_4_l608_608924


namespace dart_lands_in_central_hexagon_l608_608482

-- Definitions for the problem conditions
def side_length (s : ℝ) : ℝ := s

def area_hexagon (a : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * (a ^ 2)

def large_hexagon_area (s : ℝ) : ℝ := area_hexagon s

def small_hexagon_side (s : ℝ) : ℝ := s / 2

def small_hexagon_area (s : ℝ) : ℝ := area_hexagon (small_hexagon_side s)

-- The core theorem stating the conclusion to be proved
theorem dart_lands_in_central_hexagon (s : ℝ) : 
  (small_hexagon_area s) / (large_hexagon_area s) = 1 / 4 :=
by
  sorry

end dart_lands_in_central_hexagon_l608_608482


namespace polynomial_eval_at_one_l608_608008

theorem polynomial_eval_at_one :
  ∃ Q : ℚ[X], 
  (degree Q = 4) ∧ 
  (leading_coeff Q = 1) ∧ 
  (Q.eval (1 + real.sqrt 2) = 0) ∧ 
  (Q.eval 1 = -5) :=
begin
  sorry
end

end polynomial_eval_at_one_l608_608008


namespace _l608_608686

open EuclideanGeometry

variables {A B C O M D H : Point} {ω : Circle}

noncomputable def main_theorem (h₁ : IsNonIsoscelesNonRightTriangle A B C)
                       (h₂ : Circumcircle ω A B C)
                       (h₃ : Circumcenter O A B C)
                       (h₄ : Midpoint M B C)
                       (h₅ : CircleIntersection (circumcircle_of_triangle A O M) ω D)
                       (h₆ : Orthocenter H A B C) :
  ∠ D A H = ∠ M A O :=
by
  sorry

end _l608_608686


namespace length_increase_percentage_l608_608414

theorem length_increase_percentage
  (L W : ℝ)
  (A : ℝ := L * W)
  (A' : ℝ := 1.30000000000000004 * A)
  (new_length : ℝ := L * (1 + x / 100))
  (new_width : ℝ := W / 2)
  (area_equiv : new_length * new_width = A')
  (x : ℝ) :
  1 + x / 100 = 2.60000000000000008 :=
by
  -- Proof goes here
  sorry

end length_increase_percentage_l608_608414


namespace sufficient_and_necessary_condition_l608_608566

theorem sufficient_and_necessary_condition (a : ℝ) : 
  (0 < a ∧ a < 4) ↔ ∀ x : ℝ, (x^2 - a * x + a) > 0 :=
by sorry

end sufficient_and_necessary_condition_l608_608566


namespace number_of_clothes_hangers_l608_608090

noncomputable def total_money : ℝ := 60
noncomputable def spent_on_tissues : ℝ := 34.8
noncomputable def price_per_hanger : ℝ := 1.6

theorem number_of_clothes_hangers : 
  let remaining_money := total_money - spent_on_tissues
  let hangers := remaining_money / price_per_hanger
  Int.floor hangers = 15 := 
by
  sorry

end number_of_clothes_hangers_l608_608090


namespace find_b_l608_608335

open Real

variables {A B C a b c : ℝ}

theorem find_b 
  (hA : A = π / 4) 
  (h1 : 2 * b * sin B - c * sin C = 2 * a * sin A) 
  (h_area : 1 / 2 * b * c * sin A = 3) : 
  b = 3 := 
sorry

end find_b_l608_608335


namespace parallelogram_area_l608_608493

theorem parallelogram_area 
  (b h : ℝ)
  (h_eq_2b : h = 2 * b)
  (b_eq_10 : b = 10) :
  let A := b * h in A = 200 := by
  sorry

end parallelogram_area_l608_608493


namespace find_n_l608_608681

noncomputable def arithmetic_sum (a d : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a 0 + (n - 1) * d 0)) / 2

theorem find_n 
  (a : ℕ → ℕ) (d : ℕ → ℕ) 
  (S : ℕ → ℕ)
  (h1 : S 6 = 36)
  (h2 : S 18 = 324)
  (h3 : S (18 - 6) = 144) : 
  ∃ n : ℕ, n = 18 :=
begin
  sorry
end

end find_n_l608_608681


namespace alternating_series_sum_l608_608449

theorem alternating_series_sum : 
  (∑ i in (Finset.range 1000), (-1)^(i+1) * (i + 1)) = -500 := by
  sorry

end alternating_series_sum_l608_608449


namespace quad_root_magnitude_l608_608918

theorem quad_root_magnitude (m : ℝ) :
  (∃ x : ℝ, x^2 - x + m^2 - 4 = 0 ∧ x = 1) → m = 2 ∨ m = -2 :=
by
  sorry

end quad_root_magnitude_l608_608918


namespace fill_tank_time_l608_608039

/-- 
If pipe A fills a tank in 30 minutes, pipe B fills the same tank in 20 minutes, 
and pipe C empties it in 40 minutes, then the time it takes to fill the tank 
when all three pipes are working together is 120/7 minutes.
-/
theorem fill_tank_time 
  (rate_A : ℝ) (rate_B : ℝ) (rate_C : ℝ) (combined_rate : ℝ) (T : ℝ) :
  rate_A = 1/30 ∧ rate_B = 1/20 ∧ rate_C = -1/40 ∧ combined_rate = rate_A + rate_B + rate_C
  → T = 1 / combined_rate
  → T = 120 / 7 :=
by
  intros
  sorry

end fill_tank_time_l608_608039
