import Mathlib
import Mathlib.Algebra.Arithmetic
import Mathlib.Algebra.Basic
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.LinearEquation
import Mathlib.Algebra.Parabola
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Quadratics
import Mathlib.Algebra.Ring.Basic
import Mathlib.Algebra.Systems
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Combinatorics
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Gcd
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sign
import Mathlib.Data.Set.Basic
import Mathlib.GroupTheory.Group
import Mathlib.Logic.Basic
import Mathlib.Probability.Basic
import Mathlib.Probability.Independence
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Trigonometry.Basic
import Probability

namespace curve_is_two_rays_l795_795126

-- The parametric equations are given as follows:
def parametric_x (t : ℝ) : ℝ := t + (1/t)
def parametric_y : ℝ := -2

theorem curve_is_two_rays (t : ℝ) : 
  (parametric_x t ≤ -2 ∨ parametric_x t ≥ 2) ∧ (parametric_y = -2) :=
by
  sorry

end curve_is_two_rays_l795_795126


namespace probability_less_than_or_equal_nine_l795_795005

-- Definition corresponding to the conditions
def cardSet : Set ℕ := {1, 3, 4, 6, 7, 9}

-- Problem Statement to prove
theorem probability_less_than_or_equal_nine :
  (cardSet.filter (λ n => n ≤ 9)).card / cardSet.card = 1 := by
  sorry

end probability_less_than_or_equal_nine_l795_795005


namespace total_blue_balloons_l795_795451

theorem total_blue_balloons (joan_balloons : ℕ) (melanie_balloons : ℕ) (h_joan : joan_balloons = 40) (h_melanie : melanie_balloons = 41) : joan_balloons + melanie_balloons = 81 := by
  sorry

end total_blue_balloons_l795_795451


namespace polygon_has_7_sides_l795_795491

theorem polygon_has_7_sides (t : ℕ) (n : ℕ) : t = 5 → n = 7 :=
by
  assume ht : t = 5,
  have hn : n = 7 := sorry,
  exact hn

end polygon_has_7_sides_l795_795491


namespace melies_meat_purchase_l795_795486

-- Define the relevant variables and conditions
variable (initial_amount : ℕ) (amount_left : ℕ) (cost_per_kg : ℕ)

-- State the main theorem we want to prove
theorem melies_meat_purchase (h1 : initial_amount = 180) (h2 : amount_left = 16) (h3 : cost_per_kg = 82) :
  (initial_amount - amount_left) / cost_per_kg = 2 := by
  sorry

end melies_meat_purchase_l795_795486


namespace algebraic_expression_correct_l795_795715

variable (x y : ℤ)

theorem algebraic_expression_correct (h : (x - y) / (x + y) = 3) : (2 * (x - y)) / (x + y) - (x + y) / (3 * (x - y)) = 53 / 9 := 
by  
  sorry

end algebraic_expression_correct_l795_795715


namespace oldest_child_age_l795_795524

def avg_age (ages : List ℕ) : ℕ := (ages.sum / ages.length)

theorem oldest_child_age :
  ∀ (ages : List ℕ), 
    (avg_age ages = 10) →
    (∀ (x y : ℕ), x ∈ ages → y ∈ ages → x ≠ y) →
    (∀ (i : ℕ), i < 6 → ages.nth i + 1 = ages.nth (i + 1)) →
    ages.length = 7 →
    ages.last = 13 := 
by
  sorry

end oldest_child_age_l795_795524


namespace sum_of_coefficients_l795_795517

theorem sum_of_coefficients (a b c d : ℝ) (f : ℝ → ℝ)
    (h1 : ∀ x, f (x + 2) = 2*x^3 + 5*x^2 + 3*x + 6)
    (h2 : ∀ x, f x = a*x^3 + b*x^2 + c*x + d) :
  a + b + c + d = 6 :=
by sorry

end sum_of_coefficients_l795_795517


namespace christina_money_greater_fabien_money_l795_795116

theorem christina_money_greater_fabien_money :
  (let euro_to_dollar := 1.5;
       christina_dollars := 750;
       fabien_euros := 450;
       fabien_dollars := fabien_euros * euro_to_dollar in
   (christina_dollars - fabien_dollars) / fabien_dollars * 100 = 11.11) :=
by
  let euro_to_dollar := 1.5
  let christina_dollars := 750
  let fabien_euros := 450
  let fabien_dollars := fabien_euros * euro_to_dollar
  have h : ((christina_dollars - fabien_dollars) / fabien_dollars * 100 = 11.11) := sorry
  exact h

end christina_money_greater_fabien_money_l795_795116


namespace ratio_correct_l795_795612

def cost_of_flasks := 150
def remaining_budget := 25
def total_budget := 325
def spent_budget := total_budget - remaining_budget
def cost_of_test_tubes := 100
def cost_of_safety_gear := cost_of_test_tubes / 2
def ratio_test_tubes_flasks := cost_of_test_tubes / cost_of_flasks

theorem ratio_correct :
  spent_budget = cost_of_flasks + cost_of_test_tubes + cost_of_safety_gear → 
  ratio_test_tubes_flasks = 2 / 3 :=
by
  sorry

end ratio_correct_l795_795612


namespace roots_quadratic_eq_value_l795_795067

theorem roots_quadratic_eq_value (d e : ℝ) (h : 3 * d^2 + 4 * d - 7 = 0) (h' : 3 * e^2 + 4 * e - 7 = 0) : 
  (d - 2) * (e - 2) = 13 / 3 := 
by
  sorry

end roots_quadratic_eq_value_l795_795067


namespace number_not_perfect_square_l795_795139

theorem number_not_perfect_square 
  (n : ℕ)
  (h1 : nat.digits 10 n = 1994) 
  (h2 : count n 0 = 14)
  (h3 : ∃ x, count n 1 = x ∧ count n 2 = 2*x ∧ count n 3 = 3*x ∧ count n 4 = 4*x ∧ count n 5 = 5*x ∧ count n 6 = 6*x ∧ count n 7 = 7*x ∧ count n 8 = 8*x ∧ count n 9 = 9*x) 
  : ¬ ∃ k, k * k = n := 
sorry

end number_not_perfect_square_l795_795139


namespace parabola_distance_l795_795346

theorem parabola_distance (p : ℝ) (hp : 0 < p) (hf : ∀ P : ℝ × ℝ, P ∈ {Q : ℝ × ℝ | Q.1^2 = 2 * p * Q.2} →
  dist P (0, p / 2) = 16) (hx : ∀ P : ℝ × ℝ, P ∈ {Q : ℝ × ℝ | Q.1^2 = 2 * p * Q.2} →
  P.2 = 10) : p = 12 :=
sorry

end parabola_distance_l795_795346


namespace abs_diff_of_roots_l795_795284

theorem abs_diff_of_roots : 
  ∀ r1 r2 : ℝ, 
  (r1 + r2 = 7) ∧ (r1 * r2 = 12) → abs (r1 - r2) = 1 :=
by
  -- Assume the roots are r1 and r2
  intros r1 r2 H,
  -- Decompose the assumption H into its components
  cases H with Hsum Hprod,
  -- Calculate the square of the difference using the given identities
  have H_squared_diff : (r1 - r2)^2 = (r1 + r2)^2 - 4 * (r1 * r2),
  { sorry },
  -- Substitute the known values to find the square of the difference
  have H_squared_vals : (r1 - r2)^2 = 49 - 4 * 12,
  { sorry },
  -- Simplify to get (r1 - r2)^2 = 1
  have H1 : (r1 - r2)^2 = 1,
  { sorry },
  -- The absolute value of the difference is the square root of this result
  have abs_diff : abs (r1 - r2) = 1,
  { sorry },
  -- Conclude the proof by showing the final result matches the expected answer
  exact abs_diff

end abs_diff_of_roots_l795_795284


namespace average_of_numbers_l795_795122

theorem average_of_numbers (x : ℚ) : 
  let sum := 4950 
  let n := 99 + 1 
  (sum + x) / n = 100 * x -> 
  x = 50 / 101 := 
by
  intro h
  have h_eq : (4950 + x) / 100 = 100 * x := h
  -- Proof needed here
  sorry

end average_of_numbers_l795_795122


namespace gasoline_price_percent_increase_l795_795878

theorem gasoline_price_percent_increase 
  (highest_price : ℕ) (lowest_price : ℕ) 
  (h_highest : highest_price = 17) 
  (h_lowest : lowest_price = 10) : 
  (highest_price - lowest_price) * 100 / lowest_price = 70 := 
by 
  sorry

end gasoline_price_percent_increase_l795_795878


namespace a_seq_values_geometric_sequence_a_n_minus_2_a_seq_general_term_and_sum_l795_795376

noncomputable def a_seq : ℕ → ℝ
| 1     := 1
| (n+1) := (1 / 2) * a_seq n + 1

theorem a_seq_values :
  a_seq 2 = 3 / 2 ∧
  a_seq 3 = 7 / 4 ∧
  a_seq 4 = 15 / 8 :=
by { sorry }

theorem geometric_sequence_a_n_minus_2 :
  ∃ r : ℝ, ∀ n ≥ 2, (a_seq n - 2) = (a_seq 1 - 2) * r^(n-1) ∧ r = 1 / 2 :=
by { sorry }

theorem a_seq_general_term_and_sum (n : ℕ) :
  ∃ a_n S_n : ℝ,
  a_n = 2 - (1 / 2)^(n-1) ∧
  S_n = ∑ i in finset.range n, a_seq (i+1) ∧
  S_n = 2 * n - 2 + (1 / 2)^(n-1) :=
by { sorry }

end a_seq_values_geometric_sequence_a_n_minus_2_a_seq_general_term_and_sum_l795_795376


namespace triangle_inequality_difference_l795_795445

theorem triangle_inequality_difference :
  (∀ (x : ℤ), (x + 7 > 9) ∧ (x + 9 > 7) ∧ (7 + 9 > x) → (3 ≤ x ∧ x ≤ 15) ∧ (15 - 3 = 12)) :=
by
  sorry

end triangle_inequality_difference_l795_795445


namespace max_valid_selections_l795_795148

theorem max_valid_selections (bags beads_per_bag : ℕ) 
  (total_weight_per_bag : ℝ) (h_bags : bags = 2019) 
  (h_beads_per_bag : beads_per_bag = 2019) 
  (h_weight : total_weight_per_bag = 1) : 
  ∃ k, k = Nat.factorial 2018 := by
  use Nat.factorial 2018
  rfl

end max_valid_selections_l795_795148


namespace mean_value_of_quadrilateral_angles_l795_795964

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795964


namespace right_triangle_area_l795_795593

theorem right_triangle_area (b c : ℕ) (h : ℕ) :
  b = 8 →
  c = 10 →
  h = 6 →
  (1 / 2 : ℝ) * b * h = 24 :=
by
  intros hb hc hh
  simp [hb, hc, hh]
  norm_num
  sorry

end right_triangle_area_l795_795593


namespace five_digit_numbers_count_l795_795575

open Nat List

noncomputable def count_valid_numbers : Nat :=
  let digits := [0, 1, 2, 3, 4, 5]
  let first_digit_constraints := {d : Nat | d = 4 ∨ d = 5}
  let last_digit_constraints := {d : Nat | d = 0 ∨ d = 2 ∨ d = 4}
  
  -- Calculate all possible 5-digit numbers
  let all_valid_numbers := digits.filter (λ d => d ∈ first_digit_constraints).bind (λ first =>
                         digits.filter (λ d => d ≠ first).filter (λ d => d ∈ last_digit_constraints).bind (λ last =>
                         permutations (digits.filter (λ d => d ≠ first ∧ d ≠ last)).bind (λ rest =>
                         zipWith (++) [[first] ++ rest.dropLast ++ [last]] perm (permutations (dropLast rest))
                     )))
  
  let filtered_numbers := all_valid_numbers.filter (λ n => n.length = 5 ∧ (n.headD 0 = 4 ∨ n.headD 0 = 5) ∧ (last n = 0 ∨ last n = 2 ∨ last n = 4))
  
  filtered_numbers.length

theorem five_digit_numbers_count : count_valid_numbers = 120 :=
by sorry

end five_digit_numbers_count_l795_795575


namespace complex_number_condition_l795_795296

noncomputable def num_complex_solutions : ℕ := 35280

theorem complex_number_condition (z : ℂ) (hz : complex.abs z = 1) : 
  z ^ (nat.factorial 8) - z ^ (nat.factorial 7) ∈ ℝ ↔ num_complex_solutions = 35280 :=
by sorry

end complex_number_condition_l795_795296


namespace no_arithmetic_progression_in_squares_l795_795495

theorem no_arithmetic_progression_in_squares :
  ∀ (a d : ℕ), d > 0 → ¬ (∃ (f : ℕ → ℕ), 
    (∀ n, f n = a + n * d) ∧ 
    (∀ n, ∃ m, n ^ 2 = f m)) :=
by
  sorry

end no_arithmetic_progression_in_squares_l795_795495


namespace figure_C_perimeter_l795_795885

def is_perimeter (figure : Type) (perimeter : ℕ) : Prop :=
∃ x y : ℕ, (figure = 'A' → 6*x + 2*y = perimeter) ∧ 
           (figure = 'B' → 4*x + 6*y = perimeter) ∧
           (figure = 'C' → 2*x + 6*y = perimeter)

theorem figure_C_perimeter (hA : is_perimeter 'A' 56) (hB : is_perimeter 'B' 56) : 
  is_perimeter 'C' 40 :=
by
  sorry

end figure_C_perimeter_l795_795885


namespace parabola_properties_l795_795441

-- Problem statement
theorem parabola_properties (a c : ℝ) (A B : ℝ × ℝ) (l : ℝ → ℝ) :
  (c = -2 * a) ∧ 
  (A = (1, 0)) ∧
  (B = (-1 / 2, -9 / 4 * a)) ∧ 
  (l = λ x, 2 * x - 2) ∧
  (a < c) ∧ 
  (a < 0) →
  (c = -2 * a) ∧ 
  (B.1 < 0 ∧ B.2 < 0) ∧ 
  (0 < ∀ y, -a * x^2 + (a^2 - c + 2a) ≤ y) :=
begin
  sorry
end

end parabola_properties_l795_795441


namespace gcd_factorial_l795_795313

-- Definitions and conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- Theorem statement
theorem gcd_factorial : gcd (factorial 8) (factorial 10) = factorial 8 := by
  -- The proof is omitted
  sorry

end gcd_factorial_l795_795313


namespace distance_function_intersection_point_number_of_points_l795_795722

open Real

-- Define the ellipse
def ellipse (x y : ℝ) (b : ℝ) : Prop := (x^2 / 9 + y^2 / b^2 = 1)

-- Define the distance function|PF₁|
noncomputable def distancePF1 (x0 y0 : ℝ) : ℝ := sqrt ((x0+2)^2 + y0^2)

-- Prove distance function based on x0 under the condition b = √5
theorem distance_function (x0 : ℝ) (h : ellipse x0 (sqrt (5 - x0^2 / 9) * sqrt 5) (sqrt 5)) :
  distancePF1 x0 (sqrt (5 - x0^2 / 9) * sqrt 5) = (2 / 3) * x0 + 3 :=
  sorry

-- Intersection point N and its relationship with b
theorem intersection_point (k b : ℝ) (h : b > 0) (n : ℝ) :
  (2 * (36 - 9 * b^2) + (2 - n) * (-36 * k)) / (b^2 + 9 * k^2) = 0 → n = b^2 / 2 :=
  sorry 

-- Number of points P for which triangle F₁F₂P is an isosceles acute triangle
theorem number_of_points (b : ℝ) (h : b > 0) : (b ∈ Icc (0:ℝ) (3 * sqrt 2 / 2) → (n : ℕ) = 4) ∧
                                                  ((b > 3 * sqrt 2 / 2 ∧ b ≠ 3 * sqrt 3 / 2) ∨ (b < 3 * sqrt 2 * sqrt 2 - 2) → (n = 6)) ∧
                                                  (b = 3 * sqrt 3 / 2 ∨ (3 * sqrt 2 * sqrt 2 - 2 < b < 3) → (n = 2)) :=
  sorry

end distance_function_intersection_point_number_of_points_l795_795722


namespace mean_value_of_quadrilateral_angles_l795_795965

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795965


namespace product_lcm_gcd_l795_795297

theorem product_lcm_gcd (a b : ℕ) (h1 : nat.prime a) (h2 : b = 2^2 * 3) :
  let lcm_ab := nat.lcm a b in
  let gcd_ab := nat.gcd a b in
  lcm_ab * gcd_ab = 132 :=
by
  have ha : a = 11 := sorry  -- hypothesis that a is 11
  have hb : b = 12 := sorry  -- hypothesis that b is 12
  exact calc
    (nat.lcm a b) * (nat.gcd a b) = (11 * 12) * 1 : by rw [nat.lcm_eq_mul_div_gcd, nat.gcd_prime_prime_mul h1 (show 12 % 11 ≠ 0 from sorry)]
                            ... = 132 : by norm_num

end product_lcm_gcd_l795_795297


namespace prism_volume_l795_795781

noncomputable def point (x y z : ℝ) := (x, y, z)
noncomputable def midpoint (p1 p2 : (ℝ × ℝ × ℝ)) : (ℝ × ℝ × ℝ) :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  ( (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2 )

noncomputable def volume_of_prism (base_area height : ℝ) := base_area * height

def cube_side_length := 1
def P := midpoint (point 0 0 0) (point cube_side_length 0 0)
def Q := midpoint (point 0 0 0) (point 0 cube_side_length 0)
def R := midpoint (point 0 0 0) (point 0 0 cube_side_length)

-- Coordinates of P, Q, R following the conditions
def P_coord := (1 / 2, 0, 0)
def Q_coord := (0, 1 / 2, 0)
def R_coord := (0, 0, 1 / 2)

noncomputable def distance (p1 p2 : (ℝ × ℝ × ℝ)) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

noncomputable def area_triangle (p1 p2 p3 : (ℝ × ℝ × ℝ)) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  let (x3, y3, z3) := p3
  0.5 * real.sqrt (
    ((y2 - y1)*(z3 - z1) - (y3 - y1)*(z2 - z1))^2 +
    ((z2 - z1)*(x3 - x1) - (z3 - z1)*(x2 - x1))^2 +
    ((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))^2
  )

noncomputable def height_prism := distance R (point 0 0 (1 - 1/2))

theorem prism_volume : volume_of_prism (area_triangle P_coord Q_coord R_coord) height_prism = 3 / 16 :=
  sorry

end prism_volume_l795_795781


namespace trapezium_perimeters_l795_795940

theorem trapezium_perimeters (AB BC AD AF : ℝ)
  (h1 : AB = 30) (h2 : BC = 30) (h3 : AD = 25) (h4 : AF = 24) :
  ∃ p : ℝ, (p = 90 ∨ p = 104) :=
by
  sorry

end trapezium_perimeters_l795_795940


namespace geometric_series_sum_l795_795658

theorem geometric_series_sum : 
  let a := 2 in
  let r := 3 in
  let n := 8 in
  ∑ k in Finset.range n, a * r^k = 6560 :=
by 
  let a := 2
  let r := 3
  let n := 8
  have sum_formula : ∑ k in Finset.range n, a * r^k = a * (r^n - 1) / (r - 1) := sorry
  rw sum_formula
  calc
    a * (r^n - 1) / (r - 1) = 2 * (3^8 - 1) / (3 - 1) : by sorry
                            ... = 2 * 6560 / 2 : by sorry
                            ... = 6560 : by sorry

end geometric_series_sum_l795_795658


namespace Shiela_stars_per_bottle_l795_795106

theorem Shiela_stars_per_bottle (total_stars : ℕ) (total_classmates : ℕ) (h1 : total_stars = 45) (h2 : total_classmates = 9) :
  total_stars / total_classmates = 5 := 
by 
  sorry

end Shiela_stars_per_bottle_l795_795106


namespace perimeter_C_is_40_l795_795892

noncomputable def perimeter_of_figure_C (x y : ℝ) : ℝ :=
  2 * x + 6 * y

theorem perimeter_C_is_40 (x y : ℝ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  perimeter_of_figure_C x y = 40 :=
by
  -- Define initial conditions
  have eq1 : 3 * x + y = 28, by { rw [mul_assoc, mul_comm 3 x, add_assoc], exact (eq.div h1 2) }
  have eq2 : 2 * x + 3 * y = 28, by { rw [mul_assoc, mul_comm 2 x, add_assoc], exact (eq.div h2 2) }
  -- Assume the solutions are obtained from here
  have sol_x : x = 8, by sorry
  have sol_y : y = 4, by sorry
  -- Calculate the perimeter of figure C
  rw [perimeter_of_figure_C, sol_x, sol_y]
  norm_num
  trivial

-- Test case to ensure the code builds successfully
#eval perimeter_of_figure_C 8 4  -- Expected output: 40

end perimeter_C_is_40_l795_795892


namespace second_car_avg_mpg_l795_795426

theorem second_car_avg_mpg 
  (x y : ℝ) 
  (h1 : x + y = 75) 
  (h2 : 25 * x + 35 * y = 2275) : 
  y = 40 := 
by sorry

end second_car_avg_mpg_l795_795426


namespace hyperbola_asymptote_m_value_l795_795354

theorem hyperbola_asymptote_m_value
  (m : ℝ)
  (h1 : m > 0)
  (h2 : ∀ x y : ℝ, (5 * x - 2 * y = 0) → ((x^2 / 4) - (y^2 / m^2) = 1)) :
  m = 5 :=
sorry

end hyperbola_asymptote_m_value_l795_795354


namespace C_finishes_job_in_days_l795_795184

theorem C_finishes_job_in_days :
  ∀ (A B C : ℚ),
    (A + B = 1 / 15) →
    (A + B + C = 1 / 3) →
    1 / C = 3.75 :=
by
  intros A B C hab habc
  sorry

end C_finishes_job_in_days_l795_795184


namespace find_x_l795_795425

-- Define the conditions as given in the problem
def angle1 (x : ℝ) : ℝ := 6 * x
def angle2 (x : ℝ) : ℝ := 3 * x
def angle3 (x : ℝ) : ℝ := x
def angle4 (x : ℝ) : ℝ := 5 * x
def sum_of_angles (x : ℝ) : ℝ := angle1 x + angle2 x + angle3 x + angle4 x

-- State the problem: prove that x equals 24 given the sum of angles is 360 degrees
theorem find_x (x : ℝ) (h : sum_of_angles x = 360) : x = 24 :=
by
  sorry

end find_x_l795_795425


namespace solve_cubed_root_equation_l795_795111

theorem solve_cubed_root_equation :
  (∃ x : ℚ, (5 - 2 / x) ^ (1 / 3) = -3) ↔ x = 1 / 16 := 
by
  sorry

end solve_cubed_root_equation_l795_795111


namespace perimeter_C_l795_795899

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end perimeter_C_l795_795899


namespace min_value_x_y_l795_795013

theorem min_value_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9 / (x + 1) + 1 / (y + 1) = 1) :
  x + y ≥ 14 :=
sorry

end min_value_x_y_l795_795013


namespace total_weight_of_compound_l795_795577

variable (molecular_weight : ℕ) (moles : ℕ)

theorem total_weight_of_compound (h1 : molecular_weight = 72) (h2 : moles = 4) :
  moles * molecular_weight = 288 :=
by
  sorry

end total_weight_of_compound_l795_795577


namespace gcd_factorial_l795_795315

-- Definitions and conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- Theorem statement
theorem gcd_factorial : gcd (factorial 8) (factorial 10) = factorial 8 := by
  -- The proof is omitted
  sorry

end gcd_factorial_l795_795315


namespace Tn_lt_Sn_div_2_l795_795065

noncomputable def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
noncomputable def b (n : ℕ) : ℝ := n * (1 / 3)^n

noncomputable def S (n : ℕ) : ℝ := 
  (3 / 2) * (1 - (1 / 3)^n)

noncomputable def T (n : ℕ) : ℝ := 
  (3 / 4) * (1 - (1 / 3)^n) - (n / 2) * (1 / 3)^(n + 1)

theorem Tn_lt_Sn_div_2 (n : ℕ) : T n < S n / 2 := 
sorry

end Tn_lt_Sn_div_2_l795_795065


namespace trajectory_is_line_segment_l795_795464

noncomputable def F1 : ℝ × ℝ := (0, 0)  -- Assume some coordinates for F1
noncomputable def F2 : ℝ × ℝ := (6, 0)  -- Assume some coordinates for F2 based on the distance |F1F2| = 6

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def M_traj (M : ℝ × ℝ) : Prop := distance M F1 + distance M F2 = 6

theorem trajectory_is_line_segment : ∀ M : ℝ × ℝ, M_traj M → (M = F1 ∨ M = F2 ∨ 
  (M.1 = 3 ∧ 0 ≤ M.1 ∧ M.1 ≤ 6 ∧ M.2 = 0)) :=
by
  sorry

end trajectory_is_line_segment_l795_795464


namespace number_of_elements_in_M_l795_795835

open Set

noncomputable def M : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), 1 / Real.sqrt x - 1 / Real.sqrt y = 1 / Real.sqrt 45 ∧ x ∈ ℕ+ ∧ y ∈ ℕ+}

theorem number_of_elements_in_M : (finite M) ∧ (card M = 1) :=
  by
    sorry

end number_of_elements_in_M_l795_795835


namespace exists_subset_B_l795_795456

-- Defining the problem setup:
variable {A : Set ℤ} -- A is a set of integers

-- Conditions extracted from the problem:
axiom h1 : A.Card = 65
axiom h2 : ∀ (a ∈ A) (b ∈ A), a ≠ b → (a % 2016) ≠ (b % 2016)

-- Formulate the statement that needs to be proved:
theorem exists_subset_B (A : Set ℤ) (h1 : A.Card = 65) (h2 : ∀ (a b : ℤ), a ∈ A → b ∈ A → a ≠ b → a % 2016 ≠ b % 2016) :
  ∃ B : Set ℤ, B ⊆ A ∧ B.Card = 4 ∧ ∃ (a b c d ∈ B), (a + b - c - d) % 2016 = 0 := 
by
  sorry

end exists_subset_B_l795_795456


namespace chords_intersect_inside_circle_l795_795807

theorem chords_intersect_inside_circle (C : Finset (Set.Point → Prop)) (h : ∀ c ∈ C, ∃ d ∈ C, c ≠ d ∧  ∃ p, Midpoint p d = c) : 
  ∀ c1 c2 ∈ C, ∃ p, p ∈ ∩ C :=
sorry

end chords_intersect_inside_circle_l795_795807


namespace adult_tickets_sold_l795_795540

open Nat

theorem adult_tickets_sold (A C : ℕ) (h₁ : A + C = 522) (h₂ : 15 * A + 8 * C = 5086) :
  A = 130 :=
by
  sorry

end adult_tickets_sold_l795_795540


namespace complex_arithmetic_problem_l795_795639
open Complex

theorem complex_arithmetic_problem : (2 - 3 * Complex.I) * (2 + 3 * Complex.I) + (4 - 5 * Complex.I)^2 = 4 - 40 * Complex.I := by
  sorry

end complex_arithmetic_problem_l795_795639


namespace arithmetic_seq_k_l795_795836

theorem arithmetic_seq_k (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ) 
  (h1 : a 1 = -3)
  (h2 : a (k + 1) = 3 / 2)
  (h3 : S k = -12)
  (h4 : ∀ n, S n = n * (a 1 + a (n+1)) / 2):
  k = 13 :=
sorry

end arithmetic_seq_k_l795_795836


namespace abs_diff_of_roots_eq_one_l795_795282

theorem abs_diff_of_roots_eq_one {p q : ℝ} (h₁ : p + q = 7) (h₂ : p * q = 12) : |p - q| = 1 := 
by 
  sorry

end abs_diff_of_roots_eq_one_l795_795282


namespace angle_sum_correct_l795_795608

noncomputable def angle_sum (wzy wxy xyz : ℝ) : ℝ :=
  let wyz := wzy / 2
  let xyz := 3 * wyz
  wyz + xyz

theorem angle_sum_correct
  (wzy wxy : ℝ) (h_wzy : wzy = 50) (h_wxy : wxy = 20) :
  ∃ xyz : ℝ, 
    xyz = 3 * (wzy / 2) ∧
    angle_sum wzy wxy xyz = 110 :=
by {
  have wyz := wzy / 2,
  have xyz := 3 * wyz,
  use xyz,
  split,
  { 
    exact rfl 
  },
  {
    unfold angle_sum,
    rw [h_wzy],
    have : wyz = 50 / 2 := rfl,
    rw this,
    rw [←mul_assoc, mul_comm 3, mul_assoc, mul_assoc, ←one_mul 3],
    exact rfl,
  }
}

end angle_sum_correct_l795_795608


namespace geometric_sequence_seventh_term_l795_795876

theorem geometric_sequence_seventh_term (a r : ℝ) 
    (h1 : a * r^3 = 8) 
    (h2 : a * r^9 = 2) : 
    a * r^6 = 1 := 
by 
    sorry

end geometric_sequence_seventh_term_l795_795876


namespace multiples_of_10_between_11_and_103_l795_795756

def countMultiplesOf10 (lower_bound upper_bound : Nat) : Nat :=
  Nat.div (upper_bound - lower_bound) 10 + 1

theorem multiples_of_10_between_11_and_103 : 
  countMultiplesOf10 11 103 = 9 :=
by
  sorry

end multiples_of_10_between_11_and_103_l795_795756


namespace greatest_possible_value_of_2q_minus_r_l795_795539

theorem greatest_possible_value_of_2q_minus_r:
  ∃ (q r : ℤ), 1024 = 23 * q + r ∧ 0 < q ∧ 0 ≤ r ∧ r < 23 ∧ 2 * q - r = 76 :=
by
  use [44, 12]
  sorry

end greatest_possible_value_of_2q_minus_r_l795_795539


namespace intersection_of_A_and_B_l795_795749

noncomputable def A : Set ℝ := { x | x + 1 > 0 }
noncomputable def B : Set ℝ := { x | x^2 - x < 0 }

theorem intersection_of_A_and_B : A ∩ B = { x : ℝ | 0 < x ∧ x < 1 } := 
by 
  -- Conditions for A
  have hA : ∀ x, x ∈ A ↔ x > -1 := by 
    intro x
    rw [Set.mem_set_of_eq]
    exact iff.refl _ ,
  -- Conditions for B
  have hB : ∀ x, x ∈ B ↔ 0 < x ∧ x < 1 := by 
    intro x
    rw [Set.mem_set_of_eq]
    split_ifs ;
    exact sorry,
  -- Prove the intersection
  apply Set.ext
  intro x
  split ;
  {
    intro h
    rw [Set.mem_inter_iff Set.mem_set_of_eq Set.mem_set_of_eq] at h
    exact sorry,
  }  

end intersection_of_A_and_B_l795_795749


namespace correct_polar_equation_of_C1_correct_orthogonality_sum_value_l795_795034

namespace CurveProof

variable (α θ ρ1 ρ2 : ℝ)

def x_α : ℝ := cos α
def y_α : ℝ := 2 * sin α

def polar_equation_C1 (ρ θ : ℝ) : Prop := 
  (ρ^2) * (cos θ)^2 + (ρ^2) * (sin θ)^2 / 4 = 1

def orthogonality_condition (θ ρ1 ρ2 : ℝ) : Prop :=
  ρ1^2 * cos θ^2 + ρ1^2 * (sin θ^2) / 4 = 1 ∧
  ρ2^2 * sin θ^2 + ρ2^2 * (cos θ^2) / 4 = 1

def orthogonality_sum (ρ1 ρ2 θ : ℝ) : ℝ :=
  1 / ρ1^2 + 1 / ρ2^2 

theorem correct_polar_equation_of_C1 : 
  ∃ (ρ θ : ℝ), polar_equation_C1 ρ θ :=
sorry

theorem correct_orthogonality_sum_value (θ ρ1 ρ2 : ℝ) (h : orthogonality_condition θ ρ1 ρ2) : 
  orthogonality_sum ρ1 ρ2 θ = 5 / 4 :=
sorry

end CurveProof

end correct_polar_equation_of_C1_correct_orthogonality_sum_value_l795_795034


namespace point_below_line_range_l795_795026

theorem point_below_line_range (t : ℝ) : (2 * (-2) - 3 * t + 6 > 0) → t < (2 / 3) :=
by {
  sorry
}

end point_below_line_range_l795_795026


namespace tetrahedron_inequality_l795_795786

theorem tetrahedron_inequality 
  (A B C D : ℝ)
  (r_A R_A r_B R_B r_C R_C r_D R_D R : ℝ)
  (h_tetra : ∀ X ∈ {A, B, C, D}, acute_angle X)
  (h_radii : ∀ X ∈ {A, B, C, D}, 
    inscribed_radius X (opposite_face X) = r_X ∧ circumscribed_radius X (opposite_face X) = R_X) 
  (h_circumsphere : circumscribed_sphere_radius [A, B, C, D] = R) :
  8 * R^2 ≥ (r_A + R_A)^2 + (r_B + R_B)^2 + (r_C + R_C)^2 + (r_D + R_D)^2 :=
by
  sorry

end tetrahedron_inequality_l795_795786


namespace figure_C_perimeter_l795_795887

def is_perimeter (figure : Type) (perimeter : ℕ) : Prop :=
∃ x y : ℕ, (figure = 'A' → 6*x + 2*y = perimeter) ∧ 
           (figure = 'B' → 4*x + 6*y = perimeter) ∧
           (figure = 'C' → 2*x + 6*y = perimeter)

theorem figure_C_perimeter (hA : is_perimeter 'A' 56) (hB : is_perimeter 'B' 56) : 
  is_perimeter 'C' 40 :=
by
  sorry

end figure_C_perimeter_l795_795887


namespace axes_of_symmetry_of_regular_decagon_l795_795211

theorem axes_of_symmetry_of_regular_decagon : ∀ (n : ℕ), n = 10 → axes_of_symmetry (regular_polygon n) = n :=
by
  intros n hn
  sorry

end axes_of_symmetry_of_regular_decagon_l795_795211


namespace part1_part2_l795_795740

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≥ 0 then x^2 - 5 else 1 / (x + 1)

theorem part1 (m : ℝ) (h : f m = 4) : m = 3 ∨ m = -3 / 4 :=
by
  sorry

theorem part2 (a : ℝ) (h : f a < -6) : -7 / 6 < a ∧ a < -1 :=
by
  sorry

end part1_part2_l795_795740


namespace gcd_factorial_l795_795314

-- Definitions and conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- Theorem statement
theorem gcd_factorial : gcd (factorial 8) (factorial 10) = factorial 8 := by
  -- The proof is omitted
  sorry

end gcd_factorial_l795_795314


namespace set_intersection_l795_795344

def A := {x : ℤ | x > 2}
def B := {-1, 0, 1, 2, 3, 4 : ℤ}

theorem set_intersection : A ∩ B = {3, 4} :=
by
  -- Proof goes here
  sorry

end set_intersection_l795_795344


namespace sufficient_condition_for_lg_m_lt_1_l795_795624

theorem sufficient_condition_for_lg_m_lt_1 (m : ℝ) (h1 : m ∈ ({1, 2} : Set ℝ)) : Real.log m < 1 :=
sorry

end sufficient_condition_for_lg_m_lt_1_l795_795624


namespace perimeter_C_correct_l795_795905

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_l795_795905


namespace vect_perp_vect_norm_l795_795380

variables {θ : ℝ}

-- Problem 1
theorem vect_perp (h : (2 * cos θ - sin θ) = 0) :
  (sin θ - cos θ) / (sin θ + cos θ) = 1 / 3 := sorry

-- Problem 2
theorem vect_norm (h1 : |(cos θ - 2, sin θ + 1)| = 2) (h2 : 0 < θ ∧ θ < real.pi / 2) :
  sin (θ + real.pi / 4) = 7 * real.sqrt 2 / 10 := sorry

end vect_perp_vect_norm_l795_795380


namespace incenter_ratio_l795_795728

theorem incenter_ratio (a b : ℝ) (ha : a > b) (hb : b > 0) (M F1 F2 P N : ℝ × ℝ)
  (h_ell : (M.1 ^ 2) / (a ^ 2) + (M.2 ^ 2) / (b ^ 2) = 1)
  (h_foci : F1 = (-c, 0) ∧ F2 = (c, 0)) (h_c : c = real.sqrt (a ^ 2 - b ^ 2))
  (h_P : (P = incenter M F1 F2))
  (h_N : MP ∩ F1F2 = N) : 
  MN / NP = 1 := 
sorry

end incenter_ratio_l795_795728


namespace inequality_condition_l795_795393

theorem inequality_condition (n : ℕ) (a : Fin n → ℕ) (x : Fin n → ℝ)
    (hne_zero : ¬ (x = 0)) (h_sorted : ∀ i j : Fin n, i ≤ j → a i ≤ a j) 
    (hn_ge_two : n ≥ 2) : 
    (∑ i, a i * (x i) ^ 2 + 2 * ∑ i in Finset.range (n - 1), x i * x (i + 1) > 0) ↔ a 1 ≥ 2 := 
sorry

end inequality_condition_l795_795393


namespace meaningful_expression_range_l795_795024

theorem meaningful_expression_range {x : ℝ} : (∃ y : ℝ, y = 5 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end meaningful_expression_range_l795_795024


namespace family_vacation_rain_days_l795_795185

theorem family_vacation_rain_days (r_m r_a : ℕ) 
(h_rain_days : r_m + r_a = 13)
(clear_mornings : r_a = 11)
(clear_afternoons : r_m = 12) : 
r_m + r_a = 23 := 
by 
  sorry

end family_vacation_rain_days_l795_795185


namespace temperature_is_dependent_variable_l795_795443

theorem temperature_is_dependent_variable
  (temperature_water duration_exposure : ℝ)
  (h : ∀ t : ℝ, temperature_water = f (duration_exposure)) :
  dependent_variable temperature_water :=
sorry

def f (x : ℝ) : ℝ := sorry -- This represents the function defining the temperature dependency on duration.

end temperature_is_dependent_variable_l795_795443


namespace find_maximum_g_l795_795060

noncomputable def g (x : ℝ) := Real.sqrt(x * (100 - x)) + Real.sqrt(x * (8 - x))

theorem find_maximum_g :
  let N := Real.sqrt 736 in
  let x1 := 8 in
  0 ≤ x1 ∧ x1 ≤ 8 ∧ g x1 = N :=
by
  let x1 := 8
  let N := Real.sqrt 736
  have hx1 : 0 ≤ x1 := by norm_num
  have hx1' : x1 ≤ 8 := by norm_num
  have hN : g x1 = N := by
    dsimp [g, x1, N]
    norm_num
  exact ⟨hx1, hx1', hN⟩

end find_maximum_g_l795_795060


namespace ministry_transport_conn_l795_795096

noncomputable def make_one_way_paths (cities : Finset (Fin 1991)) 
  (roads : Finset (Fin 1991 × Fin 1991)) : Finset (Fin 1991 × Fin 1991) := sorry

theorem ministry_transport_conn (cities : Finset (Fin 1991)) 
  (roads : Finset (Fin 1991 × Fin 1991)) 
  (daily_closure : ℕ → Finset (Fin 1991 × Fin 1991)) 
  (daily_orientation : ℕ → Fin 1991 × Fin 1991) :
  ∀ t : ℕ, 
    (∀ day ∈ range t, day % 3 == 2 → (daily_orientation day ∈ roads ∧ ∀ (path ∈ daily_closure day), path ∉ (daily_closure day.succ))) 
    → (∀ city1 city2 : Fin 1991, city1 ∈ cities ∧ city2 ∈ cities → ∃ path, path ∈ make_one_way_paths cities roads 
                                               ∧ path.fst = city1 ∧ path.snd = city2) := sorry

end ministry_transport_conn_l795_795096


namespace card_probability_l795_795558

theorem card_probability :
  let cards := [0, -5, Real.pi, 2.5] in
  let num_cards := 4 in
  let num_integers := 2 in
  let prob := (num_integers : ℚ) / num_cards in
  prob = 1 / 2 :=
by
  sorry

end card_probability_l795_795558


namespace alice_does_not_lose_l795_795630

/-- 
Alice and Bob are playing chess, and each player is allowed to make two moves per turn. Alice starts.
Prove that Alice can ensure she does not lose.
-/
theorem alice_does_not_lose (chess_game : Type) (make_moves : chess_game → chess_game → chess_game)
  (alice_starts : chess_game) : 
  ∃ strategy : (chess_game → (list chess_game)), 
  ∀ bob_strategy : (chess_game → (list chess_game)), 
  ∀ game_state : chess_game,
    (strategy game_state ≠ bob_strategy game_state) ∧ 
    (strategy alice_starts ≠ bob_strategy alice_starts) :=
sorry

end alice_does_not_lose_l795_795630


namespace gcd_factorials_l795_795305

open Nat

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorials (h : ∀ n, 0 < n → factorial n = n * factorial (n - 1)) :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 :=
sorry

end gcd_factorials_l795_795305


namespace vector_equation_true_l795_795595

variables (A B C D E F O : Point)
variables [regular_hexagon : ∀ (P Q R S T U : Point), is_regular_hexagon P Q R S T U O]

variables [vector_space : vector_space ℝ (Point → Point)]
variables [vA : vector_space ℝ (A → O)]
variables [vAD : vector_space ℝ (A → D)]
variables [vFC : vector_space ℝ (F → C)]
variables [vBE : vector_space ℝ (B → E)]

-- Declare condition: O is the center of the regular hexagon ABCDEF.
axiom center_regular_hexagon : regular_hexagon A B C D E F O

-- Declare theorem to be proven
theorem vector_equation_true :
  vAD - vFC = vBE :=
sorry

end vector_equation_true_l795_795595


namespace smallest_positive_period_monotonically_increasing_interval_circumcircle_radius_l795_795367

def vec_m (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)
def vec_n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (2 * x), Real.cos x)
def f (x : ℝ) : ℝ := (vec_m x).1 * (vec_n x).1 + (vec_m x).2 * (vec_n x).2

def side_b := 1
def triangle_area := Real.sqrt 3

theorem smallest_positive_period : ∀ x, f (x + π) = f x :=
by
  sorry

theorem monotonically_increasing_interval : ∀ k : ℤ, 
  k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6 :=
by
  sorry

theorem circumcircle_radius (A : ℝ) (a b c : ℝ) : f A = 2 ∧ b = 1 ∧ (1 / 2) * b * c * Real.sin A = Real.sqrt 3 ∧
    (a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) ∧ (A = π / 3) → (R : ℝ) :=
by
  sorry

end smallest_positive_period_monotonically_increasing_interval_circumcircle_radius_l795_795367


namespace product_pattern_l795_795195

theorem product_pattern (a b : ℕ) (h1 : b < 10) (h2 : 10 - b < 10) :
    (10 * a + b) * (10 * a + (10 - b)) = 100 * a * (a + 1) + b * (10 - b) :=
by
  sorry

end product_pattern_l795_795195


namespace min_positive_sum_l795_795265

theorem min_positive_sum (b : Fin 50 → ℝ)
  (h : ∀ i, b i = 1 ∨ b i = -1) :
  ∃ T, T > 0 ∧ T = ∑ i in Finset.range 50, ∑ j in Finset.Icc i 49, b i * b j ∧ T = 7 :=
begin
  use 7,
  split,
  { linarith, },
  split,
  { sorry, },
  { sorry, }
end

end min_positive_sum_l795_795265


namespace nearest_integer_to_expression_correct_l795_795173

noncomputable def nearest_integer_to_expression : ℤ :=
  Int.floor ((3 + Real.sqrt 2) ^ 6)

theorem nearest_integer_to_expression_correct : nearest_integer_to_expression = 7414 :=
by
  sorry

end nearest_integer_to_expression_correct_l795_795173


namespace nate_age_when_ember_is_14_l795_795271

theorem nate_age_when_ember_is_14
  (nate_age : ℕ)
  (ember_age : ℕ)
  (h_half_age : ember_age = nate_age / 2)
  (h_nate_current_age : nate_age = 14) :
  nate_age + (14 - ember_age) = 21 :=
by
  sorry

end nate_age_when_ember_is_14_l795_795271


namespace xy_diff_square_l795_795471
-- Import the necessary library

-- Define the constants and expressions
def x := 1001^502 - 1001^(-502)
def y := 1001^502 + 1001^(-502)

-- Statement of the theorem to be proved
theorem xy_diff_square : x^2 - y^2 = -4 :=
by
  sorry

end xy_diff_square_l795_795471


namespace sum_of_digits_of_n_l795_795855

theorem sum_of_digits_of_n :
  ∃ n : ℕ, (log 3 (log 81 n) = log 9 (log 9 n) ∧ (n.digits 10).sum = 18) :=
sorry

end sum_of_digits_of_n_l795_795855


namespace cost_difference_l795_795453

theorem cost_difference (joy_pencils : ℕ) (colleen_pencils : ℕ) 
  (price_per_pencil_joy : ℝ) (price_per_pencil_colleen : ℝ) :
  joy_pencils = 30 →
  colleen_pencils = 50 →
  price_per_pencil_joy = 4 →
  price_per_pencil_colleen = 3.5 →
  (colleen_pencils * price_per_pencil_colleen - joy_pencils * price_per_pencil_joy) = 55 :=
by
  intros h_joy_pencils h_colleen_pencils h_price_joy h_price_colleen
  rw [h_joy_pencils, h_colleen_pencils, h_price_joy, h_price_colleen]
  norm_num
  repeat { sorry }

end cost_difference_l795_795453


namespace induction_prop_l795_795459

def majority (x : List (Fin n → Fin 2)) : Fin n → Fin 2 :=
  λ i, if 2 * (x.countp (λ y, y i = 1)) > x.length then 1 else 0

def P (S : Set (Fin n → Fin 2)) (k : Nat) : Prop :=
  ∀ t, t.length = 2 * k + 1 → List.toSet t ⊆ S → majority t ∈ S

theorem induction_prop {n : ℕ} (S : Set (Fin n → Fin 2)) (k : ℕ) 
  (h : ∀ t, t.length = 2 * k + 1 → List.toSet t ⊆ S → majority t ∈ S) :
  ∀ t, t.length = 2 * (k + 1) + 1 → List.toSet t ⊆ S → majority t ∈ S := by
  sorry

end induction_prop_l795_795459


namespace count_trailing_zeros_of_square_l795_795759

/--
Let \( n = 999,999,999,999,995 \). Prove that in the expansion of \( n^2 \),
there are 14 trailing zeros.
-/
theorem count_trailing_zeros_of_square : 
  ∃ n : ℕ, n = 999999999999995 ∧ 
  (∃ k : ℕ, k = number_of_trailing_zeros (n * n) ∧ k = 14) := 
sorry

end count_trailing_zeros_of_square_l795_795759


namespace find_polygon_sides_l795_795402

theorem find_polygon_sides (n : ℕ) (h : n - 3 = 5) : n = 8 :=
by
  sorry

end find_polygon_sides_l795_795402


namespace perimeter_C_l795_795900

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end perimeter_C_l795_795900


namespace range_of_x_l795_795017

theorem range_of_x (x : ℝ) : (x ≠ 2) ↔ ∃ y : ℝ, y = 5 / (x - 2) :=
begin
  sorry
end

end range_of_x_l795_795017


namespace Adam_daily_earnings_before_taxes_l795_795628

theorem Adam_daily_earnings_before_taxes 
  (daily_earnings taxes earnings_after_30_days : ℝ) 
  (h1 : taxes = 0.10 * daily_earnings)
  (h2 : earnings_after_30_days = 30 * (daily_earnings - taxes)) :
  daily_earnings = 40 := 
by
  have taxes_eq : taxes = 0.10 * daily_earnings from h1,
  have earnings_eq : 1080 = 30 * (0.90 * daily_earnings) from h2,
  sorry

end Adam_daily_earnings_before_taxes_l795_795628


namespace min_knights_l795_795090

noncomputable def is_lying (n : ℕ) (T : ℕ → Prop) (p : ℕ → Prop) : Prop :=
    (T n → ∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m)) ∨ (¬T n → ¬∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m ∧ m < n))

open Nat

def islanders_condition (T : ℕ → Prop) (p : ℕ → Prop) :=
  ∀ n, n < 80 → (T n ∨ ¬T n) ∧ (T n → ∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m)) ∨ (¬T n → ¬∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m))

theorem min_knights : ∀ (T : ℕ → Prop) (p : ℕ → Prop), islanders_condition T p → ∃ k, k = 70 :=    
by
    sorry

end min_knights_l795_795090


namespace min_value_objective_l795_795076

variable (x y : ℝ)

def constraints : Prop :=
  3 * x + y - 6 ≥ 0 ∧ x - y - 2 ≤ 0 ∧ y - 3 ≤ 0

def objective (x y : ℝ) : ℝ := y - 2 * x

theorem min_value_objective :
  constraints x y → ∃ x y, objective x y = -7 :=
by
  sorry

end min_value_objective_l795_795076


namespace sum_of_odds_is_square_l795_795709

theorem sum_of_odds_is_square (h1 : 1 = 1^2) 
  (h2 : 1 + 3 = 2^2) 
  (h3 : 1 + 3 + 5 = 3^2)
  (h4 : 1 + 3 + 5 + 7 = 4^2) 
  (n : ℕ) (h_pos : n > 0) : 
  1 + 3 + ... + (2 * n - 1) = n ^ 2 ∧ 
  derives_via_inductive_reasoning :=
sorry

end sum_of_odds_is_square_l795_795709


namespace smallest_sum_a_b_l795_795724

theorem smallest_sum_a_b (a b: ℕ) (h₀: 0 < a) (h₁: 0 < b) (h₂: a ≠ b) (h₃: 1 / (a: ℝ) + 1 / (b: ℝ) = 1 / 15) : a + b = 64 :=
sorry

end smallest_sum_a_b_l795_795724


namespace coefficient_x_squared_in_expansion_l795_795789

noncomputable def binomial_coeff (n k : ℕ) : ℕ := nat.choose n k

theorem coefficient_x_squared_in_expansion :
  let f (x : ℝ) := x + 3 / real.sqrt x in
  let s := 5 in
  (4^s / 2^s = 32) →
  (∃ (c : ℝ), ∃ (x^2_term : ℝ), x^2_term = binomial_coeff s 2 * 3^2 ∧ c = x^2_term) :=
by
  intros
  use binomial_coeff 5 2 * 3^2
  use 90
  split
  · exact rfl
  · exact rfl

end coefficient_x_squared_in_expansion_l795_795789


namespace sum_of_valid_students_per_row_l795_795944

theorem sum_of_valid_students_per_row :
  let divisors_of_360 := [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360] in
  let valid_values := List.filter (λ x, x * (360 / x) = 360 ∧ x >= 12 ∧ 360/x >= 15) divisors_of_360 in
  (List.sum valid_values) = 155 := 
by
  sorry

end sum_of_valid_students_per_row_l795_795944


namespace angle_UST_50_degrees_l795_795039

theorem angle_UST_50_degrees
  (Q U R S P T : Type)
  [is_equilateral_triangle Q U R]
  [is_equilateral_triangle S U R]
  [is_isosceles_triangle Q U P]
  [is_isosceles_triangle P U T]
  [is_isosceles_triangle T U S]
  (h1 : distance Q U = distance U R)
  (h2 : distance S U = distance U R)
  (h3 : distance P U = distance Q U)
  (h4 : distance T U = distance Q U)
  (h5 : distance Q P = distance P T)
  (h6 : distance T S = distance Q P) :
  angle U S T = 50 :=
sorry

end angle_UST_50_degrees_l795_795039


namespace polynomial_coeff_divisible_by_5_l795_795061

theorem polynomial_coeff_divisible_by_5
  (a b c : ℤ)
  (h : ∀ k : ℤ, (a * k^2 + b * k + c) % 5 = 0) :
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 :=
by
  sorry

end polynomial_coeff_divisible_by_5_l795_795061


namespace problem_statement_l795_795719

theorem problem_statement
  (A : Set ℕ)
  (a : ℕ → ℕ)
  (n : ℕ)
  (H1 : ∀ k, k < n → a (k + 1) > a k)
  (H2 : ∀ i j, i < j ∧ j < n → |a i - a j| ≥ (a i * a j) / 25) :
  (1 / a 1 - 1 / a n ≥ (n - 1) / 25) ∧
  (∀ i, 1 ≤ i ∧ i < n → i * (n - i) < 25) :=
sorry

end problem_statement_l795_795719


namespace mean_value_of_quadrilateral_angles_l795_795967

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795967


namespace area_of_right_triangle_l795_795191

def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0

noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |(B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)|

noncomputable def perimeter_of_triangle (A B C : ℝ × ℝ) : ℝ :=
 (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)) + (Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)) + (Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2))

-- Declaration of conditions
variables {A B C: ℝ × ℝ}
variable h_right : is_right_triangle A B C
variable h_perimeter : perimeter_of_triangle A B C = 54
variable h_AC_greater_than_10 : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) > 10
variable h_circle_tangent : ∃ O : ℝ × ℝ,
  (O.1, O.2) ∈ {(x, y) | B.1 < x ∧ x < C.1} ∧
  ((O.1 - A.1)^2 + (O.2 - A.2)^2 = 36) ∧
  ((O.1 - B.1)^2 + (O.2 - B.2)^2 = 36) ∧
  ((O.1 - C.1)^2 + (O.2 - C.2)^2 = 36)

-- The theorem statement
theorem area_of_right_triangle (A B C : ℝ × ℝ) :
  is_right_triangle A B C ∧
  perimeter_of_triangle A B C = 54 ∧
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) > 10 ∧
  (∃ O : ℝ × ℝ,
    (O.1, O.2) ∈ {(x, y) | B.1 < x ∧ x < C.1} ∧
    ((O.1 - A.1)^2 + (O.2 - A.2)^2 = 36) ∧
    ((O.1 - B.1)^2 + (O.2 - B.2)^2 = 36) ∧
    ((O.1 - C.1)^2 + (O.2 - C.2)^2 = 36)) →
  area_of_triangle A B C = 243 / 2 :=
sorry

end area_of_right_triangle_l795_795191


namespace smallest_integer_rel_prime_to_1020_l795_795578

theorem smallest_integer_rel_prime_to_1020 : ∃ n : ℕ, n > 1 ∧ n = 7 ∧ gcd n 1020 = 1 := by
  -- Here we state the theorem
  sorry

end smallest_integer_rel_prime_to_1020_l795_795578


namespace integers_between_sqrt5_and_sqrt50_l795_795001

theorem integers_between_sqrt5_and_sqrt50 : 
  (finset.Icc 3 7).card = 5 :=
by
  sorry

end integers_between_sqrt5_and_sqrt50_l795_795001


namespace ribbon_left_l795_795851

theorem ribbon_left (gifts : ℕ) (ribbon_per_gift Tom_ribbon_total : ℝ) (h1 : gifts = 8) (h2 : ribbon_per_gift = 1.5) (h3 : Tom_ribbon_total = 15) : Tom_ribbon_total - (gifts * ribbon_per_gift) = 3 := 
by
  sorry

end ribbon_left_l795_795851


namespace perimeter_C_l795_795916

theorem perimeter_C (x y : ℕ) 
  (h1 : 6 * x + 2 * y = 56)
  (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 := 
by
  sorry

end perimeter_C_l795_795916


namespace mork_tax_rate_l795_795085

theorem mork_tax_rate (M R : ℝ) (h1 : 0.15 = 0.15) (h2 : 4 * M = Mindy_income) (h3 : (R / 100 * M + 0.15 * 4 * M) = 0.21 * 5 * M):
  R = 45 :=
sorry

end mork_tax_rate_l795_795085


namespace perimeter_C_l795_795918

theorem perimeter_C (x y : ℕ) 
  (h1 : 6 * x + 2 * y = 56)
  (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 := 
by
  sorry

end perimeter_C_l795_795918


namespace mutually_exclusive_pairs_l795_795321

-- Define the events based on the conditions
def event_two_red_one_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ (drawn.count "red" = 2 ∧ drawn.count "white" = 1)

def event_one_red_two_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ (drawn.count "red" = 1 ∧ drawn.count "white" = 2)

def event_three_red (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ drawn.count "red" = 3

def event_at_least_one_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ 1 ≤ drawn.count "white"

def event_three_white (bag : List String) (drawn : List String) : Prop :=
  drawn.length = 3 ∧ drawn.count "white" = 3

-- Define mutually exclusive property
def mutually_exclusive (A B : List String → List String → Prop) (bag : List String) : Prop :=
  ∀ drawn, A bag drawn → ¬ B bag drawn

-- Define the main theorem statement
theorem mutually_exclusive_pairs (bag : List String) (condition : bag = ["red", "red", "red", "red", "red", "white", "white", "white", "white", "white"]) :
  mutually_exclusive event_three_red event_at_least_one_white bag ∧
  mutually_exclusive event_three_red event_three_white bag :=
by
  sorry

end mutually_exclusive_pairs_l795_795321


namespace simplify_tan_pi_over_24_add_tan_7pi_over_24_l795_795511

theorem simplify_tan_pi_over_24_add_tan_7pi_over_24 :
  let a := Real.tan (Real.pi / 24)
  let b := Real.tan (7 * Real.pi / 24)
  a + b = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 :=
by
  -- conditions and definitions:
  let tan_eq_sin_div_cos := ∀ x, Real.tan x = Real.sin x / Real.cos x
  let sin_add := ∀ a b, Real.sin (a + b) = Real.sin a * Real.cos b + Real.cos a * Real.sin b
  let cos_mul := ∀ a b, Real.cos a * Real.cos b = 1 / 2 * (Real.cos (a + b) + Real.cos (a - b))
  let sin_pi_over_3 := Real.sin (Real.pi / 3) = Real.sqrt 3 / 2
  let cos_pi_over_3 := Real.cos (Real.pi / 3) = 1 / 2
  let cos_pi_over_4 := Real.cos (Real.pi / 4) = Real.sqrt 2 / 2
  have cond1 := tan_eq_sin_div_cos
  have cond2 := sin_add
  have cond3 := cos_mul
  have cond4 := sin_pi_over_3
  have cond5 := cos_pi_over_3
  have cond6 := cos_pi_over_4
  sorry

end simplify_tan_pi_over_24_add_tan_7pi_over_24_l795_795511


namespace ratio_first_term_to_common_difference_l795_795180

theorem ratio_first_term_to_common_difference
  (a d : ℝ)
  (S_n : ℕ → ℝ)
  (hS_n : ∀ n, S_n n = (n / 2) * (2 * a + (n - 1) * d))
  (h : S_n 15 = 3 * S_n 10) :
  a / d = -2 :=
by
  sorry

end ratio_first_term_to_common_difference_l795_795180


namespace triangle_midpoints_collinear_l795_795662

open Set Function

-- Definitions from the conditions.
def Point := ℝ × ℝ
def Triangle (A B C : Point) : Prop := A ≠ B ∧ B ≠ C ∧ C ≠ A

structure Bisectors (A B C : Point) :=
  (FA GA : Point) -- on BC
  (FB GB : Point) -- on AC
  (FC GC : Point) -- on AB

structure Midpoints (A B C : Point) (b : Bisectors A B C) :=
  (MA : Point) -- midpoint of F_A G_A
  (MB : Point) -- midpoint of F_B G_B
  (MC : Point) -- midpoint of F_C G_C

-- Translation to Lean 4 statement.
theorem triangle_midpoints_collinear
  {A B C : Point} (hTriangle : Triangle A B C)
  (b : Bisectors A B C) (m : Midpoints A B C b) :
  ∃ (l : Line), on_line l m.MA ∧ on_line l m.MB ∧ on_line l m.MC :=
sorry

end triangle_midpoints_collinear_l795_795662


namespace simplify_expression_l795_795510

-- Define the algebraic expressions
def expr1 (x : ℝ) := (3 * x - 4) * (x + 9)
def expr2 (x : ℝ) := (x + 6) * (3 * x + 2)
def combined_expr (x : ℝ) := expr1 x + expr2 x
def result_expr (x : ℝ) := 6 * x^2 + 43 * x - 24

-- Theorem stating the equivalence
theorem simplify_expression (x : ℝ) : combined_expr x = result_expr x := 
by 
  sorry

end simplify_expression_l795_795510


namespace perimeter_C_l795_795921

theorem perimeter_C (x y : ℕ) 
  (h1 : 6 * x + 2 * y = 56)
  (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 := 
by
  sorry

end perimeter_C_l795_795921


namespace volumes_of_rotated_solids_l795_795375

theorem volumes_of_rotated_solids
  (π : ℝ)
  (b c a : ℝ)
  (h₁ : a^2 = b^2 + c^2)
  (v v₁ v₂ : ℝ)
  (hv : v = (1/3) * π * (b^2 * c^2) / a)
  (hv₁ : v₁ = (1/3) * π * c^2 * b)
  (hv₂ : v₂ = (1/3) * π * b^2 * c) :
  (1 / v^2) = (1 / v₁^2) + (1 / v₂^2) := 
by sorry

end volumes_of_rotated_solids_l795_795375


namespace max_points_earned_l795_795222

def divisible_by (a b : Nat) : Prop := b ≠ 0 ∧ a % b = 0

def points (x : Nat) : Nat :=
  (if divisible_by x 3 then 3 else 0) +
  (if divisible_by x 5 then 5 else 0) +
  (if divisible_by x 7 then 7 else 0) +
  (if divisible_by x 9 then 9 else 0) +
  (if divisible_by x 11 then 11 else 0)

theorem max_points_earned (x : Nat) (h1 : 2017 ≤ x) (h2 : x ≤ 2117) :
  points 2079 = 30 := by
  sorry

end max_points_earned_l795_795222


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795983

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795983


namespace cards_page_count_l795_795479

/-- Given the number of new and old cards for baseball, basketball, and football,
    each sport's cards must be organized with 3 cards per page.
    Prove the total number of pages required. -/
theorem cards_page_count :
  let baseball_new := 3 in
  let baseball_old := 9 in
  let basketball_new := 4 in
  let basketball_old := 6 in
  let football_new := 7 in
  let football_old := 5 in
  let total_baseball := baseball_new + baseball_old in
  let total_basketball := basketball_new + basketball_old in
  let total_football := football_new + football_old in
  let pages_baseball := (total_baseball + 2) / 3 in
  let pages_basketball := (total_basketball + 2) / 3 in
  let pages_football := (total_football + 2) / 3 in
  pages_baseball + pages_basketball + pages_football = 12 :=
by
  sorry

end cards_page_count_l795_795479


namespace product_evaluation_l795_795272

theorem product_evaluation (a : ℕ) (h : a = 7) : 
  ((a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a) = 5040 :=
by {
  cases h,
  norm_num,
}

end product_evaluation_l795_795272


namespace vendor_throws_away_28_percent_l795_795625

theorem vendor_throws_away_28_percent (n : ℕ) (h : n = 100) :
  let first_day_leftover := n * 40 / 100 in
  let first_day_thrown := n * 60 / 100 in
  let day1 := first_day_leftover in
  let second_day_sell := day1 * 50 / 100 in
  let second_day_leftover := day1 * 50 / 100 in
  let second_day_thrown := second_day_leftover in
  let total_thrown := first_day_thrown + second_day_thrown in
  total_thrown * 100 / n = 28 :=
by
  sorry

end vendor_throws_away_28_percent_l795_795625


namespace proof_of_number_of_triples_l795_795692

noncomputable def numberOfTriples : ℕ :=
  -- Definitions of the conditions
  let condition1 : ℝ × ℝ × ℝ → Prop := λ xyz, xyz.1 = 2020 - 2021 * Real.sign (xyz.2 + xyz.3)
  let condition2 : ℝ × ℝ × ℝ → Prop := λ xyz, xyz.2 = 2020 - 2021 * Real.sign (xyz.1 + xyz.3)
  let condition3 : ℝ × ℝ × ℝ → Prop := λ xyz, xyz.3 = 2020 - 2021 * Real.sign (xyz.1 + xyz.2)
  -- Counting the number of triples
  let triples : Finset (ℝ × ℝ × ℝ) := {(4041, -1, -1), (-1, 4041, -1), (-1, -1, 4041)}
  triples.card

theorem proof_of_number_of_triples :
  numberOfTriples = 3 :=
sorry  -- Proof goes here

end proof_of_number_of_triples_l795_795692


namespace perimeter_C_correct_l795_795908

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_l795_795908


namespace best_number_max_points_chosen_number_is_best_l795_795226

def is_divisible (a b : ℕ) : Prop := b % a = 0

def points (x : ℕ) : ℕ :=
  (if is_divisible 3 x then 3 else 0) +
  (if is_divisible 5 x then 5 else 0) +
  (if is_divisible 7 x then 7 else 0) +
  (if is_divisible 9 x then 9 else 0) +
  (if is_divisible 11 x then 11 else 0)

def max_points : ℕ :=
  30

def best_number : ℕ :=
  2079

theorem best_number_max_points :
  ∀ x : ℕ, 2017 ≤ x ∧ x ≤ 2117 → points x ≤ max_points :=
sorry

theorem chosen_number_is_best :
  points best_number = max_points :=
sorry

end best_number_max_points_chosen_number_is_best_l795_795226


namespace geometric_sequence_eighth_term_l795_795158

theorem geometric_sequence_eighth_term (a r : ℝ) (h₀ : a = 27) (h₁ : r = 1/3) :
  a * r^7 = 1/81 :=
by
  rw [h₀, h₁]
  sorry

end geometric_sequence_eighth_term_l795_795158


namespace gcd_fact_8_10_l795_795301

theorem gcd_fact_8_10 : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = 40320 := by
  -- No proof needed
  sorry

end gcd_fact_8_10_l795_795301


namespace sum_series_square_l795_795146

theorem sum_series_square :
  (∑ i in (Finset.range 1995), i + 1) + (∑ i in (Finset.range 1994).filter (λ x, x < 1994), 1994 - x) = (1994.5) ^ 2 :=
by
  sorry

end sum_series_square_l795_795146


namespace initial_amount_l795_795947

def spent : ℕ := 15
def left : ℕ := 63
def initial : ℕ := spent + left

theorem initial_amount : initial = 78 := by
  unfold initial
  rw [spent, left]
  norm_num
  sorry

end initial_amount_l795_795947


namespace enclosed_area_l795_795829

def f (n : ℕ) (x : ℝ) : ℝ :=
  if n = 0 then |x|
  else if n > 0 then abs (f (n - 1) x - n)
  else 0 -- This case will never happen, logically n cannot be less than 0

theorem enclosed_area (n : ℕ) : 
  let area := (4 * n^3 + 6 * n^2 - 1 + (-1)^n) / 8 in
  ∑ x in {x : ℝ | f n x = 0}, integral 0 x (f n) = area := 
by 
  sorry

end enclosed_area_l795_795829


namespace product_profit_equation_l795_795606

theorem product_profit_equation (purchase_price selling_price : ℝ) 
                                (initial_units units_decrease_per_dollar_increase : ℝ)
                                (profit : ℝ)
                                (hx : purchase_price = 35)
                                (hy : selling_price = 40)
                                (hz : initial_units = 200)
                                (hs : units_decrease_per_dollar_increase = 5)
                                (hp : profit = 1870) :
  ∃ x : ℝ, (x + (selling_price - purchase_price)) * (initial_units - units_decrease_per_dollar_increase * x) = profit :=
by { sorry }

end product_profit_equation_l795_795606


namespace zs_share_in_profit_l795_795182

noncomputable def calculateProfitShare (x_investment y_investment z_investment z_months total_profit : ℚ) : ℚ :=
  let x_invest_months := x_investment * 12
  let y_invest_months := y_investment * 12
  let z_invest_months := z_investment * z_months
  let total_invest_months := x_invest_months + y_invest_months + z_invest_months
  let z_share := z_invest_months / total_invest_months
  total_profit * z_share

theorem zs_share_in_profit :
  calculateProfitShare 36000 42000 48000 8 14190 = 2580 :=
by
  sorry

end zs_share_in_profit_l795_795182


namespace hockey_league_games_l795_795942

theorem hockey_league_games (n t : ℕ) (h1 : n = 15) (h2 : t = 1050) :
  ∃ k, ∀ team1 team2 : ℕ, team1 ≠ team2 → k = 10 :=
by
  -- Declare k as the number of times each team faces the other teams
  let k := 10
  -- Verify the total number of teams and games
  have hn : n = 15 := h1
  have ht : t = 1050 := h2
  -- For any two distinct teams, they face each other k times
  use k
  intros team1 team2 hneq
  -- Show that k equals 10 under given conditions
  exact rfl

end hockey_league_games_l795_795942


namespace probability_of_alpha_l795_795046

theorem probability_of_alpha :
  ∀ (x : ℝ), (-6 ≤ x ∧ x ≤ 6) →
  let tangent_slope := 2 * x in
  let α := real.atan tangent_slope in
  ∃ (p : ℝ), p = 11 / 12 ∧
  (integral λ x, if π / 4 ≤ α ∧ α ≤ 3 * π / 4 then 1 else 0 from -6 to 6) / 12 = p :=
sorry

end probability_of_alpha_l795_795046


namespace count_squares_mn_eq_sum_count_squares_nn_eq_formula_l795_795758

-- Define the function to count the total number of squares in an m x n grid
def count_squares (m n : ℕ) : ℕ :=
  ∑ k in Finset.range m, (m - k) * (n - k)

-- Define the function to count the total number of squares in an n x n grid
def count_squares_nn (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

-- State the theorems
theorem count_squares_mn_eq_sum (m n : ℕ) :
  count_squares m n = ∑ k in Finset.range m, (m - k) * (n - k) :=
sorry

theorem count_squares_nn_eq_formula (n : ℕ) :
  count_squares_nn n = (n * (n + 1) * (2 * n + 1) / 6) :=
sorry

end count_squares_mn_eq_sum_count_squares_nn_eq_formula_l795_795758


namespace christine_aquafaba_needed_l795_795652

-- Define the number of tablespoons per egg white
def tablespoons_per_egg_white : ℕ := 2

-- Define the number of egg whites per cake
def egg_whites_per_cake : ℕ := 8

-- Define the number of cakes
def number_of_cakes : ℕ := 2

-- Express the total amount of aquafaba needed
def aquafaba_needed : ℕ :=
  tablespoons_per_egg_white * egg_whites_per_cake * number_of_cakes

-- Statement asserting the amount of aquafaba needed is 32
theorem christine_aquafaba_needed : aquafaba_needed = 32 := by
  sorry

end christine_aquafaba_needed_l795_795652


namespace parabola_distance_l795_795745

def parabola_focus (y : ℝ) (x : ℝ) : Prop :=
  y^2 = 8 * x ∧ 
  ∃ (F : ℝ × ℝ), F = (2, 0)

def directrix (x : ℝ → ℝ) : Prop :=
  ∀ x, x = -2

def point_on_parabola (P : ℝ × ℝ) : Prop :=
  P.1^2 = 4 * P.2 

def slope_AF (A P : ℝ × ℝ) : Prop :=
  ∀ A P, A = (-2, 4) ∧ P = (2, 4) ∧ (P.2 - A.2) / (P.1 - A.1) = -1

def PF_distance (P F : ℝ × ℝ) : ℝ :=
  dist P F

theorem parabola_distance :
  ∀ P : ℝ × ℝ, ∀ F : ℝ × ℝ,
    parabola_focus P.1 P.2 → 
    directrix (λ x => x) → 
    point_on_parabola P → 
    slope_AF F P → 
    PF_distance P F = 4 := 
by 
  sorry

end parabola_distance_l795_795745


namespace percentage_increase_in_sales_l795_795580

theorem percentage_increase_in_sales (P Q : ℝ) (hP : P ≠ 0) (hQ : Q ≠ 0) :
  let new_price := 0.7 * P
  let new_quantity := (Q : ℝ) * (1 + (50 / 100))
  let new_receipts := new_price * new_quantity
  new_receipts = 1.05 * P * Q :=
by
  let x := 50
  have h : 0.7 * P * Q * (1 + x / 100) = 1.05 * P * Q :=
    sorry
  exact h

end percentage_increase_in_sales_l795_795580


namespace complex_quadrant_l795_795334

theorem complex_quadrant 
  (z : ℂ) 
  (h : (2 + 3 * Complex.I) * z = 1 + Complex.I) : 
  z.re > 0 ∧ z.im < 0 := 
sorry

end complex_quadrant_l795_795334


namespace binomial_expansion_l795_795394

theorem binomial_expansion (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ) :
  (1 + 2 * 1)^5 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 ∧
  (1 + 2 * -1)^5 = a_0 - a_1 + a_2 - a_3 + a_4 - a_5 → 
  a_0 + a_2 + a_4 = 121 :=
by
  intro h
  let h₁ := h.1
  let h₂ := h.2
  sorry

end binomial_expansion_l795_795394


namespace smallest_sum_of_three_distinct_numbers_l795_795144

theorem smallest_sum_of_three_distinct_numbers :
  ∀ (s : Set ℤ), s = {8, 27, -2, 14, -4} → (∀ a b c ∈ s, a ≠ b ∧ b ≠ c ∧ a ≠ c → a + b + c ≥ 2) :=
by sorry

end smallest_sum_of_three_distinct_numbers_l795_795144


namespace expression_1_expression_2_expression_3_expression_4_l795_795274

section problem1

variable {x : ℝ}

theorem expression_1:
  (x^2 - 1 + x)*(x^2 - 1 + 3*x) + x^2  = x^4 + 4*x^3 + 4*x^2 - 4*x - 1 :=
sorry

end problem1

section problem2

variable {x a : ℝ}

theorem expression_2:
  (x - a)^4 + 4*a^4 = (x^2 + a^2)*(x^2 - 4*a*x + 5*a^2) :=
sorry

end problem2

section problem3

variable {a : ℝ}

theorem expression_3:
  (a + 1)^4 + 2*(a + 1)^3 + a*(a + 2) = (a + 1)^4 + 2*(a + 1)^3 + 1 :=
sorry

end problem3

section problem4

variable {p : ℝ}

theorem expression_4:
  (p + 2)^4 + 2*(p^2 - 4)^2 + (p - 2)^4 = 4*p^4 :=
sorry

end problem4

end expression_1_expression_2_expression_3_expression_4_l795_795274


namespace rectangle_area_l795_795025

def length : ℝ := Real.sqrt 6
def width : ℝ := Real.sqrt 3
def area : ℝ := length * width

theorem rectangle_area : area = 3 * Real.sqrt 2 := 
by
  let l := length
  let w := width
  have h1 : l * w = Real.sqrt 6 * Real.sqrt 3 := by refl
  rw [mul_comm l w] at h1
  rw [Real.sqrt_mul] at h1
  rw [mul_comm 6 3]
  rw [Real.sqrt_mul h1]
  simp
  sorry 

end rectangle_area_l795_795025


namespace five_digit_palindrome_count_l795_795259

def is_digit (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ 9

def is_nonzero_digit (x : ℕ) : Prop := x ≥ 1 ∧ x ≤ 9

def is_palindrome (n : ℕ) : Prop :=
  let d1 := n / 10000,
      d2 := (n / 1000) % 10,
      d3 := (n / 100) % 10,
      d4 := (n / 10) % 10,
      d5 := n % 10 in
  d1 = d5 ∧ d2 = d4

def count_five_digit_palindromes : ℕ :=
  (finset.Ico 10000 100000).filter (λ n, is_palindrome n).card

theorem five_digit_palindrome_count : count_five_digit_palindromes = 900 := by
  sorry

end five_digit_palindrome_count_l795_795259


namespace best_number_max_points_chosen_number_is_best_l795_795224

def is_divisible (a b : ℕ) : Prop := b % a = 0

def points (x : ℕ) : ℕ :=
  (if is_divisible 3 x then 3 else 0) +
  (if is_divisible 5 x then 5 else 0) +
  (if is_divisible 7 x then 7 else 0) +
  (if is_divisible 9 x then 9 else 0) +
  (if is_divisible 11 x then 11 else 0)

def max_points : ℕ :=
  30

def best_number : ℕ :=
  2079

theorem best_number_max_points :
  ∀ x : ℕ, 2017 ≤ x ∧ x ≤ 2117 → points x ≤ max_points :=
sorry

theorem chosen_number_is_best :
  points best_number = max_points :=
sorry

end best_number_max_points_chosen_number_is_best_l795_795224


namespace problem_solution_l795_795397

theorem problem_solution :
  ∀ x y : ℝ, 9 * y^2 + 6 * x * y + x + 12 = 0 → (x ≤ -3 ∨ x ≥ 4) :=
  sorry

end problem_solution_l795_795397


namespace abs_diff_of_roots_l795_795283

theorem abs_diff_of_roots : 
  ∀ r1 r2 : ℝ, 
  (r1 + r2 = 7) ∧ (r1 * r2 = 12) → abs (r1 - r2) = 1 :=
by
  -- Assume the roots are r1 and r2
  intros r1 r2 H,
  -- Decompose the assumption H into its components
  cases H with Hsum Hprod,
  -- Calculate the square of the difference using the given identities
  have H_squared_diff : (r1 - r2)^2 = (r1 + r2)^2 - 4 * (r1 * r2),
  { sorry },
  -- Substitute the known values to find the square of the difference
  have H_squared_vals : (r1 - r2)^2 = 49 - 4 * 12,
  { sorry },
  -- Simplify to get (r1 - r2)^2 = 1
  have H1 : (r1 - r2)^2 = 1,
  { sorry },
  -- The absolute value of the difference is the square root of this result
  have abs_diff : abs (r1 - r2) = 1,
  { sorry },
  -- Conclude the proof by showing the final result matches the expected answer
  exact abs_diff

end abs_diff_of_roots_l795_795283


namespace find_polygon_sides_l795_795403

theorem find_polygon_sides (n : ℕ) (h : n - 3 = 5) : n = 8 :=
by
  sorry

end find_polygon_sides_l795_795403


namespace perimeter_C_l795_795901

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end perimeter_C_l795_795901


namespace smallest_divisor_subtracted_l795_795291

theorem smallest_divisor_subtracted (a b d : ℕ) (h1: a = 899830) (h2: b = 6) (h3: a - b = 899824) (h4 : 6 < d) 
(h5 : d ∣ (a - b)) : d = 8 :=
by
  sorry

end smallest_divisor_subtracted_l795_795291


namespace rhombus_diagonal_length_l795_795526

theorem rhombus_diagonal_length (d1 d2 : ℝ) (area : ℝ) 
(h_d2 : d2 = 18) (h_area : area = 126) (h_formula : area = (d1 * d2) / 2) : 
d1 = 14 :=
by
  -- We're skipping the proof steps.
  sorry

end rhombus_diagonal_length_l795_795526


namespace nearest_int_to_expr_l795_795169

theorem nearest_int_to_expr : 
  |(3 + Real.sqrt 2)^6 - 3707| < 1 :=
by 
  sorry

end nearest_int_to_expr_l795_795169


namespace no_integer_solutions_l795_795669

theorem no_integer_solutions (x y z : ℤ) (h1 : x > y) (h2 : y > z) : 
  x * (x - y) + y * (y - z) + z * (z - x) ≠ 3 := 
by
  sorry

end no_integer_solutions_l795_795669


namespace sum_of_coefficients_correct_maximum_coeff_terms_correct_l795_795790

    -- The binomial expansion condition
    def binomial_expansion (x : ℝ) (n : ℕ) : ℝ → ℝ :=
      λ x, (sqrt x + 1 / (2 * cbrt x)) ^ n

    noncomputable def sum_of_coefficients : ℝ :=
      (3 / 2) ^ 8  -- Using x = 1 simplifies the sum

    theorem sum_of_coefficients_correct :
      sum_of_coefficients = 6561 / 256 :=
    by
      sorry

    -- Terms with maximum coefficients condition
    def term_t3 (x : ℝ) : ℝ := 7 * x^(7 / 3)
    def term_t4 (x : ℝ) : ℝ := 7 * x^(3 / 2)

    theorem maximum_coeff_terms_correct (x : ℝ) :
      (binomial_expansion x 8) = λ x,
      [--- some representation of the binomial expansion terms ---] → 
      term_t3 x ∨ term_t4 x :=
    by
      sorry
    
end sum_of_coefficients_correct_maximum_coeff_terms_correct_l795_795790


namespace participants_in_book_club_and_painting_l795_795609

-- Definitions based on conditions
def total_participants : ℕ := 120

def book_club : ℕ := 80
def fun_sports : ℕ := 50
def painting : ℕ := 40

def book_club_and_fun_sports : ℕ := 20
def fun_sports_and_painting : ℕ := 10

-- Theorem statement with conditions and target answer
theorem participants_in_book_club_and_painting
  (total_participants = 120)
  (book_club = 80)
  (fun_sports = 50)
  (painting = 40)
  (book_club_and_fun_sports = 20)
  (fun_sports_and_painting = 10) :
  ∃ (book_club_and_painting : ℕ), book_club_and_painting = 20 :=
sorry

end participants_in_book_club_and_painting_l795_795609


namespace distinct_z_values_l795_795357

-- Definitions for the problem conditions
def reverse_digits (x : ℕ) : ℕ :=
  let c := x % 10
  let b := (x / 10) % 10
  let a := (x / 100) % 10
  100 * c + 10 * b + a

def valid_x (x : ℕ) : Prop := 200 ≤ x ∧ x ≤ 999
def valid_y (y : ℕ) : Prop := 100 ≤ y ∧ y ≤ 999

def z (x y : ℕ) : ℕ := x + y

-- Main theorem statement
theorem distinct_z_values : ∀ (x y : ℕ),
  valid_x x → valid_y y →
  y = reverse_digits x →
  unique_count_z (z x y) = 1878 :=
sorry

-- Helper function to count distinct values
def unique_count_z (z : ℕ → ℕ) : ℕ :=
sorry

end distinct_z_values_l795_795357


namespace find_larger_number_l795_795874

-- Definitions of the given conditions
variables (L S : ℕ)
definition condition1 : Prop := L - S = 1365
definition condition2 : Prop := L = 6 * S + 5

-- Theorem statement to prove
theorem find_larger_number (h1 : condition1 L S) (h2 : condition2 L S) : L = 1637 :=
sorry

end find_larger_number_l795_795874


namespace boat_speeds_relation_l795_795951

variables (v1 v2 : ℝ) (d0 : ℝ) (t : ℝ)
-- setting up the known values and conditions
def first_boat_speed := 25
def initial_distance := 20
def time_to_collision := 1 / 60
def distance_apart_before_collision := 0.5

-- assuming that both boats are moving towards each other and the given conditions
theorem boat_speeds_relation 
  (h1 : d0 = initial_distance)
  (h2 : v1 = first_boat_speed)
  (h3 : d0 - (v1 + v2) * time_to_collision = distance_apart_before_collision)
  (h4 : v1 + v2 = 30): 
  v1 = 25 :=
by {
    sorry
}

end boat_speeds_relation_l795_795951


namespace find_fraction_value_l795_795731

variable (a b : ℝ)
variable (ha_ne_hb : a ≠ b)
variable (root_condition : a + b - 20 = 0)

theorem find_fraction_value : (a^2 - b^2) / (2 * a - 2 * b) = 10 :=
by
  have h1 : a + b = 20 := by
    rw [← add_sub_cancel' a b, root_condition]
    add_eq_zero_iff.2 (25, 4).symm
  sorry

end find_fraction_value_l795_795731


namespace find_a2_l795_795398

theorem find_a2 (f : ℤ → ℤ) (a : ℕ → ℤ) (x : ℤ) :
  (x^2 + (x + 1)^7 = a[0] + a[1] * (x + 2) + a[2] * (x + 2)^2 + a[3] * (x + 2)^3 + a[4] * (x + 2)^4 + a[5] * (x + 2)^5 + a[6] * (x + 2)^6 + a[7] * (x + 2)^7) →
  (a[2] = -20) :=
by sorry

end find_a2_l795_795398


namespace find_t_l795_795358

open Real

noncomputable def curve_C (t : ℝ) (x : ℝ) : ℝ :=
  3 * abs (x - t)

noncomputable def circle_O (x : ℝ) (y : ℝ) : Prop :=
  x^2 + y^2 = 4

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure NaturalPoint :=
  (x : ℕ)
  (y : ℕ)

structure OnCurve (C : ℝ → ℝ → ℝ) (P : Point) : Prop :=
  (on_curve : P.y = C P.x)

def distance (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

theorem find_t (t : ℝ) (m n s p k : ℕ)
  (h_mnN : m > 0 ∧ n > 0 ∧ s > 0 ∧ p > 0)
  (P_A : Point := ⟨m, n⟩)
  (P_B : Point := ⟨s, p⟩)
  (points_on_curve : OnCurve (curve_C t) P_A ∧ OnCurve (curve_C t) P_B)
  (ratio_constant : ∀ (x y : ℝ), circle_O x y → (distance ⟨x, y⟩ P_A) / (distance ⟨x, y⟩ P_B) = k)
  (hk_gt_one : k > 1) :
  t = 4/3 := sorry

end find_t_l795_795358


namespace carpet_size_l795_795802

def length := 5
def width := 2
def area := length * width

theorem carpet_size : area = 10 := by
  sorry

end carpet_size_l795_795802


namespace elliptic_properties_l795_795341

-- Given conditions
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop := (a > b > 0) ∧ (x^2 / a^2 + y^2 / b^2 = 1)
def minor_axis_length (b : ℝ) : Prop := (2 * b = 4)
def focal_distance (a b : ℝ) : Prop := (√(a^2 - b^2) = 1)

-- Main theorem
theorem elliptic_properties :
  ∀ (a b : ℝ),
    is_ellipse a b x y →
    minor_axis_length b →
    focal_distance a b →
    (a = √5 ∧ b = 2 ∧
     (∀ (F1 A B : ℝ) (m : line_equation),
      F1 = -1 ∧ 
      m = x + 1 ∧ 
      segment_length A B a b m = 8 * √10 / 9)) :=
by sorry

end elliptic_properties_l795_795341


namespace c_ge_one_l795_795073

theorem c_ge_one (a b : ℕ) (c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : (a + 1) / (b + c) = b / a) : c ≥ 1 := 
sorry

end c_ge_one_l795_795073


namespace value_of_m_l795_795770

theorem value_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (3 / (x - 2) + (x + m) / (2 - x) = 1)) → m = 1 :=
by
  sorry

end value_of_m_l795_795770


namespace hours_worked_each_day_l795_795638

-- Given conditions
def total_hours_worked : ℕ := 18
def number_of_days_worked : ℕ := 6

-- Statement to prove
theorem hours_worked_each_day : total_hours_worked / number_of_days_worked = 3 := by
  sorry

end hours_worked_each_day_l795_795638


namespace problem_statement_l795_795300

noncomputable def f (n : ℕ) : ℝ := Real.log (n^2) / Real.log 3003

theorem problem_statement : f 33 + f 13 + f 7 = 2 := 
by
  sorry

end problem_statement_l795_795300


namespace perimeter_C_correct_l795_795904

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_l795_795904


namespace f_f_2_l795_795475

def f : ℝ → ℝ :=
  λ x, if x ≥ 1 then real.sqrt (x - 1) else x

theorem f_f_2 : f (f 2) = 0 := 
by
  sorry

end f_f_2_l795_795475


namespace perimeter_of_one_of_the_rectangles_l795_795045

noncomputable def perimeter_of_rectangle (z w : ℕ) : ℕ :=
  2 * z

theorem perimeter_of_one_of_the_rectangles (z w : ℕ) :
  ∃ P, P = perimeter_of_rectangle z w :=
by
  use 2 * z
  sorry

end perimeter_of_one_of_the_rectangles_l795_795045


namespace absolute_value_of_difference_of_quadratic_roots_l795_795288
noncomputable theory

open Real

def quadratic_roots (a b c : ℝ) : ℝ × ℝ :=
let discr := b^2 - 4 * a * c in
((−b + sqrt discr) / (2 * a), (−b - sqrt discr) / (2 * a))

theorem absolute_value_of_difference_of_quadratic_roots :
  ∀ r1 r2 : ℝ, 
  r1^2 - 7 * r1 + 12 = 0 → r2^2 - 7 * r2 + 12 = 0 →
  abs (r1 - r2) = 5 :=
by
  sorry

end absolute_value_of_difference_of_quadratic_roots_l795_795288


namespace max_value_of_gems_l795_795424

/-- Conditions -/
structure Gem :=
  (weight : ℕ)
  (value : ℕ)

def Gem1 : Gem := ⟨3, 9⟩
def Gem2 : Gem := ⟨6, 20⟩
def Gem3 : Gem := ⟨2, 5⟩

-- Laura can carry maximum of 21 pounds.
def max_weight : ℕ := 21

-- She is able to carry at least 15 of each type
def min_count := 15

/-- Prove that the maximum value Laura can carry is $69 -/
theorem max_value_of_gems : ∃ (n1 n2 n3 : ℕ), (n1 >= min_count) ∧ (n2 >= min_count) ∧ (n3 >= min_count) ∧ 
  (Gem1.weight * n1 + Gem2.weight * n2 + Gem3.weight * n3 ≤ max_weight) ∧ 
  (Gem1.value * n1 + Gem2.value * n2 + Gem3.value * n3 = 69) :=
sorry

end max_value_of_gems_l795_795424


namespace false_propositions_l795_795383

theorem false_propositions (p q : Prop) (hnp : ¬ p) (hq : q) :
  (¬ p) ∧ (¬ (p ∧ q)) ∧ (¬ ¬ q) :=
by {
  exact ⟨hnp, not_and_of_not_left q hnp, not_not_intro hq⟩
}

end false_propositions_l795_795383


namespace radius_of_C3_l795_795248

-- Define the problem context
variables (C1 C2 C3 : Type) [MetricSpace C1] [MetricSpace C2] [MetricSpace C3]
variable (O1 O2 O3 : Point)
variable (r1 r2 r3 : ℝ) -- Radii of the circles
variable (h : ℝ) -- Length HO1
variable (T : Point) -- Tangency point

-- Geometric constraints (conditions)
axiom C1_radius : r1 = 6
axiom C2_radius : r2 = 9
axiom centers_collinear : Collinear [O1, O2, O3]
axiom circles_tangent : TangentExternally C1 C2 ∧ TangentInternally C1 C3 ∧ TangentInternally C2 C3
axiom common_tangent_props : ∃ T1 T2 : Point, CommonTangentExternallyTouchingAt C1 C2 T1 T2 ∧ PerpendicularToDiameterAt T C3

-- The theorem stating the radius of C3
theorem radius_of_C3 : r3 = 11 :=
sorry

end radius_of_C3_l795_795248


namespace cal_fraction_anthony_is_two_thirds_l795_795843

def mabel_tx : ℕ := 90
def anthony_tx : ℕ := mabel_tx + (0.1 * mabel_tx).toNat
def jade_tx : ℕ := 81
def cal_tx : ℕ := jade_tx - 15
def fraction_cal_anthony : ℚ := cal_tx / anthony_tx

theorem cal_fraction_anthony_is_two_thirds :
  fraction_cal_anthony = 2 / 3 :=
by 
  sorry

end cal_fraction_anthony_is_two_thirds_l795_795843


namespace cafeteria_pies_l795_795194

theorem cafeteria_pies (initial_apples handed_out_apples apples_per_pie : ℕ)
  (h_initial : initial_apples = 50)
  (h_handed_out : handed_out_apples = 5)
  (h_apples_per_pie : apples_per_pie = 5) :
  (initial_apples - handed_out_apples) / apples_per_pie = 9 := 
by
  sorry

end cafeteria_pies_l795_795194


namespace similar_triangles_area_ratio_not_equal_similarity_ratio_l795_795956

theorem similar_triangles_area_ratio_not_equal_similarity_ratio
  (ΔABC ΔDEF : Type)
  [triangle ΔABC] [triangle ΔDEF]
  (h_similar : similar ΔABC ΔDEF)
  (k : ℝ) 
  (h_ratio : ratio_of_sides ΔABC ΔDEF = k) :
  ratio_of_areas ΔABC ΔDEF ≠ k :=
by
  sorry

end similar_triangles_area_ratio_not_equal_similarity_ratio_l795_795956


namespace angle_ACB_is_60_l795_795809

variables {A B C D E : Type*} [IsTriangle ABC] [IsTriangle ADE]
variables (Γ : Circle) [Circumcircle ABC Γ] [Circumcircle ADE Γ]
variables (orthocenter : Point → Point → Point → Point)
variables [Perpendicular (A ⟶ BC) D] [Perpendicular (B ⟶ AC) E]
variables (AB : ℝ) (DE : ℝ)
variables (angleACB : ℕ) 

-- Given conditions 
variables (h1 : D ∈ Γ) (h2 : E ∈ Γ) (h3 : AB = DE) (h4 : angleACB = 60)

theorem angle_ACB_is_60 (ABC : Triangle) :
  angle (A, C, B) = 60 :=
by 
  sorry

end angle_ACB_is_60_l795_795809


namespace stock_decrease_to_original_l795_795214

theorem stock_decrease_to_original : ∀ (x : ℝ), x > 0 → 
  let new_price := 1.40 * x in
  ∃ p : ℝ, p = 28.5714 / 100 ∧ (1 - p) * new_price = x :=
begin
  intros,
  unfold new_price,
  use 28.5714 / 100,
  split,
  { refl },
  { sorry }
end

end stock_decrease_to_original_l795_795214


namespace minimum_modulus_of_z_l795_795333

open Complex

theorem minimum_modulus_of_z (z : ℂ) (h : ∃ x : ℝ, (x : ℂ)^2 - 2*z*x + 3/4 + ⟨0, 1⟩ = 0) : |z| ≥ 1 := 
sorry

end minimum_modulus_of_z_l795_795333


namespace g_monotonically_increasing_interval_l795_795365

theorem g_monotonically_increasing_interval :
  ∀ x y : ℝ,
  (x ≥ -π / 12 ∧ x ≤ π / 12 ∧ y > x ∧ y ≤ π / 12) →
  (sin (2 * y) > sin (2 * x)) :=
begin
  sorry
end

end g_monotonically_increasing_interval_l795_795365


namespace profit_percentage_is_five_l795_795602

-- Define the cost price (CP) and selling price (SP)
def cost_price : ℝ := 60
def selling_price : ℝ := 63

-- Define the profit as selling price minus cost price
def profit : ℝ := selling_price - cost_price

-- Define the profit percentage calculation
def profit_percentage : ℝ := (profit / cost_price) * 100

-- The statement to verify
theorem profit_percentage_is_five : profit_percentage = 5 := 
by
  sorry

end profit_percentage_is_five_l795_795602


namespace nearest_int_to_expr_l795_795168

theorem nearest_int_to_expr : 
  |(3 + Real.sqrt 2)^6 - 3707| < 1 :=
by 
  sorry

end nearest_int_to_expr_l795_795168


namespace min_value_of_expression_l795_795461

theorem min_value_of_expression (x y z : ℝ) (hx : -0.5 ≤ x ∧ x ≤ 1) 
  (hy : -0.5 ≤ y ∧ y ≤ 1) (hz : -0.5 ≤ z ∧ z ≤ 1) : 
  let f := (3 / ((1 - x) * (1 - y) * (1 - z))) + (3 / ((1 + x) * (1 + y) * (1 + z))) in
  f ≥ 6 := sorry

end min_value_of_expression_l795_795461


namespace integral_binomial_condition_l795_795766

theorem integral_binomial_condition
  (a : ℝ) (h1 : a > 0) 
  (h2 : (Coeff (expand (λ x, (a * x^2 - 1 / sqrt x)^6)) 2) = 60) :
  ∫ x in -1..2, (x^2 - 2*x) = 0 :=
by 
  sorry

end integral_binomial_condition_l795_795766


namespace solve_equation_l795_795700

theorem solve_equation :
  ∀ x y : ℝ, (x + y)^2 = (x + 1) * (y - 1) → x = -1 ∧ y = 1 :=
by
  intro x y
  sorry

end solve_equation_l795_795700


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795979

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795979


namespace cos_beta_value_l795_795599

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (hα_cos : Real.cos α = 4 / 5) (hαβ_cos : Real.cos (α + β) = -16 / 65) : 
  Real.cos β = 5 / 13 := 
sorry

end cos_beta_value_l795_795599


namespace mean_value_of_quadrilateral_angles_l795_795972

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795972


namespace mean_value_of_quadrilateral_angles_l795_795971

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795971


namespace num_possible_multisets_l795_795872

-- Define the coefficient constraints
variables {b_0 b_1 b_2 b_3 b_4 b_5 : ℤ}

-- Define the polynomials p(x) and q(x) with integer coefficients
def p (x : ℂ) : ℂ := b_5 * x^5 + b_4 * x^4 + b_3 * x^3 + b_2 * x^2 + b_1 * x + b_0
def q (x : ℂ) : ℂ := b_0 * x^5 + b_1 * x^4 + b_2 * x^3 + b_3 * x^2 + b_4 * x + b_5

-- Define the roots of the polynomials; assume integer and unit imaginary roots
variables {s_1 s_2 s_3 s_4 s_5 : ℂ}

-- Define the roots condition
def roots_condition : Prop :=
  (s_1 = 1 ∨ s_1 = -1 ∨ s_1 = complex.I ∨ s_1 = -complex.I) ∧
  (s_2 = 1 ∨ s_2 = -1 ∨ s_2 = complex.I ∨ s_2 = -complex.I) ∧
  (s_3 = 1 ∨ s_3 = -1 ∨ s_3 = complex.I ∨ s_3 = -complex.I) ∧
  (s_4 = 1 ∨ s_4 = -1 ∨ s_4 = complex.I ∨ s_4 = -complex.I) ∧
  (s_5 = 1 ∨ s_5 = -1 ∨ s_5 = complex.I ∨ s_5 = -complex.I)

-- State the equality of the number of possible multisets
theorem num_possible_multisets (h : roots_condition) : 
  {s_1, s_2, s_3, s_4, s_5}.card = 42 :=
by
  sorry 

end num_possible_multisets_l795_795872


namespace final_cost_of_pens_l795_795239

noncomputable def cost_of_pens (first_dozen_cost second_dozen_cost remaining_half_dozen_cost : ℝ)
  (first_dozen_discount second_dozen_discount sales_tax_rate : ℝ) : ℝ :=
let first_dozen_discounted := first_dozen_cost * (1 - first_dozen_discount / 100) in
let second_dozen_discounted := second_dozen_cost * (1 - second_dozen_discount / 100) in
let total_cost_before_tax := first_dozen_discounted + second_dozen_discounted + remaining_half_dozen_cost in
let total_cost_with_tax := total_cost_before_tax * (1 + sales_tax_rate / 100) in
total_cost_with_tax

theorem final_cost_of_pens :
  cost_of_pens 18 16 10 10 15 8 = 44.064 :=
begin
  -- omitted proof steps for brevity
  sorry
end

end final_cost_of_pens_l795_795239


namespace perimeter_of_triangle_l795_795134

theorem perimeter_of_triangle (a b : ℝ) (hx : a = 3) (hy : b = 4)
  (hc : ∃ x, x^2 - 10 * x + 16 = 0 ∧ (x = 2 ∨ x = 8)) :
  ∃ s, (s = {a, b, c} ∧ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) ∧ perimeter = a + b + c ∧ perimeter = 9) := 
sorry

end perimeter_of_triangle_l795_795134


namespace cylinder_volume_calculation_l795_795617

-- Assuming initial conditions provided in the problem
def initial_water_level : ℝ := 30
def final_water_level : ℝ := 35
def cylinder_min_mark : ℝ := 15
def cylinder_max_mark : ℝ := 45

-- Define the proof statement that needs to be established
theorem cylinder_volume_calculation :
  final_water_level - initial_water_level = 5 →
  cylinder_max_mark - cylinder_min_mark = 30 →
  ((final_water_level - cylinder_min_mark) / (cylinder_max_mark - cylinder_min_mark)) * (final_water_level - initial_water_level) = 7.5 :=
by
  intros h1 h2,
  sorry

end cylinder_volume_calculation_l795_795617


namespace meaningful_expression_range_l795_795020

theorem meaningful_expression_range (x : ℝ) : (∃ (y : ℝ), y = 5 / (x - 2)) ↔ x ≠ 2 := 
by
  sorry

end meaningful_expression_range_l795_795020


namespace estate_problem_l795_795483

def totalEstateValue (E a b : ℝ) : Prop :=
  (a + b = (3/5) * E) ∧ 
  (a = 2 * b) ∧ 
  (3 * b = (3/5) * E) ∧ 
  (E = a + b + (3 * b) + 4000)

theorem estate_problem (E : ℝ) (a b : ℝ) :
  totalEstateValue E a b → E = 20000 :=
by
  -- The proof will be filled here
  sorry

end estate_problem_l795_795483


namespace domain_f_l795_795686

-- Define the function f(x)
def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 7*x^2 - 4*x + 2) / (x^3 - 4*x)

-- Define the domain of f(x)
def domain_of_f (x : ℝ) : Prop := x ≠ 0 ∧ x ≠ 2 ∧ x ≠ -2

-- Proof statement that domain of f is equivalent to given intervals
theorem domain_f:
  ∀ x : ℝ, domain_of_f x ↔ (x ∈ set.Ioo (-(⊤ : ℝ)) (-2) ∪ set.Ioo (-2) 0 ∪ set.Ioo 0 2 ∪ set.Ioo 2 (⊤ : ℝ)) :=
by
  sorry

end domain_f_l795_795686


namespace four_digit_numbers_with_specific_digits_l795_795136

-- Definitions for the problem
def is_four_digit_number (n: ℕ): Prop :=
  1000 ≤ n ∧ n ≤ 9999

def contains_digits_0_1_2 (n: ℕ): Prop :=
  ∀ d ∈ (nat.digits 10 n), d = 0 ∨ d = 1 ∨ d = 2

def has_two_identical_digits (n: ℕ): Prop :=
  (∃ d, d = 1 ∧ count (nat.digits 10 n) d = 2) ∨ (∃ d, d = 2 ∧ count (nat.digits 10 n) d = 2)

-- Main theorem statement
theorem four_digit_numbers_with_specific_digits : 
  {n : ℕ | is_four_digit_number n ∧ contains_digits_0_1_2 n ∧ has_two_identical_digits n}.card = 18 :=
by
  sorry

end four_digit_numbers_with_specific_digits_l795_795136


namespace nearest_integer_to_expression_correct_l795_795172

noncomputable def nearest_integer_to_expression : ℤ :=
  Int.floor ((3 + Real.sqrt 2) ^ 6)

theorem nearest_integer_to_expression_correct : nearest_integer_to_expression = 7414 :=
by
  sorry

end nearest_integer_to_expression_correct_l795_795172


namespace max_S_n_value_arithmetic_sequence_l795_795340

-- Definitions and conditions
def S_n (n : ℕ) : ℤ := 3 * n - n^2

def a_n (n : ℕ) : ℤ := 
if n = 0 then 0 else S_n n - S_n (n - 1)

-- Statement of the first part of the proof problem
theorem max_S_n_value (n : ℕ) (h : n = 1 ∨ n = 2) : S_n n = 2 :=
sorry

-- Statement of the second part of the proof problem
theorem arithmetic_sequence :
  ∀ n : ℕ, n ≥ 1 → a_n (n + 1) - a_n n = -2 :=
sorry

end max_S_n_value_arithmetic_sequence_l795_795340


namespace probability_event_A_is_3_over_8_l795_795217

-- Define the conditions and the question
def tetrahedron_faces : List ℕ := [0, 1, 2, 3]

def total_outcomes : ℕ := (tetrahedron_faces.length) ^ 2

def event_A (m n : ℕ) : Bool := (m^2 + n^2) ≤ 4

-- Calculate the number of successful outcomes
def successful_outcomes : ℕ :=
  List.foldl (λ acc (pair : ℕ × ℕ) => 
    if event_A pair.1 pair.2 then acc + 1 else acc) 
    0 
    ((List.product tetrahedron_faces tetrahedron_faces))

-- Define the probability of event A occurring
def probability_event_A : ℚ := 
  successful_outcomes.to_nat / total_outcomes.to_nat

theorem probability_event_A_is_3_over_8 :
  probability_event_A = 3/8 := by
  sorry

end probability_event_A_is_3_over_8_l795_795217


namespace range_of_x0_l795_795474

theorem range_of_x0 (x0 : ℝ)
  (hM : ∃ (N : ℝ × ℝ), N.1^2 + N.2^2 = 1 ∧ 
  ∃ θ : ℝ, θ = (1 : ℝ / 6) * Real.pi ∧ 
  ∃ (x0 : ℝ), (0 : ℝ ≤ x0 ∧ x0 ≤ 2 - θ)) : 
  (0 ≤ x0) ∧ (x0 ≤ 2) := 
sorry

end range_of_x0_l795_795474


namespace figure_C_perimeter_l795_795881

def is_perimeter (figure : Type) (perimeter : ℕ) : Prop :=
∃ x y : ℕ, (figure = 'A' → 6*x + 2*y = perimeter) ∧ 
           (figure = 'B' → 4*x + 6*y = perimeter) ∧
           (figure = 'C' → 2*x + 6*y = perimeter)

theorem figure_C_perimeter (hA : is_perimeter 'A' 56) (hB : is_perimeter 'B' 56) : 
  is_perimeter 'C' 40 :=
by
  sorry

end figure_C_perimeter_l795_795881


namespace select_athlete_is_D_l795_795645

structure Athlete :=
  (name : String)
  (average_score : Float)
  (variance : Float)

def A := Athlete.mk "A" 9.7 0.035
def B := Athlete.mk "B" 9.6 0.042
def C := Athlete.mk "C" 9.5 0.036
def D := Athlete.mk "D" 9.7 0.015

def selected_athlete (a b c d : Athlete) : Athlete :=
  if a.average_score = d.average_score ∧ d.average_score > b.average_score ∧ d.average_score > c.average_score ∧ d.variance < a.variance ∧ d.variance < b.variance ∧ d.variance < c.variance
  then d
  else if a.average_score > b.average_score ∧ a.average_score > c.average_score ∧ a.average_score > d.average_score ∧ a.variance < b.variance ∧ a.variance < c.variance
  then a
  else if b.average_score > c.average_score ∧ b.average_score > d.average_score
  then b
  else c

theorem select_athlete_is_D : selected_athlete A B C D = D :=
  by
    sorry

end select_athlete_is_D_l795_795645


namespace points_on_curve_is_parabola_l795_795857

theorem points_on_curve_is_parabola (X Y : ℝ) (h : Real.sqrt X + Real.sqrt Y = 1) :
  ∃ a b c : ℝ, Y = a * X^2 + b * X + c :=
sorry

end points_on_curve_is_parabola_l795_795857


namespace gcd_factorial_l795_795316

-- Definitions and conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

-- Theorem statement
theorem gcd_factorial : gcd (factorial 8) (factorial 10) = factorial 8 := by
  -- The proof is omitted
  sorry

end gcd_factorial_l795_795316


namespace sum_zero_l795_795828

noncomputable def f : ℝ → ℝ := sorry

theorem sum_zero :
  (∀ x : ℝ, f (-x) = -f x) →
  (∀ x : ℝ, f (x + 5) = f x) →
  f (1 / 3) = 1 →
  f (16 / 3) + f (29 / 3) + f 12 + f (-7) = 0 :=
by
  intros hodd hperiod hvalue
  sorry

end sum_zero_l795_795828


namespace ratio_of_ages_l795_795949

-- Define the conditions and the main proof goal
theorem ratio_of_ages (R J : ℕ) (Tim_age : ℕ) (h1 : Tim_age = 5) (h2 : J = R + 2) (h3 : J = Tim_age + 12) :
  R / Tim_age = 3 := 
by
  sorry

end ratio_of_ages_l795_795949


namespace vendor_apples_l795_795626

theorem vendor_apples : 
  ∀ (n : ℕ) (x : ℕ),
    let apples_left_first_day := n - (n * 60) / 100,
    let apples_left_second_day := apples_left_first_day - (apples_left_first_day * x) / 100,
    let total_thrown_away := (n * 28) / 100 in
    apples_left_first_day * x / 100 + (apples_left_second_day * 50) / 100 = total_thrown_away →
    x = 40 :=
by
  intros n x apples_left_first_day apples_left_second_day total_thrown_away h
  sorry

end vendor_apples_l795_795626


namespace find_min_fraction_l795_795143

def smallest_element (s : Set ℚ) : ℚ :=
  if h : s.nonempty
  then Classical.choose (s.exists_has_min (Classical.inhabited_of_nonempty h))
  else 0  -- This hypothetical case is irrelevant for nonempty sets.

theorem find_min_fraction :
  smallest_element {1/2, 2/3, 1/4, 5/6, 7/12} = 1/4 := by
  sorry

end find_min_fraction_l795_795143


namespace midpoint_probability_sum_equals_2881_l795_795470

def T : Set (ℤ × ℤ × ℤ) :=
  { p | (0 ≤ p.1 ∧ p.1 ≤ 3) ∧ (0 ≤ p.2.1 ∧ p.2.1 ≤ 2) ∧ (0 ≤ p.2.2 ∧ p.2.2 ≤ 5) }

lemma probability_midpoint_T (x y z : ℤ) (hx : 0 ≤ x ∧ x ≤ 3) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 5) :
  let num_pairs := 8 * 5 * 17 - 72 in
  let total_pairs := 72 * 71 in
  let probability := (num_pairs.to_nat, total_pairs.to_nat) in
  probability = (325, 2556) :=
by 
  sorry

theorem midpoint_probability_sum_equals_2881 : 
  (325 + 2556) = 2881 :=
by 
  rfl

end midpoint_probability_sum_equals_2881_l795_795470


namespace matrix_det_eq_six_l795_795672

theorem matrix_det_eq_six (x : ℝ) :
  let a := 3 * x in
  let b := 4 * x in
  let c := 2 in
  let d := 2 * x in
  (a * b - c * d) = 6 ↔ (x = -(1/3) ∨ x = 3/2) := 
  by
  intro a b c d
  simp
  apply sorry

end matrix_det_eq_six_l795_795672


namespace nearest_integer_to_power_l795_795164

theorem nearest_integer_to_power (a b : ℝ) (h1 : a = 3) (h2 : b = sqrt 2) : 
  abs ((a + b)^6 - 3707) < 0.5 :=
by
  sorry

end nearest_integer_to_power_l795_795164


namespace miquel_theorem_l795_795596

open EuclideanGeometry

/-- Miquel's theorem -/
theorem miquel_theorem 
  (ABC : Triangle) 
  (D E F : Point)
  (hD : D ∈ (segment ABC.BC))
  (hE : E ∈ (segment ABC.CA))
  (hF : F ∈ (segment ABC.AB)) 
  : ∃ G : Point, is_on_circumcircle G (circumcircle A E F) ∧ 
                 is_on_circumcircle G (circumcircle B D F) ∧ 
                 is_on_circumcircle G (circumcircle C D E) := 
sorry

end miquel_theorem_l795_795596


namespace committee_probability_l795_795123

variable (B G : ℕ) -- Number of boys and girls
variable (n : ℕ) -- Size of the committee 
variable (N : ℕ) -- Total number of members

theorem committee_probability 
  (h_B : B = 12) 
  (h_G : G = 8) 
  (h_N : N = 20) 
  (h_n : n = 4) : 
  (1 - (Nat.choose 12 4 + Nat.choose 8 4) / Nat.choose 20 4) = 4280 / 4845 := 
by sorry

end committee_probability_l795_795123


namespace sin_720_eq_zero_l795_795250

theorem sin_720_eq_zero : Real.sin (720 * Real.pi / 180) = 0 := by
  have h1 : 720 * Real.pi / 180 = 4 * Real.pi := by
    simp [← mul_div_assoc, mul_comm]
  rw [h1, Real.sin_mul_pi]
  simp

end sin_720_eq_zero_l795_795250


namespace volume_of_prism_main_theorem_l795_795218

/-- Define the main conditions in Lean 4 syntax. --/

-- Condition: A unit cube with side length 1.
def unit_cube := { x | 0 ≤ x.1 ∧ x.1 ≤ 1 ∧ 0 ≤ x.2 ∧ x.2 ≤ 1 ∧ 0 ≤ x.3 ∧ x.3 ≤ 1 }

-- Question and condition combined: Volume of the triangular prism including vertex W 
theorem volume_of_prism (W : unit_cube) : 
  (W = (1, 0, 0) ∨ W = (0, 1, 1)) ∨ 
  (W = (1, 0, 1) ∨ W = (0, 0, 0)) ∨ 
  (W = (1, 1, 0) ∨ W = (0, 1, 0)) →
  1 / 8 :=
sorry

/-- Main Theorem: Given the unit cube and conditions of the cuts, 
prove the volume of the relevant triangular prism is 1/8. --/
theorem main_theorem : 
  ∀ (W : unit_cube), (W = (1, 0, 0) ∨ W = (0, 1, 1)) ∨ 
                     (W = (1, 0, 1) ∨ W = (0, 0, 0)) ∨ 
                     (W = (1, 1, 0) ∨ W = (0, 1, 0)) → 
  volume_of_prism W :=
sorry

end volume_of_prism_main_theorem_l795_795218


namespace largest_prime_factor_of_given_numbers_l795_795581

noncomputable def factors (n : ℕ) : List ℕ := sorry

noncomputable def largest_prime_factor (n : ℕ) : ℕ :=
  (factors n).filter (λ x => Nat.Prime x) |>.maximum

theorem largest_prime_factor_of_given_numbers :
  largest_prime_factor 57 = 19 ∧
  largest_prime_factor 63 = 7 ∧
  largest_prime_factor 143 = 13 ∧
  largest_prime_factor 169 = 13 ∧
  largest_prime_factor 231 = 11 ∧
  ∀ n ∈ [57, 63, 143, 169, 231], largest_prime_factor n ≤ largest_prime_factor 57 :=
by
  sorry

end largest_prime_factor_of_given_numbers_l795_795581


namespace problem_solution_l795_795104

noncomputable def calculateLengthSum : Real :=
  let n := 200
  let AB := 5
  let CB := 4
  let AC := Real.sqrt (AB^2 + CB^2)
  let segmentLength (k : Nat) : Real := AC * (n - k) / n
  let totalLength := 2 * (Finset.sum (Finset.range (n + 1)) segmentLength) - AC
  totalLength

theorem problem_solution : calculateLengthSum = 199 * Real.sqrt 41 := by
  sorry

end problem_solution_l795_795104


namespace find_b_l795_795730

-- Definitions from the problem's conditions
variables {n : ℕ} {a0 a1 a2 : ℝ} {b : ℝ}

axiom eqn : ∀ x : ℝ, b * x^n + 1 = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + ∑ i in finset.range (n - 2 + 1), (a (i+2) * (x - 1)^(i+2))
axiom a1_ne_9 : a1 ≠ 9
axiom a2_ne_36 : a2 ≠ 36

-- The goal to prove
theorem find_b : b = 1 := 
sorry

end find_b_l795_795730


namespace dot_product_correct_l795_795290

def v : ℝ × ℝ × ℝ := (4, -3, 2)
def u : ℝ × ℝ × ℝ := (-3, 6, -4)
def k : ℝ := 2

def scaled_v : ℝ × ℝ × ℝ := (k * v.1, k * v.2, k * v.3)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2 + a.3 * b.3

theorem dot_product_correct : dot_product scaled_v u = -76 := by
  sorry

end dot_product_correct_l795_795290


namespace find_coordinates_of_M_l795_795436

theorem find_coordinates_of_M
  (a : ℝ)
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)
  (M_coords : M = (a-3, a+4))
  (N_coords : N = (real.sqrt 5, 9))
  (MN_parallel_y_axis : M.1 = N.1) :
  M = (real.sqrt 5, 7 + real.sqrt 5) :=
by
  sorry

end find_coordinates_of_M_l795_795436


namespace limit_example_eq_2_l795_795240

noncomputable def limit_example : ℕ → ℝ := λ n, (2 * n + 1) / (n - 1)

theorem limit_example_eq_2 : filter.tendsto limit_example filter.at_top (nhds 2) :=
sorry

end limit_example_eq_2_l795_795240


namespace max_area_rect_l795_795542

theorem max_area_rect (x y : ℝ) (h_perimeter : 2 * x + 2 * y = 40) : 
  x * y ≤ 100 :=
by
  sorry

end max_area_rect_l795_795542


namespace sixth_term_of_geometric_sequence_l795_795877

theorem sixth_term_of_geometric_sequence (a r : ℝ) (h1 : a * r^3 = 16) (h2 : a * r^8 = 8) : a * r^5 = 16 * real.root 4 5 :=
by
  sorry

end sixth_term_of_geometric_sequence_l795_795877


namespace coefficient_b_non_zero_l795_795130

noncomputable def Q (a b c d e : ℝ) : ℝ[X] :=
  X ^ 5 + a * X ^ 4 + b * X ^ 3 + c * X ^ 2 + d * X + e

theorem coefficient_b_non_zero
  (a b c d e p q r : ℝ)
  (hQ : Q a b c d e = X ^ 2 * (X - p) * (X - q) * (X - r))
  (hdist : p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0) :
  b ≠ 0 :=
sorry

end coefficient_b_non_zero_l795_795130


namespace roots_numerically_equal_but_opposite_signs_l795_795008

noncomputable def value_of_m (a b c : ℝ) : ℝ := (a - b) / (a + b)

theorem roots_numerically_equal_but_opposite_signs
  (a b c m : ℝ)
  (h : ∀ x : ℝ, (a ≠ 0 ∧ a + b ≠ 0) ∧ (x^2 - b*x = (ax - c) * (m - 1) / (m + 1))) 
  (root_condition : ∃ x₁ x₂ : ℝ, x₁ = -x₂ ∧ x₁ * x₂ != 0) :
  m = value_of_m a b c :=
by
  sorry

end roots_numerically_equal_but_opposite_signs_l795_795008


namespace sum_of_2020_digit_number_l795_795846

def divisible_by_eleven (n : ℕ) : Prop :=
  (n % 11 = 0)

def four_digit_extracted (l : List ℕ) : Prop :=
  ∀ k, 0 ≤ k ∧ k + 3 < l.length → divisible_by_eleven ((l.nth_le k sorry) * 1000 + (l.nth_le (k+1) sorry) * 100 + (l.nth_le (k+2) sorry) * 10 + (l.nth_le (k+3) sorry))

def sum_of_digits (l : List ℕ) : ℕ :=
  l.foldr (· + ·) 0

theorem sum_of_2020_digit_number (l : List ℕ) (hl : l.length = 2020) (h : four_digit_extracted l) : sum_of_digits l = 11110 :=
sorry

end sum_of_2020_digit_number_l795_795846


namespace ellipse_equation_fixed_points_l795_795636

theorem ellipse_equation_fixed_points (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (P : ℝ × ℝ) (hP : P = (4/3, b/3)) 
  (h_ellipse : ∀ x y, (x,y) ∈ (set_of (λ p : ℝ × ℝ, p.1^2 / a^2 + p.2^2 / b^2 = 1) )) 
  (hA : (0, b) ∈ (set_of (λ p : ℝ × ℝ, p.1^2 / a^2 + p.2^2 / b^2 = 1) )) :

  (∃ a b : ℝ, a^2 = 2 ∧ b^2 = 1 ∧
  (set_of (λ p : ℝ × ℝ, p.1^2 / a^2 + p.2^2 / b^2 = 1)) = 
  { p : ℝ × ℝ | p.1^2 / 2 + p.2^2 = 1 })

  ∧ (∃ M1 M2 : ℝ × ℝ, M1 = (1, 0) ∧ M2 = (-1, 0) ∧
  ∀ l : ℝ × ℝ → Prop, 
  (∃ k m, l = set_of (λ x : ℝ × ℝ, x.2 = k * x.1 + m) ∧ 
  ∃ p : ℝ × ℝ, p ∈ (set_of (λ p : ℝ × ℝ, p.1^2 / 2 + p.2^2 = 1) ∧ 
  ∀ p : ℝ × ℝ, (p.1, p.2) ∈ l → (abs (p.1 - M1.1) * abs (p.1 - M2.1) = 1))) :

  true := sorry

end ellipse_equation_fixed_points_l795_795636


namespace b_should_pay_l795_795592

def TotalRent : ℕ := 725
def Cost_a : ℕ := 12 * 8 * 5
def Cost_b : ℕ := 16 * 9 * 6
def Cost_c : ℕ := 18 * 6 * 7
def Cost_d : ℕ := 20 * 4 * 4
def TotalCost : ℕ := Cost_a + Cost_b + Cost_c + Cost_d
def Payment_b (Cost_b TotalCost TotalRent : ℕ) : ℕ := (Cost_b * TotalRent) / TotalCost

theorem b_should_pay :
  Payment_b Cost_b TotalCost TotalRent = 259 := 
  by
  unfold Payment_b
  -- Leaving the proof body empty as per instructions
  sorry

end b_should_pay_l795_795592


namespace nearest_int_to_expr_l795_795166

theorem nearest_int_to_expr : 
  |(3 + Real.sqrt 2)^6 - 3707| < 1 :=
by 
  sorry

end nearest_int_to_expr_l795_795166


namespace remainder_of_70th_term_modulo_6_l795_795198

noncomputable def sequence : ℕ → ℕ
| 0       := 0
| 1       := 1
| 2       := 3
| 3       := 8
| 4       := 21
| (n + 1) := 3 * (sequence n) - (sequence (n - 1))

theorem remainder_of_70th_term_modulo_6 : (sequence 69) % 6 = 4 :=
by
  sorry

end remainder_of_70th_term_modulo_6_l795_795198


namespace larger_integer_l795_795953

theorem larger_integer (a b : ℕ) (h_diff : a - b = 8) (h_prod : a * b = 224) : a = 16 :=
by
  sorry

end larger_integer_l795_795953


namespace sum_of_arithmetic_sequence_15_l795_795030

def arithmetic_sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  ∑ i in finset.range n, a i

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

theorem sum_of_arithmetic_sequence_15 (a : ℕ → ℕ) (h_arith_seq: is_arithmetic_sequence a) 
  (h_a8: a 8 = 8) : arithmetic_sequence_sum a 15 = 120 :=
by
  sorry

end sum_of_arithmetic_sequence_15_l795_795030


namespace find_area_of_triangle_ABC_l795_795050

-- Definitions of points and distances in triangle
variables {A B C P Q R S : Type}
variables [metric_space A] [metric_space B] [metric_space C]
variables [metric_space P] [metric_space Q] [metric_space R] [metric_space S]
variables [hab : {x : ℝ // x = 13}] [hac : {x : ℝ // x = 14}] [hpq : {x : ℝ // x = 7}] [hqr : {x : ℝ // x = 6}] [hrs : {x : ℝ // x = 8}]
variables [coll_P_Q_R_S : collinear P Q R S]

noncomputable def area_of_ABC : ℝ := 
  let s := (hab + hac + 15) / 2 in
  real.sqrt (s * (s - hab) * (s - hac) * (s - 15))

theorem find_area_of_triangle_ABC :
  area_of_ABC = 84 := by
  sorry

end find_area_of_triangle_ABC_l795_795050


namespace usb_drive_total_capacity_l795_795088

-- Define the conditions as α = total capacity, β = busy space (50%), γ = available space (50%)
variable (α : ℕ) -- Total capacity of the USB drive in gigabytes
variable (β γ : ℕ) -- Busy space and available space in gigabytes
variable (h1 : β = α / 2) -- 50% of total capacity is busy
variable (h2 : γ = 8)  -- 8 gigabytes are still available

-- Define the problem as a theorem that these conditions imply the total capacity
theorem usb_drive_total_capacity (h : γ = α / 2) : α = 16 :=
by
  -- defer the proof
  sorry

end usb_drive_total_capacity_l795_795088


namespace ribbon_left_after_wrapping_l795_795853

def total_ribbon_needed (gifts : ℕ) (ribbon_per_gift : ℝ) : ℝ :=
  gifts * ribbon_per_gift

def remaining_ribbon (initial_ribbon : ℝ) (used_ribbon : ℝ) : ℝ :=
  initial_ribbon - used_ribbon

theorem ribbon_left_after_wrapping : 
  ∀ (gifts : ℕ) (ribbon_per_gift initial_ribbon : ℝ),
  gifts = 8 →
  ribbon_per_gift = 1.5 →
  initial_ribbon = 15 →
  remaining_ribbon initial_ribbon (total_ribbon_needed gifts ribbon_per_gift) = 3 :=
by
  intros gifts ribbon_per_gift initial_ribbon h1 h2 h3
  rw [h1, h2, h3]
  simp [total_ribbon_needed, remaining_ribbon]
  sorry

end ribbon_left_after_wrapping_l795_795853


namespace wanda_crayons_l795_795961

variable (Dina Jacob Wanda : ℕ)

theorem wanda_crayons : Dina = 28 ∧ Jacob = Dina - 2 ∧ Dina + Jacob + Wanda = 116 → Wanda = 62 :=
by
  intro h
  sorry

end wanda_crayons_l795_795961


namespace mean_value_of_quadrilateral_angles_l795_795962

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795962


namespace integer_root_of_polynomial_l795_795137

theorem integer_root_of_polynomial (a b : ℚ) (h : (4 - 2*Real.sqrt 5) ∈ (polynomial.C(_) : polynomial ℝ).roots) :
  ∃ x : ℤ, x = -8 :=
by
  sorry

end integer_root_of_polynomial_l795_795137


namespace solution_set_x_l795_795364

def f (x : ℝ) : ℝ :=
  if 0 < x then log x / log (1 / 2) else -x^2 - 2 * x

theorem solution_set_x (x : ℝ) : f x ≤ 0 ↔ x ≥ 1 ∨ x = 0 ∨ x ≤ -2 := 
  sorry

end solution_set_x_l795_795364


namespace steves_earning_l795_795115

variable (pounds_picked : ℕ → ℕ) -- pounds picked on day i: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday

def payment_per_pound : ℕ := 2

def total_money_made : ℕ :=
  (pounds_picked 0 * payment_per_pound) + 
  (pounds_picked 1 * payment_per_pound) + 
  (pounds_picked 2 * payment_per_pound) + 
  (pounds_picked 3 * payment_per_pound)

theorem steves_earning 
  (h0 : pounds_picked 0 = 8)
  (h1 : pounds_picked 1 = 3 * pounds_picked 0)
  (h2 : pounds_picked 2 = 0)
  (h3 : pounds_picked 3 = 18) : 
  total_money_made pounds_picked = 100 := by
  sorry

end steves_earning_l795_795115


namespace cyclic_permutations_divisible_by_27_l795_795600

theorem cyclic_permutations_divisible_by_27 (n : Nat) (h : n % 3 = 0) (digits : Fin n → Fin 10)
  (H : Nat.fromDigits 10 (digits ∘ Fin.succ) % 27 = 0) :
  ∀ shift : Nat, (Nat.fromDigits 10 (digits ∘ (λ i => (i + shift) % n)) % 27 = 0) := by
  sorry

end cyclic_permutations_divisible_by_27_l795_795600


namespace no_intersection_l795_795048

def M := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }
def N (a : ℝ) := { p : ℝ × ℝ | abs (p.1 - 1) + abs (p.2 - 1) = a }

theorem no_intersection (a : ℝ) : M ∩ (N a) = ∅ ↔ a ∈ (Set.Ioo (2-Real.sqrt 2) (2+Real.sqrt 2)) := 
by 
  sorry

end no_intersection_l795_795048


namespace minimum_value_of_f_l795_795331
noncomputable theory

def f (a b x : ℝ) : ℝ := a * x ^ 3 + b * x ^ 9 + 2

theorem minimum_value_of_f (a b : ℝ) (h : (∀ x > 0, f a b x ≤ 5)) :
  ∃ c < 0, f a b c = -1 :=
sorry

end minimum_value_of_f_l795_795331


namespace log_condition_l795_795009

open Real

theorem log_condition (x : ℝ) (h1 : 3 * x > 0) (h2 : log (3 * x) 729 = x) : x = 3 ∧ ¬∃ m : ℤ, x = m^2 ∧ ¬∃ n : ℤ, x = n^3 :=
by
  sorry

end log_condition_l795_795009


namespace sum_of_D_coordinates_l795_795098

noncomputable def sum_of_coordinates_of_D (D : ℝ × ℝ) (M C : ℝ × ℝ) : ℝ :=
  D.1 + D.2

theorem sum_of_D_coordinates (D M C : ℝ × ℝ) (H_M_midpoint : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) 
                             (H_M_value : M = (5, 9)) (H_C_value : C = (11, 5)) : 
                             sum_of_coordinates_of_D D M C = 12 :=
sorry

end sum_of_D_coordinates_l795_795098


namespace most_appropriate_survey_is_D_l795_795586

-- Define the various scenarios as Lean definitions
def survey_A := "Testing whether a certain brand of fresh milk meets food hygiene standards, using a census method."
def survey_B := "Security check before taking the subway, using a sampling survey method."
def survey_C := "Understanding the sleep time of middle school students in Jiangsu Province, using a census method."
def survey_D := "Understanding the way Nanjing residents commemorate the Qingming Festival, using a sampling survey method."

-- Define the type for specifying which survey method is the most appropriate
def appropriate_survey (survey : String) : Prop := 
  survey = survey_D

-- The theorem statement proving that the most appropriate survey is D
theorem most_appropriate_survey_is_D : appropriate_survey survey_D :=
by sorry

end most_appropriate_survey_is_D_l795_795586


namespace evaluate_expression_l795_795176

theorem evaluate_expression :
  3 ^ (1 ^ (2 ^ 8)) + ((3 ^ 1) ^ 2) ^ 8 = 43046724 := 
by
  sorry

end evaluate_expression_l795_795176


namespace number_of_plain_lemonade_sold_l795_795236

theorem number_of_plain_lemonade_sold
  (price_per_plain_lemonade : ℝ)
  (earnings_strawberry_lemonade : ℝ)
  (earnings_more_plain_than_strawberry : ℝ)
  (P : ℝ)
  (H1 : price_per_plain_lemonade = 0.75)
  (H2 : earnings_strawberry_lemonade = 16)
  (H3 : earnings_more_plain_than_strawberry = 11)
  (H4 : price_per_plain_lemonade * P = earnings_strawberry_lemonade + earnings_more_plain_than_strawberry) :
  P = 36 :=
by
  sorry

end number_of_plain_lemonade_sold_l795_795236


namespace problem_l795_795407

noncomputable def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem problem (f : ℝ → ℝ) (h : isOddFunction f) : 
  f (-2012) + f (-2011) + f 0 + f 2011 + f 2012 = 0 :=
by
  sorry

end problem_l795_795407


namespace cos_300_eq_half_l795_795659

theorem cos_300_eq_half : real.cos (300 * real.pi / 180) = 1 / 2 :=
by
  have h1 : real.cos (360 * real.pi / 180) = 1, from by sorry
  have h2 : real.sin (360 * real.pi / 180) = 0, from by sorry
  have h3 : real.cos (60 * real.pi / 180) = 1 / 2, from by sorry
  -- Use the identity cos (a - b) = cos a * cos b + sin a * sin b
  calc
    real.cos (300 * real.pi / 180)
        = real.cos (360 * real.pi / 180 - 60 * real.pi / 180) : by sorry
    ... = real.cos (360 * real.pi / 180) * real.cos (60 * real.pi / 180) + real.sin (360 * real.pi / 180) * real.sin (60 * real.pi / 180) : by sorry
    ... = 1 * (1 / 2) + 0 * real.sin (60 * real.pi / 180) : by sorry
    ... = 1 / 2 : by sorry

end cos_300_eq_half_l795_795659


namespace max_intersections_cubic_polynomials_l795_795160

theorem max_intersections_cubic_polynomials (f g : ℝ → ℝ)
  (hf : ∃ (a b c d : ℝ), f = λ x, 2 * x^3 + a * x^2 + b * x + c)
  (hg : ∃ (p q r s : ℝ), g = λ x, 2 * x^3 + p * x^2 + q * x + r) :
  ∃ (max_intersections : ℕ), max_intersections = 2 :=
begin
  -- since f and g are cubic polynomials with leading coefficient 2,
  -- subtraction f - g results in a polynomial of degree at most 2
  -- hence the maximum number of points of intersection is 2
  sorry
end

end max_intersections_cubic_polynomials_l795_795160


namespace find_distance_l795_795186

-- Definitions based on conditions
def speed : ℝ := 75 -- in km/hr
def time : ℝ := 4 -- in hr

-- Statement to be proved
theorem find_distance : speed * time = 300 := by
  sorry

end find_distance_l795_795186


namespace seating_arrangements_l795_795784

theorem seating_arrangements (n : ℕ) (alice bob charlie : ℕ) :
  n = 8 → charlie ∈ {0, n - 1} → (∃! i, i ∈ {0, n-1}) →
  ∃ (arrangements : ℕ), arrangements = 7200 :=
by
  intros h1 h2 h3
  let total := 2 * nat.factorial 7
  let restricted := 2 * nat.factorial 6 * nat.factorial 2
  let acceptable := total - restricted
  use acceptable
  simp [total, restricted, acceptable]
  rw [nat.factorial_succ, nat.factorial_succ, nat.factorial_succ]
  norm_num
  sorry

end seating_arrangements_l795_795784


namespace range_of_m_l795_795476

variable (a b : ℝ)

def S (n : ℕ) := ∫ x in 0..(n : ℝ), (2 * a * x + b)

theorem range_of_m {a b m : ℝ} : (∀ n > 0, ∀ {a_n : ℝ}, (a_n^2 + (S a b n)^2 / (n : ℝ)^2) ≥ m * (a + b)^2) ↔ m ≤ 1/5 := 
sorry

end range_of_m_l795_795476


namespace value_of_expression_l795_795147

theorem value_of_expression : 9^2 - real.sqrt 9 = 78 := 
by
  sorry

end value_of_expression_l795_795147


namespace formation_coloring_l795_795702

structure MobotFormation :=
  (north_oriented : list (ℕ × ℕ))
  (east_oriented : list (ℕ × ℕ))

def correct_formation : MobotFormation :=
  { north_oriented := [(0, 0), (1, 0), (2, 2), (3, 2)],
    east_oriented := [(2, 0), (2, 1)] }

-- The main statement asserting that the correct_formation has exactly 12 ways to color the mobots using three colors
noncomputable def color_ways (formation : MobotFormation) (colors : ℕ) : ℕ :=
  if formation = correct_formation ∧ colors = 3 then 12 else sorry

theorem formation_coloring :
  ∀ (formation : MobotFormation), (formation = correct_formation) → (color_ways formation 3 = 12) :=
by
  intros,
  sorry

end formation_coloring_l795_795702


namespace carbon_is_not_allotrope_of_C60_l795_795792

theorem carbon_is_not_allotrope_of_C60 :
  (∀ (X Y : Type) [Isotope X Y], same_chemical_properties X Y) ∧
  (∀ C N : Type, (mass_number C = mass_number N) -> same_mass_number C N) ∧
  (∀ (C12 C13 C14 : Type) [Isotope C12 C13] [Isotope C12 C14], same_element_isotopes C12 C13 C14) ∧
  (¬is_allotrope (Carbon 14) (C60))
  := sorry

end carbon_is_not_allotrope_of_C60_l795_795792


namespace raj_kitchen_area_l795_795499

theorem raj_kitchen_area :
  let house_area := 1110
  let bedrooms := 4
  let bedroom_width := 11
  let bedroom_length := 11
  let bathrooms := 2
  let bathroom_width := 6
  let bathroom_length := 8
  let rooms_area := (bedrooms * bedroom_width * bedroom_length) + (bathrooms * bathroom_width * bathroom_length)
  let shared_area := house_area - rooms_area
  let kitchen_area := shared_area / 2
  in kitchen_area = 265 :=
by
  sorry

end raj_kitchen_area_l795_795499


namespace sum_infinite_series_eq_five_sixteen_l795_795241

theorem sum_infinite_series_eq_five_sixteen : 
  let series := ∑' n : ℕ, (n + 1) * (1 / 5) ^ (n + 1) 
  in series = 5 / 16 :=
by
  sorry

end sum_infinite_series_eq_five_sixteen_l795_795241


namespace total_blue_balloons_l795_795452

theorem total_blue_balloons (joan_balloons : ℕ) (melanie_balloons : ℕ) (h_joan : joan_balloons = 40) (h_melanie : melanie_balloons = 41) : joan_balloons + melanie_balloons = 81 := by
  sorry

end total_blue_balloons_l795_795452


namespace count_valid_pairs_l795_795387

def is_valid_pair (a b : ℕ) : Prop :=
  b ∣ (5 * a - 3) ∧ a ∣ (5 * b - 1)

theorem count_valid_pairs : finset.card {ab : ℕ × ℕ | is_valid_pair ab.1 ab.2} = 18 :=
  sorry

end count_valid_pairs_l795_795387


namespace twice_x_greater_than_5_l795_795679

def x : ℝ

theorem twice_x_greater_than_5 : 2 * x > 5 :=
sorry

end twice_x_greater_than_5_l795_795679


namespace blue_line_segments_count_l795_795469

def A (x y : ℕ) : Prop := 1 ≤ x ∧ x ≤ 20 ∧ 1 ≤ y ∧ y ≤ 20
def B (x y : ℕ) : Prop := 2 ≤ x ∧ x ≤ 19 ∧ 2 ≤ y ∧ y ≤ 19

def is_red (x y : ℕ) : Prop := 
  (A x y ∧ (x, y) ≠ (1, 1) ∧ (x, y) ≠ (1, 20) ∧ (x, y) ≠ (20, 1) ∧ (x, y) ≠ (20, 20)) ∧
  if B x y then 1 ≤ 1 else 0

def is_blue (x y : ℕ) : Prop := 
  A x y ∧ ¬is_red x y

def black_line_segments : ℕ := 237

def total_points_A : ℕ := 20 * 20
def red_points_A : ℕ := 219
def blue_points_A : ℕ := 181

theorem blue_line_segments_count : 
  let blue_segments := 233
  blue_segments == 233 := sorry

end blue_line_segments_count_l795_795469


namespace inverses_eval_l795_795399

noncomputable def f (x : ℝ) : ℝ := 25 / (4 + 5 * x)
 
theorem inverses_eval :
  let f_inv_3 := (4 * 3 - 25) / (-5) in
  (f_inv_3) ^ (-3) = 3375 / 2197 :=
by
  -- Set up the equation 3 = 25 / (4 + 5 * f_inv_3)
  let f_inv_3 := (4 * 3 - 25) / (-5)
  -- Compute the expression (f_inv_3) ^ (-3) which must be 3375 / 2197
  have ex3: (f_inv_3) ^ (-3) = (15 / 13) ^ 3, by sorry
  show ex3 = 3375 / 2197, by sorry
  sorry

end inverses_eval_l795_795399


namespace sector_area_l795_795870

theorem sector_area (r : ℝ) (θ : ℝ) (chord_length : ℝ) :
  (2 * r * sin (θ / 2) = chord_length) → θ = 2 → chord_length = 2 → 
  (1 / 2 * r * r * θ = 1 / (sin 1)^2) :=
by
  intro h1 h2 h3
  sorry

end sector_area_l795_795870


namespace ada_unique_solids_make_l795_795627

-- Define the conditions
def identical_cubes (cubes : set Cube) : Prop := ∀ c1 c2 ∈ cubes, c1 = c2

def glued_solids (solids : set Solid) : Prop := 
  ∀ s ∈ solids, ∃ (cubes : set Cube), 
  (|cubes| = 4) ∧ 
  (∀ c ∈ cubes, ∃ c' ∈ cubes, cube_face_coincides c c') ∧
  s = glue_cubes cubes

-- Define the goal
def number_of_unique_solids (solids : set Solid) (n : ℕ) : Prop :=
  ∃ (unique_solids : set Solid), 
  (∀ s ∈ unique_solids, s ∈ solids) ∧
  (|unique_solids| = n) ∧
  (∀ s1 s2 ∈ unique_solids, s1 ≠ s2)

-- The theorem to prove
theorem ada_unique_solids_make : 
  ∀ (cubes : set Cube) (solids : set Solid),
    identical_cubes cubes →
    glued_solids solids →
    number_of_unique_solids solids 8 := sorry

end ada_unique_solids_make_l795_795627


namespace retailer_problem_l795_795187

theorem retailer_problem :
  ∀ (r : ℝ), 
  let wholesale_price := 90 in
  let profit := 0.20 * wholesale_price in
  let selling_price_after_discount := 0.90 * r in
  selling_price_after_discount = wholesale_price + profit →
  r = 120 :=
by
  intros r wholesale_price profit selling_price_after_discount h
  sorry

end retailer_problem_l795_795187


namespace lines_concurrent_at_circumcenter_l795_795721

variable {A B C H M N O : Type}
variables [decidable_eq A] [decidable_eq B] [decidable_eq C]
variables {triangle : Triangle A B C}
variables {foot : FootOfPerpendicular A B C H}
variables {circle : CircleWithDiameter A H}
variables {intersectAB : IntersectWithSide circle AB M}
variables {intersectAC : IntersectWithSide circle AC N}
variables {L_A : LineThroughPointPerpendicular A MN}
variables {L_B : LineThroughPointPerpendicular B}
variables {L_C : LineThroughPointPerpendicular C}
variables {circumcenter : IsCircumcenter O triangle}

theorem lines_concurrent_at_circumcenter
  (h_triangle_acute : IsAcuteTriangle triangle)
  (h_foot_of_perpendicular : foot)
  (h_diameter_intersections : circle)
  (h_AB_intersect : intersectAB)
  (h_AC_intersect : intersectAC)
  (h_LA_perpendicular : L_A)
  (h_LB_perpendicular : LineThroughPointPerpendicular B (−))
  (h_LC_perpendicular : LineThroughPointPerpendicular C (−))
  (h_circumcenter : circumcenter):
  ConcurrentLines L_A L_B L_C O := sorry

end lines_concurrent_at_circumcenter_l795_795721


namespace cos_half_sum_tan_sum_l795_795711

variables {α β : ℝ}

theorem cos_half_sum (h1 : cos (α - β/2) = -2 * sqrt 7 / 7)
                     (h2 : sin (α/2 - β) = 1/2)
                     (hα : π / 2 < α ∧ α < π)
                     (hβ : 0 < β ∧ β < π / 2) :
  cos ((α + β) / 2) = -sqrt 21 / 14 := 
sorry

theorem tan_sum (h1 : cos (α - β/2) = -2 * sqrt 7 / 7)
                (h2 : sin (α/2 - β) = 1/2)
                (hα : π / 2 < α ∧ α < π)
                (hβ : 0 < β ∧ β < π / 2) :
  tan (α + β) = 5 * sqrt 3 / 11 :=
sorry

end cos_half_sum_tan_sum_l795_795711


namespace expression_eval_l795_795643

theorem expression_eval : (3^2 - 3) - (5^2 - 5) * 2 + (6^2 - 6) = -4 :=
by sorry

end expression_eval_l795_795643


namespace alexey_game_max_score_l795_795227

theorem alexey_game_max_score :
  ∃ x : ℕ, 2017 ≤ x ∧ x ≤ 2117 ∧ 
  (∃ d3, x % 3 = 0) ∧
  (∃ d7, x % 7 = 0) ∧
  (∃ d9, x % 9 = 0) ∧
  (∃ d11, x % 11 = 0) ∧
  ((x % 5 = 0 → 30) ∧ (x % 5 ≠ 0 → 30)) :=
begin
  use 2079,
  split,
  { exact nat.le_refl 2079, },
  split,
  { norm_num, },
  split,
  { use 1, norm_num, },
  split,
  { use 297, norm_num, },
  split,
  { use 231, norm_num, },
  split,
  { use 189, norm_num, },
  split,
  { exact 30, },
  exact 30,
end

end alexey_game_max_score_l795_795227


namespace perimeter_of_C_l795_795923

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end perimeter_of_C_l795_795923


namespace find_fn_f_l795_795667

-- Define the existence of g and the condition for f
variable {f g : ℝ → ℝ}
variable h : ∀ x y : ℝ, f x + f y = floor (g (x + y))

-- Formalize the proof statement to find all such functions f
theorem find_fn_f (h : ∀ x y : ℝ, f x + f y = floor (g (x + y))) :
  ∃ n : ℤ, ∀ x : ℝ, f x = n / 2 := 
sorry

end find_fn_f_l795_795667


namespace problem_I_problem_II_l795_795741

noncomputable def f (a x : ℝ) : ℝ := x^2 - (2 * a + 1) * x + a * Real.log x
noncomputable def g (a x : ℝ) : ℝ := (1 - a) * x
noncomputable def h (x : ℝ) : ℝ := (x^2 - 2 * x) / (x - Real.log x)

theorem problem_I (a : ℝ) (ha : a > 1 / 2) :
  (∀ x : ℝ, 0 < x ∧ x < 1 / 2 → deriv (f a) x > 0) ∧
  (∀ x : ℝ, 1 / 2 < x ∧ x < a → deriv (f a) x < 0) ∧
  (∀ x : ℝ, a < x → deriv (f a) x > 0) :=
sorry

theorem problem_II (a : ℝ) :
  (∃ x₀ : ℝ, 1 ≤ x₀ ∧ x₀ ≤ Real.exp 1 ∧ f a x₀ ≥ g a x₀) ↔ a ≤ (Real.exp 1 * (Real.exp 1 - 2)) / (Real.exp 1 - 1) :=
sorry

end problem_I_problem_II_l795_795741


namespace exists_product_two_distinct_primes_exists_product_three_distinct_primes_l795_795496

theorem exists_product_two_distinct_primes (n : ℕ) (h : n > 4) :
  ∃ m, n < m ∧ m < 2 * n ∧ ∃ p1 p2 : ℕ, prime p1 ∧ prime p2 ∧ p1 ≠ p2 ∧ m = p1 * p2 := sorry

theorem exists_product_three_distinct_primes (n : ℕ) (h : n > 15) :
  ∃ m, n < m ∧ m < 2 * n ∧ ∃ p1 p2 p3 : ℕ, prime p1 ∧ prime p2 ∧ prime p3 ∧ 
  p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ m = p1 * p2 * p3 := sorry

end exists_product_two_distinct_primes_exists_product_three_distinct_primes_l795_795496


namespace female_democrats_count_l795_795559

-- Definitions of given conditions as assumptions.
variables (F M D_F D_M : ℕ)

axiom total_participants : F + M = 870
axiom female_democrats : D_F = 1/2 * F
axiom male_democrats : D_M = 1/4 * M
axiom total_democrats : D_F + D_M = 1/3 * (F + M)

-- The theorem we need to prove.
theorem female_democrats_count : D_F = 145 :=
by
  suffices h : F = 290, sorry
  -- Further proof will be completed here.

end female_democrats_count_l795_795559


namespace b_geometric_a_general_l795_795824

-- The sequence {a_n} with conditions
def a : ℕ → ℚ
| 0       := 1
| 1       := 5/3
| (n + 2) := (5/3) * a (n + 1) - (2/3) * a n

-- The sequence {b_n} defined as b_n = a_{n+1} - a_n
def b : ℕ → ℚ
| n       := a (n + 1) - a n

-- b_n is a geometric sequence
theorem b_geometric : ∀ n : ℕ, b (n + 1) = (2/3) * b n := 
by
  intros
  sorry

-- General term formula for {a_n}
theorem a_general : ∀ n : ℕ, a n = 3 - 2 * (2/3) ^ (n - 1) := 
by
  intros
  sorry

end b_geometric_a_general_l795_795824


namespace geordie_commute_distance_l795_795323

structure Commute :=
  (car_toll : ℝ)
  (motorcycle_toll : ℝ)
  (mpg : ℝ)
  (gas_price : ℝ)
  (car_trips : ℕ)
  (motorcycle_trips : ℕ)
  (total_spent : ℝ)

def geordie_commute := Commute.mk 12.5 7 35 3.75 3 2 118

def calculate_commute_distance (c : Commute) : Prop :=
  let total_toll_costs := (c.car_toll * c.car_trips) + (c.motorcycle_toll * c.motorcycle_trips)
  let gas_costs := c.total_spent - total_toll_costs
  let gallons_used := gas_costs / c.gas_price
  let total_miles := gallons_used * c.mpg
  total_miles / 10 = 62.07

theorem geordie_commute_distance : calculate_commute_distance geordie_commute :=
by
  sorry

end geordie_commute_distance_l795_795323


namespace christine_needs_32_tablespoons_l795_795649

-- Define the conditions
def tablespoons_per_egg_white : ℕ := 2
def egg_whites_per_cake : ℕ := 8
def number_of_cakes : ℕ := 2

-- Define the calculation for total tablespoons of aquafaba needed
def total_tbs_aquafaba : ℕ :=
  tablespoons_per_egg_white * (egg_whites_per_cake * number_of_cakes)

-- The theorem to prove
theorem christine_needs_32_tablespoons :
  total_tbs_aquafaba = 32 :=
by 
  -- Placeholder for proof, as proof steps are not required
  sorry

end christine_needs_32_tablespoons_l795_795649


namespace median_and_mode_correct_l795_795422

noncomputable def data_set : List ℕ := [3, 6, 4, 6, 4, 3, 6, 5, 7]

def median (l : List ℕ) : ℕ :=
  let sorted := l.sorted
  sorted.nthLe (sorted.length / 2) sorry

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ (acc, freq) x =>
    if l.count x > freq then (x, l.count x)
    else acc) (0, 0)

theorem median_and_mode_correct : median data_set = 5 ∧ mode data_set = 6 :=
by
  sorry

end median_and_mode_correct_l795_795422


namespace muffins_per_box_muffins_per_box_l795_795838

theorem muffins_per_box (total_muffins : ℕ) (num_boxes : ℕ) (h1 : total_muffins = 96) (h2 : num_boxes = 8) : total_muffins / num_boxes = 12 :=
by
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul_right (by norm_num) rfl

-- The main statement without committing to a detailed proof
theorem muffins_per_box' : 96 / 8 = 12 :=
by
  norm_num

end muffins_per_box_muffins_per_box_l795_795838


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795978

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795978


namespace visitors_on_saturday_l795_795615

theorem visitors_on_saturday (S : ℕ) (h1 : S + (S + 40) = 440) : S = 200 := by
  sorry

end visitors_on_saturday_l795_795615


namespace souvenirs_sold_in_morning_l795_795588

theorem souvenirs_sold_in_morning
  (total_souvenirs : ℕ := 24)
  (morning_price : ℕ := 7)
  (total_revenue : ℕ := 120)
  (h_condition1 : ∀ x, x ≤ total_souvenirs / 2)
  (h_condition2 : ∀ x y, 7 * x + y * (24 - x) = 120)
  (h_condition3 : ∀ y, y ∈ ℕ)
  : ∃ x, x = 8 :=
by
  sorry

end souvenirs_sold_in_morning_l795_795588


namespace perfect_matching_l795_795212

open Finset

-- Define the necessary conditions of the problem
variable (n : ℕ) (teams : Finset ℕ) (days : Finset ℕ)
variable (games : days → Finset (ℕ × ℕ))
variable (winners : (d : days) → (a b : ℕ) → Prop)

-- Assert every team played every other team exactly once
axiom every_team_plays_once : ∀ (a b : ℕ), a ∈ teams → b ∈ teams → a ≠ b → ∃ d ∈ days, (a, b) ∈ games d ∨ (b, a) ∈ games d

-- Assert each day has n games with unique winners
axiom n_games_per_day : ∀ d : days, games d.card = n
axiom unique_winners_per_game : ∀ d : days, ∀ (a b : ℕ), a ≠ b → winners d a b → winners d b a → False

-- Prove the existence of a perfect matching between days and winning teams
theorem perfect_matching :
  ∃ (f : days → ℕ), (∀ d, f d ∈ teams) ∧ (∀ d d', d ≠ d' → f d ≠ f d') :=
sorry

end perfect_matching_l795_795212


namespace min_value_N_l795_795066

theorem min_value_N (a b c d e f : ℤ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) (h₄ : 0 < e) (h₅ : 0 < f)
  (h_sum : a + b + c + d + e + f = 4020) :
  ∃ N : ℤ, N = max (a + b) (max (b + c) (max (c + d) (max (d + e) (e + f)))) ∧ N = 805 :=
by
  sorry

end min_value_N_l795_795066


namespace find_b_pure_imaginary_l795_795348

-- Given conditions
variables (b : ℝ)

-- Complex numbers in Lean and the definition of product being pure imaginary
def pure_imaginary (z : ℂ) : Prop := z.re = 0

-- The specific product to evaluate
def product (b : ℝ) : ℂ := (1 + b * complex.I) * (2 + complex.I)

-- The proof statement
theorem find_b_pure_imaginary (hb : pure_imaginary (product b)) : b = 2 :=
sorry

end find_b_pure_imaginary_l795_795348


namespace find_S3n_l795_795791

variable (a : ℕ → ℝ) -- the geometric sequence (a_n)

/-- The partial sum S is defined as the sum of the first n terms of the sequence -/
def S (n : ℕ) : ℝ := ∑ i in finset.range n, a i

theorem find_S3n (S_n S_2n S_3n : ℕ → ℝ) (n : ℕ) 
  (h1 : S_n n = 48) (h2 : S_2n n = 60) 
  (geo_seq : (S_2n n - S_n n) * (S_2n n - S_n n) = S_n n * (S_3n n - S_2n n)):
  S_3n n = 63 := by 
  sorry

end find_S3n_l795_795791


namespace no_half_dimension_cuboid_l795_795107

theorem no_half_dimension_cuboid
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (a' b' c' : ℝ) (ha' : a' > 0) (hb' : b' > 0) (hc' : c' > 0) :
  ¬ (a' * b' * c' = (1 / 2) * a * b * c ∧ 2 * (a' * b' + b' * c' + c' * a') = a * b + b * c + c * a) :=
by
  sorry

end no_half_dimension_cuboid_l795_795107


namespace triangle_area_integer_count_l795_795254

/--
Given a set of all triangles OPQ where O is the origin and P, Q are distinct points
in the plane with nonnegative integer coordinates (x, y) such that 29 * x + y = 2035,
prove that the number of such distinct triangles whose area is a positive integer is 1225.
-/
theorem triangle_area_integer_count :
  let point_condition := λ (x y : ℕ), 29 * x + y = 2035,
      even_points := {x | point_condition x (2035 - 29 * x) ∧ x % 2 = 0},
      odd_points := {x | point_condition x (2035 - 29 * x) ∧ x % 2 = 1},
      even_pairs := (even_points.card.choose 2),
      odd_pairs := (odd_points.card.choose 2)
  in
  even_pairs + odd_pairs = 1225 := 
by
  let point_condition := λ x y, 29 * x + y = 2035,
  let even_points := {x | point_condition x (2035 - 29 * x) ∧ x % 2 = 0},
  let odd_points := {x | point_condition x (2035 - 29 * x) ∧ x % 2 = 1},
  have even_count : even_points.card = 36 := sorry,
  have odd_count : odd_points.card = 35 := sorry,
  have even_pairs : (finset.card even_points).choose 2 = 630 := sorry,
  have odd_pairs : (finset.card odd_points).choose 2 = 595 := sorry,
  have sum_pairs := even_pairs + odd_pairs,
  have eq_sum : sum_pairs = 1225 := by
    rw [even_pairs, odd_pairs]; exact rfl,
  exact eq_sum

end triangle_area_integer_count_l795_795254


namespace find_median_and_mode_l795_795416

def data_set := [3, 6, 4, 6, 4, 3, 6, 5, 7]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get! ((sorted.length - 1) / 2)

def mode (l : List ℕ) : ℕ :=
  l.foldl
    (λ counts n => counts.insert n (counts.find n |>.getD 0 + 1))
    (Std.HashMap.empty ℕ ℕ)
  |>.toList
  |>.foldl (λ acc (k, v) => if v > acc.2 then (k, v) else acc) (0, 0)
  |>.1

theorem find_median_and_mode :
  median data_set = 5 ∧ mode data_set = 6 :=
by
  sorry

end find_median_and_mode_l795_795416


namespace best_number_max_points_chosen_number_is_best_l795_795225

def is_divisible (a b : ℕ) : Prop := b % a = 0

def points (x : ℕ) : ℕ :=
  (if is_divisible 3 x then 3 else 0) +
  (if is_divisible 5 x then 5 else 0) +
  (if is_divisible 7 x then 7 else 0) +
  (if is_divisible 9 x then 9 else 0) +
  (if is_divisible 11 x then 11 else 0)

def max_points : ℕ :=
  30

def best_number : ℕ :=
  2079

theorem best_number_max_points :
  ∀ x : ℕ, 2017 ≤ x ∧ x ≤ 2117 → points x ≤ max_points :=
sorry

theorem chosen_number_is_best :
  points best_number = max_points :=
sorry

end best_number_max_points_chosen_number_is_best_l795_795225


namespace coins_distributed_l795_795490

theorem coins_distributed (x : ℕ) (h₁ : 2 * x = y) (h₂ : 3 * y = z) (h₃ : 4 * z = w) 
  (h₄ : x + y + z + w = 132) : x = 4 :=
by
  let y := 2 * x
  let z := 3 * y
  let w := 4 * z
  have step₁ : y = 2 * x, from h₁
  have step₂ : z = 3 * y, from h₂
  have step₃ : w = 4 * z, from h₃
  have step₄ : x + y + z + w = 132, from h₄
  sorry

end coins_distributed_l795_795490


namespace change_sum_equals_108_l795_795083

theorem change_sum_equals_108 :
  ∃ (amounts : List ℕ), (∀ a ∈ amounts, a < 100 ∧ ((a % 25 = 4) ∨ (a % 5 = 4))) ∧
    amounts.sum = 108 := 
by
  sorry

end change_sum_equals_108_l795_795083


namespace rhombus_side_length_l795_795553

variable {L S : ℝ}

theorem rhombus_side_length (hL : 0 ≤ L) (hS : 0 ≤ S) :
  (∃ m : ℝ, m = 1 / 2 * Real.sqrt (L^2 - 4 * S)) :=
sorry

end rhombus_side_length_l795_795553


namespace range_of_eccentricity_l795_795335

-- Definitions
def hyperbola (C : Type) := ∃ (F₁ F₂ M : ℝ → ℝ), ∀ t : ℝ, C t = (t, F₁ t, F₂ t, M t)

def isosceles_triangle (F₁ F₂ M : ℝ) := (F₁ = F₂) ∧ (F₁ = M) ∧ (F₂ = M)

def is_acuteangled (F₁ F₂ M : ℝ) := 
  ∀ A B C : ℝ, isosceles_triangle A B C → ((A <= B) ∧ (B <= C) → C ≤ ∥√2 + 1∥)

def eccentricity (a c : ℝ) := c / a

-- The theorem stating our mathematical problem
theorem range_of_eccentricity (C : Type) (F₁ F₂ M : ℝ → ℝ) 
  (hypC : hyperbola C)
  (hyp_triangle : ∀ t : ℝ, isosceles_triangle (F₁ t) (F₂ t) (M t))
  (hyp_acute : ∀ t : ℝ, is_acuteangled (F₁ t) (F₂ t) (M t)) :
  ∃ e : ℝ, e > 1 + √2 :=
begin
  sorry
end

end range_of_eccentricity_l795_795335


namespace probability_multiple_of_2_or_5_l795_795932

theorem probability_multiple_of_2_or_5 : 
  let total_cards := 25 
  let cards := finset.range (total_cards + 1) -- cards {0, 1, ..., 25}
  let multiples_of_2 := cards.filter (λ x, x % 2 = 0) -- multiples of 2
  let multiples_of_5 := cards.filter (λ x, x % 5 = 0) -- multiples of 5
  let multiples_of_2_or_5 := (multiples_of_2 ∪ multiples_of_5).card - (multiples_of_2 ∩ multiples_of_5).card
  let favorable_outcomes := multiples_of_2_or_5
  let total_outcomes := total_cards
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 5 := 
by
  -- proof steps to be filled
  sorry

end probability_multiple_of_2_or_5_l795_795932


namespace smallest_possible_time_for_travel_l795_795674

theorem smallest_possible_time_for_travel :
  ∃ t : ℝ, (∀ D M P : ℝ, D = 6 → M = 6 → P = 6 → 
    ∀ motorcycle_speed distance : ℝ, motorcycle_speed = 90 → distance = 135 → 
    t < 3.9) :=
  sorry

end smallest_possible_time_for_travel_l795_795674


namespace length_PQ_l795_795446

-- Define the given lengths and conditions of the triangle
variables (D E F P Q : Type) [MetricSpace D] [MetricSpace E] [MetricSpace F] [MetricSpace P] [MetricSpace Q]

def length_DE := 25
def length_EF := 29
def length_FD := 32
def lengths_similarity (x : ℝ) := x / 3

-- Define properties of points P and Q
def PQ_parallel_EF : Prop := true  -- Given that PQ is parallel to EF
def PQ_divided : Prop := lengths_similarity length_DE = length_DE / 3

-- Define the goal
theorem length_PQ (h1: PQ_parallel_EF) (h2 : PQ_divided) : lengths_similarity length_DE = 25 / 3 :=
by
  -- given the equivalence
  rw [←h2]
  sorry

end length_PQ_l795_795446


namespace seashells_after_transactions_l795_795629

def initial_seashells : ℝ := 385.5
def seashells_given_away_to_friends : ℝ := 45.75
def seashells_given_away_to_brothers : ℝ := 34.25
def seashells_sold_fraction : ℝ := 2 / 3
def seashells_traded_fraction : ℝ := 1 / 4

noncomputable def final_seashells (initial : ℝ) (given_away_friends : ℝ) (given_away_brothers : ℝ) (sold_fraction : ℝ) (traded_fraction : ℝ) : ℝ :=
  let remaining_after_giveaway := initial - (given_away_friends + given_away_brothers)
  let remaining_after_sale := remaining_after_giveaway - (sold_fraction * remaining_after_giveaway)
  remaining_after_sale - (traded_fraction * remaining_after_sale)

theorem seashells_after_transactions :
  final_seashells initial_seashells seashells_given_away_to_friends seashells_given_away_to_brothers seashells_sold_fraction seashells_traded_fraction = 76.375 :=
by
  sorry

end seashells_after_transactions_l795_795629


namespace percentage_increase_first_to_second_l795_795448

theorem percentage_increase_first_to_second (D1 D2 D3 : ℕ) (h1 : D2 = 12)
  (h2 : D3 = D2 + Nat.div (D2 * 25) 100) (h3 : D1 + D2 + D3 = 37) :
  Nat.div ((D2 - D1) * 100) D1 = 20 := by
  sorry

end percentage_increase_first_to_second_l795_795448


namespace smallest_k_satisfies_l795_795263

noncomputable def sqrt (x : ℝ) : ℝ := x ^ (1 / 2 : ℝ)

theorem smallest_k_satisfies (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (sqrt (x * y)) + (1 / 2) * (sqrt (abs (x - y))) ≥ (x + y) / 2 :=
by
  sorry

end smallest_k_satisfies_l795_795263


namespace maximize_projection_area_l795_795004

-- Define a rectangular parallelepiped, its projection, and the horizontal plane
variables {A' B' C' : Type} [rectangular_parallelepiped A' B' C']
variables {horizontal_plane : Type} [plane horizontal_plane]
variables (α : ℝ)

-- Define the maximum area condition
axiom max_projection_area : is_maximum_area_proection (A'B'C') (horizontal_plane) ↔ (α = 0)

-- Lean statement to prove the problem statement
theorem maximize_projection_area (A' B' C' : rectangular_parallelepiped) (horizontal_plane : plane):
  is_maximum_area_proection (A' B' C') (horizontal_plane) ↔ one_face_parallel_to_plane (A' B' C') (horizontal_plane) :=
begin
  sorry
end

end maximize_projection_area_l795_795004


namespace gcd_factorial_8_10_l795_795311

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_10 : Nat.gcd (factorial 8) (factorial 10) = 40320 :=
by
  -- these pre-evaluations help Lean understand the factorial values
  have fact_8 : factorial 8 = 40320 := by sorry
  have fact_10 : factorial 10 = 3628800 := by sorry
  rw [fact_8, fact_10]
  -- the actual proof gets skipped here
  sorry

end gcd_factorial_8_10_l795_795311


namespace pentagon_area_l795_795208

noncomputable def calculatePentagonArea (a b c d e : ℕ) (r s : ℕ) : ℕ :=
  a * b - (1 / 2) * r * s

theorem pentagon_area (a b c d e r s : ℕ) (h1 : a ∈ {17, 23, 24, 30, 37})
  (h2 : b ∈ {17, 23, 24, 30, 37})
  (h3 : c ∈ {17, 23, 24, 30, 37})
  (h4 : d ∈ {17, 23, 24, 30, 37})
  (h5 : e ∈ {17, 23, 24, 30, 37})
  (h6 : pairwise (≠) [a, b, c, d, e])
  (h7 : r ^ 2 + s ^ 2 = e ^ 2)
  (h8 : e = 25) -- Specific to identified Pythagorean triple (7, 24, 25)
  (h9 : r = 7)
  (h10 : s = 24) :
  calculatePentagonArea a b c d e r s = 900 := sorry

end pentagon_area_l795_795208


namespace perimeter_of_C_l795_795927

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end perimeter_of_C_l795_795927


namespace solve_abs_eq_l795_795279

theorem solve_abs_eq : ∀ x : ℚ, (|2 * x + 6| = 3 * x + 9) ↔ (x = -3) := by
  intros x
  sorry

end solve_abs_eq_l795_795279


namespace emily_vs_tim_score_difference_l795_795427

/-- Problem: In a math quiz, Emily scored the highest and Tim scored the lowest.
Prove that the difference between Emily's score and Tim's score is 7 points given their respective scores are 9 and 2. --/
theorem emily_vs_tim_score_difference :
  ∀ (emily_score tim_score : ℕ), emily_score = 9 → tim_score = 2 → (emily_score - tim_score = 7) :=
by
  intros emily_score tim_score h1 h2
  rw [h1, h2]
  rw Nat.sub_self
  assumption


end emily_vs_tim_score_difference_l795_795427


namespace find_n_l795_795042

/-- In the expansion of (1 + 3x)^n, where n is a positive integer and n >= 6, 
    if the coefficients of x^5 and x^6 are equal, then n is 7. -/
theorem find_n (n : ℕ) (h₀ : 0 < n) (h₁ : 6 ≤ n)
  (h₂ : 3^5 * Nat.choose n 5 = 3^6 * Nat.choose n 6) : 
  n = 7 := 
sorry

end find_n_l795_795042


namespace min_value_of_x_plus_y_l795_795332

theorem min_value_of_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 2 * x + 8 * y - x * y = 0) : x + y ≥ 18 :=
sorry

end min_value_of_x_plus_y_l795_795332


namespace average_age_when_youngest_born_l795_795946

theorem average_age_when_youngest_born (n : ℕ) (ages : Finₓ 7 → ℝ) (h_average : (∑ i, ages i) = 210) (h_youngest : ages 0 = 4) :
  let ages_when_born := Finₓ (6) → ℝ in
  (∑ i, ages_when_born i) / 6 = 30.3333 := 
by
  -- placeholder for the actual Lean proof
  sorry

end average_age_when_youngest_born_l795_795946


namespace numberOfWaysToArrange1225_l795_795434

def isValidArrangement (arr : List Nat) : Bool :=
  arr.length = 4 ∧ (List.last arr 0 = 5 ∨ List.last arr 0 = 0)

noncomputable def countArrangements (digits : List Nat) : Nat :=
  let permutations := (digits.permutations.filter isValidArrangement).length
  permutations

theorem numberOfWaysToArrange1225 : countArrangements [1, 2, 2, 5] = 6 := by
  sorry

end numberOfWaysToArrange1225_l795_795434


namespace jane_average_speed_l795_795879

theorem jane_average_speed :
  let total_distance := 200
  let total_time := 6
  total_distance / total_time = 100 / 3 :=
by
  sorry

end jane_average_speed_l795_795879


namespace domain_of_function_l795_795261

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := -8 * x^2 + 14 * x - 3

-- Define the domain condition
def domain_condition := { x : ℝ | quadratic x ≥ 0 }

-- State the main theorem
theorem domain_of_function : domain_condition = set.Icc (1 / 4 : ℝ) (3 / 2 : ℝ) :=
by
  sorry

end domain_of_function_l795_795261


namespace num_pos_int_values_l795_795704

theorem num_pos_int_values
  (N : ℕ) 
  (h₀ : 0 < N)
  (h₁ : ∃ (k : ℕ), 0 < k ∧ 48 = k * (N + 3)) :
  ∃ (n : ℕ), n = 7 :=
sorry

end num_pos_int_values_l795_795704


namespace distance_classroom_to_playground_l795_795576

-- Definitions of conditions
def step_length_cm := 52
def steps_taken := 176
def total_distance_cm := step_length_cm * steps_taken

-- Conversion to meters
def total_distance_m := total_distance_cm / 100

-- Statement to prove
theorem distance_classroom_to_playground : total_distance_m = 91.52 := sorry

end distance_classroom_to_playground_l795_795576


namespace range_of_x_l795_795018

theorem range_of_x (x : ℝ) : (x ≠ 2) ↔ ∃ y : ℝ, y = 5 / (x - 2) :=
begin
  sorry
end

end range_of_x_l795_795018


namespace mean_value_of_quadrilateral_angles_l795_795995

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l795_795995


namespace mean_value_of_quadrilateral_angles_l795_795963

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795963


namespace complex_multiplication_l795_795349

theorem complex_multiplication :
  let i := complex.I
  let Z1 := (2 - i : ℂ)
  let Z2 := (1 + i : ℂ)
  Z1 * Z2 = 3 + i :=
by
  sorry

end complex_multiplication_l795_795349


namespace figure_C_perimeter_l795_795886

def is_perimeter (figure : Type) (perimeter : ℕ) : Prop :=
∃ x y : ℕ, (figure = 'A' → 6*x + 2*y = perimeter) ∧ 
           (figure = 'B' → 4*x + 6*y = perimeter) ∧
           (figure = 'C' → 2*x + 6*y = perimeter)

theorem figure_C_perimeter (hA : is_perimeter 'A' 56) (hB : is_perimeter 'B' 56) : 
  is_perimeter 'C' 40 :=
by
  sorry

end figure_C_perimeter_l795_795886


namespace roots_intervals_l795_795330

variables {a b c : ℝ} {f : ℝ → ℝ}

def f (x : ℝ) : ℝ := (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a)

theorem roots_intervals (h : a < b ∧ b < c) :
  (∃ x₁ ∈ Ioo a b, f x₁ = 0) ∧ (∃ x₂ ∈ Ioo b c, f x₂ = 0) :=
sorry

end roots_intervals_l795_795330


namespace find_k_perpendicular_find_k_parallel_l795_795381

open Real

def a : ℝ × ℝ := (-3, 1)
def b : ℝ × ℝ := (1, -2)
def m (k : ℝ) := (a.1 + k * b.1, a.2 + k * b.2)
def c : ℝ × ℝ := (1, -1)

#check @prod.mk_inj_iff ℝ ℝ

-- The first problem
theorem find_k_perpendicular (k : ℝ) (h : m k = (-3 + k, 1 - 2 * k)) : k = 5 / 3 :=
sorry

-- The second problem
theorem find_k_parallel (k : ℝ) (h_parallel : ∃ l : ℝ, (m k) = (l * (k + 1), l * (-2 * k - 1))) : k = -1 / 3 :=
sorry

end find_k_perpendicular_find_k_parallel_l795_795381


namespace count_possible_n_l795_795794

-- Definitions
def AB (n : ℕ) : ℕ := 4 * n + 2
def BC (n : ℕ) : ℕ := n + 5
def AC (n : ℕ) : ℕ := 3 * n + 1

-- Given conditions
def angles_in_order (n : ℕ) : Prop :=
  AB n > AC n ∧ AC n > BC n

def valid_triangle (n : ℕ) : Prop :=
  n > 0 ∧ angles_in_order n

-- Proof statement
theorem count_possible_n : (finset.filter valid_triangle (finset.range 101)).card = 100 := by
  sorry

end count_possible_n_l795_795794


namespace numerator_of_fraction_l795_795873

theorem numerator_of_fraction (x : ℤ) (h : (x : ℚ) / (4 * x - 5) = 3 / 7) : x = 3 := 
sorry

end numerator_of_fraction_l795_795873


namespace proof_cm²_to_dm²_proof_hectares_to_km²_proof_yuan_to_yuan_jiao_fen_proof_hectares_to_m²_proof_jiao_fen_to_yuan_proof_dm²_to_m²_proof_m²_to_hectares_proof_m_to_cm_l795_795275

-- Definitions for the basic units and conversions
def cm²_to_dm² (x : ℕ) : ℕ := x / 100
def hectares_to_km² (x : ℕ) : ℕ := x / 100
def yuan_to_yuan_jiao_fen (x : ℝ) : ℕ × ℕ × ℕ :=
  let fen := (x * 100).to_nat % 10
  let jiao := ((x * 10).to_nat % 10)
  let yuan := x.to_nat
  (yuan, jiao, fen)
def hectares_to_m² (x : ℕ) : ℕ := x * 10000 * 100
def jiao_fen_to_yuan (j : ℕ) (f : ℕ) : ℝ := ↑j / 10 + ↑f / 100
def dm²_to_m² (x : ℕ) : ℕ := x / 100
def m²_to_hectares (x : ℕ) : ℕ := x / 10000
def m_to_cm (x : ℕ) : ℕ := x * 100

-- Theorem statements
theorem proof_cm²_to_dm² : cm²_to_dm² 70000 = 700 := by
  sorry

theorem proof_hectares_to_km² : hectares_to_km² 800 = 8 := by
  sorry

theorem proof_yuan_to_yuan_jiao_fen : yuan_to_yuan_jiao_fen 1.65 = (1, 6, 5) := by
  sorry

theorem proof_hectares_to_m² : hectares_to_m² 400 = 4000000 := by
  sorry

theorem proof_jiao_fen_to_yuan : jiao_fen_to_yuan 5 7 = 0.57 := by
  sorry

theorem proof_dm²_to_m² : dm²_to_m² 5000 = 50 := by
  sorry

theorem proof_m²_to_hectares : m²_to_hectares 60000 = 6 := by
  sorry

theorem proof_m_to_cm : m_to_cm 9 = 900 := by
  sorry

end proof_cm²_to_dm²_proof_hectares_to_km²_proof_yuan_to_yuan_jiao_fen_proof_hectares_to_m²_proof_jiao_fen_to_yuan_proof_dm²_to_m²_proof_m²_to_hectares_proof_m_to_cm_l795_795275


namespace number_of_ordered_triples_l795_795622

theorem number_of_ordered_triples (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 3969) (h4 : a * c = 3969^2) :
    ∃ n : ℕ, n = 12 := sorry

end number_of_ordered_triples_l795_795622


namespace tangent_line_parallel_x_axis_extreme_values_of_f_max_value_of_k_l795_795372

-- Definition of the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := x - 1 + a / Real.exp x

-- (1) Prove a = e if tangent line to curve y = f(x) at point (1, f(1)) is parallel to x-axis
theorem tangent_line_parallel_x_axis : 
  ∀ (a : ℝ), (1 - a / Real.exp 1 = 0) → a = Real.exp 1 :=
by
  intro a h
  sorry

-- (2) Prove the locations of extreme values of f(x) given specific values of a
theorem extreme_values_of_f : 
  ∀ (a : ℝ), 
  (a > 0 → (∃ x_extreme : ℝ, x_extreme = Real.log a) ∧ 
    (∀ x < Real.log a, f' x < 0) ∧ (∀ x > Real.log a, f' x > 0)) ∧ 
  (a ≤ 0 → ∀ x : ℝ, f' x > 0) :=
by
  intro a
  sorry

-- (3) Prove the maximum value of k = 1 for the function f(x) = x - 1 + 1 / exp(x) such that the line l(y = kx - 1) and curve y = f(x) have no common points
theorem max_value_of_k (k : ℝ) : 
  (a = 1) → 
  (y : ℝ -> ℝ) (l : ℝ -> ℝ), 
  y = (λ x, x - 1 + 1 / Real.exp x) → 
  l = (λ x, k * x - 1) → 
  (∀ x : ℝ, (y x ≠ l x)) → k ≤ 1 :=
by
  intro k h_a y l h_f h_l no_common_points
  sorry

end tangent_line_parallel_x_axis_extreme_values_of_f_max_value_of_k_l795_795372


namespace curve_parametric_to_triple_l795_795610

theorem curve_parametric_to_triple :
  ∃ a b c : ℚ, (∀ t : ℝ, let x := 3 * Real.cos t - 2 * Real.sin t
                           y := 3 * Real.sin t in
                           a * x^2 + b * x * y + c * y^2 = 1) ∧
             (a, b, c) = (1/9 : ℚ, 4/27 : ℚ, 5/81 : ℚ) :=
by
  sorry

end curve_parametric_to_triple_l795_795610


namespace initial_principal_amount_l795_795839

theorem initial_principal_amount : 
  let P : ℝ := 5269.48 in 
  let R : ℝ := 0.06 in 
  let T : ℝ := 9 in 
  let A : ℝ := 8110 in 
  ∃ P' : ℝ, P' = 5269.48 ∧ A = P' + (P' * R * T) / 100 := 
begin
  sorry
end

end initial_principal_amount_l795_795839


namespace appropriate_material_for_meiosis_l795_795154

theorem appropriate_material_for_meiosis (Ascaris_fertilized_eggs_mitosis : Prop)
  (Chicken_liver_cells_mitosis : Prop)
  (Mouse_testes_continuous_sperm_formation : Prop)
  (Onion_epidermis_no_divide : Prop) :
  (Mouse_testes_continuous_sperm_formation →
  (Ascaris_fertilized_eggs_mitosis →
  Chicken_liver_cells_mitosis →
  Onion_epidermis_no_divide →
  "C is the correct answer" )) := 
sorry

end appropriate_material_for_meiosis_l795_795154


namespace kitchen_area_l795_795497

-- Definitions based on conditions
def total_area : ℕ := 1110
def bedroom_area : ℕ := 11 * 11
def num_bedrooms : ℕ := 4
def bathroom_area : ℕ := 6 * 8
def num_bathrooms : ℕ := 2

-- Theorem to prove
theorem kitchen_area :
  let total_bedroom_area := bedroom_area * num_bedrooms,
      total_bathroom_area := bathroom_area * num_bathrooms,
      remaining_area := total_area - (total_bedroom_area + total_bathroom_area)
  in remaining_area / 2 = 265 := 
by
  sorry

end kitchen_area_l795_795497


namespace nearest_integer_to_power_l795_795162

theorem nearest_integer_to_power (a b : ℝ) (h1 : a = 3) (h2 : b = sqrt 2) : 
  abs ((a + b)^6 - 3707) < 0.5 :=
by
  sorry

end nearest_integer_to_power_l795_795162


namespace minimal_squares_25_grid_l795_795660

def unit_square_grid (n : ℕ) := list (list ℕ)

def color_all_lines_with_minimal_squares (n : ℕ) (grid : unit_square_grid n) : Prop :=
  ∃ (squares : ℕ), squares = 48 ∧ 
  (∀ line ∈ grid_lines n, is_colored_by squares line)

theorem minimal_squares_25_grid : color_all_lines_with_minimal_squares 25 (unit_square_grid 25) :=
  sorry

end minimal_squares_25_grid_l795_795660


namespace second_player_wins_l795_795232

/-- There are 2022 ones written along the circumference and two players take turns. 
In one move, a player erases two neighboring numbers and writes their sum instead. 
The winner is the one who obtains the number 4. If in the end only one number remains 
and it is not 4, the game ends in a draw. Prove that the second player can ensure their victory. --/
theorem second_player_wins (initial_condition : ∀ i : ℕ, 0 ≤ i < 2022 → 1 = 1) :
    (∃ n : ℕ, 0 ≤ n < 2022 ∧ initial_condition n = 4) := 
sorry

end second_player_wins_l795_795232


namespace mean_value_of_quadrilateral_angles_l795_795970

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795970


namespace find_median_and_mode_l795_795415

def data_set := [3, 6, 4, 6, 4, 3, 6, 5, 7]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get! ((sorted.length - 1) / 2)

def mode (l : List ℕ) : ℕ :=
  l.foldl
    (λ counts n => counts.insert n (counts.find n |>.getD 0 + 1))
    (Std.HashMap.empty ℕ ℕ)
  |>.toList
  |>.foldl (λ acc (k, v) => if v > acc.2 then (k, v) else acc) (0, 0)
  |>.1

theorem find_median_and_mode :
  median data_set = 5 ∧ mode data_set = 6 :=
by
  sorry

end find_median_and_mode_l795_795415


namespace interesting_seven_digit_numbers_l795_795213

theorem interesting_seven_digit_numbers :
  ∃ n : Fin 2 → ℕ, (∀ i : Fin 2, n i = 128) :=
by sorry

end interesting_seven_digit_numbers_l795_795213


namespace relationship_y1_y2_l795_795492

theorem relationship_y1_y2
    (b : ℝ) 
    (y1 y2 : ℝ)
    (h1 : y1 = - (1 / 2) * (-2) + b) 
    (h2 : y2 = - (1 / 2) * 3 + b) : 
    y1 > y2 :=
sorry

end relationship_y1_y2_l795_795492


namespace max_points_earned_l795_795221

def divisible_by (a b : Nat) : Prop := b ≠ 0 ∧ a % b = 0

def points (x : Nat) : Nat :=
  (if divisible_by x 3 then 3 else 0) +
  (if divisible_by x 5 then 5 else 0) +
  (if divisible_by x 7 then 7 else 0) +
  (if divisible_by x 9 then 9 else 0) +
  (if divisible_by x 11 then 11 else 0)

theorem max_points_earned (x : Nat) (h1 : 2017 ≤ x) (h2 : x ≤ 2117) :
  points 2079 = 30 := by
  sorry

end max_points_earned_l795_795221


namespace sum_of_remainders_l795_795235

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) : (n % 2 + n % 9) = 3 :=
by
  sorry

end sum_of_remainders_l795_795235


namespace necessary_but_not_sufficient_l795_795396

theorem necessary_but_not_sufficient (a : ℝ) : (a ≠ 1) → (a^2 ≠ 1) → (a ≠ 1) ∧ ¬((a ≠ 1) → (a^2 ≠ 1)) :=
by
  sorry

end necessary_but_not_sufficient_l795_795396


namespace hyperbola_condition_l795_795361

theorem hyperbola_condition (m : ℝ) :
  (∃ x y : ℝ, m * x^2 + (2 - m) * y^2 = 1) → m < 0 ∨ m > 2 :=
sorry

end hyperbola_condition_l795_795361


namespace find_cost_price_l795_795601

theorem find_cost_price (C : ℝ) (SP : ℝ) (M : ℝ) (h1 : SP = 1.25 * C) (h2 : 0.90 * M = SP) (h3 : SP = 65.97) : 
  C = 52.776 :=
by
  sorry

end find_cost_price_l795_795601


namespace perimeter_C_l795_795898

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end perimeter_C_l795_795898


namespace creature_token_perimeter_correct_l795_795778

/--
In a board game, a "creature" token is designed with a partial circular shape of radius 5 cm. 
The creature has an open mouth represented by a missing sector with a central angle of 90°. 
We need to prove that the perimeter of the token is \( 7.5\pi + 10 \).
-/
def creature_token_perimeter (radius : ℝ) (central_angle : ℝ) : ℝ :=
  let remaining_arc_fraction := (360 - central_angle) / 360
  let circumference := 2 * Real.pi * radius
  let arc_length := remaining_arc_fraction * circumference
  let perimeter := arc_length + 2 * radius
  perimeter

theorem creature_token_perimeter_correct :
  creature_token_perimeter 5 90 = 7.5 * Real.pi + 10 :=
by
  sorry

end creature_token_perimeter_correct_l795_795778


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795985

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795985


namespace simplify_trig_expression_l795_795860

theorem simplify_trig_expression :
  (cos 5 * cos 5 - sin 5 * sin 5) / (sin 40 * cos 40) = 2 :=
by
  sorry

end simplify_trig_expression_l795_795860


namespace angle_C_is_135_degrees_l795_795848

noncomputable section

-- Given definitions and conditions
variables (A B C H3 : Point)
variables (h3 : ℝ)

-- Conditions
axiom AH3_eq_2h3 : dist A H3 = 2 * h3
axiom BH3_eq_3h3 : dist B H3 = 3 * h3
axiom H3_on_AB : ∃ t : ℝ, t ∈ Icc 0 1 ∧ H3 = A + t • (B - A)
axiom H3_altitude : ∃ l : Line, is_altitude l H3 C

-- Proof statement
theorem angle_C_is_135_degrees : ∠ A C B = 135 :=
by
  sorry

end angle_C_is_135_degrees_l795_795848


namespace villagers_count_l795_795411

theorem villagers_count (V : ℕ) (milk_per_villager apples_per_villager bread_per_villager : ℕ) :
  156 = V * milk_per_villager ∧
  195 = V * apples_per_villager ∧
  234 = V * bread_per_villager →
  V = Nat.gcd (Nat.gcd 156 195) 234 :=
by
  sorry

end villagers_count_l795_795411


namespace abs_diff_of_roots_l795_795285

theorem abs_diff_of_roots : 
  ∀ r1 r2 : ℝ, 
  (r1 + r2 = 7) ∧ (r1 * r2 = 12) → abs (r1 - r2) = 1 :=
by
  -- Assume the roots are r1 and r2
  intros r1 r2 H,
  -- Decompose the assumption H into its components
  cases H with Hsum Hprod,
  -- Calculate the square of the difference using the given identities
  have H_squared_diff : (r1 - r2)^2 = (r1 + r2)^2 - 4 * (r1 * r2),
  { sorry },
  -- Substitute the known values to find the square of the difference
  have H_squared_vals : (r1 - r2)^2 = 49 - 4 * 12,
  { sorry },
  -- Simplify to get (r1 - r2)^2 = 1
  have H1 : (r1 - r2)^2 = 1,
  { sorry },
  -- The absolute value of the difference is the square root of this result
  have abs_diff : abs (r1 - r2) = 1,
  { sorry },
  -- Conclude the proof by showing the final result matches the expected answer
  exact abs_diff

end abs_diff_of_roots_l795_795285


namespace five_digit_palindrome_count_l795_795260

def is_digit (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ 9

def is_nonzero_digit (x : ℕ) : Prop := x ≥ 1 ∧ x ≤ 9

def is_palindrome (n : ℕ) : Prop :=
  let d1 := n / 10000,
      d2 := (n / 1000) % 10,
      d3 := (n / 100) % 10,
      d4 := (n / 10) % 10,
      d5 := n % 10 in
  d1 = d5 ∧ d2 = d4

def count_five_digit_palindromes : ℕ :=
  (finset.Ico 10000 100000).filter (λ n, is_palindrome n).card

theorem five_digit_palindrome_count : count_five_digit_palindromes = 900 := by
  sorry

end five_digit_palindrome_count_l795_795260


namespace distance_between_hands_l795_795255

theorem distance_between_hands
  (L : ℝ) (height_midpoint_to_hands : ℝ) (height_by_bead_moved_up : ℝ)
  (init_length_hypotenuse : ℝ) (total_length : L = 26)
  (bead_move_up : height_by_bead_moved_up = 8)
  (triangle_height : height_midpoint_to_hands = 5)
  (hypotenuse_length : init_length_hypotenuse = 13) :
  2 * (sqrt (init_length_hypotenuse^2 - triangle_height^2)) = 24 :=
by
  -- proof steps can be added here
  sorry

end distance_between_hands_l795_795255


namespace find_other_root_l795_795097

theorem find_other_root (z : ℂ) (z_squared : z^2 = -91 + 104 * I) (root1 : z = 7 + 10 * I) : z = -7 - 10 * I :=
by
  sorry

end find_other_root_l795_795097


namespace set_intersection_l795_795378

open Set

noncomputable def A : Set ℝ := {x | -1 ≤ x ∧ x < 1}
noncomputable def B : Set ℝ := {x | 1 < 2^x ∧ 2^x ≤ 4}

theorem set_intersection : A ∩ (compl B) = {x | -1 ≤ x ∧ x ≤ 0} :=
by
  sorry

end set_intersection_l795_795378


namespace min_knights_l795_795091

noncomputable def is_lying (n : ℕ) (T : ℕ → Prop) (p : ℕ → Prop) : Prop :=
    (T n → ∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m)) ∨ (¬T n → ¬∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m ∧ m < n))

open Nat

def islanders_condition (T : ℕ → Prop) (p : ℕ → Prop) :=
  ∀ n, n < 80 → (T n ∨ ¬T n) ∧ (T n → ∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m)) ∨ (¬T n → ¬∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m))

theorem min_knights : ∀ (T : ℕ → Prop) (p : ℕ → Prop), islanders_condition T p → ∃ k, k = 70 :=    
by
    sorry

end min_knights_l795_795091


namespace range_of_m_l795_795360

theorem range_of_m (m : ℝ) : (2 + m > 0) ∧ (1 - m > 0) ∧ (2 + m > 1 - m) → -1/2 < m ∧ m < 1 :=
by
  intros h
  sorry

end range_of_m_l795_795360


namespace correct_expression_l795_795181

def Expression : Type := String

def CorrectlyWritten (expr : Expression) : Prop :=
  match expr with
  | "ab / c" => false
  | "1(1/2)ab^2" => false
  | "a * b" => false
  | "3m" => true
  | _ => false

def Options : List Expression := ["ab / c", "1(1/2)ab^2", "a * b", "3m"]

theorem correct_expression : ∃ expr ∈ Options, CorrectlyWritten expr :=
by
  use "3m"
  split
  . repeat { try { assumption }, try { constructor } }
  . exact rfl

end correct_expression_l795_795181


namespace find_probability_of_B_l795_795547

variable {Ω : Type}             -- Underlying sample space
variable {P : MeasureTheory.ProbabilitySpace Ω} -- Probability measure

-- Definitions of events A and B
variable (A B : Set Ω)

def P_A : ℝ := 0.20
def P_A_and_B : ℝ := 0.15
def P_neither_A_nor_B : ℝ := 0.55 

theorem find_probability_of_B :
  P.to_measure (A ∪ B) = 0.20 + P.to_measure B - 0.15 →
  (1 - (0.20 + P.to_measure B - 0.15)) = 0.55 →
  P.to_measure B = 0.40 :=
by
  sorry

end find_probability_of_B_l795_795547


namespace minimize_total_cost_l795_795572

variables (a v : ℝ)

-- Condition: distances and variable cost function
def distance : ℝ := 800
def variable_cost (v : ℝ) : ℝ := (1/4) * v^2
def fixed_cost : ℝ := a

-- Condition: speed constraints and domain
def speed_domain : set ℝ := {v | 0 < v ∧ v ≤ 100}

-- Define the total cost function
def total_cost (v : ℝ) : ℝ := 800 * ((1/4) * v + a / v)

-- Statement to prove
theorem minimize_total_cost (a : ℝ) (h₀ : 0 < a) :
  (∀ v, v ∈ speed_domain → total_cost v = 800 * ((1/4) * v + a / v)) ∧
  (if (0 < a ∧ a ≤ 2500) then ∃ v, v = 2 * real.sqrt a ∧ total_cost v = 800 * ((1/4) * (2 * real.sqrt a) + a / (2 * real.sqrt a))
   else ∃ v, v = 100 ∧ total_cost v = 800 * ((1/4) * 100 + a / 100)) :=
  by
   sorry

end minimize_total_cost_l795_795572


namespace roots_cubic_polynomial_l795_795472

theorem roots_cubic_polynomial (a b c : ℝ)
  (h1 : Polynomial.root (3 * X ^ 3 - 6 * X ^ 2 + 99 * X - 2) a)
  (h2 : Polynomial.root (3 * X ^ 3 - 6 * X ^ 2 + 99 * X - 2) b)
  (h3 : Polynomial.root (3 * X ^ 3 - 6 * X ^ 2 + 99 * X - 2) c) :
  (a + b - 2) ^ 3 + (b + c - 2) ^ 3 + (c + a - 2) ^ 3 = -196 := 
sorry

end roots_cubic_polynomial_l795_795472


namespace mean_value_of_quadrilateral_angles_l795_795966

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795966


namespace indicator_light_signals_l795_795094

theorem indicator_light_signals:
  let n := 8 -- number of indicator lights
  let k := 4 -- number of indicator lights displayed
  let choices := 5.choose 2  -- number of ways to choose positions of displayed lights
  let signals_per_light := 16 -- each light can display 2^4 types of signals (2 options per light for 4 lights)
  in choices * signals_per_light = 160 := 
by 
  intro n k choices signals_per_light 
  sorry

end indicator_light_signals_l795_795094


namespace eighth_box_contains_65_books_l795_795937

theorem eighth_box_contains_65_books (total_books boxes first_seven_books per_box eighth_box : ℕ) :
  total_books = 800 →
  boxes = 8 →
  first_seven_books = 7 →
  per_box = 105 →
  eighth_box = total_books - (first_seven_books * per_box) →
  eighth_box = 65 := by
  sorry

end eighth_box_contains_65_books_l795_795937


namespace sqrt_six_greater_two_l795_795249

theorem sqrt_six_greater_two : Real.sqrt 6 > 2 :=
by
  sorry

end sqrt_six_greater_two_l795_795249


namespace even_function_a_zero_l795_795408

theorem even_function_a_zero (a : ℝ) (f : ℝ → ℝ) (h : f = (λ x, 2 - |x + a|)) 
  (even_f : ∀ x, f(-x) = f(x)) : a = 0 :=
sorry

end even_function_a_zero_l795_795408


namespace problem_solution_l795_795772

theorem problem_solution (m : ℤ) (x : ℤ) (h : 4 * x + 2 * m = 14) : x = 2 → m = 3 :=
by sorry

end problem_solution_l795_795772


namespace perimeter_C_l795_795922

theorem perimeter_C (x y : ℕ) 
  (h1 : 6 * x + 2 * y = 56)
  (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 := 
by
  sorry

end perimeter_C_l795_795922


namespace heights_proportional_l795_795746

-- Define the problem conditions
def sides_ratio (a b c : ℕ) : Prop := a / b = 3 / 4 ∧ b / c = 4 / 5

-- Define the heights
def heights_ratio (h1 h2 h3 : ℕ) : Prop := h1 / h2 = 20 / 15 ∧ h2 / h3 = 15 / 12

-- Problem statement: Given the sides ratio, prove the heights ratio
theorem heights_proportional {a b c h1 h2 h3 : ℕ} (h : sides_ratio a b c) :
  heights_ratio h1 h2 h3 :=
sorry

end heights_proportional_l795_795746


namespace triangle_height_l795_795121

theorem triangle_height (area base height : ℝ) (h1 : area = 500) (h2 : base = 50) (h3 : area = (1 / 2) * base * height) : height = 20 :=
sorry

end triangle_height_l795_795121


namespace hexagon_transformation_l795_795527

-- Define a shape composed of 36 identical small equilateral triangles
def Shape := { s : ℕ // s = 36 }

-- Define the number of triangles needed to form a hexagon
def TrianglesNeededForHexagon : ℕ := 18

-- Proof statement: Given a shape of 36 small triangles, we need 18 more triangles to form a hexagon
theorem hexagon_transformation (shape : Shape) : TrianglesNeededForHexagon = 18 :=
by
  -- This is our formalization of the problem statement which asserts
  -- that the transformation to a hexagon needs exactly 18 additional triangles.
  sorry

end hexagon_transformation_l795_795527


namespace discount_price_is_correct_l795_795936

noncomputable def first_discount_percentage : ℝ :=
let x := 18 in
x

theorem discount_price_is_correct :
  (550: ℝ) * (1 - first_discount_percentage / 100) * (0.88) = 396.88 :=
by
  -- Taking first_discount_percentage as 18
  let x := first_discount_percentage
  -- Verifying that the calculated sale price is indeed 396.88
  have h1 : (550: ℝ) * (1 - (x / 100)) * (0.88) = 396.88,
  { sorry }, -- Proof omitted
  exact h1

end discount_price_is_correct_l795_795936


namespace negation_of_forall_ge_zero_l795_795538

theorem negation_of_forall_ge_zero :
  ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 := by
  sorry

end negation_of_forall_ge_zero_l795_795538


namespace min_stool_height_l795_795230

/-
Alice needs to reach a ceiling fan switch located 15 centimeters below a 3-meter-tall ceiling.
Alice is 160 centimeters tall and can reach 50 centimeters above her head. She uses a stack of books
12 centimeters tall to assist her reach. We aim to show that the minimum height of the stool she needs is 63 centimeters.
-/

def ceiling_height_cm : ℕ := 300
def alice_height_cm : ℕ := 160
def reach_above_head_cm : ℕ := 50
def books_height_cm : ℕ := 12
def switch_below_ceiling_cm : ℕ := 15

def total_reach_with_books := alice_height_cm + reach_above_head_cm + books_height_cm
def switch_height_from_floor := ceiling_height_cm - switch_below_ceiling_cm

theorem min_stool_height : total_reach_with_books + 63 = switch_height_from_floor := by
  unfold total_reach_with_books switch_height_from_floor
  sorry

end min_stool_height_l795_795230


namespace iso_triangle_height_distance_diff_l795_795095

theorem iso_triangle_height_distance_diff
  (A B C M P Q D : Point)
  (h_iso : AB = AC)
  (h_M_on_extension : lies_on_extension BC M)
  (h_P_proj : orth_proj M AC = P)
  (h_Q_proj : orth_proj M AB = Q)
  (h_D_proj : orth_proj C AB = D)
  (CD : segment_length C D)
  : distance M Q - distance M P = CD :=
sorry

end iso_triangle_height_distance_diff_l795_795095


namespace perimeter_of_C_l795_795929

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end perimeter_of_C_l795_795929


namespace largest_integer_y_l795_795159

theorem largest_integer_y (y : ℤ) : (y / 4 + 3 / 7 : ℝ) < 9 / 4 → y ≤ 7 := by
  intros h
  sorry -- Proof needed

end largest_integer_y_l795_795159


namespace pipe_fill_time_without_leak_l795_795616

theorem pipe_fill_time_without_leak (T : ℝ) (h1 : (1 / 9 : ℝ) = 1 / T - 1 / 4.5) : T = 3 := 
by
  sorry

end pipe_fill_time_without_leak_l795_795616


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795990

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795990


namespace median_and_mode_correct_l795_795423

noncomputable def data_set : List ℕ := [3, 6, 4, 6, 4, 3, 6, 5, 7]

def median (l : List ℕ) : ℕ :=
  let sorted := l.sorted
  sorted.nthLe (sorted.length / 2) sorry

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ (acc, freq) x =>
    if l.count x > freq then (x, l.count x)
    else acc) (0, 0)

theorem median_and_mode_correct : median data_set = 5 ∧ mode data_set = 6 :=
by
  sorry

end median_and_mode_correct_l795_795423


namespace find_median_and_mode_l795_795417

def data_set := [3, 6, 4, 6, 4, 3, 6, 5, 7]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get! ((sorted.length - 1) / 2)

def mode (l : List ℕ) : ℕ :=
  l.foldl
    (λ counts n => counts.insert n (counts.find n |>.getD 0 + 1))
    (Std.HashMap.empty ℕ ℕ)
  |>.toList
  |>.foldl (λ acc (k, v) => if v > acc.2 then (k, v) else acc) (0, 0)
  |>.1

theorem find_median_and_mode :
  median data_set = 5 ∧ mode data_set = 6 :=
by
  sorry

end find_median_and_mode_l795_795417


namespace monotonically_decreasing_interval_l795_795292

variable (x : ℝ)

def quadratic_expr (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

def sqrt_function (x : ℝ) : ℝ := Real.sqrt (quadratic_expr x)

theorem monotonically_decreasing_interval :
    ∀ x, x ≤ 1 / 2 → 0 ≤ quadratic_expr x → monotonically_decreasing_on (sqrt_function) (Set.Iic (1 / 2)) := 
begin
  sorry
end

end monotonically_decreasing_interval_l795_795292


namespace constant_a5_and_S9_l795_795432

-- Defining an arithmetic sequence
def arithmetic_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

-- Defining the sum of the first n terms of the arithmetic sequence
def arithmetic_sum (a₁ d : ℝ) (n : ℕ) : ℝ := n / 2 * (2 * a₁ + (n - 1) * d)

-- Conditions given
variables (a₁ d : ℝ)
variable (C : ℝ) -- the constant value of a₂ + a₅ + a₈

-- Expressing the condition a₂ + a₅ + a₈ = C
axiom const_sum : arithmetic_seq a₁ d 2 + arithmetic_seq a₁ d 5 + arithmetic_seq a₁ d 8 = C

-- Statement to prove
theorem constant_a5_and_S9 : 
  (∃ k₅ : ℝ, ∀ (a₁ d : ℝ), arithmetic_seq a₁ d 5 = k₅) ∧ 
  (∃ k₉ : ℝ, ∀ (a₁ d : ℝ), arithmetic_sum a₁ d 9 = k₉) :=
by
  sorry

end constant_a5_and_S9_l795_795432


namespace geometric_series_sum_l795_795657

theorem geometric_series_sum : 
  let a := 2 in
  let r := 3 in
  let n := 8 in
  ∑ k in Finset.range n, a * r^k = 6560 :=
by 
  let a := 2
  let r := 3
  let n := 8
  have sum_formula : ∑ k in Finset.range n, a * r^k = a * (r^n - 1) / (r - 1) := sorry
  rw sum_formula
  calc
    a * (r^n - 1) / (r - 1) = 2 * (3^8 - 1) / (3 - 1) : by sorry
                            ... = 2 * 6560 / 2 : by sorry
                            ... = 6560 : by sorry

end geometric_series_sum_l795_795657


namespace same_number_of_friends_l795_795494

-- Definitions and conditions
def num_people (n : ℕ) := true   -- Placeholder definition to indicate the number of people
def num_friends (person : ℕ) (n : ℕ) : ℕ := sorry -- The number of friends a given person has (needs to be defined)
def friends_range (n : ℕ) := ∀ person, 0 ≤ num_friends person n ∧ num_friends person n < n

-- Theorem statement
theorem same_number_of_friends (n : ℕ) (h1 : num_people n) (h2 : friends_range n) : 
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ num_friends p1 n = num_friends p2 n :=
by
  sorry

end same_number_of_friends_l795_795494


namespace perimeter_C_l795_795909

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end perimeter_C_l795_795909


namespace sum_of_angles_quadrilateral_l795_795103

theorem sum_of_angles_quadrilateral (A B C D : Type) (angle : A → A → ℝ) (h1 : A = B) (h2 : B = C) (h3 : C = D) (h4 : D = A) : 
  (angle A B + angle B C + angle C D + angle D A = 360) :=
sorry

end sum_of_angles_quadrilateral_l795_795103


namespace meaningful_expression_range_l795_795021

theorem meaningful_expression_range (x : ℝ) : (∃ (y : ℝ), y = 5 / (x - 2)) ↔ x ≠ 2 := 
by
  sorry

end meaningful_expression_range_l795_795021


namespace people_in_room_l795_795569

theorem people_in_room (total_chairs seated_chairs total_people : ℕ) 
  (h1 : 3 * total_people = 5 * seated_chairs)
  (h2 : 4 * total_chairs = 5 * seated_chairs) 
  (h3 : total_chairs - seated_chairs = 8) : 
  total_people = 54 :=
by
  sorry

end people_in_room_l795_795569


namespace determine_values_of_a_b_l795_795739

noncomputable def find_a_b (a b : ℝ) : Prop :=
  ∀ (x : ℝ), (0 ≤ x ∧ x ≤ π / 2) → 
  (-5 ≤ 2 * a * sin (2 * x + π / 6) + a + b ∧ 
   2 * a * sin (2 * x + π / 6) + a + b ≤ 1)

theorem determine_values_of_a_b :
  ∃ (a b : ℝ), find_a_b a b ∧ ((a = 2 ∧ b = -5) ∨ (a = -2 ∧ b = 1)) :=
begin
  sorry
end

end determine_values_of_a_b_l795_795739


namespace sin_cos_identity_l795_795010

theorem sin_cos_identity (θ : ℝ) (h : Real.sin θ + Real.cos θ = 2 / 3) : Real.sin θ - Real.cos θ = (Real.sqrt 14) / 3 ∨ Real.sin θ - Real.cos θ = -(Real.sqrt 14) / 3 :=
by
  sorry

end sin_cos_identity_l795_795010


namespace max_soccer_balls_l795_795119

theorem max_soccer_balls (bought_balls : ℕ) (total_cost : ℕ) (available_money : ℕ) (unit_cost : ℕ)
    (h1 : bought_balls = 6) (h2 : total_cost = 168) (h3 : available_money = 500)
    (h4 : unit_cost = total_cost / bought_balls) :
    (available_money / unit_cost) = 17 := 
by
  sorry

end max_soccer_balls_l795_795119


namespace sequence_sum_inequality_l795_795006

theorem sequence_sum_inequality
  (n : ℕ)
  (a : fin n → ℝ)
  (h1 : ∀ i j, (i : ℕ) ≤ j → a i ≥ a j)
  (h2 : (∑ i, a i) = 1)
  (h3 : ∀ i, 0 ≤ a i) :
  (∑ i in finset.range n, (2 * (i + 1) - 1) * (a i)^2) ≤ 1 :=
sorry

end sequence_sum_inequality_l795_795006


namespace absolute_value_of_difference_of_quadratic_roots_l795_795286
noncomputable theory

open Real

def quadratic_roots (a b c : ℝ) : ℝ × ℝ :=
let discr := b^2 - 4 * a * c in
((−b + sqrt discr) / (2 * a), (−b - sqrt discr) / (2 * a))

theorem absolute_value_of_difference_of_quadratic_roots :
  ∀ r1 r2 : ℝ, 
  r1^2 - 7 * r1 + 12 = 0 → r2^2 - 7 * r2 + 12 = 0 →
  abs (r1 - r2) = 5 :=
by
  sorry

end absolute_value_of_difference_of_quadratic_roots_l795_795286


namespace probability_of_4_heads_in_6_flips_l795_795765

/-- A fair coin has equal probability of landing heads up or tails up. -/
def fair_coin : ℚ := 1 / 2

/-- Number of trials (flips) is 6. -/
def number_of_trials : ℕ := 6

/-- Number of successes (heads up) is 4. -/
def number_of_successes : ℕ := 4

theorem probability_of_4_heads_in_6_flips : 
  ∀ (p : ℚ) (n k : ℕ), 
  p = fair_coin ∧ n = number_of_trials ∧ k = number_of_successes → 
  (nat.choose n k * (p^k) * ((1 - p)^(n - k))) = 15 / 64 :=
by
  sorry

end probability_of_4_heads_in_6_flips_l795_795765


namespace c_horses_months_l795_795189

theorem c_horses_months (cost_total Rs_a Rs_b num_horses_a num_months_a num_horses_b num_months_b num_horses_c amount_paid_b : ℕ) (x : ℕ) 
  (h1 : cost_total = 841) 
  (h2 : Rs_a = 12 * 8)
  (h3 : Rs_b = 16 * 9)
  (h4 : amount_paid_b = 348)
  (h5 : 96 * (amount_paid_b / Rs_b) + (18 * x) * (amount_paid_b / Rs_b) = cost_total - amount_paid_b) :
  x = 11 :=
sorry

end c_horses_months_l795_795189


namespace max_points_of_intersection_l795_795080

-- Define the set of lines
def Lines := {L : Nat // 1 ≤ L ∧ L ≤ 120}

-- Define the special properties of certain lines
def is_parallel_to_others (L : Lines) : Prop := ∃ n : Nat, L.val = 5 * n
def passes_through_P (L : Lines) : Prop := ∃ n : Nat, L.val = 5 * n - 4
def passes_through_Q (L : Lines) : Prop := ∃ n : Nat, L.val = 3 * n - 2

-- Define the maximum number of points of intersection
theorem max_points_of_intersection : 
  ∃ max_intersections : Nat, 
    (∀ {L₁ L₂ : Lines}, L₁ ≠ L₂ → intersects_at_point L₁ L₂ P → intersects_at_point L₁ L₂ Q → 
      number_of_intersections (L₁, L₂) ≤ max_intersections) ∧ 
    max_intersections = 6589 := 
sorry

end max_points_of_intersection_l795_795080


namespace triangle_is_isosceles_l795_795776

open Real

-- Define the basic setup of the triangle and the variables involved
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides opposite to A, B, and C respectively
variables (h1 : a * cos B = b * cos A) -- Given condition: a * cos B = b * cos A

-- The theorem stating that the given condition implies the triangle is isosceles
theorem triangle_is_isosceles (h1 : a * cos B = b * cos A) : A = B :=
sorry

end triangle_is_isosceles_l795_795776


namespace christine_aquafaba_needed_l795_795654

-- Define the number of tablespoons per egg white
def tablespoons_per_egg_white : ℕ := 2

-- Define the number of egg whites per cake
def egg_whites_per_cake : ℕ := 8

-- Define the number of cakes
def number_of_cakes : ℕ := 2

-- Express the total amount of aquafaba needed
def aquafaba_needed : ℕ :=
  tablespoons_per_egg_white * egg_whites_per_cake * number_of_cakes

-- Statement asserting the amount of aquafaba needed is 32
theorem christine_aquafaba_needed : aquafaba_needed = 32 := by
  sorry

end christine_aquafaba_needed_l795_795654


namespace B_work_time_l795_795604

noncomputable def workRateA (W : ℝ): ℝ := W / 14
noncomputable def combinedWorkRate (W : ℝ): ℝ := W / 10

theorem B_work_time (W : ℝ) :
  ∃ T : ℝ, (W / T) = (combinedWorkRate W) - (workRateA W) ∧ T = 35 :=
by {
  use 35,
  sorry
}

end B_work_time_l795_795604


namespace perimeter_C_l795_795910

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end perimeter_C_l795_795910


namespace fg_neg_5_l795_795761

def f (x : ℝ) : ℝ := 3 - Real.sqrt x
def g (x : ℝ) : ℝ := 5 * x + 2 * x^2

theorem fg_neg_5 : f (g (-5)) = -2 := by
  sorry

end fg_neg_5_l795_795761


namespace min_m_n_sum_l795_795519

theorem min_m_n_sum (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : 90 * m = n ^ 3) : m + n = 120 :=
begin
  sorry,
end

end min_m_n_sum_l795_795519


namespace total_distance_traveled_l795_795866

theorem total_distance_traveled :
  let common_difference := -11
  let initial_distance := 44
  let n := 5
  let distances := List.range n |>.map (λ i => initial_distance + i * common_difference)
  ∑ k in distances, k = 110 :=
by
  sorry

end total_distance_traveled_l795_795866


namespace circle_properties_l795_795125

theorem circle_properties : 
  ∀ {x y : ℝ}, 
    x^2 + y^2 - 2 * x + 4 * y = 0 → 
    (exists (c : ℝ × ℝ) r, c = (1, -2) ∧ r = real.sqrt 5 ∧ 
    ((x - c.1)^2 + (y - c.2)^2 = r^2)) :=
by 
  sorry

end circle_properties_l795_795125


namespace max_value_of_product_l795_795726

theorem max_value_of_product
  (a : ℕ → ℝ)
  (h : ∀ i ∈ Finset.range 2015, 9 * a i > 11 * (a (i + 1))^2) :
  (∏ i in Finset.range 2016, (a i - (a (i + 1 % 2016))^2)) ≤ (1 / 4) ^ 2016 :=
sorry

end max_value_of_product_l795_795726


namespace proof_problem_l795_795007

theorem proof_problem (a b c d x : ℝ)
  (h1 : c = 6 * d)
  (h2 : 2 * a = 1 / (-b))
  (h3 : abs x = 9) :
  (2 * a * b - 6 * d + c - x / 3 = -4) ∨ (2 * a * b - 6 * d + c - x / 3 = 2) :=
by
  sorry

end proof_problem_l795_795007


namespace nearest_integer_to_power_l795_795165

theorem nearest_integer_to_power (a b : ℝ) (h1 : a = 3) (h2 : b = sqrt 2) : 
  abs ((a + b)^6 - 3707) < 0.5 :=
by
  sorry

end nearest_integer_to_power_l795_795165


namespace radius_of_inscribed_circle_l795_795120

theorem radius_of_inscribed_circle 
  (a b : ℝ)
  (S : ℝ := 24)
  (c : ℝ := 10)
  (h1 : a * b / 2 = S)
  (h2 : a^2 + b^2 = c^2) :
  let r := (a + b - c) / 2 in
  r = 2 := 
by
  sorry

end radius_of_inscribed_circle_l795_795120


namespace total_balloons_correct_l795_795450

-- Define the number of blue balloons Joan and Melanie have
def Joan_balloons : ℕ := 40
def Melanie_balloons : ℕ := 41

-- Define the total number of blue balloons
def total_balloons : ℕ := Joan_balloons + Melanie_balloons

-- Prove that the total number of blue balloons is 81
theorem total_balloons_correct : total_balloons = 81 := by
  sorry

end total_balloons_correct_l795_795450


namespace unique_zero_point_l795_795366

noncomputable def f (x : ℝ) : ℝ := Real.log2 (1 - x) + Real.log2 (1 + x)
def g (x : ℝ) : ℝ := 1 / 2 - x^2
noncomputable def h (x : ℝ) : ℝ := f x + g x

theorem unique_zero_point (a b : ℝ) (ha : 0 < a) (hb : b < 1) :
  ∃! x ∈ set.Ioo a b, h x = 0 := sorry

end unique_zero_point_l795_795366


namespace granary_circumference_l795_795033

-- Define the necessary constants and conversions
def zhang_to_chi (zhang : ℝ) : ℝ := 10 * zhang
def chi_to_cun (chi : ℝ) : ℝ := 10 * chi
def hu_to_cubic_chi (hu : ℝ) : ℝ := 1.62 * hu
def pi_approx : ℝ := 3

-- Define the given conditions
def height_chi : ℝ := 10 + 3 + (1 / 3) * (1 / 10)
def volume_hu : ℝ := 2000
def volume_cubic_chi : ℝ := hu_to_cubic_chi volume_hu

-- The base area S of the cylinder
def base_area : ℝ := volume_cubic_chi / height_chi

-- The radius of the granary base
def radius : ℝ := real.sqrt (base_area / pi_approx)

-- The circumference of the cylinder base
def circumference_chi : ℝ := 2 * pi_approx * radius

-- Convert circumference to zhang and chi
def circumference_zhang : ℝ := circumference_chi / 10
def circumference_addition (zhang_part chi_part : ℝ) : (ℝ × ℝ) :=
  let total_chi := (zhang_part * 10) + chi_part in
  (real.floor (total_chi / 10), total_chi % 10)

-- Assert that the circumference is as expected
theorem granary_circumference : circumference_addition 5 4 = (5, 4) :=
sorry

end granary_circumference_l795_795033


namespace sum_of_remainders_correct_l795_795528

noncomputable def sum_of_possible_remainders : ℕ :=
  let remainders := list.range 7 |>.map (λ m => (1110 * m + 123) % 41)
  remainders.sum

theorem sum_of_remainders_correct :
  sum_of_possible_remainders = 93 :=
by
  sorry

end sum_of_remainders_correct_l795_795528


namespace number_of_p_less_than_3025_l795_795318

-- Given Conditions
variable (p : ℕ)
variable (x y : ℕ)
variable (AB BC CD AD : ℕ)

-- Quadrilateral with right angles at B and C
def is_right_angle_quadrilateral (A B C D : ℕ) : Prop :=
  AB = 3 ∧ CD = 2 * AD ∧ p = 3 + x + 3 * y ∧
  x * x + 4 * y * y - 12 * y + 9 = y * y ∧ 
  x * x = -3 * y * y + 12 * y - 9

-- The goal is to find the number of different values of p < 3025
theorem number_of_p_less_than_3025 : 
  {p : ℕ | p < 3025 ∧ ∃ x y, is_right_angle_quadrilateral 1 2 3 4} = 40 :=
begin
  sorry
end

end number_of_p_less_than_3025_l795_795318


namespace balls_in_boxes_l795_795003

theorem balls_in_boxes : ∃ n : ℕ, n = 32 ∧ (∃ f : Fin 4 × Fin 4 → ℕ, 
  (∀ (b : Fin 4), ∑ (a : Fin 4), f (a, b) = 4)
  ∧ (∀ (a b : Fin 4), f (a, b) ≤ 4)
  ∧ (∀ (a b : Fin 4), f (a, b) ≥ 0)) := 
begin
  sorry
end

end balls_in_boxes_l795_795003


namespace exists_Brocard_point_concurrency_of_AA1_BB1_CC1_at_P_l795_795591

-- Definitions based on the problem's conditions
variables (A B C P A1 B1 C1 : Point)

-- Assume we have a triangle ABC
axiom triangle_ABC : triangle A B C

-- Similar triangles constructed externally on the sides of ABC
axioms 
  (similar_CA1B : similar_triangle (triangle C A1 B) (triangle A B C))
  (similar_CAB1 : similar_triangle (triangle C A B1) (triangle A B C))
  (similar_C1AB : similar_triangle (triangle C1 A B) (triangle A B C))

-- Proving part (a) - the existence of point P
theorem exists_Brocard_point :
  ∃ P : Point, ∠ A B P = ∠ C A P ∧ ∠ C A P = ∠ B C P :=
sorry

-- Proving part (b) - concurrency at P
theorem concurrency_of_AA1_BB1_CC1_at_P :
  ∃ P : Point, concurrency (A A1) (B B1) (C C1) ∧
  (∠ A B P = ∠ C A P ∧ ∠ C A P = ∠ B C P) :=
sorry

end exists_Brocard_point_concurrency_of_AA1_BB1_CC1_at_P_l795_795591


namespace sequence_all_positive_integers_l795_795253

noncomputable def a : ℕ → ℤ
| 0 := 1
| 1 := 1
| 2 := 1
| (n + 3) := (2019 + a (n + 2) * a (n + 1)) / a n

theorem sequence_all_positive_integers : ∀ n : ℕ, 0 < a n := 
by 
  sorry

end sequence_all_positive_integers_l795_795253


namespace ribbon_left_after_wrapping_l795_795854

def total_ribbon_needed (gifts : ℕ) (ribbon_per_gift : ℝ) : ℝ :=
  gifts * ribbon_per_gift

def remaining_ribbon (initial_ribbon : ℝ) (used_ribbon : ℝ) : ℝ :=
  initial_ribbon - used_ribbon

theorem ribbon_left_after_wrapping : 
  ∀ (gifts : ℕ) (ribbon_per_gift initial_ribbon : ℝ),
  gifts = 8 →
  ribbon_per_gift = 1.5 →
  initial_ribbon = 15 →
  remaining_ribbon initial_ribbon (total_ribbon_needed gifts ribbon_per_gift) = 3 :=
by
  intros gifts ribbon_per_gift initial_ribbon h1 h2 h3
  rw [h1, h2, h3]
  simp [total_ribbon_needed, remaining_ribbon]
  sorry

end ribbon_left_after_wrapping_l795_795854


namespace incorrect_diff_squares_l795_795233

open Classical

variables {a b x : ℝ}

theorem incorrect_diff_squares (a b x : ℝ)
  (eqA : (-a + b) * (-a - b) = a^2 - b^2)
  (eqB : (x + 1) * (1 - x) = 1 - x^2)
  (eqC : (a + b)^2 * (a - b)^2 = (a^2 - b^2)^2)
  (eqD : (2x + 3) * (2x - 3) = 2x^2 - 9) : eqD = false :=
by sorry

end incorrect_diff_squares_l795_795233


namespace mean_value_of_quadrilateral_angles_l795_795997

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l795_795997


namespace value_of_first_equation_l795_795362

variables (x y z w : ℝ)

theorem value_of_first_equation (h1 : xw + yz = 8) (h2 : (2 * x + y) * (2 * z + w) = 20) : xz + yw = 1 := by
  sorry

end value_of_first_equation_l795_795362


namespace sampling_method_is_systematic_l795_795196

def is_systematic_sampling (selected : list ℕ) : Prop :=
  selected = [3, 7, 13, 17, 23, 27, 33, 37, 43, 47, 53, 57, 63, 67, 73, 77, 83, 87, 93, 97]

theorem sampling_method_is_systematic :
  is_systematic_sampling [3, 7, 13, 17, 23, 27, 33, 37, 43, 47, 53, 57, 63, 67, 73, 77, 83, 87, 93, 97] :=
sorry

end sampling_method_is_systematic_l795_795196


namespace max_volume_is_correct_l795_795210

noncomputable def max_cylinder_volume (perimeter : ℝ) : ℝ :=
  let l_w_sum := perimeter / 2 in
  let l_opt := (2 / 3) * l_w_sum in
  let w_opt := l_w_sum - l_opt in
  let r := l_opt / (2 * Real.pi) in
  let h := w_opt in
  Real.pi * r^2 * h

theorem max_volume_is_correct :
  max_cylinder_volume 20 = 1000 / (27 * Real.pi) :=
by
  sorry

end max_volume_is_correct_l795_795210


namespace convex_octagon_triangulations_l795_795589

noncomputable def T : ℕ → ℕ 
| 3     := 1
| (n+1) := if n ≥ 3 then 2 * T n + ∑ k in finset.range (n - 2), T (k + 3) * T (n + 1 - k) else 0

theorem convex_octagon_triangulations : T 8 = 132 :=
by
  sorry

end convex_octagon_triangulations_l795_795589


namespace sum_C_D_eq_one_fifth_l795_795671

theorem sum_C_D_eq_one_fifth (D C : ℚ) :
  (∀ x : ℚ, (Dx - 13) / (x^2 - 9 * x + 20) = C / (x - 4) + 5 / (x - 5)) →
  (C + D) = 1/5 :=
by
  sorry

end sum_C_D_eq_one_fifth_l795_795671


namespace units_digit_sum_factorials_l795_795579

/-- The units digit of the sum 1! + 2! + ... + 9! is 3. -/
theorem units_digit_sum_factorials : 
  Nat.unitsDigit (1! + 2! + 3! + 4! + 5! + 6! + 7! + 8! + 9!) = 3 := 
  by 
    sorry

end units_digit_sum_factorials_l795_795579


namespace inequality_pow_gt_linear_l795_795493

theorem inequality_pow_gt_linear {a : ℝ} (n : ℕ) (h₁ : a > -1) (h₂ : a ≠ 0) (h₃ : n ≥ 2) :
  (1 + a:ℝ)^n > 1 + n * a :=
sorry

end inequality_pow_gt_linear_l795_795493


namespace count_triples_l795_795102

open Finset

variable (n : ℕ) (s : Finset (Fin n))

-- Definitions of the subsets A, B, C
variables {A B C : Finset (Fin n)}

-- Conditions
def condition1 : Prop := A ∩ B ∩ C = ∅
def condition2 : Prop := A ∩ B ≠ ∅
def condition3 : Prop := C ∩ B ≠ ∅

-- The theorem statement
theorem count_triples (n : ℕ) :
  ∃ (A B C : Finset (Fin n)),
    condition1 ∧ condition2 ∧ condition3 ∧
    (7 ^ n - 2 * 6 ^ n + 5 ^ n) :=
  sorry

end count_triples_l795_795102


namespace inf_div_p_n2n_plus_one_n_div_3_n2n_plus_one_l795_795460

theorem inf_div_p_n2n_plus_one (p : ℕ) (hp : Nat.Prime p) (h_odd : p % 2 = 1) :
  ∃ᶠ n in at_top, p ∣ (n * 2^n + 1) :=
sorry

theorem n_div_3_n2n_plus_one :
  (∃ k : ℕ, ∀ n, n = 6 * k + 1 ∨ n = 6 * k + 2 → 3 ∣ (n * 2^n + 1)) :=
sorry

end inf_div_p_n2n_plus_one_n_div_3_n2n_plus_one_l795_795460


namespace perimeter_C_l795_795895

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end perimeter_C_l795_795895


namespace mean_value_of_quadrilateral_angles_l795_795994

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l795_795994


namespace perimeter_C_l795_795914

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end perimeter_C_l795_795914


namespace hexagonal_pyramid_edge_sum_l795_795611

-- Define the conditions
def base_edge_length : ℕ := 8
def slant_height_length : ℕ := 13

-- Define a proposition for the total edge length of the regular hexagonal pyramid
theorem hexagonal_pyramid_edge_sum (h1 : base_edge_length = 8) (h2 : slant_height_length = 13) : 
  6 * base_edge_length + 6 * slant_height_length = 126 :=
by
  -- Using the conditions to derive the result
  rw [h1, h2]
  exact rfl  -- Placeholder for the proof steps

end hexagonal_pyramid_edge_sum_l795_795611


namespace sum_of_reciprocals_of_roots_l795_795641

theorem sum_of_reciprocals_of_roots 
  (r₁ r₂ : ℝ)
  (h_roots : ∀ (x : ℝ), x^2 - 17*x + 8 = 0 → (∃ r, (r = r₁ ∨ r = r₂) ∧ x = r))
  (h_sum : r₁ + r₂ = 17)
  (h_prod : r₁ * r₂ = 8) :
  1/r₁ + 1/r₂ = 17/8 := 
by
  sorry

end sum_of_reciprocals_of_roots_l795_795641


namespace boat_to_stream_ratio_l795_795613

-- Define the speed of the boat in still water and the average speed of the stream.
variables (B S : ℝ)

-- Define the condition provided: the time taken to row a distance upstream is twice the time taken downstream.
axiom rowing_time_ratio (h : 2 * (B - S) = (B + S)) : true

-- Define the theorem to prove the ratio of the speed of the boat to the average speed of the stream is 3.
theorem boat_to_stream_ratio (h : rowing_time_ratio B S) : B / S = 3 :=
sorry

end boat_to_stream_ratio_l795_795613


namespace average_of_next_consecutive_integers_l795_795567

theorem average_of_next_consecutive_integers (a : ℕ) :
  let b := (a + 1 + (a + 2) + (a + 3)) / 3 in
  let average := (b + (b + 1) + (b + 2)) / 3 in
  average = a + 3 :=
by {
  sorry
}

end average_of_next_consecutive_integers_l795_795567


namespace ratio_ANCO_to_face_is_one_fourth_l795_795031

-- Define the cube and necessary points
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (1, 0, 0)
def C : ℝ × ℝ × ℝ := (1, 1, 0)
def D : ℝ × ℝ × ℝ := (0, 1, 0)
def E : ℝ × ℝ × ℝ := (0, 0, 1)
def F : ℝ × ℝ × ℝ := (1, 0, 1)
def G : ℝ × ℝ × ℝ := (1, 1, 1)
def H : ℝ × ℝ × ℝ := (0, 1, 1)

def J : ℝ × ℝ × ℝ := ((1 / 2), 0, 0)
def K : ℝ × ℝ × ℝ := (0, 1, (1 / 2))
def L : ℝ × ℝ × ℝ := (1, 1, (1 / 2))
def M : ℝ × ℝ × ℝ := ((1 / 2), 0, 1)

def N : ℝ × ℝ × ℝ := ((1 / 4), 0, (1 / 2))
def O : ℝ × ℝ × ℝ := ((1 / 2), 1, (3 / 4))

-- Define areas and the ratio R
def area_of_face : ℝ := 1

def area_of_ANCO : ℝ := (1 / 2)

def R : ℝ := area_of_ANCO / area_of_face

-- Prove that R = 1 / 4
theorem ratio_ANCO_to_face_is_one_fourth : R = (1 / 4) :=
by
  unfold R
  unfold area_of_ANCO
  unfold area_of_face
  sorry

end ratio_ANCO_to_face_is_one_fourth_l795_795031


namespace aquafaba_needed_for_cakes_l795_795648

def tablespoons_of_aquafaba_for_egg_whites (n_egg_whites : ℕ) : ℕ :=
  2 * n_egg_whites

def total_egg_whites_needed (cakes : ℕ) (egg_whites_per_cake : ℕ) : ℕ :=
  cakes * egg_whites_per_cake

theorem aquafaba_needed_for_cakes (cakes : ℕ) (egg_whites_per_cake : ℕ) :
  tablespoons_of_aquafaba_for_egg_whites (total_egg_whites_needed cakes egg_whites_per_cake) = 32 :=
by
  have h1 : cakes = 2 := sorry
  have h2 : egg_whites_per_cake = 8 := sorry
  sorry

end aquafaba_needed_for_cakes_l795_795648


namespace f_conjecture_l795_795068

noncomputable def f (x : ℝ) : ℝ := 1 / (4^x + 2)

theorem f_conjecture (x : ℝ) : f(x) + f(1 - x) = 1 / 2 := 
by 
  sorry

end f_conjecture_l795_795068


namespace gcd_fact_8_10_l795_795303

theorem gcd_fact_8_10 : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = 40320 := by
  -- No proof needed
  sorry

end gcd_fact_8_10_l795_795303


namespace find_AE_length_l795_795806

noncomputable def length_AE {A B C D E : Type*} [ConvexQuadrilateral A B C D] 
  (angle_ABD_BCD : ∠ABD = ∠BCD) (AD : ℝ) (AD_val : AD = 1000) (BD : ℝ) (BD_val : BD = 2000) 
  (BC : ℝ) (BC_val : BC = 2001) (DC : ℝ) (DC_val : DC = 1999) 
  (angle_ABD_ECD : ∠ABD = ∠ECD) (E_on_DB : E ∈ segment D B) : ℝ :=
1000

theorem find_AE_length {A B C D E : Type*} [ConvexQuadrilateral A B C D] 
  (angle_ABD_BCD : ∠ABD = ∠BCD) (AD : ℝ) (AD_val : AD = 1000) (BD : ℝ) (BD_val : BD = 2000) 
  (BC : ℝ) (BC_val : BC = 2001) (DC : ℝ) (DC_val : DC = 1999) 
  (angle_ABD_ECD : ∠ABD = ∠ECD) (E_on_DB : E ∈ segment D B) : 
  length_AE angle_ABD_BCD AD AD_val BD BD_val BC BC_val DC DC_val angle_ABD_ECD E_on_DB = 1000 :=
sorry

end find_AE_length_l795_795806


namespace line_segment_parameterization_l795_795930

theorem line_segment_parameterization :
  ∃ (e f g h : ℤ),
    (f = 1 ∧ h = -3 ∧ e + f = -4 ∧ g + h = 6) ∧ 
    (e^2 + f^2 + g^2 + h^2 = 116) :=
begin
  use [-5, 1, 9, -3],
  split,
  { split, 
    { refl, },
    split, 
    { refl, }, 
    { split,
      { norm_num, },
      { norm_num, } } },
  norm_num,
  sorry
end

end line_segment_parameterization_l795_795930


namespace length_of_third_triangle_l795_795234

theorem length_of_third_triangle
  (T1_side_length : ℝ)
  (sum_perimeters : ℝ)
  (h1 : T1_side_length = 80)
  (h2 : sum_perimeters = 480) :
  let T2_side_length := T1_side_length / 2,
      T3_side_length := T2_side_length / 2 in
  T3_side_length = 20 :=
by
  -- proof omitted
  sorry

end length_of_third_triangle_l795_795234


namespace perimeter_C_l795_795911

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end perimeter_C_l795_795911


namespace remaining_milk_correct_l795_795142

def arranged_milk : ℝ := 21.52
def sold_milk : ℝ := 12.64
def remaining_milk (total : ℝ) (sold : ℝ) : ℝ := total - sold

theorem remaining_milk_correct :
  remaining_milk arranged_milk sold_milk = 8.88 :=
by
  sorry

end remaining_milk_correct_l795_795142


namespace unique_real_solution_k_eq_35_over_4_l795_795706

theorem unique_real_solution_k_eq_35_over_4 :
  ∃ k : ℚ, (∀ x : ℝ, (x + 5) * (x + 3) = k + 3 * x) ↔ (k = 35 / 4) :=
by
  sorry

end unique_real_solution_k_eq_35_over_4_l795_795706


namespace last_triangle_perimeter_l795_795817

variables (T₁ : Triangle) (n : ℕ)

-- Define initial triangle T1
def T₁ : Triangle := ⟨1001, 1002, 1003⟩

-- Define a function to determine the lengths AD, BE, and CF
noncomputable def next_triangle_lengths (T : Triangle) : Triangle :=
  let ⟨a, b, c⟩ := T in
  (c + b - a) / 2, (a + c - b) / 2, (a + b - c) / 2

-- Recursive function to determine the nth triangle
noncomputable def T_seq : ℕ → Triangle
| 0 => T₁
| (n+1) => next_triangle_lengths (T_seq n)

-- Function to calculate the perimeter of a triangle
def perimeter (T : Triangle) : ℚ :=
  let ⟨a, b, c⟩ := T in a + b + c

-- Function to compute the nth perimeter
noncomputable def T_perimeter (n : ℕ) : ℚ := perimeter (T_seq n)

-- Proof statement
theorem last_triangle_perimeter :
  ∃ n : ℕ, T_perimeter n = 1503 / 256 :=
sorry

end last_triangle_perimeter_l795_795817


namespace option_c_may_not_be_true_l795_795585

theorem option_c_may_not_be_true (a b c : ℝ) :
  a > b → (¬ ac^2 > bc^2) ∨ (c ≠ 0) := by
sorry

end option_c_may_not_be_true_l795_795585


namespace chen_yuxi_scores_properties_l795_795795

-- Define the list of scores
def scores : List ℝ := [9.5, 9.0, 9.0, 9.0, 10.0, 9.5, 9.0]

-- Define a function to calculate the mode
def mode (l : List ℝ) : ℝ := 
  l.groupBy id |>.maximumBy (·.length) |>.head! |>.1

-- Define a function to calculate the median
def median (l : List ℝ) : ℝ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get! (sorted.length / 2)

-- Statement to prove
theorem chen_yuxi_scores_properties :
  mode scores = 9.0 ∧ median scores = 9.0 :=
by
  sorry

end chen_yuxi_scores_properties_l795_795795


namespace f_at_2_l795_795370

def f (x : ℝ) : ℝ := 2^x + 2

theorem f_at_2 : f 2 = 6 := by
  sorry

end f_at_2_l795_795370


namespace PQRS_product_l795_795729

noncomputable def P : ℝ := (Real.sqrt 2023 + Real.sqrt 2024)
noncomputable def Q : ℝ := (-Real.sqrt 2023 - Real.sqrt 2024)
noncomputable def R : ℝ := (Real.sqrt 2023 - Real.sqrt 2024)
noncomputable def S : ℝ := (Real.sqrt 2024 - Real.sqrt 2023)

theorem PQRS_product : (P * Q * R * S) = 1 := 
by 
  sorry

end PQRS_product_l795_795729


namespace area_of_region_l795_795289

noncomputable theory

open Real

-- Definitions of the conditions
def condition1 (x y : ℝ) : Prop := abs (x + 1) ≥ abs (y + 3)
def condition2 (x y : ℝ) : Prop := x^2 + y^2 + 6*y + 2*x - 26 ≤ 64

-- The area of the region satisfying the conditions
theorem area_of_region : 
  ∃ (A : ℝ), 
  (∀ (x y : ℝ), condition1 x y → condition2 x y → A = 32 * π) := 
sorry

end area_of_region_l795_795289


namespace perimeter_C_l795_795896

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end perimeter_C_l795_795896


namespace min_value_sum_l795_795262

theorem min_value_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b) + b / (5 * c) + c / (7 * a)) ≥ (3 / (real.cbrt 105)) :=
by
  sorry

end min_value_sum_l795_795262


namespace StatementA_StatementB_StatementC_StatementD_l795_795583

section StatementA
variable (x k : ℝ)
-- Define condition for f(x)
def f (x : ℝ) (k : ℝ) := sqrt (2 * k * x^2 - 3 * k * x + k + 1)

-- Define the theorem for the range of k
theorem StatementA : (∀ x, 2 * k * x^2 - 3 * k * x + k + 1 ≥ 0) → (0 < k ∧ k ≤ 8) :=
sorry
end StatementA

section StatementB
variable (x a : ℝ)
-- Define condition for the inequality
theorem StatementB : (x ≥ 2 → x + 1/x ≥ a) → a ≤ 5/2 :=
sorry
end StatementB

section StatementC
variable (a b : ℝ)
-- Define condition for the given equation
theorem StatementC : (a > 0 ∧ b > 0 ∧ 2 * a + 8 * b = a * b) → a + b = 18 :=
sorry
end StatementC

section StatementD
variable (a : ℝ)
-- Define the piecewise function f 
def f (x : ℝ) := if x ≤ 0 then 3 * x + 5 else x + 1 / x

-- Define the condition for f(f(a)) = 2
theorem StatementD : f (f a) = 2 → a = -2 ∨ a = -4/3 :=
sorry
end StatementD

end StatementA_StatementB_StatementC_StatementD_l795_795583


namespace fraction_of_second_year_among_non_third_year_l795_795190

variable (total_students : ℕ)
variable (third_year_students : ℕ)
variable (second_year_students : ℕ)

-- Assume the total number of students is 100
def total_students := 100

-- 50 percent are third-year students
def third_year_students := 50

-- 70 percent are not second-year students, hence 30 percent are second-year students
def second_year_students := 30

-- Remaining students who are not third-year students
def non_third_year_students := total_students - third_year_students

-- Computation: fraction of second-year students who are not third-year students
def fraction_second_among_non_third : ℚ :=
  (second_year_students : ℚ) / (non_third_year_students : ℚ)

theorem fraction_of_second_year_among_non_third_year :
  fraction_second_among_non_third = 3 / 5 :=
by
  rw [fraction_second_among_non_third]
  -- Convert definitions to numbers
  trivial

end fraction_of_second_year_among_non_third_year_l795_795190


namespace expand_expression_l795_795273

variable {x y z : ℝ}

theorem expand_expression :
  (2 * x + 5) * (3 * y + 15 + 4 * z) = 6 * x * y + 30 * x + 8 * x * z + 15 * y + 20 * z + 75 :=
by
  sorry

end expand_expression_l795_795273


namespace mean_value_of_quadrilateral_angles_l795_795968

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795968


namespace pond_field_area_ratio_l795_795133

theorem pond_field_area_ratio
  (l : ℝ) (w : ℝ) (A_field : ℝ) (A_pond : ℝ)
  (h1 : l = 2 * w)
  (h2 : l = 16)
  (h3 : A_field = l * w)
  (h4 : A_pond = 8 * 8) :
  A_pond / A_field = 1 / 2 :=
by
  sorry

end pond_field_area_ratio_l795_795133


namespace line_segment_from_C_to_hypotenuse_l795_795428

theorem line_segment_from_C_to_hypotenuse :
  ∀ (BC AC: ℝ) (hyp x θ: ℝ),
    BC = 5 ∧ AC = 12 → 
    hyp = real.sqrt (BC^2 + AC^2) →
    hyp = 13 →
    θ = real.pi / 6 →
    x = (real.sqrt 3 / 2) * hyp →
    x = (13 * real.sqrt 3 / 2) :=
by
  intros BC AC hyp x θ BC_AC_eqs hyp_eq_hyp hyp_eq theta_eq x_eq
  sorry

end line_segment_from_C_to_hypotenuse_l795_795428


namespace C_is_orthocenter_of_OO1O2_l795_795032

-- Assuming a right triangle ABC with certain constructs
variables {A B C D O O1 O2 : Type}
          {triangle : Type} [right_triangle triangle A B C]
          {is_altitude : Type} [altitude is_altitude C D A B]
          {incenter : Type} [incenter O triangle ABC]
                          [incenter O1 triangle ACD]
                          [incenter O2 triangle BCD]

-- The statement representing the problem
theorem C_is_orthocenter_of_OO1O2 :
  orthocenter C (triangle O O1 O2) :=
sorry

end C_is_orthocenter_of_OO1O2_l795_795032


namespace neither_rain_nor_snow_l795_795546

theorem neither_rain_nor_snow 
  (p_rain : ℚ)
  (p_snow : ℚ)
  (independent : Prop) 
  (h_rain : p_rain = 4/10)
  (h_snow : p_snow = 1/5)
  (h_independent : independent)
  : (1 - p_rain) * (1 - p_snow) = 12 / 25 := 
by
  sorry

end neither_rain_nor_snow_l795_795546


namespace no_such_constant_l795_795827

noncomputable def f : ℚ → ℚ := sorry

theorem no_such_constant (h : ∀ x y : ℚ, ∃ k : ℤ, f (x + y) - f x - f y = k) :
  ¬ ∃ c : ℚ, ∀ x : ℚ, ∃ k : ℤ, f x - c * x = k := 
sorry

end no_such_constant_l795_795827


namespace fraction_broken_light_bulbs_l795_795113

theorem fraction_broken_light_bulbs (
  bk_foyer : ℕ, -- 10 broken light bulbs in the foyer.
  tot_kit : ℕ, -- 35 light bulbs in the kitchen.
  not_broken_total : ℕ, -- 34 light bulbs not broken in both the foyer and kitchen.
  third_foyer : ∃ x, (bk_foyer = x) ∧ (3 * x = 3 * 10) -- A third of the light bulbs in the foyer are broken (implicitly given by 10 broken bulbs).
  ): (21 = (tot_kit - (31 - bk_foyer))) →

  -- Conditions translated:
  (bk_foyer = 10) ∧
  (tot_kit = 35) ∧
  (not_broken_total = 34) →

  -- Math equivalent proof:
  fraction_broken : ((21 : ℕ) / 35) = (3 / 5) :=
sorry

end fraction_broken_light_bulbs_l795_795113


namespace perimeter_C_l795_795915

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end perimeter_C_l795_795915


namespace range_of_a_l795_795773

theorem range_of_a (a : ℝ) : (∃ x > 0, exp(x) * (x + a) < 1) → a < 1 :=
sorry

end range_of_a_l795_795773


namespace perimeter_C_l795_795917

theorem perimeter_C (x y : ℕ) 
  (h1 : 6 * x + 2 * y = 56)
  (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 := 
by
  sorry

end perimeter_C_l795_795917


namespace westward_fish_caught_fraction_l795_795598

theorem westward_fish_caught_fraction :
  let westward := 1800 in
  let eastward := 3200 in
  let north := 500 in
  let eastward_caught_fraction := (2 / 5 : ℚ) in
  let total_initial := westward + eastward + north in
  let remaining := 2870 in
  let total_caught := total_initial - remaining in
  let eastward_caught := eastward_caught_fraction * eastward in
  ∃ (x : ℚ), eastward_caught + x * westward = total_caught ∧ x = 3 / 4 :=
by
  sorry

end westward_fish_caught_fraction_l795_795598


namespace four_digit_even_numbers_with_thousand_digit_one_l795_795385

theorem four_digit_even_numbers_with_thousand_digit_one : 
  let count := 1 * 10 * 10 * 5 in
  count = 500 :=
by
  sorry

end four_digit_even_numbers_with_thousand_digit_one_l795_795385


namespace calculate_expression_l795_795242

theorem calculate_expression :
  let a := 2 * Real.sqrt 2 - Real.pi
  let e := (a) ^ 0
  let c := -4 * Real.cos (Real.pi / 3)  -- (Real.pi / 3) radians is 60 degrees
  let abs_sqrt := Real.abs (Real.sqrt 2 - 2)
  let sqrt_18 := -(3 * Real.sqrt 2)
  e + c + abs_sqrt + sqrt_18 = 1 - 4 * Real.sqrt 2 :=
by {
  sorry
}

end calculate_expression_l795_795242


namespace point_A_lies_on_plane_l795_795337

-- Define the plane equation
def plane (x y z : ℝ) : Prop := 2 * x - y + 2 * z = 7

-- Define the specific point
def point_A : Prop := plane 2 3 3

-- The theorem stating that point A lies on the plane
theorem point_A_lies_on_plane : point_A :=
by
  -- Proof skipped
  sorry

end point_A_lies_on_plane_l795_795337


namespace sum_of_third_fifth_l795_795666

noncomputable def sequence (n : ℕ) : ℕ :=
if n = 1 then 1
else (n * (n + 1) - (n - 1) * n)

theorem sum_of_third_fifth :
  let a := sequence in
  a 3 + a 5 = 16 :=
by 
  sorry

end sum_of_third_fifth_l795_795666


namespace pyramid_lateral_edge_length_l795_795734

/-- Define the properties of the pyramid -/
def pyramid_volume (h PA s : ℝ) := (1 / 3) * (s ^ 2) * h

/-- Define the conditions given in the problem -/
def given_conditions := 
  let volume := 4 / 3
  let side_length := 2
  side_length = 2 ∧ pyramid_volume 1 (sqrt 3) side_length = volume

/-- Define the problem in Lean -/
theorem pyramid_lateral_edge_length {PA : ℝ} (h PA s : ℝ) 
  (h_volume : pyramid_volume h PA s = 4 / 3)
  (h_side_length : s = 2) : PA = sqrt 3 := 
by
  sorry

end pyramid_lateral_edge_length_l795_795734


namespace dartboard_odd_sum_probability_l795_795089

theorem dartboard_odd_sum_probability :
  let innerR := 4
  let outerR := 8
  let inner_points := [3, 1, 1]
  let outer_points := [2, 3, 3]
  let total_area := π * outerR^2
  let inner_area := π * innerR^2
  let outer_area := total_area - inner_area
  let each_inner_area := inner_area / 3
  let each_outer_area := outer_area / 3
  let odd_area := 2 * each_inner_area + 2 * each_outer_area
  let even_area := each_inner_area + each_outer_area
  let P_odd := odd_area / total_area
  let P_even := even_area / total_area
  let odd_sum_prob := 2 * (P_odd * P_even)
  odd_sum_prob = 4 / 9 := by
    sorry

end dartboard_odd_sum_probability_l795_795089


namespace thabo_HNF_calculation_l795_795520

variable (THABO_BOOKS : ℕ)

-- Conditions as definitions
def total_books : ℕ := 500
def fiction_books : ℕ := total_books * 40 / 100
def non_fiction_books : ℕ := total_books * 60 / 100
def paperback_non_fiction_books (HNF : ℕ) : ℕ := HNF + 50
def total_non_fiction_books (HNF : ℕ) : ℕ := HNF + paperback_non_fiction_books HNF

-- Lean statement to prove
theorem thabo_HNF_calculation (HNF : ℕ) :
  total_books = 500 →
  fiction_books = 200 →
  non_fiction_books = 300 →
  total_non_fiction_books HNF = 300 →
  2 * HNF + 50 = 300 →
  HNF = 125 :=
by
  intros _
         _
         _
         _
         _
  sorry

end thabo_HNF_calculation_l795_795520


namespace probability_positive_slopes_l795_795825

-- Define the points as independent and uniformly chosen within the unit square
structure Point :=
  (x y : ℝ)
  (prop : 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1)

def chosen_points : list Point := [A, B, C, D]

-- Condition for positive slopes between all pairs
def positive_slope (p1 p2 : Point) : Prop :=
  p1.x < p2.x ∧ p1.y < p2.y

-- The main theorem statement
theorem probability_positive_slopes :
  ∀ (A B C D : Point),
    (independent [A, B, C, D] ∧ uniform [A.x, B.x, C.x, D.x] ∧ uniform [A.y, B.y, C.y, D.y]) →
    (probability (positive_slope A B ∧ positive_slope A C ∧ positive_slope A D ∧ 
                  positive_slope B C ∧ positive_slope B D ∧ positive_slope C D)) = (1 / 24) :=
by sorry

end probability_positive_slopes_l795_795825


namespace natural_number_equals_eleven_times_sum_of_digits_l795_795605

def sum_of_digits (n : ℕ) : ℕ :=
  (toDigits 10 n).foldr (· + ·) 0

theorem natural_number_equals_eleven_times_sum_of_digits (n : ℕ) :
  n = 11 * sum_of_digits n → n = 0 ∨ n = 198 :=
by
  sorry

end natural_number_equals_eleven_times_sum_of_digits_l795_795605


namespace intersection_eq_l795_795325

def M := {x : ℝ | x < 1}
def N := {x : ℝ | Real.log x / Real.log 2 < 1}

theorem intersection_eq : {x : ℝ | x ∈ M ∧ x ∈ N} = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_eq_l795_795325


namespace primes_fulfilling_property_l795_795278
open Int

def is_squarefree (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m*m ∣ n → m = 1

def satisfies_condition (p : ℕ) : Prop :=
  (prime p) ∧ (p ≥ 3 ∧ ∀ q : ℕ, prime q ∧ q < p → is_squarefree (p - (p / q) * q))

theorem primes_fulfilling_property :
  ∃ (ps : List ℕ), ps = [5, 7, 13] ∧ ∀ p : ℕ, satisfies_condition p ↔ p ∈ ps := by
  sorry

end primes_fulfilling_property_l795_795278


namespace part1_part2_l795_795371

def f (k : ℝ) (x : ℝ) := k * x + log (3 ^ x + 1)
def g (x : ℝ) := f (-1 / 2) x - 1 / 2 * x - log (3 ^ x - 1)

theorem part1 (h_even: ∀ x : ℝ, f k (-x) = f k x): k = -1 / 2 :=
sorry

theorem part2 : ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, g y = 0 → y = x :=
sorry

end part1_part2_l795_795371


namespace part1_proof_l795_795859

def a : ℚ := 1 / 2
def b : ℚ := -2
def expr : ℚ := 2 * (3 * a^2 * b - a * b^2) - 3 * (2 * a^2 * b - a * b^2 + a * b)

theorem part1_proof : expr = 5 := by
  unfold expr
  unfold a
  unfold b
  sorry

end part1_proof_l795_795859


namespace stickers_lost_correct_l795_795455

-- Define the initial conditions
def initial_stickers : ℕ := 93
def final_stickers : ℕ := 87

-- The statement to prove
theorem stickers_lost_correct :
  let stickers_lost := initial_stickers - final_stickers
  in stickers_lost = 6 := by
  sorry

end stickers_lost_correct_l795_795455


namespace circle_delta_values_square_delta_values_triangle_delta_values_l795_795072

-- Condition Definitions:
def K_p : Type := ℝ -- The unit circle
def K_θ : Type := ℝ -- The unit square
def T_p : Type := ℝ -- The unit equilateral triangle

-- Function definitions, representing each problem condition
noncomputable def delta (n : ℕ) (shape : Type) : ℝ := sorry

-- Statements to prove each corresponding to correct answers.

-- Circle
theorem circle_delta_values :
  (delta 1 K_p = 1) ∧
  (delta 2 K_p = 1) ∧
  (delta 3 K_p = (real.sqrt 3) / 2) ∧
  (delta 4 K_p = (real.sqrt 2) / 2) ∧
  (delta 5 K_p = (1 / 4) * real.sqrt (10 - 2 * real.sqrt 5)) ∧
  (delta 6 K_p = 1 / 2) ∧
  (delta 7 K_p = 1 / 2) := sorry

-- Square
theorem square_delta_values :
  (delta 1 K_θ = 1) ∧
  (delta 2 K_θ = real.sqrt 10 / 4) ∧
  (delta 3 K_θ = real.sqrt 130 / 16) ∧
  (delta 4 K_θ = 1 / 2) := sorry

-- Equilateral Triangle
theorem triangle_delta_values :
  (delta 1 T_p = 1) ∧
  (delta 2 T_p = 1) ∧
  (delta 3 T_p = real.sqrt 3 / 3) ∧
  (delta 4 T_p = 1 / 2) ∧
  (delta 5 T_p = 1 / 2) := sorry

end circle_delta_values_square_delta_values_triangle_delta_values_l795_795072


namespace part_a_part_b_l795_795199

-- For part (a)
theorem part_a (S : Finset ℤ) (h₁ : ∀ x ∈ S, ∃ y ∈ S, x = y + 1) (h₂ : 2 < S.card) :
  ∃ (k : ℤ), (k ∈ S) ∧ (∃ (mean : ℤ), (∑ (x : ℤ) in (S.erase k), x).nat_abs % (S.card - 1) = 0) :=
sorry

-- For part (b)
theorem part_b (S : Finset ℕ) (h₁ : S = (finset.range 100).map ⟨Nat.succ, Nat.succ_injective⟩) :
  ∃ (k : ℕ), (k ∈ S) ∧ (k = 1 ∨ k = 100) :=
sorry

end part_a_part_b_l795_795199


namespace area_of_triangle_ABC_l795_795084

-- Axiom statements representing the conditions
axiom medians_perpendicular (A B C D E G : Type) : Prop
axiom median_ad_length (A D : Type) : Prop
axiom median_be_length (B E : Type) : Prop

-- Main theorem statement
theorem area_of_triangle_ABC
  (A B C D E G : Type)
  (h1 : medians_perpendicular A B C D E G)
  (h2 : median_ad_length A D) -- AD = 18
  (h3 : median_be_length B E) -- BE = 24
  : ∃ (area : ℝ), area = 576 :=
sorry

end area_of_triangle_ABC_l795_795084


namespace taehyung_mom_age_l795_795521

variables (taehyung_age_diff_mom : ℕ) (taehyung_age_diff_brother : ℕ) (brother_age : ℕ)

theorem taehyung_mom_age 
  (h1 : taehyung_age_diff_mom = 31) 
  (h2 : taehyung_age_diff_brother = 5) 
  (h3 : brother_age = 7) 
  : 43 = brother_age + taehyung_age_diff_brother + taehyung_age_diff_mom := 
by 
  -- Proof goes here
  sorry

end taehyung_mom_age_l795_795521


namespace expenditures_ratio_l795_795140

-- Definitions based on given conditions
def P1_income : ℕ := 3000
def P1_savings : ℕ := 1200
def P2_savings : ℕ := 1200
def income_ratio : ℕ × ℕ := (5, 4)

-- Prove that the ratio of their expenditures is 3 : 2
theorem expenditures_ratio : 
  let I1 := P1_income in
  let S := P1_savings in
  let E1 := I1 - S in
  let I2 := (income_ratio.2 * I1) / income_ratio.1 in
  let E2 := I2 - P2_savings in
  E1 * 2 = E2 * 3 :=
by
  sorry

end expenditures_ratio_l795_795140


namespace helly_half_planes_helly_convex_polygons_l795_795590

-- Helly's theorem for half-planes
theorem helly_half_planes (n : ℕ) (H : Fin n → Set ℝ) 
  (h : ∀ (i j k : Fin n), (H i ∩ H j ∩ H k).Nonempty) : 
  (⋂ i, H i).Nonempty :=
sorry

-- Helly's theorem for convex polygons
theorem helly_convex_polygons (n : ℕ) (P : Fin n → Set ℝ) 
  (h : ∀ (i j k : Fin n), (P i ∩ P j ∩ P k).Nonempty) : 
  (⋂ i, P i).Nonempty :=
sorry

end helly_half_planes_helly_convex_polygons_l795_795590


namespace population_estimate_l795_795678

theorem population_estimate :
  ∃ t : ℕ, 2000 + t * 30 = 2120 ∧ (150 * 3 ^ t ≈ 10000) :=
by
  use 4
  split
  · norm_num
  · apply sorry

end population_estimate_l795_795678


namespace regular_polygon_enclosure_l795_795621

theorem regular_polygon_enclosure (m n : ℕ) (h : m = 12)
    (h_enc : ∀ p : ℝ, p = 360 / ↑n → (2 * (180 / ↑n)) = (360 / ↑m)) :
    n = 12 :=
by
  sorry

end regular_polygon_enclosure_l795_795621


namespace probability_of_stopping_at_edge_l795_795708

-- States and transitions in the grid defined here
structure Position :=
  (x : ℕ) (y : ℕ)
  (condition : x > 0 ∧ x < 5 ∧ y > 0 ∧ y < 5)

def move (pos : Position) (direction : ℕ) : Position :=
  match direction with
  | 0 => {x := pos.x, y := (pos.y + 1 - 1) % 4 + 1, condition := sorry}
  | 1 => {x := pos.x, y := (pos.y - 1 + 3) % 4 + 1, condition := sorry}
  | 2 => {x := (pos.x - 1 + 3) % 4 + 1, y := pos.y, condition := sorry}
  | 3 => {x := (pos.x + 1 - 1) % 4 + 1, y := pos.y, condition := sorry}
  | _ => pos

def is_edge (pos : Position) : Prop :=
  (pos.x = 1) ∨ (pos.x = 4) ∨ (pos.y = 1) ∨ (pos.y = 4)

-- Probability calculations encapsulated
def prob_stop_edge (start : Position) (max_hops : ℕ) : ℚ := sorry

theorem probability_of_stopping_at_edge :
  prob_stop_edge {x:= 2, y:=1, condition := sorry} 5 = 3 / 4 := sorry

end probability_of_stopping_at_edge_l795_795708


namespace find_water_mass_l795_795392

noncomputable def sulfuric_acid_solution (initial_concentration : ℝ) 
    (final_concentration : ℝ) (water_mass : ℝ) : Prop :=
water_mass = 910 ∧ initial_concentration = 130 ∧ final_concentration = 5

-- The statement that we need to prove
theorem find_water_mass : sulfuric_acid_solution 130 5 910 :=
by {
  unfold sulfuric_acid_solution, -- to ensure the statement is equivalent
  sorry
}

end find_water_mass_l795_795392


namespace concert_cost_l795_795506

noncomputable def ticket_price : ℝ := 50.0
noncomputable def processing_fee_rate : ℝ := 0.15
noncomputable def parking_fee : ℝ := 10.0
noncomputable def entrance_fee : ℝ := 5.0
def number_of_people : ℕ := 2

noncomputable def processing_fee_per_ticket : ℝ := processing_fee_rate * ticket_price
noncomputable def total_cost_per_ticket : ℝ := ticket_price + processing_fee_per_ticket
noncomputable def total_ticket_cost : ℝ := number_of_people * total_cost_per_ticket
noncomputable def total_cost_with_parking : ℝ := total_ticket_cost + parking_fee
noncomputable def total_entrance_fee : ℝ := number_of_people * entrance_fee
noncomputable def total_cost : ℝ := total_cost_with_parking + total_entrance_fee

theorem concert_cost : total_cost = 135.0 := by
  sorry

end concert_cost_l795_795506


namespace inverse_proposition_l795_795535

theorem inverse_proposition (a b : ℝ) (h1 : a < 1) (h2 : b < 1) : a + b ≠ 2 :=
by sorry

end inverse_proposition_l795_795535


namespace vector_dot_product_arcsin_symmetry_l795_795771

theorem vector_dot_product_arcsin_symmetry :
  let f : ℝ → ℝ := λ x, Real.arcsin (x - 1)
  let A : ℝ × ℝ := (1, 0)
  let O : ℝ × ℝ := (0, 0)
  let OP : ℝ × ℝ := (2, 0)
  let OA : ℝ × ℝ := (1, 0)
  (OP.1 * OA.1 + OP.2 * OA.2) = 2 :=
by {
  sorry
}

end vector_dot_product_arcsin_symmetry_l795_795771


namespace find_const_functions_l795_795276

theorem find_const_functions
  (f g : ℝ → ℝ)
  (hf : ∀ x y : ℝ, 0 < x → 0 < y → f (x^2 + y^2) = g (x * y)) :
  ∃ c : ℝ, (∀ x, 0 < x → f x = c) ∧ (∀ x, 0 < x → g x = c) :=
sorry

end find_const_functions_l795_795276


namespace shaded_region_area_l795_795439

theorem shaded_region_area :
  let total_grid_area := 13 * 5
  let unshaded_triangle_area := (1 / 2) * 13 * 5
  let shaded_region_area := total_grid_area - unshaded_triangle_area
  shaded_region_area = 32.5 :=
by
  -- Definitions corresponding to conditions
  let total_grid_area := 13 * 5
  let unshaded_triangle_area := (1 / 2) * 13 * 5
  let shaded_region_area := total_grid_area - unshaded_triangle_area
  
  -- State the expected result
  have h : shaded_region_area = 32.5 := sorry

  -- Conclude the theorem
  exact h

end shaded_region_area_l795_795439


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795987

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795987


namespace expected_winnings_is_correct_l795_795203

variable (prob_1 prob_23 prob_456 : ℚ)
variable (win_1 win_23 loss_456 : ℚ)

theorem expected_winnings_is_correct :
  prob_1 = 1/4 → 
  prob_23 = 1/2 → 
  prob_456 = 1/4 → 
  win_1 = 2 → 
  win_23 = 4 → 
  loss_456 = -3 → 
  (prob_1 * win_1 + prob_23 * win_23 + prob_456 * loss_456 = 1.75) :=
by
  intros
  sorry

end expected_winnings_is_correct_l795_795203


namespace solve_system_l795_795515

def eq1 (x y : ℝ) : Prop := x^2 * y - x * y^2 - 5 * x + 5 * y + 3 = 0
def eq2 (x y : ℝ) : Prop := x^3 * y - x * y^3 - 5 * x^2 + 5 * y^2 + 15 = 0

theorem solve_system :
  ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧ x = 4 ∧ y = 1 := by
  sorry

end solve_system_l795_795515


namespace a2023_value_l795_795565

def seq (n : ℕ) : ℚ :=
  if n = 1 then 2 else 1 - (1 / seq (n - 1))

theorem a2023_value : seq 2023 = 2 := 
  sorry

end a2023_value_l795_795565


namespace tangent_line_at_A_l795_795532

-- Define the function f(x)
def f (x : ℝ) : ℝ := x + 1 / x

-- Define the point A(2, 5/2)
def A : ℝ × ℝ := (2, 5 / 2)

-- The assertion that the equation of the tangent line is 3x - 4y + 4 = 0
theorem tangent_line_at_A :
  ∃ (c : ℝ), ∀ x y : ℝ, 3 * x - 4 * y + c = 0 ↔
  (∃ m b : ℝ, y = m * x + b ∧
  m = 1 - 1 / (2 ^ 2) ∧
  b = 5 / 2 - m * 2 ∧
  x = 3 / 4 * (x - 2) + 5 / 2) :=
sorry

end tangent_line_at_A_l795_795532


namespace triangle_ratio_sum_l795_795155

theorem triangle_ratio_sum (A B C P : Type) [tri : Triangle ABC] (h_right : is_right_triangle tri C)
  (h_angle : ∠BAC < 45) (h_AB : d A B = 4) (h_P_on_AB : on_line_seg AB P) (h_angle_condition : ∠APC = 2 * ∠ACP)
  (h_CP : d C P = 1) : 
  (let ratio := (d A P) / (d B P) in
   ∃ p q r : ℕ, ratio = p + q * real.sqrt r ∧ p + q + r = 7) :=
by
  sorry

end triangle_ratio_sum_l795_795155


namespace find_v_l795_795705

theorem find_v : ∀ (v : ℝ), (∃ x : ℝ, 5 * x^2 + 21 * x + v = 0 ∧ x = (-21 - √301) / 10) → v = 7 :=
by
  intro v h
  sorry

end find_v_l795_795705


namespace simplify_correct_l795_795509

def simplify_expression (a b : ℤ) : ℤ :=
  (30 * a + 70 * b) + (15 * a + 45 * b) - (12 * a + 60 * b)

theorem simplify_correct (a b : ℤ) : simplify_expression a b = 33 * a + 55 * b :=
by 
  sorry -- Proof to be filled in later

end simplify_correct_l795_795509


namespace quadratic_sum_a_b_l795_795939

theorem quadratic_sum_a_b (a b : ℝ) (h₁ : ∀ x, a * x^2 + b * x + 2 > 0 → x ∉ set.Ioo (-1/2) (1/3)) 
                           (h₂ : a < 0)
                           (h₃ : - b / a = -1/2 + 1/3)
                           (h₄ : 2 / a = - (1/2) * (1/3)) :
  a + b = -14 :=
begin
  sorry
end

end quadratic_sum_a_b_l795_795939


namespace intersection_point_of_line_and_plane_l795_795699

/-- The coordinates of the intersection point of the line and the plane are (4, -3, 8) -/
theorem intersection_point_of_line_and_plane :
  let x := 4, y := -3, z := 8 in
  (∃ t : ℝ, 
    x = 3 + t ∧ 
    y = -2 - t ∧ 
    z = 8) ∧ 
  5 * x + 9 * y + 4 * z - 25 = 0 :=
by 
  sorry

end intersection_point_of_line_and_plane_l795_795699


namespace arrange_books_l795_795391

-- Definitions based on the problem conditions
def math_books : Type := {M1 : String, M2 : String, M3 : String, M4 : String}
def history_books : Type := {H1 : String, H2 : String, H3 : String, H4 : String}

-- Statement of the problem as a Lean theorem
theorem arrange_books :
  let math_block : Type := {M1 : String, other_math_books : List String} in
  let history_block : Type := List String in
  let total_arrangements := 2 * factorial 3 * factorial 4 in
  total_arrangements = 288 := sorry

end arrange_books_l795_795391


namespace honey_harvest_this_year_l795_795264

def last_year_harvest : ℕ := 2479
def increase_this_year : ℕ := 6085

theorem honey_harvest_this_year : last_year_harvest + increase_this_year = 8564 :=
by {
  sorry
}

end honey_harvest_this_year_l795_795264


namespace identical_digits_divisible_l795_795614

  theorem identical_digits_divisible (n : ℕ) (hn : n > 0) : 
    ∀ a : ℕ, (10^(3^n - 1) * a / 9) % 3^n = 0 := 
  by
    intros
    sorry
  
end identical_digits_divisible_l795_795614


namespace balance_objects_l795_795556

open Finset

theorem balance_objects (weights : Fin 10 → ℕ)
  (h_pos : ∀ i, 0 < weights i)
  (h_bound : ∀ i, weights i ≤ 10)
  (h_sum : (∑ i, weights i) = 20) :
  ∃ (s : Finset (Fin 10)), (∑ i in s, weights i) = 10 :=
sorry

end balance_objects_l795_795556


namespace value_of_f_4_l795_795737

noncomputable def f : ℕ → ℕ 
| x := if x >= 10 then x - 4 else f (x + 5)

theorem value_of_f_4 : f 4 = 10 :=
by
  sorry

end value_of_f_4_l795_795737


namespace alexey_game_max_score_l795_795229

theorem alexey_game_max_score :
  ∃ x : ℕ, 2017 ≤ x ∧ x ≤ 2117 ∧ 
  (∃ d3, x % 3 = 0) ∧
  (∃ d7, x % 7 = 0) ∧
  (∃ d9, x % 9 = 0) ∧
  (∃ d11, x % 11 = 0) ∧
  ((x % 5 = 0 → 30) ∧ (x % 5 ≠ 0 → 30)) :=
begin
  use 2079,
  split,
  { exact nat.le_refl 2079, },
  split,
  { norm_num, },
  split,
  { use 1, norm_num, },
  split,
  { use 297, norm_num, },
  split,
  { use 231, norm_num, },
  split,
  { use 189, norm_num, },
  split,
  { exact 30, },
  exact 30,
end

end alexey_game_max_score_l795_795229


namespace natural_pairs_prime_l795_795683

theorem natural_pairs_prime (x y : ℕ) (p : ℕ) (hp : Nat.Prime p) (h_eq : p = xy^2 / (x + y))
  : (x, y) = (2, 2) ∨ (x, y) = (6, 2) :=
sorry

end natural_pairs_prime_l795_795683


namespace num_days_no_calls_in_2017_l795_795086

noncomputable def calls (d: ℕ) (period: ℕ) : ℕ :=
  d / period

def inclusion_exclusion 
  (n: ℕ) (a: ℕ) (b: ℕ) (c: ℕ) (ab: ℕ) (bc: ℕ) (ac: ℕ) (abc: ℕ) : ℕ :=
  a + b + c - ab - bc - ac + abc

def num_days_without_calls (total_days: ℕ) (days_with_calls: ℕ) : ℕ :=
  total_days - days_with_calls

theorem num_days_no_calls_in_2017 :
  let total_days := 365
  let g1 := 3
  let g2 := 4
  let g3 := 5
  
  let days_g1 := calls total_days g1
  let days_g2 := calls total_days g2
  let days_g3 := calls total_days g3
  
  let days_g1g2 := calls total_days (g1 * g2)
  let days_g2g3 := calls total_days (g2 * g3)
  let days_g1g3 := calls total_days (g1 * g3)
  let days_g1g2g3 := calls total_days (g1 * g2 * g3)
  
  let days_with_calls := inclusion_exclusion days_g1 days_g2 days_g3 days_g1g2 days_g2g3 days_g1g3 days_g1g2g3
  
  num_days_without_calls total_days days_with_calls = 146 :=
by
  sorry

end num_days_no_calls_in_2017_l795_795086


namespace perimeter_C_l795_795897

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end perimeter_C_l795_795897


namespace slices_per_sandwich_l795_795502

theorem slices_per_sandwich (total_sandwiches : ℕ) (total_slices : ℕ) (h1 : total_sandwiches = 5) (h2 : total_slices = 15) :
  total_slices / total_sandwiches = 3 :=
by sorry

end slices_per_sandwich_l795_795502


namespace vertical_angles_equal_l795_795219

-- Given: Definition for pairs of adjacent angles summing up to 180 degrees
def adjacent_add_to_straight_angle (α β : ℝ) : Prop := 
  α + β = 180

-- Given: Two intersecting lines forming angles
variables (α β γ δ : ℝ)

-- Given: Relationship of adjacent angles being supplementary
axiom adj1 : adjacent_add_to_straight_angle α β
axiom adj2 : adjacent_add_to_straight_angle β γ
axiom adj3 : adjacent_add_to_straight_angle γ δ
axiom adj4 : adjacent_add_to_straight_angle δ α

-- Question: Prove that vertical angles are equal
theorem vertical_angles_equal : α = γ :=
by sorry

end vertical_angles_equal_l795_795219


namespace community_structure_irrelevant_l795_795767

-- Definitions of the provided conditions and statements
def population_density_decreasing (population : Type) [inhabited population] : Prop := sorry

-- The main statement to prove
theorem community_structure_irrelevant (population : Type) [inhabited population] :
  population_density_decreasing population →
  true = (true → "The community structure is too complex" is irrelevant) :=
sorry

end community_structure_irrelevant_l795_795767


namespace angle_ABD_is_30_l795_795438

theorem angle_ABD_is_30 
  (A B C D : Type) [Euclidean_geometry A]
  (AB AC BD BC : ℝ) 
  (h1 : AB = AC) 
  (h2 : BD = BC) 
  (angle_BAC : ℝ) 
  (h3 : angle_BAC = 40) :
  ∃ (angle_ABD : ℝ), angle_ABD = 30 := 
by
  sorry 

end angle_ABD_is_30_l795_795438


namespace solve_fractional_eq_l795_795864

theorem solve_fractional_eq (x: ℝ) (h1: x ≠ -11) (h2: x ≠ -8) (h3: x ≠ -12) (h4: x ≠ -7) :
  (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) → (x = -19 / 2) :=
by
  sorry

end solve_fractional_eq_l795_795864


namespace cricket_team_average_age_l795_795594

open Real

-- Definitions based on the conditions given
def team_size := 11
def captain_age := 27
def wicket_keeper_age := 30
def remaining_players_size := team_size - 2

-- The mathematically equivalent proof problem in Lean statement
theorem cricket_team_average_age :
  ∃ A : ℝ,
    (A - 1) * remaining_players_size = (A * team_size) - (captain_age + wicket_keeper_age) ∧
    A = 24 :=
by
  sorry

end cricket_team_average_age_l795_795594


namespace employed_females_percentage_l795_795049

-- Definitions of the conditions
def employment_rate : ℝ := 0.60
def male_employment_rate : ℝ := 0.15

-- The theorem to prove
theorem employed_females_percentage : employment_rate - male_employment_rate = 0.45 := by
  sorry

end employed_females_percentage_l795_795049


namespace julian_needs_more_legos_l795_795804

-- Definitions based on the conditions
def legos_julian_has := 400
def legos_per_airplane := 240
def number_of_airplanes := 2

-- Calculate the total number of legos required for two airplane models
def total_legos_needed := legos_per_airplane * number_of_airplanes

-- Calculate the number of additional legos Julian needs
def additional_legos_needed := total_legos_needed - legos_julian_has

-- Statement that needs to be proven
theorem julian_needs_more_legos : additional_legos_needed = 80 := by
  sorry

end julian_needs_more_legos_l795_795804


namespace perimeter_C_is_40_l795_795893

noncomputable def perimeter_of_figure_C (x y : ℝ) : ℝ :=
  2 * x + 6 * y

theorem perimeter_C_is_40 (x y : ℝ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  perimeter_of_figure_C x y = 40 :=
by
  -- Define initial conditions
  have eq1 : 3 * x + y = 28, by { rw [mul_assoc, mul_comm 3 x, add_assoc], exact (eq.div h1 2) }
  have eq2 : 2 * x + 3 * y = 28, by { rw [mul_assoc, mul_comm 2 x, add_assoc], exact (eq.div h2 2) }
  -- Assume the solutions are obtained from here
  have sol_x : x = 8, by sorry
  have sol_y : y = 4, by sorry
  -- Calculate the perimeter of figure C
  rw [perimeter_of_figure_C, sol_x, sol_y]
  norm_num
  trivial

-- Test case to ensure the code builds successfully
#eval perimeter_of_figure_C 8 4  -- Expected output: 40

end perimeter_C_is_40_l795_795893


namespace minimum_knights_l795_795093

def T_shirtNumber := Fin 80 -- Represent T-shirt numbers from 1 to 80
def Islander := {i // i < 80} -- Each islander is associated with a T-shirt number

-- Knight and Liar definitions
def is_knight (i : Islander) : Prop := sorry -- Definition to be refined
def is_liar (i : Islander) : Prop := sorry -- Definition to be refined

-- Statements that islanders can make
def statement1 (i : Islander) : Prop :=
  ∃ (cnt : ℕ), cnt >= 5 ∧ ∃ (j : Islander), is_liar j ∧ j.1 > i.1 ∧ sorry

def statement2 (i : Islander) : Prop :=
  ∃ (cnt : ℕ), cnt >= 5 ∧ ∃ (j : Islander), is_liar j ∧ j.1 < i.1 ∧ sorry

-- Problem statement: Proving the minimum number of knights
theorem minimum_knights (kn : Fin 80 → bool) :
  (∀ i, if kn i then is_knight i else is_liar i) →
  (∀ i, is_knight i → (statement1 i ∨ statement2 i)) →
  (∀ i, is_liar i → ¬(statement1 i ∨ statement2 i)) →
  (∃ (k_cnt : ℕ), k_cnt = 70 ∧ sorry) :=
sorry

end minimum_knights_l795_795093


namespace smallest_n_l795_795153

theorem smallest_n (n : ℕ) (x : Fin n → ℝ) :
  (∑ i in Finset.range n, Real.sin (x i) = 0) →
  (∑ i in Finset.range n, (i + 1) * Real.sin (x i) = 100) →
  n = 20 :=
by
  sorry

end smallest_n_l795_795153


namespace question_1_question_2_question_3_l795_795359
-- Importing the Mathlib library for necessary functions

-- Definitions and assumptions based on the problem conditions
def z0 (m : ℝ) : ℂ := 1 - m * Complex.I
def z (x y : ℝ) : ℂ := x + y * Complex.I
def w (x' y' : ℝ) : ℂ := x' + y' * Complex.I

/-- The proof problem in Lean 4 to find necessary values and relationships -/
theorem question_1 (m : ℝ) (hm : m > 0) :
  (Complex.abs (z0 m) = 2 → m = Real.sqrt 3) ∧
  (∀ (x y : ℝ), ∃ (x' y' : ℝ), x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y) :=
by
  sorry

theorem question_2 (x y : ℝ) (hx : y = x + 1) :
  ∃ x' y', x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y ∧ 
  y' = (2 - Real.sqrt 3) * x' - 2 * Real.sqrt 3 + 2 :=
by
  sorry

theorem question_3 (x y : ℝ) :
  (∃ (k b : ℝ), y = k * x + b ∧ 
  (∀ (x y x' y' : ℝ), x' = x + Real.sqrt 3 * y ∧ y' = Real.sqrt 3 * x - y ∧ y' = k * x' + b → 
  y = Real.sqrt 3 / 3 * x ∨ y = - Real.sqrt 3 * x)) :=
by
  sorry

end question_1_question_2_question_3_l795_795359


namespace not_perfect_square_l795_795943

theorem not_perfect_square (N : ℕ) (hN : N = ∑ i in Finset.range 300, 10^i) : ¬ is_square N :=
begin
  sorry
end

end not_perfect_square_l795_795943


namespace average_age_proof_l795_795523

noncomputable def avg_age_of_office : ℝ := 
  let A := (70 + 144 + 56) / 14 in 
  A

theorem average_age_proof :
  let A := avg_age_of_office in
  A = 270 / 14 := 
by
  sorry

end average_age_proof_l795_795523


namespace complex_number_solution_l795_795735

theorem complex_number_solution (a b : ℤ) (z : ℂ) (h1 : z = a + b * Complex.I) (h2 : z^3 = 2 + 11 * Complex.I) : a + b = 3 :=
sorry

end complex_number_solution_l795_795735


namespace multiple_of_Roseville_population_l795_795516

noncomputable def Willowdale_population : ℕ := 2000

noncomputable def Roseville_population : ℕ :=
  (3 * Willowdale_population) - 500

noncomputable def SunCity_population : ℕ := 12000

theorem multiple_of_Roseville_population :
  ∃ m : ℕ, SunCity_population = (m * Roseville_population) + 1000 ∧ m = 2 :=
by
  sorry

end multiple_of_Roseville_population_l795_795516


namespace jeremys_school_distance_l795_795801

def distance_to_school (rush_hour_time : ℚ) (no_traffic_time : ℚ) (speed_increase : ℚ) (distance : ℚ) : Prop :=
  ∃ v : ℚ, distance = v * rush_hour_time ∧ distance = (v + speed_increase) * no_traffic_time

theorem jeremys_school_distance :
  distance_to_school (3/10 : ℚ) (1/5 : ℚ) 20 12 :=
sorry

end jeremys_school_distance_l795_795801


namespace count_of_triples_is_three_l795_795695

noncomputable def count_valid_triples : ℕ := 
  nat.count
    (λ (x y z : ℝ), 
       x = 2020 - 2021 * (sign (y + z)) ∧
       y = 2020 - 2021 * (sign (x + z)) ∧
       z = 2020 - 2021 * (sign (x + y)))
    [(4041, -1, -1), (-1, 4041, -1), (-1, -1, 4041)]

theorem count_of_triples_is_three : count_valid_triples = 3 := sorry

end count_of_triples_is_three_l795_795695


namespace count_triples_xyz_l795_795698

theorem count_triples_xyz :
  let sgn := (sign : ℝ → ℝ)
  let solutions := 
    {xyz : ℝ × ℝ × ℝ |
      let (x, y, z) := xyz in
      x = 2020 - 2021 * sgn (y + z) ∧
      y = 2020 - 2021 * sgn (x + z) ∧
      z = 2020 - 2021 * sgn (x + y)} in
  solutions.to_finset.card = 3 :=
by
  let sgn := (sign : ℝ → ℝ)
  let solutions := 
    {xyz : ℝ × ℝ × ℝ |
      let (x, y, z) := xyz in
      x = 2020 - 2021 * sgn (y + z) ∧
      y = 2020 - 2021 * sgn (x + z) ∧
      z = 2020 - 2021 * sgn (x + y)}
  have solutions_list : list (ℝ × ℝ × ℝ) := [(-1, -1, 4041), (-1, 4041, -1), (4041, -1, -1)]
  have solutions_set := solutions_list.to_finset
  have : solutions_set = solutions.to_finset := sorry
  rw this
  exact finset.card_of_list solutions_list

end count_triples_xyz_l795_795698


namespace Jorge_is_24_years_younger_l795_795059

-- Define the conditions
def Jorge_age_2005 := 16
def Simon_age_2010 := 45

-- Prove that Jorge is 24 years younger than Simon
theorem Jorge_is_24_years_younger :
  (Simon_age_2010 - (Jorge_age_2005 + 5) = 24) :=
by
  sorry

end Jorge_is_24_years_younger_l795_795059


namespace no_such_4_digit_prime_l795_795473

theorem no_such_4_digit_prime (f o g h : ℕ) (FOGH : ℕ) :
  f ≠ o ∧ f ≠ g ∧ f ≠ h ∧ o ≠ g ∧ o ≠ h ∧ g ≠ h ∧ 
  (10^3 * f + 10^2 * o + 10 * g + h = FOGH) ∧ 
  FOGH.prime ∧ 
  (10^3 * f + 10^2 * o + 10 * g + h) * (f * o * g * h) = FOGH →
  false :=
sorry

end no_such_4_digit_prime_l795_795473


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795980

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795980


namespace existence_a_b_l795_795830

theorem existence_a_b (k l c : ℕ) (hk : 0 < k) (hl : 0 < l) (hc : 0 < c) :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ b - a = c * Nat.gcd a b ∧
  (Nat.tau a / Nat.tau (a / Nat.gcd a b) * l = Nat.tau b / Nat.tau (b / Nat.gcd a b) * k) :=
by
  sorry

end existence_a_b_l795_795830


namespace sequence_sum_l795_795832

theorem sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (hn : ∀ n, a n > 0)
  (h : ∀ n, (a n)^2 + 2 * (a n) = 4 * S n + 3) :
  (a 1 = 3 ∧ ∀ n, a (n + 1) = a n + 2) → 
  let b (n : ℕ) := 1 / (a n * a (n + 1)) in 
  (∑ k in finset.range n, b k) = n / (3 * (2 * n + 3)) :=
sorry

end sequence_sum_l795_795832


namespace gcd_n_cube_minus_27_n_minus_3_l795_795703

-- Define a function to compute gcd
def gcd (a b : Nat) : Nat := a.gcd b

-- State the problem as a hypothesis and goal
theorem gcd_n_cube_minus_27_n_minus_3 (n : Nat) (h : n > 9) : gcd (n^3 - 27) (n - 3) = (n - 3) :=
by
  sorry

end gcd_n_cube_minus_27_n_minus_3_l795_795703


namespace find_d_l795_795775

noncomputable def d_length (AB BC AC : ℝ) (d : ℝ) : Prop :=
  AB = 320 ∧ BC = 375 ∧ AC = 425 ∧ 
  (375 - (15 / 17 * d + 25 / 16 * d) = d)

theorem find_d :
  ∃ d : ℝ, d_length 320 375 425 d ∧ d = 108.865 :=
begin
  use 108.865,
  unfold d_length,
  split,
  { norm_num, },
  linarith,
end

end find_d_l795_795775


namespace water_flow_into_sea_per_minute_l795_795623

noncomputable def flow_rate_kmph : ℝ := 8
noncomputable def flow_rate_m_per_minute : ℝ := flow_rate_kmph * 1000 / 60
def river_depth : ℝ := 12
def river_width : ℝ := 35
def cross_sectional_area : ℝ := river_depth * river_width
def volume_flow_per_minute : ℝ := cross_sectional_area * flow_rate_m_per_minute

theorem water_flow_into_sea_per_minute : volume_flow_per_minute = 56000 := sorry

end water_flow_into_sea_per_minute_l795_795623


namespace problem_accuracy_and_significant_figures_l795_795868

noncomputable def place_accuracy : String :=
  if (1.50e5 : Real).precRes = 1.50e3 then "thousand" else "unknown"

def significant_figures (n : Real) : List Nat :=
  [1, 5, 0]

theorem problem_accuracy_and_significant_figures :
  place_accuracy = "thousand" ∧ significant_figures (1.50e5) = [1, 5, 0] ∧ (significant_figures (1.50e5)).length = 3 :=
by
  sorry

end problem_accuracy_and_significant_figures_l795_795868


namespace quadratic_inequality_solution_l795_795707

theorem quadratic_inequality_solution (x : ℝ) : 3 * x^2 - 5 * x - 8 > 0 ↔ x < -4/3 ∨ x > 2 :=
by
  sorry

end quadratic_inequality_solution_l795_795707


namespace monotonicity_and_min_value_a_neg1_range_of_a_l795_795369

-- Define the function f(x)
def f (x a : ℝ) := (x^2 + 2 * x + a) / x

-- Problem statement (1): a = -1, prove monotonicity and minimum value
theorem monotonicity_and_min_value_a_neg1 :
  (∀ x ≥ 1, f x (-1) = x - 1/x + 2) ∧ (∀ x ≥ 1, f' x > 0) ∧ (∀ x ≥ 1, x = 1 → f x (-1) = 2) :=
by
  sorry

-- Problem statement (2): For all x in [1, +∞), f(x) > 0 implies a > -3
theorem range_of_a :
  (∀ x ≥ 1, f x a > 0) → a > -3 :=
by
  sorry

end monotonicity_and_min_value_a_neg1_range_of_a_l795_795369


namespace number_of_possible_values_r_l795_795135

noncomputable def is_closest_approx (r : ℝ) : Prop :=
  (r >= 0.2857) ∧ (r < 0.2858)

theorem number_of_possible_values_r : 
  ∃ n : ℕ, (∀ r : ℝ, is_closest_approx r ↔ r = 0.2857 ∨ r = 0.2858 ∨ r = 0.2859) ∧ n = 3 :=
by
  sorry

end number_of_possible_values_r_l795_795135


namespace gum_total_l795_795837

theorem gum_total (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) : 
  58 + x + y = 58 + x + y :=
by sorry

end gum_total_l795_795837


namespace mean_value_of_quadrilateral_angles_l795_795969

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795969


namespace length_of_cable_l795_795444

-- Conditions
def condition1 (x y z : ℝ) : Prop := x + y + z = 8
def condition2 (x y z : ℝ) : Prop := x * y + y * z + x * z = -18

-- Conclusion we want to prove
theorem length_of_cable (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 x y z) :
  4 * π * Real.sqrt (59 / 3) = 4 * π * (Real.sqrt ((x^2 + y^2 + z^2 - ((x + y + z)^2 - 4*(x*y + y*z + x*z))) / 3)) :=
sorry

end length_of_cable_l795_795444


namespace gcd_factorials_l795_795308

open Nat

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorials (h : ∀ n, 0 < n → factorial n = n * factorial (n - 1)) :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 :=
sorry

end gcd_factorials_l795_795308


namespace polygon_side_count_l795_795405

theorem polygon_side_count (n : ℕ) (h : n - 3 ≤ 5) : n = 8 :=
by {
  sorry
}

end polygon_side_count_l795_795405


namespace complement_of_A_l795_795379

def U : Set ℤ := {-1, 2, 4}
def A : Set ℤ := {-1, 4}

theorem complement_of_A : U \ A = {2} := by
  sorry

end complement_of_A_l795_795379


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795988

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795988


namespace average_rainfall_in_inches_per_hour_in_February_2020_l795_795028

theorem average_rainfall_in_inches_per_hour_in_February_2020 :
  let days_in_February_2020 := 29
  let hours_per_day := 24
  let total_rainfall := 290
  let total_hours := days_in_February_2020 * hours_per_day
  (total_rainfall : ℚ) / (total_hours : ℚ) = 290 / 696 :=
by
  simp [days_in_February_2020, hours_per_day, total_rainfall, total_hours]
  sorry

end average_rainfall_in_inches_per_hour_in_February_2020_l795_795028


namespace odd_positive_int_divides_3pow_n_plus_1_l795_795682

theorem odd_positive_int_divides_3pow_n_plus_1 (n : ℕ) (hn_odd : n % 2 = 1) (hn_pos : n > 0) : 
  n ∣ (3^n + 1) ↔ n = 1 := 
by
  sorry

end odd_positive_int_divides_3pow_n_plus_1_l795_795682


namespace count_valid_pairs_l795_795388

def is_valid_pair (a b : ℕ) : Prop :=
  b ∣ (5 * a - 3) ∧ a ∣ (5 * b - 1)

theorem count_valid_pairs : finset.card {ab : ℕ × ℕ | is_valid_pair ab.1 ab.2} = 18 :=
  sorry

end count_valid_pairs_l795_795388


namespace arithmetic_sequence_and_sum_conditions_l795_795356

open Nat

noncomputable theory

def sequence_a (n : ℕ) : ℕ := 2n - 1
def sequence_b (n : ℕ) : ℕ := (3 * n - 2) / (2 * n - 1)

theorem arithmetic_sequence_and_sum_conditions (a d : ℕ) (h_d_pos : d > 0)
  (h_2_3 : (a + d) * (a + 2 * d) = 15) (h_sum_4 : 4 * a + 6 * d = 16) 
  (b_1 : ℕ) (hb_1_eq_a1 : b_1 = a) 
  (hb_recurrence : ∀ n, b n + 1 - b n = 1 / (a_n * (a_n + 1)) →
  (∀ n, a n = 2n - 1) ∧ ∀ n, b_n = (3n - 2) / (2n - 1)) :=
begin
  sorry,
end

end arithmetic_sequence_and_sum_conditions_l795_795356


namespace tangent_line_eq_l795_795554

theorem tangent_line_eq (f : ℝ → ℝ) (x : ℝ) (h : f x = exp x * sin x) (h0 : x = 0) :
  ∃ m b, (m = 1) ∧ (b = 0) ∧ (∀ y, f y = m * y + b) := by 
  sorry

end tangent_line_eq_l795_795554


namespace T_n_correct_l795_795035

def a_n (n : ℕ) : ℤ := 2 * n - 5

def b_n (n : ℕ) : ℤ := 2^n

def C_n (n : ℕ) : ℤ := |a_n n| * b_n n

def T_n : ℕ → ℤ
| 1     => 6
| 2     => 10
| n     => if n >= 3 then 34 + (2 * n - 7) * 2^(n + 1) else 0  -- safeguard for invalid n

theorem T_n_correct (n : ℕ) (hyp : n ≥ 1) : 
  T_n n = 
  if n = 1 then 6 
  else if n = 2 then 10 
  else if n ≥ 3 then 34 + (2 * n - 7) * 2^(n + 1) 
  else 0 := 
by 
sorry

end T_n_correct_l795_795035


namespace minimal_total_cost_l795_795564

def waterway_length : ℝ := 100
def max_speed : ℝ := 50
def other_costs_per_hour : ℝ := 3240
def speed_at_ten_cost : ℝ := 10
def fuel_cost_at_ten : ℝ := 60
def proportionality_constant : ℝ := 0.06

noncomputable def total_cost (v : ℝ) : ℝ :=
  6 * v^2 + 324000 / v

theorem minimal_total_cost :
  (∃ v : ℝ, 0 < v ∧ v ≤ max_speed ∧ total_cost v = 16200) ∧ 
  (∀ v : ℝ, 0 < v ∧ v ≤ max_speed → total_cost v ≥ 16200) :=
sorry

end minimal_total_cost_l795_795564


namespace sum_of_integers_eq_28_24_23_l795_795548

theorem sum_of_integers_eq_28_24_23 
  (a b : ℕ) 
  (h1 : a * b + a + b = 143)
  (h2 : Nat.gcd a b = 1)
  (h3 : a < 30)
  (h4 : b < 30) 
  : a + b = 28 ∨ a + b = 24 ∨ a + b = 23 :=
by
  sorry

end sum_of_integers_eq_28_24_23_l795_795548


namespace range_of_x_l795_795016

theorem range_of_x (x : ℝ) : (x ≠ 2) ↔ ∃ y : ℝ, y = 5 / (x - 2) :=
begin
  sorry
end

end range_of_x_l795_795016


namespace minimum_knights_l795_795092

def T_shirtNumber := Fin 80 -- Represent T-shirt numbers from 1 to 80
def Islander := {i // i < 80} -- Each islander is associated with a T-shirt number

-- Knight and Liar definitions
def is_knight (i : Islander) : Prop := sorry -- Definition to be refined
def is_liar (i : Islander) : Prop := sorry -- Definition to be refined

-- Statements that islanders can make
def statement1 (i : Islander) : Prop :=
  ∃ (cnt : ℕ), cnt >= 5 ∧ ∃ (j : Islander), is_liar j ∧ j.1 > i.1 ∧ sorry

def statement2 (i : Islander) : Prop :=
  ∃ (cnt : ℕ), cnt >= 5 ∧ ∃ (j : Islander), is_liar j ∧ j.1 < i.1 ∧ sorry

-- Problem statement: Proving the minimum number of knights
theorem minimum_knights (kn : Fin 80 → bool) :
  (∀ i, if kn i then is_knight i else is_liar i) →
  (∀ i, is_knight i → (statement1 i ∨ statement2 i)) →
  (∀ i, is_liar i → ¬(statement1 i ∨ statement2 i)) →
  (∃ (k_cnt : ℕ), k_cnt = 70 ∧ sorry) :=
sorry

end minimum_knights_l795_795092


namespace problem_l795_795336

variable (a : ℕ → ℝ) (n m : ℕ)

-- Condition: non-negative sequence and a_{n+m} ≤ a_n + a_m
axiom condition (n m : ℕ) : a n ≥ 0 ∧ a (n + m) ≤ a n + a m

-- Theorem: for any n ≥ m
theorem problem (h : n ≥ m) : a n ≤ m * a 1 + ((n / m) - 1) * a m :=
sorry

end problem_l795_795336


namespace constant_term_in_expansion_l795_795668

theorem constant_term_in_expansion : 
  let expr := (2 * Real.sqrt x - (1 / Real.sqrt4 x)) ^ 6 in
  ∃ c : ℝ, is_constant_term expr c ∧ c = 60 :=
sorry

end constant_term_in_expansion_l795_795668


namespace correct_option_is_A_l795_795632

theorem correct_option_is_A : 
  let A := (Real.cbrt (-27) = -3)
  let B := (Real.sqrt ((-3)^2) = -3)
  let C := (Real.cbrt 125 = 5)
  let D := (Real.sqrt 25 = 5)
  A ∧ ¬B ∧ ¬C ∧ ¬D :=
by
  let A := (Real.cbrt (-27) = -3)
  let B := (Real.sqrt ((-3)^2) = -3)
  let C := (Real.cbrt 125 = 5)
  let D := (Real.sqrt 25 = 5)
  sorry

end correct_option_is_A_l795_795632


namespace ratio_of_triangle_areas_l795_795619

theorem ratio_of_triangle_areas 
  (a b : ℝ) 
  (p q n m : ℕ) 
  (hpq : p + q = n) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hp : p > 0)
  (hm : m > 0) 
  : let C_area := (a * b) / (8 * p)
    let D_area := (a * b) / (4 * m)
    C_area / D_area = 2 * m / p := 
by
  simp only [C_area, D_area],
  sorry

end ratio_of_triangle_areas_l795_795619


namespace find_c_for_degree_3_l795_795252

theorem find_c_for_degree_3 (c : ℚ) :
  let f : ℚ[X] := 1 - 12 * X + 3 * X^2 - 4 * X^3 + 5 * X^4
  let g : ℚ[X] := 3 - 2 * X + X^2 - 6 * X^3 + 11 * X^4
  (degree (f + c * g) = 3) ↔ c = (-5/11 : ℚ) :=
by sorry

end find_c_for_degree_3_l795_795252


namespace christine_aquafaba_needed_l795_795653

-- Define the number of tablespoons per egg white
def tablespoons_per_egg_white : ℕ := 2

-- Define the number of egg whites per cake
def egg_whites_per_cake : ℕ := 8

-- Define the number of cakes
def number_of_cakes : ℕ := 2

-- Express the total amount of aquafaba needed
def aquafaba_needed : ℕ :=
  tablespoons_per_egg_white * egg_whites_per_cake * number_of_cakes

-- Statement asserting the amount of aquafaba needed is 32
theorem christine_aquafaba_needed : aquafaba_needed = 32 := by
  sorry

end christine_aquafaba_needed_l795_795653


namespace students_with_D_l795_795029

noncomputable def fraction_A := 1/5
noncomputable def fraction_B := 1/4
noncomputable def fraction_C := 1/2
noncomputable def total_students := 500

theorem students_with_D : 
    fraction_A + fraction_B + fraction_C = 19/20 →
    (1 - (fraction_A + fraction_B + fraction_C)) * total_students = 25 := by
    intros h
    have h1 : 1 - (fraction_A + fraction_B + fraction_C) = 1 / 20 := by
        rw h
        norm_num
    have h2 : total_students = 500 := by
        norm_num
    norm_num at h1  
    rw h1
    have h3 : (1 / 20) * 500 = 25 := by
        norm_num
    exact h3

end students_with_D_l795_795029


namespace circles_intersect_l795_795410

-- Definition of the first circle
def circle1 (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Definition of the second circle
def circle2 (x y : ℝ) (r : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 49

-- Statement proving the range of r for which the circles intersect
theorem circles_intersect (r : ℝ) (h : r > 0) : (∃ x y : ℝ, circle1 x y r ∧ circle2 x y r) → (2 ≤ r ∧ r ≤ 12) :=
by
  -- Definition of the distance between centers and conditions for intersection
  sorry

end circles_intersect_l795_795410


namespace find_polynomials_l795_795684

theorem find_polynomials (a b c d : ℝ) (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → f (x + y) > f x + f y) :
  (f = λ x, a * x^3 + b * x^2 + c * x + d) →
  d < 0 ∧ ((a = 0 ∧ b ≥ 0) ∨ (a > 0 ∧ 8 * b^3 > 243 * a^2 * d)) :=
sorry

end find_polynomials_l795_795684


namespace arrangement_count_l795_795100

-- Number of books per language
def arabic_books : Nat := 3
def german_books : Nat := 4
def spanish_books : Nat := 3
def french_books : Nat := 2

-- Number of ways to arrange the books given the conditions
theorem arrangement_count :
  (∏ s in ([3, 2] : list ℕ), (Nat.factorial s)) * (arabic_books + german_books + spanish_books + french_books) := 362880 := 
by
  sorry

end arrangement_count_l795_795100


namespace sin_B_value_triangle_area_l795_795720

noncomputable def cos_C : ℝ := sqrt 3 / 3
noncomputable def a : ℝ := 3
noncomputable def condition : Π (b c : ℝ), (b - a) * (sin_arcsin (b / a) + sin_arcsin (c / a)) = (b - c) * sin_arcsin cos_C

theorem sin_B_value (b c : ℝ) (h : condition b c) : sin_arcsin (b / a) = (3 + sqrt 6) / 6 := 
sorry

theorem triangle_area (b c : ℝ) (h : condition b c) : (1 / 2) * a * c * sin_arcsin (b / a) = (3 * sqrt 2 + 2 * sqrt 3) / 2 :=
sorry

end sin_B_value_triangle_area_l795_795720


namespace cos_alpha_minus_270_l795_795327

open Real

theorem cos_alpha_minus_270 (α : ℝ) : 
  sin (540 * (π / 180) + α) = -4 / 5 → cos (α - 270 * (π / 180)) = -4 / 5 :=
by
  sorry

end cos_alpha_minus_270_l795_795327


namespace triangle_solutions_l795_795797

-- Definitions
def a : ℝ := 20
def b : ℝ := 28
def A : ℝ := 40 * Real.pi / 180  -- converting degrees to radians

-- Assertion (no proof)
theorem triangle_solutions : 
    (b * Real.sin A < a) ∧ (a < b) → 
    ∃ solutions, solutions = 2 := sorry

end triangle_solutions_l795_795797


namespace find_m_l795_795768

theorem find_m (m : ℕ) (h1 : 0 ≤ m ∧ m ≤ 9) (h2 : (8 + 4 + 5 + 9) - (6 + m + 3 + 7) % 11 = 0) : m = 9 :=
by
  sorry

end find_m_l795_795768


namespace max_earnings_mary_l795_795082

def wage_rate : ℝ := 8
def first_hours : ℕ := 20
def max_hours : ℕ := 80
def regular_tip_rate : ℝ := 2
def overtime_rate_increase : ℝ := 1.25
def overtime_tip_rate : ℝ := 3
def overtime_bonus_threshold : ℕ := 5
def overtime_bonus_amount : ℝ := 20

noncomputable def total_earnings (hours : ℕ) : ℝ :=
  let regular_hours := min hours first_hours
  let overtime_hours := if hours > first_hours then hours - first_hours else 0
  let overtime_blocks := overtime_hours / overtime_bonus_threshold
  let regular_earnings := regular_hours * (wage_rate + regular_tip_rate)
  let overtime_earnings := overtime_hours * (wage_rate * overtime_rate_increase + overtime_tip_rate)
  let bonuses := (overtime_blocks) * overtime_bonus_amount
  regular_earnings + overtime_earnings + bonuses

theorem max_earnings_mary : total_earnings max_hours = 1220 := by
  sorry

end max_earnings_mary_l795_795082


namespace markup_percentage_l795_795207

theorem markup_percentage (S M : ℝ) (h1 : S = 56 + M * S) (h2 : 0.80 * S - 56 = 8) : M = 0.30 :=
sorry

end markup_percentage_l795_795207


namespace inscribed_square_area_l795_795040

theorem inscribed_square_area (diameter len width : ℝ) (s : ℝ) 
  (hdiam : diameter = 2 * 14)
  (hlen : len = 12) 
  (hwidth : width = 28)
  (inscribed_rectangle : len * width ≤ diameter * diameter)
  (inscribed_square : s * s + (s + 14) * (s + 14) = 14 * 14) :
  s * s = 16 :=
by
  have radius_of_circle : radius = diameter / 2, from 
    sorry
  have radius_eq_14 : radius = 14, from 
    sorry
  have inscribed_dims: (hlen * hwidth) = 336, from 
    sorry
  have eq_rad_sq: 14 * 14 = 196, from
    sorry
  have area_square: s * s = 16, from 
    sorry
  exact area_square

end inscribed_square_area_l795_795040


namespace find_f_neg_two_l795_795117

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_equation (x : ℝ) (hx : x ≠ 0) : 3 * f (1 / x) + (2 * f x) / x = x ^ 2

theorem find_f_neg_two : f (-2) = 67 / 20 :=
by
  sorry

end find_f_neg_two_l795_795117


namespace triangle_AB_value_l795_795051

theorem triangle_AB_value 
  (A B C : ℝ)
  (h1 : ∠A = 90)
  (BC : ℝ := 20)
  (h2 : tan C = 3 * cos B)
  (hypot : AB² = BC² - AC²) :
  AB = 40 * sqrt 2 / 3 :=
by
  sorry

end triangle_AB_value_l795_795051


namespace cos_alpha_beta_half_tan_alpha_beta_l795_795712

noncomputable def cos_val : ℂ := -2 * complex.sqrt 7 / 7
noncomputable def sin_val : ℂ := 1 / 2
noncomputable def alpha_set : set ℂ := {a : ℂ | π / 2 < a ∧ a < π}
noncomputable def beta_set : set ℂ := {b : ℂ | 0 < b ∧ b < π / 2}

theorem cos_alpha_beta_half (α β : ℂ) (hα : α ∈ alpha_set) (hβ : β ∈ beta_set)
  (h1 : complex.cos (α - β / 2) = cos_val) (h2 : complex.sin (α / 2 - β) = sin_val) :
  complex.cos ((α + β) / 2) = -complex.sqrt 21 / 14 :=
sorry

theorem tan_alpha_beta (α β : ℂ) (hα : α ∈ alpha_set) (hβ : β ∈ beta_set)
  (h1 : complex.cos (α - β / 2) = cos_val) (h2 : complex.sin (α / 2 - β) = sin_val) :
  complex.tan (α + β) = 5 * complex.sqrt 3 / 11 :=
sorry

end cos_alpha_beta_half_tan_alpha_beta_l795_795712


namespace Tia_time_to_library_correct_l795_795677

def steps_per_minute_Ella := 80
def step_length_Ella := 80  -- in cm
def time_Ella_to_library := 20  -- in minutes

def steps_per_minute_Tia := 120
def step_length_Tia := 70  -- in cm

def distance_Ella := steps_per_minute_Ella * step_length_Ella * time_Ella_to_library
def speed_Tia := steps_per_minute_Tia * step_length_Tia

def time_Tia_to_library := distance_Ella / speed_Tia

theorem Tia_time_to_library_correct :
  time_Tia_to_library = 15.24 := 
by 
  -- Proof steps will go here
  sorry

end Tia_time_to_library_correct_l795_795677


namespace coefficient_x4_in_expansion_l795_795124

open Nat

theorem coefficient_x4_in_expansion (x : ℝ) :
  (coeff (expand (1 + 2 * x) 6) 4) = 240 := 
sorry

end coefficient_x4_in_expansion_l795_795124


namespace max_piece_length_total_pieces_l795_795151

-- Definitions based on the problem's conditions
def length1 : ℕ := 42
def length2 : ℕ := 63
def gcd_length : ℕ := Nat.gcd length1 length2

-- Theorem statements based on the realized correct answers
theorem max_piece_length (h1 : length1 = 42) (h2 : length2 = 63) :
  gcd_length = 21 := by
  sorry

theorem total_pieces (h1 : length1 = 42) (h2 : length2 = 63) :
  (length1 / gcd_length) + (length2 / gcd_length) = 5 := by
  sorry

end max_piece_length_total_pieces_l795_795151


namespace total_balloons_correct_l795_795449

-- Define the number of blue balloons Joan and Melanie have
def Joan_balloons : ℕ := 40
def Melanie_balloons : ℕ := 41

-- Define the total number of blue balloons
def total_balloons : ℕ := Joan_balloons + Melanie_balloons

-- Prove that the total number of blue balloons is 81
theorem total_balloons_correct : total_balloons = 81 := by
  sorry

end total_balloons_correct_l795_795449


namespace sum_mod_17_l795_795701

theorem sum_mod_17 :
  (78 + 79 + 80 + 81 + 82 + 83 + 84 + 85) % 17 = 6 :=
by
  sorry

end sum_mod_17_l795_795701


namespace coefficient_x3y3_in_expansion_of_x_plus_y_4_l795_795871

theorem coefficient_x3y3_in_expansion_of_x_plus_y_4 :
  (binomial 4 3) = 4 := 
by
  sorry

end coefficient_x3y3_in_expansion_of_x_plus_y_4_l795_795871


namespace max_distinct_exquisite_tuples_l795_795812

def is_exquisite_pair {n : ℕ} (a b : Fin n → ℤ) : Prop :=
  |∑ i, a i * b i| ≤ 1

noncomputable def max_exquisite_tuples (n : ℕ) : ℕ :=
  n^2 + n + 1

theorem max_distinct_exquisite_tuples (n : ℕ) (hn : 0 < n) :
  ∃ (S : Finset (Fin n → ℤ)), 
    (∀ (a b ∈ S), is_exquisite_pair a b) ∧ S.card = max_exquisite_tuples n :=
sorry

end max_distinct_exquisite_tuples_l795_795812


namespace perimeter_of_C_l795_795924

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end perimeter_of_C_l795_795924


namespace area_triangle_PBD_l795_795785

variable {A B C D E F G H P : Type} 

-- Define the conditions 
variables [Plane Geometry] [Area Measures]
variable (parallelogram_ABCD : parallelogram A B C D)
variable (EF_parallel_AB : EF \parallel AB)
variable (HG_parallel_AD : HG \parallel AD)
variable (area_AHPE : area (AHP E) = 5)
variable (area_PECG : area (PECG) = 16)

-- Define the theorem stating the required proof
theorem area_triangle_PBD :
  area (PBD) = 5.5 := 
sorry -- proof to be constructed

end area_triangle_PBD_l795_795785


namespace compute_expression_at_8_l795_795656

theorem compute_expression_at_8 :
  (λ x : ℝ, (x^10 - 32 * x^5 + 1024) / (x^5 - 32)) 8 = 32768 := 
by
  sorry

end compute_expression_at_8_l795_795656


namespace permutations_of_five_people_l795_795856

theorem permutations_of_five_people :
  fintype.card (equiv.perm (fin 5)) = 120 := by
  sorry

end permutations_of_five_people_l795_795856


namespace intersection_complement_correct_l795_795751

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 3, 4, 6}
def complement_B : Set ℕ := U \ B
def intersection_set : Set ℕ := A ∩ complement_B

theorem intersection_complement_correct :
  intersection_set = {2, 5} :=
by
  unfold intersection_set
  unfold A B complement_B U
  simp
  sorry

end intersection_complement_correct_l795_795751


namespace value_of_y_l795_795764

theorem value_of_y (x y : ℤ) (h1 : x^2 - 3 * x + 6 = y + 2) (h2 : x = -8) : y = 92 :=
by
  sorry

end value_of_y_l795_795764


namespace probability_edge_within_five_hops_l795_795844

def is_edge_square (n : ℕ) (coord : ℕ × ℕ) : Prop := 
  coord.1 = 1 ∨ coord.1 = n ∨ coord.2 = 1 ∨ coord.2 = n

def is_central_square (coord : ℕ × ℕ) : Prop :=
  (coord = (2, 2)) ∨ (coord = (2, 3)) ∨ (coord = (3, 2)) ∨ (coord = (3, 3))

noncomputable def probability_of_edge_in_n_hops (n : ℕ) : ℚ := sorry

theorem probability_edge_within_five_hops : probability_of_edge_in_n_hops 4 = 7 / 8 :=
sorry

end probability_edge_within_five_hops_l795_795844


namespace find_I_l795_795796

-- Define the enumeration types to represent digits
def Digits := { x : Nat // x < 10 }

noncomputable def unique_values (E I G H T F V R N : Digits) :=
  E ≠ I ∧ E ≠ G ∧ E ≠ H ∧ E ≠ T ∧ E ≠ F ∧ E ≠ V ∧ E ≠ R ∧ E ≠ N ∧
  I ≠ G ∧ I ≠ H ∧ I ≠ T ∧ I ≠ F ∧ I ≠ V ∧ I ≠ R ∧ I ≠ N ∧
  G ≠ H ∧ G ≠ T ∧ G ≠ F ∧ G ≠ V ∧ G ≠ R ∧ G ≠ N ∧
  H ≠ T ∧ H ≠ F ∧ H ≠ V ∧ H ≠ R ∧ H ≠ N ∧
  T ≠ F ∧ T ≠ V ∧ T ≠ R ∧ T ≠ N ∧
  F ≠ V ∧ F ≠ R ∧ F ≠ N ∧
  V ≠ R ∧ V ≠ N ∧
  R ≠ N

noncomputable def is_valid_addition (E I G H T F V R N : Digits) :=
  10000 * E + 1000 * I + 100 * G + 10 * H + T +
  1000 * F + 100 * I + 10 * V + E =
  10000000 * T + 1000000 * H + 100000 * I + 10000 * R + 1000 * T + 100 * E + 10 * E + N 

theorem find_I (E I G H T F V R N : Digits) (h1 : unique_values E I G H T F V R N)
  (h2 : E = 9) (h3 : G = 1 ∨ G = 3 ∨ G = 5 ∨ G = 7 ∨ G = 9)
  (h4 : is_valid_addition E I G H T F V R N) : I = 4 := 
sorry

end find_I_l795_795796


namespace hyperbola_eccentricity_proof_l795_795373

noncomputable def hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : ℝ :=
  let c := real.sqrt (a^2 + b^2)
  let e := c / a
  e

theorem hyperbola_eccentricity_proof (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1)
  (h₄ : ∃ (A B : ℝ × ℝ), A.1 + B.1 = 2 * a^2 / (b^2 - a^2) ∧ A.2 + B.2 = 2 * b^2 / (b^2 - a^2)
  ∧ collinear (A.1 + B.1, A.2 + B.2) (-3, -1) ) :
  hyperbola_eccentricity a b h₁ h₂ = 2 * real.sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_proof_l795_795373


namespace mean_value_of_quadrilateral_angles_l795_795977

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795977


namespace solution_set_a_eq_2_solution_set_a_eq_neg_2_solution_set_neg_2_lt_a_lt_2_solution_set_a_lt_neg_2_or_a_gt_2_l795_795865

def solve_inequality (a x : ℝ) : Prop :=
  a^2 * x - 6 < 4 * x + 3 * a

theorem solution_set_a_eq_2 :
  ∀ x : ℝ, solve_inequality 2 x ↔ true :=
sorry

theorem solution_set_a_eq_neg_2 :
  ∀ x : ℝ, ¬ solve_inequality (-2) x :=
sorry

theorem solution_set_neg_2_lt_a_lt_2 (a : ℝ) (h : -2 < a ∧ a < 2) :
  ∀ x : ℝ, solve_inequality a x ↔ x > 3 / (a - 2) :=
sorry

theorem solution_set_a_lt_neg_2_or_a_gt_2 (a : ℝ) (h : a < -2 ∨ a > 2) :
  ∀ x : ℝ, solve_inequality a x ↔ x < 3 / (a - 2) :=
sorry

end solution_set_a_eq_2_solution_set_a_eq_neg_2_solution_set_neg_2_lt_a_lt_2_solution_set_a_lt_neg_2_or_a_gt_2_l795_795865


namespace christine_needs_32_tablespoons_l795_795651

-- Define the conditions
def tablespoons_per_egg_white : ℕ := 2
def egg_whites_per_cake : ℕ := 8
def number_of_cakes : ℕ := 2

-- Define the calculation for total tablespoons of aquafaba needed
def total_tbs_aquafaba : ℕ :=
  tablespoons_per_egg_white * (egg_whites_per_cake * number_of_cakes)

-- The theorem to prove
theorem christine_needs_32_tablespoons :
  total_tbs_aquafaba = 32 :=
by 
  -- Placeholder for proof, as proof steps are not required
  sorry

end christine_needs_32_tablespoons_l795_795651


namespace holly_blood_pressure_pills_per_day_l795_795384

theorem holly_blood_pressure_pills_per_day (B A : ℕ) (h1 : A = 2 * B) (h2 : 2 * 7 + 7 * B + 7 * A = 77) : B = 3 :=
by
  -- Initial conditions
  have h_insulin : 2 * 7 = 14 := rfl

  -- Substitute A = 2B in the equation
  have h_subst : 2 * 7 + 7 * B + 7 * (2 * B) = 77,
  { rw h1, exact h2 },

  -- Simplify and solve for B
  simp at h_subst,
  exact sorry

end holly_blood_pressure_pills_per_day_l795_795384


namespace minerva_population_total_l795_795793

variables (h c : ℕ)

def p := 4 * h
def s := 3 * c
def d := 2 * p - 2

theorem minerva_population_total (total : ℕ) :
  total = 13 * h + 4 * c - 2 → total ≠ 223 :=
by {
  intro h1,
  sorry -- Proof omitted
}

end minerva_population_total_l795_795793


namespace geometric_mean_theorem_l795_795129

def geometric_mean_proof : Prop :=
  let a := Real.sqrt 2 + 1
  let b := Real.sqrt 2 - 1
  (let x := Real.sqrt (a * b) in x = 1 ∨ x = -1) ∧
  (let x := -Real.sqrt (a * b) in x = 1 ∨ x = -1)

theorem geometric_mean_theorem : geometric_mean_proof :=
by {
  let a := Real.sqrt 2 + 1,
  let b := Real.sqrt 2 - 1,
  have h : a * b = 1,
  {
    calc a * b = (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) : by rfl
          ... = (Real.sqrt 2)^2 - 1^2 : by { rw Real.mul_self_sqrt, exact le_of_lt (Real.sqrt_pos.2 zero_lt_two) }
          ... = 2 - 1 : by norm_num [Real.mul_self_sqrt]
          ... = 1 : by norm_num,
  },
  split,
  {
    let x := Real.sqrt (a * b),
    show x = 1 ∨ x = -1,
    rw h,
    have : Real.sqrt 1 = 1 := Real.sqrt_eq 1 zero_le_one, -- use Lean's sqrt properties
    rw this,
    exact or.inl rfl,
  },
  {
    let x := -Real.sqrt (a * b),
    show x = 1 ∨ x = -1,
    rw h,
    have : -Real.sqrt 1 = -1 := by {rw Real.sqrt_eq 1 zero_le_one, norm_num},
    rw this,
    exact or.inr rfl,
  },
} .quantity : sorry

end geometric_mean_theorem_l795_795129


namespace algebraic_identity_l795_795015

theorem algebraic_identity (theta : ℝ) (x : ℂ) (n : ℕ) (h1 : 0 < theta) (h2 : theta < π) (h3 : x + x⁻¹ = 2 * Real.cos theta) : 
  x^n + (x⁻¹)^n = 2 * Real.cos (n * theta) :=
by
  sorry

end algebraic_identity_l795_795015


namespace sum_possible_values_g_49_l795_795468

theorem sum_possible_values_g_49 :
  let f := λ x : ℝ, 5 * x^2 - 4
  let g := λ y : ℝ, if h : ∃ x : ℝ, f x = y then
                    (classical.some h)^2 + 2 * (classical.some h) + 3
                  else 0
  g(49) + g(49) = 136 / 5 :=
by
  sorry

end sum_possible_values_g_49_l795_795468


namespace calculate_distance_to_friend_l795_795056

noncomputable def distance_to_friend (d t : ℝ) : Prop :=
  (d = 45 * (t + 1)) ∧ (d = 45 + 65 * (t - 0.75))

theorem calculate_distance_to_friend : ∃ d t: ℝ, distance_to_friend d t ∧ d = 155 :=
by
  exists 155
  exists 2.4375
  sorry

end calculate_distance_to_friend_l795_795056


namespace sum_first_five_terms_geometric_reciprocal_l795_795551

theorem sum_first_five_terms_geometric_reciprocal (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a n = 2^(n-1)) →
  (S n = ∑ i in range n, (1 / a i)) →
  S 5 = 31 / 16 := 
by
  intros ha hs
  -- The proof would go here.
  sorry

end sum_first_five_terms_geometric_reciprocal_l795_795551


namespace cube_volume_increase_l795_795177

theorem cube_volume_increase (s : ℝ) (surface_area : ℝ) 
  (h1 : surface_area = 6 * s^2) (h2 : surface_area = 864) : 
  (1.5 * s)^3 = 5832 :=
by
  sorry

end cube_volume_increase_l795_795177


namespace tommy_profit_l795_795950

noncomputable def total_cost : ℝ := 220 + 375 + 180 + 50 + 30

noncomputable def tomatoes_A : ℝ := 2 * (20 - 4)
noncomputable def oranges_A : ℝ := 2 * (10 - 2)

noncomputable def tomatoes_B : ℝ := 3 * (25 - 5)
noncomputable def oranges_B : ℝ := 3 * (15 - 3)
noncomputable def apples_B : ℝ := 3 * (5 - 1)

noncomputable def tomatoes_C : ℝ := 1 * (30 - 3)
noncomputable def apples_C : ℝ := 1 * (20 - 2)

noncomputable def revenue_A : ℝ := tomatoes_A * 5 + oranges_A * 4
noncomputable def revenue_B : ℝ := tomatoes_B * 6 + oranges_B * 4.5 + apples_B * 3
noncomputable def revenue_C : ℝ := tomatoes_C * 7 + apples_C * 3.5

noncomputable def total_revenue : ℝ := revenue_A + revenue_B + revenue_C

noncomputable def profit : ℝ := total_revenue - total_cost

theorem tommy_profit : profit = 179 :=
by
    sorry

end tommy_profit_l795_795950


namespace nina_running_distance_l795_795487

theorem nina_running_distance (total_distance : ℝ) (initial_run : ℝ) (num_initial_runs : ℕ) :
  total_distance = 0.8333333333333334 →
  initial_run = 0.08333333333333333 →
  num_initial_runs = 2 →
  (total_distance - initial_run * num_initial_runs = 0.6666666666666667) :=
by
  intros h_total h_initial h_num
  sorry

end nina_running_distance_l795_795487


namespace perimeter_C_is_40_l795_795894

noncomputable def perimeter_of_figure_C (x y : ℝ) : ℝ :=
  2 * x + 6 * y

theorem perimeter_C_is_40 (x y : ℝ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  perimeter_of_figure_C x y = 40 :=
by
  -- Define initial conditions
  have eq1 : 3 * x + y = 28, by { rw [mul_assoc, mul_comm 3 x, add_assoc], exact (eq.div h1 2) }
  have eq2 : 2 * x + 3 * y = 28, by { rw [mul_assoc, mul_comm 2 x, add_assoc], exact (eq.div h2 2) }
  -- Assume the solutions are obtained from here
  have sol_x : x = 8, by sorry
  have sol_y : y = 4, by sorry
  -- Calculate the perimeter of figure C
  rw [perimeter_of_figure_C, sol_x, sol_y]
  norm_num
  trivial

-- Test case to ensure the code builds successfully
#eval perimeter_of_figure_C 8 4  -- Expected output: 40

end perimeter_C_is_40_l795_795894


namespace smallest_sector_angle_l795_795081

theorem smallest_sector_angle :
  ∃ (a1 d : ℤ), (15 * (a1 + (a1 + 14 * d)) / 2 = 360) ∧ ∀ x, x = a1 ∨ x = a1 + d ∨ ... ∨ x = a1 + 14 * d ∧ x >= 0 ∧ (∃ y, y = 3 ↔ y = a1)
:=
begin
  sorry
end

end smallest_sector_angle_l795_795081


namespace ball_return_to_A_in_seven_moves_l795_795560

-- Define the initial conditions and recurrence relation
def a : ℕ → ℕ
| 0 => 0
| 1 => 0
| n => 2 * a (n - 1) + 3 * a (n - 2)

theorem ball_return_to_A_in_seven_moves : a 7 = 1094 :=
by
    sorry

end ball_return_to_A_in_seven_moves_l795_795560


namespace count_valid_pairs_l795_795389

theorem count_valid_pairs :
  {n : ℕ // ∃ pairs : Finset (ℕ × ℕ),
    (∀ (a b : ℕ), (a, b) ∈ pairs →
    (b ∣ (5 * a - 3) ∧ a ∣ (5 * b - 1))) ∧ pairs.card = n} =
  ⟨18, _⟩ :=
sorry

end count_valid_pairs_l795_795389


namespace nate_age_when_ember_14_l795_795268

theorem nate_age_when_ember_14 (nate_age : ℕ) (ember_age : ℕ) 
    (h1 : nate_age = 14) (h2 : ember_age = nate_age / 2) 
    (h3 : ember_age = 7) (h4 : ember_14_years_later : ℕ)
    (h : ember_14_years_later = ember_age + (14 - ember_age)) :
  ember_14_years_later = 14 → (nate_age + (14 - ember_age)) = 21 := by
  intros h_ember_14
  sorry

end nate_age_when_ember_14_l795_795268


namespace trains_crossing_time_l795_795954

theorem trains_crossing_time 
  (L : ℕ) (T1 T2 : ℕ) (L = 240) (T1 = 20) (T2 = 30) : 
  ∃ T : ℕ, T = 120 :=
by
  sorry

end trains_crossing_time_l795_795954


namespace zeros_of_geometric_sequence_quadratic_l795_795011

theorem zeros_of_geometric_sequence_quadratic (a b c : ℝ) (h_geometric : b^2 = a * c) (h_pos : a * c > 0) :
  ∀ x : ℝ, a * x^2 + b * x + c ≠ 0 := by
sorry

end zeros_of_geometric_sequence_quadratic_l795_795011


namespace perimeter_C_l795_795919

theorem perimeter_C (x y : ℕ) 
  (h1 : 6 * x + 2 * y = 56)
  (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 := 
by
  sorry

end perimeter_C_l795_795919


namespace selling_price_and_profit_l795_795205

def cost_cycle : ℝ := 1400
def cost_upgrades : ℝ := 600
def profit_percentage : ℝ := 0.10
def sales_tax_percentage : ℝ := 0.05
def total_cost : ℝ := cost_cycle + cost_upgrades
def profit : ℝ := profit_percentage * total_cost
def selling_price_before_tax : ℝ := total_cost + profit
def sales_tax : ℝ := sales_tax_percentage * selling_price_before_tax
def final_selling_price : ℝ := selling_price_before_tax + sales_tax
def overall_profit : ℝ := final_selling_price - total_cost
def overall_profit_percentage : ℝ := (overall_profit / total_cost) * 100

theorem selling_price_and_profit :
  final_selling_price = 2310 ∧ overall_profit_percentage = 15.5 :=
by
  sorry

end selling_price_and_profit_l795_795205


namespace exists_divisible_pair_l795_795820

theorem exists_divisible_pair (n : ℤ) (a : Fin (n + 1) → ℤ) :
  ∃ (i j : Fin (n + 1)), i ≠ j ∧ n ∣ (a i - a j) :=
by
  sorry

end exists_divisible_pair_l795_795820


namespace b_initial_term_b_general_term_l795_795718

noncomputable def a : ℕ → ℕ 
| 0 := 1
| (n + 1) := if n % 2 = 0 then a n + 2 else a n + 1

def b (n : ℕ) : ℕ := a (2 * n)

theorem b_initial_term : b 1 = 2 :=
by {
  -- Placeholder for proof
  sorry
}

theorem b_general_term (n : ℕ) : b n = 3 * n - 1 :=
by {
  -- Placeholder for proof
  sorry
}

end b_initial_term_b_general_term_l795_795718


namespace min_reciprocal_sum_l795_795821

theorem min_reciprocal_sum (a b x y : ℝ) (h1 : 8 * x - y - 4 ≤ 0) (h2 : x + y + 1 ≥ 0) (h3 : y - 4 * x ≤ 0) 
    (ha : a > 0) (hb : b > 0) (hz : a * x + b * y = 2) : 
    1 / a + 1 / b = 9 / 2 := 
    sorry

end min_reciprocal_sum_l795_795821


namespace four_digit_numbers_count_l795_795755

theorem four_digit_numbers_count : 
  let digits := [2, 0, 2, 3] in
  (∃ (n : ℤ), 
    n ≥ 1000 ∧ n < 10000 ∧ 
    ∀ d ∈ [2, 0, 3, 2], ∃ i < 4, n.digits_base 10 !! i = d) →
  (count (λ n, 
            n >= 1000 ∧ 
            n < 10000 ∧ 
            ∀ d ∈ digits, ∃ i < 4, n.digits_base 10 !! i = d) [1000..9999] = 6) :=
begin
  sorry
end

end four_digit_numbers_count_l795_795755


namespace harmonic_set_probability_l795_795400

def is_harmonic_set (A : Set ℝ) : Prop :=
  ∀ x ∈ A, (1 / x) ∈ A

def M : Set ℝ := {-1, 0, 1 / 3, 1 / 2, 1, 2, 3, 4}

theorem harmonic_set_probability :
  let non_empty_subsets := 2^(Set.size M) - 1
  let harmonic_subsets := 15 -- as derived from pairs (2, 1/2), (3, 1/3), {1}, {-1}
  let probability := harmonic_subsets / non_empty_subsets
  probability = 1 / 17 :=
by
  sorry

end harmonic_set_probability_l795_795400


namespace net_gain_loss_is_155_l795_795603

-- Define the cost prices and gain/loss percentages
def cp_A := 1200
def cp_B := 2500
def cp_C := 3300
def cp_D := 4000
def cp_E := 5500

def loss_A := 0.15
def gain_B := 0.10
def loss_C := 0.05
def gain_D := 0.20
def loss_E := 0.10

-- Define the selling prices based on cost prices and gain/loss percentages
def sp_A := cp_A * (1 - loss_A)
def sp_B := cp_B * (1 + gain_B)
def sp_C := cp_C * (1 - loss_C)
def sp_D := cp_D * (1 + gain_D)
def sp_E := cp_E * (1 - loss_E)

-- Define the total cost price and total selling price
def total_cp := cp_A + cp_B + cp_C + cp_D + cp_E
def total_sp := sp_A + sp_B + sp_C + sp_D + sp_E

-- Define the net gain/loss
def net_gain_loss := total_sp - total_cp

-- Prove that the net gain/loss is Rs. 155
theorem net_gain_loss_is_155 : net_gain_loss = 155 := by
  sorry

end net_gain_loss_is_155_l795_795603


namespace prob_ending_even_is_correct_l795_795800

open Classical

-- Define the sample space for card picks and spin results
def card_number := {n : ℕ | 1 ≤ n ∧ n ≤ 12}
def spin_result := {s : ℕ | s = 1 ∨ s = 2 ∨ s = -1 ∨ s = -2}

-- Define the probability distribution for an even starting card
def prob_even_start := 1 / 2

-- Define the probability of ending on an even number after two spins
def prob_even_end (start : ℕ) : ℝ :=
  if start % 2 = 0 then
    (1 / 4) * (1 / 4) + (1 / 4) * (1 / 4) + (1 / 4) * (1 / 4) + (1 / 4) * (1 / 4)
  else
    (1 / 4) * (1 / 4) + (1 / 4) * (1 / 4)

-- The overall probability that Jeff ends up at an even number on the number line
def prob_end_even : ℝ :=
  prob_even_start * (prob_even_end 2) + (1 - prob_even_start) * (prob_even_end 1)

theorem prob_ending_even_is_correct : prob_end_even = 1 / 4 :=
  sorry

end prob_ending_even_is_correct_l795_795800


namespace median_and_mode_correct_l795_795421

noncomputable def data_set : List ℕ := [3, 6, 4, 6, 4, 3, 6, 5, 7]

def median (l : List ℕ) : ℕ :=
  let sorted := l.sorted
  sorted.nthLe (sorted.length / 2) sorry

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ (acc, freq) x =>
    if l.count x > freq then (x, l.count x)
    else acc) (0, 0)

theorem median_and_mode_correct : median data_set = 5 ∧ mode data_set = 6 :=
by
  sorry

end median_and_mode_correct_l795_795421


namespace figure_C_perimeter_l795_795883

def is_perimeter (figure : Type) (perimeter : ℕ) : Prop :=
∃ x y : ℕ, (figure = 'A' → 6*x + 2*y = perimeter) ∧ 
           (figure = 'B' → 4*x + 6*y = perimeter) ∧
           (figure = 'C' → 2*x + 6*y = perimeter)

theorem figure_C_perimeter (hA : is_perimeter 'A' 56) (hB : is_perimeter 'B' 56) : 
  is_perimeter 'C' 40 :=
by
  sorry

end figure_C_perimeter_l795_795883


namespace gcd_fact_8_10_l795_795302

theorem gcd_fact_8_10 : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = 40320 := by
  -- No proof needed
  sorry

end gcd_fact_8_10_l795_795302


namespace tangent_line_equation_at_x_1_intervals_of_monotonic_increase_l795_795742

noncomputable def f (x : ℝ) := x^3 - x + 3
noncomputable def df (x : ℝ) := 3 * x^2 - 1

theorem tangent_line_equation_at_x_1 : 
  let k := df 1
  let y := f 1
  (2 = k) ∧ (y = 3) ∧ ∀ x y, y - 3 = 2 * (x - 1) ↔ 2 * x - y + 1 = 0 := 
by 
  sorry

theorem intervals_of_monotonic_increase : 
  let x1 := - (Real.sqrt 3) / 3
  let x2 := (Real.sqrt 3) / 3
  ∀ x, (df x > 0 ↔ (x < x1) ∨ (x > x2)) ∧ 
       (df x < 0 ↔ (x1 < x ∧ x < x2)) := 
by 
  sorry

end tangent_line_equation_at_x_1_intervals_of_monotonic_increase_l795_795742


namespace perp_AF_to_TU_l795_795545

variables {A B C D E F T U : Point}
variables {k1 k2 k3 k4 : Circle}
variables {e : Line}

/-
  Given:
  1. \( k1 \) and \( k2 \) are circles intersecting at points \( A \) and \( B \).
  2. \( e \) is the common tangent of \( k1 \) and \( k2 \).
  3. \( T \) is the point of tangency of \( k1 \) and \( e \).
  4. \( U \) is the point of tangency of \( k2 \) and \( e \).
  5. A secant line through \( A \), parallel to \( TU \), intersects \( k1 \) at \( C \) and \( k2 \) at \( D \).
  6. \( E \) is a point on the secant line such that \( E \neq A \) and \(\frac{EC}{ED} = \frac{AC}{AD} \).
  7. \( k3 \) is the circle passing through \( A, T, U \).
  8. \( k4 \) is the circle passing through \( A, B, E \).
  9. \( F \) is a point of intersection of \( k3 \) and \( k4 \) that is different from \( A \).
-/

def circles_intersect (k1 k2 : Circle) (A B : Point) : Prop := 
  k1.contains A ∧ k2.contains A ∧ k1.contains B ∧ k2.contains B

def common_tangent (k1 k2 : Circle) (e : Line) (T U : Point) : Prop := 
  tangent k1 e T ∧ tangent k2 e U 

def secant_parallel_to_tangent (A C D : Point) (e : Line) (TU : Line) : Prop := 
  parallel (line_through A C D) TU

def point_E_condition (A C D E : Point) : Prop := 
  E ≠ A ∧ (distance E C / distance E D = distance A C / distance A D)

theorem perp_AF_to_TU 
  (h1 : circles_intersect k1 k2 A B) 
  (h2 : common_tangent k1 k2 e T U)
  (h3 : secant_parallel_to_tangent A C D e (line_through T U))
  (h4 : point_E_condition A C D E)
  (h5 : passes_through_points k3 [A, T, U])
  (h6 : passes_through_points k4 [A, B, E])
  (h7 : F ≠ A ∧ F ∈ k3 ∧ F ∈ k4) :
  perpendicular (line_through A F) (line_through T U) :=
sorry

end perp_AF_to_TU_l795_795545


namespace smallest_unpayable_amount_l795_795555

theorem smallest_unpayable_amount :
  ∀ (coins_1p coins_2p coins_3p coins_4p coins_5p : ℕ), 
    coins_1p = 1 → 
    coins_2p = 2 → 
    coins_3p = 3 → 
    coins_4p = 4 → 
    coins_5p = 5 → 
    ∃ (x : ℕ), x = 56 ∧ 
    ¬ (∃ (a b c d e : ℕ), a * 1 + b * 2 + c * 3 + d * 4 + e * 5 = x ∧ 
    a ≤ coins_1p ∧
    b ≤ coins_2p ∧
    c ≤ coins_3p ∧
    d ≤ coins_4p ∧
    e ≤ coins_5p) :=
by {
  -- Here we skip the actual proof
  sorry
}

end smallest_unpayable_amount_l795_795555


namespace common_difference_l795_795788

theorem common_difference (n : ℕ) : 
  let a := λ n : ℕ, 2 - 3 * n in
  (a (n + 1) - a n) = -3 :=
by
  intros
  rw [sub_eq_add_neg, add_assoc, <- add_assoc]
  have : a (n + 1) = 2 - 3 * (n + 1) := by rfl
  have : a n = 2 - 3 * n := by rfl
  rw [this, this]
  repeat { rw [mul_add, mul_one, sub_add_eq_sub_sub, sub_self, zero_sub, neg_neg, add_left_neg, add_zero] }
  exact rfl

end common_difference_l795_795788


namespace absolute_value_of_difference_of_quadratic_roots_l795_795287
noncomputable theory

open Real

def quadratic_roots (a b c : ℝ) : ℝ × ℝ :=
let discr := b^2 - 4 * a * c in
((−b + sqrt discr) / (2 * a), (−b - sqrt discr) / (2 * a))

theorem absolute_value_of_difference_of_quadratic_roots :
  ∀ r1 r2 : ℝ, 
  r1^2 - 7 * r1 + 12 = 0 → r2^2 - 7 * r2 + 12 = 0 →
  abs (r1 - r2) = 5 :=
by
  sorry

end absolute_value_of_difference_of_quadratic_roots_l795_795287


namespace alexey_game_max_score_l795_795228

theorem alexey_game_max_score :
  ∃ x : ℕ, 2017 ≤ x ∧ x ≤ 2117 ∧ 
  (∃ d3, x % 3 = 0) ∧
  (∃ d7, x % 7 = 0) ∧
  (∃ d9, x % 9 = 0) ∧
  (∃ d11, x % 11 = 0) ∧
  ((x % 5 = 0 → 30) ∧ (x % 5 ≠ 0 → 30)) :=
begin
  use 2079,
  split,
  { exact nat.le_refl 2079, },
  split,
  { norm_num, },
  split,
  { use 1, norm_num, },
  split,
  { use 297, norm_num, },
  split,
  { use 231, norm_num, },
  split,
  { use 189, norm_num, },
  split,
  { exact 30, },
  exact 30,
end

end alexey_game_max_score_l795_795228


namespace cookies_in_bag_l795_795114

theorem cookies_in_bag :
  ∃ (b : ℕ), (8 * 12 = 9 * b + 33) ∧ b = 7 :=
by
  use 7
  split
  { simp }
  sorry

end cookies_in_bag_l795_795114


namespace find_hyperbola_equation_l795_795717

noncomputable def hyperbola_eq (a b : ℝ) (x y : ℝ) : Prop := 
  (x^2) / (a^2) - (y^2) / (b^2) = 1

theorem find_hyperbola_equation (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b)
  (h_asymptote_ratio : b / a = sqrt 3 / 3)
  (h_distance : (|sqrt 3 * a - 3 * 0 - 0| / sqrt (sqrt 3^2 + (-3)^2)) = 1) :
  hyperbola_eq 2 (2 * sqrt 3 / 3) = fun x y => (x^2) / 4 - (y^2 * 3) / 4 := 
sorry

end find_hyperbola_equation_l795_795717


namespace expected_pairs_correct_l795_795251

-- Define the set and permutations
def set_2015 := {1..2015}
def permutation (σ : set_2015 → set_2015) : Prop :=
    ∀ i j, i ≠ j → σ i ≠ σ j

-- Define the condition of pairs
def valid_pairs (σ : set_2015 → set_2015) (i j : set_2015) : Prop :=
    i - j > 155 ∧ σ i - σ j > 266

-- Define the expected number of pairs with the given conditions
noncomputable def expected_num_pairs (σ : set_2015 → set_2015) : ℕ :=
    651222

-- The main theorem to be stated
theorem expected_pairs_correct :
    ∀ σ : set_2015 → set_2015, permutation σ →
    (∑ i in set_2015, ∑ j in set_2015, if valid_pairs σ i j then 1 else 0) / (∑ i in set_2015, ∑ j in set_2015, 1) = 651222 :=
sorry

end expected_pairs_correct_l795_795251


namespace chris_second_day_breath_time_l795_795247

theorem chris_second_day_breath_time :
  ∀ (first_day_time third_day_time : ℕ) (extra_time_per_day : ℕ),
  first_day_time = 10 →
  third_day_time = 30 →
  extra_time_per_day = 10 →
  ∃ (second_day_time : ℕ), second_day_time = first_day_time + extra_time_per_day ∧ second_day_time = 20 :=
by
  intros first_day_time third_day_time extra_time_per_day h_first_day h_third_day h_extra_time
  use first_day_time + extra_time_per_day
  split
  · rw [h_first_day, h_extra_time]
  · rw [h_first_day, h_extra_time]
    norm_num
  sorry

end chris_second_day_breath_time_l795_795247


namespace remaining_individuals_answer_yes_zero_l795_795149

theorem remaining_individuals_answer_yes_zero :
  ∀ (n : ℕ) (knight : ℕ → Prop) (friend : ℕ → ℕ),
  n = 30 →
  (∀ i, knight i ∨ ¬ knight i) →
  (∀ i, knight i ↔ ¬ knight (friend i)) →
  (∀ i, friend (friend i) = i) →
  (∀ i, i < n → (i + 2) % 2 = 0 → ((knight i ∧ (friend i + 1) % n = i + 1) ∨ (¬ knight i ∧ (friend i + 1) % n ≠ i + 1))) →
  (∀ i, ¬(i ≡ 0 [MOD 2]) → (knight i ∧ ((friend i + 1) % n <> i + 1) ∨ (¬ knight i ∧ (friend i + 1) % n = i + 1))) →
  n/2 = countp (λ i, knight i) (finset.range n) ∧
  n/2 = countp (λ i, ¬ knight i) (finset.range n) →
  0 :=
by
  sorry

end remaining_individuals_answer_yes_zero_l795_795149


namespace triangle_similarity_l795_795488

theorem triangle_similarity (A B C A1 B1 C1 : Point)
  (h1 : A1 ∈ Line(B, C))
  (h2 : B1 ∈ Line(C, A))
  (h3 : C1 ∈ Line(A, B))
  (h4 : AC1 / C1B = BA1 / A1C)
  (h5 : BA1 / A1C = CB1 / B1A)
  (h6 : ∠A = ∠B1A1C1) :
  similar (Triangle.mk A B C) (Triangle.mk A1 C1 B1) :=
sorry

end triangle_similarity_l795_795488


namespace perimeter_C_correct_l795_795907

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_l795_795907


namespace orthogonal_vectors_x_eq_1_l795_795681

def v : ℝ × ℝ × ℝ := (2, -4, -5)
def u (x : ℝ) : ℝ × ℝ × ℝ := (-3, x, -2)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

theorem orthogonal_vectors_x_eq_1 (x : ℝ) (h : dot_product v (u x) = 0) : x = 1 :=
by
  sorry

end orthogonal_vectors_x_eq_1_l795_795681


namespace hyperbola_eccentricity_is_sqrt5_l795_795374

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_eq : ∀ x, ∀ y, y^2 / a^2 - x^2 / b^2 = 1) (hasymptote : ∀ x, ∀ y, y = 1/2 * x ∨ y = -1/2 * x) : ℝ :=
  let c := Real.sqrt (a^2 + b^2) in
  let e := c / a in
  e

theorem hyperbola_eccentricity_is_sqrt5 (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_eq : ∀ x, ∀ y, y^2 / a^2 - x^2 / b^2 = 1) (hasymptote : ∀ x, ∀ y, y = 1/2 * x ∨ y = -1/2 * x)
  (h_ratio : b = 2 * a) : hyperbola_eccentricity a b ha hb h_eq hasymptote = Real.sqrt 5 :=
by
  sorry

end hyperbola_eccentricity_is_sqrt5_l795_795374


namespace problem_remainder_modulo_l795_795518

open set

theorem problem_remainder_modulo :
  (∃ (m n : ℕ), coprime m n ∧
    (let probability_event := ((∫ ℝ in Icc 0 (1 / 2011), (1 - ∑ i in range 2011, (λ x => x) i)) ^ 2012) /
                              ∫ ℝ in Iic 1, (1 - ∑ i in range 2011, (λ x => x) i)) *
                             (1 / 2011) ^ 2011,
     probability_event = (m : ℝ) / n) ∧
    (m + n) % 1000 = 1) :=
sorry

end problem_remainder_modulo_l795_795518


namespace sequence_general_term_l795_795783

noncomputable def sequence_condition (a : ℕ → ℝ) : Prop :=
  (∀ n, 2 * a n = 3 * a (n + 1)) ∧ 
  (a 2 * a 5 = 8 / 27) ∧ 
  (∀ n, 0 < a n)

theorem sequence_general_term (a : ℕ → ℝ) (h : sequence_condition a) : 
  ∀ n, a n = (2 / 3)^(n - 2) :=
by 
  sorry

end sequence_general_term_l795_795783


namespace fraction_of_number_is_three_fourths_l795_795552

theorem fraction_of_number_is_three_fourths :
  ∃ (x : ℚ), x * 8 + 2 = 8 ∧ x = 3 / 4 :=
by
  exists 3 / 4
  split
  sorry

end fraction_of_number_is_three_fourths_l795_795552


namespace time_for_one_mile_l795_795215

theorem time_for_one_mile (d v : ℝ) (mile_in_feet : ℝ) (num_circles : ℕ) 
  (circle_circumference : ℝ) (distance_in_miles : ℝ) (time : ℝ) :
  d = 50 ∧ v = 10 ∧ mile_in_feet = 5280 ∧ num_circles = 106 ∧ 
  circle_circumference = 50 * Real.pi ∧ 
  distance_in_miles = (106 * 50 * Real.pi) / 5280 ∧ 
  time = distance_in_miles / v →
  time = Real.pi / 10 :=
by {
  sorry
}

end time_for_one_mile_l795_795215


namespace simplify_to_fraction_l795_795861

noncomputable def simplify_expression (a : ℕ) : ℚ :=
    1 - (a - 1)/(a + 2) / ((a^2 - 1)/(a^2 + 2 * a))

theorem simplify_to_fraction (a : ℕ) (h : a ≠ -2 ∧ a ≠ -1 ∧ a ≠ 1):
    simplify_expression a = 1/(a+1) :=
by
    sorry

end simplify_to_fraction_l795_795861


namespace cos_half_sum_tan_sum_l795_795710

variables {α β : ℝ}

theorem cos_half_sum (h1 : cos (α - β/2) = -2 * sqrt 7 / 7)
                     (h2 : sin (α/2 - β) = 1/2)
                     (hα : π / 2 < α ∧ α < π)
                     (hβ : 0 < β ∧ β < π / 2) :
  cos ((α + β) / 2) = -sqrt 21 / 14 := 
sorry

theorem tan_sum (h1 : cos (α - β/2) = -2 * sqrt 7 / 7)
                (h2 : sin (α/2 - β) = 1/2)
                (hα : π / 2 < α ∧ α < π)
                (hβ : 0 < β ∧ β < π / 2) :
  tan (α + β) = 5 * sqrt 3 / 11 :=
sorry

end cos_half_sum_tan_sum_l795_795710


namespace gcd_factorial_8_10_l795_795310

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_10 : Nat.gcd (factorial 8) (factorial 10) = 40320 :=
by
  -- these pre-evaluations help Lean understand the factorial values
  have fact_8 : factorial 8 = 40320 := by sorry
  have fact_10 : factorial 10 = 3628800 := by sorry
  rw [fact_8, fact_10]
  -- the actual proof gets skipped here
  sorry

end gcd_factorial_8_10_l795_795310


namespace course_distance_l795_795156

noncomputable def team_r_time : ℕ := 15
def team_r_speed : ℕ := 20
def team_a_time := team_r_time - 3
def team_a_speed := team_r_speed + 5
def team_r_distance := team_r_speed * team_r_time
def team_a_distance := team_a_speed * team_a_time

theorem course_distance : team_r_distance = 300 := by
  have h1 : team_r_distance = team_a_distance := by
    calc
      team_r_distance = team_r_speed * team_r_time : by rfl
      ... = 20 * 15 : by rfl
      ... = 300 : by rfl
      ... = 25 * (team_r_time - 3) : by
        calc
          25 * (team_r_time - 3) = 25 * (15 - 3) : by rfl
                            ... = 25 * 12 : by rfl
                            ... = 300 : by rfl
  exact h1

end course_distance_l795_795156


namespace maximize_bjs_l795_795041

def distinct_digits (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

theorem maximize_bjs (爱国 创新 包容 厚德 北京精神 : ℕ) (h1: 爱国 ♯.∈ (0 : ℕ)..10) (h2: 创新 ♯.∈ (0 : ℕ)..10) (h3: 包容 ♯.∈ (0 : ℕ)..10) 
  (h4: 厚德 ♯.∈ (0 : ℕ)..10) (h5: 北京精神 ♯.∈ (1000 : ℕ)..10000) 
  (h_distinct: distinct_digits 爱国 创新 包容 厚德 北京精神)
  (h_max: 北京精神 = 9898) 
  (h_eq: 爱国 * 创新 * 包容 + 厚德 = 北京精神) : 厚德 = 98 := 
sorry

end maximize_bjs_l795_795041


namespace angle_in_equilateral_triangle_in_hexagon_is_60_degrees_l795_795127

-- Define the properties of the regular hexagon and equilateral triangle
def equilateral_triangle_angle_degrees : Prop :=
  ∀ (A B C : Type), (∃ (AB BC CA : ℝ), AB = BC ∧ BC = CA ∧
  ∀ (α β γ : ℝ), α = β ∧ β = γ ∧ α + β + γ = 180 ∧ α = 60)

def regular_hexagon_angle_degrees : Prop :=
  ∀ (A B C D E F : Type), (∃ (AB BC CD DE EF FA : ℝ), AB = BC ∧ BC = CD ∧ CD = DE ∧ DE = EF ∧ EF = FA ∧ FA = AB ∧
  ∀ (α β γ δ ε φ : ℝ), α = β ∧ β = γ ∧ γ = δ ∧ δ = ε ∧ ε = φ ∧ α + β + γ + δ + ε + φ = 720 ∧ α = 120)

-- Define the problem statement
theorem angle_in_equilateral_triangle_in_hexagon_is_60_degrees :
  equilateral_triangle_angle_degrees →
  regular_hexagon_angle_degrees →
  ∃ (A B C : Type), ∠ABC = 60 :=
by
  intro h_triangle
  intro h_hexagon
  use A B C
  sorry

end angle_in_equilateral_triangle_in_hexagon_is_60_degrees_l795_795127


namespace part1_part2_l795_795338

noncomputable def f (x a : ℝ) := Real.exp (2 * x) + 2 * Real.exp (-x) - a * x

theorem part1 {a : ℝ} (h₁ : a > 0) (h₂ : ∃ x ∈ (0:ℝ, 1:ℝ), ∃ c, c ∈ R ∧ is_min_or_max (deriv (f x a) c)) :
  a ∈ (0:ℝ, 2 * Real.exp (2) - 2 * Real.exp (-1)) :=
sorry

theorem part2 {a : ℝ} {x0 : ℝ} (h₁ : a > 0) (h₂ : differentiable ℝ (λ x, f x a)) 
(h₃ : ∃! x, f x a = 0) : 
  Real.log 2 < x0 ∧ x0 < 1 :=  
sorry

end part1_part2_l795_795338


namespace solution_set_of_inequality_l795_795298

theorem solution_set_of_inequality:
  {x : ℝ | x^2 > x} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end solution_set_of_inequality_l795_795298


namespace number_of_valid_scalene_triangles_l795_795640

noncomputable def count_scalene_triangles : ℕ :=
  finset.card {s : finset ℕ | ∃ (a b c : ℕ), a < b ∧ b < c ∧ a + b + c < 17 ∧ 
    a + b > c ∧ a + c > b ∧ b + c > a ∧ s = {a, b, c}} 

theorem number_of_valid_scalene_triangles :
  count_scalene_triangles = 7 :=
by
  sorry

end number_of_valid_scalene_triangles_l795_795640


namespace restaurant_total_cost_l795_795782

theorem restaurant_total_cost (burger_cost pizza_cost : ℕ)
    (h1 : burger_cost = 9)
    (h2 : pizza_cost = 2 * burger_cost) :
    pizza_cost + 3 * burger_cost = 45 := 
by
  sorry

end restaurant_total_cost_l795_795782


namespace probability_no_adjacent_or_across_duplicate_rolls_l795_795862

theorem probability_no_adjacent_or_across_duplicate_rolls :
  ∀ (A B C D E F : Fin 6), 
  let total_outcomes := 6 ^ 6 in
  let favorable_outcomes := 6 * 5 * 5 * 4 * 5 * 4 in
  (favorable_outcomes.toRat / total_outcomes.toRat) = (125 / 972) := by
  sorry

end probability_no_adjacent_or_across_duplicate_rolls_l795_795862


namespace perimeter_of_C_l795_795926

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end perimeter_of_C_l795_795926


namespace integer_solutions_count_l795_795386

theorem integer_solutions_count :
  (finset.univ.filter (λ x : ℤ, (x-3)^(30-(x*x)) = 1)).card = 2 :=
sorry

end integer_solutions_count_l795_795386


namespace gcd_fact_8_10_l795_795304

theorem gcd_fact_8_10 : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = 40320 := by
  -- No proof needed
  sorry

end gcd_fact_8_10_l795_795304


namespace gcd_factorials_l795_795307

open Nat

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorials (h : ∀ n, 0 < n → factorial n = n * factorial (n - 1)) :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 :=
sorry

end gcd_factorials_l795_795307


namespace sunset_time_correct_l795_795842

-- Definitions based on the problem statement
def length_of_daylight : ℕ × ℕ := (12, 36)  -- hours, minutes
def sunrise_time : ℕ × ℕ := (6, 12)        -- hours, minutes

-- Statement to prove
theorem sunset_time_correct :
  let (day_hours, day_minutes) := length_of_daylight in
  let (sunrise_hours, sunrise_minutes) := sunrise_time in
  let sunset_hours := (sunrise_hours + day_hours) % 24 in
  let sunset_minutes := (sunrise_minutes + day_minutes) % 60 in
  let sunset_extra_hours := (sunrise_minutes + day_minutes) / 60 in
  (sunset_hours + sunset_extra_hours, sunset_minutes) = (18 % 12, 48) :=
by
  sorry

end sunset_time_correct_l795_795842


namespace vector_representation_l795_795077

def a := (10 : ℝ) -- representing 10 km east
def b := (10 * Real.sqrt 3 : ℝ) -- representing 10√3 km north

theorem vector_representation : 
  (∃ (v : ℝ × ℝ), v = (10, 0) ∧ v = (0, 10 * Real.sqrt 3)) ∧ 
  (∃ (u : ℝ × ℝ), u = (v.1 - u.1, v.2 - u.2) ∧ u.1 ^ 2 + u.2 ^ 2 = 400) ∧
  (∃ (theta : ℝ), theta = Real.atan (u.2 / u.1) ∧ theta = - Real.pi / 6) :=
  sorry

end vector_representation_l795_795077


namespace triangle_dot_product_l795_795723

noncomputable def equilateral_triple := {a: ℝ × ℝ // ∃ b c: ℝ × ℝ, 
     b ≠ c ∧ abs (b - a) = abs (c - a) ∧ abs (b - c) = abs (c - a) ∧
     abs (b - c) = 2 ∧ (1: ℝ).is_nonzero ∧ is_midpoint b a c}

def e_point_exists {A B C : ℝ × ℝ} (h : equilateral_triple) [DecidableRel A B C] 
 : ∃ c: ℝ × ℝ, abs (c - B) = 1/3 * abs (A - B) ∧ abs (c - A) = 1/3 * abs (A - C) := sorry

theorem triangle_dot_product {A B C D E : ℝ × ℝ} (h_equilateral : equilateral_triple A B C) 
(m_midpoint : m = midpoint ℝ D B C) (h_point_e: E = e_point h h_equilateral m_midpoint) 
: (E - D) * (C - B) = -4/3 := sorry

end triangle_dot_product_l795_795723


namespace value_of_g_neg2_l795_795069

def g (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem value_of_g_neg2 : g (-2) = -3 := 
by sorry

end value_of_g_neg2_l795_795069


namespace triangle_congruence_side_length_l795_795328

theorem triangle_congruence_side_length
  (A B C A₁ B₁ C₁ : Type*)
  [metric_space A] [metric_space B] [metric_space C] [metric_space A₁] [metric_space B₁] [metric_space C₁]
  (h1 : congruent_triangles (triangle.mk A B C) (triangle.mk A₁ B₁ C₁))
  (h2 : dist A B = 4)
  (h3 : dist B C = 5)
  (h4 : dist A C = 6) : 
  dist B₁ C₁ = 5 :=
by {
  sorry
}

end triangle_congruence_side_length_l795_795328


namespace total_days_and_leap_years_2010_to_2015_l795_795642

theorem total_days_and_leap_years_2010_to_2015 :
  let years := [2010, 2011, 2012, 2013, 2014, 2015],
      is_leap_year (y : ℕ) : Prop := (y % 4 = 0) ∧ (y % 100 ≠ 0 ∨ y % 400 = 0),
      leap_years := list.filter is_leap_year years,
      num_leap_years := list.length leap_years,
      total_days := 5 * 365 + num_leap_years * 366
  in total_days = 2191 ∧ num_leap_years = 1 :=
by
  let years := [2010, 2011, 2012, 2013, 2014, 2015],
  let is_leap_year (y : ℕ) : Prop := (y % 4 = 0) ∧ (y % 100 ≠ 0 ∨ y % 400 = 0),
  let leap_years := list.filter is_leap_year years,
  let num_leap_years := list.length leap_years,
  let total_days := 5 * 365 + num_leap_years * 366,
  have h1 : num_leap_years = 1 := by sorry,
  have h2 : total_days = 2191 := by sorry,
  exact ⟨h2, h1⟩

end total_days_and_leap_years_2010_to_2015_l795_795642


namespace joe_bath_shop_bottles_l795_795057

theorem joe_bath_shop_bottles (b : ℕ) (n : ℕ) (m : ℕ) 
    (h1 : 5 * n = b * m)
    (h2 : 5 * n = 95)
    (h3 : b * m = 95)
    (h4 : b ≠ 1)
    (h5 : b ≠ 95): 
    b = 19 := 
by 
    sorry

end joe_bath_shop_bottles_l795_795057


namespace fitness_center_cost_effectiveness_l795_795562

noncomputable def f (x : ℝ) : ℝ := 5 * x

noncomputable def g (x : ℝ) : ℝ :=
  if 15 ≤ x ∧ x ≤ 30 then 90 
  else 2 * x + 30

def cost_comparison (x : ℝ) (h1 : 15 ≤ x) (h2 : x ≤ 40) : Prop :=
  (15 ≤ x ∧ x < 18 → f x < g x) ∧
  (x = 18 → f x = g x) ∧
  (18 < x ∧ x ≤ 40 → f x > g x)

theorem fitness_center_cost_effectiveness (x : ℝ) (h1 : 15 ≤ x) (h2 : x ≤ 40) : cost_comparison x h1 h2 :=
by
  sorry

end fitness_center_cost_effectiveness_l795_795562


namespace nearest_int_to_expr_l795_795167

theorem nearest_int_to_expr : 
  |(3 + Real.sqrt 2)^6 - 3707| < 1 :=
by 
  sorry

end nearest_int_to_expr_l795_795167


namespace caleb_counted_right_angles_l795_795244

-- Definitions for conditions
def rectangular_park_angles : ℕ := 4
def square_field_angles : ℕ := 4
def total_angles (x y : ℕ) : ℕ := x + y

-- Theorem stating the problem
theorem caleb_counted_right_angles (h : total_angles rectangular_park_angles square_field_angles = 8) : 
   "type of anges Caleb counted" = "right angles" :=
sorry

end caleb_counted_right_angles_l795_795244


namespace prove_4_4_draw_l795_795157

def is_valid_4_4_draw_possible : Prop :=
  ∃ board : Matrix (Fin 3) (Fin 3) (Option Bool),
    (∀ {i : Fin 3}, ∃ (count_black : Nat), 
      count_black = (Fin3.sum fun j => if board i j = some true then 1 else 0)) ∧
    (∀ {j : Fin 3}, ∃ (count_black : Nat), 
      count_black = (Fin3.sum fun i => if board i j = some true then 1 else 0)) ∧
    (∃ (count_black : Nat), 
      count_black = (Fin3.sum fun k => if board k k = some true then 1 else 0)) ∧
    (∃ (count_black : Nat), 
      count_black = (Fin3.sum fun k => if board k (Fin 2 - k) = some true then 1 else 0)) ∧
    ((Fin3.sum fun i => ∃ (count_black : Nat), count_black = (Fin3.sum fun j => if board i j = some true then 1 else 0) % 2 = 0) +
    (Fin3.sum fun j => ∃ (count_black : Nat), count_black = (Fin3.sum fun i => if board i j = some true then 1 else 0) % 2 = 0) +
    ((Fin3.sum fun k => if board k k = some true then 1 else 0) % 2 = 0) +
    ((Fin3.sum fun k => if board k (Fin 2 - k) = some true then 1 else 0) % 2 = 0)) = 4)

theorem prove_4_4_draw : is_valid_4_4_draw_possible :=
sorry

end prove_4_4_draw_l795_795157


namespace quadratic_roots_l795_795145

theorem quadratic_roots (x : ℝ) : (x ^ 2 - 3 = 0) → (x = Real.sqrt 3 ∨ x = -Real.sqrt 3) :=
by
  intro h
  sorry

end quadratic_roots_l795_795145


namespace grandson_age_l795_795485

variable (G F : ℕ)

-- Define the conditions given in the problem
def condition1 := F = 6 * G
def condition2 := (F + 4) + (G + 4) = 78

-- The theorem to prove
theorem grandson_age : condition1 G F → condition2 G F → G = 10 :=
by
  intros h1 h2
  sorry

end grandson_age_l795_795485


namespace business_dinner_seating_l795_795152

theorem business_dinner_seating
  (num_people : ℕ)
  (num_executives : ℕ)
  (opposite : ℕ → ℕ)
  (seated_opposite : ∀ (i : ℕ), i < num_executives → opposite i < num_people)
  (distinct_positions : ℕ) :
  num_people = 10 →
  num_executives = 5 →
  distinct_positions = (10 * 8 * 6 * 4 * 2) / 10 →
  (∃ num_arrangements : ℕ, num_arrangements = 384) :=
by
  intros h_people h_executives h_positions;
  have num_arrangements := (10 * 8 * 6 * 4 * 2) / 10;
  use num_arrangements;
  rw [h_positions];
  exact eq.refl 384;
  exact sorry -- Proof omitted

end business_dinner_seating_l795_795152


namespace figure_C_perimeter_l795_795884

def is_perimeter (figure : Type) (perimeter : ℕ) : Prop :=
∃ x y : ℕ, (figure = 'A' → 6*x + 2*y = perimeter) ∧ 
           (figure = 'B' → 4*x + 6*y = perimeter) ∧
           (figure = 'C' → 2*x + 6*y = perimeter)

theorem figure_C_perimeter (hA : is_perimeter 'A' 56) (hB : is_perimeter 'B' 56) : 
  is_perimeter 'C' 40 :=
by
  sorry

end figure_C_perimeter_l795_795884


namespace max_area_rectangle_l795_795543

theorem max_area_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 40) : x * y ≤ 100 :=
by
  sorry

end max_area_rectangle_l795_795543


namespace max_good_set_element_exists_l795_795075

def is_good_set (X : Set ℕ) : Prop :=
  ∀ x y ∈ X, x < y → x ∣ y

theorem max_good_set_element_exists :
  ∃ a ∈ ({1, 2, ..., 2016} : Set ℕ), 
  (∀ X ⊆ ({1, 2, ..., 2016} : Set ℕ), X.card = 1008 → a ∈ X → is_good_set X) ∧ a = 1008 :=
sorry

end max_good_set_element_exists_l795_795075


namespace dihedral_angle_between_adjacent_faces_l795_795529

theorem dihedral_angle_between_adjacent_faces (n : ℕ) (β : ℝ) (h1 : n > 2) (h2 : 0 < β) (h3 : β < π) :
  let γ := 2 * arccos (sin β * sin (π / n)) in
  γ = 2 * arccos (sin β * sin (π / n)) :=
by
  sorry

end dihedral_angle_between_adjacent_faces_l795_795529


namespace perimeter_C_is_40_l795_795891

noncomputable def perimeter_of_figure_C (x y : ℝ) : ℝ :=
  2 * x + 6 * y

theorem perimeter_C_is_40 (x y : ℝ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  perimeter_of_figure_C x y = 40 :=
by
  -- Define initial conditions
  have eq1 : 3 * x + y = 28, by { rw [mul_assoc, mul_comm 3 x, add_assoc], exact (eq.div h1 2) }
  have eq2 : 2 * x + 3 * y = 28, by { rw [mul_assoc, mul_comm 2 x, add_assoc], exact (eq.div h2 2) }
  -- Assume the solutions are obtained from here
  have sol_x : x = 8, by sorry
  have sol_y : y = 4, by sorry
  -- Calculate the perimeter of figure C
  rw [perimeter_of_figure_C, sol_x, sol_y]
  norm_num
  trivial

-- Test case to ensure the code builds successfully
#eval perimeter_of_figure_C 8 4  -- Expected output: 40

end perimeter_C_is_40_l795_795891


namespace half_of_a_correct_l795_795875

axiom a : ℝ

def half_of (x : ℝ) : ℝ := x / 2

def optionA : ℝ := a / 2
def optionB : ℝ := 2 * a
def optionC : ℝ := 2 + a
def optionD : ℝ := a - 2

theorem half_of_a_correct : half_of a = optionA :=
by sorry

end half_of_a_correct_l795_795875


namespace max_area_rectangle_l795_795544

theorem max_area_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 40) : x * y ≤ 100 :=
by
  sorry

end max_area_rectangle_l795_795544


namespace frequency_of_defective_parts_l795_795489

theorem frequency_of_defective_parts : 
  ∀ (n m : ℕ), n = 500 → m = 8 → (m / n : ℝ) = 0.016 :=
by
  intros n m hn hm
  rw [hn, hm]
  norm_num
  sorry

end frequency_of_defective_parts_l795_795489


namespace six_digit_quotient_l795_795750

def six_digit_number (A B : ℕ) : ℕ := 100000 * A + 97860 + B

def divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

theorem six_digit_quotient (A B : ℕ) (hA : A = 5) (hB : B = 1)
  (h9786B : divisible_by_99 (six_digit_number A B)) : 
  six_digit_number A B / 99 = 6039 := by
  sorry

end six_digit_quotient_l795_795750


namespace difference_in_areas_l795_795440

theorem difference_in_areas {AB : ℝ} (h1 : AB = 6) (hpi : Real.pi = 3.14)
  (h2 : ∃ (O : Point) (B : Point) (C : Point) (D : Point) (E : Point) (BCDE : Square), 
    E ∈ Circle O (AB / 2) ∧ angle A B E = 45) :
  let r := AB / 2,
      circle_area := 3.14 * r^2,
      square_side := (AB / 2) * Real.sqrt 2,
      square_area := square_side^2
  in circle_area - square_area = 10.26 :=
by
  let r := AB / 2
  let circle_area := 3.14 * r^2
  let square_side := (AB / 2) * Real.sqrt 2
  let square_area := square_side^2
  have h3 : circle_area = 3.14 * 3^2 := by sorry
  have h4 : square_area = 18 := by sorry
  have hyp : 3.14 * 9 - 18 = 10.26 := by norm_num
  exact hyp

end difference_in_areas_l795_795440


namespace area_of_polygon_l795_795442

-- Definitions based on the conditions of the problem
def num_sides : ℕ := 20
def total_perimeter : ℝ := 56
def sides_perpendicular (a b : ℝ) : Prop := a = b + π / 2
def all_sides_congruent (s : ℝ) (n : ℕ) (p : ℝ) : Prop := n * s = p

-- Proof problem based on the identified question and correct answer
theorem area_of_polygon (s : ℝ) : 
  (num_sides * s = total_perimeter) ∧ (∀ k : ℕ, k < num_sides → sides_perpendicular k (k+1)) ∧
  all_sides_congruent s num_sides total_perimeter →
  let area := 4 * s * s * (3 + 2) in
  area = 157.68 :=
sorry

end area_of_polygon_l795_795442


namespace mike_passing_percentage_l795_795480

theorem mike_passing_percentage (mike_score shortfall max_marks : ℝ)
  (h_mike_score : mike_score = 212)
  (h_shortfall : shortfall = 16)
  (h_max_marks : max_marks = 760) :
  (mike_score + shortfall) / max_marks * 100 = 30 :=
by
  sorry

end mike_passing_percentage_l795_795480


namespace max_rho_squared_l795_795319

variable (ρ θ : Real)

theorem max_rho_squared :
  (3 * ρ * cos(θ)^2 + 2 * ρ * sin(θ)^2 = 6 * cos(θ)) → 
  ρ^2 ≤ 4 :=
by
  sorry

end max_rho_squared_l795_795319


namespace sweets_ratio_l795_795566

theorem sweets_ratio (x : ℕ) (h1 : x + 4 + 7 = 22) : x / 22 = 1 / 2 :=
by
  sorry

end sweets_ratio_l795_795566


namespace find_a_l795_795363

noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x < 1 then 2^x + 1 else x^2 + a * x

theorem find_a (a : ℝ) (h : f a (f a 0) = 4 * a) : a = 2 := by
  sorry

end find_a_l795_795363


namespace count_triples_xyz_l795_795696

theorem count_triples_xyz :
  let sgn := (sign : ℝ → ℝ)
  let solutions := 
    {xyz : ℝ × ℝ × ℝ |
      let (x, y, z) := xyz in
      x = 2020 - 2021 * sgn (y + z) ∧
      y = 2020 - 2021 * sgn (x + z) ∧
      z = 2020 - 2021 * sgn (x + y)} in
  solutions.to_finset.card = 3 :=
by
  let sgn := (sign : ℝ → ℝ)
  let solutions := 
    {xyz : ℝ × ℝ × ℝ |
      let (x, y, z) := xyz in
      x = 2020 - 2021 * sgn (y + z) ∧
      y = 2020 - 2021 * sgn (x + z) ∧
      z = 2020 - 2021 * sgn (x + y)}
  have solutions_list : list (ℝ × ℝ × ℝ) := [(-1, -1, 4041), (-1, 4041, -1), (4041, -1, -1)]
  have solutions_set := solutions_list.to_finset
  have : solutions_set = solutions.to_finset := sorry
  rw this
  exact finset.card_of_list solutions_list

end count_triples_xyz_l795_795696


namespace count_valid_numbers_l795_795000

theorem count_valid_numbers :
  (finset.card {n : ℤ | 4000 ≤ n ∧ n < 5000 ∧ odd (n % 10)}) = 500 :=
by
  sorry

end count_valid_numbers_l795_795000


namespace prism_base_side_length_l795_795618

noncomputable def length_of_base (a b c : ℝ) : ℝ :=
  sqrt (1/3 * (a^2 + b^2 + c^2 - 2 * sqrt (a^4 + b^4 + c^4 - a^2 * b^2 - a^2 * c^2 - b^2 * c^2)))

theorem prism_base_side_length (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  ∃ d : ℝ, d^2 = 1/3 * (a^2 + b^2 + c^2 - 2 * sqrt (a^4 + b^4 + c^4 - a^2 * b^2 - a^2 * c^2 - b^2 * c^2)) :=
begin
  use length_of_base a b c,
  sorry
end

end prism_base_side_length_l795_795618


namespace probability_exactly_three_out_of_four_win_a_prize_l795_795779

-- Define the specific conditions and the final probability to check
def balls := {1, 2, 3, 4, 5, 6}
def draws := (balls × balls)
def winning_pairs := { (1, 4), (3, 4), (2, 4), (2, 6), (4, 5), (4, 6) }
def probability_of_winning := 6 / 15

-- Define binomial probability function
def binomial_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

-- Using the conditions and the equation
theorem probability_exactly_three_out_of_four_win_a_prize :
  binomial_probability 4 3 (2 / 5) = 96 / 625 :=
by sorry

end probability_exactly_three_out_of_four_win_a_prize_l795_795779


namespace length_of_second_train_l795_795574

-- Define the given conditions
def speed_first_train := 75 -- in km/h
def speed_second_train := 65 -- in km/h
def time_to_clear := 7.353697418492236 -- in seconds
def length_first_train := 121 -- in meters

-- Define the expected answer
def length_second_train := 1984 -- in meters

-- Convert speeds from km/h to m/s
def kmh_to_ms (speed : Nat) : Float := speed * (1000.0 / 3600.0)

-- Prove the length of the second train
theorem length_of_second_train :
  let relative_speed := kmh_to_ms speed_first_train + kmh_to_ms speed_second_train,
      total_distance := relative_speed * time_to_clear
  in total_distance - length_first_train = length_second_train :=
by
  sorry

end length_of_second_train_l795_795574


namespace S_n_value_l795_795466

noncomputable def S (n : ℕ) : ℕ := 
  n * (2 * a + (n - 1) * d) / 2 -- Using the arithmetic sequence sum formula

variables (a d : ℕ)

-- Conditions given:
axiom h₁ : S 3 = 3
axiom h₂ : S 9 = 24

-- Proving the conclusion:
theorem S_n_value : S 9 = 63 :=
sorry

end S_n_value_l795_795466


namespace operation_value_l795_795665

def operation (a b : ℤ) : ℤ := 3 * a - 3 * b + 4

theorem operation_value : operation 6 8 = -2 := by
  sorry

end operation_value_l795_795665


namespace proof_problem_l795_795816

variables {x y z : ℝ}

def T : set (ℝ × ℝ × ℝ) := 
{ p | ∃ (x y z : ℝ), p = (x, y, z) ∧ log 10 (x + y) = z ∧ log 10 (x^2 + y^2 + 4 * x * y) = z + 2 }

theorem proof_problem (p : ℝ × ℝ × ℝ) (hp : p ∈ T) : 
  ∃ (c d : ℝ), c = 1 ∧ d = 0 ∧ 
  (∀ (x y z : ℝ), (x, y, z) = p → x^4 + y^4 = c * 10 ^ (4 * z) + d * 10 ^ (3 * z)) :=
sorry

end proof_problem_l795_795816


namespace rose_spent_on_food_l795_795845

theorem rose_spent_on_food (T : ℝ) 
  (h_clothing : 0.5 * T = 0.5 * T)
  (h_other_items : 0.3 * T = 0.3 * T)
  (h_total_tax : 0.044 * T = 0.044 * T)
  (h_tax_clothing : 0.04 * 0.5 * T = 0.02 * T)
  (h_tax_other_items : 0.08 * 0.3 * T = 0.024 * T) :
  (0.2 * T = T - (0.5 * T + 0.3 * T)) :=
by sorry

end rose_spent_on_food_l795_795845


namespace find_A2_A7_l795_795071

theorem find_A2_A7 (A : ℕ → ℝ) (hA1A11 : A 11 - A 1 = 56)
  (hAiAi2 : ∀ i, 1 ≤ i ∧ i ≤ 9 → A (i+2) - A i ≤ 12)
  (hAjAj3 : ∀ j, 1 ≤ j ∧ j ≤ 8 → A (j+3) - A j ≥ 17) : 
  A 7 - A 2 = 29 :=
by
  sorry

end find_A2_A7_l795_795071


namespace cost_price_of_watch_l795_795188

variable (CP SP1 SP2 : ℝ)

theorem cost_price_of_watch (h1 : SP1 = 0.9 * CP)
  (h2 : SP2 = 1.04 * CP)
  (h3 : SP2 = SP1 + 200) : CP = 10000 / 7 := 
by
  sorry

end cost_price_of_watch_l795_795188


namespace even_function_b_eq_zero_l795_795012

theorem even_function_b_eq_zero (b : ℝ) :
  (∀ x : ℝ, (x^2 + b * x) = (x^2 - b * x)) → b = 0 :=
by sorry

end even_function_b_eq_zero_l795_795012


namespace range_of_q_l795_795256

noncomputable def q (x : ℝ) : ℝ :=
if prime (⌊x⌋) then x^2 - 1
else
  let z := greatest_prime_factor (⌊x⌋) in
  q z + (x - ⌊x⌋)

def range_q : set ℝ := { y | ∃ x, 2 ≤ x ∧ x ≤ 15 ∧ q x = y }

theorem range_of_q : range_q = { y | y = 3 ∨ y = 8 ∨ y = 24 ∨ y = 48 ∨ y = 120 ∨ y = 168 ∨ (3 ≤ y ∧ y ≤ 24) } :=
sorry

end range_of_q_l795_795256


namespace quadratic_with_transformed_roots_l795_795320

theorem quadratic_with_transformed_roots (p q : ℝ) :
  let a := -p + sqrt (p^2 - 4*q) / 2,
      b := -p - sqrt (p^2 - 4*q) / 2 in
  let y1 := (a + b)^2,
      y2 := (a - b)^2 in
  let P := -(2*p^2 - 4*q),
      Q := p^4 - 4*q*p^2 in
  (y - y1) * (y - y2) = y^2 + P*y + Q :=
sorry

end quadratic_with_transformed_roots_l795_795320


namespace ratio_b_to_c_l795_795880

theorem ratio_b_to_c (x a b c : ℤ) 
    (h1 : x = 100 * a + 10 * b + c)
    (h2 : a > 0)
    (h3 : 999 - x = 241) : (b : ℚ) / c = 5 / 8 :=
by
  sorry

end ratio_b_to_c_l795_795880


namespace distance_to_hole_correct_l795_795231

noncomputable def distance_to_hole := 
let first_hit := 180
let second_hit := first_hit / 2
in second_hit - 20

theorem distance_to_hole_correct : distance_to_hole = 70 := 
by
  sorry

end distance_to_hole_correct_l795_795231


namespace negation_of_proposition_l795_795537

variable (l : ℝ)

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x + l ≥ 0) ↔ (∀ x : ℝ, x + l < 0) := by
  sorry

end negation_of_proposition_l795_795537


namespace trajectory_equation_area_of_triangle_ACD_l795_795352

-- Condition 1: Distance condition forms the trajectory equation
theorem trajectory_equation (P : ℝ × ℝ) (h : |P.1 - 4| = 2 * real.sqrt ((P.1 - 1)^2 + (P.2)^2)) : 
(P.1^2 / 4 + P.2^2 / 3 = 1) := sorry

-- Definitions for points and line
def F1 : ℝ × ℝ := (1, 0)
def A : ℝ × ℝ := (2, 0)
def line_through_F1_with_slope_k (x y : ℝ) (k : ℝ) : Prop := y = k * (x - F1.1) + F1.2

-- Condition 2: The intersection points and area calculation
theorem area_of_triangle_ACD (C D : ℝ × ℝ) 
(hC : ∃ x, C = (x, x - 1) ∧ (x^2 / 4 + (x - 1)^2 / 3 = 1))
(hD : ∃ x, D = (x, x - 1) ∧ (x^2 / 4 + (x - 1)^2 / 3 = 1))
(h_area : ∃ S, S = 1/2 * real.sqrt ((A.1 - F1.1)^2 + (A.2 - F1.2)^2) * (C.2 - D.2)) :
S = 6 * real.sqrt 2 / 7 := sorry

end trajectory_equation_area_of_triangle_ACD_l795_795352


namespace solve_equation_l795_795112

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by
sorry

end solve_equation_l795_795112


namespace MrsB_students_received_A_l795_795413

theorem MrsB_students_received_A :
  let ratio_Abraham := (12 : ℚ) / 20
  let students_Berkeley := 30
  let ratio_A := (3 : ℚ) / 5
  (ratio_Abraham = ratio_A) → 
  (ratio_A * students_Berkeley) = 18 := 
by
  intro ratio_eq
  have : students_Berkeley = 30 := rfl
  rw [ratio_eq]
  calc
    (3 : ℚ) / 5 * 30 = 3 * 30 / 5 : by rw [mul_div_assoc, div_self (five_ne_zero : (5 : ℚ) ≠ 0)]
    ... = 90 / 5 : by norm_num
    ... = 18 : by norm_num
    sorry

end MrsB_students_received_A_l795_795413


namespace area_increase_l795_795620

theorem area_increase (r₁ r₂: ℝ) (A₁ A₂: ℝ) (side1 side2: ℝ) 
  (h1: side1 = 8) (h2: side2 = 12) (h3: r₁ = side2 / 2) (h4: r₂ = side1 / 2)
  (h5: A₁ = 2 * (1/2 * Real.pi * r₁ ^ 2) + 2 * (1/2 * Real.pi * r₂ ^ 2))
  (h6: A₂ = 4 * (Real.pi * r₂ ^ 2))
  (h7: A₁ = 52 * Real.pi) (h8: A₂ = 64 * Real.pi) :
  ((A₁ + A₂) - A₁) / A₁ * 100 = 123 :=
by
  sorry

end area_increase_l795_795620


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795981

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795981


namespace problem_solution_l795_795787

noncomputable def c1_rectangular_equation (x y : ℝ) : Prop :=
  (x^2) / 4 + (y^2) / 3 = 1

def parametric_curve_c2 (t α : ℝ) : ℝ × ℝ :=
  (1 + t * cos α, t * sin α)

theorem problem_solution (α : ℝ) (hα : 0 < α ∧ α < pi / 2) :
  (∃ x y : ℝ, c1_rectangular_equation x y) ∧ 
  (∃ t1 t2 : ℝ, let (x1, y1) := parametric_curve_c2 t1 α in 
                 let (x2, y2) := parametric_curve_c2 t2 α in 
                 x1 = x2 ∧ y1 = y2 ∧ abs t1 + abs t2 = 7 / 2 ∧ cos α = 2 * real.sqrt 7 / 7) := 
sorry

end problem_solution_l795_795787


namespace sets_of_earrings_l795_795850

namespace EarringsProblem

variables (magnets buttons gemstones earrings : ℕ)

theorem sets_of_earrings (h1 : gemstones = 24)
                         (h2 : gemstones = 3 * buttons)
                         (h3 : buttons = magnets / 2)
                         (h4 : earrings = magnets / 2)
                         (h5 : ∀ n : ℕ, n % 2 = 0 → ∃ k, n = 2 * k) :
  earrings = 8 :=
by
  sorry

end EarringsProblem

end sets_of_earrings_l795_795850


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795984

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795984


namespace perimeter_C_is_40_l795_795890

noncomputable def perimeter_of_figure_C (x y : ℝ) : ℝ :=
  2 * x + 6 * y

theorem perimeter_C_is_40 (x y : ℝ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  perimeter_of_figure_C x y = 40 :=
by
  -- Define initial conditions
  have eq1 : 3 * x + y = 28, by { rw [mul_assoc, mul_comm 3 x, add_assoc], exact (eq.div h1 2) }
  have eq2 : 2 * x + 3 * y = 28, by { rw [mul_assoc, mul_comm 2 x, add_assoc], exact (eq.div h2 2) }
  -- Assume the solutions are obtained from here
  have sol_x : x = 8, by sorry
  have sol_y : y = 4, by sorry
  -- Calculate the perimeter of figure C
  rw [perimeter_of_figure_C, sol_x, sol_y]
  norm_num
  trivial

-- Test case to ensure the code builds successfully
#eval perimeter_of_figure_C 8 4  -- Expected output: 40

end perimeter_C_is_40_l795_795890


namespace range_of_c_over_a_l795_795052

namespace TriangleProblem

variable {A B C : ℝ} -- internal angles
variable {a b c : ℝ} -- side lengths opposite to A, B, and C respectively

-- Conditions
axiom angle_in_triangle : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π
axiom sides_of_triangle : a > 0 ∧ b > 0 ∧ c > 0
axiom acute_triangle : A < π / 2 ∧ B < π / 2 ∧ C < π / 2
axiom given_equation : (√3 * c - 2 * sin B * sin C) = √3 * (b * sin B - a * sin A)

-- The theorem to prove
theorem range_of_c_over_a :
  \(\triangle ABC\ in_noncomputable ∧ acute A B C -> \frac{c}{a} \in \left(\frac{1}{2}, 2\right) \)
:= sorry

end TriangleProblem

end range_of_c_over_a_l795_795052


namespace find_angle_x_l795_795037

theorem find_angle_x 
  (AB CD : Type*) 
  (C : Point) 
  (D : Point) 
  (E : Point)
  (angle_ABC : ∠AB C) 
  (perpendicular : ∠CD AB = 90)
  (angle_ABDE : ∠DE AB = 52)
  (angle_DEE : ∠DEC = 62) :
  ∠DCE = 28 := 
sorry

end find_angle_x_l795_795037


namespace area_of_quadrilateral_l795_795038

-- Define the problem conditions
variables (A B C D E : Type) [plane_geometry A B C D E]

-- Define the right-angled triangles
variables (h1 : is_right_triangle A E B)
variables (h2 : is_right_triangle B E C)
variables (h3 : is_right_triangle C E D)

-- Define the specific angles in the triangles
variables (h4 : angle A E B = 60)
variables (h5 : angle B E C = 60)
variables (h6 : angle C E D = 60)

-- Define the length of AE
variable (AE : ℝ)
variable (h7 : AE = 30)

-- Define the problem to prove the area of quadrilateral ABCD
theorem area_of_quadrilateral (A B C D E : Type) [plane_geometry A B C D E]
  (h1 : is_right_triangle A E B)
  (h2 : is_right_triangle B E C)
  (h3 : is_right_triangle C E D)
  (h4 : angle A E B = 60)
  (h5 : angle B E C = 60)
  (h6 : angle C E D = 60)
  (h7 : AE = 30) :
  area_quadrilateral A B C D = 140.15625 * real.sqrt 3 :=
begin
  sorry
end

end area_of_quadrilateral_l795_795038


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795986

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795986


namespace discriminant_of_quadratic_eq_l795_795530

theorem discriminant_of_quadratic_eq : 
  ∀ (x : ℝ), x^2 - 4 * x + 3 = 0 →
  let a := 1
  let b := -4
  let c := 3
  discriminant a b c = 4 := 
by
  intro x h_eq
  let a := 1
  let b := -4
  let c := 3
  have H : discriminant a b c = b^2 - 4 * a * c := rfl
  rw [H, <-h_eq]
  -- We would have the proof steps here
  sorry

end discriminant_of_quadratic_eq_l795_795530


namespace factorization_of_a_square_minus_one_l795_795680

theorem factorization_of_a_square_minus_one (a : ℤ) : a^2 - 1 = (a + 1) * (a - 1) := 
  by sorry

end factorization_of_a_square_minus_one_l795_795680


namespace cos_theta_plus_pi_over_3_l795_795732

theorem cos_theta_plus_pi_over_3 {θ : ℝ} (h : Real.sin (θ / 2 + π / 6) = 2 / 3) :
  Real.cos (θ + π / 3) = 1 / 9 :=
by
  sorry

end cos_theta_plus_pi_over_3_l795_795732


namespace abs_diff_of_roots_eq_one_l795_795280

theorem abs_diff_of_roots_eq_one {p q : ℝ} (h₁ : p + q = 7) (h₂ : p * q = 12) : |p - q| = 1 := 
by 
  sorry

end abs_diff_of_roots_eq_one_l795_795280


namespace sales_relationship_maximize_profit_l795_795607

variable (x y w : ℤ)

-- Conditions
def condition1 := 8 ≤ x ∧ x ≤ 15
def condition2 := (9 : ℤ) = (105 : ℤ)
def condition3 := (11 : ℤ) = (95 : ℤ)
def cost_per_item := 8
def profit_eq := w = y * (x - cost_per_item)

-- Linear relationship between daily sales quantity and selling price
def linear_rel (x : ℤ) : ℤ := -5 * x + 150

-- Proof statements under given conditions

theorem sales_relationship :
  ∀ x : ℤ, (8 ≤ x ∧ x ≤ 15) →
  (y = linear_rel x) ∧
  (profit_eq w y x 8) :=
begin
  assume x,
  assume h : 8 ≤ x ∧ x ≤ 15,
  sorry
end

-- Proof to maximize the daily profit and find the maximum profit
theorem maximize_profit :
  ∀ x : ℤ, maximize_profit (w = linear_rel x * (x - 8)) := 
begin
  assume x,
  sorry
end

end sales_relationship_maximize_profit_l795_795607


namespace similar_quadrilateral_formed_by_perpendicular_feet_l795_795322

variables {α : Type*} [EuclideanGeometry α]

/-- Given a convex quadrilateral \(ABCD\) and a point \(O\) as the intersection
of its diagonals, and points \(A_1, B_1, C_1, D_1\) being the feet of the perpendiculars
dropped from vertices \(A, B, C, D\) respectively to the diagonals, proving that
quadrilateral \(A_1B_1C_1D_1\) is similar to the original quadrilateral \(ABCD\). -/
theorem similar_quadrilateral_formed_by_perpendicular_feet :
  ∀ (A B C D O A1 B1 C1 D1 : α),
    convex_quadrilateral A B C D →
    intersection_of_diagonals O A B C D →
    foot_perpendicular_from A O A1 →
    foot_perpendicular_from B O B1 →
    foot_perpendicular_from C O C1 →
    foot_perpendicular_from D O D1 →
    similar_quadrilaterals A B C D A1 B1 C1 D1 :=
sorry

end similar_quadrilateral_formed_by_perpendicular_feet_l795_795322


namespace perimeter_C_is_40_l795_795889

noncomputable def perimeter_of_figure_C (x y : ℝ) : ℝ :=
  2 * x + 6 * y

theorem perimeter_C_is_40 (x y : ℝ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  perimeter_of_figure_C x y = 40 :=
by
  -- Define initial conditions
  have eq1 : 3 * x + y = 28, by { rw [mul_assoc, mul_comm 3 x, add_assoc], exact (eq.div h1 2) }
  have eq2 : 2 * x + 3 * y = 28, by { rw [mul_assoc, mul_comm 2 x, add_assoc], exact (eq.div h2 2) }
  -- Assume the solutions are obtained from here
  have sol_x : x = 8, by sorry
  have sol_y : y = 4, by sorry
  -- Calculate the perimeter of figure C
  rw [perimeter_of_figure_C, sol_x, sol_y]
  norm_num
  trivial

-- Test case to ensure the code builds successfully
#eval perimeter_of_figure_C 8 4  -- Expected output: 40

end perimeter_C_is_40_l795_795889


namespace pentagon_area_l795_795036

structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 9 }
def C : Point := { x := 5, y := 8 }
def D : Point := { x := 8, y := 2 }
def E : Point := { x := 2, y := 2 }

-- B is the intersection of EC and AD
noncomputable def B : Point :=
  let l1 := λ p : Point, (E.y - C.y) / (E.x - C.x) * (p.x - E.x) + C.y 
  let l2 := λ p : Point, (A.y - D.y) / (A.x - D.x) * (p.x - A.x) + D.y 
  let x := (- A.y + D.y + (E.y - C.y) * E.x / (E.x - C.x) - (A.x - D.x) * E.x / (A.x - D.x))/
                    ((E.y - C.y) / (E.x - C.x) - (A.y - D.y) / (A.x - D.x))
  let y := (E.y - C.y) / (E.x - C.x) * (x - E.x) + E.y 
  { x := x, y := y }

noncomputable def shoelace_area (vertices : List Point) : ℝ :=
  let xy_pairs := vertices.zip (vertices.tail ++ [vertices.head])
  let sum1 := xy_pairs.foldl (λ acc ⟨p1, p2⟩ => acc + p1.x * p2.y) 0
  let sum2 := xy_pairs.foldl (λ acc ⟨p1, p2⟩ => acc + p1.y * p2.x) 0
  abs (sum1 - sum2) / 2

theorem pentagon_area :
  let vertices := [A, B, C, D, E] in
  shoelace_area vertices = 27 :=
by
  sorry

end pentagon_area_l795_795036


namespace mean_value_of_quadrilateral_angles_l795_795974

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795974


namespace remainder_333_pow_333_mod_11_l795_795174

theorem remainder_333_pow_333_mod_11 : (333 ^ 333) % 11 = 5 := by
  sorry

end remainder_333_pow_333_mod_11_l795_795174


namespace ratio_a_c_l795_795819

-- Definition of the function g(x)
def g (x : ℝ) : ℝ := (3 * x + 2) / (x - 4)

-- Defining the form of g⁻¹(x) to prove the ratio a / c
def g_inv_form (x : ℝ) : ℝ := (-4 * x - 2) / (x - 3)

-- The property we need to prove
theorem ratio_a_c : (a / c = -4) :=
by
  sorry

end ratio_a_c_l795_795819


namespace fifty_percent_of_number_l795_795014

-- Define the given condition
def given_condition (x : ℝ) : Prop :=
  0.6 * x = 42

-- Define the statement we need to prove
theorem fifty_percent_of_number (x : ℝ) (h : given_condition x) : 0.5 * x = 35 := by
  sorry

end fifty_percent_of_number_l795_795014


namespace sqrt_irrational_greater_than_sqrt3_l795_795587

noncomputable def sqrt_irrational_greater : Prop :=
  ∃ (x : ℝ), irrational x ∧ x > real.sqrt 3 ∧ x = real.sqrt 5

theorem sqrt_irrational_greater_than_sqrt3 : sqrt_irrational_greater :=
by
  use real.sqrt 5
  split
  · -- Proof that sqrt(5) is irrational
    sorry
  split
  · -- Proof that sqrt(5) > sqrt(3)
    sorry
  · -- Proof that x = sqrt(5)
    sorry

end sqrt_irrational_greater_than_sqrt3_l795_795587


namespace fixed_point_A_l795_795401

-- Definition of the point A
def point_A : ℝ × ℝ := (-1, 2)

theorem fixed_point_A (k : ℝ) :
  ∃ p : ℝ × ℝ, (3 + k) * p.1 + (1 - 2 * k) * p.2 + 1 + 5 * k = 0 ∧ p = point_A := 
begin
  use point_A,
  simp [point_A],
  sorry
end

end fixed_point_A_l795_795401


namespace tan_angle_BAE_unit_squares_l795_795831

theorem tan_angle_BAE_unit_squares:
  ∃ a b : ℕ, nat.gcd a b = 1 ∧ ∃ θ : ℝ,
  ABCD.unit_square ∧ AEFG.unit_square ∧
  intersection_area ABCD AEFG = 20 / 21 ∧
  angle BAE < degree_to_radian 45 ∧
  tan θ = a / b ∧
  (100 * a + b = 84041) :=
sorry

end tan_angle_BAE_unit_squares_l795_795831


namespace count_of_triples_is_three_l795_795694

noncomputable def count_valid_triples : ℕ := 
  nat.count
    (λ (x y z : ℝ), 
       x = 2020 - 2021 * (sign (y + z)) ∧
       y = 2020 - 2021 * (sign (x + z)) ∧
       z = 2020 - 2021 * (sign (x + y)))
    [(4041, -1, -1), (-1, 4041, -1), (-1, -1, 4041)]

theorem count_of_triples_is_three : count_valid_triples = 3 := sorry

end count_of_triples_is_three_l795_795694


namespace average_coins_collected_l795_795058

theorem average_coins_collected (coins_day1 : ℕ) (days : ℕ) (common_diff : ℕ) (s : List ℕ) :
  coins_day1 = 10 →
  days = 7 →
  common_diff = 10 →
  s = List.range (days) |>.map (λ i, coins_day1 + i * common_diff) →
  (s.sum / days) = 40 := by
  sorry

end average_coins_collected_l795_795058


namespace problem_part1_problem_part2_l795_795064

noncomputable def U : Set ℝ := Set.univ
noncomputable def A : Set ℝ := {x | x ≥ 1}
noncomputable def B : Set ℝ := {x | 0 < x ∧ x < 5}
noncomputable def compl (s : Set ℝ) : Set ℝ := {x | x ∉ s}

theorem problem_part1 :
  compl A ∪ B = {x | x < 5} :=
by {
  sorry
}

theorem problem_part2 :
  A ∩ compl B = {x | x ≥ 5} :=
by {
  sorry
}

end problem_part1_problem_part2_l795_795064


namespace distinct_pos_real_ints_l795_795867

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem distinct_pos_real_ints (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) (h4 : ∀ n : ℕ, (floor (n * a)) ∣ (floor (n * b))) : ∃ k l : ℤ, a = k ∧ b = l :=
by
  sorry

end distinct_pos_real_ints_l795_795867


namespace sum_div_by_24_l795_795811

theorem sum_div_by_24 (m n : ℕ) (h : ∃ k : ℤ, mn + 1 = 24 * k): (m + n) % 24 = 0 := 
by
  sorry

end sum_div_by_24_l795_795811


namespace sum_first_twelve_terms_l795_795339

theorem sum_first_twelve_terms (a : ℕ → ℕ) 
    (h : ∀ (p q : ℕ), p + q = 12 → p ≤ q → a p + a q = 2 ^ p) : 
    ∑ i in range 1 12, a i = 94 := 
sorry

end sum_first_twelve_terms_l795_795339


namespace complex_number_condition_l795_795295

noncomputable def num_complex_solutions : ℕ := 35280

theorem complex_number_condition (z : ℂ) (hz : complex.abs z = 1) : 
  z ^ (nat.factorial 8) - z ^ (nat.factorial 7) ∈ ℝ ↔ num_complex_solutions = 35280 :=
by sorry

end complex_number_condition_l795_795295


namespace original_jellybean_count_l795_795675

theorem original_jellybean_count (x : ℕ) (h : 28 = 0.343 * x) : x = 82 :=
by
  sorry

end original_jellybean_count_l795_795675


namespace man_age_difference_l795_795206

theorem man_age_difference (S M : ℕ) (h1 : S = 24) (h2 : M + 2 = 2 * (S + 2)) : M - S = 26 := by
  sorry

end man_age_difference_l795_795206


namespace count_five_digit_numbers_divisible_by_9_l795_795757

open Nat

def is_divisible_by_9 (n : ℕ) : Prop := 9 ∣ n

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem count_five_digit_numbers_divisible_by_9 : 
  { n : ℕ | is_five_digit n ∧ is_divisible_by_9 (sum_of_digits n) }.to_finset.card = 10000 :=
by
  sorry

end count_five_digit_numbers_divisible_by_9_l795_795757


namespace casey_pumping_rate_l795_795246

def total_water_needed (corn_plants pigs ducks : ℕ) (corn_plant_gallons pig_gallons duck_gallons : ℝ) : ℝ :=
  (corn_plants * corn_plant_gallons) + (pigs * pig_gallons) + (ducks * duck_gallons)

def gallons_per_minute (total_water minutes : ℝ) : ℝ :=
  total_water / minutes

theorem casey_pumping_rate :
  ∀ (corn_rows corn_plants_per_row pigs ducks : ℕ)
    (corn_plant_gallons pig_gallons duck_gallons minutes : ℝ),
    corn_rows = 4 →
    corn_plants_per_row = 15 →
    pigs = 10 →
    ducks = 20 →
    corn_plant_gallons = 0.5 →
    pig_gallons = 4 →
    duck_gallons = 0.25 →
    minutes = 25 →
    gallons_per_minute
      (total_water_needed 
        (corn_rows * corn_plants_per_row) 
        pigs 
        ducks 
        corn_plant_gallons 
        pig_gallons 
        duck_gallons)
      minutes = 3 :=
by
  intros
  sorry

end casey_pumping_rate_l795_795246


namespace ellipse_equation_hyperbola_equation_l795_795197

/-- Ellipse problem -/
def ellipse_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_equation (e a c b : ℝ) (h_c : c = 3) (h_e : e = 0.5) (h_a : a = 6) (h_b : b^2 = 27) :
  ellipse_eq x y a b := 
sorry

/-- Hyperbola problem -/
def hyperbola_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_equation (a b c : ℝ) 
  (h_c : c = 6) 
  (h_A : ∀ (x y : ℝ), (x, y) = (-5, 2) → hyperbola_eq x y a b) 
  (h_eq1 : a^2 + b^2 = 36) 
  (h_eq2 : 25 / (a^2) - 4 / (b^2) = 1) :
  hyperbola_eq x y a b :=
sorry

end ellipse_equation_hyperbola_equation_l795_795197


namespace percent_runs_by_running_l795_795200

theorem percent_runs_by_running 
  (total_runs boundaries sixes : ℕ) 
  (h_total_runs : total_runs = 150) 
  (h_boundaries : boundaries = 5) 
  (h_sixes : sixes = 5) : 
  (total_runs - (boundaries * 4 + sixes * 6)) / total_runs * 100 = 66.67 :=
by
  sorry

end percent_runs_by_running_l795_795200


namespace count_valid_pairs_l795_795390

theorem count_valid_pairs :
  {n : ℕ // ∃ pairs : Finset (ℕ × ℕ),
    (∀ (a b : ℕ), (a, b) ∈ pairs →
    (b ∣ (5 * a - 3) ∧ a ∣ (5 * b - 1))) ∧ pairs.card = n} =
  ⟨18, _⟩ :=
sorry

end count_valid_pairs_l795_795390


namespace interest_earned_l795_795477

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) := P * (1 + r) ^ t

theorem interest_earned :
  let P := 2000
  let r := 0.05
  let t := 5
  let A := compound_interest P r t
  A - P = 552.56 :=
by
  sorry

end interest_earned_l795_795477


namespace median_and_mode_of_successful_shots_l795_795420

theorem median_and_mode_of_successful_shots :
  let shots := [3, 6, 4, 6, 4, 3, 6, 5, 7]
  let sorted_shots := [3, 3, 4, 4, 5, 6, 6, 6, 7]
  let median := sorted_shots[4]  -- 4 is the index for the 5th element (0-based indexing)
  let mode := 6  -- determined by the number that appears most frequently
  median = 5 ∧ mode = 6 :=
by
  sorry

end median_and_mode_of_successful_shots_l795_795420


namespace intersection_of_A_and_B_l795_795377

def A := {0, 1, 2, 3}
def B := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
def intersection := A ∩ B

theorem intersection_of_A_and_B : intersection = {0, 1, 2} :=
by
  sorry

end intersection_of_A_and_B_l795_795377


namespace repeating_decimal_sum_l795_795175

theorem repeating_decimal_sum :
  (0.234234234... : ℝ) + (0.345345345... : ℝ) - (0.123123123... : ℝ) = (152 / 333 : ℝ) :=
sorry

end repeating_decimal_sum_l795_795175


namespace boat_upstream_time_l795_795053

theorem boat_upstream_time (v t : ℝ) (d c : ℝ) 
  (h1 : d = 24) (h2 : c = 1) (h3 : 4 * (v + c) = d) 
  (h4 : d / (v - c) = t) : t = 6 :=
by
  sorry

end boat_upstream_time_l795_795053


namespace proof_of_number_of_triples_l795_795690

noncomputable def numberOfTriples : ℕ :=
  -- Definitions of the conditions
  let condition1 : ℝ × ℝ × ℝ → Prop := λ xyz, xyz.1 = 2020 - 2021 * Real.sign (xyz.2 + xyz.3)
  let condition2 : ℝ × ℝ × ℝ → Prop := λ xyz, xyz.2 = 2020 - 2021 * Real.sign (xyz.1 + xyz.3)
  let condition3 : ℝ × ℝ × ℝ → Prop := λ xyz, xyz.3 = 2020 - 2021 * Real.sign (xyz.1 + xyz.2)
  -- Counting the number of triples
  let triples : Finset (ℝ × ℝ × ℝ) := {(4041, -1, -1), (-1, 4041, -1), (-1, -1, 4041)}
  triples.card

theorem proof_of_number_of_triples :
  numberOfTriples = 3 :=
sorry  -- Proof goes here

end proof_of_number_of_triples_l795_795690


namespace sequence_values_l795_795841

theorem sequence_values :
  ∃ (x y z: ℕ),
    let seq := [1, 3, 2, 6, 5, 15, 14, x, y, z, 122] in
    seq.nth 7 = some 42 ∧
    seq.nth 8 = some 41 ∧
    seq.nth 9 = some 123 ∧
    (∀ n, (n < 7 ∨ n = 10) → ((seq.nth n).val_or 0 * 3 = seq.nth (n+1)).val_or 0 ∨ (seq.nth (n+1)).val_or 0 = (seq.nth n).val_or 0 - 1) :=
begin
  sorry,
end

end sequence_values_l795_795841


namespace perimeter_C_l795_795912

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end perimeter_C_l795_795912


namespace anna_correct_percentage_l795_795237

theorem anna_correct_percentage :
  let test1_problems := 30
  let test1_score := 0.75
  let test2_problems := 50
  let test2_score := 0.85
  let test3_problems := 20
  let test3_score := 0.65
  let correct_test1 := test1_score * test1_problems
  let correct_test2 := test2_score * test2_problems
  let correct_test3 := test3_score * test3_problems
  let total_problems := test1_problems + test2_problems + test3_problems
  let total_correct := correct_test1 + correct_test2 + correct_test3
  (total_correct / total_problems) * 100 = 78 :=
by
  sorry

end anna_correct_percentage_l795_795237


namespace min_value2k2_minus_4n_l795_795350

-- We state the problem and set up the conditions
variable (k n : ℝ)
variable (nonneg_k : k ≥ 0)
variable (nonneg_n : n ≥ 0)
variable (eq1 : 2 * k + n = 2)

-- Main statement to prove
theorem min_value2k2_minus_4n : ∃ k n : ℝ, k ≥ 0 ∧ n ≥ 0 ∧ 2 * k + n = 2 ∧ (∀ k' n' : ℝ, k' ≥ 0 ∧ n' ≥ 0 ∧ 2 * k' + n' = 2 → 2 * k'^2 - 4 * n' ≥ -8) := 
sorry

end min_value2k2_minus_4n_l795_795350


namespace find_a_l795_795762

theorem find_a (a x : ℝ) (h : x = 3) (eqn : a * x + 4 = 1) : a = -1 :=
by
  -- Placeholder to indicate where the proof would go
  sorry

end find_a_l795_795762


namespace vector_magnitude_addition_l795_795733

variables (a b : ℝ^3)
variable (θ : ℝ)

-- Conditions
axiom proj_a_on_b : (a.dot_product b) / (b.dot_product b) = -1
axiom proj_b_on_a : (b.dot_product a) / (a.dot_product a) = -1 / 2
axiom norm_b : ∥b∥ = 1

theorem vector_magnitude_addition : ∥a + 2 • b∥ = 2 := 
by
  -- The proof steps are omitted.
  sorry

end vector_magnitude_addition_l795_795733


namespace problem_statement_l795_795725

theorem problem_statement:
  (∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1/2 ≤ 0) ∨ 
  (C₁ : ∀ x y : ℝ, x^2 / m^2 + y^2 / (2 * m + 8) = 1 → true) →
  (∃ x : ℝ, 2 * x^2 + (m - 1) * x + 1/2 ≤ 0) ∧ 
  (C₁ : ∀ x y : ℝ, x^2 / m^2 + y^2 / (2 * m + 8) = 1 → true) → false → 
  (3 ≤ m ∧ m ≤ 4) ∨ (-2 ≤ m ∧ m ≤ -1) ∨ (m ≤ -4) :=
  sorry

end problem_statement_l795_795725


namespace seq_a2015_l795_795747

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ a 2 = 6 ∧ ∀ n, a (n + 2) = a (n + 1) - a n

theorem seq_a2015 : ∀ (a : ℕ → ℤ), seq a → a 2015 = -3 :=
begin
  intros a h,
  sorry
end

end seq_a2015_l795_795747


namespace number_of_initial_daisies_l795_795631

-- Definition and conditions
def initial_daisies_growing (n : ℕ) :=
  let total_cornflowers := n - 1 in
  let initial_flowers := n + total_cornflowers in
  let total_dandelions := initial_flowers - 1 in
  4 * n - 3 = 101

theorem number_of_initial_daisies :
  (∃ n : ℕ, 4 * n - 3 = 101) ∧ (∃ n : ℕ, initial_daisies_growing n) :=
  sorry

end number_of_initial_daisies_l795_795631


namespace figure_C_perimeter_l795_795882

def is_perimeter (figure : Type) (perimeter : ℕ) : Prop :=
∃ x y : ℕ, (figure = 'A' → 6*x + 2*y = perimeter) ∧ 
           (figure = 'B' → 4*x + 6*y = perimeter) ∧
           (figure = 'C' → 2*x + 6*y = perimeter)

theorem figure_C_perimeter (hA : is_perimeter 'A' 56) (hB : is_perimeter 'B' 56) : 
  is_perimeter 'C' 40 :=
by
  sorry

end figure_C_perimeter_l795_795882


namespace tank_B_circumference_l795_795118

theorem tank_B_circumference :
  let π := Real.pi in
  ∀ (h_A h_B : ℝ) (C_A : ℝ) (volume_ratio : ℝ),
    (h_A = 5) →
    (h_B = 8) →
    (C_A = 4) →
    (volume_ratio = 0.10000000000000002) →
    let r_A := C_A / (2 * π) in
    let V_A := π * r_A^2 * h_A in
    let V_B := V_A / volume_ratio in
    let r_B := Real.sqrt (V_B / (π * h_B)) in
    2 * π * r_B = 10 :=
begin
  intros,
  sorry,
end

end tank_B_circumference_l795_795118


namespace sine_of_angle_BC1_CA1_l795_795047

variables (AB AD AA1 : ℝ)
variables (BAD BAA1 DAA1 : ℝ)

-- Conditions from the problem
def parallelepiped := 
  AB = 2 ∧ 
  AD = 2 ∧ 
  AA1 = 4 ∧ 
  BAD = 60 ∧ 
  BAA1 = 60 ∧ 
  DAA1 = 60

-- Goal: Prove that the sine value of the angle formed by BC1 and CA1 is the given value.
theorem sine_of_angle_BC1_CA1 
  (h : parallelepiped AB AD AA1 BAD BAA1 DAA1) :
  let BC1_C1 := 2 + 4 -- Just a placeholder for actual vector computation.
  let CA1_A1 := 4 - 2 - 2 -- Just a placeholder for actual vector computation.
  ∃ (sin_val : ℝ), 
  sin_val = (5*real.sqrt(7))/14 := sorry

end sine_of_angle_BC1_CA1_l795_795047


namespace inequality_solution_l795_795514

theorem inequality_solution (a : ℝ) (h : a > 0) :
  {x : ℝ | ax ^ 2 - (a + 1) * x + 1 < 0} =
    if a = 1 then ∅
    else if 0 < a ∧ a < 1 then {x : ℝ | 1 < x ∧ x < 1 / a}
    else if a > 1 then {x : ℝ | 1 / a < x ∧ x < 1} 
    else ∅ := sorry

end inequality_solution_l795_795514


namespace nearest_integer_to_expression_correct_l795_795170

noncomputable def nearest_integer_to_expression : ℤ :=
  Int.floor ((3 + Real.sqrt 2) ^ 6)

theorem nearest_integer_to_expression_correct : nearest_integer_to_expression = 7414 :=
by
  sorry

end nearest_integer_to_expression_correct_l795_795170


namespace simplify_expression_l795_795109

theorem simplify_expression (x y : ℝ) :  3 * x + 5 * x + 7 * x + 2 * y = 15 * x + 2 * y := 
by 
  sorry

end simplify_expression_l795_795109


namespace max_value_of_y_l795_795931

-- Define the function y
def y (x : ℝ) : ℝ := (Real.sin x - 2) * (Real.cos x - 2)

-- The statement of the maximum value of the function y
theorem max_value_of_y : ∃ x : ℝ, y x = 9 / 2 + 2 * Real.sqrt 2 :=
sorry

end max_value_of_y_l795_795931


namespace integer_solutions_l795_795277

theorem integer_solutions (n : ℤ) : (n^2 + 1) ∣ (n^5 + 3) ↔ n = -3 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2 := 
sorry

end integer_solutions_l795_795277


namespace calculate_total_votes_l795_795431

theorem calculate_total_votes
  (V : ℕ)
  (hA : 0.30 * V = 0)
  (hB : 0.25 * V = 0)
  (hC : 0.20 * V = 0)
  (hD : 0.25 * V = 0)
  (h_add_votes : 0.05 * (0.20 * V) = 0)
  (h_add_votes_2 : 0.05 * (0.25 * V) = 0)
  : V = 27000 :=
sorry

end calculate_total_votes_l795_795431


namespace distinct_arrangements_of_council_l795_795202

theorem distinct_arrangements_of_council : 
  let women := 9
  let men := 3
  ∃ (n : ℕ), n = 55 :=
begin
  sorry
end

end distinct_arrangements_of_council_l795_795202


namespace min_heaviest_weight_l795_795150

theorem min_heaviest_weight : 
  ∃ (w : ℕ), ∀ (weights : Fin 8 → ℕ),
    (∀ i j, i ≠ j → weights i ≠ weights j) ∧
    (∀ (a b c d : Fin 8),
      a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
      (weights a + weights b) ≠ (weights c + weights d) ∧ 
      max (max (weights a) (weights b)) (max (weights c) (weights d)) >= w) 
  → w = 34 := 
by
  sorry

end min_heaviest_weight_l795_795150


namespace students_admitted_to_universities_l795_795561

open Finset

theorem students_admitted_to_universities :
  let students := {A, B, C, D}
  let universities := {1, 2, 3, 4}
  (students.card = 4) → 
  (universities.card = 4) →
  (∃ s : Finset (Finset (Σ u : universities, students)), s.card = 144) := 
by
  sorry

end students_admitted_to_universities_l795_795561


namespace mean_value_of_quadrilateral_angles_l795_795996

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l795_795996


namespace asymptotes_equation_l795_795743

noncomputable def hyperbola_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  ∀ (m n : ℝ), 
  (m^2 / a^2 - n^2 / b^2 = 1) → 
  ∀ (P Q : ℝ × ℝ), 
  P = (-b, 0) → 
  Q = (b, 0) → 
  ∀ (M : ℝ × ℝ), 
  M = (m, n) → 
  (abs (dist P Q) = abs (dist M Q)) → 
  (tan (atan2 (n - 0) (m - b)) = -2 * sqrt 2) → 
  abs (dist P (0, 0)) / a = sqrt 41 / 5   

theorem asymptotes_equation (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : hyperbola_asymptotes a b ha hb :=
by
  sorry

end asymptotes_equation_l795_795743


namespace find_sum_of_reciprocal_extremes_l795_795935

theorem find_sum_of_reciprocal_extremes (x y : ℝ) (h : 4 * x ^ 2 - 5 * x * y + 4 * y ^ 2 = 5) : 
  let S := x ^ 2 + y ^ 2 in
  (1 / (10 / 3) + 1 / (10 / 13) = 8 / 5) := 
by
  sorry

end find_sum_of_reciprocal_extremes_l795_795935


namespace nate_age_when_ember_14_l795_795269

theorem nate_age_when_ember_14 (nate_age : ℕ) (ember_age : ℕ) 
    (h1 : nate_age = 14) (h2 : ember_age = nate_age / 2) 
    (h3 : ember_age = 7) (h4 : ember_14_years_later : ℕ)
    (h : ember_14_years_later = ember_age + (14 - ember_age)) :
  ember_14_years_later = 14 → (nate_age + (14 - ember_age)) = 21 := by
  intros h_ember_14
  sorry

end nate_age_when_ember_14_l795_795269


namespace problem_36_22_solution_l795_795959

noncomputable def B : ℕ → ℚ
| 0     := 1
| (n+1) := - (∑ i in finset.range (n + 1), nat.choose (n + 1) (i + 1) * B (n - i))

theorem problem_36_22_solution :
  B 1 = -1 ∧
  B 2 = 1 / 2 ∧
  B 3 = 1 / 6 ∧
  B 4 = 1 / 12 ∧
  B 5 = -37 / 60 :=
sorry

end problem_36_22_solution_l795_795959


namespace no_n_divisible_by_1955_l795_795673

theorem no_n_divisible_by_1955 : ∀ n : ℕ, ¬ (1955 ∣ (n^2 + n + 1)) := by
  sorry

end no_n_divisible_by_1955_l795_795673


namespace perimeter_C_correct_l795_795902

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_l795_795902


namespace all_points_collinear_l795_795814

open Set

variables {Point : Type} [Inhabited Point] [Fintype Point]

def collinear (E : Set Point) : Prop :=
  ∃ (ℓ : Line Point), ∀ p ∈ E, p ∈ ℓ

def on_line (ℓ : Line Point) (p : Point) : Prop :=
  p ∈ ℓ

noncomputable def finite_set_with_property (E : Set Point) [Fintype E] : Prop :=
  ∀ (A B : Point), A ∈ E → B ∈ E → ∃ (C : Point), C ∈ E ∧ on_line (Line.mk A B) C

theorem all_points_collinear (E : Set Point) [Fintype E] (h : finite_set_with_property E) : collinear E :=
  sorry

end all_points_collinear_l795_795814


namespace smallest_sum_three_primes_l795_795131

def is_prime (n : ℕ) := nat.prime n

noncomputable def hcf (a b : ℕ) : ℕ := nat.gcd a b

theorem smallest_sum_three_primes (Q R S a b c : ℕ) 
  (hCFQR : hcf Q R = a) 
  (hCFQS : hcf Q S = b) 
  (hCFRS : hcf R S = c) 
  (prime_a : is_prime a) 
  (prime_b : is_prime b) 
  (prime_c : is_prime c) 
  (disjoint_primes : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (Q_def : Q = a * b)
  (R_def : R = a * c)
  (S_def : S = b * c) : 
  Q + R + S = 31 :=
by sorry

end smallest_sum_three_primes_l795_795131


namespace moles_of_HCl_formed_l795_795689

-- Define the reaction
def balancedReaction (CH4 Cl2 CH3Cl HCl : ℕ) : Prop :=
  CH4 + Cl2 = CH3Cl + HCl

-- Number of moles given
def molesCH4 := 2
def molesCl2 := 4

-- Theorem statement
theorem moles_of_HCl_formed :
  ∀ CH4 Cl2 CH3Cl HCl : ℕ, balancedReaction CH4 Cl2 CH3Cl HCl →
  CH4 = molesCH4 →
  Cl2 = molesCl2 →
  HCl = 2 := sorry

end moles_of_HCl_formed_l795_795689


namespace tablet_area_difference_l795_795938

open Real -- Open the real numbers namespace

noncomputable def diagonal_to_side_length (d : ℝ) : ℝ :=
  (d^2 / 2)^0.5

theorem tablet_area_difference :
  let s8 := diagonal_to_side_length 8
  let s7 := diagonal_to_side_length 7
  (s8^2 - s7^2) * 2 = 7.5 := 
by
  -- conversion for Lean syntax
  unfold diagonal_to_side_length
  -- replace with corresponding values
  simp [real.sqrt]
  sorry

end tablet_area_difference_l795_795938


namespace nearest_integer_to_expression_correct_l795_795171

noncomputable def nearest_integer_to_expression : ℤ :=
  Int.floor ((3 + Real.sqrt 2) ^ 6)

theorem nearest_integer_to_expression_correct : nearest_integer_to_expression = 7414 :=
by
  sorry

end nearest_integer_to_expression_correct_l795_795171


namespace perpendicular_condition_parallel_condition_l795_795753

variables {R : Type*} [LinearOrderedField R]

def line1 (m n x y : R) : R := m * x + 8 * y + n
def line2 (m n x y : R) : R := 2 * x + m * y - 1 + n / 2

def lines_perpendicular (m n : R) : Prop :=
  if m = 0 then true else false

def lines_parallel (m n : R) : Prop :=
  m = 4 ∧ (n = n) ∨ m = -4 ∧ n ≠ -2

theorem perpendicular_condition (m n : R) :
  (lines_perpendicular m n) ↔ (m = 0 ∧ n ∈ set.univ) := sorry

theorem parallel_condition (m n : R) :
  (lines_parallel m n) ↔ (m = 4 ∧ n ∈ set.univ ∨ m = -4 ∧ n ≠ -2) := sorry

end perpendicular_condition_parallel_condition_l795_795753


namespace range_of_a_l795_795101

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + 1 > 0 ∨ ∃ x : ℝ, x^2 - x + a = 0) ∧ 
  ¬ (∀ x : ℝ, x^2 + a * x + 1 > 0 ∧ ∃ x : ℝ, x^2 - x + a = 0) ↔
  a ≤ -2 ∨ (1/4 < a ∧ a < 2) :=
begin
  sorry
end

end range_of_a_l795_795101


namespace sum_of_distances_squared_independent_l795_795220

noncomputable def sum_of_distances_squared (n : ℕ) (a r : ℝ) (vertices : Fin n → ℝ) : ℝ :=
  n * (a^2 + r^2)

theorem sum_of_distances_squared_independent (n : ℕ) (a r : ℝ) (P O : ℝ) (vertices : Fin n → ℝ) (h1 : ∀ i, vertices i = r) :
  (Finset.univ.sum (λ i, (P - vertices i)^2)) = sum_of_distances_squared n a r vertices :=
by
  sorry

end sum_of_distances_squared_independent_l795_795220


namespace solve_for_a4b4_l795_795676

theorem solve_for_a4b4 (
    a1 a2 a3 a4 b1 b2 b3 b4 : ℝ
) (h1 : a1 * b1 + a2 * b3 = 1) 
  (h2 : a1 * b2 + a2 * b4 = 0) 
  (h3 : a3 * b1 + a4 * b3 = 0)
  (h4 : a3 * b2 + a4 * b4 = 1)
  (h5 : a2 * b3 = 7) : 
  a4 * b4 = -6 :=
sorry

end solve_for_a4b4_l795_795676


namespace difference_in_students_specific_case_diff_l795_795780

-- Define the variables and conditions
variables (a b : ℕ)

-- Condition: a > b
axiom h1 : a > b

-- Definition of eighth grade students
def eighth_grade_students := (3 * a + b) * (2 * a + 2 * b)

-- Definition of seventh grade students
def seventh_grade_students := (2 * (a + b)) ^ 2

-- Theorem for the difference in the number of students
theorem difference_in_students : (eighth_grade_students a b) - (seventh_grade_students a b) = 2 * a^2 - 2 * b^2 :=
sorry

-- Theorem for the specific example when a = 10 and b = 2
theorem specific_case_diff : eighth_grade_students 10 2 - seventh_grade_students 10 2 = 192 :=
sorry

end difference_in_students_specific_case_diff_l795_795780


namespace sampling_method_is_systematic_l795_795204

-- Conditions
def grade := 12 -- Total number of classes
def students_per_class := 50 -- Students numbered from 1 to 50
def selected_student_numbers (class_number: ℕ) : ℕ := 14 -- Student number 14 is selected from each class

-- Question transformed into a proof problem statement
theorem sampling_method_is_systematic :
  (∀ class_number: ℕ, class_number ≤ grade -> selected_student_numbers class_number == 14) →
  (sampling_method_used = "Systematic Sampling") :=
begin
  intro _,
  sorry
end

end sampling_method_is_systematic_l795_795204


namespace triangle_center_congruence_l795_795808

open Triangle

variables (A B C H P Q R : Point)
variables (circumcircle_BCH circumcircle_CAH circumcircle_ABH : Triangle => Circle)
variables (acute : isAcuteTriangle (Triangle.mk A B C))
variables (orthocenter : isOrthocenter H (Triangle.mk A B C))
variables (P_center : P = circumcenter (Triangle.mk B C H) circumcircle_BCH)
variables (Q_center : Q = circumcenter (Triangle.mk C A H) circumcircle_CAH)
variables (R_center : R = circumcenter (Triangle.mk A B H) circumcircle_ABH)
variables (PQ_cong_AB : congruent_triangles (Triangle.mk P Q R) (Triangle.mk A B C))

theorem triangle_center_congruence :
  congruent_triangles (Triangle.mk P Q R) (Triangle.mk A B C) :=
sorry

end triangle_center_congruence_l795_795808


namespace number_of_five_digit_palindromes_l795_795257

theorem number_of_five_digit_palindromes : 
  let A := finset.range 1 10  -- 1 to 9
  let B := finset.range 0 10  -- 0 to 9
  let C := finset.range 0 10  -- 0 to 9
  A.card * B.card * C.card = 900 :=
by
  sorry

end number_of_five_digit_palindromes_l795_795257


namespace deductive_reasoning_l795_795952

theorem deductive_reasoning
    (parallel_trans : ∀ {l m n : Type}, l ∥ m → m ∥ n → l ∥ n)
    (a b c : Type)
    (ha : a ∥ b)
    (hb : b ∥ c) :
    a ∥ c :=
by 
  apply parallel_trans ha hb
-- this theorem represents that a ∥ c deduced from the conditions using deductive reasoning.

end deductive_reasoning_l795_795952


namespace pascal_binomial_sum_l795_795063

theorem pascal_binomial_sum :
  (∑ i in Finset.range 502, ((Nat.choose 502 i) / (Nat.choose 504 i))) - 
  (∑ i in Finset.range 500, ((Nat.choose 500 i) / (Nat.choose 502 i))) = 2 := 
by
  sorry

end pascal_binomial_sum_l795_795063


namespace problem_equivalent_l795_795437

-- Define the problem conditions
def an (n : ℕ) : ℤ := -4 * n + 2

-- Arithmetic sequence: given conditions
axiom arith_seq_cond1 : an 2 + an 7 = -32
axiom arith_seq_cond2 : an 3 + an 8 = -40

-- Suppose the sequence {an + bn} is geometric with first term 1 and common ratio 2
def geom_seq (n : ℕ) : ℤ := 2 ^ (n - 1)
def bn (n : ℕ) : ℤ := geom_seq n - an n

-- To prove: sum of the first n terms of {bn}, denoted as Sn
def Sn (n : ℕ) : ℤ := (n * (2 + 4 * n - 2)) / 2 + (1 - 2 ^ n) / (1 - 2)

theorem problem_equivalent (n : ℕ) :
  an 2 + an 7 = -32 ∧
  an 3 + an 8 = -40 ∧
  (∀ n : ℕ, an n + bn n = geom_seq n) →
  Sn n = 2 * n ^ 2 + 2 ^ n - 1 :=
by
  intros h
  sorry

end problem_equivalent_l795_795437


namespace ratio_fifteenth_term_l795_795465

-- Definitions of S_n and T_n based on the given conditions
def S_n (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2
def T_n (b e n : ℕ) : ℕ := n * (2 * b + (n - 1) * e) / 2

-- Statement of the problem
theorem ratio_fifteenth_term 
  (a b d e : ℕ) 
  (h : ∀ n, (S_n a d n : ℚ) / (T_n b e n : ℚ) = (9 * n + 5) / (6 * n + 31)) : 
  (a + 14 * d : ℚ) / (b + 14 * e : ℚ) = (92 : ℚ) / 71 :=
by sorry

end ratio_fifteenth_term_l795_795465


namespace mean_value_of_quadrilateral_angles_l795_795973

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795973


namespace perimeter_C_l795_795920

theorem perimeter_C (x y : ℕ) 
  (h1 : 6 * x + 2 * y = 56)
  (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 := 
by
  sorry

end perimeter_C_l795_795920


namespace sum_first_seven_terms_geometric_seq_l795_795299

theorem sum_first_seven_terms_geometric_seq :
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 2
  let S_7 := a * (1 - r^7) / (1 - r)
  S_7 = 127 / 192 := 
by
  let a := (1 : ℝ) / 3
  let r := (1 : ℝ) / 2
  let S_7 := a * (1 - r^7) / (1 - r)
  have h : S_7 = 127 / 192 := sorry
  exact h

end sum_first_seven_terms_geometric_seq_l795_795299


namespace fraction_calculation_l795_795243

noncomputable def improper_frac_1 : ℚ := 21 / 8
noncomputable def improper_frac_2 : ℚ := 33 / 14
noncomputable def improper_frac_3 : ℚ := 37 / 12
noncomputable def improper_frac_4 : ℚ := 35 / 8
noncomputable def improper_frac_5 : ℚ := 179 / 9

theorem fraction_calculation :
  (improper_frac_1 - (2 / 3) * improper_frac_2) / ((improper_frac_3 + improper_frac_4) / improper_frac_5) = 59 / 21 :=
by
  sorry

end fraction_calculation_l795_795243


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795991

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795991


namespace unique_solution_exists_l795_795736

theorem unique_solution_exists (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃! (x : ℝ), a^x = log (x) / log (1/4) :=
by
  sorry

end unique_solution_exists_l795_795736


namespace intersection_A_B_subset_A_B_l795_795748

-- Definitions for the sets A and B
def set_A (a : ℝ) : Set ℝ := {x | 2 * a - 1 ≤ x ∧ x ≤ a + 3}
def set_B : Set ℝ := {x | x < -1 ∨ x > 5}

-- First proof problem: Intersection
theorem intersection_A_B (a : ℝ) (ha : a = -2) :
  set_A a ∩ set_B = {x | -5 ≤ x ∧ x < -1} :=
sorry

-- Second proof problem: Subset
theorem subset_A_B (a : ℝ) :
  set_A a ⊆ set_B ↔ (a ≤ -4 ∨ a ≥ 3) :=
sorry

end intersection_A_B_subset_A_B_l795_795748


namespace incorrect_operation_B_l795_795582

variables (a b c : ℝ)

theorem incorrect_operation_B : (c - 2 * (a + b)) ≠ (c - 2 * a + 2 * b) := by
  sorry

end incorrect_operation_B_l795_795582


namespace occupancy_ratio_is_075_l795_795840

-- Define the necessary conditions
def total_units := 100
def rent_per_unit_per_month := 400
def total_annual_income := 360000
def potential_income := total_units * rent_per_unit_per_month * 12

-- Define the occupancy ratio calculation
def occupancy_ratio : ℝ := total_annual_income / potential_income

-- State the theorem to prove the occupancy ratio
theorem occupancy_ratio_is_075 : occupancy_ratio = 0.75 :=
by
  sorry

end occupancy_ratio_is_075_l795_795840


namespace mean_value_of_quadrilateral_angles_l795_795975

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795975


namespace solve_quartic_eq_l795_795685

theorem solve_quartic_eq :
  { x : ℂ | x^4 - 16*x^2 + 144 = 0 } =
  {ℂ (\sqrt{10} + \sqrt{5} * complex.I), -ℂ (\sqrt{10} + \sqrt{5} * complex.I),
  \sqrt{10} - \sqrt{5} * complex.I, -\sqrt{10} - \sqrt{5} * complex.I} :=
by
  sorry -- Proof is skipped

end solve_quartic_eq_l795_795685


namespace sum_possible_values_of_k_l795_795467

-- Defining the main components required for the problem
variables {a b c k : ℂ}
variables (distinct_abc : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (eq_k1 : (a + 1) / (2 - b) = k)
variables (eq_k2 : (b + 1) / (2 - c) = k)
variables (eq_k3 : (c + 1) / (2 - a) = k)

theorem sum_possible_values_of_k: 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) → 
  ((a + 1) / (2 - b) = k) → 
  ((b + 1) / (2 - c) = k) →
  ((c + 1) / (2 - a) = k) → 
  k = 1 → 
  (k = 1 + complex.I) ∨ (k = 1 - complex.I) :=
by sorry

end sum_possible_values_of_k_l795_795467


namespace letters_difference_l795_795054

theorem letters_difference :
  ∀ (m_l a_l : ℕ), m_l = 8 → a_l = 7 → (m_l - a_l) = 1 :=
by
  intros m_l a_l h_m h_a
  rw [h_m, h_a]
  rfl

end letters_difference_l795_795054


namespace fraction_nonnegative_iff_l795_795110

-- Define the quadratic polynomial q(x)
def q (x : ℝ) : ℝ := x^2 + 4 * x + 13

-- Define the fraction we are interested in
def fraction_nonnegative (x : ℝ) : Prop := (x + 4) / q(x) ≥ 0

-- Prove that the fraction is nonnegative if and only if x is at least -4
theorem fraction_nonnegative_iff (x : ℝ) : fraction_nonnegative x ↔ x ≥ -4 :=
by
  -- Skip the proof
  sorry

end fraction_nonnegative_iff_l795_795110


namespace cos_B_of_triangleABC_perimeter_of_triangleABC_l795_795777

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) :=
  a = 2 ∧ b = 5 ∧ c = 6 ∧ a*b*c = 60

theorem cos_B_of_triangleABC 
  (a b c k : ℝ) (A B C : ℝ)
  (h1: a = 2 * k) 
  (h2: b = 5 * k) 
  (h3: c = 6 * k) 
  : cos B = 25/48 := 
  sorry

theorem perimeter_of_triangleABC 
  (a b c k : ℝ) (A B C : ℝ)
  (area_triangle: 0.5 * a * c * sin B = (3*sqrt(39))/4) 
  (h1: a = 2 * k) 
  (h2: b = 5 * k) 
  (h3: c = 6 * k) 
  : a + b + c = 13 := 
  sorry

end cos_B_of_triangleABC_perimeter_of_triangleABC_l795_795777


namespace part1_part2_l795_795141

noncomputable def seq_a : ℕ → ℝ
| 0       := 2
| 1       := 3
| (n + 2) := 3 * seq_a (n + 1) - 2 * seq_a n

def seq_d (n : ℕ) : ℝ := seq_a (n + 1) - seq_a n

theorem part1 : ∃ q : ℝ, (seq_d (n+1) = q * seq_d n)  ∧ (∀ n, seq_d n = 2^(n-1)) := 
sorry

def seq_reciprocal_sum : ℕ → ℝ
| 0 := 0
| (n + 1) := seq_reciprocal_sum n + 1 / seq_a n.succ

theorem part2 : ∀ n : ℕ, seq_reciprocal_sum n < 3 / 2 :=
sorry

end part1_part2_l795_795141


namespace evaluate_power_l795_795760

theorem evaluate_power (x : ℝ) (hx : (8:ℝ)^(2 * x) = 11) : 
  2^(x + 1.5) = 11^(1 / 6) * 2 * Real.sqrt 2 :=
by 
  sorry

end evaluate_power_l795_795760


namespace tetrahedron_plane_intersection_l795_795099

-- Define the conditions for the tetrahedron/pyramid DABC.
structure TrianglePyramid (A B C D : Type) :=
  (DABC : boolean)
  (right_dihedral_angles : (triangle A B C) → (triangle B C D) → (triangle C D A) → (triangle D A B) → Prop)
  (AB_eq_BC : A ≠ B → B ≠ C → A = B → B = C → A = C)

noncomputable def circumcenter_intersection {A B C D : Type} 
  [TrianglePyramid A B C D]
  (right_angles : ∀ (triangle A B C) (triangle B C D) (triangle C D A) (triangle D A B), Prop)
  (eq_AB_BC : ∀ (A B C : A ≠ B → B ≠ C → A = B → B = C → A = C), Prop) :=
∃ (X : Type), (
  ∀ (plane_ACD plane_MOB plane_ABD plane_KOC : Type), 
  plane_perpendicular_to_face plane_ACD (face B D C) →
  plane_perpendicular_to_face plane_MOB (face A D B) →
  plane_perpendicular_to_face plane_ABD (face C D A) →
  plane_perpendicular_to_face plane_KOC (face A B C) →
  X ∈ plane_ACD ∧ X ∈ plane_MOB ∧ X ∈ plane_ABD ∧ X ∈ plane_KOC)

-- statement of the theorem
theorem tetrahedron_plane_intersection 
{A B C D : Type} [TrianglePyramid A B C D] :
  circumcenter_intersection (λ _ _ _ _ => sorry) (λ _ => sorry) :=
sorry

end tetrahedron_plane_intersection_l795_795099


namespace quadratic_roots_m_value_l795_795823

theorem quadratic_roots_m_value
  (x1 x2 m : ℝ)
  (h1 : x1^2 + 2 * x1 + m = 0)
  (h2 : x2^2 + 2 * x2 + m = 0)
  (h3 : x1 + x2 = x1 * x2 - 1) :
  m = -1 :=
sorry

end quadratic_roots_m_value_l795_795823


namespace perimeter_of_C_l795_795925

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end perimeter_of_C_l795_795925


namespace inequality_ae_pow_b_lt_be_pow_a_l795_795329

theorem inequality_ae_pow_b_lt_be_pow_a (a b : ℝ) (ha : a > 1) (hb : b > 1) (h : a > b) : a * exp b < b * exp a := 
sorry

end inequality_ae_pow_b_lt_be_pow_a_l795_795329


namespace min_distance_CD_l795_795501

def rational_path (t : ℝ) : ℝ × ℝ := (2 * Real.cos t, 2 * Real.sin t)

def irrational_path (t : ℝ) : ℝ × ℝ := (-1 + 3 * Real.cos (t / Real.sqrt 3), 3 * Real.sin (t / Real.sqrt 3))

theorem min_distance_CD : ∀ C D : ℝ × ℝ, 
  (∃ t1 : ℝ, C = rational_path t1) → (∃ t2 : ℝ, D = irrational_path t2) → 
  ∃ CD : ℝ, CD = ℝ.sqrt ((fst D - fst C) ^ 2 + (snd D - snd C) ^ 2) → CD = 2 :=
sorry

end min_distance_CD_l795_795501


namespace probability_check_on_friday_l795_795960

theorem probability_check_on_friday
    (P_no_check : ℝ) (P_check_per_day : ℝ) (P_check_friday : ℝ)
    (A : ℝ) (B : ℝ)
    (h1 : P_no_check = 1 / 2)
    (h2 : ∀ d, P_check_per_day = 1 / 2 * 1 / 5)
    (h3 : P_check_friday = 1 / 10)
    (h4 : A = P_no_check + P_check_friday)
    (h5 : A = 3 / 5)
    (h6 : A = 1)
    : (\frac{1 / 10} {3 / 5}) = 1 / 6 :=
by
    sorry

end probability_check_on_friday_l795_795960


namespace max_black_pieces_l795_795945

-- Defining a piece and its color
inductive Color
  | black
  | white

-- Defining the operation rules
-- True implies same colors, False implies different colors
def insert_piece (c1 c2 : Color) : Color :=
  if c1 = c2 then Color.white else Color.black

-- Starting conditions: initial state of pieces in a circle and the rules
variable (initial_state : List Color)
variable (circle_size : Nat)

axiom initial_conditions : initial_state.length = circle_size ∧ circle_size = 5

-- The theorem stating the maximum number of black pieces that can be present
theorem max_black_pieces {initial_state : List Color} (h : initial_conditions initial_state 5) : 
∃ (k : Nat), k ≤ 4 :=
begin
  sorry
end

end max_black_pieces_l795_795945


namespace surface_area_intersection_l795_795430

noncomputable theory

open Classical

variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

def sphere (R : ℝ) (center : Type) : Set Type := { p | dist center p = R }
def projection (A : Type) (plane : Type) : Type := sorry -- Abstracting the projection point H here

def BCD : Type := sorry -- Abstracting the plane BCD

axiom AB_AC_AD_eq : dist A B = 2 * Real.sqrt 5 ∧ dist A C = 2 * Real.sqrt 5 ∧ dist A D = 2 * Real.sqrt 5
axiom sphere_radius : (sphere 5 A).nonempty

theorem surface_area_intersection : ∃ r, r = 4 ∧ 4 * Real.pi * r^2 = 16 * Real.pi :=
by sorry

end surface_area_intersection_l795_795430


namespace perimeter_C_correct_l795_795906

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_l795_795906


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795989

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795989


namespace maximum_value_m_range_l795_795345

noncomputable def reach_hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

noncomputable def distance_to_line (x y m : ℝ) : Prop := 
  ∀ (x y : ℝ), 
  (x^2 - y^2 = 1) →
  let d := |x - y + 2| / real.sqrt (1^2 + (-1)^2) in
  d > m

theorem maximum_value_m_range : 
  ( ∀ (x y m: ℝ), 
  (x^2 - y^2 = 1) → 
  ( ∀ x y, x^2 - y^2 = 1 → (|x - y + 2| / real.sqrt (1^2 + (-1)^2)) > m )
  ) → 
  m ≤ real.sqrt 2 :=
sorry

end maximum_value_m_range_l795_795345


namespace find_halls_per_floor_l795_795478

theorem find_halls_per_floor
  (H : ℤ)
  (floors_first_wing : ℤ := 9)
  (rooms_per_hall_first_wing : ℤ := 32)
  (floors_second_wing : ℤ := 7)
  (halls_per_floor_second_wing : ℤ := 9)
  (rooms_per_hall_second_wing : ℤ := 40)
  (total_rooms : ℤ := 4248) :
  9 * H * 32 + 7 * 9 * 40 = 4248 → H = 6 :=
by
  sorry

end find_halls_per_floor_l795_795478


namespace similar_triangles_area_ratio_not_equal_similarity_ratio_l795_795955

theorem similar_triangles_area_ratio_not_equal_similarity_ratio
  (ΔABC ΔDEF : Type)
  [triangle ΔABC] [triangle ΔDEF]
  (h_similar : similar ΔABC ΔDEF)
  (k : ℝ) 
  (h_ratio : ratio_of_sides ΔABC ΔDEF = k) :
  ratio_of_areas ΔABC ΔDEF ≠ k :=
by
  sorry

end similar_triangles_area_ratio_not_equal_similarity_ratio_l795_795955


namespace product_of_areas_l795_795688

noncomputable def greatest_k (ABCD : Tetrahedron) (V : ℝ) (A1 A2 A3 : ℝ): ℝ :=
  9 / 2

theorem product_of_areas (ABCD : Tetrahedron) (V : ℝ) (A1 A2 A3 : ℝ)
  (hV : volume ABCD = V) 
  (hA1 : area (face ABC) = A1)
  (hA2 : area (face ABD) = A2)
  (hA3 : area (face ACD) = A3) : 
  A1 * A2 * A3 ≥ (greatest_k ABCD V A1 A2 A3) * V^2 :=
begin
  sorry
end

end product_of_areas_l795_795688


namespace range_of_k_l795_795132

-- Given the equation of the hyperbola and the eccentricity conditions
def hyperbola_eccentricity (k : ℝ) : Prop :=
  let e := sqrt (1 - k / 4)
  in (e > 1) ∧ (e < 3)

-- The main statement expressing the desired problem
theorem range_of_k (k : ℝ) (h : hyperbola_eccentricity k) : -32 < k ∧ k < 0 :=
sorry

end range_of_k_l795_795132


namespace greatest_value_of_sum_l795_795087

variable (a b c : ℕ)

theorem greatest_value_of_sum
    (h1 : 2022 < a)
    (h2 : 2022 < b)
    (h3 : 2022 < c)
    (h4 : ∃ k1 : ℕ, a + b = k1 * (c - 2022))
    (h5 : ∃ k2 : ℕ, a + c = k2 * (b - 2022))
    (h6 : ∃ k3 : ℕ, b + c = k3 * (a - 2022)) :
    a + b + c = 2022 * 85 := 
  sorry

end greatest_value_of_sum_l795_795087


namespace class_C_has_170_people_l795_795655

noncomputable def num_people_C (num_A : ℕ) (num_B : ℕ) (num_D : ℕ) : ℕ :=
  num_D + 10

theorem class_C_has_170_people :
  ∀ (num_B num_A num_C num_D : ℕ),
  num_A = 2 * num_B →
  num_A = num_C / 3 →
  num_B = 20 →
  num_D = num_A + 3 * num_A →
  num_C = num_D + 10 →
  num_C = 170 :=
by
  intros num_B num_A num_C num_D h₁ h₂ h₃ h₄ h₅
  rw [h₃, mul_comm] at h₁
  have h_A : num_A = 40 := by
    calc
      num_A = 2 * 20 := by rw [h₃]
      ... = 40 := by norm_num
  rw [h_A] at h₂
  have h_C : num_C = 3 * 40 := by
    calc
      num_C = 3 * num_A := by rw [← h₂, mul_comm]
      ... = 3 * 40 := by rw h_A
      ... = 120 := by norm_num
  rw [h_A] at h₄
  have h_D : num_D = 40 + 3 * 40 := by
    calc
      num_D = num_A + 3 * num_A := by rw [← h₄]
      ... = 40 + 3 * 40 := by rw h_A
      ... = 160 := by norm_num
  rw [h_C, h_D] in h₅
  calc
    num_C = 160 + 10 := by rw [← h₅]
    ... = 170 := by norm_num

end class_C_has_170_people_l795_795655


namespace booknote_unique_elements_l795_795536

def booknote_string : String := "booknote"
def booknote_set : Finset Char := { 'b', 'o', 'k', 'n', 't', 'e' }

theorem booknote_unique_elements : booknote_set.card = 6 :=
by
  sorry

end booknote_unique_elements_l795_795536


namespace probability_of_red_ball_l795_795433

theorem probability_of_red_ball :
    let total_balls := 3 + 5 + 4
    let red_balls := 3
    (red_balls / total_balls : ℚ) = 1 / 4 :=
by
    let total_balls := 3 + 5 + 4
    let red_balls := 3
    show (red_balls / total_balls : ℚ) = 1 / 4
    sorry

end probability_of_red_ball_l795_795433


namespace Jones_clothing_count_l795_795763

theorem Jones_clothing_count :
  let (pants_count := 40)
      (shirts_per_pants := 6)
      (ties_per_pants := 5)
      (socks_per_shirt := 3)
      (shirts_count := pants_count * shirts_per_pants)
      (ties_count := pants_count * ties_per_pants)
      (socks_count := shirts_count * socks_per_shirt)
      (total_clothing := pants_count + shirts_count + ties_count + socks_count)
  in  total_clothing = 1200 :=
by
  sorry

end Jones_clothing_count_l795_795763


namespace range_a_l795_795933

theorem range_a (a : ℝ) : (∀ x, x > 0 → x^2 - a * x + 1 > 0) → -2 < a ∧ a < 2 := by
  sorry

end range_a_l795_795933


namespace christine_needs_32_tablespoons_l795_795650

-- Define the conditions
def tablespoons_per_egg_white : ℕ := 2
def egg_whites_per_cake : ℕ := 8
def number_of_cakes : ℕ := 2

-- Define the calculation for total tablespoons of aquafaba needed
def total_tbs_aquafaba : ℕ :=
  tablespoons_per_egg_white * (egg_whites_per_cake * number_of_cakes)

-- The theorem to prove
theorem christine_needs_32_tablespoons :
  total_tbs_aquafaba = 32 :=
by 
  -- Placeholder for proof, as proof steps are not required
  sorry

end christine_needs_32_tablespoons_l795_795650


namespace perimeter_C_correct_l795_795903

variables (x y : ℕ)

def perimeter_A (x y : ℕ) := 6 * x + 2 * y
def perimeter_B (x y : ℕ) := 4 * x + 6 * y
def perimeter_C (x y : ℕ) := 2 * x + 6 * y

theorem perimeter_C_correct (x y : ℕ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  2 * x + 6 * y = 40 :=
sorry

end perimeter_C_correct_l795_795903


namespace congruent_triangles_have_equal_perimeters_and_areas_l795_795584

-- Definitions based on the conditions
structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (A B C : ℝ) -- angles of the triangle

def congruent_triangles (Δ1 Δ2 : Triangle) : Prop :=
  Δ1.a = Δ2.a ∧ Δ1.b = Δ2.b ∧ Δ1.c = Δ2.c ∧
  Δ1.A = Δ2.A ∧ Δ1.B = Δ2.B ∧ Δ1.C = Δ2.C

-- perimeters and areas (assuming some function calc_perimeter and calc_area for simplicity)
def perimeter (Δ : Triangle) : ℝ := Δ.a + Δ.b + Δ.c
def area (Δ : Triangle) : ℝ := sorry -- implement area calculation, e.g., using Heron's formula

-- Statement to be proved
theorem congruent_triangles_have_equal_perimeters_and_areas (Δ1 Δ2 : Triangle) :
  congruent_triangles Δ1 Δ2 →
  perimeter Δ1 = perimeter Δ2 ∧ area Δ1 = area Δ2 :=
sorry

end congruent_triangles_have_equal_perimeters_and_areas_l795_795584


namespace aquafaba_needed_for_cakes_l795_795646

def tablespoons_of_aquafaba_for_egg_whites (n_egg_whites : ℕ) : ℕ :=
  2 * n_egg_whites

def total_egg_whites_needed (cakes : ℕ) (egg_whites_per_cake : ℕ) : ℕ :=
  cakes * egg_whites_per_cake

theorem aquafaba_needed_for_cakes (cakes : ℕ) (egg_whites_per_cake : ℕ) :
  tablespoons_of_aquafaba_for_egg_whites (total_egg_whites_needed cakes egg_whites_per_cake) = 32 :=
by
  have h1 : cakes = 2 := sorry
  have h2 : egg_whites_per_cake = 8 := sorry
  sorry

end aquafaba_needed_for_cakes_l795_795646


namespace relationships_involving_correlation_l795_795534

-- Define the relationships as hypotheses
def relationship1 : Prop := "The relationship between a person's age and their wealth"
def relationship2 : Prop := "The relationship between a point on a curve and its coordinates"
def relationship3 : Prop := "The relationship between apple production and climate"
def relationship4 : Prop := "The relationship between the diameter of the cross-section and the height of the same type of tree in a forest"
def relationship5 : Prop := "The relationship between a student and their student ID number"

-- Define a predicate to indicate whether a relationship involves correlation
def involves_correlation (r : Prop) : Prop := r = relationship1 ∨ r = relationship3 ∨ r = relationship4

-- The theorem states that relationships ①③④ involve correlation
theorem relationships_involving_correlation :
  involves_correlation relationship1 ∧ involves_correlation relationship3 ∧ involves_correlation relationship4 :=
by
  -- Assuming our conditions are true, we need to prove the relationships involving correlation
  sorry

end relationships_involving_correlation_l795_795534


namespace shirt_cost_l795_795429

theorem shirt_cost (S : ℝ) (hats_cost jeans_cost total_cost : ℝ)
  (h_hats : hats_cost = 4)
  (h_jeans : jeans_cost = 10)
  (h_total : total_cost = 51)
  (h_eq : 3 * S + 2 * jeans_cost + 4 * hats_cost = total_cost) :
  S = 5 :=
by
  -- The main proof will be provided here
  sorry

end shirt_cost_l795_795429


namespace perimeter_C_is_40_l795_795888

noncomputable def perimeter_of_figure_C (x y : ℝ) : ℝ :=
  2 * x + 6 * y

theorem perimeter_C_is_40 (x y : ℝ) (h1 : 6 * x + 2 * y = 56) (h2 : 4 * x + 6 * y = 56) :
  perimeter_of_figure_C x y = 40 :=
by
  -- Define initial conditions
  have eq1 : 3 * x + y = 28, by { rw [mul_assoc, mul_comm 3 x, add_assoc], exact (eq.div h1 2) }
  have eq2 : 2 * x + 3 * y = 28, by { rw [mul_assoc, mul_comm 2 x, add_assoc], exact (eq.div h2 2) }
  -- Assume the solutions are obtained from here
  have sol_x : x = 8, by sorry
  have sol_y : y = 4, by sorry
  -- Calculate the perimeter of figure C
  rw [perimeter_of_figure_C, sol_x, sol_y]
  norm_num
  trivial

-- Test case to ensure the code builds successfully
#eval perimeter_of_figure_C 8 4  -- Expected output: 40

end perimeter_C_is_40_l795_795888


namespace tan_phi_sqrt3_l795_795714

theorem tan_phi_sqrt3 (phi : ℝ) 
  (h1 : cos(π / 2 + phi) = - (sqrt 3) / 2)
  (h2 : abs phi < π / 2) : 
  tan phi = sqrt 3 :=
sorry

end tan_phi_sqrt3_l795_795714


namespace polygon_side_count_l795_795404

theorem polygon_side_count (n : ℕ) (h : n - 3 ≤ 5) : n = 8 :=
by {
  sorry
}

end polygon_side_count_l795_795404


namespace ribbon_left_l795_795852

theorem ribbon_left (gifts : ℕ) (ribbon_per_gift Tom_ribbon_total : ℝ) (h1 : gifts = 8) (h2 : ribbon_per_gift = 1.5) (h3 : Tom_ribbon_total = 15) : Tom_ribbon_total - (gifts * ribbon_per_gift) = 3 := 
by
  sorry

end ribbon_left_l795_795852


namespace eccentricity_range_l795_795727

theorem eccentricity_range (a b c e : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : ∀ P : ℝ × ℝ, P.1^2/a^2 + P.2^2/b^2 = 1 → 
    let F1 := (-c, 0); 
    let F2 := (c, 0);
    (P.1 + c) * (P.1 - c) + (P.2)^2 = c^2) :
    (√3) / 3 ≤ e ∧ e ≤ √2 / 2 :=
sorry

end eccentricity_range_l795_795727


namespace simplify_expression_l795_795512

variable (y : ℝ)

theorem simplify_expression : 
  3 * y - 5 * y^2 + 2 + (8 - 5 * y + 2 * y^2) = -3 * y^2 - 2 * y + 10 := 
by
  sorry

end simplify_expression_l795_795512


namespace gcd_187_253_l795_795687

/-- Define the gcd function --/
def gcd (a b : ℕ) : ℕ :=
  if b = 0 then a else gcd b (a % b)

/-- Problem statement: Proving the GCD of 187 and 253 is 11 --/
theorem gcd_187_253 : gcd 187 253 = 11 := by
  sorry

end gcd_187_253_l795_795687


namespace number_of_complex_numbers_l795_795294

def isReal (z : ℂ) : Prop :=
  ∃ (θ : ℝ), z = complex.exp (θ * complex.I)

theorem number_of_complex_numbers (n : ℕ) (hn1 : n = 8!) (hn2 : n = 7!) :
  ∑ z in (unitCircle : Set ℂ), z ^ hn1 - z ^ hn2 ∈ ℝ = 20100 := sorry

end number_of_complex_numbers_l795_795294


namespace mean_value_of_quadrilateral_angles_l795_795999

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l795_795999


namespace additional_money_needed_l795_795484

theorem additional_money_needed :
  let total_budget := 500
  let budget_dresses := 300
  let budget_shoes := 150
  let budget_accessories := 50
  let extra_fraction := 2 / 5
  let discount_rate := 0.15
  let total_without_discount := 
    budget_dresses * (1 + extra_fraction) +
    budget_shoes * (1 + extra_fraction) +
    budget_accessories * (1 + extra_fraction)
  let discounted_total := total_without_discount * (1 - discount_rate)
  discounted_total > total_budget :=
sorry

end additional_money_needed_l795_795484


namespace count_triples_xyz_l795_795697

theorem count_triples_xyz :
  let sgn := (sign : ℝ → ℝ)
  let solutions := 
    {xyz : ℝ × ℝ × ℝ |
      let (x, y, z) := xyz in
      x = 2020 - 2021 * sgn (y + z) ∧
      y = 2020 - 2021 * sgn (x + z) ∧
      z = 2020 - 2021 * sgn (x + y)} in
  solutions.to_finset.card = 3 :=
by
  let sgn := (sign : ℝ → ℝ)
  let solutions := 
    {xyz : ℝ × ℝ × ℝ |
      let (x, y, z) := xyz in
      x = 2020 - 2021 * sgn (y + z) ∧
      y = 2020 - 2021 * sgn (x + z) ∧
      z = 2020 - 2021 * sgn (x + y)}
  have solutions_list : list (ℝ × ℝ × ℝ) := [(-1, -1, 4041), (-1, 4041, -1), (4041, -1, -1)]
  have solutions_set := solutions_list.to_finset
  have : solutions_set = solutions.to_finset := sorry
  rw this
  exact finset.card_of_list solutions_list

end count_triples_xyz_l795_795697


namespace solution_ellipse_l795_795815

variables {a b c : ℝ} (A B : RealPoint) (l : ℝ → ℝ) (C : RealCurve)

def is_foci_of_ellipse (F₁ F₂ : RealPoint) (a b : ℝ) : Prop :=
  ∃ F1 : ℝ×ℝ, F1 = (-c, 0) ∧ ∃ F2 : ℝ×ℝ, F2 = (c, 0) 
  ∧ F1 = F₁ ∧ F2 = F₂ ∧ c^2 = a^2 - b^2

axiom ellipse_conditions :
  ∀ (a b : ℝ), a > b ∧ b > 0 →
  ∃ (C : RealCurve), C = { p : ℝ×ℝ | p.1^2 / a^2 + p.2^2 / b^2 = 1 }

axiom line_through_focus :
  ∀ (c : ℝ), ∃ (l : ℝ → ℝ), l(x) = sqrt(3) * (x - c)

axiom distance_focus_to_line :
  ∀ (c : ℝ), (2 * sqrt(3) * c) / 2 = 2 → c = 2 / sqrt(3)

axiom distance_AB :
  A = (- sqrt 3 + 2, -1) ∧ B = (sqrt 3 + 2, 1) → |A - B|^2 = 12

theorem solution_ellipse :
  ∀ (a b c : ℝ) (F₁ F₂ : RealPoint) (l : ℝ → ℝ),
  is_foci_of_ellipse F₁ F₂ a b 
  ∧ ellipse_conditions a b
  ∧ line_through_focus c
  ∧ distance_focus_to_line c
  ∧ distance_AB A B
  ∧ (|A - B| = 2√3) →
  2 * c = 2 * (2 / sqrt 3) ∧ a^2 = 9 ∧ b^2 = 5 :=
begin
  sorry
end

end solution_ellipse_l795_795815


namespace rotation_invariant_function_l795_795463

theorem rotation_invariant_function (D : Set ℝ) (f : ℝ → ℝ) (hD : 1 ∈ D) 
  (h : ∀ x ∈ D, ∃ y ∈ D, (f x, f y) ∈ (fun (x y : ℝ) =>
    (- (1 / 2) * x + (sqrt 3 / 2) * y, (sqrt 3 / 2) * x + (1 / 2) * y)) '' D) :
  f 1 = sqrt 3 / 2 :=
sorry

end rotation_invariant_function_l795_795463


namespace nearest_integer_to_power_l795_795163

theorem nearest_integer_to_power (a b : ℝ) (h1 : a = 3) (h2 : b = sqrt 2) : 
  abs ((a + b)^6 - 3707) < 0.5 :=
by
  sorry

end nearest_integer_to_power_l795_795163


namespace min_AC_plus_BD_l795_795533

theorem min_AC_plus_BD (k : ℝ) (h : k ≠ 0) :
  (8 + 8 / k^2) + (8 + 2 * k^2) ≥ 24 :=
by
  sorry -- skipping the proof

end min_AC_plus_BD_l795_795533


namespace a1_value_a_seq_formula_l795_795834

open Nat

-- Define the sequences and the given relationship
def a_seq : ℕ → ℤ := sorry  -- Placeholder for the sequence definition
def S (n : ℕ) : ℤ := (Finset.range n).sum (λ i, a_seq i)
def T (n : ℕ) : ℤ := (Finset.range n).sum S

axiom T_condition : ∀ n : ℕ, T n = 2 * S n - n^2

-- Prove the specific value for a_1
theorem a1_value : a_seq 0 = 1 := sorry

-- Prove the general formula for the sequence a_n
theorem a_seq_formula : ∀ n : ℕ, a_seq (n + 1) = 3 * 2^n - 2 := sorry

end a1_value_a_seq_formula_l795_795834


namespace sum_abcd_l795_795818

variable (a b c d x : ℝ)

axiom eq1 : a + 2 = x
axiom eq2 : b + 3 = x
axiom eq3 : c + 4 = x
axiom eq4 : d + 5 = x
axiom eq5 : a + b + c + d + 10 = x

theorem sum_abcd : a + b + c + d = -26 / 3 :=
by
  -- We state the condition given in the problem
  sorry

end sum_abcd_l795_795818


namespace compare_sequences_l795_795833

def a_seq : ℕ → ℕ 
| 0     := 0
| (n+1) := (a_seq n)^2 + 3

def b_seq : ℕ → ℕ
| 0     := 0
| (n+1) := (b_seq n)^2 + 2^(n+1)

theorem compare_sequences : b_seq 2003 < a_seq 2003 :=
sorry

end compare_sequences_l795_795833


namespace cyclic_quadrilateral_non_similar_triangles_l795_795412

variable (A B C G D E F: Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq G] [DecidableEq D] [DecidableEq E] [DecidableEq F]

-- Let D, E, F be the midpoints of BC, AC, AB respectively
def midpoint (a b: Type) : Type := sorry
def D := midpoint B C
def E := midpoint A C
def F := midpoint A B

-- Let G be the centroid of triangle ABC
def centroid (a b c: Type) : Type := sorry
def G := centroid A B C

-- Function to check if AEGF is a cyclic quadrilateral
def isCyclic (a e g f: Type) : Prop := sorry

-- Main theorem
theorem cyclic_quadrilateral_non_similar_triangles (angle_BAC: ℝ) :
  ∃ n: ℕ, (n = 1 ∧ angle_BAC ≤ 60) ∨ (n = 0 ∧ angle_BAC > 60) :=
by
  sorry

end cyclic_quadrilateral_non_similar_triangles_l795_795412


namespace projection_equidistant_from_origin_l795_795869

theorem projection_equidistant_from_origin
  (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  let K := (a, 1 / a)
  let L := (-a, -1 / a)
  let M := (b, 1 / b)
  let P := (c, 1 / c)
  let O := (0, 0)
  let d1 := dist (proj (line_through M K) P) O
  let d2 := dist (proj (line_through M L) P) O
  in d1 = d2 :=
sorry

end projection_equidistant_from_origin_l795_795869


namespace complement_is_256_l795_795324

open Set

-- Define the sets I and A
def I : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 4}

-- Prove that the complement of A with respect to I is {2, 5, 6}
theorem complement_is_256 : I \ A = {2, 5, 6} := 
by
  sorry

end complement_is_256_l795_795324


namespace no_prime_power_palindrome_l795_795849

-- Definitions and conditions from the problem
def is_palindrome (n : ℕ) : Prop := 
  let s := n.to_string
  s = s.reverse

def eventually_periodic_mod (p : ℕ) : Prop :=
  ∃T : ℕ, ∀n, p^(n + T) % 10 = p^n % 10

theorem no_prime_power_palindrome (p : ℕ) (hp : Nat.prime p) : ¬ (∀ n : ℕ, is_palindrome (p^n)) :=
by
  sorry

end no_prime_power_palindrome_l795_795849


namespace harmony_implication_at_least_N_plus_1_zero_l795_795342

noncomputable def is_harmony (A B : ℕ → ℕ) (i : ℕ) : Prop :=
  A i = (1 / (2 * B i + 1)) * (Finset.range (2 * B i + 1)).sum (fun s => A (i + s - B i))

theorem harmony_implication_at_least_N_plus_1_zero {N : ℕ} (A B : ℕ → ℕ)
  (hN : N ≥ 2) 
  (h_nonneg_A : ∀ i, 0 ≤ A i)
  (h_nonneg_B : ∀ i, 0 ≤ B i)
  (h_periodic_A : ∀ i, A i = A ((i % N) + 1))
  (h_periodic_B : ∀ i, B i = B ((i % N) + 1))
  (h_harmony_AB : ∀ i, is_harmony A B i)
  (h_harmony_BA : ∀ i, is_harmony B A i)
  (h_not_constant_A : ¬ ∀ i j, A i = A j)
  (h_not_constant_B : ¬ ∀ i j, B i = B j) :
  Finset.card (Finset.filter (fun i => A i = 0 ∨ B i = 0) (Finset.range (N * 2))) ≥ N + 1 := by
  sorry

end harmony_implication_at_least_N_plus_1_zero_l795_795342


namespace octal_to_decimal_376_l795_795634

theorem octal_to_decimal_376 :
  let n := 376
  let d0 := n % 10  -- rightmost digit (6)
  let d1 := (n / 10) % 10  -- middle digit (7)
  let d2 := n / 100  -- leftmost digit (3)
  d0 * 8^0 + d1 * 8^1 + d2 * 8^2 = 254 :=
by
  let n := 376
  let d0 := n % 10  -- rightmost digit (6)
  let d1 := (n / 10) % 10  -- middle digit (7)
  let d2 := n / 100  -- leftmost digit (3)
  have h0 : d0 = 6 := rfl
  have h1 : d1 = 7 := rfl
  have h2 : d2 = 3 := rfl
  have eqn : d0 * 8^0 + d1 * 8^1 + d2 * 8^2 = 6 * 1 + 7 * 8 + 3 * 64 := by
    rw [h0, h1, h2]
    sorry
  show 6 * 1 + 7 * 8 + 3 * 64 = 254 from sorry

end octal_to_decimal_376_l795_795634


namespace max_fruits_is_15_l795_795664

def maxFruits (a m p : ℕ) : Prop :=
  3 * a + 4 * m + 5 * p = 50 ∧ a ≥ 1 ∧ m ≥ 1 ∧ p ≥ 1

theorem max_fruits_is_15 : ∃ a m p : ℕ, maxFruits a m p ∧ a + m + p = 15 := 
  sorry

end max_fruits_is_15_l795_795664


namespace max_area_rect_l795_795541

theorem max_area_rect (x y : ℝ) (h_perimeter : 2 * x + 2 * y = 40) : 
  x * y ≤ 100 :=
by
  sorry

end max_area_rect_l795_795541


namespace correct_options_l795_795078

def vectors_satisfy_conditions (a b : Vector ℝ) : Prop :=
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥a - 3 • b∥ = Real.sqrt 13

theorem correct_options (a b : Vector ℝ)
  (h : vectors_satisfy_conditions a b) :
  (∥a - b∥ = Real.sqrt 3) ∧ (∥a + 3 • b∥ = Real.sqrt 7) :=
by sorry

end correct_options_l795_795078


namespace kitchen_area_l795_795498

-- Definitions based on conditions
def total_area : ℕ := 1110
def bedroom_area : ℕ := 11 * 11
def num_bedrooms : ℕ := 4
def bathroom_area : ℕ := 6 * 8
def num_bathrooms : ℕ := 2

-- Theorem to prove
theorem kitchen_area :
  let total_bedroom_area := bedroom_area * num_bedrooms,
      total_bathroom_area := bathroom_area * num_bathrooms,
      remaining_area := total_area - (total_bedroom_area + total_bathroom_area)
  in remaining_area / 2 = 265 := 
by
  sorry

end kitchen_area_l795_795498


namespace median_and_mode_of_successful_shots_l795_795418

theorem median_and_mode_of_successful_shots :
  let shots := [3, 6, 4, 6, 4, 3, 6, 5, 7]
  let sorted_shots := [3, 3, 4, 4, 5, 6, 6, 6, 7]
  let median := sorted_shots[4]  -- 4 is the index for the 5th element (0-based indexing)
  let mode := 6  -- determined by the number that appears most frequently
  median = 5 ∧ mode = 6 :=
by
  sorry

end median_and_mode_of_successful_shots_l795_795418


namespace inequality_solution_l795_795738

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 3)

theorem inequality_solution :
  -3 < x ∧ x < -1 ↔ f (x^2 - 3) < f (x - 1) :=
sorry

end inequality_solution_l795_795738


namespace perimeter_C_l795_795913

theorem perimeter_C : 
  ∀ {x y : ℕ}, 
    (6 * x + 2 * y = 56) → (4 * x + 6 * y = 56) → 
    (2 * x + 6 * y = 40) :=
by
  intros x y h1 h2
  sorry

end perimeter_C_l795_795913


namespace polyhedron_euler_l795_795661

theorem polyhedron_euler {S A F : ℕ} (h : ¬ ∃ p q r s : ℝ³, coplanar p q r s) : S + F = A + 2 :=
sorry

end polyhedron_euler_l795_795661


namespace concert_cost_l795_795507

noncomputable def ticket_price : ℝ := 50.0
noncomputable def processing_fee_rate : ℝ := 0.15
noncomputable def parking_fee : ℝ := 10.0
noncomputable def entrance_fee : ℝ := 5.0
def number_of_people : ℕ := 2

noncomputable def processing_fee_per_ticket : ℝ := processing_fee_rate * ticket_price
noncomputable def total_cost_per_ticket : ℝ := ticket_price + processing_fee_per_ticket
noncomputable def total_ticket_cost : ℝ := number_of_people * total_cost_per_ticket
noncomputable def total_cost_with_parking : ℝ := total_ticket_cost + parking_fee
noncomputable def total_entrance_fee : ℝ := number_of_people * entrance_fee
noncomputable def total_cost : ℝ := total_cost_with_parking + total_entrance_fee

theorem concert_cost : total_cost = 135.0 := by
  sorry

end concert_cost_l795_795507


namespace area_smile_l795_795663

-- Definitions based on problem conditions
def radius : ℝ := 3
def semicircle_area (r : ℝ) : ℝ := (1 / 2) * Real.pi * r^2
def sector_area (r : ℝ) (angle : ℝ) : ℝ := (1 / 2) * r^2 * angle
def triangle_area (b : ℝ) (h : ℝ) : ℝ := (1 / 2) * b * h

-- Theorem stating the area of the shaded region "smile" AEFBDA
theorem area_smile :
  let r := radius,
      r_ext := 6, -- Radius of extended sectors
      angle := Real.pi / 4, -- 45 degrees in radians
      sector1_area := sector_area r_ext angle,
      sector2_area := sector_area r_ext angle,
      tri_area := triangle_area (2 * r) r, -- Triangle ABD area
      semi_area := semicircle_area r, -- Semicircle ABD area
      sector_def_area := sector_area r Real.pi -- Sector DEF area
  in sector1_area + sector2_area - tri_area + sector_def_area - semi_area = 4.5 * Real.pi :=
by
  sorry

end area_smile_l795_795663


namespace largest_common_in_range_l795_795635

-- Definitions for the problem's conditions
def first_seq (n : ℕ) : ℕ := 3 + 8 * n
def second_seq (m : ℕ) : ℕ := 5 + 9 * m

-- Statement of the theorem we are proving
theorem largest_common_in_range : 
  ∃ n m : ℕ, first_seq n = second_seq m ∧ 1 ≤ first_seq n ∧ first_seq n ≤ 200 ∧ first_seq n = 131 := by
  sorry

end largest_common_in_range_l795_795635


namespace similar_triangles_incorrect_area_ratio_l795_795958

theorem similar_triangles_incorrect_area_ratio (T1 T2 : Type) [Triangle T1] [Triangle T2] :
  (similar T1 T2) →
  ∀ (r : ℝ), similarity_ratio T1 T2 r →
  ¬(ratio_of_areas T1 T2 = r) := 
by
  sorry

end similar_triangles_incorrect_area_ratio_l795_795958


namespace general_term_formula_inequality_for_Sn_l795_795192

theorem general_term_formula (T : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) (hn : n > 0) 
  (hT : T n = 1 - a n) : a n = n / (n + 1) := 
sorry

theorem inequality_for_Sn (T : ℕ → ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) (hn : n > 0)
  (hT : ∀ i, 1 ≤ i ∧ i ≤ n → T i = 1 - a i)
  (hS : S n = ∑ i in finset.range (n+1), (T i) ^ 2) :
  a (n + 1) - 1 / 2 < S n ∧ S n < a (n + 1) - 1 / 3 := 
sorry

end general_term_formula_inequality_for_Sn_l795_795192


namespace unique_acute_triangulation_l795_795847

-- Definitions for the proof problem
def is_convex (polygon : Type) : Prop := sorry
def is_acute_triangle (triangle : Type) : Prop := sorry
def is_triangulation (polygon : Type) (triangulation : List Type) : Prop := sorry
def is_acute_triangulation (polygon : Type) (triangulation : List Type) : Prop :=
  is_triangulation polygon triangulation ∧ ∀ triangle ∈ triangulation, is_acute_triangle triangle

-- Proposition to be proved
theorem unique_acute_triangulation (n : ℕ) (polygon : Type) 
  (h₁ : is_convex polygon) (h₂ : n ≥ 3) :
  ∃! triangulation : List Type, is_acute_triangulation polygon triangulation := 
sorry

end unique_acute_triangulation_l795_795847


namespace find_inverse_sum_l795_795805

def f (x : ℝ) : ℝ := if x < 10 then x - 4 else 3 * x + 5

theorem find_inverse_sum : f⁻¹' {6} + f⁻¹' {32} = {9 + 1/3} := by
  sorry

end find_inverse_sum_l795_795805


namespace acid_solution_problem_l795_795002

theorem acid_solution_problem (w s : ℝ) (w_eq : w = 40) (s_eq : s = 200 / 9) :
  (0.25 * 40 + 0.1 * s) / (40 + w + s) = 0.15 :=
by 
  -- Immediate substitution of given values 
  have : w = 40 := w_eq,
  have : s = 200 / 9 := s_eq,
  -- Skipping the actual calculations and setting up differential steps
  sorry

end acid_solution_problem_l795_795002


namespace angle_AOD_possible_values_l795_795522

theorem angle_AOD_possible_values (x : ℝ) 
  (h1 : ∠AOB = 3 * x) 
  (h2 : ∠BOC = 3 * x) 
  (h3 : ∠COD = 3 * x) 
  (h4 : ∠AOD = x) 
  (distinct_rays : OA ≠ OB ∧ OB ≠ OC ∧ OC ≠ OD ∧ OD ≠ OA) :
  x = 36 ∨ x = 45 :=
by
  sorry

end angle_AOD_possible_values_l795_795522


namespace range_of_b_l795_795406

noncomputable def f (b x : ℝ) := log x + (x - b)^2

theorem range_of_b (b : ℝ) :
  (∃ I : set ℝ, I ⊆ set.Icc (1/2) 2 ∧ ∀ x ∈ I, deriv (f b) x > 0) ↔ b < 9/4 :=
sorry

end range_of_b_l795_795406


namespace count_of_triples_is_three_l795_795693

noncomputable def count_valid_triples : ℕ := 
  nat.count
    (λ (x y z : ℝ), 
       x = 2020 - 2021 * (sign (y + z)) ∧
       y = 2020 - 2021 * (sign (x + z)) ∧
       z = 2020 - 2021 * (sign (x + y)))
    [(4041, -1, -1), (-1, 4041, -1), (-1, -1, 4041)]

theorem count_of_triples_is_three : count_valid_triples = 3 := sorry

end count_of_triples_is_three_l795_795693


namespace axis_of_symmetry_parabola_l795_795525

theorem axis_of_symmetry_parabola {a b c : ℝ} 
(intersect1 : (-4 : ℝ), (0 : ℝ)) 
(intersect2 : (6 : ℝ), (0 : ℝ)) :
  ∃ k : ℝ, k = 1 ∧
    ∀ x y : ℝ, y = a * x^2 + b * x + c → k = (x + (-(4:ℝ)) / 2) ∨ k = (x + (6:ℝ)) / 2 :=
begin
  sorry
end

end axis_of_symmetry_parabola_l795_795525


namespace aquafaba_needed_for_cakes_l795_795647

def tablespoons_of_aquafaba_for_egg_whites (n_egg_whites : ℕ) : ℕ :=
  2 * n_egg_whites

def total_egg_whites_needed (cakes : ℕ) (egg_whites_per_cake : ℕ) : ℕ :=
  cakes * egg_whites_per_cake

theorem aquafaba_needed_for_cakes (cakes : ℕ) (egg_whites_per_cake : ℕ) :
  tablespoons_of_aquafaba_for_egg_whites (total_egg_whites_needed cakes egg_whites_per_cake) = 32 :=
by
  have h1 : cakes = 2 := sorry
  have h2 : egg_whites_per_cake = 8 := sorry
  sorry

end aquafaba_needed_for_cakes_l795_795647


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795993

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795993


namespace range_func_l795_795934

noncomputable def func (x : ℝ) : ℝ := 2^(-|x|)

theorem range_func : set.Ioo 0 1 ∪ set.Icc 1 1 = set.range func := sorry

end range_func_l795_795934


namespace instantaneous_velocity_at_t_2_l795_795138

theorem instantaneous_velocity_at_t_2 
  (t : ℝ) (x1 y1 x2 y2: ℝ) : 
  (t = 2) → 
  (x1 = 0) → (y1 = 4) → 
  (x2 = 12) → (y2 = -2) → 
  ((y2 - y1) / (x2 - x1) = -1 / 2) := 
by 
  intros ht hx1 hy1 hx2 hy2
  sorry

end instantaneous_velocity_at_t_2_l795_795138


namespace brian_total_video_length_l795_795238

theorem brian_total_video_length :
  let cat_length := 4
  let dog_length := 2 * cat_length
  let gorilla_length := cat_length ^ 2
  let elephant_length := cat_length + dog_length + gorilla_length
  let cat_dog_gorilla_elephant_sum := cat_length + dog_length + gorilla_length + elephant_length
  let penguin_length := cat_dog_gorilla_elephant_sum ^ 3
  let dolphin_length := cat_length + dog_length + gorilla_length + elephant_length + penguin_length
  let total_length := cat_length + dog_length + gorilla_length + elephant_length + penguin_length + dolphin_length
  total_length = 351344 := by
    sorry

end brian_total_video_length_l795_795238


namespace concert_cost_l795_795505

def ticket_cost : ℕ := 50
def number_of_people : ℕ := 2
def processing_fee_rate : ℝ := 0.15
def parking_fee : ℕ := 10
def per_person_entrance_fee : ℕ := 5

def total_cost : ℝ :=
  let tickets := (ticket_cost * number_of_people : ℕ)
  let processing_fee := tickets * processing_fee_rate
  let entrance_fee := per_person_entrance_fee * number_of_people
  (tickets : ℝ) + processing_fee + (parking_fee : ℝ) + (entrance_fee : ℝ)

theorem concert_cost :
  total_cost = 135 := by
  sorry

end concert_cost_l795_795505


namespace gcd_factorial_8_10_l795_795312

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_10 : Nat.gcd (factorial 8) (factorial 10) = 40320 :=
by
  -- these pre-evaluations help Lean understand the factorial values
  have fact_8 : factorial 8 = 40320 := by sorry
  have fact_10 : factorial 10 = 3628800 := by sorry
  rw [fact_8, fact_10]
  -- the actual proof gets skipped here
  sorry

end gcd_factorial_8_10_l795_795312


namespace gcd_factorials_l795_795306

open Nat

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorials (h : ∀ n, 0 < n → factorial n = n * factorial (n - 1)) :
  Nat.gcd (factorial 8) (factorial 10) = factorial 8 :=
sorry

end gcd_factorials_l795_795306


namespace first_tap_fill_time_l795_795201

theorem first_tap_fill_time (T : ℝ) (h1 : T > 0) (h2 : 12 > 0) 
  (h3 : 1/T - 1/12 = 1/12) : T = 6 :=
sorry

end first_tap_fill_time_l795_795201


namespace perimeter_of_C_l795_795928

theorem perimeter_of_C (x y : ℝ) 
  (h₁ : 6 * x + 2 * y = 56) 
  (h₂ : 4 * x + 6 * y = 56) : 
  2 * x + 6 * y = 40 :=
sorry

end perimeter_of_C_l795_795928


namespace meaningful_expression_range_l795_795019

theorem meaningful_expression_range (x : ℝ) : (∃ (y : ℝ), y = 5 / (x - 2)) ↔ x ≠ 2 := 
by
  sorry

end meaningful_expression_range_l795_795019


namespace new_boarders_joined_l795_795549

theorem new_boarders_joined (initial_boarders new_boarders initial_day_students total_boarders total_day_students: ℕ)
  (h1: initial_boarders = 60)
  (h2: initial_day_students = 150)
  (h3: total_boarders = initial_boarders + new_boarders)
  (h4: total_day_students = initial_day_students)
  (h5: 2 * initial_day_students = 5 * initial_boarders)
  (h6: 2 * total_boarders = total_day_students) :
  new_boarders = 15 :=
by
  sorry

end new_boarders_joined_l795_795549


namespace max_n_l795_795744

noncomputable def hyperbola_eqn := ∀ x y : ℝ, x^2 / 4 - y^2 = 1

structure Point :=
(x : ℝ)
(y : ℝ)

noncomputable def on_right_upper_part (P : Point) : Prop :=
2 ≤ P.x ∧ P.x ≤ 2 * Real.sqrt 5 ∧ 0 ≤ P.y

def common_diff_valid (d : ℝ) : Prop :=
1 / 5 < d ∧ d < Real.sqrt 5 / 5

def distance_to_focus (a : ℝ) (x : ℝ) : ℝ :=
(Real.sqrt 5 / 2) * x - 2

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) - a n = d

theorem max_n (a : ℕ → ℝ) (d : ℝ) :
  (∀ k : ℕ, ∃ P : Point, on_right_upper_part P ∧ a k = distance_to_focus (a k) P.x) →
  arithmetic_sequence a d →
  common_diff_valid d →
  2 ≤ a 1 / (Real.sqrt 5 / 2) - 2 ≤ (2 * Real.sqrt 5) →
  n ≤ 14 :=
sorry

end max_n_l795_795744


namespace CapeMay_more_than_twice_Daytona_l795_795644

def Daytona_sharks : ℕ := 12
def CapeMay_sharks : ℕ := 32

theorem CapeMay_more_than_twice_Daytona : CapeMay_sharks - 2 * Daytona_sharks = 8 := by
  sorry

end CapeMay_more_than_twice_Daytona_l795_795644


namespace shyne_plant_total_l795_795858

-- Conditions from the problem definition
def eggplants_per_packet := 14
def sunflowers_per_packet := 10
def tomatoes_per_packet := 16
def peas_per_packet := 20

def packets_eggplants := 4
def packets_sunflowers := 6
def packets_tomatoes := 5
def packets_peas := 7

def growth_capacity_spring := 0.7
def growth_capacity_summer := 0.8

-- Calculations for the total number of plants
def total_eggplants := packets_eggplants * eggplants_per_packet
def total_sunflowers := packets_sunflowers * sunflowers_per_packet
def total_tomatoes := packets_tomatoes * tomatoes_per_packet
def total_peas := packets_peas * peas_per_packet

def spring_eggplants := (total_eggplants * growth_capacity_spring).toNat
def spring_peas := (total_peas * growth_capacity_spring).toNat

def summer_sunflowers := (total_sunflowers * growth_capacity_summer).toNat
def summer_tomatoes := (total_tomatoes * growth_capacity_summer).toNat

def total_spring_plants := spring_eggplants + spring_peas
def total_summer_plants := summer_sunflowers + summer_tomatoes

def total_plants := total_spring_plants + total_summer_plants

-- Statement to prove
theorem shyne_plant_total : total_plants = 249 := 
by {
    sorry
}

end shyne_plant_total_l795_795858


namespace find_k_coordinates_transformation_l795_795482

theorem find_k_coordinates_transformation :
  ∃ k : ℝ, (let P := (5, 3) in
      let P' := (P.1 - 4, P.2 - 1) in
      P'.1 = 1 ∧ P'.2 = 2 ∧ P'.2 = k * P'.1 - 2) → k = 4 :=
by {
  use 4,
  sorry
}

end find_k_coordinates_transformation_l795_795482


namespace find_A_and_area_l795_795343

open Real

variable (A B C a b c : ℝ)
variable (h1 : 2 * sin A * cos B = 2 * sin C - sin B)
variable (h2 : a = 4 * sqrt 3)
variable (h3 : b + c = 8)
variable (h4 : a^2 = b^2 + c^2 - 2*b*c* cos A)

theorem find_A_and_area :
  A = π / 3 ∧ (1/2 * b * c * sin A = 4 * sqrt 3 / 3) :=
by
  sorry

end find_A_and_area_l795_795343


namespace determine_marbles_l795_795948

noncomputable def marbles_total (x : ℚ) := (4 * x + 2) + (2 * x) + (3 * x - 1)

theorem determine_marbles (x : ℚ) (h1 : marbles_total x = 47) :
  (4 * x + 2 = 202 / 9) ∧ (2 * x = 92 / 9) ∧ (3 * x - 1 = 129 / 9) :=
by
  sorry

end determine_marbles_l795_795948


namespace alpha_can_score_55_l795_795573

theorem alpha_can_score_55 (S : Finset ℕ) (hS : S = Finset.range 102 \ Finset.singleton 0) :
  ∃ (remaining : Finset ℕ), 
  (remaining.card = 2 ∧ 
   (∀ B_moves : Finset ℕ, (B_moves ⊆ S ∧ B_moves.card = 9) → 
    (∃ A_moves : Finset ℕ, (A_moves ⊆ (S \ B_moves) ∧ A_moves.card = 9 ∧ 
                           ∃ remaining_subset : Finset ℕ, 
                           (remaining_subset ⊆ (S \ (B_moves ∪ A_moves)) ∧ 
                           remaining_subset.card = 2 ∧ abs (remaining_subset.max' sorry - remaining_subset.min' sorry) = 55)))) :=
sorry

end alpha_can_score_55_l795_795573


namespace find_AD_l795_795798

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D : V) (a b : V)

def is_midpoint (D B C : V) := D = (B + C) / 2
def vector_AB (A B : V) := a
def vector_AC (A C : V) := b

theorem find_AD (A B C D : V) (a b : V)
  (h1 : is_midpoint D B C)
  (h2 : vector_AB A B = a)
  (h3 : vector_AC A C = b) :
  (D - A) = (1/2 : ℝ) • (a + b) :=
by
  sorry

end find_AD_l795_795798


namespace circle_diameter_eq_l795_795409

-- Definitions
def line (x y : ℝ) : Prop := 3 * x - 4 * y + 12 = 0
def point_A (x y : ℝ) : Prop := x = 0 ∧ y = 3
def point_B (x y : ℝ) : Prop := x = -4 ∧ y = 0
def midpoint_AB (x y : ℝ) : Prop := x = -2 ∧ y = 3 / 2 -- Midpoint of A(0,3) and B(-4,0)
def diameter_AB : ℝ := 5

-- The equation of the circle with diameter AB
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 3 * y = 0

-- The proof statement
theorem circle_diameter_eq :
  (∃ A B : ℝ × ℝ, point_A A.1 A.2 ∧ point_B B.1 B.2 ∧ 
                   midpoint_AB ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧ diameter_AB = 5) →
  (∀ x y : ℝ, circle_eq x y) :=
sorry

end circle_diameter_eq_l795_795409


namespace domain_H_eq_Icc_zero_one_l795_795353

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for the function f

theorem domain_H_eq_Icc_zero_one
  (hf : ∀ x, 0 ≤ x ∧ x ≤ 3 → ∃ y, f y = x):
  {x | (∃ y, f y = x ∧ 0 ≤ x ∧ x ≤ 3)} = set.Icc (0 : ℝ) 1 :=
by {
  sorry
}

end domain_H_eq_Icc_zero_one_l795_795353


namespace distance_between_vectors_is_d_l795_795563

variables (u1 u2 : ℝ^3)
variables (v1 v2 : ℝ^3)
variables (θ1 θ2 : ℝ)

noncomputable def vector1 := (3, -1, 2 : ℝ^3)
noncomputable def vector2 := (1, 2, 2 : ℝ^3)
noncomputable def u1_norm := ∥u1∥ = 1
noncomputable def u2_norm := ∥u2∥ = 1
noncomputable def angle_u1_v1 := u1 • vector1 = 2 * real.sqrt 3
noncomputable def angle_u1_v2 := u1 • vector2 = 3

theorem distance_between_vectors_is_d (d : ℝ) :
  u1_norm ∧ u2_norm ∧ angle_u1_v1 ∧ angle_u1_v2 → ∥u1 - u2∥ = d :=
sorry

end distance_between_vectors_is_d_l795_795563


namespace max_arithmetic_progressions_l795_795462

variable (A : Set ℝ)
variable (hA : A.card = 10)
variable (hDistinct : ∀ a b ∈ A, a ≠ b → a ≠ b)

theorem max_arithmetic_progressions (h_pos : ∀ a ∈ A, 0 < a) :
  (∃ S, S ⊆ { (a, b, c) | a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a < b ∧ b < c ∧ b - a = c - b } ∧ S.card = 20) :=
sorry

end max_arithmetic_progressions_l795_795462


namespace john_text_messages_per_day_l795_795803

theorem john_text_messages_per_day (m n : ℕ) (h1 : m = 20) (h2 : n = 245) : 
  m + n / 7 = 55 :=
by
  sorry

end john_text_messages_per_day_l795_795803


namespace AE_tangent_to_circumcircle_XYZ_l795_795062

-- Definitions for cyclic pentagon, parallel lines, midpoints and circumcircle
variables (A B C D E X Y Z : Type) 
variables [cyclic_pentagon A B C D E]
variables (BC_eq_DE : BC = DE) (parallel_AB_DE : AB ∥ DE)
variables [midpoint X B D] [midpoint Y C E] [midpoint Z A E]

theorem AE_tangent_to_circumcircle_XYZ  :
  tangent_to (circumcircle X Y Z) AE := sorry

end AE_tangent_to_circumcircle_XYZ_l795_795062


namespace circumcircle_intersection_and_simson_parallel_l795_795193

variable {A B C P O A₁ B₁ C₁ P₁ : Type*}

/-- Points A, B, C, and P lie on a circle centered at O, implies the sides of triangle A₁ B₁ C₁ are 
parallel to the lines PA, PB, and PC respectively, and through the vertices of triangle A₁ B₁ C₁, 
lines are parallel to the sides of triangle ABC. We need to prove that these lines intersect at a 
single point P₁ on the circumcircle of triangle A₁ B₁ C₁. Moreover, the Simson line of point P₁ is 
parallel to the line OP. -/
theorem circumcircle_intersection_and_simson_parallel (A B C P O A₁ B₁ C₁ P₁ : Type*)
  (h1 : Points A, B, C, and P lie on a circle centered at O)
  (h2 : The sides of triangle A₁ B₁ C₁ are parallel to the lines PA, PB, and PC respectively)
  (h3 : Through the vertices of triangle A₁ B₁ C₁, lines are drawn parallel to the sides of triangle ABC):
  (These lines intersect at a single point P₁ on the circumcircle of triangle A₁ B₁ C₁) ∧ 
  (The Simson line of point P₁ is parallel to the line OP) :=
begin
  sorry, -- The proof is omitted as per the instructions.
end

end circumcircle_intersection_and_simson_parallel_l795_795193


namespace evaluate_expression_l795_795941

theorem evaluate_expression :
  (∏ i in Finset.range 9, i) / ((Finset.sum (Finset.range 9) id) / 2) = 1120 :=
by
  -- Calculation of Numerator (∏ means product)
  have num_eq : (∏ i in Finset.range 9, if i = 0 then 1 else i) = 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1,
  { sorry }
  -- Calculation of Denominator
  have sum_eq : Finset.sum (Finset.range 9) id = 8 * 9 / 2,
  { sorry }
  -- Final fraction
  rw [num_eq, sum_eq],
  norm_num

end evaluate_expression_l795_795941


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795982

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795982


namespace correct_avg_var_l795_795414

theorem correct_avg_var
  (n : ℕ) (h_n : n = 48) 
  (original_avg : ℚ) (h_avg : original_avg = 70)
  (original_var : ℚ) (h_var : original_var = 75)
  (score_a_before correction_a correction_a_corrected : ℚ)
  (h_score_a : score_a_before = 50)
  (h_correction_a : correction_a = 80)
  (score_b_before correction_b correction_b_corrected : ℚ)
  (h_score_b : score_b_before = 100)
  (h_correction_b : correction_b = 70) :
  let corrected_avg := original_avg in
  let change_in_var := -25 in
  let corrected_var := original_var + change_in_var in
  corrected_avg = 70 ∧ corrected_var = 50 := by
  sorry

end correct_avg_var_l795_795414


namespace mean_value_of_quadrilateral_angles_l795_795976

theorem mean_value_of_quadrilateral_angles :
  let sum_of_angles := 360 in
  let number_of_angles := 4 in
  sum_of_angles / number_of_angles = 90 :=
by
  sorry

end mean_value_of_quadrilateral_angles_l795_795976


namespace cryptic_message_addition_l795_795508

/-- Proof problem for given addition equation (DEEP + POND + DEEP = DONE) in base 10
    where each letter represents a distinct digit.
-/
theorem cryptic_message_addition : 
  ∃ (P E D O N : ℕ), 
    P ≠ E ∧ E ≠ D ∧ D ≠ O ∧ O ≠ N ∧ N ≠ P ∧
    2 * P + D ≡ E [MOD 10] ∧
    2 * E + O ≡ N [MOD 10] ∧
    2 * E + N ≡ O [MOD 10] ∧
    D + P ≡ 0 [MOD 10] ∧
    let DEEP := 1000 * D + 100 * E + 10 * E + P,
        POND := 1000 * P + 100 * O + 10 * N + D,
        DONE := 1000 * D + 100 * O + 10 * N + E
    in DEEP + POND + DEEP = DONE :=
begin
  sorry
end

end cryptic_message_addition_l795_795508


namespace percentage_y_more_than_z_l795_795027

theorem percentage_y_more_than_z :
  ∀ (P y x k : ℕ),
    P = 200 →
    740 = x + y + P →
    x = (5 / 4) * y →
    y = P * (1 + k / 100) →
    k = 20 :=
by
  sorry

end percentage_y_more_than_z_l795_795027


namespace abs_diff_of_roots_eq_one_l795_795281

theorem abs_diff_of_roots_eq_one {p q : ℝ} (h₁ : p + q = 7) (h₂ : p * q = 12) : |p - q| = 1 := 
by 
  sorry

end abs_diff_of_roots_eq_one_l795_795281


namespace cos_alpha_beta_half_tan_alpha_beta_l795_795713

noncomputable def cos_val : ℂ := -2 * complex.sqrt 7 / 7
noncomputable def sin_val : ℂ := 1 / 2
noncomputable def alpha_set : set ℂ := {a : ℂ | π / 2 < a ∧ a < π}
noncomputable def beta_set : set ℂ := {b : ℂ | 0 < b ∧ b < π / 2}

theorem cos_alpha_beta_half (α β : ℂ) (hα : α ∈ alpha_set) (hβ : β ∈ beta_set)
  (h1 : complex.cos (α - β / 2) = cos_val) (h2 : complex.sin (α / 2 - β) = sin_val) :
  complex.cos ((α + β) / 2) = -complex.sqrt 21 / 14 :=
sorry

theorem tan_alpha_beta (α β : ℂ) (hα : α ∈ alpha_set) (hβ : β ∈ beta_set)
  (h1 : complex.cos (α - β / 2) = cos_val) (h2 : complex.sin (α / 2 - β) = sin_val) :
  complex.tan (α + β) = 5 * complex.sqrt 3 / 11 :=
sorry

end cos_alpha_beta_half_tan_alpha_beta_l795_795713


namespace integer_part_S_l795_795074

noncomputable def x : ℕ → ℝ
| 0     := 1 / 2
| (k+1) := x k + (x k) ^ 2

def S : ℝ := (∑ k in Finset.range 100, 1 / (x k + 1))

theorem integer_part_S : floor S = 1 := sorry

end integer_part_S_l795_795074


namespace number_of_complex_numbers_l795_795293

def isReal (z : ℂ) : Prop :=
  ∃ (θ : ℝ), z = complex.exp (θ * complex.I)

theorem number_of_complex_numbers (n : ℕ) (hn1 : n = 8!) (hn2 : n = 7!) :
  ∑ z in (unitCircle : Set ℂ), z ^ hn1 - z ^ hn2 ∈ ℝ = 20100 := sorry

end number_of_complex_numbers_l795_795293


namespace median_and_mode_of_successful_shots_l795_795419

theorem median_and_mode_of_successful_shots :
  let shots := [3, 6, 4, 6, 4, 3, 6, 5, 7]
  let sorted_shots := [3, 3, 4, 4, 5, 6, 6, 6, 7]
  let median := sorted_shots[4]  -- 4 is the index for the 5th element (0-based indexing)
  let mode := 6  -- determined by the number that appears most frequently
  median = 5 ∧ mode = 6 :=
by
  sorry

end median_and_mode_of_successful_shots_l795_795419


namespace g_one_fourth_l795_795128

noncomputable def g : ℝ → ℝ := sorry

theorem g_one_fourth :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ g x ∧ g x ≤ 1) ∧  -- g(x) is defined for 0 ≤ x ≤ 1
  g 0 = 0 ∧                                    -- g(0) = 0
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → g x ≤ g y) ∧ -- g is non-decreasing
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧ -- symmetric property
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 4) = g x / 2)   -- scaling property
  → g (1/4) = 1/2 :=
sorry

end g_one_fourth_l795_795128


namespace longest_train_length_l795_795568

theorem longest_train_length :
  ∀ (speedA : ℝ) (timeA : ℝ) (speedB : ℝ) (timeB : ℝ) (speedC : ℝ) (timeC : ℝ),
  speedA = 60 * (5 / 18) → timeA = 5 →
  speedB = 80 * (5 / 18) → timeB = 7 →
  speedC = 50 * (5 / 18) → timeC = 9 →
  speedB * timeB > speedA * timeA ∧ speedB * timeB > speedC * timeC ∧ speedB * timeB = 155.54 := by
  sorry

end longest_train_length_l795_795568


namespace angle_C_when_B_max_eq_l795_795351

-- Define the problem
theorem angle_C_when_B_max_eq:
  ∀ (A B C : ℝ),
    A > 0 ∧ A < π / 2 ∧ B > 0 ∧ B < π / 2 ∧ C > 0 ∧ C = π - (A + B) ∧
    (sin B / sin A = 2 * cos (A + B)) →
    C = 2 * π / 3 := 
sorry

end angle_C_when_B_max_eq_l795_795351


namespace molecular_weight_combined_ascorbic_citric_l795_795161

def molecular_weight_C := 12.01 -- g/mol for Carbon
def molecular_weight_H := 1.008 -- g/mol for Hydrogen
def molecular_weight_O := 16.00 -- g/mol for Oxygen

def molecules_Ascorbic : Nat := 7
def molecules_Citric : Nat := 5

def atoms_C_Ascorbic : Nat := 6
def atoms_H_Ascorbic : Nat := 8
def atoms_O_Ascorbic : Nat := 6

def atoms_C_Citric : Nat := 6
def atoms_H_Citric : Nat := 8
def atoms_O_Citric : Nat := 7

theorem molecular_weight_combined_ascorbic_citric :
  (molecules_Ascorbic * (atoms_C_Ascorbic * molecular_weight_C + atoms_H_Ascorbic * molecular_weight_H + atoms_O_Ascorbic * molecular_weight_O)) +
  (molecules_Citric * (atoms_C_Citric * molecular_weight_C + atoms_H_Citric * molecular_weight_H + atoms_O_Citric * molecular_weight_O)) = 2193.488 :=
sorry

end molecular_weight_combined_ascorbic_citric_l795_795161


namespace prop1_prop2_prop3_prop4_l795_795368

def f (x a : ℝ) := x^2 - |x + a|

theorem prop1 : ∀ a : ℝ, ¬ Monotone (λ x, f x a) := sorry

theorem prop2 : ∃ a : ℝ, ∀ x : ℝ, f x a = f (-x) a := sorry

theorem prop3 : ∀ a : ℝ, (∀ x : ℝ, f x a ≥ -5/4) → (∃ x : ℝ, f x a = -5/4) → a = -1 := sorry

theorem prop4 : ∃ a : ℝ, a < 0 ∧ (∃ x1 x2 x3 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) := sorry

end prop1_prop2_prop3_prop4_l795_795368


namespace code_conversion_l795_795597

def codeA : Nat := 1 -- A = 1
def codeB : Nat := 2 -- B = 2
def codeZ : Nat := 26 -- Z = 26

def ⟦ : Nat := 2
def ⊥ : Nat := 3
def ¬ : Nat := 4
def ⊓ : Nat := 8
def □ : Nat := 5

theorem code_conversion : (⟦ * 10000 + ⊥ * 1000 + ¬ * 100 + ⊓ * 10 + □) = 23485 := by
  sorry

end code_conversion_l795_795597


namespace cubic_expression_equals_two_l795_795395

theorem cubic_expression_equals_two (x : ℝ) (h : 2 * x ^ 2 - 3 * x - 2022 = 0) :
  2 * x ^ 3 - x ^ 2 - 2025 * x - 2020 = 2 :=
sorry

end cubic_expression_equals_two_l795_795395


namespace nate_age_when_ember_is_14_l795_795270

theorem nate_age_when_ember_is_14
  (nate_age : ℕ)
  (ember_age : ℕ)
  (h_half_age : ember_age = nate_age / 2)
  (h_nate_current_age : nate_age = 14) :
  nate_age + (14 - ember_age) = 21 :=
by
  sorry

end nate_age_when_ember_is_14_l795_795270


namespace order_variables_l795_795716

variables (a b c : ℝ)

def a := Real.log 2 / Real.log (1 / 2)
def b := Real.log (1 / 3) / Real.log (1 / 2)
def c := (1 / 2) ^ 0.3

theorem order_variables : a < c ∧ c < b := by
  have ha : a = a := rfl
  have hb : b = b := rfl
  have hc : c = c := rfl
  sorry

end order_variables_l795_795716


namespace increment_in_radius_l795_795179

theorem increment_in_radius (C1 C2 : ℝ) (hC1 : C1 = 50) (hC2 : C2 = 60) : 
  ((C2 / (2 * Real.pi)) - (C1 / (2 * Real.pi)) = (5 / Real.pi)) :=
by
  sorry

end increment_in_radius_l795_795179


namespace train_length_is_correct_l795_795216

noncomputable def train_length (speed_kmph : ℝ) (crossing_time_s : ℝ) (platform_length_m : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  let total_distance := speed_mps * crossing_time_s
  total_distance - platform_length_m

theorem train_length_is_correct :
  train_length 60 14.998800095992321 150 = 100 := by
  sorry

end train_length_is_correct_l795_795216


namespace sally_needs_8_napkins_l795_795105

theorem sally_needs_8_napkins :
  let tablecloth_length := 102
  let tablecloth_width := 54
  let napkin_length := 6
  let napkin_width := 7
  let total_material_needed := 5844
  let tablecloth_area := tablecloth_length * tablecloth_width
  let napkin_area := napkin_length * napkin_width
  let material_needed_for_napkins := total_material_needed - tablecloth_area
  let number_of_napkins := material_needed_for_napkins / napkin_area
  number_of_napkins = 8 :=
by
  sorry

end sally_needs_8_napkins_l795_795105


namespace trigonometric_identity_l795_795326

theorem trigonometric_identity (theta : ℝ) (h : Real.cos ((5 * Real.pi)/12 - theta) = 1/3) :
  Real.sin ((Real.pi)/12 + theta) = 1/3 :=
by
  sorry

end trigonometric_identity_l795_795326


namespace average_age_of_three_l795_795266

variable (Devin Eden Mom : ℕ)

-- Conditions
axiom h1 : Devin = 12
axiom h2 : Eden = 2 * Devin
axiom h3 : Mom = 2 * Eden

-- Theorem statement
theorem average_age_of_three :
  (Devin + Eden + Mom) / 3 = 28 :=
by
  rw [h1, h2, h3]
  calc
    (_ + _ + _) / 3 = 28 := sorry

end average_age_of_three_l795_795266


namespace ratio_of_speeds_l795_795570

/-- Define the time of meeting, t and distance D based on given conditions -/
variables (v1 v2 t D : ℝ)

noncomputable def time_of_meeting := t
noncomputable def distance := D
noncomputable def car1_speed := v1
noncomputable def car2_speed := v2

/-- Definition of the conditions provided in the problem statement -/
def meeting_conditions (v1 v2 t D : ℝ) : Prop :=
  (D = v1 * (t + 1)) ∧ (D = v2 * (t + 4))

/-- The ratio of the speeds of two cars meeting under given conditions is 2 -/
theorem ratio_of_speeds (v1 v2 t D : ℝ) (h : meeting_conditions v1 v2 t D) :
  v1 / v2 = 2 :=
by
  sorry

end ratio_of_speeds_l795_795570


namespace similar_triangles_CH_AH_B_CAB_l795_795813

noncomputable theory
open_locale classical

variables {A B C H_A H_B : Type*}
variables [inhabited A] [inhabited B] [inhabited C] [inhabited H_A] [inhabited H_B]
variables {triangle ABC : Type*}
variables (H_A_is_foot : is_foot_of_altitude A H_A ABC)
variables (H_B_is_foot : is_foot_of_altitude B H_B ABC)

theorem similar_triangles_CH_AH_B_CAB (ABC is_foot_of_altitude H_A A ABC) (is_foot_of_altitude H_B B ABC) :
  similar (△ CH_AH_B) (△ CAB) :=
sorry

end similar_triangles_CH_AH_B_CAB_l795_795813


namespace proof_of_number_of_triples_l795_795691

noncomputable def numberOfTriples : ℕ :=
  -- Definitions of the conditions
  let condition1 : ℝ × ℝ × ℝ → Prop := λ xyz, xyz.1 = 2020 - 2021 * Real.sign (xyz.2 + xyz.3)
  let condition2 : ℝ × ℝ × ℝ → Prop := λ xyz, xyz.2 = 2020 - 2021 * Real.sign (xyz.1 + xyz.3)
  let condition3 : ℝ × ℝ × ℝ → Prop := λ xyz, xyz.3 = 2020 - 2021 * Real.sign (xyz.1 + xyz.2)
  -- Counting the number of triples
  let triples : Finset (ℝ × ℝ × ℝ) := {(4041, -1, -1), (-1, 4041, -1), (-1, -1, 4041)}
  triples.card

theorem proof_of_number_of_triples :
  numberOfTriples = 3 :=
sorry  -- Proof goes here

end proof_of_number_of_triples_l795_795691


namespace gcd_factorial_8_10_l795_795309

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_10 : Nat.gcd (factorial 8) (factorial 10) = 40320 :=
by
  -- these pre-evaluations help Lean understand the factorial values
  have fact_8 : factorial 8 = 40320 := by sorry
  have fact_10 : factorial 10 = 3628800 := by sorry
  rw [fact_8, fact_10]
  -- the actual proof gets skipped here
  sorry

end gcd_factorial_8_10_l795_795309


namespace ends_with_14_zeros_l795_795457

theorem ends_with_14_zeros :
  let A := 2^7 * (7^14 + 1) + 2^6 * 7^11 * 10^2 + 2^6 * 7^7 * 10^4 + 2^4 * 7^3 * 10^6
  in A = 10^14 :=
by
  let A := 2^7 * (7^14 + 1) + 2^6 * 7^11 * 10^2 + 2^6 * 7^7 * 10^4 + 2^4 * 7^3 * 10^6
  sorry

end ends_with_14_zeros_l795_795457


namespace xyz_value_l795_795822

variables {x y z : ℂ}

theorem xyz_value (h1 : x * y + 2 * y = -8)
                  (h2 : y * z + 2 * z = -8)
                  (h3 : z * x + 2 * x = -8) :
  x * y * z = 32 :=
by
  sorry

end xyz_value_l795_795822


namespace mean_value_of_quadrilateral_angles_l795_795998

theorem mean_value_of_quadrilateral_angles : 
  ∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90 :=
by
  intro a b c d h
  sorry

end mean_value_of_quadrilateral_angles_l795_795998


namespace jean_spots_l795_795055

theorem jean_spots (total_spots upper_torso_spots back_hindspots sides_spots : ℕ)
  (h1 : upper_torso_spots = 30)
  (h2 : total_spots = 2 * upper_torso_spots)
  (h3 : back_hindspots = total_spots / 3)
  (h4 : sides_spots = total_spots - upper_torso_spots - back_hindspots) :
  sides_spots = 10 :=
by
  sorry

end jean_spots_l795_795055


namespace fraction_covered_is_half_l795_795267

-- Given conditions
def diameter_pizza : ℝ := 16
def salami_count : ℕ := 32
def salami_diameter : ℝ := diameter_pizza / 8
def salami_radius : ℝ := salami_diameter / 2
def pizza_radius : ℝ := diameter_pizza / 2

-- Areas
def area_salami : ℝ := π * salami_radius^2
def total_area_salami : ℝ := salami_count * area_salami
def area_pizza : ℝ := π * pizza_radius^2

-- Fraction of the pizza covered by salami
def fraction_covered : ℝ := total_area_salami / area_pizza

theorem fraction_covered_is_half : fraction_covered = 1 / 2 :=
by
  -- skip the proof
  sorry

end fraction_covered_is_half_l795_795267


namespace average_speed_lila_l795_795079

-- Definitions
def distance1 : ℝ := 50 -- miles
def speed1 : ℝ := 20 -- miles per hour
def distance2 : ℝ := 20 -- miles
def speed2 : ℝ := 40 -- miles per hour
def break_time : ℝ := 0.5 -- hours

-- Question to prove: Lila's average speed for the entire ride is 20 miles per hour
theorem average_speed_lila (d1 d2 s1 s2 bt : ℝ) 
  (h1 : d1 = distance1) (h2 : s1 = speed1) (h3 : d2 = distance2) (h4 : s2 = speed2) (h5 : bt = break_time) :
  (d1 + d2) / (d1 / s1 + d2 / s2 + bt) = 20 :=
by
  sorry

end average_speed_lila_l795_795079


namespace quotient_when_dividing_11_by_3_l795_795178

theorem quotient_when_dividing_11_by_3 :
  ∃ A : ℤ, 11 = 3 * A + 2 ∧ A = 3 :=
by
  existsi 3
  split
  repeat {sorry}

end quotient_when_dividing_11_by_3_l795_795178


namespace det_is_even_l795_795458

open Matrix

-- Definitions and assumptions based on given problem conditions
def is_finite_group (G : Type) [Group G] : Prop := Fintype G

def a (G : Type) [Group G] [Fintype G] (x : G) (y : G) : Finset(Fintype.elems(G)) → Matrix Fintype.card _ Fintype.card _
  | xi, xj => if (xi * xj⁻¹) = (xj * xi⁻¹) then 0 else 1

-- The proof problem statement
theorem det_is_even (G : Type) [Group G] [Fintype G] :
  ∀ (x : G) (y : G), (x_1 ... x_n : List G) (a_ij : Fin n → Fin n → \fin 2) (ha_ij : ∀ i j, a_ij i j = if x_i * x_j⁻¹ = x_j * x_i⁻¹ then 0 else 1),
  (det (a_ij)).val % 2 = 0 :=
by 
  sorry

end det_is_even_l795_795458


namespace exists_AB_separator_l795_795774

variables {G : Type} [graph G]
variables {A B : set (vertex G)} {𝓟 : set (path G)}

def ends_in (P : path G) (S : set (vertex G)) : Prop :=
  ∃ v ∈ S, P.ends_at v

def A_B_separator (sep : set (vertex G)) (𝓟 : set (path G)) : Prop :=
  ∀ P ∈ 𝓟, sep ∩ P.vertices ≠ ∅

theorem exists_AB_separator
  (h : ∀ W, ¬ends_in W (B \ V[𝓟])) :
  ∃ X, A_B_separator X 𝓟 :=
sorry

end exists_AB_separator_l795_795774


namespace solve_eqs_l795_795863

theorem solve_eqs (x y : ℤ) (h1 : 7 - x = 15) (h2 : y - 3 = 4 + x) : x = -8 ∧ y = -1 := 
by
  sorry

end solve_eqs_l795_795863


namespace germination_estimate_l795_795799

theorem germination_estimate (germination_rate : ℝ) (total_pounds : ℝ) 
  (hrate_nonneg : 0 ≤ germination_rate) (hrate_le_one : germination_rate ≤ 1) 
  (h_germination_value : germination_rate = 0.971) 
  (h_total_pounds_value : total_pounds = 1000) : 
  total_pounds * (1 - germination_rate) = 29 := 
by 
  sorry

end germination_estimate_l795_795799


namespace mass_not_vector_l795_795633

def is_vector (quantity : Type) : Prop :=
  ∃ (has_magnitude : Prop) (has_direction : Prop), has_magnitude ∧ has_direction

def mass_is_not_vector : Type → Prop :=
  λ Mass, ¬is_vector Mass

axiom mass : Type
axiom velocity : Type
axiom displacement : Type
axiom force : Type

theorem mass_not_vector : mass_is_not_vector mass :=
sorry

end mass_not_vector_l795_795633


namespace parallel_vectors_max_min_f_l795_795382

-- Definitions for vectors and the function f(x)
def m (x : ℝ) : ℝ × ℝ := (2 * Real.sin (x - π / 6), 1)
def n (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Part 1: If m is parallel to n, then x = π/3 or x = 4π/3
theorem parallel_vectors (x : ℝ) (hx : x ∈ Set.Icc 0 (2 * π)) 
  (hparallel : ∃ k : ℝ, m x = k • n x) : 
  x = π / 3 ∨ x = 4 * π / 3 := 
sorry

-- Part 2: Maximum and Minimum value of the dot product function f(x)
theorem max_min_f (hx : x ∈ Set.Icc (-π / 4) (π / 4)) : 
  -2 ≤ f x ∧ f x ≤ Real.sqrt 3 := 
sorry

end parallel_vectors_max_min_f_l795_795382


namespace similar_triangles_incorrect_area_ratio_l795_795957

theorem similar_triangles_incorrect_area_ratio (T1 T2 : Type) [Triangle T1] [Triangle T2] :
  (similar T1 T2) →
  ∀ (r : ℝ), similarity_ratio T1 T2 r →
  ¬(ratio_of_areas T1 T2 = r) := 
by
  sorry

end similar_triangles_incorrect_area_ratio_l795_795957


namespace min_value_pm_pn_l795_795355

def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 3

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def min_pm_pn (p m n : ℝ × ℝ) : ℝ := 
  distance p m + distance p n

theorem min_value_pm_pn : 
  ∀ P : ℝ × ℝ, ∀ M : ℝ × ℝ, ∀ N : ℝ × ℝ, P.2 = 0 → circle1 M.1 M.2 → circle2 N.1 N.2 →
  min_pm_pn P M N = 2 * Real.sqrt(10) - 1 - Real.sqrt(3) :=
sorry

end min_value_pm_pn_l795_795355


namespace distance_from_neg6_to_origin_l795_795531

theorem distance_from_neg6_to_origin :
  abs (-6) = 6 :=
by
  sorry

end distance_from_neg6_to_origin_l795_795531


namespace count_super_prime_looking_l795_795245

def is_super_prime_looking (n : ℕ) : Prop :=
  ¬ n.prime ∧
  n > 1 ∧
  (¬ ∃ k, k ∣ n ∧ (k = 2 ∨ k = 3 ∨ k = 5 ∨ k = 7))

theorem count_super_prime_looking :
  let count := (Finset.filter is_super_prime_looking (Finset.range 1200)).card
  count = 280 :=
by
  sorry

end count_super_prime_looking_l795_795245


namespace smallest_two_digit_number_l795_795183

theorem smallest_two_digit_number :
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧
            n % 12 = 0 ∧
            n % 5 = 4 ∧
            ∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ m % 12 = 0 ∧ m % 5 = 4 → n ≤ m :=
  by {
  -- proof shows the mathematical statement is true
  sorry
}

end smallest_two_digit_number_l795_795183


namespace range_of_function_l795_795670

noncomputable def f (t : ℝ) : ℝ := (1 + t) / (1 + t^2)

theorem range_of_function :
  (∀ t : ℝ, t > 0 → 0 < f t ∧ f t ≤ (√2 + 1) / 2) ∧ 
  (∀ y : ℝ, 0 < y → y ≤ (√2 + 1) / 2 → ∃ t : ℝ, t > 0 ∧ f t = y) :=
by
  sorry

end range_of_function_l795_795670


namespace simplify_expression_l795_795108

-- Define the necessary exponents and intermediate results based on problem conditions.
def four_pow_seven : ℕ := 4 ^ 7
def two_pow_six : ℕ := 2 ^ 6
def sum_one_five_and_neg_one_five : ℕ := (1^5) - ((-1)^5)
def two_pow_ten : ℕ := 2 ^ 10
def two_pow_three : ℕ := 2 ^ 3
def four_pow_two : ℕ := 4 ^ 2

-- The main theorem we want to prove
theorem simplify_expression : 
  (four_pow_seven + two_pow_six) * (sum_one_five_and_neg_one_five ^ 10) * (two_pow_three + four_pow_two) = 404225648 := 
by
  sorry

end simplify_expression_l795_795108


namespace triangle_XYZ_XM_square_sum_l795_795447

theorem triangle_XYZ_XM_square_sum (X Y Z M : Type) (YZ_len : ℝ) (XM_len : ℝ) 
  (midpoint_M : M = (Y + Z) / 2) (YZ_eq_10 : YZ_len = 10) (XM_eq_6 : XM_len = 6) :
  XY^2 + XZ^2 = 122 := 
sorry

end triangle_XYZ_XM_square_sum_l795_795447


namespace integral_inequality_l795_795810

open Real

noncomputable def twice_diff_fun (f : ℝ → ℝ) := ∃ (f' f'' : ℝ → ℝ), (∀ x, has_deriv_at f (f' x) x) ∧ (∀ x, has_deriv_at f' (f'' x) x)

theorem integral_inequality
  (f : ℝ → ℝ)
  (hf : ∀ x, -π / 2 < x ∧ x < π / 2 → has_deriv_at f (deriv f x) x ∧ has_deriv_at (deriv f) (deriv (deriv f) x) x)
  (h_con: ∀ x, -π / 2 < x ∧ x < π / 2 → (deriv (deriv f) x - f x) * tan x + 2 * deriv f x ≥ 1) :
  ∫ x in -π / 2..π / 2, f x * sin x ≥ π - 2 := by
  sorry

end integral_inequality_l795_795810


namespace solve_for_q_l795_795044

variables (T A B O C : Type) (OT TA OB CO : ℝ) (q : ℝ)

def points :=
  T = (0, 15) ∧ A = (3, 15) ∧ B = (15, 0) ∧ O = (0, 0) ∧ C = (0, q)

def areaOfTriangle (A B C : Type) (area_ABC : ℝ) :=
  area_ABC = 36

theorem solve_for_q
  (h_points : points T A B O C)
  (h_area : areaOfTriangle A B C 36) :
  q = 12.75 :=
sorry

end solve_for_q_l795_795044


namespace meaningful_expression_range_l795_795022

theorem meaningful_expression_range {x : ℝ} : (∃ y : ℝ, y = 5 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end meaningful_expression_range_l795_795022


namespace equivalent_annual_interest_rate_l795_795481

def quarterly_rate (annual_rate : ℝ) : ℝ :=
  annual_rate / 4

def equivalent_annual_rate (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate / 100) ^ 4

def convert_to_percentage (rate : ℝ) : ℝ :=
  (rate - 1) * 100

theorem equivalent_annual_interest_rate
  (annual_rate : ℝ)
  (annual_rate = 4.5) :
  convert_to_percentage (equivalent_annual_rate (quarterly_rate annual_rate)) = 4.56 :=
by 
  sorry

end equivalent_annual_interest_rate_l795_795481


namespace circles_internally_tangent_l795_795752

-- Define the given circles
def C1_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * real.sqrt 3 * x - 4 * y + 6 = 0
def C2_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * y = 0

-- Define the circles as having a certain center and radius
def C1_center := (real.sqrt 3, 2)
def C1_radius := 1

def C2_center := (0, 3)
def C2_radius := 3

-- Distance between centers 
def distance_centers := real.sqrt ((real.sqrt 3 - 0)^2 + (2 - 3)^2)

theorem circles_internally_tangent :
  distance_centers = C2_radius - C1_radius :=
by
  -- We would need to prove the actual steps, which is not required here.
  sorry

end circles_internally_tangent_l795_795752


namespace find_a_axis_of_symmetry_range_of_m_l795_795435

variable (a m x y y1 y2 y3 : ℝ)

-- Given conditions
def parabola_eq := y = a * x^2 + (2 * m - 6) * x + 1
def point_on_parabola := parabola_eq a m 1 (2 * m - 4)
def points_on_parabola := (parabola_eq a m (-m) y1) ∧ (parabola_eq a m m y2) ∧ (parabola_eq a m (m + 2) y3)
def inequality_constraints := y2 < y3 ∧ y3 ≤ y1

-- Statements to prove
theorem find_a (h : point_on_parabola) : a = 3 :=
sorry

theorem axis_of_symmetry (h : a = 3) : parabola_axis m :=
sorry

theorem range_of_m (h1 : points_on_parabola) (h2 : inequality_constraints) : 1 < m ∧ m ≤ 2 :=
sorry

end find_a_axis_of_symmetry_range_of_m_l795_795435


namespace solve_equation_l795_795513

theorem solve_equation : ∀ x : ℝ, 64 = 4 * 16^(x - 2) → x = 3 :=
by
  intros x h
  sorry

end solve_equation_l795_795513


namespace exists_constant_C_for_set_diff_l795_795826

open Set Int

def H : Set Int :=
  { i | ∃ k : ℕ, k > 0 ∧ i = floor (k * Real.sqrt 2) } \ {1, 2, 4, 5, 7, 8, ∞}

theorem exists_constant_C_for_set_diff (n : ℕ) (A : Set ℕ)
  (h1 : A ⊆ {i | 1 ≤ i ∧ i ≤ n})
  (h2 : A.card ≥ (Real.sqrt 2 * Real.sqrt 2 - Real.sqrt 2) * Real.sqrt n) :
  ∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ (a - b : ℤ) ∈ H := 
sorry

end exists_constant_C_for_set_diff_l795_795826


namespace positive_integer_solution_l795_795209

theorem positive_integer_solution (x : Int) (h_pos : x > 0) (h_cond : x + 1000 > 1000 * x) : x = 2 :=
sorry

end positive_integer_solution_l795_795209


namespace meaningful_expression_range_l795_795023

theorem meaningful_expression_range {x : ℝ} : (∃ y : ℝ, y = 5 / (x - 2)) ↔ x ≠ 2 :=
by sorry

end meaningful_expression_range_l795_795023


namespace raj_kitchen_area_l795_795500

theorem raj_kitchen_area :
  let house_area := 1110
  let bedrooms := 4
  let bedroom_width := 11
  let bedroom_length := 11
  let bathrooms := 2
  let bathroom_width := 6
  let bathroom_length := 8
  let rooms_area := (bedrooms * bedroom_width * bedroom_length) + (bathrooms * bathroom_width * bathroom_length)
  let shared_area := house_area - rooms_area
  let kitchen_area := shared_area / 2
  in kitchen_area = 265 :=
by
  sorry

end raj_kitchen_area_l795_795500


namespace problem_l795_795347

theorem problem (a : ℕ) (b : ℚ) (c : ℤ) 
  (h1 : a = 1) 
  (h2 : b = 0) 
  (h3 : abs (c) = 6) :
  (a - b + c = (7 : ℤ)) ∨ (a - b + c = (-5 : ℤ)) := by
  sorry

end problem_l795_795347


namespace valid_A_count_l795_795317

def is_digit (A : ℕ) : Prop := A < 10

def divisible (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

def count_valid_A : ℕ :=
  (Finset.filter 
    (λ A => divisible 174 A ∧ divisible (80*A + 4) 4)
    (Finset.range 10)).card

theorem valid_A_count : count_valid_A = 2 :=
by
  sorry

end valid_A_count_l795_795317


namespace number_of_five_digit_palindromes_l795_795258

theorem number_of_five_digit_palindromes : 
  let A := finset.range 1 10  -- 1 to 9
  let B := finset.range 0 10  -- 0 to 9
  let C := finset.range 0 10  -- 0 to 9
  A.card * B.card * C.card = 900 :=
by
  sorry

end number_of_five_digit_palindromes_l795_795258


namespace triangle_MAD_is_isosceles_right_at_M_l795_795070

variables (A B C D E M : Type)
open_locale classical

def is_isosceles_right_triangle_at (A B C : Type) (P : Type) : Prop :=
-- Definition of an isosceles right triangle at point P
sorry

def is_midpoint (x y m : Type) : Prop :=
-- Definition of a midpoint
sorry

theorem triangle_MAD_is_isosceles_right_at_M
  (hABC : is_isosceles_right_triangle_at A B C A)
  (hBDE : is_isosceles_right_triangle_at B D E D)
  (hMidpoint : is_midpoint C E M) :
  is_isosceles_right_triangle_at M A D M :=
sorry

end triangle_MAD_is_isosceles_right_at_M_l795_795070


namespace average_leaves_per_hour_l795_795503

theorem average_leaves_per_hour :
  let leaves_first_hour := 7
  let leaves_second_hour := 4
  let leaves_third_hour := 4
  let total_hours := 3
  let total_leaves := leaves_first_hour + leaves_second_hour + leaves_third_hour
  let average_leaves_per_hour := total_leaves / total_hours
  average_leaves_per_hour = 5 := by
  sorry

end average_leaves_per_hour_l795_795503


namespace sarah_min_correct_l795_795550

theorem sarah_min_correct (c : ℕ) (hc : c * 8 + 10 ≥ 110) : c ≥ 13 :=
sorry

end sarah_min_correct_l795_795550


namespace concert_cost_l795_795504

def ticket_cost : ℕ := 50
def number_of_people : ℕ := 2
def processing_fee_rate : ℝ := 0.15
def parking_fee : ℕ := 10
def per_person_entrance_fee : ℕ := 5

def total_cost : ℝ :=
  let tickets := (ticket_cost * number_of_people : ℕ)
  let processing_fee := tickets * processing_fee_rate
  let entrance_fee := per_person_entrance_fee * number_of_people
  (tickets : ℝ) + processing_fee + (parking_fee : ℝ) + (entrance_fee : ℝ)

theorem concert_cost :
  total_cost = 135 := by
  sorry

end concert_cost_l795_795504


namespace roots_of_unity_real_six_l795_795557

def is_real (z : ℂ) : Prop := ∃ r : ℝ, z = r

theorem roots_of_unity_real_six {z : ℂ} (hz : z^30 = 1) :
  (∑ k in finset.range 30, if is_real (z^6) then 1 else 0) = 6 := 
sorry

end roots_of_unity_real_six_l795_795557


namespace angle_bisector_length_l795_795571

theorem angle_bisector_length (ABC : Triangle)
  (r₁ r₂ : ℝ) (B C : Point) (O₁ O₂ : Point) (K L : Point)
  (r₁_pos : r₁ = 2)
  (r₂_pos : r₂ = 3)
  (inscribe_B : Circle.inscribed O₁ B ABC ∧ Circle.radius O₁ r₁)
  (inscribe_C : Circle.inscribed O₂ C ABC ∧ Circle.radius O₂ r₂)
  (touch_BC_1 : Circle.touches_side_at O₁ K B C)
  (touch_BC_2 : Circle.touches_side_at O₂ L B C)
  (distance_KL : dist K L = 7) :
  ∃ AD : LineSegment, angle_bisector_of A AD ∧ length AD = 16 :=
begin
  sorry
end

end angle_bisector_length_l795_795571


namespace percent_increase_eq_16_13_l795_795637

variable (P_init : ℝ)

def P_1 := P_init * 1.25
def P_2 := P_init * 1.55
def P_3 := P_init * 1.80

def percentIncrease := ((P_3 - P_2) / P_2) * 100

theorem percent_increase_eq_16_13 :
  (P_init ≠ 0) → abs (percentIncrease P_init - 16.13) < 0.01 :=
by
  intros
  sorry

end percent_increase_eq_16_13_l795_795637


namespace find_theta_l795_795754

open Real

noncomputable def a (θ : ℝ) : ℝ × ℝ := (cos θ, -sin θ)
noncomputable def b (θ : ℝ) : ℝ × ℝ := (3 * cos θ, sin θ)

def orthogonal (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

theorem find_theta (θ : ℝ) (hθ : θ ∈ Ioo 0 π) :
  orthogonal (a θ) (b θ) ↔ θ = π / 3 ∨ θ = 2 * π / 3 :=
sorry

end find_theta_l795_795754


namespace mean_value_of_interior_angles_of_quadrilateral_l795_795992

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l795_795992


namespace model_lighthouse_height_l795_795454

theorem model_lighthouse_height (cyl_height_actual : ℝ) (sphere_vol_actual : ℝ) (sphere_vol_model : ℝ) (proportional : ℝ) : 
  (cyl_height_actual = 60) → 
  (sphere_vol_actual = 150000) → 
  (sphere_vol_model = 0.15) → 
  (proportional = (sphere_vol_actual / sphere_vol_model).pow(1/3)) →
  (cyl_height_actual / proportional).toReal = 60 :=
begin
  sorry
end

end model_lighthouse_height_l795_795454


namespace max_points_earned_l795_795223

def divisible_by (a b : Nat) : Prop := b ≠ 0 ∧ a % b = 0

def points (x : Nat) : Nat :=
  (if divisible_by x 3 then 3 else 0) +
  (if divisible_by x 5 then 5 else 0) +
  (if divisible_by x 7 then 7 else 0) +
  (if divisible_by x 9 then 9 else 0) +
  (if divisible_by x 11 then 11 else 0)

theorem max_points_earned (x : Nat) (h1 : 2017 ≤ x) (h2 : x ≤ 2117) :
  points 2079 = 30 := by
  sorry

end max_points_earned_l795_795223


namespace range_of_k_l795_795769

noncomputable def range_of_k_ellipse (k : ℝ) : Prop :=
  ∃ (x y : ℝ), (x^2 + k * y^2 = 2) ∧ (k > 1)

theorem range_of_k (k : ℝ) :
  (∃ (x y : ℝ), x^2 + k * y^2 = 2) ∧ (∀ (x₁ y₁ x₂ y₂ : ℝ), (k * y₁^2 ≤ x₁^2) ∧ (k * y₂^2 ≥ x₂^2)) ↔ k ∈ set.Ioi 1 :=
sorry

end range_of_k_l795_795769


namespace coefficient_zero_l795_795043

theorem coefficient_zero (a : ℝ) :
  (∀ n, n ≠ - 1/2 → binomial_expansion (a - 1/real.sqrt a)^6 n = 0):= 
sorry

end coefficient_zero_l795_795043
