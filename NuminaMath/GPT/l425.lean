import Mathlib
import Mathlib.Algebra.Divisibility
import Mathlib.Algebra.Group.Basic
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.Linear
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Order.Field
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.AM_GM
import Mathlib.Analysis.Calculus.LocalExtr
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Combinatorics.Basic
import Mathlib.Data.Combinatorics.Basic
import Mathlib.Data.Finset
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Int.Basic
import Mathlib.Data.Int.ModEq
import Mathlib.Data.List.Basic
import Mathlib.Data.List.Perm
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Finset
import Mathlib.Geometry.Circle.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Geometry.Triangle.Basic
import Mathlib.LinearAlgebra.Matrix.Determinant
import Mathlib.Probability.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Topology.Basic
import data.int.basic

namespace find_AX_length_l425_425899

theorem find_AX_length (t BC AC BX : ℝ) (AX AB : ℝ)
  (h1 : t = 0.75)
  (h2 : AX = t * AB)
  (h3 : BC = 40)
  (h4 : AC = 35)
  (h5 : BX = 15) :
  AX = 105 / 8 := 
  sorry

end find_AX_length_l425_425899


namespace find_third_side_l425_425900

theorem find_third_side (a b m : ℝ) (h₁ : a = 11) (h₂ : b = 23) (h₃ : m = 10) :
  ∃ c : ℝ, c = 30 :=
by
  have h : m^2 = (1 / 4) * (2 * a^2 + 2 * b^2 - c^2) := sorry
  have eq1 : 10^2 = (1 / 4) * (2 * 11^2 + 2 * 23^2 - c^2) := sorry
  have eq2 : c = 30 := sorry
  { use 30 }

end find_third_side_l425_425900


namespace length_of_BC_l425_425173

theorem length_of_BC (AB_perp_BC : ∀ (A B C : ℝ^3), is_perpendicular AB BC)
                     (CD_perp_AD : ∀ (C D A : ℝ^3), is_perpendicular CD AD)
                     (AC_length : AC = 625)
                     (AD_length : AD = 600)
                     (angle_relation : ∀ (A B C D : ℝ^3), angle BAC = 2 * angle DAC) :
                     BC = 336 := 
sorry

end length_of_BC_l425_425173


namespace unique_solution_l425_425021

noncomputable def check_triplet (a b c : ℕ) : Prop :=
  5^a + 3^b - 2^c = 32

theorem unique_solution : ∀ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ check_triplet a b c ↔ (a = 2 ∧ b = 2 ∧ c = 1) :=
  by sorry

end unique_solution_l425_425021


namespace distance_between_closest_points_l425_425657

noncomputable def distance_closest_points :=
  let center1 : ℝ × ℝ := (5, 3)
  let center2 : ℝ × ℝ := (20, 7)
  let radius1 := center1.2  -- radius of first circle is y-coordinate of its center
  let radius2 := center2.2  -- radius of second circle is y-coordinate of its center
  let distance_centers := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  distance_centers - radius1 - radius2

theorem distance_between_closest_points :
  distance_closest_points = Real.sqrt 241 - 10 :=
sorry

end distance_between_closest_points_l425_425657


namespace perfect_squares_three_digit_divisible_by_4_count_l425_425140

theorem perfect_squares_three_digit_divisible_by_4_count : 
  ∃ (n : ℕ), (n = 11) ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ (∃ k, m = k^2) ∧ (m % 4 = 0) ↔ m ∈ {100, 144, 196, 256, 324, 400, 484, 576, 676, 784, 900}) :=
by
  existsi 11
  split
  · reflexivity
  · intro m
    split
    · intro h
      rcases h with ⟨⟨_, _, _, _⟩, _⟩
      -- details have been omitted
      sorry
    · intro h
      -- details have been omitted
      sorry

end perfect_squares_three_digit_divisible_by_4_count_l425_425140


namespace functional_equation_solution_l425_425020

noncomputable def f (x : ℝ) : ℝ := (-x^3 + x - 1) / (2 * (1-x) * x)

theorem functional_equation_solution (x : ℝ) (h : x ≠ 0 ∧ x ≠ 1) : 
  f(x) + f(1/(1-x)) = x := 
by 
  sorry

end functional_equation_solution_l425_425020


namespace factorize_expression_l425_425425

theorem factorize_expression (a : ℝ) : a^2 + 5 * a = a * (a + 5) :=
sorry

end factorize_expression_l425_425425


namespace area_of_rhombus_l425_425461

variable {ABCD : Type} [rhombus ABCD]
variable (AC BD : ℝ)

theorem area_of_rhombus (h1 : AC = 12) (h2 : BD = 16) : 
  let A := (1/2) * AC * BD in A = 96 :=
by
  sorry

end area_of_rhombus_l425_425461


namespace polynomial_identity_l425_425211

theorem polynomial_identity :
  (3 * x ^ 2 - 4 * y ^ 3) * (9 * x ^ 4 + 12 * x ^ 2 * y ^ 3 + 16 * y ^ 6) = 27 * x ^ 6 - 64 * y ^ 9 :=
by
  sorry

end polynomial_identity_l425_425211


namespace simplify_fraction_l425_425243

theorem simplify_fraction :
  (1 / ((1 / (Real.sqrt 2 + 1)) + (2 / (Real.sqrt 3 - 1)) + (3 / (Real.sqrt 5 + 2)))) =
  (1 / (Real.sqrt 2 + 2 * Real.sqrt 3 + 3 * Real.sqrt 5 - 5)) :=
by
  sorry

end simplify_fraction_l425_425243


namespace least_possible_qr_integral_l425_425334

def pq : ℝ := 7
def pr : ℝ := 15
def sr : ℝ := 10
def sq : ℝ := 25

theorem least_possible_qr_integral (QR : ℝ) (h1 : QR > pr - pq) (h2 : QR > sq - sr) : QR = 16 :=
by
  sorry

end least_possible_qr_integral_l425_425334


namespace area_sum_of_quadrilateral_l425_425894

theorem area_sum_of_quadrilateral (W X Y Z : Type*) 
  [convex_quadrilateral W X Y Z]
  (h1 : distance W Y = 6) 
  (h2 : distance Y Z = 5) 
  (h3 : distance Z X = 8) 
  (h4 : distance W Z = 8) 
  (h5 : angle Z W X = 45) :
  let a := 128, b := 15, c := 7 in a + b + c = 150 := by 
  sorry

end area_sum_of_quadrilateral_l425_425894


namespace find_a_l425_425873

theorem find_a (a b : ℝ) (h1 : 0 < a ∧ 0 < b) (h2 : a^b = b^a) (h3 : b = 4 * a) : 
  a = (4 : ℝ)^(1 / 3) :=
by
  sorry

end find_a_l425_425873


namespace find_remaining_amount_l425_425208

def remaining_amount_for_notebooks(p c_pen T : ℕ) : ℕ :=
  T - (p * c_pen)

def condition_example := ∀(p c_pen T : ℕ), p = 5 ∧ c_pen = 2 ∧ T = 30 → remaining_amount_for_notebooks p c_pen T = 20

theorem find_remaining_amount : condition_example :=
by {
  intros p c_pen T h,
  cases h,
  cases h,
  rw [h_right_right, h_right_left, h_left],
  exact rfl
}

end find_remaining_amount_l425_425208


namespace imaginary_part_of_z_l425_425085

noncomputable def z : ℂ := (3 + 2 * complex.I) / (1 + complex.I)

-- Statement to prove
theorem imaginary_part_of_z :
  complex.im (z * (1 + complex.I)) = complex.im (3 + 2 * complex.I) →
  complex.im z = -1/2 :=
by
  assume h : complex.im (z * (1 + complex.I)) = complex.im (3 + 2 * complex.I)
  sorry  -- Proof is not required.

end imaginary_part_of_z_l425_425085


namespace avg_remaining_numbers_l425_425634

-- Conditions
variable (numbers : Fin 12 → ℝ) (h_avg : (∑ i, numbers i) / 12 = 90)
variable (number_80 : ∃ j, numbers j = 80) (number_82 : ∃ k, numbers k = 82)

-- Theorem statement
theorem avg_remaining_numbers : 
  (∑ i, numbers i - 80 - 82) / 10 = 91.8 :=
by 
  sorry

end avg_remaining_numbers_l425_425634


namespace power_C_50_l425_425183

def matrixC : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, 1], ![-4, -1]]

theorem power_C_50 :
  matrixC ^ 50 = ![![4^49 + 1, 4^49], ![-4^50, -2 * 4^49 + 1]] :=
by
  sorry

end power_C_50_l425_425183


namespace solutions_eq1_solutions_eq2_l425_425625

noncomputable def equation_sol1 : Set ℝ :=
{ x | x^2 - 8 * x + 1 = 0 }

noncomputable def equation_sol2 : Set ℝ :=
{ x | x * (x - 2) - x + 2 = 0 }

theorem solutions_eq1 : ∀ x ∈ equation_sol1, x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15 :=
by
  intro x hx
  sorry

theorem solutions_eq2 : ∀ x ∈ equation_sol2, x = 2 ∨ x = 1 :=
by
  intro x hx
  sorry

end solutions_eq1_solutions_eq2_l425_425625


namespace count_three_digit_perfect_squares_divisible_by_4_l425_425120

theorem count_three_digit_perfect_squares_divisible_by_4 : ∃ n, n = 11 ∧
  (∀ k, 100 ≤ k ^ 2 ∧ k ^ 2 ≤ 999 → k ^ 2 % 4 = 0 → (k % 2 = 0 ∧ 10 ≤ k ≤ 31)) :=
sorry

end count_three_digit_perfect_squares_divisible_by_4_l425_425120


namespace part1_part2_l425_425821

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < a then 2 * a - (x + 4 / x) else x - 4 / x

theorem part1 (h : f 1 x = 3) : x = 4 := sorry

theorem part2 (a : ℝ) (ha : a ≤ -1) 
  (h : ∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ f a x1 = 3 ∧ f a x2 = 3 ∧ f a x3 = 3 ∧ x2 - x1 = x3 - x2) : 
  a = -11 / 6 := sorry

end part1_part2_l425_425821


namespace power_function_through_point_l425_425463

-- Define the condition that the power function passes through the point (2, 8)
theorem power_function_through_point (f : ℝ → ℝ) (α : ℝ) (h : ∀ x, f x = x^α) (h₂ : f 2 = 8) :
  α = 3 ∧ ∀ x, f x = x^3 :=
by
  -- Proof will be provided here
  sorry

end power_function_through_point_l425_425463


namespace problem_solution_l425_425796

theorem problem_solution :
  let x := 3^2020 in
  let value := (x + 3*x) * (3*x + 9*x) * (9*x + 27*x) * (27*x + 81*x) in
  let simplified_value := 2^8 * 3^(8086) in
  (divisors_count simplified_value) % 1000 = 783 :=
by
  sorry

end problem_solution_l425_425796


namespace length_of_rectangle_l425_425152

theorem length_of_rectangle (P L B : ℕ) (h₁ : P = 800) (h₂ : B = 300) (h₃ : P = 2 * (L + B)) : L = 100 := by
  sorry

end length_of_rectangle_l425_425152


namespace initial_weight_l425_425924

theorem initial_weight (W : ℝ) (current_weight : ℝ) (future_weight : ℝ) (months : ℝ) (additional_months : ℝ) 
  (constant_rate : Prop) :
  current_weight = 198 →
  future_weight = 170 →
  months = 3 →
  additional_months = 3.5 →
  constant_rate →
  W = 222 :=
by
  intros h_current_weight h_future_weight h_months h_additional_months h_constant_rate
  -- proof would go here
  sorry

end initial_weight_l425_425924


namespace payment_difference_is_correct_l425_425923

noncomputable def plan1_payment (principal : ℝ) (rate : ℝ) (years : ℝ) (payment_fraction : ℝ) : ℝ :=
let balance_after_6_years := principal * (1 + rate / 2) ^ (2 * 6)
let payment_at_6_years := payment_fraction * balance_after_6_years
let remaining_balance := balance_after_6_years - payment_at_6_years
let final_payment := remaining_balance * (1 + rate / 2) ^ (2 * 6)
in payment_at_6_years + final_payment

noncomputable def plan2_payment (principal : ℝ) (rate : ℝ) (years : ℝ) : ℝ :=
principal * (1 + rate) ^ years

theorem payment_difference_is_correct :
  let principal := 12000
  let rate := 0.08
  let years := 12
  let payment_fraction := 0.2
  let total_payment_plan1 := plan1_payment principal rate years payment_fraction
  let total_payment_plan2 := plan2_payment principal rate years
  abs (total_payment_plan2 - total_payment_plan1) = 1783 :=
by sorry

end payment_difference_is_correct_l425_425923


namespace local_minimum_at_one_l425_425040

noncomputable def f (c : ℝ) : ℝ := c^3 + 3 / 2 * c^2 - 6 * c + 4

theorem local_minimum_at_one : ∃ ε > (0 : ℝ), ∀ c : ℝ, |c - 1| < ε → f(c) ≥ f(1) :=
by
  sorry

end local_minimum_at_one_l425_425040


namespace bananas_to_pears_l425_425743

theorem bananas_to_pears : ∀ (cost_banana cost_apple cost_pear : ℚ),
  (5 * cost_banana = 3 * cost_apple) →
  (9 * cost_apple = 6 * cost_pear) →
  (25 * cost_banana = 10 * cost_pear) :=
by
  intros cost_banana cost_apple cost_pear h1 h2
  sorry

end bananas_to_pears_l425_425743


namespace repeating_decimal_sum_l425_425780

-- Definitions based on conditions
def x := 5 / 9  -- We derived this from 0.5 repeating as a fraction
def y := 7 / 99  -- Similarly, derived from 0.07 repeating as a fraction

-- Proposition to prove
theorem repeating_decimal_sum : x + y = 62 / 99 := by
  sorry

end repeating_decimal_sum_l425_425780


namespace smallest_positive_integer_square_begins_with_1989_l425_425797

theorem smallest_positive_integer_square_begins_with_1989 :
  ∃ (A : ℕ), (1989 * 10^0 ≤ A^2 ∧ A^2 < 1990 * 10^0) 
  ∨ (1989 * 10^1 ≤ A^2 ∧ A^2 < 1990 * 10^1) 
  ∨ (1989 * 10^2 ≤ A^2 ∧ A^2 < 1990 * 10^2)
  ∧ A = 446 :=
sorry

end smallest_positive_integer_square_begins_with_1989_l425_425797


namespace ab_sum_not_one_l425_425965

theorem ab_sum_not_one (a b : ℝ) : a^2 + 2*a*b + b^2 + a + b - 2 ≠ 0 → a + b ≠ 1 :=
by
  intros h
  sorry

end ab_sum_not_one_l425_425965


namespace probability_calculation_l425_425507

noncomputable def probability_less_than_six_miles (p q : ℝ) : ℝ :=
  if h : 0 ≤ p ∧ p ≤ 2 ∧ 7 ≤ q ∧ q ≤ 9 then
    if q - p < 6 then 1 else 0
  else 0

theorem probability_calculation :
  (∫ x in 0..2, ∫ y in 7..9, if y - x < 6 then 1 else 0) / (2 * 2) = 1 / 4 :=
sorry

end probability_calculation_l425_425507


namespace total_distance_travelled_l425_425494

theorem total_distance_travelled (distance_to_market : ℕ) (travel_time_minutes : ℕ) (speed_mph : ℕ) 
  (h1 : distance_to_market = 30) 
  (h2 : travel_time_minutes = 30) 
  (h3 : speed_mph = 20) : 
  (distance_to_market + ((travel_time_minutes / 60) * speed_mph) = 40) :=
by
  sorry

end total_distance_travelled_l425_425494


namespace angle_bisector_vs_altitude_l425_425966

-- Define the triangle and its properties
variables {A B C H L : Type}
variables {BC AC AB : ℝ} (BC_le_AC_le_AB : BC ≤ AC ∧ AC ≤ AB)
variables (altitude_AH : ℝ) (angle_bisector_CL : ℝ) 

-- Given the conditions of the problem
def given_conditions := BC ≤ AC ∧ AC ≤ AB ∧
  -- altitude dropped to the smallest side
  altitude_AH = altitude BC ∧
  -- angle bisector to the longest side
  angle_bisector_CL = bisector AB

-- Formalize the statement to be proved
theorem angle_bisector_vs_altitude {A B C H L : Type} {BC AC AB : ℝ} 
  (BC_le_AC_le_AB : BC ≤ AC ∧ AC ≤ AB)
  (altitude_AH : ℝ) (angle_bisector_CL : ℝ) 
  (h_conditions : given_conditions) :
  altitude_AH > angle_bisector_CL :=
sorry

end angle_bisector_vs_altitude_l425_425966


namespace number_of_correct_statements_l425_425903

def is_principal_view_area (edge_length : ℝ) (area : ℝ) : Prop :=
  area ∈ set.Icc (Real.sqrt 3) 8

theorem number_of_correct_statements :
  let edge_length := 2
  let statements := [
    (Real.sqrt 2),
    (2 * Real.sqrt 6 / 3),
    (Real.sqrt 3),
    2,
    4
  ]
  let correct_statements := (statements.filter (is_principal_view_area edge_length)).length
  correct_statements = 2 :=
by
  sorry

end number_of_correct_statements_l425_425903


namespace minimum_distance_l425_425859

noncomputable def f (x : ℝ) := 2 * (x + 1)
noncomputable def g (x : ℝ) := x + Real.log x

def A (m : ℝ) : ℝ × ℝ := (m, f m)
def B (n : ℝ) : ℝ × ℝ := (n, g n)

theorem minimum_distance :
  ∃ (m n : ℝ), f m = g n ∧ abs (m - n) = 3 / 2 :=
sorry

end minimum_distance_l425_425859


namespace repeating_sum_to_fraction_l425_425006

theorem repeating_sum_to_fraction :
  (0.333333333333333 ~ 1/3) ∧ 
  (0.0404040404040401 ~ 4/99) ∧ 
  (0.005005005005001 ~ 5/999) →
  (0.333333333333333 + 0.0404040404040401 + 0.005005005005001) = (112386 / 296703) := 
by
  repeat { sorry }

end repeating_sum_to_fraction_l425_425006


namespace count_oddly_powerful_under_5000_l425_425406

open Nat

def is_oddly_powerful (n : ℕ) : Prop :=
  ∃ a b : ℕ, b > 1 ∧ Odd b ∧ a^b = n

noncomputable def oddly_powerful_under_5000 : List ℕ :=
  (List.range 5000).filter is_oddly_powerful

theorem count_oddly_powerful_under_5000 :
  (oddly_powerful_under_5000.length = 20) :=
by
  sorry

end count_oddly_powerful_under_5000_l425_425406


namespace distinct_solution_count_eq_count_of_distinct_solutions_l425_425109

theorem distinct_solution_count_eq (x : ℝ) : (|x - 10| = |x - 3|) ↔ x = 6.5 :=
by
    sorry

theorem count_of_distinct_solutions : nat.card {x : ℝ // |x - 10| = |x - 3|} = 1 :=
by
    sorry

end distinct_solution_count_eq_count_of_distinct_solutions_l425_425109


namespace box_width_l425_425615

theorem box_width (rate : ℝ) (time : ℝ) (length : ℝ) (depth : ℝ) (volume : ℝ) (width : ℝ) : 
  rate = 4 ∧ time = 21 ∧ length = 7 ∧ depth = 2 ∧ volume = rate * time ∧ volume = length * width * depth → width = 6 :=
by
  sorry

end box_width_l425_425615


namespace determinant_value_l425_425394

variable (a1 b1 b2 c1 c2 c3 d1 d2 d3 d4 : ℝ)

def matrix_det : ℝ :=
  Matrix.det ![
    ![a1, b1, c1, d1],
    ![a1, b2, c2, d2],
    ![a1, b2, c3, d3],
    ![a1, b2, c3, d4]
  ]

theorem determinant_value : 
  matrix_det a1 b1 b2 c1 c2 c3 d1 d2 d3 d4 = 
  a1 * (b2 - b1) * (c3 - c2) * (d4 - d3) :=
by
  sorry

end determinant_value_l425_425394


namespace more_profitable_to_sell_fresh_l425_425705

theorem more_profitable_to_sell_fresh : 
  let weight := 49
  let fresh_price := 1.25
  let weight_after_desiccation := (2/7) * weight
  let desiccated_price := fresh_price + 2
  let revenue_fresh := weight * fresh_price
  let revenue_desiccated := weight_after_desiccation * desiccated_price
  revenue_fresh > revenue_desiccated :=
begin
  sorry
end

end more_profitable_to_sell_fresh_l425_425705


namespace perfect_squares_three_digit_divisible_by_4_count_l425_425137

theorem perfect_squares_three_digit_divisible_by_4_count : 
  ∃ (n : ℕ), (n = 11) ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ (∃ k, m = k^2) ∧ (m % 4 = 0) ↔ m ∈ {100, 144, 196, 256, 324, 400, 484, 576, 676, 784, 900}) :=
by
  existsi 11
  split
  · reflexivity
  · intro m
    split
    · intro h
      rcases h with ⟨⟨_, _, _, _⟩, _⟩
      -- details have been omitted
      sorry
    · intro h
      -- details have been omitted
      sorry

end perfect_squares_three_digit_divisible_by_4_count_l425_425137


namespace vector_scal_mult_and_addition_l425_425047

variable (a : ℝ × ℝ)
variable (b : ℝ × ℝ)

theorem vector_scal_mult_and_addition (ha : a = (3, 1)) (hb : b = (-2, 5)) :
  3 • a - 2 • b = (13, -7) :=
by
  -- Definitions and statements are correct, proof not included
  sorry

end vector_scal_mult_and_addition_l425_425047


namespace morse_code_sequences_l425_425890

theorem morse_code_sequences : 
  let number_of_sequences := 
        (2 ^ 1) + (2 ^ 2) + (2 ^ 3) + (2 ^ 4) + (2 ^ 5)
  number_of_sequences = 62 :=
by
  sorry

end morse_code_sequences_l425_425890


namespace number_of_distinct_triangles_l425_425762

open Nat

def is_point (x y : ℕ) : Prop := 37 * x + y = 2223

theorem number_of_distinct_triangles :
  let points := { (x, y) | x ∈ Finset.range (61) ∧ is_point x y }
  ∃ (P Q : (ℕ × ℕ)), P ≠ Q ∧ P ∈ points ∧ Q ∈ points ∧
  (P.1 - Q.1) % 2 = 0 ∧
  P ∈ points ∧ Q ∈ points :=
  465 + 435 = 900 :=
sorry

end number_of_distinct_triangles_l425_425762


namespace same_terminal_side_l425_425769

theorem same_terminal_side : ∃ k : ℤ, k * 360 - 60 = 300 := by
  sorry

end same_terminal_side_l425_425769


namespace generating_function_lucas_correct_l425_425675

noncomputable def generating_function_fibonacci : FormalSeries ℝ :=
  λ x => (1 : ℝ) / (1 - x - x^2)

def lucas_numbers (n : ℕ) : ℕ :=
  fib (n + 1) + fib (n - 1)

noncomputable def generating_function_lucas : FormalSeries ℝ :=
  λ x => (2 - x) / (1 - x - x^2)

theorem generating_function_lucas_correct :
  ∀ x : ℝ, generating_function_lucas x = (2 - x) / (1 - x - x^2) :=
by
  apply generating_function_lucas_correct


end generating_function_lucas_correct_l425_425675


namespace calculate_expression_l425_425391

theorem calculate_expression :
  2^(-(1/2) : ℝ) + 
  (↑(-4)^0 : ℝ) / real.sqrt 2 + 
  1 / (real.sqrt 2 - 1) - 
  real.sqrt ((1 - real.sqrt 5) ^ 0 : ℝ)
  = 2 * real.sqrt 2 :=
by sorry

end calculate_expression_l425_425391


namespace rachel_picked_apples_l425_425613

-- Defining the conditions
def original_apples : ℕ := 11
def grown_apples : ℕ := 2
def apples_left : ℕ := 6

-- Defining the equation
def equation (x : ℕ) : Prop :=
  original_apples - x + grown_apples = apples_left

-- Stating the theorem
theorem rachel_picked_apples : ∃ x : ℕ, equation x ∧ x = 7 :=
by 
  -- proof skipped 
  sorry

end rachel_picked_apples_l425_425613


namespace expected_value_correct_variance_correct_standard_deviation_correct_l425_425536

def n : ℕ := 109500
def p : ℝ := 0.51

def expected_value : ℝ := n * p
def variance : ℝ := n * p * (1 - p)
def standard_deviation : ℝ := Real.sqrt variance

theorem expected_value_correct : expected_value = 55845 := sorry
theorem variance_correct : variance = 27363.75 := sorry
theorem standard_deviation_correct : standard_deviation ≈ 165.42 := sorry

end expected_value_correct_variance_correct_standard_deviation_correct_l425_425536


namespace twelve_sided_figure_correct_area_l425_425727

def twelve_sided_figure_area : ℝ :=
  let full_squares_area := 8.0 -- Area from 8 full squares
  let triangles_area := 5 * (1.0 / 2.0) -- Area from 5 triangles each with 1/2 cm^2
  full_squares_area + triangles_area

theorem twelve_sided_figure_correct_area :
  twelve_sided_figure_area = 10.5 :=
by {
  -- We calculate the area step-by-step as described
  sorry
}

end twelve_sided_figure_correct_area_l425_425727


namespace distribution_of_earnings_l425_425244

theorem distribution_of_earnings :
  let payments := [10, 15, 20, 25, 30, 50]
  let total_earnings := payments.sum 
  let equal_share := total_earnings / 6
  50 - equal_share = 25 := by
  sorry

end distribution_of_earnings_l425_425244


namespace hay_from_grass_l425_425272

theorem hay_from_grass (moisture_grass moisture_hay dry_hay dry_grass : ℝ) (h1 : moisture_grass = 0.60)
(h2 : moisture_hay = 0.15)
(h3 : dry_grass = 1000 * (1 - moisture_grass))
(h4 : dry_hay = 0.85)
: 1000 * (1 - moisture_grass) / 0.85 = (40000 / 85) := 
by { 
  rw [h1, <-h4],
  simp [h3] 
  sorry 
}

end hay_from_grass_l425_425272


namespace least_xy_value_l425_425065

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 90 :=
by
  sorry

end least_xy_value_l425_425065


namespace orthocenter_iff_concyclic_perpendicular_l425_425905

variables {A B C D E H M N : Type*}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace H] [MetricSpace M] [MetricSpace N]

def is_acute_triangle (A B C : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ^ 2 + b ^ 2 > c ^ 2 ∧ b ^ 2 + c ^ 2 > a ^ 2 ∧ c ^ 2 + a ^ 2 > b ^ 2

def is_midpoint_of (M : Type*) (P Q : Type*) [MetricSpace P] [MetricSpace Q] [MetricSpace M] : Prop :=
  dist P M = dist Q M

def is_concyclic (B C D E : Type*) [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] : Prop :=
  ∃ (O : Type*) [MetricSpace O], dist O B = dist O C ∧ dist O C = dist O D ∧ dist O D = dist O E

def is_orthocenter (H : Type*) (A M N : Type*) [MetricSpace A] [MetricSpace M] [MetricSpace N] [MetricSpace H] : Prop :=
  ∃ (a b c : ℝ), H = orthocenter_of_triangle a b c

theorem orthocenter_iff_concyclic_perpendicular
  (A B C D E H M N : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E] [MetricSpace H] [MetricSpace M] [MetricSpace N]
  (h_acute : is_acute_triangle A B C)
  (h_D_on_AB : dist A D + dist D B = dist A B)
  (h_E_on_AC : dist A E + dist E C = dist A C)
  (h_BE_CD : BE ∩ DC = {H})
  (h_mid_M : is_midpoint_of M B D)
  (h_mid_N : is_midpoint_of N C E) :
  is_orthocenter H A M N ↔ (is_concyclic B C D E ∧ perp_Line (BE : Line) (CD : Line)) :=
sorry

end orthocenter_iff_concyclic_perpendicular_l425_425905


namespace number_of_distinct_sequences_l425_425499

theorem number_of_distinct_sequences :
  let letters := ["B", "A", "N", "A", "N", "A"]
  let possible_sequences := { s : List String // s.head? = some "B" ∧ s.getLast? = some "N" ∧ s.dedup.length = s.length ∧ ∀ c ∈ s, c ∈ letters }
  true := possible_sequences.card = 3 :=
sorry

end number_of_distinct_sequences_l425_425499


namespace central_flower_angle_l425_425714

theorem central_flower_angle : 
  let full_circle := 360
  let number_of_rays := 10
  let central_angle := full_circle / number_of_rays
  let sectors_between_east_and_wsw := 5
  measure_angle {east west_southwest : ℕ} := sectors_between_east_and_wsw * central_angle
where east := 2, west_southwest := 7 : (measure_angle) = 180 :=
by sorry

end central_flower_angle_l425_425714


namespace time_to_pass_platform_l425_425693

def length_train : ℝ := 2400
def time_tree : ℝ := 60
def length_platform : ℝ := 1800

def speed_train : ℝ := length_train / time_tree
def combined_length : ℝ := length_train + length_platform
def time_platform : ℝ := combined_length / speed_train

theorem time_to_pass_platform : time_platform = 105 := by
  -- The proof would go here
  sorry

end time_to_pass_platform_l425_425693


namespace square_side_length_l425_425659

theorem square_side_length (a : ℝ) :
  ∀ (x : ℝ),
  (∃ (M N K L : ℝ),
    (
      -- Two vertices of the square are located on the base of an isosceles triangle
      (M = 0 ∧ N = a) ∧
      -- The other two vertices are on its lateral sides
      (K = x * sqrt 3 ∧ L = a - x * sqrt 3) ∧
      -- The base of the triangle is a
      (K + L = a)
    )) →
  x = (a * (2 * sqrt 3 - 1)) / 11 :=
begin
  intros x h,
  sorry
end

end square_side_length_l425_425659


namespace recycling_drive_l425_425346

theorem recycling_drive (S : ℕ) 
  (h1 : ∀ (n : ℕ), n = 280 * S) -- Each section collected 280 kilos in two weeks
  (h2 : ∀ (t : ℕ), t = 2000 - 320) -- After the third week, they needed 320 kilos more to reach their target of 2000 kilos
  : S = 3 :=
by
  sorry

end recycling_drive_l425_425346


namespace problem_probability_l425_425413

open ProbabilityTheory

/-- Two boxes labeled as A and B, each containing four balls with labels 1, 2, 3, and 4.
    One ball is drawn from each box, and each ball has an equal chance of being drawn.
    (1) Prove that the probability that the two drawn balls have consecutive numbers is 3/8.
    (2) Prove that the probability that the sum of the numbers on the two drawn balls is divisible by 3 is 5/16. -/
theorem problem_probability :
  let outcomes := [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4), (3,1), (3,2), (3,3), (3,4), (4,1), (4,2), (4,3), (4,4)] in
  let consecutive := [(1,2), (2,1), (2,3), (3,2), (3,4), (4,3)] in
  let divisible_by_3 := [(1,2), (2,1), (2,4), (3,3), (4,2)] in
  (prob.of_finset (outcomes.to_finset) consecutive) = 3 / 8 ∧
  (prob.of_finset (outcomes.to_finset) divisible_by_3) = 5 / 16 := by
    sorry

end problem_probability_l425_425413


namespace derivative_of_f_l425_425994

def f (x : ℝ) : ℝ := Real.cos x - x^3

theorem derivative_of_f : ∀ x : ℝ, deriv f x = -Real.sin x - 3 * x^2 := 
by
  unfold f
  intro x
  apply deriv_sub
  { apply deriv_cos }
  { apply deriv_pow x (3 : ℕ) }
  sorry

end derivative_of_f_l425_425994


namespace numberOfCubesWithNoMoreThanFourNeighbors_l425_425044

def unitCubesWithAtMostFourNeighbors (a b c : ℕ) (h1 : a > 4) (h2 : b > 4) (h3 : c > 4) 
(h4 : (a - 2) * (b - 2) * (c - 2) = 836) : ℕ := 
  4 * (a - 2 + b - 2 + c - 2) + 8

theorem numberOfCubesWithNoMoreThanFourNeighbors (a b c : ℕ) 
(h1 : a > 4) (h2 : b > 4) (h3 : c > 4)
(h4 : (a - 2) * (b - 2) * (c - 2) = 836) :
  unitCubesWithAtMostFourNeighbors a b c h1 h2 h3 h4 = 144 :=
sorry

end numberOfCubesWithNoMoreThanFourNeighbors_l425_425044


namespace new_student_weight_l425_425991

theorem new_student_weight :
  let avg_weight_29 := 28
  let num_students_29 := 29
  let avg_weight_30 := 27.4
  let num_students_30 := 30
  let total_weight_29 := avg_weight_29 * num_students_29
  let total_weight_30 := avg_weight_30 * num_students_30
  let new_student_weight := total_weight_30 - total_weight_29
  new_student_weight = 10 :=
by
  sorry

end new_student_weight_l425_425991


namespace incorrect_statement_l425_425516

theorem incorrect_statement (p q : Prop) (hp : ¬ p) (hq : q) : ¬ (¬ q) :=
by
  sorry

end incorrect_statement_l425_425516


namespace sum_of_roots_eq_l425_425669

theorem sum_of_roots_eq (k : ℝ) : ∃ x1 x2 : ℝ, (2 * x1 ^ 2 - 3 * x1 + k = 7) ∧ (2 * x2 ^ 2 - 3 * x2 + k = 7) ∧ (x1 + x2 = 3 / 2) :=
by sorry

end sum_of_roots_eq_l425_425669


namespace parabola_line_dot_product_l425_425478

theorem parabola_line_dot_product (k x1 x2 y1 y2 : ℝ) 
  (h_line: ∀ x, y = k * x + 2)
  (h_parabola: ∀ x, y = (1 / 4) * x ^ 2) 
  (h_A: y1 = k * x1 + 2 ∧ y1 = (1 / 4) * x1 ^ 2)
  (h_B: y2 = k * x2 + 2 ∧ y2 = (1 / 4) * x2 ^ 2) :
  x1 * x2 + y1 * y2 = -4 := 
sorry

end parabola_line_dot_product_l425_425478


namespace simplify_factorial_expression_l425_425241

theorem simplify_factorial_expression :
  (15.factorial : ℚ) / ((12.factorial) + (3 * 10.factorial)) = 2668 := by
  sorry

end simplify_factorial_expression_l425_425241


namespace max_runs_in_ideal_t20_l425_425891

theorem max_runs_in_ideal_t20 (N : ℕ) (hN : N = 20) 
  (no_extras : ∀ (ov : ℕ), (ov < N) → ¬(wide ∨ no_ball ∨ extras ∨ overthrow))
  (max_sixes_per_over : ∀ (ov : ℕ), (ov < N) → (∀ (balls : ℕ), (balls = 6) → (sixes ≤ 3))
  (field_restrictions : ∀ (ov : ℕ), (ov < 10) → (fielders_outside_30_yard_circle ≤ 2)
   ∧ (ov ≥ 10 ∧ ov < N) → (fielders_outside_30_yard_circle ≤ 5))
  (max_overs_per_bowler : ∀ (b : Bowler), (overs_bowled_by b < 4)) :
  batsman_max_runs N = 600 :=
by sorry

end max_runs_in_ideal_t20_l425_425891


namespace largest_M_lemma_l425_425050

theorem largest_M_lemma (n : ℕ) : 
  ∃ M : ℕ, 
  (∀ (a : Fin n → ℕ), 
    (∑ i, ⌊Real.sqrt (a i)⌋) ≥ ⌊Real.sqrt (∑ i, a i + M * Finset.min' (Finset.univ.image a) ⟨0, by simp⟩)⌋) ∧
  M = ⌊(n^2 - 3n) / 3⌋ := 
sorry

end largest_M_lemma_l425_425050


namespace strawberries_left_l425_425888

-- Definitions based on conditions
def initial_amount_strawberries : ℕ := 3000 + 300  -- 3 kg 300 grams in grams
def given_amount_strawberries : ℕ := 1000 + 900  -- 1 kg 900 grams in grams

-- Theorem statement that captures the proof problem
theorem strawberries_left : initial_amount_strawberries - given_amount_strawberries = 1400 :=
by
  rw [initial_amount_strawberries, given_amount_strawberries]
  sorry

end strawberries_left_l425_425888


namespace frog_arrangements_correct_l425_425299

noncomputable def numFrogArrangements : Nat :=
  let greenPermutations := 2.factorial -- Calculation for 2!
  let redPermutations := 3.factorial -- Calculation for 3!
  let numColorArrangements := 2 -- Two distinct arrangements: (Green, Blue, Red) and (Red, Blue, Green)
  numColorArrangements * greenPermutations * redPermutations

theorem frog_arrangements_correct :
  numFrogArrangements = 24 :=
by
  sorry

end frog_arrangements_correct_l425_425299


namespace massager_usage_time_l425_425950

theorem massager_usage_time
  (vibrations_lowest : ℕ := 1600)
  (vibrations_increase_percent : ℕ := 60)
  (total_vibrations : ℕ := 768000) :
  let vibrations_highest := vibrations_lowest + (vibrations_increase_percent * vibrations_lowest) / 100
  in
  total_vibrations / vibrations_highest / 60 = 5 :=
by
  let vibrations_highest := vibrations_lowest + (vibrations_increase_percent * vibrations_lowest) / 100
  have h1 : vibrations_highest = 2560, from sorry,
  have h2 : total_vibrations / vibrations_highest = 300, from sorry,
  have h3 : 300 / 60 = 5, from sorry,
  exact h3

end massager_usage_time_l425_425950


namespace probability_A_level_l425_425347

theorem probability_A_level (p_B : ℝ) (p_C : ℝ) (h_B : p_B = 0.03) (h_C : p_C = 0.01) : 
  (1 - (p_B + p_C)) = 0.96 :=
by
  -- Proof is omitted
  sorry

end probability_A_level_l425_425347


namespace andrei_stamps_l425_425601

theorem andrei_stamps (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 5 = 3) ∧ (x % 7 = 5) ∧ (150 < x) ∧ (x ≤ 300) → 
  x = 208 :=
sorry

end andrei_stamps_l425_425601


namespace find_length_of_DE_l425_425815

-- Define the setup: five points A, B, C, D, E on a circle
variables (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]

-- Define the given distances 
def AB : ℝ := 7
def BC : ℝ := 7
def AD : ℝ := 10

-- Define the total distance AC
def AC : ℝ := AB + BC

-- Define the length DE to be solved
def DE : ℝ := 0.2

-- State the theorem to be proved given the conditions
theorem find_length_of_DE : 
  DE = 0.2 :=
sorry

end find_length_of_DE_l425_425815


namespace least_number_divisible_l425_425665

theorem least_number_divisible (n : ℕ) :
  (∃ n, (n + 3) % 24 = 0 ∧ (n + 3) % 32 = 0 ∧ (n + 3) % 36 = 0 ∧ (n + 3) % 54 = 0) →
  n = 861 :=
by
  sorry

end least_number_divisible_l425_425665


namespace find_duplicated_page_number_l425_425276

noncomputable def duplicated_page_number (n : ℕ) (incorrect_sum : ℕ) : ℕ :=
  incorrect_sum - n * (n + 1) / 2

theorem find_duplicated_page_number :
  ∃ n k, (1 <= k ∧ k <= n) ∧ ( ∃ n, (1 <= n) ∧ ( n * (n + 1) / 2 + k = 2550) )
  ∧ duplicated_page_number 70 2550 = 65 :=
by
  sorry

end find_duplicated_page_number_l425_425276


namespace repeatingDecimals_fraction_eq_l425_425005

noncomputable def repeatingDecimalsSum : ℚ :=
  let x : ℚ := 1 / 3
  let y : ℚ := 4 / 99
  let z : ℚ := 5 / 999
  x + y + z

theorem repeatingDecimals_fraction_eq : repeatingDecimalsSum = 42 / 111 :=
  sorry

end repeatingDecimals_fraction_eq_l425_425005


namespace time_period_is_12_hours_l425_425163

-- Define the conditions in the problem
def birth_rate := 8 / 2 -- people per second
def death_rate := 6 / 2 -- people per second
def net_increase := 86400 -- people

-- Define the net increase per second
def net_increase_per_second := birth_rate - death_rate

-- Total time period in seconds
def time_period_seconds := net_increase / net_increase_per_second

-- Convert the time period to hours
def time_period_hours := time_period_seconds / 3600

-- The theorem we want to state and prove
theorem time_period_is_12_hours : time_period_hours = 12 :=
by
  -- Proof goes here
  sorry

end time_period_is_12_hours_l425_425163


namespace president_statements_equals_others_l425_425296

-- Definitions
inductive Inhabitant
| knight : Inhabitant
| liar : Inhabitant

def isKnight (i : Inhabitant) : Bool :=
  match i with
  | Inhabitant.knight => true
  | Inhabitant.liar => false

def acquaintance (i j : Inhabitant) : Bool := sorry  -- symmetric relation definition

-- Given total number of inhabitants
def total_inhabitants := 2016

-- President definition
def president : Inhabitant := sorry

-- Condition that each non-presidential inhabitant has two statements
def statements (i : Inhabitant) : Bool :=
  if i = president then true
  else
    (∃ (acq_knights : ℕ) (acq_liars : ℕ),
      acq_knights % 2 = 0 ∧  -- even number of knights
      acq_liars % 2 = 1      -- odd number of liars
    )

-- Theorem
theorem president_statements_equals_others :
  (∃ (pre_knights : ℕ) (pre_liars : ℕ),
    pre_knights % 2 = 0 ∧  -- even number of knights
    pre_liars % 2 = 1      -- odd number of liars
  ) :=
sorry

end president_statements_equals_others_l425_425296


namespace simplify_expression_l425_425397

variables {x y : ℝ}
-- Ensure that x and y are not zero to avoid division by zero errors.
theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y :=
sorry

end simplify_expression_l425_425397


namespace stamps_count_l425_425602

theorem stamps_count {x : ℕ} (h1 : x % 3 = 1) (h2 : x % 5 = 3) (h3 : x % 7 = 5) (h4 : 150 < x ∧ x ≤ 300) :
  x = 208 :=
sorry

end stamps_count_l425_425602


namespace cannot_obtain_original_l425_425660

-- Definition: A 100-digit number is a sequence of 100 digits (0 to 9)
def is100DigitNumber (num : List ℕ) : Prop :=
  num.length = 100 ∧ ∀ d ∈ num, d ∈ Fin.ofNat 10

-- Definition: The sum of all pairs of digits
def pairSumProduct (num : List ℕ) : ℕ :=
  let pairs := List.pairs num
  pairs.foldl (fun prod (a, b) => prod * (a + b)) 1

-- Proposition: Given a 100-digit number, the product of the sums of all possible pairs of its digits is greater than 10^100
theorem cannot_obtain_original (num : List ℕ) (h : is100DigitNumber num) :
  pairSumProduct num > 10^100 :=
by
  sorry

end cannot_obtain_original_l425_425660


namespace find_A_plus_C_l425_425549

-- This will bring in the entirety of the necessary library and supports the digit verification and operations.

-- Definitions of digits and constraints
variables {A B C D : ℕ}

-- Given conditions in the problem
def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ 
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10

def multiplication_condition_1 (A B C D : ℕ) : Prop :=
  C * D = A

def multiplication_condition_2 (A B C D : ℕ) : Prop :=
  10 * B * D + C * D = 11 * C

-- The final problem statement
theorem find_A_plus_C (A B C D : ℕ) (h1 : distinct_digits A B C D) 
  (h2 : multiplication_condition_1 A B C D) 
  (h3 : multiplication_condition_2 A B C D) : 
  A + C = 10 :=
sorry

end find_A_plus_C_l425_425549


namespace gaussian_quadrature_l425_425429

-- Define the polynomial and its integral
noncomputable def polynomial_integral {a b : ℝ} (f : ℝ → ℝ) : ℝ :=
  ∫ x in a..b, f x

-- Define the given polynomial \( f(x) = ax^3 + bx^2 + cx + d \)
def polynomial (a b c d x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

-- Main theorem statement
theorem gaussian_quadrature (a b c d : ℝ) :
  (polynomial_integral (polynomial a b c d) (-1) 1) = 
    (polynomial a b c d (-1 / Real.sqrt 3)) + 
    (polynomial a b c d (1 / Real.sqrt 3)) :=
by
  sorry

end gaussian_quadrature_l425_425429


namespace proof_problem_l425_425056

variable (b : ℝ) (hb : 0 < b ∧ b < 2)

def ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / b^2) = 1
def point_inside (x y : ℝ) := (x = sqrt 2 ∧ y = 1)
def point_on_ellipse (x y : ℝ) := ellipse b x y

def e := sqrt (1 - (b^2 / 4))

def statement_A := 0 < e ∧ e < sqrt 2 / 2
def statement_C (Q : ℝ × ℝ) (F1 F2 : ℝ × ℝ) := (∃ Q, point_on_ellipse b Q.1 Q.2) → ¬ (∃ Q, F1.1 * F2.1 + F1.2 * F2.2 = 0)

theorem proof_problem :
  (ellipse b P.1 P.2) ∧ (point_inside P.1 P.2) ∧ (∃ Q, point_on_ellipse b Q.1 Q.2) →
  statement_A ∧ statement_C :=
by
  sorry

end proof_problem_l425_425056


namespace pentagon_diagonal_product_equalities_l425_425569

theorem pentagon_diagonal_product_equalities
  (A1 A2 A3 A4 A5 : Point)
  (convex_pentagon : is_convex_pentagon A1 A2 A3 A4 A5)
  (B1 B2 B3 B4 B5 : Point)
  (h1 : B1 = intersection (line_through A2 A4) (line_through A3 A5))
  (h2 : B2 = intersection (line_through A3 A5) (line_through A4 A1))
  (h3 : B3 = intersection (line_through A4 A1) (line_through A5 A2))
  (h4 : B4 = intersection (line_through A5 A2) (line_through A1 A3))
  (h5 : B5 = intersection (line_through A1 A3) (line_through A2 A4)) :
  (distance A1 B4 * distance A2 B5 * distance A3 B1 * distance A4 B2 * distance A5 B3 = 
   distance A1 B3 * distance A2 B4 * distance A3 B5 * distance A4 B1 * distance A5 B2) ∧
  (distance A1 B5 * distance A2 B1 * distance A3 B2 * distance A4 B3 * distance A5 B4 = 
   distance A1 B2 * distance A2 B3 * distance A3 B4 * distance A4 B5 * distance A5 B1) :=
by
  sorry

end pentagon_diagonal_product_equalities_l425_425569


namespace time_to_cross_man_l425_425343

def length_of_train : ℝ := 100 -- Length of the train in meters
def speed_of_train_kmh : ℝ := 72 -- Speed of the train in km/h

noncomputable def speed_of_train_ms : ℝ := speed_of_train_kmh * (1000 / 3600)

theorem time_to_cross_man : speed_of_train_ms = 20 → 100 / speed_of_train_ms = 5 :=
by
  intros h_speed
  rw h_speed
  norm_num
  exact rfl

end time_to_cross_man_l425_425343


namespace cosine_sine_shift_l425_425303

theorem cosine_sine_shift :
    ∀ (x : ℝ), 2 * cos (2 * x - (π / 4)) = 2 * sin ((π / 2) - (2 * (x - (π / 8)))) :=
by
  sorry

end cosine_sine_shift_l425_425303


namespace reflection_of_C_over_y_eq_x_l425_425655

def point_reflection_over_yx := ∀ (A B C : (ℝ × ℝ)), 
  A = (6, 2) → 
  B = (2, 5) → 
  C = (2, 2) → 
  (reflect_y_eq_x C) = (2, 2)
where reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

theorem reflection_of_C_over_y_eq_x :
  point_reflection_over_yx :=
by 
  sorry

end reflection_of_C_over_y_eq_x_l425_425655


namespace January1_is_Monday_l425_425161

-- Definition of the problem conditions
def January : Type := {day : ℕ // 1 ≤ day ∧ day ≤ 31}
def isMonday (d : January) : Prop := sorry -- This should define a condition on what makes a day Monday
def isThursday (d : January) : Prop := sorry -- This should define a condition on what makes a day Thursday

axiom fiveMondays : ∃ s : set January, (∀ d ∈ s, isMonday d) ∧ s.card = 5
axiom fourThursdays : ∃ t : set January, (∀ d ∈ t, isThursday d) ∧ t.card = 4

-- Problem statement to be proven: January 1 is Monday given the conditions
theorem January1_is_Monday : isMonday ⟨1, nat.one_le_succ 30⟩ :=
sorry

end January1_is_Monday_l425_425161


namespace correct_props_l425_425582

variables (m n : Set ℝ → ℝ → ℝ) (α β : Set (ℝ → ℝ → ℝ))

-- Definition of parallel and perpendicular relationships
def parallel (l : Set ℝ → ℝ → ℝ) (p : Set (ℝ → ℝ → ℝ)) : Prop := sorry
def perpendicular (l : Set ℝ → ℝ → ℝ) (p : Set (ℝ → ℝ → ℝ)) : Prop := sorry

-- Propositions
def prop_3 (m α β : Set (ℝ → ℝ → ℝ)) : Prop := perpendicular m α → parallel m β → perpendicular α β
def prop_4 (m α β : Set (ℝ → ℝ → ℝ)) : Prop := perpendicular m α → parallel α β → perpendicular m β

-- The proof problem
theorem correct_props (m n : Set (ℝ → ℝ → ℝ)) (α β : Set (ℝ → ℝ → ℝ)) :
  prop_3 m α β ∧ prop_4 m α β :=
by sorry

end correct_props_l425_425582


namespace paulina_convertibles_l425_425596

-- Definitions for conditions
def total_cars : ℕ := 125
def percentage_regular_cars : ℚ := 64 / 100
def percentage_trucks : ℚ := 8 / 100
def percentage_convertibles : ℚ := 1 - (percentage_regular_cars + percentage_trucks)

-- Theorem to prove the number of convertibles
theorem paulina_convertibles : (percentage_convertibles * total_cars) = 35 := by
  sorry

end paulina_convertibles_l425_425596


namespace complex_z_in_fourth_quadrant_l425_425882

noncomputable def complex_z : ℂ := (1 - 2 * complex.i) / (3 - complex.i)

theorem complex_z_in_fourth_quadrant :
    ∃ z : ℂ, (3 - complex.i) * z = 1 - 2 * complex.i ∧ z.re > 0 ∧ z.im < 0 := 
by
  use complex_z
  have h : (3 - complex.i) * complex_z = 1 - 2 * complex.i := by 
    sorry
  split
  exact h
  split
  calc
    complex_z.re = (↑1 + (↑1 / ↑3)*(- 2*complex.i).re) : sorry
    _ > 0 : by norm_num
  calc
    complex_z.im = (↑1 / ↑3 * (↑2) * -↑1) : sorry
    _ < 0 : by norm_num

end complex_z_in_fourth_quadrant_l425_425882


namespace remainder_of_division_l425_425773

noncomputable def dividend : Polynomial ℤ := Polynomial.C 1 * Polynomial.X^4 +
                                             Polynomial.C 3 * Polynomial.X^2 + 
                                             Polynomial.C (-4)

noncomputable def divisor : Polynomial ℤ := Polynomial.C 1 * Polynomial.X^3 +
                                            Polynomial.C (-3)

theorem remainder_of_division :
  Polynomial.modByMonic dividend divisor = Polynomial.C 3 * Polynomial.X^2 +
                                            Polynomial.C 3 * Polynomial.X +
                                            Polynomial.C (-4) :=
by
  sorry

end remainder_of_division_l425_425773


namespace digit_81_in_decimal_of_fraction_325_999_l425_425328

theorem digit_81_in_decimal_of_fraction_325_999 : 
  let decimal := (325 : ℚ) / 999 in 
  (decimal.floor_decimal_digit 81) = 5 := 
by 
  sorry

end digit_81_in_decimal_of_fraction_325_999_l425_425328


namespace equal_probability_among_children_l425_425662

theorem equal_probability_among_children
    (n : ℕ := 100)
    (p : ℝ := 0.232818)
    (k : ℕ := 18)
    (h_pos : 0 < p)
    (h_lt : p < 1)
    (num_outcomes : ℕ := 2^k) :
  ∃ (dist : Fin n → Fin num_outcomes),
    ∀ i : Fin num_outcomes, ∃ j : Fin n, dist j = i ∧ p ^ k * (1 - p) ^ (num_outcomes - k) = 1 / n :=
by
  sorry

end equal_probability_among_children_l425_425662


namespace abs_diff_squares_110_108_l425_425315

theorem abs_diff_squares_110_108 : abs ((110 : ℤ)^2 - (108 : ℤ)^2) = 436 := by
  sorry

end abs_diff_squares_110_108_l425_425315


namespace problem_statement_l425_425087

noncomputable
def ellipse_c_eq : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ a^2 = 4 ∧ b = 1 ∧ (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)

lemma line_MN_through_E (m : ℝ) : 
  let P := (4, m),
      A := (-2, 0),
      B := (2, 0),
      E := (1, 0) in
  ∀ x₁ y₁ x₂ y₂ : ℝ, 
    (y₁ = (m / 6) * (x₁ + 2)) ∧ (x₁^2 + 4*y₁^2 = 4) ∧
    (y₂ = (m / 2) * (x₂ - 2)) ∧ (x₂^2 + 4*y₂^2 = 4) →
  collinear { (x₁, y₁), (x₂, y₂), E } :=
begin
  sorry
end

theorem problem_statement : ellipse_c_eq ∧ ∀ m : ℝ, line_MN_through_E m :=
by sorry

end problem_statement_l425_425087


namespace area_triangle_AOB_eq_4_find_circle_eq_min_pb_pq_l425_425076

variable (t : ℝ) (h₀ : t ≠ 0)

def circle_eq (x y : ℝ) : Prop :=
  (x - t) ^ 2 + (y - (2 / t)) ^ 2 = t ^ 2 + 4 / t ^ 2 

variable (x y : ℝ) 
def intersect_x_axis (hx : y = 0) : Prop := 
  x = 0 ∨ x = 2 * t

def intersect_y_axis (hy : x = 0) : Prop := 
  y = 0 ∨ y = 4 / t

theorem area_triangle_AOB_eq_4 :
  ∃ (A B : ℝ × ℝ), intersect_x_axis t (A.1) (A.2) ∧ intersect_y_axis t (B.1) (B.2) ∧ 
    (1/2 * (real.abs (A.1)) * (real.abs (B.2))) = 4 :=
by 
  sorry

theorem find_circle_eq (h1: t = 2 ∨ t = -2) :
  (circle_eq 2 1 ∧ (circle_eq (-2) (-1) → False)) ∧
  ((∀ h2 : t = 2, circle_eq 2 1) ∧ (∀ h3 : t = -2, (circle_eq (-2) (-1) → False))) :=
by
  sorry

theorem min_pb_pq (h: t > 0) :
  ∀ (l : ℝ × ℝ → ℝ), (l (P:ℝ × ℝ) := x + y + 2 = 0) → ∃ Q, 
  (|PB| + |PQ| >= |B'Q|) ∧ (|B'C| - r = 2 * sqrt 5) → (2 * sqrt 5 ∧ P = (-4/3, -2/3)) :=
by
  sorry

end area_triangle_AOB_eq_4_find_circle_eq_min_pb_pq_l425_425076


namespace I_0_eq_I_m1_eq_I_2_eq_I_1_eq_I_np2_eq_I_m3_eq_I_m2_eq_I_3_eq_integral_sqrt_eq_integral_inv_squared_eq_l425_425946

section
variable (n : ℤ)

-- Definitions for the integrals
noncomputable def I (n : ℤ) := ∫ x in 0..(Real.pi / 4), (Real.cos x)^(-n)

-- Part (1)
theorem I_0_eq : I 0 = Real.pi / 4 :=
by sorry

theorem I_m1_eq : I (-1) = Real.sqrt 2 / 2 :=
by sorry

theorem I_2_eq : I 2 = 1 :=
by sorry

-- Part (2)
theorem I_1_eq : I 1 = Real.log (Real.sqrt 2 + 1) :=
by sorry

-- Part (3)
theorem I_np2_eq : I (n + 2) = (n + 1) / n * I n :=
by sorry

-- Part (4)
theorem I_m3_eq : I (-3) = Real.sqrt 2 / 3 :=
by sorry

theorem I_m2_eq : I (-2) = Real.pi / 8 :=
by sorry

theorem I_3_eq : I 3 = 2 * Real.log (Real.sqrt 2 + 1) :=
by sorry

-- Part (5)
theorem integral_sqrt_eq : ∫ x in 0..1, Real.sqrt (x^2 + 1) = 2 * Real.log (Real.sqrt 2 + 1) :=
by sorry

theorem integral_inv_squared_eq : ∫ x in 0..1, 1 / (x^2 + 1)^2 = Real.pi / 8 :=
by sorry
end

end I_0_eq_I_m1_eq_I_2_eq_I_1_eq_I_np2_eq_I_m3_eq_I_m2_eq_I_3_eq_integral_sqrt_eq_integral_inv_squared_eq_l425_425946


namespace repeating_decimals_sum_l425_425010

theorem repeating_decimals_sum : 
  (0.3333333333333333 : ℝ) + (0.0404040404040404 : ℝ) + (0.005005005005005 : ℝ) = (14 / 37 : ℝ) :=
by {
  sorry
}

end repeating_decimals_sum_l425_425010


namespace problem_statement_l425_425167

noncomputable def acute_triangle (A B C : Type) [metric_space A] (triangle : is_triangle A B C) : Prop :=
  acute_triangle A B C

def incenter (I : Type) (triangle : is_triangle A B C) : Prop :=
  is_incenter I triangle

def foot_perpendicular (D I : Type) (triangle : is_triangle A B C) : Prop :=
  foot I D (side BC triangle) ∧ perpendicular I D (side BC triangle)

def altitude_meeting (A H B I C : Type) (P Q : Type) : Prop :=
  altitude A H ∧ meets_at_altitude P Q B I C

def circumcenter (O : Type) (triangle : is_triangle P Q I) : Prop :=
  circumcenter O triangle

def extend_to_meet (A O BC : Type) (L : Type) : Prop :=
  extends AO_to_meet A O BC L

def circumcircle_meets_again (AIL BC : Type) (N : Type) : Prop :=
  circumcircle_meets AIL BC N

theorem problem_statement (A B C I D P Q O L N : Type)
  [metric_space A] [metric_space B] [metric_space C]
  [acute_triangle A B C]
  [incenter I (is_triangle A B C)]
  [foot_perpendicular D I (is_triangle A B C)]
  [altitude_meeting A H B I C P Q]
  [circumcenter O (is_triangle P Q I)]
  [extend_to_meet A O (side BC (is_triangle A B C)) L]
  [circumcircle_meets_again (triangle A I L) (side BC (is_triangle A B C)) N] :
  ∃ (BD CD BN CN : ℝ), 
  BD / CD = BN / CN := sorry

end problem_statement_l425_425167


namespace num_triangles_square_even_num_triangles_rect_even_l425_425674

-- Problem (a): Proving that the number of triangles is even 
theorem num_triangles_square_even (a : ℕ) (n : ℕ) (h : a * a = n * (3 * 4 / 2)) : 
  n % 2 = 0 :=
sorry

-- Problem (b): Proving that the number of triangles is even
theorem num_triangles_rect_even (L W k : ℕ) (hL : L = k * 2) (hW : W = k * 1) (h : L * W = k * 1 * 2 / 2) :
  k % 2 = 0 :=
sorry

end num_triangles_square_even_num_triangles_rect_even_l425_425674


namespace min_expr_value_l425_425880

theorem min_expr_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) :
  ∃ m, m = real.Inf { x | ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 1 ∧ x = 1 / a + 4 / b } ∧ m = 9 :=
by
  sorry

end min_expr_value_l425_425880


namespace complex_quadrant_l425_425810

theorem complex_quadrant (z : ℂ) (hz : z = (i / (1 - i))^2) :
    let w := 2 + z in w.im < 0 ∧ w.re > 0 :=
by
  sorry

end complex_quadrant_l425_425810


namespace largest_set_size_l425_425720

noncomputable def P : Set (ℤ × ℤ × ℤ) :=
{t | let (a, b, c) := t in a < 7 ∧ b < 7 ∧ c < 7 ∧
     (nat.prime a ∨ nat.prime b ∨ nat.prime c) ∧ 
     a + b > c ∧ a + c > b ∧ b + c > a ∧
     ∀ (u v w : ℤ), (u = a ∧ v = b ∧ w = c) →
     ∀ (x y z : ℤ), (x = a ∧ y = b ∧ z = c) → 
     (u, v, w) ≠ (x, y, z) ∧ (u, v, w) ≠ (x, y, z) ∧ (u, v, w) ≠ (y, z, x)}

theorem largest_set_size : ∃ P : Set (ℤ × ℤ × ℤ), 
  (∀ t : (ℤ × ℤ × ℤ), t ∈ P → let (a, b, c) := t in a < 7 ∧ b < 7 ∧ c < 7 ∧
     (nat.prime a ∨ nat.prime b ∨ nat.prime c) ∧ 
     a + b > c ∧ a + c > b ∧ b + c > a ∧
     ∀ (u v w : ℤ), (u = a ∧ v = b ∧ w = c) →
     ∀ (x y z : ℤ), (x = a ∧ y = b ∧ z = c) → 
     (u, v, w) ≠ (x, y, z) ∧ (u, v, w) ≠ (x, y, z) ∧ (u, v, w) ≠ (y, z, x)) ∧
  (∀ Q : Set (ℤ × ℤ × ℤ), 
     (∀ t : (ℤ × ℤ × ℤ), t ∈ Q → let (a, b, c) := t in a < 7 ∧ b < 7 ∧ c < 7 ∧
        (nat.prime a ∨ nat.prime b ∨ nat.prime c) ∧ 
        a + b > c ∧ a + c > b ∧ b + c > a ∧
        ∀ (u v w : ℤ), (u = a ∧ v = b ∧ w = c) →
        ∀ (x y z : ℤ), (x = a ∧ y = b ∧ z = c) → 
        (u, v, w) ≠ (x, y, z) ∧ (u, v, w) ≠ (x, y, z) ∧ (u, v, w) ≠ (y, z, x)) → 
     Q.card ≤ P.card) ∧ 
  P.card = 10 :=
sorry

end largest_set_size_l425_425720


namespace combined_age_l425_425971

def hezekiah_age : ℕ := 4
def ryanne_age : ℕ := hezekiah_age + 7
def jamison_age : ℕ := 2 * hezekiah_age

theorem combined_age :
  ryanne_age = 7 + hezekiah_age ∧
  hezekiah_age + (hezekiah_age + 7) = 15 ∧
  jamison_age = 2 * hezekiah_age →
  ryanne_age + hezekiah_age + jamison_age = 23 :=
by {
  intro h,
  cases h with h_ryanne h_rest,
  cases h_rest with h_sum h_jamison,
  simp [hezekiah_age, ryanne_age, jamison_age],
  sorry
}

end combined_age_l425_425971


namespace multiples_of_7_between_15_and_225_l425_425865

theorem multiples_of_7_between_15_and_225 : 
  {n : ℕ | ∃ k : ℕ, n = 7 * k ∧ 15 < n ∧ n < 225}.card = 30 := 
by 
  sorry

end multiples_of_7_between_15_and_225_l425_425865


namespace sequence_term_position_l425_425342

theorem sequence_term_position :
  ∃ n : ℕ, ∀ k : ℕ, (k = 7 + 6 * (n - 1)) → k = 2005 → n = 334 :=
by
  sorry

end sequence_term_position_l425_425342


namespace race_track_width_l425_425265

noncomputable def inner_radius (C : ℝ) : ℝ :=
  C / (2 * Real.pi)

noncomputable def track_width (C : ℝ) (R : ℝ) : ℝ :=
  R - inner_radius C

theorem race_track_width
  (C : ℝ := 440)
  (R : ℝ := 84.02817496043394)
  (W : ℝ := 14.02056077700799) :
  track_width C R ≈ W :=
by
  sorry

end race_track_width_l425_425265


namespace angle_between_vectors_l425_425786

-- Definitions of vectors
def u : ℝ × ℝ := (4, -1)
def v : ℝ × ℝ := (6, 8)

-- Function to calculate the dot product of two vectors
def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

-- Function to calculate the norm of a vector
def norm (a : ℝ × ℝ) : ℝ := Real.sqrt (a.1 * a.1 + a.2 * a.2)

-- Theorem to prove the angle between vectors u and v
theorem angle_between_vectors :
  acos ((dot_product u v) / (norm u * norm v)) = Real.arccos (8 * Real.sqrt 17 / 85) := by
  sorry

end angle_between_vectors_l425_425786


namespace betty_eggs_used_l425_425318

-- Conditions as definitions
def ratio_sugar_cream_cheese (sugar cream_cheese : ℚ) : Prop :=
  sugar / cream_cheese = 1 / 4

def ratio_vanilla_cream_cheese (vanilla cream_cheese : ℚ) : Prop :=
  vanilla / cream_cheese = 1 / 2

def ratio_eggs_vanilla (eggs vanilla : ℚ) : Prop :=
  eggs / vanilla = 2

-- Given conditions
def sugar_used : ℚ := 2 -- cups of sugar

-- The statement to prove
theorem betty_eggs_used (cream_cheese vanilla eggs : ℚ) 
  (h1 : ratio_sugar_cream_cheese sugar_used cream_cheese)
  (h2 : ratio_vanilla_cream_cheese vanilla cream_cheese)
  (h3 : ratio_eggs_vanilla eggs vanilla) :
  eggs = 8 :=
sorry

end betty_eggs_used_l425_425318


namespace demolition_time_l425_425758

variable (a b c : ℚ)

-- Given Conditions
def condition1 : Prop := b + c = 1/6
def condition2 : Prop := a + b = 1/3
def condition3 : Prop := a + c = 1/5
def condition4 : Prop := 2 * (a + b + c) = (1/3) + (1/6) + (1/5)

-- Total Time Calculation
def total_time : ℚ := 1 + (13/20) / (1/3)

theorem demolition_time 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3) 
  (h4 : condition4) :
  1 + (13/20) / (1/3) = 59/20 := 
sorry

end demolition_time_l425_425758


namespace sum_f_even_result_l425_425574

def g (k x : ℕ) := x^2 - k

def f (n : ℕ) := (2*n+1)^2 - 2*n

noncomputable def sum_f_even (n : ℕ) : ℕ := (Finset.range n).sum (λ n, f (2 * (n + 1)))

theorem sum_f_even_result :
  sum_f_even 1006 = 8124 ∨
  sum_f_even 1006 = 8136 ∨
  sum_f_even 1006 = 8148 ∨
  sum_f_even 1006 = 8160 ∨
  sum_f_even 1006 = 8172 :=
sorry

end sum_f_even_result_l425_425574


namespace triangle_ABC_angles_l425_425605

-- Define the problem conditions
variables (A B C A1 B1 C1 H : Type)
variables (incircle_touches_BC : Incircle A1 B1 C1 touches BC)
variables (angle_A_eq_50 : angle A = 50)

-- conditions stating that A1, B1, C1 are intersections with circumcircle
-- and that H is the orthocenter
variables (is_orthocenter : Orthocenter H)
variables (circumcircle_intersections : CircumcircleIntersections A1 B1 C1)

-- statement of the problem
theorem triangle_ABC_angles {A B C : Type} 
  (h1 : angle A = 50) 
  (h2 : IsAcuteAngledTriangle A B C) 
  (h3 : PointsOnCircumcircle A B C A1 B1 C1)
  (h4 : IncircleTouchesSide A1 B1 C1 BC) :
  (angle B = 60) ∧ (angle C = 70) := 
sorry

end triangle_ABC_angles_l425_425605


namespace find_parabola_eq_circle_through_fixed_points_l425_425170

open Real 

section parabola_problem

variable (p : ℝ) (h_pos : p > 0)

def parabola_eq (y x : ℝ) : Prop := y^2 = 2 * p * x 

theorem find_parabola_eq : parabola_eq 2 4 ↔ p = 2 :=
  sorry 

theorem circle_through_fixed_points :
  let C := fun (x y : ℝ) => y^2 = 4 * x,
      A := fun (k : ℝ) => (1 / k^2, 2 / k),
      l := fun (x : ℝ) => - 1,
      tangent := fun (k x : ℝ) => k * x + (1 / k),
      N : ℝ × ℝ := (-1, 1 / k - k),
      M : ℝ × ℝ := (-1, -2 * k)
  in ∀ (k : ℝ), ∃ (circle_center : ℝ × ℝ), 
      (circle_center = N ∧ ∃ r : ℝ, 
      let circle_eq := fun (x y : ℝ) => (x + 1)^2 + (y - 1 / k + k)^2 = (k + 1 / k)^2
      in circle_eq 1 0 ∧ circle_eq (-3) 0) :=
  sorry 

end parabola_problem 

end find_parabola_eq_circle_through_fixed_points_l425_425170


namespace skateboarder_speed_l425_425980

theorem skateboarder_speed (d t : ℕ) (ft_per_mile hr_to_sec : ℕ)
  (h1 : d = 660) (h2 : t = 30) (h3 : ft_per_mile = 5280) (h4 : hr_to_sec = 3600) :
  ((d / t) / ft_per_mile) * hr_to_sec = 15 :=
by sorry

end skateboarder_speed_l425_425980


namespace f_800_value_l425_425192

theorem f_800_value (f : ℝ → ℝ) (f_condition : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y) (f_400 : f 400 = 4) : f 800 = 2 :=
  sorry

end f_800_value_l425_425192


namespace measure_angle_R_l425_425915

theorem measure_angle_R (P Q R : Type) (angle_P angle_Q angle_R : R) :
  angle_P = 88 ∧ 
  angle_Q = 2 * angle_R + 18 ∧ 
  angle_P + angle_Q + angle_R = 180 
  → angle_R = 74 / 3 :=
by
  intros h
  cases h with hP hQR
  cases hQR with hQ h_sum
  sorry

end measure_angle_R_l425_425915


namespace find_a7_l425_425542

variables {a : ℕ → ℤ}
variable (h_arith_seq : ∀ n m : ℕ, a (n + m) = 2*a((n + m) / 2))
variable (h_a2 : a 2 = 2)
variable (h_sum_a4_a5 : a 4 + a 5 = 12)

theorem find_a7 : a 7 = 10 := by
  sorry

end find_a7_l425_425542


namespace three_digit_perfect_squares_divisible_by_4_count_l425_425130

theorem three_digit_perfect_squares_divisible_by_4_count : 
  (finset.filter (λ n, (100 ≤ n ∧ n ≤ 999) ∧ (∃ k, n = k * k) ∧ (n % 4 = 0)) (finset.range 1000)).card = 10 :=
by
  sorry

end three_digit_perfect_squares_divisible_by_4_count_l425_425130


namespace william_library_visits_l425_425922

variable (W : ℕ) (J : ℕ)
variable (h1 : J = 4 * W)
variable (h2 : 4 * J = 32)

theorem william_library_visits : W = 2 :=
by
  sorry

end william_library_visits_l425_425922


namespace simplify_trig_expression_l425_425619

-- Define the variables and identities
variables {α θ : Real}

-- State all conditions as hypotheses
theorem simplify_trig_expression :
  (sin (2 * Real.pi - α))^2 + (cos (Real.pi + α) * cos (Real.pi - α)) + 1 = 2 :=
by
  have h1 : sin (2 * Real.pi - α) = sin α := sorry
  have h2 : cos (Real.pi + α) = - cos α := sorry
  have h3 : cos (Real.pi - α) = - cos α := sorry
  have h4 : sin^2 θ + cos^2 θ = 1 := sorry
  sorry

end simplify_trig_expression_l425_425619


namespace number_of_rectangles_is_24_l425_425867

-- Define the rectangles on a 1x5 stripe
def rectangles_1x5 : ℕ := 1 + 2 + 3 + 4 + 5

-- Define the rectangles on a 1x4 stripe
def rectangles_1x4 : ℕ := 1 + 2 + 3 + 4

-- Define the overlap (intersection) adjustment
def overlap_adjustment : ℕ := 1

-- Total number of rectangles calculation
def total_rectangles : ℕ := rectangles_1x5 + rectangles_1x4 - overlap_adjustment

theorem number_of_rectangles_is_24 : total_rectangles = 24 := by
  sorry

end number_of_rectangles_is_24_l425_425867


namespace sum_of_exponents_l425_425511

def like_terms (m n : ℕ) : Prop :=
  ∃ a b c : ℕ, 3 * a^m * b * c^2 = -2 * a^3 * b^n * c^2

theorem sum_of_exponents (m n : ℕ) (h : like_terms m n) : m + n = 4 :=
by
  sorry

end sum_of_exponents_l425_425511


namespace projection_ratio_l425_425941

variables {V : Type*} [inner_product_space ℝ V] (v w : V)
variables (p : V) (h₁ : p = (⟪v, w⟫ / ⟪w, w⟫) • w)
variables (q : V) (h₂ : q = (⟪w, p⟫ / ⟪p, p⟫) • p)
variables (hpw : ∥p∥ / ∥w∥ = 3 / 4)

theorem projection_ratio :
  ∥q∥ / ∥p∥ = 3 / 4 :=
sorry

end projection_ratio_l425_425941


namespace competition_mode_and_median_l425_425165

-- Define the given scores
def scores : List ℝ := [95, 97, 96, 97, 99, 98]

-- Define the mode function for a list of real numbers
def mode (lst : List ℝ) : Option ℝ := 
  lst.groupBy id 
  |> List.maxBy (λ g => g.length) 
  |> Option.map List.head' 
  |> Option.join

-- Define the median function for a list of real numbers
def median (lst : List ℝ) : ℝ :=
  let sorted := lst.qsort (≤)
  if sorted.length % 2 = 0 then
    let mid1 := sorted.get! (sorted.length / 2 - 1)
    let mid2 := sorted.get! (sorted.length / 2)
    (mid1 + mid2) / 2
  else
    sorted.get! (sorted.length / 2)

-- The theorem stating the problem
theorem competition_mode_and_median : 
  mode scores = some 97 ∧ median scores = 97 := 
by
  sorry

end competition_mode_and_median_l425_425165


namespace num_perfect_square_factors_1728_l425_425278

theorem num_perfect_square_factors_1728 :
  let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n,
  is_factor (n d : ℕ) := d ∣ n,
  factors (n : ℕ) := { d : ℕ | is_factor n d },
  perfect_square_factors (n : ℕ) := { d : ℕ | is_factor n d ∧ is_perfect_square d },
  prime_factors_1728 := factors 1728,
  perfect_square_factors_1728 := perfect_square_factors 1728
  in
  perfect_square_factors_1728.to_finset.card = 8 :=
sorry

end num_perfect_square_factors_1728_l425_425278


namespace find_area_constant_l425_425369

noncomputable def projectile_curve_area_constant (v g : ℝ) : ℝ :=
  let c := (π / 8)
  in c * (v * v * v * v) / (g * g)

theorem find_area_constant (v g : ℝ) (hv : v > 0) (hg : g > 0) : 
  projectile_curve_area_constant v g = π / 8 * (v ^ 4 / g ^ 2) :=
by sorry

end find_area_constant_l425_425369


namespace remainder_sum_mod9_l425_425757

theorem remainder_sum_mod9 :
  ((2469 + 2470 + 2471 + 2472 + 2473 + 2474) % 9) = 6 := 
by 
  sorry

end remainder_sum_mod9_l425_425757


namespace binom_two_eq_l425_425663

theorem binom_two_eq (n : ℕ) (h : n ≥ 2) : nat.choose n 2 = n * (n - 1) / 2 :=
by sorry

end binom_two_eq_l425_425663


namespace angle_A_is_pi_over_3_sin_B_sin_C_l425_425154

variable (A B C a b c : ℝ) 

-- Conditions
axiom cos2A_minus_3cosBC_eq_1 : cos (2 * A) - 3 * cos (B + C) = 1
axiom area_eq_5sqrt3 : let S := (1 / 2) * b * c * sin A; S = 5 * Real.sqrt 3
axiom side_b_eq_5 : b = 5
axiom sides_a_b_c : a = Real.sqrt (b^2 + c^2 - 2*b*c*cos A)

-- Question 1: Finding measure of angle A
def angle_A_eq_pi_over_3 : Prop :=
  A = Real.pi / 3

-- Question 2: Finding value of sin B sin C
def sin_B_sin_C_eq_5_over_7 : Prop :=
  sin B * sin C = 5 / 7

-- Theorem statements
theorem angle_A_is_pi_over_3 
    (cos2A_minus_3cosBC_eq_1 : cos (2 * A) - 3 * cos (B + C) = 1) : 
    angle_A_eq_pi_over_3 A := by 
  sorry

theorem sin_B_sin_C 
    (area_eq_5sqrt3 : let S := (1 / 2) * b * c * sin A; S = 5 * Real.sqrt 3)
    (side_b_eq_5 : b = 5) : 
    sin_B_sin_C_eq_5_over_7 A B C a b c := by 
  sorry

end angle_A_is_pi_over_3_sin_B_sin_C_l425_425154


namespace complement_union_sets_l425_425489

open Set

theorem complement_union_sets :
  ∀ (U A B : Set ℕ), (U = {1, 2, 3, 4}) → (A = {2, 3}) → (B = {3, 4}) → (U \ (A ∪ B) = {1}) :=
by
  intros U A B hU hA hB
  rw [hU, hA, hB]
  simp 
  sorry

end complement_union_sets_l425_425489


namespace base_seven_sum_l425_425024

def base_seven_sum_of_product (n m : ℕ) : ℕ :=
  let product := n * m
  let digits := product.digits 7
  digits.sum

theorem base_seven_sum (k l : ℕ) (hk : k = 5 * 7 + 3) (hl : l = 343) :
  base_seven_sum_of_product k l = 11 := by
  sorry

end base_seven_sum_l425_425024


namespace beast_of_war_running_time_correct_l425_425285

def running_time_millennium : ℕ := 120

def running_time_alpha_epsilon (rt_millennium : ℕ) : ℕ := rt_millennium - 30

def running_time_beast_of_war (rt_alpha_epsilon : ℕ) : ℕ := rt_alpha_epsilon + 10

theorem beast_of_war_running_time_correct :
  running_time_beast_of_war (running_time_alpha_epsilon running_time_millennium) = 100 := by sorry

end beast_of_war_running_time_correct_l425_425285


namespace johns_money_received_l425_425925

-- Let X be the amount of money John received from his uncle
def X : ℝ := 100  -- This is the answer we aim to prove

-- Defining the conditions
def gave_to_jenna (x : ℝ) : ℝ := x * (1/4)
def bought_groceries : ℝ := 40
def remaining_money : ℝ := 35

-- John should have had 3/4 of the money before buying groceries
def had_before_groceries (x : ℝ) : ℝ := 3/4 * x

-- Prove the problem statement
theorem johns_money_received (x : ℝ) (h : remaining_money + bought_groceries = had_before_groceries x) : x = X :=
by
  sorry

end johns_money_received_l425_425925


namespace ellipse_trace_l425_425993

theorem ellipse_trace (z : ℂ) (hz : abs z = 3) : 
  ∃ a b : ℝ, (z = a + b * complex.I ∧ (a^2 + b^2 = 9) ∧
  (let x := (10 / 9) * a in 
  let y := (8 / 9) * b in
  x^2 / (10 / 3)^2 + y^2 / (8 / 3)^2 = 1)) := sorry

end ellipse_trace_l425_425993


namespace ribbon_length_per_gift_l425_425972

theorem ribbon_length_per_gift (gifts : ℕ) (initial_ribbon remaining_ribbon : ℝ) (total_used_ribbon : ℝ) (length_per_gift : ℝ):
  gifts = 8 →
  initial_ribbon = 15 →
  remaining_ribbon = 3 →
  total_used_ribbon = initial_ribbon - remaining_ribbon →
  length_per_gift = total_used_ribbon / gifts →
  length_per_gift = 1.5 :=
by
  intros
  sorry

end ribbon_length_per_gift_l425_425972


namespace sequence_sum_expression_l425_425485

-- Define the sequence and the sum of the first n terms
def a : ℕ → ℕ
| 0 := 0            -- a_0 is a dummy value as a_1 is given to be 2
| 1 := 2
| (n + 2) := 2 * S (n + 1) + 2 * (n + 1) + 2

def S : ℕ → ℕ
| 0 := 0            -- Sum of the first 0 terms is 0
| (n + 1) := S n + a (n + 1)

-- The theorem to prove
theorem sequence_sum_expression (n : ℕ) (h_pos : n ≠ 0) : S n = (3^((Int.ofNat n).toNat) - 1) - n := by
  sorry


end sequence_sum_expression_l425_425485


namespace hexagon_sides_6_units_l425_425415

theorem hexagon_sides_6_units (a b c d e f : ℕ) : 
  (a = 5) → (b = 6) → (a + b + c + d + e + f = 34) → (a = b ∨ b = c ∨ c = d ∨ d = e ∨ e = f ∨ f = a ∨ a = c ∨ b = d ∨ c = e ∨ d = f ∨ e = a ∨ f = b) → 
  (0 < (6 - a) + (6 - b) + (6 - c) + (6 - d) + (6 - e) + (6 - f)) → (a + b + c + d + e + f - ((6 - a) + (6 - b) + (6 - c) + (6 - d) + (6 - e) + (6 - f)) = 4) :=
begin
  intros ha hb hp hlen hsides,
  sorry -- proof not required, so using sorry to complete the theorem
end

end hexagon_sides_6_units_l425_425415


namespace length_more_than_breadth_by_10_l425_425270

-- Definitions based on conditions
def length : ℕ := 55
def cost_per_meter : ℚ := 26.5
def total_fencing_cost : ℚ := 5300
def perimeter : ℚ := total_fencing_cost / cost_per_meter

-- Calculate breadth (b) and difference (x)
def breadth := 45 -- This is inferred manually from the solution for completeness
def difference (b : ℚ) := length - b

-- The statement we need to prove
theorem length_more_than_breadth_by_10 :
  difference 45 = 10 :=
by
  sorry

end length_more_than_breadth_by_10_l425_425270


namespace parallel_vectors_m_eq_neg3_l425_425862

theorem parallel_vectors_m_eq_neg3
  (m : ℝ)
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h1 : a = (m + 1, -3))
  (h2 : b = (2, 3))
  (h3 : ∃ k : ℝ, a = (k * b.1, k * b.2)) :
  m = -3 := 
sorry

end parallel_vectors_m_eq_neg3_l425_425862


namespace arithmetic_sequence_general_term_b_series_sum_l425_425908

noncomputable def isArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a(n+1) = a n + d

noncomputable def isGeometricSequence (a b c : ℕ) : Prop :=
  b^2 = a * c

theorem arithmetic_sequence_general_term (a : ℕ → ℕ) (a1 : a 1 = 1)
  (geo_seq : isGeometricSequence (a 1) (a 2) (a 5))
  (not_trivial : ∀ r, r ≠ 1) :
  isArithmeticSequence a ∧ ∃ d, d ≠ 0 ∧ ∀ n, a n = 2n - 1 :=
by 
  sorry

theorem b_series_sum (a : ℕ → ℕ) (b : ℕ → ℕ) (Sn : ℕ → ℕ)
  (a1 : a 1 = 1)
  (geo_seq : isGeometricSequence (a 1) (a 2) (a 5))
  (not_trivial : ∀ r, r ≠ 1)
  (an_formula : ∀ n, a n = 2n - 1)
  (bn_def : ∀ n, b n = 1 / (a n * a (n + 1)))
  (Sn_def : ∀ n, Sn n = ∑ i in range n, b (i + 1)):
  Sn n = n / (2n + 1) :=
by 
  sorry

end arithmetic_sequence_general_term_b_series_sum_l425_425908


namespace max_m_value_l425_425202

theorem max_m_value (n : ℕ) (h1 : 1 < n) (m : ℕ) 
  (h_members : ∃ mn : ℕ, mn = m * n) 
  (h_commissions : ∃ k : ℕ, k = 2 * n ∧ ∀ (i : ℕ) (hi : i < k), ∃ c : set ℕ, c.card = m)
  (h_member_commissions : ∀ (d : ℕ) (hd : d ∈ finset.range (m * n)), ∃ s : finset ℕ, s.card = 2 ∧ ∀ j ∈ s, j < 2 * n)
  (h_commission_inter : ∀ (i j : ℕ) (hi : i < 2 * n) (hj : j < 2 * n) (hij : i ≠ j), 
    (finset.filter (λ (d : ℕ), d ∈ finset.univ.filter (λ x, d ∈ (commissions i)) ∧ d ∈ (commissions j)) finset.univ).card ≤ 1) :
  m ≤ 2 * n - 1 :=
sorry

end max_m_value_l425_425202


namespace find_a_for_extremum_at_neg3_l425_425856

theorem find_a_for_extremum_at_neg3 (a : ℝ) :
  (∀ x : ℝ, (derivative (λ x, x^3 + a * x^2 + 3 * x - 9)) x = 3 * x^2 + 2 * a * x + 3) →
  (derivative (λ x, x^3 + a * x^2 + 3 * x - 9)) (-3) = 0 →
  a = 5 :=
by
  -- Proof will go here
  sorry

end find_a_for_extremum_at_neg3_l425_425856


namespace shifted_graph_correct_l425_425974

noncomputable def original_function (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

noncomputable def shifted_function (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 3)

theorem shifted_graph_correct :
  ∀ (x : ℝ), (∃ y : ℝ, y = original_function (x - Real.pi / 4)) -> shifted_function x = original_function (x - Real.pi / 4) :=
by
  intro x h
  cases h with y hy
  sorry

end shifted_graph_correct_l425_425974


namespace train_crossing_time_l425_425107

noncomputable def time_to_cross_bridge (train_length : ℝ) (bridge_length : ℝ) (speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let speed_mps := speed_kmph * (1000 / 3600)
  total_distance / speed_mps

theorem train_crossing_time : 
  time_to_cross_bridge 110 200 60 ≈ 18.60 := 
by
  sorry

end train_crossing_time_l425_425107


namespace males_not_listening_l425_425725

theorem males_not_listening:
  ∀ (total_listeners males_listening female_non_listeners total_surveyed females_listening : ℕ),
    total_listeners = 200 →
    males_listening = 75 →
    female_non_listeners = 120 →
    total_surveyed = total_listeners + female_non_listeners + (380 - (120 + 125)) →
    females_listening = total_listeners - males_listening →
    (380 - (female_non_listeners + females_listening) - males_listening) = 185 :=
begin
  intros,
  simp,
  sorry
end 

end males_not_listening_l425_425725


namespace half_radius_of_circle_y_l425_425759

theorem half_radius_of_circle_y (Cx Cy : ℝ) (r_x r_y : ℝ) 
  (h1 : Cx = 10 * π) 
  (h2 : Cx = 2 * π * r_x) 
  (h3 : π * r_x ^ 2 = π * r_y ^ 2) :
  (1 / 2) * r_y = 2.5 := 
by
-- sorry skips the proof
sorry

end half_radius_of_circle_y_l425_425759


namespace total_cost_correct_l425_425166

def shirt_price : ℕ := 5
def hat_price : ℕ := 4
def jeans_price : ℕ := 10
def jacket_price : ℕ := 20
def shoes_price : ℕ := 15

def num_shirts : ℕ := 4
def num_jeans : ℕ := 3
def num_hats : ℕ := 4
def num_jackets : ℕ := 3
def num_shoes : ℕ := 2

def third_jacket_discount : ℕ := jacket_price / 2
def discount_per_two_shirts : ℕ := 2
def free_hat : ℕ := if num_jeans ≥ 3 then 1 else 0
def shoes_discount : ℕ := (num_shirts / 2) * discount_per_two_shirts

def total_cost : ℕ :=
  (num_shirts * shirt_price) +
  (num_jeans * jeans_price) +
  ((num_hats - free_hat) * hat_price) +
  ((num_jackets - 1) * jacket_price + third_jacket_discount) +
  (num_shoes * shoes_price - shoes_discount)

theorem total_cost_correct : total_cost = 138 := by
  sorry

end total_cost_correct_l425_425166


namespace gcd_factorials_l425_425037

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorials (n : ℕ) (h1 : n = 8) (h2 : n + 2 = 10) (h3 : n + 3 = 11) :
  nat.gcd (factorial n) (nat.gcd (factorial (n + 2)) (factorial (n + 3))) = factorial n :=
by
  simp [factorial, h1, h2, h3, nat.gcd]
  sorry

end gcd_factorials_l425_425037


namespace negation_of_universal_cosine_l425_425479

theorem negation_of_universal_cosine :
  (¬(∀ x : ℝ, cos x ≤ 1)) ↔ (∃ x : ℝ, cos x > 1) :=
sorry

end negation_of_universal_cosine_l425_425479


namespace rolling_a_6_on_10th_is_random_event_l425_425724

-- Definition of what it means for an event to be "random"
def is_random_event (event : ℕ → Prop) : Prop := 
  ∃ n : ℕ, event n

-- Condition: A die roll outcome for getting a 6
def die_roll_getting_6 (roll : ℕ) : Prop := 
  roll = 6

-- The main theorem to state the problem and the conclusion
theorem rolling_a_6_on_10th_is_random_event (event : ℕ → Prop) 
  (h : ∀ n, event n = die_roll_getting_6 n) : 
  is_random_event (event) := 
  sorry

end rolling_a_6_on_10th_is_random_event_l425_425724


namespace oliver_more_money_l425_425955

noncomputable def totalOliver : ℕ := 10 * 20 + 3 * 5
noncomputable def totalWilliam : ℕ := 15 * 10 + 4 * 5

theorem oliver_more_money : totalOliver - totalWilliam = 45 := by
  sorry

end oliver_more_money_l425_425955


namespace squirrel_walnut_count_l425_425698

-- Lean 4 statement
theorem squirrel_walnut_count :
  let initial_boy_walnuts := 12
  let gathered_walnuts := 6
  let dropped_walnuts := 1
  let initial_girl_walnuts := 0
  let brought_walnuts := 5
  let eaten_walnuts := 2
  (initial_boy_walnuts + gathered_walnuts - dropped_walnuts + initial_girl_walnuts + brought_walnuts - eaten_walnuts) = 20 :=
by
  -- Proof goes here
  sorry

end squirrel_walnut_count_l425_425698


namespace kelsey_video_count_l425_425304

variable (E U K : ℕ)

noncomputable def total_videos : ℕ := 411
noncomputable def ekon_videos : ℕ := E
noncomputable def uma_videos : ℕ := E + 17
noncomputable def kelsey_videos : ℕ := E + 43

theorem kelsey_video_count (E U K : ℕ) 
  (h1 : total_videos = ekon_videos + uma_videos + kelsey_videos)
  (h2 : uma_videos = ekon_videos + 17)
  (h3 : kelsey_videos = ekon_videos + 43)
  : kelsey_videos = 160 := 
sorry

end kelsey_video_count_l425_425304


namespace circle_tangent_exists_l425_425561

theorem circle_tangent_exists 
  {A B C D E F G H I : Type} 
  (Γ : Type) 
  [circumcircle : is_circumcircle Γ A B C] 
  (l : line) 
  [tangent_line : is_tangent l Γ A] 
  (D_on_AB : is_point_on D (segment A B)) 
  (E_on_AC : is_point_on E (segment A C)) 
  (ratio_condition : (dist B D) / (dist D A) = (dist A E) / (dist E C)) 
  (F_on_Gamma : is_point_on F Γ) 
  (G_on_Gamma : is_point_on G Γ) 
  (H : point) 
  (H_parallel_AC : parallel (line_through D H) (segment A C)) 
  (I : point) 
  (I_parallel_AB : parallel (line_through E I) (segment A B)) 
  (F_G_intersect : ∃P, is_point_on P (line_through D E) ∧ is_point_on P Γ)
  (F_G_points : F_G_intersect → (is_point_on F (line_through D E) ∧ is_point_on G (line_through D E))) :
  ∃ (γ : Type), is_circle γ ∧ passes_through γ F H G I ∧ is_tangent_to_line γ (segment B C) :=
begin
  sorry
end

end circle_tangent_exists_l425_425561


namespace cans_ounces_per_day_l425_425423

-- Definitions of the conditions
def daily_soda_cans : ℕ := 5
def daily_water_ounces : ℕ := 64
def weekly_fluid_ounces : ℕ := 868

-- Theorem statement proving the number of ounces per can of soda
theorem cans_ounces_per_day (h_soda_daily : daily_soda_cans * 7 = 35)
    (h_weekly_soda : weekly_fluid_ounces - daily_water_ounces * 7 = 420) 
    (h_total_weekly : 35 = ((daily_soda_cans * 7))):
  420 / 35 = 12 := by
  sorry

end cans_ounces_per_day_l425_425423


namespace discount_is_28_l425_425374

-- Definitions
def price_notebook : ℕ := 15
def price_planner : ℕ := 10
def num_notebooks : ℕ := 4
def num_planners : ℕ := 8
def total_cost_with_discount : ℕ := 112

-- The original cost without discount
def original_cost : ℕ := num_notebooks * price_notebook + num_planners * price_planner

-- The discount amount
def discount_amount : ℕ := original_cost - total_cost_with_discount

-- Proof statement
theorem discount_is_28 : discount_amount = 28 := by
  sorry

end discount_is_28_l425_425374


namespace cauchy_bunyakovsky_matrix_inequality_l425_425576

open Matrix

variables {α : Type*} [Field α]
variables {n m : Type*} [Fintype n] [Fintype m]
variables (X Y : Matrix n m α) (E : Matrix n n α → Matrix n n α) (E_inv : Matrix n n α → Matrix n n α)

-- Assume E[Y Yᵀ] is invertible
hypothesis (h_inv : invertible (E (Y ⬝ Yᵀ)))

noncomputable def cb_inequality : Matrix n n α :=
  E (X ⬝ Yᵀ) ⬝ (E (Y ⬝ Yᵀ))⁺ ⬝ E (Y ⬝ Xᵀ)

theorem cauchy_bunyakovsky_matrix_inequality (h_inv : invertible (E (Y ⬝ Yᵀ)))
  : cb_inequality X Y E h_inv ≤ E (X ⬝ Xᵀ) :=
sorry

end cauchy_bunyakovsky_matrix_inequality_l425_425576


namespace part_3_l425_425072

noncomputable def f (x t : ℝ) : ℝ := (2 * x - t) / (x^2 + 1)

def g (t : ℝ) : ℝ := 
  let α := (-t/4 - 1/4).sqrt
  let β := (t/4 + 1/4).sqrt
  (α + β) * (α - β) / ((α^2 + 1) * (β^2 + 1))

theorem part_3 (n : ℕ) : 1 / g 1 + 1 / g 2 + 1 / g 3 + ... + 1 / g n < 2 * ((n^2 + 1).sqrt - 1) :=
by
  sorry

end part_3_l425_425072


namespace cubes_with_two_or_three_blue_faces_l425_425358

theorem cubes_with_two_or_three_blue_faces 
  (four_inch_cube : ℝ)
  (painted_blue_faces : ℝ)
  (one_inch_cubes : ℝ) :
  (four_inch_cube = 4) →
  (painted_blue_faces = 6) →
  (one_inch_cubes = 64) →
  (num_cubes_with_two_or_three_blue_faces = 32) :=
sorry

end cubes_with_two_or_three_blue_faces_l425_425358


namespace value_of_c_l425_425514

theorem value_of_c :
  let c := 1996 * 19971997 - 1995 * 19961996
  in c = 3995992 :=
by
  sorry

end value_of_c_l425_425514


namespace boris_initial_candy_count_l425_425390

theorem boris_initial_candy_count 
  (daughter_ate : ℕ)
  (bowl_count : ℕ)
  (taken_per_bowl : ℕ)
  (candy_in_one_bowl_after_taken : ℕ) :
  daughter_ate = 8 →
  bowl_count = 4 →
  taken_per_bowl = 3 →
  candy_in_one_bowl_after_taken = 20 →
  let total_initial_candy := (candy_in_one_bowl_after_taken + taken_per_bowl) * bowl_count + daughter_ate in
  total_initial_candy = 100 :=
by
  intros h1 h2 h3 h4
  let total_initial_candy := (candy_in_one_bowl_after_taken + taken_per_bowl) * bowl_count + daughter_ate
  sorry

end boris_initial_candy_count_l425_425390


namespace perfect_square_factors_of_360_l425_425866

theorem perfect_square_factors_of_360:
  let factorization := [(2, 3), (3, 2), (5, 1)]
  let perfect_square_exponents exps := exps.all (λ e, e % 2 = 0)
  let num_perfect_square_factors := ((if 0 <= 3 then 1 else 0) + (if 2 <= 3 then 1 else 0)) *
    ((if 0 <= 2 then 1 else 0) + (if 2 <= 2 then 1 else 0)) *
    ((if 0 <= 1 then 1 else 0))
  in num_perfect_square_factors = 4
:= sorry

end perfect_square_factors_of_360_l425_425866


namespace rationalize_denominator_rationalization_example_l425_425969

theorem rationalize_denominator (a b : ℝ) (hb : b ≠ 0) (h : b ^ 2 = a) : (a / b) = ((sqrt a) / (sqrt a / sqrt b)) / (sqrt b) := 
by
  have : a = (sqrt a) ^ 2 := by rw sq_sqrt
  rw this
  rw ← mul_assoc
  rw sqrt_mul_self hb
  rw div_eq_mul_inv
  rw mul_div_assoc
  rw div_self hb
  rw mul_one
  rw sqrt_mul_self
  sorry

-- Specific example of the problem:
theorem rationalization_example : (7 : ℝ) / (sqrt 175) = (sqrt 7) / 5 := 
by
  have h1 : (175 : ℝ) = (25 * 7) := by norm_num
  rw h1
  rw sqrt_mul
  rw sqrt_mul
  rw sqrt_25
  ring
  sorry

end rationalize_denominator_rationalization_example_l425_425969


namespace calc_compound_interest_l425_425385

variable (P : ℝ) (r : ℝ) (n : ℕ)

theorem calc_compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : 
  P = 15000 →
  r = 0.045 →
  n = 7 →
  (P * (1 + r)^n).round = 20144 := by
  intros hP hr hn
  sorry

end calc_compound_interest_l425_425385


namespace proof_of_relation_l425_425058

noncomputable def k : ℝ := -1 -- Assuming some negative constant for k

variables x1 x2 x3 y1 y2 y3 : ℝ
variables (hx : x1 < x2 ∧ x2 < 0 ∧ 0 < x3)
variables (hAx : y1 = k / x1) (hBx : y2 = k / x2) (hCx : y3 = k / x3)
def inverse_proportional_relation : Prop := y2 > y1 ∧ y1 > y3

theorem proof_of_relation :
  inverse_proportional_relation x1 x2 x3 y1 y2 y3 :=
by
  sorry

end proof_of_relation_l425_425058


namespace cost_of_green_shirts_l425_425630

noncomputable def total_cost_kindergarten : ℝ := 101 * 5.8
noncomputable def total_cost_first_grade : ℝ := 113 * 5
noncomputable def total_cost_second_grade : ℝ := 107 * 5.6
noncomputable def total_cost_all_but_third : ℝ := total_cost_kindergarten + total_cost_first_grade + total_cost_second_grade
noncomputable def total_third_grade : ℝ := 2317 - total_cost_all_but_third
noncomputable def cost_per_third_grade_shirt : ℝ := total_third_grade / 108

theorem cost_of_green_shirts : cost_per_third_grade_shirt = 5.25 := sorry

end cost_of_green_shirts_l425_425630


namespace tangent_line_at_point_l425_425789

/-- 
The curve is defined by y = x^3 - x + 3
The point of tangency is (1, 3)
Prove the equation of the tangent line to the curve at the point (1, 3) is 2x - y + 1 = 0
-/
theorem tangent_line_at_point : 
  let y (x : ℝ) := x^3 - x + 3 in
  let x_tangent := 1 in let y_tangent := 3 in
  let slope := (deriv y) x_tangent in
  slope = 2 ∧ y_tangent = y x_tangent →
  ∃ m b, (m = slope ∧ b = y_tangent - m * x_tangent ∧ (∀ x y, y = m * x + b ↔ 2 * x - y + 1 = 0)) :=
begin
  sorry
end

end tangent_line_at_point_l425_425789


namespace exist_boy_danced_no_more_than_half_l425_425341

theorem exist_boy_danced_no_more_than_half :
  ∀ (boys : Fin 100) (girls : Fin 100) 
  (pair_danced : Fin 100 → Fin 100 → Bool)
  (H : ∀ g : Fin 100, ∃ s₁ s₂ : Finset (Fin 100), (s₁.disjoint s₂) ∧ (s₁ ∪ s₂ = Finset.univ) ∧ 
  (Finset.sum s₁ (λ b, if pair_danced b g then 1 else 0) = Finset.sum s₂ (λ b, if pair_danced b g then 1 else 0))), 
  ∃ k : Fin 100, (Finset.sum (Finset.univ.filter (λ g, pair_danced k g)) (λ g, 1)) / 2 ≥ Finset.sum (Finset.univ.filter (λ g, pair_danced k g)) (λ g, if g = k then 1 else 0) :=
sorry

end exist_boy_danced_no_more_than_half_l425_425341


namespace D_double_prime_coordinates_l425_425595

-- The coordinates of points A, B, C, D as given in the problem
def A : (ℝ × ℝ) := (3, 6)
def B : (ℝ × ℝ) := (5, 10)
def C : (ℝ × ℝ) := (7, 6)
def D : (ℝ × ℝ) := (5, 2)

-- Reflection across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def D' : ℝ × ℝ := reflect_x D

-- Translate the point (x, y) by (dx, dy)
def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ := (p.1 + dx, p.2 + dy)

-- Reflect across the line y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Combined translation and reflection across y = x + 2
def reflect_y_eq_x_plus_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  let p_translated := translate p 0 (-2)
  let p_reflected := reflect_y_eq_x p_translated
  translate p_reflected 0 2

def D'' : ℝ × ℝ := reflect_y_eq_x_plus_2 D'

theorem D_double_prime_coordinates : D'' = (-4, 7) := by
  sorry

end D_double_prime_coordinates_l425_425595


namespace length_more_than_breadth_l425_425268

theorem length_more_than_breadth (b : ℝ) (x : ℝ) 
  (h1 : b + x = 55) 
  (h2 : 4 * b + 2 * x = 200) 
  (h3 : (5300 : ℝ) / 26.5 = 200)
  : x = 10 := 
by
  sorry

end length_more_than_breadth_l425_425268


namespace part_I_part_II_part_III_l425_425484

def a_seq : ℕ → ℚ
| 1       := 1
| (n + 1) := if n % 2 = 1 then (1 / 2) * (a_seq n) + n else (a_seq n) - 2 * n

def b_seq (n : ℕ) : ℚ := a_seq (2 * n) - 2

def C (n : ℕ) : ℚ := n * (2 / 3) ^ n

def S : ℕ → ℚ
| 1       := C 1
| (n + 1) := S n + C (n + 1)

theorem part_I : 
  a_seq 2 = 3 / 2 ∧ a_seq 3 = -5 / 2 ∧ a_seq 4 = 7 / 4 :=
by sorry

theorem part_II :
  ∃ r : ℚ, 
  (∀ n : ℕ, b_seq (n + 1) = r * b_seq n) ∧ 
  (b_seq 1 = -1 / 2 ∧ ∀ n : ℕ, b_seq n = -((1 / 2) ^ n)) :=
by sorry

theorem part_III (n : ℕ) (h : (3 / 4) ^ n * C n = -n * b_seq n) :
  S n < 6 :=
by sorry

end part_I_part_II_part_III_l425_425484


namespace ratio_avg_speed_round_trip_l425_425695

def speed_boat := 20
def speed_current := 4
def distance := 2

theorem ratio_avg_speed_round_trip :
  let downstream_speed := speed_boat + speed_current
  let upstream_speed := speed_boat - speed_current
  let time_down := distance / downstream_speed
  let time_up := distance / upstream_speed
  let total_time := time_down + time_up
  let total_distance := distance + distance
  let avg_speed := total_distance / total_time
  avg_speed / speed_boat = 24 / 25 :=
by sorry

end ratio_avg_speed_round_trip_l425_425695


namespace prime_divisors_distinct_l425_425606

theorem prime_divisors_distinct (n : ℕ) (h : n > 0) :
  ∀ k ∈ (finset.range n).map (λ i, i + 1), 
  ∃ p_k : ℕ, prime p_k ∧ p_k ∣ (factorial n + k) ∧ 
    (∀ j ∈ (finset.range n).map (λ i, i + 1), j ≠ k → ¬ (p_k ∣ (factorial n + j))) :=
by
  sorry

end prime_divisors_distinct_l425_425606


namespace evaluate_expression_l425_425410

theorem evaluate_expression : 2 + 5 * 3^2 - 4 * 2 + 7 * 3 / 3 = 46 := by
  sorry

end evaluate_expression_l425_425410


namespace total_distance_travelled_l425_425496

-- Definition of the given conditions
def distance_to_market := 30  -- miles
def time_to_home := 0.5  -- hours
def speed_to_home := 20  -- miles per hour

-- The statement we want to prove: Total distance traveled is 40 miles.
theorem total_distance_travelled : 
  distance_to_market + speed_to_home * time_to_home = 40 := 
by 
  sorry

end total_distance_travelled_l425_425496


namespace cos_identity_proof_l425_425073

noncomputable def cos_value (θ : ℝ) (h : θ ∈ Ioo 0 (π / 2)) 
  (h₁ : cos (θ + π / 6) = 5 / 13) : ℝ := 
  cos θ
  
theorem cos_identity_proof (θ : ℝ) (h : θ ∈ Ioo 0 (π / 2)) 
  (h₁ : cos (θ + π / 6) = 5 / 13) : 
  cos_value θ h h₁ = (12 + 5 * real.sqrt 3) / 26 :=
by sorry

end cos_identity_proof_l425_425073


namespace sum_of_arithmetic_sequence_l425_425206

def f (x : ℝ) : ℝ := (x - 3)^3 + x - 1

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ)
  (h_arith : is_arithmetic_sequence a d)
  (h_nonzero : d ≠ 0)
  (h_sum_f : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) = 14) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
by 
  sorry

end sum_of_arithmetic_sequence_l425_425206


namespace Jane_Hector_meet_at_D_l425_425178

noncomputable def Jane_Hector_meet_point (s : ℝ) (t : ℝ) : Prop :=
  let jane_speed := 2 * s
  let hector_dist := s * t
  let jane_dist := jane_speed * t
  (hector_dist + jane_dist = 18) ∧ 
  (hector_dist = 6) ∧
  (jane_dist = 12)

theorem Jane_Hector_meet_at_D : ∀ (s : ℝ), ∃ (t : ℝ), Jane_Hector_meet_point s t → 
  (∃ (point : String), point = "D") :=
by 
  intros 
  use (6 / s) 
  sorry

end Jane_Hector_meet_at_D_l425_425178


namespace counties_rain_tuesday_l425_425160

-- Probability definitions for events A and B
def A : Type := { c : Type // ∃ m : Bool, m = true }  -- Counties with rain on Monday
def B : Type := { c : Type // ∃ t : Bool, t = true }  -- Counties with rain on Tuesday

-- Given conditions
def P_A : ℝ := 0.7
def P_A_inter_B : ℝ := 0.6
def P_neither : ℝ := 0.35

-- Goal: prove the percentage of counties that received some rain on Tuesday
theorem counties_rain_tuesday (P_A : ℝ) (P_A_inter_B : ℝ) (P_neither : ℝ) : 
    P_A = 0.7 ∧ P_A_inter_B = 0.6 ∧ P_neither = 0.35 → 
    ∃ P_B : ℝ, P_B = 0.55 :=
by
  sorry

end counties_rain_tuesday_l425_425160


namespace sum_of_first_five_terms_geometric_sequence_l425_425465

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * r

theorem sum_of_first_five_terms_geometric_sequence :
  (∃ a : ℕ → ℝ, ∃ q : ℝ,
    geometric_sequence a q ∧
    a 2 = 2 ∧ 
    a 3 = 4 ∧ 
    (∑ i in finset.range 5, a i) = 31) :=
by
  sorry

end sum_of_first_five_terms_geometric_sequence_l425_425465


namespace problem_statement_l425_425820

-- Define the function f
def f (x : ℝ) : ℝ :=
if x > 0 then Real.log x else 2^(-x)

-- Define the integral a
noncomputable def a : ℝ :=
∫ x in 0..1, Real.exp x + 2 * x

-- Define the logarithm base 2
noncomputable def log_base2 (x : ℝ) : ℝ :=
Real.log x / Real.log 2

-- The statement we want to prove
theorem problem_statement : f a + f (log_base2 (1 / 6)) = 7 := sorry

end problem_statement_l425_425820


namespace positive_X_solution_l425_425416

def boxtimes (X Y : ℤ) : ℤ := X^2 - 2 * X + Y^2

theorem positive_X_solution (X : ℤ) (h : boxtimes X 7 = 164) : X = 13 :=
by
  sorry

end positive_X_solution_l425_425416


namespace sin_A_plus_B_eq_max_area_eq_l425_425155

-- Conditions for problem 1 and 2
variables (A B C a b c : ℝ)
variable (h_A_B_C : A + B + C = Real.pi)
variable (h_sin_C_div_2 : Real.sin (C / 2) = 2 * Real.sqrt 2 / 3)

noncomputable def sin_A_plus_B := Real.sin (A + B)

-- Problem 1: Prove that sin(A + B) = 4 * sqrt 2 / 9
theorem sin_A_plus_B_eq : sin_A_plus_B A B = 4 * Real.sqrt 2 / 9 :=
by sorry

-- Adding additional conditions for problem 2
variable (h_a_b_sum : a + b = 2 * Real.sqrt 2)

noncomputable def area (a b C : ℝ) := (1 / 2) * a * b * (2 * Real.sin (C / 2) * (Real.cos (C / 2)))

-- Problem 2: Prove that the maximum value of the area S of triangle ABC is 4 * sqrt 2 / 9
theorem max_area_eq : ∃ S, S = area a b C ∧ S ≤ 4 * Real.sqrt 2 / 9 :=
by sorry

end sin_A_plus_B_eq_max_area_eq_l425_425155


namespace find_ages_l425_425729

def current_ages : Prop :=
  ∃ (A F : ℕ), 
    (F = 2 * A + 5) ∧ 
    (A - 6 = 1 / 3 * A) ∧ 
    (A = 9) ∧ 
    (F = 23)

theorem find_ages : current_ages :=
  by
    -- Definitions of A and F
    let A := 9
    let F := 23
    use [A, F]
    -- Prove the conditions
    split
    { -- F = 2A + 5
      calc F = 23 : rfl
         ... = 2 * 9 + 5 : rfl
    }
    split
    { -- A - 6 = (1 / 3) A
      calc A - 6 = 9 - 6 : rfl
         ... = (1 / 3) * 9 : by norm_num
    }
    split
    { -- A = 9
      exact rfl
    }
    { -- F = 23
      exact rfl
    }

end find_ages_l425_425729


namespace sum_of_repeating_decimals_l425_425015

/-- Definitions of the repeating decimals as real numbers. --/
def x : ℝ := 0.3 -- This actually represents 0.\overline{3} in Lean
def y : ℝ := 0.04 -- This actually represents 0.\overline{04} in Lean
def z : ℝ := 0.005 -- This actually represents 0.\overline{005} in Lean

/-- The theorem stating that the sum of these repeating decimals is a specific fraction. --/
theorem sum_of_repeating_decimals : x + y + z = (14 : ℝ) / 37 := 
by 
  sorry -- Placeholder for the proof

end sum_of_repeating_decimals_l425_425015


namespace rectangle_diagonal_length_l425_425839

theorem rectangle_diagonal_length :
  ∀ (length width diagonal : ℝ), length = 6 ∧ length * width = 48 ∧ diagonal = Real.sqrt (length^2 + width^2) → diagonal = 10 :=
by
  intro length width diagonal
  rintro ⟨hl, area_eq, diagonal_eq⟩
  sorry

end rectangle_diagonal_length_l425_425839


namespace probability_at_least_one_vowel_l425_425300

open Finset

def set1 : Finset Char := {'a', 'b', 'c', 'd', 'e', 'f'}
def set2 : Finset Char := {'k', 'l', 'm', 'n', 'o', 'p'}
def vowels1 : Finset Char := {'a', 'e'}
def vowels2 : Finset Char := {'o'}

theorem probability_at_least_one_vowel :
  let total_outcomes := (set1.card * set2.card : ℕ)
  let favorable_outcomes := 
        ((vowels1.card * set2.card) + ((set1.card - vowels1.card) * vowels2.card) + (vowels1.card * vowels2.card)) in
  (favorable_outcomes / total_outcomes) = (1 / 2 : ℝ) := by
  sorry

end probability_at_least_one_vowel_l425_425300


namespace sum_of_areas_alternating_colors_eq_l425_425368

noncomputable def regular_polygon (n : ℕ) : Type := { p : ℝ × ℝ // p ∈ polygon_vertices n }

-- Definitions related to the conditions
def point_inside_polygon {n : ℕ} (p : ℝ × ℝ) : Prop :=
 ∀ (v : ℝ × ℝ), v ∈ (polygon_vertices (2 * n)) → inside_polygon p v

def connected_triangles {n : ℕ} (p : ℝ × ℝ) : list (triangle ℝ) :=
 connect_point_to_vertices p (polygon_vertices (2 * n))

def alternately_colored_triangles {n : ℕ} (tris : list (triangle ℝ)) : Prop :=
 alternately_colored tris

-- The proof statement
theorem sum_of_areas_alternating_colors_eq {n : ℕ} (p : ℝ × ℝ)
  (h1 : regular_polygon (2 * n))
  (h2 : point_inside_polygon p) 
  (h3 : connected_triangles p = tris)
  (h4 : alternately_colored_triangles tris) :
  (sum_of_areas (filter_color tris blue)) = (sum_of_areas (filter_color tris red)) :=
begin
  sorry
end

end sum_of_areas_alternating_colors_eq_l425_425368


namespace sum_of_repeating_decimals_l425_425017

/-- Definitions of the repeating decimals as real numbers. --/
def x : ℝ := 0.3 -- This actually represents 0.\overline{3} in Lean
def y : ℝ := 0.04 -- This actually represents 0.\overline{04} in Lean
def z : ℝ := 0.005 -- This actually represents 0.\overline{005} in Lean

/-- The theorem stating that the sum of these repeating decimals is a specific fraction. --/
theorem sum_of_repeating_decimals : x + y + z = (14 : ℝ) / 37 := 
by 
  sorry -- Placeholder for the proof

end sum_of_repeating_decimals_l425_425017


namespace value_of_a_minus_c_l425_425518

theorem value_of_a_minus_c
  (a b c d : ℝ) 
  (h1 : (a + d + b + d) / 2 = 80)
  (h2 : (b + d + c + d) / 2 = 180)
  (h3 : d = 2 * (a - b)) :
  a - c = -200 := sorry

end value_of_a_minus_c_l425_425518


namespace original_people_l425_425958

-- Declare the original number of people in the room
variable (x : ℕ)

-- Conditions
-- One third of the people in the room left
def remaining_after_one_third_left (x : ℕ) : ℕ := (2 * x) / 3

-- One quarter of the remaining people started to dance
def dancers (remaining : ℕ) : ℕ := remaining / 4

-- Number of people not dancing
def non_dancers (remaining : ℕ) (dancers : ℕ) : ℕ := remaining - dancers

-- Given that there are 18 people not dancing
variable (remaining : ℕ) (dancers : ℕ)
axiom non_dancers_number : non_dancers remaining dancers = 18

-- Theorem to prove
theorem original_people (h_rem: remaining = remaining_after_one_third_left x) 
(h_dancers: dancers = remaining / 4) : x = 36 := by
  sorry

end original_people_l425_425958


namespace find_radius_l425_425658

noncomputable def radius_of_tangent_circles : ℝ := sqrt(5) / 3

theorem find_radius 
  (r : ℝ) 
  (h1 : (∀ (x y : ℝ), (x - r)^2 + y^2 = r^2 → 4*x^2 + 9*y^2 = 9 → (x = r ∨ x = -r)) ∧ (r > 0)) : 
  r = radius_of_tangent_circles := 
sorry

end find_radius_l425_425658


namespace circle_diameter_eq_l425_425997

noncomputable def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  ((x1 + x2) / 2, (y1 + y2) / 2)

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

noncomputable def radius (x1 y1 x2 y2 : ℝ) : ℝ :=
  distance x1 y1 x2 y2 / 2

noncomputable def circle_equation (h k r : ℝ) : (ℝ × ℝ) -> ℝ :=
  λ (p : ℝ × ℝ), (p.1 - h) ^ 2 + (p.2 - k) ^ 2 - r ^ 2

theorem circle_diameter_eq :
  circle_equation
    (midpoint 0 0 6 8).1
    (midpoint 0 0 6 8).2
    (radius 0 0 6 8)
    = λ (p : ℝ × ℝ), (p.1 - 3) ^ 2 + (p.2 - 4) ^ 2 - 25 :=
by
  sorry

end circle_diameter_eq_l425_425997


namespace inequality1_inequality2_l425_425610

variable (a b c d : ℝ)

theorem inequality1 : 
  (a + c)^2 * (b + d)^2 ≥ 2 * (a * b^2 * c + b * c^2 * d + c * d^2 * a + d * a^2 * b + 4 * a * b * c * d) :=
  sorry

theorem inequality2 : 
  (a + c)^2 * (b + d)^2 ≥ 4 * b * c * (c * d + d * a + a * b) :=
  sorry

end inequality1_inequality2_l425_425610


namespace simplify_expression_l425_425617

theorem simplify_expression (x : ℝ) : (2 * x)^5 + (3 * x) * x^4 + 2 * x^3 = 35 * x^5 + 2 * x^3 :=
by
  sorry

end simplify_expression_l425_425617


namespace andrei_stamps_l425_425600

theorem andrei_stamps (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 5 = 3) ∧ (x % 7 = 5) ∧ (150 < x) ∧ (x ≤ 300) → 
  x = 208 :=
sorry

end andrei_stamps_l425_425600


namespace find_distance_from_M_to_y_axis_l425_425260

noncomputable def distance_from_point_to_y_axis (x : ℝ) : ℝ := abs x

variables (M : ℝ × ℝ)
variables (a b c : ℝ)
variables (F1 F2 : ℝ × ℝ)

-- Define the problem conditions
def ellipse (x y : ℝ) := (x^2 / 4) + y^2 = 1
def is_focus (F : ℝ × ℝ) := F = (c, 0) ∨ F = (-c, 0)
def on_ellipse (M : ℝ × ℝ) := ellipse M.1 M.2
def orthogonal_vectors (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) :=
  (M.1 - F1.1) * (M.1 - F2.1) + (M.2 - F1.2) * (M.2 - F2.2) = 0

-- Define the problem statement
theorem find_distance_from_M_to_y_axis :
  a = 2 → b = 1 → c = real.sqrt 3 → is_focus F1 → is_focus F2 → 
  on_ellipse M → orthogonal_vectors M F1 F2 →
  distance_from_point_to_y_axis M.1 = 2 * real.sqrt 6 / 3 :=
sorry

end find_distance_from_M_to_y_axis_l425_425260


namespace count_three_digit_squares_divisible_by_4_l425_425114

def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

theorem count_three_digit_squares_divisible_by_4 : 
  let squares := {n : ℕ | is_three_digit n ∧ is_perfect_square n ∧ is_divisible_by_4 n} in
  Fintype.card squares = 10 :=
sorry

end count_three_digit_squares_divisible_by_4_l425_425114


namespace range_of_f_cos_diff_A_B_l425_425090

-- Define the function f
def f (x : Real) : Real := 2 * sin (x + π / 3) * cos x

-- Given conditions
def A : Real := π / 3
def b : Real := 2
def c : Real := 3
def a : Real := sqrt 7

-- Proving the range of f
theorem range_of_f :
  ∀ y, ∃ x, y = f (x) ↔ (y ∈ Icc (sqrt 3 - 2) / 2 ((sqrt 3 + 2) / 2)) :=
sorry

-- Proving the value of cos (A - B)
theorem cos_diff_A_B (A_is_acute : A < π / 2)
  (h : f A = sqrt 3 / 2) :
  ∀ A B, b * sin B = a * sin A → cos (A - B) = 5 * sqrt 7 / 14 :=
sorry

end range_of_f_cos_diff_A_B_l425_425090


namespace derivative_f_l425_425638

variable (x : ℝ)

def f (x : ℝ) : ℝ := exp x * cos x

theorem derivative_f : (deriv (f x) x) = exp x * (cos x - sin x) := by
  sorry

end derivative_f_l425_425638


namespace number_of_intersections_l425_425772

-- Conditions for the problem
def Line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 2
def Line2 (x y : ℝ) : Prop := 5 * x + 3 * y = 6
def Line3 (x y : ℝ) : Prop := x - 4 * y = 8

-- Statement to prove
theorem number_of_intersections : ∃ (p1 p2 p3 : ℝ × ℝ), 
  (Line1 p1.1 p1.2 ∧ Line2 p1.1 p1.2) ∧ 
  (Line1 p2.1 p2.2 ∧ Line3 p2.1 p2.2) ∧ 
  (Line2 p3.1 p3.2 ∧ Line3 p3.1 p3.2) ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 :=
sorry

end number_of_intersections_l425_425772


namespace cone_lateral_surface_area_l425_425466

theorem cone_lateral_surface_area (s : ℝ) (h_s : s = 10) 
  (h_unfold : true) : 
  ∃ A : ℝ, A = 50 * Real.pi :=
by
  use 50 * Real.pi
  -- Proof is omitted
  sorry

end cone_lateral_surface_area_l425_425466


namespace solution_set_of_inequality_l425_425032

theorem solution_set_of_inequality :
  {x : ℝ | -2 * x^2 + x + 1 > 0} = set.Ioo (-1 / 2) 1 :=
by sorry

end solution_set_of_inequality_l425_425032


namespace problem_l425_425230

theorem problem (a : ℤ) (n : ℕ) : (a + 1) ^ (2 * n + 1) + a ^ (n + 2) ∣ a ^ 2 + a + 1 :=
sorry

end problem_l425_425230


namespace people_lost_l425_425745

-- Define the given conditions
def ratio_won_to_lost : ℕ × ℕ := (4, 1)
def people_won : ℕ := 28

-- Define the proof problem
theorem people_lost (L : ℕ) (h_ratio : ratio_won_to_lost = (4, 1)) (h_won : people_won = 28) : L = 7 :=
by
  -- Skip the proof
  sorry

end people_lost_l425_425745


namespace f_definition_neg_l425_425846

variable {f : ℝ → ℝ}

-- Defining odd function property for f
def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f(-x) = -f(x)

-- Defining given condition for non-negative x
def f_definition_nonneg (f : ℝ → ℝ) := ∀ x : ℝ, x ≥ 0 → f(x) = x * (1 + x)

-- The theorem we want to prove
theorem f_definition_neg (hf_odd : odd_function f) (hf_nonneg : f_definition_nonneg f) :
  ∀ x : ℝ, x < 0 → f(x) = x * (1 - x) :=
sorry

end f_definition_neg_l425_425846


namespace sum_of_D_coordinates_l425_425227

-- Define the given points
structure Point where
  x : ℤ
  y : ℤ

-- Define the points P and C
def P : Point := ⟨1, -3⟩
def C : Point := ⟨7, 5⟩

-- Define that D is the point we are looking for
variable D : Point

-- Define the midpoint formula as a condition
def isMidpoint (M C D: Point) : Prop :=
  (M.x = (C.x + D.x) / 2) ∧ (M.y = (C.y + D.y) / 2)

-- State the theorem
theorem sum_of_D_coordinates :
  isMidpoint P C D →
  D.x + D.y = -16 :=
by
  intro midpoint_condition
  sorry

end sum_of_D_coordinates_l425_425227


namespace cubic_eq_factorization_l425_425794

theorem cubic_eq_factorization (a b c : ℝ) :
  (∃ m n : ℝ, (x^3 + a * x^2 + b * x + c = (x^2 + m) * (x + n))) ↔ (c = a * b) :=
sorry

end cubic_eq_factorization_l425_425794


namespace geometric_arithmetic_sequence_sum_l425_425847

theorem geometric_arithmetic_sequence_sum (a b : ℕ → ℕ) (q : ℕ) (h1 : a 2 = 4) (h2 : a 3 + a 4 = 24)
(h3 : b 1 = 3) (h4 : b 2 = 6)
(h5 : ∀ n : ℕ, b n - a n = (n - 1) + 1):
  (∀ n : ℕ, a n = 2 ^ n) ∧ (∀ n : ℕ, ∑ k in finset.range n.succ, b k = (n * (n + 1)) / 2 + 2 ^ (n + 1) - 2) :=
by 
  sorry

end geometric_arithmetic_sequence_sum_l425_425847


namespace bike_travel_distance_in_yards_l425_425694

variable (b t : ℝ)

theorem bike_travel_distance_in_yards (ht : t > 0) : 
  let rate_feet := b / 4,
      seconds_in_4_minutes := 4 * 60,
      distance_feet := (rate_feet / t) * seconds_in_4_minutes,
      distance_yards := distance_feet / 3 in
    distance_yards = 20 * b / t :=
by
  sorry

end bike_travel_distance_in_yards_l425_425694


namespace minimum_cost_of_pool_construction_l425_425764

-- Define the constants given in the problem
def V : ℝ := 18  -- Volume in cubic meters
def h : ℝ := 2   -- Depth in meters
def cost_bottom_per_m2 : ℝ := 200  -- Cost per square meter for the bottom
def cost_walls_per_m2 : ℝ := 150   -- Cost per square meter for the walls

-- Define the cost function
noncomputable def cost (w : ℝ) : ℝ :=
  let l := V / (w * h)
  in cost_bottom_per_m2 * l * w + cost_walls_per_m2 * (2 * (h * l) + 2 * (h * w))

-- The statement we need to prove: the minimum cost is 5400 yuan
theorem minimum_cost_of_pool_construction : 
  (∃ w : ℝ, 0 < w ∧ (∀ w', 0 < w' → cost w ≤ cost w') ∧ cost w = 5400) :=
sorry

end minimum_cost_of_pool_construction_l425_425764


namespace minimum_value_am_bn_l425_425459

theorem minimum_value_am_bn (a b m n : ℝ) (hp_a : a > 0)
    (hp_b : b > 0) (hp_m : m > 0) (hp_n : n > 0) (ha_b : a + b = 1)
    (hm_n : m * n = 2) :
    (am + bn) * (bm + an) ≥ 3/2 := by
  sorry

end minimum_value_am_bn_l425_425459


namespace lattice_points_count_l425_425708

def is_lattice_point (p : ℤ × ℤ) : Prop :=
  true

def in_region (p : ℤ × ℤ) : Prop :=
  let (x, y) := p in y = abs x ∨ y = -x^2 + 8

theorem lattice_points_count :
  { p : ℤ × ℤ | is_lattice_point p ∧ in_region p }.to_finset.card = 29 :=
by sorry

end lattice_points_count_l425_425708


namespace problem_statement_l425_425320

theorem problem_statement :
  (∀ x : ℝ, |x| < 2 → x < 3) ∧
  (∀ x : ℝ, ¬ (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ∧
  (-1 < m ∧ m < 0 → ∀ a b : ℝ, a ≠ b → (a * b > 0)) :=
by
  sorry

end problem_statement_l425_425320


namespace find_y_z_l425_425673

theorem find_y_z (x y z : ℚ) (h1 : (x + y) / (z - x) = 9 / 2) (h2 : (y + z) / (y - x) = 5) (h3 : x = 43 / 4) :
  y = 12 / 17 + 17 ∧ z = 5 / 68 + 17 := 
by sorry

end find_y_z_l425_425673


namespace common_divisors_product_eq_830390625_l425_425030

-- Open a section to keep the namespace clean
section DivisorsProduct

-- Given divisor definitions
def divisors_180 : List Int := [±1, ±2, ±3, ±4, ±5, ±6, ±9, ±10, ±12, ±15, ±18, ±20, ±30, ±36, ±45, ±60, ±90, ±180]
def divisors_45 : List Int := [±1, ±3, ±5, ±9, ±15, ±45]

-- Define the product of common divisors
def product_of_common_divisors : Int :=
  ((divisors_180 ∩ divisors_45).map (λ x => abs x)).prod

-- The proof statement
theorem common_divisors_product_eq_830390625 : product_of_common_divisors = 830390625 :=
by sorry

end DivisorsProduct

end common_divisors_product_eq_830390625_l425_425030


namespace r_minus_s_eq_two_l425_425196

-- Definitions based on conditions
def is_solution (x : ℝ) : Prop := (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3

noncomputable def r : ℝ := -3  -- x = -3 as derived in the solution
noncomputable def s : ℝ := -5  -- x = -5 as derived in the solution

-- Given that r and s are distinct solutions and r > s
lemma problem_condition : r ≠ s ∧ r > s := by
  have h1 : r ≠ s := by linarith
  have h2 : r > s := by linarith
  exact ⟨h1, h2⟩

-- The main statement to prove
theorem r_minus_s_eq_two (hr : is_solution r) (hs : is_solution s) 
  (h_distinct : r ≠ s) (h_order : r > s) : (r - s) = 2 := by
  sorry

end r_minus_s_eq_two_l425_425196


namespace ball_dance_problem_l425_425339

def exists_boy_girl_half_dances (T D : Fin 100 → ℕ) (danced : Fin 100 → Fin 100 → ℕ) : Prop :=
  ∃ k : Fin 100, danced k k ≤ T k / 2

theorem ball_dance_problem :
  ∀ (danced : Fin 100 → Fin 100 → ℕ)
  (T : Fin 100 → ℕ) (D : Fin 100 → ℕ),
  (∀ (g : Fin 100), ∃ (group1 group2 : Finset (Fin 100)),
    (group1 ∪ group2 = Finset.univ) ∧ 
    (group1 ∩ group2 = ∅) ∧ 
    (∑ x in group1, danced x g = ∑ x in group2, danced x g)) →
  exists_boy_girl_half_dances T D danced :=
by
  sorry

end ball_dance_problem_l425_425339


namespace find_z_l425_425084

variable (z : ℂ)
variable (h : z + 3 * complex.I - 3 = 6 - 3 * complex.I)

theorem find_z : z = 9 - 6 * complex.I := by
  sorry

end find_z_l425_425084


namespace will_has_16_pieces_l425_425687

def will_still_has_candies : Prop :=
∀ (boxes_purchased boxes_given pieces_per_box : ℕ), 
  boxes_purchased = 7 →
  boxes_given = 3 →
  pieces_per_box = 4 →
  (boxes_purchased - boxes_given) * pieces_per_box = 16

theorem will_has_16_pieces :
  will_still_has_candies :=
by
  intros boxes_purchased boxes_given pieces_per_box h₁ h₂ h₃
  have h₄ : (boxes_purchased - boxes_given) = 4, by rw [h₁, h₂]
  have h₅ : 4 * pieces_per_box = 16, by rw [h₃]
  rw [h₄]
  exact h₅

end will_has_16_pieces_l425_425687


namespace power_of_2_in_factorial_8_l425_425887

theorem power_of_2_in_factorial_8 :
  let x := (Nat.factorial 8) in
  ∃ i k m p : ℕ, x = 2^i * 3^k * 5^m * 7^p ∧ i + k + m + p = 11 ∧ i = 7 :=
begin
  sorry
end

end power_of_2_in_factorial_8_l425_425887


namespace coeff_of_x_squared_in_expansion_middle_term_in_expansion_l425_425809

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the conditions and the proof problems

-- Part 1: Coefficient of x^2 in the expansion
theorem coeff_of_x_squared_in_expansion :
  (binom 10 3 = binom 10 7) → 
  (∀ (x : ℝ), coeff_of_term (sqrt x + 1/(2*sqrt(sqrt(x)))) 10 2 = 105/8) :=
by sorry

-- Part 2: Middle term of the expansion
theorem middle_term_in_expansion :
  (binom 8 0 * 1 = 1 ∧
   binom 8 1 * (1/2) = 8/2 ∧
   binom 8 2 * (1/2)^2 = (8^2 - 8)/8) →
  (∀ (x : ℝ), middle_term (sqrt x + 1/(2*sqrt(sqrt(x)))) 8 = 35/8 * x) :=
by sorry

end coeff_of_x_squared_in_expansion_middle_term_in_expansion_l425_425809


namespace greatest_integer_x_of_inequality_l425_425770

theorem greatest_integer_x_of_inequality :
  ∃ x : ℤ, 7 - 5 * x + x^2 > 24 ∧ (∀ y : ℤ, 7 - 5 * y + y^2 > 24 → y ≤ x) ∧ x = 7 :=
begin
  sorry
end

end greatest_integer_x_of_inequality_l425_425770


namespace probability_student_A_advances_way1_increases_success_chance_l425_425249

/-- Student A participates in the individual qualifying round. The probability of A answering 
the first three questions correctly is 1/2 each, and the probability of answering the last 
two questions correctly is 1/3 each. Prove the probability of A advancing to the next round 
is 3/8. -/
theorem probability_student_A_advances :
  let p1 := (1/2) ^ 3,
      p2 := (3 * (1/2) ^ 3 * (1 - (2/3) ^ 2)),
      p3 := (3 * (1/2) ^ 2 * (1/3) ^ 2)
  in p1 + p2 + p3 = 3/8 :=
sorry

/-- In team challenge finals, assuming each student in a class has a constant probability of 
answering the given question correctly, denoted as p (0 < p < 1). For n ≥ 10, prove that 
Way 1 has a higher probability of class success than Way 2. -/
theorem way1_increases_success_chance (p : ℝ) (h1 : 0 < p) (h2 : p < 1) (n : ℕ) (h : n ≥ 10) :
  let P1 := p ^ n * (2 - p) ^ n,
      P2 := 1 - (1 - p ^ n) ^ 2
  in P1 > P2 :=
sorry

end probability_student_A_advances_way1_increases_success_chance_l425_425249


namespace fraction_multiplication_exponent_l425_425750

theorem fraction_multiplication_exponent :
  ( (8 : ℚ) / 9 )^2 * ( (1 : ℚ) / 3 )^2 = (64 / 729 : ℚ) := 
by
  sorry

end fraction_multiplication_exponent_l425_425750


namespace calculate_speed_of_second_fragment_l425_425357

noncomputable def speed_of_second_fragment
  (initial_speed : ℝ)
  (time_after_start : ℝ)
  (horizontal_speed_small_fragment : ℝ)
  (mass_ratio : ℕ)
  (gravity : ℝ)
  : ℝ :=
let vertical_speed_before_explosion := initial_speed - gravity * time_after_start,
    v1x := horizontal_speed_small_fragment,
    v1y := vertical_speed_before_explosion,
    v2x := - (v1x / 2),
    v2y := (3 * v1y / 2),
    magnitude_v2 := real.sqrt (v2x^2 + v2y^2) in
magnitude_v2

theorem calculate_speed_of_second_fragment :
  speed_of_second_fragment 20 1 16 2 10 = 2 * real.sqrt 41 := by
  sorry

end calculate_speed_of_second_fragment_l425_425357


namespace area_trapezoid_EFBA_l425_425688

noncomputable def rectangle_ABCD : Type :=
  {BC : ℝ × AB : ℝ // AB * BC = 20 ∧ AB = 2 * BC}

noncomputable def point_E (AD : ℝ) : ℝ := 1 / 4 * AD
noncomputable def point_F (BC : ℝ) : ℝ := 1 / 4 * BC

theorem area_trapezoid_EFBA (BC AB AD : ℝ) (h1 : AB = 2 * BC) (h2 : AB * BC = 20) :
  let E := point_E AD,
      F := point_F BC,
      EF := AB - E - F,
      height := E
  in EFBA_area EF AB height = 35 / 8 :=
by
  sorry

end area_trapezoid_EFBA_l425_425688


namespace zero_point_in_interval_l425_425850

noncomputable def f (x : ℝ) : ℝ := Real.log x - (1 / 2) ^ (x - 2)

theorem zero_point_in_interval :
  (∃ x₀ ∈ set.Ioo 2 3, f x₀ = 0) := 
by sorry

end zero_point_in_interval_l425_425850


namespace max_k_value_l425_425081

theorem max_k_value :
  let C := {p : ℝ × ℝ | (p.1 - 4)^2 + p.2^2 = 1}, -- circle equation in standard form
      line := {p : ℝ × ℝ | ∃ k : ℝ, p.2 = k * p.1 - 2} in
  ∃ k : ℝ, ∀ p ∈ line, (∃ q ∈ C, ((p.1 - q.1)^2 + (p.2 - q.2)^2 = 1) ∧ (k = 4 / 3)) :=
sorry

end max_k_value_l425_425081


namespace odd_function_at_zero_l425_425262

theorem odd_function_at_zero (f : ℝ → ℝ) (h : ∀ x : ℝ, f(-x) = -f(x)) : f(0) = 0 := 
by
  sorry

end odd_function_at_zero_l425_425262


namespace total_children_in_class_l425_425893

-- Definitions based on the conditions
def boys_born_on_same_day (boys : ℕ) : Prop :=
  boys <= 7

def girls_born_in_same_month (girls : ℕ) : Prop :=
  girls <= 12

-- Main theorem to prove
theorem total_children_in_class (boys girls : ℕ) :
  boys_born_on_same_day boys → girls_born_in_same_month girls → 
  (boys = 7 ∧ girls = 12) → boys + girls = 19 :=
by
  intros H1 H2 H3
  have h_boys : boys = 7 := H3.left
  have h_girls : girls = 12 := H3.right
  rw [h_boys, h_girls]
  exact rfl

end total_children_in_class_l425_425893


namespace sum_of_repeating_decimals_l425_425016

/-- Definitions of the repeating decimals as real numbers. --/
def x : ℝ := 0.3 -- This actually represents 0.\overline{3} in Lean
def y : ℝ := 0.04 -- This actually represents 0.\overline{04} in Lean
def z : ℝ := 0.005 -- This actually represents 0.\overline{005} in Lean

/-- The theorem stating that the sum of these repeating decimals is a specific fraction. --/
theorem sum_of_repeating_decimals : x + y + z = (14 : ℝ) / 37 := 
by 
  sorry -- Placeholder for the proof

end sum_of_repeating_decimals_l425_425016


namespace paulina_convertibles_l425_425597

-- Definitions for conditions
def total_cars : ℕ := 125
def percentage_regular_cars : ℚ := 64 / 100
def percentage_trucks : ℚ := 8 / 100
def percentage_convertibles : ℚ := 1 - (percentage_regular_cars + percentage_trucks)

-- Theorem to prove the number of convertibles
theorem paulina_convertibles : (percentage_convertibles * total_cars) = 35 := by
  sorry

end paulina_convertibles_l425_425597


namespace probability_of_two_clear_days_l425_425261

theorem probability_of_two_clear_days :
  let P_snow := 0.6 in
  let P_clear := 0.4 in
  let choose_5_2 := 10 in -- This is the binomial coefficient for choosing 2 days out of 5
  let prob_two_clears := (P_clear^2) * (P_snow^3) * choose_5_2 in
  prob_two_clears = 216 / 625 :=
by
  -- Definitions
  let P_snow := 0.6
  let P_clear := 0.4
  let choose_5_2 := 10 -- binom 5 2
  let prob_two_clears := (P_clear^2) * (P_snow^3) * choose_5_2
  -- Expected result
  have H: prob_two_clears = 216 / 625, from sorry
  exact H

end probability_of_two_clear_days_l425_425261


namespace exists_positive_n_l425_425927

theorem exists_positive_n {k : ℕ} (h_k : 0 < k) {m : ℕ} (h_m : m % 2 = 1) :
  ∃ n : ℕ, 0 < n ∧ (n^n - m) % 2^k = 0 := 
sorry

end exists_positive_n_l425_425927


namespace congruent_side_length_l425_425255

-- Define the base and area as given conditions of the isosceles triangle.
def base : ℝ := 30
def area : ℝ := 90

-- The desired length of one of the congruent sides of the isosceles triangle.
def desired_length : ℝ := 3 * Real.sqrt 29

-- Proof that the length of one of the congruent sides is as calculated
theorem congruent_side_length : ∃ (a : ℝ), 
  a = desired_length ∧ 
  (∃ (h : ℝ), 2 * area = base * h ∧ (a^2 = (base / 2)^2 + h^2)) :=
by
  -- base and area are given
  use 3 * Real.sqrt 29
  -- height is calculated to be 6, and Pythagorean theorem used to find side length
  use 6
  sorry

end congruent_side_length_l425_425255


namespace inradius_sum_eq_altitude_semiperimeter_inradius_relation_l425_425918

/-- In triangle ABC, C = π/2. CD be the altitude, 
    and the inradii of triangles BDC and ADC be r₁ and r₂
    respectively, with their semi-perimeters s₁ and s₂. 
    We want to prove: (1) r₁ + r₂ + r = h_c; 
    (2) (s₁ ± r₁)² + (s₂ ± r₂)² = (s ± r)². -/

variables {a b c h_c d : ℝ}
variables {s₁ s₂ s r₁ r₂ r : ℝ}
 
/-- Part (1): Prove that r₁ + r₂ + r = h_c -/
theorem inradius_sum_eq_altitude 
  (C_eq : C = π / 2)
  (altitude_CD : true) -- Placeholder as the altitude property is implicit in the inradius formula
  (inradius_BDC_ADC : true) -- Placeholder as we are given the inradii
  (semi_perimeters : true) -- Placeholder as we are given the semi-perimeters
  (inradius_ABC : r = (1/2) * (a * b) / ((a + b + c) / 2)) 
  ( semi_perimeter_ABC : s = (a + b + c) / 2 ) :
  r₁ + r₂ + r = h_c :=
sorry

/-- Part (2): Prove that (s₁ ± r₁)² + (s₂ ± r₂)² = (s ± r)² -/
theorem semiperimeter_inradius_relation 
  (C_eq : C = π / 2)
  (altitude_CD : true) 
  (inradius_BDC_ADC : true) 
  (semi_perimeters : true) 
  (semi_perimeter_squares : s₁^2 + s₂^2 = s^2) 
  (inradius_squares : r₁^2 + r₂^2 = r^2) 
  (sum_inradius_semiperimeter : r₁ * s₁ + r₂ * s₂ = r * s) :
  (s₁ + r₁)² + (s₂ + r₂)² = (s + r)² ∧ (s₁ - r₁)² + (s₂ - r₂)² = (s - r)² :=
sorry

end inradius_sum_eq_altitude_semiperimeter_inradius_relation_l425_425918


namespace arithmetic_to_geometric_l425_425245

theorem arithmetic_to_geometric (a : ℝ) 
(existing_AP : ∃ (d : ℝ), ∀ (n : ℕ), is_arith_prog (1, 1 + d, 1 + 2*d, ...))
(existing_GP : is_geom_prog (1, a, a^2, ...)) :
a ∈ ℕ+ := sorry             -- ℕ+ represents the set of positive integers

end arithmetic_to_geometric_l425_425245


namespace cost_of_four_pencils_and_four_pens_l425_425257

def pencil_cost : ℝ := sorry
def pen_cost : ℝ := sorry

axiom h1 : 8 * pencil_cost + 3 * pen_cost = 5.10
axiom h2 : 3 * pencil_cost + 5 * pen_cost = 4.95

theorem cost_of_four_pencils_and_four_pens : 4 * pencil_cost + 4 * pen_cost = 4.488 :=
by
  sorry

end cost_of_four_pencils_and_four_pens_l425_425257


namespace f_800_value_l425_425193

variable (f : ℝ → ℝ)
variable (f_prop : ∀ (x y : ℝ), 0 < x → 0 < y → f(x * y) = f(x) / y)
variable (f_400 : f 400 = 4)

theorem f_800_value : f 800 = 2 :=
by {
  sorry
}

end f_800_value_l425_425193


namespace fraction_division_l425_425504

theorem fraction_division:
  (1 / 4) / (1 / 8) = 2 :=
by
  sorry

end fraction_division_l425_425504


namespace solutions_eq1_solutions_eq2_l425_425626

noncomputable def equation_sol1 : Set ℝ :=
{ x | x^2 - 8 * x + 1 = 0 }

noncomputable def equation_sol2 : Set ℝ :=
{ x | x * (x - 2) - x + 2 = 0 }

theorem solutions_eq1 : ∀ x ∈ equation_sol1, x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15 :=
by
  intro x hx
  sorry

theorem solutions_eq2 : ∀ x ∈ equation_sol2, x = 2 ∨ x = 1 :=
by
  intro x hx
  sorry

end solutions_eq1_solutions_eq2_l425_425626


namespace part1_part2_l425_425936

open Nat

variables {n m : ℕ}
variables (A : finset (fin n)) (A₁ A₂ … Am : finset A)

-- Declaration of the condition that {A₁, A₂, ..., Am} are pairwise disjoint subsets of A
def pairwise_disjoint_subs (A₁ A₂ … Am : finset A) := 
  ∀ i j, i ≠ j → disjoint (Aᵢ : finset n) (Aⱼ : finset n)

-- Use a definition to denote the size of each Aᵢ
def size_fun (ᵢ : fin m): ℕ := (Aᵢ.card : nat)

theorem part1 
  (h_disjoint : pairwise_disjoint_subs A₁ A₂ … Am)
  (hn : 0 < n) :
  ∑ i in (range m), 1 / nat.choose (n, (size_fun i)) ≤ 1 := 
sorry

theorem part2 
  (h_disjoint : pairwise_disjoint_subs A₁ A₂ … Am)
  (hn : 0 < n) :
  ∑ i in (range m), (nat.choose (n, (size_fun i))) ≥ m^2 :=
sorry

end part1_part2_l425_425936


namespace repeatingDecimals_fraction_eq_l425_425003

noncomputable def repeatingDecimalsSum : ℚ :=
  let x : ℚ := 1 / 3
  let y : ℚ := 4 / 99
  let z : ℚ := 5 / 999
  x + y + z

theorem repeatingDecimals_fraction_eq : repeatingDecimalsSum = 42 / 111 :=
  sorry

end repeatingDecimals_fraction_eq_l425_425003


namespace log_sin_x_eq_a_add_log_cos_x_l425_425874

variable (b x a : ℝ)
variable (hx1 : b > 1)
variable (hx2 : tan x > 0)
variable (hx3 : log b (tan x) = a)
variable (hx4 : cos x > 0)

theorem log_sin_x_eq_a_add_log_cos_x :
  log b (sin x) = a + log b (cos x) :=
sorry

end log_sin_x_eq_a_add_log_cos_x_l425_425874


namespace sum_first_n_terms_l425_425449

def a : ℕ → ℚ
| 0       := 0
| 1       := -2
| 2       := 3
| (n + 3) := 3 * a (n + 2) + 3^((n + 3) - 1)

noncomputable def S_n (n : ℕ) := ∑ i in Finset.range n, a (i + 1)

theorem sum_first_n_terms (n : ℕ) :
  S_n n = (13 + (6 * n - 13) * 3^n) / 4 := by
  sorry

end sum_first_n_terms_l425_425449


namespace august_first_problem_answer_l425_425747

theorem august_first_problem_answer (A : ℕ)
  (h1 : 2 * A = B)
  (h2 : 3 * A - 400 = C)
  (h3 : A + B + C = 3200) : A = 600 :=
sorry

end august_first_problem_answer_l425_425747


namespace simplify_factorial_expression_l425_425240

theorem simplify_factorial_expression : 
  (15.factorial / (12.factorial + 3 * 10.factorial) = 2669) := 
  by sorry

end simplify_factorial_expression_l425_425240


namespace polygon_sides_l425_425279

theorem polygon_sides (n : ℕ) (h : 3 * n * (n * (n - 3)) = 300) : n = 10 :=
sorry

end polygon_sides_l425_425279


namespace range_of_m_l425_425151

noncomputable def root_in_interval (a b m : ℝ) : Prop :=
  ∃ x ∈ set.Icc a b, x^3 - 3 * x + m = 0

theorem range_of_m (a b : ℝ) (h : ∀ (m : ℝ), root_in_interval a b m → m ∈ set.Icc (-2 : ℝ) 2) :
  ∀ m, root_in_interval 0 2 m → m ∈ set.Icc (-2 : ℝ) 2 := sorry

end range_of_m_l425_425151


namespace set_intersection_l425_425860

def setM : Set ℝ := {x | x^2 - 1 < 0}
def setN : Set ℝ := {y | ∃ x ∈ setM, y = Real.log (x + 2)}

theorem set_intersection : setM ∩ setN = {y | 0 < y ∧ y < Real.log 3} :=
by
  sorry

end set_intersection_l425_425860


namespace f_greater_than_one_l425_425939

noncomputable def f (p x : ℝ) : ℝ := - (1 / (2 * p)) * x^2 + x

theorem f_greater_than_one (p : ℝ) (hp : 0 < p) :
  ( ∃ x : ℝ, x ∈ set.Icc 0 (4 / p) ∧ 1 < f p x ) ↔ (2 < p ∧ p < 1 + Real.sqrt 5) := 
sorry

end f_greater_than_one_l425_425939


namespace number_of_rectangles_is_24_l425_425868

-- Define the rectangles on a 1x5 stripe
def rectangles_1x5 : ℕ := 1 + 2 + 3 + 4 + 5

-- Define the rectangles on a 1x4 stripe
def rectangles_1x4 : ℕ := 1 + 2 + 3 + 4

-- Define the overlap (intersection) adjustment
def overlap_adjustment : ℕ := 1

-- Total number of rectangles calculation
def total_rectangles : ℕ := rectangles_1x5 + rectangles_1x4 - overlap_adjustment

theorem number_of_rectangles_is_24 : total_rectangles = 24 := by
  sorry

end number_of_rectangles_is_24_l425_425868


namespace part_I_part_II_l425_425827

open Real

noncomputable def alpha₁ : Real := sorry -- Placeholder for the angle α in part I
noncomputable def alpha₂ : Real := sorry -- Placeholder for the angle α in part II

-- Given a point P(-4, 3) and a point on the terminal side of angle α₁ such that tan(α₁) = -3/4
theorem part_I :
  tan α₁ = - (3 / 4) → 
  (cos (π / 2 + α₁) * sin (-π - α₁)) / (cos (11 * π / 2 - α₁) * sin (9 * π / 2 + α₁)) = - (3 / 4) :=
by 
  intro h
  sorry

-- Given vector a = (3,1) and b = (sin α, cos α) where a is parallel to b such that tan(α₂) = 3
theorem part_II :
  tan α₂ = 3 → 
  (4 * sin α₂ - 2 * cos α₂) / (5 * cos α₂ + 3 * sin α₂) = 5 / 7 :=
by 
  intro h
  sorry

end part_I_part_II_l425_425827


namespace man_l425_425324

theorem man's_rate_in_still_water 
  (V_s V_m : ℝ)
  (with_stream : V_m + V_s = 24)  -- Condition 1
  (against_stream : V_m - V_s = 10) -- Condition 2
  : V_m = 17 := 
by
  sorry

end man_l425_425324


namespace find_x_base_l425_425686

open Nat

def is_valid_digit (n : ℕ) : Prop := n < 10

def interpret_base (digits : ℕ → ℕ) (n : ℕ) : ℕ :=
  digits 2 * n^2 + digits 1 * n + digits 0

theorem find_x_base (a b c : ℕ)
  (ha : is_valid_digit a)
  (hb : is_valid_digit b)
  (hc : is_valid_digit c)
  (h : interpret_base (fun i => if i = 0 then c else if i = 1 then b else a) 20 = 2 * interpret_base (fun i => if i = 0 then c else if i = 1 then b else a) 13) :
  100 * a + 10 * b + c = 198 :=
by
  sorry

end find_x_base_l425_425686


namespace available_milk_for_me_l425_425509

def initial_milk_litres : ℝ := 1
def myeongseok_milk_litres : ℝ := 0.1
def mingu_milk_litres : ℝ := myeongseok_milk_litres + 0.2
def minjae_milk_litres : ℝ := 0.3

theorem available_milk_for_me :
  initial_milk_litres - (myeongseok_milk_litres + mingu_milk_litres + minjae_milk_litres) = 0.3 :=
by sorry

end available_milk_for_me_l425_425509


namespace goldbach_stronger_l425_425291

-- Definitions for prime numbers and the sum condition
def is_prime (n : ℕ) : Prop := nat.prime n

def sum_138 (p q : ℕ) : Prop :=
  is_prime p ∧
  is_prime q ∧
  p + q = 138 ∧
  p ≠ q

-- Main theorem to prove that the maximum difference is 124
theorem goldbach_stronger :
  ∃ (p q : ℕ), sum_138 p q ∧ q - p = 124 :=
by {
  sorry
}

end goldbach_stronger_l425_425291


namespace switch_pairs_bound_l425_425967

theorem switch_pairs_bound (odd_blocks_n odd_blocks_prev : ℕ) 
  (switch_pairs_n switch_pairs_prev : ℕ)
  (H1 : switch_pairs_n = 2 * odd_blocks_n)
  (H2 : odd_blocks_n ≤ switch_pairs_prev) : 
  switch_pairs_n ≤ 2 * switch_pairs_prev :=
by
  sorry

end switch_pairs_bound_l425_425967


namespace no_solution_for_x4_plus_y4_eq_z4_l425_425975

theorem no_solution_for_x4_plus_y4_eq_z4 :
  ∀ (x y z : ℤ), x ≠ 0 → y ≠ 0 → z ≠ 0 → gcd (gcd x y) z = 1 → x^4 + y^4 ≠ z^4 :=
sorry

end no_solution_for_x4_plus_y4_eq_z4_l425_425975


namespace complex_fraction_conjugate_modulus_l425_425145

noncomputable def z : ℂ := 4 + 3 * complex.i

theorem complex_fraction_conjugate_modulus :
  (complex.conj z) / (complex.norm z) = (4 / 5) - (3 / 5) * complex.i :=
by
  sorry

end complex_fraction_conjugate_modulus_l425_425145


namespace f_is_odd_k_range_l425_425053

noncomputable def f : ℝ → ℝ := sorry

-- Strictly monotonic
axiom strictly_monotonic : ∀ x y : ℝ, x < y → f x < f y

-- Functional equation
axiom functional_eq : ∀ x y : ℝ, f (x + y) = f x + f y

-- Given value
axiom f_one : f 1 = 2

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := sorry

theorem k_range (t : ℝ) (ht : t > 2) : ∃ k : ℝ, k < 2 * sqrt 2 - 1 ∧ f (k * log 2 t) + f (log 2 t - (log 2 t)^2 - 2) < 0 := sorry

end f_is_odd_k_range_l425_425053


namespace disjoint_sets_exist_l425_425143

open Finset

theorem disjoint_sets_exist {n k m : ℕ} (S : Finset ℕ) (A : Finset ℕ) 
  (hA : A ⊆ S) (hA_card : A.card = k) (hS : S = range (n + 1)) 
  (hn : n > (m - 1) * (Nat.choose k 2 + 1)) : 
  ∃ (t : Fin ℕ → ℕ) (H : ∀ j, j < m → t j ∈ S), 
  ∀ i j, i ≠ j → ((A.map (Function.add (t i))).disjoint (A.map (Function.add (t j)))) := 
by
  sorry

end disjoint_sets_exist_l425_425143


namespace total_goals_in_league_l425_425274

variables (g1 g2 T : ℕ)

-- Conditions
def equal_goals : Prop := g1 = g2
def players_goals : Prop := g1 = 30
def total_goals_percentage : Prop := (g1 + g2) * 5 = T

-- Theorem to prove: Given the conditions, the total number of goals T should be 300
theorem total_goals_in_league (h1 : equal_goals g1 g2) (h2 : players_goals g1) (h3 : total_goals_percentage g1 g2 T) : T = 300 :=
sorry

end total_goals_in_league_l425_425274


namespace least_possible_xy_l425_425063

theorem least_possible_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 48 :=
by
  sorry

end least_possible_xy_l425_425063


namespace right_triangle_inequality_l425_425871

-- Definitions of the conditions as Lean hypotheses
variables {a b c : ℝ}
hypothesis h_right_triangle : a * a = b * b + c * c

-- The main statement to prove
theorem right_triangle_inequality
  (h_right_triangle : a * a = b * b + c * c) :
    (a + b + c) * (1 / a + 1 / b + 1 / c) ≥ 5 + 3 * Real.sqrt 2 :=
sorry

end right_triangle_inequality_l425_425871


namespace prob_subset_abc_l425_425814

theorem prob_subset_abc {α : Type*} (s : finset α) (t : finset α) :
  s = {a, b, c, d, e} → t = {a, b, c} → 
  ∃ p : ℚ, 
  p = 1 / 4 ∧ 
  (∃ ss : finset (finset α), 
    ss = s.powerset ∧
    ∀ x ∈ ss, x ⊆ t → ↑ss.card / ↑s.powerset.card = p) :=
by convince yourself of the details later using the proof steps in pen and paper sorry

end prob_subset_abc_l425_425814


namespace largest_number_product_l425_425298

theorem largest_number_product (n : ℕ) (digits : List ℕ)
  (h_sum_squares : digits.sum (λ d, d^2) = 45)
  (h_increasing : ∀ i j, i < j → i < digits.length → j < digits.length → digits[i] < digits[j])
  (h_n : nat.ofDigits 10 digits = n) :
  (digits.sorted.digitsproduct = 24) :=
  sorry

end largest_number_product_l425_425298


namespace walnuts_left_in_burrow_l425_425697

-- Define the initial quantities
def boy_initial_walnuts : Nat := 6
def boy_dropped_walnuts : Nat := 1
def initial_burrow_walnuts : Nat := 12
def girl_added_walnuts : Nat := 5
def girl_eaten_walnuts : Nat := 2

-- Define the resulting quantity and the proof goal
theorem walnuts_left_in_burrow : boy_initial_walnuts - boy_dropped_walnuts + initial_burrow_walnuts + girl_added_walnuts - girl_eaten_walnuts = 20 :=
by
  sorry

end walnuts_left_in_burrow_l425_425697


namespace cyrus_written_pages_on_fourth_day_l425_425765

theorem cyrus_written_pages_on_fourth_day :
  ∀ (total_pages first_day second_day third_day fourth_day remaining_pages: ℕ),
  total_pages = 500 →
  first_day = 25 →
  second_day = 2 * first_day →
  third_day = 2 * second_day →
  remaining_pages = total_pages - (first_day + second_day + third_day + fourth_day) →
  remaining_pages = 315 →
  fourth_day = 10 :=
by
  intros total_pages first_day second_day third_day fourth_day remaining_pages
  intros h_total h_first h_second h_third h_remain h_needed
  sorry

end cyrus_written_pages_on_fourth_day_l425_425765


namespace min_distance_sum_l425_425842

open Real

/--
Given a point P on the parabola \( y^2 = 4 x \), let \( d_1 \) be the distance from point \( P \) to the axis of the parabola, and \( d_2 \) be the distance to the line \( x + 2 y - 12 = 0 \). The minimum value of \( d_1 + d_2 \) is \( \frac{11 \sqrt{5}}{5} \).
-/
theorem min_distance_sum : 
  ∃ P : ℝ × ℝ, (P.2^2 = 4 * P.1) ∧ (let d1 := dist (P.1, P.2) (P.1, 0) in
                                   let d2 := |P.1 + 2 * P.2 - 12| / (sqrt (1 ^ 2 + 2 ^ 2)) in
                                   d1 + d2 = 11 * sqrt 5 / 5) ∧
                                   ∀ Q : ℝ × ℝ, (Q.2^2 = 4 * Q.1) → 
                                   let d1_Q := dist (Q.1, Q.2) (Q.1, 0) in
                                   let d2_Q := |Q.1 + 2 * Q.2 - 12| / (sqrt (1 ^ 2 + 2 ^ 2)) in
                                   d1 + d2 ≤ d1_Q + d2_Q := sorry
 
end min_distance_sum_l425_425842


namespace calculate_discount_l425_425363

def membership_discount_percentage (P P_coupon P_paid : ℕ) := 
  let discount_amount := P_coupon - P_paid in
  (discount_amount * 100) / P_coupon

theorem calculate_discount 
  (P : ℕ) (C : ℕ) (P_coupon : ℕ) (P_paid : ℕ) 
  (h1 : P = 120) (h2 : C = 10) (h3 : P_coupon = P - C) (h4 : P_paid = 99) :
  membership_discount_percentage P P_coupon P_paid = 10 := 
by 
  sorry

end calculate_discount_l425_425363


namespace initial_customers_count_l425_425379

theorem initial_customers_count (n m : ℕ) (h : m = 12) (j : n = 9) : n + m = 21 :=
by
  rw [j, h]
  exact Nat.add_comm 9 12 ▸ rfl

end initial_customers_count_l425_425379


namespace domain_proof_l425_425996

noncomputable def domain_of_function (x : ℝ) : Prop :=
  sqrt ((2 + x) / (1 - x)) + sqrt (x ^ 2 - x - 2) ≤ sqrt ((2 + x) / (1 - x)) + sqrt (x ^ 2 - x - 2)

-- Conditions converted to Lean definitions
def cond1 (x : ℝ) : Prop := (2 + x) / (1 - x) ≥ 0
def cond2 (x : ℝ) : Prop := x ^ 2 - x - 2 ≥ 0

-- Desired domain of the function.
def target_domain (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ -1

-- Stating the proof problem:
theorem domain_proof (x : ℝ) (h1 : cond1 x) (h2 : cond2 x) : domain_of_function x ↔ target_domain x :=
sorry

end domain_proof_l425_425996


namespace exists_solution_for_an_l425_425805

theorem exists_solution_for_an (n : ℕ) (a : ℝ) (x_i : ℕ → ℝ) (h : ∀ i, 0 ≤ x_i i ∧ x_i i ≤ 3) :
  (∃ x ∈ Icc 0 3, (∑ i in Finset.range n, |x - x_i i| = a * n)) ↔ a = 3 / 2 :=
by
  sorry

end exists_solution_for_an_l425_425805


namespace simplify_expression_l425_425396

variable (x y : ℝ)

-- Define the proposition
theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y := 
by
  sorry

end simplify_expression_l425_425396


namespace beast_of_war_running_time_correct_l425_425286

def running_time_millennium : ℕ := 120

def running_time_alpha_epsilon (rt_millennium : ℕ) : ℕ := rt_millennium - 30

def running_time_beast_of_war (rt_alpha_epsilon : ℕ) : ℕ := rt_alpha_epsilon + 10

theorem beast_of_war_running_time_correct :
  running_time_beast_of_war (running_time_alpha_epsilon running_time_millennium) = 100 := by sorry

end beast_of_war_running_time_correct_l425_425286


namespace calculate_g_inv_sum_l425_425932

def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else x^2 - 4*x + 5

theorem calculate_g_inv_sum :
  ∀ (g_inv : ℝ → ℝ),
  (∀ y, g (g_inv y) = y) →
  g_inv (-1) + g_inv 1 + g_inv 4 = 4 :=
by
  intro g_inv h
  have h_inv_neg1 : g_inv (-1) = 3 := sorry
  have h_inv_1 : g_inv 1 = 2 := sorry
  have h_inv_4 : g_inv 4 = -1 := sorry
  calc 
    g_inv (-1) + g_inv 1 + g_inv 4
        = 3 + 2 + (-1) : by rw [h_inv_neg1, h_inv_1, h_inv_4]
    ... = 4 : by norm_num

end calculate_g_inv_sum_l425_425932


namespace find_QS_length_l425_425931

variable (P Q R S : Point)
variable (PR QR : ℝ)
variable (area_PQR : ℝ)
variable (right_angle_Q : RightAngle Q)
variable (circle_intersects_PRS : IntersectsCircleWithDiameter QR PR S)
variable (QS : ℝ)
variable (PR_value : PR = 40) 
variable (area_value : area_PQR = 200)

theorem find_QS_length : QS = 10 :=
by
  have h1 : area_PQR = (1 / 2) * PR * QS := sorry
  have h2 : area_value := sorry
  have h3 : PR_value := sorry
  have h4 : QS = area_value * 2 / PR_value := sorry
  show QS = 10 from sorry

end find_QS_length_l425_425931


namespace solution_set_equiv_l425_425929

theorem solution_set_equiv {n : ℕ} (hn : n > 0) (x : ℝ) :
  (∑ k in finset.range(2 * n) + 1, real.sqrt(x^2 - 2 * (k : ℝ) * x + (k : ℝ)^2) =
    real.abs(2 * n * x - n - 2 * n^2)) ↔ 
  (x ≥ 2 * n ∨ (∃ i, x = (i + 2 * n + 1) / 2 ∧ 0 ≤ i ∧ i < n * 2) ∨ x < (2 * n + 1) / 2) := 
sorry

end solution_set_equiv_l425_425929


namespace profit_calculation_l425_425680

noncomputable def profit_percent (purchase_price repair_costs selling_price : ℝ) : ℝ :=
  ((selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs)) * 100

theorem profit_calculation :
  ∀ (purchase_price repair_costs selling_price : ℝ),
    purchase_price = 34000 →
    repair_costs = 12000 →
    selling_price = 65000 →
    profit_percent purchase_price repair_costs selling_price ≈ 41.30 :=
by
  intros purchase_price repair_costs selling_price h1 h2 h3
  rw [h1, h2, h3]
  show profit_percent 34000 12000 65000 ≈ 41.30
  sorry

end profit_calculation_l425_425680


namespace tangent_line_equation_l425_425788

theorem tangent_line_equation :
  ∀ x y : ℝ,
  (y = x^3 - x + 3) →
  ((x = 1 ∧ y = 3) →
  (∃ c : ℝ, y = c * (x - 1) + 3 ∧ c = 2) →
  (2 * x - y + 1 = 0)) :=
begin
  intros x y h_curve h_tangent line_eq,
  sorry
end

end tangent_line_equation_l425_425788


namespace minimum_S_l425_425096

noncomputable def S (n : ℕ+) : ℤ :=
  n * n - 12 * n

theorem minimum_S : (∃ n : ℕ+, S n = -36) :=
begin
  use 6,
  simp [S],
end

end minimum_S_l425_425096


namespace calculator_display_after_120_presses_l425_425247

-- Define the transformation
def transform (x : ℝ) : ℝ := 1 / (1 - x)

-- Define the initial condition
def initial_number : ℝ := 7

-- Define the number of presses
def num_presses : ℕ := 120

-- Define the function to get the displayed number after n presses
def displayed_number (n : ℕ) (x : ℝ) : ℝ :=
  Nat.iterate transform n x

-- The main theorem stating that the number displayed after 120 presses is 7.
theorem calculator_display_after_120_presses : displayed_number num_presses initial_number = 7 := by
  -- placeholder for proof
  sorry

end calculator_display_after_120_presses_l425_425247


namespace sum_of_first_10_terms_l425_425103

variable (a : ℕ → ℕ) (n : ℕ)

def direction_vector (v : ℕ × ℕ) : Prop :=
  v = (a n.succ - a n, a n.succ - a n)

def is_direction_of_y_eq_x (v : ℕ × ℕ) : Prop :=
  v = (1, 1)

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 1

def first_term_is_five (a : ℕ → ℕ) : Prop :=
  a 1 = 5

theorem sum_of_first_10_terms (h1 : direction_vector (a n) (a n) = (1,1))
    (h2 : is_direction_of_y_eq_x (1, 1)) 
    (h3 : first_term_is_five a)
    (h4 : arithmetic_sequence a) :
  (∑ i in (Finset.range 10), a i) = 95 := 
by
  sorry

end sum_of_first_10_terms_l425_425103


namespace frac_mul_square_l425_425755

theorem frac_mul_square 
  : (8/9)^2 * (1/3)^2 = 64/729 := 
by 
  sorry

end frac_mul_square_l425_425755


namespace min_magnitude_AG_l425_425078

open Real

variables {A B C G : EuclideanGeometry.Point}
variable (centroid_G : EuclideanGeometry.is_centroid G A B C)
variable (angle_A_eq_120_deg : ∠ B A C = 2 * π / 3)
variable (dot_product_AB_AC_eq_neg2 : (A - B) • (A - C) = -2)

theorem min_magnitude_AG (A B C G : EuclideanGeometry.Point)
  (centroid_G : EuclideanGeometry.is_centroid G A B C)
  (angle_A_eq_120_deg : ∠ B A C = 2 * π / 3)
  (dot_product_AB_AC_eq_neg2 : (A - B) • (A - C) = -2) :
  ∃ (k : ℝ), k = 2/3 ∧ ∀ (x : ℝ), |(G - A)| = k :=
sorry

end min_magnitude_AG_l425_425078


namespace rotated_line_eq_l425_425641

-- Definitions and conditions from part a)
def original_line (x y : ℝ) : Prop := 2 * x - y - 4 = 0
def rotation_angle : ℝ := Real.pi / 4

-- Main theorem stating the result
theorem rotated_line_eq :
  ∃ (A B C : ℝ), A * 3 + B * 1 + C * (-6) = 0 :=
sorry

end rotated_line_eq_l425_425641


namespace average_marks_l425_425329

-- Define the conditions
variables (M P C : ℝ)
variables (h1 : M + P = 60) (h2 : C = P + 10)

-- Define the theorem statement
theorem average_marks : (M + C) / 2 = 35 :=
by {
  sorry -- Placeholder for the proof.
}

end average_marks_l425_425329


namespace ripe_bananas_weight_correct_l425_425589

/-- Conditions for the problem -/
def num_bunches1 := 6
def bananas_per_bunch1 := 8
def num_bunches2 := 5
def bananas_per_bunch2 := 7
def weight_per_banana := 100
def fraction_ripe := 3 / 4

/-- Total number of bananas in the box -/
def total_bananas : ℕ := (num_bunches1 * bananas_per_bunch1) + (num_bunches2 * bananas_per_bunch2)

/-- Number of ripe bananas (rounded down to the nearest whole banana) -/
def ripe_bananas : ℕ := Int.floor (fraction_ripe * total_bananas)

/-- Total weight of ripe bananas -/
def total_ripe_banana_weight : ℕ := ripe_bananas * weight_per_banana

theorem ripe_bananas_weight_correct : total_ripe_banana_weight = 6200 := by
  sorry

end ripe_bananas_weight_correct_l425_425589


namespace find_ellipse_equation_find_line_equation_l425_425057

-- Conditions in Lean definitions

-- The ellipse C
def ellipseC (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- The circle
def circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Given constants from problem: a, b, c, focus distance
constant a b c : ℝ
constant a_gt_b_gt_zero : a > b ∧ b > 0
constant focus_distance : c = sqrt 2

-- Existence of M on ellipse
def point_on_ellipseC (M : ℝ × ℝ) : Prop :=
  ellipseC M.1 M.2 a b

-- Line passing through focus
def line_passing_through_focus (l : ℝ → ℝ) : Prop :=
  ∃ A B, ellipseC A.1 A.2 a b ∧ ellipseC B.1 B.2 a b ∧ A ≠ B ∧ l(F1) = A ∧ l(F1) = B

-- Proof problem equivalent to find the equation of ellipse
theorem find_ellipse_equation :
  (∃ a b : ℝ, (a > b ∧ b > 0 ∧ focus_distance = 2 * sqrt 2) →
  ellipseC = (λ x y, x^2 / 4 + y^2 / 2 = 1)) :=
sorry

-- Proof problem equivalent to finding equation of the line l
theorem find_line_equation (F1: ℝ × ℝ) :
  (∃ l : ℝ → ℝ, line_passing_through_focus l ∧ 
  (∃ P, ellipseC P.1 P.2 a b ∧ (λ A B, P = (A.1 + B.1, A.2 + B.2))) →
  (l = (λ x, sqrt 2 / 2 * x + 1) ∨ l = (λ x, -sqrt 2 / 2 * x - 1))) :=
sorry

end find_ellipse_equation_find_line_equation_l425_425057


namespace law_of_sines_implies_isosceles_isosceles_does_not_imply_law_of_sines_l425_425175

noncomputable theory
open_locale classical

structure Triangle := 
(a b c : ℝ) -- sides
(A B C : ℝ) -- angles

def law_of_sines (t : Triangle) : Prop :=
(t.a / real.sin t.C = t.b / real.sin t.A) ∧
(t.b / real.sin t.A = t.c / real.sin t.B)

def is_isosceles (t : Triangle) : Prop :=
(t.a = t.b ∨ t.b = t.c ∨ t.a = t.c)

theorem law_of_sines_implies_isosceles (t : Triangle) :
  law_of_sines t → is_isosceles t :=
by {
  sorry
}

theorem isosceles_does_not_imply_law_of_sines (t : Triangle) :
  ¬ is_isosceles t → law_of_sines t :=
by {
  sorry
}

end law_of_sines_implies_isosceles_isosceles_does_not_imply_law_of_sines_l425_425175


namespace football_goal_average_increase_l425_425706

theorem football_goal_average_increase :
  ∀ (A : ℝ), 4 * A + 2 = 8 → (8 / 5) - A = 0.1 :=
by
  intro A
  intro h
  sorry -- Proof to be filled in

end football_goal_average_increase_l425_425706


namespace pentagon_distance_l425_425351

/-- Let \(ABCDE\) be a convex pentagon inscribed in a circle. Let \(a\), \(b\), and \(c\) be the distances
from point \(A\) to lines \(BC\), \(CD\), and \(DE\) respectively. The distance from \(A\) to the line \(BE\), 
denoted as \(d\), is given by \(d = \frac{a \cdot c}{b}\). -/
theorem pentagon_distance (ABCDE : Type)
  [convex_pentagon_inscribed_in_circle ABCDE]
  (A B C D E : ABCDE)
  (a b c : ℝ)
  (dist_A_BC : distance_from A (line BC) = a)
  (dist_A_CD : distance_from A (line CD) = b)
  (dist_A_DE : distance_from A (line DE) = c) :
  ∃ d : ℝ, distance_from A (line BE) = d ∧ d = (a * c) / b :=
by sorry

end pentagon_distance_l425_425351


namespace divide_L_shaped_plaque_into_four_equal_parts_l425_425926

-- Definition of an "L"-shaped plaque and the condition of symmetric cuts
def L_shaped_plaque (a b : ℕ) : Prop := (a > 0) ∧ (b > 0)

-- Statement of the proof problem
theorem divide_L_shaped_plaque_into_four_equal_parts (a b : ℕ) (h : L_shaped_plaque a b) :
  ∃ (p1 p2 : ℕ → ℕ → Prop),
    (∀ x y, p1 x y ↔ (x < a/2 ∧ y < b/2)) ∧
    (∀ x y, p2 x y ↔ (x < a/2 ∧ y >= b/2) ∨ (x >= a/2 ∧ y < b/2) ∨ (x >= a/2 ∧ y >= b/2)) :=
sorry

end divide_L_shaped_plaque_into_four_equal_parts_l425_425926


namespace samia_walking_distance_l425_425236

theorem samia_walking_distance
  (speed_bike : ℝ)
  (speed_walk : ℝ)
  (total_time : ℝ) 
  (fraction_bike : ℝ) 
  (d : ℝ)
  (walking_distance : ℝ) :
  speed_bike = 15 ∧ 
  speed_walk = 4 ∧ 
  total_time = 1 ∧ 
  fraction_bike = 2/3 ∧ 
  walking_distance = (1/3) * d ∧ 
  (53 * d / 180 = total_time) → 
  walking_distance = 1.1 := 
by 
  sorry

end samia_walking_distance_l425_425236


namespace simplify_expression_l425_425398

variables {x y : ℝ}
-- Ensure that x and y are not zero to avoid division by zero errors.
theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y :=
sorry

end simplify_expression_l425_425398


namespace area_comparison_of_triangles_l425_425988

open Real
open EuclideanGeometry

noncomputable def problem_statement (A B C A' B' C' : Point) (circumcircle : Circle Point) : Prop :=
  (triangle_inscribed_in_circle A B C circumcircle) ∧
  (angle_bisectors_meet_circle A B C A' B' C' circumcircle) →
  area_of_triangle A' B' C' ≥ area_of_triangle A B C

theorem area_comparison_of_triangles (A B C A' B' C' : Point) (circumcircle : Circle Point) :
  problem_statement A B C A' B' C' circumcircle :=
begin
  sorry
end

end area_comparison_of_triangles_l425_425988


namespace trajectory_of_P_is_ellipse_l425_425077

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem trajectory_of_P_is_ellipse
    (A B : ℝ × ℝ)
    (M : ℝ × ℝ)
    (P : ℝ × ℝ)
    (l : ℝ → ℝ)
    (hA : A = (-1, 0))
    (hB : B = (1, 0))
    (hM : distance A M = 4)
    (hl : ∃ x y, l x = y ∧ ∃ t, t ∈ ℝ → P = (t, l t) ∧ t ∈ [A.1, B.1]) : 
    ∃ (C : ℝ × ℝ → Prop),
    (∀ P, C P ↔ distance A P + distance B P = 4) ∧
    C P ∧
    (λ x y, C (x, y) → (x^2 / 4 + y^2 / 3 = 1)) := 
    sorry 

end trajectory_of_P_is_ellipse_l425_425077


namespace solution_set_of_f_greater_than_2x_plus_4_l425_425995

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, x ∈ ℝ
axiom f_at_neg1 : f (-1) = 2
axiom f_deriv_pos : ∀ x : ℝ, (deriv f x) > 2

theorem solution_set_of_f_greater_than_2x_plus_4 : { x : ℝ | f x > 2 * x + 4 } = Ioi (-1) :=
by
  sorry

end solution_set_of_f_greater_than_2x_plus_4_l425_425995


namespace f_neg_2018_l425_425472

def f : ℝ → ℝ :=
  λ x, if x > 0 then 3^(3 + (Real.log x / Real.log 2)) else 0 -- placeholder for recursive function

-- Property stating the value of the function at x <= 0
axiom f_recurrence : ∀ x ≤ 0, f x = f (x + 0.5)

-- Prove the specific value for f(-2018)
theorem f_neg_2018 : f (-2018) = 9 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end f_neg_2018_l425_425472


namespace coeff_of_x_squared_in_expansion_middle_term_in_expansion_l425_425808

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := nat.choose n k

-- Define the conditions and the proof problems

-- Part 1: Coefficient of x^2 in the expansion
theorem coeff_of_x_squared_in_expansion :
  (binom 10 3 = binom 10 7) → 
  (∀ (x : ℝ), coeff_of_term (sqrt x + 1/(2*sqrt(sqrt(x)))) 10 2 = 105/8) :=
by sorry

-- Part 2: Middle term of the expansion
theorem middle_term_in_expansion :
  (binom 8 0 * 1 = 1 ∧
   binom 8 1 * (1/2) = 8/2 ∧
   binom 8 2 * (1/2)^2 = (8^2 - 8)/8) →
  (∀ (x : ℝ), middle_term (sqrt x + 1/(2*sqrt(sqrt(x)))) 8 = 35/8 * x) :=
by sorry

end coeff_of_x_squared_in_expansion_middle_term_in_expansion_l425_425808


namespace repeatingDecimals_fraction_eq_l425_425004

noncomputable def repeatingDecimalsSum : ℚ :=
  let x : ℚ := 1 / 3
  let y : ℚ := 4 / 99
  let z : ℚ := 5 / 999
  x + y + z

theorem repeatingDecimals_fraction_eq : repeatingDecimalsSum = 42 / 111 :=
  sorry

end repeatingDecimals_fraction_eq_l425_425004


namespace lattice_points_count_l425_425501

theorem lattice_points_count : ∃ n : ℕ, n = 8 ∧ (∃ x y : ℤ, x^2 - y^2 = 51) :=
by
  sorry

end lattice_points_count_l425_425501


namespace complement_of_union_l425_425586

open Set

-- Definitions of the sets U, A and B
def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {x | x ∈ ℤ ∧ x^2 - 5 * x + 4 < 0}.to_set_of 

-- Theorem stating the required proof
theorem complement_of_union :
  compl (A ∪ B) U = {0, 4, 5} :=
by 
  sorry

end complement_of_union_l425_425586


namespace AFG_is_isosceles_l425_425568

-- Definitions of the geometric entities as per conditions

-- Axiomatically define the points and their relationships as per isosceles trapezoid
variables {A B C D E F G : Point}

-- Additional necessary conditions and axioms as per the problem statement
axiom is_isosceles_trapezoid (ABCD : IsoscelesTrapezoid A B C D) : 
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ A ≠ C ∧ B ≠ D ∧ ABCD.AB.parallel ABCD.CD

axiom inscribed_circle (ω : Circle) (BCD : Triangle B C D) (meets_at : MeetCircleAt ω BCD E) :
    ω.inscribed_in BCD ∧ E ∈ CD ∧ Circle.is_tangent_at ω CD E

axiom point_F_on_bisector (F : Point) (D A C : Point) (bisector : AngleBisector (∠ D A C) F) :
    is_internal_angle_bisector D A C F

axiom EF_perp_CD (E F CD : Line) : (Line.perpendicular_to E F CD)

axiom circumscribed_circle (circum_ACF : Circle) (ACF : Triangle A C F) (meets_at : MeetCircleAt circum_ACF ACF C G) :
    circum_ACF.circumscribed_in ACF ∧ C ∈ CD ∧ G ∈ CD

-- Theorem to prove triangle AFG is isosceles.
theorem AFG_is_isosceles (ABCD : IsoscelesTrapezoid A B C D) (ω : Circle) (BCD : Triangle B C D) 
    (triangle_inscribed : MeetCircleAt ω BCD E) (F : Point) (bisector : AngleBisector (∠ D A C) F) 
    (EF_CD_perpendicular : Line.perpendicular_to E F (Line.mk CD)) 
    (circum_ACF : Circle) (meet_ACF_at_CG : MeetCircleAt circum_ACF (Triangle.mk A C F) C G) :
    Triangle.is_isosceles (Triangle.mk A F G) :=
sorry

end AFG_is_isosceles_l425_425568


namespace petya_cut_paper_impossible_l425_425221

theorem petya_cut_paper_impossible :
  ∀ (cuts : ℕ) (vertices : ℕ), cuts = 100 → vertices = 302 → false :=
by {
  intros cuts vertices h_cuts h_vertices,
  -- We need to show that with 100 cuts, having 302 vertices is impossible
  -- Beginning with 100 cuts leading to 101 polygons with minimum 303 vertices
  sorry
}

end petya_cut_paper_impossible_l425_425221


namespace repeating_decimals_sum_l425_425011

theorem repeating_decimals_sum : 
  (0.3333333333333333 : ℝ) + (0.0404040404040404 : ℝ) + (0.005005005005005 : ℝ) = (14 / 37 : ℝ) :=
by {
  sorry
}

end repeating_decimals_sum_l425_425011


namespace circle_chord_length_equal_l425_425174

def equation_of_circle (D E F : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0

def distances_equal (D E F : ℝ) : Prop :=
  (D^2 ≠ E^2 ∧ E^2 > 4 * F) → 
  (∀ x y : ℝ, (x^2 + y^2 + D * x + E * y + F = 0) → (x = -D/2) ∧ (y = -E/2) → (abs x = abs y))

theorem circle_chord_length_equal (D E F : ℝ) (h : D^2 ≠ E^2 ∧ E^2 > 4 * F) :
  distances_equal D E F :=
by
  sorry

end circle_chord_length_equal_l425_425174


namespace find_numbers_l425_425800

def is_permutation (a b : ℕ) : Prop :=
  multiset.to_finset (multiset.of_list (nat.digits 10 a)) = 
  multiset.to_finset (multiset.of_list (nat.digits 10 b))

def consists_only_of_digit (n digit : ℕ) : Prop :=
  (nat.digits 10 n).all (λ x => x = digit)

theorem find_numbers : ∃ (a b : ℕ), 
  is_permutation a b ∧ consists_only_of_digit (abs (a - b)) 1 :=
by
  use 234567809, 345678920
  sorry

end find_numbers_l425_425800


namespace area_of_triangle_hyperbola_l425_425070

theorem area_of_triangle_hyperbola
  (a : ℝ) (ha : a > 0)
  (F1 F2 A B : ℝ × ℝ)
  (hyp : (F1.1 + F2.1) / 2 = 0)
  (hyperbola : ∀ (x y : ℝ), y^2 = 6 * (x^2 / a^2 - 1))
  (AF1_dist : real.sqrt ((A.1 - F1.1)^2 + (A.2 - F1.2)^2) = 2 * a)
  (angle_F1AF2 : ∠ F1 A F2 = 2 * real.pi / 3)
  (B_mem_hyperbola : B.2^2 = 6 * (B.1^2 / a^2 - 1)) :
  let b := real.sqrt 6 in
  area_of_triangle F1 B F2 = 6 * real.sqrt 3 :=
begin
  sorry
end

end area_of_triangle_hyperbola_l425_425070


namespace three_digit_perfect_squares_divisible_by_4_l425_425127

/-- Proving the number of three-digit perfect squares that are divisible by 4 is 6 -/
theorem three_digit_perfect_squares_divisible_by_4 : 
  (Finset.card (Finset.filter (λ n, n ∣ 4) (Finset.image (λ k, k * k) (Finset.Icc 10 31)))) = 6 :=
by
  sorry

end three_digit_perfect_squares_divisible_by_4_l425_425127


namespace min_distance_AB_l425_425930

noncomputable def circle_radius : ℝ := real.sqrt 13

noncomputable def parabola_distance (t : ℝ) : ℝ := real.sqrt ((t^4 + 73 - 24 * t))

theorem min_distance_AB :
  ∃ t : ℝ, ∀ (A : ℝ × ℝ) (B : ℝ × ℝ),
    (A = (x, y) ∧ (x - 8)^2 + (y - 3)^2 = 13) ∧
    (B = (t^2, 4 * t) ∧ (4 * t)^2 = 16 * t^2) →
      real.sqrt ((t^4 + 73 - 24 * t)) - circle_radius = real.sqrt ((∃ t, parabola_distance t)) :=
sorry

end min_distance_AB_l425_425930


namespace gold_coin_value_l425_425360

theorem gold_coin_value {G : ℝ} :
  (∀ (silver_value gold_coins silver_coins cash total_value : ℝ), 
    silver_value = 25 ∧ gold_coins = 3 ∧ silver_coins = 5 ∧ cash = 30 ∧ total_value = 305 →
    3 * G + 5 * silver_value + cash = total_value) →
  G = 50 :=
begin
  intros h,
  specialize h 25 3 5 30 305,
  simp only [*, mul_add, mul_comm (5 : ℝ)] at h,
  linarith,
end

end gold_coin_value_l425_425360


namespace problem_l425_425817

theorem problem (a b : ℝ) (h1 : 3 ^ a = 5) (h2 : log 27 3 = b) : 9 ^ (a - 3 * b) = 25 / 9 :=
by 
  sorry

end problem_l425_425817


namespace remainder_of_polynomial_division_l425_425668

theorem remainder_of_polynomial_division :
  ∀ (x : ℂ), ((x + 2) ^ 2023) % (x^2 + x + 1) = 1 :=
by
  sorry

end remainder_of_polynomial_division_l425_425668


namespace simplify_factorial_expression_l425_425242

theorem simplify_factorial_expression :
  (15.factorial : ℚ) / ((12.factorial) + (3 * 10.factorial)) = 2668 := by
  sorry

end simplify_factorial_expression_l425_425242


namespace one_fourth_div_one_eighth_l425_425502

theorem one_fourth_div_one_eighth : (1 / 4) / (1 / 8) = 2 := by
  sorry

end one_fourth_div_one_eighth_l425_425502


namespace range_of_m_l425_425570

theorem range_of_m 
  (α : ∀ x : ℝ, x^2 - 8*x + 12 > 0) 
  (β : ∀ x : ℝ, |x - m| ≤ m^2) 
  (h : ∀ x : ℝ, β x → α x): 
  -2 < m ∧ m < 1 := 
sorry

end range_of_m_l425_425570


namespace solve_equation_l425_425831

structure Point where
  x : ℝ
  y : ℝ

def hoplus (a b : Point) : Point := sorry -- Definition of the \oplus operation 

def solve_for_x : Point := ⟨(1 - Real.sqrt 2) / 2, ((1 - Real.sqrt 2) * Real.sqrt 3) / 2⟩

theorem solve_equation : 
  ∃ x : Point, (hoplus (hoplus x ⟨0,0⟩) ⟨1,1⟩) = ⟨1, -1⟩ ∧ x = solve_for_x :=
by 
  sorry

end solve_equation_l425_425831


namespace increasing_function_range_l425_425883

noncomputable def condition1 (a : ℝ) : Prop := ∀ x ∈ Ioc 0 1, (2 * x + (1/2) * a) ≥ 0
noncomputable def condition2 (a : ℝ) : Prop := ∀ x > 1, (a^x * log a) ≥ 0
noncomputable def condition3 (a : ℝ) : Prop := 1 + (1/2) * a - 2 ≤ 0

theorem increasing_function_range (a : ℝ) :
  condition1 a ∧ condition2 a ∧ condition3 a → 1 < a ∧ a ≤ 2 :=
by
  sorry

end increasing_function_range_l425_425883


namespace angle_BAC_is_45_degrees_l425_425631

open EuclideanGeometry

theorem angle_BAC_is_45_degrees
  {A B C B1 C1 O : Point}
  (h1 : IsAcuteAngle (triangle A B C))
  (h2 : AltitudesIntersectAtCircumcircle B C B1 C1 (triangle A B C))
  (h3 : B1C1PassesThroughCircumcenter B1 C1 (triangle A B C) O) : 
  Angle A B C = 45 :=
sorry

end angle_BAC_is_45_degrees_l425_425631


namespace solve_for_y_l425_425623

theorem solve_for_y (y : ℝ) (h : y^2 + 6 * y + 8 = -(y + 4) * (y + 6)) : y = -4 :=
by {
  sorry
}

end solve_for_y_l425_425623


namespace number_of_chickens_l425_425748

theorem number_of_chickens (c b : ℕ) (h1 : c + b = 9) (h2 : 2 * c + 4 * b = 26) : c = 5 :=
by
  sorry

end number_of_chickens_l425_425748


namespace external_angle_bisector_condition_l425_425101

-- Given: A triangle ABC with points A, B, and C
variables (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variable [decidable_eq A] [decidable_eq B] [decidable_eq C] [decidable_eq D]

-- Definition of the perimeter of a triangle
def perimeter (A B C : Type) [metric_space A] [metric_space B] [metric_space C] : ℝ :=
  dist A B + dist B C + dist C A

-- Definition of the external angle bisector
def external_angle_bisector (A B C : Type) [metric_space A] [metric_space B] [metric_space C] : Type :=
  { D : Type // 
      ∀ D ≠ C, (dist A D + dist B D) > (dist A C + dist B C)
  }

-- Theorem statement
theorem external_angle_bisector_condition
  (A B C : Type) [metric_space A] [metric_space B] [metric_space C] :
  ∃ (line : Type), 
    line = external_angle_bisector A B C ∧
    ∀ D ≠ C, perimeter A B D > perimeter A B C := 
sorry

end external_angle_bisector_condition_l425_425101


namespace semicircle_inscribed_quadrilateral_l425_425185

theorem semicircle_inscribed_quadrilateral 
  (x a b c : ℝ) 
  (h_amnb_inscribed : ∃ (O : ℝ), ∀ (P ∈ {AMNB : ℝ | P ∈ {AM, MN, NB}}), (P * P = x * x - O * O)) :
  x^3 - (a^2 + b^2 + c^2) * x - 2 * a * b * c = 0 := 
sorry

end semicircle_inscribed_quadrilateral_l425_425185


namespace find_x_value_l425_425934

theorem find_x_value (x y z : ℕ) (hxyz : x ≥ y ∧ y ≥ z) 
  (h_eq1 : x^2 - y^2 - z^2 + x * y = 3007)
  (h_eq2 : x^2 + 4 * y^2 + 4 * z^2 - 4 * x * y - 3 * x * z - 3 * y * z = -2901) :
  x = 59 :=
sorry

end find_x_value_l425_425934


namespace find_common_difference_l425_425907
-- Lean 4 statement

noncomputable def arith_seq_common_difference (a : ℕ → ℝ) : ℝ :=
  sorry

axiom arith_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a n = a 1 + (n - 1) * d

theorem find_common_difference (a : ℕ → ℝ) (d : ℝ)
  (h1: a 4 = 6)
  (h2: a 3 + a 5 = a 10)
  (h_seq: arith_sequence a d) :
  d = 1 :=
  sorry

end find_common_difference_l425_425907


namespace abs_neg_three_eq_three_l425_425251

theorem abs_neg_three_eq_three : abs (-3) = 3 := 
by 
  sorry

end abs_neg_three_eq_three_l425_425251


namespace equation_of_tangent_line_at_point_l425_425998

noncomputable def f (x : ℝ) : ℝ := x^2 + 1 / x

def point_of_tangency : ℝ × ℝ := (1, 2)

theorem equation_of_tangent_line_at_point : 
  let m := (derivative f 1)
  ∃ (x y : ℝ), y - 2 = m * (x - 1) ∧ x - y - 1 = 0 :=
by 
  have : m = 1 := by sorry
  use 1, 2
  split
  { rw [m, this] ; sorry }
  { sorry }

end equation_of_tangent_line_at_point_l425_425998


namespace cleaning_time_per_week_l425_425235

theorem cleaning_time_per_week : 
  let richard := 22 in
  let cory := richard + 3 in
  let blake := cory - 4 in
  let evie := richard + blake in
  let total := (richard + cory + blake + evie) * 2 in
  total = 222 := by
  -- All calculations and proofs will be skipped
  sorry   -- Placeholder for proof

end cleaning_time_per_week_l425_425235


namespace perfect_squares_three_digit_divisible_by_4_count_l425_425136

theorem perfect_squares_three_digit_divisible_by_4_count : 
  ∃ (n : ℕ), (n = 11) ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ (∃ k, m = k^2) ∧ (m % 4 = 0) ↔ m ∈ {100, 144, 196, 256, 324, 400, 484, 576, 676, 784, 900}) :=
by
  existsi 11
  split
  · reflexivity
  · intro m
    split
    · intro h
      rcases h with ⟨⟨_, _, _, _⟩, _⟩
      -- details have been omitted
      sorry
    · intro h
      -- details have been omitted
      sorry

end perfect_squares_three_digit_divisible_by_4_count_l425_425136


namespace rectangle_count_horizontal_vertical_l425_425870

theorem rectangle_count_horizontal_vertical :
  ∀ (h_strips : ℕ) (v_strips : ℕ) (intersection : ℕ), 
  h_strips = 15 → v_strips = 10 → intersection = 1 → 
  (h_strips + v_strips - intersection = 24) :=
by
  intros h_strips v_strips intersection h_strips_def v_strips_def intersection_def
  rw [h_strips_def, v_strips_def, intersection_def]
  sorry

end rectangle_count_horizontal_vertical_l425_425870


namespace solve_problems_l425_425616

variable (initial_problems : ℕ) 
variable (additional_problems : ℕ)

theorem solve_problems
  (h1 : initial_problems = 12) 
  (h2 : additional_problems = 7) : 
  initial_problems + additional_problems = 19 := 
by 
  sorry

end solve_problems_l425_425616


namespace distance_traveled_l425_425691

variable (V1 : ℝ) (C1 : ℝ) (V2 : ℝ) (C2 : ℝ) (d2 : ℝ)

-- Given conditions
axiom h1 : V1 = 50 -- 50 litres of diesel
axiom h2 : C1 = 800 -- 800 cc engine
axiom h3 : V2 = 100 -- 100 litres of diesel
axiom h4 : C2 = 1200 -- 1200 cc engine
axiom h5 : d2 = 800 -- 800 km traveled

-- Variable for distance using 50 litres of diesel and 800 cc engine
def D : ℝ := (V1 * d2 * C2) / (V2 * C1)

-- The statement to prove
theorem distance_traveled : D = 6 := by
  sorry

end distance_traveled_l425_425691


namespace strangely_powerful_count_l425_425401

def oddly_powerful (n : ℕ) : Prop :=
  ∃ a b : ℕ, b > 1 ∧ b % 2 = 1 ∧ a^b = n

theorem strangely_powerful_count :
  { n : ℕ | oddly_powerful n ∧ n < 5000 }.toFinset.card = 23 :=
by
  sorry

end strangely_powerful_count_l425_425401


namespace find_f1_find_f8_inequality_l425_425573

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom f_pos : ∀ x : ℝ, 0 < x → 0 < f x
axiom f_increasing : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f x < f y
axiom f_multiplicative : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x * f y
axiom f_of_2 : f 2 = 4

-- Statements to prove
theorem find_f1 : f 1 = 1 := sorry
theorem find_f8 : f 8 = 64 := sorry
theorem inequality : ∀ x : ℝ, 3 < x → x ≤ 7 / 2 → 16 * f (1 / (x - 3)) ≥ f (2 * x + 1) := sorry

end find_f1_find_f8_inequality_l425_425573


namespace avg_diff_l425_425519

theorem avg_diff (a x c : ℝ) (h1 : (a + x) / 2 = 40) (h2 : (x + c) / 2 = 60) :
  c - a = 40 :=
by
  sorry

end avg_diff_l425_425519


namespace problem_part_I_problem_part_II_l425_425468

-- Define the circle O with equation x^2 + y^2 = 4
def circle_O := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }

-- Define the line l with equation x = 4
def line_l := { p : ℝ × ℝ | p.1 = 4 }

-- Define point A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define point P on line l
def P (t : ℝ) : ℝ × ℝ := (4, t)

-- Define the fixed point Q
def Q : ℝ × ℝ := (1, 0)

-- (I) Proposition to prove coordinates of P and length of minor arc
theorem problem_part_I (t : ℝ) (h_tangent_length : ∀ t, ∃ C D, line_l (P t) ∧ is_tangent C D (P t)  (2 * Real.sqrt 3) ∧ ∃! z ∈ circle_O, P t z  ∧ angle (P t) O C = angle_subtended O P C) :
  (exists t : ℝ, P t = (4, 0)) ∧ (measurement {p : ℝ × ℝ | is_minor_arc P p(P t)} = (4 * Real.pi) / 3) :=
sorry

-- (II) Proposition to prove line MN passes through fixed point Q
theorem problem_part_II (t : ℝ) (M N : ℝ × ℝ) (h_intersects : ∀ t, intersects_in_circle (P t) A B M N circle_O) :
  passes_through_fixed_point (M, N) Q :=
sorry

-- Define relevant predicates and helper functions
def is_tangent (C D P) := -- implementation details
sorry

def intersects_in_circle := -- implementation details
sorry

def passes_through_fixed_point := -- implementation details
sorry

def measurement := -- implementation details
sorry

def angle_subtended := -- implementation details
sorry

end problem_part_I_problem_part_II_l425_425468


namespace sector_area_l425_425469

theorem sector_area (C : ℝ) (θ : ℝ) (r : ℝ) (S : ℝ)
  (hC : C = (8 * Real.pi / 9) + 4)
  (hθ : θ = (80 * Real.pi / 180))
  (hne : θ * r / 2 + r = C) :
  S = (1 / 2) * θ * r^2 → S = 8 * Real.pi / 9 :=
by
  sorry

end sector_area_l425_425469


namespace vincent_spent_224_l425_425661

-- Defining the given conditions as constants
def num_books_animal : ℕ := 10
def num_books_outer_space : ℕ := 1
def num_books_trains : ℕ := 3
def cost_per_book : ℕ := 16

-- Summarizing the total number of books
def total_books : ℕ := num_books_animal + num_books_outer_space + num_books_trains
-- Calculating the total cost
def total_cost : ℕ := total_books * cost_per_book

-- Lean statement to prove that Vincent spent $224
theorem vincent_spent_224 : total_cost = 224 := by
  sorry

end vincent_spent_224_l425_425661


namespace part1_part2_l425_425858

noncomputable def f (a b : ℝ) (x : ℝ) := (a * x) / (Real.exp x + 1) + b * (Real.exp(-x))

def M (a b : ℝ) := (f a b 0 = 1) ∧ (has_deriv_at (λ x, f a b x) (-1/2) 0)

theorem part1 :
  ∃ (a b : ℝ), (f a b 0 = 1) ∧ ( has_deriv_at (λ x, f a b x) (-1/2) 0 ) → a = 1 ∧ b = 1 := 
by
  sorry

theorem part2 (a b : ℝ) (H : ∀ x : ℝ, x ≠ 0 → f a b x > x / (Real.exp x - 1) + k * (Real.exp (-x))) :
  k ≤ 0 :=
by
  sorry

end part1_part2_l425_425858


namespace cards_distribution_l425_425146

theorem cards_distribution (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (extra_cards : ℕ) (people_with_extra_cards : ℕ) (people_with_fewer_cards : ℕ) :
  total_cards = 100 →
  total_people = 15 →
  total_cards / total_people = cards_per_person →
  total_cards % total_people = extra_cards →
  people_with_extra_cards = extra_cards →
  people_with_fewer_cards = total_people - people_with_extra_cards →
  people_with_fewer_cards = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end cards_distribution_l425_425146


namespace distance_AD_l425_425604

-- Define points A, B, C, D in a 2D plane
structure Point := (x : ℝ) (y : ℝ)

-- Define distances and angles given in the problem
def A : Point := ⟨0, 0⟩
def B : Point := ⟨b_x, 0⟩
def C : Point := ⟨b_x, c_y⟩
def D : Point := ⟨b_x, c_y + 10⟩

-- Given conditions and correct answer definition
def distance (p1 p2 : Point) : ℝ := real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

theorem distance_AD : distance A D = real.sqrt (562.5 + 300 * real.sqrt 3) :=
by
  -- Using the given and derived distances and angles
  let d_AC := 15
  let angle_BAC := 30
  have h1 : b_x = d_AC * real.cos (real.pi / 6), from sorry
  have h2 : c_y = d_AC * real.sin (real.pi / 6), from sorry
  -- Therefore, use the distance formula to compute AD
  sorry

end distance_AD_l425_425604


namespace three_digit_perfect_squares_divisible_by_4_count_l425_425134

theorem three_digit_perfect_squares_divisible_by_4_count : 
  (finset.filter (λ n, (100 ≤ n ∧ n ≤ 999) ∧ (∃ k, n = k * k) ∧ (n % 4 = 0)) (finset.range 1000)).card = 10 :=
by
  sorry

end three_digit_perfect_squares_divisible_by_4_count_l425_425134


namespace three_digit_perfect_squares_divisible_by_4_count_l425_425133

theorem three_digit_perfect_squares_divisible_by_4_count : 
  (finset.filter (λ n, (100 ≤ n ∧ n ≤ 999) ∧ (∃ k, n = k * k) ∧ (n % 4 = 0)) (finset.range 1000)).card = 10 :=
by
  sorry

end three_digit_perfect_squares_divisible_by_4_count_l425_425133


namespace lcm_problem_l425_425879

-- Define LCM function
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

-- The conditions given in the problem
theorem lcm_problem (x : ℕ) : (lcm (lcm 12 16) (lcm x 24)) = 144 → x = 3 :=
by
  sorry

end lcm_problem_l425_425879


namespace first_prize_winner_is_B_l425_425779

-- Definitions of the predictions
def Jia_pred (winner : String) : Prop := winner = "C" ∨ winner = "D"
def Yi_pred (winner : String) : Prop := winner = "B"
def Bing_pred (winner : String) : Prop := winner ≠ "A" ∧ winner ≠ "D"
def Ding_pred (winner : String) : Prop := winner = "C"

-- Given condition
def exactly_two_correct (winner : String) : Prop :=
  (if Jia_pred winner then 1 else 0) +
  (if Yi_pred winner then 1 else 0) +
  (if Bing_pred winner then 1 else 0) +
  (if Ding_pred winner then 1 else 0) = 2

-- Problem statement to prove
theorem first_prize_winner_is_B : ∃! winner, exactly_two_correct winner ∧ winner = "B" := by
  sorry

end first_prize_winner_is_B_l425_425779


namespace jake_balloons_proof_l425_425730

def jake_initial_balloons (j : ℕ) : Prop :=
  let allan_initial : ℕ := 3
  let allan_extra : ℕ := 2
  let total_balloons : ℕ := 10
  let allan_total : ℕ := allan_initial + allan_extra
  allan_total + j = total_balloons

theorem jake_balloons_proof : jake_initial_balloons 5 :=
by
  let allan_initial : ℕ := 3
  let allan_extra : ℕ := 2
  let total_balloons : ℕ := 10
  let allan_total : ℕ := allan_initial + allan_extra
  rw [←nat.add_sub_assoc, nat.sub_self, nat.add_zero]
  sorry

end jake_balloons_proof_l425_425730


namespace perfect_square_trinomial_m_l425_425876

-- Defining the polynomial and the concept of it being a perfect square
def is_perfect_square_trinomial (p : Polynomial ℝ) : Prop :=
∃ a : ℝ, p = (X + C a)^2 

theorem perfect_square_trinomial_m (m : ℝ) :
  is_perfect_square_trinomial (X^2 + C (m+1) * X + C 16) ↔ m = 7 ∨ m = -9 :=
by
  sorry

end perfect_square_trinomial_m_l425_425876


namespace max_handshakes_in_club_l425_425737

theorem max_handshakes_in_club (gentlemen : Finset Person) (H : gentlemen.card = 20) 
(ap : ∀ (A B C : Person), A ∈ gentlemen → B ∈ gentlemen → C ∈ gentlemen → 
acquainted A B → acquainted B C → acquainted C A → false) : 
∃ (max_handshakes : ℕ), max_handshakes = 100 := 
sorry

/--
The problem: 
The maximum number of handshakes in a club of 20 gentlemen where no three gentlemen are mutually acquainted is 100.
--/
noncomputable def max_handshakes := 100 -- Solution

end max_handshakes_in_club_l425_425737


namespace wheel_radius_l425_425281

theorem wheel_radius (π := Real.pi) (distance_covered := 351.99999999999994) (revolutions := 250) :
  let C := distance_covered / (revolutions : Real)
  let r := C / (2 * π)
  r ≈ 0.224 := 
by 
  sorry

end wheel_radius_l425_425281


namespace square_digit_substitution_l425_425547

theorem square_digit_substitution (a b c d e : ℕ)
  (h1 : abcd^2 =⟨a, b, c, d⟩.lhs)
  (h2 : digit a ∧ digit b ∧ digit c ∧ digit d ∧ digit e )
  (h3 : distinct {a, b, c, d, e}) : 
  abcd = 3957 ∨ abcd = 3967 := 
sorry

end square_digit_substitution_l425_425547


namespace sphere_volume_increase_l425_425886

theorem sphere_volume_increase 
  (r : ℝ) 
  (S : ℝ := 4 * Real.pi * r^2) 
  (V : ℝ := (4/3) * Real.pi * r^3)
  (k : ℝ := 2) 
  (h : 4 * S = 4 * Real.pi * (k * r)^2) : 
  ((4/3) * Real.pi * (2 * r)^3) = 8 * V := 
by
  sorry

end sphere_volume_increase_l425_425886


namespace mr_willson_friday_work_time_l425_425954

theorem mr_willson_friday_work_time :
  let monday := 3 / 4
  let tuesday := 1 / 2
  let wednesday := 2 / 3
  let thursday := 5 / 6
  let total_work := 4
  let time_monday_to_thursday := monday + tuesday + wednesday + thursday
  let time_friday := total_work - time_monday_to_thursday
  time_friday * 60 = 75 :=
by
  sorry

end mr_willson_friday_work_time_l425_425954


namespace brownies_left_l425_425387

theorem brownies_left (initial : ℕ) (admin_fraction : ratio) (carl_fraction : ratio) (extra_to_simon : ℕ) : 
    (initial = 20) → 
    (admin_fraction = 1 / 2) → 
    (carl_fraction = 1 / 2) → 
    (extra_to_simon = 2) → 
    let after_admin := initial * (1 - admin_fraction) in
    let after_carl := after_admin * (1 - carl_fraction) in
    (after_carl - extra_to_simon = 3) :=
begin
    intros h1 h2 h3 h4,
    rw [h1, h2, h3, h4],
    let after_admin := 20 * (1 - 1 / 2),
    let after_carl := after_admin * (1 - 1 / 2),
    have h_after_admin : after_admin = 10, by norm_num,
    rw h_after_admin,
    have h_after_carl : after_carl = 5, by norm_num [h_after_admin],
    rw h_after_carl,
    norm_num,
end

end brownies_left_l425_425387


namespace value_of_a_plus_b_l425_425646

noncomputable def verify_solution_set (a b : ℝ) : Prop :=
  ∀ x : ℝ, (ax^2 + bx + 1 > 0) ↔ (-1 < x ∧ x < 1/3)

theorem value_of_a_plus_b (a b : ℝ) (h : verify_solution_set a b) : a + b = -5 := 
by
  sorry

end value_of_a_plus_b_l425_425646


namespace min_value_of_expression_l425_425049

open Real

theorem min_value_of_expression (a : ℝ) (b : ℝ) (hb : 0 < b) : 
  ∃ m : ℝ, m = 2 ∧ (∀ a b, 0 < b → (exp a - log b) ^ 2 + (a - b) ^ 2 ≥ m) := 
begin
  use 2,
  split,
  { refl },
  {
    intro a,
    intro b,
    intro hb,
    sorry
  }
end

end min_value_of_expression_l425_425049


namespace perfect_squares_three_digit_divisible_by_4_count_l425_425135

theorem perfect_squares_three_digit_divisible_by_4_count : 
  ∃ (n : ℕ), (n = 11) ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ (∃ k, m = k^2) ∧ (m % 4 = 0) ↔ m ∈ {100, 144, 196, 256, 324, 400, 484, 576, 676, 784, 900}) :=
by
  existsi 11
  split
  · reflexivity
  · intro m
    split
    · intro h
      rcases h with ⟨⟨_, _, _, _⟩, _⟩
      -- details have been omitted
      sorry
    · intro h
      -- details have been omitted
      sorry

end perfect_squares_three_digit_divisible_by_4_count_l425_425135


namespace total_distance_traveled_l425_425588

theorem total_distance_traveled :
  let time1 := 3  -- hours
  let speed1 := 70  -- km/h
  let time2 := 4  -- hours
  let speed2 := 80  -- km/h
  let time3 := 3  -- hours
  let speed3 := 65  -- km/h
  let time4 := 2  -- hours
  let speed4 := 90  -- km/h
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4
  distance1 + distance2 + distance3 + distance4 = 905 :=
by
  sorry

end total_distance_traveled_l425_425588


namespace range_of_a_l425_425944

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * real.sin (3 * x) + real.cos (3 * x)

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |f x| ≤ a) → a ≥ 2 := by
  sorry

end range_of_a_l425_425944


namespace steaks_from_15_pounds_of_beef_l425_425209

-- Definitions for conditions
def pounds_to_ounces (pounds : ℕ) : ℕ := pounds * 16

def steaks_count (total_ounces : ℕ) (ounces_per_steak : ℕ) : ℕ := total_ounces / ounces_per_steak

-- Translate the problem to Lean statement
theorem steaks_from_15_pounds_of_beef : 
  steaks_count (pounds_to_ounces 15) 12 = 20 :=
by
  sorry

end steaks_from_15_pounds_of_beef_l425_425209


namespace eccentricity_of_ellipse_l425_425836

theorem eccentricity_of_ellipse (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a = 5 * c) : 
  eccentricity_of (ellipse a b) = 1 / 5 :=
by sorry

end eccentricity_of_ellipse_l425_425836


namespace geometric_sequence_log_sum_l425_425826

open Real

noncomputable def log_sum (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, 1 ≤ n → n ≤ 10 → 0 < a n) ∧
  (a 5 * a 6 = 2) ∧
  (∀ m n : ℕ, 1 ≤ m → m + n = 11 → a m * a n = 2) →
  (log 2 (a 1) + log 2 (a 2) + log 2 (a 3) + log 2 (a 4) + log 2 (a 5) +
   log 2 (a 6) + log 2 (a 7) + log 2 (a 8) + log 2 (a 9) + log 2 (a 10) = 5)

theorem geometric_sequence_log_sum (a : ℕ → ℝ)
  (h1 : ∀ n : ℕ, 1 ≤ n → n ≤ 10 → 0 < a n)
  (h2 : a 5 * a 6 = 2)
  (h3 : ∀ m n : ℕ, 1 ≤ m → m + n = 11 → a m * a n = 2) :
  log 2 (a 1) + log 2 (a 2) + log 2 (a 3) + log 2 (a 4) + log 2 (a 5) +
  log 2 (a 6) + log 2 (a 7) + log 2 (a 8) + log 2 (a 9) + log 2 (a 10) = 5 := 
sorry

end geometric_sequence_log_sum_l425_425826


namespace greatest_three_digit_number_l425_425664

theorem greatest_three_digit_number
  (n : ℕ) (h_3digit : 100 ≤ n ∧ n < 1000) (h_mod7 : n % 7 = 2) (h_mod4 : n % 4 = 1) :
  n = 989 :=
sorry

end greatest_three_digit_number_l425_425664


namespace pencil_total_l425_425297

theorem pencil_total (initial_pencils : ℕ) (added_pencils : ℕ) (total_pencils : ℕ) 
  (h_init : initial_pencils = 27) (h_add : added_pencils = 45) : 
  total_pencils = initial_pencils + added_pencils := by
    rw [h_init, h_add]
    rfl

end pencil_total_l425_425297


namespace abs_neg_three_eq_three_l425_425250

theorem abs_neg_three_eq_three : abs (-3) = 3 := 
by 
  sorry

end abs_neg_three_eq_three_l425_425250


namespace students_sampling_problem_l425_425719

theorem students_sampling_problem (total_students grade10_students grade11_students grade12_students : ℕ) 
  (h_total : total_students = 3500) 
  (h_grade12 : grade12_students = 2 * grade10_students) 
  (h_grade11 : grade11_students = grade10_students + 300) 
  (sampling_ratio : ℚ) 
  (h_sampling_ratio : sampling_ratio = 1 / 100) :
  let grade10_students_not_sampled := 3500 - (2 * grade10_students + (grade10_students + 300)) 
  in grade10_students_not_sampled * sampling_ratio = 8 :=
sorry

end students_sampling_problem_l425_425719


namespace cistern_filling_time_l425_425312

open Real

theorem cistern_filling_time :
  let rate1 := 1 / 10
  let rate2 := 1 / 12
  let rate3 := -1 / 25
  let rate4 := 1 / 15
  let rate5 := -1 / 30
  let combined_rate := rate1 + rate2 + rate4 + rate3 + rate5
  (300 / combined_rate) = (300 / 53) := by
  let rate1 := 1 / 10
  let rate2 := 1 / 12
  let rate3 := -1 / 25
  let rate4 := 1 / 15
  let rate5 := -1 / 30
  let combined_rate := rate1 + rate2 + rate4 + rate3 + rate5
  sorry

end cistern_filling_time_l425_425312


namespace min_norm_value_l425_425861

noncomputable def vector_min_norm (λ : ℝ) : ℝ := 
  let a := (real.cos (40 * real.pi / 180), real.sin (40 * real.pi / 180))
  let b := (real.sin (20 * real.pi / 180), real.cos (20 * real.pi / 180))
  let u := (sqrt 3 * a.1 + λ * b.1, sqrt 3 * a.2 + λ * b.2)
  sqrt (u.1^2 + u.2^2)

theorem min_norm_value : ∃ λ : ℝ, vector_min_norm λ = (sqrt 3) / 2 :=
sorry

end min_norm_value_l425_425861


namespace solve_for_y_l425_425979

theorem solve_for_y (y : ℝ) (h : 5^(2 * y) = real.sqrt 125) : y = 3 / 4 :=
sorry

end solve_for_y_l425_425979


namespace kelsey_videos_watched_l425_425307

-- Definitions of conditions
def total_videos : ℕ := 411
def ekon_less : ℕ := 17
def kelsey_more : ℕ := 43

-- Variables representing videos watched by Uma, Ekon, and Kelsey
variables (U E K : ℕ)
hypothesis total_watched : U + E + K = total_videos
hypothesis ekon_watch : E = U - ekon_less
hypothesis kelsey_watch : K = E + kelsey_more

-- The statement to be proved
theorem kelsey_videos_watched : K = 160 :=
by sorry

end kelsey_videos_watched_l425_425307


namespace prime_divisors_unique_l425_425609

theorem prime_divisors_unique
  (n : ℕ) :
  ∃ (p : ℕ → ℕ), (∀ k, (1 ≤ k ∧ k ≤ n) → (nat.prime (p k))) ∧
    ∀ k₁ k₂, (1 ≤ k₁ ∧ k₁ ≤ n) ∧ (1 ≤ k₂ ∧ k₂ ≤ n) ∧ (k₁ ≠ k₂) → (p k₁ ∣ (n! + k₁)) ∧¬(p k₁ ∣ (n! + k₂)) :=
sorry

end prime_divisors_unique_l425_425609


namespace am_gm_inequality_l425_425579

theorem am_gm_inequality (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end am_gm_inequality_l425_425579


namespace triangle_angles_l425_425917

theorem triangle_angles (h_b h_c b c : ℝ) (h_b_geq_b : h_b ≥ b) (h_c_geq_c : h_c ≥ c)
  (area1 : 1/2 * b * h_b = 1/2 * b * c * real.sin (real.pi / 2))
  (area2 : 1/2 * c * h_c = 1/2 * c * b * real.sin (real.pi / 2)) :
  ∃ α β γ : ℝ, α = real.pi / 2 ∧ β = real.pi / 4 ∧ γ = real.pi / 4 :=
by
  sorry

end triangle_angles_l425_425917


namespace cost_of_pencil_and_pens_l425_425258

variable (p q : ℝ)

def equation1 := 3 * p + 4 * q = 3.20
def equation2 := 2 * p + 3 * q = 2.50

theorem cost_of_pencil_and_pens (h1 : equation1 p q) (h2 : equation2 p q) : p + 2 * q = 1.80 := 
by 
  sorry

end cost_of_pencil_and_pens_l425_425258


namespace num_ordered_pairs_500_l425_425433

def no_zero_digit (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0

theorem num_ordered_pairs_500 : 
  (∃! (count : ℕ), count = 414 ∧ 
     (∃ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b ∧ a + b = 500 ∧ no_zero_digit a ∧ no_zero_digit b)) :=
by
  sorry

end num_ordered_pairs_500_l425_425433


namespace wholesale_cost_l425_425677

theorem wholesale_cost
  (selling_price : ℝ) (profit_percentage : ℝ) (wholesale_cost : ℝ)
  (h1 : selling_price = 28)
  (h2 : profit_percentage = 0.16) :
  wholesale_cost ≈ 24.14 :=
by
  sorry

end wholesale_cost_l425_425677


namespace phi_not_necessarily_rectangle_l425_425962

-- Define the 10 colors and the 10-cell polygon Phi
inductive Color : Type
| color0 | color1 | color2 | color3 | color4 | color5 | color6 | color7 | color8 | color9

structure Grid :=
(width : ℕ)
(height : ℕ)
(coloring : ℕ → ℕ → Color)
(h_color_range : ∀ x y, x < width → y < height → (coloring x y) ∈ {Color.color0, Color.color1, Color.color2, Color.color3, Color.color4, Color.color5, Color.color6, Color.color7, Color.color8, Color.color9})

-- Define the polygon Phi
structure Polygon :=
(cell_positions : list (ℕ × ℕ))
(h_distinct_colors : ∀ grid : Grid, ∃ placement : ℕ × ℕ, let covered_colors := (cell_positions.map (λ pos, grid.coloring (pos.1 + placement.1) (pos.2 + placement.2))) in (covered_colors.nodup))

-- Lean statement: Must Φ be a rectangle?
theorem phi_not_necessarily_rectangle (Φ : Polygon) (grid : Grid) : ∃ (non_rectangle : Polygon), non_rectangle ≠ Φ ∧ (∀ grid : Grid, ∃ placement : ℕ × ℕ, let covered_colors := (non_rectangle.cell_positions.map (λ pos, grid.coloring (pos.1 + placement.1) (pos.2 + placement.2))) in (covered_colors.nodup)) := sorry

end phi_not_necessarily_rectangle_l425_425962


namespace vectors_sum_zero_l425_425851

open Vector3

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def collinear (v₁ v₂ : V) : Prop := ∃ λ : ℝ, v₁ = λ • v₂

theorem vectors_sum_zero (a b c : V) 
    (h₁ : ¬ collinear a b)
    (h₂ : ¬ collinear b c)
    (h₃ : ¬ collinear a c)
    (h4 : collinear (a + b) c)
    (h5 : collinear (b + c) a) :
    a + b + c = 0 :=
sorry

end vectors_sum_zero_l425_425851


namespace LitteringCitationsIsNine_l425_425777

def LitteringCitationsEqualOffLeashDogs (L : ℕ) (TotalCitations : ℕ) (ParkWardenCitations : ℕ) (UnauthorizedCamping : ℕ) (ParkingFinesCoefficient : ℕ) : Prop :=
  let OffLeashDogs := L
  let SmokingProhibitedAreas := L - 5
  let CitationsSum :=
    L + OffLeashDogs + SmokingProhibitedAreas
  let ParkingFines :=
    ParkingFinesCoefficient * CitationsSum
  let TotalCitationsCalculated :=
    L + OffLeashDogs + SmokingProhibitedAreas + ParkingFines + UnauthorizedCamping
  TotalCitationsCalculated = TotalCitations

theorem LitteringCitationsIsNine (L : ℕ) (TotalCitations : ℕ) (UnauthorizedCamping : ℕ) (ParkingFinesCoefficient : ℕ) :
  LitteringCitationsEqualOffLeashDogs L TotalCitations UnauthorizedCamping ParkingFinesCoefficient → L = 9 := 
by
  intros h
  sorry

end LitteringCitationsIsNine_l425_425777


namespace sin_APC_l425_425544

theorem sin_APC (P A B C : Point) (h : P = Point.zero) (h1 : ∠ APB = 180 - ∠ APC) (h2 : sin (∠ APB) = 3 / 5) :
  sin (∠ APC) = 3 / 5 :=
sorry

end sin_APC_l425_425544


namespace area_of_polygon_formed_by_squares_l425_425294

theorem area_of_polygon_formed_by_squares 
  (a b : ℝ) (h₀ : a = 3) (h₁ : b = 9) : 
  (a * a) + (b * b) + ((a^2 + b^2)^(1/2))^2 = 193.5 := by
  have h : (3^2 : ℝ) + (9^2 : ℝ) + ((3^2 + 9^2)^(1/2))^2 = 193.5 := by
    sorry 
  exact h

end area_of_polygon_formed_by_squares_l425_425294


namespace probability_two_heads_and_die_three_l425_425322

-- Define the events
def coin_flip_outcomes := {flip₁ flip₂ flip₃ | flip₁ ∈ {0, 1} ∧ flip₂ ∈ {0, 1} ∧ flip₃ ∈ {0, 1}}
def exactly_two_heads (flips : list bool) : Prop := flips.count (· = true) = 2
def die_roll_outcomes := { 1, 2, 3, 4, 5, 6 }

-- Define probability space for coin flips and die rolls combined
def combined_outcomes := {c | c.flip₁ ∈ coin_flip_outcomes ∧ c.die_roll ∈ die_roll_outcomes}

theorem probability_two_heads_and_die_three :
  (finset.filter (λ o, exactly_two_heads o.flip₁ ∧ o.die_roll = 3) combined_outcomes).card / combined_outcomes.card = 1 / 16 := by
  sorry

end probability_two_heads_and_die_three_l425_425322


namespace determine_d_and_k_l425_425477

def A : Matrix (Fin 2) (Fin 2) ℚ := ![![3, 4], ![7, d]]

def A_inv (k : ℚ) : Matrix (Fin 2) (Fin 2) ℚ := k • ![![d, 4], ![7, 3]]

theorem determine_d_and_k :
  (∃ d k : ℚ, 
    ∃ h : A⁻¹ = A_inv k, 
    d = 0 ∧ k = 1 / 28) := by
  sorry

end determine_d_and_k_l425_425477


namespace count_three_digit_perfect_squares_divisible_by_4_l425_425119

theorem count_three_digit_perfect_squares_divisible_by_4 : ∃ n, n = 11 ∧
  (∀ k, 100 ≤ k ^ 2 ∧ k ^ 2 ≤ 999 → k ^ 2 % 4 = 0 → (k % 2 = 0 ∧ 10 ≤ k ≤ 31)) :=
sorry

end count_three_digit_perfect_squares_divisible_by_4_l425_425119


namespace ferris_wheel_seats_l425_425692

theorem ferris_wheel_seats (total_people seats_capacity : ℕ) (h1 : total_people = 8) (h2 : seats_capacity = 3) : 
  Nat.ceil ((total_people : ℚ) / (seats_capacity : ℚ)) = 3 := 
by
  sorry

end ferris_wheel_seats_l425_425692


namespace who_had_second_value_card_in_first_game_l425_425652

variable (A B C : ℕ)
variable (x y z : ℕ)
variable (points_A points_B points_C : ℕ)

-- Provided conditions
variable (h1 : x < y ∧ y < z)
variable (h2 : points_A = 20)
variable (h3 : points_B = 10)
variable (h4 : points_C = 9)
variable (number_of_games : ℕ)
variable (h5 : number_of_games = 3)
variable (h6 : A + B + C = 39)  -- This corresponds to points_A + points_B + points_C = 39.
variable (h7 : ∃ x y z, x + y + z = 13 ∧ x < y ∧ y < z)
variable (h8 : B = z)

-- Question/Proof to establish
theorem who_had_second_value_card_in_first_game :
  ∃ p : ℕ, p = C :=
sorry

end who_had_second_value_card_in_first_game_l425_425652


namespace correct_equation_l425_425717

-- Define conditions as variables in Lean
def cost_price (x : ℝ) : Prop := x > 0
def markup_percentage : ℝ := 0.40
def discount_percentage : ℝ := 0.80
def selling_price : ℝ := 240

-- Define the theorem
theorem correct_equation (x : ℝ) (hx : cost_price x) :
  x * (1 + markup_percentage) * discount_percentage = selling_price :=
by
  sorry

end correct_equation_l425_425717


namespace min_value_expression_l425_425204

noncomputable def problem_statement : Prop :=
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → xy^2z = 72 → (x^2 + 4 * x * y + 4 * y^2 + 2 * z^2) ≥ 120

theorem min_value_expression : problem_statement := by sorry

end min_value_expression_l425_425204


namespace total_distance_travelled_l425_425495

theorem total_distance_travelled (distance_to_market : ℕ) (travel_time_minutes : ℕ) (speed_mph : ℕ) 
  (h1 : distance_to_market = 30) 
  (h2 : travel_time_minutes = 30) 
  (h3 : speed_mph = 20) : 
  (distance_to_market + ((travel_time_minutes / 60) * speed_mph) = 40) :=
by
  sorry

end total_distance_travelled_l425_425495


namespace inequality_proof_l425_425823

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b + b * c + c * a ≥ 1) :
    (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ (Real.sqrt 3) / (a * b * c) :=
by
  sorry

end inequality_proof_l425_425823


namespace sum_of_real_values_l425_425293

theorem sum_of_real_values (x : ℝ) (h : |x + 3| = 3 * |x - 4|) : x = 7.5 ∨ x = 2.25 → ∑ v in ({7.5, 2.25} : Finset ℝ), v = 9.75 := by
  sorry

end sum_of_real_values_l425_425293


namespace initial_tickets_l425_425744

theorem initial_tickets (X : ℕ) (h : (X - 22) + 15 = 18) : X = 25 :=
by
  sorry

end initial_tickets_l425_425744


namespace triangle_is_right_triangle_l425_425552

theorem triangle_is_right_triangle
  (A B C a b c : ℝ)
  (∀ t : ℝ, 0 < t < π → ∃ u : ℝ, sin u = t)
  (triangle_angles_sum : A + B + C = π)
  (eqn : a * cos B + b * cos A = c * sin A)
  (sine_pos : 0 < sin A) :
  A = π / 2 := 
sorry

end triangle_is_right_triangle_l425_425552


namespace min_cubes_required_l425_425361

theorem min_cubes_required (length width height volume_cube : ℝ) 
  (h_length : length = 14.5) 
  (h_width : width = 17.8) 
  (h_height : height = 7.2) 
  (h_volume_cube : volume_cube = 3) : 
  ⌈(length * width * height) / volume_cube⌉ = 624 := sorry

end min_cubes_required_l425_425361


namespace result_fn_2017_l425_425091

noncomputable def f : ℝ → ℝ := λ x, x * exp x

def derivative (g : ℝ → ℝ) : ℝ → ℝ := 
λ x, (g x).deriv

def fn (n : ℕ) : ℝ → ℝ :=
nat.rec_on n f (λ n fn_x, derivative fn_x)

theorem result_fn_2017 (x : ℝ) :
  fn 2017 x = (x + 2017) * exp x :=
sorry

end result_fn_2017_l425_425091


namespace part1_part2_l425_425069

def setA := { x : ℝ | x^2 - 7*x + 6 ≤ 0 }
def setB (m : ℝ) (h : m > 0) := { x : ℝ | x^2 - 2*x + 1 - m^2 ≤ 0 }

theorem part1 (m : ℝ) (h : m = 1) : 
  A : { x : ℝ | 1 ≤ x ∧ x ≤ 6 },
  B : { x : ℝ | 0 ≤ x ∧ x ≤ 2 },
  x ∈ A ∩ B ↔ 1 ≤ x ∧ x ≤ 2 := 
sorry

theorem part2 (m : ℝ) (h : 0 < m) :
  (∀ x, x ∈ setA → x ∈ setB m h) ↔ 5 ≤ m := 
sorry

end part1_part2_l425_425069


namespace problem_solution_l425_425912

/-- 
Assume we have points A, B, C, D, and E as defined in the problem with the following properties:
- Triangle ABC has a right angle at C
- AC = 4
- BC = 3
- Triangle ABD has a right angle at A
- AD = 15
- Points C and D are on opposite sides of line AB
- The line through D parallel to AC meets CB extended at E.

Prove that the ratio DE/DB simplifies to 57/80 where p = 57 and q = 80, making p + q = 137.
-/
theorem problem_solution :
  ∃ (p q : ℕ), gcd p q = 1 ∧ (∃ D E : ℝ, DE/DB = p/q ∧ p + q = 137) :=
by
  sorry

end problem_solution_l425_425912


namespace probability_of_winning_l425_425701

def lottery_ticket_probability (num_digits : ℕ) (digit_choices : ℕ) (winning_num : ℕ) (prizes : ℕ → ℕ) : Prop :=
  num_digits = 7 ∧ digit_choices = 10 ∧ winning_num = 1234567 ∧
  (prizes 1 = 1 ∧ prizes 2 = 18 ∧ prizes 3 = 261) ∧
  (
    (∃ p1, p1 = 9 / (5 * 10^6)) ∧ (∃ p2, p2 = 7 / (25 * 10^4))
  )

theorem probability_of_winning :
  lottery_ticket_probability 7 10 1234567 (λ n, if n = 1 then 1 else if n = 2 then 18 else if n = 3 then 261 else 0) :=
by
  sorry

end probability_of_winning_l425_425701


namespace negation_of_proposition_l425_425097

-- Definitions using the conditions stated
def p (x : ℝ) : Prop := x^2 - x + 1/4 ≥ 0

-- The statement to prove
theorem negation_of_proposition :
  (¬ (∀ x : ℝ, p x)) = (∃ x : ℝ, ¬ p x) :=
by
  -- Proof will go here; replaced by sorry as per instruction
  sorry

end negation_of_proposition_l425_425097


namespace marsupial_protein_l425_425372

theorem marsupial_protein (absorbed : ℝ) (percent_absorbed : ℝ) (consumed : ℝ) :
  absorbed = 16 ∧ percent_absorbed = 0.4 → consumed = 40 :=
by
  sorry

end marsupial_protein_l425_425372


namespace find_b_for_continuity_at_2_l425_425943

noncomputable def f (x : ℝ) (b : ℝ) :=
if x ≤ 2 then 3 * x^2 + 1 else b * x - 6

theorem find_b_for_continuity_at_2
  (b : ℝ) 
  (h_cont : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) :
  b = 19 / 2 := by sorry

end find_b_for_continuity_at_2_l425_425943


namespace ratio_expression_x_2y_l425_425035

theorem ratio_expression_x_2y :
  ∀ (x y : ℝ), x / (2 * y) = 27 → (7 * x + 6 * y) / (x - 2 * y) = 96 / 13 :=
by
  intros x y h
  sorry

end ratio_expression_x_2y_l425_425035


namespace true_proposition_l425_425068

-- Define proposition p
def p : Prop := ∀ x : ℝ, Real.log (x^2 + 4) / Real.log 2 ≥ 2

-- Define proposition q
def q : Prop := ∀ x : ℝ, x ≥ 0 → x^(1/2) ≤ x^(1/2)

-- Theorem: true proposition is p ∨ ¬q
theorem true_proposition : p ∨ ¬q :=
by
  sorry

end true_proposition_l425_425068


namespace find_angle_A_find_perimeter_l425_425919

noncomputable theory
open Classical

variables {A B C a b c : ℝ}

-- Condition 1: Given equation involving sides and angles
def condition1 := (sqrt 3) * b * cos A = sin A * (a * cos C + c * cos A)

-- Condition 2: A is in the range (0, π)
def condition2 := 0 < A ∧ A < π

-- Condition 3: Given side 'a' value
def condition3 := a = 2 * sqrt 3

-- Condition 4: Given area of the triangle
def condition4 (area : ℝ) := area = (5 * sqrt 3) / 4

-- Theorem 1: Finding the measure of angle A
theorem find_angle_A (h1 : condition1) (h2 : condition2) : A = π / 3 :=
sorry

-- Theorem 2: Finding the perimeter of triangle ABC
theorem find_perimeter (h1 : condition1) (h2 : condition2) (h3 : condition3)
  (h4 : condition4 ((1 / 2) * b * c * sin A)) : a + b + c = 5 * sqrt 3 :=
sorry

end find_angle_A_find_perimeter_l425_425919


namespace total_distance_travelled_l425_425497

-- Definition of the given conditions
def distance_to_market := 30  -- miles
def time_to_home := 0.5  -- hours
def speed_to_home := 20  -- miles per hour

-- The statement we want to prove: Total distance traveled is 40 miles.
theorem total_distance_travelled : 
  distance_to_market + speed_to_home * time_to_home = 40 := 
by 
  sorry

end total_distance_travelled_l425_425497


namespace car_speed_return_trip_l425_425700

noncomputable def speed_return_trip (d : ℕ) (v_ab : ℕ) (v_avg : ℕ) : ℕ := 
  (2 * d * v_avg) / (2 * v_avg - v_ab)

theorem car_speed_return_trip :
  let d := 180
  let v_ab := 90
  let v_avg := 60
  speed_return_trip d v_ab v_avg = 45 :=
by
  simp [speed_return_trip]
  sorry

end car_speed_return_trip_l425_425700


namespace temperature_below_zero_l425_425408

def temperature_denotation := 
  ∀ (t_above_zero t_below_zero : ℝ),
  (t_above_zero = 2 ∧ t_below_zero = -14 ) →
  (t_above_zero = +2 → t_below_zero = -14)

theorem temperature_below_zero :
  temperature_denotation :=
by 
  intros t_above_zero t_below_zero h ht
  cases h
  rw ht
  exact h_right

end temperature_below_zero_l425_425408


namespace cubic_eq_solutions_l425_425437

noncomputable def solutions_to_cubic_eq : set ℂ :=
  {z | z^3 = 27 * complex.I}

theorem cubic_eq_solutions :
  solutions_to_cubic_eq = 
    {3 * complex.I, (-3 * complex.I + 3 * complex.sqrt 3) / 2, (-3 * complex.I - 3 * complex.sqrt 3) / 2} :=
by
  ext
  sorry

end cubic_eq_solutions_l425_425437


namespace one_thirds_in_fraction_l425_425506

theorem one_thirds_in_fraction : (9 / 5) / (1 / 3) = 27 / 5 := by
  sorry

end one_thirds_in_fraction_l425_425506


namespace ratio_of_sister_to_Aaron_l425_425380

noncomputable def Aaron_age := 15
variable (H S : ℕ)
axiom Henry_age_relation : H = 4 * S
axiom combined_age : H + S + Aaron_age = 240

theorem ratio_of_sister_to_Aaron : (S : ℚ) / Aaron_age = 3 := 
by
  -- Proof omitted
  sorry

end ratio_of_sister_to_Aaron_l425_425380


namespace f_zero_f_compare_one_three_l425_425767

section
variable {f : ℝ → ℝ}

-- Condition (i)
def functional_equation (f : ℝ → ℝ) : Prop := ∀ a b : ℝ, f(a + b) = f(a) * f(b)
-- Condition (ii)
def strictly_greater_than_one (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x > 0 → f(x) > 1

-- Existence of function satisfying given conditions
axiom exists_function (f : ℝ → ℝ) (hf1 : functional_equation f) (hf2 : strictly_greater_than_one f) : ∃ f, functional_equation f ∧ strictly_greater_than_one f

-- Proof that f(0) = 1
theorem f_zero {f : ℝ → ℝ} (hf1 : functional_equation f) (hf2 : strictly_greater_than_one f) : f 0 = 1 :=
sorry

-- Compare f(1) and f(3)
theorem f_compare_one_three {f : ℝ → ℝ} (hf1 : functional_equation f) (hf2 : strictly_greater_than_one f) : f 1 < f 3 :=
sorry

end

end f_zero_f_compare_one_three_l425_425767


namespace find_side_c_l425_425901

noncomputable def area_of_triangle (a b c : ℝ) : ℝ := 
  (1 / 4) * (Real.sqrt ((a + (b + c)) * (-a + (b + c)) * (a - (b - c)) * (a + (b - c))))

theorem find_side_c {a b c : ℝ} 
  (h₁ : a = 4)
  (h₂ : b = 5)
  (h₃ : a * b * Real.sin c / 2 = 5 * Real.sqrt 3)
  (h₄ : ∠C < π / 2) :
  c = Real.sqrt 21 :=
sorry

end find_side_c_l425_425901


namespace area_of_inner_square_l425_425722

variable (side_length : ℝ)
variable (A_x : ℝ) (B_x : ℝ)

def square_side_length := 10
def point_A := 1 / 3 * square_side_length
def point_B := 2 / 3 * square_side_length

theorem area_of_inner_square (h_side_length : side_length = square_side_length)
                             (hA : A_x = point_A)
                             (hB : B_x = point_B) :
  let side_inner_square := (point_B - point_A) / 2 in
  let area_inner_square := side_inner_square ^ 2 in
  area_inner_square = 2.79 := 
by 
  sorry

end area_of_inner_square_l425_425722


namespace kids_go_to_camp_l425_425001

theorem kids_go_to_camp (total_kids: Nat) (kids_stay_home: Nat) 
  (h1: total_kids = 1363293) (h2: kids_stay_home = 907611) : total_kids - kids_stay_home = 455682 :=
by
  have h_total : total_kids = 1363293 := h1
  have h_stay_home : kids_stay_home = 907611 := h2
  sorry

end kids_go_to_camp_l425_425001


namespace three_digit_perfect_squares_divisible_by_4_count_l425_425129

theorem three_digit_perfect_squares_divisible_by_4_count : 
  (finset.filter (λ n, (100 ≤ n ∧ n ≤ 999) ∧ (∃ k, n = k * k) ∧ (n % 4 = 0)) (finset.range 1000)).card = 10 :=
by
  sorry

end three_digit_perfect_squares_divisible_by_4_count_l425_425129


namespace equation_of_AB_l425_425835

-- Definitions based on the conditions
def circle_C (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 3

def midpoint_M (p : ℝ × ℝ) : Prop :=
  p = (1, 0)

-- The theorem to be proved
theorem equation_of_AB (x y : ℝ) (M : ℝ × ℝ) :
  circle_C x y ∧ midpoint_M M → x - y = 1 :=
by
  sorry

end equation_of_AB_l425_425835


namespace min_value_of_f_l425_425431

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * x - 3 / x

theorem min_value_of_f : ∃ x < 0, ∀ y : ℝ, y = f x → y ≥ 1 + 2 * Real.sqrt 6 :=
by
  -- Sorry is used to skip the actual proof.
  sorry

end min_value_of_f_l425_425431


namespace andre_total_payment_l425_425213

def treadmill_initial_price : ℝ := 1350
def treadmill_discount : ℝ := 0.30
def plate_initial_price : ℝ := 60
def plate_discount : ℝ := 0.15
def plate_quantity : ℝ := 2

theorem andre_total_payment :
  let treadmill_discounted_price := treadmill_initial_price * (1 - treadmill_discount)
  let plates_total_initial_price := plate_quantity * plate_initial_price
  let plates_discounted_price := plates_total_initial_price * (1 - plate_discount)
  treadmill_discounted_price + plates_discounted_price = 1047 := 
by
  sorry

end andre_total_payment_l425_425213


namespace sum_of_sides_eq_13_or_15_l425_425289

noncomputable def squares_side_lengths (b d : ℕ) : Prop :=
  15^2 = b^2 + 10^2 + d^2

theorem sum_of_sides_eq_13_or_15 :
  ∃ b d : ℕ, squares_side_lengths b d ∧ (b + d = 13 ∨ b + d = 15) :=
sorry

end sum_of_sides_eq_13_or_15_l425_425289


namespace tens_place_digit_of_ten_digit_number_l425_425440

theorem tens_place_digit_of_ten_digit_number : 
  let sequence := [123456789101112131415]
  let grouped_sequence := [1, 23, 456, 7891, 01112, 131415, 1617181, 92021222, 324252627, 2829303132]
  -- From the problem, we know the group we are interested in is formed by concatenating subsequent elements.
  let ten_digit_number := 2829303132
  -- Extract the tens place digit: 3
  ∃ (tens_place_digit : ℕ), (tens_place_digit = 3 ∧ String.sub ten_digit_number 8 1 = $["3"]) :=
sorry

end tens_place_digit_of_ten_digit_number_l425_425440


namespace find_base_r_l425_425171

theorem find_base_r : ∃ r : ℕ, 532_r + 260_r + 208_r = 1000_r ∧ r = 10 :=
by
  sorry -- Proof to be filled in later.

end find_base_r_l425_425171


namespace break_even_performances_l425_425983

theorem break_even_performances :
  ∃ x : ℕ, 16000 * x = 81000 + 7000 * x ∧ x = 9 :=
begin
  use 9,
  split,
  { linarith, },
  { refl, }
end

end break_even_performances_l425_425983


namespace kids_prefer_peas_l425_425889

variable (total_kids children_prefer_carrots children_prefer_corn : ℕ)

theorem kids_prefer_peas (H1 : children_prefer_carrots = 9)
(H2 : children_prefer_corn = 5)
(H3 : children_prefer_corn * 4 = total_kids) :
total_kids - (children_prefer_carrots + children_prefer_corn) = 6 := by
sorry

end kids_prefer_peas_l425_425889


namespace total_carrots_grown_l425_425559

theorem total_carrots_grown (joan_carrots : ℕ) (jessica_carrots : ℕ) (michael_carrots : ℕ) 
  (h1 : joan_carrots = 29) (h2 : jessica_carrots = 11) (h3 : michael_carrots = 37) : 
  joan_carrots + jessica_carrots + michael_carrots = 77 :=
by
  rw [h1, h2, h3]
  norm_num

end total_carrots_grown_l425_425559


namespace calculate_values_l425_425902

def selling_price (x : ℕ) (m n : ℤ) : ℤ :=
if 1 ≤ x ∧ x < 20 then m * (x : ℤ) + n else 30

def sales_volume (x : ℕ) : ℕ := x + 10

def sales_revenue (x : ℕ) (m n : ℤ) : ℤ :=
if 1 ≤ x ∧ x < 20 then (selling_price x m n) * (sales_volume x : ℤ)
else 30 * (sales_volume x : ℤ)

theorem calculate_values : 
  ∃ (m n : ℤ), 
  (∀ x : ℕ, (1 ≤ x ∧ x < 20 → selling_price 5 m n = 50 ∧ selling_price 10 m n = 40)) ∧
  m = -2 ∧ n = 60 ∧
  (∀ x : ℕ, 
    (1 ≤ x ∧ x < 20 → sales_revenue x m n = -2 * (x : ℤ)^2 + 40 * (x : ℤ) + 600) ∧
    (20 ≤ x ∧ x ≤ 30 → sales_revenue x m n = 30 * (x : ℤ) + 300)) ∧
  (card $ {x | 1 ≤ x ∧ x ≤ 30 ∧ sales_revenue x m n > 1000}.to_finset = 7) :=
begin
  sorry
end

end calculate_values_l425_425902


namespace y_coord_vertex_of_parabola_l425_425023

-- Define the quadratic equation of the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2 + 16 * x + 29

-- Statement to prove
theorem y_coord_vertex_of_parabola : ∃ (x : ℝ), parabola x = 2 * (x + 4)^2 - 3 := sorry

end y_coord_vertex_of_parabola_l425_425023


namespace calc_diagonal_of_rectangle_l425_425841

variable (a : ℕ) (A : ℕ)

theorem calc_diagonal_of_rectangle (h_a : a = 6) (h_A : A = 48) (H : a * a' = A) :
  ∃ d : ℕ, d = 10 :=
by
 sorry

end calc_diagonal_of_rectangle_l425_425841


namespace range_exp3_eq_l425_425283

noncomputable def exp3 (x : ℝ) : ℝ := 3^x

theorem range_exp3_eq (x : ℝ) : Set.range (exp3) = Set.Ioi 0 :=
sorry

end range_exp3_eq_l425_425283


namespace point_P_outside_circle_l425_425884

theorem point_P_outside_circle (a b : ℝ) (h : ∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) :
  a^2 + b^2 > 1 :=
sorry

end point_P_outside_circle_l425_425884


namespace min_value_x_plus_2_div_x_l425_425875

theorem min_value_x_plus_2_div_x (x : ℝ) (hx : x > 0) : x + 2 / x ≥ 2 * Real.sqrt 2 :=
sorry

end min_value_x_plus_2_div_x_l425_425875


namespace distance_traveled_by_ball_l425_425739

-- Definitions based on the problem's conditions
variable (a c : ℝ)
variable {A B : ℝ}
def is_focus_a := (A = 0)
def is_focus_b := (B = 2 * c)
def major_axis_length := (2 * a > 2 * c)

-- Main theorem statement
theorem distance_traveled_by_ball :
  A = 0 → B = 2 * c → 
  2 * a > 2 * c → 
  (4 * a = A + (2 * (a - c)) + (2 * (a + c))) :=
begin
  sorry
end

end distance_traveled_by_ball_l425_425739


namespace population_growth_111_percent_l425_425156

-- Define the conditions of the problem
def population_2000 := ∃ p : ℕ, p^2 = 121
def population_2005 := λ p : ℕ, ∃ q : ℕ, p^2 + 121 = q^2 + 16
def population_2015 := λ p q : ℕ, ∃ r : ℕ, p^2 + 346 = r^2

-- Define the percent growth calculation
def percent_growth := λ p : ℕ, (256 - 121) * 100 / 121

-- Define the proof statement
theorem population_growth_111_percent :
  (population_2000) ∧ 
  (population_2005 11) ∧ 
  (population_2015 11 15) ∧ 
  (percent_growth 11 = 111) := 
by {
  sorry
}

end population_growth_111_percent_l425_425156


namespace book_arrangements_l425_425650

/- The problem conditions are broken down into individual definitions. -/
def numArabicBooks := 2
def numGermanBooks := 3
def numSpanishBooks := 4
def totalBooks := numArabicBooks + numGermanBooks + numSpanishBooks

/- We state the problem as a theorem in Lean. -/
theorem book_arrangements : 
  (numArabicBooks = 2) ∧
  (numGermanBooks = 3) ∧
  (numSpanishBooks = 4) →
  let arabic_block_units := 1
  let spanish_block_units := 1
  let german_books_units := 3
  let total_units := arabic_block_units + spanish_block_units + german_books_units
  (Finset.card (Equiv.perm (Fin (total_units + german_books_units)))) * 
  (Finset.card (Equiv.perm (Fin numArabicBooks))) * 
  (Finset.card (Equiv.perm (Fin numSpanishBooks))) = 5760 :=
by
  intros
  sorry

end book_arrangements_l425_425650


namespace negation_exists_equiv_forall_l425_425273

theorem negation_exists_equiv_forall :
  (¬ (∃ x : ℤ, x^2 + 2*x - 1 < 0)) ↔ (∀ x : ℤ, x^2 + 2*x - 1 ≥ 0) :=
by
  sorry

end negation_exists_equiv_forall_l425_425273


namespace AE_BC_passes_through_N_MN_passes_through_fixed_point_midpoint_locus_is_H1H2_l425_425716

-- Define points A, B, O1, O2, G1, H1, H2 and segments AM, MB, AE, BC, MN, and k1, k2 as given in conditions
variable {A B M C D E F N O1 O2 G1 H1 H2 : Point}
variable {AM MB AE BC MN : Line}
variable (k1 k2 : Circle)

-- Declare the given conditions
variables [has_square AMCD] [has_square BMEF]
variables (on_same_side_AB : same_side AB C D) (on_same_side_AB : same_side AB E F)
variables (circumscribed_k1 : circumscribed_circle AMCD k1)
variables (circumscribed_k2 : circumscribed_circle BMEF k2)
variables (N_inter_k1_k2 : k1 ∩ k2 = {M, N})
variables (moves_M_along_AB : ∀ t : ℝ, M = t *• A + (1 - t) *• B)

-- Define the statements to prove:
theorem AE_BC_passes_through_N (h : Point_on N AE ∧ Point_on N BC) : Prop :=
  ∀ (M : Point), (N = intersection (k1, k2)) → Point_on N AE ∧ Point_on N BC

theorem MN_passes_through_fixed_point (h : fixed_point (N : Point) MN G1) : Prop :=
  ∀ (M : Point), (N = intersection (k1, k2)) → point_on (N, MN)

theorem midpoint_locus_is_H1H2 (h : midpoint O1 O2 = O ∧ point_on O H1H2) : Prop :=
  ∀ (M : Point), (O = midpoint O1 O2) → point_on (O, H1H2)  

end AE_BC_passes_through_N_MN_passes_through_fixed_point_midpoint_locus_is_H1H2_l425_425716


namespace one_fourth_div_one_eighth_l425_425503

theorem one_fourth_div_one_eighth : (1 / 4) / (1 / 8) = 2 := by
  sorry

end one_fourth_div_one_eighth_l425_425503


namespace lower_limit_of_prime_range_l425_425295

theorem lower_limit_of_prime_range (x : ℕ) (h1 : x ≤ 14) (h2 : ∃ p1 p2 : ℕ, prime p1 ∧ prime p2 ∧ x < p1 ∧ p1 < p2 ∧ p2 ≤ 14) : x ≤ 7 :=
sorry

end lower_limit_of_prime_range_l425_425295


namespace worker_idle_days_l425_425384

theorem worker_idle_days (W I : ℕ) 
  (h1 : 20 * W - 3 * I = 280)
  (h2 : W + I = 60) : 
  I = 40 :=
sorry

end worker_idle_days_l425_425384


namespace paint_usage_correct_l425_425382

-- Define the parameters representing paint usage and number of paintings
def largeCanvasPaint : Nat := 3
def smallCanvasPaint : Nat := 2
def largePaintings : Nat := 3
def smallPaintings : Nat := 4

-- Define the total paint used
def totalPaintUsed : Nat := largeCanvasPaint * largePaintings + smallCanvasPaint * smallPaintings

-- Prove that total paint used is 17 ounces
theorem paint_usage_correct : totalPaintUsed = 17 :=
  by
    sorry

end paint_usage_correct_l425_425382


namespace largest_solution_achieves_largest_solution_l425_425025

theorem largest_solution (x : ℝ) (hx : ⌊x⌋ = 5 + 100 * (x - ⌊x⌋)) : x ≤ 104.99 :=
by
  -- Placeholder for the proof
  sorry

theorem achieves_largest_solution : ∃ (x : ℝ), ⌊x⌋ = 5 + 100 * (x - ⌊x⌋) ∧ x = 104.99 :=
by
  -- Placeholder for the proof
  sorry

end largest_solution_achieves_largest_solution_l425_425025


namespace cowboy_shortest_distance_l425_425352

theorem cowboy_shortest_distance :
  let C := (0, 3)   -- Cowboy's initial position
  let B := (-5, -3) -- Cabin's position
  let H := (-2, 4)  -- Hill's position
  let stream := (0, 0) -- Stream's position
  distance C stream + distance stream H + distance H B = 3 + 2 * Real.sqrt 5 + Real.sqrt 58 :=
by
  sorry

end cowboy_shortest_distance_l425_425352


namespace xenon_distance_l425_425226

/-
Given an elliptical orbit where the closest point to the star (perigee) is 3 AU and the farthest point (apogee) is 15 AU,
prove that the distance from the star when Xenon is one-fourth the way from perigee to apogee along the major axis of its orbit is 4.5 AU.
-/

theorem xenon_distance 
  (perigee apogee : ℝ) 
  (h_perigee : perigee = 3) 
  (h_apogee : apogee = 15) :
  distance_along_major_axis perigee apogee 1/4 = 4.5 :=
sorry

end xenon_distance_l425_425226


namespace volume_P3_mn_4010_l425_425450

noncomputable def regular_tetrahedron := sorry

def P0_volume := 1
def volume_scaling_factor : ℚ := (1 / 3) ^ 3
def additional_volume (i : ℕ) : ℚ := (4 / 9) * (8 / 9) ^ (i - 1)

def volume (i : ℕ) : ℚ :=
if i = 0 then P0_volume else P0_volume + ∑ j in finset.range i, additional_volume j

theorem volume_P3_mn_4010 :
  volume 3 = 3281 / 729 ∧ let ⟨m, n⟩ := (3281, 729 : ℚ).num_denom_pair in int.gcd m n = 1 ∧ m + n = 4010 := 
by
  sorry

end volume_P3_mn_4010_l425_425450


namespace point_in_plane_region_l425_425733

theorem point_in_plane_region :
  let p : ℝ × ℝ := (0, 6)
  in 2 * p.1 + p.2 - 6 ≤ 0 :=
by
  -- Simplifying the left-hand side of the inequality for the point p = (0, 6)
  let x := p.1
  let y := p.2
  sorry

end point_in_plane_region_l425_425733


namespace solution_to_equation_l425_425022

theorem solution_to_equation :
  ∀ (m n : ℤ), m ≠ 0 ∧ n ≠ 0 ∧ (m^2 + n) * (m + n^2) = (m - n)^3 →
    (m, n) = (-1, -1) ∨ (m, n) = (8, -10) ∨ (m, n) = (9, -6) ∨ (m, n) = (9, -21) :=
by
  intros m n
  sorry

end solution_to_equation_l425_425022


namespace log_identity_l425_425612

theorem log_identity (a b c P : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hP : P > 0) :
  (log a P) * (log b P) + (log b P) * (log c P) + (log a P) * (log c P) = 
  (log a P) * (log b P) * (log c P) / (log (a * b * c) P) :=
by
  sorry

end log_identity_l425_425612


namespace count_three_digit_perfect_squares_divisible_by_4_l425_425118

theorem count_three_digit_perfect_squares_divisible_by_4 : ∃ n, n = 11 ∧
  (∀ k, 100 ≤ k ^ 2 ∧ k ^ 2 ≤ 999 → k ^ 2 % 4 = 0 → (k % 2 = 0 ∧ 10 ≤ k ≤ 31)) :=
sorry

end count_three_digit_perfect_squares_divisible_by_4_l425_425118


namespace plane_split_into_regions_l425_425920

def parabola1 (x : ℝ) : ℝ := 2 - x^2
def parabola2 (x : ℝ) : ℝ := x^2 - 1

theorem plane_split_into_regions : 
  let inter_parabolas := [⟨√(3/2), 1/2⟩, ⟨-√(3/2), 1/2⟩] in
  let inter_parabola1_xaxis := [⟨√2, 0⟩, ⟨-√2, 0⟩] in
  let inter_parabola2_xaxis := [⟨1, 0⟩, ⟨-1, 0⟩] in
  length inter_parabolas + length inter_parabola1_xaxis + length inter_parabola2_xaxis = 10 :=
by sorry

end plane_split_into_regions_l425_425920


namespace ratio_back_to_front_l425_425181

/-- The initial number of cars in the front parking lot -/
def front_cars : ℕ := 100

/-- The total number of cars at the end of the play -/
def total_cars_at_end : ℕ := 700

/-- The number of new cars that arrived during the play -/
def new_cars_during_play : ℕ := 300

/-- The number of cars when they arrived -/
def cars_when_arrived : ℕ := total_cars_at_end - new_cars_during_play

/-- The number of cars in the back parking lot when they arrived -/
def back_cars : ℕ := cars_when_arrived - front_cars

theorem ratio_back_to_front : back_cars / front_cars = 3 := by
  have cars_arrived := total_cars_at_end - new_cars_during_play
  have back := cars_arrived - front_cars
  rw [back, cars_arrived, front_cars]
  sorry

end ratio_back_to_front_l425_425181


namespace find_slope_of_line_l425_425080

noncomputable def slope_of_line_through_point_and_intersecting_circle (k : ℝ) : Prop :=
  let point := (-2 : ℝ, Real.sqrt 3)
  let circle := { center := (-2 : ℝ, 0), radius := 2 }
  let length_of_chord := 2 * Real.sqrt 3
  ∃ k : ℝ, ( (x : ℝ) => y - Real.sqrt 3 = k * (x + 2) ) ∧
           ( (dist : ℝ) => (abs (k * (-2) + 2 * k + Real.sqrt 3) / Real.sqrt (k^2 + 1)) = 1 ) ∧
           ( (k^2 = 2) → k = Real.sqrt 2 ∨ k = - Real.sqrt 2 )

theorem find_slope_of_line (k : ℝ) :
  slope_of_line_through_point_and_intersecting_circle k :=
by
  sorry

end find_slope_of_line_l425_425080


namespace find_point_D_l425_425228

variables (a b : ℝ)

def point_A : ℝ × ℝ := (2, 8)
def point_B : ℝ × ℝ := (0, 0)
def point_C : ℝ × ℝ := (5, 3)
def point_D : ℝ × ℝ := (a, b)

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def is_square (M N O P : ℝ × ℝ) : Prop :=
  let v1 := (N.1 - M.1, N.2 - M.2)
  let v2 := (O.1 - N.1, O.2 - N.2)
  let v3 := (P.1 - O.1, P.2 - O.2)
  let v4 := (M.1 - P.1, M.2 - P.2) in
  v1.1 * v2.1 + v1.2 * v2.2 = 0 ∧
  v2.1 * v3.1 + v2.2 * v3.2 = 0 ∧
  v3.1 * v4.1 + v3.2 * v4.2 = 0 ∧
  v4.1 * v1.1 + v4.2 * v1.2 = 0 ∧
  (v1.1 ^ 2 + v1.2 ^ 2) = (v2.1 ^ 2 + v2.2 ^ 2) ∧
  (v2.1 ^ 2 + v2.2 ^ 2) = (v3.1 ^ 2 + v3.2 ^ 2) ∧
  (v3.1 ^ 2 + v3.2 ^ 2) = (v4.1 ^ 2 + v4.2 ^ 2)

theorem find_point_D : point_D a b = (5, 3) :=
  let M1 := midpoint point_A point_B in
  let M2 := midpoint point_B point_C in
  let M3 := midpoint point_C point_D a b in
  let M4 := midpoint point_D a b point_A in
  is_square M1 M2 M3 M4
    sorry

end find_point_D_l425_425228


namespace sufficient_not_necessary_condition_l425_425771

-- Define the quadratic function
def f (x t : ℝ) : ℝ := x^2 + t * x - t

-- The proof statement about the condition for roots
theorem sufficient_not_necessary_condition (t : ℝ) :
  (t ≥ 0 → ∃ x : ℝ, f x t = 0) ∧ (∃ x : ℝ, f x t = 0 → t ≥ 0 ∨ t ≤ -4) :=
sorry

end sufficient_not_necessary_condition_l425_425771


namespace avg_remaining_numbers_l425_425635

-- Conditions
variable (numbers : Fin 12 → ℝ) (h_avg : (∑ i, numbers i) / 12 = 90)
variable (number_80 : ∃ j, numbers j = 80) (number_82 : ∃ k, numbers k = 82)

-- Theorem statement
theorem avg_remaining_numbers : 
  (∑ i, numbers i - 80 - 82) / 10 = 91.8 :=
by 
  sorry

end avg_remaining_numbers_l425_425635


namespace halfway_between_fractions_l425_425000

-- Definitions used in the conditions
def one_eighth := (1 : ℚ) / 8
def three_tenths := (3 : ℚ) / 10

-- The mathematical assertion to prove
theorem halfway_between_fractions : (one_eighth + three_tenths) / 2 = 17 / 80 := by
  sorry

end halfway_between_fractions_l425_425000


namespace vector_perpendicular_iff_l425_425863

theorem vector_perpendicular_iff (k : ℝ) :
  let a := (Real.sqrt 3, 1)
  let b := (0, 1)
  let c := (k, Real.sqrt 3)
  let ab := (Real.sqrt 3, 3)  -- a + 2b
  a.1 * c.1 + ab.2 * c.2 = 0 → k = -3 :=
by
  let a := (Real.sqrt 3, 1)
  let b := (0, 1)
  let c := (k, Real.sqrt 3)
  let ab := (Real.sqrt 3, 3)  -- a + 2b
  intro h
  sorry

end vector_perpendicular_iff_l425_425863


namespace range_of_a_l425_425446

theorem range_of_a (a : ℝ) (x : ℝ) (h₁ : a < 0) (h₂ : x^2 - 4 * a * x + 3 * a^2 < 0) :
  x^2 + 5 * x + 4 ≤ 0 ↔ - (4 : ℝ) / 3 ≤ a ∧ a ≤ -1 :=
begin
  sorry
end

end range_of_a_l425_425446


namespace perfect_squares_three_digit_divisible_by_4_count_l425_425138

theorem perfect_squares_three_digit_divisible_by_4_count : 
  ∃ (n : ℕ), (n = 11) ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ (∃ k, m = k^2) ∧ (m % 4 = 0) ↔ m ∈ {100, 144, 196, 256, 324, 400, 484, 576, 676, 784, 900}) :=
by
  existsi 11
  split
  · reflexivity
  · intro m
    split
    · intro h
      rcases h with ⟨⟨_, _, _, _⟩, _⟩
      -- details have been omitted
      sorry
    · intro h
      -- details have been omitted
      sorry

end perfect_squares_three_digit_divisible_by_4_count_l425_425138


namespace quadratic_eq_solution_1_quadratic_eq_solution_2_l425_425628

theorem quadratic_eq_solution_1 :
    ∀ (x : ℝ), x^2 - 8*x + 1 = 0 ↔ x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15 :=
by 
  sorry

theorem quadratic_eq_solution_2 :
    ∀ (x : ℝ), x * (x - 2) - x + 2 = 0 ↔ x = 1 ∨ x = 2 :=
by 
  sorry

end quadratic_eq_solution_1_quadratic_eq_solution_2_l425_425628


namespace distinguishable_rearrangements_of_contest_with_vowels_first_l425_425110

open Finset

theorem distinguishable_rearrangements_of_contest_with_vowels_first :
  let vowels := ['O', 'E'],
      consonants := ['C', 'N', 'T', 'S', 'T'],
      vowel_arrangements := 2!, -- 2! arrangements of vowels
      consonant_arrangements := 5! / 2! -- 5! arrangements of consonants accounted for repetition of 'T'
  in
  vowel_arrangements * consonant_arrangements = 120 := by
  let vowels := ['O', 'E'],
      consonants := ['C', 'N', 'T', 'S', 'T'],
      vowel_arrangements := Nat.factorial 2, -- 2! arrangements of vowels
      consonant_arrangements := Nat.factorial 5 / Nat.factorial 2 -- 5! arrangements of consonants accounted for repetition of 'T'
  show vowel_arrangements * consonant_arrangements = 120
  from calc
  2 * 60 = 120 : by sorry

#print distinguishable_rearrangements_of_contest_with_vowels_first

-- sorry statement to skip the proof.

end distinguishable_rearrangements_of_contest_with_vowels_first_l425_425110


namespace convex_quadrilateral_exists_pairs_with_intersections_l425_425689

-- Problem 1
theorem convex_quadrilateral_exists (points : Fin₅ → ℝ × ℝ) (h_no_three_collinear : ∀ (A B C : Fin₅), 
  A ≠ B → A ≠ C → B ≠ C → ¬Collinear ℝ {points A, points B, points C}) : 
  ∃ (A B C D : Fin₅), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧ convex_hull ℝ {points A, points B, points C, points D} = {points A, points B, points C, points D} := 
sorry

-- Problem 2
theorem pairs_with_intersections (n : ℕ) (points : Fin (4 * n + 1) → ℝ × ℝ) (h_no_three_collinear : ∀ (A B C : Fin (4 * n + 1)), 
  A ≠ B → A ≠ C → B ≠ C → ¬Collinear ℝ {points A, points B, points C}) : 
  ∃ (pairs : Fin (2 * n) → (ℝ × ℝ) × (ℝ × ℝ)), 
  (∀ i j, i ≠ j → segments_intersect (pairs i).1 (pairs i).2 (pairs j).1 (pairs j).2) ∧ 
    ∃ k ≥ n, distinct_segments_with_intersection (pairs k).1 (pairs k).2 := 
sorry

end convex_quadrilateral_exists_pairs_with_intersections_l425_425689


namespace train_length_l425_425726

theorem train_length
  (speed_kmph : ℕ)
  (time_sec : ℕ)
  (bridge_len_m : ℕ)
  (h_speed_kmph : speed_kmph = 45)
  (h_time_sec : time_sec = 30)
  (h_bridge_len_m : bridge_len_m = 235) :
  let speed_mps := speed_kmph * 1000 / 3600 in
  let distance := speed_mps * time_sec in
  let train_len := distance - bridge_len_m in
  train_len = 140 :=
by
  sorry

end train_length_l425_425726


namespace circles_have_equal_radii_l425_425937

-- Definitions based on the given conditions
variables (n : ℕ) (odd_n : n % 2 = 1)
variables (A : Fin n → Point)  -- Inscribed polygon vertices
variables (O : Point)  -- Center of circumcircle
variables (R : ℝ)  -- Radius of circumcircle
variables (radii : Fin n → ℝ) -- Radii of the internally tangent circles

-- Condition: A system of circles internally tangent to the given circumcircle
-- Proof statement begins
theorem circles_have_equal_radii 
  (H1 : ∀ i : Fin n, (is_tangent (circle (A i) R) (circle (A ((i + 1) % n)) R)))
  (H2 : ∀ i : Fin n, ∃ (P : Point), P ∈ (line_segment (A i) (A ((i + 1) % n)))) :
  ∀ i : Fin n, radii i = radii 0 := 
begin
  sorry
end

end circles_have_equal_radii_l425_425937


namespace billys_mom_gave_10_dollars_l425_425389

def bottle := (oz : ℕ) (cost : ℚ)

def bottles : List bottle := [
  bottle.mk 10 1,
  bottle.mk 16 2,
  bottle.mk 25 2.5,
  bottle.mk 50 5,
  bottle.mk 200 10
]

def cost_per_ounce (b : bottle) : ℚ :=
  b.cost / b.oz

theorem billys_mom_gave_10_dollars :
  ∃ (b : bottle), b ∈ bottles ∧ cost_per_ounce b = 0.05 ∧ b.cost = 10 :=
by
  /- We define the conditions based on the problem statement:
     1. The list of bottles each with their respective (oz, cost) pairs.
     2. Function to calculate cost per ounce.
     3. The goal theorem to prove that the best deal is indeed the bottle costing $10.-/
  apply Exists.intro (bottle.mk 200 10)
  sorry

end billys_mom_gave_10_dollars_l425_425389


namespace min_value_of_hyperbola_l425_425644

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sqrt 3 * a
noncomputable def c : ℝ := 2 * a
noncomputable def e : ℝ := sorry

def hyperbola (x y : ℝ) (a b : ℝ) :=
  (x^2 / a^2) - (y^2 / b^2) = 1

theorem min_value_of_hyperbola :
  (∀ (a b e : ℝ), a > 0 ∧ b > 0 ∧ b = sqrt 3 * a ∧ c = 2 * a → 
  (a^2 + e) / b ≥ 2 * sqrt 6 / 3) :=
begin
  sorry
end

end min_value_of_hyperbola_l425_425644


namespace complete_set_is_R_l425_425365

-- Define a non-empty complete set A of real numbers
def is_complete (A : set ℝ) : Prop :=
  ∃ a : ℝ, a ∈ A ∧ (∀ a b : ℝ, a + b ∈ A → a * b ∈ A)

-- Formalize the theorem statement
theorem complete_set_is_R (A : set ℝ) (hA : is_complete A) : A = set.univ :=
by
  sorry

end complete_set_is_R_l425_425365


namespace sum_cards_condition_l425_425420

theorem sum_cards_condition :
  (∑ (n : ℕ) in finset.filter (λ n, 21 * n > 168) (finset.range 10), n) = 9 :=
  sorry

end sum_cards_condition_l425_425420


namespace probability_less_than_condition_l425_425713

def diameter (sum_of_dice : ℕ) : ℝ := sum_of_dice

def area (d : ℝ) : ℝ := Real.pi * (d / 2) * (d / 2)

def circumference (d : ℝ) : ℝ := Real.pi * d

def less_than_condition (d : ℝ) : Prop :=
  area d < circumference d

def valid_d_values (d : ℕ) : Prop :=
  d = 2 ∨ d = 3

noncomputable def probability : ℝ :=
  (1 / 64) + (2 / 64)

theorem probability_less_than_condition :
  (∑ d in Finset.filter valid_d_values (Finset.range 17), 
      ite (valid_d_values d) ((1 : ℝ) / 64) 0) = 3 / 64 :=
by
  sorry

end probability_less_than_condition_l425_425713


namespace owen_sleep_hours_l425_425959

variable (hours_work : ℕ)
variable (hours_commuting : ℕ)
variable (hours_exercise : ℕ)
variable (hours_cooking : ℕ)
variable (hours_relaxation : ℕ)
variable (hours_grooming : ℚ)

def total_hours_per_day := 24
def total_hours_activities := (hours_work : ℚ) + (hours_commuting : ℚ) + (hours_exercise : ℚ) + (hours_cooking : ℚ) + (hours_relaxation : ℚ) + hours_grooming
def hours_sleep := (total_hours_per_day : ℚ) - total_hours_activities

theorem owen_sleep_hours
  (h1 : hours_work = 6)
  (h2 : hours_commuting = 2)
  (h3 : hours_exercise = 3)
  (h4 : hours_cooking = 1)
  (h5 : hours_relaxation = 3)
  (h6 : hours_grooming = (1.5 : ℚ)) :
  hours_sleep = 7.5 := by
  sorry

end owen_sleep_hours_l425_425959


namespace chord_slope_l425_425088

-- Define the ellipse and the point P(4, 2) conditions
def ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1
def P := (4 : ℝ, 2 : ℝ)

-- Problem: Prove that the slope k of the line passing through the chord with P as the midpoint is -1/2
theorem chord_slope {x1 x2 y1 y2 k : ℝ}
  (hA : ellipse x1 y1)
  (hB : ellipse x2 y2)
  (midpoint : (x1 + x2) / 2 = P.1 ∧ (y1 + y2) / 2 = P.2) :
  k = (y1 - y2) / (x1 - x2) → k = -1/2 :=
sorry

end chord_slope_l425_425088


namespace exists_point_in_k_gons_l425_425895

theorem exists_point_in_k_gons (n k : ℕ) (kgons : ℕ → Set (ℝ × ℝ)) 
  (h1 : ∀ i j, i ≠ j → (kgons i ∩ kgons j).Nonempty)
  (h2 : ∀ i j, ∃ c > 0, is_homothety (kgons i) (kgons j) c) : 
  ∃ p : ℝ × ℝ, ∃ m ≥ 1 + (n - 1) / (2 * k), 
  ∃ t : Fin m → ℕ, ∀ i, p ∈ kgons (t i) :=
sorry

end exists_point_in_k_gons_l425_425895


namespace area_between_polar_curves_l425_425756

theorem area_between_polar_curves :
  let f := λ θ : ℝ, cos θ
  let g := λ θ : ℝ, 2 * cos θ
  let α := - (Real.pi / 2)
  let β := Real.pi / 2
  (1 / 2) * ∫ θ in α..β, (g θ)^2 - (f θ)^2 = (3 * Real.pi / 4) :=
by
  sorry

end area_between_polar_curves_l425_425756


namespace number_of_laborers_l425_425350

theorem number_of_laborers (x : ℕ) :
  (18 * (x - 10)) = x → x = 11 :=
by
  assume h : 18 * (x - 10) = x
  have : 18 * (x - 10) = x, from h
  sorry

end number_of_laborers_l425_425350


namespace part1_monotonicity_part2_inequality_l425_425092

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem part1_monotonicity (a : ℝ) :
  (∀ x : ℝ, a ≤ 0 → f a x < f a (x + 1)) ∧
  (a > 0 → ∀ x : ℝ, (x < Real.log (1 / a) → f a x > f a (x + 1)) ∧
  (x > Real.log (1 / a) → f a x < f a (x + 1))) := sorry

theorem part2_inequality (a : ℝ) (ha : a > 0) :
  ∀ x : ℝ, f a x > 2 * Real.log a + (3 / 2) := sorry

end part1_monotonicity_part2_inequality_l425_425092


namespace senate_arrangement_l425_425344

def countArrangements : ℕ :=
  let totalSeats : ℕ := 14
  let democrats : ℕ := 6
  let republicans : ℕ := 6
  let independents : ℕ := 2
  -- The calculation for arrangements considering fixed elements, and permutations adjusted for rotation
  12 * (Nat.factorial 10 / 2)

theorem senate_arrangement :
  let totalSeats : ℕ := 14
  let democrats : ℕ := 6
  let republicans : ℕ := 6
  let independents : ℕ := 2
  -- Total ways to arrange the members around the table under the given conditions
  countArrangements = 21772800 :=
by
  sorry

end senate_arrangement_l425_425344


namespace hyperbola_equation_l425_425079

theorem hyperbola_equation (h1 : ∀ x y : ℝ, (x = 0 ∧ y = 0)) 
                           (h2 : ∀ a : ℝ, (2 * a = 4)) 
                           (h3 : ∀ c : ℝ, (c = 3)) : 
  ∃ b : ℝ, (b^2 = 5) ∧ (∀ x y : ℝ, (y^2 / 4) - (x^2 / b^2) = 1) :=
sorry

end hyperbola_equation_l425_425079


namespace three_element_subsets_condition_l425_425783

theorem three_element_subsets_condition 
  (n : ℕ) (h : n ≥ 4) : 
  (∀ (s : Finset (Finset (Fin n))) (hs : s.card = n) (h3 : ∀ t ∈ s, t.card = 3),
     ∃ (x y ∈ s), x ≠ y ∧ (x ∩ y).card = 1) ↔ (n % 4 ≠ 0) :=
by sorry

end three_element_subsets_condition_l425_425783


namespace correct_answer_l425_425480

variable (x : ℝ)

-- Define p and q as per conditions
def p := ∀ x > 0, Real.ln x > Real.log x
def q := ∃ x > 0, Real.sqrt x = 1 - x^2

-- Define the correct answer statement
theorem correct_answer :
  (¬p) ∧ q :=
by sorry

end correct_answer_l425_425480


namespace categorize_numbers_l425_425781

def is_negative_rational (x : ℚ) : Prop := x < 0
def is_positive_fraction (x : ℚ) : Prop := x > 0 ∧ x.denom ≠ 1
def is_non_positive_integer (x : ℤ) : Prop := x ≤ 0

theorem categorize_numbers (x : Int) (y : Rat) : 
  (x ∈ {-8, -|2|, 0 : Int} ↔ x ≤ 0) ∧ 
  (y ∈ {\frac{22}{7}, 5.4} ↔ y > 0 ∧ y.denom ≠ 1) ∧ 
  (x ∈ {-8, -|2|, 0 : Int} ↔ x < 0) :=
by 
  sorry

end categorize_numbers_l425_425781


namespace min_black_cells_l425_425898

theorem min_black_cells (n : ℕ) (is_white : ℕ → ℕ → Prop) (is_black : ℕ → ℕ → Prop) 
  (H1 : ∀ (i j : ℕ), 0 ≤ i ∧ i < n ∧ 0 ≤ j ∧ j < n → (is_white i j ∨ is_black i j))
  (H2 : ∀ (i j : ℕ), is_white i j → (is_black (i+1) j ∨ is_black (i-1) j ∨ is_black i (j+1) ∨ is_black i (j-1)))
  (H3 : ∀ i j k l, (is_black i j ∧ is_black k l) → connected (i, j) (k, l)) :
  ∃ k, k ≥ (n^2 - 2) / 3 := sorry

end min_black_cells_l425_425898


namespace length_more_than_breadth_by_10_l425_425269

-- Definitions based on conditions
def length : ℕ := 55
def cost_per_meter : ℚ := 26.5
def total_fencing_cost : ℚ := 5300
def perimeter : ℚ := total_fencing_cost / cost_per_meter

-- Calculate breadth (b) and difference (x)
def breadth := 45 -- This is inferred manually from the solution for completeness
def difference (b : ℚ) := length - b

-- The statement we need to prove
theorem length_more_than_breadth_by_10 :
  difference 45 = 10 :=
by
  sorry

end length_more_than_breadth_by_10_l425_425269


namespace diagonal_of_given_rectangular_solid_l425_425982

def diagonal_length_of_rectangular_solid (a b c : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2 + c^2)

theorem diagonal_of_given_rectangular_solid :
  diagonal_length_of_rectangular_solid 2 3 4 = Real.sqrt 29 :=
by
  exact sorry

end diagonal_of_given_rectangular_solid_l425_425982


namespace jane_sleep_hours_for_second_exam_l425_425557

theorem jane_sleep_hours_for_second_exam :
  ∀ (score1 score2 hours1 hours2 : ℝ),
  score1 * hours1 = 675 →
  (score1 + score2) / 2 = 85 →
  score2 * hours2 = 675 →
  hours2 = 135 / 19 :=
by
  intros score1 score2 hours1 hours2 h1 h2 h3
  sorry

end jane_sleep_hours_for_second_exam_l425_425557


namespace red_bowl_values_possible_l425_425656

theorem red_bowl_values_possible (r b y : ℕ) 
(h1 : r + b + y = 27)
(h2 : 15 * r + 3 * b + 18 * y = 378) : 
  r = 11 ∨ r = 16 ∨ r = 21 := 
  sorry

end red_bowl_values_possible_l425_425656


namespace smallest_four_digit_mod_35_l425_425774

theorem smallest_four_digit_mod_35 :
  ∃ x : ℤ, 1000 ≤ x ∧ x < 10000 ∧ x ≡ 4 [MOD 35] ∧ ∀ y : ℤ, 1000 ≤ y ∧ y < 10000 ∧ y ≡ 4 [MOD 35] → x ≤ y :=
sorry

end smallest_four_digit_mod_35_l425_425774


namespace calculate_speed_of_second_fragment_l425_425356

noncomputable def speed_of_second_fragment
  (initial_speed : ℝ)
  (time_after_start : ℝ)
  (horizontal_speed_small_fragment : ℝ)
  (mass_ratio : ℕ)
  (gravity : ℝ)
  : ℝ :=
let vertical_speed_before_explosion := initial_speed - gravity * time_after_start,
    v1x := horizontal_speed_small_fragment,
    v1y := vertical_speed_before_explosion,
    v2x := - (v1x / 2),
    v2y := (3 * v1y / 2),
    magnitude_v2 := real.sqrt (v2x^2 + v2y^2) in
magnitude_v2

theorem calculate_speed_of_second_fragment :
  speed_of_second_fragment 20 1 16 2 10 = 2 * real.sqrt 41 := by
  sorry

end calculate_speed_of_second_fragment_l425_425356


namespace high_temp_three_years_same_l425_425220

theorem high_temp_three_years_same
  (T : ℝ)                               -- The high temperature for the three years with the same temperature
  (temp2017 : ℝ := 79)                   -- The high temperature for 2017
  (temp2016 : ℝ := 71)                   -- The high temperature for 2016
  (average_temp : ℝ := 84)               -- The average high temperature for 5 years
  (num_years : ℕ := 5)                   -- The number of years to consider
  (years_with_same_temp : ℕ := 3)        -- The number of years with the same high temperature
  (total_temp : ℝ := average_temp * num_years) -- The sum of the high temperatures for the 5 years
  (total_known_temp : ℝ := temp2017 + temp2016) -- The known high temperatures for 2016 and 2017
  (total_for_three_years : ℝ := total_temp - total_known_temp) -- Total high temperatures for the three years
  (high_temp_per_year : ℝ := total_for_three_years / years_with_same_temp) -- High temperature per year for three years
  :
  T = 90 :=
sorry

end high_temp_three_years_same_l425_425220


namespace sum_S17_l425_425055

-- Definitions of the required arithmetic sequence elements.
variable (a1 d : ℤ)

-- Definition of the arithmetic sequence
def aₙ (n : ℤ) : ℤ := a1 + (n - 1) * d
def Sₙ (n : ℤ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

-- Theorem for the problem statement
theorem sum_S17 : (aₙ a1 d 7 + aₙ a1 d 5) = (3 + aₙ a1 d 5) → (a1 + 8 * d = 3) → Sₙ a1 d 17 = 51 :=
by
  intros h1 h2
  sorry

end sum_S17_l425_425055


namespace locus_of_B_and_C_spherical_surfaces_l425_425651

noncomputable def point_A := (A' : ℝ, A'' : ℝ)
def angle_diff := (β - γ : ℝ)
def length_bisector := (f_a : ℝ)
def ratio_b_c_a := (b + c : ℝ) / (a : ℝ)
def endpoint_D_on_first_projection_plane := true

theorem locus_of_B_and_C_spherical_surfaces 
  (A' A'' β γ f_a b c a : ℝ)
  (h_env : endpoint_D_on_first_projection_plane)
  (hc1 : point_A = (A', A''))
  (hc2 : angle_diff = β - γ)
  (hc3 : length_bisector = f_a)
  (hc4 : ratio_b_c_a = (b + c) / a) : 
  ∃ B C : ℝ × ℝ, locus_of_points B C (β - γ) f_a = spherical_surfaces :=
sorry

end locus_of_B_and_C_spherical_surfaces_l425_425651


namespace sum_of_coefficients_l425_425142

theorem sum_of_coefficients (a a1 a2 a3 a4 a5 a6 a7 : ℤ) (a_eq : (1 - 2 * (0:ℤ)) ^ 7 = a)
  (hx_eq : ∀ (x : ℤ), (1 - 2 * x) ^ 7 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) :
  a1 + a2 + a3 + a4 + a5 + a6 + a7 = -2 :=
by
  sorry

end sum_of_coefficients_l425_425142


namespace problem_statement_l425_425412

-- Define the avg function
def avg (x y : ℝ) := (x + y) / 2

-- Define the operation M
def M (x y c : ℝ) := avg x y + c

-- Definitions for properties
def isCommutative (x y c : ℝ) := M x y c = M y x c

def additionDistributesOverM (x y z c : ℝ) := 
  x + M y z c = M (x + y) (x + z) c

-- Main theorem
theorem problem_statement (c : ℝ) :
  (∀ x y, isCommutative x y c) ∧ 
  (∀ x y z, additionDistributesOverM x y z c) :=
by
  sorry

end problem_statement_l425_425412


namespace parametric_to_polar_polar_to_rectangular_max_distance_l425_425095

theorem parametric_to_polar (x y t : ℝ) (ρ θ : ℝ) :
  (x = 2 + t ∧ y = 2 - 2t) →
  (ρ * cos θ = x) ∧ (ρ * sin θ = y) →
  2 * ρ * cos θ + ρ * sin θ - 6 = 0 :=
by sorry

theorem polar_to_rectangular (ρ θ : ℝ) :
  (ρ = 2 / real.sqrt (1 + 3 * cos θ ^ 2)) →
  (ρ ^ 2 + 3 * ρ ^ 2 * cos θ ^ 2 = 4) →
  ∃ x y : ℝ, (ρ = sqrt (x^2 + y^2)) ∧ (ρ^2 + 3 * ρ^2 * (x/y)^2 = 4) :=
by sorry

theorem max_distance (θ : ℝ) :
  let ρ := 2 / real.sqrt (1 + 3 * cos θ ^ 2)
  let P_x := ρ * cos θ
  let P_y := ρ * sin θ
  let d := (1 / real.sqrt 5) * |2 * cos θ + 2 * sin θ - 6|
  let PA := d / real.sqrt(3) / 2
  |PA| = (2 * real.sqrt(15) / 15) * (2 * real.sqrt(2) - 6) :=
by sorry

end parametric_to_polar_polar_to_rectangular_max_distance_l425_425095


namespace infinitely_many_primes_dividing_sequence_l425_425246

open Nat

def f (n : ℕ) : ℤ := sorry  -- Assuming we have a nonconstant polynomial function f with integer coefficients.

def a (k : ℕ) : ℤ := (list.prod (list.map f (list.range (k+1)))) + 1

theorem infinitely_many_primes_dividing_sequence
  (f_nonconstant : nonconstant_polynomial f) 
  (f0_odd : Odd (f 0))
  (f1_odd : Odd (f 1)) :
  ∃ᶠ p in nat.primes, ∃ k, p ∣ a k := 
sorry

end infinitely_many_primes_dividing_sequence_l425_425246


namespace eccentricity_of_ellipse_l425_425830

variables (a b : ℝ) (A B C D M : ℝ × ℝ) (λ : ℝ)
def E (x y : ℝ) := (x^2) / (a^2) + (y^2) / (b^2) = 1
def M := (2, 1)
def lines := (A, C, B, D : ℝ × ℝ) -- Points of intersection
def condition1 := a > b ∧ b > 0 
def condition2 := λ > 0 ∧ λ ≠ 1
def slope_condition := (y2 - y1) / (x2 - x1) = -1 / 2

theorem eccentricity_of_ellipse (h1 : condition1(a, b))
                                (h2 : E A.1 A.2 = E M.1 M.2)
                                (h3 : E B.1 B.2 = E M.1 M.2)
                                (h4 : E C.1 C.2 = E M.1 M.2)
                                (h5 : E D.1 D.2 = E M.1 M.2)
                                (h6 : condition2 λ)
                                (h7 : slope_condition) :
  eccentricity (a b) = sqrt(1 - b^2 / a^2) := sorry

end eccentricity_of_ellipse_l425_425830


namespace max_value_f_l425_425430

noncomputable def f : ℝ → ℝ := λ x, 2^x - 16^x

theorem max_value_f : ∃ x, f x = 1 / 4 :=
by
  use (-1 / 2)
  simp [f]
  sorry

end max_value_f_l425_425430


namespace largest_divisor_of_difference_of_squares_l425_425195

theorem largest_divisor_of_difference_of_squares (m n : ℤ) (hm : m % 2 = 1) (hn : n % 2 = 1) (h : n < m) :
  ∃ k, (∀ m n : ℤ, m % 2 = 1 → n % 2 = 1 → n < m → k ∣ (m^2 - n^2)) ∧ (∀ j : ℤ, (∀ m n : ℤ, m % 2 = 1 → n % 2 = 1 → n < m → j ∣ (m^2 - n^2)) → j ≤ k) ∧ k = 8 :=
sorry

end largest_divisor_of_difference_of_squares_l425_425195


namespace prob1_prob2_prob3_l425_425414

-- Define the polar to rectangular conversion
def polar_to_rect (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

-- Theorems for conversions
theorem prob1 : ∀ (ρ θ : ℝ), ρ = cos θ + 2 * sin θ → let (x, y) := polar_to_rect ρ θ in x^2 + y^2 = x + 2 * y :=
by sorry

theorem prob2 : ∀ (ρ θ : ℝ), ρ = 1 + sin θ → let (x, y) := polar_to_rect ρ θ in x^2 + y^2 = 1 - 2 * y :=
by sorry

theorem prob3 : ∀ (ρ θ : ℝ), ρ^3 * sin θ * cos (2 * θ) = ρ^2 * cos (2 * θ) - ρ * sin θ + 1 →
  let (x, y) := polar_to_rect ρ θ in y = 1 ∨ y = √(x^2 - 1) ∨ y = -√(x^2 - 1) :=
by sorry

end prob1_prob2_prob3_l425_425414


namespace max_value_of_function_max_value_achieved_l425_425271

theorem max_value_of_function : ∀ x : ℝ, (1 + sin x) ≤ 2 :=
by
  intro x
  have h : sin x ≤ 1 := sin_le_one x
  calc
    1 + sin x ≤ 1 + 1 : add_le_add_left h 1
          ... = 2 : by rfl

theorem max_value_achieved : ∃ x : ℝ, (1 + sin x) = 2 :=
by
  use (π / 2)  -- π / 2 is one x for which sin x = 1
  calc
    1 + sin (π / 2) = 1 + 1 : by rw sin_pi_div_two
                ... = 2 : by rfl

end max_value_of_function_max_value_achieved_l425_425271


namespace find_m_l425_425317

theorem find_m (m : ℤ) (y : ℤ) : 
  (y^2 + m * y + 2) % (y - 1) = (m + 3) ∧ 
  (y^2 + m * y + 2) % (y + 1) = (3 - m) ∧
  (m + 3 = 3 - m) → m = 0 :=
sorry

end find_m_l425_425317


namespace regular_hexagon_area_l425_425970

noncomputable def hexagon_area
  (P R : ℝ × ℝ) -- vertices P and R
  (hP : P = (2, 2)) -- condition for P
  (hR : R = (10, 3)) -- condition for R
  (h_regular : True) : -- condition for regular hexagon (assume true for simplicity)
  ℝ :=
  have PR := Math.sqrt ((10 - 2)^2 + (3 - 2)^2),
  have side_square : ℝ := PR^2,
  let triangle_area := Math.sqrt 3 / 4 * side_square in
  2 * triangle_area
  

-- The statement we aim to prove
theorem regular_hexagon_area :
  hexagon_area (2, 2) (10, 3) (by rfl) (by rfl) true = 65 * Math.sqrt 3 / 2 :=
by
  sorry

end regular_hexagon_area_l425_425970


namespace hexagon_area_l425_425189

noncomputable def trapezoid_area (a b h : ℝ) : ℝ := 1/2 * (a + b) * h

noncomputable def triangle_area (b h : ℝ) : ℝ := 1/2 * b * h

theorem hexagon_area (EF GH HE FG : ℝ) (y : ℝ) (hEF : EF = 13) (hGH : GH = 21) (hHE : HE = 9) (hFG : FG = 7) (hy : y = 3 * real.sqrt 13 / 4) :
  let EFGH_area := trapezoid_area EF GH (2 * y)
  let triangle_HER_area := triangle_area HE y
  let triangle_FGS_area := triangle_area FG y
  EFGH_area - (triangle_HER_area + triangle_FGS_area) = 19.5 * real.sqrt 13 :=
by
  sorry

end hexagon_area_l425_425189


namespace calc_sqrt_expr_l425_425392

theorem calc_sqrt_expr :
  (3 + Real.sqrt 7) * (3 - Real.sqrt 7) = 2 := by
  sorry

end calc_sqrt_expr_l425_425392


namespace correct_graph_A_l425_425474

def f (x : ℝ) : ℝ :=
if -3 ≤ x ∧ x ≤ 0 then -2 - x
else if 0 ≤ x ∧ x ≤ 2 then sqrt(4 - (x - 2)^2) - 2
else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
else 0

def abs_f (x : ℝ) : ℝ :=
abs (f x)

theorem correct_graph_A : 
  (∀ x, abs_f x = |(if -3 ≤ x ∧ x ≤ 0 then 2 + x
                   else if 0 ≤ x ∧ x ≤ 2 then 2 - sqrt(4 - (x - 2)^2)
                   else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2))
    ∘ x) := sorry

end correct_graph_A_l425_425474


namespace inequality_solution_l425_425785

theorem inequality_solution (x : ℝ) :
  (x^2 / (x + 1) ≥ 3 / (x - 2) + 9 / 4) ↔ (x ∈ Set.Ioo (-∞) (-3/4) ∪ Set.Ioo 2 5) := by
  sorry

end inequality_solution_l425_425785


namespace polynomial_condition_l425_425848

theorem polynomial_condition (f : ℝ → ℝ)
  (H : ∀ x : ℝ, f(x^2 + 1) - f(x^2 - 1) = 4 * x^2 + 6) :
  ∀ x : ℝ, f(x^2 + 1) - f(x^2) = 2 * x^2 + 4 :=
sorry

end polynomial_condition_l425_425848


namespace absent_children_l425_425212

theorem absent_children (A : ℕ) (h1 : 2 * 610 = (610 - A) * 4) : A = 305 := 
by sorry

end absent_children_l425_425212


namespace downstream_distance_80_l425_425647

-- Conditions
variables (Speed_boat Speed_stream Distance_upstream : ℝ)

-- Assign given values
def speed_boat := 36 -- kmph
def speed_stream := 12 -- kmph
def distance_upstream := 40 -- km

-- Effective speeds
def speed_downstream := speed_boat + speed_stream -- kmph
def speed_upstream := speed_boat - speed_stream -- kmph

-- Downstream distance
noncomputable def distance_downstream : ℝ := 80 -- km

-- Theorem
theorem downstream_distance_80 :
  speed_boat = 36 → speed_stream = 12 → distance_upstream = 40 →
  (distance_upstream / speed_upstream = distance_downstream / speed_downstream) :=
by
  sorry

end downstream_distance_80_l425_425647


namespace pink_tulips_l425_425642

theorem pink_tulips (total_tulips : ℕ)
    (blue_ratio : ℚ) (red_ratio : ℚ)
    (h_total : total_tulips = 56)
    (h_blue_ratio : blue_ratio = 3/8)
    (h_red_ratio : red_ratio = 3/7) :
    ∃ pink_tulips : ℕ, pink_tulips = total_tulips - ((blue_ratio * total_tulips) + (red_ratio * total_tulips)) ∧ pink_tulips = 11 := by
  sorry

end pink_tulips_l425_425642


namespace isosceles_triangle_area_l425_425254

theorem isosceles_triangle_area (h : ℝ) (BC BP : ℝ) (BK : ℝ) 
  (h_eq_2Bp : h = 2 * BP) (isosceles_triangle : BK = h) :
  ∃ (S : ℝ), S = h^2 * Real.sqrt 3 :=
by
  use h^2 * Real.sqrt 3
  sorry

end isosceles_triangle_area_l425_425254


namespace max_value_sqrt_sum_l425_425935

theorem max_value_sqrt_sum (x y z : ℝ) (h_sum : x + y + z = 2)
  (h_x : x ≥ -1/2) (h_y : y ≥ -2) (h_z : z ≥ -3) :
  sqrt (6 * x + 3) + sqrt (6 * y + 12) + sqrt (6 * z + 18) ≤ 3 * sqrt 15 := 
sorry

end max_value_sqrt_sum_l425_425935


namespace sqrt_x_minus_3_defined_iff_x_geq_3_l425_425521

theorem sqrt_x_minus_3_defined_iff_x_geq_3 {x : ℝ} : (∃ y : ℝ, y = sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_x_minus_3_defined_iff_x_geq_3_l425_425521


namespace set_inter_complement_l425_425102

open Set

def U := ℤ
def M := {1, 2}
def P := {-2, -1, 0, 1, 2}

theorem set_inter_complement (U : Type) [Encodable U] [Equiv U ℤ]
  (P M : Set ℤ) : P ∩ (U \ M) = {-2, -1, 0} :=
by
  have h : P = {-2, -1, 0, 1, 2} := rfl
  have hM : M = {1, 2} := rfl
  have hcomplement : (U \ M) = {x | x ∉ {1,2}} := rfl
  have hcomp_z : {x : ℤ | x ∉ {1, 2}} = {x | ¬ (x = 1 ∨ x = 2)} := rfl
  have hintersection : {x | x ∈ P ∧ x ∉ {1,2}} = {-2, -1, 0} := by sorry
  rw [h, hM, hcomplement, hcomp_z, hintersection]

end set_inter_complement_l425_425102


namespace phil_baseball_cards_left_l425_425224

-- Step a): Define the conditions
def packs_week := 20
def weeks_year := 52
def lost_factor := 1 / 2

-- Step c): Establish the theorem statement
theorem phil_baseball_cards_left : 
  (packs_week * weeks_year * (1 - lost_factor) = 520) := 
  by
    -- proof steps will come here
    sorry

end phil_baseball_cards_left_l425_425224


namespace triangle_properties_l425_425529

-- Given conditions
def b : ℝ := 4
def c : ℝ := 5
def A : ℝ := Real.pi / 3 -- 60 degrees in radians
def a : ℝ := Real.sqrt 21
def sin2B : ℝ := (4 * Real.sqrt 3) / 7

-- Theorem statement
theorem triangle_properties : 
  let B := Real.asin (b * Real.sin A / a)
  a = Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A) ∧
  sin2B = 2 * Real.sin B * Real.cos B :=
  by
    simp only [b, c, A, a, sin2B, Real.sqrt, Real.sin, Real.cos, Real.asin],
    sorry

end triangle_properties_l425_425529


namespace probability_factor_of_5_factorial_l425_425366

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem probability_factor_of_5_factorial :
  let s := (Finset.range 40).map (λ x, x + 1)
  let favorable_outcomes := (Finset.filter (λ x, (factorial 5) % x = 0) s).card
  let total_outcomes := s.card
  favorable_outcomes / total_outcomes = 2 / 5 := 
by
  let s := (Finset.range 40).map (λ x, x + 1)
  let favorable_outcomes := (Finset.filter (λ x, (factorial 5) % x = 0) s).card
  let total_outcomes := s.card
  have h : favorable_outcomes = 16 := sorry
  have h2 : total_outcomes = 40 := sorry
  calc favorable_outcomes / total_outcomes = 16 / 40 : by rw [h, h2]
                                    ... = 2 / 5 : by norm_num

end probability_factor_of_5_factorial_l425_425366


namespace ratio_lcm_gcf_l425_425667

theorem ratio_lcm_gcf (a b : ℕ) (h₁ : a = 2^2 * 3^2 * 7) (h₂ : b = 2 * 3^2 * 5 * 7) :
  (Nat.lcm a b) / (Nat.gcd a b) = 10 := by
  sorry

end ratio_lcm_gcf_l425_425667


namespace converse_inscribed_quadrilateral_perimeter_l425_425362

    variables (A B C A' B' : Type)
    variable [metric_space A]

    def points_are_on_triangle_sides (A' : Type) (B' : Type) := 
      ∃ (A C B : Type), A' ∈ segment A C ∧ B' ∈ segment B C
    
    def perimeter_triangle (A' B' C : Type) : ℝ := sorry -- Define the perimeter calculation
    
    theorem converse_inscribed_quadrilateral_perimeter 
        (A B C A' B' : Type)
        [metric_space A]
        (h1 : points_are_on_triangle_sides A' B')
        (h2 : inscribed_quadrilateral A B A' B') : 
        perimeter_triangle A' B' C = a + b - c := 
    sorry
    
end converse_inscribed_quadrilateral_perimeter_l425_425362


namespace first_year_after_2021_with_digit_sum_5_l425_425531

def sum_of_digits (n : Nat) : Nat :=
  (toDigits 10 n).sum

theorem first_year_after_2021_with_digit_sum_5 :
  ∃ y : Nat, y > 2021 ∧ sum_of_digits y = 5 ∧ ∀ z : Nat, (z > 2021 ∧ z < y) → sum_of_digits z ≠ 5 :=
  by
    sorry

end first_year_after_2021_with_digit_sum_5_l425_425531


namespace proportional_data_points_l425_425648

theorem proportional_data_points :
  (∀ x y : ℝ, ∃ k : ℝ, y = k * x) →
  ∃ (n : ℕ), n = 2 :=
by
  intros h
  use 2
  sorry

end proportional_data_points_l425_425648


namespace value_of_ratios_l425_425916

-- Definitions of the given conditions
variable {A B C P : Type}
variable {BC CA AB DE FG HI : ℝ}
variable {a b c a' b' c' : ℝ}
variable (in_triangle : ∃ (A B C : Type), (BC = a) ∧ (CA = b) ∧ (AB = c) ∧ ∃ (P : Type), (DE = a') ∧ (FG = b') ∧ (HI = c'))

-- Statement of the problem
theorem value_of_ratios (h : in_triangle): 
  (a' / a) + (b' / b) + (c' / c) = 1 :=
sorry

end value_of_ratios_l425_425916


namespace min_tickets_to_get_lucky_l425_425108

/-- A ticket is called lucky if the sum of its first three digits equals the sum of its last three digits. -/
def is_lucky (n : ℕ) : Prop :=
  let digits := (λ d, (n / d) % 10) <$> [1, 10, 100, 1000, 10000, 100000]
  digits[0] + digits[1] + digits[2] = digits[3] + digits[4] + digits[5]

/--
  The minimum number of consecutive tickets that need to be purchased 
  from an unlimited number of tickets at a cashier to ensure obtaining at least one lucky ticket.
-/
theorem min_tickets_to_get_lucky : ∃ n, n = 1001 :=
by
  sorry

end min_tickets_to_get_lucky_l425_425108


namespace pauline_convertibles_l425_425598

noncomputable def convertibles_count (total_cars : ℕ) (percent_regular : ℝ) (percent_trucks : ℝ) : ℕ :=
  total_cars - (percent_regular * total_cars).to_nat - (percent_trucks * total_cars).to_nat

theorem pauline_convertibles :
  let total_cars := 125
  let percent_regular := 0.64
  let percent_trucks := 0.08
  convertibles_count total_cars percent_regular percent_trucks = 35 :=
by
  sorry

end pauline_convertibles_l425_425598


namespace exists_f_gcd_form_l425_425201

noncomputable def f : ℤ → ℕ := sorry

theorem exists_f_gcd_form :
  (∀ x y : ℤ, Nat.gcd (f x) (f y) = Nat.gcd (f x) (Int.natAbs (x - y))) →
  ∃ m n : ℕ, (0 < m ∧ 0 < n) ∧ (∀ x : ℤ, f x = Nat.gcd (m + Int.natAbs x) n) :=
sorry

end exists_f_gcd_form_l425_425201


namespace pauline_convertibles_l425_425599

noncomputable def convertibles_count (total_cars : ℕ) (percent_regular : ℝ) (percent_trucks : ℝ) : ℕ :=
  total_cars - (percent_regular * total_cars).to_nat - (percent_trucks * total_cars).to_nat

theorem pauline_convertibles :
  let total_cars := 125
  let percent_regular := 0.64
  let percent_trucks := 0.08
  convertibles_count total_cars percent_regular percent_trucks = 35 :=
by
  sorry

end pauline_convertibles_l425_425599


namespace line_AB_altitude_from_C_to_AB_l425_425829

-- Definition of the points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := -1, y := 5}
def B : Point := {x := -2, y := -1}
def C : Point := {x := 4, y := 3}

-- Definition of the line equation through two points
def line_equation (P Q : Point) : (ℝ × ℝ × ℝ) :=
  let a := Q.y - P.y
  let b := P.x - Q.x
  let c := P.y * Q.x - P.x * Q.y
  (a, b, c)

-- Equation of the line side AB
def line_AB_eq := (6, -1, 11)

-- Equation of the altitude to the line AB from point C
def altitude_eq_C_to_AB := (1, 6, -22)

theorem line_AB :
  line_equation A B = line_AB_eq :=
  sorry

theorem altitude_from_C_to_AB :
  let (a, b, c) := line_AB_eq
  let perpendicular_line (P : Point) : (ℝ × ℝ × ℝ) :=
    let a' := b
    let b' := -a
    let c' := -(a' * P.x + b' * P.y)
    (a', b', c')
  perpendicular_line C = altitude_eq_C_to_AB :=
  sorry

end line_AB_altitude_from_C_to_AB_l425_425829


namespace triangle_area_calculation_l425_425176

theorem triangle_area_calculation :
  ∀ (A B C D E F : Type) (mAD : ℕ) (mCE : ℕ) (ab_length : ℕ) (areaAFB : ℕ × ℕ)
  (cond1 : mAD = 20) (cond2 : mCE = 30) (cond3 : ab_length = 28)
  (cond4 : ∃ (circumcircle : Type), extend_to_circumcircle F C E circumcircle),
  areaAFB = (44, 10) → (areaAFB.fst + areaAFB.snd = 54) :=
by {
  sorry
}

end triangle_area_calculation_l425_425176


namespace right_triangle_area_l425_425164

noncomputable def area_of_triangle : ℝ := 18 * Real.sqrt 3

theorem right_triangle_area
  (hypotenuse : ℝ)
  (angle : ℝ)
  (h1 : hypotenuse = 12)
  (h2 : angle = 30) :
  ∃ A : ℝ, A = area_of_triangle :=
by
  use area_of_triangle
  simp only [area_of_triangle]
  sorry

end right_triangle_area_l425_425164


namespace part_a_part_b_l425_425723

variable (V : ℝ)

-- Basic conditions
def height := 10
def g := 10

noncomputable def time1 := (V - real.sqrt (V^2 - 200)) / 10
noncomputable def time2 := (V + real.sqrt (V^2 - 200)) / 10

-- Part A: The times for 10 meters height lie between 1 and 2 seconds
theorem part_a (h₁: height = 10) (h₂: g = 10) : 10 * real.sqrt 2 ≤ V ∧ V < 15 ↔ 
  (1 < time1 V ∧ time1 V < 2 ∧ 1 < time2 V ∧ time2 V < 2) :=
sorry

-- Part B: There is no V for which the times lie between 2 and 4 seconds
theorem part_b (h₁: height = 10) (h₂: g = 10) : ¬ (2 < time1 V ∧ time1 V < 4 ∧ 2 < time2 V ∧ time2 V < 4) :=
sorry

end part_a_part_b_l425_425723


namespace midpoint_exists_l425_425562

theorem midpoint_exists (C : set (ℝ × ℝ)) (Q : ℝ × ℝ) 
  (hC : continuous_on id C) 
  (hC_closed : is_closed C) 
  (hC_non_self_intersecting: ∀ (P1 P2 : ℝ × ℝ), P1 ∈ C → P2 ∈ C → P1 ≠ P2 → P1 ≠ P2 → id P1 ≠ P2) 
  (hQ_inside : ∃ P : ℝ × ℝ, P ∈ C ∧ Q ≠ P) :
  ∃ P1 P2 : ℝ × ℝ, P1 ∈ C ∧ P2 ∈ C ∧ Q = ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2) :=
sorry

end midpoint_exists_l425_425562


namespace hexagon_filled_correctly_l425_425782

def hexagon := Array (Array (Option ℕ))

def is_valid_hexagon (h : hexagon) : Prop :=
  -- Each number from 1 to 19 must appear exactly once
  let nums := List.filterMap id (Array.toList (Array.join h))
  nums ~ List.range' 1 19 ∧ 
  -- Each specified row sums to 38 (assuming hexagon is of proper shape)
  let rows := [
    [h[0][1], h[0][2], h[0][3]],
    [h[1][0], h[1][1], h[1][2], h[1][3]],
    [h[2][0], h[2][1], h[2][2], h[2][3], h[2][4]],
    [h[3][0], h[3][1], h[3][2], h[3][3]],
    [h[4][1], h[4][2], h[4][3]]
  ]
  rows.all (λ r, (List.sum (List.filterMap id r)) = 38)

theorem hexagon_filled_correctly : ∃ (h : hexagon), is_valid_hexagon h :=
  sorry

end hexagon_filled_correctly_l425_425782


namespace sum_of_coordinates_C_l425_425043

theorem sum_of_coordinates_C (A B D : ℝ × ℝ) (distance_condition : ℝ × ℝ → ℝ)
  (A_coord : A = (-1, 2)) (B_coord : B = (3, -6)) (D_coord : D = (7, 0))
  (distance_check : ∀ C : ℝ × ℝ, distance_condition C = 3 * distance_condition B)
  : ∀ C : ℝ × ℝ, (distance_check C) → C.1 + C.2 = 11 :=
by
  sorry

end sum_of_coordinates_C_l425_425043


namespace winter_sales_l425_425702

theorem winter_sales (T F: ℕ) (hspring hsummer hwinter: ℝ):
  F = 0.2 * T ∧ hspring = 5 ∧ hsummer = 6 ∧ hwinter = 1.1 * hsummer → 
  hwinter = 6.6 :=
by
  sorry

end winter_sales_l425_425702


namespace a_n_is_perfect_square_l425_425486

theorem a_n_is_perfect_square :
  ∀ (a b : ℕ → ℤ), a 0 = 1 → b 0 = 0 →
  (∀ n, a (n + 1) = 7 * a n + 6 * b n - 3) →
  (∀ n, b (n + 1) = 8 * a n + 7 * b n - 4) →
  ∀ n, ∃ k : ℤ, a n = k * k :=
by
  sorry

end a_n_is_perfect_square_l425_425486


namespace bike_time_to_library_l425_425508

open Nat

theorem bike_time_to_library (constant_pace : Prop)
  (time_to_park : ℕ)
  (distance_to_park : ℕ)
  (distance_to_library : ℕ)
  (h1 : constant_pace)
  (h2 : time_to_park = 30)
  (h3 : distance_to_park = 5)
  (h4 : distance_to_library = 3) :
  ∃ x : ℕ, x = 18 :=
by
  use 18
  sorry

end bike_time_to_library_l425_425508


namespace probability_p_q_prob_pq_equality_l425_425510

theorem probability_p_q (p q : ℤ) : 
  (1 ≤ p ∧ p ≤ 20) ∧ (2 * p * q - 5 * p - 3 * q = 4) ↔
  (p = 1 ∨ p = 2 ∨ p = 11) :=
sorry

theorem prob_pq_equality : 
  (finset.card {p : ℤ | 1 ≤ p ∧ p ≤ 20 ∧ ∃ q : ℤ, 2 * p * q - 5 * p - 3 * q = 4}) / 
  20 = 3 / 20 :=
sorry

end probability_p_q_prob_pq_equality_l425_425510


namespace tan_half_alpha_l425_425837

-- Definition of the given conditions
variable (α : Real) (in_fourth_quadrant : (sin α < 0) ∧ (cos α > 0))
variable (sin_cos_sum : sin α + cos α = 1/5)

-- Definition of the target statement to prove
theorem tan_half_alpha (α : Real) (in_fourth_quadrant : (sin α < 0) ∧ (cos α > 0))
(sin_cos_sum : sin α + cos α = 1/5) : tan (α / 2) = -1/3 :=
by
  sorry

end tan_half_alpha_l425_425837


namespace brocard_circle_l425_425948

theorem brocard_circle 
  (B C : (ℝ × ℝ)) (a : ℝ)
  (A : (ℝ × ℝ) → Prop)
  (φ : ℝ)
  (hBC_fixed : B = (-(a/2), 0) ∧ C = (a/2, 0))
  (hBrocard_angle : ∀ A, (A ∘ BrocardAngle A B C) = φ) :
  (A = {(x, y) : ℝ × ℝ | (x, (y - (a * cot φ / 2)))^2 = (a/2)^2 * (cot^2 φ - 3)}) := sorry

end brocard_circle_l425_425948


namespace number_of_girls_l425_425537

variable (b g d : ℕ)

-- Conditions
axiom boys_count : b = 1145
axiom difference : d = 510
axiom boys_equals_girls_plus_difference : b = g + d

-- Theorem to prove
theorem number_of_girls : g = 635 := by
  sorry

end number_of_girls_l425_425537


namespace slope_of_chord_l425_425911

theorem slope_of_chord (x y : ℝ) (h : (x^2 / 16) + (y^2 / 9) = 1) (h_midpoint : (x₁ + x₂ = 2) ∧ (y₁ + y₂ = 4)) :
  ∃ k : ℝ, k = -9 / 32 :=
by
  sorry

end slope_of_chord_l425_425911


namespace sufficient_condition_for_prop_l425_425280

theorem sufficient_condition_for_prop (a : ℝ) (e : ℝ) (h1 : ∀ x > e, a - log x < 0)
    (h2 : ∀ x > e, log x > 1) : a < 1 :=
sorry

end sufficient_condition_for_prop_l425_425280


namespace evaluate_expression_l425_425188

-- Defining the conditions and constants as per the problem statement
def factor_power_of_2 (n : ℕ) : ℕ :=
  if n % 8 = 0 then 3 else 0 -- Greatest power of 2 in 360
  
def factor_power_of_5 (n : ℕ) : ℕ :=
  if n % 5 = 0 then 1 else 0 -- Greatest power of 5 in 360

def expression (b a : ℕ) : ℚ := (2 / 3)^(b - a)

noncomputable def target_value : ℚ := 9 / 4

theorem evaluate_expression : expression (factor_power_of_5 360) (factor_power_of_2 360) = target_value := 
  by
    sorry

end evaluate_expression_l425_425188


namespace minimize_transportation_cost_l425_425640

open Real

theorem minimize_transportation_cost :
  ∃ (v : ℝ) (h : v ∈ Ioo 0 100), 
    (∀ w ∈ Ioo 0 100, (50000 / v + 5 * v) ≤ (50000 / w + 5 * w)) ∧ 
    (50000 / v + 5 * v) = 1000 :=
by
  sorry -- Proof goes here

end minimize_transportation_cost_l425_425640


namespace count_three_digit_perfect_squares_divisible_by_4_l425_425117

theorem count_three_digit_perfect_squares_divisible_by_4 : ∃ n, n = 11 ∧
  (∀ k, 100 ≤ k ^ 2 ∧ k ^ 2 ≤ 999 → k ^ 2 % 4 = 0 → (k % 2 = 0 ∧ 10 ≤ k ≤ 31)) :=
sorry

end count_three_digit_perfect_squares_divisible_by_4_l425_425117


namespace find_m_l425_425094

theorem find_m 
  (m : ℝ)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * x + 4 * y + m = 0 → 
    (x - 1)^2 + (y + 2)^2 = 5 - m)
  (line_eq : ∀ x y : ℝ, 2 * x - y - 2 = 0)
  (segment_length : (2 * Real.sqrt 5) / 5) :
  m = 4 :=
sorry

end find_m_l425_425094


namespace janet_has_five_dimes_l425_425179

theorem janet_has_five_dimes (n d q : ℕ) 
    (h1 : n + d + q = 10) 
    (h2 : d + q = 7) 
    (h3 : n + d = 8) : 
    d = 5 :=
by
  -- Proof omitted
  sorry

end janet_has_five_dimes_l425_425179


namespace problem_1_problem_2_l425_425581

-- Define the sets A, B, C
def SetA (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def SetB : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def SetC : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

-- Problem 1
theorem problem_1 (a : ℝ) : SetA a = SetB → a = 5 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (SetA a ∩ SetB).Nonempty ∧ (SetA a ∩ SetC = ∅) → a = -2 := by
  sorry

end problem_1_problem_2_l425_425581


namespace squares_on_grid_l425_425308

theorem squares_on_grid (grid : fin 3 × fin 4 → Prop) :
  (∃ f : fin 3 × fin 4 → Prop, 
    (∀ i, grid i ↔ f i) ∧
    ∑ i in finset.univ.product finset.univ, 
    grid i) = 11 := 
sorry

end squares_on_grid_l425_425308


namespace strangely_powerful_count_l425_425399

def oddly_powerful (n : ℕ) : Prop :=
  ∃ a b : ℕ, b > 1 ∧ b % 2 = 1 ∧ a^b = n

theorem strangely_powerful_count :
  { n : ℕ | oddly_powerful n ∧ n < 5000 }.toFinset.card = 23 :=
by
  sorry

end strangely_powerful_count_l425_425399


namespace tyler_saltwater_animals_l425_425314

theorem tyler_saltwater_animals :
  let type_a_animals_per_aquarium := 12 * 4 in
  let type_b_animals_per_aquarium := 18 + 10 in
  let type_c_animals_per_aquarium := 25 + 20 in
  let total_type_a_animals := 10 * type_a_animals_per_aquarium in
  let total_type_b_animals := 14 * type_b_animals_per_aquarium in
  let total_type_c_animals := 6 * type_c_animals_per_aquarium in
  total_type_a_animals + total_type_b_animals + total_type_c_animals = 1142 :=
by
  sorry

end tyler_saltwater_animals_l425_425314


namespace abs_neg_three_l425_425252

noncomputable def abs_val (a : ℤ) : ℤ :=
  if a < 0 then -a else a

theorem abs_neg_three : abs_val (-3) = 3 :=
by
  sorry

end abs_neg_three_l425_425252


namespace class_average_l425_425678

theorem class_average (n : ℕ) (h₁ : n = 100) (h₂ : 25 ≤ n) 
  (h₃ : 50 ≤ n) (h₄ : 25 * 80 + 50 * 65 + (n - 75) * 90 = 7500) :
  (25 * 80 + 50 * 65 + (n - 75) * 90) / n = 75 := 
by
  sorry

end class_average_l425_425678


namespace sum_fractions_geq_n3_div_n2_minus_1_l425_425575

variable {n : ℕ} (hn : n > 1) 
variable {x : ℕ → ℝ} (hx_pos : ∀ i, 1 ≤ i ∧ i ≤ n → x i > 0) 
variable (hx_sum : ∑ i in Finset.range n, x (i + 1) = 1) 

theorem sum_fractions_geq_n3_div_n2_minus_1 :
  (∑ i in Finset.range n, x (i + 1) / (x ((i + 1) % n + 1) - (x ((i + 1) % n + 1))^3)) 
  ≥ n^3 / (n^2 - 1) :=
sorry

end sum_fractions_geq_n3_div_n2_minus_1_l425_425575


namespace simplify_factorial_expression_l425_425239

theorem simplify_factorial_expression : 
  (15.factorial / (12.factorial + 3 * 10.factorial) = 2669) := 
  by sorry

end simplify_factorial_expression_l425_425239


namespace probability_all_1_to_5_appear_before_6_to_10_l425_425704

theorem probability_all_1_to_5_appear_before_6_to_10 :
  let roll_generator : ℕ → ℕ := sorry, -- placeholder for roll generating function
  let event_all_1_to_5_before_6_to_10 : ℕ → Prop := sorry, -- placeholder for event definition
  let probability : ℝ := sorry in -- placeholder for probability calculation
  probability = 1 / 48 :=
sorry

end probability_all_1_to_5_appear_before_6_to_10_l425_425704


namespace three_digit_perfect_squares_divisible_by_4_l425_425125

/-- Proving the number of three-digit perfect squares that are divisible by 4 is 6 -/
theorem three_digit_perfect_squares_divisible_by_4 : 
  (Finset.card (Finset.filter (λ n, n ∣ 4) (Finset.image (λ k, k * k) (Finset.Icc 10 31)))) = 6 :=
by
  sorry

end three_digit_perfect_squares_divisible_by_4_l425_425125


namespace find_a_l425_425857

noncomputable def f (x : ℝ) (a : ℝ) :=
if x >= 1 then x + a / x - 3 else log10 (x^2 + 1)

theorem find_a : ∃ a, f 1 a = f (-3) a ∧ a = 3 := by
  sorry

end find_a_l425_425857


namespace general_term_arithmetic_sequence_sum_terms_sequence_l425_425054

noncomputable def a_n (n : ℕ) : ℤ := 
  2 * (n : ℤ) - 1

theorem general_term_arithmetic_sequence :
  ∀ n : ℕ, a_n n = 2 * (n : ℤ) - 1 :=
by sorry

noncomputable def c (n : ℕ) : ℚ := 
  1 / ((2 * (n : ℤ) - 1) * (2 * (n + 1) - 1))

noncomputable def T_n (n : ℕ) : ℚ :=
  (1 / 2 : ℚ) * (1 - (1 / (2 * (n : ℤ) + 1)))

theorem sum_terms_sequence :
  ∀ n : ℕ, T_n n = (n : ℚ) / (2 * (n : ℤ) + 1) :=
by sorry

end general_term_arithmetic_sequence_sum_terms_sequence_l425_425054


namespace find_increasing_interval_l425_425018

def function_monotonically_increasing_interval 
  (f : ℝ → ℝ := λ x, Real.sin (2 * Real.pi / 3 - 2 * x)) 
  (interval : Set ℝ := Set.Icc (7 * Real.pi / 12) (13 * Real.pi / 12)) : Prop :=
  ∀ x y ∈ interval, x < y → f x < f y

theorem find_increasing_interval : 
  function_monotonically_increasing_interval := sorry

end find_increasing_interval_l425_425018


namespace expected_value_of_palindromic_substrings_floor_l425_425367

def is_palindrome (s : String) : Prop :=
  s = s.reverse

noncomputable def expected_palindromic_substrings : ℝ :=
  ∑ n in Finset.range 40, (41 - n) * 2 ^ (- (n/ 2 : ℕ))

theorem expected_value_of_palindromic_substrings_floor :
  ∃ E : ℝ, E = expected_palindromic_substrings ∧ ⌊E⌋ = 113 :=
sorry

end expected_value_of_palindromic_substrings_floor_l425_425367


namespace bags_and_candies_l425_425973

theorem bags_and_candies (f : ℕ → ℕ) (n : ℕ) :
  (Σ i in range n+1, f i) = 115 →
  (∀ i j, i < j → f i < f j) →
  (f 0 + f 1 + f 2 = 20) →
  (f (n-3) + f (n-2) + f (n-1) = 50) →
  n = 10 ∧ f 0 = 5 := 
by
  sorry

end bags_and_candies_l425_425973


namespace sum_of_prime_f_values_l425_425038

noncomputable def f (n : ℕ) := n^4 - 100 * n^2 + 576

theorem sum_of_prime_f_values :
  ∑ n in {n : ℕ | Nat.Prime (f n)}.toFinset, f n = 12505702 :=
by
  sorry

end sum_of_prime_f_values_l425_425038


namespace phil_cards_left_l425_425223

-- Conditions
def cards_per_week : ℕ := 20
def weeks_per_year : ℕ := 52

-- Total number of cards in a year
def total_cards (cards_per_week weeks_per_year : ℕ) : ℕ := cards_per_week * weeks_per_year

-- Number of cards left after losing half in fire
def cards_left (total_cards : ℕ) : ℕ := total_cards / 2

-- Theorem to prove
theorem phil_cards_left (cards_per_week weeks_per_year : ℕ) :
  cards_left (total_cards cards_per_week weeks_per_year) = 520 :=
by
  sorry

end phil_cards_left_l425_425223


namespace strangely_powerful_count_l425_425400

def oddly_powerful (n : ℕ) : Prop :=
  ∃ a b : ℕ, b > 1 ∧ b % 2 = 1 ∧ a^b = n

theorem strangely_powerful_count :
  { n : ℕ | oddly_powerful n ∧ n < 5000 }.toFinset.card = 23 :=
by
  sorry

end strangely_powerful_count_l425_425400


namespace product_of_slopes_is_constant_lambda_value_l425_425843

-- Define the given constants and variables
variables (a b c x0 y0 : ℝ) (λ : ℝ)
-- Given conditions
def is_ellipse (a b : ℝ) (x0 y0 : ℝ) := x0^2 / a^2 + y0^2 / b^2 = 1
def condition1 (a b : ℝ) : Prop := a > b ∧ b > 0

-- Point coordinates
def A : ℝ × ℝ := (-a, 0)
def B : ℝ × ℝ := (a, 0)
def F : ℝ × ℝ := (-c, 0)

-- Line slopes
def k_AP : ℝ := y0 / (x0 + a)
def k_BP : ℝ := y0 / (x0 - a)

-- First proof statement
theorem product_of_slopes_is_constant (h1 : is_ellipse a b x0 y0) (h2 : condition1 a b) : k_AP a x0 y0 * k_BP a x0 y0 = -b^2 / a^2 := sorry

-- Second proof statement
theorem lambda_value (h1 : is_ellipse a b x0 y0) (h2 : condition1 a b) (AF_eq_lambda_FB : λ := (1 : ℝ) / 3) : λ = 1/3 :=
  by sorry

end product_of_slopes_is_constant_lambda_value_l425_425843


namespace arithmetic_sequence_problem_l425_425906

theorem arithmetic_sequence_problem
  (a : ℕ → ℤ)
  (h1 : a 6 + a 9 = 16)
  (h2 : a 4 = 1)
  (h_arith : ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q) :
  a 11 = 15 :=
by
  sorry

end arithmetic_sequence_problem_l425_425906


namespace range_of_a_l425_425523

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 1| + |x - a| < 4) ↔ (-5 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l425_425523


namespace diagonal_length_l425_425282

variables (r : ℝ)
def height_cylinder := 5 * r
def volume_ratio := (20 * r^3) / (5 * π * r^3)

theorem diagonal_length (h : volume_ratio = 5 / π) : 
  sqrt (33) * r = sqrt (33) * r := 
by
  sorry

end diagonal_length_l425_425282


namespace average_of_remaining_numbers_l425_425636

theorem average_of_remaining_numbers (a : Fin 12 → ℝ) (sum_12 : ∑ i, a i = 1080) (h80 : ∃ i j, i ≠ j ∧ a i = 80 ∧ a j = 82) :
  (∑ i, a i - 80 - 82) / 10 = 91.8 :=
by
  sorry

end average_of_remaining_numbers_l425_425636


namespace impossible_configuration_l425_425177

theorem impossible_configuration :
  ∀ (points : Finset (ℝ × ℝ × ℝ)), 
  points.card = 24 ∧
  (∀ p1 p2 p3 ∈ points, p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬Collinear ℝ {p1, p2, p3}) →
  ∀ planes : Finset { s : Finset (ℝ × ℝ × ℝ) // s.card ≥ 3 },
  planes.card = 2013 ∧ 
  (∀ p1 p2 p3 ∈ points, ∃ plane ∈ planes, {p1, p2, p3} ⊆ plane.1) →
  False :=
by
  sorry

end impossible_configuration_l425_425177


namespace smallest_possible_value_l425_425326

theorem smallest_possible_value (a b c d : ℤ) 
  (h1 : a + b + c + d < 25) 
  (h2 : a > 8) 
  (h3 : b < 5) 
  (h4 : c % 2 = 1) 
  (h5 : d % 2 = 0) : 
  ∃ a' b' c' d' : ℤ, a' > 8 ∧ b' < 5 ∧ c' % 2 = 1 ∧ d' % 2 = 0 ∧ a' + b' + c' + d' < 25 ∧ (a' - b' + c' - d' = -4) := 
by 
  use 9, 4, 1, 10
  sorry

end smallest_possible_value_l425_425326


namespace problem_part1_problem_part2_l425_425052

-- Define the sequence and its properties for part (1)
def sequence_a (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < n → a (2 * n + 1) = a (2 * n - 1) + 4 ∧ a (2 * n) = a (2 * n - 1) + 8

def S (S_n : ℕ → ℕ) (a : ℕ → ℕ) : ℕ := S_n

theorem problem_part1 (a : ℕ → ℕ) (S_n : ℕ → ℕ) (n : ℕ) (h1 : ∀ k : ℕ, 0 < k → a (2 * k + 1) = a (2 * k - 1) + 4 ∧ a (2 * k) = a (2 * k - 1) + 8)
  (h2 : a 1 = 1) : 
  S_n (2 * n) = 4 * n^2 + 6 * n :=
sorry

-- Define the sequence and its properties for part (2)
def is_geometric_seq (S : ℕ → ℕ) (a : ℕ → ℕ) (q : ℕ) (a_const : ℕ) : Prop :=
  ∀ n : ℕ, S n / a n + a_const = (a_const + 1) * q ^ (n - 1)

theorem problem_part2 (a : ℕ → ℕ) (S : ℕ → ℕ) (q : ℕ) (a_const : ℕ)
  (h1 : q ≠ -1)
  (h2 : is_geometric_seq S a q a_const) :
  (∀ n : ℕ, 0 < n →  a (n + 1) / a n = 1 / q) ↔ q = 1 + 1 / a_const :=
sorry

end problem_part1_problem_part2_l425_425052


namespace simplify_vectors_l425_425977

variable (α : Type*) [AddCommGroup α]

variables (CE AC DE AD : α)

theorem simplify_vectors : CE + AC - DE - AD = (0 : α) := 
by sorry

end simplify_vectors_l425_425977


namespace vasya_wins_l425_425961

noncomputable def player_has_winning_strategy (player: string) : Prop :=
  if player = "Vasya" then true else false

theorem vasya_wins :
  player_has_winning_strategy "Vasya" :=
by
  -- Conditions
  -- Petya and Vasya are playing a game on a strip of 9 cells
  -- Petya starts first
  -- Each turn, a player writes any digit in any free cell
  -- The resulting number can start with one or more zeros
  -- Petya wins if the final number is a perfect square; otherwise, Vasya wins
  sorry

end vasya_wins_l425_425961


namespace solve_for_x_l425_425033

theorem solve_for_x : ∃ x : ℤ, x + 1 = 5 ∧ x = 4 :=
by
  sorry

end solve_for_x_l425_425033


namespace existence_of_solution_l425_425803

variable (n : ℕ) (x : ℝ) (x_i : Fin n → ℝ) (a : ℝ)

theorem existence_of_solution
  (h : ∀ i, 0 ≤ x_i i ∧ x_i i ≤ 3)
  (hx : 0 ≤ x ∧ x ≤ 3) :
  (∑ i, (| x - x_i i |) = a * n) ↔ (a = 3 / 2) :=
by sorry

end existence_of_solution_l425_425803


namespace proof_g_f_neg3_l425_425572

def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

-- Assuming g is a function that maps the results accordingly
axiom g : ℝ → ℝ

-- Given conditions
axiom h1 : g (f 3) = 12

theorem proof_g_f_neg3 : g (f (-3)) = 12 :=
by
  have f_3_eq : f 3 = 10 := by { simp [f], norm_num }
  have g_10_eq : g 10 = 12 := by { rw ← f_3_eq at h1, exact h1 }
  have f_neg3_eq : f (-3) = 28 := by { simp [f], norm_num }
  -- Hypothetical proof or assumption that g(28) = 12 based on provided solution logic
  exact sorry

end proof_g_f_neg3_l425_425572


namespace possible_values_of_a_l425_425483

variable {R : Type} [linear_ordered_field R]

noncomputable def system_of_equations (a b c d e : R) :=
  a * b + a * c + a * d + a * e = -1 ∧
  b * c + b * d + b * e + b * a = -1 ∧
  c * d + c * e + c * a + c * b = -1 ∧
  d * e + d * a + d * b + d * c = -1 ∧
  e * a + e * b + e * c + e * d = -1

theorem possible_values_of_a :
  ∀ {a b c d e : R}, system_of_equations a b c d e → 
  (a = ↑(sqrt 2) / 2 ∨ a = -↑(sqrt 2) / 2 ∨ a = ↑(sqrt 2) ∨ a = -↑(sqrt 2)) :=
by sorry

end possible_values_of_a_l425_425483


namespace measure_angle_APB_minor_arc_l425_425476

noncomputable def measure_angle (A B P : Point) : ℝ := sorry

theorem measure_angle_APB_minor_arc (A B P : Point)
  (h1 : ∃ (m : ℝ), (∀ (x y : ℝ), (y = m - x ↔ y = sqrt 3 - x)))
  (h2 : ∃ (r : ℝ), (r = sqrt 2 ∧ ∀ (x y : ℝ), x^2 + y^2 = r^2)) :
  measure_angle A B P = π / 6 :=
by sorry

end measure_angle_APB_minor_arc_l425_425476


namespace problem1_problem2_l425_425481

variables (a : ℝ) (x : ℝ)

-- Proof Problem 1
theorem problem1 (hp : ∀ x : ℝ, a * x^2 + a * x + 1 > 0) : 0 ≤ a ∧ a < 4 := 
sorry

-- Proof Problem 2
theorem problem2 (hp : ∀ x : ℝ, a * x^2 + a * x + 1 > 0 ∨ abs (2 * a - 1) < 3)
  (hnp : ¬ (∀ x : ℝ, a * x^2 + a * x + 1 > 0 ∧ abs (2 * a - 1) < 3)) :
  a ∈ set.Ioo (-1) 0 ∪ set.Icc 2 4 :=
sorry

end problem1_problem2_l425_425481


namespace trig_identity_eq_one_l425_425323

theorem trig_identity_eq_one :
  (Real.sin (160 * Real.pi / 180) + Real.sin (40 * Real.pi / 180)) *
  (Real.sin (140 * Real.pi / 180) + Real.sin (20 * Real.pi / 180)) +
  (Real.sin (50 * Real.pi / 180) - Real.sin (70 * Real.pi / 180)) *
  (Real.sin (130 * Real.pi / 180) - Real.sin (110 * Real.pi / 180)) =
  1 :=
sorry

end trig_identity_eq_one_l425_425323


namespace prism_volume_eq_400_l425_425633

noncomputable def prism_volume (a b c : ℝ) : ℝ := a * b * c

theorem prism_volume_eq_400 
  (a b c : ℝ)
  (h1 : a * b = 40)
  (h2 : a * c = 50)
  (h3 : b * c = 80) :
  prism_volume a b c = 400 :=
by
  sorry

end prism_volume_eq_400_l425_425633


namespace solve_for_x_l425_425775

theorem solve_for_x (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (h_neq : m ≠ n) :
  ∃ x : ℝ, (x + 2 * m)^2 - (x - 3 * n)^2 = 9 * (m + n)^2 ↔
  x = (5 * m^2 + 18 * m * n + 18 * n^2) / (10 * m + 6 * n) := sorry

end solve_for_x_l425_425775


namespace framed_painting_ratio_l425_425712

def painting_width := 20
def painting_height := 30

def smaller_dimension := painting_width + 2 * 5
def larger_dimension := painting_height + 4 * 5

noncomputable def ratio := (smaller_dimension : ℚ) / (larger_dimension : ℚ)

theorem framed_painting_ratio :
  ratio = 3 / 5 :=
by
  sorry

end framed_painting_ratio_l425_425712


namespace roots_of_g_on_unit_circle_l425_425200

theorem roots_of_g_on_unit_circle {n : ℕ} (a : Fin (n + 1) → ℂ) (λ : ℂ) :
  a 0 ≠ 0 → a n ≠ 0 →
  (∀ z : ℂ, z ∈ Polynomial.roots (Polynomial.mk a) → Complex.abs z < 1) →
  Complex.abs λ = 1 →
  ∀ z : ℂ, z ∈ Polynomial.roots (Polynomial.mk (λ k, a k + λ * (Complex.conj (a (n - k))))) → Complex.abs z = 1 :=
by
  sorry

end roots_of_g_on_unit_circle_l425_425200


namespace calc_diagonal_of_rectangle_l425_425840

variable (a : ℕ) (A : ℕ)

theorem calc_diagonal_of_rectangle (h_a : a = 6) (h_A : A = 48) (H : a * a' = A) :
  ∃ d : ℕ, d = 10 :=
by
 sorry

end calc_diagonal_of_rectangle_l425_425840


namespace find_x_plus_y_l425_425075

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 2023) 
                           (h2 : x + 2023 * Real.sin y = 2022) 
                           (h3 : (Real.pi / 2) ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2023 + Real.pi / 2 :=
sorry

end find_x_plus_y_l425_425075


namespace m_values_for_product_of_linear_trinomials_l425_425019

-- Define the polynomial
def p (x y z m : ℂ) := x^3 + y^3 + z^3 + m * x * y * z

-- Helper definition: Check if a polynomial can be written as a product of three linear trinomials
def is_product_of_linear (p : ℂ → ℂ → ℂ → ℂ) : Prop :=
  ∃ a b c d e f g h i : ℂ, p = λ x y z, (x + a*y + b*z) * (x + c*y + d*z) * (x + e*y + f*z)

-- The theorem we need to prove
theorem m_values_for_product_of_linear_trinomials :
  ∀ m : ℂ, (is_product_of_linear (p _ _ _ m)) ↔ (m = -3 ∨ m = -3 * (complex.root_unity 3 1) ∨ m = -3 * (complex.root_unity 3 2)) :=
by sorry

end m_values_for_product_of_linear_trinomials_l425_425019


namespace equilateral_triangle_product_l425_425277

theorem equilateral_triangle_product (c d : ℝ) (h : ((0, 0) : ℝ × ℝ), (c, 14) : ℝ × ℝ), ((d, 47) : ℝ × ℝ) are the vertices of an equilateral triangle.) :
  c * d = 1520 / 3 := 
sorry

end equilateral_triangle_product_l425_425277


namespace water_loss_per_jump_l425_425558

def pool_capacity : ℕ := 2000 -- in liters
def jump_limit : ℕ := 1000
def clean_threshold : ℝ := 0.80

theorem water_loss_per_jump :
  (pool_capacity * (1 - clean_threshold)) * 1000 / jump_limit = 400 :=
by
  -- We prove that the water lost per jump in mL is 400
  sorry

end water_loss_per_jump_l425_425558


namespace initial_amount_l425_425728

def pie_cost : Real := 6.75
def juice_cost : Real := 2.50
def gift : Real := 10.00
def mary_final : Real := 52.00

theorem initial_amount (M : Real) :
  M = mary_final + pie_cost + juice_cost + gift :=
by
  sorry

end initial_amount_l425_425728


namespace abs_neg_three_l425_425253

noncomputable def abs_val (a : ℤ) : ℤ :=
  if a < 0 then -a else a

theorem abs_neg_three : abs_val (-3) = 3 :=
by
  sorry

end abs_neg_three_l425_425253


namespace min_value_of_expression_l425_425832

theorem min_value_of_expression (a b : ℝ) (h : 2 * a + 3 * b = 4) : 4^a + 8^b ≥ 8 :=
begin
  sorry,
end

end min_value_of_expression_l425_425832


namespace trapezoid_proof_l425_425654

noncomputable def EQ (EF FG GH EH : ℕ) (h_parallel : EF = 130 ∧ GH = 34 ∧ EH = 100 ∧ FG = 72) : ℕ :=
  let p := 14400
  let q := 229
  p + q

theorem trapezoid_proof :
  ∀ (EF FG GH EH : ℕ),
  EF = 130 →
  FG = 72 →
  GH = 34 →
  EH = 100 →
  ∃ (EQ : ℚ), EQ = 14400 / 229 ∧ ∃ (p q : ℕ), EQ = p / q ∧ Nat.coprime p q ∧ p + q = 14629 :=
by
  sorry

end trapezoid_proof_l425_425654


namespace f2018_is_odd_l425_425999

/- 
Given the functions f1 and fn defined recursively as:
- f1(x) = 1/x
- fn+1(x) = 1/(x + fn(x))
Prove that f2018(x) is an odd function.
-/

def f1 (x : ℝ) : ℝ := 1 / x

def fn : ℕ → (ℝ → ℝ)
| 0 := f1
| (n+1) := λ x, 1 / (x + fn n x)

theorem f2018_is_odd : ∀ (x : ℝ), fn 2018 (-x) = - (fn 2018 x) :=
by
  sorry

end f2018_is_odd_l425_425999


namespace snow_probability_l425_425215

theorem snow_probability :
  let p1 := 1/2 in
  let p2 := 1/3 in
  let no_snow_first_five_days := (1 - p1)^5 in
  let no_snow_next_five_days := (1 - p2)^5 in
  let no_snow_ten_days := no_snow_first_five_days * no_snow_next_five_days in
  let snow_at_least_once := 1 - no_snow_ten_days in
  snow_at_least_once = 242 / 243 :=
by sorry

end snow_probability_l425_425215


namespace width_of_pool_l425_425370

-- Definitions based on the conditions
def pool_length : ℝ := 20
def total_area : ℝ := 728
def deck_width : ℝ := 3

-- The theorem we need to prove
theorem width_of_pool :
  let W := 12.43 in
  pool_length * W + 2 * deck_width * pool_length + deck_width ^ 2 = total_area :=
by
  sorry

end width_of_pool_l425_425370


namespace no_x_arcsin_sq_plus_arccos_sq_eq_one_l425_425776

open Real

theorem no_x_arcsin_sq_plus_arccos_sq_eq_one :
  ¬∃ x ∈ Icc (-1 : ℝ) 1, (arcsin x) ^ 2 + (arccos x) ^ 2 = 1 := 
by {
    -- Proof is omitted as this template requires only the statement.
    sorry
}

end no_x_arcsin_sq_plus_arccos_sq_eq_one_l425_425776


namespace symm_wrt_circumcenter_of_BCD_AH_square_eq_4h₁h₂_l425_425897

variables {V : Type} [inner_product_space ℝ V] [finite_dimensional ℝ V]

-- Define regular tetrahedron ABCD
variables {A B C D H H₁ : V}

-- Conditions
def tetrahedron.is_regular (A B C D : V) : Prop :=
  distance A B = distance B C ∧ distance B C = distance C D ∧ distance C D = distance A D ∧
  distance A C = distance B D

def point.altitude (A H : V) (Δ : set V) : Prop :=
  H ∈ Δ ∧ ∀ P ∈ Δ, ⟪A - H, P - H⟫ = 0

variables {h₁ h₂ : ℝ}

def h₁_h₂_property (h₁ h₂ : ℝ) (H₁ : V) (Δ : set V) : Prop :=
  ∃ h₁' h₂', h₁' = h₁ ∧ h₂' = h₂ ∧ ∃ P Q ∈ Δ, distance P H₁ = h₁ ∧ distance Q H₁ = h₂

-- Lean statements for the problems
theorem symm_wrt_circumcenter_of_BCD (A B C D H H₁ : V) (O₁ : V) 
  (h₁_property : h₁_h₂_property h₁ h₂ H₁ {B, C, D}) :
  tetrahedron.is_regular A B C D →
  point.altitude A H {B, C, D} →
  (∀ O, O ∈ {B, C, D} → distance O₁ O = distance O₁ (Oᵢ_Center)) →
  reflect_about_circumcenter_of_triangle {B, C, D} H H₁ :=
sorry

theorem AH_square_eq_4h₁h₂ (A B C D H H₁ : V)
  (h₁_h₂ : h₁_h₂_property h₁ h₂ H₁ {B, C, D}) :
  tetrahedron.is_regular A B C D →
  point.altitude A H {B, C, D} →
  ∥A - H∥^2 = 4 * h₁ * h₂ :=
sorry

end symm_wrt_circumcenter_of_BCD_AH_square_eq_4h₁h₂_l425_425897


namespace find_x2_plus_y2_l425_425556

open Real

theorem find_x2_plus_y2 (x y : ℝ) 
  (h1 : (x + y) ^ 4 + (x - y) ^ 4 = 4112)
  (h2 : x ^ 2 - y ^ 2 = 16) :
  x ^ 2 + y ^ 2 = 34 := 
sorry

end find_x2_plus_y2_l425_425556


namespace prime_divisors_unique_l425_425608

theorem prime_divisors_unique
  (n : ℕ) :
  ∃ (p : ℕ → ℕ), (∀ k, (1 ≤ k ∧ k ≤ n) → (nat.prime (p k))) ∧
    ∀ k₁ k₂, (1 ≤ k₁ ∧ k₁ ≤ n) ∧ (1 ≤ k₂ ∧ k₂ ≤ n) ∧ (k₁ ≠ k₂) → (p k₁ ∣ (n! + k₁)) ∧¬(p k₁ ∣ (n! + k₂)) :=
sorry

end prime_divisors_unique_l425_425608


namespace initial_amount_l425_425682

theorem initial_amount (X : ℝ) (h1 : 0.70 * X = 2800) : X = 4000 :=
by
  sorry

end initial_amount_l425_425682


namespace smallest_positive_value_of_sum_of_pairwise_products_l425_425822

-- Define the conditions: 95 numbers being either +1 or -1
def is_valid_values (a : ℕ → ℤ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ 95 → a i = 1 ∨ a i = -1

-- Define the sum of pairwise products
def sum_of_pairwise_products (a : ℕ → ℤ) : ℤ :=
  (Finset.range 95).sum (λ i, (Finset.range 95).sum (λ j, ite (i < j) (a i * a j) 0))

-- Main problem statement
theorem smallest_positive_value_of_sum_of_pairwise_products :
  ∃ a : ℕ → ℤ, is_valid_values a ∧ sum_of_pairwise_products a = 13 :=
sorry

end smallest_positive_value_of_sum_of_pairwise_products_l425_425822


namespace problem_1_problem_2_problem_3_l425_425074

noncomputable def f (x : ℝ) : ℝ := (4^x - 1) / 2^x

def g (x b : ℝ) : ℝ := (x - b) * (x - 2 * b)

def h (x b : ℝ) : ℝ :=
  if x < 1 then f x - b
  else g x b

theorem problem_1 : f x = 2^x - 2^(-x) :=
sorry

theorem problem_2 (t k : ℝ) (h_t : t ≥ 0) (h_f : f (t^2 - 2 * t) + f (2 * t^2 - k) > 0) : k < -1/3 :=
sorry

theorem problem_3 (b : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ h x1 b = 0 ∧ h x2 b = 0) ↔ (1/2 < b ∧ b < 1) ∨ (b ≥ 3/2) :=
sorry

end problem_1_problem_2_problem_3_l425_425074


namespace tangent_inequality_l425_425231

theorem tangent_inequality (n : ℕ) (h : n ≥ 2) 
  (α : Fin n → ℝ) 
  (h0 : ∀ i : Fin n, 0 < α i ∧ α i < (π / 2)) 
  (h_ord : ∀ i j : Fin n, i < j → α i < α j) :
  tan (α 0) < (∑ i, sin (α i)) / (∑ i, cos (α i)) ∧ (∑ i, sin (α i)) / (∑ i, cos (α i)) < tan (α (n - 1)) :=
begin
  sorry
end

end tangent_inequality_l425_425231


namespace luke_fish_catching_l425_425778

theorem luke_fish_catching :
  ∀ (days : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ),
  days = 30 → fillets_per_fish = 2 → total_fillets = 120 →
  (total_fillets / fillets_per_fish) / days = 2 :=
by
  intros days fillets_per_fish total_fillets days_eq fillets_eq fillets_total_eq
  sorry

end luke_fish_catching_l425_425778


namespace range_a_A_intersect_B_empty_range_a_A_union_B_eq_B_l425_425099

-- Definition of the sets A and B
def A (a : ℝ) (x : ℝ) : Prop := a - 1 < x ∧ x < 2 * a + 1
def B (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Proving range of a for A ∩ B = ∅
theorem range_a_A_intersect_B_empty (a : ℝ) :
  (¬ ∃ x : ℝ, A a x ∧ B x) ↔ (a ≤ -2 ∨ a ≥ 2 ∨ (-2 < a ∧ a ≤ -1/2)) := sorry

-- Proving range of a for A ∪ B = B
theorem range_a_A_union_B_eq_B (a : ℝ) :
  (∀ x : ℝ, A a x ∨ B x → B x) ↔ (a ≤ -2) := sorry

end range_a_A_intersect_B_empty_range_a_A_union_B_eq_B_l425_425099


namespace count_three_digit_perfect_squares_divisible_by_4_l425_425121

theorem count_three_digit_perfect_squares_divisible_by_4 : ∃ n, n = 11 ∧
  (∀ k, 100 ≤ k ^ 2 ∧ k ^ 2 ≤ 999 → k ^ 2 % 4 = 0 → (k % 2 = 0 ∧ 10 ≤ k ≤ 31)) :=
sorry

end count_three_digit_perfect_squares_divisible_by_4_l425_425121


namespace repeating_decimals_sum_l425_425013

theorem repeating_decimals_sum : 
  (0.3333333333333333 : ℝ) + (0.0404040404040404 : ℝ) + (0.005005005005005 : ℝ) = (14 / 37 : ℝ) :=
by {
  sorry
}

end repeating_decimals_sum_l425_425013


namespace min_value_xy_l425_425060

theorem min_value_xy (x y : ℕ) (h : 0 < x ∧ 0 < y) (cond : (1 : ℚ) / x + (1 : ℚ) /(3 * y) = 1 / 6) : 
  xy = 192 :=
sorry

end min_value_xy_l425_425060


namespace curve_polar_eq_and_area_l425_425453

noncomputable def parametric_curve (α : ℝ) : ℝ × ℝ :=
  (3 + 5 * Real.cos α, 4 + 5 * Real.sin α)

noncomputable def polar_coordinates_A : ℝ × ℝ := (4 + 3 * Real.sqrt 3, Real.pi / 6)
noncomputable def polar_coordinates_B : ℝ × ℝ := (8, Real.pi / 2)

theorem curve_polar_eq_and_area :
  (∀ (θ : ℝ), ∃ (ρ : ℝ), polar_coordinates_A = (ρ, θ) ∧
    (3 + 5 * Real.cos θ - 3) ^ 2 + (4 + 5 * Real.sin θ - 4) ^ 2 = 25)
  ∧ (∀ (θ : ℝ), ∃ (ρ : ℝ), polar_coordinates_B = (ρ, θ) ∧
    (3 + 5 * Real.cos θ - 3) ^ 2 + (4 + 5 * Real.sin θ - 4) ^ 2 = 25)
  ∧ (∀ (θ : ℝ), ∃ (ρ : ℝ), 
    ρ = 6 * Real.cos θ + 8 * Real.sin θ)
  ∧ (∀ (θ₁ θ₂ : ℝ) (ρ₁ ρ₂ : ℝ),
     polar_coordinates_A = (ρ₁, θ₁) → polar_coordinates_B = (ρ₂, θ₂) →
     let dist_AB := Real.sqrt ((4 + 3 * Real.sqrt 3) ^ 2 + 64 - 2 * (4 + 3 * Real.sqrt 3) * 8 * (1 / 2)) in
     let M := (3 : ℝ, 4 : ℝ) in
     let d_MAB := Real.sqrt (25 - (Real.sqrt 3 * 5 / 2) ^ 2) in
     let area_MAB := (1 / 2) * (d_MAB) * dist_AB in
     area_MAB = 25 * Real.sqrt 3 / 4) :=
sorry

end curve_polar_eq_and_area_l425_425453


namespace triangle_congruence_l425_425539

theorem triangle_congruence (x y : ℕ) (h1 : x = 6) (h2 : y = 5) : x + y = 11 :=
by
  rw [h1, h2]
  rfl

end triangle_congruence_l425_425539


namespace evaluate_expression_l425_425618

theorem evaluate_expression : 
  let a := 2
  let b := 1 / 2
  2 * (a^2 - 2 * a * b) - 3 * (a^2 - a * b - 4 * b^2) = -2 :=
by
  let a := 2
  let b := 1 / 2
  sorry

end evaluate_expression_l425_425618


namespace problem_l425_425819

noncomputable def f (ω x : Real) : Real := (Real.cos (ω * x + Real.pi / 3))

theorem problem 
  (ω : Real) 
  (hω : ω > 0) : 
  (∀ T, (T = 2) → f (Real.pi) = Real.cos (Real.pi * x + Real.pi / 3)) ∧
  (f 2 = Real.cos (2 * x + Real.pi / 3)) ∧ 
  (∀ x, (x ∈ Set.Ioo (2 * Real.pi / 3) Real.pi) → (0 < f' x)) ∧
  (Set.count(Filter.(fun x => (0 < x) ∧ (x < Real.pi) ∧ (f x = 0)) = 1)) :=
sorry

end problem_l425_425819


namespace angle_is_pi_over_3_l425_425106

variable (BA BC : ℝ × ℝ)
variable (θ : ℝ)

def vectors_BA_BC := BA = (sqrt 3 / 2, 1 / 2) ∧ BC = (0, 1)

def angle_between_vectors := cos θ = (BA.1 * BC.1 + BA.2 * BC.2) / ((real.sqrt (BA.1 ^ 2 + BA.2 ^ 2)) * (real.sqrt (BC.1 ^ 2 + BC.2 ^ 2)))

theorem angle_is_pi_over_3 (h : vectors_BA_BC BA BC) : θ = real.pi / 3 :=
by
  sorry

end angle_is_pi_over_3_l425_425106


namespace solve_equation_2021_l425_425624

theorem solve_equation_2021 (x : ℝ) (hx : 0 ≤ x) : 
  2021 * x = 2022 * (x ^ (2021 : ℕ)) ^ (1 / (2021 : ℕ)) - 1 → x = 1 := 
by
  sorry

end solve_equation_2021_l425_425624


namespace function_even_function_extrema_l425_425263

noncomputable def f (x : ℝ) : ℝ := Math.cos (2 * x) + Math.sin (x + Real.pi / 2)

theorem function_even : ∀ x : ℝ, f x = f (-x) :=
sorry

theorem function_extrema : ∃ a b : ℝ, ∀ x : ℝ, f x ≤ a ∧ f x ≥ b :=
sorry

end function_even_function_extrema_l425_425263


namespace find_a8_l425_425525

variable {a : ℕ → ℤ}

def is_geometric (s : ℕ → ℤ) (q : ℤ) : Prop :=
  ∀ n, s (n + 1) = s n * q

theorem find_a8
  (hq : ∃ q : ℤ, is_geometric (λ n, a n + 2) q)
  (h2 : a 2 = -1)
  (h4 : a 4 = 2) :
  a 8 = 62 :=
sorry

end find_a8_l425_425525


namespace walnuts_left_in_burrow_l425_425696

-- Define the initial quantities
def boy_initial_walnuts : Nat := 6
def boy_dropped_walnuts : Nat := 1
def initial_burrow_walnuts : Nat := 12
def girl_added_walnuts : Nat := 5
def girl_eaten_walnuts : Nat := 2

-- Define the resulting quantity and the proof goal
theorem walnuts_left_in_burrow : boy_initial_walnuts - boy_dropped_walnuts + initial_burrow_walnuts + girl_added_walnuts - girl_eaten_walnuts = 20 :=
by
  sorry

end walnuts_left_in_burrow_l425_425696


namespace angle_AMC_120_deg_l425_425216

-- Define the main variables and conditions
variables (A B C L K M : Point) (ABC : Triangle A B C)
  [IsEquilateral ABC]
  [L_on_AB : LiesOn L (Segment A B)]
  [K_on_BC : LiesOn K (Segment B C)]
  [M_is_intersection : Intersection M (Segment A K) (Segment C L)]
  [area_eq : Area (Triangle A M C) = Area (Quadrilateral L B K M)]

-- Formal statement of the problem in Lean 4
theorem angle_AMC_120_deg :
  ∠ A M C = 120 :=
sorry

end angle_AMC_120_deg_l425_425216


namespace miriam_pushups_ratio_l425_425952

theorem miriam_pushups_ratio :
  (∃ W : ℕ,
    let monday := 5,
        tuesday := 7,
        wednesday := W,
        thursday := (monday + tuesday + wednesday) / 2,
        total_pushups := monday + tuesday + wednesday + thursday
    in total_pushups = 39 ∧ W / 7 = 2) :=
begin
  sorry
end

end miriam_pushups_ratio_l425_425952


namespace repeating_sum_to_fraction_l425_425008

theorem repeating_sum_to_fraction :
  (0.333333333333333 ~ 1/3) ∧ 
  (0.0404040404040401 ~ 4/99) ∧ 
  (0.005005005005001 ~ 5/999) →
  (0.333333333333333 + 0.0404040404040401 + 0.005005005005001) = (112386 / 296703) := 
by
  repeat { sorry }

end repeating_sum_to_fraction_l425_425008


namespace area_quadrilateral_ABCD_l425_425467

noncomputable def circle_equation : ℝ → ℝ → ℝ := λ x y, x^2 + y^2 - 4 * x + 2 * y
noncomputable def point_E : (ℝ × ℝ) := (1, 0)
noncomputable def center := (2, -1)
noncomputable def radius := real.sqrt 5
noncomputable def diameter := 2 * (real.sqrt 5)
noncomputable def shortest_chord_length := 2 * (real.sqrt 3)

theorem area_quadrilateral_ABCD :
  let E := point_E in
  let r := radius in
  let d := diameter in
  let s := shortest_chord_length in
  (∃ (A B C D : ℝ × ℝ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ 
    B ≠ C ∧ B ≠ D ∧ 
    C ≠ D ∧
    circle_equation (A.1) (A.2) = 0 ∧
    circle_equation (B.1) (B.2) = 0 ∧
    circle_equation (C.1) (C.2) = 0 ∧
    C.1 - A.1 = d ∧
    -- Other necessary conditions capturing perpendicularity and shortest chord properties
    -- Here, we assume the chords intersect E as required by additional conditions
    true
  ) →
  let area := 1 / 2 * s * d in
  area = 2 * real.sqrt 15 :=
sorry

end area_quadrilateral_ABCD_l425_425467


namespace cost_for_3300_pens_l425_425711

noncomputable def cost_per_pack (pack_cost : ℝ) (num_pens_per_pack : ℕ) : ℝ :=
  pack_cost / num_pens_per_pack

noncomputable def total_cost (cost_per_pen : ℝ) (num_pens : ℕ) : ℝ :=
  cost_per_pen * num_pens

theorem cost_for_3300_pens (pack_cost : ℝ) (num_pens_per_pack num_pens : ℕ) (h_pack_cost : pack_cost = 45) (h_num_pens_per_pack : num_pens_per_pack = 150) (h_num_pens : num_pens = 3300) :
  total_cost (cost_per_pack pack_cost num_pens_per_pack) num_pens = 990 :=
  by
    sorry

end cost_for_3300_pens_l425_425711


namespace mark_final_friends_correct_l425_425949

def mark_initial_friends := 100

def cycle_1_keep_fraction := 0.4
def cycle_1_contact_fraction := 0.6
def cycle_1_response_fraction := 0.5

def cycle_2_keep_fraction := 0.6
def cycle_2_contact_fraction := 0.4
def cycle_2_response_fraction := 0.7

def cycle_3_keep_fraction := 0.8
def cycle_3_contact_fraction := 0.2
def cycle_3_response_fraction := 0.4

def final_friends_remaining (init_friends : ℕ) : ℕ :=
  let cycle_1_kept := (init_friends * cycle_1_keep_fraction).toInt
  let cycle_1_contacted := (init_friends * cycle_1_contact_fraction).toInt
  let cycle_1_responded := (cycle_1_contacted * cycle_1_response_fraction).toInt
  let friends_after_cycle_1 := cycle_1_kept + cycle_1_responded
  let cycle_2_kept := (friends_after_cycle_1 * cycle_2_keep_fraction).toInt
  let cycle_2_contacted := (friends_after_cycle_1 * cycle_2_contact_fraction).toInt
  let cycle_2_responded := (cycle_2_contacted * cycle_2_response_fraction).toInt
  let friends_after_cycle_2 := cycle_2_kept + cycle_2_responded
  let cycle_3_kept := (friends_after_cycle_2 * cycle_3_keep_fraction).toInt
  let cycle_3_contacted := (friends_after_cycle_2 * cycle_3_contact_fraction).toInt
  let cycle_3_responded := (cycle_3_contacted * cycle_3_response_fraction).toInt
  cycle_3_kept + cycle_3_responded

theorem mark_final_friends_correct : final_friends_remaining mark_initial_friends = 52 :=
  by sorry

end mark_final_friends_correct_l425_425949


namespace which_condition_does_not_indicate_similarity_l425_425555

noncomputable def angle_B : ℝ := 90
noncomputable def angle_B' : ℝ := 90
noncomputable def angle_A : ℝ := 30

theorem which_condition_does_not_indicate_similarity :
  ¬(∃ angle_C : ℝ, angle_C = 60 → (angle_B = angle_B' ∧ angle_A = 30 ∧ similarity_condition angle_C)) :=
sorry

end which_condition_does_not_indicate_similarity_l425_425555


namespace solution_set_of_inequality_l425_425290

theorem solution_set_of_inequality :
  { x : ℝ | (x - 2) / (x + 3) ≥ 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x ≥ 2 } := 
sorry

end solution_set_of_inequality_l425_425290


namespace find_a_l425_425083

theorem find_a (a : ℝ) : (∃ x y : ℝ, 3 * x + a * y - 5 = 0 ∧ x = 1 ∧ y = 2) → a = 1 :=
by
  intro h
  match h with
  | ⟨x, y, hx, hx1, hy2⟩ => 
    have h1 : x = 1 := hx1
    have h2 : y = 2 := hy2
    rw [h1, h2] at hx
    sorry

end find_a_l425_425083


namespace lines_LH_MG_meet_at_Γ_l425_425940

open EuclideanGeometry

variables {A B C D E G H L M : Point} {Γ : Circle}

-- Conditions
axiom circumcircle : is_circumcircle ABC Γ
axiom passing_through_A_C_circle (α : Circle) : passes_through α {A, C}
axiom meets_BC_BA (α : Circle) (D : Point) (side_BC : between B C D) (E : Point) (side_BA : between B A E) : true
axiom AD_meets_Γ_at_G : meets_on AD Γ G
axiom CE_meets_Γ_at_H : meets_on CE Γ H
axiom tangent_meets_DE_at_L_M (L_tangent : is_tangent Γ A L) (M_tangent : is_tangent Γ C M) (meets_DE : meets_on L M DE) : true

-- Proof
theorem lines_LH_MG_meet_at_Γ :
  ∃ P : Point, is_on P Γ ∧ (meets_on (line L H) (line M G) P) :=
sorry

end lines_LH_MG_meet_at_Γ_l425_425940


namespace exist_congruent_triangles_l425_425649

-- Definitions of necessary structures and conditions
structure CubeVertex where
  -- A vertex representation; could be any unique type
  index : Fin 8 

def initial_positions : Set CubeVertex := { v : CubeVertex | v.index.val < 8 }
def final_positions : Set CubeVertex := { v : CubeVertex | v.index.val < 8 }

-- Statement of the problem
theorem exist_congruent_triangles : 
  ∃ f1 f2 f3 f4 f5 f6 : CubeVertex,
    f1 ∈ initial_positions ∧ f2 ∈ initial_positions ∧ f3 ∈ initial_positions ∧ 
    f4 ∈ final_positions ∧ f5 ∈ final_positions ∧ f6 ∈ final_positions ∧ 
    congruent_triangle f1 f2 f3 f4 f5 f6 := 
  sorry

end exist_congruent_triangles_l425_425649


namespace shortest_chord_intercepted_length_of_circle_l425_425984

theorem shortest_chord_intercepted_length_of_circle :
  let C := (1, 2)
  let O := (0, 0)
  let r := 3
  let d := real.sqrt (1^2 + 2^2)
  C = (1, 2) →
  ∃ l : ℝ → ℝ, ∃ x y: ℝ,
  (∀ x y, x^2 + y^2 - 2*x - 4*y + 1 = 0) ∧ 
  (real.sqrt (x^2 + y^2) = d) →
  2 * real.sqrt (r^2 - d^2) = 4 :=
by {
    intros C O r d hC hex,
    sorry
}

end shortest_chord_intercepted_length_of_circle_l425_425984


namespace least_possible_xy_l425_425064

theorem least_possible_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 48 :=
by
  sorry

end least_possible_xy_l425_425064


namespace find_angles_DAE_DAO_l425_425551

open_locale classical

variables {A B C D O E : Type} [EuclideanGeometry A B C D O E]

noncomputable def angle_ACB : ℝ := 60
noncomputable def angle_CBA : ℝ := 70

-- Given conditions
axiom ACB_is_60_degrees : EuclideanGeometry.angle A C B = angle_ACB
axiom CBA_is_70_degrees : EuclideanGeometry.angle C B A = angle_CBA
axiom D_is_foot_of_perpendicular : EuclideanGeometry.is_perpendicular_foot A D B C
axiom O_is_circumcenter : EuclideanGeometry.is_circumcenter O A B C
axiom E_is_other_end_of_diameter : EuclideanGeometry.is_diameter_end O A E

-- To be proved
theorem find_angles_DAE_DAO :
  EuclideanGeometry.angle D A E = 10 ∧ EuclideanGeometry.angle D A O = 20 := by
  sorry

end find_angles_DAE_DAO_l425_425551


namespace sqrt_identity_trig_l425_425418

theorem sqrt_identity_trig :
  (sqrt (1 - 2 * sin (π + 2) * cos (π - 2)) = sin 2 - cos 2) :=
sorry

end sqrt_identity_trig_l425_425418


namespace three_digit_perfect_squares_divisible_by_4_count_l425_425131

theorem three_digit_perfect_squares_divisible_by_4_count : 
  (finset.filter (λ n, (100 ≤ n ∧ n ≤ 999) ∧ (∃ k, n = k * k) ∧ (n % 4 = 0)) (finset.range 1000)).card = 10 :=
by
  sorry

end three_digit_perfect_squares_divisible_by_4_count_l425_425131


namespace maximum_value_of_f_l425_425816

noncomputable def f (a : ℝ) := ∫ x in 0..1, (2 * a * x^2 - a^2 * x)

theorem maximum_value_of_f : (∃ a : ℝ, f a = ∫ x in 0..1, (2 * a * x^2 - a^2 * x) ∧ f a = 2 / 9) :=
sorry

end maximum_value_of_f_l425_425816


namespace nth_derivative_zero_l425_425186

theorem nth_derivative_zero {f : ℝ → ℝ} {n : ℕ} (h_diff : ∀ (k : ℕ), k ≤ n + 1 → Differentiable ℝ (f^[k]))
    (h0 : f 0 = 0) (h1 : f 1 = 0) (h_deriv_at_0 : ∀ (k : ℕ), k ≤ n → (f^[k] 0 = 0)) :
  ∃ x ∈ (0, 1), f^[n+1] x = 0 := 
by
  -- The proof is omitted.
  sorry

end nth_derivative_zero_l425_425186


namespace perimeter_proof_l425_425534

noncomputable def perimeter (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ (Real.sqrt 3) / 3 then 3 * Real.sqrt 6 * x
  else if x > (Real.sqrt 3) / 3 ∧ x ≤ (2 * Real.sqrt 3) / 3 then 3 * Real.sqrt 2
  else if x > (2 * Real.sqrt 3) / 3 ∧ x ≤ Real.sqrt 3 then 3 * Real.sqrt 6 * (Real.sqrt 3 - x)
  else 0

theorem perimeter_proof (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ Real.sqrt 3) :
  perimeter x = 
    if x ≤ (Real.sqrt 3) / 3 then 3 * Real.sqrt 6 * x
    else if x ≤ (2 * Real.sqrt 3) / 3 then 3 * Real.sqrt 2
    else 3 * Real.sqrt 6 * (Real.sqrt 3 - x) :=
by 
  sorry

end perimeter_proof_l425_425534


namespace not_commutative_no_identity_elem_special_computation_l425_425987

-- Definition of the binary operation
def star (x y : ℝ) := (x + 2) * (y + 1) - 3

-- Problem statements
theorem not_commutative : ∀ x y : ℝ, star x y ≠ star y x := by
  intro x y
  unfold star
  -- Expanded steps and additional proof details would follow
  sorry

theorem no_identity_elem : ¬∃ e : ℝ, ∀ x : ℝ, star x e = x := by
  intro h
  cases h with e he
  -- Expanded steps and additional proof details would follow
  sorry

theorem special_computation : star 0 1 = 1 := by
  unfold star
  -- Computation details
  calc (0 + 2) * (1 + 1) - 3 = 4 - 3 := by rfl
  ... = 1 := by rfl

end not_commutative_no_identity_elem_special_computation_l425_425987


namespace exist_boy_danced_no_more_than_half_l425_425340

theorem exist_boy_danced_no_more_than_half :
  ∀ (boys : Fin 100) (girls : Fin 100) 
  (pair_danced : Fin 100 → Fin 100 → Bool)
  (H : ∀ g : Fin 100, ∃ s₁ s₂ : Finset (Fin 100), (s₁.disjoint s₂) ∧ (s₁ ∪ s₂ = Finset.univ) ∧ 
  (Finset.sum s₁ (λ b, if pair_danced b g then 1 else 0) = Finset.sum s₂ (λ b, if pair_danced b g then 1 else 0))), 
  ∃ k : Fin 100, (Finset.sum (Finset.univ.filter (λ g, pair_danced k g)) (λ g, 1)) / 2 ≥ Finset.sum (Finset.univ.filter (λ g, pair_danced k g)) (λ g, if g = k then 1 else 0) :=
sorry

end exist_boy_danced_no_more_than_half_l425_425340


namespace product_of_slopes_l425_425491

theorem product_of_slopes (k : ℝ) :
  let l₁ := k
  let l₂ := -k
  let P := (1 : ℝ, 1 : ℝ)
  let circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = 4 }
  let chord_len_ratio := (2 * real.sqrt (4 - (k - 1)^2 / (k^2 + 1))) / (2 * real.sqrt (4 - (k + 1)^2 / (k^2 + 1)))
  (l₁ * l₂ = -9 ∨ l₁ * l₂ = -(1/9)) :=
begin
  sorry,
end

end product_of_slopes_l425_425491


namespace base_6_units_digit_l425_425256

def num1 : ℕ := 217
def num2 : ℕ := 45
def base : ℕ := 6

theorem base_6_units_digit :
  (num1 % base) * (num2 % base) % base = (num1 * num2) % base :=
by
  sorry

end base_6_units_digit_l425_425256


namespace sqrt_subtraction_eq_l425_425393

theorem sqrt_subtraction_eq : sqrt 8 - sqrt 2 = sqrt 2 :=
by 
  sorry

end sqrt_subtraction_eq_l425_425393


namespace Monroe_granola_bars_l425_425953

theorem Monroe_granola_bars (children_count : ℕ) (granola_per_child : ℕ) (parent_consumption : ℕ) 
                             (h1 : children_count = 6) (h2 : granola_per_child = 20) (h3 : parent_consumption = 80) :
                             (children_count * granola_per_child + parent_consumption = 200) :=
by
  rw [h1, h2, h3]
  norm_num

end Monroe_granola_bars_l425_425953


namespace three_digit_perfect_squares_divisible_by_4_count_l425_425132

theorem three_digit_perfect_squares_divisible_by_4_count : 
  (finset.filter (λ n, (100 ≤ n ∧ n ≤ 999) ∧ (∃ k, n = k * k) ∧ (n % 4 = 0)) (finset.range 1000)).card = 10 :=
by
  sorry

end three_digit_perfect_squares_divisible_by_4_count_l425_425132


namespace object_speed_approx_l425_425679

def distance_feet : ℝ := 300
def time_seconds : ℝ := 6
def feet_to_miles : ℝ := 1 / 5280
def seconds_to_hours : ℝ := 1 / 3600

noncomputable def speed_mph : ℝ :=
  (distance_feet * feet_to_miles) / (time_seconds * seconds_to_hours)

theorem object_speed_approx :
  |speed_mph - 34.091| < 0.001 :=
by
  sorry

end object_speed_approx_l425_425679


namespace max_coins_Martha_can_take_l425_425377

/-- 
  Suppose a total of 2010 coins are distributed in 5 boxes with quantities 
  initially forming consecutive natural numbers. Martha can perform a 
  transformation where she takes one coin from a box with at least 4 coins and 
  distributes one coin to each of the other boxes. Prove that the maximum number 
  of coins that Martha can take away is 2004.
-/
theorem max_coins_Martha_can_take : 
  ∃ (a : ℕ), 2010 = a + (a+1) + (a+2) + (a+3) + (a+4) ∧ 
  ∀ (f : ℕ → ℕ) (h : (∃ b ≥ 4, f b = 400 + b)), 
  (∃ n : ℕ, f n = 4) → (∃ n : ℕ, f n = 3) → 
  (∃ n : ℕ, f n = 2) → (∃ n : ℕ, f n = 1) → 
  (∃ m : ℕ, f m = 2004) := 
by
  sorry

end max_coins_Martha_can_take_l425_425377


namespace distance_between_countries_l425_425960

theorem distance_between_countries (total_distance : ℕ) (spain_germany : ℕ) (spain_other : ℕ) :
  total_distance = 7019 →
  spain_germany = 1615 →
  spain_other = total_distance - spain_germany →
  spain_other = 5404 :=
by
  intros h_total_distance h_spain_germany h_spain_other
  rw [h_total_distance, h_spain_germany] at h_spain_other
  exact h_spain_other

end distance_between_countries_l425_425960


namespace difference_of_squares_l425_425611

theorem difference_of_squares (n : ℕ) : (n+1)^2 - n^2 = 2*n + 1 :=
by
  sorry

end difference_of_squares_l425_425611


namespace alpha_value_l425_425046

theorem alpha_value (α : ℝ) (h : 0 ≤ α ∧ α ≤ 2 * Real.pi 
    ∧ ∃β : ℝ, β = 2 * Real.pi / 3 ∧ (Real.sin β, Real.cos β) = (Real.sin α, Real.cos α)) : 
    α = 5 * Real.pi / 3 := 
  by
    sorry

end alpha_value_l425_425046


namespace radius_of_circle_with_center_on_line_and_passing_through_points_l425_425031

theorem radius_of_circle_with_center_on_line_and_passing_through_points : 
  (∃ a b : ℝ, 2 * a + b = 0 ∧ 
              (a - 1) ^ 2 + (b - 3) ^ 2 = r ^ 2 ∧ 
              (a - 4) ^ 2 + (b - 2) ^ 2 = r ^ 2 
              → r = 5) := 
by 
  sorry

end radius_of_circle_with_center_on_line_and_passing_through_points_l425_425031


namespace sum_of_squared_residuals_is_zero_l425_425811

-- Condition: Set of observation data
variables {n : ℕ} {x y : ℕ → ℝ} (h_data : ∀ i : ℕ, i < n → y i = (1 / 3) * (x i) + 2)

-- Define the sum of squared residuals
def sum_squared_residuals (y_actual y_pred : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finRange n, (y_actual i - y_pred i) ^ 2

-- Problem statement: Prove that the sum of squared residuals is 0
theorem sum_of_squared_residuals_is_zero : sum_squared_residuals y (λ i, (1 / 3) * (x i) + 2) n = 0 :=
sorry

end sum_of_squared_residuals_is_zero_l425_425811


namespace polynomial_simplify_l425_425976

theorem polynomial_simplify (x : ℝ) :
  (2*x^5 + 3*x^3 - 5*x^2 + 8*x - 6) + (-6*x^5 + x^3 + 4*x^2 - 8*x + 7) = -4*x^5 + 4*x^3 - x^2 + 1 :=
  sorry

end polynomial_simplify_l425_425976


namespace max_value_of_expression_equality_at_max_l425_425793

theorem max_value_of_expression : ∀ x : ℝ, (0 ≤ x ∧ x ≤ 16) → (sqrt (x + 64) + sqrt (16 - x) + 2 * sqrt x ≤ 4 * sqrt 5 + 8) :=
by
  intros x hx
  sorry

theorem equality_at_max : ∀ x : ℝ, x = 16 → sqrt (x + 64) + sqrt (16 - x) + 2 * sqrt x = 4 * sqrt 5 + 8 :=
by
  intros x hx
  sorry

end max_value_of_expression_equality_at_max_l425_425793


namespace sum_f_powers_of_2_equals_55_l425_425833

open Real

-- Definition of f given by the condition
noncomputable def f (x : ℝ) : ℝ := log x / log 2

-- The theorem statement to be proved
theorem sum_f_powers_of_2_equals_55 : 
  f (3^x) = x * log 3 / log 2 → (∑ i in finset.range 10, f (2 ^ (i + 1))) = 55 :=
by
  intro h
  sorry

end sum_f_powers_of_2_equals_55_l425_425833


namespace volleyball_qualification_l425_425162

theorem volleyball_qualification :
  ∃ N : ℕ, (∀ (t : Finset (Fin 15))) (H : t.card ≥ 7, ∃ n ∈ t, n ≤ N) ↔ N = 3 := 
sorry

end volleyball_qualification_l425_425162


namespace find_B_and_cos_C_l425_425530

variable {A B C a b c : ℝ}
variable (h1 : B = Real.pi / 3)
variable (h2 : tan (A + Real.pi / 4) = 7)
variable (h3 : cos C / cos B = (2 * a - c) / b)

theorem find_B_and_cos_C 
  (h_in_triangle : 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi)
  (h_sum_angles : A + B + C = Real.pi) :
  B = Real.pi / 3 ∧ cos C = (3 * Real.sqrt 3 - 4) / 10 := by
  sorry

end find_B_and_cos_C_l425_425530


namespace Tammy_second_day_speed_l425_425248

variable (v t : ℝ)

/-- This statement represents Tammy's climbing situation -/
theorem Tammy_second_day_speed:
  (t + (t - 2) = 14) ∧
  (v * t + (v + 0.5) * (t - 2) = 52) →
  (v + 0.5 = 4) :=
by
  sorry

end Tammy_second_day_speed_l425_425248


namespace mean_variance_transformed_set_l425_425828

variable (n : ℕ) (x : ℕ → ℝ) (mean variance : ℝ)
variable (mean_def : mean = ((Finset.range n).sum x) / n)
variable (variance_def : variance = ((Finset.range n).sum (λ i, (x i - mean) ^ 2)) / n)

theorem mean_variance_transformed_set : 
  mean ((λ i, 4 * (x i) + 3)) n = 4 * mean + 3 ∧ variance ((λ i, 4 * (x i) + 3)) n = 16 * variance := 
by
  sorry

end mean_variance_transformed_set_l425_425828


namespace solve_for_x_l425_425622

theorem solve_for_x (x : ℝ) (h : (1 / 8) ^ (3 * x + 12) = 32 ^ (3 * x + 7)) : 
  x = -71 / 24 :=
sorry

end solve_for_x_l425_425622


namespace erika_flips_coin_probability_l425_425422

theorem erika_flips_coin_probability:
  let coin := {12, 24}
  let die := {1, 2, 3, 4, 5, 6}
  let p_coin_12 := (1 : ℚ) / 2
  let p_die_4 := (1 : ℚ) / 6
  let p_coin_24 := (1 : ℚ) / 2
  let p_die_2 := (1 : ℚ) / 6
  let p := p_coin_12 * p_die_4 + p_coin_24 * p_die_2
  in p = 1 / 6 := 
sorry

end erika_flips_coin_probability_l425_425422


namespace find_m_value_l425_425885

noncomputable def m_value (x : ℤ) (m : ℝ) : Prop :=
  3 * (x + 1) - 2 ≤ 4 * (x - 3) + 1 ∧
  (∃ x, x ≥ 12 ∧ (1 / 2 : ℝ) * x - m = 5)

theorem find_m_value : ∃ m : ℝ, ∀ x : ℤ, m_value x m → m = 1 :=
by
  sorry

end find_m_value_l425_425885


namespace b_5_is_2_cos_x_l425_425760

-- Define the arithmetic sequence based on the given conditions
def b (n : ℕ) : ℝ :=
  if n = 1 then sin x else
  if n = 2 then cos x else
  if n = 4 then sin x + cos x else
  sorry -- we fill in details of arithmetic sequence later if needed

-- Define the common difference
def d := cos x - sin x

-- Theorem statement
theorem b_5_is_2_cos_x (x : ℝ) 
  (h1 : b 1 = sin x)
  (h2 : b 2 = cos x)
  (h4 : b 4 = sin x + cos x) :
  b 5 = 2 * cos x :=
sorry

end b_5_is_2_cos_x_l425_425760


namespace find_constants_for_B_l425_425560
open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2, 4], ![2, 0, 2], ![4, 2, 0]]

def I3 : Matrix (Fin 3) (Fin 3) ℝ := 1

def zeros : Matrix (Fin 3) (Fin 3) ℝ := 0

theorem find_constants_for_B : 
  ∃ (s t u : ℝ), s = 0 ∧ t = -36 ∧ u = -48 ∧ (B^3 + s • B^2 + t • B + u • I3 = zeros) :=
sorry

end find_constants_for_B_l425_425560


namespace rectangle_area_intersection_l425_425583

/-- Let rectangle ABCD be defined with vertices A = (0,0), B = (0,5), C = (8,5), and D = (8,0). 
    Draw lines from A at 45 degrees and from B at -45 degrees with the horizontal. 
    Verify the area of the rectangle and the coordinates of the intersection point of the lines from A and B. -/
theorem rectangle_area_intersection :
  let A := (0 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 5 : ℝ)
  let C := (8 : ℝ, 5 : ℝ)
  let D := (8 : ℝ, 0 : ℝ)
  let lineA : (ℝ × ℝ) → Prop := λ p, p.2 = p.1 -- y = x
  let lineB : (ℝ × ℝ) → Prop := λ p, p.2 = 5 - p.1 -- y = 5 - x
  ∃ area : ℝ, area = 40 ∧ (∃ (x y : ℝ), lineA (x, y) ∧ lineB (x, y) ∧ x = 2.5 ∧ y = 2.5) :=
sorry

end rectangle_area_intersection_l425_425583


namespace initial_trees_correct_l425_425731

noncomputable def initial_trees : ℝ :=
  let T := 400 in
  let trees_cut := 0.20 * T in
  let trees_planted := 5 * trees_cut in
  let final_trees := T - trees_cut + trees_planted in
  if final_trees = 720 then T else 0

theorem initial_trees_correct :
  initial_trees = 400 :=
by
  let T := 400
  let trees_cut := 0.20 * T
  let trees_planted := 5 * trees_cut
  let final_trees := T - trees_cut + trees_planted
  have h : final_trees = 720 := by
    calc 
      final_trees = T - (0.20 * T) + (5 * (0.20 * T)) : by rfl
      _ = T - 0.20 * T + T : by ring
      _ = 1.80 * T : by ring
      _ = 720 : by norm_num
  exact if_pos h

-- Sorry is used as we assert the theorem based on the previous problem statement and derivations without providing intermediate steps.

end initial_trees_correct_l425_425731


namespace sum_of_base_b_numbers_l425_425585

theorem sum_of_base_b_numbers :
  ∀ (b : ℕ), 
    (2 * b + 1) * (2 * b + 5) * (2 * b + 6) = 7 * b^3 + 5 * b^2 + 3 * b + 6 →
    ((2 * b + 1) + (2 * b + 5) + (2 * b + 6)) = 74 := 
by {
  assume b h,
  sorry
}

end sum_of_base_b_numbers_l425_425585


namespace largest_reciprocal_l425_425671

theorem largest_reciprocal :
  let options := [1/7, 3/4, 2, 8, 100] in
  let reciprocals := options.map (fun x => 1/x) in
  (1/(7:ℝ)) ∈ options ∧ 7 ∈ reciprocals ∧ ∀ (r ∈ reciprocals), 7 ≥ r := by
  let options := [1/7, 3/4, 2, 8, 100];
  let reciprocals := options.map (fun x => 1/x);
  sorry

end largest_reciprocal_l425_425671


namespace math_problem_l425_425471

open Set

noncomputable def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℤ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- Define a theorem that matches the mathematically equivalent problem
theorem math_problem (a : ℝ) :
  (Aᶜ ∩ B) = ({7, 8, 9} : Set ℝ) ∧ (A ∪ C a = univ → 3 ≤ a ∧ a < 6) :=
  by sorry

end math_problem_l425_425471


namespace candidate_X_expected_win_percentage_l425_425158

theorem candidate_X_expected_win_percentage :
  ∀ (R : ℝ), let repub := 3 * R in
             let dem := 2 * R in
             let votes_X := 0.75 * repub + 0.15 * dem in
             let votes_Y := (1 - 0.75) * repub + (1 - 0.15) * dem in
             let total_votes := votes_X + votes_Y in
             (votes_X / total_votes) * 100 - (votes_Y / total_votes) * 100 = 2 :=
by
  intro R
  let repub := 3 * R
  let dem := 2 * R
  let votes_X := 0.75 * repub + 0.15 * dem
  let votes_Y := (1 - 0.75) * repub + (1 - 0.15) * dem
  let total_votes := votes_X + votes_Y
  have h1 : (votes_X / total_votes) * 100 = 51 := by sorry
  have h2 : (votes_Y / total_votes) * 100 = 49 := by sorry
  exact h1 - h2

end candidate_X_expected_win_percentage_l425_425158


namespace positive_integer_fraction_l425_425812

theorem positive_integer_fraction (x : ℝ) (h : x ≠ 0) : 
  (0 < x) ↔ (∃ k : ℤ, k > 0 ∧ (|x - 3 * |x||) / x = (k : ℚ)) := by 
sorry

end positive_integer_fraction_l425_425812


namespace weight_of_5_moles_H₂CO₃_l425_425666

-- Definitions based on the given conditions
def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

def num_H₂CO₃_H : ℕ := 2
def num_H₂CO₃_C : ℕ := 1
def num_H₂CO₃_O : ℕ := 3

def molecular_weight (num_H num_C num_O : ℕ) 
                     (weight_H weight_C weight_O : ℝ) : ℝ :=
  num_H * weight_H + num_C * weight_C + num_O * weight_O

-- Main proof statement
theorem weight_of_5_moles_H₂CO₃ :
  5 * molecular_weight num_H₂CO₃_H num_H₂CO₃_C num_H₂CO₃_O 
                       atomic_weight_H atomic_weight_C atomic_weight_O 
  = 310.12 := by
  sorry

end weight_of_5_moles_H₂CO₃_l425_425666


namespace solution_set_of_abs_fraction_eq_fraction_l425_425645

-- Problem Statement
theorem solution_set_of_abs_fraction_eq_fraction :
  { x : ℝ | |x / (x - 1)| = x / (x - 1) } = { x : ℝ | x ≤ 0 ∨ x > 1 } :=
by
  sorry

end solution_set_of_abs_fraction_eq_fraction_l425_425645


namespace no_three_distinct_rational_roots_l425_425580

theorem no_three_distinct_rational_roots (a b : ℝ) : 
  ¬ ∃ (u v w : ℚ), 
    u + v + w = -(2 * a + 1) ∧ 
    u * v + v * w + w * u = (2 * a^2 + 2 * a - 3) ∧ 
    u * v * w = b := sorry

end no_three_distinct_rational_roots_l425_425580


namespace standard_equation_of_ellipse_line_passes_through_fixed_point_l425_425086

-- Definition and conditions
def ellipse (x y a b : ℝ) : Prop := (x^2)/(a^2) + (y^2)/(b^2) = 1
def passes_through (x y : ℝ) : Prop := ellipse x y 2 3/2  -- ellipse with specific point
def eccentricity (a e : ℝ) : Prop := e = 1 / 2 ∧ a > 0

-- Statement of the theorem
theorem standard_equation_of_ellipse (a b : ℝ) (x y : ℝ) (e : ℝ) 
    (h1: passes_through 1 (3/2)) (h2: a > b > 0) (h3: eccentricity a e) : 
    ellipse x y 2 3 := sorry

-- Line intersecting conditions
def line_eq (k m x y : ℝ) : Prop := y = k * x + m
def line_intersects_ellipse (k m : ℝ) : Prop := ∃ x y, line_eq k m x y ∧ ellipse x y 2 3

-- Statement for fixed point of intersection
theorem line_passes_through_fixed_point (k m : ℝ)
    (h_line: line_intersects_ellipse k m)
    (h_dot_product_zero: ∀ A B D : ℝ × ℝ, ellipse A.1 A.2 2 3 ∧ ellipse B.1 B.2 2 3 ∧ (D.1 = 2) →
        (D.1 - A.1) * (D.1 - B.1) + (D.2 - A.2) * (D.2 - B.2) = 0) :
    line_eq k m (2 / 7) 0 := sorry

end standard_equation_of_ellipse_line_passes_through_fixed_point_l425_425086


namespace find_a_l425_425855

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 * (f'' a)

theorem find_a (a : ℝ) [Differentiable ℝ (fun x : ℝ => f a x)] (h : f a 1 = -1) :
  a = 1 :=
sorry

end find_a_l425_425855


namespace odd_function_decreasing_l425_425034

-- Define the function f(x)
def f (x θ : ℝ) := sin (2 * x + θ) + sqrt 3 * cos (2 * x + θ)

-- Define the interval
def interval := set.Icc (-π / 4) 0

-- Define the condition for odd function
theorem odd_function_decreasing (θ : ℝ) (h : θ = 2 * π / 3) : 
  ∀ x ∈ interval, deriv (f x θ) < 0 :=
begin
  sorry
end

end odd_function_decreasing_l425_425034


namespace committees_common_members_l425_425904

theorem committees_common_members
  (num_delegates : ℕ)
  (num_committees : ℕ)
  (committee_size : ℕ)
  (delegates_in_committees : finset (finset ℕ))
  (h_num_delegates : num_delegates = 1600)
  (h_num_committees : num_committees = 16000)
  (h_committee_size : committee_size = 80)
  (h_committees_size : ∀ C ∈ delegates_in_committees, C.card = committee_size) :
  ∃ C1 C2 ∈ delegates_in_committees, C1 ≠ C2 ∧ (C1 ∩ C2).card ≥ 4 := by
sorry

end committees_common_members_l425_425904


namespace relationship_sets_l425_425197

def M : Set := {rhombus}
def N : Set := {parallelogram}
def P : Set := {quadrilateral}
def Q : Set := {square}

theorem relationship_sets : Q ⊆ M ∧ M ⊆ N ∧ N ⊆ P :=
by
  sorry

end relationship_sets_l425_425197


namespace g_has_zeros_in_any_open_subinterval_l425_425187

open Function

theorem g_has_zeros_in_any_open_subinterval
  (g : ℝ → ℝ)
  (h1 : ∀ x ∈ Icc (2013:ℝ) 2014, ∀ y ∈ Icc (2013:ℝ) 2014, g((x + y) / 2) ≤ g(x) + g(y))
  (h2 : g 2013 = 0)
  (h3 : g 2014 = 0) :
  ∀ c d, 2013 < c → c < d → d < 2014 → ∃ z, z ∈ Ioo c d ∧ g z = 0 := 
by sorry

end g_has_zeros_in_any_open_subinterval_l425_425187


namespace total_renovation_time_l425_425653

theorem total_renovation_time :
  let bedroom_count := 3
  let bedroom_time := 4
  let kitchen_time := bedroom_time + (0.5 * bedroom_time)
  let bedroom_total_time := bedroom_count * bedroom_time
  let combined_non_living_room_time := bedroom_total_time + kitchen_time
  let living_room_time := 2 * combined_non_living_room_time
  bedroom_total_time + kitchen_time + living_room_time = 54 :=
by
  let bedroom_count := 3
  let bedroom_time := 4
  let kitchen_time := bedroom_time + (0.5 * bedroom_time)
  let bedroom_total_time := bedroom_count * bedroom_time
  let combined_non_living_room_time := bedroom_total_time + kitchen_time
  let living_room_time := 2 * combined_non_living_room_time
  show bedroom_total_time + kitchen_time + living_room_time = 54
  sorry

end total_renovation_time_l425_425653


namespace triangle_inequality_l425_425957

variables {K : Type*} [LinearOrderedField K]
variables (A B C S M : K)
variables (SA SB SC : K)
variables (p q r : K)
# enabling noncomputable logic if required
noncomputable theory

-- conditions
def barycentric_coordinates (M : K) (A B C : K) (p q r : K) : Prop :=
  M = p * A + q * B + r * C ∧ p + q + r = 1

def parallel_edges (A B C S M : K) (A₁ B₁ C₁ : K) : Prop :=
  ∃ (p q r : K), M = p * A + q * B + r * C ∧
  A₁ = SA * p ∧ B₁ = SB * q ∧ C₁ = SC * r

-- Theorem statement
theorem triangle_inequality
  (SABC: Prop)
  (A₁ B₁ C₁ : K)
  (M_on_base : barycentric_coordinates M A B C p q r)
  (par_edges: parallel_edges A B C S M A₁ B₁ C₁)
  (h1 : MA₁ = SA * p)
  (h2 : MB₁ = SB * q)
  (h3 : MC₁ = SC * r)
  : (real.sqrt (SA * p) + real.sqrt (SB * q) + real.sqrt (SC * r) ≤ real.sqrt (SA + SB + SC)) :=
sorry

end triangle_inequality_l425_425957


namespace f_800_value_l425_425191

theorem f_800_value (f : ℝ → ℝ) (f_condition : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y) (f_400 : f 400 = 4) : f 800 = 2 :=
  sorry

end f_800_value_l425_425191


namespace find_m_n_find_a_l425_425482

def quadratic_roots (x : ℝ) (m n : ℝ) : Prop := 
  x^2 + m * x - 3 = 0

theorem find_m_n {m n : ℝ} : 
  quadratic_roots (-1) m n ∧ quadratic_roots n m n → 
  m = -2 ∧ n = 3 := 
sorry

def f (x m : ℝ) : ℝ := 
  x^2 + m * x - 3

theorem find_a {a m : ℝ} (h : m = -2) : 
  f 3 m = f (2 * a - 3) m → 
  a = 1 ∨ a = 3 := 
sorry

end find_m_n_find_a_l425_425482


namespace incenter_divides_segment_l425_425632

variables (A B C I M : Type) (R r : ℝ)

-- Definitions based on conditions
def is_incenter (I : Type) (A B C : Type) : Prop := sorry
def is_circumcircle (C : Type) : Prop := sorry
def angle_bisector_intersects_at (A B C M : Type) : Prop := sorry
def divides_segment (I M : Type) (a b : ℝ) : Prop := sorry

-- Proof problem statement
theorem incenter_divides_segment (h1 : is_circumcircle C)
                                   (h2 : is_incenter I A B C)
                                   (h3 : angle_bisector_intersects_at A B C M)
                                   (h4 : divides_segment I M a b) :
  a * b = 2 * R * r :=
sorry

end incenter_divides_segment_l425_425632


namespace min_value_expression_l425_425456

theorem min_value_expression (a b m n : ℝ) 
    (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
    (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
    (h_sum_one : a + b = 1) 
    (h_prod_two : m * n = 2) :
    (a * m + b * n) * (b * m + a * n) = 2 :=
sorry

end min_value_expression_l425_425456


namespace possible_outcomes_table_tennis_match_l425_425311

theorem possible_outcomes_table_tennis_match : ∃ n : ℕ, (n = 20 ∧ 
  ∀ seq : list (fin 2), 
  (seq.length ≥ 3 ∧
   (seq.count 0 = 3 ∨ seq.count 1 = 3) ∧ 
   (seq.count 0 = 3 → seq.take_while (fun x => x = 0) ≠ [0, 0, 0]) ∧
   (seq.count 1 = 3 → seq.take_while (fun x => x = 1) ≠ [1, 1, 1]) 
  → seq.length - 1 - seq.index_of_last (3))
) sorry

end possible_outcomes_table_tennis_match_l425_425311


namespace cot_13pi_over_4_l425_425749

noncomputable def cot (x : ℝ) : ℝ := 1 / Real.tan(x)

theorem cot_13pi_over_4 : cot (13 * Real.pi / 4) = -1 :=
by
  sorry

end cot_13pi_over_4_l425_425749


namespace recipient_wins_all_l425_425538

-- Define the tournament players and their relationships
universe u
variable {α : Type u} [Fintype α] [DecidableEq α]

-- Define a structure for the tournament
structure Tournament (P : α → α → Prop) :=
  (complete : ∀ (x y : α), x ≠ y → P x y ∨ P y x)
  (asym : ∀ (x y : α), P x y → ¬ P y x)

-- Define the prize condition
def prize_condition (P : α → α → Prop) (prize : α) : Prop :=
  ∀ (z : α), P z prize →
    ∃ (y : α), P prize y ∧ P y z

-- Define the main theorem
theorem recipient_wins_all {P : α → α → Prop} (T : Tournament P) (prize : α) 
  (h : prize_condition P prize) 
  (unique_prize : ∀ (x : α), prize_condition P x → x = prize) : 
  ∀ (x : α), x ≠ prize → P prize x :=
begin
  intros x hx,
  by_contra hcontra,
  sorry -- This is where the proof would go
end

end recipient_wins_all_l425_425538


namespace fraction_of_teeth_removed_l425_425041

theorem fraction_of_teeth_removed
  (total_teeth : ℕ)
  (initial_teeth : ℕ)
  (second_fraction : ℚ)
  (third_fraction : ℚ)
  (second_removed : ℕ)
  (third_removed : ℕ)
  (fourth_removed : ℕ)
  (total_removed : ℕ)
  (first_removed : ℕ)
  (fraction_first_removed : ℚ) :
  total_teeth = 32 →
  initial_teeth = 32 →
  second_fraction = 3 / 8 →
  third_fraction = 1 / 2 →
  second_removed = 12 →
  third_removed = 16 →
  fourth_removed = 4 →
  total_removed = 40 →
  first_removed + second_removed + third_removed + fourth_removed = total_removed →
  first_removed = 8 →
  fraction_first_removed = first_removed / initial_teeth →
  fraction_first_removed = 1 / 4 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end fraction_of_teeth_removed_l425_425041


namespace three_digit_perfect_squares_divisible_by_4_l425_425124

/-- Proving the number of three-digit perfect squares that are divisible by 4 is 6 -/
theorem three_digit_perfect_squares_divisible_by_4 : 
  (Finset.card (Finset.filter (λ n, n ∣ 4) (Finset.image (λ k, k * k) (Finset.Icc 10 31)))) = 6 :=
by
  sorry

end three_digit_perfect_squares_divisible_by_4_l425_425124


namespace smallest_prime_reverse_square_l425_425798

open Nat

-- Define a function to reverse the digits of a two-digit number
def reverseDigits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

-- Define the conditions
def isTwoDigitPrime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

def isSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define the main statement
theorem smallest_prime_reverse_square : 
  ∃ P, isTwoDigitPrime P ∧ isSquare (reverseDigits P) ∧ 
       ∀ Q, isTwoDigitPrime Q ∧ isSquare (reverseDigits Q) → P ≤ Q :=
by
  sorry

end smallest_prime_reverse_square_l425_425798


namespace sum_even_digits_1_to_200_l425_425438

-- Define the function E(n) which returns the sum of even digits of n.
def E (n : ℕ) : ℕ :=
  (n.digits 10).filter (λ d, d % 2 = 0).sum

-- Define the problem: Sum of all even digits in numbers from 1 to 200.
theorem sum_even_digits_1_to_200 :
  (Finset.range 201).sum E = 800 := by
  sorry

end sum_even_digits_1_to_200_l425_425438


namespace probability_quarter_circle_l425_425233

theorem probability_quarter_circle :
  (∀ (x y : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) → (sqrt (x^2 + y^2) ≤ 1)) →
  ∃ (p : ℝ), p = π / 4 :=
by
  sorry

end probability_quarter_circle_l425_425233


namespace find_FD_length_l425_425567

variable (A B C D E F : Type) [Point A] [Point B] [Point C] [Point D] [Point E] [Point F]
variable (AB BC CD DE AD BE : Line) (angle_ABC : Angle) (distance_AB : Real) (distance_BC : Real) 
variable (distance_DE : Real) 

-- Conditions:
variable (H1 : Parallelogram A B C D)
variable (H2 : angle_ABC = 100 * π / 180)
variable (H3 : distance_AB = 20)
variable (H4 : distance_BC = 12)
variable (H5 : ExtendLineThrough D CD E DE)
variable (H6 : LengthOf DE = 6)
variable (H7 : IntersectionPoint BE AD F)

-- Goal: Prove that FD = 4
theorem find_FD_length : FD = 4 :=
by
  sorry

end find_FD_length_l425_425567


namespace paths_for_content_l425_425763

def grid := [
  [none, none, none, none, none, none, some 'C', none, none, none, none, none, none, none],
  [none, none, none, none, none, some 'C', some 'O', some 'C', none, none, none, none, none, none],
  [none, none, none, none, some 'C', some 'O', some 'N', some 'O', some 'C', none, none, none, none, none],
  [none, none, none, some 'C', some 'O', some 'N', some 'T', some 'N', some 'O', some 'C', none, none, none, none],
  [none, none, some 'C', some 'O', some 'N', some 'T', some 'E', some 'T', some 'N', some 'O', some 'C', none, none, none],
  [none, some 'C', some 'O', some 'N', some 'T', some 'E', some 'N', some 'E', some 'T', some 'N', some 'O', some 'C', none, none],
  [some 'C', some 'O', some 'N', some 'T', some 'E', some 'N', some 'T', some 'N', some 'E', some 'T', some 'N', some 'O', some 'C']
]

def spelling_paths : Nat :=
  -- Skipping the actual calculation and providing the given total for now
  127

theorem paths_for_content : spelling_paths = 127 := sorry

end paths_for_content_l425_425763


namespace eccentricity_of_hyperbola_l425_425447

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (P F : ℝ × ℝ) 
  (PF_perp_x : P.2 ≠ 0) 
  (ratio_distances : abs (P.1 - F.1) / abs (P.1 + F.1) = 1 / 3) : 
  ℝ :=
(2 * Real.sqrt 3) / 3

theorem eccentricity_of_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (P F : ℝ × ℝ)
  (hyperbola_eqn : P.1^2 / a^2 - P.2^2 / b^2 = 1)
  (PF_perp_x : P.2 ≠ 0)
  (ratio_distances : abs ((P.1 / a) - (P.2 / b)) / abs ((P.1 / a) + (P.2 / b)) = 1 / 3) :
  hyperbola_eccentricity a b ha hb P F PF_perp_x ratio_distances = (2 * Real.sqrt 3) / 3 :=
sorry

end eccentricity_of_hyperbola_l425_425447


namespace john_paint_area_l425_425180

/-- John plans to paint one wall in his living room.
    The wall is 10 feet high and 15 feet long.
    There is a 3-foot by 5-foot area on that wall that he will not have to paint due to a large painting.
    How many square feet will he need to paint? -/
theorem john_paint_area : 
    let wall_area := 10 * 15 in
    let painting_area := 3 * 5 in
    let paint_area := wall_area - painting_area in
    paint_area = 135 := 
by
    let wall_area := 10 * 15
    let painting_area := 3 * 5
    let paint_area := wall_area - painting_area
    show paint_area = 135 from sorry

end john_paint_area_l425_425180


namespace Beast_of_War_running_time_l425_425288

theorem Beast_of_War_running_time 
  (M : ℕ) 
  (AE : ℕ) 
  (BoWAC : ℕ)
  (h1 : M = 120)
  (h2 : AE = M - 30)
  (h3 : BoWAC = AE + 10) : 
  BoWAC = 100 
  := 
sorry

end Beast_of_War_running_time_l425_425288


namespace count_no_isolated_points_subsets_l425_425098

def A := {i : ℕ | 1 ≤ i ∧ i ≤ 104}

def is_isolated (S : set ℕ) (x : ℕ) : Prop := 
  x ∈ S ∧ (x - 1 ∉ S) ∧ (x + 1 ∉ S)

def no_isolated_points (S : set ℕ) : Prop := 
  ∀ x ∈ S, is_isolated S x

def five_element_subsets_with_no_isolated_points (S : set ℕ) : Prop :=
  S ⊆ {i : ℕ | 1 ≤ i ∧ i ≤ 104} ∧ S.card = 5 ∧ no_isolated_points S

theorem count_no_isolated_points_subsets : 
  {S : set ℕ | five_element_subsets_with_no_isolated_points S}.card = 10000 :=
sorry

end count_no_isolated_points_subsets_l425_425098


namespace average_of_remaining_numbers_l425_425637

theorem average_of_remaining_numbers (a : Fin 12 → ℝ) (sum_12 : ∑ i, a i = 1080) (h80 : ∃ i j, i ≠ j ∧ a i = 80 ∧ a j = 82) :
  (∑ i, a i - 80 - 82) / 10 = 91.8 :=
by
  sorry

end average_of_remaining_numbers_l425_425637


namespace impossible_all_matches_outside_own_country_l425_425621

theorem impossible_all_matches_outside_own_country (n : ℕ) (h_teams : n = 16) : 
  ¬ ∀ (T : Fin n → Fin n → Prop), (∀ i j, i ≠ j → T i j) ∧ 
  (∀ i, ∀ j, i ≠ j → T i j → T j i) ∧ 
  (∀ i, T i i = false) → 
  ∀ i, ∃ j, T i j ∧ i ≠ j :=
by
  intro H
  sorry

end impossible_all_matches_outside_own_country_l425_425621


namespace frac_mul_square_l425_425753

theorem frac_mul_square 
  : (8/9)^2 * (1/3)^2 = 64/729 := 
by 
  sorry

end frac_mul_square_l425_425753


namespace grasshopper_catched_in_finite_time_l425_425707

theorem grasshopper_catched_in_finite_time :
  ∀ (x0 y0 x1 y1 : ℤ),
  ∃ (T : ℕ), ∃ (x y : ℤ), 
  ((x = x0 + x1 * T) ∧ (y = y0 + y1 * T)) ∧ -- The hunter will catch the grasshopper at this point
  ((∀ t : ℕ, t ≤ T → (x ≠ x0 + x1 * t ∨ y ≠ y0 + y1 * t) → (x = x0 + x1 * t ∧ y = y0 + y1 * t))) :=
sorry

end grasshopper_catched_in_finite_time_l425_425707


namespace proof_problem_l425_425199

theorem proof_problem (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b ∣ c * (c ^ 2 - c + 1))
  (h5 : (c ^ 2 + 1) ∣ (a + b)) :
  (a = c ∧ b = c ^ 2 - c + 1) ∨ (a = c ^ 2 - c + 1 ∧ b = c) :=
sorry

end proof_problem_l425_425199


namespace shift_sine_function_right_by_quarter_period_l425_425238

theorem shift_sine_function_right_by_quarter_period :
  ∀ x, 2 * sin (2 * x + π / 6) = 2 * sin (2 * (x - π / 4) + π / 6) ↔ 2 * sin (2 * x - π / 3) := 
by
  intro x
  sorry

end shift_sine_function_right_by_quarter_period_l425_425238


namespace ball_dance_problem_l425_425338

def exists_boy_girl_half_dances (T D : Fin 100 → ℕ) (danced : Fin 100 → Fin 100 → ℕ) : Prop :=
  ∃ k : Fin 100, danced k k ≤ T k / 2

theorem ball_dance_problem :
  ∀ (danced : Fin 100 → Fin 100 → ℕ)
  (T : Fin 100 → ℕ) (D : Fin 100 → ℕ),
  (∀ (g : Fin 100), ∃ (group1 group2 : Finset (Fin 100)),
    (group1 ∪ group2 = Finset.univ) ∧ 
    (group1 ∩ group2 = ∅) ∧ 
    (∑ x in group1, danced x g = ∑ x in group2, danced x g)) →
  exists_boy_girl_half_dances T D danced :=
by
  sorry

end ball_dance_problem_l425_425338


namespace vector_BC_l425_425527

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2)

theorem vector_BC (BA CA BC : ℝ × ℝ) (BA_def : BA = (1, 2)) (CA_def : CA = (4, 5)) (BC_def : BC = vector_sub BA CA) : BC = (-3, -3) :=
by
  subst BA_def
  subst CA_def
  subst BC_def
  sorry

end vector_BC_l425_425527


namespace part_a_part_b_l425_425593

variables {α : Real}
variables {A B C C1 A1 B1 A' B' C': Type*} -- points in space

-- Given conditions
def cond1 : Prop := ∠(C, C1, A B) = α
def cond2 : Prop := ∠(A, A1, B C) = α
def cond3 : Prop := ∠(B, B1, C A) = α
def cond4 : Prop := intersection (A, A1) (B, B1) = C'
def cond5 : Prop := intersection (B, B1) (C, C1) = A'
def cond6 : Prop := intersection (C, C1) (A, A1) = B'

-- Part (a) proof statement
theorem part_a : (cond1 ∧ cond2 ∧ cond3 ∧ cond4 ∧ cond5 ∧ cond6) →
  orthocenter(ABC) = circumcenter(A'B'C') :=
sorry

-- Part (b) proof statement
theorem part_b : (cond1 ∧ cond2 ∧ cond3 ∧ cond4 ∧ cond5 ∧ cond6) →
  similar(ABC, A'B'C') (2 * cos α) :=
sorry

end part_a_part_b_l425_425593


namespace first_number_lcm_14_20_l425_425792

theorem first_number_lcm_14_20 (x : ℕ) (h : Nat.lcm x (Nat.lcm 14 20) = 140) : x = 1 := sorry

end first_number_lcm_14_20_l425_425792


namespace closest_integer_to_sqrt_29_l425_425266

theorem closest_integer_to_sqrt_29 : 
  let sqrt_25 := real.sqrt 25
  let sqrt_36 := real.sqrt 36
  let sqrt_29 := real.sqrt 29
  5 < sqrt_29 → sqrt_29 < 6 → 5.5 * 5.5 = 30.25 → 30.25 > 29 → Int.round sqrt_29 = 5 :=
by
  intros sqrt_25 sqrt_36 sqrt_29 h1 h2 h3 h4
  have sqrt_25_lt := h1
  have sqrt_29_lt_6 := h2
  have midpoint_sq := h3
  have midpoint_sq_gt_29 := h4
  -- The proof goes here.
  sorry

end closest_integer_to_sqrt_29_l425_425266


namespace find_f_13_l425_425335

variable (f : ℤ → ℤ)

def is_odd_function (f : ℤ → ℤ) := ∀ x : ℤ, f (-x) = -f (x)
def has_period_4 (f : ℤ → ℤ) := ∀ x : ℤ, f (x + 4) = f (x)

theorem find_f_13 (h1 : is_odd_function f) (h2 : has_period_4 f) (h3 : f (-1) = 2) : f 13 = -2 :=
by
  sorry

end find_f_13_l425_425335


namespace prime_divisors_6270_l425_425141

theorem prime_divisors_6270 : 
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1 = 2 ∧ p2 = 3 ∧ p3 = 5 ∧ p4 = 11 ∧ p5 = 19 ∧ 
  (p1 * p2 * p3 * p4 * p5 = 6270) ∧ 
  (Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ Nat.Prime p4 ∧ Nat.Prime p5) ∧ 
  (∀ q, Nat.Prime q ∧ q ∣ 6270 → (q = p1 ∨ q = p2 ∨ q = p3 ∨ q = p4 ∨ q = p5)) := 
by 
  sorry

end prime_divisors_6270_l425_425141


namespace determinant_of_cross_product_matrix_l425_425105

variables {R : Type*} [Field R] {a b c : R^3} {p q r : R}

def D (p q r : R) (a b c : R^3) : R :=
  p * q * r * (a ⬝ (b × c))

def new_matrix_det (p q r : R) (a b c : R^3) : R :=
  det ![
    [p * a × q * b, q * b × r * c, r * c × p * a]
  ]

theorem determinant_of_cross_product_matrix (a b c : R^3) (p q r : R) :
  new_matrix_det p q r a b c = p * q^2 * r^2 * (D p q r a b c) ^ 2 :=
sorry

end determinant_of_cross_product_matrix_l425_425105


namespace johnny_rate_is_4_l425_425217

noncomputable def johnny_walking_rate (distance_QY : ℝ) (matthew_rate : ℝ) (johnny_distance : ℝ) (time_difference : ℝ) : ℝ :=
let johnny_time := (distance_QY - johnny_distance - matthew_rate * time_difference) / matthew_rate in
johnny_distance / johnny_time

theorem johnny_rate_is_4 :
  johnny_walking_rate 45 3 24 1 = 4 := by
  sorry

end johnny_rate_is_4_l425_425217


namespace real_solution_implies_a_l425_425520

theorem real_solution_implies_a (a : ℝ) (x : ℝ) (i : ℂ) (h : i = complex.I) :
  ((complex.of_real (1 : ℝ) + h) * x^2 - 2 * (complex.of_real a + h) * x + (complex.of_real 5 - 3 * h)) = 0 →
  (a = 7 / 3 ∨ a = -3) :=
by {
  intros ha hb,
  sorry
}

end real_solution_implies_a_l425_425520


namespace sin_theta_geometric_progression_l425_425938

theorem sin_theta_geometric_progression (θ : ℝ) (h_acute : 0 < θ ∧ θ < π / 2)
  (h_geo : ∃ r : ℝ, (cos θ = r * Δ ∧ cos 2θ = Δ ∧ cos 3θ = r * Δ)
                    ∨ (cos 2θ = r * Δ ∧ cos 3θ = Δ ∧ cos θ = r * Δ)
                    ∨ (cos 3θ = r * Δ ∧ cos θ = Δ ∧ cos 2θ = r * Δ)) :
  sin θ = √(2) / 2 :=
by
  sorry

end sin_theta_geometric_progression_l425_425938


namespace cost_per_amulet_is_30_l425_425419

variable (days_sold : ℕ := 2)
variable (amulets_per_day : ℕ := 25)
variable (price_per_amulet : ℕ := 40)
variable (faire_percentage : ℕ := 10)
variable (profit : ℕ := 300)

def total_amulets_sold := days_sold * amulets_per_day
def total_revenue := total_amulets_sold * price_per_amulet
def faire_cut := total_revenue * faire_percentage / 100
def revenue_after_faire := total_revenue - faire_cut
def total_cost := revenue_after_faire - profit
def cost_per_amulet := total_cost / total_amulets_sold

theorem cost_per_amulet_is_30 : cost_per_amulet = 30 := by
  sorry

end cost_per_amulet_is_30_l425_425419


namespace product_of_distances_to_origin_equals_four_l425_425071

theorem product_of_distances_to_origin_equals_four
  (O : Point) (origin_eq : O = (0, 0))
  (parabola_C : Set Point) (parabola_eq : ∀ {x y : ℝ}, (x, y) ∈ parabola_C ↔ y^2 = 8 * x)
  (F : Point) (focus_eq : F = (2, 0))
  (A : Point) (A_eq : A = (2, 4))
  (P Q : Point) (P_in_parabola : P ∈ parabola_C)
  (Q_in_parabola : Q ∈ parabola_C)
  (P_ne_A : P ≠ A) (Q_ne_A : Q ≠ A)
  (line_l : Line)
  (l_through_F : ∀ {x y : ℝ}, (x, y) ∈ line_l ↔ x = m * y + 2)
  (P_on_l : ∀ {x y : ℝ}, (x, y) = P → (x, y) ∈ line_l)
  (Q_on_l : ∀ {x y : ℝ}, (x, y) = Q → (x, y) ∈ line_l)
  (M N : Point)
  (M_on_x_axis : ∃ y : ℝ, M = (x, 0))
  (N_on_x_axis : ∃ y : ℝ, N = (x, 0))
  (AP_on_M : ∀ {x y : ℝ}, onLine (A, P) (x, y) ↔ x = M)
  (AQ_on_N : ∀ {x y : ℝ}, onLine (A, Q) (x, y) ↔ x = N):
  |OM|.x * |ON|.x = 4 :=
by
  sorry

end product_of_distances_to_origin_equals_four_l425_425071


namespace parkway_elementary_fifth_grade_students_l425_425546

def total_students (num_boys : ℕ) (num_playing_soccer : ℕ) (percent_boys_playing_soccer : ℕ) (num_girls_not_playing_soccer : ℕ) : ℕ :=
  let boys_playing_soccer := percent_boys_playing_soccer * num_playing_soccer / 100
  let boys_not_playing_soccer := num_boys - boys_playing_soccer
  let total_not_playing_soccer := boys_not_playing_soccer + num_girls_not_playing_soccer
  let total_students := num_playing_soccer + total_not_playing_soccer
  total_students

theorem parkway_elementary_fifth_grade_students :
  total_students 320 250 86 65 = 420 :=
by
  simp [total_students]
  norm_num
  sorry

end parkway_elementary_fifth_grade_students_l425_425546


namespace number_of_hydrogen_atoms_is_6_l425_425348

-- Define the conditions as Lean definitions
def num_carbon_atoms : ℕ := 6
def molecular_weight : ℝ := 78
def atomic_weight_carbon : ℝ := 12.01
def atomic_weight_hydrogen : ℝ := 1.008

-- Define a function to calculate the number of hydrogen atoms
def num_hydrogen_atoms (num_carbon : ℕ) (mol_weight : ℝ) (aw_carbon : ℝ) (aw_hydrogen : ℝ) : ℝ :=
  ((mol_weight - (num_carbon * aw_carbon)) / aw_hydrogen)

-- Theorem to prove the number of hydrogen atoms
theorem number_of_hydrogen_atoms_is_6 :
  num_hydrogen_atoms num_carbon_atoms molecular_weight atomic_weight_carbon atomic_weight_hydrogen ≈ 6 :=
by
  -- The proof is skipped here; using sorry to indicate it needs to be completed.
  sorry

end number_of_hydrogen_atoms_is_6_l425_425348


namespace sum_of_repeating_decimals_l425_425014

/-- Definitions of the repeating decimals as real numbers. --/
def x : ℝ := 0.3 -- This actually represents 0.\overline{3} in Lean
def y : ℝ := 0.04 -- This actually represents 0.\overline{04} in Lean
def z : ℝ := 0.005 -- This actually represents 0.\overline{005} in Lean

/-- The theorem stating that the sum of these repeating decimals is a specific fraction. --/
theorem sum_of_repeating_decimals : x + y + z = (14 : ℝ) / 37 := 
by 
  sorry -- Placeholder for the proof

end sum_of_repeating_decimals_l425_425014


namespace find_A_l425_425910

theorem find_A (A B C : ℕ) (h1 : A = B * C + 8) (h2 : A + B + C = 2994) : A = 8 ∨ A = 2864 :=
by
  sorry

end find_A_l425_425910


namespace ratio_of_altitude_to_base_l425_425989

-- Defining the conditions given in the problem
def area : ℝ := 450
def base : ℝ := 15

-- The question corresponds to proving the ratio is 2
theorem ratio_of_altitude_to_base : 
  let altitude := area / base in 
  altitude / base = 2 :=
by
  -- Axiom is set directly to convert the problem into a statement 
  sorry

end ratio_of_altitude_to_base_l425_425989


namespace frac_mul_square_l425_425754

theorem frac_mul_square 
  : (8/9)^2 * (1/3)^2 = 64/729 := 
by 
  sorry

end frac_mul_square_l425_425754


namespace phil_baseball_cards_left_l425_425225

-- Step a): Define the conditions
def packs_week := 20
def weeks_year := 52
def lost_factor := 1 / 2

-- Step c): Establish the theorem statement
theorem phil_baseball_cards_left : 
  (packs_week * weeks_year * (1 - lost_factor) = 520) := 
  by
    -- proof steps will come here
    sorry

end phil_baseball_cards_left_l425_425225


namespace train_length_l425_425378

/-- Define the conversion from kmph to m/s -/
def kmph_to_mps (v : ℕ) : ℝ := (v * 1000) / 3600

/-- Given conditions -/
def train_speed_kmph : ℕ := 60
def time_to_cross_seconds : ℝ := 23.998080153587715
def bridge_length_meters : ℝ := 290
def total_distance_meters : ℝ := kmph_to_mps train_speed_kmph * time_to_cross_seconds

theorem train_length :
  total_distance_meters - bridge_length_meters = 110 :=
by
  sorry

end train_length_l425_425378


namespace angle_C_value_triangle_area_l425_425554

-- Define the triangle with sides a, b, c and opposite angles A, B, C
variables {a b c : ℝ} {A B C : ℝ}

-- Part 1: Prove the value of angle C
theorem angle_C_value (h : a * (cos B) * (cos C) + b * (cos A) * (cos C) = c / 2) : 
  C = π / 3 := sorry

-- Part 2: Prove the area of the triangle given c = √7 and a + b = 5
theorem triangle_area 
  (h1 : c = Real.sqrt 7) 
  (h2 : a + b = 5) 
  (C_eq_pi_div_3 : C = π / 3) : 
  (1/2) * a * b * (sin C) = (3 * Real.sqrt 3) / 2 := sorry

end angle_C_value_triangle_area_l425_425554


namespace min_value_expression_l425_425457

theorem min_value_expression (a b m n : ℝ) 
    (h_a_pos : 0 < a) (h_b_pos : 0 < b) 
    (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
    (h_sum_one : a + b = 1) 
    (h_prod_two : m * n = 2) :
    (a * m + b * n) * (b * m + a * n) = 2 :=
sorry

end min_value_expression_l425_425457


namespace problem_statement_l425_425205

noncomputable def z : ℂ := (1 + complex.I) / real.sqrt 2

theorem problem_statement :
  let sum1 := (∑ k in finset.range 1 13, z^(k^2))
  let sum2 := (∑ k in finset.range 1 13, z^(-k^2))
  (sum1 * sum2) = 36 :=
sorry

end problem_statement_l425_425205


namespace gcd_n_four_plus_sixteen_and_n_plus_three_l425_425036

theorem gcd_n_four_plus_sixteen_and_n_plus_three (n : ℕ) (hn1 : n > 9) (hn2 : n ≠ 94) :
  Nat.gcd (n^4 + 16) (n + 3) = 1 :=
by
  sorry

end gcd_n_four_plus_sixteen_and_n_plus_three_l425_425036


namespace fraction_division_l425_425505

theorem fraction_division:
  (1 / 4) / (1 / 8) = 2 :=
by
  sorry

end fraction_division_l425_425505


namespace fraction_multiplication_exponent_l425_425751

theorem fraction_multiplication_exponent :
  ( (8 : ℚ) / 9 )^2 * ( (1 : ℚ) / 3 )^2 = (64 / 729 : ℚ) := 
by
  sorry

end fraction_multiplication_exponent_l425_425751


namespace values_of_n_l425_425813

/-
  Given a natural number n and a target sum 100,
  we need to find if there exists a combination of adding and subtracting 1 through n
  such that the sum equals 100.

- A value k is representable as a sum or difference of 1 through n if the sum of the series
  can be manipulated to produce k.
- The sum of the first n natural numbers S_n = n * (n + 1) / 2 must be even and sufficiently large.
- The specific values that satisfy the conditions are of the form n = 15 + 4 * k or n = 16 + 4 * k.
-/

def exists_sum_to_100 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 15 + 4 * k ∨ n = 16 + 4 * k

theorem values_of_n (n : ℕ) : exists_sum_to_100 n ↔ (∃ (k : ℕ), n = 15 + 4 * k ∨ n = 16 + 4 * k) :=
by { sorry }

end values_of_n_l425_425813


namespace oddly_powerful_less_than_5000_count_l425_425403

noncomputable def is_oddly_powerful (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 1 ∧ b % 2 = 1 ∧ a^b = n

theorem oddly_powerful_less_than_5000_count : 
  ({ n : ℕ | n < 5000 ∧ is_oddly_powerful n }.card = 24) :=
sorry

end oddly_powerful_less_than_5000_count_l425_425403


namespace reciprocal_sum_is_1_implies_at_least_one_is_2_l425_425042

-- Lean statement for the problem
theorem reciprocal_sum_is_1_implies_at_least_one_is_2 (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_sum : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / d = 1) : 
  a = 2 ∨ b = 2 ∨ c = 2 ∨ d = 2 := 
sorry

end reciprocal_sum_is_1_implies_at_least_one_is_2_l425_425042


namespace parallelepiped_space_diagonal_and_sphere_surface_area_l425_425844

theorem parallelepiped_space_diagonal_and_sphere_surface_area
    (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = sqrt 3) 
    (vertices_on_sphere : true) :
    let D := real.sqrt (a^2 + b^2 + c^2) in
    let R := D / 2 in
    D = 4 ∧ 4 * Real.pi * R ^ 2 = 16 * Real.pi :=
by
  sorry

end parallelepiped_space_diagonal_and_sphere_surface_area_l425_425844


namespace find_r_l425_425784

noncomputable theory

def exists_set_S (r : ℝ) : Prop :=
∃ (S : set ℝ), 
  (∀ t : ℝ, (t ∈ S ∨ (t + r) ∈ S ∨ (t + 1) ∈ S) ∧ (¬ (t ∈ S ∧ (t + r) ∈ S) ∧ ¬ (t ∈ S ∧ (t + 1) ∈ S) ∧ ¬ ((t + r) ∈ S ∧ (t + 1) ∈ S))) ∧
  (∀ t : ℝ, (t ∈ S ∨ (t - r) ∈ S ∨ (t - 1) ∈ S) ∧ (¬ (t ∈ S ∧ (t - r) ∈ S) ∧ ¬ (t ∈ S ∧ (t - 1) ∈ S) ∧ ¬ ((t - r) ∈ S ∧ (t - 1) ∈ S)))

theorem find_r (r : ℝ) (hr_pos : 0 < r) (hr_lt_one : r < 1) :
  exists_set_S r ↔ irrational r ∨ ∃ a b : ℤ, r = a / b ∧ (3 ∣ (a + b)) :=
sorry

end find_r_l425_425784


namespace annular_region_area_l425_425703

/-- Define the radii of the circles -/
def r_small := 4
def r_large := 10

/-- The areas of the circles -/
def A_small := Real.pi * r_small^2
def A_large := Real.pi * r_large^2

/-- Area of the annular region is the difference of the areas of the two circles -/
theorem annular_region_area : (A_large - A_small) = 84 * Real.pi := by
  refine eq.trans _ (sub_eq_of_eq_add _)
  refine add_eq_of_eq_sub _
  have h: (10:ℝ) ^ 2 = 100 := by norm_num
  have g: (4:ℝ) ^ 2 = 16 := by norm_num
  simp [A_large, A_small, h, g]
  sorry -- Proof omitted

end annular_region_area_l425_425703


namespace min_value_x_plus_reciprocal_min_value_x_plus_reciprocal_equality_at_one_l425_425144

theorem min_value_x_plus_reciprocal (x : ℝ) (h : x > 0) : x + 1 / x ≥ 2 :=
by
  sorry

theorem min_value_x_plus_reciprocal_equality_at_one : (1 : ℝ) + 1 / 1 = 2 :=
by
  norm_num

end min_value_x_plus_reciprocal_min_value_x_plus_reciprocal_equality_at_one_l425_425144


namespace max_T_n_value_l425_425460

-- Definitions based on the conditions provided
variables {a : ℕ → ℝ} (n k l m : ℕ)
def geometric_sequence (a : ℕ → ℝ) := ∀ i, 0 < a i
def a4_condition (a : ℕ → ℝ) := a 4 = a 2 * a 5
def a5_a4_relation (a : ℕ → ℝ) := 3 * a 5 + 2 * a 4 = 1

-- The theorem to be proven
theorem max_T_n_value 
  (h_geo : geometric_sequence a) 
  (h_a4 : a4_condition a) 
  (h_a5_a4 : a5_a4_relation a) : 
  n = 3 → (a 1 * a 2 * a 3 = 27) :=
begin
  sorry
end

end max_T_n_value_l425_425460


namespace area_MPC_l425_425169

noncomputable def point := (Float, Float)

def A : point := (0, 0)
def B : point := (3, 0)
def C : point := (3, 4)
def D : point := (0, 4)

def mid (p1 p2 : point) : point := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def M : point := mid A B
def N : point := mid C D
def P : point := mid M N

def area_triangle (p1 p2 p3 : point) : Float :=
  0.5 * abs (p1.1*(p2.2 - p3.2) + p2.1*(p3.2 - p1.2) + p3.1*(p1.2 - p2.2))

theorem area_MPC : area_triangle M P C = 3 := by
  sorry

end area_MPC_l425_425169


namespace distance_between_cars_approximately_40_85_l425_425309

noncomputable def distance_between_cars : ℝ :=
  let initial_distance := 150 -- km
  let distance_first_car_on_main_road := 25 + 25 -- km (25 km initial, 25 km returning to the main road)
  let distance_second_car_on_main_road := 62 -- km
  let distance_second_car_remaining := initial_distance - distance_first_car_on_main_road - distance_second_car_on_main_road -- km
  let perpendicular_distance_first_car := 15 -- km (perpendicular distance after taking turns)
  real.sqrt (distance_second_car_remaining ^ 2 + perpendicular_distance_first_car ^ 2)

theorem distance_between_cars_approximately_40_85 :
  abs (distance_between_cars - 40.85) < 0.01 := sorry

end distance_between_cars_approximately_40_85_l425_425309


namespace trajectory_ellipse_l425_425490

/--
Given two fixed points A(-2,0) and B(2,0) in the Cartesian coordinate system, 
if a moving point P satisfies |PA| + |PB| = 6, 
then prove that the equation of the trajectory for point P is (x^2) / 9 + (y^2) / 5 = 1.
-/
theorem trajectory_ellipse (P : ℝ × ℝ)
  (A B : ℝ × ℝ)
  (hA : A = (-2, 0))
  (hB : B = (2, 0))
  (hPA_PB : dist P A + dist P B = 6) :
  (P.1 ^ 2) / 9 + (P.2 ^ 2) / 5 = 1 :=
sorry

end trajectory_ellipse_l425_425490


namespace simplest_square_root_l425_425672

theorem simplest_square_root :
  let sqrts := [(sqrt 6 : Real), (sqrt 8), (sqrt (2 / 3)), (sqrt 0.5)] in
  ∀ (s : Real), s ∈ sqrts → 
  (s = sqrt 6) = 
  (s ≠ sqrt 8 ∧ s ≠ sqrt (2 / 3) ∧ s ≠ sqrt 0.5) := 
by 
  sorry

end simplest_square_root_l425_425672


namespace find_angle_between_vectors_l425_425834

variables {E : Type*} [inner_product_space ℝ E]
variables (a b : E)

theorem find_angle_between_vectors
  (ha : ∥a∥ = 2)
  (hb : ∥b∥ = 4)
  (orth : inner_product_space.inner (a + b) a = 0) :
  real.angle a b = 2 * real.pi / 3 :=
by sorry

end find_angle_between_vectors_l425_425834


namespace sum_of_points_probabilities_l425_425670

-- Define probabilities for the sums of 2, 3, and 4
def P_A : ℚ := 1 / 36
def P_B : ℚ := 2 / 36
def P_C : ℚ := 3 / 36

-- Theorem statement
theorem sum_of_points_probabilities :
  (P_A < P_B) ∧ (P_B < P_C) :=
  sorry

end sum_of_points_probabilities_l425_425670


namespace sunzi_equations_l425_425909

theorem sunzi_equations (x y : ℝ) (h1 : x + 4.5 = y) (h2 : x + 1 = 0.5 * y) :
  x + 4.5 = y ∧ x - 1 = 0.5 * y :=
by
  exact ⟨h1, (h2)⟩

end sunzi_equations_l425_425909


namespace range_of_function_l425_425473

theorem range_of_function (a : ℝ) (m : ℝ) (h: ∀ y, ∃ x, y = (x + a) / (x^2 + 1) ↔ y ∈ Icc (-1/4) m) :
  a = 3 / 4 ∧ m = 1 :=
by
  sorry

end range_of_function_l425_425473


namespace differentiable_everywhere_l425_425928

variable {f : ℝ → ℝ}

theorem differentiable_everywhere
  (h1 : ∀ x y : ℝ, f (x + y) = f x * f y)
  (h2 : f 0 ≠ 0)
  (h3 : f 0 < ⊤) :
    ∀ x : ℝ, Differentiable ℝ f :=
by
  sorry

end differentiable_everywhere_l425_425928


namespace round_robin_10_person_tournament_l425_425371

noncomputable def num_matches (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem round_robin_10_person_tournament :
  num_matches 10 = 45 :=
by
  sorry

end round_robin_10_person_tournament_l425_425371


namespace rectangle_perimeter_l425_425614

theorem rectangle_perimeter :
  ∀ (x y : ℝ),
  x * y = 2500 →
  ∃ (a b : ℝ),
  π * a * b = 2500 * π ∧
  x^2 + y^2 = 4 * (a^2 - b^2) ∧
  (∃ k : ℝ, x = 5 * k ∧ y = 4 * k) →
  2 * (x + y) = 450 :=
by
  intros x y h_area h_cond
  rcases h_cond with ⟨a, b, h_area_ell, h_diag, ⟨k, hx, hy⟩⟩
  sorry

end rectangle_perimeter_l425_425614


namespace line_tangent_to_circle_l425_425945

-- Definitions based on conditions
def Gamma (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ (x^2 / a^2) + (y^2 / b^2) = 1}

def orthogonal (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 * p2.1 + p1.2 * p2.2 = 0

-- Problem statement following from conditions
theorem line_tangent_to_circle
    (a b : ℝ)
    (l : ℝ → ℝ → Prop) -- the line equation px + qy = 1
    (A B : ℝ × ℝ)
    (hGamma_A : (A.1^2 / a^2) + (A.2^2 / b^2) = 1)
    (hGamma_B : (B.1^2 / a^2) + (B.2^2 / b^2) = 1)
    (hLine_A : l A.1 A.2)
    (hLine_B : l B.1 B.2)
    (hOrthogonal : orthogonal A B)
  : ∃ C : ℝ, ∀ (x y : ℝ), l x y → x^2 + y^2 = C
  ∧ C = a ^ 2 * b ^ 2 / (a ^ 2 + b ^ 2) := sorry

end line_tangent_to_circle_l425_425945


namespace least_xy_value_l425_425066

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 90 :=
by
  sorry

end least_xy_value_l425_425066


namespace number_of_sheep_total_number_of_animals_l425_425591

theorem number_of_sheep (ratio_sh_horse : 5 / 7 * horses = sheep) 
    (horse_food_per_day : horses * 230 = 12880) :
    sheep = 40 :=
by
  -- These are all the given conditions
  sorry

theorem total_number_of_animals (sheep : ℕ) (horses : ℕ)
    (H1 : sheep = 40) (H2 : horses = 56) :
    sheep + horses = 96 :=
by
  -- Given conditions for the total number of animals on the farm
  sorry

end number_of_sheep_total_number_of_animals_l425_425591


namespace count_three_digit_squares_divisible_by_4_l425_425115

def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

theorem count_three_digit_squares_divisible_by_4 : 
  let squares := {n : ℕ | is_three_digit n ∧ is_perfect_square n ∧ is_divisible_by_4 n} in
  Fintype.card squares = 10 :=
sorry

end count_three_digit_squares_divisible_by_4_l425_425115


namespace inequality_proof_l425_425445

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b + b^2 / c + c^2 / a) + (a + b + c) ≥ (6 * (a^2 + b^2 + c^2) / (a + b + c)) :=
by
  sorry

end inequality_proof_l425_425445


namespace bike_distance_from_rest_l425_425345

variable (u : ℝ) (a : ℝ) (t : ℝ)

theorem bike_distance_from_rest (h1 : u = 0) (h2 : a = 0.5) (h3 : t = 8) : 
  (1 / 2 * a * t^2 = 16) :=
by
  sorry

end bike_distance_from_rest_l425_425345


namespace maximum_value_of_f_l425_425854

def f (λ : ℝ) (x : ℝ) := λ * Real.sin x + Real.cos x

theorem maximum_value_of_f (λ : ℝ) (h : ∀ x, f λ (x + π - π / 6) = f λ (π / 6 - x)) :
  ∀ x, f λ x ≤ (2 * Real.sqrt 3) / 3 := sorry

end maximum_value_of_f_l425_425854


namespace perimeter_equilateral_triangle_correct_l425_425795

noncomputable def is_equilateral_triangle_perimeter : Prop :=
  ∃ (R : ℝ), 
  (∃ (chord : ℝ) (distance : ℝ), 
  chord = 2 ∧
  distance = 3 ∧
  R^2 = distance^2 + (chord / 2)^2) ∧
  (3 * (R * sqrt 3)) = 3 * sqrt 30

theorem perimeter_equilateral_triangle_correct : is_equilateral_triangle_perimeter :=
  sorry

end perimeter_equilateral_triangle_correct_l425_425795


namespace tangent_line_equation_l425_425787

theorem tangent_line_equation :
  ∀ x y : ℝ,
  (y = x^3 - x + 3) →
  ((x = 1 ∧ y = 3) →
  (∃ c : ℝ, y = c * (x - 1) + 3 ∧ c = 2) →
  (2 * x - y + 1 = 0)) :=
begin
  intros x y h_curve h_tangent line_eq,
  sorry
end

end tangent_line_equation_l425_425787


namespace circle_diameter_l425_425533

variables {m n : ℝ}

theorem circle_diameter (h_perp : ∀ A B C D : ℝ, (A - B) * (C - D) = 0)
  (h_AD_eq_m : ∀ A D : ℝ, A - D = m)
  (h_BC_eq_n : ∀ B C : ℝ, B - C = n) :
  diameter = real.sqrt (m^2 + n^2) :=
sorry

end circle_diameter_l425_425533


namespace three_digit_perfect_squares_divisible_by_4_l425_425128

/-- Proving the number of three-digit perfect squares that are divisible by 4 is 6 -/
theorem three_digit_perfect_squares_divisible_by_4 : 
  (Finset.card (Finset.filter (λ n, n ∣ 4) (Finset.image (λ k, k * k) (Finset.Icc 10 31)))) = 6 :=
by
  sorry

end three_digit_perfect_squares_divisible_by_4_l425_425128


namespace propositions_are_true_l425_425089

-- Define the conditions (propositions) as boolean values (truth values)
def proposition1 := ∀ (l1 l2 l3 l4 : Line), 
  (skew l1 l2 ∧ intersect l3 l1 ∧ intersect l4 l2) → skew l3 l4

def proposition2 := ∀ (l1 l2 : Line) (p1 p2 : Plane),
  (parallel p1 p2 ∧ parallel l1 p1 ∧ parallel l2 p2) → parallel l1 l2

def proposition3 := ∀ (l1 l2 : Line) (p : Plane),
  (perpendicular l1 p ∧ perpendicular l2 p) → parallel l1 l2

def proposition4 := ∀ (p1 p2 : Plane) (l : Line),
  (perpendicular p1 p2 ∧ ¬ perpendicular l (intersection_line p1 p2)) → 
  ∀ (l' : Line), (l' ∈ plane p1) → ¬ perpendicular l' p2

-- Define the correct answer as boolean values
def correct_answer := proposition3 ∧ proposition4

-- The theorem stating that propositions ③ and ④ are true
theorem propositions_are_true : correct_answer := by
  split
  -- Proof for ③
  sorry
  -- Proof for ④
  sorry

end propositions_are_true_l425_425089


namespace simplify_expression_l425_425395

variable (x y : ℝ)

-- Define the proposition
theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) : 
  (6 * x^2 * y - 2 * x * y^2) / (2 * x * y) = 3 * x - y := 
by
  sorry

end simplify_expression_l425_425395


namespace ellipse_standard_equation_l425_425443

theorem ellipse_standard_equation
  (F1 F2 : ℝ × ℝ)
  (hF1 : F1 = (-1, 0))
  (hF2 : F2 = (1, 0))
  (l : ℝ → ℝ × ℝ)
  (M N : ℝ × ℝ)
  (hlM : ∃ t, l t = M ∧ F1 = l t)
  (hlN : ∃ t, l t = N ∧ F1 = l t)
  (h_perimeter : dist M F2 + dist F1 M + dist N F2 = 8) :
  ellipse_eq : (∀ x y, ((x^2) / 4) + ((y^2) / 3) = 1) :=
sorry

end ellipse_standard_equation_l425_425443


namespace total_receipts_is_1405_50_l425_425592

-- Conditions
def total_people := 754
def children_count := 388
def adults_count := total_people - children_count
def children_price := 1.50
def adults_price := 2.25

-- Question (Proof Problem)
theorem total_receipts_is_1405_50 :
  (children_price * children_count + adults_price * adults_count) = 1405.50 :=
by
  sorry

end total_receipts_is_1405_50_l425_425592


namespace length_more_than_breadth_l425_425267

theorem length_more_than_breadth (b : ℝ) (x : ℝ) 
  (h1 : b + x = 55) 
  (h2 : 4 * b + 2 * x = 200) 
  (h3 : (5300 : ℝ) / 26.5 = 200)
  : x = 10 := 
by
  sorry

end length_more_than_breadth_l425_425267


namespace count_three_digit_squares_divisible_by_4_l425_425116

def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

theorem count_three_digit_squares_divisible_by_4 : 
  let squares := {n : ℕ | is_three_digit n ∧ is_perfect_square n ∧ is_divisible_by_4 n} in
  Fintype.card squares = 10 :=
sorry

end count_three_digit_squares_divisible_by_4_l425_425116


namespace distance_AB_l425_425956

theorem distance_AB (h1 : ∃ x, (x : ℝ) + real.sqrt (144 + x^2) + 12 = 60) 
                    (h2 : ∀ x, ∃ x, (x : ℝ) = 12) : 
                    ∃ x, (x : ℝ) = 22.5 :=
by
  sorry

end distance_AB_l425_425956


namespace least_positive_integer_proof_l425_425026

-- Definitions used in the problem
noncomputable def sum_of_terms : ℝ :=
  (∑ k in finset.Ico 10 99, (1 / (Real.sin (k : ℝ) * Real.sin (k + 1))))

noncomputable def least_positive_integer :=
  1

-- The statement of the problem to be proved
theorem least_positive_integer_proof :
  sum_of_terms = 1 / Real.sin 1 :=
sorry

end least_positive_integer_proof_l425_425026


namespace complex_number_in_second_quadrant_l425_425543

-- Define the complex number
def complex_number : ℂ := (1 : ℂ) * Complex.I / (Real.sqrt 3 - Complex.I)

-- Check the quadrant of the complex number
theorem complex_number_in_second_quadrant : 
  complex_number.re < 0 ∧ complex_number.im > 0 :=
by
  -- Sorry leaves the proof empty as per instructions
  sorry

end complex_number_in_second_quadrant_l425_425543


namespace calc_a_b_50_l425_425492

noncomputable def a_seq (a_1 : ℝ) (b_seq : ℕ → ℝ) : ℕ → ℝ
| 0     := a_1
| (n+1) := a_seq n + 1 / b_seq n

noncomputable def b_seq (b_1 : ℝ) (a_seq : ℕ → ℝ) : ℕ → ℝ
| 0     := b_1
| (n+1) := b_seq n + 1 / a_seq n

theorem calc_a_b_50 (a_1 b_1 : ℝ) (h₁ : a_1 > 0) (h₂ : b_1 > 0) :
  let a_seq := a_seq a_1 (b_seq b_1 (a_seq a_1))
  let b_seq := b_seq b_1 (a_seq a_1)
  a_seq 49 + b_seq 49 > 20 :=
sorry

end calc_a_b_50_l425_425492


namespace hyperbola_eccentricity_l425_425093

theorem hyperbola_eccentricity (a b : ℝ) :
  (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ∧ (∃ y : ℝ, y = (b / a) * x) ∧
  (∃ y : ℝ, y = x^2 + 1)) →
  ∃ e : ℝ, e = sqrt 5 :=
begin
  sorry
end

end hyperbola_eccentricity_l425_425093


namespace min_square_side_length_suffices_for_circles_l425_425436

-- Define the radius of the circles
def radius : ℝ := 1

-- Define the minimum side length of the square
def min_side_length : ℝ := 2 * Real.sqrt 2 + 2

-- Definition to state that the side length a of the square can contain 5 non-overlapping circles
def is_min_sufficient_square (a : ℝ) : Prop :=
  -- Each circle has radius 1, and must fit entirely within the square
  (∀ (x₁ y₁ x₂ y₂ : ℝ), ((x₁ - x₂)^2 + (y₁ - y₂)^2 ≥ (2 * radius)^2)) ∧
  -- The minimum required side length to satisfy the non-overlapping circles
  a ≥ min_side_length

-- The main theorem stating the minimum side length required
theorem min_square_side_length_suffices_for_circles :
  ∃ (a : ℝ), is_min_sufficient_square a ∧ a = min_side_length := 
sorry

end min_square_side_length_suffices_for_circles_l425_425436


namespace kelsey_videos_watched_l425_425306

-- Definitions of conditions
def total_videos : ℕ := 411
def ekon_less : ℕ := 17
def kelsey_more : ℕ := 43

-- Variables representing videos watched by Uma, Ekon, and Kelsey
variables (U E K : ℕ)
hypothesis total_watched : U + E + K = total_videos
hypothesis ekon_watch : E = U - ekon_less
hypothesis kelsey_watch : K = E + kelsey_more

-- The statement to be proved
theorem kelsey_videos_watched : K = 160 :=
by sorry

end kelsey_videos_watched_l425_425306


namespace a_squared_plus_b_squared_equals_61_l425_425147

theorem a_squared_plus_b_squared_equals_61 (a b : ℝ) (h1 : a + b = -9) (h2 : a = 30 / b) : a^2 + b^2 = 61 :=
sorry

end a_squared_plus_b_squared_equals_61_l425_425147


namespace perimeter_paper_count_l425_425153

theorem perimeter_paper_count (n : Nat) (h : n = 10) : 
  let top_side := n
  let right_side := n - 1
  let bottom_side := n - 1
  let left_side := n - 2
  top_side + right_side + bottom_side + left_side = 36 :=
by
  sorry

end perimeter_paper_count_l425_425153


namespace number_of_correct_statements_is_zero_l425_425732

theorem number_of_correct_statements_is_zero :
  (¬(∀ m : ℝ, m is rational) ∧
   (∀ a b : ℝ, a > b → a^2 > b^2) ∧ 
   (∀ x : ℝ, x^2 - 2 * x - 3 = 0 → x = 3) ∧ 
   (∀ A B: Set, A ∩ B = B → A = ∅)) → 
  0 = 0 := 
by sorry

end number_of_correct_statements_is_zero_l425_425732


namespace three_digit_perfect_squares_divisible_by_4_l425_425123

/-- Proving the number of three-digit perfect squares that are divisible by 4 is 6 -/
theorem three_digit_perfect_squares_divisible_by_4 : 
  (Finset.card (Finset.filter (λ n, n ∣ 4) (Finset.image (λ k, k * k) (Finset.Icc 10 31)))) = 6 :=
by
  sorry

end three_digit_perfect_squares_divisible_by_4_l425_425123


namespace quadrilateral_A0_B0_C0_D0_is_trapezoid_l425_425302

variable (A B C D A_1 B_1 C_1 D_1 A_0 B_0 C_0 D_0 : Type)
variables [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B]
  [AddCommGroup C] [Module ℝ C] [AddCommGroup D] [Module ℝ D]
  [AddCommGroup A_1] [Module ℝ A_1] [AddCommGroup B_1] [Module ℝ B_1]
  [AddCommGroup C_1] [Module ℝ C_1] [AddCommGroup D_1] [Module ℝ D_1]
  [AddCommGroup A_0] [Module ℝ A_0] [AddCommGroup B_0] [Module ℝ B_0]
  [AddCommGroup C_0] [Module ℝ C_0] [AddCommGroup D_0] [Module ℝ D_0]

noncomputable def is_trapezoid (quad : Type) : Prop :=
sorry -- detailed definition of a trapezoid

theorem quadrilateral_A0_B0_C0_D0_is_trapezoid 
  (ABCD : Type)
  (h1 : is_trapezoid ABCD)
  (h2 : parallel_has_property A B C D A_1 B_1 C_1 D_1)
  (h3 : equal_ratios A_0 B_0 C_0 D_0 AA_1 BB_1 CC_1 DD_1) :
  is_trapezoid A_0 B_0 C_0 D_0 ∧ (A_0 B_0 / C_0 D_0 = A B / C D) :=
by
  sorry

end quadrilateral_A0_B0_C0_D0_is_trapezoid_l425_425302


namespace count_three_digit_squares_divisible_by_4_l425_425111

def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

theorem count_three_digit_squares_divisible_by_4 : 
  let squares := {n : ℕ | is_three_digit n ∧ is_perfect_square n ∧ is_divisible_by_4 n} in
  Fintype.card squares = 10 :=
sorry

end count_three_digit_squares_divisible_by_4_l425_425111


namespace max_choir_members_l425_425639

theorem max_choir_members (k n m : ℕ) (h1 : m = k^2 + 6) (h2 : m = n * (n + 6)) : m ≤ 112 :=
begin
  sorry -- Proof steps go here
end

end max_choir_members_l425_425639


namespace min_sum_ab_l425_425454

theorem min_sum_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : a * b = a + b + 3) : 
  a + b ≥ 6 := 
sorry

end min_sum_ab_l425_425454


namespace exists_fivespecial_even_smallest_fourdigit_special_largest_special_largest_distinct_special_l425_425710

-- A predicate to check if a number is special
def is_special (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits ≠ [] ∧
  (¬ digits.contains 0) ∧
  ((bits_head digits * 2) = digits.sum)

-- Part (a): Prove that there exists a five-digit even special number
theorem exists_fivespecial_even : ∃ n : ℕ, (50000 ≤ n ∧ n < 100000) ∧ (n % 2 = 0) ∧ is_special n := 
  sorry

-- Part (b): Prove that the smallest four-digit special number is 3111
theorem smallest_fourdigit_special: ∀ n : ℕ, is_special n → (1000 ≤ n ∧ n < 10000) → 3111 ≤ n := 
  sorry

-- Part (c): Prove that the largest special number is 9111111111
theorem largest_special: ∀ n : ℕ, is_special n → n ≤ 9111111111 := 
  sorry

-- Part (d): Prove that the largest special number with all distinct digits is 9621
theorem largest_distinct_special: ∀ n : ℕ, is_special n → (∀ digits : List ℕ, digits.nodup) → n ≤ 9621 := 
  sorry

end exists_fivespecial_even_smallest_fourdigit_special_largest_special_largest_distinct_special_l425_425710


namespace squirrel_walnut_count_l425_425699

-- Lean 4 statement
theorem squirrel_walnut_count :
  let initial_boy_walnuts := 12
  let gathered_walnuts := 6
  let dropped_walnuts := 1
  let initial_girl_walnuts := 0
  let brought_walnuts := 5
  let eaten_walnuts := 2
  (initial_boy_walnuts + gathered_walnuts - dropped_walnuts + initial_girl_walnuts + brought_walnuts - eaten_walnuts) = 20 :=
by
  -- Proof goes here
  sorry

end squirrel_walnut_count_l425_425699


namespace a_plus_b_l425_425462

open Real

-- given condition
def series_sum : ℝ := ∑' i : ℕ, (sin^2 (10 / 3^i * π / 180) / cos (30 / 3^i * π / 180))

-- main statement
theorem a_plus_b (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : (1 : ℝ) / (a + sqrt b) = series_sum) : a + b = 15 :=
sorry

end a_plus_b_l425_425462


namespace periodic_decimal_is_rational_l425_425964

theorem periodic_decimal_is_rational
  (a : ℕ → ℤ)
  (N n : ℕ)
  (h1 : ∀ i < N, 0 ≤ a i ∧ a i ≤ 9)
  (h2 : ∀ j ≥ N, a j = a (N + (j - N) % n)) :
  ∃ p q : ℤ, q ≠ 0 ∧ (∑ i in Finset.range (N + n), a i * 10^(-(i : ℤ)) : ℚ) = p / q := by
  sorry

end periodic_decimal_is_rational_l425_425964


namespace factorize_expression_l425_425424

theorem factorize_expression (a : ℝ) : a^2 + 5 * a = a * (a + 5) :=
sorry

end factorize_expression_l425_425424


namespace circles_area_difference_l425_425681

noncomputable def circumference_to_radius (C : ℝ) : ℝ :=
  C / (2 * Real.pi)

noncomputable def circle_area (r : ℝ) : ℝ :=
  Real.pi * r ^ 2

theorem circles_area_difference (C1 C2 : ℝ) (hC1 : C1 = 264) (hC2 : C2 = 352) : 
  (circle_area (circumference_to_radius C2)) - (circle_area (circumference_to_radius C1)) ≈ 4312.73 :=
by
  sorry

end circles_area_difference_l425_425681


namespace functional_eq_solution_l425_425791

theorem functional_eq_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x) →
  (∀ x : ℝ, f x = x) :=
by
  intro h
  sorry

end functional_eq_solution_l425_425791


namespace intervals_equinumerous_l425_425232

-- Definitions and statements
theorem intervals_equinumerous (a : ℝ) (h : 0 < a) : 
  ∃ (f : Set.Icc 0 1 → Set.Icc 0 a), Function.Bijective f :=
by
  sorry

end intervals_equinumerous_l425_425232


namespace sequence_sum_equals_power_l425_425577

theorem sequence_sum_equals_power (n : ℕ) (hpos : n > 0) : 
  let y : ℕ → ℕ
      | 0     := 0
      | 1     := 1
      | (k+2) := (n - 2) * y (k + 1) - (n - k - 1) * y k / (k + 1) in
  Finset.sum (Finset.range n) y = 2^(n - 2) :=
sorry

end sequence_sum_equals_power_l425_425577


namespace max_handshakes_in_club_l425_425738

theorem max_handshakes_in_club (gentlemen : Finset Person) (H : gentlemen.card = 20) 
(ap : ∀ (A B C : Person), A ∈ gentlemen → B ∈ gentlemen → C ∈ gentlemen → 
acquainted A B → acquainted B C → acquainted C A → false) : 
∃ (max_handshakes : ℕ), max_handshakes = 100 := 
sorry

/--
The problem: 
The maximum number of handshakes in a club of 20 gentlemen where no three gentlemen are mutually acquainted is 100.
--/
noncomputable def max_handshakes := 100 -- Solution

end max_handshakes_in_club_l425_425738


namespace subset_A_B_l425_425487

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 2} -- Definition of set A
def B (a : ℝ) := {x : ℝ | x > a} -- Definition of set B

theorem subset_A_B (a : ℝ) : a < 1 → A ⊆ B a :=
by
  sorry

end subset_A_B_l425_425487


namespace problem_statement_l425_425439

noncomputable def a : ℝ := (real.sqrt 15)^2 * 8^3 / real.log 256 + real.sin (real.pi / 4)

theorem problem_statement : a ≈ 1385.3225 := sorry

end problem_statement_l425_425439


namespace no_positive_integers_satisfy_log_condition_l425_425442

theorem no_positive_integers_satisfy_log_condition :
  ∀ x : ℕ, 30 < x ∧ x < 80 → ¬ (log 2 (x - 30) + log 2 (80 - x) < 3) := 
by {
  intros x h,
  exact sorry,
}

end no_positive_integers_satisfy_log_condition_l425_425442


namespace rectangle_count_horizontal_vertical_l425_425869

theorem rectangle_count_horizontal_vertical :
  ∀ (h_strips : ℕ) (v_strips : ℕ) (intersection : ℕ), 
  h_strips = 15 → v_strips = 10 → intersection = 1 → 
  (h_strips + v_strips - intersection = 24) :=
by
  intros h_strips v_strips intersection h_strips_def v_strips_def intersection_def
  rw [h_strips_def, v_strips_def, intersection_def]
  sorry

end rectangle_count_horizontal_vertical_l425_425869


namespace smallest_b_for_factorization_l425_425434

theorem smallest_b_for_factorization : ∃ (b : ℕ), (∀ p q : ℤ, (x^2 + (b * x) + 2352) = (x + p) * (x + q) → p + q = b ∧ p * q = 2352) ∧ b = 112 := 
sorry

end smallest_b_for_factorization_l425_425434


namespace M_subset_N_l425_425100

def M : Set ℝ := {x | ∃ k : ℤ, x = (k / 2) * 180 + 45}
def N : Set ℝ := {x | ∃ k : ℤ, x = (k / 4) * 180 + 45}

theorem M_subset_N : M ⊆ N :=
sorry

end M_subset_N_l425_425100


namespace count_oddly_powerful_under_5000_l425_425407

open Nat

def is_oddly_powerful (n : ℕ) : Prop :=
  ∃ a b : ℕ, b > 1 ∧ Odd b ∧ a^b = n

noncomputable def oddly_powerful_under_5000 : List ℕ :=
  (List.range 5000).filter is_oddly_powerful

theorem count_oddly_powerful_under_5000 :
  (oddly_powerful_under_5000.length = 20) :=
by
  sorry

end count_oddly_powerful_under_5000_l425_425407


namespace find_l2_and_l3_l425_425718

noncomputable def point := ℝ × ℝ

structure Ray :=
  (start : point)
  (end : point)

def reflect_across_x_axis (p : point) : point :=
  (p.1, -p.2)

def reflect_across_line (line_eq : point → Prop) (p : point) : point :=
  let x := p.1
  let y := p.2
  let y' := y - (2 * line_eq (x, y)) / (1 + 1)
  in  (x, y')

def line_equation (p1 p2 : point) : ℝ → ℝ :=
  λ x, p2.2 + (p2.2 - p1.2) / (p2.1 - p1.1) * (x - p2.1)

theorem find_l2_and_l3 :
  let M := (-1, 3)
  let P := (1, 0)
  let l1 := Ray.mk M P
  let M' := reflect_across_x_axis M
  let l2_eq := line_equation M' P  -- y = 3/2 * (x - 1)
  (l2_eq 0) = 3/2 * (0 - 1) →
  (l2_eq 2) = 3/2 * (2 - 1) →
  let N := (11/5, 9/5)
  let P' := (4, 3)
  let l3_eq := line_equation N P'  -- 2x - 3y + 1 = 0
  (l3_eq 0) = 2 * 0 - 3 * 8/5 + 1 →
  (l3_eq 5) = 2 * 5 - 3 * 2 + 1 →
  sorry

end find_l2_and_l3_l425_425718


namespace part_I_part_II_l425_425051

-- Definitions
def seq_a (n : ℕ) : ℕ :=
  if n = 1 then 2 else 4 * seq_a (n - 1)

def sum_S (n : ℕ) : ℕ :=
  if n = 1 then 2 else seq_a n + sum_S (n - 1)

def seq_b (n : ℕ) : ℤ :=
  (2 * n - 1 : ℕ)

def T (n : ℕ) : ℕ :=
  (∑ i in finset.range (n + 1), seq_b i)

-- Part (I)
theorem part_I (n : ℕ) (hn : n ≥ 1) : seq_a n = 2^(2*n - 1) := by
  sorry

-- Part (II)
theorem part_II (n : ℕ) : (∑ i in finset.range (n + 1), 1 / T i) < 2 := by
  sorry

end part_I_part_II_l425_425051


namespace total_players_l425_425337

theorem total_players (K Kho B : ℕ) : K = 10 → Kho = 15 → B = 5 → (K + Kho - B) = 20 := by
  intros hK hKho hB
  rw [hK, hKho, hB]
  rfl

end total_players_l425_425337


namespace kelsey_video_count_l425_425305

variable (E U K : ℕ)

noncomputable def total_videos : ℕ := 411
noncomputable def ekon_videos : ℕ := E
noncomputable def uma_videos : ℕ := E + 17
noncomputable def kelsey_videos : ℕ := E + 43

theorem kelsey_video_count (E U K : ℕ) 
  (h1 : total_videos = ekon_videos + uma_videos + kelsey_videos)
  (h2 : uma_videos = ekon_videos + 17)
  (h3 : kelsey_videos = ekon_videos + 43)
  : kelsey_videos = 160 := 
sorry

end kelsey_video_count_l425_425305


namespace positional_relationship_l425_425148

open Set

variable (Point Line Plane : Type) 

-- Definitions for parallel and subset relations
def is_parallel (l1 l2 : Line) : Prop := sorry
def is_subset (l : Line) (α : Plane) : Prop := sorry

variable (l m : Line) (α : Plane)

-- Definitions for the parallel line in plane condition
axiom l_parallel_m : is_parallel l m
axiom m_in_alpha : is_subset m α

-- Theorem statement to be proven
theorem positional_relationship : is_subset l α ∨ is_parallel l α := sorry

end positional_relationship_l425_425148


namespace oddly_powerful_less_than_5000_count_l425_425402

noncomputable def is_oddly_powerful (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 1 ∧ b % 2 = 1 ∧ a^b = n

theorem oddly_powerful_less_than_5000_count : 
  ({ n : ℕ | n < 5000 ∧ is_oddly_powerful n }.card = 24) :=
sorry

end oddly_powerful_less_than_5000_count_l425_425402


namespace find_integer_for_prime_l425_425427

def is_prime (n : ℤ) : Prop :=
  n > 1 ∧ ∀ m : ℤ, m > 0 → m ∣ n → m = 1 ∨ m = n

theorem find_integer_for_prime (n : ℤ) :
  is_prime (4 * n^4 + 1) ↔ n = 1 :=
by
  sorry

end find_integer_for_prime_l425_425427


namespace polynomial_roots_l425_425768

theorem polynomial_roots (α : ℝ) : 
  (α^2 + α - 1 = 0) → (α^3 - 2 * α + 1 = 0) :=
by sorry

end polynomial_roots_l425_425768


namespace train_travel_distance_l425_425359

def time_interval_minutes : ℝ := 1 + 45 / 60 -- 1 minute 45 seconds
def time_interval_hours : ℝ := time_interval_minutes / 60 -- Convert to hours (about 0.0292 hours)
def total_time_hours : ℝ := 45 / 60 -- 45 minutes in hours (0.75 hours)

theorem train_travel_distance :
  (total_time_hours / time_interval_hours).floor * 1 = 25 :=
by
  sorry

end train_travel_distance_l425_425359


namespace find_S_l425_425512

theorem find_S (R S : ℕ) (h1 : 111111111111 - 222222 = (R + S) ^ 2) (h2 : S > 0) :
  S = 333332 := 
sorry

end find_S_l425_425512


namespace seq_inequality_l425_425184

noncomputable def seq_a (A : ℕ) (a : ℕ → ℕ) : ℕ → ℕ
  | 0     => A^A
  | (n+1) => A^(a n)

noncomputable def seq_b (A : ℕ) (b : ℕ → ℕ) : ℕ → ℕ
  | 0     => A^(A+1)
  | (n+1) => 2^(b n)

theorem seq_inequality (A : ℕ) (hA : A > 1) (a b : ℕ → ℕ) :
  (∀ n, a = seq_a A a ∧ b = seq_b A b) →
  ∀ n, a n < b n :=
sorry

end seq_inequality_l425_425184


namespace correct_number_of_assertions_l425_425381

-- Definitions for the conditions
def assertion1 (a b : ℝ) : Prop :=
  (a + b) ^ 0 = 1

def assertion2 (a b : ℝ) (h1 : a < 0) (h2 : -1 < b ∧ b < 0) : Prop :=
  a * b > a * b^4 ∧ a * b^4 > a * b^2

def assertion3 (A B C D : Type) [quadrilateral A B C D]
  (S : A → B → C → ℝ) (h : equal_area_division_of_diagonals A B C D) : Prop :=
  is_parallelogram A B C D

def assertion4 (P : Type) (A B C D : rectangle) (E F G H : Type)
  (h1 : reflection P A B C D E F G H) : Prop :=
  area E F G H = 2 * area A B C D

-- Actual Theorem Statement
theorem correct_number_of_assertions {a b : ℝ} 
  (h1 : a < 0) (h2 : -1 < b ∧ b < 0)
  (A B C D : Type) [quadrilateral A B C D]
  (S : A → B → C → ℝ) (h3 : equal_area_division_of_diagonals A B C D)
  (P : Type) (rectangle : rectangle P A B C D) (E F G H : Type)
  (h4 : reflection P A B C D E F G H) :
  (¬ assertion1 a b) ∧ assertion2 a b h1 h2 ∧ assertion3 A B C D S h3 ∧ ¬ assertion4 P rectangle E F G H → 
  (number_of_true_assertions = 2) :=
sorry

end correct_number_of_assertions_l425_425381


namespace find_a2018_l425_425550

-- Definitions based on given conditions
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 0.5 ∧ ∀ n, a (n + 1) = 1 - 1 / (a n)

-- The statement to prove
theorem find_a2018 (a : ℕ → ℝ) (h : seq a) : a 2018 = -1 := by
  sorry

end find_a2018_l425_425550


namespace total_valid_numbers_l425_425947

def is_triangle (a b c : ℕ) : Prop :=
a + b > c ∧ a + c > b ∧ b + c > a

def is_isosceles (a b c : ℕ) : Prop :=
(a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

def is_equilateral (a b c : ℕ) : Prop :=
a = b ∧ b = c

def is_valid_digit (n : ℕ) : Prop :=
1 ≤ n ∧ n ≤ 9

def count_valid_numbers : ℕ :=
count (λ n : ℕ, let a := n / 100,
                    b := (n / 10) % 10,
                    c := n % 10 in
                is_valid_digit a ∧
                is_valid_digit b ∧
                is_valid_digit c ∧
                (is_triangle a b c) ∧
                (is_equilateral a b c ∨ is_isosceles a b c)) (range 1000)

theorem total_valid_numbers : count_valid_numbers = 165 :=
begin
  sorry
end

end total_valid_numbers_l425_425947


namespace annas_mom_money_l425_425386

theorem annas_mom_money :
  let cost_gum := 3 * 1
  let cost_chocolate := 5 * 1
  let cost_candy_canes := 2 * 0.5
  let total_spent := cost_gum + cost_chocolate + cost_candy_canes
  let money_left := 1
  total_spent + money_left = 10 :=
by
  let cost_gum := 3 * 1
  let cost_chocolate := 5 * 1
  let cost_candy_canes := 2 * 0.5
  let total_spent := cost_gum + cost_chocolate + cost_candy_canes
  let money_left := 1
  show total_spent + money_left = 10
  sorry

end annas_mom_money_l425_425386


namespace least_possible_xy_l425_425062

theorem least_possible_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 48 :=
by
  sorry

end least_possible_xy_l425_425062


namespace determine_exact_masses_with_4_weighings_l425_425331

-- Declare the masses of the four weights (in grams)
def masses : List ℕ := [1001, 1002, 1004, 1005]

-- Define a struct representing the balance scale
structure BalanceScale (α : Type) :=
  (weigh : α → α → comparison) -- comparison indicates whether the first α is heavier, equal, or lighter than the second α

-- Define the comparison type
inductive comparison
| heavier
| equal
| lighter

-- Define lean statement
theorem determine_exact_masses_with_4_weighings (scale : BalanceScale (List ℕ)) :
  (∀ (weighings : ℕ), weighings ≤ 4 → 
    ∃ (determination : (masses → masses) → Prop), 
      determination = λ f, ∀ perm, masses = perm.sort (≤) → f perm = masses) :=
sorry

end determine_exact_masses_with_4_weighings_l425_425331


namespace parabola_coefficients_l425_425985

theorem parabola_coefficients (a b c : ℝ)
  (h1 : ∃ k : ℝ, ∀ x : ℝ, (k = (4 - 4)^2 + 3) → (a * (x - 4)^2 + b * (x - 4) + c = (a * 0 + b * 0 + c = k) ∧ k = 3))
  (h2 : a * (2 - 4)^2 + b * (2 - 4) + c = 7) :
  a = 1 ∧ b = -8 ∧ c = 19 := by
  sorry

end parabola_coefficients_l425_425985


namespace geometric_progression_problem_l425_425301

theorem geometric_progression_problem :
  ∃ a q : ℤ, let b := a * q, c := a * q^2 in
    (q ≠ 1 ∧ a * (q - 1)^2 = 16 ∧ 16 * q = 48) ∧
    (a = 4 ∧ b = 12 ∧ c = 36) :=
by
  sorry

end geometric_progression_problem_l425_425301


namespace minimize_difference_2210_l425_425872

open Nat

theorem minimize_difference_2210 :
  ∃ (a b : ℕ), a * b = 2210 ∧ (∃ p : ℕ, p.Prime ∧ p ∣ 2210 ∧ (p ∣ a ∨ p ∣ b)) ∧ 
  ∀ (c d : ℕ), c * d = 2210 → (∃ q : ℕ, q.Prime ∧ q ∣ 2210 ∧ (q ∣ c ∨ q ∣ d)) → |a - b| ≤ |c - d| ∧ |a - b| = 31 :=
begin
  sorry
end

end minimize_difference_2210_l425_425872


namespace new_person_weight_l425_425535

theorem new_person_weight
  (initial_avg_weight : ℝ := 57)
  (num_people : ℕ := 8)
  (weight_to_replace : ℝ := 55)
  (weight_increase_first : ℝ := 1.5)
  (weight_increase_second : ℝ := 2)
  (weight_increase_third : ℝ := 2.5)
  (weight_increase_fourth : ℝ := 3)
  (weight_increase_fifth : ℝ := 3.5)
  (weight_increase_sixth : ℝ := 4)
  (weight_increase_seventh : ℝ := 4.5) :
  ∃ x : ℝ, x = 67 :=
by
  sorry

end new_person_weight_l425_425535


namespace f_of_2_minus_f_of_1_l425_425475

-- Define the conditions: the function and the point it passes through
variable (a : ℝ) (f : ℝ → ℝ)
axiom power_function (h : f(x) = x^a) : (f 9 = 3)

-- The goal is to prove that f(2) - f(1) = √2 - 1
theorem f_of_2_minus_f_of_1 (h : f 9 = 3) : f 2 - f 1 = Real.sqrt 2 - 1 :=
sorry

end f_of_2_minus_f_of_1_l425_425475


namespace thickness_and_width_l425_425746
noncomputable def channelThicknessAndWidth (L W v₀ h₀ θ g : ℝ) : ℝ × ℝ :=
let K := W * h₀ * v₀
let v := v₀ + Real.sqrt (2 * g * Real.sin θ * L)
let x := K / (v * W)
let y := K / (h₀ * v)
(x, y)

theorem thickness_and_width :
  channelThicknessAndWidth 10 3.5 1.4 0.4 (12 * Real.pi / 180) 9.81 = (0.072, 0.629) :=
by
  sorry

end thickness_and_width_l425_425746


namespace y_intercept_lineb_l425_425207

-- Define the given conditions
def line1 := λ x: ℝ, -3 * x + 6  -- Line y = -3x + 6

def is_parallel (linea lineb: ℝ → ℝ) : Prop :=
  (∃ m b1 b2, linea = (λ x, m * x + b1) ∧ lineb = (λ x, m * x + b2))

def passes_through (line: ℝ → ℝ) (p: ℝ × ℝ) : Prop :=
  line p.1 = p.2

-- Define line b
def lineb (x: ℝ) := -3 * x + 10

-- The theorem to be proved
theorem y_intercept_lineb :
  is_parallel line1 lineb ∧ passes_through lineb (↑3, ↑1) → 
  (∃ b: ℝ, lineb = (λ x, -3 * x + b) ∧ b = 10) :=
by
  sorry

end y_intercept_lineb_l425_425207


namespace math_problem_l425_425766

def op (a b : ℕ) : ℕ := (a + 2) * (b + 2) - 2

noncomputable def op_multiple : list ℕ → ℕ 
| []         := 0
| [x]        := x
| (x :: xs)  := (x + 2) * (op_multiple xs + 2) - 2

theorem math_problem : 
  (1 * 3 * 5 * 7 * 9 * 11 * 13) - op_multiple [1, 3, 5, 7, 9, 11, 13] = 2 :=
by
  have h : op (op (op (op (op (op 1 3) 5) 7) 9) 11) 13 = 
    (1 + 2) * (3 + 2) * (5 + 2) * (7 + 2) * (9 + 2) * (11 + 2) * (13 + 2) - 2, 
    from sorry,
  simp only [op, h],
  rw [mul_assoc, mul_assoc (3 * 5)],
  -- Further computations or simplification steps would go here. 
  sorry

end math_problem_l425_425766


namespace number_of_hydrogen_atoms_l425_425349

/-- 
A compound has a certain number of Hydrogen, 1 Chromium, and 4 Oxygen atoms. 
The molecular weight of the compound is 118. How many Hydrogen atoms are in the compound?
-/
theorem number_of_hydrogen_atoms
  (H Cr O : ℕ)
  (mw_H : ℕ := 1)
  (mw_Cr : ℕ := 52)
  (mw_O : ℕ := 16)
  (H_weight : ℕ := H * mw_H)
  (Cr_weight : ℕ := 1 * mw_Cr)
  (O_weight : ℕ := 4 * mw_O)
  (total_weight : ℕ := 118)
  (weight_without_H : ℕ := Cr_weight + O_weight) 
  (H_weight_calculated : ℕ := total_weight - weight_without_H) :
  H = 2 :=
  by
    sorry

end number_of_hydrogen_atoms_l425_425349


namespace proof_problem_l425_425742

-- Definitions based on the conditions
def proposition_P : Prop :=
  let f := λ x : ℝ, Real.exp (-x)
  let df := λ x : ℝ, -Real.exp (-x)
  tangent_line := λ x : ℝ, (-Real.exp 1) * (x + 1) + Real.exp (-1)
  tangent_line = λ x, -Real.exp 1 * x

def proposition_Q : Prop :=
  ∀ (f : ℝ → ℝ) (x_0 : ℝ), deriv f x_0 = 0 → (∀ x, deriv f x_0 = 0 ↔ ∀ x ≠ x_0, f x ≠ f x_0)

theorem proof_problem : proposition_P ∨ proposition_Q :=
sorry

end proof_problem_l425_425742


namespace stamps_count_l425_425603

theorem stamps_count {x : ℕ} (h1 : x % 3 = 1) (h2 : x % 5 = 3) (h3 : x % 7 = 5) (h4 : 150 < x ∧ x ≤ 300) :
  x = 208 :=
sorry

end stamps_count_l425_425603


namespace perimeter_square_III_l425_425981
-- Import the necessary Lean 4 math library

-- Define the given conditions.
def square_I_perimeter : ℝ := 16
def square_II_perimeter : ℝ := 36
def square_side_length (perimeter : ℝ) : ℝ := perimeter / 4

-- Calculate side lengths of squares I and II
def side_I : ℝ := square_side_length square_I_perimeter
def side_II : ℝ := square_side_length square_II_perimeter

-- Define the geometric mean function.
def geometric_mean (a b : ℝ) := real.sqrt (a * b)

-- Calculate the side length of square III.
def side_III : ℝ := geometric_mean side_I side_II

-- Define perimeter of square III.
def square_perimeter (side : ℝ) : ℝ := 4 * side

-- State the theorem that needs to be proved.
theorem perimeter_square_III : square_perimeter side_III = 24 := by
  sorry

end perimeter_square_III_l425_425981


namespace exists_a_b_c_l425_425564

variables (M : Finset ℕ) (A B C : Finset ℕ) (n : ℕ)

def is_partition (A B C M : Finset ℕ) : Prop :=
  M = {1, 2, ..., 3 * n} ∧
  A ∪ B ∪ C = M ∧
  disjoint A B ∧ disjoint B C ∧ disjoint A C ∧
  A.card = n ∧ B.card = n ∧ C.card = n

theorem exists_a_b_c (h : is_partition A B C M) :
  ∃ a ∈ A, ∃ b ∈ B, ∃ c ∈ C, (a = b + c ∨ b = c + a ∨ c = a + b) := 
sorry

end exists_a_b_c_l425_425564


namespace minimum_value_am_bn_l425_425458

theorem minimum_value_am_bn (a b m n : ℝ) (hp_a : a > 0)
    (hp_b : b > 0) (hp_m : m > 0) (hp_n : n > 0) (ha_b : a + b = 1)
    (hm_n : m * n = 2) :
    (am + bn) * (bm + an) ≥ 3/2 := by
  sorry

end minimum_value_am_bn_l425_425458


namespace volume_elliptical_cylinder_l425_425432

-- Defining the side lengths of the rectangle
def longer_side : ℝ := 20
def shorter_side : ℝ := 10

-- The volume of the elliptical cylinder formed by rotating the rectangle about its longer side
theorem volume_elliptical_cylinder : 
  let minor_radius := shorter_side / 2,
      height := longer_side,
      volume := Real.pi * (minor_radius^2) * height
  in volume = 500 * Real.pi :=
by
  let minor_radius := shorter_side / 2
  let height := longer_side
  let volume := Real.pi * (minor_radius^2) * height
  have : minor_radius = 5, by
    rw [shorter_side, div_eq_mul_inv, mul_inv, inv_eq_one_div, one_div]
    sorry -- Calculation omitted
  have : height = 20, by
    rw [longer_side]
    sorry -- Calculation omitted
  have : volume = Real.pi * (5^2) * 20, by
    rw [this, this]
  have : volume = 500 * Real.pi, by
    sorry -- Calculation omitted
  exact this

end volume_elliptical_cylinder_l425_425432


namespace _l425_425198

noncomputable theorem polynomial_divisible_by_3 {P : ℤ[X]} (n : ℤ) (h0 : (P.eval n) % 3 = 0)
  (h1 : (P.eval (n+1)) % 3 = 0) (h2 : (P.eval (n+2)) % 3 = 0) :
  ∀ m : ℤ, (P.eval m) % 3 = 0 := 
sorry

end _l425_425198


namespace arithmetic_square_root_of_100_eq_sqrt_10_l425_425990

def arithmetic_square_root (x : ℝ) : ℝ := real.sqrt (real.sqrt x)

theorem arithmetic_square_root_of_100_eq_sqrt_10 :
  arithmetic_square_root 100 = real.sqrt 10 :=
by
  sorry

end arithmetic_square_root_of_100_eq_sqrt_10_l425_425990


namespace cubic_yard_to_cubic_meter_and_liters_l425_425498

theorem cubic_yard_to_cubic_meter_and_liters :
  (1 : ℝ) * (0.9144 : ℝ)^3 = 0.764554 ∧ 0.764554 * 1000 = 764.554 :=
by
  sorry

end cubic_yard_to_cubic_meter_and_liters_l425_425498


namespace geometric_probability_model_l425_425968

noncomputable def probability_event (a b c : ℝ) (P : ℝ → Prop) :=
  ∫ x in a..b, if P x then 1 else 0

theorem geometric_probability_model :
  probability_event (-1) 2 (λ x, |x| ≤ 1) = 2 / 3 := 
sorry

end geometric_probability_model_l425_425968


namespace median_length_A_l425_425488

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def midpoint (p1 p2 : Point3D) : Point3D :=
  { x := (p1.x + p2.x) / 2,
    y := (p1.y + p2.y) / 2,
    z := (p1.z + p2.z) / 2 }

def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2 + (p2.z - p1.z) ^ 2)

def A : Point3D := { x := 2, y := -1, z := 4 }
def B : Point3D := { x := 3, y := 2, z := -6 }
def C : Point3D := { x := 5, y := 0, z := 2 }

def M : Point3D := midpoint B C

theorem median_length_A : distance A M = 2 * Real.sqrt 11 := 
  sorry

end median_length_A_l425_425488


namespace find_a9_l425_425168

-- Define the arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
def a_n : ℕ → ℝ := sorry   -- The sequence itself is unknown initially.

axiom a3 : a_n 3 = 5
axiom a4_a8 : a_n 4 + a_n 8 = 22

theorem find_a9 : a_n 9 = 41 :=
by
  sorry

end find_a9_l425_425168


namespace second_part_lending_time_l425_425376

theorem second_part_lending_time :
  ∀ (x y : ℝ) (P : x = 1025 ∧ y = 3 ∧ 2665 - x = 1640 ∧ 0.24 * x = 82 * y),
  y = 3 :=
by
  intros x y hP
  cases hP with hx hy
  cases hy with hx_calc hy_proof
  exact hy_proof.2.2.2

end second_part_lending_time_l425_425376


namespace number_of_quadrilaterals_even_l425_425411

/-- A structure representing the conditions of the problem -/
structure PolygonConditions (k : ℕ) :=
  (n : ℕ := 4 * k + 3)      -- The number of vertices of the polygon
  (convex : Prop)           -- The polygon is convex
  (no_three_diagonals_intersect : Prop) -- No three diagonals intersect at a single point
  (point_p_inside : Prop)   -- Point P is inside but not on diagonals

/-- The main theorem stating that the number of quadrilaterals containing point P is even -/
theorem number_of_quadrilaterals_even (k n : ℕ) (conditions : PolygonConditions k) :
  ∃ m : ℕ, (4 * k + 3) = n ∧ 2 * m = sorry :=
by
  obtain ⟨m, hm⟩ := nat.exists_eq_add_of_le (lt_of_le_of_lt (nat.zero_le k) (lt_succ_self m))
  exact ⟨m, hm⟩
  sorry

end number_of_quadrilaterals_even_l425_425411


namespace five_x_plus_four_is_25_over_7_l425_425513

theorem five_x_plus_four_is_25_over_7 (x : ℚ) (h : 5 * x - 8 = 12 * x + 15) : 5 * (x + 4) = 25 / 7 := by
  sorry

end five_x_plus_four_is_25_over_7_l425_425513


namespace smallest_value_abs_w3_plus_z3_l425_425452

noncomputable def smallest_value_w3_plus_z3 (w z : ℂ) := abs (w^3 + z^3)

theorem smallest_value_abs_w3_plus_z3 (w z : ℂ) (h1 : abs (w + z) = 2) (h2 : abs (w^2 + z^2) = 8) :
  smallest_value_w3_plus_z3 w z = 20 :=
sorry

end smallest_value_abs_w3_plus_z3_l425_425452


namespace correct_conclusion_ABC_l425_425321

-- Definitions related to the correlation coefficient r and its properties
noncomputable def linear_correlation_strong (r : ℝ) : Prop :=
  |r| ≃ 1
  
noncomputable def linear_correlation_weak (r : ℝ) : Prop :=
  |r| ≃ 0

-- Definition that means (x, y) lie on the regression line
noncomputable def mean_on_regression_line (x_bar y_bar : ℝ) (regression_line : ℝ → ℝ) : Prop :=
  regression_line x_bar = y_bar

-- Definition about the fit of the model from the residual plot
noncomputable def good_fit_model (residuals : list ℝ) : Prop :=
  -- a simple criterion based on the width of the horizontal band
  let horizontal_band_width := max (list.sum residuals) (list.length residuals) in
  horizontal_band_width < some_threshold

-- Theorem that confirms the correct answer ABC given the conditions
theorem correct_conclusion_ABC 
  (r : ℝ)
  (x_bar y_bar : ℝ)
  (regression_line : ℝ → ℝ)
  (residuals : list ℝ)
  (correct_answer : string)
  (h1 : linear_correlation_strong r)
  (h2 : linear_correlation_weak r)
  (h3 : mean_on_regression_line x_bar y_bar regression_line)
  (h4 : good_fit_model residuals) :
  correct_answer = "ABC" :=
sorry

end correct_conclusion_ABC_l425_425321


namespace review_meeting_probability_l425_425364

noncomputable def probability_successful_meeting : ℝ :=
1 / 2

theorem review_meeting_probability :
  let x y z : ℝ := sorry in
  (0 ≤ x ∧ x ≤ 2.5) →
  (0 ≤ y ∧ y ≤ 2.5) →
  (0 ≤ z ∧ z ≤ 2.5) →
  (-0.5 ≤ z - x ∧ z - x ≤ 0.5) →
  (-0.5 ≤ z - y ∧ z - y ≤ 0.5) →
  let V_cube := 15.625 in
  let V_R := V_cube / 2 in
  V_R / V_cube = probability_successful_meeting :=
by
  intros x y z h1 h2 h3 h4 h5
  sorry

end review_meeting_probability_l425_425364


namespace solve_equation_l425_425333

theorem solve_equation : 361 + 2 * 19 * 6 + 36 = 625 := by
  sorry

end solve_equation_l425_425333


namespace minimal_polynomial_correctness_l425_425029

noncomputable def minimal_polynomial (a b c d : ℚ) : Polynomial ℚ :=
  (Polynomial.X - Polynomial.C a) * (Polynomial.X - Polynomial.C b) *
  (Polynomial.X - Polynomial.C c) * (Polynomial.X - Polynomial.C d)

theorem minimal_polynomial_correctness :
  minimal_polynomial (2 + Real.sqrt 5) (2 - Real.sqrt 5) (3 + Real.sqrt 7) (3 - Real.sqrt 7) =
    Polynomial.X ^ 4 - 10 * Polynomial.X ^ 3 + 29 * Polynomial.X ^ 2 - 26 * Polynomial.X + 2 :=
by sorry

end minimal_polynomial_correctness_l425_425029


namespace sum_digits_inequality_a_sum_digits_inequality_b_max_c_k_correct_l425_425676

-- Define the sum of digits function and its properties
def S (N : ℕ) : ℕ := sorry

axiom S_add {A B : ℕ} : S (A + B) ≤ S A + S B
axiom S_sum {n : ℕ} {A₁ A₂ : fin n → ℕ} : S (finset.univ.sum A₁) ≤ (finset.univ.sum (λ i, S (A₁ i)))
axiom S_scale {n A : ℕ} : S (n * A) ≤ n * S A
axiom S_mul {A B : ℕ} : S (A * B) ≤ S A * S B

-- Part (a): The sum of the digits of K does not exceed 8 times the sum of the digits of 8K.
theorem sum_digits_inequality_a (K : ℕ) : S K ≤ 8 * S (8 * K) := sorry

-- Part (b): For k of the form 2^r 5^q, there exists c_k such that S(kN) / S(N) ≥ c_k for all N, 
-- and determine the largest suitable value for c_k.
theorem sum_digits_inequality_b (k : ℕ) (exists_c_k : ∃ r q, k = 2^r * 5^q) : 
  ∃ c_k > 0, ∀ N : ℕ, S (k * N) / S N ≥ c_k := sorry

-- Maximum value of c_k
def max_c_k (k : ℕ) (exists_c_k : ∃ r q, k = 2^r * 5^q) : ℝ := 1 / S (2^q * 5^r)

theorem max_c_k_correct (k : ℕ) (exists_c_k : ∃ r q, k = 2^r * 5^q) : 
  ∃ c_k = max_c_k k exists_c_k, ∀ N, S (k * N) / S N ≥ c_k := sorry

end sum_digits_inequality_a_sum_digits_inequality_b_max_c_k_correct_l425_425676


namespace diagonals_intersect_incenter_l425_425563

open EuclideanGeometry

variables {A₁ A₂ A₃ A₄ M B₁ B₂ B₃ B₄ : Point}

def is_incenter (M : Point) (A₁ A₂ A₃ A₄ : Point) : Prop :=
  -- Definition of incenter is needed here

def is_perpendicular (A M : Point) (g : Line) : Prop :=
  -- Definition of a line perpendicular to segment

theorem diagonals_intersect_incenter :
  ∀ (A₁ A₂ A₃ A₄ M B₁ B₂ B₃ B₄ : Point),
  is_incenter M A₁ A₂ A₃ A₄ →
  (∀ i, i ∈ {1, 2, 3, 4} → ∃ gi, is_perpendicular (list.nth_le [A₁, A₂, A₃, A₄] i _) M gi) →
  (B₁ = intersection (gi for g₁ through A₁) (gi for g₂ through A₂)) →
  (B₂ = intersection (gi for g₂ through A₂) (gi for g₃ through A₃)) →
  (B₃ = intersection (gi for g₃ through A₃) (gi for g₄ through A₄)) →
  (B₄ = intersection (gi for g₄ through A₄) (gi for g₁ through A₁)) →
  intersect_diagonals B₁ B₂ B₃ B₄ M :=
by
  sorry

end diagonals_intersect_incenter_l425_425563


namespace shawl_pricing_proof_l425_425741

-- Define the conditions and the necessary parameters
variables (y x s m v : ℝ)
variables (hy : y + 40 < s ∧ s < y + 50)
variables (hx : x + 30 < s ∧ s ≤ x + 40 - m)
variables (hm : m < 10)
variables (hro : 0.8s ≤ x + 20)
variables (hro2 : 0.8s ≤ y + 30)
variables (hros : y = 0.8s - 24)
variables (kicsi_shawl : y < 0.6s - 3 ∧ 0.6s - 3 < y + 10)
variables (kicsi_shawl2 : x - 10 < 0.6s - 3 ∧ 0.6s - 3 < x )
variables (expense : x + y - 1.2s = v)

-- Define the desired conclusion
theorem shawl_pricing_proof :
  91.16 ≤ s ∧ s ≤ 105.65 →
  0.4s + 24 + v = x ∧ 0.8s - 24 = y :=
begin
  sorry -- Proof steps are not required.
end

end shawl_pricing_proof_l425_425741


namespace probability_x_ge_1_l425_425149

theorem probability_x_ge_1 : 
  (∀ (x : ℝ), x ∈ interval (-1) 4 → ∃ P : ℝ, P = 3 / 5) :=
by 
  sorry

end probability_x_ge_1_l425_425149


namespace tammy_total_miles_l425_425986

noncomputable def miles_per_hour : ℝ := 1.527777778
noncomputable def hours_driven : ℝ := 36.0
noncomputable def total_miles := miles_per_hour * hours_driven

theorem tammy_total_miles : abs (total_miles - 55.0) < 1e-5 :=
by
  sorry

end tammy_total_miles_l425_425986


namespace min_value_xy_l425_425061

theorem min_value_xy (x y : ℕ) (h : 0 < x ∧ 0 < y) (cond : (1 : ℚ) / x + (1 : ℚ) /(3 * y) = 1 / 6) : 
  xy = 192 :=
sorry

end min_value_xy_l425_425061


namespace g_range_l425_425190

theorem g_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  let g := (λ (a b c : ℝ), (a / (a + b)) + (b / (b + c)) + (c / (c + a))) in
  1 < g a b c ∧ g a b c < 2 :=
sorry

end g_range_l425_425190


namespace count_windows_and_doors_in_palace_l425_425264

theorem count_windows_and_doors_in_palace (rooms : ℕ) (grid_size : ℕ) (external_window : ℕ) (internal_door : ℕ):
  rooms = 100 ∧ grid_size = 10 ∧ external_window = 1 ∧ internal_door = 1 →
  let external_windows := grid_size * 4 in
  let vertical_partitions := grid_size - 1 in
  let horizontal_partitions := grid_size - 1 in
  let total_partitions := vertical_partitions + horizontal_partitions in
  let total_doors := total_partitions * grid_size in
  external_windows = 40 ∧ total_doors = 180 :=
by
  intros h,
  rcases h with ⟨h1, h2, h3, h4⟩,
  have hw : 10 * 4 = 40 := by norm_num,
  have hv : 9 * 10 = 90 := by norm_num,
  have hh : 9 * 10 = 90 := by norm_num,
  have ht := hv + hh,
  split,
  { exact hw },
  { exact ht }

end count_windows_and_doors_in_palace_l425_425264


namespace coefficients_verification_l425_425818

theorem coefficients_verification :
  let a0 := -3
  let a1 := -13 -- Not required as part of the proof but shown for completeness
  let a2 := 6
  let a3 := 0 -- Filler value to ensure there is a6 value
  let a4 := 0 -- Filler value to ensure there is a6 value
  let a5 := 0 -- Filler value to ensure there is a6 value
  let a6 := 0 -- Filler value to ensure there is a6 value
  (1 + 2*x) * (x - 2)^5 = a0 + a1 * (1 - x) + a2 * (1 - x)^2 + a3 * (1 - x)^3 + a4 * (1 - x)^4 + a5 * (1 - x)^5 + a6 * (1 - x)^6 ->
  a0 = -3 ∧
  a0 + a1 + a2 + a3 + a4 + a5 + a6 = -32 :=
by
  intro a0 a1 a2 a3 a4 a5 a6 h
  exact ⟨rfl, sorry⟩

end coefficients_verification_l425_425818


namespace least_xy_value_l425_425067

theorem least_xy_value (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 90 :=
by
  sorry

end least_xy_value_l425_425067


namespace count_three_digit_squares_divisible_by_4_l425_425112

def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

theorem count_three_digit_squares_divisible_by_4 : 
  let squares := {n : ℕ | is_three_digit n ∧ is_perfect_square n ∧ is_divisible_by_4 n} in
  Fintype.card squares = 10 :=
sorry

end count_three_digit_squares_divisible_by_4_l425_425112


namespace sufficient_but_not_necessary_l425_425332

theorem sufficient_but_not_necessary (x : ℝ) : (x < -2 → x ≤ 0) → ¬(x ≤ 0 → x < -2) :=
by
  sorry

end sufficient_but_not_necessary_l425_425332


namespace max_root_absolute_value_gt_one_l425_425330

noncomputable def is_tasty (P : Polynomial ℝ) : Prop :=
  P.monic ∧ ∀ coef, coef ∈ P.coeff.support → P.coeff coef ∈ set.Icc (-1:ℝ) (1:ℝ)

theorem max_root_absolute_value_gt_one {P : Polynomial ℝ} (hP : P.monic) (h_not_tasty_mult : ∀ Q : Polynomial ℝ, Q.monic → ¬is_tasty (P * Q)) :
  ∃ r > 1, ∀ (χ ∈ P.root_set ℂ), abs χ ≤ r :=
sorry

end max_root_absolute_value_gt_one_l425_425330


namespace total_basketballs_l425_425951

theorem total_basketballs (soccer_balls : ℕ) (soccer_balls_with_holes : ℕ) (basketballs_with_holes : ℕ) (balls_without_holes : ℕ) 
  (h1 : soccer_balls = 40) 
  (h2 : soccer_balls_with_holes = 30) 
  (h3 : basketballs_with_holes = 7) 
  (h4 : balls_without_holes = 18)
  (soccer_balls_without_holes : ℕ) 
  (basketballs_without_holes : ℕ) 
  (total_basketballs : ℕ)
  (h5 : soccer_balls_without_holes = soccer_balls - soccer_balls_with_holes)
  (h6 : basketballs_without_holes = balls_without_holes - soccer_balls_without_holes)
  (h7 : total_basketballs = basketballs_without_holes + basketballs_with_holes) : 
  total_basketballs = 15 := 
sorry

end total_basketballs_l425_425951


namespace prime_multiple_of_seven_probability_l425_425426

theorem prime_multiple_of_seven_probability :
  (finset.filter (λ n, nat.prime n ∧ n % 7 = 0) (finset.range 51)).card = 1 →
  (1 / 50 : ℚ) = 1 / 50 :=
by
  sorry

end prime_multiple_of_seven_probability_l425_425426


namespace problem_l425_425933

def q (x : ℤ) : ℤ := ∑ i in finset.range 2008, x ^ i

def divisor (x : ℤ) : ℤ := x^3 + 2*x^2 + x + 1

def remainder (q : ℤ → ℤ) (divisor : ℤ → ℤ) : ℤ → ℤ := sorry -- This would be a function computing the remainder

theorem problem (q r : ℤ → ℤ) : 
  (∀ x, r x = (q x) % divisor x) →
  abs (r 2007) % 1000 = 49 := 
by
  sorry

end problem_l425_425933


namespace range_of_slope_l425_425524

open Real

theorem range_of_slope (θ : ℝ) (h : θ ∈ Icc (π / 3) (3 * π / 4)) :
  ∃ k : ℝ, k = tan θ ∧ (k ∈ Iic (-1) ∨ k ∈ Ici (sqrt 3)) :=
sorry

end range_of_slope_l425_425524


namespace solve_system_l425_425566

theorem solve_system (n : ℕ) (h_pos : 0 < n)
  (x : ℕ → ℝ) (h_nonneg : ∀ i, 1 ≤ i ∧ i ≤ n → 0 ≤ x i)
  (h_eq1 : ∑ i in Finset.range(n).map (Nat.succ), x i ^ i = n)
  (h_eq2 : ∑ i in Finset.range(n).map (Nat.succ), i * x i = (n * (n + 1)) / 2) :
  ∀ i, 1 ≤ i ∧ i ≤ n → x i = 1 :=
begin
  sorry
end

end solve_system_l425_425566


namespace avg_fuel_consumption_correct_remaining_fuel_correct_cannot_return_home_without_refueling_l425_425421

-- Average fuel consumption per kilometer
noncomputable def avgFuelConsumption (initial_fuel: ℝ) (final_fuel: ℝ) (distance: ℝ) : ℝ :=
  (initial_fuel - final_fuel) / distance

-- Relationship between remaining fuel Q and distance x
noncomputable def remainingFuel (initial_fuel: ℝ) (consumption_rate: ℝ) (distance: ℝ) : ℝ :=
  initial_fuel - consumption_rate * distance

-- Check if the car can return home without refueling
noncomputable def canReturnHome (initial_fuel: ℝ) (consumption_rate: ℝ) (round_trip_distance: ℝ) (alarm_fuel_level: ℝ) : Bool :=
  initial_fuel - consumption_rate * round_trip_distance ≥ alarm_fuel_level

-- Theorem statements to prove
theorem avg_fuel_consumption_correct :
  avgFuelConsumption 45 27 180 = 0.1 :=
sorry

theorem remaining_fuel_correct :
  ∀ x, remainingFuel 45 0.1 x = 45 - 0.1 * x :=
sorry

theorem cannot_return_home_without_refueling :
  ¬canReturnHome 45 0.1 (220 * 2) 3 :=
sorry

end avg_fuel_consumption_correct_remaining_fuel_correct_cannot_return_home_without_refueling_l425_425421


namespace equilateral_triangle_on_parabola_side_length_l425_425219

theorem equilateral_triangle_on_parabola_side_length :
  ∀ (a : ℝ), (∃ (x y : ℝ), (x = a / 2 ∧ y = (sqrt 3) / 2 * a ∧ x^2 = 2 * y)) →
    (∃ (x y : ℝ), (x = -a / 2 ∧ y = (sqrt 3) / 2 * a ∧ x^2 = 2 * y)) →
      a = 4 * sqrt 3 :=
by
  sorry

end equilateral_triangle_on_parabola_side_length_l425_425219


namespace sum_of_all_numbers_with_digits_1_to_9_l425_425229

theorem sum_of_all_numbers_with_digits_1_to_9 :
  let digits := {1, 2, 3, 4, 5, 6, 7, 8, 9},
      factorial n := if n = 0 then 1 else n * factorial (n - 1),
      geometric_sum n := (10^n - 1) / 9,
      sum_of_digits := 45 -- 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
   in sum_of_digits * geometric_sum 9 * factorial 8 = 201599999798400 := 
sorry

end sum_of_all_numbers_with_digits_1_to_9_l425_425229


namespace problem_solution_l425_425942

noncomputable def b (n : ℕ) (b₁ : ℕ) : ℕ :=
  if b (n-1) % 2 = 0 then b (n-1) / 2 else 4 * b (n-1) + 1

def valid_b₁ (b₁ : ℕ) : Prop :=
  (∀ n : ℕ, ((n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5) → 
              b n b₁ > b₁)) ∧ 
  b₁ ≤ 1003

theorem problem_solution : 
  (finset.range 1004).filter valid_b₁ = 502 := 
by
  sorry

end problem_solution_l425_425942


namespace repeating_sum_to_fraction_l425_425009

theorem repeating_sum_to_fraction :
  (0.333333333333333 ~ 1/3) ∧ 
  (0.0404040404040401 ~ 4/99) ∧ 
  (0.005005005005001 ~ 5/999) →
  (0.333333333333333 + 0.0404040404040401 + 0.005005005005001) = (112386 / 296703) := 
by
  repeat { sorry }

end repeating_sum_to_fraction_l425_425009


namespace number_of_incorrect_statements_l425_425734

def statement1 (mean variance : ℝ) (data : List ℝ) (const : ℝ) : Prop :=
  let new_data := data.map (λ x => x + const)
  mean ≠ mean ∧ variance = variance

def statement2 (x y : ℝ) : Prop :=
  let y_hat := 5 - 3 * x
  x + 1 = x + 1 → y_hat - 3 ≠ y

def statement3 (b a x_bar y_bar : ℝ) : Prop :=
  let y_hat := b * x_bar + a
  y_hat = y_bar

def statement4 : Prop :=
  let smokers := 100
  let confidence := 99
  (smokers * 0.99 = confidence / 100) → false

theorem number_of_incorrect_statements :
  ∃ incorrect_statements : ℕ, 
    incorrect_statements = 3 :=
by
  -- These are the definitions from the conditions
  let incorrect_statements := 
    if ¬ statement1 0 0 [1, 2, 3] 1 then 1 else 0 +
    if ¬ statement2 1 5 then 1 else 0 +
    if statement3 1 0 0 0 then 0 else 1 +
    if ¬ statement4 then 1 else 0
  existsi incorrect_statements
  exact sorry

end number_of_incorrect_statements_l425_425734


namespace multiple_of_chickens_l425_425045

-- Definitions according to conditions
def initial_chickens : ℕ := 4
def eggs_per_chicken_per_day : ℕ := 6
def total_weekly_eggs : ℕ := 1344

-- Now, we state the theorem we need to prove
theorem multiple_of_chickens :
  let eggs_per_chicken_per_week := eggs_per_chicken_per_day * 7 in
  let current_chickens := total_weekly_eggs / eggs_per_chicken_per_week in
  current_chickens / initial_chickens = 8 :=
by
  sorry

end multiple_of_chickens_l425_425045


namespace part1_part2_l425_425806

noncomputable def binom1_expr (x : ℝ) : ℝ := (Real.sqrt x + 1 / (2 * Real.sqrt (Real.sqrt x)))^10
noncomputable def binom2_expr (x : ℝ) : ℝ := (Real.sqrt x + 1 / (2 * Real.sqrt (Real.sqrt x)))^8

theorem part1 (x : ℝ) : 
  coefficient_of_x_squared (binom1_expr x) = 105 / 8 := sorry

theorem part2 (x : ℝ) (arith_seq : arithmetic_sequence 
  (list.first (coefficients (expansion (binom2_expr x)))) 
  (list.take 3 (coefficients (expansion (binom2_expr x))))) :
  middle_term (binom2_expr x) = (35 / 8) * x := sorry

end part1_part2_l425_425806


namespace fliers_left_l425_425683

theorem fliers_left (total : ℕ) (morning_fraction afternoon_fraction : ℚ) 
  (h1 : total = 1000)
  (h2 : morning_fraction = 1/5)
  (h3 : afternoon_fraction = 1/4) :
  let morning_sent := total * morning_fraction
  let remaining_after_morning := total - morning_sent
  let afternoon_sent := remaining_after_morning * afternoon_fraction
  let remaining_after_afternoon := remaining_after_morning - afternoon_sent
  remaining_after_afternoon = 600 :=
by
  sorry

end fliers_left_l425_425683


namespace shorter_base_of_isosceles_trapezoid_l425_425027

theorem shorter_base_of_isosceles_trapezoid
  (a b : ℝ)
  (h : a > b)
  (h_division : (a + b) / 2 = (a - b) / 2 + 10) :
  b = 10 :=
by
  sorry

end shorter_base_of_isosceles_trapezoid_l425_425027


namespace math_problem_l425_425963

theorem math_problem (a b : ℝ) :
  (a^2 - 1) * (b^2 - 1) ≥ 0 → a^2 + b^2 - 1 - a^2 * b^2 ≤ 0 :=
by
  sorry

end math_problem_l425_425963


namespace magnitude_of_z_l425_425853

noncomputable def z : ℂ := sorry

theorem magnitude_of_z (h : z * (1 + complex.I) = 2 * complex.I) : complex.abs z = real.sqrt 2 :=
by
  sorry

end magnitude_of_z_l425_425853


namespace count_three_digit_perfect_squares_divisible_by_4_l425_425122

theorem count_three_digit_perfect_squares_divisible_by_4 : ∃ n, n = 11 ∧
  (∀ k, 100 ≤ k ^ 2 ∧ k ^ 2 ≤ 999 → k ^ 2 % 4 = 0 → (k % 2 = 0 ∧ 10 ≤ k ≤ 31)) :=
sorry

end count_three_digit_perfect_squares_divisible_by_4_l425_425122


namespace repeatingDecimals_fraction_eq_l425_425002

noncomputable def repeatingDecimalsSum : ℚ :=
  let x : ℚ := 1 / 3
  let y : ℚ := 4 / 99
  let z : ℚ := 5 / 999
  x + y + z

theorem repeatingDecimals_fraction_eq : repeatingDecimalsSum = 42 / 111 :=
  sorry

end repeatingDecimals_fraction_eq_l425_425002


namespace ellipse_equation_max_AB_length_l425_425845

noncomputable def ellipse_eccentricity (a b : ℝ) (h1 : a > b > 0) : ℝ :=
  (1 - b^2 / a^2)^0.5

theorem ellipse_equation
  (hEccentricity : ellipse_eccentricity 5 3 (by norm_num) = 4/5)
  (hPointOnEllipse : ∃ a b, a = 5 ∧ b = 3 ∧ (10√2/3)^2 / (a^2) + 1^2 / (b^2) = 1) :
  ∀ x y : ℝ, x^2 / 25 + y^2 / 9 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1)^2 / 25 + (p.2)^2 / 9 = 1} :=
sorry

theorem max_AB_length
  (R : ℝ) (hR : 3 < R ∧ R < 5) :
  ∃ A B : ℝ × ℝ,
    (A.1^2 / 25 + A.2^2 / 9 = 1) ∧ (B.1^2 + B.2^2 = R^2) ∧
    ∀ l : ℝ, |√(l * (1 - 9/(R^2)) + 5 - R^2)| ≤ 2 :=
sorry

end ellipse_equation_max_AB_length_l425_425845


namespace problem_statement_l425_425578

noncomputable def exists_x_in_interval : Prop :=
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 
             ∃ (xs : Fin 100 → ℝ), 
               (∀ i, 0 ≤ xs i ∧ xs i ≤ 1) ∧
               (Finset.sum (Finset.univ) (λ i, |x - xs i|) = 50)

theorem problem_statement : exists_x_in_interval :=
sorry

end problem_statement_l425_425578


namespace number_of_yellow_highlighters_l425_425157

-- Definitions based on the given conditions
def total_highlighters : Nat := 12
def pink_highlighters : Nat := 6
def blue_highlighters : Nat := 4

-- Statement to prove the question equals the correct answer given the conditions
theorem number_of_yellow_highlighters : 
  ∃ y : Nat, y = total_highlighters - (pink_highlighters + blue_highlighters) := 
by
  -- TODO: The proof will be filled in here
  sorry

end number_of_yellow_highlighters_l425_425157


namespace max_log_sum_l425_425464

theorem max_log_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : 2 * x + 3 * y = 6) :
  (log (3/2) x + log (3/2) y) ≤ 1 :=
sorry

end max_log_sum_l425_425464


namespace parallelogram_area_l425_425417

theorem parallelogram_area (base height : ℕ) (h_base : base = 6) (h_height : height = 3) : base * height = 18 := by
  rw [h_base, h_height]
  exact rfl

end parallelogram_area_l425_425417


namespace repeating_sum_to_fraction_l425_425007

theorem repeating_sum_to_fraction :
  (0.333333333333333 ~ 1/3) ∧ 
  (0.0404040404040401 ~ 4/99) ∧ 
  (0.005005005005001 ~ 5/999) →
  (0.333333333333333 + 0.0404040404040401 + 0.005005005005001) = (112386 / 296703) := 
by
  repeat { sorry }

end repeating_sum_to_fraction_l425_425007


namespace finite_operation_ii_l425_425565

theorem finite_operation_ii (a b : ℕ) (h_distinct : a ≠ b) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  ∃ N, sorry := 
begin
  sorry
end

end finite_operation_ii_l425_425565


namespace number_of_n_for_prime_l425_425039

theorem number_of_n_for_prime (n : ℕ) : (n > 0) → ∃! n, Nat.Prime (n * (n + 2)) :=
by 
  sorry

end number_of_n_for_prime_l425_425039


namespace largest_expression_l425_425203

noncomputable def y : ℝ := 2 * 10 ^ (-2011)

theorem largest_expression : 
  (∀ z ∈ {3 + y, 3 - y, 4 * y, 4 / y, y / 4}, z ≤ 4 / y) ∧ (4 / y ∈ {3 + y, 3 - y, 4 * y, 4 / y, y / 4}) := 
by
  sorry

end largest_expression_l425_425203


namespace fraction_multiplication_exponent_l425_425752

theorem fraction_multiplication_exponent :
  ( (8 : ℚ) / 9 )^2 * ( (1 : ℚ) / 3 )^2 = (64 / 729 : ℚ) := 
by
  sorry

end fraction_multiplication_exponent_l425_425752


namespace a_plus_b_eq_2_l425_425878

theorem a_plus_b_eq_2 (a b : ℝ) 
  (h₁ : 2 = a + b) 
  (h₂ : 4 = a + b / 4) : a + b = 2 :=
by
  sorry

end a_plus_b_eq_2_l425_425878


namespace count_satisfying_integers_l425_425500

theorem count_satisfying_integers :
  ∃ (S : Finset ℤ), (∀ x ∈ S, (x^2 - x - 2)^(x + 3) = 1) ∧ S.card = 5 :=
by
  sorry

end count_satisfying_integers_l425_425500


namespace probability_odd_sum_l425_425801

-- Definitions based on the conditions
def cards : List ℕ := [1, 2, 3, 4, 5]

def is_odd_sum (a b : ℕ) : Prop := (a + b) % 2 = 1

def combinations (n k : ℕ) : ℕ := (Nat.choose n k)

-- Main statement
theorem probability_odd_sum :
  (combinations 5 2) = 10 → -- Total combinations of 2 cards from 5
  (∃ N, N = 6 ∧ (N:ℚ)/(combinations 5 2) = 3/5) :=
by 
  sorry

end probability_odd_sum_l425_425801


namespace ratio_of_similar_triangles_l425_425548

theorem ratio_of_similar_triangles
  (A B C M N : Point)
  (h_parallel : MN ∥ BC)
  (h_AM : dist A M = 4)
  (h_AB : dist A B = 7) :
  dist A N / dist A C = 4 / 7 :=
by 
  sorry

end ratio_of_similar_triangles_l425_425548


namespace haley_money_l425_425864

variable (x : ℕ)

def initial_amount : ℕ := 2
def difference : ℕ := 11
def total_amount (x : ℕ) : ℕ := x

theorem haley_money : total_amount x - initial_amount = difference → total_amount x = 13 := by
  sorry

end haley_money_l425_425864


namespace part1_part2_l425_425807

noncomputable def binom1_expr (x : ℝ) : ℝ := (Real.sqrt x + 1 / (2 * Real.sqrt (Real.sqrt x)))^10
noncomputable def binom2_expr (x : ℝ) : ℝ := (Real.sqrt x + 1 / (2 * Real.sqrt (Real.sqrt x)))^8

theorem part1 (x : ℝ) : 
  coefficient_of_x_squared (binom1_expr x) = 105 / 8 := sorry

theorem part2 (x : ℝ) (arith_seq : arithmetic_sequence 
  (list.first (coefficients (expansion (binom2_expr x)))) 
  (list.take 3 (coefficients (expansion (binom2_expr x))))) :
  middle_term (binom2_expr x) = (35 / 8) * x := sorry

end part1_part2_l425_425807


namespace third_vertex_orbit_l425_425313

noncomputable def circle_orbit (A B : ℝ × ℝ) (C : ℝ × ℝ) (c : ℝ) : Prop :=
  let (x₁, y₁) := A in
  let (x₂, y₂) := B in
  let (x, y) := C in
  ((x - x₁)^2 + (y - y₁)^2) + ((x - x₂)^2 + (y - y₂)^2) + (x₂ - x₁)^2 = 8 * (1/2 * abs (x₂ - x₁) * abs y) ∧
  ((x - x₁) = -c/2 ∧ y₁ = 0 ∧
  (x - x₂) = c/2 ∧ y₂ = 0 ∧
  ((x, y - c) ∈ circle (0, c) (c/2) ∨ (x, y + c) ∈ circle (0, -c) (c/2)))

theorem third_vertex_orbit :
  ∀ (A B C : ℝ × ℝ) (c : ℝ),
  A = (-c / 2, 0) →
  B = (c / 2, 0) →
  ((x, y) = C) →
  circle_orbit A B C c :=
sorry

end third_vertex_orbit_l425_425313


namespace proof_arithmetic_sequence_l425_425082

variable {d : ℤ} (an Sn : ℕ → ℤ)

-- Conditions
def first_term := an 1 = 1
def common_difference := d ≠ 0
def geometric_sequence := an 2 * an 9 = (an 4) ^ 2

-- Proving the results
theorem proof_arithmetic_sequence
  (h1 : first_term an)
  (h2 : common_difference d)
  (h3 : geometric_sequence an) :
  ∃ q : ℚ, (q = 5 / 2) ∧
  (∀ n, an n = 3 * n - 2) ∧
  (∀ n, Sn n = (3 * n^2 - n) / 2) :=
by
  sorry

end proof_arithmetic_sequence_l425_425082


namespace phil_cards_left_l425_425222

-- Conditions
def cards_per_week : ℕ := 20
def weeks_per_year : ℕ := 52

-- Total number of cards in a year
def total_cards (cards_per_week weeks_per_year : ℕ) : ℕ := cards_per_week * weeks_per_year

-- Number of cards left after losing half in fire
def cards_left (total_cards : ℕ) : ℕ := total_cards / 2

-- Theorem to prove
theorem phil_cards_left (cards_per_week weeks_per_year : ℕ) :
  cards_left (total_cards cards_per_week weeks_per_year) = 520 :=
by
  sorry

end phil_cards_left_l425_425222


namespace neck_couplet_scene_poem_feelings_l425_425234

def poem_neck_couplet : Prop :=
  ∀ (bright_scenery sorrowful_emotions : Prop), 
  bright_scenery → sorrowful_emotions → (sorrowful_emotions ∧ (purpose bright_scenery sorrowful_emotions = "contrast joyful scenery with sorrowful emotions"))

def poem_overall_feelings : Prop :=
  ∀ (remembrance_of_friendship reluctance_of_parting sadness_uncertainty sigh_uncertainty : Prop),
  remembrance_of_friendship → reluctance_of_parting → sadness_uncertainty → sigh_uncertainty →
  (thoughts_and_feelings convey_feelings_remembrance reluctance sadness uncertainty = 
   "remembrance_of_childhood_friendship, reluctance_of_parting, sadness_not_meeting_again, sigh_drifting_existence")

theorem neck_couplet_scene (bright_scenery sorrowful_emotions : Prop) : 
  poem_neck_couplet :=
by
  sorry

theorem poem_feelings (remembrance_of_friendship reluctance_of_parting sadness_uncertainty sigh_uncertainty : Prop) : 
  poem_overall_feelings :=
by
  sorry

end neck_couplet_scene_poem_feelings_l425_425234


namespace smile_area_eq_l425_425892

noncomputable def radius : ℝ := 2
def length_BE_AF : ℝ := 3

def area_smile (r : ℝ) (l : ℝ) : ℝ :=
  let area_sector := 1 / 2 * l^2 * (Real.pi / 4)
  let area_semicircle := 1 / 2 * Real.pi * r^2
  2 * area_sector - area_semicircle
  
theorem smile_area_eq :
  area_smile radius length_BE_AF = 5 * Real.pi / 4 := by
  sorry

end smile_area_eq_l425_425892


namespace original_employees_approx_l425_425721

-- Definitions
def reduced_employees : ℝ := 195
def reduction_percentage : ℝ := 0.13
def retained_percentage := 1 - reduction_percentage

-- Theorem statement
theorem original_employees_approx :
  reduced_employees = retained_percentage * original_employees → original_employees ≈ 224 :=
by
  sorry

end original_employees_approx_l425_425721


namespace perfect_squares_three_digit_divisible_by_4_count_l425_425139

theorem perfect_squares_three_digit_divisible_by_4_count : 
  ∃ (n : ℕ), (n = 11) ∧ (∀ m, 100 ≤ m ∧ m < 1000 ∧ (∃ k, m = k^2) ∧ (m % 4 = 0) ↔ m ∈ {100, 144, 196, 256, 324, 400, 484, 576, 676, 784, 900}) :=
by
  existsi 11
  split
  · reflexivity
  · intro m
    split
    · intro h
      rcases h with ⟨⟨_, _, _, _⟩, _⟩
      -- details have been omitted
      sorry
    · intro h
      -- details have been omitted
      sorry

end perfect_squares_three_digit_divisible_by_4_count_l425_425139


namespace largest_tan_P_in_triangle_l425_425914

noncomputable def largest_tan_P (P Q R : ℝ) (PQ QR : ℝ) :=
  PQ = 24 ∧ QR = 18 → tan P = 3 * sqrt 7 / 7

theorem largest_tan_P_in_triangle :
  ∃ (P Q R : ℝ), largest_tan_P P Q R 24 18 :=
begin
  sorry
end

end largest_tan_P_in_triangle_l425_425914


namespace count_three_digit_squares_divisible_by_4_l425_425113

def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k
def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

theorem count_three_digit_squares_divisible_by_4 : 
  let squares := {n : ℕ | is_three_digit n ∧ is_perfect_square n ∧ is_divisible_by_4 n} in
  Fintype.card squares = 10 :=
sorry

end count_three_digit_squares_divisible_by_4_l425_425113


namespace second_fragment_speed_l425_425354

-- Definitions for initial condition of the firework
def initial_vertical_speed : ℝ := 20 -- m/s
def time_until_explosion : ℝ := 1 -- second
def gravity : ℝ := 10 -- m/s^2
def mass_ratio : ℝ := 1 / 2 -- Ratio of masses ( smaller:larger = 1:2 )
def smaller_fragment_horizontal_speed : ℝ := 16 -- m/s

-- Definition for the magnitude of the speed of the second fragment immediately after the explosion
def magnitude_speed_of_second_fragment (v_x v_y : ℝ) : ℝ := (v_x^2 + v_y^2).sqrt

-- Calculate initial velocity just before explosion
def initial_velocity_before_explosion : ℝ := initial_vertical_speed - gravity * time_until_explosion

-- Function to calculate resultant magnitude of the velocity of the larger fragment
def larger_fragment_velocity_magnitude : ℝ := magnitude_speed_of_second_fragment (-8) 10 -- v_x = -8 m/s, v_y = 10 m/s

theorem second_fragment_speed : larger_fragment_velocity_magnitude = 17 := by
  sorry

end second_fragment_speed_l425_425354


namespace count_oddly_powerful_under_5000_l425_425405

open Nat

def is_oddly_powerful (n : ℕ) : Prop :=
  ∃ a b : ℕ, b > 1 ∧ Odd b ∧ a^b = n

noncomputable def oddly_powerful_under_5000 : List ℕ :=
  (List.range 5000).filter is_oddly_powerful

theorem count_oddly_powerful_under_5000 :
  (oddly_powerful_under_5000.length = 20) :=
by
  sorry

end count_oddly_powerful_under_5000_l425_425405


namespace dave_ticket_count_l425_425388

theorem dave_ticket_count :
  let initial_tickets := 11
  let spent_tickets := 5
  let won_tickets := 10
  initial_tickets - spent_tickets + won_tickets = 16 :=
by
  let initial_tickets := 11
  let spent_tickets := 5
  let won_tickets := 10
  calc
    initial_tickets - spent_tickets + won_tickets 
      = 11 - 5 + 10 : by rfl
  ... = 6 + 10        : by rfl
  ... = 16            : by rfl

end dave_ticket_count_l425_425388


namespace train_pass_time_l425_425325

/-- Given a train length of 275 meters, a train speed of 60 km/h, and a man running at 6 km/h in the opposite direction, 
prove that the time for the train to pass the man is approximately 15 seconds. -/
theorem train_pass_time : 
  ∀ (L : ℕ) (S_train S_man : ℕ),
    L = 275 ∧ S_train = 60 ∧ S_man = 6 →
    (let S_relative := (S_train + S_man) * 5 / 18 in
    let time := L / S_relative in
    time ≈ 15) :=
by
  intros,
  sorry

end train_pass_time_l425_425325


namespace max_handshakes_l425_425735

theorem max_handshakes (n : ℕ) (h1 : n = 20)
  (h2 : ∀ (set : Finset ℕ), set.card = 3 → 
   ¬ (∀ i j, i ∈ set ∧ j ∈ set → i ≠ j → acquaintances i j)) : 
  ∃ m, m = 100 ∧ is_max_handshakes n m :=
by
  sorry

end max_handshakes_l425_425735


namespace simplify_expression_l425_425620

theorem simplify_expression (c x b y d : ℝ) :
  (cx * (c^2 * x^2 + 3 * b^2 * y^2 + c^2 * y^2) + dy * (b^2 * x^2 + 3 * c^2 * x^2 + b^2 * y^2)) / (cx + dy)
  = c^2 * x^2 + d * b^2 * y^2 :=
by
  -- Expression after factoring
  have num_eq : cx * (c^2 * x^2 + 3 * b^2 * y^2 + c^2 * y^2) + dy * (b^2 * x^2 + 3 * c^2 * x^2 + b^2 * y^2)
             = (cx + dy) * (c^2 * x^2 + d * b^2 * y^2),
  from sorry,
  -- Simplifying the fraction
  calc
    (cx * (c^2 * x^2 + 3 * b^2 * y^2 + c^2 * y^2) + dy * (b^2 * x^2 + 3 * c^2 * x^2 + b^2 * y^2)) / (cx + dy)
        = (cx + dy) * (c^2 * x^2 + d * b^2 * y^2) / (cx + dy) : by rw num_eq
    ... = c^2 * x^2 + d * b^2 * y^2 : by rw div_self sorry

end simplify_expression_l425_425620


namespace total_spider_legs_l425_425373

theorem total_spider_legs (num_legs_single_spider group_spider_count: ℕ) 
      (h1: num_legs_single_spider = 8) 
      (h2: group_spider_count = (num_legs_single_spider / 2) + 10) :
      group_spider_count * num_legs_single_spider = 112 := 
by
  sorry

end total_spider_legs_l425_425373


namespace number_of_boys_l425_425353

-- Definitions based on conditions:
def num_adults := 3
def num_girls := 7
def eggs_per_adult := 3
def eggs_per_girl := 1
def total_eggs := 36
def eggs_per_boy (eggs_per_girl : ℕ) := eggs_per_girl + 1

-- The main proof statement:
theorem number_of_boys (num_adults num_girls eggs_per_adult eggs_per_girl total_eggs : ℕ) (eggs_per_boy : ℕ := eggs_per_girl + 1) :
  ∃ (num_boys : ℕ), (num_adults * eggs_per_adult + num_girls * eggs_per_girl + num_boys * eggs_per_boy eggs_per_girl = total_eggs) ∧ num_boys = 10 :=
sorry

end number_of_boys_l425_425353


namespace relatively_prime_pairs_unique_l425_425327
open Nat

theorem relatively_prime_pairs_unique :
  ∀ (m n : ℕ), Nat.gcd m n = 1 →
  (∃ f : ℕ → ℕ, ¬(IsConst f) ∧ (∀ a b : ℕ, Nat.gcd (a + b + 1) (m * f a + n * f b) > 1)) →
  (m = 1 ∧ n = 1) :=
by
  intros m n gcd_mn exists_f
  sorry

end relatively_prime_pairs_unique_l425_425327


namespace problem1_problem2_l425_425455

-- Define the first problem: if a > 0 and b > 0, and a + b = 1, then ab ≤ 1/4.
theorem problem1 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : ab ≤ 1/4 :=
by sorry

-- Define the second problem: if a > 0 and b > 0, and a + b = 1, then \frac{4}{a} + \frac{1}{b} \geq |2x - 1| - |x + 2| implies x ∈ [-6,12].
theorem problem2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) (x : ℝ) (h4 : \frac{4}{a} + \frac{1}{b} \geq |2x - 1| - |x + 2|) : 
 x ∈ set.Icc (-6 : ℝ) (12 : ℝ) :=
by sorry

end problem1_problem2_l425_425455


namespace individual_weight_l425_425159

def total_students : ℕ := 1500
def sampled_students : ℕ := 100

def individual := "the weight of each student"

theorem individual_weight :
  (total_students = 1500) →
  (sampled_students = 100) →
  individual = "the weight of each student" :=
by
  intros h1 h2
  sorry

end individual_weight_l425_425159


namespace circles_externally_tangent_l425_425104

noncomputable def center_circle1 : ℝ × ℝ := (0, 0)
noncomputable def radius_circle1 : ℝ := 1

noncomputable def center_circle2 : ℝ × ℝ := (3, 4)
noncomputable def radius_circle2 : ℝ := 4

theorem circles_externally_tangent :
  dist center_circle1 center_circle2 = radius_circle1 + radius_circle2 := by
    sorry

end circles_externally_tangent_l425_425104


namespace functional_solution_l425_425684

-- Definitions for the conditions
variables {f : ℝ → ℝ}

-- Condition 1: f(x) + f(y) ≠ 0 for all x, y ∈ ℝ
def condition1 (f : ℝ → ℝ) : Prop :=
  ∀ x y, f(x) + f(y) ≠ 0

-- Condition 2: Functional equation
def condition2 (f : ℝ → ℝ) : Prop :=
  ∀ x y, (f(x) - f(x - y)) / (f(x) + f(x + y)) + (f(x) - f(x + y)) / (f(x) + f(x - y)) = 0

-- The statement we want to prove
theorem functional_solution (f : ℝ → ℝ) (h1 : condition1 f) (h2 : condition2 f) :
  ∃ c : ℝ, c ≠ 0 ∧ ∀ x, f(x) = c :=
sorry

end functional_solution_l425_425684


namespace max_value_of_expr_l425_425685

-- Define the initial conditions and expression 
def initial_ones (n : ℕ) := List.replicate n 1

-- Given that we place "+" or ")(" between consecutive ones
def max_possible_value (n : ℕ) : ℕ := sorry

theorem max_value_of_expr : max_possible_value 2013 = 3 ^ 671 := 
sorry

end max_value_of_expr_l425_425685


namespace correct_calculation_exists_l425_425319

theorem correct_calculation_exists : ∃ C, ∀ a : ℝ, (a^2)^3 = a^6 :=
by
  use True
  intro a
  calc
    (a^2)^3 = a^(2*3) : by rw [pow_mul]
          ... = a^6   : by rw [mul_comm]

end correct_calculation_exists_l425_425319


namespace find_v_l425_425441

-- Define the operation as a Lean function
def operation (v : ℝ) : ℝ := v - v / 3

-- Define the Lean statement for the proof
theorem find_v (v : ℝ) (h : operation (operation v) = 12) : v = 27 := by
  sorry -- The proof is omitted, as requested.

end find_v_l425_425441


namespace negation_equiv_l425_425517

theorem negation_equiv (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0 :=
by
  sorry

end negation_equiv_l425_425517


namespace lily_correct_percentage_proof_l425_425210

noncomputable def Lily_correct_percentage (t : ℝ) : ℝ := 
  let max_alone := 0.7 * (t / 2)
  let max_total := 0.82 * t
  let together := max_total - max_alone
  let lily_alone := 0.85 * (t / 2)
  let lily_total := lily_alone + together
  (lily_total / t) * 100

theorem lily_correct_percentage_proof (t : ℝ) (ht : t > 0) :
    Lily_correct_percentage t = 89.5 := 
by
  let max_alone := 0.7 * (t / 2)
  let max_total := 0.82 * t
  let together := max_total - max_alone
  let lily_alone := 0.85 * (t / 2)
  let lily_total := lily_alone + together
  have lily_total_eq : lily_total = 0.895 * t :=
    calc
      lily_total   = 0.85 * (t / 2) + (0.82 * t - 0.7 * (t / 2)) : by
                     congr
      ...           = 0.425 * t + 0.82 * t - 0.35 * t             : by
                     ring
      ...           = 0.895 * t                                   : by
                     ring
  have percentage := (lily_total / t) * 100
  rw lily_total_eq
  rw div_mul_cancel _ (ne_of_gt ht)
  norm_num
  rfl
  sorry

end lily_correct_percentage_proof_l425_425210


namespace final_value_of_n_l425_425284

-- Define the initial conditions and the iterative process of the loop
def program_result (n S : ℕ) : ℕ :=
if S < 15 then
  let S' := S + n in
  let n' := n - 1 in
  program_result (n' : ℕ) S'
else
  n

-- State the theorem we want to prove with the correct answer
theorem final_value_of_n : program_result 5 0 = 0 :=
by repeat { sorry }

end final_value_of_n_l425_425284


namespace find_total_income_l425_425715

theorem find_total_income (I : ℝ) (H : (0.27 * I = 35000)) : I = 129629.63 :=
by
  sorry

end find_total_income_l425_425715


namespace min_value_xy_l425_425059

theorem min_value_xy (x y : ℕ) (h : 0 < x ∧ 0 < y) (cond : (1 : ℚ) / x + (1 : ℚ) /(3 * y) = 1 / 6) : 
  xy = 192 :=
sorry

end min_value_xy_l425_425059


namespace pulley_distance_l425_425740

theorem pulley_distance :
  ∀ (P Q R S T : ℝ) (PQ PR QS PT TQ : ℝ),
    PR = 10 ∧ QS = 6 ∧ RS = 30 ∧
    PT = RS ∧ TQ = PR - QS →
    PQ = 2 * Real.sqrt 229 :=
by
  intros P Q R S T PQ PR QS PT TQ h
  rcases h with ⟨hPR, hQS, hRS, hPT, hTQ⟩
  have h1 : PT = 30 := hRS
  have h2 : TQ = 4 := hPR - hQS
  rw [h1, h2]
  calc PQ = Real.sqrt (PT^2 + TQ^2) : by sorry
    ... = Real.sqrt (30^2 + 4^2) : by sorry
    ... = Real.sqrt 916 : by sorry
    ... = 2 * Real.sqrt 229 : by sorry

end pulley_distance_l425_425740


namespace max_handshakes_l425_425736

theorem max_handshakes (n : ℕ) (h1 : n = 20)
  (h2 : ∀ (set : Finset ℕ), set.card = 3 → 
   ¬ (∀ i j, i ∈ set ∧ j ∈ set → i ≠ j → acquaintances i j)) : 
  ∃ m, m = 100 ∧ is_max_handshakes n m :=
by
  sorry

end max_handshakes_l425_425736


namespace f_800_value_l425_425194

variable (f : ℝ → ℝ)
variable (f_prop : ∀ (x y : ℝ), 0 < x → 0 < y → f(x * y) = f(x) / y)
variable (f_400 : f 400 = 4)

theorem f_800_value : f 800 = 2 :=
by {
  sorry
}

end f_800_value_l425_425194


namespace ellipse_and_hyperbola_equations_area_of_triangle_f1pf2_l425_425383

noncomputable theory

-- Define the distance between foci
def foci_distance : ℝ := 2 * Real.sqrt 13

-- Define the properties and constants
def semi_major_axis := 7
def semi_minor_axis := Real.sqrt (49 - 13)
def ellipse_eq (x y : ℝ) : Prop := (x^2 / 49) + (y^2 / 36) = 1
def hyperbola_eq (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 4) = 1

-- Define distances PF1 and PF2
def pf1 : ℝ := 10
def pf2 : ℝ := 4

-- Define the area formula using point P and cosine theorem for triangle
def triangle_area {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (a b : ℝ) (sin_val : ℝ) : ℝ :=
  0.5 * a * b * sin_val

theorem ellipse_and_hyperbola_equations :
  ellipse_eq x y ∧ hyperbola_eq x y :=
sorry

theorem area_of_triangle_f1pf2 :
  triangle_area pf1 pf2 (3/5) = 12 :=
sorry

end ellipse_and_hyperbola_equations_area_of_triangle_f1pf2_l425_425383


namespace six_digit_positive_integers_satisfying_conditions_l425_425028

theorem six_digit_positive_integers_satisfying_conditions :
  let digits := finset.range 1 10 in
  (∑ abcd_ef in 
      finset.product
        (finset.product (digits.product digits) (digits.product digits))
        (digits.product digits),
    if (abcd_ef.1.1.1 * abcd_ef.1.1.2 + abcd_ef.1.2.1 * abcd_ef.1.2.2 +
        abcd_ef.2.1 * abcd_ef.2.2) % 2 = 0 then 1 else 0) = 280616 :=
by sorry

end six_digit_positive_integers_satisfying_conditions_l425_425028


namespace vector_sum_zero_tan_15_expression_f_sin_pi_12_domain_log_sin_smallest_positive_c_l425_425336

-- Problem 1
theorem vector_sum_zero {BC AB AC : Vector} :
  BC + AB - AC = 0 :=
sorry

-- Problem 2
theorem tan_15_expression :
  (tan 15) / (1 - (tan 15)^2) = sqrt(3) / 6 :=
sorry

-- Problem 3
theorem f_sin_pi_12 {f : ℝ → ℝ} (H : ∀ x, f (cos x) = cos (2 * x)) :
  f (sin (π / 12)) = - (sqrt 3 / 2) :=
sorry

-- Problem 4
theorem domain_log_sin (k : ℤ) :
  ∃ y, y = log (sin (x / 2)) ∧ 4 * (k : ℝ) * π < x ∧ x < 4 * (k : ℝ) * π + 2 * π :=
sorry

-- Problem 5
theorem smallest_positive_c {f : ℝ → ℝ} (H : ∀ x, f x = abs (2 * sin (2 * x - π / 6) + 1 / 2)) :
  ∃ c, 0 < c ∧ (∀ x, f (x + c) = f (x - c)) ∧ c = π / 2 :=
sorry

end vector_sum_zero_tan_15_expression_f_sin_pi_12_domain_log_sin_smallest_positive_c_l425_425336


namespace sum_arith_seq_l425_425541

-- Define the arithmetic sequence with a given starting point and common difference
def arithmetic_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + n * d

-- Given conditions
axiom a₅_eq : arithmetic_seq a₁ d 4 = 0.3
axiom a₁₂_eq : arithmetic_seq a₁ d 11 = 3.1

-- Solution answers
def a₁ : ℝ := -1.3
def d : ℝ := 0.4

-- The theorem to prove the sum of a_18 + a_19 + a_20 + a_21 + a_22 is 31.5
theorem sum_arith_seq : arithmetic_seq a₁ d 17 + arithmetic_seq a₁ d 18 + arithmetic_seq a₁ d 19 + arithmetic_seq a₁ d 20 + arithmetic_seq a₁ d 21 = 31.5 := by
  sorry

end sum_arith_seq_l425_425541


namespace find_real_x_l425_425428

def satisfies_floor_condition (x : ℝ) : Prop :=
  ⌊(x * ⌊x⌋ : ℝ)⌋ = 44

def solution_set : set ℝ :=
  {x | -6.4286 < x ∧ x <= -6.2857} ∪ {x | 7.3333 <= x ∧ x < 7.5}

theorem find_real_x :
  {x : ℝ | satisfies_floor_condition x} = solution_set :=
by
  sorry

end find_real_x_l425_425428


namespace cannot_separate_points_with_13_lines_l425_425540

-- Define the grid
def grid_size : ℕ := 8
def total_points : ℕ := grid_size * grid_size  -- 64 points
def boundary_points : ℕ := 4 * grid_size - 4    -- 28 boundary points

-- The main statement we want to prove
theorem cannot_separate_points_with_13_lines :
  ¬(∃ (lines : fin 13 → set (ℝ × ℝ)), 
    (∀ p : fin total_points, 
      ∃ l ∈ set.univ, ¬((set.prod (set.Icc 0 grid_size) (set.Icc 0 grid_size)) p ∈ l))) :=
sorry

end cannot_separate_points_with_13_lines_l425_425540


namespace find_angle_B_find_max_area_l425_425528

-- Definitions of given conditions and required proofs

variable (A B C : ℝ) -- Angles in the triangle
variable (a b c : ℝ) -- Opposite sides to angles A, B, C respectively

-- Given condition
def given_condition : Prop := (Real.sin A / a) = (Real.sqrt 3 * Real.cos B / b)

-- Required proofs
theorem find_angle_B (h : given_condition A B a b) : B = Real.pi / 3 :=
sorry

theorem find_max_area (h : given_condition A B a b) (h_b : b = 2) :
  ∃ S_max, S_max = Real.sqrt 3 ∧ (∃ D : Type, D = triangle_shape equilateral) :=
sorry

-- Auxiliary definition for triangle shape
def triangle_shape := { s : Type | ∃ t : s, true }

def equilateral : triangle_shape := ⟨unit, ⟨(), true.intro⟩⟩

end find_angle_B_find_max_area_l425_425528


namespace pages_per_inch_l425_425921

theorem pages_per_inch (number_of_books : ℕ) (average_pages_per_book : ℕ) (total_thickness : ℕ) 
                        (H1 : number_of_books = 6)
                        (H2 : average_pages_per_book = 160)
                        (H3 : total_thickness = 12) :
  (number_of_books * average_pages_per_book) / total_thickness = 80 :=
by
  -- Placeholder for proof
  sorry

end pages_per_inch_l425_425921


namespace roots_difference_eq_two_l425_425761

theorem roots_difference_eq_two (p : ℝ) :
  let f := λ (x : ℝ), x^2 - 2*p*x + (p^2 - p - 2)
  let r := (2*p + sqrt((2*p)^2 - 4*(p^2 - p - 2))) / 2
  let s := (2*p - sqrt((2*p)^2 - 4*(p^2 - p - 2))) / 2
  r - s = 2 :=
by
  sorry

end roots_difference_eq_two_l425_425761


namespace min_value_of_a_l425_425515

theorem min_value_of_a :
  (∀ x y : ℝ, x > 0 → y > 0 → (sqrt x + sqrt y ≤ sqrt 2 * sqrt (x + y))) := 
by 
  sorry

end min_value_of_a_l425_425515


namespace det_rotation_matrix_45_degrees_l425_425571

theorem det_rotation_matrix_45_degrees :
  let θ := Real.pi / 4 in
  let R := Matrix.of (λ i j, if i = 0 ∧ j = 0 then Real.cos θ else
                            if i = 0 ∧ j = 1 then -Real.sin θ else
                            if i = 1 ∧ j = 0 then Real.sin θ else
                            if i = 1 ∧ j = 1 then Real.cos θ else 0) in
  Matrix.det R = 1 :=
by
  let θ := Real.pi / 4
  let R := Matrix.of (λ i j, if i = 0 ∧ j = 0 then Real.cos θ else
                            if i = 0 ∧ j = 1 then -Real.sin θ else
                            if i = 1 ∧ j = 0 then Real.sin θ else
                            if i = 1 ∧ j = 1 then Real.cos θ else 0)
  show Matrix.det R = 1
  sorry

end det_rotation_matrix_45_degrees_l425_425571


namespace problem_ellipse_and_circle_l425_425451

def ellipse_equation (C : Type*) [metric_space C] (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1

def tangent_circle (F_2 : ℝ × ℝ) (r : ℝ) : Prop :=
  let F_2_x := F_2.1 in
  let F_2_y := F_2.2 in
  ∃ (x y : ℝ), ((x - F_2_x) ^ 2) + ( y ^ 2 ) = r

theorem problem_ellipse_and_circle
  (a b F_1 F_2 A B D : ℝ × ℝ)
  (h_ellipse_eq : ellipse_equation a b)
  (h_area_AOB : 0.5 * a * b = 2 * sqrt 2)
  (h_F1 : F_1 = (-sqrt 2, 0))
  (h_area_MF2N : ∀ (l : ℝ → ℝ), (∃ M N : ℝ × ℝ, triangle_area F_2 M N = 8/3))
  (A := (a, 0))
  (B := (0, b))
  (D := (a / 2, b / 2))
  (m : ℝ)
  (lin_eq : ∀ l, l (F_1.1) = F_1.2)
  (h_tangent_same : ∀ m, m = 1 ∨ m = -1):
  (ellipse_equation 8 4) ∧ ∀ (F_2 : ℝ × ℝ), tangent_circle (2, 0) (2 * sqrt 2) :=
sorry

end problem_ellipse_and_circle_l425_425451


namespace sequence_is_arithmetic_progression_l425_425292

theorem sequence_is_arithmetic_progression (a b : ℝ) : 
  ∀ n : ℕ, S n = a * n^2 + b * n → 
  (∃ (d : ℝ), ∀ n : ℕ, a n = S n - S (n - 1) ∧ a n = d * n + (b - a)) :=
by
  sorry

noncomputable def S (n : ℕ) : ℝ := a * n^2 + b * n

noncomputable def a (n : ℕ) : ℝ := S n - S (n - 1)

end sequence_is_arithmetic_progression_l425_425292


namespace square_mirror_side_length_l425_425375

theorem square_mirror_side_length (area_wall : ℝ) 
  (width_wall : ℝ)
  (length_wall : ℝ)
  (half_area : ℝ)
  (side_mirror : ℝ) :
  width_wall = 68 ∧
  length_wall = 85.76470588235294 ∧
  area_wall = width_wall * length_wall ∧
  half_area = area_wall / 2 ∧
  side_mirror ^ 2 = half_area →
  side_mirror = 54 :=
by {
  intros,
  sorry
}

end square_mirror_side_length_l425_425375


namespace exists_solution_for_an_l425_425804

theorem exists_solution_for_an (n : ℕ) (a : ℝ) (x_i : ℕ → ℝ) (h : ∀ i, 0 ≤ x_i i ∧ x_i i ≤ 3) :
  (∃ x ∈ Icc 0 3, (∑ i in Finset.range n, |x - x_i i| = a * n)) ↔ a = 3 / 2 :=
by
  sorry

end exists_solution_for_an_l425_425804


namespace solve_for_y_l425_425978

theorem solve_for_y : ∀ y : ℤ, 3^(y - 4) = 9^(y + 2) → y = -8 :=
  by
  intro y h
  sorry

end solve_for_y_l425_425978


namespace pages_called_this_week_l425_425182

-- Definitions as per conditions
def pages_called_last_week := 10.2
def total_pages_called := 18.8

-- Theorem to prove the solution
theorem pages_called_this_week :
  total_pages_called - pages_called_last_week = 8.6 :=
by
  sorry

end pages_called_this_week_l425_425182


namespace angle_second_or_fourth_quadrant_l425_425150

theorem angle_second_or_fourth_quadrant (θ : ℝ) (h : Real.cos θ + Real.sin θ = 1 / 2) : 
  (∃ k : ℤ, (θ ∈ Set.Ioo (π / 2 + k * π) (π + k * π)) ∨ (θ ∈ Set.Ioo (-π / 2 + k * π) (0 + k * π))) := 
  sorry

end angle_second_or_fourth_quadrant_l425_425150


namespace f_formula_l425_425448

-- Define the sequence a_n
def a (n : ℕ+) : ℚ := 1 / ((n + 1) ^ 2)

-- Define the function f
noncomputable def f (n : ℕ+) : ℚ := ∏ i in Finset.range (n + 1), (1 - a (i + 1))

-- State the theorem to be proven
theorem f_formula (n : ℕ+) : f n = (n + 2) / (2 * (n + 1)) :=
  sorry

end f_formula_l425_425448


namespace concatenated_number_divisible_by_37_l425_425629

theorem concatenated_number_divisible_by_37
  (a b : ℕ) (ha : 100 ≤ a ∧ a ≤ 999) (hb : 100 ≤ b ∧ b ≤ 999)
  (h₁ : a % 37 ≠ 0) (h₂ : b % 37 ≠ 0) (h₃ : (a + b) % 37 = 0) :
  (1000 * a + b) % 37 = 0 :=
sorry

end concatenated_number_divisible_by_37_l425_425629


namespace Beast_of_War_running_time_l425_425287

theorem Beast_of_War_running_time 
  (M : ℕ) 
  (AE : ℕ) 
  (BoWAC : ℕ)
  (h1 : M = 120)
  (h2 : AE = M - 30)
  (h3 : BoWAC = AE + 10) : 
  BoWAC = 100 
  := 
sorry

end Beast_of_War_running_time_l425_425287


namespace lateral_surface_area_of_cone_l425_425849

theorem lateral_surface_area_of_cone (r l : ℝ) (h_r : r = 3) (h_l : l = 8) :
  real.pi * r * l = 24 * real.pi := by
  rw [h_r, h_l]
  ring

end lateral_surface_area_of_cone_l425_425849


namespace find_other_root_l425_425218

theorem find_other_root (m : ℝ) (h : Polynomial.Coeff 3 2 = 3) (h1 : Polynomial.Coeff m 1 = m) (h2 : Polynomial.Coeff (-7) 0 = -7) :
  ∃ q : ℝ, 3 * q^2 + m * q = 7 ∧ q ≠ -1 :=
by
  have h3 : 3 * (-1)^2 + m * (-1) = 7 := by sorry
  exists (7/3)
  split
  · (Calculation of 3 * ((7/3)^2) + m * (7/3)) = 7 by sorry
  · by_contradiction
    assume h5 : (7/3) = -1
    have h6 : 3 * (7/3)^2 = 3 * 49/9 = 49/3 = 49/3 = 47 ≠7 by sorry

end find_other_root_l425_425218


namespace last_digit_2_pow_20_l425_425590

theorem last_digit_2_pow_20 :
  let last_digit (n : Nat) := n % 10
  in last_digit (2^20) = 6 := sorry

end last_digit_2_pow_20_l425_425590


namespace third_team_cups_l425_425237

theorem third_team_cups (required_cups : ℕ) (first_team : ℕ) (second_team : ℕ) (third_team : ℕ) :
  required_cups = 280 ∧ first_team = 90 ∧ second_team = 120 →
  third_team = required_cups - (first_team + second_team) :=
by
  intro h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end third_team_cups_l425_425237


namespace find_f_neg12_add_f_14_l425_425048

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log (Real.sqrt (x^2 - 2*x + 2) - x + 1)

theorem find_f_neg12_add_f_14 : f (-12) + f 14 = 2 :=
by
  -- The hard part, the actual proof, is left as sorry.
  sorry

end find_f_neg12_add_f_14_l425_425048


namespace swim_distance_downstream_l425_425709

theorem swim_distance_downstream (V_m : ℕ) (V_s : ℕ) 
  (V_m_eq : V_m = 10) 
  (upstream_distance : ℕ) 
  (upstream_time : ℕ) 
  (T : upstream_time = 2) 
  (D : upstream_distance = 12) 
  (upstream_eq : (V_m - V_s) * upstream_time = upstream_distance)
  : ((V_m + V_s) * T) = 28 :=
by {
  rw [V_m_eq, T, D] at upstream_eq,
  have V_s_eq : 2 * V_s = 8,
  { rw [mul_comm, mul_sub, mul_comm, mul_comm] at upstream_eq,
    simp at upstream_eq,
    exact (eq_of_sub_eq upstream_eq).resolve_left (by norm_num) },
  have V_s_val : V_s = 4,
  { rw [← two_mul, eq_comm, mul_eq_mul_right_iff] at V_s_eq,
    exact V_s_eq.resolve_right (b := 0) (by norm_num) },
  simp [V_s_val, V_m_eq, mul_comm],
  exact rfl
}

end swim_distance_downstream_l425_425709


namespace base9_first_digit_of_202211222012010111_3_is_4_l425_425992

theorem base9_first_digit_of_202211222012010111_3_is_4 :
  let y := 2 * 3^17 + 0 * 3^16 + 2 * 3^15 + 2 * 3^14 + 1 * 3^13 + 2 * 3^12 + 2 * 3^11 + 1 * 3^10 + 2 * 3^9 + 0 * 3^8 +
           1 * 3^7 + 0 * 3^6 + 1 * 3^5 + 1 * 3^4 + 1 * 3^3 + 0 * 3^2 + 1 * 3^1 + 1 * 3^0 in
  let y_base9 := (y / 9^5) % 9 in
  y_base9 = 4 :=
by sorry

end base9_first_digit_of_202211222012010111_3_is_4_l425_425992


namespace function_property_proof_l425_425824

theorem function_property_proof (f : ℝ → ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f(x) = y) ∧
  (∀ (x1 x2 : ℝ), x1 + x2 = 0 → f(x1) + f(x2) = 0) ∧
  (∀ (x : ℝ) (t : ℝ), t > 0 → f(x + t) > f(x)) ↔
  (f = λ x : ℝ, x^3) :=
by
  sorry

end function_property_proof_l425_425824


namespace no_solution_m_l425_425522

noncomputable def fractional_eq (x m : ℝ) : Prop :=
  2 / (x - 2) + m * x / (x^2 - 4) = 3 / (x + 2)

theorem no_solution_m (m : ℝ) : 
  (¬ ∃ x, fractional_eq x m) ↔ (m = -4 ∨ m = 6 ∨ m = 1) :=
sorry

end no_solution_m_l425_425522


namespace triangle_tan_sum_l425_425690

theorem triangle_tan_sum (A B C : ℝ) (h1 : A + B + C = Real.pi) (h2 : 2 * B = A + C) :
  tan(A / 2) + tan(C / 2) + Real.sqrt 3 * tan(A / 2) * tan(C / 2) = Real.sqrt 3 := 
sorry

end triangle_tan_sum_l425_425690


namespace existence_of_solution_l425_425802

variable (n : ℕ) (x : ℝ) (x_i : Fin n → ℝ) (a : ℝ)

theorem existence_of_solution
  (h : ∀ i, 0 ≤ x_i i ∧ x_i i ≤ 3)
  (hx : 0 ≤ x ∧ x ≤ 3) :
  (∑ i, (| x - x_i i |) = a * n) ↔ (a = 3 / 2) :=
by sorry

end existence_of_solution_l425_425802


namespace find_f_1988_l425_425825

def f : ℕ+ → ℕ+ := sorry

axiom functional_equation (m n : ℕ+) : f (f m + f n) = m + n

theorem find_f_1988 : f 1988 = 1988 :=
by sorry

end find_f_1988_l425_425825


namespace smallest_positive_b_exists_l425_425435

theorem smallest_positive_b_exists :
  ∃ (b : ℕ), b > 0 ∧ 
  (∃ (Q : ℤ[X]), 
  (Q.eval 1 = b) ∧ 
  (Q.eval 4 = b) ∧ 
  (Q.eval 7 = b) ∧ 
  (Q.eval 10 = b) ∧ 
  (Q.eval 2 = -2 * b) ∧ 
  (Q.eval 5 = -2 * b) ∧ 
  (Q.eval 8 = -2 * b) ∧ 
  (Q.eval 11 = -2 * b) ∧ 
  (Q.eval 3 = 3 * b) ∧ 
  (Q.eval 6 = 3 * b) ∧ 
  (Q.eval 9 = 3 * b) ∧ 
  (Q.eval 12 = 3 * b)) ∧ 
  ∀ b' (hb' : b' > 0), 
    (∃ (Q' : ℤ[X]), 
    (Q'.eval 1 = b') ∧ 
    (Q'.eval 4 = b') ∧ 
    (Q'.eval 7 = b') ∧ 
    (Q'.eval 10 = b') ∧ 
    (Q'.eval 2 = -2 * b') ∧ 
    (Q'.eval 5 = -2 * b') ∧ 
    (Q'.eval 8 = -2 * b') ∧ 
    (Q'.eval 11 = -2 * b') ∧ 
    (Q'.eval 3 = 3 * b') ∧ 
    (Q'.eval 6 = 3 * b') ∧ 
    (Q'.eval 9 = 3 * b') ∧ 
    (Q'.eval 12 = 3 * b')) → b' ≥ b :=
begin
  sorry
end

end smallest_positive_b_exists_l425_425435


namespace xy_equals_one_l425_425444

-- Define the mathematical theorem
theorem xy_equals_one (x y : ℝ) (h : x + y = 1 / x + 1 / y) (h₂ : x + y ≠ 0) : x * y = 1 := 
by
  sorry

end xy_equals_one_l425_425444


namespace PQRS_area_l425_425545

-- Define the coordinates of points P, Q, R, and S
def P : ℝ × ℝ := (-4, 2)
def Q : ℝ × ℝ := (4, 2)
def R : ℝ × ℝ := (4, -2)
def S : ℝ × ℝ := (-4, -2)

-- Define the width and height of the rectangle
def width (A B : ℝ × ℝ) : ℝ := abs (B.1 - A.1)
def height (A B : ℝ × ℝ) : ℝ := abs (A.2 - B.2)

-- Define the area of the rectangle
def area (A B C D : ℝ × ℝ) : ℝ := width A B * height A D

-- Statement of the theorem
theorem PQRS_area : area P Q R S = 32 := by
  sorry

end PQRS_area_l425_425545


namespace number_of_valid_integers_l425_425275

theorem number_of_valid_integers : 
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ t u : ℕ, n = 10 * t + u ∧ n - (10 * u + t) = 9}.finite.to_finset.card = 8 := 
sorry

end number_of_valid_integers_l425_425275


namespace angle_sine_condition_l425_425553

-- Declaration of angles A and B in the scope of a triangle such that 0 < A, B and A + B < π
theorem angle_sine_condition {A B : ℝ} (hA : 0 < A) (hB : 0 < B) (hSum : A + B < π) :
  A < B ↔ Real.sin A < Real.sin B :=
begin
  sorry,
end

end angle_sine_condition_l425_425553


namespace best_fit_model_is_model3_l425_425896

variable R2_model1 R2_model2 R2_model3 R2_model4 : ℝ

axiom R2_model1_val : R2_model1 = 0.25
axiom R2_model2_val : R2_model2 = 0.50
axiom R2_model3_val : R2_model3 = 0.98
axiom R2_model4_val : R2_model4 = 0.80

theorem best_fit_model_is_model3 :
  R2_model3 = 0.98 ∧
  (R2_model3 - 1).abs < (R2_model1 - 1).abs ∧
  (R2_model3 - 1).abs < (R2_model2 - 1).abs ∧
  (R2_model3 - 1).abs < (R2_model4 - 1).abs :=
by
  sorry

end best_fit_model_is_model3_l425_425896


namespace prob_white_ball_second_l425_425532

structure Bag :=
  (black_balls : ℕ)
  (white_balls : ℕ)

def total_balls (bag : Bag) := bag.black_balls + bag.white_balls

def prob_white_second_after_black_first (bag : Bag) : ℚ :=
  if bag.black_balls > 0 ∧ bag.white_balls > 0 ∧ total_balls bag > 1 then
    (bag.white_balls : ℚ) / (total_balls bag - 1)
  else 0

theorem prob_white_ball_second 
  (bag : Bag)
  (h_black : bag.black_balls = 4)
  (h_white : bag.white_balls = 3)
  (h_total : total_balls bag = 7) :
  prob_white_second_after_black_first bag = 1 / 2 :=
by
  sorry

end prob_white_ball_second_l425_425532


namespace angle_2a_neg3b_l425_425881

-- Define the conditions of the problem
variables {a b : ℝ} -- Assume vectors a and b are defined in ℝ

-- Define the angle between two original vectors
axiom angle_between_a_b : real.angle = 30 

-- Prove the angle between scaled vectors
theorem angle_2a_neg3b : ∀ (a b : ℝ), angle_between (2 * a) (-3 * b) = 150 :=
by
  sorry -- proof goes here

end angle_2a_neg3b_l425_425881


namespace chuck_vs_dave_ride_time_l425_425409

theorem chuck_vs_dave_ride_time (D E : ℕ) (h1 : D = 10) (h2 : E = 65) (h3 : E = 13 * C / 10) :
  (C / D = 5) :=
by
  sorry

end chuck_vs_dave_ride_time_l425_425409


namespace range_of_quadratic_l425_425877

theorem range_of_quadratic (a b c : ℝ) (h_a : a > 0) :
  let f := λ x : ℝ, a * x^2 + b * x + c in
  let f_neg1 := f (-1) in
  let f_2 := f 2 in
  let x_v := -b / (2 * a) in
  let f_x_v := c - (b^2 / (4 * a)) in
  -1 ≤ x_v ∧ x_v ≤ 2 →
  (set.range (λ x : ℝ, f x)) =
  set.Icc (min (min f_neg1 f_x_v) f_2) (max (max f_neg1 f_x_v) f_2) :=
sorry

end range_of_quadratic_l425_425877


namespace find_angle_QRS_l425_425172

variable (P Q R S : Type)
variable [linear_ordered_field P]
variable [linear_ordered_field Q]
variable [linear_ordered_field R]
variable [linear_ordered_field S]

-- Conditions
variable (PQ SQ QR: P)
variable (angle_SPQ angle_RSQ : Q)
axiom PQ_eq_SQ : PQ = SQ
axiom PQ_eq_QR : PQ = QR
axiom angle_SPQ_eq_2_angle_RSQ : angle_SPQ = 2 * angle_RSQ

-- To prove
theorem find_angle_QRS : angle_RSQ = 30 :=
by
  -- Proof skipped
  sorry

end find_angle_QRS_l425_425172


namespace tangent_line_at_point_l425_425790

/-- 
The curve is defined by y = x^3 - x + 3
The point of tangency is (1, 3)
Prove the equation of the tangent line to the curve at the point (1, 3) is 2x - y + 1 = 0
-/
theorem tangent_line_at_point : 
  let y (x : ℝ) := x^3 - x + 3 in
  let x_tangent := 1 in let y_tangent := 3 in
  let slope := (deriv y) x_tangent in
  slope = 2 ∧ y_tangent = y x_tangent →
  ∃ m b, (m = slope ∧ b = y_tangent - m * x_tangent ∧ (∀ x y, y = m * x + b ↔ 2 * x - y + 1 = 0)) :=
begin
  sorry
end

end tangent_line_at_point_l425_425790


namespace sqrt_meaningful_range_l425_425526

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y = √(x - 4)) ↔ x ≥ 4 :=
sorry

end sqrt_meaningful_range_l425_425526


namespace solve_for_x_l425_425799

theorem solve_for_x (x : ℝ) (h : sqrt (x + 12) = 10) : x = 88 :=
by
  sorry

end solve_for_x_l425_425799


namespace prime_divisors_distinct_l425_425607

theorem prime_divisors_distinct (n : ℕ) (h : n > 0) :
  ∀ k ∈ (finset.range n).map (λ i, i + 1), 
  ∃ p_k : ℕ, prime p_k ∧ p_k ∣ (factorial n + k) ∧ 
    (∀ j ∈ (finset.range n).map (λ i, i + 1), j ≠ k → ¬ (p_k ∣ (factorial n + j))) :=
by
  sorry

end prime_divisors_distinct_l425_425607


namespace problem_statement_l425_425584

def f : ℕ → ℕ 
noncomputable def N := (2014 : ℕ) ^ (2014 ^ 2014)

theorem problem_statement (f : ℕ → ℕ) :
  (∀ m n, (gcd (f m) (f n)) ≤ (gcd m n) ^ 2014) →
  (∀ n, n ≤ f n ∧ f n ≤ n + 2014) →
  ∃ N, ∀ n, n ≥ N → f n = n :=
by
  intros h1 h2
  use N
  intro n hn
  sorry

end problem_statement_l425_425584


namespace aiden_arrival_time_l425_425214

variables (a b c : ℝ)

theorem aiden_arrival_time (
  h1 : a + b + c = 120,
  h2 : 0.5 * a + 1.25 * b + 0.5 * c = 96,
  h3 : 1.25 * a + 0.75 * b + 1.25 * c = 126
) : 1.25 * a + 0.75 * b + 1.25 * c = 126 :=
by
  sorry

end aiden_arrival_time_l425_425214


namespace maggie_earnings_correct_l425_425587

def subscriptions_sold_to_parents : ℕ := 4
def subscriptions_sold_to_grandfather : ℕ := 1
def subscriptions_sold_to_next_door_neighbor : ℕ := 2
def subscriptions_sold_to_another_neighbor : ℕ := 2 * subscriptions_sold_to_next_door_neighbor
def price_per_subscription : ℕ := 5
def family_bonus_per_subscription : ℕ := 2
def neighbor_bonus_per_subscription : ℕ := 1
def base_bonus_threshold : ℕ := 10
def base_bonus : ℕ := 10
def extra_bonus_per_subscription : ℝ := 0.5

-- Define total subscriptions sold
def total_subscriptions_sold : ℕ := 
  subscriptions_sold_to_parents + subscriptions_sold_to_grandfather + 
  subscriptions_sold_to_next_door_neighbor + subscriptions_sold_to_another_neighbor

-- Define earnings from subscriptions
def earnings_from_subscriptions : ℕ := total_subscriptions_sold * price_per_subscription

-- Define bonuses
def family_bonus : ℕ :=
  (subscriptions_sold_to_parents + subscriptions_sold_to_grandfather) * family_bonus_per_subscription

def neighbor_bonus : ℕ := 
  (subscriptions_sold_to_next_door_neighbor + subscriptions_sold_to_another_neighbor) * neighbor_bonus_per_subscription

def total_bonus : ℕ := family_bonus + neighbor_bonus

-- Define additional boss bonus
def additional_boss_bonus : ℝ := 
  if total_subscriptions_sold > base_bonus_threshold then 
    base_bonus + extra_bonus_per_subscription * (total_subscriptions_sold - base_bonus_threshold) 
  else 0

-- Define total earnings
def total_earnings : ℝ :=
  earnings_from_subscriptions + total_bonus + additional_boss_bonus

-- Theorem statement
theorem maggie_earnings_correct : total_earnings = 81.5 :=
by
  unfold total_earnings
  unfold earnings_from_subscriptions
  unfold total_bonus
  unfold family_bonus
  unfold neighbor_bonus
  unfold additional_boss_bonus
  unfold total_subscriptions_sold
  simp
  norm_cast
  sorry

end maggie_earnings_correct_l425_425587


namespace hyperbola_focal_length_l425_425259

theorem hyperbola_focal_length (m : ℝ) (h : 4 - m^2 > 0) : 
  let a_sq := m^2 + 5,
      b_sq := 4 - m^2,
      c_sq := a_sq + b_sq,
      c := real.sqrt c_sq,
      focal_length := 2 * c in
  focal_length = 6 :=
by
  sorry

end hyperbola_focal_length_l425_425259


namespace beetles_on_diagonal_l425_425913

theorem beetles_on_diagonal (n : ℕ) (a : Fin n → Fin n → ℤ) 
    (h1 : ∀ i j, |a i j - a (i+1) j| ≤ 1)
    (h2 : ∀ i j, |a i j - a i (j+1)| ≤ 1) : 
  ∃ k : ℕ, ∃ l : ℕ, (k < n ∧ l < n ∧ (∀ d : Fin n, a (k + d) (l + d) = a k l)) :=
sorry

end beetles_on_diagonal_l425_425913


namespace tangent_line_at_A_tangent_line_through_B_l425_425470

open Real

noncomputable def f (x : ℝ) : ℝ := 4 / x
noncomputable def f' (x : ℝ) : ℝ := -4 / (x^2)

theorem tangent_line_at_A : 
  ∃ m b, m = -1 ∧ b = 4 ∧ (∀ x, 1 ≤ x → (x + b = 4)) :=
sorry

theorem tangent_line_through_B :
  ∃ m b, m = 4 ∧ b = -8 ∧ (∀ x, 1 ≤ x → (4*x + b = 8)) :=
sorry

end tangent_line_at_A_tangent_line_through_B_l425_425470


namespace absolute_value_of_slope_l425_425310

def line_through_point (m : ℝ) (x₀ y₀ : ℝ) : ℝ := m * x₀ + (y₀ - m * x₀)

def distance_from_line (m b x₀ y₀ : ℝ) : ℝ := |m * x₀ - y₀ + b| / Real.sqrt (m^2 + 1)

def circle1_center : ℝ × ℝ := (0, 0)
def circle1_radius : ℝ := 4
def circle2_center : ℝ × ℝ := (4, 6)
def circle2_radius : ℝ := 3
def point_on_line : ℝ × ℝ := (2, 1)

theorem absolute_value_of_slope : 
  ∃ m : ℝ, 
    let b := line_through_point m (point_on_line.1) (point_on_line.2) in
    distance_from_line m b (circle1_center.1) (circle1_center.2) = circle1_radius ∧
    distance_from_line m b (circle2_center.1) (circle2_center.2) = circle2_radius :=
sorry

end absolute_value_of_slope_l425_425310


namespace fraction_meaningful_l425_425643

theorem fraction_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 3)) ↔ x ≠ 3 :=
by
  sorry

end fraction_meaningful_l425_425643


namespace oddly_powerful_less_than_5000_count_l425_425404

noncomputable def is_oddly_powerful (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 1 ∧ b % 2 = 1 ∧ a^b = n

theorem oddly_powerful_less_than_5000_count : 
  ({ n : ℕ | n < 5000 ∧ is_oddly_powerful n }.card = 24) :=
sorry

end oddly_powerful_less_than_5000_count_l425_425404


namespace repeating_decimals_sum_l425_425012

theorem repeating_decimals_sum : 
  (0.3333333333333333 : ℝ) + (0.0404040404040404 : ℝ) + (0.005005005005005 : ℝ) = (14 / 37 : ℝ) :=
by {
  sorry
}

end repeating_decimals_sum_l425_425012


namespace rectangle_diagonal_length_l425_425838

theorem rectangle_diagonal_length :
  ∀ (length width diagonal : ℝ), length = 6 ∧ length * width = 48 ∧ diagonal = Real.sqrt (length^2 + width^2) → diagonal = 10 :=
by
  intro length width diagonal
  rintro ⟨hl, area_eq, diagonal_eq⟩
  sorry

end rectangle_diagonal_length_l425_425838


namespace three_digit_perfect_squares_divisible_by_4_l425_425126

/-- Proving the number of three-digit perfect squares that are divisible by 4 is 6 -/
theorem three_digit_perfect_squares_divisible_by_4 : 
  (Finset.card (Finset.filter (λ n, n ∣ 4) (Finset.image (λ k, k * k) (Finset.Icc 10 31)))) = 6 :=
by
  sorry

end three_digit_perfect_squares_divisible_by_4_l425_425126


namespace fifth_largest_divisor_of_1000800000_l425_425316

noncomputable def fifth_largest_divisor (n : ℕ) : ℕ :=
  let divisors := (List.range (n + 1)).filter (λ d, n % d = 0)
  List.reverse divisors !! 4

theorem fifth_largest_divisor_of_1000800000 :
  fifth_largest_divisor 1000800000 = 62550000 :=
by
  sorry

end fifth_largest_divisor_of_1000800000_l425_425316


namespace quadratic_eq_solution_1_quadratic_eq_solution_2_l425_425627

theorem quadratic_eq_solution_1 :
    ∀ (x : ℝ), x^2 - 8*x + 1 = 0 ↔ x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15 :=
by 
  sorry

theorem quadratic_eq_solution_2 :
    ∀ (x : ℝ), x * (x - 2) - x + 2 = 0 ↔ x = 1 ∨ x = 2 :=
by 
  sorry

end quadratic_eq_solution_1_quadratic_eq_solution_2_l425_425627


namespace second_fragment_speed_l425_425355

-- Definitions for initial condition of the firework
def initial_vertical_speed : ℝ := 20 -- m/s
def time_until_explosion : ℝ := 1 -- second
def gravity : ℝ := 10 -- m/s^2
def mass_ratio : ℝ := 1 / 2 -- Ratio of masses ( smaller:larger = 1:2 )
def smaller_fragment_horizontal_speed : ℝ := 16 -- m/s

-- Definition for the magnitude of the speed of the second fragment immediately after the explosion
def magnitude_speed_of_second_fragment (v_x v_y : ℝ) : ℝ := (v_x^2 + v_y^2).sqrt

-- Calculate initial velocity just before explosion
def initial_velocity_before_explosion : ℝ := initial_vertical_speed - gravity * time_until_explosion

-- Function to calculate resultant magnitude of the velocity of the larger fragment
def larger_fragment_velocity_magnitude : ℝ := magnitude_speed_of_second_fragment (-8) 10 -- v_x = -8 m/s, v_y = 10 m/s

theorem second_fragment_speed : larger_fragment_velocity_magnitude = 17 := by
  sorry

end second_fragment_speed_l425_425355


namespace brownies_maximum_l425_425493

theorem brownies_maximum (m n : ℕ) (h1 : (m - 2) * (n - 2) = 2 * (2 * m + 2 * n - 4)) :
  m * n ≤ 144 :=
sorry

end brownies_maximum_l425_425493


namespace range_of_m_l425_425852

theorem range_of_m (m : ℝ) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) 
  (h : ∀ x ∈ Icc 0 1, 2 * m - 1 < x * (m^2 - 1)) : m < 0 :=
sorry

end range_of_m_l425_425852


namespace symmetry_of_points_l425_425594

noncomputable theory

open EuclideanGeometry

variables {A B C M N D : Point}

def isosceles_triangle (A B C : Point) : Prop :=
triangle A B C ∧ (distance A B = distance A C)

def congruent_triangles (A B M N : Point) : Prop :=
triangle A B M ∧ triangle A N C ∧ (distance A M = distance A N)

def angle_bisector (A B C D : Point) : Prop :=
is_angle_bisector A B C D

theorem symmetry_of_points 
  (h_isosceles : isosceles_triangle A B C)
  (h_congruent : congruent_triangles A B M N)
  (h_bisector : angle_bisector A B C D) :
  symmetric_about_line M N D :=
sorry

end symmetry_of_points_l425_425594
