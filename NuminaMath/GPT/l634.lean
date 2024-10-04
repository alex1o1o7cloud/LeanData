import Mathlib
import Mathlib.Algebra
import Mathlib.Algebra.Group.Defs
import Mathlib.Algebra.NumberTheory.JacobiSymbol
import Mathlib.Algebra.Order
import Mathlib.Algebra.Parabola
import Mathlib.Algebra.Polynomial
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Analysis.SpecialFunctions.Integrals
import Mathlib.Analysis.SpecialFunctions.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Integral
import Mathlib.Analysis.Trigonometric.Basic
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Combinatorics.Permutations.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Graph.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Factorial
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Angle
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Geometry.Circle.Basic
import Mathlib.Geometry.Euclidean.Basic
import Mathlib.NumberTheory.ArithmeticFunction
import Mathlib.NumberTheory.Basic
import Mathlib.Tactic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Topology.Basic
import RealInnerProductSpace
import data.fintype.basic
import data.real.basic
import tactic

namespace matrix_determinant_l634_634081

theorem matrix_determinant (n : ℕ) (H : n > 0) :
  let A := λ i j : ℕ, if i = j then 8 else 3 in
  let det_A := (5^(n-1) * (3 * n + 5)) in
  Matrix.det (Matrix.of_fn A) = det_A ∧ (det_A % 10 = 0 ↔ n % 2 = 1) :=
by
  sorry

end matrix_determinant_l634_634081


namespace quadrilateral_area_l634_634626

noncomputable def area_quadrilateral (A B C D E : Point) (AC BD : Line)
  (h1 : ∠ ABC = 120)
  (h2 : ∠ ACD = 90)
  (h3 : dist A C = 10 * real.sqrt 2)
  (h4 : dist C D = 20)
  (h5 : line_intersection AC BD = E)
  (h6 : dist A E = 5 * real.sqrt 2): real :=
25 * real.sqrt 3 + 100 * real.sqrt 2

theorem quadrilateral_area (A B C D E : Point) (AC BD : Line)
  (h1 : ∠ ABC = 120)
  (h2 : ∠ ACD = 90)
  (h3 : dist A C = 10 * real.sqrt 2)
  (h4 : dist C D = 20)
  (h5 : line_intersection AC BD = E)
  (h6 : dist A E = 5 * real.sqrt 2) :
  area_quadrilateral A B C D E AC BD h1 h2 h3 h4 h5 h6 = 25 * real.sqrt 3 + 100 * real.sqrt 2 :=
by
  sorry

end quadrilateral_area_l634_634626


namespace correct_property_of_rectangle_l634_634689

theorem correct_property_of_rectangle 
  (Q : Type) 
  [IsQuadrilateral Q] 
  (h1 : ∀ q : Q, hasEqualDiagonals q → ¬isRectangle q) 
  (h2 : ∀ q : Q, hasPerpendicularDiagonals q → ¬isRhombus q) 
  (h3 : ∀ p : Parallelogram, bisectEachOther (diagonals p)) 
  (h4 : ∀ r : Rectangle, bisectEachOther (diagonals r) ∧ equalDiagonals (diagonals r))
  : isCorrectStatement(Q, "D") := 
sorry

end correct_property_of_rectangle_l634_634689


namespace gcd_lcm_eq_implies_eq_l634_634190

theorem gcd_lcm_eq_implies_eq (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  gcd a b + lcm a b = gcd a c + lcm a c → b = c :=
begin
  sorry
end

end gcd_lcm_eq_implies_eq_l634_634190


namespace math_majors_consecutive_probability_l634_634298

noncomputable def probability_math_majors_consecutive (n : ℕ) :=
  -- Number of favorable arrangements: 7! * 5!
  (Nat.factorial 7 * Nat.factorial 5) /
  -- Total arrangements: 11!
  (Nat.factorial 11)

theorem math_majors_consecutive_probability :
  probability_math_majors_consecutive 12 = 1 / 66 := 
by
  sorry

end math_majors_consecutive_probability_l634_634298


namespace oranges_initially_in_bag_l634_634330

variable (O : ℕ)

def initial_apples := 7
def initial_mangoes := 15
def taken_apples := 2
def taken_mangoes := (2 * initial_mangoes / 3 : ℝ)
def remaining_fruits := 14

-- Luisa's actions
def remaining_apples := initial_apples - taken_apples
def remaining_oranges (initial_oranges : ℕ) := initial_oranges - 2 * taken_apples
def remaining_mangoes := initial_mangoes - taken_mangoes

theorem oranges_initially_in_bag (O : ℕ) (h : O = 8) 
  (h_initial_fruits_sum : remaining_apples + remaining_oranges O + remaining_mangoes = remaining_fruits) :
  O = 8 := by
  sorry

end oranges_initially_in_bag_l634_634330


namespace perfect_square_trinomial_l634_634878

theorem perfect_square_trinomial (m : ℝ) : (∃ b : ℝ, (x^2 - 6 * x + m) = (x + b) ^ 2) → m = 9 :=
by
  sorry

end perfect_square_trinomial_l634_634878


namespace compare_abc_l634_634379

variable {f : ℝ → ℝ}
variable (a b c : ℝ)
variable hdiff : Differentiable ℝ f
variable hineq : ∀ x, 1 < x → (x - 1) * (derivative f x) - f x > 0
variable ha : a = f 2
variable hb : b = (1 / 2) * f 3
variable hc : c = (Real.sqrt 2 + 1) * f (Real.sqrt 2)

theorem compare_abc :
  c < a ∧ a < b :=
  sorry

end compare_abc_l634_634379


namespace juliet_apartment_units_digit_is_7_l634_634302

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

def has_units_digit_7 (n : ℕ) : Prop := n % 10 = 7

def exactly_three_of_four (p1 p2 p3 p4 : Prop) : Prop :=
  ((p1 ∧ p2 ∧ p3 ∧ ¬p4) ∨ (p1 ∧ p2 ∧ ¬p3 ∧ p4) ∨ (p1 ∧ ¬p2 ∧ p3 ∧ p4) ∨ (¬p1 ∧ p2 ∧ p3 ∧ p4))

theorem juliet_apartment_units_digit_is_7 :
  ∃ n : ℕ, is_two_digit n ∧
           exactly_three_of_four (Nat.Prime n) (Odd n) (n % 3 = 0) (∃ d : ℕ, d < 10 ∧ (d = 7 ∧ to_digits 10 n = (n / 10, d))). 
 qed
sorry

end juliet_apartment_units_digit_is_7_l634_634302


namespace maxValue_a1_l634_634919

variable (a_1 q : ℝ)

def isGeometricSequence (a_1 q : ℝ) : Prop :=
  a_1 ≥ 1 ∧ a_1 * q ≤ 2 ∧ a_1 * q^2 ≥ 3

theorem maxValue_a1 (h : isGeometricSequence a_1 q) : a_1 ≤ 4 / 3 := 
sorry

end maxValue_a1_l634_634919


namespace line_parametric_eq_l634_634785

variable {x y : ℝ}
variable P : ℝ × ℝ := (3, 5)
variable d : ℝ × ℝ := (4, 2)

theorem line_parametric_eq :
  ∃ (k : ℝ), (x, y) = (P.1 + k * d.1, P.2 + k * d.2) ↔ (x - 3) / 4 = (y - 5) / 2 :=
by
  sorry

end line_parametric_eq_l634_634785


namespace train_length_proof_l634_634738

-- Define the given parameters
def speed_kmph : ℝ := 72
def time_sec : ℝ := 41.24670026397888
def length_bridge_m : ℝ := 660

-- Convert speed from km/h to m/s
def speed_mps : ℝ := (speed_kmph * 1000) / 3600

-- Computed total distance covered
def total_distance : ℝ := speed_mps * time_sec

-- Length of the train
def length_of_train := total_distance - length_bridge_m

-- The statement that needs to be proved
theorem train_length_proof : length_of_train = 165.9340052795776 := by
  unfold length_of_train
  unfold total_distance
  unfold speed_mps
  calc
    ((speed_kmph * 1000) / 3600) * time_sec - length_bridge_m
      = 20 * 41.24670026397888 - 660 := by norm_num
      ... = 825.9340052795776 - 660 := by norm_num
      ... = 165.9340052795776 := by norm_num

end train_length_proof_l634_634738


namespace cos_minus_sin_l634_634469

-- Define the conditions given in the problem
def cos_alpha (α : ℝ) : ℝ := 3 / 5
def sin_alpha (α : ℝ) : ℝ := -4 / 5

-- State the goal to prove
theorem cos_minus_sin (α : ℝ) : cos_alpha α - sin_alpha α = 7 / 5 := by
  sorry

end cos_minus_sin_l634_634469


namespace calculate_angles_and_side_l634_634326

theorem calculate_angles_and_side (a b B : ℝ) (h_a : a = Real.sqrt 3) (h_b : b = Real.sqrt 2) (h_B : B = 45) :
  ∃ A C c, (A = 60 ∧ C = 75 ∧ c = (Real.sqrt 6 + Real.sqrt 2) / 2) ∨ (A = 120 ∧ C = 15 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2) :=
by sorry

end calculate_angles_and_side_l634_634326


namespace friends_division_l634_634873

def num_ways_to_divide (total_friends teams : ℕ) : ℕ :=
  4^8 - (Nat.choose 4 1) * 3^8 + (Nat.choose 4 2) * 2^8 - (Nat.choose 4 3) * 1^8

theorem friends_division (total_friends teams : ℕ) (h_friends : total_friends = 8) (h_teams : teams = 4) :
  num_ways_to_divide total_friends teams = 39824 := by
  sorry

end friends_division_l634_634873


namespace find_complex_number_l634_634707

theorem find_complex_number (z : ℂ) (h : 4 * z + 2 * conj z = 3 * real.sqrt 3 + complex.I) : 
  z = (real.sqrt 3) / 2 + (1 / 2) * complex.I :=
sorry

end find_complex_number_l634_634707


namespace house_painting_cost_l634_634566

theorem house_painting_cost :
  let judson_contrib := 500.0
  let kenny_contrib_euros := judson_contrib * 1.2 / 1.1
  let camilo_contrib_pounds := (kenny_contrib_euros * 1.1 + 200.0) / 1.3
  let camilo_contrib_usd := camilo_contrib_pounds * 1.3
  judson_contrib + kenny_contrib_euros * 1.1 + camilo_contrib_usd = 2020.0 := 
by {
  sorry
}

end house_painting_cost_l634_634566


namespace draw_triangle_with_given_perimeter_l634_634471

noncomputable def draw_triangle_with_perimeter (L A K M : Point) (p : ℝ) : Triangle :=
sorry

theorem draw_triangle_with_given_perimeter (L A K M : Point) (p : ℝ) :
  let ABC := draw_triangle_with_perimeter L A K M p in
  perimeter ABC = 2 * p :=
sorry

end draw_triangle_with_given_perimeter_l634_634471


namespace sum_of_first_8_terms_l634_634918

variable (a r : ℝ)
variable geom_seq : ℕ → ℝ := λ n, a * r^n
variable sum_first_n_terms : ℕ → ℝ
variable sum_first_3_terms : a * (1 + r + r^2) = 13
variable sum_first_5_terms : a * (1 + r + r^2 + r^3 + r^4) = 121

def sum_geom_seq (n : ℕ) : ℝ :=
  if r = 1 then a * n else a * (1 - r^n) / (1 - r)

theorem sum_of_first_8_terms
  (h3 : a * (1 + r + r^2) = 13)
  (h5 : a * (1 + r^2 + r^2 + r^3 + r^4) = 121) :
  sum_geom_seq a r 8 = 3280 := by
  sorry

end sum_of_first_8_terms_l634_634918


namespace change_is_24_l634_634614

-- Define the prices and quantities
def price_basketball_card : ℕ := 3
def price_baseball_card : ℕ := 4
def num_basketball_cards : ℕ := 2
def num_baseball_cards : ℕ := 5
def money_paid : ℕ := 50

-- Define the total cost
def total_cost : ℕ := (num_basketball_cards * price_basketball_card) + (num_baseball_cards * price_baseball_card)

-- Define the change received
def change_received : ℕ := money_paid - total_cost

-- Prove that the change received is $24
theorem change_is_24 : change_received = 24 := by
  -- the proof will go here
  sorry

end change_is_24_l634_634614


namespace petya_wins_probability_l634_634995

def stones_initial : ℕ := 16

def valid_moves : set ℕ := {1, 2, 3, 4}

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def petya_random_choice (move : ℕ) : Prop := move ∈ valid_moves

def computer_optimal_strategy (n : ℕ) : ℕ :=
  (n % 5)

noncomputable def probability_petya_wins : ℚ :=
  (1 / 4) ^ 4

theorem petya_wins_probability :
  probability_petya_wins = 1 / 256 := 
sorry

end petya_wins_probability_l634_634995


namespace lcm_24_150_is_600_l634_634784

noncomputable def lcm_24_150 : ℕ :=
  let a := 24
  let b := 150
  have h₁ : a = 2^3 * 3 := by sorry
  have h₂ : b = 2 * 3 * 5^2 := by sorry
  Nat.lcm a b

theorem lcm_24_150_is_600 : lcm_24_150 = 600 := by
  -- Use provided primes conditions to derive the result
  sorry

end lcm_24_150_is_600_l634_634784


namespace tan_add_formula_l634_634049

noncomputable def tan_subtract (a b : ℝ) : ℝ := (Real.tan a - Real.tan b) / (1 + Real.tan a * Real.tan b)
noncomputable def tan_add (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)

theorem tan_add_formula (α : ℝ) (hf : tan_subtract α (Real.pi / 4) = 1 / 4) :
  tan_add α (Real.pi / 4) = -4 :=
by
  sorry

end tan_add_formula_l634_634049


namespace guilty_prob_l634_634908

-- Defining suspects
inductive Suspect
| A
| B
| C

open Suspect

-- Constants for the problem
def looks_alike (x y : Suspect) : Prop :=
(x = A ∧ y = B) ∨ (x = B ∧ y = A)

def timid (x : Suspect) : Prop :=
x = A ∨ x = B

def bold (x : Suspect) : Prop :=
x = C

def alibi_dover (x : Suspect) : Prop :=
x = A ∨ x = B

def needs_accomplice (x : Suspect) : Prop :=
timid x

def works_alone (x : Suspect) : Prop :=
bold x

def in_bar_during_robbery (x : Suspect) : Prop :=
x = A ∨ x = B

-- Theorem to be proved
theorem guilty_prob :
  ∃ x : Suspect, (x = B) ∧ ∀ y : Suspect, y ≠ B → 
    ((y = A ∧ timid y ∧ needs_accomplice y ∧ in_bar_during_robbery y) ∨
    (y = C ∧ bold y ∧ works_alone y)) :=
by
  sorry

end guilty_prob_l634_634908


namespace min_value_of_diff_squares_l634_634199

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))

theorem min_value_of_diff_squares (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  ∃ minimum_value, minimum_value = 36 ∧ ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → (C x y z)^2 - (D x y z)^2 ≥ minimum_value :=
sorry

end min_value_of_diff_squares_l634_634199


namespace cube_root_squared_l634_634686

noncomputable def solve_for_x (x : ℝ) : Prop :=
  (x^(1/3))^2 = 81 → x = 729

theorem cube_root_squared (x : ℝ) :
  solve_for_x x :=
by
  sorry

end cube_root_squared_l634_634686


namespace part1_domain_of_f_part2_inequality_l634_634964

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (abs (x + 1) + abs (x - 1) - 4)

theorem part1_domain_of_f : {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by 
  sorry

theorem part2_inequality (a b : ℝ) (h_a : -2 < a) (h_a' : a < 2) (h_b : -2 < b) (h_b' : b < 2) 
  : 2 * abs (a + b) < abs (4 + a * b) :=
by 
  sorry

end part1_domain_of_f_part2_inequality_l634_634964


namespace part1_magnitude_of_vector_part2_projection_of_vector_l634_634078

variables (a b : ℝ → ℝ → ℝ → ℝ) -- Assuming vectors in 3D
variables (angle_ab : Real) (norm_a norm_b : Real)

#check dot_product -- Let's assume we have the dot product implementation
#check magnitude -- Let's assume we have the magnitude implementation
#check projection -- Let's assume we have the projection implementation
#check cos -- We have cosine function implemented

def given_conditions : Prop :=
  angle_ab = 120 ∧
  norm_a = 1 ∧
  norm_b = 3

noncomputable def magnitude_result : Real :=
  magnitude (λ t, 5 * (a t) - (b t))

noncomputable def projection_result : Real :=
  projection (λ t, 2 * (a t) + (b t)) (λ t, b t)

theorem part1_magnitude_of_vector : given_conditions → magnitude_result = 7 := by
  sorry

theorem part2_projection_of_vector : given_conditions → projection_result = 2 := by
  sorry

end part1_magnitude_of_vector_part2_projection_of_vector_l634_634078


namespace fill_nxn_square_with_conditions_l634_634856

open Set

variables {A B : Set ℝ} {n : ℕ}
def S (X : Set ℝ) : ℝ := ∑ i in X, i

noncomputable def k : ℕ := (A ∩ B).card

theorem fill_nxn_square_with_conditions (hA : A.card = n) (hB : B.card = n) (h_diff : A ≠ B) (h_sum : S A = S B) :
  ∃ (M : Matrix (Fin n) (Fin n) ℝ), 
    (∀ (i : Fin n), ∑ j, M i j ∈ A) ∧
    (∀ (j : Fin n), ∑ i, M i j ∈ A) ∧
    (Fintype.card {ij : (Fin n) × (Fin n) // M ij.1 ij.2 = 0} ≥ (n-1)^(2) + k) := 
sorry

end fill_nxn_square_with_conditions_l634_634856


namespace find_a_l634_634428

noncomputable def cubic_poly_has_three_distinct_positive_roots (a b : ℝ) : Prop :=
  ∃ r s t : ℝ,
  r > 0 ∧ s > 0 ∧ t > 0 ∧ r ≠ s ∧ s ≠ t ∧ r ≠ t ∧
  12*r^3 + 6*a*r^2 + 8*b*r + a = 0 ∧
  12*s^3 + 6*a*s^2 + 8*b*s + a = 0 ∧
  12*t^3 + 6*a*t^2 + 8*b*t + a = 0

def sum_of_base3_log (r s t : ℝ) : Prop := 
  real.logb 3 r + real.logb 3 s + real.logb 3 t = 5

theorem find_a (a b : ℝ) (h₁ : cubic_poly_has_three_distinct_positive_roots a b)
    (h₂ : ∃ r s t : ℝ, sum_of_base3_log r s t) :
  a = -2916 :=
sorry

end find_a_l634_634428


namespace find_f_neg_2017_l634_634829

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic_function : ∀ x : ℝ, x ≥ 0 → f (x + 2) = f x
axiom log_function : ∀ x : ℝ, 0 ≤ x ∧ x < 2 → f x = Real.log (x + 1) / Real.log 2

theorem find_f_neg_2017 : f (-2017) = 1 := by
  sorry

end find_f_neg_2017_l634_634829


namespace triangle_at_most_one_right_angle_l634_634625

theorem triangle_at_most_one_right_angle :
  (∀ (A B C : ℝ), ∃ (triangle : Triangle A B C), ¬ (triangle.angle_A = π/2 ∧ triangle.angle_B = π/2 ∨ triangle.angle_B = π/2 ∧ triangle.angle_C = π/2 ∨ triangle.angle_A = π/2 ∧ triangle.angle_C = π/2)) :=
by
  -- Assuming toward contradiction that there are at least two right angles in the triangle
  intro h
  -- Introduce a contradiction assumption
  let h := ∃ (A B C : ℝ), ∃ (triangle : Triangle A B C), (triangle.angle_A = π/2 ∧ triangle.angle_B = π/2) ∨ (triangle.angle_B = π/2 ∧ triangle.angle_C = π/2) ∨ (triangle.angle_A = π/2 ∧ triangle.angle_C = π/2)
  -- We will complete the proof by contradiction so leaving it here as sorry
  sorry

end triangle_at_most_one_right_angle_l634_634625


namespace problem1_problem2_problem3_problem4_l634_634729

namespace QuadraticFunctionProof

variables {a b c : ℝ}
def y (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions from the problem statement
def condition1 := y (-1) = -1
def condition2 := y 0 = -7/4
def condition3 := y 1 = -2

-- Problem 1: Prove y(3) = -1
theorem problem1 (ha : a = 1/4) (hb : b = -1/2) (hc : c = -7/4) : y 3 = -1 :=
by sorry

-- Problem 2: Prove minimum value of y and corresponding x
theorem problem2 (ha : a = 1/4) (hb : b = -1/2) (hc : c = -7/4) : ∃ x, x = 1 ∧ y x = -2 :=
by sorry

-- Problem 3: Compare y1 and y2 for given x1 and x2 ranges
theorem problem3 (ha : a = 1/4) (hb : b = -1/2) (hc : c = -7/4) 
  (x1 x2 : ℝ) (h1 : -1 < x1 ∧ x1 < 0) (h2 : 1 < x2 ∧ x2 < 2) : y x1 > y x2 :=
by sorry

-- Problem 4: Prove range of y when 0 ≤ x ≤ 5
theorem problem4 (ha : a = 1/4) (hb : b = -1/2) (hc : c = -7/4) : ∀ x, 0 ≤ x ∧ x ≤ 5 → -2 ≤ y x ∧ y x ≤ 2 :=
by sorry

end QuadraticFunctionProof

end problem1_problem2_problem3_problem4_l634_634729


namespace floor_div_eq_floor_div_floor_l634_634232

theorem floor_div_eq_floor_div_floor {α : ℝ} {d : ℕ} (h₁ : 0 < α) : 
  (⌊α / d⌋ = ⌊⌊α⌋ / d⌋) := 
sorry

end floor_div_eq_floor_div_floor_l634_634232


namespace house_vs_trailer_payment_difference_l634_634179

-- Definitions based on given problem conditions
def house_cost : ℝ := 480000
def trailer_cost : ℝ := 120000
def loan_term_years : ℝ := 20
def months_in_year : ℝ := 12
def total_months : ℝ := loan_term_years * months_in_year

-- Monthly payment calculations
def house_monthly_payment : ℝ := house_cost / total_months
def trailer_monthly_payment : ℝ := trailer_cost / total_months

-- The theorem we need to prove
theorem house_vs_trailer_payment_difference :
  house_monthly_payment - trailer_monthly_payment = 1500 := 
by
  sorry

end house_vs_trailer_payment_difference_l634_634179


namespace sum_first_n_terms_l634_634480

-- Define the sequence a_n using the given conditions
noncomputable def a : ℕ → ℝ
| 0     := 3
| (n+1) := a n + (n+1) + 2^(n+1)

-- Prove the sum of the first n terms equals the given formula
theorem sum_first_n_terms (n : ℕ) : 
  (∑ i in finset.range (n+1), a i) = (1 / 6) * n * (n + 1) * (n + 2) + 2^(n + 2) - 2 * (n + 2) :=
sorry

end sum_first_n_terms_l634_634480


namespace parabola_directrix_correct_l634_634019

noncomputable def parabola_directrix : Prop :=
  let eqn := λ x : ℝ, -3 * x^2 + 6 * x - 5
  let directrix := -23/12
  ∀ x : ℝ, eqn x = y → y = directrix

theorem parabola_directrix_correct : parabola_directrix :=
begin
  sorry
end

end parabola_directrix_correct_l634_634019


namespace ball_radius_approx_l634_634331

/-- Given an oval-shaped hole is 30 cm across at the narrowest point and 10 cm deep,
prove that the radius of the ball that made the hole is approximately 22.1129 cm. -/
theorem ball_radius_approx (h_width : ℝ) (h_depth : ℝ) (r : ℝ)
  (h_w : h_width = 30) (h_d : h_depth = 10) : r ≈ 22.1129 := by
    sorry

end ball_radius_approx_l634_634331


namespace binary_to_decimal_110_eq_6_l634_634766

theorem binary_to_decimal_110_eq_6 : (1 * 2^2 + 1 * 2^1 + 0 * 2^0 = 6) :=
by
  sorry

end binary_to_decimal_110_eq_6_l634_634766


namespace parabola_equation_l634_634324

theorem parabola_equation (x y : ℝ) (h_vertex : (0, 0) = (0, 0)) (h_symmetry : symmetric_about_x_axis) 
    (h_point : (-3, -6) ∈ parabola_points)
    (h_form : ∃ p : ℝ, y^2 = -2 * p * x) :
  y^2 = -12 * x :=
by 
  sorry

end parabola_equation_l634_634324


namespace percentage_increase_in_spending_l634_634516

variables (P Q : ℝ)
-- Conditions
def price_increase (P : ℝ) := 1.25 * P
def quantity_decrease (Q : ℝ) := 0.88 * Q

-- Mathemtically equivalent proof problem in Lean:
theorem percentage_increase_in_spending (P Q : ℝ) : 
  (price_increase P) * (quantity_decrease Q) / (P * Q) = 1.10 :=
by
  sorry

end percentage_increase_in_spending_l634_634516


namespace part1_unique_zero_part2_bound_l634_634476

def f (a : ℝ) (x : ℝ) : ℝ := ln ((2 / x) + a)

def F (a : ℝ) (x : ℝ) : ℝ := f a x - ln ((2 - a) * x + 3 * a - 3)

theorem part1_unique_zero (a : ℝ) :
  (∃ x : ℝ, F a x = 0 ∧ ∀ y : ℝ, F a y = 0 → y = x) ↔ 
  a ∈ Set.Ioc (-1) (4 / 3) ∪ ({2} : Set ℝ) ∪ ({5 / 2} : Set ℝ) :=
sorry

theorem part2_bound (a : ℝ) :
  (∀ m ∈ Set.Icc (3 / 4) 1, ∀ x1 x2 ∈ Set.Icc m (4 * m - 1), 
  |f a x1 - f a x2| ≤ ln 2) ↔ a ∈ Set.Ici (12 - 8 * Real.sqrt 2) :=
sorry

end part1_unique_zero_part2_bound_l634_634476


namespace find_value_l634_634473

theorem find_value (x y : ℝ) (h1 : 3 * x + y = 5) (h2 : x + 3 * y = 8) : 5 * x^2 + 11 * x * y + 5 * y^2 = 89 :=
by
  sorry

end find_value_l634_634473


namespace waiter_total_customers_l634_634754

def numCustomers (T : ℕ) (totalTips : ℕ) (tipPerCustomer : ℕ) (numNoTipCustomers : ℕ) : ℕ :=
  T + numNoTipCustomers

theorem waiter_total_customers
  (T : ℕ)
  (h1 : 3 * T = 6)
  (numNoTipCustomers : ℕ := 5)
  (total := numCustomers T 6 3 numNoTipCustomers) :
  total = 7 := by
  sorry

end waiter_total_customers_l634_634754


namespace appropriate_chart_for_milk_powder_l634_634903

-- Define the chart requirements and the correctness condition
def ChartType := String
def pie : ChartType := "pie"
def line : ChartType := "line"
def bar : ChartType := "bar"

-- The condition we need for our proof
def representsPercentagesWell (chart: ChartType) : Prop :=
  chart = pie

-- The main theorem statement
theorem appropriate_chart_for_milk_powder : representsPercentagesWell pie :=
by
  sorry

end appropriate_chart_for_milk_powder_l634_634903


namespace other_root_of_quadratic_l634_634615

theorem other_root_of_quadratic {z c : ℂ} (h : z^2 = c) (h1 : z = -6 + 3i) :
  -z = 6 - 3i :=
by
  sorry

end other_root_of_quadratic_l634_634615


namespace candy_distribution_powers_of_two_l634_634393

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def children_receive_candies (f : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ x : ℕ, ∃ k : ℕ, (k * (k + 1) / 2) % n = x

theorem candy_distribution_powers_of_two (n : ℕ) (hn : is_power_of_two n) :
  children_receive_candies (λ x, x * (x + 1) / 2 % n) n :=
sorry

end candy_distribution_powers_of_two_l634_634393


namespace part1_a_n_part2_T_20_l634_634586

-- Conditions
def seq : ℕ → ℕ := sorry -- Sequence of positive terms
def S (n : ℕ) := ∑ i in finset.range (n + 1), seq i -- Sum of the first n terms
def a (n : ℕ) : ℕ := seq n -- Term a_n
def b (n : ℕ) : ℚ :=
  if n % 2 = 1 then (a n + 1) / 2
  else 4 / (a n * (a n + 4))
def T (n : ℕ) := ∑ i in finset.range (n + 1), b i -- Sum of the first n terms of b

-- Given conditions in Lean definitions
axiom a1 : a 1 = 1
axiom a2 (n : ℕ) (hn : 2 ≤ n) : 2 * (S n + S (n - 1)) = (a n) ^ 2 + 1

-- Theorem 1: Proving a_n
theorem part1_a_n (n : ℕ) : a n = 2 * n - 1 := sorry

-- Theorem 2: Proving T_20
theorem part2_T_20 : T 20 = 13060 / 129 := sorry

end part1_a_n_part2_T_20_l634_634586


namespace colored_sectors_overlap_l634_634779

/--
Given two disks each divided into 1985 equal sectors, with 200 sectors on each disk colored arbitrarily,
and one disk is rotated by angles that are multiples of 360 degrees / 1985, 
prove that there are at least 80 positions where no more than 20 colored sectors coincide.
-/
theorem colored_sectors_overlap :
  ∀ (disks : ℕ → ℕ) (sectors_colored : ℕ),
  disks 1 = 1985 → disks 2 = 1985 →
  sectors_colored = 200 →
  ∃ (p : ℕ), p ≥ 80 ∧ (∀ (i : ℕ), (i < p → sectors_colored ≤ 20)) := 
sorry

end colored_sectors_overlap_l634_634779


namespace squares_partition_l634_634716

theorem squares_partition {k : ℕ} (S : finset (set (ℝ × ℝ))) :
  (∀ T ⊆ S, T.card = k + 1 → ∃ s t ∈ T, s ∩ t ≠ ∅) →
  ∃ P : finset (finset (set (ℝ × ℝ))), P.card ≤ 2 * k - 1 ∧ 
  (∀ p ∈ P, (∃ x : ℝ × ℝ, ∀ s ∈ p, x ∈ s) ∧ p ≠ ∅) :=
begin
  sorry,
end

end squares_partition_l634_634716


namespace independence_test_assumption_l634_634293

theorem independence_test_assumption :
  assumes independence_test : Bool -- Assume an independence test is being applied
  shows independence_assumption (gender liking_to_participate_in_sports : Type) : Prop :=
  -- State that gender and liking to participate in sports activities are unrelated.
  sorry

end independence_test_assumption_l634_634293


namespace sector_area_l634_634830

theorem sector_area (r α : ℝ) (hr : r = 2) (hα : α = 2) : (1/2) * r^2 * α = 4 :=
by
  rw [hr, hα]
  norm_num
  sorry

end sector_area_l634_634830


namespace parabola_directrix_correct_l634_634020

noncomputable def parabola_directrix : Prop :=
  let eqn := λ x : ℝ, -3 * x^2 + 6 * x - 5
  let directrix := -23/12
  ∀ x : ℝ, eqn x = y → y = directrix

theorem parabola_directrix_correct : parabola_directrix :=
begin
  sorry
end

end parabola_directrix_correct_l634_634020


namespace problem1_problem2_problem3_problem4_l634_634374

theorem problem1 : (-23 + 13 - 12) = -22 := 
by sorry

theorem problem2 : ((-2)^3 / 4 + 3 * (-5)) = -17 := 
by sorry

theorem problem3 : (-24 * (1/2 - 3/4 - 1/8)) = 9 := 
by sorry

theorem problem4 : ((2 - 7) / 5^2 + (-1)^2023 * (1/10)) = -3/10 := 
by sorry

end problem1_problem2_problem3_problem4_l634_634374


namespace rate_per_meter_for_fencing_l634_634259

/-- The length of a rectangular plot is 10 meters more than its width. 
    The cost of fencing the plot along its perimeter at a certain rate per meter is Rs. 1430. 
    The perimeter of the plot is 220 meters. 
    Prove that the rate per meter for fencing the plot is 6.5 Rs. 
 -/
theorem rate_per_meter_for_fencing (width length perimeter cost : ℝ)
  (h_length : length = width + 10)
  (h_perimeter : perimeter = 2 * (width + length))
  (h_perimeter_value : perimeter = 220)
  (h_cost : cost = 1430) :
  (cost / perimeter) = 6.5 := by
  sorry

end rate_per_meter_for_fencing_l634_634259


namespace irrational_a_or_b_l634_634645

-- Define the problem conditions and proof target
theorem irrational_a_or_b (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0)
  (h₂ : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) :
  ¬ (is_rational a ∧ is_rational b) :=
sorry

end irrational_a_or_b_l634_634645


namespace number_of_frogs_four_l634_634524

variable (Anne Brian Chris LeRoy Mike : Prop)
variable (isToad isFrog : Prop → Prop)
variable (Anne_statement : isToad Anne ↔ (exactly_three_toads {Anne, Brian, Chris, LeRoy, Mike}))
variable (Brian_statement : isToad Brian ↔ (isFrog Mike))
variable (Chris_statement : isToad Chris ↔ isToad LeRoy)
variable (LeRoy_statement : isToad LeRoy ↔ isFrog Chris)
variable (Mike_statement : isToad Mike ↔ (at_least_three_frogs {Anne, Brian, Chris, LeRoy, Mike}))

noncomputable def number_of_frogs : ℕ := 
  #{filter (λ x, isFrog x) [Anne, Brian, Chris, LeRoy, Mike]}

theorem number_of_frogs_four (hAnne : Anne)
                             (hBrian : Brian)
                             (hChris : Chris)
                             (hLeRoy : LeRoy)
                             (hMike : Mike)
                             (hToad : isToad)
                             (hFrog : isFrog)
                             (hAnne_statement : hToad Anne ↔ (exactly_three_toads {Anne, Brian, Chris, LeRoy, Mike}))
                             (hBrian_statement : hToad Brian ↔ (hFrog Mike))
                             (hChris_statement : hToad Chris ↔ hToad LeRoy)
                             (hLeRoy_statement : hToad LeRoy ↔ hFrog Chris)
                             (hMike_statement : hToad Mike ↔ (at_least_three_frogs {Anne, Brian, Chris, LeRoy, Mike})) 
  : number_of_frogs = 4 := sorry

end number_of_frogs_four_l634_634524


namespace probability_is_isosceles_right_triangle_l634_634147

open_locale classical
noncomputable theory

definition cxube_vertices : ℕ := 8
definition isosceles_right_triangles : ℕ := 24
definition total_selections : ℕ := (nat.choose 8 3)

theorem probability_is_isosceles_right_triangle :
  (isosceles_right_triangles : ℝ) / total_selections = (3 : ℝ) / 7 :=
begin
  -- Proof omitted
  sorry
end

end probability_is_isosceles_right_triangle_l634_634147


namespace probability_fourth_ball_black_l634_634334

theorem probability_fourth_ball_black :
  let total_balls := 6
  let red_balls := 3
  let black_balls := 3
  let prob_black_first_draw := black_balls / total_balls
  (prob_black_first_draw = 1 / 2) ->
  (prob_black_first_draw = (black_balls / total_balls)) ->
  (black_balls / total_balls = 1 / 2) ->
  1 / 2 = 1 / 2 :=
by
  intros
  sorry

end probability_fourth_ball_black_l634_634334


namespace part1_part2_l634_634475

noncomputable def f (x k : ℝ) : ℝ := (x ^ 2 + k * x + 1) / (x ^ 2 + 1)

theorem part1 (k : ℝ) (h : k = -4) : ∃ x > 0, f x k = -1 :=
  by sorry -- Proof goes here

theorem part2 (k : ℝ) : (∀ (x1 x2 x3 : ℝ), (0 < x1) → (0 < x2) → (0 < x3) → 
  ∃ a b c, a = f x1 k ∧ b = f x2 k ∧ c = f x3 k ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) ↔ (-1 ≤ k ∧ k ≤ 2) :=
  by sorry -- Proof goes here

end part1_part2_l634_634475


namespace weekly_allowance_value_l634_634488

-- Define John's weekly allowance
def weekly_allowance := ℝ

-- Define the conditions
def arcade_spent (A : weekly_allowance) := (3/5 : ℝ) * A
def remaining_after_arcade (A : weekly_allowance) := (2/5 : ℝ) * A
def toy_store_spent (A : weekly_allowance) := (1/3 : ℝ) * remaining_after_arcade A
def remaining_after_toy_store (A : weekly_allowance) := remaining_after_arcade A - toy_store_spent A
def candy_store_spent := 0.64

-- The proof statement
theorem weekly_allowance_value (A : weekly_allowance) :
  remaining_after_toy_store A = candy_store_spent →
  A = 2.40 :=
by
  intro h
  -- Steps and calculations would go here in a full proof
  sorry

end weekly_allowance_value_l634_634488


namespace floor_ceil_sum_l634_634409

theorem floor_ceil_sum : (⌊-3.67⌋ : Int) + (⌈30.3⌉ : Int) = 27 :=
by
  have h1 : (⌊-3.67⌋ : Int) = -4 := by sorry
  have h2 : (⌈30.3⌉ : Int) = 31 := by sorry
  exact eq.trans (by rw [h1, h2]) (by norm_num)

end floor_ceil_sum_l634_634409


namespace johns_initial_money_l634_634564

/-- John's initial money given that he gives 3/8 to his mother and 3/10 to his father,
and he has $65 left after giving away the money. Prove that he initially had $200. -/
theorem johns_initial_money 
  (M : ℕ)
  (h_left : (M : ℚ) - (3 / 8) * M - (3 / 10) * M = 65) :
  M = 200 :=
sorry

end johns_initial_money_l634_634564


namespace minimum_value_minimum_value_attained_l634_634138

noncomputable def min_value_3_pow (a b : ℝ) : ℝ := 3^a + 3^b

theorem minimum_value (a b : ℝ) (h : a + b = 2) : min_value_3_pow a b ≥ 6 :=
by
  have h₁ : 3^a + 3^b ≥ 2 * Real.sqrt (3^a * 3^b), from Real.geom_mean_le_am (3^a) (3^b),
  have h₂ : 3^a * 3^b = 3^(a + b), by rw [← Real.pow_add 3 a b],
  rw [h₂, h, Real.sqrt_mul_self (Real.pow_pos (by norm_num) 2)] at h₁,
  exact h₁

theorem minimum_value_attained (a b : ℝ) (h : a = 1) (h' : b = 1) : min_value_3_pow a b = 6 :=
by
  rw [h, h', min_value_3_pow],
  norm_num

end minimum_value_minimum_value_attained_l634_634138


namespace problem1_problem2_l634_634083

noncomputable def f (x a b : ℝ) : ℝ := x^2 - (a+1)*x + b

theorem problem1 (h : ∀ x : ℝ, f x (-4) (-10) < 0 ↔ -5 < x ∧ x < 2) : f x (-4) (-10) < 0 :=
sorry

theorem problem2 (a : ℝ) : 
  (a > 1 → ∀ x : ℝ, f x a a > 0 ↔ x < 1 ∨ x > a) ∧
  (a = 1 → ∀ x : ℝ, f x a a > 0 ↔ x ≠ 1) ∧
  (a < 1 → ∀ x : ℝ, f x a a > 0 ↔ x < a ∨ x > 1) :=
sorry

end problem1_problem2_l634_634083


namespace shiela_animal_drawings_l634_634241

theorem shiela_animal_drawings (neighbors animals : ℕ) (h_neighbors : neighbors = 6) (h_animals : animals = 54) :
  animals / neighbors = 9 :=
by
  rw [h_animals, h_neighbors]
  norm_num

end shiela_animal_drawings_l634_634241


namespace book_pages_l634_634332

theorem book_pages (P : ℕ) 
  (h1 : P / 2 + 11 + (P - (P / 2 + 11)) / 2 = 19)
  (h2 : P - (P / 2 + 11) = 2 * 19) : 
  P = 98 :=
by
  sorry

end book_pages_l634_634332


namespace rent_and_utilities_percentage_l634_634219

/-- Mrs. Snyder's previous monthly income was $1000,
    she spent 40% of her monthly income on rent and utilities,
    and her salary was increased by $600. 
    Prove that the percentage of her new monthly income
    spent on rent and utilities is 25%. --/
theorem rent_and_utilities_percentage
  (previous_income : ℝ) (rent_utilities_percentage_previous : ℝ) (salary_increase : ℝ)
  (h1 : previous_income = 1000) (h2 : rent_utilities_percentage_previous = 0.4) (h3 : salary_increase = 600) :
  let new_income := previous_income + salary_increase,
      amount_spent := rent_utilities_percentage_previous * previous_income,
      new_percentage := (amount_spent / new_income) * 100 in
  new_percentage = 25 :=
by
  sorry

end rent_and_utilities_percentage_l634_634219


namespace triangle_area_PQL_l634_634158

structure Rectangle :=
  (PQ QR: ℝ) -- Sides of the rectangle
  (RS : ℝ)
  (RJ : ℝ)
  (SK : ℝ)

def triangle_area (b h : ℝ) : ℝ :=
  (1/2) * b * h

def similar_triangles_area_ratio (base1 base2 altitude1 : ℝ) : ℝ :=
  (base2 / base1) * altitude1

theorem triangle_area_PQL :
  ∀ (rect : Rectangle),
  rect.PQ = 8 ∧ rect.QR = 4 ∧ rect.RS = 8 ∧ rect.RJ = 2 ∧ rect.SK = 3 →
  triangle_area rect.PQ (similar_triangles_area_ratio
    ((rect.RS - (rect.RJ + rect.SK)) / rect.PQ)
    rect.QR) = 16 :=
by
  intros
  sorry

end triangle_area_PQL_l634_634158


namespace olivia_change_received_l634_634611

theorem olivia_change_received 
    (cost_per_basketball_card : ℕ)
    (basketball_card_count : ℕ)
    (cost_per_baseball_card : ℕ)
    (baseball_card_count : ℕ)
    (bill_amount : ℕ) :
    basketball_card_count = 2 → 
    cost_per_basketball_card = 3 → 
    baseball_card_count = 5 →
    cost_per_baseball_card = 4 →
    bill_amount = 50 →
    bill_amount - (basketball_card_count * cost_per_basketball_card + baseball_card_count * cost_per_baseball_card) = 24 := 
by {
    intros h1 h2 h3 h4 h5,
    rw [h1, h2, h3, h4, h5],
    norm_num,
}

-- Adding a placeholder proof:
-- by sorry

end olivia_change_received_l634_634611


namespace correct_sqrt_division_l634_634316

theorem correct_sqrt_division
  (hA : sqrt 24 / sqrt 6 = 4)
  (hB : sqrt 54 / sqrt 9 = sqrt 6)
  (hC : sqrt 30 / sqrt 6 = 5)
  (hD : sqrt (4 / 7) / sqrt (1 / 49) = 7 * sqrt 2) :
  sqrt 54 / sqrt 9 = sqrt 6 :=
by
  exact hB
  -- Here we are directly using the correct condition true statement from hB
  -- Evaluations of other options (hA, hC, hD) are not needed because the correct one is known.
  sorry -- All other evaluations are skipped, Lean purely requires the final correct theorem statement.

end correct_sqrt_division_l634_634316


namespace parabola_circle_tangency_l634_634240

theorem parabola_circle_tangency :
  ∃ (r : ℝ), (∀ (k : ℕ) (hk : k < 7),
    let theta := (2 * Real.pi / 7) * k
    in ∃ (a : ℝ), (y = x^2 + r) is tangent to (y = a * x) ∧ r = 25/64) :=
sorry

end parabola_circle_tangency_l634_634240


namespace seq_a2010_l634_634552

-- Definitions and conditions
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ 
  a 2 = 3 ∧ 
  ∀ n ≥ 2, a (n + 1) = (a n * a (n - 1)) % 10

-- Proof statement
theorem seq_a2010 {a : ℕ → ℕ} (h : seq a) : a 2010 = 4 := 
  sorry

end seq_a2010_l634_634552


namespace first_train_length_l634_634358

open Real

noncomputable def length_of_first_train (speed_first_train speed_second_train : ℝ) (time : ℝ) (length_second_train : ℝ) : ℝ :=
  let relative_speed := (speed_first_train - speed_second_train) * (1000 / 3600)
  in (relative_speed * time) - length_second_train

theorem first_train_length:
  length_of_first_train 72 36 91.9926405887529 540 = 379.926405887529 :=
by
  sorry

end first_train_length_l634_634358


namespace max_a_for_increasing_f_on_interval_l634_634133

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x

theorem max_a_for_increasing_f_on_interval :
  ∀ a, (∀ x, 1 < x → 0 ≤ f' x a) → a ≤ Real.exp 1 := sorry

end max_a_for_increasing_f_on_interval_l634_634133


namespace range_of_a_l634_634100

-- Define the universal set, A, B, and the complement of A
def U := set ℝ 
def A := {x : ℝ | -1 ≤ x ∧ x < 3 }
def C_U_A := {x : ℝ | x < -1 ∨ x ≥ 3}

-- Define B based on the parameter a
def B (a : ℝ) : set ℝ :=
  if a > 0 then {x : ℝ | -2 < x ∧ x ≤ a-2}
  else if a = 0 then ∅
  else {x : ℝ | a-2 ≤ x ∧ x < 2}

-- Define the conditions for the range of a
theorem range_of_a (a : ℝ) : ((C_U_A ∪ B a) = C_U_A) ↔ (0 ≤ a ∧ a < 1) :=
sorry

end range_of_a_l634_634100


namespace duration_of_each_movie_l634_634345

-- define the conditions
def num_screens : ℕ := 6
def hours_open : ℕ := 8
def num_movies : ℕ := 24

-- define the total screening time
def total_screening_time : ℕ := num_screens * hours_open

-- define the expected duration of each movie
def movie_duration : ℕ := total_screening_time / num_movies

-- state the theorem
theorem duration_of_each_movie : movie_duration = 2 := by sorry

end duration_of_each_movie_l634_634345


namespace complex_subtraction_l634_634067

def z1 : ℂ := 3 + (1 : ℂ)
def z2 : ℂ := 2 - (1 : ℂ)

theorem complex_subtraction : z1 - z2 = 1 + 2 * (1 : ℂ) :=
by
  sorry

end complex_subtraction_l634_634067


namespace range_of_m_l634_634825

theorem range_of_m (m : ℝ) : 
  (let coefficient := (choose 10 4 : ℝ) + (choose 10 2 : ℝ) * m^2 + (choose 10 1 : ℝ) * (choose 9 2 : ℝ) * m in
  coefficient > -330) ↔ (m < -6 ∨ m > -2) := 
by { 
  sorry 
}

end range_of_m_l634_634825


namespace incorrect_negation_l634_634367

theorem incorrect_negation :
  let PA := ∀ n : ℤ, 3 ∣ n → (n % 2 = 1)
  let not_PA := ∃ n : ℤ, 3 ∣ n ∧ ¬(n % 2 = 1)
  let PB := ∃ (A B C D : ℝ × ℝ), ¬ (∃ (O : ℝ × ℝ), dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = dist O D)
  let not_PB := ∀ (A B C D : ℝ × ℝ), ∃ (O : ℝ × ℝ), dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = dist O D
  let PC := ∃ (T : Triangle), T.isEquilateral
  let not_PC := ¬ (∀ (T : Triangle), T.isEquilateral)
  let PD := ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0
  let not_PD := ∀ x : ℝ, x^2 + 2 * x + 2 > 0
  (¬ (∀ n : ℤ, 3 ∣ n → (n % 2 = 1)) ↔ ∃ n : ℤ, 3 ∣ n ∧ ¬(n % 2 = 1)) →
  (¬ (∃ (A B C D : ℝ × ℝ), ¬ (∃ (O : ℝ × ℝ), dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = dist O D)) ↔ 
    ∀ (A B C D : ℝ × ℝ), ∃ (O : ℝ × ℝ), dist O A = dist O B ∧ dist O B = dist O C ∧ dist O C = dist O D) →
  (¬ (∃ (T : Triangle), T.isEquilateral) ↔ ¬ (∀ (T : Triangle), T.isEquilateral)) →
  (¬ (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 2 > 0)) →
  false := sorry

end incorrect_negation_l634_634367


namespace house_trailer_payment_difference_l634_634177

-- Define the costs and periods
def cost_house : ℕ := 480000
def cost_trailer : ℕ := 120000
def loan_period_years : ℕ := 20
def months_per_year : ℕ := 12

-- Calculate total months
def total_months : ℕ := loan_period_years * months_per_year

-- Calculate monthly payments
def monthly_payment_house : ℕ := cost_house / total_months
def monthly_payment_trailer : ℕ := cost_trailer / total_months

-- Theorem stating the difference in monthly payments
theorem house_trailer_payment_difference :
  monthly_payment_house - monthly_payment_trailer = 1500 := by sorry

end house_trailer_payment_difference_l634_634177


namespace area_of_right_triangle_with_given_sides_l634_634888

theorem area_of_right_triangle_with_given_sides :
  let f : (ℝ → ℝ) := fun x => x^2 - 7 * x + 12
  let a := 3
  let b := 4
  let c := sqrt 7
  let hypotenuse := max a b
  let leg := min a b
in (hypotenuse = 4 ∧ leg = 3 ∧ (f(3) = 0) ∧ (f(4) = 0) → 
   (∃ (area : ℝ), (area = 6 ∨ area = (3 * sqrt 7) / 2))) :=
by
  intros
  sorry

end area_of_right_triangle_with_given_sides_l634_634888


namespace white_chocolate_bars_sold_l634_634273

theorem white_chocolate_bars_sold (W D : ℕ) (h1 : D = 15) (h2 : W / D = 4 / 3) : W = 20 :=
by
  -- This is where the proof would go.
  sorry

end white_chocolate_bars_sold_l634_634273


namespace time_for_Harish_to_paint_alone_l634_634487

theorem time_for_Harish_to_paint_alone (H : ℝ) (h1 : H > 0) (h2 :  (1 / 6 + 1 / H) = 1 / 2 ) : H = 3 :=
sorry

end time_for_Harish_to_paint_alone_l634_634487


namespace path_count_l634_634704

open_locale big_operators

def f (n s : ℕ) : ℕ :=
by simp

noncomputable
def binom (n k : ℕ) : ℕ :=
if k > n then 0 else (Finset.range k).prod (λ i, (n - i) / (i + 1))

theorem path_count (n s : ℕ) (hn : n ≥ 1) (hs : n ≥ s ∧ s ≥ 1) :
  f n s = (1 / s) * binom (n - 1) (s - 1) * binom n (s - 1) :=
sorry

end path_count_l634_634704


namespace sufficient_but_not_necessary_condition_for_increasing_log_l634_634278

theorem sufficient_but_not_necessary_condition_for_increasing_log (a : ℝ) :
  a > 2 → (∀ x > 0, monotone (λ x, log a x)) ∧ ¬ ∀ x > 0, (a > 2 → monotone (λ x, log a x)) :=
sorry

end sufficient_but_not_necessary_condition_for_increasing_log_l634_634278


namespace multiples_33_between_1_and_300_l634_634281

theorem multiples_33_between_1_and_300 : ∃ (x : ℕ), (∀ n : ℕ, n ≤ 300 → n % x = 0 → n / x ≤ 33) ∧ x = 9 :=
by
  sorry

end multiples_33_between_1_and_300_l634_634281


namespace gabby_additional_money_needed_l634_634431

theorem gabby_additional_money_needed
  (cost_makeup : ℕ := 65)
  (cost_skincare : ℕ := 45)
  (cost_hair_tool : ℕ := 55)
  (initial_savings : ℕ := 35)
  (money_from_mom : ℕ := 20)
  (money_from_dad : ℕ := 30)
  (money_from_chores : ℕ := 25) :
  (cost_makeup + cost_skincare + cost_hair_tool) - (initial_savings + money_from_mom + money_from_dad + money_from_chores) = 55 := 
by
  sorry

end gabby_additional_money_needed_l634_634431


namespace cost_of_three_books_l634_634635

-- Definitions based on the conditions
def cost_of_six_books : ℝ := 37.44
def books_to_cost (n : ℕ) := n / 6 * cost_of_six_books

-- Theorem statement proving the cost of 3 books
theorem cost_of_three_books : books_to_cost 3 = 18.72 := by
  sorry

end cost_of_three_books_l634_634635


namespace relationship_M_N_l634_634851

def is_subset (M N : Set ℝ) : Prop :=
  ∀ a : ℝ, a ∈ M → a ∈ N

def is_not_subset (N M : Set ℝ) : Prop :=
  ∃ a : ℝ, a ∈ N ∧ a ∉ M

theorem relationship_M_N : 
  let M := {x : ℝ | -2 < x ∧ x < 2},
      N := {x : ℝ | x < 2} in
  is_subset M N ∧ is_not_subset N M :=
by
  sorry

end relationship_M_N_l634_634851


namespace area_of_R_l634_634960

/-- Let ABC be an equilateral triangle of side length 2. Let ω be its circumcircle, and let
ω_A, ω_B, ω_C be circles congruent to ω centered at each of its vertices. Let R be the set 
of all points in the plane contained in exactly two of these four circles. Prove that the 
area of R is 2√3. -/
theorem area_of_R (A B C : Point2D)
                   (hABC_eq_tri: equilateral_triangle A B C 2)
                   (ω : Circle)
                   (hω_circumcircle: circumcircle ω A B C)
                   (ω_A ω_B ω_C : Circle)
                   (hωA_congr: congruent_circles ω ω_A ∧ center ω_A = A)
                   (hωB_congr: congruent_circles ω ω_B ∧ center ω_B = B)
                   (hωC_congr: congruent_circles ω ω_C ∧ center ω_C = C)
                   (R : Set Point2D)
                   (hR: ∀ p, p ∈ R ↔ (p ∈ ω ∧ p ∉ ω_A ∩ ω_B ∩ ω_C)) :
  area R = 2 * Real.sqrt 3 :=
sorry

end area_of_R_l634_634960


namespace find_d_squared_l634_634717

noncomputable def f (c d : ℝ) (z : ℂ) : ℂ := (c + d * complex.I) * z

theorem find_d_squared (c d : ℝ) (hz : ∀ z : ℂ, complex.abs (f c d z - z) = complex.abs (f c d z - (3 + 4 * complex.I))) (hcd : complex.abs (c + d * complex.I) = 7) :
  d^2 = 33 :=
sorry

end find_d_squared_l634_634717


namespace integral_circle_minus_x_l634_634009

open Set Filter

theorem integral_circle_minus_x : (∫ x in 0..1, (sqrt (1 - (x - 1) ^ 2) - x)) = (Real.pi / 4 - 1 / 2) := by
  sorry

end integral_circle_minus_x_l634_634009


namespace right_triangle_area_l634_634150

theorem right_triangle_area (hypotenuse : ℝ) (angle_ratio : ℝ) (a1 a2 : ℝ)
  (h1 : hypotenuse = 10)
  (h2 : angle_ratio = 5 / 4)
  (h3 : a1 = 50)
  (h4 : a2 = 40) :
  let A := hypotenuse * Real.sin (a2 * Real.pi / 180)
  let B := hypotenuse * Real.sin (a1 * Real.pi / 180)
  let area := 0.5 * A * B
  area = 24.63156 := by sorry

end right_triangle_area_l634_634150


namespace triangle_first_side_length_l634_634029

theorem triangle_first_side_length (x : ℕ) (h1 : x + 20 + 30 = 55) : x = 5 :=
by
  sorry

end triangle_first_side_length_l634_634029


namespace no_integer_roots_l634_634132

theorem no_integer_roots (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) : ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
  sorry

end no_integer_roots_l634_634132


namespace contradiction_proof_example_l634_634295

theorem contradiction_proof_example (a b : ℝ) (h: a ≤ b → False) : a > b :=
by sorry

end contradiction_proof_example_l634_634295


namespace geometric_inequality_l634_634152

open Real
open EuclideanGeometry

noncomputable theory

universe u

variables {Triangle : Type u} [metric_space Triangle] [normed_group Triangle] [normed_space ℝ Triangle]
variables {A B C O A' B' C' : Triangle}
variables {R : ℝ}

def is_acute_triangle (A B C : Triangle) : Prop := 
  acute_triangle A B C

def circumcenter (A B C : Triangle) : Triangle := O
def circumradius (A B C : Triangle) : ℝ := R

def extension_intersect (O : Triangle) (A B C : Triangle) (A' B' C' : Triangle) : Prop :=
  intersect (line_through A O) (circumcircle O B C) A' ∧
  intersect (line_through B O) (circumcircle O A C) B' ∧
  intersect (line_through C O) (circumcircle O A B) C'

theorem geometric_inequality
  (h_acute : is_acute_triangle A B C)
  (h_circumcenter : circumcenter A B C = O)
  (h_circumradius : circumradius A B C = R)
  (h_extension_intersect : extension_intersect O A B C A' B' C') :
  dist O A' * dist O B' * dist O C' ≥ 8 * R^3 :=
sorry

end geometric_inequality_l634_634152


namespace number_of_m_constants_l634_634774

theorem number_of_m_constants : 
  ∃ m1 m2, (∀ a b c d : ℤ, 
    let M1 := (a, b + c) in
    let M2 := (a - d, b) in
    let slope1 := c / (2 * d) in
    let slope2 := 2 * c / d in
    (slope1 = 2 ∧ M1.2 = 2 * M1.1 + 1) ∧ 
    (slope2 = m ∧ M2.2 = m * M2.1 + 2)) → 
  m1 ≠ m2 ∧ number_of_different_ms = 2 :=
begin
  sorry
end

end number_of_m_constants_l634_634774


namespace complex_quadrant_l634_634816

theorem complex_quadrant (i : ℂ) (h_i : i*i = -1) :
  let z : ℂ := (3 + i) / (1 + i) in
  (z.re > 0 ∧ z.im < 0) := by
  let one_plus_i : ℂ := 1 + i
  have h_one_plus_i_ne_zero : one_plus_i ≠ 0 := by sorry
  let z := (3 + i) / one_plus_i
  have z_value : z = 2 - i := by sorry
  show z.re > 0 ∧ z.im < 0 from by sorry

end complex_quadrant_l634_634816


namespace liam_minimum_score_l634_634968

theorem liam_minimum_score :
  let scores := [95, 85, 75, 65, 80]
  let current_average := (95 + 85 + 75 + 65 + 80) / 5
  let desired_average := current_average + 4
  let required_total := desired_average * 6
  let current_total := 95 + 85 + 75 + 65 + 80
  let needed_score := required_total - current_total
  needed_score = 104 :=
by
  simp [needed_score, required_total, current_total, desired_average, current_average, scores]
  sorry

end liam_minimum_score_l634_634968


namespace mary_age_l634_634532

theorem mary_age (M : ℕ) (h1 : ∀ t : ℕ, t = 4 → 24 = 2 * (M + t)) (h2 : 20 = 20) : M = 8 :=
by {
  have t_eq_4 := h1 4 rfl,
  norm_num at t_eq_4,
  linarith,
}

end mary_age_l634_634532


namespace area_of_sector_oab_l634_634079

noncomputable def radius {π : Real} (r : Real) :=
  let θ := (5 / 7) * π
  let perimeter := 5 * π + 14
  let equation := θ * r + 2 * r = perimeter
  r

noncomputable def area_sector_oab {π : Real} (r : Real) :=
  let θ := (5 / 7) * π
  (1 / 2) * θ * (r * r)

theorem area_of_sector_oab {π : Real} (r : Real) (h : θ * r + 2 * r = 5 * π + 14) :
  area_sector_oab 7 = 35/2 * π :=
by
  sorry

end area_of_sector_oab_l634_634079


namespace find_m_l634_634581

variable {y m : ℝ} -- define variables y and m in the reals

-- define the logarithmic conditions
axiom log8_5_eq_y : log 8 5 = y
axiom log2_125_eq_my : log 2 125 = m * y

-- state the theorem to prove m equals 9
theorem find_m (log8_5_eq_y : log 8 5 = y) (log2_125_eq_my : log 2 125 = m * y) : m = 9 := by
  sorry

end find_m_l634_634581


namespace value_of_fraction_l634_634125

-- Lean 4 statement
theorem value_of_fraction (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : (x + y) / 3 = 5 / 3 :=
by
  sorry

end value_of_fraction_l634_634125


namespace quadrilateral_segments_l634_634735

theorem quadrilateral_segments {a b c d : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a + b + c + d = 2) (h6 : 1/4 < a) (h7 : a < 1) (h8 : 1/4 < b) (h9 : b < 1)
  (h10 : 1/4 < c) (h11 : c < 1) (h12 : 1/4 < d) (h13 : d < 1) : 
  (a + b > d) ∧ (a + c > d) ∧ (a + d > c) ∧ (b + c > d) ∧ 
  (b + d > c) ∧ (c + d > a) ∧ (a + b + c > d) ∧ (a + b + d > c) ∧
  (a + c + d > b) ∧ (b + c + d > a) :=
sorry

end quadrilateral_segments_l634_634735


namespace problem_statement_l634_634494

variables {V : Type*} [inner_product_space ℝ V] (a b : V) (f : ℝ → ℝ)

-- Given conditions
def orthogonal (a b : V) : Prop := ⟪a, b⟫ = 0
def non_zero (a : V) : Prop := a ≠ 0
def different_norms (a b : V) : Prop := ∥a∥ ≠ ∥b∥

-- The function in question
def f (x : ℝ) : ℝ := ⟪(x • a + b), (x • b - a)⟫

-- The proof problem
theorem problem_statement (h_orth : orthogonal a b) 
                          (h_neq_a : non_zero a)
                          (h_neq_b : non_zero b)
                          (h_diff_norms : different_norms a b) :
  (∀ x, f x = x * (∥b∥^2 - ∥a∥^2)) ∧ (∀ x, f (-x) = -f x) :=
sorry

end problem_statement_l634_634494


namespace jeff_expected_score_l634_634181

-- Define the Uniform distribution on the interval [0, 120]
noncomputable def JeffWakeTime : ℝ → MeasureTheory.Measure ℝ := 
  MeasureTheory.Measure.diracUniform 0 120

-- Define the conditions on the score based on arrival time
def score (t : ℝ) : ℝ :=
  if h : 45 ≤ t ∧ t ≤ 60 then 50
  else if h : 60 < t ∧ t ≤ 75 then 50 - (t - 60)
  else if h : 75 < t ∧ t ≤ 120 then 0
  else 0

-- Define the expected score function based on the random variable T
noncomputable def expectedScore : ℝ :=
  ∫ (t : ℝ) in (Set.Icc 0 120), score (t + 15) ∂(JeffWakeTime t)

-- Prove that the expected score of Jeff is 13.75
theorem jeff_expected_score : expectedScore = 13.75 := 
  by
  sorry

end jeff_expected_score_l634_634181


namespace range_of_a_l634_634037

theorem range_of_a (x y : ℝ) : 
  let a := (Real.cos x) ^ 2 + (Real.cos y) ^ 2 - Real.cos (x * y) in
  a > -3 ∧ a < 3 := 
by 
  sorry

end range_of_a_l634_634037


namespace triangle_area_find_angle_C_l634_634904

variables {A B C : Angle}
variables {a b c : ℝ}

theorem triangle_area (hcosB : cos B = 3 / 5) (hac : a * c = 35) : 
  1 / 2 * a * c * sqrt(1 - (3 / 5)^2) = 14 := 
by 
  sorry

theorem find_angle_C (hcosB : cos B = 3 / 5) (hac : a * c = 35) (ha : a = 7) : 
  ∠C = π / 4 := 
by 
  sorry 

end triangle_area_find_angle_C_l634_634904


namespace veronica_pits_cherries_in_2_hours_l634_634294

theorem veronica_pits_cherries_in_2_hours :
  ∀ (pounds_cherries : ℕ) (cherries_per_pound : ℕ)
    (time_first_pound : ℕ) (cherries_first_pound : ℕ)
    (time_second_pound : ℕ) (cherries_second_pound : ℕ)
    (time_third_pound : ℕ) (cherries_third_pound : ℕ)
    (minutes_per_hour : ℕ),
  pounds_cherries = 3 →
  cherries_per_pound = 80 →
  time_first_pound = 10 →
  cherries_first_pound = 20 →
  time_second_pound = 8 →
  cherries_second_pound = 20 →
  time_third_pound = 12 →
  cherries_third_pound = 20 →
  minutes_per_hour = 60 →
  ((time_first_pound / cherries_first_pound * cherries_per_pound) + 
   (time_second_pound / cherries_second_pound * cherries_per_pound) + 
   (time_third_pound / cherries_third_pound * cherries_per_pound)) / minutes_per_hour = 2 :=
by
  intros pounds_cherries cherries_per_pound
         time_first_pound cherries_first_pound
         time_second_pound cherries_second_pound
         time_third_pound cherries_third_pound
         minutes_per_hour
         pounds_eq cherries_eq
         time1_eq cherries1_eq
         time2_eq cherries2_eq
         time3_eq cherries3_eq
         mins_eq

  -- You would insert the proof here
  sorry

end veronica_pits_cherries_in_2_hours_l634_634294


namespace homework_total_time_l634_634230

theorem homework_total_time :
  ∀ (j g p : ℕ),
  j = 18 →
  g = j - 6 →
  p = 2 * g - 4 →
  j + g + p = 50 :=
by
  intros j g p h1 h2 h3
  sorry

end homework_total_time_l634_634230


namespace evaluate_product_l634_634407

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_product : 
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 2657205 :=
by 
  sorry

end evaluate_product_l634_634407


namespace find_set_A_find_range_a_l634_634833

-- Define the universal set and the complement condition for A
def universal_set : Set ℝ := {x | true}
def complement_A : Set ℝ := {x | 2 * x^2 - 3 * x - 2 > 0}

-- Define the set A
def set_A : Set ℝ := {x | -1/2 ≤ x ∧ x ≤ 2}

-- Define the set C
def set_C (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a ≤ 0}

-- Define the proof problem for part (1)
theorem find_set_A : { x | -1 / 2 ≤ x ∧ x ≤ 2 } = { x | ¬ (2 * x^2 - 3 * x - 2 > 0) } :=
by
  sorry

-- Define the proof problem for part (2)
theorem find_range_a (a : ℝ) (C_ne_empty : (set_C a).Nonempty) (sufficient_not_necessary : ∀ x, x ∈ set_C a → x ∈ set_A → x ∈ set_A) :
  a ∈ Set.Icc (-1/8 : ℝ) 0 ∪ Set.Icc 1 (4/3 : ℝ) :=
by
  sorry

end find_set_A_find_range_a_l634_634833


namespace doubling_time_l634_634649

-- Define the initial population, final population, and time taken
def N₀ : ℕ := 1000
def N : ℕ := 500000
def t : ℝ := 35.86

-- Define the expected doubling time
def expected_T : ℝ := 4

-- State the theorem to prove the doubling time T
theorem doubling_time (T : ℝ) (h : N = N₀ * 2^(t / T)) : T ≈ expected_T := by
  sorry

end doubling_time_l634_634649


namespace find_x_plus_y_l634_634124

theorem find_x_plus_y (x y : ℝ) (hx : abs x - x + y = 6) (hy : x + abs y + y = 16) : x + y = 10 :=
sorry

end find_x_plus_y_l634_634124


namespace chord_length_greater_than_third_radius_l634_634056

theorem chord_length_greater_than_third_radius
  (O A B : Point)
  (R : ℝ)
  (circle : Circle O R)
  (angle_AOB : ∠ A O B = 20) :
  chord_length O A B > R / 3 :=
sorry

end chord_length_greater_than_third_radius_l634_634056


namespace combined_cost_price_l634_634292

def cost_price_A : ℕ := (120 + 60) / 2
def cost_price_B : ℕ := (200 + 100) / 2
def cost_price_C : ℕ := (300 + 180) / 2

def total_cost_price : ℕ := cost_price_A + cost_price_B + cost_price_C

theorem combined_cost_price :
  total_cost_price = 480 := by
  sorry

end combined_cost_price_l634_634292


namespace inverse_proportion_function_range_m_l634_634096

theorem inverse_proportion_function_range_m
  (x1 x2 y1 y2 m : ℝ)
  (h_func_A : y1 = (5 * m - 2) / x1)
  (h_func_B : y2 = (5 * m - 2) / x2)
  (h_x : x1 < x2)
  (h_x_neg : x2 < 0)
  (h_y : y1 < y2) :
  m < 2 / 5 :=
sorry

end inverse_proportion_function_range_m_l634_634096


namespace part1_part2_l634_634439

noncomputable def f : ℝ → ℝ := λ x, if x > 1 then x + 5 else 2 * x^2 + 1

theorem part1 : f (f 1) = 8 := by sorry

theorem part2 (x : ℝ) : f x = 5 → x = -Real.sqrt 2 := by
  intro h
  have h1 : ¬ (x > 1) := by
    intro h2
    have h3 : f x = x + 5 := by simp [f, h2]
    linarith
  simp [f, h1] at h
  rw [← sub_eq_zero, sub_left_inj] at h
  exact eq_neg_self_iff.mp h

end part1_part2_l634_634439


namespace figure_100_squares_l634_634780

/-- Figures 0, 1, 2, and 3 consist of 1, 7, 19, and 37 nonoverlapping unit squares, respectively.
    If the pattern continues, the number of nonoverlapping unit squares in figure 100 is 30301. -/
theorem figure_100_squares :
  ∀ (n : ℕ), (n = 0 → f n = 1) ∧
             (n = 1 → f n = 7) ∧
             (n = 2 → f n = 19) ∧
             (n = 3 → f n = 37) →
             f 100 = 30301 := sorry

end figure_100_squares_l634_634780


namespace determine_g_l634_634948

-- Definition of the custom operation
def bowtie (a b : ℝ) : ℝ := a + 3 * (Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + ...))))

-- The proof statement
theorem determine_g (g : ℝ) : bowtie 5 g = 14 → g = 6 :=
by
  sorry

end determine_g_l634_634948


namespace perimeter_of_triangle_l634_634700

-- Define the average length of the sides of the triangle
def average_length (a b c : ℕ) : ℕ := (a + b + c) / 3

-- Define the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- The theorem we want to prove
theorem perimeter_of_triangle {a b c : ℕ} (h_avg : average_length a b c = 12) : perimeter a b c = 36 :=
sorry

end perimeter_of_triangle_l634_634700


namespace proof_problem_l634_634930

theorem proof_problem :
  let M := (3, Real.pi)
  let N := (Real.sqrt 3, Real.pi / 2)
  ∃ P : ℝ × ℝ, P = (-3 / 4, 3 * Real.sqrt 3 / 4) ∧
  ∃ θ : ℝ, 
    ∀ (C : ℝ × ℝ → Prop),
      C = (λ (x y : ℝ),
            (x + 3 / 2)^2 + (y - Real.sqrt 3 / 2)^2 = 3) ∧
        (x = -3 / 2 + Real.sqrt 3 * Real.cos θ) ∧
        (y = Real.sqrt 3 / 2 + Real.sqrt 3 * Real.sin θ) ∧
        dist (-(3 * Real.sqrt 3 / 2) + (Real.sqrt 3 / 2)) 2 = Real.sqrt 3 / 2 &&
        (2 * Real.sqrt (3 - (Real.sqrt 3 / 2)^2) = 3)
  sorry

end proof_problem_l634_634930


namespace arithmetic_mean_of_two_digit_multiples_of_7_is_56_l634_634675

/-- Given the arithmetic series of all positive two-digit multiples of 7, 
prove that the arithmetic mean of this series is 56. -/

theorem arithmetic_mean_of_two_digit_multiples_of_7_is_56 :
  let a1 := 14 in
  let an := 98 in
  let d := 7 in
  let n := (an - a1) / d + 1 in
  let Sn := n * (a1 + an) / 2 in
  Sn / n = 56 :=
by
  { 
    -- Provide the required proof here
    sorry
  }

end arithmetic_mean_of_two_digit_multiples_of_7_is_56_l634_634675


namespace part1_part2_l634_634085

noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

theorem part1 (x : ℝ) (hx : 0 ≤ x) :
  f(x) ≤ x + 2 := sorry

theorem part2 (n : ℕ) (hn : 0 < n) :
  f (1 / n - 1 / (n + 1)) + 1 / Real.exp 2 * f (2 - 1 / n) ≤ 2 + 1 / n := sorry

end part1_part2_l634_634085


namespace right_triangle_area_l634_634890

def roots (a b : ℝ) : Prop :=
  a * b = 12 ∧ a + b = 7

def area (A : ℝ) : Prop :=
  A = 6 ∨ A = 3 * Real.sqrt 7 / 2

theorem right_triangle_area (a b A : ℝ) (h : roots a b) : area A := 
by 
  -- The proof steps would go here
  sorry

end right_triangle_area_l634_634890


namespace diagonals_not_parallel_to_sides_in_regular_32_gon_l634_634866

theorem diagonals_not_parallel_to_sides_in_regular_32_gon :
  let n := 32 in
  let total_diagonals := n * (n - 3) / 2 in
  let parallel_diagonals := 16 * (n - 4) / 2 in
  let non_parallel_diagonals := total_diagonals - parallel_diagonals in
  non_parallel_diagonals = 240 :=
by
  sorry

end diagonals_not_parallel_to_sides_in_regular_32_gon_l634_634866


namespace day_of_week_after_n_days_l634_634375

theorem day_of_week_after_n_days (birthday : ℕ) (n : ℕ) (day_of_week : ℕ) :
  birthday = 4 → (n % 7) = 2 → day_of_week = 6 :=
by sorry

end day_of_week_after_n_days_l634_634375


namespace evaluate_product_l634_634408

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_product : 
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 2657205 :=
by 
  sorry

end evaluate_product_l634_634408


namespace complex_abs_eq_l634_634599

noncomputable def z (w : ℂ) : ℂ := 5 / (2 + 1 * complex.I)
theorem complex_abs_eq {z : ℂ} (h : (2 + complex.I) * z = 5) : complex.abs z = real.sqrt 5 :=
sorry

end complex_abs_eq_l634_634599


namespace next_in_step_distance_l634_634174

theorem next_in_step_distance
  (jack_stride jill_stride : ℕ)
  (h1 : jack_stride = 64)
  (h2 : jill_stride = 56) :
  Nat.lcm jack_stride jill_stride = 448 := by
  sorry

end next_in_step_distance_l634_634174


namespace max_principals_and_assistant_principals_l634_634969

theorem max_principals_and_assistant_principals : 
  ∀ (years term_principal term_assistant), (years = 10) ∧ (term_principal = 3) ∧ (term_assistant = 2) 
  → ∃ n, n = 9 :=
by
  sorry

end max_principals_and_assistant_principals_l634_634969


namespace market_value_of_stock_l634_634320

def face_value : ℝ := 100
def dividend_percentage : ℝ := 0.13
def yield : ℝ := 0.08

theorem market_value_of_stock : 
  (dividend_percentage * face_value / yield) * 100 = 162.50 :=
by
  sorry

end market_value_of_stock_l634_634320


namespace evaluate_expression_l634_634410

theorem evaluate_expression (a : ℚ) (h : a = 3/2) : 
  ((5 * a^2 - 13 * a + 4) * (2 * a - 3)) = 0 := by
  sorry

end evaluate_expression_l634_634410


namespace area_of_region_l634_634959

-- Define the complex number z
def z (x y : ℝ) := complex.mk x y

-- Condition 1: Imaginary part of z is non-negative
def condition1 (x y : ℝ) := complex.im (z x y) ≥ 0

-- Condition 2: Real part of (z - i) / (1 - z) is non-negative
def condition2 (x y : ℝ) : Prop :=
  let z_minus_i := complex.sub (z x y) complex.I
  let one_minus_z := complex.sub complex.one (z x y)
  complex.re (complex.div z_minus_i one_minus_z) ≥ 0

-- The region is defined by the intersection of the above conditions
def region (x y : ℝ) : Prop := condition1 x y ∧ condition2 x y

-- Define the expected area
def expected_area : ℝ := π / 2

-- Statement of the problem: Prove that the area of the region is equal to the expected area
theorem area_of_region :
  (let points := {p : ℝ × ℝ | region p.1 p.2} in
   measure_theory.measure_space.volume (set_of region)) = expected_area :=
by
  sorry

end area_of_region_l634_634959


namespace outer_circle_radius_l634_634636

theorem outer_circle_radius 
  (r₁ : ℝ) (r₂ : ℝ)
  (h₁ : r₁ = 1)
  (h₂ : r₂ = (1 + 2 * (Real.sin (Real.pi / 10)) / (1 - Real.sin (Real.pi / 10))))
  : r₂ = (1 + Real.sin (Real.pi / 10)) / (1 - Real.sin (Real.pi / 10)) :=
  by
  rw [h₁, h₂]
  sorry

end outer_circle_radius_l634_634636


namespace solution_set_of_exponential_inequality_l634_634832

variable (f : ℝ → ℝ)

theorem solution_set_of_exponential_inequality
  (h : ∀ x, f(x) < 0 ↔ (x < -1) ∨ (x > 1/2)) :
  ∀ x, f(10^x) > 0 ↔ x < -real.log 2 :=
by
  sorry

end solution_set_of_exponential_inequality_l634_634832


namespace total_cost_is_46_8_l634_634343

def price_pork : ℝ := 6
def price_chicken : ℝ := price_pork - 2
def price_beef : ℝ := price_chicken + 4
def price_lamb : ℝ := price_pork + 3

def quantity_chicken : ℝ := 3.5
def quantity_pork : ℝ := 1.2
def quantity_beef : ℝ := 2.3
def quantity_lamb : ℝ := 0.8

def total_cost : ℝ :=
    (quantity_chicken * price_chicken) +
    (quantity_pork * price_pork) +
    (quantity_beef * price_beef) +
    (quantity_lamb * price_lamb)

theorem total_cost_is_46_8 : total_cost = 46.8 :=
by
  sorry

end total_cost_is_46_8_l634_634343


namespace possible_values_expression_l634_634438

theorem possible_values_expression 
  (a b : ℝ) 
  (h₁ : a^2 = 16) 
  (h₂ : |b| = 3) 
  (h₃ : ab < 0) : 
  (a - b)^2 + a * b^2 = 85 ∨ (a - b)^2 + a * b^2 = 13 := 
by 
  sorry

end possible_values_expression_l634_634438


namespace cartesian_equation_of_C2_distance_range_l634_634539

-- Define curve C1
def curveC1_param (θ : ℝ) : ℝ × ℝ :=
  (4 * Real.cos θ, 3 * Real.sin θ)

-- Define the polar equation ρ sin (θ + π/4) = 5√2/2
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + π / 4) = 5 * Real.sqrt 2 / 2

-- (1) Prove the Cartesian coordinate equation of curve C2
theorem cartesian_equation_of_C2 (x y : ℝ) (ρ θ : ℝ) 
  (h1 : ρ * Real.cos θ = x)
  (h2 : ρ * Real.sin θ = y)
  (h3 : polar_equation ρ θ) :
  x + y = 5 := sorry

-- (2) Prove the range of the distance d from a point M on curve C1 to curve C2
theorem distance_range (θ φ : ℝ) (M : ℝ × ℝ) 
  (h1 : M = curveC1_param θ) :
  0 ≤ (|4 * Real.cos θ + 3 * Real.sin θ - 5| / Real.sqrt 2) ∧
  (|4 * Real.cos θ + 3 * Real.sin θ - 5| / Real.sqrt 2) ≤ 5 * Real.sqrt 2 := sorry

end cartesian_equation_of_C2_distance_range_l634_634539


namespace minimize_perimeter_divide_l634_634446

noncomputable def convex_quadrilateral (A B C D : Type) : Prop := sorry

noncomputable def incenter (O A B : Type) : Type := sorry
noncomputable def excenter (O C D : Type) : Type := sorry

noncomputable def midpoint (I J: Type) : Type := sorry

noncomputable def perpendicular_projection (M BC AD: Type) : (Type × Type) := sorry

theorem minimize_perimeter_divide
  (A B C D O I J M X Y : Type)
  (h1: convex_quadrilateral A B C D)
  (h2: ∃ O, intersect (line BC) (line AD) O)
  (h3: ∃ B, on_segment O A B)
  (h4: ∃ A, on_segment O D A)
  (h5: ∃ I, I = incenter O A B)
  (h6: ∃ J, J = excenter O C D)
  (h7: ∃ M, M = midpoint I J)
  (h8: (X, Y) = perpendicular_projection M (line BC) (line AD)) :
  divides_perimeter_in_half XY A B C D ∧ minimizes_length XY A B C D := 
sorry

end minimize_perimeter_divide_l634_634446


namespace maggie_goldfish_fraction_l634_634207

theorem maggie_goldfish_fraction :
  ∀ (x : ℕ), 3*x / 5 + 20 = x → (x / 100 : ℚ) = 1 / 2 :=
by
  sorry

end maggie_goldfish_fraction_l634_634207


namespace trig_identity_l634_634119

variables {a b θ : ℝ}

theorem trig_identity (h : (sin(θ)^4 / a) + (cos(θ)^4 / b) = 1 / (2 * (a + b))) :
  (sin(θ)^6 / a^2) + (cos(θ)^6 / b^2) = 1 / (a + b)^2 :=
by
  sorry

end trig_identity_l634_634119


namespace ellipse_equation_dot_product_constant_l634_634802

-- Given circles F₁ and F₂, and Circle O
noncomputable def circle_F1 (r : ℝ) : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in (x + 1)^2 + y^2 = r^2 }

noncomputable def circle_F2 (r : ℝ) : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in (x - 1)^2 + y^2 = (4 - r)^2 }

def circle_O : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in x^2 + y^2 = (12 / 7) }

-- Define the ellipse E
noncomputable def ellipse_E : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in (x^2) / 4 + (y^2) / 3 = 1 }

-- Define the proof problem for part (1)
theorem ellipse_equation (1 ≤ r ∧ r ≤ 3) :
  ∀ (p : ℝ × ℝ), p ∈ circle_F1 r → p ∈ circle_F2 r → p ∈ ellipse_E :=
sorry

-- Define the vectors and their dot product
noncomputable def vector_dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the proof problem for part (2)
theorem dot_product_constant (A P Q : ℝ × ℝ) (r : ℝ) (hA : A ∈ circle_O)
(hPA : P ∈ ellipse_E) (hQA : Q ∈ ellipse_E) :
  vector_dot_product (⟨P.1 - A.1, P.2 - A.2⟩) (⟨Q.1 - A.1, Q.2 - A.2⟩) = - (12 / 7) :=
sorry

end ellipse_equation_dot_product_constant_l634_634802


namespace class_size_l634_634778

def S : ℝ := 30

theorem class_size (total percent_dogs_videogames percent_dogs_movies number_students_prefer_dogs : ℝ)
  (h1 : percent_dogs_videogames = 0.5)
  (h2 : percent_dogs_movies = 0.1)
  (h3 : number_students_prefer_dogs = 18)
  (h4 : total * (percent_dogs_videogames + percent_dogs_movies) = number_students_prefer_dogs) :
  total = S :=
by
  sorry

end class_size_l634_634778


namespace average_length_of_strings_l634_634978

theorem average_length_of_strings (l1 l2 l3 : ℝ) (hl1 : l1 = 2) (hl2 : l2 = 5) (hl3 : l3 = 3) : 
  (l1 + l2 + l3) / 3 = 10 / 3 :=
by
  rw [hl1, hl2, hl3]
  change (2 + 5 + 3) / 3 = 10 / 3
  sorry

end average_length_of_strings_l634_634978


namespace XiaoMing_team_award_l634_634734

def points (x : ℕ) : ℕ := 2 * x + (8 - x)

theorem XiaoMing_team_award (x : ℕ) : 2 * x + (8 - x) ≥ 12 := 
by 
  sorry

end XiaoMing_team_award_l634_634734


namespace parabola_x_intercepts_count_l634_634110

theorem parabola_x_intercepts_count :
  ∃! x, ∃ y, x = -3 * y^2 + 2 * y + 2 ∧ y = 0 :=
by
  sorry

end parabola_x_intercepts_count_l634_634110


namespace angle_between_a_plus_b_and_b_l634_634860

open Real

-- Define vectors a and b with conditions from the problem.
def vec_a (x : ℝ) : ℝ × ℝ := (x + 1, sqrt 3)
def vec_b : ℝ × ℝ := (1, 0)

-- Condition for the dot product.
def dot_product_condition (x : ℝ) : Prop := 
  let a := vec_a x in
  let b := vec_b in
  (a.1 * b.1 + a.2 * b.2 = -2)

-- Function to compute the dot product of two vectors.
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Function to compute the norm of a vector.
def norm (u : ℝ × ℝ) : ℝ :=
  sqrt (u.1 * u.1 + u.2 * u.2)

-- Function to compute the cosine of the angle between two vectors.
def cos_angle (u v : ℝ × ℝ) : ℝ :=
  dot_product u v / (norm u * norm v)

-- Main theorem: Given the conditions, prove the angle between 'a + b' and 'b'.
theorem angle_between_a_plus_b_and_b (x : ℝ) (h : dot_product_condition x) :
  let a := vec_a x in
  let b := vec_b in
  let a_plus_b := (a.1 + b.1, a.2 + b.2) in
  acos (cos_angle a_plus_b b) = 2 * pi / 3 :=
sorry

end angle_between_a_plus_b_and_b_l634_634860


namespace find_children_and_coins_l634_634012

def condition_for_child (k m remaining_coins : ℕ) : Prop :=
  ∃ (received_coins : ℕ), (received_coins = k + remaining_coins / 7 ∧ received_coins * 7 = 7 * k + remaining_coins)

def valid_distribution (n m : ℕ) : Prop :=
  ∀ k (hk : 1 ≤ k ∧ k ≤ n),
  ∃ remaining_coins,
    condition_for_child k m remaining_coins

theorem find_children_and_coins :
  ∃ n m, valid_distribution n m ∧ n = 6 ∧ m = 36 :=
sorry

end find_children_and_coins_l634_634012


namespace polynomial_remainder_example_l634_634962

noncomputable def remainder_when_divided_by : Polynomial ℚ → ℚ → ℚ
| p, c => Polynomial.eval c p

theorem polynomial_remainder_example :
  let p := (Polynomial.X ^ 9 : Polynomial ℚ)
  let q1 := Polynomial.div p (Polynomial.C (1 / 3) - Polynomial.X)
  let r1 := remainder_when_divided_by p (1 / 3)
  let q2 := Polynomial.div q1.fst (Polynomial.C (1 / 3) - Polynomial.X)
  let r2 := remainder_when_divided_by q1.fst (1 / 3)
  r2 = 1 / 4374 := 
by {
  sorry
}

end polynomial_remainder_example_l634_634962


namespace ratio_of_areas_l634_634356

noncomputable def perimeter := (P : ℝ)

def side_length_square := perimeter / 4
def side_length_hexagon := perimeter / 6

def radius_circumscribed_circle_square := (P * real.sqrt 2) / 8
def radius_circumscribed_circle_hexagon := P / 6

def area_circle_square := real.pi * (radius_circumscribed_circle_square ^ 2)
def area_circle_hexagon := real.pi * (radius_circumscribed_circle_hexagon ^ 2)

theorem ratio_of_areas (P : ℝ) (h : P > 0) :
  (area_circle_square / area_circle_hexagon) = (9 / 8) :=
by
  -- The proof is omitted
  sorry

end ratio_of_areas_l634_634356


namespace circle_equation_with_tangent_line_l634_634277

theorem circle_equation_with_tangent_line
  (center : ℝ × ℝ) (line : ℝ × ℝ × ℝ) (r : ℝ)
  (h1 : center = (2, -1))
  (h2 : line = (3, 4, -7))
  (h3 : r = (real.abs (3 * 2 + 4 * -1 - 7) / real.sqrt (3^2 + 4^2)))
  : ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 1 :=
by
  sorry

end circle_equation_with_tangent_line_l634_634277


namespace max_ratio_of_mean_70_l634_634198

theorem max_ratio_of_mean_70 (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hmean : (x + y) / 2 = 70) : (x / y ≤ 99 / 41) :=
sorry

end max_ratio_of_mean_70_l634_634198


namespace particle_position_after_1991_l634_634727

noncomputable def particle_position (n : ℕ) : ℤ × ℤ :=
  if n = 0 then (0, 1)
  else let k := n - 1 in
       let m := (k + 1) / 2 in
       if k % 2 = 0 then (-2 * m, -2 * m - 1)
       else (-2 * m, 1 + 2 * m)

theorem particle_position_after_1991 : particle_position 1991 = (-45, -32) :=
  sorry

end particle_position_after_1991_l634_634727


namespace number_of_girls_l634_634285

variable (total_children : ℕ) (boys : ℕ)

theorem number_of_girls (h1 : total_children = 117) (h2 : boys = 40) : 
  total_children - boys = 77 := by
  sorry

end number_of_girls_l634_634285


namespace sum_of_series_l634_634763

def sum_of_fourth_powers (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) * (3 * n^2 + 3 * n - 1) / 30

theorem sum_of_series (n : ℕ) (h : n = 100) :
  2 * sum_of_fourth_powers n = 40802000 :=
by {
  rw h,
  let sum := sum_of_fourth_powers 100,
  have : sum = 20401000, sorry,
  rw [← this],
  norm_num,
}

end sum_of_series_l634_634763


namespace monotonically_increasing_interval_max_height_AD_l634_634052

noncomputable def f (x : ℝ) : ℝ := 
  2 * cos x * sin (x + (π / 6)) + sqrt 3 * sin x * cos x - sin x ^ 2

-- Statement 1: Prove the monotonicity intervals of the function.
theorem monotonically_increasing_interval : 
  ∀ x, 0 < x ∧ x < π → f x > 0 ↔ (0 < x ∧ x ≤ π / 6) ∨ (2 * π / 3 ≤ x ∧ x < π) := sorry

-- Statement 2: Prove the maximum length of the height AD in triangle ABC.
-- angle A satisfies f(A) = 2.
-- AB dot AC = sqrt(3)
theorem max_height_AD (A : ℝ) (b c : ℝ) : 
  f A = 2 ∧ b * c * cos (π / 6) = sqrt 3 → 
  let a := sqrt (b^2 + c^2 - 2 * b * c * cos (π / 6)) in
  (∀ A > 0 ∧ A < π, A = π / 6) →
  let h : ℝ := 1 / (sqrt 3 - 1) in
  h = (sqrt 3 + 1) / 2 := sorry

end monotonically_increasing_interval_max_height_AD_l634_634052


namespace arrange_2022_l634_634924

def digits := [2, 0, 2, 2]

def is_multiple_of_2 (n : ℕ) : Prop := n % 2 = 0

theorem arrange_2022 : 
  let valid_numbers := {n : ℕ | (digits.permutations.map (λ d, d.foldl (λ acc x, acc * 10 + x) 0)).to_finset ∋ n ∧ is_multiple_of_2 n} in
  valid_numbers.to_finset.card = 4 :=
by sorry

end arrange_2022_l634_634924


namespace probability_three_red_balls_l634_634709

theorem probability_three_red_balls :
  (7 / 21) * (6 / 20) * (5 / 19) = 1 / 38 := by
  sorry

end probability_three_red_balls_l634_634709


namespace triangle_area_hypotenuse_l634_634740

def point := (ℝ × ℝ)

def area_of_triangle (A B C : point) : ℝ :=
(1 / 2) * abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))

def distance (P Q : point) : ℝ :=
real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

noncomputable def hypotenuse_of_triangle (A B C : point) : ℝ :=
let d1 := distance A B,
    d2 := distance B C in
real.sqrt (d1 ^ 2 + d2 ^ 2)

theorem triangle_area_hypotenuse :
  let A := (3, 1) : point,
      B := (3, 6) : point,
      C := (8, 6) : point in
  (area_of_triangle A B C = 12.5) ∧ (hypotenuse_of_triangle A B C ≈ 7.1) :=
by
  let A := (3, 1) : point,
      B := (3, 6) : point,
      C := (8, 6) : point
  sorry

end triangle_area_hypotenuse_l634_634740


namespace total_distance_is_810_l634_634712

-- Define conditions as constants
constant first_day_distance : ℕ := 100
constant second_day_distance : ℕ := 3 * first_day_distance
constant third_day_distance : ℕ := second_day_distance + 110

-- The total distance traveled in three days
constant total_distance : ℕ := first_day_distance + second_day_distance + third_day_distance

-- The theorem to prove
theorem total_distance_is_810 : total_distance = 810 := 
  by
    -- The proof steps would go here, but we insert sorry to indicate the proof is not provided
    sorry

end total_distance_is_810_l634_634712


namespace cars_cannot_meet_l634_634263

-- Define the vertices and sides in the equilateral triangle grid
structure TriangleGrid where
  vertices : Set (ℝ × ℝ)
  sides : Set ((ℝ × ℝ) × (ℝ × ℝ))
  is_equilateral : ∀ (a b c : ℝ × ℝ), a ∈ vertices → b ∈ vertices → c ∈ vertices → 
                   (a, b) ∈ sides → (b, c) ∈ sides → (c, a) ∈ sides → 
                   -- Properties to assert the equilateral nature
                   dist a b = dist b c ∧ dist b c ≡ dist c a

-- Define points A and B on the same side of a triangle
variables (A B : ℝ × ℝ)
variable (grid : TriangleGrid)
variable (same_side : (A, B) ∈ grid.sides)

-- Define the motion of cars, where car's movement can be:
--   - Continue in the same direction
--   - Turn 120 degrees to the right
--   - Turn 120 degrees to the left
inductive CarMove
| straight 
| turn_left_120
| turn_right_120

-- Define the car's journey
structure CarJourney where
  start : ℝ × ℝ
  moves : List CarMove

-- Define the two cars starting their journey simultaneously
variables (car1 car2 : CarJourney)
variable (simultaneous_start : car1.start = A ∧ car2.start = B)
variable (same_speed : ∀ t, dist (car1.moves t) car1.start = dist (car2.moves t) car2.start)

-- Define the proposition that the cars cannot meet
theorem cars_cannot_meet : ∀ t, car1.moves t ≠ car2.moves t := 
by sorry

end cars_cannot_meet_l634_634263


namespace fixed_point_on_line_l634_634261

theorem fixed_point_on_line (m x y : ℝ) (h : ∀ m : ℝ, m * x - y + 2 * m + 1 = 0) : 
  (x = -2 ∧ y = 1) :=
sorry

end fixed_point_on_line_l634_634261


namespace tangent_AB_E1_l634_634434

noncomputable def ellipse_E : set (ℝ × ℝ) := { p | (p.1^2) / 4 + (p.2^2) = 1 }
noncomputable def ellipse_E1 (m n : ℝ) : set (ℝ × ℝ) := { p | (p.1^2) / (m^2) + (p.2^2) / (n^2) = 1 }
noncomputable def point_P := (p : ℝ × ℝ)

theorem tangent_AB_E1 (m n : ℝ) (h_ecc : (√3 / 2) = 1) (h_point : point_P ∈ ellipse_E1 m n) :
  ∀ (A B : ℝ × ℝ), A ∈ ellipse_E ∧ B ∈ ellipse_E ∧ M = (point_P) ∧
  line_through point_P A B,
  is_tangent_to_ellipse E1 m n :=
sorry

end tangent_AB_E1_l634_634434


namespace coefficient_a3b2c2_l634_634165

theorem coefficient_a3b2c2 (a b c : ℤ) : 
  coefficient (a - 3 * b^2 - c) ^ 6 (a^3 * b^2 * c^2) = -180 :=
sorry

end coefficient_a3b2c2_l634_634165


namespace ratio_of_pentagon_to_rectangle_l634_634352

theorem ratio_of_pentagon_to_rectangle (p l : ℕ) 
  (h1 : 5 * p = 30) (h2 : 2 * l + 2 * 5 = 30) : 
  p / l = 3 / 5 :=
by {
  sorry 
}

end ratio_of_pentagon_to_rectangle_l634_634352


namespace sufficient_but_not_necessary_l634_634432

theorem sufficient_but_not_necessary {a b : ℝ} (h : a > b ∧ b > 0) : 
  a^2 > b^2 ∧ (¬ (a^2 > b^2 → a > b ∧ b > 0)) :=
by 
  sorry

end sufficient_but_not_necessary_l634_634432


namespace books_loaned_out_l634_634348

theorem books_loaned_out
  (B : ℕ) (L : ℕ) (R : ℕ) (P : ℚ) (books_beginning : B = 75) (books_end : L = 64) (percent_returned : R = 80):
  ∃ x : ℕ, x = 55 ∧ (B - L = (1 - P/100) * x * Nat.toRat) :=
by
  sorry

end books_loaned_out_l634_634348


namespace distance_from_highest_point_to_plane_l634_634975

-- Define the radius R
def R : Real := 20 - 10 * Real.sqrt 2

-- Define the conditions and theorem statement
theorem distance_from_highest_point_to_plane : 
  let s := 2 * R in
  let fifth_sphere_height := 2 * R in
  let max_height := fifth_sphere_height + R in
  max_height = 20 :=
by
  -- Proof is omitted
  sorry

end distance_from_highest_point_to_plane_l634_634975


namespace find_a2010_l634_634551

noncomputable def seq (n : ℕ) : ℕ :=
if n = 1 then 2 else if n = 2 then 3 else
  (seq (n - 1) * seq (n - 2)) % 10

theorem find_a2010 : seq 2010 = 4 :=
sorry

end find_a2010_l634_634551


namespace solve_congruence_l634_634243

theorem solve_congruence :
  (∃ x : ℤ, 10 * x + 3 ≡ 7 [MOD 18] ∧ (x ≡ 4 [MOD 9])) ∧ 
  let a := 4 in
  let m := 9 in
  a + m = 13 :=
by
  sorry

end solve_congruence_l634_634243


namespace hire_charges_paid_by_B_l634_634319

theorem hire_charges_paid_by_B (total_cost : ℝ) (hours_A hours_B hours_C : ℝ) (b_payment : ℝ) :
  total_cost = 720 ∧ hours_A = 9 ∧ hours_B = 10 ∧ hours_C = 13 ∧ b_payment = (total_cost / (hours_A + hours_B + hours_C)) * hours_B → b_payment = 225 :=
by
  sorry

end hire_charges_paid_by_B_l634_634319


namespace part1_part2_l634_634520

-- Define the triangle with sides a, b, c and the properties given.
variable (a b c : ℝ) (A B C : ℝ)
variable (A_ne_zero : A ≠ 0)
variable (b_cos_C a_cos_A c_cos_B : ℝ)

-- Given conditions
variable (h1 : b_cos_C = b * Real.cos C)
variable (h2 : a_cos_A = a * Real.cos A)
variable (h3 : c_cos_B = c * Real.cos B)
variable (h_seq : b_cos_C + c_cos_B = 2 * a_cos_A)
variable (A_plus_B_plus_C_eq_pi : A + B + C = Real.pi)

-- Part 1
theorem part1 : (A = Real.pi / 3) :=
by sorry

-- Part 2 with additional conditions
variable (h_a : a = 3 * Real.sqrt 2)
variable (h_bc_sum : b + c = 6)

theorem part2 : (|Real.sqrt (b ^ 2 + c ^ 2 - b * c)| = Real.sqrt 30) :=
by sorry

end part1_part2_l634_634520


namespace div_remainder_l634_634307

theorem div_remainder (x : ℕ) (h : x = 2^40) : 
  (2^160 + 160) % (2^80 + 2^40 + 1) = 159 :=
by
  sorry

end div_remainder_l634_634307


namespace jerry_books_vs_action_figures_l634_634936

-- Define the initial conditions as constants
def initial_books : ℕ := 7
def initial_action_figures : ℕ := 3
def added_action_figures : ℕ := 2

-- Define the total number of action figures after adding
def total_action_figures : ℕ := initial_action_figures + added_action_figures

-- The theorem we need to prove
theorem jerry_books_vs_action_figures : initial_books - total_action_figures = 2 :=
by
  -- Proof placeholder
  sorry

end jerry_books_vs_action_figures_l634_634936


namespace final_expression_simplified_l634_634249

variable (x : ℝ)

def f : ℝ := ((3 * x + 6) - 5 * x) / 3

theorem final_expression_simplified :
  f x = - (2 / 3) * x + 2 :=
by
  sorry

end final_expression_simplified_l634_634249


namespace trig_cos2_minus_sin2_eq_neg_sqrt5_div3_l634_634047

open Real

theorem trig_cos2_minus_sin2_eq_neg_sqrt5_div3 (α : ℝ) (hα1 : 0 < α ∧ α < π) (hα2 : sin α + cos α = sqrt 3 / 3) :
  cos α ^ 2 - sin α ^ 2 = - sqrt 5 / 3 := 
  sorry

end trig_cos2_minus_sin2_eq_neg_sqrt5_div3_l634_634047


namespace largest_c_inequality_l634_634022

theorem largest_c_inequality :
  let x := λ i, (if i < 51 then (i : ℝ) else -(i : ℝ) - 1)
  (M : ℝ) (H : M = x 51)
  (Hsum : ∑ i in (finset.range 101), x i = 0) in
  ∃ c : ℝ, (c = 5151 / 50) ∧ ( ∑ i in (finset.range 101), (x i)^2 ≥ c * M^2) :=
begin
  sorry
end

end largest_c_inequality_l634_634022


namespace log_eq_res_l634_634573

theorem log_eq_res (y m : ℝ) (h₁ : real.log 5 / real.log 8 = y) (h₂ : real.log 125 / real.log 2 = m * y) : m = 9 := 
sorry

end log_eq_res_l634_634573


namespace log2_125_eq_9y_l634_634576

theorem log2_125_eq_9y (y : ℝ) (h : Real.log 5 / Real.log 8 = y) : Real.log 125 / Real.log 2 = 9 * y :=
by
  sorry

end log2_125_eq_9y_l634_634576


namespace right_triangle_area_l634_634898

theorem right_triangle_area (a b : ℝ) (h : a^2 - 7 * a + 12 = 0 ∧ b^2 - 7 * b + 12 = 0) : 
  ∃ A : ℝ, (A = 6 ∨ A = 3 * (Real.sqrt 7 / 2)) ∧ A = 1 / 2 * a * b := 
by 
  sorry

end right_triangle_area_l634_634898


namespace solve_x_perpendicular_l634_634854

def vec_a : ℝ × ℝ := (1, 3)
def vec_b (x : ℝ) : ℝ × ℝ := (3, x)

def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem solve_x_perpendicular (x : ℝ) (h : perpendicular vec_a (vec_b x)) : x = -1 :=
by {
  sorry
}

end solve_x_perpendicular_l634_634854


namespace magnitude_projection_of_a_onto_b_l634_634859

open Real

variables (a b : EuclideanSpace ℝ) (h_a : a = ![-1, -2]) (h_b : b = ![1, 0])

-- The magnitude of the projection vector of a onto b is 1
theorem magnitude_projection_of_a_onto_b : ‖(⟪a, b⟫ / ‖b‖^2 • b)‖ = 1 :=
by
  sorry

end magnitude_projection_of_a_onto_b_l634_634859


namespace petya_wins_probability_l634_634985

theorem petya_wins_probability : 
  ∃ p : ℚ, p = 1 / 256 ∧ 
  initial_stones = 16 ∧ 
  (∀ n, (1 ≤ n ∧ n ≤ 4)) ∧ 
  (turns(1) ∨ turns(2)) ∧ 
  (last_turn_wins) ∧ 
  (Petya_starts) ∧ 
  (Petya_plays_random ∧ Computer_plays_optimally) 
    → Petya_wins_with_probability p
:= sorry

end petya_wins_probability_l634_634985


namespace circle_eq_tangent_eq_l634_634445

-- Given conditions as definitions
def A := (3, 2)
def B := (1, 4)
def line := λ (a b : ℝ), a + b - 3 = 0

-- Problem 1: Equation of the circle
def center (a b : ℝ) :=
  ((3 - a) ^ 2 + (2 - b) ^ 2 = (1 - a) ^ 2 + (4 - b) ^ 2) ∧
  line a b

theorem circle_eq (a b r : ℝ) (hc : center a b) :
  (a, b) = (1, 2) ∧ r = 2 → (x - 1) ^ 2 + (y - 2) ^ 2 = 4 :=
sorry

-- Problem 2: Equations of tangent lines
def tangent_line (k : ℝ) (M : ℝ × ℝ) (C : ℝ × ℝ) :=
  let line_eq := λ (k : ℝ) (x y : ℝ), k * x - y + (1 - k * 3) = 0 in
  let d := λ (k : ℝ) (a b : ℝ), abs (k * a + b - (1 + 3 * k)) / sqrt (k^2 + 1) in
  d k 1 2 = 2

theorem tangent_eq :
  tangent_line (3 / 4) (3, 1) (1, 2) → (3 * x - 4 * y - 5 = 0) ∨ (x = 3) :=
sorry

end circle_eq_tangent_eq_l634_634445


namespace a_n_correct_S_n_correct_l634_634482

noncomputable def a : ℕ → ℕ
| 0       := 0  -- this is unused, as a_1 corresponds to a(1) in the given sequence
| (n + 1) := if n = 0 then 1 else 2 * a n + 2

noncomputable def a_closed_form (n : ℕ) : ℕ := 3 * 2^(n-1) - 2

theorem a_n_correct (n : ℕ) : a (n+1) = a_closed_form (n+1) := sorry

noncomputable def S (n : ℕ) : ℕ := ∑ i in finset.range n, a (i+1)

noncomputable def S_closed_form (n : ℕ) : ℕ := 3 * 2^n - 2 * n - 3

theorem S_n_correct (n : ℕ) : S n = S_closed_form n := sorry

end a_n_correct_S_n_correct_l634_634482


namespace find_some_multiplier_l634_634422

theorem find_some_multiplier (m : ℕ) :
  (422 + 404)^2 - (m * 422 * 404) = 324 ↔ m = 4 :=
by
  sorry

end find_some_multiplier_l634_634422


namespace regular_polygon_of_45_deg_l634_634732

def is_regular_polygon (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 2 ∧ 360 % k = 0 ∧ n = 360 / k

def regular_polygon_is_octagon (angle : ℕ) : Prop :=
  is_regular_polygon 8 ∧ angle = 45

theorem regular_polygon_of_45_deg : regular_polygon_is_octagon 45 :=
  sorry

end regular_polygon_of_45_deg_l634_634732


namespace martha_profit_l634_634970

theorem martha_profit :
  let loaves_baked := 60
  let cost_per_loaf := 1
  let morning_price := 3
  let afternoon_price := 3 * 0.75
  let evening_price := 2
  let morning_loaves := loaves_baked / 3
  let afternoon_loaves := (loaves_baked - morning_loaves) / 2
  let evening_loaves := loaves_baked - morning_loaves - afternoon_loaves
  let morning_revenue := morning_loaves * morning_price
  let afternoon_revenue := afternoon_loaves * afternoon_price
  let evening_revenue := evening_loaves * evening_price
  let total_revenue := morning_revenue + afternoon_revenue + evening_revenue
  let total_cost := loaves_baked * cost_per_loaf
  let profit := total_revenue - total_cost
  profit = 85 := 
by
  sorry

end martha_profit_l634_634970


namespace ratio_of_sides_l634_634797

open Real

variable (s y x : ℝ)

-- Assuming the rectangles and squares conditions
def condition1 := 4 * (x * y) + s * s = 9 * (s * s)
def condition2 := x = 2 * s
def condition3 := y = s

-- Stating the theorem
theorem ratio_of_sides (h1 : condition1 s y x) (h2 : condition2 s x) (h3 : condition3 s y) :
  x / y = 2 := by
  sorry

end ratio_of_sides_l634_634797


namespace petya_wins_probability_l634_634984

theorem petya_wins_probability : 
  ∃ p : ℚ, p = 1 / 256 ∧ 
  initial_stones = 16 ∧ 
  (∀ n, (1 ≤ n ∧ n ≤ 4)) ∧ 
  (turns(1) ∨ turns(2)) ∧ 
  (last_turn_wins) ∧ 
  (Petya_starts) ∧ 
  (Petya_plays_random ∧ Computer_plays_optimally) 
    → Petya_wins_with_probability p
:= sorry

end petya_wins_probability_l634_634984


namespace minor_arc_PQ_correct_l634_634194

-- Define the radius of the circle
def radius : ℝ := 24

-- Define the given angle in degrees
def anglePRQ : ℝ := 40

-- Define the total circumference of the circle
def total_circumference : ℝ := 2 * Real.pi * radius

-- Define the central angle corresponding to arc PQ
def central_angle_PQ : ℝ := 2 * anglePRQ

-- Define the length of the minor arc PQ
def minor_arc_PQ : ℝ := total_circumference * (central_angle_PQ / 360)

-- The proof statement
theorem minor_arc_PQ_correct : minor_arc_PQ = (32 * Real.pi) / 3 :=
by
  -- Since we are not providing the proof, we use sorry to placeholder the proof
  sorry

end minor_arc_PQ_correct_l634_634194


namespace tyler_double_flips_l634_634006

theorem tyler_double_flips :
  let jen_triple_flips := 16
      jen_double_flips := 10
      tyler_triple_flips := 12
      jen_total_flips := (jen_triple_flips * 3) + (jen_double_flips * 2)
      tyler_total_flips := (0.80 * jen_total_flips).toNat
      tyler_triple_flips_total := tyler_triple_flips * 3
  in tyler_total_flips - tyler_triple_flips_total = 18 →
     18 / 2 = 9 :=
by
  intros
  sorry

end tyler_double_flips_l634_634006


namespace itzayana_height_difference_l634_634561

def Itzayana_taller_than_Zora 
(B : ℕ) (Z : ℕ) (h1 : B = 64) (h2 : Z = B - 8)
(avg_height : ℕ) (total_height : ℕ)
(h3 : avg_height = 61) (h4 : total_height = avg_height * 4)
(h5 : total_height = 244) : Prop :=
∃ (I : ℕ), (I = total_height - (Z + B + B)) ∧ (I - Z = 4)

theorem itzayana_height_difference
  (B : ℕ) (Z : ℕ) (h1 : B = 64) (h2 : Z = B - 8)
  (avg_height : ℕ) (total_height : ℕ)
  (h3 : avg_height = 61) (h4 : total_height = avg_height * 4)
  (h5 : total_height = 244) : B = 64 ∧ Z = 56 ∧ (∃ I, I = 60 ∧ I - Z = 4) :=
by {
  rw [h1, h2, h3, h4] at h5,
  have B64 : B = 64 := h1,
  have Z56 : Z = B - 8 := h2,
  have avg : avg_height = 61 := h3,
  have tot : total_height = 244 := h5,
  have total_exp : total_height = avg_height * 4 := h4,
  rw h1 at Z56,
  rw h2 at Z56,
  use 60,
  simp,
  sorry
}

end itzayana_height_difference_l634_634561


namespace quadratic_factor_conditions_l634_634271

theorem quadratic_factor_conditions (b : ℤ) :
  (∃ m n p q : ℤ, m * p = 15 ∧ n * q = 75 ∧ mq + np = b) → ∃ (c : ℤ), b = c :=
sorry

end quadratic_factor_conditions_l634_634271


namespace find_circle_radius_l634_634914

variables (a b : ℝ)
hypothesis hab : a > 0
hypothesis hbc : b > 0
hypothesis h_abc : ∀ R:ℝ, ∀ θ1 θ2: ℝ, 2 * θ1 = θ2 → sin θ1 * (R * θ1) = a → sin θ2 * (R * θ2) = b 

theorem find_circle_radius (h : 2 * (arc_length_angle AB) = (arc_length_angle AC)) : 
  radius_circle AC AB = (a ^ 2 / (sqrt (4 * a ^ 2 - b ^ 2))) :=
sorry

end find_circle_radius_l634_634914


namespace product_of_positive_real_solutions_eq_eight_l634_634507

noncomputable def product_of_positive_real_solutions : ℂ :=
  let solutions := {x : ℂ | x^8 = -256 ∧ x.re > 0}
  ∏ x in solutions, x

theorem product_of_positive_real_solutions_eq_eight :
  product_of_positive_real_solutions = 8 :=
sorry

end product_of_positive_real_solutions_eq_eight_l634_634507


namespace max_digit_sum_base_8_l634_634676

theorem max_digit_sum_base_8 (n : ℕ) (h : n < 1728) : 
  let digit_sum_base_8 (m : ℕ) : ℕ := (List.of_digits 8 m).sum in
  ∃ k : ℕ, k = digit_sum_base_8 n ∧ k ≤ 23 := sorry

end max_digit_sum_base_8_l634_634676


namespace triangle_area_l634_634154

theorem triangle_area : 
  ∀ (P Q R S : Type) (PQ PR QR : ℝ) (PS : ℝ),
  isosceles_triangle P Q R ∧ 
  altitude_bisecting PS QR ∧
  PQ = 13 ∧ PR = 13 ∧ QR = 10 ∧ QS = 5 ∧ SR = 5 ∧ PS * PS = 169 - 25 →
  1 / 2 * QR * PS = 60 :=
by 
  intros,
  sorry 

end triangle_area_l634_634154


namespace diagonals_not_parallel_to_sides_in_regular_32_gon_l634_634865

theorem diagonals_not_parallel_to_sides_in_regular_32_gon :
  let n := 32 in
  let total_diagonals := n * (n - 3) / 2 in
  let parallel_diagonals := 16 * (n - 4) / 2 in
  let non_parallel_diagonals := total_diagonals - parallel_diagonals in
  non_parallel_diagonals = 240 :=
by
  sorry

end diagonals_not_parallel_to_sides_in_regular_32_gon_l634_634865


namespace right_triangle_area_l634_634892

def roots (a b : ℝ) : Prop :=
  a * b = 12 ∧ a + b = 7

def area (A : ℝ) : Prop :=
  A = 6 ∨ A = 3 * Real.sqrt 7 / 2

theorem right_triangle_area (a b A : ℝ) (h : roots a b) : area A := 
by 
  -- The proof steps would go here
  sorry

end right_triangle_area_l634_634892


namespace area_ratio_of_similar_triangles_l634_634547

theorem area_ratio_of_similar_triangles
  (circle : Type)
  (A B C D E F : circle)
  (h_parallel : ∀ (line1 line2 : Line), (line1 ∥ line2) → (line1 = AB ∧ line2 = CD))
  (h_angle : ∀ (angle : Angle), angle.angle = ∠AFB → angle = β)
  (h_intersection : (AC ∩ BD) = E)
  (ratio : ℝ) :
  ratio = 1 :=
by
  sorry

end area_ratio_of_similar_triangles_l634_634547


namespace probability_of_prime_l634_634347

-- Define the sample space
def sample_space : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}.toFinset

-- Define the set of prime numbers within the sample space
def prime_numbers : Finset Nat := {2, 3, 5, 7}.toFinset

-- Define the probability function
def probability (event : Finset Nat) (space : Finset Nat) : Real :=
  (event.card : Real) / (space.card : Real)

-- The main theorem stating the probability that a randomly chosen number from 1 to 10 is a prime number is 0.4
theorem probability_of_prime : probability prime_numbers sample_space = 0.4 := by
  sorry

end probability_of_prime_l634_634347


namespace Petya_win_prob_is_1_over_256_l634_634997

/-!
# The probability that Petya will win given the conditions in the game "Heap of Stones".
-/

/-- Function representing the probability that Petya will win given the initial conditions.
Petya starts with 16 stones and takes a random number of stones each turn, while the computer
follows an optimal strategy. -/
noncomputable def Petya_wins_probability (initial_stones : ℕ) (random_choices : list ℕ) : ℚ :=
1 / 256

/-- Proof statement: The probability that Petya will win under the given conditions is 1/256. -/
theorem Petya_win_prob_is_1_over_256 : Petya_wins_probability 16 [1, 2, 3, 4] = 1 / 256 :=
sorry

end Petya_win_prob_is_1_over_256_l634_634997


namespace machines_working_together_l634_634662

theorem machines_working_together (x : ℝ) :
  let R_time := x + 4
  let Q_time := x + 9
  let P_time := x + 12
  (1 / P_time + 1 / Q_time + 1 / R_time) = 1 / x ↔ x = 1 := 
by
  sorry

end machines_working_together_l634_634662


namespace am_gm_inequality_l634_634203

variable {n : ℕ} (x : Fin n → ℝ)

theorem am_gm_inequality (h : ∀ i : Fin n, x i > 0) :
  (∑ i, (x i)^2 / (x ((i + 1) % n))) ≥ ∑ i, x i :=
by
  sorry

end am_gm_inequality_l634_634203


namespace number_of_solutions_is_2_l634_634647

def num_integer_solutions : ℕ :=
  let solutions := { (x, y) | |x - 2 * y - 3| + |x + y + 1| = 1 ∧ x ∈ ℤ ∧ y ∈ ℤ }
  solutions.card

theorem number_of_solutions_is_2 : num_integer_solutions = 2 := sorry

end number_of_solutions_is_2_l634_634647


namespace unique_B_cube_l634_634584

open Matrix

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := !![p, q; r, s]

theorem unique_B_cube (B : Matrix (Fin 2) (Fin 2) ℝ) (hB : B^4 = 0) :
  ∃! (C : Matrix (Fin 2) (Fin 2) ℝ), C = B^3 :=
sorry

end unique_B_cube_l634_634584


namespace find_m_l634_634852

-- Defining the sets P and Q and the condition P ∩ Q ≠ ∅
def P (m : ℕ) : Set ℕ := {0, m}
def Q : Set ℕ := {x | 2 * x^2 - 5 * x < 0 ∧ x ∈ Int.toNat)

-- Main mathematical statement
theorem find_m (m : ℕ) (h : P m ∩ Q ≠ ∅) : m = 1 ∨ m = 2 := by
  sorry

end find_m_l634_634852


namespace greatest_num_with_odd_factors_div_by_3_lt_150_l634_634973

theorem greatest_num_with_odd_factors_div_by_3_lt_150 : ∃ (n : ℕ), 
  n < 150 ∧ 
  (∃ k : ℕ, n = k * k) ∧ 
  3 ∣ n ∧
  ∀ m : ℕ, m < 150 ∧ 
  (∃ k : ℕ, m = k * k) ∧ 
  3 ∣ m → m ≤ n :=
begin
  sorry
end

end greatest_num_with_odd_factors_div_by_3_lt_150_l634_634973


namespace intersection_of_diagonals_l634_634822

-- Defining a convex hexagon and some of its properties
structure ConvexHexagon (α : Type*) :=
(A B C D E F : α)
(is_convex : convex α (set.insert A (set.insert B (set.insert C (set.insert D (set.insert E (set.singleton F)))))))

-- Definition of the diagonals bisecting the area
def bisect_area (S : ConvexHexagon ℝ) (diagonal : ℝ × ℝ → Prop) : Prop :=
-- The definition for area bisection will be added here as required
sorry

-- The theorem to prove
theorem intersection_of_diagonals (S : ConvexHexagon ℝ) 
  (h1 : bisect_area S (λ p, p = (S.A, S.D)))
  (h2 : bisect_area S (λ p, p = (S.B, S.E)))
  (h3 : bisect_area S (λ p, p = (S.C, S.F))) :
  ∃ (P : ℝ × ℝ), 
    (∀ x, x = (S.A, S.D) → x ∈ set.singleton P) ∧
    (∀ y, y = (S.B, S.E) → y ∈ set.singleton P) ∧
    (∀ z, z = (S.C, S.F) → z ∈ set.singleton P) := 
sorry

end intersection_of_diagonals_l634_634822


namespace range_of_omega_l634_634436

theorem range_of_omega (ω : ℝ) (hω : ω > 0) 
  (h_zero : ∀ x ∈ Icc (-(π / 3)) 0, f (ω * x) = sin (ω * x) + 1 ∧ (sin (ω * x) + 1) = 0 → 
    ∃ a b c ∈ Icc (-(π / 3)) 0, a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (51 / 4 : ℝ) ≤ ω ∧ ω < (75 / 4 : ℝ) := by
  sorry

end range_of_omega_l634_634436


namespace find_a_and_intersection_point_l634_634928

-- Define the necessary conditions
def parabola (a : ℝ) (x : ℝ) : ℝ := a * (x - 3)^2 - 1

def passes_through (a : ℝ) (p : ℝ × ℝ) : Prop :=
  parabola a p.1 = p.2

-- Define the main theorem
theorem find_a_and_intersection_point :
  ∃ a : ℝ, passes_through a (2, 1) ∧ a = 2 ∧ (1 = 1) :=
begin
  -- This will prove the parabola passes through the given point and matches the given condition.
  sorry
end

end find_a_and_intersection_point_l634_634928


namespace tangent_line_at_point_l634_634021
noncomputable theory

open Real

def f (x : ℝ) : ℝ := exp (3 * x) - x ^ 3

theorem tangent_line_at_point :
  ∀ (x y : ℝ), (x = 0) ∧ (y = f 0) → (3 * x - y + 1 = 0) :=
begin
  intros x y h,
  sorry
end

end tangent_line_at_point_l634_634021


namespace algebraic_expression_evaluation_l634_634818

theorem algebraic_expression_evaluation (m : ℝ) (h : m^2 - m - 3 = 0) : m^2 - m - 2 = 1 := 
by
  sorry

end algebraic_expression_evaluation_l634_634818


namespace amount_r_receives_l634_634736

theorem amount_r_receives (total_sum : ℝ) (ratios_pqrs : ℝ × ℝ × ℝ × ℝ) (ratios_qrs : ℝ × ℝ × ℝ) :
  ∃ (amount_r : ℝ), 
    total_sum = 5060 ∧
    ratios_pqrs = (3, 5, 7, 9) ∧
    ratios_qrs = (10, 11, 13) ∧ 
    amount_r ≈ 1475.83 :=
by
  sorry

end amount_r_receives_l634_634736


namespace minimum_black_edges_5x5_grid_l634_634708

-- Definitions
def small_square :=
  { edges : Finset (ℕ × ℕ) // edges.card = 4 }

def isosceles_triangle :=
  { triangles : Finset (small_square × small_square) // ∀ sq ∈ triangles, sq.1 = sq.2 }

def black_edges (sq : small_square) : Finset (ℕ × ℕ) :=
  { edge ∈ sq.edges | edge ∈ isosceles_triangle.edges }

def grid (n : ℕ) :=
  Finset (Fin n × Fin n)

-- Minimum black edges on a 5x5 grid

theorem minimum_black_edges_5x5_grid : minimum_black_edges (grid 5) = 5 :=
by
  sorry

end minimum_black_edges_5x5_grid_l634_634708


namespace no_two_right_angles_in_triangle_l634_634296

theorem no_two_right_angles_in_triangle (T : Type) [triangle T] :
  ¬ (∃ A B C : T, ∠A = 90 ∧ ∠B = 90) := by
sorry

end no_two_right_angles_in_triangle_l634_634296


namespace find_a2010_l634_634550

noncomputable def seq (n : ℕ) : ℕ :=
if n = 1 then 2 else if n = 2 then 3 else
  (seq (n - 1) * seq (n - 2)) % 10

theorem find_a2010 : seq 2010 = 4 :=
sorry

end find_a2010_l634_634550


namespace correct_statement_l634_634236

def quadratic_function (x : ℝ) : ℝ := (x - 1)^2 + 5

theorem correct_statement (x : ℝ) (h : x > 1) : 
  ∃ δ > 0, ∀ x₁ x₂, x₁ > 1 ∧ x₂ > 1 ∧ x₁ < x₂ -> quadratic_function x₁ < quadratic_function x₂ :=
begin
  sorry
end

end correct_statement_l634_634236


namespace find_f2_plus_f_l634_634091

variable {f : ℝ → ℝ}
variable (hf2 : f 2 = -5) (hf'2 : deriv f 2 = -3)

theorem find_f2_plus_f'_2 : f 2 + deriv f 2 = -8 :=
by
  rw [hf2, hf'2]
  exact rfl

end find_f2_plus_f_l634_634091


namespace odd_positives_l634_634306

theorem odd_positives (x : ℕ) (h : x = 87) : 2 * x - 1 = 173 := by
  rw [h]
  norm_num

end odd_positives_l634_634306


namespace product_of_powers_l634_634650

theorem product_of_powers :
  ((-1 : Int)^3) * ((-2 : Int)^2) = -4 := by
  sorry

end product_of_powers_l634_634650


namespace polynomial_expansion_zero_l634_634120

theorem polynomial_expansion_zero (a : Fin 2015 → ℝ) :
  (∀ x : ℝ, (1 - 2 * x) ^ 2014 = ∑ i in Finset.range 2015, a i * x ^ i) →
  ∑ i in Finset.range 2015, a i / 2 ^ i = 0 :=
by
  intro h
  have h2 := h (1 / 2)
  sorry

end polynomial_expansion_zero_l634_634120


namespace intersection_points_l634_634515

def line1 := {p : ℝ × ℝ | p.1 + p.2 + 1 = 0}
def line2 := {p : ℝ × ℝ | 2*p.1 - p.2 + 8 = 0}
def line3 (a : ℝ) := {p : ℝ × ℝ | a*p.1 + 3*p.2 - 5 = 0}

theorem intersection_points (a : ℝ) (h : ∃ p₁ p₂ : ℝ × ℝ, 
  (p₁ ∈ line1 ∧ p₁ ∈ line2) ∧ (p₂ ∈ line1 ∧ p₂ ∈ line3 a) ∧ (p₁ ≠ p₂ ∨ (p₁ ∈ line2 ∧ p₁ ∈ line3 a))) :
  a ∈ {-3, -6} :=
sorry

end intersection_points_l634_634515


namespace reflect_hyperbola_over_line_l634_634235

theorem reflect_hyperbola_over_line :
  ∀ (x y : ℝ), (x * y = 1 ∧ line_eq : y = 2 * x) → 12 * y^2 + 7 * x * y - 12 * x^2 = 25 :=
by
  intro x y h
  sorry

end reflect_hyperbola_over_line_l634_634235


namespace part1_part2_part3_l634_634884

-- Part 1
def harmonic_fraction (num denom : ℚ) : Prop :=
  ∃ a b : ℚ, num = a - 2 * b ∧ denom = a^2 - b^2 ∧ ¬(∃ x : ℚ, a - 2 * b = (a - b) * x)

theorem part1 (a b : ℚ) (h : harmonic_fraction (a - 2 * b) (a^2 - b^2)) : true :=
  by sorry

-- Part 2
theorem part2 (a : ℕ) (h : harmonic_fraction (x - 1) (x^2 + a * x + 4)) : a = 4 ∨ a = 5 :=
  by sorry

-- Part 3
theorem part3 (a b : ℚ) :
  (4 * a^2 / (a * b^2 - b^3) - a / b * 4 / b) = (4 * a / (ab - b^2)) :=
  by sorry

end part1_part2_part3_l634_634884


namespace find_two_digit_integers_l634_634103

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem find_two_digit_integers
    (a b : ℕ) :
    10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ 
    (a = b + 12 ∨ b = a + 12) ∧
    (a / 10 = b / 10 ∨ a % 10 = b % 10) ∧
    (sum_of_digits a = sum_of_digits b + 3 ∨ sum_of_digits b = sum_of_digits a + 3) :=
sorry

end find_two_digit_integers_l634_634103


namespace find_initial_typists_l634_634500

noncomputable def initial_typists (x : ℕ) : Prop :=
  let rate := 48 / (20 * x)
  (rate * 30 * 60 = 216) → x = 20

theorem find_initial_typists : ∃ x : ℕ, initial_typists x :=
by {
  use 20,
  unfold initial_typists,
  intros,
  sorry
}

end find_initial_typists_l634_634500


namespace geometric_sum_l634_634956

theorem geometric_sum (n : ℤ) (x : ℝ) : 1 + x + x^2 + ∑ i in range n, x^i = (x^(n+1) - 1) / (x - 1) := 
sorry

end geometric_sum_l634_634956


namespace right_triangle_area_l634_634900

theorem right_triangle_area (a b : ℝ) (h : a^2 - 7 * a + 12 = 0 ∧ b^2 - 7 * b + 12 = 0) : 
  ∃ A : ℝ, (A = 6 ∨ A = 3 * (Real.sqrt 7 / 2)) ∧ A = 1 / 2 * a * b := 
by 
  sorry

end right_triangle_area_l634_634900


namespace smallest_lambda_for_triangle_inequality_l634_634420

theorem smallest_lambda_for_triangle_inequality :
  ∃ λ : ℝ, λ = (2 * Real.sqrt 2 + 1) / 7 ∧
  ∀ (a b c : ℝ), a ≥ (b + c) / 3 →
    a * c + b * c - c^2 ≤ λ * (a^2 + b^2 + 3 * c^2 + 2 * a * b - 4 * b * c) :=
by
  sorry

end smallest_lambda_for_triangle_inequality_l634_634420


namespace total_snowfall_amount_l634_634143

/-- The snowfall amounts on Monday and Tuesday in different units.
  We need to convert them all to inches and determine the total amount.
-/
variables (monday_morning : ℝ) (monday_afternoon : ℝ) (tuesday_morning_cm : ℝ) (tuesday_afternoon_mm : ℝ)

def inch_to_cm : ℝ := 2.54
def cm_to_mm : ℝ := 10

noncomputable def total_snowfall_in_inches : ℝ :=
  let tuesday_morning_inches := tuesday_morning_cm / inch_to_cm
  let tuesday_afternoon_inches := (tuesday_afternoon_mm / cm_to_mm) / inch_to_cm
  monday_morning + monday_afternoon + tuesday_morning_inches + tuesday_afternoon_inches

variables (monday_morning_value : 0.125) (monday_afternoon_value : 0.5) 
          (tuesday_morning_cm_value : 1.35) (tuesday_afternoon_mm_value : 25)

theorem total_snowfall_amount :
  total_snowfall_in_inches monday_morning_value monday_afternoon_value 
                           tuesday_morning_cm_value tuesday_afternoon_mm_value ≈ 2.141 := 
sorry

end total_snowfall_amount_l634_634143


namespace min_distance_P_A_B_l634_634806

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def on_line (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  x - y + 5 = 0

def A : ℝ × ℝ := (-3, 5)
def B : ℝ × ℝ := (2, 15)

theorem min_distance_P_A_B :
  ∃ P : ℝ × ℝ, on_line P ∧ ∀ Q : ℝ × ℝ, on_line Q → distance Q A + distance Q B ≥ real.sqrt 593 :=
sorry

end min_distance_P_A_B_l634_634806


namespace polynomial_expansion_l634_634443

/-- The polynomial expansion of (2x - 1) raised to the power 2023 --/
theorem polynomial_expansion :
  ∃ a : Fin 2024 → ℤ, 
    (∀ x : ℤ, (2 * x - 1) ^ 2023 = ∑ i : Fin 2024, (a i) * x ^ i) ∧ 
    (a 0 = -1) ∧ 
    ((∑ i in Finset.range 2024, if i = 0 then 0 else a i) = 2) ∧ 
    ((∑ i in Finset.range 2024, if i = 0 then 0 else (i * (a ⟨i, _, by linarith⟩))) = 4046) :=
by
  sorry

end polynomial_expansion_l634_634443


namespace part_a_part_b_l634_634304
open Nat

-- Define nine-digit numbers with non-repeating digits from 1 to 9
def is_nine_digit_non_repeating (n : ℕ) : Prop :=
  (n >= 123456789 ∧ n <= 987654321) ∧ (∀ i j, i ≠ j → n.digits.getDigit i ≠ n.digits.getDigit j)

-- Define a conditional pair
def is_conditional_pair (a b : ℕ) : Prop :=
  a + b = 987654321 ∧ is_nine_digit_non_repeating a ∧ is_nine_digit_non_repeating b

-- Statement for part (a)
theorem part_a : ∃ a b, is_conditional_pair a b ∧ a ≠ b := 
  sorry

-- Statement for part (b)
theorem part_b : ∃ n, odd n ∧ ((∃ list, list.pairs.is_conditional_pair) = n) := 
  sorry

end part_a_part_b_l634_634304


namespace probability_of_drawing_red_ball_l634_634288

theorem probability_of_drawing_red_ball :
  (let left_drawer := (0, 5), -- (red, white)
       middle_drawer := (1, 1), -- (red, white)
       right_drawer := (2, 1)  -- (red, white)
   in (1 / 3) * (middle_drawer.1 / (middle_drawer.1 + middle_drawer.2)) +
      (1 / 3) * (right_drawer.1 / (right_drawer.1 + right_drawer.2))) = 7 / 18 :=
by
  sorry

end probability_of_drawing_red_ball_l634_634288


namespace distinct_remainders_count_l634_634871

theorem distinct_remainders_count {N : ℕ} (hN : N = 420) :
  ∃ (count : ℕ), (∀ n : ℕ, n ≥ 1 ∧ n ≤ N → ((n % 5 ≠ n % 6) ∧ (n % 5 ≠ n % 7) ∧ (n % 6 ≠ n % 7))) →
  count = 386 :=
by {
  sorry
}

end distinct_remainders_count_l634_634871


namespace sufficient_condition_parallel_plane_l634_634587

variables {m n : Type} [line m] [line n]
variables {α β : Type} [plane α] [plane β]

theorem sufficient_condition_parallel_plane (h1 : m ⊥ n) (h2 : n ⊥ α) (h3 : ¬ (m ⊆ α)) : m ∥ α :=
sorry

end sufficient_condition_parallel_plane_l634_634587


namespace mary_travel_time_l634_634217

noncomputable def ambulance_speed : ℝ := 60
noncomputable def don_speed : ℝ := 30
noncomputable def don_time : ℝ := 0.5

theorem mary_travel_time : (don_speed * don_time) / ambulance_speed * 60 = 15 := by
  sorry

end mary_travel_time_l634_634217


namespace teal_bakery_revenue_l634_634155

theorem teal_bakery_revenue :
    let pumpkin_pies := 4
    let pumpkin_pie_slices := 8
    let pumpkin_slice_price := 5
    let custard_pies := 5
    let custard_pie_slices := 6
    let custard_slice_price := 6
    let total_pumpkin_slices := pumpkin_pies * pumpkin_pie_slices
    let total_custard_slices := custard_pies * custard_pie_slices
    let pumpkin_revenue := total_pumpkin_slices * pumpkin_slice_price
    let custard_revenue := total_custard_slices * custard_slice_price
    let total_revenue := pumpkin_revenue + custard_revenue
    total_revenue = 340 :=
by
  sorry

end teal_bakery_revenue_l634_634155


namespace decimal_representation_digits_l634_634674

noncomputable def pow (a : ℝ) (b : ℝ) : ℝ := a^b

theorem decimal_representation_digits :
  let x := 10 ^ 2003 + 1 in
  (pow x (12/11) - (pow x (12/11) % 1)) * 1000 % 1000 = 909 :=
by
  sorry

end decimal_representation_digits_l634_634674


namespace college_entrance_exam_candidates_l634_634655

theorem college_entrance_exam_candidates:
  ∃ n : ℕ, n = 72 ∧ 
  (∀ (candidates gates : ℕ), 
    candidates = 4 ∧ 
    gates = 2 ∧ 
    (∀ (g1 g2 : ℕ), g1 > 0 ∧ g2 > 0) → 
    n = (nat.choose candidates 1 * nat.perm gates gates * nat.perm (candidates - 1) (candidates - 1) +
         nat.perm candidates 2 * nat.perm ((candidates - 2) / 2) ((candidates - 2) / 2))) :=
by
  sorry

end college_entrance_exam_candidates_l634_634655


namespace pairs_opposite_proof_l634_634365

-- Lean statement for mathematically equivalent proof problem
theorem pairs_opposite_proof :
  {a b c d : ℤ} (ha : a = 3^2) (hb : b = -(3^2)) (hc1 : c = -(+4)) (hc2 : d = +(-4)) (he1 : e = -3)  
  (he2 : f = -|-3|) (hg1 : g = -2^3) (hg2 : h = (-2)^3) :
  (a = 9 ∧ b = -9 ∧ a = -b) ∧ 
  (c = -4 ∧ d = -4 ∧ c ≠ -d) ∧
  (e = -3 ∧ f = -3 ∧ e ≠ -f) ∧
  (g = -8 ∧ h = -8 ∧ g ≠ -h) :=
by sorry

end pairs_opposite_proof_l634_634365


namespace monotone_increasing_intervals_min_max_values_l634_634086

noncomputable def f (x : ℝ) : ℝ := 2 * sin x * cos x + sqrt 3 * cos (2 * x) + 2

theorem monotone_increasing_intervals : 
  ∀ k : ℤ, 
  ∃ I : set ℝ, 
  I = {x | k * real.pi - 5 * real.pi / 12 ≤ x ∧ x ≤ k * real.pi + real.pi / 12} ∧ 
  (∀ x1 x2 : ℝ, x1 ∈ I ∧ x2 ∈ I ∧ x1 ≤ x2 → f x1 ≤ f x2) :=
sorry

theorem min_max_values : 
  ∀ (a b : ℝ), 
  a = -real.pi / 3 → 
  b = real.pi / 3 → 
  ∃ (min max : ℝ), 
  min = 2 - sqrt 3 ∧ 
  max = 4 ∧ 
  (∀ x ∈ set.Icc a b, min ≤ f x ∧ f x ≤ max) ∧
  (∃ x_min ∈ set.Icc a b, f x_min = min) ∧
  (∃ x_max ∈ set.Icc a b, f x_max = max) :=
sorry

end monotone_increasing_intervals_min_max_values_l634_634086


namespace pool_capacity_l634_634355

theorem pool_capacity (C : ℝ) (initial_water : ℝ) :
  0.85 * C - 0.70 * C = 300 → C = 2000 :=
by
  intro h
  sorry

end pool_capacity_l634_634355


namespace largest_power_of_3_in_e_q_l634_634000

noncomputable def q : ℝ := ∑ k in (Finset.range 8).succ, (k : ℝ) * Real.log (Nat.factorial k) / Real.log 3

theorem largest_power_of_3_in_e_q :
  ∃ n : ℕ, n = 34 ∧ ∃ m : ℝ, (e^q = 3^n * m ∧ ¬ (∃ p : ℝ, (p > m ∧ e^q = 3^n * 3 * p))) := sorry

end largest_power_of_3_in_e_q_l634_634000


namespace next_meeting_time_at_B_l634_634702

-- Definitions of conditions
def perimeter := 800 -- Perimeter of the block in meters
def t1 := 1 -- They meet for the first time after 1 minute
def AB := 100 -- Length of side AB in meters
def BC := 300 -- Length of side BC in meters
def CD := 100 -- Length of side CD in meters
def DA := 300 -- Length of side DA in meters

-- Main theorem statement
theorem next_meeting_time_at_B :
  ∃ t : ℕ, t = 9 ∧ (∃ m1 m2 : ℕ, ((t = m1 * m2 + 1) ∧ m2 = 800 / (t1 * (AB + BC + CD + DA))) ∧ m1 = 9) :=
sorry

end next_meeting_time_at_B_l634_634702


namespace Kyle_Peter_not_same_set_l634_634065

theorem Kyle_Peter_not_same_set 
  (a : ℕ → ℝ) 
  (distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (length_10 : ∀ i, i < 10) : 
  let Kyle_set := { (a i - a j)^2 | i j : ℕ, i < j, j < 10 }
  let Peter_set := { abs ((a i)^2 - (a j)^2) | i j : ℕ, i < j, j < 10 }
  in Kyle_set ≠ Peter_set :=
by sorry

end Kyle_Peter_not_same_set_l634_634065


namespace relationship_x1_x2_l634_634842

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

theorem relationship_x1_x2 (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : f x1 = f x2) :
  x1 > 2 - x2 :=
begin
  sorry
end

end relationship_x1_x2_l634_634842


namespace log_sum_of_consecutive_integers_l634_634389

theorem log_sum_of_consecutive_integers :
  ∀ (a b : ℝ), (a = 5) → (b = 6) → 5 < log 10 475728 ∧ log 10 475728 < 6 → a + b = 11 :=
by
  intros a b ha hb hlog
  rw [ha, hb]
  exact rfl

end log_sum_of_consecutive_integers_l634_634389


namespace vector_dot_product_theorem_l634_634493

noncomputable def dot_product_eq (a b c : ℝ^3) : Prop :=
  (a ⬝ b = 5) ∧ (a ⬝ c = -7) ∧ (b ⬝ c = 2)

theorem vector_dot_product_theorem (a b c : ℝ^3) (h : dot_product_eq a b c) :
  b ⬝ (5 • c - 3 • a + 4 • b) = -5 + 4 * ∥b∥^2 :=
by
  cases h with
  | intro hab hrest =>
    cases hrest with hac hbc =>
    sorry

end vector_dot_product_theorem_l634_634493


namespace ant_species_A_day_4_l634_634750

theorem ant_species_A_day_4:
  ∃ a b c : ℕ, a + b + c = 50 ∧ 16 * a + 81 * b + 625 * c = 6230 ∧ 16 * a = 736 :=
begin
  sorry
end

end ant_species_A_day_4_l634_634750


namespace area_of_right_triangle_from_roots_l634_634895

theorem area_of_right_triangle_from_roots :
  ∀ (a b : ℝ), (a^2 - 7*a + 12 = 0) → (b^2 - 7*b + 12 = 0) →
  (∃ (area : ℝ), (area = 6) ∨ (area = (3 * real.sqrt 7) / 2)) :=
by
  intros a b ha hb
  sorry

end area_of_right_triangle_from_roots_l634_634895


namespace general_formula_for_sequence_l634_634967

theorem general_formula_for_sequence :
  ∀ (a : ℕ → ℕ), (a 0 = 1) → (a 1 = 1) →
  (∀ n, 2 ≤ n → a n = 2 * a (n - 1) - a (n - 2)) →
  ∀ n, a n = (2^n - 1)^2 :=
by
  sorry

end general_formula_for_sequence_l634_634967


namespace range_of_m_l634_634095

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 - 4 * x ≥ m) → m ≤ -3 := 
sorry

end range_of_m_l634_634095


namespace option_C_qualified_l634_634360

-- Define the acceptable range
def lower_bound : ℝ := 25 - 0.2
def upper_bound : ℝ := 25 + 0.2

-- Define the option to be checked
def option_C : ℝ := 25.1

-- The theorem stating that option C is within the acceptable range
theorem option_C_qualified : lower_bound ≤ option_C ∧ option_C ≤ upper_bound := 
by 
  sorry

end option_C_qualified_l634_634360


namespace sum_of_s_r_values_l634_634595

-- Define the function r with its specific range
def r_domain := [-2, -1, 0, 1]
def r_range := {1, 3, 5, 7}

-- Define the function s with its specific domain and range
def s_domain := {0, 1, 2, 3, 4, 5}
def s (x : ℕ) : ℕ := 2 * x + 1

-- The final theorem to assert the sum of all possible values of s(r(x))
theorem sum_of_s_r_values :
  let valid_inputs := { x | x ∈ r_range ∧ x ∈ s_domain }
  let s_values := valid_inputs.image s
  s_values.sum = 21 :=
by
  sorry

end sum_of_s_r_values_l634_634595


namespace proof_problem_l634_634064

-- Definitions and Hypotheses
def ellipse (a b : ℝ) (a_gt_b : a > b) : Prop :=
  ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

def parabola_focus : ℝ × ℝ := (-1, 0)

def left_focus_at_parabola_focus (a b c : ℝ) (e : ℝ) (E_eccentricity : e = sqrt 2 / 2)
  (E_left_focus : ℝ × ℝ) (H_focus : E_left_focus = parabola_focus) : Prop :=
  c = 1 ∧ e = sqrt 2 / 2

noncomputable def get_eqn_of_ellipse (a b : ℝ) (H_eccentricity: a = sqrt 2) (H_bsq: b^2 = a^2 - 1) : Prop :=
  (a = sqrt 2 ∧ b^2 = 1) → (∀ x y : ℝ, (x^2 / 2 + y^2 = 1))

def line_intersects_ellipse (m : ℝ) (m_gt_limit: m > 3/4) : Prop :=
  ∃ (t : ℝ) (x1 y1 x2 y2 : ℝ),
    (x1 = t * y1 + m) ∧ (x2 = t * y2 + m) ∧ (y1 + y2 = -2 * t * m / (t^2 + 2)) ∧ (y1 * y2 = (m^2 - 2) / (t^2 + 2))

def pa_pb_constant (m : ℝ) (P : ℝ × ℝ) (H_point : P = (5/4, 0)) : Prop :=
  m = 1 ∧ P = (5/4, 0) →

-- Main Theorem
theorem proof_problem :
  ∃ a b c : ℝ, 
  ellipse a b (by norm_num1: sqrt 2 > 1) ∧ 
  (left_focus_at_parabola_focus a b c (sqrt 2/2) (parabola_focus) (by norm_num)) →
  (get_eqn_of_ellipse a b (by norm_num) (by norm_num)) ∧
  (line_intersects_ellipse 1 (by norm_num)) →
  (∃ t : ℝ, (sqrt 2 * (sqrt (t^2 + 1) / (t^2 + 2))) ≤ (sqrt 2 / 2))
:= by sorry

end proof_problem_l634_634064


namespace hamburger_price_l634_634046

theorem hamburger_price (P : ℝ) 
    (h1 : 2 * 4 + 2 * 2 = 12) 
    (h2 : 12 * P + 4 * P = 50) : 
    P = 3.125 := 
by
  -- sorry added to skip the proof.
  sorry

end hamburger_price_l634_634046


namespace find_slope_l634_634821

open Real

-- Given ellipse equation and line slope condition
def ellipse_eq (x y : ℝ) : Prop := x^2 + 2 * y^2 = 3

-- Definition of a line passing through the focus of the ellipse
def line_eq (k x y : ℝ) : Prop := y = k * x

-- Given the intersection points A and B such that distance AB = 2
def distance_AB_eq_two (A B : ℝ × ℝ) : Prop := 
  (euclidean_dist A B = 2)

-- Main theorem to prove |k|
theorem find_slope (k : ℝ) (A B : ℝ × ℝ) :
  (∃ A B : ℝ × ℝ, ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2 ∧ line_eq k A.1 A.2 ∧ line_eq k B.1 B.2 ∧ distance_AB_eq_two A B) → 
  |k| = sqrt (1 + sqrt 3) :=
by
  sorry

end find_slope_l634_634821


namespace solve_system_l634_634653

theorem solve_system :
  ∃ (x y : ℝ), 2 * x + y = 4 ∧ x - y = -1 ∧ x = 1 ∧ y = 2 := 
by {
  use (1 : ℝ),
  use (2 : ℝ),
  split, 
  { sorry }, -- proof that 2 * 1 + 2 = 4
  { split,
    { sorry }, -- proof that 1 - 2 = -1
    { split,
      { refl }, -- proof that x = 1
      { refl } -- proof that y = 2
    }
  }
}

end solve_system_l634_634653


namespace parabola_slope_max_k_minus_m_l634_634350

open Set

noncomputable def max_k_minus_m (m k : ℝ) : ℝ := k - m

theorem parabola_slope_max_k_minus_m
  (m : ℝ)
  (M_on_parabola : ∃ y, y^2 = m ∧ m > 0)  -- M is on the parabola y^2 = x and in the first quadrant
  (compl_intersection : ∃ A B : ℝ × ℝ, 
    A.1^2 = A.2 ∧ B.1^2 = B.2 ∧ 
    (∃ t : ℝ, A.2 - m^2 = t * (A.1 - m) ∧ B.2 - m^2 = -t * (B.1 - m)) ∧
    A ≠ B) -- Lines intersect the parabola at two points A and B with the given properties
  (k : ℝ)
  (k_def : k = (let A_y := some (exists_of_unique_of_exists (λ y, y^2 = A.1) _), 
                    B_y := some (exists_of_unique_of_exists (λ y, y^2 = B.1) _) 
                in (B_y - A_y) / (B_y^2 - A_y^2 - M_on_parabola.some))) 
  : max_k_minus_m m k ≤ - √2 := sorry

end parabola_slope_max_k_minus_m_l634_634350


namespace thompson_class_average_l634_634972

theorem thompson_class_average
  (n : ℕ) (initial_avg : ℚ) (final_avg : ℚ) (bridget_index : ℕ) (first_n_score_sum : ℚ)
  (total_students : ℕ) (final_score_sum : ℚ)
  (h1 : n = 17) -- Number of students initially graded
  (h2 : initial_avg = 76) -- Average score of the first 17 students
  (h3 : final_avg = 78) -- Average score after adding Bridget's test
  (h4 : bridget_index = 18) -- Total number of students
  (h5 : total_students = 18) -- Total number of students
  (h6 : first_n_score_sum = n * initial_avg) -- Total score of the first 17 students
  (h7 : final_score_sum = total_students * final_avg) -- Total score of the 18 students):
  -- Bridget's score
  (bridgets_score : ℚ) :
  bridgets_score = final_score_sum - first_n_score_sum :=
sorry

end thompson_class_average_l634_634972


namespace diameter_of_circle_C_l634_634376

noncomputable def diameter_C (r_C : ℝ) : ℝ := 2 * r_C

theorem diameter_of_circle_C :
  (∀ (D C : ℝ), D = 10 → (∀ A E : ℝ, A = (100 * π - E) / 2 → A = E → diameter_C (sqrt (100 / 3)) = 20 * real.sqrt 3 / 3)) := 
by
  intros D C hD A E hA hE
  rw [diameter_C]
  field_simp [hD, hA, hE]
  sorry

end diameter_of_circle_C_l634_634376


namespace david_airport_distance_l634_634769

noncomputable def distance_to_airport (t d : ℝ) : Prop :=
  40 * (t + 1.5) = d ∧
  40 + 60 * (t - 2) = d

theorem david_airport_distance : ∃ d, distance_to_airport 9 d ∧ d = 420 :=
by
  use 420
  have h1 : 40 * (9 + 1.5) = 420 := by norm_num
  have h2 : 40 + 60 * (9 - 2) = 420 := by norm_num
  exact ⟨⟨h1, h2⟩, rfl⟩

end david_airport_distance_l634_634769


namespace quad_cyclic_and_tangent_l634_634543

open Real Set

variables {A B C D : Point}
variables {AB AD CB CD : ℝ} -- lengths of sides
variables {r R : ℝ} -- radii of the incircle and circumcircle

-- Definition that AB = AD, CB = CD
def lengths_eq (AB AD CB CD : ℝ) : Prop :=
  AB = AD ∧ CB = CD

-- Condition AB ⊥ BC
def orthogonal (AB BC : ℝ) : Prop :=
  AB * BC = 0 -- Perpendicular vectors have dot product zero

-- The final statement combining all parts
theorem quad_cyclic_and_tangent (AB AD CB CD r R : ℝ) (AB_perp_BC : AB * BC = 0) :
  lengths_eq AB AD CB CD →
  cyclic_quadrilateral A B C D ∧
  externally_tangent_to_circle A B C D →
  distance_between_centers_of_incircle_and_excircle A B C D r R =
  R^2 + r^2 - r * sqrt(4 * R^2 + r^2) :=
by
  sorry

end quad_cyclic_and_tangent_l634_634543


namespace triangle_inequality_l634_634557

def sin_half_angle (θ : ℝ) : ℝ := Real.sin (θ / 2)

theorem triangle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  sin_half_angle A * sin_half_angle B * sin_half_angle C ≤ 1 / 8 :=
by
  sorry

end triangle_inequality_l634_634557


namespace arithmetic_geometric_problem_l634_634820

-- Definitions based on the conditions given in the problem

def is_arithmetic_sequence (a₁ a₂ : ℤ) (d : ℤ) : Prop :=
  a₁ = -1 + d ∧ a₂ = a₁ + d ∧ -9 = a₂ + d

def is_geometric_sequence (b₁ b₂ b₃ : ℝ) (q : ℝ) : Prop :=
  b₁ = -9 * q ∧ b₂ = b₁ * q ∧ b₃ = b₂ * q ∧ -1 = b₃ * q

theorem arithmetic_geometric_problem
  (a₁ a₂ b₁ b₂ b₃ : ℝ) (d q : ℝ)
  (arithmetic_cond : is_arithmetic_sequence a₁ a₂ d)
  (geometric_cond : is_geometric_sequence b₁ b₂ b₃ q) :
  b₂ * (a₂ - a₁) = 8 := by
sorry

end arithmetic_geometric_problem_l634_634820


namespace likes_basketball_or_cricket_or_both_l634_634915

theorem likes_basketball_or_cricket_or_both
    (students_basketball : ℕ)
    (students_cricket : ℕ)
    (students_both : ℕ) :
    students_basketball = 10 →
    students_cricket = 8 →
    students_both = 4 →
    students_basketball + students_cricket - students_both = 14 := by
  intros hB hC hBC
  rw [hB, hC, hBC]
  norm_num
  sorry

end likes_basketball_or_cricket_or_both_l634_634915


namespace lines_intersect_first_quadrant_l634_634794

theorem lines_intersect_first_quadrant (k : ℝ) :
  (∃ (x y : ℝ), 2 * x + 7 * y = 14 ∧ k * x - y = k + 1 ∧ x > 0 ∧ y > 0) ↔ k > 0 :=
by
  sorry

end lines_intersect_first_quadrant_l634_634794


namespace inequality_equivalence_l634_634297

theorem inequality_equivalence (a b : ℝ) :
  a^2 + b^2 - 1 - a^2 * b^2 ≤ 0 ↔ (a^2 - 1) * (b^2 - 1) ≥ 0 := 
begin
  sorry
end

end inequality_equivalence_l634_634297


namespace probability_of_B_l634_634195

variable {Ω : Type} -- The sample space
variable {P : Ω → Prop → ℝ} -- Probability function
variables {A B : Ω → Prop} -- Events

noncomputable def complement (A : Ω → Prop) : Ω → Prop := λ ω, ¬(A ω)

theorem probability_of_B (hP_A : P A = 1/2) 
                         (hP_B_given_A : P (λ ω, B ω | A ω) = 1/4)
                         (hP_complement_B_given_complement_A : P (λ ω, complement B ω | complement A ω) = 2/3) :
  P B = 7/24 := sorry

end probability_of_B_l634_634195


namespace prism_coloring_1995_prism_coloring_1996_l634_634559

def prism_coloring_possible (n : ℕ) : Prop :=
  ∃ (color : ℕ → ℕ),
    (∀ i, 1 ≤ color i ∧ color i ≤ 3) ∧ -- Each color is within bounds
    (∀ i, color i ≠ color ((i + 1) % n)) ∧ -- Colors on each face must be different
    (n % 3 = 0 ∨ n ≠ 1996) -- Condition for coloring

theorem prism_coloring_1995 : prism_coloring_possible 1995 :=
sorry

theorem prism_coloring_1996 : ¬prism_coloring_possible 1996 :=
sorry

end prism_coloring_1995_prism_coloring_1996_l634_634559


namespace team_formation_l634_634342

theorem team_formation (boys girls : ℕ) (team_size : ℕ) :
  boys = 4 → girls = 5 → team_size = 5 → 
  (∃ g : ℕ, 1 ≤ g ∧ g < team_size - g ∧ g ≤ girls ∧ team_size - g ≤ boys) →
  (∑ g in finset.range (girls + 1), if 1 ≤ g ∧ g < team_size - g then nat.choose boys (team_size - g) * nat.choose girls g else 0) = 45 :=
by 
  intros hb hg ht hex;
  rw [hb, hg, ht];
  sorry

end team_formation_l634_634342


namespace cube_plane_intersection_area_l634_634339

theorem cube_plane_intersection_area :
  ∀ (s : ℝ), s = 2 → 
  ∃ (A : ℝ), A = 4 ∧ (∀ (cube : set (ℝ × ℝ × ℝ)), 
    (∃ (plane : set (ℝ × ℝ × ℝ)), 
    plane.parallel (face1 : set (ℝ × ℝ × ℝ)) (face2 : set (ℝ × ℝ × ℝ)) ∧ 
    plane.halfway_between (face1 : set (ℝ × ℝ × ℝ)) (face2 : set (ℝ × ℝ × ℝ)) ∧
    (plane ∩ cube) = (square : set (ℝ × ℝ × ℝ)) ∧
  square.area = 4)) sorry

end cube_plane_intersection_area_l634_634339


namespace initial_money_eq_20_l634_634565

-- Define the conditions
variable {M : ℝ}

-- Condition 1: John spends (1/5) of his money on snacks
def spent_on_snacks (M : ℝ) : ℝ := M / 5

-- Condition 2: John spends (3/4) of the remaining money on necessities
def remaining_after_snacks (M : ℝ) : ℝ := 4 * M / 5
def spent_on_necessities (remaining_money : ℝ) : ℝ := 3 * remaining_money / 4
def remaining_after_necessities (remaining_money : ℝ) : ℝ := remaining_money / 4

-- Condition 3: John has $4 left after these expenses
def final_remaining_money : ℝ := 4

-- The main theorem to prove the initial amount of money M
theorem initial_money_eq_20 (M : ℝ) 
  (snacks: spent_on_snacks M) 
  (necessities: spent_on_necessities (remaining_after_snacks M))
  (final_remaining: remaining_after_necessities (remaining_after_snacks M) = final_remaining_money) : 
  M = 20 :=
by
  sorry

end initial_money_eq_20_l634_634565


namespace minimum_value_8_l634_634418

noncomputable def minimum_value (x : ℝ) : ℝ :=
  3 * x + 2 / x^5 + 3 / x

theorem minimum_value_8 (x : ℝ) (hx : x > 0) :
  ∃ y : ℝ, (∀ z > 0, minimum_value z ≥ y) ∧ (y = 8) :=
by
  sorry

end minimum_value_8_l634_634418


namespace area_of_right_triangle_with_given_sides_l634_634889

theorem area_of_right_triangle_with_given_sides :
  let f : (ℝ → ℝ) := fun x => x^2 - 7 * x + 12
  let a := 3
  let b := 4
  let c := sqrt 7
  let hypotenuse := max a b
  let leg := min a b
in (hypotenuse = 4 ∧ leg = 3 ∧ (f(3) = 0) ∧ (f(4) = 0) → 
   (∃ (area : ℝ), (area = 6 ∨ area = (3 * sqrt 7) / 2))) :=
by
  intros
  sorry

end area_of_right_triangle_with_given_sides_l634_634889


namespace trapezium_distance_parallel_sides_l634_634783

theorem trapezium_distance_parallel_sides (a b A : ℝ) (h : ℝ) (h1 : a = 20) (h2 : b = 18) (h3 : A = 380) :
  A = (1 / 2) * (a + b) * h → h = 20 :=
by
  intro h4
  rw [h1, h2, h3] at h4
  sorry

end trapezium_distance_parallel_sides_l634_634783


namespace problem_statement_l634_634600

noncomputable def f : ℝ → ℝ := sorry

variable (a b : ℝ)

def odd_property (f : ℝ → ℝ) := ∀ x : ℝ, f(x + 1) = -f(-(x + 1))
def even_property (f : ℝ → ℝ) := ∀ x : ℝ, f(x + 2) = f(-(x + 2))
def parabola_in_domain (f : ℝ → ℝ) (a b : ℝ) := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f(x) = a * x^2 + b
def specific_values (f : ℝ → ℝ) := f(0) + f(3) = 12

theorem problem_statement : 
  (odd_property f) →
  (even_property f) →
  (parabola_in_domain f a b) →
  (specific_values f) →
  f (17 / 2) = 5 :=
by
  intros h_odd h_even h_parabola h_specific
  sorry

end problem_statement_l634_634600


namespace max_sales_revenue_l634_634396

def f (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 40 then (1 / 4) * t + 10
  else if 40 < t ∧ t ≤ 90 then t - 20
  else 0

def g (t : ℕ) : ℝ :=
  if 1 ≤ t ∧ t ≤ 40 then -10 * t + 630
  else if 40 < t ∧ t ≤ 90 then (-1 / 10) * (t^2) + 10 * t - 10
  else 0

def S (t : ℕ) : ℝ := f t * g t

theorem max_sales_revenue : ∃ t ∈ {1, 2, ..., 90}, S t = 53045 / 8 :=
sorry

end max_sales_revenue_l634_634396


namespace math_problem_l634_634311

theorem math_problem :
  sqrt 5 * 5 ^ (1 / 3) + 15 / 5 * 3 - 9 ^ (5 / 2) = 5 ^ (5 / 6) - 234 := by
  sorry

end math_problem_l634_634311


namespace triangle_geometry_l634_634151

open EuclideanGeometry

variable {α : Type*} [MetricSpace α] [NormedGroup α] [NormedSpace ℝ α] [OrderedEuclideanGeometry α]

theorem triangle_geometry (A B C D E F G : α) :
  is_triangle_geometry A B C D E F G → 
  external_bisector_intersects_ray A B C D → 
  perpendicular_foot B A D E → 
  perpendicular_foot C A D F → 
  perpendicular_foot D A C G → 
  ∠DGE + ∠DGF = 180 :=
by
  sorry

end triangle_geometry_l634_634151


namespace sarah_copies_pages_l634_634239

theorem sarah_copies_pages : 
  ∀ (copies_per_person : ℕ) (num_people : ℕ) (pages_per_contract : ℕ), 
  copies_per_person = 5 → 
  num_people = 15 → 
  pages_per_contract = 35 → 
  copies_per_person * num_people * pages_per_contract = 2625 :=
by
  intros copies_per_person num_people pages_per_contract h1 h2 h3
  rw [h1, h2, h3]
  exact (by norm_num : 5 * 15 * 35 = 2625)

end sarah_copies_pages_l634_634239


namespace problem_l634_634169

theorem problem (t θ : ℝ)
  (x := t) (y := -1 + sqrt 3 * t)
  (ρ := 2 * sin θ + 2 * cos θ)
  (P := (0, -1)) :

  (sqrt 3 * x - y - 1 = 0) ∧
  (x^2 + y^2 - 2 * x - 2 * y = 0) ∧
  ∃ (A B : ℝ × ℝ), l ∩ C = {A, B} ∧
  (dist P A ≠ 0 ∧ dist P B ≠ 0) ∧
  (1 / dist P A + 1 / dist P B = (2 * sqrt 3 + 1) / 3) := sorry

end problem_l634_634169


namespace coefficient_x2_in_binomial_expansion_l634_634140

theorem coefficient_x2_in_binomial_expansion (n : ℕ) :
  (∑ i in Finset.range (n+1), binomial_coeff n i * 3^(n-i) * (-1)^i = 16) →
  (∃ k, (4 - 2*k = 2 ∧ binomial_coeff 4 k * 3^(4-k) * (-1)^k = -108)) :=
by
  sorry

end coefficient_x2_in_binomial_expansion_l634_634140


namespace point_of_tangency_l634_634030

-- Definitions for the given parabola equations
def parabola_1 (x : ℝ) : ℝ := x^2 + 20 * x + 72
def parabola_2 (y : ℝ) : ℝ := y^2 + 64 * y + 992

-- The proof statement
theorem point_of_tangency :
  ∃ (x y : ℝ), y = parabola_1 x ∧ x = parabola_2 y ∧ x = -9.5 ∧ y = -31.5 :=
by
  use -9.5, -31.5
  split
  · sorry -- y = parabola_1 x
  split
  · sorry -- x = parabola_2 y
  · tauto -- For x = -9.5 and y = -31.5

end point_of_tangency_l634_634030


namespace error_in_major_premise_l634_634097

-- Conditions
def f (x : ℝ) (α : ℝ) : ℝ := x^α
def y (x : ℝ) : ℝ := x^(-1)

-- Statement
theorem error_in_major_premise (α : ℝ) (x : ℝ) (h₁ : ∀ (x : ℝ), x > 0 → f x α > f (x - 1) α)
    (h₂ : ∀ (x : ℝ), x > 0 → y x > y (x - 1) → α > 0) :
    ¬(∀ α, ∀ (x : ℝ), x > 0 → f x α > f (x - 1) α) :=
sorry

end error_in_major_premise_l634_634097


namespace find_m_l634_634454

noncomputable def m : ℝ := 495 / 104

theorem find_m
  (x : ℝ)
  (h1 : log 10 (sin x) + log 10 (cos x) = -2)
  (h2 : log 10 (sin x + 2 * cos x) = (1 / 2) * (log 10 m - 2))
  : m = 475 / 100 :=
sorry

end find_m_l634_634454


namespace subset_divides_l634_634567

-- Define the set S
def S (n : ℕ) : set ℕ := {x | 1 ≤ x ∧ x ≤ n}

-- Define the subset A of S
def A (n : ℕ) := {x | x ∈ S n}

-- Define the condition on A
def condition (n : ℕ) (A : set ℕ) : Prop :=
  ∀ x y ∈ A, (x + y ∈ A) ∨ (x + y - n ∈ A)

-- State the theorem: The size of A divides n
theorem subset_divides (n : ℕ) (A : set ℕ) (hA: A ⊆ S n) (hC : condition n A) :
  ∃ k : ℕ, fintype.card (↥A) * k = n :=
sorry

end subset_divides_l634_634567


namespace rectangle_area_equals_circle_area_l634_634055

-- Definitions of points P, Q, A, B
structure Point :=
(x : ℝ)
(y : ℝ)

def O : Point := ⟨0, 0⟩
def A : Point := O 
def P : Point := O 

-- Given radius
def R : ℝ := 4

-- Definition of distance between points
def distance (p1 p2 : Point) : ℝ :=
real.sqrt ((p2.x - p1.x) ^ 2 + (p2.y - p1.y) ^ 2)

-- Define point B after rolling the circle one circumference
def B : Point := ⟨2 * real.pi * R, 0⟩

-- Define point Q vertically above B by radius R
def Q : Point := ⟨B.x, R⟩

-- Proving that the area of rectangle equals the area of the circle
theorem rectangle_area_equals_circle_area :
  let AB := distance A B,
      BQ := distance B Q,
      circle_area := real.pi * R ^ 2,
      rectangle_area := AB * BQ in
  rectangle_area = circle_area := by
  sorry

end rectangle_area_equals_circle_area_l634_634055


namespace length_AB_max_major_axis_length_l634_634848

-- Given conditions for the first question
def line1 : (ℝ × ℝ) → Prop := λ P, P.snd = -P.fst + 1
def ellipse1 (a b : ℝ) : (ℝ × ℝ) → Prop := λ P, P.fst^2 / a^2 + P.snd^2 / b^2 = 1
def eccentricity1 := sqrt 3 / 3
def focal_length1 := 1
def a1 := sqrt 3
def b1 := sqrt (3 - 1)

-- Length of segment AB
theorem length_AB : 
  ∃ A B : ℝ × ℝ, 
    line1 A ∧ ellipse1 a1 b1 A ∧ 
    line1 B ∧ ellipse1 a1 b1 B ∧ 
    dist A B = 8 * sqrt 3 / 5 := 
sorry

-- Given conditions for the second question
def ellipse2 (a b : ℝ) : (ℝ × ℝ) → Prop := λ P, P.fst^2 / a^2 + P.snd^2 / b^2 = 1
def perpendicular (A B : ℝ × ℝ) : Prop := A.fst * B.fst + A.snd * B.snd = 0
def e_min := 1/2
def e_max := sqrt 2 / 2
def max_length_major_axis := sqrt 6

-- Maximum length of major axis
theorem max_major_axis_length : 
  ∃ a : ℝ, 
    ∀ e ∈ set.Icc e_min e_max, 
    let b := sqrt (a^2 - (a * e)^2) 
    in (a > b ∧ a^2 + b^2 > 1) → 
    2 * a = max_length_major_axis := 
sorry

end length_AB_max_major_axis_length_l634_634848


namespace x_intercepts_of_parabola_l634_634112

theorem x_intercepts_of_parabola : 
  (∃ y : ℝ, x = -3 * y^2 + 2 * y + 2) → ∃ y : ℝ, y = 0 ∧ x = 2 ∧ ∀ y' ≠ 0, x ≠ -3 * y'^2 + 2 * y' + 2 :=
by
  sorry

end x_intercepts_of_parabola_l634_634112


namespace not_even_nor_odd_l634_634602

def f (x : ℝ) : ℝ := x^2

theorem not_even_nor_odd (x : ℝ) (h₁ : -1 < x) (h₂ : x ≤ 1) : ¬(∀ y, f y = f (-y)) ∧ ¬(∀ y, f y = -f (-y)) :=
by
  sorry

end not_even_nor_odd_l634_634602


namespace proof_problem_l634_634868

-- Definition of the digit count function in a given base
def digit_count (n : ℕ) (b : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log n / Nat.log b + 1

-- Definition for the given problem
def problem : Prop :=
  let n := 1296
  let base4_digits := digit_count n 4
  let base9_digits := digit_count n 9
  base4_digits - base9_digits = 2

-- The lean proof statement
theorem proof_problem : problem :=
  by
    sorry

end proof_problem_l634_634868


namespace tangent_length_from_origin_to_circle_l634_634762

variable (A B C O : Point)
variable (tangent_length : ℝ)

-- Definitions for points A, B, C and O
def A : Point := ⟨4, 5⟩
def B : Point := ⟨7, 9⟩
def C : Point := ⟨8, 6⟩
def O : Point := ⟨0, 0⟩

-- Definition for calculating distance
def euclidean_distance (p1 p2 : Point) : ℝ := 
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Proposition: The length of the tangent from the origin to the circle passing through points A, B, and C is sqrt(5330)
theorem tangent_length_from_origin_to_circle : 
  euclidean_distance A O * euclidean_distance B O = 5330 → 
  tangent_length = sqrt(5330) := by 
  sorry

end tangent_length_from_origin_to_circle_l634_634762


namespace range_of_x_for_obtuse_triangle_l634_634450

theorem range_of_x_for_obtuse_triangle :
  ∀ (x : ℝ), 1 < x ∧ x < 3 → 
  let a := 4 - x, b := 5 - x, c := 6 - x in
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ (a^2 + b^2 < c^2) :=
by intros; sorry

end range_of_x_for_obtuse_triangle_l634_634450


namespace joan_friends_kittens_l634_634183

theorem joan_friends_kittens (initial_kittens final_kittens friends_kittens : ℕ) 
  (h1 : initial_kittens = 8) 
  (h2 : final_kittens = 10) 
  (h3 : friends_kittens = 2) : 
  final_kittens - initial_kittens = friends_kittens := 
by 
  -- Sorry is used here as a placeholder to indicate where the proof would go.
  sorry

end joan_friends_kittens_l634_634183


namespace parabola_has_one_x_intercept_l634_634116

-- Define the equation of the parabola
def parabola (y : ℝ) : ℝ :=
  -3 * y ^ 2 + 2 * y + 2

-- The theorem statement asserting there is exactly one x-intercept
theorem parabola_has_one_x_intercept : ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 := by
  sorry

end parabola_has_one_x_intercept_l634_634116


namespace correct_relationship_l634_634513

variable {f : ℝ → ℝ}

-- Assume f is an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = f(x)

-- Assume f is decreasing on [1, +∞)
def decreasing_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x < y → f(y) ≤ f(x)

theorem correct_relationship (h_even : even_function f) (h_decreasing : decreasing_on_interval f) : 
  f(2) < f(-3/2) ∧ f(-3/2) < f(-1) :=
by
  sorry

end correct_relationship_l634_634513


namespace sum_of_areas_l634_634748

-- Given: Angle FAC is a right angle, CF = 12 units
-- To Prove: The sum of the areas of the squares ACDE and AFGH is 144 square units

theorem sum_of_areas (A C F : Type) [AddGroup F] [MetricSpace F] [NormedAddTorsor F ℝ]
  (angle_FAC_right : ∡(F, A, C) = π / 2) (CF_eq_12 : dist F C = 12) :
  let ACDE_area := (dist A C) ^ 2
  let AFGH_area := (dist A F) ^ 2
  ACDE_area + AFGH_area = 144 := by
  sorry

end sum_of_areas_l634_634748


namespace Marly_minimum_bills_l634_634214

theorem Marly_minimum_bills :
  ∃ (n100 n50 n20 n5 n2 : ℕ), 
    (n100 = 18) ∧ (n50 = 3) ∧ (n20 = 4) ∧ (n5 = 1) ∧ (n2 = 0) ∧
    (n100 * 100 + n50 * 50 + n20 * 20 + n5 * 5 + n2 * 2 = 3000) ∧
    (n100 >= 3) ∧ (n50 >= 2) ∧ (n20 ≤ 4) :=
begin
  use [18, 3, 4, 1, 0],
  split, refl,
  split, refl,
  split, refl,
  split, refl,
  split, refl,
  split,
  { calc 18 * 100 + 3 * 50 + 4 * 20 + 1 * 5 + 0 * 2 = 3000 : by norm_num },
  split, exact dec_trivial,
  split, exact dec_trivial,
  exact dec_trivial
end

end Marly_minimum_bills_l634_634214


namespace product_is_zero_l634_634010

theorem product_is_zero (b : ℤ) (h : b = 3) :
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b * (b + 1) * (b + 2) = 0 :=
by {
  -- Substituting b = 3
  -- (3-5) * (3-4) * (3-3) * (3-2) * (3-1) * 3 * (3+1) * (3+2)
  -- = (-2) * (-1) * 0 * 1 * 2 * 3 * 4 * 5
  -- = 0
  sorry
}

end product_is_zero_l634_634010


namespace intersection_is_M_l634_634099

def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x + 1}
def intersection : Set ℝ := M ∩ N

theorem intersection_is_M : intersection = {y : ℝ | y ≥ 1} :=
by
  -- Proof goes here
  sorry

end intersection_is_M_l634_634099


namespace multiples_of_7_in_q_l634_634693

theorem multiples_of_7_in_q (a b : ℤ) (q : set ℤ) 
  (h1 : a % 14 = 0) (h2 : b % 14 = 0) 
  (h3 : q = { x | a ≤ x ∧ x ≤ b })
  (h4 : ∃ s : finset ℤ, (∀ x ∈ s, x % 14 = 0) ∧ s.card = 14 ∧ ∀ x ∈ s, x ∈ q) : 
  ∃ t : finset ℤ, (∀ x ∈ t, x % 7 = 0) ∧ t.card = 27 ∧ ∀ x ∈ t, x ∈ q :=
by
  sorry

end multiples_of_7_in_q_l634_634693


namespace angle_neg_a_c_l634_634501

def AngleBetweenVectors (u v : Vector ℝ) : ℝ := sorry -- assumes definition of angle between vectors

-- Condition 1: The angle between the vectors a and b is 60 degrees
def angle_ab (a b : Vector ℝ) : Prop := AngleBetweenVectors a b = 60

-- Condition 2: The angle between the vectors b and c is 30 degrees
def angle_bc (b c : Vector ℝ) : Prop := AngleBetweenVectors b c = 30

-- Theorem to prove: The angle between the vectors -a and c is 210 degrees
theorem angle_neg_a_c {a b c : Vector ℝ} (h1 : angle_ab a b) (h2 : angle_bc b c) : 
  AngleBetweenVectors (-a) c = 210 :=
sorry

end angle_neg_a_c_l634_634501


namespace trigonometric_identity_l634_634792

theorem trigonometric_identity :
  (∀ θ : ℝ, θ = 12 → (√3 * tan (θ / 180 * π) - 3) * csc (θ / 180 * π) / (4 * cos (θ / 180 * π) ^ 2 - 2) = -4 * √3 ) :=
by
  intro θ hθ
  rw [hθ]
  sorry

end trigonometric_identity_l634_634792


namespace vince_customers_per_month_l634_634303

theorem vince_customers_per_month (C : ℕ) (H : 18 * C - 280 - 3.6 * C = 872) : C = 80 :=
sorry

end vince_customers_per_month_l634_634303


namespace evaluate_expression_l634_634402

noncomputable def w : ℂ := complex.exp (2 * real.pi * complex.I / 13)

theorem evaluate_expression :
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * 
  (3 - w^6) * (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * 
  (3 - w^11) * (3 - w^12) = 797161 :=
begin
  sorry
end

end evaluate_expression_l634_634402


namespace find_DC_l634_634544

variable (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
variable (distance : A → B → ℝ)
variable (AB BD BC DC : ℝ)
variable (angle_A : ℝ)
variable (sin_A sin_C : ℝ)

noncomputable def AB := 30
noncomputable def angle_A := 90
noncomputable def sin_A := 2 / 3
noncomputable def sin_C := 1 / 4

axiom triangle_ABD : angle_A = 90 ∧ sin_A = distance B D / AB

axiom triangle_BCD : sin_C = distance B D / BC

theorem find_DC :
  angle_A = 90 ∧ sin_A = 2 / 3 ∧ sin_C = 1 / 4 ∧ AB = 30 → distance D C = 20 * sqrt 15 :=
by
  intros h
  sorry

end find_DC_l634_634544


namespace max_two_terms_eq_one_l634_634628

theorem max_two_terms_eq_one (a b c x y z : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : x ≠ z) :
  ∀ (P : ℕ → ℝ), -- Define P(i) as given expressions
  ((P 1 = a * x + b * y + c * z) ∧
   (P 2 = a * x + b * z + c * y) ∧
   (P 3 = a * y + b * x + c * z) ∧
   (P 4 = a * y + b * z + c * x) ∧
   (P 5 = a * z + b * x + c * y) ∧
   (P 6 = a * z + b * y + c * x)) →
  (P 1 = 1 ∨ P 2 = 1 ∨ P 3 = 1 ∨ P 4 = 1 ∨ P 5 = 1 ∨ P 6 = 1) →
  (∃ i j, i ≠ j ∧ P i = 1 ∧ P j = 1) →
  ¬(∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ P i = 1 ∧ P j = 1 ∧ P k = 1) :=
sorry

end max_two_terms_eq_one_l634_634628


namespace equation_of_line_AB_l634_634391

noncomputable def circle_eq (x y: ℝ) : Prop := (x - 1)^2 + y^2 = 1

def point := (3, 2) : ℝ × ℝ

theorem equation_of_line_AB :
  (x + 2 * y - 3 = 0) 
  ∧ (∃ (A B : ℝ × ℝ), A ≠ B ∧ ∀ (t : ℝ), line_through point t A B ∧ tangent_to_circle A ∧ tangent_to_circle B) :=
sorry

end equation_of_line_AB_l634_634391


namespace condition_for_monotonically_decreasing_l634_634883

theorem condition_for_monotonically_decreasing (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → y ≤ 4 → f x ≥ f y) → a ≥ 8 :=
by
  let f := λ x : ℝ, x^2 - a*x - 3
  sorry

end condition_for_monotonically_decreasing_l634_634883


namespace num_points_distance_from_line_l634_634929

-- Condition: Equation of the circle in polar coordinates
def circle (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

-- Condition: Equation of the line in polar coordinates
def line (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

-- The theorem to prove
theorem num_points_distance_from_line (θ1 θ2 : ℝ) : 
  (∃ ρ1, circle ρ1 θ1 ∧ ∃ ρ2, circle ρ2 θ2 ∧ 
   Real.dist (λ θ, (2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) * Real.cos θ) (λ θ, 2) = 1) = 2 :=
sorry

end num_points_distance_from_line_l634_634929


namespace f_at_100_l634_634944

def Q_non_zero_one : Set ℚ := {q : ℚ | q ≠ 0 ∧ q ≠ 1}

def f (x : ℚ) (h : x ∈ Q_non_zero_one) : ℝ := sorry

theorem f_at_100 (h : (100 : ℚ) ∈ Q_non_zero_one) :
  (∀ x, x ∈ Q_non_zero_one → f x (by assumption) + f (1 - 1 / x) (by assumption) = 2 * Real.log (abs x)) →
  f 100 h = Real.log (100 / 99) :=
sorry

end f_at_100_l634_634944


namespace find_interval_l634_634845

theorem find_interval (f : ℝ → ℝ) (a b : ℝ) 
  (hf_eq : ∀ x, f x = -1/2 * x^2 + 13/2) 
  (hf_min : ∀ x ∈ set.Icc a b, f x ≥ 2 * a) 
  (hf_max : ∀ x ∈ set.Icc a b, f x ≤ 2 * b) :
  a = 1 ∧ b = 3 :=
sorry

end find_interval_l634_634845


namespace petya_wins_probability_l634_634993

def stones_initial : ℕ := 16

def valid_moves : set ℕ := {1, 2, 3, 4}

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def petya_random_choice (move : ℕ) : Prop := move ∈ valid_moves

def computer_optimal_strategy (n : ℕ) : ℕ :=
  (n % 5)

noncomputable def probability_petya_wins : ℚ :=
  (1 / 4) ^ 4

theorem petya_wins_probability :
  probability_petya_wins = 1 / 256 := 
sorry

end petya_wins_probability_l634_634993


namespace find_circle_equation_l634_634535

noncomputable def equation_of_circle
  (C : ℝ → ℝ → Prop)
  (passes_through_A : C 4 1)
  (tangent_at_B : ∀ x y, (C x y) → (x - y - 1 = 0 ↔ x = 2 ∧ y = 1)) : Prop :=
  ∀ x y, C x y ↔ (x - 3)^2 + y^2 = 2

theorem find_circle_equation
  (C : ℝ → ℝ → Prop)
  (hC : equation_of_circle C (C 4 1) 
    (λ x y hCxy, (x - y - 1 = 0 ↔ x = 2 ∧ y = 1))) : 
  equation_of_circle C (C 4 1) 
    (λ x y hCxy, (x - y - 1 = 0 ↔ x = 2 ∧ y = 1)) :=
  by
    intro x y
    sorry

end find_circle_equation_l634_634535


namespace angle_between_vectors_l634_634486

def vectors (a b : ℝ) : Prop :=
  |overrightarrow{a}| = 1 ∧ |overrightarrow{b}| = 4 ∧ (overrightarrow{a} ⋅ overrightarrow{b}) = 2

theorem angle_between_vectors (a b : ℝ) (h : vectors a b) : angle_between a b = π / 3 := sorry

end angle_between_vectors_l634_634486


namespace lateral_area_of_given_cone_l634_634464

noncomputable def lateral_area_cone (r h : ℝ) : ℝ :=
  let l := Real.sqrt (r^2 + h^2)
  (Real.pi * r * l)

theorem lateral_area_of_given_cone :
  lateral_area_cone 3 4 = 15 * Real.pi :=
by
  -- sorry to skip the proof
  sorry

end lateral_area_of_given_cone_l634_634464


namespace cos_A_values_l634_634492

theorem cos_A_values (A : ℝ) : 
  tan A + 2 * (1 / cos A) = 3 → (cos A = (6 + Real.sqrt 6) / 10 ∨ cos A = (6 - Real.sqrt 6) / 10) :=
by
  sorry

end cos_A_values_l634_634492


namespace find_phi_increasing_intervals_l634_634836

open Real

-- Defining the symmetry condition
noncomputable def symmetric_phi (x_sym : ℝ) (k : ℤ) (phi : ℝ): Prop :=
  2 * x_sym + phi = k * π + π / 2

-- Finding the value of phi given the conditions
theorem find_phi (x_sym : ℝ) (phi : ℝ) (k : ℤ) 
  (h_sym: symmetric_phi x_sym k phi) (h_phi_bound : -π < phi ∧ phi < 0)
  (h_xsym: x_sym = π / 8) :
  phi = -3 * π / 4 :=
by
  sorry

-- Defining the function and its increasing intervals
noncomputable def f (x : ℝ) (phi : ℝ) : ℝ := sin (2 * x + phi)

-- Finding the increasing intervals of f on the interval [0, π]
theorem increasing_intervals (phi : ℝ) 
  (h_phi: phi = -3 * π / 4) :
  ∀ x, (0 ≤ x ∧ x ≤ π) → 
    (π / 8 ≤ x ∧ x ≤ 5 * π / 8) :=
by
  sorry

end find_phi_increasing_intervals_l634_634836


namespace symmetry_of_function_l634_634777

noncomputable def f (x : ℝ) : ℝ := 1 / x - x

theorem symmetry_of_function :
  (∀ x : ℝ, x ≠ 0 → f (-x) = -f x) :=
by
  intro x hx
  unfold f
  -- We assume the domain and prove the statement
  have h1 : 1 / -x = - (1 / x), from sorry
  rw [h1, ← neg_add, neg_eq_neg_iff_eq],
  simp [-sub_eq_add_neg]
  sorry -- Prove the final step.

end symmetry_of_function_l634_634777


namespace ticket_price_l634_634974

theorem ticket_price (Olivia_money : ℕ) (Nigel_money : ℕ) (left_money : ℕ) (total_tickets : ℕ)
  (h1 : Olivia_money = 112)
  (h2 : Nigel_money = 139)
  (h3 : left_money = 83)
  (h4 : total_tickets = 6) :
  (Olivia_money + Nigel_money - left_money) / total_tickets = 28 :=
by
  sorry

end ticket_price_l634_634974


namespace lateral_edge_length_of_smaller_pyramid_l634_634801

theorem lateral_edge_length_of_smaller_pyramid
  (regular_pyramid : Prop)
  (lateral_edge_length_original : ℝ)
  (cut_parallel_to_base : Prop)
  (area_of_cross_section : ℝ)
  (h1 : regular_pyramid)
  (h2 : lateral_edge_length_original = 3)
  (h3 : cut_parallel_to_base)
  (h4 : area_of_cross_section = 1 / 9) :
  ∃ (L : ℝ), L = 1 :=
by
  use 1
  sorry

end lateral_edge_length_of_smaller_pyramid_l634_634801


namespace points_concyclic_l634_634571

-- Definitions and assumptions related to the given problem
variables {A B C D E F X Y I : Point}
-- Assumptions and conditions
variable (h_triangle : Triangle A B C)
variable (h_D : FootOfAngleBisector A B C D)
variable (h_E : FootOfAngleBisector B A C E)
variable (h_F : FootOfAngleBisector C A B F)
variable (h_I : Incenter A B C I)
variable (h_X : PerpendicularBisectorIntersects AD BE X)
variable (h_Y : PerpendicularBisectorIntersects AD CF Y)

-- The theorem to prove
theorem points_concyclic : Concylcic A I X Y :=
sorry -- proof will go here

end points_concyclic_l634_634571


namespace solution_set_of_inequality_l634_634075

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x : ℝ, f(x) = f(-x)
axiom derivative_condition : ∀ x : ℝ, deriv f x < f x
axiom symmetry_condition : ∀ x : ℝ, f(x + 1) = f(2 - x)
axiom specific_value : f 2017 = 3

theorem solution_set_of_inequality : {x : ℝ | f x < 3 * real.exp (x - 1)} = {x : ℝ | x > 1} :=
by 
  sorry

end solution_set_of_inequality_l634_634075


namespace range_of_alpha_range_of_x_plus_y_l634_634536

-- Define the Cartesian equation of curve C derived from polar equation
def curve_C : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 2 * p.1 - 3 = 0}

-- Define parametric equations for line l that passes through P(-3, 0)
def line_l (α t : ℝ) : ℝ × ℝ := (-3 + t * Real.cos α, t * Real.sin α)

-- Problem 1: Prove the range of α for which line l intersects curve C
theorem range_of_alpha (α : ℝ) (t : ℝ) (hp : line_l α t ∈ curve_C) :
  α ∈ [0, π / 6] ∪ [5 * π / 6, π) :=
sorry

-- Define the parametric equation for any point M(x, y) on the curve C
def curve_C_parametric (θ : ℝ) : ℝ × ℝ := (1 + 2 * Real.cos θ, 2 * Real.sin θ)

-- Problem 2: Prove the range of values for x + y for any point M on curve C
theorem range_of_x_plus_y (θ : ℝ) (M : ℝ × ℝ) (hM : M ∈ curve_C) :
  1 - 2 * Real.sqrt 2 ≤ M.1 + M.2 ∧ M.1 + M.2 ≤ 1 + 2 * Real.sqrt 2 :=
sorry

end range_of_alpha_range_of_x_plus_y_l634_634536


namespace orthogonal_vector_projection_l634_634196

open LinearAlgebra
open Real

variables {a b : ℝ^2}

-- Proof problem statement
theorem orthogonal_vector_projection (h₁ : dot_product a b = 0)
  (h₂ : linear_proj a ⟨4, -2⟩ = ⟨1 / 2, 1⟩) :
  linear_proj b ⟨4, -2⟩ = ⟨7 / 2, -3⟩ :=
sorry

end orthogonal_vector_projection_l634_634196


namespace prime_div_prime_totient_l634_634202

theorem prime_div_prime_totient (p n : ℕ) [h_prime : Prime p] (h_div : p ∣ n^2020) : p^2020 ∣ n^2020 :=
sorry

end prime_div_prime_totient_l634_634202


namespace rectangle_area_l634_634698

theorem rectangle_area (length diagonal : ℝ) (h_length : length = 16) (h_diagonal : diagonal = 20) : 
  ∃ width : ℝ, (length * width = 192) :=
by 
  sorry

end rectangle_area_l634_634698


namespace geometric_sequence_general_term_and_sum_l634_634467

theorem geometric_sequence_general_term_and_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) 
  (h₁ : ∀ n, a n = 2 ^ n)
  (h₂ : ∀ n, b n = 2 * n - 1)
  : (∀ n, T n = 6 + (2 * n - 3) * 2 ^ (n + 1)) :=
by {
  sorry
}

end geometric_sequence_general_term_and_sum_l634_634467


namespace hyperbola_eccentricity_range_l634_634094

-- Definition of the hyperbola
def hyperbola (x : ℝ) (y : ℝ) (a : ℝ) (b : ℝ) : Prop :=
  (x^2) / (a^2) - (y^2) / (b^2) = 1

-- Definition of the eccentricity
def eccentricity (c a : ℝ) : ℝ :=
  c / a

-- Statement to prove the range of eccentricity
theorem hyperbola_eccentricity_range
  (a b c : ℝ)
  (h_a_pos : 0 < a)
  (h_b_pos : 0 < b)
  (h_c_a : 2 * c < 3 * a)
  (h_one_lt_e : 1 < eccentricity c a)
  (P F1 F2 : ℝ × ℝ × ℝ)
  (ratio_condition : ∃ x₁ x₂ : ℝ, x₁ = 5 * x₂ ∧ x₁ - x₂ = 2 * a ∧ ∃ y₁ y₂ : ℝ, y₁ + y₂ = 3 * a ∧ y₁ - y₂ = 2 * c) :
  1 < eccentricity c a ∧ eccentricity c a < 3 / 2 :=
by
  sorry

end hyperbola_eccentricity_range_l634_634094


namespace log_eq_res_l634_634575

theorem log_eq_res (y m : ℝ) (h₁ : real.log 5 / real.log 8 = y) (h₂ : real.log 125 / real.log 2 = m * y) : m = 9 := 
sorry

end log_eq_res_l634_634575


namespace solve_problem_l634_634815

-- Define the problem conditions and question
variables {a b : ℝ} (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))

-- Define the function f
def f (x : ℝ) : ℝ := a * x ^ 2 + b * x

-- Define the interval condition
def interval_condition : Prop := a - 1 = -2 * a

-- Prove the solution
theorem solve_problem (h_even : ∀ x, f x = f (-x)) (h_interval : interval_condition) : a + b = 1 / 3 :=
by
  sorry

end solve_problem_l634_634815


namespace unit_vectors_have_equal_squares_l634_634072

variables {E : Type*} [inner_product_space ℝ E]
variables (e1 e2 : E)

-- Condition: e1 and e2 are unit vectors
def is_unit_vector (v : E) : Prop := ∥v∥ = 1

-- The proof problem
theorem unit_vectors_have_equal_squares (h1 : is_unit_vector e1) (h2 : is_unit_vector e2) : 
  (inner_product_space.has_inner.inner e1 e1) = (inner_product_space.has_inner.inner e2 e2) :=
by sorry

end unit_vectors_have_equal_squares_l634_634072


namespace function_decreasing_odd_function_m_zero_l634_634474

-- First part: Prove that the function is decreasing
theorem function_decreasing (m : ℝ) (x1 x2 : ℝ) (h : x1 < x2) :
    let f := fun x => -2 * x + m
    f x1 > f x2 :=
by
    sorry

-- Second part: Find the value of m when the function is odd
theorem odd_function_m_zero (m : ℝ) :
    (∀ x : ℝ, let f := fun x => -2 * x + m
              f (-x) = -f x) → m = 0 :=
by
    sorry

end function_decreasing_odd_function_m_zero_l634_634474


namespace unique_T_n_l634_634035

-- Define the arithmetic sequence and associated sums
def arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

def S (a : ℕ) (d : ℕ) (n : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2

def T (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (n * (n + 1) * (3 * a + (n - 1) * d)) / 6

-- Given condition
def S2023 (a : ℕ) (d : ℕ) : ℕ := 2023 * (a + 1011 * d)

-- Proof problem
theorem unique_T_n (a d : ℕ) (S2023_val : S2023 a d) : T a d 6070 = T a d 6070 :=
begin
  sorry
end

end unique_T_n_l634_634035


namespace g_neg_three_l634_634598

def g (x : ℝ) : ℝ :=
  if x < 0 then 3 * x - 4 else x^2 + 1

theorem g_neg_three : g (-3) = -13 :=
by
  sorry

end g_neg_three_l634_634598


namespace mary_water_intake_per_day_l634_634216

def glasses_ml := 250
def glasses_per_day := 6
def ml_per_l := 1000

theorem mary_water_intake_per_day : (glasses_ml * glasses_per_day) / ml_per_l = 1.5 := by
  sorry

end mary_water_intake_per_day_l634_634216


namespace card_pair_probability_sum_l634_634715

theorem card_pair_probability_sum (cards : Finset (ℕ × ℕ)) :
  (∀ n ∈ (Finset.range 1 21), cards.card = 4 * 20) →
  (cards.card = 80) →
  (∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
   ∀ n ∈ {a, b, c}, ∃ (s ∈ cards), s = (n, n) ∧ (cards.erase s).card = 4 * 20 - 2) →
  let remaining_cards := 74 in
  let total_pairs := 102 + 3 in
  let total_ways := 2701 in
  total_pairs / total_ways = 105 / 2701 ∧ 105 + 2701 = 2806 :=
begin
  intros,
  sorry
end

end card_pair_probability_sum_l634_634715


namespace quadrilateral_angle_sum_l634_634192

theorem quadrilateral_angle_sum
  (ABCD : Type*)
  (A B C D E : ABCD)
  (convex : is_convex ABCD)
  (extend_AB : ∃ E, line_through AB B ∧ line_through CD D)
  (anglesum1 : ∠AEB + ∠CED = T)
  (anglesum2 : ∠DAC + ∠BCD = T')
  (T : ℝ)
  (T' : ℝ)
  (angle_sum_q : q = T / T') :
  q > 1 :=
by
  -- Proof can be filled in here
  sorry

end quadrilateral_angle_sum_l634_634192


namespace arithmetic_to_geometric_l634_634814

theorem arithmetic_to_geometric (a1 a2 a3 a4 d : ℝ) 
  (h1 : a2 = a1 + d) (h2 : a3 = a1 + 2 * d) (h3 : a4 = a1 + 3 * d) 
  (h4 : a1 ≠ 0) (h5 : a2 ≠ 0) (h6 : a3 ≠ 0) (h7 : a4 ≠ 0) (h8 : d ≠ 0) 
  (h_geom : (∀ a b c : ℝ, (b^2 = a * c) → 
    ((a = a1 ∧ b = a2 ∧ c = a3) ∨ (a = a1 ∧ b = a2 ∧ c = a4) ∨ 
     (a = a1 ∧ b = a3 ∧ c = a4) ∨ (a = a2 ∧ b = a3 ∧ c = a4)))) : 
  (a1 / d = 1 ∨ a1 / d = -4) :=
begin
  sorry
end

end arithmetic_to_geometric_l634_634814


namespace planted_trees_l634_634175

def plants_per_tree := 20
def trees := 2
def seeds_per_plant := 1
def planting_percentage := 0.6

theorem planted_trees :
  let total_plants := trees * plants_per_tree in
  let total_seeds := total_plants * seeds_per_plant in
  total_plants == 40 ∧ total_seeds == 40 ∧ planting_percentage * total_seeds == 24 → 
  24 := sorry

end planted_trees_l634_634175


namespace area_of_right_triangle_from_roots_l634_634896

theorem area_of_right_triangle_from_roots :
  ∀ (a b : ℝ), (a^2 - 7*a + 12 = 0) → (b^2 - 7*b + 12 = 0) →
  (∃ (area : ℝ), (area = 6) ∨ (area = (3 * real.sqrt 7) / 2)) :=
by
  intros a b ha hb
  sorry

end area_of_right_triangle_from_roots_l634_634896


namespace inradius_inequality_l634_634102

open Real

theorem inradius_inequality (A B C D : Point)
  (AC : Line) (h1 : D ∈ AC)
  (r r1 r2 : ℝ)
  (H : incircle_radius A B C r)
  (H1 : incircle_radius A B D r1)
  (H2 : incircle_radius B C D r2) :
  r1 + r2 > r := sorry

end inradius_inequality_l634_634102


namespace range_of_a_l634_634134

noncomputable def f (x a : ℝ) : ℝ := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x
noncomputable def f' (x a : ℝ) : ℝ := 1 - (2 / 3) * Real.cos (2 * x) + a * Real.cos x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f' x a ≥ 0) ↔ -1 / 3 ≤ a ∧ a ≤ 1 / 3 :=
sorry

end range_of_a_l634_634134


namespace part1_part2_part3_l634_634051

open Real

namespace MathProof

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - 2 * a * x + 5

theorem part1 (h : ∀ x, f a x > 0) : a ∈ set.Ioo (-sqrt(5)) (sqrt(5)) :=
sorry

theorem part2 (h1 : a > 1)
  (h2 : ∀ x ∈ set.Icc 1 a, f a x ∈ set.Icc 1 a) : a = sqrt(2) :=
sorry

def g (a : ℝ) : ℝ :=
if a ≤ 2 then 6 - a^2 else 6 - 2 * a

theorem part3 : ∀ a, ∃ g, ∀ x ∈ set.Icc 1 (a + 1), f a x ≤ g a :=
sorry

end MathProof

end part1_part2_part3_l634_634051


namespace ratio_of_women_to_men_l634_634610

noncomputable def number_of_women (total_guests men initial_children : ℕ) : ℕ :=
total_guests - men - initial_children

theorem ratio_of_women_to_men
    (total_guests men added_children total_children : ℕ)
    (h_total : total_guests = 80)
    (h_men : men = 40)
    (h_total_children : total_children = 30)
    (h_added_children : added_children = 10) :
    (number_of_women total_guests men (total_children - added_children)) / men = 1 / 2 :=
begin
    sorry
end

end ratio_of_women_to_men_l634_634610


namespace emily_widgets_production_l634_634222

variable (w t : ℕ) (work_hours_monday work_hours_tuesday production_monday production_tuesday : ℕ)

theorem emily_widgets_production :
  (w = 2 * t) → 
  (work_hours_monday = t) →
  (work_hours_tuesday = t - 3) →
  (production_monday = w * work_hours_monday) → 
  (production_tuesday = (w + 6) * work_hours_tuesday) →
  (production_monday - production_tuesday) = 18 :=
by
  intros hw hwm hwmt hpm hpt
  sorry

end emily_widgets_production_l634_634222


namespace exists_poly_with_equal_intervals_abs_integral_l634_634706

-- Definitions for conditions
def roots_increasing (a : list ℝ) := ∀ i j, i < j → a.nth i < a.nth j

-- Main theorem statement
theorem exists_poly_with_equal_intervals_abs_integral (n : ℕ) (hn : n ≥ 3) :
  ∃ (f : ℝ → ℝ), degree f = n ∧
  (∃ a : list ℝ, length a = n ∧ roots_increasing a ∧
    ∀ i, i < n - 1 → 
    ∫ x in a.nth i .. a.nth (i + 1), |f x| = 
    ∫ x in a.nth (i + 1) .. a.nth (i + 2), |f x|) :=
by 
  sorry

end exists_poly_with_equal_intervals_abs_integral_l634_634706


namespace johns_weekly_earnings_increase_l634_634939

def combined_percentage_increase (initial final : ℕ) : ℕ :=
  ((final - initial) * 100) / initial

theorem johns_weekly_earnings_increase :
  combined_percentage_increase 40 60 = 50 :=
by
  sorry

end johns_weekly_earnings_increase_l634_634939


namespace sum_ratios_eq_l634_634554

-- Define points A, B, C, D, E, and G as well as their relationships
variables {A B C D E G : Type}

-- Given conditions
axiom BD_2DC : ∀ {BD DC : ℝ}, BD = 2 * DC
axiom AE_3EB : ∀ {AE EB : ℝ}, AE = 3 * EB
axiom AG_2GD : ∀ {AG GD : ℝ}, AG = 2 * GD

-- Mass assumptions for the given problem
noncomputable def mC := 1
noncomputable def mB := 2
noncomputable def mD := mB + 2 * mC  -- mD = B's mass + 2*C's mass
noncomputable def mA := 1
noncomputable def mE := 3 * mA + mB  -- mE = 3A's mass + B's mass
noncomputable def mG := 2 * mA + mD  -- mG = 2A's mass + D's mass

-- Ratios defined according to the problem statement
noncomputable def ratio1 := (1 : ℝ) / mE
noncomputable def ratio2 := mD / mA
noncomputable def ratio3 := mD / mG

-- The Lean theorem to state the problem and correct answer
theorem sum_ratios_eq : ratio1 + ratio2 + ratio3 = (73 / 15 : ℝ) :=
by
  unfold ratio1 ratio2 ratio3
  sorry

end sum_ratios_eq_l634_634554


namespace g_g_even_l634_634591

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

def g : ℝ → ℝ := sorry

theorem g_g_even (h : is_even_function g) : is_even_function (λ x, g (g x)) :=
by
  -- proof omitted
  sorry

end g_g_even_l634_634591


namespace angle_between_lines_l634_634160

structure PointsInSpace (A B C D : Type) :=
(equal_sides : (A.dist_to B) = (B.dist_to C) = (C.dist_to D))
(equal_angles : ∀ α, (A.angle_with B C) = α ∧ (B.angle_with C D) = α ∧ (C.angle_with D A) = α)

theorem angle_between_lines (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  [PointsInSpace A B C D] (α : ℝ) : 
  angle_between (line_through A C) (line_through B D) = α :=
sorry

#print axioms angle_between_lines

end angle_between_lines_l634_634160


namespace integer_solutions_count_l634_634040

theorem integer_solutions_count (x : ℤ) : (x^2 - 8*x + 12 < 0) → (x = 3 ∨ x = 4 ∨ x = 5) :=
by {
  sorry
}

example : ∃ n, n = 3 ∧ ∀ x : ℤ, (x^2 - 8*x + 12 < 0) → (x = 3 ∨ x = 4 ∨ x = 5) :=
by {
  use 3,
  split,
  { refl },
  { assume x h,
    have : x^2 - 8*x + 12 = (x - 2) * (x - 6),
    { ring },
    simp [this] at h,
    linarith },
  sorry
}

end integer_solutions_count_l634_634040


namespace project_completion_plans_l634_634335

theorem project_completion_plans :
  let reciprocal_A := 1 / 12
      reciprocal_B := 1 / 9
  in ((∀ x y : ℕ, (reciprocal_A + reciprocal_B) * x + reciprocal_A * y = 1 ∧ x + y ≤ 8) ∨ 
      (∀ x y : ℕ, (reciprocal_A + reciprocal_B) * x + reciprocal_B * y = 1 ∧ x + y ≤ 8)) → 
  2 := 
by sorry

end project_completion_plans_l634_634335


namespace petya_wins_probability_l634_634994

def stones_initial : ℕ := 16

def valid_moves : set ℕ := {1, 2, 3, 4}

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def petya_random_choice (move : ℕ) : Prop := move ∈ valid_moves

def computer_optimal_strategy (n : ℕ) : ℕ :=
  (n % 5)

noncomputable def probability_petya_wins : ℚ :=
  (1 / 4) ^ 4

theorem petya_wins_probability :
  probability_petya_wins = 1 / 256 := 
sorry

end petya_wins_probability_l634_634994


namespace parametric_graph_intersections_l634_634765

noncomputable def cos (t : ℝ) : ℝ := sorry
noncomputable def sin (t : ℝ) : ℝ := sorry

def parametric_x (t : ℝ) : ℝ := cos(t) + t / 3
def parametric_y (t : ℝ) : ℝ := sin(t) + t / 6

theorem parametric_graph_intersections :
  ∃ n : ℕ, n = 39 ∧ ∀ x : ℝ, (0 ≤ x ∧ x ≤ 80) → number_of_intersections(parametric_x, parametric_y, x) = n :=
sorry

end parametric_graph_intersections_l634_634765


namespace not_decreasing_intervals_of_inverse_function_l634_634088

theorem not_decreasing_intervals_of_inverse_function :
  ∀ (x : ℝ), (x ∈ (-∞, 0) ∪ (0, +∞)) → ¬ decreasing_on (λ x, 3 / x) (-∞, 0) ∪ (0, +∞) :=
by
  sorry

end not_decreasing_intervals_of_inverse_function_l634_634088


namespace max_value_of_trig_expression_l634_634023

open Real

theorem max_value_of_trig_expression : ∀ x : ℝ, 3 * cos x + 4 * sin x ≤ 5 :=
sorry

end max_value_of_trig_expression_l634_634023


namespace arithmetic_geometric_condition_l634_634380

-- Define the arithmetic sequence
noncomputable def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n-1) * d

-- Define the sum of the first n terms of the arithmetic sequence
noncomputable def sum_arith_seq (a₁ d n : ℕ) : ℕ := n * a₁ + (n * (n-1) / 2) * d

-- Given conditions and required proofs
theorem arithmetic_geometric_condition {d a₁ : ℕ} (h : d ≠ 0) (S₃ : sum_arith_seq a₁ d 3 = 9)
  (geometric_seq : (arithmetic_seq a₁ d 5)^2 = (arithmetic_seq a₁ d 3) * (arithmetic_seq a₁ d 8)) :
  d = 1 ∧ ∀ n, sum_arith_seq 2 1 n = (n^2 + 3 * n) / 2 :=
by
  sorry

end arithmetic_geometric_condition_l634_634380


namespace angle_BKC_is_30_degrees_l634_634063

-- Given a square ABCD, extend diagonal AC and mark point K such that BK = AC
variable {A B C D K : Point}
variable [square : IsSquare A B C D]
variable (h_extension : OnExtensionOfDiagonalACBeyondC A C K)
variable (h_BK_AC : Distance B K = Distance A C)

-- Define the goal to prove the angle BKC is 30 degrees
theorem angle_BKC_is_30_degrees (square : IsSquare A B C D)
  (h_extension : OnExtensionOfDiagonalACBeyondC A C K)
  (h_BK_AC : Distance B K = Distance A C) :
  angle B K C = 30 := 
sorry

end angle_BKC_is_30_degrees_l634_634063


namespace line_intersects_unit_circle_l634_634484

-- Given conditions
variables {a b θ : ℝ}
variables (h1 : a ≠ b)
variables (h2 : a^2 * sin θ + a * cos θ - π / 4 = 0)
variables (h3 : b^2 * sin θ + b * cos θ - π / 4 = 0)

-- Proof statement: the line connecting points A(a^2, a) and B(b^2, b) intersects the unit circle centered at the origin.
theorem line_intersects_unit_circle (a b θ : ℝ) (h1 : a ≠ b)
  (h2 : a^2 * sin θ + a * cos θ - π / 4 = 0)
  (h3 : b^2 * sin θ + b * cos θ - π / 4 = 0) :
  ∃ (m c : ℝ), m ≠ 0 ∧ (∀ x y ∈ ℝ, (x - (a + b) * y + ab = 0) ∧ (∃ x y ∈ ℝ, x^2 + y^2 = 1 ∧ y = m * x + c)) :=
sorry

end line_intersects_unit_circle_l634_634484


namespace mark_speed_l634_634211

-- Conditions
def distance : ℝ := 24
def time : ℝ := 4

-- Desired result
def speed (d t : ℝ) : ℝ := d / t

-- Proof statement
theorem mark_speed : speed distance time = 6 := by
  sorry

end mark_speed_l634_634211


namespace log2_125_eq_9y_l634_634578

theorem log2_125_eq_9y (y : ℝ) (h : Real.log 5 / Real.log 8 = y) : Real.log 125 / Real.log 2 = 9 * y :=
by
  sorry

end log2_125_eq_9y_l634_634578


namespace evaluate_expression_l634_634400

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_expression : 
  let p := ∏ i in Finset.range 12, (3 - w ^ (i + 1))
  p = 2391483 := 
by sorry

end evaluate_expression_l634_634400


namespace flux_vector_field_l634_634416

noncomputable def vector_field (x y z : ℝ) : EuclideanSpace3 ℝ := 
⟨x, y + z, z - y⟩

noncomputable def surface_eqn (x y z : ℝ) : Prop := 
x^2 + y^2 + z^2 = 9

noncomputable def plane_eqn (z : ℝ) : Prop := 
z = 0

theorem flux_vector_field : 
  let φ := 54 * Real.pi in
  ∫ (x y z : ℝ) in {p : EuclideanSpace3 ℝ // surface_eqn p.1 p.2 p.3 ∧ p.3 ≥ 0}, 
  (vector_field x y z) • (metric.sphere [x, y, z] 3).to_function = φ
:= sorry

end flux_vector_field_l634_634416


namespace extreme_value_theorem_zeros_range_theorem_l634_634440

-- Given conditions
def f (x k : ℝ) := Real.log x + k * x

def exists_extreme_value (x : ℝ) (k : ℝ) : Prop :=
  if k >= 0 then
    ∀ (x : ℝ), x > 0 → (f x k > (f x k)) -- no extreme value
  else ∃ (x : ℝ), x > 0 ∧ f x k = f (-1/k) k - 1

def has_two_zeros (x : ℝ) (k : ℝ) : Prop :=
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 k = 0 ∧ f x2 k = 0

-- Proof statements
theorem extreme_value_theorem (k : ℝ) : ∀ x > 0, exists_extreme_value x k :=
by
  sorry

theorem zeros_range_theorem (k : ℝ) : has_two_zeros x k ↔ -1 / Real.exp 1 < k ∧ k < 0 :=
by
  sorry

end extreme_value_theorem_zeros_range_theorem_l634_634440


namespace problem_solution_l634_634328

-- Definitions from the problem conditions
def sum_of_marked_angles (a : ℝ) : Prop := a = 900

def polygon_interior_angles (a b : ℝ) : Prop := a = (b - 2) * 180

def exponential_relationship (b c : ℝ) : Prop := 8^b = c^21

def logarithmic_relationship (c d : ℝ) : Prop := c = Real.logb d 81

-- Prove the questions equal the answers given conditions
theorem problem_solution (a b c d : ℝ) (h1 : sum_of_marked_angles a) 
    (h2 : polygon_interior_angles a b) 
    (h3 : exponential_relationship b c)
    (h4 : logarithmic_relationship c d) : 
    a = 900 ∧ b = 7 ∧ c = 2 ∧ d = 9 := 
begin
    sorry
end

end problem_solution_l634_634328


namespace min_value_of_a_plus_b_l634_634798

theorem min_value_of_a_plus_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (1 / (a + 1)) + (2 / (1 + b)) = 1) : 
  a + b ≥ 2 * Real.sqrt 2 + 1 :=
sorry

end min_value_of_a_plus_b_l634_634798


namespace actual_problem_l634_634817

-- Definitions of the predicates used
constants (Line Plane : Type) 
constants (perpendicular parallel : Line → Plane → Prop) 
constants (contained_in : Line → Plane → Prop) 
constants (skew : Line → Line → Prop)
constants (non_intersecting : Plane → Plane → Prop)

-- Given conditions
axioms 
  (m n : Line)
  (α β γ : Plane)
  (h_non_intersecting_lines : ¬ parallel m n)
  (h_non_intersecting_planes : non_intersecting α β ∧ non_intersecting α γ ∧ non_intersecting β γ)

-- Propositions
axiom prop1 : ∀ (m : Line) (α β : Plane), (perpendicular m α) ∧ (perpendicular m β) → parallel α β
axiom prop4 : ∀ (m n : Line) (α β : Plane), skew m n ∧ contained_in m α ∧ parallel m β ∧ contained_in n β ∧ parallel n α → parallel α β

theorem actual_problem : 
  (∀ (m : Line) (α β : Plane), (perpendicular m α) ∧ (perpendicular m β) → parallel α β) ∧ 
  (∀ (m n : Line) (α β : Plane), skew m n ∧ contained_in m α ∧ parallel m β ∧ contained_in n β ∧ parallel n α → parallel α β) :=
by sorry

end actual_problem_l634_634817


namespace find_expression_value_l634_634441

theorem find_expression_value (x : ℝ) (h : x^2 - 5*x = 14) : 
  (x-1)*(2*x-1) - (x+1)^2 + 1 = 15 := 
by 
  sorry

end find_expression_value_l634_634441


namespace mark_speed_l634_634210

-- Conditions
def distance : ℝ := 24
def time : ℝ := 4

-- Desired result
def speed (d t : ℝ) : ℝ := d / t

-- Proof statement
theorem mark_speed : speed distance time = 6 := by
  sorry

end mark_speed_l634_634210


namespace complex_conjugate_of_z_l634_634253

variable (z : ℂ)
variable (h : z = i * (1 - i))

theorem complex_conjugate_of_z : conj z = 1 - i :=
by
  sorry

end complex_conjugate_of_z_l634_634253


namespace area_of_triangle_is_72_l634_634772

open Real

-- Define points in ℝ²
def point (x y : ℝ) := (x, y)

-- The given points
def A := point (-4) 8
def B := point (-8) 4

-- The triangle is formed by the x-axis, y-axis, and line passing through A and B
def right_triangle_formed_by_axes_line_through (A B : ℝ × ℝ) : Prop :=
  A.fst = -4 ∧ A.snd = 8 ∧ B.fst = -8 ∧ B.snd = 4

-- Area calculation for the right triangle
def area_of_right_triangle (base height : ℝ) : ℝ :=
  1 / 2 * base * height

-- Main theorem stating the area is 72 square units
theorem area_of_triangle_is_72 :
  right_triangle_formed_by_axes_line_through A B →
  area_of_right_triangle 12 12 = 72 := sorry

end area_of_triangle_is_72_l634_634772


namespace correct_statements_l634_634366

-- Define line-to-plane parallelism in Lean
def line_parallel_to_plane (l : set (ℝ × ℝ × ℝ)) (α : set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ (x y ∈ l) (z ∈ α), x ≠ z ∧ y ≠ z

-- Euclidean parallel postulate
axiom parallel_postulate {A : (ℝ × ℝ × ℝ)} {l : set (ℝ × ℝ × ℝ)} :
  ∃! (m : set (ℝ × ℝ × ℝ)), m ≠ l ∧ ∀ (x ∈ l) (y ∈ m), x ≠ A ∧ y ≠ A ∧ x ≠ y

-- Theorem: Proof statements (1), (2), and (4) are correct and statement (3) is incorrect
theorem correct_statements {l : set (ℝ × ℝ × ℝ)} {α : set (ℝ × ℝ × ℝ)} :
  line_parallel_to_plane l α →
  (∀ (β : set (ℝ × ℝ × ℝ)), l ⊆ β ∧ β ≠ α → ∃! (m : set (ℝ × ℝ × ℝ)), (α ∩ β = m) ∧ line_parallel_to_plane l m) ∧ 
  (∀ (x ∈ α) y ∈ α, x ≠ y → ¬ ∃ (z ∈ l), x = z ∨ y = z) ∧ 
  (∀ (A : (ℝ × ℝ × ℝ)), A ∉ l → (∃! (m : set (ℝ × ℝ × ℝ)), line_parallel_to_plane m l ∧ line_parallel_to_plane m {A}) → False) ∧
  (∀ (A ∈ α), ∃! (m : set (ℝ × ℝ × ℝ)), line_parallel_to_plane m l ∧ ∀ (x ∈ m) (y ∈ α), x ≠ y → line_parallel_to_plane m α) :=
sorry

end correct_statements_l634_634366


namespace rectangle_area_x_l634_634790

noncomputable def x_value : ℝ := (7 + 2 * Real.sqrt 7) / 3

theorem rectangle_area_x (x : ℝ) 
  (hx : (x - 3) * (3 * x + 4) = 9 * x - 19) : 
  x = x_value := 
by 
  sorry

end rectangle_area_x_l634_634790


namespace domain_f_l634_634773

def floor_function (x : ℝ) : ℝ := ⌊x^2 - 5 * x + 8⌋

def f (x : ℝ) : ℝ := 1 / (floor_function x + x - 2)

theorem domain_f (x : ℝ) :
  ¬ (floor_function x + x = 2) ↔ 
  (x < 1) ∨ (1 < x ∧ x < 3) ∨ (3 < x) := sorry

end domain_f_l634_634773


namespace sin_F_l634_634159

variable {D E F : ℝ} 
variable {DEF : triangle}
variable [right_triangle DEF]
variable (sin_D : sin D = 5 / 13)
variable (sin_E : sin E = 1)

theorem sin_F (right_angle : E = 90) : sin F = 12 / 13 :=
sorry

end sin_F_l634_634159


namespace no_solutions_ordered_triples_l634_634489

theorem no_solutions_ordered_triples :
  ¬ ∃ (x y z : ℤ), 
    x^2 - 4 * x * y + 3 * y^2 - z^2 = 25 ∧
    -x^2 + 5 * y * z + 3 * z^2 = 55 ∧
    x^2 + 2 * x * y + 9 * z^2 = 150 :=
by
  sorry

end no_solutions_ordered_triples_l634_634489


namespace solve_for_m_l634_634885

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 4

theorem solve_for_m (m : ℤ) (h : ∃ x : ℝ, 2^x + x = 4 ∧ m ≤ x ∧ x ≤ m + 1) : m = 1 :=
by
  sorry

end solve_for_m_l634_634885


namespace part1_of_question_part2_of_question_l634_634601

noncomputable def ellipse (a b : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (a^2 - b^2 = (a * sqrt(3) / 3)^2)

noncomputable def vertices_distance (a b : ℝ) : Prop :=
  2 * b = 4

noncomputable def line_through_E (k : ℝ) : Affine :=
  { p := (0, 1), slope := k }

theorem part1_of_question
  (a b k x y : ℝ)
  (h1 : ellipse a b)
  (h2 : vertices_distance a b)
  (h3 : (2 + 3 * k^2) * x^2 + 6 * k * x - 9 = 0)
  (h4 : | x | = | y |)
  (h5 : x^2 / a^2 + (k * x + 1)^2 / b^2 = 1)
  : k = sqrt(6) / 3 ∨ k = -sqrt(6) / 3 :=
sorry

theorem part2_of_question
  (k : ℝ)
  (h1 : ellipse 6 4)
  (h2 : vertices_distance 6 4)
  : ∀ x1 x2 y1 y2, ¬ (x1 * (y2 + 2) = x2 * (y1 - 2)) :=
sorry 

end part1_of_question_part2_of_question_l634_634601


namespace calc_expression_l634_634756

theorem calc_expression : 
  |1 - Real.sqrt 2| - Real.sqrt 8 + (Real.sqrt 2 - 1)^0 = -Real.sqrt 2 :=
by
  sorry

end calc_expression_l634_634756


namespace find_b_l634_634808

theorem find_b (b : ℚ) : (-4 : ℚ) * (45 / 4) = -45 → (-4 + 45 / 4) = -b → b = -29 / 4 := by
  intros h1 h2
  sorry

end find_b_l634_634808


namespace john_spending_l634_634938

open Nat Real

noncomputable def cost_of_silver (silver_ounce: Real) (silver_price: Real) : Real :=
  silver_ounce * silver_price

noncomputable def quantity_of_gold (silver_ounce: Real): Real :=
  2 * silver_ounce

noncomputable def cost_per_ounce_gold (silver_price: Real) (multiplier: Real): Real :=
  silver_price * multiplier

noncomputable def cost_of_gold (gold_ounce: Real) (gold_price: Real) : Real :=
  gold_ounce * gold_price

noncomputable def total_cost (cost_silver: Real) (cost_gold: Real): Real :=
  cost_silver + cost_gold

theorem john_spending :
  let silver_ounce := 1.5
  let silver_price := 20
  let gold_multiplier := 50
  let cost_silver := cost_of_silver silver_ounce silver_price
  let gold_ounce := quantity_of_gold silver_ounce
  let gold_price := cost_per_ounce_gold silver_price gold_multiplier
  let cost_gold := cost_of_gold gold_ounce gold_price
  let total := total_cost cost_silver cost_gold
  total = 3030 :=
by
  sorry

end john_spending_l634_634938


namespace smallest_a_bound_l634_634787

theorem smallest_a_bound (a : ℕ) : (∀ n : ℕ, n > 0 → (∑ k in finset.range (2*n + 2), if k > n then 1/(k:ℚ) else 0) < a - 2007 * (1/3 : ℚ)) ↔ a = 2009 := 
sorry

end smallest_a_bound_l634_634787


namespace robin_bobin_can_meet_prescription_l634_634148

def large_gr_pill : ℝ := 11
def medium_gr_pill : ℝ := -1.1
def small_gr_pill : ℝ := -0.11
def prescribed_gr : ℝ := 20.13

theorem robin_bobin_can_meet_prescription :
  ∃ (large : ℕ) (medium : ℕ) (small : ℕ), large ≥ 1 ∧ medium ≥ 1 ∧ small ≥ 1 ∧
  large_gr_pill * large + medium_gr_pill * medium + small_gr_pill * small = prescribed_gr :=
sorry

end robin_bobin_can_meet_prescription_l634_634148


namespace joann_lollipops_l634_634937

theorem joann_lollipops : 
  ∃ (a : ℚ), 
  (7 * a  + 3 * (1 + 2 + 3 + 4 + 5 + 6) = 150) ∧ 
  (a_4 = a + 9) ∧ 
  (a_4 = 150 / 7) :=
by
  sorry

end joann_lollipops_l634_634937


namespace line_divides_circle_l634_634514

theorem line_divides_circle (k m : ℝ) :
  (∀ x y : ℝ, y = x - 1 → x^2 + y^2 + k*x + m*y - 4 = 0 → m - k = 2) :=
sorry

end line_divides_circle_l634_634514


namespace one_box_empty_two_boxes_empty_l634_634282

-- Definition of the general problem context
def distinctBalls := {b1, b2, b3, b4}
def distinctBoxes := {box1, box2, box3, box4}

-- Theorem 1: Exactly one box is empty
theorem one_box_empty : 
  (∃ (boxes : Finset distinctBoxes) (hboxes : boxes.card = 3), 
    ∑ (grp : Finset distinctBalls) (hgrp : grp.card = 2), 
    (boxes.card * grp.card * (boxes.card.factorial)) = 144) := 
sorry
  
-- Theorem 2: Exactly two boxes are empty
theorem two_boxes_empty : 
  (∃ (boxes : Finset distinctBoxes) (hboxes : boxes.card = 2), 
    ∑ (distr : (Finset distinctBalls)),
      ((∃ (h1 : distr.card = 1), ∃ (h3 : (distinctBalls \ distr).card = 3, (boxes.card * distr.card.cardinality) + 
        ∃ (h2 : distr.card = 2), ∃ (h2' : (distinctBalls \ distr).card = 2, (boxes.card * distr.card.cardinality) + 
        ∃ (h3 : distr.card = 3), ∃ (h1' : (distinctBalls \ distr).card = 1, (boxes.card * distr.card.cardinality))) = 84) :=
sorry

end one_box_empty_two_boxes_empty_l634_634282


namespace coin_flip_probability_l634_634498

theorem coin_flip_probability : 
  ∀ (prob_tails : ℚ) (seq : List (Bool × ℚ)),
    prob_tails = 1/2 →
    seq = [(true, 1/2), (true, 1/2), (false, 1/2), (false, 1/2)] →
    (seq.map Prod.snd).prod = 0.0625 :=
by 
  intros prob_tails seq htails hseq 
  sorry

end coin_flip_probability_l634_634498


namespace probability_same_flavor_l634_634369

theorem probability_same_flavor (num_flavors : ℕ) (num_bags : ℕ) (h1 : num_flavors = 4) (h2 : num_bags = 2) :
  let total_outcomes := num_flavors ^ num_bags
  let favorable_outcomes := num_flavors
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 4 :=
by
  sorry

end probability_same_flavor_l634_634369


namespace magnitude_of_angle_B_value_of_cos_A_l634_634173

variable {a b c : ℝ} {A B C : ℝ}

-- Condition 1: The sides opposite to angles A, B, and C are a, b, and c respectively.
-- Condition 2: Given b * cos C + b * sin C = a
def condition1 : Prop := b * cos C + b * sin C = a

-- Prove that angle B is π/4
theorem magnitude_of_angle_B (h1 : condition1) : B = π / 4 := sorry

-- Condition 3: Given height from B to AC is a / 4
def height_condition : Prop := a / 4 = a / 4  -- simplified placeholder

-- Prove the value of cos A when B = π / 4
theorem value_of_cos_A (h1 : condition1) (h2 : height_condition) (hB : B = π / 4) : cos A = - (sqrt 5) / 5 := sorry

end magnitude_of_angle_B_value_of_cos_A_l634_634173


namespace probability_correct_l634_634909

noncomputable def num_students : ℕ := 30
def possible_initials_set := {c : Char | 'A' ≤ c ∧ c ≤ 'Z'}
def vowels_set := {'A', 'E', 'I', 'O', 'U', 'Y'}
def total_possible_initials := possible_initials_set.size
def vowel_initials := vowels_set.size

def probability_vowel_initials : Rat :=
  vowel_initials / total_possible_initials

theorem probability_correct :
  probability_vowel_initials = 3 / 13 :=
by
  have h1 : total_possible_initials = 26 := sorry
  have h2 : vowel_initials = 6 := sorry
  rw [h1, h2]
  norm_num
  exact sorry

end probability_correct_l634_634909


namespace product_of_positive_real_solutions_eq_eight_l634_634509

noncomputable def product_of_positive_real_solutions : ℂ :=
  let solutions := {x : ℂ | x^8 = -256 ∧ x.re > 0}
  ∏ x in solutions, x

theorem product_of_positive_real_solutions_eq_eight :
  product_of_positive_real_solutions = 8 :=
sorry

end product_of_positive_real_solutions_eq_eight_l634_634509


namespace log_eq_iff_sqrt_l634_634004

-- Conditions
variables {a b c t : ℝ}
axiom (a_pos : 0 < a)
axiom (b_pos : 0 < b)
axiom (c_pos : 0 < c)
axiom (t_pos : 0 < t)
axiom (a_ne_one : a ≠ 1)
axiom (b_ne_one : b ≠ 1)
axiom (c_ne_one : c ≠ 1)
axiom (t_ne_one : t ≠ 1)
axiom (b_ne_c : b ≠ c)

-- The equivalence proof problem
theorem log_eq_iff_sqrt (a_pos : 0 < a) 
                        (b_pos : 0 < b) 
                        (c_pos : 0 < c) 
                        (t_pos : 0 < t) 
                        (a_ne_one : a ≠ 1) 
                        (b_ne_one : b ≠ 1) 
                        (c_ne_one : c ≠ 1) 
                        (t_ne_one : t ≠ 1) 
                        (b_ne_c : b ≠ c) :
    (log a t / log c t = (log a t - log b t) / (log b t - log c t)) ↔ (b = Real.sqrt (a * c)) :=
by sorry

end log_eq_iff_sqrt_l634_634004


namespace min_translation_t_for_odd_f_l634_634386

-- Definitions based on the conditions
def determinant (a₁ a₂ b₁ b₂ : ℝ) : ℝ := a₁ * b₂ - a₂ * b₁

def f (x : ℝ) : ℝ := determinant (√3) (sin (2 * x)) 1 (cos (2 * x))

-- Theorem to prove
theorem min_translation_t_for_odd_f :
  ∃ t : ℝ, (∀ x : ℝ, t > 0 → 
  f (x + t) = 0 ↔ 2 * t - π / 3 = n * π for some n ∈ ℤ) → 
  t = π / 6 :=
sorry

end min_translation_t_for_odd_f_l634_634386


namespace one_meter_eq_jumps_l634_634246

theorem one_meter_eq_jumps 
  (x y a b p q s t : ℝ) 
  (h1 : x * hops = y * skips)
  (h2 : a * jumps = b * hops)
  (h3 : p * skips = q * leaps)
  (h4 : s * leaps = t * meters) :
  1 * meters = (sp * x * a / (tq * y * b)) * jumps :=
sorry

end one_meter_eq_jumps_l634_634246


namespace sales_tax_difference_l634_634340

-- Definitions for the conditions
def item_price : ℝ := 50
def tax_rate1 : ℝ := 0.075
def tax_rate2 : ℝ := 0.05

-- Calculations based on the conditions
def tax1 := item_price * tax_rate1
def tax2 := item_price * tax_rate2

-- The proof statement
theorem sales_tax_difference :
  tax1 - tax2 = 1.25 :=
by
  sorry

end sales_tax_difference_l634_634340


namespace find_b_l634_634726

open Real

-- Definition of points as conditions
def P1 := (-3, 2 : ℝ × ℝ)
def P2 := (2, -3 : ℝ × ℝ)

-- The main theorem stating the proof problem
theorem find_b (b : ℝ) (h : ∃ k : ℝ, (k * b, k * -1) = (2 - (-3), -3 - 2)) : b = 1 :=
by
  sorry

end find_b_l634_634726


namespace smallest_n_satisfying_conditions_l634_634786

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), (n > 0) ∧ (∀ (a b : ℤ), ∃ (α β : ℝ), 
    α ≠ β ∧ (0 < α) ∧ (α < 1) ∧ (0 < β) ∧ (β < 1) ∧ (n * α^2 + a * α + b = 0) ∧ (n * β^2 + a * β + b = 0)
 ) ∧ (∀ (m : ℕ), m > 0 ∧ m < n → ¬ (∀ (a b : ℤ), ∃ (α β : ℝ), 
    α ≠ β ∧ (0 < α) ∧ (α < 1) ∧ (0 < β) ∧ (β < 1) ∧ (m * α^2 + a * α + b = 0) ∧ (m * β^2 + a * β + b = 0))) := 
sorry

end smallest_n_satisfying_conditions_l634_634786


namespace correct_statement_l634_634237

def quadratic_function (x : ℝ) : ℝ := (x - 1)^2 + 5

theorem correct_statement (x : ℝ) (h : x > 1) : 
  ∃ δ > 0, ∀ x₁ x₂, x₁ > 1 ∧ x₂ > 1 ∧ x₁ < x₂ -> quadratic_function x₁ < quadratic_function x₂ :=
begin
  sorry
end

end correct_statement_l634_634237


namespace det_special_matrix_formula_l634_634724

noncomputable def det_special_matrix (n : ℕ) : ℤ :=
  let A : matrix (fin n) (fin n) ℤ := λ i j, int.nat_abs (i.val - j.val) in
  matrix.det A

theorem det_special_matrix_formula (n : ℕ) : det_special_matrix n = (-1)^(n-1) * (n-1) * 2^(n-2) :=
by
  sorry

end det_special_matrix_formula_l634_634724


namespace a2_a3_equals_20_l634_634850

-- Sequence definition
def a_n (n : ℕ) : ℕ :=
  if n % 2 = 1 then 3 * n + 1 else 2 * n - 2

-- Proof that a_2 * a_3 = 20
theorem a2_a3_equals_20 :
  a_n 2 * a_n 3 = 20 :=
by
  sorry

end a2_a3_equals_20_l634_634850


namespace log_eq_res_l634_634574

theorem log_eq_res (y m : ℝ) (h₁ : real.log 5 / real.log 8 = y) (h₂ : real.log 125 / real.log 2 = m * y) : m = 9 := 
sorry

end log_eq_res_l634_634574


namespace shortest_path_length_l634_634955

theorem shortest_path_length 
  (C : Circle := {center := (0, 0), radius := 12})
  (P1 : Point := (8 * Real.sqrt 3, 0))
  (P2 : Point := (0, 12 * Real.sqrt 2)) :
  shortest_path_length_without_interior_intersection C P1 P2 = 12 + 4 * Real.sqrt 3 + Real.pi := 
  sorry

end shortest_path_length_l634_634955


namespace parametric_to_ordinary_l634_634268

variables (θ : ℝ) (x y : ℝ)

def parametric_to_ordinary_equation (θ x y : ℝ) : Prop :=
  x = -1 + 2 * Math.cos θ ∧ y = 2 + 2 * Math.sin θ

theorem parametric_to_ordinary :
  (∀ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * Math.PI → parametric_to_ordinary_equation θ x y) →
  (x + 1)^2 + (y - 2)^2 = 4 :=
by
  sorry

end parametric_to_ordinary_l634_634268


namespace a_seq_general_formula_sum_a_c_b_seq_arithmetic_l634_634061

-- Problem (Ⅰ): Define the sequence an and prove the general formula
def a_seq : ℕ → ℕ
| 0       => 1
| (n + 1) => 2 * a_seq n + 1

theorem a_seq_general_formula (n : ℕ) :
  a_seq n = 2 ^ (n + 1) - 1 := sorry

-- Problem (Ⅱ): Sum of the first n terms of a_n * c_n
def c_seq : ℕ → ℕ
| n => 2 * (n + 1)

def S_n (n : ℕ) : ℕ :=
  ∑ k in finset.range (n + 1), (a_seq k) * (c_seq k)

theorem sum_a_c (n : ℕ) :
  S_n n = (n - 1) * 2 ^ (n + 2) + 4 - (n * (n + 1)) := sorry

-- Problem (Ⅲ): Arithmetic sequence b_n
def b_seq : ℕ → ℕ
| 0       => 0 -- usually placeholder as sequences typically begin at 1
| 1       => 4
| (n + 1) => ((n - 1) * b_seq (n - 1) + 2) / n 

theorem b_seq_arithmetic (n : ℕ) :
  b_seq n = 2 * (n + 1) := sorry

end a_seq_general_formula_sum_a_c_b_seq_arithmetic_l634_634061


namespace f_monotonic_intervals_range_of_m_for_g_l634_634841

-- Definitions
def f (x m : ℝ) : ℝ := (1/2) * x^2 + m * x + 6 * Real.log x
def g (x m : ℝ) : ℝ := (1/2) * x^2 + m * x + 6 * Real.log x - x^2 + x

-- Conditions
axiom tangent_line_parallel (m : ℝ) : (1 : ℝ) + m + (6 : ℝ) = 2
axiom g_increasing_on_interval (m : ℝ) : ∀ x ∈ Ioo 0 1, -x + (6 / x) + m + 1 ≥ 0

-- Proof Statements
theorem f_monotonic_intervals (m : ℝ) (h : m = -5) : 
  ∀ x, (0 < x ∧ x < 2) ∨ (3 < x) → 0 < x + m + 6 / x :=
  sorry

theorem range_of_m_for_g (m : ℝ) : g_increasing_on_interval m ↔ -6 ≤ m :=
  sorry

end f_monotonic_intervals_range_of_m_for_g_l634_634841


namespace polynomial_sum_l634_634949

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def j (x : ℝ) : ℝ := x^2 - x - 3

theorem polynomial_sum (x : ℝ) : f x + g x + h x + j x = -3 * x^2 + 11 * x - 15 := by
  sorry

end polynomial_sum_l634_634949


namespace arrangement_of_multiples_l634_634071

theorem arrangement_of_multiples (x y z : ℝ) (h1 : 2^x = 3^y) (h2 : 3^y = 5^z) (h3 : 0 < x) (h4 : 0 < y) (h5 : 0 < z) :
  3 * y < 2 * x ∧ 2 * x < 5 * z :=
by
  sorry

end arrangement_of_multiples_l634_634071


namespace ratio_of_rectangular_sides_l634_634341

theorem ratio_of_rectangular_sides (x y : ℝ) (h : x + y - sqrt (x^2 + y^2) = y / 3) :
  x / y = 5 / 12 := sorry

end ratio_of_rectangular_sides_l634_634341


namespace quadratic_polynomial_root_count_l634_634058

theorem quadratic_polynomial_root_count
  (b c : ℝ)
  (D : ℝ := b^2 - 4 * c)
  (f : ℝ → ℝ := λ x, x^2 + b * x + c)
  (h : ∀ x : ℝ, f(x) = 0 → (x = (-b + real.sqrt D) / 2 ∨ x = (-b - real.sqrt D) / 2))
  (h_distinct : real.sqrt D ≠ 0) :
  ∃! x : ℝ, f(x) + f(x - real.sqrt D) = 0 :=
begin
  sorry
end

end quadratic_polynomial_root_count_l634_634058


namespace find_a_value_l634_634057

theorem find_a_value
  (P : ℝ × ℝ) -- Point P
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + 2*x - 6*y + 5 = 0) -- Circle equation
  (line_through : ∀ x y : ℝ, (line_eq : x = 1 ∧ y = 2)) -- Line through P
  (line_perpendicular : ∀ a x y : ℝ, ax + y - 1 = 0 → line_eq (1 - (1/a))) : 
  (a = 1/2) := 
by
  -- Proof to be added
  sorry

end find_a_value_l634_634057


namespace program_total_cost_l634_634322

-- Define the necessary variables and constants
def ms_to_s : Float := 0.001
def os_overhead : Float := 1.07
def cost_per_ms : Float := 0.023
def mount_cost : Float := 5.35
def time_required : Float := 1.5

-- Calculate components of the total cost
def total_cost_for_computer_time := (time_required * 1000) * cost_per_ms
def total_cost := os_overhead + total_cost_for_computer_time + mount_cost

-- State the theorem
theorem program_total_cost : total_cost = 40.92 := by
  sorry

end program_total_cost_l634_634322


namespace hyperbola_eccentricity_l634_634800

theorem hyperbola_eccentricity (a b c e : ℝ) 
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_vertices : c = a * e)
  (h_foci_distance : c^2 = a^2 + b^2)
  (h_angle : ∀ (B : ℝ × ℝ), ∠ ((-a, 0): ℝ × ℝ) B ((a, 0): ℝ × ℝ) = 60) 
  : e = sqrt 21 / 3 := 
sorry

end hyperbola_eccentricity_l634_634800


namespace average_of_remaining_quantity_l634_634634

theorem average_of_remaining_quantity (q : Fin 8 → ℝ) (h1 : (∑ i, q i) / 8 = 15)
  (h2 : (∑ i in Finset.finRange 5, q i) / 5 = 10) (h3 : (∑ i in (Finset.finRange 2).map (Function.Embedding.subtype MapFin.predicate.injOn), q i) / 2 = 22) :
  q 7 = 26 := by
    sorry

end average_of_remaining_quantity_l634_634634


namespace brianna_more_chocolates_than_alix_l634_634221

def Nick_ClosetA : ℕ := 10
def Nick_ClosetB : ℕ := 6
def Alix_ClosetA : ℕ := 3 * Nick_ClosetA
def Alix_ClosetB : ℕ := 3 * Nick_ClosetA
def Mom_Takes_From_AlixA : ℚ := (1/4:ℚ) * Alix_ClosetA
def Brianna_ClosetA : ℚ := 2 * (Nick_ClosetA + Alix_ClosetA - Mom_Takes_From_AlixA)
def Brianna_ClosetB_after : ℕ := 18
def Brianna_ClosetB : ℚ := Brianna_ClosetB_after / (0.8:ℚ)

def Brianna_Total : ℚ := Brianna_ClosetA + Brianna_ClosetB
def Alix_Total : ℚ := Alix_ClosetA + Alix_ClosetB
def Difference : ℚ := Brianna_Total - Alix_Total

theorem brianna_more_chocolates_than_alix : Difference = 35 := by
  sorry

end brianna_more_chocolates_than_alix_l634_634221


namespace petya_wins_probability_l634_634988

/-- Petya plays a game with 16 stones where players alternate in taking 1 to 4 stones. 
     Petya wins if they can take the last stone first while making random choices. 
     The computer plays optimally. The probability of Petya winning is 1 / 256. -/
theorem petya_wins_probability :
  let stones := 16 in
  let optimal_strategy := ∀ n, n % 5 = 0 in
  let random_choice_probability := (1 / 4 : ℚ) in
  let total_random_choices := 4 in
  (random_choice_probability ^ total_random_choices) = (1 / 256 : ℚ) :=
by
  sorry

end petya_wins_probability_l634_634988


namespace solve_for_x_l634_634128

theorem solve_for_x (x : ℝ) : 2 * x + 3 * x + 4 * x = 12 + 9 + 6 → x = 3 :=
by
  sorry

end solve_for_x_l634_634128


namespace petya_wins_probability_l634_634990

/-- Petya plays a game with 16 stones where players alternate in taking 1 to 4 stones. 
     Petya wins if they can take the last stone first while making random choices. 
     The computer plays optimally. The probability of Petya winning is 1 / 256. -/
theorem petya_wins_probability :
  let stones := 16 in
  let optimal_strategy := ∀ n, n % 5 = 0 in
  let random_choice_probability := (1 / 4 : ℚ) in
  let total_random_choices := 4 in
  (random_choice_probability ^ total_random_choices) = (1 / 256 : ℚ) :=
by
  sorry

end petya_wins_probability_l634_634990


namespace complex_number_in_second_quadrant_l634_634456

def f (x : ℂ) : ℂ := x^2

theorem complex_number_in_second_quadrant : 
  let c := complex.mk 1 1 in
  let cn := (f c) / (complex.mk 3 1) in
  cn.re < 0 ∧ cn.im > 0 :=
by 
  have hc : f c = (1 + complex.i)^2 := rfl
  have hcn : cn = ((1 + complex.i)^2) / (3 + complex.i) := sorry
  sorry

end complex_number_in_second_quadrant_l634_634456


namespace trapezoid_base_ratio_l634_634258

theorem trapezoid_base_ratio (a b h : ℝ) (ha_gt_hb : a > b) 
  (h_div_ratio : ∃ x y, x = a / 4 ∧ y = 3 * (a / 4) ∧ x + y = a ∧ (x : ℝ)/(y : ℝ) = 1/3) :
  a / b = 3 :=
by
  have hx : ∃ x y, x = a / 4 ∧ y = 3 * (a / 4) ∧ x + y = a ∧ (x : ℝ)/(y : ℝ) = 1/3 := h_div_ratio,
  obtain ⟨x, y, hx1, hy1, hx2, hy2⟩ := hx,
  sorry

end trapezoid_base_ratio_l634_634258


namespace symmetrical_points_form_regular_polygon_l634_634705
-- Necessary imports

-- Declarations
def is_regular_polygon {Point : Type} [inhabited Point] [has_dist Point] [has_angle Point] (p : list Point) : Prop :=
sorry  -- Define what it means for points to form a regular polygon

def symmetrical_with_respect_to (X O A : Point) : Point :=
sorry  -- Define how to find the symmetrical point

variables (Point : Type) [inhabited Point] [has_dist Point] [has_angle Point] 
variables (O : Point) (X : Point) (A : ℕ → Point) (n : ℕ) (h1 : ∀ (i : ℕ), i < n → dist O (A i) = dist O (A 0))

theorem symmetrical_points_form_regular_polygon :
  let symmetric_points := list.of_fn (λ i, symmetrical_with_respect_to X O (A i)) in
  is_regular_polygon symmetric_points :=
sorry  -- Proof of the theorem

end symmetrical_points_form_regular_polygon_l634_634705


namespace petya_wins_probability_l634_634982

theorem petya_wins_probability : 
  ∃ p : ℚ, p = 1 / 256 ∧ 
  initial_stones = 16 ∧ 
  (∀ n, (1 ≤ n ∧ n ≤ 4)) ∧ 
  (turns(1) ∨ turns(2)) ∧ 
  (last_turn_wins) ∧ 
  (Petya_starts) ∧ 
  (Petya_plays_random ∧ Computer_plays_optimally) 
    → Petya_wins_with_probability p
:= sorry

end petya_wins_probability_l634_634982


namespace right_triangle_area_l634_634899

theorem right_triangle_area (a b : ℝ) (h : a^2 - 7 * a + 12 = 0 ∧ b^2 - 7 * b + 12 = 0) : 
  ∃ A : ℝ, (A = 6 ∨ A = 3 * (Real.sqrt 7 / 2)) ∧ A = 1 / 2 * a * b := 
by 
  sorry

end right_triangle_area_l634_634899


namespace arith_seq_sum_terms_l634_634163

variable {a : ℕ → ℝ}
variable {n : ℕ}

-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n k, a (n + k) = a n + k * (a 1 - a 0)

-- Define the sum of the first 20 terms
def sum_first_20_terms (a : ℕ → ℝ) : ℝ :=
  (20 / 2) * (a 0 + a 19)

-- State the main theorem
theorem arith_seq_sum_terms (h_arith : is_arithmetic_sequence a)
  (h_sum : sum_first_20_terms a = 170) :
  a 5 + a 8 + a 10 + a 15 = 34 := 
sorry

end arith_seq_sum_terms_l634_634163


namespace solution_set_of_inequality_system_l634_634276

theorem solution_set_of_inequality_system (x : ℝ) : (x + 1 > 0) ∧ (-2 * x ≤ 6) ↔ (x > -1) := 
by 
  sorry

end solution_set_of_inequality_system_l634_634276


namespace line_through_trisection_points_l634_634206

theorem line_through_trisection_points : 
  ∀ (P Q : ℝ × ℝ), 
  let trisection1 := (2, 7), 
      trisection2 := (8, -2), 
      trisect_pts := [(4, 4), (6, 1)], 
      line_eq := (x - 4 * y + 10 = 0) in 
      (P = (4, 3) ∧ Q = (2 + (8 - 2) / 3, 7 + (-2 - 7) / 3) ∨ Q = (2 + 2 * (8 - 2) / 3, 7 + 2 * (-2 - 7) / 3)) → 
      line_eq (P.1) (P.2) = line_eq (Q.1) (Q.2) := 
by
  intros P Q trisection1 trisection2 trisect_pts line_eq h,
  sorry

end line_through_trisection_points_l634_634206


namespace ratio_of_dogs_to_cats_l634_634167

theorem ratio_of_dogs_to_cats (D C : ℕ) (hC : C = 40) (h : D + 20 = 2 * C) :
  D / Nat.gcd D C = 3 ∧ C / Nat.gcd D C = 2 :=
by
  sorry

end ratio_of_dogs_to_cats_l634_634167


namespace rational_roots_iff_a_eq_b_l634_634569

theorem rational_roots_iff_a_eq_b (a b : ℤ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x : ℚ, x^2 + (a + b)^2 * x + 4 * a * b = 1) ↔ a = b :=
by
  sorry

end rational_roots_iff_a_eq_b_l634_634569


namespace product_of_positive_real_solutions_eq_eight_l634_634508

noncomputable def product_of_positive_real_solutions : ℂ :=
  let solutions := {x : ℂ | x^8 = -256 ∧ x.re > 0}
  ∏ x in solutions, x

theorem product_of_positive_real_solutions_eq_eight :
  product_of_positive_real_solutions = 8 :=
sorry

end product_of_positive_real_solutions_eq_eight_l634_634508


namespace right_triangle_area_l634_634891

def roots (a b : ℝ) : Prop :=
  a * b = 12 ∧ a + b = 7

def area (A : ℝ) : Prop :=
  A = 6 ∨ A = 3 * Real.sqrt 7 / 2

theorem right_triangle_area (a b A : ℝ) (h : roots a b) : area A := 
by 
  -- The proof steps would go here
  sorry

end right_triangle_area_l634_634891


namespace inscribed_circle_radius_l634_634359

theorem inscribed_circle_radius (a b c : ℕ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) : 
  let s := (a + b + c) / 2 in
  let area := (a * b) / 2 in
  let r := area / s in
  r = 1 :=
by
  have h_s : s = 6 := by
    simp [h₁, h₂, h₃]
  have h_area : area = 6 := by
    simp [h₁, h₂]
  have h_r : r = 1 := by
    simp [h_s, h_area]
  sorry

end inscribed_circle_radius_l634_634359


namespace range_of_a_l634_634042

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (1/3) * (|x|^3) - a * x^2 + (2 - a) * |x| + b

theorem range_of_a (b : ℝ) :
  (∀ x : ℝ, ∃ a : ℝ, f(x, a, b).deriv = 0 → ((1 < a) ∧ (a < 2)) = 
  ∀ (a : ℝ), 1 < a ∧ a < 2 :=
sorry

end range_of_a_l634_634042


namespace greg_lent_tessa_40_l634_634107

noncomputable def initial_loan_amount (weekly_payment : ℝ) (later_loan : ℝ) (current_debt : ℝ) : ℝ :=
  (current_debt - later_loan) * 2

theorem greg_lent_tessa_40 :
  initial_loan_amount 0 10 30 = 40 :=
by
  unfold initial_loan_amount
  norm_num
  sorry

end greg_lent_tessa_40_l634_634107


namespace fraction_Renz_Miles_l634_634971

-- Given definitions and conditions
def Mitch_macarons : ℕ := 20
def Joshua_diff : ℕ := 6
def kids : ℕ := 68
def macarons_per_kid : ℕ := 2
def total_macarons_given : ℕ := kids * macarons_per_kid
def Joshua_macarons : ℕ := Mitch_macarons + Joshua_diff
def Miles_macarons : ℕ := 2 * Joshua_macarons
def Mitch_Joshua_Miles_macarons : ℕ := Mitch_macarons + Joshua_macarons + Miles_macarons
def Renz_macarons : ℕ := total_macarons_given - Mitch_Joshua_Miles_macarons

-- The theorem to prove
theorem fraction_Renz_Miles : (Renz_macarons : ℚ) / (Miles_macarons : ℚ) = 19 / 26 :=
by
  sorry

end fraction_Renz_Miles_l634_634971


namespace nine_permutations_count_l634_634819

theorem nine_permutations_count :
  ∃ S : Finset (Fin 4 → ℕ), S.card = 9 ∧
    (∀ (x : Fin 4 → ℕ), x ∈ S → 
      (∀ i : Fin 4, x i ∈ {1, 2, 3, 4}) ∧ (x 0 ∈ {1, 2, 3, 4}) ∧ 
      (x 1 ∈ {1, 2, 3, 4}) ∧ (x 2 ∈ {1, 2, 3, 4}) ∧ 
      (x 3 ∈ {1, 2, 3, 4}) ∧ 
      (|x 0 - 1| + |x 1 - 2| + |x 2 - 3| + |x 3 - 4| = 6)) :=
sorry

end nine_permutations_count_l634_634819


namespace greatest_area_difference_l634_634672

theorem greatest_area_difference (l w l' w' : ℕ) (h₁ : 2 * l + 2 * w = 200)
  (h₂ : 2 * l' + 2 * w' = 200) (hl_even : l % 2 = 0) (hw_even : w % 2 = 0)
  (hl'_even : l' % 2 = 0) (hw'_even : w' % 2 = 0) : 
  ∃ l w l' w', (l * w - l' * w').nat_abs = 2304 := 
sorry

end greatest_area_difference_l634_634672


namespace evaluate_expression_l634_634404

noncomputable def w : ℂ := complex.exp (2 * real.pi * complex.I / 13)

theorem evaluate_expression :
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * 
  (3 - w^6) * (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * 
  (3 - w^11) * (3 - w^12) = 797161 :=
begin
  sorry
end

end evaluate_expression_l634_634404


namespace product_of_solutions_is_16_l634_634504

noncomputable def product_of_solutions : ℂ :=
  let solutions : List ℂ := [Complex.polar 2 (Real.angle.ofDegrees 22.5),
                              Complex.polar 2 (Real.angle.ofDegrees 67.5),
                              Complex.polar 2 (Real.angle.ofDegrees 337.5),
                              Complex.polar 2 (Real.angle.ofDegrees 292.5)]
  solutions.foldl (· * ·) 1

theorem product_of_solutions_is_16 :
  product_of_solutions = 16 := 
sorry

end product_of_solutions_is_16_l634_634504


namespace max_area_triangle_exists_point_T_l634_634478

open Real

/-- Definition of the conditions for part 1 --/
def parabola (x y : ℝ) := y^2 = 8 * x
def line_l1 (m : ℝ) (x y : ℝ) := y = x - m

/-- Definition of the conditions for part 2 --/
def line_l2 (k m : ℝ) (x y : ℝ) := y = k * (x - m)

/-- Prove that the area of triangle QAB is maximized to \(\frac{32 \sqrt{3}}{9}\) --/
theorem max_area_triangle (m : ℝ) (hm : m < 0) :
  parabola x y →
  line_l1 m x y →
  m > -2 →
  ∃ A B Q : ℝ × ℝ, 
    Q = (-m, 0) ∧
    max_area (Q, A, B) = 32 * sqrt 3 / 9 :=
sorry

/-- Prove existence and find the coordinates of point T on x-axis for line_l2 --/
theorem exists_point_T (k m : ℝ) (hk : k ≠ 0) :
  parabola x y →
  line_l2 k m x y →
  ∃ T : ℝ × ℝ, T = (-m, 0) :=
sorry

end max_area_triangle_exists_point_T_l634_634478


namespace area_enclosed_eq_area_circle_diameter_perpendicular_division_l634_634620

variables {R r : ℝ} -- Radii
variable (CD : ℝ) -- Length of the perpendicular

-- Assume semicircle diameter is divided into two parts and semicircles are constructed
def area_enclosed_by_semicircles (R r CD : ℝ) : ℝ :=
  0.5 * (Real.pi * R^2 - Real.pi * r^2 - Real.pi * (R - r)^2)

theorem area_enclosed_eq_area_circle_diameter_perpendicular_division 
  (R r CD : ℝ) 
  (h : CD = 2 * r * (R - r)) 
  : area_enclosed_by_semicircles R r CD = Real.pi * CD^2 / 4 := 
sorry

end area_enclosed_eq_area_circle_diameter_perpendicular_division_l634_634620


namespace cyclic_sum_W_l634_634325

-- Define the power of a point with respect to a circle passing through three points
def power_of_point (P : Point) (C1 C2 C3 : Circle) : ℝ := 
  -- Assume some function here that calculates the power of P with respect to circle passing through C1, C2, C3
  sorry

-- Define the area of a triangle given three points
def area_triangle (A B C : Point) : ℝ :=
  -- Assume some function here that calculates the area of triangle ABC
  sorry

-- Define W_a
def W (A B C D : Point) :=
  let Wa := (power_of_point A (circle_through B C D)) * (area_triangle B C D)
  let Wb := (power_of_point B (circle_through A C D)) * (area_triangle A C D)
  let Wc := (power_of_point C (circle_through A B D)) * (area_triangle A B D)
  let Wd := (power_of_point D (circle_through A B C)) * (area_triangle A B C)
  Wa + Wb + Wc + Wd

theorem cyclic_sum_W (A B C D : Point) (h_convex : convex_quadrilateral A B C D):
  W A B C D = 0 :=
by
  -- Proof steps, which are omitted
  sorry

end cyclic_sum_W_l634_634325


namespace gcd_lcm_sum_l634_634776

theorem gcd_lcm_sum : 
  Nat.gcd 48 70 + Nat.lcm 18 45 = 92 := by
  have h1 : Nat.gcd 48 70 = 2 := by
    sorry
  have h2 : Nat.lcm 18 45 = 90 := by
    sorry
  rw [h1, h2]
  sorry

end gcd_lcm_sum_l634_634776


namespace base9_to_base3_conversion_l634_634381

theorem base9_to_base3_conversion : ∃ (x : Fin 9) (y : Fin 9) (z : Fin 9), 
  x = 7 ∧ y = 4 ∧ z = 5 ∧ (745 = x * 9^2 + y * 9^1 + z * 9^0) ∧
  let a : Fin 3 := 2 in
  let b : Fin 3 := 1 in
  let c : Fin 3 := 1 in
  let d : Fin 3 := 1 in
  let e : Fin 3 := 1 in
  let f : Fin 3 := 2 in
  (2 * 3^5 + 1 * 3^4 + 1 * 3^3 + 1 * 3^2 + 1 * 3^1 + 2 * 3^0) = 
  (pow 3 5 * a + pow 3 4 * b + pow 3 3 * c + pow 3 2 * d + pow 3 1 * e + pow 3 0 * f) := 
begin
  sorry
end

end base9_to_base3_conversion_l634_634381


namespace sum_c_seq_2016_eq_l634_634449

noncomputable def a_seq (n : ℕ) : ℝ := n
noncomputable def S_seq (n : ℕ) : ℝ := (n * (n + 1)) / 2
noncomputable def c_seq (n : ℕ) : ℝ := (-1) ^ n * (1 / n + 1 / (n + 1))

theorem sum_c_seq_2016_eq : 
  ∑ i in finset.range 2016, c_seq (i + 1) = -2016 / 2017 :=
sorry

end sum_c_seq_2016_eq_l634_634449


namespace probability_qualified_from_A_is_correct_l634_634548

-- Given conditions:
def p_A : ℝ := 0.7
def pass_A : ℝ := 0.95

-- Define what we need to prove:
def qualified_from_A : ℝ := p_A * pass_A

-- Theorem statement
theorem probability_qualified_from_A_is_correct :
  qualified_from_A = 0.665 :=
by
  sorry

end probability_qualified_from_A_is_correct_l634_634548


namespace not_possible_to_reach_top_right_l634_634378

-- Define the dimensions of the checkerboard
def n : ℕ := 10

-- Representation of the board as a grid of n x n squares.
structure Checkerboard :=
(height width : ℕ)
(h_proper : height = n)
(w_proper : width = n)

-- Define the start and end positions
def start_position : (ℕ × ℕ) := (1, 1)
def end_position : (ℕ × ℕ) := (n, n)

-- Define a move operation which moves token from one square to an adjacent one.
def valid_move (pos1 pos2 : ℕ × ℕ) : Prop :=
(pos1.1 = pos2.1 ∧ (pos1.2 = pos2.2 + 1 ∨ pos1.2 + 1 = pos2.2)) ∨
(pos1.2 = pos2.2 ∧ (pos1.1 = pos2.1 + 1 ∨ pos1.1 + 1 = pos2.1))

-- Define a function checking if a path covers all squares exactly once and ends at the top-right corner
def visiting_each_square_once (path : List (ℕ × ℕ)) : Prop :=
(path.head = start_position) ∧
(path.last = some end_position) ∧
(path.nodup) ∧
(path.length = n * n)

theorem not_possible_to_reach_top_right :
  ¬ ∃ path : List (ℕ × ℕ), visiting_each_square_once path ∧
  (∀ i < path.length - 1, valid_move (path.nth_le i sorry) (path.nth_le (i+1) sorry)) :=
sorry

end not_possible_to_reach_top_right_l634_634378


namespace labor_costs_are_650_l634_634180

/-
  Conditions:
  - Given:
    - Two construction workers each make $100/day.
    - The electrician makes double what a construction worker is paid.
    - The plumber makes 250% of a construction worker's salary.
  - We need to prove:
    The overall labor costs for one day is $650.
-/

noncomputable def wage_construction_worker : ℝ := 100

def wage_electrician : ℝ := 2 * wage_construction_worker

def wage_plumber : ℝ := 2.5 * wage_construction_worker

def total_daily_labor_costs : ℝ := 2 * wage_construction_worker + wage_electrician + wage_plumber

theorem labor_costs_are_650 : total_daily_labor_costs = 650 := by sorry

end labor_costs_are_650_l634_634180


namespace xiangyang_road_length_l634_634225

noncomputable def actual_length_of_xiangyang_road (scale: ℝ) (map_length: ℝ) : ℝ :=
  scale * map_length 

theorem xiangyang_road_length (scale map_length actual_length : ℝ) 
  (h_scale : scale = 10000) 
  (h_map_length : map_length = 10) 
  (h_conversion : 100 cm = 1 m) : 
  (actual_length_of_xiangyang_road scale map_length) / 100 = 1000 :=
by 
  sorry

end xiangyang_road_length_l634_634225


namespace product_sqrt_ineq_l634_634619

theorem product_sqrt_ineq (n : ℕ) (h3 : n ≥ 3) : 
  2 * (√3) * (real.rpow 4 (1/3)) * ∏ i in Ico 4 (n + 1), real.rpow i (1 / (i - 1)) > n :=
sorry

end product_sqrt_ineq_l634_634619


namespace greatest_a_inequality_l634_634189

def σ (n : ℕ) : ℕ := ∑ d in (range (n+1)).filter (λ d, n % d = 0), d
def τ (n : ℕ) : ℕ := (range (n+1)).count (λ d, n % d = 0)

theorem greatest_a_inequality (a : ℝ) (h : a = 3 * Real.sqrt 2 / 4) :
  ∀ (n : ℕ), n > 1 → (σ n : ℝ) / (τ n : ℝ) ≥ a * Real.sqrt n := 
by
  intros n hn
  sorry

end greatest_a_inequality_l634_634189


namespace problem1_problem2_problem3_l634_634963

-- Definitions for conditions in part (1)
def ellipseCondition1 (a b : ℝ) : Prop :=
  ∃ (F1 F2 A Q : ℝ × ℝ),
    F1 = (c, 0) ∧
    F2 = (-c, 0) ∧
    A = (0, b) ∧
    Q = (-b^2 / c, 0) ∧
    2 * (F1.1 - F2.1) + (F2.1 - Q.1) = 0

-- Problem (1): Prove the eccentricity e = 1/2
theorem problem1 (a b : ℝ) (h : ellipseCondition1 a b)
  (assume : a^2 = b^2 + c^2) : 
  a > 0 ∧ b > 0 → ∃ e, e = c / a ∧ e = 1 / 2 := sorry

-- Definitions for conditions in part (2)
def ellipseCondition2 (a b : ℝ) : Prop :=
  ∃ (F2 Q A : ℝ × ℝ),
    F2 = (c, 0) ∧
    Q = (-3 * c, 0) ∧
    A = (0, sqrt (3) * c)

-- Problem (2): Given circle is tangent to a line, find the ellipse equation
theorem problem2 (a b : ℝ) (h : ellipseCondition2 a b)
  (assume : a = 2 * c ∧ b = sqrt (3) * c ∧ c = 1) : 
  a > 0 ∧ b > 0 → 
  ∃ e, ∃ b, 
    e = 1 / 2 ∧
    a = 2 ∧
    b = sqrt(3) ∧ 
    (∀ x y : ℝ, 
      (x, y) ∈ ellipse a b ↔ (x^2 / 4 + y^2 / 3 = 1)) := sorry

-- Problem (3): Prove the maximum area of triangle PMN is 9/2
theorem problem3 (P : ℝ × ℝ) (h : P = (4, 0)) 
  (assume : ∀ M N : ℝ × ℝ, 
    M = ((u, v), u = v / m + 1) ∧ 
    N = ((s, t), s = t / m + 1) ∧ 
    m ∈ ℝ) : 
  ∀ M N : ℝ × ℝ,
  ∃ A : ℝ, A = (3 * sqrt(3) * sqrt(3 * m^2 + 3))/(3 * m^2 + 4))
    and ∀ λ : ℝ,  λ ≥ sqrt(3) → (6 * sqrt(3))/(λ + 1 / λ) ≤ 9/2) := sorry

end problem1_problem2_problem3_l634_634963


namespace Petya_win_prob_is_1_over_256_l634_634999

/-!
# The probability that Petya will win given the conditions in the game "Heap of Stones".
-/

/-- Function representing the probability that Petya will win given the initial conditions.
Petya starts with 16 stones and takes a random number of stones each turn, while the computer
follows an optimal strategy. -/
noncomputable def Petya_wins_probability (initial_stones : ℕ) (random_choices : list ℕ) : ℚ :=
1 / 256

/-- Proof statement: The probability that Petya will win under the given conditions is 1/256. -/
theorem Petya_win_prob_is_1_over_256 : Petya_wins_probability 16 [1, 2, 3, 4] = 1 / 256 :=
sorry

end Petya_win_prob_is_1_over_256_l634_634999


namespace inequality_sqrt_l634_634961

variable (a b c : ℝ)
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_cond : 1 / a + 1 / b + 1 / c = 1)

theorem inequality_sqrt :
  sqrt (a * b + c) + sqrt (b * c + a) + sqrt (c * a + b) ≥ sqrt (a * b * c) + sqrt a + sqrt b + sqrt c :=
  sorry

end inequality_sqrt_l634_634961


namespace joe_total_cars_l634_634563

theorem joe_total_cars (initial_cars: ℕ) (additional_cars: ℕ) :
  initial_cars = 50 ∧ additional_cars = 12 → initial_cars + additional_cars = 62 :=
by
  intros h
  rw [h.1, h.2]
  -- This step just simplifies the left part RHS to the Right part
  sorry

end joe_total_cars_l634_634563


namespace complement_intersection_l634_634101

open Finset

def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 3, 4}
def B : Finset ℕ := {3, 5}

theorem complement_intersection :
  (U \ (A ∩ B)) = {1, 2, 4, 5} :=
by sorry

end complement_intersection_l634_634101


namespace max_shareholder_percentage_l634_634145

theorem max_shareholder_percentage :
  ∃ (p : ℝ), (∀ (B C : fin (99)), B ≠ C → (B ∪ C).card = 66 → p B + p C ≥ 50) → (∀ A: fin 100, p A ≤ 25) := sorry

end max_shareholder_percentage_l634_634145


namespace population_ratio_l634_634697

-- Definitions
def population_z (Z : ℕ) : ℕ := Z
def population_y (Z : ℕ) : ℕ := 2 * population_z Z
def population_x (Z : ℕ) : ℕ := 6 * population_y Z

-- Theorem stating the ratio
theorem population_ratio (Z : ℕ) : (population_x Z) / (population_z Z) = 12 :=
  by 
  unfold population_x population_y population_z
  sorry

end population_ratio_l634_634697


namespace eccentricity_of_hyperbola_l634_634847

/-- Define hyperbola and its conditions -/
variables (a b : ℝ) (ha : a > 0) (hb : b > 0)

/-- Define the hyperbola equation -/
def hyperbola_eq : Prop := ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)

/-- Define the eccentricity of the hyperbola -/
def eccentricity (c : ℝ) : ℝ := c / a

/-- The main theorem to prove the eccentricity of the given hyperbola -/
theorem eccentricity_of_hyperbola : 
  (∃ c : ℝ, (c = real.sqrt (a^2 + b^2)) ∧ c / a = real.sqrt 2) :=
by 
  sorry

end eccentricity_of_hyperbola_l634_634847


namespace max_c_value_l634_634466

open Real

theorem max_c_value (a b c : ℝ) 
  (h1 : 2^a + 2^b ≠ 2^(a + b)) 
  (h2 : 2^a + 2^b + 2^c = 2^(a + b + c)) : 
  c ≤ 2 - log 2 3 := 
sorry

end max_c_value_l634_634466


namespace estimate_total_fish_l634_634665

theorem estimate_total_fish (n1 n2 m : ℕ) (h₁ : n1 = 40) (h₂ : n2 = 100) (h₃ : m = 5) : 
  let x := (n2 * n1) / m in x = 800 :=
by
  sorry

end estimate_total_fish_l634_634665


namespace total_cost_price_correct_l634_634354

-- Define the cost prices for the apple, orange, and banana
def cost_price_apple : ℝ := 16 * (6/5)
def cost_price_orange : ℝ := 20 * (5/4)
def cost_price_banana : ℝ := 12 * (4/3)

-- Define the total cost price as the sum of the cost prices of the apple, orange, and banana
def total_cost_price : ℝ := cost_price_apple + cost_price_orange + cost_price_banana

-- The theorem to prove
theorem total_cost_price_correct :
  total_cost_price = 60.2 := by
  sorry

end total_cost_price_correct_l634_634354


namespace proof_statement_l634_634377

noncomputable def problem_statement : Prop :=
  ∀ (ω Ω : Circle) (A B M P Q : Point) 
    (ℓ_P ℓ_Q : Line) (H : MeetsAt ω Ω A B)
    (H_M : MidpointOfArc M A B ω) (H_MP : Chord MP ω P)
    (H_MP_Ω : IntersectsAtChord MP Ω Q) 
    (H_tangentP : TangentAt ℓ_P ω P)
    (H_tangentQ : TangentAt ℓ_Q Ω Q),
  let X : Point := Intersection ℓ_P AB in
  let Y : Point := Intersection ℓ_Q AB in
  let Z : Point := Intersection ℓ_P ℓ_Q in
  ∃ (circXYZ : Circle), Circumcircle X Y Z circXYZ ∧ TangentTo circXYZ Ω 

theorem proof_statement : problem_statement := 
  sorry

end proof_statement_l634_634377


namespace fish_pond_estimation_l634_634667

/-- To estimate the number of fish in a pond, 40 fish were first caught and marked, then released back into the pond.
After the marked fish were completely mixed with the rest of the fish in the pond, 100 fish were caught again, and 5 of them were found to be marked. 
We need to prove that the total number of fish in the pond is 800. -/
theorem fish_pond_estimation
  (marked_released : ℕ)
  (fish_caught : ℕ)
  (marked_found : ℕ)
  (total_fish : ℕ)
  (h_marked_released : marked_released = 40)
  (h_fish_caught : fish_caught = 100)
  (h_marked_found : marked_found = 5)
  (h_total_fish : total_fish = 800) :
  fish_caught / marked_found = total_fish / marked_released :=
by
  rw [h_marked_released, h_fish_caught, h_marked_found, h_total_fish]
  sorry

end fish_pond_estimation_l634_634667


namespace correct_point_in_region_l634_634651

theorem correct_point_in_region :
  (3 + 2 * (1 : ℝ) < 6) ∧ ¬ (3 + 2 * (0 : ℝ) < 6) ∧ ¬ (3 + 2 * (2 : ℝ) < 6) ∧ ¬ (3 + 2 * (0 : ℝ) < 6) := 
by {
  split,
  {
    -- (1, 1) satisfies the inequation.
    sorry,
  },
  split,
  {
    -- (0, 2) does not satisfy the inequation 3 + 2 * 2 < 6.
    sorry,
  },
  split,
  {
    -- (2, 0) does not satisfy the inequation 3 + 2 * 0 < 6.
    sorry,
  },
  {
    -- (0, 0) does not satisfy the inequation 3 + 2 * 0 < 6.
    sorry,
  }
}

end correct_point_in_region_l634_634651


namespace fraction_equivalence_l634_634121

variable {m n p q : ℚ}

theorem fraction_equivalence
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5) :
  m / q = 1 :=
by {
  sorry
}

end fraction_equivalence_l634_634121


namespace gambler_win_percentage_l634_634718

theorem gambler_win_percentage :
  ∀ (T W play_extra : ℕ) (P_win_extra P_week P_current P_required : ℚ),
    T = 40 →
    P_win_extra = 0.80 →
    play_extra = 40 →
    P_week = 0.60 →
    P_required = 48 →
    (W + P_win_extra * play_extra = P_required) →
    (P_current = (W : ℚ) / T * 100) →
    P_current = 40 :=
by
  intros T W play_extra P_win_extra P_week P_current P_required h1 h2 h3 h4 h5 h6 h7
  sorry

end gambler_win_percentage_l634_634718


namespace min_distance_sum_l634_634069

-- Definitions of given lines and parabola
def line_l1 (x y : ℝ) : Prop := 4 * x - 3 * y + 16 = 0
def line_l2 (x : ℝ) : Prop := x = -1
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Definition of distances d1 and d2 from point P (on parabola) to the lines l1 and l2
def distance (x1 y1 : ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
  match line with
  | (λ x y, A * x + B * y + C = 0) => abs (A * x1 + B * y1 + C) / sqrt (A^2 + B^2)
  | _ => 0

-- Focus of the parabola
def focus : (ℝ × ℝ) := (1, 0)

-- Minimum value of distances d1 + d2
theorem min_distance_sum : ∀ (P : ℝ × ℝ),
  parabola P.1 P.2 →
  distance P.1 P.2 line_l1 + distance P.1 P.2 line_l2 = distance focus.1 focus.2 line_l1 :=
sorry

end min_distance_sum_l634_634069


namespace solution_set_of_inequality_l634_634136

theorem solution_set_of_inequality (a t : ℝ) (h1 : ∀ x : ℝ, x^2 - 2 * a * x + a > 0) : 
  a > 0 ∧ a < 1 → (a^(2*t + 1) < a^(t^2 + 2*t - 3) ↔ -2 < t ∧ t < 2) :=
by
  intro ha
  have h : (0 < a ∧ a < 1) := sorry
  exact sorry

end solution_set_of_inequality_l634_634136


namespace constant_value_AP_AQ_l634_634805

noncomputable def ellipse_trajectory (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def circle_O (x y : ℝ) : Prop :=
  (x^2 + y^2) = 12 / 7

theorem constant_value_AP_AQ (x y : ℝ) (h : circle_O x y) :
  ∃ (P Q : ℝ × ℝ), ellipse_trajectory (P.1) (P.2) ∧ ellipse_trajectory (Q.1) (Q.2) ∧ 
  ((P.1 - x) * (Q.1 - x) + (P.2 - y) * (Q.2 - y)) = - (12 / 7) :=
sorry

end constant_value_AP_AQ_l634_634805


namespace amount_per_friend_correct_l634_634605

def initial_amount : ℝ := 10.50
def spent_on_sweets : ℝ := 3.70
def remaining_amount := initial_amount - spent_on_sweets
def amount_given_per_friend := remaining_amount / 2

theorem amount_per_friend_correct :
  amount_given_per_friend = 3.40 := 
by 
  sorry

end amount_per_friend_correct_l634_634605


namespace range_of_f_in_interval_l634_634477

noncomputable def f (ω x : ℝ) : ℝ := (Real.sin(ω * x))^2 + Real.sqrt 3 * Real.sin(ω * x) * Real.sin(ω * x + Real.pi / 2)

theorem range_of_f_in_interval (ω : ℝ) (hω : ω > 0) (h_period : ∀ x, f ω x = f ω (x + Real.pi)) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 → 0 ≤ f ω x ∧ f ω x ≤ 3 / 2) :=
sorry

end range_of_f_in_interval_l634_634477


namespace total_number_of_seats_sum_of_first_20_terms_l634_634916

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℕ := n + 1

-- Define the sum of the sequence a_n for n from 1 to k
def sum_of_sequence (k : ℕ) : ℕ :=
  (k * (2 + k)) / 2

theorem total_number_of_seats : sum_of_sequence 20 = 230 := by
  -- (The proof would be written here)
  sorry

-- Define the sequence b_n where b_n = a_n / (n * (n+1)^2)
noncomputable def b (n : ℕ) : ℚ := a n / (n * (n + 1) ^ 2)

-- Define the sum of the first k terms of sequence b_n
def sum_b_sequence (k : ℕ) : ℚ :=
  ∑ i in Finset.range (k + 1), b (i + 1)

theorem sum_of_first_20_terms : sum_b_sequence 20 = 20 / 21 := by
  -- (The proof would be written here)
  sorry

end total_number_of_seats_sum_of_first_20_terms_l634_634916


namespace number_of_solutions_l634_634867

theorem number_of_solutions : 
  {m : ℤ | m ≠ 0 ∧ (1 / (|m| : ℝ) ≥ 1 / 5)}.toFinset.card = 10 := 
begin
  sorry
end

end number_of_solutions_l634_634867


namespace find_m_l634_634579

variable {y m : ℝ} -- define variables y and m in the reals

-- define the logarithmic conditions
axiom log8_5_eq_y : log 8 5 = y
axiom log2_125_eq_my : log 2 125 = m * y

-- state the theorem to prove m equals 9
theorem find_m (log8_5_eq_y : log 8 5 = y) (log2_125_eq_my : log 2 125 = m * y) : m = 9 := by
  sorry

end find_m_l634_634579


namespace matrix_B3_is_zero_unique_l634_634582

theorem matrix_B3_is_zero_unique (B : Matrix (Fin 2) (Fin 2) ℝ) (h : B^4 = 0) :
  ∃! (B3 : Matrix (Fin 2) (Fin 2) ℝ), B3 = B^3 ∧ B3 = 0 := sorry

end matrix_B3_is_zero_unique_l634_634582


namespace first_train_length_correct_l634_634673

noncomputable def length_first_train (speed_first_kmph speed_second_kmph : ℕ) (time_seconds : ℝ) (length_second_m : ℝ) : ℝ :=
  let relative_speed_mps := (speed_first_kmph + speed_second_kmph) * 1000 / 3600
  in (relative_speed_mps * time_seconds) - length_second_m

theorem first_train_length_correct :
  length_first_train 80 65 7.596633648618456 165 = 141.019 :=
by
  sorry

end first_train_length_correct_l634_634673


namespace determine_functionality_button1_l634_634659

noncomputable def describeSystem : Type :=
  { buttons : Fin 10 → Bool // ∃ i j k l m n, buttons i ∧ buttons j ∧ buttons k ∧ buttons l ∧ buttons m ∧ ¬ buttons n }

def lamp_lights_up (buttons: Fin 10 → Bool) (i j k : Fin 10) : Bool :=
  buttons i || buttons j || buttons k

theorem determine_functionality_button1 : 
  ∀ (system: describeSystem),
  ∃ (n_attempts: Nat) (attempts: Fin n_attempts → (Fin 10 × Fin 10 × Fin 10)), 
  (n_attempts ≤ 9) ∧ 
  (lamp_lights_up system.val 1 (attempts 0).1 (attempts 0).2) && 
  (lamp_lights_up system.val 1 (attempts 1).1 (attempts 1).2) && 
  (lamp_lights_up system.val 1 (attempts 2).1 (attempts 2).2) → system.val 1 = true :=
by
  sorry

end determine_functionality_button1_l634_634659


namespace area_of_right_triangle_with_given_sides_l634_634887

theorem area_of_right_triangle_with_given_sides :
  let f : (ℝ → ℝ) := fun x => x^2 - 7 * x + 12
  let a := 3
  let b := 4
  let c := sqrt 7
  let hypotenuse := max a b
  let leg := min a b
in (hypotenuse = 4 ∧ leg = 3 ∧ (f(3) = 0) ∧ (f(4) = 0) → 
   (∃ (area : ℝ), (area = 6 ∨ area = (3 * sqrt 7) / 2))) :=
by
  intros
  sorry

end area_of_right_triangle_with_given_sides_l634_634887


namespace change_combination_count_l634_634952

theorem change_combination_count :
  ∃ k : ℕ, (∑ d in finset.range 118, 
           ∑ q in finset.range 48,
           if h : n = 240 - 2 * d - 5 * q ∧ 1 ≤ d ∧ 1 ≤ q ∧ n > 0 then 1 else 0) = k := 
begin
  sorry
end

end change_combination_count_l634_634952


namespace smallest_three_digit_congruent_one_mod_37_l634_634310

theorem smallest_three_digit_congruent_one_mod_37 :
  ∃ n : ℕ, (100 ≤ n) ∧ (n < 1000) ∧ (n % 37 = 1) ∧ (∀ m : ℕ, (100 ≤ m) ∧ (m < 1000) ∧ (m % 37 = 1) → n ≤ m) :=
begin
  use 112,
  split,
  { exact le_refl 112, },
  split,
  { linarith, },
  split,
  { rw nat.mod_eq_of_lt trivial, linarith, },
  intro m,
  intros h1 h2 h3,
  rw nat.mod_eq_of_lt h2 at h3,
  exact nat.le_of_dvd (sub_pos_of_lt h1) h3,
end

end smallest_three_digit_congruent_one_mod_37_l634_634310


namespace ellipse_equation_dot_product_constant_l634_634803

-- Given circles F₁ and F₂, and Circle O
noncomputable def circle_F1 (r : ℝ) : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in (x + 1)^2 + y^2 = r^2 }

noncomputable def circle_F2 (r : ℝ) : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in (x - 1)^2 + y^2 = (4 - r)^2 }

def circle_O : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in x^2 + y^2 = (12 / 7) }

-- Define the ellipse E
noncomputable def ellipse_E : set (ℝ × ℝ) :=
  { p | let x := p.1, y := p.2 in (x^2) / 4 + (y^2) / 3 = 1 }

-- Define the proof problem for part (1)
theorem ellipse_equation (1 ≤ r ∧ r ≤ 3) :
  ∀ (p : ℝ × ℝ), p ∈ circle_F1 r → p ∈ circle_F2 r → p ∈ ellipse_E :=
sorry

-- Define the vectors and their dot product
noncomputable def vector_dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the proof problem for part (2)
theorem dot_product_constant (A P Q : ℝ × ℝ) (r : ℝ) (hA : A ∈ circle_O)
(hPA : P ∈ ellipse_E) (hQA : Q ∈ ellipse_E) :
  vector_dot_product (⟨P.1 - A.1, P.2 - A.2⟩) (⟨Q.1 - A.1, Q.2 - A.2⟩) = - (12 / 7) :=
sorry

end ellipse_equation_dot_product_constant_l634_634803


namespace candidate_failed_by_45_marks_l634_634153

-- Define the main parameters
def passing_percentage : ℚ := 45 / 100
def candidate_marks : ℝ := 180
def maximum_marks : ℝ := 500
def passing_marks : ℝ := passing_percentage * maximum_marks
def failing_marks : ℝ := passing_marks - candidate_marks

-- State the theorem to be proved
theorem candidate_failed_by_45_marks : failing_marks = 45 := by
  sorry

end candidate_failed_by_45_marks_l634_634153


namespace probability_perfect_square_or_cube_sum_of_two_8_sided_dice_l634_634680

-- Definitions of the problem's conditions.
def is_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

def is_perfect_cube (n : ℕ) : Prop :=
  n = 8

-- The probability that the sum is a perfect square or a perfect cube.
theorem probability_perfect_square_or_cube_sum_of_two_8_sided_dice :
  (probability : ℚ) =
  let total_outcomes : ℚ := 64
  let favorable_outcomes : ℚ := 19
  
  favorable_outcomes / total_outcomes = 19 / 64 :=
sorry

end probability_perfect_square_or_cube_sum_of_two_8_sided_dice_l634_634680


namespace range_of_a_l634_634139

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (4 * (x - 1) > 3 * x - 1) ∧ (5 * x > 3 * x + 2 * a) ↔ (x > 3)) ↔ (a ≤ 3) :=
by
  sorry

end range_of_a_l634_634139


namespace slopes_product_ellipse_l634_634542

theorem slopes_product_ellipse (a b : ℝ) (h : a > b > 0) 
  {x y m n : ℝ} (hA : m^2 / a^2 + n^2 / b^2 = 1) 
  (hC : x^2 / a^2 + y^2 / b^2 = 1) 
  (h_diff : (x, y) ≠ (m, n) ∧ (x, y) ≠ (-m, -n)) :
  ((y - n) / (x - m)) * ((y + n) / (x + m)) = -(b^2 / a^2) :=
sorry

end slopes_product_ellipse_l634_634542


namespace find_abs_diff_of_segments_l634_634730

noncomputable def semi_perimeter (a b c d : ℕ) := (a + b + c + d) / 2
noncomputable def area_brahmagupta (s a b c d : ℕ) := 
  real.sqrt ((s - a) * (s - b) * (s - c) * (s - d))

theorem find_abs_diff_of_segments 
  (a b c d : ℕ) (s : ℕ) 
  (x y : real) 
  (h_s : s = semi_perimeter a b c d) 
  (h_area : (area_brahmagupta s a b c d) = 24000) 
  (h_inradius : (h_area = r * s)) 
  (h_eq_sum : x + y = c) 
  (h_eq_diff : x = y + 400 / 3) : 
  abs (x - y) = 400 / 3 := 
sorry

end find_abs_diff_of_segments_l634_634730


namespace diagonals_in_convex_15gon_l634_634387

theorem diagonals_in_convex_15gon : 
  let n := 15 
  in (n * (n - 3)) / 2 = 90 := by
  let n := 15
  show (n * (n - 3)) / 2 = 90 from
    sorry

end diagonals_in_convex_15gon_l634_634387


namespace imaginary_part_of_z_l634_634503

def z : ℂ := complex.abs (complex.mk (-1) (real.sqrt 3)) + ((1 : ℂ) / (complex.mk 1 1))

theorem imaginary_part_of_z :
  z.im = -1 / 2 := by
sorry

end imaginary_part_of_z_l634_634503


namespace value_range_f_l634_634654

open Real

-- Define the function y = (x^2 - 1) / (x^2 + 1).
def f (x : ℝ) : ℝ :=
  (x^2 - 1) / (x^2 + 1)

-- The theorem stating that the value range of the function f is [-1, 1).
theorem value_range_f : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ (-1 ≤ y ∧ y < 1) :=
by
  sorry

end value_range_f_l634_634654


namespace gridtown_tom_vs_jane_l634_634907

/-- In Gridtown, all streets are 30 feet wide, and the blocks are square with each side measuring
500 feet. Jane runs around the block on the 500-foot side of the street, while Tom runs on the
opposite side of the street. Prove that for every lap around the block, Tom runs 1030π feet more
than Jane. -/
theorem gridtown_tom_vs_jane : 
  ∀ (w : ℕ) (s : ℕ), 
  w = 30 → 
  s = 500 → 
  let r_tom := s + w/2 in
  let d_tom := 2 * r_tom in
  let dist_tom_corner := (d_tom * Real.pi) / 4 in 
  4 * dist_tom_corner = 1030 * Real.pi :=
by
  intros w s w_eq s_eq r_tom d_tom dist_tom_corner
  rw [w_eq, s_eq]
  sorry

end gridtown_tom_vs_jane_l634_634907


namespace problem_1_problem_2_l634_634834

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2^(2 * x) - 2^x * a - (a + 1)

theorem problem_1 (a : ℝ) (x : ℝ) (h : a = 2) : f x 2 < 0 ↔ x < Real.log 2 3 := 
by
  sorry

theorem problem_2 (a : ℝ) (x : ℝ) (hx : 2^(2 * x) - 2^x * a - (a + 1) = 0) : a > -1 :=
by
  sorry

end problem_1_problem_2_l634_634834


namespace find_the_number_l634_634796

variable (x : ℕ)

theorem find_the_number (h : 43 + 3 * x = 58) : x = 5 :=
by 
  sorry

end find_the_number_l634_634796


namespace slices_per_person_eq_three_l634_634658

variables (num_people : ℕ) (slices_per_pizza : ℕ) (num_pizzas : ℕ)

theorem slices_per_person_eq_three (h1 : num_people = 18) (h2 : slices_per_pizza = 9) (h3 : num_pizzas = 6) : 
  (num_pizzas * slices_per_pizza) / num_people = 3 :=
sorry

end slices_per_person_eq_three_l634_634658


namespace sum_arithmetic_sequence_l634_634442

noncomputable def sequence (k : ℕ) : ℝ :=
  if k = 0 then 0 else 2 + (k - 1) * (1 / 3)

theorem sum_arithmetic_sequence (n : ℕ) (h : n > 0) :
  (Finset.range n).sum (λ k, sequence (k + 1)) = n * (n + 11) / 6 :=
by
  sorry

end sum_arithmetic_sequence_l634_634442


namespace triangle_ABC_angle_C_triangle_ABC_side_c_l634_634142

variables {a b c A B C : ℝ}

-- Define the conditions given in the problem
def condition1 (A B : ℝ) : Prop := 
  sin (A/2 - B/2)^2 + sin A * sin B = (2 + real.sqrt 2) / 4

-- Define the statement to prove the measure of angle C
theorem triangle_ABC_angle_C (A B : ℝ) 
  (h1 : ∀ A B, condition1 A B) : 
  A + B + C = real.pi ∧ cos C = real.sqrt 2 / 2 → C = real.pi / 4 := 
sorry

-- Define the conditions for side lengths and area
def area_condition (a b C : ℝ) : Prop :=
  6 = (1/2) * a * b * sin C

-- Define the statement to prove the side length c
theorem triangle_ABC_side_c (a b c : ℝ)
  (h2 : b = 4) 
  (h3 : area_condition a b (real.pi/4)) : 
  c = real.sqrt 10 :=
sorry

end triangle_ABC_angle_C_triangle_ABC_side_c_l634_634142


namespace minute_hand_angle_45min_l634_634826

theorem minute_hand_angle_45min
  (duration : ℝ)
  (h1 : duration = 45) :
  (-(3 / 4) * 2 * Real.pi = - (3 * Real.pi / 2)) :=
by
  sorry

end minute_hand_angle_45min_l634_634826


namespace lg_sin_alpha_l634_634809

-- Define the problem conditions
variables (α m n : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2)
variable (h3 : Real.log10 (1 + Real.cos α) = m)
variable (h4 : Real.log10 (1 / (1 - Real.cos α)) = n)

-- Define the target statement
theorem lg_sin_alpha (α m n : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.log10 (1 + Real.cos α) = m)
  (h4 : Real.log10 (1 / (1 - Real.cos α)) = n) : 
  Real.log10 (Real.sin α) = 1 / 2 * (m - 1 / n) := 
sorry

end lg_sin_alpha_l634_634809


namespace math_problem_l634_634572

noncomputable def num_pairs_sets : ℕ :=
  let C := {C : finset ℕ // 1 ≤ C.card ∧ C.card ≤ 14 ∧ ∀ a ∈ C, C.card ≠ a}
  let D := {D : finset ℕ // ∃ k, k.1 = 15 ∧ k.2 = C ∪ D ∧ (C ∩ D).card = 0 ∧ ∀ b ∈ D, D.card ≠ b}
  Nat.card C

theorem math_problem (M : ℕ) : M = 6476 :=
by
  have H1 : M = num_pairs_sets := sorry  -- restate the number calculation from the problem
  have H2 : num_pairs_sets = 6476 := sorry -- prove the computation
  rw [H1, H2]
  rfl

end math_problem_l634_634572


namespace middle_guards_hours_l634_634720

theorem middle_guards_hours (h1 : ∀ (first_guard_hours last_guard_hours total_night_hours : ℕ),
  first_guard_hours = 3 ∧ last_guard_hours = 2 ∧ total_night_hours = 9) : ∃ middle_guard_hours : ℕ,
  let remaining_hours := 9 - (3 + 2) in
  let middle_guard_hours := remaining_hours / 2 in
  middle_guard_hours = 2 := by
  sorry

end middle_guards_hours_l634_634720


namespace recruit_b_l634_634671

section
open BigOperators

-- Define the scores of A and B
def scoresA := [93, 91, 80, 92, 95, 89, 88, 97, 95, 93]
def scoresB := [90, 92, 88, 92, 90, 90, 84, 96, 94, 92]

-- Define a function to remove the highest and lowest scores
def remove_high_and_low (scores : List ℕ) : List ℕ :=
  let min_val := scores.minimum?
  let max_val := scores.maximum?
  scores.filter (fun x => x ≠ min_val.getD x).filter (fun x => x ≠ max_val.getD x)

-- Define the adjusted scores of A and B
def adjustedA := remove_high_and_low scoresA
def adjustedB := remove_high_and_low scoresB

-- Calculate the average of a list of scores
noncomputable def average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / (scores.length : ℚ)

def averageA := average adjustedA -- 92
def averageB := average adjustedB -- 91

-- Define the written test scores
def writtenA := 92
def writtenB := 94

-- Define the weighted combined score
noncomputable def combined_score (written : ℕ) (interview : ℚ) : ℚ :=
  0.6 * (written : ℚ) + 0.4 * interview

-- Calculate the combined scores
def combinedA := combined_score writtenA averageA
def combinedB := combined_score writtenB averageB

-- Prove that A's average interview score is greater, but B's combined score is greater, hence B should be recruited
theorem recruit_b : averageA > averageB ∧ combinedB > combinedA :=
by
  sorry
end

end recruit_b_l634_634671


namespace b_limit_to_zero_l634_634637

open Filter Real

variable (a : ℕ → ℕ) (n : ℕ)

def b (n : ℕ) : ℝ := (∏ i in Finset.range (n + 1), a i) / (2 ^ a n)

theorem b_limit_to_zero (h_an_infty : ∀ ε > 0, ∃ N, ∀ n ≥ N, (a n : ℝ) > 1/ε) (h_b : Filter.Tendsto (fun n => b a n) Filter.atTop (nhds 0)) :
  Filter.Tendsto (fun n => b a n) Filter.atTop (nhds 0) :=
sorry

end b_limit_to_zero_l634_634637


namespace points_lie_on_quadratic_l634_634568

def is_quadratic (P : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, P x = a * x^2 + b * x + c

variables (P₁ P₂ P₃ P₄ : ℝ → ℝ)
variables (q₁ q₂ q₃ q₄ : ℝ × ℝ)
variables (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ)
variables (c₁ c₂ c₃ c₄ : ℝ)

-- Define the tangency conditions
axiom tangency_cond1 : ∀ x, P₁ x - P₂ x = c₁ * (x - x₁)^2
axiom tangency_cond2 : ∀ x, P₂ x - P₃ x = c₂ * (x - x₂)^2
axiom tangency_cond3 : ∀ x, P₃ x - P₄ x = c₃ * (x - x₃)^2
axiom tangency_cond4 : ∀ x, P₄ x - P₁ x = c₄ * (x - x₄)^2

-- Distinct x-coordinates
axiom distinct_x_coords : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄

-- Prove that q₁, q₂, q₃, q₄ lie on a quadratic polynomial
theorem points_lie_on_quadratic :
  is_quadratic P₁ ∧ is_quadratic P₂ ∧ is_quadratic P₃ ∧ is_quadratic P₄ →
  y₁ = P₁ x₁ ∧ y₂ = P₂ x₂ ∧ y₃ = P₃ x₃ ∧ y₄ = P₄ x₄ →
  ∃ (f : ℝ → ℝ), is_quadratic f ∧ ∀ i ∈ {q₁, q₂, q₃, q₄}, f i.1 = i.2 :=
sorry

end points_lie_on_quadratic_l634_634568


namespace minimum_weights_divisible_l634_634308

theorem minimum_weights_divisible (S : Set ℕ) : 
  (∀ n : ℕ, (n < 9 → ¬ (n ∈ S ∧ (∃ l : List (Fin 3) → l.length = n ∧ S.sum l = 60 * m) →
  (∃ l : List (Fin 4) → l.length = n ∧ S.sum l = 60 * m) →
  (∃ l : List (Fin 5) → l.length = n ∧ S.sum l = 60 * m)))) ∧ 
  ∃ S : Set ℕ, S.size ≥ 9 ∧ 
  ∃ l₃ : List (Fin 3), l₄ : List (Fin 4), l₅ : List (Fin 5), 
    l₃.sum = 60 * m ∧ l₄.sum = 60 * m ∧ l₅.sum = 60 * m 
    → n = 9 :=
sorry

end minimum_weights_divisible_l634_634308


namespace major_axis_length_of_ellipse_l634_634260

theorem major_axis_length_of_ellipse (x y : ℝ) (h : 16 * x^2 + 9 * y^2 = 144) : 8 := by
  sorry

end major_axis_length_of_ellipse_l634_634260


namespace monotonic_intervals_l634_634843

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x - (m - 9) * x

theorem monotonic_intervals (m : ℝ) :
  (m > -3 →
    (∀ x, g' x m > 0 ↔ x < 1 - sqrt (9 + 3 * m) / 3 ∨ x > 1 + sqrt (9 + 3 * m) / 3) ∧
    (∀ x, g' x m < 0 ↔ 1 - sqrt (9 + 3 * m) / 3 ≤ x ∧ x ≤ 1 + sqrt (9 + 3 * m) / 3)) ∧
  (m = -3 →
    (∀ x, g' x m > 0) ∧
    (∀ x, g' x m < 0 → false)) ∧
  (m < -3 →
    (∀ x, g' x m > 0) ∧
    (∀ x, g' x m < 0 → false)) :=
by
  sorry

end monotonic_intervals_l634_634843


namespace part1_prove_a_and_sinC_part2_prove_cos_2A_minus_pi_over_6_l634_634906

section TriangleProofProblem

-- Variables and constants
variables (A B C : ℝ) (a b c : ℝ)
constants (sinA cosA sinC : ℝ) (area : ℝ)

-- Hypotheses
hypothesis h1 : cosA = -1 / 4
hypothesis h2 : area = 3 * Real.sqrt 15
hypothesis h3 : b - c = 2

-- Prove the value of a and sin C
theorem part1_prove_a_and_sinC (A B C a b c : ℝ) (sinA cosA sinC : ℝ) 
  (area : ℝ) (h1 : cosA = -1 / 4) (h2 : area = 3 * Real.sqrt 15) (h3 : b - c = 2) :
  a = 8 ∧ sinC = Real.sqrt 15 / 8 :=
sorry

-- Prove the value of cos (2A - π / 6)
theorem part2_prove_cos_2A_minus_pi_over_6 (A : ℝ) (cosA : ℝ) (sinA : ℝ)
  (h1 : cosA = -1 / 4) :
  Real.cos (2 * A - Real.pi / 6) = -(Real.sqrt 15 + 7 * Real.sqrt 3) / 16 :=
sorry

end TriangleProofProblem

end part1_prove_a_and_sinC_part2_prove_cos_2A_minus_pi_over_6_l634_634906


namespace x_intercepts_of_parabola_l634_634114

theorem x_intercepts_of_parabola : 
  (∃ y : ℝ, x = -3 * y^2 + 2 * y + 2) → ∃ y : ℝ, y = 0 ∧ x = 2 ∧ ∀ y' ≠ 0, x ≠ -3 * y'^2 + 2 * y' + 2 :=
by
  sorry

end x_intercepts_of_parabola_l634_634114


namespace quadratic_y_intercept_l634_634254

theorem quadratic_y_intercept :
  (∀ x : ℝ, y = x^2 - 5 * x + 1) → (∃ y : ℝ, y = 1 ∧ (0, 1) = (0, y)) :=
by
  intro h
  use 1
  split
  exact rfl
  reflexivity

end quadratic_y_intercept_l634_634254


namespace cos_225_sin_225_l634_634761

theorem cos_225 (θ : Real) (hθ : θ = 225 * Real.angle.of_deg) : Real.cos θ = - Real.sqrt 2 / 2 := by
  sorry

theorem sin_225 (θ : Real) (hθ : θ = 225 * Real.angle.of_deg) : Real.sin θ = - Real.sqrt 2 / 2 := by
  sorry

end cos_225_sin_225_l634_634761


namespace tangent_line_exists_l634_634902

noncomputable def f (x : ℝ) : ℝ := x^3 - x
noncomputable def g (a x : ℝ) : ℝ := x^2 - a^2 + a

theorem tangent_line_exists (a : ℝ) :
  (∃ m b : ℝ, ∃ x₁ x₂ : ℝ, 
    m = 3 * x₁^2 - 1 ∧
    b = x₁^3 - x₁ - (3 * x₁^2 - 1) * x₁ ∧
    m = 2 * x₂ ∧
    b = x₂^2 - a^2 + a - 2 * x₂ * x₂) ↔ 
  (a ≥ (1 - real.sqrt 5) / 2 ∧ a ≤ (1 + real.sqrt 5) / 2) := sorry

end tangent_line_exists_l634_634902


namespace lunks_for_apples_l634_634126

/-- 
If 4 lunks can be traded for 5 kunks, and 7 kunks will buy 10 apples, 
prove that 7 lunks are needed to purchase one dozen (12) apples.
-/
theorem lunks_for_apples 
  (trade : ℚ := 4 / 5)
  (cost : ℚ := 7 / 10)
  (apples : ℚ := 12) : 
  ceil ((apples * cost) * trade) = 7 :=
sorry

end lunks_for_apples_l634_634126


namespace correct_statements_B_and_D_l634_634690

-- Conditions for part B
def xi_sim_N (μ σ: ℝ) := sorry  -- A placeholder definition for the normal distribution
variable {xi: ℝ → Prop}
axiom xi_normal (P_xi_lt_6: P xi < 6 = 0.84): 
  P(3 < xi < 6) = 0.34 := sorry

-- Conditions for part D
variable (x y: ℝ)
def linearly_related (x y: ℝ) := ∃ a b, y = a * x + b
def regression_equation (a: ℝ) := 0.4 * x + a
variables (x_bar y_bar: ℝ) (a_hat: ℝ)
axiom sample_means (h1: x_bar = 4) (h2: y_bar = 3.7):
  regression_equation a_hat x_bar = y_bar := sorry

theorem correct_statements_B_and_D:
  P(3 < xi < 6) = 0.34 ∧ a_hat = 2.1 := by
  sorry

end correct_statements_B_and_D_l634_634690


namespace diagonals_not_parallel_to_sides_in_regular_32gon_l634_634863

theorem diagonals_not_parallel_to_sides_in_regular_32gon :
  ∀ (n : ℕ), n = 32 -> 
  let total_diagonals := n * (n - 3) / 2,
      pairs_of_parallel_sides := n / 2,
      diagonals_per_pair := (n - 4) / 2,
      parallel_diagonals := pairs_of_parallel_sides * diagonals_per_pair
  in total_diagonals - parallel_diagonals = 240 :=
begin
  intros n hn,
  rw hn,
  let total_diagonals := 32 * (32 - 3) / 2,
  let pairs_of_parallel_sides := 32 / 2,
  let diagonals_per_pair := (32 - 4) / 2,
  let parallel_diagonals := pairs_of_parallel_sides * diagonals_per_pair,
  calc 
    total_diagonals - parallel_diagonals 
      = 32 * 29 / 2 - (16 * 14) : by rfl
  ... = 464 - 224 : by norm_num
  ... = 240 : by norm_num,
end

end diagonals_not_parallel_to_sides_in_regular_32gon_l634_634863


namespace value_of_a4_l634_634549

open Nat

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = 2 * a n + 1

theorem value_of_a4 (a : ℕ → ℕ) (h : sequence a) : a 4 = 23 :=
by
  -- Proof to be provided or implemented
  sorry

end value_of_a4_l634_634549


namespace find_angle_A_find_bc_range_l634_634519

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  (c * (a * Real.cos B - (1/2) * b) = a^2 - b^2) ∧ (A = Real.arccos (1/2))

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h : triangle_problem a b c A B C) :
  A = Real.pi / 3 := 
sorry

theorem find_bc_range (a b c : ℝ) (A B C : ℝ) (h : triangle_problem a b c A B C) (ha : a = Real.sqrt 3) :
  b + c ∈ Set.Icc (Real.sqrt 3) (2 * Real.sqrt 3) := 
sorry

end find_angle_A_find_bc_range_l634_634519


namespace intersection_a_b_l634_634452

-- Definitions of sets A and B
def A : Set ℝ := {x | -2 < x ∧ x ≤ 2}
def B : Set ℝ := {-2, -1, 0}

-- The proof problem
theorem intersection_a_b : A ∩ B = {-1, 0} :=
by
  sorry

end intersection_a_b_l634_634452


namespace expression_for_a_expression_for_T_l634_634468

-- Definition of the sequence sum S_n
def S (n : ℕ) : ℕ := n^2 + n

-- Definition of the sequence term a_n
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | k+1 => S (k+1) - S k

-- Definition of the sum T_n for the sequence {sqrt(2)^a_n} == {2^n}
def T (n : ℕ) : ℕ :=
  2^(n + 1) - 2

-- Theorem to prove the expression for a_n
theorem expression_for_a (n : ℕ) : a n = 2 * n := by
  induction n with
  | zero => rfl  -- base case
  | succ n => -- inductive step
    calc
      a (n + 1) = S (n + 1) - S n := rfl
             ... = (n + 1)^2 + (n + 1) - (n^2 + n) := by rw [S]
             ... = 2 * (n + 1) := by linarith

-- Theorem to prove the expression for T_n
theorem expression_for_T (n : ℕ) : T n = 2^(n + 1) - 2 := by
  rfl


end expression_for_a_expression_for_T_l634_634468


namespace cot_matrix_det_l634_634954

theorem cot_matrix_det (A B C : ℝ) (h₁ : A + B + C = π) (h₂ : A ≠ π / 2) (h₃ : B ≠ π / 2) :
  det ![
    ![Real.cot A, 1, 1],
    ![1, Real.cot B, 1],
    ![1, 1, Real.cot C]
  ] = 2 := by
  sorry

end cot_matrix_det_l634_634954


namespace question1_question2_question3_l634_634098

noncomputable def a : ℕ → ℝ
| 1 := 3 / 5
| (n + 1) := 3 * (a n) / (2 * (a n) + 1)

def geomSeq (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n, a (n + 1) = r * a n

theorem question1 : geomSeq (λ n, 1 / (a n) - 1) := sorry

noncomputable def S_n (n : ℕ) : ℝ :=
(∑ i in Finset.range n, 1 / (a (i + 1)))

def max_n_lt_101 := ∃ n : ℕ, S_n n < 101 ∧ ∀ m : ℕ, S_n m < 101 → m ≤ n

theorem question2 : ∃ n : ℕ, S_n n < 101 ∧ (∀ m : ℕ, S_n m < 101 → m ≤ 99) := 
sorry

theorem question3 : ¬(∃ m s n : ℕ, m ≠ s ∧ s ≠ n ∧ m ≠ n ∧ m + n = 2 * s ∧ (a m - 1) * (a n - 1) = (a s - 1) ^ 2) :=
sorry

end question1_question2_question3_l634_634098


namespace problem_solution_l634_634640

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.cos (x - Real.pi / 4))^2 - 1

lemma problem_condition (x : ℝ) : f x = 2 * (Real.cos (x - Real.pi / 4))^2 - 1 := rfl

theorem problem_solution : (∀ x : ℝ, f(-x) = -f(x)) ∧ ∀ T > 0, (∀ x : ℝ, f(x + T) = f(x)) → T = Real.pi :=
by
  sorry

end problem_solution_l634_634640


namespace sequence_formula_l634_634171

theorem sequence_formula (a : ℕ → ℚ) (h1 : a 2 = 3 / 2) (h2 : a 3 = 7 / 3) 
  (h3 : ∀ n : ℕ, ∃ r : ℚ, (∀ m : ℕ, m ≥ 2 → (m * a m + 1) / (n * a n + 1) = r ^ (m - n))) :
  a n = (2^n - 1) / n := 
sorry

end sequence_formula_l634_634171


namespace negation_of_quadratic_inequality_l634_634644

-- Definitions
def quadratic_inequality (a : ℝ) : Prop := ∃ x : ℝ, x * x + a * x + 1 < 0

-- Theorem statement
theorem negation_of_quadratic_inequality (a : ℝ) : ¬ (quadratic_inequality a) ↔ ∀ x : ℝ, x * x + a * x + 1 ≥ 0 :=
by sorry

end negation_of_quadratic_inequality_l634_634644


namespace savings_percent_l634_634694

variables (cost_per_roll_individual : ℝ) (total_cost_package : ℝ) (number_of_rolls : ℕ)

def percent_savings_per_roll (cost_per_roll_individual total_cost_package : ℝ) (number_of_rolls : ℕ) : ℝ :=
  ((cost_per_roll_individual - total_cost_package / number_of_rolls) / cost_per_roll_individual) * 100

theorem savings_percent 
  (h1 : cost_per_roll_individual = 1) 
  (h2 : total_cost_package = 9) 
  (h3 : number_of_rolls = 12) : 
  percent_savings_per_roll cost_per_roll_individual total_cost_package number_of_rolls = 25 :=
by
  sorry

end savings_percent_l634_634694


namespace median_sequence_l634_634678

theorem median_sequence : 
  let seq := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0] in
  let sorted_seq := List.sort seq in
  let len := List.length sorted_seq in
  len = 12 → 
  (sorted_seq.nth (len / 2 - 1) + sorted_seq.nth (len / 2)) / 2 = 4.5 :=
by
  intros
  sorry

end median_sequence_l634_634678


namespace petya_wins_probability_l634_634987

/-- Petya plays a game with 16 stones where players alternate in taking 1 to 4 stones. 
     Petya wins if they can take the last stone first while making random choices. 
     The computer plays optimally. The probability of Petya winning is 1 / 256. -/
theorem petya_wins_probability :
  let stones := 16 in
  let optimal_strategy := ∀ n, n % 5 = 0 in
  let random_choice_probability := (1 / 4 : ℚ) in
  let total_random_choices := 4 in
  (random_choice_probability ^ total_random_choices) = (1 / 256 : ℚ) :=
by
  sorry

end petya_wins_probability_l634_634987


namespace angle_DEA_half_diff_beta_gamma_l634_634931

noncomputable def midpoint_shorter_arc (E : Point) (B C : Point) (circle : Circle) : Prop :=
  E = midpoint (arc BC) ∧ E ∈ circle

noncomputable def diameter_point (D E : Point) (circle : Circle) : Prop :=
  diameter_linear (D, E) ∧ circle.contains(E)])

theorem angle_DEA_half_diff_beta_gamma
  (A B C D E : Point)
  (angle : Point → Point → Point → ℝ)
  (β γ : ℝ)
  (circumcircle : Circle)
  (h1 : b ≥ c)
  (h2 : midpoint_shorter_arc E B C circumcircle)
  (h3 : diameter_point D E circumcircle)
  : angle D E A = (1 / 2) * (β - γ) :=
sorry

end angle_DEA_half_diff_beta_gamma_l634_634931


namespace contribution_amount_l634_634632

theorem contribution_amount (x : ℝ) (S : ℝ) :
  (S = 10 * x) ∧ (S = 15 * (x - 100)) → x = 300 :=
by
  sorry

end contribution_amount_l634_634632


namespace shaded_area_of_cross_in_grid_l634_634144

theorem shaded_area_of_cross_in_grid (grid_size : ℕ) (center : ℕ × ℕ) (triangle_vertices : (ℕ × ℕ) × (ℕ × ℕ) × (ℕ × ℕ)) :
  grid_size = 4 ∧ center = (2, 2) ∧
  (triangle_vertices = ((2, 2), (0, 2), (2, 0)) ∨ triangle_vertices = ((2, 2), (2, 4), (4, 2)) ∨
   triangle_vertices = ((2, 2), (2, 0), (4, 2)) ∨ triangle_vertices = ((2, 2), (0, 2), (2, 4))) →
  let area_one_triangle := 0.5 in
  let total_area := 4 * area_one_triangle in
  total_area = 2 := by
  intros _ h_center h_triangle_vertices
  sorry

end shaded_area_of_cross_in_grid_l634_634144


namespace smallest_positive_m_l634_634002

theorem smallest_positive_m (m : ℕ) (h : ∃ n : ℤ, m^3 - 90 = n * (m + 9)) : m = 12 :=
by
  sorry

end smallest_positive_m_l634_634002


namespace arccos_greater_than_arctan_l634_634015

theorem arccos_greater_than_arctan (x : ℝ) (hx1 : x ≥ -1) (hx2 : x ≤ 1) : 
    (arccos x > arctan x) ↔ (x < 1 / sqrt 3) :=
sorry

end arccos_greater_than_arctan_l634_634015


namespace area_of_right_triangle_from_roots_l634_634897

theorem area_of_right_triangle_from_roots :
  ∀ (a b : ℝ), (a^2 - 7*a + 12 = 0) → (b^2 - 7*b + 12 = 0) →
  (∃ (area : ℝ), (area = 6) ∨ (area = (3 * real.sqrt 7) / 2)) :=
by
  intros a b ha hb
  sorry

end area_of_right_triangle_from_roots_l634_634897


namespace sum_real_y_values_l634_634044

theorem sum_real_y_values :
  ∃ ylist : List ℝ,
  (∀ x y : ℝ, x^2 + x^2 * y^2 + x^2 * y^4 = 525 ∧ x + x * y + x * y^2 = 35 → y ∈ ylist) ∧ ylist.sum = 5 / 2 :=
begin
  sorry -- Proof is omitted as instructed.
end

end sum_real_y_values_l634_634044


namespace max_value_of_trig_expression_l634_634024

open Real

theorem max_value_of_trig_expression : ∀ x : ℝ, 3 * cos x + 4 * sin x ≤ 5 :=
sorry

end max_value_of_trig_expression_l634_634024


namespace sum_12_pretty_numbers_l634_634385

def is_pretty (n k : ℕ) : Prop :=
  Nat.totient n = k ∧ n % k = 0

theorem sum_12_pretty_numbers : 
  let S := (List.filter (λ n => is_pretty n 12) (List.range 500)).sum
  S = 762 ∧ S / 12 = 63.5 := by
  sorry

end sum_12_pretty_numbers_l634_634385


namespace parabola_rotation_solution_l634_634269

def parabola_rotated (θ : ℝ) : Prop := 
  ∀ θ ∈ set.Ioo 0 (Real.pi / 2), 
  let x := Real.tan θ in 
  let y := Real.cot θ in 
  let length_x := abs (x - 0) -- the length on the x-axis
  let length_y := abs (y - 0) -- the length on the y-axis
  in length_x = 1 → length_y = 2

theorem parabola_rotation_solution :
  ∃ θ : ℝ, θ ∈ set.Ioo 0 (Real.pi / 2) ∧
  parabola_rotated θ :=
begin
  sorry
end

end parabola_rotation_solution_l634_634269


namespace number_of_girls_l634_634286

variable (total_children : ℕ) (boys : ℕ)

theorem number_of_girls (h1 : total_children = 117) (h2 : boys = 40) : 
  total_children - boys = 77 := by
  sorry

end number_of_girls_l634_634286


namespace pq_work_together_in_10_days_l634_634699

theorem pq_work_together_in_10_days 
  (p q r : ℝ)
  (hq : 1/q = 1/28)
  (hr : 1/r = 1/35)
  (hp : 1/p = 1/q + 1/r) :
  1/p + 1/q = 1/10 :=
by sorry

end pq_work_together_in_10_days_l634_634699


namespace invariant_curves_are_logarithmic_l634_634427

noncomputable def T_r (r : ℝ) : ℝ × ℝ → ℝ × ℝ :=
  λ p, (2^r * p.1, r * 2^r * p.1 + 2^r * p.2)

def F : set (ℝ × ℝ → ℝ × ℝ) := {T_r r | r : ℝ}

def invariant_curve (f : ℝ → ℝ) : Prop :=
  ∀ (r : ℝ) (x : ℝ), f (2^r * x) = r * 2^r * x + 2^r * f x

theorem invariant_curves_are_logarithmic (f : ℝ → ℝ) (c : ℝ) :
  (∀ (r : ℝ) (x : ℝ), f (2^r * x) = r * 2^r * x + 2^r * f x) ↔ 
  (∀ (x : ℝ), f x = x * (Real.log (Real.abs x) / Real.log 2 + c)) :=
sorry

end invariant_curves_are_logarithmic_l634_634427


namespace seq_a2010_l634_634553

-- Definitions and conditions
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ 
  a 2 = 3 ∧ 
  ∀ n ≥ 2, a (n + 1) = (a n * a (n - 1)) % 10

-- Proof statement
theorem seq_a2010 {a : ℕ → ℕ} (h : seq a) : a 2010 = 4 := 
  sorry

end seq_a2010_l634_634553


namespace empty_jug_x_l634_634291

-- The definitions and conditions provided in the problem.
def equal_volume_jugs (V Cx Cy Cz : ℝ) : Prop :=
  V = (1/4) * Cx ∧ V = (2/3) * Cy ∧ V = (3/5) * Cz

def water_transfer {Cx Cy Cz : ℝ} (V : ℝ) : Prop :=
  -- Jug y initially holds (2/3)Cy and becomes full
  let remaining_in_x_after_filling_y := (1/4)*Cx - (2/9)*Cx in
  -- Jug z holds remaining water from x
  remaining_in_x_after_filling_y = 0

theorem empty_jug_x
  (Cx Cy Cz V : ℝ) 
  (h1 : equal_volume_jugs V Cx Cy Cz)
  : water_transfer V :=
sorry

end empty_jug_x_l634_634291


namespace parabola_focus_distance_l634_634453

theorem parabola_focus_distance :
  let F := (0 : ℝ, 1 / 2 : ℝ) in
  let M := (1 / 2 : ℝ, 0 : ℝ) in
  |(F.1 - M.1) ^ 2 + (F.2 - M.2) ^ 2| ^ (1 / 2 : ℝ) = (√2 / 2) :=
by
  let F : ℝ × ℝ := (0, 1 / 2)
  let M : ℝ × ℝ := (1 / 2, 0)
  have dist_sq : (F.1 - M.1) ^ 2 + (F.2 - M.2) ^ 2 = 1 / 2 := by sorry
  rw [Real.sqrt_eq_rpow, dist_sq]
  norm_num
  sorry

end parabola_focus_distance_l634_634453


namespace parabola_x_intercepts_count_l634_634111

theorem parabola_x_intercepts_count :
  ∃! x, ∃ y, x = -3 * y^2 + 2 * y + 2 ∧ y = 0 :=
by
  sorry

end parabola_x_intercepts_count_l634_634111


namespace sum_areas_of_squares_l634_634746

-- Definitions to capture the conditions
variables (A C F : Type) [HasAngle A C F] -- Types with the notion of angles
variables (AC AF CF : ℝ) -- The lengths are real numbers
variable (h_right : ∠ A C F = 90) -- Angle FAC is a right angle
variable (h_CF : CF = 12) -- CF = 12 units

-- The theorem stating the required proof
theorem sum_areas_of_squares : AC^2 + AF^2 = 144 :=
by
  sorry

end sum_areas_of_squares_l634_634746


namespace extreme_value_when_a_is_zero_monotonicity_of_f_when_a_lt_zero_l634_634082

def f (x : ℝ) (a : ℝ) : ℝ := (2 - a) * Real.log x + 1 / x + 2 * a * x

theorem extreme_value_when_a_is_zero : 
  (∀ x > 0, f x 0 ≥ 2 - 2 * (Real.log 2)) ∧ 
  (∃ x > 0, f x 0 = 2 - 2 * (Real.log 2)) := 
sorry

theorem monotonicity_of_f_when_a_lt_zero (a : ℝ) (h : a < 0) : 
  ( (a < -2) → 
    (∀ x > 0, x < -1/a → f x a < f (-1/a) a) ∧ 
    (∀ x > -1/a, x < 1/2 → f x a > f (-1/a) a) ∧ 
    (∀ x > 1/2 → f x a < f (1/2) a) ) ∧

  ( (a = -2) → 
    (∀ x > 0, ∀ y > x, f y a < f x a) ) ∧

  ( (-2 < a < 0) → 
    (∀ x > 0, x < 1/2 → f x a < f (1/2) a) ∧ 
    (∀ x > -1/a, x < 1/2 → f x a > f (1/2) a) ∧ 
    (∀ x > -1/a, x > 1/2 → f x a < f (1/2) a) ) :=
sorry

end extreme_value_when_a_is_zero_monotonicity_of_f_when_a_lt_zero_l634_634082


namespace num_solutions_x_squared_minus_y_squared_eq_2001_l634_634953

theorem num_solutions_x_squared_minus_y_squared_eq_2001 
  (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^2 - y^2 = 2001 ↔ (x, y) = (1001, 1000) ∨ (x, y) = (335, 332) := sorry

end num_solutions_x_squared_minus_y_squared_eq_2001_l634_634953


namespace candidate_X_votes_candidate_X_did_not_win_l634_634525

theorem candidate_X_votes (Y Z : ℕ) (hZ : Z = 25000) (hY : Y = 3 * Z / 5) (hX : X = 3 * Y / 2) : X = 22500 :=
by {
  have hY_value : Y = 15000,
  { rw hZ at hY,
    exact hY,
  },
  have hX_value : X = 22500,
  { rw hY_value at hX,
    exact hX,
  },
  exact hX_value,
}

theorem candidate_X_did_not_win (X : ℕ) (hX : X = 22500) : X < 30000 :=
by {
  rw hX,
  exact Nat.lt_succ_self 29999,
}

end candidate_X_votes_candidate_X_did_not_win_l634_634525


namespace stereographic_projection_reflection_l634_634624

noncomputable def sphere : Type := sorry
noncomputable def point_on_sphere (P : sphere) : Prop := sorry
noncomputable def reflection_on_sphere (P P' : sphere) (e : sphere) : Prop := sorry
noncomputable def arbitrary_point (E : sphere) (P P' : sphere) : Prop := E ≠ P ∧ E ≠ P'
noncomputable def tangent_plane (E : sphere) : Type := sorry
noncomputable def stereographic_projection (E : sphere) (δ : Type) : sphere → sorry := sorry
noncomputable def circle_on_plane (e : sphere) (E : sphere) (δ : Type) : Type := sorry
noncomputable def inversion_in_circle (P P' : sphere) (e_1 : Type) : Prop := sorry

theorem stereographic_projection_reflection (P P' E : sphere) (e : sphere) (δ : Type) (e_1 : Type) :
  point_on_sphere P ∧
  reflection_on_sphere P P' e ∧
  arbitrary_point E P P' ∧
  circle_on_plane e E δ = e_1 →
  inversion_in_circle P P' e_1 :=
sorry

end stereographic_projection_reflection_l634_634624


namespace perimeter_correct_l634_634166

open EuclideanGeometry

noncomputable def perimeter_of_figure : ℝ := 
  let AB : ℝ := 6
  let BC : ℝ := AB
  let AD : ℝ := AB / 2
  let DC : ℝ := AD
  let DE : ℝ := AD
  let EA : ℝ := DE
  let EF : ℝ := EA / 2
  let FG : ℝ := EF
  let GH : ℝ := FG / 2
  let HJ : ℝ := GH
  let JA : ℝ := HJ
  AB + BC + DC + DE + EF + FG + GH + HJ + JA

theorem perimeter_correct : perimeter_of_figure = 23.25 :=
by
  -- proof steps would go here, but are not required for this problem transformation
  sorry

end perimeter_correct_l634_634166


namespace log_sum_l634_634810

theorem log_sum (a b c : ℝ) (h1 : 3^a = 6) (h2 : 4^b = 6) (h3 : 5^c = 6) :
  (1 / a) + (1 / b) + (1 / c) = Real.log 60 / Real.log 6 :=
by
  sorry

end log_sum_l634_634810


namespace bad_carrots_l634_634301

-- Conditions
def carrots_picked_by_vanessa := 17
def carrots_picked_by_mom := 14
def good_carrots := 24
def total_carrots := carrots_picked_by_vanessa + carrots_picked_by_mom

-- Question and Proof
theorem bad_carrots :
  total_carrots - good_carrots = 7 :=
by
  -- Placeholder for proof
  sorry

end bad_carrots_l634_634301


namespace height_of_cylinder_is_six_l634_634280

-- Define the radius of the sphere
def radius_sphere : ℝ := 3

-- Define the diameter and radius of the cylinder
def diameter_cylinder : ℝ := 6
def radius_cylinder : ℝ := diameter_cylinder / 2

-- Define the surface area of the sphere
def surface_area_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- Define the curved surface area of the cylinder
def curved_surface_area_cylinder (r h : ℝ) : ℝ := 2 * Real.pi * r * h

-- Define the given values in the conditions
def A_sphere : ℝ := surface_area_sphere radius_sphere
def A_cylinder (h : ℝ) : ℝ := curved_surface_area_cylinder radius_cylinder h

-- Problem statement to prove that height of cylinder == 6 cm
theorem height_of_cylinder_is_six :
  ∃ h : ℝ, A_sphere = A_cylinder h ∧ h = 6 := sorry

end height_of_cylinder_is_six_l634_634280


namespace ellipse_equation_and_max_value_l634_634076

theorem ellipse_equation_and_max_value (O A N M F1 F2 P Q : ℝ × ℝ) 
  (l1 l2 : ℝ → ℝ) (k m : ℝ)
  (C1 : ∀ (x y : ℝ), x^2 + y^2 = 12)
  (tangent_condition : ∀ (x y : ℝ), l1 x = sqrt 2 * y - 6)
  (A_on_C1 : ∀ (x0 y0 : ℝ), C1 x0 y0 → x0 = A.1 ∧ y0 = A.2)
  (M_def : ∀ (x0 : ℝ), M = (x0, 0))
  (N_def : ∀ (x y x0 y0 : ℝ),
    N = (sqrt 3 / 3 * x0, 1 / 2 * y0) →
    x0 = sqrt 3 * x ∧ y0 = 2 * y)
  (l2_intersect_condition : ∀ (x y : ℝ),
    (4 * k^2 + 3) * x^2 + 8 * k * m * x + 4 * m^2 - 12 = 0 →
    m^2 = 4 * k^2 + 3)
  (dist_f1 l2 : ℝ)
  (dist_f2 l2 : ℝ)
  (dist_pq P Q : ℝ)
  (k_nonzero : k ≠ 0 → (dist_f1 + dist_f2) * dist_pq = 16 / abs m + 1 / abs m)
  (k_zero : k = 0 → (dist_f1 + dist_f2) * dist_pq = 4 * sqrt 3)
  (maximum_value : (dist_f1 + dist_f2) * dist_pq ≤ 4 * sqrt 3) :
  (∀ (x y : ℝ), N = (x, y) → (x^2 / 4 + y^2 / 3 = 1)) ∧ 
  ∃ (max_val : ℝ), max_val = 4 * sqrt 3 :=
by
  sorry

end ellipse_equation_and_max_value_l634_634076


namespace diagonals_not_parallel_to_sides_in_regular_32gon_l634_634864

theorem diagonals_not_parallel_to_sides_in_regular_32gon :
  ∀ (n : ℕ), n = 32 -> 
  let total_diagonals := n * (n - 3) / 2,
      pairs_of_parallel_sides := n / 2,
      diagonals_per_pair := (n - 4) / 2,
      parallel_diagonals := pairs_of_parallel_sides * diagonals_per_pair
  in total_diagonals - parallel_diagonals = 240 :=
begin
  intros n hn,
  rw hn,
  let total_diagonals := 32 * (32 - 3) / 2,
  let pairs_of_parallel_sides := 32 / 2,
  let diagonals_per_pair := (32 - 4) / 2,
  let parallel_diagonals := pairs_of_parallel_sides * diagonals_per_pair,
  calc 
    total_diagonals - parallel_diagonals 
      = 32 * 29 / 2 - (16 * 14) : by rfl
  ... = 464 - 224 : by norm_num
  ... = 240 : by norm_num,
end

end diagonals_not_parallel_to_sides_in_regular_32gon_l634_634864


namespace circles_intersect_at_circumcenter_l634_634188

-- Definitions and Properties
variables {A B C D E F O : Type}
variables (is_midpoint : ∀ {X Y M}, M = (X + Y) / 2)
variables (triangle_ABC : (A B C : Type))
variables (mid_AB : D = (A + B) / 2)
variables (mid_BC : E = (B + C) / 2)
variables (mid_AC : F = (A + C) / 2)
variables (circumcenter : ∀ {A B C}, O = circumcenter A B C)
variables (circle_k1 : k_1 contains_points [A, D, F])
variables (circle_k2 : k_2 contains_points [B, E, D])
variables (circle_k3 : k_3 contains_points [C, F, E])

-- Theorem Statement
theorem circles_intersect_at_circumcenter :
  O ∈ k_1 ∧ O ∈ k_2 ∧ O ∈ k_3 :=
by
  sorry

end circles_intersect_at_circumcenter_l634_634188


namespace first_issue_pages_l634_634669

-- Define the conditions
def total_pages := 220
def pages_third_issue (x : ℕ) := x + 4

-- Statement of the problem
theorem first_issue_pages (x : ℕ) (hx : 3 * x + 4 = total_pages) : x = 72 :=
sorry

end first_issue_pages_l634_634669


namespace average_of_three_strings_l634_634977

variable (len1 len2 len3 : ℝ)
variable (h1 : len1 = 2)
variable (h2 : len2 = 5)
variable (h3 : len3 = 3)

def average_length (a b c : ℝ) : ℝ := (a + b + c) / 3

theorem average_of_three_strings : average_length len1 len2 len3 = 10 / 3 := by
  rw [←h1, ←h2, ←h3]
  rw [average_length]
  norm_num
  sorry

end average_of_three_strings_l634_634977


namespace exists_m_for_n_divides_2_pow_m_plus_m_l634_634034

theorem exists_m_for_n_divides_2_pow_m_plus_m (n : ℕ) (hn : 0 < n) : 
  ∃ m : ℕ, 0 < m ∧ n ∣ 2^m + m :=
sorry

end exists_m_for_n_divides_2_pow_m_plus_m_l634_634034


namespace triangle_congruence_and_segments_equality_l634_634104

theorem triangle_congruence_and_segments_equality
    (A B C A1 B1 C1 K L K1 L1 : Type*)
    [∀ x y z : Type*, MetricSpace x]
    (AB_eq : distance A B = distance A1 B1)
    (AC_eq : distance A C = distance A1 C1)
    (angle_A_eq : angle A B C = angle A1 B1 C1)
    (AK_eq : distance A K = distance A1 K1)
    (LC_eq : distance L C = distance L1 C1) : 
    distance K L = distance K1 L1 ∧ distance A L = distance A1 L1 :=
by
  sorry

end triangle_congruence_and_segments_equality_l634_634104


namespace length_of_rectangle_l634_634131

-- Definitions based on conditions:
def side_length_square : ℝ := 4
def width_rectangle : ℝ := 8
def area_square (side : ℝ) : ℝ := side * side
def area_rectangle (width length : ℝ) : ℝ := width * length

-- The goal is to prove the length of the rectangle
theorem length_of_rectangle :
  (area_square side_length_square) = (area_rectangle width_rectangle 2) :=
by
  sorry

end length_of_rectangle_l634_634131


namespace total_lineups_is_sixty_total_lineups_excluding_A_in_doubles_l634_634336

-- Definitions for the necessary conditions:
def players : Type := fin 5 -- Five players
def team : Type := set players -- Team is a set of five players
def singles_matches : fin 2 → player := sorry -- Two singles matches
def doubles_match : fin 2 → player := sorry -- One doubles match
def disjoint : set player → set player → Prop := sorry -- Disjoint sets

-- Statement for Question 1:
theorem total_lineups_is_sixty (t : team) :
  card {l | ∃ (s : fin 2 → player) (d : fin 2 → player), -- Listing possible lineups
    (∀ i j, i ≠ j → s i ≠ s j) ∧ (∀ i j, i ≠ j → d i ≠ d j) ∧ -- No player repeats in singles/doubles
    (disjoint (s i) (d j))} = 60 := sorry

-- Additional definition for player A's constraint:
def player_A := players 0 -- Suppose player A is player 0
def cannot_play_doubles (d : fin 2 → player) : Prop := ∀ i, d i ≠ player_A

-- Statement for Question 2:
theorem total_lineups_excluding_A_in_doubles (t : team) :
  card {l | ∃ (s : fin 2 → player) (d : fin 2 → player),
    cannot_play_doubles d ∧
    (∀ i j, i ≠ j → s i ≠ s j) ∧ (∀ i j, i ≠ j → d i ≠ d j) ∧
    (disjoint (s i) (d j))} = 36 := sorry

end total_lineups_is_sixty_total_lineups_excluding_A_in_doubles_l634_634336


namespace find_number_l634_634701

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 11) : x = 5.5 :=
by
  sorry

end find_number_l634_634701


namespace tiling_possible_l634_634045

theorem tiling_possible (n : ℕ) : 
  (∃ k : ℕ, n = 2 * k ∧ (k % 2 = 1)) 
  ↔ (∃ (S T : ℕ), (S + T) * 4 = n^2 ∧ odd S) := 
sorry

end tiling_possible_l634_634045


namespace final_selling_prices_l634_634745

-- Definitions based on conditions
def cost_price_speaker : ℝ := 160
def cost_price_keyboard : ℝ := 220
def cost_price_mouse : ℝ := 120

def markup_rate_speaker : ℝ := 0.20
def markup_rate_keyboard : ℝ := 0.30
def markup_rate_mouse : ℝ := 0.40

def discount_rate_speaker : ℝ := 0.10
def discount_rate_keyboard : ℝ := 0.15
def discount_rate_mouse : ℝ := 0.25

-- Problem statement in Lean 4
theorem final_selling_prices :
  let marked_price_speaker := cost_price_speaker * (1 + markup_rate_speaker);
      discount_amount_speaker := marked_price_speaker * discount_rate_speaker;
      final_selling_price_speaker := marked_price_speaker - discount_amount_speaker;

      marked_price_keyboard := cost_price_keyboard * (1 + markup_rate_keyboard);
      discount_amount_keyboard := marked_price_keyboard * discount_rate_keyboard;
      final_selling_price_keyboard := marked_price_keyboard - discount_amount_keyboard;

      marked_price_mouse := cost_price_mouse * (1 + markup_rate_mouse);
      discount_amount_mouse := marked_price_mouse * discount_rate_mouse;
      final_selling_price_mouse := marked_price_mouse - discount_amount_mouse
  in
    final_selling_price_speaker = 172.8 ∧
    final_selling_price_keyboard = 243.1 ∧
    final_selling_price_mouse = 126 := 
by
  sorry

end final_selling_prices_l634_634745


namespace circumsphere_radius_pyramid_l634_634251

-- Definitions of the conditions in the problem
def equilateral_triangle_side_length : ℝ := 6
def perpendicular_lateral_edge_length : ℝ := 4

-- The statement of the proof problem
theorem circumsphere_radius_pyramid :
  let a := equilateral_triangle_side_length,
      h := perpendicular_lateral_edge_length,
      Q_to_vertex := a * (Real.sqrt 3) / 3,
      OQ := h / 2,
      R := Real.sqrt ((Q_to_vertex ^ 2) + (OQ ^ 2))
  in
    R = 4 :=
by
  -- Proof steps would go here
  sorry

end circumsphere_radius_pyramid_l634_634251


namespace lcm_gcd_product_12_15_l634_634373

theorem lcm_gcd_product_12_15 : 
  let a := 12
  let b := 15
  lcm a b * gcd a b = 180 :=
by
  sorry

end lcm_gcd_product_12_15_l634_634373


namespace find_prime_triplet_l634_634013

def is_geometric_sequence (x y z : ℕ) : Prop :=
  (y^2 = x * z)

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_prime_triplet :
  ∃ (a b c : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧
  a < b ∧ b < c ∧ c < 100 ∧
  is_geometric_sequence (a + 1) (b + 1) (c + 1) ∧
  (a = 17 ∧ b = 23 ∧ c = 31) :=
by
  sorry

end find_prime_triplet_l634_634013


namespace ratio_of_pentagons_inscribed_circumscribed_l634_634731

def ratio_of_areas (r : ℝ) : ℝ :=
  let sin36 := Real.sin (36 * Real.pi / 180)
  1 / (sin36 * sin36)

theorem ratio_of_pentagons_inscribed_circumscribed (r : ℝ) :
  let A_inscribed := (1 / 4) * Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) * r^2
  let A_circumscribed := (1 / 4) * Real.sqrt (5 * (5 + 2 * Real.sqrt 5)) * (r / Real.sin(36 * Real.pi / 180))^2
  (A_circumscribed / A_inscribed) = ratio_of_areas r :=
by
  sorry

end ratio_of_pentagons_inscribed_circumscribed_l634_634731


namespace middle_guards_hours_l634_634721

def total_hours := 9
def hours_first_guard := 3
def hours_last_guard := 2
def remaining_hours := total_hours - hours_first_guard - hours_last_guard
def num_middle_guards := 2

theorem middle_guards_hours : remaining_hours / num_middle_guards = 2 := by
  sorry

end middle_guards_hours_l634_634721


namespace estimate_total_fish_l634_634664

theorem estimate_total_fish (n1 n2 m : ℕ) (h₁ : n1 = 40) (h₂ : n2 = 100) (h₃ : m = 5) : 
  let x := (n2 * n1) / m in x = 800 :=
by
  sorry

end estimate_total_fish_l634_634664


namespace jessies_original_weight_l634_634562

theorem jessies_original_weight (current_weight weight_lost original_weight : ℕ) 
  (h_current: current_weight = 27) (h_lost: weight_lost = 101) 
  (h_original: original_weight = current_weight + weight_lost) : 
  original_weight = 128 :=
by
  rw [h_current, h_lost] at h_original
  exact h_original

end jessies_original_weight_l634_634562


namespace train_pass_time_l634_634923

theorem train_pass_time (length_of_train : ℝ) (speed_of_train_kmh : ℝ) (time_to_pass : ℝ) 
  (h_len : length_of_train = 180) 
  (h_speed : speed_of_train_kmh = 54) : 
  time_to_pass = 12 := 
by
  -- Conversion factor: 1 km/hr = 1000 meters / 3600 seconds
  let speed_of_train_ms := speed_of_train_kmh * (1000 / 3600)
  -- Speed after conversion
  have h_speed_converted : speed_of_train_ms = 15 := sorry
  -- Using Distance = Speed * Time: Time = Distance / Speed
  have calculation : time_to_pass = length_of_train / speed_of_train_ms := sorry
  -- Plugging in the values and simplifying
  calc 
    time_to_pass = length_of_train / speed_of_train_ms : calculation
              ... = 180 / 15 : by rw [h_len, h_speed_converted]
              ... = 12 : by norm_num

end train_pass_time_l634_634923


namespace product_of_solutions_is_16_l634_634505

noncomputable def product_of_solutions : ℂ :=
  let solutions : List ℂ := [Complex.polar 2 (Real.angle.ofDegrees 22.5),
                              Complex.polar 2 (Real.angle.ofDegrees 67.5),
                              Complex.polar 2 (Real.angle.ofDegrees 337.5),
                              Complex.polar 2 (Real.angle.ofDegrees 292.5)]
  solutions.foldl (· * ·) 1

theorem product_of_solutions_is_16 :
  product_of_solutions = 16 := 
sorry

end product_of_solutions_is_16_l634_634505


namespace circle_area_ratio_l634_634130

noncomputable def area_ratio (RX RY : ℝ) := (π * RX^2) / (π * RY^2)

theorem circle_area_ratio {RX RY : ℝ} (h : (60 / 360) * (2 * π * RX) = (20 / 360) * (2 * π * RY)) :
  area_ratio RX RY = 9 := by
  sorry

end circle_area_ratio_l634_634130


namespace ab_value_l634_634538

theorem ab_value 
  (a b : ℝ) 
  (hx : 2 = b + 1) 
  (hy : a = -3) : 
  a * b = -3 :=
by
  sorry

end ab_value_l634_634538


namespace projection_eq_4_l634_634451

def vec_proj (a b : EuclideanSpace) : ℝ :=
  (a ⬝ b) / ∥b∥

theorem projection_eq_4
  {a b : EuclideanSpace}
  (ha : ∥a∥ = 4)
  (hb : ∥b∥ = 2)
  (h_perpendicular : (a - 2 • b) ⬝ a = 0) :
  vec_proj a b = 4 :=
by
  sorry

end projection_eq_4_l634_634451


namespace tangent_line_eq_l634_634068

variables {f g : ℝ → ℝ}
noncomputable def y (x : ℝ) := (f x + 2) / (g x)

-- Conditions
axiom f_eq : f 5 = 5
axiom f'_eq : deriv f 5 = 3
axiom g_eq : g 5 = 4
axiom g'_eq : deriv g 5 = 1

-- Main Statement for Proof
theorem tangent_line_eq :
  let y_deriv := (deriv (λ x, (f x + 2) / (g x))) in
  y_deriv 5 = (5 : ℝ) / 16 ∧
  y 5 = 7 / 4 →
  ∀ x y : ℝ, (5 * x - 16 * y + 3 = 0) :=
by
  sorry

end tangent_line_eq_l634_634068


namespace isosceles_triangle_vertex_angle_l634_634922

theorem isosceles_triangle_vertex_angle (A B C : ℝ) (h_iso : A = B ∨ A = C ∨ B = C) (h_sum : A + B + C = 180) (h_one_angle : A = 50 ∨ B = 50 ∨ C = 50) :
  A = 80 ∨ B = 80 ∨ C = 80 ∨ A = 50 ∨ B = 50 ∨ C = 50 :=
by
  sorry

end isosceles_triangle_vertex_angle_l634_634922


namespace unique_B_cube_l634_634585

open Matrix

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := !![p, q; r, s]

theorem unique_B_cube (B : Matrix (Fin 2) (Fin 2) ℝ) (hB : B^4 = 0) :
  ∃! (C : Matrix (Fin 2) (Fin 2) ℝ), C = B^3 :=
sorry

end unique_B_cube_l634_634585


namespace exists_large_absolute_value_solutions_l634_634390

theorem exists_large_absolute_value_solutions : 
  ∃ (x1 x2 y1 y2 y3 y4 : ℤ), 
    x1 + x2 = y1 + y2 + y3 + y4 ∧ 
    x1^2 + x2^2 = y1^2 + y2^2 + y3^2 + y4^2 ∧ 
    x1^3 + x2^3 = y1^3 + y2^3 + y3^3 + y4^3 ∧ 
    abs x1 > 2020 ∧ abs x2 > 2020 ∧ abs y1 > 2020 ∧ abs y2 > 2020 ∧ abs y3 > 2020 ∧ abs y4 > 2020 :=
  by
  sorry

end exists_large_absolute_value_solutions_l634_634390


namespace identical_machine_production_l634_634238

-- Definitions based on given conditions
def machine_production_rate (machines : ℕ) (rate : ℕ) :=
  rate / machines

def bottles_in_minute (machines : ℕ) (rate_per_machine : ℕ) :=
  machines * rate_per_machine

def total_bottles (bottle_rate_per_minute : ℕ) (minutes : ℕ) :=
  bottle_rate_per_minute * minutes

-- Theorem to prove based on the question == answer given conditions
theorem identical_machine_production :
  ∀ (machines_initial machines_final : ℕ) (bottles_per_minute : ℕ) (minutes : ℕ),
    machines_initial = 6 →
    machines_final = 12 →
    bottles_per_minute = 270 →
    minutes = 4 →
    total_bottles (bottles_in_minute machines_final (machine_production_rate machines_initial bottles_per_minute)) minutes = 2160 := by
  intros
  sorry

end identical_machine_production_l634_634238


namespace union_P_Q_l634_634204

open Set

-- Definitions from the conditions
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {x | 3 ≤ x ∧ x < 7 ∧ x ∈ ℕ}

-- Statement to prove
theorem union_P_Q : P ∪ Q = {1, 2, 3, 4, 5, 6} :=
by
  -- Proof will go here
  sorry

end union_P_Q_l634_634204


namespace abs_sqrt_ineq_l634_634775

theorem abs_sqrt_ineq (x : ℝ) : abs (x^2 - 5) < 9 ↔ -real.sqrt 14 < x ∧ x < real.sqrt 14 :=
by
  sorry

end abs_sqrt_ineq_l634_634775


namespace value_of_f_sin_7pi_over_6_l634_634087

def f (x : ℝ) : ℝ := 4 * x^2 + 2 * x

theorem value_of_f_sin_7pi_over_6 :
  f (Real.sin (7 * Real.pi / 6)) = 0 :=
by
  sorry

end value_of_f_sin_7pi_over_6_l634_634087


namespace school_seat_total_cost_l634_634353

theorem school_seat_total_cost :
  let seats_1 := 10 * 20
      cost_1 := seats_1 * 60
      discount_1_per_group := 0.12 * (60 * 20)
      total_discount_1 := (seats_1 / 20) * discount_1_per_group
      cost_1_discounted := cost_1 - total_discount_1

      seats_2 := 10 * 15
      cost_2 := seats_2 * 50
      discount_2_per_group := 0.10 * (50 * 15)
      total_discount_2 := (seats_2 / 15) * discount_2_per_group
      extra_discount_2 := if seats_2 >= 30 then 0.03 * (cost_2 - total_discount_2) else 0
      total_discount_2 := total_discount_2 + extra_discount_2
      cost_2_discounted := cost_2 - total_discount_2

      seats_3 := 5 * 10
      cost_3 := seats_3 * 40
      discount_3_per_group := 0.08 * (40 * 10)
      total_discount_3 := (seats_3 / 10) * discount_3_per_group
      cost_3_discounted := cost_3 - total_discount_3

      total_cost := cost_1_discounted + cost_2_discounted + cost_3_discounted
  in total_cost = 18947.50 := 
by
  sorry

end school_seat_total_cost_l634_634353


namespace neither_odd_nor_even_and_min_value_at_one_l634_634090

def f (x : ℝ) : ℝ := x^3 - 3 * x + 1

theorem neither_odd_nor_even_and_min_value_at_one :
  (∀ x, f (-x) ≠ f x ∧ f (-x) ≠ - f x) ∧ ∃ x, x = 1 ∧ ∀ y, f y ≥ f x :=
by
  sorry

end neither_odd_nor_even_and_min_value_at_one_l634_634090


namespace two_digit_integers_with_prime_digits_and_first_greater_than_second_l634_634118

theorem two_digit_integers_with_prime_digits_and_first_greater_than_second :
  ∃ N : ℕ, N = 6 ∧
           (∀ x y : ℕ, (x ∈ {2, 3, 5, 7}) → (y ∈ {2, 3, 5, 7}) → (10 * x + y).is_digit ∧ x > y → N = 6) :=
by
  sorry

end two_digit_integers_with_prime_digits_and_first_greater_than_second_l634_634118


namespace angle_BAC_l634_634545

theorem angle_BAC (A B C D : Type*) (AD BD CD : ℝ) (angle_BCA : ℝ) 
  (h_AD_BD : AD = BD) (h_BD_CD : BD = CD) (h_angle_BCA : angle_BCA = 40) :
  ∃ angle_BAC : ℝ, angle_BAC = 110 := 
sorry

end angle_BAC_l634_634545


namespace trapezoid_angles_l634_634528

theorem trapezoid_angles (BC AD AC BD AB : ℝ)
  (hBCAD : AD = 4 * BC)
  (hACBD : BD = 2 * AC)
  (hPytha_AC : AB^2 + BC^2 = AC^2)
  (hPytha_BD : AB^2 + AD^2 = BD^2) :
  (∠DAB = π / 2) ∧
  (∠ABC = π / 2) ∧
  (∠CDA = Real.arctan (2 / 3)) ∧
  (∠BCD = π - Real.arctan (2 / 3)) :=
by
  sorry

end trapezoid_angles_l634_634528


namespace evaluate_expression_l634_634398

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_expression : 
  let p := ∏ i in Finset.range 12, (3 - w ^ (i + 1))
  p = 2391483 := 
by sorry

end evaluate_expression_l634_634398


namespace largest_divisor_of_n_minus_n_pow_5_l634_634425

theorem largest_divisor_of_n_minus_n_pow_5 (n : ℤ) (h_composite : ¬ prime n ∧ n > 1) : 
  ∃ k : ℤ, k = 6 ∧ ∀ n, (¬ prime n ∧ n > 1) → 6 ∣ (n^5 - n) := 
by
  sorry

end largest_divisor_of_n_minus_n_pow_5_l634_634425


namespace surface_area_ratio_l634_634872

-- Define the surface area of a sphere
def surface_area (r : ℝ) : ℝ := 4 * Real.pi * r^2

-- Given conditions
def r1 := 6
def r2 := 3

-- The theorem to prove
theorem surface_area_ratio : surface_area r1 / surface_area r2 = 4 :=
  sorry

end surface_area_ratio_l634_634872


namespace margo_total_distance_walked_l634_634209

def distance_walked (to_friend_minutes : ℕ) (return_minutes : ℕ) (average_rate_mph : ℝ) : ℝ :=
  (to_friend_minutes + return_minutes) / 60 * average_rate_mph

theorem margo_total_distance_walked : distance_walked 8 22 4 = 2 := by
  sorry

end margo_total_distance_walked_l634_634209


namespace selection_methods_for_charity_event_l634_634627

theorem selection_methods_for_charity_event :
  let n := 5
  let friday_count := 2
  let saturday_count := 1
  let sunday_count := 1
  nat.choose n friday_count * nat.choose (n - friday_count) saturday_count * nat.choose (n - friday_count - saturday_count) sunday_count = 60 := sorry

end selection_methods_for_charity_event_l634_634627


namespace max_value_y_l634_634876

theorem max_value_y (x : ℝ) (h : x < -1) : x + 1/(x + 1) ≤ -3 :=
by sorry

end max_value_y_l634_634876


namespace line_AD_parametric_equations_l634_634807

-- Define the points
def A : ℝ × ℝ × ℝ := (-3, 0, 1)
def D : ℝ × ℝ × ℝ := (1, 3, 2)

-- The theorem stating the parametric form of the line AD
theorem line_AD_parametric_equations (t : ℝ) : 
  ∃ (x y z : ℝ), (x = -3 + 4 * t) ∧ (y = 3 * t) ∧ (z = 1 + t) :=
by {
  use [(-3 + 4 * t), (3 * t), (1 + t)],
  simp,
}

end line_AD_parametric_equations_l634_634807


namespace difference_of_radii_l634_634272

-- Definitions based on the conditions provided
def radius_smaller_circle : ℝ := r
def ratio_of_areas : ℝ := 4

-- Introducing the Lean formula for the problem statement
theorem difference_of_radii (r : ℝ) : ∃ R : ℝ, (ratio_of_areas = 4) ∧ (R - radius_smaller_circle = r) := sorry

end difference_of_radii_l634_634272


namespace solve_equation_a_solve_equation_b_l634_634696

-- Part (a) Lean statement
theorem solve_equation_a (x : ℝ) : 
  x + real.sqrt ((x + 1) * (x + 2)) = 3 ↔ x = 7 / 9 := sorry

-- Part (b) Lean statement
theorem solve_equation_b (x : ℝ) : 
  x + real.sqrt ((x - 1) * x) + real.sqrt (x * (x + 1)) + real.sqrt ((x + 1) * (x - 1)) = 3 ↔ x = 25 / 24 := sorry

end solve_equation_a_solve_equation_b_l634_634696


namespace percentage_of_x_is_y_l634_634127

theorem percentage_of_x_is_y (x y : ℝ) (h : 0.5 * (x - y) = 0.4 * (x + y)) : y = 0.1111 * x := 
sorry

end percentage_of_x_is_y_l634_634127


namespace sum_areas_of_squares_l634_634747

-- Definitions to capture the conditions
variables (A C F : Type) [HasAngle A C F] -- Types with the notion of angles
variables (AC AF CF : ℝ) -- The lengths are real numbers
variable (h_right : ∠ A C F = 90) -- Angle FAC is a right angle
variable (h_CF : CF = 12) -- CF = 12 units

-- The theorem stating the required proof
theorem sum_areas_of_squares : AC^2 + AF^2 = 144 :=
by
  sorry

end sum_areas_of_squares_l634_634747


namespace sum_reciprocals_roots_eq_one_l634_634043

theorem sum_reciprocals_roots_eq_one (x1 x2 : ℝ) (h : Polynomial) :
  (h = Polynomial.C 6 - Polynomial.C 6 * Polynomial.X + Polynomial.X ^ 2) →
  (x1 + x2 = 6) → 
  (x1 * x2 = 6) →
  (1 / x1 + 1 / x2 = 1) :=
begin
  intros,
  rw [←div_eq_mul_inv, ←div_eq_mul_inv],
  calc
    1 / x1 + 1 / x2 = (x1 + x2) / (x1 * x2) : sorry
                  ... = 6 / 6                 : sorry
                  ... = 1                     : sorry
end

end sum_reciprocals_roots_eq_one_l634_634043


namespace g_at_5_l634_634256

def g : ℝ → ℝ := sorry

axiom g_property : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 3 * x + 2

theorem g_at_5 : g 5 = -20 :=
by {
  apply sorry
}

end g_at_5_l634_634256


namespace count_idempotent_mappings_l634_634483

-- Definitions based on conditions
def A : Set := {1, 2}
def is_idempotent_mapping (f : A → A) : Prop :=
  ∀ x : A, f (f x) = f x

-- The proof problem statement
theorem count_idempotent_mappings : 
  {f : A → A // is_idempotent_mapping f}.to_finset.card = 3 :=
sorry

end count_idempotent_mappings_l634_634483


namespace students_count_l634_634656

theorem students_count {x : ℕ} (h : x(x-1)(8-x) = 30) : x = 3 :=
by
  sorry

end students_count_l634_634656


namespace number_of_winning_scores_l634_634523

-- Define the problem conditions
variable (n : ℕ) (team1_scores team2_scores : Finset ℕ)

-- Define the total number of runners
def total_runners := 12

-- Define the sum of placements
def sum_placements : ℕ := (total_runners * (total_runners + 1)) / 2

-- Define the threshold for the winning score
def winning_threshold : ℕ := sum_placements / 2

-- Define the minimum score for a team
def min_score : ℕ := 1 + 2 + 3 + 4 + 5 + 6

-- Prove that the number of different possible winning scores is 19
theorem number_of_winning_scores : 
  Finset.card (Finset.range (winning_threshold + 1) \ Finset.range min_score) = 19 :=
by
  sorry -- Proof to be filled in

end number_of_winning_scores_l634_634523


namespace pattern_generalization_l634_634609

theorem pattern_generalization (n : ℕ) (h : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 :=
by
  -- TODO: The proof will be filled in later
  sorry

end pattern_generalization_l634_634609


namespace right_triangle_area_l634_634893

def roots (a b : ℝ) : Prop :=
  a * b = 12 ∧ a + b = 7

def area (A : ℝ) : Prop :=
  A = 6 ∨ A = 3 * Real.sqrt 7 / 2

theorem right_triangle_area (a b A : ℝ) (h : roots a b) : area A := 
by 
  -- The proof steps would go here
  sorry

end right_triangle_area_l634_634893


namespace kim_knit_8_sweaters_on_monday_l634_634940

variable (M : ℝ)

def sweaters_on_tuesday := M + 2
def sweaters_on_wednesday_and_thursday := (M + 2) - 4
def sweaters_on_friday := M / 2

def total_sweaters
  := M + sweaters_on_tuesday M + 2 * sweaters_on_wednesday_and_thursday M + sweaters_on_friday M

theorem kim_knit_8_sweaters_on_monday
  (h1 : total_sweaters M = 34) : M = 8 :=
sorry

end kim_knit_8_sweaters_on_monday_l634_634940


namespace number_of_common_terms_up_to_six_l634_634168

open Nat

-- Definitions of the sequences and their initial conditions.
def a (n : ℕ) : ℕ :=
  match n with
  | 0     => 1
  | 1     => 2
  | (n+2) => a n + a (n+1)

def b (n : ℕ) : ℕ :=
  match n with
  | 0     => 2
  | 1     => 1
  | (n+2) => b n + b (n+1)

-- Definition stating that the set of common terms in two sequences up to n is {2, 1, 3}
def common_terms_up_to_n (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ i => a i = b i)

-- Main statement: Prove that the number of common terms in the sequences up to 6 is 3.
theorem number_of_common_terms_up_to_six : (common_terms_up_to_n 6).card = 3 :=
  by
    sorry

end number_of_common_terms_up_to_six_l634_634168


namespace polar_to_rectangular_l634_634383

open Real
open Real.Angle

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 4) (h_θ : θ = 5 * pi / 3) :
    (r * cos θ, r * sin θ) = (2, -2 * sqrt 3) := by
  rw [h_r, h_θ]
  have h_cos : cos (5 * pi / 3) = 1 / 2 := sorry
  have h_sin : sin (5 * pi / 3) = -sqrt 3 / 2 := sorry
  rw [h_cos, h_sin]
  norm_num
  split
  norm_num
  ring
  norm_num
  ring

end polar_to_rectangular_l634_634383


namespace total_income_l634_634728

theorem total_income (I : ℝ) 
  (h1 : I * 0.225 = 40000) : 
  I = 177777.78 :=
by
  sorry

end total_income_l634_634728


namespace combined_books_total_l634_634186

def keith_books : ℕ := 20
def jason_books : ℕ := 21
def amanda_books : ℕ := 15
def sophie_books : ℕ := 30

def total_books := keith_books + jason_books + amanda_books + sophie_books

theorem combined_books_total : total_books = 86 := 
by sorry

end combined_books_total_l634_634186


namespace curves_share_x_intercept_on_Ox_l634_634537

def C1 (t : ℝ) : ℝ × ℝ := (t + 1, 1 - 2 * t)
def C2 (a θ : ℝ) : ℝ × ℝ := (a * Real.sin θ, 3 * Real.cos θ)
def x_intercept (p : ℝ × ℝ) : ℝ := p.1

theorem curves_share_x_intercept_on_Ox (a : ℝ) (ha : a > 0) :
  (∃ t, ∃ θ, C1 t = C2 a θ ∧ (C1 t).snd = 0) → a = 3 / 2 :=
by
  sorry

end curves_share_x_intercept_on_Ox_l634_634537


namespace mary_age_proof_l634_634533

theorem mary_age_proof (suzy_age_now : ℕ) (H1 : suzy_age_now = 20) (H2 : ∀ (years : ℕ), years = 4 → (suzy_age_now + years) = 2 * (mary_age + years)) : mary_age = 8 :=
by
  sorry

end mary_age_proof_l634_634533


namespace smallest_percent_error_l634_634184

noncomputable def actualDiameter := 30
noncomputable def measurementErrorPercent := 25

theorem smallest_percent_error :
  let actual_area := real.pi * (actualDiameter / 2)^2
  let lower_bound_area := real.pi * ((actualDiameter * (1 - measurementErrorPercent / 100)) / 2)^2
  (actual_area - lower_bound_area) / actual_area * 100 = 43.75 :=
by
  -- Proof omitted
  sorry

end smallest_percent_error_l634_634184


namespace product_of_roots_with_pos_real_part_l634_634511

-- Definitions based on the conditions in the problem
def roots (n : ℕ) (z : ℂ) : Set ℂ := {x | x^n = z}
def real_part_pos (x : ℂ) : Prop := x.re > 0

-- Main theorem based on the question
theorem product_of_roots_with_pos_real_part :
  (∏ x in (roots 8 (-256 : ℂ)).filter real_part_pos, x) = 16 :=
  sorry

end product_of_roots_with_pos_real_part_l634_634511


namespace trajectory_P_inclination_AB_CD_complementary_line_CD_fixed_point_l634_634433

section math_problem

variable {x y : ℝ}

def point_F := (1, 0) : ℝ × ℝ
def line_l (x : ℝ) := (x = -1)
def point_P (x y : ℝ) := (x, y) : ℝ × ℝ
def point_Q (y : ℝ) := (-1, y) : ℝ × ℝ

def QP (x y : ℝ) := (x + 1, 0)
def QF (x y : ℝ) := (2, -y)
def FP (x y : ℝ) := (x - 1, y)
def FQ (x y : ℝ) := (-2, y)

axiom condition_QF_FP (x y : ℝ) : QP x y • QF x y = FP x y • FQ x y

theorem trajectory_P :
  ∀ (x y : ℝ), (QP x y • QF x y = FP x y • FQ x y) → y^2 = 4 * x :=
by
  intros
  apply sorry

def point_M := (-1, 0) : ℝ × ℝ 

theorem inclination_AB_CD_complementary :
  ∀ (n y1 y2 y3 : ℝ), 
  (y1 * y2 = -4) → 
  (y1 * y3 = 4) →
  let k_AB := (4 / (y1 + y2)),
      k_CD := (-4 / (y1 + y2)) 
  in (k_CD + k_AB = 0) :=
by
  intros
  apply sorry

theorem line_CD_fixed_point (y1 y2 : ℝ) :
  (y1 * y2 = -4) → 
  ∃ (fixed_point : ℝ × ℝ),
  fixed_point = (1, 0) :=
by
  intros
  apply sorry

end math_problem

end trajectory_P_inclination_AB_CD_complementary_line_CD_fixed_point_l634_634433


namespace imaginary_part_not_neg_1_l634_634799

theorem imaginary_part_not_neg_1 (z : ℂ) (h : (complex.I - 1) * z = 2) : complex.im z ≠ -1 := sorry

end imaginary_part_not_neg_1_l634_634799


namespace equation_of_line_through_point_with_slope_l634_634638

-- Define the point and the slope
def point1 : ℝ × ℝ := (1, 3)
def slope : ℝ := 3

-- Define the point-slope form of the line
def point_slope_form (p : ℝ × ℝ) (m : ℝ) : ℝ × ℝ → Prop :=
  λ q, (q.snd - p.snd = m * (q.fst - p.fst))

-- The theorem to prove the equation of the line passing through point (1, 3) with slope 3
theorem equation_of_line_through_point_with_slope :
  point_slope_form point1 slope (λ x y, y - 3 = 3 * (x - 1)) :=
sorry

end equation_of_line_through_point_with_slope_l634_634638


namespace equal_segments_in_collinear_triplet_l634_634556

theorem equal_segments_in_collinear_triplet (A B C X Y : Point)
  (h_collinear : Collinear {B, X, Y, C})
  (h_bx_xy : dist B X = dist X Y)
  (h_xy_yc : dist X Y = dist Y C) :
  dist B X = dist X Y ∧ dist X Y = dist Y C :=
by
  sorry

end equal_segments_in_collinear_triplet_l634_634556


namespace sum_of_digits_of_s_l634_634951

noncomputable def num_trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem sum_of_digits_of_s :
  let s := 10 + 11 + 17 + 18 in
  let digit_sum := s.digits.sum in
  digit_sum = 11 := 
by
  let n_values := [10, 11, 17, 18]
  let s := n_values.sum 
  let digit_sum := (s.digits).sum
  have h1 : s = 56 := by sorry
  have h2 : digit_sum = 11 := by sorry
  exact h2

end sum_of_digits_of_s_l634_634951


namespace solution_l634_634255

noncomputable def problem_statement : Prop :=
  let m := 15
  let r := 65536
  m * r = 983040

theorem solution : problem_statement :=
by {
  let m := 15
  let r := 65536
  exact eq.refl (m * r)
}

end solution_l634_634255


namespace shaded_area_percentage_l634_634315

theorem shaded_area_percentage (n_shaded : ℕ) (n_total : ℕ) (hn_shaded : n_shaded = 21) (hn_total : n_total = 36) :
  ((n_shaded : ℚ) / (n_total : ℚ)) * 100 = 58.33 :=
by
  sorry

end shaded_area_percentage_l634_634315


namespace maximum_charlie_four_day_success_ratio_l634_634363

theorem maximum_charlie_four_day_success_ratio :
  ∃ (p q x y : ℕ), 
    (0 < p ∧ p < 120 ∧ 0 < q ∧ q < 80 ∧ x ≠ 200 ∧ y ≠ 200) ∧
    (15 * p + 10 * q < 2400) ∧ 
    (p + q = 239) ∧ (x + y = 400) ∧ 
    (p / x < 120 / 200 ∧ q / y < 80 / 200) ∧
    (p + q < 240) := 
begin
  sorry,
end

end maximum_charlie_four_day_success_ratio_l634_634363


namespace probability_collinear_dots_l634_634149

theorem probability_collinear_dots 
  (rows : ℕ) (cols : ℕ) (total_dots : ℕ) (collinear_sets : ℕ) (total_ways : ℕ) : 
  rows = 5 → cols = 4 → total_dots = 20 → collinear_sets = 20 → total_ways = 4845 → 
  (collinear_sets : ℚ) / total_ways = 4 / 969 :=
by
  intros hrows hcols htotal_dots hcollinear_sets htotal_ways
  sorry

end probability_collinear_dots_l634_634149


namespace evaluate_expression_l634_634403

noncomputable def w : ℂ := complex.exp (2 * real.pi * complex.I / 13)

theorem evaluate_expression :
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * 
  (3 - w^6) * (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * 
  (3 - w^11) * (3 - w^12) = 797161 :=
begin
  sorry
end

end evaluate_expression_l634_634403


namespace total_distance_of_ship_l634_634710

-- Define the conditions
def first_day_distance : ℕ := 100
def second_day_distance := 3 * first_day_distance
def third_day_distance := second_day_distance + 110
def total_distance := first_day_distance + second_day_distance + third_day_distance

-- Theorem stating that given the conditions the total distance traveled is 810 miles
theorem total_distance_of_ship :
  total_distance = 810 := by
  sorry

end total_distance_of_ship_l634_634710


namespace total_distance_is_810_l634_634713

-- Define conditions as constants
constant first_day_distance : ℕ := 100
constant second_day_distance : ℕ := 3 * first_day_distance
constant third_day_distance : ℕ := second_day_distance + 110

-- The total distance traveled in three days
constant total_distance : ℕ := first_day_distance + second_day_distance + third_day_distance

-- The theorem to prove
theorem total_distance_is_810 : total_distance = 810 := 
  by
    -- The proof steps would go here, but we insert sorry to indicate the proof is not provided
    sorry

end total_distance_is_810_l634_634713


namespace sum_of_solutions_l634_634570

section proof_problem

variables {n : ℕ} {x y : ℝ}
def is_solution (x y : ℝ) : Prop :=
  abs (x - 4) = abs (y - 8) ∧ abs (x - 8) = 3 * abs (y - 2)

theorem sum_of_solutions :
  let solutions := finset.univ.filter (λ p : ℝ × ℝ, is_solution p.1 p.2) in
  (solutions.sum (λ p, p.1 + p.2)) = 6 :=
by
  sorry

end proof_problem

end sum_of_solutions_l634_634570


namespace circumcenter_lies_on_AD_l634_634558

open EuclideanGeometry

variables {A B C D X Y : Point}
variables {ABC_circle_center : Point}

-- Given conditions
variables (hD_inside_angle_XAY : InsideAngle (∠ (X - A - Y)) D)
variables (hB_on_AX : OnRay (A - X) B)
variables (hC_on_AY : OnRay (A - Y) C)
variables (h_angle_ABC_XBD : ∠ (A - B - C) = ∠ (X - B - D))
variables (h_angle_ACB_YCD : ∠ (A - C - B) = ∠ (Y - C - D))

-- Prove that the center of circumscribed circle of triangle ABC lies on line segment AD
theorem circumcenter_lies_on_AD :
  OnSegment A D ABC_circle_center → 
  Circumcenter (Triangle.mk A B C) = ABC_circle_center :=
by
  sorry

end circumcenter_lies_on_AD_l634_634558


namespace parabola_has_one_x_intercept_l634_634117

-- Define the equation of the parabola
def parabola (y : ℝ) : ℝ :=
  -3 * y ^ 2 + 2 * y + 2

-- The theorem statement asserting there is exactly one x-intercept
theorem parabola_has_one_x_intercept : ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 := by
  sorry

end parabola_has_one_x_intercept_l634_634117


namespace PA_perp_BC_l634_634752

variables {A B C P : Point} -- vertices of triangle and point of intersection
variables {O1 O2 : Circle} -- two circles
variables {E F G H : Point} -- points of tangency

-- Conditions
axiom tangent_points : TangentToCircle A B C O1 O2 E F G H -- Tangent points of the circles to the sides of the triangle
axiom intersect_P : IntersectAt E G F H P -- Lines EG and FH intersect at P

-- Theorem to prove
theorem PA_perp_BC (triangle_ABC : Triangle ABC) :
  PA ⊥ BC :=
begin
  -- Proof needs to be constructed
  sorry
end

end PA_perp_BC_l634_634752


namespace problem_statement_l634_634541

-- Define the arithmetic sequence and the conditions
noncomputable def a : ℕ → ℝ := sorry
axiom a_arith_seq : ∃ d : ℝ, ∀ n m : ℕ, a (n + m) = a n + m • d
axiom condition : a 4 + a 10 + a 16 = 30

-- State the theorem
theorem problem_statement : a 18 - 2 * a 14 = -10 :=
sorry

end problem_statement_l634_634541


namespace average_length_of_strings_l634_634979

theorem average_length_of_strings (l1 l2 l3 : ℝ) (hl1 : l1 = 2) (hl2 : l2 = 5) (hl3 : l3 = 3) : 
  (l1 + l2 + l3) / 3 = 10 / 3 :=
by
  rw [hl1, hl2, hl3]
  change (2 + 5 + 3) / 3 = 10 / 3
  sorry

end average_length_of_strings_l634_634979


namespace initial_cakes_count_l634_634424

theorem initial_cakes_count (f : ℕ) (a b : ℕ) 
  (condition1 : f = 5)
  (condition2 : ∀ i, i ∈ Finset.range f → a = 4)
  (condition3 : ∀ i, i ∈ Finset.range f → b = 20 / 2)
  (condition4 : f * a = 2 * b) : 
  b = 40 := 
by
  sorry

end initial_cakes_count_l634_634424


namespace circles_tangent_ON_eq_2OM_l634_634170

noncomputable def curve_C (theta : ℝ) : ℝ × ℝ :=
  if theta = π / 4 then (cos theta, sin theta) else (0, 0)

noncomputable def curve_C1 (theta : ℝ) : ℝ × ℝ :=
  let rho := 2 * sin theta in (rho * cos theta, rho * sin theta)

noncomputable def curve_C2 (theta : ℝ) : ℝ × ℝ :=
  let rho := 4 * sin theta in (rho * cos theta, rho * sin theta)

noncomputable def point_M : ℝ × ℝ :=
  (1, 1)  -- Intersection of C and C1

noncomputable def point_N : ℝ × ℝ :=
  (2, 2)  -- Intersection of C and C2

theorem circles_tangent :
  let C1 := curve_C1
  let C2 := curve_C2
  let center1 := (0, 1)
  let center2 := (0, 2)
  let radius1 := 1
  let radius2 := 2
  dist center1 center2 = abs(radius2 - radius1) :=
sorry

theorem ON_eq_2OM :
  let O := (0, 0)
  dist O point_N = 2 * dist O point_M :=
sorry

end circles_tangent_ON_eq_2OM_l634_634170


namespace seventeen_divides_9x_plus_5y_l634_634957

theorem seventeen_divides_9x_plus_5y (x y : ℤ) (h : 17 ∣ (2 * x + 3 * y)) : 17 ∣ (9 * x + 5 * y) :=
sorry

end seventeen_divides_9x_plus_5y_l634_634957


namespace area_of_right_triangle_with_given_sides_l634_634886

theorem area_of_right_triangle_with_given_sides :
  let f : (ℝ → ℝ) := fun x => x^2 - 7 * x + 12
  let a := 3
  let b := 4
  let c := sqrt 7
  let hypotenuse := max a b
  let leg := min a b
in (hypotenuse = 4 ∧ leg = 3 ∧ (f(3) = 0) ∧ (f(4) = 0) → 
   (∃ (area : ℝ), (area = 6 ∨ area = (3 * sqrt 7) / 2))) :=
by
  intros
  sorry

end area_of_right_triangle_with_given_sides_l634_634886


namespace solve_x_l634_634001

noncomputable def y : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def x : ℝ := y^3

theorem solve_x :
    (sqrt (x + sqrt (x + sqrt (x + ...))) = sqrt[4] (x * sqrt[4] (x * sqrt[4] (x * ...)))) →
    x = (7 + 3 * Real.sqrt 5) / 8 :=
by
    sorry

end solve_x_l634_634001


namespace seventh_root_of_unity_value_l634_634457

-- Let z be a complex number such that z is a 7th root of unity and z ≠ 1.
variable (z : ℂ)
hypothesis h_root : z^7 = 1
hypothesis h_ne : z ≠ 1

-- Define u to be z + z^2 + z^4
noncomputable def u := z + z^2 + z^4

-- Stating the theorem
theorem seventh_root_of_unity_value :
  u = (Complex.mk (-1 / 2) (sqrt 7 / 2)) ∨ u = (Complex.mk (-1 / 2) (-sqrt 7 / 2)) :=
by sorry

end seventh_root_of_unity_value_l634_634457


namespace multiplicative_inverse_l634_634245

def A : ℕ := 123456
def B : ℕ := 162738
def N : ℕ := 503339
def modulo : ℕ := 1000000

theorem multiplicative_inverse :
  (A * B * N) % modulo = 1 :=
by
  -- placeholder for proof
  sorry

end multiplicative_inverse_l634_634245


namespace shop_profit_correct_l634_634182

def profit_per_tire_repair : ℕ := 20 - 5
def total_tire_repairs : ℕ := 300
def profit_per_complex_repair : ℕ := 300 - 50
def total_complex_repairs : ℕ := 2
def retail_profit : ℕ := 2000
def fixed_expenses : ℕ := 4000

theorem shop_profit_correct :
  profit_per_tire_repair * total_tire_repairs +
  profit_per_complex_repair * total_complex_repairs +
  retail_profit - fixed_expenses = 3000 :=
by
  sorry

end shop_profit_correct_l634_634182


namespace parabola_has_one_x_intercept_l634_634115

-- Define the equation of the parabola
def parabola (y : ℝ) : ℝ :=
  -3 * y ^ 2 + 2 * y + 2

-- The theorem statement asserting there is exactly one x-intercept
theorem parabola_has_one_x_intercept : ∃! x : ℝ, ∃ y : ℝ, parabola y = x ∧ y = 0 := by
  sorry

end parabola_has_one_x_intercept_l634_634115


namespace mabel_piggy_bank_amount_l634_634412

theorem mabel_piggy_bank_amount (n : ℕ) (h : n = 7) : 
  let quarters_per_year (k : ℕ) := k * 0.25 
  in (∑ k in finset.range (n+1), quarters_per_year k) = 7 := 
by 
  sorry

end mabel_piggy_bank_amount_l634_634412


namespace Petya_win_prob_is_1_over_256_l634_634998

/-!
# The probability that Petya will win given the conditions in the game "Heap of Stones".
-/

/-- Function representing the probability that Petya will win given the initial conditions.
Petya starts with 16 stones and takes a random number of stones each turn, while the computer
follows an optimal strategy. -/
noncomputable def Petya_wins_probability (initial_stones : ℕ) (random_choices : list ℕ) : ℚ :=
1 / 256

/-- Proof statement: The probability that Petya will win under the given conditions is 1/256. -/
theorem Petya_win_prob_is_1_over_256 : Petya_wins_probability 16 [1, 2, 3, 4] = 1 / 256 :=
sorry

end Petya_win_prob_is_1_over_256_l634_634998


namespace tangent_line_at_1_g_extrema_l634_634327

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x + 1
noncomputable def g (a b x : ℝ) : ℝ := (3 * x^2 - 3 * x - 3) * exp(-x)

theorem tangent_line_at_1 (a b : ℝ) (h1 : deriv (f a b) 1 = 2 * a) (h2 : deriv (f a b) 2 = -b) :
  ∀ x y : ℝ, (6 * x + 2 * y - 1 = 0) ↔ y = (3 * 1^2 + 2 * a * 1 + b + 1) :=
by
  sorry

theorem g_extrema (a b : ℝ) (h1 : deriv (f a b) 1 = 2 * a) (h2 : deriv (f a b) 2 = -b) :
  (∀ x : ℝ, x = 0 → g a b x = -3) ∧ (∀ x : ℝ, x = 3 → g a b x = 15 * exp(-3)) :=
by
  sorry

end tangent_line_at_1_g_extrema_l634_634327


namespace max_min_value_l634_634458

open Real

theorem max_min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ (M : ℝ), M = sqrt 2 ∧ M = min (1 / a) (min (2 / b) (min (4 / c) (cbrt (a * b * c)))) :=
sorry

end max_min_value_l634_634458


namespace derivative_at_2_l634_634053

def f (x : ℝ) : ℝ := x + 1 / x

theorem derivative_at_2 : (deriv f 2) = 3 / 4 :=
by
  sorry

end derivative_at_2_l634_634053


namespace books_loaned_out_l634_634695

/-- 
Given:
- There are 75 books in a special collection at the beginning of the month.
- By the end of the month, 70 percent of books that were loaned out are returned.
- There are 60 books in the special collection at the end of the month.
Prove:
- The number of books loaned out during the month is 50.
-/
theorem books_loaned_out (x : ℝ) (h1 : 75 - 0.3 * x = 60) : x = 50 :=
by
  sorry

end books_loaned_out_l634_634695


namespace evaluate_product_l634_634405

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_product : 
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 2657205 :=
by 
  sorry

end evaluate_product_l634_634405


namespace perimeter_parallelogram_ADEF_l634_634522

-- Defining the conditions
variables (A B C D E F : Point)
variables (AB AC BC DE EF : Line)
variables (hAB : length AB = 30) (hAC : length AC = 30) (hBC : length BC = 28)
variables (hD : (D : Point) ∈ AB) (hE : (E : Point) ∈ BC) (hF : (F : Point) ∈ AC)
variables (hDE_parallel_AC : parallel DE AC) (hEF_parallel_AB : parallel EF AB)

-- Statement to be proved
theorem perimeter_parallelogram_ADEF : 
  hAB ∧ hAC ∧ hBC ∧ hD ∧ hE ∧ hF ∧ hDE_parallel_AC ∧ hEF_parallel_AB → 
  perimeter (A, D, E, F) = 60 :=
sorry

end perimeter_parallelogram_ADEF_l634_634522


namespace flower_count_l634_634733

theorem flower_count 
  (row_front : ℕ) (row_back : ℕ) (col_left : ℕ) (col_right : ℕ)
  (total_rows : ℕ := row_front + 1 + row_back)
  (total_cols : ℕ := col_left + 1 + col_right)
  (total_flowers : ℕ := total_rows * total_cols) :
  row_front = 6 → row_back = 15 → col_left = 8 → col_right = 12 → total_flowers = 462 := 
by 
  intros hrow_front hrow_back hcol_left hcol_right
  rw [hrow_front, hrow_back, hcol_left, hcol_right]
  simp [total_rows, total_cols]
  norm_num

#print flower_count

end flower_count_l634_634733


namespace sin_angle_sum_l634_634470

theorem sin_angle_sum (α : ℝ) (h : ∃ x y : ℝ, x = -3 ∧ y = 4 ∧ (x^2 + y^2 = 25 ∧ sin α = y / 5 ∧ cos α = x / 5)) :
  sin (α + π / 4) = (√2) / 10 :=
by
  sorry

end sin_angle_sum_l634_634470


namespace constant_value_AP_AQ_l634_634804

noncomputable def ellipse_trajectory (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

noncomputable def circle_O (x y : ℝ) : Prop :=
  (x^2 + y^2) = 12 / 7

theorem constant_value_AP_AQ (x y : ℝ) (h : circle_O x y) :
  ∃ (P Q : ℝ × ℝ), ellipse_trajectory (P.1) (P.2) ∧ ellipse_trajectory (Q.1) (Q.2) ∧ 
  ((P.1 - x) * (Q.1 - x) + (P.2 - y) * (Q.2 - y)) = - (12 / 7) :=
sorry

end constant_value_AP_AQ_l634_634804


namespace negation_of_p_is_universal_l634_634479

-- Define the proposition p
def p : Prop := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

-- The proof statement for the negation of p
theorem negation_of_p_is_universal : ¬p ↔ ∀ x : ℝ, Real.exp x - x - 1 > 0 :=
by sorry

end negation_of_p_is_universal_l634_634479


namespace cos_x_plus_2y_equals_1_l634_634444

theorem cos_x_plus_2y_equals_1
  (x y : ℝ)
  (a : ℝ)
  (hxy : x ∈ set.Icc (-(Real.pi / 4)) (Real.pi / 4))
  (hyx : y ∈ set.Icc (-(Real.pi / 4)) (Real.pi / 4))
  (h1 : x^3 + Real.sin x = 2 * a)
  (h2 : 4 * y^3 + Real.sin y * Real.cos y = -a) :
  Real.cos (x + 2 * y) = 1 := 
by
  sorry

end cos_x_plus_2y_equals_1_l634_634444


namespace countable_or_finite_nonintersecting_circles_l634_634623

theorem countable_or_finite_nonintersecting_circles {Circle : Type} :
  (∀ c : Circle, ∃ p : ℚ × ℚ, p ∈ c) →
  (∀ c1 c2 : Circle, c1 ≠ c2 → ¬ ∃ p : ℚ × ℚ, p ∈ c1 ∧ p ∈ c2) →
  ∃ (S : Set Circle), (S.Finite ∨ S.Countable) :=
by
  assume (h_rational : ∀ c : Circle, ∃ p : ℚ × ℚ, p ∈ c)
  (h_distinct : ∀ c1 c2 : Circle, c1 ≠ c2 → ¬ ∃ p : ℚ × ℚ, p ∈ c1 ∧ p ∈ c2)

  sorry

end countable_or_finite_nonintersecting_circles_l634_634623


namespace math_problem_l634_634455
open Nat

-- Definitions of the sequence and conditions

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sequence_conditions (a : ℕ → ℤ) : Prop :=
  arithmetic_seq a ∧
  a 2 + a 8 = -4 ∧
  a 6 = 2

-- Definition of b_n
def b_seq (a : ℕ → ℤ) (b : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, b n = 1 / (a n * a (n + 1))

-- General term formula to be proven
def general_term_formula (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = 4 n - 22

-- Sum of the first n terms to be proven
def sum_first_n_terms (b : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, S n = -n / (36 * (2 * n - 9))

-- Combined statement
theorem math_problem (a : ℕ → ℤ) (b : ℕ → ℚ) (S : ℕ → ℚ) :
  sequence_conditions a →
  b_seq a b →
  (general_term_formula a ∧ sum_first_n_terms b S) :=
by
  sorry

end math_problem_l634_634455


namespace candy_distribution_powers_of_two_l634_634392

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def children_receive_candies (f : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ x : ℕ, ∃ k : ℕ, (k * (k + 1) / 2) % n = x

theorem candy_distribution_powers_of_two (n : ℕ) (hn : is_power_of_two n) :
  children_receive_candies (λ x, x * (x + 1) / 2 % n) n :=
sorry

end candy_distribution_powers_of_two_l634_634392


namespace find_A_l634_634016

def condition_on_A (A : ℝ) : Prop :=
  ∀ n : ℕ, ∀ m : ℤ, (m * m ≤ ⌈(A ∧ n : ℝ)⌉) ∧ (⌈(A ∧ n : ℝ)⌉ ≤ (m+1) * (m+1)) →
    (abs (⌈(A ∧ n : ℝ)⌉ - m * m) = 2 ∨ abs (⌈(A ∧ n : ℝ)⌉ - (m + 1) * (m + 1)) = 2)

theorem find_A : ∃ A : ℝ, condition_on_A A :=
  ⟨2, by sorry⟩

end find_A_l634_634016


namespace unique_third_rectangle_exists_l634_634300

-- Define the given rectangles.
def rect1_length : ℕ := 3
def rect1_width : ℕ := 8
def rect2_length : ℕ := 2
def rect2_width : ℕ := 5

-- Define the areas of the given rectangles.
def area_rect1 : ℕ := rect1_length * rect1_width
def area_rect2 : ℕ := rect2_length * rect2_width

-- Define the total area covered by the two given rectangles.
def total_area_without_third : ℕ := area_rect1 + area_rect2

-- We need to prove that there exists one unique configuration for the third rectangle.
theorem unique_third_rectangle_exists (a b : ℕ) : 
  (total_area_without_third + a * b = 34) → 
  (a * b = 4) → 
  (a = 4 ∧ b = 1 ∨ a = 1 ∧ b = 4) :=
by sorry

end unique_third_rectangle_exists_l634_634300


namespace system_solution_exists_l634_634771

theorem system_solution_exists (m : ℝ) : m ≠ 1 →
  ∃ x y z : ℝ, y = m * x + z + 2 ∧ y = (3 * m - 2) * x + z + 5 :=
begin
  sorry
end

end system_solution_exists_l634_634771


namespace new_gross_profit_percentage_l634_634714

variable (C : ℝ)
variable (SP_old SP_new : ℝ)
variable (GP_new : ℝ)

def cost_of_product (old_sp : ℝ) (profit_percentage : ℝ) : ℝ := old_sp / (1 + profit_percentage)

def gross_profit (new_sp cost : ℝ) : ℝ := new_sp - cost

def gross_profit_percentage (gp cost : ℝ) : ℝ := (gp / cost) * 100

theorem new_gross_profit_percentage :
  SP_old = 88 →
  SP_new = 92 →
  SP_old = 1.10 * C →
  gross_profit_percentage (gross_profit SP_new (cost_of_product SP_old 0.10)) (cost_of_product SP_old 0.10) = 15 := by
    sorry

end new_gross_profit_percentage_l634_634714


namespace cos_sum_zero_l634_634958

theorem cos_sum_zero (x y z w : ℝ) 
  (h1 : cos x + cos y + cos z + cos w = 0)
  (h2 : sin x + sin y + sin z + sin w = 0) : 
  cos (2 * x) + cos (2 * y) + cos (2 * z) + cos (2 * w) = 0 := 
by 
  sorry

end cos_sum_zero_l634_634958


namespace Mary_sleep_hours_for_avg_score_l634_634215

def sleep_score_inverse_relation (sleep1 score1 sleep2 score2 : ℝ) : Prop :=
  sleep1 * score1 = sleep2 * score2

theorem Mary_sleep_hours_for_avg_score (h1 s1 s2 : ℝ) (h_eq : h1 = 6) (s1_eq : s1 = 60)
  (avg_score_cond : (s1 + s2) / 2 = 75) :
  ∃ h2 : ℝ, sleep_score_inverse_relation h1 s1 h2 s2 ∧ h2 = 4 := 
by
  sorry

end Mary_sleep_hours_for_avg_score_l634_634215


namespace triple_application_of_a_l634_634129

def a (k : ℕ) : ℕ := (2 * k + 1) ^ k

theorem triple_application_of_a (k : ℕ) (h : k = 0) : a (a (a k)) = 343 :=
by
  rw h
  have h₁ : a 0 = 1 := by sorry
  have h₂ : a 1 = 3 := by sorry
  have h₃ : a 3 = 343 := by sorry
  rw [h₁, h₂, h₃]
  sorry

end triple_application_of_a_l634_634129


namespace eccentricity_of_ellipse_l634_634193

open Real

theorem eccentricity_of_ellipse :
  ∀ (a b : ℝ) (h : a > b > 0)
    (F1 F2 M : ℝ × ℝ) (gamma : ℝ) 
    (h_ellipse : ∀ {x y}, x^2 / a^2 + y^2 / b^2 = 1)
    (hf1f2 : dist M F1 = 2 * dist M F2)
    (h_right_triangle : dist M F1 ^ 2 + dist F1 F2 ^ 2 = dist M F2 ^ 2 ∨ dist M F1 ^ 2 + dist M F2 ^ 2 = dist F1 F2 ^ 2 ∨ dist F2 F1 ^ 2 + dist M F2 ^ 2 = dist M F1 ^ 2),
  let e := dist F1 F2 / (2 * a) in
  e = sqrt 3 / 3 ∨ e = sqrt 5 / 3 :=
begin
  intros,
  sorry
end

end eccentricity_of_ellipse_l634_634193


namespace select_four_numbers_l634_634448

theorem select_four_numbers (nums : List ℕ) (h_len : nums.length = 48) (h_prime_factors : ∃ p : List ℕ, p.length = 10 ∧ ∀ n ∈ nums, ∀ q ∈ p, q.prime ∧ (q ∣ n)) : 
  ∃ subset : List ℕ, subset.length = 4 ∧ (∏ x in subset, x) = k^2 for some k : ℕ :=
by
  sorry

end select_four_numbers_l634_634448


namespace workers_contribution_l634_634318

theorem workers_contribution (W C : ℕ) 
    (h1 : W * C = 300000) 
    (h2 : W * (C + 50) = 325000) : 
    W = 500 :=
by
    sorry

end workers_contribution_l634_634318


namespace find_g_l634_634648

-- Definitions for auxiliary functions to check if n is a power of 2
def is_power_of_2 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def g (n : ℕ) : ℕ :=
  if is_power_of_2 n then 2 else 3

theorem find_g (n : ℕ) (h : n ≥ 2) :
  g(n) = if is_power_of_2 n then 2 else 3 :=
begin
  -- Proof goes here
  sorry
end

end find_g_l634_634648


namespace smallest_positive_x_eq_l634_634309

theorem smallest_positive_x_eq : 
  ∃ (x : ℝ), (x > 0) ∧ (sqrt (3 * x) = 5 * x + 1) ∧ (x = ( -7 - sqrt 349) / 50) :=
begin
  sorry
end

end smallest_positive_x_eq_l634_634309


namespace fg_of_2_eq_81_l634_634495

def f (x : ℝ) : ℝ := x ^ 2
def g (x : ℝ) : ℝ := x ^ 2 + 2 * x + 1

theorem fg_of_2_eq_81 : f (g 2) = 81 := by
  sorry

end fg_of_2_eq_81_l634_634495


namespace sum_of_areas_l634_634749

-- Given: Angle FAC is a right angle, CF = 12 units
-- To Prove: The sum of the areas of the squares ACDE and AFGH is 144 square units

theorem sum_of_areas (A C F : Type) [AddGroup F] [MetricSpace F] [NormedAddTorsor F ℝ]
  (angle_FAC_right : ∡(F, A, C) = π / 2) (CF_eq_12 : dist F C = 12) :
  let ACDE_area := (dist A C) ^ 2
  let AFGH_area := (dist A F) ^ 2
  ACDE_area + AFGH_area = 144 := by
  sorry

end sum_of_areas_l634_634749


namespace sin_minus_cos_value_l634_634074

theorem sin_minus_cos_value (θ : ℝ) (hθ_pos : 0 < θ) (hθ_bound : θ < π / 4)
  (h_sum : sin θ + cos θ = 4 / 3) : sin θ - cos θ = - sqrt 2 / 3 := 
sorry

end sin_minus_cos_value_l634_634074


namespace cheryl_expense_problem_l634_634757

noncomputable def cheryl_phone_expense_difference : ℝ :=
  let E := 800 -- Electricity bill cost $800
  let C := E + X -- Cheryl's cell phone expense is $800 + X
  let G := C * 1.2 -- Golf tournament cost is 20% more than cell phone expense
  let payment := 1440 -- Total amount Cheryl paid is $1440
  G = payment → X = 400

theorem cheryl_expense_problem :
  cheryl_phone_expense_difference := sorry

end cheryl_expense_problem_l634_634757


namespace equal_daily_mileage_l634_634299

def total_boston : ℝ := 840
def total_atlanta : ℝ := 440
def days_boston_hilly : ℝ := 2
def days_boston_rain : ℝ := 2
def days_boston_rest : ℝ := 1
def days_boston_normal : ℝ := 2
def days_atlanta_hilly : ℝ := 1
def days_atlanta_rain : ℝ := 1
def days_atlanta_rest : ℝ := 1
def days_atlanta_normal : ℝ := 4

noncomputable def daily_mileage := ∀ x: ℝ, 2 * 0.8 * x + 2 * 0.9 * x + 2 * x = total_boston ∧ (0.8 * x + 0.9 * x + 4 * x = total_atlanta)

theorem equal_daily_mileage : daily_mileage 77.19 :=
by
  sorry

end equal_daily_mileage_l634_634299


namespace problem1_problem2_1_problem2_2_l634_634059

-- Define the quadratic function and conditions
def quadratic (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c

-- Problem 1: Expression of the quadratic function given vertex
theorem problem1 (b c : ℝ) : (quadratic 2 b c = 0) ∧ (∀ x : ℝ, quadratic x b c = (x - 2)^2) ↔ (b = -4) ∧ (c = 4) := sorry

-- Problem 2.1: Given n < -5 and y1 = y2, range of b + c
theorem problem2_1 (n y1 y2 b c : ℝ) (h1 : n < -5) (h2 : quadratic (3*n - 4) b c = y1)
  (h3 : quadratic (5*n + 6) b c = y2) (h4 : y1 = y2) : b + c < -38 := sorry

-- Problem 2.2: Given n < -5 and c > 0, compare values of y1 and y2
theorem problem2_2 (n y1 y2 b c : ℝ) (h1 : n < -5) (h2 : c > 0) 
  (h3 : quadratic (3*n - 4) b c = y1) (h4 : quadratic (5*n + 6) b c = y2) : y1 < y2 := sorry

end problem1_problem2_1_problem2_2_l634_634059


namespace largest_circle_on_chessboard_without_white_intersection_l634_634005

noncomputable def largest_circle_radius : ℝ :=
  1 / 2 * sqrt 10

theorem largest_circle_on_chessboard_without_white_intersection :
  ∀ (radius : ℝ), 
    (∃ (circle_center : ℝ × ℝ), 
      ∀ (x y : ℝ), (x, y) ∈ circle_center → 
      (1 / 2 * sqrt 10 <= radius)
    ) → 
    radius = 1 / 2 * sqrt 10 :=
by
  intro radius
  intro h
  sorry

end largest_circle_on_chessboard_without_white_intersection_l634_634005


namespace homework_total_time_l634_634229

theorem homework_total_time :
  ∀ (j g p : ℕ),
  j = 18 →
  g = j - 6 →
  p = 2 * g - 4 →
  j + g + p = 50 :=
by
  intros j g p h1 h2 h3
  sorry

end homework_total_time_l634_634229


namespace trigonometric_condition_l634_634521

theorem trigonometric_condition (A B C : ℝ) 
    (h1 : cos A + sin A = cos B + sin B) 
    (h2 : C = 90 * Real.pi / 180): 
    (cos A + sin A = cos B + sin B) ↔ (C = 90 * Real.pi / 180) :=
begin
  sorry
end

end trigonometric_condition_l634_634521


namespace sum_series_rel_prime_l634_634197

theorem sum_series_rel_prime (a b : ℕ) (h_rel_prime : Nat.gcd a b = 1) :
  (∑ i in (Finset.range ∞), (2 * i + 1) / 2 ^ (2 * i + 2) +
   ∑ i in (Finset.range ∞), (2 * i + 3) / 3 ^ (2 * i + 1) = a / b) → 
  a + b = 165 :=
by
  sorry

end sum_series_rel_prime_l634_634197


namespace sisters_meeting_days_l634_634162

open Finset

def eldest_returns_home_on_days : Finset ℕ := (range 20).map ⟨λ x, x * 5 + 5, sorry⟩
def middle_returns_home_on_days : Finset ℕ := (range 25).map ⟨λ x, x * 4 + 4, sorry⟩
def youngest_returns_home_on_days : Finset ℕ := (range 33).map ⟨λ x, x * 3 + 3, sorry⟩

def days_within_100 : Finset ℕ := range 101

theorem sisters_meeting_days :
  card ((eldest_returns_home_on_days ∪ middle_returns_home_on_days ∪ youngest_returns_home_on_days) ∩ days_within_100) = 60 :=
by
  sorry

end sisters_meeting_days_l634_634162


namespace ratio_breadth_length_l634_634252

theorem ratio_breadth_length (A l : ℕ) (hl : l = 60) (hA : A = 2400) :
  let b := A / l in b / l = 2 / 3 := by
  sorry

end ratio_breadth_length_l634_634252


namespace B_current_age_l634_634723

theorem B_current_age (A B : ℕ) (h1 : A = B + 15) (h2 : A - 5 = 2 * (B - 5)) : B = 20 :=
by sorry

end B_current_age_l634_634723


namespace largest_integer_less_than_a8_l634_634590

def a1 : ℝ := 3
def recurrence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ n > 1, 4 * (a (n-1) ^ 2 + a n ^ 2) = 10 * a (n-1) * a n - 9

theorem largest_integer_less_than_a8 {a : ℕ → ℝ} (h₁ : a 1 = a1) 
  (h₂ : recurrence a) : 
  ∑ x in (Icc 0 8) \ {8}, x < floor (a 8) := sorry

end largest_integer_less_than_a8_l634_634590


namespace petya_wins_probability_l634_634983

theorem petya_wins_probability : 
  ∃ p : ℚ, p = 1 / 256 ∧ 
  initial_stones = 16 ∧ 
  (∀ n, (1 ≤ n ∧ n ≤ 4)) ∧ 
  (turns(1) ∨ turns(2)) ∧ 
  (last_turn_wins) ∧ 
  (Petya_starts) ∧ 
  (Petya_plays_random ∧ Computer_plays_optimally) 
    → Petya_wins_with_probability p
:= sorry

end petya_wins_probability_l634_634983


namespace all_children_receive_candy_l634_634395

-- Define f(x) function
def f (x n : ℕ) : ℕ := ((x * (x + 1)) / 2) % n

-- Define the problem statement: prove that all children receive at least one candy if n is a power of 2.
theorem all_children_receive_candy (n : ℕ) (h : ∃ m, n = 2^m) : 
    ∀ i : ℕ, i < n → ∃ x : ℕ, i = f x n := 
sorry

end all_children_receive_candy_l634_634395


namespace part1_max_min_part2_root_sum_range_l634_634840

/-- Part (I) --/
noncomputable def f_part1 (x : ℝ) := x * abs (x + 2) + 5

theorem part1_max_min :
  let f := f_part1 in
  (∀ x ∈ set.Icc (-3 : ℝ) 0, 2 ≤ f x ∧ f x ≤ 5) :=
begin
  intro f,
  intros x hx,
  -- Proof omitted, for the coder's proof steps to follow
  sorry
end

/-- Part (II) --/
noncomputable def f_part2 (x a : ℝ) := 
  if x >= 2 * a then x^2 - 2 * a * x + a^2 - 4 * a
  else -x^2 + 2 * a * x + a^2 - 4 * a

theorem part2_root_sum_range {a x1 x2 x3 : ℝ} (h : a > 0)
  (hx1x2 : f_part2 x1 a = 0 ∧ f_part2 x2 a = 0 ∧ f_part2 x3 a = 0)
  (hx_distinct : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  let s := (1 / x1) + (1 / x2) + (1 / x3) in  
  (∃ a, (2 < a ∧ a < 4) ∧ s > (1 + real.sqrt 2) / 2) :=
begin
  intros s,
  -- Proof omitted, for the coder's proof steps to follow
  sorry
end

end part1_max_min_part2_root_sum_range_l634_634840


namespace symmetric_circle_eq_l634_634066

-- Define the given circle C
def circle_C : set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 1}

-- Define the line l
def line_l : set (ℝ × ℝ) := {p | p.1 - p.2 = 2}

-- Define the symmetric circle equation to prove it matches the condition
def symmetric_circle : set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 + 1)^2 = 1}

-- The theorem to prove that symmetric_circle is indeed the symmetric to circle_C w.r.t. line_l
theorem symmetric_circle_eq :
  symmetric_circle = {p | (p.1 - 3)^2 + (p.2 + 1)^2 = 1} :=
begin
  sorry
end

end symmetric_circle_eq_l634_634066


namespace trapezoid_proof_l634_634323

variables {R : ℝ} {A B C D E K : ℝ}

-- Conditions
def is_isosceles_trapezoid_circumscribed_around_circle (ABCD : ℝ) (R : ℝ) (E K : ℝ) : Prop :=
  ∃ O, 
  let M := (A + B) / 2 in
  ((C - D) / 2) ^ 2 + R ^ 2 = ((A - B) / 2) ^ 2 + R ^ 2 ∧
  (E + K) / 2 = M ∧ 
  (2 * R) * cos (π / 3) = R -- Angle condition simplified by cos(60°) = 1/2

-- Theorem to prove
theorem trapezoid_proof (h1 : is_isosceles_trapezoid_circumscribed_around_circle ABCD R E K)
: (E + K) / 2 = (A + B) / 2 ∧ 
  let area_ABKE := 3 * (R^2 * sqrt 3 / 4) in
  area_ABKE = (9 * R^2 * sqrt 3) / 4 :=
sorry

end trapezoid_proof_l634_634323


namespace triangle_area_l634_634739

theorem triangle_area (a b c : ℝ) (h₁ : a = 10) (h₂ : b = 24) (h₃ : c = 26) (h₄ : a ^ 2 + b ^ 2 = c ^ 2) :
  let area := (1 / 2) * a * b in
  area = 120 :=
by
  sorry

end triangle_area_l634_634739


namespace find_triangle_side_value_find_triangle_tan_value_l634_634905

noncomputable def triangle_side_value (A B C : ℝ) (a b c : ℝ) : Prop :=
  C = 2 * Real.pi / 3 ∧
  c = 5 ∧
  a = Real.sqrt 5 * b * Real.sin A ∧
  b = 2 * Real.sqrt 15 / 3

noncomputable def triangle_tan_value (B : ℝ) : Prop :=
  Real.tan (B + Real.pi / 4) = 3

theorem find_triangle_side_value (A B C a b c : ℝ) :
  triangle_side_value A B C a b c := by sorry

theorem find_triangle_tan_value (B : ℝ) :
  triangle_tan_value B := by sorry

end find_triangle_side_value_find_triangle_tan_value_l634_634905


namespace playground_girls_l634_634283

theorem playground_girls (total_children boys girls : ℕ) (h1 : boys = 40) (h2 : total_children = 117) (h3 : total_children = boys + girls) : girls = 77 := 
by 
  sorry

end playground_girls_l634_634283


namespace Petya_win_prob_is_1_over_256_l634_634996

/-!
# The probability that Petya will win given the conditions in the game "Heap of Stones".
-/

/-- Function representing the probability that Petya will win given the initial conditions.
Petya starts with 16 stones and takes a random number of stones each turn, while the computer
follows an optimal strategy. -/
noncomputable def Petya_wins_probability (initial_stones : ℕ) (random_choices : list ℕ) : ℚ :=
1 / 256

/-- Proof statement: The probability that Petya will win under the given conditions is 1/256. -/
theorem Petya_win_prob_is_1_over_256 : Petya_wins_probability 16 [1, 2, 3, 4] = 1 / 256 :=
sorry

end Petya_win_prob_is_1_over_256_l634_634996


namespace star_operation_result_l634_634039

def set_minus (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∧ x ∉ B}

def set_star (A B : Set ℝ) : Set ℝ :=
  set_minus A B ∪ set_minus B A

def A : Set ℝ := { y : ℝ | y ≥ 0 }
def B : Set ℝ := { x : ℝ | -3 ≤ x ∧ x ≤ 3 }

theorem star_operation_result :
  set_star A B = {x : ℝ | (-3 ≤ x ∧ x < 0) ∨ (x > 3)} :=
  sorry

end star_operation_result_l634_634039


namespace M_intersect_N_eq_M_l634_634485

def M (x y : ℝ) : Prop := x^2 + 2*x + y^2 ≤ 0
def N (x y : ℝ) (a : ℝ) : Prop := y ≥ x + a

theorem M_intersect_N_eq_M (a : ℝ) : (M ∩ (λ (xy : ℝ × ℝ), N xy.1 xy.2 a)) = M → a ≤ 1 - Real.sqrt 2 :=
sorry

end M_intersect_N_eq_M_l634_634485


namespace geometric_sum_four_terms_l634_634831

/-- 
Given that the sequence {a_n} is a geometric sequence with the sum of its 
first n terms denoted as S_n, if S_4=1 and S_8=4, prove that a_{13}+a_{14}+a_{15}+a_{16}=27 
-/ 
theorem geometric_sum_four_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ (n : ℕ), S (n + 1) = a (n + 1) + S n) 
  (h2 : S 4 = 1) 
  (h3 : S 8 = 4) 
  : (a 13) + (a 14) + (a 15) + (a 16) = 27 := 
sorry

end geometric_sum_four_terms_l634_634831


namespace expand_expression_l634_634413

variable (x : ℝ)

theorem expand_expression : (9 * x + 4) * (2 * x ^ 2) = 18 * x ^ 3 + 8 * x ^ 2 :=
by sorry

end expand_expression_l634_634413


namespace perpendicular_planes_sufficient_condition_l634_634048

-- Definitions of alpha and beta being distinct planes, and m being a line such that m ⊆ alpha
variables (α β : Plane) (m : Line)

-- Conditions
variables (h_distinct : α ≠ β) (h_m_in_alpha: m ⊆ α)

-- Target condition: Prove that "m ⊥ β" is a sufficient but not necessary condition for "α ⊥ β"
theorem perpendicular_planes_sufficient_condition :
  (m ⊥ β → α ⊥ β) ∧ (α ⊥ β → ¬ (m ⊥ β)) :=
sorry

end perpendicular_planes_sufficient_condition_l634_634048


namespace solve_equation_l634_634244

theorem solve_equation : ∀ x : ℝ, 2 * (3 * x - 1) = 7 - (x - 5) → x = 2 :=
by
  intro x h
  sorry

end solve_equation_l634_634244


namespace divisor_of_136_l634_634224

theorem divisor_of_136 (d : ℕ) (h : 136 = 9 * d + 1) : d = 15 := 
by {
  -- Since the solution steps are skipped, we use sorry to indicate a placeholder.
  sorry
}

end divisor_of_136_l634_634224


namespace remainder_of_expression_l634_634419

theorem remainder_of_expression :
  (8 * 7^19 + 1^19) % 9 = 3 :=
  by
    sorry

end remainder_of_expression_l634_634419


namespace wolves_hunting_l634_634657

theorem wolves_hunting 
  (pack_size : ℕ)
  (meat_per_wolf_daily : ℕ)
  (hunting_days : ℕ)
  (meat_per_deer : ℕ)
  (additional_wolves : ℕ) :
  pack_size = additional_wolves + 1 → 
  meat_per_wolf_daily = 8 →
  hunting_days = 5 →
  meat_per_deer = 200 →
  ∀ deer_needed : ℕ, (pack_size * meat_per_wolf_daily * hunting_days) ≤ (deer_needed * meat_per_deer) ∧
  (deer_needed * meat_per_deer) < ((deer_needed + 1) * meat_per_deer) →
  deer_needed = 4 →
  pack_size = 17 →
  ∃ wolves_hunting : ℕ, wolves_hunting = deer_needed :=
begin
  sorry
end

end wolves_hunting_l634_634657


namespace smallest_n_for_equal_triangles_l634_634604

/-- Problem statement:
Given a convex n-gon, what is the smallest value of n for which it is possible to divide it into 
n equal triangles using line segments from an interior point to the vertices?

Mathematically equivalent proof problem:
Prove that the smallest value of n for which a convex n-gon can be divided into 
n equal triangles using line segments from an interior point to the vertices is n = 5.
-/
theorem smallest_n_for_equal_triangles (n : ℕ) (polygon : convex_ngon n) 
  (h : ∃ interior_point, ∀ vertex ∈ vertices polygon, 
  divides_into_equal_triangles polygon interior_point vertex) : n = 5 := 
sorry

end smallest_n_for_equal_triangles_l634_634604


namespace cycle_exists_l634_634223

def Country := Type
def City := Type

variable (M N : ℕ) -- Number of countries and cities
variable [Fintype Country] [Fintype City]
variable (belongs_to : City → Country) -- City to Country mapping
variable (connected : City → City → Prop) -- Road connectivity

hypothesis (h1 : ∀ c : Country, 3 ≤ Fintype.card {x : City // belongs_to x = c})
hypothesis (h2 : ∀ c : Country, ∀ x : {x // belongs_to x = c}, 
                   Fintype.card {y : City // connected x.1 y ∧ belongs_to y = c} 
                   ≥ Fintype.card {y : City // belongs_to y = c} / 2)
hypothesis (h3 : ∀ x : City, ∃! y : City, connected x y ∧ belongs_to x ≠ belongs_to y)
hypothesis (h4 : ∀ c1 c2 : Country, c1 ≠ c2 → 
                   Fintype.card {⟨x, y⟩ : City × City // connected x y ∧ belongs_to x = c1 ∧ belongs_to y = c2} ≤ 2)
hypothesis (h5 : ∀ c1 c2 : Country, 
                   Fintype.card {x : City // belongs_to x = c1} + 
                   Fintype.card {x : City // belongs_to x = c2} < 2 * M → 
                   ∃ x : City, ∃ y : City, connected x y ∧ belongs_to x = c1 ∧ belongs_to y = c2)

theorem cycle_exists : ∃ (C : List City), 
  (∀ x ∈ C, ∃ (i : Fin C.length), x = C.nth i) ∧ 
  (∀ ⦃i j : Fin C.length⦄, i ≠ j → C.nth i ≠ C.nth j) ∧ 
  (∀ i : Fin C.length, connected (C.nth i) (C.nth (i + 1) : _)) ∧ 
  C.length ≥ M + N / 2 :=
sorry

end cycle_exists_l634_634223


namespace triangle_expression_negative_l634_634875

variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (h1 : a + b > c)
variable (h2 : a + c > b)
variable (h3 : b + c > a)

theorem triangle_expression_negative : a^2 - c^2 - 2 * a * b + b^2 < 0 :=
by
  have h4 : a > b - c, from (h1.trans (lt_add_of_pos_right c (hb.trans_le (le_self_add hb)))) sorry,
  have h5 : b - c > a, from lt_of_sub_neg (sub_lt_sub_right h2 c) sorry,
  sorry

end triangle_expression_negative_l634_634875


namespace find_function_f_l634_634965

-- Define the problem in Lean 4
theorem find_function_f (f : ℝ → ℝ) : 
  (f 0 = 1) → 
  ((∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2)) → 
  (∀ x : ℝ, f x = x + 1) :=
  by
    intros h₁ h₂
    sorry

end find_function_f_l634_634965


namespace remainder_division_l634_634685

noncomputable def findRemainder (q : Polynomial ℝ) : Polynomial ℝ :=
  if (q.eval 2 = 3) ∧ (q.eval (-3) = 2) then 
    (1 / 5) * Polynomial.X + 13 / 5
  else 
    0

theorem remainder_division (q : Polynomial ℝ) (h1 : q.eval 2 = 3) (h2 : q.eval (-3) = 2) :
  ∃ (c d : ℝ), (q = ((Polynomial.X - 2) * (Polynomial.X + 3)) * (Polynomial.X^0) + (c * Polynomial.X + d)) ∧
  c = 1 / 5 ∧ d = 13 / 5 :=
by
  sorry

end remainder_division_l634_634685


namespace parallel_lines_l634_634926

theorem parallel_lines (m : ℝ) :
  (∀ x y : ℝ, x + (m + 1) * y = 2 - m → m * x + 2 * y = -8)
  → m = 1 :=
begin
  sorry,
end

end parallel_lines_l634_634926


namespace find_P_l634_634033

def custom_operation (m n : ℤ) := m^n + n^m

theorem find_P (P : ℤ) (h : custom_operation 2 P = 100) : P = 6 := 
by
  sorry

end find_P_l634_634033


namespace min_omega_six_l634_634257

noncomputable def min_omega (ω : ℝ) : Prop :=
  ω > 0 ∧  ∃ k : ℤ, ω = 6 * k ∧ ω = min {ω' : ℝ | ω' > 0 ∧ ∃ k : ℤ, ω' = 6 * k}

theorem min_omega_six : min_omega 6 := 
by 
  sorry

end min_omega_six_l634_634257


namespace minimum_value_at_extremum_l634_634877

noncomputable def f (x a : ℝ) : ℝ := (x^2 + a * x - 1) * Real.exp (x - 1)

theorem minimum_value_at_extremum (a : ℝ) :
  (∃ x : ℝ, x = -2 ∧ (∀ h : ℝ, x ≠ h → f(x, a) > f(x, a) ∨ f(x, a) < f(x, a))) → f (-2, a) = -1 := 
sorry

end minimum_value_at_extremum_l634_634877


namespace incorrect_implication_of_angle_A_eq_60_l634_634459

-- Definitions and assumptions for parallelogram ABCD
variables {A B C D : Type}
variables [parallelogram A B C D]

-- Conditions for the options in the problem
variable (angle_A_eq_60 : angle A = 60)
variable (AB_eq_BC : AB = BC)
variable (AC_perp_BD : AC ⊥ BD)
variable (AC_eq_BD : AC = BD)

-- Proving that angle A = 60 degrees does not necessarily imply that ABCD is a rhombus
theorem incorrect_implication_of_angle_A_eq_60 : ¬(rhombus A B C D) :=
sorry

end incorrect_implication_of_angle_A_eq_60_l634_634459


namespace petya_wins_probability_l634_634989

/-- Petya plays a game with 16 stones where players alternate in taking 1 to 4 stones. 
     Petya wins if they can take the last stone first while making random choices. 
     The computer plays optimally. The probability of Petya winning is 1 / 256. -/
theorem petya_wins_probability :
  let stones := 16 in
  let optimal_strategy := ∀ n, n % 5 = 0 in
  let random_choice_probability := (1 / 4 : ℚ) in
  let total_random_choices := 4 in
  (random_choice_probability ^ total_random_choices) = (1 / 256 : ℚ) :=
by
  sorry

end petya_wins_probability_l634_634989


namespace polynomial_divisibility_l634_634423

-- Define the root properties and polynomial divisibility
theorem polynomial_divisibility (C D : ℤ) (hC : C = -1) (hD : D = -1) :
  (∀ α, α^2 - α + 1 = 0 → α^3 = 1 → α^103 + C * α^2 + D = 0) :=
by
  intros α h_alpha h_alpha_cube
  have h_alpha101 : α^103 = α := by rw [←mul_pow, h_alpha_cube, one_pow, h_alpha_cube]
  rw [h_alpha101, hC, ←h_alpha, hD]
  sorry

end polynomial_divisibility_l634_634423


namespace smallest_value_of_abs_z_minus_i_l634_634201

theorem smallest_value_of_abs_z_minus_i (z : ℂ) (h : |z^2 - 4| = |z * (z - 2 * complex.I)|) : 
  ∃ (ε : ℂ), (|ε - complex.I| = 1) := sorry

end smallest_value_of_abs_z_minus_i_l634_634201


namespace ellipse_eccentricity_range_l634_634812

theorem ellipse_eccentricity_range {a b c e : ℝ} (h : a > b ∧ b > 0) 
  (he : c^2 = -c^2 - 2a^2 * (1 - b^2 / a^2)) :
  ∃ e, e ∈ set.Ico (sqrt 3 / 3) (sqrt 2 / 2) :=
by 
  sorry

end ellipse_eccentricity_range_l634_634812


namespace unit_vector_perpendicular_to_a_l634_634122

def unit_vector_perpendicular (a : ℝ × ℝ) (u : ℝ × ℝ) : Prop :=
  (a.1 * u.1 + a.2 * u.2 = 0) ∧ (u.1^2 + u.2^2 = 1)

theorem unit_vector_perpendicular_to_a :
  unit_vector_perpendicular (2, -2) (1 / real.sqrt 2, 1 / real.sqrt 2) ∨ 
  unit_vector_perpendicular (2, -2) (-1 / real.sqrt 2, -1 / real.sqrt 2) :=
sorry

end unit_vector_perpendicular_to_a_l634_634122


namespace count_multiples_5_or_7_but_not_10_l634_634490

theorem count_multiples_5_or_7_but_not_10 (n : ℕ) (h : n = 200) : 
  (nat.filter (λ x, (x % 5 = 0 ∨ x % 7 = 0) ∧ x % 10 ≠ 0) (list.range (n+1))).length = 43 :=
sorry

end count_multiples_5_or_7_but_not_10_l634_634490


namespace negation_of_prop_l634_634266

-- Define the original proposition
def prop (x : ℝ) : Prop := x^2 - x + 2 ≥ 0

-- State the negation of the original proposition
theorem negation_of_prop : (¬ ∀ x : ℝ, prop x) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) := 
by
  sorry

end negation_of_prop_l634_634266


namespace TV_cost_is_1700_l634_634941

def hourlyRate : ℝ := 10
def workHoursPerWeek : ℝ := 30
def weeksPerMonth : ℝ := 4
def additionalHours : ℝ := 50

def weeklyEarnings : ℝ := hourlyRate * workHoursPerWeek
def monthlyEarnings : ℝ := weeklyEarnings * weeksPerMonth
def additionalEarnings : ℝ := hourlyRate * additionalHours

def TVCost : ℝ := monthlyEarnings + additionalEarnings

theorem TV_cost_is_1700 : TVCost = 1700 := sorry

end TV_cost_is_1700_l634_634941


namespace cone_volume_given_conditions_l634_634641

-- Define the conditions
def isIsoscelesRightTriangle (r l : ℝ) : Prop :=
  (l = √2 * r)

def lateralSurfaceArea (r l : ℝ) : ℝ :=
  π * r * l

-- The volume of the cone given radius, slant height, and height
def coneVolume (r h : ℝ) : ℝ :=
  (1 / 3) * π * r^2 * h

-- Main theorem to prove
theorem cone_volume_given_conditions (r l h : ℝ)
  (h_isosceles : isIsoscelesRightTriangle r l)
  (h_lateral_area : lateralSurfaceArea r l = 16 * √2 * π)
  (h_height : h = √(l^2 - r^2))
  : coneVolume r h = 64 * π / 3 :=
by
  sorry

end cone_volume_given_conditions_l634_634641


namespace find_x_l634_634630

-- conditions
variable (k : ℝ)
variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

-- proportional relationship
def proportional_relationship (k x y z : ℝ) : Prop := 
  x = (k * y^2) / z

-- initial conditions
def initial_conditions (k : ℝ) : Prop := 
  proportional_relationship k 6 1 3

-- prove x = 24 when y = 2 and z = 3 under given conditions
theorem find_x (k : ℝ) (h : initial_conditions k) : 
  proportional_relationship k 24 2 3 :=
sorry

end find_x_l634_634630


namespace no_hamiltonian_path_l634_634917

-- Definitions derived from conditions:
def airports := (fin 2014)
def divides_country_equally (a b : airports) : Prop :=
  -- Placeholder for the actual condition that the line passing through a and b
  -- divides the country into two parts with exactly 1006 airports each.
  sorry

def G : simple_graph airports := {
  adj := λ a b, divides_country_equally a b,
  sym := sorry,  -- symmetry: if a is connected to b, then b is connected to a
  loopless := sorry  -- no loops: no vertex is connected to itself
}

-- The main theorem we need to prove:
theorem no_hamiltonian_path (G : simple_graph airports)
  (hG : ∀ a b : airports, G.adj a b ↔ divides_country_equally a b) :
  ¬(∃ p : list airports, p.nodup ∧ p.length = 2014 ∧ ∀ v ∈ p, v.degree G = 1 ∨ p = [p.head, ..., v, ..., p.last]) :=
sorry

end no_hamiltonian_path_l634_634917


namespace no_real_roots_in_interval_l634_634943

variable {a b c : ℝ}

theorem no_real_roots_in_interval (ha : 0 < a) (h : 12 * a + 5 * b + 2 * c > 0) :
  ¬ ∃ α β, (2 < α ∧ α < 3) ∧ (2 < β ∧ β < 3) ∧ a * α^2 + b * α + c = 0 ∧ a * β^2 + b * β + c = 0 := by
  sorry

end no_real_roots_in_interval_l634_634943


namespace oliforum_problem_l634_634108

variable {S : Set ℕ} (h : ℕ → ℕ → ℕ) (k : ℕ → ℕ)

/-- Conditions given in the problem -/
axiom h_k_condition_i : ∀ x, k(x) = h 0 x
axiom h_k_condition_ii : k 0 = 0
axiom h_k_condition_iii : ∀ x₁ x₂, h (k x₁) x₂ = x₁

/-- Conclusion: the functions h(x, y) = y and k(x) = x satisfy all conditions -/
theorem oliforum_problem : (∀ x y, h x y = y) ∧ (∀ x, k x = x) :=
by
  sorry

end oliforum_problem_l634_634108


namespace businessman_earnings_l634_634652

theorem businessman_earnings : 
  let P : ℝ := 1000
  let day1_stock := 1000 / P
  let day2_stock := 1000 / (P * 1.1)
  let day3_stock := 1000 / (P * 1.1^2)
  let value_on_day4 stock := stock * (P * 1.1^3)
  let total_earnings := value_on_day4 day1_stock + value_on_day4 day2_stock + value_on_day4 day3_stock
  total_earnings = 3641 := sorry

end businessman_earnings_l634_634652


namespace num_five_digit_palindromes_with_even_middle_l634_634870

theorem num_five_digit_palindromes_with_even_middle :
  (∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ ∃ c', c = 2 * c' ∧ 0 ≤ c' ∧ c' ≤ 4 ∧ 10000 * a + 1000 * b + 100 * c + 10 * b + a ≤ 99999) →
  9 * 10 * 5 = 450 :=
by
  sorry

end num_five_digit_palindromes_with_even_middle_l634_634870


namespace log_sum_solution_l634_634242

theorem log_sum_solution (x : ℝ) (h : log 2 x + log 8 x + log 16 x = 10) : x = 2 ^ (120 / 19) := by sorry

end log_sum_solution_l634_634242


namespace shaded_area_percentage_l634_634314

theorem shaded_area_percentage (n_shaded : ℕ) (n_total : ℕ) (hn_shaded : n_shaded = 21) (hn_total : n_total = 36) :
  ((n_shaded : ℚ) / (n_total : ℚ)) * 100 = 58.33 :=
by
  sorry

end shaded_area_percentage_l634_634314


namespace slope_of_line_l634_634137

theorem slope_of_line : ∀ (A B : ℝ × ℝ), A = (1, 2) → B = (2, 5) → ∃ k : ℝ, k = 3 ∧ k = (B.snd - A.snd) / (B.fst - A.fst) :=
by
  intros A B hA hB
  rw [hA, hB]
  use 3
  split
  {
    rfl
  }
  {
    simp
  }

end slope_of_line_l634_634137


namespace sin_arccos_l634_634759

theorem sin_arccos {adj hyp : ℝ} (h_adj : adj = 3) (h_hyp : hyp = 5) :
    sin (arccos (adj / hyp)) = 4 / 5 :=
by
  sorry

end sin_arccos_l634_634759


namespace shortest_path_surface_unit_cube_l634_634703

open Real

-- Define the concept of a unit cube and its geometry properties.
-- We have a unit cube with edge length 1.
def unit_cube := { x : ℝ × ℝ × ℝ // 0 ≤ x.1 ∧ x.1 ≤ 1 ∧ 0 ≤ x.2 ∧ x.2 ≤ 1 ∧ 0 ≤ x.3 ∧ x.3 ≤ 1 }

-- Define vertices of unit cube in the context of surface pathfinding.
def vertex_A : unit_cube := ⟨(0, 0, 0), by simp⟩
def vertex_C1 : unit_cube := ⟨(1, 1, 1), by simp⟩

-- The shortest path on the surface of the cube between opposite vertices
-- is found by calculating the correct unfolding distance.
theorem shortest_path_surface_unit_cube : (λ A C1 : unit_cube, dist A.val C1.val = sqrt 5) vertex_A vertex_C1 :=
by sorry

end shortest_path_surface_unit_cube_l634_634703


namespace find_coefficients_sum_l634_634038

theorem find_coefficients_sum (a_0 a_1 a_2 a_3 : ℝ) (h : ∀ x : ℝ, x^3 = a_0 + a_1 * (x-2) + a_2 * (x-2)^2 + a_3 * (x-2)^3) :
  a_1 + a_2 + a_3 = 19 :=
by
  sorry

end find_coefficients_sum_l634_634038


namespace d_no_inverse_l634_634622

variable {α : Type*}
variables (d e : α) (x : α)

-- Given condition
axiom de_zero : d * e = 0

-- Statement to prove
theorem d_no_inverse (h : x * d = 1 → x * d * e = e) : ¬ ∃ x, x * d = 1 :=
by {
  sorry
}

end d_no_inverse_l634_634622


namespace fraction_value_l634_634435

theorem fraction_value
  (x y z : ℝ)
  (h1 : x / 2 = y / 3)
  (h2 : y / 3 = z / 5)
  (h3 : 2 * x + y ≠ 0) :
  (x + y - 3 * z) / (2 * x + y) = -10 / 7 := by
  -- Add sorry to skip the proof.
  sorry

end fraction_value_l634_634435


namespace avg_of_divisibles_7_to_49_div_6_l634_634755

def is_divisible_by (a b : Nat) : Prop := a % b = 0

def avg_of_divisibles_in_range (start end divisor : Nat) : Nat :=
  let nums := List.range' (start + 1) (end + 1) |>.filter (λ n => is_divisible_by n divisor)
  nums.sum / nums.length

theorem avg_of_divisibles_7_to_49_div_6 : avg_of_divisibles_in_range 7 49 6 = 30 := by
  sorry

end avg_of_divisibles_7_to_49_div_6_l634_634755


namespace quadrilateral_divided_similarity_iff_trapezoid_l634_634305

noncomputable def convex_quadrilateral (A B C D : Type) : Prop := sorry
noncomputable def is_trapezoid (A B C D : Type) : Prop := sorry
noncomputable def similar_quadrilaterals (E F A B C D : Type) : Prop := sorry

theorem quadrilateral_divided_similarity_iff_trapezoid {A B C D E F : Type}
  (h1 : convex_quadrilateral A B C D)
  (h2 : similar_quadrilaterals E F A B C D): 
  is_trapezoid A B C D ↔ similar_quadrilaterals E F A B C D :=
sorry

end quadrilateral_divided_similarity_iff_trapezoid_l634_634305


namespace dot_product_is_one_l634_634077

noncomputable def vector_a : ℝ × ℝ × ℝ := sorry
noncomputable def vector_b : ℝ × ℝ × ℝ := sorry
def theta : ℝ := real.pi / 3
def norm_a : ℝ := 1
def norm_b : ℝ := 2

theorem dot_product_is_one (ha : ∥vector_a∥ = norm_a) 
                           (hb : ∥vector_b∥ = norm_b) 
                           (ht : real.angle (⟨vector_a.1, vector_a.2, vector_a.3⟩) (⟨vector_b.1, vector_b.2, vector_b.3⟩) = theta) : 
    (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 + vector_a.3 * vector_b.3) = 1 := 
by sorry

end dot_product_is_one_l634_634077


namespace circle_radius_l634_634911

theorem circle_radius (a b : ℝ) (h_ab_length : ∃ R : ℝ, ∀ α : ℝ, arc_length_AB = α ∧ arc_length_AC = 2 * α) :
  ∃ R : ℝ, R = a^2 / sqrt (4 * a^2 - b^2) :=
sorry

end circle_radius_l634_634911


namespace blue_eyed_blonds_greater_than_population_proportion_l634_634146

variables {G_B Γ B N : ℝ}

theorem blue_eyed_blonds_greater_than_population_proportion (h : G_B / Γ > B / N) : G_B / B > Γ / N :=
sorry

end blue_eyed_blonds_greater_than_population_proportion_l634_634146


namespace recruits_line_l634_634274

theorem recruits_line
  (x y z : ℕ) 
  (hx : x + y + z + 3 = 211) 
  (hx_peter : x = 50) 
  (hy_nikolai : y = 100) 
  (hz_denis : z = 170) 
  (hxy_ratio : x = 4 * z) : 
  x + y + z + 3 = 211 :=
by
  sorry

end recruits_line_l634_634274


namespace product_of_solutions_is_16_l634_634506

noncomputable def product_of_solutions : ℂ :=
  let solutions : List ℂ := [Complex.polar 2 (Real.angle.ofDegrees 22.5),
                              Complex.polar 2 (Real.angle.ofDegrees 67.5),
                              Complex.polar 2 (Real.angle.ofDegrees 337.5),
                              Complex.polar 2 (Real.angle.ofDegrees 292.5)]
  solutions.foldl (· * ·) 1

theorem product_of_solutions_is_16 :
  product_of_solutions = 16 := 
sorry

end product_of_solutions_is_16_l634_634506


namespace huangs_tax_is_65_yuan_l634_634362

noncomputable def monthly_salary : ℝ := 2900
noncomputable def tax_free_portion : ℝ := 2000
noncomputable def tax_rate_5_percent : ℝ := 0.05
noncomputable def tax_rate_10_percent : ℝ := 0.10

noncomputable def taxable_income_amount (income : ℝ) (exemption : ℝ) : ℝ := income - exemption

noncomputable def personal_income_tax (income : ℝ) : ℝ :=
  let taxable_income := taxable_income_amount income tax_free_portion
  if taxable_income ≤ 500 then
    taxable_income * tax_rate_5_percent
  else
    (500 * tax_rate_5_percent) + ((taxable_income - 500) * tax_rate_10_percent)

theorem huangs_tax_is_65_yuan : personal_income_tax monthly_salary = 65 :=
by
  sorry

end huangs_tax_is_65_yuan_l634_634362


namespace club_election_problem_l634_634980

theorem club_election_problem
  (total_members : ℕ) (boys : ℕ) (girls : ℕ) 
  (president : total_members) 
  (vice_president : total_members)
  (secretary : total_members) :
  total_members = 24 →
  boys = 12 →
  girls = 12 →
  (boys + girls = total_members) →
  (president ∈ (boys ∪ girls)) →
  (vice_president ∈ (boys ∪ girls) ∧ vice_president ≠ president ∧ (⊕ president) = (¬⊕ vice_president)) →
  (secretary ∈ (boys ∪ girls) ∧ secretary ≠ vice_president ∧ (⊕ secretary) = (⊕ vice_president)) →
  ∃ n, n = 24 * 12 * 11 :=
sorry

end club_election_problem_l634_634980


namespace probability_three_even_dice_l634_634934

theorem probability_three_even_dice :
  let p_even := 3 / 5
  let p_odd := 2 / 5
  let ways_to_choose_three_even := Nat.choose 7 3
  let probability_exactly_three_even := ways_to_choose_three_even * (p_even ^ 3) * (p_odd ^ 4)
  probability_exactly_three_even = 15120 / 78125 :=
begin
  sorry
end

end probability_three_even_dice_l634_634934


namespace find_a_add_b_l634_634496

theorem find_a_add_b (a b : ℝ) 
  (h1 : ∀ (x : ℝ), y = a + b / (x^2 + 1))
  (h2 : (y = 3) → (x = 1)) 
  (h3 : (y = 2) → (x = 0)) : a + b = 2 :=
by
  sorry

end find_a_add_b_l634_634496


namespace Bernardo_wins_with_27_sum_of_digits_is_9_l634_634526

def operation_Bernardo (n : ℕ) : ℕ := 3 * n
def operation_Silvia (n : ℕ) : ℕ := n - 30

def smallest_initial_number_for_Bernardo_win : ℕ := 27

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits.sum

theorem Bernardo_wins_with_27 :
  (operation_Bernardo (operation_Silvia (operation_Bernardo (operation_Silvia (operation_Bernardo (operation_Silvia (operation_Bernardo (operation_Silvia (operation_Bernardo smallest_initial_number_for_Bernardo_win))))))))) < 1000 :=
  sorry

theorem sum_of_digits_is_9 :
  sum_of_digits smallest_initial_number_for_Bernardo_win = 9 :=
  sorry

end Bernardo_wins_with_27_sum_of_digits_is_9_l634_634526


namespace first_two_digits_of_1666_l634_634546

/-- Lean 4 statement for the given problem -/
theorem first_two_digits_of_1666 (y k : ℕ) (H_nonzero_k : k ≠ 0) (H_nonzero_y : y ≠ 0) (H_y_six : y = 6) :
  (1666 / 100) = 16 := by
  sorry

end first_two_digits_of_1666_l634_634546


namespace max_value_5x_minus_25x_l634_634417

noncomputable def max_value_of_expression : ℝ :=
  (1 / 4 : ℝ)

theorem max_value_5x_minus_25x :
  ∃ x : ℝ, ∀ y : ℝ, y = 5^x → (5^y - 25^y) ≤ max_value_of_expression :=
sorry

end max_value_5x_minus_25x_l634_634417


namespace max_weight_of_crates_on_trip_l634_634737

def max_crates : ℕ := 5
def min_crate_weight : ℕ := 150

theorem max_weight_of_crates_on_trip : max_crates * min_crate_weight = 750 := by
  sorry

end max_weight_of_crates_on_trip_l634_634737


namespace identify_stolen_treasure_l634_634329

-- Define the magic square arrangement
def magic_square (bags : ℕ → ℕ) :=
  bags 0 + bags 1 + bags 2 = 15 ∧
  bags 3 + bags 4 + bags 5 = 15 ∧
  bags 6 + bags 7 + bags 8 = 15 ∧
  bags 0 + bags 3 + bags 6 = 15 ∧
  bags 1 + bags 4 + bags 7 = 15 ∧
  bags 2 + bags 5 + bags 8 = 15 ∧
  bags 0 + bags 4 + bags 8 = 15 ∧
  bags 2 + bags 4 + bags 6 = 15

-- Define the stolen treasure detection function
def stolen_treasure (bags : ℕ → ℕ) : Prop :=
  ∃ altered_bag_idx : ℕ, (bags altered_bag_idx ≠ altered_bag_idx + 1)

-- The main theorem
theorem identify_stolen_treasure (bags : ℕ → ℕ) (h_magic_square : magic_square bags) : ∃ altered_bag_idx : ℕ, stolen_treasure bags :=
sorry

end identify_stolen_treasure_l634_634329


namespace positive_root_of_real_root_l634_634663

theorem positive_root_of_real_root (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : b^2 - 4*a*c ≥ 0) (h2 : c^2 - 4*b*a ≥ 0) (h3 : a^2 - 4*c*b ≥ 0) : 
  ∀ (p q r : ℝ), (p = a ∧ q = b ∧ r = c) ∨ (p = b ∧ q = c ∧ r = a) ∨ (p = c ∧ q = a ∧ r = b) →
  (∃ x : ℝ, x > 0 ∧ p*x^2 + q*x + r = 0) :=
by 
  sorry

end positive_root_of_real_root_l634_634663


namespace sqrt_neg_square_real_l634_634764

theorem sqrt_neg_square_real : ∃! (x : ℝ), -(x + 2) ^ 2 = 0 := by
  sorry

end sqrt_neg_square_real_l634_634764


namespace problem_solution_l634_634811

theorem problem_solution :
  ∀ (x y : ℚ), 
  4 * x + y = 20 ∧ x + 2 * y = 17 → 
  5 * x^2 + 18 * x * y + 5 * y^2 = 696 + 5/7 := 
by 
  sorry

end problem_solution_l634_634811


namespace lights_ratio_l634_634156

theorem lights_ratio (M S L : ℕ) (h1 : M = 12) (h2 : S = M + 10) (h3 : 118 = (S * 1) + (M * 2) + (L * 3)) :
  L = 24 ∧ L / M = 2 :=
by
  sorry

end lights_ratio_l634_634156


namespace alice_bob_meet_l634_634743

theorem alice_bob_meet (t : ℝ) 
(h1 : ∀ s : ℝ, s = 30 * t) 
(h2 : ∀ b : ℝ, b = 29.5 * 60 ∨ b = 30.5 * 60)
(h3 : ∀ a : ℝ, a = 30 * t)
(h4 : ∀ a b : ℝ, a = b):
(t = 59 ∨ t = 61) :=
by
  sorry

end alice_bob_meet_l634_634743


namespace tree_prices_solution_planting_schemes_l634_634106

open Nat

structure TreePrices (x y : ℕ) :=
  (h1 : x + 2*y = 42)
  (h2 : 2*x + y = 48)

structure PlantingScheme (a : ℕ) :=
  (total_trees : a + (20-a) = 20)
  (cost_condition : 18*a + 12*(20-a) ≤ 312)
  (tree_count_condition : a ≥ 20-a)

theorem tree_prices_solution : ∃ (x y : ℕ), TreePrices x y ∧ x = 18 ∧ y = 12 :=
by
  exists 18
  exists 12
  split
  . constructor
    · exact rfl
    · exact rfl
  . split
    . rfl
    . rfl

theorem planting_schemes : ∃ a, PlantingScheme a ∧ (10 ≤ a ∧ a ≤ 12) :=
by
  exists 10
  split
  . split
    . exact rfl
    . sorry
    . sorry
  . split
    . exact sorry
    . exact sorry

  exists 11
  split
  . split
    . exact rfl
    . sorry
    . sorry
  . split
    . exact sorry
    . exact sorry

  exists 12
  split
  . split
    . exact rfl
    . sorry
    . sorry
  . split
    . exact sorry
    . exact sorry

end tree_prices_solution_planting_schemes_l634_634106


namespace problem1_problem2_l634_634839

noncomputable def f (x a : ℝ) : ℝ := x^2 + 2*x + a

theorem problem1 (x : ℝ) : f x (-2) > 1 ↔ x ∈ (-∞ : Set ℝ) ∪ {x | x > 1} :=
by
  sorry

theorem problem2 (a : ℝ) : (∀ x ≥ 1, f x a > 0) ↔ a ∈ (-3 : ℝ, +∞ : ℝ) :=
by
  sorry

end problem1_problem2_l634_634839


namespace intervals_of_monotonicity_smallest_positive_integer_for_two_zeros_derivative_at_midpoint_l634_634089

-- Define the function f
def f (a : ℝ) (x : ℝ) := x^2 - (a - 2) * x - a * Real.log x

-- (1) Prove the intervals of monotonicity for f
theorem intervals_of_monotonicity (a : ℝ) :
  (∀ x > 0, (a ≤ 0 → f' a x > 0) ∧ 
             (a > 0 → (x > a / 2 → f' a x > 0) ∧ (0 < x < a / 2 → f' a x < 0))) := sorry

-- (2) Prove that the smallest positive integer a for which f(x) has two zeros is 3
theorem smallest_positive_integer_for_two_zeros :
  ∃ a : ℤ, a > 0 ∧ (∀ x > 0, f a x = 0) ∧ (a = 3) := sorry

-- (3) Given f(x) = c with distinct roots x1 and x2, prove f'( (x1 + x2) / 2 ) > 0
theorem derivative_at_midpoint (a c x1 x2 : ℝ) (h1 : 0 < a) (h2 : f a x1 = c) (h3 : f a x2 = c) (h4 : x1 ≠ x2) :
  f a ((x1 + x2) / 2) > 0 := sorry

end intervals_of_monotonicity_smallest_positive_integer_for_two_zeros_derivative_at_midpoint_l634_634089


namespace container_value_is_315_l634_634338

variable (p : ℕ) -- Number of pennies

def dimes := 3 * p
def quarters := 6 * p

def total_value : ℕ := 1 * p + 10 * dimes + 25 * quarters

theorem container_value_is_315 (h : 181 * p = 315 * 100) : total_value p = 31500 := by
  sorry

end container_value_is_315_l634_634338


namespace find_r_l634_634491

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = Real.log 9 / Real.log 3 := by
  sorry

end find_r_l634_634491


namespace sequence_sum_l634_634062

theorem sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, 0 < a n)
  → (∀ n : ℕ, S (n + 1) = S n + a (n + 1)) 
  → (∀ n : ℕ, a (n+1)^2 = a n * a (n+2))
  → S 3 = 13
  → a 1 = 1
  → (a 3 + a 4) / (a 1 + a 2) = 9 :=
sorry

end sequence_sum_l634_634062


namespace max_surface_area_difference_l634_634861

open Function

noncomputable def small_cuboid_surface_area (length width height : ℕ) : ℕ :=
  (length * width + length * height + width * height) * 2

theorem max_surface_area_difference :
  let original := small_cuboid_surface_area 3 2 1
  original = 22 ∧
  let final1 := small_cuboid_surface_area 6 2 1
  final1 = 32 ∧
  let final2 := small_cuboid_surface_area 4 3 1
  final2 = 31 ∧
  let final3 := small_cuboid_surface_area 3 2 2
  final3 = 22 →
  Max.max (Max.max (final1 - original) (final2 - original)) (final3 - original) = 10 :=
by 
  intros original original_eq final1 final1_eq final2 final2_eq final3 final3_eq,
  sorry

end max_surface_area_difference_l634_634861


namespace angle_of_skew_lines_in_range_l634_634250

noncomputable def angle_between_skew_lines (θ : ℝ) (θ_range : 0 < θ ∧ θ ≤ 90) : Prop :=
  θ ∈ (Set.Ioc 0 90)

-- We assume the existence of such an angle θ formed by two skew lines
theorem angle_of_skew_lines_in_range (θ : ℝ) (h_skew : true) : angle_between_skew_lines θ (⟨sorry, sorry⟩) :=
  sorry

end angle_of_skew_lines_in_range_l634_634250


namespace second_planner_cheaper_l634_634227

theorem second_planner_cheaper (x : ℕ) :
  (∀ x, 250 + 15 * x < 150 + 18 * x → x ≥ 34) :=
by
  intros x h
  sorry

end second_planner_cheaper_l634_634227


namespace monitor_width_l634_634208

theorem monitor_width (d w h : ℝ) (h_ratio : w / h = 16 / 9) (h_diag : d = 24) :
  w = 384 / Real.sqrt 337 :=
by
  sorry

end monitor_width_l634_634208


namespace cos_value_of_geometric_sequence_l634_634080

theorem cos_value_of_geometric_sequence (a : ℕ → ℝ) (r : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * r)
  (h2 : a 1 * a 13 + 2 * (a 7) ^ 2 = 5 * Real.pi) :
  Real.cos (a 2 * a 12) = 1 / 2 := 
sorry

end cos_value_of_geometric_sequence_l634_634080


namespace cylindrical_to_rectangular_and_distance_l634_634382

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

noncomputable def distance_from_origin (x y z : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2 + z^2)

theorem cylindrical_to_rectangular_and_distance :
  let cylindrical_coords := (7 : ℝ, (5 * Real.pi / 6 : ℝ), -3 : ℝ)
  let rectangular_coords := cylindrical_to_rectangular cylindrical_coords.1 cylindrical_coords.2 cylindrical_coords.3
  rectangular_coords = (-7 * Real.sqrt 3 / 2 : ℝ, 7 / 2 : ℝ, -3 : ℝ) ∧ 
  distance_from_origin rectangular_coords.1 rectangular_coords.2 rectangular_coords.3 = Real.sqrt 58 :=
by
  sorry

end cylindrical_to_rectangular_and_distance_l634_634382


namespace mark_speed_l634_634213

theorem mark_speed (distance : ℕ) (time : ℕ) (h1 : distance = 24) (h2 : time = 4) : distance / time = 6 :=
by
  rw [h1, h2]
  norm_num

end mark_speed_l634_634213


namespace cos_double_angle_identity_solve_problem_l634_634789

theorem cos_double_angle_identity : 2 * cos (π / 8) ^ 2 - 1 = cos (π / 4) :=
by sorry

theorem solve_problem : 2 * cos (π / 8) ^ 2 - 1 = sqrt 2 / 2 :=
by {
  have h := cos_double_angle_identity,
  rw cos_pi_div_four at h,
  exact h,
}

end cos_double_angle_identity_solve_problem_l634_634789


namespace distinct_division_l634_634014

theorem distinct_division (n : ℕ) (h : n ≥ 11) : 
  ∃ (A B C D E F G H I : ℕ), 
    A = 1 ∧ 
    list.nodup [A, B, C, D, E, F, G, H, I] ∧ 
    ∀ i ∈ [B, C, D, E, F, G, H, I], i > 0 ∧
    distinct_lengths n :=
sorry

end distinct_division_l634_634014


namespace middle_guards_hours_l634_634719

theorem middle_guards_hours (h1 : ∀ (first_guard_hours last_guard_hours total_night_hours : ℕ),
  first_guard_hours = 3 ∧ last_guard_hours = 2 ∧ total_night_hours = 9) : ∃ middle_guard_hours : ℕ,
  let remaining_hours := 9 - (3 + 2) in
  let middle_guard_hours := remaining_hours / 2 in
  middle_guard_hours = 2 := by
  sorry

end middle_guards_hours_l634_634719


namespace dot_product_sum_eq_l634_634946

open Real EuclideanSpace

noncomputable def u : ℝ^3 := sorry
noncomputable def v : ℝ^3 := sorry
noncomputable def w : ℝ^3 := sorry

axiom norm_u : ‖u‖ = 2
axiom norm_v : ‖v‖ = 3
axiom norm_w : ‖w‖ = 1
axiom vec_eq_zero : u + 2 • v + 3 • w = 0

theorem dot_product_sum_eq : 
  (u ⋅ v + v ⋅ w + w ⋅ u) = -49 / 44 := 
sorry

end dot_product_sum_eq_l634_634946


namespace triangular_region_area_l634_634741

theorem triangular_region_area :
  let x_intercept := 4
  let y_intercept := 6
  let area := (1 / 2) * x_intercept * y_intercept
  area = 12 :=
by
  sorry

end triangular_region_area_l634_634741


namespace wallets_inside_each_other_solution_l634_634289

-- Define the wallets and coin situation
variables (Wallet : Type) (Coin : Type)
variables (A B : Wallet) (coin : Coin)

-- Conditions
def wallet_contains_coin (W : Wallet) (c : Coin) : Prop := sorry -- Definition of a wallet containing a coin

-- The initial conditions
axiom H1 : wallet_contains_coin A coin
axiom H2 : wallet_contains_coin B coin

-- Given there is only one coin in total
variable (uniqueCoin : Coin = coin)

-- The statement we want to prove
theorem wallets_inside_each_other_solution :
  ∃ (B_in_A : Wallet), (wallet_contains_coin A B_in_A) ∧ (wallet_contains_coin B container_Coin) :=
sorry

end wallets_inside_each_other_solution_l634_634289


namespace hyperbola_eccentricity_l634_634827

theorem hyperbola_eccentricity
  (p a b : ℝ) (hp : 0 < p) (ha : 0 < a) (hb : 0 < b) (c : ℝ) (h1 : c = p / 2) 
  (h2 : ∀ P : ℝ × ℝ, P = (c, b^2 / a) → 2 * (b^2 / a) = 4 * c):
  let e := c / a in e = (sqrt 2) + 1 := 
  sorry

end hyperbola_eccentricity_l634_634827


namespace relationship_between_ys_l634_634499

theorem relationship_between_ys :
  ∀ (y1 y2 y3 : ℝ),
    (y1 = - (6 / (-2))) ∧ (y2 = - (6 / (-1))) ∧ (y3 = - (6 / 3)) →
    y2 > y1 ∧ y1 > y3 :=
by sorry

end relationship_between_ys_l634_634499


namespace find_circle_radius_l634_634913

variables (a b : ℝ)
hypothesis hab : a > 0
hypothesis hbc : b > 0
hypothesis h_abc : ∀ R:ℝ, ∀ θ1 θ2: ℝ, 2 * θ1 = θ2 → sin θ1 * (R * θ1) = a → sin θ2 * (R * θ2) = b 

theorem find_circle_radius (h : 2 * (arc_length_angle AB) = (arc_length_angle AC)) : 
  radius_circle AC AB = (a ^ 2 / (sqrt (4 * a ^ 2 - b ^ 2))) :=
sorry

end find_circle_radius_l634_634913


namespace problem_solution_l634_634481

def seq_a (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * n - 1

def seq_b (n : ℕ) : ℕ :=
  if n = 0 then 0 else (seq_a n) * (seq_a (n + 1))

def seq_c (n : ℕ) : ℚ :=
  if n = 0 then 0 else 1 / seq_b n

def sum_seq_c (n : ℕ) : ℚ :=
  if n = 0 then 0 else (finset.range n).sum (λ i, seq_c (i + 1))

theorem problem_solution (n : ℕ) (h1 : ∀ n, seq_a (n + 1) = seq_a n + 2) (h2 : seq_a 2 = 3) :
  (∃ m, seq_b n = m ∧ seq_c n = 1 / m) ∧
  sum_seq_c n = (n : ℚ) / (2 * n + 1)
:=
by
  unfold seq_a seq_b seq_c sum_seq_c
  sorry

end problem_solution_l634_634481


namespace equal_distances_l634_634596

variables {A B C D I I_A : Type}
variables [NoncomputableSpace A B C D I I_A]

-- Assuming we have a type for points and we represent conditions as follows

variables (triangle_ABC : Triangle A B C) (Γ : Circumcircle A B C)
variables (incenter_I : Incenter A B C I) (excenter_I_A : Excenter A B C I_A)
variables (angle_bisector_AD : AngleBisector A D)

-- Lean statement to show the desired equality
theorem equal_distances (h1 : OnCircumcircle D Γ)
                        (h2 : OnAngleBisector A D) :
  dist D B = dist D C ∧ dist D B = dist D I ∧ dist D I = dist D I_A :=
begin
  sorry
end

end equal_distances_l634_634596


namespace problem_part1_problem_part2_problem_part3_l634_634018

-- Define the circle passing through origin and (6,0) and tangent to line y=1
def circle_C_center : Point := {x := 3, y := -4}
def circle_C_radius : ℝ := 5
def circle_C_equation (x y : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 25

-- Define line l and point (2, -2) condition
def line_l (x y : ℝ) : Prop := x - 2*y + 4 = 0
def point_Q : Point := {x := 2, y := -2}

-- Fixed points on circle E passing through intersection with x-axis
def fixed_point1 : Point := {x := 16, y := 0}
def fixed_point2 : Point := {x := -4, y := 0}

theorem problem_part1 (x y : ℝ) : circle_C_equation x y ↔ (x - 3)^2 + (y + 4)^2 = 25 :=
by infer_instance

theorem problem_part2 (x y : ℝ) (a b : ℝ) : 
    (|PT| = |PQ| → line_l a b) ↔ (a-2*b+4=0) :=
by infer_instance

theorem problem_part3 (x y : ℝ) : 
    (circle_C_equation (-4,0) ∧
     (∃ y1 y2, y1*y2 = -100 ∧ ∀ y, (x - 6)^2 + (y - (y1+y2)/2)^2 = (y1-y2)^2/4)) →
    (∃ P : Point, P = fixed_point1 ∨ P = fixed_point2) :=
by infer_instance

end problem_part1_problem_part2_problem_part3_l634_634018


namespace shaded_percentage_six_by_six_grid_l634_634313

theorem shaded_percentage_six_by_six_grid (total_squares shaded_squares : ℕ)
    (h_total : total_squares = 36) (h_shaded : shaded_squares = 21) : 
    (shaded_squares.to_rat / total_squares.to_rat) * 100 = 58.33 := 
by
  sorry

end shaded_percentage_six_by_six_grid_l634_634313


namespace recyclable_cans_and_bottles_collected_l634_634660

-- Define the conditions in Lean
def people_at_picnic : ℕ := 90
def soda_cans : ℕ := 50
def plastic_bottles_sparkling_water : ℕ := 50
def glass_bottles_juice : ℕ := 50
def guests_drank_soda : ℕ := people_at_picnic / 2
def guests_drank_sparkling_water : ℕ := people_at_picnic / 3
def juice_consumed : ℕ := (glass_bottles_juice * 4) / 5

-- The theorem statement
theorem recyclable_cans_and_bottles_collected :
  (soda_cans + guests_drank_sparkling_water + juice_consumed) = 120 :=
by
  sorry

end recyclable_cans_and_bottles_collected_l634_634660


namespace num_true_converse_props_l634_634744

def cond_1 : Prop := ∀ l1 l2 : Line, consecutive_interior_angles_supplementary l1 l2 → parallel l1 l2
def cond_2 : Prop := ∀ Δ1 Δ2 : Triangle, congruent Δ1 Δ2 → perimeter Δ1 = perimeter Δ2
def cond_3 : Prop := ∀ α β : Angle, equal α β → vertical_angles α β
def cond_4 : Prop := ∀ m n : ℤ, m = n → m^2 = n^2

def converse_cond_1 : Prop := ∀ l1 l2 : Line, parallel l1 l2 → consecutive_interior_angles_supplementary l1 l2
def converse_cond_2 : Prop := ∀ Δ1 Δ2 : Triangle, perimeter Δ1 = perimeter Δ2 → congruent Δ1 Δ2
def converse_cond_3 : Prop := ∀ α β : Angle, vertical_angles α β → equal α β
def converse_cond_4 : Prop := ∀ m n : ℤ, m^2 = n^2 → m = n

theorem num_true_converse_props : (converse_cond_1 ∧ converse_cond_3) ∧ ¬(converse_cond_2 ∧ converse_cond_4) → true_converse_prop_count = 2 := by
  sorry

end num_true_converse_props_l634_634744


namespace complex_div_conjugate_l634_634791

theorem complex_div_conjugate (a b : ℂ) (h1 : a = 2 - I) (h2 : b = 1 + 2 * I) :
    a / b = -I := by
  sorry

end complex_div_conjugate_l634_634791


namespace vectors_form_basis_l634_634073

open Real

noncomputable def e1 : ℝ × ℝ := (1, 0)
noncomputable def e2 : ℝ × ℝ := (0, 1)

def not_collinear (u v : ℝ × ℝ) : Prop :=
  ∀ k : ℝ, u ≠ k • v

def vector_a (e1 e2 : ℝ × ℝ) : ℝ × ℝ := 
  (e1.1 + 2 * e2.1, e1.2 + 2 * e2.2)

def vector_b (e1 e2 : ℝ × ℝ) (λ : ℝ) : ℝ × ℝ := 
  (2 * e1.1 + λ * e2.1, 2 * e1.2 + λ * e2.2)

theorem vectors_form_basis (h : not_collinear e1 e2) (λ : ℝ) :
  λ ≠ 4 ↔ not_collinear (vector_a e1 e2) (vector_b e1 e2 λ) := 
by 
  sorry

end vectors_form_basis_l634_634073


namespace total_distance_of_ship_l634_634711

-- Define the conditions
def first_day_distance : ℕ := 100
def second_day_distance := 3 * first_day_distance
def third_day_distance := second_day_distance + 110
def total_distance := first_day_distance + second_day_distance + third_day_distance

-- Theorem stating that given the conditions the total distance traveled is 810 miles
theorem total_distance_of_ship :
  total_distance = 810 := by
  sorry

end total_distance_of_ship_l634_634711


namespace curve_self_intersection_l634_634751

def curve_crosses_itself_at_point (x y : ℝ) : Prop :=
∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ (t₁^2 - 4 = x) ∧ (t₁^3 - 6 * t₁ + 7 = y) ∧ (t₂^2 - 4 = x) ∧ (t₂^3 - 6 * t₂ + 7 = y)

theorem curve_self_intersection : curve_crosses_itself_at_point 2 7 :=
sorry

end curve_self_intersection_l634_634751


namespace investment_order_l634_634616

noncomputable def initial_value_AA := 200
noncomputable def initial_value_BB := 150
noncomputable def initial_value_CC := 100

noncomputable def first_year_AA := initial_value_AA * 1.15
noncomputable def first_year_BB := initial_value_BB * 0.80
def first_year_CC := initial_value_CC

noncomputable def second_year_AA := first_year_AA * 0.90
noncomputable def second_year_BB := first_year_BB * 1.30
def second_year_CC := first_year_CC

noncomputable def dividends_AA := first_year_AA * 0.05

noncomputable def final_value_AA := second_year_AA + dividends_AA
noncomputable def final_value_BB := second_year_BB
def final_value_CC := second_year_CC

theorem investment_order :
  final_value_CC < final_value_BB ∧ final_value_BB < final_value_AA :=
by
  unfold final_value_AA final_value_BB final_value_CC dividends_AA second_year_AA second_year_BB first_year_AA first_year_BB initial_value_AA initial_value_BB initial_value_CC
  linarith

end investment_order_l634_634616


namespace sin_shift_l634_634668

theorem sin_shift (x : ℝ) : (sin (2 * x + 1)) = sin (2 * (x + 1 / 2)) := 
by sorry

end sin_shift_l634_634668


namespace product_of_roots_with_pos_real_part_l634_634512

-- Definitions based on the conditions in the problem
def roots (n : ℕ) (z : ℂ) : Set ℂ := {x | x^n = z}
def real_part_pos (x : ℂ) : Prop := x.re > 0

-- Main theorem based on the question
theorem product_of_roots_with_pos_real_part :
  (∏ x in (roots 8 (-256 : ℂ)).filter real_part_pos, x) = 16 :=
  sorry

end product_of_roots_with_pos_real_part_l634_634512


namespace length_of_BC_l634_634226

-- Defining constants and parameters
def radius := 14
def sin_alpha := (Real.sqrt 33) / 7
def alpha := Real.asin sin_alpha
def cos_alpha := Real.sqrt (1 - sin_alpha ^ 2)

-- Hypothesis: B and C lie on the circle, and the angles are as described.
theorem length_of_BC :
  let ℝ := 14 in -- Radius of the circle
  let α := Real.asin (Real.sqrt 33 / 7) in -- The angle α in radians
  let cos_alpha := Real.sqrt (1 - (Real.sqrt 33 / 7) ^ 2) in
  2 * radius * cos_alpha = 16 := 
sorry

end length_of_BC_l634_634226


namespace cost_difference_is_35_88_usd_l634_634607

/-
  Mr. Llesis bought 50 kilograms of rice at different prices per kilogram from various suppliers.
  He bought:
  - 15 kilograms at €1.2 per kilogram from Supplier A
  - 10 kilograms at €1.4 per kilogram from Supplier B
  - 12 kilograms at €1.6 per kilogram from Supplier C
  - 8 kilograms at €1.9 per kilogram from Supplier D
  - 5 kilograms at €2.3 per kilogram from Supplier E

  He kept 7/10 of the total rice in storage and gave the rest to Mr. Everest.
  The current conversion rate is €1 = $1.15.
  
  Prove that the difference in cost in US dollars between the rice kept and the rice given away is $35.88.
-/

def euros_to_usd (euros : ℚ) : ℚ :=
  euros * (115 / 100)

def total_cost : ℚ := 
  (15 * 1.2) + (10 * 1.4) + (12 * 1.6) + (8 * 1.9) + (5 * 2.3)

def cost_kept : ℚ := (7/10) * total_cost
def cost_given : ℚ := (3/10) * total_cost

theorem cost_difference_is_35_88_usd :
  euros_to_usd cost_kept - euros_to_usd cost_given = 35.88 := 
sorry

end cost_difference_is_35_88_usd_l634_634607


namespace olivia_change_received_l634_634612

theorem olivia_change_received 
    (cost_per_basketball_card : ℕ)
    (basketball_card_count : ℕ)
    (cost_per_baseball_card : ℕ)
    (baseball_card_count : ℕ)
    (bill_amount : ℕ) :
    basketball_card_count = 2 → 
    cost_per_basketball_card = 3 → 
    baseball_card_count = 5 →
    cost_per_baseball_card = 4 →
    bill_amount = 50 →
    bill_amount - (basketball_card_count * cost_per_basketball_card + baseball_card_count * cost_per_baseball_card) = 24 := 
by {
    intros h1 h2 h3 h4 h5,
    rw [h1, h2, h3, h4, h5],
    norm_num,
}

-- Adding a placeholder proof:
-- by sorry

end olivia_change_received_l634_634612


namespace solution_form_l634_634781

-- Define a function from positive reals to positive reals
def f : ℝ → ℝ := sorry

-- Define our positive reals
def ℝ₊ := { x : ℝ // 0 < x }

-- Define the main condition of the problem
def condition (f : ℝ₊ → ℝ₊) : Prop :=
  ∀ x y : ℝ₊, f x * f (⟨y.val * f x, mul_pos y.property (f x).property⟩) = f ⟨x.val + y.val, add_pos x.property y.property⟩

-- Prove that any function satisfying the condition has the form described
theorem solution_form (f : ℝ₊ → ℝ₊) (h : condition f) :
  ∃ λ : ℝ, 0 < λ ∧ (∀ x : ℝ₊, f x = ⟨1 / (1 + λ * x.val), div_pos one_pos (add_pos one_pos (mul_pos zero_lt_one x.property))⟩) :=
sorry

end solution_form_l634_634781


namespace hyperbola_eccentricity_sqrt2_l634_634966

variables {a b c m n : ℝ}

def hyperbola (a b : ℝ) := forall (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1

def left_focus (c : ℝ) := -(c, 0 : ℝ × ℝ)

def point_M (c a b: ℝ) := -(c, b * c / a : ℝ × ℝ)

def point_N (c a b: ℝ) := -(c, -b * c / a : ℝ × ℝ)

def point_P (c a b: ℝ) := -(c, b^2 / a : ℝ × ℝ)

def vector_relation (m n : ℝ) (a b c: ℝ) := -(c, b^2 / a : ℝ × ℝ) = m * -(c, b * c / a : ℝ × ℝ) + n * -(c, -b * c / a : ℝ × ℝ)

def relation_mn (m n : ℝ) := m * n = 1 / 8

noncomputable def eccentricity (c a : ℝ) := c / a

theorem hyperbola_eccentricity_sqrt2 (a b c : ℝ) (h1: a > 0) (h2: b > 0)
(h3 : hyperbola a b)
(h4 : left_focus c)
(h5 : point_M c a b)
(h6 : point_N c a b)
(h7 : point_P c a b)
(h8 : vector_relation m n a b c)
(h9 : relation_mn m n) :
eccentricity c a = real.sqrt 2 := sorry

end hyperbola_eccentricity_sqrt2_l634_634966


namespace sector_area_proof_l634_634460

-- Define variables for the central angle, arc length, and derived radius
variables (θ L : ℝ) (r A: ℝ)

-- Define the conditions given in the problem
def central_angle_condition : Prop := θ = 2
def arc_length_condition : Prop := L = 4
def radius_condition : Prop := r = L / θ

-- Define the formula for the area of the sector
def area_of_sector_condition : Prop := A = (1 / 2) * r^2 * θ

-- The theorem that needs to be proved
theorem sector_area_proof :
  central_angle_condition θ ∧ arc_length_condition L ∧ radius_condition θ L r ∧ area_of_sector_condition r θ A → A = 4 :=
by
  sorry

end sector_area_proof_l634_634460


namespace acute_angle_between_planes_l634_634017

noncomputable def normal_vector1 : ℝ × ℝ × ℝ := (2, -1, -3)
noncomputable def normal_vector2 : ℝ × ℝ × ℝ := (1, 1, 0)

open RealInnerProductSpace

theorem acute_angle_between_planes : 
  let n1 := normal_vector1
  let n2 := normal_vector2
  let cos_phi := (n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3) / 
                (real.sqrt (n1.1^2 + n1.2^2 + n1.3^2) * real.sqrt (n2.1^2 + n2.2^2 + n2.3^2))
  ∃ α, α = real.arccos cos_phi ∧ α = real.arccos (1 / (2 * real.sqrt 7)) :=
by
  let n1 := normal_vector1
  let n2 := normal_vector2
  have cos_phi_def : cos_phi = (n1.1 * n2.1 + n1.2 * n2.2 + n1.3 * n2.3) / 
                        (real.sqrt (n1.1^2 + n1.2^2 + n1.3^2) * real.sqrt (n2.1^2 + n2.2^2 + n2.3^2)), from sorry,
  obtain ⟨α, hα⟩ := ⟨real.arccos cos_phi, by sorry⟩,
  exact ⟨α, hα.1, by sorry⟩

end acute_angle_between_planes_l634_634017


namespace petya_wins_probability_l634_634991

def stones_initial : ℕ := 16

def valid_moves : set ℕ := {1, 2, 3, 4}

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def petya_random_choice (move : ℕ) : Prop := move ∈ valid_moves

def computer_optimal_strategy (n : ℕ) : ℕ :=
  (n % 5)

noncomputable def probability_petya_wins : ℚ :=
  (1 / 4) ^ 4

theorem petya_wins_probability :
  probability_petya_wins = 1 / 256 := 
sorry

end petya_wins_probability_l634_634991


namespace matrix_B3_is_zero_unique_l634_634583

theorem matrix_B3_is_zero_unique (B : Matrix (Fin 2) (Fin 2) ℝ) (h : B^4 = 0) :
  ∃! (B3 : Matrix (Fin 2) (Fin 2) ℝ), B3 = B^3 ∧ B3 = 0 := sorry

end matrix_B3_is_zero_unique_l634_634583


namespace simplified_expression_l634_634692

theorem simplified_expression (x : ℝ) (h : x ≥ 0) :
    Real.root 4 (6 * x * (5 + 2 * Real.sqrt 6)) * Real.sqrt (3 * Real.sqrt (2 * x) - 2 * Real.sqrt (3 * x)) = 
    Real.sqrt (6 * x) :=
sorry

end simplified_expression_l634_634692


namespace sum_of_proper_divisors_less_than_100_of_780_l634_634003

def is_divisor (n d : ℕ) : Bool :=
  d ∣ n

def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ d => d ∣ n ∧ d < n)

def proper_divisors_less_than (n bound : ℕ) : List ℕ :=
  (proper_divisors n).filter (λ d => d < bound)

def sum_list (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => acc + x) 0

theorem sum_of_proper_divisors_less_than_100_of_780 :
  sum_list (proper_divisors_less_than 780 100) = 428 :=
by
  sorry

end sum_of_proper_divisors_less_than_100_of_780_l634_634003


namespace intersection_line_slope_eq_one_l634_634032

theorem intersection_line_slope_eq_one :
  let C₁ := {p : ℝ × ℝ | let (x, y) := p in x^2 + y^2 - 6 * x + 4 * y - 20 = 0}
  let C₂ := {p : ℝ × ℝ | let (x, y) := p in x^2 + y^2 - 10 * x + 8 * y + 24 = 0}
  let L := {p : ℝ × ℝ | let (x, y) := p in y - x = -11}
  (∃ p : ℝ × ℝ, p ∈ C₁ ∧ p ∈ C₂) → -- To ensure the circles intersect
  (∀ p : ℝ × ℝ, p ∈ C₁ ∧ p ∈ C₂ → p ∈ L) →
  ∃ m : ℝ, m = 1 :=
by
  intros C₁ C₂ L h_int h_L
  use 1
  sorry

end intersection_line_slope_eq_one_l634_634032


namespace range_of_f_l634_634135

noncomputable def f (θ x : ℝ) : ℝ := 2 * sin (x + 2 * θ) * cos x

theorem range_of_f (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2) (h : f θ 0 = 2) : 
  ∀ x, 0 ≤ f θ x ∧ f θ x ≤ 2 :=
sorry

end range_of_f_l634_634135


namespace race_duration_l634_634275

theorem race_duration 
  (lap_distance : ℕ) (laps : ℕ)
  (award_per_hundred_meters : ℝ) (earn_rate_per_minute : ℝ)
  (total_distance : ℕ) (total_award : ℝ) (duration : ℝ) :
  lap_distance = 100 →
  laps = 24 →
  award_per_hundred_meters = 3.5 →
  earn_rate_per_minute = 7 →
  total_distance = lap_distance * laps →
  total_award = (total_distance / 100) * award_per_hundred_meters →
  duration = total_award / earn_rate_per_minute →
  duration = 12 := 
by 
  intros;
  sorry

end race_duration_l634_634275


namespace circle_radius_l634_634912

theorem circle_radius (a b : ℝ) (h_ab_length : ∃ R : ℝ, ∀ α : ℝ, arc_length_AB = α ∧ arc_length_AC = 2 * α) :
  ∃ R : ℝ, R = a^2 / sqrt (4 * a^2 - b^2) :=
sorry

end circle_radius_l634_634912


namespace hyperbola_eccentricity_l634_634828

-- Definitions translated from conditions
noncomputable def parabola_focus : ℝ × ℝ := (0, -Real.sqrt 5)
noncomputable def a : ℝ := 2
noncomputable def c : ℝ := Real.sqrt 5

-- Eccentricity formula for the hyperbola
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

-- Statement to be proved
theorem hyperbola_eccentricity :
  eccentricity c a = Real.sqrt 5 / 2 :=
by
  sorry

end hyperbola_eccentricity_l634_634828


namespace exists_num_div_k_exists_ones_div_k_l634_634234

theorem exists_num_div_k (k : ℕ) (h_pos : k > 0) : 
  ∃ n : ℕ, (∀ d ∈ n.digits 10, d = 0 ∨ d = 1) ∧ k ∣ n :=
sorry

theorem exists_ones_div_k (k : ℕ) (h_pos : k > 0) (h_coprime : Nat.gcd k 10 = 1) : 
  ∃ n : ℕ, (∀ d ∈ n.digits 10, d = 1) ∧ k ∣ n :=
sorry

end exists_num_div_k_exists_ones_div_k_l634_634234


namespace not_polynomial_VX_l634_634770

theorem not_polynomial_VX (P : Polynomial ℝ) :
  (P : ℝ[X]) ≠ (λ X : ℝ, (1 / (X^2 + 1))) :=
sorry

end not_polynomial_VX_l634_634770


namespace arithmetic_series_sum_l634_634788

variable (a₁ aₙ d S : ℝ)
variable (n : ℕ)

-- Defining the conditions (a₁, aₙ, d, and the formula for arithmetic series sum)
def first_term : a₁ = 10 := sorry
def last_term : aₙ = 70 := sorry
def common_diff : d = 1 / 7 := sorry

-- Equation to find number of terms (n)
def find_n : 70 = 10 + (n - 1) * (1 / 7) := sorry

-- Formula for the sum of an arithmetic series
def series_sum : S = (n * (10 + 70)) / 2 := sorry

-- The proof problem statement
theorem arithmetic_series_sum : 
  a₁ = 10 → 
  aₙ = 70 → 
  d = 1 / 7 → 
  (70 = 10 + (n - 1) * (1 / 7)) → 
  S = (n * (10 + 70)) / 2 → 
  S = 16840 := by 
  intros h1 h2 h3 h4 h5 
  -- proof steps would go here
  sorry

end arithmetic_series_sum_l634_634788


namespace max_value_trig_expression_l634_634027

theorem max_value_trig_expression : ∀ x : ℝ, (3 * Real.cos x + 4 * Real.sin x) ≤ 5 := 
sorry

end max_value_trig_expression_l634_634027


namespace petya_wins_probability_l634_634981

theorem petya_wins_probability : 
  ∃ p : ℚ, p = 1 / 256 ∧ 
  initial_stones = 16 ∧ 
  (∀ n, (1 ≤ n ∧ n ≤ 4)) ∧ 
  (turns(1) ∨ turns(2)) ∧ 
  (last_turn_wins) ∧ 
  (Petya_starts) ∧ 
  (Petya_plays_random ∧ Computer_plays_optimally) 
    → Petya_wins_with_probability p
:= sorry

end petya_wins_probability_l634_634981


namespace prob_area_twice_circumference_l634_634290

theorem prob_area_twice_circumference: 
  (let D_roll := ([1, 2, 3, 4, 5, 6].sum + 1 + [1, 2, 3, 4, 5, 6, 7, 8].sum) in 
    D_roll = 8) -> 
    1 / 96 :=
  by
    sorry

end prob_area_twice_circumference_l634_634290


namespace m_n_p_sum_l634_634855

-- Defining the conditions
def perpendicular_lines (L1 L2 : ℝ × ℝ × ℝ) : Prop := 
  let (a, b, c) := L1 in
  let (d, e, f) := L2 in 
  (a * e) + (b * d) = 0

def point_on_line (p : ℝ × ℝ) (L : ℝ × ℝ × ℝ) : Prop :=
  let (x, y) := p in
  let (a, b, c) := L in
  (a * x + b * y + c = 0)

-- Specific lines and point conditions
def L1 : ℝ × ℝ × ℝ := (2, m, -1)
def L2 : ℝ × ℝ × ℝ := (3, -2, n)
def P := (2, p)

-- Proving the main theorem
theorem m_n_p_sum 
  (m n p : ℝ)
  (H_perpendicular: perpendicular_lines L1 L2)
  (H_on_L1: point_on_line P L1)
  (H_on_L2: point_on_line P L2):
  m + n + p = -6 := by
  sorry

end m_n_p_sum_l634_634855


namespace bisect_AC_by_PQ_l634_634643

-- Definitions for points, vectors, and midpoints
variables (Point : Type) [AddCommGroup Point] [Module ℝ Point]

def midpoint (P Q : Point) : Point := (P + Q) / 2

-- Points A, B, C, D
variables (A B C D : Point)

-- Midpoints M, N, P, Q, and E
noncomputable def M : Point := midpoint A B
noncomputable def N : Point := midpoint C D
noncomputable def P : Point := midpoint B D
noncomputable def Q : Point := midpoint M N
noncomputable def E : Point := midpoint A C

-- The statement we need to prove
theorem bisect_AC_by_PQ : ∃ λ (t : Point), E = P + t • (Q - P) :=
sorry

end bisect_AC_by_PQ_l634_634643


namespace cube_painting_possible_l634_634560

/-- It is possible to paint the faces of a cube with three colors such that each color is
    present on the cube, but from any viewpoint where you can see three faces that share a 
    common vertex, it is impossible to see faces of all three colors simultaneously. -/
theorem cube_painting_possible : ∃ (color : fin 6 → fin 3),
  (∀ i, ∃ j, color i = j) ∧
  (∀ v : fin 8, ∃ c1 c2 : fin 3, 
    (∃ f1 f2 f3 : fin 6, shares_vertex v f1 f2 f3 ∧ 
     color f1 = c1 ∧ color f2 = c1 ∧ color f2 = c2) ∧
    ¬(∃ c3 : fin 3, c1 ≠ c3 ∧ c2 ≠ c3)) :=
sorry

end cube_painting_possible_l634_634560


namespace hyperbola_equation_dot_product_zero_triangle_area_l634_634447

/-- Given a hyperbola centered at the origin, with foci on the coordinate axes,
    eccentricity e = √2, and passing through the point P(4, -√10),
    prove that its equation is x^2 - y^2 = 6. -/
theorem hyperbola_equation :
    let e := real.sqrt 2 in
    let P := (4 : ℝ, - real.sqrt 10) in
    ∃ λ : ℝ, (λ = 6) ∧ (P.1 ^ 2 - P.2 ^ 2 = λ) := sorry

/-- Given the same hyperbola by its equation x^2 - y^2 = 6,
    and a point M(3, m) on the hyperbola,
    prove that the dot product of vectors MF₁ and MF₂ is 0. -/
theorem dot_product_zero (m : ℝ) (M := (3, m)) :
    let F₁ := (- real.sqrt 6, 0) in
    let F₂ := (real.sqrt 6, 0) in
    (M.1 ^ 2 - M.2 ^ 2 = 6) →
    let MF₁ := (M.1 - F₁.1, M.2 - F₁.2) in
    let MF₂ := (M.1 - F₂.1, M.2 - F₂.2) in
    (MF₁.1 * MF₂.1 + MF₁.2 * MF₂.2) = 0 := sorry

/-- Calculate the area of the triangle formed by the points F₁, M, and F₂
    for point M(3, m) on the hyperbola x^2 - y^2 = 6. -/
theorem triangle_area (m : ℝ) :
    let e := real.sqrt 2 in
    let F₁ := (- real.sqrt 6, 0) in
    let F₂ := (real.sqrt 6, 0) in
    ∃ (M : ℝ × ℝ), (M = (3, m)) ∧ (M.1 ^ 2 - M.2 ^ 2 = 6) →
    let base := 2 * real.sqrt 6 in
    let height := abs m in
    (1 / 2 * base * height = 3 * real.sqrt 2) := sorry

end hyperbola_equation_dot_product_zero_triangle_area_l634_634447


namespace angle_OQP_90_l634_634384

open EuclideanGeometry

variables {A B C D P Q O : Point}

-- Define the statements of being inscribed and intersecting
def isCyclicQuadrilateral (A B C D : Point) : Prop :=
  Cyclic_Quadrilateral A B C D

def intersectAt (AC BD : Line) (P : Point) : Prop :=
  ∃ A C B D : Point, P ∈ AC ∧ P ∈ BD

def circumcirclesIntersect (ABP CDP : Circumcircle) (P Q : Point) : Prop :=
  ∃ A B C D : Point, P ≠ Q ∧ P ∈ ABP ∧ P ∈ CDP ∧ Q ∈ ABP ∧ Q ∈ CDP

-- The main theorem statement
theorem angle_OQP_90 (h_cyclic : isCyclicQuadrilateral A B C D)
  (h_intersect_AC_BD : intersectAt (Line.mk A C) (Line.mk B D) P)
  (h_circumcircle_intersect : circumcirclesIntersect (Circumcircle.mk A B P) (Circumcircle.mk C D P) P Q)
  (h_distinct : O ≠ P ∧ O ≠ Q ∧ P ≠ Q) :
  angle O Q P = 90° :=
sorry

end angle_OQP_90_l634_634384


namespace recast_cylinder_to_sphere_radius_l634_634218

theorem recast_cylinder_to_sphere_radius :
  ∀ (r : ℝ), (π * 2^2 * 9) = (4/3 * π * r^3) → r = 3 :=
by
  intros r h
  have cylinder_volume : 36 * π = π * 2^2 * 9 := by norm_num
  have volume_eq : 36 * π = 4 / 3 * π * r^3 := by linarith
  sorry

end recast_cylinder_to_sphere_radius_l634_634218


namespace arithmetic_sequence_sum_l634_634279

/-- Let {a_n} be an arithmetic sequence and S_n the sum of its first n terms.
   Given a_1 - a_5 - a_10 - a_15 + a_19 = 2, prove that S_19 = -38. --/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h2 : a 1 - a 5 - a 10 - a 15 + a 19 = 2) :
  S 19 = -38 := 
sorry

end arithmetic_sequence_sum_l634_634279


namespace fraction_sum_eq_l634_634317

theorem fraction_sum_eq
  (a b : ℚ) 
  (h1 : a = 20 / 24)
  (h2 : b = 20 / 25) :
  a + b = 49 / 30 := 
by {
  rw [h1, h2],
  -- Simplifying the fractions
  have ha : a = 5 / 6, sorry,
  have hb : b = 4 / 5, sorry,
  -- Adding the fractions with common denominator
  have sum_eq : 5 / 6 + 4 / 5 = 49 / 30, sorry,
  exact sum_eq
}

end fraction_sum_eq_l634_634317


namespace common_tangents_intersect_on_line_l634_634621

theorem common_tangents_intersect_on_line (O₁ O₂ : Point) (r₁ r₂ : ℝ) 
  (h₁ : r₁ > 0) (h₂ : r₂ > 0) : 
  ∃ M : Point, (M lies on the line through O₁ and O₂) ∧ 
    (M is the intersection point of the common external tangents) ∧ 
    (M is also the intersection point of the common internal tangents) :=
sorry

end common_tangents_intersect_on_line_l634_634621


namespace oil_bill_for_Jan_l634_634921

def F_to_J (F J : ℝ) := 3 * J = 2 * F
def F_to_M (F M : ℝ) := 4 * M = 5 * F
def F_new (F F_new : ℝ) := F_new = 1.15 * F
def M_new (M M_new : ℝ) := M_new = 1.10 * M
def F_plus_20_to_J (F J : ℝ) := 5 * J = 3 * (F + 20)
def F_plus_20_to_M (F M : ℝ) := 2 * M = 3 * (F + 20)

theorem oil_bill_for_Jan (J F M F_new M_new : ℝ) 
  (h1 : F_to_J F J)
  (h2 : F_to_M F M)
  (h3 : F_new F F_new)
  (h4 : M_new M M_new)
  (h5 : F_plus_20_to_J F J)
  (h6 : F_plus_20_to_M F M) : 
  J = 120 :=
by
  sorry

end oil_bill_for_Jan_l634_634921


namespace regular_polygon_sides_l634_634670

theorem regular_polygon_sides (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C]
  (angle_A angle_B angle_C : ℝ)
  (is_circle_inscribed_triangle : angle_B = 3 * angle_A ∧ angle_C = 3 * angle_A ∧ angle_B + angle_C + angle_A = 180)
  (n : ℕ)
  (is_regular_polygon : B = C ∧ angle_B = 3 * angle_A ∧ angle_C = 3 * angle_A) :
  n = 9 := sorry

end regular_polygon_sides_l634_634670


namespace twin_primes_up_to_100_count_l634_634880

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ k : ℕ, (2 ≤ k ∧ k < n) → n % k ≠ 0

def is_twin_prime_pair (p1 p2 : ℕ) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ (p2 = p1 + 2 ∨ p1 = p2 + 2)

def twin_prime_pairs_up_to_100 : Finset (ℕ × ℕ) :=
  (Finset.Icc 2 100).filter (λ p => is_twin_prime_pair p.1 p.2)

theorem twin_primes_up_to_100_count : twin_prime_pairs_up_to_100.card = 8 := by
  sorry

end twin_primes_up_to_100_count_l634_634880


namespace ratio_5_to_1_to_9_l634_634684

def has_ratio (a b : ℕ) (r : ℕ) : Prop := a * 1 = b * r

theorem ratio_5_to_1_to_9 : ∃ x : ℕ, has_ratio x 9 5 :=
by {
  use 45,
  unfold has_ratio,
  exact eq.refl (45 * 1),
}

end ratio_5_to_1_to_9_l634_634684


namespace tangent_line_eq_l634_634639

noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp x

noncomputable def f' (x : ℝ) : ℝ := (x : ℝ) * Real.exp x

theorem tangent_line_eq (x : ℝ) (h : x = 0) : 
  ∃ (c : ℝ), (1 : ℝ) = 1 ∧ f x = c ∧ f' x = 0 ∧ (∀ y, y = c) :=
by
  sorry

end tangent_line_eq_l634_634639


namespace cos_30_2α_l634_634813

variable (α : ℝ)
noncomputable theory

def cos_75_deg := Real.cos (75 * Real.pi / 180)
def sin_75_deg := Real.sin (75 * Real.pi / 180)

-- Defining the matrix determinant
def matrix_det : ℝ := cos_75_deg * Real.cos α + sin_75_deg * Real.sin α

-- Given condition in Lean
axiom determinant_condition : matrix_det = 1 / 3

-- The theorem to be proved
theorem cos_30_2α (α : ℝ) (h : matrix_det = 1 / 3) : Real.cos (30 * Real.pi / 180 + 2 * α) = 7 / 9 := 
  sorry

end cos_30_2α_l634_634813


namespace find_c_l634_634518

variable (a b tanC : ℝ)
variable (C c : ℝ)

namespace TriangleProblem

def given_conditions (a b tanC C : ℝ) :=
  a = 3 ∧ b = sqrt 7 ∧ tanC = sqrt 3 / 2 ∧ C = arctan tanC

theorem find_c : given_conditions 3 (sqrt 7) (sqrt 3 / 2) (arctan (sqrt 3 / 2)) →
  c = 2 := by
  sorry

end TriangleProblem

end find_c_l634_634518


namespace area_of_right_triangle_from_roots_l634_634894

theorem area_of_right_triangle_from_roots :
  ∀ (a b : ℝ), (a^2 - 7*a + 12 = 0) → (b^2 - 7*b + 12 = 0) →
  (∃ (area : ℝ), (area = 6) ∨ (area = (3 * real.sqrt 7) / 2)) :=
by
  intros a b ha hb
  sorry

end area_of_right_triangle_from_roots_l634_634894


namespace team_selection_ways_l634_634608

theorem team_selection_ways :
  let boys := 10
  let girls := 12
  let team_size_boys := 4
  let team_size_girls := 4
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose boys team_size_boys * choose girls team_size_girls = 103950 :=
by
  let boys := 10
  let girls := 12
  let team_size_boys := 4
  let team_size_girls := 4
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  sorry

end team_selection_ways_l634_634608


namespace part1_line_equation_part2_line_equation_l634_634725

def point (α : Type _) := prod α α

variables {α : Type _} [LinearOrderedField α]

-- Conditions in part (1)
def line_l_through_P1_and_P2 (l : α → α) : Prop := l 1 = 2

def line_l_intersects_positive_x (l : α → α) (A : point α) : Prop := A.2 = 0 ∧ 0 < A.1

def line_l_intersects_positive_y (l : α → α) (B : point α) : Prop := B.1 = 0 ∧ 0 < B.2

def vector_AP_eq_3vector_PB (A B P : point α) : Prop :=
(A.1 - P.1, A.2 - P.2) = (3 * (P.1 - B.1), 3 * (P.2 - B.2))

-- Conditions in part (2)
def area_of_triangle_AOB_eq_4 (A B : point α) : Prop :=
1/2 * abs (A.1 * B.2 - A.2 * B.1) = 4

theorem part1_line_equation (l : α → α) (P A B : point α) 
  (h1 : line_l_through_P1_and_P2 l)
  (h2 : line_l_intersects_positive_x l A)
  (h3 : line_l_intersects_positive_y l B)
  (h4 : vector_AP_eq_3vector_PB A B P) :
  ∃ a b c, a • P.1 + b • P.2 + c = (0 : α) ∧ a = 2 ∧ b = 3 ∧ c = -8 := sorry

theorem part2_line_equation (l : α → α) (A B : point α) 
  (h1 : line_l_through_P1_and_P2 l)
  (h2 : line_l_intersects_positive_x l A)
  (h3 : line_l_intersects_positive_y l B)
  (h4 : area_of_triangle_AOB_eq_4 A B) :
  ∃ a b c, a • (1 : α) + b • (2 : α) + c = 0 ∧ a = 2 ∧ b = 1 ∧ c = -4 := sorry

end part1_line_equation_part2_line_equation_l634_634725


namespace mark_speed_l634_634212

theorem mark_speed (distance : ℕ) (time : ℕ) (h1 : distance = 24) (h2 : time = 4) : distance / time = 6 :=
by
  rw [h1, h2]
  norm_num

end mark_speed_l634_634212


namespace mary_age_proof_l634_634534

theorem mary_age_proof (suzy_age_now : ℕ) (H1 : suzy_age_now = 20) (H2 : ∀ (years : ℕ), years = 4 → (suzy_age_now + years) = 2 * (mary_age + years)) : mary_age = 8 :=
by
  sorry

end mary_age_proof_l634_634534


namespace find_number_l634_634879

theorem find_number (x : ℝ) (h : 0.35 * x = 0.50 * x - 24) : x = 160 :=
by
  sorry

end find_number_l634_634879


namespace range_of_b_l634_634838

open Real

noncomputable def f (x b : ℝ) := exp x * (x - b)
noncomputable def f'' (x b : ℝ) := exp x * (x - b + 2)

theorem range_of_b (b : ℝ) :
  (∃ x ∈ Icc (1/2) 2, f(x, b) + x * f''(x, b) > 0) → b < 8/3 :=
by
  sorry

end range_of_b_l634_634838


namespace find_a_value_l634_634461

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a x

theorem find_a_value (h₁ : 0 < a) (h₂ : a < 1) (h₃ : ∀ x ∈ set.Icc 2 8, f a x = log a x) :
  ∃ a, 0 < a ∧ a < 1 ∧
  (∀ x ∈ set.Icc 2 8, f a x = log a x) ∧
  (f a 2 - f a 8 = 2) → a = 0.5 :=
sorry

end find_a_value_l634_634461


namespace trader_loss_percentage_l634_634321
noncomputable def calculateLossPercentage (sp1 sp2 : ℝ) :=
  let cp1 := sp1 / 1.12
  let cp2 := sp2 / 0.88
  let tcp := cp1 + cp2
  let tsp := sp1 + sp2
  let profit_or_loss := tsp - tcp
  (profit_or_loss / tcp) * 100

theorem trader_loss_percentage (sp : ℝ) (h : sp = 325475) :
  calculateLossPercentage sp sp ≈ -1.438 := by
  sorry

end trader_loss_percentage_l634_634321


namespace consecutive_integers_equation_l634_634164

theorem consecutive_integers_equation
  (X Y : ℕ)
  (h_consecutive : Y = X + 1)
  (h_equation : 2 * X^2 + 4 * X + 5 * Y + 3 = (X + Y)^2 + 9 * (X + Y) + 4) :
  X + Y = 15 := by
  sorry

end consecutive_integers_equation_l634_634164


namespace mary_age_l634_634531

theorem mary_age (M : ℕ) (h1 : ∀ t : ℕ, t = 4 → 24 = 2 * (M + t)) (h2 : 20 = 20) : M = 8 :=
by {
  have t_eq_4 := h1 4 rfl,
  norm_num at t_eq_4,
  linarith,
}

end mary_age_l634_634531


namespace problem1_problem2_l634_634070

-- Definitions for the given problem
def point (x y : ℝ) := (x, y)
def line (a b c : ℝ) := (x y : ℝ) → a * x + b * y + c = 0

-- Given conditions
def M := point 3 0
def l₁ := line 2 (-1) (-2)
def l₂ := line 1 1 3

-- Problem statement 1
def line_intersect_bisected (P Q : ℝ × ℝ) (l : line) :=
  line 8 (-1) (-24)

-- Problem statement 2
def reflection_of_line (l₃ : line) :=
  line 1 (-2) (-5)

-- Prove the problems
theorem problem1 : ∃ l, l = line 8 (-1) (-24) ∧ 
  (∃ P Q, P ≠ Q ∧ P ∈ l₁ ∧ Q ∈ l₂ ∧ 
  let mp := (P.1 + Q.1) / 2 in 
  let mq := (P.2 + Q.2) / 2 in 
  point mp mq = M) :=
sorry

theorem problem2 : ∃ l₃, l₃ = line 1 (-2) (-5) ∧ 
  let intersection := point (-(1/3)) (-(8/3)) in 
  (∃ m, (m * intersection.1 + (m - 1) * intersection.2 = -8 / 3 ∧ 1 - m = 2 * m - 1)) :=
sorry

end problem1_problem2_l634_634070


namespace area_of_circle_correct_l634_634617

noncomputable def area_of_circle (A B : (ℝ × ℝ)) (tangent_meeting_point_x_axis : (ℝ × ℝ)) : ℝ :=
  let D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  let slope_AB := (B.2 - A.2) / (B.1 - A.1) in
  let slope_CD := -1 / slope_AB in
  let x_intersect := tangent_meeting_point_x_axis.1 in
  let C := (x_intersect, 0) in
  let AC := real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) in
  let AD := real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) in
  let DC := real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) in
  let OA := real.sqrt (AC * AD / DC) in
  real.pi * OA^2

theorem area_of_circle_correct :
  let A := (6, 13) in
  let B := (12, 11) in
  ∃ x_intersect, (area_of_circle A B (x_intersect, 0)) = 85 * real.pi / 8 :=
sorry

end area_of_circle_correct_l634_634617


namespace all_children_receive_candy_l634_634394

-- Define f(x) function
def f (x n : ℕ) : ℕ := ((x * (x + 1)) / 2) % n

-- Define the problem statement: prove that all children receive at least one candy if n is a power of 2.
theorem all_children_receive_candy (n : ℕ) (h : ∃ m, n = 2^m) : 
    ∀ i : ℕ, i < n → ∃ x : ℕ, i = f x n := 
sorry

end all_children_receive_candy_l634_634394


namespace words_with_at_least_one_consonant_l634_634862

open Finset

structure FiveLetterWordsCondition where
  length : ℕ
  letters : Finset ℝ
  consonants : Finset ℝ
  vowels : Finset ℝ

def fiveLetterWordsCondition : FiveLetterWordsCondition :=
{ length := 5,
  letters := {0, 1, 2, 3, 4},  -- Just an example representation of A, B, C, D, E by integers
  consonants := {1, 2, 3},     -- Representing B, C, D
  vowels := {0, 4}             -- Representing A, E
}

theorem words_with_at_least_one_consonant : ∃ n, n = 3093 :=
by
  sorry

end words_with_at_least_one_consonant_l634_634862


namespace mono_increasing_interval_range_on_interval_l634_634858

noncomputable def a (x : ℝ) : ℝ × ℝ := (sqrt 3 * cos x, 0)
noncomputable def b (x : ℝ) : ℝ × ℝ := (0, sin x)
noncomputable def f (x : ℝ) : ℝ := 
  let ab2 := (sqrt 3 * cos x)^2 + (sin x)^2 -- |a|^2 + |b|^2 since a·b = 0
  ab2 + sqrt 3 * sin (2 * x)

theorem mono_increasing_interval : ∃ k : ℤ, ∀ x : ℝ, k * π - π / 3 ≤ x ∧ x ≤ k * π + π / 6 →
  2 * sin (2 * x + π / 6) + 2 = f x := 
sorry

theorem range_on_interval : ∀ x ∈ set.Ioo (-π / 4) (π / 4), -- interval (-π/4, π/4)
  2 - sqrt 3 < f x ∧ f x ≤ 4 :=
sorry

end mono_increasing_interval_range_on_interval_l634_634858


namespace magnitude_of_angle_B_l634_634555

variable (A B C a b c : ℝ)

-- Definitions for the sides of the triangle opposite to angles A, B, and C
def sides_opposite_to_angles : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

-- Given Vectors
def vector_m := (a + b, Real.sin C)
def vector_n := (Real.sqrt 3 * a + c, Real.sin B - Real.sin A)

-- Vectors are parallel
def vectors_parallel : Prop :=
  (a + b) * (Real.sin B - Real.sin A) = Real.sin C * (Real.sqrt 3 * a + c)

-- The hypothesis
axiom sides_opposite_to_angles_ax : sides_opposite_to_angles a b c
axiom vectors_parallel_ax : vectors_parallel a b c B

-- The theorem to be proved
theorem magnitude_of_angle_B : B = (5 * Real.pi) / 6 :=
  by
    sorry

end magnitude_of_angle_B_l634_634555


namespace house_trailer_payment_difference_l634_634176

-- Define the costs and periods
def cost_house : ℕ := 480000
def cost_trailer : ℕ := 120000
def loan_period_years : ℕ := 20
def months_per_year : ℕ := 12

-- Calculate total months
def total_months : ℕ := loan_period_years * months_per_year

-- Calculate monthly payments
def monthly_payment_house : ℕ := cost_house / total_months
def monthly_payment_trailer : ℕ := cost_trailer / total_months

-- Theorem stating the difference in monthly payments
theorem house_trailer_payment_difference :
  monthly_payment_house - monthly_payment_trailer = 1500 := by sorry

end house_trailer_payment_difference_l634_634176


namespace no_positive_integer_solutions_l634_634782

theorem no_positive_integer_solutions :
  ∀ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 → 3 * x^2 ≠ 5^y + 2^z - 1 :=
by
  intros x y z h,
  sorry

end no_positive_integer_solutions_l634_634782


namespace evaluate_expression_l634_634399

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_expression : 
  let p := ∏ i in Finset.range 12, (3 - w ^ (i + 1))
  p = 2391483 := 
by sorry

end evaluate_expression_l634_634399


namespace line_intersects_ellipse_l634_634462

theorem line_intersects_ellipse (m : ℝ) (k : ℝ) :
  (∀ k : ℝ, ∃ x y : ℝ, 2 * k * x - y + 1 = 0 ∧ (x^2) / 9 + (y^2) / m = 1) ↔ 
  m ∈ set.Ico 1 9 ∪ set.Ioi 9 :=
by
  sorry

end line_intersects_ellipse_l634_634462


namespace area_of_region_enclosed_by_even_circles_l634_634753

theorem area_of_region_enclosed_by_even_circles :
  (∑ k in finset.range_univ (λ n => ⟨n, by sorry⟩), if k.1 % 2 = 0 then (-1)^(k.1+1) * (π / k.1.factorial) else 0) = π / real.exp 1 :=
sorry

end area_of_region_enclosed_by_even_circles_l634_634753


namespace least_divisor_for_perfect_square_l634_634677

theorem least_divisor_for_perfect_square : 
  ∃ d : ℕ, (∀ n : ℕ, n > 0 → 16800 / d = n * n) ∧ d = 21 := 
sorry

end least_divisor_for_perfect_square_l634_634677


namespace ordered_pairs_count_is_894_l634_634869

noncomputable def count_ordered_pairs : ℕ :=
  let pairs := { p : ℝ × ℤ // (p.1 > 0) ∧ (3 ≤ p.2 ∧ p.2 ≤ 300) ∧ ((Real.log p.1 / Real.log p.2).pow 2023 = Real.log (p.1 ^ 2023) / Real.log p.2) } in
  Set.card pairs.to_set

theorem ordered_pairs_count_is_894 : count_ordered_pairs = 894 :=
  sorry

end ordered_pairs_count_is_894_l634_634869


namespace change_is_24_l634_634613

-- Define the prices and quantities
def price_basketball_card : ℕ := 3
def price_baseball_card : ℕ := 4
def num_basketball_cards : ℕ := 2
def num_baseball_cards : ℕ := 5
def money_paid : ℕ := 50

-- Define the total cost
def total_cost : ℕ := (num_basketball_cards * price_basketball_card) + (num_baseball_cards * price_baseball_card)

-- Define the change received
def change_received : ℕ := money_paid - total_cost

-- Prove that the change received is $24
theorem change_is_24 : change_received = 24 := by
  -- the proof will go here
  sorry

end change_is_24_l634_634613


namespace good_point_iff_diagonal_bisect_l634_634351

-- Definition of a convex quadrilateral and necessary structures
structure ConvexQuadrilateral (V : Type*) [AddGroup V] [Module ℝ V] :=
(A B C D : V)
(convex : Convex ℝ ({A, B, C, D} : Set V))

-- Definition of a good point within a convex quadrilateral
def good_point {V : Type*} [AddGroup V] [Module ℝ V] (Q : ConvexQuadrilateral V) (O : V) :=
(S : V) : S ∈ ({Q.A, Q.B, Q.C, Q.D} : set V) → ∃ (S1 S2 : V) (Area : ℝ),
   (S1 = Q.A ∧ S2 = Q.C ∧ Parallelogram O S1 S2 O) ∧
   ∃ (S3 S4 : V), (S3 = Q.B ∧ S4 = Q.D ∧ AreaTriangle O S3 S4 = Area)

-- Proposition A: Existence of a good point
def Proposition_A {V : Type*} [AddGroup V] [Module ℝ V] (Q : ConvexQuadrilateral V) :=
∃ (O : V), good_point Q O

-- Proposition B: One diagonal bisects the other
def Proposition_B {V : Type*} [AddGroup V] [Module ℝ V] (Q : ConvexQuadrilateral V) :=
∃ (E F : V), (E = midpoint ℝ Q.A Q.C ∧ F = midpoint ℝ Q.B Q.D) ∧ E = F

-- Stating the theorem that Proposition A is necessary and sufficient for Proposition B
theorem good_point_iff_diagonal_bisect {V : Type*} [AddGroup V] [Module ℝ V] :
  ∀ (Q : ConvexQuadrilateral V), Proposition_A Q ↔ Proposition_B Q :=
by
  sorry

end good_point_iff_diagonal_bisect_l634_634351


namespace integer_solution_system_eq_det_l634_634589

theorem integer_solution_system_eq_det (a b c d : ℤ) 
  (h : ∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) : 
  a * d - b * c = 1 ∨ a * d - b * c = -1 :=
by
  sorry

end integer_solution_system_eq_det_l634_634589


namespace extreme_values_monotonic_interval_l634_634835

noncomputable def f (x : ℝ) := 2 * x^3 - 3 * x^2 - 12 * x + 3

theorem extreme_values :
  (f (-1) = 10) ∧ (f (2) = -17) ∧ 
  (∀ x : ℝ, (f'(x) = 6 * x^2 - 6 * x - 12) →
     (f'(-1) = 0) ∧ (f'(2) = 0)) := 
by 
  sorry

theorem monotonic_interval (m : ℝ) (h : ∀ x ∈ Icc m (m+4), f'(x) > 0 ∨ f'(x) < 0) :
    m ≤ -5 ∨ m ≥ 2 :=
by 
  sorry

end extreme_values_monotonic_interval_l634_634835


namespace circles_tangency_l634_634371

theorem circles_tangency
    (O O1 O2 : Point)
    (S T A B P Q O' : Point)
    (h1 : Circle O)
    (h2 : Circle O1)
    (h3 : Circle O2)
    (tangent_O1_S : tangent h1 h2 S)
    (tangent_O2_T : tangent h1 h3 T)
    (intersect_AB : Intersect h2 h3 A B)
    (extension_AB_P : on_line_extension A B P)
    (intersect_PS_Q : Intersects (Line_segment P S) h2 Q)
    (intersect_O1Q_OP_O' : Intersects (Line_segment O1 Q) (Line_segment O P) O')
    (circle_O' : Circle O' O'Q) :
  tangent circle_O' h3 :=
begin
  sorry
end

end circles_tangency_l634_634371


namespace brianna_marbles_lost_l634_634372

theorem brianna_marbles_lost
  (total_marbles : ℕ)
  (remaining_marbles : ℕ)
  (L : ℕ)
  (gave_away : ℕ)
  (dog_ate : ℚ)
  (h1 : total_marbles = 24)
  (h2 : remaining_marbles = 10)
  (h3 : gave_away = 2 * L)
  (h4 : dog_ate = L / 2)
  (h5 : total_marbles - remaining_marbles = L + gave_away + dog_ate) : L = 4 := 
by
  sorry

end brianna_marbles_lost_l634_634372


namespace monotone_intervals_and_extrema_l634_634837

def f (x : ℝ) : ℝ := sqrt 2 * (Real.cos (2 * x - pi / 4))

theorem monotone_intervals_and_extrema :
  (∀ x ∈ Icc (-pi / 8) (pi / 8), Monotone (λ x, f x)) ∧
  (∀ x ∈ Icc (pi / 8) (pi / 2), Antitone (λ x, f x)) ∧
  (∀ x ∈ Icc (-pi / 8) (pi / 2), ∃ y ∈ Icc (-pi / 8) (pi / 2), f y = -1) ∧
  (∀ x ∈ Icc (-pi / 8) (pi / 2), ∃ y ∈ Icc (-pi / 8) (pi / 2), f y = sqrt 2) :=
sorry

end monotone_intervals_and_extrema_l634_634837


namespace remainder_zero_l634_634123

noncomputable def poly1 : ℚ[X] := X^6
noncomputable def poly2 : ℚ[X] := X - (1 / 3 : ℚ)

-- remainder when x^6 is divided by x - (1/3)
noncomputable def r1 : ℚ := polynomial.eval (1 / 3) poly1

-- quotient q1 when x^6 is divided by x - (1/3)
noncomputable def q1 : ℚ[X] := polynomial.div poly1 poly2

-- remainder when q1 is divided by x - (1/3)
noncomputable def r2 : ℚ := polynomial.eval (1 / 3) q1

theorem remainder_zero : r2 = 0 :=
sorry

end remainder_zero_l634_634123


namespace necessary_condition_l634_634823

noncomputable def conditions (α β : Plane) (l a b : Line) : Prop :=
  ∃ l, l ∈ α ∧ l ∈ β ∧ ∃ a, a ∈ α ∧ ∃ b, b ∈ β ∧ perp b l

theorem necessary_condition (α β : Plane) (l a b : Line) :
  conditions α β l a b → (α ⊥ β ↔ a ⊥ b) :=
begin
  sorry
end

end necessary_condition_l634_634823


namespace shaded_percentage_six_by_six_grid_l634_634312

theorem shaded_percentage_six_by_six_grid (total_squares shaded_squares : ℕ)
    (h_total : total_squares = 36) (h_shaded : shaded_squares = 21) : 
    (shaded_squares.to_rat / total_squares.to_rat) * 100 = 58.33 := 
by
  sorry

end shaded_percentage_six_by_six_grid_l634_634312


namespace identify_incorrect_value_in_table_l634_634157

def first_differences (lst : List ℕ) : List ℕ :=
  List.zipWith (-) (List.drop 1 lst) lst

def second_differences (lst : List ℕ) : List ℕ :=
  first_differences (first_differences lst)

def incorrect_value (values : List ℕ) (v : ℕ) : Prop :=
  second_differences values ≠ List.repeat 2 (List.length values - 2)

def initial_values : List ℕ := [6300, 6481, 6664, 6851, 7040, 7231, 7424, 7619, 7816]

theorem identify_incorrect_value_in_table :
  incorrect_value initial_values 6851 :=
by
  unfold incorrect_value
  unfold second_differences
  unfold first_differences
  -- The remaining proof steps would follow the discrepancy in the second differences.
  sorry

end identify_incorrect_value_in_table_l634_634157


namespace find_tan_C_find_b_given_area_l634_634141

variables {A B C : ℝ}
variables {a b c : ℝ}
variables [fact (A = π / 4)] [fact (b^2 - a^2 = 1/2 * c^2)]

/- Proof statement for (1) -/
theorem find_tan_C : 
  ∀ {A B C a b c : ℝ}, 
    A = π / 4 → 
    b^2 - a^2 = 1/2 * c^2 → 
    tan C = 2 := 
sorry

/- Proof statement for (2) -/
theorem find_b_given_area : 
  ∀ {A B C a b c : ℝ}, 
    A = π / 4 → 
    (b^2 - a^2 = 1/2 * c^2) → 
    (1/2 * a * b * sin C = 3) → 
    b = 3 := 
sorry

end find_tan_C_find_b_given_area_l634_634141


namespace goldie_hours_worked_l634_634105

def hourly_rate : ℕ := 5
def hours_last_week : ℕ := 20
def total_earnings : ℕ := 250
def earnings_last_week : ℕ := hours_last_week * hourly_rate := by
  sorry
def earnings_this_week : ℕ := total_earnings - earnings_last_week := by
  sorry
def hours_this_week : ℕ := earnings_this_week / hourly_rate := by
  sorry

theorem goldie_hours_worked : 
  earnings_last_week = hours_last_week * hourly_rate ∧
  earnings_this_week = total_earnings - earnings_last_week ∧
  hours_this_week = earnings_this_week / hourly_rate → 
  hours_this_week = 30 := 
by
  sorry

end goldie_hours_worked_l634_634105


namespace parallel_condition_l634_634950

-- Define the lines l and m, and the plane α
variables {l m : Type} [Line l] [Line m]
variables {α : Type} [Plane α]

-- Define conditions
axiom l_not_subset_alpha : ¬(l ⊆ α)
axiom m_subset_alpha : m ⊆ α

-- Define the parallelism properties
def parallel (x y : Type) [Line x] [Line y] : Prop := sorry
def parallel_plane (x : Type) [Line x] (y : Type) [Plane y] : Prop := sorry

-- THEOREM STATEMENT
theorem parallel_condition :
  parallel_plane l α → (parallel l m ∧ ¬(parallel m l) ∧ parallel_plane l α) :=
begin
  sorry
end

end parallel_condition_l634_634950


namespace ticket_ratio_proof_l634_634631

-- Define the initial number of tickets Tate has.
def initial_tate_tickets : ℕ := 32

-- Define the additional tickets Tate buys.
def additional_tickets : ℕ := 2

-- Define the total tickets they have together.
def combined_tickets : ℕ := 51

-- Calculate Tate's total number of tickets after buying more tickets.
def total_tate_tickets := initial_tate_tickets + additional_tickets

-- Define the number of tickets Peyton has.
def peyton_tickets := combined_tickets - total_tate_tickets

-- Define the ratio of Peyton's tickets to Tate's tickets.
def tickets_ratio := peyton_tickets / total_tate_tickets

theorem ticket_ratio_proof : tickets_ratio = 1 / 2 :=
by
  unfold tickets_ratio peyton_tickets total_tate_tickets initial_tate_tickets additional_tickets
  norm_num
  sorry

end ticket_ratio_proof_l634_634631


namespace solve_for_a_l634_634874

noncomputable def integral_expression (a : ℝ) : ℝ :=
∫ x in 1..a, (2 * x - 1 / x)

noncomputable def evaluated_integral (a : ℝ) : ℝ :=
  (a^2 - log a) - (1 - log 1)

theorem solve_for_a (a : ℝ) (h1 : 1 < a) (h2 : integral_expression a = 3 - log 2) : a = 2 :=
begin
  unfold integral_expression at h2,
  unfold evaluated_integral at h2,
  sorry
end

end solve_for_a_l634_634874


namespace number_of_common_tangents_l634_634646

-- Define the first circle
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 6 * y - 48 = 0

-- Define the second circle
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 8 * y - 44 = 0

-- The theorem proving that the number of common tangents is 2
theorem number_of_common_tangents : 
  ∃ C1 C2 : (ℝ × ℝ) → Prop, 
    (∀ x y : ℝ, C1 (x, y) ↔ circle1 x y) ∧ 
    (∀ x y : ℝ, C2 (x, y) ↔ circle2 x y) ∧
    (let r1 := 8 in let r2 := 8 in let d := real.sqrt 74 in |r1 - r2| < d ∧ d < r1 + r2) →
    2 := by sorry

end number_of_common_tangents_l634_634646


namespace library_total_books_l634_634344

theorem library_total_books
    (T : ℕ)
    (h1 : 0.40 * T = 104) :
  T = 260 := 
by 
suffices : T = 104 / 0.40 by sorry
sorry

end library_total_books_l634_634344


namespace collinear_unit_vectors_l634_634853

namespace MyVectorProof

-- Define the initial vector a
def vec_a : ℝ × ℝ × ℝ := (-3, 1, real.sqrt 6)

-- Calculate the magnitude (norm) of vector a
noncomputable def vec_a_magnitude : ℝ :=
  real.sqrt ((-3)^2 + 1^2 + (real.sqrt 6)^2)

-- Define the unit vector that is collinear with vec_a (both possibilities)
def unit_vec_1 : ℝ × ℝ × ℝ := (-3 / vec_a_magnitude, 1 / vec_a_magnitude, real.sqrt 6 / vec_a_magnitude)
def unit_vec_2 : ℝ × ℝ × ℝ := (3 / vec_a_magnitude, -1 / vec_a_magnitude, -real.sqrt 6 / vec_a_magnitude)

-- Statement to be proved
theorem collinear_unit_vectors :
  vec_a_magnitude = 4 →
  (unit_vec_1 = (-3 / 4, 1 / 4, real.sqrt 6 / 4)) ∧ (unit_vec_2 = (3 / 4, -1 / 4, -real.sqrt 6 / 4)) :=
by
  intro h
  sorry

end MyVectorProof

end collinear_unit_vectors_l634_634853


namespace lottery_probability_l634_634262

theorem lottery_probability :
  let MegaBallProbability := (1 : ℝ) / 30
  let WinnerSequenceProbability := (1 : ℝ) / 50 * (1 : ℝ) / 49 * (1 : ℝ) / 48 * (1 : ℝ) / 47 * (1 : ℝ) / 46
  MegaBallProbability * WinnerSequenceProbability = (1 : ℝ) / 841500000 :=
by
  let MegaBallProbability := (1 : ℝ) / 30
  let WinnerSequenceProbability := (1 : ℝ) / 50 * (1 : ℝ) / 49 * (1 : ℝ) / 48 * (1 : ℝ) / 47 * (1 : ℝ) / 46
  have h1 : MegaBallProbability = (1 : ℝ) / 30 := rfl
  have h2 : WinnerSequenceProbability = (1 : ℝ) / 50 * (1 : ℝ) / 49 * (1 : ℝ) / 48 * (1 : ℝ) / 47 * (1 : ℝ) / 46 := rfl
  rw [h1, h2]
  sorry

end lottery_probability_l634_634262


namespace sum_2017_terms_eq_5042_l634_634370

-- Definitions for the conditions in the problem
def equal_sum_sequence (a : ℕ → ℕ) (c : ℕ) :=
  ∀ n : ℕ, a n + a (n + 1) = c

-- The given sequence with the specified initial conditions and common sum
def a {a : ℕ → ℕ} (h : equal_sum_sequence a 5) (ha1 : a 1 = 2) : ℕ := sorry

-- The sum of the first 2017 terms of the sequence
def sum_first_n_terms {a : ℕ → ℕ} (S : ℕ) (h1 : equal_sum_sequence a 5) (ha1 : a 1 = 2) (hn : S = 2017) : Prop :=
  (∑ i in finset.range S, a i) = 5042

-- The theorem statement
theorem sum_2017_terms_eq_5042 : ∀ {a : ℕ → ℕ} (h1 : equal_sum_sequence a 5) (ha1 : a 1 = 2),
  sum_first_n_terms 2017 h1 ha1 := sorry

end sum_2017_terms_eq_5042_l634_634370


namespace ellipse_properties_l634_634415

noncomputable def a_square : ℝ := 2
noncomputable def b_square : ℝ := 9 / 8
noncomputable def c_square : ℝ := a_square - b_square
noncomputable def c : ℝ := Real.sqrt c_square
noncomputable def distance_between_foci : ℝ := 2 * c
noncomputable def eccentricity : ℝ := c / Real.sqrt a_square

theorem ellipse_properties :
  (distance_between_foci = Real.sqrt 14) ∧ (eccentricity = Real.sqrt 7 / 4) := by
  sorry

end ellipse_properties_l634_634415


namespace range_f_range_a_l634_634084

def f (x : ℝ) := abs (x + 2) - abs (x - 1)

def g (a : ℝ) (s : ℝ) := (a * s^2 - 3 * s + 3) / s

theorem range_f :
  set.range f = set.Icc (-3 : ℝ) 3 := sorry

theorem range_a (a : ℝ) :
  ( ∀ (s : ℝ) (t : ℝ), s > 0 → g a s ≥ f t ) → a ≥ 3 := sorry

end range_f_range_a_l634_634084


namespace percent_of_red_candies_remain_l634_634333

open_locale classical

noncomputable def percent_red_candies_remain (n : ℕ) : ℕ :=
  let total_candies := 6 * n in
  let remaining_after_green := 5 * n in
  let remaining_after_orange := remaining_after_green - n / 2 in
  let remaining_after_purple := remaining_after_orange - (2 * n) / 3 in
  let remaining_after_half_other := remaining_after_purple - 2 * n in
  let required_remaining := (32 * total_candies) / 100 in
  if remaining_after_half_other <= required_remaining then
    (n / 2)
  else
    sorry

theorem percent_of_red_candies_remain (n : ℕ) (n > 0) :
  percent_red_candies_remain n = n / 2 := by 
    sorry

end percent_of_red_candies_remain_l634_634333


namespace product_of_roots_with_pos_real_part_l634_634510

-- Definitions based on the conditions in the problem
def roots (n : ℕ) (z : ℂ) : Set ℂ := {x | x^n = z}
def real_part_pos (x : ℂ) : Prop := x.re > 0

-- Main theorem based on the question
theorem product_of_roots_with_pos_real_part :
  (∏ x in (roots 8 (-256 : ℂ)).filter real_part_pos, x) = 16 :=
  sorry

end product_of_roots_with_pos_real_part_l634_634510


namespace max_pairs_condition_l634_634247

-- Define the problem conditions in Lean
def ordered_pairs (n : ℕ) : Type := {p : ℕ × ℕ // p.fst < n ∧ p.snd < n}

def max_good_pairs := 197

theorem max_pairs_condition (s : Finset (ℕ × ℕ)) (h : s.card = 100)
  (h_distinct : s.nodup) :
  (∀ i j, (i ≠ j ∧ 1 ≤ i ∧ i < j ∧ j ≤ 100) → (∃ p1 p2, s.to_list.nth i = some p1 ∧ s.to_list.nth j = some p2 ∧ |p1.fst * p2.snd - p2.fst * p1.snd| = 1)) →
  s.card ≤ max_good_pairs :=
sorry

end max_pairs_condition_l634_634247


namespace gg_even_of_g_even_l634_634594

-- Define what it means for a function to be even
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = f(x)

-- Theorem statement: Prove that if g is even, then g(g(x)) is even
theorem gg_even_of_g_even (g : ℝ → ℝ) (h : even_function g) : even_function (λ x, g(g(x))) :=
  sorry

end gg_even_of_g_even_l634_634594


namespace even_func_min_value_l634_634882

theorem even_func_min_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_neq_a : a ≠ 1) (h_neq_b : b ≠ 1) (h_even : ∀ x : ℝ, a^x + b^x = a^(-x) + b^(-x)) :
  ab = 1 → (∃ y : ℝ, y = (1 / a + 4 / b) ∧ y = 4) :=
by
  sorry

end even_func_min_value_l634_634882


namespace astroid_trajectory_l634_634231

theorem astroid_trajectory (r R ω1 ω2 : ℝ) (t : ℝ)
  (hR : R = 4 * r) (hω : ω2 = -3 * ω1)
  (hx : r * cos(ω1 * t) + r * cos(ω2 * t) = 4 * r * cos(t)^3)
  (hy : r * sin(ω1 * t) + r * sin(ω2 * t) = 4 * r * sin(t)^3) :
  (r^(-2/3) * (r * cos(ω1 * t) + r * cos(ω2 * t))^2/3 + r^(-2/3) * (r * sin(ω1 * t) + r * sin(ω2 * t))^2/3 = (4 * r)^(2/3)) :=
by
  sorry

end astroid_trajectory_l634_634231


namespace log2_125_eq_9y_l634_634577

theorem log2_125_eq_9y (y : ℝ) (h : Real.log 5 / Real.log 8 = y) : Real.log 125 / Real.log 2 = 9 * y :=
by
  sorry

end log2_125_eq_9y_l634_634577


namespace other_root_l634_634041

theorem other_root (m : ℝ) :
  (∃ (a : ℝ), a ^ 2 - 4 * a + m + 2 = 0 ∧ a = -1) →
  (∃ (b : ℝ), b ^ 2 - 4 * b + m + 2 = 0 ∧ b = 5) :=
by
  intro h
  cases h with a ha
  cases ha with ha1 ha2
  use 5
  sorry

end other_root_l634_634041


namespace five_digit_even_with_adjacent_23_or_32_l634_634795

def is_even (n : ℕ) : Prop := n % 2 = 0

def adjacent (n m : ℕ) (l : List ℕ) : Prop :=
  ∃ (i : ℕ), i < l.length - 1 ∧ l.nth i = some n ∧ l.nth (i + 1) = some m

noncomputable def count_adj_even_five_digit_numbers : ℕ :=
  ((List.permutations [1, 2, 3, 4, 5]).filter 
   (λ l, is_even (List.last l 0) ∧ (adjacent 2 3 l ∨ adjacent 3 2 l))).length

theorem five_digit_even_with_adjacent_23_or_32 : count_adj_even_five_digit_numbers = 18 := 
by 
  sorry

end five_digit_even_with_adjacent_23_or_32_l634_634795


namespace distance_between_stripes_l634_634357

theorem distance_between_stripes 
  (width_curbs : ℝ)
  (curb_length_between_stripes : ℝ)
  (stripe_length : ℝ)
  (area_parallelogram : ℝ)
  (d : ℝ)
  (h_curbs_parallel : width_curbs = 40)
  (h_curb_length_between_stripes : curb_length_between_stripes = 15)
  (h_stripe_length : stripe_length = 50)
  (h_area_parallelogram : area_parallelogram = curb_length_between_stripes * width_curbs) :
  area_parallelogram = stripe_length * d → d = 12 := by
  -- given conditions
  have h1 : area_parallelogram = 600 := by
    rw [h_curb_length_between_stripes, h_curbs_parallel]
    simp [h_area_parallelogram]
  have h2 : area_parallelogram = stripe_length * d := by
    assumption
  rw [h1, h_stripe_length] at h2
  sorry

end distance_between_stripes_l634_634357


namespace compare_exponents_l634_634429

noncomputable def a : ℝ := 6 ^ Real.log 5
noncomputable def b : ℝ := 7 ^ Real.log 4
noncomputable def c : ℝ := 8 ^ Real.log 3

def f (x : ℝ) : ℝ := Real.log x * Real.log (11 - x)

theorem compare_exponents : a > b ∧ b > c := by
  -- Proof placeholder
  sorry

end compare_exponents_l634_634429


namespace count_non_adjacent_chords_l634_634633

/-- Ten points are marked on the circumference of a circle. 
    However, no two adjacent points can be connected by a chord. 
    How many different chords can be drawn under this restriction? --/
theorem count_non_adjacent_chords (n : ℕ) (h : n = 10) : 
  let total_chords := (n * (n - 1)) / 2,
      adjacent_chords := n,
      non_adjacent_chords := total_chords - adjacent_chords
  in non_adjacent_chords = 35 :=
by
  sorry

end count_non_adjacent_chords_l634_634633


namespace cut_7x7_board_to_squares_l634_634361

-- Define the problem conditions
def is_integer (n : ℤ) : Prop := n.to_nat = n

def is_possible_to_cut_and_form (a b c : ℤ) : Prop :=
  a^2 + b^2 + c^2 = 49 ∧ a < b ∧ b < c ∧ is_integer a ∧ is_integer b ∧ is_integer c

-- The theorem to check if it's possible to create 3 squares of sizes 3, 4, and 5
theorem cut_7x7_board_to_squares : 
  ∃ (a b c : ℤ), is_possible_to_cut_and_form a b c :=
begin
  use [3, 4, 5],
  simp [is_possible_to_cut_and_form, is_integer],
  norm_num,
end

end cut_7x7_board_to_squares_l634_634361


namespace perimeter_pqr_30_l634_634187

theorem perimeter_pqr_30 (AB BC CA : ℕ) (hAB : AB = 13) (hBC : BC = 14) (hCA : CA = 15)
    (AQR BPR CPQ PQR : Type) (p : Type → ℕ)
    (h2 : p(AQR) = (4 / 5 : ℚ) * p(PQR))
    (h3 : p(BPR) = (4 / 5 : ℚ) * p(PQR))
    (h4 : p(CPQ) = (4 / 5 : ℚ) * p(PQR))
    (h_sum : p(AQR) + p(BPR) + p(CPQ) - p(PQR) = AB + BC + CA) :
    p(PQR) = 30 := by
  sorry

end perimeter_pqr_30_l634_634187


namespace fish_pond_estimation_l634_634666

/-- To estimate the number of fish in a pond, 40 fish were first caught and marked, then released back into the pond.
After the marked fish were completely mixed with the rest of the fish in the pond, 100 fish were caught again, and 5 of them were found to be marked. 
We need to prove that the total number of fish in the pond is 800. -/
theorem fish_pond_estimation
  (marked_released : ℕ)
  (fish_caught : ℕ)
  (marked_found : ℕ)
  (total_fish : ℕ)
  (h_marked_released : marked_released = 40)
  (h_fish_caught : fish_caught = 100)
  (h_marked_found : marked_found = 5)
  (h_total_fish : total_fish = 800) :
  fish_caught / marked_found = total_fish / marked_released :=
by
  rw [h_marked_released, h_fish_caught, h_marked_found, h_total_fish]
  sorry

end fish_pond_estimation_l634_634666


namespace det_A_is_one_l634_634411

open Matrix

variables (α β : ℝ)

def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![cos (α + π/4) * cos (β + π/4), cos (α + π/4) * sin (β + π/4), -sin (α + π/4)],
    ![-sin (β + π/4), cos (β + π/4), 0],
    ![sin (α + π/4) * cos (β + π/4), sin (α + π/4) * sin (β + π/4), cos (α + π/4)]
  ]

theorem det_A_is_one : det (A α β) = 1 :=
  sorry

end det_A_is_one_l634_634411


namespace largest_d_l634_634472

variable (a b c d : ℤ)

def condition : Prop := a + 2 = b - 1 ∧ a + 2 = c + 3 ∧ a + 2 = d - 4

theorem largest_d (h : condition a b c d) : d > a ∧ d > b ∧ d > c :=
by
  -- Assuming the condition holds, we need to prove d > a, d > b, and d > c
  sorry

end largest_d_l634_634472


namespace max_value_trig_expression_l634_634026

theorem max_value_trig_expression : ∀ x : ℝ, (3 * Real.cos x + 4 * Real.sin x) ≤ 5 := 
sorry

end max_value_trig_expression_l634_634026


namespace parabola_x_intercepts_count_l634_634109

theorem parabola_x_intercepts_count :
  ∃! x, ∃ y, x = -3 * y^2 + 2 * y + 2 ∧ y = 0 :=
by
  sorry

end parabola_x_intercepts_count_l634_634109


namespace initial_oranges_count_l634_634287

theorem initial_oranges_count
  (initial_apples : ℕ := 50)
  (apple_cost : ℝ := 0.80)
  (orange_cost : ℝ := 0.50)
  (total_earnings : ℝ := 49)
  (remaining_apples : ℕ := 10)
  (remaining_oranges : ℕ := 6)
  : initial_oranges = 40 := 
by
  sorry

end initial_oranges_count_l634_634287


namespace sequence_recurrence_l634_634629

theorem sequence_recurrence (v : ℕ → ℝ) (h_rec : ∀ n, v (n + 2) = 3 * v (n + 1) + 2 * v n) 
    (h_v3 : v 3 = 8) (h_v6 : v 6 = 245) : v 5 = 70 :=
sorry

end sequence_recurrence_l634_634629


namespace camp_children_l634_634540

noncomputable def number_of_children (C : ℝ) (B : ℝ) : Prop :=
  (0.10 * C = 0.05 * (C + B))

axiom additional_boys : ℝ := 49.99999999999997
axiom current_children : ℝ := 50

theorem camp_children : number_of_children current_children additional_boys :=
by
  unfold number_of_children
  have h : 0.10 * current_children = current_children / 10 := by ring
  rw [h]
  have h' : 0.05 * (current_children + additional_boys) = (current_children + 49.99999999999997) / 20 := by ring
  rw [h']
  sorry

end camp_children_l634_634540


namespace right_triangle_area_l634_634901

theorem right_triangle_area (a b : ℝ) (h : a^2 - 7 * a + 12 = 0 ∧ b^2 - 7 * b + 12 = 0) : 
  ∃ A : ℝ, (A = 6 ∨ A = 3 * (Real.sqrt 7 / 2)) ∧ A = 1 / 2 * a * b := 
by 
  sorry

end right_triangle_area_l634_634901


namespace proof_tan_C_and_area_l634_634932

variable {A B C : ℝ} {a b c : ℝ}

def tan_C_eq_2 (a b c : ℝ) (C : ℝ) : Prop :=
  a = sqrt 10 ∧ a^2 + b^2 - c^2 = a * b * sin C ∧ a * cos B + b * sin A = c → tan C = 2

def area_eq_6 (a b c : ℝ) (C : ℝ) : Prop :=
  a = sqrt 10 ∧ a^2 + b^2 - c^2 = a * b * sin C ∧ a * cos B + b * sin A = c →
  (1/2) * a * b * sin C = 6

theorem proof_tan_C_and_area :
  tan_C_eq_2 a b c C ∧ area_eq_6 a b c C :=
by
  sorry

end proof_tan_C_and_area_l634_634932


namespace pebbles_collected_by_day_20_l634_634220

theorem pebbles_collected_by_day_20 (n : ℕ) (h : n = 20) : 
    ∑ k in range(n+1), k = 210 :=
by 
  sorry

end pebbles_collected_by_day_20_l634_634220


namespace age_in_1930_l634_634349

/-- A person's age at the time of their death (y) was one 31st of their birth year,
and we want to prove the person's age in 1930 (x). -/
theorem age_in_1930 (x y : ℕ) (h : 31 * y + x = 1930) (hx : 0 < x) (hxy : x < y) :
  x = 39 :=
sorry

end age_in_1930_l634_634349


namespace largest_n_for_sin_cos_l634_634793

theorem largest_n_for_sin_cos (n : ℕ) (h : n = 3) :
  (∀ x ∈ Ioc (0 : ℝ) (π / 2), sin x ^ n + cos x ^ n > 1 / 2) :=
sorry

end largest_n_for_sin_cos_l634_634793


namespace max_value_trig_expression_l634_634028

theorem max_value_trig_expression : ∀ x : ℝ, (3 * Real.cos x + 4 * Real.sin x) ≤ 5 := 
sorry

end max_value_trig_expression_l634_634028


namespace math_problem_l634_634426

theorem math_problem (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 :=
sorry

end math_problem_l634_634426


namespace solution_set_l634_634463

variable {f : ℝ → ℝ}

-- Given conditions
def odd_function := ∀ x, f(-x) = -f(x)
def differentiable_function := Differentiable ℝ f
def condition_2f_plus_xf' (x : ℝ) : Prop := 2 * f x + x * (deriv f x) > x^2

-- derived definition
def F (x : ℝ) : ℝ := x^2 * f x

-- Proof statement
theorem solution_set (h_odd : odd_function f)
  (h_diff : differentiable_function f)
  (h_condition : ∀ x > 0, condition_2f_plus_xf' f x) :
  {x | (x + 2014)^2 * f (x + 2014) + 4 * f (-2) < 0} = set.Iio (-2012) := sorry

end solution_set_l634_634463


namespace Julia_total_payment_l634_634185

namespace CarRental

def daily_rate : ℝ := 30
def mileage_rate : ℝ := 0.25
def num_days : ℝ := 3
def num_miles : ℝ := 500

def daily_cost : ℝ := daily_rate * num_days
def mileage_cost : ℝ := mileage_rate * num_miles
def total_cost : ℝ := daily_cost + mileage_cost

theorem Julia_total_payment : total_cost = 215 := by
  sorry

end CarRental

end Julia_total_payment_l634_634185


namespace find_a_b_minimum_value_l634_634844

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 + b * x^2

/-- Given the function y = f(x) = ax^3 + bx^2, when x = 1, it has a maximum value of 3 -/
def condition1 (a b : ℝ) : Prop :=
  f 1 a b = 3 ∧ (3 * a + 2 * b = 0)

/-- Find the values of the real numbers a and b -/
theorem find_a_b : ∃ (a b : ℝ), condition1 a b :=
sorry

/-- Find the minimum value of the function -/
theorem minimum_value : ∀ (a b : ℝ), condition1 a b → (∃ x_min, ∀ x, f x a b ≥ f x_min a b) :=
sorry

end find_a_b_minimum_value_l634_634844


namespace range_f_l634_634388

noncomputable def f (x : ℝ) : ℝ := Real.cos (x - π / 3)

theorem range_f : set.range f = set.Icc (1 / 2) 1 :=
by
  sorry

end range_f_l634_634388


namespace solve_temperature_l634_634497

theorem solve_temperature(F : ℝ) : 
  (25 = (4 / 9) * (F - 40)) → 
  F = 96.25 ∧ (C = (5 / 9) * (F - 32) → C ≈ 35.7) :=
by
  sorry

end solve_temperature_l634_634497


namespace expected_value_of_winnings_l634_634606

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_perfect_square (n : ℕ) : Prop :=
  n = 1 ∨ n = 4

def winnings (n : ℕ) : ℤ :=
  if is_prime n then n
  else if is_perfect_square n then -n
  else 0

def expected_value : ℚ :=
  (1 / 2 : ℚ) * (2 + 3 + 5 + 7) - (1 / 4 : ℚ) * (1 + 4) + (1 / 4 : ℚ) * 0

theorem expected_value_of_winnings : expected_value = 29 / 4 := 
  sorry

end expected_value_of_winnings_l634_634606


namespace remainder_div_x_plus_2_eq_98_l634_634682

def polynomial := polynomial ℤ

def f : polynomial := polynomial.C (-20) + polynomial.X * (polynomial.C 5 + polynomial.X * (polynomial.C 12 + polynomial.X * (polynomial.C (-8) + polynomial.X)))

theorem remainder_div_x_plus_2_eq_98 : (f.eval (-2) = 98) :=
by sorry

end remainder_div_x_plus_2_eq_98_l634_634682


namespace pages_copied_l634_634933

-- Define the assumptions
def cost_per_pages (cent_per_pages: ℕ) : Prop := 
  5 * cent_per_pages = 7 * 1

def total_cents (dollars: ℕ) (cents: ℕ) : Prop :=
  cents = dollars * 100

-- The problem to prove
theorem pages_copied (dollars: ℕ) (cents: ℕ) (cent_per_pages: ℕ) : 
  cost_per_pages cent_per_pages → total_cents dollars cents → dollars = 35 → cents = 3500 → 
  3500 * (5/7 : ℚ) = 2500 :=
by
  sorry

end pages_copied_l634_634933


namespace whole_process_time_is_9_l634_634767

variable (BleachingTime : ℕ)
variable (DyeingTime : ℕ)

-- Conditions
axiom bleachingTime_is_3 : BleachingTime = 3
axiom dyeingTime_is_twice_bleachingTime : DyeingTime = 2 * BleachingTime

-- Question and Proof Problem
theorem whole_process_time_is_9 (BleachingTime : ℕ) (DyeingTime : ℕ)
  (h1 : BleachingTime = 3) (h2 : DyeingTime = 2 * BleachingTime) : 
  (BleachingTime + DyeingTime) = 9 :=
  by
  sorry

end whole_process_time_is_9_l634_634767


namespace solution_opposite_numbers_l634_634364

theorem solution_opposite_numbers (x y : ℤ) (h1 : 2 * x + 3 * y - 4 = 0) (h2 : x = -y) : x = -4 ∧ y = 4 :=
by
  sorry

end solution_opposite_numbers_l634_634364


namespace part1_l634_634927

variable (θ : ℝ)

noncomputable def a : ℝ × ℝ := (2 * Real.sin θ, 1)
noncomputable def b : ℝ × ℝ := (1, Real.sin (θ + Real.pi / 3))

theorem part1 : (a · b = 0) → Real.tan θ = -Real.sqrt 3 / 5 := by
  sorry

end part1_l634_634927


namespace gg_even_of_g_even_l634_634593

-- Define what it means for a function to be even
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = f(x)

-- Theorem statement: Prove that if g is even, then g(g(x)) is even
theorem gg_even_of_g_even (g : ℝ → ℝ) (h : even_function g) : even_function (λ x, g(g(x))) :=
  sorry

end gg_even_of_g_even_l634_634593


namespace magnitude_of_b_l634_634857

open Real

noncomputable def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

def angle_cosine (u v : ℝ × ℝ × ℝ) : ℝ :=
  (u.1 * v.1 + u.2 * v.2 + u.3 * v.3) / 
  (vector_magnitude u * vector_magnitude v)

theorem magnitude_of_b 
  (a b : ℝ × ℝ × ℝ) 
  (ha : vector_magnitude a = 1) 
  (hab : vector_magnitude (a - b) = sqrt 3 / 2) 
  (θ : angle_cosine a b = 1 / 2) :
  vector_magnitude b = 1 / 2 := 
sorry

end magnitude_of_b_l634_634857


namespace angle_sum_around_point_l634_634683

theorem angle_sum_around_point {x : ℝ} (h : 2 * x + 210 = 360) : x = 75 :=
by
  sorry

end angle_sum_around_point_l634_634683


namespace polynomial_roots_l634_634031

theorem polynomial_roots :
  ∀ x : ℝ, (4 * x^4 - 28 * x^3 + 53 * x^2 - 28 * x + 4 = 0) ↔ (x = 4 ∨ x = 2 ∨ x = 1/4 ∨ x = 1/2) := 
by
  sorry

end polynomial_roots_l634_634031


namespace inequality_f_l634_634036

variable {f : ℝ → ℝ}

-- Assuming the conditions provided
axiom diff_f : Differentiable ℝ f
axiom cond : ∀ x : ℝ, (x - 1) * (deriv f x) ≥ 0

-- The theorem to be proved
theorem inequality_f (hf : Differentiable ℝ f) (hcond : ∀ x : ℝ, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 :=
sorry

end inequality_f_l634_634036


namespace intersection_cardinality_l634_634881

noncomputable def setA : Set ℝ := {1/2, 1/3, 1/5}
noncomputable def setB : Set ℝ := {0.2, 0.4, 0.5, 0.7}

theorem intersection_cardinality : (setA ∩ setB).card = 2 := by
  sorry

end intersection_cardinality_l634_634881


namespace city_plan_exists_l634_634161

-- Definitions of vertices and their connections
structure Vertex : Type :=
(id : ℕ)

-- Define the edges in the graph
structure Edge : Type :=
(v1 v2 : Vertex)

-- Define a graph as a collection of vertices and edges
structure Graph : Type :=
(vertices : List Vertex)
(edges : List Edge)

-- Define the conditions for the graph to satisfy
def is_planar (G : Graph) : Prop := sorry -- Placeholder for the actual planar graph definition

def degree_three (v : Vertex) (G : Graph) : Prop :=
(G.edges.count (λ e => e.v1 = v ∨ e.v2 = v) = 3)

def no_edges_intersect (G : Graph) : Prop := sorry -- Placeholder for checking intersecting edges

def valid_angles (G : Graph) : Prop := sorry -- Placeholder for checking angles condition

-- Vertex list
def vertices : List Vertex :=
[⟨1⟩, ⟨2⟩, ⟨3⟩, ⟨4⟩, ⟨5⟩, ⟨6⟩]

-- Edges list representing solution
def edges : List Edge :=
[⟨⟨1⟩, ⟨2⟩⟩, ⟨⟨1⟩, ⟨3⟩⟩, ⟨⟨1⟩, ⟨6⟩⟩,
 ⟨⟨2⟩, ⟨3⟩⟩, ⟨⟨2⟩, ⟨4⟩⟩,
 ⟨⟨3⟩, ⟨4⟩⟩,
 ⟨⟨4⟩, ⟨5⟩⟩,
 ⟨⟨5⟩, ⟨6⟩⟩,
 ⟨⟨6⟩, ⟨2⟩⟩]

-- Define the graph
def graph : Graph :=
{ vertices := vertices,
  edges := edges }

-- The proof statement
theorem city_plan_exists : 
    ∃ G : Graph, 
      G.vertices.length = 6 ∧
      (∀ v : Vertex, v ∈ G.vertices → degree_three v G) ∧
      is_planar G ∧
      no_edges_intersect G ∧
      valid_angles G :=
    ⟨graph, by
      split; sorry⟩

end city_plan_exists_l634_634161


namespace evaluate_expression_l634_634397

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_expression : 
  let p := ∏ i in Finset.range 12, (3 - w ^ (i + 1))
  p = 2391483 := 
by sorry

end evaluate_expression_l634_634397


namespace complex_purely_imaginary_eq_one_l634_634502

-- Definition: A complex number z = a + bi is purely imaginary if and only if its real part is zero.
def purely_imaginary (z : ℂ) : Prop := z.re = 0

-- Statement: Given the complex number (x^2 - 1) + (x + 1)i is purely imaginary, prove that x = 1.
theorem complex_purely_imaginary_eq_one (x : ℝ) (h : purely_imaginary ((x^2 - 1) + (x + 1) * complex.I)) : x = 1 :=
by {
  sorry -- Proof is omitted as per instructions.
}

end complex_purely_imaginary_eq_one_l634_634502


namespace x_intercepts_of_parabola_l634_634113

theorem x_intercepts_of_parabola : 
  (∃ y : ℝ, x = -3 * y^2 + 2 * y + 2) → ∃ y : ℝ, y = 0 ∧ x = 2 ∧ ∀ y' ≠ 0, x ≠ -3 * y'^2 + 2 * y' + 2 :=
by
  sorry

end x_intercepts_of_parabola_l634_634113


namespace employee_Y_base_pay_l634_634430

theorem employee_Y_base_pay (P : ℝ) (h1 : 1.2 * P + P * 1.1 + P * 1.08 + P = P * 4.38)
                            (h2 : 2 * 1.5 * 1.2 * P = 3.6 * P)
                            (h3 : P * 4.38 + 100 + 3.6 * P = 1800) :
  P = 213.03 :=
by
  sorry

end employee_Y_base_pay_l634_634430


namespace c_investment_l634_634742

theorem c_investment 
  (A_investment B_investment : ℝ)
  (C_share total_profit : ℝ)
  (hA : A_investment = 8000)
  (hB : B_investment = 4000)
  (hC_share : C_share = 36000)
  (h_profit : total_profit = 252000) :
  ∃ (x : ℝ), (x / 4000) / (2 + 1 + x / 4000) = (36000 / 252000) ∧ x = 2000 :=
by
  sorry

end c_investment_l634_634742


namespace mean_of_three_l634_634264

variable (p q r : ℚ)

theorem mean_of_three (h1 : (p + q) / 2 = 13)
                      (h2 : (q + r) / 2 = 16)
                      (h3 : (r + p) / 2 = 7) :
                      (p + q + r) / 3 = 12 :=
by
  sorry

end mean_of_three_l634_634264


namespace answer_l634_634849

variable {R : Type} [LinearOrderedField R]

def p (x : R) : Prop := |x| ≥ 0

def q : Prop := ∃ x : R, x = 1 ∧ x^2 + x + 1 = 0

theorem answer (h1 : ∀ x : R, p x) (h2 : ¬q) : p 0 ∧ ¬q :=
by
  split
  . apply h1
  . exact h2

end answer_l634_634849


namespace petya_wins_probability_l634_634992

def stones_initial : ℕ := 16

def valid_moves : set ℕ := {1, 2, 3, 4}

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def petya_random_choice (move : ℕ) : Prop := move ∈ valid_moves

def computer_optimal_strategy (n : ℕ) : ℕ :=
  (n % 5)

noncomputable def probability_petya_wins : ℚ :=
  (1 / 4) ^ 4

theorem petya_wins_probability :
  probability_petya_wins = 1 / 256 := 
sorry

end petya_wins_probability_l634_634992


namespace evaluate_expression_l634_634401

noncomputable def w : ℂ := complex.exp (2 * real.pi * complex.I / 13)

theorem evaluate_expression :
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * 
  (3 - w^6) * (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * 
  (3 - w^11) * (3 - w^12) = 797161 :=
begin
  sorry
end

end evaluate_expression_l634_634401


namespace six_digit_numbers_with_condition_l634_634270

theorem six_digit_numbers_with_condition : 
  ∃ (N : finset ℕ), 
    (∀ n ∈ N, n ∈ finset.range 1000000 ∧ ∀ i j, i < j → digit n i < digit n j) ∧ 
    (∀ n ∈ N, (3 ∈ digits n) ∧ (4 ∈ digits n) ∧ (5 ∈ digits n)) ∧ 
    (∀ n ∈ N, n % 6 = 0) ∧ 
    N.card = 3 := 
sorry

end six_digit_numbers_with_condition_l634_634270


namespace middle_guards_hours_l634_634722

def total_hours := 9
def hours_first_guard := 3
def hours_last_guard := 2
def remaining_hours := total_hours - hours_first_guard - hours_last_guard
def num_middle_guards := 2

theorem middle_guards_hours : remaining_hours / num_middle_guards = 2 := by
  sorry

end middle_guards_hours_l634_634722


namespace lowest_sewage_pollution_index_max_sewage_pollution_index_range_l634_634337

-- Conditions
def f (x : ℝ) (a : ℝ) : ℝ := abs (log (x + 1) / log 25 - a) + 2 * a + 1

-- The first part of the problem
theorem lowest_sewage_pollution_index (a : ℝ) (h_a : a = 1 / 2) : 
  ∃ x, 0 ≤ x ∧ x ≤ 24 ∧ f x a = 2 :=
by
  use 4
  sorry

-- The second part of the problem
theorem max_sewage_pollution_index_range : 
  ∀ a, (0 < a ∧ a < 1) → 
  (∀ x, 0 ≤ x ∧ x ≤ 24 → f x a ≤ 3) → (0 < a ∧ a ≤ 2 / 3) :=
by
  intros a ha h
  sorry  

end lowest_sewage_pollution_index_max_sewage_pollution_index_range_l634_634337


namespace power_mod_l634_634681

theorem power_mod (a k n r : ℕ) : a = 23 ∧ k = 2023 ∧ n = 29 ∧ r = 24 → (a^k % n = r) := 
by
  intro h
  cases h with ha hk
  cases hk with hk hn
  cases hn with hn hr
  rw [ha, hk, hn, hr]
  sorry

end power_mod_l634_634681


namespace number_of_non_isolated_4_subsets_l634_634945

-- Definition of the set S
def S := {1, 2, 3, 4, 5, 6}

-- Definition of an isolated element in a subset A of S
def is_isolated (A : Set ℕ) (x : ℕ) : Prop :=
  x ∈ A ∧ x - 1 ∉ A ∧ x + 1 ∉ A

-- Definition of the condition that no element in A is isolated
def no_isolated_elements (A : Set ℕ) : Prop :=
  ∀ x ∈ A, ¬ is_isolated A x

-- The main statement to prove
theorem number_of_non_isolated_4_subsets : 
  ∃ (A : Finset (Finset ℕ)), 
    (∀ a ∈ A, a.card = 4 ∧ no_isolated_elements a) ∧ 
    A.card = 6 :=
sorry

end number_of_non_isolated_4_subsets_l634_634945


namespace yoki_cans_collected_l634_634008

theorem yoki_cans_collected (total_cans LaDonna_cans Prikya_cans Avi_cans : ℕ) (half_Avi_cans Yoki_cans : ℕ) 
    (h1 : total_cans = 85) 
    (h2 : LaDonna_cans = 25) 
    (h3 : Prikya_cans = 2 * LaDonna_cans - 3) 
    (h4 : Avi_cans = 8) 
    (h5 : half_Avi_cans = Avi_cans / 2) 
    (h6 : total_cans = LaDonna_cans + Prikya_cans + half_Avi_cans + Yoki_cans) :
    Yoki_cans = 9 := sorry

end yoki_cans_collected_l634_634008


namespace money_left_unshredded_is_one_l634_634942

-- Defining the conditions
def total_earned : ℕ := 28
def fraction_spent_on_milkshake : ℚ := 1/7
def money_spent_on_milkshake : ℕ := total_earned * fraction_spent_on_milkshake
def remaining_after_milkshake : ℕ := total_earned - money_spent_on_milkshake
def fraction_saved : ℚ := 1/2
def money_saved : ℕ := remaining_after_milkshake * fraction_saved
def money_left_in_wallet : ℕ := remaining_after_milkshake - money_saved
def money_lost_due_to_dog : ℕ := 11
def money_unshredded : ℕ := money_left_in_wallet - money_lost_due_to_dog

-- Statement of the theorem
theorem money_left_unshredded_is_one : money_unshredded = 1 :=
by
  -- Skipping the proof for now with sorry
  sorry

end money_left_unshredded_is_one_l634_634942


namespace arc_length_gt_diameter_l634_634758

theorem arc_length_gt_diameter
  (C C' : Topology.Circle) -- Defining circles C and C'
  (A B : Topology.Point) -- Points of intersection A and B
  (h1 : Topology.IntersectAt C C' A B) -- Circles intersect at points A and B
  (h2 : Topology.ArcDividesArea C' A B C) -- Arc AB of C' divides area of C into two equal parts
  : Topology.ArcLength C' A B > Topology.Diameter C := -- Theorem statement
sorry

end arc_length_gt_diameter_l634_634758


namespace problem_1_problem_2_l634_634050

theorem problem_1 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^3 + b^3 = 2) : (a + b) * (a^5 + b^5) ≥ 4 :=
sorry

theorem problem_2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a^3 + b^3 = 2) : a + b ≤ 2 :=
sorry

end problem_1_problem_2_l634_634050


namespace beth_route_longer_by_5_miles_l634_634935

def average_speed_jerry := 40
def time_jerry_minutes := 30
def average_speed_beth := 30
def additional_time_beth_minutes := 20

def time_jerry_hours := time_jerry_minutes / 60.0
def distance_jerry := average_speed_jerry * time_jerry_hours

def time_beth_minutes := time_jerry_minutes + additional_time_beth_minutes
def time_beth_hours := time_beth_minutes / 60.0
def distance_beth := average_speed_beth * time_beth_hours

def route_difference := distance_beth - distance_jerry

theorem beth_route_longer_by_5_miles : route_difference = 5 := 
sorry

end beth_route_longer_by_5_miles_l634_634935


namespace max_k_value_l634_634060

-- Definitions of given conditions
def a_seq (n : ℕ) : ℕ := n + 2

def b_seq (n : ℕ) : ℕ := 2 * n + 3

def C_seq (n : ℕ) : ℝ := 1 / (2 * (a_seq n) - 3) * (2 * (b_seq n) - 8)

def T_seq (n : ℕ) : ℝ := (1 / 4) * (1 - 1 / (2 * n + 1))

-- Theorem to prove
theorem max_k_value : ∃ k : ℕ, k = 8 ∧ ∀ n : ℕ, T_seq n > k / 54 :=
by
  sorry

end max_k_value_l634_634060


namespace triangle_to_rectangle_l634_634768

theorem triangle_to_rectangle 
  (A B C D E F G : Point) 
  (h1: is_largest_angle A A B C)
  (h2: D = midpoint A B)
  (h3: E = midpoint A C)
  (h4: perpendicular_bisector D BC F)
  (h5: perpendicular_bisector E BC G)
  (h6: F = intersection (perpendicular_bisector D BC) BC)
  (h7: G = intersection (perpendicular_bisector E BC) BC)
  : dissect_triangle A B C D E F G ∧ three_right_angled_triangles A D F G ∧ forms_rectangle A D F D G E F :=
sorry

end triangle_to_rectangle_l634_634768


namespace linear_functions_exist_l634_634414

-- Definitions of P(x) and Q(x) as given in the conditions
def P (x : ℝ) : ℝ := (4/21) * x + (5/21)
def Q (x : ℝ) : ℝ := (-4/21) * x + (11/21)

-- Main theorem stating the required equality
theorem linear_functions_exist :
  ∀ (x : ℝ), 
    (P x) * (2 * x^3 - 7 * x^2 + 7 * x - 2) + (Q x) * (2 * x^3 + x^2 + x - 1) = 2 * x - 1 :=
by
  sorry

end linear_functions_exist_l634_634414


namespace igor_ratio_not_possible_l634_634517

theorem igor_ratio_not_possible :
  ∀ n : ℕ, let S₁ := 10 * n + 45,
               S₂ := 10 * n + 145
           in (S₁ : ℚ) / S₂ ≠ 0.8 :=
by
  intros n S₁ S₂
  dsimp only [S₁, S₂]
  sorry

end igor_ratio_not_possible_l634_634517


namespace average_of_three_strings_l634_634976

variable (len1 len2 len3 : ℝ)
variable (h1 : len1 = 2)
variable (h2 : len2 = 5)
variable (h3 : len3 = 3)

def average_length (a b c : ℝ) : ℝ := (a + b + c) / 3

theorem average_of_three_strings : average_length len1 len2 len3 = 10 / 3 := by
  rw [←h1, ←h2, ←h3]
  rw [average_length]
  norm_num
  sorry

end average_of_three_strings_l634_634976


namespace recyclable_cans_and_bottles_collected_l634_634661

-- Define the conditions in Lean
def people_at_picnic : ℕ := 90
def soda_cans : ℕ := 50
def plastic_bottles_sparkling_water : ℕ := 50
def glass_bottles_juice : ℕ := 50
def guests_drank_soda : ℕ := people_at_picnic / 2
def guests_drank_sparkling_water : ℕ := people_at_picnic / 3
def juice_consumed : ℕ := (glass_bottles_juice * 4) / 5

-- The theorem statement
theorem recyclable_cans_and_bottles_collected :
  (soda_cans + guests_drank_sparkling_water + juice_consumed) = 120 :=
by
  sorry

end recyclable_cans_and_bottles_collected_l634_634661


namespace walnuts_cost_check_l634_634011

namespace ShoppingProblem

variable (price_apples_per_kg : ℝ) (kg_apples : ℝ) (packs_sugar : ℝ) (total_payment : ℝ) (price_per_pack_sugar : ℝ) (total_payment_apples_sugar : ℝ) (price_500g_walnuts : ℝ)

-- Conditions
def condition1 : price_apples_per_kg = 2 := rfl
def condition2 : kg_apples = 5 := rfl
def condition3 : packs_sugar = 3 := rfl
def condition4 : price_per_pack_sugar = price_apples_per_kg - 1 := by simp [condition1]
def condition5 : total_payment = 16 := rfl
def condition6 : total_payment_apples_sugar = (kg_apples * price_apples_per_kg) + (packs_sugar * price_per_pack_sugar) := by simp [condition1, condition2, condition3, condition4]

-- Remaining amount for walnuts
def remaining_for_walnuts := total_payment - total_payment_apples_sugar

def cost_one_kg_walnuts_is_6 : Prop :=
  ∃ (price_walnuts_per_kg : ℝ), price_500g_walnuts = remaining_for_walnuts ∧ price_walnuts_per_kg = price_500g_walnuts * 2

theorem walnuts_cost_check : cost_one_kg_walnuts_is_6 :=
  have h : price_500g_walnuts = (16 - ((5 * 2) + (3 * 1))) := by simp [condition1, condition2, condition3, condition4, condition5, condition6]
  have hk : (16 - ((5 * 2) + (3 * 1))) = 3 := by simp
  have hw : h = rfl := by simp
  exists.intro 6 sorry

end ShoppingProblem

end walnuts_cost_check_l634_634011


namespace find_a_for_perpendicular_lines_l634_634421

-- slopes
def slope_first_line : ℚ := 2

def slope_second_line (a : ℚ) : ℚ := -a / 18

-- condition for perpendicular lines
def perpendicular_lines (a : ℚ) : Prop :=
  slope_first_line * slope_second_line(a) = -1

-- define the problem statement
theorem find_a_for_perpendicular_lines :
  (∃ a : ℚ, perpendicular_lines a) ↔ a = 9 := sorry

end find_a_for_perpendicular_lines_l634_634421


namespace geometric_sequence_sum_l634_634947

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = a n * q)
  (h1 : a 0 + a 1 = 1) (h2 : a 1 + a 2 = 2) : a 5 + a 6 = 32 :=
sorry

end geometric_sequence_sum_l634_634947


namespace petya_wins_probability_l634_634986

/-- Petya plays a game with 16 stones where players alternate in taking 1 to 4 stones. 
     Petya wins if they can take the last stone first while making random choices. 
     The computer plays optimally. The probability of Petya winning is 1 / 256. -/
theorem petya_wins_probability :
  let stones := 16 in
  let optimal_strategy := ∀ n, n % 5 = 0 in
  let random_choice_probability := (1 / 4 : ℚ) in
  let total_random_choices := 4 in
  (random_choice_probability ^ total_random_choices) = (1 / 256 : ℚ) :=
by
  sorry

end petya_wins_probability_l634_634986


namespace size_relationship_l634_634437

noncomputable def a : ℝ := 1 + Real.sqrt 7
noncomputable def b : ℝ := Real.sqrt 3 + Real.sqrt 5
noncomputable def c : ℝ := 4

theorem size_relationship : a < b ∧ b < c := by
  sorry

end size_relationship_l634_634437


namespace DE_equals_8_l634_634920

variable (DEF : Triangle)
variable (D E F : Point)
variable (right_triangle_DEF : IsRightTriangle DEF D E F)
variable (hypotenuse_DF : Hypotenuse DEF DF)
variable (cos_E : cos E = (8 * sqrt 145) / 145)
variable (DF_length : length DF = sqrt 145)

theorem DE_equals_8 :
  length DE = 8 :=
sorry

end DE_equals_8_l634_634920


namespace determine_x_l634_634054

variable {m x : ℝ}

theorem determine_x (h₁ : m > 25)
    (h₂ : ((m / 100) * m = (m - 20) / 100 * (m + x))) : 
    x = 20 * m / (m - 20) := 
sorry

end determine_x_l634_634054


namespace max_value_of_trig_expression_l634_634025

open Real

theorem max_value_of_trig_expression : ∀ x : ℝ, 3 * cos x + 4 * sin x ≤ 5 :=
sorry

end max_value_of_trig_expression_l634_634025


namespace basketball_lineup_count_l634_634228

theorem basketball_lineup_count :
  let n := 12
  let positions := 5
  (12 * 11 * 10 * 9 * 8 = 95_040) :=
by
  sorry

end basketball_lineup_count_l634_634228


namespace correct_conclusions_l634_634687

-- Theorem statement proving conclusions ① and ③
theorem correct_conclusions (a b c d : ℝ) (h_a_gt_b : a > b) (h_c_lt_d : c < d) (h_b_pos : b > 0) :
  (a - c > b - d)
  ∧ ((a > 0) → (b > 0) → (cbrt a > cbrt b)) :=
by
  sorry

end correct_conclusions_l634_634687


namespace general_formula_sum_of_reciprocals_l634_634172

-- Definitions for the arithmetic sequence and the conditions
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (2 * a 0 + (n - 1) * (a 1 - a 0))) / 2

def geometric_sequence (S : ℕ → ℤ) : Prop :=
  S 1 * S 4 = S 2 ^ 2

def condition_1 (a : ℕ → ℤ) : Prop :=
  a 3 = 2 * a 1 + 2

-- Proving the two parts of the problem
theorem general_formula (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ) : 
  arithmetic_sequence a d → 
  sum_of_first_n_terms S a →
  geometric_sequence S → 
  condition_1 a →
  a n = 4 * n - 2 :=
  sorry

theorem sum_of_reciprocals (n : ℕ) (a : ℕ → ℤ) :
  (∀ k, a k = 4 * k - 2) → 
  ∑ k in finRange n, (1 : ℚ) / (a k * a (k + 1)) = 𝟏 / 𝟖 * (1 - 𝟏 / (2 * n + 1)) :=
  sorry

end general_formula_sum_of_reciprocals_l634_634172


namespace interval_of_monotonic_decrease_l634_634200

noncomputable def f : ℝ → ℝ := sorry  -- assume f exists as a decreasing function
noncomputable def g (x : ℝ) := x^2 - 2*x + 3

-- We assume that f is decreasing, i.e., for all x, y such that x ≤ y, we have f(y) ≤ f(x)
axiom f_decreasing : ∀ x y : ℝ, x ≤ y → f(y) ≤ f(x)

-- g is increasing on [1, +∞), we state it as follows
axiom g_increasing_on_1_inf : ∀ x y : ℝ, 1 ≤ x → x ≤ y → g(x) ≤ g(y)

theorem interval_of_monotonic_decrease :
  ∀ x y : ℝ, 1 ≤ x → x ≤ y → f(g(y)) ≤ f(g(x)) :=
by
  sorry

end interval_of_monotonic_decrease_l634_634200


namespace g_g_even_l634_634592

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

def g : ℝ → ℝ := sorry

theorem g_g_even (h : is_even_function g) : is_even_function (λ x, g (g x)) :=
by
  -- proof omitted
  sorry

end g_g_even_l634_634592


namespace playground_girls_l634_634284

theorem playground_girls (total_children boys girls : ℕ) (h1 : boys = 40) (h2 : total_children = 117) (h3 : total_children = boys + girls) : girls = 77 := 
by 
  sorry

end playground_girls_l634_634284


namespace product_of_chords_lengths_l634_634597

def A : ℂ := 3
def B : ℂ := -3
def ω : ℂ := complex.exp (2 * real.pi * complex.I / 18)
def D (k : ℕ) (hk : k > 0 ∧ k < 9) : ℂ := 3 * (ω^k)

theorem product_of_chords_lengths :
  (∏ (i : ℕ) in finset.range 1 9, complex.abs (A - D i (sorry))) *
  (∏ (i : ℕ) in finset.range 1 9, complex.abs (B - D i (sorry))) = 774840978 := 
sorry

end product_of_chords_lengths_l634_634597


namespace find_lambda_l634_634824

def vectors_and_angle_conditions (a b: Vector ℝ) (m: ℝ) (h₁: 0 < m) (h₂: ∥a∥ = m) (h₃: ∥b∥ = 2 * m) (h₄: ∠ a b = real.pi * 2 / 3) : Prop :=
  a ⬝ (a - λ • b) = 0 → λ = -1

theorem find_lambda (a b: Vector ℝ) (m: ℝ) (h₁: 0 < m) (h₂: ∥a∥ = m) (h₃: ∥b∥ = 2 * m) (h₄: ∠ a b = real.pi * 2 / 3) (h₅: a ⬝ (a - λ • b) = 0) : λ = -1 :=
sorry

end find_lambda_l634_634824


namespace trajectory_of_center_of_moving_circle_l634_634346

theorem trajectory_of_center_of_moving_circle (M : ℝ × ℝ) :
  (∀ (M : ℝ × ℝ), (∃ r > 0, ((M.2 - 2) = r ∧ dist (M.1, M.2) (0, -3) = r + 1))) →
  (∃ y : ℝ, (M.1)^2 = -12 * y) :=
by
  sorry

end trajectory_of_center_of_moving_circle_l634_634346


namespace monotonic_increasing_quadratic_l634_634265

theorem monotonic_increasing_quadratic (b : ℝ) (c : ℝ) :
  (∀ x y : ℝ, (0 ≤ x → x ≤ y → (x^2 + b*x + c) ≤ (y^2 + b*y + c))) ↔ (b ≥ 0) :=
sorry  -- Proof is omitted

end monotonic_increasing_quadratic_l634_634265


namespace infinite_sequence_no_square_factors_l634_634233

/-
  Prove that there exist infinitely many positive integers \( n_1 < n_2 < \cdots \)
  such that for all \( i \neq j \), \( n_i + n_j \) has no square factors other than 1.
-/

theorem infinite_sequence_no_square_factors :
  ∃ (n : ℕ → ℕ), (∀ (i j : ℕ), i ≠ j → ∀ p : ℕ, p ≠ 1 → p^2 ∣ (n i + n j) → false) ∧
    ∀ k : ℕ, n k < n (k + 1) :=
sorry

end infinite_sequence_no_square_factors_l634_634233


namespace min_decimal_digits_l634_634679

theorem min_decimal_digits : 
  (let frac := (987654321 : ℚ) / (2^30 * 5^3) in 
   ∃ n : ℕ, n = 30 ∧ frac = (frac.num / frac.denom) * 10^{-n}) :=
sorry

end min_decimal_digits_l634_634679


namespace acute_triangle_angle_C_acute_triangle_sum_ab_l634_634529

open Real

theorem acute_triangle_angle_C
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_acute : a^2 + b^2 > c^2)
  (eq1 : sqrt 3 * a = 2 * c * sin (1/2 * Real.pi))
  (h_area : c = sqrt 7)
  (h_area2 : 1 / 2 * a * b * sin (1/3 * Real.pi) = 3 * sqrt 3 / 2) :
  (C : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → ℝ)
   := sorry

theorem acute_triangle_sum_ab
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_acute : a^2 + b^2 > c^2)
  (h_area : c = sqrt 7)
  (h_area2 : 1 / 2 * a * b * sin (1/3 * Real.pi) = 3 * sqrt 3 / 2)
  (C_eq : C = 1/3 * Real.pi) :
  a + b = 5 := sorry

end acute_triangle_angle_C_acute_triangle_sum_ab_l634_634529


namespace profit_percentage_on_cp_l634_634368

variable (CP MP SP P : ℝ)

theorem profit_percentage_on_cp :
  CP = 47.50 → MP = 65 → SP = 0.95 * MP → 
  let P := SP - CP in 
  (P / CP) * 100 = 30 :=
by
  intros h_cp h_mp h_sp
  have h1 : SP = 0.95 * MP := h_sp
  have h2 : CP = 47.50 := h_cp
  have h3 : MP = 65 := h_mp
  sorry

end profit_percentage_on_cp_l634_634368


namespace evaluate_product_l634_634406

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_product : 
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 2657205 :=
by 
  sorry

end evaluate_product_l634_634406


namespace sqrt_eq_cond_l634_634588

theorem sqrt_eq_cond (a b c : ℕ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
    (not_perfect_square_a : ¬(∃ n : ℕ, n * n = a)) (not_perfect_square_b : ¬(∃ n : ℕ, n * n = b))
    (not_perfect_square_c : ¬(∃ n : ℕ, n * n = c)) :
    (Real.sqrt a + Real.sqrt b = Real.sqrt c) →
    (2 * Real.sqrt (a * b) = c - (a + b) ∧ (∃ k : ℕ, a * b = k * k)) :=
sorry

end sqrt_eq_cond_l634_634588


namespace lyceum_student_count_l634_634925

-- We define the variables and conditions
def percentage_not_homework (students : ℕ) := students * 76 / 100
def fraction_forget_shoes (students : ℕ) := students * 5 / 37
def lcm_25_37 := 925

-- Finally, we state the proof problem
theorem lyceum_student_count (students : ℕ)
  (h_homework : percentage_not_homework students)
  (h_shoes : fraction_forget_shoes students)
  (h_range : students > 1000 ∧ students < 2000) :
  ∃ k : ℕ, students = 925 * k ∧ 1000 < students ∧ students < 2000 :=
by
  sorry

end lyceum_student_count_l634_634925


namespace find_m_l634_634580

variable {y m : ℝ} -- define variables y and m in the reals

-- define the logarithmic conditions
axiom log8_5_eq_y : log 8 5 = y
axiom log2_125_eq_my : log 2 125 = m * y

-- state the theorem to prove m equals 9
theorem find_m (log8_5_eq_y : log 8 5 = y) (log2_125_eq_my : log 2 125 = m * y) : m = 9 := by
  sorry

end find_m_l634_634580


namespace find_2016th_smallest_n_l634_634191

def a_n (n : ℕ) : ℕ := sorry -- Placeholder for the actual relevant function

def satisfies_condition (n : ℕ) : Prop :=
  a_n n ≡ 1 [MOD 5]

theorem find_2016th_smallest_n :
  (∃ n, ∀ m < 2016, satisfies_condition m → satisfies_condition n ∧ n < m)
  ∧ 
  satisfies_condition 2016 → 
  2016 = 475756 :=
sorry

end find_2016th_smallest_n_l634_634191


namespace edward_earnings_l634_634007

theorem edward_earnings 
  (regular_hours : ℕ) (total_hours_worked : ℕ) (regular_pay_rate : ℝ) (overtime_multiplier : ℝ)
  (h1 : regular_hours = 40)
  (h2 : total_hours_worked = 45)
  (h3 : regular_pay_rate = 7)
  (h4 : overtime_multiplier = 2) :
  let overtime_hours := total_hours_worked - regular_hours in
  let overtime_pay_rate := regular_pay_rate * overtime_multiplier in
  let regular_earnings := regular_hours * regular_pay_rate in
  let overtime_earnings := overtime_hours * overtime_pay_rate in
  let total_earnings := regular_earnings + overtime_earnings in
  total_earnings = 350 := 
by 
  rw [h1, h2, h3, h4]
  sorry

end edward_earnings_l634_634007


namespace part1_part2_a_part2_b_part2_c_l634_634092

noncomputable def f (x a : ℝ) := Real.exp x - x - a

theorem part1 (x : ℝ) : f x 0 > x := 
by 
  -- here would be the proof
  sorry

theorem part2_a (a : ℝ) : a > 1 → ∃ z₁ z₂ : ℝ, f z₁ a = 0 ∧ f z₂ a = 0 ∧ z₁ ≠ z₂ := 
by 
  -- here would be the proof
  sorry

theorem part2_b (a : ℝ) : a < 1 → ¬ (∃ z : ℝ, f z a = 0) := 
by 
  -- here would be the proof
  sorry

theorem part2_c : f 0 1 = 0 := 
by 
  -- here would be the proof
  sorry

end part1_part2_a_part2_b_part2_c_l634_634092


namespace sum_of_two_youngest_is_14_l634_634642

-- Conditions
def mean_age (ages : List ℕ) : ℕ := (ages.sum / ages.length)
def median_age (ages : List ℕ) : ℕ := ages.nth_le (ages.length / 2) sorry

theorem sum_of_two_youngest_is_14 (ages : List ℕ) (h_mean : ages.length = 5 ∧ mean_age ages = 10) 
  (h_median : ages.nth_le 2 sorry = 12) :
  ages.sorted.take 2.sum = 14 :=
by
  sorry

end sum_of_two_youngest_is_14_l634_634642


namespace number_divisors_l634_634267

theorem number_divisors (p : ℕ) (h : p = 2^56 - 1) : ∃ x y : ℕ, 95 ≤ x ∧ x ≤ 105 ∧ 95 ≤ y ∧ y ≤ 105 ∧ p % x = 0 ∧ p % y = 0 ∧ x = 101 ∧ y = 127 :=
by {
  sorry
}

end number_divisors_l634_634267


namespace lateral_area_of_given_cone_l634_634465

noncomputable def lateral_area_cone (r h : ℝ) : ℝ :=
  let l := Real.sqrt (r^2 + h^2)
  (Real.pi * r * l)

theorem lateral_area_of_given_cone :
  lateral_area_cone 3 4 = 15 * Real.pi :=
by
  -- sorry to skip the proof
  sorry

end lateral_area_of_given_cone_l634_634465


namespace find_angle_C_l634_634603

theorem find_angle_C 
  (A B C : ℝ) 
  (h_sum_angles : A + B + C = π)
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hm : ℝ × ℝ := (sqrt 3 * sin A, sin B))
  (hn : ℝ × ℝ := (cos B, sqrt 3 * cos A))
  (h_dot_product : (hm.1 * hn.1 + hm.2 * hn.2) = 1 + cos (A + B)) :
  C = 2 * π / 3 :=
begin
  -- proof to be added
  sorry
end

end find_angle_C_l634_634603


namespace outfits_count_l634_634691

def red_shirts := 8
def green_shirts := 7
def blue_pants := 8
def red_hats := 10
def green_hats := 9
def black_belts := 5
def brown_belts := 4

theorem outfits_count :
  (red_shirts * blue_pants * green_hats * brown_belts) 
  + (green_shirts * blue_pants * red_hats * black_belts) = 5104 := 
by 
  calc 
  8 * 8 * 9 * 4 + 7 * 8 * 10 * 5 = 2304 + 2800 : by norm_num
                         ... = 5104          : by norm_num

end outfits_count_l634_634691


namespace house_vs_trailer_payment_difference_l634_634178

-- Definitions based on given problem conditions
def house_cost : ℝ := 480000
def trailer_cost : ℝ := 120000
def loan_term_years : ℝ := 20
def months_in_year : ℝ := 12
def total_months : ℝ := loan_term_years * months_in_year

-- Monthly payment calculations
def house_monthly_payment : ℝ := house_cost / total_months
def trailer_monthly_payment : ℝ := trailer_cost / total_months

-- The theorem we need to prove
theorem house_vs_trailer_payment_difference :
  house_monthly_payment - trailer_monthly_payment = 1500 := 
by
  sorry

end house_vs_trailer_payment_difference_l634_634178


namespace problem_1_problem_2_l634_634846

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * a * x^2 + 2 * x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x - g x a
noncomputable def h' (x : ℝ) (a : ℝ) : ℝ := (1 / x) - a * x - 2
noncomputable def G (x : ℝ) : ℝ := ((1 / x) - 1) ^ 2 - 1

theorem problem_1 (a : ℝ): 
  (∃ x : ℝ, 0 < x ∧ h' x a < 0) ↔ a > -1 :=
by sorry

theorem problem_2 (a : ℝ):
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → h' x a ≤ 0) ↔ a ≥ -(7 / 16) :=
by sorry

end problem_1_problem_2_l634_634846


namespace area_of_circle_correct_l634_634618

noncomputable def area_of_circle (A B : (ℝ × ℝ)) (tangent_meeting_point_x_axis : (ℝ × ℝ)) : ℝ :=
  let D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  let slope_AB := (B.2 - A.2) / (B.1 - A.1) in
  let slope_CD := -1 / slope_AB in
  let x_intersect := tangent_meeting_point_x_axis.1 in
  let C := (x_intersect, 0) in
  let AC := real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) in
  let AD := real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) in
  let DC := real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) in
  let OA := real.sqrt (AC * AD / DC) in
  real.pi * OA^2

theorem area_of_circle_correct :
  let A := (6, 13) in
  let B := (12, 11) in
  ∃ x_intersect, (area_of_circle A B (x_intersect, 0)) = 85 * real.pi / 8 :=
sorry

end area_of_circle_correct_l634_634618


namespace effective_days_21_minimal_m_l634_634910

-- Define the function f(x)
def f (x : ℝ) : ℝ :=
  if (0 < x) ∧ (x ≤ 5) then x^2 / 25 + 2
  else (x + 19) / (2*x - 2)

-- Define the concentration of the purification agent
def concentration (m x : ℝ) : ℝ := m * f x

-- Define the effective purification condition
def effective_purification (m x : ℝ) : Prop := concentration m x ≥ 5

-- Define the optimal purification condition
def optimal_purification (m x : ℝ) : Prop := 5 ≤ concentration m x ∧ concentration m x ≤ 10

-- Given m = 5, prove the water maintains effective purification for 21 days
theorem effective_days_21 : effective_purification 5 21 := sorry

-- Prove the minimum value of m for optimal purification within 9 days is 20/7
theorem minimal_m : ∀ m, (∀ x, x ≤ 9 → optimal_purification m x) ↔ ((20/7) ≤ m ∧ m ≤ (10/3)) := sorry

end effective_days_21_minimal_m_l634_634910


namespace roots_sum_squares_l634_634760

theorem roots_sum_squares (a b c : ℝ) (h₁ : Polynomial.eval a (3 * X^3 - 2 * X^2 + 5 * X + 15) = 0)
  (h₂ : Polynomial.eval b (3 * X^3 - 2 * X^2 + 5 * X + 15) = 0)
  (h₃ : Polynomial.eval c (3 * X^3 - 2 * X^2 + 5 * X + 15) = 0) :
  a^2 + b^2 + c^2 = -26 / 9 :=
sorry

end roots_sum_squares_l634_634760


namespace etienne_diana_value_difference_l634_634248

theorem etienne_diana_value_difference :
  let euro_to_dollar : ℝ := 1.5
  let diana_dollars : ℝ := 600
  let etienne_euros : ℝ := 450
  let etienne_dollars := etienne_euros * euro_to_dollar
  let percentage_difference := ((diana_dollars - etienne_dollars) / etienne_dollars) * 100
  percentage_difference ≈ -11.11 :=
by
  sorry

end etienne_diana_value_difference_l634_634248


namespace intersection_of_M_and_complement_N_l634_634205

def M : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def N : Set ℝ := { x | 2 * x < 2 }
def complement_N : Set ℝ := { x | x ≥ 1 }

theorem intersection_of_M_and_complement_N : M ∩ complement_N = { x | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_of_M_and_complement_N_l634_634205


namespace S_2016_value_l634_634093

def a (n : ℕ) : ℝ := n * Real.cos (n * Real.pi / 2)

def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)

theorem S_2016_value : S 2016 = 1008 := 
by
  sorry

end S_2016_value_l634_634093


namespace power_equality_l634_634688

-- Definitions based on conditions
def nine := 3^2

-- Theorem stating the given mathematical problem
theorem power_equality : nine^4 = 3^8 := by
  sorry

end power_equality_l634_634688


namespace isosceles_trapezoid_area_l634_634530

theorem isosceles_trapezoid_area (a b : ℝ) : 
  ∃ (T : ℝ), (∀ (AB CD : ℝ), AB = a → CD = b → ∠BAC = 45 → T = (a^2 - b^2) / 4) :=
sorry

end isosceles_trapezoid_area_l634_634530


namespace area_of_rhombus_l634_634527

structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 2 5.5
def B := Point.mk 8.5 1
def C := Point.mk 2 (-3.5)
def D := Point.mk (-4.5) 1

def vector (p q : Point) := (q.x - p.x, q.y - p.y)

def AC := vector A C
def BD := vector B D

def cross_product (u v : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (0, 0, u.1 * v.2 - u.2 * v.1)

def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

def area_parallelogram (u v : ℝ × ℝ) : ℝ :=
  magnitude (cross_product u v)

def area_rhombus (u v : ℝ × ℝ) : ℝ :=
  area_parallelogram u v / 2

theorem area_of_rhombus : area_rhombus AC BD = 58.5 := by
  sorry

end area_of_rhombus_l634_634527
