import Mathlib

namespace NUMINAMATH_CALUDE_breakfast_cost_is_correct_l3655_365589

/-- Calculates the total cost of breakfast for Francis, Kiera, and David --/
def total_breakfast_cost (muffin_price fruit_cup_price coffee_price : ℚ)
  (discount_rate : ℚ) (voucher : ℚ) : ℚ :=
  let francis_cost := 2 * muffin_price + 2 * fruit_cup_price + coffee_price -
    discount_rate * (2 * muffin_price + fruit_cup_price)
  let kiera_cost := 2 * muffin_price + fruit_cup_price + coffee_price -
    discount_rate * (2 * muffin_price + fruit_cup_price)
  let david_cost := 3 * muffin_price + fruit_cup_price + coffee_price - voucher
  francis_cost + kiera_cost + david_cost

/-- Theorem stating that the total breakfast cost is $27.10 --/
theorem breakfast_cost_is_correct :
  total_breakfast_cost 2 3 1.5 0.1 2 = 27.1 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_cost_is_correct_l3655_365589


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3655_365514

theorem arithmetic_calculation : 15 * 35 + 50 * 15 - 5 * 15 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3655_365514


namespace NUMINAMATH_CALUDE_final_price_percentage_l3655_365523

/-- The price change over 5 years -/
def price_change (p : ℝ) : ℝ :=
  p * 1.3 * 0.8 * 1.25 * 0.9 * 1.15

/-- Theorem stating the final price is 134.55% of the original price -/
theorem final_price_percentage (p : ℝ) (hp : p > 0) :
  price_change p / p = 1.3455 := by
  sorry

end NUMINAMATH_CALUDE_final_price_percentage_l3655_365523


namespace NUMINAMATH_CALUDE_original_denominator_proof_l3655_365566

theorem original_denominator_proof : 
  ∀ d : ℚ, (5 : ℚ) / (d + 4) = (1 : ℚ) / 3 → d = 11 := by
  sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l3655_365566


namespace NUMINAMATH_CALUDE_worm_length_difference_l3655_365556

theorem worm_length_difference : 
  let longer_worm : ℝ := 0.8
  let shorter_worm : ℝ := 0.1
  longer_worm - shorter_worm = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_worm_length_difference_l3655_365556


namespace NUMINAMATH_CALUDE_adams_purchase_cost_l3655_365562

/-- The cost of Adam's purchases given the quantities and prices of nuts and dried fruits -/
theorem adams_purchase_cost (nuts_quantity : ℝ) (nuts_price : ℝ) (fruits_quantity : ℝ) (fruits_price : ℝ) 
  (h1 : nuts_quantity = 3)
  (h2 : nuts_price = 12)
  (h3 : fruits_quantity = 2.5)
  (h4 : fruits_price = 8) :
  nuts_quantity * nuts_price + fruits_quantity * fruits_price = 56 := by
  sorry

end NUMINAMATH_CALUDE_adams_purchase_cost_l3655_365562


namespace NUMINAMATH_CALUDE_cos_equation_solution_l3655_365571

theorem cos_equation_solution (x : ℝ) : 
  (Real.cos (2 * x) - 3 * Real.cos (4 * x))^2 = 16 + (Real.cos (5 * x))^2 → 
  ∃ k : ℤ, x = π / 2 + k * π :=
by sorry

end NUMINAMATH_CALUDE_cos_equation_solution_l3655_365571


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3655_365521

def M : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def N : Set ℝ := {x | x < 0}

theorem intersection_complement_theorem : 
  M ∩ (Set.univ \ N) = {x : ℝ | 0 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3655_365521


namespace NUMINAMATH_CALUDE_ellipse_properties_hyperbola_properties_l3655_365524

/-- An ellipse with the given properties -/
def ellipse (x y : ℝ) : Prop :=
  x^2 / 15 + y^2 / 10 = 1

/-- A hyperbola with the given properties -/
def hyperbola (x y : ℝ) : Prop :=
  x^2 / (35/9) - y^2 / 35 = 1

theorem ellipse_properties :
  (∀ x y, ellipse x y → (x^2 / 9 + y^2 / 4 = 1 → 
    ∃ c, c^2 = 5 ∧ (∀ x' y', x'^2 / 9 + y'^2 / 4 = 1 → 
      (x' - c)^2 + y'^2 = (x' + c)^2 + y'^2))) ∧
  ellipse (-3) 2 := by sorry

theorem hyperbola_properties :
  (∀ x y, hyperbola x y → (y = 3*x ∨ y = -3*x)) ∧
  hyperbola 2 (-1) := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_hyperbola_properties_l3655_365524


namespace NUMINAMATH_CALUDE_gcd_problem_l3655_365509

theorem gcd_problem (b : ℤ) (h : 2373 ∣ b) : 
  Nat.gcd (Int.natAbs (b^2 + 13*b + 40)) (Int.natAbs (b + 5)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3655_365509


namespace NUMINAMATH_CALUDE_last_name_length_proof_l3655_365526

/-- Given information about the lengths of last names, prove the length of another person's last name --/
theorem last_name_length_proof (samantha_length bobbie_length other_length : ℕ) : 
  samantha_length = 7 →
  bobbie_length = samantha_length + 3 →
  bobbie_length - 2 = 2 * other_length →
  other_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_name_length_proof_l3655_365526


namespace NUMINAMATH_CALUDE_solve_for_k_l3655_365559

/-- The function f(x) = 4x³ - 3x² + 2x + 5 -/
def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 + 2 * x + 5

/-- The function g(x) = x³ - (k+1)x² - 7x - 8 -/
def g (k x : ℝ) : ℝ := x^3 - (k + 1) * x^2 - 7 * x - 8

/-- If f(5) - g(5) = 24, then k = -16.36 -/
theorem solve_for_k : ∃ k : ℝ, f 5 - g k 5 = 24 ∧ k = -16.36 := by sorry

end NUMINAMATH_CALUDE_solve_for_k_l3655_365559


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3655_365597

theorem complex_number_in_second_quadrant : 
  let z : ℂ := (Complex.I / (1 + Complex.I)) + (1 + Complex.I * Real.sqrt 3) ^ 2
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l3655_365597


namespace NUMINAMATH_CALUDE_symmetric_probability_l3655_365513

/-- Represents a standard die with faces labeled 1 to 6 -/
def StandardDie : Type := Fin 6

/-- The number of dice being rolled -/
def numDice : Nat := 9

/-- The sum we are comparing to -/
def targetSum : Nat := 14

/-- The sum we want to prove has the same probability as the target sum -/
def symmetricSum : Nat := 49

/-- Function to calculate the probability of a specific sum occurring when rolling n dice -/
noncomputable def probabilityOfSum (n : Nat) (sum : Nat) : ℚ := sorry

theorem symmetric_probability :
  probabilityOfSum numDice targetSum = probabilityOfSum numDice symmetricSum := by sorry

end NUMINAMATH_CALUDE_symmetric_probability_l3655_365513


namespace NUMINAMATH_CALUDE_sum_of_digits_Y_squared_l3655_365508

/-- The number of digits in 222222222 -/
def n : ℕ := 9

/-- The digit repeated in 222222222 -/
def d : ℕ := 2

/-- The number 222222222 -/
def Y : ℕ := d * (10^n - 1) / 9

/-- Sum of digits function -/
def sum_of_digits (x : ℕ) : ℕ := sorry

/-- Theorem: The sum of the digits of the square of 222222222 is 162 -/
theorem sum_of_digits_Y_squared : sum_of_digits (Y^2) = 162 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_Y_squared_l3655_365508


namespace NUMINAMATH_CALUDE_percentage_pies_with_forks_l3655_365572

def total_pies : ℕ := 2000
def pies_not_with_forks : ℕ := 640

theorem percentage_pies_with_forks :
  (total_pies - pies_not_with_forks : ℚ) / total_pies * 100 = 68 := by
  sorry

end NUMINAMATH_CALUDE_percentage_pies_with_forks_l3655_365572


namespace NUMINAMATH_CALUDE_jerry_age_l3655_365592

/-- Given that Mickey's age is 17 years and Mickey's age is 3 years less than 250% of Jerry's age,
    prove that Jerry's age is 8 years. -/
theorem jerry_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 17 → 
  mickey_age = (250 * jerry_age) / 100 - 3 → 
  jerry_age = 8 :=
by sorry

end NUMINAMATH_CALUDE_jerry_age_l3655_365592


namespace NUMINAMATH_CALUDE_cuboids_painted_l3655_365538

theorem cuboids_painted (total_faces : ℕ) (faces_per_cuboid : ℕ) : 
  total_faces = 60 → faces_per_cuboid = 6 → total_faces / faces_per_cuboid = 10 := by
  sorry

end NUMINAMATH_CALUDE_cuboids_painted_l3655_365538


namespace NUMINAMATH_CALUDE_pythagorean_triple_properties_l3655_365587

/-- Given a Pythagorean triple (a, b, c) where c is the hypotenuse,
    prove that certain expressions are perfect squares and
    that certain equations are solvable in integers. -/
theorem pythagorean_triple_properties (a b c : ℤ) 
  (h : a^2 + b^2 = c^2) : -- Pythagorean triple condition
  (∃ (k₁ k₂ k₃ k₄ : ℤ), 
    2*(c-a)*(c-b) = k₁^2 ∧ 
    2*(c-a)*(c+b) = k₂^2 ∧ 
    2*(c+a)*(c-b) = k₃^2 ∧ 
    2*(c+a)*(c+b) = k₄^2) ∧ 
  (∃ (x y : ℤ), 
    x + y + (2*x*y).sqrt = c ∧ 
    x + y - (2*x*y).sqrt = c) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_properties_l3655_365587


namespace NUMINAMATH_CALUDE_parabola_vertex_l3655_365593

/-- A quadratic function f(x) = 2x^2 + px + q with roots at -6 and 4 -/
def f (p q : ℝ) (x : ℝ) : ℝ := 2 * x^2 + p * x + q

theorem parabola_vertex (p q : ℝ) :
  (∀ x ∈ Set.Icc (-6 : ℝ) 4, f p q x ≥ 0) ∧
  (∀ x ∉ Set.Icc (-6 : ℝ) 4, f p q x < 0) →
  ∃ vertex : ℝ × ℝ, vertex = (-1, -50) ∧
    ∀ x : ℝ, f p q x ≥ f p q (-1) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3655_365593


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3655_365584

theorem complex_equation_solution (z : ℂ) : 
  (1 : ℂ) + Complex.I * Real.sqrt 3 = z * ((1 : ℂ) - Complex.I * Real.sqrt 3) →
  z = -(1/2 : ℂ) + Complex.I * (Real.sqrt 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3655_365584


namespace NUMINAMATH_CALUDE_largest_lcm_with_15_l3655_365511

theorem largest_lcm_with_15 : 
  (Finset.image (fun n => Nat.lcm 15 n) {3, 5, 6, 9, 10, 15}).max = some 30 := by
  sorry

end NUMINAMATH_CALUDE_largest_lcm_with_15_l3655_365511


namespace NUMINAMATH_CALUDE_slope_of_CD_is_one_l3655_365558

/-- Given a line y = kx (k > 0) passing through the origin and intersecting the curve y = e^(x-1)
    at two distinct points A(x₁, y₁) and B(x₂, y₂) where x₁ > 0 and x₂ > 0, and points C(x₁, ln x₁)
    and D(x₂, ln x₂) on the curve y = ln x, prove that the slope of line CD is 1. -/
theorem slope_of_CD_is_one (k x₁ x₂ : ℝ) (hk : k > 0) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (hy₁ : k * x₁ = Real.exp (x₁ - 1)) (hy₂ : k * x₂ = Real.exp (x₂ - 1)) :
  (Real.log x₂ - Real.log x₁) / (x₂ - x₁) = 1 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_CD_is_one_l3655_365558


namespace NUMINAMATH_CALUDE_chess_tournament_players_l3655_365598

/-- The number of players in the chess tournament -/
def num_players : ℕ := 45

/-- The total score of all players in the tournament -/
def total_score : ℕ := 1980

/-- Theorem stating that the number of players is correct given the total score -/
theorem chess_tournament_players :
  num_players * (num_players - 1) = total_score :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l3655_365598


namespace NUMINAMATH_CALUDE_product_expansion_l3655_365518

theorem product_expansion (x : ℝ) : 
  (3*x - 4) * (2*x^2 + 3*x - 1) = 6*x^3 + x^2 - 15*x + 4 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l3655_365518


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3655_365535

theorem complex_equation_solution (z : ℂ) :
  (z - 1) * Complex.I = 1 + Complex.I → z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3655_365535


namespace NUMINAMATH_CALUDE_prob_at_least_two_fruits_l3655_365555

/-- The number of fruit types available -/
def num_fruit_types : ℕ := 4

/-- The number of meals in a day -/
def num_meals : ℕ := 3

/-- The probability of choosing a specific fruit for one meal -/
def prob_one_fruit : ℚ := 1 / num_fruit_types

/-- The probability of eating the same fruit for all meals -/
def prob_same_fruit : ℚ := prob_one_fruit ^ num_meals

/-- The probability of eating at least two different kinds of fruit in a day -/
theorem prob_at_least_two_fruits : 
  1 - (num_fruit_types : ℚ) * prob_same_fruit = 15 / 16 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_two_fruits_l3655_365555


namespace NUMINAMATH_CALUDE_no_real_b_for_single_solution_l3655_365517

theorem no_real_b_for_single_solution :
  ¬ ∃ b : ℝ, ∃! x : ℝ, |x^2 + 3*b*x + 5*b| ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_no_real_b_for_single_solution_l3655_365517


namespace NUMINAMATH_CALUDE_max_revenue_is_50_l3655_365527

def neighborhood_A_homes : ℕ := 10
def neighborhood_A_boxes_per_home : ℕ := 2
def neighborhood_B_homes : ℕ := 5
def neighborhood_B_boxes_per_home : ℕ := 5
def price_per_box : ℕ := 2

def revenue_A : ℕ := neighborhood_A_homes * neighborhood_A_boxes_per_home * price_per_box
def revenue_B : ℕ := neighborhood_B_homes * neighborhood_B_boxes_per_home * price_per_box

theorem max_revenue_is_50 : max revenue_A revenue_B = 50 := by
  sorry

end NUMINAMATH_CALUDE_max_revenue_is_50_l3655_365527


namespace NUMINAMATH_CALUDE_inequality_solution_l3655_365564

theorem inequality_solution (p q r : ℝ) 
  (h1 : ∀ x : ℝ, (x - p) * (x - q) / (x - r) ≥ 0 ↔ x ≤ -5 ∨ (20 ≤ x ∧ x ≤ 30))
  (h2 : p < q) : 
  p + 2*q + 3*r = 65 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3655_365564


namespace NUMINAMATH_CALUDE_student_A_selection_probability_l3655_365520

/-- The number of students -/
def n : ℕ := 5

/-- The number of students to be selected -/
def k : ℕ := 2

/-- The probability of selecting student A -/
def prob_A : ℚ := 2/5

/-- The combination function -/
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem student_A_selection_probability :
  (combination (n - 1) (k - 1) : ℚ) / (combination n k : ℚ) = prob_A :=
sorry

end NUMINAMATH_CALUDE_student_A_selection_probability_l3655_365520


namespace NUMINAMATH_CALUDE_midpoint_square_area_l3655_365545

theorem midpoint_square_area (A B C D : ℝ × ℝ) : 
  A = (0, 0) → 
  B = (1, 0) → 
  C = (1, 1) → 
  D = (0, 1) → 
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let N := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let P := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  let Q := ((D.1 + A.1) / 2, (D.2 + A.2) / 2)
  (M.1 - N.1)^2 + (M.2 - N.2)^2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_square_area_l3655_365545


namespace NUMINAMATH_CALUDE_triangle_point_inequalities_l3655_365561

/-- Given a triangle ABC and a point P, prove two inequalities involving side lengths and distances --/
theorem triangle_point_inequalities 
  (A B C P : ℝ × ℝ) -- Points in 2D plane
  (a b c : ℝ) -- Side lengths of triangle ABC
  (α β γ : ℝ) -- Distances from P to A, B, C respectively
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) -- Triangle inequality
  (h_a : a = Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) -- Definition of side length a
  (h_b : b = Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)) -- Definition of side length b
  (h_c : c = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) -- Definition of side length c
  (h_α : α = Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)) -- Definition of distance α
  (h_β : β = Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)) -- Definition of distance β
  (h_γ : γ = Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)) -- Definition of distance γ
  : (a * β * γ + b * γ * α + c * α * β ≥ a * b * c) ∧ 
    (α * b * c + β * c * a + γ * a * b ≥ Real.sqrt 3 * a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_triangle_point_inequalities_l3655_365561


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3655_365580

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b → a > b - 1) ∧
  (∃ a b : ℝ, a > b - 1 ∧ ¬(a > b)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3655_365580


namespace NUMINAMATH_CALUDE_min_rental_cost_l3655_365581

/-- Represents the rental arrangement for buses --/
structure RentalArrangement where
  typeA : ℕ
  typeB : ℕ

/-- Checks if a rental arrangement is valid according to the given constraints --/
def is_valid_arrangement (arr : RentalArrangement) : Prop :=
  36 * arr.typeA + 60 * arr.typeB ≥ 900 ∧
  arr.typeA + arr.typeB ≤ 21 ∧
  arr.typeB - arr.typeA ≤ 7

/-- Calculates the total cost for a given rental arrangement --/
def total_cost (arr : RentalArrangement) : ℕ :=
  1600 * arr.typeA + 2400 * arr.typeB

/-- Theorem stating that the minimum rental cost is 36800 yuan --/
theorem min_rental_cost :
  ∃ (arr : RentalArrangement),
    is_valid_arrangement arr ∧
    total_cost arr = 36800 ∧
    ∀ (other : RentalArrangement),
      is_valid_arrangement other →
      total_cost other ≥ 36800 :=
sorry

end NUMINAMATH_CALUDE_min_rental_cost_l3655_365581


namespace NUMINAMATH_CALUDE_chessboard_divisibility_theorem_l3655_365574

/-- Represents a chessboard with natural numbers -/
def Chessboard := Matrix (Fin 8) (Fin 8) ℕ

/-- Represents an operation on the chessboard -/
inductive Operation
  | inc_3x3 (i j : Fin 6) : Operation
  | inc_4x4 (i j : Fin 5) : Operation

/-- Applies an operation to a chessboard -/
def apply_operation (board : Chessboard) (op : Operation) : Chessboard :=
  match op with
  | Operation.inc_3x3 i j => sorry
  | Operation.inc_4x4 i j => sorry

/-- Checks if all elements in the chessboard are divisible by 10 -/
def all_divisible_by_10 (board : Chessboard) : Prop :=
  ∀ i j, board i j % 10 = 0

/-- Main theorem: There exists a sequence of operations that makes all numbers divisible by 10 -/
theorem chessboard_divisibility_theorem (initial_board : Chessboard) :
  ∃ (ops : List Operation), all_divisible_by_10 (ops.foldl apply_operation initial_board) :=
sorry

end NUMINAMATH_CALUDE_chessboard_divisibility_theorem_l3655_365574


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3655_365532

theorem arithmetic_expression_evaluation : 6 + 15 / 3 - 4^2 + 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3655_365532


namespace NUMINAMATH_CALUDE_determine_x_with_gcd_queries_l3655_365578

theorem determine_x_with_gcd_queries :
  ∀ X : ℕ+, X ≤ 100 →
  ∃ (queries : Fin 7 → ℕ+ × ℕ+),
    (∀ i, (queries i).1 < 100 ∧ (queries i).2 < 100) ∧
    ∀ Y : ℕ+, Y ≤ 100 →
      (∀ i, Nat.gcd (X + (queries i).1) (queries i).2 = Nat.gcd (Y + (queries i).1) (queries i).2) →
      X = Y := by
  sorry

end NUMINAMATH_CALUDE_determine_x_with_gcd_queries_l3655_365578


namespace NUMINAMATH_CALUDE_smallest_p_satisfying_gcd_conditions_l3655_365540

theorem smallest_p_satisfying_gcd_conditions : 
  ∃ (p : ℕ), 
    p > 1500 ∧ 
    Nat.gcd 90 (p + 150) = 30 ∧ 
    Nat.gcd (p + 90) 150 = 75 ∧ 
    (∀ (q : ℕ), q > 1500 → Nat.gcd 90 (q + 150) = 30 → Nat.gcd (q + 90) 150 = 75 → p ≤ q) ∧
    p = 1560 :=
by sorry

end NUMINAMATH_CALUDE_smallest_p_satisfying_gcd_conditions_l3655_365540


namespace NUMINAMATH_CALUDE_president_and_committee_10_people_l3655_365591

/-- The number of ways to choose a president and a 3-person committee from a group of n people,
    where the president cannot be part of the committee -/
def choose_president_and_committee (n : ℕ) : ℕ :=
  n * (Nat.choose (n - 1) 3)

/-- Theorem stating that choosing a president and a 3-person committee from 10 people,
    where the president cannot be part of the committee, can be done in 840 ways -/
theorem president_and_committee_10_people :
  choose_president_and_committee 10 = 840 := by
  sorry

end NUMINAMATH_CALUDE_president_and_committee_10_people_l3655_365591


namespace NUMINAMATH_CALUDE_credit_card_more_beneficial_l3655_365588

/-- Represents the purchase amount in rubles -/
def purchase_amount : ℝ := 8000

/-- Represents the credit card cashback rate -/
def credit_cashback_rate : ℝ := 0.005

/-- Represents the debit card cashback rate -/
def debit_cashback_rate : ℝ := 0.0075

/-- Represents the monthly interest rate on the debit card -/
def debit_interest_rate : ℝ := 0.005

/-- Calculates the total income when using the credit card -/
def credit_card_income (amount : ℝ) : ℝ :=
  amount * credit_cashback_rate + amount * debit_interest_rate

/-- Calculates the total income when using the debit card -/
def debit_card_income (amount : ℝ) : ℝ :=
  amount * debit_cashback_rate

/-- Theorem stating that using the credit card is more beneficial -/
theorem credit_card_more_beneficial :
  credit_card_income purchase_amount > debit_card_income purchase_amount :=
sorry


end NUMINAMATH_CALUDE_credit_card_more_beneficial_l3655_365588


namespace NUMINAMATH_CALUDE_parabola_directrix_l3655_365549

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y = 3 * x^2 - 6 * x + 1

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop :=
  y = -25 / 12

/-- Theorem stating that the given directrix equation is correct for the parabola -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), parabola_equation x y → ∃ (d : ℝ), directrix_equation d ∧ 
  (d = y - 1 / (4 * 3) - (y - 1 / (4 * 3) - (-2 - 1 / (4 * 3)))) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3655_365549


namespace NUMINAMATH_CALUDE_special_triangle_angles_special_triangle_exists_l3655_365543

/-- A triangle with specific properties -/
structure SpecialTriangle where
  -- Angles of the triangle
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  -- Condition: The sum of angles in a triangle is 180°
  angle_sum : angleA + angleB + angleC = 180
  -- Condition: Angle C is a right angle
  right_angleC : angleC = 90
  -- Condition: Angle A is one-fourth of angle B
  angle_relation : angleA = angleB / 3

/-- Theorem stating the angles of the special triangle -/
theorem special_triangle_angles (t : SpecialTriangle) :
  t.angleA = 22.5 ∧ t.angleB = 67.5 ∧ t.angleC = 90 := by
  sorry

/-- The existence of such a triangle -/
theorem special_triangle_exists : ∃ t : SpecialTriangle, True := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_angles_special_triangle_exists_l3655_365543


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3655_365503

theorem simplify_polynomial (b : ℝ) : (1 : ℝ) * (3 * b) * (4 * b^2) * (5 * b^3) * (6 * b^4) = 360 * b^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3655_365503


namespace NUMINAMATH_CALUDE_coconut_grove_theorem_l3655_365502

theorem coconut_grove_theorem (x : ℝ) : 
  ((x + 4) * 60 + x * 120 + (x - 4) * 180) / (3 * x) = 100 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_theorem_l3655_365502


namespace NUMINAMATH_CALUDE_series_evaluation_l3655_365541

noncomputable def series_sum : ℝ := ∑' k, (k : ℝ) / (4 ^ k)

theorem series_evaluation : series_sum = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_series_evaluation_l3655_365541


namespace NUMINAMATH_CALUDE_textbook_order_cost_l3655_365530

/-- The total cost of ordering English and geography textbooks -/
def total_cost (english_count : ℕ) (geography_count : ℕ) (english_price : ℚ) (geography_price : ℚ) : ℚ :=
  english_count * english_price + geography_count * geography_price

/-- Theorem stating that the total cost of the textbook order is $630.00 -/
theorem textbook_order_cost :
  total_cost 35 35 (7.5 : ℚ) (10.5 : ℚ) = 630 := by
  sorry

end NUMINAMATH_CALUDE_textbook_order_cost_l3655_365530


namespace NUMINAMATH_CALUDE_min_value_of_f_l3655_365506

/-- The quadratic function f(x) = (x-1)^2 + 2 -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem: The minimum value of f(x) = (x-1)^2 + 2 is 2 -/
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ 2 ∧ ∃ x₀ : ℝ, f x₀ = 2 :=
sorry


end NUMINAMATH_CALUDE_min_value_of_f_l3655_365506


namespace NUMINAMATH_CALUDE_sparrows_among_non_pigeons_l3655_365547

theorem sparrows_among_non_pigeons (sparrows : ℝ) (pigeons : ℝ) (parrots : ℝ) (crows : ℝ)
  (h1 : sparrows = 0.4)
  (h2 : pigeons = 0.2)
  (h3 : parrots = 0.15)
  (h4 : crows = 0.25)
  (h5 : sparrows + pigeons + parrots + crows = 1) :
  sparrows / (1 - pigeons) = 0.5 := by
sorry

end NUMINAMATH_CALUDE_sparrows_among_non_pigeons_l3655_365547


namespace NUMINAMATH_CALUDE_mango_loss_percentage_l3655_365553

/-- Calculates the percentage of loss for a fruit seller selling mangoes. -/
theorem mango_loss_percentage 
  (loss_price : ℝ) 
  (profit_price : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : loss_price = 8)
  (h2 : profit_price = 10.5)
  (h3 : profit_percentage = 5) : 
  (loss_price - profit_price / (1 + profit_percentage / 100)) / (profit_price / (1 + profit_percentage / 100)) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_mango_loss_percentage_l3655_365553


namespace NUMINAMATH_CALUDE_line_parallel_plane_relationship_l3655_365548

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line) (α : Plane) : Prop := sorry

/-- Defines when a line is contained within a plane -/
def line_in_plane (a : Line) (α : Plane) : Prop := sorry

/-- Defines when two lines are parallel -/
def parallel_lines (l1 l2 : Line) : Prop := sorry

/-- Defines when two lines are skew -/
def skew_lines (l1 l2 : Line) : Prop := sorry

/-- Theorem: If a line is parallel to a plane, and another line is contained within that plane,
    then the two lines are either parallel or skew -/
theorem line_parallel_plane_relationship (l a : Line) (α : Plane) 
  (h1 : parallel_line_plane l α) (h2 : line_in_plane a α) :
  parallel_lines l a ∨ skew_lines l a := by sorry

end NUMINAMATH_CALUDE_line_parallel_plane_relationship_l3655_365548


namespace NUMINAMATH_CALUDE_specific_gold_cube_profit_l3655_365582

/-- Calculates the profit from selling a gold cube -/
def gold_cube_profit (side_length : ℝ) (density : ℝ) (buy_price : ℝ) (sell_multiplier : ℝ) : ℝ :=
  let volume := side_length ^ 3
  let mass := volume * density
  let cost := mass * buy_price
  let sell_price := cost * sell_multiplier
  sell_price - cost

/-- Theorem stating the profit for a specific gold cube -/
theorem specific_gold_cube_profit :
  gold_cube_profit 6 19 60 1.5 = 123120 := by
  sorry

end NUMINAMATH_CALUDE_specific_gold_cube_profit_l3655_365582


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3655_365585

theorem tangent_line_to_circle (x y : ℝ) : 
  (x^2 + y^2 - 4*x = 0) →  -- circle equation
  (x = 1 ∧ y = Real.sqrt 3) →  -- point of tangency
  (x - Real.sqrt 3 * y + 2 = 0)  -- equation of tangent line
:= by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3655_365585


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l3655_365519

theorem min_sum_reciprocals (n : ℕ+) (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b = 2) : 
  (1 / (1 + a ^ n.val)) + (1 / (1 + b ^ n.val)) ≥ 1 ∧ 
  ((1 / (1 + a ^ n.val)) + (1 / (1 + b ^ n.val)) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l3655_365519


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3655_365544

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0, b > 0, and eccentricity e = 2,
    the equation of its asymptotes is y = ±√3x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let asymptotes := fun (x y : ℝ) ↦ y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x
  e = 2 → (∀ x y, hyperbola x y ↔ asymptotes x y) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3655_365544


namespace NUMINAMATH_CALUDE_nth_monomial_form_l3655_365596

/-- A sequence of monomials is defined as follows:
    1st term: a
    2nd term: 3a²
    3rd term: 5a³
    4th term: 7a⁴
    5th term: 9a⁵
    ...
    This function represents the coefficient of the nth term in this sequence. -/
def monomial_coefficient (n : ℕ) : ℕ := 2 * n - 1

/-- This function represents the exponent of 'a' in the nth term of the sequence. -/
def monomial_exponent (n : ℕ) : ℕ := n

/-- This theorem states that the nth term of the sequence is (2n - 1)aⁿ -/
theorem nth_monomial_form (n : ℕ) (a : ℝ) :
  monomial_coefficient n * a ^ monomial_exponent n = (2 * n - 1) * a ^ n :=
sorry

end NUMINAMATH_CALUDE_nth_monomial_form_l3655_365596


namespace NUMINAMATH_CALUDE_rent_percentage_last_year_l3655_365539

theorem rent_percentage_last_year (E : ℝ) (P : ℝ) : 
  E > 0 → 
  (0.30 * (1.25 * E) = 1.875 * (P / 100) * E) → 
  P = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_rent_percentage_last_year_l3655_365539


namespace NUMINAMATH_CALUDE_polygon_sides_l3655_365570

theorem polygon_sides (n : ℕ) (x : ℝ) : 
  n ≥ 3 →
  0 < x →
  x < 180 →
  (n - 2) * 180 - x + (180 - x) = 500 →
  n = 4 ∨ n = 5 :=
sorry

end NUMINAMATH_CALUDE_polygon_sides_l3655_365570


namespace NUMINAMATH_CALUDE_total_cable_cost_neighborhood_cable_cost_l3655_365573

/-- The total cost of cable for a neighborhood with the given street configuration and cable requirements. -/
theorem total_cable_cost (ew_streets : ℕ) (ew_length : ℝ) (ns_streets : ℕ) (ns_length : ℝ) 
  (cable_per_mile : ℝ) (cost_per_mile : ℝ) : ℝ :=
  let total_street_length := ew_streets * ew_length + ns_streets * ns_length
  let total_cable_length := total_street_length * cable_per_mile
  total_cable_length * cost_per_mile

/-- The total cost of cable for the specific neighborhood described in the problem. -/
theorem neighborhood_cable_cost : total_cable_cost 18 2 10 4 5 2000 = 760000 := by
  sorry

end NUMINAMATH_CALUDE_total_cable_cost_neighborhood_cable_cost_l3655_365573


namespace NUMINAMATH_CALUDE_remainder_two_power_1000_mod_17_l3655_365542

theorem remainder_two_power_1000_mod_17 : 2^1000 % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_two_power_1000_mod_17_l3655_365542


namespace NUMINAMATH_CALUDE_sues_necklace_beads_l3655_365512

theorem sues_necklace_beads (purple : ℕ) (blue : ℕ) (green : ℕ) 
  (h1 : purple = 7)
  (h2 : blue = 2 * purple)
  (h3 : green = blue + 11) :
  purple + blue + green = 46 := by
  sorry

end NUMINAMATH_CALUDE_sues_necklace_beads_l3655_365512


namespace NUMINAMATH_CALUDE_perfect_square_factorization_l3655_365577

/-- Perfect square formula check -/
def isPerfectSquare (a b c : ℝ) : Prop :=
  ∃ (k : ℝ), a * c = (b / 2) ^ 2 ∧ a > 0

theorem perfect_square_factorization :
  ¬ isPerfectSquare 1 (1/4 : ℝ) (1/4) ∧
  ¬ isPerfectSquare 1 (2 : ℝ) (-1) ∧
  ¬ isPerfectSquare 1 (1 : ℝ) 1 ∧
  isPerfectSquare 4 (4 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_factorization_l3655_365577


namespace NUMINAMATH_CALUDE_lulu_ice_cream_expense_l3655_365583

theorem lulu_ice_cream_expense (initial_amount : ℝ) (ice_cream_cost : ℝ) (final_cash : ℝ) :
  initial_amount = 65 →
  final_cash = 24 →
  final_cash = (4/5) * (1/2) * (initial_amount - ice_cream_cost) →
  ice_cream_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_lulu_ice_cream_expense_l3655_365583


namespace NUMINAMATH_CALUDE_geometric_subsequence_k4_l3655_365510

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n : ℕ, a (n + 1) = a n + d

/-- A subsequence of an arithmetic sequence that forms a geometric sequence -/
structure GeometricSubsequence (as : ArithmeticSequence) where
  k : ℕ → ℕ
  q : ℝ
  h_geom : ∀ n : ℕ, as.a (k (n + 1)) = q * as.a (k n)
  h_k1 : k 1 ≠ 1
  h_k2 : k 2 ≠ 2
  h_k3 : k 3 ≠ 6

/-- The main theorem -/
theorem geometric_subsequence_k4 (as : ArithmeticSequence) (gs : GeometricSubsequence as) :
  gs.k 4 = 22 := by
  sorry

end NUMINAMATH_CALUDE_geometric_subsequence_k4_l3655_365510


namespace NUMINAMATH_CALUDE_park_area_change_l3655_365594

theorem park_area_change (original_area : ℝ) (length_decrease_percent : ℝ) (width_increase_percent : ℝ) :
  original_area = 600 →
  length_decrease_percent = 20 →
  width_increase_percent = 30 →
  let new_length_factor := 1 - length_decrease_percent / 100
  let new_width_factor := 1 + width_increase_percent / 100
  let new_area := original_area * new_length_factor * new_width_factor
  new_area = 624 := by sorry

end NUMINAMATH_CALUDE_park_area_change_l3655_365594


namespace NUMINAMATH_CALUDE_rectangles_with_one_gray_cell_l3655_365516

/-- The number of rectangles containing exactly one gray cell in a checkered rectangle -/
theorem rectangles_with_one_gray_cell 
  (total_gray_cells : ℕ) 
  (cells_with_four_rectangles : ℕ) 
  (cells_with_eight_rectangles : ℕ) 
  (h1 : total_gray_cells = 40)
  (h2 : cells_with_four_rectangles = 36)
  (h3 : cells_with_eight_rectangles = 4)
  (h4 : total_gray_cells = cells_with_four_rectangles + cells_with_eight_rectangles) :
  cells_with_four_rectangles * 4 + cells_with_eight_rectangles * 8 = 176 := by
sorry

end NUMINAMATH_CALUDE_rectangles_with_one_gray_cell_l3655_365516


namespace NUMINAMATH_CALUDE_sum_of_cubes_equation_l3655_365515

theorem sum_of_cubes_equation (x y : ℝ) : 
  x^3 + 21*x*y + y^3 = 343 → (x + y = 7 ∨ x + y = -14) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equation_l3655_365515


namespace NUMINAMATH_CALUDE_euro_equation_solution_l3655_365568

def euro (x y : ℝ) : ℝ := 2 * x * y

theorem euro_equation_solution (x : ℝ) : 
  euro 9 (euro 4 x) = 720 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_euro_equation_solution_l3655_365568


namespace NUMINAMATH_CALUDE_find_x_l3655_365590

-- Define the relationship between y, x, and a
def relationship (y x a k : ℝ) : Prop := y^4 * Real.sqrt x = k / a

-- Theorem statement
theorem find_x (k : ℝ) : 
  (∃ y x a, relationship y x a k ∧ y = 1 ∧ x = 16 ∧ a = 2) →
  (∀ y x a, relationship y x a k → y = 2 → a = 4 → x = 1/64) :=
by sorry

end NUMINAMATH_CALUDE_find_x_l3655_365590


namespace NUMINAMATH_CALUDE_comparison_of_b_and_c_l3655_365500

theorem comparison_of_b_and_c (a b c : ℝ) 
  (h1 : 2*a^3 - b^3 + 2*c^3 - 6*a^2*b + 3*a*b^2 - 3*a*c^2 - 3*b*c^2 + 6*a*b*c = 0)
  (h2 : a < b) : 
  b < c ∧ c < 2*b - a := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_b_and_c_l3655_365500


namespace NUMINAMATH_CALUDE_P_intersect_Q_equals_y_leq_2_l3655_365586

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x, y = -x + 2}

-- Theorem statement
theorem P_intersect_Q_equals_y_leq_2 : P ∩ Q = {y | y ≤ 2} := by sorry

end NUMINAMATH_CALUDE_P_intersect_Q_equals_y_leq_2_l3655_365586


namespace NUMINAMATH_CALUDE_bill_calculation_l3655_365537

def restaurant_bill (num_friends : ℕ) (extra_payment : ℕ) : Prop :=
  num_friends > 0 ∧ 
  ∃ (total_bill : ℕ), 
    total_bill = num_friends * (total_bill / num_friends + extra_payment * (num_friends - 1) / num_friends)

theorem bill_calculation :
  restaurant_bill 6 3 → ∃ (total_bill : ℕ), total_bill = 90 :=
by sorry

end NUMINAMATH_CALUDE_bill_calculation_l3655_365537


namespace NUMINAMATH_CALUDE_tens_digit_of_2035_pow_2037_minus_2039_l3655_365525

theorem tens_digit_of_2035_pow_2037_minus_2039 : ∃ n : ℕ, n < 10 ∧ n * 10 + 3 = (2035^2037 - 2039) % 100 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_2035_pow_2037_minus_2039_l3655_365525


namespace NUMINAMATH_CALUDE_election_votes_theorem_l3655_365554

theorem election_votes_theorem :
  ∀ (total_votes : ℕ) (valid_votes : ℕ) (candidate1_votes : ℕ) (candidate2_votes : ℕ),
    valid_votes = (80 * total_votes) / 100 →
    candidate1_votes = (55 * valid_votes) / 100 →
    candidate2_votes = 2700 →
    candidate1_votes + candidate2_votes = valid_votes →
    total_votes = 7500 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l3655_365554


namespace NUMINAMATH_CALUDE_dandelion_count_l3655_365504

/-- Proves the original number of yellow and white dandelions given the initial and final conditions --/
theorem dandelion_count : ∀ y w : ℕ,
  y + w = 35 →
  y - 2 = 2 * (w - 6) →
  y = 20 ∧ w = 15 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_count_l3655_365504


namespace NUMINAMATH_CALUDE_expression_value_l3655_365569

theorem expression_value : (2^200 + 5^201)^2 - (2^200 - 5^201)^2 = 20 * 10^201 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3655_365569


namespace NUMINAMATH_CALUDE_unique_two_digit_square_l3655_365536

theorem unique_two_digit_square : ∃! n : ℕ,
  10 ≤ n ∧ n < 100 ∧
  1000 ≤ n^2 ∧ n^2 < 10000 ∧
  (∃ a b : ℕ, n^2 = 1100 * a + 11 * b ∧ 0 < a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10) ∧
  n = 88 := by
sorry

end NUMINAMATH_CALUDE_unique_two_digit_square_l3655_365536


namespace NUMINAMATH_CALUDE_population_in_scientific_notation_l3655_365567

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem population_in_scientific_notation :
  let population : ℝ := 4.6e9
  toScientificNotation population = ScientificNotation.mk 4.6 9 := by
  sorry

end NUMINAMATH_CALUDE_population_in_scientific_notation_l3655_365567


namespace NUMINAMATH_CALUDE_quarters_remaining_l3655_365599

theorem quarters_remaining (initial_quarters : ℕ) (payment_dollars : ℕ) (quarters_per_dollar : ℕ) : 
  initial_quarters = 160 →
  payment_dollars = 35 →
  quarters_per_dollar = 4 →
  initial_quarters - (payment_dollars * quarters_per_dollar) = 20 :=
by sorry

end NUMINAMATH_CALUDE_quarters_remaining_l3655_365599


namespace NUMINAMATH_CALUDE_even_increasing_function_property_l3655_365531

/-- A function that is even on ℝ and increasing on (-∞, 0] -/
def EvenIncreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y)

/-- Theorem stating that for an even function increasing on (-∞, 0],
    if f(a) ≤ f(2-a), then a ≥ 1 -/
theorem even_increasing_function_property (f : ℝ → ℝ) (a : ℝ) 
    (h1 : EvenIncreasingFunction f) (h2 : f a ≤ f (2 - a)) : 
    a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_even_increasing_function_property_l3655_365531


namespace NUMINAMATH_CALUDE_white_marbles_added_l3655_365533

/-- Proves that the number of white marbles added to a bag is 4, given the initial marble counts and the resulting probability of drawing a black or gold marble. -/
theorem white_marbles_added (black gold purple red : ℕ) 
  (h_black : black = 3)
  (h_gold : gold = 6)
  (h_purple : purple = 2)
  (h_red : red = 6)
  (h_prob : (black + gold : ℚ) / (black + gold + purple + red + w) = 3 / 7)
  : w = 4 := by
  sorry

end NUMINAMATH_CALUDE_white_marbles_added_l3655_365533


namespace NUMINAMATH_CALUDE_probability_theorem_l3655_365501

def total_balls : ℕ := 6
def new_balls : ℕ := 4
def old_balls : ℕ := 2

def probability_one_new_one_old : ℚ :=
  (new_balls * old_balls) / (total_balls * (total_balls - 1) / 2)

theorem probability_theorem :
  probability_one_new_one_old = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l3655_365501


namespace NUMINAMATH_CALUDE_water_left_after_experiment_l3655_365563

-- Define the initial amount of water
def initial_water : ℚ := 2

-- Define the amount of water used in the experiment
def water_used : ℚ := 7/6

-- Theorem to prove
theorem water_left_after_experiment :
  initial_water - water_used = 5/6 := by sorry

end NUMINAMATH_CALUDE_water_left_after_experiment_l3655_365563


namespace NUMINAMATH_CALUDE_unique_quadratic_root_l3655_365595

theorem unique_quadratic_root (m : ℝ) : 
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) → (m = 0 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_root_l3655_365595


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_quadratic_range_l3655_365552

theorem greatest_integer_b_for_quadratic_range (b : ℤ) : 
  (∀ x : ℝ, x^2 + b*x + 15 ≠ -6) ↔ b ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_quadratic_range_l3655_365552


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3655_365534

theorem inequality_solution_set (a : ℝ) (h : 2*a + 1 < 0) :
  {x : ℝ | x^2 - 4*a*x - 5*a^2 > 0} = {x : ℝ | x < 5*a ∨ x > -a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3655_365534


namespace NUMINAMATH_CALUDE_algebra_test_female_students_l3655_365575

theorem algebra_test_female_students 
  (total_average : ℝ) 
  (male_count : ℕ) 
  (male_average : ℝ) 
  (female_average : ℝ) 
  (h1 : total_average = 90) 
  (h2 : male_count = 8) 
  (h3 : male_average = 83) 
  (h4 : female_average = 92) : 
  ∃ (female_count : ℕ), 
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧ 
    female_count = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_algebra_test_female_students_l3655_365575


namespace NUMINAMATH_CALUDE_player_a_winning_strategy_l3655_365551

/-- Represents a player in the game -/
inductive Player
| A
| B

/-- Represents a cubic polynomial ax^3 + bx^2 + cx + d -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_nonzero : a ≠ 0

/-- Represents the state of the game -/
structure GameState where
  polynomial : CubicPolynomial
  current_player : Player
  moves_left : Nat

/-- Represents a move in the game -/
structure Move where
  value : ℤ
  position : Nat

/-- Function to apply a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- Predicate to check if a polynomial has three distinct integer roots -/
def has_three_distinct_integer_roots (p : CubicPolynomial) : Prop :=
  sorry

/-- Theorem stating that Player A has a winning strategy -/
theorem player_a_winning_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (initial_state : GameState),
      initial_state.current_player = Player.A →
      initial_state.moves_left = 3 →
      ∀ (b_moves : Fin 2 → Move),
        let final_state := apply_move (apply_move (apply_move initial_state (strategy initial_state)) (b_moves 0)) (b_moves 1)
        has_three_distinct_integer_roots final_state.polynomial :=
  sorry

end NUMINAMATH_CALUDE_player_a_winning_strategy_l3655_365551


namespace NUMINAMATH_CALUDE_walkway_problem_l3655_365565

/-- Represents the walkway scenario -/
structure Walkway where
  length : ℝ
  time_with : ℝ
  time_against : ℝ

/-- Calculates the time to walk when the walkway is not moving -/
noncomputable def time_stationary (w : Walkway) : ℝ :=
  w.length * 2 * w.time_with * w.time_against / (w.time_against + w.time_with) / w.time_with

/-- Theorem statement for the walkway problem -/
theorem walkway_problem (w : Walkway) 
  (h1 : w.length = 100)
  (h2 : w.time_with = 25)
  (h3 : w.time_against = 150) :
  abs (time_stationary w - 300 / 7) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_walkway_problem_l3655_365565


namespace NUMINAMATH_CALUDE_monday_temperature_l3655_365579

def sunday_temp : ℝ := 40
def tuesday_temp : ℝ := 65
def wednesday_temp : ℝ := 36
def thursday_temp : ℝ := 82
def friday_temp : ℝ := 72
def saturday_temp : ℝ := 26
def average_temp : ℝ := 53
def days_in_week : ℕ := 7

theorem monday_temperature (monday_temp : ℝ) :
  (sunday_temp + monday_temp + tuesday_temp + wednesday_temp + thursday_temp + friday_temp + saturday_temp) / days_in_week = average_temp →
  monday_temp = 50 := by
sorry

end NUMINAMATH_CALUDE_monday_temperature_l3655_365579


namespace NUMINAMATH_CALUDE_sin_negative_four_thirds_pi_l3655_365522

theorem sin_negative_four_thirds_pi : 
  Real.sin (-(4/3) * Real.pi) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_four_thirds_pi_l3655_365522


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3655_365505

theorem sum_of_cubes (x y z : ℝ) 
  (sum_eq : x + y + z = 2) 
  (sum_prod_eq : x*y + y*z + z*x = -3) 
  (prod_eq : x*y*z = 2) : 
  x^3 + y^3 + z^3 = 32 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3655_365505


namespace NUMINAMATH_CALUDE_new_edition_has_450_pages_l3655_365576

/-- The number of pages in the old edition of the Geometry book. -/
def old_edition_pages : ℕ := 340

/-- The difference in pages between twice the old edition and the new edition. -/
def page_difference : ℕ := 230

/-- The number of pages in the new edition of the Geometry book. -/
def new_edition_pages : ℕ := 2 * old_edition_pages - page_difference

/-- Theorem stating that the new edition of the Geometry book has 450 pages. -/
theorem new_edition_has_450_pages : new_edition_pages = 450 := by
  sorry

end NUMINAMATH_CALUDE_new_edition_has_450_pages_l3655_365576


namespace NUMINAMATH_CALUDE_lunch_break_duration_l3655_365557

/-- Represents the painting rate of an individual or group in terms of house percentage per hour -/
structure PaintingRate where
  rate : ℝ
  (nonneg : rate ≥ 0)

/-- Represents the duration of work in hours -/
def workDuration (startTime endTime : ℝ) : ℝ := endTime - startTime

/-- Represents the percentage of house painted given a painting rate and work duration -/
def percentPainted (r : PaintingRate) (duration : ℝ) : ℝ := r.rate * duration

theorem lunch_break_duration (paula : PaintingRate) (helpers : PaintingRate) 
  (lunchBreak : ℝ) : 
  -- Monday's condition
  percentPainted (PaintingRate.mk (paula.rate + helpers.rate) (by sorry)) (workDuration 8 16 - lunchBreak) = 0.5 →
  -- Tuesday's condition
  percentPainted helpers (workDuration 8 14.2 - lunchBreak) = 0.24 →
  -- Wednesday's condition
  percentPainted paula (workDuration 8 19.2 - lunchBreak) = 0.26 →
  -- Conclusion
  lunchBreak * 60 = 48 := by sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l3655_365557


namespace NUMINAMATH_CALUDE_max_value_expression_l3655_365560

theorem max_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 ∧
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c)^2 / (a^2 + b^2 + c^2) = 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3655_365560


namespace NUMINAMATH_CALUDE_jack_books_theorem_l3655_365529

/-- Calculates the average number of pages per book given the total thickness,
    pages per inch, and number of books. -/
def average_pages_per_book (total_thickness : ℕ) (pages_per_inch : ℕ) (num_books : ℕ) : ℚ :=
  (total_thickness * pages_per_inch : ℚ) / num_books

/-- Theorem stating that for a stack of books 12 inches thick,
    with 80 pages per inch and 6 books in total,
    the average number of pages per book is 160. -/
theorem jack_books_theorem :
  average_pages_per_book 12 80 6 = 160 := by
  sorry

end NUMINAMATH_CALUDE_jack_books_theorem_l3655_365529


namespace NUMINAMATH_CALUDE_spinner_probabilities_l3655_365528

theorem spinner_probabilities : ∃ (p_C p_D : ℚ), 
  p_C ≥ 0 ∧ p_D ≥ 0 ∧ (1 : ℚ)/4 + (1 : ℚ)/3 + p_C + p_D = 1 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probabilities_l3655_365528


namespace NUMINAMATH_CALUDE_no_quadratic_trinomials_satisfying_equation_l3655_365507

/-- A quadratic trinomial -/
structure QuadraticTrinomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluate a quadratic trinomial at a given value -/
def QuadraticTrinomial.eval (p : QuadraticTrinomial) (x : ℤ) : ℤ :=
  p.a * x^2 + p.b * x + p.c

/-- Statement: There do not exist quadratic trinomials P, Q, R such that
    for all integers x and y, there exists an integer z satisfying P(x) + Q(y) = R(z) -/
theorem no_quadratic_trinomials_satisfying_equation :
  ¬∃ (P Q R : QuadraticTrinomial), ∀ (x y : ℤ), ∃ (z : ℤ),
    P.eval x + Q.eval y = R.eval z := by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_trinomials_satisfying_equation_l3655_365507


namespace NUMINAMATH_CALUDE_prob_even_sum_two_balls_l3655_365550

def num_balls : ℕ := 20

def is_even (n : ℕ) : Prop := n % 2 = 0

theorem prob_even_sum_two_balls :
  let total_outcomes := num_balls * (num_balls - 1)
  let favorable_outcomes := (num_balls / 2) * (num_balls / 2 - 1) * 2
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 19 := by sorry

end NUMINAMATH_CALUDE_prob_even_sum_two_balls_l3655_365550


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3655_365546

theorem right_triangle_hypotenuse (m1 m2 : ℝ) (h_m1 : m1 = 6) (h_m2 : m2 = Real.sqrt 50) :
  ∃ a b h : ℝ,
    a > 0 ∧ b > 0 ∧
    m1^2 = a^2 + (b/2)^2 ∧
    m2^2 = b^2 + (a/2)^2 ∧
    h^2 = (2*a)^2 + (2*b)^2 ∧
    h = Real.sqrt 275.2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3655_365546
