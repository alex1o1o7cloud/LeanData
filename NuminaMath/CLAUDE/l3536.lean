import Mathlib

namespace NUMINAMATH_CALUDE_abs_opposite_equal_l3536_353618

theorem abs_opposite_equal (x : ℝ) : |x| = |-x| := by sorry

end NUMINAMATH_CALUDE_abs_opposite_equal_l3536_353618


namespace NUMINAMATH_CALUDE_park_tree_removal_l3536_353621

/-- The number of trees removed from a park -/
def trees_removed (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem: Given 6 initial trees and 2 remaining trees, 4 trees are removed -/
theorem park_tree_removal :
  trees_removed 6 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_park_tree_removal_l3536_353621


namespace NUMINAMATH_CALUDE_sin_4x_eq_sin_2x_solutions_l3536_353622

open Set
open Real

theorem sin_4x_eq_sin_2x_solutions :
  let S := {x : ℝ | 0 < x ∧ x < (3/2)*π ∧ sin (4*x) = sin (2*x)}
  S = {π/6, π/2, π, 5*π/6, 7*π/6} := by sorry

end NUMINAMATH_CALUDE_sin_4x_eq_sin_2x_solutions_l3536_353622


namespace NUMINAMATH_CALUDE_sum_le_product_plus_two_l3536_353654

theorem sum_le_product_plus_two (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  x + y + z ≤ x*y*z + 2 := by sorry

end NUMINAMATH_CALUDE_sum_le_product_plus_two_l3536_353654


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l3536_353683

theorem geometric_progression_ratio (a b c d : ℝ) : 
  0 < a → a < b → b < c → c < d → d = 2*a →
  (d - a) * (a^2 / (b - a) + b^2 / (c - b) + c^2 / (d - c)) = (a + b + c)^2 →
  b * c * d / a^3 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l3536_353683


namespace NUMINAMATH_CALUDE_cubic_factorization_l3536_353672

theorem cubic_factorization (y : ℝ) : y^3 - 4*y^2 + 4*y = y*(y-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3536_353672


namespace NUMINAMATH_CALUDE_special_sequence_a10_l3536_353625

/-- A sequence of positive real numbers satisfying aₚ₊ₖ = aₚ · aₖ for all positive integers p and q -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ p q : ℕ, p > 0 → q > 0 → a (p + q) = a p * a q)

theorem special_sequence_a10 (a : ℕ → ℝ) (h : SpecialSequence a) (h8 : a 8 = 16) : 
  a 10 = 32 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_a10_l3536_353625


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_true_l3536_353647

theorem quadratic_inequality_always_true (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0) ↔ -1 < m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_true_l3536_353647


namespace NUMINAMATH_CALUDE_brians_books_l3536_353623

theorem brians_books (x : ℕ) : 
  x + 2 * 15 + (x + 2 * 15) / 2 = 75 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_brians_books_l3536_353623


namespace NUMINAMATH_CALUDE_polynomial_condition_implies_linear_l3536_353644

/-- A polynomial with real coefficients -/
def RealPolynomial : Type := ℝ → ℝ

/-- The condition that P(x + y) is rational when P(x) and P(y) are rational -/
def SatisfiesCondition (P : RealPolynomial) : Prop :=
  ∀ x y : ℝ, (∃ q₁ q₂ : ℚ, P x = q₁ ∧ P y = q₂) → ∃ q : ℚ, P (x + y) = q

/-- The theorem stating that polynomials satisfying the condition must be linear with rational coefficients -/
theorem polynomial_condition_implies_linear
  (P : RealPolynomial)
  (h : SatisfiesCondition P) :
  ∃ a b : ℚ, ∀ x : ℝ, P x = a * x + b :=
sorry

end NUMINAMATH_CALUDE_polynomial_condition_implies_linear_l3536_353644


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3536_353693

theorem geometric_sequence_sum (n : ℕ) :
  let a : ℝ := 1
  let r : ℝ := 1/2
  let sum : ℝ := a * (1 - r^n) / (1 - r)
  sum = 31/16 → n = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3536_353693


namespace NUMINAMATH_CALUDE_hexagon_coins_proof_l3536_353662

/-- The number of coins needed to construct a hexagon with side length n -/
def hexagon_coins (n : ℕ) : ℕ := 3 * n * (n - 1) + 1

theorem hexagon_coins_proof :
  (hexagon_coins 2 = 7) ∧
  (hexagon_coins 3 = 19) ∧
  (hexagon_coins 10 = 271) :=
by sorry

end NUMINAMATH_CALUDE_hexagon_coins_proof_l3536_353662


namespace NUMINAMATH_CALUDE_percentage_60to69_is_20_percent_l3536_353627

/-- Represents the score ranges in the class --/
inductive ScoreRange
  | Below60
  | Range60to69
  | Range70to79
  | Range80to89
  | Range90to100

/-- The frequency of students for each score range --/
def frequency (range : ScoreRange) : Nat :=
  match range with
  | .Below60 => 2
  | .Range60to69 => 5
  | .Range70to79 => 6
  | .Range80to89 => 8
  | .Range90to100 => 4

/-- The total number of students in the class --/
def totalStudents : Nat :=
  frequency ScoreRange.Below60 +
  frequency ScoreRange.Range60to69 +
  frequency ScoreRange.Range70to79 +
  frequency ScoreRange.Range80to89 +
  frequency ScoreRange.Range90to100

/-- The percentage of students in the 60%-69% range --/
def percentageIn60to69Range : Rat :=
  (frequency ScoreRange.Range60to69 : Rat) / (totalStudents : Rat) * 100

theorem percentage_60to69_is_20_percent :
  percentageIn60to69Range = 20 := by
  sorry

#eval percentageIn60to69Range

end NUMINAMATH_CALUDE_percentage_60to69_is_20_percent_l3536_353627


namespace NUMINAMATH_CALUDE_total_balloons_l3536_353608

theorem total_balloons (tom_balloons sara_balloons : ℕ) 
  (h1 : tom_balloons = 9) 
  (h2 : sara_balloons = 8) : 
  tom_balloons + sara_balloons = 17 := by
sorry

end NUMINAMATH_CALUDE_total_balloons_l3536_353608


namespace NUMINAMATH_CALUDE_bottles_used_second_game_l3536_353675

theorem bottles_used_second_game :
  let total_bottles : ℕ := 10 * 20
  let bottles_used_first_game : ℕ := 70
  let bottles_left_after_second_game : ℕ := 20
  let bottles_used_second_game : ℕ := total_bottles - bottles_used_first_game - bottles_left_after_second_game
  bottles_used_second_game = 110 := by sorry

end NUMINAMATH_CALUDE_bottles_used_second_game_l3536_353675


namespace NUMINAMATH_CALUDE_inequality_solution_l3536_353674

theorem inequality_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (x / (x - 1) + (x + 3) / (2 * x) ≥ 4) ↔ (0 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3536_353674


namespace NUMINAMATH_CALUDE_four_composition_odd_l3536_353657

-- Define a type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem four_composition_odd (f : RealFunction) (h : IsOdd f) :
  IsOdd (fun x ↦ f (f (f (f x)))) := by
  sorry

end NUMINAMATH_CALUDE_four_composition_odd_l3536_353657


namespace NUMINAMATH_CALUDE_businessmen_beverage_problem_l3536_353694

theorem businessmen_beverage_problem (total : ℕ) (coffee tea soda : ℕ) 
  (coffee_tea coffee_soda tea_soda : ℕ) (all_three : ℕ) 
  (h_total : total = 30)
  (h_coffee : coffee = 15)
  (h_tea : tea = 13)
  (h_soda : soda = 8)
  (h_coffee_tea : coffee_tea = 6)
  (h_coffee_soda : coffee_soda = 2)
  (h_tea_soda : tea_soda = 3)
  (h_all_three : all_three = 1) : 
  total - (coffee + tea + soda - coffee_tea - coffee_soda - tea_soda + all_three) = 4 := by
sorry

end NUMINAMATH_CALUDE_businessmen_beverage_problem_l3536_353694


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3536_353666

def i : ℂ := Complex.I

theorem complex_equation_solution (x : ℝ) (h : (1 - 2*i) * (x + i) = 4 - 3*i) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3536_353666


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3536_353630

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, (1/2 : ℝ) * x^2 - 2*m*x + 4*m + 1 = 0 ∧ 
   ∀ y : ℝ, (1/2 : ℝ) * y^2 - 2*m*y + 4*m + 1 = 0 → y = x) → 
  m^2 - 2*m = 1/2 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3536_353630


namespace NUMINAMATH_CALUDE_least_N_mod_1000_l3536_353624

/-- Sum of digits in base-five representation -/
def f (n : ℕ) : ℕ := sorry

/-- Sum of digits in base-seven representation -/
def g (n : ℕ) : ℕ := sorry

/-- The least value of n such that g(n) ≥ 10 -/
def N : ℕ := sorry

theorem least_N_mod_1000 : N % 1000 = 781 := by sorry

end NUMINAMATH_CALUDE_least_N_mod_1000_l3536_353624


namespace NUMINAMATH_CALUDE_jens_age_difference_l3536_353680

/-- Proves that the difference between 3 times Jen's son's current age and Jen's current age is 7 years -/
theorem jens_age_difference (jen_age_at_birth : ℕ) (son_current_age : ℕ) (jen_current_age : ℕ) : 
  jen_age_at_birth = 25 →
  son_current_age = 16 →
  jen_current_age = 41 →
  3 * son_current_age - jen_current_age = 7 := by
  sorry

end NUMINAMATH_CALUDE_jens_age_difference_l3536_353680


namespace NUMINAMATH_CALUDE_ninety_eight_squared_l3536_353656

theorem ninety_eight_squared : 98 * 98 = 9604 := by
  sorry

end NUMINAMATH_CALUDE_ninety_eight_squared_l3536_353656


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3536_353681

/-- Triangle ABC with given properties -/
structure TriangleABC where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  side_angle_correspondence : True  -- Sides a, b, c are opposite to angles A, B, C respectively
  cosine_relation : b * (1 - 2 * Real.cos A) = 2 * a * Real.cos B
  a_value : a = 1
  tan_A_value : Real.tan A = 2 * Real.sqrt 2

/-- Main theorem about the properties of TriangleABC -/
theorem triangle_abc_properties (t : TriangleABC) :
  t.b = 2 * t.c ∧ 
  (1/2 * t.b * t.c * Real.sin t.A : ℝ) = 2 * Real.sqrt 2 / 11 :=
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3536_353681


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3536_353603

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x + 2*y - 3 = 0
def l₂ (a x y : ℝ) : Prop := 2*x - a*y + 3 = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, l₁ x y → l₂ a x y → x = x

-- Theorem statement
theorem parallel_lines_a_value :
  ∀ a : ℝ, parallel a → a = -4 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3536_353603


namespace NUMINAMATH_CALUDE_boys_left_bakery_l3536_353600

/-- The number of boys who left the bakery --/
def boys_who_left (initial_children : ℕ) (girls_came_in : ℕ) (final_children : ℕ) : ℕ :=
  initial_children + girls_came_in - final_children

theorem boys_left_bakery (initial_children : ℕ) (girls_came_in : ℕ) (final_children : ℕ)
  (h1 : initial_children = 85)
  (h2 : girls_came_in = 24)
  (h3 : final_children = 78) :
  boys_who_left initial_children girls_came_in final_children = 31 := by
  sorry

#eval boys_who_left 85 24 78

end NUMINAMATH_CALUDE_boys_left_bakery_l3536_353600


namespace NUMINAMATH_CALUDE_factorization_equality_l3536_353669

theorem factorization_equality (x y : ℝ) : x + x^2 - y - y^2 = (x + y + 1) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3536_353669


namespace NUMINAMATH_CALUDE_line_equation_and_intersection_l3536_353651

/-- The slope of the first line -/
def m : ℚ := 3 / 4

/-- The y-intercept of the first line -/
def b : ℚ := 3 / 2

/-- The slope of the second line -/
def m' : ℚ := -1

/-- The y-intercept of the second line -/
def b' : ℚ := 7

/-- The x-coordinate of the intersection point -/
def x_intersect : ℚ := 11 / 7

/-- The y-coordinate of the intersection point -/
def y_intersect : ℚ := 25 / 7

theorem line_equation_and_intersection :
  (∀ x y : ℚ, 3 * (x - 2) + (-4) * (y - 3) = 0 ↔ y = m * x + b) ∧
  (m * x_intersect + b = m' * x_intersect + b') ∧
  (y_intersect = m * x_intersect + b) ∧
  (y_intersect = m' * x_intersect + b') := by
  sorry

end NUMINAMATH_CALUDE_line_equation_and_intersection_l3536_353651


namespace NUMINAMATH_CALUDE_factorization_x4_3x2_1_l3536_353677

theorem factorization_x4_3x2_1 (x : ℝ) :
  x^4 - 3*x^2 + 1 = (x^2 + x - 1) * (x^2 - x - 1) := by sorry

end NUMINAMATH_CALUDE_factorization_x4_3x2_1_l3536_353677


namespace NUMINAMATH_CALUDE_initial_girls_count_l3536_353663

theorem initial_girls_count (total : ℕ) (initial_girls : ℕ) : 
  (initial_girls : ℚ) / total = 35 / 100 →
  ((initial_girls : ℚ) - 3) / (total : ℚ) = 25 / 100 →
  initial_girls = 11 := by
  sorry

end NUMINAMATH_CALUDE_initial_girls_count_l3536_353663


namespace NUMINAMATH_CALUDE_find_divisor_l3536_353691

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 16698 →
  quotient = 89 →
  remainder = 14 →
  divisor * quotient + remainder = dividend →
  divisor = 187 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l3536_353691


namespace NUMINAMATH_CALUDE_perfect_square_fraction_count_l3536_353668

theorem perfect_square_fraction_count : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, n ≠ 20 ∧ ∃ k : ℤ, (n : ℚ) / (20 - n) = k^2) ∧ 
    Finset.card S = 4 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_fraction_count_l3536_353668


namespace NUMINAMATH_CALUDE_new_to_original_detergent_water_ratio_l3536_353658

/-- Represents the ratio of bleach, detergent, and water in a solution -/
structure SolutionRatio where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- The original solution ratio -/
def originalRatio : SolutionRatio :=
  { bleach := 2, detergent := 40, water := 100 }

/-- The new amount of water in liters -/
def newWaterAmount : ℚ := 300

/-- The new amount of detergent in liters -/
def newDetergentAmount : ℚ := 60

/-- The factor by which the bleach to detergent ratio is increased -/
def bleachDetergentIncreaseFactor : ℚ := 3

theorem new_to_original_detergent_water_ratio :
  (newDetergentAmount / (originalRatio.water * newWaterAmount / originalRatio.water)) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_new_to_original_detergent_water_ratio_l3536_353658


namespace NUMINAMATH_CALUDE_negation_existential_geq_zero_l3536_353652

theorem negation_existential_geq_zero :
  ¬(∃ x : ℝ, x + 1 ≥ 0) ↔ ∀ x : ℝ, x + 1 < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_existential_geq_zero_l3536_353652


namespace NUMINAMATH_CALUDE_first_candidate_percentage_l3536_353655

/-- Given an election between two candidates where the total number of votes is 600
    and the second candidate received 240 votes, prove that the first candidate
    received 60% of the votes. -/
theorem first_candidate_percentage (total_votes : ℕ) (second_candidate_votes : ℕ)
  (h1 : total_votes = 600)
  (h2 : second_candidate_votes = 240) :
  (total_votes - second_candidate_votes : ℚ) / total_votes * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_first_candidate_percentage_l3536_353655


namespace NUMINAMATH_CALUDE_triangle_side_values_l3536_353684

theorem triangle_side_values (A B C : Real) (a b c : Real) :
  c = Real.sqrt 3 →
  C = π / 3 →
  Real.sin B = 2 * Real.sin A →
  a ^ 2 + b ^ 2 - a * b = 3 →
  (a = 1 ∧ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_values_l3536_353684


namespace NUMINAMATH_CALUDE_complex_ratio_max_value_l3536_353646

theorem complex_ratio_max_value (z : ℂ) (h : Complex.abs z = 2) :
  (Complex.abs (z^2 - z + 1)) / (Complex.abs (2*z - 1 - Complex.I * Real.sqrt 3)) ≤ 3/2 ∧
  ∃ w : ℂ, Complex.abs w = 2 ∧
    (Complex.abs (w^2 - w + 1)) / (Complex.abs (2*w - 1 - Complex.I * Real.sqrt 3)) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_complex_ratio_max_value_l3536_353646


namespace NUMINAMATH_CALUDE_contrapositive_quadratic_inequality_l3536_353639

theorem contrapositive_quadratic_inequality :
  (∀ x : ℝ, x^2 + x - 6 > 0 → x < -3 ∨ x > 2) ↔
  (∀ x : ℝ, x ≥ -3 ∧ x ≤ 2 → x^2 + x - 6 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_quadratic_inequality_l3536_353639


namespace NUMINAMATH_CALUDE_c_7_equals_448_l3536_353633

def a (n : ℕ) : ℕ := n

def b (n : ℕ) : ℕ := 2^(n-1)

def c (n : ℕ) : ℕ := a n * b n

theorem c_7_equals_448 : c 7 = 448 := by
  sorry

end NUMINAMATH_CALUDE_c_7_equals_448_l3536_353633


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3536_353638

theorem polynomial_divisibility (k l m n : ℕ) : 
  ∃ q : Polynomial ℤ, (X^4*k + X^(4*l+1) + X^(4*m+2) + X^(4*n+3)) = (X^3 + X^2 + X + 1) * q := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3536_353638


namespace NUMINAMATH_CALUDE_lion_king_earnings_l3536_353642

/-- Represents movie financial data in millions of dollars -/
structure MovieData where
  productionCost : ℝ
  boxOfficeEarnings : ℝ
  profit : ℝ

/-- The Lion King's box office earnings -/
def lionKingEarnings : ℝ := 200

theorem lion_king_earnings (starWars lionKing : MovieData) :
  starWars.productionCost = 25 →
  starWars.boxOfficeEarnings = 405 →
  lionKing.productionCost = 10 →
  lionKing.profit = (starWars.boxOfficeEarnings - starWars.productionCost) / 2 →
  lionKing.boxOfficeEarnings = lionKingEarnings := by
  sorry

end NUMINAMATH_CALUDE_lion_king_earnings_l3536_353642


namespace NUMINAMATH_CALUDE_third_term_value_l3536_353648

/-- An arithmetic sequence with five terms -/
structure ArithmeticSequence :=
  (a : ℝ)  -- First term
  (d : ℝ)  -- Common difference

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithmeticSequence) : ℝ := seq.a + 2 * seq.d

theorem third_term_value :
  ∀ seq : ArithmeticSequence,
  seq.a = 12 ∧ seq.a + 4 * seq.d = 32 →
  third_term seq = 22 :=
by sorry

end NUMINAMATH_CALUDE_third_term_value_l3536_353648


namespace NUMINAMATH_CALUDE_only_solutions_l3536_353616

/-- A four-digit number is composed of two two-digit numbers x and y -/
def is_valid_four_digit (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x ≥ 10 ∧ x < 100 ∧ y ≥ 10 ∧ y < 100 ∧ n = 100 * x + y

/-- The condition that the square of the sum of x and y equals the four-digit number -/
def satisfies_condition (n : ℕ) : Prop :=
  ∃ (x y : ℕ), is_valid_four_digit n ∧ (x + y)^2 = n

/-- The theorem stating that 3025 and 2025 are the only solutions -/
theorem only_solutions : ∀ (n : ℕ), satisfies_condition n ↔ (n = 3025 ∨ n = 2025) :=
sorry

end NUMINAMATH_CALUDE_only_solutions_l3536_353616


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l3536_353631

theorem smallest_solution_quadratic (x : ℝ) :
  (6 * x^2 - 29 * x + 35 = 0) → (x ≥ 7/3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l3536_353631


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3536_353667

theorem linear_equation_solution (x y : ℝ) : 
  3 * x - y = 5 → y = 3 * x - 5 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3536_353667


namespace NUMINAMATH_CALUDE_garden_roller_diameter_l3536_353687

/-- The diameter of a garden roller given its length, area covered, and number of revolutions. -/
theorem garden_roller_diameter
  (length : ℝ)
  (area_covered : ℝ)
  (revolutions : ℕ)
  (h1 : length = 2)
  (h2 : area_covered = 52.8)
  (h3 : revolutions = 6)
  (h4 : Real.pi = 22 / 7) :
  ∃ (diameter : ℝ), diameter = 1.4 ∧ 
    area_covered = revolutions * Real.pi * diameter * length :=
by sorry

end NUMINAMATH_CALUDE_garden_roller_diameter_l3536_353687


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l3536_353698

theorem intersection_point_k_value :
  ∀ (x y k : ℝ),
  (x = -7.5) →
  (-3 * x + y = k) →
  (0.3 * x + y = 12) →
  (k = 36.75) := by
sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l3536_353698


namespace NUMINAMATH_CALUDE_division_of_decimals_l3536_353637

theorem division_of_decimals : (0.25 : ℚ) / (0.005 : ℚ) = 50 := by
  sorry

end NUMINAMATH_CALUDE_division_of_decimals_l3536_353637


namespace NUMINAMATH_CALUDE_three_numbers_problem_l3536_353682

theorem three_numbers_problem :
  let x : ℚ := 1/9
  let y : ℚ := 1/6
  let z : ℚ := 1/3
  (x + y + z = 11/18) ∧
  (1/x + 1/y + 1/z = 18) ∧
  (2 * (1/y) = 1/x + 1/z) :=
by sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l3536_353682


namespace NUMINAMATH_CALUDE_min_abs_z_complex_l3536_353620

/-- Given a complex number z satisfying |z - 5i| + |z - 3| = 7, 
    the minimum value of |z| is 15/7 -/
theorem min_abs_z_complex (z : ℂ) (h : Complex.abs (z - 5*Complex.I) + Complex.abs (z - 3) = 7) :
  ∃ (w : ℂ), Complex.abs w = 15/7 ∧ ∀ (v : ℂ), Complex.abs (v - 5*Complex.I) + Complex.abs (v - 3) = 7 → Complex.abs w ≤ Complex.abs v :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_complex_l3536_353620


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3536_353690

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 2

-- Define the theorem
theorem ellipse_and_line_intersection :
  ∀ (a b : ℝ),
  a > b ∧ b > 0 ∧
  2 * b = 2 ∧
  (Real.sqrt 6) / 3 = Real.sqrt (a^2 - b^2) / a →
  (∀ (x y : ℝ), ellipse a b x y ↔ x^2 / 3 + y^2 = 1) ∧
  (∀ (k : ℝ),
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse a b x₁ y₁ ∧
      ellipse a b x₂ y₂ ∧
      line k x₁ y₁ ∧
      line k x₂ y₂ ∧
      x₁ ≠ x₂ ∧
      x₁ * x₂ + y₁ * y₂ > 0) ↔
    (k > 1 ∧ k < Real.sqrt 13 / Real.sqrt 3) ∨
    (k < -1 ∧ k > -Real.sqrt 13 / Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l3536_353690


namespace NUMINAMATH_CALUDE_calculator_key_functions_l3536_353615

/-- Represents the keys on a calculator --/
inductive CalculatorKey
  | ON_C
  | OFF
  | Other

/-- Represents the functions of calculator keys --/
inductive KeyFunction
  | ClearScreen
  | PowerOff
  | Other

/-- Maps calculator keys to their functions --/
def key_function : CalculatorKey → KeyFunction
  | CalculatorKey.ON_C => KeyFunction.ClearScreen
  | CalculatorKey.OFF => KeyFunction.PowerOff
  | CalculatorKey.Other => KeyFunction.Other

theorem calculator_key_functions :
  (key_function CalculatorKey.ON_C = KeyFunction.ClearScreen) ∧
  (key_function CalculatorKey.OFF = KeyFunction.PowerOff) :=
by sorry

end NUMINAMATH_CALUDE_calculator_key_functions_l3536_353615


namespace NUMINAMATH_CALUDE_three_digit_numbers_equation_l3536_353605

theorem three_digit_numbers_equation : 
  ∃! (A B : ℕ), 
    100 ≤ A ∧ A < 1000 ∧
    100 ≤ B ∧ B < 1000 ∧
    1000 * A + B = 3 * A * B := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_equation_l3536_353605


namespace NUMINAMATH_CALUDE_root_product_theorem_l3536_353601

theorem root_product_theorem (a b m p q : ℝ) : 
  (a^2 - m*a + 5 = 0) → 
  (b^2 - m*b + 5 = 0) → 
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) → 
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) → 
  q = 36/5 := by
sorry

end NUMINAMATH_CALUDE_root_product_theorem_l3536_353601


namespace NUMINAMATH_CALUDE_total_cards_after_addition_l3536_353614

theorem total_cards_after_addition (initial_playing_cards initial_id_cards additional_playing_cards additional_id_cards : ℕ) :
  initial_playing_cards = 9 →
  initial_id_cards = 4 →
  additional_playing_cards = 6 →
  additional_id_cards = 3 →
  initial_playing_cards + initial_id_cards + additional_playing_cards + additional_id_cards = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_total_cards_after_addition_l3536_353614


namespace NUMINAMATH_CALUDE_rectangle_area_l3536_353629

theorem rectangle_area (a c : ℝ) (ha : a = 15) (hc : c = 17) : 
  ∃ b : ℝ, a * b = 120 ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3536_353629


namespace NUMINAMATH_CALUDE_invisible_square_exists_l3536_353635

/-- A point (x, y) is invisible if gcd(x, y) > 1 -/
def invisible (x y : ℤ) : Prop := Nat.gcd x.natAbs y.natAbs > 1

/-- For any natural number L, there exists integers a and b such that
    for all integers i and j where 0 ≤ i, j ≤ L, the point (a+i, b+j) is invisible -/
theorem invisible_square_exists (L : ℕ) :
  ∃ a b : ℤ, ∀ i j : ℤ, 0 ≤ i ∧ i ≤ L ∧ 0 ≤ j ∧ j ≤ L →
    invisible (a + i) (b + j) := by
  sorry

end NUMINAMATH_CALUDE_invisible_square_exists_l3536_353635


namespace NUMINAMATH_CALUDE_tomatoes_picked_yesterday_l3536_353659

def initial_tomatoes : ℕ := 160
def tomatoes_left_after_yesterday : ℕ := 104

theorem tomatoes_picked_yesterday :
  initial_tomatoes - tomatoes_left_after_yesterday = 56 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_picked_yesterday_l3536_353659


namespace NUMINAMATH_CALUDE_distance_difference_l3536_353606

/-- The difference in distances from Q to the intersection points of a line and a parabola -/
theorem distance_difference (Q : ℝ × ℝ) (C D : ℝ × ℝ) : 
  Q.1 = 2 ∧ Q.2 = 0 →
  C.2 - 2 * C.1 + 4 = 0 →
  D.2 - 2 * D.1 + 4 = 0 →
  C.2^2 = 3 * C.1 + 4 →
  D.2^2 = 3 * D.1 + 4 →
  |((C.1 - Q.1)^2 + (C.2 - Q.2)^2).sqrt - ((D.1 - Q.1)^2 + (D.2 - Q.2)^2).sqrt| = 
  |2 * (5 : ℝ).sqrt - (8.90625 : ℝ).sqrt| :=
by sorry

end NUMINAMATH_CALUDE_distance_difference_l3536_353606


namespace NUMINAMATH_CALUDE_shaded_area_is_60_l3536_353645

/-- Represents a point in a 2D grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a rectangular grid -/
structure Grid where
  width : ℕ
  height : ℕ

/-- Represents the shaded region in the grid -/
structure ShadedRegion where
  grid : Grid
  points : List Point

/-- Calculates the area of the shaded region -/
def shadedArea (region : ShadedRegion) : ℕ :=
  sorry

/-- The specific grid and shaded region from the problem -/
def problemGrid : Grid :=
  { width := 15, height := 5 }

def problemShadedRegion : ShadedRegion :=
  { grid := problemGrid,
    points := [
      { x := 0, y := 0 },   -- bottom left corner
      { x := 4, y := 3 },   -- first point
      { x := 9, y := 5 },   -- second point
      { x := 15, y := 5 }   -- top right corner
    ] }

/-- The main theorem to prove -/
theorem shaded_area_is_60 :
  shadedArea problemShadedRegion = 60 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_is_60_l3536_353645


namespace NUMINAMATH_CALUDE_cistern_filling_time_l3536_353643

theorem cistern_filling_time (capacity : ℝ) (fill_time : ℝ) (empty_time : ℝ) :
  fill_time = 10 →
  empty_time = 15 →
  (capacity / fill_time - capacity / empty_time) * (fill_time * empty_time / (empty_time - fill_time)) = capacity :=
by
  sorry

#check cistern_filling_time

end NUMINAMATH_CALUDE_cistern_filling_time_l3536_353643


namespace NUMINAMATH_CALUDE_diana_earnings_ratio_l3536_353613

/-- Diana's earnings over three months --/
def DianaEarnings (july : ℕ) (august_multiple : ℕ) : Prop :=
  let august := july * august_multiple
  let september := 2 * august
  july + august + september = 1500

theorem diana_earnings_ratio : 
  DianaEarnings 150 3 ∧ 
  ∀ x : ℕ, DianaEarnings 150 x → x = 3 :=
by sorry

end NUMINAMATH_CALUDE_diana_earnings_ratio_l3536_353613


namespace NUMINAMATH_CALUDE_unbroken_seashells_l3536_353649

def total_seashells : ℕ := 7
def broken_seashells : ℕ := 4

theorem unbroken_seashells :
  total_seashells - broken_seashells = 3 := by sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l3536_353649


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3536_353640

def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 10}
def B : Set ℝ := {x | 2 < x ∧ x < 7}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 3 ≤ x ∧ x < 7} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3536_353640


namespace NUMINAMATH_CALUDE_fourth_power_of_nested_square_root_l3536_353670

theorem fourth_power_of_nested_square_root : (Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))^4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_nested_square_root_l3536_353670


namespace NUMINAMATH_CALUDE_remainder_theorem_l3536_353685

theorem remainder_theorem : ∃ q : ℕ, 2^160 + 160 = q * (2^80 + 2^40 + 1) + 159 :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3536_353685


namespace NUMINAMATH_CALUDE_max_safe_caffeine_value_l3536_353695

/-- The maximum safe amount of caffeine one can consume per day -/
def max_safe_caffeine : ℕ := sorry

/-- The amount of caffeine in one energy drink (in mg) -/
def caffeine_per_drink : ℕ := 120

/-- The number of energy drinks Brandy consumes -/
def drinks_consumed : ℕ := 4

/-- The additional amount of caffeine Brandy can safely consume (in mg) -/
def additional_safe_caffeine : ℕ := 20

/-- Theorem stating the maximum safe amount of caffeine one can consume per day -/
theorem max_safe_caffeine_value : 
  max_safe_caffeine = caffeine_per_drink * drinks_consumed + additional_safe_caffeine := by
  sorry

end NUMINAMATH_CALUDE_max_safe_caffeine_value_l3536_353695


namespace NUMINAMATH_CALUDE_equation_describes_cone_l3536_353696

/-- Cylindrical coordinates -/
structure CylindricalCoord where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- Predicate for points satisfying θ = 2z -/
def SatisfiesEquation (p : CylindricalCoord) : Prop :=
  p.θ = 2 * p.z

/-- Predicate for points on a cone -/
def OnCone (p : CylindricalCoord) (α : ℝ) : Prop :=
  p.r = α * p.z

theorem equation_describes_cone :
  ∃ α : ℝ, ∀ p : CylindricalCoord, SatisfiesEquation p → OnCone p α :=
sorry

end NUMINAMATH_CALUDE_equation_describes_cone_l3536_353696


namespace NUMINAMATH_CALUDE_gus_buys_two_dozen_l3536_353661

def golf_balls_per_dozen : ℕ := 12

def dans_dozens : ℕ := 5
def chris_golf_balls : ℕ := 48
def total_golf_balls : ℕ := 132

def gus_dozens : ℕ := total_golf_balls / golf_balls_per_dozen - dans_dozens - chris_golf_balls / golf_balls_per_dozen

theorem gus_buys_two_dozen : gus_dozens = 2 := by
  sorry

end NUMINAMATH_CALUDE_gus_buys_two_dozen_l3536_353661


namespace NUMINAMATH_CALUDE_max_value_abc_fraction_l3536_353676

theorem max_value_abc_fraction (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^2) ≤ (1 : ℝ) / 4 := by
sorry

end NUMINAMATH_CALUDE_max_value_abc_fraction_l3536_353676


namespace NUMINAMATH_CALUDE_two_digit_triple_sum_product_l3536_353689

def digit_sum (n : ℕ) : ℕ := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def all_digits_different (p q r : ℕ) : Prop :=
  let digits := [p / 10, p % 10, q / 10, q % 10, r / 10, r % 10]
  ∀ i j, i ≠ j → digits.nthLe i sorry ≠ digits.nthLe j sorry

theorem two_digit_triple_sum_product (p q r : ℕ) : 
  is_two_digit p ∧ is_two_digit q ∧ is_two_digit r ∧ 
  all_digits_different p q r ∧
  p * q * digit_sum r = p * digit_sum q * r ∧
  p * digit_sum q * r = digit_sum p * q * r →
  ((p = 12 ∧ q = 36 ∧ r = 48) ∨ (p = 21 ∧ q = 63 ∧ r = 84)) :=
sorry

end NUMINAMATH_CALUDE_two_digit_triple_sum_product_l3536_353689


namespace NUMINAMATH_CALUDE_smallest_triangle_area_l3536_353609

/-- The smallest area of a triangle with given vertices -/
theorem smallest_triangle_area :
  let A : ℝ × ℝ × ℝ := (-1, 1, 2)
  let B : ℝ × ℝ × ℝ := (1, 2, 3)
  let C : ℝ → ℝ → ℝ × ℝ × ℝ := fun t s ↦ (t, s, 1)
  let triangle_area (t s : ℝ) : ℝ :=
    (1 / 2) * Real.sqrt ((s^2) + ((-t-3)^2) + ((2*s-t-2)^2))
  ∃ (min_area : ℝ), min_area = Real.sqrt 58 / 2 ∧
    ∀ (t s : ℝ), triangle_area t s ≥ min_area :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_triangle_area_l3536_353609


namespace NUMINAMATH_CALUDE_coin_toss_probability_l3536_353634

theorem coin_toss_probability : 
  let n : ℕ := 5  -- Total number of coins
  let k : ℕ := 3  -- Number of heads we want
  let p : ℚ := 1/2  -- Probability of getting heads on a single toss
  Nat.choose n k * p^n = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l3536_353634


namespace NUMINAMATH_CALUDE_inequality_proof_l3536_353632

theorem inequality_proof (K x : ℝ) (hK : K > 1) (hx_pos : x > 0) (hx_bound : x < π / K) :
  (Real.sin (K * x) / Real.sin x) < K * Real.exp (-(K^2 - 1) * x^2 / 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3536_353632


namespace NUMINAMATH_CALUDE_game_points_sum_l3536_353673

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [6, 2, 5, 3, 4]
def carlos_rolls : List ℕ := [3, 2, 2, 6, 1]

theorem game_points_sum : 
  (List.sum (List.map g allie_rolls)) + (List.sum (List.map g carlos_rolls)) = 44 := by
  sorry

end NUMINAMATH_CALUDE_game_points_sum_l3536_353673


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l3536_353660

theorem fraction_equals_zero (x y : ℝ) :
  (x - 5) / (5 * x + y) = 0 ∧ y ≠ -5 * x → x = 5 ∧ y ≠ -25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l3536_353660


namespace NUMINAMATH_CALUDE_angle_A_measure_l3536_353692

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem angle_A_measure (t : Triangle) : 
  t.a = Real.sqrt 3 → t.b = 1 → t.B = π / 6 → t.A = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_angle_A_measure_l3536_353692


namespace NUMINAMATH_CALUDE_number_problem_l3536_353697

theorem number_problem (N p q : ℝ) 
  (h1 : N / p = 4)
  (h2 : N / q = 18)
  (h3 : p - q = 0.5833333333333334) : 
  N = 3 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l3536_353697


namespace NUMINAMATH_CALUDE_largest_in_set_l3536_353611

def S : Set ℝ := {0.109, 0.2, 0.111, 0.114, 0.19}

theorem largest_in_set : ∀ x ∈ S, x ≤ 0.2 := by sorry

end NUMINAMATH_CALUDE_largest_in_set_l3536_353611


namespace NUMINAMATH_CALUDE_evan_needs_seven_l3536_353686

-- Define the given amounts
def david_found : ℕ := 12
def evan_initial : ℕ := 1
def watch_cost : ℕ := 20

-- Define Evan's total after receiving money from David
def evan_total : ℕ := evan_initial + david_found

-- Theorem to prove
theorem evan_needs_seven : watch_cost - evan_total = 7 := by
  sorry

end NUMINAMATH_CALUDE_evan_needs_seven_l3536_353686


namespace NUMINAMATH_CALUDE_minimum_fourth_round_score_l3536_353610

def minimum_average_score : ℝ := 96
def number_of_rounds : ℕ := 4
def first_round_score : ℝ := 95
def second_round_score : ℝ := 97
def third_round_score : ℝ := 94

theorem minimum_fourth_round_score :
  let total_required_score := minimum_average_score * number_of_rounds
  let sum_of_first_three_rounds := first_round_score + second_round_score + third_round_score
  let minimum_fourth_round_score := total_required_score - sum_of_first_three_rounds
  minimum_fourth_round_score = 98 := by sorry

end NUMINAMATH_CALUDE_minimum_fourth_round_score_l3536_353610


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3536_353665

/-- The circle C with center (1, 0) and radius 5 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 25}

/-- The point M -/
def M : ℝ × ℝ := (-3, 3)

/-- The proposed tangent line -/
def tangentLine (x y : ℝ) : Prop :=
  4 * x - 3 * y + 21 = 0

/-- Theorem stating that the proposed line is tangent to C at M -/
theorem tangent_line_to_circle :
  (M ∈ C) ∧
  (∃ (p : ℝ × ℝ), p ∈ C ∧ p ≠ M ∧ tangentLine p.1 p.2) ∧
  (∀ (q : ℝ × ℝ), q ∈ C → q ≠ M → ¬tangentLine q.1 q.2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3536_353665


namespace NUMINAMATH_CALUDE_digital_root_of_8_pow_1989_l3536_353679

def digital_root (n : ℕ) : ℕ :=
  if n % 9 = 0 then 9 else n % 9

theorem digital_root_of_8_pow_1989 :
  digital_root (8^1989) = 8 :=
sorry

end NUMINAMATH_CALUDE_digital_root_of_8_pow_1989_l3536_353679


namespace NUMINAMATH_CALUDE_mark_to_jaydon_ratio_l3536_353612

/-- Represents the number of cans brought by each person -/
structure Cans where
  rachel : ℕ
  jaydon : ℕ
  mark : ℕ

/-- The conditions of the food drive problem -/
def FoodDrive (c : Cans) : Prop :=
  c.mark = 100 ∧
  c.jaydon = 2 * c.rachel + 5 ∧
  c.rachel + c.jaydon + c.mark = 135

/-- The theorem to be proved -/
theorem mark_to_jaydon_ratio (c : Cans) (h : FoodDrive c) : 
  c.mark / c.jaydon = 4 := by
  sorry

#check mark_to_jaydon_ratio

end NUMINAMATH_CALUDE_mark_to_jaydon_ratio_l3536_353612


namespace NUMINAMATH_CALUDE_starting_lineup_count_l3536_353688

def total_members : ℕ := 12
def offensive_linemen : ℕ := 4
def quick_reflex_players : ℕ := 2

def starting_lineup_combinations : ℕ := offensive_linemen * quick_reflex_players * 1 * (total_members - 3)

theorem starting_lineup_count : starting_lineup_combinations = 72 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l3536_353688


namespace NUMINAMATH_CALUDE_temperature_difference_qianan_l3536_353653

/-- The temperature difference between two times of day -/
def temperature_difference (temp1 : Int) (temp2 : Int) : Int :=
  temp2 - temp1

/-- Proof that the temperature difference between 10 a.m. and midnight is 9°C -/
theorem temperature_difference_qianan : 
  let midnight_temp : Int := -4
  let morning_temp : Int := 5
  temperature_difference midnight_temp morning_temp = 9 := by
sorry

end NUMINAMATH_CALUDE_temperature_difference_qianan_l3536_353653


namespace NUMINAMATH_CALUDE_unique_b_l3536_353678

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the properties of b
def b_properties (b : ℝ) : Prop :=
  b > 1 ∧ 
  (∀ x, x ∈ Set.Icc 1 b ↔ f x ∈ Set.Icc 1 b) ∧
  (∀ x, x ∈ Set.Icc 1 b → f x ≤ b)

-- Theorem statement
theorem unique_b : ∃! b, b_properties b ∧ b = 2 := by sorry

end NUMINAMATH_CALUDE_unique_b_l3536_353678


namespace NUMINAMATH_CALUDE_cos_five_pi_sixth_minus_alpha_l3536_353617

theorem cos_five_pi_sixth_minus_alpha (α : ℝ) (h : Real.cos (π / 6 + α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 - α) = -(Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_cos_five_pi_sixth_minus_alpha_l3536_353617


namespace NUMINAMATH_CALUDE_least_number_divisibility_l3536_353636

theorem least_number_divisibility (x : ℕ) : x = 10315 ↔ 
  (∀ y : ℕ, y < x → ¬((1024 + y) % (17 * 23 * 29) = 0)) ∧ 
  ((1024 + x) % (17 * 23 * 29) = 0) := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l3536_353636


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l3536_353671

/-- A parabola y = ax^2 + bx + 2 is tangent to the line y = 2x + 3 if and only if a = -1 and b = 4 -/
theorem parabola_tangent_to_line (a b : ℝ) : 
  (∃ x : ℝ, ax^2 + bx + 2 = 2*x + 3 ∧ 
   ∀ y : ℝ, y ≠ x → ax^2 + bx + 2 ≠ 2*y + 3) ↔ 
  (a = -1 ∧ b = 4) :=
sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l3536_353671


namespace NUMINAMATH_CALUDE_debate_pairs_l3536_353607

theorem debate_pairs (n : ℕ) (h : n = 12) : Nat.choose n 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_debate_pairs_l3536_353607


namespace NUMINAMATH_CALUDE_E_72_with_4_equals_9_l3536_353628

/-- The number of ways to express an integer as a product of integers greater than 1 -/
def E (n : ℕ) : ℕ := sorry

/-- The number of ways to express 72 as a product of integers greater than 1,
    including at least one factor of 4, where the order of factors matters -/
def E_72_with_4 : ℕ := sorry

/-- The prime factorization of 72 -/
def prime_factorization_72 : List ℕ := [2, 2, 2, 3, 3]

theorem E_72_with_4_equals_9 : E_72_with_4 = 9 := by sorry

end NUMINAMATH_CALUDE_E_72_with_4_equals_9_l3536_353628


namespace NUMINAMATH_CALUDE_sector_central_angle_l3536_353602

/-- Given a circle with circumference 2π + 2 and a sector of that circle with arc length 2π - 2,
    the central angle of the sector is π - 1. -/
theorem sector_central_angle (r : ℝ) (α : ℝ) 
    (h_circumference : 2 * π * r = 2 * π + 2)
    (h_arc_length : r * α = 2 * π - 2) : 
  α = π - 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3536_353602


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l3536_353641

-- Define an arithmetic sequence of integers
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define an increasing sequence
def is_increasing_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

-- Theorem statement
theorem arithmetic_sequence_product (a : ℕ → ℤ) :
  is_arithmetic_sequence a →
  is_increasing_sequence a →
  a 4 * a 5 = 45 →
  a 3 * a 6 = 13 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l3536_353641


namespace NUMINAMATH_CALUDE_rectangle_rhombus_perimeter_ratio_l3536_353699

/-- The ratio of the perimeter of a 3 by 2 rectangle to the perimeter of a rhombus
    formed by rearranging four congruent right-angled triangles that the rectangle
    is split into is 1:1. -/
theorem rectangle_rhombus_perimeter_ratio :
  let rectangle_length : ℝ := 3
  let rectangle_width : ℝ := 2
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  let triangle_leg1 := rectangle_length / 2
  let triangle_leg2 := rectangle_width
  let triangle_hypotenuse := Real.sqrt (triangle_leg1^2 + triangle_leg2^2)
  let rhombus_side := triangle_hypotenuse
  let rhombus_perimeter := 4 * rhombus_side
  rectangle_perimeter / rhombus_perimeter = 1 := by
sorry

end NUMINAMATH_CALUDE_rectangle_rhombus_perimeter_ratio_l3536_353699


namespace NUMINAMATH_CALUDE_blue_balls_removed_l3536_353604

theorem blue_balls_removed (total_balls : Nat) (initial_blue : Nat) (final_probability : Rat) :
  total_balls = 15 →
  initial_blue = 7 →
  final_probability = 1/3 →
  ∃ (removed : Nat), removed = 3 ∧
    (initial_blue - removed : Rat) / (total_balls - removed : Rat) = final_probability :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_removed_l3536_353604


namespace NUMINAMATH_CALUDE_student_average_age_l3536_353664

theorem student_average_age 
  (num_students : ℕ) 
  (teacher_age : ℕ) 
  (average_with_teacher : ℚ) :
  num_students = 20 →
  teacher_age = 36 →
  average_with_teacher = 16 →
  (num_students * (average_with_teacher : ℚ) + teacher_age) / (num_students + 1 : ℚ) = average_with_teacher →
  (num_students * (average_with_teacher : ℚ) + teacher_age - teacher_age) / num_students = 15 := by
  sorry

end NUMINAMATH_CALUDE_student_average_age_l3536_353664


namespace NUMINAMATH_CALUDE_gathering_handshakes_l3536_353626

/-- Represents a gathering of couples -/
structure Gathering where
  couples : Nat
  people : Nat
  men : Nat
  women : Nat

/-- Calculates the number of handshakes in a gathering -/
def handshakes (g : Gathering) : Nat :=
  g.men * (g.women - 1)

/-- Theorem: In a gathering of 7 couples with the given handshake rules, 
    the total number of handshakes is 42 -/
theorem gathering_handshakes :
  ∀ g : Gathering, 
    g.couples = 7 →
    g.people = 2 * g.couples →
    g.men = g.couples →
    g.women = g.couples →
    handshakes g = 42 := by
  sorry

end NUMINAMATH_CALUDE_gathering_handshakes_l3536_353626


namespace NUMINAMATH_CALUDE_money_problem_l3536_353650

theorem money_problem (a b : ℝ) 
  (h1 : 6 * a + b > 78)
  (h2 : 4 * a - b = 42)
  (h3 : a ≥ 0)  -- Assuming money can't be negative
  (h4 : b ≥ 0)  -- Assuming money can't be negative
  : a > 12 ∧ b > 6 :=
by
  sorry

end NUMINAMATH_CALUDE_money_problem_l3536_353650


namespace NUMINAMATH_CALUDE_cost_of_paints_paint_set_cost_l3536_353619

theorem cost_of_paints (total_spent : ℕ) (num_classes : ℕ) (folders_per_class : ℕ) 
  (pencils_per_class : ℕ) (pencils_per_eraser : ℕ) (folder_cost : ℕ) (pencil_cost : ℕ) 
  (eraser_cost : ℕ) : ℕ :=
  let num_folders := num_classes * folders_per_class
  let num_pencils := num_classes * pencils_per_class
  let num_erasers := num_pencils / pencils_per_eraser
  let folders_total_cost := num_folders * folder_cost
  let pencils_total_cost := num_pencils * pencil_cost
  let erasers_total_cost := num_erasers * eraser_cost
  let supplies_cost := folders_total_cost + pencils_total_cost + erasers_total_cost
  total_spent - supplies_cost

theorem paint_set_cost : cost_of_paints 80 6 1 3 6 6 2 1 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_paints_paint_set_cost_l3536_353619
