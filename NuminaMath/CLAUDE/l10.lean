import Mathlib

namespace NUMINAMATH_CALUDE_false_premise_implications_l10_1037

theorem false_premise_implications :
  ∃ (p : Prop) (q r : Prop), 
    (¬p) ∧ (p → q) ∧ (p → r) ∧ q ∧ (¬r) := by
  -- Let p be the false premise 5 = -5
  let p := (5 = -5)
  -- Let q be the true conclusion 25 = 25
  let q := (25 = 25)
  -- Let r be the false conclusion 125 = -125
  let r := (125 = -125)
  
  have h1 : ¬p := by sorry
  have h2 : p → q := by sorry
  have h3 : p → r := by sorry
  have h4 : q := by sorry
  have h5 : ¬r := by sorry

  exact ⟨p, q, r, h1, h2, h3, h4, h5⟩

#check false_premise_implications

end NUMINAMATH_CALUDE_false_premise_implications_l10_1037


namespace NUMINAMATH_CALUDE_work_completion_time_l10_1040

theorem work_completion_time (b_days : ℝ) (a_wage_ratio : ℝ) (a_days : ℝ) : 
  b_days = 15 →
  a_wage_ratio = 3/5 →
  a_wage_ratio = (1/a_days) / (1/a_days + 1/b_days) →
  a_days = 10 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l10_1040


namespace NUMINAMATH_CALUDE_dividend_calculation_l10_1086

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 18)
  (h2 : quotient = 9)
  (h3 : remainder = 4) :
  divisor * quotient + remainder = 166 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l10_1086


namespace NUMINAMATH_CALUDE_trapezium_area_l10_1078

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 15) (hh : h = 14) :
  (a + b) * h / 2 = 245 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_area_l10_1078


namespace NUMINAMATH_CALUDE_area_trapezoid_equals_rectangle_l10_1052

-- Define the points
variable (P Q R S T : ℝ × ℝ)

-- Define the shapes
def rectangle_PQRS : Set (ℝ × ℝ) := sorry
def trapezoid_TQSR : Set (ℝ × ℝ) := sorry

-- Define the area function
noncomputable def area : Set (ℝ × ℝ) → ℝ := sorry

-- State the theorem
theorem area_trapezoid_equals_rectangle
  (h1 : area rectangle_PQRS = 20)
  (h2 : trapezoid_TQSR ⊆ rectangle_PQRS)
  (h3 : P = (0, 0))
  (h4 : Q = (5, 0))
  (h5 : R = (5, 4))
  (h6 : S = (0, 4))
  (h7 : T = (2, 4)) :
  area trapezoid_TQSR = area rectangle_PQRS :=
by sorry

end NUMINAMATH_CALUDE_area_trapezoid_equals_rectangle_l10_1052


namespace NUMINAMATH_CALUDE_square_coverage_l10_1015

/-- The smallest number of 3-by-4 rectangles needed to cover a square region exactly -/
def min_rectangles : ℕ := 12

/-- The side length of the square region -/
def square_side : ℕ := 12

/-- The width of each rectangle -/
def rectangle_width : ℕ := 3

/-- The height of each rectangle -/
def rectangle_height : ℕ := 4

theorem square_coverage :
  (square_side * square_side) = (min_rectangles * rectangle_width * rectangle_height) ∧
  (square_side % rectangle_width = 0) ∧
  (square_side % rectangle_height = 0) ∧
  ∀ n : ℕ, n < min_rectangles →
    (n * rectangle_width * rectangle_height) < (square_side * square_side) :=
by sorry

end NUMINAMATH_CALUDE_square_coverage_l10_1015


namespace NUMINAMATH_CALUDE_tomatoes_picked_yesterday_l10_1092

def initial_tomatoes : ℕ := 160
def tomatoes_left_after_yesterday : ℕ := 104

theorem tomatoes_picked_yesterday :
  initial_tomatoes - tomatoes_left_after_yesterday = 56 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_picked_yesterday_l10_1092


namespace NUMINAMATH_CALUDE_pentagonal_tiles_count_l10_1046

theorem pentagonal_tiles_count (t p : ℕ) : 
  t + p = 30 →  -- Total number of tiles
  3 * t + 5 * p = 100 →  -- Total number of edges
  p = 5  -- Number of pentagonal tiles
  := by sorry

end NUMINAMATH_CALUDE_pentagonal_tiles_count_l10_1046


namespace NUMINAMATH_CALUDE_chord_length_l10_1090

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def are_externally_tangent (c1 c2 : Circle) : Prop := sorry

def is_internally_tangent (c1 c2 : Circle) : Prop := sorry

def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop := sorry

def is_common_external_tangent (c1 c2 c3 : Circle) (chord : ℝ × ℝ → ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem chord_length 
  (c1 c2 c3 : Circle)
  (chord : ℝ × ℝ → ℝ × ℝ)
  (h1 : are_externally_tangent c1 c2)
  (h2 : is_internally_tangent c1 c3)
  (h3 : is_internally_tangent c2 c3)
  (h4 : c1.radius = 3)
  (h5 : c2.radius = 9)
  (h6 : are_collinear c1.center c2.center c3.center)
  (h7 : is_common_external_tangent c1 c2 c3 chord) :
  ∃ (a b : ℝ × ℝ), chord a = b ∧ Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 18 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l10_1090


namespace NUMINAMATH_CALUDE_zoe_pool_cleaning_earnings_l10_1042

/-- Represents Zoe's earnings and babysitting frequencies -/
structure ZoeEarnings where
  total : ℕ
  zachary_earnings : ℕ
  julie_freq : ℕ
  zachary_freq : ℕ
  chloe_freq : ℕ

/-- Calculates Zoe's earnings from pool cleaning -/
def pool_cleaning_earnings (e : ZoeEarnings) : ℕ :=
  e.total - (e.zachary_earnings * (1 + 3 + 5))

/-- Theorem stating that Zoe's pool cleaning earnings are $2,600 -/
theorem zoe_pool_cleaning_earnings :
  ∀ e : ZoeEarnings,
    e.total = 8000 ∧
    e.zachary_earnings = 600 ∧
    e.julie_freq = 3 * e.zachary_freq ∧
    e.zachary_freq * 5 = e.chloe_freq →
    pool_cleaning_earnings e = 2600 :=
by
  sorry


end NUMINAMATH_CALUDE_zoe_pool_cleaning_earnings_l10_1042


namespace NUMINAMATH_CALUDE_new_to_original_detergent_water_ratio_l10_1091

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

end NUMINAMATH_CALUDE_new_to_original_detergent_water_ratio_l10_1091


namespace NUMINAMATH_CALUDE_josh_spending_l10_1047

/-- Josh's spending problem -/
theorem josh_spending (x y : ℝ) : 
  (x - 1.75 - y = 6) → y = x - 7.75 := by
sorry

end NUMINAMATH_CALUDE_josh_spending_l10_1047


namespace NUMINAMATH_CALUDE_gcf_2835_9150_l10_1088

theorem gcf_2835_9150 : Nat.gcd 2835 9150 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_2835_9150_l10_1088


namespace NUMINAMATH_CALUDE_race_completion_time_l10_1073

theorem race_completion_time (total_runners : ℕ) (avg_time_all : ℝ) (fastest_time : ℝ) : 
  total_runners = 4 →
  avg_time_all = 30 →
  fastest_time = 15 →
  (((avg_time_all * total_runners) - fastest_time) / (total_runners - 1) : ℝ) = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_race_completion_time_l10_1073


namespace NUMINAMATH_CALUDE_isosceles_triangle_exists_l10_1059

/-- Isosceles triangle type -/
structure IsoscelesTriangle where
  /-- Base length of the isosceles triangle -/
  base : ℝ
  /-- Length of the equal sides of the isosceles triangle -/
  side : ℝ
  /-- Height of the isosceles triangle -/
  height : ℝ
  /-- Condition: base and side are positive -/
  base_pos : 0 < base
  side_pos : 0 < side
  /-- Condition: height is positive -/
  height_pos : 0 < height
  /-- Condition: triangle inequality -/
  triangle_ineq : base < 2 * side

/-- Theorem: Given a perimeter and a height, an isosceles triangle exists -/
theorem isosceles_triangle_exists (perimeter : ℝ) (height : ℝ) 
  (perimeter_pos : 0 < perimeter) (height_pos : 0 < height) : 
  ∃ (t : IsoscelesTriangle), t.base + 2 * t.side = perimeter ∧ t.height = height := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_exists_l10_1059


namespace NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l10_1067

/-- Two lines are parallel if their slopes are equal -/
def parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * b₂ = a₂ * b₁

/-- Line l₁ with equation x + (1+m)y = 2 - m -/
def l₁ (m : ℝ) (x y : ℝ) : Prop :=
  x + (1+m)*y = 2 - m

/-- Line l₂ with equation 2mx + 4y + 16 = 0 -/
def l₂ (m : ℝ) (x y : ℝ) : Prop :=
  2*m*x + 4*y + 16 = 0

theorem parallel_lines_m_equals_one :
  ∀ m : ℝ, parallel 1 (1+m) (2*m) 4 → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_equals_one_l10_1067


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l10_1049

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns true if all numbers in the range [a, b] are nonprime, false otherwise -/
def allNonPrime (a b : ℕ) : Prop := sorry

theorem smallest_prime_after_six_nonprimes : 
  ∃ (k : ℕ), 
    isPrime 97 ∧ 
    (∀ p < 97, isPrime p → ¬(allNonPrime (p + 1) (p + 6))) ∧
    allNonPrime 91 96 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l10_1049


namespace NUMINAMATH_CALUDE_correct_num_students_l10_1053

/-- The number of students in a class with incorrectly entered marks -/
def num_students : ℕ := by sorry

/-- The total increase in marks due to incorrect entry -/
def total_mark_increase : ℕ := 44

/-- The increase in class average due to incorrect entry -/
def average_increase : ℚ := 1/2

theorem correct_num_students :
  num_students = 88 ∧
  (total_mark_increase : ℚ) = num_students * average_increase := by sorry

end NUMINAMATH_CALUDE_correct_num_students_l10_1053


namespace NUMINAMATH_CALUDE_partnership_investment_l10_1084

theorem partnership_investment (a c total_profit c_profit : ℚ) (ha : a = 45000) (hc : c = 72000) (htotal : total_profit = 60000) (hc_profit : c_profit = 24000) :
  ∃ b : ℚ, 
    (c_profit / total_profit = c / (a + b + c)) ∧
    b = 63000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_l10_1084


namespace NUMINAMATH_CALUDE_no_divisibility_pairs_l10_1093

theorem no_divisibility_pairs : ¬∃ (m n : ℕ+), (m.val * n.val ∣ 3^m.val + 1) ∧ (m.val * n.val ∣ 3^n.val + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_divisibility_pairs_l10_1093


namespace NUMINAMATH_CALUDE_quadratic_polynomial_property_l10_1060

/-- A quadratic polynomial with a common root property -/
structure QuadraticPolynomial where
  P : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, P x = a * x^2 + b * x + c
  common_root : ∃ t : ℝ, P t = 0 ∧ P (P (P t)) = 0

/-- 
For any quadratic polynomial P(x) where P(x) and P(P(P(x))) have a common root, 
P(0)P(1) = 0
-/
theorem quadratic_polynomial_property (p : QuadraticPolynomial) : 
  p.P 0 * p.P 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_property_l10_1060


namespace NUMINAMATH_CALUDE_cheaper_module_cost_l10_1063

-- Define the total number of modules
def total_modules : ℕ := 22

-- Define the number of cheaper modules
def cheaper_modules : ℕ := 21

-- Define the cost of the expensive module
def expensive_module_cost : ℚ := 10

-- Define the total stock value
def total_stock_value : ℚ := 62.5

-- Theorem to prove
theorem cheaper_module_cost :
  ∃ (x : ℚ), x > 0 ∧ x < expensive_module_cost ∧
  x * cheaper_modules + expensive_module_cost = total_stock_value ∧
  x = 2.5 := by sorry

end NUMINAMATH_CALUDE_cheaper_module_cost_l10_1063


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l10_1006

theorem rectangle_perimeter (area : ℝ) (width : ℝ) (length : ℝ) :
  area = 500 →
  length = 2 * width →
  area = length * width →
  2 * (length + width) = 30 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l10_1006


namespace NUMINAMATH_CALUDE_wxyz_unique_product_l10_1019

/-- Represents a letter of the alphabet -/
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J | K | L | M
| N | O | P | Q | R | S | T | U | V | W | X | Y | Z

/-- Assigns a numeric value to each letter -/
def letterValue : Letter → Nat
| Letter.A => 1  | Letter.B => 2  | Letter.C => 3  | Letter.D => 4
| Letter.E => 5  | Letter.F => 6  | Letter.G => 7  | Letter.H => 8
| Letter.I => 9  | Letter.J => 10 | Letter.K => 11 | Letter.L => 12
| Letter.M => 13 | Letter.N => 14 | Letter.O => 15 | Letter.P => 16
| Letter.Q => 17 | Letter.R => 18 | Letter.S => 19 | Letter.T => 20
| Letter.U => 21 | Letter.V => 22 | Letter.W => 23 | Letter.X => 24
| Letter.Y => 25 | Letter.Z => 26

/-- Represents a four-letter sequence -/
structure FourLetterSequence :=
  (first second third fourth : Letter)

/-- Calculates the product of a four-letter sequence -/
def sequenceProduct (seq : FourLetterSequence) : Nat :=
  (letterValue seq.first) * (letterValue seq.second) * (letterValue seq.third) * (letterValue seq.fourth)

/-- States that WXYZ is the unique four-letter sequence with a product of 29700 -/
theorem wxyz_unique_product :
  ∀ (seq : FourLetterSequence),
    sequenceProduct seq = 29700 →
    seq = FourLetterSequence.mk Letter.W Letter.X Letter.Y Letter.Z :=
by sorry

end NUMINAMATH_CALUDE_wxyz_unique_product_l10_1019


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_19_l10_1077

theorem smallest_k_for_64_power_gt_4_19 : ∃ k : ℕ, k = 7 ∧ 
  (∀ m : ℕ, 64^m > 4^19 → m ≥ k) ∧ 
  64^k > 4^19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_19_l10_1077


namespace NUMINAMATH_CALUDE_min_value_of_function_l10_1041

theorem min_value_of_function (x : ℝ) : 
  (x^2 + 5) / Real.sqrt (x^2 + 4) ≥ 5/2 ∧ 
  ∃ y : ℝ, (y^2 + 5) / Real.sqrt (y^2 + 4) = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l10_1041


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l10_1071

/-- Two vectors are parallel if their corresponding components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (2, 4)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l10_1071


namespace NUMINAMATH_CALUDE_no_natural_solution_l10_1034

theorem no_natural_solution : ¬∃ (m n : ℕ), m * n * (m + n) = 2020 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l10_1034


namespace NUMINAMATH_CALUDE_point_quadrant_relation_l10_1010

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Checks if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem stating that if P(a+b, ab) is in the second quadrant, 
    then Q(-a, b) is in the fourth quadrant -/
theorem point_quadrant_relation (a b : ℝ) :
  isInSecondQuadrant (Point.mk (a + b) (a * b)) →
  isInFourthQuadrant (Point.mk (-a) b) :=
by
  sorry


end NUMINAMATH_CALUDE_point_quadrant_relation_l10_1010


namespace NUMINAMATH_CALUDE_Q_formula_l10_1069

def T (n : ℕ) : ℕ := (n * (n + 1)) / 2

def Q (n : ℕ) : ℚ :=
  if n < 2 then 0
  else Finset.prod (Finset.range (n - 1)) (fun k => (T (k + 2) : ℚ) / ((T (k + 3) : ℚ) - 1))

theorem Q_formula (n : ℕ) (h : n ≥ 2) : Q n = 2 / (n + 3) := by
  sorry

end NUMINAMATH_CALUDE_Q_formula_l10_1069


namespace NUMINAMATH_CALUDE_polygon_sides_when_interior_thrice_exterior_l10_1017

theorem polygon_sides_when_interior_thrice_exterior : ∀ n : ℕ,
  (n ≥ 3) →
  (180 * (n - 2) = 3 * 360) →
  n = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_when_interior_thrice_exterior_l10_1017


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l10_1061

/-- Calculate the total cost for a group at a restaurant where adults pay and kids eat free -/
theorem restaurant_bill_calculation (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : 
  total_people = 12 →
  num_kids = 7 →
  adult_meal_cost = 3 →
  (total_people - num_kids) * adult_meal_cost = 15 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l10_1061


namespace NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l10_1065

theorem tan_ratio_from_sin_sum_diff (x y : ℝ) 
  (h1 : Real.sin (x + y) = 5/8) 
  (h2 : Real.sin (x - y) = 1/4) : 
  Real.tan x / Real.tan y = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_from_sin_sum_diff_l10_1065


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l10_1058

theorem square_area_with_four_circles (r : ℝ) (h : r = 3) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l10_1058


namespace NUMINAMATH_CALUDE_samara_friends_average_alligators_l10_1002

/-- Given a group of people searching for alligators, calculate the average number
    of alligators seen by friends, given the total number seen, the number seen by
    one person, and the number of friends. -/
def average_alligators_seen_by_friends 
  (total_alligators : ℕ) 
  (alligators_seen_by_one : ℕ) 
  (num_friends : ℕ) : ℚ :=
  (total_alligators - alligators_seen_by_one) / num_friends

/-- Prove that given the specific values from the problem, 
    the average number of alligators seen by each friend is 10. -/
theorem samara_friends_average_alligators :
  average_alligators_seen_by_friends 50 20 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_samara_friends_average_alligators_l10_1002


namespace NUMINAMATH_CALUDE_percentage_of_sum_l10_1054

theorem percentage_of_sum (x y : ℝ) (P : ℝ) 
  (h1 : 0.5 * (x - y) = (P / 100) * (x + y)) 
  (h2 : y = 0.25 * x) : 
  P = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_sum_l10_1054


namespace NUMINAMATH_CALUDE_fibonacci_geometric_sequence_l10_1011

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the theorem
theorem fibonacci_geometric_sequence (a k p : ℕ) :
  (∃ r : ℚ, r > 1 ∧ fib k = r * fib a ∧ fib p = r * fib k) →  -- Geometric sequence condition
  (a < k ∧ k < p) →  -- Increasing order condition
  (a + k + p = 2010) →  -- Sum condition
  a = 669 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_geometric_sequence_l10_1011


namespace NUMINAMATH_CALUDE_arithmetic_sum_1_to_19_l10_1016

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ aₙ : ℕ) (d : ℕ) : ℕ := 
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Proof that the sum of the arithmetic sequence 1, 3, 5, ..., 17, 19 is 100 -/
theorem arithmetic_sum_1_to_19 : arithmetic_sum 1 19 2 = 100 := by
  sorry

#eval arithmetic_sum 1 19 2

end NUMINAMATH_CALUDE_arithmetic_sum_1_to_19_l10_1016


namespace NUMINAMATH_CALUDE_fifth_power_equality_l10_1033

theorem fifth_power_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_equality_l10_1033


namespace NUMINAMATH_CALUDE_original_price_correct_l10_1004

/-- The original price of water bottles that satisfies the given conditions --/
def original_price : ℝ :=
  let number_of_bottles : ℕ := 60
  let reduced_price : ℝ := 1.85
  let shortfall : ℝ := 9
  2

theorem original_price_correct :
  let number_of_bottles : ℕ := 60
  let reduced_price : ℝ := 1.85
  let shortfall : ℝ := 9
  (number_of_bottles : ℝ) * original_price = 
    (number_of_bottles : ℝ) * reduced_price + shortfall :=
by
  sorry

#eval original_price

end NUMINAMATH_CALUDE_original_price_correct_l10_1004


namespace NUMINAMATH_CALUDE_conservation_center_count_l10_1030

/-- The number of turtles in a conservation center -/
def total_turtles (green : ℕ) (hawksbill : ℕ) : ℕ := green + hawksbill

/-- The number of hawksbill turtles is twice more than the number of green turtles -/
def hawksbill_count (green : ℕ) : ℕ := green + 2 * green

theorem conservation_center_count :
  let green := 800
  let hawksbill := hawksbill_count green
  total_turtles green hawksbill = 3200 := by sorry

end NUMINAMATH_CALUDE_conservation_center_count_l10_1030


namespace NUMINAMATH_CALUDE_inequality_and_max_value_l10_1085

theorem inequality_and_max_value :
  (∀ a b c d : ℝ, (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = 5 → ∀ x : ℝ, 2*a + b ≤ x → x ≤ 5) :=
by sorry


end NUMINAMATH_CALUDE_inequality_and_max_value_l10_1085


namespace NUMINAMATH_CALUDE_y_value_proof_l10_1032

theorem y_value_proof (x y : ℕ+) 
  (h1 : y = (x : ℚ) * (1/4 : ℚ) * (1/2 : ℚ))
  (h2 : (y : ℚ) * (x : ℚ) / 100 = 100) :
  y = 35 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l10_1032


namespace NUMINAMATH_CALUDE_consecutive_squareful_numbers_l10_1035

/-- A natural number is squareful if it has a square divisor greater than 1 -/
def IsSquareful (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 1 ∧ m * m ∣ n

/-- For any natural number k, there exist k consecutive squareful numbers -/
theorem consecutive_squareful_numbers :
  ∀ k : ℕ, ∃ n : ℕ, ∀ i : ℕ, i < k → IsSquareful (n + i) :=
sorry

end NUMINAMATH_CALUDE_consecutive_squareful_numbers_l10_1035


namespace NUMINAMATH_CALUDE_fred_grew_nine_onions_l10_1003

/-- The number of onions Sally grew -/
def sally_onions : ℕ := 5

/-- The number of onions Sally and Fred gave away -/
def onions_given_away : ℕ := 4

/-- The number of onions Sally and Fred have remaining -/
def onions_remaining : ℕ := 10

/-- The number of onions Fred grew -/
def fred_onions : ℕ := sally_onions + onions_given_away + onions_remaining - sally_onions - onions_given_away

theorem fred_grew_nine_onions : fred_onions = 9 := by
  sorry

end NUMINAMATH_CALUDE_fred_grew_nine_onions_l10_1003


namespace NUMINAMATH_CALUDE_hamburger_combinations_l10_1048

/-- The number of different condiments available. -/
def num_condiments : ℕ := 10

/-- The number of patty options available. -/
def num_patty_options : ℕ := 4

/-- The total number of different hamburger combinations. -/
def total_combinations : ℕ := 2^num_condiments * num_patty_options

/-- Theorem stating that the total number of different hamburger combinations is 4096. -/
theorem hamburger_combinations : total_combinations = 4096 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_combinations_l10_1048


namespace NUMINAMATH_CALUDE_karen_tagalongs_sales_l10_1000

/-- The number of cases Karen picked up -/
def num_cases : ℕ := 3

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 12

/-- The total number of boxes Karen sold -/
def total_boxes : ℕ := num_cases * boxes_per_case

theorem karen_tagalongs_sales : total_boxes = 36 := by
  sorry

end NUMINAMATH_CALUDE_karen_tagalongs_sales_l10_1000


namespace NUMINAMATH_CALUDE_largest_box_size_l10_1007

theorem largest_box_size (olivia noah liam : ℕ) 
  (h_olivia : olivia = 48)
  (h_noah : noah = 60)
  (h_liam : liam = 72) :
  Nat.gcd olivia (Nat.gcd noah liam) = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_box_size_l10_1007


namespace NUMINAMATH_CALUDE_nearest_whole_number_l10_1023

theorem nearest_whole_number (x : ℝ) (h : x = 24567.4999997) :
  round x = 24567 := by sorry

end NUMINAMATH_CALUDE_nearest_whole_number_l10_1023


namespace NUMINAMATH_CALUDE_circle_sum_formula_l10_1075

/-- The sum of numbers on a circle after n divisions -/
def circle_sum (n : ℕ) : ℝ :=
  2 * 3^n

/-- Theorem stating the sum of numbers on the circle after n divisions -/
theorem circle_sum_formula (n : ℕ) : circle_sum n = 2 * 3^n := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_formula_l10_1075


namespace NUMINAMATH_CALUDE_trigonometric_product_equals_one_l10_1064

theorem trigonometric_product_equals_one : 
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equals_one_l10_1064


namespace NUMINAMATH_CALUDE_total_segment_length_l10_1072

-- Define the grid dimensions
def grid_width : ℕ := 5
def grid_height : ℕ := 6

-- Define the number of unit squares
def total_squares : ℕ := 30

-- Define the lengths of the six line segments
def segment_lengths : List ℕ := [5, 1, 4, 2, 3, 3]

-- Theorem statement
theorem total_segment_length :
  grid_width = 5 ∧ 
  grid_height = 6 ∧ 
  total_squares = 30 ∧ 
  segment_lengths = [5, 1, 4, 2, 3, 3] →
  List.sum segment_lengths = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_segment_length_l10_1072


namespace NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_implies_negative_k_and_b_l10_1089

/-- A linear function that does not pass through the first quadrant -/
structure LinearFunctionNotInFirstQuadrant where
  k : ℝ
  b : ℝ
  not_in_first_quadrant : ∀ x y : ℝ, y = k * x + b → ¬(x > 0 ∧ y > 0)

/-- Theorem: If a linear function y = kx + b does not pass through the first quadrant, then k < 0 and b < 0 -/
theorem linear_function_not_in_first_quadrant_implies_negative_k_and_b 
  (f : LinearFunctionNotInFirstQuadrant) : f.k < 0 ∧ f.b < 0 := by
  sorry


end NUMINAMATH_CALUDE_linear_function_not_in_first_quadrant_implies_negative_k_and_b_l10_1089


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l10_1097

/-- Given vectors a and b in R^2, prove that their difference has magnitude 5 -/
theorem vector_difference_magnitude (a b : ℝ × ℝ) : 
  a = (2, 1) → b = (-2, 4) → ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l10_1097


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l10_1008

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, x^2 + 2*x = 0 ↔ x = x₁ ∨ x = x₂) ∧ x₁ = 0 ∧ x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l10_1008


namespace NUMINAMATH_CALUDE_christina_transfer_l10_1009

/-- The amount Christina transferred out of her bank account -/
def amount_transferred (initial_balance final_balance : ℕ) : ℕ :=
  initial_balance - final_balance

/-- Theorem stating that Christina transferred $69 out of her bank account -/
theorem christina_transfer :
  amount_transferred 27004 26935 = 69 := by
  sorry

end NUMINAMATH_CALUDE_christina_transfer_l10_1009


namespace NUMINAMATH_CALUDE_double_then_half_sixteen_l10_1039

theorem double_then_half_sixteen : 
  let initial_number := 16
  let doubled := initial_number * 2
  let halved := doubled / 2
  halved = 2^4 := by sorry

end NUMINAMATH_CALUDE_double_then_half_sixteen_l10_1039


namespace NUMINAMATH_CALUDE_three_digit_self_repeating_powers_l10_1080

theorem three_digit_self_repeating_powers : 
  {N : ℕ | 100 ≤ N ∧ N < 1000 ∧ ∀ k : ℕ, k ≥ 1 → N^k % 1000 = N} = {376, 625} := by
  sorry

end NUMINAMATH_CALUDE_three_digit_self_repeating_powers_l10_1080


namespace NUMINAMATH_CALUDE_wrapping_paper_division_l10_1031

theorem wrapping_paper_division (total_used : ℚ) (num_presents : ℕ) (paper_per_present : ℚ) :
  total_used = 5 / 8 →
  num_presents = 5 →
  paper_per_present * num_presents = total_used →
  paper_per_present = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_division_l10_1031


namespace NUMINAMATH_CALUDE_no_solution_exists_l10_1095

theorem no_solution_exists : ¬ ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a + b ≠ 0 ∧ 1 / a + 2 / b = 3 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l10_1095


namespace NUMINAMATH_CALUDE_quadratic_curve_focal_distance_l10_1099

theorem quadratic_curve_focal_distance (a : ℝ) (h1 : a ≠ 0) :
  (∃ (x y : ℝ), x^2 + a*y^2 + a^2 = 0) ∧
  (∃ (c : ℝ), c = 2 ∧ c^2 = a^2 + (-a)) →
  a = (1 - Real.sqrt 17) / 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_curve_focal_distance_l10_1099


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l10_1044

theorem sin_cos_sixth_power_sum (θ : Real) (h : Real.cos (2 * θ) = 1 / 5) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_sum_l10_1044


namespace NUMINAMATH_CALUDE_volleyball_team_combinations_l10_1022

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def starters : ℕ := 6

def choose_starters (n k : ℕ) : ℕ := Nat.choose n k

theorem volleyball_team_combinations : 
  choose_starters (total_players - quadruplets) starters + 
  quadruplets * choose_starters (total_players - quadruplets) (starters - 1) + 
  Nat.choose quadruplets 2 * choose_starters (total_players - quadruplets) (starters - 2) = 7062 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_combinations_l10_1022


namespace NUMINAMATH_CALUDE_cookies_per_day_l10_1057

-- Define the problem parameters
def cookie_cost : ℕ := 19
def total_spent : ℕ := 2356
def days_in_march : ℕ := 31

-- Define the theorem
theorem cookies_per_day :
  (total_spent / cookie_cost) / days_in_march = 4 := by
  sorry


end NUMINAMATH_CALUDE_cookies_per_day_l10_1057


namespace NUMINAMATH_CALUDE_triangle_problem_l10_1025

noncomputable def f (x φ : Real) : Real := 2 * Real.sin x * (Real.cos (φ / 2))^2 + Real.cos x * Real.sin φ - Real.sin x

theorem triangle_problem (φ : Real) (A B C : Real) (a b c : Real) :
  (0 < φ) ∧ (φ < Real.pi) ∧
  (∀ x, f x φ ≥ f Real.pi φ) ∧
  (a = 1) ∧ (b = Real.sqrt 2) ∧
  (f A φ = Real.sqrt 3 / 2) ∧
  (a / Real.sin A = b / Real.sin B) ∧
  (A + B + C = Real.pi) →
  (φ = Real.pi / 2) ∧
  (∀ x, f x φ = Real.cos x) ∧
  (C = 7 * Real.pi / 12) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l10_1025


namespace NUMINAMATH_CALUDE_closed_polygonal_line_links_divisible_by_four_l10_1050

/-- Represents a link in the polygonal line -/
structure Link where
  direction : Bool  -- True for horizontal, False for vertical
  length : Nat
  is_odd : Odd length

/-- Represents a closed polygonal line on a square grid -/
structure PolygonalLine where
  links : List Link
  is_closed : links.length > 0

/-- The main theorem to prove -/
theorem closed_polygonal_line_links_divisible_by_four (p : PolygonalLine) :
  4 ∣ p.links.length :=
sorry

end NUMINAMATH_CALUDE_closed_polygonal_line_links_divisible_by_four_l10_1050


namespace NUMINAMATH_CALUDE_polynomial_coefficient_B_l10_1082

-- Define the polynomial
def polynomial (z A B C D : ℤ) : ℤ := z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 20

-- Define the property that all roots are positive integers
def all_roots_positive_integers (p : ℤ → ℤ) : Prop :=
  ∀ r : ℤ, p r = 0 → r > 0

-- State the theorem
theorem polynomial_coefficient_B :
  ∀ A B C D : ℤ,
  all_roots_positive_integers (polynomial · A B C D) →
  B = -160 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_B_l10_1082


namespace NUMINAMATH_CALUDE_range_of_a_l10_1043

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 5, log10 (x^2 + a*x) = 1) → 
  a ∈ Set.Icc (-3) 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l10_1043


namespace NUMINAMATH_CALUDE_characterize_u_l10_1038

/-- A function is strictly monotonic if it preserves the order relation -/
def StrictlyMonotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The main theorem statement -/
theorem characterize_u (u : ℝ → ℝ) :
  (∃ f : ℝ → ℝ, StrictlyMonotonic f ∧
    (∀ x y : ℝ, f (x + y) = f x * u y + f y)) →
  (∃ a : ℝ, ∀ x : ℝ, u x = Real.exp (a * x)) :=
by sorry

end NUMINAMATH_CALUDE_characterize_u_l10_1038


namespace NUMINAMATH_CALUDE_rita_calculation_l10_1029

theorem rita_calculation (a b c : ℝ) 
  (h1 : a - (2*b - 3*c) = 23) 
  (h2 : a - 2*b - 3*c = 5) : 
  a - 2*b = 14 := by
sorry

end NUMINAMATH_CALUDE_rita_calculation_l10_1029


namespace NUMINAMATH_CALUDE_vertices_form_hyperbola_branch_l10_1020

/-- Given a real number k and a constant c, the set of vertices (x_t, y_t) of the parabola
    y = t^2 x^2 + 2ktx + c for varying t forms one branch of a hyperbola. -/
theorem vertices_form_hyperbola_branch (k : ℝ) (c : ℝ) :
  ∃ (A B C D : ℝ), A ≠ 0 ∧
    (∀ x_t y_t : ℝ, x_t ≠ 0 →
      (∃ t : ℝ, y_t = t^2 * x_t^2 + 2*k*t*x_t + c ∧
                x_t = -k/t) →
      A * x_t * y_t + B * x_t + C * y_t + D = 0) :=
sorry

end NUMINAMATH_CALUDE_vertices_form_hyperbola_branch_l10_1020


namespace NUMINAMATH_CALUDE_simplify_expression_l10_1018

theorem simplify_expression : (2^5 + 4^4) * (2^2 - (-2)^3)^8 = 123876479488 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l10_1018


namespace NUMINAMATH_CALUDE_factor_expression_l10_1036

theorem factor_expression (x : ℝ) : 36 * x^2 + 24 * x = 12 * x * (3 * x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l10_1036


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l10_1087

/-- The area of an equilateral triangle with altitude √15 is 5√3 square units. -/
theorem equilateral_triangle_area (h : ℝ) (altitude_eq : h = Real.sqrt 15) :
  let side : ℝ := 2 * Real.sqrt 5
  let area : ℝ := (side * h) / 2
  area = 5 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l10_1087


namespace NUMINAMATH_CALUDE_special_triangle_sides_l10_1001

/-- A triangle ABC with specific properties -/
structure SpecialTriangle where
  /-- Vertex B of the triangle -/
  B : ℝ × ℝ
  /-- Equation of the altitude on side AB: ax + by + c = 0 -/
  altitude : ℝ × ℝ × ℝ
  /-- Equation of the angle bisector of angle A: dx + ey + f = 0 -/
  angle_bisector : ℝ × ℝ × ℝ

/-- Theorem about the equations of sides in a special triangle -/
theorem special_triangle_sides 
  (t : SpecialTriangle) 
  (h1 : t.B = (-2, 0))
  (h2 : t.altitude = (1, 3, -26))
  (h3 : t.angle_bisector = (1, 1, -2)) :
  ∃ (AB AC : ℝ × ℝ × ℝ),
    AB = (3, -1, 6) ∧ 
    AC = (1, -3, 10) := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sides_l10_1001


namespace NUMINAMATH_CALUDE_arithmetic_sequence_value_l10_1076

def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem arithmetic_sequence_value :
  ∀ a : ℝ, is_arithmetic_sequence 2 a 10 → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_value_l10_1076


namespace NUMINAMATH_CALUDE_chord_equation_l10_1068

/-- The equation of a line containing a chord of the ellipse x^2/2 + y^2 = 1,
    passing through and bisected by the point (1/2, 1/2) -/
theorem chord_equation (x y : ℝ) : 
  (∃ (x1 y1 x2 y2 : ℝ),
    -- Ellipse equation
    x1^2 / 2 + y1^2 = 1 ∧ 
    x2^2 / 2 + y2^2 = 1 ∧
    -- Point P is on the ellipse
    (1/2)^2 / 2 + (1/2)^2 = 1 ∧
    -- P is the midpoint of the chord
    (x1 + x2) / 2 = 1/2 ∧
    (y1 + y2) / 2 = 1/2 ∧
    -- The line passes through P
    y - 1/2 = (y - 1/2) / (x - 1/2) * (x - 1/2)) →
  2*x + 4*y - 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l10_1068


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l10_1062

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -2}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l10_1062


namespace NUMINAMATH_CALUDE_solution_to_equation_l10_1024

theorem solution_to_equation (x : ℝ) : 
  (((32 : ℝ) ^ (x - 2) / (8 : ℝ) ^ (x - 2)) ^ 2 = (1024 : ℝ) ^ (2 * x - 1)) ↔ x = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l10_1024


namespace NUMINAMATH_CALUDE_blue_marbles_total_l10_1021

/-- The number of blue marbles Jason has -/
def jason_blue : ℕ := 44

/-- The number of blue marbles Tom has -/
def tom_blue : ℕ := 24

/-- The total number of blue marbles Jason and Tom have together -/
def total_blue : ℕ := jason_blue + tom_blue

theorem blue_marbles_total : total_blue = 68 := by sorry

end NUMINAMATH_CALUDE_blue_marbles_total_l10_1021


namespace NUMINAMATH_CALUDE_horner_method_result_l10_1074

def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

theorem horner_method_result : f (-4) = 220 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_result_l10_1074


namespace NUMINAMATH_CALUDE_range_of_m_l10_1096

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - m ≥ 0

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m*x + 1 > 0

-- Theorem statement
theorem range_of_m (m : ℝ) :
  p m ∧ q m → -2 < m ∧ m ≤ 1 :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_m_l10_1096


namespace NUMINAMATH_CALUDE_circle_radius_theorem_l10_1014

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the circle
structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define the points E and F on the sides of the triangle
def E (triangle : Triangle) : Point := sorry
def F (triangle : Triangle) : Point := sorry

-- Define the angles
def angle_ABC (triangle : Triangle) : ℝ := sorry
def angle_AEC (triangle : Triangle) (circle : Circle) : ℝ := sorry
def angle_BAF (triangle : Triangle) (circle : Circle) : ℝ := sorry

-- Define the length of AC
def length_AC (triangle : Triangle) : ℝ := sorry

-- State the theorem
theorem circle_radius_theorem (triangle : Triangle) (circle : Circle) :
  angle_ABC triangle = 72 →
  angle_AEC triangle circle = 5 * angle_BAF triangle circle →
  length_AC triangle = 6 →
  circle.radius = 3 := by sorry

end NUMINAMATH_CALUDE_circle_radius_theorem_l10_1014


namespace NUMINAMATH_CALUDE_base7_sum_property_l10_1081

/-- A digit in base 7 is a natural number less than 7 -/
def Digit7 : Type := { n : ℕ // n < 7 }

/-- Convert a three-digit number in base 7 to its decimal representation -/
def toDecimal (d e f : Digit7) : ℕ := 49 * d.val + 7 * e.val + f.val

/-- The sum of three permutations of a three-digit number in base 7 -/
def sumPermutations (d e f : Digit7) : ℕ :=
  toDecimal d e f + toDecimal e f d + toDecimal f d e

theorem base7_sum_property (d e f : Digit7) 
  (h_distinct : d ≠ e ∧ d ≠ f ∧ e ≠ f) 
  (h_nonzero : d.val ≠ 0 ∧ e.val ≠ 0 ∧ f.val ≠ 0)
  (h_sum : sumPermutations d e f = 400 * d.val) :
  e.val + f.val = 6 :=
sorry

end NUMINAMATH_CALUDE_base7_sum_property_l10_1081


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_l10_1005

/-- Given two vectors a and b, where a is parallel to b, 
    prove that the minimum value of 3/x + 2/y is 8 -/
theorem min_value_parallel_vectors (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, y - 1)
  (∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1) →  -- parallelism condition
  (3 / x + 2 / y) ≥ 8 ∧ 
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 2 / y₀ = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_l10_1005


namespace NUMINAMATH_CALUDE_fib_2006_mod_10_l10_1028

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_2006_mod_10 : fib 2006 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fib_2006_mod_10_l10_1028


namespace NUMINAMATH_CALUDE_cos_54_degrees_l10_1051

theorem cos_54_degrees : Real.cos (54 * π / 180) = Real.sqrt ((3 + Real.sqrt 5) / 8) := by
  sorry

end NUMINAMATH_CALUDE_cos_54_degrees_l10_1051


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l10_1079

theorem divisibility_equivalence (n : ℤ) : 
  let A := n % 1000
  let B := n / 1000
  let k := A - B
  (7 ∣ n ∨ 11 ∣ n ∨ 13 ∣ n) ↔ (7 ∣ k ∨ 11 ∣ k ∨ 13 ∣ k) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l10_1079


namespace NUMINAMATH_CALUDE_area_ratio_ACEG_to_hexadecagon_l10_1027

/-- Regular hexadecagon with vertices ABCDEFGHIJKLMNOP -/
structure RegularHexadecagon where
  vertices : Fin 16 → ℝ × ℝ
  is_regular : sorry -- Additional properties to ensure it's a regular hexadecagon

/-- Area of a regular hexadecagon -/
def area_hexadecagon (h : RegularHexadecagon) : ℝ := sorry

/-- Quadrilateral ACEG formed by connecting every fourth vertex of the hexadecagon -/
def quadrilateral_ACEG (h : RegularHexadecagon) : Set (ℝ × ℝ) := sorry

/-- Area of quadrilateral ACEG -/
def area_ACEG (h : RegularHexadecagon) : ℝ := sorry

/-- The main theorem: The ratio of the area of ACEG to the area of the hexadecagon is √2/2 -/
theorem area_ratio_ACEG_to_hexadecagon (h : RegularHexadecagon) :
  (area_ACEG h) / (area_hexadecagon h) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_ACEG_to_hexadecagon_l10_1027


namespace NUMINAMATH_CALUDE_complex_arithmetic_result_l10_1045

theorem complex_arithmetic_result : 
  ((2 - 3*I) + (4 + 6*I)) * (-1 + 2*I) = -12 + 9*I :=
by sorry

end NUMINAMATH_CALUDE_complex_arithmetic_result_l10_1045


namespace NUMINAMATH_CALUDE_decoration_sets_count_l10_1056

/-- Represents a decoration set with balloons and ribbons -/
structure DecorationSet where
  balloons : ℕ
  ribbons : ℕ

/-- The cost of a decoration set -/
def cost (set : DecorationSet) : ℕ := 4 * set.balloons + 6 * set.ribbons

/-- Predicate for valid decoration sets -/
def isValid (set : DecorationSet) : Prop :=
  cost set = 120 ∧ Even set.balloons

theorem decoration_sets_count :
  ∃! (sets : Finset DecorationSet), 
    (∀ s ∈ sets, isValid s) ∧ 
    (∀ s, isValid s → s ∈ sets) ∧
    Finset.card sets = 2 := by
  sorry

end NUMINAMATH_CALUDE_decoration_sets_count_l10_1056


namespace NUMINAMATH_CALUDE_sum_of_coefficients_without_x_l10_1094

theorem sum_of_coefficients_without_x (x y : ℝ) : 
  (fun x y => (1 - x - 5*y)^5) 0 1 = -1024 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_without_x_l10_1094


namespace NUMINAMATH_CALUDE_sequence_2013_value_l10_1013

def is_valid_sequence (a : ℕ → ℤ) : Prop :=
  ∀ (p k : ℕ), Nat.Prime p → k > 0 → a (p * k + 1) = p * a k - 3 * a p + 13

theorem sequence_2013_value (a : ℕ → ℤ) (h : is_valid_sequence a) : a 2013 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sequence_2013_value_l10_1013


namespace NUMINAMATH_CALUDE_amelia_position_100_l10_1083

/-- Represents a position on the coordinate plane -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a direction -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Defines Amelia's movement pattern -/
def ameliaMove (n : Nat) : Position :=
  sorry

/-- Theorem stating Amelia's position at p₁₀₀ -/
theorem amelia_position_100 : ameliaMove 100 = Position.mk 0 19 := by
  sorry

end NUMINAMATH_CALUDE_amelia_position_100_l10_1083


namespace NUMINAMATH_CALUDE_book_sale_price_l10_1098

-- Define the total number of books
def total_books : ℕ := 150

-- Define the number of unsold books
def unsold_books : ℕ := 50

-- Define the total amount received
def total_amount : ℕ := 500

-- Define the fraction of books sold
def fraction_sold : ℚ := 2/3

-- Theorem to prove
theorem book_sale_price :
  let sold_books := total_books - unsold_books
  let price_per_book := total_amount / sold_books
  fraction_sold * total_books = sold_books ∧
  price_per_book = 5 := by
sorry

end NUMINAMATH_CALUDE_book_sale_price_l10_1098


namespace NUMINAMATH_CALUDE_set_one_two_three_not_triangle_l10_1066

/-- Triangle Inequality Theorem: A set of three positive real numbers a, b, and c can form a triangle
    if and only if the sum of any two is greater than the third. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The set {1, 2, 3} cannot form a triangle. -/
theorem set_one_two_three_not_triangle :
  ¬ can_form_triangle 1 2 3 := by
  sorry

#check set_one_two_three_not_triangle

end NUMINAMATH_CALUDE_set_one_two_three_not_triangle_l10_1066


namespace NUMINAMATH_CALUDE_jack_needs_five_rocks_l10_1055

-- Define the weights and rock weight
def jack_weight : ℕ := 60
def anna_weight : ℕ := 40
def rock_weight : ℕ := 4

-- Define the function to calculate the number of rocks
def num_rocks (jack_w anna_w rock_w : ℕ) : ℕ :=
  (jack_w - anna_w) / rock_w

-- Theorem statement
theorem jack_needs_five_rocks :
  num_rocks jack_weight anna_weight rock_weight = 5 := by
  sorry

end NUMINAMATH_CALUDE_jack_needs_five_rocks_l10_1055


namespace NUMINAMATH_CALUDE_fiona_casey_hoodies_l10_1012

/-- The number of hoodies Fiona and Casey own together -/
def total_hoodies (fiona_hoodies : ℕ) (casey_extra_hoodies : ℕ) : ℕ :=
  fiona_hoodies + (fiona_hoodies + casey_extra_hoodies)

/-- Theorem stating that Fiona and Casey own 8 hoodies in total -/
theorem fiona_casey_hoodies : total_hoodies 3 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fiona_casey_hoodies_l10_1012


namespace NUMINAMATH_CALUDE_imo_problem_6_l10_1026

theorem imo_problem_6 (n : ℕ) (hn : n ≥ 2) :
  (∀ k : ℕ, k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) →
  (∀ k : ℕ, k ≤ n - 2 → Nat.Prime (k^2 + k + n)) := by
  sorry

end NUMINAMATH_CALUDE_imo_problem_6_l10_1026


namespace NUMINAMATH_CALUDE_average_problem_l10_1070

theorem average_problem (x : ℝ) : (2 + 76 + x) / 3 = 5 → x = -63 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l10_1070
