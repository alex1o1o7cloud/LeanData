import Mathlib

namespace NUMINAMATH_CALUDE_fraction_simplification_l31_3105

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) : (5 * x) / (x - 1) - 5 / (x - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l31_3105


namespace NUMINAMATH_CALUDE_probability_continuous_stripe_is_two_over_81_l31_3153

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron :=
  (faces : Fin 4 → Face)

/-- Represents a face of the tetrahedron -/
structure Face :=
  (vertices : Fin 3 → Vertex)
  (stripe_start : Vertex)

/-- Represents a vertex of a face -/
inductive Vertex
| A | B | C

/-- Represents a stripe configuration on the tetrahedron -/
def StripeConfiguration := RegularTetrahedron

/-- Checks if a stripe configuration forms a continuous stripe around the tetrahedron -/
def is_continuous_stripe (config : StripeConfiguration) : Prop :=
  sorry

/-- The total number of possible stripe configurations -/
def total_configurations : ℕ := 81

/-- The number of stripe configurations that form a continuous stripe -/
def continuous_stripe_configurations : ℕ := 2

/-- The probability of a continuous stripe encircling the tetrahedron -/
def probability_continuous_stripe : ℚ :=
  continuous_stripe_configurations / total_configurations

theorem probability_continuous_stripe_is_two_over_81 :
  probability_continuous_stripe = 2 / 81 :=
sorry

end NUMINAMATH_CALUDE_probability_continuous_stripe_is_two_over_81_l31_3153


namespace NUMINAMATH_CALUDE_sin_2alpha_l31_3133

theorem sin_2alpha (α : ℝ) (h : Real.sin (α + π/4) = 1/3) : Real.sin (2*α) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_l31_3133


namespace NUMINAMATH_CALUDE_ninth_term_of_geometric_sequence_l31_3132

/-- A geometric sequence of positive real numbers -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem ninth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_fifth : a 5 = 32)
  (h_eleventh : a 11 = 2) :
  a 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_geometric_sequence_l31_3132


namespace NUMINAMATH_CALUDE_athul_rowing_problem_l31_3199

/-- Athul's rowing problem -/
theorem athul_rowing_problem 
  (v : ℝ) -- Athul's speed in still water (km/h)
  (d : ℝ) -- Distance rowed upstream (km)
  (h1 : v + 1 = 24 / 4) -- Downstream speed equation
  (h2 : v - 1 = d / 4) -- Upstream speed equation
  : d = 16 := by
  sorry

end NUMINAMATH_CALUDE_athul_rowing_problem_l31_3199


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l31_3161

theorem cubic_equation_solution :
  ∃ y : ℝ, y > 0 ∧ 5 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + y^(1/3) ∧ y = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l31_3161


namespace NUMINAMATH_CALUDE_trigonometric_identity_l31_3186

theorem trigonometric_identity (α : Real) : 
  Real.sin α ^ 2 + Real.cos (π / 6 + α) ^ 2 + Real.sin α * Real.cos (π / 6 + α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l31_3186


namespace NUMINAMATH_CALUDE_existence_of_irrational_powers_with_integer_result_l31_3129

theorem existence_of_irrational_powers_with_integer_result :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ Irrational a ∧ Irrational b ∧ ∃ (n : ℤ), a^b = n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_irrational_powers_with_integer_result_l31_3129


namespace NUMINAMATH_CALUDE_distance_traveled_l31_3156

/-- 
Given a speed of 20 km/hr and a time of 8 hr, prove that the distance traveled is 160 km.
-/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 20) (h2 : time = 8) :
  speed * time = 160 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l31_3156


namespace NUMINAMATH_CALUDE_anthony_transaction_percentage_l31_3102

theorem anthony_transaction_percentage (mabel_transactions cal_transactions anthony_transactions jade_transactions : ℕ) :
  mabel_transactions = 90 →
  cal_transactions = (2 : ℚ) / 3 * anthony_transactions →
  jade_transactions = cal_transactions + 16 →
  jade_transactions = 82 →
  (anthony_transactions : ℚ) / mabel_transactions - 1 = (1 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_anthony_transaction_percentage_l31_3102


namespace NUMINAMATH_CALUDE_water_cube_product_l31_3197

/-- Definition of a water cube number -/
def V (a b c : ℝ) : ℝ := a^3 + b^3 + c^3 - 3*a*b*c

/-- Theorem: The product of two water cube numbers is a water cube number -/
theorem water_cube_product (a b c x y z : ℝ) :
  V a b c * V x y z = V (a*x + b*y + c*z) (b*x + c*y + a*z) (c*x + a*y + b*z) := by
  sorry

end NUMINAMATH_CALUDE_water_cube_product_l31_3197


namespace NUMINAMATH_CALUDE_optimal_transport_solution_l31_3171

/-- Represents a vehicle type with its carrying capacity and freight cost. -/
structure VehicleType where
  capacity : ℕ
  cost : ℕ

/-- Represents the transportation problem. -/
structure TransportProblem where
  totalVegetables : ℕ
  totalVehicles : ℕ
  vehicleA : VehicleType
  vehicleB : VehicleType
  vehicleC : VehicleType

/-- Represents a solution to the transportation problem. -/
structure TransportSolution where
  numA : ℕ
  numB : ℕ
  numC : ℕ
  totalCost : ℕ

/-- Checks if a solution is valid for a given problem. -/
def isValidSolution (problem : TransportProblem) (solution : TransportSolution) : Prop :=
  solution.numA + solution.numB + solution.numC = problem.totalVehicles ∧
  solution.numA * problem.vehicleA.capacity +
  solution.numB * problem.vehicleB.capacity +
  solution.numC * problem.vehicleC.capacity ≥ problem.totalVegetables ∧
  solution.totalCost = solution.numA * problem.vehicleA.cost +
                       solution.numB * problem.vehicleB.cost +
                       solution.numC * problem.vehicleC.cost

/-- Theorem stating the optimal solution for the given problem. -/
theorem optimal_transport_solution (problem : TransportProblem)
  (h1 : problem.totalVegetables = 240)
  (h2 : problem.totalVehicles = 16)
  (h3 : problem.vehicleA = ⟨10, 800⟩)
  (h4 : problem.vehicleB = ⟨16, 1000⟩)
  (h5 : problem.vehicleC = ⟨20, 1200⟩) :
  ∃ (solution : TransportSolution),
    isValidSolution problem solution ∧
    solution.numA = 4 ∧
    solution.numB = 10 ∧
    solution.numC = 2 ∧
    solution.totalCost = 15600 ∧
    (∀ (otherSolution : TransportSolution),
      isValidSolution problem otherSolution →
      otherSolution.totalCost ≥ solution.totalCost) :=
sorry

end NUMINAMATH_CALUDE_optimal_transport_solution_l31_3171


namespace NUMINAMATH_CALUDE_expression_value_l31_3141

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 + z^2 + 2*x*y*z = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l31_3141


namespace NUMINAMATH_CALUDE_sum_of_x_y_z_l31_3170

theorem sum_of_x_y_z (x y z : ℝ) : y = 3*x → z = 2*y → x + y + z = 10*x := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_y_z_l31_3170


namespace NUMINAMATH_CALUDE_school_height_ratio_l31_3168

theorem school_height_ratio (total_avg : ℝ) (female_avg : ℝ) (male_avg : ℝ)
  (h_total : total_avg = 180)
  (h_female : female_avg = 170)
  (h_male : male_avg = 182) :
  ∃ (m w : ℝ), m > 0 ∧ w > 0 ∧ m / w = 5 ∧
    male_avg * m + female_avg * w = total_avg * (m + w) :=
by
  sorry

end NUMINAMATH_CALUDE_school_height_ratio_l31_3168


namespace NUMINAMATH_CALUDE_soccer_balls_count_l31_3185

/-- The cost of a football in dollars -/
def football_cost : ℝ := 35

/-- The cost of a soccer ball in dollars -/
def soccer_ball_cost : ℝ := 50

/-- The number of soccer balls in the first set -/
def soccer_balls_in_first_set : ℕ := 1

theorem soccer_balls_count : 
  3 * football_cost + soccer_balls_in_first_set * soccer_ball_cost = 155 ∧
  2 * football_cost + 3 * soccer_ball_cost = 220 →
  soccer_balls_in_first_set = 1 := by
  sorry

end NUMINAMATH_CALUDE_soccer_balls_count_l31_3185


namespace NUMINAMATH_CALUDE_one_third_between_one_eighth_and_one_third_l31_3167

def one_third_between (a b : ℚ) : ℚ :=
  (1 - 1/3) * a + 1/3 * b

theorem one_third_between_one_eighth_and_one_third :
  one_third_between (1/8) (1/3) = 7/36 := by
  sorry

end NUMINAMATH_CALUDE_one_third_between_one_eighth_and_one_third_l31_3167


namespace NUMINAMATH_CALUDE_no_perfect_square_3000_001_l31_3101

theorem no_perfect_square_3000_001 (n : ℕ) : ¬ ∃ k : ℤ, (3 * 10^n + 1 : ℤ) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_3000_001_l31_3101


namespace NUMINAMATH_CALUDE_discount_profit_relation_l31_3164

/-- Proves that if an item sold at a 10% discount yields a gross profit of 20% of the cost,
    then selling the item without a discount would yield a gross profit of 33.33% of the cost. -/
theorem discount_profit_relation (cost : ℝ) (original_price : ℝ) :
  original_price * 0.9 = cost * 1.2 →
  (original_price - cost) / cost * 100 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_discount_profit_relation_l31_3164


namespace NUMINAMATH_CALUDE_group_size_l31_3108

/-- An international group consisting of Chinese, Americans, and Australians -/
structure InternationalGroup where
  chinese : ℕ
  americans : ℕ
  australians : ℕ

/-- The total number of people in the group -/
def InternationalGroup.total (group : InternationalGroup) : ℕ :=
  group.chinese + group.americans + group.australians

theorem group_size (group : InternationalGroup) 
  (h1 : group.chinese = 22)
  (h2 : group.americans = 16)
  (h3 : group.australians = 11) :
  group.total = 49 := by
  sorry

#check group_size

end NUMINAMATH_CALUDE_group_size_l31_3108


namespace NUMINAMATH_CALUDE_sum_of_integers_l31_3194

theorem sum_of_integers (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  (6 - a) * (6 - b) * (6 - c) * (6 - d) * (6 - e) = 45 →
  a + b + c + d + e = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l31_3194


namespace NUMINAMATH_CALUDE_jade_savings_l31_3136

def monthly_income : ℝ := 1600

def living_expenses_ratio : ℝ := 0.75
def insurance_ratio : ℝ := 0.2

def savings (income : ℝ) (living_ratio : ℝ) (insurance_ratio : ℝ) : ℝ :=
  income - (income * living_ratio) - (income * insurance_ratio)

theorem jade_savings : savings monthly_income living_expenses_ratio insurance_ratio = 80 := by
  sorry

end NUMINAMATH_CALUDE_jade_savings_l31_3136


namespace NUMINAMATH_CALUDE_stating_final_values_theorem_l31_3176

/-- 
Given initial values a = 1 and b = 3, this function performs the operations
a = a + b and b = b * a, and returns the final values of a and b.
-/
def calculate_final_values (a b : ℕ) : ℕ × ℕ :=
  let a' := a + b
  let b' := b * a'
  (a', b')

/-- 
Theorem stating that given initial values a = 1 and b = 3, 
the final values after performing the operations are a = 4 and b = 12.
-/
theorem final_values_theorem : 
  calculate_final_values 1 3 = (4, 12) := by
  sorry

#eval calculate_final_values 1 3

end NUMINAMATH_CALUDE_stating_final_values_theorem_l31_3176


namespace NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l31_3173

/-- Given that 9 oranges weigh the same as 6 apples, prove that 36 oranges weigh the same as 24 apples -/
theorem orange_apple_weight_equivalence 
  (orange_weight apple_weight : ℚ) 
  (h : 9 * orange_weight = 6 * apple_weight) : 
  36 * orange_weight = 24 * apple_weight := by
sorry

end NUMINAMATH_CALUDE_orange_apple_weight_equivalence_l31_3173


namespace NUMINAMATH_CALUDE_prob_at_least_one_female_l31_3159

/-- The probability of selecting at least one female student when randomly choosing 2 students
    from a group of 3 males and 1 female is equal to 1/2. -/
theorem prob_at_least_one_female (total_students : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (team_size : ℕ) (h1 : total_students = male_students + female_students) 
  (h2 : total_students = 4) (h3 : male_students = 3) (h4 : female_students = 1) (h5 : team_size = 2) :
  1 - (Nat.choose male_students team_size : ℚ) / (Nat.choose total_students team_size : ℚ) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_female_l31_3159


namespace NUMINAMATH_CALUDE_k_value_l31_3195

theorem k_value (x : ℝ) (h1 : x ≠ 0) (h2 : 24 / x = k) : k = 24 / x := by
  sorry

end NUMINAMATH_CALUDE_k_value_l31_3195


namespace NUMINAMATH_CALUDE_perpendicular_segments_s_value_l31_3110

/-- Given two perpendicular line segments PQ and PR, where P(4, 2), R(0, 1), and Q(2, s),
    prove that s = 10 -/
theorem perpendicular_segments_s_value (P Q R : ℝ × ℝ) (s : ℝ) : 
  P = (4, 2) →
  R = (0, 1) →
  Q = (2, s) →
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0 →
  s = 10 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_segments_s_value_l31_3110


namespace NUMINAMATH_CALUDE_dvaneft_percentage_range_l31_3135

/-- Represents the share packages in the auction --/
structure SharePackages where
  razneft : ℕ
  dvaneft : ℕ
  trineft : ℕ

/-- Represents the prices of individual shares --/
structure SharePrices where
  razneft : ℝ
  dvaneft : ℝ
  trineft : ℝ

/-- Main theorem about the percentage range of Dvaneft shares --/
theorem dvaneft_percentage_range 
  (packages : SharePackages) 
  (prices : SharePrices) : 
  /- Total shares of Razneft and Dvaneft equals shares of Trineft -/
  (packages.razneft + packages.dvaneft = packages.trineft) → 
  /- Dvaneft package is 3 times cheaper than Razneft package -/
  (3 * prices.dvaneft * packages.dvaneft = prices.razneft * packages.razneft) → 
  /- Total cost of Razneft and Dvaneft equals cost of Trineft -/
  (prices.razneft * packages.razneft + prices.dvaneft * packages.dvaneft = 
   prices.trineft * packages.trineft) → 
  /- Price difference between Razneft and Dvaneft share is between 10,000 and 18,000 -/
  (10000 ≤ prices.razneft - prices.dvaneft ∧ prices.razneft - prices.dvaneft ≤ 18000) → 
  /- Price of Trineft share is between 18,000 and 42,000 -/
  (18000 ≤ prices.trineft ∧ prices.trineft ≤ 42000) → 
  /- The percentage of Dvaneft shares is between 15% and 25% -/
  (15 ≤ 100 * packages.dvaneft / (packages.razneft + packages.dvaneft + packages.trineft) ∧
   100 * packages.dvaneft / (packages.razneft + packages.dvaneft + packages.trineft) ≤ 25) :=
by sorry

end NUMINAMATH_CALUDE_dvaneft_percentage_range_l31_3135


namespace NUMINAMATH_CALUDE_equation_solution_l31_3106

theorem equation_solution : 
  {x : ℝ | x^2 + (x-1)*(x+3) = 3*x + 5} = {-2, 2} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l31_3106


namespace NUMINAMATH_CALUDE_student_count_equation_l31_3189

/-- Represents the number of pens per box for the first type of pen -/
def pens_per_box_1 : ℕ := 8

/-- Represents the number of pens per box for the second type of pen -/
def pens_per_box_2 : ℕ := 12

/-- Represents the number of students without pens if x boxes of type 1 are bought -/
def students_without_pens : ℕ := 3

/-- Represents the number of fewer boxes that can be bought of type 2 -/
def fewer_boxes_type_2 : ℕ := 2

/-- Represents the number of pens left in the last box of type 2 -/
def pens_left_type_2 : ℕ := 1

theorem student_count_equation (x : ℕ) : 
  pens_per_box_1 * x + students_without_pens = 
  pens_per_box_2 * (x - fewer_boxes_type_2) - pens_left_type_2 := by
  sorry

end NUMINAMATH_CALUDE_student_count_equation_l31_3189


namespace NUMINAMATH_CALUDE_pizza_combinations_l31_3104

theorem pizza_combinations : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l31_3104


namespace NUMINAMATH_CALUDE_circle_symmetry_max_k_l31_3115

/-- Given a circle C with center (a,b) and radius 2 passing through (0,2),
    and a line 2x-ky-k=0 with respect to which two points on C are symmetric,
    the maximum value of k is 4√5/5 -/
theorem circle_symmetry_max_k :
  ∀ (a b k : ℝ),
  (a^2 + (b-2)^2 = 4) →  -- circle equation passing through (0,2)
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    ((x₁ - a)^2 + (y₁ - b)^2 = 4) ∧  -- point 1 on circle
    ((x₂ - a)^2 + (y₂ - b)^2 = 4) ∧  -- point 2 on circle
    (2*((x₁ + x₂)/2) - k*((y₁ + y₂)/2) - k = 0) ∧  -- midpoint on line
    (2*a - k*b - k = 0)) →  -- line passes through center
  k ≤ 4 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_max_k_l31_3115


namespace NUMINAMATH_CALUDE_polygon_sides_from_diagonals_l31_3175

theorem polygon_sides_from_diagonals :
  ∃ (n : ℕ), n > 2 ∧ (n * (n - 3)) / 2 = 15 ∧ 
  (∀ (m : ℕ), m > 2 → (m * (m - 3)) / 2 = 15 → m = n) ∧
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_from_diagonals_l31_3175


namespace NUMINAMATH_CALUDE_min_max_perimeter_12_pieces_l31_3155

/-- Represents a rectangular piece with length and width in centimeters -/
structure Piece where
  length : ℝ
  width : ℝ

/-- Represents a collection of identical rectangular pieces -/
structure PieceCollection where
  piece : Piece
  count : ℕ

/-- Calculates the area of a rectangular piece -/
def pieceArea (p : Piece) : ℝ := p.length * p.width

/-- Calculates the total area of a collection of pieces -/
def totalArea (pc : PieceCollection) : ℝ := (pieceArea pc.piece) * pc.count

/-- Calculates the perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: Minimum and maximum perimeter of rectangle formed by 12 pieces of 4x3 cm -/
theorem min_max_perimeter_12_pieces :
  let pieces : PieceCollection := ⟨⟨4, 3⟩, 12⟩
  let area : ℝ := totalArea pieces
  ∃ (min_perim max_perim : ℝ),
    min_perim = 48 ∧
    max_perim = 102 ∧
    (∀ (l w : ℝ), l * w = area → rectanglePerimeter l w ≥ min_perim) ∧
    (∃ (l w : ℝ), l * w = area ∧ rectanglePerimeter l w = max_perim) :=
by sorry

end NUMINAMATH_CALUDE_min_max_perimeter_12_pieces_l31_3155


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l31_3123

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  seventh_term : a 7 = -8
  seventeenth_term : a 17 = -28

/-- The general term formula for the arithmetic sequence -/
def generalTerm (seq : ArithmeticSequence) : ℕ → ℤ := 
  fun n => -2 * n + 6

/-- The sum of the first n terms of the arithmetic sequence -/
def sumOfTerms (seq : ArithmeticSequence) : ℕ → ℤ :=
  fun n => -n^2 + 5*n

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = generalTerm seq n) ∧ 
  (∃ k, ∀ n, sumOfTerms seq n ≤ sumOfTerms seq k) ∧
  (sumOfTerms seq 2 = 6 ∧ sumOfTerms seq 3 = 6) := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l31_3123


namespace NUMINAMATH_CALUDE_circle_center_coordinate_product_l31_3181

/-- Given two points as endpoints of a circle's diameter, 
    calculate the product of the coordinates of the circle's center -/
theorem circle_center_coordinate_product 
  (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) 
  (h1 : p1 = (7, -8)) 
  (h2 : p2 = (-2, 3)) : 
  let center := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (center.1 * center.2) = -25/4 := by
sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_product_l31_3181


namespace NUMINAMATH_CALUDE_meter_to_skips_conversion_l31_3127

/-- Proves that 1 meter is equivalent to (g*b*f*d)/(a*e*h*c) skips given the measurement relationships -/
theorem meter_to_skips_conversion
  (a b c d e f g h : ℝ)
  (hops_to_skips : a * 1 = b)
  (jumps_to_hops : c * 1 = d)
  (leaps_to_jumps : e * 1 = f)
  (leaps_to_meters : g * 1 = h)
  (a_pos : 0 < a)
  (c_pos : 0 < c)
  (e_pos : 0 < e)
  (h_pos : 0 < h) :
  1 = (g * b * f * d) / (a * e * h * c) :=
sorry

end NUMINAMATH_CALUDE_meter_to_skips_conversion_l31_3127


namespace NUMINAMATH_CALUDE_painting_height_l31_3160

theorem painting_height (wall_height wall_width painting_width : ℝ) 
  (wall_area painting_area : ℝ) (painting_percentage : ℝ) :
  wall_height = 5 →
  wall_width = 10 →
  painting_width = 4 →
  painting_percentage = 0.16 →
  wall_area = wall_height * wall_width →
  painting_area = painting_percentage * wall_area →
  painting_area = painting_width * 2 :=
by
  sorry

end NUMINAMATH_CALUDE_painting_height_l31_3160


namespace NUMINAMATH_CALUDE_tangent_line_at_one_max_value_min_value_f_properties_l31_3146

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - x^2 + 1

-- Theorem for the tangent line equation
theorem tangent_line_at_one : 
  ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ (x = 1 ∧ y = f 1) ∨ (y - f 1 = m * (x - 1)) :=
sorry

-- Theorem for the maximum value
theorem max_value :
  ∃ x_max, f x_max = 1 ∧ ∀ x, f x ≤ 1 :=
sorry

-- Theorem for the minimum value
theorem min_value :
  ∃ x_min, f x_min = 23/27 ∧ ∀ x, f x ≥ 23/27 :=
sorry

-- Theorem combining all results
theorem f_properties :
  (∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ (x = 1 ∧ y = f 1) ∨ (y - f 1 = m * (x - 1))) ∧
  (∃ x_max, f x_max = 1 ∧ ∀ x, f x ≤ 1) ∧
  (∃ x_min, f x_min = 23/27 ∧ ∀ x, f x ≥ 23/27) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_max_value_min_value_f_properties_l31_3146


namespace NUMINAMATH_CALUDE_september_birth_percentage_l31_3134

theorem september_birth_percentage (total_authors : ℕ) (september_authors : ℕ) :
  total_authors = 120 →
  september_authors = 15 →
  (september_authors : ℚ) / (total_authors : ℚ) * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_september_birth_percentage_l31_3134


namespace NUMINAMATH_CALUDE_cricket_team_size_l31_3182

theorem cricket_team_size :
  ∀ n : ℕ,
  n > 0 →
  let captain_age : ℕ := 25
  let wicket_keeper_age : ℕ := captain_age + 3
  let team_average_age : ℕ := 22
  let remaining_average_age : ℝ := team_average_age - 1
  (n : ℝ) * team_average_age = captain_age + wicket_keeper_age + (n - 2 : ℝ) * remaining_average_age →
  n = 11 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_size_l31_3182


namespace NUMINAMATH_CALUDE_interval_partition_existence_l31_3137

theorem interval_partition_existence : ∃ (x : Fin 10 → ℝ), 
  (∀ i, x i ∈ Set.Icc (0 : ℝ) 1) ∧ 
  (∀ k : Fin 9, k.val + 2 ≤ 10 → 
    ∀ i j : Fin (k.val + 2), i ≠ j → 
      ⌊(k.val + 2 : ℝ) * x i⌋ ≠ ⌊(k.val + 2 : ℝ) * x j⌋) :=
sorry

end NUMINAMATH_CALUDE_interval_partition_existence_l31_3137


namespace NUMINAMATH_CALUDE_minimum_final_percentage_is_60_percent_l31_3109

def total_points : ℕ := 700
def passing_threshold : ℚ := 70 / 100
def problem_set_score : ℕ := 100
def midterm1_score : ℚ := 60 / 100
def midterm2_score : ℚ := 70 / 100
def midterm3_score : ℚ := 80 / 100
def final_exam_points : ℕ := 300

def minimum_final_percentage (total : ℕ) (threshold : ℚ) (problem_set : ℕ) 
  (mid1 mid2 mid3 : ℚ) (final_points : ℕ) : ℚ :=
  -- Definition of the function to calculate the minimum final percentage
  sorry

theorem minimum_final_percentage_is_60_percent :
  minimum_final_percentage total_points passing_threshold problem_set_score
    midterm1_score midterm2_score midterm3_score final_exam_points = 60 / 100 :=
by sorry

end NUMINAMATH_CALUDE_minimum_final_percentage_is_60_percent_l31_3109


namespace NUMINAMATH_CALUDE_paiges_team_size_l31_3125

theorem paiges_team_size (total_points : ℕ) (paige_points : ℕ) (others_points : ℕ) :
  total_points = 41 →
  paige_points = 11 →
  others_points = 6 →
  ∃ (team_size : ℕ), team_size = (total_points - paige_points) / others_points + 1 ∧ team_size = 6 :=
by sorry

end NUMINAMATH_CALUDE_paiges_team_size_l31_3125


namespace NUMINAMATH_CALUDE_hexagonal_prism_square_pyramid_edge_lengths_l31_3158

/-- Represents a regular hexagonal prism -/
structure HexagonalPrism where
  edge_length : ℝ
  total_edge_length : ℝ
  edge_count : ℕ := 18
  h_total_edge : total_edge_length = edge_length * edge_count

/-- Represents a square pyramid -/
structure SquarePyramid where
  edge_length : ℝ
  total_edge_length : ℝ
  edge_count : ℕ := 8
  h_total_edge : total_edge_length = edge_length * edge_count

/-- Theorem stating the relationship between the total edge lengths of a hexagonal prism and a square pyramid with the same edge length -/
theorem hexagonal_prism_square_pyramid_edge_lengths 
  (h : HexagonalPrism) (p : SquarePyramid) 
  (h_same_edge : h.edge_length = p.edge_length) 
  (h_total_81 : h.total_edge_length = 81) : 
  p.total_edge_length = 36 := by
  sorry

end NUMINAMATH_CALUDE_hexagonal_prism_square_pyramid_edge_lengths_l31_3158


namespace NUMINAMATH_CALUDE_common_root_quadratic_equations_l31_3166

theorem common_root_quadratic_equations (p : ℝ) :
  (p > 0 ∧
   ∃ x : ℝ, (3 * x^2 - 4 * p * x + 9 = 0) ∧ (x^2 - 2 * p * x + 5 = 0)) ↔
  p = 3 :=
by sorry

end NUMINAMATH_CALUDE_common_root_quadratic_equations_l31_3166


namespace NUMINAMATH_CALUDE_max_projection_area_unit_cube_max_projection_area_unit_cube_proof_l31_3143

/-- The maximum area of the orthogonal projection of a unit cube onto any plane -/
theorem max_projection_area_unit_cube : ℝ :=
  2 * Real.sqrt 3

/-- Theorem: The maximum area of the orthogonal projection of a unit cube onto any plane is 2√3 -/
theorem max_projection_area_unit_cube_proof :
  max_projection_area_unit_cube = 2 * Real.sqrt 3 := by
  sorry

#check max_projection_area_unit_cube_proof

end NUMINAMATH_CALUDE_max_projection_area_unit_cube_max_projection_area_unit_cube_proof_l31_3143


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l31_3180

def A : Set ℤ := {1, 2}
def B : Set ℤ := {x | x^2 - 5*x + 4 < 0}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l31_3180


namespace NUMINAMATH_CALUDE_remainder_17_power_53_mod_5_l31_3124

theorem remainder_17_power_53_mod_5 : 17^53 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_power_53_mod_5_l31_3124


namespace NUMINAMATH_CALUDE_expression_upper_bound_l31_3139

theorem expression_upper_bound (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 2) :
  Real.sqrt (a^3 + (2-b)^3) + Real.sqrt (b^3 + (2-c)^3) + 
  Real.sqrt (c^3 + (2-d)^3) + Real.sqrt (d^3 + (3-a)^3) ≤ 5 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_upper_bound_l31_3139


namespace NUMINAMATH_CALUDE_isosceles_triangle_circumradius_l31_3113

/-- Given an isosceles triangle with base angle α, if the altitude to the base exceeds
    the radius of the inscribed circle by m, then the radius of the circumscribed circle
    is m / (4 * sin²(α/2)). -/
theorem isosceles_triangle_circumradius (α m : ℝ) (h_α : 0 < α ∧ α < π) (h_m : m > 0) :
  let altitude := inradius + m
  circumradius = m / (4 * Real.sin (α / 2) ^ 2) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_circumradius_l31_3113


namespace NUMINAMATH_CALUDE_inequality_solution_set_l31_3112

theorem inequality_solution_set (a : ℝ) :
  (∀ x, (a - x) * (x - 1) < 0 ↔ 
    (a > 1 ∧ (x > a ∨ x < 1)) ∨
    (a < 1 ∧ (x > 1 ∨ x < a)) ∨
    (a = 1 ∧ x ≠ 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l31_3112


namespace NUMINAMATH_CALUDE_ending_number_is_300_l31_3169

theorem ending_number_is_300 (ending_number : ℕ) : 
  (∃ (multiples : List ℕ), 
    multiples.length = 67 ∧ 
    (∀ n ∈ multiples, n % 3 = 0) ∧
    (∀ n ∈ multiples, 100 ≤ n ∧ n ≤ ending_number) ∧
    (∀ n, 100 ≤ n ∧ n ≤ ending_number ∧ n % 3 = 0 → n ∈ multiples)) →
  ending_number = 300 := by
sorry

end NUMINAMATH_CALUDE_ending_number_is_300_l31_3169


namespace NUMINAMATH_CALUDE_a_minus_b_equals_plus_minus_eight_l31_3111

theorem a_minus_b_equals_plus_minus_eight (a b : ℚ) : 
  (|a| = 5) → (|b| = 3) → (a * b < 0) → (a - b = 8 ∨ a - b = -8) := by
  sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_plus_minus_eight_l31_3111


namespace NUMINAMATH_CALUDE_functional_equation_solution_l31_3177

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x - (1/2) * f (x/2) = x^2

/-- The theorem stating that the function satisfying the equation is f(x) = (8/7) * x^2 -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) :
  f = fun x ↦ (8/7) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l31_3177


namespace NUMINAMATH_CALUDE_max_vertex_product_sum_l31_3150

/-- The set of numbers that can be assigned to the faces of the cube -/
def CubeNumbers : Finset ℕ := {1, 2, 3, 4, 8, 9}

/-- A valid assignment of numbers to the faces of a cube -/
structure CubeAssignment where
  faces : Fin 6 → ℕ
  valid : ∀ i, faces i ∈ CubeNumbers
  distinct : ∀ i j, i ≠ j → faces i ≠ faces j

/-- The sum of products at the vertices of a cube given a face assignment -/
def vertexProductSum (assignment : CubeAssignment) : ℕ :=
  let a := assignment.faces 0
  let b := assignment.faces 1
  let c := assignment.faces 2
  let d := assignment.faces 3
  let e := assignment.faces 4
  let f := assignment.faces 5
  (a + b) * (c + d) * (e + f)

/-- The maximum sum of products at the vertices of a cube -/
theorem max_vertex_product_sum :
  ∃ (assignment : CubeAssignment), ∀ (other : CubeAssignment),
    vertexProductSum assignment ≥ vertexProductSum other ∧
    vertexProductSum assignment = 729 :=
  sorry

end NUMINAMATH_CALUDE_max_vertex_product_sum_l31_3150


namespace NUMINAMATH_CALUDE_sum_exradii_equals_four_circumradius_plus_inradius_l31_3157

/-- Given a triangle with exradii r_a, r_b, r_c, circumradius R, and inradius r,
    prove that the sum of the exradii equals four times the circumradius plus the inradius. -/
theorem sum_exradii_equals_four_circumradius_plus_inradius 
  (r_a r_b r_c R r : ℝ) :
  r_a > 0 → r_b > 0 → r_c > 0 → R > 0 → r > 0 →
  r_a + r_b + r_c = 4 * R + r := by
  sorry

end NUMINAMATH_CALUDE_sum_exradii_equals_four_circumradius_plus_inradius_l31_3157


namespace NUMINAMATH_CALUDE_correct_answer_l31_3174

-- Define proposition p
def p : Prop := ∀ a b c : ℝ, a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ x : ℝ, x^2 - x + 1 ≤ 0

-- Theorem to prove
theorem correct_answer : p ∨ (¬q) := by sorry

end NUMINAMATH_CALUDE_correct_answer_l31_3174


namespace NUMINAMATH_CALUDE_lcm_24_90_35_l31_3126

theorem lcm_24_90_35 : Nat.lcm 24 (Nat.lcm 90 35) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_90_35_l31_3126


namespace NUMINAMATH_CALUDE_abcd_congruence_l31_3131

theorem abcd_congruence (a b c d : ℕ) 
  (h1 : a < 7) (h2 : b < 7) (h3 : c < 7) (h4 : d < 7)
  (c1 : (a + 2*b + 3*c + 4*d) % 7 = 1)
  (c2 : (2*a + 3*b + c + 2*d) % 7 = 5)
  (c3 : (3*a + b + 2*c + 3*d) % 7 = 3)
  (c4 : (4*a + 2*b + d + c) % 7 = 2) :
  (a * b * c * d) % 7 = 0 := by
sorry

end NUMINAMATH_CALUDE_abcd_congruence_l31_3131


namespace NUMINAMATH_CALUDE_residue_negative_999_mod_25_l31_3191

theorem residue_negative_999_mod_25 : Int.mod (-999) 25 = 1 := by
  sorry

end NUMINAMATH_CALUDE_residue_negative_999_mod_25_l31_3191


namespace NUMINAMATH_CALUDE_melanie_gave_27_apples_l31_3183

/-- The number of apples Joan picked from the orchard -/
def apples_picked : ℕ := 43

/-- The total number of apples Joan has now -/
def total_apples : ℕ := 70

/-- The number of apples Melanie gave to Joan -/
def apples_from_melanie : ℕ := total_apples - apples_picked

theorem melanie_gave_27_apples : apples_from_melanie = 27 := by
  sorry

end NUMINAMATH_CALUDE_melanie_gave_27_apples_l31_3183


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l31_3152

/-- The constant term in the expansion of (x + 1/x)^4 -/
def constant_term : ℕ := 6

/-- Represents a geometric sequence -/
def geometric_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, a (n + m) = a n * a m

theorem geometric_sequence_product
  (a : ℕ → ℕ)
  (h_geo : geometric_sequence a)
  (h_a5 : a 5 = constant_term) :
  a 3 * a 7 = 36 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l31_3152


namespace NUMINAMATH_CALUDE_ellipse_properties_l31_3107

-- Define the ellipse C
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the focal distance
def focal_distance (c : ℝ) : Prop :=
  c = 2 * Real.sqrt 3

-- Define the intersection points with y-axis
def y_intersections (b : ℝ) : Prop :=
  b = 1

-- Define the standard form of the ellipse
def standard_form (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop :=
  e = Real.sqrt 3 / 2

-- Define the range of x-coordinate for point P
def x_range (x : ℝ) : Prop :=
  24 / 13 < x ∧ x ≤ 2

-- Define the maximum value of |EF|
def max_ef (ef : ℝ) : Prop :=
  ef = 1

theorem ellipse_properties (a b c : ℝ) :
  a > b ∧ b > 0 ∧
  focal_distance c ∧
  y_intersections b →
  (∃ x y, ellipse x y a b ∧ standard_form x y) ∧
  (∃ e, eccentricity e) ∧
  (∃ x, x_range x) ∧
  (∃ ef, max_ef ef) :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l31_3107


namespace NUMINAMATH_CALUDE_fish_value_in_honey_l31_3193

/-- Represents the value of one fish in terms of jars of honey -/
def fish_value (fish_to_bread : ℚ) (bread_to_honey : ℚ) : ℚ :=
  (3 / 4) * bread_to_honey

/-- Theorem stating the value of one fish in jars of honey -/
theorem fish_value_in_honey 
  (h1 : fish_to_bread = 3 / 4)  -- 4 fish = 3 loaves of bread
  (h2 : bread_to_honey = 3)     -- 1 loaf of bread = 3 jars of honey
  : fish_value fish_to_bread bread_to_honey = 9 / 4 := by
  sorry

#eval fish_value (3 / 4) 3  -- Should evaluate to 2.25

end NUMINAMATH_CALUDE_fish_value_in_honey_l31_3193


namespace NUMINAMATH_CALUDE_black_area_after_changes_l31_3151

/-- Represents the fraction of the square that is black -/
def black_fraction : ℕ → ℚ
  | 0 => 1/2  -- Initially half the square is black
  | (n+1) => (3/4) * black_fraction n  -- Each change keeps 3/4 of the previous black area

/-- The number of changes applied to the square -/
def num_changes : ℕ := 6

theorem black_area_after_changes :
  black_fraction num_changes = 729/8192 := by
  sorry

end NUMINAMATH_CALUDE_black_area_after_changes_l31_3151


namespace NUMINAMATH_CALUDE_hundred_thousand_scientific_notation_l31_3116

-- Define scientific notation
def scientific_notation (n : ℝ) (x : ℝ) (y : ℤ) : Prop :=
  n = x * (10 : ℝ) ^ y ∧ 1 ≤ x ∧ x < 10

-- Theorem statement
theorem hundred_thousand_scientific_notation :
  scientific_notation 100000 1 5 :=
by sorry

end NUMINAMATH_CALUDE_hundred_thousand_scientific_notation_l31_3116


namespace NUMINAMATH_CALUDE_distribute_five_balls_two_boxes_l31_3149

/-- The number of ways to distribute n indistinguishable balls into 2 indistinguishable boxes -/
def distribute_balls (n : ℕ) : ℕ :=
  (n + 1) / 2 + 1

/-- Theorem: There are 3 ways to distribute 5 indistinguishable balls into 2 indistinguishable boxes -/
theorem distribute_five_balls_two_boxes : distribute_balls 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_two_boxes_l31_3149


namespace NUMINAMATH_CALUDE_spending_problem_l31_3118

theorem spending_problem (initial_amount : ℚ) : 
  (2 / 5 : ℚ) * initial_amount = 600 → initial_amount = 1500 := by
  sorry

end NUMINAMATH_CALUDE_spending_problem_l31_3118


namespace NUMINAMATH_CALUDE_unreachable_from_2_2_2_reachable_from_3_3_3_l31_3162

/-- The operation that replaces one number with the difference between the sum of the other two and 1 -/
def operation (x y z : ℤ) : ℤ × ℤ × ℤ → Prop :=
  fun w => (w = (y + z - 1, y, z)) ∨ (w = (x, x + z - 1, z)) ∨ (w = (x, y, x + y - 1))

/-- The relation that represents the repeated application of the operation -/
inductive reachable : ℤ × ℤ × ℤ → ℤ × ℤ × ℤ → Prop
  | refl {x} : reachable x x
  | step {x y z} (h : reachable x y) (o : operation y.1 y.2.1 y.2.2 z) : reachable x z

theorem unreachable_from_2_2_2 :
  ¬ reachable (2, 2, 2) (17, 1999, 2105) :=
sorry

theorem reachable_from_3_3_3 :
  reachable (3, 3, 3) (17, 1999, 2105) :=
sorry

end NUMINAMATH_CALUDE_unreachable_from_2_2_2_reachable_from_3_3_3_l31_3162


namespace NUMINAMATH_CALUDE_container_volume_ratio_l31_3117

theorem container_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/5 : ℝ) * V₁ = (2/3 : ℝ) * V₂ →
  V₁ / V₂ = 10/9 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l31_3117


namespace NUMINAMATH_CALUDE_min_pieces_to_control_l31_3138

/-- Represents a rhombus game board -/
structure GameBoard where
  angle : ℝ
  side_divisions : ℕ

/-- Represents a piece on the game board -/
structure Piece where
  position : ℕ × ℕ

/-- Checks if a piece controls a given position -/
def controls (p : Piece) (pos : ℕ × ℕ) : Prop := sorry

/-- Checks if a set of pieces controls all positions on the board -/
def controls_all (pieces : Finset Piece) (board : GameBoard) : Prop := sorry

/-- The main theorem stating the minimum number of pieces required -/
theorem min_pieces_to_control (board : GameBoard) :
  board.angle = 60 ∧ board.side_divisions = 9 →
  ∃ (pieces : Finset Piece),
    pieces.card = 6 ∧
    controls_all pieces board ∧
    ∀ (other_pieces : Finset Piece),
      controls_all other_pieces board →
      other_pieces.card ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_pieces_to_control_l31_3138


namespace NUMINAMATH_CALUDE_fifth_term_geometric_sequence_l31_3190

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem fifth_term_geometric_sequence :
  let a₁ : ℚ := 2
  let a₂ : ℚ := 1/4
  let r : ℚ := a₂ / a₁
  geometric_sequence a₁ r 5 = 1/2048 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_geometric_sequence_l31_3190


namespace NUMINAMATH_CALUDE_west_representation_l31_3148

-- Define a type for direction
inductive Direction
  | East
  | West

-- Define a function to represent distance with direction
def representDistance (dir : Direction) (distance : ℝ) : ℝ :=
  match dir with
  | Direction.East => distance
  | Direction.West => -distance

-- State the theorem
theorem west_representation :
  representDistance Direction.East 80 = 80 →
  representDistance Direction.West 200 = -200 :=
by sorry

end NUMINAMATH_CALUDE_west_representation_l31_3148


namespace NUMINAMATH_CALUDE_mnp_value_l31_3187

theorem mnp_value (a b x y : ℝ) (m n p : ℤ) 
  (h : a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1)) 
  (h_equiv : (a^m*x - a^n)*(a^p*y - a^3) = a^5*b^5) : 
  m * n * p = 32 := by
  sorry

end NUMINAMATH_CALUDE_mnp_value_l31_3187


namespace NUMINAMATH_CALUDE_characterize_valid_triples_l31_3188

def is_valid_triple (a b c : ℕ+) : Prop :=
  (1 : ℚ) / a + 2 / b + 3 / c = 1 ∧
  Nat.Prime a ∧
  a ≤ b ∧ b ≤ c

def valid_triples : Set (ℕ+ × ℕ+ × ℕ+) :=
  {(2, 5, 30), (2, 6, 18), (2, 7, 14), (2, 8, 12), (2, 10, 10),
   (3, 4, 18), (3, 6, 9), (5, 4, 10)}

theorem characterize_valid_triples :
  ∀ a b c : ℕ+, is_valid_triple a b c ↔ (a, b, c) ∈ valid_triples := by
  sorry

end NUMINAMATH_CALUDE_characterize_valid_triples_l31_3188


namespace NUMINAMATH_CALUDE_fraction_reduction_l31_3130

theorem fraction_reduction (a b d : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hd : d ≠ 0) (hsum : a + b + d ≠ 0) :
  (a^2 + b^2 - d^2 + 2*a*b) / (a^2 + d^2 - b^2 + 2*a*d) = (a + b - d) / (a + d - b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_reduction_l31_3130


namespace NUMINAMATH_CALUDE_smallest_square_cover_l31_3103

/-- The side length of the smallest square that can be covered by 2-by-4 rectangles -/
def smallest_square_side : ℕ := 8

/-- The area of a 2-by-4 rectangle -/
def rectangle_area : ℕ := 2 * 4

/-- The number of 2-by-4 rectangles needed to cover the smallest square -/
def num_rectangles : ℕ := smallest_square_side^2 / rectangle_area

theorem smallest_square_cover :
  (∀ n : ℕ, n < smallest_square_side → n^2 % rectangle_area ≠ 0) ∧
  smallest_square_side^2 % rectangle_area = 0 ∧
  num_rectangles = 8 := by sorry

end NUMINAMATH_CALUDE_smallest_square_cover_l31_3103


namespace NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt3_over_2_l31_3172

theorem cos_sin_sum_equals_sqrt3_over_2 :
  Real.cos (6 * π / 180) * Real.cos (36 * π / 180) + 
  Real.sin (6 * π / 180) * Real.cos (54 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_equals_sqrt3_over_2_l31_3172


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l31_3114

def A : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}
def B : Set ℝ := {x : ℝ | (x+4)*(x-2) > 0}

theorem set_intersection_theorem : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l31_3114


namespace NUMINAMATH_CALUDE_female_democrats_count_l31_3144

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) 
  (h1 : female + male = total)
  (h2 : total = 720)
  (h3 : female / 2 + male / 4 = total / 3) :
  female / 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_female_democrats_count_l31_3144


namespace NUMINAMATH_CALUDE_power_inequality_l31_3198

theorem power_inequality (a b : ℕ) (ha : a ≥ 2) (hb : b ≥ 3) :
  a^b + 1 ≥ b * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l31_3198


namespace NUMINAMATH_CALUDE_chris_fishing_trips_l31_3184

theorem chris_fishing_trips (brian_trips : ℕ) (chris_trips : ℕ) (brian_fish_per_trip : ℕ) (total_fish : ℕ) :
  brian_trips = 2 * chris_trips →
  brian_fish_per_trip = 400 →
  total_fish = 13600 →
  brian_fish_per_trip * brian_trips + (chris_trips * (brian_fish_per_trip * 7 / 5)) = total_fish →
  chris_trips = 10 := by
sorry

end NUMINAMATH_CALUDE_chris_fishing_trips_l31_3184


namespace NUMINAMATH_CALUDE_units_digit_problem_l31_3147

/-- Given a positive even integer with a positive units digit,
    if the units digit of its cube minus the units digit of its square is 0,
    then the number needed to be added to its units digit to get 10 is 4. -/
theorem units_digit_problem (p : ℕ) : 
  p > 0 → 
  Even p → 
  p % 10 > 0 → 
  p % 10 < 10 → 
  (p^3 % 10) - (p^2 % 10) = 0 → 
  10 - (p % 10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l31_3147


namespace NUMINAMATH_CALUDE_custom_mul_one_neg_three_l31_3192

-- Define the custom multiplication operation
def custom_mul (a b : ℤ) : ℤ := a^2 + 2*a*b - b^2

-- Theorem statement
theorem custom_mul_one_neg_three :
  custom_mul 1 (-3) = -14 :=
by
  sorry

end NUMINAMATH_CALUDE_custom_mul_one_neg_three_l31_3192


namespace NUMINAMATH_CALUDE_reciprocal_roots_equation_l31_3145

theorem reciprocal_roots_equation (m n : ℝ) (hn : n ≠ 0) :
  let original_eq := fun x => x^2 + m*x + n
  let reciprocal_eq := fun x => n*x^2 + m*x + 1
  ∀ x, original_eq x = 0 → reciprocal_eq (1/x) = 0 :=
sorry


end NUMINAMATH_CALUDE_reciprocal_roots_equation_l31_3145


namespace NUMINAMATH_CALUDE_equation_equivalence_l31_3165

theorem equation_equivalence (x : ℝ) (P : ℝ) (h : 3 * (4 * x + 5 * Real.pi) = P) :
  6 * (8 * x + 10 * Real.pi) = 4 * P := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l31_3165


namespace NUMINAMATH_CALUDE_smallest_among_four_numbers_l31_3100

theorem smallest_among_four_numbers :
  let a : ℝ := -Real.sqrt 3
  let b : ℝ := 0
  let c : ℝ := 2
  let d : ℝ := -3
  d < a ∧ d < b ∧ d < c := by sorry

end NUMINAMATH_CALUDE_smallest_among_four_numbers_l31_3100


namespace NUMINAMATH_CALUDE_triangle_properties_l31_3120

/-- Given a triangle ABC with the specified properties, prove the cosine of angle B and the perimeter. -/
theorem triangle_properties (A B C : ℝ) (AB BC AC : ℝ) : 
  C = 2 * A →
  Real.cos A = 3 / 4 →
  2 * (AB * BC * Real.cos B) = -27 →
  AB = 6 →
  BC = 4 →
  AC = 5 →
  Real.cos B = 9 / 16 ∧ AB + BC + AC = 15 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l31_3120


namespace NUMINAMATH_CALUDE_inscribed_sphere_sum_l31_3121

/-- A sphere inscribed in a right cone with base radius 15 cm and height 30 cm -/
structure InscribedSphere :=
  (b : ℝ)
  (d : ℝ)
  (radius : ℝ)
  (cone_base_radius : ℝ)
  (cone_height : ℝ)
  (radius_eq : radius = b * (Real.sqrt d - 1))
  (cone_base_radius_eq : cone_base_radius = 15)
  (cone_height_eq : cone_height = 30)

/-- Theorem stating that b + d = 12.5 for the inscribed sphere -/
theorem inscribed_sphere_sum (s : InscribedSphere) : s.b + s.d = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_sum_l31_3121


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l31_3163

theorem power_mod_seventeen : 5^2023 % 17 = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l31_3163


namespace NUMINAMATH_CALUDE_bat_ball_cost_difference_l31_3140

/-- The cost difference between a ball and a bat -/
def cost_difference (x y : ℝ) : ℝ := y - x

/-- The problem statement -/
theorem bat_ball_cost_difference :
  ∀ x y : ℝ,
  (2 * x + 3 * y = 1300) →
  (3 * x + 2 * y = 1200) →
  cost_difference x y = 100 := by
sorry

end NUMINAMATH_CALUDE_bat_ball_cost_difference_l31_3140


namespace NUMINAMATH_CALUDE_chess_master_exhibition_l31_3178

theorem chess_master_exhibition (x : ℝ) 
  (h1 : 0.1 * x + 8 + 0.1 * (0.9 * x - 8) + 2 + 7 = x) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_chess_master_exhibition_l31_3178


namespace NUMINAMATH_CALUDE_equation_C_is_linear_l31_3122

/-- Definition of a linear equation in one variable -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

/-- The equation 2x + 3 = 7 is linear -/
theorem equation_C_is_linear : is_linear_equation (λ x => 2 * x + 3) :=
by
  sorry

#check equation_C_is_linear

end NUMINAMATH_CALUDE_equation_C_is_linear_l31_3122


namespace NUMINAMATH_CALUDE_average_age_increase_l31_3142

theorem average_age_increase (initial_men : ℕ) (replaced_men_ages : List ℕ) (women_avg_age : ℚ) : 
  initial_men = 8 →
  replaced_men_ages = [20, 10] →
  women_avg_age = 23 →
  (((initial_men : ℚ) * women_avg_age + (women_avg_age * 2 - replaced_men_ages.sum)) / initial_men) - women_avg_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l31_3142


namespace NUMINAMATH_CALUDE_test_configuration_theorem_l31_3154

/-- Represents the fraction of problems that are difficult and the fraction of students who perform well -/
structure TestConfiguration (α : ℚ) :=
  (difficult_problems : ℚ)
  (well_performing_students : ℚ)
  (difficult_problems_ge : difficult_problems ≥ α)
  (well_performing_students_ge : well_performing_students ≥ α)

/-- Theorem stating the existence and non-existence of certain test configurations -/
theorem test_configuration_theorem :
  (∃ (config : TestConfiguration (2/3)), True) ∧
  (¬ ∃ (config : TestConfiguration (3/4)), True) ∧
  (¬ ∃ (config : TestConfiguration (7/10^7)), True) := by
  sorry

end NUMINAMATH_CALUDE_test_configuration_theorem_l31_3154


namespace NUMINAMATH_CALUDE_enthalpy_change_is_236_l31_3128

-- Define bond dissociation energies
def CC_bond_energy : ℝ := 347
def CO_bond_energy : ℝ := 358
def OH_bond_energy_alcohol : ℝ := 463
def CO_double_bond_energy : ℝ := 745
def OH_bond_energy_acid : ℝ := 467
def OO_double_bond_energy : ℝ := 498
def OH_bond_energy_water : ℝ := 467

-- Define the reaction components
def moles_CH3CH2OH : ℝ := 1
def moles_O2 : ℝ := 1.5
def moles_H2O : ℝ := 1
def moles_CH3COOH : ℝ := 1

-- Define the function to calculate enthalpy change
def calculate_enthalpy_change : ℝ :=
  let bonds_broken := CC_bond_energy + CO_bond_energy + OH_bond_energy_alcohol + moles_O2 * OO_double_bond_energy
  let bonds_formed := CO_double_bond_energy + OH_bond_energy_acid + OH_bond_energy_water
  bonds_broken - bonds_formed

-- Theorem statement
theorem enthalpy_change_is_236 : calculate_enthalpy_change = 236 := by
  sorry

end NUMINAMATH_CALUDE_enthalpy_change_is_236_l31_3128


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l31_3196

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  ((x₁ + 1) * (x₁ - 1) = 2 * x₁ + 3) ∧ ((x₂ + 1) * (x₂ - 1) = 2 * x₂ + 3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l31_3196


namespace NUMINAMATH_CALUDE_lesser_fraction_l31_3119

theorem lesser_fraction (x y : ℚ) (h_sum : x + y = 3/4) (h_product : x * y = 1/8) : 
  min x y = 1/4 := by sorry

end NUMINAMATH_CALUDE_lesser_fraction_l31_3119


namespace NUMINAMATH_CALUDE_common_roots_product_l31_3179

-- Define the two polynomial functions
def f (x : ℝ) : ℝ := x^3 + 3*x + 20
def g (x : ℝ) : ℝ := x^3 + 4*x^2 + 80

-- Define the property of having common roots
def has_common_roots (p q : ℝ → ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ p x = 0 ∧ p y = 0 ∧ q x = 0 ∧ q y = 0

-- Theorem statement
theorem common_roots_product :
  has_common_roots f g →
  ∃ (x y : ℝ), x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ g x = 0 ∧ g y = 0 ∧ x * y = 20 :=
by sorry

end NUMINAMATH_CALUDE_common_roots_product_l31_3179
