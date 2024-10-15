import Mathlib

namespace NUMINAMATH_CALUDE_r_lower_bound_l2938_293818

theorem r_lower_bound (a b c d : ℕ+) (r : ℚ) :
  r = 1 - (a : ℚ) / b - (c : ℚ) / d →
  a + c ≤ 1982 →
  r ≥ 0 →
  r > 1 / (1983 : ℚ)^3 := by
  sorry

end NUMINAMATH_CALUDE_r_lower_bound_l2938_293818


namespace NUMINAMATH_CALUDE_parallel_resistors_l2938_293812

/-- Given two resistors connected in parallel with resistances x and y,
    where the combined resistance r satisfies 1/r = 1/x + 1/y,
    prove that when x = 4 ohms and r = 2.4 ohms, y = 6 ohms. -/
theorem parallel_resistors (x y r : ℝ) 
  (hx : x = 4)
  (hr : r = 2.4)
  (h_combined : 1 / r = 1 / x + 1 / y) :
  y = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_resistors_l2938_293812


namespace NUMINAMATH_CALUDE_fish_count_l2938_293849

theorem fish_count (total_pets dogs cats : ℕ) (h1 : total_pets = 149) (h2 : dogs = 43) (h3 : cats = 34) :
  total_pets - (dogs + cats) = 72 := by
sorry

end NUMINAMATH_CALUDE_fish_count_l2938_293849


namespace NUMINAMATH_CALUDE_kyunghwan_spent_most_l2938_293868

def initial_amount : ℕ := 20000

def seunga_remaining : ℕ := initial_amount / 4
def kyunghwan_remaining : ℕ := initial_amount / 8
def doyun_remaining : ℕ := initial_amount / 5

def seunga_spent : ℕ := initial_amount - seunga_remaining
def kyunghwan_spent : ℕ := initial_amount - kyunghwan_remaining
def doyun_spent : ℕ := initial_amount - doyun_remaining

theorem kyunghwan_spent_most : 
  kyunghwan_spent > seunga_spent ∧ kyunghwan_spent > doyun_spent :=
by sorry

end NUMINAMATH_CALUDE_kyunghwan_spent_most_l2938_293868


namespace NUMINAMATH_CALUDE_solve_for_y_l2938_293845

theorem solve_for_y (x y : ℝ) (h1 : 2 * (x - y) = 32) (h2 : x + y = -4) : y = -10 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2938_293845


namespace NUMINAMATH_CALUDE_unpainted_area_calculation_l2938_293863

theorem unpainted_area_calculation (board_width1 board_width2 : ℝ) 
  (angle : ℝ) (h1 : board_width1 = 5) (h2 : board_width2 = 8) 
  (h3 : angle = 45 * π / 180) : 
  board_width1 * (board_width2 * Real.sin angle) = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_area_calculation_l2938_293863


namespace NUMINAMATH_CALUDE_min_perimeter_is_8_meters_l2938_293859

/-- Represents the side length of a square tile in centimeters -/
def tileSideLength : ℕ := 40

/-- Represents the total number of tiles -/
def totalTiles : ℕ := 24

/-- Calculates the perimeter of a rectangle given its length and width in tile units -/
def perimeterInTiles (length width : ℕ) : ℕ := 2 * (length + width)

/-- Checks if the given dimensions form a valid rectangle using all tiles -/
def isValidRectangle (length width : ℕ) : Prop := length * width = totalTiles

/-- Theorem: The minimum perimeter of a rectangular arrangement of 24 square tiles,
    each with side length 40 cm, is 8 meters -/
theorem min_perimeter_is_8_meters :
  ∃ (length width : ℕ),
    isValidRectangle length width ∧
    ∀ (l w : ℕ), isValidRectangle l w →
      perimeterInTiles length width ≤ perimeterInTiles l w ∧
      perimeterInTiles length width * tileSideLength = 800 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_is_8_meters_l2938_293859


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2938_293839

theorem min_value_quadratic (x y : ℝ) : 3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 ≥ 39 / 4 ∧
  (3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + 9 = 39 / 4 ↔ x = 1 / 2 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2938_293839


namespace NUMINAMATH_CALUDE_manager_chef_wage_difference_l2938_293861

/-- Represents the hourly wages at Joe's Steakhouse -/
structure SteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ

/-- Defines the wage relationships at Joe's Steakhouse -/
def valid_steakhouse_wages (w : SteakhouseWages) : Prop :=
  w.manager = 8.50 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.dishwasher * 1.22

/-- Theorem stating the wage difference between manager and chef -/
theorem manager_chef_wage_difference (w : SteakhouseWages) 
  (h : valid_steakhouse_wages w) : 
  w.manager - w.chef = 3.315 := by
  sorry

end NUMINAMATH_CALUDE_manager_chef_wage_difference_l2938_293861


namespace NUMINAMATH_CALUDE_largest_tank_volume_width_l2938_293843

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank -/
structure GasTank where
  radius : ℝ
  height : ℝ

/-- Checks if a gas tank fits inside a crate -/
def tankFitsInCrate (tank : GasTank) (crate : CrateDimensions) : Prop :=
  (2 * tank.radius ≤ crate.length ∧ 2 * tank.radius ≤ crate.width) ∨
  (2 * tank.radius ≤ crate.length ∧ 2 * tank.radius ≤ crate.height) ∨
  (2 * tank.radius ≤ crate.width ∧ 2 * tank.radius ≤ crate.height)

/-- Theorem: The width of the crate must be 8 feet for the largest possible tank volume -/
theorem largest_tank_volume_width (x : ℝ) :
  let crate := CrateDimensions.mk 6 x 10
  let tank := GasTank.mk 4 (min (min 6 x) 10)
  tankFitsInCrate tank crate → x = 8 := by
  sorry

#check largest_tank_volume_width

end NUMINAMATH_CALUDE_largest_tank_volume_width_l2938_293843


namespace NUMINAMATH_CALUDE_new_room_ratio_l2938_293867

/-- The ratio of a new room's size to the combined size of a bedroom and bathroom -/
theorem new_room_ratio (bedroom_size bathroom_size new_room_size : ℝ) 
  (h1 : bedroom_size = 309)
  (h2 : bathroom_size = 150)
  (h3 : new_room_size = 918) :
  new_room_size / (bedroom_size + bathroom_size) = 2 := by
  sorry

end NUMINAMATH_CALUDE_new_room_ratio_l2938_293867


namespace NUMINAMATH_CALUDE_octahedron_non_prime_sum_pairs_l2938_293883

-- Define the type for die faces
def DieFace := Fin 8

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := sorry

-- Define a function to get the value on a die face
def faceValue (face : DieFace) : ℕ := face.val + 1

-- Define a function to check if the sum of two face values is not prime
def sumNotPrime (face1 face2 : DieFace) : Prop :=
  ¬(isPrime (faceValue face1 + faceValue face2))

-- The main theorem
theorem octahedron_non_prime_sum_pairs :
  ∃ (pairs : Finset (DieFace × DieFace)),
    pairs.card = 8 ∧
    (∀ (pair : DieFace × DieFace), pair ∈ pairs → sumNotPrime pair.1 pair.2) ∧
    (∀ (face1 face2 : DieFace), 
      face1 ≠ face2 → sumNotPrime face1 face2 → 
      (face1, face2) ∈ pairs ∨ (face2, face1) ∈ pairs) :=
sorry

end NUMINAMATH_CALUDE_octahedron_non_prime_sum_pairs_l2938_293883


namespace NUMINAMATH_CALUDE_committee_size_is_four_l2938_293817

/-- The number of boys in the total group -/
def num_boys : ℕ := 5

/-- The number of girls in the total group -/
def num_girls : ℕ := 6

/-- The number of boys required in each committee -/
def boys_in_committee : ℕ := 2

/-- The number of girls required in each committee -/
def girls_in_committee : ℕ := 2

/-- The total number of possible committees -/
def total_committees : ℕ := 150

/-- The number of people in each committee -/
def committee_size : ℕ := boys_in_committee + girls_in_committee

theorem committee_size_is_four :
  committee_size = 4 :=
by sorry

end NUMINAMATH_CALUDE_committee_size_is_four_l2938_293817


namespace NUMINAMATH_CALUDE_competition_participants_l2938_293823

theorem competition_participants : 
  ∀ n : ℕ, 
  (100 < n ∧ n < 200) ∧ 
  (∃ k : ℕ, n = 4 * k - 2) ∧
  (∃ l : ℕ, n = 5 * l - 3) ∧
  (∃ m : ℕ, n = 6 * m - 4) →
  n = 122 ∨ n = 182 := by
sorry

end NUMINAMATH_CALUDE_competition_participants_l2938_293823


namespace NUMINAMATH_CALUDE_decimal_places_relation_l2938_293855

/-- Represents a decimal number -/
structure Decimal where
  integerPart : ℤ
  fractionalPart : ℕ
  decimalPlaces : ℕ

/-- Represents the result of decimal multiplication -/
structure DecimalMultiplicationResult where
  product : Decimal
  factor1 : Decimal
  factor2 : Decimal

/-- Rules of decimal multiplication -/
axiom decimal_multiplication_rule (result : DecimalMultiplicationResult) :
  result.product.decimalPlaces = result.factor1.decimalPlaces + result.factor2.decimalPlaces

/-- Theorem: The number of decimal places in a product is related to the number of decimal places in its factors -/
theorem decimal_places_relation :
  ∃ (result : DecimalMultiplicationResult),
    result.product.decimalPlaces ≠ result.factor1.decimalPlaces ∨
    result.product.decimalPlaces ≠ result.factor2.decimalPlaces :=
  sorry

end NUMINAMATH_CALUDE_decimal_places_relation_l2938_293855


namespace NUMINAMATH_CALUDE_prob_exactly_two_ones_value_l2938_293809

def num_dice : ℕ := 12
def num_sides : ℕ := 6
def target_outcome : ℕ := 1
def num_target : ℕ := 2

def prob_exactly_two_ones : ℚ :=
  (num_dice.choose num_target) *
  (1 / num_sides) ^ num_target *
  ((num_sides - 1) / num_sides) ^ (num_dice - num_target)

theorem prob_exactly_two_ones_value :
  prob_exactly_two_ones = (66 * 5^10) / 6^12 := by
  sorry

end NUMINAMATH_CALUDE_prob_exactly_two_ones_value_l2938_293809


namespace NUMINAMATH_CALUDE_sum_of_two_equals_zero_l2938_293858

theorem sum_of_two_equals_zero (a b c d : ℝ) 
  (h1 : a^3 + b^3 + c^3 + d^3 = 0) 
  (h2 : a + b + c + d = 0) : 
  a + b = 0 ∨ c + d = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_equals_zero_l2938_293858


namespace NUMINAMATH_CALUDE_divisibility_by_13_l2938_293891

theorem divisibility_by_13 (N : ℕ) (x : ℕ) : 
  (N = 2 * 10^2022 + x * 10^2000 + 23) →
  (N % 13 = 0) →
  (x = 3) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_13_l2938_293891


namespace NUMINAMATH_CALUDE_sector_area_from_arc_length_l2938_293829

/-- Given a circle with radius 6cm and an arc length of 25.12cm, 
    the area of the sector formed by this arc is 75.36 cm². -/
theorem sector_area_from_arc_length : 
  let r : ℝ := 6  -- radius in cm
  let arc_length : ℝ := 25.12  -- arc length in cm
  let π : ℝ := Real.pi
  let central_angle : ℝ := arc_length / r  -- angle in radians
  let sector_area : ℝ := 0.5 * r^2 * central_angle
  sector_area = 75.36 := by
  sorry

#check sector_area_from_arc_length

end NUMINAMATH_CALUDE_sector_area_from_arc_length_l2938_293829


namespace NUMINAMATH_CALUDE_geometric_sequence_12th_term_l2938_293810

/-- In a geometric sequence, given the 4th and 8th terms, prove the 12th term -/
theorem geometric_sequence_12th_term 
  (a : ℕ → ℝ) -- The sequence
  (h1 : a 4 = 2) -- 4th term is 2
  (h2 : a 8 = 162) -- 8th term is 162
  (h3 : ∀ n : ℕ, n ≥ 1 → ∃ r : ℝ, a (n + 1) = a n * r) -- Definition of geometric sequence
  : a 12 = 13122 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_12th_term_l2938_293810


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2938_293836

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2)  -- Pythagorean theorem
  (h2 : c = 10)           -- Hypotenuse is 10
  (h3 : a = 6)            -- One side is 6
  : b = 8 :=              -- Prove the other side is 8
by sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2938_293836


namespace NUMINAMATH_CALUDE_first_question_percentage_l2938_293813

/-- Given a class of students taking a test with two questions, this theorem proves
    the percentage of students who answered the first question correctly. -/
theorem first_question_percentage
  (second_correct : ℝ)
  (neither_correct : ℝ)
  (both_correct : ℝ)
  (h1 : second_correct = 65)
  (h2 : neither_correct = 20)
  (h3 : both_correct = 60) :
  ∃ (first_correct : ℝ),
    first_correct = 75 ∧
    first_correct + second_correct - both_correct = 100 - neither_correct :=
by
  sorry


end NUMINAMATH_CALUDE_first_question_percentage_l2938_293813


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l2938_293808

theorem geometric_sequence_proof (a : ℕ → ℝ) (q : ℝ) (h_positive : q > 0) :
  (∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  a 2 * a 6 = 9 * a 4 →             -- given condition
  a 2 = 1 →                         -- given condition
  (q = 3 ∧ ∀ n : ℕ, a n = 3^(n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_l2938_293808


namespace NUMINAMATH_CALUDE_largest_of_three_numbers_l2938_293840

theorem largest_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (sum_products_eq : x*y + x*z + y*z = -6)
  (product_eq : x*y*z = -8) :
  ∃ max_val : ℝ, max_val = (1 + Real.sqrt 17) / 2 ∧ 
  max_val = max x (max y z) := by
sorry

end NUMINAMATH_CALUDE_largest_of_three_numbers_l2938_293840


namespace NUMINAMATH_CALUDE_tan_3_expression_zero_l2938_293869

theorem tan_3_expression_zero (θ : Real) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_3_expression_zero_l2938_293869


namespace NUMINAMATH_CALUDE_circle_line_intersection_equivalence_l2938_293893

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Reflection of a point over a line -/
def reflect_point (p : ℝ × ℝ) (l : Line) : ℝ × ℝ := sorry

/-- Reflection of a circle over a line -/
def reflect_circle (c : Circle) (l : Line) : Circle := sorry

/-- Intersection points of a circle and a line -/
def circle_line_intersection (c : Circle) (l : Line) : Set (ℝ × ℝ) := sorry

/-- Intersection points of two circles -/
def circle_circle_intersection (c1 c2 : Circle) : Set (ℝ × ℝ) := sorry

/-- Main theorem -/
theorem circle_line_intersection_equivalence 
  (k : Circle) (e : Line) (O A B : ℝ × ℝ) :
  O ≠ A ∧ O ≠ B ∧ A ≠ B ∧  -- O is not on line e
  e.point1 = A ∧ e.point2 = B ∧ 
  k.center = O →
  circle_line_intersection k e = circle_circle_intersection k (reflect_circle k e) := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_equivalence_l2938_293893


namespace NUMINAMATH_CALUDE_square_of_1023_l2938_293864

theorem square_of_1023 : 1023^2 = 1046529 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1023_l2938_293864


namespace NUMINAMATH_CALUDE_find_a_l2938_293816

-- Define the set A
def A (a : ℤ) : Set ℤ := {12, a^2 + 4*a, a - 2}

-- Theorem statement
theorem find_a : ∀ a : ℤ, -3 ∈ A a → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l2938_293816


namespace NUMINAMATH_CALUDE_fourth_term_is_375_l2938_293892

/-- A geometric sequence of positive integers with first term 3 and third term 75 -/
structure GeometricSequence where
  a : ℕ+  -- first term
  r : ℕ+  -- common ratio
  third_term_eq : a * r^2 = 75
  first_term_eq : a = 3

/-- The fourth term of the geometric sequence is 375 -/
theorem fourth_term_is_375 (seq : GeometricSequence) : seq.a * seq.r^3 = 375 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_375_l2938_293892


namespace NUMINAMATH_CALUDE_xxyy_perfect_square_l2938_293896

theorem xxyy_perfect_square : 
  ∃! (x y : Nat), x < 10 ∧ y < 10 ∧ 
  (1100 * x + 11 * y = 88 * 88) := by
sorry

end NUMINAMATH_CALUDE_xxyy_perfect_square_l2938_293896


namespace NUMINAMATH_CALUDE_olivia_math_problem_l2938_293838

theorem olivia_math_problem (x : ℝ) 
  (h1 : 7 * x + 3 = 31) : 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_olivia_math_problem_l2938_293838


namespace NUMINAMATH_CALUDE_cars_with_airbag_l2938_293851

theorem cars_with_airbag (total : ℕ) (power_windows : ℕ) (both : ℕ) (neither : ℕ) :
  total = 65 →
  power_windows = 30 →
  both = 12 →
  neither = 2 →
  total - neither = power_windows + (total - power_windows - neither) - both :=
by sorry

end NUMINAMATH_CALUDE_cars_with_airbag_l2938_293851


namespace NUMINAMATH_CALUDE_greatest_number_with_odd_factors_under_200_l2938_293824

theorem greatest_number_with_odd_factors_under_200 :
  ∀ n : ℕ, n < 200 → (∃ k : ℕ, n = k^2) →
  ∀ m : ℕ, m < 200 → (∃ l : ℕ, m = l^2) → m ≤ n →
  n = 196 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_odd_factors_under_200_l2938_293824


namespace NUMINAMATH_CALUDE_jim_apples_count_l2938_293841

theorem jim_apples_count : ∀ (j : ℕ), 
  (j + 60 + 40) / 3 = 2 * j → j = 200 := by
  sorry

end NUMINAMATH_CALUDE_jim_apples_count_l2938_293841


namespace NUMINAMATH_CALUDE_factor_polynomial_l2938_293832

theorem factor_polynomial (k : ℤ) : 
  (∀ x : ℝ, (x + k) ∣ (3 * x^2 + 14 * x + 8)) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2938_293832


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2938_293835

theorem quadratic_inequality_range (c : ℝ) : 
  (¬ ∀ x : ℝ, c ≤ -1/2 → x^2 + 4*c*x + 1 > 0) → c ≤ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2938_293835


namespace NUMINAMATH_CALUDE_tangent_angle_at_x_1_l2938_293872

/-- The angle of inclination of the tangent to the curve y = x³ - 2x + m at x = 1 is 45° -/
theorem tangent_angle_at_x_1 (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 2*x + m
  let f' : ℝ → ℝ := λ x => 3*x^2 - 2
  let slope : ℝ := f' 1
  Real.arctan slope = π/4 := by sorry

end NUMINAMATH_CALUDE_tangent_angle_at_x_1_l2938_293872


namespace NUMINAMATH_CALUDE_range_of_f_l2938_293825

noncomputable def f (x : ℝ) : ℝ := (x^2 + 4*x + 3) / (x + 2)

theorem range_of_f :
  ∀ y : ℝ, ∃ x : ℝ, x ≠ -2 ∧ f x = y :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2938_293825


namespace NUMINAMATH_CALUDE_complement_union_problem_l2938_293877

def U : Finset Nat := {0, 1, 2, 3, 4, 5}
def A : Finset Nat := {1, 2, 3, 5}
def B : Finset Nat := {2, 4}

theorem complement_union_problem :
  (U \ A) ∪ B = {0, 2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_problem_l2938_293877


namespace NUMINAMATH_CALUDE_parallel_implications_l2938_293821

-- Define the types for points and lines
variable (Point Line : Type)

-- Define a function to check if a point is on a line
variable (on_line : Point → Line → Prop)

-- Define a function to check if two lines are parallel
variable (parallel : Line → Line → Prop)

-- Define a function to create a line from two points
variable (line_from_points : Point → Point → Line)

-- Define the theorem
theorem parallel_implications
  (l l' : Line) (O A B C A' B' C' : Point)
  (h1 : on_line A l) (h2 : on_line B l) (h3 : on_line C l)
  (h4 : on_line A' l') (h5 : on_line B' l') (h6 : on_line C' l')
  (h7 : parallel (line_from_points A B') (line_from_points A' B))
  (h8 : parallel (line_from_points A C') (line_from_points A' C)) :
  parallel (line_from_points B C') (line_from_points B' C) :=
sorry

end NUMINAMATH_CALUDE_parallel_implications_l2938_293821


namespace NUMINAMATH_CALUDE_square_perimeter_doubled_l2938_293870

theorem square_perimeter_doubled (area : ℝ) (h : area = 900) : 
  let side_length := Real.sqrt area
  let initial_perimeter := 4 * side_length
  let doubled_perimeter := 2 * initial_perimeter
  doubled_perimeter = 240 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_doubled_l2938_293870


namespace NUMINAMATH_CALUDE_simplify_xy_expression_l2938_293871

theorem simplify_xy_expression (x y : ℝ) : 4 * x * y - 2 * x * y = 2 * x * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_xy_expression_l2938_293871


namespace NUMINAMATH_CALUDE_sequence_proof_l2938_293881

theorem sequence_proof (a : Fin 8 → ℕ) 
  (h1 : ∀ i : Fin 6, a i + a (i + 1) + a (i + 2) = 100)
  (h2 : a 0 = 20)
  (h3 : a 7 = 16) :
  a = ![20, 16, 64, 20, 16, 64, 20, 16] := by
sorry

end NUMINAMATH_CALUDE_sequence_proof_l2938_293881


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_line_l2938_293852

/-- Given a line (x/a) + (y/b) = 1 where a > 0 and b > 0, 
    and the line passes through the point (1, 1),
    the minimum value of a + b is 4. -/
theorem min_sum_of_reciprocal_line (a b : ℝ) : 
  a > 0 → b > 0 → (1 / a + 1 / b = 1) → 
  ∀ (a' b' : ℝ), a' > 0 → b' > 0 → (1 / a' + 1 / b' = 1) → 
  a + b ≤ a' + b' ∧ a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_line_l2938_293852


namespace NUMINAMATH_CALUDE_sets_properties_l2938_293820

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | ∃ y, y = x^2 + 1}
def B : Set ℝ := {y : ℝ | ∃ x, y = x^2 + 1}
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 + 1}

-- Theorem stating the properties of A, B, and C
theorem sets_properties :
  (A = Set.univ) ∧
  (B = {y : ℝ | y ≥ 1}) ∧
  (C = {p : ℝ × ℝ | p.2 = p.1^2 + 1}) :=
by sorry

end NUMINAMATH_CALUDE_sets_properties_l2938_293820


namespace NUMINAMATH_CALUDE_special_hexagon_area_l2938_293865

/-- A hexagon with specific side lengths that can be divided into a rectangle and two triangles -/
structure SpecialHexagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  side6 : ℝ
  rectangle_width : ℝ
  rectangle_height : ℝ
  triangle_base : ℝ
  triangle_height : ℝ
  side1_eq : side1 = 20
  side2_eq : side2 = 15
  side3_eq : side3 = 22
  side4_eq : side4 = 27
  side5_eq : side5 = 18
  side6_eq : side6 = 15
  rectangle_width_eq : rectangle_width = 18
  rectangle_height_eq : rectangle_height = 22
  triangle_base_eq : triangle_base = 18
  triangle_height_eq : triangle_height = 15

/-- The area of the special hexagon is 666 square units -/
theorem special_hexagon_area (h : SpecialHexagon) : 
  h.rectangle_width * h.rectangle_height + 2 * (1/2 * h.triangle_base * h.triangle_height) = 666 := by
  sorry

end NUMINAMATH_CALUDE_special_hexagon_area_l2938_293865


namespace NUMINAMATH_CALUDE_first_saline_concentration_l2938_293899

theorem first_saline_concentration 
  (desired_concentration : ℝ)
  (total_volume : ℝ)
  (first_volume : ℝ)
  (second_volume : ℝ)
  (second_concentration : ℝ)
  (h1 : desired_concentration = 3.24)
  (h2 : total_volume = 5)
  (h3 : first_volume = 3.6)
  (h4 : second_volume = 1.4)
  (h5 : second_concentration = 9)
  (h6 : total_volume = first_volume + second_volume)
  : ∃ (first_concentration : ℝ),
    first_concentration = 1 ∧
    desired_concentration * total_volume = 
      first_concentration * first_volume + second_concentration * second_volume :=
by sorry

end NUMINAMATH_CALUDE_first_saline_concentration_l2938_293899


namespace NUMINAMATH_CALUDE_share_ratio_B_to_C_l2938_293805

def total_amount : ℕ := 510
def share_A : ℕ := 360
def share_B : ℕ := 90
def share_C : ℕ := 60

theorem share_ratio_B_to_C : 
  (share_B : ℚ) / (share_C : ℚ) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_share_ratio_B_to_C_l2938_293805


namespace NUMINAMATH_CALUDE_min_distance_point_to_line_l2938_293897

/-- The minimum distance between a point (1,0) and the line x - y + 5 = 0 is 3√2 -/
theorem min_distance_point_to_line : 
  let F : ℝ × ℝ := (1, 0)
  let line (x y : ℝ) : Prop := x - y + 5 = 0
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 ∧ 
    ∀ (P : ℝ × ℝ), line P.1 P.2 → Real.sqrt ((F.1 - P.1)^2 + (F.2 - P.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_to_line_l2938_293897


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l2938_293895

theorem stewart_farm_ratio : 
  ∀ (sheep horses : ℕ) (horse_food_per_day total_horse_food : ℕ),
    sheep = 8 →
    horse_food_per_day = 230 →
    total_horse_food = 12880 →
    horses * horse_food_per_day = total_horse_food →
    sheep.gcd horses = 1 →
    sheep / horses = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_ratio_l2938_293895


namespace NUMINAMATH_CALUDE_fathers_age_l2938_293853

/-- Proves that the father's age is 30 given the conditions of the problem -/
theorem fathers_age (man_age : ℝ) (father_age : ℝ) : 
  man_age = (2/5) * father_age ∧ 
  man_age + 6 = (1/2) * (father_age + 6) → 
  father_age = 30 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_l2938_293853


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2938_293875

/-- Given that (2,3) is reflected across y = mx + b to (10,7), prove m + b = 15 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = (2 + 10) / 2 ∧ y = (3 + 7) / 2 ∧ y = m * x + b) →
  (m = -(10 - 2) / (7 - 3)) →
  m + b = 15 := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l2938_293875


namespace NUMINAMATH_CALUDE_coinciding_rest_days_count_l2938_293880

/-- Chris's schedule cycle length -/
def chris_cycle : ℕ := 6

/-- Dana's schedule cycle length -/
def dana_cycle : ℕ := 6

/-- Number of days Chris works in his cycle -/
def chris_work_days : ℕ := 4

/-- Number of days Dana works in her cycle -/
def dana_work_days : ℕ := 5

/-- Total number of days to consider -/
def total_days : ℕ := 1200

/-- The number of times Chris and Dana have rest-days on the same day -/
def coinciding_rest_days : ℕ := total_days / chris_cycle

theorem coinciding_rest_days_count :
  coinciding_rest_days = 200 :=
sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_count_l2938_293880


namespace NUMINAMATH_CALUDE_third_term_is_16_l2938_293814

/-- Geometric sequence with common ratio 2 and sum of first 4 terms equal to 60 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 2 * a n) ∧ 
  (a 1 + a 2 + a 3 + a 4 = 60)

/-- The third term of the geometric sequence is 16 -/
theorem third_term_is_16 (a : ℕ → ℝ) (h : geometric_sequence a) : a 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_16_l2938_293814


namespace NUMINAMATH_CALUDE_points_three_units_from_negative_two_l2938_293856

theorem points_three_units_from_negative_two (x : ℝ) : 
  (x = 1 ∨ x = -5) ↔ |x + 2| = 3 :=
by sorry

end NUMINAMATH_CALUDE_points_three_units_from_negative_two_l2938_293856


namespace NUMINAMATH_CALUDE_expression_is_integer_not_necessarily_natural_l2938_293833

theorem expression_is_integer_not_necessarily_natural : ∃ (n : ℤ), 
  (((1 + Real.sqrt 1991)^100 - (1 - Real.sqrt 1991)^100) / Real.sqrt 1991 = n) ∧ 
  (n ≠ 0 ∨ n < 0) := by
  sorry

end NUMINAMATH_CALUDE_expression_is_integer_not_necessarily_natural_l2938_293833


namespace NUMINAMATH_CALUDE_winnie_lollipop_distribution_l2938_293890

/-- Winnie's lollipop distribution problem -/
theorem winnie_lollipop_distribution 
  (total_lollipops : ℕ) 
  (num_friends : ℕ) 
  (h1 : total_lollipops = 72 + 89 + 23 + 316) 
  (h2 : num_friends = 14) : 
  total_lollipops % num_friends = 10 := by
  sorry

end NUMINAMATH_CALUDE_winnie_lollipop_distribution_l2938_293890


namespace NUMINAMATH_CALUDE_lake_depth_for_specific_cone_l2938_293801

/-- Represents a conical hill partially submerged in a lake -/
structure SubmergedCone where
  total_height : ℝ
  volume_ratio_above_water : ℝ

/-- Calculates the depth of the lake at the base of a partially submerged conical hill -/
def lake_depth (cone : SubmergedCone) : ℝ :=
  cone.total_height * (1 - (1 - cone.volume_ratio_above_water) ^ (1/3))

theorem lake_depth_for_specific_cone :
  let cone : SubmergedCone := ⟨5000, 1/5⟩
  lake_depth cone = 660 := by
  sorry

end NUMINAMATH_CALUDE_lake_depth_for_specific_cone_l2938_293801


namespace NUMINAMATH_CALUDE_olivers_candy_l2938_293811

/-- Oliver's Halloween candy problem -/
theorem olivers_candy (initial_candy : ℕ) : 
  (initial_candy - 10 = 68) → initial_candy = 78 := by
  sorry

end NUMINAMATH_CALUDE_olivers_candy_l2938_293811


namespace NUMINAMATH_CALUDE_perimeterDifference_l2938_293873

/-- Calculates the perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- Calculates the perimeter of an L-shaped formation (2x2 square missing 1x1 square) -/
def lShapePerimeter : ℕ := 5

/-- Calculates the perimeter of Figure 1 (composite of 3x1 rectangle and L-shape) -/
def figure1Perimeter : ℕ :=
  rectanglePerimeter 3 1 + lShapePerimeter

/-- Calculates the perimeter of Figure 2 (6x2 rectangle) -/
def figure2Perimeter : ℕ :=
  rectanglePerimeter 6 2

/-- The main theorem stating the positive difference in perimeters -/
theorem perimeterDifference :
  (max figure1Perimeter figure2Perimeter) - (min figure1Perimeter figure2Perimeter) = 3 := by
  sorry

end NUMINAMATH_CALUDE_perimeterDifference_l2938_293873


namespace NUMINAMATH_CALUDE_givenEquationIsQuadratic_l2938_293882

/-- Represents a polynomial equation with one variable -/
structure PolynomialEquation :=
  (a b c : ℝ)

/-- Defines a quadratic equation with one variable -/
def IsQuadraticOneVariable (eq : PolynomialEquation) : Prop :=
  eq.a ≠ 0

/-- The specific equation we're considering -/
def givenEquation : PolynomialEquation :=
  { a := 1, b := 1, c := 3 }

/-- Theorem stating that the given equation is a quadratic equation with one variable -/
theorem givenEquationIsQuadratic : IsQuadraticOneVariable givenEquation := by
  sorry


end NUMINAMATH_CALUDE_givenEquationIsQuadratic_l2938_293882


namespace NUMINAMATH_CALUDE_supplement_triple_angle_l2938_293848

theorem supplement_triple_angle : ∃ (x : ℝ), x > 0 ∧ x < 180 ∧ x = 3 * (180 - x) ∧ x = 135 := by
  sorry

end NUMINAMATH_CALUDE_supplement_triple_angle_l2938_293848


namespace NUMINAMATH_CALUDE_plate_cup_cost_l2938_293894

/-- Given the cost of 20 plates and 40 cups, calculate the cost of 100 plates and 200 cups -/
theorem plate_cup_cost (plate_cost cup_cost : ℝ) : 
  20 * plate_cost + 40 * cup_cost = 1.50 → 
  100 * plate_cost + 200 * cup_cost = 7.50 := by
  sorry

end NUMINAMATH_CALUDE_plate_cup_cost_l2938_293894


namespace NUMINAMATH_CALUDE_reflection_point_properties_l2938_293844

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A concave spherical mirror -/
structure SphericalMirror where
  radius : ℝ
  center : Point

/-- The reflection point on a spherical mirror -/
def reflection_point (mirror : SphericalMirror) (A B : Point) : Point :=
  sorry

/-- Theorem: The reflection point satisfies the sphere equation and reflection equation -/
theorem reflection_point_properties (mirror : SphericalMirror) (A B : Point) :
  let X := reflection_point mirror A B
  (X.x^2 + X.y^2 = mirror.radius^2) ∧
  ((A.x * B.y + B.x * A.y) * (X.x^2 - X.y^2) - 
   2 * (A.x * B.x - A.y * B.y) * X.x * X.y + 
   mirror.radius^2 * ((A.x + B.x) * X.y - (A.y + B.y) * X.x) = 0) := by
  sorry


end NUMINAMATH_CALUDE_reflection_point_properties_l2938_293844


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l2938_293837

theorem circle_tangent_to_line (m : ℝ) (h : m ≥ 0) :
  ∃ (x y : ℝ), x^2 + y^2 = m ∧ x + y = Real.sqrt (2 * m) ∧
  ∀ (x' y' : ℝ), x'^2 + y'^2 = m → x' + y' ≤ Real.sqrt (2 * m) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l2938_293837


namespace NUMINAMATH_CALUDE_product_nonnegative_implies_lower_bound_l2938_293874

open Real

theorem product_nonnegative_implies_lower_bound (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  (∀ x : ℝ, x > 0 → (log (a * x) - 1) * (exp x - b) ≥ 0) →
  a * b ≥ exp 2 :=
by sorry

end NUMINAMATH_CALUDE_product_nonnegative_implies_lower_bound_l2938_293874


namespace NUMINAMATH_CALUDE_min_original_tables_l2938_293802

/-- Given a restaurant scenario with customers and tables, prove that the minimum number of original tables is 3. -/
theorem min_original_tables (X Y Z A B C : ℕ) : 
  X = Z + A + B + C →  -- Total customers equals those who left plus those who remained
  Y ≥ 3 :=             -- The original number of tables is at least 3
by sorry

end NUMINAMATH_CALUDE_min_original_tables_l2938_293802


namespace NUMINAMATH_CALUDE_work_hours_constant_l2938_293807

/-- Represents the work schedule for a week -/
structure WorkSchedule where
  days_per_week : ℕ
  initial_hours_task1 : ℕ
  initial_hours_task2 : ℕ
  hours_reduction_task1 : ℕ

/-- Calculates the total weekly work hours -/
def total_weekly_hours (schedule : WorkSchedule) : ℕ :=
  schedule.days_per_week * (schedule.initial_hours_task1 + schedule.initial_hours_task2)

/-- Theorem stating that the total weekly work hours remain constant after redistribution -/
theorem work_hours_constant (schedule : WorkSchedule) 
  (h1 : schedule.days_per_week = 5)
  (h2 : schedule.initial_hours_task1 = 5)
  (h3 : schedule.initial_hours_task2 = 3)
  (h4 : schedule.hours_reduction_task1 = 5) :
  total_weekly_hours schedule = 40 := by
  sorry

#eval total_weekly_hours { days_per_week := 5, initial_hours_task1 := 5, initial_hours_task2 := 3, hours_reduction_task1 := 5 }

end NUMINAMATH_CALUDE_work_hours_constant_l2938_293807


namespace NUMINAMATH_CALUDE_urn_probability_l2938_293885

def Urn := Nat × Nat -- (white balls, black balls)

def initial_urn : Urn := (2, 1)

def operation (u : Urn) : Urn → Prop :=
  fun u' => (u'.1 = u.1 + 1 ∧ u'.2 = u.2) ∨ (u'.1 = u.1 ∧ u'.2 = u.2 + 1)

def final_urn (u : Urn) : Prop := u.1 = 4 ∧ u.2 = 4

def probability_of_drawing_white (u : Urn) : ℚ := u.1 / (u.1 + u.2)

def probability_of_drawing_black (u : Urn) : ℚ := u.2 / (u.1 + u.2)

theorem urn_probability : 
  ∃ (u₁ u₂ u₃ u₄ : Urn),
    operation initial_urn u₁ ∧
    operation u₁ u₂ ∧
    operation u₂ u₃ ∧
    operation u₃ u₄ ∧
    final_urn u₄ ∧
    (probability_of_drawing_white initial_urn *
     probability_of_drawing_white u₁ *
     probability_of_drawing_black u₂ *
     probability_of_drawing_black u₃ +
     probability_of_drawing_white initial_urn *
     probability_of_drawing_black u₁ *
     probability_of_drawing_white u₂ *
     probability_of_drawing_black u₃ +
     probability_of_drawing_white initial_urn *
     probability_of_drawing_black u₁ *
     probability_of_drawing_black u₂ *
     probability_of_drawing_white u₃ +
     probability_of_drawing_black initial_urn *
     probability_of_drawing_white u₁ *
     probability_of_drawing_white u₂ *
     probability_of_drawing_black u₃ +
     probability_of_drawing_black initial_urn *
     probability_of_drawing_white u₁ *
     probability_of_drawing_black u₂ *
     probability_of_drawing_white u₃ +
     probability_of_drawing_black initial_urn *
     probability_of_drawing_black u₁ *
     probability_of_drawing_white u₂ *
     probability_of_drawing_white u₃) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_urn_probability_l2938_293885


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l2938_293831

theorem subtraction_preserves_inequality (a b c : ℝ) (h : a > b) : b - c < a - c := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l2938_293831


namespace NUMINAMATH_CALUDE_l_shapes_on_8x8_board_l2938_293876

/-- Represents a square checkerboard -/
structure Checkerboard :=
  (size : Nat)

/-- Represents an L-shape on the checkerboard -/
structure LShape :=
  (x : Nat) (y : Nat) (orientation : Nat)

/-- The number of different L-shapes on a checkerboard -/
def count_l_shapes (board : Checkerboard) : Nat :=
  sorry

theorem l_shapes_on_8x8_board :
  ∃ (board : Checkerboard),
    board.size = 8 ∧ count_l_shapes board = 196 :=
  sorry

end NUMINAMATH_CALUDE_l_shapes_on_8x8_board_l2938_293876


namespace NUMINAMATH_CALUDE_sum_of_cubes_divisible_by_nine_l2938_293842

theorem sum_of_cubes_divisible_by_nine (x : ℤ) : 
  ∃ k : ℤ, (x - 1)^3 + x^3 + (x + 1)^3 = 9 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_divisible_by_nine_l2938_293842


namespace NUMINAMATH_CALUDE_probability_neither_mix_l2938_293857

/-- Represents the set of buyers -/
def Buyers : Type := Unit

/-- The total number of buyers -/
def total_buyers : ℕ := 100

/-- The number of buyers who purchase cake mix -/
def cake_mix_buyers : ℕ := 50

/-- The number of buyers who purchase muffin mix -/
def muffin_mix_buyers : ℕ := 40

/-- The number of buyers who purchase both cake mix and muffin mix -/
def both_mix_buyers : ℕ := 15

/-- The probability of selecting a buyer who purchases neither cake mix nor muffin mix -/
theorem probability_neither_mix (b : Buyers) : 
  (total_buyers - (cake_mix_buyers + muffin_mix_buyers - both_mix_buyers)) / total_buyers = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_neither_mix_l2938_293857


namespace NUMINAMATH_CALUDE_savings_calculation_l2938_293886

theorem savings_calculation (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : 
  income = 19000 → 
  income_ratio = 5 → 
  expenditure_ratio = 4 → 
  income - (income * expenditure_ratio / income_ratio) = 3800 := by
sorry

end NUMINAMATH_CALUDE_savings_calculation_l2938_293886


namespace NUMINAMATH_CALUDE_stream_speed_stream_speed_problem_l2938_293850

/-- The speed of a stream given a swimmer's still water speed and relative upstream/downstream times -/
theorem stream_speed (still_speed : ℝ) (upstream_downstream_ratio : ℝ) : ℝ :=
  let stream_speed := (still_speed * (upstream_downstream_ratio - 1)) / (upstream_downstream_ratio + 1)
  stream_speed

/-- Proves that the speed of the stream is 0.5 km/h given the conditions -/
theorem stream_speed_problem : stream_speed 1.5 2 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_stream_speed_problem_l2938_293850


namespace NUMINAMATH_CALUDE_james_pizza_slices_l2938_293866

theorem james_pizza_slices :
  let total_slices : ℕ := 20
  let tom_slices : ℕ := 5
  let alice_slices : ℕ := 3
  let bob_slices : ℕ := 4
  let friends_slices : ℕ := tom_slices + alice_slices + bob_slices
  let remaining_slices : ℕ := total_slices - friends_slices
  let james_slices : ℕ := remaining_slices / 2
  james_slices = 4 := by sorry

end NUMINAMATH_CALUDE_james_pizza_slices_l2938_293866


namespace NUMINAMATH_CALUDE_min_distance_line_curve_l2938_293830

/-- The minimum distance between a point on the line 2x - y + 6 = 0 and
    a point on the curve y = 2ln(x) + 2 is 6√5/5 -/
theorem min_distance_line_curve :
  let line := {(x, y) : ℝ × ℝ | 2 * x - y + 6 = 0}
  let curve := {(x, y) : ℝ × ℝ | y = 2 * Real.log x + 2}
  ∃ (d : ℝ), d = 6 * Real.sqrt 5 / 5 ∧
    ∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line → q ∈ curve →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_curve_l2938_293830


namespace NUMINAMATH_CALUDE_rice_stock_calculation_l2938_293822

theorem rice_stock_calculation (initial_stock sold restocked : ℕ) : 
  initial_stock = 55 → sold = 23 → restocked = 132 → 
  initial_stock - sold + restocked = 164 := by
  sorry

end NUMINAMATH_CALUDE_rice_stock_calculation_l2938_293822


namespace NUMINAMATH_CALUDE_f_4_3_2_1_l2938_293834

/-- The mapping f from (a₁, a₂, a₃, a₄) to (b₁, b₂, b₃, b₄) based on the equation
    x^4 + a₁x³ + a₂x² + a₃x + a₄ = (x+1)^4 + b₁(x+1)³ + b₂(x+1)² + b₃(x+1) + b₄ -/
def f (a₁ a₂ a₃ a₄ : ℝ) : ℝ × ℝ × ℝ × ℝ := sorry

/-- Theorem stating that f(4, 3, 2, 1) = (0, -3, 4, -1) -/
theorem f_4_3_2_1 : f 4 3 2 1 = (0, -3, 4, -1) := by sorry

end NUMINAMATH_CALUDE_f_4_3_2_1_l2938_293834


namespace NUMINAMATH_CALUDE_gcd_bn_bn_plus_2_is_one_max_en_is_one_l2938_293827

theorem gcd_bn_bn_plus_2_is_one (n : ℕ) : 
  Nat.gcd ((10^n - 2) / 8) ((10^(n+2) - 2) / 8) = 1 := by
  sorry

theorem max_en_is_one : 
  ∀ n : ℕ, (∃ k : ℕ, Nat.gcd ((10^n - 2) / 8) ((10^(n+2) - 2) / 8) = k) → 
  Nat.gcd ((10^n - 2) / 8) ((10^(n+2) - 2) / 8) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_bn_bn_plus_2_is_one_max_en_is_one_l2938_293827


namespace NUMINAMATH_CALUDE_right_triangle_area_l2938_293828

theorem right_triangle_area (base height : ℝ) (h1 : base = 3) (h2 : height = 4) :
  (1/2 : ℝ) * base * height = 6 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2938_293828


namespace NUMINAMATH_CALUDE_inequality_solution_range_l2938_293819

theorem inequality_solution_range :
  ∀ a : ℝ, (∃ x : ℝ, |x + 1| - |x - 3| < a) ↔ a > -4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l2938_293819


namespace NUMINAMATH_CALUDE_folded_paper_area_l2938_293847

/-- The area of a folded rectangular paper -/
theorem folded_paper_area (length width : ℝ) (h_length : length = 17) (h_width : width = 8) :
  let original_area := length * width
  let folded_triangle_area := (1/2) * width * width
  original_area - folded_triangle_area = 104 :=
by
  sorry


end NUMINAMATH_CALUDE_folded_paper_area_l2938_293847


namespace NUMINAMATH_CALUDE_triangle_inequalities_l2938_293884

/-- Given a triangle with sides a, b, c, prove two inequalities about its radii and semiperimeter. -/
theorem triangle_inequalities 
  (a b c r R r_a r_b r_c p S : ℝ) 
  (h1 : 4 * R + r = r_a + r_b + r_c)
  (h2 : R - 2 * r ≥ 0)
  (h3 : r_a + r_b + r_c = p * r * (1 / (p - a) + 1 / (p - b) + 1 / (p - c)))
  (h4 : 1 / (p - a) + 1 / (p - b) + 1 / (p - c) = (a * b + b * c + c * a - p ^ 2) / S)
  (h5 : p = (a + b + c) / 2)
  (h6 : 2 * (a * b + b * c + c * a) - (a ^ 2 + b ^ 2 + c ^ 2) ≥ 4 * Real.sqrt 3 * S)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ R > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0 ∧ p > 0 ∧ S > 0) :
  (5 * R - r ≥ Real.sqrt 3 * p) ∧ 
  (4 * R - r_a ≥ (p - a) * (Real.sqrt 3 + (a ^ 2 + (b - c) ^ 2) / (2 * S))) := by
  sorry


end NUMINAMATH_CALUDE_triangle_inequalities_l2938_293884


namespace NUMINAMATH_CALUDE_atLeastOneWhite_oneRedOneBlack_mutually_exclusive_not_exhaustive_l2938_293878

-- Define the set of possible ball colors
inductive Color
| Red
| White
| Black

-- Define the bag contents
def bag : Multiset Color :=
  Multiset.replicate 3 Color.Red + Multiset.replicate 2 Color.White + Multiset.replicate 1 Color.Black

-- Define a draw as a pair of colors
def Draw := (Color × Color)

-- Define the event "At least one white ball"
def atLeastOneWhite (draw : Draw) : Prop :=
  draw.1 = Color.White ∨ draw.2 = Color.White

-- Define the event "one red ball and one black ball"
def oneRedOneBlack (draw : Draw) : Prop :=
  (draw.1 = Color.Red ∧ draw.2 = Color.Black) ∨ (draw.1 = Color.Black ∧ draw.2 = Color.Red)

-- Theorem stating that the events are mutually exclusive but not exhaustive
theorem atLeastOneWhite_oneRedOneBlack_mutually_exclusive_not_exhaustive :
  (∀ (draw : Draw), ¬(atLeastOneWhite draw ∧ oneRedOneBlack draw)) ∧
  (∃ (draw : Draw), ¬atLeastOneWhite draw ∧ ¬oneRedOneBlack draw) :=
sorry

end NUMINAMATH_CALUDE_atLeastOneWhite_oneRedOneBlack_mutually_exclusive_not_exhaustive_l2938_293878


namespace NUMINAMATH_CALUDE_sequence_formula_l2938_293815

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define the properties of the sequence
def PropertyOne (a : Sequence) : Prop :=
  ∀ m n : ℕ, m > n → a (m - n) = a m - a n

def PropertyTwo (a : Sequence) : Prop :=
  ∀ m n : ℕ, m > n → a m > a n

-- State the theorem
theorem sequence_formula (a : Sequence) 
  (h1 : PropertyOne a) (h2 : PropertyTwo a) : 
  ∃ k : ℝ, k > 0 ∧ ∀ n : ℕ, a n = k * n := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l2938_293815


namespace NUMINAMATH_CALUDE_product_abcd_l2938_293826

theorem product_abcd (a b c d : ℚ) : 
  (2 * a + 3 * b + 5 * c + 8 * d = 45) →
  (4 * (d + c) = b) →
  (4 * b + c = a) →
  (c + 1 = d) →
  (a * b * c * d = (1511 / 103) * (332 / 103) * (-7 / 103) * (96 / 103)) := by
  sorry

end NUMINAMATH_CALUDE_product_abcd_l2938_293826


namespace NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l2938_293854

theorem inverse_proportion_percentage_change 
  (x y p : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hp : p > 0) 
  (h_inverse : ∃ k, k > 0 ∧ x * y = k) :
  let x' := x * (1 + 2*p/100)
  let y' := y * 100 / (100 + 2*p)
  (y - y') / y * 100 = 200 * p / (100 + 2*p) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_percentage_change_l2938_293854


namespace NUMINAMATH_CALUDE_decimal_expansion_of_prime_reciprocal_l2938_293800

/-- The type of natural numbers greater than 1 -/
def PositiveNatGT1 := { n : ℕ // n > 1 }

/-- The period of a rational number's decimal expansion -/
def decimalPeriod (q : ℚ) : ℕ := sorry

/-- The nth digit in the decimal expansion of a rational number -/
def nthDecimalDigit (q : ℚ) (n : ℕ) : Fin 10 := sorry

theorem decimal_expansion_of_prime_reciprocal (p : PositiveNatGT1) 
  (h_prime : Nat.Prime p.val) 
  (h_period : decimalPeriod (1 / p.val) = 200) : 
  nthDecimalDigit (1 / p.val) 101 = 9 := by sorry

end NUMINAMATH_CALUDE_decimal_expansion_of_prime_reciprocal_l2938_293800


namespace NUMINAMATH_CALUDE_transylvania_statements_l2938_293860

/-- Represents a statement that can be made by a resident of Transylvania -/
structure Statement :=
  (proposition : Prop)

/-- Defines what it means for one statement to be the converse of another -/
def is_converse (X Y : Statement) : Prop :=
  ∃ P Q : Prop, X.proposition = (P → Q) ∧ Y.proposition = (Q → P)

/-- Defines the property that asserting one statement implies the truth of another -/
def implies_truth (X Y : Statement) : Prop :=
  ∀ (resident : Prop), (resident → X.proposition) → Y.proposition

/-- The main theorem stating the existence of two statements satisfying the given conditions -/
theorem transylvania_statements : ∃ (X Y : Statement),
  is_converse X Y ∧
  (¬ (X.proposition → Y.proposition)) ∧
  (¬ (Y.proposition → X.proposition)) ∧
  implies_truth X Y ∧
  implies_truth Y X := by
  sorry

end NUMINAMATH_CALUDE_transylvania_statements_l2938_293860


namespace NUMINAMATH_CALUDE_smallest_cut_length_for_non_triangle_l2938_293803

theorem smallest_cut_length_for_non_triangle : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (∀ (y : ℕ), y < x → (9 - y) + (16 - y) > (18 - y)) ∧
  ((9 - x) + (16 - x) ≤ (18 - x)) ∧ 
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cut_length_for_non_triangle_l2938_293803


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2938_293804

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

/-- The theorem states that if the ratio of S_{3n} to S_n is constant
    for all positive integers n, then the first term of the sequence is 5/2 -/
theorem arithmetic_sequence_first_term
  (h : ∃ (c : ℚ), ∀ (n : ℕ), n > 0 → S a (3 * n) / S a n = c) :
  a = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2938_293804


namespace NUMINAMATH_CALUDE_exists_monochromatic_equilateral_triangle_l2938_293806

/-- A color type representing red or blue -/
inductive Color
  | Red
  | Blue

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A coloring function that assigns a color to each point in the plane -/
def Coloring := Point → Color

/-- Predicate to check if three points form an equilateral triangle -/
def IsEquilateralTriangle (p1 p2 p3 : Point) : Prop := sorry

/-- Theorem stating that in any coloring of the plane, there exist three points
    of the same color forming an equilateral triangle -/
theorem exists_monochromatic_equilateral_triangle (c : Coloring) :
  ∃ (p1 p2 p3 : Point) (col : Color),
    c p1 = col ∧ c p2 = col ∧ c p3 = col ∧
    IsEquilateralTriangle p1 p2 p3 := by
  sorry

end NUMINAMATH_CALUDE_exists_monochromatic_equilateral_triangle_l2938_293806


namespace NUMINAMATH_CALUDE_diamond_properties_l2938_293862

def diamond (a b : ℤ) : ℤ := a^2 - 2*b

theorem diamond_properties :
  (diamond (-1) 2 = -3) ∧
  (∃ a b : ℤ, diamond a b ≠ diamond b a) := by sorry

end NUMINAMATH_CALUDE_diamond_properties_l2938_293862


namespace NUMINAMATH_CALUDE_banana_arrangements_count_l2938_293887

/-- The number of distinct arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct arrangements of the letters in "BANANA" is 60 -/
theorem banana_arrangements_count : banana_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_count_l2938_293887


namespace NUMINAMATH_CALUDE_adult_ticket_price_l2938_293889

/-- Proves that the price of an adult ticket is $32 given the specified conditions -/
theorem adult_ticket_price
  (num_adults : ℕ)
  (num_children : ℕ)
  (total_amount : ℕ)
  (h_adults : num_adults = 400)
  (h_children : num_children = 200)
  (h_total : total_amount = 16000)
  (h_price_ratio : ∃ (child_price : ℕ), 
    total_amount = num_adults * (2 * child_price) + num_children * child_price) :
  ∃ (adult_price : ℕ), adult_price = 32 ∧
    total_amount = num_adults * adult_price + num_children * (adult_price / 2) :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l2938_293889


namespace NUMINAMATH_CALUDE_x_gt_one_necessary_not_sufficient_for_x_gt_two_l2938_293888

theorem x_gt_one_necessary_not_sufficient_for_x_gt_two :
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_necessary_not_sufficient_for_x_gt_two_l2938_293888


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2938_293879

/-- The eccentricity of a hyperbola given its equation and a point it passes through -/
theorem hyperbola_eccentricity (m : ℝ) (h : 2 - 4 / m = 1) : 
  Real.sqrt (1 + 4 / 2) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2938_293879


namespace NUMINAMATH_CALUDE_factorial_less_than_power_l2938_293898

theorem factorial_less_than_power (n : ℕ) (h : n > 1) : 
  Nat.factorial n < ((n + 1) / 2 : ℚ) ^ n := by
  sorry

end NUMINAMATH_CALUDE_factorial_less_than_power_l2938_293898


namespace NUMINAMATH_CALUDE_union_of_sets_l2938_293846

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 4, 6}
  A ∪ B = {1, 2, 4, 6} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l2938_293846
