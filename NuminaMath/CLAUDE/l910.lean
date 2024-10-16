import Mathlib

namespace NUMINAMATH_CALUDE_negative_cube_inequality_l910_91039

theorem negative_cube_inequality (a : ℝ) (h : a < 0) : a^3 ≠ (-a)^3 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_inequality_l910_91039


namespace NUMINAMATH_CALUDE_fractional_parts_sum_l910_91087

theorem fractional_parts_sum (x : ℝ) (h : x^3 + 1/x^3 = 18) :
  (x - ⌊x⌋) + (1/x - ⌊1/x⌋) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fractional_parts_sum_l910_91087


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l910_91009

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

/-- Theorem: In an arithmetic sequence where a_3 + a_7 = 38, the sum a_2 + a_4 + a_6 + a_8 = 76 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 3 + a 7 = 38) : 
  a 2 + a 4 + a 6 + a 8 = 76 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l910_91009


namespace NUMINAMATH_CALUDE_lemon_pie_degree_measure_l910_91031

theorem lemon_pie_degree_measure (total_students : ℕ) (chocolate_pref : ℕ) (apple_pref : ℕ) (blueberry_pref : ℕ) 
  (h_total : total_students = 45)
  (h_chocolate : chocolate_pref = 15)
  (h_apple : apple_pref = 10)
  (h_blueberry : blueberry_pref = 9)
  (h_remaining : (total_students - (chocolate_pref + apple_pref + blueberry_pref)) % 2 = 0) :
  let remaining := total_students - (chocolate_pref + apple_pref + blueberry_pref)
  let lemon_pref := remaining / 2
  ↑lemon_pref / ↑total_students * 360 = 48 := by
sorry

end NUMINAMATH_CALUDE_lemon_pie_degree_measure_l910_91031


namespace NUMINAMATH_CALUDE_stadium_length_in_feet_l910_91020

/-- Converts yards to feet using the standard conversion factor. -/
def yards_to_feet (yards : ℕ) : ℕ := yards * 3

/-- The length of the sports stadium in yards. -/
def stadium_length_yards : ℕ := 61

theorem stadium_length_in_feet :
  yards_to_feet stadium_length_yards = 183 := by
  sorry

end NUMINAMATH_CALUDE_stadium_length_in_feet_l910_91020


namespace NUMINAMATH_CALUDE_points_difference_is_integer_impossible_score_difference_l910_91058

/-- Represents the possible outcomes of a chess game -/
inductive GameOutcome
  | Victory
  | Draw
  | Defeat

/-- Calculates the points scored for a given game outcome -/
def points_scored (outcome : GameOutcome) : ℚ :=
  match outcome with
  | GameOutcome.Victory => 1
  | GameOutcome.Draw => 1/2
  | GameOutcome.Defeat => 0

/-- Calculates the points lost for a given game outcome -/
def points_lost (outcome : GameOutcome) : ℚ :=
  match outcome with
  | GameOutcome.Victory => 0
  | GameOutcome.Draw => 1/2
  | GameOutcome.Defeat => 1

/-- Represents a sequence of game outcomes in a chess tournament -/
def Tournament := List GameOutcome

/-- Calculates the total points scored in a tournament -/
def total_points_scored (tournament : Tournament) : ℚ :=
  tournament.map points_scored |>.sum

/-- Calculates the total points lost in a tournament -/
def total_points_lost (tournament : Tournament) : ℚ :=
  tournament.map points_lost |>.sum

/-- Theorem: The difference between points scored and points lost in any chess tournament is always an integer -/
theorem points_difference_is_integer (tournament : Tournament) :
  ∃ n : ℤ, total_points_scored tournament - total_points_lost tournament = n :=
sorry

/-- Corollary: It's impossible to have scored exactly 3.5 points more than lost -/
theorem impossible_score_difference (tournament : Tournament) :
  total_points_scored tournament - total_points_lost tournament ≠ 7/2 :=
sorry

end NUMINAMATH_CALUDE_points_difference_is_integer_impossible_score_difference_l910_91058


namespace NUMINAMATH_CALUDE_sequence_theorem_l910_91053

def sequence_property (a : ℕ → ℕ) : Prop :=
  (∀ n, a n ∈ ({0, 1} : Set ℕ)) ∧
  (∀ n, a n + a (n + 1) ≠ a (n + 2) + a (n + 3)) ∧
  (∀ n, a n + a (n + 1) + a (n + 2) ≠ a (n + 3) + a (n + 4) + a (n + 5))

theorem sequence_theorem (a : ℕ → ℕ) (h : sequence_property a) (h1 : a 1 = 0) :
  a 2020 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_theorem_l910_91053


namespace NUMINAMATH_CALUDE_valid_quadruples_l910_91054

def is_valid_quadruple (p q r n : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ 
  ¬(3 ∣ (p + q)) ∧
  p + q = r * (p - q)^n

theorem valid_quadruples :
  ∀ p q r n : ℕ,
    is_valid_quadruple p q r n →
    ((p = 2 ∧ q = 3 ∧ r = 5 ∧ Even n) ∨
     (p = 3 ∧ q = 2 ∧ r = 5) ∨
     (p = 5 ∧ q = 3 ∧ r = 1 ∧ n = 3) ∨
     (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 2) ∨
     (p = 5 ∧ q = 3 ∧ r = 8 ∧ n = 1) ∨
     (p = 3 ∧ q = 5 ∧ r = 1 ∧ n = 3) ∨
     (p = 3 ∧ q = 5 ∧ r = 2 ∧ n = 2) ∨
     (p = 3 ∧ q = 5 ∧ r = 8 ∧ n = 1)) :=
by sorry


end NUMINAMATH_CALUDE_valid_quadruples_l910_91054


namespace NUMINAMATH_CALUDE_tree_height_proof_l910_91040

/-- Proves that a tree with a current height of 180 inches, which is 50% taller than its original height, had an original height of 10 feet. -/
theorem tree_height_proof (current_height : ℝ) (growth_factor : ℝ) (inches_per_foot : ℝ) :
  current_height = 180 ∧
  growth_factor = 1.5 ∧
  inches_per_foot = 12 →
  current_height / growth_factor / inches_per_foot = 10 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_proof_l910_91040


namespace NUMINAMATH_CALUDE_product_equality_l910_91000

theorem product_equality (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l910_91000


namespace NUMINAMATH_CALUDE_Q_equals_two_three_four_l910_91056

-- Define the set P
def P : Set ℕ := {1, 2}

-- Define the set Q
def Q : Set ℕ := {z | ∃ x y, x ∈ P ∧ y ∈ P ∧ z = x + y}

-- Theorem statement
theorem Q_equals_two_three_four : Q = {2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_Q_equals_two_three_four_l910_91056


namespace NUMINAMATH_CALUDE_canoe_trip_average_distance_l910_91013

/-- Proves that given a 6-day canoe trip with a total distance of 168 km, 
    where 3/7 of the distance is completed in 3 days, 
    the average distance per day for the remaining days is 32 km. -/
theorem canoe_trip_average_distance 
  (total_distance : ℝ) 
  (total_days : ℕ) 
  (completed_fraction : ℚ) 
  (completed_days : ℕ) 
  (h1 : total_distance = 168)
  (h2 : total_days = 6)
  (h3 : completed_fraction = 3/7)
  (h4 : completed_days = 3) : 
  (total_distance * (1 - completed_fraction)) / (total_days - completed_days) = 32 := by
  sorry

end NUMINAMATH_CALUDE_canoe_trip_average_distance_l910_91013


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l910_91062

/-- Represents different types of sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a sampling technique -/
structure SamplingTechnique where
  method : SamplingMethod
  description : String

/-- Represents the survey conducted by the school -/
structure Survey where
  totalStudents : Nat
  technique1 : SamplingTechnique
  technique2 : SamplingTechnique

/-- The actual survey conducted by the school -/
def schoolSurvey : Survey :=
  { totalStudents := 200,
    technique1 := 
      { method := SamplingMethod.SimpleRandom,
        description := "Random selection of 20 students by the student council" },
    technique2 := 
      { method := SamplingMethod.Systematic,
        description := "Students numbered from 001 to 200, those with last digit 2 are selected" }
  }

/-- Theorem stating that the sampling methods are correctly identified -/
theorem correct_sampling_methods :
  schoolSurvey.technique1.method = SamplingMethod.SimpleRandom ∧
  schoolSurvey.technique2.method = SamplingMethod.Systematic :=
by sorry


end NUMINAMATH_CALUDE_correct_sampling_methods_l910_91062


namespace NUMINAMATH_CALUDE_total_cost_special_requirement_l910_91055

/-- The number of ways to choose 3 consecutive numbers from 01 to 10 -/
def consecutive_three_from_ten : Nat := 8

/-- The number of ways to choose 2 consecutive numbers from 11 to 20 -/
def consecutive_two_from_ten : Nat := 9

/-- The number of ways to choose 1 number from 21 to 30 -/
def one_from_ten : Nat := 10

/-- The number of ways to choose 1 number from 31 to 36 -/
def one_from_six : Nat := 6

/-- The cost of a single entry in yuan -/
def entry_cost : Nat := 2

/-- Theorem: The total cost of purchasing all possible entries meeting the special requirement is 8640 yuan -/
theorem total_cost_special_requirement : 
  consecutive_three_from_ten * consecutive_two_from_ten * one_from_ten * one_from_six * entry_cost = 8640 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_special_requirement_l910_91055


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l910_91061

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the shaded quadrilateral -/
structure ShadedQuadrilateral where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- The side length of the large square -/
def largeSideLength : ℝ := 10

/-- The side length of each small square in the grid -/
def smallSideLength : ℝ := 2

/-- The number of squares in each row/column of the grid -/
def gridSize : ℕ := 5

/-- Function to calculate the area of the shaded quadrilateral -/
def shadedArea (quad : ShadedQuadrilateral) : ℝ := sorry

/-- Function to create the shaded quadrilateral based on the problem description -/
def createShadedQuadrilateral : ShadedQuadrilateral := sorry

/-- Theorem stating the ratio of shaded area to large square area -/
theorem shaded_area_ratio :
  let quad := createShadedQuadrilateral
  shadedArea quad / (largeSideLength ^ 2) = 1 / 50 := by sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l910_91061


namespace NUMINAMATH_CALUDE_functional_equation_solution_l910_91014

/-- A function satisfying g(xy) = xg(y) for all real numbers x and y -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) = x * g y

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h1 : FunctionalEquation g) (h2 : g 1 = 30) : 
  g 50 = 1500 ∧ g 0.5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l910_91014


namespace NUMINAMATH_CALUDE_school_committee_formation_l910_91081

theorem school_committee_formation (n_students : ℕ) (n_teachers : ℕ) (committee_size : ℕ) : 
  n_students = 11 → n_teachers = 3 → committee_size = 8 →
  (Nat.choose (n_students + n_teachers) committee_size) - (Nat.choose n_students committee_size) = 2838 :=
by sorry

end NUMINAMATH_CALUDE_school_committee_formation_l910_91081


namespace NUMINAMATH_CALUDE_sum_xyz_is_zero_l910_91010

theorem sum_xyz_is_zero (x y z : ℝ) 
  (eq1 : x + y = 2*x + z)
  (eq2 : x - 2*y = 4*z)
  (eq3 : y = 6*z) : 
  x + y + z = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_xyz_is_zero_l910_91010


namespace NUMINAMATH_CALUDE_alices_number_l910_91022

theorem alices_number (y : ℝ) : 3 * (3 * y + 15) = 135 → y = 10 := by
  sorry

end NUMINAMATH_CALUDE_alices_number_l910_91022


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l910_91036

/-- Given a hyperbola and a circle, if the length of the chord intercepted on the hyperbola's
    asymptotes by the circle is 2, then the eccentricity of the hyperbola is √6/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6*x + 5 = 0}
  let asymptotes := {(x, y) : ℝ × ℝ | y = b/a * x ∨ y = -b/a * x}
  let chord_length := 2
  chord_length = Real.sqrt (4 - 9 * b^2 / (a^2 + b^2)) * 2 →
  (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l910_91036


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_l910_91078

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : (1 : ℝ) * (2 * x^2)^3 - x^2 * x^4 = 7 * x^6 := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) : (a + b)^2 - b * (2 * a + b) = a^2 := by sorry

-- Problem 3
theorem simplify_expression_3 (x : ℝ) : (x + 1) * (x - 1) - x^2 = -1 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_l910_91078


namespace NUMINAMATH_CALUDE_absolute_value_five_l910_91012

theorem absolute_value_five (x : ℝ) : |x| = 5 → x = 5 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_five_l910_91012


namespace NUMINAMATH_CALUDE_square_has_most_symmetry_l910_91082

-- Define the types of figures
inductive Figure
  | EquilateralTriangle
  | NonSquareRhombus
  | NonSquareRectangle
  | IsoscelesTrapezoid
  | Square

-- Function to get the number of lines of symmetry for each figure
def linesOfSymmetry (f : Figure) : ℕ :=
  match f with
  | Figure.EquilateralTriangle => 3
  | Figure.NonSquareRhombus => 2
  | Figure.NonSquareRectangle => 2
  | Figure.IsoscelesTrapezoid => 1
  | Figure.Square => 4

-- Theorem stating that the square has the greatest number of lines of symmetry
theorem square_has_most_symmetry :
  ∀ f : Figure, linesOfSymmetry Figure.Square ≥ linesOfSymmetry f :=
by
  sorry


end NUMINAMATH_CALUDE_square_has_most_symmetry_l910_91082


namespace NUMINAMATH_CALUDE_rectangle_square_overlap_l910_91025

/-- Given a rectangle ABCD and a square IJKL, if the rectangle shares 40% of its area with the square,
    and the square shares 25% of its area with the rectangle, then the ratio of the length of AB to AD
    in the rectangle is 8. -/
theorem rectangle_square_overlap (AB AD s : ℝ) (h1 : AB > 0) (h2 : AD > 0) (h3 : s > 0) : 
  (0.4 * AB * AD = 0.25 * s^2) → AB / AD = 8 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_square_overlap_l910_91025


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l910_91072

/-- The complex number z = (1 - 2i)^2 is in the third quadrant of the complex plane -/
theorem complex_number_in_third_quadrant : 
  let z : ℂ := (1 - 2*I)^2
  (z.re < 0) ∧ (z.im < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l910_91072


namespace NUMINAMATH_CALUDE_infinitely_many_M_exist_l910_91043

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Check if a natural number has no zero digits -/
def hasNoZeroDigits (n : ℕ) : Prop := sorry

/-- The main theorem -/
theorem infinitely_many_M_exist (N : ℕ) (hN : N > 0) :
  ∀ k : ℕ, ∃ M : ℕ, M > k ∧ hasNoZeroDigits M ∧ sumOfDigits (N * M) = sumOfDigits M :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_M_exist_l910_91043


namespace NUMINAMATH_CALUDE_expression_evaluation_l910_91077

theorem expression_evaluation (x y : ℤ) (hx : x = -2) (hy : y = -3) :
  (x - 3*y)^2 + (x - 2*y)*(x + 2*y) - x*(2*x - 5*y) - y = 42 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l910_91077


namespace NUMINAMATH_CALUDE_expression_value_l910_91074

theorem expression_value (a b : ℚ) (h : 4 * b = 3 + 4 * a) :
  a + (a - (a - (a - b) - b) - b) - b = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l910_91074


namespace NUMINAMATH_CALUDE_one_volleyball_outside_range_l910_91011

def volleyball_weights : List ℝ := [275, 263, 278, 270, 261, 277, 282, 269]
def standard_weight : ℝ := 270
def tolerance : ℝ := 10

theorem one_volleyball_outside_range : 
  (volleyball_weights.filter (λ w => w < standard_weight - tolerance ∨ 
                                     w > standard_weight + tolerance)).length = 1 :=
by sorry

end NUMINAMATH_CALUDE_one_volleyball_outside_range_l910_91011


namespace NUMINAMATH_CALUDE_distance_difference_l910_91050

/-- The width of the streets in Longtown -/
def street_width : ℝ := 30

/-- The length of the longer side of the block -/
def block_length : ℝ := 500

/-- The length of the shorter side of the block -/
def block_width : ℝ := 300

/-- The distance Jenny runs around the block -/
def jenny_distance : ℝ := 2 * (block_length + block_width)

/-- The distance Jeremy runs around the block -/
def jeremy_distance : ℝ := 2 * ((block_length + 2 * street_width) + (block_width + 2 * street_width))

/-- Theorem stating the difference in distance run by Jeremy and Jenny -/
theorem distance_difference : jeremy_distance - jenny_distance = 240 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l910_91050


namespace NUMINAMATH_CALUDE_right_triangle_area_l910_91089

/-- The area of a right triangle with a leg of 28 inches and a hypotenuse of 30 inches is 28√29 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 28) (h2 : c = 30) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 28 * Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l910_91089


namespace NUMINAMATH_CALUDE_equal_solutions_iff_n_eq_neg_one_third_l910_91046

theorem equal_solutions_iff_n_eq_neg_one_third 
  (x y n : ℝ) : 
  (2 * x - 5 * y = 3 * n + 7 ∧ x - 3 * y = 4) → 
  (∃! (x y : ℝ), 2 * x - 5 * y = 3 * n + 7 ∧ x - 3 * y = 4) ↔ 
  n = -1/3 := by
sorry

end NUMINAMATH_CALUDE_equal_solutions_iff_n_eq_neg_one_third_l910_91046


namespace NUMINAMATH_CALUDE_odd_sum_even_product_l910_91032

theorem odd_sum_even_product (a b : ℤ) : 
  Odd (a + b) → Even (a * b) := by sorry

end NUMINAMATH_CALUDE_odd_sum_even_product_l910_91032


namespace NUMINAMATH_CALUDE_gcd_180_270_l910_91059

theorem gcd_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_270_l910_91059


namespace NUMINAMATH_CALUDE_average_marks_equals_85_l910_91096

def english_marks : ℕ := 86
def math_marks : ℕ := 89
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 87
def biology_marks : ℕ := 81

def total_marks : ℕ := english_marks + math_marks + physics_marks + chemistry_marks + biology_marks
def num_subjects : ℕ := 5

theorem average_marks_equals_85 : (total_marks : ℚ) / num_subjects = 85 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_equals_85_l910_91096


namespace NUMINAMATH_CALUDE_train_speed_l910_91092

/-- The speed of a train given specific passing times -/
theorem train_speed (t_pole : ℝ) (t_stationary : ℝ) (l_stationary : ℝ) 
  (h_pole : t_pole = 10)
  (h_stationary : t_stationary = 30)
  (h_length : l_stationary = 600) :
  ∃ v : ℝ, v = 30 ∧ v * t_pole = v * t_stationary - l_stationary :=
by sorry

end NUMINAMATH_CALUDE_train_speed_l910_91092


namespace NUMINAMATH_CALUDE_log_sum_40_25_l910_91069

theorem log_sum_40_25 : Real.log 40 + Real.log 25 = 3 * Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_40_25_l910_91069


namespace NUMINAMATH_CALUDE_average_speed_calculation_l910_91094

/-- Proves that the average speed for a 60-mile trip with specified conditions is 30 mph -/
theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (speed_increase : ℝ) :
  total_distance = 60 →
  first_half_speed = 24 →
  speed_increase = 16 →
  let second_half_speed := first_half_speed + speed_increase
  let first_half_time := (total_distance / 2) / first_half_speed
  let second_half_time := (total_distance / 2) / second_half_speed
  let total_time := first_half_time + second_half_time
  total_distance / total_time = 30 := by
  sorry

#check average_speed_calculation

end NUMINAMATH_CALUDE_average_speed_calculation_l910_91094


namespace NUMINAMATH_CALUDE_inequality_proof_l910_91034

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) :
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 ∧
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry


end NUMINAMATH_CALUDE_inequality_proof_l910_91034


namespace NUMINAMATH_CALUDE_system_solution_exists_and_unique_l910_91080

theorem system_solution_exists_and_unique :
  ∃! (x y z : ℝ), 
    8 * (x^3 + y^3 + z^3) = 73 ∧
    2 * (x^2 + y^2 + z^2) = 3 * (x*y + y*z + z*x) ∧
    x * y * z = 1 ∧
    x = 1 ∧ y = 2 ∧ z = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_and_unique_l910_91080


namespace NUMINAMATH_CALUDE_inverse_proportion_l910_91023

/-- Given that α is inversely proportional to β, prove that when α = 5 for β = 10, 
    then α = 25/2 for β = 4 -/
theorem inverse_proportion (α β : ℝ) (k : ℝ) (h1 : α * β = k) 
    (h2 : 5 * 10 = k) : 
  4 * (25/2 : ℝ) = k := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l910_91023


namespace NUMINAMATH_CALUDE_line_parallel_to_x_axis_l910_91008

/-- A line through two points (x₁, y₁) and (x₂, y₂) is parallel to the x-axis if and only if y₁ = y₂ -/
def parallel_to_x_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ = y₂

/-- The problem statement -/
theorem line_parallel_to_x_axis (k : ℝ) :
  parallel_to_x_axis 3 (2*k + 1) 8 (4*k - 5) ↔ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_to_x_axis_l910_91008


namespace NUMINAMATH_CALUDE_speed_increase_for_time_reduction_l910_91071

/-- Proves that increasing speed by 20 km/h reduces trip time by 30 minutes for a 60 km journey at 40 km/h -/
theorem speed_increase_for_time_reduction (distance : ℝ) (initial_speed : ℝ) (time_reduction : ℝ) :
  distance = 60 →
  initial_speed = 40 →
  time_reduction = 0.5 →
  ∃ (speed_increase : ℝ),
    speed_increase = 20 ∧
    (distance / (initial_speed + speed_increase)) = (distance / initial_speed) - time_reduction :=
by sorry

end NUMINAMATH_CALUDE_speed_increase_for_time_reduction_l910_91071


namespace NUMINAMATH_CALUDE_correlation_count_correlated_relationships_l910_91037

/-- Represents a relationship between two quantities -/
structure Relationship where
  name : String
  has_correlation : Bool

/-- The set of relationships given in the problem -/
def relationships : List Relationship := [
  ⟨"cube volume-edge length", false⟩,
  ⟨"yield-fertilizer", true⟩,
  ⟨"height-age", true⟩,
  ⟨"expenses-income", true⟩,
  ⟨"electricity consumption-price", false⟩
]

/-- The correct answer is that exactly three relationships have correlations -/
theorem correlation_count :
  (relationships.filter (fun r => r.has_correlation)).length = 3 := by
  sorry

/-- The relationships with correlations are yield-fertilizer, height-age, and expenses-income -/
theorem correlated_relationships :
  (relationships.filter (fun r => r.has_correlation)).map (fun r => r.name) =
    ["yield-fertilizer", "height-age", "expenses-income"] := by
  sorry

end NUMINAMATH_CALUDE_correlation_count_correlated_relationships_l910_91037


namespace NUMINAMATH_CALUDE_lines_intersection_l910_91088

def line1 (t : ℝ) : ℝ × ℝ := (1 + 2*t, 2 - 3*t)
def line2 (u : ℝ) : ℝ × ℝ := (-1 + 3*u, 4 + u)

theorem lines_intersection :
  ∃! p : ℝ × ℝ, (∃ t : ℝ, line1 t = p) ∧ (∃ u : ℝ, line2 u = p) :=
  by
    use (-5/11, 46/11)
    sorry

#check lines_intersection

end NUMINAMATH_CALUDE_lines_intersection_l910_91088


namespace NUMINAMATH_CALUDE_sum_of_y_values_l910_91001

/-- Given 5 sets of data points, prove the sum of y values -/
theorem sum_of_y_values 
  (x₁ x₂ x₃ x₄ x₅ y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (h_sum_x : x₁ + x₂ + x₃ + x₄ + x₅ = 150) 
  (h_regression : ∀ x, (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = x → 
    (y₁ + y₂ + y₃ + y₄ + y₅) / 5 = 0.67 * x + 24.9) : 
  y₁ + y₂ + y₃ + y₄ + y₅ = 225 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_y_values_l910_91001


namespace NUMINAMATH_CALUDE_abc_inequality_l910_91085

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 + a*b*c = 4) : a + b + c ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l910_91085


namespace NUMINAMATH_CALUDE_union_of_specific_sets_l910_91095

theorem union_of_specific_sets :
  let A : Set ℕ := {2, 5, 6}
  let B : Set ℕ := {3, 5}
  A ∪ B = {2, 3, 5, 6} := by
sorry

end NUMINAMATH_CALUDE_union_of_specific_sets_l910_91095


namespace NUMINAMATH_CALUDE_staircase_shape_perimeter_l910_91007

/-- A shape formed by cutting out a staircase from a rectangle --/
structure StaircaseShape where
  width : ℝ
  height : ℝ
  step_size : ℝ
  num_steps : ℕ
  total_area : ℝ

/-- Calculate the perimeter of a StaircaseShape --/
def perimeter (shape : StaircaseShape) : ℝ :=
  shape.width + shape.height + shape.step_size * (2 * shape.num_steps)

/-- The main theorem --/
theorem staircase_shape_perimeter : 
  ∀ (shape : StaircaseShape), 
    shape.width = 11 ∧ 
    shape.step_size = 2 ∧ 
    shape.num_steps = 10 ∧ 
    shape.total_area = 130 →
    perimeter shape = 54.45 := by
  sorry


end NUMINAMATH_CALUDE_staircase_shape_perimeter_l910_91007


namespace NUMINAMATH_CALUDE_added_number_after_doubling_l910_91019

theorem added_number_after_doubling (x : ℕ) (y : ℕ) (h : x = 19) :
  3 * (2 * x + y) = 129 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_added_number_after_doubling_l910_91019


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_boat_speed_in_still_water_proof_l910_91051

/-- Given a boat that travels 15 km/hr along a stream and 5 km/hr against the same stream,
    its speed in still water is 10 km/hr. -/
theorem boat_speed_in_still_water : ℝ → ℝ → Prop :=
  fun (along_stream : ℝ) (against_stream : ℝ) =>
    along_stream = 15 ∧ against_stream = 5 →
    ∃ (boat_speed stream_speed : ℝ),
      boat_speed + stream_speed = along_stream ∧
      boat_speed - stream_speed = against_stream ∧
      boat_speed = 10

/-- Proof of the theorem -/
theorem boat_speed_in_still_water_proof :
  boat_speed_in_still_water 15 5 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_boat_speed_in_still_water_proof_l910_91051


namespace NUMINAMATH_CALUDE_smallest_number_with_conditions_l910_91097

theorem smallest_number_with_conditions : 
  ∃ (n : ℕ), n = 2102 ∧ 
  (11 ∣ n) ∧ 
  (∀ i : ℕ, 3 ≤ i → i ≤ 7 → n % i = 2) ∧
  (∀ m : ℕ, m < n → ¬((11 ∣ m) ∧ (∀ i : ℕ, 3 ≤ i → i ≤ 7 → m % i = 2))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_conditions_l910_91097


namespace NUMINAMATH_CALUDE_five_consecutive_not_square_l910_91073

theorem five_consecutive_not_square (n : ℕ) : 
  ∃ (m : ℕ), n * (n + 1) * (n + 2) * (n + 3) * (n + 4) ≠ m^2 := by
sorry

end NUMINAMATH_CALUDE_five_consecutive_not_square_l910_91073


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l910_91049

theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 27) : 
  r - p = 34 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l910_91049


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l910_91052

/-- Given two hyperbolas with the same asymptotes, prove the value of T -/
theorem hyperbola_asymptotes (T : ℚ) : 
  (∀ x y, y^2 / 49 - x^2 / 25 = 1 → 
    ∃ k, y = k * x ∧ k^2 = 49 / 25) ∧
  (∀ x y, x^2 / T - y^2 / 18 = 1 → 
    ∃ k, y = k * x ∧ k^2 = 18 / T) →
  T = 450 / 49 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l910_91052


namespace NUMINAMATH_CALUDE_no_prime_solution_l910_91033

theorem no_prime_solution : ¬ ∃ (p : ℕ), 
  Nat.Prime p ∧ 
  p > 8 ∧ 
  2 * p^3 + 7 * p^2 + 6 * p + 20 = 6 * p^2 + 19 * p + 10 := by
sorry

end NUMINAMATH_CALUDE_no_prime_solution_l910_91033


namespace NUMINAMATH_CALUDE_crayons_added_l910_91063

theorem crayons_added (initial : ℕ) (final : ℕ) (added : ℕ) : 
  initial = 7 → final = 10 → initial + added = final → added = 3 := by
  sorry

end NUMINAMATH_CALUDE_crayons_added_l910_91063


namespace NUMINAMATH_CALUDE_chipmunk_increase_l910_91041

/-- Proves that the number of chipmunks increased by 50 given the initial counts, doubling of beavers, and total animal count. -/
theorem chipmunk_increase (initial_beavers initial_chipmunks total_animals : ℕ) 
  (h1 : initial_beavers = 20)
  (h2 : initial_chipmunks = 40)
  (h3 : total_animals = 130) :
  (total_animals - 2 * initial_beavers) - initial_chipmunks = 50 := by
  sorry

#check chipmunk_increase

end NUMINAMATH_CALUDE_chipmunk_increase_l910_91041


namespace NUMINAMATH_CALUDE_magician_earnings_proof_l910_91057

def magician_earnings (price : ℕ) (initial_decks : ℕ) (final_decks : ℕ) : ℕ :=
  (initial_decks - final_decks) * price

theorem magician_earnings_proof (price : ℕ) (initial_decks : ℕ) (final_decks : ℕ) 
  (h1 : price = 2)
  (h2 : initial_decks = 5)
  (h3 : final_decks = 3) :
  magician_earnings price initial_decks final_decks = 4 := by
  sorry

end NUMINAMATH_CALUDE_magician_earnings_proof_l910_91057


namespace NUMINAMATH_CALUDE_triangle_side_length_l910_91083

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 5 →
  c = 2 →
  Real.cos A = 2/3 →
  b = 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l910_91083


namespace NUMINAMATH_CALUDE_matt_card_trade_profit_l910_91064

/-- Represents the profit made from trading cards -/
def card_trade_profit (cards_traded : ℕ) (value_per_traded_card : ℕ) 
                      (received_cards_1 : ℕ) (value_per_received_card_1 : ℕ)
                      (received_cards_2 : ℕ) (value_per_received_card_2 : ℕ) : ℤ :=
  (received_cards_1 * value_per_received_card_1 + received_cards_2 * value_per_received_card_2) -
  (cards_traded * value_per_traded_card)

/-- The profit Matt makes from trading two $6 cards for three $2 cards and one $9 card is $3 -/
theorem matt_card_trade_profit : card_trade_profit 2 6 3 2 1 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_matt_card_trade_profit_l910_91064


namespace NUMINAMATH_CALUDE_proportion_third_number_l910_91029

theorem proportion_third_number (y : ℝ) : 
  (0.75 : ℝ) / 1.35 = y / 9 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_third_number_l910_91029


namespace NUMINAMATH_CALUDE_households_with_bike_only_l910_91068

theorem households_with_bike_only 
  (total : ℕ) 
  (neither : ℕ) 
  (both : ℕ) 
  (with_car : ℕ) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 20)
  (h4 : with_car = 44) :
  total - neither - with_car + both = 35 := by
  sorry

end NUMINAMATH_CALUDE_households_with_bike_only_l910_91068


namespace NUMINAMATH_CALUDE_systems_solution_l910_91017

theorem systems_solution :
  -- System (1)
  (∃ x y : ℝ, y = 2 * x ∧ 3 * y + 2 * x = 8 ∧ x = 1 ∧ y = 2) ∧
  -- System (2)
  (∃ s t : ℝ, 2 * s - 3 * t = 2 ∧ (s + 2 * t) / 3 = 3 / 2 ∧ s = 5 / 2 ∧ t = 1) :=
by
  sorry

#check systems_solution

end NUMINAMATH_CALUDE_systems_solution_l910_91017


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l910_91093

theorem square_perimeter_problem (A B C : ℝ) : 
  -- A, B, and C represent the side lengths of squares A, B, and C respectively
  (4 * A = 20) →  -- Perimeter of A is 20 units
  (4 * B = 32) →  -- Perimeter of B is 32 units
  (C = A / 2 + 2 * B) →  -- Side length of C definition
  (4 * C = 74) -- Perimeter of C is 74 units
  := by sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l910_91093


namespace NUMINAMATH_CALUDE_log_inequality_l910_91042

theorem log_inequality (a b : ℝ) : Real.log a > Real.log b → a > b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l910_91042


namespace NUMINAMATH_CALUDE_f_properties_l910_91048

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.tan x

def is_in_domain (x : ℝ) : Prop :=
  ∀ k : ℤ, x ≠ k * Real.pi + Real.pi / 2

def is_period (p : ℝ) : Prop :=
  ∀ x : ℝ, is_in_domain x → f (x + p) = f x

theorem f_properties :
  (∀ x : ℝ, is_in_domain x ↔ ∃ y : ℝ, f y = f x) ∧
  ¬ is_period Real.pi ∧
  is_period (2 * Real.pi) := by sorry

end NUMINAMATH_CALUDE_f_properties_l910_91048


namespace NUMINAMATH_CALUDE_inequality_system_solution_l910_91098

theorem inequality_system_solution (x : ℝ) : 
  (5 * x - 1 > 3 * (x + 1) ∧ (1/2) * x - 1 ≤ 7 - (3/2) * x) ↔ (2 < x ∧ x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l910_91098


namespace NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l910_91066

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * 3600

/-- Theorem: A space shuttle orbiting at 4 km/s is traveling at 14400 km/h -/
theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 4 = 14400 := by
  sorry

#eval km_per_second_to_km_per_hour 4

end NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l910_91066


namespace NUMINAMATH_CALUDE_system_unique_solution_l910_91027

/-- The system of equations has a unique solution -/
theorem system_unique_solution :
  ∃! (x₁ x₂ x₃ : ℝ),
    3 * x₁ + 4 * x₂ + 3 * x₃ = 0 ∧
    x₁ - x₂ + x₃ = 0 ∧
    x₁ + 3 * x₂ - x₃ = -2 ∧
    x₁ + 2 * x₂ + 3 * x₃ = 2 ∧
    x₁ = 1 ∧ x₂ = 0 ∧ x₃ = 1 := by
  sorry


end NUMINAMATH_CALUDE_system_unique_solution_l910_91027


namespace NUMINAMATH_CALUDE_stating_max_regions_correct_l910_91065

/-- 
Given two points A and B on a plane, with m lines passing through A and n lines passing through B,
this function calculates the maximum number of regions these m+n lines can divide the plane into.
-/
def max_regions (m n : ℕ) : ℕ :=
  m * n + 2 * m + 2 * n - 1

/-- 
Theorem stating that for any positive natural numbers m and n, 
the maximum number of regions formed by m+n lines (m through point A, n through point B) 
is given by the function max_regions.
-/
theorem max_regions_correct (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  max_regions m n = m * n + 2 * m + 2 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_stating_max_regions_correct_l910_91065


namespace NUMINAMATH_CALUDE_range_of_a_l910_91024

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) ↔ 
  a ≤ -2 ∨ a = 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l910_91024


namespace NUMINAMATH_CALUDE_unique_valid_n_l910_91038

def is_valid_n (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    n = 10 * a + b ∧
    100 * a + 10 * c + b = 6 * n

theorem unique_valid_n :
  ∃! n : ℕ, n ≥ 10 ∧ is_valid_n n ∧ n = 18 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_n_l910_91038


namespace NUMINAMATH_CALUDE_no_solution_exists_l910_91030

theorem no_solution_exists (k : ℕ) (hk : k > 1) : ¬ ∃ n : ℕ+, ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → n / i = 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l910_91030


namespace NUMINAMATH_CALUDE_periodic_odd_function_sum_l910_91005

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def MinimumPositivePeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  HasPeriod f p ∧ p > 0 ∧ ∀ q, 0 < q ∧ q < p → ¬HasPeriod f q

theorem periodic_odd_function_sum (f : ℝ → ℝ) :
  IsOdd f →
  MinimumPositivePeriod f 3 →
  (∀ x, f x = Real.log (1 - x)) →
  f 2010 + f 2011 = 1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_odd_function_sum_l910_91005


namespace NUMINAMATH_CALUDE_ninas_homework_is_40_l910_91026

/-- The amount of Nina's total homework given Ruby's homework and the ratios --/
def ninas_total_homework (rubys_math_homework : ℕ) (rubys_reading_homework : ℕ) 
  (math_ratio : ℕ) (reading_ratio : ℕ) : ℕ :=
  math_ratio * rubys_math_homework + reading_ratio * rubys_reading_homework

/-- Theorem stating that Nina's total homework is 40 given the problem conditions --/
theorem ninas_homework_is_40 :
  ninas_total_homework 6 2 4 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ninas_homework_is_40_l910_91026


namespace NUMINAMATH_CALUDE_minimal_additional_squares_l910_91003

/-- A point on the grid --/
structure Point where
  x : Nat
  y : Nat

/-- The grid configuration --/
structure Grid where
  size : Nat
  shaded : List Point

/-- Check if a point is within the grid --/
def inGrid (p : Point) (g : Grid) : Prop :=
  p.x < g.size ∧ p.y < g.size

/-- Check if a point is shaded --/
def isShaded (p : Point) (g : Grid) : Prop :=
  p ∈ g.shaded

/-- Reflect a point horizontally --/
def reflectHorizontal (p : Point) (g : Grid) : Point :=
  ⟨p.x, g.size - 1 - p.y⟩

/-- Reflect a point vertically --/
def reflectVertical (p : Point) (g : Grid) : Point :=
  ⟨g.size - 1 - p.x, p.y⟩

/-- Check if the grid has horizontal symmetry --/
def hasHorizontalSymmetry (g : Grid) : Prop :=
  ∀ p, inGrid p g → (isShaded p g ↔ isShaded (reflectHorizontal p g) g)

/-- Check if the grid has vertical symmetry --/
def hasVerticalSymmetry (g : Grid) : Prop :=
  ∀ p, inGrid p g → (isShaded p g ↔ isShaded (reflectVertical p g) g)

/-- The initial grid configuration --/
def initialGrid : Grid :=
  { size := 6
  , shaded := [⟨0,5⟩, ⟨2,3⟩, ⟨3,2⟩, ⟨5,0⟩] }

/-- The theorem to prove --/
theorem minimal_additional_squares :
  ∃ (additionalSquares : List Point),
    additionalSquares.length = 1 ∧
    let newGrid : Grid := { size := initialGrid.size, shaded := initialGrid.shaded ++ additionalSquares }
    hasHorizontalSymmetry newGrid ∧ hasVerticalSymmetry newGrid ∧
    ∀ (otherSquares : List Point),
      otherSquares.length < additionalSquares.length →
      let otherGrid : Grid := { size := initialGrid.size, shaded := initialGrid.shaded ++ otherSquares }
      ¬(hasHorizontalSymmetry otherGrid ∧ hasVerticalSymmetry otherGrid) :=
by sorry

end NUMINAMATH_CALUDE_minimal_additional_squares_l910_91003


namespace NUMINAMATH_CALUDE_meals_neither_kosher_nor_vegan_l910_91002

/-- Proves the number of meals that are neither kosher nor vegan -/
theorem meals_neither_kosher_nor_vegan 
  (total_clients : ℕ) 
  (vegan_meals : ℕ) 
  (kosher_meals : ℕ) 
  (both_vegan_and_kosher : ℕ) 
  (h1 : total_clients = 30)
  (h2 : vegan_meals = 7)
  (h3 : kosher_meals = 8)
  (h4 : both_vegan_and_kosher = 3) :
  total_clients - (vegan_meals + kosher_meals - both_vegan_and_kosher) = 18 :=
by
  sorry

#check meals_neither_kosher_nor_vegan

end NUMINAMATH_CALUDE_meals_neither_kosher_nor_vegan_l910_91002


namespace NUMINAMATH_CALUDE_largest_coefficient_term_l910_91006

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The general term in the binomial expansion -/
def binomialTerm (n k : ℕ) (a b : ℝ) : ℝ := 
  (binomial n k : ℝ) * (a ^ (n - k)) * (b ^ k)

/-- The coefficient of the k-th term in the expansion of (2+3x)^10 -/
def coefficientTerm (k : ℕ) : ℝ := 
  (binomial 10 k : ℝ) * (2 ^ (10 - k)) * (3 ^ k)

theorem largest_coefficient_term :
  ∃ (k : ℕ), k = 5 ∧ 
  ∀ (j : ℕ), j ≠ k → coefficientTerm k ≥ coefficientTerm j :=
sorry

end NUMINAMATH_CALUDE_largest_coefficient_term_l910_91006


namespace NUMINAMATH_CALUDE_planes_parallel_if_perp_lines_parallel_l910_91045

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perp_lines_parallel
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : perpendicular m β)
  (h3 : parallel l m) :
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perp_lines_parallel_l910_91045


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l910_91060

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 12 n = Nat.choose 12 (2*n - 3)) → (n = 3 ∨ n = 5) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l910_91060


namespace NUMINAMATH_CALUDE_triangle_equilateral_proof_l910_91021

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if certain conditions are met, the triangle is equilateral with A = π/3. -/
theorem triangle_equilateral_proof (a b c A B C : ℝ) : 
  0 < A ∧ A < π →  -- Angle A is between 0 and π
  0 < B ∧ B < π →  -- Angle B is between 0 and π
  0 < C ∧ C < π →  -- Angle C is between 0 and π
  A + B + C = π →  -- Sum of angles in a triangle
  2 * a * Real.cos B = 2 * c - b →  -- Given condition
  (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3 / 4 →  -- Area condition
  a = Real.sqrt 3 →  -- Given side length
  A = π/3 ∧ a = b ∧ b = c := by sorry

end NUMINAMATH_CALUDE_triangle_equilateral_proof_l910_91021


namespace NUMINAMATH_CALUDE_small_cuboids_needed_for_large_l910_91047

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℕ :=
  d.width * d.length * d.height

/-- The dimensions of the large cuboid -/
def largeCuboid : CuboidDimensions :=
  { width := 24, length := 15, height := 28 }

/-- The dimensions of the small cuboid -/
def smallCuboid : CuboidDimensions :=
  { width := 4, length := 5, height := 7 }

/-- Theorem stating that 72 small cuboids are needed to create the large cuboid -/
theorem small_cuboids_needed_for_large : 
  (cuboidVolume largeCuboid) / (cuboidVolume smallCuboid) = 72 := by
  sorry

end NUMINAMATH_CALUDE_small_cuboids_needed_for_large_l910_91047


namespace NUMINAMATH_CALUDE_inequality_proof_l910_91004

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = 2) :
  ((a + b) * (a^5 + b^5) ≥ 4) ∧ (a + b ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l910_91004


namespace NUMINAMATH_CALUDE_apartment_occupancy_theorem_l910_91018

/-- Represents an apartment complex with identical buildings -/
structure ApartmentComplex where
  num_buildings : ℕ
  studio_per_building : ℕ
  two_person_per_building : ℕ
  four_person_per_building : ℕ
  occupancy_rate : ℚ

/-- Calculates the number of people living in the apartment complex at the given occupancy rate -/
def occupancy (complex : ApartmentComplex) : ℕ :=
  let max_per_building := 
    complex.studio_per_building + 
    2 * complex.two_person_per_building + 
    4 * complex.four_person_per_building
  let total_max := complex.num_buildings * max_per_building
  ⌊(total_max : ℚ) * complex.occupancy_rate⌋.toNat

theorem apartment_occupancy_theorem (complex : ApartmentComplex) 
  (h1 : complex.num_buildings = 4)
  (h2 : complex.studio_per_building = 10)
  (h3 : complex.two_person_per_building = 20)
  (h4 : complex.four_person_per_building = 5)
  (h5 : complex.occupancy_rate = 3/4) :
  occupancy complex = 210 := by
  sorry

end NUMINAMATH_CALUDE_apartment_occupancy_theorem_l910_91018


namespace NUMINAMATH_CALUDE_three_tetrominoes_with_symmetry_l910_91084

-- Define the set of tetrominoes
inductive Tetromino
| I -- Line
| O -- Square
| T
| S
| Z

-- Define a function to check if a tetromino has reflectional symmetry
def has_reflectional_symmetry : Tetromino → Bool
| Tetromino.I => true
| Tetromino.O => true
| Tetromino.T => true
| Tetromino.S => false
| Tetromino.Z => false

-- Define the set of all tetrominoes
def all_tetrominoes : List Tetromino :=
  [Tetromino.I, Tetromino.O, Tetromino.T, Tetromino.S, Tetromino.Z]

-- Theorem: Exactly 3 tetrominoes have reflectional symmetry
theorem three_tetrominoes_with_symmetry :
  (all_tetrominoes.filter has_reflectional_symmetry).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_tetrominoes_with_symmetry_l910_91084


namespace NUMINAMATH_CALUDE_factorial_fraction_l910_91035

theorem factorial_fraction (N : ℕ) :
  (Nat.factorial (N + 2)) / (Nat.factorial (N + 3) - Nat.factorial (N + 2)) = 1 / (N + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_l910_91035


namespace NUMINAMATH_CALUDE_miss_evans_class_size_l910_91015

theorem miss_evans_class_size :
  let total_contribution : ℕ := 90
  let class_funds : ℕ := 14
  let student_contribution : ℕ := 4
  let remaining_contribution := total_contribution - class_funds
  let num_students := remaining_contribution / student_contribution
  num_students = 19 := by sorry

end NUMINAMATH_CALUDE_miss_evans_class_size_l910_91015


namespace NUMINAMATH_CALUDE_eight_prof_sequences_l910_91079

/-- The number of professors --/
def n : ℕ := 8

/-- The number of distinct sequences for scheduling n professors,
    where one specific professor must present before another specific professor --/
def num_sequences (n : ℕ) : ℕ := n.factorial / 2

/-- Theorem stating that the number of distinct sequences for scheduling 8 professors,
    where one specific professor must present before another specific professor,
    is equal to 8! / 2 --/
theorem eight_prof_sequences :
  num_sequences n = 20160 := by sorry

end NUMINAMATH_CALUDE_eight_prof_sequences_l910_91079


namespace NUMINAMATH_CALUDE_equal_chord_circle_exists_l910_91067

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The incenter of a triangle --/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The length of a chord formed by the intersection of a circle and a line segment --/
def chordLength (c : Circle) (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: For any triangle, there exists a circle that cuts chords of equal length from its sides --/
theorem equal_chord_circle_exists (t : Triangle) : 
  ∃ (c : Circle), 
    chordLength c t.A t.B = chordLength c t.B t.C ∧ 
    chordLength c t.B t.C = chordLength c t.C t.A := by
  sorry

end NUMINAMATH_CALUDE_equal_chord_circle_exists_l910_91067


namespace NUMINAMATH_CALUDE_opposite_of_abs_neg_half_l910_91091

theorem opposite_of_abs_neg_half : 
  -(|(-0.5 : ℝ)|) = -0.5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_abs_neg_half_l910_91091


namespace NUMINAMATH_CALUDE_max_area_APBQ_l910_91028

noncomputable section

-- Define the Cartesian coordinate system
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (-1, 0)

-- Define the distance ratio condition
def distance_ratio (P : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 + 1)^2 + P.2^2) / |P.1 + 2|

-- Define the trajectory C
def C : Set (ℝ × ℝ) :=
  {P | distance_ratio P = Real.sqrt 2 / 2}

-- Define the circle C₁
def C₁ : Set (ℝ × ℝ) :=
  {P | (P.1 - 4)^2 + P.2^2 = 32}

-- Define a chord AB of C passing through F
def chord_AB (m : ℝ) : Set (ℝ × ℝ) :=
  {P | P ∈ C ∧ P.1 = m * P.2 - 1}

-- Define the midpoint M of AB
def M (m : ℝ) : ℝ × ℝ :=
  (-2 / (m^2 + 2), m / (m^2 + 2))

-- Define the line OM
def line_OM (m : ℝ) : Set (ℝ × ℝ) :=
  {P | P.2 = (m / (m^2 + 2)) * P.1}

-- Define the intersection points P and Q
def P_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {P | P ∈ C₁ ∧ P ∈ line_OM m}

-- Define the area of quadrilateral APBQ
def area_APBQ (m : ℝ) : ℝ :=
  8 * Real.sqrt 2 * Real.sqrt ((m^2 + 8) * (m^2 + 1) / (m^2 + 4)^2)

-- Theorem statement
theorem max_area_APBQ :
  ∃ m : ℝ, ∀ n : ℝ, area_APBQ m ≥ area_APBQ n ∧ area_APBQ m = 14 * Real.sqrt 6 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_area_APBQ_l910_91028


namespace NUMINAMATH_CALUDE_sum_of_fractions_l910_91090

theorem sum_of_fractions : (1 : ℚ) / 9 + (1 : ℚ) / 11 = 20 / 99 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l910_91090


namespace NUMINAMATH_CALUDE_nth_root_inequality_l910_91075

theorem nth_root_inequality (m n : ℕ) (h1 : m > n) (h2 : n ≥ 2) :
  (m : ℝ) ^ (1 / n : ℝ) - (n : ℝ) ^ (1 / m : ℝ) > 1 / (m * n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_nth_root_inequality_l910_91075


namespace NUMINAMATH_CALUDE_circle_line_no_intersection_l910_91086

theorem circle_line_no_intersection (b : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 = 2 → y ≠ x + b) ↔ (b > 2 ∨ b < -2) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_no_intersection_l910_91086


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l910_91016

theorem billion_to_scientific_notation :
  let billion : ℝ := 10^9
  8.26 * billion = 8.26 * 10^9 := by
  sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l910_91016


namespace NUMINAMATH_CALUDE_specific_triangle_toothpicks_l910_91076

/-- Represents the configuration of a large equilateral triangle made of small triangles --/
structure TriangleConfig where
  rows : Nat
  base_triangles : Nat
  double_count_start : Nat

/-- Calculates the total number of toothpicks required for a given triangle configuration --/
def total_toothpicks (config : TriangleConfig) : Nat :=
  sorry

/-- Theorem stating that the specific configuration requires 1617 toothpicks --/
theorem specific_triangle_toothpicks :
  let config : TriangleConfig := {
    rows := 5,
    base_triangles := 100,
    double_count_start := 2
  }
  total_toothpicks config = 1617 := by
  sorry

end NUMINAMATH_CALUDE_specific_triangle_toothpicks_l910_91076


namespace NUMINAMATH_CALUDE_ribbon_length_difference_l910_91099

/-- Represents the dimensions of a box in centimeters -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the ribbon length for the first method -/
def ribbonLengthMethod1 (box : BoxDimensions) : ℝ :=
  2 * box.length + 2 * box.width + 4 * box.height + 24

/-- Calculates the ribbon length for the second method -/
def ribbonLengthMethod2 (box : BoxDimensions) : ℝ :=
  2 * box.length + 4 * box.width + 2 * box.height + 24

/-- Theorem stating that the difference in ribbon lengths equals one side of the box -/
theorem ribbon_length_difference (box : BoxDimensions) 
    (h1 : box.length = 22) 
    (h2 : box.width = 22) 
    (h3 : box.height = 11) : 
  ribbonLengthMethod2 box - ribbonLengthMethod1 box = box.length := by
  sorry

end NUMINAMATH_CALUDE_ribbon_length_difference_l910_91099


namespace NUMINAMATH_CALUDE_range_of_a_l910_91044

theorem range_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1/x + 2/y = 1) (h_ineq : ∀ x y, x > 0 → y > 0 → 1/x + 2/y = 1 → x + 2*y > a^2 + 8*a) : 
  -4 - 2*Real.sqrt 6 < a ∧ a < -4 + 2*Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l910_91044


namespace NUMINAMATH_CALUDE_smallest_valid_n_l910_91070

def is_valid_pairing (n : ℕ) : Prop :=
  ∃ (f : ℕ → ℕ), 
    (∀ i ∈ Finset.range 1008, f i ≠ f (2017 - i) ∧ f i ∈ Finset.range 2016 ∧ f (2017 - i) ∈ Finset.range 2016) ∧
    (∀ i ∈ Finset.range 1008, (i + 1) * (2017 - i) ≤ n)

theorem smallest_valid_n : (∀ m < 1017072, ¬ is_valid_pairing m) ∧ is_valid_pairing 1017072 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l910_91070
