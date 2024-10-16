import Mathlib

namespace NUMINAMATH_CALUDE_flower_bed_area_l3748_374893

theorem flower_bed_area (total_posts : ℕ) (post_spacing : ℝ) 
  (h1 : total_posts = 24)
  (h2 : post_spacing = 5)
  (h3 : ∃ (short_side long_side : ℕ), 
    short_side + 1 + long_side + 1 = total_posts ∧ 
    long_side + 1 = 3 * (short_side + 1)) :
  (short_side * post_spacing) * (long_side * post_spacing) = 600 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_area_l3748_374893


namespace NUMINAMATH_CALUDE_black_squares_21st_row_l3748_374868

/-- Represents the number of squares in a row of the stair-step figure -/
def squares_in_row (n : ℕ) : ℕ := 2 * n

/-- Represents the number of black squares in a row of the stair-step figure -/
def black_squares_in_row (n : ℕ) : ℕ := 2 * (squares_in_row n / 4)

theorem black_squares_21st_row :
  black_squares_in_row 21 = 20 := by
  sorry

end NUMINAMATH_CALUDE_black_squares_21st_row_l3748_374868


namespace NUMINAMATH_CALUDE_max_second_term_arithmetic_sequence_l3748_374818

theorem max_second_term_arithmetic_sequence (a d : ℕ) (h1 : 0 < a) (h2 : 0 < d) :
  (a + (a + d) + (a + 2*d) + (a + 3*d) = 58) →
  ∀ b e : ℕ, (0 < b) → (0 < e) →
  (b + (b + e) + (b + 2*e) + (b + 3*e) = 58) →
  (a + d ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_max_second_term_arithmetic_sequence_l3748_374818


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l3748_374802

theorem arithmetic_mean_of_first_four_primes_reciprocals :
  let first_four_primes := [2, 3, 5, 7]
  let reciprocals := first_four_primes.map (λ x => 1 / x)
  (reciprocals.sum / 4 : ℚ) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_first_four_primes_reciprocals_l3748_374802


namespace NUMINAMATH_CALUDE_road_sign_difference_l3748_374813

/-- Represents the number of road signs at each intersection -/
structure RoadSigns where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- The conditions of the road sign problem -/
def roadSignProblem (rs : RoadSigns) : Prop :=
  rs.first = 40 ∧
  rs.second = rs.first + rs.first / 4 ∧
  rs.third = 2 * rs.second ∧
  rs.fourth < rs.third ∧
  rs.first + rs.second + rs.third + rs.fourth = 270

theorem road_sign_difference (rs : RoadSigns) 
  (h : roadSignProblem rs) : rs.third - rs.fourth = 20 := by
  sorry

end NUMINAMATH_CALUDE_road_sign_difference_l3748_374813


namespace NUMINAMATH_CALUDE_balls_after_1500_steps_l3748_374888

/-- Represents the state of boxes with balls -/
def BoxState := List Nat

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : Nat) : List Nat :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List Nat) : Nat :=
  sorry

/-- Simulates the ball placement process for a given number of steps -/
def simulateBallPlacement (steps : Nat) : BoxState :=
  sorry

/-- Counts the total number of balls in a BoxState -/
def countBalls (state : BoxState) : Nat :=
  sorry

/-- Theorem stating that the number of balls after 1500 steps
    is equal to the sum of digits of 1500 in base-4 -/
theorem balls_after_1500_steps :
  countBalls (simulateBallPlacement 1500) = sumDigits (toBase4 1500) :=
sorry

end NUMINAMATH_CALUDE_balls_after_1500_steps_l3748_374888


namespace NUMINAMATH_CALUDE_outside_trash_count_l3748_374830

def total_trash : ℕ := 1576
def classroom_trash : ℕ := 344

theorem outside_trash_count : total_trash - classroom_trash = 1232 := by
  sorry

end NUMINAMATH_CALUDE_outside_trash_count_l3748_374830


namespace NUMINAMATH_CALUDE_polynomial_remainder_l3748_374856

def p (x : ℝ) : ℝ := 4*x^5 - 3*x^4 + 5*x^3 - 7*x^2 + 3*x - 10

theorem polynomial_remainder : p 2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l3748_374856


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l3748_374896

theorem sqrt_3_irrational :
  ∀ (a b c : ℚ), (a = 1/2 ∧ b = 1/5 ∧ c = -5) →
  ¬ ∃ (p q : ℤ), q ≠ 0 ∧ Real.sqrt 3 = p / q := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l3748_374896


namespace NUMINAMATH_CALUDE_min_value_expression_l3748_374851

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + 4*b^2 + 1/(a*b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3748_374851


namespace NUMINAMATH_CALUDE_fraction_division_result_l3748_374843

theorem fraction_division_result (a : ℝ) 
  (h1 : a^2 + 4*a + 4 ≠ 0) 
  (h2 : a^2 + 5*a + 6 ≠ 0) : 
  (a^2 - 4) / (a^2 + 4*a + 4) / ((a^2 + a - 6) / (a^2 + 5*a + 6)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_result_l3748_374843


namespace NUMINAMATH_CALUDE_unique_solution_fourth_root_equation_l3748_374820

theorem unique_solution_fourth_root_equation :
  ∃! x : ℝ, (((4 - x) ^ (1/4) : ℝ) + ((x - 2) ^ (1/2) : ℝ) = 2) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_fourth_root_equation_l3748_374820


namespace NUMINAMATH_CALUDE_original_blueberry_count_l3748_374824

/-- Represents the number of blueberry jelly beans Camilla originally had -/
def blueberry : ℕ := sorry

/-- Represents the number of cherry jelly beans Camilla originally had -/
def cherry : ℕ := sorry

/-- Theorem stating the original number of blueberry jelly beans -/
theorem original_blueberry_count : blueberry = 30 := by
  have h1 : blueberry = 3 * cherry := sorry
  have h2 : blueberry - 20 = 2 * (cherry - 5) := sorry
  sorry


end NUMINAMATH_CALUDE_original_blueberry_count_l3748_374824


namespace NUMINAMATH_CALUDE_factor_63x_minus_21_l3748_374861

theorem factor_63x_minus_21 : ∀ x : ℝ, 63 * x - 21 = 21 * (3 * x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_63x_minus_21_l3748_374861


namespace NUMINAMATH_CALUDE_triangle_and_circle_symmetry_l3748_374880

-- Define the point A
def A : ℝ × ℝ := (4, -3)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the vector OA
def OA : ℝ × ℝ := A

-- Define the vector AB
def AB : ℝ × ℝ := (6, 8)

-- Define point B
def B : ℝ × ℝ := (A.1 + AB.1, A.2 + AB.2)

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 10

-- Theorem statement
theorem triangle_and_circle_symmetry :
  -- A is the right-angle vertex of triangle OAB
  (OA.1 * AB.1 + OA.2 * AB.2 = 0) →
  -- |AB| = 2|OA|
  (AB.1^2 + AB.2^2 = 4 * (OA.1^2 + OA.2^2)) →
  -- The ordinate of point B is greater than 0
  (B.2 > 0) →
  -- AB has coordinates (6, 8)
  (AB = (6, 8)) ∧
  -- The equation of the symmetric circle is correct
  (∀ x y, symmetric_circle x y ↔
    ∃ x' y', original_circle x' y' ∧
      -- x and y are symmetric to x' and y' with respect to line OB
      ((x + x') / 2 = B.1 * ((y + y') / 2) / B.2)) :=
sorry

end NUMINAMATH_CALUDE_triangle_and_circle_symmetry_l3748_374880


namespace NUMINAMATH_CALUDE_problem_solution_l3748_374886

/-- The surface area of an open box formed by removing square corners from a rectangular sheet. -/
def boxSurfaceArea (length width corner_size : ℕ) : ℕ :=
  length * width - 4 * (corner_size * corner_size)

/-- Theorem stating that the surface area of the box described in the problem is 500 square units. -/
theorem problem_solution :
  boxSurfaceArea 30 20 5 = 500 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3748_374886


namespace NUMINAMATH_CALUDE_sum_a_d_equals_five_l3748_374853

theorem sum_a_d_equals_five 
  (a b c d : ℤ) 
  (eq1 : a + b = 11) 
  (eq2 : b + c = 9) 
  (eq3 : c + d = 3) : 
  a + d = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_equals_five_l3748_374853


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3748_374878

theorem imaginary_part_of_complex_fraction : Complex.im ((1 + Complex.I) / (1 - Complex.I)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3748_374878


namespace NUMINAMATH_CALUDE_total_students_on_ride_l3748_374810

theorem total_students_on_ride (seats_per_ride : ℕ) (empty_seats : ℕ) (num_rides : ℕ) : 
  seats_per_ride = 15 → empty_seats = 3 → num_rides = 18 →
  (seats_per_ride - empty_seats) * num_rides = 216 := by
  sorry

end NUMINAMATH_CALUDE_total_students_on_ride_l3748_374810


namespace NUMINAMATH_CALUDE_total_wheels_l3748_374858

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of adults riding bicycles -/
def adults_on_bicycles : ℕ := 6

/-- The number of children riding tricycles -/
def children_on_tricycles : ℕ := 15

/-- The total number of wheels Dimitri saw at the park -/
theorem total_wheels : 
  bicycle_wheels * adults_on_bicycles + tricycle_wheels * children_on_tricycles = 57 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_l3748_374858


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3748_374867

theorem inequality_solution_set (x : ℝ) :
  (-2 < (x^2 - 16*x + 15) / (x^2 - 4*x + 5) ∧ (x^2 - 16*x + 15) / (x^2 - 4*x + 5) < 2) ↔
  (x < -13.041 ∨ x > -0.959) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3748_374867


namespace NUMINAMATH_CALUDE_first_operation_result_l3748_374827

theorem first_operation_result (x : ℝ) : (x - 24) / 10 = 3 → (x - 5) / 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_first_operation_result_l3748_374827


namespace NUMINAMATH_CALUDE_sphere_radius_equals_cone_lateral_area_l3748_374841

theorem sphere_radius_equals_cone_lateral_area 
  (cone_height : ℝ) 
  (cone_base_radius : ℝ) 
  (sphere_radius : ℝ) :
  cone_height = 3 →
  cone_base_radius = 4 →
  (4 * Real.pi * sphere_radius^2) = (Real.pi * cone_base_radius * (cone_height^2 + cone_base_radius^2).sqrt) →
  sphere_radius = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_equals_cone_lateral_area_l3748_374841


namespace NUMINAMATH_CALUDE_quadrilateral_side_length_l3748_374897

/-- Represents a quadrilateral with sides a, b, c, d --/
structure Quadrilateral :=
  (a b c d : ℝ)

/-- Represents the properties of the specific quadrilateral in the problem --/
def ProblemQuadrilateral (q : Quadrilateral) (x y : ℕ) : Prop :=
  q.a = 20 ∧ 
  q.a = x^2 + y^2 ∧ 
  q.b = x ∧ 
  q.c = y ∧ 
  x > 0 ∧ 
  y > 0 ∧ 
  q.d ≥ q.a ∧ 
  q.d ≥ q.b ∧ 
  q.d ≥ q.c

theorem quadrilateral_side_length 
  (q : Quadrilateral) 
  (x y : ℕ) 
  (h : ProblemQuadrilateral q x y) : 
  q.d = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_side_length_l3748_374897


namespace NUMINAMATH_CALUDE_digit_sum_l3748_374890

theorem digit_sum (a b : ℕ) : 
  a < 10 → b < 10 → (32 * a + 300) * (10 * b + 4) = 1486 → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_l3748_374890


namespace NUMINAMATH_CALUDE_mary_regular_hours_l3748_374823

/-- Represents Mary's work schedule and pay structure --/
structure MaryWork where
  maxHours : Nat
  regularRate : ℝ
  overtimeRateIncrease : ℝ
  maxEarnings : ℝ

/-- Calculates Mary's earnings based on regular hours worked --/
def calculateEarnings (work : MaryWork) (regularHours : ℝ) : ℝ :=
  let overtimeRate := work.regularRate * (1 + work.overtimeRateIncrease)
  let overtimeHours := work.maxHours - regularHours
  regularHours * work.regularRate + overtimeHours * overtimeRate

/-- Theorem stating that Mary works 20 hours at her regular rate to maximize earnings --/
theorem mary_regular_hours (work : MaryWork)
    (h1 : work.maxHours = 50)
    (h2 : work.regularRate = 8)
    (h3 : work.overtimeRateIncrease = 0.25)
    (h4 : work.maxEarnings = 460) :
    ∃ (regularHours : ℝ), regularHours = 20 ∧
    calculateEarnings work regularHours = work.maxEarnings :=
  sorry


end NUMINAMATH_CALUDE_mary_regular_hours_l3748_374823


namespace NUMINAMATH_CALUDE_project_selection_count_l3748_374835

def num_key_projects : ℕ := 4
def num_general_projects : ℕ := 6
def projects_to_select : ℕ := 3

def select_projects (n k : ℕ) : ℕ := Nat.choose n k

theorem project_selection_count : 
  (select_projects (num_general_projects - 1) (projects_to_select - 1) * 
   select_projects (num_key_projects - 1) (projects_to_select - 1)) +
  (select_projects (num_key_projects - 1) 1 * 
   select_projects (num_general_projects - 1) 1) = 45 := by sorry

end NUMINAMATH_CALUDE_project_selection_count_l3748_374835


namespace NUMINAMATH_CALUDE_integer_solution_equation_l3748_374817

theorem integer_solution_equation :
  ∀ x y : ℤ, (3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3) ↔ 
  ((x = -6 ∧ y = -6) ∨ (x = 0 ∧ y = 0) ∨ (x = 6 ∧ y = 6)) := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_equation_l3748_374817


namespace NUMINAMATH_CALUDE_unique_tangent_circle_existence_l3748_374806

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the given elements
variable (M : Point) -- Given point
variable (O : Point) -- Center of the given circle
variable (r : ℝ) -- Radius of the given circle
variable (N : Point) -- Point on the given circle

-- Define the condition that N is on the given circle
def is_on_circle (P : Point) (C : Circle) : Prop :=
  (P.x - C.center.x)^2 + (P.y - C.center.y)^2 = C.radius^2

-- Define tangency between two circles
def are_tangent (C1 C2 : Circle) : Prop :=
  (C1.center.x - C2.center.x)^2 + (C1.center.y - C2.center.y)^2 = (C1.radius + C2.radius)^2

-- State the theorem
theorem unique_tangent_circle_existence 
  (h_N_on_circle : is_on_circle N { center := O, radius := r }) :
  ∃! C : Circle, (is_on_circle M C) ∧ 
                 (are_tangent C { center := O, radius := r }) ∧ 
                 (is_on_circle N C) := by
  sorry

end NUMINAMATH_CALUDE_unique_tangent_circle_existence_l3748_374806


namespace NUMINAMATH_CALUDE_shortest_distance_between_inscribed_circles_shortest_distance_proof_l3748_374815

/-- The shortest distance between two circles inscribed in two of nine identical squares 
    (each with side length 1) that form a larger square -/
theorem shortest_distance_between_inscribed_circles : ℝ :=
  let large_square_side : ℝ := 3
  let small_square_side : ℝ := 1
  let num_small_squares : ℕ := 9
  let circle_radius : ℝ := small_square_side / 2
  2 * Real.sqrt 2 - 1

/-- Proof of the shortest distance between the inscribed circles -/
theorem shortest_distance_proof :
  shortest_distance_between_inscribed_circles = 2 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_between_inscribed_circles_shortest_distance_proof_l3748_374815


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l3748_374879

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a population with subgroups -/
structure Population where
  total : Nat
  subgroups : List Nat
  sample_size : Nat

/-- Represents a simple population without subgroups -/
structure SimplePopulation where
  total : Nat
  sample_size : Nat

def student_population : Population :=
  { total := 1200
  , subgroups := [400, 600, 200]
  , sample_size := 120 }

def parent_population : SimplePopulation :=
  { total := 10
  , sample_size := 3 }

/-- Determines the best sampling method for a given population -/
def best_sampling_method (pop : Population) : SamplingMethod :=
  sorry

/-- Determines the best sampling method for a simple population -/
def best_simple_sampling_method (pop : SimplePopulation) : SamplingMethod :=
  sorry

theorem correct_sampling_methods :
  (best_sampling_method student_population = SamplingMethod.Stratified) ∧
  (best_simple_sampling_method parent_population = SamplingMethod.SimpleRandom) :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l3748_374879


namespace NUMINAMATH_CALUDE_gift_expenses_calculation_l3748_374857

/-- Calculates the total amount spent on gift wrapping, taxes, and other expenses given the following conditions:
  * Jeremy bought presents for 5 people
  * Total spent: $930
  * Gift costs: $400 (mother), $280 (father), $100 (sister), $60 (brother), $50 (best friend)
  * Gift wrapping fee: 7% of each gift's price
  * Tax rate: 9%
  * Other miscellaneous expenses: $40
-/
theorem gift_expenses_calculation (total_spent : ℝ) (gift_mother gift_father gift_sister gift_brother gift_friend : ℝ)
  (wrapping_rate tax_rate : ℝ) (misc_expenses : ℝ) :
  total_spent = 930 ∧
  gift_mother = 400 ∧ gift_father = 280 ∧ gift_sister = 100 ∧ gift_brother = 60 ∧ gift_friend = 50 ∧
  wrapping_rate = 0.07 ∧ tax_rate = 0.09 ∧ misc_expenses = 40 →
  (gift_mother + gift_father + gift_sister + gift_brother + gift_friend) * wrapping_rate +
  (gift_mother + gift_father + gift_sister + gift_brother + gift_friend) * tax_rate +
  misc_expenses = 182.40 := by
  sorry

end NUMINAMATH_CALUDE_gift_expenses_calculation_l3748_374857


namespace NUMINAMATH_CALUDE_vertical_line_properties_l3748_374883

/-- A line passing through two points with the same x-coordinate but different y-coordinates has an undefined slope and its x-intercept is equal to the common x-coordinate. -/
theorem vertical_line_properties (x y₁ y₂ : ℝ) (h : y₁ ≠ y₂) :
  let C : ℝ × ℝ := (x, y₁)
  let D : ℝ × ℝ := (x, y₂)
  let line := {P : ℝ × ℝ | ∃ t : ℝ, P = (1 - t) • C + t • D}
  (∀ P Q : ℝ × ℝ, P ∈ line → Q ∈ line → P.1 ≠ Q.1 → (Q.2 - P.2) / (Q.1 - P.1) = (0 : ℝ)/0) ∧
  (∃ y : ℝ, (x, y) ∈ line) :=
by sorry

end NUMINAMATH_CALUDE_vertical_line_properties_l3748_374883


namespace NUMINAMATH_CALUDE_two_different_color_chips_probability_l3748_374840

/-- Represents the colors of chips in the bag -/
inductive ChipColor
  | Blue
  | Red
  | Yellow

/-- Represents the state of the bag of chips -/
structure ChipBag where
  blue : Nat
  red : Nat
  yellow : Nat

/-- Calculates the total number of chips in the bag -/
def ChipBag.total (bag : ChipBag) : Nat :=
  bag.blue + bag.red + bag.yellow

/-- Calculates the probability of drawing a specific color -/
def drawProbability (bag : ChipBag) (color : ChipColor) : Rat :=
  match color with
  | ChipColor.Blue => bag.blue / bag.total
  | ChipColor.Red => bag.red / bag.total
  | ChipColor.Yellow => bag.yellow / bag.total

/-- Calculates the probability of drawing two different colored chips -/
def differentColorProbability (bag : ChipBag) : Rat :=
  let blueFirst := drawProbability bag ChipColor.Blue * (1 - drawProbability bag ChipColor.Blue / 2)
  let redFirst := drawProbability bag ChipColor.Red * (1 - drawProbability bag ChipColor.Red)
  let yellowFirst := drawProbability bag ChipColor.Yellow * (1 - drawProbability bag ChipColor.Yellow)
  blueFirst + redFirst + yellowFirst

theorem two_different_color_chips_probability :
  let initialBag : ChipBag := { blue := 7, red := 5, yellow := 4 }
  differentColorProbability initialBag = 381 / 512 := by
  sorry


end NUMINAMATH_CALUDE_two_different_color_chips_probability_l3748_374840


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3748_374874

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  n ≤ 999 ∧ n ≥ 100 ∧ 17 ∣ n → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3748_374874


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l3748_374891

/-- A sequence is geometric if the ratio of consecutive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties
  (a : ℕ → ℝ) (h : IsGeometric a) :
  ∃ q : ℝ,
    (IsGeometric (fun n ↦ (a n)^3)) ∧
    (∀ p : ℝ, p ≠ 0 → IsGeometric (fun n ↦ p * a n)) ∧
    (IsGeometric (fun n ↦ a n * a (n + 1))) ∧
    (IsGeometric (fun n ↦ a n + a (n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l3748_374891


namespace NUMINAMATH_CALUDE_sheets_exceed_500_at_step_31_l3748_374889

def sheets_after_steps (initial_sheets : ℕ) (steps : ℕ) : ℕ :=
  initial_sheets + steps * (steps + 1) / 2

theorem sheets_exceed_500_at_step_31 :
  sheets_after_steps 10 31 > 500 ∧ sheets_after_steps 10 30 ≤ 500 := by
  sorry

end NUMINAMATH_CALUDE_sheets_exceed_500_at_step_31_l3748_374889


namespace NUMINAMATH_CALUDE_james_total_toys_l3748_374814

/-- The number of toy cars James buys -/
def toy_cars : ℕ := 20

/-- The number of toy soldiers James buys -/
def toy_soldiers : ℕ := 2 * toy_cars

/-- The total number of toys James buys -/
def total_toys : ℕ := toy_cars + toy_soldiers

theorem james_total_toys : total_toys = 60 := by sorry

end NUMINAMATH_CALUDE_james_total_toys_l3748_374814


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3748_374885

theorem circle_area_ratio (diameter_R diameter_S area_R area_S : ℝ) :
  diameter_R = 0.6 * diameter_S →
  area_R = π * (diameter_R / 2)^2 →
  area_S = π * (diameter_S / 2)^2 →
  area_R / area_S = 0.36 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3748_374885


namespace NUMINAMATH_CALUDE_apple_distribution_l3748_374809

/-- The number of ways to distribute n apples among k people, with each person receiving at least m apples -/
def distribution_ways (n k m : ℕ) : ℕ :=
  Nat.choose (n - k * m + k - 1) (k - 1)

/-- The problem statement -/
theorem apple_distribution :
  distribution_ways 30 3 3 = 253 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l3748_374809


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l3748_374875

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  mon_wed_fri_hours : ℕ
  tue_thu_hours : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week --/
def total_weekly_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.mon_wed_fri_hours + 2 * schedule.tue_thu_hours

/-- Calculates the hourly wage given a work schedule --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_weekly_hours schedule)

/-- Sheila's actual work schedule --/
def sheila_schedule : WorkSchedule :=
  { mon_wed_fri_hours := 8
  , tue_thu_hours := 6
  , weekly_earnings := 504 }

/-- Theorem stating that Sheila's hourly wage is $14 --/
theorem sheila_hourly_wage :
  hourly_wage sheila_schedule = 14 := by
  sorry


end NUMINAMATH_CALUDE_sheila_hourly_wage_l3748_374875


namespace NUMINAMATH_CALUDE_train_length_calculation_l3748_374845

/-- Given a train traveling at a certain speed that crosses a bridge of known length in a specific time, calculate the length of the train. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_speed = 45 * 1000 / 3600 →
  bridge_length = 215 →
  crossing_time = 30 →
  (train_speed * crossing_time) - bridge_length = 160 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l3748_374845


namespace NUMINAMATH_CALUDE_x_times_one_minus_f_equals_one_l3748_374836

/-- Given x = (3 + √8)^1000, n = ⌊x⌋, and f = x - n, prove that x(1 - f) = 1 -/
theorem x_times_one_minus_f_equals_one :
  let x : ℝ := (3 + Real.sqrt 8) ^ 1000
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by sorry

end NUMINAMATH_CALUDE_x_times_one_minus_f_equals_one_l3748_374836


namespace NUMINAMATH_CALUDE_political_science_majors_l3748_374838

/-- Represents the number of applicants who majored in political science -/
def P : ℕ := sorry

theorem political_science_majors :
  let total_applicants : ℕ := 40
  let high_gpa : ℕ := 20
  let not_ps_low_gpa : ℕ := 10
  let ps_high_gpa : ℕ := 5
  P = 15 := by sorry

end NUMINAMATH_CALUDE_political_science_majors_l3748_374838


namespace NUMINAMATH_CALUDE_probability_sum_less_than_product_l3748_374884

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def valid_number (n : ℕ) : Prop := is_even n ∧ 0 < n ∧ n ≤ 10

def valid_pair (a b : ℕ) : Prop := valid_number a ∧ valid_number b ∧ a + b < a * b

def total_pairs : ℕ := 25

def valid_pairs : ℕ := 16

theorem probability_sum_less_than_product :
  (valid_pairs : ℚ) / total_pairs = 16 / 25 := by sorry

end NUMINAMATH_CALUDE_probability_sum_less_than_product_l3748_374884


namespace NUMINAMATH_CALUDE_trapezoid_bases_l3748_374826

/-- Given a trapezoid with midline 6 and difference between bases 4, prove the bases are 4 and 8 -/
theorem trapezoid_bases (a b : ℝ) : 
  (a + b) / 2 = 6 → -- midline is 6
  a - b = 4 →       -- difference between bases is 4
  (a = 8 ∧ b = 4) := by
sorry

end NUMINAMATH_CALUDE_trapezoid_bases_l3748_374826


namespace NUMINAMATH_CALUDE_factorization_correctness_l3748_374822

theorem factorization_correctness (x y : ℝ) : 
  (∃! n : ℕ, n = (if x^3 + 2*x*y + x = x*(x^2 + 2*y) then 1 else 0) + 
             (if x^2 + 4*x + 4 = (x + 2)^2 then 1 else 0) + 
             (if -x^2 + y^2 = (x + y)*(x - y) then 1 else 0) ∧ 
             n = 1) := by sorry

end NUMINAMATH_CALUDE_factorization_correctness_l3748_374822


namespace NUMINAMATH_CALUDE_quadratic_solution_inequality_solution_l3748_374876

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 2 * x^2 + x - 2 = 0

-- Define the inequality system
def inequality_system (x : ℝ) : Prop := x + 3 > -2*x ∧ 2*x - 5 < 1

-- Theorem for the quadratic equation solution
theorem quadratic_solution :
  ∃ x1 x2 : ℝ, x1 = (-1 + Real.sqrt 17) / 4 ∧
              x2 = (-1 - Real.sqrt 17) / 4 ∧
              quadratic_equation x1 ∧
              quadratic_equation x2 :=
sorry

-- Theorem for the inequality system solution
theorem inequality_solution :
  ∀ x : ℝ, inequality_system x ↔ -1 < x ∧ x < 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_inequality_solution_l3748_374876


namespace NUMINAMATH_CALUDE_systematic_sampling_example_l3748_374839

/-- Systematic sampling function -/
def systematicSample (totalItems : ℕ) (numGroups : ℕ) (startGroup : ℕ) (startNum : ℕ) (targetGroup : ℕ) : ℕ :=
  startNum + (targetGroup - startGroup) * (totalItems / numGroups)

/-- Theorem: In systematic sampling of 200 items into 40 groups, 
    if the 5th group draws 24, then the 9th group draws 44 -/
theorem systematic_sampling_example :
  systematicSample 200 40 5 24 9 = 44 := by
  sorry

#eval systematicSample 200 40 5 24 9

end NUMINAMATH_CALUDE_systematic_sampling_example_l3748_374839


namespace NUMINAMATH_CALUDE_correct_multiple_choice_count_l3748_374869

/-- Represents the citizenship test with multiple-choice and fill-in-the-blank questions. -/
structure CitizenshipTest where
  totalQuestions : ℕ
  multipleChoiceTime : ℕ
  fillInBlankTime : ℕ
  totalStudyTime : ℕ

/-- Calculates the number of multiple-choice questions on the test. -/
def multipleChoiceCount (test : CitizenshipTest) : ℕ :=
  30

/-- Theorem stating that for the given test parameters, 
    the number of multiple-choice questions is 30. -/
theorem correct_multiple_choice_count 
  (test : CitizenshipTest)
  (h1 : test.totalQuestions = 60)
  (h2 : test.multipleChoiceTime = 15)
  (h3 : test.fillInBlankTime = 25)
  (h4 : test.totalStudyTime = 1200) :
  multipleChoiceCount test = 30 := by
  sorry

#eval multipleChoiceCount {
  totalQuestions := 60,
  multipleChoiceTime := 15,
  fillInBlankTime := 25,
  totalStudyTime := 1200
}

end NUMINAMATH_CALUDE_correct_multiple_choice_count_l3748_374869


namespace NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l3748_374832

-- Define the first equation
def equation1 (x : ℝ) : Prop := 2 / x = 3 / (x + 2)

-- Define the second equation
def equation2 (x : ℝ) : Prop := 5 / (x - 2) + 1 = (x - 7) / (2 - x)

-- Theorem for the first equation
theorem equation1_solution :
  ∃! x : ℝ, equation1 x ∧ x ≠ 0 ∧ x + 2 ≠ 0 := by sorry

-- Theorem for the second equation
theorem equation2_no_solution :
  ¬∃ x : ℝ, equation2 x ∧ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_equation1_solution_equation2_no_solution_l3748_374832


namespace NUMINAMATH_CALUDE_divide_powers_l3748_374872

theorem divide_powers (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  2 * x^2 * y^3 / (x * y^2) = 2 * x * y := by
  sorry

end NUMINAMATH_CALUDE_divide_powers_l3748_374872


namespace NUMINAMATH_CALUDE_find_d_l3748_374828

theorem find_d (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + 4 = d + Real.sqrt (a + b + c - d + 3)) : 
  d = 75/16 := by
sorry

end NUMINAMATH_CALUDE_find_d_l3748_374828


namespace NUMINAMATH_CALUDE_circle_M_properties_l3748_374871

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 2*y - 2 = 0

-- Define the center of a circle
def is_center (cx cy : ℝ) (circle : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, circle x y ↔ (x - cx)^2 + (y - cy)^2 = (cx^2 + cy^2 + 2*cx + 2*cy - 2)

-- Define a line passing through two points
def line_through (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

-- Define the chord length
def chord_length (circle : ℝ → ℝ → Prop) (line : ℝ → ℝ → Prop) : ℝ :=
  sorry -- Definition of chord length

theorem circle_M_properties :
  -- The center of circle M is at (-1, -1)
  is_center (-1) (-1) circle_M ∧
  -- The line x + y = 0 passes through (0, 0) and intersects circle M with the shortest chord
  (∀ a b, line_through 0 0 a b ≠ line_through 0 0 1 (-1) →
    chord_length circle_M (line_through 0 0 a b) ≥
    chord_length circle_M (line_through 0 0 1 (-1))) :=
by sorry

end NUMINAMATH_CALUDE_circle_M_properties_l3748_374871


namespace NUMINAMATH_CALUDE_pt_length_in_quadrilateral_l3748_374846

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Calculates the length between two points -/
def distance (A B : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (A B C : Point) : ℝ := sorry

/-- Theorem: In a convex quadrilateral PQRS, given specific side lengths and conditions, 
    the length of PT can be determined -/
theorem pt_length_in_quadrilateral 
  (PQRS : Quadrilateral)
  (T : Point)
  (convex : sorry) -- Assumption that PQRS is convex
  (pq_length : distance PQRS.P PQRS.Q = 8)
  (rs_length : distance PQRS.R PQRS.S = 14)
  (pr_length : distance PQRS.P PQRS.R = 18)
  (qs_length : distance PQRS.Q PQRS.S = 12)
  (T_on_PR : sorry) -- Assumption that T is on PR
  (T_on_QS : sorry) -- Assumption that T is on QS
  (equal_areas : triangleArea PQRS.P T PQRS.R = triangleArea PQRS.Q T PQRS.S) :
  distance PQRS.P T = 72 / 11 := by sorry

end NUMINAMATH_CALUDE_pt_length_in_quadrilateral_l3748_374846


namespace NUMINAMATH_CALUDE_maplewood_population_estimate_l3748_374811

theorem maplewood_population_estimate :
  ∀ (avg_population : ℝ),
  (25 : ℝ) > 0 →
  6200 ≤ avg_population →
  avg_population ≤ 6800 →
  ∃ (total_population : ℝ),
  total_population = 25 * avg_population ∧
  total_population = 162500 :=
by sorry

end NUMINAMATH_CALUDE_maplewood_population_estimate_l3748_374811


namespace NUMINAMATH_CALUDE_continued_fraction_value_l3748_374866

theorem continued_fraction_value : ∃ x : ℝ, x = 3 + 5 / (2 + 5 / x) ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_continued_fraction_value_l3748_374866


namespace NUMINAMATH_CALUDE_nina_tomato_harvest_l3748_374855

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ :=
  d.length * d.width

/-- Represents the planting and yield information for tomatoes -/
structure TomatoInfo where
  plantsPerSquareFoot : ℝ
  tomatoesPerPlant : ℝ

/-- Calculates the total number of tomatoes expected from a garden -/
def expectedTomatoes (d : GardenDimensions) (t : TomatoInfo) : ℝ :=
  gardenArea d * t.plantsPerSquareFoot * t.tomatoesPerPlant

/-- Theorem stating the expected tomato harvest for Nina's garden -/
theorem nina_tomato_harvest :
  let garden := GardenDimensions.mk 10 20
  let tomato := TomatoInfo.mk 5 10
  expectedTomatoes garden tomato = 10000 := by
  sorry


end NUMINAMATH_CALUDE_nina_tomato_harvest_l3748_374855


namespace NUMINAMATH_CALUDE_annual_cost_difference_is_5525_l3748_374870

def annual_cost_difference : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ := 
  fun clarinet_rate clarinet_hours piano_rate piano_hours violin_rate violin_hours 
      singing_rate singing_hours weeks_per_year =>
    let weeks_with_lessons := weeks_per_year - 2
    let clarinet_cost := clarinet_rate * clarinet_hours * weeks_with_lessons
    let piano_cost := (piano_rate * piano_hours * weeks_with_lessons * 9) / 10
    let violin_cost := (violin_rate * violin_hours * weeks_with_lessons * 85) / 100
    let singing_cost := singing_rate * singing_hours * weeks_with_lessons
    piano_cost + violin_cost + singing_cost - clarinet_cost

theorem annual_cost_difference_is_5525 :
  annual_cost_difference 40 3 28 5 35 2 45 1 52 = 5525 := by
  sorry

end NUMINAMATH_CALUDE_annual_cost_difference_is_5525_l3748_374870


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_30_l3748_374895

def complement (α : ℝ) : ℝ := 90 - α

def supplement (α : ℝ) : ℝ := 180 - α

theorem supplement_of_complement_of_30 :
  supplement (complement 30) = 120 := by sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_30_l3748_374895


namespace NUMINAMATH_CALUDE_two_numbers_product_cube_sum_l3748_374812

theorem two_numbers_product_cube_sum : ∃ (a b : ℚ), 
  (∃ (x : ℚ), a + (a * b) = x^3) ∧ 
  (∃ (y : ℚ), b + (a * b) = y^3) ∧ 
  a = 112/13 ∧ 
  b = 27/169 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_product_cube_sum_l3748_374812


namespace NUMINAMATH_CALUDE_cubic_sum_of_quadratic_roots_l3748_374844

theorem cubic_sum_of_quadratic_roots :
  ∀ x₁ x₂ : ℝ,
  (x₁^2 + 4*x₁ + 2 = 0) →
  (x₂^2 + 4*x₂ + 2 = 0) →
  (x₁ ≠ x₂) →
  x₁^3 + 14*x₂ + 55 = 7 :=
by sorry

end NUMINAMATH_CALUDE_cubic_sum_of_quadratic_roots_l3748_374844


namespace NUMINAMATH_CALUDE_shaded_area_of_rectangle_l3748_374852

/-- The area of the shaded part of a rectangle with specific properties -/
theorem shaded_area_of_rectangle (base height total_area : ℝ) : 
  base = 7 →
  height = 4 →
  total_area = 56 →
  total_area - 2 * (base * height / 2) = 28 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_of_rectangle_l3748_374852


namespace NUMINAMATH_CALUDE_rose_difference_after_changes_l3748_374834

/-- Calculates the difference in red roses between two people after changes -/
def rose_difference (santiago_initial : ℕ) (garrett_initial : ℕ) (given_away : ℕ) (received : ℕ) : ℕ :=
  (santiago_initial - given_away + received) - (garrett_initial - given_away + received)

theorem rose_difference_after_changes :
  rose_difference 58 24 10 5 = 34 := by
  sorry

end NUMINAMATH_CALUDE_rose_difference_after_changes_l3748_374834


namespace NUMINAMATH_CALUDE_cupboard_has_35_slots_l3748_374831

/-- Represents a cupboard with shelves and slots -/
structure Cupboard where
  shelves : ℕ
  slots_per_shelf : ℕ

/-- Represents the position of a plate in the cupboard -/
structure PlatePosition where
  shelf_from_top : ℕ
  shelf_from_bottom : ℕ
  slot_from_left : ℕ
  slot_from_right : ℕ

/-- Calculates the total number of slots in a cupboard -/
def total_slots (c : Cupboard) : ℕ := c.shelves * c.slots_per_shelf

/-- Theorem: Given the position of a plate, the cupboard has 35 slots -/
theorem cupboard_has_35_slots (pos : PlatePosition) 
  (h1 : pos.shelf_from_top = 2)
  (h2 : pos.shelf_from_bottom = 4)
  (h3 : pos.slot_from_left = 1)
  (h4 : pos.slot_from_right = 7) :
  ∃ c : Cupboard, total_slots c = 35 := by
  sorry

end NUMINAMATH_CALUDE_cupboard_has_35_slots_l3748_374831


namespace NUMINAMATH_CALUDE_torus_grid_piece_placement_impossible_l3748_374816

theorem torus_grid_piece_placement_impossible :
  ∀ (a b c : ℕ) (x y z : ℕ),
    a + b + c = 50 →
    2 * a ≤ x ∧ x ≤ 2 * b →
    2 * b ≤ y ∧ y ≤ 2 * c →
    2 * c ≤ z ∧ z ≤ 2 * a →
    False :=
by sorry

end NUMINAMATH_CALUDE_torus_grid_piece_placement_impossible_l3748_374816


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3748_374882

/-- Given a quadratic function y = x^2 + px + q + r where the minimum value is -r, 
    prove that q = p^2 / 4 -/
theorem quadratic_minimum (p q r : ℝ) : 
  (∀ x, x^2 + p*x + q + r ≥ -r) → 
  (∃ x, x^2 + p*x + q + r = -r) → 
  q = p^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3748_374882


namespace NUMINAMATH_CALUDE_circle_area_theorem_l3748_374837

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (4, 10)
def B : ℝ × ℝ := (10, 8)

-- State that A and B lie on circle ω
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω

-- Define the tangent lines at A and B
def tangent_A : Set (ℝ × ℝ) := sorry
def tangent_B : Set (ℝ × ℝ) := sorry

-- Define the intersection point of tangent lines
def intersection : ℝ × ℝ := sorry

-- State that the intersection point is on the x-axis
axiom intersection_on_x_axis : intersection.2 = 0

-- Define the area of a circle
def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem circle_area_theorem : circle_area ω = 100 * π / 9 := by sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l3748_374837


namespace NUMINAMATH_CALUDE_calc_expression_equality_simplify_fraction_equality_l3748_374848

-- Part 1
theorem calc_expression_equality : 
  (-1/2)⁻¹ + Real.sqrt 2 * Real.sqrt 6 - (π - 3)^0 + abs (Real.sqrt 3 - 2) = -1 + Real.sqrt 3 := by sorry

-- Part 2
theorem simplify_fraction_equality (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) : 
  (x^2 - 1) / (x + 1) / ((x^2 - 2*x + 1) / (x^2 - x)) = x / (x - 1) := by sorry

end NUMINAMATH_CALUDE_calc_expression_equality_simplify_fraction_equality_l3748_374848


namespace NUMINAMATH_CALUDE_union_M_complement_N_l3748_374859

def U : Finset ℕ := {1, 2, 3, 4, 5, 6}
def M : Finset ℕ := {1, 3, 4}
def N : Finset ℕ := {3, 5, 6}

theorem union_M_complement_N : M ∪ (U \ N) = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_M_complement_N_l3748_374859


namespace NUMINAMATH_CALUDE_altara_population_2040_l3748_374862

/-- Represents the population of Altara at a given year -/
def population (year : ℕ) : ℕ :=
  sorry

theorem altara_population_2040 :
  (population 2020 = 500) →
  (∀ y : ℕ, y ≥ 2020 → population (y + 10) = 2 * population y) →
  population 2040 = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_altara_population_2040_l3748_374862


namespace NUMINAMATH_CALUDE_function_properties_l3748_374865

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * (Real.log x - a) + a

theorem function_properties :
  (∃ a > 0, ∀ x > 0, f a x ≥ 0) ∧
  (∃ a > 0, ∃ x > 0, f a x ≤ 0) ∧
  (∀ a > 0, ∃ x > 0, f a x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3748_374865


namespace NUMINAMATH_CALUDE_quadratic_discriminant_zero_implies_geometric_progression_l3748_374805

/-- Given a quadratic equation ax^2 + 2bx + c = 0 with discriminant zero,
    prove that a, b, and c form a geometric progression -/
theorem quadratic_discriminant_zero_implies_geometric_progression
  (a b c : ℝ) (h : a ≠ 0) :
  (2 * b)^2 - 4 * a * c = 0 →
  ∃ (r : ℝ), r ≠ 0 ∧ b = a * r ∧ c = b * r :=
by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_zero_implies_geometric_progression_l3748_374805


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l3748_374807

theorem trigonometric_equation_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)),
    (∀ (a b c : ℝ), (a, b, c) ∈ solutions ↔
      (c ∈ Set.Icc 0 (2 * Real.pi) ∧
       ∀ x : ℝ, 2 * Real.sin (3 * x - Real.pi / 3) = a * Real.sin (b * x + c))) ∧
    Finset.card solutions = 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l3748_374807


namespace NUMINAMATH_CALUDE_no_solution_implies_m_equals_six_l3748_374877

/-- If the equation (3x - m) / (x - 2) = 1 has no solution, then m = 6 -/
theorem no_solution_implies_m_equals_six (m : ℝ) : 
  (∀ x : ℝ, (3 * x - m) / (x - 2) ≠ 1) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_equals_six_l3748_374877


namespace NUMINAMATH_CALUDE_min_c_value_l3748_374887

theorem min_c_value (a b c : ℕ+) (h1 : a < b) (h2 : b < 2*b) (h3 : 2*b < c)
  (h4 : ∃! (x y : ℝ), 3*x + y = 3000 ∧ y = |x - a| + |x - b| + |x - 2*b| + |x - c|) :
  c ≥ 502 ∧ ∃ (a' b' : ℕ+), a' < b' ∧ b' < 2*b' ∧ 2*b' < 502 ∧
    ∃! (x y : ℝ), 3*x + y = 3000 ∧ y = |x - a'| + |x - b'| + |x - 2*b'| + |x - 502| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l3748_374887


namespace NUMINAMATH_CALUDE_circle_radius_from_circumference_l3748_374829

/-- The radius of a circle with circumference 100π cm is 50 cm. -/
theorem circle_radius_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 100 * π → r = 50 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_circumference_l3748_374829


namespace NUMINAMATH_CALUDE_inequality_proof_l3748_374894

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + 4*a/(b+c)) * (1 + 4*b/(c+a)) * (1 + 4*c/(a+b)) > 25 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3748_374894


namespace NUMINAMATH_CALUDE_system_solution_l3748_374803

theorem system_solution :
  ∃ (x y : ℝ), x + y = 5 ∧ 2 * x - y = 1 ∧ x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3748_374803


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3748_374847

/-- A quadratic radical is in its simplest form if it has no fractions inside 
    the radical and no coefficients outside the radical. -/
def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = Real.sqrt n ∧ n ≠ 0 ∧ n ≠ 1 ∧ ∀ (m : ℕ), m * m ≤ n → m = 1

theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical (Real.sqrt 5) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 0.2) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt (1/2)) ∧
  ¬is_simplest_quadratic_radical (Real.sqrt 12) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3748_374847


namespace NUMINAMATH_CALUDE_rug_overlap_problem_l3748_374850

/-- Given three rugs with a combined area of 200 square meters, prove that the area
    covered by exactly two layers of rug is 5 square meters when:
    1. The rugs cover a floor area of 138 square meters when overlapped.
    2. The area covered by exactly some layers of rug is 24 square meters.
    3. The area covered by three layers of rug is 19 square meters. -/
theorem rug_overlap_problem (total_area : ℝ) (covered_area : ℝ) (some_layers_area : ℝ) (three_layers_area : ℝ)
    (h1 : total_area = 200)
    (h2 : covered_area = 138)
    (h3 : some_layers_area = 24)
    (h4 : three_layers_area = 19) :
    total_area - (covered_area + some_layers_area) = 5 := by
  sorry

end NUMINAMATH_CALUDE_rug_overlap_problem_l3748_374850


namespace NUMINAMATH_CALUDE_solution_of_cubic_system_l3748_374808

theorem solution_of_cubic_system :
  ∀ x y : ℝ, x + y = 1 ∧ x^3 + y^3 = 19 →
  (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) := by
sorry

end NUMINAMATH_CALUDE_solution_of_cubic_system_l3748_374808


namespace NUMINAMATH_CALUDE_arc_length_for_72_degrees_l3748_374898

theorem arc_length_for_72_degrees (d : ℝ) (θ_deg : ℝ) (l : ℝ) : 
  d = 4 →  -- diameter is 4 cm
  θ_deg = 72 →  -- central angle is 72°
  l = d / 2 * (θ_deg * π / 180) →  -- arc length formula
  l = 4 * π / 5 :=  -- arc length is 4π/5 cm
by sorry

end NUMINAMATH_CALUDE_arc_length_for_72_degrees_l3748_374898


namespace NUMINAMATH_CALUDE_function_value_symmetry_l3748_374854

/-- Given a function f(x) = ax^5 - bx^3 + cx where a, b, c are real numbers,
    if f(-3) = 7, then f(3) = -7 -/
theorem function_value_symmetry (a b c : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^5 - b * x^3 + c * x)
    (h2 : f (-3) = 7) : 
  f 3 = -7 := by
  sorry

end NUMINAMATH_CALUDE_function_value_symmetry_l3748_374854


namespace NUMINAMATH_CALUDE_inequality_proof_l3748_374800

theorem inequality_proof (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) 
  (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  (a + b + c + d + 1)^2 ≥ 4 * (a^2 + b^2 + c^2 + d^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3748_374800


namespace NUMINAMATH_CALUDE_salaries_degrees_l3748_374821

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  transportation : ℝ
  research_development : ℝ
  utilities : ℝ
  equipment : ℝ
  supplies : ℝ
  salaries : ℝ

/-- The total budget percentage should sum to 100% -/
axiom budget_sum (b : BudgetAllocation) : 
  b.transportation + b.research_development + b.utilities + b.equipment + b.supplies + b.salaries = 100

/-- The given budget allocation -/
def company_budget : BudgetAllocation where
  transportation := 20
  research_development := 9
  utilities := 5
  equipment := 4
  supplies := 2
  salaries := 100 - (20 + 9 + 5 + 4 + 2)

/-- The number of degrees in a full circle -/
def full_circle : ℝ := 360

/-- Theorem: The number of degrees representing salaries in the circle graph is 216 -/
theorem salaries_degrees : 
  (company_budget.salaries / 100) * full_circle = 216 := by sorry

end NUMINAMATH_CALUDE_salaries_degrees_l3748_374821


namespace NUMINAMATH_CALUDE_midpoint_sum_and_product_l3748_374881

/-- Given a line segment with endpoints (8, 15) and (-2, -3), 
    prove that the sum of the coordinates of the midpoint is 9 
    and the product of the coordinates of the midpoint is 18. -/
theorem midpoint_sum_and_product : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := 15
  let x₂ : ℝ := -2
  let y₂ : ℝ := -3
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  (midpoint_x + midpoint_y = 9) ∧ (midpoint_x * midpoint_y = 18) := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_and_product_l3748_374881


namespace NUMINAMATH_CALUDE_stacy_berries_l3748_374864

theorem stacy_berries (steve_initial : ℕ) (steve_takes : ℕ) (difference : ℕ) : 
  steve_initial = 21 → steve_takes = 4 → difference = 7 → 
  ∃ stacy_initial : ℕ, stacy_initial = 32 ∧ 
    steve_initial + steve_takes = stacy_initial - difference :=
by sorry

end NUMINAMATH_CALUDE_stacy_berries_l3748_374864


namespace NUMINAMATH_CALUDE_answer_key_combinations_l3748_374873

/-- The number of answer choices for each multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- The number of true-false questions -/
def true_false_questions : ℕ := 5

/-- The number of multiple-choice questions -/
def multiple_choice_questions : ℕ := 2

/-- The total number of possible true-false answer combinations -/
def total_true_false_combinations : ℕ := 2^true_false_questions

/-- The number of true-false combinations where all answers are the same -/
def same_answer_combinations : ℕ := 2

/-- The number of valid true-false combinations (excluding all same answers) -/
def valid_true_false_combinations : ℕ := total_true_false_combinations - same_answer_combinations

/-- The total number of possible multiple-choice answer combinations -/
def multiple_choice_combinations : ℕ := multiple_choice_options^multiple_choice_questions

/-- The theorem stating the total number of ways to create the answer key -/
theorem answer_key_combinations : 
  valid_true_false_combinations * multiple_choice_combinations = 480 := by
  sorry

end NUMINAMATH_CALUDE_answer_key_combinations_l3748_374873


namespace NUMINAMATH_CALUDE_fraction_sum_product_l3748_374892

theorem fraction_sum_product : (3 / 5 + 4 / 15) * (2 / 3) = 26 / 45 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_product_l3748_374892


namespace NUMINAMATH_CALUDE_parabola_through_negative_x_l3748_374860

/-- A parabola passing through the point (-2, 3) cannot have a standard equation of the form y^2 = 2px where p > 0 -/
theorem parabola_through_negative_x (p : ℝ) (h : p > 0) : ¬ (3^2 = 2 * p * (-2)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_negative_x_l3748_374860


namespace NUMINAMATH_CALUDE_room_area_in_sqm_l3748_374842

-- Define the room dimensions
def room_length : Real := 18
def room_width : Real := 9

-- Define the conversion factor
def sqft_to_sqm : Real := 10.7639

-- Theorem statement
theorem room_area_in_sqm :
  let area_sqft := room_length * room_width
  let area_sqm := area_sqft / sqft_to_sqm
  ⌊area_sqm⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_room_area_in_sqm_l3748_374842


namespace NUMINAMATH_CALUDE_power_equality_l3748_374833

theorem power_equality : 32^5 * 4^5 = 2^35 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3748_374833


namespace NUMINAMATH_CALUDE_last_four_average_l3748_374899

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 62 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 4 = 67.25 :=
by sorry

end NUMINAMATH_CALUDE_last_four_average_l3748_374899


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l3748_374801

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) :
  x^2 + (1 / x)^2 = 23 := by
sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l3748_374801


namespace NUMINAMATH_CALUDE_derivative_of_odd_is_even_l3748_374825

/-- A function f : ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The derivative of an odd function is an even function -/
theorem derivative_of_odd_is_even (f : ℝ → ℝ) (hf : IsOdd f) (hf' : Differentiable ℝ f) :
  IsEven (deriv f) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_odd_is_even_l3748_374825


namespace NUMINAMATH_CALUDE_simplified_irrational_expression_l3748_374804

theorem simplified_irrational_expression :
  ∃ (a b c : ℤ), 
    (c > 0) ∧ 
    (∀ (a' b' c' : ℤ), c' > 0 → 
      Real.sqrt 11 + 2 / Real.sqrt 11 + Real.sqrt 2 + 3 / Real.sqrt 2 = (a' * Real.sqrt 11 + b' * Real.sqrt 2) / c' → 
      c ≤ c') ∧
    Real.sqrt 11 + 2 / Real.sqrt 11 + Real.sqrt 2 + 3 / Real.sqrt 2 = (a * Real.sqrt 11 + b * Real.sqrt 2) / c ∧
    a = 11 ∧ b = 44 ∧ c = 22 := by
  sorry

end NUMINAMATH_CALUDE_simplified_irrational_expression_l3748_374804


namespace NUMINAMATH_CALUDE_number_exceeds_fraction_by_40_l3748_374863

theorem number_exceeds_fraction_by_40 (x : ℝ) : x = (3 / 8) * x + 40 → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeds_fraction_by_40_l3748_374863


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3748_374849

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let A := ((x - y) * Real.sqrt (x^2 + y^2) + 
            (y - z) * Real.sqrt (y^2 + z^2) + 
            (z - x) * Real.sqrt (z^2 + x^2) + 
            Real.sqrt 2) / 
           ((x - y)^2 + (y - z)^2 + (z - x)^2 + 2)
  A ≤ 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3748_374849


namespace NUMINAMATH_CALUDE_function_equality_l3748_374819

theorem function_equality (f : ℝ → ℝ) (h : ∀ x, f x = 3 * x - 5) :
  2 * (f 3) - 10 = f (3 - 2) := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l3748_374819
