import Mathlib

namespace local_face_value_difference_l345_34577

def numeral : ℕ := 96348621

theorem local_face_value_difference :
  let digit : ℕ := 8
  let position : ℕ := 5  -- 1-indexed from right, so 8 is in the 5th position
  let local_value : ℕ := digit * (10 ^ (position - 1))
  let face_value : ℕ := digit
  local_value - face_value = 79992 :=
by sorry

end local_face_value_difference_l345_34577


namespace imaginary_part_of_z_div_i_l345_34568

-- Define the complex number z
def z : ℂ := Complex.mk 1 (-3)

-- Theorem statement
theorem imaginary_part_of_z_div_i : (z / Complex.I).im = -1 := by
  sorry

end imaginary_part_of_z_div_i_l345_34568


namespace opposite_face_is_D_l345_34502

/-- Represents the labels of the faces of a cube --/
inductive FaceLabel
  | A | B | C | D | E | F

/-- Represents the positions of faces on a cube --/
inductive Position
  | Top | Bottom | Left | Right | Front | Back

/-- Represents a cube with labeled faces --/
structure Cube where
  faces : Position → FaceLabel

/-- Defines the opposite position for each position on the cube --/
def oppositePosition : Position → Position
  | Position.Top => Position.Bottom
  | Position.Bottom => Position.Top
  | Position.Left => Position.Right
  | Position.Right => Position.Left
  | Position.Front => Position.Back
  | Position.Back => Position.Front

/-- Theorem stating that in a cube where C is on top and B is to its right, 
    the face opposite to A is labeled D --/
theorem opposite_face_is_D (cube : Cube) 
  (h1 : cube.faces Position.Top = FaceLabel.C) 
  (h2 : cube.faces Position.Right = FaceLabel.B) : 
  ∃ p : Position, cube.faces p = FaceLabel.A ∧ 
  cube.faces (oppositePosition p) = FaceLabel.D := by
  sorry

end opposite_face_is_D_l345_34502


namespace integral_reciprocal_sqrt_one_minus_x_squared_l345_34543

open Real MeasureTheory

theorem integral_reciprocal_sqrt_one_minus_x_squared : 
  ∫ x in (Set.Icc 0 (1 / Real.sqrt 2)), 1 / ((1 - x^2) * Real.sqrt (1 - x^2)) = 1 := by
  sorry

end integral_reciprocal_sqrt_one_minus_x_squared_l345_34543


namespace cloth_trimming_l345_34500

/-- Given a square piece of cloth with side length 22 feet, prove that after trimming 6 feet from two opposite edges and 5 feet from the other two edges, the remaining area is 272 square feet. -/
theorem cloth_trimming (original_length : ℕ) (trim_1 : ℕ) (trim_2 : ℕ) : 
  original_length = 22 → 
  trim_1 = 6 → 
  trim_2 = 5 → 
  (original_length - trim_1) * (original_length - trim_2) = 272 := by
sorry

end cloth_trimming_l345_34500


namespace share_change_l345_34531

theorem share_change (total money : ℝ) (ostap_share kisa_share : ℝ) 
  (h1 : ostap_share + kisa_share = total)
  (h2 : ostap_share = 1.5 * kisa_share) :
  let new_ostap_share := 1.5 * ostap_share
  let new_kisa_share := total - new_ostap_share
  new_kisa_share = 0.25 * kisa_share := by
sorry

end share_change_l345_34531


namespace root_of_equation_l345_34554

theorem root_of_equation (x : ℝ) : 
  (2 * x^3 - 3 * x^2 - 13 * x + 10) * (x - 1) = 0 ↔ x = 1 := by
  sorry

end root_of_equation_l345_34554


namespace complex_magnitude_l345_34516

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - 2 * w) = 15)
  (h2 : Complex.abs (2 * z + 3 * w) = 10)
  (h3 : Complex.abs (z - w) = 3) :
  Complex.abs z = 4.5 := by sorry

end complex_magnitude_l345_34516


namespace difference_of_percentages_l345_34590

theorem difference_of_percentages (x y : ℝ) : 
  0.60 * (50 + x) - 0.45 * (30 + y) = 16.5 + 0.60 * x - 0.45 * y :=
by sorry

end difference_of_percentages_l345_34590


namespace four_digit_equal_digits_l345_34501

theorem four_digit_equal_digits (n : ℤ) : 12 * n^2 + 12 * n + 11 = 5555 ↔ n = 21 ∨ n = -22 := by
  sorry

end four_digit_equal_digits_l345_34501


namespace ceiling_floor_difference_l345_34529

theorem ceiling_floor_difference : 
  ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌈(-34 : ℝ) / 4⌉⌋ = 0 := by
  sorry

end ceiling_floor_difference_l345_34529


namespace boat_license_count_l345_34585

/-- The number of possible letters for a boat license -/
def num_letters : Nat := 3

/-- The number of possible digits for each position in a boat license -/
def num_digits : Nat := 10

/-- The number of digit positions in a boat license -/
def num_positions : Nat := 5

/-- The total number of possible boat licenses -/
def total_licenses : Nat := num_letters * (num_digits ^ num_positions)

theorem boat_license_count : total_licenses = 300000 := by
  sorry

end boat_license_count_l345_34585


namespace expression_evaluation_l345_34539

theorem expression_evaluation : 
  let a := (1/4 + 1/12 - 7/18 - 1/36)
  let b := 1/36
  (b / a + a / b) = -10/3 := by sorry

end expression_evaluation_l345_34539


namespace gcd_cube_plus_five_cube_l345_34546

theorem gcd_cube_plus_five_cube (n : ℕ) (h : n > 2^5) : Nat.gcd (n^3 + 5^3) (n + 5) = 1 := by
  sorry

end gcd_cube_plus_five_cube_l345_34546


namespace sum_equals_twelve_l345_34547

theorem sum_equals_twelve (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 := by
  sorry

end sum_equals_twelve_l345_34547


namespace cube_sum_theorem_l345_34514

/-- Represents a cube with integers on its faces -/
structure Cube where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+
  f : ℕ+

/-- The sum of vertex products for a cube -/
def vertexSum (cube : Cube) : ℕ :=
  2 * (cube.a * cube.b * cube.c +
       cube.a * cube.b * cube.f +
       cube.d * cube.b * cube.c +
       cube.d * cube.b * cube.f)

/-- The sum of face numbers for a cube -/
def faceSum (cube : Cube) : ℕ :=
  cube.a + cube.b + cube.c + cube.d + cube.e + cube.f

/-- Theorem stating the relationship between vertex sum and face sum -/
theorem cube_sum_theorem (cube : Cube) 
  (h1 : vertexSum cube = 1332)
  (h2 : cube.b = cube.e) : 
  faceSum cube = 47 := by
  sorry

end cube_sum_theorem_l345_34514


namespace circular_seating_arrangement_l345_34519

/-- Given a circular arrangement of students where the 5th position
    is opposite the 20th position, prove that there are 32 students in total. -/
theorem circular_seating_arrangement (n : ℕ) 
  (h : n > 0)  -- Ensure positive number of students
  (opposite : ∀ (a b : ℕ), a ≤ n → b ≤ n → (a + n / 2) % n = b % n → a = 5 ∧ b = 20) :
  n = 32 := by
  sorry

end circular_seating_arrangement_l345_34519


namespace largest_base5_5digit_in_base10_l345_34599

/-- The largest base-5 number with five digits -/
def largest_base5_5digit : ℕ := 44444

/-- Convert a base-5 number to base 10 -/
def base5_to_base10 (n : ℕ) : ℕ :=
  (n / 10000) * 5^4 + ((n / 1000) % 5) * 5^3 + ((n / 100) % 5) * 5^2 + ((n / 10) % 5) * 5^1 + (n % 5) * 5^0

theorem largest_base5_5digit_in_base10 :
  base5_to_base10 largest_base5_5digit = 3124 := by
  sorry

end largest_base5_5digit_in_base10_l345_34599


namespace vertex_y_is_negative_three_l345_34565

/-- Quadratic function f(x) = 2x^2 - 4x - 1 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 1

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

theorem vertex_y_is_negative_three :
  f vertex_x = -3 := by
  sorry

end vertex_y_is_negative_three_l345_34565


namespace infinite_primes_with_solutions_l345_34597

theorem infinite_primes_with_solutions (S : Finset Nat) (h : ∀ p ∈ S, Nat.Prime p) :
  ∃ p : Nat, p ∉ S ∧ Nat.Prime p ∧ ∃ x : ℤ, x^2 + x + 1 = p := by
  sorry

end infinite_primes_with_solutions_l345_34597


namespace square_perimeter_from_area_l345_34518

theorem square_perimeter_from_area (area : ℝ) (side : ℝ) (perimeter : ℝ) :
  area = 144 →
  side * side = area →
  perimeter = 4 * side →
  perimeter = 48 := by
sorry

end square_perimeter_from_area_l345_34518


namespace irregular_hexagon_perimeter_l345_34584

/-- An irregular hexagon with specific angle measurements and equal side lengths -/
structure IrregularHexagon where
  -- Side length of the hexagon
  side_length : ℝ
  -- Assumption that all sides are equal
  all_sides_equal : True
  -- Three nonadjacent angles measure 120°
  three_angles_120 : True
  -- The other three angles measure 60°
  three_angles_60 : True
  -- The enclosed area of the hexagon
  area : ℝ
  -- The area is 24
  area_is_24 : area = 24

/-- The perimeter of an irregular hexagon with the given conditions -/
def perimeter (h : IrregularHexagon) : ℝ := 6 * h.side_length

/-- Theorem stating that the perimeter of the irregular hexagon is 24 / (3^(1/4)) -/
theorem irregular_hexagon_perimeter (h : IrregularHexagon) : 
  perimeter h = 24 / Real.rpow 3 (1/4) := by
  sorry

end irregular_hexagon_perimeter_l345_34584


namespace smallest_k_for_digit_sum_945_l345_34598

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The number formed by k repetitions of the digit 7 -/
def repeated_sevens (k : ℕ) : ℕ := sorry

theorem smallest_k_for_digit_sum_945 :
  (∀ k < 312, sum_of_digits (7 * repeated_sevens k) < 945) ∧
  sum_of_digits (7 * repeated_sevens 312) = 945 := by sorry

end smallest_k_for_digit_sum_945_l345_34598


namespace gwen_candy_weight_l345_34593

/-- The amount of candy Gwen received given the total amount and Frank's amount -/
def gwens_candy (total frank : ℕ) : ℕ := total - frank

/-- Theorem stating that Gwen received 7 pounds of candy -/
theorem gwen_candy_weight :
  let total := 17
  let frank := 10
  gwens_candy total frank = 7 := by sorry

end gwen_candy_weight_l345_34593


namespace max_d_value_l345_34574

def a (n : ℕ) : ℕ := 100 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value : ∃ (k : ℕ), ∀ (n : ℕ), n > 0 → d n ≤ k ∧ ∃ (m : ℕ), m > 0 ∧ d m = k :=
sorry

end max_d_value_l345_34574


namespace negation_of_existence_l345_34586

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) :=
by sorry

end negation_of_existence_l345_34586


namespace function_symmetric_about_two_lines_is_periodic_l345_34580

/-- Given a function f: ℝ → ℝ that is symmetric about x = a and x = b (where a ≠ b),
    prove that f is periodic with period 2b - 2a. -/
theorem function_symmetric_about_two_lines_is_periodic
  (f : ℝ → ℝ) (a b : ℝ) (h_neq : a ≠ b)
  (h_sym_a : ∀ x, f (a - x) = f (a + x))
  (h_sym_b : ∀ x, f (b - x) = f (b + x)) :
  ∀ x, f x = f (x + 2*b - 2*a) :=
sorry

end function_symmetric_about_two_lines_is_periodic_l345_34580


namespace robin_gum_packages_l345_34595

/-- The number of pieces of gum in each package -/
def pieces_per_package : ℕ := 15

/-- The total number of pieces of gum Robin has -/
def total_pieces : ℕ := 135

/-- The number of packages Robin has -/
def num_packages : ℕ := total_pieces / pieces_per_package

theorem robin_gum_packages : num_packages = 9 := by
  sorry

end robin_gum_packages_l345_34595


namespace not_reach_54_after_60_ops_l345_34511

/-- Represents the possible operations on the board number -/
inductive Operation
| MultTwo
| DivTwo
| MultThree
| DivThree

/-- Applies an operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.MultTwo => n * 2
  | Operation.DivTwo => n / 2
  | Operation.MultThree => n * 3
  | Operation.DivThree => n / 3

/-- Applies a sequence of operations to a number -/
def applyOperations (n : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation n

/-- Theorem: After 60 operations starting from 12, it's impossible to reach 54 -/
theorem not_reach_54_after_60_ops (ops : List Operation) :
  ops.length = 60 → applyOperations 12 ops ≠ 54 := by
  sorry

end not_reach_54_after_60_ops_l345_34511


namespace unique_number_satisfying_conditions_l345_34576

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Reverses the digits of a ThreeDigitNumber -/
def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.units
  tens := n.tens
  units := n.hundreds
  is_valid := by sorry

theorem unique_number_satisfying_conditions :
  ∃! (n : ThreeDigitNumber),
    n.hundreds + n.tens + n.units = 20 ∧
    ∃ (m : ThreeDigitNumber),
      n.toNat - 16 = m.toNat ∧
      m = n.reverse ∧
    n.hundreds = 9 ∧ n.tens = 7 ∧ n.units = 4 := by sorry

end unique_number_satisfying_conditions_l345_34576


namespace sets_problem_l345_34545

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}
def B : Set ℝ := {x | 0 < x ∧ x < 5}
def C (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ 2*m}

-- Theorem statement
theorem sets_problem :
  (∀ x : ℝ, x ∈ A ∩ B ↔ 4 ≤ x ∧ x < 5) ∧
  (∀ x : ℝ, x ∈ (Set.univ \ A) ∪ B ↔ -1 < x ∧ x < 5) ∧
  (∀ m : ℝ, B ∩ C m = C m ↔ m < -2 ∨ (2 < m ∧ m < 5/2)) :=
sorry

end sets_problem_l345_34545


namespace light_travel_distance_l345_34596

/-- The distance light travels in one year (in miles) -/
def light_year : ℕ := 5870000000000

/-- The number of years we're calculating for -/
def years : ℕ := 500

/-- The expected distance light travels in 500 years (in miles) -/
def expected_distance : ℕ := 2935 * (10^12)

theorem light_travel_distance :
  (light_year * years : ℕ) = expected_distance := by
  sorry

end light_travel_distance_l345_34596


namespace percentage_of_older_female_students_l345_34542

theorem percentage_of_older_female_students
  (total_students : ℝ)
  (h1 : total_students > 0)
  (h2 : 0.4 * total_students = male_students)
  (h3 : 0.5 * male_students = older_male_students)
  (h4 : 0.56 * total_students = younger_students)
  : 0.4 * (total_students - male_students) = older_female_students :=
by
  sorry

end percentage_of_older_female_students_l345_34542


namespace fraction_equality_implies_zero_l345_34571

theorem fraction_equality_implies_zero (x : ℝ) : 
  (4 + x) / (6 + x) = (2 + x) / (3 + x) → x = 0 := by
  sorry

end fraction_equality_implies_zero_l345_34571


namespace cross_section_properties_l345_34587

/-- Regular triangular prism with given dimensions -/
structure RegularTriangularPrism where
  base_side_length : ℝ
  height : ℝ

/-- Cross-section of the prism -/
structure CrossSection where
  area : ℝ
  angle_with_base : ℝ

/-- Theorem about the cross-section of a specific regular triangular prism -/
theorem cross_section_properties (prism : RegularTriangularPrism) 
  (h1 : prism.base_side_length = 6)
  (h2 : prism.height = (1/3) * Real.sqrt 7) :
  ∃ (cs : CrossSection), cs.area = 39/4 ∧ cs.angle_with_base = 30 * π / 180 := by
  sorry

end cross_section_properties_l345_34587


namespace interest_rate_calculation_l345_34582

/-- The compound interest formula: A = P * (1 + r)^n -/
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem interest_rate_calculation
  (P : ℝ)  -- Principal amount
  (r : ℝ)  -- Rate of interest (as a decimal)
  (h1 : compound_interest P r 3 = 800)
  (h2 : compound_interest P r 4 = 820) :
  r = 0.025 := by
sorry

end interest_rate_calculation_l345_34582


namespace problem_solution_l345_34508

theorem problem_solution (A B : Set ℝ) (a b : ℝ) : 
  A = {2, 3} →
  B = {x : ℝ | x^2 + a*x + b = 0} →
  A ∩ B = {2} →
  A ∪ B = A →
  (a + b = 0 ∨ a + b = 1) :=
by sorry

end problem_solution_l345_34508


namespace rectangle_length_l345_34513

theorem rectangle_length (w l : ℝ) (h1 : w > 0) (h2 : l > 0) : 
  (2*l + 2*w) / w = 5 → l * w = 150 → l = 15 := by
  sorry

end rectangle_length_l345_34513


namespace max_sum_of_squares_l345_34533

theorem max_sum_of_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 →
  n ∈ Finset.range 1982 →
  (n^2 - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end max_sum_of_squares_l345_34533


namespace gcf_of_45_135_60_l345_34559

theorem gcf_of_45_135_60 : Nat.gcd 45 (Nat.gcd 135 60) = 15 := by
  sorry

end gcf_of_45_135_60_l345_34559


namespace buckingham_palace_visitors_l345_34520

def visitors_previous_day : ℕ := 100
def additional_visitors : ℕ := 566

theorem buckingham_palace_visitors :
  visitors_previous_day + additional_visitors = 666 := by
  sorry

end buckingham_palace_visitors_l345_34520


namespace no_numbers_equal_seven_times_digit_sum_l345_34523

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem no_numbers_equal_seven_times_digit_sum :
  ∀ n : ℕ, n > 0 ∧ n < 2000 → n ≠ 7 * (sum_of_digits n) :=
by
  sorry

end no_numbers_equal_seven_times_digit_sum_l345_34523


namespace min_draws_for_eighteen_l345_34579

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  purple : Nat

/-- The minimum number of balls needed to guarantee at least n of a single color -/
def minDraws (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The specific ball counts in our problem -/
def problemCounts : BallCounts :=
  { red := 34, green := 25, yellow := 18, blue := 21, purple := 13 }

/-- The theorem stating the minimum number of draws needed -/
theorem min_draws_for_eighteen (counts : BallCounts) :
  counts = problemCounts → minDraws counts 18 = 82 :=
  sorry

end min_draws_for_eighteen_l345_34579


namespace problem_statement_l345_34509

theorem problem_statement (x y : ℝ) 
  (h1 : (4 : ℝ)^y = 1 / (8 * (Real.sqrt 2)^(x + 2)))
  (h2 : (9 : ℝ)^x * (3 : ℝ)^y = 3 * Real.sqrt 3) :
  (5 : ℝ)^(x + y) = 1 / Real.sqrt 5 := by
  sorry

end problem_statement_l345_34509


namespace distance_B_to_center_l345_34515

/-- A circle with radius √52 and points A, B, C satisfying given conditions -/
structure NotchedCircle where
  -- Define the circle
  center : ℝ × ℝ
  radius : ℝ
  radius_eq : radius = Real.sqrt 52

  -- Define points A, B, C
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

  -- Conditions
  on_circle_A : (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2
  on_circle_B : (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2
  on_circle_C : (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2

  AB_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64  -- 8^2 = 64
  BC_length : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 16  -- 4^2 = 16

  right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0

/-- The square of the distance from point B to the center of the circle is 20 -/
theorem distance_B_to_center (nc : NotchedCircle) :
  (nc.B.1 - nc.center.1)^2 + (nc.B.2 - nc.center.2)^2 = 20 := by
  sorry

end distance_B_to_center_l345_34515


namespace geometric_arithmetic_sequence_sum_squares_l345_34541

theorem geometric_arithmetic_sequence_sum_squares (x y z : ℝ) 
  (h1 : (4*y)^2 = (3*x)*(5*z))  -- Geometric sequence condition
  (h2 : y^2 = (x^2 + z^2)/2)    -- Arithmetic sequence condition
  : x^2 + z^2 = 16 :=
by sorry

end geometric_arithmetic_sequence_sum_squares_l345_34541


namespace raffle_prize_calculation_l345_34504

theorem raffle_prize_calculation (kept_amount : ℝ) (kept_percentage : ℝ) (total_prize : ℝ) : 
  kept_amount = 80 → kept_percentage = 0.80 → kept_amount = kept_percentage * total_prize → 
  total_prize = 100 := by
sorry

end raffle_prize_calculation_l345_34504


namespace unique_function_solution_l345_34560

/-- A function f: ℕ → ℤ is an increasing function that satisfies the given conditions -/
def IsValidFunction (f : ℕ → ℤ) : Prop :=
  (∀ m n : ℕ, m < n → f m < f n) ∧ 
  (f 2 = 7) ∧
  (∀ m n : ℕ, f (m * n) = f m + f n + f m * f n)

/-- The theorem stating that the only function satisfying the conditions is f(n) = n³ - 1 -/
theorem unique_function_solution :
  ∀ f : ℕ → ℤ, IsValidFunction f → ∀ n : ℕ, f n = n^3 - 1 :=
by sorry

end unique_function_solution_l345_34560


namespace integer_triples_theorem_l345_34535

def satisfies_conditions (a b c : ℤ) : Prop :=
  a + b + c = 24 ∧ a^2 + b^2 + c^2 = 210 ∧ a * b * c = 440

def solution_set : Set (ℤ × ℤ × ℤ) :=
  {(11, 8, 5), (8, 11, 5), (8, 5, 11), (5, 8, 11), (11, 5, 8), (5, 11, 8)}

theorem integer_triples_theorem :
  ∀ (a b c : ℤ), satisfies_conditions a b c ↔ (a, b, c) ∈ solution_set :=
by sorry

end integer_triples_theorem_l345_34535


namespace union_and_intersection_when_a_is_two_range_of_a_for_necessary_but_not_sufficient_l345_34534

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B (a : ℝ) : Set ℝ := {x | 2 - a < x ∧ x < 2 + a}

-- Theorem for part (1)
theorem union_and_intersection_when_a_is_two :
  (A ∪ B 2 = {x | -1 < x ∧ x < 4}) ∧ (A ∩ B 2 = {x | 0 < x ∧ x < 3}) := by sorry

-- Theorem for part (2)
theorem range_of_a_for_necessary_but_not_sufficient :
  {a : ℝ | ∀ x, x ∈ B a → x ∈ A} ∩ {a : ℝ | ∃ x, x ∈ B a ∧ x ∉ A} = Set.Iic 1 := by sorry

end union_and_intersection_when_a_is_two_range_of_a_for_necessary_but_not_sufficient_l345_34534


namespace magnitude_of_z_l345_34537

def z : ℂ := (1 + Complex.I) * (2 - Complex.I)

theorem magnitude_of_z : Complex.abs z = Real.sqrt 10 := by
  sorry

end magnitude_of_z_l345_34537


namespace unique_absolute_value_complex_root_l345_34556

theorem unique_absolute_value_complex_root : ∃! r : ℝ, 
  (∃ z : ℂ, z^2 - 6*z + 20 = 0 ∧ Complex.abs z = r) ∧ r ≥ 3 := by
  sorry

end unique_absolute_value_complex_root_l345_34556


namespace simplify_expression_l345_34557

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end simplify_expression_l345_34557


namespace scooter_safety_gear_cost_increase_l345_34561

/-- The percent increase in the combined cost of a scooter and safety gear set --/
theorem scooter_safety_gear_cost_increase (scooter_cost safety_gear_cost : ℝ)
  (scooter_increase safety_gear_increase : ℝ) :
  scooter_cost = 200 →
  safety_gear_cost = 50 →
  scooter_increase = 0.08 →
  safety_gear_increase = 0.15 →
  let new_scooter_cost := scooter_cost * (1 + scooter_increase)
  let new_safety_gear_cost := safety_gear_cost * (1 + safety_gear_increase)
  let total_original_cost := scooter_cost + safety_gear_cost
  let total_new_cost := new_scooter_cost + new_safety_gear_cost
  let percent_increase := (total_new_cost - total_original_cost) / total_original_cost * 100
  ∃ ε > 0, |percent_increase - 9| < ε :=
by sorry

end scooter_safety_gear_cost_increase_l345_34561


namespace industrial_park_investment_l345_34506

theorem industrial_park_investment
  (total_investment : ℝ)
  (return_rate_A : ℝ)
  (return_rate_B : ℝ)
  (total_return : ℝ)
  (h1 : total_investment = 2000)
  (h2 : return_rate_A = 0.054)
  (h3 : return_rate_B = 0.0828)
  (h4 : total_return = 122.4)
  : ∃ (investment_A investment_B : ℝ),
    investment_A + investment_B = total_investment ∧
    investment_A * return_rate_A + investment_B * return_rate_B = total_return ∧
    investment_A = 1500 ∧
    investment_B = 500 := by
  sorry

end industrial_park_investment_l345_34506


namespace thirteen_divides_six_digit_reverse_perm_l345_34575

/-- A 6-digit positive integer whose first three digits are a permutation of its last three digits taken in reverse order. -/
def SixDigitReversePerm : Type :=
  {n : ℕ // 100000 ≤ n ∧ n < 1000000 ∧ ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ a ≠ 0 ∧
    (n = 100000*a + 10000*b + 1000*c + 100*c + 10*b + a ∨
     n = 100000*a + 10000*c + 1000*b + 100*b + 10*c + a ∨
     n = 100000*b + 10000*a + 1000*c + 100*c + 10*a + b ∨
     n = 100000*b + 10000*c + 1000*a + 100*a + 10*c + b ∨
     n = 100000*c + 10000*a + 1000*b + 100*b + 10*a + c ∨
     n = 100000*c + 10000*b + 1000*a + 100*a + 10*b + c)}

theorem thirteen_divides_six_digit_reverse_perm (x : SixDigitReversePerm) :
  13 ∣ x.val :=
sorry

end thirteen_divides_six_digit_reverse_perm_l345_34575


namespace joe_cars_count_l345_34505

/-- Proves that Joe will have 62 cars after getting 12 more cars, given he initially had 50 cars. -/
theorem joe_cars_count (initial_cars : ℕ) (additional_cars : ℕ) : 
  initial_cars = 50 → additional_cars = 12 → initial_cars + additional_cars = 62 := by
  sorry

end joe_cars_count_l345_34505


namespace different_tens_digit_probability_l345_34566

def lower_bound : ℕ := 30
def upper_bound : ℕ := 89
def num_integers : ℕ := 7

def favorable_outcomes : ℕ := 27000000
def total_outcomes : ℕ := Nat.choose (upper_bound - lower_bound + 1) num_integers

theorem different_tens_digit_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 6750 / 9655173 := by
  sorry

end different_tens_digit_probability_l345_34566


namespace bryans_collection_total_l345_34551

/-- Calculates the total number of reading materials in Bryan's collection --/
def total_reading_materials (
  num_shelves : ℕ
) (
  books_per_shelf : ℕ
) (
  magazines_per_shelf : ℕ
) (
  newspapers_per_shelf : ℕ
) (
  graphic_novels_per_shelf : ℕ
) : ℕ :=
  num_shelves * (books_per_shelf + magazines_per_shelf + newspapers_per_shelf + graphic_novels_per_shelf)

/-- Proves that Bryan's collection contains 4810 reading materials --/
theorem bryans_collection_total :
  total_reading_materials 37 23 61 17 29 = 4810 := by
  sorry

end bryans_collection_total_l345_34551


namespace triangle_properties_l345_34563

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.b = 5 ∧
  t.c = Real.sqrt 7 ∧
  4 * (Real.sin ((t.A + t.B) / 2))^2 - Real.cos (2 * t.C) = 7/2 ∧
  t.A + t.B + t.C = Real.pi

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 3 ∧ (1/2 * t.a * t.b * Real.sin t.C) = (3 * Real.sqrt 3) / 2 := by
  sorry

end triangle_properties_l345_34563


namespace fifth_row_sum_in_spiral_grid_l345_34562

/-- Represents a spiral arrangement of numbers in a square grid -/
def SpiralGrid (n : ℕ) := Matrix (Fin n) (Fin n) ℕ

/-- Creates a spiral grid of size n × n with numbers from 1 to n^2 -/
def createSpiralGrid (n : ℕ) : SpiralGrid n :=
  sorry

/-- Returns the numbers in a specific row of the spiral grid -/
def getRowNumbers (grid : SpiralGrid 20) (row : Fin 20) : List ℕ :=
  sorry

/-- Theorem: In a 20x20 spiral grid, the sum of the greatest and least numbers 
    in the fifth row is 565 -/
theorem fifth_row_sum_in_spiral_grid :
  let grid := createSpiralGrid 20
  let fifthRowNumbers := getRowNumbers grid 4
  (List.maximum fifthRowNumbers).getD 0 + (List.minimum fifthRowNumbers).getD 0 = 565 := by
  sorry

end fifth_row_sum_in_spiral_grid_l345_34562


namespace arithmetic_sequence_with_geometric_sum_l345_34538

/-- The sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a : ℕ → ℚ) : ℚ := (n : ℚ) * (a 1 + a n) / 2

/-- An arithmetic sequence with common difference -1 -/
def arithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∀ n, a (n + 1) = a n - 1

/-- S_1, S_2, S_4 form a geometric sequence -/
def geometricSequence (a : ℕ → ℚ) : Prop :=
  (S 2 a) ^ 2 = (S 1 a) * (S 4 a)

theorem arithmetic_sequence_with_geometric_sum 
  (a : ℕ → ℚ) 
  (h1 : arithmeticSequence a) 
  (h2 : geometricSequence a) : 
  ∀ n, a n = 1/2 - n := by
  sorry

end arithmetic_sequence_with_geometric_sum_l345_34538


namespace sum_between_15_and_16_l345_34522

theorem sum_between_15_and_16 : 
  let a : ℚ := 10/3
  let b : ℚ := 19/4
  let c : ℚ := 123/20
  15 < a + b + c ∧ a + b + c < 16 := by
  sorry

end sum_between_15_and_16_l345_34522


namespace sum_base6_numbers_l345_34592

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The sum of the given base 6 numbers equals 3153₆ --/
theorem sum_base6_numbers :
  let n1 := [4, 3, 2, 1]  -- 1234₆
  let n2 := [4, 5, 6]     -- 654₆
  let n3 := [1, 2, 3]     -- 321₆
  let n4 := [6, 5]        -- 56₆
  base10ToBase6 (base6ToBase10 n1 + base6ToBase10 n2 + base6ToBase10 n3 + base6ToBase10 n4) = [3, 1, 5, 3] :=
by sorry

end sum_base6_numbers_l345_34592


namespace student_desk_arrangement_impossibility_l345_34527

theorem student_desk_arrangement_impossibility :
  ∀ (total_students total_desks : ℕ) 
    (girls boys : ℕ) 
    (girls_with_boys boys_with_girls : ℕ),
  total_students = 450 →
  total_desks = 225 →
  girls + boys = total_students →
  2 * total_desks = total_students →
  2 * girls_with_boys = girls →
  2 * boys_with_girls = boys →
  False :=
by sorry

end student_desk_arrangement_impossibility_l345_34527


namespace m_range_proof_l345_34581

theorem m_range_proof (m : ℝ) (h_pos : m > 0) : 
  (∀ x : ℝ, x ≤ -1 → (3 * m - 1) * 2^x < 1) → 
  m < 1 :=
by sorry

end m_range_proof_l345_34581


namespace classroom_size_is_81_l345_34573

/-- Represents the number of students in a classroom with specific shirt and shorts conditions. -/
def classroom_size : ℕ → Prop := fun n =>
  ∃ (striped checkered shorts : ℕ),
    -- Total number of students
    n = striped + checkered
    -- Two-thirds wear striped shirts, one-third wear checkered shirts
    ∧ 3 * striped = 2 * n
    ∧ 3 * checkered = n
    -- Shorts condition
    ∧ shorts = checkered + 19
    -- Striped shirts condition
    ∧ striped = shorts + 8

/-- The number of students in the classroom satisfying the given conditions is 81. -/
theorem classroom_size_is_81 : classroom_size 81 := by
  sorry

end classroom_size_is_81_l345_34573


namespace new_student_weight_l345_34578

/-- 
Given 5 students with an initial total weight W, 
if replacing two students weighing x and y with a new student 
causes the average weight to decrease by 8 kg, 
then the new student's weight is 40 kg less than x + y.
-/
theorem new_student_weight 
  (W : ℝ) -- Initial total weight of 5 students
  (x y : ℝ) -- Weights of the two replaced students
  (new_avg : ℝ) -- New average weight after replacement
  (h1 : new_avg = (W - x - y + (x + y - 40)) / 5) -- New average calculation
  (h2 : W / 5 - new_avg = 8) -- Average weight decrease
  : x + y - 40 = (x + y) - 40 := by sorry

end new_student_weight_l345_34578


namespace twenty_four_is_forty_eight_percent_of_fifty_l345_34555

theorem twenty_four_is_forty_eight_percent_of_fifty :
  ∃ x : ℝ, (24 : ℝ) / x = 48 / 100 ∧ x = 50 := by
  sorry

end twenty_four_is_forty_eight_percent_of_fifty_l345_34555


namespace locus_properties_l345_34507

/-- The locus of point R in a specific geometric configuration -/
def locus_equation (a b c x y : ℝ) : Prop :=
  b^2 * x^2 - 2*a*b*x*y + a*(a - c)*y^2 - b^2*c*x + 2*a*b*c*y = 0

/-- The type of curve represented by the locus equation -/
inductive CurveType
  | Ellipse
  | Hyperbola

/-- Theorem stating the properties of the locus and its curve type -/
theorem locus_properties (a b c : ℝ) (h1 : b > 0) (h2 : c > 0) (h3 : a ≠ c) :
  ∃ (curve_type : CurveType),
    (∀ x y : ℝ, locus_equation a b c x y) ∧
    ((a < 0 → curve_type = CurveType.Ellipse) ∧
     (a > 0 → curve_type = CurveType.Hyperbola)) :=
by sorry

end locus_properties_l345_34507


namespace alex_bill_correct_l345_34536

/-- Calculates the cell phone bill based on the given parameters. -/
def calculate_bill (base_cost : ℚ) (text_cost : ℚ) (extra_minute_cost : ℚ) 
                   (discount : ℚ) (texts_sent : ℕ) (hours_talked : ℕ) : ℚ :=
  let text_charge := text_cost * texts_sent
  let extra_minutes := max (hours_talked * 60 - 25 * 60) 0
  let extra_minute_charge := extra_minute_cost * extra_minutes
  let subtotal := base_cost + text_charge + extra_minute_charge
  let final_bill := if hours_talked > 35 then subtotal - discount else subtotal
  final_bill

theorem alex_bill_correct :
  calculate_bill 30 0.1 0.12 5 150 36 = 119.2 := by
  sorry

end alex_bill_correct_l345_34536


namespace festival_worker_assignment_l345_34589

def number_of_workers : ℕ := 6

def number_of_desks : ℕ := 2

def min_workers_per_desk : ℕ := 2

def ways_to_assign_workers (n : ℕ) (k : ℕ) (min_per_group : ℕ) : ℕ :=
  sorry

theorem festival_worker_assignment :
  ways_to_assign_workers number_of_workers number_of_desks min_workers_per_desk = 28 :=
sorry

end festival_worker_assignment_l345_34589


namespace max_sphere_radius_in_glass_l345_34567

theorem max_sphere_radius_in_glass (x : ℝ) :
  let r := (3 * 2^(1/3)) / 4
  let glass_curve := fun x => x^4
  let sphere_equation := fun (x y : ℝ) => x^2 + (y - r)^2 = r^2
  (∃ y, y = glass_curve x ∧ sphere_equation x y) ∧
  (∀ r' > r, ∃ x y, y < glass_curve x ∧ x^2 + (y - r')^2 = r'^2) ∧
  sphere_equation 0 0 :=
by sorry

end max_sphere_radius_in_glass_l345_34567


namespace money_distribution_l345_34588

theorem money_distribution (total : ℝ) (p q r : ℝ) : 
  p + q + r = total →
  p / q = 3 / 7 →
  q / r = 7 / 12 →
  q - p = 2400 →
  r - q = 3000 :=
by sorry

end money_distribution_l345_34588


namespace room_occupancy_l345_34550

theorem room_occupancy (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) : 
  (5 : ℚ) / 6 * total_people = seated_people →
  (5 : ℚ) / 6 * total_chairs = seated_people →
  total_chairs - seated_people = 10 →
  total_people = 60 := by
sorry

end room_occupancy_l345_34550


namespace joe_fruit_probability_l345_34503

/-- The number of fruit options Joe has -/
def num_fruits : ℕ := 4

/-- The number of meals Joe has in a day -/
def num_meals : ℕ := 3

/-- The probability of choosing any specific fruit for a meal -/
def prob_single_fruit : ℚ := 1 / num_fruits

/-- The probability of eating the same fruit for all meals -/
def prob_same_fruit : ℚ := num_fruits * prob_single_fruit ^ num_meals

/-- The probability of eating at least two different kinds of fruit in a day -/
def prob_different_fruits : ℚ := 1 - prob_same_fruit

theorem joe_fruit_probability :
  prob_different_fruits = 15 / 16 :=
sorry

end joe_fruit_probability_l345_34503


namespace fifth_term_is_89_l345_34544

def sequence_rule (seq : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → seq (n + 1) = (seq n + seq (n + 2)) / 3

theorem fifth_term_is_89 (seq : ℕ → ℕ) (h_rule : sequence_rule seq) 
  (h_first : seq 1 = 2) (h_fourth : seq 4 = 34) : seq 5 = 89 := by
  sorry

end fifth_term_is_89_l345_34544


namespace point_on_x_axis_with_distance_3_from_y_axis_l345_34510

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance function from a point to the y-axis
def distanceToYAxis (p : Point2D) : ℝ := |p.x|

-- State the theorem
theorem point_on_x_axis_with_distance_3_from_y_axis 
  (P : Point2D) 
  (h1 : P.y = 0)  -- P is on the x-axis
  (h2 : distanceToYAxis P = 3) : 
  (P.x = 3 ∧ P.y = 0) ∨ (P.x = -3 ∧ P.y = 0) := by
  sorry


end point_on_x_axis_with_distance_3_from_y_axis_l345_34510


namespace exponent_equality_l345_34583

theorem exponent_equality (a b c d : ℝ) (x y z q : ℝ) 
  (h1 : a^(2*x) = c^(3*q)) 
  (h2 : a^(2*x) = b) 
  (h3 : c^(4*y) = a^(5*z)) 
  (h4 : c^(4*y) = d) : 
  2*x * 5*z = 3*q * 4*y := by
sorry

end exponent_equality_l345_34583


namespace x_value_l345_34521

theorem x_value (x y : ℝ) (h1 : 2 * x - y = 14) (h2 : y = 2) : x = 8 := by
  sorry

end x_value_l345_34521


namespace range_of_f_l345_34540

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 1)

theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, x^2 ≠ 1 ∧ f x = y} = {y : ℝ | y < 0 ∨ y > 0} := by
  sorry

end range_of_f_l345_34540


namespace circle_center_coordinate_sum_l345_34572

/-- Given a circle with equation x^2 + y^2 - 6x + 14y = -28, 
    the sum of the x-coordinate and y-coordinate of its center is -4 -/
theorem circle_center_coordinate_sum : 
  ∃ (h k : ℝ), (∀ (x y : ℝ), x^2 + y^2 - 6*x + 14*y = -28 ↔ (x - h)^2 + (y - k)^2 = 30) ∧ h + k = -4 := by
  sorry

end circle_center_coordinate_sum_l345_34572


namespace green_balls_removal_l345_34591

theorem green_balls_removal (total : ℕ) (initial_green_percentage : ℚ) 
  (final_green_percentage : ℚ) (removed : ℕ) : 
  total = 600 →
  initial_green_percentage = 7/10 →
  final_green_percentage = 3/5 →
  removed = 150 →
  (initial_green_percentage * total - removed) / (total - removed) = final_green_percentage := by
sorry

end green_balls_removal_l345_34591


namespace correct_num_dogs_l345_34530

/-- Represents the number of dogs Carly worked on --/
def num_dogs : ℕ := 11

/-- Represents the total number of nails trimmed --/
def total_nails : ℕ := 164

/-- Represents the number of dogs with three legs --/
def three_legged_dogs : ℕ := 3

/-- Represents the number of dogs with three nails on one paw --/
def three_nailed_dogs : ℕ := 2

/-- Represents the number of dogs with an extra nail on one paw --/
def extra_nailed_dogs : ℕ := 1

/-- Represents the number of nails on a regular dog --/
def nails_per_regular_dog : ℕ := 4 * 4

/-- Theorem stating that the number of dogs is correct given the conditions --/
theorem correct_num_dogs :
  num_dogs * nails_per_regular_dog
  - three_legged_dogs * 4
  - three_nailed_dogs
  + extra_nailed_dogs
  = total_nails :=
by sorry

end correct_num_dogs_l345_34530


namespace vector_problem_l345_34570

/-- Define a 2D vector -/
def Vector2D := ℝ × ℝ

/-- Check if two vectors are collinear -/
def collinear (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Dot product of two vectors -/
def dot_product (v w : Vector2D) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Check if two vectors are perpendicular -/
def perpendicular (v w : Vector2D) : Prop :=
  dot_product v w = 0

/-- The main theorem -/
theorem vector_problem :
  ∀ (m : ℝ),
  let a : Vector2D := (2, 1)
  let b : Vector2D := (3, -1)
  let c : Vector2D := (3, m)
  (collinear a c → m = 3/2) ∧
  (perpendicular (a.1 - 2*b.1, a.2 - 2*b.2) c → m = 4) :=
sorry

end vector_problem_l345_34570


namespace smallest_odd_polygon_is_seven_l345_34524

/-- A polygon with an odd number of sides that can be divided into parallelograms -/
structure OddPolygon where
  sides : ℕ
  is_odd : Odd sides
  divisible_into_parallelograms : Bool

/-- The smallest number of sides for an OddPolygon -/
def smallest_odd_polygon_sides : ℕ := 7

/-- Theorem stating that the smallest number of sides for an OddPolygon is 7 -/
theorem smallest_odd_polygon_is_seven :
  ∀ (p : OddPolygon), p.divisible_into_parallelograms → p.sides ≥ smallest_odd_polygon_sides :=
by sorry

end smallest_odd_polygon_is_seven_l345_34524


namespace book_profit_rate_l345_34553

/-- Calculate the rate of profit given cost price and selling price -/
def rate_of_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem: The rate of profit for a book bought at Rs 50 and sold at Rs 90 is 80% -/
theorem book_profit_rate : rate_of_profit 50 90 = 80 := by
  sorry

end book_profit_rate_l345_34553


namespace functional_equation_problem_l345_34526

theorem functional_equation_problem (f g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + g y) = -x + y + 1) :
  ∀ x y : ℝ, g (x + f y) = -x + y - 1 := by
sorry

end functional_equation_problem_l345_34526


namespace hyperbola_asymptote_l345_34525

/-- Given a hyperbola with equation x²/a² - y²/4 = 1 where a > 0,
    if one of its asymptote equations is y = -2x, then a = 1 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) : 
  (∃ x y : ℝ, x^2/a^2 - y^2/4 = 1 ∧ y = -2*x) → a = 1 := by
  sorry

end hyperbola_asymptote_l345_34525


namespace valid_allocations_count_l345_34548

/-- The number of male volunteers -/
def num_males : ℕ := 4

/-- The number of female volunteers -/
def num_females : ℕ := 3

/-- The total number of volunteers -/
def total_volunteers : ℕ := num_males + num_females

/-- The maximum number of people allowed in a group -/
def max_group_size : ℕ := 5

/-- A function to calculate the number of valid allocation plans -/
def count_valid_allocations : ℕ :=
  let three_four_split := (Nat.choose total_volunteers 3 - 1) * 2
  let two_five_split := (Nat.choose total_volunteers 2 - Nat.choose num_females 2) * 2
  three_four_split + two_five_split

/-- Theorem stating that the number of valid allocation plans is 104 -/
theorem valid_allocations_count : count_valid_allocations = 104 := by
  sorry


end valid_allocations_count_l345_34548


namespace area_perimeter_ratio_equal_l345_34569

/-- An isosceles trapezoid inscribed in a circle -/
structure InscribedIsoscelesTrapezoid where
  /-- Radius of the circle -/
  R : ℝ
  /-- Perimeter of the trapezoid -/
  P : ℝ
  /-- Radius is positive -/
  R_pos : R > 0
  /-- Perimeter is positive -/
  P_pos : P > 0

/-- Theorem: The ratio of the area of the trapezoid to the area of the circle
    is equal to the ratio of the perimeter of the trapezoid to the circumference of the circle -/
theorem area_perimeter_ratio_equal
  (trap : InscribedIsoscelesTrapezoid) :
  (trap.P * trap.R / 2) / (Real.pi * trap.R^2) = trap.P / (2 * Real.pi * trap.R) :=
sorry

end area_perimeter_ratio_equal_l345_34569


namespace inequality_condition_l345_34552

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 + x) = f (2 - x)) ∧
  (∀ x y, x < y → x ≤ 2 → y ≤ 2 → f x < f y)

/-- The main theorem -/
theorem inequality_condition (f : ℝ → ℝ) (a : ℝ) 
  (h : special_function f) : 
  f (a^2 + 3*a + 2) < f (a^2 - a + 2) ↔ a > -1 ∧ a ≠ 0 := by
  sorry

end inequality_condition_l345_34552


namespace max_value_3x_4y_l345_34517

theorem max_value_3x_4y (x y : ℝ) : 
  y^2 = (1 - x) * (1 + x) → 
  ∃ (M : ℝ), M = 5 ∧ ∀ (x' y' : ℝ), y'^2 = (1 - x') * (1 + x') → 3*x' + 4*y' ≤ M :=
sorry

end max_value_3x_4y_l345_34517


namespace people_speaking_neither_language_l345_34528

theorem people_speaking_neither_language (total : ℕ) (latin : ℕ) (french : ℕ) (both : ℕ) 
  (h_total : total = 25)
  (h_latin : latin = 13)
  (h_french : french = 15)
  (h_both : both = 9)
  : total - (latin + french - both) = 6 := by
  sorry

end people_speaking_neither_language_l345_34528


namespace stratified_sampling_problem_l345_34558

theorem stratified_sampling_problem (total_population : ℕ) (first_stratum : ℕ) (sample_first_stratum : ℕ) (total_sample : ℕ) :
  total_population = 1500 →
  first_stratum = 700 →
  sample_first_stratum = 14 →
  (sample_first_stratum : ℚ) / total_sample = (first_stratum : ℚ) / total_population →
  total_sample = 30 :=
by
  sorry

#check stratified_sampling_problem

end stratified_sampling_problem_l345_34558


namespace modular_arithmetic_expression_l345_34594

theorem modular_arithmetic_expression : (240 * 15 - 33 * 8 + 6) % 18 = 12 := by
  sorry

end modular_arithmetic_expression_l345_34594


namespace sqrt_twelve_minus_sqrt_twentyseven_equals_negative_sqrt_three_l345_34512

theorem sqrt_twelve_minus_sqrt_twentyseven_equals_negative_sqrt_three :
  Real.sqrt 12 - Real.sqrt 27 = -Real.sqrt 3 := by
  sorry

end sqrt_twelve_minus_sqrt_twentyseven_equals_negative_sqrt_three_l345_34512


namespace tan_585_degrees_l345_34564

theorem tan_585_degrees : Real.tan (585 * π / 180) = 1 := by
  sorry

end tan_585_degrees_l345_34564


namespace rectangular_plot_area_l345_34532

/-- A rectangular plot with length thrice its width and width of 12 meters has an area of 432 square meters. -/
theorem rectangular_plot_area : 
  ∀ (width length area : ℝ),
  width = 12 →
  length = 3 * width →
  area = length * width →
  area = 432 := by
sorry

end rectangular_plot_area_l345_34532


namespace arithmetic_series_sum_l345_34549

theorem arithmetic_series_sum (A : ℕ) : A = 380 := by
  sorry

#check arithmetic_series_sum

end arithmetic_series_sum_l345_34549
