import Mathlib

namespace NUMINAMATH_CALUDE_moms_ice_cream_scoops_pierre_ice_cream_problem_l2952_295282

/-- Given the cost of ice cream scoops and the total bill, calculate the number of scoops Pierre's mom gets. -/
theorem moms_ice_cream_scoops (cost_per_scoop : ℕ) (pierres_scoops : ℕ) (total_bill : ℕ) : ℕ :=
  let moms_scoops := (total_bill - cost_per_scoop * pierres_scoops) / cost_per_scoop
  moms_scoops

/-- Prove that given the specific conditions, Pierre's mom gets 4 scoops of ice cream. -/
theorem pierre_ice_cream_problem :
  moms_ice_cream_scoops 2 3 14 = 4 := by
  sorry

end NUMINAMATH_CALUDE_moms_ice_cream_scoops_pierre_ice_cream_problem_l2952_295282


namespace NUMINAMATH_CALUDE_unique_four_digit_difference_l2952_295268

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  h1 : a < 10
  h2 : b < 10
  h3 : c < 10
  h4 : d < 10
  h5 : a > b
  h6 : b > c
  h7 : c > d

/-- Converts a FourDigitNumber to its decimal representation -/
def toDecimal (n : FourDigitNumber) : Nat :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- The difference between a number and its reverse -/
def difference (n : FourDigitNumber) : Int :=
  (toDecimal n) - (1000 * n.d + 100 * n.c + 10 * n.b + n.a)

theorem unique_four_digit_difference (n : FourDigitNumber) :
  0 < difference n ∧ difference n < 10000 → n.a = 7 ∧ n.b = 6 ∧ n.c = 4 ∧ n.d = 1 :=
by
  sorry

#eval toDecimal { a := 7, b := 6, c := 4, d := 1, h1 := by norm_num, h2 := by norm_num, h3 := by norm_num, h4 := by norm_num, h5 := by norm_num, h6 := by norm_num, h7 := by norm_num }

end NUMINAMATH_CALUDE_unique_four_digit_difference_l2952_295268


namespace NUMINAMATH_CALUDE_vector_calculation_l2952_295290

def a : Fin 2 → ℚ := ![1, 1]
def b : Fin 2 → ℚ := ![1, -1]

theorem vector_calculation : (1/2 : ℚ) • a - (3/2 : ℚ) • b = ![-1, 2] := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l2952_295290


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2952_295217

theorem sqrt_equation_solution (a b : ℕ+) (h1 : a < b) :
  Real.sqrt (1 + Real.sqrt (25 + 20 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b →
  a = 2 ∧ b = 8 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2952_295217


namespace NUMINAMATH_CALUDE_number_problem_l2952_295200

theorem number_problem (n p q : ℝ) 
  (h1 : n / p = 6)
  (h2 : n / q = 15)
  (h3 : p - q = 0.3) :
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l2952_295200


namespace NUMINAMATH_CALUDE_kitchen_floor_theorem_l2952_295240

/-- Calculates the area of the kitchen floor given the total mopping time,
    mopping rate, and bathroom floor area. -/
def kitchen_floor_area (total_time : ℕ) (mopping_rate : ℕ) (bathroom_area : ℕ) : ℕ :=
  total_time * mopping_rate - bathroom_area

/-- Proves that the kitchen floor area is 80 square feet given the specified conditions. -/
theorem kitchen_floor_theorem :
  kitchen_floor_area 13 8 24 = 80 := by
  sorry

#eval kitchen_floor_area 13 8 24

end NUMINAMATH_CALUDE_kitchen_floor_theorem_l2952_295240


namespace NUMINAMATH_CALUDE_school_fee_calculation_l2952_295298

def mother_money : ℚ :=
  2 * 100 + 1 * 50 + 5 * 20 + 3 * 10 + 4 * 5 + 6 * 0.25 + 10 * 0.1 + 5 * 0.05

def father_money : ℚ :=
  3 * 100 + 4 * 50 + 2 * 20 + 1 * 10 + 6 * 5 + 8 * 0.25 + 7 * 0.1 + 3 * 0.05

def school_fee : ℚ := mother_money + father_money

theorem school_fee_calculation : school_fee = 985.60 := by
  sorry

end NUMINAMATH_CALUDE_school_fee_calculation_l2952_295298


namespace NUMINAMATH_CALUDE_eight_percent_of_fifty_l2952_295284

theorem eight_percent_of_fifty : ∃ x : ℝ, x = 50 * 0.08 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_eight_percent_of_fifty_l2952_295284


namespace NUMINAMATH_CALUDE_x_intercepts_count_l2952_295238

theorem x_intercepts_count : 
  (⌊(100000 : ℝ) / Real.pi⌋ : ℤ) - (⌊(10000 : ℝ) / Real.pi⌋ : ℤ) = 28648 := by
  sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l2952_295238


namespace NUMINAMATH_CALUDE_john_vacation_expenses_l2952_295218

def octal_to_decimal (n : Nat) : Nat :=
  let digits := n.digits 8
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * (8 ^ i)) 0

theorem john_vacation_expenses :
  octal_to_decimal 5555 - 1500 = 1425 := by
  sorry

end NUMINAMATH_CALUDE_john_vacation_expenses_l2952_295218


namespace NUMINAMATH_CALUDE_value_of_a_l2952_295248

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x^2 + 14
def g (x : ℝ) : ℝ := x^3 - 4

-- State the theorem
theorem value_of_a (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 18) :
  a = (4 + 2 * Real.sqrt 3 / 3) ^ (1/3) := by
  sorry


end NUMINAMATH_CALUDE_value_of_a_l2952_295248


namespace NUMINAMATH_CALUDE_circle_equation_l2952_295228

/-- A circle C with points A and B, and a chord intercepted by a line --/
structure CircleWithPointsAndChord where
  -- Center of the circle
  center : ℝ × ℝ
  -- Radius of the circle
  radius : ℝ
  -- Point A on the circle
  pointA : ℝ × ℝ
  -- Point B on the circle
  pointB : ℝ × ℝ
  -- Length of the chord intercepted by the line x-y-2=0
  chordLength : ℝ
  -- Ensure A and B are on the circle
  h_pointA_on_circle : (pointA.1 - center.1)^2 + (pointA.2 - center.2)^2 = radius^2
  h_pointB_on_circle : (pointB.1 - center.1)^2 + (pointB.2 - center.2)^2 = radius^2
  -- Ensure the chord length is correct
  h_chord_length : chordLength = Real.sqrt 2

/-- The theorem stating that the circle satisfying the given conditions has the equation (x-1)² + y² = 1 --/
theorem circle_equation (c : CircleWithPointsAndChord) 
  (h_pointA : c.pointA = (1, 1)) 
  (h_pointB : c.pointB = (2, 0)) :
  c.center = (1, 0) ∧ c.radius = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l2952_295228


namespace NUMINAMATH_CALUDE_rational_equation_solution_l2952_295239

theorem rational_equation_solution : 
  ∃ (x : ℚ), (x + 11) / (x - 4) = (x - 3) / (x + 7) ∧ x = -13/5 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l2952_295239


namespace NUMINAMATH_CALUDE_club_average_age_l2952_295231

/-- Represents the average age of a group of people -/
def average_age (total_age : ℕ) (num_people : ℕ) : ℚ :=
  (total_age : ℚ) / (num_people : ℚ)

/-- Represents the total age of a group of people -/
def total_age (avg_age : ℕ) (num_people : ℕ) : ℕ :=
  avg_age * num_people

theorem club_average_age 
  (num_women : ℕ) (women_avg_age : ℕ) 
  (num_men : ℕ) (men_avg_age : ℕ) 
  (num_children : ℕ) (children_avg_age : ℕ) :
  num_women = 12 → 
  women_avg_age = 32 → 
  num_men = 18 → 
  men_avg_age = 36 → 
  num_children = 20 → 
  children_avg_age = 10 → 
  average_age 
    (total_age women_avg_age num_women + 
     total_age men_avg_age num_men + 
     total_age children_avg_age num_children)
    (num_women + num_men + num_children) = 24 := by
  sorry

end NUMINAMATH_CALUDE_club_average_age_l2952_295231


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l2952_295259

theorem factorization_of_difference_of_squares (a b : ℝ) :
  3 * a^2 - 3 * b^2 = 3 * (a + b) * (a - b) := by sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l2952_295259


namespace NUMINAMATH_CALUDE_time_for_order_l2952_295250

/-- Represents the time it takes to make one shirt -/
def shirt_time : ℝ := 1

/-- Represents the time it takes to make one pair of pants -/
def pants_time : ℝ := 2 * shirt_time

/-- Represents the time it takes to make one jacket -/
def jacket_time : ℝ := 3 * shirt_time

/-- The total time to make 2 shirts, 3 pairs of pants, and 4 jackets is 10 hours -/
axiom total_time_10 : 2 * shirt_time + 3 * pants_time + 4 * jacket_time = 10

/-- Theorem: It takes 20 working hours to make 14 shirts, 10 pairs of pants, and 2 jackets -/
theorem time_for_order : 14 * shirt_time + 10 * pants_time + 2 * jacket_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_time_for_order_l2952_295250


namespace NUMINAMATH_CALUDE_unique_solution_l2952_295260

/-- Floor function: greatest integer less than or equal to x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  3 * x + 5 * (floor x) - 2017 = 0

/-- The theorem stating the unique solution -/
theorem unique_solution :
  ∃! x : ℝ, equation x ∧ x = 252 + 1/3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l2952_295260


namespace NUMINAMATH_CALUDE_circle_line_segments_l2952_295280

/-- The number of line segments formed by joining each pair of n distinct points on a circle -/
def lineSegments (n : ℕ) : ℕ := n.choose 2

/-- There are 8 distinct points on a circle -/
def numPoints : ℕ := 8

theorem circle_line_segments :
  lineSegments numPoints = 28 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_segments_l2952_295280


namespace NUMINAMATH_CALUDE_sum_fraction_bounds_l2952_295245

theorem sum_fraction_bounds (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let S := (a / (a + b + d)) + (b / (b + c + a)) + (c / (c + d + b)) + (d / (d + a + c))
  1 < S ∧ S < 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_fraction_bounds_l2952_295245


namespace NUMINAMATH_CALUDE_exists_equal_area_split_line_l2952_295297

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the four circles
def circles : List Circle := [
  { center := (14, 92), radius := 5 },
  { center := (17, 76), radius := 5 },
  { center := (19, 84), radius := 5 },
  { center := (25, 90), radius := 5 }
]

-- Define a line passing through a point with a given slope
structure Line where
  point : ℝ × ℝ
  slope : ℝ

-- Function to calculate the area of a circle segment cut by a line
def circleSegmentArea (c : Circle) (l : Line) : ℝ := sorry

-- Function to calculate the total area of circle segments on one side of the line
def totalSegmentArea (cs : List Circle) (l : Line) : ℝ := sorry

-- Theorem statement
theorem exists_equal_area_split_line :
  ∃ m : ℝ, let l := { point := (17, 76), slope := m }
    totalSegmentArea circles l = (1/2) * (List.sum (circles.map (fun c => π * c.radius^2))) :=
sorry

end NUMINAMATH_CALUDE_exists_equal_area_split_line_l2952_295297


namespace NUMINAMATH_CALUDE_tenth_grade_enrollment_l2952_295202

/-- Represents the number of students enrolled only in science class -/
def students_only_science (total_students science_students art_students : ℕ) : ℕ :=
  science_students - (science_students + art_students - total_students)

/-- Theorem stating that given the conditions, 65 students are enrolled only in science class -/
theorem tenth_grade_enrollment (total_students science_students art_students : ℕ) 
  (h1 : total_students = 140)
  (h2 : science_students = 100)
  (h3 : art_students = 75) :
  students_only_science total_students science_students art_students = 65 := by
  sorry

#eval students_only_science 140 100 75

end NUMINAMATH_CALUDE_tenth_grade_enrollment_l2952_295202


namespace NUMINAMATH_CALUDE_count_ordered_pairs_l2952_295215

/-- The number of ordered pairs of positive integers (x,y) satisfying xy = 1944 -/
def num_ordered_pairs : ℕ := 24

/-- The prime factorization of 1944 -/
def prime_factorization_1944 : List (ℕ × ℕ) := [(2, 3), (3, 5)]

/-- Theorem stating that the number of ordered pairs (x,y) of positive integers
    satisfying xy = 1944 is equal to 24, given the prime factorization of 1944 -/
theorem count_ordered_pairs :
  (∀ (x y : ℕ), x * y = 1944 → x > 0 ∧ y > 0) →
  prime_factorization_1944 = [(2, 3), (3, 5)] →
  num_ordered_pairs = 24 := by
  sorry

#check count_ordered_pairs

end NUMINAMATH_CALUDE_count_ordered_pairs_l2952_295215


namespace NUMINAMATH_CALUDE_mean_of_numbers_l2952_295279

def numbers : List ℝ := [13, 8, 13, 21, 7, 23]

theorem mean_of_numbers : (numbers.sum / numbers.length : ℝ) = 14.1666667 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_numbers_l2952_295279


namespace NUMINAMATH_CALUDE_race_finish_positions_l2952_295291

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  position : ℝ

/-- Represents the state of the race -/
structure RaceState where
  a : Runner
  b : Runner
  c : Runner

/-- The race is 100 meters long -/
def race_length : ℝ := 100

theorem race_finish_positions (initial : RaceState) 
  (h1 : initial.a.position = race_length) 
  (h2 : initial.b.position = race_length - 5)
  (h3 : initial.c.position = race_length - 10)
  (h4 : ∀ r : Runner, r.speed > 0) :
  ∃ (final : RaceState), 
    final.b.position = race_length ∧ 
    final.c.position = race_length - (5 * 5 / 19) := by
  sorry

end NUMINAMATH_CALUDE_race_finish_positions_l2952_295291


namespace NUMINAMATH_CALUDE_monomial_count_l2952_295277

/-- A function that determines if an expression is a monomial -/
def isMonomial (expr : String) : Bool :=
  match expr with
  | "-1" => true
  | "-1/2*a^2" => true
  | "2/3*x^2*y" => true
  | "a*b^2/π" => true
  | "ab/c" => false
  | "3a-b" => false
  | "0" => true
  | "(x-1)/2" => false
  | _ => false

/-- The list of expressions to check -/
def expressions : List String :=
  ["-1", "-1/2*a^2", "2/3*x^2*y", "a*b^2/π", "ab/c", "3a-b", "0", "(x-1)/2"]

/-- Counts the number of monomials in the list of expressions -/
def countMonomials (exprs : List String) : Nat :=
  exprs.filter isMonomial |>.length

theorem monomial_count :
  countMonomials expressions = 5 := by sorry

end NUMINAMATH_CALUDE_monomial_count_l2952_295277


namespace NUMINAMATH_CALUDE_fraction_of_task_completed_l2952_295271

theorem fraction_of_task_completed (total_time minutes : ℕ) (h : total_time = 60) (h2 : minutes = 15) :
  (minutes : ℚ) / total_time = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_task_completed_l2952_295271


namespace NUMINAMATH_CALUDE_tan_half_product_l2952_295227

theorem tan_half_product (a b : ℝ) :
  7 * (Real.cos a + Real.cos b) + 2 * (Real.cos a * Real.cos b + 1) + 3 * Real.sin a * Real.sin b = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = -2 ∨ Real.tan (a / 2) * Real.tan (b / 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_product_l2952_295227


namespace NUMINAMATH_CALUDE_g_solution_set_m_range_l2952_295257

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x - 8
def g (x : ℝ) : ℝ := 2*x^2 - 4*x - 16

-- Theorem for the solution set of g(x) < 0
theorem g_solution_set :
  {x : ℝ | g x < 0} = {x : ℝ | -2 < x ∧ x < 4} := by sorry

-- Theorem for the range of m
theorem m_range (m : ℝ) :
  (∀ x > 2, f x ≥ (m + 2) * x - m - 15) ↔ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_g_solution_set_m_range_l2952_295257


namespace NUMINAMATH_CALUDE_fourth_person_height_l2952_295258

/-- Proves that given four people with heights in increasing order, where the difference
    between consecutive heights is 2, 2, and 6 inches respectively, and the average
    height is 77 inches, the height of the fourth person is 83 inches. -/
theorem fourth_person_height
  (h₁ h₂ h₃ h₄ : ℝ)
  (height_order : h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄)
  (diff_1_2 : h₂ - h₁ = 2)
  (diff_2_3 : h₃ - h₂ = 2)
  (diff_3_4 : h₄ - h₃ = 6)
  (avg_height : (h₁ + h₂ + h₃ + h₄) / 4 = 77) :
  h₄ = 83 := by
  sorry

end NUMINAMATH_CALUDE_fourth_person_height_l2952_295258


namespace NUMINAMATH_CALUDE_simplify_expression_l2952_295265

/-- For all real numbers z, (2-3z) - (3+4z) = -1-7z -/
theorem simplify_expression (z : ℝ) : (2 - 3*z) - (3 + 4*z) = -1 - 7*z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2952_295265


namespace NUMINAMATH_CALUDE_certain_number_proof_l2952_295249

theorem certain_number_proof : ∃ x : ℝ, (0.80 * x = 0.50 * 960) ∧ (x = 600) := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2952_295249


namespace NUMINAMATH_CALUDE_smallest_m_for_inequality_l2952_295242

theorem smallest_m_for_inequality : 
  ∃ (m : ℝ), (∀ (a b c : ℕ+), a + b + c = 1 → 
    m * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1) ∧ 
  (∀ (m' : ℝ), m' < m → 
    ∃ (a b c : ℕ+), a + b + c = 1 ∧ 
    m' * (a^3 + b^3 + c^3) < 6 * (a^2 + b^2 + c^2) + 1) ∧
  m = 27 := by
sorry

end NUMINAMATH_CALUDE_smallest_m_for_inequality_l2952_295242


namespace NUMINAMATH_CALUDE_red_cells_count_l2952_295209

/-- Represents the dimensions of the grid -/
structure GridDim where
  rows : Nat
  cols : Nat

/-- Represents the painter's movement -/
structure Movement where
  left : Nat
  down : Nat

/-- Calculates the number of distinct cells visited before returning to the start -/
def distinctCellsVisited (dim : GridDim) (move : Movement) : Nat :=
  Nat.lcm dim.rows dim.cols

/-- The main theorem stating the number of red cells on the grid -/
theorem red_cells_count (dim : GridDim) (move : Movement) 
  (h1 : dim.rows = 2000) 
  (h2 : dim.cols = 70) 
  (h3 : move.left = 1) 
  (h4 : move.down = 1) : 
  distinctCellsVisited dim move = 14000 := by
  sorry

#eval distinctCellsVisited ⟨2000, 70⟩ ⟨1, 1⟩

end NUMINAMATH_CALUDE_red_cells_count_l2952_295209


namespace NUMINAMATH_CALUDE_magnitude_2a_equals_6_l2952_295208

def a : Fin 3 → ℝ := ![-1, 2, 2]

theorem magnitude_2a_equals_6 : ‖(2 : ℝ) • a‖ = 6 := by sorry

end NUMINAMATH_CALUDE_magnitude_2a_equals_6_l2952_295208


namespace NUMINAMATH_CALUDE_covering_circles_highest_point_covered_l2952_295276

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The highest point of a circle -/
def highestPoint (c : Circle) : ℝ × ℝ :=
  (c.center.1, c.center.2 + c.radius)

/-- Check if a point is inside or on a circle -/
def isInside (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 ≤ c.radius^2

/-- A set of 101 unit circles where the first 100 cover the 101st -/
structure CoveringCircles where
  circles : Fin 101 → Circle
  all_unit : ∀ i, (circles i).radius = 1
  last_covered : ∀ p, isInside p (circles 100) → ∃ i < 100, isInside p (circles i)
  all_distinct : ∀ i j, i ≠ j → circles i ≠ circles j

theorem covering_circles_highest_point_covered (cc : CoveringCircles) :
  ∃ i j, i < 100 ∧ j < 100 ∧ i ≠ j ∧
    isInside (highestPoint (cc.circles j)) (cc.circles i) :=
  sorry

end NUMINAMATH_CALUDE_covering_circles_highest_point_covered_l2952_295276


namespace NUMINAMATH_CALUDE_cosine_sum_equality_l2952_295246

theorem cosine_sum_equality : 
  Real.cos (43 * π / 180) * Real.cos (77 * π / 180) - 
  Real.sin (43 * π / 180) * Real.sin (77 * π / 180) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_equality_l2952_295246


namespace NUMINAMATH_CALUDE_parabola_c_value_l2952_295269

/-- A parabola with equation y = 2x^2 + bx + c passing through (-2, 20) and (2, 28) has c = 16 -/
theorem parabola_c_value (b c : ℝ) : 
  (∀ x y : ℝ, y = 2 * x^2 + b * x + c → 
    ((x = -2 ∧ y = 20) ∨ (x = 2 ∧ y = 28))) → 
  c = 16 := by sorry

end NUMINAMATH_CALUDE_parabola_c_value_l2952_295269


namespace NUMINAMATH_CALUDE_probability_one_and_three_faces_l2952_295211

/-- Represents a cube with side length 5, assembled from unit cubes -/
def LargeCube := Fin 5 → Fin 5 → Fin 5 → Bool

/-- The number of unit cubes in the large cube -/
def totalUnitCubes : ℕ := 125

/-- The number of unit cubes with exactly one painted face -/
def oneRedFaceCubes : ℕ := 26

/-- The number of unit cubes with exactly three painted faces -/
def threeRedFaceCubes : ℕ := 4

/-- The probability of selecting one cube with one red face and one with three red faces -/
def probabilityOneAndThree : ℚ := 52 / 3875

theorem probability_one_and_three_faces (cube : LargeCube) :
  probabilityOneAndThree = (oneRedFaceCubes * threeRedFaceCubes : ℚ) / (totalUnitCubes.choose 2) :=
sorry

end NUMINAMATH_CALUDE_probability_one_and_three_faces_l2952_295211


namespace NUMINAMATH_CALUDE_factorial_difference_l2952_295256

theorem factorial_difference : Nat.factorial 9 - Nat.factorial 8 = 322560 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l2952_295256


namespace NUMINAMATH_CALUDE_peter_soda_purchase_l2952_295294

/-- The cost of soda per ounce in dollars -/
def soda_cost_per_ounce : ℚ := 25 / 100

/-- The amount Peter brought in dollars -/
def initial_amount : ℚ := 2

/-- The amount Peter left with in dollars -/
def remaining_amount : ℚ := 1 / 2

/-- The number of ounces of soda Peter bought -/
def soda_ounces : ℚ := (initial_amount - remaining_amount) / soda_cost_per_ounce

theorem peter_soda_purchase : soda_ounces = 6 := by
  sorry

end NUMINAMATH_CALUDE_peter_soda_purchase_l2952_295294


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2952_295267

theorem sum_of_xyz (x y z : ℕ+) (h : x + 2*x*y + 3*x*y*z = 115) : x + y + z = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2952_295267


namespace NUMINAMATH_CALUDE_six_solved_only_b_l2952_295252

/-- Represents the number of students who solved specific combinations of problems -/
structure ProblemSolvers where
  a : ℕ  -- only A
  b : ℕ  -- only B
  c : ℕ  -- only C
  ab : ℕ  -- A and B
  bc : ℕ  -- B and C
  ca : ℕ  -- C and A
  abc : ℕ  -- all three

/-- The conditions of the math competition problem -/
def competition_conditions (s : ProblemSolvers) : Prop :=
  -- Total number of students is 25
  s.a + s.b + s.c + s.ab + s.bc + s.ca + s.abc = 25 ∧
  -- Among students who didn't solve A, those who solved B is twice those who solved C
  s.b + s.bc = 2 * (s.c + s.bc) ∧
  -- Among students who solved A, those who solved only A is one more than those who solved A and others
  s.a = (s.ab + s.ca + s.abc) + 1 ∧
  -- Among students who solved only one problem, half didn't solve A
  s.a = s.b + s.c

/-- The theorem stating that 6 students solved only problem B -/
theorem six_solved_only_b :
  ∃ (s : ProblemSolvers), competition_conditions s ∧ s.b = 6 :=
sorry

end NUMINAMATH_CALUDE_six_solved_only_b_l2952_295252


namespace NUMINAMATH_CALUDE_expression_undefined_l2952_295221

theorem expression_undefined (θ : ℝ) (h1 : θ > 0) (h2 : θ + 90 = 180) : 
  ¬∃x : ℝ, x = (Real.sin θ + Real.sin (2*θ) + Real.sin (3*θ) + Real.sin (4*θ)) / 
            (Real.cos (θ/2) * Real.cos θ * Real.cos (2*θ)) := by
  sorry

end NUMINAMATH_CALUDE_expression_undefined_l2952_295221


namespace NUMINAMATH_CALUDE_sum_of_vectors_is_zero_l2952_295207

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given vectors a, b, c in a real vector space V, prove that their sum is zero
    under the given conditions. -/
theorem sum_of_vectors_is_zero (a b c : V)
  (not_collinear_ab : ¬ Collinear ℝ ({0, a, b} : Set V))
  (not_collinear_bc : ¬ Collinear ℝ ({0, b, c} : Set V))
  (not_collinear_ca : ¬ Collinear ℝ ({0, c, a} : Set V))
  (collinear_ab_c : Collinear ℝ ({0, a + b, c} : Set V))
  (collinear_bc_a : Collinear ℝ ({0, b + c, a} : Set V)) :
  a + b + c = (0 : V) := by
sorry

end NUMINAMATH_CALUDE_sum_of_vectors_is_zero_l2952_295207


namespace NUMINAMATH_CALUDE_orange_juice_percentage_l2952_295244

def pear_juice_yield : ℚ := 10 / 2
def orange_juice_yield : ℚ := 6 / 3
def pears_used : ℕ := 4
def oranges_used : ℕ := 6

theorem orange_juice_percentage : 
  (oranges_used * orange_juice_yield) / ((pears_used * pear_juice_yield) + (oranges_used * orange_juice_yield)) = 375 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_orange_juice_percentage_l2952_295244


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2952_295232

/-- Quadratic function passing through (2,3) -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 2) * x + 3

theorem quadratic_function_properties :
  ∃ a : ℝ,
  (f a 2 = 3) ∧
  (∀ x : ℝ, 0 < x → x < 3 → 2 ≤ f 0 x ∧ f 0 x < 6) ∧
  (∀ m y₁ y₂ : ℝ, f 0 (m - 1) = y₁ → f 0 m = y₂ → y₁ > y₂ → m < 3/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2952_295232


namespace NUMINAMATH_CALUDE_max_sum_under_constraints_l2952_295201

theorem max_sum_under_constraints :
  ∃ (M : ℝ), M = 32/17 ∧
  (∀ x y : ℝ, 5*x + 3*y ≤ 9 → 3*x + 5*y ≤ 11 → x + y ≤ M) ∧
  (∃ x y : ℝ, 5*x + 3*y ≤ 9 ∧ 3*x + 5*y ≤ 11 ∧ x + y = M) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_under_constraints_l2952_295201


namespace NUMINAMATH_CALUDE_sum_of_cubes_over_product_l2952_295270

theorem sum_of_cubes_over_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hsum : x + y + z = 0) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_over_product_l2952_295270


namespace NUMINAMATH_CALUDE_kylie_coins_to_laura_l2952_295233

/-- The number of coins Kylie collected from her piggy bank -/
def piggy_bank_coins : ℕ := 15

/-- The number of coins Kylie collected from her brother -/
def brother_coins : ℕ := 13

/-- The number of coins Kylie collected from her father -/
def father_coins : ℕ := 8

/-- The number of coins Kylie had left after giving some to Laura -/
def coins_left : ℕ := 15

/-- The total number of coins Kylie collected -/
def total_coins : ℕ := piggy_bank_coins + brother_coins + father_coins

/-- The number of coins Kylie gave to Laura -/
def coins_given_to_laura : ℕ := total_coins - coins_left

theorem kylie_coins_to_laura : coins_given_to_laura = 21 := by
  sorry

end NUMINAMATH_CALUDE_kylie_coins_to_laura_l2952_295233


namespace NUMINAMATH_CALUDE_tom_initial_investment_l2952_295296

/-- Represents the investment scenario of Tom and Jose's shop --/
structure ShopInvestment where
  tom_initial : ℕ  -- Tom's initial investment
  jose_investment : ℕ := 4500  -- Jose's investment
  total_profit : ℕ := 5400  -- Total profit after one year
  jose_profit : ℕ := 3000  -- Jose's share of the profit
  tom_months : ℕ := 12  -- Months Tom invested
  jose_months : ℕ := 10  -- Months Jose invested

/-- Theorem stating that Tom's initial investment was 3000 --/
theorem tom_initial_investment (shop : ShopInvestment) : shop.tom_initial = 3000 := by
  sorry

end NUMINAMATH_CALUDE_tom_initial_investment_l2952_295296


namespace NUMINAMATH_CALUDE_range_of_s_l2952_295262

/-- A decreasing function with central symmetry property -/
def DecreasingSymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x > f y) ∧
  (∀ x, f (x - 1) = -f (2 - x))

/-- The main theorem -/
theorem range_of_s (f : ℝ → ℝ) (h : DecreasingSymmetricFunction f) :
  ∀ s : ℝ, f (s^2 - 2*s) + f (2 - s) ≤ 0 → s ≤ 1 ∨ s ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_s_l2952_295262


namespace NUMINAMATH_CALUDE_greatest_integer_gcd_30_is_10_l2952_295293

theorem greatest_integer_gcd_30_is_10 : 
  ∃ n : ℕ, n < 100 ∧ Nat.gcd n 30 = 10 ∧ ∀ m : ℕ, m < 100 → Nat.gcd m 30 = 10 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_gcd_30_is_10_l2952_295293


namespace NUMINAMATH_CALUDE_ellipse_theorem_l2952_295222

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the major axis length -/
def majorAxisLength (a : ℝ) : Prop :=
  2 * a = 2 * Real.sqrt 2

/-- Definition of the range for point N's x-coordinate -/
def NxRange (x : ℝ) : Prop :=
  -1/4 < x ∧ x < 0

/-- Main theorem -/
theorem ellipse_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : majorAxisLength a) :
  (∀ x y : ℝ, ellipse x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  (∀ N A B : ℝ × ℝ,
    NxRange N.1 →
    (∃ k : ℝ, 
      ellipse A.1 A.2 (Real.sqrt 2) 1 ∧
      ellipse B.1 B.2 (Real.sqrt 2) 1 ∧
      A.2 = k * (A.1 + 1) ∧
      B.2 = k * (B.1 + 1)) →
    3 * Real.sqrt 2 / 2 < Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) < 2 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l2952_295222


namespace NUMINAMATH_CALUDE_complex_absolute_value_product_l2952_295281

theorem complex_absolute_value_product : Complex.abs (5 - 3 * Complex.I) * Complex.abs (5 + 3 * Complex.I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_product_l2952_295281


namespace NUMINAMATH_CALUDE_count_solutions_2x_3y_763_l2952_295299

theorem count_solutions_2x_3y_763 : 
  (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 + 3 * p.2 = 763 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 764) (Finset.range 764))).card = 127 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_2x_3y_763_l2952_295299


namespace NUMINAMATH_CALUDE_triangle_similarity_FC_length_l2952_295261

theorem triangle_similarity_FC_length
  (DC : ℝ) (CB : ℝ) (AD : ℝ) (AB : ℝ) (ED : ℝ) (FC : ℝ)
  (h1 : DC = 10)
  (h2 : CB = 5)
  (h3 : AB = (1/3) * AD)
  (h4 : ED = (4/5) * AD)
  : FC = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_FC_length_l2952_295261


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l2952_295214

theorem sum_of_cubes_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -4) : 
  a^3 + b^3 = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l2952_295214


namespace NUMINAMATH_CALUDE_correct_delivery_probability_l2952_295235

/-- The number of houses and packages -/
def n : ℕ := 5

/-- The number of correctly delivered packages -/
def k : ℕ := 3

/-- Probability of exactly k out of n packages being delivered to the correct houses -/
def prob_correct_delivery (n k : ℕ) : ℚ :=
  (Nat.choose n k * (Nat.factorial k) * (Nat.factorial (n - k) / Nat.factorial n)) / Nat.factorial n

theorem correct_delivery_probability :
  prob_correct_delivery n k = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_correct_delivery_probability_l2952_295235


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l2952_295285

theorem circle_diameter_from_area :
  ∀ (A r d : ℝ),
  A = 225 * Real.pi →
  A = Real.pi * r^2 →
  d = 2 * r →
  d = 30 := by sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l2952_295285


namespace NUMINAMATH_CALUDE_sugar_measurement_l2952_295292

theorem sugar_measurement (sugar_needed : ℚ) (cup_capacity : ℚ) : 
  sugar_needed = 5/2 ∧ cup_capacity = 1/4 → sugar_needed / cup_capacity = 10 := by
  sorry

end NUMINAMATH_CALUDE_sugar_measurement_l2952_295292


namespace NUMINAMATH_CALUDE_sum_of_possible_x_values_l2952_295236

/-- An isosceles triangle with two angles of 50° and x° --/
structure IsoscelesTriangle where
  x : ℝ
  is_isosceles : Bool
  has_50_degree_angle : Bool
  has_x_degree_angle : Bool

/-- The sum of angles in a triangle is 180° --/
axiom angle_sum (t : IsoscelesTriangle) : t.x + 50 + (180 - t.x - 50) = 180

/-- In an isosceles triangle, at least two angles are equal --/
axiom isosceles_equal_angles (t : IsoscelesTriangle) : t.is_isosceles → 
  (t.x = 50 ∨ t.x = (180 - 50) / 2 ∨ t.x = 180 - 2 * 50)

/-- The theorem to be proved --/
theorem sum_of_possible_x_values : 
  ∀ t : IsoscelesTriangle, t.is_isosceles ∧ t.has_50_degree_angle ∧ t.has_x_degree_angle → 
    50 + (180 - 50) / 2 + (180 - 2 * 50) = 195 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_x_values_l2952_295236


namespace NUMINAMATH_CALUDE_rectangle_square_area_ratio_l2952_295266

/-- Given a square S and a rectangle R, where the longer side of R is 20% more than
    the side of S, the shorter side of R is 20% less than the side of S, and the
    diagonal of R is 10% longer than the diagonal of S, prove that the ratio of
    the area of R to the area of S is 24/25. -/
theorem rectangle_square_area_ratio 
  (S : Real) -- Side length of square S
  (R_long : Real) -- Longer side of rectangle R
  (R_short : Real) -- Shorter side of rectangle R
  (R_diag : Real) -- Diagonal of rectangle R
  (h1 : R_long = 1.2 * S) -- Longer side of R is 20% more than side of S
  (h2 : R_short = 0.8 * S) -- Shorter side of R is 20% less than side of S
  (h3 : R_diag = 1.1 * S * Real.sqrt 2) -- Diagonal of R is 10% longer than diagonal of S
  : (R_long * R_short) / (S * S) = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_square_area_ratio_l2952_295266


namespace NUMINAMATH_CALUDE_competition_selection_count_l2952_295286

def male_count : ℕ := 5
def female_count : ℕ := 3
def selection_size : ℕ := 3

def selection_count : ℕ := 45

theorem competition_selection_count :
  (Nat.choose female_count 2 * Nat.choose male_count 1) +
  (Nat.choose female_count 1 * Nat.choose male_count 2) = selection_count :=
by sorry

end NUMINAMATH_CALUDE_competition_selection_count_l2952_295286


namespace NUMINAMATH_CALUDE_unique_congruent_integer_l2952_295223

theorem unique_congruent_integer : ∃! n : ℤ, 5 ≤ n ∧ n ≤ 10 ∧ n ≡ 12345 [ZMOD 6] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruent_integer_l2952_295223


namespace NUMINAMATH_CALUDE_maggie_spent_170_l2952_295226

/-- The total amount Maggie spent on books and magazines -/
def total_spent (num_books num_magazines book_price magazine_price : ℕ) : ℕ :=
  num_books * book_price + num_magazines * magazine_price

/-- Theorem stating that Maggie spent $170 in total -/
theorem maggie_spent_170 :
  total_spent 10 10 15 2 = 170 := by
  sorry

end NUMINAMATH_CALUDE_maggie_spent_170_l2952_295226


namespace NUMINAMATH_CALUDE_birds_remaining_proof_l2952_295247

/-- Calculates the number of birds remaining on a fence after some fly away. -/
def birdsRemaining (initialBirds flownAway : ℝ) : ℝ :=
  initialBirds - flownAway

/-- Theorem stating that the number of birds remaining is the difference between
    the initial number and the number that flew away. -/
theorem birds_remaining_proof (initialBirds flownAway : ℝ) :
  birdsRemaining initialBirds flownAway = initialBirds - flownAway := by
  sorry

/-- Example calculation for the specific problem -/
example : birdsRemaining 15.3 6.5 = 8.8 := by
  sorry

end NUMINAMATH_CALUDE_birds_remaining_proof_l2952_295247


namespace NUMINAMATH_CALUDE_no_four_consecutive_integers_product_perfect_square_l2952_295225

theorem no_four_consecutive_integers_product_perfect_square :
  ∀ x : ℕ+, ∃ y : ℕ+, x * (x + 1) * (x + 2) * (x + 3) = y^2 → False :=
by sorry

end NUMINAMATH_CALUDE_no_four_consecutive_integers_product_perfect_square_l2952_295225


namespace NUMINAMATH_CALUDE_fraction_sum_integer_implies_fractions_integer_l2952_295243

theorem fraction_sum_integer_implies_fractions_integer 
  (x y : ℕ+) 
  (h : ∃ (k : ℤ), (x.val^2 - 1 : ℤ) / (y.val + 1) + (y.val^2 - 1 : ℤ) / (x.val + 1) = k) :
  (∃ (m : ℤ), (x.val^2 - 1 : ℤ) / (y.val + 1) = m) ∧ 
  (∃ (n : ℤ), (y.val^2 - 1 : ℤ) / (x.val + 1) = n) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_integer_implies_fractions_integer_l2952_295243


namespace NUMINAMATH_CALUDE_vector_operations_and_parallelism_l2952_295210

/-- Given three vectors in R², prove the results of vector operations and parallelism condition. -/
theorem vector_operations_and_parallelism 
  (a b c : ℝ × ℝ) 
  (ha : a = (3, 2)) 
  (hb : b = (-1, 2)) 
  (hc : c = (4, 1)) : 
  (3 • a + b - 2 • c = (0, 6)) ∧ 
  (∃ k : ℝ, k = -16/13 ∧ ∃ t : ℝ, t • (a + k • c) = 2 • b - a) := by
sorry


end NUMINAMATH_CALUDE_vector_operations_and_parallelism_l2952_295210


namespace NUMINAMATH_CALUDE_mrs_hilt_pencil_purchase_l2952_295263

/-- Given Mrs. Hilt's purchases at the school store, prove the number of pencils she bought. -/
theorem mrs_hilt_pencil_purchase
  (total_spent : ℕ)
  (notebook_cost : ℕ)
  (ruler_cost : ℕ)
  (pencil_cost : ℕ)
  (h1 : total_spent = 74)
  (h2 : notebook_cost = 35)
  (h3 : ruler_cost = 18)
  (h4 : pencil_cost = 7)
  : (total_spent - notebook_cost - ruler_cost) / pencil_cost = 3 :=
by sorry

end NUMINAMATH_CALUDE_mrs_hilt_pencil_purchase_l2952_295263


namespace NUMINAMATH_CALUDE_smallest_integer_half_square_third_cube_l2952_295224

theorem smallest_integer_half_square_third_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (a : ℕ), n / 2 = a * a) ∧ 
  (∃ (b : ℕ), n / 3 = b * b * b) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), m / 2 = x * x) → 
    (∃ (y : ℕ), m / 3 = y * y * y) → 
    m ≥ n) ∧
  n = 648 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_half_square_third_cube_l2952_295224


namespace NUMINAMATH_CALUDE_yarn_length_difference_l2952_295229

theorem yarn_length_difference (green_length red_length : ℝ) : 
  green_length = 156 →
  red_length > 3 * green_length →
  green_length + red_length = 632 →
  red_length - 3 * green_length = 8 := by
sorry

end NUMINAMATH_CALUDE_yarn_length_difference_l2952_295229


namespace NUMINAMATH_CALUDE_quadratic_roots_arithmetic_sequence_l2952_295254

theorem quadratic_roots_arithmetic_sequence (a b : ℚ) : 
  a ≠ b →
  (∃ x₁ x₂ x₃ x₄ : ℚ, 
    (x₁^2 - x₁ + a = 0 ∧ x₂^2 - x₂ + a = 0) ∧ 
    (x₃^2 - x₃ + b = 0 ∧ x₄^2 - x₄ + b = 0) ∧
    (∃ d : ℚ, x₁ = 1/4 ∧ x₂ = x₁ + d ∧ x₃ = x₂ + d ∧ x₄ = x₃ + d)) →
  a + b = 31/72 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_arithmetic_sequence_l2952_295254


namespace NUMINAMATH_CALUDE_percentage_of_number_seventy_six_point_five_percent_of_1287_l2952_295219

theorem percentage_of_number (x : ℝ) (y : ℝ) (z : ℝ) (h : z = x * (y / 100)) :
  z = x * (y / 100) := by
  sorry

theorem seventy_six_point_five_percent_of_1287 :
  (76.5 / 100) * 1287 = 984.495 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_number_seventy_six_point_five_percent_of_1287_l2952_295219


namespace NUMINAMATH_CALUDE_mothers_day_discount_l2952_295275

theorem mothers_day_discount (original_price : ℝ) (final_price : ℝ) 
  (additional_discount : ℝ) (h1 : original_price = 125) 
  (h2 : final_price = 108) (h3 : additional_discount = 0.04) : 
  ∃ (initial_discount : ℝ), 
    final_price = (1 - additional_discount) * (original_price * (1 - initial_discount)) ∧ 
    initial_discount = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_mothers_day_discount_l2952_295275


namespace NUMINAMATH_CALUDE_regions_for_twenty_points_l2952_295295

/-- The number of regions created by chords in a circle --/
def num_regions (n : ℕ) : ℕ :=
  let vertices := n + (n.choose 4)
  let edges := (n * (n - 1) + 2 * (n.choose 4)) / 2
  edges - vertices + 1

/-- Theorem stating the number of regions for 20 points --/
theorem regions_for_twenty_points :
  num_regions 20 = 5036 := by
  sorry

end NUMINAMATH_CALUDE_regions_for_twenty_points_l2952_295295


namespace NUMINAMATH_CALUDE_same_color_probability_l2952_295205

/-- The probability of drawing two balls of the same color from a bag containing green and white balls. -/
theorem same_color_probability (green white : ℕ) (h : green = 9 ∧ white = 8) :
  let total := green + white
  let p_green := green * (green - 1) / (total * (total - 1))
  let p_white := white * (white - 1) / (total * (total - 1))
  p_green + p_white = 8 / 17 := by
  sorry

#check same_color_probability

end NUMINAMATH_CALUDE_same_color_probability_l2952_295205


namespace NUMINAMATH_CALUDE_equation_proof_l2952_295216

theorem equation_proof : (8 - 2) + 5 * (3 - 2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l2952_295216


namespace NUMINAMATH_CALUDE_range_of_quadratic_expression_l2952_295203

theorem range_of_quadratic_expression (x : ℝ) :
  ((x - 1) * (x - 2) < 2) →
  ∃ y, y = (x + 1) * (x - 3) ∧ -4 ≤ y ∧ y < 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_quadratic_expression_l2952_295203


namespace NUMINAMATH_CALUDE_opposite_sign_sum_l2952_295253

theorem opposite_sign_sum (x y : ℝ) :
  (|x + 2| + |y - 4| = 0) → (x + y - 3 = -1) := by
  sorry

end NUMINAMATH_CALUDE_opposite_sign_sum_l2952_295253


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2952_295241

theorem quadratic_inequality (y : ℝ) : y^2 - 9*y + 14 ≤ 0 ↔ 2 ≤ y ∧ y ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2952_295241


namespace NUMINAMATH_CALUDE_charles_journey_l2952_295264

/-- Represents the distance traveled by Charles -/
def total_distance : ℝ := 1800

/-- Represents the speed for the first half of the journey -/
def speed1 : ℝ := 90

/-- Represents the speed for the second half of the journey -/
def speed2 : ℝ := 180

/-- Represents the total time of the journey -/
def total_time : ℝ := 30

/-- Theorem stating that given the conditions of Charles' journey, the total distance is 1800 miles -/
theorem charles_journey :
  (total_distance / 2 / speed1 + total_distance / 2 / speed2 = total_time) →
  total_distance = 1800 :=
by sorry

end NUMINAMATH_CALUDE_charles_journey_l2952_295264


namespace NUMINAMATH_CALUDE_half_percent_to_decimal_l2952_295230

theorem half_percent_to_decimal : (1 / 2 : ℚ) / 100 = (0.005 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_half_percent_to_decimal_l2952_295230


namespace NUMINAMATH_CALUDE_class_size_l2952_295237

theorem class_size (poor_vision_percentage : ℝ) (glasses_percentage : ℝ) (glasses_count : ℕ) :
  poor_vision_percentage = 0.4 →
  glasses_percentage = 0.7 →
  glasses_count = 21 →
  ∃ total_students : ℕ, 
    (poor_vision_percentage * glasses_percentage * total_students : ℝ) = glasses_count ∧
    total_students = 75 :=
by sorry

end NUMINAMATH_CALUDE_class_size_l2952_295237


namespace NUMINAMATH_CALUDE_tommys_pencils_l2952_295212

/-- Represents the contents of Tommy's pencil case -/
structure PencilCase where
  total_items : ℕ
  num_pencils : ℕ
  num_pens : ℕ
  num_erasers : ℕ

/-- Theorem stating the number of pencils in Tommy's pencil case -/
theorem tommys_pencils (pc : PencilCase) 
  (h1 : pc.total_items = 13)
  (h2 : pc.num_pens = 2 * pc.num_pencils)
  (h3 : pc.num_erasers = 1)
  (h4 : pc.total_items = pc.num_pencils + pc.num_pens + pc.num_erasers) :
  pc.num_pencils = 4 := by
  sorry

end NUMINAMATH_CALUDE_tommys_pencils_l2952_295212


namespace NUMINAMATH_CALUDE_simplify_trig_expression_simplify_trig_product_l2952_295278

-- Part 1
theorem simplify_trig_expression (α : Real) :
  (Real.sin (α - π/2) + Real.cos (3*π/2 + α)) / (Real.sin (π - α) + Real.cos (3*π + α)) =
  1 / 0 := by sorry

-- Part 2
theorem simplify_trig_product :
  Real.sin (40 * π/180) * (Real.tan (10 * π/180) - Real.sqrt 3) =
  -Real.sin (80 * π/180) / Real.cos (10 * π/180) := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_simplify_trig_product_l2952_295278


namespace NUMINAMATH_CALUDE_following_pierre_better_than_guessing_l2952_295255

-- Define the probability of Pierre giving correct information
def pierre_correct_prob : ℚ := 3/4

-- Define the probability of Pierre giving incorrect information
def pierre_incorrect_prob : ℚ := 1/4

-- Define the probability of Jean guessing correctly for one event
def jean_guess_prob : ℚ := 1/2

-- Define the probability of Jean getting both dates correct when following Pierre's advice
def jean_correct_following_pierre : ℚ :=
  pierre_correct_prob * (pierre_correct_prob * pierre_correct_prob) +
  pierre_incorrect_prob * (pierre_incorrect_prob * pierre_incorrect_prob)

-- Define the probability of Jean getting both dates correct when guessing randomly
def jean_correct_guessing : ℚ := jean_guess_prob * jean_guess_prob

-- Theorem stating that following Pierre's advice is better than guessing randomly
theorem following_pierre_better_than_guessing :
  jean_correct_following_pierre > jean_correct_guessing :=
by sorry

end NUMINAMATH_CALUDE_following_pierre_better_than_guessing_l2952_295255


namespace NUMINAMATH_CALUDE_coin_collection_problem_l2952_295288

theorem coin_collection_problem :
  ∀ (n d q : ℕ),
    n + d + q = 30 →
    d = n + 4 →
    5 * n + 10 * d + 25 * q = 410 →
    q = n + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_coin_collection_problem_l2952_295288


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l2952_295204

/-- The inradius of a right triangle with side lengths 5, 12, and 13 is 2. -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 5 → b = 12 → c = 13 →
  a^2 + b^2 = c^2 →
  r = (a * b) / (2 * (a + b + c)) →
  r = 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l2952_295204


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l2952_295213

theorem polynomial_expansion_equality (x : ℝ) :
  (3*x^2 + 4*x + 8)*(x + 2) - (x + 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x + 2)*(x + 6) =
  6*x^3 + 20*x^2 + 6*x - 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l2952_295213


namespace NUMINAMATH_CALUDE_number_of_valid_choices_is_84_l2952_295273

/-- Represents a digit from 1 to 9 -/
def Digit := Fin 9

/-- The number of ways to choose three different digits a, b, c from 1 to 9 such that a < b < c -/
def NumberOfValidChoices : ℕ := sorry

/-- The theorem stating that the number of valid choices is 84 -/
theorem number_of_valid_choices_is_84 : NumberOfValidChoices = 84 := by sorry

end NUMINAMATH_CALUDE_number_of_valid_choices_is_84_l2952_295273


namespace NUMINAMATH_CALUDE_inequality_proof_l2952_295274

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a + b + c + d + 8 / (a * b + b * c + c * d + d * a) ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2952_295274


namespace NUMINAMATH_CALUDE_power_two_33_mod_9_l2952_295287

theorem power_two_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_two_33_mod_9_l2952_295287


namespace NUMINAMATH_CALUDE_covering_recurrence_l2952_295272

/-- Number of ways to cover a 2 × n rectangle with 1 × 2 pieces -/
def coveringWays : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => coveringWays (n + 1) + coveringWays n

/-- The recurrence relation for covering a 2 × n rectangle with 1 × 2 pieces -/
theorem covering_recurrence (n : ℕ) (h : n ≥ 2) :
  coveringWays n = coveringWays (n - 1) + coveringWays (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_covering_recurrence_l2952_295272


namespace NUMINAMATH_CALUDE_tan_equality_solution_l2952_295234

theorem tan_equality_solution (n : ℤ) (h1 : -180 < n) (h2 : n < 180) 
  (h3 : Real.tan (n * π / 180) = Real.tan (123 * π / 180)) : 
  n = 123 ∨ n = -57 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_solution_l2952_295234


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2952_295251

theorem hyperbola_eccentricity (a b e : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = Real.sqrt (2 * e - 1) * x) →
  e = 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2952_295251


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l2952_295206

theorem cubic_equation_solutions :
  ∀ (x y z n : ℕ), x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 →
  ((x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 3) ∨
   (x = 1 ∧ y = 2 ∧ z = 3 ∧ n = 1) ∨
   (x = 1 ∧ y = 3 ∧ z = 2 ∧ n = 1) ∨
   (x = 2 ∧ y = 1 ∧ z = 3 ∧ n = 1) ∨
   (x = 2 ∧ y = 3 ∧ z = 1 ∧ n = 1) ∨
   (x = 3 ∧ y = 1 ∧ z = 2 ∧ n = 1) ∨
   (x = 3 ∧ y = 2 ∧ z = 1 ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l2952_295206


namespace NUMINAMATH_CALUDE_largest_remainder_l2952_295283

theorem largest_remainder (A B : ℕ) : 
  (A / 13 = 33) → (A % 13 = B) → (∀ C : ℕ, (C / 13 = 33) → (C % 13 ≤ B)) → A = 441 :=
by sorry

end NUMINAMATH_CALUDE_largest_remainder_l2952_295283


namespace NUMINAMATH_CALUDE_E_parity_l2952_295289

def E : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => E (n + 1) + E n

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem E_parity : is_odd (E 2021) ∧ is_even (E 2022) ∧ is_odd (E 2023) := by sorry

end NUMINAMATH_CALUDE_E_parity_l2952_295289


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l2952_295220

/-- 
Given a quadratic equation x^2 + 8x + k = 0 with nonzero roots in the ratio 3:1,
prove that k = 12
-/
theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 8*x + k = 0 ∧ y^2 + 8*y + k = 0) → k = 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l2952_295220
