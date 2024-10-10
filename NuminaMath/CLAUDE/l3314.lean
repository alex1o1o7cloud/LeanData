import Mathlib

namespace new_average_weight_l3314_331465

theorem new_average_weight 
  (initial_students : ℕ) 
  (initial_average : ℝ) 
  (new_student_weight : ℝ) : 
  initial_students = 19 → 
  initial_average = 15 → 
  new_student_weight = 3 → 
  (initial_students * initial_average + new_student_weight) / (initial_students + 1) = 14.4 := by
  sorry

end new_average_weight_l3314_331465


namespace calculator_addition_correct_l3314_331462

/-- Represents a calculator button --/
inductive CalculatorButton
  | Digit (n : Nat)
  | Plus
  | Equals

/-- Represents a sequence of button presses on a calculator --/
def ButtonSequence := List CalculatorButton

/-- Evaluates a sequence of button presses and returns the result --/
def evaluate (seq : ButtonSequence) : Nat :=
  sorry

/-- The correct sequence of button presses to calculate 569 + 728 --/
def correctSequence : ButtonSequence :=
  [CalculatorButton.Digit 569, CalculatorButton.Plus, CalculatorButton.Digit 728, CalculatorButton.Equals]

theorem calculator_addition_correct :
  evaluate correctSequence = 569 + 728 :=
sorry

end calculator_addition_correct_l3314_331462


namespace pizza_combinations_l3314_331475

/-- The number of pizza toppings available. -/
def num_toppings : ℕ := 8

/-- The number of incompatible topping pairs. -/
def num_incompatible_pairs : ℕ := 1

/-- Calculates the number of combinations of n items taken k at a time. -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of possible one-topping and two-topping pizzas, given the number of toppings
    and the number of incompatible pairs. -/
def total_pizzas (n incompatible : ℕ) : ℕ :=
  n + combinations n 2 - incompatible

/-- Theorem stating that the total number of possible one-topping and two-topping pizzas
    is 35, given 8 toppings and 1 incompatible pair. -/
theorem pizza_combinations :
  total_pizzas num_toppings num_incompatible_pairs = 35 := by
  sorry

end pizza_combinations_l3314_331475


namespace volume_of_specific_cuboid_l3314_331419

/-- The volume of a cuboid with given edge lengths. -/
def cuboid_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a cuboid with edges 2 cm, 5 cm, and 3 cm is 30 cubic centimeters. -/
theorem volume_of_specific_cuboid : 
  cuboid_volume 2 5 3 = 30 := by
  sorry

end volume_of_specific_cuboid_l3314_331419


namespace garden_circle_diameter_l3314_331485

/-- Given a circular ground with a 2-metre broad garden around it,
    if the area of the garden is 226.19467105846502 square metres,
    then the diameter of the circular ground is 34 metres. -/
theorem garden_circle_diameter (r : ℝ) : 
  (π * ((r + 2)^2 - r^2) = 226.19467105846502) → 
  (2 * r = 34) := by
  sorry

end garden_circle_diameter_l3314_331485


namespace parallel_line_slope_l3314_331477

/-- Given a line with equation 5x - 3y = 9, prove that the slope of any parallel line is 5/3 -/
theorem parallel_line_slope (x y : ℝ) (h : 5 * x - 3 * y = 9) :
  ∃ (m : ℝ), m = 5 / 3 ∧ ∀ (x₁ y₁ : ℝ), (5 * x₁ - 3 * y₁ = 9) → (y₁ - y) = m * (x₁ - x) :=
sorry

end parallel_line_slope_l3314_331477


namespace f_positive_at_one_l3314_331414

def f (a : ℝ) (x : ℝ) : ℝ := -3 * x^2 + a * (6 - a) * x + 6

theorem f_positive_at_one (a : ℝ) :
  f a 1 > 0 ↔ 3 - 2 * Real.sqrt 3 < a ∧ a < 3 + 2 * Real.sqrt 3 :=
sorry

end f_positive_at_one_l3314_331414


namespace hexagon_area_l3314_331450

/-- Regular hexagon with vertices A at (0,0) and C at (10,2) -/
structure RegularHexagon where
  A : ℝ × ℝ
  C : ℝ × ℝ
  is_regular : Bool
  A_is_origin : A = (0, 0)
  C_coordinates : C = (10, 2)

/-- The area of a regular hexagon -/
def area (h : RegularHexagon) : ℝ := sorry

/-- Theorem stating that the area of the specified regular hexagon is 52√3 -/
theorem hexagon_area (h : RegularHexagon) : area h = 52 * Real.sqrt 3 := by sorry

end hexagon_area_l3314_331450


namespace basketball_spectators_l3314_331454

theorem basketball_spectators (total : ℕ) (children : ℕ) 
  (h1 : total = 10000)
  (h2 : children = 2500)
  (h3 : children = 5 * (total - children - (total - children - children) / 5)) :
  total - children - (total - children - children) / 5 = 7000 := by
  sorry

end basketball_spectators_l3314_331454


namespace cube_sum_divisible_by_nine_l3314_331438

theorem cube_sum_divisible_by_nine (n : ℕ+) :
  ∃ k : ℤ, (n : ℤ)^3 + (n + 1 : ℤ)^3 + (n + 2 : ℤ)^3 = 9 * k :=
by sorry

end cube_sum_divisible_by_nine_l3314_331438


namespace complex_expression_equality_l3314_331490

-- Define the complex number z
def z : ℂ := 1 + Complex.I

-- State the theorem
theorem complex_expression_equality : (2 / z) + z^2 = 1 + Complex.I := by
  sorry

end complex_expression_equality_l3314_331490


namespace mistaken_polynomial_calculation_l3314_331423

/-- Given a polynomial P such that P + (x^2 - 3x + 5) = 5x^2 - 2x + 4,
    prove that P = 4x^2 + x - 1 and P - (x^2 - 3x + 5) = 3x^2 + 4x - 6 -/
theorem mistaken_polynomial_calculation (P : ℝ → ℝ) 
  (h : ∀ x, P x + (x^2 - 3*x + 5) = 5*x^2 - 2*x + 4) : 
  (∀ x, P x = 4*x^2 + x - 1) ∧ 
  (∀ x, P x - (x^2 - 3*x + 5) = 3*x^2 + 4*x - 6) := by
  sorry

end mistaken_polynomial_calculation_l3314_331423


namespace exponent_multiplication_l3314_331480

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end exponent_multiplication_l3314_331480


namespace trip_time_is_ten_weeks_l3314_331428

/-- Calculates the total time spent on a trip visiting three countries -/
def totalTripTime (firstStay : ℕ) (otherStaysMultiplier : ℕ) : ℕ :=
  firstStay + 2 * otherStaysMultiplier * firstStay

/-- Proves that the total trip time is 10 weeks given the specified conditions -/
theorem trip_time_is_ten_weeks :
  totalTripTime 2 2 = 10 := by
  sorry

#eval totalTripTime 2 2

end trip_time_is_ten_weeks_l3314_331428


namespace f_six_of_two_l3314_331437

def f (x : ℝ) : ℝ := 3 * x - 1

def f_iter (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => f (f_iter n x)

theorem f_six_of_two : f_iter 6 2 = 1094 := by sorry

end f_six_of_two_l3314_331437


namespace imaginary_part_of_complex_fraction_l3314_331411

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (-3 + I) / (2 + I) → z.im = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l3314_331411


namespace gummies_cost_gummies_cost_proof_l3314_331407

theorem gummies_cost (lollipop_count : ℕ) (lollipop_price : ℚ) 
                      (gummies_count : ℕ) (initial_amount : ℚ) 
                      (remaining_amount : ℚ) : ℚ :=
  let total_spent := initial_amount - remaining_amount
  let lollipop_total := ↑lollipop_count * lollipop_price
  let gummies_total := total_spent - lollipop_total
  gummies_total / ↑gummies_count

#check gummies_cost 4 (3/2) 2 15 5 = 2

theorem gummies_cost_proof :
  gummies_cost 4 (3/2) 2 15 5 = 2 := by
  sorry

end gummies_cost_gummies_cost_proof_l3314_331407


namespace complex_cube_root_l3314_331497

theorem complex_cube_root : ∃ (z : ℂ), z^2 + 2 = 0 ∧ (z^3 = 2 * Real.sqrt 2 * I ∨ z^3 = -2 * Real.sqrt 2 * I) := by
  sorry

end complex_cube_root_l3314_331497


namespace quadratic_solution_l3314_331448

theorem quadratic_solution (x : ℝ) : x^2 - 4*x + 3 = 0 ∧ x ≥ 0 → x = 1 ∨ x = 3 := by
  sorry

end quadratic_solution_l3314_331448


namespace total_books_is_91_l3314_331403

/-- Calculates the total number of books sold over three days given the conditions -/
def total_books_sold (tuesday_sales : ℕ) : ℕ :=
  let wednesday_sales := 3 * tuesday_sales
  let thursday_sales := 3 * wednesday_sales
  tuesday_sales + wednesday_sales + thursday_sales

/-- Theorem stating that the total number of books sold over three days is 91 -/
theorem total_books_is_91 : total_books_sold 7 = 91 := by
  sorry

end total_books_is_91_l3314_331403


namespace nonagon_diagonal_count_l3314_331427

/-- The number of sides in a nonagon -/
def nonagon_sides : ℕ := 9

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonal_count : nonagon_diagonals = 27 := by
  sorry

end nonagon_diagonal_count_l3314_331427


namespace rhombus_perimeter_l3314_331463

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 16) (h2 : d2 = 30) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 68 := by
  sorry

end rhombus_perimeter_l3314_331463


namespace smallest_valid_number_proof_l3314_331435

/-- Checks if a natural number contains all digits from 0 to 9 --/
def containsAllDigits (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, n / 10^k % 10 = d

/-- The smallest 12-digit number divisible by 36 containing all digits --/
def smallestValidNumber : ℕ := 100023457896

theorem smallest_valid_number_proof :
  (smallestValidNumber ≥ 10^11) ∧ 
  (smallestValidNumber < 10^12) ∧
  (smallestValidNumber % 36 = 0) ∧
  containsAllDigits smallestValidNumber ∧
  ∀ m : ℕ, m ≥ 10^11 ∧ m < 10^12 ∧ m % 36 = 0 ∧ containsAllDigits m → m ≥ smallestValidNumber :=
by sorry

end smallest_valid_number_proof_l3314_331435


namespace intersection_M_N_l3314_331476

def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def N : Set ℝ := {-2, 0, 2}

theorem intersection_M_N : M ∩ N = {0, 2} := by
  sorry

end intersection_M_N_l3314_331476


namespace arithmetic_sequence_sum_divisibility_l3314_331471

theorem arithmetic_sequence_sum_divisibility :
  ∀ (a d : ℕ+), ∃ (k : ℕ+), (12 * a + 66 * d : ℕ) = 6 * k ∧
  ∀ (m : ℕ+), m < 6 → ∃ (a' d' : ℕ+), ¬(∃ (k' : ℕ+), (12 * a' + 66 * d' : ℕ) = m * k') :=
sorry

end arithmetic_sequence_sum_divisibility_l3314_331471


namespace min_value_quadratic_l3314_331481

theorem min_value_quadratic (x : ℝ) :
  ∃ (min_z : ℝ), ∀ (z : ℝ), z = 4 * x^2 + 8 * x + 16 → z ≥ min_z ∧ ∃ (x₀ : ℝ), 4 * x₀^2 + 8 * x₀ + 16 = min_z :=
by sorry

end min_value_quadratic_l3314_331481


namespace solve_for_a_l3314_331491

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- State the theorem
theorem solve_for_a (a : ℝ) (h1 : a > 1) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 ↔ |f a (2*x + a) - 2*f a x| ≤ 2) →
  a = 3 := by
sorry

end solve_for_a_l3314_331491


namespace complex_equation_solution_l3314_331482

theorem complex_equation_solution (z : ℂ) : (z - 2) * (1 + Complex.I) = 1 - Complex.I → z = 2 - Complex.I := by
  sorry

end complex_equation_solution_l3314_331482


namespace taller_tree_height_l3314_331441

theorem taller_tree_height (h_taller h_shorter : ℝ) : 
  h_taller - h_shorter = 18 →
  h_shorter / h_taller = 5 / 6 →
  h_taller = 108 := by
sorry

end taller_tree_height_l3314_331441


namespace max_boxes_of_paint_A_l3314_331426

/-- The maximum number of boxes of paint A that can be purchased given the conditions -/
theorem max_boxes_of_paint_A : ℕ :=
  let price_A : ℕ := 24  -- Price of paint A in yuan
  let price_B : ℕ := 16  -- Price of paint B in yuan
  let total_boxes : ℕ := 200  -- Total number of boxes to be purchased
  let max_cost : ℕ := 3920  -- Maximum total cost in yuan
  let max_A : ℕ := 90  -- Maximum number of boxes of paint A (to be proved)

  have h1 : price_A + 2 * price_B = 56 := by sorry
  have h2 : 2 * price_A + price_B = 64 := by sorry
  have h3 : ∀ m : ℕ, m ≤ total_boxes → 
    price_A * m + price_B * (total_boxes - m) ≤ max_cost → 
    m ≤ max_A := by sorry

  max_A

end max_boxes_of_paint_A_l3314_331426


namespace maximal_ratio_of_primes_l3314_331418

theorem maximal_ratio_of_primes (p q : ℕ) : 
  Prime p → Prime q → p > q → ¬(240 ∣ p^4 - q^4) → 
  (∃ (r : ℚ), r = q / p ∧ r ≤ 2/3 ∧ ∀ (s : ℚ), s = q / p → s ≤ r) :=
sorry

end maximal_ratio_of_primes_l3314_331418


namespace novel_reading_distribution_l3314_331473

/-- Represents the reading assignment for three friends -/
structure ReadingAssignment where
  total_pages : ℕ
  alice_speed : ℕ
  bob_speed : ℕ
  chandra_speed : ℕ
  alice_pages : ℕ
  bob_pages : ℕ
  chandra_pages : ℕ

/-- Theorem stating the correct distribution of pages for the given conditions -/
theorem novel_reading_distribution (assignment : ReadingAssignment) :
  assignment.total_pages = 912 ∧
  assignment.alice_speed = 40 ∧
  assignment.bob_speed = 60 ∧
  assignment.chandra_speed = 48 ∧
  assignment.chandra_pages = 420 →
  assignment.alice_pages = 295 ∧
  assignment.bob_pages = 197 ∧
  assignment.alice_pages + assignment.bob_pages + assignment.chandra_pages = assignment.total_pages :=
by sorry

end novel_reading_distribution_l3314_331473


namespace sophomore_count_l3314_331456

theorem sophomore_count (total : ℕ) (sophomore_percent : ℚ) (senior_percent : ℚ)
  (h_total : total = 50)
  (h_sophomore_percent : sophomore_percent = 1/5)
  (h_senior_percent : senior_percent = 1/4)
  (h_team_equal : ∃ (team_size : ℕ), 
    sophomore_percent * (total - seniors) = ↑team_size ∧
    senior_percent * seniors = ↑team_size)
  (seniors : ℕ) :
  total - seniors = 22 :=
sorry

end sophomore_count_l3314_331456


namespace intersection_of_M_and_N_l3314_331469

def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

theorem intersection_of_M_and_N :
  M ∩ N = {(3, -1)} := by
  sorry

end intersection_of_M_and_N_l3314_331469


namespace parallel_quadrilateral_coordinates_l3314_331483

/-- A quadrilateral with parallel sides and non-intersecting diagonals -/
structure ParallelQuadrilateral (a b c d : ℝ) :=
  (xC : ℝ)
  (xD : ℝ)
  (yC : ℝ)
  (side_AB : ℝ := a)
  (side_BC : ℝ := b)
  (side_CD : ℝ := c)
  (side_DA : ℝ := d)
  (parallel : yC = yC)  -- AB parallel to CD
  (non_intersecting : c = xC - xD)  -- BC and DA do not intersect
  (length_BC : b^2 = xC^2 + yC^2)
  (length_AD : d^2 = (xD + a)^2 + yC^2)

/-- The x-coordinates of points C and D in a parallel quadrilateral -/
theorem parallel_quadrilateral_coordinates
  (a b c d : ℝ) (quad : ParallelQuadrilateral a b c d)
  (h_a : a ≠ c) :
  quad.xD = (d^2 - b^2 - a^2 + c^2) / (2*(a - c)) ∧
  quad.xC = (d^2 - b^2 - a^2 + c^2) / (2*(a - c)) + c :=
sorry

end parallel_quadrilateral_coordinates_l3314_331483


namespace total_shaded_area_l3314_331429

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a right triangle with two equal legs -/
structure RightTriangle where
  leg : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  r.width * r.height

/-- Calculates the area of a right triangle -/
def rightTriangleArea (t : RightTriangle) : ℝ :=
  0.5 * t.leg * t.leg

/-- Represents the overlap between rectangles -/
def rectangleOverlap : ℝ := 20

/-- Represents the fraction of triangle overlap with rectangles -/
def triangleOverlapFraction : ℝ := 0.5

/-- Theorem stating the total shaded area -/
theorem total_shaded_area (r1 r2 : Rectangle) (t : RightTriangle) :
  let totalArea := rectangleArea r1 + rectangleArea r2 - rectangleOverlap
  let triangleCorrection := triangleOverlapFraction * rightTriangleArea t
  totalArea - triangleCorrection = 70.75 :=
by
  sorry

#check total_shaded_area (Rectangle.mk 4 12) (Rectangle.mk 5 9) (RightTriangle.mk 3)

end total_shaded_area_l3314_331429


namespace pencils_per_child_l3314_331496

theorem pencils_per_child (total_children : ℕ) (total_pencils : ℕ) 
  (h1 : total_children = 9) 
  (h2 : total_pencils = 18) : 
  total_pencils / total_children = 2 := by
  sorry

end pencils_per_child_l3314_331496


namespace count_words_to_1000_l3314_331434

def word_count_1_to_99 : Nat := 171

def word_count_100_to_999 : Nat := 486 + 1944

def word_count_1000 : Nat := 37

theorem count_words_to_1000 :
  word_count_1_to_99 + word_count_100_to_999 + word_count_1000 = 2611 :=
by sorry

end count_words_to_1000_l3314_331434


namespace fraction_problem_l3314_331420

theorem fraction_problem (p q : ℚ) : 
  p = 4 → 
  (1 : ℚ)/7 + (2*q - p)/(2*q + p) = 0.5714285714285714 → 
  q = 5 := by
sorry

end fraction_problem_l3314_331420


namespace f_geq_one_range_f_lt_a_plus_two_l3314_331440

/-- The quadratic function f(x) = ax² + (2-a)x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (2 - a) * x + a

/-- Theorem stating the range of a for which f(x) ≥ 1 holds for all real x -/
theorem f_geq_one_range (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 1) ↔ a ≥ 2 * Real.sqrt 3 / 3 := by sorry

/-- Helper function to describe the solution set of f(x) < a+2 -/
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x | x < 1}
  else if a > 0 then {x | -2/a < x ∧ x < 1}
  else if -2 < a ∧ a < 0 then {x | x < 1 ∨ x > -2/a}
  else if a = -2 then Set.univ
  else {x | x < -2/a ∨ x > 1}

/-- Theorem stating the solution set of f(x) < a+2 -/
theorem f_lt_a_plus_two (a : ℝ) (x : ℝ) :
  f a x < a + 2 ↔ x ∈ solution_set a := by sorry

end f_geq_one_range_f_lt_a_plus_two_l3314_331440


namespace harolds_books_ratio_l3314_331401

theorem harolds_books_ratio (h m : ℝ) : 
  h > 0 ∧ m > 0 → 
  (1/3 : ℝ) * h + (1/2 : ℝ) * m = (5/6 : ℝ) * m → 
  h / m = 1 := by
sorry

end harolds_books_ratio_l3314_331401


namespace gcd_459_357_l3314_331446

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l3314_331446


namespace angle_ABF_is_right_l3314_331453

/-- An ellipse with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e_eq : e = (Real.sqrt 5 - 1) / 2

/-- The angle ABF in an ellipse, where A is the left vertex, 
    F is the right focus, and B is one endpoint of the minor axis -/
def angle_ABF (E : Ellipse) : ℝ := sorry

/-- Theorem: In an ellipse with the given properties, the angle ABF is 90° -/
theorem angle_ABF_is_right (E : Ellipse) : angle_ABF E = 90 := by sorry

end angle_ABF_is_right_l3314_331453


namespace teacher_pay_per_period_l3314_331424

/-- Calculates the pay per period for a teacher given their work schedule and total earnings --/
theorem teacher_pay_per_period 
  (periods_per_day : ℕ)
  (days_per_month : ℕ)
  (months_worked : ℕ)
  (total_earnings : ℕ)
  (h1 : periods_per_day = 5)
  (h2 : days_per_month = 24)
  (h3 : months_worked = 6)
  (h4 : total_earnings = 3600) :
  total_earnings / (periods_per_day * days_per_month * months_worked) = 5 := by
  sorry

#eval 3600 / (5 * 24 * 6)  -- This should output 5

end teacher_pay_per_period_l3314_331424


namespace cube_root_plus_square_root_l3314_331417

theorem cube_root_plus_square_root : 
  ∃ (x : ℝ), (x = 4 ∨ x = -8) ∧ x = ((-64 : ℝ)^(1/2))^(1/3) + (36 : ℝ)^(1/2) :=
sorry

end cube_root_plus_square_root_l3314_331417


namespace rectangular_hall_dimension_difference_l3314_331410

/-- Proves that for a rectangular hall with width being half the length and area 288 sq. m, 
    the difference between length and width is 12 meters -/
theorem rectangular_hall_dimension_difference 
  (length width : ℝ) 
  (h1 : width = length / 2) 
  (h2 : length * width = 288) : 
  length - width = 12 := by
  sorry

end rectangular_hall_dimension_difference_l3314_331410


namespace speed_ratio_is_two_l3314_331499

/-- Represents the driving scenario for Daniel's commute --/
structure DrivingScenario where
  x : ℝ  -- Speed on Sunday in miles per hour
  y : ℝ  -- Speed for first 32 miles on Monday in miles per hour
  total_distance : ℝ  -- Total distance in miles
  first_part_distance : ℝ  -- Distance of first part on Monday in miles

/-- The theorem stating the ratio of speeds --/
theorem speed_ratio_is_two (scenario : DrivingScenario) : 
  scenario.x > 0 → 
  scenario.y > 0 → 
  scenario.total_distance = 60 → 
  scenario.first_part_distance = 32 → 
  (scenario.first_part_distance / scenario.y + (scenario.total_distance - scenario.first_part_distance) / (scenario.x / 2)) = 
    1.2 * (scenario.total_distance / scenario.x) → 
  scenario.y / scenario.x = 2 := by
  sorry

end speed_ratio_is_two_l3314_331499


namespace sunny_gave_away_two_cakes_l3314_331421

/-- The number of cakes Sunny initially baked -/
def initial_cakes : ℕ := 8

/-- The number of candles Sunny puts on each remaining cake -/
def candles_per_cake : ℕ := 6

/-- The total number of candles Sunny uses -/
def total_candles : ℕ := 36

/-- The number of cakes Sunny gave away -/
def cakes_given_away : ℕ := initial_cakes - (total_candles / candles_per_cake)

theorem sunny_gave_away_two_cakes : cakes_given_away = 2 := by
  sorry

end sunny_gave_away_two_cakes_l3314_331421


namespace possible_values_of_a_l3314_331492

def A (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A a ∪ B = B) ↔ (a = -1/2 ∨ a = 0 ∨ a = -1) :=
by sorry

end possible_values_of_a_l3314_331492


namespace verandah_width_is_two_l3314_331464

/-- Represents the dimensions of a rectangular room with a surrounding verandah. -/
structure RoomWithVerandah where
  roomLength : ℝ
  roomWidth : ℝ
  verandahWidth : ℝ

/-- Calculates the area of the verandah given the room dimensions. -/
def verandahArea (r : RoomWithVerandah) : ℝ :=
  (r.roomLength + 2 * r.verandahWidth) * (r.roomWidth + 2 * r.verandahWidth) - r.roomLength * r.roomWidth

/-- Theorem stating that for a room of 15m x 12m with a verandah of area 124 sq m, the verandah width is 2m. -/
theorem verandah_width_is_two :
  ∃ (r : RoomWithVerandah), r.roomLength = 15 ∧ r.roomWidth = 12 ∧ verandahArea r = 124 ∧ r.verandahWidth = 2 :=
by sorry

end verandah_width_is_two_l3314_331464


namespace window_savings_theorem_l3314_331402

/-- Represents the savings when purchasing windows together vs separately --/
def windowSavings (windowPrice : ℕ) (daveWindows : ℕ) (dougWindows : ℕ) : ℕ :=
  let batchSize := 10
  let freeWindows := 2
  let separateCost := 
    (((daveWindows + batchSize - 1) / batchSize * batchSize - freeWindows) * windowPrice)
    + (((dougWindows + batchSize - 1) / batchSize * batchSize - freeWindows) * windowPrice)
  let jointWindows := daveWindows + dougWindows
  let jointCost := ((jointWindows + batchSize - 1) / batchSize * batchSize - freeWindows * (jointWindows / batchSize)) * windowPrice
  separateCost - jointCost

/-- Theorem stating the savings when Dave and Doug purchase windows together --/
theorem window_savings_theorem : 
  windowSavings 120 9 11 = 120 := by
  sorry

end window_savings_theorem_l3314_331402


namespace circle_through_points_l3314_331472

-- Define the points
def O : ℝ × ℝ := (0, 0)
def M1 : ℝ × ℝ := (1, 1)
def M2 : ℝ × ℝ := (4, 2)

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Theorem statement
theorem circle_through_points : 
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (4, -3) ∧ 
    radius = 5 ∧
    O ∈ Circle center radius ∧
    M1 ∈ Circle center radius ∧
    M2 ∈ Circle center radius ∧
    Circle center radius = {p : ℝ × ℝ | (p.1 - 4)^2 + (p.2 + 3)^2 = 25} := by
  sorry

end circle_through_points_l3314_331472


namespace line_circle_intersection_l3314_331442

/-- The intersection of a line and a circle with specific properties implies a unique value for the parameter a. -/
theorem line_circle_intersection (a : ℝ) (h_a_pos : a > 0) : 
  (∃ A B : ℝ × ℝ, 
    (A.2 = A.1 + 2*a ∧ B.2 = B.1 + 2*a) ∧ 
    (A.1^2 + A.2^2 - 2*a*A.2 - 2 = 0 ∧ B.1^2 + B.2^2 - 2*a*B.2 - 2 = 0) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 12)) →
  a = Real.sqrt 2 := by
sorry

end line_circle_intersection_l3314_331442


namespace hydrogen_weight_in_H2CrO4_l3314_331495

def atomic_weight_H : ℝ := 1.008
def molecular_weight_H2CrO4 : ℝ := 118

theorem hydrogen_weight_in_H2CrO4 :
  let hydrogen_count : ℕ := 2
  let hydrogen_weight : ℝ := atomic_weight_H * hydrogen_count
  hydrogen_weight = 2.016 := by sorry

end hydrogen_weight_in_H2CrO4_l3314_331495


namespace platform_length_l3314_331460

/-- The length of a platform given train specifications and crossing time -/
theorem platform_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmh = 55 →
  crossing_time = 50.395968322534195 →
  ∃ platform_length : ℝ, abs (platform_length - 520) < 0.01 := by
  sorry

end platform_length_l3314_331460


namespace line_through_circle_center_l3314_331431

theorem line_through_circle_center (a : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0 ∧ 3*x + y + a = 0 ∧ 
   ∀ cx cy : ℝ, (cx - x)^2 + (cy - y)^2 ≤ (cx + 1)^2 + (cy - 2)^2) → 
  a = 1 := by
sorry

end line_through_circle_center_l3314_331431


namespace cloth_profit_theorem_l3314_331412

/-- Calculates the profit per meter of cloth (rounded to the nearest rupee) -/
def profit_per_meter (meters : ℕ) (total_selling_price : ℚ) (cost_price_per_meter : ℚ) : ℕ :=
  let total_cost_price := meters * cost_price_per_meter
  let total_profit := total_selling_price - total_cost_price
  let profit_per_meter := total_profit / meters
  (profit_per_meter + 1/2).floor.toNat

/-- The profit per meter of cloth is 29 rupees -/
theorem cloth_profit_theorem :
  profit_per_meter 78 6788 (58.02564102564102) = 29 := by
  sorry

end cloth_profit_theorem_l3314_331412


namespace quadratic_equation_result_l3314_331406

theorem quadratic_equation_result (a : ℝ) (h : a^2 - 4*a - 12 = 0) : 
  -2*a^2 + 8*a + 40 = 16 := by
  sorry

end quadratic_equation_result_l3314_331406


namespace vegetable_cost_l3314_331466

def initial_amount : ℤ := 100
def roast_cost : ℤ := 17
def remaining_amount : ℤ := 72

theorem vegetable_cost :
  initial_amount - roast_cost - remaining_amount = 11 := by
  sorry

end vegetable_cost_l3314_331466


namespace mixture_qualified_probability_l3314_331451

theorem mixture_qualified_probability 
  (batch1_defective_rate : ℝ)
  (batch2_defective_rate : ℝ)
  (mix_ratio1 : ℝ)
  (mix_ratio2 : ℝ)
  (h1 : batch1_defective_rate = 0.05)
  (h2 : batch2_defective_rate = 0.15)
  (h3 : mix_ratio1 = 3)
  (h4 : mix_ratio2 = 2) :
  let total_ratio := mix_ratio1 + mix_ratio2
  let batch1_qualified_rate := 1 - batch1_defective_rate
  let batch2_qualified_rate := 1 - batch2_defective_rate
  let mixture_qualified_rate := 
    (batch1_qualified_rate * mix_ratio1 + batch2_qualified_rate * mix_ratio2) / total_ratio
  mixture_qualified_rate = 0.91 := by
sorry

end mixture_qualified_probability_l3314_331451


namespace students_left_after_dropout_l3314_331405

/-- Calculates the number of students left after some drop out -/
def studentsLeft (initialBoys initialGirls boysDropped girlsDropped : ℕ) : ℕ :=
  (initialBoys - boysDropped) + (initialGirls - girlsDropped)

/-- Theorem: Given 14 boys and 10 girls initially, if 4 boys and 3 girls drop out, 17 students are left -/
theorem students_left_after_dropout : studentsLeft 14 10 4 3 = 17 := by
  sorry

end students_left_after_dropout_l3314_331405


namespace shopkeeper_milk_ounces_l3314_331474

/-- Calculates the total amount of milk in ounces bought by a shopkeeper -/
theorem shopkeeper_milk_ounces 
  (packets : ℕ) 
  (ml_per_packet : ℕ) 
  (ml_per_ounce : ℕ) 
  (h1 : packets = 150)
  (h2 : ml_per_packet = 250)
  (h3 : ml_per_ounce = 30) : 
  (packets * ml_per_packet) / ml_per_ounce = 1250 := by
  sorry

#check shopkeeper_milk_ounces

end shopkeeper_milk_ounces_l3314_331474


namespace range_of_a_l3314_331457

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x < -1}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a < x ∧ x < a + 3}

-- Define the complement of A
def A_complement : Set ℝ := {x : ℝ | x ≥ -1}

-- Theorem statement
theorem range_of_a (a : ℝ) : B a ⊆ A_complement ↔ a ≥ -1/2 := by
  sorry

end range_of_a_l3314_331457


namespace quadratic_roots_sum_of_squares_l3314_331486

theorem quadratic_roots_sum_of_squares (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 8 = 0) → 
  (x₂^2 + 2*x₂ - 8 = 0) → 
  (x₁^2 + x₂^2 = 20) := by
  sorry

end quadratic_roots_sum_of_squares_l3314_331486


namespace parallel_reasoning_is_deductive_l3314_331409

-- Define a type for lines
structure Line : Type :=
  (id : ℕ)

-- Define a parallel relation between lines
def parallel (l1 l2 : Line) : Prop := sorry

-- Define the property of transitivity for parallel lines
axiom parallel_transitive : ∀ (x y z : Line), parallel x y → parallel y z → parallel x z

-- Given lines a, b, and c
variable (a b c : Line)

-- Given that a is parallel to b, and b is parallel to c
axiom a_parallel_b : parallel a b
axiom b_parallel_c : parallel b c

-- Define deductive reasoning
def is_deductive_reasoning (conclusion : Prop) : Prop := sorry

-- Theorem to prove
theorem parallel_reasoning_is_deductive : 
  is_deductive_reasoning (parallel a c) := sorry

end parallel_reasoning_is_deductive_l3314_331409


namespace hotel_operations_cost_l3314_331487

/-- Proves that the total cost of operations is $100 given the specified conditions --/
theorem hotel_operations_cost (cost : ℝ) (payments : ℝ) (loss : ℝ) : 
  payments = (3/4) * cost → 
  loss = 25 → 
  payments + loss = cost → 
  cost = 100 := by
  sorry

end hotel_operations_cost_l3314_331487


namespace small_cube_edge_length_l3314_331489

/-- Given a cube with volume 1000 cm³, if 8 small cubes of equal size are cut off from its corners
    such that the remaining volume is 488 cm³, then the edge length of each small cube is 4 cm. -/
theorem small_cube_edge_length (x : ℝ) : 
  (1000 : ℝ) - 8 * x^3 = 488 → x = 4 := by sorry

end small_cube_edge_length_l3314_331489


namespace probability_letter_in_mathematics_l3314_331443

def alphabet : Finset Char := sorry

def mathematics : String := "MATHEMATICS"

def unique_letters (s : String) : Finset Char :=
  s.toList.toFinset

theorem probability_letter_in_mathematics :
  (unique_letters mathematics).card / alphabet.card = 4 / 13 := by
  sorry

end probability_letter_in_mathematics_l3314_331443


namespace larger_circle_radius_is_32_l3314_331415

/-- Two concentric circles with chord properties -/
structure ConcentricCircles where
  r : ℝ  -- radius of the smaller circle
  AB : ℝ  -- length of AB
  h_ratio : r > 0  -- radius is positive
  h_AB : AB = 16  -- given length of AB

/-- The radius of the larger circle in the concentric circles setup -/
def larger_circle_radius (c : ConcentricCircles) : ℝ := 4 * c.r

theorem larger_circle_radius_is_32 (c : ConcentricCircles) : 
  larger_circle_radius c = 32 := by
  sorry

#check larger_circle_radius_is_32

end larger_circle_radius_is_32_l3314_331415


namespace intersection_of_A_and_B_l3314_331404

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

end intersection_of_A_and_B_l3314_331404


namespace girls_in_senior_year_l3314_331444

theorem girls_in_senior_year 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (girls_boys_diff : ℕ) 
  (h1 : total_students = 1200)
  (h2 : sample_size = 100)
  (h3 : girls_boys_diff = 20) :
  let boys_in_sample := (sample_size + girls_boys_diff) / 2
  let girls_in_sample := sample_size - boys_in_sample
  let sampling_ratio := sample_size / total_students
  (girls_in_sample * (total_students / sample_size) : ℚ) = 480 := by
sorry

end girls_in_senior_year_l3314_331444


namespace factorization_cubic_quadratic_l3314_331488

theorem factorization_cubic_quadratic (a : ℝ) : a^3 - 2*a^2 = a^2*(a - 2) := by
  sorry

end factorization_cubic_quadratic_l3314_331488


namespace journey_results_correct_l3314_331455

/-- Truck's journey between Town A and Village B -/
structure TruckJourney where
  uphill_distance : ℝ
  downhill_distance : ℝ
  flat_distance : ℝ
  round_trip_time_diff : ℝ
  uphill_speed_ratio : ℝ
  downhill_speed_ratio : ℝ
  flat_speed_ratio : ℝ

/-- Calculated speeds and times for the journey -/
structure JourneyResults where
  uphill_speed : ℝ
  downhill_speed : ℝ
  flat_speed : ℝ
  time_a_to_b : ℝ
  time_b_to_a : ℝ

/-- Theorem stating the correctness of the calculated results -/
theorem journey_results_correct (j : TruckJourney)
  (res : JourneyResults)
  (h1 : j.uphill_distance = 20)
  (h2 : j.downhill_distance = 14)
  (h3 : j.flat_distance = 5)
  (h4 : j.round_trip_time_diff = 1/6)
  (h5 : j.uphill_speed_ratio = 3)
  (h6 : j.downhill_speed_ratio = 6)
  (h7 : j.flat_speed_ratio = 5)
  (h8 : res.uphill_speed = 18)
  (h9 : res.downhill_speed = 36)
  (h10 : res.flat_speed = 30)
  (h11 : res.time_a_to_b = 5/3)
  (h12 : res.time_b_to_a = 3/2) :
  (j.uphill_distance / res.uphill_speed +
   j.downhill_distance / res.downhill_speed +
   j.flat_distance / res.flat_speed) -
  (j.uphill_distance / res.downhill_speed +
   j.downhill_distance / res.uphill_speed +
   j.flat_distance / res.flat_speed) = j.round_trip_time_diff ∧
  res.time_a_to_b =
    j.uphill_distance / res.uphill_speed +
    j.downhill_distance / res.downhill_speed +
    j.flat_distance / res.flat_speed ∧
  res.time_b_to_a =
    j.uphill_distance / res.downhill_speed +
    j.downhill_distance / res.uphill_speed +
    j.flat_distance / res.flat_speed ∧
  res.uphill_speed / res.downhill_speed = j.uphill_speed_ratio / j.downhill_speed_ratio ∧
  res.downhill_speed / res.flat_speed = j.downhill_speed_ratio / j.flat_speed_ratio :=
by sorry


end journey_results_correct_l3314_331455


namespace three_lines_determine_plane_l3314_331449

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane where
  -- Define properties of a plane

/-- Represents the intersection of two lines -/
def intersect (l1 l2 : Line3D) : Prop :=
  sorry

/-- Represents that three lines have no common point -/
def no_common_point (l1 l2 l3 : Line3D) : Prop :=
  sorry

/-- Represents that a plane contains a line -/
def plane_contains_line (p : Plane) (l : Line3D) : Prop :=
  sorry

/-- Three lines intersecting in pairs without a common point determine a unique plane -/
theorem three_lines_determine_plane (l1 l2 l3 : Line3D) :
  intersect l1 l2 ∧ intersect l2 l3 ∧ intersect l3 l1 ∧ no_common_point l1 l2 l3 →
  ∃! p : Plane, plane_contains_line p l1 ∧ plane_contains_line p l2 ∧ plane_contains_line p l3 :=
sorry

end three_lines_determine_plane_l3314_331449


namespace robotics_club_proof_l3314_331461

theorem robotics_club_proof (total : ℕ) (programming : ℕ) (electronics : ℕ) (both : ℕ)
  (h1 : total = 120)
  (h2 : programming = 80)
  (h3 : electronics = 50)
  (h4 : both = 15) :
  total - (programming + electronics - both) = 5 := by
  sorry

end robotics_club_proof_l3314_331461


namespace right_triangle_area_and_perimeter_l3314_331436

theorem right_triangle_area_and_perimeter : 
  ∀ (triangle : Set ℝ) (leg1 leg2 hypotenuse : ℝ),
  -- Conditions
  leg1 = 30 →
  leg2 = 45 →
  hypotenuse^2 = leg1^2 + leg2^2 →
  -- Definitions
  let area := (1/2) * leg1 * leg2
  let perimeter := leg1 + leg2 + hypotenuse
  -- Theorem
  area = 675 ∧ perimeter = 129 := by
sorry

end right_triangle_area_and_perimeter_l3314_331436


namespace original_number_problem_l3314_331478

theorem original_number_problem (x : ℝ) :
  1 - 1/x = 5/2 → x = -2/3 := by
  sorry

end original_number_problem_l3314_331478


namespace min_PQ_ratio_approaches_infinity_l3314_331439

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

-- Define points
variable (X Y M P Q : V)

-- Define conditions
variable (h1 : M = (X + Y) / 2)
variable (h2 : ∃ (t k : ℝ) (d : V), P = Y + t • d ∧ Q = Y - k • d ∧ t > 0 ∧ k > 0)
variable (h3 : ‖X - Q‖ = 2 * ‖M - P‖)
variable (h4 : ‖X - Y‖ / 2 < ‖M - P‖ ∧ ‖M - P‖ < 3 * ‖X - Y‖ / 2)

-- Theorem statement
theorem min_PQ_ratio_approaches_infinity :
  ∀ ε > 0, ∃ δ > 0, ∀ P' Q' : V,
    ‖P' - Q'‖ < ‖P - Q‖ + δ →
    ‖P' - Y‖ / ‖Q' - Y‖ > 1 / ε :=
sorry

end min_PQ_ratio_approaches_infinity_l3314_331439


namespace class_size_proof_l3314_331468

theorem class_size_proof (total_average : ℝ) (group1_size : ℕ) (group1_average : ℝ)
                         (group2_size : ℕ) (group2_average : ℝ) (last_student_age : ℕ) :
  total_average = 15 →
  group1_size = 5 →
  group1_average = 14 →
  group2_size = 9 →
  group2_average = 16 →
  last_student_age = 11 →
  ∃ (total_students : ℕ), total_students = 15 ∧
    (total_students : ℝ) * total_average =
      (group1_size : ℝ) * group1_average +
      (group2_size : ℝ) * group2_average +
      last_student_age :=
by
  sorry

#check class_size_proof

end class_size_proof_l3314_331468


namespace purely_imaginary_complex_number_l3314_331416

theorem purely_imaginary_complex_number (a : ℝ) : 
  (a^2 - a - 2 = 0) ∧ (|a - 1| - 1 ≠ 0) → a = -1 := by sorry

end purely_imaginary_complex_number_l3314_331416


namespace smallest_prime_10_less_than_perfect_square_l3314_331494

/-- A number is a perfect square if it's the square of an integer -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2

/-- The smallest prime that is 10 less than a perfect square -/
theorem smallest_prime_10_less_than_perfect_square :
  (∃ a : ℕ, Nat.Prime a ∧
    (∃ b : ℕ, is_perfect_square b ∧ a = b - 10) ∧
    (∀ a' : ℕ, a' < a →
      ¬(Nat.Prime a' ∧ ∃ b' : ℕ, is_perfect_square b' ∧ a' = b' - 10))) →
  (∃ a : ℕ, a = 71 ∧ Nat.Prime a ∧
    (∃ b : ℕ, is_perfect_square b ∧ a = b - 10) ∧
    (∀ a' : ℕ, a' < a →
      ¬(Nat.Prime a' ∧ ∃ b' : ℕ, is_perfect_square b' ∧ a' = b' - 10))) :=
by
  sorry

end smallest_prime_10_less_than_perfect_square_l3314_331494


namespace infinite_integers_with_noncounting_divisors_l3314_331413

theorem infinite_integers_with_noncounting_divisors :
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
    (∀ a ∈ S, a ≥ 1 ∧ 
      (∀ n : ℕ, n ≥ 1 → ¬(Nat.card (Nat.divisors a) = n))) := by
  sorry

end infinite_integers_with_noncounting_divisors_l3314_331413


namespace stratified_sample_sum_eq_six_l3314_331430

/-- Represents the number of varieties in each food category -/
def food_categories : List Nat := [40, 10, 30, 20]

/-- The total number of food varieties -/
def total_varieties : Nat := food_categories.sum

/-- The sample size for food safety inspection -/
def sample_size : Nat := 20

/-- Calculates the number of samples for a given category size -/
def stratified_sample (category_size : Nat) : Nat :=
  (sample_size * category_size) / total_varieties

/-- Theorem: The sum of stratified samples from the second and fourth categories is 6 -/
theorem stratified_sample_sum_eq_six :
  stratified_sample (food_categories[1]) + stratified_sample (food_categories[3]) = 6 := by
  sorry

end stratified_sample_sum_eq_six_l3314_331430


namespace total_students_l3314_331470

/-- The number of students in the three classrooms -/
structure ClassroomCounts where
  tina : ℕ
  maura : ℕ
  zack : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (c : ClassroomCounts) : Prop :=
  c.tina = c.maura ∧
  c.zack = (c.tina + c.maura) / 2 ∧
  c.zack - 1 = 22

/-- The theorem stating the total number of students -/
theorem total_students (c : ClassroomCounts) 
  (h : satisfies_conditions c) : c.tina + c.maura + c.zack = 69 := by
  sorry

end total_students_l3314_331470


namespace min_value_arithmetic_sequence_l3314_331447

/-- Given three positive real numbers forming an arithmetic sequence,
    the sum of their ratio and its reciprocal is at least 5/2 -/
theorem min_value_arithmetic_sequence (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_arith : b - a = c - b) : (a + c) / b + b / (a + c) ≥ 5 / 2 := by
  sorry

end min_value_arithmetic_sequence_l3314_331447


namespace same_solution_implies_value_l3314_331422

theorem same_solution_implies_value (a b : ℝ) :
  (∃ x y : ℝ, 5 * x + y = 3 ∧ a * x + 5 * y = 4 ∧ x - 2 * y = 5 ∧ 5 * x + b * y = 1) →
  1/2 * a - b = 5 := by
sorry

end same_solution_implies_value_l3314_331422


namespace modulus_of_Z_l3314_331452

/-- The modulus of the complex number Z = 1/(1+i) + i^3 is equal to √10/2 -/
theorem modulus_of_Z : Complex.abs (1 / (1 + Complex.I) + Complex.I^3) = Real.sqrt 10 / 2 := by
  sorry

end modulus_of_Z_l3314_331452


namespace parallel_vectors_problem_l3314_331408

/-- Given two vectors a and b in R², where a is parallel to (2a + b), prove that the second component of b is 4 and m = 2. -/
theorem parallel_vectors_problem (a b : ℝ × ℝ) (m : ℝ) :
  a = (1, 2) →
  b = (m, 4) →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • (2 • a + b) →
  m = 2 := by
  sorry

end parallel_vectors_problem_l3314_331408


namespace no_numbers_satisfying_conditions_l3314_331459

theorem no_numbers_satisfying_conditions : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 300 →
    (6 ∣ n ∧ 8 ∣ n) → (4 ∣ n ∨ 11 ∣ n) :=
by sorry

end no_numbers_satisfying_conditions_l3314_331459


namespace probability_theorem_l3314_331432

/-- Represents the enrollment data for language classes --/
structure LanguageEnrollment where
  total : ℕ
  french : ℕ
  spanish : ℕ
  german : ℕ
  french_and_spanish : ℕ
  spanish_and_german : ℕ
  french_and_german : ℕ
  all_three : ℕ

/-- Calculates the probability of selecting two students that cover all three languages --/
def probability_all_languages (e : LanguageEnrollment) : ℚ :=
  1 - (132 : ℚ) / (435 : ℚ)

/-- Theorem stating the probability of selecting two students covering all three languages --/
theorem probability_theorem (e : LanguageEnrollment) 
  (h1 : e.total = 30)
  (h2 : e.french = 20)
  (h3 : e.spanish = 18)
  (h4 : e.german = 10)
  (h5 : e.french_and_spanish = 12)
  (h6 : e.spanish_and_german = 5)
  (h7 : e.french_and_german = 4)
  (h8 : e.all_three = 3) :
  probability_all_languages e = 101 / 145 := by
  sorry

end probability_theorem_l3314_331432


namespace quadratic_equation_two_distinct_roots_l3314_331493

theorem quadratic_equation_two_distinct_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 3*k*x₁ - 2 = 0 ∧ x₂^2 - 3*k*x₂ - 2 = 0 :=
by sorry

end quadratic_equation_two_distinct_roots_l3314_331493


namespace fraction_simplification_l3314_331458

theorem fraction_simplification (x m n : ℝ) (hx : x ≠ 0) (hmn : m + n ≠ 0) :
  x / (x * (m + n)) = 1 / (m + n) := by
  sorry

end fraction_simplification_l3314_331458


namespace right_triangle_3_4_5_l3314_331433

theorem right_triangle_3_4_5 : ∃ (a b c : ℕ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 := by
  sorry

end right_triangle_3_4_5_l3314_331433


namespace teacher_books_l3314_331498

theorem teacher_books (num_children : ℕ) (books_per_child : ℕ) (total_books : ℕ) : 
  num_children = 10 → books_per_child = 7 → total_books = 78 →
  total_books - (num_children * books_per_child) = 8 := by
sorry

end teacher_books_l3314_331498


namespace numerical_puzzle_solution_l3314_331467

theorem numerical_puzzle_solution :
  ∃! (A B C D E F : ℕ),
    A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧ E ≤ 9 ∧ F ≤ 9 ∧
    (10 * A + A) * (10 * A + B) = 1000 * C + 100 * D + 10 * E + F ∧
    (10 * C + C) * (100 * C + 10 * E + F) = 1000 * C + 100 * D + 10 * E + F ∧
    A = 4 ∧ B = 5 ∧ C = 1 ∧ D = 9 ∧ E = 8 ∧ F = 0 :=
by sorry

end numerical_puzzle_solution_l3314_331467


namespace class_size_l3314_331400

/-- Represents the number of students in a class with English and German courses -/
structure ClassEnrollment where
  total : ℕ
  english : ℕ
  german : ℕ
  both : ℕ
  onlyEnglish : ℕ

/-- Theorem stating the total number of students given the enrollment conditions -/
theorem class_size (c : ClassEnrollment)
  (h1 : c.both = 12)
  (h2 : c.german = 22)
  (h3 : c.onlyEnglish = 23)
  (h4 : c.total = c.english + c.german - c.both)
  (h5 : c.english = c.onlyEnglish + c.both) :
  c.total = 45 := by
  sorry

#check class_size

end class_size_l3314_331400


namespace x_value_l3314_331445

theorem x_value (x y : ℝ) (h : x / (x - 2) = (y^2 + 3*y + 1) / (y^2 + 3*y - 1)) : 
  x = 2*y^2 + 6*y + 2 := by
  sorry

end x_value_l3314_331445


namespace solve_for_y_l3314_331479

theorem solve_for_y (x y : ℝ) (h1 : x^(3*y) = 64) (h2 : x = 8) : y = 2/3 := by
  sorry

end solve_for_y_l3314_331479


namespace soccer_camp_ratio_l3314_331425

theorem soccer_camp_ratio (total_kids : ℕ) (afternoon_kids : ℕ) 
  (h1 : total_kids = 2000)
  (h2 : afternoon_kids = 750)
  (h3 : ∃ (morning_kids : ℕ), 4 * morning_kids = total_soccer_kids - afternoon_kids) :
  ∃ (total_soccer_kids : ℕ), 
    2 * total_soccer_kids = total_kids ∧ 
    4 * afternoon_kids = 3 * total_soccer_kids := by
sorry


end soccer_camp_ratio_l3314_331425


namespace polynomial_evaluation_l3314_331484

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 18 = 0 ∧ x^3 - 3*x^2 - 9*x + 5 = 59 := by
  sorry

end polynomial_evaluation_l3314_331484
