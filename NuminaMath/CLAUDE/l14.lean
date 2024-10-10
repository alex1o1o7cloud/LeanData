import Mathlib

namespace agricultural_machinery_growth_rate_l14_1455

/-- The average growth rate for May and June in an agricultural machinery factory --/
theorem agricultural_machinery_growth_rate :
  ∀ (april_production : ℕ) (total_production : ℕ) (growth_rate : ℝ),
  april_production = 500 →
  total_production = 1820 →
  april_production + 
    april_production * (1 + growth_rate) + 
    april_production * (1 + growth_rate)^2 = total_production →
  growth_rate = 0.2 := by
sorry

end agricultural_machinery_growth_rate_l14_1455


namespace simplify_and_evaluate_simplify_to_polynomial_l14_1411

-- Problem 1
theorem simplify_and_evaluate (a : ℚ) : 
  a = -2 → (a^2 + a) / (a^2 - 3*a) / ((a^2 - 1) / (a - 3)) - 1 / (a + 1) = 2/3 := by
  sorry

-- Problem 2
theorem simplify_to_polynomial (x : ℚ) : 
  (x^2 - 1) / (x - 4) / ((x + 1) / (4 - x)) = 1 - x := by
  sorry

end simplify_and_evaluate_simplify_to_polynomial_l14_1411


namespace square_sum_reciprocal_l14_1476

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := by
  sorry

end square_sum_reciprocal_l14_1476


namespace gcd_2475_7350_l14_1472

theorem gcd_2475_7350 : Nat.gcd 2475 7350 = 225 := by sorry

end gcd_2475_7350_l14_1472


namespace square_sum_equals_product_l14_1468

theorem square_sum_equals_product (x y z t : ℤ) :
  x^2 + y^2 + z^2 + t^2 = 2*x*y*z*t → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end square_sum_equals_product_l14_1468


namespace tetrahedron_with_two_square_intersections_l14_1408

/-- A tetrahedron in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- The intersection of a tetrahedron and a plane -/
def intersection (t : Tetrahedron) (p : Plane) : Set (ℝ × ℝ × ℝ) :=
  sorry

/-- Predicate to check if a set of points forms a square -/
def is_square (s : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry

/-- The side length of a square -/
def side_length (s : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem stating the existence of a tetrahedron with the desired properties -/
theorem tetrahedron_with_two_square_intersections :
  ∃ (t : Tetrahedron) (p1 p2 : Plane),
    p1 ≠ p2 ∧
    is_square (intersection t p1) ∧
    is_square (intersection t p2) ∧
    side_length (intersection t p1) ≤ 1 ∧
    side_length (intersection t p2) ≥ 100 :=
  sorry

end tetrahedron_with_two_square_intersections_l14_1408


namespace point_difference_on_plane_l14_1494

/-- Given two points on a plane, prove that the difference in their x and z coordinates are 3 and 0 respectively. -/
theorem point_difference_on_plane (m n z p q : ℝ) (k : ℝ) (hk : k ≠ 0) :
  (m = n / 6 - 2 / 5 + z / k) →
  (m + p = (n + 18) / 6 - 2 / 5 + (z + q) / k) →
  p = 3 ∧ q = 0 := by
  sorry

end point_difference_on_plane_l14_1494


namespace trig_special_angles_sum_l14_1457

theorem trig_special_angles_sum : 
  Real.sin (π / 2) + 2 * Real.cos 0 - 3 * Real.sin (3 * π / 2) + 10 * Real.cos π = -4 := by
  sorry

end trig_special_angles_sum_l14_1457


namespace garden_area_with_fountain_garden_area_calculation_l14_1417

/-- Calculates the new available area for planting in a rectangular garden with a circular fountain -/
theorem garden_area_with_fountain (perimeter : ℝ) (side : ℝ) (fountain_radius : ℝ) : ℝ :=
  let length := (perimeter - 2 * side) / 2
  let garden_area := length * side
  let fountain_area := Real.pi * fountain_radius^2
  garden_area - fountain_area

/-- Proves that the new available area for planting is approximately 37185.84 square meters -/
theorem garden_area_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |garden_area_with_fountain 950 100 10 - 37185.84| < ε :=
sorry

end garden_area_with_fountain_garden_area_calculation_l14_1417


namespace cube_can_be_threaded_tetrahedron_can_be_threaded_l14_1413

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a 2D point
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a frame (cube or tetrahedron)
structure Frame where
  vertices : List Point3D
  edges : List (Point3D × Point3D)

-- Define a hole in the plane
structure Hole where
  boundary : Point2D → Bool

-- Function to check if a hole is valid (closed and non-self-intersecting)
def isValidHole (h : Hole) : Prop :=
  sorry

-- Function to check if a frame can be threaded through a hole
def canThreadThrough (f : Frame) (h : Hole) : Prop :=
  sorry

-- Theorem for cube
theorem cube_can_be_threaded :
  ∃ (cubef : Frame) (h : Hole), isValidHole h ∧ canThreadThrough cubef h :=
sorry

-- Theorem for tetrahedron
theorem tetrahedron_can_be_threaded :
  ∃ (tetf : Frame) (h : Hole), isValidHole h ∧ canThreadThrough tetf h :=
sorry

end cube_can_be_threaded_tetrahedron_can_be_threaded_l14_1413


namespace max_product_863_l14_1473

/-- A type representing the digits we can use -/
inductive Digit
  | three
  | five
  | six
  | eight
  | nine

/-- A function to convert our Digit type to a natural number -/
def digit_to_nat (d : Digit) : Nat :=
  match d with
  | Digit.three => 3
  | Digit.five => 5
  | Digit.six => 6
  | Digit.eight => 8
  | Digit.nine => 9

/-- A type representing a valid combination of digits -/
structure DigitCombination where
  a : Digit
  b : Digit
  c : Digit
  d : Digit
  e : Digit
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

/-- Calculate the product of the three-digit and two-digit numbers -/
def calculate_product (combo : DigitCombination) : Nat :=
  (100 * digit_to_nat combo.a + 10 * digit_to_nat combo.b + digit_to_nat combo.c) *
  (10 * digit_to_nat combo.d + digit_to_nat combo.e)

/-- The main theorem -/
theorem max_product_863 :
  ∀ combo : DigitCombination,
    calculate_product combo ≤ calculate_product
      { a := Digit.eight
      , b := Digit.six
      , c := Digit.three
      , d := Digit.nine
      , e := Digit.five
      , all_different := by simp } :=
by
  sorry


end max_product_863_l14_1473


namespace cube_root_of_negative_eight_l14_1488

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end cube_root_of_negative_eight_l14_1488


namespace cos_alpha_plus_five_sixths_pi_l14_1401

theorem cos_alpha_plus_five_sixths_pi (α : Real) 
  (h : Real.sin (α + π / 3) = 1 / 4) : 
  Real.cos (α + 5 * π / 6) = -1 / 4 := by
  sorry

end cos_alpha_plus_five_sixths_pi_l14_1401


namespace speed_equivalence_l14_1410

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in meters per second -/
def speed_mps : ℝ := 18.334799999999998

/-- The speed in kilometers per hour -/
def speed_kmph : ℝ := 66.00528

/-- Theorem stating that the given speed in km/h is equivalent to the speed in m/s -/
theorem speed_equivalence : speed_kmph = speed_mps * mps_to_kmph := by
  sorry

end speed_equivalence_l14_1410


namespace unique_element_implies_a_value_l14_1403

def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

theorem unique_element_implies_a_value (a : ℝ) : (∃! x, x ∈ A a) → a = 0 ∨ a = 1 := by
  sorry

end unique_element_implies_a_value_l14_1403


namespace arithmetic_sequence_eighth_term_l14_1425

/-- 
Given an arithmetic sequence where:
- The first term is 2/3
- The second term is 1
- The third term is 4/3

Prove that the eighth term of this sequence is 3.
-/
theorem arithmetic_sequence_eighth_term : 
  ∀ (a : ℕ → ℚ), 
    (a 1 = 2/3) →
    (a 2 = 1) →
    (a 3 = 4/3) →
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →
    a 8 = 3 := by
  sorry

end arithmetic_sequence_eighth_term_l14_1425


namespace simplify_fraction_sum_powers_of_half_l14_1465

theorem simplify_fraction_sum_powers_of_half :
  1 / (1 / ((1/2)^0) + 1 / ((1/2)^1) + 1 / ((1/2)^2) + 1 / ((1/2)^3)) = 1 / 15 := by
  sorry

end simplify_fraction_sum_powers_of_half_l14_1465


namespace circle_ratio_l14_1499

theorem circle_ratio (r R a b : ℝ) (hr : r > 0) (hR : R > r) (ha : a > 0) (hb : b > 0) 
  (h : π * R^2 = (a / b) * (π * R^2 - π * r^2)) :
  R / r = Real.sqrt a / Real.sqrt (a - b) := by
sorry

end circle_ratio_l14_1499


namespace intersection_M_N_l14_1427

def M : Set ℝ := {x | (x - 3) / (x + 1) ≤ 0}
def N : Set ℝ := {-3, -1, 1, 3, 5}

theorem intersection_M_N : M ∩ N = {1, 3} := by
  sorry

end intersection_M_N_l14_1427


namespace jenny_mike_earnings_l14_1497

theorem jenny_mike_earnings (t : ℝ) : 
  (t + 3) * (4 * t - 6) = (4 * t - 7) * (t + 3) + 3 → t = 3 := by
  sorry

end jenny_mike_earnings_l14_1497


namespace average_team_goals_l14_1459

-- Define the average goals per game for each player
def carter_goals : ℚ := 4
def shelby_goals : ℚ := carter_goals / 2
def judah_goals : ℚ := 2 * shelby_goals - 3
def morgan_goals : ℚ := judah_goals + 1
def alex_goals : ℚ := carter_goals / 2 - 2
def taylor_goals : ℚ := 1 / 3

-- Define the total goals per game for the team
def team_goals : ℚ := carter_goals + shelby_goals + judah_goals + morgan_goals + alex_goals + taylor_goals

-- Theorem statement
theorem average_team_goals : team_goals = 28 / 3 := by
  sorry

end average_team_goals_l14_1459


namespace dividend_proof_l14_1481

theorem dividend_proof (divisor quotient dividend : ℕ) : 
  divisor = 12 → quotient = 999809 → dividend = 11997708 → 
  dividend / divisor = quotient ∧ dividend % divisor = 0 := by
  sorry

end dividend_proof_l14_1481


namespace height_on_hypotenuse_l14_1405

/-- Given a right triangle with legs of lengths 3 and 4, 
    the height on the hypotenuse is 12/5 -/
theorem height_on_hypotenuse (a b c h : ℝ) : 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → h * c = 2 * (a * b / 2) → h = 12/5 := by sorry

end height_on_hypotenuse_l14_1405


namespace price_difference_year_l14_1454

def price_P (n : ℕ) : ℚ := 420/100 + 40/100 * n
def price_Q (n : ℕ) : ℚ := 630/100 + 15/100 * n

theorem price_difference_year : 
  ∃ n : ℕ, price_P n = price_Q n + 40/100 ∧ n = 10 :=
sorry

end price_difference_year_l14_1454


namespace sin_cos_identity_l14_1430

theorem sin_cos_identity : 
  (4 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180)) / Real.cos (20 * π / 180) - Real.tan (20 * π / 180) = Real.sqrt 3 := by
  sorry

end sin_cos_identity_l14_1430


namespace sum_of_two_smallest_prime_factors_of_294_l14_1474

theorem sum_of_two_smallest_prime_factors_of_294 :
  ∃ (p q : ℕ), 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p < q ∧
    p ∣ 294 ∧ 
    q ∣ 294 ∧
    (∀ (r : ℕ), Nat.Prime r → r ∣ 294 → r = p ∨ r ≥ q) ∧
    p + q = 5 :=
by sorry

end sum_of_two_smallest_prime_factors_of_294_l14_1474


namespace total_stars_l14_1485

theorem total_stars (num_students : ℕ) (stars_per_student : ℕ) 
  (h1 : num_students = 186) 
  (h2 : stars_per_student = 5) : 
  num_students * stars_per_student = 930 := by
  sorry

end total_stars_l14_1485


namespace equation_solutions_l14_1433

theorem equation_solutions : 
  ∃! (s : Set ℝ), (∀ x ∈ s, |x - 2| = |x - 5| + |x - 8|) ∧ s = {5, 11} := by
sorry

end equation_solutions_l14_1433


namespace max_product_value_l14_1492

/-- An arithmetic sequence with specific conditions -/
structure ArithmeticSequence where
  a : ℕ+ → ℝ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_condition1 : a 1 + a 3 = 30
  sum_condition2 : a 2 + a 4 = 10

/-- The product of the first n terms of the sequence -/
def product (seq : ArithmeticSequence) (n : ℕ+) : ℝ :=
  (Finset.range n).prod (fun i => seq.a ⟨i + 1, Nat.succ_pos i⟩)

/-- The theorem stating the maximum value of the product -/
theorem max_product_value (seq : ArithmeticSequence) :
  ∃ max_val : ℝ, max_val = 729 ∧ ∀ n : ℕ+, product seq n ≤ max_val :=
sorry

end max_product_value_l14_1492


namespace B_subset_A_l14_1469

def A : Set ℕ := {2, 0, 3}
def B : Set ℕ := {2, 3}

theorem B_subset_A : B ⊆ A := by sorry

end B_subset_A_l14_1469


namespace sum_of_odd_function_at_specific_points_l14_1404

/-- A function is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The sum of v(-3.14), v(-1.57), v(1.57), and v(3.14) is zero for any odd function v -/
theorem sum_of_odd_function_at_specific_points (v : ℝ → ℝ) (h : IsOdd v) :
  v (-3.14) + v (-1.57) + v 1.57 + v 3.14 = 0 := by
  sorry


end sum_of_odd_function_at_specific_points_l14_1404


namespace integer_solutions_of_inequality_system_l14_1446

theorem integer_solutions_of_inequality_system :
  {x : ℤ | x + 2 > 0 ∧ 2 * x - 1 ≤ 0} = {-1, 0} := by
sorry

end integer_solutions_of_inequality_system_l14_1446


namespace min_games_correct_l14_1482

/-- Represents a chess tournament between two schools -/
structure ChessTournament where
  white_rook_students : ℕ
  black_elephant_students : ℕ
  games_per_white_student : ℕ
  total_games : ℕ

/-- The specific tournament described in the problem -/
def tournament : ChessTournament :=
  { white_rook_students := 15
  , black_elephant_students := 20
  , games_per_white_student := 20
  , total_games := 300 }

/-- The minimum number of games after which one can guarantee
    that at least one White Rook student has played all their games -/
def min_games_for_guarantee (t : ChessTournament) : ℕ :=
  (t.white_rook_students - 1) * t.games_per_white_student

theorem min_games_correct (t : ChessTournament) :
  min_games_for_guarantee t = (t.white_rook_students - 1) * t.games_per_white_student ∧
  min_games_for_guarantee t < t.total_games ∧
  ∀ n, n < min_games_for_guarantee t → 
    ∃ i j, i < t.white_rook_students ∧ j < t.games_per_white_student ∧
           n < i * t.games_per_white_student + j :=
by sorry

#eval min_games_for_guarantee tournament  -- Should output 280

end min_games_correct_l14_1482


namespace non_adjacent_book_arrangements_l14_1419

/-- Represents the number of books of each subject -/
structure BookCounts where
  chinese : Nat
  math : Nat
  physics : Nat

/-- Calculates the total number of books -/
def totalBooks (counts : BookCounts) : Nat :=
  counts.chinese + counts.math + counts.physics

/-- Calculates the number of permutations of n items -/
def permutations (n : Nat) : Nat :=
  Nat.factorial n

/-- Calculates the number of arrangements where books of the same subject are not adjacent -/
def nonAdjacentArrangements (counts : BookCounts) : Nat :=
  let total := totalBooks counts
  let allArrangements := permutations total
  let chineseAdjacent := (permutations (total - counts.chinese + 1)) * (permutations counts.chinese)
  let mathAdjacent := (permutations (total - counts.math + 1)) * (permutations counts.math)
  let bothAdjacent := (permutations (total - counts.chinese - counts.math + 2)) * 
                      (permutations counts.chinese) * (permutations counts.math)
  allArrangements - chineseAdjacent - mathAdjacent + bothAdjacent

theorem non_adjacent_book_arrangements :
  let counts : BookCounts := { chinese := 2, math := 2, physics := 1 }
  nonAdjacentArrangements counts = 48 := by
  sorry

end non_adjacent_book_arrangements_l14_1419


namespace salary_D_value_l14_1451

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_E : ℕ := 9000
def average_salary : ℕ := 8000
def num_people : ℕ := 5

theorem salary_D_value :
  ∃ (salary_D : ℕ),
    (salary_A + salary_B + salary_C + salary_D + salary_E) / num_people = average_salary ∧
    salary_D = 9000 := by
  sorry

end salary_D_value_l14_1451


namespace quadratic_equal_roots_l14_1453

/-- 
Given a quadratic equation x^2 + bx + 4 = 0 with two equal real roots,
prove that b = 4 or b = -4.
-/
theorem quadratic_equal_roots (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 4 = 0 ∧ 
   ∀ y : ℝ, y^2 + b*y + 4 = 0 → y = x) → 
  b = 4 ∨ b = -4 := by
sorry

end quadratic_equal_roots_l14_1453


namespace divisibility_of_7386038_l14_1444

theorem divisibility_of_7386038 : ∃ (k : ℕ), 7386038 = 7 * k := by sorry

end divisibility_of_7386038_l14_1444


namespace paulson_spending_percentage_l14_1478

theorem paulson_spending_percentage 
  (income_increase : Real) 
  (expenditure_increase : Real) 
  (savings_increase : Real) : 
  income_increase = 0.20 → 
  expenditure_increase = 0.10 → 
  savings_increase = 0.50 → 
  ∃ (original_income : Real) (spending_percentage : Real),
    spending_percentage = 0.75 ∧ 
    original_income > 0 ∧
    (1 + income_increase) * original_income - 
    (1 + expenditure_increase) * spending_percentage * original_income = 
    (1 + savings_increase) * (original_income - spending_percentage * original_income) :=
by sorry

end paulson_spending_percentage_l14_1478


namespace unique_prime_digit_l14_1428

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

/-- The six-digit number as a function of B -/
def number (B : ℕ) : ℕ := 304200 + B

/-- Theorem stating that there is a unique B that makes the number prime, and it's 1 -/
theorem unique_prime_digit :
  ∃! B : ℕ, B < 10 ∧ isPrime (number B) ∧ B = 1 :=
sorry

end unique_prime_digit_l14_1428


namespace factors_lcm_gcd_of_24_60_180_l14_1431

def numbers : List Nat := [24, 60, 180]

theorem factors_lcm_gcd_of_24_60_180 :
  (∃ (common_factors : List Nat), common_factors.length = 6 ∧ 
    ∀ n ∈ common_factors, ∀ m ∈ numbers, n ∣ m) ∧
  Nat.lcm 24 (Nat.lcm 60 180) = 180 ∧
  Nat.gcd 24 (Nat.gcd 60 180) = 12 := by
  sorry

end factors_lcm_gcd_of_24_60_180_l14_1431


namespace bennetts_brothers_l14_1438

theorem bennetts_brothers (aaron_brothers : ℕ) (bennett_brothers : ℕ) : 
  aaron_brothers = 4 → 
  bennett_brothers = 2 * aaron_brothers - 2 → 
  bennett_brothers = 6 := by
sorry

end bennetts_brothers_l14_1438


namespace distance_B_to_x_axis_l14_1470

def point_B : ℝ × ℝ := (2, -3)

def distance_to_x_axis (p : ℝ × ℝ) : ℝ := |p.2|

theorem distance_B_to_x_axis :
  distance_to_x_axis point_B = 3 := by
  sorry

end distance_B_to_x_axis_l14_1470


namespace problem_solution_l14_1462

theorem problem_solution (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 3) (hy : 3 ≤ y ∧ y ≤ 5) :
  (4 ≤ x + y ∧ x + y ≤ 8) ∧
  (∀ a b, (1 ≤ a ∧ a ≤ 3 ∧ 3 ≤ b ∧ b ≤ 5) → x + y + 1/x + 16/y ≤ a + b + 1/a + 16/b) ∧
  (∃ a b, (1 ≤ a ∧ a ≤ 3 ∧ 3 ≤ b ∧ b ≤ 5) ∧ x + y + 1/x + 16/y = a + b + 1/a + 16/b ∧ a + b + 1/a + 16/b = 10) :=
by
  sorry


end problem_solution_l14_1462


namespace line_contains_point_l14_1420

/-- Given a line equation 2 - kx = -4y that contains the point (2, -1), prove that k = -1 -/
theorem line_contains_point (k : ℝ) : 
  (2 - k * 2 = -4 * (-1)) → k = -1 := by
  sorry

end line_contains_point_l14_1420


namespace quadrilateral_interior_angles_mean_l14_1450

/-- The mean value of the measures of the four interior angles of any quadrilateral is 90°. -/
theorem quadrilateral_interior_angles_mean :
  let sum_of_angles : ℝ := 360
  let number_of_angles : ℕ := 4
  (sum_of_angles / number_of_angles : ℝ) = 90 := by
  sorry

end quadrilateral_interior_angles_mean_l14_1450


namespace total_distance_four_runners_l14_1458

/-- The total distance run by four runners, where one runner ran 51 miles
    and the other three ran the same distance of 48 miles each, is 195 miles. -/
theorem total_distance_four_runners :
  ∀ (katarina tomas tyler harriet : ℕ),
    katarina = 51 →
    tomas = 48 →
    tyler = 48 →
    harriet = 48 →
    katarina + tomas + tyler + harriet = 195 :=
by
  sorry

end total_distance_four_runners_l14_1458


namespace orange_count_l14_1489

/-- Given a fruit farm that packs oranges, calculate the total number of oranges. -/
theorem orange_count (oranges_per_box : ℝ) (boxes_per_day : ℝ) 
  (h1 : oranges_per_box = 10.0) 
  (h2 : boxes_per_day = 2650.0) : 
  oranges_per_box * boxes_per_day = 26500.0 := by
  sorry

#check orange_count

end orange_count_l14_1489


namespace travel_time_difference_l14_1424

/-- Represents the travel times for different modes of transportation --/
structure TravelTimes where
  drivingTimeMinutes : ℕ
  driveToAirportMinutes : ℕ
  waitToBoardMinutes : ℕ
  exitPlaneMinutes : ℕ

/-- Calculates the total airplane travel time --/
def airplaneTravelTime (t : TravelTimes) : ℕ :=
  t.driveToAirportMinutes + t.waitToBoardMinutes + (t.drivingTimeMinutes / 3) + t.exitPlaneMinutes

/-- Theorem stating the time difference between driving and flying --/
theorem travel_time_difference (t : TravelTimes) 
  (h1 : t.drivingTimeMinutes = 195)
  (h2 : t.driveToAirportMinutes = 10)
  (h3 : t.waitToBoardMinutes = 20)
  (h4 : t.exitPlaneMinutes = 10) :
  t.drivingTimeMinutes - airplaneTravelTime t = 90 := by
  sorry


end travel_time_difference_l14_1424


namespace selection_theorem_l14_1436

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of girls -/
def num_girls : ℕ := 5

/-- The total number of boys -/
def num_boys : ℕ := 7

/-- The number of representatives to be selected -/
def num_representatives : ℕ := 5

theorem selection_theorem :
  /- At least one girl is selected -/
  (choose (num_girls + num_boys) num_representatives - choose num_boys num_representatives = 771) ∧
  /- Boy A and Girl B are selected -/
  (choose (num_girls + num_boys - 2) (num_representatives - 2) = 120) ∧
  /- At least one of Boy A or Girl B is selected -/
  (choose (num_girls + num_boys) num_representatives - choose (num_girls + num_boys - 2) num_representatives = 540) :=
by sorry

end selection_theorem_l14_1436


namespace auction_price_problem_l14_1416

theorem auction_price_problem (tv_initial_cost : ℝ) (tv_price_increase_ratio : ℝ) 
  (phone_price_increase_ratio : ℝ) (total_received : ℝ) :
  tv_initial_cost = 500 →
  tv_price_increase_ratio = 2 / 5 →
  phone_price_increase_ratio = 0.4 →
  total_received = 1260 →
  ∃ (phone_initial_price : ℝ),
    phone_initial_price = 400 ∧
    total_received = tv_initial_cost * (1 + tv_price_increase_ratio) + 
                     phone_initial_price * (1 + phone_price_increase_ratio) :=
by sorry

end auction_price_problem_l14_1416


namespace lizas_rent_calculation_l14_1464

def initial_balance : ℚ := 800
def paycheck : ℚ := 1500
def electricity_bill : ℚ := 117
def internet_bill : ℚ := 100
def phone_bill : ℚ := 70
def final_balance : ℚ := 1563

theorem lizas_rent_calculation :
  ∃ (rent : ℚ), 
    initial_balance - rent + paycheck - electricity_bill - internet_bill - phone_bill = final_balance ∧
    rent = 450 :=
by sorry

end lizas_rent_calculation_l14_1464


namespace binary_rep_156_ones_minus_zeros_eq_zero_l14_1484

/-- Converts a natural number to its binary representation as a list of booleans -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Counts the number of true values in a list of booleans -/
def countOnes (l : List Bool) : ℕ :=
  l.filter id |>.length

/-- Counts the number of false values in a list of booleans -/
def countZeros (l : List Bool) : ℕ :=
  l.filter not |>.length

theorem binary_rep_156_ones_minus_zeros_eq_zero :
  let binary := toBinary 156
  let y := countOnes binary
  let x := countZeros binary
  y - x = 0 := by sorry

end binary_rep_156_ones_minus_zeros_eq_zero_l14_1484


namespace roots_of_equation_l14_1483

theorem roots_of_equation (x : ℝ) : 
  (x + 2)^2 = 8 ↔ x = 2 * Real.sqrt 2 - 2 ∨ x = -2 * Real.sqrt 2 - 2 := by
  sorry

end roots_of_equation_l14_1483


namespace students_over_capacity_l14_1495

/-- Calculates the number of students over capacity given the initial conditions --/
theorem students_over_capacity
  (ratio : ℚ)
  (teachers : ℕ)
  (increase_percent : ℚ)
  (capacity : ℕ)
  (h_ratio : ratio = 27.5)
  (h_teachers : teachers = 42)
  (h_increase : increase_percent = 0.15)
  (h_capacity : capacity = 1300) :
  ⌊(ratio * teachers) * (1 + increase_percent)⌋ - capacity = 28 :=
by sorry

end students_over_capacity_l14_1495


namespace probability_smaller_divides_larger_l14_1423

def S : Finset ℕ := {1, 2, 3, 6, 9}

def divides_pairs (S : Finset ℕ) : Finset (ℕ × ℕ) :=
  S.product S |>.filter (fun p => p.1 < p.2 ∧ p.2 % p.1 = 0)

theorem probability_smaller_divides_larger :
  (divides_pairs S).card / (S.product S |>.filter (fun p => p.1 ≠ p.2)).card = 3 / 5 := by
  sorry

end probability_smaller_divides_larger_l14_1423


namespace problem_statements_l14_1415

theorem problem_statements (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (ab ≤ 1 → 1/a + 1/b ≥ 2) ∧
  (a + b = 4 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 4 ∧ 1/x + 9/y ≤ 1/a + 9/b ∧ 1/a + 9/b = 4) ∧
  (a^2 + b^2 = 4 → ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = 4 → x*y ≤ a*b ∧ a*b = 2) ∧
  ¬(2*a + b = 1 → ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2*x + y = 1 → x*y ≤ a*b ∧ a*b = Real.sqrt 2 / 2) :=
by sorry

end problem_statements_l14_1415


namespace digit_addition_subtraction_problem_l14_1480

/- Define digits as natural numbers from 0 to 9 -/
def Digit := {n : ℕ // n ≤ 9}

/- Define a function to convert a two-digit number to its value -/
def twoDigitValue (tens : Digit) (ones : Digit) : ℕ := 10 * tens.val + ones.val

theorem digit_addition_subtraction_problem (A B C D : Digit) :
  (twoDigitValue A B + twoDigitValue C A = twoDigitValue D A) ∧
  (twoDigitValue A B - twoDigitValue C A = A.val) →
  D.val = 9 := by
  sorry

end digit_addition_subtraction_problem_l14_1480


namespace parallelogram_bisector_l14_1448

-- Define the parallelogram
def parallelogram : List (ℝ × ℝ) := [(5, 25), (5, 50), (14, 58), (14, 33)]

-- Define the property of the line
def divides_equally (m n : ℕ) : Prop :=
  let slope := m / n
  ∃ (b : ℝ), 
    (25 + b) / 5 = (58 - b) / 14 ∧ 
    (25 + b) / 5 = slope ∧
    (b > -25 ∧ b < 33)  -- Ensure the line intersects the parallelogram

-- Main theorem
theorem parallelogram_bisector :
  ∃ (m n : ℕ), 
    m.Coprime n ∧
    divides_equally m n ∧
    m = 71 ∧ n = 19 ∧
    m + n = 90 := by sorry

end parallelogram_bisector_l14_1448


namespace even_function_sum_l14_1466

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x

-- Define the property of being an even function
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (-2*a) (3*a - 1), f b x = f b (-x)) →
  a + b = 1 :=
by sorry

end even_function_sum_l14_1466


namespace max_value_2x_plus_y_l14_1447

theorem max_value_2x_plus_y (x y : ℝ) (h : 4 * x^2 + y^2 + x*y = 5) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ ∀ (z : ℝ), 2 * x + y ≤ z → z ≤ M :=
sorry

end max_value_2x_plus_y_l14_1447


namespace largest_c_for_no_integer_in_interval_l14_1412

theorem largest_c_for_no_integer_in_interval :
  ∃ (c : ℝ), c = 6 - 4 * Real.sqrt 2 ∧
  (∀ (n : ℕ), ∀ (k : ℤ),
    (n : ℝ) * Real.sqrt 2 - c / (n : ℝ) < (k : ℝ) →
    (k : ℝ) < (n : ℝ) * Real.sqrt 2 + c / (n : ℝ)) ∧
  (∀ (c' : ℝ), c' > c →
    ∃ (n : ℕ), ∃ (k : ℤ),
      (n : ℝ) * Real.sqrt 2 - c' / (n : ℝ) ≤ (k : ℝ) ∧
      (k : ℝ) ≤ (n : ℝ) * Real.sqrt 2 + c' / (n : ℝ)) :=
sorry

end largest_c_for_no_integer_in_interval_l14_1412


namespace vector_properties_l14_1471

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-2, 1]
def c : Fin 2 → ℝ := ![3, -5]

theorem vector_properties :
  (∃ (k : ℝ), a + 2 • b = k • c) ∧
  ‖a + c‖ = 2 * ‖b‖ := by
sorry

end vector_properties_l14_1471


namespace jelly_servings_count_jelly_servings_mixed_number_l14_1432

-- Define the total amount of jelly in tablespoons
def total_jelly : ℚ := 113 / 3

-- Define the serving size in tablespoons
def serving_size : ℚ := 3 / 2

-- Define the number of servings
def num_servings : ℚ := total_jelly / serving_size

-- Theorem to prove
theorem jelly_servings_count :
  num_servings = 226 / 9 := by sorry

-- Proof that the result is equivalent to 25 1/9
theorem jelly_servings_mixed_number :
  ∃ (n : ℕ) (m : ℚ), n = 25 ∧ m = 1 / 9 ∧ num_servings = n + m := by sorry

end jelly_servings_count_jelly_servings_mixed_number_l14_1432


namespace metal_waste_problem_l14_1493

/-- Given a rectangle with length twice its width, prove that the area wasted
    when cutting out a maximum circular piece and then a maximum square piece
    from that circle is 3/2 of the original rectangle's area. -/
theorem metal_waste_problem (w : ℝ) (hw : w > 0) :
  let rectangle_area := 2 * w^2
  let circle_area := π * (w/2)^2
  let square_area := (w * Real.sqrt 2 / 2)^2
  let waste_area := rectangle_area - square_area
  waste_area = (3/2) * rectangle_area := by
  sorry

end metal_waste_problem_l14_1493


namespace inequality_proof_l14_1442

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 1 ≥ x*y + x + y := by
sorry

end inequality_proof_l14_1442


namespace spectators_count_l14_1487

/-- The number of wristbands distributed -/
def total_wristbands : ℕ := 250

/-- The number of wristbands each person received -/
def wristbands_per_person : ℕ := 2

/-- The number of people who watched the game -/
def spectators : ℕ := total_wristbands / wristbands_per_person

theorem spectators_count : spectators = 125 := by
  sorry

end spectators_count_l14_1487


namespace square_2209_product_l14_1409

theorem square_2209_product (x : ℤ) (h : x^2 = 2209) : (x + 2) * (x - 2) = 2205 := by
  sorry

end square_2209_product_l14_1409


namespace speed_ratio_l14_1437

-- Define the speeds of A and B
def speed_A : ℝ := sorry
def speed_B : ℝ := sorry

-- Define the initial position of B
def initial_B_position : ℝ := 600

-- Define the time when they are first equidistant
def first_equidistant_time : ℝ := 3

-- Define the time when they are second equidistant
def second_equidistant_time : ℝ := 12

-- Define the condition for being equidistant at the first time
def first_equidistant_condition : Prop :=
  (first_equidistant_time * speed_A) = abs (-initial_B_position + first_equidistant_time * speed_B)

-- Define the condition for being equidistant at the second time
def second_equidistant_condition : Prop :=
  (second_equidistant_time * speed_A) = abs (-initial_B_position + second_equidistant_time * speed_B)

-- Theorem stating that the ratio of speeds is 1:5
theorem speed_ratio : 
  first_equidistant_condition → second_equidistant_condition → speed_A / speed_B = 1 / 5 := by sorry

end speed_ratio_l14_1437


namespace barbier_theorem_for_delta_curves_l14_1445

-- Define a Δ-curve
class DeltaCurve where
  height : ℝ
  is_convex : Bool
  can_rotate_in_triangle : Bool
  always_touches_sides : Bool

-- Define the length of a Δ-curve
def length_of_delta_curve (K : DeltaCurve) : ℝ := sorry

-- Define the approximation of a Δ-curve by circular arcs
def approximate_by_circular_arcs (K : DeltaCurve) (n : ℕ) : DeltaCurve := sorry

-- Theorem: The length of any Δ-curve with height h is 2πh/3
theorem barbier_theorem_for_delta_curves (K : DeltaCurve) :
  length_of_delta_curve K = 2 * Real.pi * K.height / 3 := by sorry

end barbier_theorem_for_delta_curves_l14_1445


namespace cake_slices_problem_l14_1414

theorem cake_slices_problem (num_cakes : ℕ) (price_per_slice donation1 donation2 total_raised : ℚ) :
  num_cakes = 10 →
  price_per_slice = 1 →
  donation1 = 1/2 →
  donation2 = 1/4 →
  total_raised = 140 →
  ∃ (slices_per_cake : ℕ), 
    slices_per_cake = 8 ∧
    (num_cakes * slices_per_cake : ℚ) * (price_per_slice + donation1 + donation2) = total_raised :=
by sorry

end cake_slices_problem_l14_1414


namespace inverse_of_matrix_A_l14_1449

theorem inverse_of_matrix_A :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; 1, 2]
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![1, -2; -1/2, 3/2]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end inverse_of_matrix_A_l14_1449


namespace lotus_growth_model_l14_1434

def y (x : ℕ) : ℚ := (32 / 3) * (3 / 2) ^ x

theorem lotus_growth_model :
  (y 2 = 24) ∧ 
  (y 3 = 36) ∧ 
  (∀ n : ℕ, y n ≤ 10 * y 0 → n ≤ 5) ∧
  (y 6 > 10 * y 0) := by
  sorry

end lotus_growth_model_l14_1434


namespace farmer_land_problem_l14_1418

theorem farmer_land_problem (original_land : ℚ) : 
  (9 / 10 : ℚ) * original_land = 10 → original_land = 11 + 1 / 9 := by
  sorry

end farmer_land_problem_l14_1418


namespace triangle_abc_properties_l14_1435

/-- Triangle ABC with specific properties -/
structure TriangleABC where
  -- Sides opposite to angles A, B, C
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles A, B, C
  A : ℝ
  B : ℝ
  C : ℝ
  -- Given conditions
  side_angle_relation : (2 * a - b) * Real.cos C = c * Real.cos B
  c_value : c = 2
  area : (1/2) * a * b * Real.sin C = Real.sqrt 3
  -- Triangle properties
  angle_sum : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

/-- Main theorem about the properties of Triangle ABC -/
theorem triangle_abc_properties (t : TriangleABC) : 
  t.C = Real.pi / 3 ∧ t.a + t.b + t.c = 6 := by
  sorry


end triangle_abc_properties_l14_1435


namespace f_of_3_equals_0_l14_1490

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2 - 2*(x - 1)

-- State the theorem
theorem f_of_3_equals_0 : f 3 = 0 := by
  sorry

end f_of_3_equals_0_l14_1490


namespace perpendicular_planes_from_parallel_lines_l14_1441

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes_from_parallel_lines
  (l m : Line) (α β : Plane)
  (h1 : perpendicular_line_plane l α)
  (h2 : contained_in m β)
  (h3 : parallel_lines l m) :
  perpendicular_planes α β :=
sorry

end perpendicular_planes_from_parallel_lines_l14_1441


namespace no_sum_of_three_different_squares_128_l14_1460

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def sum_of_three_different_squares (n : ℕ) : Prop :=
  ∃ a b c : ℕ, 
    is_perfect_square a ∧ 
    is_perfect_square b ∧ 
    is_perfect_square c ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = n

theorem no_sum_of_three_different_squares_128 : 
  ¬(sum_of_three_different_squares 128) := by
  sorry

end no_sum_of_three_different_squares_128_l14_1460


namespace pencil_arrangement_theorem_l14_1407

def yellow_pencils : ℕ := 6
def red_pencils : ℕ := 3
def blue_pencils : ℕ := 4

def total_pencils : ℕ := yellow_pencils + red_pencils + blue_pencils

def total_arrangements : ℕ := Nat.factorial total_pencils / (Nat.factorial yellow_pencils * Nat.factorial red_pencils * Nat.factorial blue_pencils)

def arrangements_with_adjacent_blue : ℕ := Nat.factorial (total_pencils - blue_pencils + 1) / (Nat.factorial yellow_pencils * Nat.factorial red_pencils)

theorem pencil_arrangement_theorem :
  total_arrangements - arrangements_with_adjacent_blue = 274400 := by
  sorry

end pencil_arrangement_theorem_l14_1407


namespace exponential_equation_solution_l14_1456

theorem exponential_equation_solution :
  ∃ y : ℝ, (3 : ℝ) ^ (y - 4) = 9 ^ (y + 2) → y = -8 := by
  sorry

end exponential_equation_solution_l14_1456


namespace range_of_m_l14_1439

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 2 4 ∧ x^2 - 2*x - 2 - m < 0) → m > -2 := by
  sorry

end range_of_m_l14_1439


namespace max_value_of_f_l14_1426

/-- The quadratic function f(x) = -2x^2 + 16x - 14 -/
def f (x : ℝ) : ℝ := -2 * x^2 + 16 * x - 14

/-- Theorem: The maximum value of f(x) = -2x^2 + 16x - 14 is -14 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = -14 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end max_value_of_f_l14_1426


namespace integer_fraction_theorem_l14_1422

def is_valid_pair (a b : ℕ) : Prop :=
  a ≥ 1 ∧ b ≥ 1 ∧
  (∃ (k₁ k₂ : ℤ), (a^2 + b : ℤ) = k₁ * (b^2 - a) ∧ (b^2 + a : ℤ) = k₂ * (a^2 - b))

def solution_set : Set (ℕ × ℕ) :=
  {(2,2), (3,3), (1,2), (2,1), (2,3), (3,2)}

theorem integer_fraction_theorem :
  ∀ (a b : ℕ), is_valid_pair a b ↔ (a, b) ∈ solution_set := by sorry

end integer_fraction_theorem_l14_1422


namespace ellipse_max_value_l14_1486

/-- The maximum value of x + 2y for points on the ellipse x^2/16 + y^2/12 = 1 is 8 -/
theorem ellipse_max_value (x y : ℝ) : 
  x^2/16 + y^2/12 = 1 → x + 2*y ≤ 8 := by sorry

end ellipse_max_value_l14_1486


namespace initial_lot_cost_l14_1421

/-- Represents the cost and composition of a lot of tickets -/
structure TicketLot where
  firstClass : ℕ
  secondClass : ℕ
  firstClassCost : ℕ
  secondClassCost : ℕ

/-- Calculates the total cost of a ticket lot -/
def totalCost (lot : TicketLot) : ℕ :=
  lot.firstClass * lot.firstClassCost + lot.secondClass * lot.secondClassCost

/-- Theorem: The cost of the initial lot of tickets is 110 Rs -/
theorem initial_lot_cost (initialLot interchangedLot : TicketLot) : 
  initialLot.firstClass + initialLot.secondClass = 18 →
  initialLot.firstClassCost = 10 →
  initialLot.secondClassCost = 3 →
  interchangedLot.firstClass = initialLot.secondClass →
  interchangedLot.secondClass = initialLot.firstClass →
  interchangedLot.firstClassCost = initialLot.firstClassCost →
  interchangedLot.secondClassCost = initialLot.secondClassCost →
  totalCost interchangedLot = 124 →
  totalCost initialLot = 110 := by
  sorry

end initial_lot_cost_l14_1421


namespace total_fish_caught_l14_1402

def leo_fish : ℕ := 40
def agrey_fish : ℕ := leo_fish + 20

theorem total_fish_caught : leo_fish + agrey_fish = 100 := by
  sorry

end total_fish_caught_l14_1402


namespace gcd_n4_plus_16_and_n_plus_3_l14_1452

theorem gcd_n4_plus_16_and_n_plus_3 (n : ℕ) (h1 : n > 9) (h2 : n ≠ 94) :
  Nat.gcd (n^4 + 16) (n + 3) = 1 := by
  sorry

end gcd_n4_plus_16_and_n_plus_3_l14_1452


namespace triangle_abc_properties_l14_1479

theorem triangle_abc_properties (a b : ℝ) (A B C : ℝ) :
  -- Conditions
  (a^2 - 2 * Real.sqrt 3 * a + 2 = 0) →
  (b^2 - 2 * Real.sqrt 3 * b + 2 = 0) →
  (2 * Real.cos (A + B) = 1) →
  -- Conclusions
  (C = 120 * π / 180) ∧
  (Real.sqrt ((a^2 + b^2 + a*b) : ℝ) = Real.sqrt 10) :=
by sorry

end triangle_abc_properties_l14_1479


namespace nara_height_l14_1461

/-- Given the heights of Sangheon, Chiho, and Nara, prove Nara's height -/
theorem nara_height (sangheon_height : ℝ) (chiho_diff : ℝ) (nara_diff : ℝ) :
  sangheon_height = 1.56 →
  chiho_diff = 0.14 →
  nara_diff = 0.27 →
  sangheon_height - chiho_diff + nara_diff = 1.69 := by
sorry

end nara_height_l14_1461


namespace consecutive_squares_divisible_by_five_l14_1440

theorem consecutive_squares_divisible_by_five (n : ℤ) :
  ∃ k : ℤ, (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 5 * k := by
  sorry

end consecutive_squares_divisible_by_five_l14_1440


namespace sector_area_l14_1475

/-- Given a circular sector with perimeter 8 cm and central angle 2 radians, its area is 4 cm² -/
theorem sector_area (r : ℝ) (l : ℝ) : 
  l + 2 * r = 8 →  -- Perimeter condition
  l = 2 * r →      -- Arc length condition (derived from central angle)
  (1 / 2) * 2 * r^2 = 4 := by  -- Area calculation
sorry

end sector_area_l14_1475


namespace correct_calculation_l14_1463

theorem correct_calculation : 
  (Real.sqrt 27 / Real.sqrt 3 = 3) ∧ 
  (Real.sqrt 2 + Real.sqrt 3 ≠ Real.sqrt 5) ∧ 
  (5 * Real.sqrt 2 - 4 * Real.sqrt 2 ≠ 1) ∧ 
  (2 * Real.sqrt 3 * 3 * Real.sqrt 3 ≠ 6 * Real.sqrt 3) :=
by sorry


end correct_calculation_l14_1463


namespace cylinder_surface_area_l14_1467

/-- The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches. -/
theorem cylinder_surface_area : 
  let h : ℝ := 8
  let r : ℝ := 3
  let lateral_area := 2 * π * r * h
  let base_area := π * r^2
  let total_surface_area := lateral_area + 2 * base_area
  total_surface_area = 66 * π := by sorry

end cylinder_surface_area_l14_1467


namespace tangent_slope_values_l14_1429

-- Define the curve
def curve (x : ℝ) : ℝ := x^2

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 2 * x

-- Define the tangent line equation
def tangent_line (t : ℝ) (x : ℝ) : ℝ := t^2 + 2*t*(x - t)

-- Theorem statement
theorem tangent_slope_values :
  ∃ (t : ℝ), (tangent_line t 1 = 0) ∧ 
  ((curve_derivative t = 0) ∨ (curve_derivative t = 4)) :=
sorry

end tangent_slope_values_l14_1429


namespace system_of_equations_sum_l14_1491

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 5 * x + b * y + c * z = 0)
  (eq2 : a * x + 7 * y + c * z = 0)
  (eq3 : a * x + b * y + 9 * z = 0)
  (ha : a ≠ 5)
  (hx : x ≠ 0) :
  a / (a - 5) + b / (b - 7) + c / (c - 9) = 1 := by
  sorry

end system_of_equations_sum_l14_1491


namespace average_problem_l14_1496

theorem average_problem (y : ℝ) : 
  (15 + 25 + 35 + y) / 4 = 30 → y = 45 := by
sorry

end average_problem_l14_1496


namespace lines_parallel_iff_l14_1477

-- Define the lines
def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y + 2 * a = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop := ∀ (x y : ℝ), line1 a x y ↔ ∃ (k : ℝ), line2 a (x + k) (y + k)

-- Theorem statement
theorem lines_parallel_iff : ∀ (a : ℝ), parallel a ↔ a = -1 := by sorry

end lines_parallel_iff_l14_1477


namespace right_triangle_side_length_l14_1406

theorem right_triangle_side_length 
  (A B C : Real) 
  (BC : Real) 
  (h1 : A = Real.pi / 2) 
  (h2 : BC = 10) 
  (h3 : Real.tan C = 3 * Real.cos B) : 
  ∃ AB : Real, AB = 20 * Real.sqrt 2 / 3 := by
  sorry

end right_triangle_side_length_l14_1406


namespace cosine_of_angle_between_vectors_l14_1498

/-- Given two planar vectors a and b satisfying certain conditions, 
    prove that the cosine of the angle between them is -√10/10 -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) : 
  (2 • a + b = (3, 3)) → 
  (a - 2 • b = (-1, 4)) → 
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = -Real.sqrt 10 / 10 := by
sorry

end cosine_of_angle_between_vectors_l14_1498


namespace zeros_when_a_1_b_neg_2_range_of_a_for_two_distinct_zeros_l14_1443

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + b - 1

-- Theorem for part (1)
theorem zeros_when_a_1_b_neg_2 :
  let f := f 1 (-2)
  ∀ x, f x = 0 ↔ x = 3 ∨ x = -1 := by sorry

-- Theorem for part (2)
theorem range_of_a_for_two_distinct_zeros :
  ∀ a : ℝ, (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) ↔ 0 < a ∧ a < 1 := by sorry

end zeros_when_a_1_b_neg_2_range_of_a_for_two_distinct_zeros_l14_1443


namespace tan_pi_4_minus_theta_l14_1400

theorem tan_pi_4_minus_theta (θ : Real) 
  (h1 : θ > -π/2 ∧ θ < 0) 
  (h2 : Real.cos (2*θ) - 3*Real.sin (θ - π/2) = 1) : 
  Real.tan (π/4 - θ) = -2 - Real.sqrt 3 := by
  sorry

end tan_pi_4_minus_theta_l14_1400
