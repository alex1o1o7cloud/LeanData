import Mathlib

namespace NUMINAMATH_CALUDE_total_money_l2091_209151

/-- Given that A and C together have 200, B and C together have 350, and C has 200,
    prove that the total amount of money A, B, and C have between them is 350. -/
theorem total_money (A B C : ℕ) 
  (hAC : A + C = 200)
  (hBC : B + C = 350)
  (hC : C = 200) : 
  A + B + C = 350 := by
sorry

end NUMINAMATH_CALUDE_total_money_l2091_209151


namespace NUMINAMATH_CALUDE_triangle_properties_l2091_209129

/-- Given a triangle ABC with angle C = π/4 and the relation 2sin²A - 1 = sin²B,
    prove that tan B = 2 and if side b = 1, the area of the triangle is 3/8 -/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  C = π/4 →
  2 * Real.sin A ^ 2 - 1 = Real.sin B ^ 2 →
  Real.tan B = 2 ∧
  (b = 1 → (1/2) * a * b * Real.sin C = 3/8) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2091_209129


namespace NUMINAMATH_CALUDE_angle_calculations_l2091_209136

theorem angle_calculations (α : Real) (h : Real.tan α = -3/7) :
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -3/7 ∧
  2 + Real.sin α * Real.cos α - Real.cos α ^ 2 = 23/29 := by
sorry

end NUMINAMATH_CALUDE_angle_calculations_l2091_209136


namespace NUMINAMATH_CALUDE_davis_items_left_l2091_209124

/-- The number of items Miss Davis has left after distributing popsicle sticks and straws --/
def items_left (popsicle_sticks_per_group : ℕ) (straws_per_group : ℕ) (num_groups : ℕ) (total_items : ℕ) : ℕ :=
  total_items - (popsicle_sticks_per_group + straws_per_group) * num_groups

/-- Theorem stating that Miss Davis has 150 items left --/
theorem davis_items_left :
  items_left 15 20 10 500 = 150 := by
  sorry

end NUMINAMATH_CALUDE_davis_items_left_l2091_209124


namespace NUMINAMATH_CALUDE_parabola_equation_l2091_209131

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 9 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, -3)

-- Define the parabola properties
structure Parabola where
  -- The coordinate axes are the axes of symmetry
  symmetry_axes : Prop
  -- The origin is the vertex
  vertex_at_origin : Prop
  -- The parabola passes through the center of the circle
  passes_through_center : Prop

-- Theorem statement
theorem parabola_equation (p : Parabola) :
  (∀ x y : ℝ, y^2 = 9*x) ∨ (∀ x y : ℝ, x^2 = -1/3*y) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l2091_209131


namespace NUMINAMATH_CALUDE_number_difference_l2091_209180

theorem number_difference (L S : ℕ) (h1 : L > S) (h2 : L = 1650) (h3 : L = 5 * S + 5) : L - S = 1321 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2091_209180


namespace NUMINAMATH_CALUDE_sports_conference_games_l2091_209188

/-- The number of games in a sports conference season --/
def conference_games (total_teams : ℕ) (teams_per_division : ℕ) (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let divisions := total_teams / teams_per_division
  let intra_division_total := divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games
  let inter_division_total := (total_teams * (total_teams - teams_per_division) / 2) * inter_division_games
  intra_division_total + inter_division_total

theorem sports_conference_games :
  conference_games 16 8 3 2 = 296 := by
  sorry

end NUMINAMATH_CALUDE_sports_conference_games_l2091_209188


namespace NUMINAMATH_CALUDE_intersection_M_N_l2091_209168

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 2 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2, -1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2091_209168


namespace NUMINAMATH_CALUDE_cheetah_catches_deer_l2091_209190

/-- Proves that a cheetah catches up with a deer in 10 minutes given specific conditions -/
theorem cheetah_catches_deer (deer_speed cheetah_speed : ℝ) 
  (time_difference : ℝ) (catch_up_time : ℝ) : 
  deer_speed = 50 → 
  cheetah_speed = 60 → 
  time_difference = 2 / 60 → 
  (deer_speed * time_difference) / (cheetah_speed - deer_speed) = catch_up_time →
  catch_up_time = 1 / 6 := by
  sorry

#check cheetah_catches_deer

end NUMINAMATH_CALUDE_cheetah_catches_deer_l2091_209190


namespace NUMINAMATH_CALUDE_solution_set_exponential_inequality_l2091_209121

theorem solution_set_exponential_inequality :
  ∀ x : ℝ, (2 : ℝ) ^ (x^2 - 5*x + 5) > (1/2 : ℝ) ↔ x < 2 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_exponential_inequality_l2091_209121


namespace NUMINAMATH_CALUDE_book_price_increase_percentage_l2091_209145

theorem book_price_increase_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 300)
  (h2 : new_price = 480) :
  (new_price - original_price) / original_price * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_book_price_increase_percentage_l2091_209145


namespace NUMINAMATH_CALUDE_brick_count_for_wall_l2091_209159

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  length * width * height

/-- Converts meters to centimeters -/
def meters_to_cm (m : ℝ) : ℝ :=
  m * 100

theorem brick_count_for_wall :
  let brick_length : ℝ := 20
  let brick_width : ℝ := 10
  let brick_height : ℝ := 7.5
  let wall_length : ℝ := 27
  let wall_width : ℝ := 2
  let wall_height : ℝ := 0.75
  let brick_volume : ℝ := volume brick_length brick_width brick_height
  let wall_volume : ℝ := volume (meters_to_cm wall_length) (meters_to_cm wall_width) (meters_to_cm wall_height)
  (wall_volume / brick_volume : ℝ) = 27000 :=
by sorry

end NUMINAMATH_CALUDE_brick_count_for_wall_l2091_209159


namespace NUMINAMATH_CALUDE_lcm_6_15_l2091_209164

theorem lcm_6_15 : Nat.lcm 6 15 = 30 := by
  sorry

end NUMINAMATH_CALUDE_lcm_6_15_l2091_209164


namespace NUMINAMATH_CALUDE_expression_simplification_l2091_209160

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (((2 * x - 1) / (x + 1) - x + 1) / ((x - 2) / (x^2 + 2*x + 1))) = -2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2091_209160


namespace NUMINAMATH_CALUDE_fib_odd_index_not_divisible_by_4k_plus_3_prime_l2091_209161

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define prime numbers of the form 4k + 3
def isPrime4kPlus3 (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ k : ℕ, p = 4 * k + 3

-- Theorem statement
theorem fib_odd_index_not_divisible_by_4k_plus_3_prime (n : ℕ) (p : ℕ) 
  (h_prime : isPrime4kPlus3 p) : ¬(p ∣ fib (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_fib_odd_index_not_divisible_by_4k_plus_3_prime_l2091_209161


namespace NUMINAMATH_CALUDE_cycle_transactions_result_l2091_209101

/-- Calculates the final amount after three cycle transactions -/
def final_amount (initial_cost : ℝ) (loss1 gain2 gain3 : ℝ) : ℝ :=
  let selling_price1 := initial_cost * (1 - loss1)
  let selling_price2 := selling_price1 * (1 + gain2)
  selling_price2 * (1 + gain3)

/-- Theorem stating the final amount after three cycle transactions -/
theorem cycle_transactions_result :
  final_amount 1600 0.12 0.15 0.20 = 1943.04 := by
  sorry

#eval final_amount 1600 0.12 0.15 0.20

end NUMINAMATH_CALUDE_cycle_transactions_result_l2091_209101


namespace NUMINAMATH_CALUDE_johns_allowance_l2091_209179

/-- John's weekly allowance problem -/
theorem johns_allowance (A : ℚ) : A = 4.80 :=
  let arcade_spent := (3 : ℚ) / 5
  let arcade_remaining := 1 - arcade_spent
  let toy_store_spent := (1 : ℚ) / 3 * arcade_remaining
  let candy_store_remaining := arcade_remaining - toy_store_spent
  have h1 : arcade_remaining = (2 : ℚ) / 5 := by sorry
  have h2 : candy_store_remaining = (4 : ℚ) / 15 := by sorry
  have h3 : candy_store_remaining * A = 1.28 := by sorry
  sorry

#eval (4.80 : ℚ)

end NUMINAMATH_CALUDE_johns_allowance_l2091_209179


namespace NUMINAMATH_CALUDE_point_on_transformed_plane_l2091_209198

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Apply a similarity transformation to a plane -/
def similarityTransform (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Check if a point lies on a plane -/
def pointOnPlane (point : Point) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

theorem point_on_transformed_plane :
  let originalPlane : Plane := { a := 3, b := -1, c := 2, d := 4 }
  let k : ℝ := 1/2
  let transformedPlane := similarityTransform originalPlane k
  let pointA : Point := { x := -1, y := 1, z := 1 }
  pointOnPlane pointA transformedPlane := by sorry

end NUMINAMATH_CALUDE_point_on_transformed_plane_l2091_209198


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_and_24_l2091_209187

theorem smallest_divisible_by_15_and_24 : ∃ n : ℕ, (n > 0 ∧ n % 15 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 15 = 0 ∧ m % 24 = 0) → n ≤ m) ∧ n = 120 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_and_24_l2091_209187


namespace NUMINAMATH_CALUDE_maria_savings_l2091_209113

/-- Calculates the amount left in Maria's savings after buying sweaters and scarves. -/
def amount_left (sweater_price scarf_price : ℕ) (num_sweaters num_scarves : ℕ) (initial_savings : ℕ) : ℕ :=
  initial_savings - (sweater_price * num_sweaters + scarf_price * num_scarves)

/-- Proves that Maria will have $200 left in her savings after buying sweaters and scarves. -/
theorem maria_savings : amount_left 30 20 6 6 500 = 200 := by
  sorry

end NUMINAMATH_CALUDE_maria_savings_l2091_209113


namespace NUMINAMATH_CALUDE_freds_dark_blue_marbles_l2091_209138

/-- Proves that the number of dark blue marbles is 6 given the conditions of Fred's marble collection. -/
theorem freds_dark_blue_marbles :
  let total_marbles : ℕ := 63
  let red_marbles : ℕ := 38
  let green_marbles : ℕ := red_marbles / 2
  let dark_blue_marbles : ℕ := total_marbles - red_marbles - green_marbles
  dark_blue_marbles = 6 := by
  sorry

end NUMINAMATH_CALUDE_freds_dark_blue_marbles_l2091_209138


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_is_585_dual_base_palindrome_properties_no_smaller_dual_base_palindrome_l2091_209122

/-- Checks if a number is a palindrome in the given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Finds the smallest number greater than 10 that is a palindrome in both base 2 and base 3 -/
def smallestDualBasePalindrome : ℕ := sorry

theorem smallest_dual_base_palindrome_is_585 :
  smallestDualBasePalindrome = 585 := by sorry

theorem dual_base_palindrome_properties (n : ℕ) :
  n = smallestDualBasePalindrome →
  n > 10 ∧ isPalindrome n 2 ∧ isPalindrome n 3 := by sorry

theorem no_smaller_dual_base_palindrome (n : ℕ) :
  10 < n ∧ n < smallestDualBasePalindrome →
  ¬(isPalindrome n 2 ∧ isPalindrome n 3) := by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_is_585_dual_base_palindrome_properties_no_smaller_dual_base_palindrome_l2091_209122


namespace NUMINAMATH_CALUDE_simplify_fraction_l2091_209171

theorem simplify_fraction : (144 : ℚ) / 1008 = 1 / 7 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2091_209171


namespace NUMINAMATH_CALUDE_digit_sum_difference_l2091_209123

/-- Represents a two-digit number -/
def TwoDigitNumber (tens ones : Nat) : Nat := 10 * tens + ones

theorem digit_sum_difference (A B C D : Nat) (E F : Nat) 
  (h1 : TwoDigitNumber A B + TwoDigitNumber C D = TwoDigitNumber A E)
  (h2 : TwoDigitNumber A B - TwoDigitNumber D C = TwoDigitNumber A F)
  (h3 : A < 10) (h4 : B < 10) (h5 : C < 10) (h6 : D < 10) : E = 9 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_difference_l2091_209123


namespace NUMINAMATH_CALUDE_rotation_theorem_l2091_209142

-- Define a function f with an inverse
variable (f : ℝ → ℝ)
variable (hf : Function.Bijective f)

-- Define the rotation transformation
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, p.1)

-- State the theorem
theorem rotation_theorem :
  ∀ x y : ℝ, y = f x ↔ (rotate90 (x, y)).2 = -(Function.invFun f) (rotate90 (x, y)).1 :=
by sorry

end NUMINAMATH_CALUDE_rotation_theorem_l2091_209142


namespace NUMINAMATH_CALUDE_min_value_and_ellipse_l2091_209118

theorem min_value_and_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : ∀ x : ℝ, x > 0 → (a + b) * x - 1 ≤ x^2) :
  (∀ c d : ℝ, c > 0 → d > 0 → 1 / c + 1 / d ≥ 2) ∧
  (1 / a^2 + 1 / b^2 > 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_and_ellipse_l2091_209118


namespace NUMINAMATH_CALUDE_cos_2a_given_tan_a_l2091_209108

theorem cos_2a_given_tan_a (a : ℝ) (h : Real.tan a = 2) : Real.cos (2 * a) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_2a_given_tan_a_l2091_209108


namespace NUMINAMATH_CALUDE_olivia_remaining_money_l2091_209117

theorem olivia_remaining_money (initial_amount spent_amount : ℕ) 
  (h1 : initial_amount = 128)
  (h2 : spent_amount = 38) :
  initial_amount - spent_amount = 90 := by
sorry

end NUMINAMATH_CALUDE_olivia_remaining_money_l2091_209117


namespace NUMINAMATH_CALUDE_solution_set_correct_l2091_209175

/-- The solution set for the system of equations:
    x + y + z = 2
    (x+y)(y+z) + (y+z)(z+x) + (z+x)(x+y) = 1
    x²(y+z) + y²(z+x) + z²(x+y) = -6 -/
def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(0, 3, -1), (0, -1, 3), (3, 0, -1), (3, -1, 0), (-1, 0, 3), (-1, 3, 0)}

/-- The system of equations -/
def satisfies_equations (x y z : ℝ) : Prop :=
  x + y + z = 2 ∧
  (x+y)*(y+z) + (y+z)*(z+x) + (z+x)*(x+y) = 1 ∧
  x^2*(y+z) + y^2*(z+x) + z^2*(x+y) = -6

theorem solution_set_correct :
  ∀ (x y z : ℝ), (x, y, z) ∈ solution_set ↔ satisfies_equations x y z :=
sorry

end NUMINAMATH_CALUDE_solution_set_correct_l2091_209175


namespace NUMINAMATH_CALUDE_average_weight_increase_l2091_209150

/-- 
Proves that replacing a person weighing 65 kg with a person weighing 97 kg 
in a group of 10 people increases the average weight by 3.2 kg
-/
theorem average_weight_increase (initial_average : ℝ) : 
  let initial_total := 10 * initial_average
  let new_total := initial_total - 65 + 97
  let new_average := new_total / 10
  new_average - initial_average = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_l2091_209150


namespace NUMINAMATH_CALUDE_alices_number_l2091_209167

theorem alices_number (n : ℕ) 
  (h1 : n % 243 = 0)
  (h2 : n % 36 = 0)
  (h3 : 1000 < n ∧ n < 3000) :
  n = 1944 ∨ n = 2916 := by
sorry

end NUMINAMATH_CALUDE_alices_number_l2091_209167


namespace NUMINAMATH_CALUDE_tower_surface_area_l2091_209181

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ

/-- Calculates the volume of a cube -/
def Cube.volume (c : Cube) : ℝ := c.sideLength ^ 3

/-- Calculates the surface area of a cube -/
def Cube.surfaceArea (c : Cube) : ℝ := 6 * c.sideLength ^ 2

/-- Represents the tower of cubes -/
structure CubeTower where
  cubes : List Cube
  isDecreasing : ∀ i j, i < j → (cubes.get i).volume > (cubes.get j).volume
  thirdCubeShifted : True

/-- Calculates the total surface area of the tower -/
def CubeTower.totalSurfaceArea (t : CubeTower) : ℝ :=
  let visibleFaces := [5, 5, 4.5] ++ List.replicate 5 4 ++ [5]
  List.sum (List.zipWith (λ c f => f * c.sideLength ^ 2) t.cubes visibleFaces)

/-- The theorem to be proved -/
theorem tower_surface_area (t : CubeTower) 
  (h1 : t.cubes.length = 9)
  (h2 : List.map Cube.volume t.cubes = [512, 343, 216, 125, 64, 27, 8, 1, 0.125]) :
  t.totalSurfaceArea = 948.25 := by
  sorry

end NUMINAMATH_CALUDE_tower_surface_area_l2091_209181


namespace NUMINAMATH_CALUDE_mixed_selection_probability_l2091_209135

/-- Represents the number of volunteers from each grade -/
structure Volunteers where
  first_grade : ℕ
  second_grade : ℕ

/-- Represents the number of temporary leaders selected from each grade -/
structure Leaders where
  first_grade : ℕ
  second_grade : ℕ

/-- Calculates the number of leaders proportionally selected from each grade -/
def selectLeaders (v : Volunteers) : Leaders :=
  { first_grade := (5 * v.first_grade) / (v.first_grade + v.second_grade),
    second_grade := (5 * v.second_grade) / (v.first_grade + v.second_grade) }

/-- Calculates the probability of selecting one leader from each grade -/
def probabilityOfMixedSelection (l : Leaders) : ℚ :=
  (l.first_grade * l.second_grade : ℚ) / ((l.first_grade + l.second_grade) * (l.first_grade + l.second_grade - 1) / 2 : ℚ)

theorem mixed_selection_probability 
  (v : Volunteers) 
  (h1 : v.first_grade = 150) 
  (h2 : v.second_grade = 100) : 
  probabilityOfMixedSelection (selectLeaders v) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_mixed_selection_probability_l2091_209135


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2091_209143

theorem polynomial_division_theorem (x : ℝ) :
  8 * x^3 - 4 * x^2 + 6 * x - 15 = (x - 3) * (8 * x^2 + 20 * x + 66) + 183 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2091_209143


namespace NUMINAMATH_CALUDE_min_value_problem_l2091_209163

theorem min_value_problem (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2*y = 1) :
  ∃ m : ℝ, m = 8/9 ∧ ∀ x' y' : ℝ, x' ≥ 0 → y' ≥ 0 → x' + 2*y' = 1 → 2*x' + 3*y'^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l2091_209163


namespace NUMINAMATH_CALUDE_fullTimeAndYearCount_l2091_209134

/-- Represents a company with employees. -/
structure Company where
  total : ℕ
  fullTime : ℕ
  atLeastYear : ℕ
  neitherFullTimeNorYear : ℕ

/-- The number of full-time employees who have worked at least a year. -/
def fullTimeAndYear (c : Company) : ℕ :=
  c.fullTime + c.atLeastYear - c.total + c.neitherFullTimeNorYear

/-- Theorem stating the number of full-time employees who have worked at least a year. -/
theorem fullTimeAndYearCount (c : Company) 
    (h1 : c.total = 130)
    (h2 : c.fullTime = 80)
    (h3 : c.atLeastYear = 100)
    (h4 : c.neitherFullTimeNorYear = 20) :
    fullTimeAndYear c = 70 := by
  sorry

end NUMINAMATH_CALUDE_fullTimeAndYearCount_l2091_209134


namespace NUMINAMATH_CALUDE_motorboat_travel_time_l2091_209193

/-- Represents the time (in hours) for the motorboat to travel from dock C to dock D -/
def motorboat_time_to_D : ℝ := 5.5

/-- Represents the total journey time in hours -/
def total_journey_time : ℝ := 12

/-- Represents the time (in hours) the motorboat stops at dock E -/
def stop_time_at_E : ℝ := 1

theorem motorboat_travel_time :
  motorboat_time_to_D = (total_journey_time - stop_time_at_E) / 2 :=
sorry

end NUMINAMATH_CALUDE_motorboat_travel_time_l2091_209193


namespace NUMINAMATH_CALUDE_brick_height_calculation_l2091_209183

/-- Prove that the height of each brick is 67.5 cm, given the wall dimensions,
    brick dimensions (except height), and the number of bricks needed. -/
theorem brick_height_calculation (wall_length wall_width wall_height : ℝ)
                                 (brick_length brick_width : ℝ)
                                 (num_bricks : ℕ) :
  wall_length = 900 →
  wall_width = 600 →
  wall_height = 22.5 →
  brick_length = 25 →
  brick_width = 11.25 →
  num_bricks = 7200 →
  ∃ (brick_height : ℝ),
    brick_height = 67.5 ∧
    wall_length * wall_width * wall_height =
      num_bricks * brick_length * brick_width * brick_height :=
by
  sorry

end NUMINAMATH_CALUDE_brick_height_calculation_l2091_209183


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2091_209106

theorem quadratic_inequality (y : ℝ) : y^2 - 9*y + 20 < 0 ↔ 4 < y ∧ y < 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2091_209106


namespace NUMINAMATH_CALUDE_a_minus_c_equals_296_l2091_209177

theorem a_minus_c_equals_296 (A B C : ℤ) 
  (h1 : A = B - 397)
  (h2 : A = 742)
  (h3 : B = C + 693) : 
  A - C = 296 := by
  sorry

end NUMINAMATH_CALUDE_a_minus_c_equals_296_l2091_209177


namespace NUMINAMATH_CALUDE_pen_pencil_difference_l2091_209194

theorem pen_pencil_difference :
  ∀ (pens pencils : ℕ),
    pens * 6 = pencils * 5 →  -- ratio of pens to pencils is 5:6
    pencils = 30 →            -- there are 30 pencils
    pencils - pens = 5        -- prove that there are 5 more pencils than pens
:= by sorry

end NUMINAMATH_CALUDE_pen_pencil_difference_l2091_209194


namespace NUMINAMATH_CALUDE_smallest_AAB_value_l2091_209191

def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem smallest_AAB_value :
  ∀ A B : ℕ,
  is_digit A →
  is_digit B →
  two_digit (10 * A + B) →
  three_digit (100 * A + 10 * A + B) →
  (10 * A + B : ℚ) = (1 / 7) * (100 * A + 10 * A + B) →
  ∀ A' B' : ℕ,
  is_digit A' →
  is_digit B' →
  two_digit (10 * A' + B') →
  three_digit (100 * A' + 10 * A' + B') →
  (10 * A' + B' : ℚ) = (1 / 7) * (100 * A' + 10 * A' + B') →
  100 * A + 10 * A + B ≤ 100 * A' + 10 * A' + B' →
  100 * A + 10 * A + B = 332 :=
by sorry

end NUMINAMATH_CALUDE_smallest_AAB_value_l2091_209191


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l2091_209148

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ+, (42 * x.val + 9) % 15 = 3 ∧
  ∀ y : ℕ+, (42 * y.val + 9) % 15 = 3 → x ≤ y ∧
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l2091_209148


namespace NUMINAMATH_CALUDE_second_bucket_contents_l2091_209178

def bucket_contents : List ℕ := [11, 13, 12, 16, 10]

theorem second_bucket_contents (h : ∃ x ∈ bucket_contents, x + 10 = 23) :
  (List.sum bucket_contents) - 23 = 39 := by
  sorry

end NUMINAMATH_CALUDE_second_bucket_contents_l2091_209178


namespace NUMINAMATH_CALUDE_largest_three_digit_product_l2091_209137

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

theorem largest_three_digit_product : ∃ (m x y : ℕ),
  100 ≤ m ∧ m < 1000 ∧
  isPrime x ∧ isPrime y ∧ isPrime (10 * x - y) ∧
  x < 10 ∧ y < 10 ∧ x ≠ y ∧
  m = x * y * (10 * x - y) ∧
  ∀ (m' x' y' : ℕ),
    100 ≤ m' ∧ m' < 1000 →
    isPrime x' ∧ isPrime y' ∧ isPrime (10 * x' - y') →
    x' < 10 ∧ y' < 10 ∧ x' ≠ y' →
    m' = x' * y' * (10 * x' - y') →
    m' ≤ m ∧
  m = 705 := by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_product_l2091_209137


namespace NUMINAMATH_CALUDE_minimum_students_l2091_209133

theorem minimum_students (b g : ℕ) : 
  (3 * b = 8 * g) →  -- From the equation (3/4)b = 2(2/3)g simplified
  (b ≥ 1) →          -- At least one boy
  (g ≥ 1) →          -- At least one girl
  (∀ b' g', (3 * b' = 8 * g') → b' + g' ≥ b + g) →  -- Minimum condition
  b + g = 25 := by
sorry

end NUMINAMATH_CALUDE_minimum_students_l2091_209133


namespace NUMINAMATH_CALUDE_odd_function_extension_l2091_209196

-- Define an odd function f
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_extension
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_positive : ∀ x > 0, f x = Real.exp x) :
  ∀ x < 0, f x = -Real.exp (-x) := by
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l2091_209196


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l2091_209182

def num_chairs : ℕ := 12
def num_students : ℕ := 5
def num_professors : ℕ := 4
def available_positions : ℕ := 6

theorem seating_arrangements_count :
  (Nat.choose available_positions num_professors) * (Nat.factorial num_professors) = 360 :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l2091_209182


namespace NUMINAMATH_CALUDE_common_chords_concur_l2091_209197

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Three pairwise intersecting circles --/
structure ThreeIntersectingCircles where
  c1 : Circle
  c2 : Circle
  c3 : Circle
  intersect_12 : c1.center.1 ^ 2 + c1.center.2 ^ 2 ≠ c2.center.1 ^ 2 + c2.center.2 ^ 2 ∨ c1.center ≠ c2.center
  intersect_23 : c2.center.1 ^ 2 + c2.center.2 ^ 2 ≠ c3.center.1 ^ 2 + c3.center.2 ^ 2 ∨ c2.center ≠ c3.center
  intersect_31 : c3.center.1 ^ 2 + c3.center.2 ^ 2 ≠ c1.center.1 ^ 2 + c1.center.2 ^ 2 ∨ c3.center ≠ c1.center

/-- A line in a plane, represented by ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The common chord of two intersecting circles --/
def commonChord (c1 c2 : Circle) : Line := sorry

/-- Three lines concur if they all pass through a single point --/
def concur (l1 l2 l3 : Line) : Prop := sorry

/-- The theorem: The common chords of three pairwise intersecting circles concur --/
theorem common_chords_concur (circles : ThreeIntersectingCircles) :
  let chord12 := commonChord circles.c1 circles.c2
  let chord23 := commonChord circles.c2 circles.c3
  let chord31 := commonChord circles.c3 circles.c1
  concur chord12 chord23 chord31 := by sorry

end NUMINAMATH_CALUDE_common_chords_concur_l2091_209197


namespace NUMINAMATH_CALUDE_problem_solution_l2091_209156

def f (x : ℝ) : ℝ := 2 * x - 4
def g (x : ℝ) : ℝ := -x + 4

theorem problem_solution :
  (f 1 = -2 ∧ g 1 = 3) ∧
  (∀ x, f x * g x = -2 * x^2 + 12 * x - 16) ∧
  (Set.Icc 2 4 = {x | f x * g x = 0}) ∧
  (∀ x y, x < 3 ∧ y < 3 ∧ x < y → f x * g x < f y * g y) ∧
  (∀ x y, x > 3 ∧ y > 3 ∧ x < y → f x * g x > f y * g y) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2091_209156


namespace NUMINAMATH_CALUDE_parallel_transitive_l2091_209158

-- Define the concept of straight lines
variable (Line : Type)

-- Define the parallel relationship between lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitive_l2091_209158


namespace NUMINAMATH_CALUDE_doubling_condition_iff_triangle_or_quadrilateral_l2091_209100

/-- The sum of interior angles of an n-sided polygon is (n-2) * 180°. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A polygon satisfies the doubling condition if the sum of angles after doubling
    the sides is an integer multiple of the original sum of angles. -/
def satisfies_doubling_condition (m : ℕ) : Prop :=
  ∃ k : ℕ, sum_interior_angles (2 * m) = k * sum_interior_angles m

/-- Theorem: A polygon satisfies the doubling condition if and only if
    it has 3 or 4 sides. -/
theorem doubling_condition_iff_triangle_or_quadrilateral (m : ℕ) :
  satisfies_doubling_condition m ↔ m = 3 ∨ m = 4 :=
sorry

end NUMINAMATH_CALUDE_doubling_condition_iff_triangle_or_quadrilateral_l2091_209100


namespace NUMINAMATH_CALUDE_monday_pages_to_reach_average_l2091_209192

def target_average : ℕ := 50
def days_in_week : ℕ := 7
def known_pages : List ℕ := [43, 28, 0, 70, 56, 88]

theorem monday_pages_to_reach_average :
  ∃ (monday_pages : ℕ),
    (monday_pages + known_pages.sum) / days_in_week = target_average ∧
    monday_pages = 65 := by
  sorry

end NUMINAMATH_CALUDE_monday_pages_to_reach_average_l2091_209192


namespace NUMINAMATH_CALUDE_decimal_123_in_base7_has_three_consecutive_digits_l2091_209195

/-- Represents a number in base 7 --/
def Base7 := Nat

/-- Converts a decimal number to base 7 --/
def toBase7 (n : Nat) : Base7 :=
  sorry

/-- Checks if a Base7 number has three consecutive digits --/
def hasThreeConsecutiveDigits (n : Base7) : Prop :=
  sorry

/-- The decimal number we're working with --/
def decimalNumber : Nat := 123

theorem decimal_123_in_base7_has_three_consecutive_digits :
  hasThreeConsecutiveDigits (toBase7 decimalNumber) :=
sorry

end NUMINAMATH_CALUDE_decimal_123_in_base7_has_three_consecutive_digits_l2091_209195


namespace NUMINAMATH_CALUDE_probability_x_less_than_2y_is_five_sixths_l2091_209173

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The specific rectangle in the problem --/
def problemRectangle : Rectangle where
  x_min := 0
  x_max := 6
  y_min := 0
  y_max := 2
  h_x := by norm_num
  h_y := by norm_num

/-- The probability of selecting a point (x,y) from the rectangle such that x < 2y --/
def probabilityXLessThan2Y (r : Rectangle) : ℝ :=
  sorry

theorem probability_x_less_than_2y_is_five_sixths :
  probabilityXLessThan2Y problemRectangle = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_less_than_2y_is_five_sixths_l2091_209173


namespace NUMINAMATH_CALUDE_smallest_base_not_divisible_by_five_l2091_209199

theorem smallest_base_not_divisible_by_five : 
  ∃ (b : ℕ), b > 2 ∧ b = 6 ∧ ¬(5 ∣ (2 * b^3 - 1)) ∧
  ∀ (k : ℕ), 2 < k ∧ k < b → (5 ∣ (2 * k^3 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_not_divisible_by_five_l2091_209199


namespace NUMINAMATH_CALUDE_polynomial_equality_l2091_209126

/-- Given a polynomial Q(x) = Q(0) + Q(1)x + Q(3)x^2 where Q(-1) = 2, 
    prove that Q(x) = 0.6x^2 - 2x - 0.6 -/
theorem polynomial_equality (Q : ℝ → ℝ) (h1 : ∀ x, Q x = Q 0 + Q 1 * x + Q 3 * x^2)
    (h2 : Q (-1) = 2) : ∀ x, Q x = 0.6 * x^2 - 2 * x - 0.6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2091_209126


namespace NUMINAMATH_CALUDE_clothing_sale_theorem_l2091_209157

/-- The marked price of an item of clothing --/
def marked_price : ℝ := 300

/-- The loss per item when sold at 40% of marked price --/
def loss_at_40_percent : ℝ := 30

/-- The profit per item when sold at 70% of marked price --/
def profit_at_70_percent : ℝ := 60

/-- The maximum discount percentage that can be offered without incurring a loss --/
def max_discount_percent : ℝ := 50

theorem clothing_sale_theorem :
  (0.4 * marked_price - loss_at_40_percent = 0.7 * marked_price + profit_at_70_percent) ∧
  (max_discount_percent / 100 * marked_price = 0.4 * marked_price + loss_at_40_percent) := by
  sorry

end NUMINAMATH_CALUDE_clothing_sale_theorem_l2091_209157


namespace NUMINAMATH_CALUDE_soccer_boys_percentage_l2091_209130

theorem soccer_boys_percentage (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) (girls_not_playing : ℕ)
  (h1 : total_students = 420)
  (h2 : boys = 320)
  (h3 : soccer_players = 250)
  (h4 : girls_not_playing = 65) :
  (boys - (total_students - boys - girls_not_playing)) / soccer_players * 100 = 86 := by
  sorry

end NUMINAMATH_CALUDE_soccer_boys_percentage_l2091_209130


namespace NUMINAMATH_CALUDE_remainder_problem_l2091_209144

theorem remainder_problem : (29 * 171997^2000) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2091_209144


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l2091_209146

theorem divisibility_equivalence (a m x n : ℕ) :
  m ∣ n ↔ (x^m - a^m) ∣ (x^n - a^n) := by sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l2091_209146


namespace NUMINAMATH_CALUDE_vector_subtraction_l2091_209184

/-- Given complex numbers z1 and z2 representing vectors OA and OB respectively,
    prove that the complex number representing BA is equal to 5-5i. -/
theorem vector_subtraction (z1 z2 : ℂ) (h1 : z1 = 2 - 3*I) (h2 : z2 = -3 + 2*I) :
  z1 - z2 = 5 - 5*I := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2091_209184


namespace NUMINAMATH_CALUDE_total_animals_savanna_l2091_209132

-- Define the number of animals in Safari National Park
def safari_lions : ℕ := 100
def safari_snakes : ℕ := safari_lions / 2
def safari_giraffes : ℕ := safari_snakes - 10
def safari_elephants : ℕ := safari_lions / 4

-- Define the number of animals in Savanna National Park
def savanna_lions : ℕ := safari_lions * 2
def savanna_snakes : ℕ := safari_snakes * 3
def savanna_giraffes : ℕ := safari_giraffes + 20
def savanna_elephants : ℕ := safari_elephants * 5
def savanna_zebras : ℕ := (savanna_lions + savanna_snakes) / 2

-- Theorem statement
theorem total_animals_savanna : 
  savanna_lions + savanna_snakes + savanna_giraffes + savanna_elephants + savanna_zebras = 710 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_savanna_l2091_209132


namespace NUMINAMATH_CALUDE_curve_and_tangent_l2091_209109

noncomputable section

-- Define the curve C
def C (k : ℝ) (x y : ℝ) : Prop :=
  x^(2/3) + y^(2/3) = k^(2/3)

-- Define the line segment AB
def AB (k : ℝ) (α β : ℝ) : Prop :=
  α^2 + β^2 = k^2

-- Define the midpoint M of AB
def M (α β : ℝ) : ℝ × ℝ :=
  (α^3 / (α^2 + β^2), β^3 / (α^2 + β^2))

-- State the theorem
theorem curve_and_tangent (k : ℝ) (h : k > 0) :
  ∀ α β : ℝ, AB k α β →
  let (x, y) := M α β
  (C k x y) ∧
  (∃ t : ℝ, t * α + (1 - t) * 0 = x ∧ t * 0 + (1 - t) * β = y) :=
sorry

end

end NUMINAMATH_CALUDE_curve_and_tangent_l2091_209109


namespace NUMINAMATH_CALUDE_three_triples_l2091_209186

/-- Least common multiple of two positive integers -/
def lcm (a b : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a,b,c) satisfying the given conditions -/
def count_triples : ℕ := sorry

/-- Theorem stating that there are exactly 3 ordered triples satisfying the conditions -/
theorem three_triples : count_triples = 3 := by sorry

end NUMINAMATH_CALUDE_three_triples_l2091_209186


namespace NUMINAMATH_CALUDE_megan_carrots_count_l2091_209120

/-- The total number of carrots Megan has after picking, throwing out some, and picking more. -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) : ℕ :=
  initial - thrown_out + picked_next_day

/-- Theorem stating that Megan's total carrots can be calculated using the given formula. -/
theorem megan_carrots_count (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ)
    (h1 : initial ≥ thrown_out) :
  total_carrots initial thrown_out picked_next_day = initial - thrown_out + picked_next_day :=
by
  sorry

#eval total_carrots 19 4 46  -- Should evaluate to 61

end NUMINAMATH_CALUDE_megan_carrots_count_l2091_209120


namespace NUMINAMATH_CALUDE_min_sum_xy_l2091_209103

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : x + y ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_xy_l2091_209103


namespace NUMINAMATH_CALUDE_sticker_distribution_l2091_209149

/-- The number of ways to distribute n identical objects into k groups,
    with each group containing at least one object -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- The problem statement -/
theorem sticker_distribution :
  distribute 10 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2091_209149


namespace NUMINAMATH_CALUDE_function_transformation_l2091_209112

/-- Given a function f where f(2) = 0, prove that g(x) = f(x-3)+1 passes through (5, 1) -/
theorem function_transformation (f : ℝ → ℝ) (h : f 2 = 0) :
  let g := λ x => f (x - 3) + 1
  g 5 = 1 := by
sorry

end NUMINAMATH_CALUDE_function_transformation_l2091_209112


namespace NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_solutions_l2091_209141

-- Define the function f(x) = |x-a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | |x - 1| > (1/2) * (x + 1)} = {x : ℝ | x > 3 ∨ x < 1/3} := by sorry

-- Part 2
theorem range_of_a_for_solutions :
  ∀ a : ℝ, (∃ x : ℝ, f a x + |x - 2| ≤ 3) ↔ -1 ≤ a ∧ a ≤ 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_a_equals_one_range_of_a_for_solutions_l2091_209141


namespace NUMINAMATH_CALUDE_max_pons_is_eleven_l2091_209107

/-- Represents the number of items purchased -/
structure Purchase where
  pans : ℕ
  pins : ℕ
  pons : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  3 * p.pans + 5 * p.pins + 8 * p.pons

/-- Checks if a purchase is valid (at least one of each item and total cost is $100) -/
def isValidPurchase (p : Purchase) : Prop :=
  p.pans ≥ 1 ∧ p.pins ≥ 1 ∧ p.pons ≥ 1 ∧ totalCost p = 100

/-- Theorem: The maximum number of pons in a valid purchase is 11 -/
theorem max_pons_is_eleven :
  ∀ p : Purchase, isValidPurchase p → p.pons ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_pons_is_eleven_l2091_209107


namespace NUMINAMATH_CALUDE_prove_distance_l2091_209140

def distance_between_cities : ℝ → Prop := λ d =>
  let speed_ab : ℝ := 40
  let speed_ba : ℝ := 49.99999999999999
  let total_time : ℝ := 5 + 24 / 60
  (d / speed_ab + d / speed_ba) = total_time

theorem prove_distance : distance_between_cities 120 := by
  sorry

end NUMINAMATH_CALUDE_prove_distance_l2091_209140


namespace NUMINAMATH_CALUDE_pen_profit_percentage_l2091_209105

/-- Calculates the profit percentage for a retailer selling pens -/
theorem pen_profit_percentage 
  (num_pens : ℕ) 
  (cost_pens : ℕ) 
  (discount_percent : ℚ) : 
  num_pens = 60 → 
  cost_pens = 36 → 
  discount_percent = 1/100 →
  (((num_pens : ℚ) * (1 - discount_percent) - cost_pens) / cost_pens) * 100 = 65 := by
  sorry

#check pen_profit_percentage

end NUMINAMATH_CALUDE_pen_profit_percentage_l2091_209105


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l2091_209128

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 5 →
  set1_mean = 15 →
  set2_count = 8 →
  set2_mean = 20 →
  let total_count := set1_count + set2_count
  let total_sum := set1_count * set1_mean + set2_count * set2_mean
  (total_sum / total_count : ℚ) = 235 / 13 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l2091_209128


namespace NUMINAMATH_CALUDE_pasta_preference_ratio_is_two_l2091_209104

/-- The ratio of students preferring spaghetti to those preferring manicotti -/
def pasta_preference_ratio (spaghetti_count : ℕ) (manicotti_count : ℕ) : ℚ :=
  spaghetti_count / manicotti_count

/-- The total number of students surveyed -/
def total_students : ℕ := 800

/-- The number of students who preferred spaghetti -/
def spaghetti_preference : ℕ := 320

/-- The number of students who preferred manicotti -/
def manicotti_preference : ℕ := 160

theorem pasta_preference_ratio_is_two :
  pasta_preference_ratio spaghetti_preference manicotti_preference = 2 := by
  sorry

end NUMINAMATH_CALUDE_pasta_preference_ratio_is_two_l2091_209104


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2091_209153

theorem modulus_of_complex_fraction : 
  let i : ℂ := Complex.I
  Complex.abs ((1 + 3 * i) / (1 - i)) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2091_209153


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2091_209189

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, (Complex.I : ℂ) * b = (2 + a * Complex.I) / (1 - Complex.I)) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2091_209189


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l2091_209169

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  area : ℝ

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.diagonal1 * r.diagonal2) / 2

theorem rhombus_longer_diagonal (r : Rhombus) 
  (h1 : r.diagonal1 = 12)
  (h2 : r.area = 120) :
  r.diagonal2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l2091_209169


namespace NUMINAMATH_CALUDE_vector_addition_l2091_209114

/-- Given two vectors AB and BC in ℝ², prove that AC = AB + BC -/
theorem vector_addition (AB BC : ℝ × ℝ) : 
  AB = (2, -1) → BC = (-4, 1) → AB + BC = (-2, 0) := by sorry

end NUMINAMATH_CALUDE_vector_addition_l2091_209114


namespace NUMINAMATH_CALUDE_modulo_problem_l2091_209172

theorem modulo_problem (n : ℕ) : 
  (215 * 789) % 75 = n ∧ 0 ≤ n ∧ n < 75 → n = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_modulo_problem_l2091_209172


namespace NUMINAMATH_CALUDE_games_purchased_l2091_209139

theorem games_purchased (total_income : ℕ) (expense : ℕ) (game_cost : ℕ) :
  total_income = 69 →
  expense = 24 →
  game_cost = 5 →
  (total_income - expense) / game_cost = 9 :=
by sorry

end NUMINAMATH_CALUDE_games_purchased_l2091_209139


namespace NUMINAMATH_CALUDE_not_always_cylinder_l2091_209110

/-- A cylinder in 3D space -/
structure Cylinder where
  base : Set (ℝ × ℝ)  -- Base of the cylinder
  height : ℝ          -- Height of the cylinder

/-- A plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ  -- Normal vector of the plane
  point : ℝ × ℝ × ℝ   -- A point on the plane

/-- Two planes are parallel if their normal vectors are parallel -/
def parallel (p1 p2 : Plane) : Prop :=
  ∃ (k : ℝ), p1.normal = k • p2.normal

/-- The result of cutting a cylinder with two parallel planes -/
def cut_cylinder (c : Cylinder) (p1 p2 : Plane) : Set (ℝ × ℝ × ℝ) :=
  sorry  -- Definition of the cut cylinder

/-- Theorem: Cutting a cylinder with two arbitrary parallel planes 
    does not always result in a cylinder -/
theorem not_always_cylinder (c : Cylinder) :
  ∃ (p1 p2 : Plane), parallel p1 p2 ∧ ¬∃ (c' : Cylinder), cut_cylinder c p1 p2 = {(x, y, z) | (x, y) ∈ c'.base ∧ 0 ≤ z ∧ z ≤ c'.height} :=
sorry


end NUMINAMATH_CALUDE_not_always_cylinder_l2091_209110


namespace NUMINAMATH_CALUDE_tangent_circles_m_range_l2091_209111

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y + 25 - m^2 = 0

-- Define the property of being externally tangent
def externally_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y m

-- State the theorem
theorem tangent_circles_m_range :
  ∀ m : ℝ, externally_tangent m ↔ m ∈ Set.Ioo (-4) 0 ∪ Set.Ioo 0 4 :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_m_range_l2091_209111


namespace NUMINAMATH_CALUDE_cos_equality_angle_l2091_209154

theorem cos_equality_angle (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (280 * π / 180) → n = 80 := by
sorry

end NUMINAMATH_CALUDE_cos_equality_angle_l2091_209154


namespace NUMINAMATH_CALUDE_product_repeating_third_and_nine_l2091_209174

/-- The repeating decimal 0.3̄ -/
def repeating_third : ℚ := 1/3

/-- Theorem stating that the product of 0.3̄ and 9 is 3 -/
theorem product_repeating_third_and_nine :
  repeating_third * 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_repeating_third_and_nine_l2091_209174


namespace NUMINAMATH_CALUDE_max_value_polynomial_l2091_209170

theorem max_value_polynomial (a b : ℝ) (h : a + b = 5) :
  ∃ M : ℝ, M = 6084 / 17 ∧ 
  ∀ x y : ℝ, x + y = 5 → 
  x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ M ∧
  ∃ a b : ℝ, a + b = 5 ∧ 
  a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 = M :=
sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l2091_209170


namespace NUMINAMATH_CALUDE_total_amount_after_two_years_l2091_209155

/-- Calculates the total amount returned after compound interest --/
def totalAmountAfterCompoundInterest (principal : ℝ) (rate : ℝ) (time : ℕ) (compoundInterest : ℝ) : ℝ :=
  principal + compoundInterest

/-- Theorem stating the total amount returned after two years of compound interest --/
theorem total_amount_after_two_years 
  (principal : ℝ) 
  (rate : ℝ) 
  (compoundInterest : ℝ) 
  (h1 : rate = 0.05) 
  (h2 : compoundInterest = 246) 
  (h3 : principal * ((1 + rate)^2 - 1) = compoundInterest) : 
  totalAmountAfterCompoundInterest principal rate 2 compoundInterest = 2646 := by
  sorry

#check total_amount_after_two_years

end NUMINAMATH_CALUDE_total_amount_after_two_years_l2091_209155


namespace NUMINAMATH_CALUDE_area_square_on_hypotenuse_for_24cm_l2091_209152

/-- An isosceles right triangle with an inscribed square -/
structure TriangleWithSquare where
  /-- Side length of the inscribed square touching the right angle -/
  s : ℝ
  /-- The square touches the right angle vertex -/
  touches_right_angle : s > 0
  /-- The opposite side of the square is parallel to the hypotenuse -/
  parallel_to_hypotenuse : True

/-- The area of a square inscribed along the hypotenuse of the triangle -/
def area_square_on_hypotenuse (t : TriangleWithSquare) : ℝ :=
  t.s ^ 2

theorem area_square_on_hypotenuse_for_24cm (t : TriangleWithSquare) 
  (h : t.s = 24) : area_square_on_hypotenuse t = 576 := by
  sorry

end NUMINAMATH_CALUDE_area_square_on_hypotenuse_for_24cm_l2091_209152


namespace NUMINAMATH_CALUDE_absolute_value_equation_range_l2091_209166

theorem absolute_value_equation_range :
  ∀ x : ℝ, (|3*x - 2| + |3*x + 1| = 3) ↔ (-1/3 ≤ x ∧ x ≤ 2/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_range_l2091_209166


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_24_and_1024_l2091_209102

theorem smallest_n_divisible_by_24_and_1024 :
  ∃ n : ℕ+, (∀ m : ℕ+, m < n → (¬(24 ∣ m^2) ∨ ¬(1024 ∣ m^3))) ∧
  (24 ∣ n^2) ∧ (1024 ∣ n^3) ∧ n = 48 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_24_and_1024_l2091_209102


namespace NUMINAMATH_CALUDE_meadow_diaper_earnings_l2091_209185

/-- Calculates the total money earned from selling diapers -/
def total_money (boxes : ℕ) (packs_per_box : ℕ) (diapers_per_pack : ℕ) (price_per_diaper : ℕ) : ℕ :=
  boxes * packs_per_box * diapers_per_pack * price_per_diaper

/-- Proves that Meadow's total earnings from selling diapers is $960,000 -/
theorem meadow_diaper_earnings :
  total_money 30 40 160 5 = 960000 := by
  sorry

#eval total_money 30 40 160 5

end NUMINAMATH_CALUDE_meadow_diaper_earnings_l2091_209185


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2091_209147

theorem sum_with_radical_conjugate : 
  let x : ℝ := 10 - Real.sqrt 2018
  let y : ℝ := 10 + Real.sqrt 2018  -- Definition of radical conjugate
  x + y = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l2091_209147


namespace NUMINAMATH_CALUDE_x_squared_congruence_l2091_209116

theorem x_squared_congruence (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 25]) 
  (h2 : 4 * x ≡ 20 [ZMOD 25]) : 
  x^2 ≡ 0 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_congruence_l2091_209116


namespace NUMINAMATH_CALUDE_walking_distance_multiple_l2091_209162

/-- Prove that the multiple M is 4 given the walking distances of Rajesh and Hiro -/
theorem walking_distance_multiple (total_distance hiro_distance rajesh_distance : ℝ) 
  (h1 : total_distance = 25)
  (h2 : rajesh_distance = 18)
  (h3 : total_distance = hiro_distance + rajesh_distance)
  (h4 : ∃ M : ℝ, rajesh_distance = M * hiro_distance - 10) :
  ∃ M : ℝ, M = 4 ∧ rajesh_distance = M * hiro_distance - 10 := by
  sorry

end NUMINAMATH_CALUDE_walking_distance_multiple_l2091_209162


namespace NUMINAMATH_CALUDE_no_both_squares_l2091_209165

theorem no_both_squares : ¬∃ (x y : ℕ+), 
  ∃ (a b : ℕ+), (x^2 + 2*y : ℕ) = a^2 ∧ (y^2 + 2*x : ℕ) = b^2 := by
  sorry

end NUMINAMATH_CALUDE_no_both_squares_l2091_209165


namespace NUMINAMATH_CALUDE_square_pens_area_ratio_l2091_209115

/-- Given four congruent square pens with side length s, prove that the ratio of their
    total area to the area of a single square pen formed by reusing the same amount
    of fencing is 1/4. -/
theorem square_pens_area_ratio (s : ℝ) (h : s > 0) : 
  (4 * s^2) / ((4 * s)^2) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_square_pens_area_ratio_l2091_209115


namespace NUMINAMATH_CALUDE_intersection_point_power_l2091_209119

theorem intersection_point_power (n : ℕ) (x₀ y₀ : ℝ) (hn : n ≥ 2) 
  (h1 : y₀^2 = n * x₀ - 1) (h2 : y₀ = x₀) :
  ∀ m : ℕ, m > 0 → ∃ k : ℕ, k ≥ 2 ∧ (x₀^m)^2 = k * (x₀^m) - 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_power_l2091_209119


namespace NUMINAMATH_CALUDE_nine_squared_minus_sqrt_nine_l2091_209125

theorem nine_squared_minus_sqrt_nine : 9^2 - Real.sqrt 9 = 78 := by
  sorry

end NUMINAMATH_CALUDE_nine_squared_minus_sqrt_nine_l2091_209125


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2091_209176

theorem simple_interest_rate_calculation (P A T : ℝ) (h1 : P = 750) (h2 : A = 1050) (h3 : T = 5) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = 8 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l2091_209176


namespace NUMINAMATH_CALUDE_absolute_value_sqrt_two_plus_half_inverse_l2091_209127

theorem absolute_value_sqrt_two_plus_half_inverse :
  |1 - Real.sqrt 2| + (1/2)⁻¹ = Real.sqrt 2 + 1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_sqrt_two_plus_half_inverse_l2091_209127
