import Mathlib

namespace arithmetic_progression_sum_l4020_402036

def arithmetic_progression (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_progression_sum (a : ℕ → ℝ) :
  arithmetic_progression a → a 5 = 5 → a 3 + a 7 = 10 := by
  sorry

end arithmetic_progression_sum_l4020_402036


namespace horse_distance_in_day_l4020_402074

/-- The distance a horse can run in one day -/
def horse_distance (speed : ℝ) (hours_per_day : ℝ) : ℝ :=
  speed * hours_per_day

/-- Theorem: A horse running at 10 miles/hour for 24 hours covers 240 miles -/
theorem horse_distance_in_day :
  horse_distance 10 24 = 240 := by
  sorry

end horse_distance_in_day_l4020_402074


namespace collinear_probability_in_5x5_grid_l4020_402033

/-- The number of dots in each row and column of the grid -/
def gridSize : ℕ := 5

/-- The total number of dots in the grid -/
def totalDots : ℕ := gridSize * gridSize

/-- The number of dots to be chosen -/
def chosenDots : ℕ := 4

/-- The number of ways to choose 4 collinear dots -/
def collinearWays : ℕ := 2 * gridSize + 2

/-- The total number of ways to choose 4 dots from 25 -/
def totalWays : ℕ := Nat.choose totalDots chosenDots

/-- The probability of choosing 4 collinear dots -/
def collinearProbability : ℚ := collinearWays / totalWays

theorem collinear_probability_in_5x5_grid :
  collinearProbability = 12 / 12650 :=
sorry

end collinear_probability_in_5x5_grid_l4020_402033


namespace Q_divisible_by_three_l4020_402027

def Q (x p q : ℤ) : ℤ := x^3 - x + (p+1)*x + q

theorem Q_divisible_by_three (p q : ℤ) 
  (h1 : 3 ∣ (p + 1)) 
  (h2 : 3 ∣ q) : 
  ∀ x : ℤ, 3 ∣ Q x p q := by
sorry

end Q_divisible_by_three_l4020_402027


namespace abcdef_hex_bits_l4020_402070

def hex_to_decimal (h : String) : ℕ := 
  match h with
  | "A" => 10
  | "B" => 11
  | "C" => 12
  | "D" => 13
  | "E" => 14
  | "F" => 15
  | _ => 0  -- This case should never be reached for valid hex digits

theorem abcdef_hex_bits : 
  let decimal : ℕ := 
    (hex_to_decimal "A") * (16^5) +
    (hex_to_decimal "B") * (16^4) +
    (hex_to_decimal "C") * (16^3) +
    (hex_to_decimal "D") * (16^2) +
    (hex_to_decimal "E") * (16^1) +
    (hex_to_decimal "F")
  ∃ n : ℕ, 2^n ≤ decimal ∧ decimal < 2^(n+1) ∧ n + 1 = 24 :=
by sorry

end abcdef_hex_bits_l4020_402070


namespace compound_weight_proof_l4020_402011

/-- Molar mass of Nitrogen in g/mol -/
def N_mass : ℝ := 14.01

/-- Molar mass of Hydrogen in g/mol -/
def H_mass : ℝ := 1.01

/-- Molar mass of Iodine in g/mol -/
def I_mass : ℝ := 126.90

/-- Molar mass of Oxygen in g/mol -/
def O_mass : ℝ := 16.00

/-- Molar mass of NH4I in g/mol -/
def NH4I_mass : ℝ := N_mass + 4 * H_mass + I_mass

/-- Molar mass of H2O in g/mol -/
def H2O_mass : ℝ := 2 * H_mass + O_mass

/-- Number of moles of NH4I -/
def NH4I_moles : ℝ := 15

/-- Number of moles of H2O -/
def H2O_moles : ℝ := 7

/-- Total weight of the compound (NH4I·H2O) in grams -/
def total_weight : ℝ := NH4I_moles * NH4I_mass + H2O_moles * H2O_mass

theorem compound_weight_proof : total_weight = 2300.39 := by
  sorry

end compound_weight_proof_l4020_402011


namespace square_perimeter_l4020_402045

/-- A square with area 484 cm² has a perimeter of 88 cm. -/
theorem square_perimeter (s : ℝ) (h1 : s > 0) (h2 : s^2 = 484) : 4 * s = 88 := by
  sorry

end square_perimeter_l4020_402045


namespace candy_distribution_theorem_l4020_402035

/-- The number of ways to distribute candies into boxes. -/
def distribute_candy (candies boxes : ℕ) : ℕ := sorry

/-- The number of ways to distribute candies into boxes with no adjacent empty boxes. -/
def distribute_candy_no_adjacent_empty (candies boxes : ℕ) : ℕ := sorry

/-- Theorem: There are 34 ways to distribute 10 pieces of candy into 5 boxes
    such that no two adjacent boxes are empty. -/
theorem candy_distribution_theorem :
  distribute_candy_no_adjacent_empty 10 5 = 34 := by sorry

end candy_distribution_theorem_l4020_402035


namespace eu_countries_2012_is_set_l4020_402095

/-- A type representing countries -/
def Country : Type := String

/-- A predicate that determines if a country was in the EU in 2012 -/
def WasEUMemberIn2012 (c : Country) : Prop := sorry

/-- The set of all EU countries in 2012 -/
def EUCountries2012 : Set Country :=
  {c : Country | WasEUMemberIn2012 c}

/-- A property that determines if a collection can form a set -/
def CanFormSet (S : Set α) : Prop :=
  ∀ x, x ∈ S → (∃ p : Prop, p ↔ x ∈ S)

theorem eu_countries_2012_is_set :
  CanFormSet EUCountries2012 :=
sorry

end eu_countries_2012_is_set_l4020_402095


namespace smallest_integer_satisfying_inequality_l4020_402016

theorem smallest_integer_satisfying_inequality :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 13*m + 22 ≤ 0 → n ≤ m) ∧ n^2 - 13*n + 22 ≤ 0 ∧ n = 2 := by
  sorry

end smallest_integer_satisfying_inequality_l4020_402016


namespace first_class_average_mark_l4020_402093

theorem first_class_average_mark (x : ℝ) : 
  (25 * x + 30 * 60) / 55 = 50.90909090909091 → x = 40 := by
  sorry

end first_class_average_mark_l4020_402093


namespace benny_work_hours_l4020_402024

/-- Given a person who works a fixed number of hours per day for a certain number of days,
    calculate the total number of hours worked. -/
def totalHoursWorked (hoursPerDay : ℕ) (numberOfDays : ℕ) : ℕ :=
  hoursPerDay * numberOfDays

/-- Theorem stating that working 3 hours per day for 6 days results in 18 total hours worked. -/
theorem benny_work_hours :
  totalHoursWorked 3 6 = 18 := by
  sorry

end benny_work_hours_l4020_402024


namespace cubic_equation_real_root_l4020_402015

theorem cubic_equation_real_root (k : ℝ) (hk : k ≠ 0) :
  ∃ x : ℝ, x^3 + k*x + k^2 = 0 :=
sorry

end cubic_equation_real_root_l4020_402015


namespace complex_product_equals_369_l4020_402060

theorem complex_product_equals_369 (x : ℂ) : 
  x = Complex.exp (2 * Real.pi * I / 9) →
  (3 * x + x^3) * (3 * x^3 + x^9) * (3 * x^6 + x^18) = 369 := by
  sorry

end complex_product_equals_369_l4020_402060


namespace abs_even_and_increasing_l4020_402066

-- Define the function
def f (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_even_and_increasing : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x ≤ f y) :=
by sorry

end abs_even_and_increasing_l4020_402066


namespace rectangle_formation_count_l4020_402013

theorem rectangle_formation_count (h : ℕ) (v : ℕ) : h = 5 → v = 6 → Nat.choose h 2 * Nat.choose v 2 = 150 := by
  sorry

end rectangle_formation_count_l4020_402013


namespace roses_left_unsold_l4020_402078

theorem roses_left_unsold (price : ℕ) (initial : ℕ) (earned : ℕ) : 
  price = 7 → initial = 9 → earned = 35 → initial - (earned / price) = 4 := by
  sorry

end roses_left_unsold_l4020_402078


namespace sum_in_B_l4020_402075

-- Define set A
def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

-- Define set B
def B : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}

-- Define set C (although not used in the theorem, it's part of the original problem)
def C : Set ℤ := {x | ∃ k : ℤ, x = 4 * k + 1}

-- Theorem statement
theorem sum_in_B (a b : ℤ) (ha : a ∈ A) (hb : b ∈ B) : a + b ∈ B := by
  sorry

end sum_in_B_l4020_402075


namespace difference_of_squares_special_case_l4020_402049

theorem difference_of_squares_special_case : (500 : ℤ) * 500 - 499 * 501 = 1 := by
  sorry

end difference_of_squares_special_case_l4020_402049


namespace center_is_ten_l4020_402082

/-- Represents a 4x4 array of integers -/
def Array4x4 := Fin 4 → Fin 4 → ℕ

/-- Checks if two positions in the array share an edge -/
def share_edge (p q : Fin 4 × Fin 4) : Prop :=
  (p.1 = q.1 ∧ (p.2.val + 1 = q.2.val ∨ p.2.val = q.2.val + 1)) ∨
  (p.2 = q.2 ∧ (p.1.val + 1 = q.1.val ∨ p.1.val = q.1.val + 1))

/-- Defines a valid array according to the problem conditions -/
def valid_array (a : Array4x4) : Prop :=
  (∀ n : Fin 16, ∃ i j : Fin 4, a i j = n.val + 1) ∧
  (∀ n : Fin 15, ∃ i j k l : Fin 4, 
    a i j = n.val + 1 ∧ 
    a k l = n.val + 2 ∧ 
    share_edge (i, j) (k, l)) ∧
  (a 0 0 + a 0 3 + a 3 0 + a 3 3 = 34)

/-- The main theorem to prove -/
theorem center_is_ten (a : Array4x4) (h : valid_array a) : 
  a 1 1 = 10 ∨ a 1 2 = 10 ∨ a 2 1 = 10 ∨ a 2 2 = 10 := by
  sorry

end center_is_ten_l4020_402082


namespace at_least_one_not_greater_than_neg_one_l4020_402086

theorem at_least_one_not_greater_than_neg_one (a b c d : ℝ) 
  (sum_eq : a + b + c + d = -2)
  (sum_products_eq : a*b + a*c + a*d + b*c + b*d + c*d = 0) :
  min a (min b (min c d)) ≤ -1 := by
sorry

end at_least_one_not_greater_than_neg_one_l4020_402086


namespace transform_f_to_g_l4020_402061

-- Define the original function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f ((1 - x) / 2) + 1

-- Theorem stating the transformations
theorem transform_f_to_g :
  ∀ x : ℝ,
  -- Reflection across y-axis
  g (-x) = f ((1 + x) / 2) + 1 ∧
  -- Horizontal stretch by factor 2
  g (2 * x) = f ((1 - 2*x) / 2) + 1 ∧
  -- Horizontal shift right by 0.5 units
  g (x - 0.5) = f ((1 - (x - 0.5)) / 2) + 1 ∧
  -- Vertical shift up by 1 unit
  g x = f ((1 - x) / 2) + 1 :=
by sorry

end transform_f_to_g_l4020_402061


namespace prime_sum_and_seven_sum_squares_l4020_402054

theorem prime_sum_and_seven_sum_squares (p q : ℕ) : 
  Prime p → Prime q → 
  ∃ x y : ℕ, x^2 = p + q ∧ y^2 = p + 7*q → 
  p = 2 := by
sorry

end prime_sum_and_seven_sum_squares_l4020_402054


namespace intersected_cubes_count_l4020_402062

/-- Represents a cube composed of unit cubes -/
structure LargeCube where
  side_length : ℕ
  total_cubes : ℕ
  h_total : total_cubes = side_length ^ 3

/-- Represents a plane intersecting the cube -/
structure IntersectingPlane where
  perpendicular_to_diagonal : Bool
  bisects_diagonal : Bool

/-- Counts the number of unit cubes intersected by the plane -/
def count_intersected_cubes (cube : LargeCube) (plane : IntersectingPlane) : ℕ :=
  sorry

/-- Theorem: A plane perpendicular to and bisecting an internal diagonal 
    of a 4x4x4 cube intersects exactly 32 unit cubes -/
theorem intersected_cubes_count 
  (cube : LargeCube) 
  (plane : IntersectingPlane) 
  (h_side : cube.side_length = 4)
  (h_perp : plane.perpendicular_to_diagonal = true)
  (h_bisect : plane.bisects_diagonal = true) : 
  count_intersected_cubes cube plane = 32 := by
  sorry

end intersected_cubes_count_l4020_402062


namespace jackson_running_distance_l4020_402098

/-- Calculate the final running distance after doubling the initial distance for a given number of weeks -/
def finalDistance (initialDistance : ℕ) (weeks : ℕ) : ℕ :=
  initialDistance * (2 ^ weeks)

/-- Theorem stating that starting with 3 miles and doubling for 4 weeks results in 24 miles -/
theorem jackson_running_distance : finalDistance 3 4 = 24 := by
  sorry

end jackson_running_distance_l4020_402098


namespace polynomial_difference_theorem_l4020_402055

/-- Given two polynomials that differ in terms of x^2 and y^2, 
    prove the values of m and n and the result of a specific expression. -/
theorem polynomial_difference_theorem (m n : ℝ) : 
  (∀ x y : ℝ, 2 * (m * x^2 - 2 * y^2) - (x - 2 * y) - (x - n * y^2 - 2 * x^2) = 0) →
  m = -1 ∧ n = 4 ∧ (4 * m^2 * n - 3 * m * n^2) - 2 * (m^2 * n + m * n^2) = -72 := by
  sorry

end polynomial_difference_theorem_l4020_402055


namespace sin_plus_sin_sqrt2_not_periodic_l4020_402040

/-- The function x ↦ sin x + sin (√2 x) is not periodic -/
theorem sin_plus_sin_sqrt2_not_periodic :
  ¬ ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, Real.sin x + Real.sin (Real.sqrt 2 * x) = Real.sin (x + p) + Real.sin (Real.sqrt 2 * (x + p)) := by
  sorry

end sin_plus_sin_sqrt2_not_periodic_l4020_402040


namespace competitive_examination_selection_l4020_402065

theorem competitive_examination_selection (total_candidates : ℕ) 
  (selection_rate_A : ℚ) (selection_rate_B : ℚ) : 
  total_candidates = 8000 →
  selection_rate_A = 6 / 100 →
  selection_rate_B = 7 / 100 →
  (selection_rate_B * total_candidates : ℚ) - (selection_rate_A * total_candidates : ℚ) = 80 := by
  sorry

end competitive_examination_selection_l4020_402065


namespace line_intersection_regions_l4020_402000

theorem line_intersection_regions (h s : ℕ+) : 
  (s + 1) * (s + 2 * h) = 3984 ↔ (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) :=
sorry

end line_intersection_regions_l4020_402000


namespace problem_1_problem_2_l4020_402072

-- Problem 1
theorem problem_1 (x y : ℝ) (h1 : x * y = 5) (h2 : x + y = 6) :
  (x - y)^2 = 16 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : (2016 - a) * (2017 - a) = 5) :
  (a - 2016)^2 + (2017 - a)^2 = 11 := by sorry

end problem_1_problem_2_l4020_402072


namespace percent_more_and_less_equal_l4020_402089

theorem percent_more_and_less_equal (x : ℝ) : x = 138.67 →
  (80 + 0.3 * 80 : ℝ) = (x - 0.25 * x) := by sorry

end percent_more_and_less_equal_l4020_402089


namespace angle_of_inclination_for_unit_slope_l4020_402003

/-- Given a line with slope of absolute value 1, its angle of inclination is either 45° or 135°. -/
theorem angle_of_inclination_for_unit_slope (slope : ℝ) (h : |slope| = 1) :
  let angle := Real.arctan slope
  angle = π/4 ∨ angle = 3*π/4 := by
sorry

end angle_of_inclination_for_unit_slope_l4020_402003


namespace odd_even_sum_reciprocal_l4020_402071

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function g is even if g(-x) = g(x) for all x -/
def IsEven (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = g x

/-- Given f is odd, g is even, and f(x) + g(x) = 1 / (x - 1), prove f(3) = 3/8 -/
theorem odd_even_sum_reciprocal (f g : ℝ → ℝ) 
    (hodd : IsOdd f) (heven : IsEven g) 
    (hsum : ∀ x ≠ 1, f x + g x = 1 / (x - 1)) : 
    f 3 = 3/8 := by
  sorry

end odd_even_sum_reciprocal_l4020_402071


namespace student_ratio_l4020_402056

theorem student_ratio (total : ℕ) (on_bleachers : ℕ) 
  (h1 : total = 26) (h2 : on_bleachers = 4) : 
  (total - on_bleachers : ℚ) / total = 11 / 13 := by
  sorry

end student_ratio_l4020_402056


namespace three_equal_differences_exist_l4020_402076

theorem three_equal_differences_exist (a : Fin 19 → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_bound : ∀ i, a i < 91) :
  ∃ i j k l m n, i ≠ j ∧ k ≠ l ∧ m ≠ n ∧
    i ≠ k ∧ i ≠ m ∧ k ≠ m ∧
    a j - a i = a l - a k ∧ a n - a m = a j - a i :=
sorry

end three_equal_differences_exist_l4020_402076


namespace middle_guard_hours_l4020_402077

theorem middle_guard_hours (total_hours : ℕ) (num_guards : ℕ) (first_guard_hours : ℕ) (last_guard_hours : ℕ) :
  total_hours = 9 ∧ num_guards = 4 ∧ first_guard_hours = 3 ∧ last_guard_hours = 2 →
  (total_hours - first_guard_hours - last_guard_hours) / (num_guards - 2) = 2 := by
  sorry

end middle_guard_hours_l4020_402077


namespace blake_apples_cost_l4020_402042

/-- The amount Blake spent on apples -/
def apples_cost (total : ℕ) (change : ℕ) (oranges : ℕ) (mangoes : ℕ) : ℕ :=
  total - change - (oranges + mangoes)

/-- Theorem: Blake spent $50 on apples -/
theorem blake_apples_cost :
  apples_cost 300 150 40 60 = 50 := by
  sorry

end blake_apples_cost_l4020_402042


namespace machine_production_l4020_402002

/-- Given the production rate of 6 machines, calculate the production of 8 machines in 4 minutes -/
theorem machine_production 
  (rate : ℕ) -- Production rate per minute for 6 machines
  (h1 : rate = 270) -- 6 machines produce 270 bottles per minute
  : (8 * 4 * (rate / 6) : ℕ) = 1440 := by
  sorry

end machine_production_l4020_402002


namespace smallest_percent_increase_between_3_and_4_l4020_402091

def question_values : List ℕ := [100, 300, 600, 900, 1500, 2400]

def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def consecutive_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  List.zip l (List.tail l)

theorem smallest_percent_increase_between_3_and_4 :
  let pairs := consecutive_pairs question_values
  let increases := pairs.map (fun (a, b) => percent_increase a b)
  increases.argmin id = some 2 := by sorry

end smallest_percent_increase_between_3_and_4_l4020_402091


namespace solution_set_inequality_l4020_402009

theorem solution_set_inequality (x : ℝ) :
  (x^2 - |x| > 0) ↔ (x < -1 ∨ x > 1) :=
sorry

end solution_set_inequality_l4020_402009


namespace sphere_triangle_distance_is_four_l4020_402094

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Represents a triangle with given side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Calculates the distance between the center of a sphere and the plane of a triangle tangent to it -/
def sphereTriangleDistance (s : Sphere) (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating the distance between the sphere's center and the triangle's plane -/
theorem sphere_triangle_distance_is_four :
  ∀ (s : Sphere) (t : Triangle),
    s.radius = 8 ∧
    t.side1 = 13 ∧ t.side2 = 14 ∧ t.side3 = 15 →
    sphereTriangleDistance s t = 4 :=
by
  sorry

end sphere_triangle_distance_is_four_l4020_402094


namespace balls_after_500_steps_l4020_402046

/-- Represents the state of boxes after a certain number of steps -/
def BoxState := Nat

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : Nat) : List Nat :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List Nat) : Nat :=
  sorry

/-- Simulates the ball placement process for a given number of steps -/
def simulateSteps (steps : Nat) : BoxState :=
  sorry

theorem balls_after_500_steps :
  simulateSteps 500 = sumDigits (toBase4 500) :=
sorry

end balls_after_500_steps_l4020_402046


namespace expression_equality_l4020_402043

theorem expression_equality : -2^2 + Real.sqrt 8 - 3 + 1/3 = -20/3 + 2 * Real.sqrt 2 := by
  sorry

end expression_equality_l4020_402043


namespace number_of_men_l4020_402087

theorem number_of_men (M : ℕ) (W : ℝ) : 
  (W / (M * 40) = W / ((M - 5) * 50)) → M = 25 := by
  sorry

end number_of_men_l4020_402087


namespace initial_bird_families_l4020_402041

/-- The number of bird families that flew away for winter. -/
def flew_away : ℕ := 7

/-- The difference between the number of bird families that stayed and those that flew away. -/
def difference : ℕ := 73

/-- The total number of bird families initially living near the mountain. -/
def total_families : ℕ := flew_away + (flew_away + difference)

theorem initial_bird_families :
  total_families = 87 :=
sorry

end initial_bird_families_l4020_402041


namespace seventy_fifth_term_is_298_l4020_402084

def arithmetic_sequence (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

theorem seventy_fifth_term_is_298 : arithmetic_sequence 2 4 75 = 298 := by
  sorry

end seventy_fifth_term_is_298_l4020_402084


namespace reciprocal_opposite_theorem_l4020_402048

theorem reciprocal_opposite_theorem (a b c d : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  : (c + d)^2 - a * b = -1 := by
  sorry

end reciprocal_opposite_theorem_l4020_402048


namespace zoo_fraction_l4020_402028

/-- Given a zoo with various animals, prove that the fraction of elephants
    to the sum of parrots and snakes is 1/2. -/
theorem zoo_fraction (parrots snakes monkeys elephants zebras : ℕ) 
  (h1 : parrots = 8)
  (h2 : snakes = 3 * parrots)
  (h3 : monkeys = 2 * snakes)
  (h4 : ∃ f : ℚ, elephants = f * (parrots + snakes))
  (h5 : zebras + 3 = elephants)
  (h6 : monkeys - zebras = 35) :
  ∃ f : ℚ, elephants = f * (parrots + snakes) ∧ f = 1/2 := by
  sorry

end zoo_fraction_l4020_402028


namespace negation_at_most_four_l4020_402010

-- Define "at most four" for natural numbers
def at_most_four (n : ℕ) : Prop := n ≤ 4

-- Define "at least five" for natural numbers
def at_least_five (n : ℕ) : Prop := n ≥ 5

-- Theorem stating that the negation of "at most four" is equivalent to "at least five"
theorem negation_at_most_four (n : ℕ) : ¬(at_most_four n) ↔ at_least_five n := by
  sorry

end negation_at_most_four_l4020_402010


namespace min_values_ab_l4020_402014

theorem min_values_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ 1/x + 2/y < 9) = False ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ x^2 + y^2 < 1/5) = False :=
by sorry

end min_values_ab_l4020_402014


namespace angle_sum_theorem_l4020_402099

theorem angle_sum_theorem (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 1/3) (h4 : Real.cos β = 3/5) :
  α + 2*β = π - Real.arctan (13/9) := by sorry

end angle_sum_theorem_l4020_402099


namespace hyperbola_foci_distance_l4020_402012

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- First asymptote: y = x + 2 -/
  asymptote1 : ℝ → ℝ
  /-- Second asymptote: y = 4 - x -/
  asymptote2 : ℝ → ℝ
  /-- The hyperbola passes through the point (4, 4) -/
  passes_through : ℝ × ℝ
  /-- Conditions for the asymptotes -/
  h_asymptote1 : ∀ x, asymptote1 x = x + 2
  h_asymptote2 : ∀ x, asymptote2 x = 4 - x
  h_passes_through : passes_through = (4, 4)

/-- The distance between the foci of the hyperbola -/
def foci_distance (h : Hyperbola) : ℝ := 8

/-- Theorem stating that the distance between the foci of the given hyperbola is 8 -/
theorem hyperbola_foci_distance (h : Hyperbola) :
  foci_distance h = 8 := by sorry

end hyperbola_foci_distance_l4020_402012


namespace multiply_to_target_l4020_402081

theorem multiply_to_target (x : ℕ) : x * 586645 = 5865863355 → x = 9999 := by
  sorry

end multiply_to_target_l4020_402081


namespace certain_number_proof_l4020_402097

theorem certain_number_proof (h1 : 268 * 74 = 19732) (n : ℝ) (h2 : 2.68 * n = 1.9832) : n = 0.74 := by
  sorry

end certain_number_proof_l4020_402097


namespace pizza_toppings_l4020_402050

theorem pizza_toppings (total_slices cheese_slices onion_slices : ℕ) 
  (h_total : total_slices = 16)
  (h_cheese : cheese_slices = 9)
  (h_onion : onion_slices = 13)
  (h_at_least_one : cheese_slices + onion_slices ≥ total_slices) :
  ∃ (both_toppings : ℕ), 
    both_toppings = cheese_slices + onion_slices - total_slices ∧
    both_toppings = 6 :=
by sorry

end pizza_toppings_l4020_402050


namespace partnership_profit_theorem_l4020_402044

/-- Represents an investment in a partnership business -/
structure Investment where
  amount : ℕ
  duration : ℕ

/-- Calculates the total profit of a partnership business -/
def calculateTotalProfit (investments : List Investment) (cProfit : ℕ) : ℕ :=
  let totalCapitalMonths := investments.foldl (fun acc inv => acc + inv.amount * inv.duration) 0
  let cCapitalMonths := (investments.find? (fun inv => inv.amount = 6000 ∧ inv.duration = 6)).map (fun inv => inv.amount * inv.duration)
  match cCapitalMonths with
  | some cm => totalCapitalMonths * cProfit / cm
  | none => 0

theorem partnership_profit_theorem (investments : List Investment) (cProfit : ℕ) :
  investments = [
    ⟨8000, 12⟩,  -- A's investment
    ⟨4000, 8⟩,   -- B's investment
    ⟨6000, 6⟩,   -- C's investment
    ⟨10000, 9⟩   -- D's investment
  ] ∧ cProfit = 36000 →
  calculateTotalProfit investments cProfit = 254000 := by
  sorry

#eval calculateTotalProfit [⟨8000, 12⟩, ⟨4000, 8⟩, ⟨6000, 6⟩, ⟨10000, 9⟩] 36000

end partnership_profit_theorem_l4020_402044


namespace division_problem_l4020_402053

theorem division_problem (x y z : ℕ) : 
  x > 0 → 
  x = 7 * y + 3 → 
  2 * x = 3 * y * z + 2 → 
  11 * y - x = 1 → 
  z = 6 := by
sorry

end division_problem_l4020_402053


namespace dataset_mode_l4020_402017

def dataset : List Nat := [3, 1, 3, 0, 3, 2, 1, 2]

def mode (l : List Nat) : Nat :=
  l.foldl (λ acc x => if l.count x > l.count acc then x else acc) 0

theorem dataset_mode :
  mode dataset = 3 := by
  sorry

end dataset_mode_l4020_402017


namespace regular_square_pyramid_volume_l4020_402022

/-- The volume of a regular square pyramid with base edge length 2 and side edge length √6 is 8/3. -/
theorem regular_square_pyramid_volume :
  ∀ (base_edge side_edge volume : ℝ),
    base_edge = 2 →
    side_edge = Real.sqrt 6 →
    volume = (1 / 3) * base_edge ^ 2 * Real.sqrt (side_edge ^ 2 - (base_edge ^ 2 / 2)) →
    volume = 8 / 3 := by
  sorry

end regular_square_pyramid_volume_l4020_402022


namespace greatest_multiple_of_nine_with_unique_digits_mod_100_l4020_402038

/-- A function that checks if all digits of a natural number are unique -/
def has_unique_digits (n : ℕ) : Prop := sorry

/-- The greatest integer multiple of 9 with all unique digits -/
def M : ℕ := sorry

theorem greatest_multiple_of_nine_with_unique_digits_mod_100 :
  M % 9 = 0 ∧ has_unique_digits M ∧ (∀ k : ℕ, k % 9 = 0 → has_unique_digits k → k ≤ M) →
  M % 100 = 81 := by sorry

end greatest_multiple_of_nine_with_unique_digits_mod_100_l4020_402038


namespace inequality_solution_set_l4020_402067

theorem inequality_solution_set (a : ℝ) :
  let solution_set := {x : ℝ | x^2 - (a^2 + a)*x + a^3 < 0}
  (a = 0 ∨ a = 1 → solution_set = ∅) ∧
  (0 < a ∧ a < 1 → solution_set = {x : ℝ | a^2 < x ∧ x < a}) ∧
  ((a < 0 ∨ a > 1) → solution_set = {x : ℝ | a < x ∧ x < a^2}) :=
by sorry

end inequality_solution_set_l4020_402067


namespace boys_to_girls_ratio_l4020_402096

theorem boys_to_girls_ratio : 
  ∀ (boys girls : ℕ), 
    boys = 40 →
    girls = boys + 64 →
    (boys : ℚ) / (girls : ℚ) = 5 / 13 := by
  sorry

end boys_to_girls_ratio_l4020_402096


namespace product_105_95_l4020_402026

theorem product_105_95 : 105 * 95 = 9975 := by
  sorry

end product_105_95_l4020_402026


namespace gcd_288_123_l4020_402057

theorem gcd_288_123 : Nat.gcd 288 123 = 3 := by
  sorry

end gcd_288_123_l4020_402057


namespace blueberry_picking_total_l4020_402023

/-- The total number of pints of blueberries picked by Annie, Kathryn, and Ben -/
def total_pints (annie kathryn ben : ℕ) : ℕ := annie + kathryn + ben

/-- Theorem stating the total number of pints picked given the conditions -/
theorem blueberry_picking_total :
  ∀ (annie kathryn ben : ℕ),
  annie = 8 →
  kathryn = annie + 2 →
  ben = kathryn - 3 →
  total_pints annie kathryn ben = 25 :=
by
  sorry

end blueberry_picking_total_l4020_402023


namespace optimal_sampling_methods_for_school_scenario_l4020_402037

/-- Represents the different blood types --/
inductive BloodType
| O
| A
| B
| AB

/-- Represents the available sampling methods --/
inductive SamplingMethod
| Random
| Systematic
| Stratified

/-- Structure representing the school scenario --/
structure SchoolScenario where
  total_students : Nat
  blood_type_distribution : BloodType → Nat
  sample_size_blood_study : Nat
  soccer_team_size : Nat
  sample_size_soccer_study : Nat

/-- Determines the optimal sampling method for a given scenario and study type --/
def optimal_sampling_method (scenario : SchoolScenario) (is_blood_study : Bool) : SamplingMethod :=
  if is_blood_study then SamplingMethod.Stratified else SamplingMethod.Random

/-- Theorem stating the optimal sampling methods for the given school scenario --/
theorem optimal_sampling_methods_for_school_scenario 
  (scenario : SchoolScenario)
  (h1 : scenario.total_students = 500)
  (h2 : scenario.blood_type_distribution BloodType.O = 200)
  (h3 : scenario.blood_type_distribution BloodType.A = 125)
  (h4 : scenario.blood_type_distribution BloodType.B = 125)
  (h5 : scenario.blood_type_distribution BloodType.AB = 50)
  (h6 : scenario.sample_size_blood_study = 20)
  (h7 : scenario.soccer_team_size = 11)
  (h8 : scenario.sample_size_soccer_study = 2) :
  (optimal_sampling_method scenario true = SamplingMethod.Stratified) ∧
  (optimal_sampling_method scenario false = SamplingMethod.Random) :=
sorry

end optimal_sampling_methods_for_school_scenario_l4020_402037


namespace sphere_volume_in_cone_l4020_402030

/-- A right circular cone with a sphere inscribed inside it -/
structure ConeWithSphere where
  /-- The diameter of the cone's base in inches -/
  base_diameter : ℝ
  /-- The vertex angle of the cross-section triangle in degrees -/
  vertex_angle : ℝ

/-- Calculate the volume of the inscribed sphere -/
def sphere_volume (cone : ConeWithSphere) : ℝ :=
  sorry

/-- Theorem stating the volume of the inscribed sphere in the given cone -/
theorem sphere_volume_in_cone (cone : ConeWithSphere) 
  (h1 : cone.base_diameter = 24)
  (h2 : cone.vertex_angle = 90) : 
  sphere_volume cone = 2304 * Real.pi :=
sorry

end sphere_volume_in_cone_l4020_402030


namespace orthocenter_proof_l4020_402063

/-- A point in a 2D plane -/
structure Point := (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral := (A B C D : Point)

/-- A triangle defined by three points -/
structure Triangle := (D P Q : Point)

/-- Checks if a quadrilateral is a rhombus -/
def is_rhombus (quad : Quadrilateral) : Prop := sorry

/-- Checks if a quadrilateral is a parallelogram -/
def is_parallelogram (quad : Quadrilateral) : Prop := sorry

/-- Checks if a point lies inside a quadrilateral -/
def point_inside (P : Point) (quad : Quadrilateral) : Prop := sorry

/-- Checks if two line segments have equal length -/
def segments_equal (A B C D : Point) : Prop := sorry

/-- Checks if a point is the orthocenter of a triangle -/
def is_orthocenter (P : Point) (tri : Triangle) : Prop := sorry

theorem orthocenter_proof (A B C D P Q : Point) :
  let ABCD := Quadrilateral.mk A B C D
  let APQC := Quadrilateral.mk A P Q C
  let DPQ := Triangle.mk D P Q
  is_rhombus ABCD →
  is_parallelogram APQC →
  point_inside B APQC →
  segments_equal A P A B →
  is_orthocenter B DPQ := by
  sorry

end orthocenter_proof_l4020_402063


namespace cube_root_unity_product_l4020_402092

theorem cube_root_unity_product (ω : ℂ) : 
  ω ≠ 1 → ω^3 = 1 → (1 - ω + ω^2) * (1 + ω - ω^2) = 4 := by sorry

end cube_root_unity_product_l4020_402092


namespace picnic_men_count_l4020_402079

/-- Represents the number of people at a picnic -/
structure PicnicAttendance where
  total : Nat
  men : Nat
  women : Nat
  adults : Nat
  children : Nat

/-- Defines the conditions for a valid picnic attendance -/
def ValidPicnicAttendance (p : PicnicAttendance) : Prop :=
  p.total = 200 ∧
  p.men = p.women + 20 ∧
  p.adults = p.children + 20 ∧
  p.total = p.men + p.women + p.children ∧
  p.adults = p.men + p.women

theorem picnic_men_count (p : PicnicAttendance) (h : ValidPicnicAttendance p) : p.men = 65 := by
  sorry

end picnic_men_count_l4020_402079


namespace first_year_after_2010_with_digit_sum_15_l4020_402051

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2010 ∧ digit_sum year = 15

theorem first_year_after_2010_with_digit_sum_15 :
  ∀ year : ℕ, is_valid_year year → year ≥ 2049 :=
sorry

end first_year_after_2010_with_digit_sum_15_l4020_402051


namespace mardi_gras_necklaces_mardi_gras_necklaces_proof_l4020_402085

theorem mardi_gras_necklaces : Int → Int → Int → Prop :=
  fun boudreaux rhonda latch =>
    boudreaux = 12 ∧
    rhonda = boudreaux / 2 ∧
    latch = 3 * rhonda - 4 →
    latch = 14

-- The proof is omitted
theorem mardi_gras_necklaces_proof : mardi_gras_necklaces 12 6 14 := by
  sorry

end mardi_gras_necklaces_mardi_gras_necklaces_proof_l4020_402085


namespace pq_equals_10_l4020_402029

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ × ℝ)

-- Define properties of the triangle
def isRightAngled (t : Triangle) : Prop := sorry
def anglePRQ (t : Triangle) : ℝ := sorry
def lengthPR (t : Triangle) : ℝ := sorry
def lengthPQ (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem pq_equals_10 (t : Triangle) 
  (h1 : isRightAngled t) 
  (h2 : anglePRQ t = 45) 
  (h3 : lengthPR t = 10) : 
  lengthPQ t = 10 := by sorry

end pq_equals_10_l4020_402029


namespace simplify_polynomial_l4020_402080

theorem simplify_polynomial (r : ℝ) : (2 * r^2 + 5 * r - 3) - (r^2 + 4 * r - 6) = r^2 + r + 3 := by
  sorry

end simplify_polynomial_l4020_402080


namespace intersection_nonempty_implies_a_greater_than_neg_one_l4020_402083

open Set

theorem intersection_nonempty_implies_a_greater_than_neg_one (a : ℝ) :
  let M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
  let N : Set ℝ := {y | y < a}
  (M ∩ N).Nonempty → a > -1 := by
  sorry

end intersection_nonempty_implies_a_greater_than_neg_one_l4020_402083


namespace pencil_cost_l4020_402008

/-- Given that 120 pencils cost $36, prove that 3000 pencils cost $900 -/
theorem pencil_cost (cost_120 : ℕ) (quantity : ℕ) (h1 : cost_120 = 36) (h2 : quantity = 3000) :
  (cost_120 * quantity) / 120 = 900 := by
  sorry

end pencil_cost_l4020_402008


namespace sqrt_equation_solution_l4020_402001

theorem sqrt_equation_solution : ∃! x : ℝ, Real.sqrt (2 * x + 3) = x := by
  sorry

end sqrt_equation_solution_l4020_402001


namespace four_row_arrangement_has_fourteen_triangles_l4020_402007

/-- Represents a triangular arrangement of smaller triangles. -/
structure TriangularArrangement where
  rows : Nat
  bottom_row_triangles : Nat

/-- Calculates the total number of triangles in the arrangement. -/
def total_triangles (arr : TriangularArrangement) : Nat :=
  sorry

/-- Theorem stating that a triangular arrangement with 4 rows and 4 triangles
    in the bottom row has a total of 14 triangles. -/
theorem four_row_arrangement_has_fourteen_triangles :
  ∀ (arr : TriangularArrangement),
    arr.rows = 4 →
    arr.bottom_row_triangles = 4 →
    total_triangles arr = 14 :=
  sorry

end four_row_arrangement_has_fourteen_triangles_l4020_402007


namespace nobel_prize_laureates_l4020_402058

theorem nobel_prize_laureates (total_scientists : ℕ) 
                               (wolf_prize : ℕ) 
                               (wolf_and_nobel : ℕ) 
                               (non_wolf_nobel_diff : ℕ) : 
  total_scientists = 50 → 
  wolf_prize = 31 → 
  wolf_and_nobel = 12 → 
  non_wolf_nobel_diff = 3 → 
  (total_scientists - wolf_prize + wolf_and_nobel : ℕ) = 23 := by
  sorry

end nobel_prize_laureates_l4020_402058


namespace alices_money_is_64_dollars_l4020_402069

/-- Represents the value of Alice's money after exchanging quarters for nickels -/
def alices_money_value (num_quarters : ℕ) (iron_nickel_percentage : ℚ) 
  (iron_nickel_value : ℚ) (regular_nickel_value : ℚ) : ℚ :=
  let total_cents := num_quarters * 25
  let total_nickels := total_cents / 5
  let iron_nickels := iron_nickel_percentage * total_nickels
  let regular_nickels := total_nickels - iron_nickels
  iron_nickels * iron_nickel_value + regular_nickels * regular_nickel_value

/-- Theorem stating that Alice's money value after exchange is $64 -/
theorem alices_money_is_64_dollars :
  alices_money_value 20 (1/5) 300 (5/100) = 6400/100 := by
  sorry

end alices_money_is_64_dollars_l4020_402069


namespace fraction_evaluation_l4020_402068

theorem fraction_evaluation : 
  (1 - (1/4 + 1/5)) / (1 - 2/3) = 33/20 := by sorry

end fraction_evaluation_l4020_402068


namespace equation_solution_l4020_402032

theorem equation_solution : 
  ∃! x : ℚ, x ≠ 3 ∧ (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ∧ x = -7/6 := by
  sorry

end equation_solution_l4020_402032


namespace inequality_bound_l4020_402020

theorem inequality_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (a / x + b / y) > M) →
  M < a + b + 2 * Real.sqrt (a * b) :=
by sorry

end inequality_bound_l4020_402020


namespace floor_plus_self_unique_l4020_402019

theorem floor_plus_self_unique (r : ℝ) : ⌊r⌋ + r = 15.75 ↔ r = 7.25 := by
  sorry

end floor_plus_self_unique_l4020_402019


namespace cos_double_angle_special_case_l4020_402059

/-- Given an angle θ formed by the positive x-axis and a line passing through 
    the origin and the point (-3,1), prove that cos(2θ) = 4/5 -/
theorem cos_double_angle_special_case : 
  ∀ θ : Real, 
  (∃ (x y : Real), x = -3 ∧ y = 1 ∧ x = Real.cos θ * Real.sqrt (x^2 + y^2) ∧ 
                    y = Real.sin θ * Real.sqrt (x^2 + y^2)) → 
  Real.cos (2 * θ) = 4/5 := by
  sorry

end cos_double_angle_special_case_l4020_402059


namespace f_properties_l4020_402018

-- Define the function f
def f (x b c : ℝ) : ℝ := x * |x| + b * x + c

-- Theorem statement
theorem f_properties :
  (∀ x, f x 0 0 = -f (-x) 0 0) ∧
  (∀ x, f x 0 (0 : ℝ) = 0 → x = 0) ∧
  (∀ x, f (x - 0) b c = f (-x - 0) b c + 2 * c) ∧
  (∃ b c, ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x b c = 0 ∧ f y b c = 0 ∧ f z b c = 0) :=
by sorry

end f_properties_l4020_402018


namespace hexagon_triangle_quadrilateral_area_ratio_l4020_402064

/-- A regular hexagon with vertices labeled A to F. -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  vertices : Fin 3 → ℝ × ℝ

/-- A quadrilateral -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- The area of a polygon -/
noncomputable def area {n : ℕ} (vertices : Fin n → ℝ × ℝ) : ℝ := sorry

theorem hexagon_triangle_quadrilateral_area_ratio
  (h : RegularHexagon)
  (triangles : Fin 6 → EquilateralTriangle)
  (quad : Quadrilateral) :
  (∀ i, area (triangles i).vertices = area (triangles 0).vertices) →
  (quad.vertices 0 = h.vertices 0) →
  (quad.vertices 1 = h.vertices 2) →
  (quad.vertices 2 = h.vertices 4) →
  (quad.vertices 3 = h.vertices 1) →
  area (triangles 0).vertices / area quad.vertices = 1 / 4 := by
  sorry

end hexagon_triangle_quadrilateral_area_ratio_l4020_402064


namespace cat_mouse_positions_after_196_moves_l4020_402006

/-- Represents the four squares in the grid --/
inductive Square
| TopLeft
| TopRight
| BottomLeft
| BottomRight

/-- Represents the eight outer segments of the squares --/
inductive Segment
| TopLeft
| TopMiddle
| TopRight
| RightMiddle
| BottomRight
| BottomMiddle
| BottomLeft
| LeftMiddle

/-- The cat's position after a given number of moves --/
def catPosition (moves : Nat) : Square :=
  match moves % 4 with
  | 0 => Square.TopLeft
  | 1 => Square.BottomLeft
  | 2 => Square.BottomRight
  | 3 => Square.TopRight
  | _ => Square.TopLeft  -- This case is unreachable, but needed for exhaustiveness

/-- The mouse's position after a given number of moves --/
def mousePosition (moves : Nat) : Segment :=
  match moves % 8 with
  | 0 => Segment.TopMiddle
  | 1 => Segment.TopRight
  | 2 => Segment.RightMiddle
  | 3 => Segment.BottomRight
  | 4 => Segment.BottomMiddle
  | 5 => Segment.BottomLeft
  | 6 => Segment.LeftMiddle
  | 7 => Segment.TopLeft
  | _ => Segment.TopMiddle  -- This case is unreachable, but needed for exhaustiveness

theorem cat_mouse_positions_after_196_moves :
  catPosition 196 = Square.TopLeft ∧ mousePosition 196 = Segment.BottomMiddle := by
  sorry


end cat_mouse_positions_after_196_moves_l4020_402006


namespace remainder_problem_l4020_402047

theorem remainder_problem (n : ℕ) (h1 : (1661 - 10) % n = 0) (h2 : (2045 - 13) % n = 0) (h3 : n = 127) : 
  13 = 2045 % n :=
by sorry

end remainder_problem_l4020_402047


namespace third_score_proof_l4020_402034

/-- Given three scores with an average of 122, where two scores are 118 and 125, prove the third score is 123. -/
theorem third_score_proof (average : ℝ) (score1 score2 : ℝ) (h_average : average = 122) 
  (h_score1 : score1 = 118) (h_score2 : score2 = 125) : 
  3 * average - (score1 + score2) = 123 := by
  sorry

end third_score_proof_l4020_402034


namespace remainder_sum_l4020_402073

theorem remainder_sum (c d : ℤ) 
  (hc : c % 60 = 53) 
  (hd : d % 42 = 35) : 
  (c + d) % 21 = 4 := by
  sorry

end remainder_sum_l4020_402073


namespace quadratic_roots_properties_l4020_402004

theorem quadratic_roots_properties (x₁ x₂ k m : ℝ) : 
  x₁ + x₂ + x₁ * x₂ = 2 * m + k →
  (x₁ - 1) * (x₂ - 1) = m + 1 - k →
  x₁ - x₂ = 1 →
  k - m = 1 →
  (k^2 > 4 * m) ∧ ((m = 0 ∧ k = 1) ∨ (m = 2 ∧ k = 3)) :=
by sorry

end quadratic_roots_properties_l4020_402004


namespace connie_gave_juan_marbles_l4020_402031

/-- The number of marbles Connie gave to Juan -/
def marbles_given (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Proof that Connie gave 183 marbles to Juan -/
theorem connie_gave_juan_marbles : marbles_given 776 593 = 183 := by
  sorry

end connie_gave_juan_marbles_l4020_402031


namespace greatest_prime_factor_of_391_l4020_402090

theorem greatest_prime_factor_of_391 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 391 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 391 → q ≤ p :=
by sorry

end greatest_prime_factor_of_391_l4020_402090


namespace tank_depth_l4020_402025

/-- Calculates the depth of a tank given its dimensions and plastering costs -/
theorem tank_depth (length width : ℝ) (plaster_cost_per_sqm total_cost : ℝ) : 
  length = 25 → 
  width = 12 → 
  plaster_cost_per_sqm = 0.75 → 
  total_cost = 558 → 
  ∃ depth : ℝ, 
    plaster_cost_per_sqm * (2 * (length * depth) + 2 * (width * depth) + (length * width)) = total_cost ∧ 
    depth = 6 := by
  sorry

end tank_depth_l4020_402025


namespace regina_farm_sale_price_l4020_402088

/-- Calculates the total sale price of animals on Regina's farm -/
def total_sale_price (num_cows : ℕ) (cow_price pig_price goat_price chicken_price rabbit_price : ℕ) : ℕ :=
  let num_pigs := 4 * num_cows
  let num_goats := num_pigs / 2
  let num_chickens := 2 * num_cows
  let num_rabbits := 30
  num_cows * cow_price + num_pigs * pig_price + num_goats * goat_price + 
  num_chickens * chicken_price + num_rabbits * rabbit_price

/-- Theorem stating that the total sale price of all animals on Regina's farm is $74,750 -/
theorem regina_farm_sale_price :
  total_sale_price 20 800 400 600 50 25 = 74750 := by
  sorry

end regina_farm_sale_price_l4020_402088


namespace consecutive_integers_sum_l4020_402005

theorem consecutive_integers_sum (n : ℤ) : 
  (n - 1) * n * (n + 1) = 336 → (n - 1) + n + (n + 1) = 21 := by
sorry

end consecutive_integers_sum_l4020_402005


namespace mangoes_harvested_l4020_402039

theorem mangoes_harvested (neighbors : ℕ) (mangoes_per_neighbor : ℕ) 
  (h1 : neighbors = 8)
  (h2 : mangoes_per_neighbor = 35)
  (h3 : ∃ (total : ℕ), total / 2 = neighbors * mangoes_per_neighbor) :
  ∃ (total : ℕ), total = 560 := by
  sorry

end mangoes_harvested_l4020_402039


namespace date_statistics_order_l4020_402021

def date_counts : List (Nat × Nat) :=
  (List.range 29).map (λ n => (n + 1, 12)) ++
  [(30, 11), (31, 7)]

def total_count : Nat := date_counts.foldl (λ acc (_, count) => acc + count) 0

def sum_of_values : Nat := date_counts.foldl (λ acc (date, count) => acc + date * count) 0

def mean : ℚ := sum_of_values / total_count

def median : ℚ :=
  let mid_point := (total_count + 1) / 2
  16

def median_of_modes : Nat := 15

theorem date_statistics_order :
  (median_of_modes : ℚ) < mean ∧ mean < median := by sorry

end date_statistics_order_l4020_402021


namespace girls_in_class_l4020_402052

/-- Represents the number of girls in a class given the total number of students and the ratio of girls to boys. -/
def number_of_girls (total : ℕ) (girl_ratio : ℕ) (boy_ratio : ℕ) : ℕ :=
  (total * girl_ratio) / (girl_ratio + boy_ratio)

/-- Theorem stating that in a class of 63 students with a girl-to-boy ratio of 4:3, there are 36 girls. -/
theorem girls_in_class : number_of_girls 63 4 3 = 36 := by
  sorry

end girls_in_class_l4020_402052
