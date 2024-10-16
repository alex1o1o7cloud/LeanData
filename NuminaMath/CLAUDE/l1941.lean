import Mathlib

namespace NUMINAMATH_CALUDE_least_multiple_and_digit_sum_l1941_194163

def least_multiple_of_17_gt_500 : ℕ := 510

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem least_multiple_and_digit_sum :
  (least_multiple_of_17_gt_500 % 17 = 0) ∧
  (least_multiple_of_17_gt_500 > 500) ∧
  (∀ m : ℕ, m % 17 = 0 ∧ m > 500 → m ≥ least_multiple_of_17_gt_500) ∧
  (sum_of_digits least_multiple_of_17_gt_500 = 6) :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_and_digit_sum_l1941_194163


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l1941_194100

theorem sum_of_solutions_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (a * r₁^2 + b * r₁ + c = 0) ∧ (a * r₂^2 + b * r₂ + c = 0) →
  r₁ + r₂ = -b / a := by
  sorry

theorem sum_of_solutions_specific_quadratic :
  let r₁ := (-(-15) + Real.sqrt ((-15)^2 - 4*(-1)*54)) / (2*(-1))
  let r₂ := (-(-15) - Real.sqrt ((-15)^2 - 4*(-1)*54)) / (2*(-1))
  (-r₁^2 - 15*r₁ + 54 = 0) ∧ (-r₂^2 - 15*r₂ + 54 = 0) →
  r₁ + r₂ = -15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_sum_of_solutions_specific_quadratic_l1941_194100


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1941_194112

theorem complex_equation_solution :
  ∀ z : ℂ, (1 + 2*I)*z = 4 + 3*I → z = 2 - I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1941_194112


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_is_one_l1941_194108

theorem fraction_zero_implies_x_is_one (x : ℝ) :
  (x - 1) / (x - 3) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_is_one_l1941_194108


namespace NUMINAMATH_CALUDE_divisibility_by_seven_l1941_194143

theorem divisibility_by_seven (n : ℕ) : 
  ∃ k : ℤ, ((-8)^(2019 : ℕ) + (-8)^(2018 : ℕ)) = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_seven_l1941_194143


namespace NUMINAMATH_CALUDE_alloy_problem_solution_l1941_194149

/-- Represents the copper-tin alloy problem -/
structure AlloyProblem where
  mass1 : ℝ  -- Mass of the first alloy
  copper1 : ℝ  -- Copper percentage in the first alloy
  mass2 : ℝ  -- Mass of the second alloy
  copper2 : ℝ  -- Copper percentage in the second alloy
  targetMass : ℝ  -- Target mass of the resulting alloy

/-- Represents the solution to the copper-tin alloy problem -/
structure AlloySolution where
  pMin : ℝ  -- Minimum percentage of copper in the resulting alloy
  pMax : ℝ  -- Maximum percentage of copper in the resulting alloy
  mass1 : ℝ → ℝ  -- Function to calculate mass of the first alloy
  mass2 : ℝ → ℝ  -- Function to calculate mass of the second alloy

/-- Theorem stating the solution to the copper-tin alloy problem -/
theorem alloy_problem_solution (problem : AlloyProblem) 
  (h1 : problem.mass1 = 4) 
  (h2 : problem.copper1 = 40) 
  (h3 : problem.mass2 = 6) 
  (h4 : problem.copper2 = 30) 
  (h5 : problem.targetMass = 8) :
  ∃ (solution : AlloySolution),
    solution.pMin = 32.5 ∧
    solution.pMax = 35 ∧
    (∀ p, solution.mass1 p = 0.8 * p - 24) ∧
    (∀ p, solution.mass2 p = 32 - 0.8 * p) ∧
    (∀ p, 32.5 ≤ p → p ≤ 35 → 
      0 ≤ solution.mass1 p ∧ 
      solution.mass1 p ≤ problem.mass1 ∧
      0 ≤ solution.mass2 p ∧ 
      solution.mass2 p ≤ problem.mass2 ∧
      solution.mass1 p + solution.mass2 p = problem.targetMass ∧
      solution.mass1 p * (problem.copper1 / 100) + solution.mass2 p * (problem.copper2 / 100) = 
        problem.targetMass * (p / 100)) :=
by
  sorry

end NUMINAMATH_CALUDE_alloy_problem_solution_l1941_194149


namespace NUMINAMATH_CALUDE_not_linear_in_M_f_expression_for_negative_range_sin_k_in_M_iff_l1941_194172

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∃ (T : ℝ) (hT : T ≠ 0), ∀ x, f (x + T) = T * f x}

-- Theorem 1
theorem not_linear_in_M : ¬(λ x : ℝ => x) ∈ M := by sorry

-- Theorem 2
theorem f_expression_for_negative_range 
  (f : ℝ → ℝ) (hf : f ∈ M) (hT : ∃ T, T = 2 ∧ ∀ x, 1 < x → x < 2 → f x = x + Real.log x) :
  ∀ x, -3 < x → x < -2 → f x = (1/4) * (x + 4 + Real.log (x + 4)) := by sorry

-- Theorem 3
theorem sin_k_in_M_iff (k : ℝ) : 
  (λ x : ℝ => Real.sin (k * x)) ∈ M ↔ ∃ m : ℤ, k = m * Real.pi := by sorry

end NUMINAMATH_CALUDE_not_linear_in_M_f_expression_for_negative_range_sin_k_in_M_iff_l1941_194172


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1941_194183

theorem greatest_sum_consecutive_integers (n : ℤ) : 
  (n * (n + 1) < 360) → (∀ m : ℤ, m > n → m * (m + 1) ≥ 360) → n + (n + 1) = 37 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1941_194183


namespace NUMINAMATH_CALUDE_square_root_five_plus_one_squared_minus_two_times_plus_seven_equals_eleven_l1941_194119

theorem square_root_five_plus_one_squared_minus_two_times_plus_seven_equals_eleven :
  (Real.sqrt 5 + 1)^2 - 2 * (Real.sqrt 5 + 1) + 7 = 11 := by sorry

end NUMINAMATH_CALUDE_square_root_five_plus_one_squared_minus_two_times_plus_seven_equals_eleven_l1941_194119


namespace NUMINAMATH_CALUDE_solution_set_of_f_positive_l1941_194109

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 5*x + 6

-- State the theorem
theorem solution_set_of_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x > 3 ∨ x < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_positive_l1941_194109


namespace NUMINAMATH_CALUDE_total_yardage_progress_l1941_194107

def team_a_moves : List Int := [-5, 8, -3, 6]
def team_b_moves : List Int := [4, -2, 9, -7]

theorem total_yardage_progress : 
  (team_a_moves.sum + team_b_moves.sum) = 10 := by sorry

end NUMINAMATH_CALUDE_total_yardage_progress_l1941_194107


namespace NUMINAMATH_CALUDE_max_chesslike_subsquares_l1941_194144

/-- Represents the color of a square on the board -/
inductive Color
| Red
| Green

/-- Represents a 6x6 board -/
def Board := Fin 6 → Fin 6 → Color

/-- Checks if four adjacent squares in a given direction are of the same color -/
def fourAdjacentSameColor (board : Board) : Bool := sorry

/-- Checks if a 2x2 subsquare is chesslike -/
def isChesslike (board : Board) (row col : Fin 5) : Bool := sorry

/-- Counts the number of chesslike 2x2 subsquares on the board -/
def countChesslike (board : Board) : Nat := sorry

/-- Theorem: The maximal number of chesslike 2x2 subsquares on a 6x6 board 
    with the given constraints is 25 -/
theorem max_chesslike_subsquares (board : Board) 
  (h : ¬fourAdjacentSameColor board) : 
  (∃ (b : Board), ¬fourAdjacentSameColor b ∧ countChesslike b = 25) ∧ 
  (∀ (b : Board), ¬fourAdjacentSameColor b → countChesslike b ≤ 25) := by
  sorry

end NUMINAMATH_CALUDE_max_chesslike_subsquares_l1941_194144


namespace NUMINAMATH_CALUDE_gym_time_is_two_hours_l1941_194102

/-- Represents the daily schedule of a working mom --/
structure DailySchedule where
  wakeTime : Nat
  sleepTime : Nat
  workHours : Nat
  cookingTime : Real
  bathTime : Real
  homeworkTime : Real
  lunchPackingTime : Real
  cleaningTime : Real
  leisureTime : Real

/-- Calculates the total awake hours in a day --/
def awakeHours (schedule : DailySchedule) : Nat :=
  schedule.sleepTime - schedule.wakeTime

/-- Calculates the total time spent on activities excluding work and gym --/
def otherActivitiesTime (schedule : DailySchedule) : Real :=
  schedule.cookingTime + schedule.bathTime + schedule.homeworkTime +
  schedule.lunchPackingTime + schedule.cleaningTime + schedule.leisureTime

/-- Theorem: The working mom spends 2 hours at the gym --/
theorem gym_time_is_two_hours (schedule : DailySchedule) 
    (h1 : schedule.wakeTime = 7)
    (h2 : schedule.sleepTime = 23)
    (h3 : schedule.workHours = 8)
    (h4 : schedule.workHours = awakeHours schedule / 2)
    (h5 : schedule.cookingTime = 1.5)
    (h6 : schedule.bathTime = 0.5)
    (h7 : schedule.homeworkTime = 1)
    (h8 : schedule.lunchPackingTime = 0.5)
    (h9 : schedule.cleaningTime = 0.5)
    (h10 : schedule.leisureTime = 2) :
    awakeHours schedule - schedule.workHours - otherActivitiesTime schedule = 2 := by
  sorry


end NUMINAMATH_CALUDE_gym_time_is_two_hours_l1941_194102


namespace NUMINAMATH_CALUDE_solve_for_a_l1941_194166

theorem solve_for_a : ∀ a : ℝ, (∃ x : ℝ, x = 1 ∧ 2 * x - a = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1941_194166


namespace NUMINAMATH_CALUDE_fibonacci_sum_cube_square_l1941_194196

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define a predicate for Fibonacci numbers
def isFibonacci (n : ℕ) : Prop := ∃ k, fib k = n

-- Define the theorem
theorem fibonacci_sum_cube_square :
  ∀ a b : ℕ,
  isFibonacci a ∧ 49 < a ∧ a < 61 ∧
  isFibonacci b ∧ 59 < b ∧ b < 71 →
  a^3 + b^2 = 170096 :=
sorry

end NUMINAMATH_CALUDE_fibonacci_sum_cube_square_l1941_194196


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1941_194122

/-- Given a hyperbola with equation x²/4 - y²/12 = 1, prove its eccentricity and asymptotes. -/
theorem hyperbola_properties :
  let a := 2
  let b := 2 * Real.sqrt 3
  let c := 4
  let e := c / a
  let asymptote (x : ℝ) := Real.sqrt 3 * x
  (∀ x y : ℝ, x^2/4 - y^2/12 = 1 →
    (e = 2 ∧
    (∀ x : ℝ, y = asymptote x ∨ y = -asymptote x))) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1941_194122


namespace NUMINAMATH_CALUDE_mary_ray_difference_l1941_194195

/-- The number of chickens taken by each person -/
structure ChickenDistribution where
  john : ℕ
  mary : ℕ
  ray : ℕ

/-- The conditions of the chicken distribution problem -/
def valid_distribution (d : ChickenDistribution) : Prop :=
  d.john = d.mary + 5 ∧
  d.ray < d.mary ∧
  d.ray = 10 ∧
  d.john = d.ray + 11

/-- The theorem stating the difference between Mary's and Ray's chickens -/
theorem mary_ray_difference (d : ChickenDistribution) 
  (h : valid_distribution d) : d.mary - d.ray = 6 := by
  sorry

#check mary_ray_difference

end NUMINAMATH_CALUDE_mary_ray_difference_l1941_194195


namespace NUMINAMATH_CALUDE_min_value_theorem_l1941_194128

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 2/y + 3/z = 1) :
  x + y/2 + z/3 ≥ 9 ∧ (x + y/2 + z/3 = 9 ↔ x = 3 ∧ y = 6 ∧ z = 9) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1941_194128


namespace NUMINAMATH_CALUDE_fraction_simplification_l1941_194126

theorem fraction_simplification :
  (5 - Real.sqrt 4) / (5 + Real.sqrt 4) = 3 / 7 :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1941_194126


namespace NUMINAMATH_CALUDE_total_water_poured_l1941_194199

/-- 
Given two bottles with capacities of 4 and 8 cups respectively, 
if they are filled to the same fraction of their capacity and 
5.333333333333333 cups of water are poured into the 8-cup bottle, 
then the total amount of water poured into both bottles is 8 cups.
-/
theorem total_water_poured (bottle1_capacity bottle2_capacity : ℝ) 
  (water_in_bottle2 : ℝ) : 
  bottle1_capacity = 4 →
  bottle2_capacity = 8 →
  water_in_bottle2 = 5.333333333333333 →
  (water_in_bottle2 / bottle2_capacity) * bottle1_capacity + water_in_bottle2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_water_poured_l1941_194199


namespace NUMINAMATH_CALUDE_xy_range_and_min_x_plus_2y_l1941_194129

theorem xy_range_and_min_x_plus_2y (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x + y + x*y = 3) : 
  (0 < x*y ∧ x*y ≤ 1) ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → a + b + a*b = 3 → x + 2*y ≤ a + 2*b) ∧
  (∃ c d : ℝ, c > 0 ∧ d > 0 ∧ c + d + c*d = 3 ∧ c + 2*d = 4*Real.sqrt 2 - 3) :=
by sorry

end NUMINAMATH_CALUDE_xy_range_and_min_x_plus_2y_l1941_194129


namespace NUMINAMATH_CALUDE_inhabitable_earth_surface_fraction_l1941_194191

theorem inhabitable_earth_surface_fraction :
  let total_surface := 1
  let land_fraction := (1 : ℚ) / 3
  let inhabitable_land_fraction := (2 : ℚ) / 3
  (land_fraction * inhabitable_land_fraction : ℚ) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inhabitable_earth_surface_fraction_l1941_194191


namespace NUMINAMATH_CALUDE_quiz_smallest_n_l1941_194170

/-- The smallest possible value of n in the quiz problem -/
theorem quiz_smallest_n : ∃ (n : ℤ), n = 89 ∧ 
  ∀ (m : ℕ+) (n' : ℤ),
  (m : ℤ) * (n' + 2) - m * (m + 1) = 2009 →
  n ≤ n' :=
by sorry

end NUMINAMATH_CALUDE_quiz_smallest_n_l1941_194170


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_25_l1941_194161

theorem cubic_fraction_equals_25 (a b : ℝ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b^3)^2 / (a^2 - a*b + b^2)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_25_l1941_194161


namespace NUMINAMATH_CALUDE_tree_height_difference_l1941_194190

theorem tree_height_difference : 
  let pine_height : ℚ := 49/4
  let maple_height : ℚ := 75/4
  maple_height - pine_height = 13/2 := by sorry

end NUMINAMATH_CALUDE_tree_height_difference_l1941_194190


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l1941_194140

theorem multiplication_puzzle :
  ∀ P Q R : ℕ,
    P ≠ Q → P ≠ R → Q ≠ R →
    P < 10 → Q < 10 → R < 10 →
    (100 * P + 10 * P + Q) * Q = 1000 * R + 100 * Q + 50 + Q →
    P + Q + R = 17 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l1941_194140


namespace NUMINAMATH_CALUDE_kayak_trip_remaining_fraction_l1941_194114

/-- Given a kayak trip with total distance and distance paddled before lunch,
    calculate the fraction of the trip remaining after lunch -/
theorem kayak_trip_remaining_fraction
  (total_distance : ℝ)
  (distance_before_lunch : ℝ)
  (h1 : total_distance = 36)
  (h2 : distance_before_lunch = 12)
  : (total_distance - distance_before_lunch) / total_distance = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_kayak_trip_remaining_fraction_l1941_194114


namespace NUMINAMATH_CALUDE_rationalize_and_sum_l1941_194137

theorem rationalize_and_sum : ∃ (A B C D E F : ℤ),
  (F > 0) ∧
  (∃ (k : ℚ), k ≠ 0 ∧ 
    k * (1 / (Real.sqrt 5 + Real.sqrt 3 + Real.sqrt 11)) = 
    (A * Real.sqrt 5 + B * Real.sqrt 3 + C * Real.sqrt 11 + D * Real.sqrt E) / F) ∧
  (A + B + C + D + E + F = 196) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_sum_l1941_194137


namespace NUMINAMATH_CALUDE_unique_prime_evaluation_l1941_194160

theorem unique_prime_evaluation (T : ℕ) (h : T = 2161) :
  ∃! p : ℕ, Prime p ∧ ∃ n : ℤ, n^4 - 898*n^2 + T - 2160 = p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_evaluation_l1941_194160


namespace NUMINAMATH_CALUDE_probability_black_then_white_l1941_194111

/-- The probability of drawing a black ball first and then a white ball from a bag. -/
theorem probability_black_then_white (white_balls black_balls : ℕ) : 
  white_balls = 7 → black_balls = 3 → 
  (black_balls : ℚ) / (white_balls + black_balls) * 
  (white_balls : ℚ) / (white_balls + black_balls - 1) = 7 / 30 := by
  sorry

#check probability_black_then_white

end NUMINAMATH_CALUDE_probability_black_then_white_l1941_194111


namespace NUMINAMATH_CALUDE_parabolas_common_point_l1941_194165

/-- A parabola in the family y = -x^2 + px + q -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- The y-coordinate of a point on a parabola given its x-coordinate -/
def Parabola.y_coord (para : Parabola) (x : ℝ) : ℝ :=
  -x^2 + para.p * x + para.q

/-- The condition that the vertex of a parabola lies on y = x^2 -/
def vertex_on_curve (para : Parabola) : Prop :=
  ∃ a : ℝ, para.y_coord a = a^2

theorem parabolas_common_point :
  ∀ p : ℝ, ∃ para : Parabola, 
    vertex_on_curve para ∧ 
    para.p = p ∧
    para.y_coord 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabolas_common_point_l1941_194165


namespace NUMINAMATH_CALUDE_solution_set_M_range_of_k_l1941_194141

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 4| - |x - 3|

-- Theorem for the solution set M
theorem solution_set_M : 
  {x : ℝ | f x ≤ 2} = Set.Icc (-1) 3 := by sorry

-- Theorem for the range of k
theorem range_of_k : 
  {k : ℝ | ∃ x, k^2 - 4*k - 3*f x = 0} = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_M_range_of_k_l1941_194141


namespace NUMINAMATH_CALUDE_cube_edge_length_l1941_194115

theorem cube_edge_length (box_edge : ℝ) (num_cubes : ℕ) (cube_edge : ℝ) : 
  box_edge = 1 →
  num_cubes = 8 →
  num_cubes = (box_edge / cube_edge) ^ 3 →
  cube_edge * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1941_194115


namespace NUMINAMATH_CALUDE_cubic_root_sum_fourth_power_l1941_194186

theorem cubic_root_sum_fourth_power (p q r : ℝ) : 
  (p^3 - p^2 + 2*p - 3 = 0) → 
  (q^3 - q^2 + 2*q - 3 = 0) → 
  (r^3 - r^2 + 2*r - 3 = 0) → 
  p^4 + q^4 + r^4 = 13 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_fourth_power_l1941_194186


namespace NUMINAMATH_CALUDE_calories_per_slice_l1941_194139

/-- Given a pizza with 8 slices, where half the pizza contains 1200 calories,
    prove that each slice contains 300 calories. -/
theorem calories_per_slice (total_slices : ℕ) (eaten_fraction : ℚ) (total_calories : ℕ) :
  total_slices = 8 →
  eaten_fraction = 1/2 →
  total_calories = 1200 →
  (total_calories : ℚ) / (eaten_fraction * total_slices) = 300 := by
sorry

end NUMINAMATH_CALUDE_calories_per_slice_l1941_194139


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l1941_194104

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem f_derivative_at_zero : 
  (deriv f) 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l1941_194104


namespace NUMINAMATH_CALUDE_square_perimeter_l1941_194173

theorem square_perimeter (side_length : ℝ) (h : side_length = 13) : 
  4 * side_length = 52 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1941_194173


namespace NUMINAMATH_CALUDE_ice_skating_falls_l1941_194193

theorem ice_skating_falls (steven_falls sonya_falls : ℕ) 
  (h1 : steven_falls = 3)
  (h2 : sonya_falls = 6) : 
  (steven_falls + 13) / 2 - sonya_falls = 2 := by
  sorry

end NUMINAMATH_CALUDE_ice_skating_falls_l1941_194193


namespace NUMINAMATH_CALUDE_complex_simplification_l1941_194156

theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  7 * (4 - 2*i) + 4*i * (7 - 2*i) = 36 + 14*i := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l1941_194156


namespace NUMINAMATH_CALUDE_projection_coplanarity_l1941_194162

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a pyramid with a quadrilateral base -/
structure Pyramid where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  E : Point3D

/-- Checks if two line segments are perpendicular -/
def isPerpendicular (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def intersectionPoint (p1 p2 p3 p4 : Point3D) : Point3D := sorry

/-- Checks if a point is the height of a pyramid -/
def isHeight (p : Point3D) (pyr : Pyramid) : Prop := sorry

/-- Projects a point onto a plane defined by three points -/
def projectOntoPlane (p : Point3D) (p1 p2 p3 : Point3D) : Point3D := sorry

/-- Checks if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

theorem projection_coplanarity (pyr : Pyramid) : 
  let M := intersectionPoint pyr.A pyr.C pyr.B pyr.D
  isPerpendicular pyr.A pyr.C pyr.B pyr.D ∧ 
  isHeight (intersectionPoint pyr.E M pyr.A pyr.C) pyr →
  areCoplanar 
    (projectOntoPlane M pyr.E pyr.A pyr.B)
    (projectOntoPlane M pyr.E pyr.B pyr.C)
    (projectOntoPlane M pyr.E pyr.C pyr.D)
    (projectOntoPlane M pyr.E pyr.D pyr.A) := by
  sorry

end NUMINAMATH_CALUDE_projection_coplanarity_l1941_194162


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1941_194152

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) =
  (x^2 + 6*x + 1) * (x^2 + 6*x + 37) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1941_194152


namespace NUMINAMATH_CALUDE_pure_imaginary_solutions_l1941_194121

theorem pure_imaginary_solutions (x : ℂ) :
  (x^4 - 4*x^3 + 6*x^2 - 40*x - 64 = 0) ∧ (∃ k : ℝ, x = k * Complex.I) ↔
  x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_solutions_l1941_194121


namespace NUMINAMATH_CALUDE_unique_solution_system_l1941_194138

theorem unique_solution_system (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.cos (π * x)^2 + 2 * Real.sin (π * y) = 1)
  (h2 : Real.sin (π * x) + Real.sin (π * y) = 0)
  (h3 : x^2 - y^2 = 12) :
  x = 4 ∧ y = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1941_194138


namespace NUMINAMATH_CALUDE_existence_of_prime_and_cube_root_l1941_194153

theorem existence_of_prime_and_cube_root (n : ℕ+) :
  ∃ (p : ℕ) (m : ℤ), 
    Nat.Prime p ∧ 
    p % 6 = 5 ∧ 
    ¬(p ∣ n.val) ∧ 
    n.val % p = (m ^ 3) % p :=
by sorry

end NUMINAMATH_CALUDE_existence_of_prime_and_cube_root_l1941_194153


namespace NUMINAMATH_CALUDE_circles_intersection_theorem_l1941_194175

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Circle : Type :=
  (center : Point)
  (radius : ℝ)

-- Define the given conditions
def O₁ : Point := sorry
def O₂ : Point := sorry
def A : Point := sorry
def B : Point := sorry
def P : Point := sorry
def Q : Point := sorry

def circle₁ : Circle := ⟨O₁, sorry⟩
def circle₂ : Circle := ⟨O₂, sorry⟩

-- Define the necessary predicates
def intersect (c₁ c₂ : Circle) (p : Point) : Prop := sorry
def on_circle (c : Circle) (p : Point) : Prop := sorry
def on_segment (p₁ p₂ p : Point) : Prop := sorry

-- State the theorem
theorem circles_intersection_theorem :
  intersect circle₁ circle₂ A ∧
  intersect circle₁ circle₂ B ∧
  on_circle circle₁ Q ∧
  on_circle circle₂ P ∧
  (∃ (c : Circle), on_circle c O₁ ∧ on_circle c A ∧ on_circle c O₂ ∧ on_circle c P ∧ on_circle c Q) →
  on_segment O₁ Q B ∧ on_segment O₂ P B :=
sorry

end NUMINAMATH_CALUDE_circles_intersection_theorem_l1941_194175


namespace NUMINAMATH_CALUDE_carries_payment_is_correct_l1941_194171

/-- Calculates Carrie's payment for clothes given the quantities and prices of items, and her mom's contribution ratio --/
def carries_payment (shirt_qty : ℕ) (shirt_price : ℚ) 
                    (pants_qty : ℕ) (pants_price : ℚ)
                    (jacket_qty : ℕ) (jacket_price : ℚ)
                    (skirt_qty : ℕ) (skirt_price : ℚ)
                    (shoes_qty : ℕ) (shoes_price : ℚ)
                    (mom_ratio : ℚ) : ℚ :=
  let total_cost := shirt_qty * shirt_price + 
                    pants_qty * pants_price + 
                    jacket_qty * jacket_price + 
                    skirt_qty * skirt_price + 
                    shoes_qty * shoes_price
  total_cost - (mom_ratio * total_cost)

/-- Theorem: Carrie's payment for clothes is $228.67 --/
theorem carries_payment_is_correct : 
  carries_payment 8 12 4 25 4 75 3 30 2 50 (2/3) = 228.67 := by
  sorry

end NUMINAMATH_CALUDE_carries_payment_is_correct_l1941_194171


namespace NUMINAMATH_CALUDE_cube_three_times_cube_six_l1941_194124

theorem cube_three_times_cube_six : 3^3 * 6^3 = 5832 := by
  sorry

end NUMINAMATH_CALUDE_cube_three_times_cube_six_l1941_194124


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l1941_194159

/-- Given a wire cut into two pieces of lengths a and b, where a forms a square
    and b forms a circle, and the perimeter of the square equals the circumference
    of the circle, prove that a/b = 1. -/
theorem wire_cut_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (4 * (a / 4) = 2 * Real.pi * (b / (2 * Real.pi))) → a / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l1941_194159


namespace NUMINAMATH_CALUDE_no_valid_base_solution_l1941_194106

theorem no_valid_base_solution : 
  ¬∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ 
    (4 * x + 9 = 4 * y + 1) ∧ 
    (4 * x^2 + 7 * x + 7 = 3 * y^2 + 2 * y + 9) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_base_solution_l1941_194106


namespace NUMINAMATH_CALUDE_eliana_steps_l1941_194182

def day1_steps : ℕ := 200 + 300

def day2_steps (d1 : ℕ) : ℕ := d1 * d1

def day3_steps (d1 d2 : ℕ) : ℕ := d1 + d2 + 100

def total_steps (d1 d2 d3 : ℕ) : ℕ := d1 + d2 + d3

theorem eliana_steps :
  let d1 := day1_steps
  let d2 := day2_steps d1
  let d3 := day3_steps d1 d2
  total_steps d1 d2 d3 = 501100 := by
  sorry

end NUMINAMATH_CALUDE_eliana_steps_l1941_194182


namespace NUMINAMATH_CALUDE_at_least_five_primes_in_cubic_l1941_194151

theorem at_least_five_primes_in_cubic (f : ℕ → ℕ) : 
  (∀ n : ℕ, f n = n^3 - 10*n^2 + 31*n - 17) →
  ∃ (a b c d e : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧
    Nat.Prime (f a) ∧ Nat.Prime (f b) ∧ Nat.Prime (f c) ∧ Nat.Prime (f d) ∧ Nat.Prime (f e) :=
by sorry

end NUMINAMATH_CALUDE_at_least_five_primes_in_cubic_l1941_194151


namespace NUMINAMATH_CALUDE_max_daily_profit_daily_profit_correct_l1941_194117

/-- Represents the daily profit function for a store selling an item --/
def daily_profit (x : ℕ) : ℝ :=
  if x ≤ 30 then -x^2 + 54*x + 640
  else -40*x + 2560

/-- Theorem stating the maximum daily profit and the day it occurs --/
theorem max_daily_profit :
  ∃ (max_profit : ℝ) (max_day : ℕ),
    max_profit = 1369 ∧ 
    max_day = 27 ∧
    (∀ x : ℕ, 1 ≤ x ∧ x ≤ 60 → daily_profit x ≤ max_profit) ∧
    daily_profit max_day = max_profit :=
  sorry

/-- Cost price of the item --/
def cost_price : ℝ := 30

/-- Selling price function --/
def selling_price (x : ℕ) : ℝ :=
  if x ≤ 30 then 0.5 * x + 35
  else 50

/-- Quantity sold function --/
def quantity_sold (x : ℕ) : ℝ := 128 - 2 * x

/-- Verifies that the daily_profit function is correct --/
theorem daily_profit_correct (x : ℕ) (h : 1 ≤ x ∧ x ≤ 60) :
  daily_profit x = (selling_price x - cost_price) * quantity_sold x :=
  sorry

end NUMINAMATH_CALUDE_max_daily_profit_daily_profit_correct_l1941_194117


namespace NUMINAMATH_CALUDE_unique_solution_is_one_point_five_l1941_194157

/-- Given that (3a+2b)x^2+ax+b=0 is a linear equation in x with a unique solution, prove that x = 1.5 -/
theorem unique_solution_is_one_point_five (a b x : ℝ) :
  ((3*a + 2*b) * x^2 + a*x + b = 0) →  -- The equation
  (∃! x, (3*a + 2*b) * x^2 + a*x + b = 0) →  -- Unique solution exists
  (∀ y, (3*a + 2*b) * y^2 + a*y + b = 0 → y = x) →  -- Linear equation condition
  x = 1.5 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_is_one_point_five_l1941_194157


namespace NUMINAMATH_CALUDE_daughters_age_l1941_194127

theorem daughters_age (mother_age : ℕ) (daughter_age : ℕ) : 
  mother_age = 42 → 
  (mother_age + 9) = 3 * (daughter_age + 9) → 
  daughter_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_daughters_age_l1941_194127


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l1941_194120

/-- The number of diagonals in a polygon with n vertices -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- An octagon with one vertex removed is equivalent to a heptagon -/
def heptagon_vertices : ℕ := 8 - 1

theorem heptagon_diagonals : diagonals heptagon_vertices = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l1941_194120


namespace NUMINAMATH_CALUDE_triangle_area_from_lines_l1941_194168

/-- The area of the triangle formed by the intersection of three lines -/
theorem triangle_area_from_lines :
  let line1 : ℝ → ℝ := λ x => 3 * x - 4
  let line2 : ℝ → ℝ := λ x => -2 * x + 16
  let y_axis : ℝ → ℝ := λ x => 0
  let intersection_x : ℝ := 4
  let intersection_y : ℝ := line1 intersection_x
  let y_intercept1 : ℝ := line1 0
  let y_intercept2 : ℝ := line2 0
  let base : ℝ := y_intercept2 - y_intercept1
  let height : ℝ := intersection_x
  let area : ℝ := (1 / 2) * base * height
  area = 40 := by sorry

end NUMINAMATH_CALUDE_triangle_area_from_lines_l1941_194168


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1941_194187

theorem quadratic_equation_roots (p : ℝ) (x₁ x₂ : ℝ) : 
  p > 0 → 
  x₁^2 + p*x₁ + 1 = 0 → 
  x₂^2 + p*x₂ + 1 = 0 → 
  |x₁^2 - x₂^2| = p → 
  p = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1941_194187


namespace NUMINAMATH_CALUDE_homework_problem_solution_l1941_194118

theorem homework_problem_solution :
  ∃ (a b c d : ℤ),
    a ≤ -1 ∧ b ≤ -1 ∧ c ≤ -1 ∧ d ≤ -1 ∧
    -a - b = -a * b ∧
    c * d = -182 * (1 / (-c - d)) :=
by sorry

end NUMINAMATH_CALUDE_homework_problem_solution_l1941_194118


namespace NUMINAMATH_CALUDE_room_occupancy_l1941_194131

theorem room_occupancy (empty_chairs : ℕ) (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) :
  empty_chairs = 14 →
  empty_chairs * 2 = total_chairs →
  seated_people = total_chairs - empty_chairs →
  seated_people = (2 : ℚ) / 3 * total_people →
  total_people = 21 := by
sorry

end NUMINAMATH_CALUDE_room_occupancy_l1941_194131


namespace NUMINAMATH_CALUDE_cube_sum_eq_neg_26_l1941_194177

/-- ω is a nonreal complex number that is a cube root of unity -/
def ω : ℂ :=
  sorry

/-- ω is a nonreal cube root of unity -/
axiom ω_cube_root : ω ^ 3 = 1 ∧ ω ≠ 1

/-- The main theorem to prove -/
theorem cube_sum_eq_neg_26 :
  (1 + ω + 2 * ω^2)^3 + (1 - 2*ω + ω^2)^3 = -26 :=
sorry

end NUMINAMATH_CALUDE_cube_sum_eq_neg_26_l1941_194177


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l1941_194194

theorem gcd_digits_bound (a b : ℕ) : 
  10000 ≤ a ∧ a < 100000 ∧ 
  10000 ≤ b ∧ b < 100000 ∧ 
  100000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000 →
  Nat.gcd a b < 100 :=
by sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l1941_194194


namespace NUMINAMATH_CALUDE_pauls_pencil_stock_l1941_194197

/-- Calculates the number of pencils in stock at the end of the week -/
def pencils_in_stock_end_of_week (
  daily_production : ℕ)
  (working_days : ℕ)
  (initial_stock : ℕ)
  (sold_pencils : ℕ) : ℕ :=
  daily_production * working_days + initial_stock - sold_pencils

/-- Proves that Paul has 230 pencils in stock at the end of the week -/
theorem pauls_pencil_stock : 
  pencils_in_stock_end_of_week 100 5 80 350 = 230 := by
  sorry

end NUMINAMATH_CALUDE_pauls_pencil_stock_l1941_194197


namespace NUMINAMATH_CALUDE_problem_solution_l1941_194132

-- Definition of additive inverse
def additive_inverse (x y : ℝ) : Prop := x + y = 0

-- Definition of real roots for a quadratic equation
def has_real_roots (a b c : ℝ) : Prop := b^2 - 4*a*c ≥ 0

theorem problem_solution :
  -- Proposition 1
  (∀ x y : ℝ, additive_inverse x y → x + y = 0) ∧
  -- Proposition 3
  (∀ q : ℝ, ¬(has_real_roots 1 2 q) → q > 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1941_194132


namespace NUMINAMATH_CALUDE_paulas_remaining_money_l1941_194174

/-- Calculates the remaining money after shopping given the initial amount, 
    number of shirts, cost per shirt, and cost of pants. -/
def remaining_money (initial : ℕ) (num_shirts : ℕ) (shirt_cost : ℕ) (pants_cost : ℕ) : ℕ :=
  initial - (num_shirts * shirt_cost + pants_cost)

/-- Theorem stating that Paula's remaining money after shopping is $74 -/
theorem paulas_remaining_money :
  remaining_money 109 2 11 13 = 74 := by
  sorry

end NUMINAMATH_CALUDE_paulas_remaining_money_l1941_194174


namespace NUMINAMATH_CALUDE_product_of_smallest_primes_smallestOneDigitPrimes_are_prime_smallestTwoDigitPrime_is_prime_l1941_194169

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define the two smallest one-digit primes
def smallestOneDigitPrimes : Fin 2 → ℕ
| 0 => 2
| 1 => 3

-- Define the smallest two-digit prime
def smallestTwoDigitPrime : ℕ := 11

-- Theorem statement
theorem product_of_smallest_primes : 
  (smallestOneDigitPrimes 0) * (smallestOneDigitPrimes 1) * smallestTwoDigitPrime = 66 :=
by
  sorry

-- Prove that the defined numbers are indeed prime
theorem smallestOneDigitPrimes_are_prime :
  ∀ i : Fin 2, isPrime (smallestOneDigitPrimes i) :=
by
  sorry

theorem smallestTwoDigitPrime_is_prime :
  isPrime smallestTwoDigitPrime :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_smallest_primes_smallestOneDigitPrimes_are_prime_smallestTwoDigitPrime_is_prime_l1941_194169


namespace NUMINAMATH_CALUDE_abigail_savings_l1941_194164

/-- Calculates the monthly savings given the total savings and number of months. -/
def monthly_savings (total_savings : ℕ) (num_months : ℕ) : ℕ :=
  total_savings / num_months

/-- Theorem stating that given a total savings of 48000 over 12 months, 
    the monthly savings is 4000. -/
theorem abigail_savings : monthly_savings 48000 12 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_abigail_savings_l1941_194164


namespace NUMINAMATH_CALUDE_unique_solution_phi_sigma_pow_two_l1941_194150

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

/-- Sum of divisors function -/
def sigma : ℕ → ℕ := sorry

/-- The equation φ(σ(2^x)) = 2^x has only one solution in the natural numbers, and that solution is x = 1 -/
theorem unique_solution_phi_sigma_pow_two : 
  ∃! x : ℕ, phi (sigma (2^x)) = 2^x ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_phi_sigma_pow_two_l1941_194150


namespace NUMINAMATH_CALUDE_additional_cars_needed_l1941_194180

def current_cars : ℕ := 37
def cars_per_row : ℕ := 9

theorem additional_cars_needed :
  let next_multiple := ((current_cars + cars_per_row - 1) / cars_per_row) * cars_per_row
  next_multiple - current_cars = 8 := by
  sorry

end NUMINAMATH_CALUDE_additional_cars_needed_l1941_194180


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1941_194113

def rahul_future_age : ℕ := 26
def years_to_future : ℕ := 2
def deepak_age : ℕ := 18

theorem age_ratio_proof :
  let rahul_age := rahul_future_age - years_to_future
  (rahul_age : ℚ) / deepak_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l1941_194113


namespace NUMINAMATH_CALUDE_anands_income_is_2000_l1941_194134

/-- Represents the financial data of a person --/
structure FinancialData where
  income : ℕ
  expenditure : ℕ
  savings : ℕ

/-- Proves that Anand's income is 2000 given the conditions --/
theorem anands_income_is_2000 
  (anand balu : FinancialData)
  (income_ratio : anand.income * 4 = balu.income * 5)
  (expenditure_ratio : anand.expenditure * 2 = balu.expenditure * 3)
  (anand_savings : anand.income - anand.expenditure = 800)
  (balu_savings : balu.income - balu.expenditure = 800) :
  anand.income = 2000 := by
  sorry

#check anands_income_is_2000

end NUMINAMATH_CALUDE_anands_income_is_2000_l1941_194134


namespace NUMINAMATH_CALUDE_kolya_twos_count_l1941_194105

/-- Represents the grades of a student -/
structure Grades where
  fives : ℕ
  fours : ℕ
  threes : ℕ
  twos : ℕ

/-- Calculates the average grade -/
def averageGrade (g : Grades) : ℚ :=
  (5 * g.fives + 4 * g.fours + 3 * g.threes + 2 * g.twos) / 20

theorem kolya_twos_count 
  (kolya vasya : Grades)
  (total_grades : kolya.fives + kolya.fours + kolya.threes + kolya.twos = 20)
  (vasya_total : vasya.fives + vasya.fours + vasya.threes + vasya.twos = 20)
  (fives_eq : kolya.fives = vasya.fours)
  (fours_eq : kolya.fours = vasya.threes)
  (threes_eq : kolya.threes = vasya.twos)
  (twos_eq : kolya.twos = vasya.fives)
  (avg_eq : averageGrade kolya = averageGrade vasya) :
  kolya.twos = 5 := by
sorry

end NUMINAMATH_CALUDE_kolya_twos_count_l1941_194105


namespace NUMINAMATH_CALUDE_attendees_equal_22_l1941_194130

/-- Represents the total number of people who attended a performance given ticket prices and total revenue --/
def total_attendees (adult_price child_price : ℕ) (total_revenue : ℕ) (num_children : ℕ) : ℕ :=
  let num_adults := (total_revenue - num_children * child_price) / adult_price
  num_adults + num_children

/-- Theorem stating that given the specific conditions, the total number of attendees is 22 --/
theorem attendees_equal_22 :
  total_attendees 8 1 50 18 = 22 := by
  sorry

end NUMINAMATH_CALUDE_attendees_equal_22_l1941_194130


namespace NUMINAMATH_CALUDE_square_root_divided_by_19_l1941_194181

theorem square_root_divided_by_19 : 
  Real.sqrt 5776 / 19 = 4 := by sorry

end NUMINAMATH_CALUDE_square_root_divided_by_19_l1941_194181


namespace NUMINAMATH_CALUDE_even_q_l1941_194123

theorem even_q (p q : ℕ) 
  (h1 : ∃ (n : ℕ), n^2 = 2*p - q) 
  (h2 : ∃ (m : ℕ), m^2 = 2*p + q) : 
  Even q := by
sorry

end NUMINAMATH_CALUDE_even_q_l1941_194123


namespace NUMINAMATH_CALUDE_f_property_l1941_194155

/-- Represents a number with k digits, all being 1 -/
def rep_ones (k : ℕ) : ℕ :=
  (10^k - 1) / 9

/-- The function f(x) = 9x^2 + 2x -/
def f (x : ℕ) : ℕ :=
  9 * x^2 + 2 * x

/-- Theorem stating the property of f for numbers with all digits being 1 -/
theorem f_property (k : ℕ) :
  f (rep_ones k) = rep_ones (2 * k) :=
sorry

end NUMINAMATH_CALUDE_f_property_l1941_194155


namespace NUMINAMATH_CALUDE_notebook_cost_l1941_194146

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : 
  total_students = 36 →
  total_cost = 2772 →
  ∃ (buying_students : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat),
    buying_students > total_students / 2 ∧
    notebooks_per_student > 2 ∧
    cost_per_notebook = 2 * notebooks_per_student ∧
    buying_students * notebooks_per_student * cost_per_notebook = total_cost ∧
    cost_per_notebook = 12 :=
by sorry

end NUMINAMATH_CALUDE_notebook_cost_l1941_194146


namespace NUMINAMATH_CALUDE_round_robin_tournament_teams_l1941_194154

/-- The number of games in a round-robin tournament with n teams -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 28 games, there are 8 teams -/
theorem round_robin_tournament_teams : ∃ (n : ℕ), n > 0 ∧ num_games n = 28 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_teams_l1941_194154


namespace NUMINAMATH_CALUDE_exactly_six_expressions_l1941_194147

/-- Represents an expression using three identical digits --/
inductive ThreeDigitExpr (d : ℕ)
| add : ThreeDigitExpr d
| sub : ThreeDigitExpr d
| mul : ThreeDigitExpr d
| div : ThreeDigitExpr d
| exp : ThreeDigitExpr d
| sqrt : ThreeDigitExpr d
| floor : ThreeDigitExpr d
| fact : ThreeDigitExpr d

/-- Evaluates a ThreeDigitExpr to a real number --/
def eval {d : ℕ} : ThreeDigitExpr d → ℝ
| ThreeDigitExpr.add => sorry
| ThreeDigitExpr.sub => sorry
| ThreeDigitExpr.mul => sorry
| ThreeDigitExpr.div => sorry
| ThreeDigitExpr.exp => sorry
| ThreeDigitExpr.sqrt => sorry
| ThreeDigitExpr.floor => sorry
| ThreeDigitExpr.fact => sorry

/-- Predicate for valid expressions that evaluate to 24 --/
def isValid (d : ℕ) (e : ThreeDigitExpr d) : Prop :=
  d ≠ 8 ∧ eval e = 24

/-- The main theorem stating there are exactly 6 valid expressions --/
theorem exactly_six_expressions :
  ∃ (exprs : Finset (Σ (d : ℕ), ThreeDigitExpr d)),
    exprs.card = 6 ∧
    (∀ (d : ℕ) (e : ThreeDigitExpr d), isValid d e ↔ (⟨d, e⟩ : Σ (d : ℕ), ThreeDigitExpr d) ∈ exprs) :=
sorry

end NUMINAMATH_CALUDE_exactly_six_expressions_l1941_194147


namespace NUMINAMATH_CALUDE_labourer_savings_l1941_194136

/-- Calculates the amount saved by a labourer after clearing debt -/
theorem labourer_savings (monthly_income : ℕ) (initial_expenditure : ℕ) (reduced_expenditure : ℕ) : 
  monthly_income = 78 → 
  initial_expenditure = 85 → 
  reduced_expenditure = 60 → 
  (4 * monthly_income - (4 * reduced_expenditure + (6 * initial_expenditure - 6 * monthly_income))) = 30 := by
sorry

end NUMINAMATH_CALUDE_labourer_savings_l1941_194136


namespace NUMINAMATH_CALUDE_complex_expression_equals_one_l1941_194178

theorem complex_expression_equals_one : 
  Real.sqrt 6 / Real.sqrt 2 + |1 - Real.sqrt 3| - Real.sqrt 12 + (1/2)⁻¹ = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_one_l1941_194178


namespace NUMINAMATH_CALUDE_zach_cookies_l1941_194188

/-- The number of cookies Zach baked over three days --/
def total_cookies (monday tuesday wednesday : ℕ) : ℕ :=
  monday + tuesday + wednesday

/-- Theorem stating the total number of cookies Zach had after three days --/
theorem zach_cookies : ∃ (monday tuesday wednesday : ℕ),
  monday = 32 ∧
  tuesday = monday / 2 ∧
  wednesday = tuesday * 3 - 4 ∧
  total_cookies monday tuesday wednesday = 92 := by
  sorry

end NUMINAMATH_CALUDE_zach_cookies_l1941_194188


namespace NUMINAMATH_CALUDE_flag_arrangement_count_flag_arrangement_remainder_l1941_194103

def M (b g : ℕ) : ℕ :=
  (b - 1) * Nat.choose (b + 2) g - 2 * Nat.choose (b + 1) g

theorem flag_arrangement_count :
  M 14 11 = 54054 :=
by sorry

theorem flag_arrangement_remainder :
  M 14 11 % 1000 = 54 :=
by sorry

end NUMINAMATH_CALUDE_flag_arrangement_count_flag_arrangement_remainder_l1941_194103


namespace NUMINAMATH_CALUDE_octagonal_pyramid_cross_section_distance_l1941_194148

-- Define the pyramid and cross sections
structure OctagonalPyramid where
  crossSection1Area : ℝ
  crossSection2Area : ℝ
  planeDistance : ℝ

-- Define the theorem
theorem octagonal_pyramid_cross_section_distance
  (pyramid : OctagonalPyramid)
  (h1 : pyramid.crossSection1Area = 324 * Real.sqrt 2)
  (h2 : pyramid.crossSection2Area = 648 * Real.sqrt 2)
  (h3 : pyramid.planeDistance = 12)
  : ∃ (distance : ℝ), distance = 24 + 12 * Real.sqrt 2 ∧
    distance = (pyramid.planeDistance) / (1 - Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_octagonal_pyramid_cross_section_distance_l1941_194148


namespace NUMINAMATH_CALUDE_worker_c_completion_time_l1941_194185

/-- Given workers a, b, and c who can complete a work in the specified times,
    prove that c can complete the work alone in 40 days. -/
theorem worker_c_completion_time
  (total_work : ℝ)
  (time_a : ℝ) (time_b : ℝ) (time_c : ℝ)
  (total_time : ℝ) (c_left_early : ℝ)
  (h_time_a : time_a = 30)
  (h_time_b : time_b = 30)
  (h_total_time : total_time = 12)
  (h_c_left_early : c_left_early = 4)
  (h_work_completed : (total_work / time_a + total_work / time_b + total_work / time_c) *
    (total_time - c_left_early) +
    (total_work / time_a + total_work / time_b) * c_left_early = total_work) :
  time_c = 40 := by
sorry

end NUMINAMATH_CALUDE_worker_c_completion_time_l1941_194185


namespace NUMINAMATH_CALUDE_total_cds_count_l1941_194176

/-- The number of CDs Dawn has -/
def dawn_cds : ℕ := 10

/-- The number of CDs Kristine has -/
def kristine_cds : ℕ := dawn_cds + 7

/-- The number of CDs Mark has -/
def mark_cds : ℕ := 2 * kristine_cds

/-- The total number of CDs owned by Dawn, Kristine, and Mark -/
def total_cds : ℕ := dawn_cds + kristine_cds + mark_cds

theorem total_cds_count : total_cds = 61 := by
  sorry

end NUMINAMATH_CALUDE_total_cds_count_l1941_194176


namespace NUMINAMATH_CALUDE_find_divisor_l1941_194133

theorem find_divisor (dividend quotient remainder : ℕ) (h : dividend = quotient * 3 + remainder) :
  3 = dividend / quotient :=
by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1941_194133


namespace NUMINAMATH_CALUDE_m_range_l1941_194142

theorem m_range (m : ℝ) : 
  let M := Set.Iic m
  let P := {x : ℝ | x ≥ -1}
  M ∩ P = ∅ → m < -1 :=
by
  sorry

end NUMINAMATH_CALUDE_m_range_l1941_194142


namespace NUMINAMATH_CALUDE_final_balloon_count_l1941_194189

def total_balloons (brooke_initial : ℕ) (brooke_added : ℕ) (tracy_initial : ℕ) (tracy_added : ℕ) : ℕ :=
  (brooke_initial + brooke_added) + ((tracy_initial + tracy_added) / 2)

theorem final_balloon_count :
  total_balloons 12 8 6 24 = 35 := by
  sorry

end NUMINAMATH_CALUDE_final_balloon_count_l1941_194189


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1941_194167

theorem sufficient_but_not_necessary : 
  (∃ x : ℝ, x > 0 ∧ x ≤ 1) ∧ 
  (∀ x : ℝ, x > 1 → x > 0) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1941_194167


namespace NUMINAMATH_CALUDE_lucy_integers_l1941_194145

theorem lucy_integers (x y : ℤ) (h1 : 3 * x + 2 * y = 85) (h2 : x = 19 ∨ y = 19) : 
  (x = 19 ∧ y = 14) ∨ (y = 19 ∧ x = 14) :=
sorry

end NUMINAMATH_CALUDE_lucy_integers_l1941_194145


namespace NUMINAMATH_CALUDE_largest_b_value_l1941_194198

theorem largest_b_value (a b : ℤ) (h1 : 29 < a ∧ a < 41) (h2 : 39 < b) 
  (h3 : (a : ℚ) / b - (30 : ℚ) / b = 0.4) : b ≤ 75 :=
sorry

end NUMINAMATH_CALUDE_largest_b_value_l1941_194198


namespace NUMINAMATH_CALUDE_exactly_one_valid_number_l1941_194179

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- 3-digit whole number
  (n / 100 + (n / 10) % 10 + n % 10 = 28) ∧  -- digit-sum is 28
  (n % 10 < 7) ∧  -- units digit is less than 7
  (n % 2 = 0)  -- units digit is an even number

theorem exactly_one_valid_number : 
  ∃! n : ℕ, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_exactly_one_valid_number_l1941_194179


namespace NUMINAMATH_CALUDE_S_minimized_at_two_l1941_194158

/-- The area S(a) bounded by a line and a parabola -/
noncomputable def S (a : ℝ) : ℝ :=
  (1/6) * ((a^2 - 4*a + 8) ^ (3/2))

/-- The theorem stating that S(a) is minimized when a = 2 -/
theorem S_minimized_at_two :
  ∃ (a : ℝ), 0 ≤ a ∧ a ≤ 6 ∧ ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 6 → S a ≤ S x :=
by
  -- The proof goes here
  sorry

#check S_minimized_at_two

end NUMINAMATH_CALUDE_S_minimized_at_two_l1941_194158


namespace NUMINAMATH_CALUDE_distance_difference_l1941_194116

-- Define the distances
def john_distance : ℝ := 0.7
def nina_distance : ℝ := 0.4

-- Theorem statement
theorem distance_difference : john_distance - nina_distance = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l1941_194116


namespace NUMINAMATH_CALUDE_principal_amount_calculation_l1941_194125

/-- Given a principal amount and an interest rate, if increasing the rate by 1%
    results in an additional interest of 63 over 3 years, then the principal amount is 2100. -/
theorem principal_amount_calculation (P R : ℝ) (h : P * (R + 1) * 3 / 100 - P * R * 3 / 100 = 63) :
  P = 2100 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_calculation_l1941_194125


namespace NUMINAMATH_CALUDE_odd_function_zero_at_origin_l1941_194192

-- Define the function f on the interval [-1, 1]
def f : ℝ → ℝ := sorry

-- Define the property of being an odd function
def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, f (-x) = -f x

-- State the theorem
theorem odd_function_zero_at_origin (h : isOdd f) : f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_zero_at_origin_l1941_194192


namespace NUMINAMATH_CALUDE_system_solution_l1941_194184

theorem system_solution :
  ∀ x y : ℂ,
  (x^2 + y^2 = x*y ∧ x + y = x*y) ↔
  ((x = 0 ∧ y = 0) ∨
   (x = (3 + Complex.I * Real.sqrt 3) / 2 ∧ y = (3 - Complex.I * Real.sqrt 3) / 2) ∨
   (x = (3 - Complex.I * Real.sqrt 3) / 2 ∧ y = (3 + Complex.I * Real.sqrt 3) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1941_194184


namespace NUMINAMATH_CALUDE_vertex_on_x_axis_l1941_194110

/-- A parabola with equation y = x^2 - 12x + c -/
def parabola (c : ℝ) (x : ℝ) : ℝ := x^2 - 12*x + c

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 6

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y (c : ℝ) : ℝ := parabola c vertex_x

/-- The vertex lies on the x-axis if and only if c = 36 -/
theorem vertex_on_x_axis (c : ℝ) : vertex_y c = 0 ↔ c = 36 := by
  sorry

end NUMINAMATH_CALUDE_vertex_on_x_axis_l1941_194110


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1941_194135

theorem sufficient_not_necessary_condition (x y z : ℝ) :
  (∀ z ≠ 0, x * z^2024 < y * z^2024 → x < y) ∧
  ¬(∀ x y : ℝ, x < y → ∀ z : ℝ, x * z^2024 < y * z^2024) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1941_194135


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1941_194101

/-- Given an arithmetic sequence with common difference 2, if a₁, a₃, a₄ form a geometric sequence, then a₂ = -6 -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →  -- a₁, a₃, a₄ form a geometric sequence
  a 2 = -6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1941_194101
