import Mathlib

namespace solution_range_l302_30222

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, a * x < 6 ∧ (3 * x - 6 * a) / 2 > a / 3 - 1) → 
  a ≤ -3/2 := by
sorry

end solution_range_l302_30222


namespace square_9801_difference_of_squares_l302_30288

theorem square_9801_difference_of_squares (x : ℤ) (h : x^2 = 9801) :
  (x + 1) * (x - 1) = 9800 := by
sorry

end square_9801_difference_of_squares_l302_30288


namespace x_fourth_gt_x_minus_half_l302_30278

theorem x_fourth_gt_x_minus_half (x : ℝ) : x^4 - x + (1/2 : ℝ) > 0 := by
  sorry

end x_fourth_gt_x_minus_half_l302_30278


namespace point_quadrant_l302_30280

/-- Given that point A(a, -b) is in the first quadrant, prove that point B(a, b) is in the fourth quadrant -/
theorem point_quadrant (a b : ℝ) : 
  (a > 0 ∧ -b > 0) → (a > 0 ∧ b < 0) :=
by sorry

end point_quadrant_l302_30280


namespace factor_expression_l302_30235

theorem factor_expression (x y : ℝ) : 100 - 25 * x^2 + 16 * y^2 = (10 - 5*x + 4*y) * (10 + 5*x - 4*y) := by
  sorry

end factor_expression_l302_30235


namespace quadratic_equation_solution_l302_30241

theorem quadratic_equation_solution : 
  let x₁ := 2 + Real.sqrt 5
  let x₂ := 2 - Real.sqrt 5
  (x₁^2 - 4*x₁ - 1 = 0) ∧ (x₂^2 - 4*x₂ - 1 = 0) := by
sorry

end quadratic_equation_solution_l302_30241


namespace two_digit_integers_mod_seven_l302_30210

theorem two_digit_integers_mod_seven : 
  (Finset.filter (fun n => n ≥ 10 ∧ n < 100 ∧ n % 7 = 3) (Finset.range 100)).card = 13 := by
  sorry

end two_digit_integers_mod_seven_l302_30210


namespace taimour_painting_time_l302_30217

theorem taimour_painting_time (jamshid_rate taimour_rate : ℝ) 
  (h1 : jamshid_rate = 2 * taimour_rate) 
  (h2 : jamshid_rate + taimour_rate = 1 / 6) : 
  taimour_rate = 1 / 18 :=
by sorry

end taimour_painting_time_l302_30217


namespace interest_rate_calculation_l302_30237

theorem interest_rate_calculation (simple_interest principal time_period : ℚ) 
  (h1 : simple_interest = 4016.25)
  (h2 : time_period = 5)
  (h3 : principal = 80325) :
  simple_interest * 100 / (principal * time_period) = 0.01 := by
  sorry

end interest_rate_calculation_l302_30237


namespace walnut_trees_planted_l302_30282

theorem walnut_trees_planted (current : ℕ) (final : ℕ) (planted : ℕ) : 
  current = 4 → final = 10 → planted = final - current → planted = 6 := by sorry

end walnut_trees_planted_l302_30282


namespace prod_mod_seven_l302_30259

theorem prod_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end prod_mod_seven_l302_30259


namespace coral_age_conversion_l302_30269

/-- Converts an octal digit to decimal --/
def octal_to_decimal (digit : Nat) : Nat :=
  if digit < 8 then digit else 0

/-- Converts an octal number to decimal --/
def octal_to_decimal_number (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + octal_to_decimal digit * 8^i) 0

theorem coral_age_conversion :
  octal_to_decimal_number [7, 3, 4] = 476 := by
  sorry

end coral_age_conversion_l302_30269


namespace sequence_nth_term_l302_30244

theorem sequence_nth_term (u : ℕ → ℝ) (u₀ a b : ℝ) (h : ∀ n : ℕ, u (n + 1) = a * u n + b) :
  ∀ n : ℕ, u n = if a = 1
    then u₀ + n * b
    else a^n * u₀ + b * (1 - a^(n + 1)) / (1 - a) :=
by sorry

end sequence_nth_term_l302_30244


namespace impossibility_of_tiling_101_square_l302_30236

theorem impossibility_of_tiling_101_square : ¬ ∃ (a b : ℕ), 4*a + 9*b = 101*101 := by sorry

end impossibility_of_tiling_101_square_l302_30236


namespace S_remainder_mod_1000_l302_30248

/-- The sum of all three-digit positive integers from 500 to 999 with all digits distinct -/
def S : ℕ := sorry

/-- Theorem stating that the remainder of S divided by 1000 is 720 -/
theorem S_remainder_mod_1000 : S % 1000 = 720 := by sorry

end S_remainder_mod_1000_l302_30248


namespace mass_percentage_H_is_correct_l302_30279

/-- The mass percentage of H in a certain compound -/
def mass_percentage_H : ℝ := 1.69

/-- Theorem stating that the mass percentage of H is 1.69% -/
theorem mass_percentage_H_is_correct : mass_percentage_H = 1.69 := by
  sorry

end mass_percentage_H_is_correct_l302_30279


namespace train_speed_conversion_l302_30225

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- Speed of the train in meters per second -/
def train_speed_mps : ℝ := 52.5042

/-- Theorem stating the conversion of train speed from m/s to km/h -/
theorem train_speed_conversion :
  train_speed_mps * mps_to_kmph = 189.01512 := by sorry

end train_speed_conversion_l302_30225


namespace white_balls_count_l302_30218

theorem white_balls_count (total : ℕ) (prob : ℚ) (w : ℕ) : 
  total = 15 → 
  prob = 1 / 21 → 
  (w : ℚ) / total * ((w - 1) : ℚ) / (total - 1) = prob → 
  w = 5 := by sorry

end white_balls_count_l302_30218


namespace modulus_of_z_l302_30245

open Complex

theorem modulus_of_z (z : ℂ) (h : (1 - I) * z = 2 * I) : abs z = Real.sqrt 2 := by
  sorry

end modulus_of_z_l302_30245


namespace plane_equation_3d_l302_30293

/-- Definition of a line in 2D Cartesian coordinate system -/
def is_line_2d (A B C : ℝ) : Prop :=
  A^2 + B^2 ≠ 0

/-- Definition of a plane in 3D Cartesian coordinate system -/
def is_plane_3d (A B C D : ℝ) : Prop :=
  A^2 + B^2 + C^2 ≠ 0

/-- Theorem stating the equation of a plane in 3D Cartesian coordinate system -/
theorem plane_equation_3d (A B C D : ℝ) :
  is_plane_3d A B C D ↔ ∃ (x y z : ℝ), A*x + B*y + C*z + D = 0 :=
by sorry

end plane_equation_3d_l302_30293


namespace y_derivative_l302_30239

noncomputable def y (x : ℝ) : ℝ := 
  (1/12) * Real.log ((x^4 - x^2 + 1) / (x^2 + 1)^2) - 
  (1 / (2 * Real.sqrt 3)) * Real.arctan (Real.sqrt 3 / (2*x^2 - 1))

theorem y_derivative (x : ℝ) : 
  deriv y x = x^3 / ((x^4 - x^2 + 1) * (x^2 + 1)) :=
by sorry

end y_derivative_l302_30239


namespace bike_clamps_promotion_l302_30228

/-- The number of bike clamps given per bicycle purchase -/
def clamps_per_bike (morning_bikes : ℕ) (afternoon_bikes : ℕ) (total_clamps : ℕ) : ℚ :=
  total_clamps / (morning_bikes + afternoon_bikes)

/-- Theorem stating that the number of bike clamps given per bicycle purchase is 2 -/
theorem bike_clamps_promotion (morning_bikes afternoon_bikes total_clamps : ℕ)
  (h1 : morning_bikes = 19)
  (h2 : afternoon_bikes = 27)
  (h3 : total_clamps = 92) :
  clamps_per_bike morning_bikes afternoon_bikes total_clamps = 2 := by
  sorry

end bike_clamps_promotion_l302_30228


namespace supplement_of_complement_of_half_right_angle_l302_30211

/-- Given an angle that is half of 90 degrees, prove that the degree measure of
    the supplement of its complement is 135 degrees. -/
theorem supplement_of_complement_of_half_right_angle :
  let α : ℝ := 90 / 2
  let complement_α : ℝ := 90 - α
  let supplement_complement_α : ℝ := 180 - complement_α
  supplement_complement_α = 135 := by
  sorry

end supplement_of_complement_of_half_right_angle_l302_30211


namespace sample_size_major_C_l302_30255

/-- Represents the number of students in each major -/
structure CollegeMajors where
  A : Nat
  B : Nat
  C : Nat
  D : Nat

/-- Calculates the total number of students across all majors -/
def totalStudents (majors : CollegeMajors) : Nat :=
  majors.A + majors.B + majors.C + majors.D

/-- Calculates the number of students to be sampled from a specific major -/
def sampleSize (majors : CollegeMajors) (totalSample : Nat) (majorSize : Nat) : Nat :=
  (majorSize * totalSample) / totalStudents majors

/-- Theorem: The number of students to be sampled from major C is 16 -/
theorem sample_size_major_C :
  let majors : CollegeMajors := { A := 150, B := 150, C := 400, D := 300 }
  let totalSample : Nat := 40
  sampleSize majors totalSample majors.C = 16 := by
  sorry

end sample_size_major_C_l302_30255


namespace unique_base_solution_l302_30221

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldr (λ (i, d) acc => acc + d * b^i) 0

/-- The equation 142₂ + 163₂ = 315₂ holds in base b --/
def equation_holds (b : Nat) : Prop :=
  to_decimal [1, 4, 2] b + to_decimal [1, 6, 3] b = to_decimal [3, 1, 5] b

theorem unique_base_solution :
  ∃! b : Nat, b > 6 ∧ equation_holds b :=
sorry

end unique_base_solution_l302_30221


namespace highest_a_divisible_by_8_first_digit_is_three_l302_30230

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def last_three_digits (n : ℕ) : ℕ := n % 1000

theorem highest_a_divisible_by_8 :
  ∃ (a : ℕ), a ≤ 9 ∧
  is_divisible_by_8 (3 * 100000 + a * 1000 + 524) ∧
  (∀ (b : ℕ), b ≤ 9 → b > a →
    ¬is_divisible_by_8 (3 * 100000 + b * 1000 + 524)) ∧
  a = 8 :=
sorry

theorem first_digit_is_three :
  ∀ (a : ℕ), a ≤ 9 →
  (3 * 100000 + a * 1000 + 524) / 100000 = 3 :=
sorry

end highest_a_divisible_by_8_first_digit_is_three_l302_30230


namespace prob_sum_less_than_10_given_first_6_l302_30231

/-- The probability that the sum of two dice is less than 10, given that the first die shows 6 -/
theorem prob_sum_less_than_10_given_first_6 :
  let outcomes : Finset ℕ := Finset.range 6
  let favorable_outcomes : Finset ℕ := Finset.filter (λ x => x + 6 < 10) outcomes
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ) = 1 / 2 := by
  sorry

end prob_sum_less_than_10_given_first_6_l302_30231


namespace counterexample_exists_l302_30205

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ Nat.Prime (n + 2) := by
  sorry

end counterexample_exists_l302_30205


namespace min_abs_z_l302_30289

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 5*Complex.I) + Complex.abs (z - 7) = 10) :
  ∃ (w : ℂ), Complex.abs (z - 5*Complex.I) + Complex.abs (z - 7) = 10 → Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 35 / Real.sqrt 74 :=
by sorry

end min_abs_z_l302_30289


namespace dana_earnings_l302_30262

def hourly_rate : ℝ := 13
def friday_hours : ℝ := 9
def saturday_hours : ℝ := 10
def sunday_hours : ℝ := 3

theorem dana_earnings : 
  hourly_rate * (friday_hours + saturday_hours + sunday_hours) = 286 := by
  sorry

end dana_earnings_l302_30262


namespace rectangles_bounded_by_lines_l302_30273

/-- The number of rectangles bounded by p parallel lines and q perpendicular lines -/
def num_rectangles (p q : ℕ) : ℚ :=
  (p * q * (p - 1) * (q - 1)) / 4

/-- Theorem stating the number of rectangles bounded by p parallel lines and q perpendicular lines -/
theorem rectangles_bounded_by_lines (p q : ℕ) :
  num_rectangles p q = (p * q * (p - 1) * (q - 1)) / 4 := by
  sorry

end rectangles_bounded_by_lines_l302_30273


namespace bright_numbers_l302_30299

def isBright (x : ℕ) : Prop :=
  ∃ a b : ℕ, x = a^2 + b^3

theorem bright_numbers (r s : ℕ+) :
  (∃ f : ℕ → ℕ, StrictMono f ∧ ∀ i, isBright (r + f i) ∧ isBright (s + f i)) ∧
  (∃ g : ℕ → ℕ, StrictMono g ∧ ∀ i, isBright (r * g i) ∧ isBright (s * g i)) := by
  sorry

end bright_numbers_l302_30299


namespace number_line_segment_sum_l302_30290

theorem number_line_segment_sum : 
  ∀ (P V : ℝ) (Q R S T U : ℝ),
  P = 3 →
  V = 33 →
  Q - P = R - Q → R - Q = S - R → S - R = T - S → T - S = U - T → U - T = V - U →
  (S - P) + (V - T) = 25 := by
sorry

end number_line_segment_sum_l302_30290


namespace snakes_in_pond_l302_30298

/-- The number of alligators in the pond -/
def num_alligators : ℕ := 10

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := 56

/-- The number of eyes each alligator has -/
def eyes_per_alligator : ℕ := 2

/-- The number of eyes each snake has -/
def eyes_per_snake : ℕ := 2

/-- The number of snakes in the pond -/
def num_snakes : ℕ := (total_eyes - num_alligators * eyes_per_alligator) / eyes_per_snake

theorem snakes_in_pond : num_snakes = 18 := by
  sorry

end snakes_in_pond_l302_30298


namespace black_car_speed_proof_l302_30229

/-- The speed of the red car in miles per hour -/
def red_car_speed : ℝ := 10

/-- The initial distance between the cars in miles -/
def initial_distance : ℝ := 20

/-- The time it takes for the black car to overtake the red car in hours -/
def overtake_time : ℝ := 0.5

/-- The speed of the black car in miles per hour -/
def black_car_speed : ℝ := 50

theorem black_car_speed_proof :
  red_car_speed * overtake_time + initial_distance = black_car_speed * overtake_time :=
by sorry

end black_car_speed_proof_l302_30229


namespace distinct_triangles_in_square_pyramid_l302_30277

-- Define the number of vertices in a square pyramid
def num_vertices : ℕ := 5

-- Define the number of vertices needed to form a triangle
def vertices_per_triangle : ℕ := 3

-- Define the function to calculate combinations
def combinations (n k : ℕ) : ℕ := 
  (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem distinct_triangles_in_square_pyramid :
  combinations num_vertices vertices_per_triangle = 10 := by
  sorry

end distinct_triangles_in_square_pyramid_l302_30277


namespace ratio_equality_l302_30276

theorem ratio_equality (x : ℝ) : (1 : ℝ) / 3 = (5 : ℝ) / (3 * x) → x = 5 := by
  sorry

end ratio_equality_l302_30276


namespace x_power_3a_plus_2b_l302_30294

theorem x_power_3a_plus_2b (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3*a + 2*b) = 72 := by
  sorry

end x_power_3a_plus_2b_l302_30294


namespace range_of_function_l302_30291

theorem range_of_function (y : ℝ) : 
  (∃ x : ℝ, x ≠ 0 ∧ y = x + 4 / x) → y ≤ -4 ∨ y ≥ 4 := by
  sorry

end range_of_function_l302_30291


namespace group_meal_cost_example_l302_30213

def group_meal_cost (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

theorem group_meal_cost_example : group_meal_cost 9 2 2 = 14 := by
  sorry

end group_meal_cost_example_l302_30213


namespace exactly_three_valid_combinations_l302_30292

/-- Represents the number of pairs of socks at each price point -/
structure SockCombination :=
  (x : ℕ)  -- Number of 18 yuan socks
  (y : ℕ)  -- Number of 30 yuan socks
  (z : ℕ)  -- Number of 39 yuan socks

/-- Checks if a combination is valid according to the problem constraints -/
def isValidCombination (c : SockCombination) : Prop :=
  18 * c.x + 30 * c.y + 39 * c.z = 100 ∧
  18 * c.x + 30 * c.y + 39 * c.z > 95

/-- The main theorem stating that there are exactly 3 valid combinations -/
theorem exactly_three_valid_combinations :
  ∃! (s : Finset SockCombination), 
    (∀ c ∈ s, isValidCombination c) ∧ 
    s.card = 3 := by
  sorry

end exactly_three_valid_combinations_l302_30292


namespace prop_p_prop_q_l302_30200

-- Define the set of real numbers excluding 1
def RealExcludingOne : Set ℝ := {x : ℝ | x ∈ (Set.Ioo 0 1) ∪ (Set.Ioi 1)}

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Proposition p
theorem prop_p : ∀ a ∈ RealExcludingOne, log a 1 = 0 := by sorry

-- Proposition q
theorem prop_q : ∀ x : ℕ, x^3 ≥ x^2 := by sorry

end prop_p_prop_q_l302_30200


namespace min_value_expression_l302_30297

theorem min_value_expression (x : ℝ) (h : x > 10) :
  (x^2 + 36) / (x - 10) ≥ 4 * Real.sqrt 34 + 20 ∧
  (x^2 + 36) / (x - 10) = 4 * Real.sqrt 34 + 20 ↔ x = 10 + 2 * Real.sqrt 34 :=
by sorry

end min_value_expression_l302_30297


namespace correct_employee_count_l302_30267

/-- The number of employees in John's company --/
def number_of_employees : ℕ := 85

/-- The cost of each turkey in dollars --/
def cost_per_turkey : ℕ := 25

/-- The total amount spent on turkeys in dollars --/
def total_spent : ℕ := 2125

/-- Theorem stating that the number of employees is correct given the conditions --/
theorem correct_employee_count :
  number_of_employees * cost_per_turkey = total_spent :=
by sorry

end correct_employee_count_l302_30267


namespace candies_equalization_l302_30227

theorem candies_equalization (basket_a basket_b added : ℕ) : 
  basket_a = 8 → basket_b = 17 → basket_a + added = basket_b → added = 9 := by
sorry

end candies_equalization_l302_30227


namespace train_speed_l302_30247

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 450) (h2 : time = 12) :
  length / time = 37.5 := by
  sorry

end train_speed_l302_30247


namespace coin_division_problem_l302_30272

theorem coin_division_problem (n : ℕ) : 
  (n > 0) →
  (n % 8 = 5) → 
  (n % 7 = 2) → 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 5 ∨ m % 7 ≠ 2)) →
  (n % 9 = 1) :=
by sorry

end coin_division_problem_l302_30272


namespace fathers_age_l302_30208

theorem fathers_age (father daughter : ℕ) 
  (h1 : father = 4 * daughter)
  (h2 : father + daughter + 10 = 50) : 
  father = 32 := by
  sorry

end fathers_age_l302_30208


namespace cubic_equation_with_complex_root_l302_30220

theorem cubic_equation_with_complex_root (k : ℝ) : 
  (∃ (z : ℂ), z^3 + 2*(k-1)*z^2 + 9*z + 5*(k-1) = 0 ∧ Complex.abs z = Real.sqrt 5) →
  (k = -1 ∨ k = 3) :=
by sorry

end cubic_equation_with_complex_root_l302_30220


namespace exercise_weights_after_training_l302_30250

def calculate_final_weight (initial_weight : ℝ) (changes : List ℝ) : ℝ :=
  changes.foldl (λ acc change => acc * (1 + change)) initial_weight

def bench_press_changes : List ℝ := [-0.8, 0.6, -0.2, 2.0]
def squat_changes : List ℝ := [-0.5, 0.4, 1.0]
def deadlift_changes : List ℝ := [-0.3, 0.8, -0.4, 0.5]

theorem exercise_weights_after_training (initial_bench : ℝ) (initial_squat : ℝ) (initial_deadlift : ℝ) 
    (h1 : initial_bench = 500) 
    (h2 : initial_squat = 400) 
    (h3 : initial_deadlift = 600) :
  (calculate_final_weight initial_bench bench_press_changes = 384) ∧
  (calculate_final_weight initial_squat squat_changes = 560) ∧
  (calculate_final_weight initial_deadlift deadlift_changes = 680.4) := by
  sorry

#eval calculate_final_weight 500 bench_press_changes
#eval calculate_final_weight 400 squat_changes
#eval calculate_final_weight 600 deadlift_changes

end exercise_weights_after_training_l302_30250


namespace sqrt_expressions_equality_l302_30271

theorem sqrt_expressions_equality :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    (Real.sqrt (24 * a) - Real.sqrt (18 * b)) - Real.sqrt (6 * c) = 
    Real.sqrt (6 * c) - 3 * Real.sqrt (2 * b)) ∧
  (∀ d e f : ℝ, d > 0 → e > 0 → f > 0 →
    2 * Real.sqrt (12 * d) * Real.sqrt ((1 / 8) * e) + 5 * Real.sqrt (2 * f) = 
    Real.sqrt (6 * d) + 5 * Real.sqrt (2 * f)) :=
by sorry

end sqrt_expressions_equality_l302_30271


namespace parallel_transitive_perpendicular_to_parallel_l302_30263

/-- A type representing lines in three-dimensional space -/
structure Line3D where
  -- Add necessary fields here
  -- This is just a placeholder structure

/-- Parallel relation between two lines in 3D space -/
def parallel (l m : Line3D) : Prop :=
  sorry

/-- Perpendicular relation between two lines in 3D space -/
def perpendicular (l m : Line3D) : Prop :=
  sorry

/-- Theorem: If two lines are parallel to the same line, they are parallel to each other -/
theorem parallel_transitive (l m n : Line3D) :
  parallel l m → parallel m n → parallel l n :=
sorry

/-- Theorem: If a line is perpendicular to one of two parallel lines, it is perpendicular to the other -/
theorem perpendicular_to_parallel (l m n : Line3D) :
  perpendicular l m → parallel m n → perpendicular l n :=
sorry

end parallel_transitive_perpendicular_to_parallel_l302_30263


namespace largest_fraction_addition_l302_30252

def is_proper_fraction (n d : ℤ) : Prop := 0 < n ∧ n < d

def denominator_less_than_8 (d : ℤ) : Prop := 0 < d ∧ d < 8

theorem largest_fraction_addition :
  ∀ n d : ℤ,
    is_proper_fraction n d →
    denominator_less_than_8 d →
    is_proper_fraction (6 * n + d) (6 * d) →
    n * 7 ≤ 5 * d :=
by sorry

end largest_fraction_addition_l302_30252


namespace father_son_age_sum_l302_30253

theorem father_son_age_sum :
  ∀ (F S : ℕ),
  F > 0 ∧ S > 0 →
  F / S = 7 / 4 →
  (F + 10) / (S + 10) = 5 / 3 →
  F + S = 220 :=
by
  sorry

end father_son_age_sum_l302_30253


namespace reasoning_forms_mapping_l302_30281

/-- Represents the different forms of reasoning -/
inductive ReasoningForm
  | Inductive
  | Deductive
  | Analogical

/-- Represents the different reasoning descriptions -/
inductive ReasoningDescription
  | SpecificToSpecific
  | PartToWholeOrIndividualToGeneral
  | GeneralToSpecific

/-- Maps a reasoning description to its corresponding reasoning form -/
def descriptionToForm (d : ReasoningDescription) : ReasoningForm :=
  match d with
  | ReasoningDescription.SpecificToSpecific => ReasoningForm.Analogical
  | ReasoningDescription.PartToWholeOrIndividualToGeneral => ReasoningForm.Inductive
  | ReasoningDescription.GeneralToSpecific => ReasoningForm.Deductive

theorem reasoning_forms_mapping :
  (descriptionToForm ReasoningDescription.SpecificToSpecific = ReasoningForm.Analogical) ∧
  (descriptionToForm ReasoningDescription.PartToWholeOrIndividualToGeneral = ReasoningForm.Inductive) ∧
  (descriptionToForm ReasoningDescription.GeneralToSpecific = ReasoningForm.Deductive) :=
sorry

end reasoning_forms_mapping_l302_30281


namespace polynomial_value_l302_30261

theorem polynomial_value (x y : ℝ) (h : x - 2*y + 3 = 8) : x - 2*y = 5 := by
  sorry

end polynomial_value_l302_30261


namespace trigonometric_identity_l302_30207

theorem trigonometric_identity : 
  let a : Real := 2 * Real.pi / 3
  Real.sin (Real.pi - a / 2) + Real.tan (a - 5 * Real.pi / 12) = (2 + Real.sqrt 3) / 2 := by
  sorry

end trigonometric_identity_l302_30207


namespace cubic_inequality_l302_30284

theorem cubic_inequality (x : ℝ) (h : x ≥ 1000000) :
  x^3 + x + 1 ≤ x^4 / 1000000 := by
  sorry

end cubic_inequality_l302_30284


namespace logarithmic_equation_solution_l302_30249

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithmic_equation_solution :
  ∃! x : ℝ, x > 0 ∧ 
    log_base 4 (x - 1) + log_base (Real.sqrt 4) (x^2 - 1) + log_base (1/4) (x - 1) = 2 ∧
    x = Real.sqrt 5 := by
  sorry

end logarithmic_equation_solution_l302_30249


namespace russel_carousel_rides_l302_30209

/-- The number of times Russel rode the carousel -/
def carousel_rides (total_tickets jen_games shooting_cost carousel_cost : ℕ) : ℕ :=
  (total_tickets - jen_games * shooting_cost) / carousel_cost

/-- Proof that Russel rode the carousel 3 times -/
theorem russel_carousel_rides : 
  carousel_rides 19 2 5 3 = 3 := by
  sorry

end russel_carousel_rides_l302_30209


namespace bryan_bookshelves_l302_30256

/-- The number of books in each of Bryan's bookshelves -/
def books_per_shelf : ℕ := 27

/-- The total number of books Bryan has -/
def total_books : ℕ := 621

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := total_books / books_per_shelf

theorem bryan_bookshelves : num_bookshelves = 23 := by
  sorry

end bryan_bookshelves_l302_30256


namespace geometric_sequence_b_value_l302_30246

theorem geometric_sequence_b_value (b : ℝ) (h_positive : b > 0) 
  (h_sequence : ∃ r : ℝ, 250 * r = b ∧ b * r = 81 / 50) : 
  b = 9 * Real.sqrt 5 := by
sorry

end geometric_sequence_b_value_l302_30246


namespace right_triangle_bisector_properties_l302_30296

/-- Represents a right-angled triangle with an angle bisector --/
structure RightTriangleWithBisector where
  -- AC and BC are the legs, AB is the hypotenuse
  -- D is the point where the angle bisector from A intersects BC
  α : Real  -- angle BAC
  β : Real  -- angle ABC
  k : Real  -- ratio AD/DB

/-- Theorem about properties of a right-angled triangle with angle bisector --/
theorem right_triangle_bisector_properties (t : RightTriangleWithBisector) :
  -- 1. The problem has a solution for all k > 0
  (t.k > 0) →
  -- 2. The triangle is isosceles when k = √(2 + √2)
  (t.α = π/4 ↔ t.k = Real.sqrt (2 + Real.sqrt 2)) ∧
  -- 3. When k = 7/2, α = arccos(7/8) and β = π/2 - arccos(7/8)
  (t.k = 7/2 →
    t.α = Real.arccos (7/8) ∧
    t.β = π/2 - Real.arccos (7/8)) := by
  sorry

end right_triangle_bisector_properties_l302_30296


namespace defective_more_likely_from_machine2_l302_30295

-- Define the probabilities
def p_machine1 : ℝ := 0.8
def p_machine2 : ℝ := 0.2
def p_defect_machine1 : ℝ := 0.01
def p_defect_machine2 : ℝ := 0.05

-- Define the events
def B1 := "part manufactured by first machine"
def B2 := "part manufactured by second machine"
def A := "part is defective"

-- Define the probability of a part being defective
def p_defective : ℝ := p_machine1 * p_defect_machine1 + p_machine2 * p_defect_machine2

-- Theorem to prove
theorem defective_more_likely_from_machine2 :
  (p_machine2 * p_defect_machine2) / p_defective > (p_machine1 * p_defect_machine1) / p_defective :=
sorry

end defective_more_likely_from_machine2_l302_30295


namespace inscribed_square_area_l302_30219

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- The square is inscribed in the region bound by the parabola and x-axis -/
def is_inscribed_square (s : ℝ) : Prop :=
  ∃ (center : ℝ), 
    parabola (center - s) = 0 ∧
    parabola (center + s) = 0 ∧
    parabola (center + s) = 2*s

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area :
  ∃ (s : ℝ), is_inscribed_square s ∧ (2*s)^2 = 64 - 16*Real.sqrt 5 :=
sorry

end inscribed_square_area_l302_30219


namespace factorization_equality_l302_30266

theorem factorization_equality (a b : ℝ) : a * b^3 - 4 * a * b = a * b * (b + 2) * (b - 2) := by
  sorry

end factorization_equality_l302_30266


namespace solution_set_when_a_is_2_range_of_a_when_x_in_1_to_3_l302_30224

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a| - |2*x - 1|

-- Part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 + 3 ≥ 0} = {x : ℝ | -4 ≤ x ∧ x ≤ 2} := by sorry

-- Part 2
theorem range_of_a_when_x_in_1_to_3 :
  {a : ℝ | ∀ x ∈ Set.Icc 1 3, f x a ≤ 3} = Set.Icc (-3) 5 := by sorry

end solution_set_when_a_is_2_range_of_a_when_x_in_1_to_3_l302_30224


namespace bad_carrots_l302_30270

theorem bad_carrots (vanessa_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : 
  vanessa_carrots = 17 → mom_carrots = 14 → good_carrots = 24 → 
  vanessa_carrots + mom_carrots - good_carrots = 7 := by
sorry

end bad_carrots_l302_30270


namespace f_min_value_inequality_proof_l302_30258

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x + 1) + abs (x - 2)

-- Theorem for the minimum value of f
theorem f_min_value : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 3 := by sorry

-- Theorem for the inequality
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq : a + b + c = 3) :
  b^2 / a + c^2 / b + a^2 / c ≥ 3 := by sorry

end f_min_value_inequality_proof_l302_30258


namespace fancy_shape_charge_proof_l302_30203

/-- The cost to trim up a single boxwood -/
def trim_cost : ℚ := 5

/-- The total number of boxwoods -/
def total_boxwoods : ℕ := 30

/-- The number of boxwoods to be trimmed into fancy shapes -/
def fancy_boxwoods : ℕ := 4

/-- The total charge for the job -/
def total_charge : ℚ := 210

/-- The charge for trimming a boxwood into a fancy shape -/
def fancy_shape_charge : ℚ := 15

theorem fancy_shape_charge_proof :
  fancy_shape_charge * fancy_boxwoods + trim_cost * total_boxwoods = total_charge :=
sorry

end fancy_shape_charge_proof_l302_30203


namespace total_homework_time_l302_30238

def jacob_time : ℕ := 18

def greg_time (jacob_time : ℕ) : ℕ := jacob_time - 6

def patrick_time (greg_time : ℕ) : ℕ := 2 * greg_time - 4

def samantha_time (patrick_time : ℕ) : ℕ := (3 * patrick_time) / 2

theorem total_homework_time :
  jacob_time + greg_time jacob_time + patrick_time (greg_time jacob_time) + samantha_time (patrick_time (greg_time jacob_time)) = 80 := by
  sorry

end total_homework_time_l302_30238


namespace polar_bear_trout_consumption_l302_30264

/-- The amount of fish eaten daily by the polar bear -/
def total_fish : ℝ := 0.6

/-- The amount of salmon eaten daily by the polar bear -/
def salmon : ℝ := 0.4

/-- The amount of trout eaten daily by the polar bear -/
def trout : ℝ := total_fish - salmon

theorem polar_bear_trout_consumption : trout = 0.2 := by
  sorry

end polar_bear_trout_consumption_l302_30264


namespace umbrella_boots_probability_l302_30275

theorem umbrella_boots_probability
  (total_umbrellas : ℕ)
  (total_boots : ℕ)
  (prob_boots_and_umbrella : ℚ)
  (h1 : total_umbrellas = 40)
  (h2 : total_boots = 60)
  (h3 : prob_boots_and_umbrella = 1/3) :
  (prob_boots_and_umbrella * total_boots : ℚ) / total_umbrellas = 1/2 :=
sorry

end umbrella_boots_probability_l302_30275


namespace units_digit_of_2_pow_2012_l302_30265

theorem units_digit_of_2_pow_2012 : ∃ n : ℕ, 2^2012 ≡ 6 [ZMOD 10] := by
  sorry

end units_digit_of_2_pow_2012_l302_30265


namespace expand_and_simplify_l302_30287

theorem expand_and_simplify (x : ℝ) : 2 * (x + 3) * (x + 8) = 2 * x^2 + 22 * x + 48 := by
  sorry

end expand_and_simplify_l302_30287


namespace evaluate_g_l302_30201

/-- The function g(x) = 3x^2 - 5x + 7 -/
def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

/-- Theorem: 3g(2) + 2g(-4) = 177 -/
theorem evaluate_g : 3 * g 2 + 2 * g (-4) = 177 := by sorry

end evaluate_g_l302_30201


namespace cos_300_degrees_l302_30254

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end cos_300_degrees_l302_30254


namespace diamond_equation_solution_l302_30216

/-- Definition of the diamond operation -/
def diamond (a b : ℚ) : ℚ := a * b + 3 * b - a

/-- Theorem stating that if 4 ◇ y = 44, then y = 48/7 -/
theorem diamond_equation_solution :
  diamond 4 y = 44 → y = 48 / 7 := by
  sorry

end diamond_equation_solution_l302_30216


namespace mr_langsley_arrival_time_l302_30283

-- Define a custom time type
structure Time where
  hour : Nat
  minute : Nat

-- Define addition operation for Time
def Time.add (t1 t2 : Time) : Time :=
  let totalMinutes := t1.hour * 60 + t1.minute + t2.hour * 60 + t2.minute
  { hour := totalMinutes / 60, minute := totalMinutes % 60 }

-- Define the problem parameters
def pickup_time : Time := { hour := 6, minute := 0 }
def time_to_first_station : Time := { hour := 0, minute := 40 }
def time_from_first_station_to_work : Time := { hour := 2, minute := 20 }

-- Theorem to prove
theorem mr_langsley_arrival_time :
  (pickup_time.add time_to_first_station).add time_from_first_station_to_work = { hour := 9, minute := 0 } := by
  sorry


end mr_langsley_arrival_time_l302_30283


namespace jones_earnings_proof_l302_30214

/-- Dr. Jones' monthly earnings in dollars -/
def monthly_earnings : ℝ := 6000

/-- Dr. Jones' monthly expenses and savings -/
theorem jones_earnings_proof :
  monthly_earnings - (
    640 +  -- House rental
    380 +  -- Food expense
    (monthly_earnings / 4) +  -- Electric and water bill
    (monthly_earnings / 5)  -- Insurances
  ) = 2280  -- Remaining money after expenses
  := by sorry

end jones_earnings_proof_l302_30214


namespace max_profit_price_l302_30226

/-- Represents the profit function for a product -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 280 * x - 1600

/-- The initial purchase price of the product -/
def initial_purchase_price : ℝ := 8

/-- The initial selling price of the product -/
def initial_selling_price : ℝ := 10

/-- The initial daily sales volume -/
def initial_daily_sales : ℝ := 100

/-- The decrease in daily sales for each yuan increase in price -/
def sales_decrease_rate : ℝ := 10

/-- Theorem: The selling price that maximizes profit is 14 yuan -/
theorem max_profit_price : 
  ∃ (x : ℝ), x > initial_selling_price ∧ 
  ∀ (y : ℝ), y > initial_selling_price → profit_function x ≥ profit_function y :=
sorry

end max_profit_price_l302_30226


namespace soccer_team_biology_count_l302_30257

theorem soccer_team_biology_count :
  ∀ (total_players physics_count chemistry_count all_three_count physics_and_chemistry_count : ℕ),
    total_players = 15 →
    physics_count = 8 →
    chemistry_count = 6 →
    all_three_count = 3 →
    physics_and_chemistry_count = 4 →
    ∃ (biology_count : ℕ),
      biology_count = 9 ∧
      biology_count = total_players - (physics_count - physics_and_chemistry_count) - (chemistry_count - physics_and_chemistry_count) :=
by
  sorry

#check soccer_team_biology_count

end soccer_team_biology_count_l302_30257


namespace square_with_semicircles_perimeter_l302_30240

theorem square_with_semicircles_perimeter (π : Real) (h : π > 0) :
  let side_length := 2 / π
  let semicircle_radius := side_length / 2
  let semicircle_arc_length := π * semicircle_radius
  4 * semicircle_arc_length = 4 := by sorry

end square_with_semicircles_perimeter_l302_30240


namespace four_numbers_sum_l302_30285

theorem four_numbers_sum (a b c d : ℤ) :
  a + b + c = 21 ∧
  a + b + d = 28 ∧
  a + c + d = 29 ∧
  b + c + d = 30 →
  a = 6 ∧ b = 7 ∧ c = 8 ∧ d = 15 := by
sorry

end four_numbers_sum_l302_30285


namespace water_remaining_l302_30233

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → 
  used = 11/8 → 
  remaining = initial - used → 
  remaining = 13/8 :=
by
  sorry

#eval (13/8 : ℚ) -- To show that 13/8 is equivalent to 1 5/8

end water_remaining_l302_30233


namespace bacteria_growth_rate_l302_30232

/-- The growth rate of a bacteria colony -/
def growth_rate : ℝ := 2

/-- The number of days for a single colony to reach the habitat's limit -/
def single_colony_days : ℕ := 22

/-- The number of days for two colonies to reach the habitat's limit -/
def double_colony_days : ℕ := 21

/-- The theorem stating the growth rate of the bacteria colony -/
theorem bacteria_growth_rate :
  (growth_rate ^ single_colony_days : ℝ) = 2 * (growth_rate ^ double_colony_days : ℝ) :=
sorry

end bacteria_growth_rate_l302_30232


namespace smallest_side_length_is_correct_l302_30242

/-- Represents a triangle ABC with a point D on AC --/
structure TriangleABCD where
  -- The side length of the equilateral triangle
  side_length : ℕ
  -- The length of CD
  cd_length : ℕ
  -- Ensures that CD is not longer than AC
  h_cd_le_side : cd_length ≤ side_length

/-- The smallest possible side length of an equilateral triangle ABC 
    with a point D on AC such that BD is perpendicular to AC, 
    BD² = 65, and AC and CD are integers --/
def smallest_side_length : ℕ := 8

theorem smallest_side_length_is_correct (t : TriangleABCD) : 
  (t.side_length : ℝ)^2 / 4 + 65 = (t.side_length : ℝ)^2 →
  t.side_length ≥ smallest_side_length := by
  sorry

#check smallest_side_length_is_correct

end smallest_side_length_is_correct_l302_30242


namespace solve_for_z_l302_30274

-- Define the € operation
def euro (x y : ℝ) : ℝ := 2 * x * y

-- Theorem statement
theorem solve_for_z : ∃ z : ℝ, euro (euro 4 5) z = 560 ∧ z = 7 := by
  sorry

end solve_for_z_l302_30274


namespace tomato_price_theorem_l302_30243

/-- The original price per pound of tomatoes -/
def original_price : ℝ := 0.80

/-- The percentage of tomatoes that can be sold -/
def sellable_percentage : ℝ := 0.90

/-- The selling price per pound of tomatoes -/
def selling_price : ℝ := 0.96

/-- The profit percentage of the cost -/
def profit_percentage : ℝ := 0.08

/-- Theorem stating that the original price satisfies the given conditions -/
theorem tomato_price_theorem :
  selling_price * sellable_percentage = 
  original_price * (1 + profit_percentage) :=
by sorry

end tomato_price_theorem_l302_30243


namespace games_expenditure_l302_30268

def allowance : ℚ := 48

def clothes_fraction : ℚ := 1/4
def books_fraction : ℚ := 1/3
def snacks_fraction : ℚ := 1/6

def amount_on_games : ℚ := allowance - (clothes_fraction * allowance + books_fraction * allowance + snacks_fraction * allowance)

theorem games_expenditure : amount_on_games = 12 := by
  sorry

end games_expenditure_l302_30268


namespace point_upper_left_region_range_l302_30286

theorem point_upper_left_region_range (t : ℝ) : 
  (2 : ℝ) - 2 * t + 4 ≤ 0 → t ≥ 3 := by
  sorry

end point_upper_left_region_range_l302_30286


namespace union_complement_equality_l302_30223

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | x^2 ≤ 4}
def B : Set ℝ := {x | x < 1}

theorem union_complement_equality : A ∪ (U \ B) = {x : ℝ | x ≥ -2} := by sorry

end union_complement_equality_l302_30223


namespace intersection_count_l302_30212

-- Define the line L
def line_L (x y : ℝ) : Prop := y = 2 + Real.sqrt 3 - Real.sqrt 3 * x

-- Define the ellipse C'
def ellipse_C' (x y : ℝ) : Prop := 4 * x^2 + y^2 = 16

-- Define a point on the line L
def point_on_L : Prop := line_L 1 2

-- Theorem statement
theorem intersection_count :
  point_on_L →
  ∃ (p q : ℝ × ℝ),
    p ≠ q ∧
    line_L p.1 p.2 ∧
    line_L q.1 q.2 ∧
    ellipse_C' p.1 p.2 ∧
    ellipse_C' q.1 q.2 ∧
    ∀ (r : ℝ × ℝ), line_L r.1 r.2 ∧ ellipse_C' r.1 r.2 → r = p ∨ r = q :=
by
  sorry

end intersection_count_l302_30212


namespace largest_divisor_of_fifth_power_minus_self_l302_30260

/-- A number is composite if it has a proper divisor -/
def IsComposite (n : ℕ) : Prop := ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- The largest integer that always divides n^5 - n for all composite n -/
def LargestCommonDivisor : ℕ := 6

theorem largest_divisor_of_fifth_power_minus_self :
  ∀ n : ℕ, IsComposite n → (n^5 - n) % LargestCommonDivisor = 0 ∧
  ∀ k : ℕ, k > LargestCommonDivisor → ∃ m : ℕ, IsComposite m ∧ (m^5 - m) % k ≠ 0 := by
  sorry

end largest_divisor_of_fifth_power_minus_self_l302_30260


namespace consecutive_integers_square_sum_l302_30202

theorem consecutive_integers_square_sum : 
  ∀ (a b c : ℕ), 
    a > 0 → 
    b = a + 1 → 
    c = b + 1 → 
    a * b * c = 6 * (a + b + c) → 
    a^2 + b^2 + c^2 = 77 := by
  sorry

end consecutive_integers_square_sum_l302_30202


namespace p_sufficient_not_necessary_l302_30215

def p (x₁ x₂ : ℝ) : Prop := x₁^2 + 5*x₁ - 6 = 0 ∧ x₂^2 + 5*x₂ - 6 = 0

def q (x₁ x₂ : ℝ) : Prop := x₁ + x₂ = -5

theorem p_sufficient_not_necessary :
  (∀ x₁ x₂, p x₁ x₂ → q x₁ x₂) ∧ (∃ y₁ y₂, q y₁ y₂ ∧ ¬p y₁ y₂) := by sorry

end p_sufficient_not_necessary_l302_30215


namespace sum_powers_of_i_l302_30206

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_powers_of_i :
  i^300 + i^301 + i^302 + i^303 + i^304 + i^305 + i^306 + i^307 = 0 := by
  sorry

end sum_powers_of_i_l302_30206


namespace side_face_area_l302_30204

/-- A rectangular box with specific proportions and volume -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  front_half_top : length * width = 0.5 * (length * height)
  top_1_5_side : length * height = 1.5 * (width * height)
  volume : length * width * height = 3000

/-- The area of the side face of the box is 200 -/
theorem side_face_area (b : Box) : b.width * b.height = 200 := by
  sorry

end side_face_area_l302_30204


namespace total_installments_count_l302_30234

/-- Proves that the total number of installments is 52 given the specified payment conditions -/
theorem total_installments_count (first_25_payment : ℝ) (remaining_payment : ℝ) (average_payment : ℝ) :
  first_25_payment = 500 →
  remaining_payment = 600 →
  average_payment = 551.9230769230769 →
  ∃ n : ℕ, n = 52 ∧ 
    n * average_payment = 25 * first_25_payment + (n - 25) * remaining_payment :=
by sorry

end total_installments_count_l302_30234


namespace max_dot_product_on_circle_l302_30251

theorem max_dot_product_on_circle :
  ∀ (P : ℝ × ℝ),
  P.1^2 + P.2^2 = 1 →
  let A : ℝ × ℝ := (-2, 0)
  let O : ℝ × ℝ := (0, 0)
  let AO : ℝ × ℝ := (O.1 - A.1, O.2 - A.2)
  let AP : ℝ × ℝ := (P.1 - A.1, P.2 - A.2)
  (AO.1 * AP.1 + AO.2 * AP.2) ≤ 6 :=
by sorry

end max_dot_product_on_circle_l302_30251
