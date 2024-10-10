import Mathlib

namespace min_distance_complex_l3002_300237

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z - (1 + 2*I)) = 2) :
  ∃ (min_val : ℝ), (∀ (w : ℂ), Complex.abs (w - (1 + 2*I)) = 2 → Complex.abs (w + 1) ≥ min_val) ∧
                   (∃ (z₀ : ℂ), Complex.abs (z₀ - (1 + 2*I)) = 2 ∧ Complex.abs (z₀ + 1) = min_val) ∧
                   min_val = 2 * Real.sqrt 2 - 2 :=
sorry

end min_distance_complex_l3002_300237


namespace train_length_l3002_300263

/-- Calculates the length of a train given its speed, tunnel length, and time to pass through -/
theorem train_length (train_speed : ℝ) (tunnel_length : ℝ) (time_to_pass : ℝ) :
  train_speed = 72 →
  tunnel_length = 3.5 →
  time_to_pass = 3 / 60 →
  (train_speed * time_to_pass - tunnel_length) * 1000 = 100 := by
  sorry

end train_length_l3002_300263


namespace volleyball_team_math_players_l3002_300248

/-- The number of players taking mathematics in a volleyball team -/
def players_taking_mathematics (total_players : ℕ) (physics_players : ℕ) (both_subjects : ℕ) : ℕ :=
  total_players - (physics_players - both_subjects)

/-- Theorem stating the number of players taking mathematics -/
theorem volleyball_team_math_players :
  let total_players : ℕ := 30
  let physics_players : ℕ := 15
  let both_subjects : ℕ := 6
  players_taking_mathematics total_players physics_players both_subjects = 21 := by
  sorry

#check volleyball_team_math_players

end volleyball_team_math_players_l3002_300248


namespace binomial_coeff_not_coprime_l3002_300292

theorem binomial_coeff_not_coprime (n k l : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : 1 ≤ l) (h4 : l < n) :
  Nat.gcd (Nat.choose n k) (Nat.choose n l) > 1 := by
  sorry

end binomial_coeff_not_coprime_l3002_300292


namespace type_a_cubes_count_l3002_300269

/-- Represents the dimensions of the rectangular solid -/
def solid_dimensions : Fin 3 → ℕ
  | 0 => 120
  | 1 => 350
  | 2 => 400
  | _ => 0

/-- Calculates the number of cubes traversed by the diagonal -/
def total_cubes_traversed : ℕ := sorry

/-- The number of type A cubes traversed by the diagonal -/
def type_a_cubes : ℕ := total_cubes_traversed / 2

theorem type_a_cubes_count : type_a_cubes = 390 := by sorry

end type_a_cubes_count_l3002_300269


namespace sin_cos_difference_l3002_300293

theorem sin_cos_difference (x : Real) :
  (Real.sin x)^3 - (Real.cos x)^3 = -1 → Real.sin x - Real.cos x = -1 := by
sorry

end sin_cos_difference_l3002_300293


namespace john_ray_difference_l3002_300288

/-- The number of chickens each person took -/
structure ChickenCount where
  mary : ℕ
  john : ℕ
  ray : ℕ

/-- The conditions of the chicken problem -/
def chicken_problem (c : ChickenCount) : Prop :=
  c.john = c.mary + 5 ∧
  c.mary = c.ray + 6 ∧
  c.ray = 10

/-- The theorem stating John took 11 more chickens than Ray -/
theorem john_ray_difference (c : ChickenCount) 
  (h : chicken_problem c) : c.john - c.ray = 11 := by
  sorry


end john_ray_difference_l3002_300288


namespace product_of_x_values_l3002_300276

theorem product_of_x_values (x : ℝ) : 
  (|12 / x + 3| = 2) → (∃ y : ℝ, (|12 / y + 3| = 2) ∧ x * y = 144 / 5) :=
by sorry

end product_of_x_values_l3002_300276


namespace car_trip_duration_l3002_300255

/-- Represents the duration of a car trip with varying speeds -/
def car_trip (initial_speed initial_duration additional_speed average_speed : ℝ) : Prop :=
  ∃ (total_time additional_time : ℝ),
    total_time > 0 ∧
    additional_time ≥ 0 ∧
    total_time = initial_duration + additional_time ∧
    (initial_speed * initial_duration + additional_speed * additional_time) / total_time = average_speed

/-- The car trip lasts 12 hours given the specified conditions -/
theorem car_trip_duration :
  car_trip 45 4 75 65 → ∃ (total_time : ℝ), total_time = 12 :=
by sorry

end car_trip_duration_l3002_300255


namespace intersection_and_center_l3002_300262

-- Define the square ABCD
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (4, 4)
def D : ℝ × ℝ := (4, 0)

-- Define the lines
def line_from_A (x : ℝ) : ℝ := x
def line_from_B (x : ℝ) : ℝ := 4 - x

-- Define the intersection point
def intersection_point : ℝ × ℝ := (2, 2)

theorem intersection_and_center :
  (∀ x : ℝ, line_from_A x = line_from_B x → x = intersection_point.1) ∧
  (line_from_A intersection_point.1 = intersection_point.2) ∧
  (intersection_point.1 = (C.1 - A.1) / 2) ∧
  (intersection_point.2 = (C.2 - A.2) / 2) :=
sorry

end intersection_and_center_l3002_300262


namespace three_integers_difference_l3002_300273

theorem three_integers_difference (x y z : ℕ+) 
  (sum_xy : x + y = 998)
  (sum_xz : x + z = 1050)
  (sum_yz : y + z = 1234) :
  max x (max y z) - min x (min y z) = 236 := by
sorry

end three_integers_difference_l3002_300273


namespace snail_problem_l3002_300281

/-- The number of snails originally in Centerville -/
def original_snails : ℕ := 11760

/-- The number of snails removed from Centerville -/
def removed_snails : ℕ := 3482

/-- The number of snails remaining in Centerville -/
def remaining_snails : ℕ := original_snails - removed_snails

theorem snail_problem : remaining_snails = 8278 := by
  sorry

end snail_problem_l3002_300281


namespace max_ab_value_l3002_300243

theorem max_ab_value (a b : ℝ) : 
  (∀ x : ℤ, (20 * x + a > 0 ∧ 15 * x - b ≤ 0) ↔ (x = 2 ∨ x = 3 ∨ x = 4)) →
  (∃ (a' b' : ℝ), a' * b' = -1200 ∧ ∀ (a'' b'' : ℝ), 
    (∀ x : ℤ, (20 * x + a'' > 0 ∧ 15 * x - b'' ≤ 0) ↔ (x = 2 ∨ x = 3 ∨ x = 4)) →
    a'' * b'' ≤ a' * b') :=
by sorry

end max_ab_value_l3002_300243


namespace solution_inequality_minimum_value_l3002_300254

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - |x - 4|

-- Theorem for the solution of f(x) > 3
theorem solution_inequality (x : ℝ) : f x > 3 ↔ x > 2 := by sorry

-- Theorem for the minimum value of f(x)
theorem minimum_value : ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = 0 := by sorry

end solution_inequality_minimum_value_l3002_300254


namespace smallest_next_divisor_l3002_300258

theorem smallest_next_divisor (m : ℕ) : 
  m % 2 = 0 ∧ 
  1000 ≤ m ∧ m < 10000 ∧ 
  m % 391 = 0 → 
  (∃ (d : ℕ), d ∣ m ∧ d > 391 ∧ d ≤ 782 ∧ ∀ (x : ℕ), x ∣ m ∧ x > 391 → x ≥ d) ∧
  782 ∣ m :=
by sorry

end smallest_next_divisor_l3002_300258


namespace road_width_calculation_l3002_300209

/-- Represents the width of the roads in meters -/
def road_width : ℝ := 10

/-- The length of the lawn in meters -/
def lawn_length : ℝ := 80

/-- The breadth of the lawn in meters -/
def lawn_breadth : ℝ := 60

/-- The cost per square meter in Rupees -/
def cost_per_sq_m : ℝ := 5

/-- The total cost of traveling the two roads in Rupees -/
def total_cost : ℝ := 6500

theorem road_width_calculation :
  (lawn_length * road_width + lawn_breadth * road_width - road_width^2) * cost_per_sq_m = total_cost :=
sorry

end road_width_calculation_l3002_300209


namespace binomial_parameters_unique_l3002_300290

/-- A random variable following a binomial distribution -/
structure BinomialRandomVariable where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (ξ : BinomialRandomVariable) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRandomVariable) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_parameters_unique 
  (ξ : BinomialRandomVariable) 
  (h_exp : expectation ξ = 2.4)
  (h_var : variance ξ = 1.44) : 
  ξ.n = 6 ∧ ξ.p = 0.4 := by
  sorry

end binomial_parameters_unique_l3002_300290


namespace half_abs_diff_squares_21_19_l3002_300220

theorem half_abs_diff_squares_21_19 : (1/2 : ℝ) * |21^2 - 19^2| = 40 := by
  sorry

end half_abs_diff_squares_21_19_l3002_300220


namespace not_divisible_by_81_l3002_300282

theorem not_divisible_by_81 (n : ℤ) : ¬(81 ∣ (n^3 - 9*n + 27)) := by
  sorry

end not_divisible_by_81_l3002_300282


namespace sum_of_digits_of_power_l3002_300214

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_of_power : 
  tens_digit ((4 + 3) ^ 12) + ones_digit ((4 + 3) ^ 12) = 1 := by
  sorry

end sum_of_digits_of_power_l3002_300214


namespace existence_of_stabilization_l3002_300285

-- Define the function type
def PositiveIntegerFunction := ℕ+ → ℕ+

-- Define the conditions on the function
def SatisfiesConditions (f : PositiveIntegerFunction) : Prop :=
  (∀ m n : ℕ+, Nat.gcd (f m) (f n) ≤ (Nat.gcd m n) ^ 2014) ∧
  (∀ n : ℕ+, n ≤ f n ∧ f n ≤ n + 2014)

-- State the theorem
theorem existence_of_stabilization (f : PositiveIntegerFunction) 
  (h : SatisfiesConditions f) : 
  ∃ N : ℕ+, ∀ n : ℕ+, n ≥ N → f n = n := by
  sorry

end existence_of_stabilization_l3002_300285


namespace min_side_length_with_integer_altitude_l3002_300278

theorem min_side_length_with_integer_altitude (a b c h x y : ℕ) :
  -- Triangle with integer side lengths
  (a > 0) ∧ (b > 0) ∧ (c > 0) →
  -- Altitude h divides side b into segments x and y
  (x + y = b) →
  -- Difference between segments is 7
  (y = x + 7) →
  -- Pythagorean theorem for altitude
  (a^2 - y^2 = c^2 - x^2) →
  -- Altitude is an integer
  (h^2 = a^2 - y^2) →
  -- b is the minimum side length
  (∀ b' : ℕ, b' < b → ¬∃ a' c' h' x' y' : ℕ,
    (a' > 0) ∧ (b' > 0) ∧ (c' > 0) ∧
    (x' + y' = b') ∧ (y' = x' + 7) ∧
    (a'^2 - y'^2 = c'^2 - x'^2) ∧
    (h'^2 = a'^2 - y'^2)) →
  -- Conclusion: minimum side length is 25
  b = 25 := by sorry

end min_side_length_with_integer_altitude_l3002_300278


namespace fuel_savings_l3002_300240

theorem fuel_savings (old_efficiency : ℝ) (old_cost : ℝ) 
  (h_old_positive : old_efficiency > 0) (h_cost_positive : old_cost > 0) : 
  let new_efficiency := old_efficiency * (1 + 0.6)
  let new_cost := old_cost * 1.25
  let old_trip_cost := old_cost
  let new_trip_cost := (old_efficiency / new_efficiency) * new_cost
  let savings_percent := (old_trip_cost - new_trip_cost) / old_trip_cost * 100
  savings_percent = 21.875 := by
sorry


end fuel_savings_l3002_300240


namespace gcd_78_36_l3002_300270

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := by
  sorry

end gcd_78_36_l3002_300270


namespace greatest_common_piece_length_l3002_300235

theorem greatest_common_piece_length :
  let rope_lengths : List Nat := [45, 60, 75, 90]
  Nat.gcd (Nat.gcd (Nat.gcd 45 60) 75) 90 = 15 := by sorry

end greatest_common_piece_length_l3002_300235


namespace fraction_simplification_l3002_300268

theorem fraction_simplification (m : ℝ) (hm : m ≠ 0 ∧ m ≠ 1) : 
  (m - 1) / m / ((m - 1) / (m^2)) = m := by
  sorry

end fraction_simplification_l3002_300268


namespace abc_product_l3002_300228

theorem abc_product (a b c : ℕ+) 
  (h1 : a * b = 13)
  (h2 : b * c = 52)
  (h3 : c * a = 4) :
  a * b * c = 52 := by
sorry

end abc_product_l3002_300228


namespace sum_products_sides_projections_equality_l3002_300200

/-- Represents a convex polygon in a 2D plane -/
structure ConvexPolygon where
  -- Add necessary fields here
  -- This is a placeholder structure

/-- Calculates the sum of products of side lengths and projected widths -/
def sumProductsSidesProjections (P Q : ConvexPolygon) : ℝ :=
  -- Placeholder definition
  0

/-- Theorem stating the equality of sumProductsSidesProjections for two polygons -/
theorem sum_products_sides_projections_equality (P Q : ConvexPolygon) :
  sumProductsSidesProjections P Q = sumProductsSidesProjections Q P :=
by
  sorry

#check sum_products_sides_projections_equality

end sum_products_sides_projections_equality_l3002_300200


namespace arthur_walking_distance_l3002_300211

/-- Calculates the total distance walked given the number of blocks and block length -/
def total_distance (blocks_south : ℕ) (blocks_west : ℕ) (block_length : ℚ) : ℚ :=
  (blocks_south + blocks_west : ℚ) * block_length

/-- Theorem: Arthur's total walking distance is 4.5 miles -/
theorem arthur_walking_distance :
  let blocks_south : ℕ := 8
  let blocks_west : ℕ := 10
  let block_length : ℚ := 1/4
  total_distance blocks_south blocks_west block_length = 4.5 := by
  sorry

end arthur_walking_distance_l3002_300211


namespace cost_per_meat_type_l3002_300251

/-- Calculates the cost per type of sliced meat in a 4-pack with rush delivery --/
theorem cost_per_meat_type (base_cost : ℝ) (rush_delivery_rate : ℝ) (num_types : ℕ) :
  base_cost = 40 →
  rush_delivery_rate = 0.3 →
  num_types = 4 →
  (base_cost + base_cost * rush_delivery_rate) / num_types = 13 := by
  sorry

end cost_per_meat_type_l3002_300251


namespace total_teaching_years_l3002_300294

/-- The combined total of years taught by Virginia, Adrienne, and Dennis -/
def combinedYears (adrienne virginia dennis : ℕ) : ℕ := adrienne + virginia + dennis

theorem total_teaching_years :
  ∀ (adrienne virginia dennis : ℕ),
  virginia = adrienne + 9 →
  virginia = dennis - 9 →
  dennis = 43 →
  combinedYears adrienne virginia dennis = 102 :=
by
  sorry

end total_teaching_years_l3002_300294


namespace colored_ball_probability_l3002_300272

/-- The probability of drawing a colored ball from an urn -/
theorem colored_ball_probability (total : ℕ) (blue green white : ℕ)
  (h_total : total = blue + green + white)
  (h_blue : blue = 15)
  (h_green : green = 5)
  (h_white : white = 20) :
  (blue + green : ℚ) / total = 1 / 2 :=
by sorry

end colored_ball_probability_l3002_300272


namespace cos_sin_identity_l3002_300210

theorem cos_sin_identity : 
  Real.cos (14 * π / 180) * Real.cos (59 * π / 180) + 
  Real.sin (14 * π / 180) * Real.sin (121 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end cos_sin_identity_l3002_300210


namespace regression_lines_intersect_l3002_300250

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The point where a regression line passes through -/
def passingPoint (l : RegressionLine) (x : ℝ) : ℝ × ℝ :=
  (x, l.slope * x + l.intercept)

/-- Theorem: Two regression lines with the same average x and y values intersect at (a, b) -/
theorem regression_lines_intersect (l₁ l₂ : RegressionLine) (a b : ℝ) 
  (h₁ : passingPoint l₁ a = (a, b))
  (h₂ : passingPoint l₂ a = (a, b)) :
  ∃ (x y : ℝ), passingPoint l₁ x = (x, y) ∧ passingPoint l₂ x = (x, y) ∧ x = a ∧ y = b :=
sorry

end regression_lines_intersect_l3002_300250


namespace chess_tournament_games_l3002_300247

/-- Calculate the number of games in a round-robin tournament stage -/
def gamesInRoundRobin (n : ℕ) : ℕ := n * (n - 1)

/-- Calculate the number of games in a knockout tournament stage -/
def gamesInKnockout (n : ℕ) : ℕ := n - 1

/-- The total number of games in the chess tournament -/
def totalGames : ℕ :=
  gamesInRoundRobin 20 + gamesInRoundRobin 10 + gamesInKnockout 4

theorem chess_tournament_games :
  totalGames = 474 := by sorry

end chess_tournament_games_l3002_300247


namespace student_marks_calculation_l3002_300291

/-- Calculates the marks obtained by a student who failed an exam. -/
theorem student_marks_calculation
  (total_marks : ℕ)
  (passing_percentage : ℚ)
  (failing_margin : ℕ)
  (h_total : total_marks = 500)
  (h_passing : passing_percentage = 40 / 100)
  (h_failing : failing_margin = 50) :
  (total_marks : ℚ) * passing_percentage - failing_margin = 150 :=
by sorry

end student_marks_calculation_l3002_300291


namespace tangent_line_at_zero_l3002_300205

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x

theorem tangent_line_at_zero : 
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    y = m * x + b ∧ 
    (∃ (h : ℝ), h ≠ 0 ∧ (f (0 + h) - f 0) / h = m) ∧
    f 0 = b ∧
    m = 1 ∧ b = 0 :=
sorry

end tangent_line_at_zero_l3002_300205


namespace common_point_l3002_300231

/-- A function of the form f(x) = x^2 + ax + b where a + b = 2021 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + (2021 - a)

/-- Theorem: All functions f(x) = x^2 + ax + b where a + b = 2021 have a common point at (1, 2022) -/
theorem common_point : ∀ a : ℝ, f a 1 = 2022 := by
  sorry

end common_point_l3002_300231


namespace bowl_water_problem_l3002_300222

theorem bowl_water_problem (C : ℝ) (h1 : C > 0) : 
  C / 2 + 4 = 0.7 * C → 0.7 * C = 14 := by
  sorry

end bowl_water_problem_l3002_300222


namespace johns_height_l3002_300265

/-- Given the heights of John, Lena, and Rebeca, prove John's height is 152 cm -/
theorem johns_height (john lena rebeca : ℕ) 
  (h1 : john = lena + 15)
  (h2 : john + 6 = rebeca)
  (h3 : lena + rebeca = 295) :
  john = 152 := by sorry

end johns_height_l3002_300265


namespace greatest_prime_factor_of_sum_factorials_l3002_300286

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem greatest_prime_factor_of_sum_factorials :
  ∃ p : ℕ, is_prime p ∧ p ∣ (factorial 15 + factorial 17) ∧
    ∀ q : ℕ, is_prime q → q ∣ (factorial 15 + factorial 17) → q ≤ p :=
by sorry

end greatest_prime_factor_of_sum_factorials_l3002_300286


namespace smallest_square_area_l3002_300241

/-- Given three squares arranged as described in the problem, 
    this theorem relates the area of the smallest square to that of the middle square. -/
theorem smallest_square_area 
  (largest_square_area : ℝ) 
  (middle_square_area : ℝ) 
  (h1 : largest_square_area = 1) 
  (h2 : 0 < middle_square_area) 
  (h3 : middle_square_area < 1) :
  ∃ (smallest_square_area : ℝ), 
    smallest_square_area = ((1 - middle_square_area) / 2)^2 ∧ 
    0 < smallest_square_area ∧
    smallest_square_area < middle_square_area := by
  sorry

end smallest_square_area_l3002_300241


namespace heroes_on_large_sheets_l3002_300279

/-- Represents the number of pictures that can be drawn on a sheet of paper. -/
structure SheetCapacity where
  small : ℕ
  large : ℕ
  large_twice_small : large = 2 * small

/-- Represents the distribution of pictures drawn during the lunch break. -/
structure PictureDistribution where
  total : ℕ
  on_back : ℕ
  on_front : ℕ
  total_sum : total = on_back + on_front
  half_on_back : on_back = total / 2

/-- Represents the time spent drawing during the lunch break. -/
structure DrawingTime where
  break_duration : ℕ
  time_per_drawing : ℕ
  time_left : ℕ
  total_drawing_time : ℕ
  drawing_time_calc : total_drawing_time = break_duration - time_left

/-- The main theorem to prove. -/
theorem heroes_on_large_sheets
  (sheet_capacity : SheetCapacity)
  (picture_dist : PictureDistribution)
  (drawing_time : DrawingTime)
  (h1 : picture_dist.total = 20)
  (h2 : drawing_time.break_duration = 75)
  (h3 : drawing_time.time_per_drawing = 5)
  (h4 : drawing_time.time_left = 5)
  : ∃ (n : ℕ), n = 6 ∧ n * sheet_capacity.small = picture_dist.on_front / 2 :=
sorry

end heroes_on_large_sheets_l3002_300279


namespace circle_circumference_with_inscribed_rectangle_l3002_300252

theorem circle_circumference_with_inscribed_rectangle :
  let rectangle_width : ℝ := 9
  let rectangle_height : ℝ := 12
  let diagonal : ℝ := (rectangle_width^2 + rectangle_height^2).sqrt
  let diameter : ℝ := diagonal
  let circumference : ℝ := π * diameter
  circumference = 15 * π :=
by sorry

end circle_circumference_with_inscribed_rectangle_l3002_300252


namespace area_of_ABCD_l3002_300287

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The composed rectangle ABCD -/
def ABCD : Rectangle := { width := 10, height := 15 }

/-- One of the smaller identical rectangles -/
def SmallRect : Rectangle := { width := 5, height := 10 }

theorem area_of_ABCD : ABCD.area = 150 := by sorry

end area_of_ABCD_l3002_300287


namespace jill_peaches_l3002_300245

theorem jill_peaches (steven_peaches : ℕ) (jake_fewer : ℕ) (jake_more : ℕ)
  (h1 : steven_peaches = 14)
  (h2 : jake_fewer = 6)
  (h3 : jake_more = 3)
  : steven_peaches - jake_fewer - jake_more = 5 := by
  sorry

end jill_peaches_l3002_300245


namespace cooking_oil_problem_l3002_300232

theorem cooking_oil_problem (X : ℝ) : 
  (X - ((2/5) * X + 300)) - ((1/2) * (X - ((2/5) * X + 300)) - 200) = 800 →
  X = 2500 :=
by
  sorry

#check cooking_oil_problem

end cooking_oil_problem_l3002_300232


namespace max_tickets_purchasable_l3002_300289

theorem max_tickets_purchasable (ticket_price : ℕ) (budget : ℕ) : 
  ticket_price = 15 → budget = 150 → 
  (∀ n : ℕ, n * ticket_price ≤ budget → n ≤ 10) ∧ 
  10 * ticket_price ≤ budget :=
by sorry

end max_tickets_purchasable_l3002_300289


namespace boys_neither_happy_nor_sad_l3002_300213

/-- Given a group of children with various emotional states and genders, 
    prove the number of boys who are neither happy nor sad. -/
theorem boys_neither_happy_nor_sad 
  (total_children : ℕ) 
  (happy_children : ℕ) 
  (sad_children : ℕ) 
  (neither_children : ℕ) 
  (total_boys : ℕ) 
  (total_girls : ℕ) 
  (happy_boys : ℕ) 
  (sad_girls : ℕ) 
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neither_children = 20)
  (h5 : total_boys = 17)
  (h6 : total_girls = 43)
  (h7 : happy_boys = 6)
  (h8 : sad_girls = 4)
  (h9 : total_children = happy_children + sad_children + neither_children)
  (h10 : total_children = total_boys + total_girls) :
  total_boys - (happy_boys + (sad_children - sad_girls)) = 5 := by
  sorry

end boys_neither_happy_nor_sad_l3002_300213


namespace right_triangle_hypotenuse_l3002_300261

/-- A right-angled triangle with specific properties -/
structure RightTriangle where
  -- The lengths of the two legs
  a : ℝ
  b : ℝ
  -- The midpoints of the legs
  m : ℝ × ℝ
  n : ℝ × ℝ
  -- Conditions
  right_angle : a^2 + b^2 = (a + b)^2 / 2
  m_midpoint : m = (a/2, 0)
  n_midpoint : n = (0, b/2)
  xn_length : a^2 + (b/2)^2 = 22^2
  ym_length : b^2 + (a/2)^2 = 31^2

/-- The theorem to be proved -/
theorem right_triangle_hypotenuse (t : RightTriangle) : 
  Real.sqrt (t.a^2 + t.b^2) = 34 := by
  sorry

end right_triangle_hypotenuse_l3002_300261


namespace g_one_equals_three_l3002_300242

-- Define f as an odd function
def f : ℝ → ℝ := sorry

-- Define g as an even function
def g : ℝ → ℝ := sorry

-- Axiom for odd function
axiom f_odd : ∀ x : ℝ, f (-x) = -f x

-- Axiom for even function
axiom g_even : ∀ x : ℝ, g (-x) = g x

-- Given conditions
axiom condition1 : f (-1) + g 1 = 2
axiom condition2 : f 1 + g (-1) = 4

-- Theorem to prove
theorem g_one_equals_three : g 1 = 3 := by sorry

end g_one_equals_three_l3002_300242


namespace waiter_customers_l3002_300227

theorem waiter_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) 
  (h1 : num_tables = 7)
  (h2 : women_per_table = 7)
  (h3 : men_per_table = 2) :
  num_tables * (women_per_table + men_per_table) = 63 := by
  sorry

end waiter_customers_l3002_300227


namespace franks_initial_money_l3002_300202

/-- Frank's lamp purchase problem -/
theorem franks_initial_money (cheapest_lamp : ℕ) (expensive_multiplier : ℕ) (remaining_money : ℕ) : 
  cheapest_lamp = 20 →
  expensive_multiplier = 3 →
  remaining_money = 30 →
  cheapest_lamp * expensive_multiplier + remaining_money = 90 := by
  sorry

end franks_initial_money_l3002_300202


namespace turner_amusement_park_tickets_l3002_300219

/-- Calculates the total number of tickets needed for a multi-day amusement park visit -/
def total_tickets (days : ℕ) 
                  (rollercoaster_rides_per_day : ℕ) 
                  (catapult_rides_per_day : ℕ) 
                  (ferris_wheel_rides_per_day : ℕ) 
                  (rollercoaster_tickets_per_ride : ℕ) 
                  (catapult_tickets_per_ride : ℕ) 
                  (ferris_wheel_tickets_per_ride : ℕ) : ℕ :=
  days * (rollercoaster_rides_per_day * rollercoaster_tickets_per_ride +
          catapult_rides_per_day * catapult_tickets_per_ride +
          ferris_wheel_rides_per_day * ferris_wheel_tickets_per_ride)

theorem turner_amusement_park_tickets : 
  total_tickets 3 3 2 1 4 4 1 = 63 := by
  sorry

end turner_amusement_park_tickets_l3002_300219


namespace complex_modulus_l3002_300275

theorem complex_modulus (z : ℂ) : z + 2*I = (3 - I^3) / (1 + I) → Complex.abs z = Real.sqrt 13 := by
  sorry

end complex_modulus_l3002_300275


namespace not_all_squares_congruent_l3002_300234

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define congruence for squares
def congruent (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem statement
theorem not_all_squares_congruent :
  ¬ (∀ s1 s2 : Square, congruent s1 s2) :=
sorry

end not_all_squares_congruent_l3002_300234


namespace simplify_expression_l3002_300218

theorem simplify_expression (s r : ℝ) : 
  (2 * s^2 + 4 * r - 5) - (s^2 + 6 * r - 8) = s^2 - 2 * r + 3 := by
  sorry

end simplify_expression_l3002_300218


namespace weekly_earnings_increase_l3002_300216

/-- Calculates the percentage increase between two amounts -/
def percentageIncrease (originalAmount newAmount : ℚ) : ℚ :=
  ((newAmount - originalAmount) / originalAmount) * 100

theorem weekly_earnings_increase (originalAmount newAmount : ℚ) 
  (h1 : originalAmount = 40)
  (h2 : newAmount = 80) :
  percentageIncrease originalAmount newAmount = 100 := by
  sorry

#eval percentageIncrease 40 80

end weekly_earnings_increase_l3002_300216


namespace remainder_3045_div_32_l3002_300299

theorem remainder_3045_div_32 : 3045 % 32 = 5 := by sorry

end remainder_3045_div_32_l3002_300299


namespace system_condition_l3002_300224

theorem system_condition : 
  (∀ x y : ℝ, x > 2 ∧ y > 3 → x + y > 5 ∧ x * y > 6) ∧ 
  (∃ x y : ℝ, x + y > 5 ∧ x * y > 6 ∧ ¬(x > 2 ∧ y > 3)) := by
sorry

end system_condition_l3002_300224


namespace expression_one_evaluation_l3002_300259

theorem expression_one_evaluation : 8 / (-2) - (-4) * (-3) = -16 := by sorry

end expression_one_evaluation_l3002_300259


namespace B_equals_C_equals_A_union_complement_B_l3002_300207

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | x^2 ≥ 9}
def B : Set ℝ := {x | (x - 7) / (x + 1) ≤ 0}
def C : Set ℝ := {x | |x - 2| < 4}
def U : Set ℝ := Set.univ

-- Theorem statements
theorem B_equals : B = {x | -1 < x ∧ x ≤ 7} := by sorry

theorem C_equals : C = {x | -2 < x ∧ x < 6} := by sorry

theorem A_union_complement_B :
  A ∪ (U \ B) = {x | x ≥ 3 ∨ x ≤ -1} := by sorry

end B_equals_C_equals_A_union_complement_B_l3002_300207


namespace intersection_M_N_l3002_300271

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l3002_300271


namespace problem_statement_l3002_300206

/-- Given real numbers a and b satisfying the conditions, 
    prove the minimum value of m and the inequality for x, y, z -/
theorem problem_statement 
  (a b : ℝ) 
  (h1 : a * b > 0) 
  (h2 : a^2 * b = 2) 
  (m : ℝ := a * b + a^2) : 
  (∃ (t : ℝ), t = 3 ∧ ∀ m', m' = a * b + a^2 → m' ≥ t) ∧ 
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → |x + 2*y + 2*z| ≤ 3) := by
  sorry

end problem_statement_l3002_300206


namespace original_price_calculation_l3002_300223

theorem original_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 220)
  (h2 : profit_percentage = 0.1) : 
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 + profit_percentage) ∧ 
    original_price = 200 := by
  sorry

end original_price_calculation_l3002_300223


namespace equation_value_l3002_300277

theorem equation_value (x y : ℚ) 
  (eq1 : 5 * x + 6 * y = 7) 
  (eq2 : 3 * x + 5 * y = 6) : 
  x + 4 * y = 5 := by
sorry

end equation_value_l3002_300277


namespace rect_to_cylindrical_l3002_300266

/-- Conversion from rectangular to cylindrical coordinates --/
theorem rect_to_cylindrical :
  let x : ℝ := -4
  let y : ℝ := -4 * Real.sqrt 3
  let z : ℝ := -3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 4 * Real.pi / 3
  (r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) →
  (x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ z = z) :=
by sorry

end rect_to_cylindrical_l3002_300266


namespace cube_difference_152_l3002_300253

theorem cube_difference_152 : ∃! n : ℤ, 
  (∃ a : ℤ, a > 0 ∧ n - 76 = a^3) ∧ 
  (∃ b : ℤ, b > 0 ∧ n + 76 = b^3) :=
by
  sorry

end cube_difference_152_l3002_300253


namespace correct_average_l3002_300274

/-- Given 10 numbers with an initial average of 40.2, if one number is 17 greater than
    it should be and another number is 13 instead of 31, then the correct average is 40.3. -/
theorem correct_average (n : ℕ) (initial_avg : ℚ) (error1 error2 : ℚ) (correct2 : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  error1 = 17 →
  error2 = 13 →
  correct2 = 31 →
  (n : ℚ) * initial_avg - error1 - error2 + correct2 = n * 40.3 :=
by sorry

end correct_average_l3002_300274


namespace sound_speed_model_fits_data_sound_speed_model_unique_l3002_300256

/-- Represents the relationship between temperature and sound speed -/
def sound_speed_model (x : ℝ) : ℝ := 330 + 0.6 * x

/-- The set of data points for temperature and sound speed -/
def data_points : List (ℝ × ℝ) := [
  (-20, 318), (-10, 324), (0, 330), (10, 336), (20, 342), (30, 348)
]

/-- Theorem stating that the sound_speed_model fits the given data points -/
theorem sound_speed_model_fits_data : 
  ∀ (point : ℝ × ℝ), point ∈ data_points → 
    sound_speed_model point.1 = point.2 := by
  sorry

/-- Theorem stating that the sound_speed_model is the unique linear model fitting the data -/
theorem sound_speed_model_unique : 
  ∀ (a b : ℝ), (∀ (point : ℝ × ℝ), point ∈ data_points → 
    a + b * point.1 = point.2) → a = 330 ∧ b = 0.6 := by
  sorry

end sound_speed_model_fits_data_sound_speed_model_unique_l3002_300256


namespace problem_solution_l3002_300217

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 8) :
  b / (a + b) + c / (b + c) + a / (c + a) = 11.5 := by
  sorry

end problem_solution_l3002_300217


namespace function_inequality_solution_set_l3002_300280

open Set
open Function

theorem function_inequality_solution_set
  (f : ℝ → ℝ)
  (h_domain : ∀ x, x > 0 → DifferentiableAt ℝ f x)
  (h_ineq : ∀ x, x > 0 → x * deriv f x > f x)
  (h_f2 : f 2 = 0) :
  {x : ℝ | x > 0 ∧ f x < 0} = Ioo 0 2 := by
sorry

end function_inequality_solution_set_l3002_300280


namespace average_remaining_is_70_l3002_300236

/-- Represents the number of travelers checks of each denomination -/
structure TravelersChecks where
  fifty : ℕ
  hundred : ℕ

/-- The problem setup for travelers checks -/
def travelersChecksProblem (tc : TravelersChecks) : Prop :=
  tc.fifty + tc.hundred = 30 ∧
  50 * tc.fifty + 100 * tc.hundred = 1800

/-- The average amount of remaining checks after spending 15 $50 checks -/
def averageRemainingAmount (tc : TravelersChecks) : ℚ :=
  (50 * (tc.fifty - 15) + 100 * tc.hundred) / (tc.fifty + tc.hundred - 15)

/-- Theorem stating that the average amount of remaining checks is $70 -/
theorem average_remaining_is_70 (tc : TravelersChecks) :
  travelersChecksProblem tc → averageRemainingAmount tc = 70 := by
  sorry

end average_remaining_is_70_l3002_300236


namespace sheep_buying_problem_l3002_300225

/-- Represents the sheep buying problem from "The Nine Chapters on the Mathematical Art" --/
theorem sheep_buying_problem (x y : ℤ) : 
  (∀ (contribution shortage : ℤ), contribution = 5 ∧ shortage = 45 → contribution * x + shortage = y) ∧
  (∀ (contribution surplus : ℤ), contribution = 7 ∧ surplus = 3 → contribution * x - surplus = y) ↔
  (5 * x + 45 = y ∧ 7 * x - 3 = y) :=
sorry


end sheep_buying_problem_l3002_300225


namespace at_least_one_composite_l3002_300283

theorem at_least_one_composite (a b c : ℕ) 
  (h_odd_a : Odd a) (h_odd_b : Odd b) (h_odd_c : Odd c)
  (h_positive_a : 0 < a) (h_positive_b : 0 < b) (h_positive_c : 0 < c)
  (h_not_square : ¬∃k, a = k^2)
  (h_equation : a^2 + a + 1 = 3 * (b^2 + b + 1) * (c^2 + c + 1)) :
  (∃k > 1, k ∣ (b^2 + b + 1)) ∨ (∃k > 1, k ∣ (c^2 + c + 1)) :=
by sorry

end at_least_one_composite_l3002_300283


namespace house_construction_delay_l3002_300298

/-- Represents the construction of a house -/
structure HouseConstruction where
  totalDays : ℕ
  initialMen : ℕ
  additionalMen : ℕ
  daysBeforeAddition : ℕ

/-- Calculates the total man-days of work for the house construction -/
def totalManDays (h : HouseConstruction) : ℕ :=
  h.initialMen * h.totalDays

/-- Calculates the days behind schedule without additional men -/
def daysBehindSchedule (h : HouseConstruction) : ℕ :=
  let totalWork := h.initialMen * h.daysBeforeAddition + (h.initialMen + h.additionalMen) * (h.totalDays - h.daysBeforeAddition)
  totalWork / h.initialMen - h.totalDays

/-- Theorem stating that the construction would be 80 days behind schedule without additional men -/
theorem house_construction_delay (h : HouseConstruction) 
  (h_total_days : h.totalDays = 100)
  (h_initial_men : h.initialMen = 100)
  (h_additional_men : h.additionalMen = 100)
  (h_days_before_addition : h.daysBeforeAddition = 20) :
  daysBehindSchedule h = 80 := by
  sorry

#eval daysBehindSchedule { totalDays := 100, initialMen := 100, additionalMen := 100, daysBeforeAddition := 20 }

end house_construction_delay_l3002_300298


namespace largest_valid_number_l3002_300249

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ (a b : ℕ), a ≠ b ∧ a < 10 ∧ b < 10 ∧
    n = 7000 + 100 * a + 20 + b ∧
    n % 30 = 0

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 7920 :=
sorry

end largest_valid_number_l3002_300249


namespace johns_final_push_time_l3002_300260

/-- The time of John's final push in a race, given the initial and final distances between
    John and Steve, and their respective speeds. -/
theorem johns_final_push_time 
  (initial_distance : ℝ) 
  (john_speed : ℝ) 
  (steve_speed : ℝ) 
  (final_distance : ℝ) : 
  initial_distance = 16 →
  john_speed = 4.2 →
  steve_speed = 3.7 →
  final_distance = 2 →
  ∃ t : ℝ, t = 15 / 7 ∧ john_speed * t = initial_distance + final_distance :=
by
  sorry

#check johns_final_push_time

end johns_final_push_time_l3002_300260


namespace expression_evaluation_l3002_300295

theorem expression_evaluation :
  let x : ℝ := 4
  let y : ℝ := -1/2
  2 * x^2 * y - (5 * x * y^2 + 2 * (x^2 * y - 3 * x * y^2 + 1)) = -1 :=
by sorry

end expression_evaluation_l3002_300295


namespace hyperbola_eccentricity_l3002_300239

/-- The eccentricity of a hyperbola with equation y²/9 - x²/4 = 1 is √13/3 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = Real.sqrt 13 / 3 ∧
  ∀ x y : ℝ, y^2 / 9 - x^2 / 4 = 1 → 
  e = Real.sqrt ((3:ℝ)^2 + (2:ℝ)^2) / 3 := by
  sorry

end hyperbola_eccentricity_l3002_300239


namespace odd_digits_365_base5_l3002_300230

/-- Counts the number of odd digits in the base-5 representation of a natural number -/
def countOddDigitsBase5 (n : ℕ) : ℕ :=
  sorry

theorem odd_digits_365_base5 : countOddDigitsBase5 365 = 1 := by
  sorry

end odd_digits_365_base5_l3002_300230


namespace bin_drawing_probability_l3002_300204

def bin_probability (black white : ℕ) : ℚ :=
  let total := black + white
  let favorable := (black.choose 2 * white) + (black * white.choose 2)
  favorable / total.choose 3

theorem bin_drawing_probability :
  bin_probability 10 4 = 60 / 91 := by
  sorry

end bin_drawing_probability_l3002_300204


namespace inequality_solution_set_l3002_300226

theorem inequality_solution_set (t m : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + t < 0 ↔ 1 < x ∧ x < m) → 
  t = 2 ∧ m = 2 := by
sorry

end inequality_solution_set_l3002_300226


namespace geometric_sequence_property_l3002_300233

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Three terms form a geometric sequence -/
def FormGeometricSequence (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  FormGeometricSequence (a 3) (a 6) (a 9) :=
sorry

end geometric_sequence_property_l3002_300233


namespace fib_150_mod_5_l3002_300297

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The 150th Fibonacci number modulo 5 is 0 -/
theorem fib_150_mod_5 : fib 149 % 5 = 0 := by
  sorry

end fib_150_mod_5_l3002_300297


namespace sum_interior_angles_convex_polygon_l3002_300246

/-- The sum of interior angles of a convex polygon with n sides, in degrees -/
def sumInteriorAngles (n : ℕ) : ℝ :=
  180 * (n - 2)

/-- Theorem: The sum of interior angles of a convex n-gon is 180 * (n - 2) degrees -/
theorem sum_interior_angles_convex_polygon (n : ℕ) (h : n ≥ 3) :
  sumInteriorAngles n = 180 * (n - 2) := by
  sorry

#check sum_interior_angles_convex_polygon

end sum_interior_angles_convex_polygon_l3002_300246


namespace line_x_eq_1_properties_l3002_300201

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-axis in the 2D plane -/
def x_axis : Line := { a := 0, b := 1, c := 0 }

/-- Check if a line passes through a point -/
def Line.passes_through (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem -/
theorem line_x_eq_1_properties :
  ∃ (l : Line),
    (∀ (x y : ℝ), l.passes_through (x, y) ↔ x = 1) ∧
    l.passes_through (1, 2) ∧
    l.perpendicular x_axis := by
  sorry

end line_x_eq_1_properties_l3002_300201


namespace simplify_sqrt_difference_l3002_300208

theorem simplify_sqrt_difference : (Real.sqrt 300 / Real.sqrt 75) - (Real.sqrt 128 / Real.sqrt 32) = 0 := by
  sorry

end simplify_sqrt_difference_l3002_300208


namespace complex_number_simplification_l3002_300229

theorem complex_number_simplification (i : ℂ) (h : i^2 = -1) :
  (2 + i^3) / (1 - i) = (3 + i) / 2 := by sorry

end complex_number_simplification_l3002_300229


namespace cos_alpha_plus_pi_sixth_l3002_300203

theorem cos_alpha_plus_pi_sixth (α : Real) :
  (∃ (x y : Real), x = 1 ∧ y = 2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos (α + π/6) = (Real.sqrt 15 - 2 * Real.sqrt 5) / 10 := by
sorry

end cos_alpha_plus_pi_sixth_l3002_300203


namespace line_parallel_or_in_plane_l3002_300244

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  
/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here

/-- Parallel relation between two lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Main theorem -/
theorem line_parallel_or_in_plane (a b : Line3D) (α : Plane3D) 
  (h1 : parallel_lines a b) (h2 : parallel_line_plane a α) :
  parallel_line_plane b α ∨ line_in_plane b α :=
sorry

end line_parallel_or_in_plane_l3002_300244


namespace initial_bushes_count_l3002_300215

/-- The number of orchid bushes to be planted today -/
def bushes_to_plant : ℕ := 4

/-- The final number of orchid bushes after planting -/
def final_bushes : ℕ := 6

/-- The initial number of orchid bushes in the park -/
def initial_bushes : ℕ := final_bushes - bushes_to_plant

theorem initial_bushes_count : initial_bushes = 2 := by
  sorry

end initial_bushes_count_l3002_300215


namespace uncle_fyodor_wins_l3002_300257

/-- Represents the state of a sandwich (with or without sausage) -/
inductive SandwichState
  | WithSausage
  | WithoutSausage

/-- Represents a player in the game -/
inductive Player
  | UncleFyodor
  | Matroskin

/-- The game state -/
structure GameState where
  sandwiches : List SandwichState
  currentPlayer : Player
  fyodorMoves : Nat
  matroskinMoves : Nat

/-- A move in the game -/
inductive Move
  | EatSandwich : Move  -- For Uncle Fyodor
  | RemoveSausage : Nat → Move  -- For Matroskin, with sandwich index

/-- Function to apply a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Function to check if the game is over -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Function to determine the winner -/
def getWinner (state : GameState) : Option Player :=
  sorry

/-- Theorem stating that Uncle Fyodor can always win for N = 2^100 - 1 -/
theorem uncle_fyodor_wins :
  ∀ (initialState : GameState),
    initialState.sandwiches.length = 100 * (2^100 - 1) →
    initialState.currentPlayer = Player.UncleFyodor →
    initialState.fyodorMoves = 0 →
    initialState.matroskinMoves = 0 →
    ∀ (matroskinStrategy : GameState → Move),
      ∃ (fyodorStrategy : GameState → Move),
        let finalState := sorry  -- Play out the game using the strategies
        getWinner finalState = some Player.UncleFyodor :=
  sorry


end uncle_fyodor_wins_l3002_300257


namespace angle_AOC_equals_negative_150_l3002_300238

-- Define the rotation angles
def counterclockwise_rotation : ℝ := 120
def clockwise_rotation : ℝ := 270

-- Define the resulting angle
def angle_AOC : ℝ := counterclockwise_rotation - clockwise_rotation

-- Theorem statement
theorem angle_AOC_equals_negative_150 : angle_AOC = -150 := by
  sorry

end angle_AOC_equals_negative_150_l3002_300238


namespace wood_amount_correct_l3002_300296

/-- The amount of wood (in cubic meters) that two workers need to saw and chop in one day -/
def wood_amount : ℚ := 40 / 13

/-- The amount of wood (in cubic meters) that two workers can saw in one day -/
def saw_capacity : ℚ := 5

/-- The amount of wood (in cubic meters) that two workers can chop in one day -/
def chop_capacity : ℚ := 8

/-- Theorem stating that the wood_amount is the correct amount of wood that two workers 
    need to saw in order to have enough time to chop it for the remainder of the day -/
theorem wood_amount_correct : 
  wood_amount / saw_capacity + wood_amount / chop_capacity = 1 := by
  sorry


end wood_amount_correct_l3002_300296


namespace arithmetic_sequence_constant_l3002_300267

theorem arithmetic_sequence_constant (x y z k : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x ≠ y → y ≠ z → x ≠ z →
  k ≠ 1 →
  k * x = y →
  let u := y / x
  let v := z / y
  (u - 1/v) - (v - 1/u) = (v - 1/u) - (1/u - u) →
  ∃ (k' : ℝ), k' * x = z ∧ 2 * k / k' - 2 * k + k^2 / k' - 1 / k = 0 :=
by sorry

end arithmetic_sequence_constant_l3002_300267


namespace part_one_part_two_l3002_300212

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 3| + |x - a|

-- Part 1
theorem part_one : 
  {x : ℝ | f 4 x = 7} = Set.Icc (-3) 4 := by sorry

-- Part 2
theorem part_two (h : a > 0) :
  {x : ℝ | f a x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} → a = 1 := by sorry

end part_one_part_two_l3002_300212


namespace circle_equation_tangent_to_line_l3002_300264

/-- The equation of a circle with center (0, 1) tangent to the line y = 2 is x^2 + (y-1)^2 = 1 -/
theorem circle_equation_tangent_to_line (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ x^2 + (y - 1)^2 = r^2 ∧ |2 - 1| = r) → 
  x^2 + (y - 1)^2 = 1 := by
sorry

end circle_equation_tangent_to_line_l3002_300264


namespace company_employees_l3002_300221

theorem company_employees (total : ℕ) 
  (h1 : (60 : ℚ) / 100 * total = (total : ℚ) - (40 : ℚ) / 100 * total)
  (h2 : (20 : ℚ) / 100 * total = (40 : ℚ) / 100 * total / 2)
  (h3 : (20 : ℚ) / 100 * total = 20) :
  total = 100 := by
sorry

end company_employees_l3002_300221


namespace wendy_recycling_points_l3002_300284

/-- Calculates the points earned from recycling bags -/
def points_earned (total_bags : ℕ) (unrecycled_bags : ℕ) (points_per_bag : ℕ) : ℕ :=
  (total_bags - unrecycled_bags) * points_per_bag

/-- Proves that Wendy earns 210 points from recycling bags -/
theorem wendy_recycling_points :
  let total_bags : ℕ := 25
  let unrecycled_bags : ℕ := 4
  let points_per_bag : ℕ := 10
  points_earned total_bags unrecycled_bags points_per_bag = 210 :=
by
  sorry

end wendy_recycling_points_l3002_300284
