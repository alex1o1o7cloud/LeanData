import Mathlib

namespace rotated_square_base_vertex_on_line_l700_70060

/-- Represents a square with side length 2 inches -/
structure Square :=
  (side : ℝ)
  (is_two_inch : side = 2)

/-- Represents the configuration of three squares -/
structure SquareConfiguration :=
  (left : Square)
  (center : Square)
  (right : Square)
  (rotation_angle : ℝ)
  (is_thirty_degrees : rotation_angle = π / 6)

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The base vertex of the rotated square after lowering -/
def base_vertex (config : SquareConfiguration) : Point :=
  sorry

theorem rotated_square_base_vertex_on_line (config : SquareConfiguration) :
  (base_vertex config).y = 0 := by
  sorry

end rotated_square_base_vertex_on_line_l700_70060


namespace bagged_sugar_weight_recording_l700_70047

/-- Represents the recording of a bag's weight difference from the standard -/
def weightDifference (standardWeight actual : ℕ) : ℤ :=
  (actual : ℤ) - (standardWeight : ℤ)

/-- Proves that a bag weighing 498 grams should be recorded as -3 grams when the standard is 501 grams -/
theorem bagged_sugar_weight_recording :
  let standardWeight : ℕ := 501
  let actualWeight : ℕ := 498
  weightDifference standardWeight actualWeight = -3 := by
sorry

end bagged_sugar_weight_recording_l700_70047


namespace vasya_counts_more_apples_CD_l700_70062

/-- Represents the number of apple trees around the circular lake -/
def n : ℕ := sorry

/-- Represents the total number of apples on all trees -/
def m : ℕ := sorry

/-- Represents the number of trees Vasya counts from A to B -/
def vasya_trees_AB : ℕ := n / 3

/-- Represents the number of trees Petya counts from A to B -/
def petya_trees_AB : ℕ := 2 * n / 3

/-- Represents the number of apples Vasya counts from A to B -/
def vasya_apples_AB : ℕ := m / 8

/-- Represents the number of apples Petya counts from A to B -/
def petya_apples_AB : ℕ := 7 * m / 8

/-- Represents the number of trees Vasya counts from B to C -/
def vasya_trees_BC : ℕ := n / 3

/-- Represents the number of trees Petya counts from B to C -/
def petya_trees_BC : ℕ := 2 * n / 3

/-- Represents the number of apples Vasya counts from B to C -/
def vasya_apples_BC : ℕ := m / 8

/-- Represents the number of apples Petya counts from B to C -/
def petya_apples_BC : ℕ := 7 * m / 8

/-- Represents the number of trees Vasya counts from C to D -/
def vasya_trees_CD : ℕ := n / 3

/-- Represents the number of trees Petya counts from C to D -/
def petya_trees_CD : ℕ := 2 * n / 3

/-- Theorem stating that Vasya counts 3 times more apples than Petya from C to D -/
theorem vasya_counts_more_apples_CD :
  (m - vasya_apples_AB - vasya_apples_BC) = 3 * (m - petya_apples_AB - petya_apples_BC) :=
by sorry

end vasya_counts_more_apples_CD_l700_70062


namespace function_always_one_l700_70066

theorem function_always_one (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, n > 0 → f (n + f n) = f n)
  (h2 : ∃ n₀ : ℕ, f n₀ = 1) : 
  ∀ n : ℕ, f n = 1 := by
sorry

end function_always_one_l700_70066


namespace infinite_triplets_existence_l700_70067

theorem infinite_triplets_existence : ∀ n : ℕ, ∃ p : ℕ, ∃ q₁ q₂ : ℤ,
  0 < p ∧ p ≤ 2 * n^2 ∧ 
  |p * Real.sqrt 2 - q₁| * |p * Real.sqrt 3 - q₂| ≤ 1 / (4 * ↑n^2) :=
sorry

end infinite_triplets_existence_l700_70067


namespace rational_cube_sum_zero_l700_70048

theorem rational_cube_sum_zero (x y z : ℚ) 
  (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : 
  x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end rational_cube_sum_zero_l700_70048


namespace x_plus_y_value_l700_70098

theorem x_plus_y_value (x y : ℤ) (h1 : x - y = 36) (h2 : x = 20) : x + y = 4 := by
  sorry

end x_plus_y_value_l700_70098


namespace initial_bananas_per_child_l700_70078

theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) :
  total_children = 640 →
  absent_children = 320 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ),
    total_children * initial_bananas = (total_children - absent_children) * (initial_bananas + extra_bananas) ∧
    initial_bananas = 2 :=
by sorry

end initial_bananas_per_child_l700_70078


namespace geometric_sum_problem_l700_70002

/-- Sum of a geometric sequence -/
def geometric_sum (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sum_problem : geometric_sum (1/3) (1/2) 8 = 255/384 := by
  sorry

end geometric_sum_problem_l700_70002


namespace diophantine_equation_solutions_l700_70083

theorem diophantine_equation_solutions (n : ℕ+) :
  ∃ (S : Finset (ℤ × ℤ)), S.card ≥ n ∧ ∀ (p : ℤ × ℤ), p ∈ S → p.1^2 + 15 * p.2^2 = 4^(n : ℕ) :=
sorry

end diophantine_equation_solutions_l700_70083


namespace planes_perpendicular_l700_70018

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (m n : Line) (α β : Plane) :
  parallel m n → perpendicular n β → subset m α → perp_planes α β :=
sorry

end planes_perpendicular_l700_70018


namespace percent_of_a_is_4b_l700_70016

theorem percent_of_a_is_4b (a b : ℝ) (h : a = 1.2 * b) : (4 * b) / a = 10/3 := by
  sorry

end percent_of_a_is_4b_l700_70016


namespace ratio_x_to_2y_l700_70023

theorem ratio_x_to_2y (x y : ℝ) (h : (7 * x + 5 * y) / (x - 2 * y) = 26) : 
  x / (2 * y) = 3 / 2 := by
sorry

end ratio_x_to_2y_l700_70023


namespace circle_inscribed_angles_sum_l700_70003

theorem circle_inscribed_angles_sum (n : ℕ) (x y : ℝ) : 
  n = 16 → 
  x = 3 * (360 / n) / 2 → 
  y = 5 * (360 / n) / 2 → 
  x + y = 90 := by
sorry

end circle_inscribed_angles_sum_l700_70003


namespace circle_area_ratio_l700_70001

theorem circle_area_ratio (s r : ℝ) (hs : s > 0) (hr : r > 0) (h : r = 0.6 * s) : 
  (π * (r / 2)^2) / (π * (s / 2)^2) = 0.36 := by
  sorry

end circle_area_ratio_l700_70001


namespace fourth_power_of_nested_square_roots_l700_70014

theorem fourth_power_of_nested_square_roots : 
  (Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2)))^4 = 6 + 4 * Real.sqrt (2 + Real.sqrt 2) + Real.sqrt 2 := by
  sorry

end fourth_power_of_nested_square_roots_l700_70014


namespace number_of_triangles_triangles_in_figure_l700_70027

/-- The number of triangles in a figure with 9 lines and 25 intersection points -/
theorem number_of_triangles (num_lines : ℕ) (num_intersections : ℕ) : ℕ :=
  let total_combinations := (num_lines.choose 3)
  total_combinations - num_intersections

/-- Proof that the number of triangles in the given figure is 59 -/
theorem triangles_in_figure : number_of_triangles 9 25 = 59 := by
  sorry

end number_of_triangles_triangles_in_figure_l700_70027


namespace fraction_inequality_l700_70053

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  a / d < b / c := by
  sorry

end fraction_inequality_l700_70053


namespace serve_meals_eq_945_l700_70071

/-- The number of ways to serve meals to 10 people with exactly 2 correct matches -/
def serve_meals : ℕ :=
  let total_people : ℕ := 10
  let pasta_orders : ℕ := 5
  let salad_orders : ℕ := 5
  let correct_matches : ℕ := 2
  -- The actual calculation is not implemented, just the problem statement
  945

/-- Theorem stating that serve_meals equals 945 -/
theorem serve_meals_eq_945 : serve_meals = 945 := by
  sorry

end serve_meals_eq_945_l700_70071


namespace min_horizontal_distance_l700_70031

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

-- Define the set of x-coordinates for points P
def P : Set ℝ := {x | f x = 5}

-- Define the set of x-coordinates for points Q
def Q : Set ℝ := {x | f x = -2}

-- State the theorem
theorem min_horizontal_distance :
  ∃ (p q : ℝ), p ∈ P ∧ q ∈ Q ∧
  ∀ (p' q' : ℝ), p' ∈ P → q' ∈ Q →
  |p - q| ≤ |p' - q'| ∧
  |p - q| = |Real.sqrt 6 - Real.sqrt 3| :=
sorry

end min_horizontal_distance_l700_70031


namespace power_product_cube_l700_70063

theorem power_product_cube (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end power_product_cube_l700_70063


namespace line_perpendicular_to_triangle_sides_l700_70043

-- Define a triangle in a plane
structure Triangle :=
  (A B C : Point)

-- Define a line
structure Line :=
  (p q : Point)

-- Define perpendicularity between a line and a side of a triangle
def perpendicular (l : Line) (t : Triangle) (side : Fin 3) : Prop := sorry

theorem line_perpendicular_to_triangle_sides 
  (t : Triangle) (l : Line) :
  perpendicular l t 0 → perpendicular l t 1 → perpendicular l t 2 := by
  sorry

end line_perpendicular_to_triangle_sides_l700_70043


namespace dog_grouping_theorem_l700_70074

/-- The number of ways to divide 12 dogs into specified groups -/
def dog_grouping_count : ℕ := sorry

/-- Total number of dogs -/
def total_dogs : ℕ := 12

/-- Size of the first group (including Fluffy) -/
def group1_size : ℕ := 3

/-- Size of the second group (including Nipper) -/
def group2_size : ℕ := 5

/-- Size of the third group (including Spot) -/
def group3_size : ℕ := 4

/-- Theorem stating the correct number of ways to group the dogs -/
theorem dog_grouping_theorem : 
  dog_grouping_count = 20160 ∧
  total_dogs = group1_size + group2_size + group3_size ∧
  group1_size = 3 ∧
  group2_size = 5 ∧
  group3_size = 4 := by sorry

end dog_grouping_theorem_l700_70074


namespace factorial_divisibility_l700_70011

/-- The number of ones in the binary representation of a natural number -/
def binary_ones (n : ℕ) : ℕ := sorry

theorem factorial_divisibility (n : ℕ) (h : binary_ones n = 1995) :
  ∃ k : ℕ, n! = k * 2^(n - 1995) :=
sorry

end factorial_divisibility_l700_70011


namespace cylindrical_can_volume_condition_l700_70007

/-- The value of y that satisfies the volume condition for a cylindrical can --/
theorem cylindrical_can_volume_condition (π : ℝ) (h : π > 0) : 
  ∃! y : ℝ, y > 0 ∧ 
    π * (5 + y)^2 * (4 + y) = π * (5 + 2*y)^2 * 4 ∧
    y = Real.sqrt 76 - 5 := by
  sorry

end cylindrical_can_volume_condition_l700_70007


namespace log_problem_l700_70093

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_problem (x : ℝ) (h : log 8 (3 * x) = 3) :
  log x 125 = 3 / (9 * log 5 2 - log 5 3) := by
  sorry

end log_problem_l700_70093


namespace block_size_correct_l700_70082

/-- The number of squares on a standard chessboard -/
def standardChessboardSize : Nat := 64

/-- The number of squares removed from the chessboard -/
def removedSquares : Nat := 2

/-- The number of rectangular blocks that can be placed on the modified chessboard -/
def numberOfBlocks : Nat := 30

/-- The size of the rectangular block in squares -/
def blockSize : Nat := 2

/-- Theorem stating that the given block size is correct for the modified chessboard -/
theorem block_size_correct :
  blockSize * numberOfBlocks ≤ standardChessboardSize - removedSquares ∧
  (blockSize + 1) * numberOfBlocks > standardChessboardSize - removedSquares :=
sorry

end block_size_correct_l700_70082


namespace four_digit_divisible_by_eleven_l700_70057

theorem four_digit_divisible_by_eleven (B : ℕ) : 
  (4000 + 100 * B + 10 * B + 6) % 11 = 0 → B = 5 := by
  sorry

end four_digit_divisible_by_eleven_l700_70057


namespace tangency_triangle_area_l700_70070

/-- A right triangle with legs 3 and 4 -/
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  is_right : leg1 = 3 ∧ leg2 = 4

/-- The incircle of a triangle -/
structure Incircle (t : RightTriangle) where
  center : ℝ × ℝ
  radius : ℝ

/-- Points of tangency of the incircle with the sides of the triangle -/
structure TangencyPoints (t : RightTriangle) (i : Incircle t) where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- The area of a triangle -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: The area of the triangle formed by the points of tangency is 6/5 -/
theorem tangency_triangle_area (t : RightTriangle) (i : Incircle t) (tp : TangencyPoints t i) :
  triangleArea tp.point1 tp.point2 tp.point3 = 6/5 := by sorry

end tangency_triangle_area_l700_70070


namespace intersection_slope_inequality_l700_70030

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := x * (1 + Real.log x)

-- Define the derivative of f
def f' (x : ℝ) : ℝ := Real.log x + 2

-- Theorem statement
theorem intersection_slope_inequality (x₁ x₂ k : ℝ) (h1 : 0 < x₁) (h2 : x₁ < x₂) 
  (h3 : k = (f' x₂ - f' x₁) / (x₂ - x₁)) : 
  x₁ < 1 / k ∧ 1 / k < x₂ := by
  sorry

end

end intersection_slope_inequality_l700_70030


namespace tank_capacity_after_adding_gas_l700_70033

/-- Given a tank with a capacity of 54 gallons, initially filled to 3/4 of its capacity,
    prove that after adding 9 gallons of gasoline, the tank will be filled to 23/25 of its capacity. -/
theorem tank_capacity_after_adding_gas (tank_capacity : ℚ) (initial_fill : ℚ) (added_gas : ℚ) :
  tank_capacity = 54 →
  initial_fill = 3 / 4 →
  added_gas = 9 →
  (initial_fill * tank_capacity + added_gas) / tank_capacity = 23 / 25 := by
  sorry

end tank_capacity_after_adding_gas_l700_70033


namespace margaret_fraction_of_dollar_l700_70084

-- Define the amounts for each person
def lance_cents : ℕ := 70
def guy_cents : ℕ := 50 + 10  -- Two quarters and a dime
def bill_cents : ℕ := 6 * 10  -- Six dimes
def total_cents : ℕ := 265

-- Define Margaret's amount
def margaret_cents : ℕ := total_cents - (lance_cents + guy_cents + bill_cents)

-- Theorem to prove
theorem margaret_fraction_of_dollar : 
  (margaret_cents : ℚ) / 100 = 3 / 4 := by sorry

end margaret_fraction_of_dollar_l700_70084


namespace pencils_in_drawer_l700_70019

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 34

/-- The number of pencils Dan took from the drawer -/
def pencils_taken : ℕ := 22

/-- The number of pencils remaining in the drawer -/
def remaining_pencils : ℕ := initial_pencils - pencils_taken

theorem pencils_in_drawer : remaining_pencils = 12 := by
  sorry

end pencils_in_drawer_l700_70019


namespace determinant_equality_l700_70009

theorem determinant_equality (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = -3 →
  Matrix.det !![p + 2*r, q + 2*s; r, s] = -3 := by
  sorry

end determinant_equality_l700_70009


namespace sum_is_zero_l700_70086

theorem sum_is_zero (x y z : ℝ) 
  (h1 : x^2 = y + 2) 
  (h2 : y^2 = z + 2) 
  (h3 : z^2 = x + 2) : 
  x + y + z = 0 := by
sorry

end sum_is_zero_l700_70086


namespace angle_of_inclination_negative_sqrt_three_line_l700_70046

theorem angle_of_inclination_negative_sqrt_three_line :
  let line : ℝ → ℝ := λ x ↦ -Real.sqrt 3 * x + 1
  let slope : ℝ := -Real.sqrt 3
  let angle_of_inclination : ℝ := Real.arctan (-Real.sqrt 3)
  (0 ≤ angle_of_inclination) ∧ (angle_of_inclination < π) →
  angle_of_inclination = 2 * π / 3 :=
by
  sorry


end angle_of_inclination_negative_sqrt_three_line_l700_70046


namespace contrapositive_square_sum_l700_70076

theorem contrapositive_square_sum (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
sorry

end contrapositive_square_sum_l700_70076


namespace muffin_goal_remaining_l700_70005

def muffin_problem (goal : ℕ) (morning_sales : ℕ) (afternoon_sales : ℕ) : ℕ :=
  goal - (morning_sales + afternoon_sales)

theorem muffin_goal_remaining :
  muffin_problem 20 12 4 = 4 := by
  sorry

end muffin_goal_remaining_l700_70005


namespace population_less_than_15_percent_in_fifth_year_l700_70029

def population_decrease_rate : ℝ := 0.35
def target_population_ratio : ℝ := 0.15

def population_after_n_years (n : ℕ) : ℝ :=
  (1 - population_decrease_rate) ^ n

theorem population_less_than_15_percent_in_fifth_year :
  (∀ k < 5, population_after_n_years k > target_population_ratio) ∧
  population_after_n_years 5 < target_population_ratio :=
sorry

end population_less_than_15_percent_in_fifth_year_l700_70029


namespace obstacle_course_probability_l700_70039

def pass_rate_1 : ℝ := 0.8
def pass_rate_2 : ℝ := 0.7
def pass_rate_3 : ℝ := 0.6

theorem obstacle_course_probability :
  let prob_pass_two := pass_rate_1 * pass_rate_2 * (1 - pass_rate_3)
  prob_pass_two = 0.224 := by
sorry

end obstacle_course_probability_l700_70039


namespace hamster_ratio_l700_70099

/-- Proves that the ratio of male hamsters to total hamsters is 1:3 given the specified conditions --/
theorem hamster_ratio (total_pets : ℕ) (total_gerbils : ℕ) (total_males : ℕ) 
  (h1 : total_pets = 92)
  (h2 : total_gerbils = 68)
  (h3 : total_males = 25)
  (h4 : total_gerbils * 1/4 = total_gerbils / 4) -- One-quarter of gerbils are male
  (h5 : total_pets = total_gerbils + (total_pets - total_gerbils)) -- Total pets consist of gerbils and hamsters
  : (total_males - total_gerbils / 4) / (total_pets - total_gerbils) = 1 / 3 := by
  sorry


end hamster_ratio_l700_70099


namespace product_eleven_one_seventeenth_thirtyfour_l700_70038

theorem product_eleven_one_seventeenth_thirtyfour : 11 * (1 / 17) * 34 = 22 := by
  sorry

end product_eleven_one_seventeenth_thirtyfour_l700_70038


namespace smallest_possible_campers_l700_70040

/-- Represents the number of campers participating in different combinations of activities -/
structure CampActivities where
  only_canoeing : ℕ
  canoeing_swimming : ℕ
  only_swimming : ℕ
  canoeing_fishing : ℕ
  swimming_fishing : ℕ
  only_fishing : ℕ

/-- Represents the camp with its activities and camper counts -/
structure Camp where
  activities : CampActivities
  no_activity : ℕ

/-- Calculates the total number of campers in the camp -/
def total_campers (camp : Camp) : ℕ :=
  camp.activities.only_canoeing +
  camp.activities.canoeing_swimming +
  camp.activities.only_swimming +
  camp.activities.canoeing_fishing +
  camp.activities.swimming_fishing +
  camp.activities.only_fishing +
  camp.no_activity

/-- Checks if the camp satisfies the given conditions -/
def satisfies_conditions (camp : Camp) : Prop :=
  (camp.activities.only_canoeing + camp.activities.canoeing_swimming + camp.activities.canoeing_fishing = 15) ∧
  (camp.activities.canoeing_swimming + camp.activities.only_swimming + camp.activities.swimming_fishing = 22) ∧
  (camp.activities.canoeing_fishing + camp.activities.swimming_fishing + camp.activities.only_fishing = 12) ∧
  (camp.no_activity = 9)

theorem smallest_possible_campers :
  ∀ camp : Camp,
    satisfies_conditions camp →
    total_campers camp ≥ 34 :=
by sorry

#check smallest_possible_campers

end smallest_possible_campers_l700_70040


namespace rhombus_area_l700_70064

/-- The area of a rhombus with diagonals of length 3 and 4 is 6 -/
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 3) (h2 : d2 = 4) : 
  (1 / 2 : ℝ) * d1 * d2 = 6 := by
  sorry

#check rhombus_area

end rhombus_area_l700_70064


namespace binomial_coefficient_two_l700_70052

theorem binomial_coefficient_two (n : ℕ) (h : n > 1) : 
  Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end binomial_coefficient_two_l700_70052


namespace people_after_increase_l700_70010

-- Define the initial conditions
def initial_people : ℕ := 5
def initial_houses : ℕ := 5
def initial_days : ℕ := 5

-- Define the new conditions
def new_houses : ℕ := 100
def new_days : ℕ := 5

-- Define the function to calculate the number of people needed
def people_needed (houses : ℕ) (days : ℕ) : ℕ :=
  (houses * initial_people * initial_days) / (initial_houses * days)

-- Theorem statement
theorem people_after_increase :
  people_needed new_houses new_days = 100 :=
by sorry

end people_after_increase_l700_70010


namespace triangle_properties_l700_70081

/-- Theorem about a triangle ABC with specific angle and side properties -/
theorem triangle_properties (A B C : Real) (a b c : Real) (D : Real) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Given equation
  Real.sin A ^ 2 + Real.sin B ^ 2 - Real.sin C ^ 2 = Real.sqrt 2 * Real.sin A * Real.sin B →
  -- Additional conditions
  Real.cos B = 3 / 5 →
  0 < D ∧ D < 1 →  -- Representing CD = 4BD as D = 4/(1+4) = 4/5
  -- Area condition (using scaled version to avoid square root)
  a * c * D * Real.sin A = 14 / 5 →
  -- Conclusions
  C = π / 4 ∧ a = 2 := by
  sorry

#check triangle_properties

end triangle_properties_l700_70081


namespace replaced_person_weight_l700_70091

/-- The weight of the replaced person given the conditions of the problem -/
def weight_of_replaced_person (initial_count : ℕ) (average_increase : ℚ) (new_person_weight : ℚ) : ℚ :=
  new_person_weight - (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the replaced person is 65 kg -/
theorem replaced_person_weight :
  weight_of_replaced_person 8 (5/2) 85 = 65 := by
  sorry

end replaced_person_weight_l700_70091


namespace jills_race_time_l700_70035

/-- Proves that Jill's race time is 32 seconds -/
theorem jills_race_time (jack_first_half : ℕ) (jack_second_half : ℕ) (time_difference : ℕ) :
  jack_first_half = 19 →
  jack_second_half = 6 →
  time_difference = 7 →
  jack_first_half + jack_second_half + time_difference = 32 :=
by sorry

end jills_race_time_l700_70035


namespace min_value_f_f_decreasing_sum_lower_bound_l700_70015

noncomputable section

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b * x^2 + x

-- Part 1
theorem min_value_f (x : ℝ) (h : x > 0) : 
  f (-1) 0 x ≥ 1 :=
sorry

-- Part 2
def f_special (x : ℝ) : ℝ := Real.log x - x^2 + x

theorem f_decreasing (x : ℝ) (h : x > 1) :
  ∀ y > x, f_special y < f_special x :=
sorry

-- Part 3
theorem sum_lower_bound (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0)
  (h : f 1 1 x₁ + f 1 1 x₂ + x₁ * x₂ = 0) :
  x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end min_value_f_f_decreasing_sum_lower_bound_l700_70015


namespace hoodie_price_l700_70079

/-- Proves that the price of the hoodie is $80 given the conditions of Celina's hiking equipment purchase. -/
theorem hoodie_price (total_spent : ℝ) (boots_original : ℝ) (boots_discount : ℝ) (flashlight_ratio : ℝ) 
  (h_total : total_spent = 195)
  (h_boots_original : boots_original = 110)
  (h_boots_discount : boots_discount = 0.1)
  (h_flashlight : flashlight_ratio = 0.2) : 
  ∃ (hoodie_price : ℝ), 
    hoodie_price = 80 ∧ 
    (boots_original * (1 - boots_discount) + flashlight_ratio * hoodie_price + hoodie_price = total_spent) :=
by
  sorry


end hoodie_price_l700_70079


namespace farm_animal_difference_l700_70042

/-- Proves that the difference between the number of goats and pigs is 33 -/
theorem farm_animal_difference : 
  let goats : ℕ := 66
  let chickens : ℕ := 2 * goats
  let ducks : ℕ := (goats + chickens) / 2
  let pigs : ℕ := ducks / 3
  goats - pigs = 33 := by sorry

end farm_animal_difference_l700_70042


namespace fish_disappeared_l700_70036

def original_goldfish : ℕ := 7
def original_catfish : ℕ := 12
def original_guppies : ℕ := 8
def original_angelfish : ℕ := 5
def current_total : ℕ := 27

theorem fish_disappeared : 
  original_goldfish + original_catfish + original_guppies + original_angelfish - current_total = 5 := by
  sorry

end fish_disappeared_l700_70036


namespace number_equivalence_l700_70008

theorem number_equivalence : ∃ x : ℕ, 
  x = 1 ∧ 
  x = 62 ∧ 
  x = 363 ∧ 
  x = 3634 ∧ 
  x = 365 ∧ 
  36 = 2 ∧ 
  x = 2 :=
by sorry

end number_equivalence_l700_70008


namespace max_value_of_sum_products_l700_70020

theorem max_value_of_sum_products (w x y z : ℝ) : 
  w ≥ 0 → x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  w + x + y + z = 200 →
  w * x + x * y + y * z + w * z ≤ 10000 :=
by
  sorry

end max_value_of_sum_products_l700_70020


namespace chocolate_box_count_l700_70037

theorem chocolate_box_count : ∀ (total caramels nougats truffles peanut_clusters : ℕ),
  caramels = 3 →
  nougats = 2 * caramels →
  truffles = caramels + 6 →
  peanut_clusters = total - (caramels + nougats + truffles) →
  (peanut_clusters : ℚ) / total = 64 / 100 →
  total = 50 := by sorry

end chocolate_box_count_l700_70037


namespace intersection_circle_radius_squared_l700_70050

/-- The parabolas y = (x - 2)^2 and x + 6 = (y + 1)^2 intersect at four points. 
    All four points lie on a circle. This theorem proves that the radius squared 
    of this circle is 1/4. -/
theorem intersection_circle_radius_squared (x y : ℝ) : 
  (y = (x - 2)^2 ∧ x + 6 = (y + 1)^2) → 
  ((x - 3/2)^2 + (y + 3/2)^2 = 1/4) :=
by sorry

end intersection_circle_radius_squared_l700_70050


namespace drawings_on_last_page_is_sixty_l700_70026

/-- Represents the problem of rearranging drawings in notebooks --/
structure NotebookProblem where
  initial_notebooks : ℕ
  pages_per_notebook : ℕ
  initial_drawings_per_page : ℕ
  new_drawings_per_page : ℕ
  filled_notebooks : ℕ
  filled_pages_in_last_notebook : ℕ

/-- Calculate the number of drawings on the last page of the partially filled notebook --/
def drawings_on_last_page (p : NotebookProblem) : ℕ :=
  let total_drawings := p.initial_notebooks * p.pages_per_notebook * p.initial_drawings_per_page
  let filled_pages := p.filled_notebooks * p.pages_per_notebook + p.filled_pages_in_last_notebook
  let drawings_on_filled_pages := filled_pages * p.new_drawings_per_page
  total_drawings - drawings_on_filled_pages

/-- The main theorem stating that for the given problem, there are 60 drawings on the last page --/
theorem drawings_on_last_page_is_sixty :
  let p : NotebookProblem := {
    initial_notebooks := 5,
    pages_per_notebook := 60,
    initial_drawings_per_page := 8,
    new_drawings_per_page := 12,
    filled_notebooks := 3,
    filled_pages_in_last_notebook := 45
  }
  drawings_on_last_page p = 60 := by
  sorry


end drawings_on_last_page_is_sixty_l700_70026


namespace average_income_proof_l700_70049

def family_size : ℕ := 4

def income_1 : ℕ := 8000
def income_2 : ℕ := 15000
def income_3 : ℕ := 6000
def income_4 : ℕ := 11000

def total_income : ℕ := income_1 + income_2 + income_3 + income_4

theorem average_income_proof :
  total_income / family_size = 10000 := by
  sorry

end average_income_proof_l700_70049


namespace fourth_divisor_l700_70073

theorem fourth_divisor (n : Nat) (h1 : n = 9600) (h2 : n % 15 = 0) (h3 : n % 25 = 0) (h4 : n % 40 = 0) :
  ∃ m : Nat, m = 16 ∧ n % m = 0 ∧ ∀ k : Nat, k > m → n % k = 0 → (k % 15 = 0 ∨ k % 25 = 0 ∨ k % 40 = 0) :=
by sorry

end fourth_divisor_l700_70073


namespace cricket_run_rate_theorem_l700_70094

/-- Represents a cricket game scenario -/
structure CricketGame where
  total_overs : ℕ
  first_overs : ℕ
  first_run_rate : ℚ
  target : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_overs
  let runs_in_first_overs := game.first_run_rate * game.first_overs
  let remaining_runs := game.target - runs_in_first_overs
  remaining_runs / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.total_overs = 50)
  (h2 : game.first_overs = 10)
  (h3 : game.first_run_rate = 3.6)
  (h4 : game.target = 282) :
  required_run_rate game = 6.15 := by
  sorry

end cricket_run_rate_theorem_l700_70094


namespace derivative_at_one_implies_a_value_l700_70032

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 4 * x^2 + 3 * x

theorem derivative_at_one_implies_a_value (a : ℝ) :
  (∀ x, HasDerivAt (f a) ((3 * a * x^2) + 8 * x + 3) x) →
  HasDerivAt (f a) 2 1 →
  a = -3 := by sorry

end derivative_at_one_implies_a_value_l700_70032


namespace hexagonal_prism_intersection_area_l700_70061

-- Define the hexagonal prism
structure HexagonalPrism :=
  (height : ℝ)
  (side_length : ℝ)

-- Define the plane
structure Plane :=
  (normal : ℝ × ℝ × ℝ)
  (point : ℝ × ℝ × ℝ)

-- Define the area of intersection
def area_of_intersection (prism : HexagonalPrism) (plane : Plane) : ℝ := sorry

-- Theorem statement
theorem hexagonal_prism_intersection_area 
  (prism : HexagonalPrism) 
  (plane : Plane) 
  (h1 : prism.height = 5) 
  (h2 : prism.side_length = 6) 
  (h3 : plane.point = (6, 0, 0)) 
  (h4 : (∃ (t : ℝ), plane.point = (-3, 3 * Real.sqrt 3, 5))) 
  (h5 : (∃ (t : ℝ), plane.point = (-3, -3 * Real.sqrt 3, 0))) : 
  area_of_intersection prism plane = 6 * Real.sqrt 399 := by sorry

end hexagonal_prism_intersection_area_l700_70061


namespace nicks_nacks_nocks_conversion_l700_70088

/-- Given the conversion rates between nicks, nacks, and nocks, 
    prove that 40 nocks is equal to 160/3 nicks. -/
theorem nicks_nacks_nocks_conversion 
  (h1 : (5 : ℚ) * nick = 3 * nack)
  (h2 : (4 : ℚ) * nack = 5 * nock)
  : (40 : ℚ) * nock = 160 / 3 * nick :=
by sorry

end nicks_nacks_nocks_conversion_l700_70088


namespace cosine_tangent_ratio_equals_two_l700_70092

theorem cosine_tangent_ratio_equals_two : 
  (Real.cos (10 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180))) / 
  (Real.cos (50 * π / 180)) = 2 := by
sorry

end cosine_tangent_ratio_equals_two_l700_70092


namespace no_solutions_for_equation_l700_70022

theorem no_solutions_for_equation (x : ℝ) : 
  x > 6 → 
  ¬(Real.sqrt (x + 6 * Real.sqrt (x - 6)) + 3 = Real.sqrt (x - 6 * Real.sqrt (x - 6)) + 3) :=
by
  sorry

end no_solutions_for_equation_l700_70022


namespace range_of_sum_reciprocals_l700_70068

theorem range_of_sum_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : x + 4 * y + 1 / x + 1 / y = 10) :
  1 ≤ 1 / x + 1 / y ∧ 1 / x + 1 / y ≤ 9 := by
  sorry

end range_of_sum_reciprocals_l700_70068


namespace probability_all_red_fourth_draw_correct_l700_70089

/-- Represents the number of white balls initially in the bag -/
def initial_white_balls : ℕ := 8

/-- Represents the number of red balls initially in the bag -/
def initial_red_balls : ℕ := 2

/-- Represents the total number of balls initially in the bag -/
def total_balls : ℕ := initial_white_balls + initial_red_balls

/-- Represents the probability of drawing all red balls exactly after the 4th draw -/
def probability_all_red_fourth_draw : ℝ := 0.0434

/-- Theorem stating the probability of drawing all red balls exactly after the 4th draw -/
theorem probability_all_red_fourth_draw_correct :
  probability_all_red_fourth_draw = 
    (initial_red_balls / total_balls) * 
    ((initial_white_balls + 1) / total_balls) * 
    (initial_red_balls / total_balls) * 
    (1 / (initial_white_balls + 1)) := by
  sorry

end probability_all_red_fourth_draw_correct_l700_70089


namespace max_red_socks_l700_70051

theorem max_red_socks (r g : ℕ) : 
  let t := r + g
  (t ≤ 3000) → 
  (r * (r - 1) + g * (g - 1)) / (t * (t - 1)) = 3/5 →
  r ≤ 1199 :=
sorry

end max_red_socks_l700_70051


namespace goods_selection_theorem_l700_70034

def total_goods : ℕ := 35
def counterfeit_goods : ℕ := 15
def selection_size : ℕ := 3

theorem goods_selection_theorem :
  (Nat.choose (total_goods - 1) (selection_size - 1) = 561) ∧
  (Nat.choose (total_goods - 1) selection_size = 5984) ∧
  (Nat.choose counterfeit_goods 2 * Nat.choose (total_goods - counterfeit_goods) 1 = 2100) ∧
  (Nat.choose counterfeit_goods 2 * Nat.choose (total_goods - counterfeit_goods) 1 + 
   Nat.choose counterfeit_goods 3 = 2555) ∧
  (Nat.choose counterfeit_goods 2 * Nat.choose (total_goods - counterfeit_goods) 1 + 
   Nat.choose counterfeit_goods 1 * Nat.choose (total_goods - counterfeit_goods) 2 + 
   Nat.choose (total_goods - counterfeit_goods) 3 = 6090) := by
  sorry


end goods_selection_theorem_l700_70034


namespace gcd_problem_l700_70006

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 7768) :
  Int.gcd (7 * b^2 + 55 * b + 125) (3 * b + 10) = 10 := by
  sorry

end gcd_problem_l700_70006


namespace log_equality_ratio_l700_70085

theorem log_equality_ratio (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : Real.log a / Real.log 8 = Real.log b / Real.log 18 ∧ 
       Real.log a / Real.log 8 = Real.log (a + b) / Real.log 32) : 
  b / a = (3 + 2 * (Real.log 3 / Real.log 2)) / (1 + 2 * (Real.log 3 / Real.log 2) + 5) := by
sorry

end log_equality_ratio_l700_70085


namespace unique_k_for_inequality_l700_70072

theorem unique_k_for_inequality :
  ∃! k : ℝ, ∀ t : ℝ, t ∈ Set.Ioo (-1) 1 →
    (1 + t) ^ k * (1 - t) ^ (1 - k) ≤ 1 :=
by
  -- The proof goes here
  sorry

end unique_k_for_inequality_l700_70072


namespace ab_plus_one_gt_a_plus_b_l700_70058

-- Define the set M
def M : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem ab_plus_one_gt_a_plus_b (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  a * b + 1 > a + b := by
  sorry

end ab_plus_one_gt_a_plus_b_l700_70058


namespace stevens_apple_peach_difference_prove_stevens_apple_peach_difference_l700_70054

/-- Given that Jake has 3 fewer peaches and 4 more apples than Steven, and Steven has 19 apples,
    prove that the difference between Steven's apples and peaches is 19 - P,
    where P is the number of peaches Steven has. -/
theorem stevens_apple_peach_difference (P : ℕ) : ℕ → Prop :=
  let steven_apples : ℕ := 19
  let steven_peaches : ℕ := P
  let jake_peaches : ℕ := P - 3
  let jake_apples : ℕ := steven_apples + 4
  λ _ => steven_apples - steven_peaches = 19 - P

/-- Proof of the theorem -/
theorem prove_stevens_apple_peach_difference (P : ℕ) :
  stevens_apple_peach_difference P P :=
by
  sorry

end stevens_apple_peach_difference_prove_stevens_apple_peach_difference_l700_70054


namespace common_ratio_is_two_l700_70044

/-- An increasing geometric sequence with specific conditions -/
structure IncreasingGeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  is_increasing : q > 1
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * q
  a2_eq_2 : a 2 = 2
  a4_minus_a3_eq_4 : a 4 - a 3 = 4

/-- The common ratio of the increasing geometric sequence is 2 -/
theorem common_ratio_is_two (seq : IncreasingGeometricSequence) : seq.q = 2 := by
  sorry

end common_ratio_is_two_l700_70044


namespace jazel_sticks_total_length_l700_70056

/-- Given Jazel's three sticks with specified lengths, prove that their total length is 14 centimeters. -/
theorem jazel_sticks_total_length :
  let first_stick : ℕ := 3
  let second_stick : ℕ := 2 * first_stick
  let third_stick : ℕ := second_stick - 1
  first_stick + second_stick + third_stick = 14 := by
  sorry

end jazel_sticks_total_length_l700_70056


namespace students_liking_both_channels_l700_70025

theorem students_liking_both_channels
  (total : ℕ)
  (sports : ℕ)
  (arts : ℕ)
  (neither : ℕ)
  (h1 : total = 100)
  (h2 : sports = 68)
  (h3 : arts = 55)
  (h4 : neither = 3)
  : (sports + arts) - (total - neither) = 26 :=
by sorry

end students_liking_both_channels_l700_70025


namespace perpendicular_distance_extrema_l700_70095

/-- Given two points on a line, prove that the sum of j values for (6, j) 
    that maximize and minimize squared perpendicular distances to the line is 13 -/
theorem perpendicular_distance_extrema (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : x₁ = 2 ∧ y₁ = 9) (h₂ : x₂ = 14 ∧ y₂ = 20) : 
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  let y_line := m * 6 + b
  let j_max := ⌈y_line⌉ 
  let j_min := ⌊y_line⌋
  j_max + j_min = 13 := by
  sorry

end perpendicular_distance_extrema_l700_70095


namespace sin_585_degrees_l700_70017

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_585_degrees_l700_70017


namespace chord_equation_l700_70024

/-- Given a parabola and a chord, prove the equation of the line containing the chord -/
theorem chord_equation (x₁ x₂ y₁ y₂ : ℝ) : 
  (x₁^2 = -2*y₁) →  -- Point A on parabola
  (x₂^2 = -2*y₂) →  -- Point B on parabola
  (x₁ + x₂ = -2) →  -- Sum of x-coordinates
  ((x₁ + x₂)/2 = -1) →  -- x-coordinate of midpoint
  ((y₁ + y₂)/2 = -5) →  -- y-coordinate of midpoint
  ∃ (m b : ℝ), ∀ x y, (y = m*x + b) ↔ (y - y₁)*(x₂ - x₁) = (x - x₁)*(y₂ - y₁) ∧ m = 1 ∧ b = -4 :=
sorry

end chord_equation_l700_70024


namespace point_on_circle_l700_70087

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def arc_length (θ : ℝ) : ℝ := θ

theorem point_on_circle (P Q : ℝ × ℝ) :
  P = (1, 0) →
  unit_circle P.1 P.2 →
  unit_circle Q.1 Q.2 →
  arc_length (4 * π / 3) = abs (Real.arccos P.1 - Real.arccos Q.1) →
  Q = (-1/2, Real.sqrt 3 / 2) := by
  sorry

end point_on_circle_l700_70087


namespace greatest_integer_difference_l700_70075

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 10) :
  (⌊y⌋ - ⌈x⌉ : ℤ) ≤ 5 ∧ ∃ (x' y' : ℝ), 3 < x' ∧ x' < 6 ∧ 6 < y' ∧ y' < 10 ∧ ⌊y'⌋ - ⌈x'⌉ = 5 := by
  sorry

end greatest_integer_difference_l700_70075


namespace sum_of_reciprocals_l700_70013

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  1 / x + 1 / y = 6 := by
  sorry

end sum_of_reciprocals_l700_70013


namespace school_sections_l700_70097

/-- Given a school with 408 boys and 216 girls, prove that when divided into equal sections
    of either boys or girls alone, the total number of sections formed is 26. -/
theorem school_sections (num_boys num_girls : ℕ) 
    (h_boys : num_boys = 408) 
    (h_girls : num_girls = 216) : 
    (num_boys / (Nat.gcd num_boys num_girls)) + (num_girls / (Nat.gcd num_boys num_girls)) = 26 := by
  sorry

end school_sections_l700_70097


namespace three_heads_before_four_tails_l700_70055

/-- The probability of encountering 3 heads before 4 tails in repeated fair coin flips -/
def probability_three_heads_before_four_tails : ℚ := 4/7

/-- A fair coin has equal probability of heads and tails -/
axiom fair_coin : ℚ

/-- The probability of heads for a fair coin is 1/2 -/
axiom fair_coin_probability : fair_coin = 1/2

theorem three_heads_before_four_tails :
  probability_three_heads_before_four_tails = 4/7 :=
by sorry

end three_heads_before_four_tails_l700_70055


namespace integer_roots_of_polynomial_l700_70096

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 30

def possible_roots : Set ℤ := {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30}

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | ∃ (y : ℤ), polynomial b₂ b₁ x = 0} = possible_roots :=
sorry

end integer_roots_of_polynomial_l700_70096


namespace tan_value_fourth_quadrant_l700_70080

/-- An angle in the fourth quadrant -/
structure FourthQuadrantAngle where
  α : Real
  in_fourth_quadrant : α > -π/2 ∧ α < 0

/-- A point on the terminal side of an angle -/
structure TerminalPoint where
  x : Real
  y : Real

/-- Properties of the angle α -/
structure AngleProperties (α : FourthQuadrantAngle) where
  terminal_point : TerminalPoint
  x_coord : terminal_point.x = 4
  sin_value : Real.sin α.α = terminal_point.y / 5

theorem tan_value_fourth_quadrant (α : FourthQuadrantAngle) 
  (props : AngleProperties α) : Real.tan α.α = -3/4 := by
  sorry

end tan_value_fourth_quadrant_l700_70080


namespace total_questions_is_60_l700_70059

/-- Represents the citizenship test study problem --/
def CitizenshipTestStudy : Prop :=
  let multipleChoice : ℕ := 30
  let fillInBlank : ℕ := 30
  let multipleChoiceTime : ℕ := 15
  let fillInBlankTime : ℕ := 25
  let totalStudyTime : ℕ := 20 * 60

  (multipleChoice * multipleChoiceTime + fillInBlank * fillInBlankTime = totalStudyTime) →
  (multipleChoice + fillInBlank = 60)

/-- Theorem stating that the total number of questions on the test is 60 --/
theorem total_questions_is_60 : CitizenshipTestStudy := by
  sorry

end total_questions_is_60_l700_70059


namespace complex_fraction_equality_l700_70065

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) :
  ∃ ζ : ℂ, ζ^3 = 1 ∧ ζ ≠ 1 ∧ (a^9 + b^9) / (a - b)^9 = 2 / (81 * (ζ - 1)) := by
  sorry

end complex_fraction_equality_l700_70065


namespace sin_2x_equiv_cos_2x_shifted_l700_70045

theorem sin_2x_equiv_cos_2x_shifted (x : ℝ) : 
  Real.sin (2 * x) = Real.cos (2 * (x - Real.pi / 4)) := by
  sorry

end sin_2x_equiv_cos_2x_shifted_l700_70045


namespace arithmetic_calculations_l700_70000

theorem arithmetic_calculations :
  (((1 : ℤ) - 32 - 11 + (-9) - (-16)) = -36) ∧
  (-(1 : ℚ)^4 - |0 - 1| * 2 - (-3)^2 / (-3/2) = 3) := by sorry

end arithmetic_calculations_l700_70000


namespace expand_algebraic_expression_l700_70069

theorem expand_algebraic_expression (a b : ℝ) : 3*a*(5*a - 2*b) = 15*a^2 - 6*a*b := by
  sorry

end expand_algebraic_expression_l700_70069


namespace equation_solution_l700_70021

theorem equation_solution :
  let x : ℝ := -Real.sqrt 3
  let y : ℝ := 4
  x^2 + 2 * Real.sqrt 3 * x + y - 4 * Real.sqrt y + 7 = 0 :=
by
  sorry

end equation_solution_l700_70021


namespace base6_divisible_by_13_l700_70090

def base6_to_base10 (d : Nat) : Nat :=
  2 * 6^3 + d * 6^2 + d * 6 + 3

theorem base6_divisible_by_13 (d : Nat) :
  d ≤ 5 → (base6_to_base10 d % 13 = 0 ↔ d = 5) := by
  sorry

end base6_divisible_by_13_l700_70090


namespace sqrt_six_and_quarter_equals_five_halves_l700_70077

theorem sqrt_six_and_quarter_equals_five_halves :
  Real.sqrt (6 + 1/4) = 5/2 := by sorry

end sqrt_six_and_quarter_equals_five_halves_l700_70077


namespace intersection_point_of_linear_function_and_inverse_l700_70041

-- Define the function f
def f (b : ℤ) : ℝ → ℝ := λ x ↦ 4 * x + b

-- Define the theorem
theorem intersection_point_of_linear_function_and_inverse
  (b : ℤ) (a : ℤ) :
  (f b (-4) = a ∧ f b a = -4) → a = -4 :=
by sorry

end intersection_point_of_linear_function_and_inverse_l700_70041


namespace quadratic_solution_property_l700_70004

theorem quadratic_solution_property (a b : ℝ) : 
  (a * 1^2 + b * 1 + 1 = 0) → (3 - a - b = 4) := by
  sorry

end quadratic_solution_property_l700_70004


namespace base3_to_base10_equality_l700_70028

/-- Converts a base-3 number to base-10 --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

/-- The base-3 representation of the number --/
def base3Number : List Nat := [1, 2, 0, 1, 2]

/-- Theorem stating that the base-3 number 12012 is equal to 140 in base-10 --/
theorem base3_to_base10_equality : base3ToBase10 base3Number = 140 := by
  sorry

end base3_to_base10_equality_l700_70028


namespace surface_area_of_stacked_cubes_l700_70012

/-- The surface area of a cube formed by stacking smaller cubes -/
theorem surface_area_of_stacked_cubes (n : Nat) (side_length : Real) :
  n > 0 →
  side_length > 0 →
  n = 27 →
  side_length = 3 →
  let large_cube_side := (n ^ (1 / 3 : Real)) * side_length
  6 * large_cube_side ^ 2 = 486 := by
  sorry

end surface_area_of_stacked_cubes_l700_70012
