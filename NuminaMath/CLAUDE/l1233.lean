import Mathlib

namespace x_squared_plus_reciprocal_squared_l1233_123363

theorem x_squared_plus_reciprocal_squared (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end x_squared_plus_reciprocal_squared_l1233_123363


namespace persimmon_basket_weight_l1233_123346

theorem persimmon_basket_weight (total_weight half_weight : ℝ)
  (h1 : total_weight = 62)
  (h2 : half_weight = 34)
  (h3 : ∃ (basket_weight persimmon_weight : ℝ),
    basket_weight + persimmon_weight = total_weight ∧
    basket_weight + persimmon_weight / 2 = half_weight) :
  ∃ (basket_weight : ℝ), basket_weight = 6 := by
sorry

end persimmon_basket_weight_l1233_123346


namespace student_correct_answers_l1233_123343

/-- Represents a multiple-choice test with scoring rules -/
structure MCTest where
  total_questions : ℕ
  correct_points : ℕ
  incorrect_points : ℕ

/-- Represents a student's test result -/
structure TestResult where
  test : MCTest
  total_score : ℤ

/-- Calculates the number of correctly answered questions -/
def correct_answers (result : TestResult) : ℕ :=
  sorry

/-- Theorem stating the problem and its solution -/
theorem student_correct_answers
  (test : MCTest)
  (result : TestResult)
  (h1 : test.total_questions = 25)
  (h2 : test.correct_points = 4)
  (h3 : test.incorrect_points = 1)
  (h4 : result.test = test)
  (h5 : result.total_score = 85) :
  correct_answers result = 22 := by
  sorry

end student_correct_answers_l1233_123343


namespace manny_marbles_l1233_123307

theorem manny_marbles (total_marbles : ℕ) (marbles_per_pack : ℕ) (kept_packs : ℕ) (neil_fraction : ℚ) :
  total_marbles = 400 →
  marbles_per_pack = 10 →
  kept_packs = 25 →
  neil_fraction = 1/8 →
  let total_packs := total_marbles / marbles_per_pack
  let given_packs := total_packs - kept_packs
  let neil_packs := neil_fraction * total_packs
  let manny_packs := given_packs - neil_packs
  manny_packs / total_packs = 1/4 := by sorry

end manny_marbles_l1233_123307


namespace courtyard_tile_cost_l1233_123351

/-- Calculates the total cost of tiles for a courtyard --/
def total_tile_cost (length width : ℝ) (tiles_per_sqft : ℝ) 
  (green_tile_percentage : ℝ) (green_tile_cost red_tile_cost : ℝ) : ℝ :=
  let area := length * width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := total_tiles * green_tile_percentage
  let red_tiles := total_tiles - green_tiles
  (green_tiles * green_tile_cost) + (red_tiles * red_tile_cost)

/-- Theorem stating the total cost of tiles for the given courtyard specifications --/
theorem courtyard_tile_cost :
  total_tile_cost 10 25 4 0.4 3 1.5 = 2100 := by
  sorry

end courtyard_tile_cost_l1233_123351


namespace max_value_n_is_3210_l1233_123314

/-- S(a) represents the sum of the digits of a natural number a -/
def S (a : ℕ) : ℕ := sorry

/-- allDigitsDifferent n is true if all digits of n are different -/
def allDigitsDifferent (n : ℕ) : Prop := sorry

/-- maxValueN is the maximum value of n satisfying the given conditions -/
def maxValueN : ℕ := 3210

theorem max_value_n_is_3210 :
  ∀ n : ℕ, allDigitsDifferent n → S (3 * n) = 3 * S n → n ≤ maxValueN := by
  sorry

end max_value_n_is_3210_l1233_123314


namespace b_k_divisible_by_9_count_l1233_123330

/-- The sequence b_n is defined as the number obtained by concatenating
    integers from 1 to n and subtracting n -/
def b (n : ℕ) : ℕ := sorry

/-- g(n) represents the sum of digits of n -/
def g (n : ℕ) : ℕ := sorry

/-- The number of b_k divisible by 9 for 1 ≤ k ≤ 100 -/
def count_divisible_by_9 : ℕ := sorry

theorem b_k_divisible_by_9_count :
  count_divisible_by_9 = 22 := by sorry

end b_k_divisible_by_9_count_l1233_123330


namespace ratio_problem_l1233_123369

theorem ratio_problem (x : ℚ) : x / 8 = 6 / (4 * 60) ↔ x = 1 / 5 := by
  sorry

end ratio_problem_l1233_123369


namespace largest_sum_is_five_sixths_l1233_123370

theorem largest_sum_is_five_sixths :
  let sums : List ℚ := [1/3 + 1/2, 1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/7, 1/3 + 1/9]
  ∀ x ∈ sums, x ≤ 5/6 ∧ (5/6 ∈ sums) := by
  sorry

end largest_sum_is_five_sixths_l1233_123370


namespace cuboid_s_value_l1233_123382

/-- Represents a cuboid with adjacent face areas a, b, and s, 
    whose vertices lie on a sphere with surface area sa -/
structure Cuboid where
  a : ℝ
  b : ℝ
  s : ℝ
  sa : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < s ∧ 0 < sa
  h_sphere : sa = 152 * Real.pi
  h_face1 : a * b = 6
  h_face2 : b * (s / b) = 10
  h_vertices_on_sphere : ∃ (r : ℝ), 
    a^2 + b^2 + (s / b)^2 = 4 * r^2 ∧ sa = 4 * Real.pi * r^2

/-- The theorem stating that for a cuboid satisfying the given conditions, s must equal 15 -/
theorem cuboid_s_value (c : Cuboid) : c.s = 15 := by
  sorry

end cuboid_s_value_l1233_123382


namespace x_minus_y_equals_three_l1233_123359

theorem x_minus_y_equals_three (x y : ℝ) 
  (h1 : x + y = 8) 
  (h2 : x^2 - y^2 = 24) : 
  x - y = 3 := by
sorry

end x_minus_y_equals_three_l1233_123359


namespace circle_radius_increase_l1233_123318

theorem circle_radius_increase (r n : ℝ) : 
  r > 0 → r > n → π * (r + n)^2 = 3 * π * r^2 → r = n * (Real.sqrt 3 - 1) / 2 := by
  sorry

end circle_radius_increase_l1233_123318


namespace translate_line_example_l1233_123338

/-- Represents a line in the form y = mx + b -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translates a line vertically by a given amount -/
def translate_line (l : Line) (y_shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + y_shift }

/-- The theorem stating that translating y = 3x - 3 upwards by 5 units results in y = 3x + 2 -/
theorem translate_line_example :
  let original_line : Line := { slope := 3, intercept := -3 }
  let translated_line := translate_line original_line 5
  translated_line = { slope := 3, intercept := 2 } := by
  sorry

end translate_line_example_l1233_123338


namespace kay_weight_training_time_l1233_123317

/-- Represents the weekly exercise schedule -/
structure ExerciseSchedule where
  total_time : ℕ
  aerobics_ratio : ℕ
  weight_training_ratio : ℕ

/-- Calculates the time spent on weight training given an exercise schedule -/
def weight_training_time (schedule : ExerciseSchedule) : ℕ :=
  (schedule.total_time * schedule.weight_training_ratio) / (schedule.aerobics_ratio + schedule.weight_training_ratio)

/-- Theorem: Given Kay's exercise schedule, she spends 100 minutes on weight training -/
theorem kay_weight_training_time :
  let kay_schedule : ExerciseSchedule := {
    total_time := 250,
    aerobics_ratio := 3,
    weight_training_ratio := 2
  }
  weight_training_time kay_schedule = 100 := by
  sorry

end kay_weight_training_time_l1233_123317


namespace calc_expression_equality_simplify_fraction_equality_l1233_123394

-- Part 1
theorem calc_expression_equality : 
  (-1/2)⁻¹ + Real.sqrt 2 * Real.sqrt 6 - (π - 3)^0 + abs (Real.sqrt 3 - 2) = -1 + Real.sqrt 3 := by sorry

-- Part 2
theorem simplify_fraction_equality (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) : 
  (x^2 - 1) / (x + 1) / ((x^2 - 2*x + 1) / (x^2 - x)) = x / (x - 1) := by sorry

end calc_expression_equality_simplify_fraction_equality_l1233_123394


namespace original_decimal_l1233_123393

theorem original_decimal (x : ℝ) : (1000 * x) / 100 = 12.5 → x = 1.25 := by
  sorry

end original_decimal_l1233_123393


namespace orange_bags_weight_l1233_123348

/-- If 12 bags of oranges weigh 24 pounds, then 8 bags of oranges weigh 16 pounds. -/
theorem orange_bags_weight (weight_12_bags : ℝ) (h : weight_12_bags = 24) : 
  (8 / 12) * weight_12_bags = 16 := by
  sorry

end orange_bags_weight_l1233_123348


namespace intersection_complement_theorem_l1233_123306

-- Define the sets
def A : Set ℝ := {y | ∃ x, y = x^2}
def B : Set ℝ := {x | x > 3}

-- State the theorem
theorem intersection_complement_theorem :
  A ∩ (Set.univ \ B) = Set.Icc 0 3 := by sorry

end intersection_complement_theorem_l1233_123306


namespace min_students_for_given_data_l1233_123323

/-- Represents the number of students receiving A's on each day of the week -/
structure GradeData where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- The minimum number of students in the class given the grade data -/
def minStudents (data : GradeData) : Nat :=
  max (data.monday + data.tuesday)
    (max (data.tuesday + data.wednesday)
      (max (data.wednesday + data.thursday)
        (data.thursday + data.friday)))

/-- Theorem stating the minimum number of students given the specific grade data -/
theorem min_students_for_given_data :
  let data : GradeData := {
    monday := 5,
    tuesday := 8,
    wednesday := 6,
    thursday := 4,
    friday := 9
  }
  minStudents data = 14 := by sorry

end min_students_for_given_data_l1233_123323


namespace sum_a_d_g_equals_six_l1233_123342

-- Define the variables
variable (a b c d e f g : ℤ)

-- State the theorem
theorem sum_a_d_g_equals_six 
  (eq1 : a + b + e = 7)
  (eq2 : b + c + f = 10)
  (eq3 : c + d + g = 6)
  (eq4 : e + f + g = 9) :
  a + d + g = 6 := by
  sorry

end sum_a_d_g_equals_six_l1233_123342


namespace machine_no_repair_l1233_123311

/-- Represents the state of a portion measuring machine -/
structure PortionMachine where
  max_deviation : ℝ
  nominal_mass : ℝ
  unreadable_deviation_bound : ℝ
  standard_deviation : ℝ

/-- Determines if a portion measuring machine requires repair -/
def requires_repair (m : PortionMachine) : Prop :=
  m.max_deviation > 0.1 * m.nominal_mass ∨
  m.unreadable_deviation_bound ≥ m.max_deviation ∨
  m.standard_deviation > m.max_deviation

/-- Theorem stating that the given machine does not require repair -/
theorem machine_no_repair (m : PortionMachine)
  (h1 : m.max_deviation = 37)
  (h2 : m.max_deviation ≤ 0.1 * m.nominal_mass)
  (h3 : m.unreadable_deviation_bound < m.max_deviation)
  (h4 : m.standard_deviation ≤ m.max_deviation) :
  ¬(requires_repair m) :=
sorry

end machine_no_repair_l1233_123311


namespace jump_rope_total_l1233_123333

theorem jump_rope_total (taehyung_jumps_per_day : ℕ) (taehyung_days : ℕ) 
                        (namjoon_jumps_per_day : ℕ) (namjoon_days : ℕ) :
  taehyung_jumps_per_day = 56 →
  taehyung_days = 3 →
  namjoon_jumps_per_day = 35 →
  namjoon_days = 4 →
  taehyung_jumps_per_day * taehyung_days + namjoon_jumps_per_day * namjoon_days = 308 :=
by
  sorry

end jump_rope_total_l1233_123333


namespace eve_distance_difference_l1233_123327

def running_intervals : List ℝ := [0.75, 0.85, 0.95]
def walking_intervals : List ℝ := [0.50, 0.65, 0.75, 0.80]

theorem eve_distance_difference :
  (running_intervals.sum - walking_intervals.sum) = -0.15 := by
  sorry

end eve_distance_difference_l1233_123327


namespace sin_cos_difference_65_35_l1233_123334

theorem sin_cos_difference_65_35 :
  Real.sin (65 * π / 180) * Real.cos (35 * π / 180) -
  Real.cos (65 * π / 180) * Real.sin (35 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_difference_65_35_l1233_123334


namespace max_value_3sin2x_l1233_123377

theorem max_value_3sin2x :
  ∀ x : ℝ, 3 * Real.sin (2 * x) ≤ 3 ∧ ∃ x₀ : ℝ, 3 * Real.sin (2 * x₀) = 3 :=
by sorry

end max_value_3sin2x_l1233_123377


namespace truck_rental_theorem_l1233_123304

/-- Represents the number of trucks on a rental lot -/
structure TruckLot where
  monday : ℕ
  rented : ℕ
  returned : ℕ
  saturday : ℕ

/-- Conditions for the truck rental problem -/
def truck_rental_conditions (lot : TruckLot) : Prop :=
  lot.monday = 20 ∧
  lot.rented ≤ 20 ∧
  lot.returned = lot.rented / 2 ∧
  lot.saturday = lot.monday - lot.rented + lot.returned

theorem truck_rental_theorem (lot : TruckLot) :
  truck_rental_conditions lot → lot.saturday = 10 :=
by
  sorry

#check truck_rental_theorem

end truck_rental_theorem_l1233_123304


namespace giyun_distance_to_school_l1233_123375

/-- The distance between Giyun's house and school -/
def distance_to_school (step_length : ℝ) (steps_per_minute : ℕ) (time_taken : ℕ) : ℝ :=
  step_length * (steps_per_minute : ℝ) * time_taken

/-- Theorem stating the distance between Giyun's house and school -/
theorem giyun_distance_to_school :
  distance_to_school 0.75 70 13 = 682.5 := by
  sorry

end giyun_distance_to_school_l1233_123375


namespace digits_after_decimal_point_l1233_123366

theorem digits_after_decimal_point : ∃ (n : ℕ), 
  (5^8 : ℚ) / (2^5 * 10^6) = (n : ℚ) / 10^11 ∧ 
  0 < n ∧ 
  n < 10^11 := by
sorry

end digits_after_decimal_point_l1233_123366


namespace cara_right_neighbors_l1233_123361

/-- The number of Cara's friends -/
def num_friends : ℕ := 7

/-- The number of different friends who can sit immediately to Cara's right -/
def num_right_neighbors : ℕ := num_friends

theorem cara_right_neighbors :
  num_right_neighbors = num_friends :=
by sorry

end cara_right_neighbors_l1233_123361


namespace probability_all_white_balls_l1233_123310

theorem probability_all_white_balls (total_balls : ℕ) (white_balls : ℕ) (drawn_balls : ℕ) :
  total_balls = 11 →
  white_balls = 5 →
  drawn_balls = 5 →
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 1 / 462 := by
sorry

end probability_all_white_balls_l1233_123310


namespace task_completion_probability_l1233_123381

theorem task_completion_probability (p_task1 p_task1_not_task2 : ℝ) 
  (h1 : p_task1 = 3/8)
  (h2 : p_task1_not_task2 = 0.15)
  (h3 : 0 ≤ p_task1 ∧ p_task1 ≤ 1)
  (h4 : 0 ≤ p_task1_not_task2 ∧ p_task1_not_task2 ≤ 1) :
  ∃ p_task2 : ℝ, p_task2 = 0.6 ∧ 0 ≤ p_task2 ∧ p_task2 ≤ 1 :=
by sorry

end task_completion_probability_l1233_123381


namespace checkerboard_exists_l1233_123328

/-- Represents a cell on the board -/
inductive Cell
| Black
| White

/-- Represents the board -/
def Board := Fin 100 → Fin 100 → Cell

/-- Checks if a cell is adjacent to the border -/
def isBorderAdjacent (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- Checks if a 2x2 square is monochromatic -/
def isMonochromatic (board : Board) (i j : Fin 100) : Prop :=
  ∃ c : Cell, 
    board i j = c ∧ board (i+1) j = c ∧ 
    board i (j+1) = c ∧ board (i+1) (j+1) = c

/-- Checks if a 2x2 square has a checkerboard pattern -/
def isCheckerboard (board : Board) (i j : Fin 100) : Prop :=
  (board i j = board (i+1) (j+1) ∧ board (i+1) j = board i (j+1)) ∧
  (board i j ≠ board (i+1) j)

/-- The main theorem -/
theorem checkerboard_exists (board : Board) 
  (border_black : ∀ i j : Fin 100, isBorderAdjacent i j → board i j = Cell.Black)
  (no_monochromatic : ∀ i j : Fin 100, ¬isMonochromatic board i j) :
  ∃ i j : Fin 100, isCheckerboard board i j :=
sorry

end checkerboard_exists_l1233_123328


namespace vertical_equality_puzzle_l1233_123303

theorem vertical_equality_puzzle :
  ∃ (a b c d e f g h i j : ℕ),
    a = 1 ∧ b = 9 ∧ c = 8 ∧ d = 5 ∧ e = 4 ∧ f = 0 ∧ g = 6 ∧ h = 7 ∧ i = 2 ∧ j = 3 ∧
    (100 * a + 10 * b + c) - (10 * d + c) = (100 * a + 10 * e + f) ∧
    g * h = 10 * e + i ∧
    (10 * j + j) + (10 * g + d) = 10 * b + c ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j :=
by
  sorry

end vertical_equality_puzzle_l1233_123303


namespace a_plus_b_value_l1233_123340

theorem a_plus_b_value (a b : ℝ) 
  (h1 : |(-a)| = |(-1)|) 
  (h2 : b^2 = 9)
  (h3 : |a - b| = b - a) : 
  a + b = 2 ∨ a + b = 4 := by
sorry

end a_plus_b_value_l1233_123340


namespace juice_theorem_l1233_123354

def juice_problem (sam_initial ben_initial sam_consumed ben_consumed sam_received : ℚ) : Prop :=
  let sam_final := sam_consumed + sam_received
  let ben_final := ben_consumed - sam_received
  sam_initial = 12 ∧
  ben_initial = sam_initial + 8 ∧
  sam_consumed = 2 / 3 * sam_initial ∧
  ben_consumed = 2 / 3 * ben_initial ∧
  sam_received = (1 / 2 * (ben_initial - ben_consumed)) + 1 ∧
  sam_final = ben_final ∧
  sam_initial + ben_initial = 32

theorem juice_theorem :
  ∃ (sam_initial ben_initial sam_consumed ben_consumed sam_received : ℚ),
    juice_problem sam_initial ben_initial sam_consumed ben_consumed sam_received :=
by
  sorry

#check juice_theorem

end juice_theorem_l1233_123354


namespace min_triangle_area_l1233_123335

/-- A point in the 2D Cartesian plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- Definition of a rectangle OABC with O at origin and B at (9, 8) -/
def rectangle : Set IntPoint :=
  {p : IntPoint | 0 ≤ p.x ∧ p.x ≤ 9 ∧ 0 ≤ p.y ∧ p.y ≤ 8}

/-- Area of triangle OBX given point X -/
def triangleArea (X : IntPoint) : ℚ :=
  (1 / 2 : ℚ) * |9 * X.y - 8 * X.x|

/-- Theorem stating the minimum area of triangle OBX -/
theorem min_triangle_area :
  ∃ (min_area : ℚ), min_area = 1/2 ∧
  ∀ (X : IntPoint), X ∈ rectangle → triangleArea X ≥ min_area :=
sorry

end min_triangle_area_l1233_123335


namespace min_a_value_l1233_123389

-- Define the functions f and g
def f (a x : ℝ) : ℝ := a * x^3
def g (x : ℝ) : ℝ := 9 * x^2 + 3 * x - 1

-- State the theorem
theorem min_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, f a x ≥ g x) → a ≥ 11 := by
  sorry

end min_a_value_l1233_123389


namespace complex_modulus_problem_l1233_123350

theorem complex_modulus_problem (z : ℂ) : 
  z * (1 + Complex.I) = 4 - 2 * Complex.I → Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_modulus_problem_l1233_123350


namespace function_ordering_l1233_123398

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the property of being monotonically decreasing on an interval
def is_monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

-- State the theorem
theorem function_ordering (h1 : is_even f) (h2 : is_monotone_decreasing_on (fun x => f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 :=
sorry

end function_ordering_l1233_123398


namespace rug_inner_length_is_four_l1233_123321

/-- Represents a rectangular rug with three nested regions -/
structure Rug where
  inner_width : ℝ
  inner_length : ℝ
  middle_width : ℝ
  middle_length : ℝ
  outer_width : ℝ
  outer_length : ℝ

/-- Calculates the area of a rectangle -/
def area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop := b - a = c - b

theorem rug_inner_length_is_four (r : Rug) : 
  r.inner_width = 2 ∧ 
  r.middle_width = r.inner_width + 4 ∧ 
  r.outer_width = r.middle_width + 4 ∧
  r.middle_length = r.inner_length + 4 ∧
  r.outer_length = r.middle_length + 4 ∧
  isArithmeticProgression 
    (area r.inner_width r.inner_length)
    (area r.middle_width r.middle_length - area r.inner_width r.inner_length)
    (area r.outer_width r.outer_length - area r.middle_width r.middle_length) →
  r.inner_length = 4 := by
sorry

end rug_inner_length_is_four_l1233_123321


namespace remaining_payment_l1233_123309

theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (h1 : deposit = 80) (h2 : deposit_percentage = 0.1) :
  let total_cost := deposit / deposit_percentage
  total_cost - deposit = 720 := by
sorry

end remaining_payment_l1233_123309


namespace inverse_proportion_problem_l1233_123341

/-- Given two real numbers are inversely proportional, if one is 40 when the other is 5,
    then the first is 25 when the second is 8. -/
theorem inverse_proportion_problem (r s : ℝ) (h : ∃ k : ℝ, r * s = k) 
    (h1 : ∃ r0 : ℝ, r0 * 5 = 40 ∧ r0 * s = r * s) : 
    r * 8 = 25 * s := by
  sorry

end inverse_proportion_problem_l1233_123341


namespace min_value_expression_l1233_123397

theorem min_value_expression (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  (x^2 / (y - 1)) + (y^2 / (z - 1)) + (z^2 / (x - 1)) ≥ 12 ∧
  ((x^2 / (y - 1)) + (y^2 / (z - 1)) + (z^2 / (x - 1)) = 12 ↔ x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

end min_value_expression_l1233_123397


namespace hexagon_division_divisible_by_three_l1233_123395

/-- A regular hexagon divided into congruent parallelograms -/
structure RegularHexagonDivision where
  /-- The number of congruent parallelograms -/
  N : ℕ
  /-- The hexagon is divided into N congruent parallelograms -/
  is_division : N > 0

/-- Theorem: The number of congruent parallelograms in a regular hexagon division is divisible by 3 -/
theorem hexagon_division_divisible_by_three (h : RegularHexagonDivision) : 
  ∃ k : ℕ, h.N = 3 * k := by
  sorry

end hexagon_division_divisible_by_three_l1233_123395


namespace positive_derivative_implies_increasing_exists_increasing_with_nonpositive_derivative_l1233_123353

open Set
open Function

-- Define a differentiable function on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define monotonically increasing
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Part 1: If f'(x) > 0 for all x, then f is monotonically increasing
theorem positive_derivative_implies_increasing :
  (∀ x, deriv f x > 0) → MonotonicallyIncreasing f :=
sorry

-- Part 2: There exists a monotonically increasing function with f'(x) ≤ 0 for some x
theorem exists_increasing_with_nonpositive_derivative :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ MonotonicallyIncreasing f ∧ ∃ x, deriv f x ≤ 0 :=
sorry

end positive_derivative_implies_increasing_exists_increasing_with_nonpositive_derivative_l1233_123353


namespace rectangle_ratio_theorem_l1233_123358

theorem rectangle_ratio_theorem (s : ℝ) (x y : ℝ) (h1 : s > 0) (h2 : x > 0) (h3 : y > 0) : 
  (s + 2*y = 3*s) → (x + y = 3*s) → (x / y = 2) := by
  sorry

#check rectangle_ratio_theorem

end rectangle_ratio_theorem_l1233_123358


namespace roots_of_p_l1233_123374

def p (x : ℝ) : ℝ := x * (x + 3)^2 * (5 - x)

theorem roots_of_p :
  ∀ x : ℝ, p x = 0 ↔ x = 0 ∨ x = -3 ∨ x = 5 := by
  sorry

end roots_of_p_l1233_123374


namespace tan_theta_minus_pi_fourth_l1233_123332

/-- Given that θ is in the fourth quadrant and sin(θ + π/4) = 5/13, 
    prove that tan(θ - π/4) = -12/5 -/
theorem tan_theta_minus_pi_fourth (θ : Real) 
  (h1 : π < θ ∧ θ < 2*π) -- θ is in the fourth quadrant
  (h2 : Real.sin (θ + π/4) = 5/13) : 
  Real.tan (θ - π/4) = -12/5 := by
sorry

end tan_theta_minus_pi_fourth_l1233_123332


namespace stone_162_is_12_l1233_123313

/-- The number of stones in the circular arrangement -/
def n : ℕ := 15

/-- The count we're interested in -/
def target_count : ℕ := 162

/-- The function that maps a count to its corresponding stone number -/
def stone_number (count : ℕ) : ℕ := 
  if count % n = 0 then n else count % n

theorem stone_162_is_12 : stone_number target_count = 12 := by
  sorry

end stone_162_is_12_l1233_123313


namespace polynomial_factorization_l1233_123302

theorem polynomial_factorization (x : ℝ) : 
  x^4 - 4*x^3 + 6*x^2 - 4*x + 1 = (x - 1)^4 := by
  sorry

end polynomial_factorization_l1233_123302


namespace not_necessary_not_sufficient_condition_l1233_123312

theorem not_necessary_not_sufficient_condition (a b : ℝ) : 
  ¬(((a ≠ 5 ∧ b ≠ -5) → (a + b ≠ 0)) ∧ ((a + b ≠ 0) → (a ≠ 5 ∧ b ≠ -5))) := by
  sorry

end not_necessary_not_sufficient_condition_l1233_123312


namespace remaining_budget_calculation_l1233_123305

def total_budget : ℝ := 80000000
def infrastructure_percentage : ℝ := 0.30
def public_transportation : ℝ := 10000000
def healthcare_percentage : ℝ := 0.15

theorem remaining_budget_calculation :
  total_budget - (infrastructure_percentage * total_budget + public_transportation + healthcare_percentage * total_budget) = 34000000 := by
  sorry

end remaining_budget_calculation_l1233_123305


namespace expression_value_l1233_123325

theorem expression_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^7 + y^7 + z^7) / (x*y*z*(x*y + x*z + y*z)) = -7 := by
  sorry

end expression_value_l1233_123325


namespace m_range_l1233_123380

def f (m : ℝ) (x : ℝ) := 2*x^2 - 2*(m-2)*x + 3*m - 1

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

def is_ellipse_with_foci_on_y_axis (m : ℝ) : Prop :=
  m + 1 > 0 ∧ 9 - m > m + 1

def prop_p (m : ℝ) : Prop := is_increasing (f m) 1 2

def prop_q (m : ℝ) : Prop := is_ellipse_with_foci_on_y_axis m

theorem m_range (m : ℝ) 
  (h1 : prop_p m ∨ prop_q m) 
  (h2 : ¬(prop_p m ∧ prop_q m)) 
  (h3 : ¬¬(prop_p m)) : 
  m ≤ -1 ∨ m = 4 := by sorry

end m_range_l1233_123380


namespace arithmetic_sequence_sum_remainder_l1233_123337

theorem arithmetic_sequence_sum_remainder (n : ℕ) (a d : ℤ) (h : n = 2013) (h1 : a = 105) (h2 : d = 35) :
  (n * (2 * a + (n - 1) * d) / 2) % 12 = 3 := by
  sorry

end arithmetic_sequence_sum_remainder_l1233_123337


namespace possible_x_values_l1233_123324

def A (x y : ℕ+) : ℕ := x^2 + y^2 + 2*x - 2*y + 2

def B (x : ℕ+) : ℤ := x^2 - 5*x + 5

theorem possible_x_values :
  ∀ x y : ℕ+, (B x)^(A x y) = 1 → x ∈ ({1, 2, 3, 4} : Set ℕ+) :=
sorry

end possible_x_values_l1233_123324


namespace probability_intersection_l1233_123336

theorem probability_intersection (A B : ℝ) (union : ℝ) (h1 : 0 ≤ A ∧ A ≤ 1) (h2 : 0 ≤ B ∧ B ≤ 1) (h3 : 0 ≤ union ∧ union ≤ 1) :
  ∃ intersection : ℝ, 0 ≤ intersection ∧ intersection ≤ 1 ∧ union = A + B - intersection :=
by sorry

end probability_intersection_l1233_123336


namespace reciprocal_of_repeating_decimal_one_third_l1233_123352

theorem reciprocal_of_repeating_decimal_one_third (x : ℚ) : 
  (∀ n : ℕ, (10 * x - x) * 10^n = 3 * 10^n - 3) → 
  (1 / x = 3) :=
by sorry

end reciprocal_of_repeating_decimal_one_third_l1233_123352


namespace optimal_carriages_and_passengers_l1233_123320

/-- The daily round trips as a function of the number of carriages -/
def daily_trips (x : ℕ) : ℝ :=
  -3 * x + 28

/-- The daily operating number of passengers as a function of the number of carriages -/
def daily_passengers (x : ℕ) : ℝ :=
  110 * x * daily_trips x

/-- The set of valid carriage numbers -/
def valid_carriages : Set ℕ :=
  {x | 1 ≤ x ∧ x ≤ 9}

theorem optimal_carriages_and_passengers :
  ∀ x ∈ valid_carriages,
    daily_passengers 5 ≥ daily_passengers x ∧
    daily_passengers 5 = 14300 :=
by sorry

end optimal_carriages_and_passengers_l1233_123320


namespace base_conversion_sum_l1233_123347

-- Define a function to convert a number from base 8 to base 10
def base8To10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

-- Define a function to convert a number from base 13 to base 10
def base13To10 (n : Nat) (c : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds * 13^2 + c * 13^1 + ones * 13^0

theorem base_conversion_sum :
  base8To10 537 + base13To10 405 12 = 1188 := by
  sorry

end base_conversion_sum_l1233_123347


namespace pascal_leibniz_relation_l1233_123378

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Element of the Leibniz triangle -/
def leibniz (n k : ℕ) : ℚ := 1 / ((n + 1 : ℚ) * (binomial n k))

/-- Theorem stating the relationship between Pascal's and Leibniz's triangles -/
theorem pascal_leibniz_relation (n k : ℕ) (h : k ≤ n) :
  leibniz n k = 1 / ((n + 1 : ℚ) * (binomial n k)) := by
  sorry

end pascal_leibniz_relation_l1233_123378


namespace student_calculation_l1233_123300

theorem student_calculation (chosen_number : ℕ) : 
  chosen_number = 40 → chosen_number * 7 - 150 = 130 := by
  sorry

end student_calculation_l1233_123300


namespace sum_seven_is_thirtyfive_l1233_123383

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  sum_property : a 2 + a 10 = 16
  eighth_term : a 8 = 11

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n * (seq.a 1 + seq.a n)) / 2

/-- The main theorem to prove -/
theorem sum_seven_is_thirtyfive (seq : ArithmeticSequence) : 
  sum_n seq 7 = 35 := by
  sorry

end sum_seven_is_thirtyfive_l1233_123383


namespace intersection_A_complement_B_for_m_3_find_m_for_given_intersection_l1233_123345

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x - 5 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Part 1
theorem intersection_A_complement_B_for_m_3 : 
  A ∩ (Set.univ \ B 3) = {x | x = -1 ∨ (3 ≤ x ∧ x ≤ 5)} := by sorry

-- Part 2
theorem find_m_for_given_intersection :
  ∃ m : ℝ, A ∩ B m = {x | -1 ≤ x ∧ x < 4} ∧ m = 8 := by sorry

end intersection_A_complement_B_for_m_3_find_m_for_given_intersection_l1233_123345


namespace max_consecutive_integers_sum_l1233_123326

theorem max_consecutive_integers_sum (k : ℕ) : k ≤ 81 ↔ ∃ n : ℕ, 2 * 3^8 = (k * (2 * n + k + 1)) / 2 := by sorry

end max_consecutive_integers_sum_l1233_123326


namespace opposite_numbers_abs_l1233_123355

theorem opposite_numbers_abs (m n : ℝ) : m + n = 0 → |m + n - 1| = 1 := by sorry

end opposite_numbers_abs_l1233_123355


namespace bridge_length_l1233_123365

/-- Given a train of length 150 meters traveling at 45 km/hr that crosses a bridge in 30 seconds,
    the length of the bridge is 225 meters. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 225 := by
  sorry

#check bridge_length

end bridge_length_l1233_123365


namespace floor_abs_negative_l1233_123344

theorem floor_abs_negative : ⌊|(-47.6:ℝ)|⌋ = 47 := by sorry

end floor_abs_negative_l1233_123344


namespace sandro_children_l1233_123391

/-- Calculates the total number of children for a person with a given number of sons
    and a ratio of daughters to sons. -/
def totalChildren (numSons : ℕ) (daughterToSonRatio : ℕ) : ℕ :=
  numSons + numSons * daughterToSonRatio

/-- Theorem stating that for a person with 3 sons and 6 times as many daughters as sons,
    the total number of children is 21. -/
theorem sandro_children :
  totalChildren 3 6 = 21 := by
  sorry

end sandro_children_l1233_123391


namespace base4_subtraction_l1233_123392

/-- Converts a natural number to its base 4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Subtracts two lists of digits in base 4 -/
def subtractBase4 (a b : List ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem base4_subtraction :
  let a := 207
  let b := 85
  let a_base4 := toBase4 a
  let b_base4 := toBase4 b
  let diff_base4 := subtractBase4 a_base4 b_base4
  fromBase4 diff_base4 = fromBase4 [1, 2, 3, 2] :=
by sorry

end base4_subtraction_l1233_123392


namespace mom_has_enough_money_l1233_123399

/-- Proves that the amount of money mom brought is sufficient to buy the discounted clothing item -/
theorem mom_has_enough_money (mom_money : ℝ) (original_price : ℝ) 
  (h1 : mom_money = 230)
  (h2 : original_price = 268)
  : mom_money ≥ 0.8 * original_price := by
  sorry

end mom_has_enough_money_l1233_123399


namespace weekly_caloric_deficit_l1233_123371

def monday_calories : ℕ := 2500
def tuesday_calories : ℕ := 2600
def wednesday_calories : ℕ := 2400
def thursday_calories : ℕ := 2700
def friday_calories : ℕ := 2300
def saturday_calories : ℕ := 3500
def sunday_calories : ℕ := 2400

def monday_exercise : ℕ := 1000
def tuesday_exercise : ℕ := 1200
def wednesday_exercise : ℕ := 1300
def thursday_exercise : ℕ := 1600
def friday_exercise : ℕ := 1000
def saturday_exercise : ℕ := 0
def sunday_exercise : ℕ := 1200

def total_weekly_calories : ℕ := monday_calories + tuesday_calories + wednesday_calories + thursday_calories + friday_calories + saturday_calories + sunday_calories

def total_weekly_net_calories : ℕ := 
  (monday_calories - monday_exercise) + 
  (tuesday_calories - tuesday_exercise) + 
  (wednesday_calories - wednesday_exercise) + 
  (thursday_calories - thursday_exercise) + 
  (friday_calories - friday_exercise) + 
  (saturday_calories - saturday_exercise) + 
  (sunday_calories - sunday_exercise)

theorem weekly_caloric_deficit : 
  total_weekly_calories - total_weekly_net_calories = 6800 := by
  sorry

end weekly_caloric_deficit_l1233_123371


namespace num_paths_A_to_B_l1233_123360

/-- Represents the number of red arrows from Point A -/
def num_red_arrows : ℕ := 3

/-- Represents the number of blue arrows connected to each red arrow -/
def blue_per_red : ℕ := 2

/-- Represents the number of green arrows connected to each blue arrow -/
def green_per_blue : ℕ := 2

/-- Represents the number of orange arrows connected to each green arrow -/
def orange_per_green : ℕ := 1

/-- Represents the number of ways to reach each blue arrow from a red arrow -/
def ways_to_blue : ℕ := 3

/-- Represents the number of ways to reach each green arrow from a blue arrow -/
def ways_to_green : ℕ := 4

/-- Represents the number of ways to reach each orange arrow from a green arrow -/
def ways_to_orange : ℕ := 5

/-- Theorem stating that the number of paths from A to B is 1440 -/
theorem num_paths_A_to_B : 
  num_red_arrows * blue_per_red * green_per_blue * orange_per_green * 
  ways_to_blue * ways_to_green * ways_to_orange = 1440 := by
  sorry

#check num_paths_A_to_B

end num_paths_A_to_B_l1233_123360


namespace binomial_expansion_ratio_l1233_123388

theorem binomial_expansion_ratio : 
  let n : ℕ := 10
  let k : ℕ := 5
  let a : ℕ := Nat.choose n k
  let b : ℤ := -Nat.choose n 3 * (-2)^3
  (b : ℚ) / a = -80 / 21 := by sorry

end binomial_expansion_ratio_l1233_123388


namespace sum_five_consecutive_squares_not_perfect_square_l1233_123357

theorem sum_five_consecutive_squares_not_perfect_square (n : ℤ) : 
  ∃ (k : ℤ), (5 * n^2 + 10) ≠ k^2 := by
sorry

end sum_five_consecutive_squares_not_perfect_square_l1233_123357


namespace robins_hair_length_l1233_123396

/-- Calculates the final hair length after growth and cutting -/
def finalHairLength (initial growth cut : ℝ) : ℝ :=
  initial + growth - cut

/-- Theorem stating that given the initial conditions, the final hair length is 2 inches -/
theorem robins_hair_length :
  finalHairLength 14 8 20 = 2 := by
  sorry

end robins_hair_length_l1233_123396


namespace quadratic_transformation_l1233_123384

theorem quadratic_transformation (x : ℝ) : 
  (4 * x^2 - 16 * x - 400 = 0) → 
  (∃ p q : ℝ, (x + p)^2 = q ∧ q = 104) :=
by sorry

end quadratic_transformation_l1233_123384


namespace fraction_evaluation_l1233_123339

theorem fraction_evaluation : (20 + 15) / (30 - 25) = 7 := by
  sorry

end fraction_evaluation_l1233_123339


namespace water_speed_swimming_problem_l1233_123386

/-- Proves that the speed of water is 2 km/h given the conditions of the swimming problem. -/
theorem water_speed_swimming_problem (still_water_speed : ℝ) (distance : ℝ) (time : ℝ) (water_speed : ℝ) :
  still_water_speed = 4 →
  distance = 12 →
  time = 6 →
  distance = (still_water_speed - water_speed) * time →
  water_speed = 2 := by
  sorry

end water_speed_swimming_problem_l1233_123386


namespace max_digit_sum_for_reciprocal_l1233_123316

theorem max_digit_sum_for_reciprocal (a b c z : ℕ) : 
  (a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9) →  -- a, b, and c are digits
  (100 * a + 10 * b + c = 1000 / z) →  -- 0.abc = 1/z
  (0 < z ∧ z ≤ 15) →  -- 0 < z ≤ 15
  (∀ w, (w ≤ 9 ∧ w ≤ 9 ∧ w ≤ 9) → 
        (100 * w + 10 * w + w = 1000 / z) → 
        (0 < z ∧ z ≤ 15) → 
        a + b + c ≥ w + w + w) →
  a + b + c = 8 :=
by sorry

end max_digit_sum_for_reciprocal_l1233_123316


namespace expression_evaluation_l1233_123301

theorem expression_evaluation (b : ℚ) (h : b = -3) :
  (3 * b⁻¹ + b⁻¹ / 3) / b = 10 / 27 := by
  sorry

end expression_evaluation_l1233_123301


namespace polynomial_simplification_l1233_123379

theorem polynomial_simplification (y : ℝ) : 
  (4 * y^10 + 6 * y^9 + 3 * y^8) + (2 * y^12 + 5 * y^10 + y^9 + y^7 + 4 * y^4 + 7 * y + 9) = 
  2 * y^12 + 9 * y^10 + 7 * y^9 + 3 * y^8 + y^7 + 4 * y^4 + 7 * y + 9 := by
sorry

end polynomial_simplification_l1233_123379


namespace defective_products_m1_l1233_123308

theorem defective_products_m1 (m1_production m2_production m3_production : ℝ)
  (m2_defective m3_defective : ℝ) (non_defective_total : ℝ) :
  m1_production = 25 ∧ 
  m2_production = 35 ∧ 
  m3_production = 40 ∧ 
  m2_defective = 4 ∧ 
  m3_defective = 5 ∧ 
  non_defective_total = 96.1 →
  (100 - non_defective_total - (m2_production * m2_defective / 100 + m3_production * m3_defective / 100)) / m1_production * 100 = 2 := by
  sorry

end defective_products_m1_l1233_123308


namespace bus_profit_analysis_l1233_123387

/-- Represents the daily profit of a bus company -/
def daily_profit (x : ℕ) : ℤ :=
  2 * x - 600

theorem bus_profit_analysis :
  (∀ x : ℕ, x ≥ 300 → daily_profit x ≥ 0) ∧
  (∀ x : ℕ, daily_profit x = 2 * x - 600) ∧
  (daily_profit 800 = 1000) :=
sorry

end bus_profit_analysis_l1233_123387


namespace partition_set_exists_l1233_123372

theorem partition_set_exists (n : ℕ) (h : n ≥ 3) :
  ∃ (S : Finset ℕ), 
    (Finset.card S = 2 * n) ∧ 
    (∀ m : ℕ, 2 ≤ m ∧ m ≤ n → 
      ∃ (A : Finset ℕ), 
        A ⊆ S ∧ 
        Finset.card A = m ∧ 
        (∃ (B : Finset ℕ), B = S \ A ∧ Finset.sum A id = Finset.sum B id)) :=
by sorry

end partition_set_exists_l1233_123372


namespace half_percent_is_point_zero_zero_five_l1233_123356

/-- Converts a percentage to its decimal representation -/
def percent_to_decimal (p : ℚ) : ℚ := p / 100

/-- States that 1/2 % is equal to 0.005 -/
theorem half_percent_is_point_zero_zero_five :
  percent_to_decimal (1/2) = 5/1000 := by sorry

end half_percent_is_point_zero_zero_five_l1233_123356


namespace arithmetic_sequence_nth_term_l1233_123319

theorem arithmetic_sequence_nth_term (a₁ a₂ aₙ n : ℤ) : 
  a₁ = 11 → a₂ = 8 → aₙ = -49 → 
  (∀ k : ℕ, k > 0 → a₁ + (k - 1) * (a₂ - a₁) = aₙ ↔ k = n) →
  n = 21 := by
sorry

end arithmetic_sequence_nth_term_l1233_123319


namespace total_cost_is_24_l1233_123362

/-- The cost of one gold ring in dollars -/
def ring_cost : ℕ := 12

/-- The number of index fingers a person has -/
def index_fingers : ℕ := 2

/-- The total cost of buying gold rings for all index fingers -/
def total_cost : ℕ := ring_cost * index_fingers

/-- Theorem: The total cost of buying gold rings for all index fingers is 24 dollars -/
theorem total_cost_is_24 : total_cost = 24 := by sorry

end total_cost_is_24_l1233_123362


namespace calvins_haircuts_l1233_123364

/-- The number of haircuts Calvin has gotten so far -/
def haircuts_gotten : ℕ := 8

/-- The number of additional haircuts Calvin needs to reach his goal -/
def haircuts_needed : ℕ := 2

/-- The percentage of progress Calvin has made towards his goal -/
def progress_percentage : ℚ := 80 / 100

theorem calvins_haircuts : 
  (haircuts_gotten : ℚ) / (haircuts_gotten + haircuts_needed) = progress_percentage := by
  sorry

end calvins_haircuts_l1233_123364


namespace min_bottles_to_fill_jumbo_l1233_123349

def jumbo_capacity : ℕ := 1200
def regular_capacity : ℕ := 75
def mini_capacity : ℕ := 50

theorem min_bottles_to_fill_jumbo :
  (jumbo_capacity / regular_capacity = 16 ∧ jumbo_capacity % regular_capacity = 0) ∧
  (jumbo_capacity / mini_capacity = 24 ∧ jumbo_capacity % mini_capacity = 0) :=
by sorry

end min_bottles_to_fill_jumbo_l1233_123349


namespace play_dough_quantity_l1233_123373

def lego_price : ℕ := 250
def sword_price : ℕ := 120
def dough_price : ℕ := 35
def lego_quantity : ℕ := 3
def sword_quantity : ℕ := 7
def total_paid : ℕ := 1940

theorem play_dough_quantity :
  (total_paid - (lego_price * lego_quantity + sword_price * sword_quantity)) / dough_price = 10 := by
  sorry

end play_dough_quantity_l1233_123373


namespace expand_expression_l1233_123329

theorem expand_expression (x : ℝ) : (5*x - 3) * (x^3 + 4*x) = 5*x^4 - 3*x^3 + 20*x^2 - 12*x := by
  sorry

end expand_expression_l1233_123329


namespace double_real_value_interest_rate_l1233_123385

/-- Proves that the given formula for the annual compound interest rate 
    results in doubling the real value of an initial sum after 22 years, 
    considering inflation and taxes. -/
theorem double_real_value_interest_rate 
  (X : ℝ) -- Annual inflation rate
  (Y : ℝ) -- Annual tax rate on earned interest
  (r : ℝ) -- Annual compound interest rate
  (h_X : X > 0)
  (h_Y : 0 ≤ Y ∧ Y < 1)
  (h_r : r = ((2 * (1 + X)) ^ (1 / 22) - 1) / (1 - Y)) :
  (∀ P : ℝ, P > 0 → 
    P * (1 + r * (1 - Y))^22 / (1 + X)^22 = 2 * P) :=
sorry

end double_real_value_interest_rate_l1233_123385


namespace stream_speed_l1233_123331

theorem stream_speed (boat_speed : ℝ) (stream_speed : ℝ) : 
  boat_speed = 39 →
  (1 / (boat_speed - stream_speed)) = (2 * (1 / (boat_speed + stream_speed))) →
  stream_speed = 13 := by
  sorry

end stream_speed_l1233_123331


namespace ten_candies_distribution_l1233_123367

/-- The number of ways to distribute n candies over days, with at least one candy per day -/
def candy_distribution (n : ℕ) : ℕ := 2^(n - 1)

/-- Theorem: The number of ways to distribute 10 candies over days, with at least one candy per day, is 512 -/
theorem ten_candies_distribution : candy_distribution 10 = 512 := by
  sorry

end ten_candies_distribution_l1233_123367


namespace decagon_diagonal_intersections_l1233_123315

/-- A regular decagon is a 10-sided polygon -/
def regular_decagon : ℕ := 10

/-- Number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def interior_intersection_points (n : ℕ) : ℕ := choose n 4

theorem decagon_diagonal_intersections :
  interior_intersection_points regular_decagon = 210 :=
sorry

end decagon_diagonal_intersections_l1233_123315


namespace unique_positive_solution_exists_distinct_real_solution_l1233_123322

-- Define the system of equations
def equation_system (x y z : ℝ) : Prop :=
  x * y + y * z + z * x = 12 ∧ x * y * z - x - y - z = 2

-- Theorem for unique positive solution
theorem unique_positive_solution :
  ∃! (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ equation_system x y z ∧ x = 2 ∧ y = 2 ∧ z = 2 :=
sorry

-- Theorem for existence of distinct real solution
theorem exists_distinct_real_solution :
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ equation_system x y z :=
sorry

end unique_positive_solution_exists_distinct_real_solution_l1233_123322


namespace square_difference_identity_l1233_123376

theorem square_difference_identity :
  287 * 287 + 269 * 269 - 2 * 287 * 269 = 324 := by
  sorry

end square_difference_identity_l1233_123376


namespace rectangle_area_rectangle_area_is_270_l1233_123368

theorem rectangle_area (square_area : ℝ) (rectangle_length : ℝ) : ℝ :=
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_breadth := (3 / 5) * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_is_270 :
  rectangle_area 2025 10 = 270 := by
  sorry

end rectangle_area_rectangle_area_is_270_l1233_123368


namespace production_growth_rate_l1233_123390

theorem production_growth_rate (initial_volume : ℝ) (final_volume : ℝ) (years : ℕ) (growth_rate : ℝ) : 
  initial_volume = 1000000 → 
  final_volume = 1210000 → 
  years = 2 →
  initial_volume * (1 + growth_rate) ^ years = final_volume →
  growth_rate = 0.1 := by
sorry

end production_growth_rate_l1233_123390
