import Mathlib

namespace NUMINAMATH_CALUDE_problem_statement_l3686_368694

theorem problem_statement (a b : ℝ) (h : Real.exp a + Real.exp b = 4) :
  a + b ≤ 2 * Real.log 2 ∧ Real.exp a + b ≤ 3 ∧ Real.exp (2 * a) + Real.exp (2 * b) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3686_368694


namespace NUMINAMATH_CALUDE_divisibility_by_17_and_32_l3686_368641

theorem divisibility_by_17_and_32 (n : ℕ) (hn : n > 0) :
  (∃ k : ℤ, 5 * 3^(4*n + 1) + 2^(6*n + 1) = 17 * k) ∧
  (∃ m : ℤ, 5^2 * 7^(2*n + 1) + 3^(4*n) = 32 * m) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_17_and_32_l3686_368641


namespace NUMINAMATH_CALUDE_simplify_nested_radicals_l3686_368646

theorem simplify_nested_radicals : 
  Real.sqrt (8 + 4 * Real.sqrt 3) + Real.sqrt (8 - 4 * Real.sqrt 3) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_radicals_l3686_368646


namespace NUMINAMATH_CALUDE_school_survey_result_l3686_368686

/-- Calculates the number of girls in a school based on stratified sampling -/
def girlsInSchool (totalStudents sampleSize girlsInSample : ℕ) : ℕ :=
  (girlsInSample * totalStudents) / sampleSize

/-- Theorem stating that given the problem conditions, the number of girls in the school is 760 -/
theorem school_survey_result :
  let totalStudents : ℕ := 1600
  let sampleSize : ℕ := 200
  let girlsInSample : ℕ := 95
  girlsInSchool totalStudents sampleSize girlsInSample = 760 := by
  sorry

end NUMINAMATH_CALUDE_school_survey_result_l3686_368686


namespace NUMINAMATH_CALUDE_fair_attendance_ratio_l3686_368687

theorem fair_attendance_ratio :
  let this_year : ℕ := 600
  let total_three_years : ℕ := 2800
  let next_year : ℕ := (total_three_years - this_year + 200) / 2
  let last_year : ℕ := next_year - 200
  (next_year : ℚ) / this_year = 2 :=
by sorry

end NUMINAMATH_CALUDE_fair_attendance_ratio_l3686_368687


namespace NUMINAMATH_CALUDE_shortest_path_l3686_368625

/-- Represents an elevator in the building --/
inductive Elevator
| A | B | C | D | E | F | G | H | I | J

/-- Represents a floor in the building --/
inductive Floor
| First | Second

/-- Represents a location on a floor --/
structure Location where
  floor : Floor
  x : ℕ
  y : ℕ

/-- Defines the building layout --/
def building_layout : List (Elevator × Location) := sorry

/-- Defines the entrance location --/
def entrance : Location := sorry

/-- Defines the exit location --/
def exit : Location := sorry

/-- Determines if an elevator leads to a confined room --/
def is_confined (e : Elevator) : Bool := sorry

/-- Calculates the distance between two locations --/
def distance (l1 l2 : Location) : ℕ := sorry

/-- Determines if a path is valid (uses only non-confined elevators) --/
def is_valid_path (path : List Elevator) : Bool := sorry

/-- Calculates the total distance of a path --/
def path_distance (path : List Elevator) : ℕ := sorry

/-- The theorem to be proved --/
theorem shortest_path :
  let path := [Elevator.B, Elevator.J, Elevator.G]
  is_valid_path path ∧
  (∀ other_path, is_valid_path other_path → path_distance path ≤ path_distance other_path) :=
sorry

end NUMINAMATH_CALUDE_shortest_path_l3686_368625


namespace NUMINAMATH_CALUDE_circle_radius_l3686_368682

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the axis of symmetry
def axis_of_symmetry (x : ℝ) : Prop := x = -1

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 3 = 0

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- State the theorem
theorem circle_radius :
  ∀ C : Circle,
  (C.center.1 = -1) →  -- Center is on the axis of symmetry
  (C.center.2 ≠ 0) →  -- Center is not on the x-axis
  (C.center.1 + C.radius)^2 + C.center.2^2 = C.radius^2 →  -- Circle passes through the focus
  (∃ (x y : ℝ), tangent_line x y ∧ 
    ((x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2)) →  -- Circle is tangent to the line
  C.radius = 14 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_l3686_368682


namespace NUMINAMATH_CALUDE_average_of_multiples_of_10_l3686_368603

def multiples_of_10 : List ℕ := List.range 30 |>.map (fun n => 10 * (n + 1))

theorem average_of_multiples_of_10 :
  (List.sum multiples_of_10) / (List.length multiples_of_10) = 155 := by
  sorry

#eval (List.sum multiples_of_10) / (List.length multiples_of_10)

end NUMINAMATH_CALUDE_average_of_multiples_of_10_l3686_368603


namespace NUMINAMATH_CALUDE_max_min_product_l3686_368672

theorem max_min_product (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 12 → 
  a * b + b * c + c * a = 30 → 
  (min (a * b) (min (b * c) (c * a))) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l3686_368672


namespace NUMINAMATH_CALUDE_jeff_cabinets_l3686_368627

/-- The total number of cabinets after Jeff's installation --/
def total_cabinets (initial : ℕ) (counters : ℕ) (extra : ℕ) : ℕ :=
  initial + counters * (2 * initial) + extra

/-- Proof that Jeff has 26 cabinets after the installation --/
theorem jeff_cabinets : total_cabinets 3 3 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_jeff_cabinets_l3686_368627


namespace NUMINAMATH_CALUDE_veronica_yellow_balls_l3686_368673

theorem veronica_yellow_balls :
  let total_balls : ℕ := 60
  let yellow_balls : ℕ := 27
  let brown_balls : ℕ := 33
  (yellow_balls : ℚ) / total_balls = 45 / 100 ∧
  brown_balls + yellow_balls = total_balls →
  yellow_balls = 27 :=
by sorry

end NUMINAMATH_CALUDE_veronica_yellow_balls_l3686_368673


namespace NUMINAMATH_CALUDE_ray_point_distance_product_l3686_368657

/-- Given two points on a ray from the origin, prove that the product of their distances
    equals the sum of the products of their coordinates. -/
theorem ray_point_distance_product (x₁ y₁ z₁ x₂ y₂ z₂ r₁ r₂ : ℝ) 
  (h₁ : r₁ = Real.sqrt (x₁^2 + y₁^2 + z₁^2))
  (h₂ : r₂ = Real.sqrt (x₂^2 + y₂^2 + z₂^2))
  (h_collinear : ∃ (t : ℝ), t > 0 ∧ x₂ = t * x₁ ∧ y₂ = t * y₁ ∧ z₂ = t * z₁) :
  r₁ * r₂ = x₁ * x₂ + y₁ * y₂ + z₁ * z₂ := by
  sorry

end NUMINAMATH_CALUDE_ray_point_distance_product_l3686_368657


namespace NUMINAMATH_CALUDE_no_real_m_for_equal_roots_l3686_368655

theorem no_real_m_for_equal_roots : 
  ¬∃ (m : ℝ), ∃ (x : ℝ), 
    (x * (x + 2) - (m + 3)) / ((x + 2) * (m + 2)) = x / m ∧
    ∀ (y : ℝ), (y * (y + 2) - (m + 3)) / ((y + 2) * (m + 2)) = y / m → y = x :=
by sorry

end NUMINAMATH_CALUDE_no_real_m_for_equal_roots_l3686_368655


namespace NUMINAMATH_CALUDE_exists_non_square_product_l3686_368684

theorem exists_non_square_product (a b : ℤ) (ha : a > 1) (hb : b > 1) (hab : a ≠ b) :
  ∃ n : ℕ+, ¬∃ m : ℤ, (a^n.val - 1) * (b^n.val - 1) = m^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_non_square_product_l3686_368684


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l3686_368675

def units_digit (n : ℤ) : ℕ := n.natAbs % 10

theorem units_digit_of_expression : units_digit (7 * 17 * 1977 - 7^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l3686_368675


namespace NUMINAMATH_CALUDE_jordan_oreos_l3686_368663

theorem jordan_oreos (jordan : ℕ) (james : ℕ) : 
  james = 2 * jordan + 3 →
  jordan + james = 36 →
  jordan = 11 := by
sorry

end NUMINAMATH_CALUDE_jordan_oreos_l3686_368663


namespace NUMINAMATH_CALUDE_knickknack_weight_is_six_l3686_368658

-- Define the given conditions
def bookcase_max_weight : ℝ := 80
def hardcover_count : ℕ := 70
def hardcover_weight : ℝ := 0.5
def textbook_count : ℕ := 30
def textbook_weight : ℝ := 2
def knickknack_count : ℕ := 3
def weight_over_limit : ℝ := 33

-- Define the total weight of the collection
def total_weight : ℝ := bookcase_max_weight + weight_over_limit

-- Define the weight of hardcover books and textbooks
def books_weight : ℝ := hardcover_count * hardcover_weight + textbook_count * textbook_weight

-- Define the total weight of knick-knacks
def knickknacks_total_weight : ℝ := total_weight - books_weight

-- Theorem to prove
theorem knickknack_weight_is_six :
  knickknacks_total_weight / knickknack_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_knickknack_weight_is_six_l3686_368658


namespace NUMINAMATH_CALUDE_farmer_goats_problem_l3686_368642

/-- Represents the number of additional goats needed to make half of the animals goats -/
def additional_goats (cows sheep initial_goats : ℕ) : ℕ :=
  let total := cows + sheep + initial_goats
  (total - 2 * initial_goats)

theorem farmer_goats_problem (cows sheep initial_goats : ℕ) 
  (h_cows : cows = 7)
  (h_sheep : sheep = 8)
  (h_initial_goats : initial_goats = 6) :
  additional_goats cows sheep initial_goats = 9 := by
  sorry

#eval additional_goats 7 8 6

end NUMINAMATH_CALUDE_farmer_goats_problem_l3686_368642


namespace NUMINAMATH_CALUDE_triangle_special_area_l3686_368629

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the angles form an arithmetic sequence and a, c, 4/√3 * b form a geometric sequence,
    then the area of the triangle is √3/2 * a² -/
theorem triangle_special_area (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = Real.pi →
  B = (A + C) / 2 →
  ∃ (q : ℝ), q > 0 ∧ c = q * a ∧ 4 / Real.sqrt 3 * b = q^2 * a →
  (1 / 2) * a * b * Real.sin C = Real.sqrt 3 / 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_area_l3686_368629


namespace NUMINAMATH_CALUDE_vector_equality_l3686_368691

def vector_a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def vector_b : Fin 2 → ℝ := ![1, -2]

theorem vector_equality (x : ℝ) : 
  ‖vector_a x + vector_b‖ = ‖vector_a x - vector_b‖ → x = 2 := by
sorry

end NUMINAMATH_CALUDE_vector_equality_l3686_368691


namespace NUMINAMATH_CALUDE_average_of_numbers_l3686_368665

theorem average_of_numbers : 
  let numbers : List ℕ := [744, 745, 747, 748, 749, 752, 752, 753, 755, 755]
  (numbers.sum / numbers.length : ℚ) = 700 := by
sorry

end NUMINAMATH_CALUDE_average_of_numbers_l3686_368665


namespace NUMINAMATH_CALUDE_degrees_to_radians_l3686_368652

theorem degrees_to_radians (π : Real) (h : π = 180) : 
  (240 : Real) * π / 180 = 4 * π / 3 := by sorry

end NUMINAMATH_CALUDE_degrees_to_radians_l3686_368652


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3686_368605

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 8) * (x - 6) = -50 + k * x) ↔ 
  (k = -10 + 2 * Real.sqrt 6 ∨ k = -10 - 2 * Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3686_368605


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l3686_368699

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 4 * x * y) :
  1 / x + 1 / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l3686_368699


namespace NUMINAMATH_CALUDE_david_crunches_l3686_368678

theorem david_crunches (zachary_crunches : ℕ) (david_less : ℕ) 
  (h1 : zachary_crunches = 17)
  (h2 : david_less = 13) : 
  zachary_crunches - david_less = 4 := by
  sorry

end NUMINAMATH_CALUDE_david_crunches_l3686_368678


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3686_368604

theorem nested_fraction_evaluation : 
  2 + (1 / (2 + (1 / (2 + 2)))) = 22 / 9 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3686_368604


namespace NUMINAMATH_CALUDE_article_price_calculation_l3686_368677

theorem article_price_calculation (P : ℝ) : 
  P * 0.75 * 0.85 * 1.10 * 1.05 = 1226.25 → P = 1843.75 := by
  sorry

end NUMINAMATH_CALUDE_article_price_calculation_l3686_368677


namespace NUMINAMATH_CALUDE_max_difference_when_a_is_one_sum_geq_three_iff_abs_a_geq_one_l3686_368620

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x - 2*a|
def g (a x : ℝ) : ℝ := |x + a|

-- Part 1
theorem max_difference_when_a_is_one :
  ∃ (m : ℝ), ∀ (x : ℝ), f 1 x - g 1 x ≤ m ∧ ∃ (y : ℝ), f 1 y - g 1 y = m ∧ m = 3 :=
sorry

-- Part 2
theorem sum_geq_three_iff_abs_a_geq_one (a : ℝ) :
  (∀ x : ℝ, f a x + g a x ≥ 3) ↔ |a| ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_max_difference_when_a_is_one_sum_geq_three_iff_abs_a_geq_one_l3686_368620


namespace NUMINAMATH_CALUDE_bells_toll_together_l3686_368679

theorem bells_toll_together (a b c d : ℕ) 
  (ha : a = 5) (hb : b = 8) (hc : c = 11) (hd : d = 15) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 1320 := by
  sorry

end NUMINAMATH_CALUDE_bells_toll_together_l3686_368679


namespace NUMINAMATH_CALUDE_xy_range_l3686_368644

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x + 2/x + 3*y + 4/y = 10) : 1 ≤ x*y ∧ x*y ≤ 8/3 := by
  sorry

end NUMINAMATH_CALUDE_xy_range_l3686_368644


namespace NUMINAMATH_CALUDE_even_function_decreasing_interval_l3686_368637

def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem even_function_decreasing_interval (k : ℝ) :
  (∀ x : ℝ, f k x = f k (-x)) →
  (∃ a : ℝ, ∀ x y : ℝ, x < y ∧ y ≤ 0 → f k x > f k y) ∧
  (∀ x y : ℝ, 0 < x ∧ x < y → f k x < f k y) :=
sorry

end NUMINAMATH_CALUDE_even_function_decreasing_interval_l3686_368637


namespace NUMINAMATH_CALUDE_optimal_profit_is_1368_l3686_368622

/-- Represents the types of apples -/
inductive AppleType
| A
| B
| C

/-- Represents the configuration of cars for each apple type -/
structure CarConfiguration where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- The total number of cars -/
def totalCars : ℕ := 40

/-- The total tons of apples -/
def totalTons : ℕ := 200

/-- Returns the tons per car for a given apple type -/
def tonsPerCar (t : AppleType) : ℕ :=
  match t with
  | AppleType.A => 6
  | AppleType.B => 5
  | AppleType.C => 4

/-- Returns the profit per ton for a given apple type -/
def profitPerTon (t : AppleType) : ℕ :=
  match t with
  | AppleType.A => 5
  | AppleType.B => 7
  | AppleType.C => 8

/-- Checks if a car configuration is valid -/
def isValidConfiguration (config : CarConfiguration) : Prop :=
  config.typeA + config.typeB + config.typeC = totalCars ∧
  config.typeA * tonsPerCar AppleType.A + 
  config.typeB * tonsPerCar AppleType.B + 
  config.typeC * tonsPerCar AppleType.C = totalTons ∧
  config.typeA ≥ 4 ∧ config.typeB ≥ 4 ∧ config.typeC ≥ 4

/-- Calculates the profit for a given car configuration -/
def calculateProfit (config : CarConfiguration) : ℕ :=
  config.typeA * tonsPerCar AppleType.A * profitPerTon AppleType.A +
  config.typeB * tonsPerCar AppleType.B * profitPerTon AppleType.B +
  config.typeC * tonsPerCar AppleType.C * profitPerTon AppleType.C

/-- The optimal car configuration -/
def optimalConfig : CarConfiguration :=
  { typeA := 4, typeB := 32, typeC := 4 }

theorem optimal_profit_is_1368 :
  isValidConfiguration optimalConfig ∧
  calculateProfit optimalConfig = 1368 ∧
  ∀ (config : CarConfiguration), 
    isValidConfiguration config → 
    calculateProfit config ≤ calculateProfit optimalConfig :=
by sorry

end NUMINAMATH_CALUDE_optimal_profit_is_1368_l3686_368622


namespace NUMINAMATH_CALUDE_xyz_equals_one_l3686_368666

theorem xyz_equals_one (x y z : ℝ) 
  (eq1 : x + 1/y = 4)
  (eq2 : y + 1/z = 1)
  (eq3 : z + 1/x = 7/3) :
  x * y * z = 1 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_one_l3686_368666


namespace NUMINAMATH_CALUDE_rotational_symmetry_180_l3686_368624

/-- Represents a 2D shape -/
structure Shape :=
  (points : Set (ℝ × ℝ))

/-- Represents a rotation of a shape -/
def rotate (s : Shape) (angle : ℝ) : Shape :=
  sorry

/-- Defines rotational symmetry for a shape -/
def is_rotationally_symmetric (s : Shape) (angle : ℝ) : Prop :=
  rotate s angle = s

/-- The original L-like shape -/
def original_shape : Shape :=
  sorry

/-- Theorem: The shape rotated 180 degrees is rotationally symmetric to the original shape -/
theorem rotational_symmetry_180 :
  is_rotationally_symmetric (rotate original_shape π) π :=
sorry

end NUMINAMATH_CALUDE_rotational_symmetry_180_l3686_368624


namespace NUMINAMATH_CALUDE_cube_rotation_different_face_l3686_368649

-- Define a cube face
inductive CubeFace
  | Top
  | Bottom
  | Front
  | Back
  | Left
  | Right

-- Define a cube position
structure CubePosition :=
  (location : ℝ × ℝ × ℝ)
  (bottom_face : CubeFace)

-- Define a cube rotation
inductive CubeRotation
  | RollForward
  | RollBackward
  | RollLeft
  | RollRight

-- Define a function that applies a rotation to a cube position
def apply_rotation (pos : CubePosition) (rot : CubeRotation) : CubePosition :=
  sorry

-- Define the theorem
theorem cube_rotation_different_face :
  ∃ (initial_pos final_pos : CubePosition) (rotations : List CubeRotation),
    (initial_pos.location = final_pos.location) ∧
    (initial_pos.bottom_face ≠ final_pos.bottom_face) ∧
    (final_pos = rotations.foldl apply_rotation initial_pos) :=
  sorry

end NUMINAMATH_CALUDE_cube_rotation_different_face_l3686_368649


namespace NUMINAMATH_CALUDE_train_length_l3686_368639

/-- Given a train that crosses a signal post in 40 seconds and takes 2 minutes to cross a 1.8 kilometer
    long bridge at constant speed, the length of the train is 900 meters. -/
theorem train_length (signal_time : ℝ) (bridge_time : ℝ) (bridge_length : ℝ) :
  signal_time = 40 →
  bridge_time = 120 →
  bridge_length = 1800 →
  ∃ (train_length : ℝ) (speed : ℝ),
    train_length = speed * signal_time ∧
    train_length + bridge_length = speed * bridge_time ∧
    train_length = 900 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3686_368639


namespace NUMINAMATH_CALUDE_marble_selection_probability_l3686_368631

def total_marbles : ℕ := 3 + 2 + 2
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 2
def green_marbles : ℕ := 2
def selected_marbles : ℕ := 4

theorem marble_selection_probability :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1) /
  (Nat.choose total_marbles selected_marbles) = 12 / 35 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_probability_l3686_368631


namespace NUMINAMATH_CALUDE_yamaimo_moving_problem_l3686_368618

/-- The Yamaimo family's moving problem -/
theorem yamaimo_moving_problem (initial_weight : ℝ) (initial_book_percentage : ℝ) 
  (final_book_percentage : ℝ) (new_weight : ℝ) : 
  initial_weight = 100 →
  initial_book_percentage = 99 / 100 →
  final_book_percentage = 95 / 100 →
  initial_weight * initial_book_percentage = 
    new_weight * final_book_percentage →
  initial_weight * (1 - initial_book_percentage) = 
    new_weight * (1 - final_book_percentage) →
  new_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_yamaimo_moving_problem_l3686_368618


namespace NUMINAMATH_CALUDE_rectangle_perimeters_l3686_368656

/-- The perimeter of a rectangle given its width and height. -/
def rectanglePerimeter (width height : ℝ) : ℝ := 2 * (width + height)

/-- The theorem stating the perimeters of the three rectangles formed from four photographs. -/
theorem rectangle_perimeters (photo_perimeter : ℝ) 
  (h1 : photo_perimeter = 20)
  (h2 : ∃ (w h : ℝ), rectanglePerimeter w h = photo_perimeter ∧ 
                      rectanglePerimeter (2*w) (2*h) = 40 ∧
                      rectanglePerimeter (4*w) h = 44 ∧
                      rectanglePerimeter w (4*h) = 56) :
  ∃ (p1 p2 p3 : ℝ), p1 = 40 ∧ p2 = 44 ∧ p3 = 56 ∧
    (p1 = 40 ∨ p1 = 44 ∨ p1 = 56) ∧
    (p2 = 40 ∨ p2 = 44 ∨ p2 = 56) ∧
    (p3 = 40 ∨ p3 = 44 ∨ p3 = 56) ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeters_l3686_368656


namespace NUMINAMATH_CALUDE_divisibility_of_linear_combination_l3686_368668

theorem divisibility_of_linear_combination (a b c : ℕ+) : 
  ∃ (r s : ℕ+), (Nat.gcd r s = 1) ∧ (∃ k : ℤ, (a : ℤ) * (r : ℤ) + (b : ℤ) * (s : ℤ) = k * (c : ℤ)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_linear_combination_l3686_368668


namespace NUMINAMATH_CALUDE_prob_two_even_correct_l3686_368602

/-- The total number of balls -/
def total_balls : ℕ := 17

/-- The number of even-numbered balls -/
def even_balls : ℕ := 8

/-- The probability of drawing two even-numbered balls without replacement -/
def prob_two_even : ℚ := 7 / 34

theorem prob_two_even_correct :
  (even_balls : ℚ) / total_balls * (even_balls - 1) / (total_balls - 1) = prob_two_even := by
  sorry

end NUMINAMATH_CALUDE_prob_two_even_correct_l3686_368602


namespace NUMINAMATH_CALUDE_rope_around_cylinders_l3686_368695

theorem rope_around_cylinders (rope_length : ℝ) (r1 r2 : ℝ) (rounds1 : ℕ) :
  r1 = 14 →
  r2 = 20 →
  rounds1 = 70 →
  rope_length = 2 * π * r1 * (rounds1 : ℝ) →
  ∃ (rounds2 : ℕ), rounds2 = 49 ∧ rope_length = 2 * π * r2 * (rounds2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_rope_around_cylinders_l3686_368695


namespace NUMINAMATH_CALUDE_chessboard_rearrangement_l3686_368667

def knight_distance (a b : ℕ × ℕ) : ℕ :=
  sorry

def king_distance (a b : ℕ × ℕ) : ℕ :=
  sorry

def is_valid_rearrangement (N : ℕ) (f : (ℕ × ℕ) → (ℕ × ℕ)) : Prop :=
  ∀ a b : ℕ × ℕ, a.1 < N ∧ a.2 < N ∧ b.1 < N ∧ b.2 < N →
    knight_distance a b = 1 → king_distance (f a) (f b) = 1

theorem chessboard_rearrangement :
  (∃ f : (ℕ × ℕ) → (ℕ × ℕ), is_valid_rearrangement 3 f) ∧
  (¬ ∃ f : (ℕ × ℕ) → (ℕ × ℕ), is_valid_rearrangement 8 f) :=
sorry

end NUMINAMATH_CALUDE_chessboard_rearrangement_l3686_368667


namespace NUMINAMATH_CALUDE_student_arrangement_equality_l3686_368609

/-- The number of ways to arrange k items out of n items -/
def arrange (n k : ℕ) : ℕ := sorry

theorem student_arrangement_equality (n : ℕ) :
  arrange (2*n) (2*n) = arrange (2*n) n * arrange n n := by sorry

end NUMINAMATH_CALUDE_student_arrangement_equality_l3686_368609


namespace NUMINAMATH_CALUDE_money_problem_l3686_368651

theorem money_problem (a b : ℝ) 
  (h1 : 4 * a + b = 68)
  (h2 : 2 * a - b < 16)
  (h3 : a + b > 22) :
  a < 14 ∧ b > 12 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l3686_368651


namespace NUMINAMATH_CALUDE_study_method_is_algorithm_statements_are_not_algorithms_l3686_368693

/-- Represents a series of steps or instructions -/
structure Procedure where
  steps : List String

/-- Represents a statement or fact -/
structure Statement where
  content : String

/-- Definition of an algorithm -/
def is_algorithm (p : Procedure) : Prop :=
  p.steps.length > 0 ∧ ∀ s ∈ p.steps, s ≠ ""

theorem study_method_is_algorithm (study_method : Procedure)
  (h1 : study_method.steps = ["Preview before class", 
                              "Listen carefully and take good notes during class", 
                              "Review first and then do homework after class", 
                              "Do appropriate exercises"]) : 
  is_algorithm study_method := by sorry

theorem statements_are_not_algorithms (s : Statement) : 
  ¬ is_algorithm ⟨[s.content]⟩ := by sorry

#check study_method_is_algorithm
#check statements_are_not_algorithms

end NUMINAMATH_CALUDE_study_method_is_algorithm_statements_are_not_algorithms_l3686_368693


namespace NUMINAMATH_CALUDE_complex_number_location_l3686_368696

theorem complex_number_location :
  ∀ z : ℂ, z / (1 + Complex.I) = Complex.I →
  (z.re < 0 ∧ z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3686_368696


namespace NUMINAMATH_CALUDE_inequality_proof_equality_conditions_l3686_368635

theorem inequality_proof (x y : ℝ) (h1 : x ≥ y) (h2 : y ≥ 1) :
  (x / Real.sqrt (x + y)) + (y / Real.sqrt (y + 1)) + (1 / Real.sqrt (x + 1)) ≥
  (y / Real.sqrt (x + y)) + (x / Real.sqrt (x + 1)) + (1 / Real.sqrt (y + 1)) :=
sorry

theorem equality_conditions (x y : ℝ) (h1 : x ≥ y) (h2 : y ≥ 1) :
  (x / Real.sqrt (x + y)) + (y / Real.sqrt (y + 1)) + (1 / Real.sqrt (x + 1)) =
  (y / Real.sqrt (x + y)) + (x / Real.sqrt (x + 1)) + (1 / Real.sqrt (y + 1)) ↔
  (x = y ∨ x = 1 ∨ y = 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_conditions_l3686_368635


namespace NUMINAMATH_CALUDE_count_four_digit_divisible_by_15_eq_600_l3686_368640

/-- The count of positive four-digit integers divisible by 15 -/
def count_four_digit_divisible_by_15 : ℕ :=
  (Finset.filter (fun n => n % 15 = 0) (Finset.range 9000)).card

/-- Theorem stating that the count of positive four-digit integers divisible by 15 is 600 -/
theorem count_four_digit_divisible_by_15_eq_600 :
  count_four_digit_divisible_by_15 = 600 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_divisible_by_15_eq_600_l3686_368640


namespace NUMINAMATH_CALUDE_octagon_sectors_area_l3686_368638

/-- The area of the region inside a regular octagon with side length 8 but outside
    the circular sectors with radius 4 centered at each vertex. -/
theorem octagon_sectors_area : 
  let side_length : ℝ := 8
  let sector_radius : ℝ := 4
  let octagon_area : ℝ := 8 * (1/2 * side_length^2 * Real.sqrt ((1 - Real.sqrt 2 / 2) / 2))
  let sectors_area : ℝ := 8 * (π * sector_radius^2 / 8)
  octagon_area - sectors_area = 256 * Real.sqrt ((1 - Real.sqrt 2 / 2) / 2) - 16 * π :=
by sorry

end NUMINAMATH_CALUDE_octagon_sectors_area_l3686_368638


namespace NUMINAMATH_CALUDE_fraction_problem_l3686_368680

theorem fraction_problem (a b c d e f : ℚ) :
  (∃ (k : ℚ), a = k * 1 ∧ b = k * 2 ∧ c = k * 5) →
  (∃ (m : ℚ), d = m * 1 ∧ e = m * 3 ∧ f = m * 7) →
  (a / d + b / e + c / f) / 3 = 200 / 441 →
  a / d = 4 / 7 ∧ b / e = 8 / 21 ∧ c / f = 20 / 49 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l3686_368680


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3686_368630

theorem inequality_equivalence (x : ℝ) : 3 * x^2 - 5 * x > 9 ↔ x < -1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3686_368630


namespace NUMINAMATH_CALUDE_inequality_proof_l3686_368650

theorem inequality_proof (x y : ℝ) (h1 : x ≥ y) (h2 : y ≥ 1) :
  (x / Real.sqrt (x + y)) + (y / Real.sqrt (y + 1)) + (1 / Real.sqrt (x + 1)) ≥ 
  (y / Real.sqrt (x + y)) + (x / Real.sqrt (x + 1)) + (1 / Real.sqrt (y + 1)) ∧
  ((x = y ∨ y = 1) ↔ 
    (x / Real.sqrt (x + y)) + (y / Real.sqrt (y + 1)) + (1 / Real.sqrt (x + 1)) = 
    (y / Real.sqrt (x + y)) + (x / Real.sqrt (x + 1)) + (1 / Real.sqrt (y + 1))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3686_368650


namespace NUMINAMATH_CALUDE_three_distinct_zeroes_l3686_368647

/-- The piecewise function f(x) as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then |2^x - 1| else 3 / (x - 1)

/-- The theorem stating the condition for three distinct zeroes -/
theorem three_distinct_zeroes (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = a ∧ f y = a ∧ f z = a) ↔ 0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_three_distinct_zeroes_l3686_368647


namespace NUMINAMATH_CALUDE_min_sum_nested_sqrt_l3686_368643

theorem min_sum_nested_sqrt (a b c : ℕ+) (k : ℕ) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a : ℝ) * Real.sqrt ((b : ℝ) * Real.sqrt (c : ℝ)) = (k : ℝ)^2 →
  (∀ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z →
    (x : ℝ) * Real.sqrt ((y : ℝ) * Real.sqrt (z : ℝ)) = (k : ℝ)^2 →
    (x : ℕ) + (y : ℕ) + (z : ℕ) ≥ (a : ℕ) + (b : ℕ) + (c : ℕ)) →
  (a : ℕ) + (b : ℕ) + (c : ℕ) = 7 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_nested_sqrt_l3686_368643


namespace NUMINAMATH_CALUDE_gamma_interval_for_f_l3686_368616

def f (x : ℝ) : ℝ := -x * abs x + 2 * x

def is_gamma_interval (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  ∀ y ∈ Set.Icc (1 / n) (1 / m), ∃ x ∈ Set.Icc m n, f x = y

theorem gamma_interval_for_f :
  let m : ℝ := 1
  let n : ℝ := (1 + Real.sqrt 5) / 2
  m < n ∧ 
  Set.Icc m n ⊆ Set.Ioi 1 ∧ 
  is_gamma_interval f m n := by sorry

end NUMINAMATH_CALUDE_gamma_interval_for_f_l3686_368616


namespace NUMINAMATH_CALUDE_f_half_equals_half_l3686_368653

-- Define the function f
def f : ℝ → ℝ := fun x ↦ 1 - 2 * x^2

-- State the theorem
theorem f_half_equals_half :
  (∀ x : ℝ, f (Real.sin x) = Real.cos (2 * x)) →
  f (1/2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_half_equals_half_l3686_368653


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l3686_368674

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents the possible cuts of the plywood -/
inductive PlywoodCut
  | Vertical
  | Horizontal
  | Mixed

theorem plywood_cut_perimeter_difference :
  let plywood : Rectangle := { width := 6, height := 9 }
  let possible_cuts : List PlywoodCut := [PlywoodCut.Vertical, PlywoodCut.Horizontal, PlywoodCut.Mixed]
  let cut_rectangles : PlywoodCut → Rectangle
    | PlywoodCut.Vertical => { width := 1, height := 9 }
    | PlywoodCut.Horizontal => { width := 1, height := 6 }
    | PlywoodCut.Mixed => { width := 2, height := 3 }
  let perimeters : List ℝ := possible_cuts.map (fun cut => perimeter (cut_rectangles cut))
  (∃ (max_perimeter min_perimeter : ℝ),
    max_perimeter ∈ perimeters ∧
    min_perimeter ∈ perimeters ∧
    max_perimeter = perimeters.maximum ∧
    min_perimeter = perimeters.minimum ∧
    max_perimeter - min_perimeter = 10) := by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l3686_368674


namespace NUMINAMATH_CALUDE_prob_wind_given_rain_l3686_368608

theorem prob_wind_given_rain (prob_rain prob_wind_and_rain : ℚ) 
  (h1 : prob_rain = 4/15)
  (h2 : prob_wind_and_rain = 1/10) :
  prob_wind_and_rain / prob_rain = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_wind_given_rain_l3686_368608


namespace NUMINAMATH_CALUDE_min_value_of_z_l3686_368662

theorem min_value_of_z (x y : ℝ) : 
  ∃ (m : ℝ), m = 8 ∧ ∀ (x y : ℝ), 3*x^2 + 4*y^2 + 12*x - 8*y + 3*x*y + 30 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_z_l3686_368662


namespace NUMINAMATH_CALUDE_students_liking_both_sports_l3686_368681

/-- Given a class of students with information about their sports preferences,
    prove the number of students who like both basketball and table tennis. -/
theorem students_liking_both_sports
  (total : ℕ)
  (basketball : ℕ)
  (table_tennis : ℕ)
  (neither : ℕ)
  (h1 : total = 40)
  (h2 : basketball = 20)
  (h3 : table_tennis = 15)
  (h4 : neither = 8)
  : ∃ x : ℕ, x = 3 ∧ basketball + table_tennis - x + neither = total :=
by sorry

end NUMINAMATH_CALUDE_students_liking_both_sports_l3686_368681


namespace NUMINAMATH_CALUDE_counterexample_exists_l3686_368614

theorem counterexample_exists : ∃ x : ℝ, x > 1 ∧ x + 1 / (x - 1) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3686_368614


namespace NUMINAMATH_CALUDE_alpha_value_l3686_368661

theorem alpha_value (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^4 - 1) = 5 * Complex.abs (α - 1))
  (h4 : Real.cos (Complex.arg α) = 1/2) :
  α = (-1 + Real.sqrt 33) / 4 + Complex.I * Real.sqrt (3 * ((-1 + Real.sqrt 33) / 4)^2) ∨
  α = (-1 - Real.sqrt 33) / 4 + Complex.I * Real.sqrt (3 * ((-1 - Real.sqrt 33) / 4)^2) :=
by sorry

end NUMINAMATH_CALUDE_alpha_value_l3686_368661


namespace NUMINAMATH_CALUDE_valid_numeral_count_l3686_368645

def is_single_digit_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def count_valid_numerals : ℕ :=
  let three_digit_count := 4 * 10 * 4
  let four_digit_count := 4 * 10 * 10 * 4
  three_digit_count + four_digit_count

theorem valid_numeral_count :
  count_valid_numerals = 1760 :=
sorry

end NUMINAMATH_CALUDE_valid_numeral_count_l3686_368645


namespace NUMINAMATH_CALUDE_fiftycentchange_l3686_368659

/-- Represents the different types of U.S. coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Represents a combination of coins --/
def CoinCombination := List Coin

/-- Checks if a coin combination is valid for 50 cents --/
def isValidCombination (c : CoinCombination) : Bool := sorry

/-- Counts the number of quarters in a combination --/
def countQuarters (c : CoinCombination) : Nat := sorry

/-- Generates all valid coin combinations for 50 cents --/
def allCombinations : List CoinCombination := sorry

/-- The main theorem stating that there are 47 ways to make change for 50 cents --/
theorem fiftycentchange : 
  (allCombinations.filter (fun c => isValidCombination c ∧ countQuarters c ≤ 1)).length = 47 := by
  sorry

end NUMINAMATH_CALUDE_fiftycentchange_l3686_368659


namespace NUMINAMATH_CALUDE_one_less_than_negative_one_l3686_368660

theorem one_less_than_negative_one : (-1 : ℤ) - 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_one_less_than_negative_one_l3686_368660


namespace NUMINAMATH_CALUDE_marcos_strawberries_weight_l3686_368636

theorem marcos_strawberries_weight (total_weight dad_weight : ℝ) 
  (h1 : total_weight = 20)
  (h2 : dad_weight = 17) :
  total_weight - dad_weight = 3 := by
sorry

end NUMINAMATH_CALUDE_marcos_strawberries_weight_l3686_368636


namespace NUMINAMATH_CALUDE_tangent_line_range_l3686_368669

/-- Given k > 0, if a line can always be drawn through the point (3, 1) to be tangent
    to the circle (x-2k)^2 + (y-k)^2 = k, then k ∈ (0, 1) ∪ (2, +∞) -/
theorem tangent_line_range (k : ℝ) (h_pos : k > 0) 
  (h_tangent : ∀ (x y : ℝ), (x - 2*k)^2 + (y - k)^2 = k → 
    ∃ (m b : ℝ), y = m*x + b ∧ (3 - 2*k)^2 + (1 - k)^2 ≥ k) :
  k ∈ Set.Ioo 0 1 ∪ Set.Ioi 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_range_l3686_368669


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3686_368601

/-- The ratio of the side length of a regular pentagon to the width of a rectangle with the same perimeter -/
theorem pentagon_rectangle_ratio : 
  ∀ (pentagon_side rectangle_width : ℝ),
  pentagon_side > 0 → 
  rectangle_width > 0 →
  5 * pentagon_side = 20 →
  6 * rectangle_width = 20 →
  pentagon_side / rectangle_width = 6 / 5 := by
sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l3686_368601


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_square_roots_l3686_368611

theorem max_value_of_sum_of_square_roots (a b c : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c)
  (sum_condition : a + b + c = 1) :
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) ≤ Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_square_roots_l3686_368611


namespace NUMINAMATH_CALUDE_product_sum_and_reciprocals_geq_nine_l3686_368690

theorem product_sum_and_reciprocals_geq_nine (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_and_reciprocals_geq_nine_l3686_368690


namespace NUMINAMATH_CALUDE_end_of_year_deposits_l3686_368617

/-- Accumulated capital for end-of-year deposits given beginning-of-year deposits -/
theorem end_of_year_deposits (P r : ℝ) (n : ℕ) (K : ℝ) :
  P > 0 → r > 0 → n > 0 →
  K = P * ((1 + r/100)^n - 1) / (r/100) * (1 + r/100) →
  ∃ K', K' = P * ((1 + r/100)^n - 1) / (r/100) ∧ K' = K / (1 + r/100) := by
  sorry

end NUMINAMATH_CALUDE_end_of_year_deposits_l3686_368617


namespace NUMINAMATH_CALUDE_shuai_shuai_memorization_l3686_368626

/-- The number of words memorized by Shuai Shuai over 7 days -/
def total_words : ℕ := 198

/-- The number of words memorized in the first 3 days -/
def first_three_days : ℕ := 44

/-- The number of words memorized on the fourth day -/
def fourth_day : ℕ := 10

/-- The number of words memorized in the last 3 days -/
def last_three_days : ℕ := 45

/-- Theorem stating the conditions and the result -/
theorem shuai_shuai_memorization :
  (first_three_days + fourth_day + last_three_days = total_words) ∧
  (first_three_days = (4 : ℚ) / 5 * (fourth_day + last_three_days)) ∧
  (first_three_days + fourth_day = (6 : ℚ) / 5 * last_three_days) ∧
  (total_words > 100) ∧
  (total_words < 200) :=
by sorry

end NUMINAMATH_CALUDE_shuai_shuai_memorization_l3686_368626


namespace NUMINAMATH_CALUDE_total_book_price_l3686_368634

theorem total_book_price (total_books : ℕ) (math_books : ℕ) (math_price : ℕ) (history_price : ℕ) :
  total_books = 90 →
  math_books = 54 →
  math_price = 4 →
  history_price = 5 →
  (math_books * math_price + (total_books - math_books) * history_price) = 396 :=
by sorry

end NUMINAMATH_CALUDE_total_book_price_l3686_368634


namespace NUMINAMATH_CALUDE_parallelogram_area_l3686_368670

-- Define the lines
def L1 (x y : ℝ) : Prop := y = 2
def L2 (x y : ℝ) : Prop := y = -2
def L3 (x y : ℝ) : Prop := 4 * x + 7 * y - 10 = 0
def L4 (x y : ℝ) : Prop := 4 * x + 7 * y + 20 = 0

-- Define the vertices of the parallelogram
def A : ℝ × ℝ := (-1.5, -2)
def B : ℝ × ℝ := (6, -2)
def C : ℝ × ℝ := (-1, 2)
def D : ℝ × ℝ := (-8.5, 2)

-- State the theorem
theorem parallelogram_area : 
  (A.1 = -1.5 ∧ A.2 = -2) →
  (B.1 = 6 ∧ B.2 = -2) →
  (C.1 = -1 ∧ C.2 = 2) →
  (D.1 = -8.5 ∧ D.2 = 2) →
  L1 C.1 C.2 →
  L1 D.1 D.2 →
  L2 A.1 A.2 →
  L2 B.1 B.2 →
  L3 A.1 A.2 →
  L3 C.1 C.2 →
  L4 B.1 B.2 →
  L4 D.1 D.2 →
  (B.1 - A.1) * (C.2 - A.2) = 30 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3686_368670


namespace NUMINAMATH_CALUDE_difference_of_squares_l3686_368632

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3686_368632


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l3686_368683

theorem ratio_of_sum_to_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (h : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l3686_368683


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l3686_368654

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  x * (x - 4) = 2 * x - 8 ↔ x = 4 ∨ x = 2 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  (2 * x) / (2 * x - 3) - 4 / (2 * x + 3) = 1 ↔ x = 10.5 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solution_l3686_368654


namespace NUMINAMATH_CALUDE_simplify_expression_l3686_368698

theorem simplify_expression : (0.3 * 0.2) / (0.4 * 0.5) - (0.1 * 0.6) = 0.24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3686_368698


namespace NUMINAMATH_CALUDE_mabel_handled_90_transactions_l3686_368628

-- Define the number of transactions for each person
def mabel_transactions : ℕ := 90
def anthony_transactions : ℕ := (110 * mabel_transactions) / 100
def cal_transactions : ℕ := (2 * anthony_transactions) / 3
def jade_transactions : ℕ := 80

-- State the theorem
theorem mabel_handled_90_transactions :
  mabel_transactions = 90 ∧
  anthony_transactions = (110 * mabel_transactions) / 100 ∧
  cal_transactions = (2 * anthony_transactions) / 3 ∧
  jade_transactions = cal_transactions + 14 ∧
  jade_transactions = 80 := by
  sorry


end NUMINAMATH_CALUDE_mabel_handled_90_transactions_l3686_368628


namespace NUMINAMATH_CALUDE_donut_distribution_theorem_l3686_368676

/-- A structure representing the donut distribution problem -/
structure DonutDistribution where
  num_boxes : ℕ
  total_donuts : ℕ
  num_flavors : ℕ

/-- The result of the donut distribution -/
structure DistributionResult where
  extra_donuts : ℕ
  donuts_per_flavor_per_box : ℕ

/-- Function to calculate the distribution result -/
def calculate_distribution (d : DonutDistribution) : DistributionResult :=
  { extra_donuts := d.total_donuts % d.num_boxes,
    donuts_per_flavor_per_box := (d.total_donuts / d.num_flavors) / d.num_boxes }

/-- Theorem stating the correct distribution for the given problem -/
theorem donut_distribution_theorem (d : DonutDistribution) 
  (h1 : d.num_boxes = 12)
  (h2 : d.total_donuts = 125)
  (h3 : d.num_flavors = 5) :
  let result := calculate_distribution d
  result.extra_donuts = 5 ∧ result.donuts_per_flavor_per_box = 2 := by
  sorry

#check donut_distribution_theorem

end NUMINAMATH_CALUDE_donut_distribution_theorem_l3686_368676


namespace NUMINAMATH_CALUDE_wall_volume_is_86436_l3686_368606

def wall_volume (width : ℝ) : ℝ :=
  let height := 6 * width
  let length := 7 * height
  width * height * length

theorem wall_volume_is_86436 :
  wall_volume 7 = 86436 :=
by sorry

end NUMINAMATH_CALUDE_wall_volume_is_86436_l3686_368606


namespace NUMINAMATH_CALUDE_solution_of_system_l3686_368633

theorem solution_of_system (x y : ℝ) : 
  (x^2 - x*y + y^2 = 49*(x - y) ∧ x^2 + x*y + y^2 = 76*(x + y)) ↔ 
  ((x = 0 ∧ y = 0) ∨ (x = 40 ∧ y = -24)) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l3686_368633


namespace NUMINAMATH_CALUDE_monthly_savings_proof_l3686_368685

-- Define income tax rate
def income_tax_rate : ℚ := 13/100

-- Define salaries and pensions (in rubles)
def ivan_salary : ℕ := 55000
def vasilisa_salary : ℕ := 45000
def mother_salary : ℕ := 18000
def mother_pension : ℕ := 10000
def father_salary : ℕ := 20000
def son_state_scholarship : ℕ := 3000
def son_nonstate_scholarship : ℕ := 15000

-- Define monthly expenses (in rubles)
def monthly_expenses : ℕ := 74000

-- Function to calculate net income after tax
def net_income (gross_income : ℕ) : ℚ :=
  (gross_income : ℚ) * (1 - income_tax_rate)

-- Total monthly net income before 01.05.2018
def total_net_income_before_may : ℚ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  net_income mother_salary + net_income father_salary + son_state_scholarship

-- Total monthly net income from 01.05.2018 to 31.08.2018
def total_net_income_may_to_aug : ℚ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  mother_pension + net_income father_salary + son_state_scholarship

-- Total monthly net income from 01.09.2018 for 1 year
def total_net_income_from_sep : ℚ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  mother_pension + net_income father_salary + 
  son_state_scholarship + net_income son_nonstate_scholarship

-- Theorem to prove monthly savings for different periods
theorem monthly_savings_proof :
  (total_net_income_before_may - monthly_expenses = 49060) ∧
  (total_net_income_may_to_aug - monthly_expenses = 43400) ∧
  (total_net_income_from_sep - monthly_expenses = 56450) := by
  sorry


end NUMINAMATH_CALUDE_monthly_savings_proof_l3686_368685


namespace NUMINAMATH_CALUDE_rita_trust_fund_growth_l3686_368613

/-- Calculates the final amount in a trust fund after compound interest -/
def trustFundGrowth (initialInvestment : ℝ) (interestRate : ℝ) (years : ℕ) : ℝ :=
  initialInvestment * (1 + interestRate) ^ years

/-- Theorem stating the approximate value of Rita's trust fund after 25 years -/
theorem rita_trust_fund_growth :
  let initialInvestment : ℝ := 5000
  let interestRate : ℝ := 0.03
  let years : ℕ := 25
  let finalAmount := trustFundGrowth initialInvestment interestRate years
  ∃ ε > 0, abs (finalAmount - 10468.87) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_rita_trust_fund_growth_l3686_368613


namespace NUMINAMATH_CALUDE_equation_solution_l3686_368610

theorem equation_solution (x : ℚ) : 
  (1 : ℚ) / 3 + 1 / x = (3 : ℚ) / 4 → x = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3686_368610


namespace NUMINAMATH_CALUDE_sum_reciprocals_equals_one_plus_reciprocal_product_l3686_368671

theorem sum_reciprocals_equals_one_plus_reciprocal_product 
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y + 1) : 
  1 / x + 1 / y = 1 + 1 / (x * y) := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equals_one_plus_reciprocal_product_l3686_368671


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l3686_368697

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x - 3 else -2 + Real.log x

theorem f_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l3686_368697


namespace NUMINAMATH_CALUDE_min_planks_for_color_condition_l3686_368692

/-- Represents a fence with colored planks. -/
structure Fence where
  n : ℕ                            -- number of planks
  colors : Fin n → Fin 100         -- color of each plank

/-- Checks if the fence satisfies the color condition. -/
def satisfiesColorCondition (f : Fence) : Prop :=
  ∀ (i j : Fin 100), i ≠ j →
    ∃ (p q : Fin f.n), p < q ∧ f.colors p = i ∧ f.colors q = j

/-- The theorem stating the minimum number of planks required. -/
theorem min_planks_for_color_condition :
  (∃ (f : Fence), satisfiesColorCondition f) →
  (∀ (f : Fence), satisfiesColorCondition f → f.n ≥ 199) ∧
  (∃ (f : Fence), f.n = 199 ∧ satisfiesColorCondition f) :=
sorry

end NUMINAMATH_CALUDE_min_planks_for_color_condition_l3686_368692


namespace NUMINAMATH_CALUDE_part_one_part_two_l3686_368688

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| - |x + 1|

-- Theorem for part I
theorem part_one : 
  ∀ x : ℝ, f (-1/2) x ≤ -1 ↔ x ≥ 1/4 := by sorry

-- Theorem for part II
theorem part_two :
  (∃ a : ℝ, ∀ x : ℝ, f a x ≤ 2*a) ∧ 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≤ 2*a) → a ≥ 1/3) ∧
  (∀ x : ℝ, f (1/3) x ≤ 2*(1/3)) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3686_368688


namespace NUMINAMATH_CALUDE_circle_tangency_problem_l3686_368612

theorem circle_tangency_problem (C D : ℕ) : 
  C = 144 →
  (∃ (S : Finset ℕ), S = {s : ℕ | s < C ∧ C % s = 0 ∧ s ≠ C} ∧ S.card = 14) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangency_problem_l3686_368612


namespace NUMINAMATH_CALUDE_price_is_400_l3686_368621

/-- The price per phone sold by Aliyah and Vivienne -/
def price_per_phone (vivienne_phones : ℕ) (aliyah_extra_phones : ℕ) (total_revenue : ℕ) : ℚ :=
  total_revenue / (vivienne_phones + (vivienne_phones + aliyah_extra_phones))

/-- Theorem stating that the price per phone is $400 -/
theorem price_is_400 :
  price_per_phone 40 10 36000 = 400 := by
  sorry

end NUMINAMATH_CALUDE_price_is_400_l3686_368621


namespace NUMINAMATH_CALUDE_acute_angle_measure_l3686_368600

theorem acute_angle_measure (x : ℝ) : 
  0 < x → x < 90 → (90 - x = (180 - x) / 2 + 20) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_measure_l3686_368600


namespace NUMINAMATH_CALUDE_horner_method_operations_l3686_368615

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The number of operations required by Horner's method -/
def horner_operations (n : ℕ) : ℕ × ℕ :=
  (n, n)

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1 -/
def f_coeffs : List ℝ := [5, 4, 3, 2, 1, 1]

theorem horner_method_operations :
  let (mults, adds) := horner_operations (f_coeffs.length - 1)
  mults ≤ 5 ∧ adds = 5 := by sorry

end NUMINAMATH_CALUDE_horner_method_operations_l3686_368615


namespace NUMINAMATH_CALUDE_ladder_slide_speed_l3686_368664

theorem ladder_slide_speed (x y : ℝ) (dx_dt : ℝ) (h1 : x^2 + y^2 = 5^2) 
  (h2 : x = 1.4) (h3 : dx_dt = 3) : 
  ∃ dy_dt : ℝ, 2*x*dx_dt + 2*y*dy_dt = 0 ∧ |dy_dt| = 0.875 := by
sorry

end NUMINAMATH_CALUDE_ladder_slide_speed_l3686_368664


namespace NUMINAMATH_CALUDE_symmetry_about_x_axis_l3686_368623

/-- A point in the 2D plane represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis for two points. -/
def symmetricAboutXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- The theorem stating that (2, -3) is symmetric to (2, 3) with respect to the x-axis. -/
theorem symmetry_about_x_axis :
  symmetricAboutXAxis (Point.mk 2 3) (Point.mk 2 (-3)) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_about_x_axis_l3686_368623


namespace NUMINAMATH_CALUDE_sector_area_l3686_368607

theorem sector_area (r : ℝ) (θ : ℝ) (h : r = 2) (k : θ = π / 3) :
  (1 / 2) * r^2 * θ = (2 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3686_368607


namespace NUMINAMATH_CALUDE_prime_even_intersection_l3686_368689

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def isEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def P : Set ℕ := {n : ℕ | isPrime n}
def Q : Set ℕ := {n : ℕ | isEven n}

theorem prime_even_intersection : P ∩ Q = {2} := by sorry

end NUMINAMATH_CALUDE_prime_even_intersection_l3686_368689


namespace NUMINAMATH_CALUDE_min_production_quantity_to_break_even_l3686_368619

-- Define the total cost function
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

-- Define the sales revenue function
def sales_revenue (x : ℝ) : ℝ := 250 * x

-- Define the break-even condition
def breaks_even (x : ℝ) : Prop := sales_revenue x ≥ total_cost x

-- Theorem statement
theorem min_production_quantity_to_break_even :
  ∃ (x : ℝ), x = 150 ∧ x ∈ Set.Ioo 0 240 ∧ breaks_even x ∧
  ∀ (y : ℝ), y ∈ Set.Ioo 0 240 → breaks_even y → y ≥ x :=
sorry

end NUMINAMATH_CALUDE_min_production_quantity_to_break_even_l3686_368619


namespace NUMINAMATH_CALUDE_trajectory_intersection_fixed_point_l3686_368648

/-- The trajectory of a point equidistant from a fixed point and a fixed line -/
def Trajectory : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- A line not perpendicular to the x-axis -/
structure Line where
  t : ℝ
  m : ℝ
  h : t ≠ 0

/-- The condition that a line intersects the trajectory at two distinct points -/
def intersects_trajectory (l : Line) : Prop :=
  ∃ p q : ℝ × ℝ, p ≠ q ∧ p ∈ Trajectory ∧ q ∈ Trajectory ∧
    p.1 = l.t * p.2 + l.m ∧ q.1 = l.t * q.2 + l.m

/-- The condition that the x-axis is the angle bisector of ∠PBQ -/
def x_axis_bisects (l : Line) : Prop :=
  ∀ p q : ℝ × ℝ, p ≠ q → p ∈ Trajectory → q ∈ Trajectory →
    p.1 = l.t * p.2 + l.m → q.1 = l.t * q.2 + l.m →
    p.2 / (p.1 + 3) + q.2 / (q.1 + 3) = 0

/-- The main theorem -/
theorem trajectory_intersection_fixed_point :
  ∀ l : Line, intersects_trajectory l → x_axis_bisects l →
    l.m = 3 :=
sorry

end NUMINAMATH_CALUDE_trajectory_intersection_fixed_point_l3686_368648
