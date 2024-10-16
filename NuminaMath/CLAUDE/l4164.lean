import Mathlib

namespace NUMINAMATH_CALUDE_science_fair_competition_l4164_416418

theorem science_fair_competition (k h n : ℕ) : 
  h = (3 * k) / 5 →
  n = 2 * (k + h) →
  k + h + n = 240 →
  k = 50 ∧ h = 30 ∧ n = 160 := by
sorry

end NUMINAMATH_CALUDE_science_fair_competition_l4164_416418


namespace NUMINAMATH_CALUDE_cos_two_alpha_value_l4164_416441

theorem cos_two_alpha_value (α : ℝ) (h : Real.sin (π / 2 - α) = 1 / 4) : 
  Real.cos (2 * α) = -7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_value_l4164_416441


namespace NUMINAMATH_CALUDE_cube_root_equation_l4164_416489

theorem cube_root_equation : ∃ A : ℝ, 32 * A * A * A = 42592 ∧ A = 11 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_l4164_416489


namespace NUMINAMATH_CALUDE_practice_problems_count_l4164_416451

theorem practice_problems_count (N : ℕ) 
  (h1 : N > 0)
  (h2 : (4 / 5 : ℚ) * (3 / 4 : ℚ) * (2 / 3 : ℚ) * N = 24) : N = 60 := by
  sorry

#check practice_problems_count

end NUMINAMATH_CALUDE_practice_problems_count_l4164_416451


namespace NUMINAMATH_CALUDE_leading_coefficient_of_g_l4164_416457

/-- A polynomial g satisfying g(x + 1) - g(x) = 4x + 6 for all x has a leading coefficient of 2 -/
theorem leading_coefficient_of_g (g : ℝ → ℝ) (hg : ∀ x, g (x + 1) - g x = 4 * x + 6) :
  ∃ (a b c : ℝ), (∀ x, g x = 2 * x^2 + a * x + b) ∧ c = 2 ∧ c ≠ 0 ∧ 
  (∀ d, (∀ x, g x = d * x^2 + a * x + b) → d ≤ c) := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_g_l4164_416457


namespace NUMINAMATH_CALUDE_complement_A_in_U_l4164_416414

def U : Set ℤ := {-3, -1, 0, 1, 3}

def A : Set ℤ := {x | x^2 - 2*x - 3 = 0}

theorem complement_A_in_U : 
  {x ∈ U | x ∉ A} = {-3, 0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l4164_416414


namespace NUMINAMATH_CALUDE_isosceles_triangle_condition_l4164_416422

theorem isosceles_triangle_condition (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_condition : 2 * Real.cos B * Real.sin A = Real.sin C) : A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_condition_l4164_416422


namespace NUMINAMATH_CALUDE_complex_number_properties_l4164_416427

theorem complex_number_properties : 
  (∃ (s₁ s₂ : Prop) (s₃ s₄ : Prop), 
    s₁ ∧ s₂ ∧ ¬s₃ ∧ ¬s₄ ∧
    s₁ = (∀ z₁ z₂ : ℂ, z₁ * z₂ = z₂ * z₁) ∧
    s₂ = (∀ z₁ z₂ : ℂ, Complex.abs (z₁ * z₂) = Complex.abs z₁ * Complex.abs z₂) ∧
    s₃ = (∀ z : ℂ, Complex.abs z = 1 → z = 1 ∨ z = -1) ∧
    s₄ = (∀ z : ℂ, (Complex.abs z)^2 = z^2)) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l4164_416427


namespace NUMINAMATH_CALUDE_cube_sum_l4164_416420

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where

/-- The number of faces in a cube -/
def Cube.faces (c : Cube) : ℕ := 6

/-- The number of edges in a cube -/
def Cube.edges (c : Cube) : ℕ := 12

/-- The number of vertices in a cube -/
def Cube.vertices (c : Cube) : ℕ := 8

/-- The sum of faces, edges, and vertices in a cube is 26 -/
theorem cube_sum (c : Cube) : c.faces + c.edges + c.vertices = 26 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_l4164_416420


namespace NUMINAMATH_CALUDE_mark_to_jenna_ratio_l4164_416442

/-- The number of math problems in the homework -/
def total_problems : ℕ := 20

/-- The number of problems Martha finished -/
def martha_problems : ℕ := 2

/-- The number of problems Jenna finished -/
def jenna_problems : ℕ := 4 * martha_problems - 2

/-- The number of problems Angela finished -/
def angela_problems : ℕ := 9

/-- The number of problems Mark finished -/
def mark_problems : ℕ := total_problems - (martha_problems + jenna_problems + angela_problems)

/-- Theorem stating the ratio of problems Mark finished to problems Jenna finished -/
theorem mark_to_jenna_ratio : 
  (mark_problems : ℚ) / jenna_problems = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_mark_to_jenna_ratio_l4164_416442


namespace NUMINAMATH_CALUDE_factorial_square_root_squared_l4164_416484

theorem factorial_square_root_squared : (((4 * 3 * 2 * 1) * (3 * 2 * 1) : ℕ).sqrt ^ 2 : ℝ) = 144 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_squared_l4164_416484


namespace NUMINAMATH_CALUDE_brie_blouses_l4164_416474

/-- The number of blouses Brie has -/
def num_blouses : ℕ := sorry

/-- The number of skirts Brie has -/
def num_skirts : ℕ := 6

/-- The number of slacks Brie has -/
def num_slacks : ℕ := 8

/-- The percentage of blouses in the hamper -/
def blouse_hamper_percent : ℚ := 75 / 100

/-- The percentage of skirts in the hamper -/
def skirt_hamper_percent : ℚ := 50 / 100

/-- The percentage of slacks in the hamper -/
def slack_hamper_percent : ℚ := 25 / 100

/-- The total number of clothes to be washed -/
def clothes_to_wash : ℕ := 14

theorem brie_blouses : 
  num_blouses = 12 := by sorry

end NUMINAMATH_CALUDE_brie_blouses_l4164_416474


namespace NUMINAMATH_CALUDE_first_class_equipment_amount_l4164_416413

/-- Represents the amount of equipment -/
structure Equipment where
  higherClass : ℕ
  firstClass : ℕ

/-- The initial distribution of equipment at two sites -/
structure InitialDistribution where
  site1 : Equipment
  site2 : Equipment

/-- The final distribution of equipment after transfers -/
structure FinalDistribution where
  site1 : Equipment
  site2 : Equipment

/-- Transfers equipment between sites according to the problem description -/
def transfer (init : InitialDistribution) : FinalDistribution :=
  sorry

/-- The conditions of the problem -/
def problemConditions (init : InitialDistribution) (final : FinalDistribution) : Prop :=
  init.site1.firstClass = 0 ∧
  init.site2.higherClass = 0 ∧
  init.site1.higherClass < init.site2.firstClass ∧
  final = transfer init ∧
  final.site1.higherClass = final.site2.higherClass + 26 ∧
  final.site2.higherClass + final.site2.firstClass > 
    (init.site2.higherClass + init.site2.firstClass) * 21 / 20

theorem first_class_equipment_amount 
  (init : InitialDistribution) 
  (final : FinalDistribution) 
  (h : problemConditions init final) : 
  init.site2.firstClass = 60 :=
sorry

end NUMINAMATH_CALUDE_first_class_equipment_amount_l4164_416413


namespace NUMINAMATH_CALUDE_sqrt_32_div_sqrt_2_eq_4_l4164_416411

theorem sqrt_32_div_sqrt_2_eq_4 : Real.sqrt 32 / Real.sqrt 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_32_div_sqrt_2_eq_4_l4164_416411


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l4164_416437

theorem rectangle_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 160) :
  let square_side := square_perimeter / 4
  let rect_length := square_side
  let rect_width := square_side / 4
  let rect_perimeter := 2 * (rect_length + rect_width)
  rect_perimeter = 100 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l4164_416437


namespace NUMINAMATH_CALUDE_mans_age_puzzle_l4164_416432

theorem mans_age_puzzle (A : ℕ) (h : A = 72) :
  ∃ N : ℕ, (A + 6) * N - (A - 6) * N = A ∧ N = 6 := by
  sorry

end NUMINAMATH_CALUDE_mans_age_puzzle_l4164_416432


namespace NUMINAMATH_CALUDE_cos_shifted_angle_l4164_416429

theorem cos_shifted_angle (α : Real) (h : Real.sin (α + π/6) = 1/3) :
  Real.cos (α + 2*π/3) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_shifted_angle_l4164_416429


namespace NUMINAMATH_CALUDE_division_problem_l4164_416410

theorem division_problem (n : ℕ) : 
  n / 16 = 10 ∧ n % 16 = 1 → n = 161 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4164_416410


namespace NUMINAMATH_CALUDE_complex_expression_value_l4164_416438

theorem complex_expression_value : 
  ∃ (i : ℂ), i^2 = -1 ∧ i^3 * (1 + i)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_value_l4164_416438


namespace NUMINAMATH_CALUDE_joan_initial_balloons_l4164_416415

/-- The number of balloons Joan lost -/
def lost_balloons : ℕ := 2

/-- The number of balloons Joan currently has -/
def current_balloons : ℕ := 7

/-- The initial number of balloons Joan had -/
def initial_balloons : ℕ := current_balloons + lost_balloons

theorem joan_initial_balloons : initial_balloons = 9 := by
  sorry

end NUMINAMATH_CALUDE_joan_initial_balloons_l4164_416415


namespace NUMINAMATH_CALUDE_perpendicular_condition_l4164_416405

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the relation of a line being within a plane
variable (line_in_plane : Line → Plane → Prop)

theorem perpendicular_condition (α β : Plane) (m : Line) 
  (h1 : α ≠ β) 
  (h2 : line_in_plane m α) : 
  (∀ m, line_in_plane m α → perp_line_plane m β → perp_planes α β) ∧ 
  (∃ m, line_in_plane m α ∧ perp_planes α β ∧ ¬perp_line_plane m β) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l4164_416405


namespace NUMINAMATH_CALUDE_profit_and_marginal_profit_maxima_l4164_416477

def R (x : ℕ) : ℝ := 3000 * x - 20 * x^2
def C (x : ℕ) : ℝ := 600 * x + 2000
def p (x : ℕ) : ℝ := R x - C x
def Mp (x : ℕ) : ℝ := p (x + 1) - p x

theorem profit_and_marginal_profit_maxima 
  (h : ∀ x : ℕ, 0 < x ∧ x ≤ 100) :
  (∃ x : ℕ, p x = 74000 ∧ ∀ y : ℕ, p y ≤ 74000) ∧
  (∃ x : ℕ, Mp x = 2340 ∧ ∀ y : ℕ, Mp y ≤ 2340) :=
sorry

end NUMINAMATH_CALUDE_profit_and_marginal_profit_maxima_l4164_416477


namespace NUMINAMATH_CALUDE_cube_division_theorem_l4164_416439

/-- A point in 3D space represented by rational coordinates -/
structure RationalPoint where
  x : ℚ
  y : ℚ
  z : ℚ

/-- Represents that a point is inside the unit cube -/
def insideUnitCube (p : RationalPoint) : Prop :=
  0 < p.x ∧ p.x < 1 ∧ 0 < p.y ∧ p.y < 1 ∧ 0 < p.z ∧ p.z < 1

theorem cube_division_theorem (points : Finset RationalPoint) 
    (h : points.card = 2003) 
    (h_inside : ∀ p ∈ points, insideUnitCube p) :
    ∃ (n : ℕ), n > 2003 ∧ 
    ∀ p ∈ points, ∃ (i j k : ℕ), 
      i < n ∧ j < n ∧ k < n ∧
      (i : ℚ) / n < p.x ∧ p.x < ((i + 1) : ℚ) / n ∧
      (j : ℚ) / n < p.y ∧ p.y < ((j + 1) : ℚ) / n ∧
      (k : ℚ) / n < p.z ∧ p.z < ((k + 1) : ℚ) / n :=
by sorry


end NUMINAMATH_CALUDE_cube_division_theorem_l4164_416439


namespace NUMINAMATH_CALUDE_intersection_value_l4164_416417

theorem intersection_value (A B : Set ℝ) (a : ℝ) :
  A = {x : ℝ | x ≤ 1} →
  B = {x : ℝ | x ≥ a} →
  A ∩ B = {1} →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_value_l4164_416417


namespace NUMINAMATH_CALUDE_problem_one_l4164_416407

theorem problem_one : Real.sqrt 12 + (-2024)^0 - 4 * Real.sin (60 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_l4164_416407


namespace NUMINAMATH_CALUDE_rhombus_count_in_triangle_l4164_416433

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ

/-- Represents a rhombus composed of smaller equilateral triangles -/
structure Rhombus where
  num_triangles : ℕ

/-- The number of rhombuses in one direction -/
def rhombuses_in_one_direction (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The main theorem -/
theorem rhombus_count_in_triangle (big_triangle : EquilateralTriangle) 
  (small_triangle : EquilateralTriangle) (rhombus : Rhombus) : 
  big_triangle.side_length = 10 →
  small_triangle.side_length = 1 →
  rhombus.num_triangles = 8 →
  (rhombuses_in_one_direction 7) * 3 = 84 := by
  sorry

#check rhombus_count_in_triangle

end NUMINAMATH_CALUDE_rhombus_count_in_triangle_l4164_416433


namespace NUMINAMATH_CALUDE_boys_employed_is_50_l4164_416440

/-- Represents the roadway construction scenario --/
structure RoadwayConstruction where
  totalLength : ℝ
  totalTime : ℝ
  initialLength : ℝ
  initialTime : ℝ
  initialMen : ℕ
  initialHours : ℝ
  overtimeHours : ℝ
  boyEfficiency : ℝ

/-- Calculates the number of boys employed in the roadway construction --/
def calculateBoysEmployed (rc : RoadwayConstruction) : ℕ :=
  sorry

/-- Theorem stating that the number of boys employed is 50 --/
theorem boys_employed_is_50 (rc : RoadwayConstruction) : 
  rc.totalLength = 15 ∧ 
  rc.totalTime = 40 ∧ 
  rc.initialLength = 3 ∧ 
  rc.initialTime = 10 ∧ 
  rc.initialMen = 180 ∧ 
  rc.initialHours = 8 ∧ 
  rc.overtimeHours = 1 ∧ 
  rc.boyEfficiency = 2/3 → 
  calculateBoysEmployed rc = 50 := by
  sorry

end NUMINAMATH_CALUDE_boys_employed_is_50_l4164_416440


namespace NUMINAMATH_CALUDE_three_not_in_range_iff_c_in_open_interval_l4164_416493

-- Define the function g
def g (c : ℝ) (x : ℝ) : ℝ := x^2 + c*x + 4

-- State the theorem
theorem three_not_in_range_iff_c_in_open_interval :
  ∀ c : ℝ, (∀ x : ℝ, g c x ≠ 3) ↔ c ∈ Set.Ioo (-2) 2 := by sorry

end NUMINAMATH_CALUDE_three_not_in_range_iff_c_in_open_interval_l4164_416493


namespace NUMINAMATH_CALUDE_stream_speed_l4164_416472

theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  boat_speed = 18 →
  downstream_distance = 48 →
  upstream_distance = 32 →
  ∃ (time : ℝ), time > 0 ∧
    time * (boat_speed + 3.6) = downstream_distance ∧
    time * (boat_speed - 3.6) = upstream_distance :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l4164_416472


namespace NUMINAMATH_CALUDE_system_solution_ratio_l4164_416416

theorem system_solution_ratio (a b x y : ℝ) : 
  8 * x - 6 * y = a →
  9 * x - 12 * y = b →
  x ≠ 0 →
  y ≠ 0 →
  b ≠ 0 →
  a / b = 8 / 9 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l4164_416416


namespace NUMINAMATH_CALUDE_dividend_calculation_l4164_416499

theorem dividend_calculation (remainder quotient divisor dividend : ℕ) : 
  remainder = 8 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + 3 →
  dividend = divisor * quotient + remainder →
  dividend = 251 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l4164_416499


namespace NUMINAMATH_CALUDE_triangle_area_l4164_416476

theorem triangle_area (R : ℝ) (A : ℝ) (b c : ℝ) (h1 : R = 4) (h2 : A = π / 3) (h3 : b - c = 4) :
  let S := (1 / 2) * b * c * Real.sin A
  S = 8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l4164_416476


namespace NUMINAMATH_CALUDE_ball_338_in_cup_360_l4164_416409

/-- The number of cups in the circle. -/
def n : ℕ := 1000

/-- The step size for placing balls. -/
def step : ℕ := 7

/-- The index of the ball we're interested in. -/
def ball_index : ℕ := 338

/-- Function to calculate the cup number for a given ball index. -/
def cup_number (k : ℕ) : ℕ :=
  (1 + step * (k - 1)) % n

theorem ball_338_in_cup_360 : cup_number ball_index = 360 := by
  sorry

end NUMINAMATH_CALUDE_ball_338_in_cup_360_l4164_416409


namespace NUMINAMATH_CALUDE_zoe_calorie_intake_l4164_416412

-- Define the quantities
def strawberries : ℕ := 12
def yogurt_ounces : ℕ := 6
def calories_per_strawberry : ℕ := 4
def calories_per_yogurt_ounce : ℕ := 17

-- Define the total calories
def total_calories : ℕ := strawberries * calories_per_strawberry + yogurt_ounces * calories_per_yogurt_ounce

-- Theorem statement
theorem zoe_calorie_intake : total_calories = 150 := by
  sorry

end NUMINAMATH_CALUDE_zoe_calorie_intake_l4164_416412


namespace NUMINAMATH_CALUDE_inequality_proof_l4164_416498

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + 8 / (x * y) + y^2 ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l4164_416498


namespace NUMINAMATH_CALUDE_universal_set_determination_l4164_416467

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 5}
def complementA : Set Nat := {2, 4, 6}

theorem universal_set_determination :
  (A ⊆ U) ∧ (complementA ⊆ U) ∧ (A ∪ complementA = U) ∧ (A ∩ complementA = ∅) →
  U = {1, 2, 3, 4, 5, 6} :=
by sorry

end NUMINAMATH_CALUDE_universal_set_determination_l4164_416467


namespace NUMINAMATH_CALUDE_digits_of_product_l4164_416448

theorem digits_of_product : ∃ n : ℕ, n > 0 ∧ (2^15 * 5^10 * 12 : ℕ) < 10^n ∧ (2^15 * 5^10 * 12 : ℕ) ≥ 10^(n-1) ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_product_l4164_416448


namespace NUMINAMATH_CALUDE_root_implies_a_range_l4164_416400

theorem root_implies_a_range :
  ∀ a : ℝ, (∃ x : ℝ, 9^(-|x - 2|) - 4 * 3^(-|x - 2|) - a = 0) → -3 ≤ a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_a_range_l4164_416400


namespace NUMINAMATH_CALUDE_train_length_l4164_416444

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 72 → time_s = 8 → speed_kmh * (1000 / 3600) * time_s = 160 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l4164_416444


namespace NUMINAMATH_CALUDE_iron_volume_change_l4164_416465

/-- If the volume of iron reduces by 1/34 when solidifying, then the volume increases by 1/33 when melting back to its original state. -/
theorem iron_volume_change (V : ℝ) (V_block : ℝ) (h : V_block = V * (1 - 1/34)) :
  (V - V_block) / V_block = 1/33 := by
sorry

end NUMINAMATH_CALUDE_iron_volume_change_l4164_416465


namespace NUMINAMATH_CALUDE_point_not_on_line_l4164_416428

theorem point_not_on_line (a c : ℝ) (h : a * c > 0) : 
  ¬ (∃ (x y : ℝ), x = 2023 ∧ y = 0 ∧ y = a * x + c) :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_line_l4164_416428


namespace NUMINAMATH_CALUDE_m_range_l4164_416479

def p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧
    m * x₁^2 - x₁ + m - 4 = 0 ∧
    m * x₂^2 - x₂ + m - 4 = 0

theorem m_range :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m) → Real.sqrt 2 + 1 ≤ m ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_m_range_l4164_416479


namespace NUMINAMATH_CALUDE_sum_absolute_value_l4164_416463

theorem sum_absolute_value (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h : x₁ + 1 = x₂ + 2 ∧ x₂ + 2 = x₃ + 3 ∧ x₃ + 3 = x₄ + 4 ∧ x₄ + 4 = x₅ + 5 ∧ 
       x₅ + 5 = x₁ + x₂ + x₃ + x₄ + x₅ + 6) : 
  |x₁ + x₂ + x₃ + x₄ + x₅| = 3.75 := by
sorry

end NUMINAMATH_CALUDE_sum_absolute_value_l4164_416463


namespace NUMINAMATH_CALUDE_max_value_of_function_l4164_416462

theorem max_value_of_function (x y z : ℝ) (h : x^2 + y^2 + z^2 ≠ 0) :
  (x*y + 2*y*z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l4164_416462


namespace NUMINAMATH_CALUDE_range_of_a_l4164_416459

theorem range_of_a (p q : ℝ → Prop) :
  (∀ x, q x → p x) ∧
  (∃ x, p x ∧ ¬q x) ∧
  (∀ x, q x ↔ -x^2 + 5*x - 6 > 0) ∧
  (∀ x a, p x ↔ |x - a| < 4) →
  ∃ a_min a_max, a_min = -1 ∧ a_max = 6 ∧ ∀ a, (a_min < a ∧ a < a_max) → 
    (∃ x, p x) ∧ (∃ x, ¬p x) := by sorry


end NUMINAMATH_CALUDE_range_of_a_l4164_416459


namespace NUMINAMATH_CALUDE_largest_N_for_dispersive_connective_perm_l4164_416435

/-- The set of residues modulo 17 -/
def X : Set ℕ := {x | x < 17}

/-- Two numbers in X are adjacent if they differ by 1 or are 0 and 16 -/
def adjacent (a b : ℕ) : Prop :=
  (a ∈ X ∧ b ∈ X) ∧ ((a + 1 ≡ b [ZMOD 17]) ∨ (b + 1 ≡ a [ZMOD 17]))

/-- A permutation on X -/
def permutation_on_X (p : ℕ → ℕ) : Prop :=
  Function.Bijective p ∧ ∀ x, x ∈ X → p x ∈ X

/-- A permutation is dispersive if it never maps adjacent values to adjacent values -/
def dispersive (p : ℕ → ℕ) : Prop :=
  permutation_on_X p ∧ ∀ a b, adjacent a b → ¬adjacent (p a) (p b)

/-- A permutation is connective if it always maps adjacent values to adjacent values -/
def connective (p : ℕ → ℕ) : Prop :=
  permutation_on_X p ∧ ∀ a b, adjacent a b → adjacent (p a) (p b)

/-- The composition of a permutation with itself n times -/
def iterate_perm (p : ℕ → ℕ) : ℕ → (ℕ → ℕ)
  | 0 => id
  | n + 1 => p ∘ (iterate_perm p n)

/-- The theorem stating the largest N for which the described permutation exists -/
theorem largest_N_for_dispersive_connective_perm :
  ∃ (p : ℕ → ℕ), permutation_on_X p ∧
    (∀ k < 8, dispersive (iterate_perm p k)) ∧
    connective (iterate_perm p 8) ∧
    ∀ (q : ℕ → ℕ) (m : ℕ),
      (permutation_on_X q ∧
       (∀ k < m, dispersive (iterate_perm q k)) ∧
       connective (iterate_perm q m)) →
      m ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_largest_N_for_dispersive_connective_perm_l4164_416435


namespace NUMINAMATH_CALUDE_correct_value_for_square_l4164_416449

theorem correct_value_for_square (x : ℕ) : 60 + x * 5 = 500 ↔ x = 88 :=
by sorry

end NUMINAMATH_CALUDE_correct_value_for_square_l4164_416449


namespace NUMINAMATH_CALUDE_int_tan_triangle_values_l4164_416455

-- Define a triangle with integer tangents
structure IntTanTriangle where
  α : Real
  β : Real
  γ : Real
  tan_α : Int
  tan_β : Int
  tan_γ : Int
  sum_angles : α + β + γ = Real.pi
  tan_α_def : Real.tan α = tan_α
  tan_β_def : Real.tan β = tan_β
  tan_γ_def : Real.tan γ = tan_γ

-- Theorem statement
theorem int_tan_triangle_values (t : IntTanTriangle) :
  (t.tan_α = 1 ∧ t.tan_β = 2 ∧ t.tan_γ = 3) ∨
  (t.tan_α = 1 ∧ t.tan_β = 3 ∧ t.tan_γ = 2) ∨
  (t.tan_α = 2 ∧ t.tan_β = 1 ∧ t.tan_γ = 3) ∨
  (t.tan_α = 2 ∧ t.tan_β = 3 ∧ t.tan_γ = 1) ∨
  (t.tan_α = 3 ∧ t.tan_β = 1 ∧ t.tan_γ = 2) ∨
  (t.tan_α = 3 ∧ t.tan_β = 2 ∧ t.tan_γ = 1) :=
by sorry

end NUMINAMATH_CALUDE_int_tan_triangle_values_l4164_416455


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l4164_416497

/-- A rhombus with side length 51 and shorter diagonal 48 has a longer diagonal of 90 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diag : ℝ) (longer_diag : ℝ) : 
  side = 51 → shorter_diag = 48 → longer_diag = 90 → 
  side^2 = (shorter_diag/2)^2 + (longer_diag/2)^2 := by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l4164_416497


namespace NUMINAMATH_CALUDE_max_m_value_l4164_416481

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_eq : 2/a + 1/b = 1/4) (h_ineq : ∀ m : ℝ, 2*a + b ≥ 4*m) : 
  ∃ m_max : ℝ, m_max = 9 ∧ ∀ m : ℝ, (∀ x : ℝ, 2*a + b ≥ 4*x → m ≤ x) → m ≤ m_max :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l4164_416481


namespace NUMINAMATH_CALUDE_min_value_theorem_l4164_416468

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 + 2*x)/(y - 2) + (y^2 + 2*y)/(x - 2) ≥ 22 ∧
  ((x^2 + 2*x)/(y - 2) + (y^2 + 2*y)/(x - 2) = 22 ↔ x = 3 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4164_416468


namespace NUMINAMATH_CALUDE_number_problem_l4164_416447

theorem number_problem (x : ℚ) : x + (-5/12) - (-5/2) = 1/3 → x = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4164_416447


namespace NUMINAMATH_CALUDE_quadratic_discriminant_perfect_square_l4164_416488

theorem quadratic_discriminant_perfect_square 
  (a b c t : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : a * t^2 + b * t + c = 0) : 
  b^2 - 4*a*c = (2*a*t + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_perfect_square_l4164_416488


namespace NUMINAMATH_CALUDE_relationship_between_x_and_y_l4164_416401

theorem relationship_between_x_and_y (x y : ℝ) 
  (h1 : 2 * x - y > x + 1) 
  (h2 : x + 2 * y < 2 * y - 3) : 
  x < -3 ∧ y < -4 ∧ x > y + 1 := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_x_and_y_l4164_416401


namespace NUMINAMATH_CALUDE_sum_x_y_z_l4164_416478

theorem sum_x_y_z (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 3 * y + x) : 
  x + y + z = 14 * x := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_z_l4164_416478


namespace NUMINAMATH_CALUDE_states_fraction_proof_l4164_416406

theorem states_fraction_proof (total_states : ℕ) (decade_states : ℕ) :
  total_states = 22 →
  decade_states = 8 →
  (decade_states : ℚ) / total_states = 4 / 11 := by
sorry

end NUMINAMATH_CALUDE_states_fraction_proof_l4164_416406


namespace NUMINAMATH_CALUDE_prob_at_least_six_heads_in_eight_flips_l4164_416483

/-- The probability of getting at least 6 heads in 8 fair coin flips -/
theorem prob_at_least_six_heads_in_eight_flips :
  let n : ℕ := 8  -- number of coin flips
  let k : ℕ := 6  -- minimum number of heads
  let p : ℚ := 1/2  -- probability of heads for a fair coin
  Finset.sum (Finset.range (n - k + 1)) (λ i => (n.choose (k + i)) * p^(k + i) * (1 - p)^(n - (k + i))) = 37/256 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_six_heads_in_eight_flips_l4164_416483


namespace NUMINAMATH_CALUDE_shared_angle_measure_l4164_416491

/-- A configuration of a regular pentagon sharing a side with an equilateral triangle -/
structure PentagonTriangleConfig where
  /-- The measure of an interior angle of the regular pentagon in degrees -/
  pentagon_angle : ℝ
  /-- The measure of an interior angle of the equilateral triangle in degrees -/
  triangle_angle : ℝ
  /-- The condition that the pentagon is regular -/
  pentagon_regular : pentagon_angle = 108
  /-- The condition that the triangle is equilateral -/
  triangle_equilateral : triangle_angle = 60

/-- The theorem stating that the angle formed by the shared side and the adjacent sides is 6 degrees -/
theorem shared_angle_measure (config : PentagonTriangleConfig) :
  let total_angle := config.pentagon_angle + config.triangle_angle
  let shared_angle := (180 - total_angle) / 2
  shared_angle = 6 := by sorry

end NUMINAMATH_CALUDE_shared_angle_measure_l4164_416491


namespace NUMINAMATH_CALUDE_bridge_length_l4164_416426

/-- The length of a bridge given specific train crossing conditions -/
theorem bridge_length (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 100 →
  crossing_time = 36 →
  train_speed = 40 →
  train_speed * crossing_time - train_length = 1340 := by
sorry

end NUMINAMATH_CALUDE_bridge_length_l4164_416426


namespace NUMINAMATH_CALUDE_sum_equals_seventeen_l4164_416495

theorem sum_equals_seventeen 
  (a b c d : ℝ) 
  (h1 : a * (c + d) + b * (c + d) = 42) 
  (h2 : c + d = 3) : 
  a + b + c + d = 17 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_seventeen_l4164_416495


namespace NUMINAMATH_CALUDE_cubic_parabola_x_intercepts_l4164_416482

theorem cubic_parabola_x_intercepts :
  ∃! x : ℝ, x = -3 * 0^3 + 2 * 0^2 - 0 + 2 :=
sorry

end NUMINAMATH_CALUDE_cubic_parabola_x_intercepts_l4164_416482


namespace NUMINAMATH_CALUDE_vendor_division_l4164_416443

theorem vendor_division (account_balance : Nat) (min_addition : Nat) (num_vendors : Nat) : 
  account_balance = 329864 →
  min_addition = 4 →
  num_vendors = 20 →
  (∀ k < num_vendors, account_balance % k ≠ 0 ∨ (account_balance + min_addition) % k ≠ 0) ∧
  account_balance % num_vendors ≠ 0 ∧
  (account_balance + min_addition) % num_vendors = 0 :=
by sorry

end NUMINAMATH_CALUDE_vendor_division_l4164_416443


namespace NUMINAMATH_CALUDE_remaining_surface_area_after_removal_l4164_416402

/-- The remaining surface area of a cube after removing a smaller cube from its corner --/
theorem remaining_surface_area_after_removal (a b : ℝ) (ha : a > 0) (hb : b > 0) (hba : b < 3*a) :
  6 * (3*a)^2 - 3 * b^2 + 3 * b^2 = 54 * a^2 := by
  sorry

#check remaining_surface_area_after_removal

end NUMINAMATH_CALUDE_remaining_surface_area_after_removal_l4164_416402


namespace NUMINAMATH_CALUDE_f_zeros_iff_a_range_l4164_416430

/-- The cubic function f(x) = x³ - ax² + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x

/-- The statement that f(x) has less than 3 zeros -/
def has_less_than_three_zeros (a : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), ∀ (x : ℝ), f a x = 0 → x = x₁ ∨ x = x₂

/-- The main theorem: f(x) has less than 3 zeros iff a ∈ (-∞, 3] -/
theorem f_zeros_iff_a_range :
  ∀ (a : ℝ), has_less_than_three_zeros a ↔ a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_f_zeros_iff_a_range_l4164_416430


namespace NUMINAMATH_CALUDE_cos_18_degrees_l4164_416431

theorem cos_18_degrees : Real.cos (18 * π / 180) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_18_degrees_l4164_416431


namespace NUMINAMATH_CALUDE_sallys_onions_l4164_416469

theorem sallys_onions (fred_onions : ℕ) (given_to_sara : ℕ) (remaining_onions : ℕ) : ℕ :=
  sorry

end NUMINAMATH_CALUDE_sallys_onions_l4164_416469


namespace NUMINAMATH_CALUDE_binomial_14_11_l4164_416446

theorem binomial_14_11 : Nat.choose 14 11 = 364 := by
  sorry

end NUMINAMATH_CALUDE_binomial_14_11_l4164_416446


namespace NUMINAMATH_CALUDE_banana_fraction_proof_l4164_416452

theorem banana_fraction_proof (jefferson_bananas : ℕ) (walter_bananas : ℚ) (f : ℚ) :
  jefferson_bananas = 56 →
  walter_bananas = 56 - 56 * f →
  (56 + (56 - 56 * f)) / 2 = 49 →
  f = 1/4 := by
sorry

end NUMINAMATH_CALUDE_banana_fraction_proof_l4164_416452


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l4164_416486

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * (1 + major_minor_ratio)

/-- Theorem: The major axis length of the ellipse is 7.2 -/
theorem ellipse_major_axis_length :
  major_axis_length 2 0.8 = 7.2 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l4164_416486


namespace NUMINAMATH_CALUDE_smallest_d_correct_l4164_416464

/-- The smallest possible value of d satisfying the triangle and square perimeter conditions -/
def smallest_d : ℕ :=
  let d : ℕ := 675
  d

theorem smallest_d_correct :
  let d := smallest_d
  -- The perimeter of the equilateral triangle exceeds the perimeter of the square by 2023 cm
  ∀ s : ℝ, 3 * (s + d) - 4 * s = 2023 →
  -- The square has a perimeter greater than 0 cm
  (s > 0) →
  -- d is a multiple of 3
  (d % 3 = 0) →
  -- d is the smallest value satisfying these conditions
  ∀ d' : ℕ, d' < d →
    (∀ s : ℝ, 3 * (s + d') - 4 * s = 2023 → s > 0 → d' % 3 = 0 → False) :=
by sorry

#eval smallest_d

end NUMINAMATH_CALUDE_smallest_d_correct_l4164_416464


namespace NUMINAMATH_CALUDE_range_of_a_l4164_416456

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 5*x - 6 ≤ 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - 4*a^2 ≤ 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a ≥ 0) →
  (∀ x, ¬(q x a) → ¬(p x)) →
  (∃ x, ¬(p x) ∧ (q x a)) →
  a ≥ 5/2 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4164_416456


namespace NUMINAMATH_CALUDE_soccer_substitutions_remainder_l4164_416496

/-- Represents the number of ways to make substitutions in a soccer game -/
def substitutions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => 11 * (12 - m) * substitutions m

/-- The total number of ways to make 0 to 3 substitutions -/
def total_substitutions : ℕ :=
  substitutions 0 + substitutions 1 + substitutions 2 + substitutions 3

theorem soccer_substitutions_remainder :
  total_substitutions ≡ 122 [MOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_soccer_substitutions_remainder_l4164_416496


namespace NUMINAMATH_CALUDE_age_of_replaced_man_l4164_416492

theorem age_of_replaced_man (n : ℕ) (A : ℝ) (increase : ℝ) (known_man_age : ℕ) (women_avg_age : ℝ) :
  n = 10 ∧ 
  increase = 6 ∧ 
  known_man_age = 22 ∧ 
  women_avg_age = 50 →
  ∃ x : ℕ, x = 18 ∧ 
    n * (A + increase) = (n - 2) * A + 2 * women_avg_age ∧
    n * A = (n - 2) * A + known_man_age + x :=
by sorry

end NUMINAMATH_CALUDE_age_of_replaced_man_l4164_416492


namespace NUMINAMATH_CALUDE_bacon_suggestion_count_l4164_416423

theorem bacon_suggestion_count (mashed_potatoes : ℕ) (bacon : ℕ) : 
  mashed_potatoes = 479 → 
  bacon = mashed_potatoes + 10 → 
  bacon = 489 := by
sorry

end NUMINAMATH_CALUDE_bacon_suggestion_count_l4164_416423


namespace NUMINAMATH_CALUDE_complex_modulus_product_l4164_416424

theorem complex_modulus_product : Complex.abs (5 - 3*Complex.I) * Complex.abs (5 + 3*Complex.I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l4164_416424


namespace NUMINAMATH_CALUDE_percentage_of_375_l4164_416450

theorem percentage_of_375 (x : ℝ) :
  (x / 100) * 375 = 5.4375 → x = 1.45 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_375_l4164_416450


namespace NUMINAMATH_CALUDE_system_of_equations_l4164_416490

/-- Given a system of equations with parameters n and m, prove specific values of m for different conditions. -/
theorem system_of_equations (n m x y : ℤ) : 
  (n * x + (n + 1) * y = n + 2) → 
  (x - 2 * y + m * x = -5) →
  (
    (n = 1 ∧ x + 2 * y = 3 ∧ x + y = 2 → m = -4) ∧
    (n = 3 ∧ ∃ (x y : ℤ), n * x + (n + 1) * y = n + 2 ∧ x - 2 * y + m * x = -5 → m = -2 ∨ m = 0)
  ) := by sorry

end NUMINAMATH_CALUDE_system_of_equations_l4164_416490


namespace NUMINAMATH_CALUDE_village_population_l4164_416466

theorem village_population (population : ℕ) : 
  (60 : ℕ) * population = 23040 * 100 → population = 38400 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l4164_416466


namespace NUMINAMATH_CALUDE_power_equation_l4164_416436

theorem power_equation (x y z : ℕ) : 
  3^x * 4^y = z → x - y = 9 → x = 9 → z = 19683 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_l4164_416436


namespace NUMINAMATH_CALUDE_chocolate_bar_eating_ways_l4164_416453

/-- Represents a chocolate bar of size m × n -/
structure ChocolateBar (m n : ℕ) where
  size : Fin m × Fin n → Bool

/-- Represents the state of eating a chocolate bar -/
structure EatingState (m n : ℕ) where
  bar : ChocolateBar m n
  eaten : Fin m × Fin n → Bool

/-- Checks if a piece can be eaten (has no more than two shared sides with uneaten pieces) -/
def canEat (state : EatingState m n) (pos : Fin m × Fin n) : Bool :=
  sorry

/-- Counts the number of ways to eat the chocolate bar -/
def countEatingWays (m n : ℕ) : ℕ :=
  sorry

/-- The main theorem: there are 6720 ways to eat a 2 × 4 chocolate bar -/
theorem chocolate_bar_eating_ways :
  countEatingWays 2 4 = 6720 :=
sorry

end NUMINAMATH_CALUDE_chocolate_bar_eating_ways_l4164_416453


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l4164_416461

theorem simplify_trig_expression (x : ℝ) : 
  (3 + 3 * Real.sin x - 3 * Real.cos x) / (3 + 3 * Real.sin x + 3 * Real.cos x) = Real.tan (x / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l4164_416461


namespace NUMINAMATH_CALUDE_game_download_time_l4164_416460

theorem game_download_time (total_size : ℕ) (downloaded : ℕ) (speed : ℕ) : 
  total_size = 880 → downloaded = 310 → speed = 3 → 
  (total_size - downloaded) / speed = 190 := by
  sorry

end NUMINAMATH_CALUDE_game_download_time_l4164_416460


namespace NUMINAMATH_CALUDE_peytons_score_l4164_416434

theorem peytons_score (n : ℕ) (avg_14 : ℚ) (avg_15 : ℚ) (peyton_score : ℚ) : 
  n = 15 → 
  avg_14 = 80 → 
  avg_15 = 81 → 
  (n - 1) * avg_14 + peyton_score = n * avg_15 →
  peyton_score = 95 := by
  sorry

end NUMINAMATH_CALUDE_peytons_score_l4164_416434


namespace NUMINAMATH_CALUDE_rap_song_requests_l4164_416404

/-- Represents the number of song requests for different genres in a night --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of rap song requests given the conditions --/
theorem rap_song_requests (s : SongRequests) : s.rap = 2 :=
  by
  have h1 : s.total = 30 := by sorry
  have h2 : s.electropop = s.total / 2 := by sorry
  have h3 : s.dance = s.electropop / 3 := by sorry
  have h4 : s.rock = 5 := by sorry
  have h5 : s.oldies = s.rock - 3 := by sorry
  have h6 : s.dj_choice = s.oldies / 2 := by sorry
  have h7 : s.total = s.electropop + s.dance + s.rock + s.oldies + s.dj_choice + s.rap := by sorry
  sorry

end NUMINAMATH_CALUDE_rap_song_requests_l4164_416404


namespace NUMINAMATH_CALUDE_primes_between_30_and_50_l4164_416471

/-- Count of prime numbers in a given range -/
def countPrimes (a b : ℕ) : ℕ :=
  (Finset.range (b - a + 1)).filter (fun i => Nat.Prime (i + a)) |>.card

/-- The theorem stating that there are 5 prime numbers between 30 and 50 -/
theorem primes_between_30_and_50 : countPrimes 31 49 = 5 := by
  sorry

end NUMINAMATH_CALUDE_primes_between_30_and_50_l4164_416471


namespace NUMINAMATH_CALUDE_remainder_theorem_l4164_416445

theorem remainder_theorem (N : ℤ) : 
  (N % 779 = 47) → (N % 19 = 9) := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l4164_416445


namespace NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l4164_416485

theorem min_value_of_function (x : ℝ) : 3 * x^2 + 6 / (x^2 + 1) ≥ 6 * Real.sqrt 2 - 3 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, 3 * x^2 + 6 / (x^2 + 1) = 6 * Real.sqrt 2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_min_value_achievable_l4164_416485


namespace NUMINAMATH_CALUDE_last_two_digits_a_2015_l4164_416480

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => if n % 2 = 0 then a n + 2 else 2 * a n

theorem last_two_digits_a_2015 : a 2015 % 100 = 72 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_a_2015_l4164_416480


namespace NUMINAMATH_CALUDE_greifswald_schools_l4164_416408

-- Define the schools
inductive School
| A
| B
| C

-- Define the student type
structure Student where
  id : Nat
  school : School

-- Define the knowing relation
def knows (s1 s2 : Student) : Prop := sorry

-- Define the set of all students
def AllStudents : Set Student := sorry

-- State the conditions
axiom non_empty_schools :
  ∃ (a b c : Student), a.school = School.A ∧ b.school = School.B ∧ c.school = School.C

axiom knowing_condition :
  ∀ (a b c : Student),
    a.school = School.A → b.school = School.B → c.school = School.C →
    ((knows a b ∧ knows a c ∧ ¬knows b c) ∨
     (knows a b ∧ ¬knows a c ∧ knows b c) ∨
     (¬knows a b ∧ knows a c ∧ knows b c))

-- State the theorem to be proved
theorem greifswald_schools :
  (∃ (a : Student), a.school = School.A ∧ ∀ (b : Student), b.school = School.B → knows a b) ∨
  (∃ (b : Student), b.school = School.B ∧ ∀ (c : Student), c.school = School.C → knows b c) ∨
  (∃ (c : Student), c.school = School.C ∧ ∀ (a : Student), a.school = School.A → knows c a) :=
by
  sorry

end NUMINAMATH_CALUDE_greifswald_schools_l4164_416408


namespace NUMINAMATH_CALUDE_five_consecutive_not_square_l4164_416403

theorem five_consecutive_not_square (n : ℤ) : 
  ∃ (m : ℤ), (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) ≠ m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_five_consecutive_not_square_l4164_416403


namespace NUMINAMATH_CALUDE_common_root_condition_l4164_416419

theorem common_root_condition (m : ℝ) : 
  (∃ x : ℝ, m * x - 1000 = 1001 ∧ 1001 * x = m - 1000 * x) ↔ (m = 2001 ∨ m = -2001) := by
  sorry

end NUMINAMATH_CALUDE_common_root_condition_l4164_416419


namespace NUMINAMATH_CALUDE_unique_prime_between_squares_l4164_416458

theorem unique_prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p = n^2 + 4 ∧ 
  ∃ m : ℕ, p + 7 = (n + 1)^2 ∧ 
  p = 29 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_between_squares_l4164_416458


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l4164_416475

/-- Systematic sampling function that returns the nth element of the sample -/
def systematicSample (populationSize sampleSize start n : ℕ) : ℕ :=
  start + (populationSize / sampleSize) * n

/-- Theorem: In a systematic sample of size 5 from a population of 55,
    if students 3, 25, and 47 are in the sample,
    then the other two students in the sample have numbers 14 and 36 -/
theorem systematic_sample_theorem :
  let populationSize : ℕ := 55
  let sampleSize : ℕ := 5
  let start : ℕ := 3
  (systematicSample populationSize sampleSize start 0 = 3) →
  (systematicSample populationSize sampleSize start 2 = 25) →
  (systematicSample populationSize sampleSize start 4 = 47) →
  (systematicSample populationSize sampleSize start 1 = 14) ∧
  (systematicSample populationSize sampleSize start 3 = 36) :=
by
  sorry


end NUMINAMATH_CALUDE_systematic_sample_theorem_l4164_416475


namespace NUMINAMATH_CALUDE_reading_time_per_disc_l4164_416473

/-- Proves that given a total reading time of 480 minutes, disc capacity of 60 minutes,
    and the conditions of using the smallest possible number of discs with equal reading time on each disc,
    the reading time per disc is 60 minutes. -/
theorem reading_time_per_disc (total_time : ℕ) (disc_capacity : ℕ) (reading_per_disc : ℕ) :
  total_time = 480 →
  disc_capacity = 60 →
  reading_per_disc * (total_time / disc_capacity) = total_time →
  reading_per_disc ≤ disc_capacity →
  reading_per_disc = 60 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_per_disc_l4164_416473


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l4164_416454

/-- Systematic sampling function that returns true if the number is in the sample -/
def in_systematic_sample (total : ℕ) (sample_size : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (total / sample_size) + 1

/-- Theorem stating that in a systematic sample of size 5 from 60 numbered parts,
    if 4, 16, 40, and 52 are in the sample, then 28 must also be in the sample -/
theorem systematic_sample_theorem :
  let total := 60
  let sample_size := 5
  (in_systematic_sample total sample_size 4) →
  (in_systematic_sample total sample_size 16) →
  (in_systematic_sample total sample_size 40) →
  (in_systematic_sample total sample_size 52) →
  (in_systematic_sample total sample_size 28) :=
by
  sorry

#check systematic_sample_theorem

end NUMINAMATH_CALUDE_systematic_sample_theorem_l4164_416454


namespace NUMINAMATH_CALUDE_billys_restaurant_bill_l4164_416494

/-- Calculates the total bill for a group at Billy's Restaurant -/
def calculate_bill (num_adults : ℕ) (num_children : ℕ) (cost_per_meal : ℕ) : ℕ :=
  (num_adults + num_children) * cost_per_meal

/-- Proves that the bill for 2 adults and 5 children, with meals costing $3 each, is $21 -/
theorem billys_restaurant_bill :
  calculate_bill 2 5 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_billys_restaurant_bill_l4164_416494


namespace NUMINAMATH_CALUDE_lucas_numbers_l4164_416421

theorem lucas_numbers (a b : ℤ) : 
  (3 * a + 4 * b = 161) → 
  ((a = 17 ∨ b = 17) → (a = 31 ∨ b = 31)) :=
by sorry

end NUMINAMATH_CALUDE_lucas_numbers_l4164_416421


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4164_416470

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (2 - 3 * z) = 9 :=
by
  -- The unique solution is z = -79/3
  use -79/3
  constructor
  · -- Prove that -79/3 satisfies the equation
    sorry
  · -- Prove that any z satisfying the equation must equal -79/3
    sorry

#check sqrt_equation_solution

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4164_416470


namespace NUMINAMATH_CALUDE_function_properties_l4164_416487

noncomputable def f (ω θ : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x + θ)

theorem function_properties (ω θ : ℝ) (h_ω : ω > 0) (h_θ : 0 ≤ θ ∧ θ ≤ π/2)
  (h_intersect : f ω θ 0 = Real.sqrt 3)
  (h_period : ∀ x, f ω θ (x + π) = f ω θ x) :
  (θ = π/6 ∧ ω = 2) ∧
  (∃ x₀ ∈ Set.Icc (π/2) π,
    let y₀ := Real.sqrt 3 / 2
    let x₁ := 2 * x₀ - π/2
    let y₁ := f ω θ x₁
    y₀ = (y₁ + 0) / 2 ∧ (x₀ = 2*π/3 ∨ x₀ = 3*π/4)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4164_416487


namespace NUMINAMATH_CALUDE_exists_n_ratio_f_g_eq_2012_l4164_416425

/-- The number of divisors of n which are perfect squares -/
def f (n : ℕ+) : ℕ := sorry

/-- The number of divisors of n which are perfect cubes -/
def g (n : ℕ+) : ℕ := sorry

/-- There exists a positive integer n such that f(n) / g(n) = 2012 -/
theorem exists_n_ratio_f_g_eq_2012 : ∃ n : ℕ+, (f n : ℚ) / (g n : ℚ) = 2012 := by sorry

end NUMINAMATH_CALUDE_exists_n_ratio_f_g_eq_2012_l4164_416425
