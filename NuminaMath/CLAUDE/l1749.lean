import Mathlib

namespace NUMINAMATH_CALUDE_inequality_system_solution_l1749_174906

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2*x ∧ x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1749_174906


namespace NUMINAMATH_CALUDE_dorothy_age_ratio_l1749_174958

/-- Given Dorothy's sister's age and the condition about their future ages,
    prove that Dorothy is currently 3 times as old as her sister. -/
theorem dorothy_age_ratio (sister_age : ℕ) (dorothy_age : ℕ) : 
  sister_age = 5 →
  dorothy_age + 5 = 2 * (sister_age + 5) →
  dorothy_age / sister_age = 3 := by
  sorry

end NUMINAMATH_CALUDE_dorothy_age_ratio_l1749_174958


namespace NUMINAMATH_CALUDE_parallel_lines_l1749_174959

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if two lines are coincident -/
def coincident (l1 l2 : Line) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ l1.a = k * l2.a ∧ l1.b = k * l2.b ∧ l1.c = k * l2.c

/-- The main theorem -/
theorem parallel_lines (a : ℝ) : 
  let l1 : Line := ⟨a, 3, a^2 - 5⟩
  let l2 : Line := ⟨1, a - 2, 4⟩
  (parallel l1 l2 ∧ ¬coincident l1 l2) ↔ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_l1749_174959


namespace NUMINAMATH_CALUDE_concentric_circles_circumference_difference_l1749_174998

/-- The difference in circumferences of two concentric circles -/
theorem concentric_circles_circumference_difference 
  (inner_diameter : ℝ) 
  (distance_between_circles : ℝ) : 
  inner_diameter = 100 → 
  distance_between_circles = 15 → 
  (inner_diameter + 2 * distance_between_circles) * π - inner_diameter * π = 30 * π := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_circumference_difference_l1749_174998


namespace NUMINAMATH_CALUDE_one_cow_one_bag_days_l1749_174932

/-- Given that 34 cows eat 34 bags of husk in 34 days, 
    prove that one cow will eat one bag of husk in 34 days. -/
theorem one_cow_one_bag_days : 
  ∀ (cows bags days : ℕ), 
  cows = 34 → bags = 34 → days = 34 →
  (cows * bags = cows * days) →
  1 * days = 34 := by
  sorry

end NUMINAMATH_CALUDE_one_cow_one_bag_days_l1749_174932


namespace NUMINAMATH_CALUDE_max_value_theorem_l1749_174971

theorem max_value_theorem (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (x_ge : x ≥ -1)
  (y_ge : y ≥ -2)
  (z_ge : z ≥ -4) :
  (∀ a b c : ℝ, a + b + c = 3 → a ≥ -1 → b ≥ -2 → c ≥ -4 →
    Real.sqrt (4*a + 4) + Real.sqrt (4*b + 8) + Real.sqrt (4*c + 16) ≤
    Real.sqrt (4*x + 4) + Real.sqrt (4*y + 8) + Real.sqrt (4*z + 16)) ∧
  Real.sqrt (4*x + 4) + Real.sqrt (4*y + 8) + Real.sqrt (4*z + 16) = 2 * Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1749_174971


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l1749_174910

/-- Represents the speed of a boat in still water -/
def boat_speed : ℝ := 15

/-- Represents the speed of the stream -/
def stream_speed : ℝ := 10

/-- Represents the downstream distance traveled -/
def downstream_distance : ℝ := 100

/-- Represents the upstream distance traveled -/
def upstream_distance : ℝ := 75

/-- Represents the time taken for downstream travel -/
def downstream_time : ℝ := 4

/-- Represents the time taken for upstream travel -/
def upstream_time : ℝ := 15

theorem stream_speed_calculation :
  (downstream_distance / downstream_time = boat_speed + stream_speed) ∧
  (upstream_distance / upstream_time = boat_speed - stream_speed) →
  stream_speed = 10 := by sorry

end NUMINAMATH_CALUDE_stream_speed_calculation_l1749_174910


namespace NUMINAMATH_CALUDE_bella_steps_l1749_174989

/-- The distance between Bella's and Ella's houses in feet -/
def distance : ℕ := 15840

/-- Bella's speed relative to Ella's -/
def speed_ratio : ℚ := 1 / 3

/-- The number of feet Bella covers in one step -/
def feet_per_step : ℕ := 3

/-- The number of steps Bella takes before meeting Ella -/
def steps_taken : ℕ := 1320

theorem bella_steps :
  distance * speed_ratio / (1 + speed_ratio) / feet_per_step = steps_taken := by
  sorry

end NUMINAMATH_CALUDE_bella_steps_l1749_174989


namespace NUMINAMATH_CALUDE_water_evaporation_per_day_l1749_174979

/-- Proves that given the initial conditions, the amount of water evaporated per day is correct -/
theorem water_evaporation_per_day 
  (initial_water : ℝ) 
  (evaporation_percentage : ℝ) 
  (days : ℕ) 
  (h1 : initial_water = 10) 
  (h2 : evaporation_percentage = 7.000000000000001) 
  (h3 : days = 50) : 
  (initial_water * evaporation_percentage / 100) / days = 0.014000000000000002 := by
  sorry

#check water_evaporation_per_day

end NUMINAMATH_CALUDE_water_evaporation_per_day_l1749_174979


namespace NUMINAMATH_CALUDE_workshop_average_salary_l1749_174912

theorem workshop_average_salary 
  (num_technicians : ℕ)
  (num_total_workers : ℕ)
  (avg_salary_technicians : ℚ)
  (avg_salary_others : ℚ)
  (h1 : num_technicians = 7)
  (h2 : num_total_workers = 22)
  (h3 : avg_salary_technicians = 1000)
  (h4 : avg_salary_others = 780) :
  (num_technicians * avg_salary_technicians + (num_total_workers - num_technicians) * avg_salary_others) / num_total_workers = 850 := by
sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l1749_174912


namespace NUMINAMATH_CALUDE_flock_size_lcm_equals_min_ducks_l1749_174945

/-- Represents the flock size of ducks -/
def duck_flock_size : ℕ := 18

/-- Represents the flock size of seagulls -/
def seagull_flock_size : ℕ := 10

/-- Represents the smallest number of ducks observed -/
def min_ducks_observed : ℕ := 90

/-- Theorem stating that the least common multiple of the flock sizes
    is equal to the smallest number of ducks observed -/
theorem flock_size_lcm_equals_min_ducks :
  Nat.lcm duck_flock_size seagull_flock_size = min_ducks_observed := by
  sorry

end NUMINAMATH_CALUDE_flock_size_lcm_equals_min_ducks_l1749_174945


namespace NUMINAMATH_CALUDE_space_diagonals_of_Q_l1749_174914

/-- A convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  let total_line_segments := Q.vertices.choose 2
  let face_diagonals := 2 * Q.quadrilateral_faces
  total_line_segments - Q.edges - face_diagonals

/-- The specific polyhedron Q described in the problem -/
def Q : ConvexPolyhedron :=
  { vertices := 30
  , edges := 72
  , faces := 44
  , triangular_faces := 30
  , quadrilateral_faces := 14 }

theorem space_diagonals_of_Q :
  space_diagonals Q = 335 := by sorry

end NUMINAMATH_CALUDE_space_diagonals_of_Q_l1749_174914


namespace NUMINAMATH_CALUDE_similar_triangle_lines_count_l1749_174990

/-- A triangle in a 2D plane -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- Predicate to check if a point is inside a triangle -/
def isInside (P : Point) (T : Triangle) : Prop := sorry

/-- A line in a 2D plane -/
structure Line :=
  (point : Point)
  (direction : ℝ × ℝ)

/-- Predicate to check if a line intersects a triangle -/
def intersects (L : Line) (T : Triangle) : Prop := sorry

/-- Predicate to check if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop := sorry

/-- Function to count the number of lines through a point inside a triangle
    that intersect the triangle and form similar triangles -/
def countSimilarTriangleLines (T : Triangle) (P : Point) : ℕ := sorry

/-- Theorem stating that the number of lines through a point inside a triangle
    that intersect the triangle and form similar triangles is 6 -/
theorem similar_triangle_lines_count (T : Triangle) (P : Point) 
  (h : isInside P T) : countSimilarTriangleLines T P = 6 := by sorry

end NUMINAMATH_CALUDE_similar_triangle_lines_count_l1749_174990


namespace NUMINAMATH_CALUDE_quadratic_has_two_real_roots_roots_difference_implies_m_values_l1749_174957

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 - (m-1)*x + m - 2

-- Theorem 1: The quadratic equation always has two real roots
theorem quadratic_has_two_real_roots (m : ℝ) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic x1 m = 0 ∧ quadratic x2 m = 0 :=
sorry

-- Theorem 2: When the difference between the roots is 3, m = 0 or m = 6
theorem roots_difference_implies_m_values :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic x1 0 = 0 ∧ quadratic x2 0 = 0 ∧ |x1 - x2| = 3 ∨
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ quadratic x1 6 = 0 ∧ quadratic x2 6 = 0 ∧ |x1 - x2| = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_two_real_roots_roots_difference_implies_m_values_l1749_174957


namespace NUMINAMATH_CALUDE_fourteen_sided_figure_area_l1749_174943

/-- A fourteen-sided figure constructed on a 1 cm × 1 cm grid -/
structure FourteenSidedFigure where
  /-- The number of full unit squares inside the figure -/
  full_squares : ℕ
  /-- The number of small right-angled triangles along the boundaries -/
  boundary_triangles : ℕ
  /-- The figure has 14 sides -/
  sides : ℕ
  sides_eq : sides = 14

/-- The area of the fourteen-sided figure is 16 cm² -/
theorem fourteen_sided_figure_area (f : FourteenSidedFigure) : 
  f.full_squares + f.boundary_triangles / 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourteen_sided_figure_area_l1749_174943


namespace NUMINAMATH_CALUDE_problem_solution_l1749_174962

-- Statement ①
def statement1 (a b c : ℝ) : Prop :=
  (a > b → c^2 * a > c^2 * b)

-- Statement ②
def statement2 (m : ℝ) : Prop :=
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0)

-- Statement ③
def statement3 (x y : ℝ) : Prop :=
  (x + y = 5 → x^2 - y^2 - 3*x + 7*y = 10)

theorem problem_solution :
  (¬ ∀ a b c : ℝ, ¬statement1 a b c) ∧
  (∀ m : ℝ, ¬(∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) ∧
  ((∀ x y : ℝ, x + y = 5 → x^2 - y^2 - 3*x + 7*y = 10) ∧
   ¬(∀ x y : ℝ, x^2 - y^2 - 3*x + 7*y = 10 → x + y = 5)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1749_174962


namespace NUMINAMATH_CALUDE_quadratic_inequality_minimum_l1749_174955

theorem quadratic_inequality_minimum (b c : ℝ) : 
  (∀ x, (x^2 - (b+2)*x + c < 0) ↔ (2 < x ∧ x < 3)) →
  (∃ min : ℝ, min = 3 ∧ 
    ∀ x > 1, (x^2 - b*x + c) / (x - 1) ≥ min ∧ 
    ∃ x₀ > 1, (x₀^2 - b*x₀ + c) / (x₀ - 1) = min) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_minimum_l1749_174955


namespace NUMINAMATH_CALUDE_horner_method_op_count_for_f_l1749_174922

/-- Horner's method operation count for polynomial evaluation -/
def horner_op_count (coeffs : List ℝ) : ℕ :=
  match coeffs with
  | [] => 0
  | [_] => 0
  | _ :: tail => 2 * (tail.length)

/-- The polynomial f(x) = x^5 + 4x^4 + 3x^3 + 2x^2 + 1 -/
def f_coeffs : List ℝ := [1, 4, 3, 2, 0, 1]

theorem horner_method_op_count_for_f :
  horner_op_count f_coeffs = 8 := by sorry

end NUMINAMATH_CALUDE_horner_method_op_count_for_f_l1749_174922


namespace NUMINAMATH_CALUDE_burger_filler_percentage_l1749_174947

/-- Given a burger with specified total weight and filler weights, 
    calculate the percentage that is not filler -/
theorem burger_filler_percentage 
  (total_weight : ℝ) 
  (vegetable_filler : ℝ) 
  (grain_filler : ℝ) 
  (h1 : total_weight = 180) 
  (h2 : vegetable_filler = 45) 
  (h3 : grain_filler = 15) : 
  (total_weight - (vegetable_filler + grain_filler)) / total_weight = 2/3 := by
sorry

#eval (180 - (45 + 15)) / 180

end NUMINAMATH_CALUDE_burger_filler_percentage_l1749_174947


namespace NUMINAMATH_CALUDE_quadratic_function_determination_l1749_174999

/-- Given real numbers a, b, c, if f(x) = ax^2 + bx + c, g(x) = ax + b, 
    and the maximum value of g(x) is 2 when -1 ≤ x ≤ 1, then f(x) = 2x^2 - 1 -/
theorem quadratic_function_determination (a b c : ℝ) 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^2 + b * x + c)
  (h_g : ∀ x, g x = a * x + b)
  (h_max : ∀ x, -1 ≤ x → x ≤ 1 → g x ≤ 2)
  (h_reaches_max : ∃ x, -1 ≤ x ∧ x ≤ 1 ∧ g x = 2) :
  ∀ x, f x = 2 * x^2 - 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_determination_l1749_174999


namespace NUMINAMATH_CALUDE_sqrt_9800_simplification_l1749_174907

theorem sqrt_9800_simplification : Real.sqrt 9800 = 70 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_9800_simplification_l1749_174907


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l1749_174937

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 5 * x - 2

-- Define the solution set of f(x) > 0
def solution_set (a : ℝ) : Set ℝ := {x | f a x > 0}

-- Define the second quadratic function
def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 5 * x + a^2 - 1

-- State the theorem
theorem quadratic_inequality_problem (a : ℝ) :
  solution_set a = {x | 1/2 < x ∧ x < 2} →
  (a = -2 ∧ {x | g a x > 0} = {x | -3 < x ∧ x < 1/2}) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l1749_174937


namespace NUMINAMATH_CALUDE_modular_home_cost_l1749_174919

/-- Calculates the cost of a modular home given specific module costs and sizes. -/
theorem modular_home_cost 
  (kitchen_size : ℕ) (kitchen_cost : ℕ)
  (bathroom_size : ℕ) (bathroom_cost : ℕ)
  (other_cost_per_sqft : ℕ)
  (total_size : ℕ) (num_bathrooms : ℕ) : 
  kitchen_size = 400 →
  kitchen_cost = 20000 →
  bathroom_size = 150 →
  bathroom_cost = 12000 →
  other_cost_per_sqft = 100 →
  total_size = 2000 →
  num_bathrooms = 2 →
  (kitchen_cost + num_bathrooms * bathroom_cost + 
   (total_size - kitchen_size - num_bathrooms * bathroom_size) * other_cost_per_sqft) = 174000 :=
by sorry

end NUMINAMATH_CALUDE_modular_home_cost_l1749_174919


namespace NUMINAMATH_CALUDE_ratio_p_to_q_l1749_174995

def total_ways : ℕ := 6^24

def ways_p : ℕ := Nat.choose 6 2 * Nat.choose 24 2 * Nat.choose 22 6 * 
                  Nat.choose 16 4 * Nat.choose 12 4 * Nat.choose 8 4 * Nat.choose 4 4

def ways_q : ℕ := Nat.choose 6 2 * Nat.choose 24 3 * Nat.choose 21 3 * 
                  Nat.choose 18 4 * Nat.choose 14 4 * Nat.choose 10 4 * Nat.choose 6 4

def p : ℚ := ways_p / total_ways
def q : ℚ := ways_q / total_ways

theorem ratio_p_to_q : p / q = ways_p / ways_q := by
  sorry

end NUMINAMATH_CALUDE_ratio_p_to_q_l1749_174995


namespace NUMINAMATH_CALUDE_train_speed_problem_l1749_174933

/-- Proves the speed of the second train given the conditions of the problem -/
theorem train_speed_problem (train_length : ℝ) (train1_speed : ℝ) (passing_time : ℝ) :
  train_length = 210 →
  train1_speed = 90 →
  passing_time = 8.64 →
  ∃ train2_speed : ℝ,
    train2_speed = 85 ∧
    (train_length * 2) / passing_time * 3.6 = train1_speed + train2_speed :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1749_174933


namespace NUMINAMATH_CALUDE_weight_of_doubled_cube_l1749_174938

/-- Given a cubical block of metal weighing 8 pounds, proves that another cube of the same metal with sides twice as long will weigh 64 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (h : s > 0) : 
  let original_weight : ℝ := 8
  let original_volume : ℝ := s^3
  let density : ℝ := original_weight / original_volume
  let new_side_length : ℝ := 2 * s
  let new_volume : ℝ := new_side_length^3
  let new_weight : ℝ := density * new_volume
  new_weight = 64 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_doubled_cube_l1749_174938


namespace NUMINAMATH_CALUDE_negative_discriminant_implies_no_real_roots_l1749_174952

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation (α : Type*) [Field α] where
  a : α
  b : α
  c : α

/-- Calculates the discriminant of a quadratic equation -/
def discriminant {α : Type*} [Field α] (eq : QuadraticEquation α) : α :=
  eq.b ^ 2 - 4 * eq.a * eq.c

/-- Represents the property of having real roots -/
def has_real_roots {α : Type*} [Field α] (eq : QuadraticEquation α) : Prop :=
  ∃ x : α, eq.a * x ^ 2 + eq.b * x + eq.c = 0

theorem negative_discriminant_implies_no_real_roots 
  {k : ℝ} (eq : QuadraticEquation ℝ) 
  (h_eq : eq = { a := 3, b := -4 * Real.sqrt 3, c := k }) 
  (h_discr : discriminant eq < 0) : 
  ¬ has_real_roots eq :=
sorry

end NUMINAMATH_CALUDE_negative_discriminant_implies_no_real_roots_l1749_174952


namespace NUMINAMATH_CALUDE_intersection_count_l1749_174936

-- Define the equations
def eq1 (x y : ℝ) : Prop := (x + 2*y - 10) * (x - 4*y + 8) = 0
def eq2 (x y : ℝ) : Prop := (2*x - y - 1) * (5*x + 3*y - 15) = 0

-- Define a function to count distinct intersection points
noncomputable def count_intersections : ℕ :=
  -- Implementation details are omitted
  sorry

-- Theorem statement
theorem intersection_count : count_intersections = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l1749_174936


namespace NUMINAMATH_CALUDE_target_shopping_total_l1749_174949

/-- The total amount spent by Christy and Tanya at Target -/
def total_spent (face_moisturizer_price : ℕ) (body_lotion_price : ℕ) 
  (face_moisturizer_count : ℕ) (body_lotion_count : ℕ) : ℕ :=
  let tanya_spent := face_moisturizer_price * face_moisturizer_count + 
                     body_lotion_price * body_lotion_count
  2 * tanya_spent

/-- Theorem stating the total amount spent by Christy and Tanya -/
theorem target_shopping_total : 
  total_spent 50 60 2 4 = 1020 := by
  sorry

#eval total_spent 50 60 2 4

end NUMINAMATH_CALUDE_target_shopping_total_l1749_174949


namespace NUMINAMATH_CALUDE_ball_arrangement_count_l1749_174904

/-- The number of ways to arrange 8 balls in a row, with 5 red balls (3 of which must be consecutive) and 3 white balls. -/
def ball_arrangements : ℕ := 30

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The number of red balls -/
def red_balls : ℕ := 5

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of consecutive red balls -/
def consecutive_red_balls : ℕ := 3

theorem ball_arrangement_count : 
  ball_arrangements = (Nat.choose (total_balls - consecutive_red_balls + 1) white_balls) * 
                      (Nat.choose (total_balls - white_balls - consecutive_red_balls + 1) 1) / 
                      (Nat.factorial (red_balls - consecutive_red_balls)) :=
sorry

end NUMINAMATH_CALUDE_ball_arrangement_count_l1749_174904


namespace NUMINAMATH_CALUDE_statement_analysis_l1749_174916

-- Define the types of statements
inductive StatementType
  | Universal
  | Existential

-- Define a structure to represent a statement
structure Statement where
  content : String
  type : StatementType
  isTrue : Bool

-- Define the statements
def statement1 : Statement := {
  content := "The diagonals of a square are perpendicular bisectors of each other",
  type := StatementType.Universal,
  isTrue := true
}

def statement2 : Statement := {
  content := "All Chinese people speak Chinese",
  type := StatementType.Universal,
  isTrue := false
}

def statement3 : Statement := {
  content := "Some numbers are greater than their squares",
  type := StatementType.Existential,
  isTrue := true
}

def statement4 : Statement := {
  content := "Some real numbers have irrational square roots",
  type := StatementType.Existential,
  isTrue := true
}

-- Theorem to prove
theorem statement_analysis : 
  (statement1.type = StatementType.Universal ∧ statement1.isTrue) ∧
  (statement2.type = StatementType.Universal ∧ ¬statement2.isTrue) ∧
  (statement3.type = StatementType.Existential ∧ statement3.isTrue) ∧
  (statement4.type = StatementType.Existential ∧ statement4.isTrue) := by
  sorry


end NUMINAMATH_CALUDE_statement_analysis_l1749_174916


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l1749_174968

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (45 * x + 13) % 17 = 5 % 17 ∧
  ∀ (y : ℕ), y > 0 ∧ (45 * y + 13) % 17 = 5 % 17 → x ≤ y :=
by
  use 11
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l1749_174968


namespace NUMINAMATH_CALUDE_cone_height_is_sqrt_3_l1749_174982

-- Define the cone structure
structure Cone where
  base_radius : ℝ
  height : ℝ
  slant_height : ℝ

-- Define the property of the cone's lateral surface
def lateral_surface_is_semicircle (c : Cone) : Prop :=
  c.slant_height = 2

-- Theorem statement
theorem cone_height_is_sqrt_3 (c : Cone) 
  (h_semicircle : lateral_surface_is_semicircle c) : 
  c.height = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_cone_height_is_sqrt_3_l1749_174982


namespace NUMINAMATH_CALUDE_complex_abs_power_six_l1749_174902

theorem complex_abs_power_six : Complex.abs ((2 : ℂ) + 2 * Complex.I * Real.sqrt 3) ^ 6 = 4096 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_power_six_l1749_174902


namespace NUMINAMATH_CALUDE_fraction_comparison_geometric_sum_comparison_l1749_174960

theorem fraction_comparison (α β : ℝ) (hα : α = 1.00000000004) (hβ : β = 1.00000000002) :
  (1 + β) / (1 + β + β^2) > (1 + α) / (1 + α + α^2) := by sorry

theorem geometric_sum_comparison {a b : ℝ} {n : ℕ} (hab : a > b) (hb : b > 0) (hn : n > 0) :
  (b^n - 1) / (b^(n+1) - 1) > (a^n - 1) / (a^(n+1) - 1) := by sorry

end NUMINAMATH_CALUDE_fraction_comparison_geometric_sum_comparison_l1749_174960


namespace NUMINAMATH_CALUDE_no_integer_coefficients_l1749_174900

theorem no_integer_coefficients : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_coefficients_l1749_174900


namespace NUMINAMATH_CALUDE_basketball_score_increase_l1749_174953

theorem basketball_score_increase (junior_score : ℕ) (total_score : ℕ) 
  (h1 : junior_score = 260) 
  (h2 : total_score = 572) : 
  (((total_score - junior_score) : ℚ) / junior_score) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_increase_l1749_174953


namespace NUMINAMATH_CALUDE_museum_paintings_l1749_174988

theorem museum_paintings (initial : ℕ) (left : ℕ) (removed : ℕ) :
  initial = 1795 →
  left = 1322 →
  removed = initial - left →
  removed = 473 :=
by sorry

end NUMINAMATH_CALUDE_museum_paintings_l1749_174988


namespace NUMINAMATH_CALUDE_not_prime_n_l1749_174996

theorem not_prime_n (p a b c n : ℕ) : 
  Nat.Prime p → 
  0 < a → 0 < b → 0 < c → 0 < n →
  a < p → b < p → c < p →
  p^2 ∣ (a + (n-1) * b) →
  p^2 ∣ (b + (n-1) * c) →
  p^2 ∣ (c + (n-1) * a) →
  ¬(Nat.Prime n) :=
by sorry


end NUMINAMATH_CALUDE_not_prime_n_l1749_174996


namespace NUMINAMATH_CALUDE_f_inequality_l1749_174901

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * (x - 1) / (x + 1)

theorem f_inequality (a : ℝ) : 
  (∀ x : ℝ, x > 0 ∧ x ≠ 1 → ((x + 1) * Real.log x + 2 * a) / ((x + 1)^2) < Real.log x / (x - 1)) ↔ 
  a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_f_inequality_l1749_174901


namespace NUMINAMATH_CALUDE_largest_inexpressible_is_19_l1749_174921

/-- Represents the value of a coin in soldi -/
inductive Coin : Type
| five : Coin
| six : Coin

/-- Checks if a natural number can be expressed as a sum of multiples of 5 and 6 -/
def canExpress (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 5 * a + 6 * b

/-- The largest value that cannot be expressed as a sum of multiples of 5 and 6 -/
def largestInexpressible : ℕ := 19

theorem largest_inexpressible_is_19 :
  largestInexpressible = 19 ∧
  ¬(canExpress largestInexpressible) ∧
  ∀ n : ℕ, n > largestInexpressible → n ≤ 50 → canExpress n :=
by sorry

end NUMINAMATH_CALUDE_largest_inexpressible_is_19_l1749_174921


namespace NUMINAMATH_CALUDE_running_time_difference_l1749_174940

/-- The time difference for running 5 miles between new and old shoes -/
theorem running_time_difference 
  (old_shoe_time : ℕ) -- Time to run one mile in old shoes
  (new_shoe_time : ℕ) -- Time to run one mile in new shoes
  (distance : ℕ) -- Distance to run in miles
  (h1 : old_shoe_time = 10)
  (h2 : new_shoe_time = 13)
  (h3 : distance = 5) :
  new_shoe_time * distance - old_shoe_time * distance = 15 :=
by sorry

end NUMINAMATH_CALUDE_running_time_difference_l1749_174940


namespace NUMINAMATH_CALUDE_rotation_exists_l1749_174994

-- Define a 3D point
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle in 3D space
structure Triangle3D where
  A : Point3D
  O : Point3D
  B : Point3D

-- Define congruence for triangles
def congruent (t1 t2 : Triangle3D) : Prop :=
  (t1.A.x - t1.O.x)^2 + (t1.A.y - t1.O.y)^2 + (t1.A.z - t1.O.z)^2 =
    (t2.A.x - t2.O.x)^2 + (t2.A.y - t2.O.y)^2 + (t2.A.z - t2.O.z)^2 ∧
  (t1.B.x - t1.O.x)^2 + (t1.B.y - t1.O.y)^2 + (t1.B.z - t1.O.z)^2 =
    (t2.B.x - t2.O.x)^2 + (t2.B.y - t2.O.y)^2 + (t2.B.z - t2.O.z)^2

-- Define when two triangles are not in the same plane
def not_coplanar (t1 t2 : Triangle3D) : Prop :=
  ¬ ∃ (a b c d : ℝ),
    a * (t1.A.x - t1.O.x) + b * (t1.A.y - t1.O.y) + c * (t1.A.z - t1.O.z) + d = 0 ∧
    a * (t1.B.x - t1.O.x) + b * (t1.B.y - t1.O.y) + c * (t1.B.z - t1.O.z) + d = 0 ∧
    a * (t2.A.x - t2.O.x) + b * (t2.A.y - t2.O.y) + c * (t2.A.z - t2.O.z) + d = 0 ∧
    a * (t2.B.x - t2.O.x) + b * (t2.B.y - t2.O.y) + c * (t2.B.z - t2.O.z) + d = 0

-- Define rotation in 3D space
structure Rotation3D where
  axis : Point3D
  angle : ℝ

-- Theorem statement
theorem rotation_exists (t1 t2 : Triangle3D)
  (h1 : congruent t1 t2)
  (h2 : t1.O = t2.O)
  (h3 : not_coplanar t1 t2) :
  ∃ (r : Rotation3D), r.axis.x * (t1.A.x - t1.O.x) + r.axis.y * (t1.A.y - t1.O.y) + r.axis.z * (t1.A.z - t1.O.z) = 0 ∧
                      r.axis.x * (t1.B.x - t1.O.x) + r.axis.y * (t1.B.y - t1.O.y) + r.axis.z * (t1.B.z - t1.O.z) = 0 :=
by sorry

end NUMINAMATH_CALUDE_rotation_exists_l1749_174994


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1749_174969

/-- The line y - 1 = k(x - 1) intersects the circle x^2 + y^2 - 2y = 0 for any real number k -/
theorem line_intersects_circle (k : ℝ) : ∃ (x y : ℝ), 
  (y - 1 = k * (x - 1)) ∧ (x^2 + y^2 - 2*y = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1749_174969


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l1749_174980

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 8*x + 7 = 0 ↔ (x + 4)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l1749_174980


namespace NUMINAMATH_CALUDE_fourth_machine_works_twelve_hours_l1749_174993

/-- Represents a factory with machines producing material. -/
structure Factory where
  num_original_machines : ℕ
  hours_per_day_original : ℕ
  production_rate : ℕ
  price_per_kg : ℕ
  total_revenue : ℕ

/-- Calculates the hours worked by the fourth machine. -/
def fourth_machine_hours (f : Factory) : ℕ :=
  let original_production := f.num_original_machines * f.hours_per_day_original * f.production_rate
  let original_revenue := original_production * f.price_per_kg
  let fourth_machine_revenue := f.total_revenue - original_revenue
  let fourth_machine_production := fourth_machine_revenue / f.price_per_kg
  fourth_machine_production / f.production_rate

/-- Theorem stating the fourth machine works 12 hours a day. -/
theorem fourth_machine_works_twelve_hours (f : Factory) 
  (h1 : f.num_original_machines = 3)
  (h2 : f.hours_per_day_original = 23)
  (h3 : f.production_rate = 2)
  (h4 : f.price_per_kg = 50)
  (h5 : f.total_revenue = 8100) :
  fourth_machine_hours f = 12 := by
  sorry

end NUMINAMATH_CALUDE_fourth_machine_works_twelve_hours_l1749_174993


namespace NUMINAMATH_CALUDE_lowest_price_per_component_l1749_174986

/-- The lowest price per component that covers all costs for a computer manufacturer --/
theorem lowest_price_per_component 
  (cost_per_component : ℝ) 
  (shipping_cost_per_unit : ℝ) 
  (fixed_monthly_costs : ℝ) 
  (components_per_month : ℕ) 
  (h1 : cost_per_component = 80)
  (h2 : shipping_cost_per_unit = 7)
  (h3 : fixed_monthly_costs = 16500)
  (h4 : components_per_month = 150) : 
  ∃ (price : ℝ), price = 197 ∧ 
    price * (components_per_month : ℝ) = 
      (cost_per_component + shipping_cost_per_unit) * (components_per_month : ℝ) + fixed_monthly_costs :=
by sorry

end NUMINAMATH_CALUDE_lowest_price_per_component_l1749_174986


namespace NUMINAMATH_CALUDE_min_n_is_15_l1749_174923

/-- A type representing the vertices of a regular 9-sided polygon -/
inductive Vertex : Type
  | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9

/-- A function type representing an assignment of integers to vertices -/
def Assignment := Vertex → Fin 9

/-- Predicate to check if an assignment is valid (each integer used once) -/
def is_valid_assignment (f : Assignment) : Prop :=
  ∀ i j : Vertex, i ≠ j → f i ≠ f j

/-- Function to get the next vertex in cyclic order -/
def next_vertex : Vertex → Vertex
  | Vertex.v1 => Vertex.v2
  | Vertex.v2 => Vertex.v3
  | Vertex.v3 => Vertex.v4
  | Vertex.v4 => Vertex.v5
  | Vertex.v5 => Vertex.v6
  | Vertex.v6 => Vertex.v7
  | Vertex.v7 => Vertex.v8
  | Vertex.v8 => Vertex.v9
  | Vertex.v9 => Vertex.v1

/-- Predicate to check if the sum of any three consecutive vertices does not exceed n -/
def satisfies_sum_condition (f : Assignment) (n : ℕ) : Prop :=
  ∀ v : Vertex, (f v).val + 1 + (f (next_vertex v)).val + 1 + (f (next_vertex (next_vertex v))).val + 1 ≤ n

/-- The main theorem: the minimum value of n is 15 -/
theorem min_n_is_15 :
  ∃ (f : Assignment), is_valid_assignment f ∧ satisfies_sum_condition f 15 ∧
  ∀ (m : ℕ), m < 15 → ¬∃ (g : Assignment), is_valid_assignment g ∧ satisfies_sum_condition g m :=
sorry

end NUMINAMATH_CALUDE_min_n_is_15_l1749_174923


namespace NUMINAMATH_CALUDE_condition_implies_increasing_l1749_174905

def IsIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem condition_implies_increasing (a : ℕ → ℝ) 
    (h : ∀ n : ℕ, a (n + 1) > |a n|) : 
  IsIncreasing a := by
  sorry

end NUMINAMATH_CALUDE_condition_implies_increasing_l1749_174905


namespace NUMINAMATH_CALUDE_only_shanxi_spirit_census_l1749_174961

-- Define the survey types
inductive SurveyType
  | Census
  | Sample

-- Define the survey options
inductive SurveyOption
  | ArtilleryShells
  | TVRatings
  | FishSpecies
  | ShanxiSpiritAwareness

-- Function to determine the appropriate survey type for each option
def appropriateSurveyType (option : SurveyOption) : SurveyType :=
  match option with
  | SurveyOption.ArtilleryShells => SurveyType.Sample
  | SurveyOption.TVRatings => SurveyType.Sample
  | SurveyOption.FishSpecies => SurveyType.Sample
  | SurveyOption.ShanxiSpiritAwareness => SurveyType.Census

-- Theorem stating that only ShanxiSpiritAwareness is suitable for a census survey
theorem only_shanxi_spirit_census :
  ∀ (option : SurveyOption),
    appropriateSurveyType option = SurveyType.Census ↔ option = SurveyOption.ShanxiSpiritAwareness :=
by
  sorry


end NUMINAMATH_CALUDE_only_shanxi_spirit_census_l1749_174961


namespace NUMINAMATH_CALUDE_max_unique_sundaes_l1749_174920

/-- The number of ice cream flavors --/
def num_flavors : ℕ := 8

/-- The number of flavors that must be served together --/
def num_paired_flavors : ℕ := 2

/-- The number of distinct choices after pairing --/
def num_choices : ℕ := num_flavors - num_paired_flavors + 1

/-- The number of scoops in a sundae --/
def scoops_per_sundae : ℕ := 2

theorem max_unique_sundaes :
  (Nat.choose (num_choices - 1) (scoops_per_sundae - 1)) + 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_unique_sundaes_l1749_174920


namespace NUMINAMATH_CALUDE_factorization_equality_l1749_174997

theorem factorization_equality (y : ℝ) : 49 - 16 * y^2 + 8 * y = (7 - 4 * y) * (7 + 4 * y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1749_174997


namespace NUMINAMATH_CALUDE_snail_movement_bound_l1749_174963

/-- Represents the movement of a snail over time -/
structure SnailMovement where
  /-- The total observation time in minutes -/
  total_time : ℝ
  /-- The movement function: time → distance -/
  movement : ℝ → ℝ
  /-- Ensures the movement is non-negative -/
  non_negative : ∀ t, 0 ≤ movement t
  /-- Ensures the movement is monotonically increasing -/
  monotone : ∀ t₁ t₂, t₁ ≤ t₂ → movement t₁ ≤ movement t₂

/-- The observation condition: for any 1-minute interval, the snail moves exactly 1 meter -/
def observation_condition (sm : SnailMovement) : Prop :=
  ∀ t, 0 ≤ t ∧ t + 1 ≤ sm.total_time → sm.movement (t + 1) - sm.movement t = 1

/-- The theorem statement -/
theorem snail_movement_bound (sm : SnailMovement) 
    (h_time : sm.total_time = 6)
    (h_obs : observation_condition sm) :
    sm.movement sm.total_time ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_snail_movement_bound_l1749_174963


namespace NUMINAMATH_CALUDE_isosceles_triangle_determination_l1749_174946

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is isosceles with AB = AC -/
def isIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2

/-- The incenter of a triangle -/
def incenter (t : Triangle) : Point :=
  sorry

/-- The centroid of a triangle -/
def centroid (t : Triangle) : Point :=
  sorry

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point :=
  sorry

theorem isosceles_triangle_determination
  (I M H : Point) :
  ∃! (t : Triangle), isIsosceles t ∧
    incenter t = I ∧
    centroid t = M ∧
    orthocenter t = H :=
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_determination_l1749_174946


namespace NUMINAMATH_CALUDE_quadratic_polynomial_problem_l1749_174964

theorem quadratic_polynomial_problem (p : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c) →
  (∀ x, (x - 2) * (x + 2) * (x - 9) ∣ (p x)^3 - x) →
  p 14 = -36 / 79 := by
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_problem_l1749_174964


namespace NUMINAMATH_CALUDE_triangle_radii_inequality_l1749_174903

theorem triangle_radii_inequality (a b c r r_a r_b r_c : ℝ) 
    (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
    (h_inradius : r = (a * b * c) / (4 * (a + b + c) * (a * b + b * c + c * a - a^2 - b^2 - c^2)^(1/2)))
    (h_exradius_a : r_a = (a * (b + c - a)) / (4 * (a + b + c) * (a * b + b * c + c * a - a^2 - b^2 - c^2)^(1/2)))
    (h_exradius_b : r_b = (b * (c + a - b)) / (4 * (a + b + c) * (a * b + b * c + c * a - a^2 - b^2 - c^2)^(1/2)))
    (h_exradius_c : r_c = (c * (a + b - c)) / (4 * (a + b + c) * (a * b + b * c + c * a - a^2 - b^2 - c^2)^(1/2))) :
  (a + b + c) / (a^2 + b^2 + c^2)^(1/2) ≤ 2 * (r_a^2 + r_b^2 + r_c^2)^(1/2) / (r_a + r_b + r_c - 3 * r) := by
sorry

end NUMINAMATH_CALUDE_triangle_radii_inequality_l1749_174903


namespace NUMINAMATH_CALUDE_special_line_equation_l1749_174925

/-- A line passing through point A(-3, 4) with x-intercept twice the y-intercept -/
structure SpecialLine where
  -- The slope-intercept form of the line: y = mx + b
  slope : ℝ
  y_intercept : ℝ
  -- The line passes through (-3, 4)
  point_condition : 4 = slope * (-3) + y_intercept
  -- The x-intercept is twice the y-intercept
  intercept_condition : -2 * y_intercept = y_intercept / slope

/-- The equation of the special line is either 3y + 4x = 0 or 2x - y - 5 = 0 -/
theorem special_line_equation (L : SpecialLine) :
  (3 * L.slope + 4 = 0 ∧ 3 * L.y_intercept = 0) ∨
  (2 = L.slope ∧ -5 = L.y_intercept) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l1749_174925


namespace NUMINAMATH_CALUDE_problem_solution_l1749_174970

theorem problem_solution : (-1 : ℚ)^51 + 2^(4^2 + 5^2 - 7^2) = -127/128 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1749_174970


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1749_174909

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 2 = (1 : ℝ) / 4 →
  a 2 * a 8 = 4 * (a 5 - 1) →
  a 4 + a 5 + a 6 + a 7 + a 8 = 31 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1749_174909


namespace NUMINAMATH_CALUDE_french_exam_vocab_study_l1749_174915

/-- Represents the French exam vocabulary problem -/
theorem french_exam_vocab_study (total_words : ℕ) (recall_rate : ℚ) (guess_rate : ℚ) (target_score : ℚ) :
  let min_words : ℕ := 712
  total_words = 800 ∧ recall_rate = 1 ∧ guess_rate = 1/10 ∧ target_score = 9/10 →
  (↑min_words : ℚ) + guess_rate * (total_words - min_words) ≥ target_score * total_words ∧
  ∀ (x : ℕ), x < min_words →
    (↑x : ℚ) + guess_rate * (total_words - x) < target_score * total_words :=
by sorry

end NUMINAMATH_CALUDE_french_exam_vocab_study_l1749_174915


namespace NUMINAMATH_CALUDE_cara_card_is_five_l1749_174942

def is_valid_sequence (a b c d : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ c < d ∧ a + b + c + d = 20

def alan_statement (a : ℕ) : Prop :=
  ∃ b c d, is_valid_sequence a b c d ∧
  ∃ b' c' d', b' ≠ b ∧ is_valid_sequence a b' c' d'

def bella_statement (a b : ℕ) : Prop :=
  ∃ c d, is_valid_sequence a b c d ∧
  ∃ c' d', c' ≠ c ∧ is_valid_sequence a b c' d'

def cara_statement (a b c : ℕ) : Prop :=
  ∃ d, is_valid_sequence a b c d ∧
  ∃ d', d' ≠ d ∧ is_valid_sequence a b c d'

def david_statement (a b c d : ℕ) : Prop :=
  is_valid_sequence a b c d ∧
  ∃ a' b' c', a' ≠ a ∧ is_valid_sequence a' b' c' d

theorem cara_card_is_five :
  ∀ a b c d : ℕ,
    is_valid_sequence a b c d →
    alan_statement a →
    bella_statement a b →
    cara_statement a b c →
    david_statement a b c d →
    c = 5 := by
  sorry

end NUMINAMATH_CALUDE_cara_card_is_five_l1749_174942


namespace NUMINAMATH_CALUDE_quadratic_residue_minus_one_l1749_174954

theorem quadratic_residue_minus_one (p : Nat) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  (∃ x : Nat, x^2 ≡ -1 [ZMOD p]) ↔ p ≡ 1 [ZMOD 4] := by
  sorry

end NUMINAMATH_CALUDE_quadratic_residue_minus_one_l1749_174954


namespace NUMINAMATH_CALUDE_part_one_part_two_l1749_174939

-- Define propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem part_one (x : ℝ) (h : p x 1 ∧ q x) : 2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : ∀ x, ¬(p x a) → ¬(q x)) 
  (h_not_necessary : ∃ x, ¬(p x a) ∧ q x) : 1 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1749_174939


namespace NUMINAMATH_CALUDE_inequality_implies_lower_bound_l1749_174908

theorem inequality_implies_lower_bound (a : ℝ) : 
  (∀ x : ℝ, x > 0 → x / (x^2 + 3*x + 1) ≤ a) → a ≥ 1/5 :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_lower_bound_l1749_174908


namespace NUMINAMATH_CALUDE_pyramid_fifth_face_sum_l1749_174928

/-- Represents a labeling of a square-based pyramid -/
structure PyramidLabeling where
  vertices : Fin 5 → Nat
  sum_to_15 : (vertices 0) + (vertices 1) + (vertices 2) + (vertices 3) + (vertices 4) = 15
  all_different : ∀ i j, i ≠ j → vertices i ≠ vertices j

/-- Represents the sums of faces in the pyramid -/
structure FaceSums (l : PyramidLabeling) where
  sums : Fin 5 → Nat
  four_given_sums : {7, 8, 9, 10} ⊆ (Finset.image sums Finset.univ)

theorem pyramid_fifth_face_sum (l : PyramidLabeling) (s : FaceSums l) :
  ∃ i, s.sums i = 13 :=
sorry

end NUMINAMATH_CALUDE_pyramid_fifth_face_sum_l1749_174928


namespace NUMINAMATH_CALUDE_power_two_33_mod_9_l1749_174976

theorem power_two_33_mod_9 : 2^33 % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_power_two_33_mod_9_l1749_174976


namespace NUMINAMATH_CALUDE_second_arrangement_column_size_l1749_174984

/-- Represents a group of people that can be arranged in columns. -/
structure PeopleGroup where
  /-- The total number of people in the group -/
  total : ℕ
  /-- The number of columns formed when 30 people stand in each column -/
  columns_with_30 : ℕ
  /-- The number of columns formed in the second arrangement -/
  columns_in_second : ℕ
  /-- Ensures that 30 people per column forms the specified number of columns -/
  h_first_arrangement : total = 30 * columns_with_30

/-- 
Given a group of people where 30 people per column forms 16 columns,
if the same group is rearranged into 12 columns,
then there will be 40 people in each column of the second arrangement.
-/
theorem second_arrangement_column_size (g : PeopleGroup) 
    (h_16_columns : g.columns_with_30 = 16)
    (h_12_columns : g.columns_in_second = 12) :
    g.total / g.columns_in_second = 40 := by
  sorry

end NUMINAMATH_CALUDE_second_arrangement_column_size_l1749_174984


namespace NUMINAMATH_CALUDE_linear_function_value_l1749_174991

theorem linear_function_value (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : f 1 = 5) 
  (h2 : f 2 = 8) 
  (h3 : f 3 = 11) 
  (h_linear : ∀ x, f x = a * x + b) : 
  f 4 = 14 := by
sorry

end NUMINAMATH_CALUDE_linear_function_value_l1749_174991


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1749_174924

-- Define the hyperbola
structure Hyperbola where
  center : ℝ × ℝ := (0, 0)
  focus_on_y_axis : Bool
  semi_minor_axis : ℝ
  eccentricity : ℝ

-- Define the asymptote equation type
structure AsymptoticEquation where
  slope : ℝ

-- Theorem statement
theorem hyperbola_asymptotes 
  (h : Hyperbola) 
  (h_center : h.center = (0, 0))
  (h_focus : h.focus_on_y_axis = true)
  (h_semi_minor : h.semi_minor_axis = 4 * Real.sqrt 2)
  (h_eccentricity : h.eccentricity = 3) :
  ∃ (eq : AsymptoticEquation), eq.slope = Real.sqrt 2 / 4 ∨ eq.slope = -(Real.sqrt 2 / 4) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1749_174924


namespace NUMINAMATH_CALUDE_least_number_of_cans_l1749_174975

def maaza_liters : ℕ := 40
def pepsi_liters : ℕ := 144
def sprite_liters : ℕ := 368

theorem least_number_of_cans : 
  ∃ (can_size : ℕ), 
    can_size > 0 ∧
    maaza_liters % can_size = 0 ∧
    pepsi_liters % can_size = 0 ∧
    sprite_liters % can_size = 0 ∧
    (maaza_liters / can_size + pepsi_liters / can_size + sprite_liters / can_size = 69) ∧
    ∀ (other_size : ℕ), 
      other_size > 0 →
      maaza_liters % other_size = 0 →
      pepsi_liters % other_size = 0 →
      sprite_liters % other_size = 0 →
      (maaza_liters / other_size + pepsi_liters / other_size + sprite_liters / other_size ≥ 69) :=
by
  sorry

end NUMINAMATH_CALUDE_least_number_of_cans_l1749_174975


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1749_174911

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the third and fourth terms is 1/2 -/
def third_fourth_sum (a : ℕ → ℚ) : Prop :=
  a 3 + a 4 = 1/2

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a → third_fourth_sum a → a 1 + a 6 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1749_174911


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1749_174929

/-- Given two vectors a and b in ℝ², prove that if a = (1, 2) and b = (m, 1) are perpendicular, then m = -2 -/
theorem perpendicular_vectors_m_value (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![m, 1]
  (∀ i, i < 2 → a i * b i = 0) → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1749_174929


namespace NUMINAMATH_CALUDE_min_value_fraction_l1749_174972

theorem min_value_fraction (a b : ℝ) (h1 : a > 2*b) (h2 : b > 0) :
  (a^4 + 1) / (b * (a - 2*b)) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1749_174972


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l1749_174934

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l1749_174934


namespace NUMINAMATH_CALUDE_base_k_conversion_uniqueness_l1749_174941

theorem base_k_conversion_uniqueness :
  ∃! (k : ℕ), k ≥ 4 ∧ 1 * k^2 + 3 * k + 2 = 30 := by sorry

end NUMINAMATH_CALUDE_base_k_conversion_uniqueness_l1749_174941


namespace NUMINAMATH_CALUDE_proportion_solution_l1749_174951

theorem proportion_solution (x : ℝ) : (0.75 / x = 5 / 9) → x = 1.35 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l1749_174951


namespace NUMINAMATH_CALUDE_sodium_atom_diameter_scientific_notation_l1749_174973

theorem sodium_atom_diameter_scientific_notation :
  ∃ (n : ℤ), (0.0000000599 : ℝ) = 5.99 * (10 : ℝ) ^ n → n = -8 := by
  sorry

end NUMINAMATH_CALUDE_sodium_atom_diameter_scientific_notation_l1749_174973


namespace NUMINAMATH_CALUDE_property_tax_increase_l1749_174974

/-- Represents the property tax increase in Township K --/
theorem property_tax_increase 
  (tax_rate : ℝ) 
  (initial_value : ℝ) 
  (new_value : ℝ) 
  (h1 : tax_rate = 0.1)
  (h2 : initial_value = 20000)
  (h3 : new_value = 28000) : 
  new_value * tax_rate - initial_value * tax_rate = 800 := by
  sorry

#check property_tax_increase

end NUMINAMATH_CALUDE_property_tax_increase_l1749_174974


namespace NUMINAMATH_CALUDE_sprint_jog_difference_value_l1749_174966

/-- The difference between Darnel's total sprinting distance and total jogging distance -/
def sprint_jog_difference : ℝ :=
  let sprint1 := 0.8932
  let sprint2 := 0.9821
  let sprint3 := 1.2534
  let jog1 := 0.7683
  let jog2 := 0.4356
  let jog3 := 0.6549
  (sprint1 + sprint2 + sprint3) - (jog1 + jog2 + jog3)

/-- Theorem stating that the difference between Darnel's total sprinting distance and total jogging distance is 1.2699 laps -/
theorem sprint_jog_difference_value : sprint_jog_difference = 1.2699 := by
  sorry

end NUMINAMATH_CALUDE_sprint_jog_difference_value_l1749_174966


namespace NUMINAMATH_CALUDE_symmetry_condition_l1749_174944

/-- A curve in the xy-plane represented by the equation x^2 + y^2 + Dx + Ey + F = 0 -/
structure Curve (D E F : ℝ) where
  condition : D^2 + E^2 - 4*F > 0

/-- Predicate for a curve being symmetric about the line y = x -/
def is_symmetric_about_y_eq_x (c : Curve D E F) : Prop :=
  D = E

/-- Theorem stating the condition for symmetry about y = x -/
theorem symmetry_condition (D E F : ℝ) (c : Curve D E F) :
  is_symmetric_about_y_eq_x c ↔ D = E :=
sorry

end NUMINAMATH_CALUDE_symmetry_condition_l1749_174944


namespace NUMINAMATH_CALUDE_smallest_to_large_square_area_ratio_l1749_174948

/-- Represents a square with a given side length -/
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The area of a square -/
def Square.area (s : Square) : ℝ := s.side_length ^ 2

theorem smallest_to_large_square_area_ratio :
  ∀ (large : Square),
  ∃ (middle smallest : Square),
  (middle.side_length = large.side_length / 2) ∧
  (smallest.side_length = middle.side_length / 2) →
  smallest.area / large.area = 1 / 16 :=
by
  sorry

#check smallest_to_large_square_area_ratio

end NUMINAMATH_CALUDE_smallest_to_large_square_area_ratio_l1749_174948


namespace NUMINAMATH_CALUDE_parallelepiped_properties_l1749_174927

/-- Properties of a parallelepiped -/
structure Parallelepiped where
  projection : ℝ
  height : ℝ
  rhombus_area : ℝ
  rhombus_diagonal : ℝ

/-- Calculate the lateral surface area of the parallelepiped -/
def lateral_surface_area (p : Parallelepiped) : ℝ := sorry

/-- Calculate the volume of the parallelepiped -/
def volume (p : Parallelepiped) : ℝ := sorry

/-- Theorem stating the lateral surface area and volume of the given parallelepiped -/
theorem parallelepiped_properties (p : Parallelepiped) 
  (h1 : p.projection = 5)
  (h2 : p.height = 12)
  (h3 : p.rhombus_area = 24)
  (h4 : p.rhombus_diagonal = 8) :
  lateral_surface_area p = 260 ∧ volume p = 312 := by sorry

end NUMINAMATH_CALUDE_parallelepiped_properties_l1749_174927


namespace NUMINAMATH_CALUDE_max_excellent_boys_100_l1749_174950

/-- Represents a person with height and weight -/
structure Person where
  height : ℝ
  weight : ℝ

/-- Defines the "not worse than" relation between two people -/
def notWorseThan (a b : Person) : Prop :=
  a.height > b.height ∨ a.weight > b.weight

/-- Defines an "excellent boy" as someone who is not worse than all others -/
def excellentBoy (p : Person) (group : Finset Person) : Prop :=
  ∀ q ∈ group, p ≠ q → notWorseThan p q

/-- The main theorem: The maximum number of excellent boys in a group of 100 is 100 -/
theorem max_excellent_boys_100 :
  ∃ (group : Finset Person), group.card = 100 ∧
  ∃ (excellent : Finset Person), excellent ⊆ group ∧ excellent.card = 100 ∧
  ∀ p ∈ excellent, excellentBoy p group :=
sorry

end NUMINAMATH_CALUDE_max_excellent_boys_100_l1749_174950


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l1749_174935

theorem book_arrangement_proof :
  let total_books : ℕ := 11
  let geometry_books : ℕ := 5
  let number_theory_books : ℕ := 6
  Nat.choose total_books geometry_books = 462 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l1749_174935


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l1749_174913

/-- Given an inverse proportion function y = (k+1)/x passing through the point (1, -2),
    prove that the value of k is -3. -/
theorem inverse_proportion_k_value (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, x ≠ 0 → f x = (k + 1) / x) ∧ f 1 = -2) → k = -3 :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l1749_174913


namespace NUMINAMATH_CALUDE_big_crash_frequency_is_20_l1749_174978

/-- Represents the frequency of big crashes in seconds -/
def big_crash_frequency (total_accidents : ℕ) (total_time : ℕ) (collision_frequency : ℕ) : ℕ :=
  let regular_collisions := total_time / collision_frequency
  let big_crashes := total_accidents - regular_collisions
  total_time / big_crashes

/-- Theorem stating the frequency of big crashes given the problem conditions -/
theorem big_crash_frequency_is_20 :
  big_crash_frequency 36 (4 * 60) 10 = 20 := by
  sorry

#eval big_crash_frequency 36 (4 * 60) 10

end NUMINAMATH_CALUDE_big_crash_frequency_is_20_l1749_174978


namespace NUMINAMATH_CALUDE_expression_evaluation_l1749_174931

theorem expression_evaluation :
  (∀ x : ℤ, x = -2 → (3*x + 1)*(2*x - 3) - (6*x - 5)*(x - 4) = -67) ∧
  (∀ x y : ℤ, x = 1 ∧ y = 2 → (2*x - y)*(x + y) - 2*x*(-2*x + 3*y) + 6*x*(-x - 5/2*y) = -44) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1749_174931


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l1749_174965

theorem quadratic_roots_expression (a b : ℝ) : 
  (3 * a^2 + 2 * a - 2 = 0) →
  (3 * b^2 + 2 * b - 2 = 0) →
  (2 * a / (a^2 - b^2) - 1 / (a - b) = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l1749_174965


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1749_174987

/-- Given a geometric sequence with first term 3 and second term -1/2, 
    prove that its sixth term is -1/2592 -/
theorem sixth_term_of_geometric_sequence (a₁ a₂ : ℚ) (h₁ : a₁ = 3) (h₂ : a₂ = -1/2) :
  let r := a₂ / a₁
  let a_n (n : ℕ) := a₁ * r^(n - 1)
  a_n 6 = -1/2592 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l1749_174987


namespace NUMINAMATH_CALUDE_range_of_m_l1749_174918

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x y : ℝ, (x + y - m = 0) → ((x - 1)^2 + y^2 = 1) → False

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, (x₁^2 - x₁ + m - 4 = 0) ∧ (x₂^2 - x₂ + m - 4 = 0) ∧ (x₁ * x₂ < 0)

-- Main theorem
theorem range_of_m : 
  ∀ m : ℝ, (∀ m' : ℝ, p m' ∨ q m') → ¬(p m) → (1 - Real.sqrt 2 ≤ m ∧ m ≤ 1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1749_174918


namespace NUMINAMATH_CALUDE_dog_age_64_human_years_l1749_174967

/-- Calculates the age of a dog in dog years given its age in human years -/
def dogAge (humanYears : ℕ) : ℕ :=
  if humanYears ≤ 15 then 1
  else if humanYears ≤ 24 then 2
  else 2 + (humanYears - 24) / 5

/-- Theorem stating that a dog that has lived 64 human years is 10 years old in dog years -/
theorem dog_age_64_human_years : dogAge 64 = 10 := by
  sorry

end NUMINAMATH_CALUDE_dog_age_64_human_years_l1749_174967


namespace NUMINAMATH_CALUDE_duck_average_l1749_174930

theorem duck_average (adelaide ephraim kolton : ℕ) : 
  adelaide = 30 →
  adelaide = 2 * ephraim →
  kolton = ephraim + 45 →
  (adelaide + ephraim + kolton) / 3 = 35 := by
sorry

end NUMINAMATH_CALUDE_duck_average_l1749_174930


namespace NUMINAMATH_CALUDE_red_rose_value_l1749_174926

def total_flowers : ℕ := 400
def tulips : ℕ := 120
def white_roses : ℕ := 80
def selling_price : ℚ := 75

def roses : ℕ := total_flowers - tulips
def red_roses : ℕ := roses - white_roses
def roses_to_sell : ℕ := red_roses / 2

theorem red_rose_value (total_flowers tulips white_roses selling_price : ℕ) 
  (h1 : total_flowers = 400)
  (h2 : tulips = 120)
  (h3 : white_roses = 80)
  (h4 : selling_price = 75) :
  (selling_price : ℚ) / roses_to_sell = 3/4 := by
  sorry

#eval (75 : ℚ) / 100  -- To verify the result is indeed 0.75

end NUMINAMATH_CALUDE_red_rose_value_l1749_174926


namespace NUMINAMATH_CALUDE_difference_largest_smallest_l1749_174977

def digits : List Nat := [6, 2, 5]

def largest_number (digits : List Nat) : Nat :=
  sorry

def smallest_number (digits : List Nat) : Nat :=
  sorry

theorem difference_largest_smallest :
  largest_number digits - smallest_number digits = 396 := by
  sorry

end NUMINAMATH_CALUDE_difference_largest_smallest_l1749_174977


namespace NUMINAMATH_CALUDE_eldora_paper_clips_count_l1749_174983

/-- The cost of one box of paper clips in dollars -/
def paper_clip_cost : ℚ := 185 / 100

/-- The total cost of Eldora's purchase in dollars -/
def eldora_total : ℚ := 5540 / 100

/-- The number of packages of index cards Eldora bought -/
def eldora_index_cards : ℕ := 7

/-- The total cost of Finn's purchase in dollars -/
def finn_total : ℚ := 6170 / 100

/-- The number of boxes of paper clips Finn bought -/
def finn_paper_clips : ℕ := 12

/-- The number of packages of index cards Finn bought -/
def finn_index_cards : ℕ := 10

/-- The number of boxes of paper clips Eldora bought -/
def eldora_paper_clips : ℕ := 15

theorem eldora_paper_clips_count :
  ∃ (index_card_cost : ℚ),
    index_card_cost * finn_index_cards + paper_clip_cost * finn_paper_clips = finn_total ∧
    index_card_cost * eldora_index_cards + paper_clip_cost * eldora_paper_clips = eldora_total :=
by sorry

end NUMINAMATH_CALUDE_eldora_paper_clips_count_l1749_174983


namespace NUMINAMATH_CALUDE_eccentricity_decreases_as_a_increases_ellipse_approaches_circle_l1749_174992

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse where
  a : ℝ
  h_a_pos : 1 < a
  h_a_bound : a < 2 + Real.sqrt 5

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.a^2 - 1) / (4 * e.a))

/-- Theorem: As 'a' increases, the eccentricity decreases -/
theorem eccentricity_decreases_as_a_increases (e1 e2 : Ellipse) 
    (h : e1.a < e2.a) : eccentricity e2 < eccentricity e1 := by
  sorry

/-- Corollary: As 'a' increases, the ellipse becomes closer to a circle -/
theorem ellipse_approaches_circle (e1 e2 : Ellipse) (h : e1.a < e2.a) :
    ∃ (c : ℝ), 0 < c ∧ c < 1 ∧ eccentricity e2 < c * eccentricity e1 := by
  sorry

end NUMINAMATH_CALUDE_eccentricity_decreases_as_a_increases_ellipse_approaches_circle_l1749_174992


namespace NUMINAMATH_CALUDE_square_difference_nonnegative_l1749_174981

theorem square_difference_nonnegative (a b : ℝ) : (a - b)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_nonnegative_l1749_174981


namespace NUMINAMATH_CALUDE_max_value_of_f_l1749_174956

def f (x : ℝ) (a : ℝ) := -x^2 + 4*x + a

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≥ -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = -2) →
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f x a ≥ f y a) →
  (∃ x ∈ Set.Icc 0 1, f x a = 1) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1749_174956


namespace NUMINAMATH_CALUDE_custom_mul_properties_l1749_174917

/-- Custom multiplication operation -/
def custom_mul (m : ℚ) (x y : ℚ) : ℚ := (x * y) / (m * x + 2 * y)

/-- Theorem stating the properties of the custom multiplication -/
theorem custom_mul_properties :
  ∃ (m : ℚ), 
    (custom_mul m 1 2 = 2/5) ∧
    (m = 1) ∧
    (custom_mul m 2 6 = 6/7) := by sorry

end NUMINAMATH_CALUDE_custom_mul_properties_l1749_174917


namespace NUMINAMATH_CALUDE_model_height_calculation_l1749_174985

/-- The height of the Eiffel Tower in meters -/
def eiffel_height : ℝ := 320

/-- The capacity of the Eiffel Tower's observation deck in number of people -/
def eiffel_capacity : ℝ := 800

/-- The space required per person in square meters -/
def space_per_person : ℝ := 1

/-- The equivalent capacity of Mira's model in number of people -/
def model_capacity : ℝ := 0.8

/-- The height of Mira's model in meters -/
def model_height : ℝ := 10.12

theorem model_height_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (model_height - eiffel_height * (model_capacity / eiffel_capacity).sqrt) < ε :=
sorry

end NUMINAMATH_CALUDE_model_height_calculation_l1749_174985
