import Mathlib

namespace NUMINAMATH_CALUDE_function_transformation_l1549_154942

theorem function_transformation (g : ℝ → ℝ) (h : ∀ x, g (x + 2) = 2 * x + 3) :
  ∀ x, g x = 2 * x - 1 := by
sorry

end NUMINAMATH_CALUDE_function_transformation_l1549_154942


namespace NUMINAMATH_CALUDE_range_of_f_l1549_154965

def f (x : ℝ) : ℝ := 4 * (x - 1)^2 - 1

theorem range_of_f : 
  ∀ y ∈ Set.Icc (-1 : ℝ) 15, ∃ x ∈ Set.Ico (-1 : ℝ) 2, f x = y ∧
  ∀ x ∈ Set.Ico (-1 : ℝ) 2, f x ∈ Set.Icc (-1 : ℝ) 15 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l1549_154965


namespace NUMINAMATH_CALUDE_no_45_degree_rectangle_with_odd_intersections_l1549_154998

/-- Represents a point on a 2D grid --/
structure GridPoint where
  x : ℝ
  y : ℝ

/-- Represents a rectangle on a grid --/
structure GridRectangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint
  D : GridPoint

/-- Checks if a point is on a grid line --/
def isOnGridLine (p : GridPoint) : Prop :=
  ∃ n : ℤ, p.x = n ∨ p.y = n

/-- Checks if a line segment intersects the grid at a 45° angle --/
def intersectsAt45Degrees (p1 p2 : GridPoint) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ p2.x - p1.x = k ∧ p2.y - p1.y = k

/-- Counts the number of grid lines intersected by a line segment --/
noncomputable def gridLinesIntersected (p1 p2 : GridPoint) : ℕ :=
  sorry

/-- Main theorem: No rectangle exists with the given properties --/
theorem no_45_degree_rectangle_with_odd_intersections :
  ¬ ∃ (rect : GridRectangle),
    (¬ isOnGridLine rect.A) ∧ (¬ isOnGridLine rect.B) ∧ 
    (¬ isOnGridLine rect.C) ∧ (¬ isOnGridLine rect.D) ∧
    (intersectsAt45Degrees rect.A rect.B) ∧ 
    (intersectsAt45Degrees rect.B rect.C) ∧
    (intersectsAt45Degrees rect.C rect.D) ∧ 
    (intersectsAt45Degrees rect.D rect.A) ∧
    (Odd (gridLinesIntersected rect.A rect.B)) ∧
    (Odd (gridLinesIntersected rect.B rect.C)) ∧
    (Odd (gridLinesIntersected rect.C rect.D)) ∧
    (Odd (gridLinesIntersected rect.D rect.A)) :=
by
  sorry

end NUMINAMATH_CALUDE_no_45_degree_rectangle_with_odd_intersections_l1549_154998


namespace NUMINAMATH_CALUDE_proportional_relation_l1549_154959

/-- Given that x is directly proportional to y^4 and y is inversely proportional to z^(1/3),
    prove that if x = 8 when z = 27, then x = 81/32 when z = 64 -/
theorem proportional_relation (x y z : ℝ) (k₁ k₂ : ℝ) 
    (h1 : x = k₁ * y^4)
    (h2 : y = k₂ / z^(1/3))
    (h3 : x = 8 ∧ z = 27) :
    z = 64 → x = 81/32 := by
  sorry

end NUMINAMATH_CALUDE_proportional_relation_l1549_154959


namespace NUMINAMATH_CALUDE_toddler_count_problem_l1549_154929

/-- The actual number of toddlers given Bill's count and errors -/
def actual_toddler_count (counted : ℕ) (double_counted : ℕ) (hidden : ℕ) : ℕ :=
  counted - double_counted + hidden

/-- Theorem stating the actual number of toddlers in the given scenario -/
theorem toddler_count_problem : 
  actual_toddler_count 34 10 4 = 28 := by
  sorry

end NUMINAMATH_CALUDE_toddler_count_problem_l1549_154929


namespace NUMINAMATH_CALUDE_brandon_card_count_l1549_154982

theorem brandon_card_count (malcom_cards : ℕ) (brandon_cards : ℕ) : 
  (malcom_cards = brandon_cards + 8) →
  (malcom_cards / 2 = 14) →
  brandon_cards = 20 := by
sorry

end NUMINAMATH_CALUDE_brandon_card_count_l1549_154982


namespace NUMINAMATH_CALUDE_arrow_transformation_l1549_154949

-- Define the possible orientations of an arrow
inductive ArrowOrientation
  | Right
  | Left
  | Up
  | Down

-- Define the possible directions an arrow can point
inductive ArrowDirection
  | Right
  | Left
  | Up

-- Define the arrow shape
structure Arrow where
  orientation : ArrowOrientation
  direction : ArrowDirection

-- Define the transformations
def horizontalFlip (a : Arrow) : Arrow :=
  match a.direction with
  | ArrowDirection.Right => { orientation := a.orientation, direction := ArrowDirection.Left }
  | ArrowDirection.Left => { orientation := a.orientation, direction := ArrowDirection.Right }
  | ArrowDirection.Up => a

def rotate180 (a : Arrow) : Arrow :=
  match a.orientation with
  | ArrowOrientation.Right => { orientation := ArrowOrientation.Left, direction := a.direction }
  | ArrowOrientation.Left => { orientation := ArrowOrientation.Right, direction := a.direction }
  | ArrowOrientation.Up => { orientation := ArrowOrientation.Down, direction := a.direction }
  | ArrowOrientation.Down => { orientation := ArrowOrientation.Up, direction := a.direction }

-- Theorem statement
theorem arrow_transformation (a : Arrow) 
  (h1 : a.orientation = ArrowOrientation.Right) 
  (h2 : a.direction = ArrowDirection.Right) : 
  (rotate180 (horizontalFlip a)) = 
  { orientation := ArrowOrientation.Left, direction := ArrowDirection.Right } :=
by sorry

end NUMINAMATH_CALUDE_arrow_transformation_l1549_154949


namespace NUMINAMATH_CALUDE_sarahs_initial_trucks_l1549_154951

/-- Given that Sarah gave away 13 trucks and has 38 trucks remaining,
    prove that she initially had 51 trucks. -/
theorem sarahs_initial_trucks :
  ∀ (initial_trucks given_trucks remaining_trucks : ℕ),
    given_trucks = 13 →
    remaining_trucks = 38 →
    initial_trucks = given_trucks + remaining_trucks →
    initial_trucks = 51 :=
by sorry

end NUMINAMATH_CALUDE_sarahs_initial_trucks_l1549_154951


namespace NUMINAMATH_CALUDE_evaluate_expression_l1549_154962

theorem evaluate_expression : 3000 * (3000^3001)^2 = 3000^6003 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1549_154962


namespace NUMINAMATH_CALUDE_prob_connected_formula_l1549_154985

/-- The number of vertices in the graph -/
def n : ℕ := 20

/-- The number of edges removed -/
def k : ℕ := 35

/-- The total number of edges in a complete graph with n vertices -/
def total_edges : ℕ := n * (n - 1) / 2

/-- The probability that the graph remains connected after removing k edges -/
def prob_connected : ℚ :=
  1 - (n : ℚ) * (Nat.choose (total_edges - n + 1) (k - n + 1) : ℚ) / (Nat.choose total_edges k : ℚ)

theorem prob_connected_formula :
  prob_connected = 1 - (20 : ℚ) * (Nat.choose 171 16 : ℚ) / (Nat.choose 190 35 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_prob_connected_formula_l1549_154985


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_over_sums_l1549_154947

theorem min_value_sum_of_squares_over_sums (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_sum : a + b + c = 9) : 
  (a^2 + b^2)/(a + b) + (a^2 + c^2)/(a + c) + (b^2 + c^2)/(b + c) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_over_sums_l1549_154947


namespace NUMINAMATH_CALUDE_video_game_points_l1549_154924

/-- The number of points earned for defeating one enemy in a video game -/
def points_per_enemy (total_enemies : ℕ) (enemies_defeated : ℕ) (total_points : ℕ) : ℚ :=
  total_points / enemies_defeated

theorem video_game_points :
  let total_enemies : ℕ := 7
  let enemies_defeated : ℕ := total_enemies - 2
  let total_points : ℕ := 40
  points_per_enemy total_enemies enemies_defeated total_points = 8 := by
  sorry

end NUMINAMATH_CALUDE_video_game_points_l1549_154924


namespace NUMINAMATH_CALUDE_average_of_abcd_l1549_154901

theorem average_of_abcd (a b c d : ℝ) : 
  (4 + 6 + 9 + a + b + c + d) / 7 = 20 → (a + b + c + d) / 4 = 30.25 := by
  sorry

end NUMINAMATH_CALUDE_average_of_abcd_l1549_154901


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l1549_154953

/-- Calculate the profit percentage given the selling price and cost price -/
theorem profit_percentage_calculation (selling_price cost_price : ℚ) :
  selling_price = 1800 ∧ cost_price = 1500 →
  (selling_price - cost_price) / cost_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l1549_154953


namespace NUMINAMATH_CALUDE_min_overlap_mozart_bach_l1549_154994

theorem min_overlap_mozart_bach (total : ℕ) (mozart : ℕ) (bach : ℕ) 
  (h_total : total = 200)
  (h_mozart : mozart = 160)
  (h_bach : bach = 145)
  : mozart + bach - total ≥ 105 := by
  sorry

end NUMINAMATH_CALUDE_min_overlap_mozart_bach_l1549_154994


namespace NUMINAMATH_CALUDE_prob_shortest_diagonal_icosagon_l1549_154908

/-- A regular polygon with n sides -/
structure RegularPolygon where
  n : ℕ
  h : n ≥ 3

/-- The number of diagonals in a regular polygon -/
def num_diagonals (p : RegularPolygon) : ℕ := p.n * (p.n - 3) / 2

/-- The number of shortest diagonals in a regular polygon -/
def num_shortest_diagonals (p : RegularPolygon) : ℕ := p.n / 2

/-- An icosagon is a regular polygon with 20 sides -/
def icosagon : RegularPolygon where
  n := 20
  h := by norm_num

/-- The probability of selecting a shortest diagonal in an icosagon -/
def prob_shortest_diagonal (p : RegularPolygon) : ℚ :=
  (num_shortest_diagonals p : ℚ) / (num_diagonals p : ℚ)

theorem prob_shortest_diagonal_icosagon :
  prob_shortest_diagonal icosagon = 1 / 17 := by
  sorry

end NUMINAMATH_CALUDE_prob_shortest_diagonal_icosagon_l1549_154908


namespace NUMINAMATH_CALUDE_clock_digit_sum_probability_l1549_154909

def total_times : ℕ := 1440
def times_with_sum_23 : ℕ := 4

theorem clock_digit_sum_probability :
  (times_with_sum_23 : ℚ) / total_times = 1 / 360 := by
  sorry

end NUMINAMATH_CALUDE_clock_digit_sum_probability_l1549_154909


namespace NUMINAMATH_CALUDE_total_air_conditioner_sales_l1549_154928

theorem total_air_conditioner_sales (june_sales : ℕ) (july_increase : ℚ) : 
  june_sales = 96 →
  july_increase = 1/3 →
  june_sales + (june_sales * (1 + july_increase)).floor = 224 := by
  sorry

end NUMINAMATH_CALUDE_total_air_conditioner_sales_l1549_154928


namespace NUMINAMATH_CALUDE_two_number_difference_l1549_154964

theorem two_number_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) :
  y - x = 80 / 7 := by
  sorry

end NUMINAMATH_CALUDE_two_number_difference_l1549_154964


namespace NUMINAMATH_CALUDE_max_d_value_l1549_154995

def is_valid_number (d e : Nat) : Prop :=
  d ≤ 9 ∧ e ≤ 9 ∧ (808450 + 100000 * d + e) % 45 = 0

theorem max_d_value :
  ∃ (d : Nat), is_valid_number d 2 ∧
  ∀ (d' : Nat), is_valid_number d' 2 → d' ≤ d :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l1549_154995


namespace NUMINAMATH_CALUDE_complex_number_location_l1549_154972

theorem complex_number_location (z : ℂ) (h : z * (1 + Complex.I)^2 = 1 - Complex.I) :
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l1549_154972


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1549_154990

theorem complex_fraction_simplification :
  (1 + 3*Complex.I) / (1 - Complex.I) = -1 + 2*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1549_154990


namespace NUMINAMATH_CALUDE_max_distance_complex_l1549_154950

theorem max_distance_complex (w : ℂ) (h : Complex.abs w = 3) :
  ∃ (max_dist : ℝ), max_dist = 729 + 81 * Real.sqrt 5 ∧
  ∀ (z : ℂ), Complex.abs z = 3 → Complex.abs ((1 + 2*I)*z^4 - z^6) ≤ max_dist :=
sorry

end NUMINAMATH_CALUDE_max_distance_complex_l1549_154950


namespace NUMINAMATH_CALUDE_henri_total_time_l1549_154983

/-- Represents the total time Henri has for watching movies and reading -/
def total_time : ℝ := 8

/-- Duration of the first movie Henri watches -/
def movie1_duration : ℝ := 3.5

/-- Duration of the second movie Henri watches -/
def movie2_duration : ℝ := 1.5

/-- Henri's reading speed in words per minute -/
def reading_speed : ℝ := 10

/-- Number of words Henri reads -/
def words_read : ℝ := 1800

/-- Theorem stating that Henri's total time for movies and reading is 8 hours -/
theorem henri_total_time : 
  movie1_duration + movie2_duration + (words_read / reading_speed) / 60 = total_time := by
  sorry

end NUMINAMATH_CALUDE_henri_total_time_l1549_154983


namespace NUMINAMATH_CALUDE_opposite_pairs_l1549_154935

theorem opposite_pairs :
  (∀ x : ℝ, -|x| = -x ∧ -(-x) = x) ∧
  (-|-3| = -(-(-3))) ∧
  (3 ≠ -|-3|) ∧
  (-3 ≠ -(-1/3)) ∧
  (-3 ≠ -(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_opposite_pairs_l1549_154935


namespace NUMINAMATH_CALUDE_thomas_salary_l1549_154903

/-- Given the average salaries of two groups, prove Thomas's salary -/
theorem thomas_salary (raj_salary roshan_salary thomas_salary : ℕ) : 
  (raj_salary + roshan_salary) / 2 = 4000 →
  (raj_salary + roshan_salary + thomas_salary) / 3 = 5000 →
  thomas_salary = 7000 := by
  sorry

end NUMINAMATH_CALUDE_thomas_salary_l1549_154903


namespace NUMINAMATH_CALUDE_line_plane_relationships_l1549_154986

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the relationships between planes
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the relationship between lines
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)

-- Define the given conditions
variable (l m : Line) (α β : Plane)
variable (h1 : perpendicular l α)
variable (h2 : parallel m β)

-- State the theorem
theorem line_plane_relationships :
  (plane_parallel α β → line_perpendicular l m) ∧
  (line_parallel l m → plane_perpendicular α β) :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationships_l1549_154986


namespace NUMINAMATH_CALUDE_first_grade_enrollment_l1549_154977

theorem first_grade_enrollment :
  ∃ (n : ℕ),
    200 ≤ n ∧ n ≤ 300 ∧
    ∃ (r : ℕ), n = 25 * r + 10 ∧
    ∃ (l : ℕ), n = 30 * l - 15 ∧
    n = 285 := by
  sorry

end NUMINAMATH_CALUDE_first_grade_enrollment_l1549_154977


namespace NUMINAMATH_CALUDE_equal_x_y_l1549_154926

-- Define the geometric configuration
structure GeometricConfiguration where
  a₁ : ℝ
  a₂ : ℝ
  b₁ : ℝ
  b₂ : ℝ
  x : ℝ
  y : ℝ

-- Define the theorem
theorem equal_x_y (config : GeometricConfiguration) 
  (h1 : config.a₁ = config.a₂) 
  (h2 : config.b₁ = config.b₂) : 
  config.x = config.y := by
  sorry


end NUMINAMATH_CALUDE_equal_x_y_l1549_154926


namespace NUMINAMATH_CALUDE_bike_race_distance_difference_l1549_154919

/-- Represents a cyclist with their distance traveled and time taken -/
structure Cyclist where
  distance : ℝ
  time : ℝ

/-- The difference in distance traveled between two cyclists -/
def distanceDifference (c1 c2 : Cyclist) : ℝ :=
  c1.distance - c2.distance

theorem bike_race_distance_difference :
  let carlos : Cyclist := { distance := 70, time := 5 }
  let dana : Cyclist := { distance := 50, time := 5 }
  distanceDifference carlos dana = 20 := by
  sorry

end NUMINAMATH_CALUDE_bike_race_distance_difference_l1549_154919


namespace NUMINAMATH_CALUDE_angela_action_figures_l1549_154996

theorem angela_action_figures (initial : ℕ) : 
  (initial : ℚ) * (3/4) * (2/3) = 12 → initial = 24 := by
  sorry

end NUMINAMATH_CALUDE_angela_action_figures_l1549_154996


namespace NUMINAMATH_CALUDE_vegetable_difference_is_30_l1549_154967

/-- Calculates the difference between initial and remaining vegetables after exchanges --/
def vegetable_difference (
  initial_tomatoes : ℕ)
  (initial_carrots : ℕ)
  (initial_cucumbers : ℕ)
  (initial_bell_peppers : ℕ)
  (picked_tomatoes : ℕ)
  (picked_carrots : ℕ)
  (picked_cucumbers : ℕ)
  (picked_bell_peppers : ℕ)
  (neighbor1_tomatoes : ℕ)
  (neighbor1_carrots : ℕ)
  (neighbor2_tomatoes : ℕ)
  (neighbor2_cucumbers : ℕ)
  (neighbor2_radishes : ℕ)
  (neighbor3_bell_peppers : ℕ) : ℕ :=
  let initial_total := initial_tomatoes + initial_carrots + initial_cucumbers + initial_bell_peppers
  let remaining_tomatoes := initial_tomatoes - picked_tomatoes - neighbor1_tomatoes - neighbor2_tomatoes
  let remaining_carrots := initial_carrots - picked_carrots - neighbor1_carrots
  let remaining_cucumbers := initial_cucumbers - picked_cucumbers - neighbor2_cucumbers
  let remaining_bell_peppers := initial_bell_peppers - picked_bell_peppers - neighbor3_bell_peppers
  let remaining_total := remaining_tomatoes + remaining_carrots + remaining_cucumbers + remaining_bell_peppers + neighbor2_radishes
  initial_total - remaining_total

/-- The difference between initial and remaining vegetables is 30 --/
theorem vegetable_difference_is_30 : 
  vegetable_difference 17 13 8 15 5 6 3 8 3 2 2 3 5 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_difference_is_30_l1549_154967


namespace NUMINAMATH_CALUDE_forty_platforms_required_l1549_154921

/-- The minimum number of platforms required to transport granite slabs -/
def min_platforms (num_slabs_7ton : ℕ) (num_slabs_9ton : ℕ) (max_platform_capacity : ℕ) : ℕ :=
  let total_weight := num_slabs_7ton * 7 + num_slabs_9ton * 9
  (total_weight + max_platform_capacity - 1) / max_platform_capacity

/-- Theorem stating that 40 platforms are required for the given conditions -/
theorem forty_platforms_required :
  min_platforms 120 80 40 = 40 ∧
  ∀ n : ℕ, n < 40 → ¬ (120 * 7 + 80 * 9 ≤ n * 40) :=
by sorry

end NUMINAMATH_CALUDE_forty_platforms_required_l1549_154921


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l1549_154918

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_solution (a : ℕ → ℝ) (q : ℝ) 
  (h1 : geometric_sequence a q)
  (h2 : a 5 - a 1 = 15)
  (h3 : a 4 - a 2 = 6) :
  (q = 2 ∧ a 3 = 4) ∨ (q = 1/2 ∧ a 3 = -4) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l1549_154918


namespace NUMINAMATH_CALUDE_tangent_slope_at_x_one_l1549_154913

noncomputable def f (x : ℝ) := x^2 / 4 - Real.log x + 1

theorem tangent_slope_at_x_one (x : ℝ) (h : x > 0) :
  (deriv f x = -1/2) → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_x_one_l1549_154913


namespace NUMINAMATH_CALUDE_train_length_calculation_l1549_154968

/-- Calculates the length of a train given its speed and time to cross a pole. -/
theorem train_length_calculation (speed_km_hr : ℝ) (time_seconds : ℝ) : 
  speed_km_hr = 30 → time_seconds = 12 → 
  ∃ (length_meters : ℝ), 
    (abs (length_meters - 100) < 1) ∧ 
    (length_meters = speed_km_hr * (1000 / 3600) * time_seconds) := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1549_154968


namespace NUMINAMATH_CALUDE_table_permutation_exists_l1549_154915

/-- Represents a 2 × n table of real numbers -/
def Table (n : ℕ) := Fin 2 → Fin n → ℝ

/-- Calculates the sum of a column in the table -/
def columnSum (t : Table n) (j : Fin n) : ℝ :=
  (t 0 j) + (t 1 j)

/-- Calculates the sum of a row in the table -/
def rowSum (t : Table n) (i : Fin 2) : ℝ :=
  Finset.sum (Finset.univ : Finset (Fin n)) (λ j => t i j)

/-- States that all column sums in a table are different -/
def distinctColumnSums (t : Table n) : Prop :=
  ∀ j k : Fin n, j ≠ k → columnSum t j ≠ columnSum t k

/-- Represents a permutation of table elements -/
def tablePermutation (n : ℕ) := Fin 2 → Fin n → Fin 2 × Fin n

/-- Applies a permutation to a table -/
def applyPermutation (t : Table n) (p : tablePermutation n) : Table n :=
  λ i j => let (i', j') := p i j; t i' j'

theorem table_permutation_exists (n : ℕ) (h : n > 2) (t : Table n) 
  (hd : distinctColumnSums t) :
  ∃ p : tablePermutation n, 
    distinctColumnSums (applyPermutation t p) ∧ 
    rowSum (applyPermutation t p) 0 ≠ rowSum (applyPermutation t p) 1 :=
  sorry

end NUMINAMATH_CALUDE_table_permutation_exists_l1549_154915


namespace NUMINAMATH_CALUDE_betty_age_l1549_154910

/-- Represents the ages of Albert, Mary, and Betty --/
structure Ages where
  albert : ℕ
  mary : ℕ
  betty : ℕ

/-- The conditions given in the problem --/
def age_conditions (ages : Ages) : Prop :=
  ages.albert = 2 * ages.mary ∧
  ages.albert = 4 * ages.betty ∧
  ages.mary = ages.albert - 22

/-- The theorem stating Betty's age --/
theorem betty_age (ages : Ages) (h : age_conditions ages) : ages.betty = 11 := by
  sorry

end NUMINAMATH_CALUDE_betty_age_l1549_154910


namespace NUMINAMATH_CALUDE_camera_price_theorem_l1549_154946

/-- The sticker price of the camera -/
def sticker_price : ℝ := 666.67

/-- The price at Store X after discount and rebate -/
def store_x_price (p : ℝ) : ℝ := 0.80 * p - 50

/-- The price at Store Y after discount -/
def store_y_price (p : ℝ) : ℝ := 0.65 * p

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem camera_price_theorem : 
  store_y_price sticker_price - store_x_price sticker_price = 40 := by
  sorry


end NUMINAMATH_CALUDE_camera_price_theorem_l1549_154946


namespace NUMINAMATH_CALUDE_tangent_line_perpendicular_range_of_m_l1549_154931

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((x + a) * Real.log x) / (x + 1)

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ :=
  ((Real.log x + (x + a) / x) * (x + 1) - (x + a) * Real.log x) / ((x + 1)^2)

-- Theorem for part (I)
theorem tangent_line_perpendicular (a : ℝ) :
  f_derivative a 1 = 1/2 → a = 0 :=
by sorry

-- Theorem for part (II)
theorem range_of_m (m : ℝ) :
  (∀ x ≥ 1, f 0 x ≤ m * (x - 1)) ↔ m ≥ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_perpendicular_range_of_m_l1549_154931


namespace NUMINAMATH_CALUDE_x_minus_25_is_perfect_square_l1549_154978

theorem x_minus_25_is_perfect_square (n : ℕ) :
  let x := 10^(2*n + 4) + 10^(n + 3) + 50
  ∃ k : ℕ, x - 25 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_25_is_perfect_square_l1549_154978


namespace NUMINAMATH_CALUDE_situp_difference_l1549_154941

/-- The number of sit-ups Ken can do -/
def ken_situps : ℕ := 20

/-- The number of sit-ups Nathan can do -/
def nathan_situps : ℕ := 2 * ken_situps

/-- The number of sit-ups Bob can do -/
def bob_situps : ℕ := (ken_situps + nathan_situps) / 2

/-- The number of sit-ups Emma can do -/
def emma_situps : ℕ := bob_situps / 3

/-- The theorem stating the difference in sit-ups between the group (Nathan, Bob, Emma) and Ken -/
theorem situp_difference : nathan_situps + bob_situps + emma_situps - ken_situps = 60 := by
  sorry

end NUMINAMATH_CALUDE_situp_difference_l1549_154941


namespace NUMINAMATH_CALUDE_symmetric_point_y_axis_P_l1549_154922

/-- Given a point P in 3D space, this function returns its symmetric point with respect to the y-axis -/
def symmetricPointYAxis (P : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := P
  (-x, y, z)

/-- Theorem stating that the symmetric point of P(1,2,-1) with respect to the y-axis is (-1,2,1) -/
theorem symmetric_point_y_axis_P :
  symmetricPointYAxis (1, 2, -1) = (-1, 2, -1) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_y_axis_P_l1549_154922


namespace NUMINAMATH_CALUDE_racketSalesEarnings_l1549_154954

/-- The total amount earned from selling rackets, given the average price and number of pairs sold. -/
def totalEarnings (avgPrice : ℝ) (pairsSold : ℕ) : ℝ :=
  avgPrice * pairsSold

/-- Proof that the total earnings from selling rackets is $588, given the specified conditions. -/
theorem racketSalesEarnings :
  let avgPrice : ℝ := 9.8
  let pairsSold : ℕ := 60
  totalEarnings avgPrice pairsSold = 588 := by
  sorry

end NUMINAMATH_CALUDE_racketSalesEarnings_l1549_154954


namespace NUMINAMATH_CALUDE_half_sum_squares_even_odd_l1549_154999

theorem half_sum_squares_even_odd (a b : ℤ) :
  (∃ x y : ℤ, (4 * a^2 + 4 * b^2) / 2 = x^2 + y^2) ∨
  (∃ x y : ℤ, ((2 * a + 1)^2 + (2 * b + 1)^2) / 2 = x^2 + y^2) :=
by sorry

end NUMINAMATH_CALUDE_half_sum_squares_even_odd_l1549_154999


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_l1549_154923

-- Define the quadrilateral ABCD and point P
structure Quadrilateral :=
  (A B C D P : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def area (q : Quadrilateral) : ℝ := sorry

def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

def diagonals_orthogonal (q : Quadrilateral) : Prop := sorry

def perimeter (q : Quadrilateral) : ℝ := sorry

-- State the theorem
theorem quadrilateral_perimeter 
  (q : Quadrilateral)
  (h_convex : is_convex q)
  (h_area : area q = 2601)
  (h_PA : distance q.P q.A = 25)
  (h_PB : distance q.P q.B = 35)
  (h_PC : distance q.P q.C = 30)
  (h_PD : distance q.P q.D = 50)
  (h_ortho : diagonals_orthogonal q) :
  perimeter q = Real.sqrt 1850 + Real.sqrt 2125 + Real.sqrt 3400 + Real.sqrt 3125 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_l1549_154923


namespace NUMINAMATH_CALUDE_ellipse_angle_tangent_product_l1549_154956

/-- Given an ellipse with eccentricity e and a point P on the ellipse,
    if α is the angle PF₁F₂ and β is the angle PF₂F₁, where F₁ and F₂ are the foci,
    then tan(α/2) * tan(β/2) = (1 - e) / (1 + e) -/
theorem ellipse_angle_tangent_product (a b : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (α β e : ℝ)
  (h_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (h_foci : F₁ ≠ F₂)
  (h_eccentricity : e = Real.sqrt (a^2 - b^2) / a)
  (h_angle_α : α = Real.arccos ((P.1 - F₁.1) * (F₂.1 - F₁.1) + (P.2 - F₁.2) * (F₂.2 - F₁.2)) /
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)))
  (h_angle_β : β = Real.arccos ((P.1 - F₂.1) * (F₁.1 - F₂.1) + (P.2 - F₂.2) * (F₁.2 - F₂.2)) /
    (Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) * Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)))
  : Real.tan (α/2) * Real.tan (β/2) = (1 - e) / (1 + e) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_angle_tangent_product_l1549_154956


namespace NUMINAMATH_CALUDE_expression_evaluation_l1549_154992

theorem expression_evaluation (x y : ℕ) (h1 : x = 3) (h2 : y = 2) :
  5 * x^(y + 1) + 6 * y^(x + 1) = 231 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1549_154992


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1549_154991

/-- Given a cube with surface area 150 cm², prove its volume is 125 cm³ -/
theorem cube_volume_from_surface_area :
  ∀ (side_length : ℝ),
  (6 : ℝ) * side_length^2 = 150 →
  side_length^3 = 125 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l1549_154991


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l1549_154944

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l1549_154944


namespace NUMINAMATH_CALUDE_divisible_by_prime_l1549_154988

/-- Sequence of polynomials Q -/
def Q : ℕ → (ℤ → ℤ)
| 0 => λ x => 1
| 1 => λ x => x
| (n + 2) => λ x => x * Q (n + 1) x + (n + 1) * Q n x

/-- Theorem statement -/
theorem divisible_by_prime (p : ℕ) (hp : p.Prime) (hp2 : p > 2) :
  ∀ x : ℤ, (Q p x - x ^ p) % p = 0 := by sorry

end NUMINAMATH_CALUDE_divisible_by_prime_l1549_154988


namespace NUMINAMATH_CALUDE_train_speed_before_increase_l1549_154981

/-- The average speed before a train's speed increase, given the travel times before and after
    the increase, and the amount of speed increase. -/
theorem train_speed_before_increase 
  (time_after : ℝ) 
  (time_before : ℝ) 
  (speed_increase : ℝ) 
  (h1 : time_after = 10) 
  (h2 : time_before = 12) 
  (h3 : speed_increase = 20) :
  let speed_before := (time_after * (time_before * speed_increase) / (time_before - time_after))
  speed_before = 100 := by
sorry

end NUMINAMATH_CALUDE_train_speed_before_increase_l1549_154981


namespace NUMINAMATH_CALUDE_road_width_calculation_l1549_154904

/-- Calculates the width of roads on a rectangular lawn given the dimensions and cost --/
theorem road_width_calculation (length width total_cost cost_per_sqm : ℝ) : 
  length = 80 →
  width = 60 →
  total_cost = 3900 →
  cost_per_sqm = 3 →
  let road_area := total_cost / cost_per_sqm
  let road_width := road_area / (length + width)
  road_width = 65 / 7 := by
  sorry

end NUMINAMATH_CALUDE_road_width_calculation_l1549_154904


namespace NUMINAMATH_CALUDE_three_numbers_sum_l1549_154945

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- a, b, c are in ascending order
  (a + b + c) / 3 = a + 8 ∧  -- mean is 8 more than least
  (a + b + c) / 3 = c - 20 ∧  -- mean is 20 less than greatest
  b = 10  -- median is 10
  → a + b + c = 66 := by sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l1549_154945


namespace NUMINAMATH_CALUDE_complex_power_2013_l1549_154993

def i : ℂ := Complex.I

theorem complex_power_2013 : ((1 + i) / (1 - i)) ^ 2013 = i := by sorry

end NUMINAMATH_CALUDE_complex_power_2013_l1549_154993


namespace NUMINAMATH_CALUDE_robin_seeds_count_robin_seeds_is_150_l1549_154973

theorem robin_seeds_count : ℕ → ℕ → Prop :=
  fun (robin_bushes sparrow_bushes : ℕ) =>
    (robin_bushes = sparrow_bushes + 5) →
    (5 * robin_bushes = 6 * sparrow_bushes) →
    (5 * robin_bushes = 150)

/-- The number of seeds hidden by the robin is 150 -/
theorem robin_seeds_is_150 : ∃ (robin_bushes sparrow_bushes : ℕ),
  robin_seeds_count robin_bushes sparrow_bushes :=
by
  sorry

#check robin_seeds_is_150

end NUMINAMATH_CALUDE_robin_seeds_count_robin_seeds_is_150_l1549_154973


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l1549_154917

theorem magnitude_of_complex_power : 
  Complex.abs ((3 : ℂ) + (2 : ℂ) * Complex.I) ^ 6 = 2197 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l1549_154917


namespace NUMINAMATH_CALUDE_new_average_age_l1549_154969

/-- Calculates the new average age of a group after new members join -/
theorem new_average_age
  (initial_count : ℕ)
  (initial_avg_age : ℚ)
  (new_count : ℕ)
  (new_avg_age : ℚ)
  (h1 : initial_count = 20)
  (h2 : initial_avg_age = 16)
  (h3 : new_count = 20)
  (h4 : new_avg_age = 15) :
  let total_initial_age := initial_count * initial_avg_age
  let total_new_age := new_count * new_avg_age
  let total_count := initial_count + new_count
  let new_avg := (total_initial_age + total_new_age) / total_count
  new_avg = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l1549_154969


namespace NUMINAMATH_CALUDE_total_books_on_shelves_l1549_154984

/-- Given 150 book shelves with 15 books each, the total number of books is 2250. -/
theorem total_books_on_shelves (num_shelves : ℕ) (books_per_shelf : ℕ) : 
  num_shelves = 150 → books_per_shelf = 15 → num_shelves * books_per_shelf = 2250 := by
  sorry

end NUMINAMATH_CALUDE_total_books_on_shelves_l1549_154984


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l1549_154970

theorem geometric_arithmetic_sequence (a₁ : ℝ) (h : a₁ ≠ 0) :
  ∃! (s : Finset ℝ), s.card = 2 ∧
    ∀ q ∈ s, 2 * (a₁ * q^4) = 4 * a₁ + (-2 * (a₁ * q^2)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l1549_154970


namespace NUMINAMATH_CALUDE_drain_time_to_half_l1549_154932

/-- Represents the remaining water volume in the pool after draining for a given time. -/
def remaining_water (t : ℝ) : ℝ := 300 - 25 * t

/-- Proves that it takes 6 hours to drain the pool from 300 m³ to 150 m³. -/
theorem drain_time_to_half : ∃ t : ℝ, t = 6 ∧ remaining_water t = 150 := by
  sorry

end NUMINAMATH_CALUDE_drain_time_to_half_l1549_154932


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l1549_154920

/-- Given a boat traveling downstream, calculate the speed of the stream. -/
theorem stream_speed_calculation 
  (boat_speed : ℝ)           -- Speed of the boat in still water
  (distance : ℝ)             -- Distance traveled downstream
  (time : ℝ)                 -- Time taken to travel downstream
  (h1 : boat_speed = 5)      -- Boat speed is 5 km/hr
  (h2 : distance = 100)      -- Distance is 100 km
  (h3 : time = 10)           -- Time taken is 10 hours
  : ∃ (stream_speed : ℝ), 
    stream_speed = 5 ∧ 
    distance = (boat_speed + stream_speed) * time :=
by sorry


end NUMINAMATH_CALUDE_stream_speed_calculation_l1549_154920


namespace NUMINAMATH_CALUDE_pretzels_john_ate_l1549_154948

/-- Given a bowl of pretzels and information about how many pretzels three people ate,
    prove how many pretzels John ate. -/
theorem pretzels_john_ate (total : ℕ) (john alan marcus : ℕ) 
    (h1 : total = 95)
    (h2 : alan = john - 9)
    (h3 : marcus = john + 12)
    (h4 : marcus = 40) :
    john = 28 := by sorry

end NUMINAMATH_CALUDE_pretzels_john_ate_l1549_154948


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l1549_154966

theorem fraction_zero_implies_x_equals_one (x : ℝ) :
  x ≠ -1 →
  (x^2 - 1) / (x + 1) = 0 →
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l1549_154966


namespace NUMINAMATH_CALUDE_total_vacations_and_classes_l1549_154902

/-- Represents the number of classes Kelvin has -/
def kelvin_classes : ℕ := 90

/-- Represents the cost of each of Kelvin's classes in dollars -/
def kelvin_class_cost : ℕ := 75

/-- Represents Grant's maximum budget for vacations in dollars -/
def grant_max_budget : ℕ := 100000

/-- Theorem stating that the sum of Grant's vacations and Kelvin's classes is 450 -/
theorem total_vacations_and_classes : 
  ∃ (grant_vacations : ℕ),
    grant_vacations = 4 * kelvin_classes ∧ 
    grant_vacations * (2 * kelvin_class_cost) ≤ grant_max_budget ∧
    grant_vacations + kelvin_classes = 450 := by
  sorry

end NUMINAMATH_CALUDE_total_vacations_and_classes_l1549_154902


namespace NUMINAMATH_CALUDE_license_plate_combinations_l1549_154936

def letter_count : ℕ := 26
def digit_count : ℕ := 10
def letter_positions : ℕ := 4
def digit_positions : ℕ := 2

theorem license_plate_combinations : 
  (Nat.choose letter_count 2 * 2 * Nat.choose letter_positions 2 * digit_count ^ digit_positions) = 390000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l1549_154936


namespace NUMINAMATH_CALUDE_mutually_exclusive_but_not_complementary_l1549_154905

/-- A bag containing red and black balls -/
structure Bag where
  red : ℕ
  black : ℕ

/-- The sample space of drawing two balls from the bag -/
def SampleSpace (b : Bag) := Fin (b.red + b.black) × Fin (b.red + b.black - 1)

/-- Event of drawing exactly one black ball -/
def ExactlyOneBlack (b : Bag) : Set (SampleSpace b) := sorry

/-- Event of drawing exactly two black balls -/
def ExactlyTwoBlack (b : Bag) : Set (SampleSpace b) := sorry

/-- Two events are mutually exclusive -/
def MutuallyExclusive {α : Type*} (A B : Set α) : Prop :=
  A ∩ B = ∅

/-- Two events are complementary -/
def Complementary {α : Type*} (A B : Set α) : Prop :=
  MutuallyExclusive A B ∧ A ∪ B = Set.univ

/-- The main theorem -/
theorem mutually_exclusive_but_not_complementary :
  let b : Bag := ⟨2, 2⟩
  MutuallyExclusive (ExactlyOneBlack b) (ExactlyTwoBlack b) ∧
  ¬Complementary (ExactlyOneBlack b) (ExactlyTwoBlack b) := by sorry

end NUMINAMATH_CALUDE_mutually_exclusive_but_not_complementary_l1549_154905


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_implies_m_range_l1549_154937

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate. -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The theorem states that if the point P(m-3, m-2) is in the second quadrant,
    then m is strictly between 2 and 3. -/
theorem point_in_second_quadrant_implies_m_range
  (m : ℝ)
  (h : is_in_second_quadrant (m - 3) (m - 2)) :
  2 < m ∧ m < 3 :=
sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_implies_m_range_l1549_154937


namespace NUMINAMATH_CALUDE_max_fourth_power_sum_l1549_154938

theorem max_fourth_power_sum (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) :
  ∃ (m : ℝ), m = 64 / (4^(1/3)) ∧ a^4 + b^4 + c^4 + d^4 ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_fourth_power_sum_l1549_154938


namespace NUMINAMATH_CALUDE_book_selection_count_l1549_154955

theorem book_selection_count (A B : Type) [Fintype A] [Fintype B] 
  (h1 : Fintype.card A = 4) (h2 : Fintype.card B = 5) : 
  Fintype.card (A × B) = 20 := by
  sorry

end NUMINAMATH_CALUDE_book_selection_count_l1549_154955


namespace NUMINAMATH_CALUDE_attached_pyramids_volume_l1549_154914

/-- A solid formed by two attached pyramids -/
structure AttachedPyramids where
  /-- Length of each edge in the square-based pyramid -/
  base_edge_length : ℝ
  /-- Total length of all edges in the resulting solid -/
  total_edge_length : ℝ

/-- The volume of the attached pyramids solid -/
noncomputable def volume (ap : AttachedPyramids) : ℝ :=
  2 * Real.sqrt 2

/-- Theorem stating the volume of the attached pyramids solid -/
theorem attached_pyramids_volume (ap : AttachedPyramids) 
  (h1 : ap.base_edge_length = 2)
  (h2 : ap.total_edge_length = 18) :
  volume ap = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_attached_pyramids_volume_l1549_154914


namespace NUMINAMATH_CALUDE_coefficient_a7_equals_negative_eight_l1549_154952

theorem coefficient_a7_equals_negative_eight :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ),
  (∀ x : ℝ, (x - 2)^8 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
                        a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8) →
  a₇ = -8 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a7_equals_negative_eight_l1549_154952


namespace NUMINAMATH_CALUDE_locus_equation_l1549_154906

def point_A : ℝ × ℝ := (4, 0)
def point_B : ℝ × ℝ := (1, 0)

theorem locus_equation (x y : ℝ) :
  let M := (x, y)
  let dist_MA := Real.sqrt ((x - point_A.1)^2 + (y - point_A.2)^2)
  let dist_MB := Real.sqrt ((x - point_B.1)^2 + (y - point_B.2)^2)
  dist_MA = (1/2) * dist_MB → x^2 + y^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_locus_equation_l1549_154906


namespace NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l1549_154911

/-- The focus of a parabola y = ax^2 + k is at (0, 1/(4a) + k) -/
theorem parabola_focus (a k : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ := (0, 1/(4*a) + k)
  ∀ x y : ℝ, y = a * x^2 + k → (x - f.1)^2 + (y - f.2)^2 = (y - k + 1/(4*a))^2 :=
sorry

/-- The focus of the parabola y = 9x^2 + 6 is at (0, 217/36) -/
theorem focus_of_specific_parabola :
  let f : ℝ × ℝ := (0, 217/36)
  ∀ x y : ℝ, y = 9 * x^2 + 6 → (x - f.1)^2 + (y - f.2)^2 = (y - 6 + 1/36)^2 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l1549_154911


namespace NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l1549_154963

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def sumInteriorNumbers (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_triangle_interior_sum :
  sumInteriorNumbers 4 = 6 ∧
  sumInteriorNumbers 5 = 14 →
  sumInteriorNumbers 7 = 62 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l1549_154963


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1549_154971

/-- Given a triangle DEF with side lengths DE = 8, DF = 5, and EF = 9,
    the radius of its inscribed circle is 6√11/11. -/
theorem inscribed_circle_radius (DE DF EF : ℝ) (hDE : DE = 8) (hDF : DF = 5) (hEF : EF = 9) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  area / s = 6 * Real.sqrt 11 / 11 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1549_154971


namespace NUMINAMATH_CALUDE_pie_fraction_not_eaten_l1549_154930

theorem pie_fraction_not_eaten
  (lara_ate : ℚ)
  (ryan_ate : ℚ)
  (cassie_ate_remaining : ℚ)
  (h1 : lara_ate = 1/4)
  (h2 : ryan_ate = 3/10)
  (h3 : cassie_ate_remaining = 2/3)
  : 1 - (lara_ate + ryan_ate + cassie_ate_remaining * (1 - lara_ate - ryan_ate)) = 3/20 := by
  sorry

#check pie_fraction_not_eaten

end NUMINAMATH_CALUDE_pie_fraction_not_eaten_l1549_154930


namespace NUMINAMATH_CALUDE_triangle_inequality_proof_l1549_154933

/-- A structure representing a set of three line segments. -/
structure LineSegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The triangle inequality theorem for a set of line segments. -/
def satisfies_triangle_inequality (s : LineSegmentSet) : Prop :=
  s.a + s.b > s.c ∧ s.b + s.c > s.a ∧ s.c + s.a > s.b

/-- The set of line segments that can form a triangle. -/
def triangle_set : LineSegmentSet :=
  { a := 3, b := 4, c := 5 }

/-- The sets of line segments that cannot form triangles. -/
def non_triangle_sets : List LineSegmentSet :=
  [{ a := 1, b := 2, c := 3 },
   { a := 4, b := 5, c := 10 },
   { a := 6, b := 9, c := 2 }]

theorem triangle_inequality_proof :
  satisfies_triangle_inequality triangle_set ∧
  ∀ s ∈ non_triangle_sets, ¬satisfies_triangle_inequality s :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_proof_l1549_154933


namespace NUMINAMATH_CALUDE_four_distinct_roots_l1549_154925

/-- The equation x^2 - 4|x| + 5 = m has four distinct real roots if and only if 1 < m < 5 -/
theorem four_distinct_roots (m : ℝ) :
  (∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (∀ x : ℝ, x^2 - 4 * |x| + 5 = m ↔ x = a ∨ x = b ∨ x = c ∨ x = d)) ↔
  1 < m ∧ m < 5 := by
  sorry

end NUMINAMATH_CALUDE_four_distinct_roots_l1549_154925


namespace NUMINAMATH_CALUDE_expression_value_l1549_154979

theorem expression_value :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 4 + 6 - 8 + 10 - 12 + 14) = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1549_154979


namespace NUMINAMATH_CALUDE_cut_square_problem_l1549_154934

/-- Given a square with integer side length and four isosceles right triangles
    cut from its corners, if the total area of the cut triangles is 40 square centimeters,
    then the area of the remaining rectangle is 24 square centimeters. -/
theorem cut_square_problem (s a b : ℕ) : 
  s = a + b →  -- The side length of the square is the sum of the leg lengths
  a^2 + b^2 = 40 →  -- The total area of cut triangles is 40
  s^2 - (a^2 + b^2) = 24 :=  -- The area of the remaining rectangle is 24
by sorry

end NUMINAMATH_CALUDE_cut_square_problem_l1549_154934


namespace NUMINAMATH_CALUDE_square_root_of_ten_thousand_l1549_154974

theorem square_root_of_ten_thousand : Real.sqrt 10000 = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_ten_thousand_l1549_154974


namespace NUMINAMATH_CALUDE_gcf_of_40_120_80_l1549_154939

theorem gcf_of_40_120_80 : Nat.gcd 40 (Nat.gcd 120 80) = 40 := by sorry

end NUMINAMATH_CALUDE_gcf_of_40_120_80_l1549_154939


namespace NUMINAMATH_CALUDE_rug_area_theorem_l1549_154961

/-- Given three overlapping rugs, prove their combined area is 212 square meters -/
theorem rug_area_theorem (total_covered_area single_layer_area double_layer_area triple_layer_area : ℝ) :
  total_covered_area = 140 →
  double_layer_area = 24 →
  triple_layer_area = 24 →
  single_layer_area = total_covered_area - double_layer_area - triple_layer_area →
  single_layer_area + 2 * double_layer_area + 3 * triple_layer_area = 212 :=
by sorry

end NUMINAMATH_CALUDE_rug_area_theorem_l1549_154961


namespace NUMINAMATH_CALUDE_problem_G6_1_l1549_154980

theorem problem_G6_1 (p : ℝ) : 
  p = (21^3 - 11^3) / (21^2 + 21*11 + 11^2) → p = 10 := by
  sorry


end NUMINAMATH_CALUDE_problem_G6_1_l1549_154980


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1549_154912

/-- A hyperbola with center at the origin, transverse axis on the y-axis, and one focus at (0, 6) -/
structure Hyperbola where
  center : ℝ × ℝ
  transverse_axis : ℝ → ℝ × ℝ
  focus : ℝ × ℝ
  h_center : center = (0, 0)
  h_transverse : ∀ x, transverse_axis x = (0, x)
  h_focus : focus = (0, 6)

/-- The equation of the hyperbola is y^2 - x^2 = 18 -/
theorem hyperbola_equation (h : Hyperbola) : 
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | y^2 - x^2 = 18} ↔ 
  ∃ t : ℝ, h.transverse_axis t = (x, y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1549_154912


namespace NUMINAMATH_CALUDE_map_scale_conversion_l1549_154943

/-- Given a map scale where 8 cm represents 40 km, prove that 20 cm represents 100 km -/
theorem map_scale_conversion (map_scale : ℝ → ℝ) 
  (h1 : map_scale 8 = 40) -- 8 cm represents 40 km
  (h2 : ∀ x y : ℝ, map_scale (x + y) = map_scale x + map_scale y) -- Linear scaling
  (h3 : ∀ x : ℝ, map_scale x ≥ 0) -- Non-negative scaling
  : map_scale 20 = 100 := by sorry

end NUMINAMATH_CALUDE_map_scale_conversion_l1549_154943


namespace NUMINAMATH_CALUDE_num_paths_upper_bound_l1549_154916

/-- Represents a rectangular grid city -/
structure City where
  length : ℕ
  width : ℕ

/-- The number of possible paths from southwest to northeast corner -/
def num_paths (c : City) : ℕ := sorry

/-- The theorem to be proved -/
theorem num_paths_upper_bound (c : City) :
  num_paths c ≤ 2^(c.length * c.width) := by sorry

end NUMINAMATH_CALUDE_num_paths_upper_bound_l1549_154916


namespace NUMINAMATH_CALUDE_non_negative_xy_l1549_154960

theorem non_negative_xy (x y : ℝ) :
  |x^2 + y^2 - 4*x - 4*y + 5| = |2*x + 2*y - 4| → x ≥ 0 ∧ y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_non_negative_xy_l1549_154960


namespace NUMINAMATH_CALUDE_min_value_of_z_l1549_154976

/-- The objective function to be minimized -/
def z (x y : ℝ) : ℝ := 2*x + 5*y

/-- The feasible region defined by the given constraints -/
def feasible_region (x y : ℝ) : Prop :=
  x - y + 2 ≥ 0 ∧ 2*x + 3*y - 6 ≥ 0 ∧ 3*x + 2*y - 9 ≤ 0

/-- Theorem stating that the minimum value of z in the feasible region is 6 -/
theorem min_value_of_z : 
  ∀ x y : ℝ, feasible_region x y → z x y ≥ 6 ∧ ∃ x₀ y₀ : ℝ, feasible_region x₀ y₀ ∧ z x₀ y₀ = 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l1549_154976


namespace NUMINAMATH_CALUDE_unique_satisfying_pair_l1549_154940

/-- A pair of real numbers satisfying both arithmetic and geometric progression conditions -/
def SatisfyingPair (a b : ℝ) : Prop :=
  -- Arithmetic progression condition
  (15 : ℝ) - a = a - b ∧ a - b = b - (a * b) ∧
  -- Geometric progression condition
  ∃ r : ℝ, a * b = 15 * r^3 ∧ r > 0

/-- Theorem stating that (15, 15) is the only pair satisfying both conditions -/
theorem unique_satisfying_pair :
  ∀ a b : ℝ, SatisfyingPair a b → a = 15 ∧ b = 15 :=
by sorry

end NUMINAMATH_CALUDE_unique_satisfying_pair_l1549_154940


namespace NUMINAMATH_CALUDE_gcd_98_63_l1549_154958

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_98_63_l1549_154958


namespace NUMINAMATH_CALUDE_billion_to_scientific_notation_l1549_154975

theorem billion_to_scientific_notation :
  (6.1 : ℝ) * 1000000000 = (6.1 : ℝ) * (10 ^ 8) :=
by sorry

end NUMINAMATH_CALUDE_billion_to_scientific_notation_l1549_154975


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l1549_154989

/-- A line with equal intercepts on both coordinate axes passing through (-3, -2) -/
structure EqualInterceptLine where
  -- The slope of the line
  k : ℝ
  -- The y-intercept of the line
  b : ℝ
  -- The line passes through (-3, -2)
  point_condition : -2 = k * (-3) + b
  -- The line has equal intercepts on both axes
  equal_intercepts : k * b + b = b

/-- The equation of an EqualInterceptLine is either 2x - 3y = 0 or x + y + 5 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (∀ x y, y = l.k * x + l.b → 2 * x - 3 * y = 0) ∨
  (∀ x y, y = l.k * x + l.b → x + y + 5 = 0) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l1549_154989


namespace NUMINAMATH_CALUDE_cooking_oil_problem_l1549_154907

theorem cooking_oil_problem (X : ℝ) : 
  (X - ((2/5) * X + 300)) - ((1/2) * (X - ((2/5) * X + 300)) - 200) = 800 →
  X = 2500 :=
by
  sorry

#check cooking_oil_problem

end NUMINAMATH_CALUDE_cooking_oil_problem_l1549_154907


namespace NUMINAMATH_CALUDE_f_properties_l1549_154900

noncomputable section

variables {f : ℝ → ℝ} {a : ℝ}

-- f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- f is symmetric about x = 1
def symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f (2 - x) = f x

-- f satisfies the multiplicative property for x₁, x₂ ∈ [0, 1/2]
def multiplicative_property (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ∈ Set.Icc 0 (1/2) → x₂ ∈ Set.Icc 0 (1/2) → f (x₁ + x₂) = f x₁ * f x₂

theorem f_properties (heven : even_function f) (hsym : symmetric_about_one f)
    (hmult : multiplicative_property f) (hf1 : f 1 = a) (ha : a > 0) :
    f (1/2) = Real.sqrt a ∧ f (1/4) = Real.sqrt (Real.sqrt a) ∧ ∀ x, f (x + 2) = f x := by
  sorry

end

end NUMINAMATH_CALUDE_f_properties_l1549_154900


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l1549_154997

theorem complementary_angles_difference (x y : ℝ) : 
  x + y = 90 → x = 5 * y → |x - y| = 60 := by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l1549_154997


namespace NUMINAMATH_CALUDE_absolute_value_equality_l1549_154987

theorem absolute_value_equality : |5 - 3| = -(3 - 5) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l1549_154987


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l1549_154957

/-- The coordinates of a point symmetric to P(2,3) with respect to the x-axis are (2,-3) -/
theorem symmetric_point_x_axis : 
  let P : ℝ × ℝ := (2, 3)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point P = (2, -3) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l1549_154957


namespace NUMINAMATH_CALUDE_license_plate_increase_l1549_154927

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^5
  let new_plates := 26^4 * 10^4 * 2
  (new_plates : ℚ) / old_plates = 135.2 := by
sorry

end NUMINAMATH_CALUDE_license_plate_increase_l1549_154927
