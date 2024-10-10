import Mathlib

namespace shifts_needed_is_six_l2532_253262

/-- Represents the problem of assigning workers to shifts -/
structure ShiftAssignment where
  total_workers : ℕ
  workers_per_shift : ℕ
  total_assignments : ℕ

/-- Calculates the number of shifts needed -/
def number_of_shifts (assignment : ShiftAssignment) : ℕ :=
  assignment.total_workers / assignment.workers_per_shift

/-- Theorem stating that the number of shifts is 6 for the given conditions -/
theorem shifts_needed_is_six (assignment : ShiftAssignment) 
  (h1 : assignment.total_workers = 12)
  (h2 : assignment.workers_per_shift = 2)
  (h3 : assignment.total_assignments = 23760) :
  number_of_shifts assignment = 6 := by
  sorry

#eval number_of_shifts ⟨12, 2, 23760⟩

end shifts_needed_is_six_l2532_253262


namespace evaluate_expression_l2532_253206

theorem evaluate_expression (a : ℝ) :
  let x : ℝ := a + 9
  (x - a + 5) = 14 := by sorry

end evaluate_expression_l2532_253206


namespace chinese_english_difference_l2532_253249

/-- The number of hours Ryan spends learning English daily -/
def english_hours : ℕ := 6

/-- The number of hours Ryan spends learning Chinese daily -/
def chinese_hours : ℕ := 7

/-- The difference in hours between Chinese and English learning time -/
def learning_difference : ℕ := chinese_hours - english_hours

theorem chinese_english_difference :
  learning_difference = 1 :=
by sorry

end chinese_english_difference_l2532_253249


namespace linear_function_translation_l2532_253222

/-- Given a linear function y = 2x, when translated 3 units to the right along the x-axis,
    the resulting function is y = 2x - 6. -/
theorem linear_function_translation (x y : ℝ) :
  (y = 2 * x) →
  (y = 2 * (x - 3)) →
  (y = 2 * x - 6) :=
by sorry

end linear_function_translation_l2532_253222


namespace no_grasshopper_overlap_l2532_253246

/-- Represents the position of a grasshopper -/
structure Position where
  x : ℤ
  y : ℤ

/-- Represents the state of all four grasshoppers -/
structure GrasshopperState where
  g1 : Position
  g2 : Position
  g3 : Position
  g4 : Position

/-- Calculates the center of mass of three positions -/
def centerOfMass (p1 p2 p3 : Position) : Position :=
  { x := (p1.x + p2.x + p3.x) / 3,
    y := (p1.y + p2.y + p3.y) / 3 }

/-- Calculates the symmetric position of a point with respect to another point -/
def symmetricPosition (p center : Position) : Position :=
  { x := 2 * center.x - p.x,
    y := 2 * center.y - p.y }

/-- Performs a single jump for one grasshopper -/
def jump (state : GrasshopperState) (jumper : Fin 4) : GrasshopperState :=
  match jumper with
  | 0 => { state with g1 := symmetricPosition state.g1 (centerOfMass state.g2 state.g3 state.g4) }
  | 1 => { state with g2 := symmetricPosition state.g2 (centerOfMass state.g1 state.g3 state.g4) }
  | 2 => { state with g3 := symmetricPosition state.g3 (centerOfMass state.g1 state.g2 state.g4) }
  | 3 => { state with g4 := symmetricPosition state.g4 (centerOfMass state.g1 state.g2 state.g3) }

/-- Checks if any two grasshoppers are at the same position -/
def hasOverlap (state : GrasshopperState) : Prop :=
  state.g1 = state.g2 ∨ state.g1 = state.g3 ∨ state.g1 = state.g4 ∨
  state.g2 = state.g3 ∨ state.g2 = state.g4 ∨
  state.g3 = state.g4

/-- Initial state of the grasshoppers on a square -/
def initialState (n : ℕ) : GrasshopperState :=
  { g1 := { x := 0,     y := 0 },
    g2 := { x := 3^n,   y := 0 },
    g3 := { x := 3^n,   y := 3^n },
    g4 := { x := 0,     y := 3^n } }

/-- The main theorem stating that no overlap occurs after any number of jumps -/
theorem no_grasshopper_overlap (n : ℕ) :
  ∀ (jumps : List (Fin 4)), ¬(hasOverlap (jumps.foldl jump (initialState n))) :=
sorry

end no_grasshopper_overlap_l2532_253246


namespace parallel_vectors_sum_magnitude_l2532_253210

theorem parallel_vectors_sum_magnitude (x : ℝ) :
  let a : Fin 2 → ℝ := ![4, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), a = k • b) →
  ‖a + b‖ = 3 * Real.sqrt 5 := by
  sorry

end parallel_vectors_sum_magnitude_l2532_253210


namespace journey_speed_proof_l2532_253297

/-- Proves that given a journey of 120 miles in 90 minutes, where the average speed
    for the first 30 minutes was 70 mph and for the second 30 minutes was 75 mph,
    the average speed for the last 30 minutes must be 95 mph. -/
theorem journey_speed_proof (total_distance : ℝ) (total_time : ℝ) 
    (speed1 : ℝ) (speed2 : ℝ) (time_segment : ℝ) :
    total_distance = 120 →
    total_time = 1.5 →
    speed1 = 70 →
    speed2 = 75 →
    time_segment = 0.5 →
    speed1 * time_segment + speed2 * time_segment + 
    ((total_distance - (speed1 * time_segment + speed2 * time_segment)) / time_segment) = 
    total_distance / total_time :=
by sorry

end journey_speed_proof_l2532_253297


namespace park_bench_spaces_l2532_253283

/-- Calculates the number of available spaces on benches in a park. -/
def availableSpaces (numBenches : ℕ) (capacityPerBench : ℕ) (peopleSitting : ℕ) : ℕ :=
  numBenches * capacityPerBench - peopleSitting

/-- Theorem stating that there are 120 available spaces on the benches. -/
theorem park_bench_spaces :
  availableSpaces 50 4 80 = 120 := by
  sorry

end park_bench_spaces_l2532_253283


namespace negation_equivalence_l2532_253278

theorem negation_equivalence : 
  (¬(∀ x : ℝ, |x| ≥ 2 → (x ≥ 2 ∨ x ≤ -2))) ↔ 
  (∀ x : ℝ, |x| < 2 → (-2 < x ∧ x < 2)) :=
sorry

end negation_equivalence_l2532_253278


namespace inequality_proof_l2532_253241

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 1) : 
  (1/x) + (4/y) + (9/z) ≥ 36 := by
sorry

end inequality_proof_l2532_253241


namespace sum_of_xyz_equals_sqrt_13_l2532_253201

theorem sum_of_xyz_equals_sqrt_13 (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + y^2 + x*y = 3)
  (eq2 : y^2 + z^2 + y*z = 4)
  (eq3 : z^2 + x^2 + z*x = 7) :
  x + y + z = Real.sqrt 13 := by
sorry

end sum_of_xyz_equals_sqrt_13_l2532_253201


namespace grape_price_l2532_253252

/-- The price of each box of grapes given the following conditions:
  * 60 bundles of asparagus at $3.00 each
  * 40 boxes of grapes
  * 700 apples at $0.50 each
  * Total worth of the produce is $630
-/
theorem grape_price (asparagus_bundles : ℕ) (asparagus_price : ℚ)
                    (grape_boxes : ℕ) (apple_count : ℕ) (apple_price : ℚ)
                    (total_worth : ℚ) :
  asparagus_bundles = 60 →
  asparagus_price = 3 →
  grape_boxes = 40 →
  apple_count = 700 →
  apple_price = 1/2 →
  total_worth = 630 →
  (total_worth - (asparagus_bundles * asparagus_price + apple_count * apple_price)) / grape_boxes = 5/2 :=
by sorry

end grape_price_l2532_253252


namespace average_weight_b_c_l2532_253226

theorem average_weight_b_c (a b c : ℝ) : 
  (a + b + c) / 3 = 30 → 
  (a + b) / 2 = 25 → 
  b = 16 → 
  (b + c) / 2 = 28 := by
  sorry

end average_weight_b_c_l2532_253226


namespace zigzag_angle_l2532_253284

theorem zigzag_angle (ACB FEG DCE DEC : Real) (h1 : ACB = 80)
  (h2 : FEG = 64) (h3 : DCE + 80 + 14 = 180) (h4 : DEC + 64 + 33 = 180) :
  180 - DCE - DEC = 11 := by
  sorry

end zigzag_angle_l2532_253284


namespace cotton_candy_to_candy_bar_ratio_l2532_253286

/-- The price of candy bars, caramel, and cotton candy -/
structure CandyPrices where
  caramel : ℝ
  candy_bar : ℝ
  cotton_candy : ℝ

/-- The conditions of the candy pricing problem -/
def candy_pricing_conditions (p : CandyPrices) : Prop :=
  p.candy_bar = 2 * p.caramel ∧
  p.caramel = 3 ∧
  6 * p.candy_bar + 3 * p.caramel + p.cotton_candy = 57

/-- The theorem stating the ratio of cotton candy price to 4 candy bars -/
theorem cotton_candy_to_candy_bar_ratio (p : CandyPrices) 
  (h : candy_pricing_conditions p) : 
  p.cotton_candy / (4 * p.candy_bar) = 1 / 2 := by
  sorry

end cotton_candy_to_candy_bar_ratio_l2532_253286


namespace arc_length_central_angle_l2532_253254

theorem arc_length_central_angle (r : ℝ) (θ : ℝ) (h : θ = π / 2) :
  let circum := 2 * π * r
  let arc_length := (θ / (2 * π)) * circum
  r = 15 → arc_length = 7.5 * π := by
  sorry

end arc_length_central_angle_l2532_253254


namespace picture_book_shelves_l2532_253236

theorem picture_book_shelves (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ) : 
  books_per_shelf = 7 → 
  mystery_shelves = 8 → 
  total_books = 70 → 
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 2 := by
sorry

end picture_book_shelves_l2532_253236


namespace regular_polygon_with_150_degree_angles_l2532_253255

theorem regular_polygon_with_150_degree_angles (n : ℕ) : 
  (n ≥ 3) →  -- Ensuring the polygon has at least 3 sides
  (∀ angle : ℝ, angle = 150 → 180 * (n - 2) = n * angle) →
  n = 12 := by
sorry

end regular_polygon_with_150_degree_angles_l2532_253255


namespace rectangle_dimensions_l2532_253231

theorem rectangle_dimensions : ∀ x y : ℝ,
  y = 2 * x →
  2 * (x + y) = 2 * (x * y) →
  x = (3 : ℝ) / 2 ∧ y = 3 := by
  sorry

end rectangle_dimensions_l2532_253231


namespace geometric_sequence_common_ratio_l2532_253214

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 8)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1))
  (h3 : a 4 = a 3 * a 5) :
  a 2 / a 1 = 1/2 := by
sorry

end geometric_sequence_common_ratio_l2532_253214


namespace problem_solution_l2532_253202

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 4*x - 5 ≤ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem problem_solution :
  (∀ m : ℝ, m > 0 → (∀ x : ℝ, p x → q x m) → m ≥ 4) ∧
  (∀ x : ℝ, (p x ∨ q x 5) ∧ ¬(p x ∧ q x 5) → (x ∈ Set.Icc (-4 : ℝ) (-1) ∪ Set.Ioc 5 6)) :=
sorry

end problem_solution_l2532_253202


namespace correct_division_result_l2532_253225

theorem correct_division_result (incorrect_divisor incorrect_quotient correct_divisor : ℕ) 
  (h1 : incorrect_divisor = 72)
  (h2 : incorrect_quotient = 24)
  (h3 : correct_divisor = 36) :
  (incorrect_divisor * incorrect_quotient) / correct_divisor = 48 :=
by sorry

end correct_division_result_l2532_253225


namespace greatest_n_value_l2532_253276

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 6400) : n ≤ 7 ∧ ∃ m : ℤ, m = 7 ∧ 101 * m^2 ≤ 6400 := by
  sorry

end greatest_n_value_l2532_253276


namespace total_students_l2532_253251

theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 120) : 
  boys + girls = 312 := by
sorry

end total_students_l2532_253251


namespace inequality_solution_set_l2532_253207

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - x - 2) / (x - 4) ≥ 3 ↔ x > 4 :=
sorry

end inequality_solution_set_l2532_253207


namespace farm_tax_percentage_l2532_253273

/-- Given a village's farm tax collection and information about Mr. Willam's tax and land,
    prove that the percentage of cultivated land taxed is 12.5%. -/
theorem farm_tax_percentage (total_tax village_tax willam_tax : ℚ) (willam_land_percentage : ℚ) :
  total_tax = 4000 →
  willam_tax = 500 →
  willam_land_percentage = 20833333333333332 / 100000000000000000 →
  (willam_tax / total_tax) * 100 = 125 / 10 :=
by sorry

end farm_tax_percentage_l2532_253273


namespace initial_apples_count_l2532_253242

/-- The number of apples initially on the tree -/
def initial_apples : ℕ := sorry

/-- The number of apples picked from the tree -/
def picked_apples : ℕ := 7

/-- The number of new apples that grew on the tree -/
def new_apples : ℕ := 2

/-- The number of apples remaining on the tree -/
def remaining_apples : ℕ := 6

/-- Theorem stating that the initial number of apples is 11 -/
theorem initial_apples_count : initial_apples = 11 := by
  sorry

end initial_apples_count_l2532_253242


namespace third_day_temperature_l2532_253208

/-- Given the average temperature of three days and the temperatures of two of those days,
    calculate the temperature of the third day. -/
theorem third_day_temperature
  (avg_temp : ℚ)
  (day1_temp : ℚ)
  (day3_temp : ℚ)
  (h1 : avg_temp = -7)
  (h2 : day1_temp = -14)
  (h3 : day3_temp = 1)
  : (3 * avg_temp - day1_temp - day3_temp : ℚ) = -8 := by
  sorry

end third_day_temperature_l2532_253208


namespace trent_travel_distance_l2532_253264

/-- The distance Trent walked from his house to the bus stop -/
def distance_to_bus_stop : ℕ := 4

/-- The distance Trent rode the bus to the library -/
def distance_on_bus : ℕ := 7

/-- The total distance Trent traveled in blocks -/
def total_distance : ℕ := 2 * (distance_to_bus_stop + distance_on_bus)

theorem trent_travel_distance : total_distance = 22 := by
  sorry

end trent_travel_distance_l2532_253264


namespace ratio_of_numbers_l2532_253229

theorem ratio_of_numbers (x y : ℝ) (h : x > y) (h' : (x + y) / (x - y) = 4 / 3) :
  x / y = 7 := by
  sorry

end ratio_of_numbers_l2532_253229


namespace abs_func_no_opposite_signs_l2532_253215

-- Define the absolute value function
def abs_func (x : ℝ) : ℝ := |x|

-- Theorem statement
theorem abs_func_no_opposite_signs :
  ∀ (a b : ℝ), (abs_func a) * (abs_func b) ≥ 0 := by sorry

end abs_func_no_opposite_signs_l2532_253215


namespace task_completion_probability_l2532_253217

theorem task_completion_probability 
  (p1 : ℚ) (p2 : ℚ) 
  (h1 : p1 = 3 / 8) 
  (h2 : p2 = 3 / 5) : 
  p1 * (1 - p2) = 3 / 20 := by
sorry

end task_completion_probability_l2532_253217


namespace wine_cost_theorem_l2532_253223

/-- The cost of a bottle of wine with a cork -/
def wineWithCorkCost (corkCost : ℝ) (extraCost : ℝ) : ℝ :=
  corkCost + (corkCost + extraCost)

/-- Theorem: The cost of a bottle of wine with a cork is $6.10 -/
theorem wine_cost_theorem (corkCost : ℝ) (extraCost : ℝ)
  (h1 : corkCost = 2.05)
  (h2 : extraCost = 2.00) :
  wineWithCorkCost corkCost extraCost = 6.10 := by
  sorry

#eval wineWithCorkCost 2.05 2.00

end wine_cost_theorem_l2532_253223


namespace monotonic_decreasing_interval_of_f_l2532_253294

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 4*x + 3)

theorem monotonic_decreasing_interval_of_f :
  ∀ x₁ x₂, x₁ < x₂ → x₁ < 1 → x₂ < 1 → f x₁ > f x₂ :=
by sorry

end monotonic_decreasing_interval_of_f_l2532_253294


namespace ratio_problem_l2532_253244

theorem ratio_problem (a b : ℚ) (h1 : b / a = 5) (h2 : b = 18 - 3 * a) : a = 9 / 4 := by
  sorry

end ratio_problem_l2532_253244


namespace parabola_directrix_l2532_253250

/-- Given a parabola with equation y = -1/4 * x^2, its directrix has the equation y = 1 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = -1/4 * x^2) → (∃ (k : ℝ), k = 1 ∧ k = y + 1/4) :=
by sorry

end parabola_directrix_l2532_253250


namespace sum_of_number_and_its_square_l2532_253227

theorem sum_of_number_and_its_square (x : ℕ) : x = 14 → x + x^2 = 210 := by
  sorry

end sum_of_number_and_its_square_l2532_253227


namespace rectangle_problem_l2532_253247

theorem rectangle_problem (a b k l : ℕ) (h1 : k * l = 47 * (a + b)) 
  (h2 : a * k = b * l) : k = 2256 := by
  sorry

end rectangle_problem_l2532_253247


namespace diagonal_smallest_angle_at_midpoints_l2532_253271

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with side length a -/
structure Cube where
  a : ℝ
  center : Point3D

/-- Calculates the angle at which the diagonal is seen from a point on the cube's surface -/
noncomputable def angleFromPoint (c : Cube) (p : Point3D) : ℝ := sorry

/-- Checks if a point is on the surface of the cube -/
def isOnSurface (c : Cube) (p : Point3D) : Prop := sorry

/-- Calculates the midpoints of the cube's faces -/
def faceMidpoints (c : Cube) : List Point3D := sorry

/-- Main theorem: The diagonal is seen at the smallest angle from the midpoints of the cube's faces -/
theorem diagonal_smallest_angle_at_midpoints (c : Cube) :
  ∀ p : Point3D, isOnSurface c p →
    (p ∉ faceMidpoints c → 
      ∀ m ∈ faceMidpoints c, angleFromPoint c p > angleFromPoint c m) :=
sorry

end diagonal_smallest_angle_at_midpoints_l2532_253271


namespace water_bucket_addition_l2532_253287

theorem water_bucket_addition (initial_water : Real) (added_water : Real) :
  initial_water = 3 ∧ added_water = 6.8 → initial_water + added_water = 9.8 := by
  sorry

end water_bucket_addition_l2532_253287


namespace smallest_number_with_remainders_l2532_253265

theorem smallest_number_with_remainders : ∃ n : ℕ,
  (n % 10 = 9) ∧
  (n % 9 = 8) ∧
  (n % 8 = 7) ∧
  (n % 7 = 6) ∧
  (n % 6 = 5) ∧
  (n % 5 = 4) ∧
  (n % 4 = 3) ∧
  (n % 3 = 2) ∧
  (n % 2 = 1) ∧
  (∀ m : ℕ, m < n →
    ¬((m % 10 = 9) ∧
      (m % 9 = 8) ∧
      (m % 8 = 7) ∧
      (m % 7 = 6) ∧
      (m % 6 = 5) ∧
      (m % 5 = 4) ∧
      (m % 4 = 3) ∧
      (m % 3 = 2) ∧
      (m % 2 = 1))) ∧
  n = 2519 :=
by sorry

end smallest_number_with_remainders_l2532_253265


namespace range_of_a_l2532_253256

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x - a ≥ 0) → a ≤ 8 := by
  sorry

end range_of_a_l2532_253256


namespace total_chocolates_in_large_box_l2532_253285

/-- Represents the number of small boxes in the large box -/
def num_small_boxes : ℕ := 19

/-- Represents the number of chocolate bars in each small box -/
def chocolates_per_small_box : ℕ := 25

/-- Theorem stating that the total number of chocolate bars in the large box is 475 -/
theorem total_chocolates_in_large_box : 
  num_small_boxes * chocolates_per_small_box = 475 := by
  sorry

end total_chocolates_in_large_box_l2532_253285


namespace product_in_M_l2532_253272

/-- The set M of differences of squares of integers -/
def M : Set ℤ := {x | ∃ a b : ℤ, x = a^2 - b^2}

/-- Theorem: The product of any two elements in M is also in M -/
theorem product_in_M (p q : ℤ) (hp : p ∈ M) (hq : q ∈ M) : p * q ∈ M := by
  sorry

end product_in_M_l2532_253272


namespace complex_number_in_third_quadrant_l2532_253233

theorem complex_number_in_third_quadrant : 
  let z : ℂ := Complex.I * ((-2 : ℂ) + Complex.I)
  (z.re < 0) ∧ (z.im < 0) := by sorry

end complex_number_in_third_quadrant_l2532_253233


namespace troll_count_l2532_253234

theorem troll_count (P B T : ℕ) : 
  P = 6 → 
  B = 4 * P - 6 → 
  T = B / 2 → 
  P + B + T = 33 := by
sorry

end troll_count_l2532_253234


namespace line_equation_l2532_253293

-- Define a line in 2D space
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define when a point lies on a line
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem line_equation (l : Line2D) :
  (pointOnLine ⟨0, 0⟩ l) →
  (perpendicular l ⟨1, -1, -3⟩) →
  l = ⟨1, 1, 0⟩ := by
  sorry

end line_equation_l2532_253293


namespace parallel_perpendicular_plane_l2532_253279

/-- Two lines are parallel -/
def parallel (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular (l : Line) (α : Plane) : Prop := sorry

/-- Main theorem: If two lines are parallel and one is perpendicular to a plane, 
    then the other is also perpendicular to that plane -/
theorem parallel_perpendicular_plane (m n : Line) (α : Plane) :
  parallel m n → perpendicular m α → perpendicular n α := by sorry

end parallel_perpendicular_plane_l2532_253279


namespace two_digit_gcd_theorem_l2532_253295

/-- Represents a two-digit decimal number as a pair of natural numbers (a, b) -/
def TwoDigitNumber := { p : ℕ × ℕ // p.1 ≤ 9 ∧ p.2 ≤ 9 ∧ p.1 ≠ 0 }

/-- Converts a two-digit number (a, b) to its decimal representation ab -/
def toDecimal (n : TwoDigitNumber) : ℕ :=
  10 * n.val.1 + n.val.2

/-- Converts a two-digit number (a, b) to its reversed decimal representation ba -/
def toReversedDecimal (n : TwoDigitNumber) : ℕ :=
  10 * n.val.2 + n.val.1

/-- Checks if a two-digit number satisfies the GCD condition -/
def satisfiesGCDCondition (n : TwoDigitNumber) : Prop :=
  Nat.gcd (toDecimal n) (toReversedDecimal n) = n.val.1^2 - n.val.2^2

theorem two_digit_gcd_theorem :
  ∃ (n1 n2 : TwoDigitNumber),
    satisfiesGCDCondition n1 ∧
    satisfiesGCDCondition n2 ∧
    toDecimal n1 = 21 ∧
    toDecimal n2 = 54 ∧
    (∀ (n : TwoDigitNumber), satisfiesGCDCondition n → (toDecimal n = 21 ∨ toDecimal n = 54)) :=
  sorry

end two_digit_gcd_theorem_l2532_253295


namespace similar_quadratic_radicals_l2532_253212

def are_similar_quadratic_radicals (a b : ℝ) : Prop :=
  ∃ (k : ℚ), a = k * b

theorem similar_quadratic_radicals :
  are_similar_quadratic_radicals (Real.sqrt 18) (Real.sqrt 72) ∧
  ¬ are_similar_quadratic_radicals (Real.sqrt 12) (Real.sqrt 18) ∧
  ¬ are_similar_quadratic_radicals (Real.sqrt 20) (Real.sqrt 50) ∧
  ¬ are_similar_quadratic_radicals (Real.sqrt 24) (Real.sqrt 32) :=
by sorry

end similar_quadratic_radicals_l2532_253212


namespace quadratic_inequality_implies_a_bound_l2532_253296

theorem quadratic_inequality_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 2, a * x^2 - 2*x + 2 > 0) →
  a > 1/2 := by
  sorry

end quadratic_inequality_implies_a_bound_l2532_253296


namespace business_partnership_timing_l2532_253269

/-- Represents the number of months after A started that B joined the business. -/
def months_until_b_joined : ℕ → Prop :=
  fun x =>
    let a_investment := 3500 * 12
    let b_investment := 21000 * (12 - x)
    a_investment * 3 = b_investment * 2

theorem business_partnership_timing :
  ∃ x : ℕ, months_until_b_joined x ∧ x = 9 := by
  sorry

end business_partnership_timing_l2532_253269


namespace orange_box_capacity_l2532_253270

/-- 
Given two boxes for carrying oranges, where:
- The first box has a capacity of 80 and is filled 3/4 full
- The second box has an unknown capacity C and is filled 3/5 full
- The total number of oranges in both boxes is 90

This theorem proves that the capacity C of the second box is 50.
-/
theorem orange_box_capacity 
  (box1_capacity : ℕ) 
  (box1_fill : ℚ) 
  (box2_fill : ℚ) 
  (total_oranges : ℕ) 
  (h1 : box1_capacity = 80)
  (h2 : box1_fill = 3/4)
  (h3 : box2_fill = 3/5)
  (h4 : total_oranges = 90) :
  ∃ (C : ℕ), box1_fill * box1_capacity + box2_fill * C = total_oranges ∧ C = 50 := by
sorry

end orange_box_capacity_l2532_253270


namespace intersection_complement_equals_set_l2532_253203

def U : Set ℕ := {x | x^2 - 4*x - 5 ≤ 0}
def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 3, 5}

theorem intersection_complement_equals_set : A ∩ (U \ B) = {0, 2} := by
  sorry

end intersection_complement_equals_set_l2532_253203


namespace product_mod_seven_l2532_253228

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end product_mod_seven_l2532_253228


namespace photograph_perimeter_l2532_253282

theorem photograph_perimeter (w h m : ℝ) 
  (border_1 : (w + 2) * (h + 2) = m)
  (border_3 : (w + 6) * (h + 6) = m + 52) :
  2 * w + 2 * h = 10 := by
  sorry

end photograph_perimeter_l2532_253282


namespace adults_cookie_fraction_l2532_253245

theorem adults_cookie_fraction (total_cookies : ℕ) (num_children : ℕ) (cookies_per_child : ℕ) :
  total_cookies = 120 →
  num_children = 4 →
  cookies_per_child = 20 →
  (total_cookies - num_children * cookies_per_child : ℚ) / total_cookies = 1 / 3 := by
  sorry

end adults_cookie_fraction_l2532_253245


namespace min_value_implies_a_f_less_than_x_squared_l2532_253267

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

theorem min_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ 3/2) ∧
  (∃ x ∈ Set.Icc 1 (Real.exp 1), f a x = 3/2) →
  a = -Real.sqrt (Real.exp 1) :=
sorry

theorem f_less_than_x_squared (a : ℝ) :
  (∀ x > 1, f a x < x^2) →
  a ≥ -1 :=
sorry

end min_value_implies_a_f_less_than_x_squared_l2532_253267


namespace incorrect_propositions_l2532_253213

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop := sorry

/-- A point lies on a line -/
def point_on_line (p : Point3D) (l : Line3D) : Prop := sorry

/-- A point lies on a plane -/
def point_on_plane (p : Point3D) (plane : Plane3D) : Prop := sorry

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Line3D) : Prop := sorry

/-- Two lines are skew (not parallel and not intersecting) -/
def skew (l1 l2 : Line3D) : Prop := sorry

/-- A line lies on a plane -/
def line_on_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Three points determine a plane -/
def points_determine_plane (p1 p2 p3 : Point3D) : Prop := sorry

theorem incorrect_propositions :
  -- Proposition ③: Three points on two intersecting lines determine a plane
  ¬ (∀ (l1 l2 : Line3D) (p1 p2 p3 : Point3D),
    intersect l1 l2 →
    point_on_line p1 l1 →
    point_on_line p2 l1 →
    point_on_line p3 l2 →
    points_determine_plane p1 p2 p3) ∧
  -- Proposition ④: Two perpendicular lines are coplanar
  ¬ (∀ (l1 l2 : Line3D),
    perpendicular l1 l2 →
    ∃ (p : Plane3D), line_on_plane l1 p ∧ line_on_plane l2 p) :=
by sorry

end incorrect_propositions_l2532_253213


namespace f_derivative_at_one_l2532_253260

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem f_derivative_at_one :
  (deriv f) 1 = 2 * Real.exp 1 := by
sorry

end f_derivative_at_one_l2532_253260


namespace percentage_equality_l2532_253253

theorem percentage_equality (x y : ℝ) (h1 : 3 * x = 3/4 * y) (h2 : x = 20) : y + 10 = 90 := by
  sorry

end percentage_equality_l2532_253253


namespace min_students_same_score_l2532_253239

theorem min_students_same_score (total_students : ℕ) (min_score max_score : ℕ) :
  total_students = 8000 →
  min_score = 30 →
  max_score = 83 →
  ∃ (score : ℕ), min_score ≤ score ∧ score ≤ max_score ∧
    (∃ (students_with_score : ℕ), students_with_score ≥ 149 ∧
      (∀ (s : ℕ), min_score ≤ s ∧ s ≤ max_score →
        (∃ (students : ℕ), students ≤ students_with_score))) :=
by sorry

end min_students_same_score_l2532_253239


namespace simple_interest_ratio_l2532_253259

/-- Simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- The ratio of final amount to initial amount after simple interest --/
def final_to_initial_ratio (rate : ℝ) (time : ℝ) : ℝ :=
  1 + rate * time

theorem simple_interest_ratio :
  let rate : ℝ := 0.1
  let time : ℝ := 10
  final_to_initial_ratio rate time = 2 := by
  sorry

end simple_interest_ratio_l2532_253259


namespace carol_initial_blocks_l2532_253200

/-- The number of blocks Carol started with -/
def initial_blocks : ℕ := sorry

/-- The number of blocks Carol lost -/
def lost_blocks : ℕ := 25

/-- The number of blocks Carol ended with -/
def final_blocks : ℕ := 17

/-- Theorem stating that Carol started with 42 blocks -/
theorem carol_initial_blocks : initial_blocks = 42 := by sorry

end carol_initial_blocks_l2532_253200


namespace stock_percentage_problem_l2532_253204

/-- Calculates the percentage of a stock given income, investment, and stock price. -/
def stock_percentage (income : ℚ) (investment : ℚ) (stock_price : ℚ) : ℚ :=
  (income * stock_price) / investment

/-- Theorem stating that given the specific values in the problem, the stock percentage is 30%. -/
theorem stock_percentage_problem :
  let income : ℚ := 500
  let investment : ℚ := 1500
  let stock_price : ℚ := 90
  stock_percentage income investment stock_price = 30 := by sorry

end stock_percentage_problem_l2532_253204


namespace train_crossing_time_l2532_253219

theorem train_crossing_time (length : ℝ) (time_second : ℝ) (crossing_time : ℝ) : 
  length = 120 →
  time_second = 12 →
  crossing_time = 10.909090909090908 →
  (length / (length / time_second + (2 * length) / crossing_time - length / time_second)) = 10 :=
by sorry

end train_crossing_time_l2532_253219


namespace inequality_solution_set_l2532_253240

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - x - 6) / (x - 1) > 0 ↔ (-2 < x ∧ x < 1) ∨ x > 3 :=
sorry

end inequality_solution_set_l2532_253240


namespace stanley_run_distance_l2532_253268

def distance_walked : ℝ := 0.2
def additional_distance : ℝ := 0.2

theorem stanley_run_distance :
  distance_walked + additional_distance = 0.4 := by sorry

end stanley_run_distance_l2532_253268


namespace min_value_reciprocal_sum_l2532_253211

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1) + x + Real.sin x

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : f (4 * a) + f (b - 9) = 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → f (4 * x) + f (y - 9) = 0 → 1 / x + 1 / y ≥ 1) ∧ 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ f (4 * x) + f (y - 9) = 0 ∧ 1 / x + 1 / y = 1) :=
by sorry

end min_value_reciprocal_sum_l2532_253211


namespace detergent_calculation_l2532_253216

/-- Calculates the total amount of detergent used for washing clothes -/
theorem detergent_calculation (total_clothes cotton_clothes woolen_clothes : ℝ)
  (cotton_detergent wool_detergent : ℝ) : 
  total_clothes = cotton_clothes + woolen_clothes →
  cotton_clothes = 4 →
  woolen_clothes = 5 →
  cotton_detergent = 2 →
  wool_detergent = 1.5 →
  cotton_clothes * cotton_detergent + woolen_clothes * wool_detergent = 15.5 := by
  sorry

end detergent_calculation_l2532_253216


namespace ratio_fraction_equality_l2532_253280

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end ratio_fraction_equality_l2532_253280


namespace cricket_bat_profit_percentage_l2532_253232

theorem cricket_bat_profit_percentage
  (cost_price_A : ℝ)
  (selling_price_C : ℝ)
  (profit_percentage_B : ℝ)
  (h1 : cost_price_A = 156)
  (h2 : selling_price_C = 234)
  (h3 : profit_percentage_B = 25)
  : (((selling_price_C / (1 + profit_percentage_B / 100)) - cost_price_A) / cost_price_A) * 100 = 20 := by
  sorry

end cricket_bat_profit_percentage_l2532_253232


namespace roots_equation_sum_l2532_253220

theorem roots_equation_sum (α β : ℝ) : 
  α^2 - 3*α - 4 = 0 → β^2 - 3*β - 4 = 0 → 3*α^3 + 7*β^4 = 1591 :=
by
  sorry

end roots_equation_sum_l2532_253220


namespace f_min_value_inequality_proof_l2532_253218

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem for the minimum value of f(x)
theorem f_min_value : ∃ (a : ℝ), ∀ (x : ℝ), f x ≥ a ∧ ∃ (y : ℝ), f y = a :=
sorry

-- Theorem for the inequality
theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m^3 + n^3 = 2) : m + n ≤ 2 :=
sorry

end f_min_value_inequality_proof_l2532_253218


namespace infinitely_many_a_making_n4_plus_a_composite_l2532_253275

theorem infinitely_many_a_making_n4_plus_a_composite :
  ∀ k : ℕ, k > 1 → ∃ a : ℕ, a = 4 * k^4 ∧ ∀ n : ℕ, ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ n^4 + a = x * y :=
sorry

end infinitely_many_a_making_n4_plus_a_composite_l2532_253275


namespace plot_length_is_60_l2532_253298

/-- Represents a rectangular plot with its dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencingCostPerMetre : ℝ
  totalFencingCost : ℝ

/-- The length of the plot is 20 metres more than its breadth. -/
def lengthCondition (plot : RectangularPlot) : Prop :=
  plot.length = plot.breadth + 20

/-- The cost of fencing the plot at the given rate equals the total fencing cost. -/
def fencingCostCondition (plot : RectangularPlot) : Prop :=
  plot.fencingCostPerMetre * (2 * plot.length + 2 * plot.breadth) = plot.totalFencingCost

/-- The theorem stating that under the given conditions, the length of the plot is 60 metres. -/
theorem plot_length_is_60 (plot : RectangularPlot)
    (h1 : lengthCondition plot)
    (h2 : fencingCostCondition plot)
    (h3 : plot.fencingCostPerMetre = 26.5)
    (h4 : plot.totalFencingCost = 5300) :
    plot.length = 60 := by
  sorry


end plot_length_is_60_l2532_253298


namespace janes_drinks_l2532_253299

theorem janes_drinks (b m d : ℕ) : 
  b + m + d = 5 →
  (90 * b + 40 * m + 30 * d) % 100 = 0 →
  d = 4 := by
sorry

end janes_drinks_l2532_253299


namespace paper_area_difference_l2532_253205

/-- The combined area (front and back) of a rectangular sheet of paper -/
def combinedArea (length width : ℝ) : ℝ := 2 * length * width

/-- The difference in combined area between two rectangular sheets of paper -/
def areaDifference (l1 w1 l2 w2 : ℝ) : ℝ :=
  combinedArea l1 w1 - combinedArea l2 w2

theorem paper_area_difference :
  areaDifference 11 19 9.5 11 = 209 := by sorry

end paper_area_difference_l2532_253205


namespace no_valid_x_l2532_253209

/-- A circle in the xy-plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point is on a circle --/
def is_on_circle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

theorem no_valid_x : 
  ∀ x : ℝ, ¬∃ (c : Circle), 
    c.center = (15, 0) ∧ 
    c.radius = 15 ∧
    is_on_circle c (x, 18) ∧ 
    is_on_circle c (x, -18) := by
  sorry

end no_valid_x_l2532_253209


namespace percentage_calculation_l2532_253266

theorem percentage_calculation (N : ℝ) (P : ℝ) 
  (h1 : N = 125) 
  (h2 : N = (P / 100) * N + 105) : 
  P = 16 := by
  sorry

end percentage_calculation_l2532_253266


namespace evaluate_expression_1_evaluate_expression_2_l2532_253248

-- Question 1
theorem evaluate_expression_1 :
  (3 * Real.sqrt 27 - 2 * Real.sqrt 12) * (2 * Real.sqrt (5 + 1/3) + 3 * Real.sqrt (8 + 1/3)) = 115 := by
  sorry

-- Question 2
theorem evaluate_expression_2 :
  (5 * Real.sqrt 21 - 3 * Real.sqrt 15) / (5 * Real.sqrt (2 + 2/3) - 3 * Real.sqrt (1 + 2/3)) = 3 := by
  sorry

end evaluate_expression_1_evaluate_expression_2_l2532_253248


namespace quadratic_equation_solution_l2532_253237

theorem quadratic_equation_solution :
  let x₁ : ℝ := (3 + Real.sqrt 41) / 4
  let x₂ : ℝ := (3 - Real.sqrt 41) / 4
  2 * x₁^2 - 3 * x₁ - 4 = 0 ∧ 2 * x₂^2 - 3 * x₂ - 4 = 0 := by
  sorry

end quadratic_equation_solution_l2532_253237


namespace vector_midpoint_dot_product_l2532_253235

def problem (a b : ℝ × ℝ) : Prop :=
  let m : ℝ × ℝ := (4, 10)
  m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2) ∧
  a.1 * b.1 + a.2 * b.2 = 10 →
  a.1^2 + a.2^2 + b.1^2 + b.2^2 = 444

theorem vector_midpoint_dot_product :
  ∀ a b : ℝ × ℝ, problem a b :=
by
  sorry

end vector_midpoint_dot_product_l2532_253235


namespace x_minus_y_value_l2532_253263

theorem x_minus_y_value (x y : ℝ) (h : x^2 + 6*x + 9 + Real.sqrt (y - 3) = 0) : 
  x - y = -6 := by
sorry

end x_minus_y_value_l2532_253263


namespace jerry_action_figures_count_l2532_253257

/-- Calculates the total number of action figures on Jerry's shelf --/
def total_action_figures (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that the total number of action figures is the sum of initial and added figures --/
theorem jerry_action_figures_count (initial : ℕ) (added : ℕ) :
  total_action_figures initial added = initial + added :=
by sorry

end jerry_action_figures_count_l2532_253257


namespace orvin_balloon_purchase_l2532_253261

/-- Represents the cost of balloons in cents -/
def regular_price : ℕ := 200

/-- Represents the total amount of money Orvin has in cents -/
def total_money : ℕ := 40 * regular_price

/-- Represents the cost of a pair of balloons (one at regular price, one at half price) in cents -/
def pair_cost : ℕ := regular_price + regular_price / 2

/-- The maximum number of balloons Orvin can buy -/
def max_balloons : ℕ := 2 * (total_money / pair_cost)

theorem orvin_balloon_purchase :
  max_balloons = 52 := by sorry

end orvin_balloon_purchase_l2532_253261


namespace bisection_next_point_l2532_253258

-- Define the function f(x) = x^3 + x - 3
def f (x : ℝ) : ℝ := x^3 + x - 3

-- Define the initial interval
def a : ℝ := 0
def b : ℝ := 2

-- Define the first midpoint
def m₁ : ℝ := 1

-- Theorem statement
theorem bisection_next_point :
  f a < 0 ∧ f b > 0 ∧ f m₁ < 0 →
  (a + b) / 2 = 1.5 := by sorry

end bisection_next_point_l2532_253258


namespace trigonometric_identities_l2532_253274

theorem trigonometric_identities :
  (Real.sin (75 * π / 180))^2 - (Real.cos (75 * π / 180))^2 = Real.sqrt 3 / 2 ∧
  Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 4 :=
by sorry

end trigonometric_identities_l2532_253274


namespace mikes_shopping_cost_l2532_253243

/-- The total amount Mike spent on shopping --/
def total_spent (food_cost wallet_cost shirt_cost : ℝ) : ℝ :=
  food_cost + wallet_cost + shirt_cost

/-- Theorem stating the total amount Mike spent on shopping --/
theorem mikes_shopping_cost :
  ∀ (food_cost wallet_cost shirt_cost : ℝ),
    food_cost = 30 →
    wallet_cost = food_cost + 60 →
    shirt_cost = wallet_cost / 3 →
    total_spent food_cost wallet_cost shirt_cost = 150 := by
  sorry

end mikes_shopping_cost_l2532_253243


namespace sequence_property_l2532_253290

theorem sequence_property (n : ℕ) (x : ℕ → ℚ) (h_n : n ≥ 7) 
  (h_def : ∀ k > 1, x k = 1 / (1 - x (k-1)))
  (h_x2 : x 2 = 5) : 
  x 7 = 4/5 := by
sorry

end sequence_property_l2532_253290


namespace business_partnership_gains_l2532_253277

/-- Represents the investment and gain of a partner in the business. -/
structure Partner where
  investment : ℕ
  time : ℕ
  gain : ℕ

/-- Represents the business partnership with four partners. -/
def BusinessPartnership (nandan gopal vishal krishan : Partner) : Prop :=
  -- Investment ratios
  krishan.investment = 6 * nandan.investment ∧
  gopal.investment = 3 * nandan.investment ∧
  vishal.investment = 2 * nandan.investment ∧
  -- Time ratios
  krishan.time = 2 * nandan.time ∧
  gopal.time = 3 * nandan.time ∧
  vishal.time = nandan.time ∧
  -- Nandan's gain
  nandan.gain = 6000 ∧
  -- Gain proportionality
  krishan.gain * nandan.investment * nandan.time = nandan.gain * krishan.investment * krishan.time ∧
  gopal.gain * nandan.investment * nandan.time = nandan.gain * gopal.investment * gopal.time ∧
  vishal.gain * nandan.investment * nandan.time = nandan.gain * vishal.investment * vishal.time

/-- The theorem to be proved -/
theorem business_partnership_gains 
  (nandan gopal vishal krishan : Partner) 
  (h : BusinessPartnership nandan gopal vishal krishan) : 
  krishan.gain = 72000 ∧ 
  gopal.gain = 54000 ∧ 
  vishal.gain = 12000 ∧ 
  nandan.gain + gopal.gain + vishal.gain + krishan.gain = 144000 := by
  sorry

end business_partnership_gains_l2532_253277


namespace problem_solution_l2532_253230

/-- The function f as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 else x^2 + a*x

/-- Theorem stating the solution to the problem -/
theorem problem_solution (a : ℝ) : f a (f a 0) = 4 * a → a = 2 := by
  sorry

end problem_solution_l2532_253230


namespace gold_alloy_composition_l2532_253289

/-- Proves that adding 12 ounces of pure gold to an alloy weighing 48 ounces
    that is 25% gold will result in an alloy that is 40% gold. -/
theorem gold_alloy_composition (initial_weight : ℝ) (initial_gold_percentage : ℝ) 
    (final_gold_percentage : ℝ) (added_gold : ℝ) : 
  initial_weight = 48 →
  initial_gold_percentage = 0.25 →
  final_gold_percentage = 0.40 →
  added_gold = 12 →
  (initial_weight * initial_gold_percentage + added_gold) / (initial_weight + added_gold) = final_gold_percentage :=
by
  sorry

end gold_alloy_composition_l2532_253289


namespace milk_sold_in_fl_oz_l2532_253288

def monday_morning_milk : ℕ := 150 * 250 + 40 * 300 + 50 * 350
def monday_evening_milk : ℕ := 50 * 400 + 25 * 500 + 25 * 450
def tuesday_morning_milk : ℕ := 24 * 300 + 18 * 350 + 18 * 400
def tuesday_evening_milk : ℕ := 50 * 450 + 70 * 500 + 80 * 550

def total_milk_bought : ℕ := monday_morning_milk + monday_evening_milk + tuesday_morning_milk + tuesday_evening_milk
def remaining_milk : ℕ := 84000
def ml_per_fl_oz : ℕ := 30

theorem milk_sold_in_fl_oz :
  (total_milk_bought - remaining_milk) / ml_per_fl_oz = 4215 := by sorry

end milk_sold_in_fl_oz_l2532_253288


namespace division_reduction_l2532_253224

theorem division_reduction (x : ℕ) (h : x > 0) : 36 / x = 36 - 24 → x = 3 := by
  sorry

end division_reduction_l2532_253224


namespace square_sum_nonzero_implies_nonzero_element_l2532_253281

theorem square_sum_nonzero_implies_nonzero_element (a b : ℝ) : 
  a^2 + b^2 ≠ 0 → (a ≠ 0 ∨ b ≠ 0) := by
  sorry

end square_sum_nonzero_implies_nonzero_element_l2532_253281


namespace dog_food_calculation_l2532_253292

theorem dog_food_calculation (num_dogs : ℕ) (food_per_dog : ℕ) (vacation_days : ℕ) :
  num_dogs = 4 →
  food_per_dog = 250 →
  vacation_days = 14 →
  (num_dogs * food_per_dog * vacation_days : ℕ) / 1000 = 14 := by
  sorry

end dog_food_calculation_l2532_253292


namespace total_tiles_l2532_253238

theorem total_tiles (yellow blue purple white : ℕ) : 
  yellow = 3 → 
  blue = yellow + 1 → 
  purple = 6 → 
  white = 7 → 
  yellow + blue + purple + white = 20 := by
sorry

end total_tiles_l2532_253238


namespace completing_square_equivalence_l2532_253221

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 2*x = 2 ↔ (x - 1)^2 = 3 :=
by sorry

end completing_square_equivalence_l2532_253221


namespace rectangle_breadth_ratio_l2532_253291

/-- Given a rectangle where the length is halved and the area is reduced by 50%,
    prove that the ratio of new breadth to original breadth is 0.5 -/
theorem rectangle_breadth_ratio
  (L B : ℝ)  -- Original length and breadth
  (L' B' : ℝ) -- New length and breadth
  (h1 : L' = L / 2)  -- New length is half of original
  (h2 : L' * B' = (L * B) / 2)  -- New area is half of original
  : B' / B = 0.5 := by
  sorry


end rectangle_breadth_ratio_l2532_253291
