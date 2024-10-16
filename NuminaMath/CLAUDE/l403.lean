import Mathlib

namespace NUMINAMATH_CALUDE_oranges_per_box_l403_40341

/-- Given a fruit farm that packs 2650 oranges into 265 boxes,
    prove that each box contains 10 oranges. -/
theorem oranges_per_box :
  let total_oranges : ℕ := 2650
  let total_boxes : ℕ := 265
  total_oranges / total_boxes = 10 := by
sorry

end NUMINAMATH_CALUDE_oranges_per_box_l403_40341


namespace NUMINAMATH_CALUDE_inequality_solution_l403_40346

theorem inequality_solution (x : ℝ) : 
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (1 + 2 * x)) ↔ 
  x ∈ Set.Icc (-12/7) (-3/4) ∧ x ≠ -3/4 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l403_40346


namespace NUMINAMATH_CALUDE_find_b_l403_40399

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x + 4

-- Define the closed interval [2, 2b]
def interval (b : ℝ) : Set ℝ := Set.Icc 2 (2*b)

-- Theorem statement
theorem find_b : 
  ∃ (b : ℝ), b > 1 ∧ 
  (∀ x ∈ interval b, f x ∈ interval b) ∧
  (∀ y ∈ interval b, ∃ x ∈ interval b, f x = y) ∧
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l403_40399


namespace NUMINAMATH_CALUDE_only_event3_mutually_exclusive_l403_40329

-- Define the set of numbers
def NumberSet : Set Nat := {n | 1 ≤ n ∧ n ≤ 9}

-- Define the sample space
def SampleSpace : Set (Nat × Nat) :=
  {pair | pair.1 ∈ NumberSet ∧ pair.2 ∈ NumberSet ∧ pair.1 ≠ pair.2}

-- Define event ①
def Event1 (pair : Nat × Nat) : Prop :=
  (pair.1 % 2 = 0 ∧ pair.2 % 2 = 1) ∨ (pair.1 % 2 = 1 ∧ pair.2 % 2 = 0)

-- Define event ②
def Event2 (pair : Nat × Nat) : Prop :=
  (pair.1 % 2 = 1 ∨ pair.2 % 2 = 1) ∧ (pair.1 % 2 = 1 ∧ pair.2 % 2 = 1)

-- Define event ③
def Event3 (pair : Nat × Nat) : Prop :=
  (pair.1 % 2 = 1 ∨ pair.2 % 2 = 1) ∧ (pair.1 % 2 = 0 ∧ pair.2 % 2 = 0)

-- Define event ④
def Event4 (pair : Nat × Nat) : Prop :=
  (pair.1 % 2 = 1 ∨ pair.2 % 2 = 1) ∧ (pair.1 % 2 = 0 ∨ pair.2 % 2 = 0)

-- Theorem stating that only Event3 is mutually exclusive with other events
theorem only_event3_mutually_exclusive :
  ∀ (pair : Nat × Nat), pair ∈ SampleSpace →
    (¬(Event1 pair ∧ Event3 pair) ∧
     ¬(Event2 pair ∧ Event3 pair) ∧
     ¬(Event4 pair ∧ Event3 pair)) ∧
    ((Event1 pair ∧ Event2 pair) ∨
     (Event1 pair ∧ Event4 pair) ∨
     (Event2 pair ∧ Event4 pair)) :=
by sorry

end NUMINAMATH_CALUDE_only_event3_mutually_exclusive_l403_40329


namespace NUMINAMATH_CALUDE_tile_c_in_rectangle_three_l403_40370

/-- Represents the four sides of a tile -/
structure TileSides :=
  (top : Nat)
  (right : Nat)
  (bottom : Nat)
  (left : Nat)

/-- Represents a tile with its sides -/
inductive Tile
| A
| B
| C
| D

/-- Represents the four rectangles -/
inductive Rectangle
| One
| Two
| Three
| Four

/-- Function to get the sides of a tile -/
def getTileSides (t : Tile) : TileSides :=
  match t with
  | Tile.A => ⟨6, 1, 3, 2⟩
  | Tile.B => ⟨3, 6, 2, 0⟩
  | Tile.C => ⟨4, 0, 5, 6⟩
  | Tile.D => ⟨2, 5, 1, 4⟩

/-- Predicate to check if two tiles can be placed adjacent to each other -/
def canBePlacedAdjacent (t1 t2 : Tile) (side : Nat → Nat) : Prop :=
  side (getTileSides t1).right = side (getTileSides t2).left

/-- The main theorem stating that Tile C must be placed in Rectangle 3 -/
theorem tile_c_in_rectangle_three :
  ∃ (placement : Tile → Rectangle),
    placement Tile.C = Rectangle.Three ∧
    (∀ t1 t2 : Tile, t1 ≠ t2 → placement t1 ≠ placement t2) ∧
    (∀ t1 t2 : Tile, 
      (placement t1 = Rectangle.One ∧ placement t2 = Rectangle.Two) ∨
      (placement t1 = Rectangle.Two ∧ placement t2 = Rectangle.Three) ∨
      (placement t1 = Rectangle.Three ∧ placement t2 = Rectangle.Four) →
      canBePlacedAdjacent t1 t2 id) := by
  sorry

end NUMINAMATH_CALUDE_tile_c_in_rectangle_three_l403_40370


namespace NUMINAMATH_CALUDE_shadow_problem_l403_40369

theorem shadow_problem (cube_edge : ℝ) (shadow_area : ℝ) (y : ℝ) : 
  cube_edge = 2 →
  shadow_area = 200 →
  y > 0 →
  y = (Real.sqrt (shadow_area + cube_edge ^ 2)) →
  ⌊1000 * y⌋ = 14280 := by
sorry

end NUMINAMATH_CALUDE_shadow_problem_l403_40369


namespace NUMINAMATH_CALUDE_midpoint_complex_coordinates_l403_40354

theorem midpoint_complex_coordinates (A B C : ℂ) :
  A = 6 + 5*I ∧ B = -2 + 3*I ∧ C = (A + B) / 2 →
  C = 2 + 4*I :=
by sorry

end NUMINAMATH_CALUDE_midpoint_complex_coordinates_l403_40354


namespace NUMINAMATH_CALUDE_swan_population_after_ten_years_l403_40305

/-- The number of swans after a given number of years, given an initial population
    that doubles every 2 years -/
def swan_population (initial_population : ℕ) (years : ℕ) : ℕ :=
  initial_population * (2 ^ (years / 2))

/-- Theorem stating that given an initial population of 15 swans that doubles every 2 years,
    the population after 10 years will be 480 swans -/
theorem swan_population_after_ten_years :
  swan_population 15 10 = 480 := by
  sorry

end NUMINAMATH_CALUDE_swan_population_after_ten_years_l403_40305


namespace NUMINAMATH_CALUDE_polynomial_factorization_l403_40307

theorem polynomial_factorization (x : ℝ) : 
  x^9 - 6*x^6 + 12*x^3 - 8 = (x^3 - 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l403_40307


namespace NUMINAMATH_CALUDE_solve_system_l403_40321

/-- Given a system of equations, prove that y and z have specific values -/
theorem solve_system (x y z : ℚ) 
  (eq1 : (x + y) / (z - x) = 9/2)
  (eq2 : (y + z) / (y - x) = 5)
  (eq3 : x = 43/4) :
  y = 305/17 ∧ z = 1165/68 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l403_40321


namespace NUMINAMATH_CALUDE_measuring_rod_with_rope_l403_40313

theorem measuring_rod_with_rope (x y : ℝ) 
  (h1 : x - y = 5)
  (h2 : y - (1/2) * x = 5) : 
  x - y = 5 ∧ y - (1/2) * x = 5 := by
  sorry

end NUMINAMATH_CALUDE_measuring_rod_with_rope_l403_40313


namespace NUMINAMATH_CALUDE_dave_apps_left_l403_40377

/-- The number of files Dave has left on his phone -/
def files_left : ℕ := 4

/-- The number of apps Dave has left on his phone -/
def apps_left : ℕ := files_left + 17

/-- Theorem: Dave has 21 apps left on his phone -/
theorem dave_apps_left : apps_left = 21 := by
  sorry

end NUMINAMATH_CALUDE_dave_apps_left_l403_40377


namespace NUMINAMATH_CALUDE_cuboid_surface_area_l403_40333

theorem cuboid_surface_area (h : ℝ) (sum_edges : ℝ) (surface_area : ℝ) : 
  sum_edges = 100 ∧ 
  20 * h = sum_edges ∧ 
  surface_area = 2 * (2*h * 2*h + 2*h * h + 2*h * h) → 
  surface_area = 400 := by
sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_l403_40333


namespace NUMINAMATH_CALUDE_roof_dimension_difference_l403_40325

theorem roof_dimension_difference (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 4 * width →
  width * length = 676 →
  length - width = 39 := by
sorry

end NUMINAMATH_CALUDE_roof_dimension_difference_l403_40325


namespace NUMINAMATH_CALUDE_optimal_warehouse_location_l403_40364

/-- The optimal warehouse location problem -/
theorem optimal_warehouse_location 
  (y₁ : ℝ → ℝ) (y₂ : ℝ → ℝ) (k₁ k₂ : ℝ) (h₁ : ∀ x > 0, y₁ x = k₁ / x) 
  (h₂ : ∀ x > 0, y₂ x = k₂ * x) (h₃ : k₁ > 0) (h₄ : k₂ > 0)
  (h₅ : y₁ 10 = 4) (h₆ : y₂ 10 = 16) :
  ∃ x₀ > 0, ∀ x > 0, y₁ x + y₂ x ≥ y₁ x₀ + y₂ x₀ ∧ x₀ = 5 :=
sorry

end NUMINAMATH_CALUDE_optimal_warehouse_location_l403_40364


namespace NUMINAMATH_CALUDE_john_task_completion_l403_40331

-- Define the start time and end time of the first three tasks
def start_time : Nat := 9 * 60  -- 9:00 AM in minutes
def end_three_tasks : Nat := 12 * 60 + 15  -- 12:15 PM in minutes

-- Define the number of tasks
def num_tasks : Nat := 4

-- Define the theorem
theorem john_task_completion 
  (h1 : end_three_tasks - start_time = (num_tasks - 1) * ((end_three_tasks - start_time) / (num_tasks - 1)))
  (h2 : (end_three_tasks - start_time) % (num_tasks - 1) = 0) :
  end_three_tasks + ((end_three_tasks - start_time) / (num_tasks - 1)) = 13 * 60 + 20 := by
sorry


end NUMINAMATH_CALUDE_john_task_completion_l403_40331


namespace NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l403_40362

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else a^(x - 1)

-- State the theorem
theorem range_of_a_for_monotonic_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Ioo 2 4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l403_40362


namespace NUMINAMATH_CALUDE_lines_parallel_if_perpendicular_to_parallel_planes_l403_40319

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (lineParallel : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_if_perpendicular_to_parallel_planes 
  (α β : Plane) (a b : Line)
  (h_distinct_planes : α ≠ β)
  (h_distinct_lines : a ≠ b)
  (h_a_perp_α : perpendicular a α)
  (h_b_perp_β : perpendicular b β)
  (h_α_parallel_β : parallel α β) :
  lineParallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_if_perpendicular_to_parallel_planes_l403_40319


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l403_40396

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 < 0) → a ∈ Set.Iio (-2) ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l403_40396


namespace NUMINAMATH_CALUDE_expression_evaluation_l403_40358

theorem expression_evaluation (x y : ℝ) (h : x * y ≠ 0) :
  (x^4 + 1) / x^2 * (y^4 + 1) / y^2 - (x^4 - 1) / y^2 * (y^4 - 1) / x^2 = 2 * x^2 / y^2 + 2 * y^2 / x^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l403_40358


namespace NUMINAMATH_CALUDE_fifty_second_digit_of_1_17_l403_40349

-- Define the decimal representation of 1/17
def decimal_rep_1_17 : ℚ := 1 / 17

-- Define the length of the repeating sequence
def repeat_length : ℕ := 16

-- Define the position we're interested in
def target_position : ℕ := 52

-- Define the function to get the nth digit after the decimal point
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem fifty_second_digit_of_1_17 : 
  nth_digit target_position = 8 := by sorry

end NUMINAMATH_CALUDE_fifty_second_digit_of_1_17_l403_40349


namespace NUMINAMATH_CALUDE_circle_area_20cm_diameter_l403_40344

/-- The area of a circle with diameter 20 cm is 314 square cm, given π = 3.14 -/
theorem circle_area_20cm_diameter (π : ℝ) (h : π = 3.14) :
  let d : ℝ := 20
  let r : ℝ := d / 2
  π * r^2 = 314 := by sorry

end NUMINAMATH_CALUDE_circle_area_20cm_diameter_l403_40344


namespace NUMINAMATH_CALUDE_log_product_reciprocal_l403_40392

theorem log_product_reciprocal (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) :
  Real.log a / Real.log b * (Real.log b / Real.log a) = 1 :=
by sorry

end NUMINAMATH_CALUDE_log_product_reciprocal_l403_40392


namespace NUMINAMATH_CALUDE_minimum_value_implies_k_l403_40365

/-- Given that k is a positive constant and the minimum value of the function
    y = x^2 + k/x (where x > 0) is 3, prove that k = 2. -/
theorem minimum_value_implies_k (k : ℝ) (h1 : k > 0) :
  (∀ x > 0, x^2 + k/x ≥ 3) ∧ (∃ x > 0, x^2 + k/x = 3) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_implies_k_l403_40365


namespace NUMINAMATH_CALUDE_abacus_problem_solution_l403_40393

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ // n ≥ 100 ∧ n < 1000 }

/-- Check if a three-digit number has distinct digits -/
def has_distinct_digits (n : ThreeDigitNumber) : Prop :=
  let digits := [n.val / 100, (n.val / 10) % 10, n.val % 10]
  digits.Nodup

/-- The abacus problem solution -/
theorem abacus_problem_solution :
  ∃! (top bottom : ThreeDigitNumber),
    has_distinct_digits top ∧
    ∃ (k : ℕ), k > 1 ∧ top.val = k * bottom.val ∧
    top.val + bottom.val = 1110 ∧
    top.val = 925 := by
  sorry

end NUMINAMATH_CALUDE_abacus_problem_solution_l403_40393


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l403_40316

/-- Proves that a boat's speed in still water is 20 km/hr given specific conditions -/
theorem boat_speed_in_still_water :
  let current_speed : ℝ := 3
  let downstream_distance : ℝ := 9.2
  let downstream_time : ℝ := 24 / 60
  let downstream_speed : ℝ → ℝ := λ v => v + current_speed
  ∃ (v : ℝ), downstream_speed v * downstream_time = downstream_distance ∧ v = 20 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l403_40316


namespace NUMINAMATH_CALUDE_shelly_thread_calculation_l403_40308

/-- The number of friends Shelly made in classes -/
def class_friends : ℕ := 10

/-- The number of friends Shelly made from after-school clubs -/
def club_friends : ℕ := 2 * class_friends

/-- The amount of thread needed for each keychain for class friends (in inches) -/
def class_thread_per_keychain : ℕ := 16

/-- The amount of thread needed for each keychain for after-school club friends (in inches) -/
def club_thread_per_keychain : ℕ := 20

/-- The total amount of thread Shelly needs (in inches) -/
def total_thread_needed : ℕ := class_friends * class_thread_per_keychain + club_friends * club_thread_per_keychain

theorem shelly_thread_calculation :
  total_thread_needed = 560 := by
  sorry

end NUMINAMATH_CALUDE_shelly_thread_calculation_l403_40308


namespace NUMINAMATH_CALUDE_sum_of_polygon_sides_l403_40343

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a quadrilateral -/
def quadrilateral_sides : ℕ := 4

/-- Theorem: The sum of the sides of a hexagon, triangle, and quadrilateral is 13 -/
theorem sum_of_polygon_sides : 
  hexagon_sides + triangle_sides + quadrilateral_sides = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polygon_sides_l403_40343


namespace NUMINAMATH_CALUDE_line_intersects_circle_l403_40342

/-- Given a point (x₀, y₀) outside the circle x² + y² = r², 
    prove that the line x₀x + y₀y = r² intersects the circle. -/
theorem line_intersects_circle (x₀ y₀ r : ℝ) (h : x₀^2 + y₀^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ x₀*x + y₀*y = r^2 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l403_40342


namespace NUMINAMATH_CALUDE_square_root_problem_l403_40378

theorem square_root_problem (x y z : ℝ) 
  (h1 : Real.sqrt (2 * x + 1) = 0)
  (h2 : Real.sqrt y = 4)
  (h3 : z^3 = -27) :
  {r : ℝ | r^2 = 2*x + y + z} = {2 * Real.sqrt 3, -2 * Real.sqrt 3} := by
sorry

end NUMINAMATH_CALUDE_square_root_problem_l403_40378


namespace NUMINAMATH_CALUDE_inequality_and_equality_cases_l403_40380

theorem inequality_and_equality_cases (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 ∧
  ((a - b) * (b - c) * (a - c) = 2 ↔ 
    ((a = 2 ∧ b = 1 ∧ c = 0) ∨ 
     (a = 1 ∧ b = 0 ∧ c = 2) ∨ 
     (a = 0 ∧ b = 2 ∧ c = 1))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_cases_l403_40380


namespace NUMINAMATH_CALUDE_quadratic_function_range_l403_40353

/-- A quadratic function passing through (1,0) and (0,1) with vertex in second quadrant -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  point_zero_one : 1 = c
  point_one_zero : 0 = a + b + c
  vertex_second_quadrant : b < 0 ∧ a < 0

/-- The range of a - b + c for the given quadratic function -/
theorem quadratic_function_range (f : QuadraticFunction) : 
  0 < f.a - f.b + f.c ∧ f.a - f.b + f.c < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l403_40353


namespace NUMINAMATH_CALUDE_granger_grocery_bill_l403_40371

/-- Calculates the total cost of a grocery shopping trip -/
def total_cost (spam_price peanut_butter_price bread_price : ℕ) 
               (spam_quantity peanut_butter_quantity bread_quantity : ℕ) : ℕ :=
  spam_price * spam_quantity + 
  peanut_butter_price * peanut_butter_quantity + 
  bread_price * bread_quantity

/-- Proves that Granger's grocery bill is $59 -/
theorem granger_grocery_bill : 
  total_cost 3 5 2 12 3 4 = 59 := by
  sorry

end NUMINAMATH_CALUDE_granger_grocery_bill_l403_40371


namespace NUMINAMATH_CALUDE_min_value_of_2x_plus_y_l403_40389

theorem min_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 3) :
  2*x + y ≥ 8/3 ∧ (2*x + y = 8/3 ↔ x = 2/3 ∧ y = 4/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_2x_plus_y_l403_40389


namespace NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_two_l403_40302

theorem abs_ratio_eq_sqrt_two (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 6*a*b) :
  |((a + b) / (a - b))| = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_ratio_eq_sqrt_two_l403_40302


namespace NUMINAMATH_CALUDE_sqrt_of_36_l403_40330

theorem sqrt_of_36 : Real.sqrt 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_36_l403_40330


namespace NUMINAMATH_CALUDE_ned_shirts_problem_l403_40337

theorem ned_shirts_problem (total_shirts : ℕ) (short_sleeve : ℕ) (washed_shirts : ℕ)
  (h1 : total_shirts = 30)
  (h2 : short_sleeve = 9)
  (h3 : washed_shirts = 29) :
  total_shirts - short_sleeve - (total_shirts - washed_shirts) = 19 := by
  sorry

end NUMINAMATH_CALUDE_ned_shirts_problem_l403_40337


namespace NUMINAMATH_CALUDE_square_plus_double_equals_one_implies_double_square_plus_quadruple_plus_one_equals_three_l403_40314

theorem square_plus_double_equals_one_implies_double_square_plus_quadruple_plus_one_equals_three
  (a : ℝ) (h : a^2 + 2*a = 1) : 2*a^2 + 4*a + 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_double_equals_one_implies_double_square_plus_quadruple_plus_one_equals_three_l403_40314


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l403_40391

theorem square_rectangle_area_relation :
  ∃ (x₁ x₂ : ℝ),
    (x₁ - 2) * (x₁ + 5) = 3 * (x₁ - 3)^2 ∧
    (x₂ - 2) * (x₂ + 5) = 3 * (x₂ - 3)^2 ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l403_40391


namespace NUMINAMATH_CALUDE_exam_results_l403_40384

theorem exam_results (failed_hindi : ℝ) (failed_english : ℝ) (failed_both : ℝ)
  (h1 : failed_hindi = 20)
  (h2 : failed_english = 70)
  (h3 : failed_both = 10) :
  100 - (failed_hindi + failed_english - failed_both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_exam_results_l403_40384


namespace NUMINAMATH_CALUDE_house_to_school_distance_house_to_school_distance_is_60_l403_40381

/-- The distance between a house and a school, given travel times at different speeds -/
theorem house_to_school_distance : ℝ :=
  let speed_slow : ℝ := 10  -- km/hr
  let speed_fast : ℝ := 20  -- km/hr
  let time_late : ℝ := 2    -- hours
  let time_early : ℝ := 1   -- hours
  let distance : ℝ := 60    -- km

  have h1 : distance = speed_slow * (distance / speed_slow + time_late) := by sorry
  have h2 : distance = speed_fast * (distance / speed_fast - time_early) := by sorry

  distance

/-- The proof that the distance is indeed 60 km -/
theorem house_to_school_distance_is_60 : house_to_school_distance = 60 := by sorry

end NUMINAMATH_CALUDE_house_to_school_distance_house_to_school_distance_is_60_l403_40381


namespace NUMINAMATH_CALUDE_probability_two_red_one_black_l403_40366

def num_red_balls : ℕ := 4
def num_black_balls : ℕ := 3
def total_balls : ℕ := num_red_balls + num_black_balls
def num_draws : ℕ := 3

def prob_red : ℚ := num_red_balls / total_balls
def prob_black : ℚ := num_black_balls / total_balls

def prob_two_red_one_black : ℚ := 3 * (prob_red * prob_red * prob_black)

theorem probability_two_red_one_black : 
  prob_two_red_one_black = 144 / 343 := by sorry

end NUMINAMATH_CALUDE_probability_two_red_one_black_l403_40366


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l403_40324

/-- The polynomial function we're considering -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

/-- Theorem stating that 1, -1, and 3 are the only roots of the polynomial -/
theorem roots_of_polynomial :
  (∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -1 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l403_40324


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l403_40300

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - a + 1 = 0) → 
  (b^3 - b + 1 = 0) → 
  (c^3 - c + 1 = 0) → 
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = -2) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l403_40300


namespace NUMINAMATH_CALUDE_edge_enlargement_equals_graph_scale_l403_40374

/-- A graph is represented by a set of edges, where each edge has a length. -/
structure Graph where
  edges : Set (ℝ)

/-- Enlarging a graph by multiplying each edge length by a factor. -/
def enlarge (g : Graph) (factor : ℝ) : Graph :=
  { edges := g.edges.image (· * factor) }

/-- The scale factor of a transformation that multiplies each edge by 4. -/
def scale_factor : ℝ := 4

theorem edge_enlargement_equals_graph_scale (g : Graph) :
  enlarge g scale_factor = enlarge g scale_factor :=
by
  sorry

end NUMINAMATH_CALUDE_edge_enlargement_equals_graph_scale_l403_40374


namespace NUMINAMATH_CALUDE_parabola_ellipse_focus_l403_40340

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

-- Define the parabola
def parabola (x y p : ℝ) : Prop := y^2 = 2 * p * x

-- Define the right focus of the ellipse
def right_focus (x y : ℝ) : Prop := ellipse x y ∧ x > 0 ∧ y = 0

-- Define the focus of the parabola
def parabola_focus (x y p : ℝ) : Prop := x = p / 2 ∧ y = 0

-- The main theorem
theorem parabola_ellipse_focus (p : ℝ) :
  p > 0 →
  (∃ x y, right_focus x y ∧ parabola_focus x y p) →
  p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_ellipse_focus_l403_40340


namespace NUMINAMATH_CALUDE_complement_A_union_B_l403_40301

def U : Set ℕ := {n | n > 0 ∧ n < 9}
def A : Set ℕ := {n ∈ U | n % 2 = 1}
def B : Set ℕ := {n ∈ U | n % 3 = 0}

theorem complement_A_union_B : (U \ (A ∪ B)) = {2, 4, 8} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_l403_40301


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l403_40332

/-- Represents the distance walked in a single direction -/
structure DirectionalWalk where
  blocks : ℕ
  direction : String

/-- Calculates the total distance walked given a list of directional walks and the length of each block in miles -/
def totalDistance (walks : List DirectionalWalk) (blockLength : ℚ) : ℚ :=
  (walks.map (·.blocks)).sum * blockLength

theorem arthur_walk_distance :
  let eastWalk : DirectionalWalk := ⟨8, "east"⟩
  let northWalk : DirectionalWalk := ⟨10, "north"⟩
  let westWalk : DirectionalWalk := ⟨3, "west"⟩
  let walks : List DirectionalWalk := [eastWalk, northWalk, westWalk]
  let blockLength : ℚ := 1/3
  totalDistance walks blockLength = 7 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l403_40332


namespace NUMINAMATH_CALUDE_drummer_tosses_six_sets_l403_40368

/-- Calculates the number of drum stick sets tossed to the audience after each show -/
def drumSticksTossedPerShow (setsPerShow : ℕ) (totalNights : ℕ) (totalSetsUsed : ℕ) : ℕ :=
  ((totalSetsUsed - setsPerShow * totalNights) / totalNights)

/-- Theorem: Given the conditions, the drummer tosses 6 sets of drum sticks after each show -/
theorem drummer_tosses_six_sets :
  drumSticksTossedPerShow 5 30 330 = 6 := by
  sorry

end NUMINAMATH_CALUDE_drummer_tosses_six_sets_l403_40368


namespace NUMINAMATH_CALUDE_equal_piles_coin_count_l403_40388

theorem equal_piles_coin_count (total_coins : ℕ) (num_quarter_piles : ℕ) (num_dime_piles : ℕ) :
  total_coins = 42 →
  num_quarter_piles = 3 →
  num_dime_piles = 3 →
  ∃ (coins_per_pile : ℕ),
    total_coins = coins_per_pile * (num_quarter_piles + num_dime_piles) ∧
    coins_per_pile = 7 :=
by sorry

end NUMINAMATH_CALUDE_equal_piles_coin_count_l403_40388


namespace NUMINAMATH_CALUDE_line_condition_l403_40386

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Checks if two points are on the same side of a line x - y + 2 = 0 -/
def sameSideOfLine (p1 p2 : Point) : Prop :=
  (p1.x - p1.y + 2) * (p2.x - p2.y + 2) > 0

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- A point on the line y = kx + b -/
def pointOnLine (l : Line) (x : ℝ) : Point :=
  ⟨x, l.k * x + l.b⟩

theorem line_condition (l : Line) : 
  (∀ x : ℝ, sameSideOfLine (pointOnLine l x) origin) → 
  l.k = 1 ∧ l.b < 2 := by
  sorry

end NUMINAMATH_CALUDE_line_condition_l403_40386


namespace NUMINAMATH_CALUDE_f_and_g_properties_l403_40318

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1| - |1 - x|
def g (a b x : ℝ) : ℝ := |x + a^2| + |x - b^2|

-- State the theorem
theorem f_and_g_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x, x ∈ {x : ℝ | f x ≥ 1} ↔ x ∈ Set.Ici (1/2)) ∧
  (∀ x, f x ≤ g a b x) := by
  sorry

end NUMINAMATH_CALUDE_f_and_g_properties_l403_40318


namespace NUMINAMATH_CALUDE_tan_theta_value_l403_40357

theorem tan_theta_value (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π / 2))
  (h2 : 12 / Real.sin θ + 12 / Real.cos θ = 35) :
  Real.tan θ = 3/4 ∨ Real.tan θ = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l403_40357


namespace NUMINAMATH_CALUDE_age_problem_l403_40360

theorem age_problem (parent_age son_age : ℕ) : 
  parent_age = 3 * son_age ∧ 
  parent_age + 5 = (5/2) * (son_age + 5) →
  parent_age = 45 ∧ son_age = 15 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l403_40360


namespace NUMINAMATH_CALUDE_grocery_store_inventory_l403_40317

theorem grocery_store_inventory (apples regular_soda diet_soda : ℕ) 
  (h1 : apples = 36)
  (h2 : regular_soda = 80)
  (h3 : diet_soda = 54) :
  regular_soda + diet_soda - apples = 98 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_inventory_l403_40317


namespace NUMINAMATH_CALUDE_cubic_equations_common_root_l403_40397

/-- Given real numbers a, b, c, if every pair of equations from 
    x³ - ax² + b = 0, x³ - bx² + c = 0, x³ - cx² + a = 0 has a common root, 
    then a = b = c. -/
theorem cubic_equations_common_root (a b c : ℝ) 
  (h1 : ∃ x : ℝ, x^3 - a*x^2 + b = 0 ∧ x^3 - b*x^2 + c = 0)
  (h2 : ∃ x : ℝ, x^3 - b*x^2 + c = 0 ∧ x^3 - c*x^2 + a = 0)
  (h3 : ∃ x : ℝ, x^3 - c*x^2 + a = 0 ∧ x^3 - a*x^2 + b = 0) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_cubic_equations_common_root_l403_40397


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l403_40385

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence, if a₁ + a₉ = 8, then a₂ + a₈ = 8 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 + a 9 = 8 → a 2 + a 8 = 8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l403_40385


namespace NUMINAMATH_CALUDE_supplementary_angle_of_60_degrees_l403_40345

theorem supplementary_angle_of_60_degrees (α : ℝ) : 
  α = 60 → 180 - α = 120 := by
  sorry

end NUMINAMATH_CALUDE_supplementary_angle_of_60_degrees_l403_40345


namespace NUMINAMATH_CALUDE_sin_18_cos_12_plus_cos_18_sin_12_l403_40382

theorem sin_18_cos_12_plus_cos_18_sin_12 :
  Real.sin (18 * π / 180) * Real.cos (12 * π / 180) + 
  Real.cos (18 * π / 180) * Real.sin (12 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_18_cos_12_plus_cos_18_sin_12_l403_40382


namespace NUMINAMATH_CALUDE_waitress_average_orders_per_hour_l403_40310

theorem waitress_average_orders_per_hour
  (hourly_wage : ℝ)
  (tip_rate : ℝ)
  (num_shifts : ℕ)
  (hours_per_shift : ℕ)
  (total_earnings : ℝ)
  (h1 : hourly_wage = 4)
  (h2 : tip_rate = 0.15)
  (h3 : num_shifts = 3)
  (h4 : hours_per_shift = 8)
  (h5 : total_earnings = 240) :
  let total_hours : ℕ := num_shifts * hours_per_shift
  let wage_earnings : ℝ := hourly_wage * total_hours
  let tip_earnings : ℝ := total_earnings - wage_earnings
  let total_orders : ℝ := tip_earnings / tip_rate
  let avg_orders_per_hour : ℝ := total_orders / total_hours
  avg_orders_per_hour = 40 := by
sorry

end NUMINAMATH_CALUDE_waitress_average_orders_per_hour_l403_40310


namespace NUMINAMATH_CALUDE_min_value_inequality_l403_40312

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  x / y + 1 / x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l403_40312


namespace NUMINAMATH_CALUDE_multiplication_factor_exists_l403_40327

theorem multiplication_factor_exists (x : ℝ) (hx : x = 2.6666666666666665) :
  ∃ y : ℝ, Real.sqrt ((x * y) / 3) = x ∧ abs (y - 8) < 0.0000001 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_factor_exists_l403_40327


namespace NUMINAMATH_CALUDE_z₂_in_fourth_quadrant_z₂_equals_z₁_times_ni_l403_40356

-- Define complex numbers z₁ and z₂
def z₁ (m : ℝ) : ℂ := m + Complex.I
def z₂ (m : ℝ) : ℂ := m + (m - 2) * Complex.I

-- Theorem 1: If z₂ is in the fourth quadrant, then 0 < m < 2
theorem z₂_in_fourth_quadrant (m : ℝ) :
  (z₂ m).re > 0 ∧ (z₂ m).im < 0 → 0 < m ∧ m < 2 := by sorry

-- Theorem 2: If z₂ = z₁ · ni, then (m = 1 and n = -1) or (m = -2 and n = 2)
theorem z₂_equals_z₁_times_ni (m n : ℝ) :
  z₂ m = z₁ m * (n * Complex.I) →
  (m = 1 ∧ n = -1) ∨ (m = -2 ∧ n = 2) := by sorry

end NUMINAMATH_CALUDE_z₂_in_fourth_quadrant_z₂_equals_z₁_times_ni_l403_40356


namespace NUMINAMATH_CALUDE_fencing_cost_proof_l403_40359

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length breadth cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_meter

/-- Proves that the total cost of fencing the given rectangular plot is 5300 -/
theorem fencing_cost_proof (length breadth cost_per_meter : ℝ) 
  (h1 : length = 64)
  (h2 : breadth = length - 28)
  (h3 : cost_per_meter = 26.5) :
  total_fencing_cost length breadth cost_per_meter = 5300 := by
  sorry

#eval total_fencing_cost 64 36 26.5

end NUMINAMATH_CALUDE_fencing_cost_proof_l403_40359


namespace NUMINAMATH_CALUDE_twenty_pancakes_in_24_minutes_l403_40320

/-- Represents the pancake production and consumption rates of a family -/
structure PancakeFamily where
  dad_rate : ℚ  -- Dad's pancake production rate per hour
  mom_rate : ℚ  -- Mom's pancake production rate per hour
  petya_rate : ℚ  -- Petya's pancake consumption rate per 15 minutes
  vasya_multiplier : ℚ  -- Vasya's consumption rate multiplier relative to Petya

/-- Calculates the minimum time (in minutes) required for at least 20 pancakes to remain uneaten -/
def min_time_for_20_pancakes (family : PancakeFamily) : ℚ :=
  sorry

/-- The main theorem stating that 24 minutes is the minimum time for 20 pancakes to remain uneaten -/
theorem twenty_pancakes_in_24_minutes (family : PancakeFamily) 
  (h1 : family.dad_rate = 70)
  (h2 : family.mom_rate = 100)
  (h3 : family.petya_rate = 10)
  (h4 : family.vasya_multiplier = 2) :
  min_time_for_20_pancakes family = 24 := by
  sorry

end NUMINAMATH_CALUDE_twenty_pancakes_in_24_minutes_l403_40320


namespace NUMINAMATH_CALUDE_trash_in_classrooms_l403_40303

theorem trash_in_classrooms (total_trash : ℕ) (outside_trash : ℕ) 
  (h1 : total_trash = 1576) 
  (h2 : outside_trash = 1232) : 
  total_trash - outside_trash = 344 := by
  sorry

end NUMINAMATH_CALUDE_trash_in_classrooms_l403_40303


namespace NUMINAMATH_CALUDE_rectangle_length_l403_40309

/-- Given a rectangle with area 28 square centimeters and width 4 centimeters, its length is 7 centimeters. -/
theorem rectangle_length (area width : ℝ) (h_area : area = 28) (h_width : width = 4) :
  area / width = 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l403_40309


namespace NUMINAMATH_CALUDE_difference_of_sums_l403_40394

/-- Sum of first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- Sum of first n even numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The number of odd numbers from 1 to 2011 -/
def oddCount : ℕ := 1006

/-- The number of even numbers from 2 to 2010 -/
def evenCount : ℕ := 1005

theorem difference_of_sums : sumOddNumbers oddCount - sumEvenNumbers evenCount = 1006 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_sums_l403_40394


namespace NUMINAMATH_CALUDE_lasagna_profit_proof_l403_40322

/-- Calculates the profit after expenses for selling lasagna pans -/
def profit_after_expenses (num_pans : ℕ) (cost_per_pan : ℚ) (price_per_pan : ℚ) : ℚ :=
  num_pans * (price_per_pan - cost_per_pan)

/-- Proves that the profit after expenses for selling 20 pans of lasagna is $300.00 -/
theorem lasagna_profit_proof :
  profit_after_expenses 20 10 25 = 300 := by
  sorry

end NUMINAMATH_CALUDE_lasagna_profit_proof_l403_40322


namespace NUMINAMATH_CALUDE_gcd_abcd_plus_dcba_eq_one_l403_40361

def abcd_plus_dcba (a : ℤ) : ℤ :=
  let b := a^2 + 1
  let c := a^2 + 2
  let d := a^2 + 3
  2111 * a^2 + 1001 * a + 3333

theorem gcd_abcd_plus_dcba_eq_one :
  ∃ (a : ℤ), ∀ (x : ℤ), x ∣ abcd_plus_dcba a → x = 1 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_gcd_abcd_plus_dcba_eq_one_l403_40361


namespace NUMINAMATH_CALUDE_factorization_proof_l403_40339

theorem factorization_proof (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 3) * (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l403_40339


namespace NUMINAMATH_CALUDE_divisibility_theorem_l403_40398

theorem divisibility_theorem (a b : ℕ+) (h : (7^2009 : ℕ) ∣ (a^2 + b^2)) :
  (7^2010 : ℕ) ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l403_40398


namespace NUMINAMATH_CALUDE_sad_children_count_l403_40383

theorem sad_children_count (total : ℕ) (happy : ℕ) (neither : ℕ) :
  total = 60 →
  happy = 30 →
  neither = 20 →
  total - (happy + neither) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_sad_children_count_l403_40383


namespace NUMINAMATH_CALUDE_triangle_side_sum_l403_40306

theorem triangle_side_sum (a b c : ℝ) (A B C : ℝ) :
  B = π / 6 →
  b = 1 →
  a * c = 2 * Real.sqrt 3 →
  a + c = 2 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l403_40306


namespace NUMINAMATH_CALUDE_dog_food_consumption_l403_40372

/-- The amount of dog food eaten by two dogs per day, given that each dog eats 0.125 scoop per day. -/
theorem dog_food_consumption (dog1_consumption dog2_consumption : ℝ) 
  (h1 : dog1_consumption = 0.125)
  (h2 : dog2_consumption = 0.125) : 
  dog1_consumption + dog2_consumption = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_dog_food_consumption_l403_40372


namespace NUMINAMATH_CALUDE_range_of_x_l403_40334

theorem range_of_x (x : ℝ) : 4 * x - 12 ≥ 0 → x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l403_40334


namespace NUMINAMATH_CALUDE_cube_has_eight_vertices_l403_40387

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the internal structure of a cube for this problem

/-- The number of vertices in a cube -/
def num_vertices (c : Cube) : ℕ := 8

/-- Theorem: A cube has 8 vertices -/
theorem cube_has_eight_vertices (c : Cube) : num_vertices c = 8 := by
  sorry

end NUMINAMATH_CALUDE_cube_has_eight_vertices_l403_40387


namespace NUMINAMATH_CALUDE_distance_between_cities_l403_40375

/-- Represents the travel scenario of two cars between two cities -/
structure TravelScenario where
  v : ℝ  -- Speed of car A in km/min
  x : ℝ  -- Total travel time for car A in minutes
  d : ℝ  -- Distance between the two cities in km

/-- Conditions of the travel scenario -/
def travel_conditions (s : TravelScenario) : Prop :=
  -- Both cars travel the same distance in first 5 minutes
  -- Car B's speed reduces to 2/5 of original after 5 minutes
  -- Car B arrives 15 minutes after car A
  (5 * s.v - 25) / 2 = s.x - 5 + 15 ∧
  -- If failure occurred 4 km farther, B would arrive 10 minutes after A
  25 - 10 / s.v = 20 - 4 / s.v ∧
  -- Total distance is speed multiplied by time
  s.d = s.v * s.x

/-- The main theorem stating the distance between the cities -/
theorem distance_between_cities :
  ∀ s : TravelScenario, travel_conditions s → s.d = 18 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_cities_l403_40375


namespace NUMINAMATH_CALUDE_parabola_point_distance_l403_40323

/-- Parabola type representing y = -ax²/6 + ax + c -/
structure Parabola where
  a : ℝ
  c : ℝ
  h_a : a < 0

/-- Point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y = -p.a * x^2 / 6 + p.a * x + p.c

/-- Theorem statement -/
theorem parabola_point_distance (p : Parabola) 
  (A B C : ParabolaPoint p) 
  (h_B_vertex : B.y = 3 * p.a / 2 + p.c) 
  (h_y_order : A.y > C.y ∧ C.y > B.y) :
  |A.x - B.x| ≥ |B.x - C.x| := by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l403_40323


namespace NUMINAMATH_CALUDE_valentine_card_cost_l403_40379

def total_students : ℕ := 30
def valentine_percentage : ℚ := 60 / 100
def initial_money : ℚ := 40
def spending_percentage : ℚ := 90 / 100

theorem valentine_card_cost :
  let students_receiving := total_students * valentine_percentage
  let money_spent := initial_money * spending_percentage
  let cost_per_card := money_spent / students_receiving
  cost_per_card = 2 := by sorry

end NUMINAMATH_CALUDE_valentine_card_cost_l403_40379


namespace NUMINAMATH_CALUDE_hockey_league_teams_l403_40338

/-- The number of teams in a hockey league --/
def num_teams : ℕ := 18

/-- The number of times each team faces every other team --/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season --/
def total_games : ℕ := 1530

/-- Theorem: Given the conditions, the number of teams in the league is 18 --/
theorem hockey_league_teams :
  (num_teams * (num_teams - 1) * games_per_pair) / 2 = total_games :=
sorry

end NUMINAMATH_CALUDE_hockey_league_teams_l403_40338


namespace NUMINAMATH_CALUDE_problem_solution_l403_40373

def A : Set ℤ := {-2, 3, 4, 6}
def B (a : ℤ) : Set ℤ := {3, a, a^2}

theorem problem_solution (a : ℤ) : 
  (B a ⊆ A → a = 2) ∧ 
  (A ∩ B a = {3, 4} → a = 2 ∨ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l403_40373


namespace NUMINAMATH_CALUDE_binomial_plus_ten_l403_40326

theorem binomial_plus_ten : (Nat.choose 20 4) + 10 = 4855 := by
  sorry

end NUMINAMATH_CALUDE_binomial_plus_ten_l403_40326


namespace NUMINAMATH_CALUDE_ginger_water_bottle_capacity_l403_40367

/-- Proves that Ginger's water bottle holds 2 cups given the problem conditions -/
theorem ginger_water_bottle_capacity 
  (hours_worked : ℕ) 
  (bottles_for_plants : ℕ) 
  (total_cups_used : ℕ) 
  (h1 : hours_worked = 8)
  (h2 : bottles_for_plants = 5)
  (h3 : total_cups_used = 26) :
  (total_cups_used : ℚ) / (hours_worked + bottles_for_plants : ℚ) = 2 := by
  sorry

#check ginger_water_bottle_capacity

end NUMINAMATH_CALUDE_ginger_water_bottle_capacity_l403_40367


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l403_40395

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geometric : geometric_sequence a q)
  (h_product : a 1 * a 2 * a 3 = 27)
  (h_sum : a 2 + a 4 = 30) :
  q = 3 ∨ q = -3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l403_40395


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l403_40311

/-- Given a quadratic expression 3x^2 + 9x + 20, prove that when expressed in the form a(x - h)^2 + k, the value of h is -3/2. -/
theorem quadratic_completion_of_square (x : ℝ) :
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l403_40311


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l403_40390

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, |x| < 1 → x^2 + x - 6 < 0) ∧
  (∃ x : ℝ, x^2 + x - 6 < 0 ∧ ¬(|x| < 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l403_40390


namespace NUMINAMATH_CALUDE_range_of_m_l403_40376

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = x - 2) →
  (∀ x, g x = x^2 - 2*m*x + 4) →
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc 4 5, g x₁ = f x₂) →
  m ∈ Set.Icc (5/4) (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l403_40376


namespace NUMINAMATH_CALUDE_fermat_number_large_prime_factor_l403_40355

/-- Fermat number -/
def F (n : ℕ) : ℕ := 2^(2^n) + 1

/-- Theorem: For n ≥ 3, F_n has a prime factor greater than 2^(n+2)(n+1) -/
theorem fermat_number_large_prime_factor (n : ℕ) (h : n ≥ 3) :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ F n ∧ p > 2^(n+2) * (n+1) := by
  sorry

end NUMINAMATH_CALUDE_fermat_number_large_prime_factor_l403_40355


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l403_40335

/-- The ratio of the area of the inscribed circle to the area of a right triangle -/
theorem inscribed_circle_area_ratio (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let a := (3 / 5) * h
  let b := (4 / 5) * h
  let triangle_area := (1 / 2) * a * b
  let circle_area := π * r^2
  circle_area / triangle_area = (5 * π * r) / (12 * h) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l403_40335


namespace NUMINAMATH_CALUDE_distinct_collections_l403_40315

/-- Represents the letter counts in CALCULATOR --/
structure LetterCounts where
  a : Nat
  c : Nat
  l : Nat
  other_vowels : Nat
  other_consonants : Nat

/-- Represents a selection of letters --/
structure Selection where
  a : Nat
  c : Nat
  l : Nat
  other_vowels : Nat
  other_consonants : Nat

/-- Checks if a selection is valid --/
def is_valid_selection (s : Selection) : Prop :=
  s.a + s.other_vowels = 3 ∧ 
  s.c + s.l + s.other_consonants = 6

/-- Counts distinct vowel selections --/
def count_vowel_selections (total : LetterCounts) : Nat :=
  3 -- This is a simplification based on the problem's specifics

/-- Counts distinct consonant selections --/
noncomputable def count_consonant_selections (total : LetterCounts) : Nat :=
  sorry -- This would be calculated based on the combinations in the solution

/-- The main theorem --/
theorem distinct_collections (total : LetterCounts) 
  (h1 : total.a = 2)
  (h2 : total.c = 2)
  (h3 : total.l = 2)
  (h4 : total.other_vowels = 2)
  (h5 : total.other_consonants = 2) :
  (count_vowel_selections total) * (count_consonant_selections total) = 
  3 * (count_consonant_selections total) := by
  sorry

#check distinct_collections

end NUMINAMATH_CALUDE_distinct_collections_l403_40315


namespace NUMINAMATH_CALUDE_units_digit_G_1000_l403_40304

/-- The sequence G_n is defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_G_1000 :
  units_digit (3^(3^1000)) = 1 →
  units_digit (G 1000) = 2 :=
sorry

end NUMINAMATH_CALUDE_units_digit_G_1000_l403_40304


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l403_40363

theorem cube_root_equation_solution :
  ∃ (x y z : ℕ+),
    (4 * ((7 : ℝ)^(1/3) - (6 : ℝ)^(1/3))^(1/2) = x^(1/3) + y^(1/3) - z^(1/3)) ∧
    (x + y + z = 51) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l403_40363


namespace NUMINAMATH_CALUDE_complex_modulus_l403_40328

theorem complex_modulus (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l403_40328


namespace NUMINAMATH_CALUDE_equation_describes_two_lines_l403_40352

/-- The set of points satisfying the equation (x-y)^2 = x^2 + y^2 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - p.2)^2 = p.1^2 + p.2^2}

/-- The union of x-axis and y-axis -/
def TwoLines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0}

theorem equation_describes_two_lines : S = TwoLines := by
  sorry

end NUMINAMATH_CALUDE_equation_describes_two_lines_l403_40352


namespace NUMINAMATH_CALUDE_evaluate_expression_l403_40351

theorem evaluate_expression : -(16 / 2 * 8 - 72 + 4^2) = -8 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l403_40351


namespace NUMINAMATH_CALUDE_solution_in_quadrant_I_l403_40350

/-- A point (x, y) lies in Quadrant I if both x and y are positive -/
def in_quadrant_I (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The system of equations -/
def system_equations (k x y : ℝ) : Prop :=
  2 * x - y = 5 ∧ k * x^2 + y = 4

theorem solution_in_quadrant_I (k : ℝ) :
  (∃ x y : ℝ, system_equations k x y ∧ in_quadrant_I x y) ↔ k > 0 :=
sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_I_l403_40350


namespace NUMINAMATH_CALUDE_new_savings_approx_400_l403_40348

/-- Represents the monthly salary in rupees -/
def monthly_salary : ℝ := 7272.727272727273

/-- Represents the initial savings rate as a decimal -/
def initial_savings_rate : ℝ := 0.10

/-- Represents the expense increase rate as a decimal -/
def expense_increase_rate : ℝ := 0.05

/-- Calculates the new monthly savings after the expense increase -/
def new_monthly_savings : ℝ :=
  monthly_salary * (1 - (1 - initial_savings_rate) * (1 + expense_increase_rate))

/-- Theorem stating that the new monthly savings is approximately 400 rupees -/
theorem new_savings_approx_400 :
  ∃ ε > 0, |new_monthly_savings - 400| < ε :=
sorry

end NUMINAMATH_CALUDE_new_savings_approx_400_l403_40348


namespace NUMINAMATH_CALUDE_johns_sleep_theorem_l403_40336

theorem johns_sleep_theorem :
  let days_in_week : ℕ := 7
  let short_sleep_days : ℕ := 2
  let short_sleep_hours : ℝ := 3
  let recommended_sleep : ℝ := 8
  let sleep_percentage : ℝ := 0.6

  let normal_sleep_days : ℕ := days_in_week - short_sleep_days
  let normal_sleep_hours : ℝ := sleep_percentage * recommended_sleep
  
  let total_sleep : ℝ := 
    (short_sleep_days : ℝ) * short_sleep_hours + 
    (normal_sleep_days : ℝ) * normal_sleep_hours

  total_sleep = 30 := by sorry

end NUMINAMATH_CALUDE_johns_sleep_theorem_l403_40336


namespace NUMINAMATH_CALUDE_min_sum_squares_cube_edges_l403_40347

/-- Represents a cube with 8 vertices -/
structure Cube :=
  (v1 v2 v3 v4 v5 v6 v7 v8 : ℝ)

/-- Calculates the sum of squares of differences on the edges of a cube -/
def sumOfSquaresOfDifferences (c : Cube) : ℝ :=
  (c.v1 - c.v2)^2 + (c.v1 - c.v3)^2 + (c.v1 - c.v5)^2 +
  (c.v2 - c.v4)^2 + (c.v2 - c.v6)^2 +
  (c.v3 - c.v4)^2 + (c.v3 - c.v7)^2 +
  (c.v4 - c.v8)^2 +
  (c.v5 - c.v6)^2 + (c.v5 - c.v7)^2 +
  (c.v6 - c.v8)^2 +
  (c.v7 - c.v8)^2

/-- Theorem stating the minimum sum of squares of differences on cube edges -/
theorem min_sum_squares_cube_edges :
  ∃ (c : Cube),
    c.v1 = 0 ∧
    c.v8 = 2013 ∧
    c.v2 = 2013/2 ∧
    c.v3 = 2013/2 ∧
    c.v4 = 2013/2 ∧
    c.v5 = 2013/2 ∧
    c.v6 = 2013/2 ∧
    c.v7 = 2013/2 ∧
    sumOfSquaresOfDifferences c = (3 * 2013^2) / 2 ∧
    ∀ (c' : Cube), c'.v1 = 0 ∧ c'.v8 = 2013 →
      sumOfSquaresOfDifferences c' ≥ sumOfSquaresOfDifferences c :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_cube_edges_l403_40347
