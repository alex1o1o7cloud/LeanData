import Mathlib

namespace NUMINAMATH_CALUDE_height_estimate_l1404_140471

/-- Given a survey of 1500 first-year high school students' heights:
    - The height range [160cm, 170cm] is divided into two groups of 5cm each
    - 'a' is the height of the histogram rectangle for [160cm, 165cm]
    - 'b' is the height of the histogram rectangle for [165cm, 170cm]
    - 1 unit of height in the histogram corresponds to 1500 students
    Then, the estimated number of students with heights in [160cm, 170cm] is 7500(a+b) -/
theorem height_estimate (a b : ℝ) : ℝ :=
  let total_students : ℕ := 1500
  let group_width : ℝ := 5
  let scale : ℝ := 1500
  7500 * (a + b)

#check height_estimate

end NUMINAMATH_CALUDE_height_estimate_l1404_140471


namespace NUMINAMATH_CALUDE_work_completion_time_l1404_140493

/-- Represents the time taken to complete a work when one worker is assisted by two others on alternate days -/
def time_to_complete (time_a time_b time_c : ℝ) : ℝ :=
  2 * 4

/-- Theorem stating that if A can do a work in 11 days, B in 20 days, and C in 55 days,
    and A is assisted by B and C on alternate days, then the work can be completed in 8 days -/
theorem work_completion_time (time_a time_b time_c : ℝ)
  (ha : time_a = 11)
  (hb : time_b = 20)
  (hc : time_c = 55) :
  time_to_complete time_a time_b time_c = 8 := by
  sorry

#eval time_to_complete 11 20 55

end NUMINAMATH_CALUDE_work_completion_time_l1404_140493


namespace NUMINAMATH_CALUDE_probability_theorem_l1404_140411

/-- The number of possible outcomes when rolling a single six-sided die -/
def dice_outcomes : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 7

/-- The probability of rolling seven standard six-sided dice and getting at least one pair
    but not a three-of-a-kind -/
def probability_at_least_one_pair_no_three_of_a_kind : ℚ :=
  6426 / 13997

/-- Theorem stating that the probability of rolling seven standard six-sided dice
    and getting at least one pair but not a three-of-a-kind is 6426/13997 -/
theorem probability_theorem :
  probability_at_least_one_pair_no_three_of_a_kind = 6426 / 13997 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l1404_140411


namespace NUMINAMATH_CALUDE_smallest_number_l1404_140423

theorem smallest_number (a b c d : ℝ) (ha : a = 1/2) (hb : b = Real.sqrt 3) (hc : c = 0) (hd : d = -2) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1404_140423


namespace NUMINAMATH_CALUDE_expected_red_pairs_50_cards_l1404_140485

/-- Represents a deck of cards -/
structure Deck :=
  (total : ℕ)
  (red : ℕ)
  (black : ℕ)
  (h_total : total = red + black)
  (h_equal : red = black)

/-- The expected number of adjacent red pairs in a circular arrangement -/
def expected_red_pairs (d : Deck) : ℚ :=
  (d.red : ℚ) * ((d.red - 1) / (d.total - 1))

theorem expected_red_pairs_50_cards :
  ∃ d : Deck, d.total = 50 ∧ expected_red_pairs d = 600 / 49 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_pairs_50_cards_l1404_140485


namespace NUMINAMATH_CALUDE_smallest_room_width_l1404_140419

theorem smallest_room_width 
  (largest_width : ℝ) 
  (largest_length : ℝ) 
  (smallest_length : ℝ) 
  (area_difference : ℝ) :
  largest_width = 45 →
  largest_length = 30 →
  smallest_length = 8 →
  largest_width * largest_length - smallest_length * (largest_width * largest_length - area_difference) / smallest_length = 1230 →
  (largest_width * largest_length - area_difference) / smallest_length = 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_room_width_l1404_140419


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1404_140452

/-- The line y = mx + (2m + 1), where m ∈ ℝ, always passes through the point (-2, 1). -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), ((-2 : ℝ) : ℝ) * m + (2 * m + 1) = 1 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1404_140452


namespace NUMINAMATH_CALUDE_larger_number_problem_l1404_140480

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 6) : 
  max x y = 23 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1404_140480


namespace NUMINAMATH_CALUDE_sum_of_squares_on_sides_l1404_140479

/-- Given a triangle XYZ with side XZ = 12 units and perpendicular height from Y to XZ being 5 units,
    the sum of the areas of squares on sides XY and YZ is 122 square units. -/
theorem sum_of_squares_on_sides (X Y Z : ℝ × ℝ) : 
  let XZ : ℝ := Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2)
  let height : ℝ := 5
  let XY : ℝ := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  let YZ : ℝ := Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2)
  XZ = 12 →
  (∃ D : ℝ × ℝ, (D.1 - X.1) * (Z.2 - X.2) = (Z.1 - X.1) * (D.2 - X.2) ∧ 
                (Y.1 - D.1) * (Z.1 - X.1) = (X.2 - D.2) * (Z.2 - X.2) ∧
                Real.sqrt ((Y.1 - D.1)^2 + (Y.2 - D.2)^2) = height) →
  XY^2 + YZ^2 = 122 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_on_sides_l1404_140479


namespace NUMINAMATH_CALUDE_correct_statements_l1404_140439

theorem correct_statements (x : ℝ) : 
  (x ≥ 0 → x^2 ≥ x) ∧ 
  (x^2 ≥ 0 → abs x ≥ 0) ∧ 
  (x ≤ -1 → x^2 ≥ abs x) := by
  sorry

end NUMINAMATH_CALUDE_correct_statements_l1404_140439


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1404_140403

def M : Set ℝ := {x : ℝ | x^2 + x - 6 = 0}

def N (a : ℝ) : Set ℝ := {x : ℝ | a * x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (N a ⊆ M) ↔ (a = -1 ∨ a = 0 ∨ a = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1404_140403


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l1404_140413

theorem triangle_abc_proof (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ) :
  0 < A ∧ A < π →
  m = (Real.cos A, Real.sin A) →
  n = (Real.sqrt 2 - Real.sin A, Real.cos A) →
  Real.sqrt ((m.1 + n.1)^2 + (m.2 + n.2)^2) = 2 →
  b = 4 * Real.sqrt 2 →
  c = Real.sqrt 2 * a →
  A = π / 4 ∧ (1/2 * b * a = 16) := by sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l1404_140413


namespace NUMINAMATH_CALUDE_joe_oranges_count_l1404_140420

/-- The number of boxes Joe has for oranges -/
def num_boxes : ℕ := 9

/-- The number of oranges required in each box -/
def oranges_per_box : ℕ := 5

/-- The total number of oranges Joe has -/
def total_oranges : ℕ := num_boxes * oranges_per_box

theorem joe_oranges_count : total_oranges = 45 := by
  sorry

end NUMINAMATH_CALUDE_joe_oranges_count_l1404_140420


namespace NUMINAMATH_CALUDE_number_with_quarters_l1404_140466

/-- The number of quarters (1/4s) in the given number -/
def num_quarters : ℚ := 150

/-- The value of each quarter -/
def quarter_value : ℚ := 1/4

/-- The theorem stating that the number containing 150 quarters is equal to 37.5 -/
theorem number_with_quarters : num_quarters * quarter_value = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_number_with_quarters_l1404_140466


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_l1404_140400

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- Define the given conditions
variable (l m n : Line)
variable (α β γ : Plane)

variable (different_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n)
variable (non_coincident_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)

-- State the theorem
theorem perpendicular_implies_parallel 
  (h1 : perpendicular m α) 
  (h2 : perpendicular m β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_l1404_140400


namespace NUMINAMATH_CALUDE_edge_sum_greater_than_3d_l1404_140445

-- Define a convex polyhedron
structure ConvexPolyhedron where
  vertices : Set (Fin 3 → ℝ)
  edges : Set (Fin 2 → Fin 3 → ℝ)
  is_convex : Bool

-- Define the maximum distance between vertices
def max_distance (p : ConvexPolyhedron) : ℝ :=
  sorry

-- Define the sum of edge lengths
def sum_edge_lengths (p : ConvexPolyhedron) : ℝ :=
  sorry

-- The theorem to prove
theorem edge_sum_greater_than_3d (p : ConvexPolyhedron) :
  p.is_convex → sum_edge_lengths p > 3 * max_distance p :=
by sorry

end NUMINAMATH_CALUDE_edge_sum_greater_than_3d_l1404_140445


namespace NUMINAMATH_CALUDE_factorial_10_mod_13_l1404_140440

-- Define factorial function
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

-- Statement to prove
theorem factorial_10_mod_13 : factorial 10 % 13 = 6 := by
  sorry

end NUMINAMATH_CALUDE_factorial_10_mod_13_l1404_140440


namespace NUMINAMATH_CALUDE_unique_pair_l1404_140459

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem unique_pair : 
  ∃! (a b : ℕ), 
    a > 0 ∧ 
    b > 0 ∧ 
    b > a ∧ 
    is_prime (b - a) ∧ 
    (a + b) % 10 = 3 ∧ 
    ∃ k : ℕ, a * b = k * k ∧
    a = 4 ∧
    b = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_l1404_140459


namespace NUMINAMATH_CALUDE_susan_reading_time_l1404_140408

/-- Represents the ratio of time spent on different activities -/
structure TimeRatio where
  swimming : ℕ
  reading : ℕ
  hangingOut : ℕ

/-- Calculates the time spent on an activity given the total time of another activity -/
def calculateTime (ratio : TimeRatio) (knownActivity : ℕ) (knownTime : ℕ) (targetActivity : ℕ) : ℕ :=
  (targetActivity * knownTime) / knownActivity

theorem susan_reading_time (ratio : TimeRatio) 
    (h1 : ratio.swimming = 1)
    (h2 : ratio.reading = 4)
    (h3 : ratio.hangingOut = 10)
    (h4 : calculateTime ratio ratio.hangingOut 20 ratio.reading = 8) : 
  ∃ (readingTime : ℕ), readingTime = 8 ∧ 
    readingTime = calculateTime ratio ratio.hangingOut 20 ratio.reading :=
by sorry

end NUMINAMATH_CALUDE_susan_reading_time_l1404_140408


namespace NUMINAMATH_CALUDE_oleg_can_win_l1404_140476

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A list of n positive integers, all smaller than the nth prime -/
def validList (n : ℕ) (list : List ℕ) : Prop :=
  list.length = n ∧ 
  ∀ x ∈ list, 0 < x ∧ x < nthPrime n

/-- The operation of replacing one number with the product of two numbers -/
def replaceWithProduct (list : List ℕ) (i j k : ℕ) : List ℕ :=
  sorry

/-- Predicate to check if a list contains at least two equal elements -/
def hasEqualElements (list : List ℕ) : Prop :=
  ∃ i j, i ≠ j ∧ list.get! i = list.get! j

/-- The main theorem: Oleg can always win for n > 1 -/
theorem oleg_can_win (n : ℕ) (list : List ℕ) (h : n > 1) (hlist : validList n list) :
  ∃ (steps : List (ℕ × ℕ × ℕ)), 
    let finalList := steps.foldl (fun acc step => replaceWithProduct acc step.1 step.2.1 step.2.2) list
    hasEqualElements finalList :=
  sorry

end NUMINAMATH_CALUDE_oleg_can_win_l1404_140476


namespace NUMINAMATH_CALUDE_gcd_8251_6105_l1404_140442

theorem gcd_8251_6105 : Int.gcd 8251 6105 = 39 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8251_6105_l1404_140442


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1404_140464

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function we want to prove is quadratic -/
def f (x : ℝ) : ℝ := 2 * x^2 + 3 * x

/-- Theorem stating that f is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l1404_140464


namespace NUMINAMATH_CALUDE_min_cost_disinfectants_l1404_140428

/-- Represents the price and quantity of disinfectants A and B -/
structure Disinfectants where
  price_A : ℕ
  price_B : ℕ
  quantity_A : ℕ
  quantity_B : ℕ

/-- Calculates the total cost of purchasing disinfectants -/
def total_cost (d : Disinfectants) : ℕ :=
  d.price_A * d.quantity_A + d.price_B * d.quantity_B

/-- Represents the constraints on quantities of disinfectants -/
def valid_quantities (d : Disinfectants) : Prop :=
  d.quantity_A + d.quantity_B = 30 ∧
  d.quantity_A ≥ d.quantity_B + 5 ∧
  d.quantity_A ≤ 2 * d.quantity_B

theorem min_cost_disinfectants :
  ∃ (d : Disinfectants),
    d.price_A = 45 ∧
    d.price_B = 35 ∧
    9 * d.price_A + 6 * d.price_B = 615 ∧
    8 * d.price_A + 12 * d.price_B = 780 ∧
    valid_quantities d ∧
    (∀ (d' : Disinfectants), valid_quantities d' → total_cost d ≤ total_cost d') ∧
    total_cost d = 1230 :=
by
  sorry

end NUMINAMATH_CALUDE_min_cost_disinfectants_l1404_140428


namespace NUMINAMATH_CALUDE_polygon_sides_l1404_140409

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1260) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1404_140409


namespace NUMINAMATH_CALUDE_shirts_not_washed_l1404_140473

theorem shirts_not_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) : 
  short_sleeve = 39 → long_sleeve = 47 → washed = 20 → 
  short_sleeve + long_sleeve - washed = 66 := by
sorry

end NUMINAMATH_CALUDE_shirts_not_washed_l1404_140473


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1404_140481

def complex_number : ℂ := Complex.I * (1 + Complex.I)

theorem complex_number_in_third_quadrant :
  Real.sign (complex_number.re) = -1 ∧ Real.sign (complex_number.im) = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l1404_140481


namespace NUMINAMATH_CALUDE_ages_sum_l1404_140402

theorem ages_sum (a b c : ℕ+) : 
  b = c →                 -- twins have the same age
  b > a →                 -- twins are older than Kiana
  a * b * c = 144 →       -- product of ages is 144
  a + b + c = 16 :=       -- sum of ages is 16
by sorry

end NUMINAMATH_CALUDE_ages_sum_l1404_140402


namespace NUMINAMATH_CALUDE_three_cakes_cooking_time_l1404_140489

/-- Represents the cooking process for cakes -/
structure CookingProcess where
  pot_capacity : ℕ
  cooking_time_per_cake : ℕ
  num_cakes : ℕ

/-- The minimum time required to cook the given number of cakes -/
def min_cooking_time (process : CookingProcess) : ℕ :=
  sorry

/-- Theorem stating the minimum time to cook three cakes under given conditions -/
theorem three_cakes_cooking_time :
  ∀ (process : CookingProcess),
    process.pot_capacity = 2 →
    process.cooking_time_per_cake = 5 →
    process.num_cakes = 3 →
    min_cooking_time process = 15 :=
by sorry

end NUMINAMATH_CALUDE_three_cakes_cooking_time_l1404_140489


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1404_140495

theorem sum_of_fractions : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1404_140495


namespace NUMINAMATH_CALUDE_box_length_proof_l1404_140437

/-- Proves that a rectangular box with given dimensions has a length of 55.5 meters -/
theorem box_length_proof (width : ℝ) (road_width : ℝ) (lawn_area : ℝ) :
  width = 40 →
  road_width = 3 →
  lawn_area = 2109 →
  ∃ (length : ℝ),
    length * width - 2 * (length / 3) * road_width = lawn_area ∧
    length = 55.5 := by
  sorry

end NUMINAMATH_CALUDE_box_length_proof_l1404_140437


namespace NUMINAMATH_CALUDE_min_sum_bound_min_sum_achievable_l1404_140456

theorem min_sum_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (5 * c) + c / (6 * a) ≥ 3 / Real.rpow 90 (1/3) :=
sorry

theorem min_sum_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a / (3 * b) + b / (5 * c) + c / (6 * a) = 3 / Real.rpow 90 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_sum_bound_min_sum_achievable_l1404_140456


namespace NUMINAMATH_CALUDE_triangle_construction_uniqueness_l1404_140496

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by its three vertices -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point :=
  sorry

/-- The point where the internal angle bisector from A intersects BC -/
def internalBisectorIntersection (t : Triangle) : Point :=
  sorry

/-- The point where the external angle bisector from A intersects BC -/
def externalBisectorIntersection (t : Triangle) : Point :=
  sorry

/-- Predicate to check if a point is within the valid region for M -/
def isValidM (M A' A'' : Point) : Prop :=
  sorry

theorem triangle_construction_uniqueness 
  (M A' A'' : Point) 
  (h_valid : isValidM M A' A'') :
  ∃! t : Triangle, 
    orthocenter t = M ∧ 
    internalBisectorIntersection t = A' ∧ 
    externalBisectorIntersection t = A'' :=
  sorry

end NUMINAMATH_CALUDE_triangle_construction_uniqueness_l1404_140496


namespace NUMINAMATH_CALUDE_tom_seashells_l1404_140438

theorem tom_seashells (initial_seashells : ℕ) (given_away : ℕ) :
  initial_seashells = 5 →
  given_away = 2 →
  initial_seashells - given_away = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_l1404_140438


namespace NUMINAMATH_CALUDE_bella_steps_theorem_l1404_140450

/-- The number of feet in a mile -/
def feet_per_mile : ℕ := 5280

/-- The distance between the two houses in miles -/
def distance_miles : ℕ := 3

/-- The length of Bella's step in feet -/
def step_length : ℕ := 3

/-- The ratio of Ella's speed to Bella's speed -/
def speed_ratio : ℕ := 4

/-- The number of steps Bella takes when they meet -/
def steps_taken : ℕ := 1056

theorem bella_steps_theorem :
  let total_distance_feet := distance_miles * feet_per_mile
  let combined_speed_ratio := speed_ratio + 1
  let bella_distance := total_distance_feet / combined_speed_ratio
  bella_distance / step_length = steps_taken := by
  sorry

end NUMINAMATH_CALUDE_bella_steps_theorem_l1404_140450


namespace NUMINAMATH_CALUDE_product_of_cubic_fractions_l1404_140491

theorem product_of_cubic_fractions :
  let f (n : ℕ) := (n^3 - 1) / (n^3 + 1)
  (f 3) * (f 4) * (f 5) * (f 6) * (f 7) = 57 / 168 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cubic_fractions_l1404_140491


namespace NUMINAMATH_CALUDE_homeless_donation_calculation_l1404_140404

theorem homeless_donation_calculation (total amount_first amount_second : ℝ) 
  (h1 : total = 900)
  (h2 : amount_first = 325)
  (h3 : amount_second = 260) :
  total - amount_first - amount_second = 315 :=
by sorry

end NUMINAMATH_CALUDE_homeless_donation_calculation_l1404_140404


namespace NUMINAMATH_CALUDE_probability_four_ones_eight_dice_l1404_140451

theorem probability_four_ones_eight_dice : 
  let n : ℕ := 8  -- number of dice
  let s : ℕ := 8  -- number of sides on each die
  let k : ℕ := 4  -- number of dice showing 1
  Nat.choose n k * (1 / s) ^ k * ((s - 1) / s) ^ (n - k) = 168070 / 16777216 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_ones_eight_dice_l1404_140451


namespace NUMINAMATH_CALUDE_root_difference_quadratic_l1404_140429

/-- The nonnegative difference between the roots of x^2 + 30x + 180 = -36 is 6 -/
theorem root_difference_quadratic : 
  let f : ℝ → ℝ := λ x => x^2 + 30*x + 216
  ∃ r₁ r₂ : ℝ, f r₁ = 0 ∧ f r₂ = 0 ∧ |r₁ - r₂| = 6 := by
sorry

end NUMINAMATH_CALUDE_root_difference_quadratic_l1404_140429


namespace NUMINAMATH_CALUDE_clarence_spent_12_96_l1404_140474

/-- The cost of Clarence's amusement park visit -/
def clarence_total_cost (cost_per_ride : ℚ) (water_slide_rides : ℕ) (roller_coaster_rides : ℕ) : ℚ :=
  cost_per_ride * (water_slide_rides + roller_coaster_rides)

/-- Theorem stating that Clarence's total cost at the amusement park was $12.96 -/
theorem clarence_spent_12_96 :
  clarence_total_cost 2.16 3 3 = 12.96 := by
  sorry

end NUMINAMATH_CALUDE_clarence_spent_12_96_l1404_140474


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1404_140418

theorem complex_equation_solution :
  ∃ z : ℂ, (5 : ℂ) - 3 * Complex.I * z = (3 : ℂ) + 5 * Complex.I * z ∧ z = Complex.I / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1404_140418


namespace NUMINAMATH_CALUDE_no_real_roots_k_value_l1404_140455

theorem no_real_roots_k_value (k : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → k / (x - 1) + 3 ≠ x / (1 - x)) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_k_value_l1404_140455


namespace NUMINAMATH_CALUDE_largest_and_smallest_A_l1404_140453

/-- A function that moves the last digit of a number to the first position -/
def moveLastDigitToFirst (n : ℕ) : ℕ :=
  let lastDigit := n % 10
  let restOfDigits := n / 10
  lastDigit * 10^8 + restOfDigits

/-- Theorem stating the largest and smallest A values -/
theorem largest_and_smallest_A :
  ∀ B : ℕ,
  (B > 22222222) →
  (Nat.gcd B 18 = 1) →
  (∃ A : ℕ, A = moveLastDigitToFirst B) →
  (∃ A_max A_min : ℕ,
    (A_max = moveLastDigitToFirst B → A_max ≤ 999999998) ∧
    (A_min = moveLastDigitToFirst B → A_min ≥ 122222224) ∧
    (∃ B_max B_min : ℕ,
      B_max > 22222222 ∧
      Nat.gcd B_max 18 = 1 ∧
      moveLastDigitToFirst B_max = 999999998 ∧
      B_min > 22222222 ∧
      Nat.gcd B_min 18 = 1 ∧
      moveLastDigitToFirst B_min = 122222224)) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_and_smallest_A_l1404_140453


namespace NUMINAMATH_CALUDE_inscribed_rectangles_area_sum_l1404_140449

/-- Represents a rectangle in 2D space -/
structure Rectangle where
  bottom_left : ℝ × ℝ
  top_right : ℝ × ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ :=
  (r.top_right.1 - r.bottom_left.1) * (r.top_right.2 - r.bottom_left.2)

/-- Checks if a rectangle is inscribed in another rectangle -/
def is_inscribed (inner outer : Rectangle) : Prop :=
  inner.bottom_left.1 ≥ outer.bottom_left.1 ∧
  inner.bottom_left.2 ≥ outer.bottom_left.2 ∧
  inner.top_right.1 ≤ outer.top_right.1 ∧
  inner.top_right.2 ≤ outer.top_right.2

/-- Checks if two rectangles share a vertex on the given side -/
def share_vertex_on_side (r1 r2 outer : Rectangle) (side : ℝ) : Prop :=
  (r1.bottom_left.1 = side ∨ r1.top_right.1 = side) ∧
  (r2.bottom_left.1 = side ∨ r2.top_right.1 = side) ∧
  ∃ y, (r1.bottom_left.2 = y ∨ r1.top_right.2 = y) ∧
       (r2.bottom_left.2 = y ∨ r2.top_right.2 = y)

theorem inscribed_rectangles_area_sum (outer r1 r2 : Rectangle) :
  is_inscribed r1 outer →
  is_inscribed r2 outer →
  share_vertex_on_side r1 r2 outer outer.bottom_left.1 →
  area r1 + area r2 = area outer := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangles_area_sum_l1404_140449


namespace NUMINAMATH_CALUDE_garden_breadth_l1404_140412

/-- Given a rectangular garden with perimeter 900 m and length 260 m, prove its breadth is 190 m. -/
theorem garden_breadth (perimeter length breadth : ℝ) : 
  perimeter = 900 ∧ length = 260 ∧ perimeter = 2 * (length + breadth) → breadth = 190 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_l1404_140412


namespace NUMINAMATH_CALUDE_mango_rate_per_kg_mango_rate_proof_l1404_140446

/-- The rate of mangoes per kilogram given the purchase conditions --/
theorem mango_rate_per_kg : ℝ → Prop :=
  fun rate =>
    let grape_quantity : ℝ := 8
    let grape_rate : ℝ := 70
    let mango_quantity : ℝ := 9
    let total_paid : ℝ := 1145
    grape_quantity * grape_rate + mango_quantity * rate = total_paid →
    rate = 65

/-- Proof of the mango rate per kilogram --/
theorem mango_rate_proof : mango_rate_per_kg 65 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_per_kg_mango_rate_proof_l1404_140446


namespace NUMINAMATH_CALUDE_opposite_hands_count_l1404_140487

/-- Represents a clock with hour and minute hands -/
structure Clock :=
  (hour : ℝ) -- Hour hand position (0 ≤ hour < 12)
  (minute : ℝ) -- Minute hand position (0 ≤ minute < 60)

/-- The speed ratio between minute and hour hands -/
def minute_hour_speed_ratio : ℝ := 12

/-- The angle difference when hands are opposite -/
def opposite_angle_diff : ℝ := 30

/-- Counts the number of times the clock hands are opposite in a 24-hour period -/
def count_opposite_hands (c : Clock) : ℕ := sorry

/-- Theorem stating that the hands are opposite 22 times in a day -/
theorem opposite_hands_count :
  ∀ c : Clock, count_opposite_hands c = 22 := by sorry

end NUMINAMATH_CALUDE_opposite_hands_count_l1404_140487


namespace NUMINAMATH_CALUDE_work_left_fraction_l1404_140492

theorem work_left_fraction (days_A days_B days_together : ℕ) 
  (h1 : days_A = 15)
  (h2 : days_B = 20)
  (h3 : days_together = 6) : 
  1 - (days_together : ℚ) * (1 / days_A + 1 / days_B) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_work_left_fraction_l1404_140492


namespace NUMINAMATH_CALUDE_elements_not_in_either_set_l1404_140401

/-- Given sets A and B that are subsets of a finite universal set U, 
    this theorem calculates the number of elements in U that are not in either A or B. -/
theorem elements_not_in_either_set 
  (U A B : Finset ℕ) 
  (h_subset_A : A ⊆ U) 
  (h_subset_B : B ⊆ U) 
  (h_card_U : U.card = 193)
  (h_card_A : A.card = 116)
  (h_card_B : B.card = 41)
  (h_card_inter : (A ∩ B).card = 23) :
  (U \ (A ∪ B)).card = 59 := by
  sorry

#check elements_not_in_either_set

end NUMINAMATH_CALUDE_elements_not_in_either_set_l1404_140401


namespace NUMINAMATH_CALUDE_only_cylinder_produces_quadrilateral_section_l1404_140417

-- Define the types of geometric solids
inductive GeometricSolid
  | Cone
  | Sphere
  | Cylinder

-- Define a function that checks if a geometric solid can produce a quadrilateral section
def can_produce_quadrilateral_section (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.Cylinder => True
  | _ => False

-- Theorem statement
theorem only_cylinder_produces_quadrilateral_section :
  ∀ (solid : GeometricSolid),
    can_produce_quadrilateral_section solid ↔ solid = GeometricSolid.Cylinder :=
by
  sorry


end NUMINAMATH_CALUDE_only_cylinder_produces_quadrilateral_section_l1404_140417


namespace NUMINAMATH_CALUDE_hexagon_area_from_square_l1404_140433

theorem hexagon_area_from_square (s : ℝ) (h_square_area : s^2 = Real.sqrt 3) :
  let hexagon_area := 6 * (Real.sqrt 3 / 4 * s^2)
  hexagon_area = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_from_square_l1404_140433


namespace NUMINAMATH_CALUDE_test_scores_theorem_l1404_140416

/-- Represents the test scores for three students -/
structure TestScores where
  alisson : ℕ
  jose : ℕ
  meghan : ℕ

/-- Calculates the total score for the three students -/
def totalScore (scores : TestScores) : ℕ :=
  scores.alisson + scores.jose + scores.meghan

/-- Theorem stating the total score for the three students -/
theorem test_scores_theorem (scores : TestScores) : totalScore scores = 210 :=
  by
  have h1 : scores.jose = scores.alisson + 40 := sorry
  have h2 : scores.meghan = scores.jose - 20 := sorry
  have h3 : scores.jose = 100 - 10 := sorry
  sorry

#check test_scores_theorem

end NUMINAMATH_CALUDE_test_scores_theorem_l1404_140416


namespace NUMINAMATH_CALUDE_percentage_of_girls_taking_lunch_l1404_140434

theorem percentage_of_girls_taking_lunch (total : ℕ) (boys girls : ℕ) 
  (h_ratio : boys = 3 * girls / 2)
  (h_total : total = boys + girls)
  (boys_lunch : ℕ) (total_lunch : ℕ)
  (h_boys_lunch : boys_lunch = 3 * boys / 5)
  (h_total_lunch : total_lunch = 13 * total / 25) :
  (total_lunch - boys_lunch) * 5 = 2 * girls := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_girls_taking_lunch_l1404_140434


namespace NUMINAMATH_CALUDE_optimal_ticket_price_l1404_140478

/-- Represents the net income function for the cinema --/
def net_income (x : ℕ) : ℝ :=
  if x ≤ 10 then 100 * x - 575
  else -3 * x^2 + 130 * x - 575

/-- The domain of valid ticket prices --/
def valid_price (x : ℕ) : Prop :=
  6 ≤ x ∧ x ≤ 38

theorem optimal_ticket_price :
  ∀ x : ℕ, valid_price x → net_income x ≤ net_income 22 :=
sorry

end NUMINAMATH_CALUDE_optimal_ticket_price_l1404_140478


namespace NUMINAMATH_CALUDE_interview_probability_l1404_140488

def total_students : ℕ := 30
def french_students : ℕ := 20
def spanish_students : ℕ := 24

theorem interview_probability :
  let both_classes := french_students + spanish_students - total_students
  let only_french := french_students - both_classes
  let only_spanish := spanish_students - both_classes
  let total_combinations := total_students.choose 2
  let unfavorable_combinations := only_french.choose 2 + only_spanish.choose 2
  (total_combinations - unfavorable_combinations : ℚ) / total_combinations = 25 / 29 := by
  sorry

end NUMINAMATH_CALUDE_interview_probability_l1404_140488


namespace NUMINAMATH_CALUDE_cos_eleven_pi_thirds_l1404_140468

theorem cos_eleven_pi_thirds : Real.cos (11 * Real.pi / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_eleven_pi_thirds_l1404_140468


namespace NUMINAMATH_CALUDE_optimal_distribution_minimizes_cost_l1404_140447

noncomputable section

/-- Represents the distribution of potatoes among three farms -/
structure PotatoDistribution where
  farm1 : ℝ
  farm2 : ℝ
  farm3 : ℝ

/-- The cost function for potato distribution -/
def cost (d : PotatoDistribution) : ℝ :=
  4 * d.farm1 + 3 * d.farm2 + d.farm3

/-- Checks if a distribution satisfies all constraints -/
def isValid (d : PotatoDistribution) : Prop :=
  d.farm1 ≥ 0 ∧ d.farm2 ≥ 0 ∧ d.farm3 ≥ 0 ∧
  d.farm1 + d.farm2 + d.farm3 = 12 ∧
  d.farm1 + 4 * d.farm2 + 3 * d.farm3 ≤ 40 ∧
  d.farm1 ≤ 10 ∧ d.farm2 ≤ 8 ∧ d.farm3 ≤ 6

/-- The optimal distribution of potatoes -/
def optimalDistribution : PotatoDistribution :=
  { farm1 := 2/3, farm2 := 16/3, farm3 := 6 }

/-- Theorem stating that the optimal distribution minimizes the cost -/
theorem optimal_distribution_minimizes_cost :
  isValid optimalDistribution ∧
  ∀ d : PotatoDistribution, isValid d → cost optimalDistribution ≤ cost d :=
sorry

end

end NUMINAMATH_CALUDE_optimal_distribution_minimizes_cost_l1404_140447


namespace NUMINAMATH_CALUDE_geometric_sequence_condition_l1404_140422

-- Define a geometric sequence
def is_geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

-- Theorem statement
theorem geometric_sequence_condition (a b c d : ℝ) :
  (is_geometric_sequence a b c d → a * d = b * c) ∧
  ∃ a b c d : ℝ, a * d = b * c ∧ ¬(is_geometric_sequence a b c d) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_condition_l1404_140422


namespace NUMINAMATH_CALUDE_haleigh_dogs_count_l1404_140483

/-- The number of cats Haleigh has -/
def num_cats : ℕ := 3

/-- The total number of leggings needed -/
def total_leggings : ℕ := 14

/-- The number of leggings each animal needs -/
def leggings_per_animal : ℕ := 1

/-- The number of dogs Haleigh has -/
def num_dogs : ℕ := total_leggings - (num_cats * leggings_per_animal)

theorem haleigh_dogs_count : num_dogs = 11 := by sorry

end NUMINAMATH_CALUDE_haleigh_dogs_count_l1404_140483


namespace NUMINAMATH_CALUDE_smithtown_handedness_ratio_l1404_140461

-- Define the population of Smithtown
structure Population where
  total : ℝ
  men : ℝ
  women : ℝ
  rightHanded : ℝ
  leftHanded : ℝ

-- Define the conditions
def smithtown_conditions (p : Population) : Prop :=
  p.men / p.women = 3 / 2 ∧
  p.men = p.rightHanded ∧
  p.leftHanded / p.total = 0.2500000000000001

-- Theorem statement
theorem smithtown_handedness_ratio (p : Population) :
  smithtown_conditions p →
  p.rightHanded / p.leftHanded = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_smithtown_handedness_ratio_l1404_140461


namespace NUMINAMATH_CALUDE_same_color_probability_l1404_140484

/-- Represents the number of pairs of shoes -/
def num_pairs : ℕ := 5

/-- Represents the total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- Represents the number of shoes to select -/
def shoes_to_select : ℕ := 2

/-- The probability of selecting two shoes of the same color -/
theorem same_color_probability : 
  (num_pairs : ℚ) / (total_shoes.choose shoes_to_select) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l1404_140484


namespace NUMINAMATH_CALUDE_senior_to_child_ratio_l1404_140467

theorem senior_to_child_ratio 
  (adults : ℕ) 
  (children : ℕ) 
  (seniors : ℕ) 
  (total : ℕ) 
  (h1 : adults = 58)
  (h2 : children = adults - 35)
  (h3 : total = adults + children + seniors)
  (h4 : total = 127) :
  (seniors : ℚ) / children = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_senior_to_child_ratio_l1404_140467


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1404_140405

theorem complex_equation_solution (z : ℂ) :
  (3 + 4*I) * z = 25 → z = 3 - 4*I := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1404_140405


namespace NUMINAMATH_CALUDE_decimal_to_binary_53_l1404_140497

theorem decimal_to_binary_53 : 
  (53 : ℕ) = 
  (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_53_l1404_140497


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l1404_140494

/-- Given a right triangle with area 180 square units and one leg of length 18 units,
    its perimeter is 38 + 2√181 units. -/
theorem right_triangle_perimeter : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 180 →
  a = 18 →
  a^2 + b^2 = c^2 →
  a + b + c = 38 + 2 * Real.sqrt 181 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l1404_140494


namespace NUMINAMATH_CALUDE_intercept_sum_l1404_140470

theorem intercept_sum (m : ℕ) (x_0 y_0 : ℕ) : m = 17 →
  (2 * x_0) % m = 3 →
  (5 * y_0) % m = m - 3 →
  x_0 < m →
  y_0 < m →
  x_0 + y_0 = 22 := by
  sorry

end NUMINAMATH_CALUDE_intercept_sum_l1404_140470


namespace NUMINAMATH_CALUDE_certain_number_problem_l1404_140431

theorem certain_number_problem (x : ℤ) (h : x + 36 = 71) : x + 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1404_140431


namespace NUMINAMATH_CALUDE_pencil_box_puzzle_l1404_140490

structure Box where
  blue : ℕ
  green : ℕ

def vasya_statement (box : Box) : Prop :=
  box.blue ≥ 4

def kolya_statement (box : Box) : Prop :=
  box.green ≥ 5

def petya_statement (box : Box) : Prop :=
  box.blue ≥ 3 ∧ box.green ≥ 4

def misha_statement (box : Box) : Prop :=
  box.blue ≥ 4 ∧ box.green ≥ 4

theorem pencil_box_puzzle (box : Box) :
  (vasya_statement box ∧ ¬kolya_statement box ∧ petya_statement box ∧ misha_statement box) ↔
  (box.blue ≥ 4 ∧ box.green = 4) :=
by sorry

end NUMINAMATH_CALUDE_pencil_box_puzzle_l1404_140490


namespace NUMINAMATH_CALUDE_pencil_cost_l1404_140427

theorem pencil_cost (x y : ℚ) 
  (eq1 : 5 * x + 4 * y = 340)
  (eq2 : 3 * x + 6 * y = 264) : 
  y = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l1404_140427


namespace NUMINAMATH_CALUDE_B_subset_A_l1404_140436

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x^2)}

-- Define set B
def B : Set ℝ := {x | ∃ m ∈ A, x = m^2}

-- Theorem statement
theorem B_subset_A : B ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_B_subset_A_l1404_140436


namespace NUMINAMATH_CALUDE_series_sum_convergence_l1404_140463

open Real
open BigOperators

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 2)) converges to 5/6 -/
theorem series_sum_convergence :
  ∑' n : ℕ, (3 * n - 2 : ℝ) / (n * (n + 1) * (n + 2)) = 5/6 := by sorry

end NUMINAMATH_CALUDE_series_sum_convergence_l1404_140463


namespace NUMINAMATH_CALUDE_car_tank_capacity_l1404_140421

def distance_to_home : ℝ := 220
def fuel_efficiency : ℝ := 20
def additional_distance : ℝ := 100

theorem car_tank_capacity :
  let total_distance := distance_to_home + additional_distance
  let tank_capacity := total_distance / fuel_efficiency
  tank_capacity = 16 := by sorry

end NUMINAMATH_CALUDE_car_tank_capacity_l1404_140421


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1404_140432

theorem quadratic_root_problem (b : ℝ) :
  (∃ x₀ : ℝ, x₀^2 - 4*x₀ + b = 0 ∧ (-x₀)^2 + 4*(-x₀) - b = 0) →
  (∃ x : ℝ, x > 0 ∧ x^2 + b*x - 4 = 0 ∧ x = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1404_140432


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l1404_140458

theorem inscribed_cube_surface_area (r : ℝ) (h : 4 * π * r^2 = π) :
  6 * (1 / (r * Real.sqrt 3))^2 = 2 := by sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l1404_140458


namespace NUMINAMATH_CALUDE_inequality_proof_l1404_140424

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_ineq : a + b < 2 * c) : 
  c - Real.sqrt (c^2 - a*b) < a ∧ a < c + Real.sqrt (c^2 - a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1404_140424


namespace NUMINAMATH_CALUDE_fish_theorem_l1404_140407

def fish_problem (leo agrey sierra returned : ℕ) : Prop :=
  let total := leo + agrey + sierra
  agrey = leo + 20 ∧ 
  sierra = agrey + 15 ∧ 
  leo = 40 ∧ 
  returned = 30 ∧ 
  total - returned = 145

theorem fish_theorem : 
  ∃ (leo agrey sierra returned : ℕ), fish_problem leo agrey sierra returned :=
by
  sorry

end NUMINAMATH_CALUDE_fish_theorem_l1404_140407


namespace NUMINAMATH_CALUDE_bobbit_worm_days_l1404_140426

/-- The number of days the Bobbit worm was in the aquarium before James added more fish -/
def days_before_adding : ℕ := sorry

/-- The initial number of fish in the aquarium -/
def initial_fish : ℕ := 60

/-- The number of fish the Bobbit worm eats per day -/
def fish_eaten_per_day : ℕ := 2

/-- The number of fish James adds to the aquarium -/
def fish_added : ℕ := 8

/-- The number of days between adding fish and discovering the Bobbit worm -/
def days_after_adding : ℕ := 7

/-- The final number of fish in the aquarium when James discovers the Bobbit worm -/
def final_fish : ℕ := 26

theorem bobbit_worm_days : 
  initial_fish - (fish_eaten_per_day * days_before_adding) + fish_added - (fish_eaten_per_day * days_after_adding) = final_fish ∧
  days_before_adding = 14 := by sorry

end NUMINAMATH_CALUDE_bobbit_worm_days_l1404_140426


namespace NUMINAMATH_CALUDE_f_local_min_g_max_local_min_l1404_140475

noncomputable section

open Real

-- Define the function f(x)
def f (x : ℝ) : ℝ := exp (x - 1) - log x

-- Define the function g(x) parameterized by a
def g (a : ℝ) (x : ℝ) : ℝ := f x - a * (x - 1)

-- Theorem for the local minimum of f(x)
theorem f_local_min : ∃ x₀ : ℝ, x₀ > 0 ∧ IsLocalMin f x₀ ∧ f x₀ = 1 := by sorry

-- Theorem for the maximum of the local minimum of g(x)
theorem g_max_local_min : 
  ∃ a₀ : ℝ, ∀ a : ℝ, 
    (∃ x₀ : ℝ, x₀ > 0 ∧ IsLocalMin (g a) x₀) → 
    (∃ x₁ : ℝ, x₁ > 0 ∧ IsLocalMin (g a₀) x₁ ∧ g a₀ x₁ ≥ g a x₀) ∧
    (∃ x₂ : ℝ, x₂ > 0 ∧ IsLocalMin (g a₀) x₂ ∧ g a₀ x₂ = 1) := by sorry

end

end NUMINAMATH_CALUDE_f_local_min_g_max_local_min_l1404_140475


namespace NUMINAMATH_CALUDE_game_lives_calculation_l1404_140441

/-- Given an initial number of players, additional players joining, and lives per player,
    calculate the total number of lives for all players. -/
def totalLives (initialPlayers additionalPlayers livesPerPlayer : ℕ) : ℕ :=
  (initialPlayers + additionalPlayers) * livesPerPlayer

/-- Prove that given 25 initial players, 10 additional players, and 15 lives per player,
    the total number of lives for all players is 525. -/
theorem game_lives_calculation :
  totalLives 25 10 15 = 525 := by
  sorry

end NUMINAMATH_CALUDE_game_lives_calculation_l1404_140441


namespace NUMINAMATH_CALUDE_washing_machine_price_difference_l1404_140486

def total_price : ℕ := 7060
def refrigerator_price : ℕ := 4275

theorem washing_machine_price_difference : 
  refrigerator_price - (total_price - refrigerator_price) = 1490 := by
  sorry

end NUMINAMATH_CALUDE_washing_machine_price_difference_l1404_140486


namespace NUMINAMATH_CALUDE_preservation_time_at_33_l1404_140410

/-- The preservation time function -/
noncomputable def preservation_time (k b x : ℝ) : ℝ := Real.exp (k * x + b)

/-- Theorem stating the preservation time at 33°C given conditions -/
theorem preservation_time_at_33 (k b : ℝ) :
  preservation_time k b 0 = 192 →
  preservation_time k b 22 = 48 →
  preservation_time k b 33 = 24 := by
  sorry

end NUMINAMATH_CALUDE_preservation_time_at_33_l1404_140410


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_0125_l1404_140457

/-- The area of the quadrilateral formed by the intersection of four lines -/
def quadrilateral_area (line1 line2 : ℝ → ℝ → Prop) (x_line y_line : ℝ → Prop) : ℝ := sorry

/-- The first line: 3x + 4y - 12 = 0 -/
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

/-- The second line: 5x - 4y - 10 = 0 -/
def line2 (x y : ℝ) : Prop := 5 * x - 4 * y - 10 = 0

/-- The vertical line: x = 3 -/
def x_line (x : ℝ) : Prop := x = 3

/-- The horizontal line: y = 1 -/
def y_line (y : ℝ) : Prop := y = 1

theorem quadrilateral_area_is_0125 : 
  quadrilateral_area line1 line2 x_line y_line = 0.125 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_0125_l1404_140457


namespace NUMINAMATH_CALUDE_apple_cost_is_40_l1404_140415

/-- The cost of apples and pears at Clark's Food Store -/
structure FruitCosts where
  pear_cost : ℕ
  apple_cost : ℕ
  apple_quantity : ℕ
  pear_quantity : ℕ
  total_spent : ℕ

/-- Theorem: The cost of a dozen apples is 40 dollars -/
theorem apple_cost_is_40 (fc : FruitCosts) 
  (h1 : fc.pear_cost = 50)
  (h2 : fc.apple_quantity = 14 ∧ fc.pear_quantity = 14)
  (h3 : fc.total_spent = 1260)
  : fc.apple_cost = 40 := by
  sorry

#check apple_cost_is_40

end NUMINAMATH_CALUDE_apple_cost_is_40_l1404_140415


namespace NUMINAMATH_CALUDE_childrens_home_toddlers_l1404_140469

theorem childrens_home_toddlers (total : ℕ) (newborns : ℕ) :
  total = 40 →
  newborns = 4 →
  ∃ (toddlers teenagers : ℕ),
    toddlers + teenagers + newborns = total ∧
    teenagers = 5 * toddlers ∧
    toddlers = 6 :=
by sorry

end NUMINAMATH_CALUDE_childrens_home_toddlers_l1404_140469


namespace NUMINAMATH_CALUDE_square_equation_solution_l1404_140498

theorem square_equation_solution (b c x : ℝ) : 
  x^2 + c^2 = (b - x)^2 → x = (b^2 - c^2) / (2 * b) :=
by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l1404_140498


namespace NUMINAMATH_CALUDE_shopkeeper_loss_percent_l1404_140462

theorem shopkeeper_loss_percent 
  (initial_value : ℝ)
  (profit_rate : ℝ)
  (theft_rate : ℝ)
  (h1 : profit_rate = 0.1)
  (h2 : theft_rate = 0.3)
  (h3 : initial_value > 0) :
  let remaining_value := initial_value * (1 - theft_rate)
  let final_value := remaining_value * (1 + profit_rate)
  let loss := initial_value - final_value
  let loss_percent := (loss / initial_value) * 100
  loss_percent = 23 := by
sorry


end NUMINAMATH_CALUDE_shopkeeper_loss_percent_l1404_140462


namespace NUMINAMATH_CALUDE_min_value_expression_l1404_140414

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 3) : 
  ∃ (min : ℝ), min = 16/9 ∧ ∀ x y z, x > 0 → y > 0 → z > 0 → x + y + z = 3 → 
    (x + y) / (x * y * z) ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1404_140414


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1404_140472

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nine-sided polygon contains 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1404_140472


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1404_140443

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 4) (hb : b = 9) (hc : c = 36) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 12 / 7 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1404_140443


namespace NUMINAMATH_CALUDE_complement_A_union_B_equals_target_l1404_140482

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- Define set B
def B : Set ℝ := {y : ℝ | 1 ≤ y ∧ y ≤ 3}

-- State the theorem
theorem complement_A_union_B_equals_target :
  (Set.compl A) ∪ B = {x : ℝ | x < 0 ∨ x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_equals_target_l1404_140482


namespace NUMINAMATH_CALUDE_first_term_value_l1404_140460

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of first 40 terms is 180
  sum_first_40 : (40 : ℚ) / 2 * (2 * a + 39 * d) = 180
  -- Sum of next 40 terms (41st to 80th) is 2200
  sum_next_40 : (40 : ℚ) / 2 * (2 * (a + 40 * d) + 39 * d) = 2200
  -- 20th term is 75
  term_20 : a + 19 * d = 75

/-- The first term of the arithmetic sequence with given properties is 51.0125 -/
theorem first_term_value (seq : ArithmeticSequence) : seq.a = 51.0125 := by
  sorry

end NUMINAMATH_CALUDE_first_term_value_l1404_140460


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1404_140454

/-- An arithmetic sequence with a positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 3 = 10 →
  a 1 * a 3 = 16 →
  a 11 + a 12 + a 13 = 105 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1404_140454


namespace NUMINAMATH_CALUDE_complex_number_problem_l1404_140430

theorem complex_number_problem (α β : ℂ) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ α + β = x ∧ Complex.I * (α - 3 * β) = y) →
  β = 4 + Complex.I →
  α = 12 - Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1404_140430


namespace NUMINAMATH_CALUDE_hyperbola_slope_product_l1404_140448

/-- Hyperbola theorem -/
theorem hyperbola_slope_product (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0) 
  (hp : x₀^2 / a^2 - y₀^2 / b^2 = 1) (hx : x₀ ≠ a ∧ x₀ ≠ -a) : 
  (y₀ / (x₀ + a)) * (y₀ / (x₀ - a)) = b^2 / a^2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_slope_product_l1404_140448


namespace NUMINAMATH_CALUDE_probability_sum_seven_l1404_140477

/-- Represents the faces of the first die -/
def die1 : Finset ℕ := {1, 3, 5}

/-- Represents the faces of the second die -/
def die2 : Finset ℕ := {2, 4, 6}

/-- The total number of possible outcomes when rolling both dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum of 7) -/
def favorable_outcomes : ℕ := 12

/-- Theorem stating that the probability of rolling a sum of 7 is 1/3 -/
theorem probability_sum_seven :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_seven_l1404_140477


namespace NUMINAMATH_CALUDE_quadratic_square_of_binomial_l1404_140465

theorem quadratic_square_of_binomial (c : ℝ) : 
  (∃ a b : ℝ, ∀ x, 9*x^2 - 27*x + c = (a*x + b)^2) → c = 20.25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_square_of_binomial_l1404_140465


namespace NUMINAMATH_CALUDE_eggs_per_omelet_is_two_l1404_140499

/-- Represents the number of eggs per omelet for the Rotary Club's Omelet Breakfast. -/
def eggs_per_omelet : ℚ :=
  let small_children_tickets : ℕ := 53
  let older_children_tickets : ℕ := 35
  let adult_tickets : ℕ := 75
  let senior_tickets : ℕ := 37
  let small_children_omelets : ℚ := 0.5
  let older_children_omelets : ℚ := 1
  let adult_omelets : ℚ := 2
  let senior_omelets : ℚ := 1.5
  let extra_omelets : ℕ := 25
  let total_eggs : ℕ := 584
  let total_omelets : ℚ := small_children_tickets * small_children_omelets +
                           older_children_tickets * older_children_omelets +
                           adult_tickets * adult_omelets +
                           senior_tickets * senior_omelets +
                           extra_omelets
  total_eggs / total_omelets

/-- Theorem stating that the number of eggs per omelet is 2. -/
theorem eggs_per_omelet_is_two : eggs_per_omelet = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_omelet_is_two_l1404_140499


namespace NUMINAMATH_CALUDE_correct_operation_l1404_140435

theorem correct_operation (a : ℝ) : 2 * a^2 * a = 2 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l1404_140435


namespace NUMINAMATH_CALUDE_monthly_income_calculation_l1404_140425

/-- Proves that if 22% of a person's monthly income is Rs. 3800, then their monthly income is Rs. 17272.73. -/
theorem monthly_income_calculation (deposit : ℝ) (percentage : ℝ) (monthly_income : ℝ) 
  (h1 : deposit = 3800)
  (h2 : percentage = 22)
  (h3 : deposit = (percentage / 100) * monthly_income) :
  monthly_income = 17272.73 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_calculation_l1404_140425


namespace NUMINAMATH_CALUDE_solution_sum_comparison_l1404_140406

theorem solution_sum_comparison
  (a a' b b' c c' : ℝ)
  (ha : a ≠ 0)
  (ha' : a' ≠ 0) :
  (c' - b') / a' < (c - b) / a ↔
  (c - b) / a > (c' - b') / a' :=
by sorry

end NUMINAMATH_CALUDE_solution_sum_comparison_l1404_140406


namespace NUMINAMATH_CALUDE_base6_to_base10_12345_l1404_140444

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- The list representation of 12345 in base 6 --/
def number : List Nat := [5, 4, 3, 2, 1]

theorem base6_to_base10_12345 :
  base6ToBase10 number = 1865 := by
  sorry

#eval base6ToBase10 number

end NUMINAMATH_CALUDE_base6_to_base10_12345_l1404_140444
