import Mathlib

namespace NUMINAMATH_CALUDE_repeating_fraction_sixteen_equals_five_thirty_ninths_l204_20482

/-- Given a positive integer k, this function returns the value of the infinite geometric series
    4/k + 5/k^2 + 4/k^3 + 5/k^4 + ... -/
def repeating_fraction (k : ℕ) : ℚ :=
  (4 * k + 5) / (k^2 - 1)

/-- The theorem states that for k = 16, the repeating fraction equals 5/39 -/
theorem repeating_fraction_sixteen_equals_five_thirty_ninths :
  repeating_fraction 16 = 5 / 39 := by
  sorry

end NUMINAMATH_CALUDE_repeating_fraction_sixteen_equals_five_thirty_ninths_l204_20482


namespace NUMINAMATH_CALUDE_positive_function_from_condition_l204_20409

theorem positive_function_from_condition (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h' : ∀ x : ℝ, f x + x * deriv f x > 0) : 
  ∀ x : ℝ, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_function_from_condition_l204_20409


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l204_20431

/-- Given a line with equation y + 3 = -3(x - 5), prove that the sum of its x-intercept and y-intercept is 16 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y + 3 = -3 * (x - 5)) → 
  ∃ (x_int y_int : ℝ), 
    (y_int + 3 = -3 * (x_int - 5)) ∧ 
    (0 + 3 = -3 * (x_int - 5)) ∧ 
    (y_int + 3 = -3 * (0 - 5)) ∧ 
    (x_int + y_int = 16) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l204_20431


namespace NUMINAMATH_CALUDE_inequality_implies_a_range_l204_20477

theorem inequality_implies_a_range :
  (∀ x : ℝ, (3 : ℝ)^(x^2 - 2*a*x) > (1/3 : ℝ)^(x + 1)) →
  -1/2 < a ∧ a < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_a_range_l204_20477


namespace NUMINAMATH_CALUDE_scientific_notation_of_169200000000_l204_20487

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The number we want to convert to scientific notation -/
def number : ℝ := 169200000000

/-- Theorem stating that the scientific notation of 169200000000 is 1.692 × 10^11 -/
theorem scientific_notation_of_169200000000 :
  toScientificNotation number = ScientificNotation.mk 1.692 11 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_169200000000_l204_20487


namespace NUMINAMATH_CALUDE_domain_of_log2_l204_20496

-- Define the logarithm function with base 2
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Theorem stating that the domain of log₂x is the set of all positive real numbers
theorem domain_of_log2 :
  {x : ℝ | ∃ y, log2 x = y} = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_domain_of_log2_l204_20496


namespace NUMINAMATH_CALUDE_intersection_theorem_l204_20461

/-- The line x + y = k intersects the circle x^2 + y^2 = 4 at points A and B. -/
def intersectionPoints (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  (A.1 + A.2 = k) ∧ (B.1 + B.2 = k) ∧
  (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4)

/-- The length of AB equals the length of OA + OB, where O is the origin. -/
def lengthCondition (A B : ℝ × ℝ) : Prop :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 + B.1)^2 + (A.2 + B.2)^2

/-- Main theorem: If the conditions are satisfied, then k = 2. -/
theorem intersection_theorem (k : ℝ) (A B : ℝ × ℝ) 
  (h1 : k > 0)
  (h2 : intersectionPoints k A B)
  (h3 : lengthCondition A B) : 
  k = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l204_20461


namespace NUMINAMATH_CALUDE_trapezoid_construction_l204_20405

/-- Represents a trapezoid with sides a, b, c where a ∥ c -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  a_parallel_c : True  -- Represents the condition a ∥ c

/-- The condition that angle γ is twice as large as angle α -/
def angle_condition (t : Trapezoid) : Prop :=
  ∃ (α : ℝ), t.b * Real.sin (2 * α) = t.a - t.c

theorem trapezoid_construction (t : Trapezoid) 
  (h : angle_condition t) : 
  (t.b ≠ t.a - t.c → False) ∧
  (t.b = t.a - t.c → ∀ (ε : ℝ), ∃ (t' : Trapezoid), 
    t'.a = t.a ∧ t'.b = t.b ∧ t'.c = t.c ∧ 
    angle_condition t' ∧ t' ≠ t) :=
sorry

end NUMINAMATH_CALUDE_trapezoid_construction_l204_20405


namespace NUMINAMATH_CALUDE_problem_statement_l204_20419

theorem problem_statement (x y z w : ℝ) 
  (eq1 : 2^x + y = 7)
  (eq2 : 2^8 = y + x)
  (eq3 : z = Real.sin (x - y))
  (eq4 : w = 3 * (y + z)) :
  ∃ (result : ℝ), (x + y + z + w) / 4 = result := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l204_20419


namespace NUMINAMATH_CALUDE_base_b_number_not_divisible_by_four_l204_20488

theorem base_b_number_not_divisible_by_four (b : ℕ) : b ∈ ({4, 5, 6, 7, 8} : Finset ℕ) →
  (b^3 + b^2 - b + 2) % 4 ≠ 0 ↔ b ∈ ({4, 5, 7, 8} : Finset ℕ) := by
  sorry

end NUMINAMATH_CALUDE_base_b_number_not_divisible_by_four_l204_20488


namespace NUMINAMATH_CALUDE_saree_price_problem_l204_20446

theorem saree_price_problem (P : ℝ) : 
  P * (1 - 0.1) * (1 - 0.05) = 171 → P = 200 := by
  sorry

end NUMINAMATH_CALUDE_saree_price_problem_l204_20446


namespace NUMINAMATH_CALUDE_pentagon_side_length_l204_20429

/-- Given an equilateral triangle with side length 9/20 cm, prove that a regular pentagon with the same perimeter has side length 27/100 cm. -/
theorem pentagon_side_length (triangle_side : ℝ) (pentagon_side : ℝ) : 
  triangle_side = 9/20 → 
  3 * triangle_side = 5 * pentagon_side → 
  pentagon_side = 27/100 := by sorry

end NUMINAMATH_CALUDE_pentagon_side_length_l204_20429


namespace NUMINAMATH_CALUDE_equation_solution_l204_20485

theorem equation_solution (x : ℝ) : 
  x^2 + 3*x + 2 ≠ 0 →
  (-x^2 = (4*x + 2) / (x^2 + 3*x + 2)) ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l204_20485


namespace NUMINAMATH_CALUDE_equal_perimeter_interior_tiles_l204_20425

/-- Represents a rectangular room with dimensions m × n -/
structure Room where
  m : ℕ
  n : ℕ
  h : m ≤ n

/-- The number of tiles on the perimeter of the room -/
def perimeterTiles (r : Room) : ℕ := 2 * r.m + 2 * r.n - 4

/-- The number of tiles in the interior of the room -/
def interiorTiles (r : Room) : ℕ := r.m * r.n - perimeterTiles r

/-- Predicate to check if a room has equal number of perimeter and interior tiles -/
def hasEqualTiles (r : Room) : Prop := perimeterTiles r = interiorTiles r

/-- The theorem stating that (5,12) and (6,8) are the only solutions -/
theorem equal_perimeter_interior_tiles :
  ∀ r : Room, hasEqualTiles r ↔ (r.m = 5 ∧ r.n = 12) ∨ (r.m = 6 ∧ r.n = 8) := by sorry

end NUMINAMATH_CALUDE_equal_perimeter_interior_tiles_l204_20425


namespace NUMINAMATH_CALUDE_line_segment_length_l204_20497

/-- The length of a line segment with endpoints (1,2) and (4,10) is √73. -/
theorem line_segment_length : Real.sqrt ((4 - 1)^2 + (10 - 2)^2) = Real.sqrt 73 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_length_l204_20497


namespace NUMINAMATH_CALUDE_investment_plans_count_l204_20449

/-- The number of ways to distribute 3 distinct projects among 5 cities, 
    with no more than 2 projects per city -/
def investmentPlans : ℕ := 120

/-- The number of candidate cities -/
def numCities : ℕ := 5

/-- The number of projects to be distributed -/
def numProjects : ℕ := 3

/-- The maximum number of projects allowed in a single city -/
def maxProjectsPerCity : ℕ := 2

theorem investment_plans_count :
  investmentPlans = 
    (numCities.choose numProjects) + 
    (numProjects.choose 2) * numCities * (numCities - 1) := by
  sorry

end NUMINAMATH_CALUDE_investment_plans_count_l204_20449


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l204_20418

/-- The continued fraction equation representing the given expression -/
def continued_fraction_equation (x : ℝ) : Prop :=
  x = 3 + 5 / (2 + 5 / x)

/-- The theorem stating that 5 is the solution to the continued fraction equation -/
theorem continued_fraction_solution :
  ∃ (x : ℝ), continued_fraction_equation x ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l204_20418


namespace NUMINAMATH_CALUDE_trailing_zeros_of_product_factorials_mod_100_l204_20441

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def product_of_factorials (n : ℕ) : ℕ := (List.range n).foldl (fun acc i => acc * factorial (i + 1)) 1

def trailing_zeros (n : ℕ) : ℕ := 
  if n = 0 then 0 else (n.digits 10).reverse.takeWhile (· = 0) |>.length

theorem trailing_zeros_of_product_factorials_mod_100 :
  trailing_zeros (product_of_factorials 50) % 100 = 12 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_product_factorials_mod_100_l204_20441


namespace NUMINAMATH_CALUDE_max_rectangle_area_l204_20478

theorem max_rectangle_area (perimeter : ℕ) (h_perimeter : perimeter = 156) :
  ∃ (length width : ℕ),
    2 * (length + width) = perimeter ∧
    ∀ (l w : ℕ), 2 * (l + w) = perimeter → l * w ≤ length * width ∧
    length * width = 1521 := by
  sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l204_20478


namespace NUMINAMATH_CALUDE_range_of_x_l204_20447

theorem range_of_x (x : ℝ) : 
  (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → |3*a + b| + |a - b| ≥ |a| * (|x - 1| + |x + 1|)) 
  ↔ x ∈ Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_l204_20447


namespace NUMINAMATH_CALUDE_connie_marbles_to_juan_l204_20401

/-- Represents the number of marbles Connie gave to Juan -/
def marbles_given_to_juan (initial_marbles : ℕ) (remaining_marbles : ℕ) : ℕ :=
  initial_marbles - remaining_marbles

/-- Proves that Connie gave 73 marbles to Juan -/
theorem connie_marbles_to_juan :
  marbles_given_to_juan 143 70 = 73 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_to_juan_l204_20401


namespace NUMINAMATH_CALUDE_factor_expression_l204_20450

theorem factor_expression (a b c : ℝ) : 
  ((a^2 - b^2)^4 + (b^2 - c^2)^4 + (c^2 - a^2)^4) / ((a - b)^4 + (b - c)^4 + (c - a)^4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l204_20450


namespace NUMINAMATH_CALUDE_largest_number_l204_20484

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 99 / 100)
  (hb : b = 9099 / 10000)
  (hc : c = 9 / 10)
  (hd : d = 909 / 1000)
  (he : e = 9009 / 10000) :
  a > b ∧ a > c ∧ a > d ∧ a > e :=
sorry

end NUMINAMATH_CALUDE_largest_number_l204_20484


namespace NUMINAMATH_CALUDE_range_of_abc_l204_20460

theorem range_of_abc (a b c : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) (h4 : 2 < c) (h5 : c < 3) :
  ∀ x, (∃ a' b' c', -1 < a' ∧ a' < b' ∧ b' < 1 ∧ 2 < c' ∧ c' < 3 ∧ x = (a' - b') * c') → -6 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_abc_l204_20460


namespace NUMINAMATH_CALUDE_problem_statement_l204_20490

theorem problem_statement (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = 3) :
  3 * a^2 * b + 3 * a * b^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l204_20490


namespace NUMINAMATH_CALUDE_quadratic_radical_range_l204_20445

theorem quadratic_radical_range : 
  {x : ℝ | ∃ y : ℝ, y^2 = 3*x - 1} = {x : ℝ | x ≥ 1/3} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_range_l204_20445


namespace NUMINAMATH_CALUDE_raspberry_ratio_l204_20459

theorem raspberry_ratio (total_berries : ℕ) (blackberries : ℕ) (blueberries : ℕ) :
  total_berries = 42 →
  blackberries = total_berries / 3 →
  blueberries = 7 →
  (total_berries - blackberries - blueberries) * 2 = total_berries := by
  sorry

end NUMINAMATH_CALUDE_raspberry_ratio_l204_20459


namespace NUMINAMATH_CALUDE_triangle_side_length_l204_20448

/-- Prove that in a triangle ABC where angles A, B, C form an arithmetic sequence,
    if A = 75° and b = √3, then a = (√6 + √2) / 2. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  -- Angles form an arithmetic sequence
  (B - A = C - B) → 
  -- A = 75°
  (A = 75 * π / 180) →
  -- b = √3
  (b = Real.sqrt 3) →
  -- Triangle inequality
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  -- Sum of angles in a triangle is π
  (A + B + C = π) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) →
  -- Conclusion: a = (√6 + √2) / 2
  a = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
    sorry

end NUMINAMATH_CALUDE_triangle_side_length_l204_20448


namespace NUMINAMATH_CALUDE_find_other_number_l204_20493

theorem find_other_number (a b : ℕ+) (h1 : Nat.lcm a b = 5040) (h2 : Nat.gcd a b = 12) (h3 : a = 240) : b = 252 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l204_20493


namespace NUMINAMATH_CALUDE_repetend_5_17_l204_20483

def repetend_of_5_17 : List Nat := [2, 9, 4, 1, 1, 7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5, 2, 9]

theorem repetend_5_17 :
  ∃ (k : ℕ), (5 : ℚ) / 17 = (k : ℚ) / 10^18 + 
  (List.sum (List.zipWith (λ (d i : ℕ) => (d : ℚ) / 10^(i+1)) repetend_of_5_17 (List.range 18))) *
  (1 / (1 - 1 / 10^18)) :=
by
  sorry

end NUMINAMATH_CALUDE_repetend_5_17_l204_20483


namespace NUMINAMATH_CALUDE_equation_solution_l204_20427

theorem equation_solution : ∃ x : ℚ, (1 / 3 + 1 / x = 7 / 9 + 1) ∧ (x = 9 / 13) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l204_20427


namespace NUMINAMATH_CALUDE_line_slope_range_l204_20476

/-- The range of m for a line x - my + √3m = 0 with a point M satisfying certain conditions -/
theorem line_slope_range (m : ℝ) : 
  (∃ (x y : ℝ), x - m * y + Real.sqrt 3 * m = 0 ∧ 
    y^2 = 3 * x^2 - 3) →
  (m ≤ -Real.sqrt 6 / 6 ∨ m ≥ Real.sqrt 6 / 6) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_range_l204_20476


namespace NUMINAMATH_CALUDE_value_calculation_l204_20438

theorem value_calculation (number : ℕ) (value : ℕ) 
  (h1 : value = 5 * number) 
  (h2 : number = 20) : 
  value = 100 := by
sorry

end NUMINAMATH_CALUDE_value_calculation_l204_20438


namespace NUMINAMATH_CALUDE_opposite_numbers_solution_l204_20444

theorem opposite_numbers_solution (x : ℝ) : 2 * (x - 3) = -(4 * (1 - x)) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_solution_l204_20444


namespace NUMINAMATH_CALUDE_box_width_proof_l204_20481

theorem box_width_proof (length width height : ℕ) (cubes : ℕ) : 
  length = 15 → height = 13 → cubes = 3120 → cubes = length * width * height → width = 16 := by
  sorry

end NUMINAMATH_CALUDE_box_width_proof_l204_20481


namespace NUMINAMATH_CALUDE_swan_population_l204_20404

/-- The number of swans doubles every 2 years -/
def doubles_every_two_years (S : ℕ → ℕ) : Prop :=
  ∀ n, S (n + 2) = 2 * S n

/-- In 10 years, there will be 480 swans -/
def swans_in_ten_years (S : ℕ → ℕ) : Prop :=
  S 10 = 480

/-- The current number of swans -/
def current_swans : ℕ := 15

theorem swan_population (S : ℕ → ℕ) 
  (h1 : doubles_every_two_years S) 
  (h2 : swans_in_ten_years S) : 
  S 0 = current_swans := by
  sorry

end NUMINAMATH_CALUDE_swan_population_l204_20404


namespace NUMINAMATH_CALUDE_square_difference_equality_l204_20479

theorem square_difference_equality : (19 + 15)^2 - (19 - 15)^2 = 1140 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l204_20479


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l204_20494

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 12 m and height 6 m is 72 sq m -/
theorem parallelogram_area_example : parallelogram_area 12 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l204_20494


namespace NUMINAMATH_CALUDE_hcd_problem_l204_20434

theorem hcd_problem : (Nat.gcd 12348 2448 * 3) - 14 = 94 := by
  sorry

end NUMINAMATH_CALUDE_hcd_problem_l204_20434


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_two_l204_20406

-- Define the curve
def curve (x : ℝ) : ℝ := 2 * x^2

-- Define the slope of the tangent line at a point
def tangent_slope (x : ℝ) : ℝ := 4 * x

-- Theorem statement
theorem tangent_slope_at_point_two :
  tangent_slope 2 = 8 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_two_l204_20406


namespace NUMINAMATH_CALUDE_pencil_cost_l204_20468

theorem pencil_cost (x y : ℚ) 
  (eq1 : 4 * x + 3 * y = 224)
  (eq2 : 2 * x + 5 * y = 154) : 
  y = 12 := by
sorry

end NUMINAMATH_CALUDE_pencil_cost_l204_20468


namespace NUMINAMATH_CALUDE_lifesaving_test_percentage_l204_20408

/-- The percentage of swim club members who have passed the lifesaving test -/
def percentage_passed : ℝ := 30

theorem lifesaving_test_percentage :
  let total_members : ℕ := 60
  let not_passed_with_course : ℕ := 12
  let not_passed_without_course : ℕ := 30
  percentage_passed = 30 ∧
  percentage_passed = (total_members - (not_passed_with_course + not_passed_without_course)) / total_members * 100 :=
by sorry

end NUMINAMATH_CALUDE_lifesaving_test_percentage_l204_20408


namespace NUMINAMATH_CALUDE_total_hamburger_combinations_l204_20432

/-- The number of available condiments -/
def num_condiments : ℕ := 10

/-- The number of choices for meat patties -/
def patty_choices : ℕ := 4

/-- Theorem stating the total number of hamburger combinations -/
theorem total_hamburger_combinations :
  2^num_condiments * patty_choices = 4096 := by
  sorry

end NUMINAMATH_CALUDE_total_hamburger_combinations_l204_20432


namespace NUMINAMATH_CALUDE_graph_vertical_shift_l204_20471

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the vertical shift operation
def verticalShift (g : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := fun x ↦ g x - c

-- Theorem statement
theorem graph_vertical_shift (x : ℝ) : 
  (verticalShift f 2) x = f x - 2 := by sorry

end NUMINAMATH_CALUDE_graph_vertical_shift_l204_20471


namespace NUMINAMATH_CALUDE_initial_cows_count_l204_20457

theorem initial_cows_count (initial_pigs : ℕ) (initial_goats : ℕ) 
  (added_cows : ℕ) (added_pigs : ℕ) (added_goats : ℕ) (total_after : ℕ) :
  initial_pigs = 3 →
  initial_goats = 6 →
  added_cows = 3 →
  added_pigs = 5 →
  added_goats = 2 →
  total_after = 21 →
  ∃ initial_cows : ℕ, initial_cows = 2 ∧ 
    initial_cows + initial_pigs + initial_goats + added_cows + added_pigs + added_goats = total_after :=
by sorry

end NUMINAMATH_CALUDE_initial_cows_count_l204_20457


namespace NUMINAMATH_CALUDE_library_shelves_l204_20499

theorem library_shelves (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 14240) (h2 : books_per_shelf = 8) :
  total_books / books_per_shelf = 1780 := by
  sorry

end NUMINAMATH_CALUDE_library_shelves_l204_20499


namespace NUMINAMATH_CALUDE_jays_change_is_twenty_l204_20439

/-- The change Jay received after purchasing items and paying with a fifty-dollar bill -/
def jays_change (book_price pen_price ruler_price paid_amount : ℕ) : ℕ :=
  paid_amount - (book_price + pen_price + ruler_price)

/-- Theorem stating that Jay's change is $20 given the specific prices and payment amount -/
theorem jays_change_is_twenty :
  jays_change 25 4 1 50 = 20 := by
  sorry

end NUMINAMATH_CALUDE_jays_change_is_twenty_l204_20439


namespace NUMINAMATH_CALUDE_sum_of_squares_l204_20472

theorem sum_of_squares (a b c : ℝ) (h1 : a * b + b * c + a * c = 72) (h2 : a + b + c = 14) :
  a^2 + b^2 + c^2 = 52 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l204_20472


namespace NUMINAMATH_CALUDE_harry_pencils_left_l204_20420

/-- Calculates the number of pencils left with Harry given the initial conditions. -/
def pencils_left_with_harry (anna_pencils : ℕ) (harry_lost : ℕ) : ℕ :=
  2 * anna_pencils - harry_lost

/-- Proves that Harry has 81 pencils left given the initial conditions. -/
theorem harry_pencils_left :
  pencils_left_with_harry 50 19 = 81 := by
  sorry

#eval pencils_left_with_harry 50 19

end NUMINAMATH_CALUDE_harry_pencils_left_l204_20420


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l204_20453

theorem perpendicular_vectors (m : ℚ) : 
  let a : ℚ × ℚ := (-2, m)
  let b : ℚ × ℚ := (-1, 3)
  (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 = 0 → m = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l204_20453


namespace NUMINAMATH_CALUDE_initial_books_borrowed_l204_20467

/-- Represents the number of books Mary has at each stage --/
def books_count (initial : ℕ) : ℕ → ℕ
| 0 => initial  -- Initial number of books
| 1 => initial - 3 + 5  -- After first library visit
| 2 => initial - 3 + 5 - 2 + 7  -- After second library visit
| _ => 0  -- We don't need values beyond stage 2

/-- The theorem stating the initial number of books Mary borrowed --/
theorem initial_books_borrowed :
  ∃ (initial : ℕ), books_count initial 2 = 12 ∧ initial = 5 := by
  sorry


end NUMINAMATH_CALUDE_initial_books_borrowed_l204_20467


namespace NUMINAMATH_CALUDE_trip_time_difference_l204_20430

theorem trip_time_difference (distance1 distance2 speed : ℝ) 
  (h1 : distance1 = 240)
  (h2 : distance2 = 420)
  (h3 : speed = 60) :
  distance2 / speed - distance1 / speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_difference_l204_20430


namespace NUMINAMATH_CALUDE_tracy_candies_l204_20410

theorem tracy_candies (initial_candies : ℕ) : 
  (∃ (sister_took : ℕ),
    initial_candies > 0 ∧
    sister_took ≥ 2 ∧ 
    sister_took ≤ 6 ∧
    (initial_candies * 3 / 4) * 2 / 3 - 40 - sister_took = 10) →
  initial_candies = 108 :=
by sorry

end NUMINAMATH_CALUDE_tracy_candies_l204_20410


namespace NUMINAMATH_CALUDE_product_modulo_300_l204_20462

theorem product_modulo_300 : (2025 * 1233) % 300 = 75 := by
  sorry

end NUMINAMATH_CALUDE_product_modulo_300_l204_20462


namespace NUMINAMATH_CALUDE_cloth_profit_proof_l204_20403

def cloth_problem (selling_price total_meters cost_price_per_meter : ℕ) : Prop :=
  let total_cost := total_meters * cost_price_per_meter
  let total_profit := selling_price - total_cost
  let profit_per_meter := total_profit / total_meters
  profit_per_meter = 5

theorem cloth_profit_proof :
  cloth_problem 8925 85 100 := by
  sorry

end NUMINAMATH_CALUDE_cloth_profit_proof_l204_20403


namespace NUMINAMATH_CALUDE_triangle_problem_l204_20400

theorem triangle_problem (A B C : Real) (a b c : Real) :
  a + c = 5 →
  a > c →
  b = 3 →
  Real.cos B = 1/3 →
  a = 3 ∧ c = 2 ∧ Real.cos (A + B) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l204_20400


namespace NUMINAMATH_CALUDE_projection_a_on_b_is_sqrt_5_l204_20451

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-2, 4]

theorem projection_a_on_b_is_sqrt_5 :
  let dot_product := (a 0) * (b 0) + (a 1) * (b 1)
  let magnitude_b := Real.sqrt ((b 0)^2 + (b 1)^2)
  dot_product / magnitude_b = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_projection_a_on_b_is_sqrt_5_l204_20451


namespace NUMINAMATH_CALUDE_corner_sum_is_200_l204_20455

/-- Represents a 9x9 grid filled with numbers from 10 to 90 --/
def Grid := Fin 9 → Fin 9 → ℕ

/-- The grid is filled sequentially from 10 to 90 --/
def sequential_fill (g : Grid) : Prop :=
  ∀ i j, g i j = i.val * 9 + j.val + 10

/-- The sum of the numbers in the four corners of the grid --/
def corner_sum (g : Grid) : ℕ :=
  g 0 0 + g 0 8 + g 8 0 + g 8 8

/-- Theorem stating that the sum of the numbers in the four corners is 200 --/
theorem corner_sum_is_200 (g : Grid) (h : sequential_fill g) : corner_sum g = 200 := by
  sorry

end NUMINAMATH_CALUDE_corner_sum_is_200_l204_20455


namespace NUMINAMATH_CALUDE_denominator_value_l204_20426

theorem denominator_value (x : ℝ) (h : (1 / x) ^ 1 = 0.25) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_denominator_value_l204_20426


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l204_20475

theorem fraction_sum_equals_decimal : 
  2/5 + 3/25 + 4/125 + 1/625 = 0.5536 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l204_20475


namespace NUMINAMATH_CALUDE_investment_gain_percentage_l204_20407

-- Define the initial investment
def initial_investment : ℝ := 100

-- Define the first year loss percentage
def first_year_loss_percent : ℝ := 10

-- Define the second year gain percentage
def second_year_gain_percent : ℝ := 25

-- Theorem to prove the overall gain percentage
theorem investment_gain_percentage :
  let first_year_amount := initial_investment * (1 - first_year_loss_percent / 100)
  let second_year_amount := first_year_amount * (1 + second_year_gain_percent / 100)
  let overall_gain_percent := (second_year_amount - initial_investment) / initial_investment * 100
  overall_gain_percent = 12.5 := by
sorry

end NUMINAMATH_CALUDE_investment_gain_percentage_l204_20407


namespace NUMINAMATH_CALUDE_find_A_l204_20421

theorem find_A : ∃ A : ℤ, A + 10 = 15 ∧ A = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l204_20421


namespace NUMINAMATH_CALUDE_div_chain_equals_four_l204_20452

theorem div_chain_equals_four : (((120 / 5) / 3) / 2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_div_chain_equals_four_l204_20452


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l204_20458

theorem quadratic_roots_property (a : ℝ) (x₁ x₂ : ℝ) : 
  (x₁ ≠ x₂) →
  (x₁^2 + a*x₁ + 2 = 0) →
  (x₂^2 + a*x₂ + 2 = 0) →
  (x₁^3 + 14/x₂^2 = x₂^3 + 14/x₁^2) →
  (a = 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l204_20458


namespace NUMINAMATH_CALUDE_initial_books_count_l204_20443

/-- The number of people who borrowed books on the first day -/
def borrowers : ℕ := 5

/-- The number of books each person borrowed on the first day -/
def books_per_borrower : ℕ := 2

/-- The number of books borrowed on the second day -/
def second_day_borrowed : ℕ := 20

/-- The number of books remaining on the shelf after the second day -/
def remaining_books : ℕ := 70

/-- The initial number of books on the shelf -/
def initial_books : ℕ := borrowers * books_per_borrower + second_day_borrowed + remaining_books

theorem initial_books_count : initial_books = 100 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_count_l204_20443


namespace NUMINAMATH_CALUDE_marble_distribution_l204_20469

theorem marble_distribution (x : ℚ) 
  (total_marbles : ℕ) 
  (first_boy : ℚ → ℚ) 
  (second_boy : ℚ → ℚ) 
  (third_boy : ℚ → ℚ) 
  (h1 : first_boy x = 4 * x + 2)
  (h2 : second_boy x = 2 * x)
  (h3 : third_boy x = 3 * x - 1)
  (h4 : total_marbles = 47)
  (h5 : (first_boy x + second_boy x + third_boy x : ℚ) = total_marbles) :
  (first_boy x, second_boy x, third_boy x) = (202/9, 92/9, 129/9) := by
sorry

end NUMINAMATH_CALUDE_marble_distribution_l204_20469


namespace NUMINAMATH_CALUDE_game_points_l204_20492

/-- The number of points earned in a video game level --/
def points_earned (total_enemies : ℕ) (enemies_left : ℕ) (points_per_enemy : ℕ) : ℕ :=
  (total_enemies - enemies_left) * points_per_enemy

/-- Theorem: In a level with 8 enemies, destroying all but 6 of them, with 5 points per enemy, results in 10 points --/
theorem game_points : points_earned 8 6 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_game_points_l204_20492


namespace NUMINAMATH_CALUDE_negation_of_existence_inequality_l204_20440

theorem negation_of_existence_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_inequality_l204_20440


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_closest_integer_to_cube_root_of_sum_of_cubes_l204_20436

theorem closest_integer_to_cube_root (x : ℝ) : 
  ∃ n : ℤ, ∀ m : ℤ, |x - n| ≤ |x - m| := by sorry

theorem closest_integer_to_cube_root_of_sum_of_cubes : 
  ∃ n : ℤ, (∀ m : ℤ, |((7 : ℝ)^3 + 9^3)^(1/3) - n| ≤ |((7 : ℝ)^3 + 9^3)^(1/3) - m|) ∧ n = 10 := by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_closest_integer_to_cube_root_of_sum_of_cubes_l204_20436


namespace NUMINAMATH_CALUDE_triangle_max_value_l204_20435

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that if a² + b² = √3ab + c² and AB = 1, then the maximum value of AC + √3BC is 2√7 -/
theorem triangle_max_value (a b c : ℝ) (A B C : ℝ) :
  a^2 + b^2 = Real.sqrt 3 * a * b + c^2 →
  a = 1 →  -- AB = 1
  ∃ (AC BC : ℝ), AC + Real.sqrt 3 * BC ≤ 2 * Real.sqrt 7 ∧
    ∃ (AC' BC' : ℝ), AC' + Real.sqrt 3 * BC' = 2 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_value_l204_20435


namespace NUMINAMATH_CALUDE_sum_of_m_for_integer_solutions_l204_20414

theorem sum_of_m_for_integer_solutions : ∃ (S : Finset Int),
  (∀ m : Int, m ∈ S ↔ 
    (∃ x y : Int, x^2 - m*x + 15 = 0 ∧ y^2 - m*y + 15 = 0 ∧ x ≠ y)) ∧
  (S.sum id = 48) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_m_for_integer_solutions_l204_20414


namespace NUMINAMATH_CALUDE_inscribed_box_sphere_radius_l204_20491

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  s : ℝ  -- radius of the sphere
  a : ℝ  -- length of the box
  b : ℝ  -- width of the box
  c : ℝ  -- height of the box

/-- The sum of the lengths of the 12 edges of the box -/
def edgeSum (box : InscribedBox) : ℝ := 4 * (box.a + box.b + box.c)

/-- The surface area of the box -/
def surfaceArea (box : InscribedBox) : ℝ := 2 * (box.a * box.b + box.b * box.c + box.c * box.a)

/-- The theorem stating the relationship between the box dimensions and the sphere radius -/
theorem inscribed_box_sphere_radius (box : InscribedBox) 
    (h1 : edgeSum box = 160) 
    (h2 : surfaceArea box = 600) : 
    box.s = 5 * Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_inscribed_box_sphere_radius_l204_20491


namespace NUMINAMATH_CALUDE_h_odd_f_increasing_inequality_solution_l204_20498

noncomputable section

variable (f : ℝ → ℝ)
variable (h : ℝ → ℝ)

axiom f_property : ∀ x y : ℝ, f x + f y = f (x + y) + 1
axiom f_positive : ∀ x : ℝ, x > 0 → f x > 1
axiom h_def : ∀ x : ℝ, h x = f x - 1

theorem h_odd : ∀ x : ℝ, h (-x) = -h x := by sorry

theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ > x₂ → f x₁ > f x₂ := by sorry

theorem inequality_solution (t : ℝ) :
  (t = 1 → ¬∃ x, f (x^2) - f (3*t*x) + f (2*t^2 + 2*t - x) < 1) ∧
  (t > 1 → ∀ x, f (x^2) - f (3*t*x) + f (2*t^2 + 2*t - x) < 1 ↔ t+1 < x ∧ x < 2*t) ∧
  (t < 1 → ∀ x, f (x^2) - f (3*t*x) + f (2*t^2 + 2*t - x) < 1 ↔ 2*t < x ∧ x < t+1) := by sorry

end

end NUMINAMATH_CALUDE_h_odd_f_increasing_inequality_solution_l204_20498


namespace NUMINAMATH_CALUDE_power_function_through_point_l204_20424

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f (1/2) = 8) : 
  f 2 = 1/8 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l204_20424


namespace NUMINAMATH_CALUDE_money_market_investment_ratio_l204_20454

def initial_amount : ℚ := 25
def amount_to_mom : ℚ := 8
def num_items : ℕ := 5
def item_cost : ℚ := 1/2
def final_amount : ℚ := 6

theorem money_market_investment_ratio :
  let remaining_after_mom := initial_amount - amount_to_mom
  let spent_on_items := num_items * item_cost
  let before_investment := remaining_after_mom - spent_on_items
  let invested := before_investment - final_amount
  (invested : ℚ) / remaining_after_mom = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_money_market_investment_ratio_l204_20454


namespace NUMINAMATH_CALUDE_paper_area_problem_l204_20437

theorem paper_area_problem (L : ℝ) : 
  2 * (11 * L) = 2 * (8.5 * 11) + 100 ↔ L = 287 / 22 := by sorry

end NUMINAMATH_CALUDE_paper_area_problem_l204_20437


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l204_20473

/-- Given a geometric sequence {a_n} with first three terms a-1, a+1, a+2, 
    prove that its general formula is a_n = -1/(2^(n-3)) -/
theorem geometric_sequence_formula (a : ℝ) (a_n : ℕ → ℝ) :
  a_n 1 = a - 1 →
  a_n 2 = a + 1 →
  a_n 3 = a + 2 →
  (∀ n : ℕ, n ≥ 1 → a_n (n + 1) / a_n n = a_n 2 / a_n 1) →
  ∀ n : ℕ, n ≥ 1 → a_n n = -1 / (2^(n - 3)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l204_20473


namespace NUMINAMATH_CALUDE_base_seven_digits_of_4300_l204_20413

theorem base_seven_digits_of_4300 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 4300 ∧ 4300 < 7^n ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_digits_of_4300_l204_20413


namespace NUMINAMATH_CALUDE_tan_sum_equality_l204_20433

theorem tan_sum_equality (A B : ℝ) 
  (h1 : A + B = (5 / 4) * Real.pi)
  (h2 : ∀ k : ℤ, A ≠ k * Real.pi + Real.pi / 2)
  (h3 : ∀ k : ℤ, B ≠ k * Real.pi + Real.pi / 2) :
  (1 + Real.tan A) * (1 + Real.tan B) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_equality_l204_20433


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l204_20412

theorem largest_constant_inequality (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  ∃ m : ℝ, m = 2 ∧ 
  (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    Real.sqrt (a / (b + c + d + e)) + 
    Real.sqrt (b / (a + c + d + e)) + 
    Real.sqrt (c / (a + b + d + e)) + 
    Real.sqrt (d / (a + b + c + e)) > m) ∧
  (∀ m' : ℝ, m' > m → 
    ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
      Real.sqrt (a / (b + c + d + e)) + 
      Real.sqrt (b / (a + c + d + e)) + 
      Real.sqrt (c / (a + b + d + e)) + 
      Real.sqrt (d / (a + b + c + e)) ≤ m') :=
by sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l204_20412


namespace NUMINAMATH_CALUDE_complex_modulus_evaluation_l204_20474

theorem complex_modulus_evaluation :
  Complex.abs (3 - 5*I + (-2 + (3/4)*I)) = (Real.sqrt 305) / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_evaluation_l204_20474


namespace NUMINAMATH_CALUDE_special_right_triangle_legs_lengths_l204_20423

/-- A right triangle with a point on the hypotenuse equidistant from both legs -/
structure SpecialRightTriangle where
  /-- Length of the first segment of the divided hypotenuse -/
  segment1 : ℝ
  /-- Length of the second segment of the divided hypotenuse -/
  segment2 : ℝ
  /-- The point divides the hypotenuse into the given segments -/
  hypotenuse_division : segment1 + segment2 = 70
  /-- The segments are positive -/
  segment1_pos : segment1 > 0
  segment2_pos : segment2 > 0

/-- The lengths of the legs of the special right triangle -/
def legs_lengths (t : SpecialRightTriangle) : ℝ × ℝ :=
  (42, 56)

/-- Theorem stating that the legs of the special right triangle have lengths 42 and 56 -/
theorem special_right_triangle_legs_lengths (t : SpecialRightTriangle)
    (h1 : t.segment1 = 30) (h2 : t.segment2 = 40) :
    legs_lengths t = (42, 56) := by
  sorry

end NUMINAMATH_CALUDE_special_right_triangle_legs_lengths_l204_20423


namespace NUMINAMATH_CALUDE_base4_1010_equals_68_l204_20416

/-- Converts a base-4 digit to its decimal value -/
def base4ToDecimal (digit : Nat) : Nat :=
  match digit with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | _ => 0  -- Default case for invalid digits

/-- Converts a list of base-4 digits to a decimal number -/
def convertBase4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + base4ToDecimal d * (4 ^ i)) 0

theorem base4_1010_equals_68 : 
  convertBase4ToDecimal [0, 1, 0, 1] = 68 := by
  sorry

#eval convertBase4ToDecimal [0, 1, 0, 1]

end NUMINAMATH_CALUDE_base4_1010_equals_68_l204_20416


namespace NUMINAMATH_CALUDE_complex_power_six_l204_20486

theorem complex_power_six (i : ℂ) (h : i^2 = -1) : (1 + i)^6 = -8*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_six_l204_20486


namespace NUMINAMATH_CALUDE_tenth_row_sum_l204_20415

/-- The function representing the first term of the n-th row -/
def f (n : ℕ) : ℕ := 2 * n^2 - 3 * n + 3

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem tenth_row_sum :
  let first_term : ℕ := f 10
  let num_terms : ℕ := 2 * 10
  let common_diff : ℕ := 2
  arithmetic_sum first_term common_diff num_terms = 3840 := by
sorry

#eval arithmetic_sum (f 10) 2 (2 * 10)

end NUMINAMATH_CALUDE_tenth_row_sum_l204_20415


namespace NUMINAMATH_CALUDE_negation_of_existential_quadratic_inequality_l204_20463

theorem negation_of_existential_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x ≤ 1) ↔ (∀ x : ℝ, x^2 + 2*x > 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_quadratic_inequality_l204_20463


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_for_260_l204_20489

/-- Given a natural number n, returns the sum of exponents in its binary representation -/
def sumOfExponents (n : ℕ) : ℕ := sorry

/-- Checks if a natural number n can be expressed as a sum of at least three distinct powers of 2 -/
def hasAtLeastThreeDistinctPowers (n : ℕ) : Prop := sorry

theorem least_sum_of_exponents_for_260 :
  ∀ k : ℕ, (hasAtLeastThreeDistinctPowers 260 ∧ sumOfExponents 260 = k) → k ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_for_260_l204_20489


namespace NUMINAMATH_CALUDE_irrational_power_congruence_l204_20465

theorem irrational_power_congruence :
  ∀ (k : ℕ), k ≥ 2 →
  ∃ (r : ℝ), Irrational r ∧
    ∀ (m : ℕ), (⌊r^m⌋ : ℤ) ≡ -1 [ZMOD k] :=
sorry

end NUMINAMATH_CALUDE_irrational_power_congruence_l204_20465


namespace NUMINAMATH_CALUDE_rhinos_count_l204_20402

/-- The number of animals Erica saw during her safari --/
def total_animals : ℕ := 20

/-- The number of lions seen on Saturday --/
def lions : ℕ := 3

/-- The number of elephants seen on Saturday --/
def elephants : ℕ := 2

/-- The number of buffaloes seen on Sunday --/
def buffaloes : ℕ := 2

/-- The number of leopards seen on Sunday --/
def leopards : ℕ := 5

/-- The number of warthogs seen on Monday --/
def warthogs : ℕ := 3

/-- The number of rhinos seen on Monday --/
def rhinos : ℕ := total_animals - (lions + elephants + buffaloes + leopards + warthogs)

theorem rhinos_count : rhinos = 5 := by
  sorry

end NUMINAMATH_CALUDE_rhinos_count_l204_20402


namespace NUMINAMATH_CALUDE_smallest_valid_coloring_distance_l204_20480

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The set of points inside and on the edges of a regular hexagon with side length 1 -/
def S : Set Point := sorry

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- A 3-coloring of points -/
def Coloring := Point → Fin 3

/-- A valid coloring respecting the distance r -/
def valid_coloring (c : Coloring) (r : ℝ) : Prop :=
  ∀ p q : Point, p ∈ S → q ∈ S → c p = c q → distance p q < r

/-- The existence of a valid coloring -/
def exists_valid_coloring (r : ℝ) : Prop :=
  ∃ c : Coloring, valid_coloring c r

/-- The theorem stating that 3/2 is the smallest r for which a valid 3-coloring exists -/
theorem smallest_valid_coloring_distance :
  (∀ r < 3/2, ¬ exists_valid_coloring r) ∧ exists_valid_coloring (3/2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_coloring_distance_l204_20480


namespace NUMINAMATH_CALUDE_yard_length_26_trees_l204_20417

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 26 equally spaced trees,
    where the distance between consecutive trees is 15 meters, is 375 meters -/
theorem yard_length_26_trees :
  yard_length 26 15 = 375 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_26_trees_l204_20417


namespace NUMINAMATH_CALUDE_cinnamon_nutmeg_difference_l204_20411

theorem cinnamon_nutmeg_difference :
  let cinnamon : Float := 0.6666666666666666
  let nutmeg : Float := 0.5
  cinnamon - nutmeg = 0.1666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_cinnamon_nutmeg_difference_l204_20411


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l204_20466

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 25| + |x - 21| = |2*x - 46| + |x - 17| ∧ x = 67/3 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l204_20466


namespace NUMINAMATH_CALUDE_train_length_l204_20428

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 180 → time_s = 20 → length_m = 1000 → 
  length_m = (speed_kmh * (5/18)) * time_s := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l204_20428


namespace NUMINAMATH_CALUDE_pants_fabric_usage_l204_20464

/-- Proves that each pair of pants uses 5 yards of fabric given the conditions of Jenson and Kingsley's tailoring business. -/
theorem pants_fabric_usage
  (shirts_per_day : ℕ)
  (pants_per_day : ℕ)
  (fabric_per_shirt : ℕ)
  (total_fabric : ℕ)
  (days : ℕ)
  (h1 : shirts_per_day = 3)
  (h2 : pants_per_day = 5)
  (h3 : fabric_per_shirt = 2)
  (h4 : total_fabric = 93)
  (h5 : days = 3) :
  (total_fabric - shirts_per_day * days * fabric_per_shirt) / (pants_per_day * days) = 5 :=
sorry

end NUMINAMATH_CALUDE_pants_fabric_usage_l204_20464


namespace NUMINAMATH_CALUDE_hotel_rooms_l204_20495

theorem hotel_rooms (total_lamps : ℕ) (lamps_per_room : ℕ) (h1 : total_lamps = 147) (h2 : lamps_per_room = 7) :
  total_lamps / lamps_per_room = 21 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rooms_l204_20495


namespace NUMINAMATH_CALUDE_ttakji_count_l204_20456

theorem ttakji_count (n : ℕ) (h : n^2 + 36 = (n + 1)^2 + 3) : n^2 + 36 = 292 := by
  sorry

end NUMINAMATH_CALUDE_ttakji_count_l204_20456


namespace NUMINAMATH_CALUDE_simplest_common_denominator_l204_20422

variable (a : ℝ)
variable (h : a ≠ 0)

theorem simplest_common_denominator : 
  lcm (2 * a) (a ^ 2) = 2 * (a ^ 2) :=
sorry

end NUMINAMATH_CALUDE_simplest_common_denominator_l204_20422


namespace NUMINAMATH_CALUDE_baseball_card_ratio_l204_20442

theorem baseball_card_ratio (rob_total : ℕ) (jess_doubles : ℕ) : 
  rob_total = 24 →
  jess_doubles = 40 →
  (jess_doubles : ℚ) / ((rob_total : ℚ) / 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_baseball_card_ratio_l204_20442


namespace NUMINAMATH_CALUDE_two_color_cubes_count_l204_20470

/-- Represents a cube with painted stripes -/
structure StripedCube where
  edge_length : ℕ
  stripe_count : ℕ

/-- Counts the number of smaller cubes with exactly two faces painted with different colors -/
def count_two_color_cubes (cube : StripedCube) : ℕ :=
  sorry

/-- Theorem stating the correct number of two-color cubes for a 6x6x6 cube with three stripes -/
theorem two_color_cubes_count (cube : StripedCube) :
  cube.edge_length = 6 ∧ cube.stripe_count = 3 →
  count_two_color_cubes cube = 12 :=
by sorry

end NUMINAMATH_CALUDE_two_color_cubes_count_l204_20470
