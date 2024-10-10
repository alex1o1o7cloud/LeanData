import Mathlib

namespace investment_gain_percentage_l2853_285360

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

end investment_gain_percentage_l2853_285360


namespace pentagon_side_length_l2853_285353

/-- Given an equilateral triangle with side length 9/20 cm, prove that a regular pentagon with the same perimeter has side length 27/100 cm. -/
theorem pentagon_side_length (triangle_side : ℝ) (pentagon_side : ℝ) : 
  triangle_side = 9/20 → 
  3 * triangle_side = 5 * pentagon_side → 
  pentagon_side = 27/100 := by sorry

end pentagon_side_length_l2853_285353


namespace square_root_product_l2853_285330

theorem square_root_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) := by
  sorry

end square_root_product_l2853_285330


namespace yard_length_26_trees_l2853_285379

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 26 equally spaced trees,
    where the distance between consecutive trees is 15 meters, is 375 meters -/
theorem yard_length_26_trees :
  yard_length 26 15 = 375 := by
  sorry

end yard_length_26_trees_l2853_285379


namespace investment_plans_count_l2853_285310

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

end investment_plans_count_l2853_285310


namespace store_annual_profits_l2853_285391

/-- Calculates the annual profits given the profits for each quarter -/
def annual_profits (q1 q2 q3 q4 : ℕ) : ℕ :=
  q1 + q2 + q3 + q4

/-- Theorem stating that the annual profits are $8,000 given the quarterly profits -/
theorem store_annual_profits :
  let q1 : ℕ := 1500
  let q2 : ℕ := 1500
  let q3 : ℕ := 3000
  let q4 : ℕ := 2000
  annual_profits q1 q2 q3 q4 = 8000 := by
  sorry

end store_annual_profits_l2853_285391


namespace room_width_calculation_l2853_285323

theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 9 →
  cost_per_sqm = 900 →
  total_cost = 38475 →
  (total_cost / cost_per_sqm) / length = 4.75 := by
  sorry

end room_width_calculation_l2853_285323


namespace product_modulo_300_l2853_285375

theorem product_modulo_300 : (2025 * 1233) % 300 = 75 := by
  sorry

end product_modulo_300_l2853_285375


namespace paper_area_problem_l2853_285373

theorem paper_area_problem (L : ℝ) : 
  2 * (11 * L) = 2 * (8.5 * 11) + 100 ↔ L = 287 / 22 := by sorry

end paper_area_problem_l2853_285373


namespace ttakji_count_l2853_285312

theorem ttakji_count (n : ℕ) (h : n^2 + 36 = (n + 1)^2 + 3) : n^2 + 36 = 292 := by
  sorry

end ttakji_count_l2853_285312


namespace bats_against_left_handed_correct_l2853_285398

/-- Represents a baseball player's batting statistics -/
structure BattingStats where
  total_bats : ℕ
  total_hits : ℕ
  left_handed_avg : ℚ
  right_handed_avg : ℚ

/-- Calculates the number of bats against left-handed pitchers -/
def bats_against_left_handed (stats : BattingStats) : ℕ :=
  sorry

/-- Theorem stating the correct number of bats against left-handed pitchers -/
theorem bats_against_left_handed_correct (stats : BattingStats) 
  (h1 : stats.total_bats = 600)
  (h2 : stats.total_hits = 192)
  (h3 : stats.left_handed_avg = 1/4)
  (h4 : stats.right_handed_avg = 7/20)
  (h5 : (stats.total_hits : ℚ) / stats.total_bats = 8/25) :
  bats_against_left_handed stats = 180 :=
sorry

end bats_against_left_handed_correct_l2853_285398


namespace negation_of_existential_quadratic_inequality_l2853_285335

theorem negation_of_existential_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x ≤ 1) ↔ (∀ x : ℝ, x^2 + 2*x > 1) := by sorry

end negation_of_existential_quadratic_inequality_l2853_285335


namespace initial_books_borrowed_l2853_285385

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


end initial_books_borrowed_l2853_285385


namespace find_A_l2853_285300

theorem find_A : ∃ A : ℤ, A + 10 = 15 ∧ A = 5 := by
  sorry

end find_A_l2853_285300


namespace lifesaving_test_percentage_l2853_285315

/-- The percentage of swim club members who have passed the lifesaving test -/
def percentage_passed : ℝ := 30

theorem lifesaving_test_percentage :
  let total_members : ℕ := 60
  let not_passed_with_course : ℕ := 12
  let not_passed_without_course : ℕ := 30
  percentage_passed = 30 ∧
  percentage_passed = (total_members - (not_passed_with_course + not_passed_without_course)) / total_members * 100 :=
by sorry

end lifesaving_test_percentage_l2853_285315


namespace all_roots_of_polynomial_l2853_285370

/-- The polynomial function we're considering -/
def f (x : ℝ) : ℝ := x^3 - x^2 - 4*x + 4

/-- The set of roots we claim are correct -/
def roots : Set ℝ := {-2, 1, 2}

/-- Theorem stating that the given set contains all roots of the polynomial -/
theorem all_roots_of_polynomial :
  ∀ x : ℝ, f x = 0 ↔ x ∈ roots := by sorry

end all_roots_of_polynomial_l2853_285370


namespace total_hamburger_combinations_l2853_285327

/-- The number of available condiments -/
def num_condiments : ℕ := 10

/-- The number of choices for meat patties -/
def patty_choices : ℕ := 4

/-- Theorem stating the total number of hamburger combinations -/
theorem total_hamburger_combinations :
  2^num_condiments * patty_choices = 4096 := by
  sorry

end total_hamburger_combinations_l2853_285327


namespace max_squares_after_triangles_l2853_285324

/-- Represents the number of matchsticks used to form triangles efficiently -/
def triangleMatchsticks : ℕ := 13

/-- Represents the total number of matchsticks available -/
def totalMatchsticks : ℕ := 24

/-- Represents the number of matchsticks required to form a square -/
def matchsticksPerSquare : ℕ := 4

/-- Represents the number of triangles to be formed -/
def numTriangles : ℕ := 6

/-- Theorem stating the maximum number of squares that can be formed -/
theorem max_squares_after_triangles :
  (totalMatchsticks - triangleMatchsticks) / matchsticksPerSquare = 4 :=
sorry

end max_squares_after_triangles_l2853_285324


namespace range_of_x_l2853_285381

theorem range_of_x (x : ℝ) : 
  (∀ (a b : ℝ), a ≠ 0 → b ≠ 0 → |3*a + b| + |a - b| ≥ |a| * (|x - 1| + |x + 1|)) 
  ↔ x ∈ Set.Icc (-2) 2 :=
sorry

end range_of_x_l2853_285381


namespace raspberry_ratio_l2853_285378

theorem raspberry_ratio (total_berries : ℕ) (blackberries : ℕ) (blueberries : ℕ) :
  total_berries = 42 →
  blackberries = total_berries / 3 →
  blueberries = 7 →
  (total_berries - blackberries - blueberries) * 2 = total_berries := by
  sorry

end raspberry_ratio_l2853_285378


namespace positive_function_from_condition_l2853_285316

theorem positive_function_from_condition (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h' : ∀ x : ℝ, f x + x * deriv f x > 0) : 
  ∀ x : ℝ, f x > 0 := by
  sorry

end positive_function_from_condition_l2853_285316


namespace square_difference_equality_l2853_285308

theorem square_difference_equality : (19 + 15)^2 - (19 - 15)^2 = 1140 := by
  sorry

end square_difference_equality_l2853_285308


namespace closest_integer_to_cube_root_closest_integer_to_cube_root_of_sum_of_cubes_l2853_285340

theorem closest_integer_to_cube_root (x : ℝ) : 
  ∃ n : ℤ, ∀ m : ℤ, |x - n| ≤ |x - m| := by sorry

theorem closest_integer_to_cube_root_of_sum_of_cubes : 
  ∃ n : ℤ, (∀ m : ℤ, |((7 : ℝ)^3 + 9^3)^(1/3) - n| ≤ |((7 : ℝ)^3 + 9^3)^(1/3) - m|) ∧ n = 10 := by sorry

end closest_integer_to_cube_root_closest_integer_to_cube_root_of_sum_of_cubes_l2853_285340


namespace arithmetic_sequence_property_l2853_285345

-- Define an arithmetic sequence
def isArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℤ)
  (h_arithmetic : isArithmeticSequence a)
  (h_a4 : a 4 = -4)
  (h_a8 : a 8 = 4) :
  a 12 = 12 := by
  sorry

end arithmetic_sequence_property_l2853_285345


namespace saree_price_problem_l2853_285349

theorem saree_price_problem (P : ℝ) : 
  P * (1 - 0.1) * (1 - 0.05) = 171 → P = 200 := by
  sorry

end saree_price_problem_l2853_285349


namespace power_function_through_point_l2853_285333

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x ^ a

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f (1/2) = 8) : 
  f 2 = 1/8 := by
sorry

end power_function_through_point_l2853_285333


namespace quadratic_radical_range_l2853_285337

theorem quadratic_radical_range : 
  {x : ℝ | ∃ y : ℝ, y^2 = 3*x - 1} = {x : ℝ | x ≥ 1/3} := by
  sorry

end quadratic_radical_range_l2853_285337


namespace special_right_triangle_legs_lengths_l2853_285332

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

end special_right_triangle_legs_lengths_l2853_285332


namespace equation_solution_l2853_285331

theorem equation_solution (x : ℝ) : 
  x^2 + 3*x + 2 ≠ 0 →
  (-x^2 = (4*x + 2) / (x^2 + 3*x + 2)) ↔ x = -1 := by
  sorry

end equation_solution_l2853_285331


namespace complex_power_six_l2853_285313

theorem complex_power_six (i : ℂ) (h : i^2 = -1) : (1 + i)^6 = -8*i := by
  sorry

end complex_power_six_l2853_285313


namespace value_calculation_l2853_285374

theorem value_calculation (number : ℕ) (value : ℕ) 
  (h1 : value = 5 * number) 
  (h2 : number = 20) : 
  value = 100 := by
sorry

end value_calculation_l2853_285374


namespace equal_perimeter_interior_tiles_l2853_285334

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

end equal_perimeter_interior_tiles_l2853_285334


namespace quadratic_roots_property_l2853_285377

theorem quadratic_roots_property (a : ℝ) (x₁ x₂ : ℝ) : 
  (x₁ ≠ x₂) →
  (x₁^2 + a*x₁ + 2 = 0) →
  (x₂^2 + a*x₂ + 2 = 0) →
  (x₁^3 + 14/x₂^2 = x₂^3 + 14/x₁^2) →
  (a = 4) := by
  sorry

end quadratic_roots_property_l2853_285377


namespace box_width_proof_l2853_285303

theorem box_width_proof (length width height : ℕ) (cubes : ℕ) : 
  length = 15 → height = 13 → cubes = 3120 → cubes = length * width * height → width = 16 := by
  sorry

end box_width_proof_l2853_285303


namespace dice_roll_probability_l2853_285329

def is_valid_roll (a b : Nat) : Prop :=
  a ≤ 6 ∧ b ≤ 6 ∧ a + b ≤ 10 ∧ (a > 3 ∨ b > 3)

def total_outcomes : Nat := 36

def valid_outcomes : Nat := 24

theorem dice_roll_probability : 
  (valid_outcomes : ℚ) / total_outcomes = 2 / 3 := by sorry

end dice_roll_probability_l2853_285329


namespace corner_sum_is_200_l2853_285311

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

end corner_sum_is_200_l2853_285311


namespace range_of_abc_l2853_285371

theorem range_of_abc (a b c : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) (h4 : 2 < c) (h5 : c < 3) :
  ∀ x, (∃ a' b' c', -1 < a' ∧ a' < b' ∧ b' < 1 ∧ 2 < c' ∧ c' < 3 ∧ x = (a' - b') * c') → -6 < x ∧ x < 0 := by
  sorry

end range_of_abc_l2853_285371


namespace hcd_problem_l2853_285338

theorem hcd_problem : (Nat.gcd 12348 2448 * 3) - 14 = 94 := by
  sorry

end hcd_problem_l2853_285338


namespace swan_population_l2853_285366

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

end swan_population_l2853_285366


namespace intersection_theorem_l2853_285346

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

end intersection_theorem_l2853_285346


namespace base_seven_digits_of_4300_l2853_285387

theorem base_seven_digits_of_4300 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 4300 ∧ 4300 < 7^n ∧ n = 5 := by
  sorry

end base_seven_digits_of_4300_l2853_285387


namespace scientific_notation_of_169200000000_l2853_285314

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

end scientific_notation_of_169200000000_l2853_285314


namespace opening_night_customers_count_l2853_285347

/-- Represents the revenue and customer data for a movie theater on a specific day. -/
structure TheaterData where
  matineePrice : ℕ
  eveningPrice : ℕ
  openingNightPrice : ℕ
  popcornPrice : ℕ
  matineeCustomers : ℕ
  eveningCustomers : ℕ
  totalRevenue : ℕ

/-- Calculates the number of opening night customers given theater data. -/
def openingNightCustomers (data : TheaterData) : ℕ :=
  let totalCustomers := data.matineeCustomers + data.eveningCustomers + (data.totalRevenue - 
    (data.matineePrice * data.matineeCustomers + 
     data.eveningPrice * data.eveningCustomers + 
     (data.popcornPrice * (data.matineeCustomers + data.eveningCustomers)) / 2)) / data.openingNightPrice
  (data.totalRevenue - 
   (data.matineePrice * data.matineeCustomers + 
    data.eveningPrice * data.eveningCustomers + 
    data.popcornPrice * totalCustomers / 2)) / data.openingNightPrice

theorem opening_night_customers_count (data : TheaterData) 
  (h1 : data.matineePrice = 5)
  (h2 : data.eveningPrice = 7)
  (h3 : data.openingNightPrice = 10)
  (h4 : data.popcornPrice = 10)
  (h5 : data.matineeCustomers = 32)
  (h6 : data.eveningCustomers = 40)
  (h7 : data.totalRevenue = 1670) :
  openingNightCustomers data = 58 := by
  sorry

#eval openingNightCustomers {
  matineePrice := 5,
  eveningPrice := 7,
  openingNightPrice := 10,
  popcornPrice := 10,
  matineeCustomers := 32,
  eveningCustomers := 40,
  totalRevenue := 1670
}

end opening_night_customers_count_l2853_285347


namespace rhinos_count_l2853_285364

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

end rhinos_count_l2853_285364


namespace smallest_valid_coloring_distance_l2853_285302

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

end smallest_valid_coloring_distance_l2853_285302


namespace triangle_max_value_l2853_285339

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that if a² + b² = √3ab + c² and AB = 1, then the maximum value of AC + √3BC is 2√7 -/
theorem triangle_max_value (a b c : ℝ) (A B C : ℝ) :
  a^2 + b^2 = Real.sqrt 3 * a * b + c^2 →
  a = 1 →  -- AB = 1
  ∃ (AC BC : ℝ), AC + Real.sqrt 3 * BC ≤ 2 * Real.sqrt 7 ∧
    ∃ (AC' BC' : ℝ), AC' + Real.sqrt 3 * BC' = 2 * Real.sqrt 7 :=
by sorry

end triangle_max_value_l2853_285339


namespace largest_number_l2853_285318

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 99 / 100)
  (hb : b = 9099 / 10000)
  (hc : c = 9 / 10)
  (hd : d = 909 / 1000)
  (he : e = 9009 / 10000) :
  a > b ∧ a > c ∧ a > d ∧ a > e :=
sorry

end largest_number_l2853_285318


namespace systematic_sample_fourth_number_l2853_285394

/-- Represents a systematic sampling scheme -/
structure SystematicSample where
  total : ℕ        -- Total number of items
  sampleSize : ℕ   -- Size of the sample
  step : ℕ         -- Step size for systematic sampling
  first : ℕ        -- First sample number

/-- Generates the nth sample number in a systematic sample -/
def nthSample (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first + (n - 1) * s.step

/-- Checks if a number is in the sample -/
def isInSample (s : SystematicSample) (num : ℕ) : Prop :=
  ∃ n : ℕ, n ≤ s.sampleSize ∧ nthSample s n = num

theorem systematic_sample_fourth_number
  (s : SystematicSample)
  (h_total : s.total = 52)
  (h_size : s.sampleSize = 4)
  (h_7 : isInSample s 7)
  (h_33 : isInSample s 33)
  (h_46 : isInSample s 46) :
  isInSample s 20 :=
sorry

end systematic_sample_fourth_number_l2853_285394


namespace initial_books_count_l2853_285344

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

end initial_books_count_l2853_285344


namespace equation_solution_l2853_285351

theorem equation_solution : ∃ x : ℚ, (1 / 3 + 1 / x = 7 / 9 + 1) ∧ (x = 9 / 13) := by
  sorry

end equation_solution_l2853_285351


namespace square_area_error_l2853_285397

theorem square_area_error (s : ℝ) (h : s > 0) : 
  let measured_side := s * 1.06
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 12.36 := by
sorry

end square_area_error_l2853_285397


namespace marble_distribution_l2853_285361

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

end marble_distribution_l2853_285361


namespace geometric_sequence_formula_l2853_285321

/-- Given a geometric sequence {a_n} with first three terms a-1, a+1, a+2, 
    prove that its general formula is a_n = -1/(2^(n-3)) -/
theorem geometric_sequence_formula (a : ℝ) (a_n : ℕ → ℝ) :
  a_n 1 = a - 1 →
  a_n 2 = a + 1 →
  a_n 3 = a + 2 →
  (∀ n : ℕ, n ≥ 1 → a_n (n + 1) / a_n n = a_n 2 / a_n 1) →
  ∀ n : ℕ, n ≥ 1 → a_n n = -1 / (2^(n - 3)) :=
by sorry

end geometric_sequence_formula_l2853_285321


namespace absolute_value_equation_solution_l2853_285384

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 25| + |x - 21| = |2*x - 46| + |x - 17| ∧ x = 67/3 :=
by sorry

end absolute_value_equation_solution_l2853_285384


namespace money_market_investment_ratio_l2853_285363

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

end money_market_investment_ratio_l2853_285363


namespace pants_fabric_usage_l2853_285336

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

end pants_fabric_usage_l2853_285336


namespace largest_constant_inequality_l2853_285304

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

end largest_constant_inequality_l2853_285304


namespace tenth_row_sum_l2853_285306

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

end tenth_row_sum_l2853_285306


namespace tracy_candies_l2853_285389

theorem tracy_candies (initial_candies : ℕ) : 
  (∃ (sister_took : ℕ),
    initial_candies > 0 ∧
    sister_took ≥ 2 ∧ 
    sister_took ≤ 6 ∧
    (initial_candies * 3 / 4) * 2 / 3 - 40 - sister_took = 10) →
  initial_candies = 108 :=
by sorry

end tracy_candies_l2853_285389


namespace miles_walked_approx_2250_l2853_285395

/-- Represents a pedometer with a maximum reading before reset --/
structure Pedometer where
  max_reading : ℕ
  steps_per_mile : ℕ

/-- Represents the pedometer readings over a year --/
structure YearlyReading where
  resets : ℕ
  final_reading : ℕ

/-- Calculates the total miles walked based on pedometer data --/
def total_miles_walked (p : Pedometer) (yr : YearlyReading) : ℚ :=
  let total_steps : ℕ := p.max_reading * yr.resets + yr.final_reading + 1
  (total_steps : ℚ) / p.steps_per_mile

/-- Theorem stating that the total miles walked is approximately 2250 --/
theorem miles_walked_approx_2250 (p : Pedometer) (yr : YearlyReading) :
  p.max_reading = 99999 →
  p.steps_per_mile = 1600 →
  yr.resets = 36 →
  yr.final_reading = 25000 →
  2249 < total_miles_walked p yr ∧ total_miles_walked p yr < 2251 :=
sorry

end miles_walked_approx_2250_l2853_285395


namespace irrational_power_congruence_l2853_285383

theorem irrational_power_congruence :
  ∀ (k : ℕ), k ≥ 2 →
  ∃ (r : ℝ), Irrational r ∧
    ∀ (m : ℕ), (⌊r^m⌋ : ℤ) ≡ -1 [ZMOD k] :=
sorry

end irrational_power_congruence_l2853_285383


namespace trapezoid_construction_l2853_285358

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

end trapezoid_construction_l2853_285358


namespace div_chain_equals_four_l2853_285342

theorem div_chain_equals_four : (((120 / 5) / 3) / 2) = 4 := by
  sorry

end div_chain_equals_four_l2853_285342


namespace connie_marbles_to_juan_l2853_285393

/-- Represents the number of marbles Connie gave to Juan -/
def marbles_given_to_juan (initial_marbles : ℕ) (remaining_marbles : ℕ) : ℕ :=
  initial_marbles - remaining_marbles

/-- Proves that Connie gave 73 marbles to Juan -/
theorem connie_marbles_to_juan :
  marbles_given_to_juan 143 70 = 73 := by
  sorry

end connie_marbles_to_juan_l2853_285393


namespace perpendicular_vectors_l2853_285362

theorem perpendicular_vectors (m : ℚ) : 
  let a : ℚ × ℚ := (-2, m)
  let b : ℚ × ℚ := (-1, 3)
  (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 = 0 → m = 8/3 := by
  sorry

end perpendicular_vectors_l2853_285362


namespace base_b_number_not_divisible_by_four_l2853_285388

theorem base_b_number_not_divisible_by_four (b : ℕ) : b ∈ ({4, 5, 6, 7, 8} : Finset ℕ) →
  (b^3 + b^2 - b + 2) % 4 ≠ 0 ↔ b ∈ ({4, 5, 7, 8} : Finset ℕ) := by
  sorry

end base_b_number_not_divisible_by_four_l2853_285388


namespace pi_estimation_l2853_285369

theorem pi_estimation (n : ℕ) (m : ℕ) (h1 : n = 120) (h2 : m = 34) :
  let π_estimate := 4 * (m / n + 1 / 2)
  π_estimate = 47 / 15 := by
  sorry

end pi_estimation_l2853_285369


namespace repetend_5_17_l2853_285317

def repetend_of_5_17 : List Nat := [2, 9, 4, 1, 1, 7, 6, 4, 7, 0, 5, 8, 8, 2, 3, 5, 2, 9]

theorem repetend_5_17 :
  ∃ (k : ℕ), (5 : ℚ) / 17 = (k : ℚ) / 10^18 + 
  (List.sum (List.zipWith (λ (d i : ℕ) => (d : ℚ) / 10^(i+1)) repetend_of_5_17 (List.range 18))) *
  (1 / (1 - 1 / 10^18)) :=
by
  sorry

end repetend_5_17_l2853_285317


namespace tan_sum_equality_l2853_285328

theorem tan_sum_equality (A B : ℝ) 
  (h1 : A + B = (5 / 4) * Real.pi)
  (h2 : ∀ k : ℤ, A ≠ k * Real.pi + Real.pi / 2)
  (h3 : ∀ k : ℤ, B ≠ k * Real.pi + Real.pi / 2) :
  (1 + Real.tan A) * (1 + Real.tan B) = 2 := by
  sorry

end tan_sum_equality_l2853_285328


namespace cloth_profit_proof_l2853_285365

def cloth_problem (selling_price total_meters cost_price_per_meter : ℕ) : Prop :=
  let total_cost := total_meters * cost_price_per_meter
  let total_profit := selling_price - total_cost
  let profit_per_meter := total_profit / total_meters
  profit_per_meter = 5

theorem cloth_profit_proof :
  cloth_problem 8925 85 100 := by
  sorry

end cloth_profit_proof_l2853_285365


namespace largest_x_for_prime_f_l2853_285356

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def f (x : ℤ) : ℤ := |4*x^2 - 41*x + 21|

theorem largest_x_for_prime_f :
  ∀ x : ℤ, x > 2 → ¬(is_prime (f x).toNat) ∧ is_prime (f 2).toNat :=
sorry

end largest_x_for_prime_f_l2853_285356


namespace sum_of_squares_l2853_285320

theorem sum_of_squares (a b c : ℝ) (h1 : a * b + b * c + a * c = 72) (h2 : a + b + c = 14) :
  a^2 + b^2 + c^2 = 52 := by
sorry

end sum_of_squares_l2853_285320


namespace factor_expression_l2853_285376

theorem factor_expression (a b c : ℝ) : 
  ((a^2 - b^2)^4 + (b^2 - c^2)^4 + (c^2 - a^2)^4) / ((a - b)^4 + (b - c)^4 + (c - a)^4) = 1 := by
  sorry

end factor_expression_l2853_285376


namespace product_of_base8_digits_7354_l2853_285357

/-- The base 8 representation of a natural number -/
def base8Representation (n : ℕ) : List ℕ :=
  sorry

/-- The product of a list of natural numbers -/
def productList (l : List ℕ) : ℕ :=
  sorry

theorem product_of_base8_digits_7354 :
  productList (base8Representation 7354) = 0 :=
sorry

end product_of_base8_digits_7354_l2853_285357


namespace continued_fraction_solution_l2853_285380

/-- The continued fraction equation representing the given expression -/
def continued_fraction_equation (x : ℝ) : Prop :=
  x = 3 + 5 / (2 + 5 / x)

/-- The theorem stating that 5 is the solution to the continued fraction equation -/
theorem continued_fraction_solution :
  ∃ (x : ℝ), continued_fraction_equation x ∧ x = 5 := by
  sorry

end continued_fraction_solution_l2853_285380


namespace triangle_side_length_l2853_285382

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

end triangle_side_length_l2853_285382


namespace denominator_value_l2853_285350

theorem denominator_value (x : ℝ) (h : (1 / x) ^ 1 = 0.25) : x = 4 := by
  sorry

end denominator_value_l2853_285350


namespace opposite_numbers_solution_l2853_285386

theorem opposite_numbers_solution (x : ℝ) : 2 * (x - 3) = -(4 * (1 - x)) → x = -1 := by
  sorry

end opposite_numbers_solution_l2853_285386


namespace train_length_l2853_285352

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 180 → time_s = 20 → length_m = 1000 → 
  length_m = (speed_kmh * (5/18)) * time_s := by
  sorry

#check train_length

end train_length_l2853_285352


namespace cinnamon_nutmeg_difference_l2853_285390

theorem cinnamon_nutmeg_difference :
  let cinnamon : Float := 0.6666666666666666
  let nutmeg : Float := 0.5
  cinnamon - nutmeg = 0.1666666666666666 := by
  sorry

end cinnamon_nutmeg_difference_l2853_285390


namespace original_element_l2853_285348

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 2 * p.2, 2 * p.1 - p.2)

/-- Theorem: If f(x, y) = (3, 1), then (x, y) = (1, 1) -/
theorem original_element (x y : ℝ) (h : f (x, y) = (3, 1)) : (x, y) = (1, 1) := by
  sorry

end original_element_l2853_285348


namespace base4_1010_equals_68_l2853_285307

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

end base4_1010_equals_68_l2853_285307


namespace tangent_line_circle_range_l2853_285396

theorem tangent_line_circle_range (m n : ℝ) : 
  (∃ (x y : ℝ), (m + 1) * x + (n + 1) * y - 2 = 0 ∧ (x - 1)^2 + (y - 1)^2 = 1) →
  ((m + n ≤ 2 - 2 * Real.sqrt 2) ∨ (m + n ≥ 2 + 2 * Real.sqrt 2)) :=
by sorry

end tangent_line_circle_range_l2853_285396


namespace imaginary_part_of_complex_fraction_l2853_285354

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (1 + I) / (4 + 3 * I) → z.im = 1 / 25 := by
  sorry

end imaginary_part_of_complex_fraction_l2853_285354


namespace initial_cows_count_l2853_285372

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

end initial_cows_count_l2853_285372


namespace triangle_problem_l2853_285392

theorem triangle_problem (A B C : Real) (a b c : Real) :
  a + c = 5 →
  a > c →
  b = 3 →
  Real.cos B = 1/3 →
  a = 3 ∧ c = 2 ∧ Real.cos (A + B) = -7/9 := by
  sorry

end triangle_problem_l2853_285392


namespace problem_statement_l2853_285301

theorem problem_statement (x y z w : ℝ) 
  (eq1 : 2^x + y = 7)
  (eq2 : 2^8 = y + x)
  (eq3 : z = Real.sin (x - y))
  (eq4 : w = 3 * (y + z)) :
  ∃ (result : ℝ), (x + y + z + w) / 4 = result := by
  sorry

end problem_statement_l2853_285301


namespace graph_vertical_shift_l2853_285319

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the vertical shift operation
def verticalShift (g : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := fun x ↦ g x - c

-- Theorem statement
theorem graph_vertical_shift (x : ℝ) : 
  (verticalShift f 2) x = f x - 2 := by sorry

end graph_vertical_shift_l2853_285319


namespace trip_time_difference_l2853_285309

theorem trip_time_difference (distance1 distance2 speed : ℝ) 
  (h1 : distance1 = 240)
  (h2 : distance2 = 420)
  (h3 : speed = 60) :
  distance2 / speed - distance1 / speed = 3 := by
  sorry

end trip_time_difference_l2853_285309


namespace triangle_problem_l2853_285399

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a + b = 6 →
  c = 2 →
  Real.cos C = 7/9 →
  a = 3 ∧ b = 3 ∧ (1/2 * a * b * Real.sin C = 2 * Real.sqrt 2) :=
by sorry

end triangle_problem_l2853_285399


namespace inequality_implies_a_range_l2853_285322

theorem inequality_implies_a_range :
  (∀ x : ℝ, (3 : ℝ)^(x^2 - 2*a*x) > (1/3 : ℝ)^(x + 1)) →
  -1/2 < a ∧ a < 3/2 :=
by sorry

end inequality_implies_a_range_l2853_285322


namespace sample_customers_l2853_285325

theorem sample_customers (samples_per_box : ℕ) (boxes_opened : ℕ) (samples_left : ℕ) : 
  samples_per_box = 20 →
  boxes_opened = 12 →
  samples_left = 5 →
  (samples_per_box * boxes_opened - samples_left) = 235 :=
by
  sorry

end sample_customers_l2853_285325


namespace line_intercepts_sum_l2853_285326

/-- Given a line with equation y + 3 = -3(x - 5), prove that the sum of its x-intercept and y-intercept is 16 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y + 3 = -3 * (x - 5)) → 
  ∃ (x_int y_int : ℝ), 
    (y_int + 3 = -3 * (x_int - 5)) ∧ 
    (0 + 3 = -3 * (x_int - 5)) ∧ 
    (y_int + 3 = -3 * (0 - 5)) ∧ 
    (x_int + y_int = 16) := by
  sorry

end line_intercepts_sum_l2853_285326


namespace baseball_card_ratio_l2853_285343

theorem baseball_card_ratio (rob_total : ℕ) (jess_doubles : ℕ) : 
  rob_total = 24 →
  jess_doubles = 40 →
  (jess_doubles : ℚ) / ((rob_total : ℚ) / 3) = 5 := by
  sorry

end baseball_card_ratio_l2853_285343


namespace simplest_common_denominator_l2853_285368

variable (a : ℝ)
variable (h : a ≠ 0)

theorem simplest_common_denominator : 
  lcm (2 * a) (a ^ 2) = 2 * (a ^ 2) :=
sorry

end simplest_common_denominator_l2853_285368


namespace power_expansion_l2853_285355

theorem power_expansion (x : ℝ) : (3*x)^2 * x^2 = 9*x^4 := by
  sorry

end power_expansion_l2853_285355


namespace tangent_slope_at_point_two_l2853_285359

-- Define the curve
def curve (x : ℝ) : ℝ := 2 * x^2

-- Define the slope of the tangent line at a point
def tangent_slope (x : ℝ) : ℝ := 4 * x

-- Theorem statement
theorem tangent_slope_at_point_two :
  tangent_slope 2 = 8 :=
sorry

end tangent_slope_at_point_two_l2853_285359


namespace projection_a_on_b_is_sqrt_5_l2853_285341

def a : Fin 2 → ℝ := ![1, 3]
def b : Fin 2 → ℝ := ![-2, 4]

theorem projection_a_on_b_is_sqrt_5 :
  let dot_product := (a 0) * (b 0) + (a 1) * (b 1)
  let magnitude_b := Real.sqrt ((b 0)^2 + (b 1)^2)
  dot_product / magnitude_b = Real.sqrt 5 := by sorry

end projection_a_on_b_is_sqrt_5_l2853_285341


namespace sum_of_m_for_integer_solutions_l2853_285305

theorem sum_of_m_for_integer_solutions : ∃ (S : Finset Int),
  (∀ m : Int, m ∈ S ↔ 
    (∃ x y : Int, x^2 - m*x + 15 = 0 ∧ y^2 - m*y + 15 = 0 ∧ x ≠ y)) ∧
  (S.sum id = 48) := by
  sorry

end sum_of_m_for_integer_solutions_l2853_285305


namespace max_rectangle_area_l2853_285367

theorem max_rectangle_area (perimeter : ℕ) (h_perimeter : perimeter = 156) :
  ∃ (length width : ℕ),
    2 * (length + width) = perimeter ∧
    ∀ (l w : ℕ), 2 * (l + w) = perimeter → l * w ≤ length * width ∧
    length * width = 1521 := by
  sorry

end max_rectangle_area_l2853_285367
