import Mathlib

namespace NUMINAMATH_CALUDE_root_sum_of_coefficients_l255_25585

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def isRoot (z : ℂ) (b c : ℝ) : Prop :=
  z^2 + b * z + c = 0

-- Theorem statement
theorem root_sum_of_coefficients :
  ∀ (b c : ℝ), isRoot (2 + i) b c → b + c = 1 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_of_coefficients_l255_25585


namespace NUMINAMATH_CALUDE_painting_sale_difference_l255_25513

def previous_painting_sale : ℕ := 9000
def recent_painting_sale : ℕ := 44000

theorem painting_sale_difference : 
  (5 * previous_painting_sale + previous_painting_sale) - recent_painting_sale = 10000 := by
  sorry

end NUMINAMATH_CALUDE_painting_sale_difference_l255_25513


namespace NUMINAMATH_CALUDE_books_unchanged_l255_25532

/-- Represents the number of items before and after a garage sale. -/
structure GarageSale where
  initial_books : ℕ
  initial_pens : ℕ
  sold_pens : ℕ
  final_pens : ℕ

/-- Theorem stating that the number of books remains unchanged after the garage sale. -/
theorem books_unchanged (sale : GarageSale) 
  (h1 : sale.initial_books = 51)
  (h2 : sale.initial_pens = 106)
  (h3 : sale.sold_pens = 92)
  (h4 : sale.final_pens = 14)
  (h5 : sale.initial_pens - sale.sold_pens = sale.final_pens) :
  sale.initial_books = 51 := by
  sorry

end NUMINAMATH_CALUDE_books_unchanged_l255_25532


namespace NUMINAMATH_CALUDE_reciprocal_product_theorem_l255_25599

theorem reciprocal_product_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 6 * x * y) : (1 / x) * (1 / y) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_product_theorem_l255_25599


namespace NUMINAMATH_CALUDE_area_relationship_l255_25571

/-- The area of an isosceles triangle with sides 17, 17, and 16. -/
def P : ℝ := sorry

/-- The area of a right triangle with legs 15 and 20. -/
def Q : ℝ := sorry

/-- Theorem stating the relationship between P and Q. -/
theorem area_relationship : P = (4/5) * Q := by sorry

end NUMINAMATH_CALUDE_area_relationship_l255_25571


namespace NUMINAMATH_CALUDE_net_profit_calculation_l255_25595

def calculate_net_profit (basil_seed_cost mint_seed_cost zinnia_seed_cost : ℚ)
  (potting_soil_cost packaging_cost : ℚ)
  (sellers_fee_rate sales_tax_rate : ℚ)
  (basil_yield mint_yield zinnia_yield : ℕ)
  (basil_germination mint_germination zinnia_germination : ℚ)
  (healthy_basil_price healthy_mint_price healthy_zinnia_price : ℚ)
  (small_basil_price small_mint_price small_zinnia_price : ℚ)
  (healthy_basil_sold small_basil_sold : ℕ)
  (healthy_mint_sold small_mint_sold : ℕ)
  (healthy_zinnia_sold small_zinnia_sold : ℕ) : ℚ :=
  let total_revenue := 
    healthy_basil_price * healthy_basil_sold + small_basil_price * small_basil_sold +
    healthy_mint_price * healthy_mint_sold + small_mint_price * small_mint_sold +
    healthy_zinnia_price * healthy_zinnia_sold + small_zinnia_price * small_zinnia_sold
  let total_expenses := 
    basil_seed_cost + mint_seed_cost + zinnia_seed_cost + potting_soil_cost + packaging_cost
  let sellers_fee := sellers_fee_rate * total_revenue
  let sales_tax := sales_tax_rate * total_revenue
  total_revenue - total_expenses - sellers_fee - sales_tax

theorem net_profit_calculation : 
  calculate_net_profit 2 3 7 15 5 (1/10) (1/20)
    20 15 10 (4/5) (3/4) (7/10)
    5 6 10 3 4 7
    12 8 10 4 5 2 = 158.4 := by sorry

end NUMINAMATH_CALUDE_net_profit_calculation_l255_25595


namespace NUMINAMATH_CALUDE_binomial_150_150_l255_25574

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_l255_25574


namespace NUMINAMATH_CALUDE_smallest_m_with_divisible_digit_sum_l255_25527

/-- Represents the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Checks if a number's digit sum is divisible by 6 -/
def hasSumDivisibleBy6 (n : ℕ) : Prop :=
  (digitSum n) % 6 = 0

/-- Main theorem: 9 is the smallest m satisfying the condition -/
theorem smallest_m_with_divisible_digit_sum : 
  ∀ (start : ℕ), ∃ (i : ℕ), i < 9 ∧ hasSumDivisibleBy6 (start + i) ∧
  ∀ (m : ℕ), m < 9 → ∃ (start' : ℕ), ∀ (j : ℕ), j < m → ¬hasSumDivisibleBy6 (start' + j) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_with_divisible_digit_sum_l255_25527


namespace NUMINAMATH_CALUDE_lamp_height_difference_example_l255_25515

/-- The height difference between two lamps -/
def lamp_height_difference (new_height old_height : ℝ) : ℝ :=
  new_height - old_height

/-- Theorem: The height difference between a new lamp of 2.33 feet and an old lamp of 1 foot is 1.33 feet -/
theorem lamp_height_difference_example :
  lamp_height_difference 2.33 1 = 1.33 := by
  sorry

end NUMINAMATH_CALUDE_lamp_height_difference_example_l255_25515


namespace NUMINAMATH_CALUDE_train_speed_calculation_l255_25534

theorem train_speed_calculation (t m s : ℝ) (ht : t > 0) (hm : m > 0) (hs : s > 0) :
  let m₁ := (Real.sqrt (t * m * (4 * s + t * m)) - t * m) / (2 * t)
  ∃ (t₁ : ℝ), t₁ > 0 ∧ m₁ * t₁ = s ∧ (m₁ + m) * (t₁ - t) = s :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l255_25534


namespace NUMINAMATH_CALUDE_possible_values_of_a_l255_25551

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

-- Define the theorem
theorem possible_values_of_a :
  ∀ a : ℝ, (M ∩ N a = N a) → (a = -1 ∨ a = 0 ∨ a = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l255_25551


namespace NUMINAMATH_CALUDE_a_not_square_l255_25526

/-- Sequence definition -/
def a : ℕ → ℚ
  | 0 => 2016
  | n + 1 => a n + 2 / (a n)

/-- Theorem statement -/
theorem a_not_square : ∀ n : ℕ, ¬ ∃ q : ℚ, a n = q ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_not_square_l255_25526


namespace NUMINAMATH_CALUDE_empty_solution_set_l255_25589

theorem empty_solution_set : ∀ x : ℝ, ¬(2 * x - x^2 > 5) := by
  sorry

end NUMINAMATH_CALUDE_empty_solution_set_l255_25589


namespace NUMINAMATH_CALUDE_distribution_difference_l255_25586

theorem distribution_difference (total : ℕ) (p q r s : ℕ) : 
  total = 1000 →
  p = 2 * q →
  s = 4 * r →
  q = r →
  p + q + r + s = total →
  s - p = 250 := by
sorry

end NUMINAMATH_CALUDE_distribution_difference_l255_25586


namespace NUMINAMATH_CALUDE_no_solution_exists_l255_25507

theorem no_solution_exists : ¬ ∃ (a b : ℤ), a^2 = b^15 + 1004 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l255_25507


namespace NUMINAMATH_CALUDE_sandy_dog_puppies_l255_25569

/-- The number of puppies Sandy now has -/
def total_puppies : ℕ := 12

/-- The number of puppies Sandy's friend gave her -/
def friend_puppies : ℕ := 4

/-- The number of puppies Sandy's dog initially had -/
def initial_puppies : ℕ := total_puppies - friend_puppies

theorem sandy_dog_puppies : initial_puppies = 8 := by
  sorry

end NUMINAMATH_CALUDE_sandy_dog_puppies_l255_25569


namespace NUMINAMATH_CALUDE_cookie_drop_count_l255_25561

/-- Represents the number of cookies of each type made by Alice and Bob --/
structure CookieCount where
  chocolate_chip : ℕ
  sugar : ℕ
  oatmeal_raisin : ℕ
  peanut_butter : ℕ
  snickerdoodle : ℕ
  white_chocolate_macadamia : ℕ

/-- Calculates the total number of cookies --/
def total_cookies (c : CookieCount) : ℕ :=
  c.chocolate_chip + c.sugar + c.oatmeal_raisin + c.peanut_butter + c.snickerdoodle + c.white_chocolate_macadamia

theorem cookie_drop_count 
  (initial_cookies : CookieCount)
  (initial_dropped : CookieCount)
  (additional_cookies : CookieCount)
  (final_edible_cookies : ℕ) :
  total_cookies initial_cookies + total_cookies additional_cookies - final_edible_cookies = 139 :=
by sorry

end NUMINAMATH_CALUDE_cookie_drop_count_l255_25561


namespace NUMINAMATH_CALUDE_total_jelly_beans_l255_25536

/-- The number of jelly beans needed to fill a large drinking glass -/
def large_glass_beans : ℕ := 50

/-- The number of jelly beans needed to fill a small drinking glass -/
def small_glass_beans : ℕ := large_glass_beans / 2

/-- The number of large drinking glasses -/
def num_large_glasses : ℕ := 5

/-- The number of small drinking glasses -/
def num_small_glasses : ℕ := 3

/-- Theorem stating the total number of jelly beans needed to fill all glasses -/
theorem total_jelly_beans :
  num_large_glasses * large_glass_beans + num_small_glasses * small_glass_beans = 325 := by
  sorry

end NUMINAMATH_CALUDE_total_jelly_beans_l255_25536


namespace NUMINAMATH_CALUDE_board_coverage_problem_boards_l255_25577

/-- Represents a rectangular board --/
structure Board where
  rows : ℕ
  cols : ℕ

/-- Checks if a board can be completely covered by dominoes --/
def canCoverWithDominoes (b : Board) : Prop :=
  (b.rows * b.cols) % 2 = 0

/-- Theorem stating that a board can be covered iff its area is even --/
theorem board_coverage (b : Board) :
  canCoverWithDominoes b ↔ (b.rows * b.cols) % 2 = 0 := by sorry

/-- Function to check if a board can be covered --/
def checkBoard (b : Board) : Bool :=
  (b.rows * b.cols) % 2 = 0

/-- Theorem for the specific boards in the problem --/
theorem problem_boards :
  (¬ checkBoard ⟨5, 5⟩) ∧
  (checkBoard ⟨4, 6⟩) ∧
  (¬ checkBoard ⟨3, 7⟩) ∧
  (checkBoard ⟨5, 6⟩) ∧
  (checkBoard ⟨3, 8⟩) := by sorry

end NUMINAMATH_CALUDE_board_coverage_problem_boards_l255_25577


namespace NUMINAMATH_CALUDE_cube_folding_preserves_adjacency_l255_25545

/-- Represents a face of the cube -/
inductive Face : Type
| One
| Two
| Three
| Four
| Five
| Six

/-- Represents the net of the cube -/
structure CubeNet :=
(faces : List Face)
(adjacent : Face → Face → Bool)

/-- Represents the folded cube -/
structure FoldedCube :=
(faces : List Face)
(adjacent : Face → Face → Bool)

/-- Theorem stating that the face adjacencies in the folded cube
    must match the adjacencies in the original net -/
theorem cube_folding_preserves_adjacency (net : CubeNet) (cube : FoldedCube) :
  (net.faces = cube.faces) →
  (∀ (f1 f2 : Face), net.adjacent f1 f2 = cube.adjacent f1 f2) :=
sorry

end NUMINAMATH_CALUDE_cube_folding_preserves_adjacency_l255_25545


namespace NUMINAMATH_CALUDE_beach_trip_duration_l255_25558

-- Define the variables
def seashells_per_day : ℕ := 7
def total_seashells : ℕ := 35

-- Define the function to calculate the number of days
def days_at_beach : ℕ := total_seashells / seashells_per_day

-- Theorem statement
theorem beach_trip_duration : days_at_beach = 5 := by
  sorry

end NUMINAMATH_CALUDE_beach_trip_duration_l255_25558


namespace NUMINAMATH_CALUDE_basketball_game_score_l255_25598

/-- Represents the score of a team in a basketball game -/
structure Score :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if a sequence is geometric with common ratio r -/
def isGeometric (s : Score) (r : ℚ) : Prop :=
  s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- Checks if a sequence is arithmetic with common difference d -/
def isArithmetic (s : Score) (d : ℕ) : Prop :=
  s.q2 = s.q1 + d ∧ s.q3 = s.q2 + d ∧ s.q4 = s.q3 + d

/-- The main theorem -/
theorem basketball_game_score 
  (sharks lions : Score) 
  (r : ℚ) 
  (d : ℕ) : 
  sharks.q1 = lions.q1 →  -- Tied at first quarter
  isGeometric sharks r →  -- Sharks scored in geometric sequence
  isArithmetic lions d →  -- Lions scored in arithmetic sequence
  (sharks.q1 + sharks.q2 + sharks.q3 + sharks.q4) = 
    (lions.q1 + lions.q2 + lions.q3 + lions.q4 + 2) →  -- Sharks won by 2 points
  sharks.q1 + sharks.q2 + sharks.q3 + sharks.q4 ≤ 120 →  -- Sharks' total ≤ 120
  lions.q1 + lions.q2 + lions.q3 + lions.q4 ≤ 120 →  -- Lions' total ≤ 120
  sharks.q1 + sharks.q2 + lions.q1 + lions.q2 = 45  -- First half total is 45
  := by sorry

end NUMINAMATH_CALUDE_basketball_game_score_l255_25598


namespace NUMINAMATH_CALUDE_expected_total_cost_is_350_l255_25550

/-- Represents the outcome of a single test -/
inductive TestResult
| Defective
| NonDefective

/-- Represents the possible total costs of the testing process -/
inductive TotalCost
| Cost200
| Cost300
| Cost400

/-- The probability of getting a specific test result -/
def testProbability (result : TestResult) : ℚ :=
  match result with
  | TestResult.Defective => 2/5
  | TestResult.NonDefective => 3/5

/-- The probability of getting a specific total cost -/
def costProbability (cost : TotalCost) : ℚ :=
  match cost with
  | TotalCost.Cost200 => 1/10
  | TotalCost.Cost300 => 3/10
  | TotalCost.Cost400 => 3/5

/-- The cost in yuan for a specific total cost outcome -/
def costValue (cost : TotalCost) : ℚ :=
  match cost with
  | TotalCost.Cost200 => 200
  | TotalCost.Cost300 => 300
  | TotalCost.Cost400 => 400

/-- The expected value of the total cost -/
def expectedTotalCost : ℚ :=
  (costValue TotalCost.Cost200 * costProbability TotalCost.Cost200) +
  (costValue TotalCost.Cost300 * costProbability TotalCost.Cost300) +
  (costValue TotalCost.Cost400 * costProbability TotalCost.Cost400)

theorem expected_total_cost_is_350 :
  expectedTotalCost = 350 := by sorry

end NUMINAMATH_CALUDE_expected_total_cost_is_350_l255_25550


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l255_25548

/-- The number of ways to distribute n indistinguishable objects into k distinct categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of combinations of n items taken k at a time -/
def combinations (n k : ℕ) : ℕ := Nat.choose n k

theorem ice_cream_combinations : 
  distribute 5 4 = combinations 8 3 := by sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l255_25548


namespace NUMINAMATH_CALUDE_projectile_max_height_l255_25505

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- Theorem stating that the maximum height of the projectile is 161 meters -/
theorem projectile_max_height :
  ∃ t : ℝ, h t = 161 ∧ ∀ s : ℝ, h s ≤ h t :=
sorry

end NUMINAMATH_CALUDE_projectile_max_height_l255_25505


namespace NUMINAMATH_CALUDE_system_solution_range_l255_25579

theorem system_solution_range (x y m : ℝ) : 
  (x + 2*y = m + 4) →
  (2*x + y = 2*m - 1) →
  (x + y < 2) →
  (x - y < 4) →
  m < 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_range_l255_25579


namespace NUMINAMATH_CALUDE_chantel_bracelets_l255_25529

def bracelet_problem (
  days1 : ℕ) (bracelets_per_day1 : ℕ) (give_away1 : ℕ)
  (days2 : ℕ) (bracelets_per_day2 : ℕ) (give_away2 : ℕ)
  (days3 : ℕ) (bracelets_per_day3 : ℕ)
  (days4 : ℕ) (bracelets_per_day4 : ℕ) (give_away3 : ℕ) : ℕ :=
  (days1 * bracelets_per_day1 - give_away1 +
   days2 * bracelets_per_day2 - give_away2 +
   days3 * bracelets_per_day3 +
   days4 * bracelets_per_day4 - give_away3)

theorem chantel_bracelets :
  bracelet_problem 7 4 8 10 5 12 4 6 2 3 10 = 78 := by
  sorry

end NUMINAMATH_CALUDE_chantel_bracelets_l255_25529


namespace NUMINAMATH_CALUDE_three_greater_than_negative_five_l255_25584

theorem three_greater_than_negative_five :
  3 > -5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_three_greater_than_negative_five_l255_25584


namespace NUMINAMATH_CALUDE_parabola_through_point_l255_25582

/-- The value of 'a' for a parabola y = ax^2 passing through (-1, 2) -/
theorem parabola_through_point (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) → 2 = a * (-1)^2 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_through_point_l255_25582


namespace NUMINAMATH_CALUDE_jens_son_age_l255_25539

theorem jens_son_age :
  ∀ (sons_age : ℕ),
  (41 : ℕ) = 25 + sons_age →  -- Jen was 25 when her son was born, and she's 41 now
  (41 : ℕ) = 3 * sons_age - 7 →  -- Jen's age is 7 less than 3 times her son's age
  sons_age = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_jens_son_age_l255_25539


namespace NUMINAMATH_CALUDE_min_value_theorem_l255_25575

/-- The minimum value of 1/m + 2/n given the constraints -/
theorem min_value_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (1 / m + 2 / n) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l255_25575


namespace NUMINAMATH_CALUDE_bookstore_new_releases_fraction_l255_25522

/-- Represents a bookstore inventory --/
structure Bookstore where
  total : ℕ
  historicalFiction : ℕ
  historicalFictionNewReleases : ℕ
  otherNewReleases : ℕ

/-- Calculates the fraction of new releases that are historical fiction --/
def newReleasesFraction (store : Bookstore) : ℚ :=
  store.historicalFictionNewReleases / (store.historicalFictionNewReleases + store.otherNewReleases)

theorem bookstore_new_releases_fraction :
  ∀ (store : Bookstore),
    store.total > 0 →
    store.historicalFiction = (2 * store.total) / 5 →
    store.historicalFictionNewReleases = (2 * store.historicalFiction) / 5 →
    store.otherNewReleases = (7 * (store.total - store.historicalFiction)) / 10 →
    newReleasesFraction store = 8 / 29 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_new_releases_fraction_l255_25522


namespace NUMINAMATH_CALUDE_theorem_A_theorem_B_theorem_C_theorem_D_l255_25521

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (intersection : Plane → Plane → Line)

-- Define the planes and lines
variable (α β : Plane)
variable (m n : Line)

-- Axioms for the relations
axiom different_planes : α ≠ β
axiom different_lines : m ≠ n

-- Theorem A
theorem theorem_A : 
  parallel_planes α β → perpendicular_plane_line α m → perpendicular_plane_line β m :=
sorry

-- Theorem B
theorem theorem_B :
  perpendicular_plane_line α m → perpendicular_plane_line α n → parallel_lines m n :=
sorry

-- Theorem C
theorem theorem_C :
  perpendicular_planes α β → intersection α β = n → ¬parallel_line_plane m α → 
  perpendicular_lines m n → perpendicular_plane_line β m :=
sorry

-- Theorem D (which should be false)
theorem theorem_D :
  parallel_line_plane m α → parallel_line_plane n α → 
  parallel_line_plane m β → parallel_line_plane n β → 
  ¬(parallel_planes α β) :=
sorry

end NUMINAMATH_CALUDE_theorem_A_theorem_B_theorem_C_theorem_D_l255_25521


namespace NUMINAMATH_CALUDE_parabola_point_focus_distance_l255_25517

/-- A point on a parabola and its distance to the focus -/
theorem parabola_point_focus_distance :
  ∀ (x y : ℝ),
  y^2 = 8*x →  -- Point (x, y) is on the parabola y^2 = 8x
  x = 4 →      -- The x-coordinate of the point is 4
  Real.sqrt ((x - 2)^2 + y^2) = 6 :=  -- The distance to the focus (2, 0) is 6
by sorry

end NUMINAMATH_CALUDE_parabola_point_focus_distance_l255_25517


namespace NUMINAMATH_CALUDE_ryosuke_trip_cost_l255_25528

/-- Calculates the cost of gas for a trip given the odometer readings, fuel efficiency, and gas price -/
def gas_cost (initial_reading final_reading : ℕ) (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  let distance := final_reading - initial_reading
  let gas_used := (distance : ℚ) / fuel_efficiency
  gas_used * gas_price

/-- Proves that the cost of gas for Ryosuke's trip is $5.04 -/
theorem ryosuke_trip_cost :
  let initial_reading : ℕ := 74580
  let final_reading : ℕ := 74610
  let fuel_efficiency : ℚ := 25
  let gas_price : ℚ := 21/5
  gas_cost initial_reading final_reading fuel_efficiency gas_price = 504/100 := by
  sorry

end NUMINAMATH_CALUDE_ryosuke_trip_cost_l255_25528


namespace NUMINAMATH_CALUDE_frog_distribution_l255_25594

/-- Represents the three lakes in the problem -/
inductive Lake
| Crystal
| Lassie
| Emerald

/-- Represents the three frog species in the problem -/
inductive Species
| A
| B
| C

/-- The number of frogs of a given species in a given lake -/
def frog_count (l : Lake) (s : Species) : ℕ :=
  match l, s with
  | Lake.Lassie, Species.A => 45
  | Lake.Lassie, Species.B => 35
  | Lake.Lassie, Species.C => 25
  | Lake.Crystal, Species.A => 36
  | Lake.Crystal, Species.B => 39
  | Lake.Crystal, Species.C => 25
  | Lake.Emerald, Species.A => 59
  | Lake.Emerald, Species.B => 70
  | Lake.Emerald, Species.C => 38

/-- The total number of frogs of a given species across all lakes -/
def total_frogs (s : Species) : ℕ :=
  (frog_count Lake.Crystal s) + (frog_count Lake.Lassie s) + (frog_count Lake.Emerald s)

theorem frog_distribution :
  (total_frogs Species.A = 140) ∧
  (total_frogs Species.B = 144) ∧
  (total_frogs Species.C = 88) :=
by sorry


end NUMINAMATH_CALUDE_frog_distribution_l255_25594


namespace NUMINAMATH_CALUDE_room_height_calculation_l255_25516

/-- Calculates the height of a room given its dimensions, door and window sizes, and whitewashing cost. -/
theorem room_height_calculation (room_length room_width : ℝ)
  (door_length door_width : ℝ)
  (window_length window_width : ℝ)
  (num_windows : ℕ)
  (whitewash_cost_per_sqft : ℝ)
  (total_cost : ℝ)
  (h : room_length = 25 ∧ room_width = 15 ∧ 
       door_length = 6 ∧ door_width = 3 ∧
       window_length = 4 ∧ window_width = 3 ∧
       num_windows = 3 ∧
       whitewash_cost_per_sqft = 5 ∧
       total_cost = 4530) :
  ∃ (room_height : ℝ),
    room_height = 12 ∧
    total_cost = whitewash_cost_per_sqft * 
      (2 * (room_length + room_width) * room_height - 
       (door_length * door_width + num_windows * window_length * window_width)) :=
by sorry

end NUMINAMATH_CALUDE_room_height_calculation_l255_25516


namespace NUMINAMATH_CALUDE_cyclic_inequality_l255_25542

theorem cyclic_inequality (x y z m n : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hm : m > 0) (hn : n > 0)
  (hmn : m + n ≥ 2) : 
  x * Real.sqrt (y * z * (x + m * y) * (x + n * z)) + 
  y * Real.sqrt (x * z * (y + m * x) * (y + n * z)) + 
  z * Real.sqrt (x * y * (z + m * x) * (z + n * y)) ≤ 
  (3 * (m + n) / 8) * (x + y) * (y + z) * (z + x) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l255_25542


namespace NUMINAMATH_CALUDE_special_op_is_addition_l255_25514

/-- A binary operation on real numbers satisfying (a * b) * c = a + b + c -/
def special_op (a b : ℝ) : ℝ := sorry

/-- The property of the special operation -/
axiom special_op_property (a b c : ℝ) : special_op (special_op a b) c = a + b + c

/-- Theorem: The special operation is equivalent to addition -/
theorem special_op_is_addition (a b : ℝ) : special_op a b = a + b := by sorry

end NUMINAMATH_CALUDE_special_op_is_addition_l255_25514


namespace NUMINAMATH_CALUDE_hexagon_enclosure_octagon_enclosure_l255_25520

-- Define the shapes
def Square (sideLength : ℝ) : Type := Unit
def RegularHexagon (sideLength : ℝ) : Type := Unit
def Circle (diameter : ℝ) : Type := Unit

-- Define the derived shapes
def Hexagon (s : Square 1) : Type := Unit
def Octagon (h : RegularHexagon (Real.sqrt 3 / 3)) : Type := Unit

-- Define the enclosure property
def CanEnclose (shape : Type) (figure : Circle 1) : Prop := sorry

-- State the theorems
theorem hexagon_enclosure (s : Square 1) (f : Circle 1) :
  CanEnclose (Hexagon s) f := sorry

theorem octagon_enclosure (h : RegularHexagon (Real.sqrt 3 / 3)) (f : Circle 1) :
  CanEnclose (Octagon h) f := sorry

end NUMINAMATH_CALUDE_hexagon_enclosure_octagon_enclosure_l255_25520


namespace NUMINAMATH_CALUDE_max_abc_value_l255_25512

theorem max_abc_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b + c + a * b = (a + c) * (b + c))
  (h2 : a + b + c = 2) :
  a * b * c ≤ 8 / 27 :=
sorry

end NUMINAMATH_CALUDE_max_abc_value_l255_25512


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l255_25518

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ)
  (germination_rate_plot1 total_germination_rate : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate_plot1 = 30 / 100 →
  total_germination_rate = 32 / 100 →
  (germination_rate_plot1 * seeds_plot1 + (35 / 100) * seeds_plot2) / (seeds_plot1 + seeds_plot2) = total_germination_rate :=
by sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l255_25518


namespace NUMINAMATH_CALUDE_factor_polynomial_l255_25540

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l255_25540


namespace NUMINAMATH_CALUDE_max_leftover_candy_l255_25593

theorem max_leftover_candy (y : ℕ) : ∃ (q r : ℕ), y = 6 * q + r ∧ r < 6 ∧ r ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_candy_l255_25593


namespace NUMINAMATH_CALUDE_iphone_price_drop_l255_25547

/-- Calculates the final price of an iPhone after two consecutive price drops -/
theorem iphone_price_drop (initial_price : ℝ) (first_drop : ℝ) (second_drop : ℝ) :
  initial_price = 1000 ∧ first_drop = 0.1 ∧ second_drop = 0.2 →
  initial_price * (1 - first_drop) * (1 - second_drop) = 720 := by
  sorry


end NUMINAMATH_CALUDE_iphone_price_drop_l255_25547


namespace NUMINAMATH_CALUDE_total_material_is_correct_l255_25564

/-- The amount of sand required for the renovation project in truck-loads -/
def sand : ℚ := 0.16666666666666666

/-- The amount of dirt required for the renovation project in truck-loads -/
def dirt : ℚ := 0.3333333333333333

/-- The amount of cement required for the renovation project in truck-loads -/
def cement : ℚ := 0.16666666666666666

/-- The total amount of material required for the renovation project in truck-loads -/
def total_material : ℚ := sand + dirt + cement

/-- Theorem stating that the total amount of material required is 0.6666666666666666 truck-loads -/
theorem total_material_is_correct : total_material = 0.6666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_total_material_is_correct_l255_25564


namespace NUMINAMATH_CALUDE_inequality_solution_l255_25501

-- Define the inequality
def inequality (x a : ℝ) : Prop := x^2 - 2*a*x - 3*a^2 < 0

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x | inequality x a}

-- Theorem statement
theorem inequality_solution :
  ∀ a : ℝ,
    (a = 0 → solution_set a = ∅) ∧
    (a > 0 → solution_set a = {x | -a < x ∧ x < 3*a}) ∧
    (a < 0 → solution_set a = {x | 3*a < x ∧ x < -a}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l255_25501


namespace NUMINAMATH_CALUDE_largest_multiple_of_nine_less_than_hundred_l255_25573

theorem largest_multiple_of_nine_less_than_hundred : 
  ∃ (n : ℕ), n * 9 = 99 ∧ 
  ∀ (m : ℕ), m * 9 < 100 → m * 9 ≤ 99 := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_nine_less_than_hundred_l255_25573


namespace NUMINAMATH_CALUDE_total_distance_equals_expected_l255_25567

/-- The initial travel distance per year in kilometers -/
def initial_distance : ℝ := 983400000000

/-- The factor by which the speed increases every 50 years -/
def speed_increase_factor : ℝ := 2

/-- The number of years for each speed increase -/
def years_per_increase : ℕ := 50

/-- The total number of years of travel -/
def total_years : ℕ := 150

/-- The function to calculate the total distance traveled -/
def total_distance : ℝ := 
  initial_distance * years_per_increase * (1 + speed_increase_factor + speed_increase_factor^2)

theorem total_distance_equals_expected : 
  total_distance = 3.4718e14 := by sorry

end NUMINAMATH_CALUDE_total_distance_equals_expected_l255_25567


namespace NUMINAMATH_CALUDE_cross_section_area_is_21_over_8_l255_25535

/-- Right prism with specific properties -/
structure RightPrism where
  -- Base triangle
  base_hypotenuse : ℝ
  base_angle_B : ℝ
  base_angle_C : ℝ
  -- Cutting plane properties
  distance_C_to_plane : ℝ

/-- The cross-section of the prism -/
def cross_section_area (prism : RightPrism) : ℝ :=
  sorry

/-- Main theorem: The area of the cross-section is 21/8 -/
theorem cross_section_area_is_21_over_8 (prism : RightPrism) 
  (h1 : prism.base_hypotenuse = Real.sqrt 14)
  (h2 : prism.base_angle_B = 90)
  (h3 : prism.base_angle_C = 30)
  (h4 : prism.distance_C_to_plane = 2) :
  cross_section_area prism = 21 / 8 :=
sorry

end NUMINAMATH_CALUDE_cross_section_area_is_21_over_8_l255_25535


namespace NUMINAMATH_CALUDE_max_pen_area_l255_25570

/-- The maximum area of a rectangular pen with one side against a wall,
    given 30 meters of fencing for the other three sides. -/
theorem max_pen_area (total_fence : ℝ) (h_total_fence : total_fence = 30) :
  ∃ (width height : ℝ),
    width > 0 ∧
    height > 0 ∧
    width + 2 * height = total_fence ∧
    ∀ (w h : ℝ), w > 0 → h > 0 → w + 2 * h = total_fence →
      w * h ≤ width * height ∧
      width * height = 112 :=
by sorry

end NUMINAMATH_CALUDE_max_pen_area_l255_25570


namespace NUMINAMATH_CALUDE_probability_same_group_four_people_prove_probability_same_group_four_people_l255_25541

/-- The probability that two specific people are in the same group when four people are divided into two groups. -/
theorem probability_same_group_four_people : ℚ :=
  5 / 6

/-- Proof that the probability of two specific people being in the same group when four people are divided into two groups is 5/6. -/
theorem prove_probability_same_group_four_people :
  probability_same_group_four_people = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_group_four_people_prove_probability_same_group_four_people_l255_25541


namespace NUMINAMATH_CALUDE_total_clients_l255_25503

/-- Represents the number of clients needing vegan meals -/
def vegan : ℕ := 7

/-- Represents the number of clients needing kosher meals -/
def kosher : ℕ := 8

/-- Represents the number of clients needing both vegan and kosher meals -/
def both : ℕ := 3

/-- Represents the number of clients needing neither vegan nor kosher meals -/
def neither : ℕ := 18

/-- Theorem stating that the total number of clients is 30 -/
theorem total_clients : vegan + kosher - both + neither = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_clients_l255_25503


namespace NUMINAMATH_CALUDE_min_abs_z_on_circle_l255_25587

theorem min_abs_z_on_circle (z : ℂ) (h : Complex.abs (z - (1 + Complex.I)) = 1) :
  ∃ (w : ℂ), Complex.abs (w - (1 + Complex.I)) = 1 ∧
             Complex.abs w = Real.sqrt 2 - 1 ∧
             ∀ (v : ℂ), Complex.abs (v - (1 + Complex.I)) = 1 → Complex.abs w ≤ Complex.abs v :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_on_circle_l255_25587


namespace NUMINAMATH_CALUDE_joe_money_left_l255_25524

def initial_amount : ℕ := 56
def notebooks_bought : ℕ := 7
def books_bought : ℕ := 2
def notebook_cost : ℕ := 4
def book_cost : ℕ := 7

theorem joe_money_left : 
  initial_amount - (notebooks_bought * notebook_cost + books_bought * book_cost) = 14 := by
  sorry

end NUMINAMATH_CALUDE_joe_money_left_l255_25524


namespace NUMINAMATH_CALUDE_shaded_area_octagon_semicircles_l255_25509

/-- The area of the shaded region inside a regular octagon but outside eight semicircles -/
theorem shaded_area_octagon_semicircles : 
  let s : ℝ := 4  -- side length of the octagon
  let octagon_area : ℝ := 2 * (1 + Real.sqrt 2) * s^2
  let semicircle_area : ℝ := π * (s/2)^2 / 2
  let total_semicircle_area : ℝ := 8 * semicircle_area
  octagon_area - total_semicircle_area = 32 * (1 + Real.sqrt 2) - 16 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_octagon_semicircles_l255_25509


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l255_25531

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 6 > 0} = {x : ℝ | x < -3 ∨ x > 2} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l255_25531


namespace NUMINAMATH_CALUDE_clara_pill_cost_l255_25581

/-- The cost of pills for Clara's treatment --/
def pill_cost (blue_cost : ℚ) : Prop :=
  let days : ℕ := 10
  let red_cost : ℚ := blue_cost - 2
  let daily_cost : ℚ := blue_cost + red_cost
  let total_cost : ℚ := 480
  (days : ℚ) * daily_cost = total_cost ∧ blue_cost = 25

theorem clara_pill_cost : ∃ (blue_cost : ℚ), pill_cost blue_cost := by
  sorry

end NUMINAMATH_CALUDE_clara_pill_cost_l255_25581


namespace NUMINAMATH_CALUDE_D_96_l255_25506

/-- D(n) is the number of ways of writing n as a product of integers greater than 1, where the order matters -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem: D(96) = 112 -/
theorem D_96 : D 96 = 112 := by sorry

end NUMINAMATH_CALUDE_D_96_l255_25506


namespace NUMINAMATH_CALUDE_gloria_pine_tree_price_l255_25508

/-- Proves that the price per pine tree is $200 given the conditions of Gloria's cabin purchase --/
theorem gloria_pine_tree_price :
  let cabin_price : ℕ := 129000
  let initial_cash : ℕ := 150
  let cypress_trees : ℕ := 20
  let pine_trees : ℕ := 600
  let maple_trees : ℕ := 24
  let cypress_price : ℕ := 100
  let maple_price : ℕ := 300
  let remaining_cash : ℕ := 350
  let pine_price : ℕ := (cabin_price - initial_cash + remaining_cash - 
    (cypress_trees * cypress_price + maple_trees * maple_price)) / pine_trees
  pine_price = 200 := by sorry

end NUMINAMATH_CALUDE_gloria_pine_tree_price_l255_25508


namespace NUMINAMATH_CALUDE_max_value_x2_y2_z3_max_value_achieved_l255_25519

theorem max_value_x2_y2_z3 (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_constraint : x + 2*y + 3*z = 1) : 
  x^2 + y^2 + z^3 ≤ 1 :=
by sorry

theorem max_value_achieved (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_constraint : x + 2*y + 3*z = 1) : 
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + 2*b + 3*c = 1 ∧ a^2 + b^2 + c^3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x2_y2_z3_max_value_achieved_l255_25519


namespace NUMINAMATH_CALUDE_percentage_increase_l255_25597

theorem percentage_increase (x y : ℝ) (h : x > y) :
  (x - y) / y * 100 = 50 → x = 132 ∧ y = 88 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l255_25597


namespace NUMINAMATH_CALUDE_min_value_theorem_l255_25500

theorem min_value_theorem (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ 3 * Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l255_25500


namespace NUMINAMATH_CALUDE_coefficient_x6_sum_binomial_expansions_l255_25596

theorem coefficient_x6_sum_binomial_expansions :
  let f (n : ℕ) := (1 + X : Polynomial ℚ)^n
  let expansion := f 5 + f 6 + f 7
  (expansion.coeff 6 : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x6_sum_binomial_expansions_l255_25596


namespace NUMINAMATH_CALUDE_farmer_feed_expenditure_l255_25562

theorem farmer_feed_expenditure (initial_amount : ℝ) :
  (initial_amount * 0.4 / 0.5) + (initial_amount * 0.6) = 49 →
  initial_amount = 35 := by
sorry

end NUMINAMATH_CALUDE_farmer_feed_expenditure_l255_25562


namespace NUMINAMATH_CALUDE_is_projection_matrix_l255_25560

def projection_matrix (A : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  A * A = A

theorem is_projection_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![11/12, 12/25; 12/25, 13/25]
  projection_matrix A := by
  sorry

end NUMINAMATH_CALUDE_is_projection_matrix_l255_25560


namespace NUMINAMATH_CALUDE_band_competition_l255_25566

theorem band_competition (flute trumpet trombone drummer clarinet french_horn : ℕ) : 
  trumpet = 3 * flute ∧ 
  trombone = trumpet - 8 ∧ 
  drummer = trombone + 11 ∧ 
  clarinet = 2 * flute ∧ 
  french_horn = trombone + 3 ∧ 
  flute + trumpet + trombone + drummer + clarinet + french_horn = 65 → 
  flute = 6 := by sorry

end NUMINAMATH_CALUDE_band_competition_l255_25566


namespace NUMINAMATH_CALUDE_triangle_inequality_l255_25591

/-- Given a triangle ABC with point P inside, prove the inequality involving
    sides and distances from P to the sides. -/
theorem triangle_inequality (a b c d₁ d₂ d₃ S_ABC : ℝ) 
    (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
    (h₄ : d₁ > 0) (h₅ : d₂ > 0) (h₆ : d₃ > 0)
    (h₇ : S_ABC > 0)
    (h₈ : S_ABC = (1/2) * (a * d₁ + b * d₂ + c * d₃)) :
  (a / d₁) + (b / d₂) + (c / d₃) ≥ (a + b + c)^2 / (2 * S_ABC) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l255_25591


namespace NUMINAMATH_CALUDE_a_range_l255_25510

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

def range_of_a (a : ℝ) : Prop := (1 ≤ a ∧ a < 2) ∨ a ≤ -2

theorem a_range (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) : range_of_a a :=
sorry

end NUMINAMATH_CALUDE_a_range_l255_25510


namespace NUMINAMATH_CALUDE_triangle_with_semiprime_angles_l255_25543

/-- A number is semi-prime if it's a product of exactly two primes (not necessarily distinct) -/
def IsSemiPrime (n : ℕ) : Prop := ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p * q

/-- The smallest semi-prime number -/
def SmallestSemiPrime : ℕ := 4

theorem triangle_with_semiprime_angles (p q : ℕ) :
  p = 2 * q →
  IsSemiPrime p →
  IsSemiPrime q →
  (p = SmallestSemiPrime ∨ q = SmallestSemiPrime) →
  ∃ x : ℕ, x = 168 ∧ p + q + x = 180 :=
sorry

end NUMINAMATH_CALUDE_triangle_with_semiprime_angles_l255_25543


namespace NUMINAMATH_CALUDE_lcm_1560_1040_l255_25538

theorem lcm_1560_1040 : Nat.lcm 1560 1040 = 3120 := by
  sorry

end NUMINAMATH_CALUDE_lcm_1560_1040_l255_25538


namespace NUMINAMATH_CALUDE_max_abc_value_l255_25572

theorem max_abc_value (a b c : ℕ+) 
  (h1 : a * b + b * c = 518)
  (h2 : a * b - a * c = 360) :
  (∀ x y z : ℕ+, x * y + y * z = 518 → x * y - x * z = 360 → a * b * c ≥ x * y * z) ∧ 
  a * b * c = 1008 :=
sorry

end NUMINAMATH_CALUDE_max_abc_value_l255_25572


namespace NUMINAMATH_CALUDE_product_of_successive_numbers_l255_25583

theorem product_of_successive_numbers : 
  let x : ℝ := 97.49871794028884
  let y : ℝ := x + 1
  abs (x * y - 9603) < 0.001 := by
sorry

end NUMINAMATH_CALUDE_product_of_successive_numbers_l255_25583


namespace NUMINAMATH_CALUDE_johnny_age_puzzle_l255_25568

/-- Represents Johnny's age now -/
def current_age : ℕ := 8

/-- Represents the number of years into the future Johnny is referring to -/
def future_years : ℕ := 2

/-- Theorem stating that the number of years into the future Johnny was referring to is correct -/
theorem johnny_age_puzzle :
  (current_age + future_years = 2 * (current_age - 3)) ∧
  (future_years = 2) := by
  sorry

end NUMINAMATH_CALUDE_johnny_age_puzzle_l255_25568


namespace NUMINAMATH_CALUDE_f_sin_pi_12_l255_25555

theorem f_sin_pi_12 (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (π / 12)) = - (Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sin_pi_12_l255_25555


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l255_25533

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = d

theorem sequence_is_arithmetic (a : ℕ → ℝ) 
    (h : ∀ n, 3 * a (n + 1) = 3 * a n + 1) : 
    is_arithmetic_sequence a (1/3) := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l255_25533


namespace NUMINAMATH_CALUDE_probability_qualified_bulb_factory_A_l255_25576

/-- The probability of purchasing a qualified light bulb produced by Factory A from the market -/
theorem probability_qualified_bulb_factory_A
  (factory_A_production_rate : ℝ)
  (factory_B_production_rate : ℝ)
  (factory_A_pass_rate : ℝ)
  (factory_B_pass_rate : ℝ)
  (h1 : factory_A_production_rate = 0.7)
  (h2 : factory_B_production_rate = 0.3)
  (h3 : factory_A_pass_rate = 0.95)
  (h4 : factory_B_pass_rate = 0.8)
  (h5 : factory_A_production_rate + factory_B_production_rate = 1) :
  factory_A_production_rate * factory_A_pass_rate = 0.665 := by
  sorry


end NUMINAMATH_CALUDE_probability_qualified_bulb_factory_A_l255_25576


namespace NUMINAMATH_CALUDE_Q_satisfies_conditions_l255_25578

/-- A polynomial Q(x) with the given properties -/
def Q (x : ℝ) : ℝ := 4 - x + x^2

/-- The theorem stating that Q(x) satisfies the given conditions -/
theorem Q_satisfies_conditions :
  (Q (-2) = 2) ∧
  (∀ x, Q x = Q 0 + Q 1 * x + Q 2 * x^2 + Q 3 * x^3) :=
by sorry

end NUMINAMATH_CALUDE_Q_satisfies_conditions_l255_25578


namespace NUMINAMATH_CALUDE_expenditure_savings_ratio_l255_25544

/-- Given an income, expenditure, and savings, proves that the ratio of expenditure to savings is 1.5:1 -/
theorem expenditure_savings_ratio 
  (income expenditure savings : ℝ) 
  (h1 : income = expenditure + savings)
  (h2 : 1.15 * income = 1.21 * expenditure + 1.06 * savings) : 
  expenditure = 1.5 * savings := by
  sorry

end NUMINAMATH_CALUDE_expenditure_savings_ratio_l255_25544


namespace NUMINAMATH_CALUDE_expenditure_ratio_l255_25556

-- Define the incomes and expenditures
def uma_income : ℚ := 20000
def bala_income : ℚ := 15000
def uma_expenditure : ℚ := 15000
def bala_expenditure : ℚ := 10000
def savings : ℚ := 5000

-- Define the theorem
theorem expenditure_ratio :
  (uma_income / bala_income = 4 / 3) →
  (uma_income = 20000) →
  (uma_income - uma_expenditure = savings) →
  (bala_income - bala_expenditure = savings) →
  (uma_expenditure / bala_expenditure = 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_expenditure_ratio_l255_25556


namespace NUMINAMATH_CALUDE_balog_theorem_l255_25530

theorem balog_theorem (q : ℕ+) (A : Finset ℤ) :
  ∃ (C_q : ℕ), (A.card + q * A.card : ℤ) ≥ ((q + 1) * A.card : ℤ) - C_q :=
by sorry

end NUMINAMATH_CALUDE_balog_theorem_l255_25530


namespace NUMINAMATH_CALUDE_f_sum_over_sum_positive_l255_25559

noncomputable def f (x : ℝ) : ℝ := x^3 - Real.log (Real.sqrt (x^2 + 1) - x)

theorem f_sum_over_sum_positive (a b : ℝ) (h : a + b ≠ 0) :
  (f a + f b) / (a + b) > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_over_sum_positive_l255_25559


namespace NUMINAMATH_CALUDE_discount_is_twenty_percent_l255_25592

/-- Calculates the discount percentage given the original price, quantity, tax rate, and final price --/
def calculate_discount_percentage (original_price quantity : ℕ) (tax_rate final_price : ℚ) : ℚ :=
  let discounted_price := final_price / (1 + tax_rate) / quantity
  let discount_amount := original_price - discounted_price
  (discount_amount / original_price) * 100

/-- The discount percentage is 20% given the problem conditions --/
theorem discount_is_twenty_percent :
  calculate_discount_percentage 45 10 (1/10) 396 = 20 := by
  sorry

end NUMINAMATH_CALUDE_discount_is_twenty_percent_l255_25592


namespace NUMINAMATH_CALUDE_henrysFriendMoney_l255_25557

/-- Calculates the amount of money Henry's friend has -/
def friendsMoney (henryInitial : ℕ) (henryEarned : ℕ) (totalCombined : ℕ) : ℕ :=
  totalCombined - (henryInitial + henryEarned)

/-- Theorem: Henry's friend has 13 dollars -/
theorem henrysFriendMoney : friendsMoney 5 2 20 = 13 := by
  sorry

end NUMINAMATH_CALUDE_henrysFriendMoney_l255_25557


namespace NUMINAMATH_CALUDE_train_speed_calculation_l255_25590

/-- Proves that a train of length 60 m crossing an electric pole in 1.4998800095992322 seconds has a speed of approximately 11.112 km/hr -/
theorem train_speed_calculation (train_length : Real) (crossing_time : Real) 
  (h1 : train_length = 60) 
  (h2 : crossing_time = 1.4998800095992322) : 
  ∃ (speed : Real), abs (speed - 11.112) < 0.001 ∧ 
  speed = (train_length / crossing_time) * (3600 / 1000) := by
  sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l255_25590


namespace NUMINAMATH_CALUDE_special_function_property_l255_25525

/-- A function f: ℝ → ℝ satisfying specific properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (x - 1) = -f (-x - 1)) ∧  -- f(x-1) is odd
  (∀ x, f (x + 1) = f (-x + 1)) ∧  -- f(x+1) is even
  (∀ x, x > -1 ∧ x < 1 → f x = -Real.exp x)  -- f(x) = -e^x for x ∈ (-1,1)

/-- Theorem stating the property of the special function -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  ∀ x, f (2 * x) = f (2 * x + 8) :=
by sorry

end NUMINAMATH_CALUDE_special_function_property_l255_25525


namespace NUMINAMATH_CALUDE_line_equation_and_range_l255_25549

/-- A line passing through two points -/
structure Line where
  k : ℝ
  b : ℝ

/-- The y-coordinate of a point on the line given its x-coordinate -/
def Line.y_at (l : Line) (x : ℝ) : ℝ := l.k * x + l.b

theorem line_equation_and_range (l : Line) 
  (h1 : l.y_at (-1) = 2)
  (h2 : l.y_at 2 = 5) :
  (∀ x, l.y_at x = x + 3) ∧ 
  (∀ x, l.y_at x > 0 ↔ x > -3) := by
  sorry


end NUMINAMATH_CALUDE_line_equation_and_range_l255_25549


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_l255_25523

theorem smallest_undefined_inverse (b : ℕ) : b = 6 ↔ 
  (b > 0) ∧ 
  (∀ x : ℕ, x * b % 30 ≠ 1) ∧ 
  (∀ y : ℕ, y * b % 42 ≠ 1) ∧ 
  (∀ c < b, c > 0 → (∃ x : ℕ, x * c % 30 = 1) ∨ (∃ y : ℕ, y * c % 42 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_l255_25523


namespace NUMINAMATH_CALUDE_seating_theorem_l255_25580

/-- The number of ways to seat 2 students in a row of 5 desks with at least one empty desk between them -/
def seating_arrangements : ℕ := 12

/-- The number of desks in the row -/
def num_desks : ℕ := 5

/-- The number of students to be seated -/
def num_students : ℕ := 2

/-- Minimum number of empty desks between students -/
def min_empty_desks : ℕ := 1

theorem seating_theorem :
  seating_arrangements = 12 ∧
  num_desks = 5 ∧
  num_students = 2 ∧
  min_empty_desks = 1 →
  seating_arrangements = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l255_25580


namespace NUMINAMATH_CALUDE_sqrt_fifteen_over_two_equals_half_sqrt_thirty_l255_25546

theorem sqrt_fifteen_over_two_equals_half_sqrt_thirty :
  Real.sqrt (15 / 2) = (1 / 2) * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fifteen_over_two_equals_half_sqrt_thirty_l255_25546


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l255_25563

/-- The radius of the inscribed circle in a triangle with sides 26, 15, and 17 is √6 -/
theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 26) (hb : b = 15) (hc : c = 17) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l255_25563


namespace NUMINAMATH_CALUDE_prop_1_prop_3_prop_4_l255_25504

open Real

-- Define the second quadrant
def second_quadrant (θ : ℝ) : Prop := π/2 < θ ∧ θ < π

-- Proposition 1
theorem prop_1 (θ : ℝ) (h : second_quadrant θ) : sin θ * tan θ < 0 := by
  sorry

-- Proposition 3
theorem prop_3 : sin 1 * cos 2 * tan 3 > 0 := by
  sorry

-- Proposition 4
theorem prop_4 (θ : ℝ) (h : 3*π/2 < θ ∧ θ < 2*π) : sin (π + θ) > 0 := by
  sorry

end NUMINAMATH_CALUDE_prop_1_prop_3_prop_4_l255_25504


namespace NUMINAMATH_CALUDE_f_of_f_10_eq_2_l255_25565

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else Real.log x / Real.log 2

theorem f_of_f_10_eq_2 : f (f 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_f_10_eq_2_l255_25565


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l255_25553

theorem simplify_trig_expression : 
  (Real.sqrt (1 - 2 * Real.sin (20 * π / 180) * Real.cos (20 * π / 180))) / 
  (Real.cos (20 * π / 180) - Real.sqrt (1 - Real.cos (160 * π / 180) ^ 2)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l255_25553


namespace NUMINAMATH_CALUDE_gold_coin_distribution_l255_25537

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def problem_conditions (w : MerchantWealth) : Prop :=
  (w.ierema + 70 = w.yuliy) ∧ (w.foma - 40 = w.yuliy)

/-- The theorem to prove -/
theorem gold_coin_distribution (w : MerchantWealth) 
  (h : problem_conditions w) : w.foma - 55 = w.ierema + 55 := by
  sorry

#check gold_coin_distribution

end NUMINAMATH_CALUDE_gold_coin_distribution_l255_25537


namespace NUMINAMATH_CALUDE_printer_X_time_l255_25511

/-- The time (in hours) it takes for printer Y to complete the job alone -/
def time_Y : ℝ := 12

/-- The time (in hours) it takes for printer Z to complete the job alone -/
def time_Z : ℝ := 8

/-- The ratio of the time it takes printer X alone to the time it takes printers Y and Z together -/
def ratio : ℝ := 3.333333333333333

theorem printer_X_time : ∃ (time_X : ℝ), time_X = 16 ∧
  ratio = time_X / (1 / (1 / time_Y + 1 / time_Z)) :=
sorry

end NUMINAMATH_CALUDE_printer_X_time_l255_25511


namespace NUMINAMATH_CALUDE_parallelepiped_dimensions_l255_25554

theorem parallelepiped_dimensions (n : ℕ) (h1 : n > 6) : 
  (n - 2) * (n - 4) * (n - 6) = (2 / 3) * n * (n - 2) * (n - 4) → n = 18 := by
sorry

end NUMINAMATH_CALUDE_parallelepiped_dimensions_l255_25554


namespace NUMINAMATH_CALUDE_octahedron_triangle_count_l255_25588

/-- A regular octahedron -/
structure RegularOctahedron where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 6 × Fin 6)
  vertex_count : vertices.card = 6
  edge_count : edges.card = 12
  edge_validity : ∀ e ∈ edges, e.1 ≠ e.2 ∧ e.1 ∈ vertices ∧ e.2 ∈ vertices

/-- A triangle on the octahedron -/
structure OctahedronTriangle (O : RegularOctahedron) where
  vertices : Finset (Fin 6)
  vertex_count : vertices.card = 3
  vertex_validity : vertices ⊆ O.vertices
  edge_shared : ∃ e ∈ O.edges, (e.1 ∈ vertices ∧ e.2 ∈ vertices)

/-- The set of all valid triangles on the octahedron -/
def validTriangles (O : RegularOctahedron) : Set (OctahedronTriangle O) :=
  {t | t.vertices ⊆ O.vertices ∧ ∃ e ∈ O.edges, (e.1 ∈ t.vertices ∧ e.2 ∈ t.vertices)}

theorem octahedron_triangle_count (O : RegularOctahedron) :
  (validTriangles O).ncard = 12 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_triangle_count_l255_25588


namespace NUMINAMATH_CALUDE_angle_c_in_triangle_l255_25552

theorem angle_c_in_triangle (A B C : ℝ) (h1 : A + B = 80) (h2 : A + B + C = 180) : C = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_in_triangle_l255_25552


namespace NUMINAMATH_CALUDE_investment_average_interest_rate_l255_25502

/-- Prove that given a total investment split between two interest rates with equal annual returns, the average rate of interest is as calculated. -/
theorem investment_average_interest_rate 
  (total_investment : ℝ)
  (rate1 rate2 : ℝ)
  (h1 : total_investment = 6000)
  (h2 : rate1 = 0.03)
  (h3 : rate2 = 0.07)
  (h4 : ∃ (x : ℝ), x * rate2 = (total_investment - x) * rate1) :
  (rate1 * (total_investment - (180 / 0.1)) + rate2 * (180 / 0.1)) / total_investment = 0.042 := by
  sorry

end NUMINAMATH_CALUDE_investment_average_interest_rate_l255_25502
