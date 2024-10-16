import Mathlib

namespace NUMINAMATH_CALUDE_money_distribution_l392_39264

theorem money_distribution (a b c d e : ℕ) : 
  a + b + c + d + e = 1000 →
  a + c = 300 →
  b + c = 200 →
  d + e = 350 →
  a + d = 250 →
  b + e = 150 →
  a + b + c = 400 →
  (a = 200 ∧ b = 100 ∧ c = 100 ∧ d = 50 ∧ e = 300) :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l392_39264


namespace NUMINAMATH_CALUDE_sector_area_from_arc_length_l392_39259

/-- Given a circle with radius 6cm and an arc length of 25.12cm, 
    the area of the sector formed by this arc is 75.36 cm². -/
theorem sector_area_from_arc_length : 
  let r : ℝ := 6  -- radius in cm
  let arc_length : ℝ := 25.12  -- arc length in cm
  let π : ℝ := Real.pi
  let central_angle : ℝ := arc_length / r  -- angle in radians
  let sector_area : ℝ := 0.5 * r^2 * central_angle
  sector_area = 75.36 := by
  sorry

#check sector_area_from_arc_length

end NUMINAMATH_CALUDE_sector_area_from_arc_length_l392_39259


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l392_39254

theorem system_of_equations_sum (x y z : ℝ) 
  (eq1 : y + z = 16 - 4*x)
  (eq2 : x + z = -18 - 4*y)
  (eq3 : x + y = 13 - 4*z) :
  2*x + 2*y + 2*z = 11/3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l392_39254


namespace NUMINAMATH_CALUDE_first_tree_growth_rate_l392_39296

/-- The daily growth rate of the first tree -/
def first_tree_growth : ℝ := 1

/-- The daily growth rate of the second tree -/
def second_tree_growth : ℝ := 2 * first_tree_growth

/-- The daily growth rate of the third tree -/
def third_tree_growth : ℝ := 2

/-- The daily growth rate of the fourth tree -/
def fourth_tree_growth : ℝ := 3

/-- The number of days the trees grew -/
def days : ℕ := 4

/-- The total growth of all trees -/
def total_growth : ℝ := 32

theorem first_tree_growth_rate :
  first_tree_growth * days +
  second_tree_growth * days +
  third_tree_growth * days +
  fourth_tree_growth * days = total_growth :=
by sorry

end NUMINAMATH_CALUDE_first_tree_growth_rate_l392_39296


namespace NUMINAMATH_CALUDE_peggy_left_knee_bandages_l392_39278

/-- The number of bandages Peggy used on her left knee -/
def bandages_on_left_knee (initial_bandages : ℕ) (remaining_bandages : ℕ) (right_knee_bandages : ℕ) : ℕ :=
  initial_bandages - remaining_bandages - right_knee_bandages

/-- Proof that Peggy used 2 bandages on her left knee -/
theorem peggy_left_knee_bandages : 
  let initial_bandages := 24 - 8
  let remaining_bandages := 11
  let right_knee_bandages := 3
  bandages_on_left_knee initial_bandages remaining_bandages right_knee_bandages = 2 := by
sorry

#eval bandages_on_left_knee (24 - 8) 11 3

end NUMINAMATH_CALUDE_peggy_left_knee_bandages_l392_39278


namespace NUMINAMATH_CALUDE_apple_pear_ratio_l392_39251

theorem apple_pear_ratio (apples oranges pears : ℕ) 
  (h1 : oranges = 3 * apples) 
  (h2 : pears = 4 * oranges) : 
  apples = (1 : ℚ) / 12 * pears :=
by sorry

end NUMINAMATH_CALUDE_apple_pear_ratio_l392_39251


namespace NUMINAMATH_CALUDE_unmanned_supermarket_prices_l392_39294

/-- Represents the unit price of keychains in yuan -/
def keychain_price : ℝ := 24

/-- Represents the unit price of plush toys in yuan -/
def plush_toy_price : ℝ := 36

/-- The total number of items bought -/
def total_items : ℕ := 15

/-- The total amount spent on keychains in yuan -/
def total_keychain_cost : ℝ := 240

/-- The total amount spent on plush toys in yuan -/
def total_plush_toy_cost : ℝ := 180

theorem unmanned_supermarket_prices :
  (total_keychain_cost / keychain_price + total_plush_toy_cost / plush_toy_price = total_items) ∧
  (plush_toy_price = 1.5 * keychain_price) := by
  sorry

end NUMINAMATH_CALUDE_unmanned_supermarket_prices_l392_39294


namespace NUMINAMATH_CALUDE_equation_solution_l392_39273

theorem equation_solution (x : ℝ) : x ≠ 3 →
  (x - 7 = (4 * |x - 3|) / (x - 3)) ↔ x = 11 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l392_39273


namespace NUMINAMATH_CALUDE_simplify_expression_l392_39227

theorem simplify_expression (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 2) (h3 : x ≠ -2) :
  (x - 1 - 3 / (x + 1)) / ((x^2 - 4) / (x^2 + 2*x + 1)) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l392_39227


namespace NUMINAMATH_CALUDE_expression_evaluation_l392_39248

theorem expression_evaluation : 
  let x := Real.sqrt ((9^9 + 3^12) / (9^5 + 3^13))
  ∃ ε > 0, abs (x - 15.3) < ε ∧ ε < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l392_39248


namespace NUMINAMATH_CALUDE_compound_interest_approximation_l392_39204

/-- Approximation of compound interest using Binomial Theorem -/
theorem compound_interest_approximation
  (K : ℝ) (p : ℝ) (n : ℕ) :
  let r := p / 100
  let Kn := K * (1 + r)^n
  let approx := K * (1 + n*r + (n*(n-1)/2) * r^2 + (n*(n-1)*(n-2)/6) * r^3)
  ∃ (ε : ℝ), ε > 0 ∧ |Kn - approx| < ε * Kn :=
sorry

end NUMINAMATH_CALUDE_compound_interest_approximation_l392_39204


namespace NUMINAMATH_CALUDE_planes_parallel_if_skew_lines_parallel_l392_39281

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relation for planes and lines
variable (parallel : Plane → Plane → Prop)
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the containment relation for lines in planes
variable (contains : Plane → Line → Prop)

-- Define the skew relation for lines
variable (skew : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_if_skew_lines_parallel
  (α β : Plane) (a b : Line)
  (h1 : contains α a)
  (h2 : contains β b)
  (h3 : lineParallelPlane a β)
  (h4 : lineParallelPlane b α)
  (h5 : skew a b) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_skew_lines_parallel_l392_39281


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l392_39271

theorem angle_sum_around_point (x : ℝ) : 
  x > 0 ∧ 150 > 0 ∧ 
  x + x + 150 = 360 →
  x = 105 := by sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l392_39271


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l392_39234

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

/-- The theorem states that if the ratio of S_{3n} to S_n is constant
    for all positive integers n, then the first term of the sequence is 5/2 -/
theorem arithmetic_sequence_first_term
  (h : ∃ (c : ℚ), ∀ (n : ℕ), n > 0 → S a (3 * n) / S a n = c) :
  a = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l392_39234


namespace NUMINAMATH_CALUDE_double_angle_sine_15_degrees_l392_39297

theorem double_angle_sine_15_degrees :
  2 * Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_double_angle_sine_15_degrees_l392_39297


namespace NUMINAMATH_CALUDE_distinct_z_values_l392_39289

def is_valid_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def reverse_digits (n : ℕ) : ℕ := 
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  100 * c + 10 * b + a

def z (x : ℕ) : ℕ := Int.natAbs (x - reverse_digits x)

theorem distinct_z_values : 
  ∃ (S : Finset ℕ), (∀ x, is_valid_number x → z x ∈ S) ∧ S.card = 10 := by
sorry

end NUMINAMATH_CALUDE_distinct_z_values_l392_39289


namespace NUMINAMATH_CALUDE_square_tiles_count_l392_39286

/-- Represents a collection of triangular and square tiles. -/
structure TileCollection where
  triangles : ℕ
  squares : ℕ
  total_tiles : ℕ
  total_edges : ℕ
  tiles_sum : triangles + squares = total_tiles
  edges_sum : 3 * triangles + 4 * squares = total_edges

/-- Theorem stating that in a collection of 32 tiles with 110 edges, there are 14 square tiles. -/
theorem square_tiles_count (tc : TileCollection) 
  (h1 : tc.total_tiles = 32) 
  (h2 : tc.total_edges = 110) : 
  tc.squares = 14 := by
  sorry

#check square_tiles_count

end NUMINAMATH_CALUDE_square_tiles_count_l392_39286


namespace NUMINAMATH_CALUDE_x_minus_y_value_l392_39299

theorem x_minus_y_value (x y : ℝ) 
  (hx : |x| = 4)
  (hy : |y| = 2)
  (hxy : x * y < 0) :
  x - y = 6 ∨ x - y = -6 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l392_39299


namespace NUMINAMATH_CALUDE_largest_tank_volume_width_l392_39274

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank -/
structure GasTank where
  radius : ℝ
  height : ℝ

/-- Checks if a gas tank fits inside a crate -/
def tankFitsInCrate (tank : GasTank) (crate : CrateDimensions) : Prop :=
  (2 * tank.radius ≤ crate.length ∧ 2 * tank.radius ≤ crate.width) ∨
  (2 * tank.radius ≤ crate.length ∧ 2 * tank.radius ≤ crate.height) ∨
  (2 * tank.radius ≤ crate.width ∧ 2 * tank.radius ≤ crate.height)

/-- Theorem: The width of the crate must be 8 feet for the largest possible tank volume -/
theorem largest_tank_volume_width (x : ℝ) :
  let crate := CrateDimensions.mk 6 x 10
  let tank := GasTank.mk 4 (min (min 6 x) 10)
  tankFitsInCrate tank crate → x = 8 := by
  sorry

#check largest_tank_volume_width

end NUMINAMATH_CALUDE_largest_tank_volume_width_l392_39274


namespace NUMINAMATH_CALUDE_min_profit_is_128_l392_39267

/-- The profit function for a stationery item -/
def profit (x : ℝ) : ℝ :=
  let y := -2 * x + 60
  y * (x - 10)

/-- The theorem stating the minimum profit -/
theorem min_profit_is_128 :
  ∃ (x_min : ℝ), 15 ≤ x_min ∧ x_min ≤ 26 ∧
  ∀ (x : ℝ), 15 ≤ x → x ≤ 26 → profit x_min ≤ profit x ∧
  profit x_min = 128 :=
sorry

end NUMINAMATH_CALUDE_min_profit_is_128_l392_39267


namespace NUMINAMATH_CALUDE_f_zero_values_l392_39224

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = 2 * f x * f y

/-- The theorem stating the possible values of f(0) -/
theorem f_zero_values (f : ℝ → ℝ) (h : FunctionalEquation f) :
    f 0 = 0 ∨ f 0 = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_values_l392_39224


namespace NUMINAMATH_CALUDE_inequality_solution_l392_39256

theorem inequality_solution (x : ℝ) : (10 * x^2 + 1 < 7 * x) ∧ ((2 * x - 7) / (-3 * x + 1) > 0) ↔ x > 1/3 ∧ x < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l392_39256


namespace NUMINAMATH_CALUDE_circumscribable_with_special_area_is_inscribable_l392_39249

-- Define a quadrilateral
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  area : ℝ

-- Define the properties of being circumscribable and inscribable
def is_circumscribable (q : Quadrilateral) : Prop := sorry
def is_inscribable (q : Quadrilateral) : Prop := sorry

-- State the theorem
theorem circumscribable_with_special_area_is_inscribable (q : Quadrilateral) :
  is_circumscribable q →
  q.area = Real.sqrt (q.a * q.b * q.c * q.d) →
  is_inscribable q := by sorry

end NUMINAMATH_CALUDE_circumscribable_with_special_area_is_inscribable_l392_39249


namespace NUMINAMATH_CALUDE_point_on_right_branch_l392_39268

/-- 
Given a point P(a, b) on the hyperbola x² - 4y² = m (m ≠ 0),
if a - 2b > 0 and a + 2b > 0, then a > 0.
-/
theorem point_on_right_branch 
  (m : ℝ) (hm : m ≠ 0)
  (a b : ℝ) 
  (h_hyperbola : a^2 - 4*b^2 = m)
  (h_diff : a - 2*b > 0)
  (h_sum : a + 2*b > 0) : 
  a > 0 := by sorry

end NUMINAMATH_CALUDE_point_on_right_branch_l392_39268


namespace NUMINAMATH_CALUDE_mona_grouped_before_l392_39291

/-- Represents the game groups Mona joined -/
structure GameGroups where
  totalGroups : ℕ
  playersPerGroup : ℕ
  uniquePlayers : ℕ
  knownPlayersInOneGroup : ℕ

/-- Calculates the number of players Mona had grouped with before in a specific group -/
def playersGroupedBefore (g : GameGroups) : ℕ :=
  g.totalGroups * g.playersPerGroup - g.uniquePlayers - g.knownPlayersInOneGroup

/-- Theorem stating the number of players Mona had grouped with before in a specific group -/
theorem mona_grouped_before (g : GameGroups) 
  (h1 : g.totalGroups = 9)
  (h2 : g.playersPerGroup = 4)
  (h3 : g.uniquePlayers = 33)
  (h4 : g.knownPlayersInOneGroup = 1) :
  playersGroupedBefore g = 2 := by
    sorry

end NUMINAMATH_CALUDE_mona_grouped_before_l392_39291


namespace NUMINAMATH_CALUDE_count_pairs_equals_210_l392_39284

def count_pairs : ℕ := 
  (Finset.range 20).sum (fun a => 21 - a)

theorem count_pairs_equals_210 : count_pairs = 210 := by sorry

end NUMINAMATH_CALUDE_count_pairs_equals_210_l392_39284


namespace NUMINAMATH_CALUDE_N_subset_M_l392_39261

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {1, 4}

theorem N_subset_M : N ⊆ M := by sorry

end NUMINAMATH_CALUDE_N_subset_M_l392_39261


namespace NUMINAMATH_CALUDE_sum_of_powers_l392_39228

theorem sum_of_powers (n : ℕ) :
  n^5 + n^5 + n^5 + n^5 + n^5 = 5 * n^5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l392_39228


namespace NUMINAMATH_CALUDE_smiths_bakery_pies_l392_39230

/-- The number of pies sold by Mcgee's Bakery -/
def mcgees_pies : ℕ := 16

/-- The number of pies sold by Smith's Bakery -/
def smiths_pies : ℕ := 4 * mcgees_pies + 6

/-- Theorem stating that Smith's Bakery sold 70 pies -/
theorem smiths_bakery_pies : smiths_pies = 70 := by
  sorry

end NUMINAMATH_CALUDE_smiths_bakery_pies_l392_39230


namespace NUMINAMATH_CALUDE_paint_combinations_l392_39252

theorem paint_combinations (n m k : ℕ) (hn : n = 10) (hm : m = 3) (hk : k = 2) :
  (n.choose m) * k^m = 960 := by
  sorry

end NUMINAMATH_CALUDE_paint_combinations_l392_39252


namespace NUMINAMATH_CALUDE_clinton_shoes_count_l392_39245

/-- Clinton's wardrobe inventory problem -/
theorem clinton_shoes_count :
  ∀ (shoes belts hats : ℕ),
  shoes = 2 * belts →
  belts = hats + 2 →
  hats = 5 →
  shoes = 14 :=
by sorry

end NUMINAMATH_CALUDE_clinton_shoes_count_l392_39245


namespace NUMINAMATH_CALUDE_subletter_monthly_rent_subletter_rent_is_400_l392_39213

/-- Calculates the monthly rent for each subletter given the number of subletters,
    John's monthly rent, and John's annual profit. -/
theorem subletter_monthly_rent 
  (num_subletters : ℕ) 
  (john_monthly_rent : ℕ) 
  (john_annual_profit : ℕ) : ℕ :=
  let total_annual_rent := john_monthly_rent * 12 + john_annual_profit
  total_annual_rent / (num_subletters * 12)

/-- Proves that each subletter pays $400 per month given the specific conditions. -/
theorem subletter_rent_is_400 :
  subletter_monthly_rent 3 900 3600 = 400 := by
  sorry

end NUMINAMATH_CALUDE_subletter_monthly_rent_subletter_rent_is_400_l392_39213


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l392_39238

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def arrange_books (total : ℕ) (math_copies : ℕ) (physics_copies : ℕ) : ℕ :=
  factorial total / (factorial math_copies * factorial physics_copies)

theorem book_arrangement_proof :
  arrange_books 7 3 2 = 420 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l392_39238


namespace NUMINAMATH_CALUDE_peanut_butter_calories_value_l392_39240

/-- The number of calories in a serving of peanut butter -/
def peanut_butter_calories : ℕ := sorry

/-- The number of calories in a piece of bread -/
def bread_calories : ℕ := 100

/-- The total number of calories for breakfast -/
def total_calories : ℕ := 500

/-- The number of servings of peanut butter -/
def peanut_butter_servings : ℕ := 2

/-- The number of pieces of bread -/
def bread_pieces : ℕ := 1

theorem peanut_butter_calories_value : 
  bread_calories * bread_pieces + peanut_butter_calories * peanut_butter_servings = total_calories ∧ 
  peanut_butter_calories = 200 := by sorry

end NUMINAMATH_CALUDE_peanut_butter_calories_value_l392_39240


namespace NUMINAMATH_CALUDE_basketball_win_percentage_l392_39253

theorem basketball_win_percentage (total_games : ℕ) (first_games : ℕ) (first_wins : ℕ) (target_percentage : ℚ) : 
  total_games = 110 →
  first_games = 60 →
  first_wins = 45 →
  target_percentage = 3/4 →
  ∃ (remaining_wins : ℕ), 
    remaining_wins = 38 ∧ 
    (first_wins + remaining_wins : ℚ) / total_games = target_percentage :=
by sorry

end NUMINAMATH_CALUDE_basketball_win_percentage_l392_39253


namespace NUMINAMATH_CALUDE_min_value_problem_l392_39276

theorem min_value_problem (x y : ℝ) (h1 : x * y + 1 = 4 * x + y) (h2 : x > 1) :
  ∃ (min : ℝ), min = 27 ∧ ∀ z, z = (x + 1) * (y + 2) → z ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l392_39276


namespace NUMINAMATH_CALUDE_trig_fraction_value_l392_39270

theorem trig_fraction_value (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + Real.sin α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_fraction_value_l392_39270


namespace NUMINAMATH_CALUDE_true_discount_for_given_values_l392_39282

/-- Given a banker's discount and a sum due, calculate the true discount -/
def true_discount (bankers_discount : ℚ) (sum_due : ℚ) : ℚ :=
  bankers_discount / (1 + bankers_discount / sum_due)

/-- Theorem stating that for the given values, the true discount is 120 -/
theorem true_discount_for_given_values :
  true_discount 144 720 = 120 := by
  sorry

end NUMINAMATH_CALUDE_true_discount_for_given_values_l392_39282


namespace NUMINAMATH_CALUDE_raft_cannot_turn_l392_39295

/-- A raft is a shape with a measurable area -/
class Raft :=
  (area : ℝ)

/-- A canal is a path with a width and ability to turn -/
class Canal :=
  (width : ℝ)
  (turn_angle : ℝ)

/-- Determines if a raft can turn in a given canal -/
def can_turn (r : Raft) (c : Canal) : Prop :=
  sorry

/-- Theorem: A raft with area ≥ 2√2 cannot turn in a canal of width 1 with a 90° turn -/
theorem raft_cannot_turn (r : Raft) (c : Canal) :
  r.area ≥ 2 * Real.sqrt 2 →
  c.width = 1 →
  c.turn_angle = Real.pi / 2 →
  ¬(can_turn r c) :=
sorry

end NUMINAMATH_CALUDE_raft_cannot_turn_l392_39295


namespace NUMINAMATH_CALUDE_range_of_f_on_interval_l392_39279

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 1

-- State the theorem
theorem range_of_f_on_interval (m : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ ∧ x₂ ≤ -2 → f m x₁ > f m x₂) ∧ 
  (∀ x₁ x₂, -2 ≤ x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) →
  Set.Icc (f m 1) (f m 2) = Set.Icc (-11) 33 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_on_interval_l392_39279


namespace NUMINAMATH_CALUDE_min_value_problem1_min_value_problem2_l392_39232

theorem min_value_problem1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + y = 1) :
  2*x + 1/(3*y) ≥ (13 + 4*Real.sqrt 3) / 3 :=
sorry

theorem min_value_problem2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 1) :
  1/(2*x) + x/(y+1) ≥ 5/4 :=
sorry

end NUMINAMATH_CALUDE_min_value_problem1_min_value_problem2_l392_39232


namespace NUMINAMATH_CALUDE_cost_price_percentage_l392_39242

theorem cost_price_percentage (marked_price cost_price selling_price : ℝ) : 
  marked_price > 0 →
  cost_price > 0 →
  selling_price = marked_price * 0.9 →
  selling_price = cost_price * (1 + 20 / 700) →
  cost_price / marked_price = 0.875 := by
sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l392_39242


namespace NUMINAMATH_CALUDE_non_empty_proper_subsets_of_A_l392_39237

def A : Set ℕ := {2, 3}

theorem non_empty_proper_subsets_of_A :
  {s : Set ℕ | s ⊆ A ∧ s ≠ ∅ ∧ s ≠ A} = {{2}, {3}} := by sorry

end NUMINAMATH_CALUDE_non_empty_proper_subsets_of_A_l392_39237


namespace NUMINAMATH_CALUDE_max_value_of_fraction_difference_l392_39277

theorem max_value_of_fraction_difference (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h : 4 * a - b ≥ 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → 4 * x - y ≥ 2 → 1 / x - 1 / y ≤ 1 / 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x - y ≥ 2 ∧ 1 / x - 1 / y = 1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_difference_l392_39277


namespace NUMINAMATH_CALUDE_unwatered_bushes_l392_39208

def total_bushes : ℕ := 2006

def bushes_watered_by_vitya (n : ℕ) : ℕ := n / 2
def bushes_watered_by_anya (n : ℕ) : ℕ := n / 2
def bushes_watered_by_both : ℕ := 3

theorem unwatered_bushes :
  total_bushes - (bushes_watered_by_vitya total_bushes + bushes_watered_by_anya total_bushes - bushes_watered_by_both) = 3 := by
  sorry

end NUMINAMATH_CALUDE_unwatered_bushes_l392_39208


namespace NUMINAMATH_CALUDE_polynomial_sum_l392_39225

-- Define the polynomial P
def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) :
  P a b c d 1 = 2000 →
  P a b c d 2 = 4000 →
  P a b c d 3 = 6000 →
  P a b c d 9 + P a b c d (-5) = 12704 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l392_39225


namespace NUMINAMATH_CALUDE_cody_second_week_books_l392_39226

/-- The number of books Cody read in the second week -/
def books_second_week (total_books : ℕ) (first_week : ℕ) (after_second : ℕ) (total_weeks : ℕ) : ℕ :=
  total_books - first_week - (after_second * (total_weeks - 2))

/-- Theorem stating that Cody read 3 books in the second week -/
theorem cody_second_week_books :
  books_second_week 54 6 9 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cody_second_week_books_l392_39226


namespace NUMINAMATH_CALUDE_negation_equivalence_l392_39241

-- Define a type for polyhedra
structure Polyhedron where
  faces : Set Face

-- Define a type for faces
inductive Face
  | Triangle
  | Quadrilateral
  | Pentagon
  | Other

-- Define the original proposition
def original_proposition : Prop :=
  ∀ p : Polyhedron, ∃ f ∈ p.faces, f = Face.Triangle ∨ f = Face.Quadrilateral ∨ f = Face.Pentagon

-- Define the negation
def negation : Prop :=
  ∃ p : Polyhedron, ∀ f ∈ p.faces, f ≠ Face.Triangle ∧ f ≠ Face.Quadrilateral ∧ f ≠ Face.Pentagon

-- Theorem stating the equivalence
theorem negation_equivalence : ¬original_proposition ↔ negation := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l392_39241


namespace NUMINAMATH_CALUDE_ship_ratio_l392_39219

/-- Given the conditions of ships in a busy port, prove the ratio of sailboats to fishing boats -/
theorem ship_ratio : 
  ∀ (cruise cargo sailboats fishing : ℕ),
  cruise = 4 →
  cargo = 2 * cruise →
  sailboats = cargo + 6 →
  cruise + cargo + sailboats + fishing = 28 →
  sailboats / fishing = 7 ∧ fishing ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ship_ratio_l392_39219


namespace NUMINAMATH_CALUDE_modulo_23_equivalence_l392_39265

theorem modulo_23_equivalence (n : ℤ) : 0 ≤ n ∧ n < 23 ∧ -207 ≡ n [ZMOD 23] → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_modulo_23_equivalence_l392_39265


namespace NUMINAMATH_CALUDE_inequality_proof_l392_39246

theorem inequality_proof (x y z : ℝ) (hx : x ∈ Set.Icc 0 1) (hy : y ∈ Set.Icc 0 1) (hz : z ∈ Set.Icc 0 1) :
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l392_39246


namespace NUMINAMATH_CALUDE_isosceles_triangle_conditions_l392_39216

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that each of the following conditions implies that the triangle is isosceles. -/
theorem isosceles_triangle_conditions (a b c A B C : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C)
  (h_angle_sum : A + B + C = Real.pi) : 
  (a * Real.cos B = b * Real.cos A → a = b ∨ b = c ∨ a = c) ∧ 
  (Real.cos B * Real.cos C = (1 - Real.cos A) / 2 → a = b ∨ b = c ∨ a = c) ∧
  (a / Real.sin B + b / Real.sin A ≤ 2 * c → a = b ∨ b = c ∨ a = c) := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_conditions_l392_39216


namespace NUMINAMATH_CALUDE_problem_statement_l392_39202

theorem problem_statement (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l392_39202


namespace NUMINAMATH_CALUDE_min_sum_of_product_l392_39272

theorem min_sum_of_product (a b : ℤ) (h : a * b = 144) : 
  ∀ x y : ℤ, x * y = 144 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 144 ∧ a₀ + b₀ = -145 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l392_39272


namespace NUMINAMATH_CALUDE_sin_cos_sum_14_16_l392_39255

theorem sin_cos_sum_14_16 : 
  Real.sin (14 * π / 180) * Real.cos (16 * π / 180) + 
  Real.cos (14 * π / 180) * Real.sin (16 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_14_16_l392_39255


namespace NUMINAMATH_CALUDE_target_hitting_probability_l392_39209

theorem target_hitting_probability : 
  let p_single_hit : ℚ := 1/2
  let total_shots : ℕ := 7
  let total_hits : ℕ := 3
  let consecutive_hits : ℕ := 2

  -- Probability of exactly 3 hits out of 7 shots
  let p_total_hits : ℚ := (Nat.choose total_shots total_hits : ℚ) * p_single_hit ^ total_shots

  -- Number of ways to arrange 2 consecutive hits out of 3 in 7 shots
  let arrangements : ℕ := Nat.descFactorial (total_shots - consecutive_hits) consecutive_hits

  -- Final probability
  (arrangements : ℚ) * p_single_hit ^ total_shots = 5/32 :=
by sorry

end NUMINAMATH_CALUDE_target_hitting_probability_l392_39209


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l392_39223

theorem unique_modular_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 6 ∧ n ≡ 12345 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l392_39223


namespace NUMINAMATH_CALUDE_shortest_player_height_l392_39239

/-- Given the height of the tallest player and the difference in height between
    the tallest and shortest players, calculate the height of the shortest player. -/
theorem shortest_player_height
  (tallest_height : ℝ)
  (height_difference : ℝ)
  (h1 : tallest_height = 77.75)
  (h2 : height_difference = 9.5) :
  tallest_height - height_difference = 68.25 := by
  sorry

#check shortest_player_height

end NUMINAMATH_CALUDE_shortest_player_height_l392_39239


namespace NUMINAMATH_CALUDE_set_operations_l392_39287

-- Define the sets A, B, and C
def A : Set ℤ := {x : ℤ | -6 ≤ x ∧ x ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

-- Define the theorem
theorem set_operations :
  (A ∩ (B ∩ C) = {3}) ∧
  (A ∩ (A \ (B ∪ C)) = {-6, -5, -4, -3, -2, -1, 0}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l392_39287


namespace NUMINAMATH_CALUDE_line_slope_l392_39207

theorem line_slope (x y : ℝ) : 
  (x / 4 - y / 3 = 1) → (∃ b : ℝ, y = (3/4) * x + b) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l392_39207


namespace NUMINAMATH_CALUDE_halfway_point_between_fractions_l392_39231

theorem halfway_point_between_fractions :
  (1 / 12 + 1 / 15) / 2 = 3 / 40 := by
  sorry

end NUMINAMATH_CALUDE_halfway_point_between_fractions_l392_39231


namespace NUMINAMATH_CALUDE_max_profits_l392_39220

def total_profit (x : ℕ+) : ℚ := -x^2 + 18*x - 36

def average_annual_profit (x : ℕ+) : ℚ := (total_profit x) / x

theorem max_profits :
  (∃ (x_max : ℕ+), ∀ (x : ℕ+), total_profit x ≤ total_profit x_max ∧ 
    total_profit x_max = 45) ∧
  (∃ (x_avg_max : ℕ+), ∀ (x : ℕ+), average_annual_profit x ≤ average_annual_profit x_avg_max ∧ 
    average_annual_profit x_avg_max = 6) :=
by sorry

end NUMINAMATH_CALUDE_max_profits_l392_39220


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l392_39201

theorem square_area_equal_perimeter_triangle (a b c : ℝ) (h_triangle : a = 7.2 ∧ b = 9.5 ∧ c = 11.3) :
  let triangle_perimeter := a + b + c
  let square_side := triangle_perimeter / 4
  square_side ^ 2 = 49 := by
sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_triangle_l392_39201


namespace NUMINAMATH_CALUDE_similar_polygons_ratio_l392_39215

theorem similar_polygons_ratio (A₁ A₂ P₁ P₂ : ℝ) (h_positive : A₁ > 0 ∧ A₂ > 0 ∧ P₁ > 0 ∧ P₂ > 0) :
  A₁ / A₂ = 5 → P₁ / P₂ = m → 5 / m = Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_similar_polygons_ratio_l392_39215


namespace NUMINAMATH_CALUDE_reflection_point_properties_l392_39275

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A concave spherical mirror -/
structure SphericalMirror where
  radius : ℝ
  center : Point

/-- The reflection point on a spherical mirror -/
def reflection_point (mirror : SphericalMirror) (A B : Point) : Point :=
  sorry

/-- Theorem: The reflection point satisfies the sphere equation and reflection equation -/
theorem reflection_point_properties (mirror : SphericalMirror) (A B : Point) :
  let X := reflection_point mirror A B
  (X.x^2 + X.y^2 = mirror.radius^2) ∧
  ((A.x * B.y + B.x * A.y) * (X.x^2 - X.y^2) - 
   2 * (A.x * B.x - A.y * B.y) * X.x * X.y + 
   mirror.radius^2 * ((A.x + B.x) * X.y - (A.y + B.y) * X.x) = 0) := by
  sorry


end NUMINAMATH_CALUDE_reflection_point_properties_l392_39275


namespace NUMINAMATH_CALUDE_sales_function_satisfies_data_profit_192_implies_price_18_max_profit_at_19_l392_39283

-- Define the linear function
def sales_function (x : ℝ) : ℝ := -2 * x + 60

-- Define the profit function
def profit_function (x : ℝ) : ℝ := (x - 10) * (sales_function x)

-- Theorem 1: The sales function satisfies the given data points
theorem sales_function_satisfies_data : 
  sales_function 12 = 36 ∧ sales_function 13 = 34 := by sorry

-- Theorem 2: When the daily profit is 192 yuan, the selling price is 18 yuan
theorem profit_192_implies_price_18 :
  ∀ x, 10 ≤ x ∧ x ≤ 19 → profit_function x = 192 → x = 18 := by sorry

-- Theorem 3: The maximum daily profit occurs at a selling price of 19 yuan and equals 198 yuan
theorem max_profit_at_19 :
  (∀ x, 10 ≤ x ∧ x ≤ 19 → profit_function x ≤ profit_function 19) ∧
  profit_function 19 = 198 := by sorry

end NUMINAMATH_CALUDE_sales_function_satisfies_data_profit_192_implies_price_18_max_profit_at_19_l392_39283


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l392_39244

/-- Given a store's pricing strategy and profit margin, 
    prove the initial markup percentage. -/
theorem initial_markup_percentage 
  (initial_cost : ℝ) 
  (markup_percentage : ℝ) 
  (new_year_markup : ℝ) 
  (february_discount : ℝ) 
  (february_profit : ℝ) 
  (h1 : new_year_markup = 0.25) 
  (h2 : february_discount = 0.09) 
  (h3 : february_profit = 0.365) : 
  1.365 = (1 + markup_percentage) * 1.25 * 0.91 := by
  sorry

#check initial_markup_percentage

end NUMINAMATH_CALUDE_initial_markup_percentage_l392_39244


namespace NUMINAMATH_CALUDE_pencil_count_l392_39260

/-- The number of pencils Mitchell has -/
def mitchell_pencils : ℕ := 30

/-- The number of pencils Antonio has -/
def antonio_pencils : ℕ := mitchell_pencils - (mitchell_pencils * 20 / 100)

/-- The number of pencils Elizabeth has -/
def elizabeth_pencils : ℕ := 2 * antonio_pencils

/-- The total number of pencils Mitchell, Antonio, and Elizabeth have together -/
def total_pencils : ℕ := mitchell_pencils + antonio_pencils + elizabeth_pencils

theorem pencil_count : total_pencils = 102 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l392_39260


namespace NUMINAMATH_CALUDE_geometric_sum_n_terms_l392_39269

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_n_terms (a r : ℚ) (n : ℕ) (h1 : a = 1/3) (h2 : r = 1/3) :
  geometric_sum a r n = 80/243 ↔ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_n_terms_l392_39269


namespace NUMINAMATH_CALUDE_hexagon_coin_rotations_l392_39288

/-- Represents a configuration of coins on a table -/
structure CoinConfiguration where
  num_coins : Nat
  is_closed_chain : Bool

/-- Represents the motion of a rolling coin -/
structure RollingCoin where
  rotations : Nat

/-- Calculates the number of rotations a coin makes when rolling around a hexagon of coins -/
def calculate_rotations (config : CoinConfiguration) : RollingCoin :=
  sorry

/-- Theorem: A coin rolling around a hexagon of coins makes 4 complete rotations -/
theorem hexagon_coin_rotations :
  ∀ (config : CoinConfiguration),
    config.num_coins = 6 ∧ config.is_closed_chain →
    (calculate_rotations config).rotations = 4 :=
  sorry

end NUMINAMATH_CALUDE_hexagon_coin_rotations_l392_39288


namespace NUMINAMATH_CALUDE_polynomial_equality_conditions_l392_39218

theorem polynomial_equality_conditions (A B C p q : ℝ) :
  (∀ x : ℝ, A * x^4 + B * x^2 + C = A * (x^2 + p * x + q) * (x^2 - p * x + q)) →
  (A * (2 * q - p^2) = B ∧ A * q^2 = C) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_conditions_l392_39218


namespace NUMINAMATH_CALUDE_john_supermarket_spending_l392_39247

theorem john_supermarket_spending : 
  ∀ (total : ℚ),
  (1 / 2 : ℚ) * total + (1 / 3 : ℚ) * total + (1 / 10 : ℚ) * total + 5 = total →
  total = 75 := by
sorry

end NUMINAMATH_CALUDE_john_supermarket_spending_l392_39247


namespace NUMINAMATH_CALUDE_vector_magnitude_equation_l392_39210

theorem vector_magnitude_equation (k : ℝ) : 
  ‖k • (⟨3, -4⟩ : ℝ × ℝ) + ⟨5, -6⟩‖ = 5 * Real.sqrt 5 ↔ k = 17/25 ∨ k = -19/5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_equation_l392_39210


namespace NUMINAMATH_CALUDE_savings_growth_l392_39292

/-- Given an initial savings amount that grows over time with a fixed annual interest rate,
    this theorem states the relationship between the initial amount, final amount,
    time period, and the annual interest rate. -/
theorem savings_growth (initial_amount final_amount : ℝ) (years : ℕ) (rate : ℝ) :
  initial_amount > 0 →
  final_amount > initial_amount →
  years > 0 →
  rate > 0 →
  rate < 1 →
  final_amount = initial_amount + initial_amount * years * rate →
  initial_amount = 3000 →
  final_amount = 3243 →
  years = 3 →
  rate = x / 100 →
  3000 + 3000 * 3 * (x / 100) = 3243 :=
by sorry

end NUMINAMATH_CALUDE_savings_growth_l392_39292


namespace NUMINAMATH_CALUDE_jessica_attended_games_l392_39206

theorem jessica_attended_games (total_games missed_games : ℕ) 
  (h1 : total_games = 6)
  (h2 : missed_games = 4) :
  total_games - missed_games = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_attended_games_l392_39206


namespace NUMINAMATH_CALUDE_geometry_propositions_l392_39258

theorem geometry_propositions (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) : 
  (p₁ ∧ p₄) ∧ ¬(p₁ ∧ p₂) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l392_39258


namespace NUMINAMATH_CALUDE_coin_stack_order_l392_39243

-- Define the type for coins
inductive Coin | A | B | C | D | E

-- Define the covering relation
def covers (x y : Coin) : Prop := sorry

-- Define the partial covering relation
def partially_covers (x y : Coin) : Prop := sorry

-- Define the order relation
def above (x y : Coin) : Prop := sorry

-- State the theorem
theorem coin_stack_order :
  (partially_covers Coin.A Coin.B) →
  (covers Coin.C Coin.A) →
  (covers Coin.C Coin.D) →
  (covers Coin.D Coin.B) →
  (¬ covers Coin.D Coin.E) →
  (covers Coin.C Coin.E) →
  (∀ x, ¬ covers Coin.E x) →
  (above Coin.C Coin.E) ∧
  (above Coin.E Coin.A) ∧
  (above Coin.E Coin.D) ∧
  (above Coin.A Coin.B) ∧
  (above Coin.D Coin.B) :=
by sorry

end NUMINAMATH_CALUDE_coin_stack_order_l392_39243


namespace NUMINAMATH_CALUDE_monic_quartic_with_specific_roots_l392_39221

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 10*x^3 + 25*x^2 + 2*x - 12

-- State the theorem
theorem monic_quartic_with_specific_roots :
  -- The polynomial is monic
  (∀ x, p x = x^4 - 10*x^3 + 25*x^2 + 2*x - 12) ∧
  -- The polynomial has rational coefficients
  (∃ a b c d : ℚ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d) ∧
  -- 3 + √5 is a root
  p (3 + Real.sqrt 5) = 0 ∧
  -- 2 - √7 is a root
  p (2 - Real.sqrt 7) = 0 :=
by sorry


end NUMINAMATH_CALUDE_monic_quartic_with_specific_roots_l392_39221


namespace NUMINAMATH_CALUDE_cube_sum_given_sum_and_diff_l392_39257

theorem cube_sum_given_sum_and_diff (a b : ℝ) (h1 : a + b = 12) (h2 : a - b = 4) :
  a^3 + b^3 = 1344 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_given_sum_and_diff_l392_39257


namespace NUMINAMATH_CALUDE_remaining_water_fills_glasses_l392_39217

theorem remaining_water_fills_glasses (total_water : ℕ) (glass_5oz : ℕ) (glass_8oz : ℕ) (glass_4oz : ℕ) :
  total_water = 122 →
  glass_5oz = 6 →
  glass_8oz = 4 →
  glass_4oz * 4 = total_water - (glass_5oz * 5 + glass_8oz * 8) →
  glass_4oz = 15 := by
sorry

end NUMINAMATH_CALUDE_remaining_water_fills_glasses_l392_39217


namespace NUMINAMATH_CALUDE_toy_poodle_height_l392_39205

/-- The height of the toy poodle given the heights of other poodle types -/
theorem toy_poodle_height (h_standard : ℕ) (h_mini : ℕ) (h_toy : ℕ)
  (standard_mini : h_standard = h_mini + 8)
  (mini_toy : h_mini = h_toy + 6)
  (standard_height : h_standard = 28) :
  h_toy = 14 := by
  sorry

end NUMINAMATH_CALUDE_toy_poodle_height_l392_39205


namespace NUMINAMATH_CALUDE_invalid_domain_l392_39266

def f (x : ℝ) : ℝ := x^2

def N : Set ℝ := {1, 2}

theorem invalid_domain : ¬(∀ x ∈ ({1, Real.sqrt 2, 2} : Set ℝ), f x ∈ N) := by
  sorry

end NUMINAMATH_CALUDE_invalid_domain_l392_39266


namespace NUMINAMATH_CALUDE_remainder_is_18_l392_39203

/-- A cubic polynomial p(x) with coefficients a and b. -/
def p (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + a * x^2 + b * x + 12

/-- The theorem stating that the remainder when p(x) is divided by x-1 is 18. -/
theorem remainder_is_18 (a b : ℝ) :
  (x + 2 ∣ p a b x) → (x - 3 ∣ p a b x) → p a b 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_is_18_l392_39203


namespace NUMINAMATH_CALUDE_special_sequence_sum_property_l392_39229

/-- A sequence of pairwise distinct nonnegative integers satisfying the given conditions -/
def SpecialSequence (b : ℕ → ℕ) : Prop :=
  (∀ i j, i ≠ j → b i ≠ b j) ∧ 
  (b 0 = 0) ∧ 
  (∀ n > 0, b n < 2 * n)

/-- The main theorem -/
theorem special_sequence_sum_property (b : ℕ → ℕ) (h : SpecialSequence b) :
  ∀ m : ℕ, ∃ k ℓ : ℕ, b k + b ℓ = m := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_sum_property_l392_39229


namespace NUMINAMATH_CALUDE_remainder_4536_div_32_l392_39236

theorem remainder_4536_div_32 : 4536 % 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4536_div_32_l392_39236


namespace NUMINAMATH_CALUDE_escalator_time_l392_39290

/-- Time taken for a person to cover the length of a moving escalator -/
theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) 
  (h1 : escalator_speed = 15)
  (h2 : person_speed = 5)
  (h3 : escalator_length = 180) :
  escalator_length / (escalator_speed + person_speed) = 9 := by
  sorry

end NUMINAMATH_CALUDE_escalator_time_l392_39290


namespace NUMINAMATH_CALUDE_cathys_final_balance_l392_39222

def cathys_money (initial_balance dad_contribution : ℕ) : ℕ :=
  initial_balance + dad_contribution + 2 * dad_contribution

theorem cathys_final_balance :
  cathys_money 12 25 = 87 :=
by sorry

end NUMINAMATH_CALUDE_cathys_final_balance_l392_39222


namespace NUMINAMATH_CALUDE_cyclic_inequality_l392_39211

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + a*b + b^2)) + (b^3 / (b^2 + b*c + c^2)) + (c^3 / (c^2 + a*c + a^2)) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l392_39211


namespace NUMINAMATH_CALUDE_student_arrangements_l392_39285

/-- The number of students in the row -/
def n : ℕ := 7

/-- The number of arrangements where A and B must stand together -/
def arrangements_together : ℕ := 1440

/-- The number of arrangements where A is not at the head and B is not at the end -/
def arrangements_not_head_end : ℕ := 3720

/-- The number of arrangements where there is exactly one person between A and B -/
def arrangements_one_between : ℕ := 1200

/-- Theorem stating the correct number of arrangements for each situation -/
theorem student_arrangements :
  (arrangements_together = 1440) ∧
  (arrangements_not_head_end = 3720) ∧
  (arrangements_one_between = 1200) := by sorry

end NUMINAMATH_CALUDE_student_arrangements_l392_39285


namespace NUMINAMATH_CALUDE_olive_oil_price_increase_l392_39235

def highest_price : ℝ := 24
def lowest_price : ℝ := 16

theorem olive_oil_price_increase :
  (highest_price - lowest_price) / lowest_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_olive_oil_price_increase_l392_39235


namespace NUMINAMATH_CALUDE_friday_to_thursday_ratio_l392_39298

def thursday_sales : ℝ := 210
def saturday_sales : ℝ := 150
def average_daily_sales : ℝ := 260

theorem friday_to_thursday_ratio :
  let total_sales := average_daily_sales * 3
  let friday_sales := total_sales - thursday_sales - saturday_sales
  friday_sales / thursday_sales = 2 := by sorry

end NUMINAMATH_CALUDE_friday_to_thursday_ratio_l392_39298


namespace NUMINAMATH_CALUDE_expression_simplification_l392_39214

theorem expression_simplification (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  ((2*x + y) * (2*x - y) - (2*x - y)^2 - y*(x - 2*y)) / (2*x) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l392_39214


namespace NUMINAMATH_CALUDE_hexagon_sixth_angle_l392_39280

/-- The sum of angles in a hexagon -/
def hexagon_angle_sum : ℝ := 720

/-- The given angles in the hexagon -/
def given_angles : List ℝ := [150, 110, 120, 130, 100]

/-- Theorem: In a hexagon where five angles are 150°, 110°, 120°, 130°, and 100°, 
    the measure of the sixth angle is 110°. -/
theorem hexagon_sixth_angle : 
  hexagon_angle_sum - (given_angles.sum) = 110 := by sorry

end NUMINAMATH_CALUDE_hexagon_sixth_angle_l392_39280


namespace NUMINAMATH_CALUDE_polynomial_simplification_l392_39212

theorem polynomial_simplification (x : ℝ) :
  (3 * x^2 + 8 * x - 5) - (2 * x^2 + 3 * x - 15) = x^2 + 5 * x + 10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l392_39212


namespace NUMINAMATH_CALUDE_range_of_a_l392_39250

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 4}
def B : Set ℝ := {x : ℝ | 3 * x - 1 < x + 5}
def C (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

-- Define the complement of A with respect to ℝ
def complementA : Set ℝ := {x : ℝ | x ≤ 1 ∨ x ≥ 4}

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (complementA ∩ C a = C a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l392_39250


namespace NUMINAMATH_CALUDE_square_of_1023_l392_39263

theorem square_of_1023 : 1023^2 = 1046529 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1023_l392_39263


namespace NUMINAMATH_CALUDE_project_distribution_count_l392_39262

/-- The number of ways to distribute 8 projects among 4 companies -/
def distribute_projects : ℕ :=
  -- Total projects
  let total := 8
  -- Projects for each company
  let company_A := 3
  let company_B := 1
  let company_C := 2
  let company_D := 2
  -- The actual calculation would go here
  1680

/-- Theorem stating that the number of ways to distribute the projects is 1680 -/
theorem project_distribution_count : distribute_projects = 1680 := by
  sorry

end NUMINAMATH_CALUDE_project_distribution_count_l392_39262


namespace NUMINAMATH_CALUDE_condition_for_inequality_l392_39293

theorem condition_for_inequality (a b : ℝ) : 
  let x := a^2 * b^2 + 5
  let y := 2*a*b - a^2 - 4*a
  x > y → a*b ≠ 1 ∨ a ≠ -2 := by
sorry

end NUMINAMATH_CALUDE_condition_for_inequality_l392_39293


namespace NUMINAMATH_CALUDE_greatest_number_with_odd_factors_under_200_l392_39233

theorem greatest_number_with_odd_factors_under_200 :
  ∀ n : ℕ, n < 200 → (∃ k : ℕ, n = k^2) →
  ∀ m : ℕ, m < 200 → (∃ l : ℕ, m = l^2) → m ≤ n →
  n = 196 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_odd_factors_under_200_l392_39233


namespace NUMINAMATH_CALUDE_mitzi_bowling_score_l392_39200

/-- Proves that given three bowlers with an average score of 106, where one bowler scores 120 
    and another scores 85, the third bowler's score must be 113. -/
theorem mitzi_bowling_score (average_score gretchen_score beth_score : ℕ) 
    (h1 : average_score = 106)
    (h2 : gretchen_score = 120)
    (h3 : beth_score = 85) : 
  ∃ mitzi_score : ℕ, mitzi_score = 113 ∧ 
    (gretchen_score + beth_score + mitzi_score) / 3 = average_score :=
by
  sorry


end NUMINAMATH_CALUDE_mitzi_bowling_score_l392_39200
