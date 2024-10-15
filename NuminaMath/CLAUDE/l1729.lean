import Mathlib

namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1729_172945

/-- The largest possible angle in a triangle with two sides of length 2 and the third side greater than 4 --/
theorem largest_angle_in_special_triangle :
  ∀ (a b c : ℝ) (C : ℝ),
    a = 2 →
    b = 2 →
    c > 4 →
    C = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) →
    ∀ ε > 0, C < 180 - ε := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1729_172945


namespace NUMINAMATH_CALUDE_equation_solution_l1729_172905

theorem equation_solution : ∃ (x₁ x₂ : ℝ), x₁ = -2 ∧ x₂ = 3 ∧
  ∀ x : ℝ, (x + 2)^2 - 5*(x + 2) = 0 ↔ x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1729_172905


namespace NUMINAMATH_CALUDE_cookout_2006_attendance_l1729_172954

def cookout_2004 : ℕ := 60

def cookout_2005 : ℕ := cookout_2004 / 2

def cookout_2006 : ℕ := (cookout_2005 * 2) / 3

theorem cookout_2006_attendance : cookout_2006 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cookout_2006_attendance_l1729_172954


namespace NUMINAMATH_CALUDE_milk_storage_calculation_l1729_172970

/-- Calculates the final amount of milk in a storage tank given initial amount,
    pumping out rate and duration, and adding rate and duration. -/
def final_milk_amount (initial : ℝ) (pump_rate : ℝ) (pump_duration : ℝ) 
                       (add_rate : ℝ) (add_duration : ℝ) : ℝ :=
  initial - pump_rate * pump_duration + add_rate * add_duration

/-- Theorem stating that given the specific conditions from the problem,
    the final amount of milk in the storage tank is 28,980 gallons. -/
theorem milk_storage_calculation :
  final_milk_amount 30000 2880 4 1500 7 = 28980 := by
  sorry

end NUMINAMATH_CALUDE_milk_storage_calculation_l1729_172970


namespace NUMINAMATH_CALUDE_blue_markers_count_l1729_172912

def total_markers : ℕ := 3343
def red_markers : ℕ := 2315

theorem blue_markers_count : total_markers - red_markers = 1028 := by
  sorry

end NUMINAMATH_CALUDE_blue_markers_count_l1729_172912


namespace NUMINAMATH_CALUDE_quadratic_root_l1729_172941

theorem quadratic_root (a b c : ℝ) (h1 : 4 * a - 2 * b + c = 0) (h2 : a ≠ 0) :
  a * (-2)^2 + b * (-2) + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_l1729_172941


namespace NUMINAMATH_CALUDE_square_side_length_l1729_172935

theorem square_side_length (perimeter : ℝ) (h : perimeter = 28) : 
  perimeter / 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1729_172935


namespace NUMINAMATH_CALUDE_natalia_crates_l1729_172993

theorem natalia_crates (novels : ℕ) (comics : ℕ) (documentaries : ℕ) (albums : ℕ) (crate_capacity : ℕ) :
  novels = 145 →
  comics = 271 →
  documentaries = 419 →
  albums = 209 →
  crate_capacity = 9 →
  (novels + comics + documentaries + albums + crate_capacity - 1) / crate_capacity = 116 :=
by sorry

end NUMINAMATH_CALUDE_natalia_crates_l1729_172993


namespace NUMINAMATH_CALUDE_evenBlueFaceCubesFor6x3x2_l1729_172917

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of cubes with an even number of blue faces -/
def evenBlueFaceCubes (b : Block) : ℕ :=
  let edgeCubes := 4 * (b.length + b.width + b.height - 6)
  let internalCubes := (b.length - 2) * (b.width - 2) * (b.height - 2)
  edgeCubes + internalCubes

/-- The main theorem stating that a 6x3x2 block has 20 cubes with an even number of blue faces -/
theorem evenBlueFaceCubesFor6x3x2 : 
  evenBlueFaceCubes { length := 6, width := 3, height := 2 } = 20 := by
  sorry

end NUMINAMATH_CALUDE_evenBlueFaceCubesFor6x3x2_l1729_172917


namespace NUMINAMATH_CALUDE_computer_price_decrease_l1729_172984

/-- The price of a computer after a certain number of years, given an initial price and a constant rate of decrease every two years. -/
def price_after_years (initial_price : ℝ) (decrease_rate : ℝ) (years : ℕ) : ℝ :=
  initial_price * (1 - decrease_rate) ^ (years / 2)

/-- Theorem stating that a computer with an initial price of 8100 yuan, decreasing by one-third every two years, will cost 2400 yuan after 6 years. -/
theorem computer_price_decrease (initial_price : ℝ) (years : ℕ) :
  initial_price = 8100 →
  years = 6 →
  price_after_years initial_price (1/3) years = 2400 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_decrease_l1729_172984


namespace NUMINAMATH_CALUDE_games_per_month_is_seven_l1729_172951

/-- Represents the number of baseball games in a season. -/
def games_per_season : ℕ := 14

/-- Represents the number of months in a season. -/
def months_per_season : ℕ := 2

/-- Calculates the number of baseball games played in a month. -/
def games_per_month : ℕ := games_per_season / months_per_season

/-- Theorem stating that the number of baseball games played in a month is 7. -/
theorem games_per_month_is_seven : games_per_month = 7 := by sorry

end NUMINAMATH_CALUDE_games_per_month_is_seven_l1729_172951


namespace NUMINAMATH_CALUDE_mixed_fraction_product_l1729_172949

theorem mixed_fraction_product (X Y : ℕ) : 
  (5 + 1 / X) * (Y + 1 / 2) = 43 → X = 17 ∧ Y = 8 :=
sorry

end NUMINAMATH_CALUDE_mixed_fraction_product_l1729_172949


namespace NUMINAMATH_CALUDE_ant_walk_probability_l1729_172926

/-- A point on a 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- The ant's walk on the lattice -/
def AntWalk :=
  { p : LatticePoint // |p.x| + |p.y| ≥ 2 }

/-- Probability measure on the ant's walk -/
noncomputable def ProbMeasure : Type := AntWalk → ℝ

/-- The starting point of the ant -/
def start : LatticePoint := ⟨1, 0⟩

/-- The target end point -/
def target : LatticePoint := ⟨1, 1⟩

/-- Adjacent points are those that differ by 1 in exactly one coordinate -/
def adjacent (p q : LatticePoint) : Prop :=
  (abs (p.x - q.x) + abs (p.y - q.y) = 1)

/-- The probability measure satisfies the uniform distribution on adjacent points -/
axiom uniform_distribution (μ : ProbMeasure) (p : LatticePoint) :
  ∀ q : LatticePoint, adjacent p q → μ ⟨q, sorry⟩ = (1 : ℝ) / 4

/-- The main theorem: probability of reaching (1,1) from (1,0) is 7/24 -/
theorem ant_walk_probability (μ : ProbMeasure) :
  μ ⟨target, sorry⟩ = 7 / 24 := by sorry

end NUMINAMATH_CALUDE_ant_walk_probability_l1729_172926


namespace NUMINAMATH_CALUDE_speed_of_current_l1729_172950

/-- Calculates the speed of the current given the rowing speed in still water,
    distance covered downstream, and time taken. -/
theorem speed_of_current
  (rowing_speed : ℝ)
  (distance : ℝ)
  (time : ℝ)
  (h1 : rowing_speed = 120)
  (h2 : distance = 0.5)
  (h3 : time = 9.99920006399488 / 3600) :
  rowing_speed + (distance / time - rowing_speed) = 180 :=
by sorry

#check speed_of_current

end NUMINAMATH_CALUDE_speed_of_current_l1729_172950


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1729_172921

/-- The complex number i(2-i) is located in the first quadrant of the complex plane. -/
theorem complex_number_quadrant : ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ Complex.I * (2 - Complex.I) = Complex.mk x y := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1729_172921


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l1729_172940

-- Define the triangle ABC
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (0, 6)

-- Define the line parallel to BC passing through A
def line_parallel_BC (x y : ℝ) : Prop :=
  x + 2*y - 8 = 0

-- Define the lines passing through B equidistant from A and C
def line_equidistant_1 (x y : ℝ) : Prop :=
  3*x - 2*y - 4 = 0

def line_equidistant_2 (x y : ℝ) : Prop :=
  3*x + 2*y - 44 = 0

theorem triangle_ABC_properties :
  (∀ x y, line_parallel_BC x y ↔ (x - A.1) * (C.2 - B.2) = (y - A.2) * (C.1 - B.1)) ∧
  (∀ x y, (line_equidistant_1 x y ∨ line_equidistant_2 x y) ↔
    ((x - A.1)^2 + (y - A.2)^2 = (x - C.1)^2 + (y - C.2)^2 ∧ x = B.1 ∧ y = B.2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l1729_172940


namespace NUMINAMATH_CALUDE_min_people_for_company2_cheaper_l1729_172953

/-- Company 1's pricing function -/
def company1_cost (people : ℕ) : ℕ := 150 + 18 * people

/-- Company 2's pricing function -/
def company2_cost (people : ℕ) : ℕ := 250 + 15 * people

/-- Theorem stating the minimum number of people for Company 2 to be cheaper -/
theorem min_people_for_company2_cheaper :
  (company2_cost 34 < company1_cost 34) ∧
  (company1_cost 33 ≤ company2_cost 33) := by
  sorry

end NUMINAMATH_CALUDE_min_people_for_company2_cheaper_l1729_172953


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1729_172922

/-- Given a geometric sequence {a_n} where a₁ = 1 and a₅ = 81, prove that a₃ = 9 -/
theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_a1 : a 1 = 1) 
  (h_a5 : a 5 = 81) : 
  a 3 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1729_172922


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_b_terms_l1729_172995

def b (n : ℕ) : ℕ := n.factorial + 2^n + n

theorem max_gcd_consecutive_b_terms :
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (b n) (b (n + 1)) ≤ k ∧
  ∃ (m : ℕ), Nat.gcd (b m) (b (m + 1)) = k ∧
  k = 2 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_b_terms_l1729_172995


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1729_172943

theorem inequality_solution_set (x : ℝ) :
  (Set.Ioo (-1/5 : ℝ) (8/15 : ℝ)) = {x | (7/30 : ℝ) + |x - 13/60| < 11/20} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1729_172943


namespace NUMINAMATH_CALUDE_fraction_simplification_l1729_172939

theorem fraction_simplification : (1922^2 - 1913^2) / (1930^2 - 1905^2) = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1729_172939


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l1729_172990

/-- Given a hyperbola (x^2 / a^2) - (y^2 / 81) = 1 with a > 0, if one of its asymptotes is y = 3x, then a = 3 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / 81 = 1 → ∃ k : ℝ, y = k * x ∧ |k| = 9 / a) →
  (∃ x y : ℝ, x^2 / a^2 - y^2 / 81 = 1 ∧ y = 3 * x) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l1729_172990


namespace NUMINAMATH_CALUDE_sum_of_roots_l1729_172900

theorem sum_of_roots (k : ℝ) (a₁ a₂ : ℝ) (h₁ : a₁ ≠ a₂) 
  (h₂ : 5 * a₁^2 + k * a₁ - 2 = 0) (h₃ : 5 * a₂^2 + k * a₂ - 2 = 0) : 
  a₁ + a₂ = -k / 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1729_172900


namespace NUMINAMATH_CALUDE_coin_arrangement_strategy_exists_l1729_172902

/-- Represents a strategy for arranging coins by weight --/
structure CoinArrangementStrategy where
  /-- Function that decides which coins to compare at each step --/
  compareCoins : ℕ → (ℕ × ℕ)
  /-- Maximum number of comparisons needed --/
  maxComparisons : ℕ

/-- Represents the expected number of comparisons for a strategy --/
def expectedComparisons (strategy : CoinArrangementStrategy) : ℚ :=
  sorry

/-- There exists a strategy to arrange 4 coins with expected comparisons less than 4.8 --/
theorem coin_arrangement_strategy_exists :
  ∃ (strategy : CoinArrangementStrategy),
    strategy.maxComparisons ≤ 4 ∧ expectedComparisons strategy < 24/5 := by
  sorry

end NUMINAMATH_CALUDE_coin_arrangement_strategy_exists_l1729_172902


namespace NUMINAMATH_CALUDE_probability_theorem_l1729_172906

/-- Probability of reaching point (0, n) for a particle with given movement rules -/
def probability_reach_n (n : ℕ) : ℚ :=
  2/3 + 1/12 * (1 - (-1/3)^(n-1))

/-- Movement rules for the particle -/
structure MovementRules where
  prob_move_1 : ℚ := 2/3
  prob_move_2 : ℚ := 1/3
  vector_1 : Fin 2 → ℤ := ![0, 1]
  vector_2 : Fin 2 → ℤ := ![0, 2]

theorem probability_theorem (n : ℕ) (rules : MovementRules) :
  probability_reach_n n =
  2/3 + 1/12 * (1 - (-1/3)^(n-1)) :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l1729_172906


namespace NUMINAMATH_CALUDE_factors_of_180_l1729_172919

/-- The number of distinct positive factors of 180 -/
def num_factors_180 : ℕ := sorry

/-- Theorem stating that the number of distinct positive factors of 180 is 18 -/
theorem factors_of_180 : num_factors_180 = 18 := by sorry

end NUMINAMATH_CALUDE_factors_of_180_l1729_172919


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1729_172929

/-- Given the compound interest for the second and third years, calculate the interest rate. -/
theorem interest_rate_calculation (CI2 CI3 : ℝ) (h1 : CI2 = 1200) (h2 : CI3 = 1272) :
  ∃ (r : ℝ), r = 0.06 ∧ CI3 - CI2 = CI2 * r :=
by sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1729_172929


namespace NUMINAMATH_CALUDE_distributive_law_addition_over_multiplication_not_hold_l1729_172913

-- Define the pair type
def Pair := ℝ × ℝ

-- Define addition operation
def add : Pair → Pair → Pair
  | (x₁, y₁), (x₂, y₂) => (x₁ + x₂, y₁ + y₂)

-- Define multiplication operation
def mul : Pair → Pair → Pair
  | (x₁, y₁), (x₂, y₂) => (x₁ * x₂ - y₁ * y₂, x₁ * y₂ + y₁ * x₂)

-- Statement: Distributive law of addition over multiplication does NOT hold
theorem distributive_law_addition_over_multiplication_not_hold :
  ∃ a b c : Pair, add a (mul b c) ≠ mul (add a b) (add a c) := by
  sorry

end NUMINAMATH_CALUDE_distributive_law_addition_over_multiplication_not_hold_l1729_172913


namespace NUMINAMATH_CALUDE_horizontal_shift_l1729_172933

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the shift amount
variable (a : ℝ)

-- Define a point (x, y) on the original graph
variable (x y : ℝ)

-- Theorem statement
theorem horizontal_shift (h : y = f x) :
  y = f (x - a) ↔ y = f ((x + a) - a) :=
sorry

end NUMINAMATH_CALUDE_horizontal_shift_l1729_172933


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l1729_172938

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Returns true if two lines are parallel. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Returns true if a point lies on a line. -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The given line y = -3x + 6 -/
def givenLine : Line :=
  { slope := -3, yIntercept := 6 }

theorem y_intercept_of_parallel_line :
  ∀ (b : Line),
    parallel b givenLine →
    pointOnLine b 3 (-4) →
    b.yIntercept = 5 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l1729_172938


namespace NUMINAMATH_CALUDE_flagpole_height_l1729_172976

/-- Given a person and a flagpole under the same lighting conditions, 
    we can determine the height of the flagpole using the ratio of heights to shadow lengths. -/
theorem flagpole_height
  (person_height : ℝ)
  (person_shadow : ℝ)
  (flagpole_shadow : ℝ)
  (h_person_height : person_height = 1.6)
  (h_person_shadow : person_shadow = 0.4)
  (h_flagpole_shadow : flagpole_shadow = 5)
  (h_positive : person_height > 0 ∧ person_shadow > 0 ∧ flagpole_shadow > 0) :
  (person_height / person_shadow) * flagpole_shadow = 20 :=
sorry

#check flagpole_height

end NUMINAMATH_CALUDE_flagpole_height_l1729_172976


namespace NUMINAMATH_CALUDE_guests_who_stayed_l1729_172930

-- Define the initial conditions
def total_guests : ℕ := 50
def men_guests : ℕ := 15

-- Define the number of guests who left
def men_left : ℕ := men_guests / 5
def children_left : ℕ := 4

-- Theorem to prove
theorem guests_who_stayed :
  let women_guests := total_guests / 2
  let children_guests := total_guests - women_guests - men_guests
  let guests_who_stayed := total_guests - men_left - children_left
  guests_who_stayed = 43 := by sorry

end NUMINAMATH_CALUDE_guests_who_stayed_l1729_172930


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1729_172999

/-- The speed of a boat in still water, given the rate of current and distance travelled downstream. -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  current_speed = 4 →
  downstream_distance = 10.4 →
  downstream_time = 24 / 60 →
  ∃ (boat_speed : ℝ), boat_speed = 22 ∧ downstream_distance = (boat_speed + current_speed) * downstream_time :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1729_172999


namespace NUMINAMATH_CALUDE_magic_8_ball_three_out_of_six_l1729_172988

def magic_8_ball_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem magic_8_ball_three_out_of_six :
  magic_8_ball_probability 6 3 (1/4) = 135/1024 := by
  sorry

end NUMINAMATH_CALUDE_magic_8_ball_three_out_of_six_l1729_172988


namespace NUMINAMATH_CALUDE_classroom_fraction_l1729_172992

theorem classroom_fraction (total : ℕ) (absent_fraction : ℚ) (canteen : ℕ) : 
  total = 40 → 
  absent_fraction = 1 / 10 → 
  canteen = 9 → 
  (total - (absent_fraction * total).num - canteen : ℚ) / (total - (absent_fraction * total).num) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_classroom_fraction_l1729_172992


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1729_172915

/-- The number of diagonals in a regular polygon with n sides -/
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l1729_172915


namespace NUMINAMATH_CALUDE_intersection_trig_functions_l1729_172936

theorem intersection_trig_functions (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) 
  (h3 : 6 * Real.cos x = 9 * Real.tan x) : Real.sin x = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_trig_functions_l1729_172936


namespace NUMINAMATH_CALUDE_chord_equation_l1729_172931

/-- A chord of the hyperbola x^2 - y^2 = 1 with midpoint (2, 1) -/
structure Chord where
  /-- First endpoint of the chord -/
  p1 : ℝ × ℝ
  /-- Second endpoint of the chord -/
  p2 : ℝ × ℝ
  /-- The chord endpoints lie on the hyperbola -/
  h1 : p1.1^2 - p1.2^2 = 1
  h2 : p2.1^2 - p2.2^2 = 1
  /-- The midpoint of the chord is (2, 1) -/
  h3 : (p1.1 + p2.1) / 2 = 2
  h4 : (p1.2 + p2.2) / 2 = 1

/-- The equation of the line containing the chord is y = 2x - 3 -/
theorem chord_equation (c : Chord) : 
  ∃ (m b : ℝ), m = 2 ∧ b = -3 ∧ ∀ (x y : ℝ), y = m * x + b ↔ 
  ∃ (t : ℝ), x = (1 - t) * c.p1.1 + t * c.p2.1 ∧ y = (1 - t) * c.p1.2 + t * c.p2.2 :=
sorry

end NUMINAMATH_CALUDE_chord_equation_l1729_172931


namespace NUMINAMATH_CALUDE_unique_real_roots_l1729_172960

def n : ℕ := 2016

-- Define geometric progression
def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ i : ℕ, i < n → a (i + 1) = r * a i

-- Define arithmetic progression
def is_arithmetic_progression (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : ℕ, i < n → b (i + 1) = b i + d

-- Define quadratic polynomial
def P (a b : ℕ → ℝ) (i : ℕ) (x : ℝ) : ℝ :=
  x^2 + a i * x + b i

-- Define discriminant
def discriminant (a b : ℕ → ℝ) (i : ℕ) : ℝ :=
  (a i)^2 - 4 * b i

-- Theorem statement
theorem unique_real_roots
  (a : ℕ → ℝ) (b : ℕ → ℝ) (k : ℕ)
  (h_geo : is_geometric_progression a)
  (h_arith : is_arithmetic_progression b)
  (h_unique : ∀ i : ℕ, i ≤ n → i ≠ k → discriminant a b i < 0)
  (h_real : discriminant a b k ≥ 0) :
  k = 1 ∨ k = n := by sorry

end NUMINAMATH_CALUDE_unique_real_roots_l1729_172960


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_l1729_172998

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extreme_value_implies_a (a : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x - (-3)| < ε → f a x ≤ f a (-3) ∨ f a x ≥ f a (-3)) →
  a = 5 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_l1729_172998


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_l1729_172944

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 7 = 0 → m ≥ n) ∧
  n = 105 := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_l1729_172944


namespace NUMINAMATH_CALUDE_quadratic_vertex_property_l1729_172959

/-- Given a quadratic function y = -x^2 + 2x + n with vertex (m, 1), prove m - n = 1 -/
theorem quadratic_vertex_property (n m : ℝ) : 
  (∀ x, -x^2 + 2*x + n = -(x - m)^2 + 1) → m - n = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_property_l1729_172959


namespace NUMINAMATH_CALUDE_largest_even_n_inequality_l1729_172983

theorem largest_even_n_inequality (n : ℕ) : 
  (n = 8 ∧ n % 2 = 0) ↔ 
  (∀ x : ℝ, (Real.sin x)^(2*n) + (Real.cos x)^(2*n) + (Real.tan x)^2 ≥ 1/n) ∧
  (∀ m : ℕ, m > n → m % 2 = 0 → 
    ∃ x : ℝ, (Real.sin x)^(2*m) + (Real.cos x)^(2*m) + (Real.tan x)^2 < 1/m) :=
by sorry

end NUMINAMATH_CALUDE_largest_even_n_inequality_l1729_172983


namespace NUMINAMATH_CALUDE_power_two_greater_than_square_five_is_smallest_smallest_n_zero_l1729_172903

theorem power_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := by sorry

theorem five_is_smallest (n : ℕ) (h : n < 5) : 2^n ≤ n^2 := by sorry

theorem smallest_n_zero : ∃ (n₀ : ℕ), (∀ (n : ℕ), n ≥ n₀ → 2^n > n^2) ∧ 
  (∀ (m : ℕ), m < n₀ → 2^m ≤ m^2) := by
  use 5
  constructor
  · exact power_two_greater_than_square
  · exact five_is_smallest

end NUMINAMATH_CALUDE_power_two_greater_than_square_five_is_smallest_smallest_n_zero_l1729_172903


namespace NUMINAMATH_CALUDE_professors_women_tenured_or_both_l1729_172927

theorem professors_women_tenured_or_both (
  women_percentage : Real)
  (tenured_percentage : Real)
  (men_tenured_percentage : Real)
  (h1 : women_percentage = 0.69)
  (h2 : tenured_percentage = 0.70)
  (h3 : men_tenured_percentage = 0.52) :
  women_percentage + tenured_percentage - (tenured_percentage - men_tenured_percentage * (1 - women_percentage)) = 0.8512 := by
  sorry

end NUMINAMATH_CALUDE_professors_women_tenured_or_both_l1729_172927


namespace NUMINAMATH_CALUDE_product_first_two_terms_l1729_172978

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℝ
  /-- The common difference between consecutive terms -/
  d : ℝ
  /-- The seventh term of the sequence is 20 -/
  seventh_term : a₁ + 6 * d = 20
  /-- The common difference is 2 -/
  common_diff : d = 2

/-- The product of the first two terms of the arithmetic sequence is 80 -/
theorem product_first_two_terms (seq : ArithmeticSequence) :
  seq.a₁ * (seq.a₁ + seq.d) = 80 := by
  sorry


end NUMINAMATH_CALUDE_product_first_two_terms_l1729_172978


namespace NUMINAMATH_CALUDE_f_2013_plus_f_neg_2014_l1729_172968

open Real

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_2 (f : ℝ → ℝ) : Prop := ∀ x ≥ 0, f (x + 2) = f x

def matches_exp_minus_one_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 2 → f x = exp x - 1

theorem f_2013_plus_f_neg_2014 (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : is_periodic_2 f)
  (h_match : matches_exp_minus_one_on_interval f) :
  f 2013 + f (-2014) = exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2013_plus_f_neg_2014_l1729_172968


namespace NUMINAMATH_CALUDE_square_sum_of_powers_l1729_172955

theorem square_sum_of_powers (a b : ℕ+) : 
  (∃ n : ℕ, 2^(a : ℕ) + 3^(b : ℕ) = n^2) ↔ a = 4 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_powers_l1729_172955


namespace NUMINAMATH_CALUDE_yarn_length_multiple_l1729_172980

theorem yarn_length_multiple (green_length red_length total_length x : ℝ) : 
  green_length = 156 →
  red_length = green_length * x + 8 →
  total_length = green_length + red_length →
  total_length = 632 →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_yarn_length_multiple_l1729_172980


namespace NUMINAMATH_CALUDE_min_shots_battleship_l1729_172934

/-- Represents a cell on the grid -/
structure Cell :=
  (row : Nat)
  (col : Nat)

/-- Represents a ship placement on the grid -/
structure Ship :=
  (start : Cell)
  (horizontal : Bool)

/-- The grid size -/
def gridSize : Nat := 5

/-- The ship length -/
def shipLength : Nat := 4

/-- Checks if a ship placement is valid on the grid -/
def isValidShip (s : Ship) : Prop :=
  s.start.row ≥ 1 ∧ s.start.row ≤ gridSize ∧
  s.start.col ≥ 1 ∧ s.start.col ≤ gridSize ∧
  (if s.horizontal
   then s.start.col + shipLength - 1 ≤ gridSize
   else s.start.row + shipLength - 1 ≤ gridSize)

/-- Checks if a shot hits a ship -/
def hitShip (shot : Cell) (s : Ship) : Prop :=
  if s.horizontal
  then shot.row = s.start.row ∧ shot.col ≥ s.start.col ∧ shot.col < s.start.col + shipLength
  else shot.col = s.start.col ∧ shot.row ≥ s.start.row ∧ shot.row < s.start.row + shipLength

/-- The main theorem: 6 shots are sufficient and necessary -/
theorem min_shots_battleship :
  ∃ (shots : Finset Cell),
    shots.card = 6 ∧
    (∀ s : Ship, isValidShip s → ∃ shot ∈ shots, hitShip shot s) ∧
    (∀ (shots' : Finset Cell), shots'.card < 6 →
      ∃ s : Ship, isValidShip s ∧ ∀ shot ∈ shots', ¬hitShip shot s) :=
sorry

end NUMINAMATH_CALUDE_min_shots_battleship_l1729_172934


namespace NUMINAMATH_CALUDE_multiply_33333_33334_l1729_172932

theorem multiply_33333_33334 : 33333 * 33334 = 1111122222 := by
  sorry

end NUMINAMATH_CALUDE_multiply_33333_33334_l1729_172932


namespace NUMINAMATH_CALUDE_prime_sequence_equality_l1729_172997

theorem prime_sequence_equality (p : ℕ) 
  (h1 : Nat.Prime p) 
  (h2 : Nat.Prime (p + 10)) 
  (h3 : Nat.Prime (p + 14)) 
  (h4 : Nat.Prime (2 * p + 1)) 
  (h5 : Nat.Prime (4 * p + 1)) : p = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_sequence_equality_l1729_172997


namespace NUMINAMATH_CALUDE_power_inequality_l1729_172965

theorem power_inequality (a b m n : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 0) (hn : n > 0) :
  a^(m+n) + b^(m+n) ≥ a^m * b^n + a^n * b^m := by sorry

end NUMINAMATH_CALUDE_power_inequality_l1729_172965


namespace NUMINAMATH_CALUDE_root_conditions_imply_relation_l1729_172986

/-- Given two equations with specific root conditions, prove a relation between constants c and d -/
theorem root_conditions_imply_relation (c d : ℝ) : 
  (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    ((r₁ + c) * (r₁ + d) * (r₁ - 7)) / ((r₁ + 4)^2) = 0 ∧
    ((r₂ + c) * (r₂ + d) * (r₂ - 7)) / ((r₂ + 4)^2) = 0 ∧
    ((r₃ + c) * (r₃ + d) * (r₃ - 7)) / ((r₃ + 4)^2) = 0) →
  (∃! (r : ℝ), ((r + 2*c) * (r + 5) * (r + 8)) / ((r + d) * (r - 7)) = 0) →
  100 * c + d = 408 := by
sorry

end NUMINAMATH_CALUDE_root_conditions_imply_relation_l1729_172986


namespace NUMINAMATH_CALUDE_number_equals_sixteen_l1729_172909

theorem number_equals_sixteen (x y : ℝ) (h1 : |x| = 9*x - y) (h2 : x = 2) : y = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_equals_sixteen_l1729_172909


namespace NUMINAMATH_CALUDE_ceiling_sum_evaluation_l1729_172910

theorem ceiling_sum_evaluation : 
  ⌈Real.sqrt (16/9 : ℝ)⌉ + ⌈(16/9 : ℝ)⌉ + ⌈(16/9 : ℝ)^2⌉ + ⌈(16/9 : ℝ)^(1/3)⌉ = 10 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_evaluation_l1729_172910


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1729_172964

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity :
  ∀ (a : ℝ), a > 0 →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / 4 = 1) →
  (∀ (x y : ℝ), y^2 = 12*x) →
  (∃ (xf : ℝ), xf = 3 ∧ (∀ (y : ℝ), x^2 / a^2 - y^2 / 4 = 1 → (x - xf)^2 + y^2 = (3*a/5)^2)) →
  3 * Real.sqrt 5 / 5 = 3 / Real.sqrt (a^2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1729_172964


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l1729_172966

/-- Given two points on a linear function, prove that y₁ > y₂ -/
theorem y1_greater_than_y2 (y₁ y₂ : ℝ) : 
  (y₁ = -(-1) + 1) → (y₂ = -(2) + 1) → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l1729_172966


namespace NUMINAMATH_CALUDE_clare_remaining_money_l1729_172928

-- Define the initial amount and item costs
def initial_amount : ℚ := 47
def bread_cost : ℚ := 2
def milk_cost : ℚ := 2
def cereal_cost : ℚ := 3
def apple_cost : ℚ := 4

-- Define the quantities of each item
def bread_quantity : ℕ := 4
def milk_quantity : ℕ := 2
def cereal_quantity : ℕ := 3
def apple_quantity : ℕ := 1

-- Define the discount and tax rates
def discount_rate : ℚ := 0.1
def tax_rate : ℚ := 0.05

-- Define the function to calculate the remaining money
def calculate_remaining_money : ℚ :=
  let total_cost := bread_cost * bread_quantity + milk_cost * milk_quantity + 
                    cereal_cost * cereal_quantity + apple_cost * apple_quantity
  let discounted_cost := total_cost * (1 - discount_rate)
  let tax_amount := discounted_cost * tax_rate
  let final_cost := discounted_cost + tax_amount
  initial_amount - final_cost

-- Theorem statement
theorem clare_remaining_money :
  calculate_remaining_money = 23.37 := by sorry

end NUMINAMATH_CALUDE_clare_remaining_money_l1729_172928


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_condition_l1729_172973

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n * (a 1 + a n)) / 2

/-- The theorem stating that m = 5 for the given conditions -/
theorem arithmetic_sequence_sum_condition (seq : ArithmeticSequence) (m : ℕ) :
  m > 1 →
  seq.S (m - 1) = -2 →
  seq.S m = 0 →
  seq.S (m + 1) = 3 →
  m = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_condition_l1729_172973


namespace NUMINAMATH_CALUDE_min_value_quadratic_ratio_l1729_172961

/-- A quadratic function -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The derivative of a quadratic function -/
def quadratic_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem min_value_quadratic_ratio 
  (a b c : ℝ) 
  (h1 : quadratic_derivative a b 0 > 0)
  (h2 : ∀ x, quadratic a b c x ≥ 0) :
  (quadratic a b c 1) / (quadratic_derivative a b 0) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_ratio_l1729_172961


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1729_172982

theorem log_sum_equals_two : Real.log 3 / Real.log 6 + Real.log 4 / Real.log 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1729_172982


namespace NUMINAMATH_CALUDE_max_female_students_with_4_teachers_min_total_people_l1729_172977

/-- Represents a study group composition --/
structure StudyGroup where
  male_students : ℕ
  female_students : ℕ
  teachers : ℕ

/-- Checks if a study group satisfies the given conditions --/
def is_valid_group (g : StudyGroup) : Prop :=
  g.male_students > g.female_students ∧
  g.female_students > g.teachers ∧
  2 * g.teachers > g.male_students

/-- The maximum number of female students when there are 4 teachers is 6 --/
theorem max_female_students_with_4_teachers :
  ∀ g : StudyGroup, is_valid_group g → g.teachers = 4 → g.female_students ≤ 6 := by
  sorry

/-- The minimum number of people in a valid study group is 12 --/
theorem min_total_people :
  ∀ g : StudyGroup, is_valid_group g →
    g.male_students + g.female_students + g.teachers ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_max_female_students_with_4_teachers_min_total_people_l1729_172977


namespace NUMINAMATH_CALUDE_bob_candy_count_l1729_172956

/-- Bob's share of items -/
structure BobsShare where
  chewing_gums : ℕ
  chocolate_bars : ℕ
  assorted_candies : ℕ

/-- Definition of Bob's relationship and actions -/
structure BobInfo where
  is_sams_neighbor : Prop
  accompanies_sam : Prop
  share : BobsShare

/-- Theorem stating the number of candies Bob got -/
theorem bob_candy_count (bob : BobInfo) 
  (h1 : bob.is_sams_neighbor)
  (h2 : bob.accompanies_sam)
  (h3 : bob.share.chewing_gums = 15)
  (h4 : bob.share.chocolate_bars = 20)
  (h5 : bob.share.assorted_candies = 15) : 
  bob.share.assorted_candies = 15 := by
  sorry

end NUMINAMATH_CALUDE_bob_candy_count_l1729_172956


namespace NUMINAMATH_CALUDE_blue_tile_fraction_is_three_fourths_l1729_172962

/-- Represents the tiling pattern of an 8x8 square -/
structure TilingPattern :=
  (size : Nat)
  (blue_tiles_per_corner : Nat)
  (total_corners : Nat)

/-- The fraction of blue tiles in the tiling pattern -/
def blue_tile_fraction (pattern : TilingPattern) : Rat :=
  let total_blue_tiles := pattern.blue_tiles_per_corner * pattern.total_corners
  let total_tiles := pattern.size * pattern.size
  total_blue_tiles / total_tiles

/-- Theorem stating that the fraction of blue tiles in the given pattern is 3/4 -/
theorem blue_tile_fraction_is_three_fourths (pattern : TilingPattern) 
  (h1 : pattern.size = 8)
  (h2 : pattern.blue_tiles_per_corner = 12)
  (h3 : pattern.total_corners = 4) : 
  blue_tile_fraction pattern = 3/4 := by
  sorry

#check blue_tile_fraction_is_three_fourths

end NUMINAMATH_CALUDE_blue_tile_fraction_is_three_fourths_l1729_172962


namespace NUMINAMATH_CALUDE_solve_for_m_l1729_172981

/-- 
If 2x + m = 6 and x = 2, then m = 2
-/
theorem solve_for_m (x m : ℝ) (eq : 2 * x + m = 6) (sol : x = 2) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l1729_172981


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l1729_172914

theorem average_of_three_numbers (x : ℝ) (h1 : x = 33) : (x + 4*x + 2*x) / 3 = 77 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l1729_172914


namespace NUMINAMATH_CALUDE_max_product_given_sum_l1729_172991

theorem max_product_given_sum (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 2 → ∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → a * b ≥ x * y :=
sorry

end NUMINAMATH_CALUDE_max_product_given_sum_l1729_172991


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_target_l1729_172948

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (x - 1) < 0}
def N : Set ℝ := {x | 2 * x^2 - 3 * x ≤ 0}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Define the open-closed interval (1, 3/2]
def target_set : Set ℝ := Set.Ioc 1 (3/2)

-- Theorem statement
theorem M_intersect_N_equals_target : M_intersect_N = target_set := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_target_l1729_172948


namespace NUMINAMATH_CALUDE_inverse_sum_theorem_l1729_172994

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- State the theorem
theorem inverse_sum_theorem : 
  ∃ (a b : ℝ), f a = 9 ∧ f b = -64 ∧ a + b = -5 := by sorry

end NUMINAMATH_CALUDE_inverse_sum_theorem_l1729_172994


namespace NUMINAMATH_CALUDE_min_value_of_E_l1729_172972

theorem min_value_of_E :
  ∃ (E : ℝ), (∀ (x : ℝ), |x - 4| + |E| + |x - 5| ≥ 12) ∧
  (∀ (F : ℝ), (∀ (x : ℝ), |x - 4| + |F| + |x - 5| ≥ 12) → |F| ≥ |E|) ∧
  |E| = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_E_l1729_172972


namespace NUMINAMATH_CALUDE_fraction_simplification_l1729_172989

theorem fraction_simplification (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (1 / y) / (1 / x) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1729_172989


namespace NUMINAMATH_CALUDE_trapezoid_longer_side_length_l1729_172963

/-- Given a square of side length s divided into a pentagon and three congruent trapezoids,
    if all four shapes have equal area, then the length of the longer parallel side
    of each trapezoid is s/2 -/
theorem trapezoid_longer_side_length (s : ℝ) (s_pos : s > 0) :
  let square_area := s^2
  let shape_area := square_area / 4
  let trapezoid_height := s / 2
  ∃ x : ℝ,
    x > 0 ∧
    x < s ∧
    shape_area = (x + s/2) * trapezoid_height / 2 ∧
    x = s / 2 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_longer_side_length_l1729_172963


namespace NUMINAMATH_CALUDE_monster_count_monster_count_proof_l1729_172918

theorem monster_count : ℕ → Prop :=
  fun m : ℕ =>
    ∃ s : ℕ,
      s = 4 * m + 3 ∧
      s = 5 * m - 6 →
      m = 9

-- The proof is omitted
theorem monster_count_proof : monster_count 9 := by
  sorry

end NUMINAMATH_CALUDE_monster_count_monster_count_proof_l1729_172918


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l1729_172947

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- Question 1
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Question 2
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l1729_172947


namespace NUMINAMATH_CALUDE_paper_stack_thickness_sheets_in_six_cm_stack_l1729_172996

/-- Calculates the number of sheets in a stack of paper given the thickness of the stack and the number of sheets per unit thickness. -/
def sheets_in_stack (stack_thickness : ℝ) (sheets_per_unit : ℝ) : ℝ :=
  stack_thickness * sheets_per_unit

theorem paper_stack_thickness (bundle_sheets : ℝ) (bundle_thickness : ℝ) (stack_thickness : ℝ) :
  bundle_sheets > 0 → bundle_thickness > 0 → stack_thickness > 0 →
  sheets_in_stack stack_thickness (bundle_sheets / bundle_thickness) = 
    (stack_thickness / bundle_thickness) * bundle_sheets := by
  sorry

/-- The main theorem that proves the number of sheets in a 6 cm stack given a 400-sheet bundle is 4 cm thick. -/
theorem sheets_in_six_cm_stack : 
  sheets_in_stack 6 (400 / 4) = 600 := by
  sorry

end NUMINAMATH_CALUDE_paper_stack_thickness_sheets_in_six_cm_stack_l1729_172996


namespace NUMINAMATH_CALUDE_shekars_english_marks_l1729_172904

/-- Represents the marks scored in each subject -/
structure Marks where
  mathematics : ℕ
  science : ℕ
  socialStudies : ℕ
  biology : ℕ
  english : ℕ

/-- Calculates the average of a list of natural numbers -/
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

/-- Theorem: Given Shekar's marks in other subjects and his average, his English marks are 67 -/
theorem shekars_english_marks (m : Marks) (h1 : m.mathematics = 76) (h2 : m.science = 65)
    (h3 : m.socialStudies = 82) (h4 : m.biology = 75)
    (h5 : average [m.mathematics, m.science, m.socialStudies, m.biology, m.english] = 73) :
    m.english = 67 := by
  sorry

#check shekars_english_marks

end NUMINAMATH_CALUDE_shekars_english_marks_l1729_172904


namespace NUMINAMATH_CALUDE_total_hours_spent_l1729_172924

def project_time : ℕ := 300
def research_time : ℕ := 45
def presentation_time : ℕ := 75

def total_minutes : ℕ := project_time + research_time + presentation_time

theorem total_hours_spent : (total_minutes : ℚ) / 60 = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_hours_spent_l1729_172924


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l1729_172907

theorem indefinite_integral_proof (x : ℝ) : 
  (deriv (fun x => -1/2 * (2 - 3*x) * Real.cos (2*x) - 3/4 * Real.sin (2*x))) x 
  = (2 - 3*x) * Real.sin (2*x) := by
sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l1729_172907


namespace NUMINAMATH_CALUDE_chord_length_exists_point_P_l1729_172923

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 16

-- Define point F
def F : ℝ × ℝ := (-2, 0)

-- Define the line x = -4
def line_x_eq_neg_4 (x y : ℝ) : Prop := x = -4

-- Theorem 1: Length of the chord
theorem chord_length :
  ∃ (G : ℝ × ℝ), C₁ G.1 G.2 →
  ∃ (T : ℝ × ℝ), line_x_eq_neg_4 T.1 T.2 →
  (G.1 - F.1 = T.1 - G.1 ∧ G.2 - F.2 = T.2 - G.2) →
  ∃ (chord_length : ℝ), chord_length = 7 :=
sorry

-- Theorem 2: Existence of point P
theorem exists_point_P :
  ∃ (P : ℝ × ℝ), P = (4, 0) ∧
  ∀ (G : ℝ × ℝ), C₁ G.1 G.2 →
  (G.1 - P.1)^2 + (G.2 - P.2)^2 = 4 * ((G.1 - F.1)^2 + (G.2 - F.2)^2) :=
sorry

end NUMINAMATH_CALUDE_chord_length_exists_point_P_l1729_172923


namespace NUMINAMATH_CALUDE_old_supervisor_salary_proof_l1729_172937

/-- Calculates the old supervisor's salary given the initial and new average salaries,
    number of workers, and new supervisor's salary. -/
def old_supervisor_salary (initial_avg : ℚ) (new_avg : ℚ) (num_workers : ℕ) 
  (new_supervisor_salary : ℚ) : ℚ :=
  (initial_avg * (num_workers + 1) - new_avg * (num_workers + 1) + new_supervisor_salary)

/-- Proves that the old supervisor's salary was $870 given the problem conditions. -/
theorem old_supervisor_salary_proof :
  old_supervisor_salary 430 440 8 960 = 870 := by
  sorry

#eval old_supervisor_salary 430 440 8 960

end NUMINAMATH_CALUDE_old_supervisor_salary_proof_l1729_172937


namespace NUMINAMATH_CALUDE_skittles_per_friend_l1729_172967

def total_skittles : ℕ := 40
def num_friends : ℕ := 5

theorem skittles_per_friend :
  total_skittles / num_friends = 8 := by sorry

end NUMINAMATH_CALUDE_skittles_per_friend_l1729_172967


namespace NUMINAMATH_CALUDE_icosagon_diagonals_l1729_172979

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in an icosagon (20-sided polygon) is 170 -/
theorem icosagon_diagonals : num_diagonals 20 = 170 := by
  sorry

end NUMINAMATH_CALUDE_icosagon_diagonals_l1729_172979


namespace NUMINAMATH_CALUDE_prob_second_red_given_first_red_l1729_172958

/-- The probability of drawing a red ball on the second draw, given that a red ball was drawn on the first -/
theorem prob_second_red_given_first_red 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (white_balls : ℕ) 
  (h1 : total_balls = 6)
  (h2 : red_balls = 4)
  (h3 : white_balls = 2)
  (h4 : total_balls = red_balls + white_balls) :
  (red_balls - 1 : ℚ) / (total_balls - 1) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_second_red_given_first_red_l1729_172958


namespace NUMINAMATH_CALUDE_simple_interest_investment_l1729_172916

/-- Proves that an initial investment of $1000 with 10% simple interest over 3 years results in $1300 --/
theorem simple_interest_investment (P : ℝ) : 
  (P * (1 + 0.1 * 3) = 1300) → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_investment_l1729_172916


namespace NUMINAMATH_CALUDE_number_puzzle_l1729_172952

theorem number_puzzle (a b : ℕ) : 
  a + b = 21875 →
  (a % 5 = 0 ∨ b % 5 = 0) →
  b = 10 * a + 5 →
  b - a = 17893 := by
sorry

end NUMINAMATH_CALUDE_number_puzzle_l1729_172952


namespace NUMINAMATH_CALUDE_difference_of_squares_l1729_172985

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1729_172985


namespace NUMINAMATH_CALUDE_h_j_h_3_eq_86_l1729_172942

def h (x : ℝ) : ℝ := 2 * x + 2

def j (x : ℝ) : ℝ := 5 * x + 2

theorem h_j_h_3_eq_86 : h (j (h 3)) = 86 := by
  sorry

end NUMINAMATH_CALUDE_h_j_h_3_eq_86_l1729_172942


namespace NUMINAMATH_CALUDE_reading_rate_difference_l1729_172911

-- Define the given information
def songhee_pages : ℕ := 288
def songhee_days : ℕ := 12
def eunju_pages : ℕ := 243
def eunju_days : ℕ := 9

-- Define the daily reading rates
def songhee_rate : ℚ := songhee_pages / songhee_days
def eunju_rate : ℚ := eunju_pages / eunju_days

-- Theorem statement
theorem reading_rate_difference : eunju_rate - songhee_rate = 3 := by
  sorry

end NUMINAMATH_CALUDE_reading_rate_difference_l1729_172911


namespace NUMINAMATH_CALUDE_expression_value_l1729_172925

theorem expression_value (x y : ℝ) (h : |x + 1| + (2 * y - 4)^2 = 0) :
  (2 * x^2 * y - 3 * x * y) - 2 * (x^2 * y - x * y + 1/2 * x * y^2) + x * y = 4 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l1729_172925


namespace NUMINAMATH_CALUDE_car_storm_distance_time_l1729_172908

/-- The time when a car traveling north at 3/4 mile per minute is 30 miles away from the center of a storm
    moving southeast at 3/4√2 mile per minute, given that at t=0 the storm's center is 150 miles due east of the car. -/
theorem car_storm_distance_time : ∃ t : ℝ,
  (27 / 32 : ℝ) * t^2 - (450 * Real.sqrt 2 / 2) * t + 21600 = 0 :=
by sorry

end NUMINAMATH_CALUDE_car_storm_distance_time_l1729_172908


namespace NUMINAMATH_CALUDE_friend_money_pooling_l1729_172969

/-- Represents the money pooling problem with 4 friends --/
theorem friend_money_pooling
  (peter john quincy andrew : ℕ)  -- Money amounts for each friend
  (h1 : peter = 320)              -- Peter has $320
  (h2 : peter = 2 * john)         -- Peter has twice as much as John
  (h3 : quincy > peter)           -- Quincy has more than Peter
  (h4 : andrew = (115 * quincy) / 100)  -- Andrew has 15% more than Quincy
  (h5 : peter + john + quincy + andrew = 1211)  -- Total money after spending $1200
  : quincy - peter = 20 :=
by sorry

end NUMINAMATH_CALUDE_friend_money_pooling_l1729_172969


namespace NUMINAMATH_CALUDE_coconuts_yield_five_l1729_172957

/-- The number of coconuts each tree yields -/
def coconuts_per_tree (price_per_coconut : ℚ) (total_amount : ℚ) (num_trees : ℕ) : ℚ :=
  (total_amount / price_per_coconut) / num_trees

/-- Proof that each tree yields 5 coconuts given the conditions -/
theorem coconuts_yield_five :
  coconuts_per_tree 3 90 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_coconuts_yield_five_l1729_172957


namespace NUMINAMATH_CALUDE_total_money_made_l1729_172971

/-- The total money made from selling items is the sum of the products of price and quantity for each item. -/
theorem total_money_made 
  (smoothie_price cake_price : ℚ) 
  (smoothie_quantity cake_quantity : ℕ) :
  smoothie_price * smoothie_quantity + cake_price * cake_quantity =
  (smoothie_price * smoothie_quantity + cake_price * cake_quantity : ℚ) :=
by sorry

/-- Scott's total earnings from selling smoothies and cakes -/
def scotts_earnings : ℚ :=
  let smoothie_price : ℚ := 3
  let cake_price : ℚ := 2
  let smoothie_quantity : ℕ := 40
  let cake_quantity : ℕ := 18
  smoothie_price * smoothie_quantity + cake_price * cake_quantity

#eval scotts_earnings -- Expected output: 156

end NUMINAMATH_CALUDE_total_money_made_l1729_172971


namespace NUMINAMATH_CALUDE_equation_solutions_l1729_172975

theorem equation_solutions (a : ℝ) (h : a < 0) :
  ∃! (s : Finset ℝ), s.card = 4 ∧
  (∀ x ∈ s, -π < x ∧ x < π) ∧
  (∀ x ∈ s, (a - 1) * (Real.sin (2 * x) + Real.cos x) + (a + 1) * (Real.sin x - Real.cos (2 * x)) = 0) ∧
  (∀ x, -π < x ∧ x < π →
    (a - 1) * (Real.sin (2 * x) + Real.cos x) + (a + 1) * (Real.sin x - Real.cos (2 * x)) = 0 →
    x ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1729_172975


namespace NUMINAMATH_CALUDE_baking_difference_l1729_172987

/-- Given a recipe and current state of baking, calculate the difference between
    remaining sugar and flour to be added. -/
def sugar_flour_difference (total_flour total_sugar added_flour : ℕ) : ℕ :=
  total_sugar - (total_flour - added_flour)

/-- Theorem stating the difference between remaining sugar and flour to be added
    for the given recipe and current state. -/
theorem baking_difference : sugar_flour_difference 9 11 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_baking_difference_l1729_172987


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l1729_172920

theorem cubic_root_equation_solution :
  ∃ x : ℝ, x > 0 ∧ 3 * (2 + x)^(1/3) + 4 * (2 - x)^(1/3) = 6 ∧ |x - 2.096| < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l1729_172920


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1729_172946

theorem perfect_square_condition (x : ℤ) : 
  ∃ (y : ℤ), x * (x + 1) * (x + 7) * (x + 8) = y^2 ↔ 
  x = -9 ∨ x = -8 ∨ x = -7 ∨ x = -4 ∨ x = -1 ∨ x = 0 ∨ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1729_172946


namespace NUMINAMATH_CALUDE_geometric_distribution_variance_l1729_172974

/-- A random variable following a geometric distribution with parameter p -/
def GeometricDistribution (p : ℝ) := { X : ℝ → ℝ // 0 < p ∧ p ≤ 1 }

/-- The variance of a random variable -/
def variance (X : ℝ → ℝ) : ℝ := sorry

/-- The theorem stating that the variance of a geometric distribution is (1-p)/p^2 -/
theorem geometric_distribution_variance (p : ℝ) (X : GeometricDistribution p) :
  variance X.val = (1 - p) / p^2 := by sorry

end NUMINAMATH_CALUDE_geometric_distribution_variance_l1729_172974


namespace NUMINAMATH_CALUDE_only_x0_is_perfect_square_l1729_172901

-- Define the sequence (x_n)
def x : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * x (n + 1) - x n

-- Define a perfect square
def isPerfectSquare (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

-- Theorem statement
theorem only_x0_is_perfect_square :
  ∀ n : ℕ, isPerfectSquare (x n) → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_only_x0_is_perfect_square_l1729_172901
