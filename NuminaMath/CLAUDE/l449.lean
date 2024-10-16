import Mathlib

namespace NUMINAMATH_CALUDE_even_decreasing_comparison_l449_44906

-- Define an even function that is decreasing on (-∞, 0)
def even_decreasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, x < y ∧ y ≤ 0 → f y < f x)

-- Theorem statement
theorem even_decreasing_comparison 
  (f : ℝ → ℝ) 
  (h : even_decreasing_function f) : 
  f 2 < f (-3) := by
sorry

end NUMINAMATH_CALUDE_even_decreasing_comparison_l449_44906


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l449_44949

/-- A sequence a : ℕ → ℝ is geometric if there exists a common ratio r such that
    a(n+1) = r * a(n) for all n ≥ 1 -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  IsGeometric a →
  (a 1)^2 - 2*(a 1) - 3 = 0 →
  (a 4)^2 - 2*(a 4) - 3 = 0 →
  a 2 * a 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l449_44949


namespace NUMINAMATH_CALUDE_negation_of_forall_leq_is_exists_gt_l449_44919

theorem negation_of_forall_leq_is_exists_gt (p : (n : ℕ) → n^2 ≤ 2^n → Prop) :
  (¬ ∀ n : ℕ, n^2 ≤ 2^n) ↔ (∃ n : ℕ, n^2 > 2^n) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_leq_is_exists_gt_l449_44919


namespace NUMINAMATH_CALUDE_johnny_table_planks_l449_44954

/-- The number of planks needed for a table's surface -/
def planks_for_surface (total_tables : ℕ) (total_planks : ℕ) (legs_per_table : ℕ) : ℕ :=
  (total_planks / total_tables) - legs_per_table

theorem johnny_table_planks 
  (total_tables : ℕ) 
  (total_planks : ℕ) 
  (legs_per_table : ℕ) 
  (h1 : total_tables = 5) 
  (h2 : total_planks = 45) 
  (h3 : legs_per_table = 4) : 
  planks_for_surface total_tables total_planks legs_per_table = 5 := by
sorry

end NUMINAMATH_CALUDE_johnny_table_planks_l449_44954


namespace NUMINAMATH_CALUDE_rectangle_side_length_l449_44937

theorem rectangle_side_length (a b d : ℝ) : 
  a = 4 →
  a / b = 2 * (b / d) →
  d^2 = a^2 + b^2 →
  b = Real.sqrt (2 + 4 * Real.sqrt 17) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l449_44937


namespace NUMINAMATH_CALUDE_stratified_sampling_expectation_l449_44964

theorem stratified_sampling_expectation
  (total_population : ℕ)
  (sample_size : ℕ)
  (category_size : ℕ)
  (h1 : total_population = 100)
  (h2 : sample_size = 20)
  (h3 : category_size = 30) :
  (sample_size : ℚ) / total_population * category_size = 6 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_expectation_l449_44964


namespace NUMINAMATH_CALUDE_inequality_proof_l449_44960

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 3) : 
  a / (a + 5) + b / (b + 5) + c / (c + 5) ≤ 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l449_44960


namespace NUMINAMATH_CALUDE_max_second_term_arithmetic_sequence_l449_44914

theorem max_second_term_arithmetic_sequence :
  ∀ (a d : ℕ),
    a > 0 →
    d > 0 →
    a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 60 →
    ∀ (b e : ℕ),
      b > 0 →
      e > 0 →
      b + (b + e) + (b + 2*e) + (b + 3*e) + (b + 4*e) = 60 →
      (a + d) ≤ 7 ∧
      (∃ a d : ℕ, a > 0 ∧ d > 0 ∧ a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 60 ∧ a + d = 7) :=
by sorry

#check max_second_term_arithmetic_sequence

end NUMINAMATH_CALUDE_max_second_term_arithmetic_sequence_l449_44914


namespace NUMINAMATH_CALUDE_isabella_house_paintable_area_l449_44993

-- Define the problem parameters
def num_bedrooms : ℕ := 4
def room_length : ℝ := 15
def room_width : ℝ := 12
def room_height : ℝ := 9
def unpaintable_area : ℝ := 80

-- Define the function to calculate the paintable area
def paintable_area : ℝ :=
  let total_wall_area := num_bedrooms * (2 * (room_length * room_height + room_width * room_height))
  total_wall_area - (num_bedrooms * unpaintable_area)

-- State the theorem
theorem isabella_house_paintable_area :
  paintable_area = 1624 := by sorry

end NUMINAMATH_CALUDE_isabella_house_paintable_area_l449_44993


namespace NUMINAMATH_CALUDE_lemon_production_increase_l449_44941

/-- Represents the lemon production data for normal and engineered trees -/
structure LemonProduction where
  normal_lemons_per_year : ℕ
  grove_size : ℕ
  total_lemons : ℕ
  years : ℕ

/-- Calculates the percentage increase in lemon production -/
def percentage_increase (data : LemonProduction) : ℚ :=
  let normal_total := data.normal_lemons_per_year * data.years
  let engineered_per_tree := data.total_lemons / data.grove_size
  ((engineered_per_tree - normal_total) / normal_total) * 100

/-- Theorem stating the percentage increase in lemon production -/
theorem lemon_production_increase (data : LemonProduction) 
  (h1 : data.normal_lemons_per_year = 60)
  (h2 : data.grove_size = 1500)
  (h3 : data.total_lemons = 675000)
  (h4 : data.years = 5) :
  percentage_increase data = 50 := by
  sorry

end NUMINAMATH_CALUDE_lemon_production_increase_l449_44941


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l449_44917

/-- Given a principal amount P, prove that if the compound interest at 4% for 2 years is $612,
    then the simple interest at 4% for 2 years is $600. -/
theorem simple_interest_calculation (P : ℝ) : 
  P * (1 + 0.04)^2 - P = 612 → P * 0.04 * 2 = 600 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l449_44917


namespace NUMINAMATH_CALUDE_soccer_game_analysis_l449_44904

-- Define the players
inductive Player : Type
| Amandine : Player
| Bobby : Player
| Charles : Player

-- Define the game structure
structure Game where
  total_phases : ℕ
  amandine_field : ℕ
  bobby_field : ℕ
  charles_goalkeeper : ℕ

-- Define the theorem
theorem soccer_game_analysis (g : Game) 
  (h1 : g.amandine_field = 12)
  (h2 : g.bobby_field = 21)
  (h3 : g.charles_goalkeeper = 8)
  (h4 : g.total_phases = g.amandine_field + (g.total_phases - g.amandine_field))
  (h5 : g.total_phases = g.bobby_field + (g.total_phases - g.bobby_field))
  (h6 : g.total_phases = (g.total_phases - g.charles_goalkeeper) + g.charles_goalkeeper) :
  g.total_phases = 25 ∧ (∃ n : ℕ, n = 6 ∧ n % 2 = 0 ∧ (n + 1) ≤ g.total_phases - g.amandine_field) := by
  sorry


end NUMINAMATH_CALUDE_soccer_game_analysis_l449_44904


namespace NUMINAMATH_CALUDE_systematic_sampling_l449_44927

theorem systematic_sampling (total_students : Nat) (sample_size : Nat) (part_size : Nat) (first_drawn : Nat) :
  total_students = 1000 →
  sample_size = 50 →
  part_size = 20 →
  first_drawn = 15 →
  (third_drawn : Nat) = 55 :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_l449_44927


namespace NUMINAMATH_CALUDE_smallest_subset_size_for_divisibility_l449_44982

theorem smallest_subset_size_for_divisibility : ∃ (n : ℕ),
  n = 337 ∧
  (∀ (S : Finset ℕ), S ⊆ Finset.range 2005 → S.card = n →
    ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ 2004 ∣ (a^2 - b^2)) ∧
  (∀ (m : ℕ), m < n →
    ∃ (T : Finset ℕ), T ⊆ Finset.range 2005 ∧ T.card = m ∧
      ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → ¬(2004 ∣ (a^2 - b^2))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_subset_size_for_divisibility_l449_44982


namespace NUMINAMATH_CALUDE_max_representations_l449_44953

/-- The number of colors used for coding -/
def n : ℕ := 5

/-- The number of ways to choose a single color -/
def single_color_choices (m : ℕ) : ℕ := m

/-- The number of ways to choose a pair of two different colors -/
def color_pair_choices (m : ℕ) : ℕ := m * (m - 1) / 2

/-- The total number of unique representations -/
def total_representations (m : ℕ) : ℕ := single_color_choices m + color_pair_choices m

/-- Theorem: Given 5 colors, the maximum number of unique representations is 15 -/
theorem max_representations : total_representations n = 15 := by
  sorry

end NUMINAMATH_CALUDE_max_representations_l449_44953


namespace NUMINAMATH_CALUDE_tanner_savings_l449_44934

/-- The amount of money Tanner is left with after saving and spending -/
def money_left (september_savings october_savings november_savings video_game_cost : ℕ) : ℕ :=
  september_savings + october_savings + november_savings - video_game_cost

/-- Theorem stating that Tanner is left with $41 -/
theorem tanner_savings : 
  money_left 17 48 25 49 = 41 := by
  sorry

end NUMINAMATH_CALUDE_tanner_savings_l449_44934


namespace NUMINAMATH_CALUDE_bathtub_fill_time_l449_44985

/-- Represents the filling and draining rates of a bathtub -/
structure BathtubRates where
  cold_fill_time : ℚ
  hot_fill_time : ℚ
  drain_time : ℚ

/-- Calculates the time to fill the bathtub with both taps open and drain unplugged -/
def fill_time (rates : BathtubRates) : ℚ :=
  1 / ((1 / rates.cold_fill_time) + (1 / rates.hot_fill_time) - (1 / rates.drain_time))

/-- Theorem: Given the specified filling and draining rates, the bathtub will fill in 5 minutes -/
theorem bathtub_fill_time (rates : BathtubRates) 
  (h1 : rates.cold_fill_time = 20 / 3)
  (h2 : rates.hot_fill_time = 8)
  (h3 : rates.drain_time = 40 / 3) :
  fill_time rates = 5 := by
  sorry

#eval fill_time { cold_fill_time := 20 / 3, hot_fill_time := 8, drain_time := 40 / 3 }

end NUMINAMATH_CALUDE_bathtub_fill_time_l449_44985


namespace NUMINAMATH_CALUDE_cake_distribution_l449_44952

theorem cake_distribution (n : ℕ) (most least : ℚ) : 
  most = 1/11 → least = 1/14 → (∀ x, least ≤ x ∧ x ≤ most) → 
  (n : ℚ) * least ≤ 1 ∧ 1 ≤ (n : ℚ) * most → n = 12 ∨ n = 13 := by
  sorry

#check cake_distribution

end NUMINAMATH_CALUDE_cake_distribution_l449_44952


namespace NUMINAMATH_CALUDE_remainder_sum_of_powers_l449_44955

theorem remainder_sum_of_powers (n : ℕ) : (9^4 + 8^5 + 7^6) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_of_powers_l449_44955


namespace NUMINAMATH_CALUDE_unique_consecutive_sum_20_l449_44900

/-- A set of consecutive positive integers -/
def ConsecutiveSet (start : ℕ) (length : ℕ) : Set ℕ :=
  {n : ℕ | start ≤ n ∧ n < start + length}

/-- The sum of a set of consecutive positive integers -/
def ConsecutiveSum (start : ℕ) (length : ℕ) : ℕ :=
  (length * (2 * start + length - 1)) / 2

/-- Theorem: There exists exactly one set of consecutive positive integers with sum 20 -/
theorem unique_consecutive_sum_20 : 
  ∃! p : ℕ × ℕ, 2 ≤ p.2 ∧ ConsecutiveSum p.1 p.2 = 20 :=
sorry

end NUMINAMATH_CALUDE_unique_consecutive_sum_20_l449_44900


namespace NUMINAMATH_CALUDE_sock_selection_with_red_l449_44981

def total_socks : ℕ := 7
def socks_to_select : ℕ := 3

theorem sock_selection_with_red (total_socks : ℕ) (socks_to_select : ℕ) : 
  total_socks = 7 → socks_to_select = 3 → 
  (Nat.choose total_socks socks_to_select) - (Nat.choose (total_socks - 1) socks_to_select) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_with_red_l449_44981


namespace NUMINAMATH_CALUDE_uvw_sum_squared_product_bound_l449_44922

theorem uvw_sum_squared_product_bound (u v w : ℝ) 
  (h_nonneg : u ≥ 0 ∧ v ≥ 0 ∧ w ≥ 0) 
  (h_sum : u + v + w = 2) : 
  0 ≤ u^2 * v^2 + v^2 * w^2 + w^2 * u^2 ∧ 
  u^2 * v^2 + v^2 * w^2 + w^2 * u^2 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_uvw_sum_squared_product_bound_l449_44922


namespace NUMINAMATH_CALUDE_series_sum_equals_one_l449_44959

open Real

noncomputable def seriesSum : ℝ := ∑' k, (k^2 : ℝ) / 3^k

theorem series_sum_equals_one : seriesSum = 1 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_l449_44959


namespace NUMINAMATH_CALUDE_chord_length_theorem_l449_44942

/-- Representation of a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Check if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c2.radius - c1.radius)^2

/-- Check if three points are collinear -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem chord_length_theorem (C1 C2 C3 : Circle) : 
  are_externally_tangent C1 C2 →
  is_internally_tangent C1 C3 →
  is_internally_tangent C2 C3 →
  C1.radius = 3 →
  C2.radius = 9 →
  are_collinear C1.center C2.center C3.center →
  ∃ (chord : ℝ), chord = 6 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_theorem_l449_44942


namespace NUMINAMATH_CALUDE_circle_center_radius_l449_44944

theorem circle_center_radius (x y : ℝ) :
  x^2 + y^2 - 4*x = 0 → (∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (2, 0) ∧ radius = 2 ∧ 
    (x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_l449_44944


namespace NUMINAMATH_CALUDE_largest_prime_divisor_test_l449_44973

theorem largest_prime_divisor_test (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1050) :
  Prime n → ∀ p, Prime p ∧ p > 31 → ¬(p ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_test_l449_44973


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l449_44997

def base_seven_to_decimal (n : ℕ) : ℕ := 
  2 * 7^6 + 1 * 7^5 + 0 * 7^4 + 2 * 7^3 + 0 * 7^2 + 1 * 7^1 + 2 * 7^0

def number : ℕ := base_seven_to_decimal 2102012

theorem largest_prime_divisor :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ number ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ number → q ≤ p :=
by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l449_44997


namespace NUMINAMATH_CALUDE_six_digit_number_divisibility_l449_44935

theorem six_digit_number_divisibility (W : ℕ) :
  (100000 ≤ W) ∧ (W < 1000000) ∧
  (∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    W = 100000*a + 10000*b + 1000*c + 200*a + 20*b + 2*c) →
  2 ∣ W :=
by sorry

end NUMINAMATH_CALUDE_six_digit_number_divisibility_l449_44935


namespace NUMINAMATH_CALUDE_spaceship_reach_boundary_l449_44988

/-- A path in 3D space --/
structure Path3D where
  points : List (ℝ × ℝ × ℝ)

/-- Distance of a point from a plane --/
def distanceFromPlane (point : ℝ × ℝ × ℝ) (plane : ℝ × ℝ × ℝ → ℝ) : ℝ :=
  sorry

/-- Length of a path --/
def pathLength (path : Path3D) : ℝ :=
  sorry

/-- Check if a path reaches the boundary plane --/
def reachesBoundary (path : Path3D) (boundaryPlane : ℝ × ℝ × ℝ → ℝ) : Prop :=
  sorry

/-- The main theorem --/
theorem spaceship_reach_boundary (a : ℝ) (startPoint : ℝ × ℝ × ℝ) (boundaryPlane : ℝ × ℝ × ℝ → ℝ) 
    (h : distanceFromPlane startPoint boundaryPlane = a) :
    ∃ (path : Path3D), pathLength path ≤ 14 * a ∧ reachesBoundary path boundaryPlane :=
  sorry

end NUMINAMATH_CALUDE_spaceship_reach_boundary_l449_44988


namespace NUMINAMATH_CALUDE_fraction_of_5000_l449_44921

theorem fraction_of_5000 : 
  ∃ (f : ℚ), (f * (1/2 * (2/5 * 5000)) = 750.0000000000001) ∧ (f = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_5000_l449_44921


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l449_44915

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (x - 1/x) - 2 * Real.log x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 2 + 2/(x^2) - 2/x

-- Theorem statement
theorem tangent_line_at_one (x y : ℝ) :
  (y = f x) → (x = 1) → (2*x - y - 2 = 0) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_one_l449_44915


namespace NUMINAMATH_CALUDE_point_division_ratios_l449_44938

/-- Given two points A and B on a line, there exist points M and N such that
    AM:MB = 2:1 and AN:NB = 1:3 respectively. -/
theorem point_division_ratios (A B : ℝ) : 
  (∃ M : ℝ, |A - M| / |M - B| = 2) ∧ 
  (∃ N : ℝ, |A - N| / |N - B| = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_point_division_ratios_l449_44938


namespace NUMINAMATH_CALUDE_p_is_true_l449_44947

theorem p_is_true (h1 : ¬(p ∧ q)) (h2 : ¬(¬p)) : p := by
  sorry

end NUMINAMATH_CALUDE_p_is_true_l449_44947


namespace NUMINAMATH_CALUDE_coefficient_x90_is_minus_one_l449_44963

/-- The sequence of factors in the polynomial expansion -/
def factors : List (ℕ → ℤ) := [
  (λ n => if n = 1 then -1 else 1),
  (λ n => if n = 2 then -2 else 1),
  (λ n => if n = 3 then -3 else 1),
  (λ n => if n = 4 then -4 else 1),
  (λ n => if n = 5 then -5 else 1),
  (λ n => if n = 6 then -6 else 1),
  (λ n => if n = 7 then -7 else 1),
  (λ n => if n = 8 then -8 else 1),
  (λ n => if n = 9 then -9 else 1),
  (λ n => if n = 10 then -10 else 1),
  (λ n => if n = 11 then -11 else 1),
  (λ n => if n = 13 then -13 else 1)
]

/-- The coefficient of x^90 in the expansion -/
def coefficient_x90 : ℤ := -1

/-- Theorem stating that the coefficient of x^90 in the expansion is -1 -/
theorem coefficient_x90_is_minus_one :
  coefficient_x90 = -1 := by sorry

end NUMINAMATH_CALUDE_coefficient_x90_is_minus_one_l449_44963


namespace NUMINAMATH_CALUDE_polyline_segment_bound_l449_44939

/-- Represents a grid paper with square side length 1 -/
structure GridPaper where
  -- Additional structure properties can be added if needed

/-- Represents a point on the grid paper -/
structure GridPoint where
  -- Additional point properties can be added if needed

/-- Represents a polyline segment on the grid paper -/
structure PolylineSegment where
  start : GridPoint
  length : ℕ
  -- Additional segment properties can be added if needed

/-- 
  P_k denotes the number of different polyline segments of length k 
  starting from a fixed point O on a grid paper, where each segment 
  lies along the grid lines
-/
def P (grid : GridPaper) (O : GridPoint) (k : ℕ) : ℕ :=
  sorry -- Definition of P_k

/-- 
  Theorem: For all natural numbers k, the number of different polyline 
  segments of length k starting from a fixed point O on a grid paper 
  with square side length 1, where each segment lies along the grid lines, 
  is less than 2 × 3^k
-/
theorem polyline_segment_bound 
  (grid : GridPaper) (O : GridPoint) : 
  ∀ k : ℕ, P grid O k < 2 * 3^k := by
  sorry


end NUMINAMATH_CALUDE_polyline_segment_bound_l449_44939


namespace NUMINAMATH_CALUDE_well_climbing_slip_distance_l449_44912

/-- Calculates the daily slip distance in a well-climbing scenario -/
theorem well_climbing_slip_distance
  (well_depth : ℝ)
  (daily_climb : ℝ)
  (days_to_exit : ℕ)
  (h_depth : well_depth = 30)
  (h_climb : daily_climb = 4)
  (h_days : days_to_exit = 27) :
  ∃ (slip_distance : ℝ),
    slip_distance = 3 ∧
    well_depth = (daily_climb - slip_distance) * (days_to_exit - 1 : ℝ) + daily_climb :=
by sorry

end NUMINAMATH_CALUDE_well_climbing_slip_distance_l449_44912


namespace NUMINAMATH_CALUDE_x_minus_y_value_l449_44910

theorem x_minus_y_value (x y : ℚ) 
  (eq1 : 3015 * x + 3020 * y = 3024)
  (eq2 : 3017 * x + 3022 * y = 3026) : 
  x - y = -13/5 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l449_44910


namespace NUMINAMATH_CALUDE_gcd_lcm_product_75_125_l449_44994

theorem gcd_lcm_product_75_125 : 
  (Nat.gcd 75 125) * (Nat.lcm 75 125) = 9375 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_75_125_l449_44994


namespace NUMINAMATH_CALUDE_triangle_count_l449_44975

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of points on the circle -/
def num_points : ℕ := 9

/-- The number of points needed to form a triangle -/
def points_per_triangle : ℕ := 3

/-- The number of different triangles that can be formed -/
def num_triangles : ℕ := binomial num_points points_per_triangle

theorem triangle_count : num_triangles = 84 := by sorry

end NUMINAMATH_CALUDE_triangle_count_l449_44975


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_min_value_attained_l449_44930

/-- Given two parallel vectors a and b, where a = (1,3) and b = (x,1-y),
    and x and y are positive real numbers, 
    the minimum value of 3/x + 1/y is 16 -/
theorem min_value_parallel_vectors (x y : ℝ) : 
  x > 0 → y > 0 → (1 : ℝ) / x = 3 / (1 - y) → (3 / x + 1 / y) ≥ 16 := by
  sorry

/-- The minimum value is attained when 3/x + 1/y = 16 -/
theorem min_value_attained (x y : ℝ) : 
  x > 0 → y > 0 → (1 : ℝ) / x = 3 / (1 - y) → 
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ (1 : ℝ) / x₀ = 3 / (1 - y₀) ∧ 3 / x₀ + 1 / y₀ = 16) := by
  sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_min_value_attained_l449_44930


namespace NUMINAMATH_CALUDE_prime_power_equation_l449_44992

theorem prime_power_equation (p : ℕ) (x y : ℕ) (h_prime : Nat.Prime p) (h_eq : x^4 - y^4 = p * (x^3 - y^3)) :
  (x = 0 ∧ y = p) ∨ (x = p ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_prime_power_equation_l449_44992


namespace NUMINAMATH_CALUDE_weight_difference_l449_44978

theorem weight_difference (jim steve stan : ℕ) 
  (h1 : steve < stan)
  (h2 : steve = jim - 8)
  (h3 : jim = 110)
  (h4 : stan + steve + jim = 319) :
  stan - steve = 5 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l449_44978


namespace NUMINAMATH_CALUDE_path_area_is_775_l449_44971

/-- Represents the dimensions of a rectangular field with a surrounding path -/
structure FieldWithPath where
  field_length : ℝ
  field_width : ℝ
  path_width : ℝ

/-- Calculates the area of the path surrounding a rectangular field -/
def path_area (f : FieldWithPath) : ℝ :=
  (f.field_length + 2 * f.path_width) * (f.field_width + 2 * f.path_width) -
  f.field_length * f.field_width

/-- Theorem stating that the area of the path for the given field dimensions is 775 sq m -/
theorem path_area_is_775 :
  let f : FieldWithPath := {
    field_length := 95,
    field_width := 55,
    path_width := 2.5
  }
  path_area f = 775 := by sorry

end NUMINAMATH_CALUDE_path_area_is_775_l449_44971


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l449_44961

/-- The cost of chocolate bars for a scout camp -/
theorem chocolate_bar_cost (chocolate_bar_cost : ℝ) (sections_per_bar : ℕ) (num_scouts : ℕ) (smores_per_scout : ℕ) : 
  chocolate_bar_cost = 1.5 →
  sections_per_bar = 3 →
  num_scouts = 15 →
  smores_per_scout = 2 →
  (num_scouts * smores_per_scout : ℝ) / sections_per_bar * chocolate_bar_cost = 15 := by
  sorry

#check chocolate_bar_cost

end NUMINAMATH_CALUDE_chocolate_bar_cost_l449_44961


namespace NUMINAMATH_CALUDE_find_n_l449_44996

theorem find_n : ∃ n : ℕ, (1/5 : ℝ)^n * (1/4 : ℝ)^18 = 1 / (2 * (10 : ℝ)^35) → n = 35 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l449_44996


namespace NUMINAMATH_CALUDE_equal_sums_exist_l449_44972

/-- Represents a 3x3 grid with values from {-1, 0, 1} -/
def Grid := Matrix (Fin 3) (Fin 3) (Fin 3)

/-- Computes the sum of a row in the grid -/
def rowSum (g : Grid) (i : Fin 3) : ℤ := sorry

/-- Computes the sum of a column in the grid -/
def colSum (g : Grid) (j : Fin 3) : ℤ := sorry

/-- Computes the sum of the main diagonal -/
def mainDiagSum (g : Grid) : ℤ := sorry

/-- Computes the sum of the anti-diagonal -/
def antiDiagSum (g : Grid) : ℤ := sorry

/-- All possible sums in the grid -/
def allSums (g : Grid) : List ℤ := 
  [rowSum g 0, rowSum g 1, rowSum g 2, 
   colSum g 0, colSum g 1, colSum g 2, 
   mainDiagSum g, antiDiagSum g]

theorem equal_sums_exist (g : Grid) : 
  ∃ (i j : Fin 8), i ≠ j ∧ (allSums g).get i = (allSums g).get j := by sorry

end NUMINAMATH_CALUDE_equal_sums_exist_l449_44972


namespace NUMINAMATH_CALUDE_original_number_l449_44956

theorem original_number (x : ℝ) : (x * 1.2 = 480) → x = 400 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l449_44956


namespace NUMINAMATH_CALUDE_power_mod_23_l449_44967

theorem power_mod_23 : 17^1988 % 23 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_23_l449_44967


namespace NUMINAMATH_CALUDE_intersection_locus_l449_44916

/-- The locus of the intersection point of two lines in a Cartesian coordinate system -/
theorem intersection_locus (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) :
  ∀ (x y : ℝ), 
  (∃ c : ℝ, c ≠ 0 ∧ 
    (y = (a / c) * x) ∧ 
    (x / b + y / c = 1)) →
  ((x - b / 2)^2 / (b^2 / 4) + y^2 / (a * b / 4) = 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_locus_l449_44916


namespace NUMINAMATH_CALUDE_max_stamps_purchasable_l449_44911

def stamp_price : ℕ := 37
def available_funds : ℕ := 4000  -- $40 in cents

theorem max_stamps_purchasable : 
  ∀ n : ℕ, n * stamp_price ≤ available_funds ∧ 
  (∀ m : ℕ, m * stamp_price ≤ available_funds → m ≤ n) → 
  n = 108 := by
sorry

end NUMINAMATH_CALUDE_max_stamps_purchasable_l449_44911


namespace NUMINAMATH_CALUDE_difference_61st_terms_arithmetic_sequences_l449_44990

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

theorem difference_61st_terms_arithmetic_sequences :
  let C := arithmetic_sequence 45 15
  let D := arithmetic_sequence 45 (-15)
  |C 61 - D 61| = 1800 := by
sorry

end NUMINAMATH_CALUDE_difference_61st_terms_arithmetic_sequences_l449_44990


namespace NUMINAMATH_CALUDE_volume_of_specific_open_box_l449_44928

/-- Calculates the volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
def openBoxVolume (sheetLength sheetWidth cutSize : ℝ) : ℝ :=
  (sheetLength - 2 * cutSize) * (sheetWidth - 2 * cutSize) * cutSize

/-- Theorem stating that the volume of the specific open box is 5120 m³. -/
theorem volume_of_specific_open_box :
  openBoxVolume 48 36 8 = 5120 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_open_box_l449_44928


namespace NUMINAMATH_CALUDE_queen_then_diamond_probability_l449_44936

/-- Standard deck of cards --/
def standard_deck : ℕ := 52

/-- Number of Queens in a standard deck --/
def num_queens : ℕ := 4

/-- Number of diamonds in a standard deck --/
def num_diamonds : ℕ := 13

/-- Probability of drawing a Queen as the first card and a diamond as the second --/
def prob_queen_then_diamond : ℚ := 52 / 221

theorem queen_then_diamond_probability :
  prob_queen_then_diamond = (num_queens / standard_deck) * (num_diamonds / (standard_deck - 1)) :=
sorry

end NUMINAMATH_CALUDE_queen_then_diamond_probability_l449_44936


namespace NUMINAMATH_CALUDE_total_watching_time_l449_44989

-- Define the TV series data
def pride_and_prejudice_episodes : ℕ := 6
def pride_and_prejudice_duration : ℕ := 50
def breaking_bad_episodes : ℕ := 62
def breaking_bad_duration : ℕ := 47
def stranger_things_episodes : ℕ := 33
def stranger_things_duration : ℕ := 51

-- Calculate total watching time in minutes
def total_minutes : ℕ := 
  pride_and_prejudice_episodes * pride_and_prejudice_duration +
  breaking_bad_episodes * breaking_bad_duration +
  stranger_things_episodes * stranger_things_duration

-- Convert minutes to hours and round to nearest whole number
def total_hours : ℕ := (total_minutes + 30) / 60

-- Theorem to prove
theorem total_watching_time : total_hours = 82 := by
  sorry

end NUMINAMATH_CALUDE_total_watching_time_l449_44989


namespace NUMINAMATH_CALUDE_savings_is_84_l449_44923

/-- Represents the pricing structure and needs for a window purchase scenario -/
structure WindowPurchase where
  regular_price : ℕ  -- Regular price per window
  free_window_threshold : ℕ  -- Number of windows purchased to get one free
  bulk_discount_threshold : ℕ  -- Number of windows needed for bulk discount
  bulk_discount_rate : ℚ  -- Discount rate for bulk purchases
  alice_needs : ℕ  -- Number of windows Alice needs
  bob_needs : ℕ  -- Number of windows Bob needs

/-- Calculates the price for a given number of windows -/
def calculate_price (wp : WindowPurchase) (num_windows : ℕ) : ℚ :=
  let paid_windows := num_windows - (num_windows / wp.free_window_threshold)
  let base_price := (paid_windows * wp.regular_price : ℚ)
  if num_windows ≥ wp.bulk_discount_threshold
  then base_price * (1 - wp.bulk_discount_rate)
  else base_price

/-- Calculates the savings when purchasing windows together versus separately -/
def savings_difference (wp : WindowPurchase) : ℚ :=
  let separate_cost := calculate_price wp wp.alice_needs + calculate_price wp wp.bob_needs
  let combined_cost := calculate_price wp (wp.alice_needs + wp.bob_needs)
  (wp.alice_needs + wp.bob_needs) * wp.regular_price - separate_cost - 
  ((wp.alice_needs + wp.bob_needs) * wp.regular_price - combined_cost)

/-- The main theorem stating the savings difference -/
theorem savings_is_84 (wp : WindowPurchase) 
  (h1 : wp.regular_price = 120)
  (h2 : wp.free_window_threshold = 5)
  (h3 : wp.bulk_discount_threshold = 10)
  (h4 : wp.bulk_discount_rate = 1/10)
  (h5 : wp.alice_needs = 9)
  (h6 : wp.bob_needs = 11) : 
  savings_difference wp = 84 := by
  sorry

end NUMINAMATH_CALUDE_savings_is_84_l449_44923


namespace NUMINAMATH_CALUDE_geometric_propositions_l449_44913

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the basic relations
variable (belongs_to : Point → Line → Prop)
variable (lies_in : Point → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (intersect_lines : Line → Line → Prop)
variable (intersect_line_plane : Line → Plane → Prop)
variable (intersect_planes : Plane → Plane → Line)

-- Define the theorem
theorem geometric_propositions 
  (a b : Line) (α β γ : Plane) :
  (-- Proposition 2
   intersect_lines a b ∧ 
   (¬ line_in_plane a α) ∧ (¬ line_in_plane a β) ∧
   (¬ line_in_plane b α) ∧ (¬ line_in_plane b β) ∧
   parallel_line_plane a α ∧ parallel_line_plane a β ∧
   parallel_line_plane b α ∧ parallel_line_plane b β →
   parallel_planes α β) ∧
  (-- Proposition 3
   line_in_plane a α ∧ 
   parallel_line_plane a β ∧
   intersect_planes α β = b →
   parallel_lines a b) :=
sorry

end NUMINAMATH_CALUDE_geometric_propositions_l449_44913


namespace NUMINAMATH_CALUDE_no_solution_for_room_occupancy_l449_44933

theorem no_solution_for_room_occupancy : ¬∃ (x y : ℕ), x + y = 2019 ∧ 2 * x - y = 2018 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_room_occupancy_l449_44933


namespace NUMINAMATH_CALUDE_cos_value_from_tan_sin_relation_l449_44979

theorem cos_value_from_tan_sin_relation (θ : Real) 
  (h1 : 6 * Real.tan θ = 5 * Real.sin θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.cos θ = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_cos_value_from_tan_sin_relation_l449_44979


namespace NUMINAMATH_CALUDE_range_of_sum_l449_44926

theorem range_of_sum (a b c : ℝ) 
  (sum_condition : a + b + c = 1) 
  (square_sum_condition : a^2 + b^2 + c^2 = 1) : 
  0 ≤ a + b ∧ a + b ≤ 4/3 := by
sorry

end NUMINAMATH_CALUDE_range_of_sum_l449_44926


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l449_44908

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l449_44908


namespace NUMINAMATH_CALUDE_common_root_implies_zero_l449_44969

theorem common_root_implies_zero (a b : ℝ) : 
  (∃ r : ℝ, r^2 + a*r + b^2 = 0 ∧ r^2 + b*r + a^2 = 0) → 
  ¬(a ≠ 0 ∧ b ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_common_root_implies_zero_l449_44969


namespace NUMINAMATH_CALUDE_inner_triangle_perimeter_l449_44977

theorem inner_triangle_perimeter (a : ℝ) (h : a = 8) :
  let outer_leg := a
  let inner_leg := a - 1
  let inner_hypotenuse := inner_leg * Real.sqrt 2
  let inner_perimeter := 2 * inner_leg + inner_hypotenuse
  inner_perimeter = 14 + 7 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inner_triangle_perimeter_l449_44977


namespace NUMINAMATH_CALUDE_square_root_subtraction_l449_44940

theorem square_root_subtraction : Real.sqrt 81 - Real.sqrt 144 * 3 = -27 := by
  sorry

end NUMINAMATH_CALUDE_square_root_subtraction_l449_44940


namespace NUMINAMATH_CALUDE_function_value_range_l449_44987

theorem function_value_range (a : ℝ) : 
  (∃ x y : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧ y ∈ Set.Icc (-1 : ℝ) 1 ∧ 
   (a * x + 2 * a + 1) * (a * y + 2 * a + 1) < 0) ↔ 
  a ∈ Set.Ioo (-(1/3) : ℝ) (-1) :=
by sorry

end NUMINAMATH_CALUDE_function_value_range_l449_44987


namespace NUMINAMATH_CALUDE_good_carrots_count_l449_44943

theorem good_carrots_count (nancy_carrots : ℕ) (mom_carrots : ℕ) (bad_carrots : ℕ) :
  nancy_carrots = 38 →
  mom_carrots = 47 →
  bad_carrots = 14 →
  nancy_carrots + mom_carrots - bad_carrots = 71 := by
sorry

end NUMINAMATH_CALUDE_good_carrots_count_l449_44943


namespace NUMINAMATH_CALUDE_summer_program_students_l449_44945

theorem summer_program_students : ∃! n : ℕ, 0 < n ∧ n < 500 ∧ n % 25 = 24 ∧ n % 21 = 14 ∧ n = 449 := by
  sorry

end NUMINAMATH_CALUDE_summer_program_students_l449_44945


namespace NUMINAMATH_CALUDE_roots_of_equation_l449_44951

theorem roots_of_equation (x : ℝ) : 
  (x - 3)^2 = 4 ↔ x = 5 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l449_44951


namespace NUMINAMATH_CALUDE_wire_ratio_l449_44966

/-- Given a wire of total length 60 cm with a shorter piece of 20 cm,
    prove that the ratio of the shorter piece to the longer piece is 1/2. -/
theorem wire_ratio (total_length : ℝ) (shorter_piece : ℝ) 
  (h1 : total_length = 60)
  (h2 : shorter_piece = 20)
  (h3 : shorter_piece < total_length) :
  shorter_piece / (total_length - shorter_piece) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_wire_ratio_l449_44966


namespace NUMINAMATH_CALUDE_max_min_x_plus_y_l449_44965

theorem max_min_x_plus_y (x y : ℝ) :
  (|x + 2| + |1 - x| = 9 - |y - 5| - |1 + y|) →
  (∃ (a b : ℝ), (∀ z w : ℝ, |z + 2| + |1 - z| = 9 - |w - 5| - |1 + w| → x + y ≤ a ∧ b ≤ z + w) ∧
                 a = 6 ∧ b = -3) :=
by sorry

end NUMINAMATH_CALUDE_max_min_x_plus_y_l449_44965


namespace NUMINAMATH_CALUDE_subsets_common_element_probability_l449_44968

def S : Finset ℕ := {1, 2, 3, 4}

theorem subsets_common_element_probability :
  let subsets := S.powerset
  let total_pairs := subsets.card * subsets.card
  let disjoint_pairs := (Finset.range 5).sum (λ k => (Nat.choose 4 k) * (2^(4 - k)))
  (total_pairs - disjoint_pairs) / total_pairs = 175 / 256 := by
sorry

end NUMINAMATH_CALUDE_subsets_common_element_probability_l449_44968


namespace NUMINAMATH_CALUDE_product_of_three_digit_numbers_l449_44909

theorem product_of_three_digit_numbers : ∃ (I K S : Nat), 
  (I ≠ 0 ∧ K ≠ 0 ∧ S ≠ 0) ∧  -- non-zero digits
  (I ≠ K ∧ K ≠ S ∧ I ≠ S) ∧  -- distinct digits
  (I < 10 ∧ K < 10 ∧ S < 10) ∧  -- single digits
  ((100 * I + 10 * K + S) * (100 * K + 10 * S + I) = 100602) ∧  -- product
  (100602 % 10 = S) ∧  -- ends with S
  (100602 / 100 = I * 10 + K) ∧  -- after removing zeros, IKS remains
  (S = 2 ∧ K = 6 ∧ I = 1)  -- specific values that satisfy the conditions
:= by sorry

end NUMINAMATH_CALUDE_product_of_three_digit_numbers_l449_44909


namespace NUMINAMATH_CALUDE_g_sum_equals_negative_two_l449_44957

/-- Piecewise function g(x, y) -/
noncomputable def g (x y : ℝ) : ℝ :=
  if x - y ≤ 1 then (x^2 * y - x + 3) / (3 * x)
  else (x^2 * y - y - 3) / (-3 * y)

/-- Theorem stating that g(3,2) + g(4,1) = -2 -/
theorem g_sum_equals_negative_two : g 3 2 + g 4 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_equals_negative_two_l449_44957


namespace NUMINAMATH_CALUDE_students_per_table_unchanged_l449_44902

theorem students_per_table_unchanged (tables : ℝ) (initial_students_per_table : ℝ) 
  (h1 : tables = 34.0) 
  (h2 : initial_students_per_table = 6.0) : 
  (tables * initial_students_per_table) / tables = initial_students_per_table := by
  sorry

#check students_per_table_unchanged

end NUMINAMATH_CALUDE_students_per_table_unchanged_l449_44902


namespace NUMINAMATH_CALUDE_mikes_coins_value_l449_44958

/-- Represents the number of coins Mike has -/
def total_coins : ℕ := 17

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Calculates the total value of Mike's coins in cents -/
def total_value (dimes quarters : ℕ) : ℕ :=
  dimes * dime_value + quarters * quarter_value

theorem mikes_coins_value :
  ∃ (dimes quarters : ℕ),
    dimes + quarters = total_coins ∧
    quarters + 3 = 2 * dimes ∧
    total_value dimes quarters = 345 := by
  sorry

end NUMINAMATH_CALUDE_mikes_coins_value_l449_44958


namespace NUMINAMATH_CALUDE_no_integer_solutions_for_fermat_like_equation_l449_44974

theorem no_integer_solutions_for_fermat_like_equation (n : ℕ) (hn : n ≥ 2) :
  ¬∃ (x y z : ℤ), x^2 + y^2 = z^n := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_for_fermat_like_equation_l449_44974


namespace NUMINAMATH_CALUDE_hayden_ironing_time_l449_44998

/-- The total ironing time over 4 weeks given daily ironing times and weekly frequency -/
def total_ironing_time (shirt_time pants_time days_per_week num_weeks : ℕ) : ℕ :=
  (shirt_time + pants_time) * days_per_week * num_weeks

/-- Theorem stating that Hayden's total ironing time over 4 weeks is 160 minutes -/
theorem hayden_ironing_time :
  total_ironing_time 5 3 5 4 = 160 := by
  sorry

#eval total_ironing_time 5 3 5 4

end NUMINAMATH_CALUDE_hayden_ironing_time_l449_44998


namespace NUMINAMATH_CALUDE_third_vertex_coordinates_l449_44999

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem third_vertex_coordinates (y : ℝ) :
  y < 0 →
  triangleArea ⟨8, 6⟩ ⟨0, 0⟩ ⟨0, y⟩ = 48 →
  y = -12 := by
  sorry

end NUMINAMATH_CALUDE_third_vertex_coordinates_l449_44999


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l449_44991

theorem diophantine_equation_solutions (k x y z : ℕ+) : 
  (x^2 + y^2 + z^2 = k * x * y * z) ↔ (k = 1 ∨ k = 3) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l449_44991


namespace NUMINAMATH_CALUDE_quadratic_inequality_l449_44995

theorem quadratic_inequality (x : ℝ) : x^2 - 3*x - 18 < 0 ↔ -3 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l449_44995


namespace NUMINAMATH_CALUDE_vector_b_coordinates_l449_44918

theorem vector_b_coordinates (a b : ℝ × ℝ) :
  a = (Real.sqrt 3, Real.sqrt 5) →
  (a.1 * b.1 + a.2 * b.2 = 0) →
  (b.1^2 + b.2^2 = 4) →
  (b = (-Real.sqrt 10 / 2, Real.sqrt 6 / 2) ∨ b = (Real.sqrt 10 / 2, -Real.sqrt 6 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_vector_b_coordinates_l449_44918


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l449_44924

theorem lcm_gcd_problem (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 18) :
  ∃ (a' c' : ℕ), Nat.lcm a' b = 20 ∧ Nat.lcm b c' = 18 ∧
    ∀ (x y : ℕ), Nat.lcm x b = 20 → Nat.lcm b y = 18 →
      Nat.lcm a' c' + 2 * Nat.gcd a' c' ≤ Nat.lcm x y + 2 * Nat.gcd x y :=
by sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l449_44924


namespace NUMINAMATH_CALUDE_martha_initial_marbles_l449_44946

/-- Represents the number of marbles each person had initially -/
structure InitialMarbles where
  dilan : Nat
  martha : Nat
  phillip : Nat
  veronica : Nat

/-- Represents the state of marbles before and after redistribution -/
structure MarbleDistribution where
  initial : InitialMarbles
  final : Nat

def total_marbles (d : MarbleDistribution) : Nat :=
  d.initial.dilan + d.initial.martha + d.initial.phillip + d.initial.veronica

theorem martha_initial_marbles (d : MarbleDistribution) 
  (h1 : d.initial.dilan = 14)
  (h2 : d.initial.phillip = 19)
  (h3 : d.initial.veronica = 7)
  (h4 : d.final = 15)
  (h5 : total_marbles d = 4 * d.final) :
  d.initial.martha = 20 := by
  sorry

#check martha_initial_marbles

end NUMINAMATH_CALUDE_martha_initial_marbles_l449_44946


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1821_l449_44920

theorem smallest_prime_factor_of_1821 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1821 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1821 → p ≤ q :=
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1821_l449_44920


namespace NUMINAMATH_CALUDE_sum_squares_inequality_l449_44984

theorem sum_squares_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 3) :
  a + b + c ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_inequality_l449_44984


namespace NUMINAMATH_CALUDE_mark_additional_spending_l449_44962

def mark_spending (initial_amount : ℝ) (additional_first_store : ℝ) : Prop :=
  let half_spent := initial_amount / 2
  let remaining_after_half := initial_amount - half_spent
  let remaining_after_first := remaining_after_half - additional_first_store
  let third_spent := initial_amount / 3
  let remaining_after_second := remaining_after_first - third_spent - 16
  remaining_after_second = 0

theorem mark_additional_spending :
  mark_spending 180 14 := by sorry

end NUMINAMATH_CALUDE_mark_additional_spending_l449_44962


namespace NUMINAMATH_CALUDE_raft_distance_l449_44907

/-- Given a motorboat that travels downstream and upstream in equal time,
    this theorem proves the distance a raft travels with the stream. -/
theorem raft_distance (t : ℝ) (vb vs : ℝ) : t > 0 →
  (vb + vs) * t = 90 →
  (vb - vs) * t = 70 →
  vs * t = 10 := by
  sorry

end NUMINAMATH_CALUDE_raft_distance_l449_44907


namespace NUMINAMATH_CALUDE_focus_of_parabola_l449_44929

/-- A parabola is defined by the equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of a parabola is a point on its axis of symmetry -/
def Focus (p : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Theorem: The focus of the parabola y^2 = 4x is at the point (2, 0) -/
theorem focus_of_parabola :
  Focus Parabola = (2, 0) := by sorry

end NUMINAMATH_CALUDE_focus_of_parabola_l449_44929


namespace NUMINAMATH_CALUDE_polynomial_simplification_l449_44901

theorem polynomial_simplification (x : ℝ) :
  (14 * x^12 + 8 * x^9 + 3 * x^8) + (2 * x^14 - x^12 + 2 * x^9 + 5 * x^5 + 7 * x^2 + 6) =
  2 * x^14 + 13 * x^12 + 10 * x^9 + 3 * x^8 + 5 * x^5 + 7 * x^2 + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l449_44901


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l449_44925

theorem cubic_roots_sum (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) →
  (b^3 - 2*b^2 + 3*b - 4 = 0) →
  (c^3 - 2*c^2 + 3*c - 4 = 0) →
  a ≠ b → b ≠ c → a ≠ c →
  1/(a*(b^2 + c^2 - a^2)) + 1/(b*(c^2 + a^2 - b^2)) + 1/(c*(a^2 + b^2 - c^2)) = -1/8 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l449_44925


namespace NUMINAMATH_CALUDE_least_sum_of_valid_pair_l449_44948

def is_valid_pair (a b : ℕ+) : Prop :=
  Nat.gcd (a + b) 330 = 1 ∧
  (a : ℕ) ^ (a : ℕ) % (b : ℕ) ^ (b : ℕ) = 0 ∧
  (a : ℕ) % (b : ℕ) ≠ 0

theorem least_sum_of_valid_pair :
  ∃ (a b : ℕ+), is_valid_pair a b ∧
    ∀ (a' b' : ℕ+), is_valid_pair a' b' → a + b ≤ a' + b' ∧
    a + b = 357 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_valid_pair_l449_44948


namespace NUMINAMATH_CALUDE_smallest_gcd_qr_l449_44931

theorem smallest_gcd_qr (p q r : ℕ+) (h1 : Nat.gcd p q = 300) (h2 : Nat.gcd p r = 450) :
  ∃ (q' r' : ℕ+), Nat.gcd p q' = 300 ∧ Nat.gcd p r' = 450 ∧ Nat.gcd q' r' = 150 ∧
  ∀ (q'' r'' : ℕ+), Nat.gcd p q'' = 300 → Nat.gcd p r'' = 450 → Nat.gcd q'' r'' ≥ 150 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_qr_l449_44931


namespace NUMINAMATH_CALUDE_cube_surface_division_l449_44980

-- Define a cube
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 12)
  faces : Finset (Fin 6)
  bodyDiagonals : Finset (Fin 4)

-- Define a plane
structure Plane where
  normal : Vector ℝ 3

-- Define the function to erect planes perpendicular to body diagonals
def erectPerpendicularPlanes (c : Cube) : Finset Plane := sorry

-- Define the function to count surface parts
def countSurfaceParts (c : Cube) (planes : Finset Plane) : ℕ := sorry

-- Theorem statement
theorem cube_surface_division (c : Cube) :
  let perpendicularPlanes := erectPerpendicularPlanes c
  countSurfaceParts c perpendicularPlanes = 14 := by sorry

end NUMINAMATH_CALUDE_cube_surface_division_l449_44980


namespace NUMINAMATH_CALUDE_product_divisible_by_five_l449_44970

theorem product_divisible_by_five (a b : ℕ+) :
  (5 ∣ (a * b)) → ¬(¬(5 ∣ a) ∧ ¬(5 ∣ b)) := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_five_l449_44970


namespace NUMINAMATH_CALUDE_complex_number_properties_l449_44903

theorem complex_number_properties (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  ((a + b)^2 = a^2 + 2*a*b + b^2) ∧ 
  (a^2 = a*b → a = b) ∧ 
  (∃ c : ℂ, c ≠ 0 ∧ c + 1/c = 0) ∧
  (∃ x y : ℂ, Complex.abs x = Complex.abs y ∧ x ≠ y ∧ x ≠ -y) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l449_44903


namespace NUMINAMATH_CALUDE_group_contains_perfect_square_diff_l449_44986

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem group_contains_perfect_square_diff :
  ∀ (partition : Fin 3 → Set ℕ),
    (∀ n : ℕ, n ≤ 46 → ∃ i : Fin 3, n ∈ partition i) →
    (∀ i j : Fin 3, i ≠ j → partition i ∩ partition j = ∅) →
    (∀ i : Fin 3, partition i ⊆ Finset.range 47) →
    ∃ (i : Fin 3) (a b : ℕ), 
      a ∈ partition i ∧ 
      b ∈ partition i ∧ 
      a ≠ b ∧ 
      is_perfect_square (max a b - min a b) :=
by
  sorry

#check group_contains_perfect_square_diff

end NUMINAMATH_CALUDE_group_contains_perfect_square_diff_l449_44986


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l449_44976

/-- For a cube with volume 8y cubic units and surface area 6y square units, y = 64 -/
theorem cube_volume_surface_area (y : ℝ) (h1 : y > 0) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*y ∧ 6*s^2 = 6*y) → y = 64 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l449_44976


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l449_44932

/-- The ratio of volumes of cylinders formed from a rectangle --/
theorem cylinder_volume_ratio (w h : ℝ) (hw : w = 9) (hh : h = 12) :
  let v1 := π * (w / (2 * π))^2 * h
  let v2 := π * (h / (2 * π))^2 * w
  max v1 v2 / min v1 v2 = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l449_44932


namespace NUMINAMATH_CALUDE_tricycle_count_l449_44983

/-- Represents the number of vehicles of each type -/
structure VehicleCounts where
  bicycles : ℕ
  tricycles : ℕ
  scooters : ℕ

/-- The total number of children -/
def totalChildren : ℕ := 10

/-- The total number of wheels -/
def totalWheels : ℕ := 25

/-- Calculates the total number of children given the vehicle counts -/
def countChildren (v : VehicleCounts) : ℕ :=
  v.bicycles + v.tricycles + v.scooters

/-- Calculates the total number of wheels given the vehicle counts -/
def countWheels (v : VehicleCounts) : ℕ :=
  2 * v.bicycles + 3 * v.tricycles + v.scooters

/-- Theorem stating that the number of tricycles is 5 -/
theorem tricycle_count :
  ∃ (v : VehicleCounts),
    countChildren v = totalChildren ∧
    countWheels v = totalWheels ∧
    v.tricycles = 5 := by
  sorry

end NUMINAMATH_CALUDE_tricycle_count_l449_44983


namespace NUMINAMATH_CALUDE_workers_combined_efficiency_l449_44950

/-- Given two workers a and b, where a can finish a job in 18 days and b can finish the same job in half the time of a, prove that they can complete 1/6 of the job together in one day. -/
theorem workers_combined_efficiency (a b : ℝ) : 
  a > 0 → b > 0 → (a = 1 / 18) → (b = 2 * a) → a + b = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_workers_combined_efficiency_l449_44950


namespace NUMINAMATH_CALUDE_triangle_inequality_triangle_equality_l449_44905

/-- Triangle ABC with sides a, b, c, where a ≥ b and a ≥ c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_ge_b : a ≥ b
  a_ge_c : a ≥ c
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

/-- Circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

/-- Inradius of a triangle -/
def inradius (t : Triangle) : ℝ := sorry

/-- Length of centroidal axis from vertex A -/
def centroidal_axis_length (t : Triangle) : ℝ := sorry

/-- Altitude from vertex A to side BC -/
def altitude_a (t : Triangle) : ℝ := sorry

/-- A triangle is equilateral if all sides are equal -/
def is_equilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

theorem triangle_inequality (t : Triangle) :
  circumradius t / (2 * inradius t) ≥ centroidal_axis_length t / altitude_a t :=
sorry

theorem triangle_equality (t : Triangle) :
  circumradius t / (2 * inradius t) = centroidal_axis_length t / altitude_a t ↔ is_equilateral t :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_triangle_equality_l449_44905
