import Mathlib

namespace NUMINAMATH_CALUDE_no_such_hexagon_exists_l1556_155626

-- Define a hexagon as a collection of 6 points in 2D space
def Hexagon := (Fin 6 → ℝ × ℝ)

-- Define a predicate for convexity
def is_convex (h : Hexagon) : Prop := sorry

-- Define a predicate for a point being inside a hexagon
def is_inside (p : ℝ × ℝ) (h : Hexagon) : Prop := sorry

-- Define a function to calculate the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define a function to calculate the length of a hexagon side
def side_length (h : Hexagon) (i : Fin 6) : ℝ := 
  distance (h i) (h ((i + 1) % 6))

-- Theorem statement
theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon) (m : ℝ × ℝ),
    is_convex h ∧
    is_inside m h ∧
    (∀ i : Fin 6, side_length h i > 1) ∧
    (∀ i : Fin 6, distance m (h i) < 1) :=
sorry

end NUMINAMATH_CALUDE_no_such_hexagon_exists_l1556_155626


namespace NUMINAMATH_CALUDE_inverse_proportion_ratios_l1556_155698

/-- Given c is inversely proportional to d and c * d = k for a constant k,
    prove the relationship between d₁, d₂, and d₃. -/
theorem inverse_proportion_ratios 
  (c d : ℝ → ℝ) (k : ℝ) 
  (h_inverse : ∀ x, c x * d x = k) 
  (c₁ c₂ c₃ d₁ d₂ d₃ : ℝ) 
  (h_c_ratio : c₁ / c₂ = 4 / 5)
  (h_c₃ : c₃ = 2 * c₁) :
  d₁ / d₂ = 5 / 4 ∧ d₃ = d₁ / 2 := by
  sorry


end NUMINAMATH_CALUDE_inverse_proportion_ratios_l1556_155698


namespace NUMINAMATH_CALUDE_equidistant_point_coordinates_l1556_155654

/-- A point with coordinates (4-a, 2a+1) that has equal distances to both coordinate axes -/
structure EquidistantPoint where
  a : ℝ
  equal_distance : |4 - a| = |2*a + 1|

theorem equidistant_point_coordinates (P : EquidistantPoint) :
  (P.a = 1 ∧ (4 - P.a, 2*P.a + 1) = (3, 3)) ∨
  (P.a = -5 ∧ (4 - P.a, 2*P.a + 1) = (9, -9)) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_coordinates_l1556_155654


namespace NUMINAMATH_CALUDE_base_k_representation_l1556_155676

theorem base_k_representation (k : ℕ) (hk : k > 0) : 
  (8 : ℚ) / 63 = (k + 5 : ℚ) / (k^2 - 1) → k = 17 := by
  sorry

end NUMINAMATH_CALUDE_base_k_representation_l1556_155676


namespace NUMINAMATH_CALUDE_percentage_problem_l1556_155695

theorem percentage_problem (P : ℝ) : 
  P * 140 = (4/5) * 140 - 21 → P = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1556_155695


namespace NUMINAMATH_CALUDE_base8_digit_sum_l1556_155616

/-- Represents a digit in base 8 -/
def Digit8 : Type := { n : ℕ // n > 0 ∧ n < 8 }

/-- Converts a three-digit number in base 8 to its decimal equivalent -/
def toDecimal (p q r : Digit8) : ℕ := 64 * p.val + 8 * q.val + r.val

/-- The sum of three permutations of digits in base 8 -/
def sumPermutations (p q r : Digit8) : ℕ :=
  toDecimal p q r + toDecimal r q p + toDecimal q p r

/-- The value of PPP0 in base 8 -/
def ppp0 (p : Digit8) : ℕ := 512 * p.val + 64 * p.val + 8 * p.val

theorem base8_digit_sum (p q r : Digit8) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (h_sum : sumPermutations p q r = ppp0 p) : 
  q.val + r.val = 7 := by sorry

end NUMINAMATH_CALUDE_base8_digit_sum_l1556_155616


namespace NUMINAMATH_CALUDE_optimal_production_time_l1556_155623

def shaping_time : ℕ := 15
def firing_time : ℕ := 30
def total_items : ℕ := 75
def total_workers : ℕ := 13

def production_time (shaping_workers : ℕ) (firing_workers : ℕ) : ℕ :=
  let shaping_rounds := (total_items + shaping_workers - 1) / shaping_workers
  let firing_rounds := (total_items + firing_workers - 1) / firing_workers
  max (shaping_rounds * shaping_time) (firing_rounds * firing_time)

theorem optimal_production_time :
  ∃ (shaping_workers firing_workers : ℕ),
    shaping_workers + firing_workers = total_workers ∧
    ∀ (s f : ℕ), s + f = total_workers →
      production_time shaping_workers firing_workers ≤ production_time s f ∧
      production_time shaping_workers firing_workers = 325 :=
by sorry

end NUMINAMATH_CALUDE_optimal_production_time_l1556_155623


namespace NUMINAMATH_CALUDE_power_difference_equals_eight_l1556_155691

theorem power_difference_equals_eight : 4^2 - 2^3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equals_eight_l1556_155691


namespace NUMINAMATH_CALUDE_max_profit_l1556_155651

noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10 * x^2 + 300 * x
  else if x ≥ 40 then (901 * x^2 - 9450 * x + 10000) / x
  else 0

noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then -10 * x^2 + 600 * x - 260
  else if x ≥ 40 then (-x^2 + 9190 * x - 10000) / x
  else 0

theorem max_profit (x : ℝ) :
  (∀ y, y > 0 → W y ≤ W 100) ∧ W 100 = 8990 := by sorry

end NUMINAMATH_CALUDE_max_profit_l1556_155651


namespace NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l1556_155697

/-- The area of a right triangle with sides of length 8 and 3 is 12 -/
theorem right_triangle_area : ℝ → ℝ → ℝ → Prop :=
  fun side1 side2 area =>
    side1 = 8 ∧ side2 = 3 ∧ area = (1 / 2) * side1 * side2 → area = 12

/-- Proof of the theorem -/
theorem right_triangle_area_proof : right_triangle_area 8 3 12 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_right_triangle_area_proof_l1556_155697


namespace NUMINAMATH_CALUDE_lcm_of_330_and_210_l1556_155672

theorem lcm_of_330_and_210 (hcf : ℕ) (a b lcm : ℕ) : 
  hcf = 30 → a = 330 → b = 210 → lcm = Nat.lcm a b → lcm = 2310 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_330_and_210_l1556_155672


namespace NUMINAMATH_CALUDE_root_implies_u_value_l1556_155610

theorem root_implies_u_value (u : ℝ) :
  (3 * (((-15 - Real.sqrt 145) / 6) ^ 2) + 15 * ((-15 - Real.sqrt 145) / 6) + u = 0) →
  u = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_root_implies_u_value_l1556_155610


namespace NUMINAMATH_CALUDE_equation_solutions_l1556_155631

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, (x₁ = (-1 + Real.sqrt 13) / 3 ∧ x₂ = (-1 - Real.sqrt 13) / 3) ∧
    (3 * x₁^2 = 4 - 2*x₁ ∧ 3 * x₂^2 = 4 - 2*x₂)) ∧
  (∃ x₁ x₂ : ℝ, (x₁ = 7 ∧ x₂ = -8) ∧
    (x₁ * (x₁ - 7) = 8 * (7 - x₁) ∧ x₂ * (x₂ - 7) = 8 * (7 - x₂))) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1556_155631


namespace NUMINAMATH_CALUDE_range_of_a_l1556_155655

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 2 3 → (a * x^2 + 2 * y^2) / (x * y) - 1 > 0) → 
  a ∈ Set.Ioi (-1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1556_155655


namespace NUMINAMATH_CALUDE_smallest_a_value_l1556_155637

theorem smallest_a_value (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) 
  (h : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) :
  ∀ a' ≥ 0, (∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x)) → a' ≥ 17 := by
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l1556_155637


namespace NUMINAMATH_CALUDE_increase_decrease_threshold_l1556_155642

theorem increase_decrease_threshold (S x y : ℝ) 
  (hS : S > 0) (hxy : x > y) (hy : y > 0) : 
  ((S * (1 + x/100) + 15) * (1 - y/100) > S + 10) ↔ 
  (x > y + (x*y/100) + 500 - (1500*y/S)) :=
sorry

end NUMINAMATH_CALUDE_increase_decrease_threshold_l1556_155642


namespace NUMINAMATH_CALUDE_short_pencil_cost_proof_l1556_155687

/-- The cost of a short pencil in dollars -/
def short_pencil_cost : ℚ := 0.4

/-- The cost of a pencil with eraser in dollars -/
def eraser_pencil_cost : ℚ := 0.8

/-- The cost of a regular pencil in dollars -/
def regular_pencil_cost : ℚ := 0.5

/-- The number of pencils with eraser sold -/
def eraser_pencils_sold : ℕ := 200

/-- The number of regular pencils sold -/
def regular_pencils_sold : ℕ := 40

/-- The number of short pencils sold -/
def short_pencils_sold : ℕ := 35

/-- The total revenue from all sales in dollars -/
def total_revenue : ℚ := 194

theorem short_pencil_cost_proof :
  short_pencil_cost * short_pencils_sold +
  eraser_pencil_cost * eraser_pencils_sold +
  regular_pencil_cost * regular_pencils_sold = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_short_pencil_cost_proof_l1556_155687


namespace NUMINAMATH_CALUDE_subset_iff_a_eq_one_l1556_155661

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_iff_a_eq_one (a : ℝ) : A a ⊆ B a ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_iff_a_eq_one_l1556_155661


namespace NUMINAMATH_CALUDE_soda_transaction_result_l1556_155649

def soda_transaction (initial_cans : ℕ) : ℕ × ℕ :=
  let jeff_takes := 6
  let jeff_returns := jeff_takes / 2
  let after_jeff := initial_cans - jeff_takes + jeff_returns
  let tim_buys := after_jeff / 3
  let store_bonus := tim_buys / 4
  let after_store := after_jeff + tim_buys + store_bonus
  let sarah_takes := after_store / 5
  let end_of_day := after_store - sarah_takes
  let sarah_returns := sarah_takes * 2
  let next_day := end_of_day + sarah_returns
  (end_of_day, next_day)

theorem soda_transaction_result :
  soda_transaction 22 = (21, 31) := by sorry

end NUMINAMATH_CALUDE_soda_transaction_result_l1556_155649


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l1556_155696

theorem unique_modular_congruence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 17 ∧ -250 ≡ n [ZMOD 17] ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l1556_155696


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1556_155682

theorem rectangle_diagonal (perimeter : ℝ) (length_ratio width_ratio : ℕ) 
  (h_perimeter : perimeter = 72) 
  (h_ratio : length_ratio = 5 ∧ width_ratio = 4) : 
  ∃ (length width : ℝ),
    2 * (length + width) = perimeter ∧ 
    length * width_ratio = width * length_ratio ∧
    length^2 + width^2 = 656 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1556_155682


namespace NUMINAMATH_CALUDE_non_adjacent_arrangements_l1556_155629

def number_of_people : ℕ := 6

def number_of_gaps (n : ℕ) : ℕ := n + 1

def permutations (n : ℕ) : ℕ := n.factorial

def arrangements_with_gaps (n : ℕ) : ℕ :=
  permutations (n - 2) * (number_of_gaps (n - 2)).choose 2

theorem non_adjacent_arrangements :
  arrangements_with_gaps number_of_people = 480 := by sorry

end NUMINAMATH_CALUDE_non_adjacent_arrangements_l1556_155629


namespace NUMINAMATH_CALUDE_ping_pong_games_l1556_155624

theorem ping_pong_games (total_games : ℕ) (frankie_games carla_games : ℕ) : 
  total_games = 30 →
  frankie_games + carla_games = total_games →
  frankie_games = carla_games / 2 →
  carla_games = 20 := by
sorry

end NUMINAMATH_CALUDE_ping_pong_games_l1556_155624


namespace NUMINAMATH_CALUDE_sum_bc_equals_nine_l1556_155615

theorem sum_bc_equals_nine 
  (h1 : a + b = 16) 
  (h2 : c + d = 3) 
  (h3 : a + d = 10) : 
  b + c = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_bc_equals_nine_l1556_155615


namespace NUMINAMATH_CALUDE_custom_operation_properties_l1556_155619

-- Define the custom operation *
noncomputable def customMul (x y : ℝ) : ℝ :=
  if x = 0 then |y|
  else if y = 0 then |x|
  else if (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) then |x| + |y|
  else -(|x| + |y|)

-- Theorem statement
theorem custom_operation_properties :
  (∀ a : ℝ, customMul (-15) (customMul 3 0) = -18) ∧
  (∀ a : ℝ, 
    (a < 0 → customMul 3 a + a = 2 * a - 3) ∧
    (a = 0 → customMul 3 a + a = 3) ∧
    (a > 0 → customMul 3 a + a = 2 * a + 3)) :=
by sorry

end NUMINAMATH_CALUDE_custom_operation_properties_l1556_155619


namespace NUMINAMATH_CALUDE_second_polygon_sides_l1556_155685

theorem second_polygon_sides (p1 p2 : ℕ → ℝ) (n2 : ℕ) :
  (∀ k : ℕ, p1 k = p2 k) →  -- Same perimeter
  (p1 45 = 45 * (3 * p2 n2)) →  -- First polygon has 45 sides and 3 times the side length
  n2 * p2 n2 = p2 n2 * 135 →  -- Perimeter of second polygon
  n2 = 135 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l1556_155685


namespace NUMINAMATH_CALUDE_dinner_seating_arrangements_l1556_155663

theorem dinner_seating_arrangements (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 7) :
  (Nat.choose n k) * Nat.factorial (k - 1) = 25920 := by
  sorry

end NUMINAMATH_CALUDE_dinner_seating_arrangements_l1556_155663


namespace NUMINAMATH_CALUDE_min_cost_garden_l1556_155609

-- Define the plot dimensions
def plot1_area : ℝ := 5 * 2
def plot2_area : ℝ := 4 * 4
def plot3_area : ℝ := 3 * 3
def plot4_area : ℝ := 6 * 1
def plot5_area : ℝ := 4 * 1

-- Define flower costs
def sunflower_cost : ℝ := 0.75
def tulip_cost : ℝ := 1.25
def rose_cost : ℝ := 1.75
def orchid_cost : ℝ := 2.25
def hydrangea_cost : ℝ := 2.75

-- Define the total garden area
def total_area : ℝ := plot1_area + plot2_area + plot3_area + plot4_area + plot5_area

-- Theorem statement
theorem min_cost_garden (cost_function : ℝ → ℝ → ℝ) 
  (hf : ∀ (area : ℝ) (cost : ℝ), cost_function area cost = area * cost) :
  ∃ (a1 a2 a3 a4 a5 : ℝ),
    a1 + a2 + a3 + a4 + a5 = total_area ∧ 
    a1 ≥ 0 ∧ a2 ≥ 0 ∧ a3 ≥ 0 ∧ a4 ≥ 0 ∧ a5 ≥ 0 ∧
    cost_function a1 sunflower_cost + 
    cost_function a2 tulip_cost + 
    cost_function a3 rose_cost + 
    cost_function a4 orchid_cost + 
    cost_function a5 hydrangea_cost = 64.75 ∧
    ∀ (b1 b2 b3 b4 b5 : ℝ),
      b1 + b2 + b3 + b4 + b5 = total_area →
      b1 ≥ 0 → b2 ≥ 0 → b3 ≥ 0 → b4 ≥ 0 → b5 ≥ 0 →
      cost_function b1 sunflower_cost + 
      cost_function b2 tulip_cost + 
      cost_function b3 rose_cost + 
      cost_function b4 orchid_cost + 
      cost_function b5 hydrangea_cost ≥ 64.75 :=
by sorry

end NUMINAMATH_CALUDE_min_cost_garden_l1556_155609


namespace NUMINAMATH_CALUDE_num_triangles_equals_closest_integer_l1556_155647

/-- The number of distinct triangles in a regular n-gon -/
def num_triangles (n : ℕ) : ℕ := sorry

/-- The integer closest to n^2/12 -/
def closest_integer (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of distinct triangles in a regular n-gon
    is equal to the integer closest to n^2/12 -/
theorem num_triangles_equals_closest_integer (n : ℕ) (h : n ≥ 3) :
  num_triangles n = closest_integer n := by sorry

end NUMINAMATH_CALUDE_num_triangles_equals_closest_integer_l1556_155647


namespace NUMINAMATH_CALUDE_intersection_A_B_l1556_155608

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x | 2 * x^2 - 9 * x + 9 ≤ 0}

theorem intersection_A_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1556_155608


namespace NUMINAMATH_CALUDE_sqrt_two_minus_one_abs_plus_pi_minus_one_pow_zero_l1556_155601

theorem sqrt_two_minus_one_abs_plus_pi_minus_one_pow_zero (π : ℝ) : 
  |Real.sqrt 2 - 1| + (π - 1)^0 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_one_abs_plus_pi_minus_one_pow_zero_l1556_155601


namespace NUMINAMATH_CALUDE_soccer_lineup_combinations_l1556_155689

def num_goalkeepers : ℕ := 3
def num_defenders : ℕ := 5
def num_midfielders : ℕ := 8
def num_forwards : ℕ := 4

theorem soccer_lineup_combinations : 
  num_goalkeepers * num_defenders * num_midfielders * (num_forwards * (num_forwards - 1)) = 1440 :=
by sorry

end NUMINAMATH_CALUDE_soccer_lineup_combinations_l1556_155689


namespace NUMINAMATH_CALUDE_area_30_60_90_triangle_l1556_155684

/-- The area of a 30-60-90 triangle with hypotenuse 6 is 9√3/2 -/
theorem area_30_60_90_triangle (h : Real) (A : Real) : 
  h = 6 → -- hypotenuse is 6 units
  A = (9 * Real.sqrt 3) / 2 → -- area is 9√3/2 square units
  ∃ (s1 s2 : Real), -- there exist two sides s1 and s2 such that
    s1^2 + s2^2 = h^2 ∧ -- Pythagorean theorem
    s1 = h / 2 ∧ -- shortest side is half the hypotenuse
    s2 = s1 * Real.sqrt 3 ∧ -- longer side is √3 times the shorter side
    A = (1 / 2) * s1 * s2 -- area formula
  := by sorry

end NUMINAMATH_CALUDE_area_30_60_90_triangle_l1556_155684


namespace NUMINAMATH_CALUDE_van_distance_van_distance_proof_l1556_155662

/-- The distance covered by a van under specific conditions -/
theorem van_distance : ℝ :=
  let initial_time : ℝ := 6
  let new_time_factor : ℝ := 3/2
  let new_speed : ℝ := 28
  let distance := new_speed * (new_time_factor * initial_time)
  252

/-- Proof that the van's distance is 252 km -/
theorem van_distance_proof : van_distance = 252 := by
  sorry

end NUMINAMATH_CALUDE_van_distance_van_distance_proof_l1556_155662


namespace NUMINAMATH_CALUDE_chetan_score_percentage_l1556_155686

theorem chetan_score_percentage (max_score : ℕ) (amar_percent : ℚ) (bhavan_percent : ℚ) (average_mark : ℕ) :
  max_score = 900 →
  amar_percent = 64/100 →
  bhavan_percent = 36/100 →
  average_mark = 432 →
  ∃ (chetan_percent : ℚ), 
    (amar_percent + bhavan_percent + chetan_percent) * max_score / 3 = average_mark ∧
    chetan_percent = 44/100 :=
by sorry

end NUMINAMATH_CALUDE_chetan_score_percentage_l1556_155686


namespace NUMINAMATH_CALUDE_activity_popularity_ranking_l1556_155660

/-- Represents the popularity of an activity as a fraction --/
structure ActivityPopularity where
  numerator : ℕ
  denominator : ℕ
  denominator_pos : denominator > 0

/-- The three activities in the festival --/
inductive Activity
  | Dance
  | Painting
  | ClayModeling

/-- Given popularity data for the activities --/
def popularity : Activity → ActivityPopularity
  | Activity.Dance => ⟨3, 8, by norm_num⟩
  | Activity.Painting => ⟨5, 16, by norm_num⟩
  | Activity.ClayModeling => ⟨9, 24, by norm_num⟩

/-- Convert a fraction to a common denominator --/
def toCommonDenominator (ap : ActivityPopularity) (lcd : ℕ) : ℚ :=
  (ap.numerator : ℚ) * (lcd / ap.denominator) / lcd

/-- The least common denominator of all activities' fractions --/
def leastCommonDenominator : ℕ := 48

theorem activity_popularity_ranking :
  let commonDance := toCommonDenominator (popularity Activity.Dance) leastCommonDenominator
  let commonPainting := toCommonDenominator (popularity Activity.Painting) leastCommonDenominator
  let commonClayModeling := toCommonDenominator (popularity Activity.ClayModeling) leastCommonDenominator
  (commonDance = commonClayModeling) ∧ (commonDance > commonPainting) := by
  sorry

#check activity_popularity_ranking

end NUMINAMATH_CALUDE_activity_popularity_ranking_l1556_155660


namespace NUMINAMATH_CALUDE_sum_1_to_15_mod_11_l1556_155683

theorem sum_1_to_15_mod_11 : (List.range 15).sum % 11 = 10 := by sorry

end NUMINAMATH_CALUDE_sum_1_to_15_mod_11_l1556_155683


namespace NUMINAMATH_CALUDE_circumscribed_radius_of_specific_trapezoid_l1556_155674

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  lateral : ℝ

/-- The radius of the circumscribed circle of an isosceles trapezoid -/
def circumscribedRadius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that the radius of the circumscribed circle of the given isosceles trapezoid is 5√2 -/
theorem circumscribed_radius_of_specific_trapezoid :
  let t : IsoscelesTrapezoid := { base1 := 2, base2 := 14, lateral := 10 }
  circumscribedRadius t = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_radius_of_specific_trapezoid_l1556_155674


namespace NUMINAMATH_CALUDE_rabbit_carrots_l1556_155607

theorem rabbit_carrots (rabbit_per_burrow deer_per_burrow : ℕ)
  (rabbit_burrows deer_burrows : ℕ) :
  rabbit_per_burrow = 4 →
  deer_per_burrow = 6 →
  rabbit_per_burrow * rabbit_burrows = deer_per_burrow * deer_burrows →
  rabbit_burrows = deer_burrows + 3 →
  rabbit_per_burrow * rabbit_burrows = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_rabbit_carrots_l1556_155607


namespace NUMINAMATH_CALUDE_circle_radius_l1556_155666

theorem circle_radius (x y : ℝ) :
  (16 * x^2 + 32 * x + 16 * y^2 - 48 * y + 76 = 0) →
  ∃ (center_x center_y : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l1556_155666


namespace NUMINAMATH_CALUDE_fraction_division_equality_l1556_155645

theorem fraction_division_equality : (3 : ℚ) / 7 / (5 / 2) = 6 / 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_equality_l1556_155645


namespace NUMINAMATH_CALUDE_sin_is_2_type_function_x_plus_cos_is_2_type_function_l1556_155680

-- Define what it means for a function to be a t-type function
def is_t_type_function (f : ℝ → ℝ) (t : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (deriv f x₁) + (deriv f x₂) = t

-- State the theorem for sin x
theorem sin_is_2_type_function :
  is_t_type_function Real.sin 2 :=
sorry

-- State the theorem for x + cos x
theorem x_plus_cos_is_2_type_function :
  is_t_type_function (fun x => x + Real.cos x) 2 :=
sorry

end NUMINAMATH_CALUDE_sin_is_2_type_function_x_plus_cos_is_2_type_function_l1556_155680


namespace NUMINAMATH_CALUDE_sum_of_digits_power_two_l1556_155640

/-- Sum of digits function -/
def s (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem sum_of_digits_power_two : 
  (∀ n : ℕ, (n - s n) % 9 = 0) → 
  (2^2009 < 10^672) → 
  s (s (s (2^2009))) = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_two_l1556_155640


namespace NUMINAMATH_CALUDE_sum_of_differences_mod_1000_l1556_155641

def S : Finset ℕ := Finset.range 11

def pairDifference (i j : ℕ) : ℕ := 
  if i < j then 2^j - 2^i else 2^i - 2^j

def N : ℕ := Finset.sum (S.product S) (fun (p : ℕ × ℕ) => pairDifference p.1 p.2)

theorem sum_of_differences_mod_1000 : N % 1000 = 304 := by sorry

end NUMINAMATH_CALUDE_sum_of_differences_mod_1000_l1556_155641


namespace NUMINAMATH_CALUDE_dvd_pack_cost_l1556_155678

/-- Given that 6 packs of DVDs can be bought with 120 dollars, 
    prove that each pack costs 20 dollars. -/
theorem dvd_pack_cost (total_cost : ℕ) (num_packs : ℕ) (cost_per_pack : ℕ) 
    (h1 : total_cost = 120) 
    (h2 : num_packs = 6) 
    (h3 : total_cost = num_packs * cost_per_pack) : 
  cost_per_pack = 20 := by
  sorry

end NUMINAMATH_CALUDE_dvd_pack_cost_l1556_155678


namespace NUMINAMATH_CALUDE_santa_gift_combinations_l1556_155693

theorem santa_gift_combinations (n : ℤ) : 30 ∣ (n^5 - n) := by
  sorry

end NUMINAMATH_CALUDE_santa_gift_combinations_l1556_155693


namespace NUMINAMATH_CALUDE_existential_vs_universal_quantifier_l1556_155653

theorem existential_vs_universal_quantifier :
  ¬(∀ (x₀ : ℝ), x₀^2 > 3 ↔ ∃ (x₀ : ℝ), x₀^2 > 3) :=
by sorry

end NUMINAMATH_CALUDE_existential_vs_universal_quantifier_l1556_155653


namespace NUMINAMATH_CALUDE_factor_expression_l1556_155604

theorem factor_expression (y z : ℝ) : 3 * y^2 - 75 * z^2 = 3 * (y + 5 * z) * (y - 5 * z) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1556_155604


namespace NUMINAMATH_CALUDE_xiao_ming_final_score_l1556_155648

/-- Calculates the final score of a speech contest given individual scores and weights -/
def final_score (speech_image : ℝ) (content : ℝ) (effectiveness : ℝ) 
  (weight_image : ℝ) (weight_content : ℝ) (weight_effectiveness : ℝ) : ℝ :=
  speech_image * weight_image + content * weight_content + effectiveness * weight_effectiveness

/-- Xiao Ming's speech contest scores and weights -/
def xiao_ming_scores : ℝ × ℝ × ℝ := (9, 8, 8)
def xiao_ming_weights : ℝ × ℝ × ℝ := (0.3, 0.4, 0.3)

theorem xiao_ming_final_score :
  final_score xiao_ming_scores.1 xiao_ming_scores.2.1 xiao_ming_scores.2.2
              xiao_ming_weights.1 xiao_ming_weights.2.1 xiao_ming_weights.2.2 = 8.3 := by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_final_score_l1556_155648


namespace NUMINAMATH_CALUDE_equation_equivalence_l1556_155611

theorem equation_equivalence (x y : ℝ) : (2 * x - y = 3) ↔ (y = 2 * x - 3) := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1556_155611


namespace NUMINAMATH_CALUDE_problem_statement_l1556_155668

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) - 2 * (Real.sin (ω * x / 2))^2

theorem problem_statement 
  (ω : ℝ) 
  (h_ω_pos : ω > 0)
  (h_period : ∀ x, f ω (x + 3 * Real.pi) = f ω x)
  (a b c A B C : ℝ)
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi)
  (h_b : b = 2)
  (h_fA : f ω A = Real.sqrt 3 - 1)
  (h_sides : Real.sqrt 3 * a = 2 * b * Real.sin A) :
  (∃ x ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, ∀ y ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, f ω y ≥ f ω x) ∧ 
  (∃ x ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, ∀ y ∈ Set.Icc (-3 * Real.pi / 4) Real.pi, f ω y ≤ f ω x) ∧
  (1/2 * a * b * Real.sin C = (3 + Real.sqrt 3) / 3) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1556_155668


namespace NUMINAMATH_CALUDE_brick_length_calculation_l1556_155652

theorem brick_length_calculation (courtyard_length : Real) (courtyard_width : Real)
  (brick_width : Real) (total_bricks : Nat) :
  courtyard_length = 18 ∧ 
  courtyard_width = 12 ∧
  brick_width = 0.06 ∧
  total_bricks = 30000 →
  ∃ brick_length : Real,
    brick_length = 0.12 ∧
    courtyard_length * courtyard_width * 10000 = total_bricks * brick_length * brick_width :=
by sorry

end NUMINAMATH_CALUDE_brick_length_calculation_l1556_155652


namespace NUMINAMATH_CALUDE_confectioner_pastry_count_l1556_155618

theorem confectioner_pastry_count :
  ∀ (P : ℕ),
  (P / 28 : ℚ) - (P / 49 : ℚ) = 6 →
  P = 378 :=
by
  sorry

end NUMINAMATH_CALUDE_confectioner_pastry_count_l1556_155618


namespace NUMINAMATH_CALUDE_quadratic_sum_l1556_155603

/-- Given a quadratic x^2 - 20x + 36 that can be written as (x + b)^2 + c,
    prove that b + c = -74 -/
theorem quadratic_sum (b c : ℝ) : 
  (∀ x, x^2 - 20*x + 36 = (x + b)^2 + c) → b + c = -74 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l1556_155603


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1556_155657

theorem right_triangle_third_side (a b c : ℝ) : 
  a = 8 ∧ b = 15 ∧ a > 0 ∧ b > 0 ∧ c > 0 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c = 17 ∨ c = Real.sqrt 161 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1556_155657


namespace NUMINAMATH_CALUDE_m_derivative_l1556_155632

noncomputable def m (x : ℝ) : ℝ := (2^x) / (1 + x)

theorem m_derivative (x : ℝ) : 
  deriv m x = (2^x * (1 + x) * Real.log 2 - 2^x) / (1 + x)^2 :=
by sorry

end NUMINAMATH_CALUDE_m_derivative_l1556_155632


namespace NUMINAMATH_CALUDE_problem_solution_l1556_155671

theorem problem_solution : 101^4 - 4 * 101^3 + 6 * 101^2 - 4 * 101 + 1 = 100000000 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1556_155671


namespace NUMINAMATH_CALUDE_line_circle_intersection_count_l1556_155617

/-- The number of intersection points between a line and a circle -/
theorem line_circle_intersection_count (k : ℝ) : 
  ∃ (p q : ℝ × ℝ), p ≠ q ∧ 
  (∀ (x y : ℝ), (k * x - y - k = 0 ∧ x^2 + y^2 = 2) ↔ (x, y) = p ∨ (x, y) = q) :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_count_l1556_155617


namespace NUMINAMATH_CALUDE_existence_of_A_l1556_155688

/-- An increasing sequence of positive integers -/
def IncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The growth condition for the sequence -/
def GrowthCondition (a : ℕ → ℕ) (M : ℝ) : Prop :=
  ∀ n : ℕ, 0 < a (n + 1) - a n ∧ (a (n + 1) - a n : ℝ) < M * (a n : ℝ) ^ (5/8)

/-- The main theorem -/
theorem existence_of_A (a : ℕ → ℕ) (M : ℝ) 
    (h_inc : IncreasingSequence a) 
    (h_growth : GrowthCondition a M) :
    ∃ A : ℝ, ∀ k : ℕ, ∃ n : ℕ, ⌊A ^ (3^k)⌋ = a n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_A_l1556_155688


namespace NUMINAMATH_CALUDE_problem_proof_l1556_155643

theorem problem_proof : (-24) * (1/3 - 5/6 + 3/8) = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_proof_l1556_155643


namespace NUMINAMATH_CALUDE_tangent_parallel_to_4x_l1556_155694

/-- The curve function f(x) = x^3 + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_parallel_to_4x :
  ∃ x : ℝ, f x = 0 ∧ f' x = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_to_4x_l1556_155694


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l1556_155673

/-- A circle with a given center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle
  tangentPoint1 : ℝ × ℝ
  tangentPoint2 : ℝ × ℝ

/-- The y-intercept of a line tangent to two specific circles -/
def yIntercept (line : TangentLine) : ℝ :=
  sorry

/-- The main theorem stating the y-intercept of the tangent line -/
theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 0), radius := 3 }
  let c2 : Circle := { center := (6, 0), radius := 1 }
  ∀ (line : TangentLine),
    line.circle1 = c1 →
    line.circle2 = c2 →
    line.tangentPoint1.1 > 3 →
    line.tangentPoint1.2 > 0 →
    line.tangentPoint2.1 > 6 →
    line.tangentPoint2.2 > 0 →
    yIntercept line = 6 * Real.sqrt 2 :=
  by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l1556_155673


namespace NUMINAMATH_CALUDE_first_group_size_correct_l1556_155605

/-- The number of beavers in the first group -/
def first_group_size : ℕ := 20

/-- The time taken by the first group to build the dam (in hours) -/
def first_group_time : ℕ := 3

/-- The number of beavers in the second group -/
def second_group_size : ℕ := 12

/-- The time taken by the second group to build the dam (in hours) -/
def second_group_time : ℕ := 5

/-- Theorem stating that the first group size is correct -/
theorem first_group_size_correct : 
  first_group_size * first_group_time = second_group_size * second_group_time :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_correct_l1556_155605


namespace NUMINAMATH_CALUDE_simple_interest_rate_percent_l1556_155628

/-- Simple interest calculation -/
theorem simple_interest_rate_percent 
  (principal : ℝ) 
  (interest : ℝ) 
  (time : ℝ) 
  (h1 : principal = 720)
  (h2 : interest = 180)
  (h3 : time = 4)
  : (interest * 100) / (principal * time) = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_percent_l1556_155628


namespace NUMINAMATH_CALUDE_tan_alpha_negative_three_l1556_155644

theorem tan_alpha_negative_three (α : Real) (h : Real.tan α = -3) :
  (Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α) = 3 ∧
  Real.sin α ^ 2 + Real.sin α * Real.cos α + 2 = 13/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_negative_three_l1556_155644


namespace NUMINAMATH_CALUDE_equation_solution_equivalence_l1556_155650

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(-1, 2, 5), (-1, -5, -2), (-2, 1, 5), (-2, -5, -1), (5, 1, -2), (5, 2, -1)}

def satisfies_equations (x y z : ℝ) : Prop :=
  x - y + z = 2 ∧ x^2 + y^2 + z^2 = 30 ∧ x^3 - y^3 + z^3 = 116

theorem equation_solution_equivalence :
  ∀ x y z : ℝ, satisfies_equations x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_equivalence_l1556_155650


namespace NUMINAMATH_CALUDE_digits_of_8_pow_20_times_5_pow_18_l1556_155614

/-- The number of digits in a positive integer n in base b -/
def num_digits (n : ℕ) (b : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log b n + 1

/-- Theorem: The number of digits in 8^20 * 5^18 in base 10 is 31 -/
theorem digits_of_8_pow_20_times_5_pow_18 :
  num_digits (8^20 * 5^18) 10 = 31 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_8_pow_20_times_5_pow_18_l1556_155614


namespace NUMINAMATH_CALUDE_parallel_planes_lines_l1556_155620

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_parallel_line : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_lines 
  (α β : Plane) (m n : Line) :
  parallel α β →
  line_parallel_plane m α →
  line_parallel_line n m →
  ¬ line_in_plane n β →
  line_parallel_plane n β :=
by sorry

end NUMINAMATH_CALUDE_parallel_planes_lines_l1556_155620


namespace NUMINAMATH_CALUDE_equation_solution_l1556_155667

theorem equation_solution : ∃! x : ℝ, (3 - x) / (x - 4) + 1 / (4 - x) = 1 ∧ x ≠ 4 :=
  by sorry

end NUMINAMATH_CALUDE_equation_solution_l1556_155667


namespace NUMINAMATH_CALUDE_math_test_paper_probability_l1556_155646

theorem math_test_paper_probability :
  let total_papers : ℕ := 12
  let math_papers : ℕ := 4
  let probability := math_papers / total_papers
  probability = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_math_test_paper_probability_l1556_155646


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1556_155670

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 9

-- Define the condition for no real roots
def has_no_real_roots (m : ℝ) : Prop := ∀ x : ℝ, f m x ≠ 0

-- Define the given interval
def p (m : ℝ) : Prop := -6 ≤ m ∧ m ≤ 6

-- Theorem statement
theorem sufficient_not_necessary :
  (∀ m : ℝ, p m → has_no_real_roots m) ∧
  ¬(∀ m : ℝ, has_no_real_roots m → p m) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1556_155670


namespace NUMINAMATH_CALUDE_rectangular_equation_of_C_chord_length_l1556_155633

-- Define the polar curve C
def polar_curve (ρ θ : ℝ) : Prop := ρ * Real.sin θ = 8 * Real.cos θ

-- Define the line l
def line_l (t x y : ℝ) : Prop := x = 2 + t ∧ y = Real.sqrt 3 * t

-- Theorem for the rectangular equation of curve C
theorem rectangular_equation_of_C (x y : ℝ) :
  (∃ ρ θ, polar_curve ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  y^2 = 8*x :=
sorry

-- Theorem for the chord length |AB|
theorem chord_length (A B : ℝ × ℝ) :
  (∃ t₁, line_l t₁ A.1 A.2 ∧ A.2^2 = 8*A.1) →
  (∃ t₂, line_l t₂ B.1 B.2 ∧ B.2^2 = 8*B.1) →
  A ≠ B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 32/3 :=
sorry

end NUMINAMATH_CALUDE_rectangular_equation_of_C_chord_length_l1556_155633


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1556_155606

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (2 * α) = -Real.sin α) : 
  Real.tan α = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1556_155606


namespace NUMINAMATH_CALUDE_cone_properties_l1556_155677

/-- Properties of a right circular cone -/
theorem cone_properties (V h r l : ℝ) (hV : V = 16 * Real.pi) (hh : h = 6) 
  (hVol : (1/3) * Real.pi * r^2 * h = V) 
  (hSlant : l^2 = r^2 + h^2) : 
  2 * Real.pi * r = 4 * Real.sqrt 2 * Real.pi ∧ 
  Real.pi * r * l = 4 * Real.sqrt 22 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_cone_properties_l1556_155677


namespace NUMINAMATH_CALUDE_square_land_multiple_l1556_155613

theorem square_land_multiple (a p k : ℝ) : 
  a > 0 → 
  p > 0 → 
  p = 36 → 
  a = (p / 4) ^ 2 → 
  5 * a = k * p + 45 → 
  k = 10 := by
sorry

end NUMINAMATH_CALUDE_square_land_multiple_l1556_155613


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1556_155679

theorem trigonometric_identity (A B C : ℝ) : 
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 2 + 2 * Real.cos A * Real.cos B * Real.cos C := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1556_155679


namespace NUMINAMATH_CALUDE_min_tan_product_acute_triangle_l1556_155621

theorem min_tan_product_acute_triangle (A B C : ℝ) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_sin : Real.sin A = 3 * Real.sin B * Real.sin C) : 
  (∀ A' B' C', 0 < A' ∧ 0 < B' ∧ 0 < C' ∧ A' + B' + C' = π → 
    Real.sin A' = 3 * Real.sin B' * Real.sin C' → 
    12 ≤ Real.tan A' * Real.tan B' * Real.tan C') ∧
  (∃ A₀ B₀ C₀, 0 < A₀ ∧ 0 < B₀ ∧ 0 < C₀ ∧ A₀ + B₀ + C₀ = π ∧
    Real.sin A₀ = 3 * Real.sin B₀ * Real.sin C₀ ∧
    Real.tan A₀ * Real.tan B₀ * Real.tan C₀ = 12) := by
  sorry

end NUMINAMATH_CALUDE_min_tan_product_acute_triangle_l1556_155621


namespace NUMINAMATH_CALUDE_two_roots_iff_a_eq_twenty_l1556_155658

/-- The quadratic equation in x parametrized by a -/
def f (a : ℝ) (x : ℝ) : ℝ := a^2 * (x - 2) + a * (39 - 20*x) + 20

/-- The condition for at least two distinct roots -/
def has_at_least_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0

/-- The main theorem -/
theorem two_roots_iff_a_eq_twenty :
  ∀ a : ℝ, has_at_least_two_distinct_roots a ↔ a = 20 := by sorry

end NUMINAMATH_CALUDE_two_roots_iff_a_eq_twenty_l1556_155658


namespace NUMINAMATH_CALUDE_line_properties_l1556_155675

-- Define the lines l₁ and l₂
def l₁ (a b : ℝ) (x y : ℝ) : Prop := a * x - b * y + 4 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a - 1) * x + y + b = 0

-- Define perpendicularity of lines
def perpendicular (a b : ℝ) : Prop := a * (a - 1) - b = 0

-- Define parallel lines
def parallel (a b : ℝ) : Prop := a / b = 1 - a

-- Define point M
def point_M (a b : ℝ) : Prop := l₁ a b (-3) (-1)

-- Define equal distance from origin to both lines
def equal_distance (b : ℝ) : Prop := 4 / b = b

theorem line_properties (a b : ℝ) :
  (perpendicular a b ∧ point_M a b → a = 2 ∧ b = 2) ∧
  (parallel a b ∧ equal_distance b → (a = 2 ∧ b = -2) ∨ (a = 2/3 ∧ b = 2)) :=
sorry

end NUMINAMATH_CALUDE_line_properties_l1556_155675


namespace NUMINAMATH_CALUDE_inequality_proof_l1556_155625

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b + b / c + c / a)^2 ≥ (3/2) * ((a + b) / c + (b + c) / a + (c + a) / b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1556_155625


namespace NUMINAMATH_CALUDE_twelve_tone_equal_temperament_l1556_155622

theorem twelve_tone_equal_temperament (a : ℕ → ℝ) :
  (∀ n : ℕ, 1 ≤ n → n ≤ 13 → a n > 0) →
  (∀ n : ℕ, 1 < n → n ≤ 13 → a n / a (n-1) = a 2 / a 1) →
  a 1 = 1 →
  a 13 = 2 →
  a 5 = 2^(1/3) :=
sorry

end NUMINAMATH_CALUDE_twelve_tone_equal_temperament_l1556_155622


namespace NUMINAMATH_CALUDE_binomial_16_choose_5_l1556_155639

theorem binomial_16_choose_5 : Nat.choose 16 5 = 4368 := by
  sorry

end NUMINAMATH_CALUDE_binomial_16_choose_5_l1556_155639


namespace NUMINAMATH_CALUDE_total_bathing_suits_l1556_155669

theorem total_bathing_suits (men_suits women_suits : ℕ) 
  (h1 : men_suits = 14797) 
  (h2 : women_suits = 4969) : 
  men_suits + women_suits = 19766 := by
  sorry

end NUMINAMATH_CALUDE_total_bathing_suits_l1556_155669


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1556_155656

-- Define the condition for an ellipse with foci on the x-axis
def is_ellipse_x_foci (m : ℝ) : Prop := m^2 - 1 > 3

-- Define the given condition
def given_condition (m : ℝ) : Prop := m^2 > 5

-- Theorem statement
theorem sufficient_not_necessary :
  (∀ m : ℝ, given_condition m → is_ellipse_x_foci m) ∧
  ¬(∀ m : ℝ, is_ellipse_x_foci m → given_condition m) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1556_155656


namespace NUMINAMATH_CALUDE_math_is_90_average_l1556_155690

/-- Represents the scores in three subjects -/
structure Scores where
  physics : ℝ
  chemistry : ℝ
  mathematics : ℝ

/-- Represents the conditions given in the problem -/
def satisfiesConditions (s : Scores) : Prop :=
  s.physics = 80 ∧
  (s.physics + s.chemistry + s.mathematics) / 3 = 80 ∧
  (s.physics + s.chemistry) / 2 = 70 ∧
  ∃ x, (s.physics + x) / 2 = 90 ∧ (x = s.chemistry ∨ x = s.mathematics)

/-- Theorem stating that mathematics is the subject averaging 90 with physics -/
theorem math_is_90_average (s : Scores) (h : satisfiesConditions s) :
  (s.physics + s.mathematics) / 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_math_is_90_average_l1556_155690


namespace NUMINAMATH_CALUDE_hyperbola_condition_l1556_155681

/-- Defines a hyperbola in terms of its equation -/
def is_hyperbola (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), m * x^2 + n * y^2 = 1 ∧ 
  ∀ (a b : ℝ), a^2 / (1/m) - b^2 / (1/n) = 1 ∨ a^2 / (1/n) - b^2 / (1/m) = 1

/-- Theorem stating that mn < 0 is a necessary and sufficient condition for mx^2 + ny^2 = 1 to represent a hyperbola -/
theorem hyperbola_condition (m n : ℝ) :
  is_hyperbola m n ↔ m * n < 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l1556_155681


namespace NUMINAMATH_CALUDE_fraction_decomposition_l1556_155600

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 4/3) :
  (7 * x - 13) / (3 * x^2 + 2 * x - 8) = 27 / (10 * (x + 2)) - 11 / (10 * (3 * x - 4)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l1556_155600


namespace NUMINAMATH_CALUDE_hypercube_diagonal_count_l1556_155627

/-- A hypercube is a 4-dimensional cube -/
structure Hypercube where
  vertices : Finset (Fin 16)
  edges : Finset (Fin 16 × Fin 16)
  vertex_count : vertices.card = 16
  edge_count : edges.card = 32

/-- A diagonal in a hypercube is a segment joining two vertices not joined by an edge -/
def Diagonal (h : Hypercube) (v w : Fin 16) : Prop :=
  v ∈ h.vertices ∧ w ∈ h.vertices ∧ v ≠ w ∧ (v, w) ∉ h.edges

/-- The number of diagonals in a hypercube -/
def DiagonalCount (h : Hypercube) : ℕ :=
  (h.vertices.card.choose 2) - h.edges.card

/-- Theorem: A hypercube has 408 diagonals -/
theorem hypercube_diagonal_count (h : Hypercube) : DiagonalCount h = 408 := by
  sorry

end NUMINAMATH_CALUDE_hypercube_diagonal_count_l1556_155627


namespace NUMINAMATH_CALUDE_recipe_total_cups_l1556_155664

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the amount of flour -/
def totalIngredients (ratio : RecipeRatio) (flourCups : ℕ) : ℕ :=
  let partSize := flourCups / ratio.flour
  (ratio.butter + ratio.flour + ratio.sugar) * partSize

/-- Theorem: Given a recipe with ratio 2:5:3 and 10 cups of flour, the total ingredients is 20 cups -/
theorem recipe_total_cups (ratio : RecipeRatio) (h1 : ratio.butter = 2) (h2 : ratio.flour = 5) (h3 : ratio.sugar = 3) :
  totalIngredients ratio 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l1556_155664


namespace NUMINAMATH_CALUDE_triangle_inequality_l1556_155636

/-- Given a triangle with semi-perimeter s, circumradius R, and inradius r,
    prove the inequality relating these quantities. -/
theorem triangle_inequality (s R r : ℝ) (hs : s > 0) (hR : R > 0) (hr : r > 0) :
  2 * Real.sqrt (r * (r + 4 * R)) < 2 * s ∧ 
  2 * s ≤ Real.sqrt (4 * (r + 2 * R)^2 + 2 * R^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1556_155636


namespace NUMINAMATH_CALUDE_five_balls_three_boxes_l1556_155638

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ways_to_distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 66 ways to put 5 distinguishable balls into 3 indistinguishable boxes -/
theorem five_balls_three_boxes : ways_to_distribute 5 3 = 66 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_three_boxes_l1556_155638


namespace NUMINAMATH_CALUDE_triangle_equilateral_l1556_155612

theorem triangle_equilateral (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_equation : 2 * (a * b^2 + b * c^2 + c * a^2) = a^2 * b + b^2 * c + c^2 * a + 3 * a * b * c) : 
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l1556_155612


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l1556_155634

theorem pure_imaginary_fraction (a : ℝ) : 
  let z : ℂ := (a - Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = -1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l1556_155634


namespace NUMINAMATH_CALUDE_abc_inequality_l1556_155659

theorem abc_inequality (a b c : Real) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (eq_a : Real.exp a = 9 * a * Real.log 11)
  (eq_b : Real.exp b = 10 * b * Real.log 10)
  (eq_c : Real.exp c = 11 * c * Real.log 9) :
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l1556_155659


namespace NUMINAMATH_CALUDE_handshakes_four_and_n_l1556_155699

/-- Calculates the number of handshakes for n people -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

theorem handshakes_four_and_n :
  (handshakes 4 = 6) ∧
  (∀ n : ℕ, handshakes n = n * (n - 1) / 2) :=
by sorry

#check handshakes_four_and_n

end NUMINAMATH_CALUDE_handshakes_four_and_n_l1556_155699


namespace NUMINAMATH_CALUDE_sum_series_result_l1556_155692

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def sum_series (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => (↑(double_factorial (2*i+1)) / ↑(double_factorial (2*i+2)) + 1 / 2^(i+1)))

theorem sum_series_result : 
  ∃ (a b : ℕ), b % 2 = 1 ∧ 
    (∃ (num : ℕ), sum_series 2023 = num / (2^a * b : ℚ)) ∧
    a * b / 10 = 4039 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_series_result_l1556_155692


namespace NUMINAMATH_CALUDE_log_base_2_derivative_l1556_155602

open Real

theorem log_base_2_derivative (x : ℝ) (h : x > 0) : 
  deriv (λ x => log x / log 2) x = 1 / (x * log 2) := by
  sorry

end NUMINAMATH_CALUDE_log_base_2_derivative_l1556_155602


namespace NUMINAMATH_CALUDE_complex_power_difference_l1556_155635

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) :
  (1 + 2*i)^24 - (1 - 2*i)^24 = 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1556_155635


namespace NUMINAMATH_CALUDE_circumradius_inradius_ratio_rational_l1556_155630

/-- Given a triangle with rational side lengths, prove that the ratio of its circumradius to inradius is rational. -/
theorem circumradius_inradius_ratio_rational 
  (a b c : ℚ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let p : ℚ := (a + b + c) / 2
  ∃ (q : ℚ), q = a * b * c / (4 * (p - a) * (p - b) * (p - c)) :=
sorry

end NUMINAMATH_CALUDE_circumradius_inradius_ratio_rational_l1556_155630


namespace NUMINAMATH_CALUDE_prime_sum_product_l1556_155665

theorem prime_sum_product : ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p + q = 95 ∧ p * q = 178 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_product_l1556_155665
