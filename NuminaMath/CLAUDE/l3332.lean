import Mathlib

namespace NUMINAMATH_CALUDE_cubes_not_touching_foil_l3332_333223

/-- Represents the dimensions of a rectangular prism --/
structure PrismDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of cubes in a rectangular prism --/
def cubeCount (d : PrismDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Theorem: The number of cubes not touching tin foil in the described prism is 128 --/
theorem cubes_not_touching_foil : ∃ (inner outer : PrismDimensions),
  -- The width of the foil-covered prism is 10 inches
  outer.width = 10 ∧
  -- The width of the inner figure is twice its length and height
  inner.width = 2 * inner.length ∧
  inner.width = 2 * inner.height ∧
  -- There is a 1-inch layer of cubes touching the foil on all sides
  outer.length = inner.length + 2 ∧
  outer.width = inner.width + 2 ∧
  outer.height = inner.height + 2 ∧
  -- The number of cubes not touching any tin foil is 128
  cubeCount inner = 128 := by
  sorry


end NUMINAMATH_CALUDE_cubes_not_touching_foil_l3332_333223


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l3332_333218

theorem consecutive_integers_square_sum : 
  ∀ a : ℤ, a > 0 → 
  ((a - 1) * a * (a + 1) = 12 * (3 * a)) → 
  ((a - 1)^2 + a^2 + (a + 1)^2 = 77) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l3332_333218


namespace NUMINAMATH_CALUDE_rabbits_total_distance_l3332_333231

/-- The total distance hopped by two rabbits in a given time -/
def total_distance (white_speed brown_speed time : ℕ) : ℕ :=
  (white_speed * time) + (brown_speed * time)

/-- Theorem: The total distance hopped by the white and brown rabbits in 5 minutes is 135 meters -/
theorem rabbits_total_distance :
  total_distance 15 12 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_total_distance_l3332_333231


namespace NUMINAMATH_CALUDE_negation_of_even_sum_false_l3332_333220

theorem negation_of_even_sum_false : 
  ¬(∀ a b : ℤ, (¬(Even a ∧ Even b) → ¬Even (a + b))) := by sorry

end NUMINAMATH_CALUDE_negation_of_even_sum_false_l3332_333220


namespace NUMINAMATH_CALUDE_work_completion_l3332_333266

/-- Represents the total amount of work in man-days -/
def total_work : ℕ := 10 * 80

/-- The number of days it takes for the second group to complete the work -/
def days_second_group : ℕ := 40

/-- Calculates the number of men needed to complete the work in a given number of days -/
def men_needed (days : ℕ) : ℕ := total_work / days

theorem work_completion :
  men_needed days_second_group = 20 :=
sorry

end NUMINAMATH_CALUDE_work_completion_l3332_333266


namespace NUMINAMATH_CALUDE_total_sibling_age_l3332_333211

/-- Represents the ages of the siblings -/
structure SiblingAges where
  susan : ℝ
  arthur : ℝ
  tom : ℝ
  bob : ℝ
  emily : ℝ
  david : ℝ
  youngest : ℝ

/-- Theorem stating the total age of the siblings -/
theorem total_sibling_age (ages : SiblingAges) : 
  ages.arthur = ages.susan + 2 →
  ages.tom = ages.bob - 3 →
  ages.emily = ages.susan / 2 →
  ages.david = (ages.arthur + ages.tom + ages.emily) / 3 →
  ages.susan - ages.tom = 2 * (ages.emily - ages.david) →
  ages.bob = 11 →
  ages.susan = 15 →
  ages.emily = ages.youngest + 2.5 →
  ages.susan + ages.arthur + ages.tom + ages.bob + ages.emily + ages.david + ages.youngest = 74.5 := by
  sorry


end NUMINAMATH_CALUDE_total_sibling_age_l3332_333211


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l3332_333239

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 → -- angles are complementary
  a = 4 * b → -- ratio of measures is 4:1
  abs (a - b) = 54 := by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l3332_333239


namespace NUMINAMATH_CALUDE_polynomial_identity_l3332_333253

theorem polynomial_identity : ∀ x : ℝ, 
  (x^2 + 3*x + 2) * (x + 3) = (x + 1) * (x^2 + 5*x + 6) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3332_333253


namespace NUMINAMATH_CALUDE_min_colors_is_23_l3332_333229

/-- A coloring scheme for boxes of balls -/
structure ColoringScheme where
  n : ℕ  -- number of colors
  boxes : Fin 8 → Fin 6 → Fin n  -- coloring function

/-- Predicate to check if a coloring scheme is valid -/
def is_valid_coloring (c : ColoringScheme) : Prop :=
  -- No two balls in the same box have the same color
  (∀ i : Fin 8, ∀ j k : Fin 6, j ≠ k → c.boxes i j ≠ c.boxes i k) ∧
  -- No two colors occur together in more than one box
  (∀ i j : Fin 8, i ≠ j → ∀ c1 c2 : Fin c.n, c1 ≠ c2 →
    (∃ k : Fin 6, c.boxes i k = c1 ∧ ∃ l : Fin 6, c.boxes i l = c2) →
    ¬(∃ m : Fin 6, c.boxes j m = c1 ∧ ∃ n : Fin 6, c.boxes j n = c2))

/-- The main theorem: the minimum number of colors is 23 -/
theorem min_colors_is_23 :
  (∃ c : ColoringScheme, c.n = 23 ∧ is_valid_coloring c) ∧
  (∀ c : ColoringScheme, c.n < 23 → ¬is_valid_coloring c) := by
  sorry

end NUMINAMATH_CALUDE_min_colors_is_23_l3332_333229


namespace NUMINAMATH_CALUDE_nephews_count_l3332_333224

/-- The number of nephews Alden and Vihaan have together -/
def total_nephews (alden_past : ℕ) (alden_ratio : ℕ) (vihaan_diff : ℕ) : ℕ :=
  let alden_current := alden_past * alden_ratio
  let vihaan_current := alden_current + vihaan_diff
  alden_current + vihaan_current

/-- Proof that Alden and Vihaan have 600 nephews together -/
theorem nephews_count : total_nephews 80 3 120 = 600 := by
  sorry

end NUMINAMATH_CALUDE_nephews_count_l3332_333224


namespace NUMINAMATH_CALUDE_power_sum_l3332_333268

theorem power_sum (a m n : ℝ) (h1 : a^m = 4) (h2 : a^n = 8) : a^(m+n) = 32 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_l3332_333268


namespace NUMINAMATH_CALUDE_sqrt_two_plus_sqrt_l3332_333277

theorem sqrt_two_plus_sqrt : ∃ y : ℝ, y = Real.sqrt (2 + y) → y = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_two_plus_sqrt_l3332_333277


namespace NUMINAMATH_CALUDE_total_flowers_ratio_l3332_333225

/-- Represents the number of pots in the garden -/
def num_pots : ℕ := 350

/-- Represents the ratio of flowers to total items in a pot -/
def flower_ratio : ℚ := 3 / 5

/-- Represents the number of flowers in a single pot -/
def flowers_per_pot (total_items : ℕ) : ℚ := flower_ratio * total_items

theorem total_flowers_ratio (total_items_per_pot : ℕ) :
  (num_pots : ℚ) * flowers_per_pot total_items_per_pot = 
  flower_ratio * ((num_pots : ℚ) * total_items_per_pot) := by sorry

end NUMINAMATH_CALUDE_total_flowers_ratio_l3332_333225


namespace NUMINAMATH_CALUDE_midnight_temperature_l3332_333213

/-- Given the temperature changes throughout the day in a certain city, 
    prove that the temperature at midnight is 24°C. -/
theorem midnight_temperature 
  (morning_temp : ℝ)
  (afternoon_increase : ℝ)
  (midnight_decrease : ℝ)
  (h1 : morning_temp = 30)
  (h2 : afternoon_increase = 1)
  (h3 : midnight_decrease = 7) :
  morning_temp + afternoon_increase - midnight_decrease = 24 :=
by sorry

end NUMINAMATH_CALUDE_midnight_temperature_l3332_333213


namespace NUMINAMATH_CALUDE_banana_count_l3332_333272

/-- The number of bananas in the fruit shop. -/
def bananas : ℕ := 30

/-- The number of apples in the fruit shop. -/
def apples : ℕ := 4 * bananas

/-- The number of persimmons in the fruit shop. -/
def persimmons : ℕ := 3 * bananas

/-- Theorem stating that the number of bananas is 30, given the conditions. -/
theorem banana_count : bananas = 30 := by
  have h1 : apples + persimmons = 210 := by sorry
  sorry

end NUMINAMATH_CALUDE_banana_count_l3332_333272


namespace NUMINAMATH_CALUDE_find_k_l3332_333248

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 5 * x + 6
def g (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 - k * x + 1

-- State the theorem
theorem find_k : ∃ k : ℝ, f 5 - g k 5 = 30 ∧ k = -10 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l3332_333248


namespace NUMINAMATH_CALUDE_radius_is_five_l3332_333282

/-- A configuration of tangent lines to a circle -/
structure TangentConfiguration where
  -- The radius of the circle
  r : ℝ
  -- The length of tangent line AB
  ab : ℝ
  -- The length of tangent line CD
  cd : ℝ
  -- The length of tangent line EF
  ef : ℝ
  -- AB and CD are parallel
  parallel_ab_cd : True
  -- A, C, and D are points of tangency
  tangency_points : True
  -- EF intersects AB and CD
  ef_intersects : True
  -- Tangency point for EF falls between AB and CD
  ef_tangency_between : True
  -- Given lengths
  ab_length : ab = 7
  cd_length : cd = 12
  ef_length : ef = 25

/-- The theorem stating that the radius is 5 given the configuration -/
theorem radius_is_five (config : TangentConfiguration) : config.r = 5 := by
  sorry

end NUMINAMATH_CALUDE_radius_is_five_l3332_333282


namespace NUMINAMATH_CALUDE_find_g_of_x_l3332_333236

theorem find_g_of_x (x : ℝ) (g : ℝ → ℝ) : 
  (2 * x^5 + 4 * x^3 - 3 * x + 5 + g x = 3 * x^4 + 7 * x^2 - 2 * x - 4) → 
  (g x = -2 * x^5 + 3 * x^4 - 4 * x^3 + 7 * x^2 - x - 9) := by
sorry

end NUMINAMATH_CALUDE_find_g_of_x_l3332_333236


namespace NUMINAMATH_CALUDE_reciprocal_equal_reciprocal_equal_opposite_sign_l3332_333214

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem for numbers equal to their own reciprocal
theorem reciprocal_equal (x : ℝ) : x = 1 / x ↔ x = 1 ∨ x = -1 := by sorry

-- Theorem for numbers equal to their own reciprocal with opposite sign
theorem reciprocal_equal_opposite_sign (y : ℂ) : y = -1 / y ↔ y = i ∨ y = -i := by sorry

end NUMINAMATH_CALUDE_reciprocal_equal_reciprocal_equal_opposite_sign_l3332_333214


namespace NUMINAMATH_CALUDE_car_highway_efficiency_l3332_333284

/-- Represents the fuel efficiency of a car -/
structure CarFuelEfficiency where
  highway : ℝ  -- Miles per gallon on highway
  city : ℝ     -- Miles per gallon in city

/-- The car's fuel efficiency satisfies the given conditions -/
def satisfies_conditions (car : CarFuelEfficiency) : Prop :=
  car.city = 20 ∧
  (4 / car.highway + 4 / car.city) = (8 / car.highway) * 1.4000000000000001

/-- The theorem stating that the car's highway fuel efficiency is 36 miles per gallon -/
theorem car_highway_efficiency :
  ∃ (car : CarFuelEfficiency), satisfies_conditions car ∧ car.highway = 36 := by
  sorry

end NUMINAMATH_CALUDE_car_highway_efficiency_l3332_333284


namespace NUMINAMATH_CALUDE_grid_column_contains_all_numbers_l3332_333228

/-- Represents the state of the grid after a certain number of transformations -/
structure GridState (n : ℕ) :=
  (grid : Fin n → Fin n → Fin n)

/-- Represents the transformation rule for the grid -/
def transform_row (n k m : ℕ) (row : Fin n → Fin n) : Fin n → Fin n :=
  sorry

/-- Fills the grid according to the given rule -/
def fill_grid (n k m : ℕ) : ℕ → GridState n :=
  sorry

theorem grid_column_contains_all_numbers
  (n k m : ℕ) 
  (h_m_gt_k : m > k) 
  (h_coprime : Nat.Coprime m (n - k)) :
  ∀ (col : Fin n), 
    ∃ (rows : Finset (Fin n)), 
      rows.card = n ∧ 
      (∀ i : Fin n, ∃ row ∈ rows, (fill_grid n k m n).grid row col = i) :=
sorry

end NUMINAMATH_CALUDE_grid_column_contains_all_numbers_l3332_333228


namespace NUMINAMATH_CALUDE_handshakes_in_specific_gathering_l3332_333219

/-- Represents a gathering of people -/
structure Gathering where
  total : Nat
  group1 : Nat
  group2 : Nat
  group1_knows_each_other : Bool
  group2_knows_no_one : Bool

/-- Calculates the number of handshakes in a gathering -/
def count_handshakes (g : Gathering) : Nat :=
  if g.group1_knows_each_other && g.group2_knows_no_one then
    (g.group2 * (g.total - 1)) / 2
  else
    0  -- This case is not relevant for our specific problem

theorem handshakes_in_specific_gathering :
  let g : Gathering := {
    total := 30,
    group1 := 20,
    group2 := 10,
    group1_knows_each_other := true,
    group2_knows_no_one := true
  }
  count_handshakes g = 145 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_in_specific_gathering_l3332_333219


namespace NUMINAMATH_CALUDE_calculate_expression_l3332_333238

theorem calculate_expression : -Real.sqrt 4 + |Real.sqrt 2 - 2| - (2023 : ℝ)^0 = -2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3332_333238


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3332_333205

-- Problem 1
theorem problem_1 (x : ℝ) : (-2 * x^2)^3 + x^2 * x^4 - (-3 * x^3)^2 = -16 * x^6 := by
  sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : (a - b)^2 * (b - a)^4 + (b - a)^3 * (a - b)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3332_333205


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3332_333206

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  is_positive : ∀ n, a n > 0

/-- Theorem: In an arithmetic sequence with positive terms, if 2a₆ + 2a₈ = a₇², then a₇ = 4 -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
    (h : 2 * seq.a 6 + 2 * seq.a 8 = (seq.a 7) ^ 2) : 
    seq.a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3332_333206


namespace NUMINAMATH_CALUDE_largest_base_for_12_cubed_digit_sum_base_8_sum_not_9_l3332_333275

def base_representation (n : ℕ) (b : ℕ) : List ℕ :=
  sorry

def sum_of_digits (n : ℕ) (b : ℕ) : ℕ :=
  (base_representation n b).sum

def power_in_base (base : ℕ) (n : ℕ) (power : ℕ) : ℕ :=
  sorry

theorem largest_base_for_12_cubed_digit_sum :
  ∀ b : ℕ, b > 8 → sum_of_digits (power_in_base b 12 3) b = 9 :=
by sorry

theorem base_8_sum_not_9 :
  sum_of_digits (power_in_base 8 12 3) 8 ≠ 9 :=
by sorry

end NUMINAMATH_CALUDE_largest_base_for_12_cubed_digit_sum_base_8_sum_not_9_l3332_333275


namespace NUMINAMATH_CALUDE_base_five_digits_of_1297_l3332_333210

theorem base_five_digits_of_1297 : ∃ n : ℕ, n > 0 ∧ 5^(n-1) ≤ 1297 ∧ 1297 < 5^n ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_base_five_digits_of_1297_l3332_333210


namespace NUMINAMATH_CALUDE_liam_strawberry_candies_l3332_333209

theorem liam_strawberry_candies :
  ∀ (s g : ℕ),
  s = 3 * g →                     -- Initial condition
  s - 15 = 4 * (g - 15) →         -- Condition after giving away candies
  s = 135 :=                      -- Conclusion to prove
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_liam_strawberry_candies_l3332_333209


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3332_333269

theorem min_value_of_expression (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b < 0) 
  (h3 : a - b = 5) : 
  ∃ (m : ℝ), m = 1/2 ∧ ∀ x, x = 1/(a+1) + 1/(2-b) → x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3332_333269


namespace NUMINAMATH_CALUDE_grid_paths_4_3_l3332_333299

/-- The number of paths on a grid from (0,0) to (m,n) using exactly (m+n) steps -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) n

theorem grid_paths_4_3 : gridPaths 4 3 = 35 := by sorry

end NUMINAMATH_CALUDE_grid_paths_4_3_l3332_333299


namespace NUMINAMATH_CALUDE_train_platform_ratio_l3332_333276

/-- Proves that the ratio of train length to platform length is 1:1 given the specified conditions --/
theorem train_platform_ratio (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) :
  train_speed = 216 * (1000 / 3600) →
  train_length = 1800 →
  crossing_time = 60 →
  ∃ (platform_length : ℝ), train_length / platform_length = 1 := by
  sorry


end NUMINAMATH_CALUDE_train_platform_ratio_l3332_333276


namespace NUMINAMATH_CALUDE_fruit_bowl_oranges_l3332_333258

theorem fruit_bowl_oranges :
  let bananas : ℕ := 7
  let apples : ℕ := 2 * bananas
  let pears : ℕ := 4
  let grapes : ℕ := apples / 2
  let total_fruits : ℕ := 40
  let oranges : ℕ := total_fruits - (bananas + apples + pears + grapes)
  oranges = 8 := by sorry

end NUMINAMATH_CALUDE_fruit_bowl_oranges_l3332_333258


namespace NUMINAMATH_CALUDE_ellipse_intersection_slope_product_l3332_333283

/-- Given a line l passing through M(-2,0) with slope k1 (k1 ≠ 0) intersecting 
    the ellipse x^2 + 2y^2 = 4 at P1 and P2, and P being the midpoint of P1P2,
    if k2 is the slope of OP, then k1k2 = -1/2 -/
theorem ellipse_intersection_slope_product 
  (k1 : ℝ) (P1 P2 P : ℝ × ℝ) (k2 : ℝ)
  (h1 : k1 ≠ 0)
  (h2 : P1.1^2 + 2*P1.2^2 = 4)
  (h3 : P2.1^2 + 2*P2.2^2 = 4)
  (h4 : P1.2 = k1 * (P1.1 + 2))
  (h5 : P2.2 = k1 * (P2.1 + 2))
  (h6 : P = ((P1.1 + P2.1)/2, (P1.2 + P2.2)/2))
  (h7 : k2 = P.2 / P.1) :
  k1 * k2 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_slope_product_l3332_333283


namespace NUMINAMATH_CALUDE_equation_solution_l3332_333267

theorem equation_solution : ∃ x : ℚ, (1 / 3 - 1 / 4 : ℚ) = 1 / (2 * x) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3332_333267


namespace NUMINAMATH_CALUDE_platform_length_l3332_333215

/-- Given a train of length 300 meters that takes 18 seconds to cross a post
    and 39 seconds to cross a platform, the length of the platform is 350 meters. -/
theorem platform_length (train_length : ℝ) (post_time : ℝ) (platform_time : ℝ) :
  train_length = 300 →
  post_time = 18 →
  platform_time = 39 →
  ∃ (platform_length : ℝ),
    platform_length = 350 ∧
    (train_length / post_time) * platform_time = train_length + platform_length :=
by
  sorry


end NUMINAMATH_CALUDE_platform_length_l3332_333215


namespace NUMINAMATH_CALUDE_fourth_side_distance_l3332_333298

/-- Given a square and a point inside it, if the distances from the point to three sides are 4, 7, and 12,
    then the distance to the fourth side is either 9 or 15. -/
theorem fourth_side_distance (s : ℝ) (d1 d2 d3 d4 : ℝ) : 
  s > 0 ∧ d1 = 4 ∧ d2 = 7 ∧ d3 = 12 ∧ 
  d1 + d2 + d3 + d4 = s → 
  d4 = 9 ∨ d4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_distance_l3332_333298


namespace NUMINAMATH_CALUDE_circle_area_increase_l3332_333222

theorem circle_area_increase (r : ℝ) (h : r > 0) : 
  let new_radius := 1.12 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  new_area = 1.2544 * original_area := by
sorry

end NUMINAMATH_CALUDE_circle_area_increase_l3332_333222


namespace NUMINAMATH_CALUDE_sum_of_angles_F_and_C_l3332_333261

-- Define the circle and points
variable (circle : Circle ℝ)
variable (A B C D E : circle.sphere)

-- Define the arcs and their measures
variable (arc_AB arc_DE : circle.sphere)
variable (measure_AB measure_DE : ℝ)

-- Define point F as intersection of chords
variable (F : circle.sphere)

-- Hypotheses
variable (h1 : measure_AB = 60)
variable (h2 : measure_DE = 72)
variable (h3 : F ∈ (circle.chord A C) ∩ (circle.chord B D))

-- Theorem statement
theorem sum_of_angles_F_and_C :
  ∃ (angle_F angle_C : ℝ),
    angle_F + angle_C = 42 ∧
    angle_F = abs ((measure circle.arc A C - measure circle.arc B D) / 2) ∧
    angle_C = measure_DE / 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_angles_F_and_C_l3332_333261


namespace NUMINAMATH_CALUDE_probability_at_least_two_successes_probability_one_from_pair_contest_probabilities_l3332_333245

/-- The probability of getting at least two successes in three independent trials 
    with a 50% success rate for each trial -/
theorem probability_at_least_two_successes : ℝ := by sorry

/-- The probability of selecting exactly one item from a specific pair 
    when selecting 2 out of 4 items -/
theorem probability_one_from_pair : ℝ := by sorry

/-- Proof of the probabilities for the contest scenario -/
theorem contest_probabilities : 
  probability_at_least_two_successes = 1/2 ∧ 
  probability_one_from_pair = 2/3 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_two_successes_probability_one_from_pair_contest_probabilities_l3332_333245


namespace NUMINAMATH_CALUDE_function_value_2024_l3332_333251

def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem function_value_2024 (a b c : ℝ) 
  (h2021 : f a b c 2021 = 2021)
  (h2022 : f a b c 2022 = 2022)
  (h2023 : f a b c 2023 = 2023) :
  f a b c 2024 = 2030 := by
  sorry

end NUMINAMATH_CALUDE_function_value_2024_l3332_333251


namespace NUMINAMATH_CALUDE_sequence_difference_l3332_333264

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

def geometric_sequence (g₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := g₁ * r^(n - 1 : ℕ)

theorem sequence_difference : 
  let a₁ := 3
  let a₂ := 11
  let g₁ := 2
  let g₂ := 10
  let d := a₂ - a₁
  let r := g₂ / g₁
  |arithmetic_sequence a₁ d 100 - geometric_sequence g₁ r 4| = 545 := by
sorry

end NUMINAMATH_CALUDE_sequence_difference_l3332_333264


namespace NUMINAMATH_CALUDE_right_shift_two_units_l3332_333278

/-- Represents a linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- Transformation that moves a function horizontally -/
def horizontalShift (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { m := f.m, b := f.b - f.m * shift }

theorem right_shift_two_units (f : LinearFunction) :
  f.m = 2 ∧ f.b = 1 →
  (horizontalShift f 2).m = 2 ∧ (horizontalShift f 2).b = -3 := by
  sorry

end NUMINAMATH_CALUDE_right_shift_two_units_l3332_333278


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l3332_333290

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_second : a 2 = 4)
  (h_sixth : a 6 = 2) :
  a 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l3332_333290


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3332_333291

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_positive_geometric_sequence a →
  a 1 * a 3 + 2 * a 2 * a 4 + a 2 * a 6 = 9 →
  a 2 + a 4 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3332_333291


namespace NUMINAMATH_CALUDE_tim_sleep_hours_l3332_333237

/-- The number of hours Tim sleeps each day for the first two days -/
def initial_sleep_hours : ℕ := 6

/-- The number of days Tim sleeps for the initial period -/
def initial_days : ℕ := 2

/-- The total number of days Tim sleeps -/
def total_days : ℕ := 4

/-- The total number of hours Tim sleeps over all days -/
def total_sleep_hours : ℕ := 32

/-- Theorem stating that Tim slept 10 hours each for the next 2 days -/
theorem tim_sleep_hours :
  (total_sleep_hours - initial_sleep_hours * initial_days) / (total_days - initial_days) = 10 :=
sorry

end NUMINAMATH_CALUDE_tim_sleep_hours_l3332_333237


namespace NUMINAMATH_CALUDE_max_investment_at_lower_rate_l3332_333256

theorem max_investment_at_lower_rate 
  (total_investment : ℝ) 
  (low_rate high_rate : ℝ) 
  (min_interest : ℝ) 
  (h1 : total_investment = 25000)
  (h2 : low_rate = 0.07)
  (h3 : high_rate = 0.12)
  (h4 : min_interest = 2450) :
  let max_low_investment := 11000
  ∀ x : ℝ, 
    0 ≤ x ∧ 
    x ≤ total_investment ∧ 
    low_rate * x + high_rate * (total_investment - x) ≥ min_interest →
    x ≤ max_low_investment := by
sorry

end NUMINAMATH_CALUDE_max_investment_at_lower_rate_l3332_333256


namespace NUMINAMATH_CALUDE_garden_length_l3332_333265

theorem garden_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = 2 + 3 * width →
  perimeter = 2 * length + 2 * width →
  perimeter = 100 →
  length = 38 := by
sorry

end NUMINAMATH_CALUDE_garden_length_l3332_333265


namespace NUMINAMATH_CALUDE_cooking_and_yoga_count_l3332_333246

/-- Represents the number of people in various curriculum groups -/
structure CurriculumGroups where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  allCurriculums : ℕ
  cookingAndWeaving : ℕ

/-- Theorem stating the number of people studying both cooking and yoga -/
theorem cooking_and_yoga_count (g : CurriculumGroups) 
  (h1 : g.yoga = 25)
  (h2 : g.cooking = 18)
  (h3 : g.weaving = 10)
  (h4 : g.cookingOnly = 4)
  (h5 : g.allCurriculums = 4)
  (h6 : g.cookingAndWeaving = 5) :
  g.cooking - g.cookingOnly - g.cookingAndWeaving + g.allCurriculums = 9 :=
by sorry

end NUMINAMATH_CALUDE_cooking_and_yoga_count_l3332_333246


namespace NUMINAMATH_CALUDE_power_multiplication_subtraction_l3332_333200

theorem power_multiplication_subtraction (a : ℝ) : 4 * a * a^3 - a^4 = 3 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_subtraction_l3332_333200


namespace NUMINAMATH_CALUDE_david_airport_distance_l3332_333226

/-- The distance from David's home to the airport --/
def airport_distance : ℝ := by sorry

/-- David's initial speed --/
def initial_speed : ℝ := 35

/-- David's speed increase --/
def speed_increase : ℝ := 15

/-- Time saved by increasing speed --/
def time_saved : ℝ := 1.5

/-- Time early --/
def time_early : ℝ := 0.5

theorem david_airport_distance :
  airport_distance = initial_speed * (airport_distance / initial_speed) +
  (initial_speed + speed_increase) * (time_saved - time_early) ∧
  airport_distance = 210 := by sorry

end NUMINAMATH_CALUDE_david_airport_distance_l3332_333226


namespace NUMINAMATH_CALUDE_couch_price_l3332_333232

theorem couch_price (chair_price : ℝ) 
  (h1 : chair_price + 3 * chair_price + 5 * (3 * chair_price) = 380) : 
  5 * (3 * chair_price) = 300 := by
  sorry

end NUMINAMATH_CALUDE_couch_price_l3332_333232


namespace NUMINAMATH_CALUDE_angle_sum_properties_l3332_333287

/-- Given two obtuse angles α and β whose terminal sides intersect the unit circle at points
    with x-coordinates -√2/10 and -2√5/5 respectively, prove that tan(α+β) = -5/3 and α+2β = 9π/4 -/
theorem angle_sum_properties (α β : Real) (hα : α > π/2) (hβ : β > π/2)
  (hA : Real.cos α = -Real.sqrt 2 / 10)
  (hB : Real.cos β = -2 * Real.sqrt 5 / 5) :
  Real.tan (α + β) = -5/3 ∧ α + 2*β = 9*π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_properties_l3332_333287


namespace NUMINAMATH_CALUDE_conic_is_parabola_l3332_333201

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Defines the equation of the conic section -/
def conicEquation (p : Point2D) : Prop :=
  |p.y - 3| = Real.sqrt ((p.x + 4)^2 + (p.y - 1)^2)

/-- Defines a parabola in the form y = ax² + bx + c where a ≠ 0 -/
def isParabola (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- Theorem stating that the given equation represents a parabola -/
theorem conic_is_parabola :
  ∃ f : ℝ → ℝ, (∀ p : Point2D, conicEquation p ↔ p.y = f p.x) ∧ isParabola f :=
sorry

end NUMINAMATH_CALUDE_conic_is_parabola_l3332_333201


namespace NUMINAMATH_CALUDE_spinner_probability_l3332_333234

def spinner_numbers : List ℕ := [4, 6, 7, 11, 12, 13, 17, 18]

def total_sections : ℕ := 8

def favorable_outcomes : ℕ := (spinner_numbers.filter (λ x => x > 10)).length

theorem spinner_probability : 
  (favorable_outcomes : ℚ) / total_sections = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3332_333234


namespace NUMINAMATH_CALUDE_distinct_fm_pairs_count_l3332_333285

/-- Represents the gender of a person -/
inductive Gender
| Male
| Female

/-- Represents a seating arrangement of 5 people around a round table -/
def SeatingArrangement := Vector Gender 5

/-- Counts the number of people sitting next to at least one female -/
def count_next_to_female (arrangement : SeatingArrangement) : Nat :=
  sorry

/-- Counts the number of people sitting next to at least one male -/
def count_next_to_male (arrangement : SeatingArrangement) : Nat :=
  sorry

/-- Generates all distinct seating arrangements -/
def all_distinct_arrangements : List SeatingArrangement :=
  sorry

/-- The main theorem stating that there are exactly 8 distinct (f, m) pairs -/
theorem distinct_fm_pairs_count :
  (all_distinct_arrangements.map (λ arr => (count_next_to_female arr, count_next_to_male arr))).toFinset.card = 8 :=
  sorry

end NUMINAMATH_CALUDE_distinct_fm_pairs_count_l3332_333285


namespace NUMINAMATH_CALUDE_expression_evaluation_l3332_333260

theorem expression_evaluation :
  let x : ℝ := 2
  let expr := (x^2 + x) / (x^2 - 2*x + 1) / ((2 / (x - 1)) - (1 / x))
  expr = 4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3332_333260


namespace NUMINAMATH_CALUDE_only_equation_II_has_nontrivial_solution_l3332_333242

theorem only_equation_II_has_nontrivial_solution :
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) ∧
  (Real.sqrt (a^2 + b^2 + c^2) = c) ∧
  (∀ (x y z : ℝ), (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) →
    (Real.sqrt (x^2 + y^2 + z^2) ≠ 0) ∧
    (Real.sqrt (x^2 + y^2 + z^2) ≠ x + y + z) ∧
    (Real.sqrt (x^2 + y^2 + z^2) ≠ x*y*z)) :=
by sorry

end NUMINAMATH_CALUDE_only_equation_II_has_nontrivial_solution_l3332_333242


namespace NUMINAMATH_CALUDE_absent_students_probability_l3332_333280

theorem absent_students_probability 
  (p_absent : ℝ) 
  (h_p_absent : p_absent = 1 / 10) 
  (p_present : ℝ) 
  (h_p_present : p_present = 1 - p_absent) 
  (n_students : ℕ) 
  (h_n_students : n_students = 3) :
  (n_students.choose 2 : ℝ) * p_absent^2 * p_present = 27 / 1000 := by
sorry

end NUMINAMATH_CALUDE_absent_students_probability_l3332_333280


namespace NUMINAMATH_CALUDE_tuesday_greatest_diff_greatest_diff_day_is_tuesday_l3332_333247

-- Define the temperature difference for each day
def monday_diff : ℤ := 5 - 2
def tuesday_diff : ℤ := 4 - (-1)
def wednesday_diff : ℤ := 0 - (-4)

-- Theorem stating that Tuesday has the greatest temperature difference
theorem tuesday_greatest_diff : 
  tuesday_diff > monday_diff ∧ tuesday_diff > wednesday_diff :=
by
  sorry

-- Define a function to get the day with the greatest temperature difference
def day_with_greatest_diff : String :=
  if tuesday_diff > monday_diff ∧ tuesday_diff > wednesday_diff then
    "Tuesday"
  else if monday_diff > tuesday_diff ∧ monday_diff > wednesday_diff then
    "Monday"
  else
    "Wednesday"

-- Theorem stating that the day with the greatest temperature difference is Tuesday
theorem greatest_diff_day_is_tuesday : 
  day_with_greatest_diff = "Tuesday" :=
by
  sorry

end NUMINAMATH_CALUDE_tuesday_greatest_diff_greatest_diff_day_is_tuesday_l3332_333247


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3332_333230

/-- Given vectors a and b, if (-2a + b) is parallel to (a + kb), then k = -1/2 -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (-3, 1)) 
  (h2 : b = (1, -2)) 
  (h_parallel : ∃ (t : ℝ), t • (-2 • a + b) = (a + k • b)) :
  k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3332_333230


namespace NUMINAMATH_CALUDE_digit_222_of_55_div_777_l3332_333270

/-- The decimal representation of a rational number -/
def decimal_representation (n d : ℕ) : ℕ → ℕ :=
  sorry

/-- The length of the repeating block in the decimal representation of a rational number -/
def repeating_block_length (n d : ℕ) : ℕ :=
  sorry

theorem digit_222_of_55_div_777 :
  decimal_representation 55 777 222 = 7 :=
sorry

end NUMINAMATH_CALUDE_digit_222_of_55_div_777_l3332_333270


namespace NUMINAMATH_CALUDE_length_of_cd_l3332_333271

/-- Given a line segment CD with points M and N on it, prove that CD has length 57.6 -/
theorem length_of_cd (C D M N : ℝ × ℝ) : 
  (∃ t : ℝ, M = (1 - t) • C + t • D ∧ 0 < t ∧ t < 1/2) →  -- M is on CD and same side of midpoint
  (∃ s : ℝ, N = (1 - s) • C + s • D ∧ 0 < s ∧ s < 1/2) →  -- N is on CD and same side of midpoint
  (dist C M) / (dist M D) = 3/5 →                         -- M divides CD in ratio 3:5
  (dist C N) / (dist N D) = 4/5 →                         -- N divides CD in ratio 4:5
  dist M N = 4 →                                          -- Length of MN is 4
  dist C D = 57.6 :=                                      -- Length of CD is 57.6
by sorry

end NUMINAMATH_CALUDE_length_of_cd_l3332_333271


namespace NUMINAMATH_CALUDE_oil_leak_total_l3332_333216

/-- The total amount of oil leaked from three pipes -/
def total_oil_leaked (pipe1_before pipe1_during pipe2_before pipe2_during pipe3_before pipe3_rate pipe3_hours : ℕ) : ℕ :=
  pipe1_before + pipe1_during + pipe2_before + pipe2_during + pipe3_before + pipe3_rate * pipe3_hours

/-- Theorem stating that the total amount of oil leaked is 32,975 liters -/
theorem oil_leak_total :
  total_oil_leaked 6522 2443 8712 3894 9654 250 7 = 32975 := by
  sorry

end NUMINAMATH_CALUDE_oil_leak_total_l3332_333216


namespace NUMINAMATH_CALUDE_half_of_1_01_l3332_333221

theorem half_of_1_01 : (1.01 : ℝ) / 2 = 0.505 := by
  sorry

end NUMINAMATH_CALUDE_half_of_1_01_l3332_333221


namespace NUMINAMATH_CALUDE_students_between_50_and_90_count_l3332_333204

/-- Represents the distribution of student scores -/
structure ScoreDistribution where
  total_students : ℕ
  mean : ℝ
  std_dev : ℝ
  above_90 : ℕ

/-- Calculates the number of students scoring between 50 and 90 -/
def students_between_50_and_90 (d : ScoreDistribution) : ℕ :=
  d.total_students - 2 * d.above_90

/-- Theorem stating the number of students scoring between 50 and 90 -/
theorem students_between_50_and_90_count
  (d : ScoreDistribution)
  (h1 : d.total_students = 10000)
  (h2 : d.mean = 70)
  (h3 : d.std_dev = 10)
  (h4 : d.above_90 = 230) :
  students_between_50_and_90 d = 9540 := by
  sorry

#check students_between_50_and_90_count

end NUMINAMATH_CALUDE_students_between_50_and_90_count_l3332_333204


namespace NUMINAMATH_CALUDE_prime_arithmetic_sequence_ones_digit_l3332_333295

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def ones_digit (n : ℕ) : ℕ := n % 10

theorem prime_arithmetic_sequence_ones_digit 
  (p q r : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hr : is_prime r) 
  (h_seq : q = p + 4 ∧ r = q + 4) 
  (h_p_gt_5 : p > 5) :
  ones_digit p = 3 ∨ ones_digit p = 9 :=
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_sequence_ones_digit_l3332_333295


namespace NUMINAMATH_CALUDE_billiard_path_to_top_left_l3332_333249

/-- Represents a point in a 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Represents a rectangular lattice -/
structure RectangularLattice where
  width : ℕ
  height : ℕ

def billiardTable : RectangularLattice := { width := 1965, height := 26 }

/-- Checks if a point is on the top edge of the lattice -/
def isTopEdge (l : RectangularLattice) (p : LatticePoint) : Prop :=
  p.x = 0 ∧ p.y = l.height

/-- Represents a line with slope 1 starting from (0, 0) -/
def slopeLine (x : ℤ) : LatticePoint :=
  { x := x, y := x }

theorem billiard_path_to_top_left :
  ∃ (n : ℕ), isTopEdge billiardTable (slopeLine (n * billiardTable.width)) := by
  sorry

end NUMINAMATH_CALUDE_billiard_path_to_top_left_l3332_333249


namespace NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l3332_333292

/-- A fair coin is tossed eight times. -/
def coin_tosses : ℕ := 8

/-- The coin is fair, meaning the probability of heads is 1/2. -/
def fair_coin_prob : ℚ := 1/2

/-- The number of heads we're looking for. -/
def target_heads : ℕ := 3

/-- The probability of getting exactly three heads in eight tosses of a fair coin. -/
theorem three_heads_in_eight_tosses : 
  (Nat.choose coin_tosses target_heads : ℚ) * fair_coin_prob^target_heads * (1 - fair_coin_prob)^(coin_tosses - target_heads) = 7/32 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l3332_333292


namespace NUMINAMATH_CALUDE_product_301_52_base7_units_digit_l3332_333255

theorem product_301_52_base7_units_digit (a b : ℕ) (ha : a = 301) (hb : b = 52) :
  (a * b) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_301_52_base7_units_digit_l3332_333255


namespace NUMINAMATH_CALUDE_three_sixes_is_random_event_l3332_333294

/-- Represents the possible outcomes of rolling a die -/
inductive DieOutcome
  | One
  | Two
  | Three
  | Four
  | Five
  | Six

/-- Represents the result of rolling 3 dice simultaneously -/
structure ThreeDiceRoll :=
  (first second third : DieOutcome)

/-- Defines the event of getting three 6s -/
def allSixes (roll : ThreeDiceRoll) : Prop :=
  roll.first = DieOutcome.Six ∧ 
  roll.second = DieOutcome.Six ∧ 
  roll.third = DieOutcome.Six

/-- Theorem stating that rolling three 6s is a random event -/
theorem three_sixes_is_random_event : 
  ∃ (roll : ThreeDiceRoll), allSixes roll ∧
  ∃ (roll' : ThreeDiceRoll), ¬allSixes roll' :=
sorry

end NUMINAMATH_CALUDE_three_sixes_is_random_event_l3332_333294


namespace NUMINAMATH_CALUDE_complement_of_P_l3332_333274

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set P
def P : Set ℝ := {x | x - 1 < 0}

-- State the theorem
theorem complement_of_P : Set.compl P = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_l3332_333274


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3332_333293

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((i - 1) / (i + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3332_333293


namespace NUMINAMATH_CALUDE_xy_value_l3332_333297

theorem xy_value (x y : ℝ) (h : (x + y)^2 - (x - y)^2 = 20) : x * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3332_333297


namespace NUMINAMATH_CALUDE_inscribed_squares_side_length_difference_l3332_333296

/-- Given a circle with radius R and a chord at distance h from the center,
    prove that the difference in side lengths of two squares inscribed in the
    segments formed by the chord is 8h/5. Each square has two adjacent vertices
    on the chord and two on the circle arc. -/
theorem inscribed_squares_side_length_difference
  (R h : ℝ) (h_pos : 0 < h) (h_lt_R : h < R) :
  ∃ x y : ℝ,
    (0 < x) ∧ (0 < y) ∧
    ((2 * x - h)^2 + x^2 = R^2) ∧
    ((2 * y + h)^2 + y^2 = R^2) ∧
    (2 * x - 2 * y = 8 * h / 5) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_squares_side_length_difference_l3332_333296


namespace NUMINAMATH_CALUDE_only_100_factorial_makes_perfect_square_l3332_333244

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

def product_of_remaining_factorials (k : ℕ) : ℕ :=
  (List.range 200).foldl (λ acc i => if i + 1 ≠ k then acc * factorial (i + 1) else acc) 1

theorem only_100_factorial_makes_perfect_square :
  ∀ k : ℕ, k ≤ 200 →
    (is_perfect_square (product_of_remaining_factorials k) ↔ k = 100) :=
by sorry

end NUMINAMATH_CALUDE_only_100_factorial_makes_perfect_square_l3332_333244


namespace NUMINAMATH_CALUDE_identity_unique_l3332_333217

-- Define a group structure
class MyGroup (G : Type) where
  mul : G → G → G
  one : G
  inv : G → G
  mul_assoc : ∀ a b c : G, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a : G, mul one a = a
  mul_one : ∀ a : G, mul a one = a
  mul_left_inv : ∀ a : G, mul (inv a) a = one

-- State the theorem
theorem identity_unique {G : Type} [MyGroup G] (e e' : G)
    (h1 : ∀ g : G, MyGroup.mul e g = g ∧ MyGroup.mul g e = g)
    (h2 : ∀ g : G, MyGroup.mul e' g = g ∧ MyGroup.mul g e' = g) :
    e = e' := by sorry

end NUMINAMATH_CALUDE_identity_unique_l3332_333217


namespace NUMINAMATH_CALUDE_andrena_christel_doll_difference_l3332_333289

/-- Proves that Andrena has 2 more dolls than Christel after gift exchanges -/
theorem andrena_christel_doll_difference :
  -- Initial conditions
  ∀ (debelyn_initial christel_initial andrena_initial : ℕ),
  debelyn_initial = 20 →
  christel_initial = 24 →
  -- Gift exchanges
  ∀ (debelyn_to_andrena christel_to_andrena : ℕ),
  debelyn_to_andrena = 2 →
  christel_to_andrena = 5 →
  -- Final condition
  andrena_initial + debelyn_to_andrena + christel_to_andrena =
    debelyn_initial - debelyn_to_andrena + 3 →
  -- Conclusion
  (andrena_initial + debelyn_to_andrena + christel_to_andrena) -
    (christel_initial - christel_to_andrena) = 2 :=
by sorry

end NUMINAMATH_CALUDE_andrena_christel_doll_difference_l3332_333289


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3332_333227

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 4 + a 5 + a 6 = 168 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3332_333227


namespace NUMINAMATH_CALUDE_sharp_composition_50_l3332_333259

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.5 * N + 1

-- Theorem statement
theorem sharp_composition_50 : sharp (sharp (sharp 50)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sharp_composition_50_l3332_333259


namespace NUMINAMATH_CALUDE_simplify_power_expression_l3332_333286

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^4 = 81 * x^16 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l3332_333286


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3332_333288

/-- For an ellipse where the length of the major axis is twice its focal length, the eccentricity is 1/2. -/
theorem ellipse_eccentricity (a c : ℝ) (h : a = 2 * c) : c / a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3332_333288


namespace NUMINAMATH_CALUDE_sock_profit_calculation_l3332_333250

/-- Calculates the total profit from selling socks given specific conditions. -/
theorem sock_profit_calculation : 
  let total_pairs : ℕ := 9
  let cost_per_pair : ℚ := 2
  let purchase_discount : ℚ := 0.1
  let profit_percentage_4_pairs : ℚ := 0.25
  let profit_per_pair_5_pairs : ℚ := 0.2
  let sales_tax : ℚ := 0.05

  let discounted_cost := total_pairs * cost_per_pair * (1 - purchase_discount)
  let selling_price_4_pairs := 4 * cost_per_pair * (1 + profit_percentage_4_pairs)
  let selling_price_5_pairs := 5 * cost_per_pair + 5 * profit_per_pair_5_pairs
  let total_selling_price := (selling_price_4_pairs + selling_price_5_pairs) * (1 + sales_tax)
  let total_profit := total_selling_price - discounted_cost

  total_profit = 5.85 := by sorry

end NUMINAMATH_CALUDE_sock_profit_calculation_l3332_333250


namespace NUMINAMATH_CALUDE_saturday_hourly_rate_l3332_333263

/-- Calculates the hourly rate for Saturday work given the following conditions:
  * After-school hourly rate is $4.00
  * Total weekly hours worked is 18
  * Total weekly earnings is $88.00
  * Saturday hours worked is 8.0
-/
theorem saturday_hourly_rate
  (after_school_rate : ℝ)
  (total_hours : ℝ)
  (total_earnings : ℝ)
  (saturday_hours : ℝ)
  (h1 : after_school_rate = 4)
  (h2 : total_hours = 18)
  (h3 : total_earnings = 88)
  (h4 : saturday_hours = 8) :
  (total_earnings - after_school_rate * (total_hours - saturday_hours)) / saturday_hours = 6 :=
by sorry

end NUMINAMATH_CALUDE_saturday_hourly_rate_l3332_333263


namespace NUMINAMATH_CALUDE_gumball_range_l3332_333262

theorem gumball_range (x : ℤ) : 
  let carolyn := 17
  let lew := 12
  let total := carolyn + lew + x
  let avg := total / 3
  (19 ≤ avg ∧ avg ≤ 25) →
  (max x - min x = 18) :=
by sorry

end NUMINAMATH_CALUDE_gumball_range_l3332_333262


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l3332_333254

theorem quadratic_roots_product (p q : ℝ) : 
  (3 * p ^ 2 + 9 * p - 21 = 0) → 
  (3 * q ^ 2 + 9 * q - 21 = 0) → 
  (3 * p - 4) * (6 * q - 8) = 122 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l3332_333254


namespace NUMINAMATH_CALUDE_original_number_proof_l3332_333202

theorem original_number_proof (N : ℤ) : (N + 1) % 9 = 0 → N = 8 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3332_333202


namespace NUMINAMATH_CALUDE_range_of_f_l3332_333257

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f :
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3332_333257


namespace NUMINAMATH_CALUDE_unique_solution_condition_inequality_condition_l3332_333207

/-- The function f(x) = x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The function g(x) = a|x-1| -/
def g (a x : ℝ) : ℝ := a * |x - 1|

/-- If |f(x)| = g(x) has only one real solution, then a < 0 -/
theorem unique_solution_condition (a : ℝ) :
  (∃! x : ℝ, |f x| = g a x) → a < 0 := by sorry

/-- If f(x) ≥ g(x) for all x ∈ ℝ, then a ≤ -2 -/
theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) → a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_inequality_condition_l3332_333207


namespace NUMINAMATH_CALUDE_liquid_x_percentage_in_solution_b_l3332_333252

/-- Given two solutions A and B, where liquid X makes up 0.8% of solution A,
    and a mixture of 300g of A and 700g of B results in a solution with 1.5% of liquid X,
    prove that liquid X makes up 1.8% of solution B. -/
theorem liquid_x_percentage_in_solution_b : 
  let percent_x_in_a : ℝ := 0.008
  let mass_a : ℝ := 300
  let mass_b : ℝ := 700
  let percent_x_in_mixture : ℝ := 0.015
  let percent_x_in_b : ℝ := (percent_x_in_mixture * (mass_a + mass_b) - percent_x_in_a * mass_a) / mass_b
  percent_x_in_b = 0.018 := by
  sorry

end NUMINAMATH_CALUDE_liquid_x_percentage_in_solution_b_l3332_333252


namespace NUMINAMATH_CALUDE_difference_value_l3332_333241

theorem difference_value (x : ℝ) (h : x = -10) : 2 * x - (-8) = -12 := by
  sorry

end NUMINAMATH_CALUDE_difference_value_l3332_333241


namespace NUMINAMATH_CALUDE_mountain_height_theorem_l3332_333203

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the problem setup -/
structure MountainProblem where
  A : Point3D
  C : Point3D
  P : Point3D
  F : Point3D
  AC_distance : ℝ
  AP_distance : ℝ
  C_elevation : ℝ
  AC_angle : ℝ
  AP_angle : ℝ
  magnetic_declination : ℝ
  latitude : ℝ

/-- The main theorem to prove -/
theorem mountain_height_theorem (problem : MountainProblem) :
  problem.AC_distance = 2200 →
  problem.AP_distance = 400 →
  problem.C_elevation = 550 →
  problem.AC_angle = 71 →
  problem.AP_angle = 64 →
  problem.magnetic_declination = 2 →
  problem.latitude = 49 →
  ∃ (height : ℝ), abs (height - 420) < 1 ∧ height = problem.A.z :=
sorry


end NUMINAMATH_CALUDE_mountain_height_theorem_l3332_333203


namespace NUMINAMATH_CALUDE_students_taking_both_languages_l3332_333273

theorem students_taking_both_languages (total : ℕ) (french : ℕ) (german : ℕ) (neither : ℕ) :
  total = 79 →
  french = 41 →
  german = 22 →
  neither = 25 →
  french + german - (total - neither) = 9 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_both_languages_l3332_333273


namespace NUMINAMATH_CALUDE_sqrt_two_three_power_l3332_333240

/-- Given that (√2 + √3)^(2n-1) = aₙ√2 + bₙ√3 and aₙ₊₁ = paₙ + qbₙ for n ∈ ℕ₊,
    prove that p + q = 11 and 2aₙ² - 3bₙ² = -1 -/
theorem sqrt_two_three_power (n : ℕ) (hn : n > 0) 
  (a b : ℕ → ℝ) (p q : ℝ)
  (h1 : ∀ n, (Real.sqrt 2 + Real.sqrt 3) ^ (2 * n - 1) = a n * Real.sqrt 2 + b n * Real.sqrt 3)
  (h2 : ∀ n, a (n + 1) = p * a n + q * b n) :
  p + q = 11 ∧ ∀ n, 2 * (a n)^2 - 3 * (b n)^2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_three_power_l3332_333240


namespace NUMINAMATH_CALUDE_soccer_team_enrollment_l3332_333233

theorem soccer_team_enrollment (total : ℕ) (physics : ℕ) (both : ℕ) (mathematics : ℕ)
  (h1 : total = 15)
  (h2 : physics = 9)
  (h3 : both = 3)
  (h4 : physics + mathematics - both = total) :
  mathematics = 9 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_enrollment_l3332_333233


namespace NUMINAMATH_CALUDE_smallest_circle_radius_l3332_333243

/-- The radius of the smallest circle containing a triangle with sides 7, 9, and 12 -/
theorem smallest_circle_radius (a b c : ℝ) (ha : a = 7) (hb : b = 9) (hc : c = 12) :
  let R := max a (max b c) / 2
  R = 6 := by sorry

end NUMINAMATH_CALUDE_smallest_circle_radius_l3332_333243


namespace NUMINAMATH_CALUDE_complex_equation_real_l3332_333235

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_real (a : ℝ) : 
  (((2 * a : ℂ) / (1 + i) + 1 + i).im = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_real_l3332_333235


namespace NUMINAMATH_CALUDE_max_intersections_after_300_turns_l3332_333212

/-- The number of intersections formed by n lines on a plane -/
def num_intersections (n : ℕ) : ℕ := n.choose 2

/-- The number of lines after 300 turns -/
def total_lines : ℕ := 3 + 300

theorem max_intersections_after_300_turns :
  num_intersections total_lines = 45853 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_after_300_turns_l3332_333212


namespace NUMINAMATH_CALUDE_max_bc_value_l3332_333281

theorem max_bc_value (a b c : ℂ) 
  (h : ∀ z : ℂ, Complex.abs z ≤ 1 → Complex.abs (a * z^2 + b * z + c) ≤ 1) : 
  Complex.abs (b * c) ≤ (3 * Real.sqrt 3) / 16 := by
  sorry

end NUMINAMATH_CALUDE_max_bc_value_l3332_333281


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l3332_333208

def f (x : ℝ) := x^2

theorem f_is_even_and_increasing :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 ≤ x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l3332_333208


namespace NUMINAMATH_CALUDE_boundary_length_square_l3332_333279

/-- The length of the boundary formed by semi-circle arcs and line segments on a square with area 144 square units, where each side is divided into four equal parts -/
theorem boundary_length_square (square_area : ℝ) (side_divisions : ℕ) : square_area = 144 ∧ side_divisions = 4 → ∃ (boundary_length : ℝ), boundary_length = 12 * Real.pi + 24 := by
  sorry

end NUMINAMATH_CALUDE_boundary_length_square_l3332_333279
