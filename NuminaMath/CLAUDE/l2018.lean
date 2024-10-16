import Mathlib

namespace NUMINAMATH_CALUDE_whitewash_cost_l2018_201845

/-- Calculate the cost of white washing a room with given dimensions and openings. -/
theorem whitewash_cost (room_length room_width room_height : ℝ)
                       (door_length door_width : ℝ)
                       (window_length window_width : ℝ)
                       (rate : ℝ) :
  room_length = 25 →
  room_width = 15 →
  room_height = 12 →
  door_length = 6 →
  door_width = 3 →
  window_length = 4 →
  window_width = 3 →
  rate = 5 →
  (2 * (room_length * room_height + room_width * room_height) -
   (door_length * door_width + 3 * window_length * window_width)) * rate = 4530 := by
  sorry

#check whitewash_cost

end NUMINAMATH_CALUDE_whitewash_cost_l2018_201845


namespace NUMINAMATH_CALUDE_drums_filled_per_day_l2018_201816

/-- Given the total number of drums filled and the number of days, 
    calculate the number of drums filled per day -/
def drums_per_day (total_drums : ℕ) (num_days : ℕ) : ℕ :=
  total_drums / num_days

/-- Theorem stating that given 6264 drums filled in 58 days, 
    the number of drums filled per day is 108 -/
theorem drums_filled_per_day : 
  drums_per_day 6264 58 = 108 := by
  sorry

#eval drums_per_day 6264 58

end NUMINAMATH_CALUDE_drums_filled_per_day_l2018_201816


namespace NUMINAMATH_CALUDE_ratio_calculation_l2018_201877

theorem ratio_calculation (P M Q R N : ℚ) :
  R = (40 / 100) * M →
  M = (25 / 100) * Q →
  Q = (30 / 100) * P →
  N = (60 / 100) * P →
  R / N = 1 / 20 := by
sorry

end NUMINAMATH_CALUDE_ratio_calculation_l2018_201877


namespace NUMINAMATH_CALUDE_speaker_is_tweedledee_l2018_201808

-- Define the brothers
inductive Brother
| Tweedledum
| Tweedledee

-- Define the card suits
inductive Suit
| Black
| Red

-- Define the speaker
structure Speaker where
  identity : Brother
  card : Suit

-- Define the statement made by the speaker
def statement (s : Speaker) : Prop :=
  s.identity = Brother.Tweedledum → s.card ≠ Suit.Black

-- Theorem: The speaker must be Tweedledee
theorem speaker_is_tweedledee (s : Speaker) (h : statement s) : 
  s.identity = Brother.Tweedledee :=
sorry

end NUMINAMATH_CALUDE_speaker_is_tweedledee_l2018_201808


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l2018_201844

theorem bowling_ball_weight (canoe_weight : ℝ) (h1 : canoe_weight = 35) :
  let total_canoe_weight := 2 * canoe_weight
  let bowling_ball_weight := total_canoe_weight / 9
  bowling_ball_weight = 70 / 9 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l2018_201844


namespace NUMINAMATH_CALUDE_A_initial_investment_l2018_201831

/-- Represents the initial investment of A in rupees -/
def A_investment : ℝ := 27000

/-- Represents the investment of B in rupees -/
def B_investment : ℝ := 36000

/-- Represents the number of months in a year -/
def months_in_year : ℝ := 12

/-- Represents the number of months after which B joined -/
def B_join_time : ℝ := 7.5

/-- Represents the ratio of profit sharing between A and B -/
def profit_ratio : ℝ := 2

theorem A_initial_investment :
  A_investment * months_in_year = 
  profit_ratio * B_investment * (months_in_year - B_join_time) :=
by sorry

end NUMINAMATH_CALUDE_A_initial_investment_l2018_201831


namespace NUMINAMATH_CALUDE_exactville_running_difference_l2018_201857

/-- Represents the town layout with square blocks and streets --/
structure TownLayout where
  block_side : ℝ  -- Side length of a square block
  street_width : ℝ  -- Width of the streets

/-- Calculates the difference in running distance between outer and inner paths --/
def running_distance_difference (town : TownLayout) : ℝ :=
  4 * (town.block_side + 2 * town.street_width) - 4 * town.block_side

/-- Theorem stating the difference in running distance for Exactville --/
theorem exactville_running_difference :
  let town : TownLayout := { block_side := 500, street_width := 25 }
  running_distance_difference town = 200 := by
  sorry

end NUMINAMATH_CALUDE_exactville_running_difference_l2018_201857


namespace NUMINAMATH_CALUDE_animal_shelter_multiple_l2018_201896

theorem animal_shelter_multiple (puppies kittens : ℕ) (h1 : puppies = 32) (h2 : kittens = 78)
  (h3 : ∃ x : ℕ, kittens = x * puppies + 14) : 
  ∃ x : ℕ, x = 2 ∧ kittens = x * puppies + 14 := by
  sorry

end NUMINAMATH_CALUDE_animal_shelter_multiple_l2018_201896


namespace NUMINAMATH_CALUDE_function_range_and_triangle_area_l2018_201860

noncomputable def f (x : ℝ) := Real.sqrt 3 * (Real.sin x)^2 + Real.sin x * Real.cos x

theorem function_range_and_triangle_area 
  (A B C : ℝ) (a b c : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 3), f x ∈ Set.Icc 0 (Real.sqrt 3)) ∧
  (f (A / 2) = Real.sqrt 3 / 2) ∧
  (a = 4) ∧
  (b + c = 5) →
  (Set.range (fun x => f x) = Set.Icc 0 (Real.sqrt 3)) ∧
  (1 / 2 * b * c * Real.sin A = 3 * Real.sqrt 3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_function_range_and_triangle_area_l2018_201860


namespace NUMINAMATH_CALUDE_min_distance_vectors_l2018_201807

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem min_distance_vectors (a b : ℝ × ℝ) 
  (h1 : angle_between_vectors a b = 2 * Real.pi / 3)
  (h2 : a.1 * b.1 + a.2 * b.2 = -1) : 
  ∀ (c d : ℝ × ℝ), angle_between_vectors c d = 2 * Real.pi / 3 → 
  c.1 * d.1 + c.2 * d.2 = -1 → 
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ≤ Real.sqrt ((c.1 - d.1)^2 + (c.2 - d.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_vectors_l2018_201807


namespace NUMINAMATH_CALUDE_minimum_cards_to_turn_l2018_201827

/-- Represents a card with a letter on one side and a number on the other -/
structure Card where
  letter : Char
  number : Nat

/-- Checks if a character is a vowel -/
def isVowel (c : Char) : Bool :=
  c ∈ ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']

/-- Checks if a number is even -/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/-- Checks if a card satisfies the condition: 
    if it has a vowel, it must have an even number -/
def satisfiesCondition (card : Card) : Bool :=
  ¬(isVowel card.letter) || isEven card.number

/-- Represents the set of cards on the table -/
def cardSet : Finset Card := sorry

/-- The number of cards that need to be turned over -/
def cardsToTurn : Nat := sorry

theorem minimum_cards_to_turn : 
  (∀ card ∈ cardSet, satisfiesCondition card) → cardsToTurn = 3 := by
  sorry

end NUMINAMATH_CALUDE_minimum_cards_to_turn_l2018_201827


namespace NUMINAMATH_CALUDE_min_rectangles_cover_square_l2018_201835

/-- The smallest number of 3-by-4 non-overlapping rectangles needed to cover a square region -/
def min_rectangles : ℕ := 16

/-- The width of each rectangle -/
def rectangle_width : ℕ := 4

/-- The height of each rectangle -/
def rectangle_height : ℕ := 3

/-- The side length of the square region -/
def square_side : ℕ := 12

theorem min_rectangles_cover_square :
  (min_rectangles * rectangle_width * rectangle_height = square_side * square_side) ∧
  (square_side % rectangle_height = 0) ∧
  (∀ n : ℕ, n < min_rectangles →
    n * rectangle_width * rectangle_height < square_side * square_side) := by
  sorry

#check min_rectangles_cover_square

end NUMINAMATH_CALUDE_min_rectangles_cover_square_l2018_201835


namespace NUMINAMATH_CALUDE_cross_product_result_l2018_201814

def u : ℝ × ℝ × ℝ := (-3, 4, 2)
def v : ℝ × ℝ × ℝ := (8, -5, 6)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  (a₂ * b₃ - a₃ * b₂, a₃ * b₁ - a₁ * b₃, a₁ * b₂ - a₂ * b₁)

theorem cross_product_result : cross_product u v = (34, -34, -17) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_result_l2018_201814


namespace NUMINAMATH_CALUDE_sin_beta_value_l2018_201899

theorem sin_beta_value (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos α = 4 / 5)
  (h4 : Real.cos (α + β) = 3 / 5) : 
  Real.sin β = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_beta_value_l2018_201899


namespace NUMINAMATH_CALUDE_john_balloons_l2018_201834

/-- The number of balloons John bought -/
def num_balloons : ℕ := sorry

/-- The volume of air each balloon holds in liters -/
def air_per_balloon : ℕ := 10

/-- The volume of gas in each tank in liters -/
def gas_per_tank : ℕ := 500

/-- The number of tanks John needs to fill all balloons -/
def num_tanks : ℕ := 20

theorem john_balloons :
  num_balloons = 1000 :=
by sorry

end NUMINAMATH_CALUDE_john_balloons_l2018_201834


namespace NUMINAMATH_CALUDE_ratio_expression_value_l2018_201837

theorem ratio_expression_value (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 15 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_expression_value_l2018_201837


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2018_201812

theorem arithmetic_calculation : 12 - (-18) + (-11) - 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2018_201812


namespace NUMINAMATH_CALUDE_sqrt_cube_root_equality_l2018_201847

theorem sqrt_cube_root_equality (a : ℝ) (h : a > 0) : 
  Real.sqrt (a * Real.rpow a (1/3)) = Real.rpow a (2/3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_cube_root_equality_l2018_201847


namespace NUMINAMATH_CALUDE_negative_one_third_less_than_negative_point_three_l2018_201804

theorem negative_one_third_less_than_negative_point_three : -1/3 < -0.3 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_third_less_than_negative_point_three_l2018_201804


namespace NUMINAMATH_CALUDE_turnip_potato_ratio_l2018_201843

theorem turnip_potato_ratio (total_potatoes : ℝ) (total_turnips : ℝ) (base_potatoes : ℝ) 
  (h1 : total_potatoes = 20)
  (h2 : total_turnips = 8)
  (h3 : base_potatoes = 5) :
  (base_potatoes / total_potatoes) * total_turnips = 2 := by
  sorry

end NUMINAMATH_CALUDE_turnip_potato_ratio_l2018_201843


namespace NUMINAMATH_CALUDE_phone_bill_increase_l2018_201842

theorem phone_bill_increase (usual_bill : ℝ) (increase_rate : ℝ) (months : ℕ) : 
  usual_bill = 50 → 
  increase_rate = 0.1 → 
  months = 12 → 
  (usual_bill + usual_bill * increase_rate) * months = 660 := by
sorry

end NUMINAMATH_CALUDE_phone_bill_increase_l2018_201842


namespace NUMINAMATH_CALUDE_rectangle_area_l2018_201853

/-- A rectangle with diagonal d and length three times its width has area 3d²/10 -/
theorem rectangle_area (d : ℝ) (w : ℝ) (h : w > 0) : 
  w^2 + (3*w)^2 = d^2 → w * (3*w) = (3 * d^2) / 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2018_201853


namespace NUMINAMATH_CALUDE_min_tiles_for_square_l2018_201892

theorem min_tiles_for_square (tile_width : ℕ) (tile_height : ℕ) : 
  tile_width = 12 →
  tile_height = 15 →
  ∃ (square_side : ℕ) (num_tiles : ℕ),
    square_side % tile_width = 0 ∧
    square_side % tile_height = 0 ∧
    num_tiles = (square_side * square_side) / (tile_width * tile_height) ∧
    num_tiles = 20 ∧
    ∀ (smaller_side : ℕ) (smaller_num_tiles : ℕ),
      smaller_side < square_side →
      smaller_side % tile_width = 0 →
      smaller_side % tile_height = 0 →
      smaller_num_tiles = (smaller_side * smaller_side) / (tile_width * tile_height) →
      smaller_num_tiles < num_tiles :=
by sorry

end NUMINAMATH_CALUDE_min_tiles_for_square_l2018_201892


namespace NUMINAMATH_CALUDE_total_jelly_beans_l2018_201859

/-- The number of vanilla jelly beans -/
def vanilla : ℕ := 120

/-- The number of grape jelly beans -/
def grape : ℕ := 5 * vanilla + 50

/-- The number of strawberry jelly beans -/
def strawberry : ℕ := (2 * vanilla) / 3

/-- The total number of jelly beans -/
def total : ℕ := grape + vanilla + strawberry

/-- Theorem stating that the total number of jelly beans is 850 -/
theorem total_jelly_beans : total = 850 := by
  sorry

end NUMINAMATH_CALUDE_total_jelly_beans_l2018_201859


namespace NUMINAMATH_CALUDE_math_club_trips_l2018_201823

/-- Represents a math club with field trips -/
structure MathClub where
  total_students : ℕ
  students_per_trip : ℕ
  (total_students_pos : total_students > 0)
  (students_per_trip_pos : students_per_trip > 0)
  (students_per_trip_le_total : students_per_trip ≤ total_students)

/-- The minimum number of trips for one student to meet all others -/
def min_trips_for_one (club : MathClub) : ℕ :=
  (club.total_students - 1 + club.students_per_trip - 2) / (club.students_per_trip - 1)

/-- The minimum number of trips for all pairs to meet -/
def min_trips_for_all_pairs (club : MathClub) : ℕ :=
  (club.total_students * (club.total_students - 1)) / (club.students_per_trip * (club.students_per_trip - 1))

theorem math_club_trips (club : MathClub) 
  (h1 : club.total_students = 12) 
  (h2 : club.students_per_trip = 6) : 
  min_trips_for_one club = 3 ∧ min_trips_for_all_pairs club = 6 := by
  sorry

#eval min_trips_for_one ⟨12, 6, by norm_num, by norm_num, by norm_num⟩
#eval min_trips_for_all_pairs ⟨12, 6, by norm_num, by norm_num, by norm_num⟩

end NUMINAMATH_CALUDE_math_club_trips_l2018_201823


namespace NUMINAMATH_CALUDE_harriet_driving_speed_l2018_201861

/-- Harriet's driving problem -/
theorem harriet_driving_speed 
  (total_time : ℝ) 
  (time_to_b : ℝ) 
  (speed_back : ℝ) : 
  total_time = 5 → 
  time_to_b = 192 / 60 → 
  speed_back = 160 → 
  (total_time - time_to_b) * speed_back / time_to_b = 90 := by
  sorry

end NUMINAMATH_CALUDE_harriet_driving_speed_l2018_201861


namespace NUMINAMATH_CALUDE_factor_expression_l2018_201824

theorem factor_expression (x : ℝ) : x^2 * (x + 3) + 3 * (x + 3) = (x^2 + 3) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2018_201824


namespace NUMINAMATH_CALUDE_parallel_segments_l2018_201800

/-- Given four points on a Cartesian plane, if AB is parallel to XY, then k = -6 -/
theorem parallel_segments (k : ℝ) : 
  let A : ℝ × ℝ := (-6, 0)
  let B : ℝ × ℝ := (0, -6)
  let X : ℝ × ℝ := (0, 10)
  let Y : ℝ × ℝ := (16, k)
  let slope_AB := (B.2 - A.2) / (B.1 - A.1)
  let slope_XY := (Y.2 - X.2) / (Y.1 - X.1)
  slope_AB = slope_XY → k = -6 := by
sorry

end NUMINAMATH_CALUDE_parallel_segments_l2018_201800


namespace NUMINAMATH_CALUDE_min_max_pairwise_sum_l2018_201826

/-- Given positive integers a, b, c, d, and e that sum to 2020,
    the minimum value of the maximum of their pairwise sums is 1010. -/
theorem min_max_pairwise_sum :
  ∀ a b c d e : ℕ+,
  a + b + c + d + e = 2020 →
  1010 ≤ max (a + b) (max (b + c) (max (c + d) (d + e))) ∧
  ∃ a b c d e : ℕ+,
    a + b + c + d + e = 2020 ∧
    max (a + b) (max (b + c) (max (c + d) (d + e))) = 1010 :=
by sorry

end NUMINAMATH_CALUDE_min_max_pairwise_sum_l2018_201826


namespace NUMINAMATH_CALUDE_products_produced_is_twenty_l2018_201865

/-- Calculates the number of products produced given fixed cost, marginal cost, and total cost. -/
def products_produced (fixed_cost marginal_cost total_cost : ℚ) : ℚ :=
  (total_cost - fixed_cost) / marginal_cost

/-- Theorem stating that the number of products produced is 20 given the specified costs. -/
theorem products_produced_is_twenty :
  products_produced 12000 200 16000 = 20 := by
  sorry

#eval products_produced 12000 200 16000

end NUMINAMATH_CALUDE_products_produced_is_twenty_l2018_201865


namespace NUMINAMATH_CALUDE_toothpick_pattern_l2018_201833

/-- 
Given an arithmetic sequence where:
- The first term is 5
- The common difference is 4
Prove that the 250th term is 1001
-/
theorem toothpick_pattern (n : ℕ) (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) :
  n = 250 → a₁ = 5 → d = 4 → aₙ = a₁ + (n - 1) * d → aₙ = 1001 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_pattern_l2018_201833


namespace NUMINAMATH_CALUDE_target_probability_l2018_201841

def prob_A : ℚ := 2/3
def prob_B : ℚ := 1/2
def num_shots : ℕ := 4

theorem target_probability : 
  let prob_A_2 := (num_shots.choose 2) * prob_A^2 * (1 - prob_A)^2
  let prob_B_3 := (num_shots.choose 3) * prob_B^3 * (1 - prob_B)
  prob_A_2 * prob_B_3 = 2/27 := by sorry

end NUMINAMATH_CALUDE_target_probability_l2018_201841


namespace NUMINAMATH_CALUDE_smallest_inverse_undefined_twenty_two_satisfies_smallest_inverse_undefined_is_22_l2018_201829

theorem smallest_inverse_undefined (a : ℕ) : a > 0 ∧ 
  (∀ x : ℕ, x * a % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * a % 77 ≠ 1) → 
  a ≥ 22 := by
sorry

theorem twenty_two_satisfies : 
  (∀ x : ℕ, x * 22 % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * 22 % 77 ≠ 1) := by
sorry

theorem smallest_inverse_undefined_is_22 : 
  ∃! a : ℕ, a > 0 ∧ 
  (∀ x : ℕ, x * a % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * a % 77 ≠ 1) ∧ 
  ∀ b : ℕ, b > 0 ∧ 
  (∀ x : ℕ, x * b % 36 ≠ 1) ∧ 
  (∀ y : ℕ, y * b % 77 ≠ 1) → 
  a ≤ b := by
sorry

end NUMINAMATH_CALUDE_smallest_inverse_undefined_twenty_two_satisfies_smallest_inverse_undefined_is_22_l2018_201829


namespace NUMINAMATH_CALUDE_relationship_between_x_and_z_l2018_201871

theorem relationship_between_x_and_z (x y z : ℝ) 
  (h1 : x = 1.027 * y) 
  (h2 : y = 0.45 * z) : 
  x = 0.46215 * z := by
sorry

end NUMINAMATH_CALUDE_relationship_between_x_and_z_l2018_201871


namespace NUMINAMATH_CALUDE_less_than_minus_l2018_201881

theorem less_than_minus (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a < a - b := by
  sorry

end NUMINAMATH_CALUDE_less_than_minus_l2018_201881


namespace NUMINAMATH_CALUDE_interior_angle_regular_octagon_l2018_201880

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ) (sum_interior_angles : ℝ) (interior_angle : ℝ),
  n = 8 →
  sum_interior_angles = (n - 2) * 180 →
  interior_angle = sum_interior_angles / n →
  interior_angle = 135 := by
sorry

end NUMINAMATH_CALUDE_interior_angle_regular_octagon_l2018_201880


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2018_201839

theorem quadratic_root_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ s₁ s₂ : ℝ, (s₁ + s₂ = -c ∧ s₁ * s₂ = a) ∧
               (3*s₁ + 3*s₂ = -a ∧ 9*s₁*s₂ = b)) →
  b / c = 27 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2018_201839


namespace NUMINAMATH_CALUDE_negation_equivalence_l2018_201897

theorem negation_equivalence : 
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2018_201897


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l2018_201811

theorem smallest_number_with_remainders : ∃ (n : ℕ), n > 0 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  (∀ m : ℕ, m > 0 ∧
    m % 4 = 3 ∧
    m % 5 = 4 ∧
    m % 6 = 5 ∧
    m % 7 = 6 ∧
    m % 8 = 7 ∧
    m % 9 = 8 → m ≥ n) ∧
  n = 2519 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l2018_201811


namespace NUMINAMATH_CALUDE_alex_speed_l2018_201887

/-- Given the running speeds of Rick, Jen, Mark, and Alex, prove Alex's speed -/
theorem alex_speed (rick_speed : ℚ) (jen_ratio : ℚ) (mark_ratio : ℚ) (alex_ratio : ℚ)
  (h1 : rick_speed = 5)
  (h2 : jen_ratio = 3 / 4)
  (h3 : mark_ratio = 4 / 3)
  (h4 : alex_ratio = 5 / 6) :
  alex_ratio * mark_ratio * jen_ratio * rick_speed = 25 / 6 := by
  sorry

end NUMINAMATH_CALUDE_alex_speed_l2018_201887


namespace NUMINAMATH_CALUDE_pair_operation_result_l2018_201813

-- Define the equality for pairs of real numbers
def pair_eq (a b c d : ℝ) : Prop := a = c ∧ b = d

-- Define the "Ä" operation
def op_triangle (a b c d : ℝ) : ℝ × ℝ := (a*c + b*d, b*c - a*d)

-- Define the "Å" operation
def op_pentagon (a b c d : ℝ) : ℝ × ℝ := (a + c, b + d)

-- State the theorem
theorem pair_operation_result (x y : ℝ) :
  op_triangle 3 4 x y = (11, -2) →
  op_pentagon 3 4 x y = (4, 6) := by
  sorry

end NUMINAMATH_CALUDE_pair_operation_result_l2018_201813


namespace NUMINAMATH_CALUDE_solve_equation_l2018_201882

theorem solve_equation (x : ℚ) (h : (1 / 4 : ℚ) - (1 / 6 : ℚ) = 4 / x) : x = 48 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2018_201882


namespace NUMINAMATH_CALUDE_original_number_proof_l2018_201874

theorem original_number_proof (x : ℝ) : 3 * (2 * x + 9) = 57 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2018_201874


namespace NUMINAMATH_CALUDE_mary_towel_count_l2018_201889

/-- Proves that Mary has 4 towels given the conditions of the problem --/
theorem mary_towel_count :
  ∀ (mary_towel_count frances_towel_count : ℕ)
    (total_weight mary_towel_weight frances_towel_weight : ℚ),
  mary_towel_count = 4 * frances_towel_count →
  total_weight = 60 →
  frances_towel_weight = 128 / 16 →
  total_weight = mary_towel_weight + frances_towel_weight →
  mary_towel_weight = mary_towel_count * (frances_towel_weight / frances_towel_count) →
  mary_towel_count = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_mary_towel_count_l2018_201889


namespace NUMINAMATH_CALUDE_broken_flagpole_l2018_201895

theorem broken_flagpole (h : ℝ) (d : ℝ) (x : ℝ) : 
  h = 6 → d = 2 → x * x + d * d = (h - x) * (h - x) → x = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_broken_flagpole_l2018_201895


namespace NUMINAMATH_CALUDE_special_sum_equals_250_l2018_201852

/-- The sum of two arithmetic sequences with 5 terms each, where the first sequence starts at 3 and increases by 10, and the second sequence starts at 7 and increases by 10 -/
def special_sum : ℕ := (3+13+23+33+43)+(7+17+27+37+47)

/-- Theorem stating that the special sum equals 250 -/
theorem special_sum_equals_250 : special_sum = 250 := by
  sorry

end NUMINAMATH_CALUDE_special_sum_equals_250_l2018_201852


namespace NUMINAMATH_CALUDE_centipede_sock_shoe_orders_l2018_201821

/-- Represents the number of legs of the centipede -/
def num_legs : ℕ := 10

/-- Represents the total number of items (socks and shoes) -/
def total_items : ℕ := 2 * num_legs

/-- Represents the number of valid orders to put on socks and shoes -/
def valid_orders : ℕ := Nat.factorial total_items / (2^num_legs)

/-- Theorem stating the number of valid orders for the centipede to put on socks and shoes -/
theorem centipede_sock_shoe_orders :
  valid_orders = Nat.factorial total_items / (2^num_legs) :=
by sorry

end NUMINAMATH_CALUDE_centipede_sock_shoe_orders_l2018_201821


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l2018_201805

theorem sqrt_sum_fractions : Real.sqrt ((1 : ℝ) / 8 + 1 / 18) = Real.sqrt 26 / 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l2018_201805


namespace NUMINAMATH_CALUDE_absolute_value_and_exponents_calculation_l2018_201893

theorem absolute_value_and_exponents_calculation : 
  |(-5 : ℝ)| + (1/3)⁻¹ - (π - 2)^0 = 7 := by sorry

end NUMINAMATH_CALUDE_absolute_value_and_exponents_calculation_l2018_201893


namespace NUMINAMATH_CALUDE_garden_length_l2018_201868

/-- Proves that a rectangular garden with length twice its width and perimeter 300 yards has a length of 100 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width →  -- Length is twice the width
  2 * length + 2 * width = 300 →  -- Perimeter is 300 yards
  length = 100 := by  -- Prove that length is 100 yards
sorry

end NUMINAMATH_CALUDE_garden_length_l2018_201868


namespace NUMINAMATH_CALUDE_song_ratio_is_two_to_one_l2018_201898

/-- Represents the number of songs on Aisha's mp3 player at different stages --/
structure SongCount where
  initial : ℕ
  afterWeek : ℕ
  added : ℕ
  removed : ℕ
  final : ℕ

/-- Calculates the ratio of added songs to songs after the first two weeks --/
def songRatio (s : SongCount) : ℚ :=
  s.added / (s.initial + s.afterWeek)

/-- Theorem stating the ratio of added songs to songs after the first two weeks --/
theorem song_ratio_is_two_to_one (s : SongCount)
  (h1 : s.initial = 500)
  (h2 : s.afterWeek = 500)
  (h3 : s.removed = 50)
  (h4 : s.final = 2950)
  (h5 : s.initial + s.afterWeek + s.added - s.removed = s.final) :
  songRatio s = 2 := by
  sorry

#check song_ratio_is_two_to_one

end NUMINAMATH_CALUDE_song_ratio_is_two_to_one_l2018_201898


namespace NUMINAMATH_CALUDE_number_equation_l2018_201872

theorem number_equation (x : ℝ) : 3 * x = (26 - x) + 10 ↔ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l2018_201872


namespace NUMINAMATH_CALUDE_jessa_cupcakes_l2018_201810

/-- The number of cupcakes needed for a given number of classes and students per class -/
def cupcakes_needed (num_classes : ℕ) (students_per_class : ℕ) : ℕ :=
  num_classes * students_per_class

theorem jessa_cupcakes : 
  let fourth_grade_cupcakes := cupcakes_needed 3 30
  let pe_class_cupcakes := cupcakes_needed 1 50
  fourth_grade_cupcakes + pe_class_cupcakes = 140 := by
  sorry

end NUMINAMATH_CALUDE_jessa_cupcakes_l2018_201810


namespace NUMINAMATH_CALUDE_ned_friend_games_l2018_201806

/-- The number of games Ned bought from his friend -/
def games_from_friend : ℕ := 50

/-- The number of games Ned bought at the garage sale -/
def garage_sale_games : ℕ := 27

/-- The number of games that didn't work -/
def non_working_games : ℕ := 74

/-- The number of good games Ned ended up with -/
def good_games : ℕ := 3

/-- Theorem stating that the number of games Ned bought from his friend is 50 -/
theorem ned_friend_games : 
  games_from_friend = 50 ∧
  games_from_friend + garage_sale_games = non_working_games + good_games :=
by sorry

end NUMINAMATH_CALUDE_ned_friend_games_l2018_201806


namespace NUMINAMATH_CALUDE_remainder_of_five_n_mod_eleven_l2018_201855

theorem remainder_of_five_n_mod_eleven (n : ℤ) (h : n % 11 = 1) : (5 * n) % 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_five_n_mod_eleven_l2018_201855


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l2018_201820

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 30 →
  c^2 = a^2 + b^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l2018_201820


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l2018_201815

theorem sum_of_x_and_y_equals_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 20) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l2018_201815


namespace NUMINAMATH_CALUDE_ab_100_necessary_not_sufficient_for_log_sum_2_l2018_201803

theorem ab_100_necessary_not_sufficient_for_log_sum_2 :
  (∀ a b : ℝ, (Real.log a + Real.log b = 2) → (a * b = 100)) ∧
  (∃ a b : ℝ, a * b = 100 ∧ Real.log a + Real.log b ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_ab_100_necessary_not_sufficient_for_log_sum_2_l2018_201803


namespace NUMINAMATH_CALUDE_sticker_count_l2018_201828

/-- Given a number of stickers per page and a number of pages, 
    calculate the total number of stickers -/
def total_stickers (stickers_per_page : ℕ) (num_pages : ℕ) : ℕ :=
  stickers_per_page * num_pages

/-- Theorem: The total number of stickers is 220 when there are 10 stickers per page and 22 pages -/
theorem sticker_count : total_stickers 10 22 = 220 := by
  sorry

end NUMINAMATH_CALUDE_sticker_count_l2018_201828


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l2018_201830

theorem reciprocal_of_negative_two :
  (∃ x : ℚ, -2 * x = 1) ∧ (∀ x : ℚ, -2 * x = 1 → x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l2018_201830


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l2018_201851

theorem quadratic_equation_real_roots (k : ℝ) : 
  k > 0 → ∃ x : ℝ, x^2 + 2*x - k = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l2018_201851


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2018_201886

theorem quadratic_factorization (a b c : ℤ) : 
  (∃ b c : ℤ, ∀ x : ℤ, (x - a) * (x - 6) + 1 = (x + b) * (x + c)) ↔ a = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2018_201886


namespace NUMINAMATH_CALUDE_sampling_more_suitable_for_large_population_l2018_201801

/-- Represents a survey method -/
inductive SurveyMethod
  | Comprehensive
  | Sampling

/-- Represents the characteristics of a survey -/
structure SurveyCharacteristics where
  populationSize : ℕ
  isSurveyingLargePopulation : Bool

/-- Determines the most suitable survey method based on survey characteristics -/
def mostSuitableSurveyMethod (characteristics : SurveyCharacteristics) : SurveyMethod :=
  if characteristics.isSurveyingLargePopulation then
    SurveyMethod.Sampling
  else
    SurveyMethod.Comprehensive

/-- Theorem: For a large population survey, sampling is more suitable than comprehensive -/
theorem sampling_more_suitable_for_large_population 
  (characteristics : SurveyCharacteristics) 
  (h : characteristics.isSurveyingLargePopulation = true) : 
  mostSuitableSurveyMethod characteristics = SurveyMethod.Sampling :=
by
  sorry

end NUMINAMATH_CALUDE_sampling_more_suitable_for_large_population_l2018_201801


namespace NUMINAMATH_CALUDE_room_tile_coverage_l2018_201848

-- Define the room dimensions
def room_length : ℕ := 12
def room_width : ℕ := 20

-- Define the number of tiles
def num_tiles : ℕ := 40

-- Define the size of each tile
def tile_size : ℕ := 1

-- Theorem to prove
theorem room_tile_coverage : 
  (num_tiles : ℚ) / (room_length * room_width) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_room_tile_coverage_l2018_201848


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2018_201878

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (2 - Complex.I) * (1 + 2 * Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2018_201878


namespace NUMINAMATH_CALUDE_M_intersect_N_l2018_201888

def M : Set ℝ := {x | -2 ≤ x - 1 ∧ x - 1 ≤ 2}

def N : Set ℝ := {x | ∃ k : ℕ+, x = 2 * k - 1}

theorem M_intersect_N : M ∩ N = {1, 3} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l2018_201888


namespace NUMINAMATH_CALUDE_derivative_y_x_l2018_201825

noncomputable def x (t : ℝ) : ℝ := Real.log (1 / Real.tan t)
noncomputable def y (t : ℝ) : ℝ := 1 / (Real.cos t)^2

theorem derivative_y_x (t : ℝ) (h : Real.cos t ≠ 0) (h' : Real.sin t ≠ 0) :
  deriv y t / deriv x t = -2 * (Real.tan t)^2 :=
by sorry

end NUMINAMATH_CALUDE_derivative_y_x_l2018_201825


namespace NUMINAMATH_CALUDE_jogging_ninth_day_l2018_201883

def minutes_jogged_6_days : ℕ := 6 * 80
def minutes_jogged_2_days : ℕ := 2 * 105
def total_minutes_8_days : ℕ := minutes_jogged_6_days + minutes_jogged_2_days
def desired_average : ℕ := 100
def total_days : ℕ := 9

theorem jogging_ninth_day :
  desired_average * total_days - total_minutes_8_days = 210 := by
  sorry

end NUMINAMATH_CALUDE_jogging_ninth_day_l2018_201883


namespace NUMINAMATH_CALUDE_smallest_n_for_cube_assembly_l2018_201838

theorem smallest_n_for_cube_assembly (n : ℕ) : 
  (∀ m : ℕ, m < n → m^3 < (2*m)^3 - (2*m - 2)^3) ∧ 
  n^3 ≥ (2*n)^3 - (2*n - 2)^3 → 
  n = 23 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_cube_assembly_l2018_201838


namespace NUMINAMATH_CALUDE_water_tank_theorem_l2018_201846

/-- Represents the water tank problem --/
def WaterTankProblem (maxCapacity initialLossRate initialLossDuration
                      secondaryLossRate secondaryLossDuration
                      refillRate refillDuration : ℕ) : Prop :=
  let initialLoss := initialLossRate * initialLossDuration
  let secondaryLoss := secondaryLossRate * secondaryLossDuration
  let totalLoss := initialLoss + secondaryLoss
  let remainingWater := maxCapacity - totalLoss
  let refillAmount := refillRate * refillDuration
  let finalWaterAmount := remainingWater + refillAmount
  maxCapacity - finalWaterAmount = 140000

/-- The water tank theorem --/
theorem water_tank_theorem : WaterTankProblem 350000 32000 5 10000 10 40000 3 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_theorem_l2018_201846


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2018_201890

theorem exponent_multiplication (a : ℝ) : a^4 * a^3 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2018_201890


namespace NUMINAMATH_CALUDE_division_problem_l2018_201879

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 689 →
  divisor = 36 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  quotient = 19 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2018_201879


namespace NUMINAMATH_CALUDE_simplify_expression_l2018_201873

theorem simplify_expression (x y : ℝ) : 7*x + 3*y + 4 - 2*x + 9 + 5*y = 5*x + 8*y + 13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2018_201873


namespace NUMINAMATH_CALUDE_triangle_properties_l2018_201817

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.cos t.A * Real.cos t.C + 1 = 2 * Real.sin t.A * Real.sin t.C)
  (h2 : t.a + t.c = 3 * Real.sqrt 3 / 2)
  (h3 : t.b = Real.sqrt 3) :
  t.B = π / 3 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B) = 5 * Real.sqrt 3 / 16 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l2018_201817


namespace NUMINAMATH_CALUDE_harry_travel_time_l2018_201819

theorem harry_travel_time (initial_bus_time remaining_bus_time : ℕ) 
  (h1 : initial_bus_time = 15)
  (h2 : remaining_bus_time = 25) : 
  let total_bus_time := initial_bus_time + remaining_bus_time
  let walking_time := total_bus_time / 2
  initial_bus_time + remaining_bus_time + walking_time = 60 := by
sorry

end NUMINAMATH_CALUDE_harry_travel_time_l2018_201819


namespace NUMINAMATH_CALUDE_rectangles_on_4x3_grid_l2018_201862

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of rectangles that can be formed on a grid. -/
def rectangles_on_grid (columns rows : ℕ) : ℕ :=
  binomial columns 2 * binomial rows 2

/-- Theorem: The number of rectangles on a 4x3 grid is 18. -/
theorem rectangles_on_4x3_grid : rectangles_on_grid 4 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_on_4x3_grid_l2018_201862


namespace NUMINAMATH_CALUDE_permutation_fraction_equality_l2018_201850

def A (n m : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial (n - m))

theorem permutation_fraction_equality : 
  (2 * A 8 5 + 7 * A 8 4) / (A 8 8 - A 9 5) = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_permutation_fraction_equality_l2018_201850


namespace NUMINAMATH_CALUDE_problem_statement_l2018_201884

def prop_p (m : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, 2 * x - 2 ≥ m^2 - 3 * m

def prop_q (m : ℝ) (a : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (-1) 1, m ≤ a * x

theorem problem_statement (m : ℝ) :
  (prop_p m → m ∈ Set.Icc 1 2) ∧
  (¬(prop_p m ∧ prop_q m 1) ∧ (prop_p m ∨ prop_q m 1) →
    m < 1 ∨ (1 < m ∧ m ≤ 2)) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2018_201884


namespace NUMINAMATH_CALUDE_inequality_proof_l2018_201864

theorem inequality_proof (x : ℝ) (h1 : x > 4/3) (h2 : x ≠ -5) (h3 : x ≠ 4/3) :
  (6*x^2 + 18*x - 60) / ((3*x - 4)*(x + 5)) < 2 :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2018_201864


namespace NUMINAMATH_CALUDE_square_cut_from_rectangle_l2018_201832

theorem square_cut_from_rectangle (a b x : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0) (h4 : x ≤ min a b) :
  (2 * (a + b) + 2 * x = a * b) ∧ (a * b - x^2 = 2 * (a + b)) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_cut_from_rectangle_l2018_201832


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l2018_201885

/-- The area of a circle with circumference 31.4 meters is 246.49/π square meters -/
theorem circle_area_from_circumference :
  let circumference : ℝ := 31.4
  let radius : ℝ := circumference / (2 * Real.pi)
  let area : ℝ := Real.pi * radius^2
  area = 246.49 / Real.pi := by sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l2018_201885


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l2018_201858

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, 2 * a * y - 1 = 0 ↔ (3 * a - 1) * x + y - 1 = 0) → 
  a = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l2018_201858


namespace NUMINAMATH_CALUDE_james_vegetable_consumption_l2018_201822

/-- Represents James' vegetable consumption --/
structure VegetableConsumption where
  asparagus : Real
  broccoli : Real
  kale : Real

/-- Calculates the total weekly consumption given daily consumption of asparagus and broccoli --/
def weekly_consumption (daily : VegetableConsumption) : Real :=
  (daily.asparagus + daily.broccoli) * 7

/-- James' initial daily consumption --/
def initial_daily : VegetableConsumption :=
  { asparagus := 0.25, broccoli := 0.25, kale := 0 }

/-- James' consumption after doubling asparagus and broccoli and adding kale --/
def final_weekly : VegetableConsumption :=
  { asparagus := initial_daily.asparagus * 2 * 7,
    broccoli := initial_daily.broccoli * 2 * 7,
    kale := 3 }

/-- Theorem stating James' final weekly vegetable consumption --/
theorem james_vegetable_consumption :
  final_weekly.asparagus + final_weekly.broccoli + final_weekly.kale = 10 := by
  sorry


end NUMINAMATH_CALUDE_james_vegetable_consumption_l2018_201822


namespace NUMINAMATH_CALUDE_rectangle_rotation_path_length_l2018_201870

/-- The length of the path traveled by point A in a rectangle ABCD after three 90° rotations -/
theorem rectangle_rotation_path_length (AB BC : ℝ) (h1 : AB = 3) (h2 : BC = 5) : 
  let diagonal := Real.sqrt (AB^2 + BC^2)
  let single_rotation_arc := π * diagonal / 2
  3 * single_rotation_arc = (3 * π * Real.sqrt 34) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_rotation_path_length_l2018_201870


namespace NUMINAMATH_CALUDE_negative_integer_squared_plus_self_equals_twelve_l2018_201863

theorem negative_integer_squared_plus_self_equals_twelve (N : ℤ) : 
  N < 0 → N^2 + N = 12 → N = -4 := by sorry

end NUMINAMATH_CALUDE_negative_integer_squared_plus_self_equals_twelve_l2018_201863


namespace NUMINAMATH_CALUDE_triangle_max_value_l2018_201876

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where the area is √3 and cos(C) / cos(B) = c / (2a - b),
    the maximum value of 1/(b+1) + 9/(a+9) is 3/5. -/
theorem triangle_max_value (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 →
  Real.cos C / Real.cos B = c / (2*a - b) →
  (∃ (x : ℝ), (1/(b+1) + 9/(a+9) ≤ x) ∧ 
   (∀ (y : ℝ), 1/(b+1) + 9/(a+9) ≤ y → x ≤ y)) →
  (1/(b+1) + 9/(a+9)) ≤ 3/5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_value_l2018_201876


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_with_8_and_0_l2018_201867

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 8 ∨ d = 0

def count_digit (n : ℕ) (d : ℕ) : ℕ :=
  (n.digits 10).filter (· = d) |>.length

theorem largest_multiple_of_15_with_8_and_0 :
  ∃ m : ℕ,
    m > 0 ∧
    15 ∣ m ∧
    is_valid_number m ∧
    count_digit m 8 = 6 ∧
    count_digit m 0 = 1 ∧
    m / 15 = 592592 ∧
    ∀ n : ℕ, n > m → ¬(15 ∣ n ∧ is_valid_number n) :=
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_with_8_and_0_l2018_201867


namespace NUMINAMATH_CALUDE_equation_solutions_l2018_201854

def is_solution (x y : ℤ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 7

theorem equation_solutions :
  ∃! (s : Finset (ℤ × ℤ)), s.card = 5 ∧ ∀ (p : ℤ × ℤ), p ∈ s ↔ is_solution p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l2018_201854


namespace NUMINAMATH_CALUDE_beads_per_necklace_is_20_l2018_201818

/-- The number of beads needed to make one necklace -/
def beads_per_necklace : ℕ := sorry

/-- The number of necklaces made on Monday -/
def monday_necklaces : ℕ := 10

/-- The number of necklaces made on Tuesday -/
def tuesday_necklaces : ℕ := 2

/-- The number of bracelets made on Wednesday -/
def wednesday_bracelets : ℕ := 5

/-- The number of earrings made on Wednesday -/
def wednesday_earrings : ℕ := 7

/-- The number of beads needed for one bracelet -/
def beads_per_bracelet : ℕ := 10

/-- The number of beads needed for one earring -/
def beads_per_earring : ℕ := 5

/-- The total number of beads used -/
def total_beads : ℕ := 325

theorem beads_per_necklace_is_20 : 
  beads_per_necklace = 20 :=
by
  have h1 : monday_necklaces * beads_per_necklace + 
            tuesday_necklaces * beads_per_necklace + 
            wednesday_bracelets * beads_per_bracelet + 
            wednesday_earrings * beads_per_earring = total_beads := by sorry
  sorry

end NUMINAMATH_CALUDE_beads_per_necklace_is_20_l2018_201818


namespace NUMINAMATH_CALUDE_find_number_l2018_201836

theorem find_number : ∃ x : ℤ, x - 27 = 49 ∧ x = 76 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2018_201836


namespace NUMINAMATH_CALUDE_g_75_solutions_l2018_201894

-- Define g₀
def g₀ (x : ℝ) : ℝ := x + |x - 150| - |x + 150|

-- Define gₙ recursively
def g (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

-- Theorem statement
theorem g_75_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, g 75 x = 0) ∧
                    (∀ x ∉ S, g 75 x ≠ 0) ∧
                    Finset.card S = 4 := by
  sorry

end NUMINAMATH_CALUDE_g_75_solutions_l2018_201894


namespace NUMINAMATH_CALUDE_volume_is_360_l2018_201869

/-- A rectangular parallelepiped with edge lengths 4, 6, and 15 -/
structure RectangularParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  length_eq : length = 4
  width_eq : width = 6
  height_eq : height = 15

/-- The volume of a rectangular parallelepiped -/
def volume (rp : RectangularParallelepiped) : ℝ :=
  rp.length * rp.width * rp.height

/-- Theorem: The volume of the given rectangular parallelepiped is 360 cubic units -/
theorem volume_is_360 (rp : RectangularParallelepiped) : volume rp = 360 := by
  sorry

end NUMINAMATH_CALUDE_volume_is_360_l2018_201869


namespace NUMINAMATH_CALUDE_cubic_expansion_equals_cube_problem_solution_l2018_201875

theorem cubic_expansion_equals_cube (n : ℕ) : n^3 + 3*(n^2) + 3*n + 1 = (n + 1)^3 := by sorry

theorem problem_solution : 98^3 + 3*(98^2) + 3*98 + 1 = 970299 := by sorry

end NUMINAMATH_CALUDE_cubic_expansion_equals_cube_problem_solution_l2018_201875


namespace NUMINAMATH_CALUDE_gcd_and_sum_of_digits_l2018_201891

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem gcd_and_sum_of_digits : 
  let diff1 := [18330 - 3510, 23790 - 3510, 23790 - 18330]
  let diff2 := [14660 - 5680, 19050 - 5680, 19050 - 14660]
  let gcd1 := Nat.gcd (Nat.gcd (diff1.get! 0) (diff1.get! 1)) (diff1.get! 2)
  let gcd2 := Nat.gcd (Nat.gcd (diff2.get! 0) (diff2.get! 1)) (diff2.get! 2)
  let n := Nat.gcd gcd1 gcd2
  n = 130 ∧ sum_of_digits n = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_and_sum_of_digits_l2018_201891


namespace NUMINAMATH_CALUDE_angle_measure_l2018_201849

/-- The measure of an angle in degrees, given that its supplement is four times its complement. -/
theorem angle_measure : ∃ x : ℝ, 
  (0 < x) ∧ (x < 180) ∧ 
  (180 - x = 4 * (90 - x)) ∧ 
  (x = 60) := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l2018_201849


namespace NUMINAMATH_CALUDE_water_experiment_result_l2018_201809

/-- Calculates the remaining water after an experiment and addition. -/
def remaining_water (initial : ℚ) (used : ℚ) (added : ℚ) : ℚ :=
  initial - used + added

/-- Proves that given the specific amounts in the problem, the remaining water is 13/6 gallons. -/
theorem water_experiment_result :
  remaining_water 3 (4/3) (1/2) = 13/6 := by
  sorry

end NUMINAMATH_CALUDE_water_experiment_result_l2018_201809


namespace NUMINAMATH_CALUDE_total_pens_count_l2018_201866

/-- The number of black pens bought by the teacher -/
def black_pens : ℕ := 7

/-- The number of blue pens bought by the teacher -/
def blue_pens : ℕ := 9

/-- The number of red pens bought by the teacher -/
def red_pens : ℕ := 5

/-- The total number of pens bought by the teacher -/
def total_pens : ℕ := black_pens + blue_pens + red_pens

theorem total_pens_count : total_pens = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_pens_count_l2018_201866


namespace NUMINAMATH_CALUDE_complex_number_modulus_l2018_201856

theorem complex_number_modulus (i : ℂ) (h : i * i = -1) :
  let z : ℂ := i * (1 - i)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l2018_201856


namespace NUMINAMATH_CALUDE_final_price_is_91_percent_l2018_201802

/-- Represents the price increase factor -/
def price_increase : ℝ := 1.4

/-- Represents the discount factor -/
def discount : ℝ := 0.65

/-- Theorem stating that the final price after increase and discount is 91% of the original price -/
theorem final_price_is_91_percent (original_price : ℝ) :
  discount * (price_increase * original_price) = 0.91 * original_price := by
  sorry

end NUMINAMATH_CALUDE_final_price_is_91_percent_l2018_201802


namespace NUMINAMATH_CALUDE_integral_roots_system_l2018_201840

theorem integral_roots_system : ∃! (x y z : ℕ),
  (z^x = y^(3*x)) ∧
  (2^z = 8 * 4^x) ∧
  (x + y + z = 18) ∧
  x = 6 ∧ y = 2 ∧ z = 15 := by sorry

end NUMINAMATH_CALUDE_integral_roots_system_l2018_201840
