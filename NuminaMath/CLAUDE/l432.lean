import Mathlib

namespace NUMINAMATH_CALUDE_cubic_equation_solution_l432_43229

theorem cubic_equation_solution : 
  ∃! x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ∧ x^2 + 4*x + 4 > 0 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l432_43229


namespace NUMINAMATH_CALUDE_exactly_one_of_each_survives_l432_43220

-- Define the number of trees of each type
def num_trees_A : ℕ := 2
def num_trees_B : ℕ := 2

-- Define the survival rates
def survival_rate_A : ℚ := 2/3
def survival_rate_B : ℚ := 1/2

-- Define the probability of exactly one tree of type A surviving
def prob_one_A_survives : ℚ := 
  (num_trees_A.choose 1 : ℚ) * survival_rate_A * (1 - survival_rate_A)

-- Define the probability of exactly one tree of type B surviving
def prob_one_B_survives : ℚ := 
  (num_trees_B.choose 1 : ℚ) * survival_rate_B * (1 - survival_rate_B)

-- State the theorem
theorem exactly_one_of_each_survives : 
  prob_one_A_survives * prob_one_B_survives = 2/9 := by sorry

end NUMINAMATH_CALUDE_exactly_one_of_each_survives_l432_43220


namespace NUMINAMATH_CALUDE_knights_on_red_chairs_l432_43236

/-- Represents the type of chair occupant -/
inductive Occupant
| Knight
| Liar

/-- Represents the color of a chair -/
inductive ChairColor
| Blue
| Red

/-- Represents the state of the room -/
structure RoomState where
  totalChairs : ℕ
  knights : ℕ
  liars : ℕ
  knightsOnRed : ℕ
  liarsOnBlue : ℕ

/-- The initial state of the room -/
def initialState : RoomState :=
  { totalChairs := 20
  , knights := 20 - (20 : ℕ) / 2  -- Arbitrary split between knights and liars
  , liars := (20 : ℕ) / 2
  , knightsOnRed := 0
  , liarsOnBlue := 0 }

/-- The state of the room after rearrangement -/
def finalState (initial : RoomState) : RoomState :=
  { totalChairs := initial.totalChairs
  , knights := initial.knights
  , liars := initial.liars
  , knightsOnRed := (initial.totalChairs : ℕ) / 2 - initial.liars
  , liarsOnBlue := (initial.totalChairs : ℕ) / 2 - (initial.knights - ((initial.totalChairs : ℕ) / 2 - initial.liars)) }

theorem knights_on_red_chairs (initial : RoomState) :
  (finalState initial).knightsOnRed = 5 := by
  sorry

end NUMINAMATH_CALUDE_knights_on_red_chairs_l432_43236


namespace NUMINAMATH_CALUDE_three_circles_cover_horizon_two_circles_cannot_cover_horizon_l432_43206

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_positive : radius > 0

-- Define a point in 2D space
def Point : Type := ℝ × ℝ

-- Define a ray emanating from a point
structure Ray where
  origin : Point
  direction : ℝ × ℝ
  direction_nonzero : direction ≠ (0, 0)

-- Function to check if two circles are non-overlapping and non-touching
def non_overlapping_non_touching (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let distance := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance > c1.radius + c2.radius

-- Function to check if a point is outside a circle
def point_outside_circle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 > c.radius^2

-- Function to check if a ray intersects a circle
def ray_intersects_circle (r : Ray) (c : Circle) : Prop :=
  sorry  -- The actual implementation would go here

-- Theorem for three circles covering the horizon
theorem three_circles_cover_horizon :
  ∃ (c1 c2 c3 : Circle) (p : Point),
    non_overlapping_non_touching c1 c2 ∧
    non_overlapping_non_touching c1 c3 ∧
    non_overlapping_non_touching c2 c3 ∧
    point_outside_circle p c1 ∧
    point_outside_circle p c2 ∧
    point_outside_circle p c3 ∧
    ∀ (r : Ray), r.origin = p →
      ray_intersects_circle r c1 ∨
      ray_intersects_circle r c2 ∨
      ray_intersects_circle r c3 :=
  sorry

-- Theorem for two circles not covering the horizon
theorem two_circles_cannot_cover_horizon :
  ¬ ∃ (c1 c2 : Circle) (p : Point),
    non_overlapping_non_touching c1 c2 ∧
    point_outside_circle p c1 ∧
    point_outside_circle p c2 ∧
    ∀ (r : Ray), r.origin = p →
      ray_intersects_circle r c1 ∨
      ray_intersects_circle r c2 :=
  sorry

end NUMINAMATH_CALUDE_three_circles_cover_horizon_two_circles_cannot_cover_horizon_l432_43206


namespace NUMINAMATH_CALUDE_find_number_l432_43288

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem find_number (xy : ℕ) (h1 : is_two_digit xy) 
  (h2 : (xy / 10) + (xy % 10) = 8)
  (h3 : reverse_digits xy - xy = 18) : xy = 35 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l432_43288


namespace NUMINAMATH_CALUDE_planet_combinations_correct_l432_43269

/-- The number of different combinations of planets that can be occupied. -/
def planetCombinations : ℕ :=
  let earthLike := 7
  let marsLike := 8
  let earthUnits := 3
  let marsUnits := 1
  let totalUnits := 21
  2941

/-- Theorem stating that the number of planet combinations is correct. -/
theorem planet_combinations_correct : planetCombinations = 2941 := by
  sorry

end NUMINAMATH_CALUDE_planet_combinations_correct_l432_43269


namespace NUMINAMATH_CALUDE_abhays_speed_l432_43265

/-- Proves that Abhay's speed is 10.5 km/h given the problem conditions --/
theorem abhays_speed (distance : ℝ) (abhay_speed sameer_speed : ℝ) 
  (h1 : distance = 42)
  (h2 : distance / abhay_speed = distance / sameer_speed + 2)
  (h3 : distance / (2 * abhay_speed) = distance / sameer_speed - 1) :
  abhay_speed = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_abhays_speed_l432_43265


namespace NUMINAMATH_CALUDE_product_of_four_numbers_l432_43211

theorem product_of_four_numbers (E F G H : ℝ) :
  E > 0 → F > 0 → G > 0 → H > 0 →
  E + F + G + H = 50 →
  E - 3 = F + 3 ∧ E - 3 = G * 3 ∧ E - 3 = H / 3 →
  E * F * G * H = 7461.9140625 := by
sorry

end NUMINAMATH_CALUDE_product_of_four_numbers_l432_43211


namespace NUMINAMATH_CALUDE_binomial_15_4_l432_43278

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_4_l432_43278


namespace NUMINAMATH_CALUDE_walking_distance_l432_43203

-- Define the total journey time in hours
def total_time : ℚ := 50 / 60

-- Define the speeds in km/h
def bike_speed : ℚ := 20
def walk_speed : ℚ := 4

-- Define the function to calculate the total time given a distance x
def journey_time (x : ℚ) : ℚ := x / (2 * bike_speed) + x / (2 * walk_speed)

-- State the theorem
theorem walking_distance : 
  ∃ (x : ℚ), journey_time x = total_time ∧ 
  (round (10 * (x / 2)) / 10 : ℚ) = 28 / 10 :=
sorry

end NUMINAMATH_CALUDE_walking_distance_l432_43203


namespace NUMINAMATH_CALUDE_closest_sum_to_zero_l432_43297

def S : Finset Int := {5, 19, -6, 0, -4}

theorem closest_sum_to_zero (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∀ x y z, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → 
  |a + b + c| ≤ |x + y + z| ∧ (∃ p q r, p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ |p + q + r| = 1) :=
by sorry

end NUMINAMATH_CALUDE_closest_sum_to_zero_l432_43297


namespace NUMINAMATH_CALUDE_pythagorean_triple_3_4_5_l432_43217

theorem pythagorean_triple_3_4_5 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧ a = 3 ∧ b = 4 ∧ c = 5 := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_3_4_5_l432_43217


namespace NUMINAMATH_CALUDE_probability_three_primes_l432_43258

def num_dice : ℕ := 5
def num_primes : ℕ := 3
def prob_prime : ℚ := 2/5

theorem probability_three_primes :
  (num_dice.choose num_primes : ℚ) * prob_prime^num_primes * (1 - prob_prime)^(num_dice - num_primes) = 720/3125 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_primes_l432_43258


namespace NUMINAMATH_CALUDE_four_digit_integer_problem_l432_43232

theorem four_digit_integer_problem (n : ℕ) (a b c d : ℕ) :
  n = a * 1000 + b * 100 + c * 10 + d →
  a ≥ 1 →
  a ≤ 9 →
  b ≤ 9 →
  c ≤ 9 →
  d ≤ 9 →
  a + b + c + d = 16 →
  b + c = 10 →
  a - d = 2 →
  n % 11 = 0 →
  n = 4462 := by
sorry

end NUMINAMATH_CALUDE_four_digit_integer_problem_l432_43232


namespace NUMINAMATH_CALUDE_sophie_goal_theorem_l432_43204

def sophie_marks : List ℚ := [73/100, 82/100, 85/100]

def total_tests : ℕ := 5

def goal_average : ℚ := 80/100

def pair_D : List ℚ := [73/100, 83/100]
def pair_A : List ℚ := [79/100, 82/100]
def pair_B : List ℚ := [70/100, 91/100]
def pair_C : List ℚ := [76/100, 86/100]

theorem sophie_goal_theorem :
  (sophie_marks.sum + pair_D.sum) / total_tests < goal_average ∧
  (sophie_marks.sum + pair_A.sum) / total_tests ≥ goal_average ∧
  (sophie_marks.sum + pair_B.sum) / total_tests ≥ goal_average ∧
  (sophie_marks.sum + pair_C.sum) / total_tests ≥ goal_average :=
by sorry

end NUMINAMATH_CALUDE_sophie_goal_theorem_l432_43204


namespace NUMINAMATH_CALUDE_sum_of_triangle_ops_l432_43280

/-- Operation on three numbers as defined in the problem -/
def triangle_op (a b c : ℕ) : ℕ := a * b - c

/-- Theorem stating the sum of results from two specific triangles -/
theorem sum_of_triangle_ops : 
  triangle_op 4 2 3 + triangle_op 3 5 1 = 19 := by sorry

end NUMINAMATH_CALUDE_sum_of_triangle_ops_l432_43280


namespace NUMINAMATH_CALUDE_somu_father_age_ratio_l432_43285

/-- Represents the ages of Somu and his father -/
structure Ages where
  somu : ℕ
  father : ℕ

/-- The condition that Somu's age 8 years ago was one-fifth of his father's age 8 years ago -/
def age_relation (ages : Ages) : Prop :=
  ages.somu - 8 = (ages.father - 8) / 5

/-- The theorem stating the ratio of Somu's age to his father's age -/
theorem somu_father_age_ratio :
  ∀ (ages : Ages),
  ages.somu = 16 →
  age_relation ages →
  (ages.somu : ℚ) / (ages.father : ℚ) = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_somu_father_age_ratio_l432_43285


namespace NUMINAMATH_CALUDE_three_digit_numbers_decreasing_by_factor_of_six_l432_43296

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ (a b c : ℕ), 
    n = 100 * a + 10 * b + c ∧
    a ≠ 0 ∧
    10 * b + c = (100 * a + 10 * b + c) / 6

theorem three_digit_numbers_decreasing_by_factor_of_six : 
  {n : ℕ | is_valid_number n} = {120, 240, 360, 480} := by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_decreasing_by_factor_of_six_l432_43296


namespace NUMINAMATH_CALUDE_complex_multiplication_l432_43271

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l432_43271


namespace NUMINAMATH_CALUDE_linear_function_inequality_l432_43242

theorem linear_function_inequality (a b : ℝ) (h1 : a > 0) (h2 : -2 * a + b = 0) :
  ∀ x : ℝ, a * x > b ↔ x > 2 :=
by sorry

end NUMINAMATH_CALUDE_linear_function_inequality_l432_43242


namespace NUMINAMATH_CALUDE_room_tiles_count_l432_43272

/-- Calculates the number of tiles needed for a room with given specifications -/
def calculate_tiles (room_length room_width border_width tile_size column_size : ℕ) : ℕ :=
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let border_tiles := 2 * ((room_length / tile_size) + (room_width / tile_size) - 4)
  let inner_tiles := (inner_length * inner_width) / (tile_size * tile_size)
  let column_tiles := (column_size * column_size + tile_size * tile_size - 1) / (tile_size * tile_size)
  border_tiles + inner_tiles + column_tiles

/-- Theorem stating that the number of tiles for the given room specification is 78 -/
theorem room_tiles_count : calculate_tiles 15 20 2 2 1 = 78 := by
  sorry

end NUMINAMATH_CALUDE_room_tiles_count_l432_43272


namespace NUMINAMATH_CALUDE_valid_fractions_characterization_l432_43222

def is_valid_fraction (n d : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ d ≥ 10 ∧ d < 100 ∧
  (∃ (a b c : Nat), (a > 0 ∧ b > 0 ∧ c > 0) ∧
    ((n = 10 * a + b ∧ d = 10 * a + c ∧ n * c = d * b) ∨
     (n = 10 * a + b ∧ d = 10 * c + b ∧ n * c = d * a) ∨
     (n = 10 * a + c ∧ d = 10 * b + c ∧ n * b = d * a)))

theorem valid_fractions_characterization :
  {p : Nat × Nat | is_valid_fraction p.1 p.2} =
  {(26, 65), (16, 64), (19, 95), (49, 98)} := by
  sorry

end NUMINAMATH_CALUDE_valid_fractions_characterization_l432_43222


namespace NUMINAMATH_CALUDE_problem_statement_l432_43298

theorem problem_statement (x y : ℝ) (h1 : x + 3*y = 5) (h2 : 2*x - y = 2) :
  2*x^2 + 5*x*y - 3*y^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l432_43298


namespace NUMINAMATH_CALUDE_valid_paths_count_l432_43213

/-- Represents a point in the Cartesian plane -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a move direction -/
inductive Move
  | Right
  | Up
  | Diagonal

/-- Defines a valid path on the Cartesian plane -/
def ValidPath (start finish : Point) (path : List Move) : Prop :=
  -- Path starts at the start point and ends at the finish point
  -- Each move is valid according to the problem conditions
  -- No right-angle turns in the path
  sorry

/-- Counts the number of valid paths between two points -/
def CountValidPaths (start finish : Point) : ℕ :=
  sorry

theorem valid_paths_count :
  CountValidPaths (Point.mk 0 0) (Point.mk 6 6) = 128 := by
  sorry

end NUMINAMATH_CALUDE_valid_paths_count_l432_43213


namespace NUMINAMATH_CALUDE_copper_needed_in_mixture_l432_43240

/-- Given a manufacturing mixture with specified percentages of materials,
    this theorem calculates the amount of copper needed when a certain amount of lead is used. -/
theorem copper_needed_in_mixture (total : ℝ) (cobalt_percent lead_percent copper_percent : ℝ) 
    (lead_amount : ℝ) (copper_amount : ℝ) : 
  cobalt_percent = 0.15 →
  lead_percent = 0.25 →
  copper_percent = 0.60 →
  cobalt_percent + lead_percent + copper_percent = 1 →
  lead_amount = 5 →
  total * lead_percent = lead_amount →
  copper_amount = total * copper_percent →
  copper_amount = 12 := by
sorry

end NUMINAMATH_CALUDE_copper_needed_in_mixture_l432_43240


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l432_43230

/-- Given an arithmetic sequence {aₙ} where a₁ = 1 and the common difference d = 2,
    prove that a₈ = 15. -/
theorem arithmetic_sequence_eighth_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = 2) →  -- Common difference is 2
    a 1 = 1 →                    -- First term is 1
    a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l432_43230


namespace NUMINAMATH_CALUDE_count_flippy_divisible_by_25_is_24_l432_43215

/-- A flippy number alternates between two distinct digits. -/
def is_flippy (n : ℕ) : Prop := sorry

/-- Checks if a number is six digits long. -/
def is_six_digit (n : ℕ) : Prop := sorry

/-- Counts the number of six-digit flippy numbers divisible by 25. -/
def count_flippy_divisible_by_25 : ℕ := sorry

/-- Theorem stating that the count of six-digit flippy numbers divisible by 25 is 24. -/
theorem count_flippy_divisible_by_25_is_24 : count_flippy_divisible_by_25 = 24 := by sorry

end NUMINAMATH_CALUDE_count_flippy_divisible_by_25_is_24_l432_43215


namespace NUMINAMATH_CALUDE_smallest_terminating_n_is_correct_l432_43246

/-- A fraction a/b is a terminating decimal if b has only 2 and 5 as prime factors -/
def IsTerminatingDecimal (a b : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ b → p = 2 ∨ p = 5

/-- The smallest positive integer n such that n/(n+150) is a terminating decimal -/
def SmallestTerminatingN : ℕ := 10

theorem smallest_terminating_n_is_correct :
  IsTerminatingDecimal SmallestTerminatingN (SmallestTerminatingN + 150) ∧
  ∀ m : ℕ, 0 < m → m < SmallestTerminatingN →
    ¬IsTerminatingDecimal m (m + 150) := by
  sorry

end NUMINAMATH_CALUDE_smallest_terminating_n_is_correct_l432_43246


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l432_43267

/-- A tree that triples its height each year --/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem: A tree that triples its height each year and reaches 81 feet after 4 years
    will have a height of 9 feet after 2 years --/
theorem tree_height_after_two_years
  (h : tree_height (tree_height h₀ 2) 2 = 81)
  (h₀ : ℝ) :
  tree_height h₀ 2 = 9 :=
sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l432_43267


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l432_43293

theorem arithmetic_geometric_mean_problem (a b : ℝ) 
  (h1 : (a + b) / 2 = 24) 
  (h2 : Real.sqrt (a * b) = 2 * Real.sqrt 110) : 
  a^2 + b^2 = 1424 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l432_43293


namespace NUMINAMATH_CALUDE_inequality_holds_for_n_2_and_8_l432_43254

theorem inequality_holds_for_n_2_and_8 (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  ((Real.exp (2 * x)) / (Real.log y)^2 > (x / y)^2) ∧
  ((Real.exp (2 * x)) / (Real.log y)^2 > (x / y)^8) :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_for_n_2_and_8_l432_43254


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l432_43241

/-- Given two similar right triangles, where the first triangle has a side of 18 units
    and a hypotenuse of 30 units, and the second triangle has a hypotenuse of 60 units,
    the side in the second triangle corresponding to the 18-unit side in the first triangle
    is 36 units long. -/
theorem similar_triangles_side_length (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  a^2 + 18^2 = 30^2 →
  c^2 + d^2 = 60^2 →
  30 / 60 = 18 / d →
  d = 36 := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l432_43241


namespace NUMINAMATH_CALUDE_complex_equation_solution_l432_43248

theorem complex_equation_solution (z : ℂ) : (3 - 4*I + z)*I = 2 + I → z = -2 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l432_43248


namespace NUMINAMATH_CALUDE_dinner_cost_is_36_l432_43263

/-- Represents the farming scenario with kids planting corn --/
structure FarmingScenario where
  kids : ℕ
  ears_per_row : ℕ
  seeds_per_bag : ℕ
  seeds_per_ear : ℕ
  pay_per_row : ℚ
  bags_per_kid : ℕ

/-- Calculates the cost of dinner per kid based on the farming scenario --/
def dinner_cost_per_kid (scenario : FarmingScenario) : ℚ :=
  let ears_per_bag := scenario.seeds_per_bag / scenario.seeds_per_ear
  let total_ears := scenario.bags_per_kid * ears_per_bag
  let rows_planted := total_ears / scenario.ears_per_row
  let earnings := rows_planted * scenario.pay_per_row
  earnings / 2

/-- Theorem stating that the dinner cost per kid is $36 given the specific scenario --/
theorem dinner_cost_is_36 (scenario : FarmingScenario) 
  (h1 : scenario.kids = 4)
  (h2 : scenario.ears_per_row = 70)
  (h3 : scenario.seeds_per_bag = 48)
  (h4 : scenario.seeds_per_ear = 2)
  (h5 : scenario.pay_per_row = 3/2)
  (h6 : scenario.bags_per_kid = 140) :
  dinner_cost_per_kid scenario = 36 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_is_36_l432_43263


namespace NUMINAMATH_CALUDE_binomial_coefficient_modulo_prime_l432_43290

theorem binomial_coefficient_modulo_prime (p n q : ℕ) : 
  Prime p → 
  0 < n → 
  0 < q → 
  (n ≠ q * (p - 1) → Nat.choose n (p - 1) % p = 0) ∧
  (n = q * (p - 1) → Nat.choose n (p - 1) % p = 1) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_modulo_prime_l432_43290


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_six_l432_43228

theorem sum_of_solutions_eq_six :
  ∃ (N₁ N₂ : ℝ), N₁ * (N₁ - 6) = -7 ∧ N₂ * (N₂ - 6) = -7 ∧ N₁ + N₂ = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_six_l432_43228


namespace NUMINAMATH_CALUDE_smallest_positive_difference_l432_43261

theorem smallest_positive_difference (a b : ℤ) (h : 17 * a + 6 * b = 13) :
  ∃ (k : ℤ), a - b = 17 + 23 * k ∧ ∀ (m : ℤ), m > 0 → m = a - b → m ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_difference_l432_43261


namespace NUMINAMATH_CALUDE_sin_cos_pi_12_star_l432_43281

-- Define the * operation
def star (a b : ℝ) : ℝ := a^2 - a*b - b^2

-- State the theorem
theorem sin_cos_pi_12_star :
  star (Real.sin (π/12)) (Real.cos (π/12)) = -(1 + 2*Real.sqrt 3)/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_pi_12_star_l432_43281


namespace NUMINAMATH_CALUDE_negative_y_ceil_floor_product_l432_43221

theorem negative_y_ceil_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 :=
by sorry

end NUMINAMATH_CALUDE_negative_y_ceil_floor_product_l432_43221


namespace NUMINAMATH_CALUDE_difference_is_one_over_1650_l432_43237

/-- The repeating decimal 0.060606... -/
def repeating_decimal : ℚ := 2 / 33

/-- The terminating decimal 0.06 -/
def terminating_decimal : ℚ := 6 / 100

/-- The difference between the repeating decimal and the terminating decimal -/
def difference : ℚ := repeating_decimal - terminating_decimal

theorem difference_is_one_over_1650 : difference = 1 / 1650 := by
  sorry

end NUMINAMATH_CALUDE_difference_is_one_over_1650_l432_43237


namespace NUMINAMATH_CALUDE_specific_wall_has_30_bricks_l432_43291

/-- Represents a brick wall with a specific pattern -/
structure BrickWall where
  num_rows : ℕ
  bottom_row_bricks : ℕ
  brick_decrease : ℕ

/-- Calculates the total number of bricks in the wall -/
def total_bricks (wall : BrickWall) : ℕ :=
  sorry

/-- Theorem stating that a specific brick wall configuration has 30 bricks in total -/
theorem specific_wall_has_30_bricks :
  let wall : BrickWall := {
    num_rows := 5,
    bottom_row_bricks := 8,
    brick_decrease := 1
  }
  total_bricks wall = 30 := by
  sorry

end NUMINAMATH_CALUDE_specific_wall_has_30_bricks_l432_43291


namespace NUMINAMATH_CALUDE_total_wall_area_to_paint_l432_43279

def living_room_width : ℝ := 40
def living_room_length : ℝ := 40
def bedroom_width : ℝ := 10
def bedroom_length : ℝ := 12
def wall_height : ℝ := 10
def living_room_walls_to_paint : ℕ := 3
def bedroom_walls_to_paint : ℕ := 4

theorem total_wall_area_to_paint :
  (living_room_walls_to_paint * living_room_width * wall_height) +
  (bedroom_walls_to_paint * bedroom_width * wall_height) +
  (bedroom_walls_to_paint * bedroom_length * wall_height) -
  (2 * bedroom_width * wall_height) = 1640 := by sorry

end NUMINAMATH_CALUDE_total_wall_area_to_paint_l432_43279


namespace NUMINAMATH_CALUDE_solve_system_for_y_l432_43202

theorem solve_system_for_y :
  ∃ (x y : ℚ), (3 * x - y = 24 ∧ x + 2 * y = 10) → y = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_for_y_l432_43202


namespace NUMINAMATH_CALUDE_range_of_2a_minus_b_l432_43208

theorem range_of_2a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 3) (hb : 2 < b ∧ b < 4) :
  -2 < 2*a - b ∧ 2*a - b < 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_2a_minus_b_l432_43208


namespace NUMINAMATH_CALUDE_perpendicular_lines_l432_43251

/-- Two lines are perpendicular if the sum of the products of their corresponding coefficients is zero -/
def perpendicular (a1 b1 a2 b2 : ℝ) : Prop :=
  a1 * a2 + b1 * b2 = 0

/-- The problem statement -/
theorem perpendicular_lines (a : ℝ) :
  perpendicular 2 1 1 a → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l432_43251


namespace NUMINAMATH_CALUDE_binomial_plus_ten_l432_43266

theorem binomial_plus_ten : (Nat.choose 15 12) + 10 = 465 := by
  sorry

end NUMINAMATH_CALUDE_binomial_plus_ten_l432_43266


namespace NUMINAMATH_CALUDE_unique_line_intersection_l432_43264

theorem unique_line_intersection (m b : ℝ) : 
  (∃! k, ∃ y₁ y₂, y₁ = k^2 + 4*k + 4 ∧ y₂ = m*k + b ∧ |y₁ - y₂| = 4) ∧
  (m * 2 + b = 8) ∧
  (b ≠ 0) →
  m = 12 ∧ b = -16 := by sorry

end NUMINAMATH_CALUDE_unique_line_intersection_l432_43264


namespace NUMINAMATH_CALUDE_felicity_lollipop_collection_l432_43225

/-- The number of sticks needed to finish the fort -/
def total_sticks : ℕ := 400

/-- The number of times Felicity's family goes to the store per week -/
def store_visits_per_week : ℕ := 3

/-- The percentage of completion of the fort -/
def fort_completion_percentage : ℚ := 60 / 100

/-- The number of weeks Felicity has been collecting lollipops -/
def collection_weeks : ℕ := 80

theorem felicity_lollipop_collection :
  collection_weeks = (fort_completion_percentage * total_sticks) / store_visits_per_week := by
  sorry

end NUMINAMATH_CALUDE_felicity_lollipop_collection_l432_43225


namespace NUMINAMATH_CALUDE_x_value_l432_43226

theorem x_value (x : ℝ) : x = 88 * (1 + 0.3) → x = 114.4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l432_43226


namespace NUMINAMATH_CALUDE_max_value_polynomial_l432_43255

/-- Given real numbers x and y such that x + y = 5, 
    the maximum value of x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 is 30625/44 -/
theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (z w : ℝ), z + w = 5 ∧ 
    ∀ (a b : ℝ), a + b = 5 → 
      z^5*w + z^4*w^2 + z^3*w^3 + z^2*w^4 + z*w^5 ≥ a^5*b + a^4*b^2 + a^3*b^3 + a^2*b^4 + a*b^5) ∧
  (∀ (a b : ℝ), a + b = 5 → 
    x^5*y + x^4*y^2 + x^3*y^3 + x^2*y^4 + x*y^5 ≤ 30625/44) :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l432_43255


namespace NUMINAMATH_CALUDE_combustion_reaction_l432_43283

/-- Represents the balanced chemical equation for the combustion of methane with chlorine and oxygen -/
structure BalancedEquation where
  ch4 : ℕ
  cl2 : ℕ
  o2 : ℕ
  co2 : ℕ
  hcl : ℕ
  h2o : ℕ
  balanced : ch4 = 1 ∧ cl2 = 4 ∧ o2 = 4 ∧ co2 = 1 ∧ hcl = 4 ∧ h2o = 2

/-- Represents the given quantities and products in the reaction -/
structure ReactionQuantities where
  ch4_given : ℕ
  cl2_given : ℕ
  co2_produced : ℕ
  hcl_produced : ℕ

/-- Theorem stating the required amount of O2 and produced amount of H2O -/
theorem combustion_reaction 
  (eq : BalancedEquation) 
  (quant : ReactionQuantities) 
  (h_ch4 : quant.ch4_given = 24) 
  (h_cl2 : quant.cl2_given = 48) 
  (h_co2 : quant.co2_produced = 24) 
  (h_hcl : quant.hcl_produced = 48) :
  ∃ (o2_required h2o_produced : ℕ), 
    o2_required = 96 ∧ 
    h2o_produced = 48 :=
  sorry

end NUMINAMATH_CALUDE_combustion_reaction_l432_43283


namespace NUMINAMATH_CALUDE_sequence_property_l432_43207

theorem sequence_property (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  (∀ n, S n = n * (n - 40)) →
  (∀ n, a n = S n - S (n - 1)) →
  a 19 < 0 ∧ a 21 > 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l432_43207


namespace NUMINAMATH_CALUDE_razorback_tshirt_profit_l432_43270

/-- The amount made per t-shirt, given the number of t-shirts sold and the total amount made from t-shirts. -/
def amount_per_tshirt (num_tshirts : ℕ) (total_amount : ℕ) : ℚ :=
  total_amount / num_tshirts

/-- Theorem stating that the amount made per t-shirt is $62. -/
theorem razorback_tshirt_profit : amount_per_tshirt 183 11346 = 62 := by
  sorry

end NUMINAMATH_CALUDE_razorback_tshirt_profit_l432_43270


namespace NUMINAMATH_CALUDE_election_votes_l432_43295

theorem election_votes (total_votes : ℕ) : 
  (∃ (winner_votes loser_votes : ℕ),
    winner_votes + loser_votes = total_votes ∧
    winner_votes = (70 * total_votes) / 100 ∧
    loser_votes = (30 * total_votes) / 100 ∧
    winner_votes - loser_votes = 180) →
  total_votes = 450 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_l432_43295


namespace NUMINAMATH_CALUDE_jose_profit_share_l432_43286

/-- Calculates the share of profit for an investor in a partnership --/
def calculate_profit_share (investment : ℕ) (months : ℕ) (total_profit : ℕ) (total_capital_months : ℕ) : ℕ :=
  (investment * months * total_profit) / total_capital_months

theorem jose_profit_share :
  let tom_investment : ℕ := 30000
  let tom_months : ℕ := 12
  let jose_investment : ℕ := 45000
  let jose_months : ℕ := 10
  let total_profit : ℕ := 36000
  let total_capital_months : ℕ := tom_investment * tom_months + jose_investment * jose_months
  
  calculate_profit_share jose_investment jose_months total_profit total_capital_months = 20000 := by
  sorry


end NUMINAMATH_CALUDE_jose_profit_share_l432_43286


namespace NUMINAMATH_CALUDE_money_sharing_l432_43201

theorem money_sharing (john_share : ℕ) (jose_share : ℕ) (binoy_share : ℕ) 
  (h1 : john_share = 1400)
  (h2 : jose_share = 2 * john_share)
  (h3 : binoy_share = 3 * john_share) :
  john_share + jose_share + binoy_share = 8400 := by
  sorry

end NUMINAMATH_CALUDE_money_sharing_l432_43201


namespace NUMINAMATH_CALUDE_a0_value_sum_of_all_coefficients_sum_of_odd_coefficients_l432_43277

-- Define the polynomial coefficients
variable (a : Fin 8 → ℤ)

-- Define the equality condition
axiom expansion_equality : ∀ x : ℝ, (2*x - 1)^7 = (Finset.range 8).sum (λ i => a i * x^i)

-- Theorem statements
theorem a0_value : a 0 = -1 := by sorry

theorem sum_of_all_coefficients : (Finset.range 8).sum (λ i => a i) - a 0 = 2 := by sorry

theorem sum_of_odd_coefficients : a 1 + a 3 + a 5 + a 7 = -126 := by sorry

end NUMINAMATH_CALUDE_a0_value_sum_of_all_coefficients_sum_of_odd_coefficients_l432_43277


namespace NUMINAMATH_CALUDE_clock_setting_time_l432_43282

/-- Represents a 24-hour clock time -/
structure ClockTime where
  hours : ℕ
  minutes : ℕ
  valid : hours < 24 ∧ minutes < 60

/-- Adds minutes to a clock time, wrapping around if necessary -/
def addMinutes (t : ClockTime) (m : ℤ) : ClockTime :=
  sorry

/-- Subtracts minutes from a clock time, wrapping around if necessary -/
def subtractMinutes (t : ClockTime) (m : ℕ) : ClockTime :=
  sorry

theorem clock_setting_time 
  (initial_time : ClockTime)
  (elapsed_hours : ℕ)
  (gain_rate : ℕ)
  (loss_rate : ℕ)
  (h : elapsed_hours = 20)
  (hgain : gain_rate = 1)
  (hloss : loss_rate = 2)
  (hfinal_diff : addMinutes initial_time (elapsed_hours * gain_rate) = 
                 addMinutes (subtractMinutes initial_time (elapsed_hours * loss_rate)) 60)
  (hfinal_time : addMinutes initial_time (elapsed_hours * gain_rate) = 
                 { hours := 12, minutes := 0, valid := sorry }) :
  initial_time = { hours := 15, minutes := 40, valid := sorry } :=
sorry

end NUMINAMATH_CALUDE_clock_setting_time_l432_43282


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l432_43260

theorem quadratic_minimum_value (k : ℝ) : 
  (∀ x y : ℝ, 5*x^2 - 8*k*x*y + (4*k^2 + 3)*y^2 - 10*x - 6*y + 9 ≥ 0) ∧ 
  (∃ x y : ℝ, 5*x^2 - 8*k*x*y + (4*k^2 + 3)*y^2 - 10*x - 6*y + 9 = 0) →
  k = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l432_43260


namespace NUMINAMATH_CALUDE_activities_alignment_period_l432_43216

def activity_frequencies : List Nat := [6, 4, 16, 12, 8, 13, 17]

theorem activities_alignment_period :
  Nat.lcm (List.foldl Nat.lcm 1 activity_frequencies) = 10608 := by
  sorry

end NUMINAMATH_CALUDE_activities_alignment_period_l432_43216


namespace NUMINAMATH_CALUDE_f_minimum_value_tangent_line_equation_l432_43210

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - x - 2 * Real.log x + 1/2

-- Theorem for the minimum value of f(x)
theorem f_minimum_value :
  ∃ (x_min : ℝ), x_min > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≥ f x_min ∧ f x_min = -2 * Real.log 2 + 1/2 :=
sorry

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (x₀ : ℝ), x₀ > 0 ∧
  (2 * x₀ + f x₀ - 2 = 0) ∧
  ∀ (x : ℝ), 2 * x + f x₀ + (x - x₀) * (x₀ - 1 - 2 / x₀) - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_tangent_line_equation_l432_43210


namespace NUMINAMATH_CALUDE_probability_roots_different_signs_l432_43273

def S : Set ℕ := {1, 2, 3, 4, 5, 6}

def quadratic_equation (a b x : ℝ) : Prop :=
  x^2 - 2*(a-3)*x + 9 - b^2 = 0

def roots_different_signs (a b : ℝ) : Prop :=
  (9 - b^2 < 0) ∧ (4*(a-3)^2 - 4*(9-b^2) > 0)

def count_valid_pairs : ℕ := 18

def total_pairs : ℕ := 36

theorem probability_roots_different_signs :
  (count_valid_pairs : ℚ) / (total_pairs : ℚ) = 1/2 :=
sorry

end NUMINAMATH_CALUDE_probability_roots_different_signs_l432_43273


namespace NUMINAMATH_CALUDE_point_inside_circle_l432_43289

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    eccentricity e = √2, right focus F(c, 0), and an equation ax² - bx - c = 0
    with roots x₁ and x₂, prove that the point P(x₁, x₂) is inside the circle x² + y² = 8 -/
theorem point_inside_circle (a b c : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_eccentricity : c / a = Real.sqrt 2)
  (h_focus : c > 0)
  (h_roots : a * x₁^2 - b * x₁ - c = 0 ∧ a * x₂^2 - b * x₂ - c = 0) :
  x₁^2 + x₂^2 < 8 := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_l432_43289


namespace NUMINAMATH_CALUDE_elephants_viewing_time_l432_43276

def zoo_visit (total_time seals_time penguins_multiplier : ℕ) : ℕ :=
  total_time - (seals_time + seals_time * penguins_multiplier)

theorem elephants_viewing_time :
  zoo_visit 130 13 8 = 13 := by
  sorry

end NUMINAMATH_CALUDE_elephants_viewing_time_l432_43276


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l432_43209

/-- An ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point on the ellipse -/
def PointOnEllipse (C : Ellipse) := ℝ × ℝ

/-- The angle F₁MF₂ for a point M on the ellipse -/
def angle (C : Ellipse) (M : PointOnEllipse C) : ℝ := sorry

/-- The eccentricity of an ellipse -/
def eccentricity (C : Ellipse) : ℝ := sorry

/-- Theorem: If there exists a point M on ellipse C such that ∠F₁MF₂ = π/3,
    then the eccentricity e of C satisfies 1/2 ≤ e < 1 -/
theorem ellipse_eccentricity_range (C : Ellipse) :
  (∃ M : PointOnEllipse C, angle C M = π / 3) →
  let e := eccentricity C
  1 / 2 ≤ e ∧ e < 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l432_43209


namespace NUMINAMATH_CALUDE_S_description_l432_43218

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 
    let x := p.1
    let y := p.2
    ((5 = x + 3 ∧ (y - 6 ≤ 5 ∨ y - 6 = 5 / 2)) ∨
     (5 = y - 6 ∧ (x + 3 ≤ 5 ∨ x + 3 = 5 / 2)) ∨
     (x + 3 = y - 6 ∧ 5 = (x + 3) / 2))}

-- Define what it means to be parts of a right triangle
def isPartsOfRightTriangle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b c : ℝ × ℝ,
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    a.1 = b.1 ∧ b.2 = c.2 ∧
    (c.1 - a.1) * (b.2 - a.2) = 0

-- Define what it means to have a separate point
def hasSeparatePoint (S : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ S ∧ ∀ q ∈ S, q ≠ p → ‖p - q‖ > 0

-- Theorem statement
theorem S_description :
  isPartsOfRightTriangle S ∧ hasSeparatePoint S :=
sorry

end NUMINAMATH_CALUDE_S_description_l432_43218


namespace NUMINAMATH_CALUDE_fiftieth_term_of_specific_sequence_l432_43214

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem fiftieth_term_of_specific_sequence :
  arithmetic_sequence 3 2 50 = 101 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_specific_sequence_l432_43214


namespace NUMINAMATH_CALUDE_divisibility_by_six_l432_43299

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_875a (a : ℕ) : ℕ := 8750 + a

theorem divisibility_by_six (a : ℕ) (h : is_single_digit a) : 
  (number_875a a) % 6 = 0 ↔ a = 4 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l432_43299


namespace NUMINAMATH_CALUDE_total_roses_theorem_l432_43256

/-- The number of bouquets to be made -/
def num_bouquets : ℕ := 5

/-- The number of table decorations to be made -/
def num_table_decorations : ℕ := 7

/-- The number of white roses used in each bouquet -/
def roses_per_bouquet : ℕ := 5

/-- The number of white roses used in each table decoration -/
def roses_per_table_decoration : ℕ := 12

/-- The total number of white roses needed for all bouquets and table decorations -/
def total_roses_needed : ℕ := num_bouquets * roses_per_bouquet + num_table_decorations * roses_per_table_decoration

theorem total_roses_theorem : total_roses_needed = 109 := by
  sorry

end NUMINAMATH_CALUDE_total_roses_theorem_l432_43256


namespace NUMINAMATH_CALUDE_train_crossing_time_l432_43224

/-- Time for a train to cross a man moving in the opposite direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 150 →
  train_speed = 25 →
  man_speed = 2 →
  (train_length / ((train_speed + man_speed) * (5/18))) = 20 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l432_43224


namespace NUMINAMATH_CALUDE_dave_spent_43_tickets_l432_43294

/-- The number of tickets Dave started with -/
def initial_tickets : ℕ := 98

/-- The number of tickets Dave had left after buying the stuffed tiger -/
def remaining_tickets : ℕ := 55

/-- The number of tickets Dave spent on the stuffed tiger -/
def spent_tickets : ℕ := initial_tickets - remaining_tickets

theorem dave_spent_43_tickets : spent_tickets = 43 := by
  sorry

end NUMINAMATH_CALUDE_dave_spent_43_tickets_l432_43294


namespace NUMINAMATH_CALUDE_percentage_six_plus_years_l432_43231

/-- Represents the number of marks for each tenure range --/
structure TenureMarks where
  lessThan1Year : ℕ
  oneToTwo : ℕ
  twoToThree : ℕ
  threeToFour : ℕ
  fourToFive : ℕ
  fiveToSix : ℕ
  sixToSeven : ℕ
  sevenToEight : ℕ
  eightToNine : ℕ
  nineToTen : ℕ

/-- Calculates the total number of marks --/
def totalMarks (tm : TenureMarks) : ℕ :=
  tm.lessThan1Year + tm.oneToTwo + tm.twoToThree + tm.threeToFour +
  tm.fourToFive + tm.fiveToSix + tm.sixToSeven + tm.sevenToEight +
  tm.eightToNine + tm.nineToTen

/-- Calculates the number of marks for employees with 6 or more years --/
def marksForSixPlusYears (tm : TenureMarks) : ℕ :=
  tm.sixToSeven + tm.sevenToEight + tm.eightToNine + tm.nineToTen

/-- The main theorem stating that the percentage of employees working for 6 years or more is 17.14% --/
theorem percentage_six_plus_years (tm : TenureMarks) 
  (h : tm = { lessThan1Year := 6, oneToTwo := 6, twoToThree := 7, threeToFour := 4,
              fourToFive := 3, fiveToSix := 3, sixToSeven := 3, sevenToEight := 1,
              eightToNine := 1, nineToTen := 1 }) : 
  (marksForSixPlusYears tm : ℚ) / (totalMarks tm : ℚ) * 100 = 17.14 := by
  sorry

end NUMINAMATH_CALUDE_percentage_six_plus_years_l432_43231


namespace NUMINAMATH_CALUDE_bobs_remaining_amount_l432_43235

/-- Calculates the remaining amount after Bob's spending over three days. -/
def remaining_amount (initial : ℚ) (mon_frac : ℚ) (tue_frac : ℚ) (wed_frac : ℚ) : ℚ :=
  let after_mon := initial * (1 - mon_frac)
  let after_tue := after_mon * (1 - tue_frac)
  after_tue * (1 - wed_frac)

/-- Theorem stating that Bob's remaining amount is $20 after three days of spending. -/
theorem bobs_remaining_amount :
  remaining_amount 80 (1/2) (1/5) (3/8) = 20 := by
  sorry

end NUMINAMATH_CALUDE_bobs_remaining_amount_l432_43235


namespace NUMINAMATH_CALUDE_deductive_reasoning_properties_l432_43250

-- Define the properties of deductive reasoning
def is_general_to_specific (r : Type) : Prop := sorry
def conclusion_always_correct (r : Type) : Prop := sorry
def has_syllogism_form (r : Type) : Prop := sorry
def correctness_depends_on_premises_and_form (r : Type) : Prop := sorry

-- Define deductive reasoning
def deductive_reasoning : Type := sorry

-- Theorem stating that exactly 3 out of 4 statements are correct
theorem deductive_reasoning_properties :
  is_general_to_specific deductive_reasoning ∧
  ¬conclusion_always_correct deductive_reasoning ∧
  has_syllogism_form deductive_reasoning ∧
  correctness_depends_on_premises_and_form deductive_reasoning :=
sorry

end NUMINAMATH_CALUDE_deductive_reasoning_properties_l432_43250


namespace NUMINAMATH_CALUDE_min_guests_at_banquet_l432_43284

/-- The minimum number of guests at a football banquet given the total food consumed and maximum consumption per guest -/
theorem min_guests_at_banquet (total_food : ℝ) (max_per_guest : ℝ) (h1 : total_food = 319) (h2 : max_per_guest = 2.0) : ℕ := by
  sorry

end NUMINAMATH_CALUDE_min_guests_at_banquet_l432_43284


namespace NUMINAMATH_CALUDE_fourth_month_sale_l432_43287

/-- Proves that the sale in the fourth month is 5399, given the sales for other months and the required average. -/
theorem fourth_month_sale
  (sale1 sale2 sale3 sale5 sale6 : ℕ)
  (average : ℕ)
  (h1 : sale1 = 5124)
  (h2 : sale2 = 5366)
  (h3 : sale3 = 5808)
  (h5 : sale5 = 6124)
  (h6 : sale6 = 4579)
  (h_avg : average = 5400)
  (h_total : sale1 + sale2 + sale3 + sale5 + sale6 + (6 * average - (sale1 + sale2 + sale3 + sale5 + sale6)) = 6 * average) :
  6 * average - (sale1 + sale2 + sale3 + sale5 + sale6) = 5399 :=
by sorry


end NUMINAMATH_CALUDE_fourth_month_sale_l432_43287


namespace NUMINAMATH_CALUDE_valid_pairs_count_l432_43205

/-- A function that counts the number of valid (a,b) pairs -/
def count_valid_pairs : ℕ :=
  (Finset.range 50).sum (fun a => 
    Nat.ceil (((a + 1) : ℕ) / 2))

/-- The main theorem stating that there are exactly 75 valid pairs -/
theorem valid_pairs_count : count_valid_pairs = 75 := by
  sorry

end NUMINAMATH_CALUDE_valid_pairs_count_l432_43205


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l432_43253

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_planes 
  (a : Line) (α β : Plane) :
  perpendicular a α → parallel α β → perpendicular a β :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l432_43253


namespace NUMINAMATH_CALUDE_adjacent_knights_probability_l432_43275

/-- The number of knights seated at the round table -/
def num_knights : ℕ := 25

/-- The number of knights chosen -/
def chosen_knights : ℕ := 3

/-- The probability of choosing at least two adjacent knights -/
def P : ℚ := 21/92

/-- Theorem stating the probability of choosing at least two adjacent knights -/
theorem adjacent_knights_probability :
  (
    let total_choices := Nat.choose num_knights chosen_knights
    let adjacent_triples := num_knights
    let adjacent_pairs := num_knights * (num_knights - 2 * chosen_knights + 1)
    let favorable_outcomes := adjacent_triples + adjacent_pairs
    (favorable_outcomes : ℚ) / total_choices
  ) = P := by sorry

end NUMINAMATH_CALUDE_adjacent_knights_probability_l432_43275


namespace NUMINAMATH_CALUDE_treehouse_planks_l432_43223

theorem treehouse_planks (total : ℕ) (storage_fraction : ℚ) (parents_fraction : ℚ) (friends : ℕ) 
  (h1 : total = 200)
  (h2 : storage_fraction = 1/4)
  (h3 : parents_fraction = 1/2)
  (h4 : friends = 20) :
  total - (↑total * storage_fraction).num - (↑total * parents_fraction).num - friends = 30 := by
  sorry

end NUMINAMATH_CALUDE_treehouse_planks_l432_43223


namespace NUMINAMATH_CALUDE_surveyor_distance_theorem_l432_43227

/-- The distance traveled by the surveyor when he heard the blast -/
def surveyorDistance : ℝ := 122

/-- The time it takes for the fuse to burn (in seconds) -/
def fuseTime : ℝ := 20

/-- The speed of the surveyor (in yards per second) -/
def surveyorSpeed : ℝ := 6

/-- The speed of sound (in feet per second) -/
def soundSpeed : ℝ := 960

/-- Conversion factor from yards to feet -/
def yardsToFeet : ℝ := 3

theorem surveyor_distance_theorem :
  let t := (soundSpeed * fuseTime) / (soundSpeed - surveyorSpeed * yardsToFeet)
  surveyorDistance = surveyorSpeed * t := by sorry

end NUMINAMATH_CALUDE_surveyor_distance_theorem_l432_43227


namespace NUMINAMATH_CALUDE_pairball_playing_time_l432_43238

theorem pairball_playing_time (total_time : ℕ) (num_children : ℕ) : 
  total_time = 90 → num_children = 5 → (total_time * 2) / num_children = 36 := by
  sorry

end NUMINAMATH_CALUDE_pairball_playing_time_l432_43238


namespace NUMINAMATH_CALUDE_leading_coefficient_of_p_l432_43239

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := 5*(x^5 - 3*x^4 + 2*x^3) - 6*(x^5 + x^3 + 1) + 2*(3*x^5 - x^4 + x^2)

/-- The leading coefficient of a polynomial -/
def leadingCoefficient (p : ℝ → ℝ) : ℝ :=
  sorry -- Definition of leading coefficient

theorem leading_coefficient_of_p :
  leadingCoefficient p = 5 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_p_l432_43239


namespace NUMINAMATH_CALUDE_expansion_properties_l432_43268

theorem expansion_properties (n : ℕ) : 
  (∃ a b : ℚ, 
    (1 : ℚ) = a ∧ 
    (n : ℚ) * (1 / 2 : ℚ) = a + b ∧ 
    (n * (n - 1) / 2 : ℚ) * (1 / 4 : ℚ) = a + 2 * b) → 
  n = 8 ∧ (2 : ℕ) ^ n = 256 :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l432_43268


namespace NUMINAMATH_CALUDE_algebra_test_average_l432_43262

theorem algebra_test_average : ∀ (male_count female_count : ℕ) 
  (male_avg female_avg overall_avg : ℚ),
  male_count = 8 →
  female_count = 28 →
  male_avg = 83 →
  female_avg = 92 →
  overall_avg = (male_count * male_avg + female_count * female_avg) / (male_count + female_count) →
  overall_avg = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_algebra_test_average_l432_43262


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l432_43245

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nonagon has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l432_43245


namespace NUMINAMATH_CALUDE_card_58_is_6_l432_43200

/-- The sequence of playing cards -/
def card_sequence : ℕ → ℕ :=
  fun n => (n - 1) % 13 + 1

/-- The 58th card in the sequence -/
def card_58 : ℕ := card_sequence 58

theorem card_58_is_6 : card_58 = 6 := by
  sorry

end NUMINAMATH_CALUDE_card_58_is_6_l432_43200


namespace NUMINAMATH_CALUDE_expression_simplification_l432_43243

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 3 + 1) :
  ((a^2 / (a - 2) - 1 / (a - 2)) / ((a^2 - 2*a + 1) / (a - 2))) = (3 + 2 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l432_43243


namespace NUMINAMATH_CALUDE_pairings_count_l432_43219

/-- The number of bowls -/
def num_bowls : ℕ := 5

/-- The number of glasses -/
def num_glasses : ℕ := 5

/-- The number of possible pairings when choosing one bowl and one glass -/
def num_pairings : ℕ := num_bowls * num_glasses

/-- Theorem stating that the number of possible pairings is 25 -/
theorem pairings_count : num_pairings = 25 := by
  sorry

end NUMINAMATH_CALUDE_pairings_count_l432_43219


namespace NUMINAMATH_CALUDE_vector_simplification_l432_43234

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification 
  (O P Q S : V) : 
  (O - P) - (Q - P) + (P - S) + (S - P) = O - Q := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l432_43234


namespace NUMINAMATH_CALUDE_cubic_equation_properties_l432_43257

theorem cubic_equation_properties :
  (∀ x y : ℕ, x^3 + y = y^3 + x → x = y) ∧
  (∃ x y : ℚ, x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + y = y^3 + x) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_properties_l432_43257


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l432_43274

/-- A sequence a, b, c forms a geometric sequence if there exists a non-zero real number r such that b = ar and c = br -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r

theorem geometric_sequence_property (a b c : ℝ) :
  (IsGeometricSequence a b c → b^2 = a * c) ∧
  ¬(b^2 = a * c → IsGeometricSequence a b c) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l432_43274


namespace NUMINAMATH_CALUDE_quadratic_one_root_iff_discriminant_zero_unique_positive_m_for_one_root_l432_43292

/-- A quadratic equation has exactly one real root if and only if its discriminant is zero -/
theorem quadratic_one_root_iff_discriminant_zero {a b c : ℝ} (ha : a ≠ 0) :
  (∃! x, a * x^2 + b * x + c = 0) ↔ b^2 - 4*a*c = 0 := by sorry

theorem unique_positive_m_for_one_root :
  ∃! (m : ℝ), m > 0 ∧ (∃! x, x^2 + 6*m*x + 2*m = 0) := by
  use 2/9
  constructor
  · sorry
  · sorry

end NUMINAMATH_CALUDE_quadratic_one_root_iff_discriminant_zero_unique_positive_m_for_one_root_l432_43292


namespace NUMINAMATH_CALUDE_janet_extra_fica_tax_l432_43252

/-- Represents Janet's employment situation -/
structure Employment where
  hours_per_week : ℕ
  current_hourly_rate : ℚ
  freelance_hourly_rate : ℚ
  healthcare_premium_per_month : ℚ
  weeks_per_month : ℕ
  additional_monthly_income_freelancing : ℚ

/-- Calculates the extra weekly FICA tax for freelancing -/
def extra_weekly_fica_tax (e : Employment) : ℚ :=
  let current_monthly_income := e.current_hourly_rate * e.hours_per_week * e.weeks_per_month
  let freelance_monthly_income := e.freelance_hourly_rate * e.hours_per_week * e.weeks_per_month
  let extra_income := freelance_monthly_income - current_monthly_income
  let extra_income_after_healthcare := extra_income - e.healthcare_premium_per_month
  let monthly_fica_tax := e.additional_monthly_income_freelancing - extra_income_after_healthcare
  monthly_fica_tax / e.weeks_per_month

/-- Theorem stating that Janet's extra weekly FICA tax for freelancing is $25 -/
theorem janet_extra_fica_tax :
  let janet : Employment := {
    hours_per_week := 40,
    current_hourly_rate := 30,
    freelance_hourly_rate := 40,
    healthcare_premium_per_month := 400,
    weeks_per_month := 4,
    additional_monthly_income_freelancing := 1100
  }
  extra_weekly_fica_tax janet = 25 := by
  sorry

end NUMINAMATH_CALUDE_janet_extra_fica_tax_l432_43252


namespace NUMINAMATH_CALUDE_system_solution_l432_43247

theorem system_solution (x y : Real) (k₁ k₂ : Int) : 
  (Real.sqrt 2 * Real.sin x = Real.sin y) →
  (Real.sqrt 2 * Real.cos x = Real.sqrt 3 * Real.cos y) →
  (∃ n₁ n₂ : Int, x = n₁ * π / 6 + k₂ * π ∧ y = n₂ * π / 4 + k₁ * π ∧ 
   (n₁ = 1 ∨ n₁ = -1) ∧ (n₂ = 1 ∨ n₂ = -1) ∧ k₁ % 2 = k₂ % 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l432_43247


namespace NUMINAMATH_CALUDE_loot_box_loss_l432_43233

theorem loot_box_loss (cost_per_box : ℝ) (avg_value_per_box : ℝ) (total_spent : ℝ)
  (h1 : cost_per_box = 5)
  (h2 : avg_value_per_box = 3.5)
  (h3 : total_spent = 40) :
  (total_spent / cost_per_box) * (cost_per_box - avg_value_per_box) = 12 :=
by sorry

end NUMINAMATH_CALUDE_loot_box_loss_l432_43233


namespace NUMINAMATH_CALUDE_two_n_squared_lt_three_to_n_l432_43259

theorem two_n_squared_lt_three_to_n (n : ℕ+) : 2 * n.val ^ 2 < 3 ^ n.val := by sorry

end NUMINAMATH_CALUDE_two_n_squared_lt_three_to_n_l432_43259


namespace NUMINAMATH_CALUDE_book_sale_fraction_l432_43212

/-- Given a book sale where some books were sold for $2 each, 36 books remained unsold,
    and the total amount received was $144, prove that 2/3 of the books were sold. -/
theorem book_sale_fraction (B : ℕ) (h1 : B > 36) : 
  2 * (B - 36) = 144 → (B - 36 : ℚ) / B = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_fraction_l432_43212


namespace NUMINAMATH_CALUDE_vanessa_birthday_money_l432_43249

theorem vanessa_birthday_money (money : ℕ) : 
  (∃ k : ℕ, money = 9 * k + 1) ↔ 
  (∃ n : ℕ, money = 9 * n + 1 ∧ 
    ∀ m : ℕ, m < n → money ≥ 9 * m + 1) :=
by sorry

end NUMINAMATH_CALUDE_vanessa_birthday_money_l432_43249


namespace NUMINAMATH_CALUDE_negation_to_original_proposition_l432_43244

theorem negation_to_original_proposition :
  (¬ (∃ x : ℝ, x < 1 ∧ x^2 < 1)) ↔ (∀ x : ℝ, x < 1 → x^2 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_to_original_proposition_l432_43244
