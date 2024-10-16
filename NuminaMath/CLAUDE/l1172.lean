import Mathlib

namespace NUMINAMATH_CALUDE_stack_surface_area_l1172_117256

/-- Calculates the external surface area of a stack of cubes -/
def external_surface_area (volumes : List ℕ) : ℕ :=
  let side_lengths := volumes.map (fun v => v^(1/3))
  let surface_areas := side_lengths.map (fun s => 6 * s^2)
  let adjusted_areas := surface_areas.zip side_lengths
    |> List.map (fun (area, s) => area - s^2)
  adjusted_areas.sum + 6 * (volumes.head!^(1/3))^2

/-- The volumes of the cubes in the stack -/
def cube_volumes : List ℕ := [512, 343, 216, 125, 64, 27, 8, 1]

theorem stack_surface_area :
  external_surface_area cube_volumes = 1021 := by
  sorry

end NUMINAMATH_CALUDE_stack_surface_area_l1172_117256


namespace NUMINAMATH_CALUDE_teachers_distribution_arrangements_l1172_117206

/-- The number of ways to distribute teachers between two classes -/
def distribute_teachers (total_teachers : ℕ) (max_per_class : ℕ) : ℕ :=
  let equal_distribution := 1
  let unequal_distribution := 2 * (Nat.choose total_teachers max_per_class)
  equal_distribution + unequal_distribution

/-- Theorem stating that distributing 6 teachers with a maximum of 4 per class results in 31 arrangements -/
theorem teachers_distribution_arrangements :
  distribute_teachers 6 4 = 31 := by
  sorry

end NUMINAMATH_CALUDE_teachers_distribution_arrangements_l1172_117206


namespace NUMINAMATH_CALUDE_cricket_overs_calculation_l1172_117284

theorem cricket_overs_calculation (total_target : ℝ) (initial_rate : ℝ) (required_rate : ℝ) 
  (remaining_overs : ℝ) (h1 : total_target = 262) (h2 : initial_rate = 3.2) 
  (h3 : required_rate = 5.75) (h4 : remaining_overs = 40) : 
  ∃ (initial_overs : ℝ), initial_overs = 10 ∧ 
  total_target = initial_rate * initial_overs + required_rate * remaining_overs :=
by
  sorry

end NUMINAMATH_CALUDE_cricket_overs_calculation_l1172_117284


namespace NUMINAMATH_CALUDE_min_value_expression_l1172_117280

theorem min_value_expression (x y z : ℝ) (h : 2 * x * y + y * z > 0) :
  (x^2 + y^2 + z^2) / (2 * x * y + y * z) ≥ 3 ∧
  ∃ (x₀ y₀ z₀ : ℝ), 2 * x₀ * y₀ + y₀ * z₀ > 0 ∧
    (x₀^2 + y₀^2 + z₀^2) / (2 * x₀ * y₀ + y₀ * z₀) = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1172_117280


namespace NUMINAMATH_CALUDE_pencil_count_l1172_117214

theorem pencil_count (initial : ℕ) (nancy_added : ℕ) (steven_added : ℕ)
  (h1 : initial = 138)
  (h2 : nancy_added = 256)
  (h3 : steven_added = 97) :
  initial + nancy_added + steven_added = 491 :=
by sorry

end NUMINAMATH_CALUDE_pencil_count_l1172_117214


namespace NUMINAMATH_CALUDE_roses_per_girl_l1172_117279

theorem roses_per_girl (total_students : Nat) (total_plants : Nat) (total_birches : Nat) 
  (h1 : total_students = 24)
  (h2 : total_plants = 24)
  (h3 : total_birches = 6)
  (h4 : total_birches * 3 ≤ total_students) :
  ∃ (roses_per_girl : Nat), 
    roses_per_girl * (total_students - total_birches * 3) = total_plants - total_birches ∧ 
    roses_per_girl = 3 := by
  sorry

end NUMINAMATH_CALUDE_roses_per_girl_l1172_117279


namespace NUMINAMATH_CALUDE_inequality_part_1_inequality_part_2_l1172_117203

-- Part I
theorem inequality_part_1 : 
  ∀ x : ℝ, (|x - 3| + |x + 5| ≥ 2 * |x + 5|) ↔ (x ≤ -1) := by sorry

-- Part II
theorem inequality_part_2 : 
  ∀ a : ℝ, (∀ x : ℝ, |x - a| + |x + 5| ≥ 6) ↔ (a ≥ 1 ∨ a ≤ -11) := by sorry

end NUMINAMATH_CALUDE_inequality_part_1_inequality_part_2_l1172_117203


namespace NUMINAMATH_CALUDE_max_principals_is_five_l1172_117259

/-- Represents the duration of the entire period in years -/
def total_period : ℕ := 15

/-- Represents the length of each principal's term in years -/
def term_length : ℕ := 3

/-- Calculates the maximum number of principals that can serve in the given period -/
def max_principals : ℕ := total_period / term_length

theorem max_principals_is_five : max_principals = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_principals_is_five_l1172_117259


namespace NUMINAMATH_CALUDE_red_pepper_weight_l1172_117265

theorem red_pepper_weight (total_weight green_weight : ℚ) 
  (h1 : total_weight = 0.66)
  (h2 : green_weight = 0.33) :
  total_weight - green_weight = 0.33 := by
sorry

end NUMINAMATH_CALUDE_red_pepper_weight_l1172_117265


namespace NUMINAMATH_CALUDE_extended_triangle_PQ_length_l1172_117223

/-- Triangle ABC with extended sides and intersection points -/
structure ExtendedTriangle where
  -- Side lengths of triangle ABC
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Extended segments
  DA : ℝ
  BE : ℝ
  -- Intersection points with circumcircle of CDE
  PQ : ℝ

/-- Theorem stating the length of PQ in the given configuration -/
theorem extended_triangle_PQ_length 
  (triangle : ExtendedTriangle)
  (h1 : triangle.AB = 15)
  (h2 : triangle.BC = 18)
  (h3 : triangle.CA = 20)
  (h4 : triangle.DA = triangle.AB)
  (h5 : triangle.BE = triangle.AB)
  : triangle.PQ = 37 := by
  sorry

#check extended_triangle_PQ_length

end NUMINAMATH_CALUDE_extended_triangle_PQ_length_l1172_117223


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l1172_117260

/-- An isosceles trapezoid with given base lengths and area -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- The length of the side of an isosceles trapezoid -/
def side_length (t : IsoscelesTrapezoid) : ℝ :=
  sorry

theorem isosceles_trapezoid_side_length :
  ∀ t : IsoscelesTrapezoid,
    t.base1 = 11 ∧ t.base2 = 17 ∧ t.area = 56 →
    side_length t = 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l1172_117260


namespace NUMINAMATH_CALUDE_edward_pipe_usage_l1172_117281

/- Define the problem parameters -/
def total_washers : ℕ := 20
def remaining_washers : ℕ := 4
def washers_per_bolt : ℕ := 2
def feet_per_bolt : ℕ := 5

/- Define the function to calculate feet of pipe used -/
def feet_of_pipe_used (total_washers remaining_washers washers_per_bolt feet_per_bolt : ℕ) : ℕ :=
  let washers_used := total_washers - remaining_washers
  let bolts_used := washers_used / washers_per_bolt
  bolts_used * feet_per_bolt

/- Theorem statement -/
theorem edward_pipe_usage :
  feet_of_pipe_used total_washers remaining_washers washers_per_bolt feet_per_bolt = 40 := by
  sorry

end NUMINAMATH_CALUDE_edward_pipe_usage_l1172_117281


namespace NUMINAMATH_CALUDE_power_sum_equals_eighteen_l1172_117289

theorem power_sum_equals_eighteen :
  (-3)^4 + (-3)^2 + (-3)^1 + 3^1 - 3^4 + 3^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_eighteen_l1172_117289


namespace NUMINAMATH_CALUDE_find_q_l1172_117268

theorem find_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 8) :
  q = 4 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_find_q_l1172_117268


namespace NUMINAMATH_CALUDE_parabola_shift_sum_l1172_117253

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift_sum (p : Parabola) :
  (shift (shift p 1 0) 0 2) = { a := 1, b := -4, c := 5 } →
  p.a + p.b + p.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_sum_l1172_117253


namespace NUMINAMATH_CALUDE_money_left_relation_l1172_117275

/-- The relationship between money left and masks bought -/
theorem money_left_relation (initial_amount : ℝ) (mask_price : ℝ) (x : ℝ) (y : ℝ) :
  initial_amount = 60 →
  mask_price = 2 →
  y = initial_amount - mask_price * x →
  y = 60 - 2 * x :=
by sorry

end NUMINAMATH_CALUDE_money_left_relation_l1172_117275


namespace NUMINAMATH_CALUDE_evaluate_expression_l1172_117255

theorem evaluate_expression : 
  Real.sqrt (9 / 4) - Real.sqrt (8 / 9) + Real.sqrt 1 = (15 - 4 * Real.sqrt 2) / 6 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1172_117255


namespace NUMINAMATH_CALUDE_pool_capacity_l1172_117210

theorem pool_capacity (current_water : ℝ) (h1 : current_water > 0) 
  (h2 : current_water + 300 = 0.8 * 1875) 
  (h3 : current_water + 300 = 1.25 * current_water) : 
  1875 = 1875 := by
  sorry

end NUMINAMATH_CALUDE_pool_capacity_l1172_117210


namespace NUMINAMATH_CALUDE_positive_root_k_values_negative_solution_k_range_l1172_117286

-- Define the equation
def equation (x k : ℝ) : Prop :=
  4 / (x + 1) + 3 / (x - 1) = k / (x^2 - 1)

-- Part 1: Positive root case
theorem positive_root_k_values (k : ℝ) :
  (∃ x > 0, equation x k) → (k = 6 ∨ k = -8) := by sorry

-- Part 2: Negative solution case
theorem negative_solution_k_range (k : ℝ) :
  (∃ x < 0, equation x k) → (k < -1 ∧ k ≠ -8) := by sorry

end NUMINAMATH_CALUDE_positive_root_k_values_negative_solution_k_range_l1172_117286


namespace NUMINAMATH_CALUDE_fifty_paise_coins_count_l1172_117216

/-- Represents the types of coins in the bag -/
inductive CoinType
  | OneRupee
  | FiftyPaise
  | TwentyFivePaise

/-- Represents the bag of coins -/
structure CoinBag where
  numCoins : CoinType → ℕ
  totalValue : ℚ
  equalCoins : ∀ (c1 c2 : CoinType), numCoins c1 = numCoins c2

def coinValue : CoinType → ℚ
  | CoinType.OneRupee => 1
  | CoinType.FiftyPaise => 1/2
  | CoinType.TwentyFivePaise => 1/4

theorem fifty_paise_coins_count (bag : CoinBag) 
  (h1 : bag.totalValue = 105)
  (h2 : bag.numCoins CoinType.OneRupee = 60) :
  bag.numCoins CoinType.FiftyPaise = 60 := by
  sorry

end NUMINAMATH_CALUDE_fifty_paise_coins_count_l1172_117216


namespace NUMINAMATH_CALUDE_xyz_inequality_and_sum_l1172_117276

theorem xyz_inequality_and_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 8) :
  ((x + y < 7) → (x / (1 + x) + y / (1 + y) > 2 * Real.sqrt ((x * y) / (x * y + 8)))) ∧
  (⌈(1 / Real.sqrt (1 + x) + 1 / Real.sqrt (1 + y) + 1 / Real.sqrt (1 + z))⌉ = 2) := by
sorry

end NUMINAMATH_CALUDE_xyz_inequality_and_sum_l1172_117276


namespace NUMINAMATH_CALUDE_min_value_expression_l1172_117226

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 3/2) :
  2 * x^2 + 4 * x * y + 9 * y^2 + 10 * y * z + 3 * z^2 ≥ 27 / 2^(4/9) * Real.rpow 90 (1/9) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1172_117226


namespace NUMINAMATH_CALUDE_point_A_movement_l1172_117250

def point_movement (initial_x initial_y right_movement down_movement : ℝ) : ℝ × ℝ :=
  (initial_x + right_movement, initial_y - down_movement)

theorem point_A_movement :
  point_movement 1 0 2 3 = (3, -3) := by sorry

end NUMINAMATH_CALUDE_point_A_movement_l1172_117250


namespace NUMINAMATH_CALUDE_unique_solution_exists_l1172_117235

/-- A function satisfying the given functional equation for a constant k -/
def SatisfiesEquation (f : ℝ → ℝ) (k : ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) = x + y + k

/-- The theorem stating the uniqueness and form of the solution -/
theorem unique_solution_exists (k : ℝ) :
  ∃! f : ℝ → ℝ, SatisfiesEquation f k ∧ ∀ x : ℝ, f x = x - k :=
sorry

end NUMINAMATH_CALUDE_unique_solution_exists_l1172_117235


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1172_117207

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (a^3 - 15*a^2 + 25*a - 10 = 0) →
  (b^3 - 15*b^2 + 25*b - 10 = 0) →
  (c^3 - 15*c^2 + 25*c - 10 = 0) →
  a^2 + b^2 + c^2 = 175 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1172_117207


namespace NUMINAMATH_CALUDE_base5_1234_equals_194_l1172_117242

def base5_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

theorem base5_1234_equals_194 :
  base5_to_decimal [4, 3, 2, 1] = 194 := by
  sorry

end NUMINAMATH_CALUDE_base5_1234_equals_194_l1172_117242


namespace NUMINAMATH_CALUDE_min_a_value_l1172_117262

theorem min_a_value (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x + Real.sqrt (x * y) ≤ a * (x + y)) → 
  a ≥ (Real.sqrt 2 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_a_value_l1172_117262


namespace NUMINAMATH_CALUDE_sector_angle_l1172_117272

/-- Given a circle with radius 12 meters and a sector with area 50.28571428571428 square meters,
    the angle at the center of the circle is 40 degrees. -/
theorem sector_angle (r : ℝ) (area : ℝ) (h1 : r = 12) (h2 : area = 50.28571428571428) :
  (area * 360) / (π * r^2) = 40 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l1172_117272


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l1172_117295

theorem largest_n_divisibility (n : ℕ) : (n + 1) ∣ (n^3 + 10) → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l1172_117295


namespace NUMINAMATH_CALUDE_circle_inequality_l1172_117264

/-- Given a circle with diameter AC = 1, AB tangent to the circle, and BC intersecting the circle again at D,
    prove that if AB = a and CD = b, then 1/(a^2 + 1/2) < b/a < 1/a^2 -/
theorem circle_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  1 / (a^2 + 1/2) < b / a ∧ b / a < 1 / a^2 := by
  sorry

#check circle_inequality

end NUMINAMATH_CALUDE_circle_inequality_l1172_117264


namespace NUMINAMATH_CALUDE_cube_diagonal_length_l1172_117202

theorem cube_diagonal_length (S : ℝ) (h : S = 864) :
  ∃ (d : ℝ), d = 12 * Real.sqrt 3 ∧ d^2 = 3 * (S / 6) :=
by sorry

end NUMINAMATH_CALUDE_cube_diagonal_length_l1172_117202


namespace NUMINAMATH_CALUDE_min_questions_for_phone_number_l1172_117221

theorem min_questions_for_phone_number (n : ℕ) (h : n = 100000) :
  ∃ k : ℕ, k = 17 ∧ 2^k ≥ n ∧ ∀ m : ℕ, m < k → 2^m < n :=
by sorry

end NUMINAMATH_CALUDE_min_questions_for_phone_number_l1172_117221


namespace NUMINAMATH_CALUDE_prob_sum_18_correct_l1172_117287

/-- The number of faces on each die -/
def num_faces : ℕ := 7

/-- The target sum we're aiming for -/
def target_sum : ℕ := 18

/-- The number of dice being rolled -/
def num_dice : ℕ := 3

/-- The probability of rolling a sum of 18 with three 7-faced dice -/
def prob_sum_18 : ℚ := 4 / 343

/-- Theorem stating that the probability of rolling a sum of 18 
    with three 7-faced dice is 4/343 -/
theorem prob_sum_18_correct :
  prob_sum_18 = (num_favorable_outcomes : ℚ) / (num_faces ^ num_dice) :=
sorry

end NUMINAMATH_CALUDE_prob_sum_18_correct_l1172_117287


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_l1172_117246

/-- Sum of an arithmetic series -/
def arithmetic_sum (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- The sum of the arithmetic series with first term 22, last term 73, and common difference 3/7 is 5700 -/
theorem arithmetic_series_sum :
  arithmetic_sum 22 73 (3/7) = 5700 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_l1172_117246


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1172_117230

theorem complex_number_in_second_quadrant :
  let z : ℂ := (-2 + 3 * Complex.I) / (3 - 4 * Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1172_117230


namespace NUMINAMATH_CALUDE_parking_theorem_l1172_117208

/-- The number of ways to arrange n distinct objects in k positions --/
def arrange (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items --/
def choose (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to park cars in a row with contiguous empty spaces --/
def parkingArrangements (total_spaces : ℕ) (cars : ℕ) : ℕ :=
  arrange cars cars * choose (cars + 1) 1

theorem parking_theorem :
  parkingArrangements 12 8 = arrange 8 8 * choose 9 1 := by sorry

end NUMINAMATH_CALUDE_parking_theorem_l1172_117208


namespace NUMINAMATH_CALUDE_right_triangles_shared_hypotenuse_l1172_117270

theorem right_triangles_shared_hypotenuse (b : ℝ) (h : b ≥ Real.sqrt 3) :
  let BC : ℝ := 1
  let AC : ℝ := b
  let AD : ℝ := 2
  let AB : ℝ := Real.sqrt (AC^2 + BC^2)
  let BD : ℝ := Real.sqrt (AB^2 - AD^2)
  BD = Real.sqrt (b^2 - 3) :=
by sorry

end NUMINAMATH_CALUDE_right_triangles_shared_hypotenuse_l1172_117270


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1172_117243

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1172_117243


namespace NUMINAMATH_CALUDE_regular_decagon_interior_angle_regular_decagon_interior_angle_is_144_l1172_117217

/-- The measure of each interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle : ℝ :=
  let n : ℕ := 10  -- number of sides in a decagon
  let sum_of_angles : ℝ := (n - 2) * 180  -- sum of interior angles formula
  let angle_measure : ℝ := sum_of_angles / n  -- measure of each angle (sum divided by number of sides)
  angle_measure

/-- Proof that the measure of each interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle_is_144 : 
  regular_decagon_interior_angle = 144 := by
  sorry


end NUMINAMATH_CALUDE_regular_decagon_interior_angle_regular_decagon_interior_angle_is_144_l1172_117217


namespace NUMINAMATH_CALUDE_unicorn_rope_length_l1172_117291

theorem unicorn_rope_length (rope_length : ℝ) (tower_radius : ℝ) (rope_end_distance : ℝ) 
  (h1 : rope_length = 24)
  (h2 : tower_radius = 10)
  (h3 : rope_end_distance = 6) :
  rope_length - 2 * Real.sqrt (rope_length^2 - tower_radius^2) = 24 - 2 * Real.sqrt 119 :=
by sorry

end NUMINAMATH_CALUDE_unicorn_rope_length_l1172_117291


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1172_117218

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum1 : a 1 + a 2 + a 3 = -24) (h_sum2 : a 18 + a 19 + a 20 = 78) : 
  a 1 + a 20 = 18 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1172_117218


namespace NUMINAMATH_CALUDE_number_problem_l1172_117296

theorem number_problem : ∃ (x : ℝ), x = 40 ∧ 0.8 * x > (4/5 * 15 + 20) := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1172_117296


namespace NUMINAMATH_CALUDE_total_distance_is_164_l1172_117282

-- Define the parameters
def flat_speed : ℝ := 20
def flat_time : ℝ := 4.5
def uphill_speed : ℝ := 12
def uphill_time : ℝ := 2.5
def downhill_speed : ℝ := 24
def downhill_time : ℝ := 1.5
def walking_distance : ℝ := 8

-- Define the function to calculate distance
def distance (speed time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem total_distance_is_164 :
  distance flat_speed flat_time +
  distance uphill_speed uphill_time +
  distance downhill_speed downhill_time +
  walking_distance = 164 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_164_l1172_117282


namespace NUMINAMATH_CALUDE_multiplication_formula_examples_l1172_117200

theorem multiplication_formula_examples :
  (203 * 197 = 39991) ∧ ((-69.9)^2 = 4886.01) := by sorry

end NUMINAMATH_CALUDE_multiplication_formula_examples_l1172_117200


namespace NUMINAMATH_CALUDE_angle_between_vectors_is_acute_l1172_117213

theorem angle_between_vectors_is_acute (A B C : ℝ) (p q : ℝ × ℝ) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  A + B + C = π →
  p = (Real.cos A, Real.sin A) →
  q = (-Real.cos B, Real.sin B) →
  ∃ α, 0 < α ∧ α < π/2 ∧ Real.cos α = p.1 * q.1 + p.2 * q.2 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_is_acute_l1172_117213


namespace NUMINAMATH_CALUDE_three_true_propositions_l1172_117233

-- Definition of reciprocals
def reciprocals (x y : ℝ) : Prop := x * y = 1

-- Definition of congruent triangles
def congruent_triangles (t1 t2 : Set ℝ × Set ℝ) : Prop := sorry

-- Definition of equal area triangles
def equal_area_triangles (t1 t2 : Set ℝ × Set ℝ) : Prop := sorry

-- Definition of real solutions for quadratic equation
def has_real_solutions (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + m = 0

theorem three_true_propositions :
  (∀ x y : ℝ, reciprocals x y → x * y = 1) ∧
  (∀ t1 t2 : Set ℝ × Set ℝ, ¬(congruent_triangles t1 t2) → ¬(equal_area_triangles t1 t2)) ∧
  (∀ m : ℝ, ¬(has_real_solutions m) → m > 1) :=
sorry

end NUMINAMATH_CALUDE_three_true_propositions_l1172_117233


namespace NUMINAMATH_CALUDE_average_math_chem_score_l1172_117249

theorem average_math_chem_score (math physics chem : ℕ) : 
  math + physics = 60 → 
  chem = physics + 20 → 
  (math + chem) / 2 = 40 := by
sorry

end NUMINAMATH_CALUDE_average_math_chem_score_l1172_117249


namespace NUMINAMATH_CALUDE_correct_propositions_l1172_117248

theorem correct_propositions : 
  -- Proposition 2
  (∀ a b : ℝ, a > 0 ∧ b > 0 → (a + b) / 2 ≥ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≥ (a * b) / (a + b)) ∧
  -- Proposition 3
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) ∧
  -- Proposition 4
  (Real.log 9 * Real.log 11 < 1) ∧
  -- Proposition 5
  (∀ a b : ℝ, a > b ∧ 1/a > 1/b → a > 0 ∧ b < 0) ∧
  -- Proposition 1 (incorrect)
  ¬(∀ a b : ℝ, a < b ∧ b < 0 → 1/a < 1/b) ∧
  -- Proposition 6 (incorrect)
  ¬(∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 → (x + 2*y ≥ 6 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 1/y₀ = 1 ∧ x₀ + 2*y₀ = 6)) :=
by sorry


end NUMINAMATH_CALUDE_correct_propositions_l1172_117248


namespace NUMINAMATH_CALUDE_negation_equivalence_l1172_117293

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, 2 * x^2 + x + m ≤ 0) ↔ (∀ x : ℤ, 2 * x^2 + x + m > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1172_117293


namespace NUMINAMATH_CALUDE_no_natural_solution_l1172_117278

theorem no_natural_solution : ¬ ∃ (m n : ℕ), (1 : ℚ) / m + (1 : ℚ) / n = (7 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_l1172_117278


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1172_117285

theorem rectangle_dimensions : ∃ (x y : ℝ), 
  y = x + 3 ∧ 
  2 * (2 * (x + y)) = x * y ∧ 
  x = 8 ∧ 
  y = 11 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l1172_117285


namespace NUMINAMATH_CALUDE_apple_banana_cost_l1172_117245

/-- The total cost of buying apples and bananas -/
def total_cost (a b : ℝ) : ℝ := 3 * a + 2 * b

/-- Theorem stating that the total cost of buying 3 kg of apples at 'a' yuan/kg
    and 2 kg of bananas at 'b' yuan/kg is (3a + 2b) yuan -/
theorem apple_banana_cost (a b : ℝ) :
  total_cost a b = 3 * a + 2 * b := by sorry

end NUMINAMATH_CALUDE_apple_banana_cost_l1172_117245


namespace NUMINAMATH_CALUDE_jack_reading_pages_l1172_117201

theorem jack_reading_pages (pages_per_booklet : ℕ) (number_of_booklets : ℕ) (total_pages : ℕ) :
  pages_per_booklet = 9 →
  number_of_booklets = 49 →
  total_pages = pages_per_booklet * number_of_booklets →
  total_pages = 441 :=
by sorry

end NUMINAMATH_CALUDE_jack_reading_pages_l1172_117201


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l1172_117204

theorem complex_fraction_evaluation : 
  (0.125 / 0.25 + (1 + 9/16) / 2.5) / 
  ((10 - 22 / 2.3) * 0.46 + 1.6) + 
  (17/20 + 1.9) * 0.5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l1172_117204


namespace NUMINAMATH_CALUDE_tenth_finger_is_two_l1172_117240

-- Define the functions f and g
def f : ℕ → ℕ
| 4 => 3
| 1 => 8
| 7 => 2
| _ => 0  -- Default case

def g : ℕ → ℕ
| 3 => 1
| 8 => 7
| 2 => 1
| _ => 0  -- Default case

-- Define a function that applies f and g alternately n times
def applyAlternately (n : ℕ) (start : ℕ) : ℕ :=
  match n with
  | 0 => start
  | n + 1 => if n % 2 = 0 then g (applyAlternately n start) else f (applyAlternately n start)

-- Theorem statement
theorem tenth_finger_is_two : applyAlternately 9 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tenth_finger_is_two_l1172_117240


namespace NUMINAMATH_CALUDE_rhombus_diagonals_property_inequality_or_equality_l1172_117209

-- Definition for rhombus properties
def diagonals_perpendicular (r : Type) : Prop := sorry
def diagonals_bisect (r : Type) : Prop := sorry

-- Theorem for the first compound proposition
theorem rhombus_diagonals_property :
  ∀ (r : Type), diagonals_perpendicular r ∧ diagonals_bisect r :=
sorry

-- Theorem for the second compound proposition
theorem inequality_or_equality : 2 < 3 ∨ 2 = 3 :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_property_inequality_or_equality_l1172_117209


namespace NUMINAMATH_CALUDE_lcm_18_45_l1172_117266

theorem lcm_18_45 : Nat.lcm 18 45 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_45_l1172_117266


namespace NUMINAMATH_CALUDE_prob_one_common_correct_l1172_117277

/-- The number of numbers in the lottery -/
def total_numbers : ℕ := 45

/-- The number of numbers each participant chooses -/
def chosen_numbers : ℕ := 6

/-- Calculates the probability of exactly one common number between two independently chosen combinations -/
def prob_one_common : ℚ :=
  (chosen_numbers : ℚ) * (Nat.choose (total_numbers - chosen_numbers) (chosen_numbers - 1) : ℚ) /
  (Nat.choose total_numbers chosen_numbers : ℚ)

/-- Theorem stating that the probability of exactly one common number is correct -/
theorem prob_one_common_correct :
  prob_one_common = (6 : ℚ) * (Nat.choose 39 5 : ℚ) / (Nat.choose 45 6 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_prob_one_common_correct_l1172_117277


namespace NUMINAMATH_CALUDE_good_bulbs_count_l1172_117258

def total_bulbs : ℕ := 10
def num_lamps : ℕ := 3
def prob_lighted : ℚ := 29/30

def num_good_bulbs : ℕ := 6

theorem good_bulbs_count :
  (1 : ℚ) - (Nat.choose (total_bulbs - num_good_bulbs) num_lamps : ℚ) / (Nat.choose total_bulbs num_lamps) = prob_lighted :=
sorry

end NUMINAMATH_CALUDE_good_bulbs_count_l1172_117258


namespace NUMINAMATH_CALUDE_complex_square_sum_l1172_117236

theorem complex_square_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (2 - i)^2 = a + b * i^3 → a + b = 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_l1172_117236


namespace NUMINAMATH_CALUDE_distance_from_origin_l1172_117241

theorem distance_from_origin (x y : ℝ) (h1 : y = 20) 
  (h2 : Real.sqrt ((x - 2)^2 + (y - 15)^2) = 15) (h3 : x > 2) : 
  Real.sqrt (x^2 + y^2) = Real.sqrt (604 + 40 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_distance_from_origin_l1172_117241


namespace NUMINAMATH_CALUDE_jerica_louis_age_ratio_l1172_117237

theorem jerica_louis_age_ratio :
  ∀ (jerica_age louis_age matilda_age : ℕ),
    louis_age = 14 →
    matilda_age = 35 →
    matilda_age = jerica_age + 7 →
    ∃ k : ℕ, jerica_age = k * louis_age →
    jerica_age / louis_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerica_louis_age_ratio_l1172_117237


namespace NUMINAMATH_CALUDE_probability_is_correct_l1172_117232

/-- Represents a unit cube within the larger cube -/
structure UnitCube where
  painted_faces : Nat
  deriving Repr

/-- Represents the larger 5x5x5 cube -/
def LargeCube : Type := Array UnitCube

/-- Creates a large cube with the specified painting configuration -/
def create_large_cube : LargeCube :=
  sorry

/-- Calculates the number of ways to choose 2 items from n items -/
def choose_two (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the probability of selecting one cube with two painted faces
    and one cube with no painted faces -/
def probability_two_and_none (cube : LargeCube) : Rat :=
  sorry

theorem probability_is_correct (cube : LargeCube) :
  probability_two_and_none (create_large_cube) = 187 / 3875 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_correct_l1172_117232


namespace NUMINAMATH_CALUDE_paving_cost_example_l1172_117274

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

theorem paving_cost_example :
  paving_cost 5.5 4 700 = 15400 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_example_l1172_117274


namespace NUMINAMATH_CALUDE_add_like_terms_l1172_117222

theorem add_like_terms (a : ℝ) : 3 * a + 2 * a = 5 * a := by
  sorry

end NUMINAMATH_CALUDE_add_like_terms_l1172_117222


namespace NUMINAMATH_CALUDE_record_4800_steps_l1172_117244

/-- The standard number of steps per day -/
def standard : ℕ := 5000

/-- Function to calculate the recorded steps -/
def recordedSteps (actualSteps : ℕ) : ℤ :=
  (actualSteps : ℤ) - standard

/-- Theorem stating that 4800 steps should be recorded as -200 -/
theorem record_4800_steps :
  recordedSteps 4800 = -200 := by sorry

end NUMINAMATH_CALUDE_record_4800_steps_l1172_117244


namespace NUMINAMATH_CALUDE_max_visible_cubes_9x9x9_l1172_117238

/-- Represents a cube formed by unit cubes -/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point -/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  let face_area := cube.size ^ 2
  let total_faces := 3 * face_area
  let shared_edges := 3 * (cube.size - 1)
  let corner_cube := 1
  total_faces - shared_edges + corner_cube

/-- Theorem stating that for a 9x9x9 cube, the maximum number of visible unit cubes is 220 -/
theorem max_visible_cubes_9x9x9 :
  max_visible_cubes ⟨9⟩ = 220 := by
  sorry

#eval max_visible_cubes ⟨9⟩

end NUMINAMATH_CALUDE_max_visible_cubes_9x9x9_l1172_117238


namespace NUMINAMATH_CALUDE_tan_105_degrees_l1172_117261

theorem tan_105_degrees : Real.tan (105 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l1172_117261


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1172_117299

/-- Given three consecutive odd integers where the sum of the first and third is 150,
    prove that the second integer is 75. -/
theorem consecutive_odd_integers_sum (n : ℤ) : 
  (∃ (a b c : ℤ), a + 2 = b ∧ b + 2 = c ∧ Odd a ∧ Odd b ∧ Odd c ∧ a + c = 150) →
  b = 75 := by
sorry


end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1172_117299


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1172_117247

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x + 8)*(x - 6) = -54 + k*x) ↔ 
  (k = 6*Real.sqrt 2 - 10 ∨ k = -6*Real.sqrt 2 - 10) := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1172_117247


namespace NUMINAMATH_CALUDE_bug_probability_after_12_meters_l1172_117267

/-- Probability of the bug being at vertex A after crawling n meters -/
def P (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => (1 - P n) / 3

/-- Edge length of the tetrahedron in meters -/
def edgeLength : ℕ := 2

/-- Number of edges traversed after 12 meters -/
def edgesTraversed : ℕ := 12 / edgeLength

theorem bug_probability_after_12_meters :
  P edgesTraversed = 44287 / 177147 := by sorry

end NUMINAMATH_CALUDE_bug_probability_after_12_meters_l1172_117267


namespace NUMINAMATH_CALUDE_min_repetitions_divisible_by_15_l1172_117283

def repeated_2002_plus_15 (n : ℕ) : ℕ :=
  2002 * (10^(4*n) - 1) / 9 * 10 + 15

theorem min_repetitions_divisible_by_15 :
  ∀ k : ℕ, k < 3 → ¬(repeated_2002_plus_15 k % 15 = 0) ∧
  repeated_2002_plus_15 3 % 15 = 0 :=
sorry

end NUMINAMATH_CALUDE_min_repetitions_divisible_by_15_l1172_117283


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_property_l1172_117251

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_correct : ∀ n, S n = (n : ℚ) * (a 1 + a n) / 2

/-- If S_2 / S_4 = 1/3 for an arithmetic sequence, then S_4 / S_8 = 3/10 -/
theorem arithmetic_sequence_ratio_property (seq : ArithmeticSequence) 
    (h : seq.S 2 / seq.S 4 = 1/3) : 
    seq.S 4 / seq.S 8 = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_property_l1172_117251


namespace NUMINAMATH_CALUDE_optimal_distribution_theorem_l1172_117229

/-- Represents the total value of the estate in talents -/
def estate_value : ℚ := 210

/-- Represents the fraction of the estate allocated to the son if only a son is born -/
def son_fraction : ℚ := 2/3

/-- Represents the fraction of the estate allocated to the daughter if only a daughter is born -/
def daughter_fraction : ℚ := 1/3

/-- Represents the optimal fraction of the estate allocated to the son when twins are born -/
def optimal_son_fraction : ℚ := 4/7

/-- Represents the optimal fraction of the estate allocated to the daughter when twins are born -/
def optimal_daughter_fraction : ℚ := 1/7

/-- Represents the optimal fraction of the estate allocated to the mother when twins are born -/
def optimal_mother_fraction : ℚ := 2/7

/-- Theorem stating that the optimal distribution is the best approximation of the will's conditions -/
theorem optimal_distribution_theorem :
  optimal_son_fraction + optimal_daughter_fraction + optimal_mother_fraction = 1 ∧
  optimal_son_fraction * estate_value + 
  optimal_daughter_fraction * estate_value + 
  optimal_mother_fraction * estate_value = estate_value ∧
  optimal_son_fraction > optimal_daughter_fraction ∧
  optimal_son_fraction < son_fraction ∧
  optimal_daughter_fraction < daughter_fraction :=
sorry

end NUMINAMATH_CALUDE_optimal_distribution_theorem_l1172_117229


namespace NUMINAMATH_CALUDE_monotonicity_and_minimum_l1172_117220

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (-x) * (a * x^2 + a + 1)

def f_derivative (a : ℝ) (x : ℝ) : ℝ := 
  Real.exp (-x) * (-a * x^2 + 2 * a * x - a - 1)

theorem monotonicity_and_minimum :
  (∀ x, a ≥ 0 → f_derivative a x < 0) ∧
  (a < 0 → ∃ r₁ r₂, r₁ < r₂ ∧ r₂ < 0 ∧
    (∀ x, x < r₁ → f_derivative a x > 0) ∧
    (∀ x, r₁ < x ∧ x < r₂ → f_derivative a x < 0) ∧
    (∀ x, x > r₂ → f_derivative a x > 0)) ∧
  (-1 < a ∧ a < 0 → ∀ x, 1 ≤ x ∧ x ≤ 2 → f a x ≥ f a 2) :=
by sorry

end

end NUMINAMATH_CALUDE_monotonicity_and_minimum_l1172_117220


namespace NUMINAMATH_CALUDE_b_age_is_27_l1172_117252

/-- The ages of four people A, B, C, and D. -/
structure Ages where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The conditions of the problem. -/
def problem_conditions (ages : Ages) : Prop :=
  (ages.a + ages.b + ages.c + ages.d) / 4 = 28 ∧
  (ages.a + ages.c) / 2 = 29 ∧
  (2 * ages.b + 3 * ages.d) / 5 = 27 ∧
  ages.a = 1.1 * (ages.a / 1.1) ∧
  ages.c = 1.1 * (ages.c / 1.1) ∧
  ages.b = 1.15 * (ages.b / 1.15) ∧
  ages.d = 1.15 * (ages.d / 1.15)

/-- The theorem stating that given the problem conditions, B's age is 27. -/
theorem b_age_is_27 (ages : Ages) (h : problem_conditions ages) : ages.b = 27 := by
  sorry

end NUMINAMATH_CALUDE_b_age_is_27_l1172_117252


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l1172_117224

theorem correct_average_after_error_correction 
  (n : ℕ) 
  (incorrect_avg : ℚ) 
  (incorrect_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 16 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 55 →
  (n * incorrect_avg - incorrect_num + correct_num) / n = 19 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l1172_117224


namespace NUMINAMATH_CALUDE_reachable_region_characterization_l1172_117288

/-- A particle's position in a 2D plane -/
structure Particle where
  x : ℝ
  y : ℝ

/-- The speed of the particle along the x-axis -/
def x_speed : ℝ := 2

/-- The speed of the particle elsewhere -/
def other_speed : ℝ := 1

/-- The time limit for the particle's movement -/
def time_limit : ℝ := 1

/-- Check if a point is within the reachable region -/
def is_reachable (p : Particle) : Prop :=
  let o := Particle.mk 0 0
  let a := Particle.mk (1/2) (Real.sqrt 3 / 2)
  let b := Particle.mk 2 0
  let c := Particle.mk 1 0
  (p.x ≥ 0 ∧ p.y ≥ 0) ∧  -- First quadrant
  ((p.x ≤ 2 ∧ p.y ≤ (Real.sqrt 3 * (1 - p.x/2))) ∨  -- Triangle OAB
   (p.x^2 + p.y^2 ≤ 1 ∧ p.y ≥ 0 ∧ p.x ≥ p.y/Real.sqrt 3))  -- Sector OAC

/-- The main theorem stating that a point is reachable if and only if it's in the defined region -/
theorem reachable_region_characterization (p : Particle) :
  (∃ (path : ℝ → Particle), path 0 = Particle.mk 0 0 ∧
    (∀ t, 0 ≤ t ∧ t ≤ time_limit →
      (path t).x^2 + (path t).y^2 ≤ (x_speed * t)^2 ∨
      (path t).x^2 + (path t).y^2 ≤ (other_speed * t)^2) ∧
    path time_limit = p) ↔
  is_reachable p :=
sorry

end NUMINAMATH_CALUDE_reachable_region_characterization_l1172_117288


namespace NUMINAMATH_CALUDE_triathlete_average_speed_l1172_117215

-- Define the problem parameters
def total_distance : Real := 8
def running_distance : Real := 4
def swimming_distance : Real := 4
def running_speed : Real := 10
def swimming_speed : Real := 6

-- Define the theorem
theorem triathlete_average_speed :
  let running_time := running_distance / running_speed
  let swimming_time := swimming_distance / swimming_speed
  let total_time := running_time + swimming_time
  let average_speed_mph := total_distance / total_time
  let average_speed_mpm := average_speed_mph / 60
  average_speed_mpm = 0.125 := by sorry

end NUMINAMATH_CALUDE_triathlete_average_speed_l1172_117215


namespace NUMINAMATH_CALUDE_shadow_problem_l1172_117257

/-- Given a cube with edge length 2 cm and a point light source x cm above an upper vertex,
    if the shadow area (excluding the area beneath the cube) is 192 sq cm,
    then the greatest integer not exceeding 1000x is 12000. -/
theorem shadow_problem (x : ℝ) : 
  let cube_edge : ℝ := 2
  let shadow_area : ℝ := 192
  let total_shadow_area : ℝ := shadow_area + cube_edge^2
  let shadow_side : ℝ := Real.sqrt total_shadow_area
  x = cube_edge * (shadow_side - cube_edge) / cube_edge →
  Int.floor (1000 * x) = 12000 := by
sorry

end NUMINAMATH_CALUDE_shadow_problem_l1172_117257


namespace NUMINAMATH_CALUDE_tree_planting_problem_l1172_117298

theorem tree_planting_problem (a o c m : ℕ) 
  (ha : a = 47)
  (ho : o = 27)
  (hm : m = a * o)
  (hc : c = a - 15) :
  a = 47 ∧ o = 27 ∧ c = 32 ∧ m = 1269 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_problem_l1172_117298


namespace NUMINAMATH_CALUDE_rectangle_area_l1172_117297

/-- The area of a rectangle with length twice its width and perimeter equal to a triangle with sides 7, 10, and 11 is 392/9 -/
theorem rectangle_area (w : ℝ) (h : 2 * (2 * w + w) = 7 + 10 + 11) : w * (2 * w) = 392 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1172_117297


namespace NUMINAMATH_CALUDE_fruit_boxes_problem_l1172_117294

theorem fruit_boxes_problem (total_pears : ℕ) : 
  (∃ (fruits_per_box : ℕ), 
    fruits_per_box = 12 + total_pears / 9 ∧ 
    fruits_per_box = (12 + total_pears) / 3 ∧
    fruits_per_box = 16) := by
  sorry

end NUMINAMATH_CALUDE_fruit_boxes_problem_l1172_117294


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l1172_117234

/-- A rectangular garden with length three times its width and area 507 square meters has a width of 13 meters. -/
theorem rectangular_garden_width (width length area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 507 →
  width = 13 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l1172_117234


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l1172_117239

/-- Given a train crossing a bridge, calculate the length of the bridge -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 140 ∧ train_speed_kmh = 45 ∧ crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 235 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l1172_117239


namespace NUMINAMATH_CALUDE_l_shaped_paper_area_l1172_117290

/-- The area of an "L" shaped paper formed by cutting rectangles from a larger rectangle --/
theorem l_shaped_paper_area (original_length original_width cut1_length cut1_width cut2_length cut2_width : ℕ) 
  (h1 : original_length = 10)
  (h2 : original_width = 7)
  (h3 : cut1_length = 3)
  (h4 : cut1_width = 2)
  (h5 : cut2_length = 2)
  (h6 : cut2_width = 4) :
  original_length * original_width - cut1_length * cut1_width - cut2_length * cut2_width = 56 := by
  sorry

end NUMINAMATH_CALUDE_l_shaped_paper_area_l1172_117290


namespace NUMINAMATH_CALUDE_two_ones_in_twelve_dice_l1172_117228

def probability_two_ones (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem two_ones_in_twelve_dice :
  probability_two_ones 12 2 (1/6) = (66 * 5^10 : ℚ) / (36 * 6^10 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_two_ones_in_twelve_dice_l1172_117228


namespace NUMINAMATH_CALUDE_locus_of_symmetric_points_l1172_117269

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) -/
def origin : Point2D := ⟨0, 0⟩

/-- Check if a point is on the x-axis -/
def isOnXAxis (p : Point2D) : Prop := p.y = 0

/-- Check if a point is on the y-axis -/
def isOnYAxis (p : Point2D) : Prop := p.x = 0

/-- Check if three points form a right angle -/
def isRightAngle (p1 p2 p3 : Point2D) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

/-- The line symmetric to a point with respect to the coordinate axes -/
def symmetricLine (m : Point2D) : Set Point2D :=
  {n : Point2D | n.x * m.y = n.y * m.x}

/-- The main theorem -/
theorem locus_of_symmetric_points (m : Point2D) 
  (h1 : m ≠ origin) 
  (h2 : ¬isOnXAxis m) 
  (h3 : ¬isOnYAxis m) :
  ∀ (p q : Point2D), 
    isOnXAxis p → isOnYAxis q → isRightAngle p m q →
    ∃ (n : Point2D), n ∈ symmetricLine m :=
by sorry

end NUMINAMATH_CALUDE_locus_of_symmetric_points_l1172_117269


namespace NUMINAMATH_CALUDE_jasons_tip_is_two_dollars_l1172_117211

/-- Calculates the tip amount given the check amount, tax rate, and customer payment. -/
def calculate_tip (check_amount : ℝ) (tax_rate : ℝ) (customer_payment : ℝ) : ℝ :=
  let total_with_tax := check_amount * (1 + tax_rate)
  customer_payment - total_with_tax

/-- Proves that given the specific conditions, Jason's tip is $2.00 -/
theorem jasons_tip_is_two_dollars :
  calculate_tip 15 0.2 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jasons_tip_is_two_dollars_l1172_117211


namespace NUMINAMATH_CALUDE_eraser_difference_l1172_117271

/-- Proves that the difference between Rachel's erasers and one-half of Tanya's red erasers is 5 -/
theorem eraser_difference (tanya_total : ℕ) (tanya_red : ℕ) (rachel : ℕ) 
  (h1 : tanya_total = 20)
  (h2 : tanya_red = tanya_total / 2)
  (h3 : rachel = tanya_red) :
  rachel - tanya_red / 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_eraser_difference_l1172_117271


namespace NUMINAMATH_CALUDE_integer_ratio_problem_l1172_117231

theorem integer_ratio_problem (s l : ℕ) : 
  s = 32 →
  ∃ k : ℕ, l = k * s →
  s + l = 96 →
  l / s = 2 :=
by sorry

end NUMINAMATH_CALUDE_integer_ratio_problem_l1172_117231


namespace NUMINAMATH_CALUDE_tree_planting_seedlings_l1172_117205

theorem tree_planting_seedlings : 
  ∃ (x : ℕ), 
    (∃ (n : ℕ), x - 6 = 5 * n) ∧ 
    (∃ (m : ℕ), x + 9 = 6 * m) ∧ 
    x = 81 := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_seedlings_l1172_117205


namespace NUMINAMATH_CALUDE_function_bound_l1172_117292

theorem function_bound (x : ℝ) : 
  1/2 ≤ (x^2 + x + 1) / (x^2 + 1) ∧ (x^2 + x + 1) / (x^2 + 1) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l1172_117292


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1172_117225

theorem sphere_volume_from_surface_area (O : Set ℝ) (surface_area : ℝ) (volume : ℝ) :
  (∃ (r : ℝ), surface_area = 4 * Real.pi * r^2) →
  surface_area = 4 * Real.pi →
  (∃ (r : ℝ), volume = (4 / 3) * Real.pi * r^3) →
  volume = (4 / 3) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1172_117225


namespace NUMINAMATH_CALUDE_log_850_between_consecutive_integers_l1172_117263

theorem log_850_between_consecutive_integers :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 850 / Real.log 10 ∧ Real.log 850 / Real.log 10 < b ∧ a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_850_between_consecutive_integers_l1172_117263


namespace NUMINAMATH_CALUDE_probability_three_blue_pens_l1172_117273

def total_pens : ℕ := 15
def blue_pens : ℕ := 8
def red_pens : ℕ := 7
def num_trials : ℕ := 7
def num_blue_picks : ℕ := 3

def prob_blue : ℚ := blue_pens / total_pens
def prob_red : ℚ := red_pens / total_pens

def binomial_coefficient (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

theorem probability_three_blue_pens :
  (binomial_coefficient num_trials num_blue_picks : ℚ) *
  (prob_blue ^ num_blue_picks) *
  (prob_red ^ (num_trials - num_blue_picks)) =
  43025920 / 170859375 := by sorry

end NUMINAMATH_CALUDE_probability_three_blue_pens_l1172_117273


namespace NUMINAMATH_CALUDE_shaded_area_is_24_5_l1172_117227

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  bottomLeft : Point
  baseLength : ℝ

/-- Calculates the area of the shaded region -/
def shadedArea (square : Square) (triangle : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating the area of the shaded region -/
theorem shaded_area_is_24_5 (square : Square) (triangle : IsoscelesTriangle) :
  square.bottomLeft = Point.mk 0 0 →
  square.sideLength = 7 →
  triangle.bottomLeft = Point.mk 7 0 →
  triangle.baseLength = 7 →
  shadedArea square triangle = 24.5 :=
by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_24_5_l1172_117227


namespace NUMINAMATH_CALUDE_power_of_two_plus_two_eq_rational_square_l1172_117254

theorem power_of_two_plus_two_eq_rational_square (r : ℚ) :
  (∃ z : ℤ, 2^z + 2 = r^2) ↔ (r = 2 ∨ r = -2 ∨ r = 3/2 ∨ r = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_plus_two_eq_rational_square_l1172_117254


namespace NUMINAMATH_CALUDE_smallest_divisor_k_divisibility_at_126_smallest_k_is_126_l1172_117212

def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_divisor_k : ∀ k : ℕ, k > 0 → (∀ z : ℂ, f z ∣ (z^k - 1)) → k ≥ 126 :=
by sorry

theorem divisibility_at_126 : ∀ z : ℂ, f z ∣ (z^126 - 1) :=
by sorry

theorem smallest_k_is_126 : (∀ z : ℂ, f z ∣ (z^126 - 1)) ∧ 
  (∀ k : ℕ, k > 0 → k < 126 → ∃ z : ℂ, ¬(f z ∣ (z^k - 1))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisor_k_divisibility_at_126_smallest_k_is_126_l1172_117212


namespace NUMINAMATH_CALUDE_exists_m_f_even_l1172_117219

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = x^2 + mx -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x

/-- There exists an m ∈ ℝ such that f(x) = x^2 + mx is an even function -/
theorem exists_m_f_even : ∃ m : ℝ, IsEven (f m) := by
  sorry

end NUMINAMATH_CALUDE_exists_m_f_even_l1172_117219
