import Mathlib

namespace NUMINAMATH_CALUDE_hospital_bill_ambulance_cost_l621_62139

theorem hospital_bill_ambulance_cost 
  (total_bill : ℝ)
  (medication_percentage : ℝ)
  (overnight_percentage : ℝ)
  (food_cost : ℝ)
  (h1 : total_bill = 5000)
  (h2 : medication_percentage = 0.5)
  (h3 : overnight_percentage = 0.25)
  (h4 : food_cost = 175) :
  let medication_cost := medication_percentage * total_bill
  let remaining_after_medication := total_bill - medication_cost
  let overnight_cost := overnight_percentage * remaining_after_medication
  let remaining_after_overnight := remaining_after_medication - overnight_cost
  let ambulance_cost := remaining_after_overnight - food_cost
  ambulance_cost = 1700 := by sorry

end NUMINAMATH_CALUDE_hospital_bill_ambulance_cost_l621_62139


namespace NUMINAMATH_CALUDE_three_points_with_midpoint_l621_62196

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points on a line
structure Point where
  position : ℝ
  color : Color

-- Define the theorem
theorem three_points_with_midpoint
  (line : Set Point)
  (h_nonempty : Set.Nonempty line)
  (h_two_colors : ∃ p q : Point, p ∈ line ∧ q ∈ line ∧ p.color ≠ q.color)
  (h_one_color : ∀ p : Point, p ∈ line → (p.color = Color.Red ∨ p.color = Color.Blue)) :
  ∃ p q r : Point,
    p ∈ line ∧ q ∈ line ∧ r ∈ line ∧
    p.color = q.color ∧ q.color = r.color ∧
    q.position = (p.position + r.position) / 2 :=
sorry

end NUMINAMATH_CALUDE_three_points_with_midpoint_l621_62196


namespace NUMINAMATH_CALUDE_f_of_4_equals_15_l621_62127

/-- A function f(x) = cx^2 + dx + 3 satisfying f(1) = 3 and f(2) = 5 -/
def f (c d : ℝ) (x : ℝ) : ℝ := c * x^2 + d * x + 3

/-- The theorem stating that f(4) = 15 given the conditions -/
theorem f_of_4_equals_15 (c d : ℝ) :
  f c d 1 = 3 → f c d 2 = 5 → f c d 4 = 15 := by
  sorry

#check f_of_4_equals_15

end NUMINAMATH_CALUDE_f_of_4_equals_15_l621_62127


namespace NUMINAMATH_CALUDE_markers_count_l621_62124

/-- Given a ratio of pens : pencils : markers as 2 : 2 : 5, and 10 pens, the number of markers is 25. -/
theorem markers_count (pens pencils markers : ℕ) : 
  pens = 10 → 
  pens * 5 = markers * 2 → 
  markers = 25 := by
  sorry

end NUMINAMATH_CALUDE_markers_count_l621_62124


namespace NUMINAMATH_CALUDE_average_height_of_trees_l621_62115

theorem average_height_of_trees (elm_height oak_height pine_height : ℚ) : 
  elm_height = 35 / 3 →
  oak_height = 107 / 6 →
  pine_height = 31 / 2 →
  (elm_height + oak_height + pine_height) / 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_height_of_trees_l621_62115


namespace NUMINAMATH_CALUDE_better_misspellings_l621_62136

/-- The word to be considered -/
def word : String := "better"

/-- The number of distinct letters in the word -/
def distinct_letters : Nat := 4

/-- The total number of letters in the word -/
def total_letters : Nat := 6

/-- The number of repeated letters in the word -/
def repeated_letters : Nat := 2

/-- The number of repetitions for each repeated letter -/
def repetitions : Nat := 2

/-- The number of misspellings of the word "better" -/
def misspellings : Nat := 179

theorem better_misspellings :
  (Nat.factorial total_letters / (Nat.factorial repetitions ^ repeated_letters)) - 1 = misspellings :=
sorry

end NUMINAMATH_CALUDE_better_misspellings_l621_62136


namespace NUMINAMATH_CALUDE_area_covered_by_strips_l621_62193

/-- The area covered by overlapping rectangular strips -/
theorem area_covered_by_strips (n : ℕ) (length width overlap_length : ℝ) : 
  n = 5 → 
  length = 12 → 
  width = 1 → 
  overlap_length = 2 → 
  (n : ℝ) * length * width - (n.choose 2 : ℝ) * overlap_length * width = 40 := by
  sorry

#check area_covered_by_strips

end NUMINAMATH_CALUDE_area_covered_by_strips_l621_62193


namespace NUMINAMATH_CALUDE_find_other_number_l621_62176

theorem find_other_number (a b : ℤ) : 
  3 * a + 2 * b = 105 → (a = 15 ∨ b = 15) → (a = 30 ∨ b = 30) := by
sorry

end NUMINAMATH_CALUDE_find_other_number_l621_62176


namespace NUMINAMATH_CALUDE_shoe_problem_contradiction_l621_62175

theorem shoe_problem_contradiction (becky bobby bonny : ℕ) : 
  (bonny = 2 * becky - 5) →
  (bobby = 3 * becky) →
  (bonny = bobby) →
  False :=
by sorry

end NUMINAMATH_CALUDE_shoe_problem_contradiction_l621_62175


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l621_62131

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ r₁ r₂ : ℝ, (r₁ + r₂ = -p ∧ r₁ * r₂ = m) ∧
               (3 * r₁ + 3 * r₂ = -m ∧ 9 * r₁ * r₂ = n)) →
  n / p = 27 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l621_62131


namespace NUMINAMATH_CALUDE_sum_of_squares_l621_62101

theorem sum_of_squares (x y z : ℕ+) : 
  (x : ℕ) + y + z = 24 →
  Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10 →
  ∃! s : ℕ, s = x^2 + y^2 + z^2 ∧ s = 296 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l621_62101


namespace NUMINAMATH_CALUDE_equation1_representation_equation2_representation_l621_62138

-- Define the equations
def equation1 (x y : ℝ) : Prop := 4 * x^2 + 8 * y^2 + 8 * y * |y| = 1
def equation2 (x y : ℝ) : Prop := 2 * x^2 - 4 * x + 2 + 2 * (x - 1) * |x - 1| + 8 * y^2 - 8 * y * |y| = 1

-- Define the regions for equation1
def upper_ellipse (x y : ℝ) : Prop := y ≥ 0 ∧ 4 * x^2 + 16 * y^2 = 1
def vertical_lines (x y : ℝ) : Prop := y < 0 ∧ (x = 1/2 ∨ x = -1/2)

-- Define the regions for equation2
def elliptic_part (x y : ℝ) : Prop := x ≥ 1 ∧ 4 * (x - 1)^2 + 16 * y^2 = 1
def vertical_section (x y : ℝ) : Prop := x < 1 ∧ y = -1/4

-- Theorem statements
theorem equation1_representation :
  ∀ x y : ℝ, equation1 x y ↔ (upper_ellipse x y ∨ vertical_lines x y) :=
sorry

theorem equation2_representation :
  ∀ x y : ℝ, equation2 x y ↔ (elliptic_part x y ∨ vertical_section x y) :=
sorry

end NUMINAMATH_CALUDE_equation1_representation_equation2_representation_l621_62138


namespace NUMINAMATH_CALUDE_taxi_cost_formula_correct_l621_62160

/-- Represents the total cost in dollars for a taxi ride -/
def taxiCost (T : ℕ) : ℤ :=
  10 + 5 * T - 10 * (if T > 5 then 1 else 0)

/-- Theorem stating the correctness of the taxi cost formula -/
theorem taxi_cost_formula_correct (T : ℕ) :
  taxiCost T = 10 + 5 * T - 10 * (if T > 5 then 1 else 0) := by
  sorry

#check taxi_cost_formula_correct

end NUMINAMATH_CALUDE_taxi_cost_formula_correct_l621_62160


namespace NUMINAMATH_CALUDE_largest_package_size_l621_62195

theorem largest_package_size (lucy_markers emma_markers : ℕ) 
  (h1 : lucy_markers = 54)
  (h2 : emma_markers = 36) :
  Nat.gcd lucy_markers emma_markers = 18 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l621_62195


namespace NUMINAMATH_CALUDE_last_week_viewers_correct_l621_62134

/-- The number of people who watched the baseball games last week -/
def last_week_viewers : ℕ := 200

/-- The number of people who watched the second game this week -/
def second_game_viewers : ℕ := 80

/-- The number of people who watched the first game this week -/
def first_game_viewers : ℕ := second_game_viewers - 20

/-- The number of people who watched the third game this week -/
def third_game_viewers : ℕ := second_game_viewers + 15

/-- The total number of people who watched the games this week -/
def this_week_total : ℕ := first_game_viewers + second_game_viewers + third_game_viewers

/-- The difference in viewers between this week and last week -/
def viewer_difference : ℕ := 35

theorem last_week_viewers_correct : 
  last_week_viewers = this_week_total - viewer_difference := by
  sorry

end NUMINAMATH_CALUDE_last_week_viewers_correct_l621_62134


namespace NUMINAMATH_CALUDE_annie_total_blocks_l621_62149

/-- The total number of blocks Annie traveled -/
def total_blocks : ℕ :=
  let house_to_bus := 5
  let bus_to_train := 7
  let train_to_friend := 10
  let friend_to_coffee := 4
  2 * (house_to_bus + bus_to_train + train_to_friend) + 2 * friend_to_coffee

/-- Theorem stating that Annie traveled 52 blocks in total -/
theorem annie_total_blocks : total_blocks = 52 := by
  sorry

end NUMINAMATH_CALUDE_annie_total_blocks_l621_62149


namespace NUMINAMATH_CALUDE_mirror_image_properties_l621_62137

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define mirror image operations
def mirrorY (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

def mirrorX (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

def mirrorOrigin (p : Point2D) : Point2D :=
  { x := -p.x, y := -p.y }

def mirrorYEqualsX (p : Point2D) : Point2D :=
  { x := p.y, y := p.x }

def mirrorYEqualsNegX (p : Point2D) : Point2D :=
  { x := -p.y, y := -p.x }

-- Theorem stating the mirror image properties
theorem mirror_image_properties (p : Point2D) :
  (mirrorY p = { x := -p.x, y := p.y }) ∧
  (mirrorX p = { x := p.x, y := -p.y }) ∧
  (mirrorOrigin p = { x := -p.x, y := -p.y }) ∧
  (mirrorYEqualsX p = { x := p.y, y := p.x }) ∧
  (mirrorYEqualsNegX p = { x := -p.y, y := -p.x }) :=
by sorry

end NUMINAMATH_CALUDE_mirror_image_properties_l621_62137


namespace NUMINAMATH_CALUDE_units_digit_of_quotient_l621_62122

theorem units_digit_of_quotient (n : ℕ) : (2^2023 + 3^2023) % 7 = 0 → 
  (2^2023 + 3^2023) / 7 % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_quotient_l621_62122


namespace NUMINAMATH_CALUDE_range_of_b_minus_a_l621_62164

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem range_of_b_minus_a (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, f x ∈ Set.Icc (-1) 3) →
  (∃ x ∈ Set.Icc a b, f x = -1) →
  (∃ x ∈ Set.Icc a b, f x = 3) →
  b - a ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_minus_a_l621_62164


namespace NUMINAMATH_CALUDE_sarah_tuesday_pencils_l621_62117

/-- The number of pencils Sarah bought on Monday -/
def monday_pencils : ℕ := 20

/-- The number of pencils Sarah bought on Tuesday -/
def tuesday_pencils : ℕ := 18

/-- The total number of pencils Sarah has -/
def total_pencils : ℕ := 92

/-- Theorem: Sarah bought 18 pencils on Tuesday -/
theorem sarah_tuesday_pencils :
  monday_pencils + tuesday_pencils + 3 * tuesday_pencils = total_pencils :=
by sorry

end NUMINAMATH_CALUDE_sarah_tuesday_pencils_l621_62117


namespace NUMINAMATH_CALUDE_equation_roots_l621_62130

def equation (x : ℝ) : ℝ := x * (x + 2)^2 * (3 - x) * (5 + x)

theorem equation_roots : 
  {x : ℝ | equation x = 0} = {0, -2, 3, -5} := by sorry

end NUMINAMATH_CALUDE_equation_roots_l621_62130


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l621_62190

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((3 / 8 : ℚ) + (5 / 9 : ℚ)) = 67 / 144 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l621_62190


namespace NUMINAMATH_CALUDE_lemniscate_symmetric_origin_lemniscate_max_distance_squared_lemniscate_unique_equidistant_point_l621_62109

-- Define the lemniscate curve
def Lemniscate (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p; ((x + a)^2 + y^2) * ((x - a)^2 + y^2) = a^4}

-- Statement 1: Symmetry with respect to the origin
theorem lemniscate_symmetric_origin (a : ℝ) (h : a > 0) :
  ∀ (p : ℝ × ℝ), p ∈ Lemniscate a ↔ (-p.1, -p.2) ∈ Lemniscate a :=
sorry

-- Statement 2: Maximum value of |PO|^2 - a^2
theorem lemniscate_max_distance_squared (a : ℝ) (h : a > 0) :
  ∃ (p : ℝ × ℝ), p ∈ Lemniscate a ∧
    ∀ (q : ℝ × ℝ), q ∈ Lemniscate a → (p.1^2 + p.2^2) - a^2 ≥ (q.1^2 + q.2^2) - a^2 ∧
    (p.1^2 + p.2^2) - a^2 = a^2 :=
sorry

-- Statement 3: Unique point equidistant from focal points
theorem lemniscate_unique_equidistant_point (a : ℝ) (h : a > 0) :
  ∃! (p : ℝ × ℝ), p ∈ Lemniscate a ∧
    (p.1 + a)^2 + p.2^2 = (p.1 - a)^2 + p.2^2 :=
sorry

end NUMINAMATH_CALUDE_lemniscate_symmetric_origin_lemniscate_max_distance_squared_lemniscate_unique_equidistant_point_l621_62109


namespace NUMINAMATH_CALUDE_percentage_problem_l621_62192

theorem percentage_problem (P : ℝ) : P = 30 :=
by
  -- Define the condition from the problem
  have h1 : P / 100 * 100 = 50 / 100 * 40 + 10 := by sorry
  
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l621_62192


namespace NUMINAMATH_CALUDE_birds_and_storks_l621_62121

theorem birds_and_storks (initial_birds : ℕ) (storks : ℕ) (additional_birds : ℕ) : 
  initial_birds = 3 → storks = 4 → additional_birds = 2 →
  (initial_birds + additional_birds) - storks = 1 := by
  sorry

end NUMINAMATH_CALUDE_birds_and_storks_l621_62121


namespace NUMINAMATH_CALUDE_reciprocal_sum_l621_62110

theorem reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 = b^2 + b*c) (h2 : b^2 = c^2 + a*c) : 
  1/c = 1/a + 1/b := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_l621_62110


namespace NUMINAMATH_CALUDE_apples_per_box_l621_62156

/-- Proves that the number of apples per box is 50 given the total number of apples,
    the desired amount to take home, and the price per box. -/
theorem apples_per_box
  (total_apples : ℕ)
  (take_home_amount : ℕ)
  (price_per_box : ℕ)
  (h1 : total_apples = 10000)
  (h2 : take_home_amount = 7000)
  (h3 : price_per_box = 35) :
  total_apples / (take_home_amount / price_per_box) = 50 := by
  sorry

#check apples_per_box

end NUMINAMATH_CALUDE_apples_per_box_l621_62156


namespace NUMINAMATH_CALUDE_binary_multiplication_example_l621_62159

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNum := List Nat

/-- Converts a decimal number to its binary representation -/
def to_binary (n : Nat) : BinaryNum :=
  sorry

/-- Converts a binary number to its decimal representation -/
def to_decimal (b : BinaryNum) : Nat :=
  sorry

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNum) : BinaryNum :=
  sorry

theorem binary_multiplication_example :
  let a : BinaryNum := [1, 0, 1, 0, 1]  -- 10101₂
  let b : BinaryNum := [1, 0, 1]        -- 101₂
  let result : BinaryNum := [1, 1, 0, 1, 0, 0, 1]  -- 1101001₂
  binary_multiply a b = result :=
by sorry

end NUMINAMATH_CALUDE_binary_multiplication_example_l621_62159


namespace NUMINAMATH_CALUDE_modulus_of_z_l621_62118

open Complex

theorem modulus_of_z (z : ℂ) (h : (1 - I) * z = 2 * I) : abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l621_62118


namespace NUMINAMATH_CALUDE_max_students_distribution_l621_62180

def stationery_A : ℕ := 38
def stationery_B : ℕ := 78
def stationery_C : ℕ := 128

def remaining_A : ℕ := 2
def remaining_B : ℕ := 6
def remaining_C : ℕ := 20

def distributed_A : ℕ := stationery_A - remaining_A
def distributed_B : ℕ := stationery_B - remaining_B
def distributed_C : ℕ := stationery_C - remaining_C

theorem max_students_distribution :
  ∃ (n : ℕ), n > 0 ∧ 
    distributed_A % n = 0 ∧
    distributed_B % n = 0 ∧
    distributed_C % n = 0 ∧
    ∀ (m : ℕ), m > n →
      (distributed_A % m ≠ 0 ∨
       distributed_B % m ≠ 0 ∨
       distributed_C % m ≠ 0) →
    n = 36 :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l621_62180


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l621_62169

/-- The distance between the foci of a hyperbola with equation y²/25 - x²/16 = 1 is 2√41 -/
theorem hyperbola_foci_distance : 
  let a : ℝ := 5
  let b : ℝ := 4
  let c : ℝ := Real.sqrt (a^2 + b^2)
  2 * c = 2 * Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l621_62169


namespace NUMINAMATH_CALUDE_ratio_of_sums_is_301_480_l621_62199

/-- Calculate the sum of an arithmetic sequence -/
def sum_arithmetic (a1 : ℚ) (d : ℚ) (an : ℚ) : ℚ :=
  let n : ℚ := (an - a1) / d + 1
  n * (a1 + an) / 2

/-- The ratio of sums of two specific arithmetic sequences -/
def ratio_of_sums : ℚ :=
  (sum_arithmetic 2 3 41) / (sum_arithmetic 4 4 60)

theorem ratio_of_sums_is_301_480 : ratio_of_sums = 301 / 480 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sums_is_301_480_l621_62199


namespace NUMINAMATH_CALUDE_parabola_directrix_l621_62129

/-- Given a parabola y = -3x^2 + 6x - 7, its directrix is y = -47/12 -/
theorem parabola_directrix (x y : ℝ) : 
  y = -3 * x^2 + 6 * x - 7 → 
  ∃ (k : ℝ), k = -47/12 ∧ (∀ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y → 
    (p.1 - 1)^2 + (p.2 - k)^2 = (p.2 + 4)^2 / 9) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l621_62129


namespace NUMINAMATH_CALUDE_victor_stickers_l621_62153

theorem victor_stickers (flower_stickers : ℕ) (total_stickers : ℕ) (animal_stickers : ℕ) : 
  flower_stickers = 8 → 
  total_stickers = 14 → 
  animal_stickers < flower_stickers → 
  flower_stickers + animal_stickers = total_stickers →
  animal_stickers = 6 := by
sorry

end NUMINAMATH_CALUDE_victor_stickers_l621_62153


namespace NUMINAMATH_CALUDE_tower_count_remainder_l621_62163

/-- Represents a cube with an edge length --/
structure Cube where
  edge_length : ℕ

/-- Represents a tower of cubes --/
inductive Tower : Type
  | empty : Tower
  | cons : Cube → Tower → Tower

/-- Checks if a tower is valid according to the rules --/
def is_valid_tower : Tower → Bool
  | Tower.empty => true
  | Tower.cons c Tower.empty => true
  | Tower.cons c1 (Tower.cons c2 t) =>
    c1.edge_length ≤ c2.edge_length + 3 && is_valid_tower (Tower.cons c2 t)

/-- The set of cubes with edge lengths from 1 to 10 --/
def cube_set : List Cube :=
  List.map (λ k => ⟨k⟩) (List.range 10)

/-- Counts the number of valid towers that can be constructed --/
def count_valid_towers (cubes : List Cube) : ℕ :=
  sorry  -- Implementation details omitted

/-- The main theorem --/
theorem tower_count_remainder (U : ℕ) :
  U = count_valid_towers cube_set →
  U % 1000 = 536 :=
sorry

end NUMINAMATH_CALUDE_tower_count_remainder_l621_62163


namespace NUMINAMATH_CALUDE_at_least_one_equation_has_solution_l621_62120

theorem at_least_one_equation_has_solution (a b c : ℝ) : 
  ¬(c^2 > a^2 + b^2 ∧ b^2 - 16*a*c < 0) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_equation_has_solution_l621_62120


namespace NUMINAMATH_CALUDE_goose_egg_hatching_rate_l621_62167

theorem goose_egg_hatching_rate : 
  ∀ (total_eggs : ℕ) (hatched_eggs : ℕ),
    (hatched_eggs : ℚ) / total_eggs = 1 →
    (3 : ℚ) / 4 * ((2 : ℚ) / 5 * hatched_eggs) = 180 →
    hatched_eggs ≤ total_eggs →
    (hatched_eggs : ℚ) / total_eggs = 1 := by
  sorry

end NUMINAMATH_CALUDE_goose_egg_hatching_rate_l621_62167


namespace NUMINAMATH_CALUDE_sector_central_angle_l621_62106

/-- Given a sector with perimeter 4 and area 1, its central angle is 2 radians -/
theorem sector_central_angle (r l : ℝ) (h1 : l + 2*r = 4) (h2 : (1/2)*l*r = 1) :
  l / r = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l621_62106


namespace NUMINAMATH_CALUDE_angle_between_skew_medians_l621_62123

/-- A regular tetrahedron with edge length a -/
structure RegularTetrahedron (a : ℝ) where
  edge_length : a > 0

/-- A median of a face in a regular tetrahedron -/
structure FaceMedian (t : RegularTetrahedron a) where
  start_vertex : ℝ × ℝ × ℝ
  end_point : ℝ × ℝ × ℝ

/-- The angle between two vectors in ℝ³ -/
def angle_between (v w : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Two face medians are skew if they're not on the same face -/
def are_skew_medians (m1 m2 : FaceMedian t) : Prop := sorry

theorem angle_between_skew_medians (t : RegularTetrahedron a) 
  (m1 m2 : FaceMedian t) (h : are_skew_medians m1 m2) : 
  angle_between (m1.end_point - m1.start_vertex) (m2.end_point - m2.start_vertex) = Real.arccos (1/6) := by
  sorry

end NUMINAMATH_CALUDE_angle_between_skew_medians_l621_62123


namespace NUMINAMATH_CALUDE_polynomial_remainder_l621_62140

def polynomial (x : ℝ) : ℝ := 3*x^8 - x^7 - 7*x^5 + 3*x^3 + 4*x^2 - 12*x - 1

def divisor (x : ℝ) : ℝ := 3*x - 9

theorem polynomial_remainder :
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ),
    polynomial x = (divisor x) * (q x) + 15951 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l621_62140


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_rectangular_solid_l621_62183

/-- The surface area of a sphere circumscribing a rectangular solid with dimensions √3, √2, and 1 is 6π. -/
theorem sphere_surface_area_of_circumscribed_rectangular_solid :
  let length : ℝ := Real.sqrt 3
  let width : ℝ := Real.sqrt 2
  let height : ℝ := 1
  let diagonal : ℝ := Real.sqrt (length ^ 2 + width ^ 2 + height ^ 2)
  let radius : ℝ := diagonal / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 6 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_circumscribed_rectangular_solid_l621_62183


namespace NUMINAMATH_CALUDE_percent_of_a_l621_62155

theorem percent_of_a (a b c : ℝ) (h1 : b = 0.5 * a) (h2 : c = 0.5 * b) :
  c = 0.25 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_of_a_l621_62155


namespace NUMINAMATH_CALUDE_tangent_line_equation_l621_62186

/-- The equation of the tangent line to the curve y = x^3 - 3x^2 + 1 at the point (1, -1) is y = -3x + 2 -/
theorem tangent_line_equation (x y : ℝ) : 
  y = x^3 - 3*x^2 + 1 → -- curve equation
  (1 : ℝ)^3 - 3*(1 : ℝ)^2 + 1 = -1 → -- point (1, -1) satisfies the curve equation
  ∃ (m b : ℝ), 
    (∀ t, y = m*t + b → (t - 1)*(3*(1 : ℝ)^2 - 6*(1 : ℝ)) = y + 1) ∧ -- point-slope form of tangent line
    m = -3 ∧ b = 2 -- coefficients of the tangent line equation
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l621_62186


namespace NUMINAMATH_CALUDE_lcm_36_65_l621_62147

theorem lcm_36_65 : Nat.lcm 36 65 = 2340 := by
  sorry

end NUMINAMATH_CALUDE_lcm_36_65_l621_62147


namespace NUMINAMATH_CALUDE_pizza_varieties_count_l621_62187

/-- Represents the number of base pizza flavors -/
def num_flavors : ℕ := 8

/-- Represents the number of extra topping options -/
def num_toppings : ℕ := 5

/-- Calculates the number of valid topping combinations -/
def valid_topping_combinations : ℕ :=
  (num_toppings) +  -- 1 topping
  (num_toppings.choose 2 - 1) +  -- 2 toppings, excluding onions with jalapeños
  (num_toppings.choose 3 - 3)  -- 3 toppings, excluding combinations with both onions and jalapeños

/-- The total number of pizza varieties -/
def total_varieties : ℕ := num_flavors * valid_topping_combinations

theorem pizza_varieties_count :
  total_varieties = 168 := by sorry

end NUMINAMATH_CALUDE_pizza_varieties_count_l621_62187


namespace NUMINAMATH_CALUDE_unique_cube_prime_l621_62189

theorem unique_cube_prime (n : ℕ) : (∃ p : ℕ, Nat.Prime p ∧ 2^n + n^2 + 25 = p^3) ↔ n = 6 :=
  sorry

end NUMINAMATH_CALUDE_unique_cube_prime_l621_62189


namespace NUMINAMATH_CALUDE_group_meal_cost_example_l621_62148

def group_meal_cost (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

theorem group_meal_cost_example : group_meal_cost 9 2 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_group_meal_cost_example_l621_62148


namespace NUMINAMATH_CALUDE_decimal_equality_and_unit_l621_62168

/-- Represents a number with its counting unit -/
structure NumberWithUnit where
  value : ℝ
  unit : ℝ

/-- The statement we want to prove false -/
def statement (a b : NumberWithUnit) : Prop :=
  a.value = b.value ∧ a.unit = b.unit

/-- The theorem to prove -/
theorem decimal_equality_and_unit (a b : NumberWithUnit) 
  (h1 : a.value = b.value)
  (h2 : a.unit = 1)
  (h3 : b.unit = 0.1) : 
  ¬(statement a b) := by
  sorry

#check decimal_equality_and_unit

end NUMINAMATH_CALUDE_decimal_equality_and_unit_l621_62168


namespace NUMINAMATH_CALUDE_urn_problem_l621_62144

theorem urn_problem (M : ℕ) : 
  (5 / 12 : ℚ) * (10 / (10 + M)) + (7 / 12 : ℚ) * (M / (10 + M)) = 62 / 100 → M = 7 :=
by sorry

end NUMINAMATH_CALUDE_urn_problem_l621_62144


namespace NUMINAMATH_CALUDE_min_words_to_learn_l621_62179

/-- Represents the French vocabulary exam setup -/
structure FrenchExam where
  totalWords : ℕ
  guessSuccessRate : ℚ
  targetScore : ℚ

/-- Calculates the exam score based on the number of words learned -/
def examScore (exam : FrenchExam) (wordsLearned : ℕ) : ℚ :=
  let correctGuesses := exam.guessSuccessRate * (exam.totalWords - wordsLearned)
  (wordsLearned + correctGuesses) / exam.totalWords

/-- Theorem stating the minimum number of words to learn for the given exam conditions -/
theorem min_words_to_learn (exam : FrenchExam) 
    (h1 : exam.totalWords = 800)
    (h2 : exam.guessSuccessRate = 1/20)
    (h3 : exam.targetScore = 9/10) : 
    ∀ n : ℕ, (∀ m : ℕ, m < n → examScore exam m < exam.targetScore) ∧ 
              examScore exam n ≥ exam.targetScore ↔ n = 716 := by
  sorry

end NUMINAMATH_CALUDE_min_words_to_learn_l621_62179


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l621_62178

theorem consecutive_odd_integers_sum (x : ℤ) : 
  (x % 2 = 1) →  -- x is odd
  (x + (x + 2) + (x + 4) ≥ 51) →  -- sum is at least 51
  (x ≥ 15) ∧  -- x is at least 15
  (∀ y : ℤ, (y % 2 = 1) ∧ (y + (y + 2) + (y + 4) ≥ 51) → y ≥ x) -- x is the smallest such integer
  := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l621_62178


namespace NUMINAMATH_CALUDE_sum_of_squares_of_quadratic_solutions_l621_62162

theorem sum_of_squares_of_quadratic_solutions : 
  let a : ℝ := -2
  let b : ℝ := -4
  let c : ℝ := -42
  let α : ℝ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let β : ℝ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  α^2 + β^2 = 46 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_quadratic_solutions_l621_62162


namespace NUMINAMATH_CALUDE_books_per_day_calculation_l621_62158

/-- Calculates the number of books read per day given the total books read and the number of reading days. -/
def books_per_day (total_books : ℕ) (reading_days : ℕ) : ℚ :=
  (total_books : ℚ) / (reading_days : ℚ)

/-- Represents the reading habits of a person over a period of weeks. -/
structure ReadingHabit where
  days_per_week : ℕ
  weeks : ℕ
  total_books : ℕ

theorem books_per_day_calculation (habit : ReadingHabit) 
    (h1 : habit.days_per_week = 2)
    (h2 : habit.weeks = 6)
    (h3 : habit.total_books = 48) :
  books_per_day habit.total_books (habit.days_per_week * habit.weeks) = 4 := by
  sorry

end NUMINAMATH_CALUDE_books_per_day_calculation_l621_62158


namespace NUMINAMATH_CALUDE_restocking_theorem_l621_62102

/-- Calculates the amount of ingredients needed to restock --/
def ingredients_to_buy (initial_flour initial_sugar initial_chips : ℕ)
                       (mon_flour mon_sugar mon_chips : ℕ)
                       (tue_flour tue_sugar tue_chips : ℕ)
                       (wed_flour wed_chips : ℕ)
                       (full_flour full_sugar full_chips : ℕ) :
                       (ℕ × ℕ × ℕ) :=
  let remaining_flour := initial_flour - mon_flour - tue_flour
  let spilled_flour := remaining_flour / 2
  let final_flour := if spilled_flour > wed_flour then spilled_flour - wed_flour else 0
  let flour_to_buy := full_flour + (if spilled_flour > wed_flour then 0 else wed_flour - spilled_flour)
  let sugar_to_buy := full_sugar - (initial_sugar - mon_sugar - tue_sugar)
  let chips_to_buy := full_chips - (initial_chips - mon_chips - tue_chips - wed_chips)
  (flour_to_buy, sugar_to_buy, chips_to_buy)

theorem restocking_theorem :
  ingredients_to_buy 500 300 400 150 120 200 240 90 150 100 90 500 300 400 = (545, 210, 440) := by
  sorry

end NUMINAMATH_CALUDE_restocking_theorem_l621_62102


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l621_62141

theorem gcd_power_two_minus_one :
  Nat.gcd (2^1510 - 1) (2^1500 - 1) = 2^10 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l621_62141


namespace NUMINAMATH_CALUDE_box_dimensions_sum_l621_62107

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Defines the properties of the rectangular box -/
def validBox (d : BoxDimensions) : Prop :=
  d.A * d.B = 18 ∧ d.A * d.C = 32 ∧ d.B * d.C = 50

/-- Theorem stating that the sum of dimensions is approximately 57.28 -/
theorem box_dimensions_sum (d : BoxDimensions) (h : validBox d) :
  ∃ ε > 0, |d.A + d.B + d.C - 57.28| < ε :=
sorry

end NUMINAMATH_CALUDE_box_dimensions_sum_l621_62107


namespace NUMINAMATH_CALUDE_inequality_equivalence_l621_62191

def f (x : ℝ) : ℝ := x * abs x

theorem inequality_equivalence (m : ℝ) : 
  (∀ x ≥ 1, f (x + m) + m * f x < 0) ↔ m ∈ Set.Iic (-1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l621_62191


namespace NUMINAMATH_CALUDE_chef_guests_problem_l621_62194

theorem chef_guests_problem (adults children seniors : ℕ) : 
  children = adults - 35 →
  seniors = 2 * children →
  adults + children + seniors = 127 →
  adults = 58 := by
  sorry

end NUMINAMATH_CALUDE_chef_guests_problem_l621_62194


namespace NUMINAMATH_CALUDE_employee_salary_problem_l621_62170

/-- Proves that given 20 employees, if adding a manager's salary of 3400
    increases the average salary by 100, then the initial average salary
    of the employees is 1300. -/
theorem employee_salary_problem (n : ℕ) (manager_salary : ℕ) (salary_increase : ℕ) 
    (h1 : n = 20)
    (h2 : manager_salary = 3400)
    (h3 : salary_increase = 100) :
    ∃ (initial_avg : ℕ),
      initial_avg * n + manager_salary = (initial_avg + salary_increase) * (n + 1) ∧
      initial_avg = 1300 := by
  sorry

end NUMINAMATH_CALUDE_employee_salary_problem_l621_62170


namespace NUMINAMATH_CALUDE_cube_edge_length_l621_62119

theorem cube_edge_length (material_volume : ℕ) (num_cubes : ℕ) (edge_length : ℕ) : 
  material_volume = 12 * 18 * 6 →
  num_cubes = 48 →
  material_volume = num_cubes * edge_length * edge_length * edge_length →
  edge_length = 3 := by
sorry

end NUMINAMATH_CALUDE_cube_edge_length_l621_62119


namespace NUMINAMATH_CALUDE_carpooling_arrangements_count_l621_62116

/-- Represents the last digit of a license plate -/
inductive LicensePlateEnding
| Nine
| Zero
| Two
| One
| Five

/-- Represents a day in the carpooling period -/
inductive Day
| Five
| Six
| Seven
| Eight
| Nine

def is_odd_day (d : Day) : Bool :=
  match d with
  | Day.Five | Day.Seven | Day.Nine => true
  | _ => false

def is_even_ending (e : LicensePlateEnding) : Bool :=
  match e with
  | LicensePlateEnding.Zero | LicensePlateEnding.Two => true
  | _ => false

def is_valid_car (d : Day) (e : LicensePlateEnding) : Bool :=
  (is_odd_day d && !is_even_ending e) || (!is_odd_day d && is_even_ending e)

/-- Represents a carpooling arrangement for the 5-day period -/
def CarpoolingArrangement := Day → LicensePlateEnding

def is_valid_arrangement (arr : CarpoolingArrangement) : Prop :=
  (∀ d, is_valid_car d (arr d)) ∧
  (∃! d, arr d = LicensePlateEnding.Nine)

def number_of_arrangements : ℕ := sorry

theorem carpooling_arrangements_count :
  number_of_arrangements = 80 := by sorry

end NUMINAMATH_CALUDE_carpooling_arrangements_count_l621_62116


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l621_62146

theorem line_segment_endpoint (y : ℝ) : 
  y > 0 → 
  Real.sqrt ((2 - (-6))^2 + (y - 5)^2) = 10 → 
  y = 11 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l621_62146


namespace NUMINAMATH_CALUDE_chocolate_lollipop_cost_equivalence_l621_62172

/-- Proves that the cost of one pack of chocolate equals the cost of 4 lollipops -/
theorem chocolate_lollipop_cost_equivalence 
  (lollipop_count : ℕ) 
  (chocolate_pack_count : ℕ)
  (lollipop_cost : ℕ)
  (bills_given : ℕ)
  (bill_value : ℕ)
  (change_received : ℕ)
  (h1 : lollipop_count = 4)
  (h2 : chocolate_pack_count = 6)
  (h3 : lollipop_cost = 2)
  (h4 : bills_given = 6)
  (h5 : bill_value = 10)
  (h6 : change_received = 4) :
  (bills_given * bill_value - change_received - lollipop_count * lollipop_cost) / chocolate_pack_count = 4 * lollipop_cost :=
by sorry

end NUMINAMATH_CALUDE_chocolate_lollipop_cost_equivalence_l621_62172


namespace NUMINAMATH_CALUDE_weight_lifting_multiple_l621_62108

theorem weight_lifting_multiple (rodney roger ron : ℕ) (m : ℕ) : 
  rodney + roger + ron = 239 →
  rodney = 2 * roger →
  roger = m * ron - 7 →
  rodney = 146 →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_weight_lifting_multiple_l621_62108


namespace NUMINAMATH_CALUDE_tailwind_speed_l621_62154

/-- Given a plane's ground speeds with and against a tailwind, calculate the speed of the tailwind. -/
theorem tailwind_speed (speed_with_wind speed_against_wind : ℝ) 
  (h1 : speed_with_wind = 460)
  (h2 : speed_against_wind = 310) :
  ∃ (plane_speed wind_speed : ℝ),
    plane_speed + wind_speed = speed_with_wind ∧
    plane_speed - wind_speed = speed_against_wind ∧
    wind_speed = 75 := by
  sorry

end NUMINAMATH_CALUDE_tailwind_speed_l621_62154


namespace NUMINAMATH_CALUDE_ashtons_remaining_items_l621_62113

def pencil_boxes : ℕ := 3
def pencils_per_box : ℕ := 14
def pen_boxes : ℕ := 2
def pens_per_box : ℕ := 10
def pencils_to_brother : ℕ := 6
def pencils_to_friends : ℕ := 12
def pens_to_friends : ℕ := 8

theorem ashtons_remaining_items :
  let initial_pencils := pencil_boxes * pencils_per_box
  let initial_pens := pen_boxes * pens_per_box
  let remaining_pencils := initial_pencils - pencils_to_brother - pencils_to_friends
  let remaining_pens := initial_pens - pens_to_friends
  remaining_pencils + remaining_pens = 36 := by
  sorry

end NUMINAMATH_CALUDE_ashtons_remaining_items_l621_62113


namespace NUMINAMATH_CALUDE_sin_450_degrees_l621_62125

theorem sin_450_degrees : Real.sin (450 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_450_degrees_l621_62125


namespace NUMINAMATH_CALUDE_sixth_power_of_complex_root_of_unity_l621_62145

theorem sixth_power_of_complex_root_of_unity (z : ℂ) : 
  z = (-1 + Complex.I * Real.sqrt 3) / 2 → z^6 = (1 : ℂ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_of_complex_root_of_unity_l621_62145


namespace NUMINAMATH_CALUDE_two_sided_icing_cubes_count_l621_62143

/-- Represents a 3D coordinate --/
structure Coord where
  x : Nat
  y : Nat
  z : Nat

/-- Represents a cake with dimensions and icing information --/
structure Cake where
  dim : Nat
  hasIcingTop : Bool
  hasIcingBottom : Bool
  hasIcingSides : Bool

/-- Counts the number of unit cubes with icing on exactly two sides --/
def countTwoSidedIcingCubes (c : Cake) : Nat :=
  sorry

/-- The main theorem to prove --/
theorem two_sided_icing_cubes_count (c : Cake) : 
  c.dim = 4 ∧ c.hasIcingTop ∧ ¬c.hasIcingBottom ∧ c.hasIcingSides → 
  countTwoSidedIcingCubes c = 20 := by
  sorry

end NUMINAMATH_CALUDE_two_sided_icing_cubes_count_l621_62143


namespace NUMINAMATH_CALUDE_encoded_equation_unique_solution_l621_62100

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ // 100 ≤ n ∧ n < 1000 }

/-- The encoded equation -/
def EncodedEquation (Δ square triangle circle : Digit) : Prop :=
  ∃ (base : TwoDigitNumber) (result : ThreeDigitNumber),
    base.val = 10 * Δ.val + square.val ∧
    result.val = 100 * square.val + 10 * circle.val + square.val ∧
    base.val ^ triangle.val = result.val

theorem encoded_equation_unique_solution :
  ∃! (Δ square triangle circle : Digit), EncodedEquation Δ square triangle circle :=
sorry

end NUMINAMATH_CALUDE_encoded_equation_unique_solution_l621_62100


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l621_62185

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : 1 / x + 1 / y = 5) 
  (h4 : 1 / x - 1 / y = -9) : 
  x + y = -5/14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l621_62185


namespace NUMINAMATH_CALUDE_positive_real_inequality_general_real_inequality_l621_62128

-- Part 1
theorem positive_real_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 / b ≥ 2*a - b := by sorry

-- Part 2
theorem general_real_inequality (a b : ℝ) :
  a^2 + b^2 + 3 ≥ a*b + Real.sqrt 3 * (a + b) := by sorry

end NUMINAMATH_CALUDE_positive_real_inequality_general_real_inequality_l621_62128


namespace NUMINAMATH_CALUDE_base_with_final_digit_two_l621_62197

theorem base_with_final_digit_two : 
  ∃! b : ℕ, 2 ≤ b ∧ b ≤ 20 ∧ 625 % b = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_with_final_digit_two_l621_62197


namespace NUMINAMATH_CALUDE_pool_water_removal_l621_62126

/-- Calculates the volume of water removed from a rectangular pool in gallons -/
def water_removed (length width height : ℝ) (conversion_factor : ℝ) : ℝ :=
  length * width * height * conversion_factor

theorem pool_water_removal :
  let length : ℝ := 60
  let width : ℝ := 10
  let height : ℝ := 0.5
  let conversion_factor : ℝ := 7.5
  water_removed length width height conversion_factor = 2250 := by
sorry

end NUMINAMATH_CALUDE_pool_water_removal_l621_62126


namespace NUMINAMATH_CALUDE_gunther_free_time_l621_62198

/-- Represents the time in minutes for each cleaning task -/
structure CleaningTasks where
  vacuum : ℕ
  dust : ℕ
  mop : ℕ
  brush_cat : ℕ

/-- Calculates the total cleaning time in hours -/
def total_cleaning_time (tasks : CleaningTasks) (num_cats : ℕ) : ℚ :=
  (tasks.vacuum + tasks.dust + tasks.mop + tasks.brush_cat * num_cats) / 60

/-- Theorem: If Gunther has no cats and 30 minutes of free time left after cleaning,
    his initial free time was 2.75 hours -/
theorem gunther_free_time (tasks : CleaningTasks) 
    (h1 : tasks.vacuum = 45)
    (h2 : tasks.dust = 60)
    (h3 : tasks.mop = 30)
    (h4 : tasks.brush_cat = 5)
    (h5 : total_cleaning_time tasks 0 + 0.5 = 2.75) : True :=
  sorry

end NUMINAMATH_CALUDE_gunther_free_time_l621_62198


namespace NUMINAMATH_CALUDE_complex_modulus_theorem_l621_62161

theorem complex_modulus_theorem (z : ℂ) (h : z * Complex.I = 1 - Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_theorem_l621_62161


namespace NUMINAMATH_CALUDE_odd_function_extrema_l621_62104

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

-- State the theorem
theorem odd_function_extrema :
  ∀ (a b c : ℝ),
  (∀ x, f a b c (-x) = -(f a b c x)) →  -- f is odd
  (f a b c 1 = 2) →                     -- maximum value of 2 at x = 1
  (∀ x, f a b c x ≤ f a b c 1) →        -- global maximum at x = 1
  (∃ (f_max f_min : ℝ),
    (∀ x ∈ Set.Icc (-4) 3, f (-1) 0 3 x ≤ f_max) ∧
    (∀ x ∈ Set.Icc (-4) 3, f_min ≤ f (-1) 0 3 x) ∧
    f_max = 52 ∧ f_min = -18) :=
by sorry


end NUMINAMATH_CALUDE_odd_function_extrema_l621_62104


namespace NUMINAMATH_CALUDE_extreme_points_range_l621_62182

open Real

/-- The function f(x) with parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + 1 + a * log x

/-- The derivative of f(x) with respect to x -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 2*x - 2 + a / x

theorem extreme_points_range (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ ∧ 
  (∃ (x : ℝ), f' a x = 0) ∧
  (∀ (x : ℝ), x ≤ x₁ ∨ x₂ ≤ x ∨ f' a x ≠ 0) →
  (1 - 2 * log 2) / 4 < f a x₂ ∧ f a x₂ < 0 :=
sorry

end NUMINAMATH_CALUDE_extreme_points_range_l621_62182


namespace NUMINAMATH_CALUDE_max_sum_squared_distances_l621_62177

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def is_unit_vector (v : E) : Prop := ‖v‖ = 1

theorem max_sum_squared_distances (a b c : E) 
  (ha : is_unit_vector a) (hb : is_unit_vector b) (hc : is_unit_vector c) :
  ‖a - b‖^2 + ‖a - c‖^2 + ‖b - c‖^2 ≤ 9 ∧ 
  ∃ (a' b' c' : E), is_unit_vector a' ∧ is_unit_vector b' ∧ is_unit_vector c' ∧
    ‖a' - b'‖^2 + ‖a' - c'‖^2 + ‖b' - c'‖^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_squared_distances_l621_62177


namespace NUMINAMATH_CALUDE_sqrt_fraction_equiv_neg_x_l621_62132

theorem sqrt_fraction_equiv_neg_x (x : ℝ) (h : x < 0) :
  Real.sqrt (x / (1 - (x - 1) / x)) = -x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equiv_neg_x_l621_62132


namespace NUMINAMATH_CALUDE_ax5_plus_by5_l621_62112

theorem ax5_plus_by5 (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 30)
  (h4 : a * x^4 + b * y^4 = 80) :
  a * x^5 + b * y^5 = 6200 / 29 := by
sorry

end NUMINAMATH_CALUDE_ax5_plus_by5_l621_62112


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l621_62151

theorem sum_of_cubes_of_roots (a b : ℝ) (α β : ℝ) : 
  (α^2 + a*α + b = 0) → (β^2 + a*β + b = 0) → α^3 + β^3 = -(a^3 - 3*a*b) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l621_62151


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l621_62184

/-- Given two points P₁ and P₂ that are symmetric with respect to the origin,
    prove that m - n = 8. -/
theorem symmetric_points_difference (m n : ℝ) : 
  (∃ (P₁ P₂ : ℝ × ℝ), 
    P₁ = (2 - m, 5) ∧ 
    P₂ = (3, 2*n + 1) ∧ 
    P₁.1 = -P₂.1 ∧ 
    P₁.2 = -P₂.2) → 
  m - n = 8 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l621_62184


namespace NUMINAMATH_CALUDE_mixture_ratio_l621_62142

/-- Given a mixture of liquids p and q, prove that the initial ratio is 3:2 -/
theorem mixture_ratio (p q : ℝ) : 
  p + q = 25 →                      -- Initial total volume
  p / (q + 2) = 5 / 4 →             -- Ratio after adding 2 liters of q
  p / q = 3 / 2 :=                  -- Initial ratio
by sorry

end NUMINAMATH_CALUDE_mixture_ratio_l621_62142


namespace NUMINAMATH_CALUDE_sum_six_consecutive_integers_l621_62165

theorem sum_six_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_consecutive_integers_l621_62165


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l621_62105

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence satisfying certain conditions, its 4th term equals 8. -/
theorem fourth_term_of_geometric_sequence (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h_sum : a 6 + a 2 = 34) 
    (h_diff : a 6 - a 2 = 30) : 
  a 4 = 8 := by
  sorry


end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l621_62105


namespace NUMINAMATH_CALUDE_tissue_paper_count_l621_62135

theorem tissue_paper_count (remaining : ℕ) (used : ℕ) (initial : ℕ) : 
  remaining = 93 → used = 4 → initial = remaining + used :=
by sorry

end NUMINAMATH_CALUDE_tissue_paper_count_l621_62135


namespace NUMINAMATH_CALUDE_intersection_implies_b_range_l621_62171

/-- The set M represents an ellipse -/
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}

/-- The set N represents a family of lines parameterized by m and b -/
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

/-- The theorem states that if M intersects N for all m, then b is in the specified range -/
theorem intersection_implies_b_range :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) → b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_b_range_l621_62171


namespace NUMINAMATH_CALUDE_solve_system_l621_62188

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  q = -25 / 11 := by sorry

end NUMINAMATH_CALUDE_solve_system_l621_62188


namespace NUMINAMATH_CALUDE_remainder_sum_l621_62111

theorem remainder_sum (a b : ℤ) 
  (ha : a % 80 = 74) 
  (hb : b % 120 = 114) : 
  (a + b) % 40 = 28 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l621_62111


namespace NUMINAMATH_CALUDE_basketball_cost_l621_62150

theorem basketball_cost (initial_amount : ℕ) (jersey_cost : ℕ) (jersey_count : ℕ) (shorts_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 50 →
  jersey_cost = 2 →
  jersey_count = 5 →
  shorts_cost = 8 →
  remaining_amount = 14 →
  initial_amount - (jersey_cost * jersey_count + shorts_cost + remaining_amount) = 18 :=
by sorry

end NUMINAMATH_CALUDE_basketball_cost_l621_62150


namespace NUMINAMATH_CALUDE_special_quadrilateral_angles_l621_62157

/-- A quadrilateral with three equal sides and two specific angles -/
structure SpecialQuadrilateral where
  -- Three equal sides
  side : ℝ
  side_positive : side > 0
  -- Two angles formed by the equal sides
  angle1 : ℝ
  angle2 : ℝ
  -- Angle conditions
  angle1_is_90 : angle1 = 90
  angle2_is_150 : angle2 = 150

/-- The other two angles of the special quadrilateral -/
def other_angles (q : SpecialQuadrilateral) : ℝ × ℝ :=
  (45, 75)

/-- Theorem stating that the other two angles are 45° and 75° -/
theorem special_quadrilateral_angles (q : SpecialQuadrilateral) :
  other_angles q = (45, 75) := by
  sorry

end NUMINAMATH_CALUDE_special_quadrilateral_angles_l621_62157


namespace NUMINAMATH_CALUDE_line_slope_k_l621_62173

/-- Given a line passing through the points (-1, -4) and (4, k) with slope k, prove that k = 1 -/
theorem line_slope_k (k : ℝ) : 
  (k - (-4)) / (4 - (-1)) = k → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_k_l621_62173


namespace NUMINAMATH_CALUDE_school_play_ticket_ratio_l621_62103

theorem school_play_ticket_ratio :
  ∀ (total_tickets student_tickets adult_tickets : ℕ),
    total_tickets = 366 →
    adult_tickets = 122 →
    total_tickets = student_tickets + adult_tickets →
    ∃ (k : ℕ), student_tickets = k * adult_tickets →
    (student_tickets : ℚ) / (adult_tickets : ℚ) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_school_play_ticket_ratio_l621_62103


namespace NUMINAMATH_CALUDE_circle_center_distance_l621_62152

theorem circle_center_distance (x y : ℝ) :
  x^2 + y^2 = 8*x - 2*y + 23 →
  Real.sqrt ((4 - (-3))^2 + (-1 - 4)^2) = Real.sqrt 74 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_distance_l621_62152


namespace NUMINAMATH_CALUDE_complex_unit_vector_l621_62181

theorem complex_unit_vector (z : ℂ) (h : z = 3 + 4*I) : z / Complex.abs z = 3/5 + 4/5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_unit_vector_l621_62181


namespace NUMINAMATH_CALUDE_george_carries_two_buckets_l621_62174

/-- The number of buckets George can carry each round -/
def george_buckets : ℕ := 2

/-- The number of buckets Harry can carry each round -/
def harry_buckets : ℕ := 3

/-- The total number of buckets needed to fill the pool -/
def total_buckets : ℕ := 110

/-- The number of rounds needed to fill the pool -/
def total_rounds : ℕ := 22

theorem george_carries_two_buckets :
  george_buckets = 2 ∧
  harry_buckets * total_rounds + george_buckets * total_rounds = total_buckets :=
sorry

end NUMINAMATH_CALUDE_george_carries_two_buckets_l621_62174


namespace NUMINAMATH_CALUDE_shortest_piece_length_l621_62133

theorem shortest_piece_length (total_length : ℝ) (piece1 piece2 piece3 : ℝ) : 
  total_length = 138 →
  piece1 + piece2 + piece3 = total_length →
  piece1 = 2 * piece2 →
  piece2 = 3 * piece3 →
  piece3 = 13.8 := by
sorry

end NUMINAMATH_CALUDE_shortest_piece_length_l621_62133


namespace NUMINAMATH_CALUDE_divisible_by_1998_digit_sum_l621_62166

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n, if n is divisible by 1998, 
    then the sum of its digits is greater than or equal to 27 -/
theorem divisible_by_1998_digit_sum (n : ℕ) : 
  n % 1998 = 0 → sum_of_digits n ≥ 27 := by sorry

end NUMINAMATH_CALUDE_divisible_by_1998_digit_sum_l621_62166


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l621_62114

/-- An arithmetic sequence -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The theorem to prove -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h : arithmeticSequence a) 
    (h_sum : a 5 + a 10 = 12) : 
  3 * a 7 + a 9 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l621_62114
