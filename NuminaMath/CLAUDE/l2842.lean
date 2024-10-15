import Mathlib

namespace NUMINAMATH_CALUDE_coin_toss_is_random_event_l2842_284231

/-- Represents the outcome of a coin toss -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents a random event -/
class RandomEvent (α : Type) where
  /-- The probability of the event occurring is between 0 and 1, exclusive -/
  prob_between_zero_and_one : ∃ (p : ℝ), 0 < p ∧ p < 1

/-- Definition of a coin toss -/
def coinToss : Set CoinOutcome := {CoinOutcome.Heads, CoinOutcome.Tails}

/-- Theorem: Tossing a coin is a random event -/
theorem coin_toss_is_random_event : RandomEvent coinToss := by
  sorry


end NUMINAMATH_CALUDE_coin_toss_is_random_event_l2842_284231


namespace NUMINAMATH_CALUDE_num_planes_is_one_or_three_l2842_284294

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  distinct : point1 ≠ point2

/-- Three pairwise parallel lines -/
structure ThreeParallelLines where
  line1 : Line3D
  line2 : Line3D
  line3 : Line3D
  parallel12 : line1.point2 - line1.point1 = line2.point2 - line2.point1
  parallel23 : line2.point2 - line2.point1 = line3.point2 - line3.point1
  parallel31 : line3.point2 - line3.point1 = line1.point2 - line1.point1

/-- The number of planes determined by three pairwise parallel lines -/
def num_planes_from_parallel_lines (lines : ThreeParallelLines) : Fin 4 :=
  sorry

/-- Theorem: The number of planes determined by three pairwise parallel lines is either 1 or 3 -/
theorem num_planes_is_one_or_three (lines : ThreeParallelLines) :
  (num_planes_from_parallel_lines lines = 1) ∨ (num_planes_from_parallel_lines lines = 3) :=
sorry

end NUMINAMATH_CALUDE_num_planes_is_one_or_three_l2842_284294


namespace NUMINAMATH_CALUDE_four_students_same_group_probability_l2842_284281

/-- The number of groups -/
def num_groups : ℕ := 4

/-- The probability of a student being assigned to a specific group -/
def prob_assigned_to_group : ℚ := 1 / num_groups

/-- The probability of four specific students being assigned to the same group -/
def prob_four_students_same_group : ℚ := prob_assigned_to_group ^ 3

theorem four_students_same_group_probability :
  prob_four_students_same_group = 1 / 64 := by sorry

end NUMINAMATH_CALUDE_four_students_same_group_probability_l2842_284281


namespace NUMINAMATH_CALUDE_roots_of_cosine_equation_l2842_284287

theorem roots_of_cosine_equation :
  let f (t : ℝ) := 32 * t^5 - 40 * t^3 + 10 * t - Real.sqrt 3
  (f (Real.cos (6 * π / 180)) = 0) →
  (f (Real.cos (66 * π / 180)) = 0) ∧
  (f (Real.cos (78 * π / 180)) = 0) ∧
  (f (Real.cos (138 * π / 180)) = 0) ∧
  (f (Real.cos (150 * π / 180)) = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_cosine_equation_l2842_284287


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l2842_284262

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles : 
  let center1 : ℝ × ℝ := (5, 3)
  let radius1 : ℝ := 12
  let center2 : ℝ × ℝ := (2, -1)
  let radius2 : ℝ := 6
  let distance_between_centers : ℝ := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  let shortest_distance : ℝ := max 0 (distance_between_centers - (radius1 + radius2))
  shortest_distance = 1 :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_between_circles_l2842_284262


namespace NUMINAMATH_CALUDE_fraction_equality_l2842_284227

theorem fraction_equality (x y z m : ℝ) 
  (h1 : 5 / (x + y) = m / (x + z)) 
  (h2 : m / (x + z) = 13 / (z - y)) : 
  m = 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2842_284227


namespace NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l2842_284267

/-- An arithmetic sequence -/
def ArithmeticSequence (b : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_k_value
  (b : ℕ → ℚ)
  (h_arith : ArithmeticSequence b)
  (h_sum1 : b 5 + b 8 + b 11 = 21)
  (h_sum2 : (Finset.range 11).sum (fun i => b (i + 5)) = 121)
  (h_bk : ∃ k : ℕ, b k = 23) :
  ∃ k : ℕ, b k = 23 ∧ k = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_k_value_l2842_284267


namespace NUMINAMATH_CALUDE_triangle_base_length_l2842_284278

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 13.5 → height = 6 → area = (base * height) / 2 → base = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l2842_284278


namespace NUMINAMATH_CALUDE_exists_non_isosceles_with_four_equal_subtriangles_l2842_284237

/-- A triangle represented by its three vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- An interior point of a triangle -/
def InteriorPoint (t : Triangle) := ℝ × ℝ

/-- Predicate to check if a triangle is isosceles -/
def IsIsosceles (t : Triangle) : Prop := sorry

/-- Predicate to check if a point is inside a triangle -/
def IsInside (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Function to create 6 triangles by connecting an interior point to vertices and drawing perpendiculars -/
def CreateSubTriangles (t : Triangle) (p : InteriorPoint t) : List Triangle := sorry

/-- Predicate to check if 4 out of 6 triangles in a list are equal -/
def FourOutOfSixEqual (triangles : List Triangle) : Prop := sorry

/-- Theorem stating that there exists a non-isosceles triangle with an interior point
    such that 4 out of 6 resulting triangles are equal -/
theorem exists_non_isosceles_with_four_equal_subtriangles :
  ∃ (t : Triangle) (p : InteriorPoint t),
    ¬IsIsosceles t ∧
    IsInside p t ∧
    FourOutOfSixEqual (CreateSubTriangles t p) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_isosceles_with_four_equal_subtriangles_l2842_284237


namespace NUMINAMATH_CALUDE_intersection_M_N_l2842_284273

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | x^2 = 2*x}

theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2842_284273


namespace NUMINAMATH_CALUDE_crabapple_sequences_count_l2842_284201

/-- The number of ways to select 5 students from a group of 13 students,
    where the order matters and no student is selected more than once. -/
def crabapple_sequences : ℕ :=
  13 * 12 * 11 * 10 * 9

/-- Theorem stating that the number of crabapple recipient sequences is 154,440. -/
theorem crabapple_sequences_count : crabapple_sequences = 154440 := by
  sorry

end NUMINAMATH_CALUDE_crabapple_sequences_count_l2842_284201


namespace NUMINAMATH_CALUDE_hexagon_area_is_32_l2842_284282

/-- A hexagon surrounded by four right triangles forming a rectangle -/
structure HexagonWithTriangles where
  -- Side length of the hexagon
  side_length : ℝ
  -- Height of each triangle
  triangle_height : ℝ
  -- The shape forms a rectangle
  is_rectangle : Bool
  -- There are four identical right triangles
  triangle_count : Nat
  -- The triangles are identical and right-angled
  triangles_identical_right : Bool

/-- The area of the hexagon given its structure -/
def hexagon_area (h : HexagonWithTriangles) : ℝ :=
  sorry

/-- Theorem stating the area of the hexagon is 32 square units -/
theorem hexagon_area_is_32 (h : HexagonWithTriangles) 
  (h_side : h.side_length = 2)
  (h_height : h.triangle_height = 4)
  (h_rect : h.is_rectangle = true)
  (h_count : h.triangle_count = 4)
  (h_tri : h.triangles_identical_right = true) :
  hexagon_area h = 32 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_is_32_l2842_284282


namespace NUMINAMATH_CALUDE_athlete_C_is_best_l2842_284219

structure Athlete where
  name : String
  average_score : ℝ
  variance : ℝ

def athletes : List Athlete := [
  ⟨"A", 7, 0.9⟩,
  ⟨"B", 8, 1.1⟩,
  ⟨"C", 8, 0.9⟩,
  ⟨"D", 7, 1.0⟩
]

def has_best_performance_and_stability (a : Athlete) (athletes : List Athlete) : Prop :=
  ∀ b ∈ athletes, 
    (a.average_score > b.average_score) ∨ 
    (a.average_score = b.average_score ∧ a.variance ≤ b.variance)

theorem athlete_C_is_best : 
  ∃ a ∈ athletes, a.name = "C" ∧ has_best_performance_and_stability a athletes := by
  sorry

end NUMINAMATH_CALUDE_athlete_C_is_best_l2842_284219


namespace NUMINAMATH_CALUDE_harvest_calculation_l2842_284288

/-- Represents the harvest schedule and quantities for oranges and apples -/
structure HarvestData where
  total_days : ℕ
  orange_sacks : ℕ
  apple_sacks : ℕ
  orange_interval : ℕ
  apple_interval : ℕ

/-- Calculates the number of sacks harvested per day when both fruits are harvested together -/
def sacks_per_joint_harvest_day (data : HarvestData) : ℚ :=
  let orange_days := data.total_days / data.orange_interval
  let apple_days := data.total_days / data.apple_interval
  let orange_per_day := data.orange_sacks / orange_days
  let apple_per_day := data.apple_sacks / apple_days
  orange_per_day + apple_per_day

/-- The main theorem stating the result of the harvest calculation -/
theorem harvest_calculation (data : HarvestData) 
  (h1 : data.total_days = 20)
  (h2 : data.orange_sacks = 56)
  (h3 : data.apple_sacks = 35)
  (h4 : data.orange_interval = 2)
  (h5 : data.apple_interval = 3) :
  sacks_per_joint_harvest_day data = 11.4333 := by
  sorry

end NUMINAMATH_CALUDE_harvest_calculation_l2842_284288


namespace NUMINAMATH_CALUDE_min_value_implies_m_range_l2842_284275

-- Define the piecewise function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then (x - m)^2 - 2 else 2*x^3 - 3*x^2

-- State the theorem
theorem min_value_implies_m_range (m : ℝ) :
  (∀ x, f m x ≥ -1) ∧ (∃ x, f m x = -1) → m ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_implies_m_range_l2842_284275


namespace NUMINAMATH_CALUDE_average_and_differences_l2842_284222

theorem average_and_differences (x : ℝ) : 
  (45 + x) / 2 = 38 → |x - 45| + |x - 30| = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_and_differences_l2842_284222


namespace NUMINAMATH_CALUDE_amoeba_problem_l2842_284224

/-- The number of amoebas after n days, given an initial population and split factor --/
def amoeba_population (initial : ℕ) (split_factor : ℕ) (days : ℕ) : ℕ :=
  initial * split_factor ^ days

/-- Theorem: Given 2 initial amoebas that split into 3 each day, after 5 days there will be 486 amoebas --/
theorem amoeba_problem :
  amoeba_population 2 3 5 = 486 := by
  sorry

#eval amoeba_population 2 3 5

end NUMINAMATH_CALUDE_amoeba_problem_l2842_284224


namespace NUMINAMATH_CALUDE_board_cutting_l2842_284230

theorem board_cutting (total_length shorter_length : ℝ) 
  (h1 : total_length = 120)
  (h2 : shorter_length = 35)
  (h3 : ∃ longer_length, longer_length + shorter_length = total_length ∧ 
    ∃ x, longer_length = 2 * shorter_length + x) :
  ∃ longer_length x, 
    longer_length + shorter_length = total_length ∧ 
    longer_length = 2 * shorter_length + x ∧ 
    x = 15 :=
by sorry

end NUMINAMATH_CALUDE_board_cutting_l2842_284230


namespace NUMINAMATH_CALUDE_initial_fish_caught_per_day_l2842_284255

-- Define the initial colony size
def initial_colony_size : ℕ := sorry

-- Define the colony size after the first year (doubled)
def first_year_size : ℕ := 2 * initial_colony_size

-- Define the colony size after the second year (tripled from first year)
def second_year_size : ℕ := 3 * first_year_size

-- Define the current colony size (after third year)
def current_colony_size : ℕ := 1077

-- Define the increase in the third year
def third_year_increase : ℕ := 129

-- Define the fish consumption per penguin per day
def fish_per_penguin : ℚ := 3/2

-- Theorem stating the initial number of fish caught per day
theorem initial_fish_caught_per_day :
  (initial_colony_size : ℚ) * fish_per_penguin = 237 :=
by sorry

end NUMINAMATH_CALUDE_initial_fish_caught_per_day_l2842_284255


namespace NUMINAMATH_CALUDE_sum_of_squares_power_l2842_284277

theorem sum_of_squares_power (a p q : ℤ) (h : a = p^2 + q^2) :
  ∀ k : ℕ+, ∃ x y : ℤ, a^(k : ℕ) = x^2 + y^2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_power_l2842_284277


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l2842_284285

theorem unique_solution_floor_equation :
  ∃! (x : ℝ), x > 0 ∧ x * ↑(⌊x⌋) = 72 ∧ x = 9 := by sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l2842_284285


namespace NUMINAMATH_CALUDE_rectangle_perimeter_is_48_l2842_284216

/-- A rectangle can be cut into two squares with side length 8 cm -/
structure Rectangle where
  length : ℝ
  width : ℝ
  is_cut_into_squares : length = 2 * width
  square_side : ℝ
  square_side_eq : square_side = width

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: The perimeter of the rectangle is 48 cm -/
theorem rectangle_perimeter_is_48 (r : Rectangle) (h : r.square_side = 8) : perimeter r = 48 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_is_48_l2842_284216


namespace NUMINAMATH_CALUDE_two_real_roots_iff_m_nonpositive_m_values_given_roots_relationship_l2842_284214

/-- Given a quadratic equation x^2 - 2x + m + 1 = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 - 2*x + m + 1 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (-2)^2 - 4*(m + 1)

/-- The condition for two real roots -/
def has_two_real_roots (m : ℝ) : Prop :=
  discriminant m ≥ 0

/-- The relationship between the roots and m -/
def roots_relationship (x₁ x₂ m : ℝ) : Prop :=
  x₁ + 3*x₂ = 2*m + 8

/-- Theorem 1: The equation has two real roots iff m ≤ 0 -/
theorem two_real_roots_iff_m_nonpositive (m : ℝ) :
  has_two_real_roots m ↔ m ≤ 0 :=
sorry

/-- Theorem 2: If the roots satisfy the given relationship, then m = -1 or m = -2 -/
theorem m_values_given_roots_relationship (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_equation x₁ m ∧ quadratic_equation x₂ m ∧ roots_relationship x₁ x₂ m) →
  (m = -1 ∨ m = -2) :=
sorry

end NUMINAMATH_CALUDE_two_real_roots_iff_m_nonpositive_m_values_given_roots_relationship_l2842_284214


namespace NUMINAMATH_CALUDE_max_expression_c_value_l2842_284207

theorem max_expression_c_value (a b c : ℕ) : 
  a ∈ ({1, 2, 4} : Set ℕ) →
  b ∈ ({1, 2, 4} : Set ℕ) →
  c ∈ ({1, 2, 4} : Set ℕ) →
  a ≠ b → b ≠ c → a ≠ c →
  (∀ x y z : ℕ, x ∈ ({1, 2, 4} : Set ℕ) → y ∈ ({1, 2, 4} : Set ℕ) → z ∈ ({1, 2, 4} : Set ℕ) →
    x ≠ y → y ≠ z → x ≠ z → (x / 2) / (y / z : ℚ) ≤ (a / 2) / (b / c : ℚ)) →
  (a / 2) / (b / c : ℚ) = 4 →
  c = 2 := by sorry

end NUMINAMATH_CALUDE_max_expression_c_value_l2842_284207


namespace NUMINAMATH_CALUDE_new_ratio_second_term_l2842_284212

def original_ratio : Rat × Rat := (4, 15)
def number_to_add : ℕ := 29

theorem new_ratio_second_term :
  let new_ratio := (original_ratio.1 + number_to_add, original_ratio.2 + number_to_add)
  new_ratio.2 = 44 := by sorry

end NUMINAMATH_CALUDE_new_ratio_second_term_l2842_284212


namespace NUMINAMATH_CALUDE_locus_of_midpoint_of_tangent_l2842_284250

/-- Given two circles with centers at (0, 0) and (a, 0), prove that the locus of the midpoint
    of their common outer tangent is part of a specific circle. -/
theorem locus_of_midpoint_of_tangent (a c : ℝ) (h₁ : a > c) (h₂ : c > 0) :
  ∃ (x y : ℝ), 
    4 * x^2 + 4 * y^2 - 4 * a * x + a^2 = c^2 ∧ 
    (a^2 - c^2) / (2 * a) ≤ x ∧ 
    x ≤ (a^2 + c^2) / (2 * a) ∧ 
    y > 0 :=
sorry

end NUMINAMATH_CALUDE_locus_of_midpoint_of_tangent_l2842_284250


namespace NUMINAMATH_CALUDE_optionC_most_suitable_l2842_284289

/-- Represents a sampling option with population size and sample size -/
structure SamplingOption where
  populationSize : ℕ
  sampleSize : ℕ

/-- Determines if a sampling option is suitable for simple random sampling -/
def isSuitableForSimpleRandomSampling (option : SamplingOption) : Prop :=
  option.populationSize ≤ 30 ∧ option.sampleSize ≤ 5

/-- The given sampling options -/
def optionA : SamplingOption := ⟨1320, 300⟩
def optionB : SamplingOption := ⟨1135, 50⟩
def optionC : SamplingOption := ⟨30, 5⟩
def optionD : SamplingOption := ⟨5000, 200⟩

/-- Theorem stating that Option C is most suitable for simple random sampling -/
theorem optionC_most_suitable :
  isSuitableForSimpleRandomSampling optionC ∧
  ¬isSuitableForSimpleRandomSampling optionA ∧
  ¬isSuitableForSimpleRandomSampling optionB ∧
  ¬isSuitableForSimpleRandomSampling optionD :=
by sorry

end NUMINAMATH_CALUDE_optionC_most_suitable_l2842_284289


namespace NUMINAMATH_CALUDE_trig_identity_proof_l2842_284247

theorem trig_identity_proof :
  6 * Real.cos (10 * π / 180) * Real.cos (50 * π / 180) * Real.cos (70 * π / 180) +
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) =
  6 * (1 + Real.sqrt 3) / 8 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l2842_284247


namespace NUMINAMATH_CALUDE_contractor_problem_l2842_284264

/-- A contractor problem -/
theorem contractor_problem (daily_wage : ℚ) (daily_fine : ℚ) (total_earnings : ℚ) (absent_days : ℕ) :
  daily_wage = 25 →
  daily_fine = (15/2) →
  total_earnings = 555 →
  absent_days = 6 →
  ∃ (total_days : ℕ), total_days = 24 ∧ 
    daily_wage * (total_days - absent_days : ℚ) - daily_fine * absent_days = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_contractor_problem_l2842_284264


namespace NUMINAMATH_CALUDE_triangle_value_l2842_284242

theorem triangle_value (triangle p : ℤ) 
  (eq1 : triangle + p = 73)
  (eq2 : (triangle + p) + 2*p = 157) : 
  triangle = 31 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l2842_284242


namespace NUMINAMATH_CALUDE_max_volume_corner_cut_box_l2842_284280

/-- The maximum volume of an open-top box formed by cutting identical squares from the corners of a rectangular cardboard -/
theorem max_volume_corner_cut_box (a b : ℝ) (ha : a = 10) (hb : b = 16) :
  let V := fun x => (a - 2*x) * (b - 2*x) * x
  ∃ (x : ℝ), x > 0 ∧ x < a/2 ∧ x < b/2 ∧
    (∀ y, y > 0 → y < a/2 → y < b/2 → V y ≤ V x) ∧
    V x = 144 :=
sorry

end NUMINAMATH_CALUDE_max_volume_corner_cut_box_l2842_284280


namespace NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l2842_284266

/-- Converts a base 8 number to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

/-- Checks if a number is a 3-digit base 8 number -/
def is_3digit_base8 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), is_3digit_base8 n ∧ 
             base8_to_base10 n % 7 = 0 ∧
             ∀ (m : ℕ), is_3digit_base8 m ∧ base8_to_base10 m % 7 = 0 → m ≤ n :=
by
  use 777
  sorry

end NUMINAMATH_CALUDE_greatest_3digit_base8_divisible_by_7_l2842_284266


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2842_284253

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℕ, n^2 < 2^n) ↔ (∃ n₀ : ℕ, n₀^2 ≥ 2^n₀) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2842_284253


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2842_284269

theorem quadratic_inequality_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 2)*x - k + 4 > 0) ↔ 
  k > -2 * Real.sqrt 3 ∧ k < 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2842_284269


namespace NUMINAMATH_CALUDE_boat_license_combinations_l2842_284235

/-- The number of possible letters for a boat license -/
def letter_choices : ℕ := 4

/-- The number of possible choices for the first digit of a boat license -/
def first_digit_choices : ℕ := 8

/-- The number of possible choices for each of the remaining digits of a boat license -/
def other_digit_choices : ℕ := 10

/-- The number of digits in a boat license after the letter -/
def num_digits : ℕ := 7

/-- Theorem: The number of possible boat license combinations is 32,000,000 -/
theorem boat_license_combinations :
  letter_choices * first_digit_choices * (other_digit_choices ^ (num_digits - 1)) = 32000000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_combinations_l2842_284235


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2842_284202

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 3}

theorem complement_intersection_theorem : 
  {4, 5, 6} = (U \ M) ∩ (U \ N) := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2842_284202


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l2842_284240

theorem polynomial_identity_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x : ℝ, 1728 * x^4 + 64 = (a * x^3 + b * x^2 + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 416 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l2842_284240


namespace NUMINAMATH_CALUDE_college_strength_l2842_284200

theorem college_strength (cricket_players basketball_players both : ℕ) 
  (h1 : cricket_players = 500)
  (h2 : basketball_players = 600)
  (h3 : both = 220) :
  cricket_players + basketball_players - both = 880 :=
by sorry

end NUMINAMATH_CALUDE_college_strength_l2842_284200


namespace NUMINAMATH_CALUDE_train_ticket_types_l2842_284274

/-- The number of ticket types needed for a train route -/
def ticket_types (stops_between : ℕ) : ℕ :=
  let total_stops := stops_between + 2
  total_stops * (total_stops - 1)

/-- Theorem: For a train route with 3 stops between two end cities, 
    the number of ticket types needed is 20 -/
theorem train_ticket_types : ticket_types 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_ticket_types_l2842_284274


namespace NUMINAMATH_CALUDE_fraction_simplification_l2842_284276

theorem fraction_simplification (x y : ℚ) 
  (hx : x = 4 / 6) (hy : y = 8 / 10) : 
  (6 * x + 8 * y) / (48 * x * y) = 13 / 32 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2842_284276


namespace NUMINAMATH_CALUDE_points_per_game_l2842_284203

theorem points_per_game (total_points : ℕ) (num_games : ℕ) (points_per_game : ℕ) : 
  total_points = 81 → 
  num_games = 3 → 
  total_points = num_games * points_per_game → 
  points_per_game = 27 := by
sorry

end NUMINAMATH_CALUDE_points_per_game_l2842_284203


namespace NUMINAMATH_CALUDE_morgan_lunch_change_l2842_284209

/-- Calculates the change Morgan receives from his lunch order --/
theorem morgan_lunch_change : 
  let hamburger : ℚ := 5.75
  let onion_rings : ℚ := 2.50
  let smoothie : ℚ := 3.25
  let side_salad : ℚ := 3.75
  let chocolate_cake : ℚ := 4.20
  let discount_rate : ℚ := 0.10
  let tax_rate : ℚ := 0.06
  let payment : ℚ := 50

  let total_before_discount : ℚ := hamburger + onion_rings + smoothie + side_salad + chocolate_cake
  let discount : ℚ := (side_salad + chocolate_cake) * discount_rate
  let total_after_discount : ℚ := total_before_discount - discount
  let tax : ℚ := total_after_discount * tax_rate
  let final_total : ℚ := total_after_discount + tax
  let change : ℚ := payment - final_total

  change = 30.34 := by sorry

end NUMINAMATH_CALUDE_morgan_lunch_change_l2842_284209


namespace NUMINAMATH_CALUDE_baez_marbles_l2842_284297

theorem baez_marbles (p : ℝ) : 
  25 > 0 ∧ 0 ≤ p ∧ p ≤ 100 ∧ 2 * ((100 - p) / 100 * 25) = 60 → p = 20 :=
by sorry

end NUMINAMATH_CALUDE_baez_marbles_l2842_284297


namespace NUMINAMATH_CALUDE_expression_simplification_l2842_284218

theorem expression_simplification (x : ℝ) (h : x = 8) :
  (2 * x) / (x + 1) - ((2 * x + 4) / (x^2 - 1)) / ((x + 2) / (x^2 - 2*x + 1)) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2842_284218


namespace NUMINAMATH_CALUDE_rotate_point_around_OA_l2842_284217

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Rotate a point around a ray by a given angle -/
def rotateAroundRay (p : Point3D) (origin : Point3D) (axis : Point3D) (angle : ℝ) : Point3D :=
  sorry

/-- The theorem to prove -/
theorem rotate_point_around_OA : 
  let A : Point3D := ⟨1, 1, 1⟩
  let P : Point3D := ⟨1, 1, 0⟩
  let O : Point3D := ⟨0, 0, 0⟩
  let angle : ℝ := π / 3  -- 60 degrees in radians
  let rotated_P : Point3D := rotateAroundRay P O A angle
  rotated_P = ⟨1/3, 4/3, 1/3⟩ := by sorry

end NUMINAMATH_CALUDE_rotate_point_around_OA_l2842_284217


namespace NUMINAMATH_CALUDE_customer_total_cost_l2842_284299

-- Define the quantities and prices of items
def riqing_quantity : ℕ := 24
def riqing_price : ℚ := 1.80
def riqing_discount : ℚ := 0.8

def kangshifu_quantity : ℕ := 6
def kangshifu_price : ℚ := 1.70
def kangshifu_discount : ℚ := 0.8

def shanlin_quantity : ℕ := 5
def shanlin_price : ℚ := 3.40
def shanlin_discount : ℚ := 1  -- No discount

def shuanghui_quantity : ℕ := 3
def shuanghui_price : ℚ := 11.20
def shuanghui_discount : ℚ := 0.9

-- Define the total cost function
def total_cost : ℚ :=
  riqing_quantity * riqing_price * riqing_discount +
  kangshifu_quantity * kangshifu_price * kangshifu_discount +
  shanlin_quantity * shanlin_price * shanlin_discount +
  shuanghui_quantity * shuanghui_price * shuanghui_discount

-- Theorem statement
theorem customer_total_cost : total_cost = 89.96 := by
  sorry

end NUMINAMATH_CALUDE_customer_total_cost_l2842_284299


namespace NUMINAMATH_CALUDE_quadratic_equation_general_form_l2842_284245

theorem quadratic_equation_general_form :
  ∀ x : ℝ, (1 + 3 * x) * (x - 3) = 2 * x^2 + 1 ↔ x^2 - 8 * x - 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_general_form_l2842_284245


namespace NUMINAMATH_CALUDE_percentage_difference_l2842_284244

theorem percentage_difference : (0.6 * 50) - (0.42 * 30) = 17.4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2842_284244


namespace NUMINAMATH_CALUDE_quadratic_solution_properties_l2842_284271

theorem quadratic_solution_properties :
  ∀ (y₁ y₂ : ℝ), y₁^2 - 1500*y₁ + 750 = 0 ∧ y₂^2 - 1500*y₂ + 750 = 0 →
  y₁ + y₂ = 1500 ∧ y₁ * y₂ = 750 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_properties_l2842_284271


namespace NUMINAMATH_CALUDE_ben_age_is_five_l2842_284293

/-- Represents the ages of Amy, Ben, and Chris -/
structure Ages where
  amy : ℝ
  ben : ℝ
  chris : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 12
  (ages.amy + ages.ben + ages.chris) / 3 = 12 ∧
  -- Four years ago, Chris was twice as old as Amy was then
  ages.chris - 4 = 2 * (ages.amy - 4) ∧
  -- In 5 years, Ben's age will be 3/4 of Amy's age at that time
  ages.ben + 5 = 3 / 4 * (ages.amy + 5)

/-- The theorem to be proved -/
theorem ben_age_is_five :
  ∃ (ages : Ages), satisfies_conditions ages ∧ ages.ben = 5 := by
  sorry

end NUMINAMATH_CALUDE_ben_age_is_five_l2842_284293


namespace NUMINAMATH_CALUDE_correct_operation_is_subtraction_l2842_284257

-- Define the possible operations
inductive Operation
  | Add
  | Multiply
  | Divide
  | Subtract

-- Function to apply the operation
def applyOperation (op : Operation) (a b : ℤ) : ℤ :=
  match op with
  | Operation.Add => a + b
  | Operation.Multiply => a * b
  | Operation.Divide => a / b
  | Operation.Subtract => a - b

-- Theorem statement
theorem correct_operation_is_subtraction :
  ∃! op : Operation, (applyOperation op 8 4) + 6 - (3 - 2) = 9 :=
by sorry

end NUMINAMATH_CALUDE_correct_operation_is_subtraction_l2842_284257


namespace NUMINAMATH_CALUDE_convex_quadrilaterals_12_points_l2842_284265

/-- The number of different convex quadrilaterals that can be drawn from n distinct points
    on the circumference of a circle, where each vertex of the quadrilateral must be one
    of these n points. -/
def convex_quadrilaterals (n : ℕ) : ℕ := Nat.choose n 4

/-- Theorem stating that the number of convex quadrilaterals from 12 points is 3960 -/
theorem convex_quadrilaterals_12_points :
  convex_quadrilaterals 12 = 3960 := by
  sorry

#eval convex_quadrilaterals 12

end NUMINAMATH_CALUDE_convex_quadrilaterals_12_points_l2842_284265


namespace NUMINAMATH_CALUDE_divisors_of_fermat_like_number_l2842_284258

-- Define a function to represent the product of the first n primes in a list
def primeProduct : List Nat → Nat
  | [] => 1
  | p::ps => p * primeProduct ps

-- Define the main theorem
theorem divisors_of_fermat_like_number (n : Nat) (primes : List Nat) 
  (h_distinct : List.Pairwise (·≠·) primes)
  (h_prime : ∀ p ∈ primes, Nat.Prime p)
  (h_greater_than_three : ∀ p ∈ primes, p > 3)
  (h_length : primes.length = n) :
  (Nat.divisors (2^(primeProduct primes) + 1)).card ≥ 4^n := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_fermat_like_number_l2842_284258


namespace NUMINAMATH_CALUDE_total_spending_is_638_l2842_284229

/-- The total spending of Elizabeth, Emma, and Elsa -/
def total_spending (emma_spending : ℕ) : ℕ :=
  let elsa_spending := 2 * emma_spending
  let elizabeth_spending := 4 * elsa_spending
  emma_spending + elsa_spending + elizabeth_spending

/-- Theorem: The total spending is $638 when Emma spent $58 -/
theorem total_spending_is_638 : total_spending 58 = 638 := by
  sorry

end NUMINAMATH_CALUDE_total_spending_is_638_l2842_284229


namespace NUMINAMATH_CALUDE_sum_of_seventh_powers_l2842_284254

theorem sum_of_seventh_powers (α β γ : ℂ) 
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 6)
  (h3 : α^3 + β^3 + γ^3 = 14) :
  α^7 + β^7 + γ^7 = -98 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seventh_powers_l2842_284254


namespace NUMINAMATH_CALUDE_number_equation_solution_l2842_284238

theorem number_equation_solution : ∃ x : ℝ, (3 * x - 1 = 2 * x) ∧ (x = 1) := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l2842_284238


namespace NUMINAMATH_CALUDE_extremum_conditions_another_extremum_l2842_284256

/-- The function f with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f with respect to x -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_conditions (a b : ℝ) : 
  (f a b (-1) = 8 ∧ f' a b (-1) = 0) → (a = -2 ∧ b = -7) :=
by sorry

theorem another_extremum : 
  f (-2) (-7) (7/3) = -284/27 ∧ 
  (∀ x : ℝ, x ≠ -1 ∧ x ≠ 7/3 → |f (-2) (-7) x| ≤ |f (-2) (-7) (7/3)|) :=
by sorry

end NUMINAMATH_CALUDE_extremum_conditions_another_extremum_l2842_284256


namespace NUMINAMATH_CALUDE_mixture_ratio_after_replacement_l2842_284223

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the state of the liquid mixture -/
structure LiquidMixture where
  ratioAB : Ratio
  volumeA : ℝ
  totalVolume : ℝ

def initialMixture : LiquidMixture :=
  { ratioAB := { numerator := 4, denominator := 1 }
  , volumeA := 32
  , totalVolume := 40 }

def replacementVolume : ℝ := 20

/-- Calculates the new ratio after replacing some mixture with liquid B -/
def newRatio (initial : LiquidMixture) (replace : ℝ) : Ratio :=
  { numerator := 2
  , denominator := 3 }

theorem mixture_ratio_after_replacement :
  newRatio initialMixture replacementVolume = { numerator := 2, denominator := 3 } :=
sorry

end NUMINAMATH_CALUDE_mixture_ratio_after_replacement_l2842_284223


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l2842_284234

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 5123) % 12 = 2900 % 12 ∧
  ∀ (y : ℕ), y > 0 → (y + 5123) % 12 = 2900 % 12 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l2842_284234


namespace NUMINAMATH_CALUDE_fish_count_l2842_284296

def billy_fish : ℕ := 10

def tony_fish (billy : ℕ) : ℕ := 3 * billy

def sarah_fish (tony : ℕ) : ℕ := tony + 5

def bobby_fish (sarah : ℕ) : ℕ := 2 * sarah

def total_fish (billy tony sarah bobby : ℕ) : ℕ := billy + tony + sarah + bobby

theorem fish_count :
  total_fish billy_fish 
             (tony_fish billy_fish) 
             (sarah_fish (tony_fish billy_fish)) 
             (bobby_fish (sarah_fish (tony_fish billy_fish))) = 145 := by
  sorry

end NUMINAMATH_CALUDE_fish_count_l2842_284296


namespace NUMINAMATH_CALUDE_p_neither_necessary_nor_sufficient_for_q_l2842_284239

theorem p_neither_necessary_nor_sufficient_for_q (a : ℝ) : 
  (∃ x, x < 0 ∧ x > x^2) ∧ 
  (∃ y, y < 0 ∧ ¬(y > y^2)) ∧ 
  (∃ z, z > z^2 ∧ ¬(z < 0)) := by
sorry

end NUMINAMATH_CALUDE_p_neither_necessary_nor_sufficient_for_q_l2842_284239


namespace NUMINAMATH_CALUDE_largest_multiple_of_daytona_sharks_l2842_284213

def daytona_sharks : ℕ := 12
def cape_may_sharks : ℕ := 32

theorem largest_multiple_of_daytona_sharks : 
  ∃ (m : ℕ), m * daytona_sharks < cape_may_sharks ∧ 
  ∀ (n : ℕ), n * daytona_sharks < cape_may_sharks → n ≤ m ∧ 
  m = 2 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_daytona_sharks_l2842_284213


namespace NUMINAMATH_CALUDE_marble_count_theorem_l2842_284211

theorem marble_count_theorem (g y : ℚ) :
  (g - 3) / (g + y - 3) = 1 / 6 →
  g / (g + y - 4) = 1 / 4 →
  g + y = 18 :=
by sorry

end NUMINAMATH_CALUDE_marble_count_theorem_l2842_284211


namespace NUMINAMATH_CALUDE_john_works_fifty_weeks_l2842_284225

/-- Represents the number of weeks John works in a year -/
def weeks_worked (patients_hospital1 : ℕ) (patients_hospital2_increase : ℚ) 
  (days_per_week : ℕ) (total_patients_per_year : ℕ) : ℚ :=
  let patients_hospital2 := patients_hospital1 * (1 + patients_hospital2_increase)
  let patients_per_week := (patients_hospital1 + patients_hospital2) * days_per_week
  total_patients_per_year / patients_per_week

/-- Theorem stating that John works 50 weeks a year given the problem conditions -/
theorem john_works_fifty_weeks :
  weeks_worked 20 (1/5 : ℚ) 5 11000 = 50 := by sorry

end NUMINAMATH_CALUDE_john_works_fifty_weeks_l2842_284225


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_fourth_l2842_284205

theorem arctan_sum_equals_pi_fourth (m : ℕ+) : 
  (Real.arctan (1/7) + Real.arctan (1/8) + Real.arctan (1/9) + Real.arctan (1/m.val : ℝ) = π/4) → m = 133 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_fourth_l2842_284205


namespace NUMINAMATH_CALUDE_white_ring_weight_l2842_284236

/-- Given the weights of three plastic rings (orange, purple, and white) and their total weight,
    this theorem proves that the weight of the white ring is equal to the total weight
    minus the sum of the orange and purple ring weights. -/
theorem white_ring_weight 
  (orange_weight : ℝ) 
  (purple_weight : ℝ) 
  (total_weight : ℝ) 
  (h1 : orange_weight = 0.08333333333333333)
  (h2 : purple_weight = 0.3333333333333333)
  (h3 : total_weight = 0.8333333333) :
  total_weight - (orange_weight + purple_weight) = 0.41666666663333337 := by
  sorry

#eval Float.toString (0.8333333333 - (0.08333333333333333 + 0.3333333333333333))

end NUMINAMATH_CALUDE_white_ring_weight_l2842_284236


namespace NUMINAMATH_CALUDE_line_slope_perpendicular_lines_b_value_l2842_284246

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
axiom perpendicular_lines_slope_product (m₁ m₂ : ℝ) : 
  m₁ * m₂ = -1 ↔ (∃ (x₁ y₁ x₂ y₂ : ℝ), y₁ = m₁ * x₁ ∧ y₂ = m₂ * x₂ ∧ (y₂ - y₁) * (x₂ - x₁) = 0)

/-- The slope of a line ax + by + c = 0 where b ≠ 0 is -a/b -/
theorem line_slope (a b c : ℝ) (hb : b ≠ 0) :
  ∃ m : ℝ, m = -a / b ∧ ∀ x y : ℝ, a * x + b * y + c = 0 → y = m * x - c / b :=
sorry

theorem perpendicular_lines_b_value : 
  ∀ b : ℝ, (∃ x₁ y₁ x₂ y₂ : ℝ, 
    2 * x₁ + 3 * y₁ - 4 = 0 ∧ 
    b * x₂ + 3 * y₂ - 4 = 0 ∧ 
    (y₂ - y₁) * (x₂ - x₁) = 0) → 
  b = -9/2 :=
sorry

end NUMINAMATH_CALUDE_line_slope_perpendicular_lines_b_value_l2842_284246


namespace NUMINAMATH_CALUDE_number_division_l2842_284232

theorem number_division (x : ℚ) : x / 4 = 12 → x / 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l2842_284232


namespace NUMINAMATH_CALUDE_no_real_solutions_l2842_284259

theorem no_real_solutions : ∀ x : ℝ, ¬∃ y : ℝ, 
  (y = 3 * x - 1) ∧ (4 * y^2 + y + 3 = 3 * (8 * x^2 + 3 * y + 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2842_284259


namespace NUMINAMATH_CALUDE_burger_nonfiller_percentage_l2842_284251

/-- Given a burger with a total weight and filler weight, calculate the percentage that is not filler -/
theorem burger_nonfiller_percentage
  (total_weight : ℝ)
  (filler_weight : ℝ)
  (h1 : total_weight = 180)
  (h2 : filler_weight = 45)
  : (total_weight - filler_weight) / total_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_burger_nonfiller_percentage_l2842_284251


namespace NUMINAMATH_CALUDE_correct_average_l2842_284298

theorem correct_average (n : Nat) (incorrect_avg : ℚ) (incorrect_num : ℚ) (correct_num : ℚ) :
  n = 10 →
  incorrect_avg = 16 →
  incorrect_num = 26 →
  correct_num = 46 →
  (n : ℚ) * incorrect_avg + (correct_num - incorrect_num) = n * 18 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l2842_284298


namespace NUMINAMATH_CALUDE_composite_expression_l2842_284283

/-- A positive integer is composite if it can be expressed as a product of two integers,
    each greater than or equal to 2. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≥ 2 ∧ b ≥ 2 ∧ n = a * b

/-- Every composite number can be expressed as xy + xz + yz + 1,
    where x, y, and z are positive integers. -/
theorem composite_expression (c : ℕ) (h : IsComposite c) :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ c = x * y + x * z + y * z + 1 :=
sorry

end NUMINAMATH_CALUDE_composite_expression_l2842_284283


namespace NUMINAMATH_CALUDE_point_order_on_parabola_l2842_284272

/-- Parabola equation y = (x-1)^2 - 2 -/
def parabola (x y : ℝ) : Prop := y = (x - 1)^2 - 2

theorem point_order_on_parabola (a b c d : ℝ) :
  parabola a 2 →
  parabola b 6 →
  parabola c d →
  d < 1 →
  a < 0 →
  b > 0 →
  a < c ∧ c < b :=
sorry

end NUMINAMATH_CALUDE_point_order_on_parabola_l2842_284272


namespace NUMINAMATH_CALUDE_smallest_number_proof_l2842_284241

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  b = 28 →                 -- Median is 28
  c = b + 6 →              -- Largest number is 6 more than median
  a < b ∧ b < c →          -- Ordering of numbers
  a = 28 :=                -- Smallest number is 28
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l2842_284241


namespace NUMINAMATH_CALUDE_min_y_l2842_284263

variable (a b c d : ℝ)
variable (x : ℝ)

def y (x : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + c*(x - d)^2

theorem min_y :
  ∃ (x_min : ℝ), (∀ (x : ℝ), y x_min ≤ y x) ∧ x_min = (a + b + c*d) / (2 + c) :=
sorry

end NUMINAMATH_CALUDE_min_y_l2842_284263


namespace NUMINAMATH_CALUDE_first_hundred_contains_all_naturals_l2842_284261

/-- A sequence of 200 numbers partitioned into blue and red -/
def Sequence := Fin 200 → ℕ

/-- The property that blue numbers are in ascending order from 1 to 100 -/
def BlueAscending (s : Sequence) : Prop :=
  ∃ (blue : Fin 200 → Bool),
    (∀ i : Fin 200, blue i → s i ∈ Finset.range 101) ∧
    (∀ i j : Fin 200, i < j → blue i → blue j → s i < s j)

/-- The property that red numbers are in descending order from 100 to 1 -/
def RedDescending (s : Sequence) : Prop :=
  ∃ (red : Fin 200 → Bool),
    (∀ i : Fin 200, red i → s i ∈ Finset.range 101) ∧
    (∀ i j : Fin 200, i < j → red i → red j → s i > s j)

/-- The main theorem -/
theorem first_hundred_contains_all_naturals (s : Sequence)
    (h1 : BlueAscending s) (h2 : RedDescending s) :
    ∀ n : ℕ, n ∈ Finset.range 101 → ∃ i : Fin 100, s i = n :=
  sorry

end NUMINAMATH_CALUDE_first_hundred_contains_all_naturals_l2842_284261


namespace NUMINAMATH_CALUDE_max_value_fraction_max_value_achievable_l2842_284295

theorem max_value_fraction (y : ℝ) :
  y^2 / (y^4 + 4*y^3 + y^2 + 8*y + 16) ≤ 1/25 :=
by sorry

theorem max_value_achievable :
  ∃ y : ℝ, y^2 / (y^4 + 4*y^3 + y^2 + 8*y + 16) = 1/25 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_max_value_achievable_l2842_284295


namespace NUMINAMATH_CALUDE_principal_is_7500_l2842_284221

/-- Calculates the compound interest amount -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Proves that the principal is 7500 given the conditions -/
theorem principal_is_7500 
  (rate : ℝ) 
  (time : ℕ) 
  (interest : ℝ) 
  (h_rate : rate = 0.04) 
  (h_time : time = 2) 
  (h_interest : interest = 612) : 
  ∃ (principal : ℝ), principal = 7500 ∧ compound_interest principal rate time = interest :=
sorry

end NUMINAMATH_CALUDE_principal_is_7500_l2842_284221


namespace NUMINAMATH_CALUDE_function_inequality_range_l2842_284228

open Real

theorem function_inequality_range (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : f 0 = 1)
  (h3 : ∀ x, 3 * f x = f' x - 3) :
  {x | 4 * f x > f' x} = {x | x > log 2 / 3} := by sorry

end NUMINAMATH_CALUDE_function_inequality_range_l2842_284228


namespace NUMINAMATH_CALUDE_seventeen_in_binary_l2842_284290

theorem seventeen_in_binary : 
  (17 : ℕ).digits 2 = [1, 0, 0, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_seventeen_in_binary_l2842_284290


namespace NUMINAMATH_CALUDE_workshop_workers_l2842_284248

/-- The total number of workers in a workshop given specific salary conditions -/
theorem workshop_workers (average_salary : ℝ) (technician_salary : ℝ) (other_salary : ℝ) 
  (num_technicians : ℕ) :
  average_salary = 8000 →
  technician_salary = 12000 →
  other_salary = 6000 →
  num_technicians = 7 →
  ∃ (total_workers : ℕ), 
    (total_workers : ℝ) * average_salary = 
      (num_technicians : ℝ) * technician_salary + 
      ((total_workers - num_technicians) : ℝ) * other_salary ∧
    total_workers = 21 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l2842_284248


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2842_284260

theorem polynomial_coefficient_sum : 
  ∀ A B C D : ℝ, 
  (∀ x : ℝ, (x + 2) * (3 * x^2 - x + 5) = A * x^3 + B * x^2 + C * x + D) → 
  A + B + C + D = 21 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2842_284260


namespace NUMINAMATH_CALUDE_expand_expression_l2842_284220

theorem expand_expression (x y : ℝ) : 12 * (3 * x - 4 * y + 2) = 36 * x - 48 * y + 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2842_284220


namespace NUMINAMATH_CALUDE_linda_furniture_spending_l2842_284210

theorem linda_furniture_spending (original_savings : ℝ) (tv_cost : ℝ) 
  (h1 : original_savings = 1800)
  (h2 : tv_cost = 450) :
  (original_savings - tv_cost) / original_savings = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_linda_furniture_spending_l2842_284210


namespace NUMINAMATH_CALUDE_segment_length_line_circle_l2842_284226

/-- The length of the segment cut by a line from a circle -/
theorem segment_length_line_circle (a b c : ℝ) (x₀ y₀ r : ℝ) : 
  (∀ x y, (x - x₀)^2 + (y - y₀)^2 = r^2 → a*x + b*y + c = 0 → 
    2 * Real.sqrt (r^2 - (a*x₀ + b*y₀ + c)^2 / (a^2 + b^2)) = Real.sqrt 3) →
  x₀ = 1 ∧ y₀ = 0 ∧ r = 1 ∧ a = 1 ∧ b = Real.sqrt 3 ∧ c = -2 :=
by sorry

end NUMINAMATH_CALUDE_segment_length_line_circle_l2842_284226


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l2842_284270

theorem zeros_before_first_nonzero_digit (n : ℕ) (d : ℕ) (h : d = 2^7 * 5^9) :
  (∃ k : ℕ, (3 : ℚ) / d = (k : ℚ) / 10^9 ∧ 1 ≤ k ∧ k < 10) →
  (∃ m : ℕ, (3 : ℚ) / d = (m : ℚ) / 10^8 ∧ 10 ≤ m) →
  n = 8 :=
sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_digit_l2842_284270


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l2842_284291

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_surface_area := 6 * L^2
  let new_edge_length := 1.5 * L
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area * 100 = 125 := by
sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l2842_284291


namespace NUMINAMATH_CALUDE_count_integers_between_cubes_l2842_284206

theorem count_integers_between_cubes : 
  ∃ (n : ℕ), n = 37 ∧ 
  (∀ k : ℤ, (11.1 : ℝ)^3 < k ∧ k < (11.2 : ℝ)^3 ↔ 
   (⌊(11.1 : ℝ)^3⌋ + 1 : ℤ) ≤ k ∧ k ≤ (⌊(11.2 : ℝ)^3⌋ : ℤ)) ∧
  n = ⌊(11.2 : ℝ)^3⌋ - ⌊(11.1 : ℝ)^3⌋ :=
by sorry

end NUMINAMATH_CALUDE_count_integers_between_cubes_l2842_284206


namespace NUMINAMATH_CALUDE_infinite_solutions_diophantine_equation_l2842_284215

theorem infinite_solutions_diophantine_equation :
  ∃ f g h : ℕ → ℕ,
    (∀ t : ℕ, (f t)^2 + (g t)^3 = (h t)^5) ∧
    (∀ t₁ t₂ : ℕ, t₁ ≠ t₂ → (f t₁, g t₁, h t₁) ≠ (f t₂, g t₂, h t₂)) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_diophantine_equation_l2842_284215


namespace NUMINAMATH_CALUDE_yi_number_is_seven_eighths_l2842_284268

def card_numbers : Finset ℚ := {1/2, 3/4, 7/8, 15/16, 31/32}

def jia_statement (x : ℚ) : Prop :=
  x ∈ card_numbers ∧ x ≠ 1/2 ∧ x ≠ 31/32

def yi_statement (y : ℚ) : Prop :=
  y ∈ card_numbers ∧ y ≠ 3/4 ∧ y ≠ 15/16

theorem yi_number_is_seven_eighths :
  ∀ (x y : ℚ), x ∈ card_numbers → y ∈ card_numbers → x ≠ y →
  jia_statement x → yi_statement y → y = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_yi_number_is_seven_eighths_l2842_284268


namespace NUMINAMATH_CALUDE_inequality_proof_l2842_284279

theorem inequality_proof (x : ℝ) : 2 * (5 * x + 3) ≤ x - 3 * (1 - 2 * x) → x ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2842_284279


namespace NUMINAMATH_CALUDE_root_sum_square_l2842_284286

theorem root_sum_square (α β : ℝ) : 
  (α^2 + α - 2023 = 0) → 
  (β^2 + β - 2023 = 0) → 
  α^2 + 2*α + β = 2022 := by
sorry

end NUMINAMATH_CALUDE_root_sum_square_l2842_284286


namespace NUMINAMATH_CALUDE_same_terminal_side_angle_with_same_terminal_side_l2842_284208

theorem same_terminal_side (x y : Real) : 
  x = y + 2 * Real.pi * ↑(Int.floor ((x - y) / (2 * Real.pi))) → 
  ∃ k : ℤ, y = x + 2 * Real.pi * k := by
  sorry

theorem angle_with_same_terminal_side : 
  ∃ k : ℤ, -π/3 = 5*π/3 + 2*π*k := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_angle_with_same_terminal_side_l2842_284208


namespace NUMINAMATH_CALUDE_raisin_nut_cost_ratio_l2842_284249

theorem raisin_nut_cost_ratio :
  ∀ (r n : ℝ),
  r > 0 →
  n > 0 →
  (5 * r) / (5 * r + 4 * n) = 0.29411764705882354 →
  n / r = 3 :=
by sorry

end NUMINAMATH_CALUDE_raisin_nut_cost_ratio_l2842_284249


namespace NUMINAMATH_CALUDE_phone_number_a_is_five_l2842_284233

/-- Represents a valid 10-digit telephone number -/
structure PhoneNumber where
  digits : Fin 10 → Fin 10
  all_different : ∀ i j, i ≠ j → digits i ≠ digits j
  decreasing_abc : digits 0 > digits 1 ∧ digits 1 > digits 2
  decreasing_def : digits 3 > digits 4 ∧ digits 4 > digits 5
  decreasing_ghij : digits 6 > digits 7 ∧ digits 7 > digits 8 ∧ digits 8 > digits 9
  consecutive_def : ∃ n : ℕ, digits 3 = n + 2 ∧ digits 4 = n + 1 ∧ digits 5 = n
  consecutive_ghij : ∃ n : ℕ, digits 6 = n + 3 ∧ digits 7 = n + 2 ∧ digits 8 = n + 1 ∧ digits 9 = n
  sum_abc : digits 0 + digits 1 + digits 2 = 10

theorem phone_number_a_is_five (p : PhoneNumber) : p.digits 0 = 5 := by
  sorry

end NUMINAMATH_CALUDE_phone_number_a_is_five_l2842_284233


namespace NUMINAMATH_CALUDE_last_number_proof_l2842_284292

theorem last_number_proof (A B C D : ℝ) 
  (h1 : (A + B + C) / 3 = 6)
  (h2 : (B + C + D) / 3 = 3)
  (h3 : A + D = 13) :
  D = 2 := by
sorry

end NUMINAMATH_CALUDE_last_number_proof_l2842_284292


namespace NUMINAMATH_CALUDE_complex_magnitude_l2842_284284

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I) = 4 - 2 * Complex.I) :
  Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2842_284284


namespace NUMINAMATH_CALUDE_triangle_side_length_l2842_284204

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = Real.pi / 3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2842_284204


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_of_250_l2842_284243

theorem closest_integer_to_cube_root_of_250 :
  ∀ n : ℤ, |n^3 - 250| ≥ |6^3 - 250| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_of_250_l2842_284243


namespace NUMINAMATH_CALUDE_grid_recoloring_theorem_l2842_284252

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents the grid -/
def Grid := Fin 99 → Fin 99 → Color

/-- Represents a row or column index -/
def Index := Fin 99

/-- Represents a recoloring operation -/
inductive RecolorOp
| Row (i : Index)
| Col (j : Index)

/-- Applies a recoloring operation to a grid -/
def applyRecolor (g : Grid) (op : RecolorOp) : Grid :=
  sorry

/-- Checks if all cells in the grid have the same color -/
def isMonochromatic (g : Grid) : Prop :=
  sorry

/-- The main theorem -/
theorem grid_recoloring_theorem (g : Grid) :
  ∃ (ops : List RecolorOp), isMonochromatic (ops.foldl applyRecolor g) :=
sorry

end NUMINAMATH_CALUDE_grid_recoloring_theorem_l2842_284252
