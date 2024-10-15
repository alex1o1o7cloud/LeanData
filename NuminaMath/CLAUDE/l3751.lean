import Mathlib

namespace NUMINAMATH_CALUDE_prob_same_group_is_one_third_l3751_375108

/-- The number of interest groups -/
def num_groups : ℕ := 3

/-- The set of all possible outcomes when two students choose interest groups -/
def total_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range num_groups) (Finset.range num_groups)

/-- The set of outcomes where both students choose the same group -/
def same_group_outcomes : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 = p.2) total_outcomes

/-- The probability of two students choosing the same interest group -/
def prob_same_group : ℚ :=
  (same_group_outcomes.card : ℚ) / (total_outcomes.card : ℚ)

theorem prob_same_group_is_one_third :
  prob_same_group = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_same_group_is_one_third_l3751_375108


namespace NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l3751_375141

/-- A curve represented by the equation mx^2 + ny^2 = 1 -/
structure Curve (m n : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1

/-- Predicate to check if a curve is an ellipse -/
def IsEllipse (c : Curve m n) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

/-- The main theorem stating that mn > 0 is a necessary but not sufficient condition for the curve to be an ellipse -/
theorem mn_positive_necessary_not_sufficient (m n : ℝ) :
  (∀ (c : Curve m n), IsEllipse c → m * n > 0) ∧
  ¬(∀ (c : Curve m n), m * n > 0 → IsEllipse c) :=
sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l3751_375141


namespace NUMINAMATH_CALUDE_initial_candies_are_52_or_56_l3751_375147

def initial_candies : Set ℕ :=
  {x : ℕ | 
    -- The number of candies after Tracy ate 1/4
    ∃ (a : ℕ), 3 * x = 4 * a ∧
    -- The number of candies after giving 1/3 to Sam
    ∃ (b : ℕ), 2 * a = 3 * b ∧
    -- The number of candies after Tracy and her dad ate 20
    b ≥ 20 ∧
    -- The number of candies after Tracy's sister took 2 to 6
    ∃ (c : ℕ), b - 20 - c = 4 ∧ 2 ≤ c ∧ c ≤ 6
  }

theorem initial_candies_are_52_or_56 : initial_candies = {52, 56} := by sorry

end NUMINAMATH_CALUDE_initial_candies_are_52_or_56_l3751_375147


namespace NUMINAMATH_CALUDE_smoothie_cost_l3751_375191

def burger_cost : ℝ := 5
def sandwich_cost : ℝ := 4
def total_order_cost : ℝ := 17
def num_smoothies : ℕ := 2

theorem smoothie_cost :
  let non_smoothie_cost := burger_cost + sandwich_cost
  let smoothie_total_cost := total_order_cost - non_smoothie_cost
  let smoothie_cost := smoothie_total_cost / num_smoothies
  smoothie_cost = 4 := by sorry

end NUMINAMATH_CALUDE_smoothie_cost_l3751_375191


namespace NUMINAMATH_CALUDE_probability_not_adjacent_l3751_375137

-- Define the total number of chairs
def total_chairs : ℕ := 10

-- Define the number of available chairs (excluding first and last)
def available_chairs : ℕ := total_chairs - 2

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the number of adjacent pairs in the available chairs
def adjacent_pairs : ℕ := available_chairs - 1

-- Theorem statement
theorem probability_not_adjacent :
  (1 : ℚ) - (adjacent_pairs : ℚ) / (choose available_chairs 2) = 3/4 :=
sorry

end NUMINAMATH_CALUDE_probability_not_adjacent_l3751_375137


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3751_375120

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area : 
  let r : ℝ := 3
  let h : ℝ := 4
  let l : ℝ := Real.sqrt (r^2 + h^2)
  π * r * l = 15 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3751_375120


namespace NUMINAMATH_CALUDE_company_donation_problem_l3751_375163

theorem company_donation_problem (donation_A donation_B : ℕ) 
  (average_difference : ℕ) (percentage_difference : ℚ) :
  donation_A = 60000 →
  donation_B = 60000 →
  average_difference = 40 →
  percentage_difference = 1/5 →
  ∃ (people_A people_B : ℕ),
    people_A = (1 + percentage_difference) * people_B ∧
    (donation_B : ℚ) / people_B - (donation_A : ℚ) / people_A = average_difference ∧
    people_A = 300 ∧
    people_B = 250 := by
  sorry

end NUMINAMATH_CALUDE_company_donation_problem_l3751_375163


namespace NUMINAMATH_CALUDE_distinct_roots_quadratic_l3751_375167

theorem distinct_roots_quadratic (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 - m*x₁ - 2 = 0) ∧ 
  (x₂^2 - m*x₂ - 2 = 0) := by
sorry

end NUMINAMATH_CALUDE_distinct_roots_quadratic_l3751_375167


namespace NUMINAMATH_CALUDE_intersection_segment_length_l3751_375136

/-- The length of the line segment formed by the intersection of a line and an ellipse -/
theorem intersection_segment_length 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (h_ecc : (a^2 - b^2) / a^2 = 1/2) 
  (h_focal : 2 * Real.sqrt (a^2 - b^2) = 2) : 
  ∃ (A B : ℝ × ℝ), 
    (A.2 = -A.1 + 1 ∧ A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧ 
    (B.2 = -B.1 + 1 ∧ B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧ 
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 4 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l3751_375136


namespace NUMINAMATH_CALUDE_system_of_inequalities_l3751_375151

theorem system_of_inequalities (x : ℝ) :
  (2 * x ≤ 6 - x) ∧ (3 * x + 1 > 2 * (x - 1)) → -3 < x ∧ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l3751_375151


namespace NUMINAMATH_CALUDE_circle_inequality_l3751_375124

theorem circle_inequality (a b c d : ℝ) (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a * b + c * d = 1)
  (h1 : x1^2 + y1^2 = 1) (h2 : x2^2 + y2^2 = 1) 
  (h3 : x3^2 + y3^2 = 1) (h4 : x4^2 + y4^2 = 1) :
  (a*y1 + b*y2 + c*y3 + d*y4)^2 + (a*x4 + b*x3 + c*x2 + d*x1)^2 
  ≤ 2 * ((a^2 + b^2)/(a*b) + (c^2 + d^2)/(c*d)) := by
  sorry

end NUMINAMATH_CALUDE_circle_inequality_l3751_375124


namespace NUMINAMATH_CALUDE_mans_speed_against_stream_l3751_375125

theorem mans_speed_against_stream 
  (rate : ℝ) 
  (speed_with_stream : ℝ) 
  (h1 : rate = 4) 
  (h2 : speed_with_stream = 12) : 
  abs (rate - (speed_with_stream - rate)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_against_stream_l3751_375125


namespace NUMINAMATH_CALUDE_remainder_theorem_l3751_375184

theorem remainder_theorem (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) (h4 : u + v < y) :
  (x + 3 * u * y + u) % y = u + v :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3751_375184


namespace NUMINAMATH_CALUDE_number_times_99_equals_2376_l3751_375199

theorem number_times_99_equals_2376 : ∃ x : ℕ, x * 99 = 2376 ∧ x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_times_99_equals_2376_l3751_375199


namespace NUMINAMATH_CALUDE_problem_statement_l3751_375164

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 1 / y) :
  (x - 1 / x) * (y + 1 / y) = x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3751_375164


namespace NUMINAMATH_CALUDE_greatest_prime_factor_f_36_l3751_375155

def f (m : ℕ) : ℕ := Finset.prod (Finset.filter (λ x => Even x) (Finset.range (m + 1))) id

theorem greatest_prime_factor_f_36 :
  ∃ (p : ℕ), Prime p ∧ p ∣ f 36 ∧ ∀ (q : ℕ), Prime q → q ∣ f 36 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_f_36_l3751_375155


namespace NUMINAMATH_CALUDE_longer_diagonal_squared_is_80_l3751_375172

/-- Represents a parallelogram LMNO with specific properties -/
structure Parallelogram where
  area : ℝ
  xy : ℝ
  zw : ℝ
  (area_positive : area > 0)
  (xy_positive : xy > 0)
  (zw_positive : zw > 0)

/-- The square of the longer diagonal of the parallelogram -/
def longer_diagonal_squared (p : Parallelogram) : ℝ := sorry

/-- Theorem stating the square of the longer diagonal equals 80 -/
theorem longer_diagonal_squared_is_80 (p : Parallelogram) 
  (h1 : p.area = 24) 
  (h2 : p.xy = 8) 
  (h3 : p.zw = 10) : 
  longer_diagonal_squared p = 80 := by sorry

end NUMINAMATH_CALUDE_longer_diagonal_squared_is_80_l3751_375172


namespace NUMINAMATH_CALUDE_imaginary_number_condition_l3751_375118

theorem imaginary_number_condition (x : ℝ) : 
  let z : ℂ := Complex.mk (x^2 - 1) (x - 1)
  (z.re = 0 ∧ z.im ≠ 0) → x = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_number_condition_l3751_375118


namespace NUMINAMATH_CALUDE_train_crossing_time_l3751_375189

/- Define the train speed in m/s -/
def train_speed : ℝ := 20

/- Define the time to pass a man on the platform in seconds -/
def time_pass_man : ℝ := 18

/- Define the length of the platform in meters -/
def platform_length : ℝ := 260

/- Calculate the length of the train -/
def train_length : ℝ := train_speed * time_pass_man

/- Calculate the total distance the train needs to travel -/
def total_distance : ℝ := train_length + platform_length

/- Theorem: The time for the train to cross the platform is 31 seconds -/
theorem train_crossing_time : 
  total_distance / train_speed = 31 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3751_375189


namespace NUMINAMATH_CALUDE_bob_age_proof_l3751_375130

theorem bob_age_proof :
  ∃! x : ℕ, 
    x > 0 ∧ 
    ∃ y : ℕ, x - 2 = y^2 ∧
    ∃ z : ℕ, x + 2 = z^3 ∧
    x = 123 := by
  sorry

end NUMINAMATH_CALUDE_bob_age_proof_l3751_375130


namespace NUMINAMATH_CALUDE_largest_fraction_l3751_375196

theorem largest_fraction : 
  let fractions := [5/11, 9/20, 23/47, 105/209, 205/409]
  ∀ x ∈ fractions, (105 : ℚ) / 209 ≥ x := by sorry

end NUMINAMATH_CALUDE_largest_fraction_l3751_375196


namespace NUMINAMATH_CALUDE_combined_cost_theorem_l3751_375134

-- Define the cost prices of the two articles
def cost_price_1 : ℝ := sorry
def cost_price_2 : ℝ := sorry

-- Define the conditions
def condition_1 : Prop :=
  (350 - cost_price_1) = 1.12 * (280 - cost_price_1)

def condition_2 : Prop :=
  (420 - cost_price_2) = 1.08 * (380 - cost_price_2)

-- Theorem to prove
theorem combined_cost_theorem :
  condition_1 ∧ condition_2 → cost_price_1 + cost_price_2 = 423.33 :=
by sorry

end NUMINAMATH_CALUDE_combined_cost_theorem_l3751_375134


namespace NUMINAMATH_CALUDE_num_spiders_is_one_l3751_375176

/-- The number of spiders in a pet shop. -/
def num_spiders : ℕ :=
  let num_birds : ℕ := 3
  let num_dogs : ℕ := 5
  let num_snakes : ℕ := 4
  let total_legs : ℕ := 34
  let bird_legs : ℕ := 2
  let dog_legs : ℕ := 4
  let snake_legs : ℕ := 0
  let spider_legs : ℕ := 8
  (total_legs - (num_birds * bird_legs + num_dogs * dog_legs + num_snakes * snake_legs)) / spider_legs

theorem num_spiders_is_one : num_spiders = 1 := by
  sorry

end NUMINAMATH_CALUDE_num_spiders_is_one_l3751_375176


namespace NUMINAMATH_CALUDE_last_digit_of_product_l3751_375153

theorem last_digit_of_product (n : ℕ) : 
  (3^2001 * 7^2002 * 13^2003) % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_product_l3751_375153


namespace NUMINAMATH_CALUDE_circle_center_l3751_375109

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2,
    where (h, k) is the center and r is the radius. -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_center (x y : ℝ) :
  CircleEquation 3 0 3 x y → (3, 0) = (3, 0) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3751_375109


namespace NUMINAMATH_CALUDE_alice_and_dave_weight_l3751_375146

theorem alice_and_dave_weight
  (alice_bob : ℝ)
  (bob_charlie : ℝ)
  (charlie_dave : ℝ)
  (h1 : alice_bob = 230)
  (h2 : bob_charlie = 220)
  (h3 : charlie_dave = 250) :
  ∃ (alice dave : ℝ), alice + dave = 260 :=
by
  sorry

end NUMINAMATH_CALUDE_alice_and_dave_weight_l3751_375146


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3751_375100

-- Problem 1
theorem problem_1 : (-3/7) + 1/5 + 2/7 + (-6/5) = -8/7 := by sorry

-- Problem 2
theorem problem_2 : -(-1) + 3^2 / (1-4) * 2 = -5 := by sorry

-- Problem 3
theorem problem_3 : (-1/6)^2 / ((1/2 - 1/3)^2) / |(-6)|^2 = 1/36 := by sorry

-- Problem 4
theorem problem_4 : (-1)^1000 - 2.45 * 8 + 2.55 * (-8) = -39 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3751_375100


namespace NUMINAMATH_CALUDE_simplify_expression_l3751_375182

theorem simplify_expression (r : ℝ) : 120 * r - 68 * r + 15 * r = 67 * r := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3751_375182


namespace NUMINAMATH_CALUDE_next_joint_work_day_is_360_l3751_375115

/-- Represents the work schedule of a staff member -/
structure WorkSchedule where
  cycle : Nat
  deriving Repr

/-- Represents the community center with its staff members -/
structure CommunityCenter where
  alan : WorkSchedule
  berta : WorkSchedule
  carlos : WorkSchedule
  dora : WorkSchedule

/-- Calculates the next day when all staff members work together -/
def nextJointWorkDay (center : CommunityCenter) : Nat :=
  Nat.lcm center.alan.cycle (Nat.lcm center.berta.cycle (Nat.lcm center.carlos.cycle center.dora.cycle))

/-- The main theorem: proving that the next joint work day is 360 days from today -/
theorem next_joint_work_day_is_360 (center : CommunityCenter) 
  (h1 : center.alan.cycle = 5)
  (h2 : center.berta.cycle = 6)
  (h3 : center.carlos.cycle = 8)
  (h4 : center.dora.cycle = 9) :
  nextJointWorkDay center = 360 := by
  sorry

end NUMINAMATH_CALUDE_next_joint_work_day_is_360_l3751_375115


namespace NUMINAMATH_CALUDE_cut_triangles_perimeter_sum_l3751_375105

theorem cut_triangles_perimeter_sum (large_perimeter hexagon_perimeter : ℝ) :
  large_perimeter = 60 →
  hexagon_perimeter = 40 →
  ∃ (x y z : ℝ),
    x + y + z = large_perimeter / 3 - hexagon_perimeter / 3 ∧
    3 * (x + y + z) = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_cut_triangles_perimeter_sum_l3751_375105


namespace NUMINAMATH_CALUDE_pool_filling_rate_l3751_375177

/-- Given a pool with the following properties:
  * Capacity: 60 gallons
  * Leak rate: 0.1 gallons per minute
  * Filling time: 40 minutes
  Prove that the rate at which water is provided to fill the pool is 1.6 gallons per minute. -/
theorem pool_filling_rate 
  (capacity : ℝ) 
  (leak_rate : ℝ) 
  (filling_time : ℝ) 
  (h1 : capacity = 60) 
  (h2 : leak_rate = 0.1) 
  (h3 : filling_time = 40) : 
  ∃ (fill_rate : ℝ), 
    fill_rate = 1.6 ∧ 
    (fill_rate - leak_rate) * filling_time = capacity :=
by sorry

end NUMINAMATH_CALUDE_pool_filling_rate_l3751_375177


namespace NUMINAMATH_CALUDE_parabola_properties_l3751_375190

theorem parabola_properties (a b c : ℝ) (h1 : 0 < a) (h2 : a < c) 
  (h3 : a + b + c = 0) : 
  (2 * a + b < 0 ∧ 
   ∃ x : ℝ, x > 1 ∧ (2 * a * x + b ≤ 0) ∧
   (b^2 + 4 * a^2 > 0)) ∧
  ¬(∀ x : ℝ, x > 1 → 2 * a * x + b > 0) := by
sorry

end NUMINAMATH_CALUDE_parabola_properties_l3751_375190


namespace NUMINAMATH_CALUDE_oxen_count_l3751_375121

/-- The number of oxen in the first group that can plough 1/7th of a field in 2 days. -/
def first_group : ℕ := sorry

/-- The time it takes for the first group to plough the entire field. -/
def total_time : ℕ := 14

/-- The fraction of the field ploughed by the first group. -/
def ploughed_fraction : ℚ := 1/7

/-- The number of oxen in the second group. -/
def second_group : ℕ := 18

/-- The time it takes for the second group to plough the remaining field. -/
def remaining_time : ℕ := 20

/-- The fraction of the field ploughed by the second group. -/
def remaining_fraction : ℚ := 6/7

theorem oxen_count :
  (first_group * total_time) / 1 = (second_group * remaining_time) / remaining_fraction →
  first_group = 30 := by sorry

end NUMINAMATH_CALUDE_oxen_count_l3751_375121


namespace NUMINAMATH_CALUDE_continuous_function_satisfying_integral_equation_is_constant_l3751_375195

/-- A continuous function satisfying the given integral equation is constant -/
theorem continuous_function_satisfying_integral_equation_is_constant 
  (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ a b : ℝ, (a^2 + a*b + b^2) * ∫ x in a..b, f x = 3 * ∫ x in a..b, x^2 * f x) : 
  ∃ C : ℝ, ∀ x : ℝ, f x = C := by
sorry

end NUMINAMATH_CALUDE_continuous_function_satisfying_integral_equation_is_constant_l3751_375195


namespace NUMINAMATH_CALUDE_cannot_tile_8x9_with_6x1_l3751_375145

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a rectangular tile -/
structure Tile :=
  (length : ℕ)
  (width : ℕ)

/-- Defines what it means for a board to be tileable by a given tile -/
def is_tileable (b : Board) (t : Tile) : Prop :=
  ∃ (n : ℕ), n * (t.length * t.width) = b.rows * b.cols ∧
  (t.length ∣ b.rows ∨ t.length ∣ b.cols) ∧
  (t.width ∣ b.rows ∨ t.width ∣ b.cols)

/-- The main theorem stating that an 8x9 board cannot be tiled with 6x1 tiles -/
theorem cannot_tile_8x9_with_6x1 :
  ¬ is_tileable (Board.mk 8 9) (Tile.mk 6 1) :=
sorry

end NUMINAMATH_CALUDE_cannot_tile_8x9_with_6x1_l3751_375145


namespace NUMINAMATH_CALUDE_surface_area_unchanged_l3751_375131

/-- Represents a cube with given side length -/
structure Cube where
  side : ℝ
  side_pos : side > 0

/-- Calculates the surface area of a cube -/
def surfaceArea (c : Cube) : ℝ := 6 * c.side^2

/-- Represents the original cube -/
def originalCube : Cube := ⟨4, by norm_num⟩

/-- Represents a corner cube -/
def cornerCube : Cube := ⟨2, by norm_num⟩

/-- The number of corners in a cube -/
def numCorners : ℕ := 8

theorem surface_area_unchanged : 
  surfaceArea originalCube = surfaceArea originalCube - numCorners * (
    3 * cornerCube.side^2 - 3 * cornerCube.side^2
  ) := by sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_l3751_375131


namespace NUMINAMATH_CALUDE_not_not_or_implies_or_at_least_one_true_l3751_375161

theorem not_not_or_implies_or (p q : Prop) : ¬¬(p ∨ q) → (p ∨ q) := by
  sorry

theorem at_least_one_true (p q : Prop) : ¬¬(p ∨ q) → (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_not_not_or_implies_or_at_least_one_true_l3751_375161


namespace NUMINAMATH_CALUDE_coal_shoveling_time_l3751_375113

/-- Given that 10 people can shovel 10,000 pounds of coal in 10 days,
    prove that 5 people will take 80 days to shovel 40,000 pounds of coal. -/
theorem coal_shoveling_time 
  (people : ℕ) 
  (days : ℕ) 
  (coal : ℕ) 
  (h1 : people = 10) 
  (h2 : days = 10) 
  (h3 : coal = 10000) :
  (people / 2) * (coal * 4 / (people * days)) = 80 := by
  sorry

#check coal_shoveling_time

end NUMINAMATH_CALUDE_coal_shoveling_time_l3751_375113


namespace NUMINAMATH_CALUDE_point_distance_theorem_l3751_375168

theorem point_distance_theorem (x y : ℝ) (h1 : x > 2) :
  y = 14 ∧ (x - 2)^2 + (y - 8)^2 = 12^2 →
  x^2 + y^2 = 284 := by sorry

end NUMINAMATH_CALUDE_point_distance_theorem_l3751_375168


namespace NUMINAMATH_CALUDE_ellipse_tangent_product_l3751_375114

/-- An ellipse with its key points -/
structure Ellipse where
  A : ℝ × ℝ  -- Major axis endpoint
  B : ℝ × ℝ  -- Minor axis endpoint
  F₁ : ℝ × ℝ  -- Focus 1
  F₂ : ℝ × ℝ  -- Focus 2

/-- Vector dot product -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Vector from point to point -/
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

/-- Tangent of angle between three points -/
noncomputable def tan_angle (p q r : ℝ × ℝ) : ℝ :=
  let v1 := vector q p
  let v2 := vector q r
  (v1.2 * v2.1 - v1.1 * v2.2) / (v1.1 * v2.1 + v1.2 * v2.2)

/-- Main theorem -/
theorem ellipse_tangent_product (Γ : Ellipse) 
  (h : dot_product (vector Γ.A Γ.F₁) (vector Γ.A Γ.F₂) + 
       dot_product (vector Γ.B Γ.F₁) (vector Γ.B Γ.F₂) = 0) : 
  tan_angle Γ.A Γ.B Γ.F₁ * tan_angle Γ.A Γ.B Γ.F₂ = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_tangent_product_l3751_375114


namespace NUMINAMATH_CALUDE_brother_grade_is_two_l3751_375138

structure Brother where
  grade : ℕ

structure Grandmother where
  sneeze : Bool

def tells_truth (b : Brother) (statement : ℕ) : Prop :=
  b.grade = statement

def grandmother_sneezes (g : Grandmother) (b : Brother) (statement : ℕ) : Prop :=
  tells_truth b statement → g.sneeze = true

theorem brother_grade_is_two (b : Brother) (g : Grandmother) :
  grandmother_sneezes g b 5 ∧ g.sneeze = false →
  grandmother_sneezes g b 4 ∧ g.sneeze = true →
  grandmother_sneezes g b 3 ∧ g.sneeze = false →
  b.grade = 2 := by
  sorry

end NUMINAMATH_CALUDE_brother_grade_is_two_l3751_375138


namespace NUMINAMATH_CALUDE_slide_ratio_problem_l3751_375166

/-- Given that x boys initially went down a slide, y more boys joined them later,
    and the ratio of boys who went down the slide to boys who watched (z) is 5:3,
    prove that z = 21 when x = 22 and y = 13. -/
theorem slide_ratio_problem (x y : ℕ) (z : ℚ) 
    (h1 : x = 22)
    (h2 : y = 13)
    (h3 : (5 : ℚ) / 3 = (x + y : ℚ) / z) : 
  z = 21 := by
  sorry

end NUMINAMATH_CALUDE_slide_ratio_problem_l3751_375166


namespace NUMINAMATH_CALUDE_d_neither_sufficient_nor_necessary_for_a_l3751_375116

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
variable (h1 : (A → B) ∧ ¬(B → A))  -- A is sufficient but not necessary for B
variable (h2 : (B ↔ C))             -- B is necessary and sufficient for C
variable (h3 : (D → C) ∧ ¬(C → D))  -- C is necessary but not sufficient for D

-- Theorem to prove
theorem d_neither_sufficient_nor_necessary_for_a :
  ¬((D → A) ∧ (A → D)) :=
sorry

end NUMINAMATH_CALUDE_d_neither_sufficient_nor_necessary_for_a_l3751_375116


namespace NUMINAMATH_CALUDE_ace_then_king_probability_l3751_375129

/-- The number of cards in a standard deck -/
def deckSize : ℕ := 52

/-- The number of Aces in a standard deck -/
def numAces : ℕ := 4

/-- The number of Kings in a standard deck -/
def numKings : ℕ := 4

/-- The probability of drawing an Ace followed by a King from a standard deck -/
theorem ace_then_king_probability :
  (numAces / deckSize) * (numKings / (deckSize - 1)) = 4 / 663 := by
  sorry


end NUMINAMATH_CALUDE_ace_then_king_probability_l3751_375129


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3751_375179

theorem quadratic_inequality (a b c d : ℝ) 
  (h1 : b > d) 
  (h2 : b > 0) 
  (h3 : d > 0) 
  (h4 : Real.sqrt (a^2 - 4*b) > Real.sqrt (c^2 - 4*d)) : 
  a^2 - c^2 > b - d := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3751_375179


namespace NUMINAMATH_CALUDE_animal_lifespans_l3751_375173

theorem animal_lifespans (bat hamster frog tortoise : ℝ) : 
  hamster = bat - 6 →
  frog = 4 * hamster →
  tortoise = 2 * bat →
  bat + hamster + frog + tortoise = 62 →
  bat = 11.5 := by
sorry

end NUMINAMATH_CALUDE_animal_lifespans_l3751_375173


namespace NUMINAMATH_CALUDE_expression_equals_five_halves_l3751_375148

theorem expression_equals_five_halves :
  Real.sqrt 12 - 2 * Real.cos (π / 6) + |Real.sqrt 3 - 2| + 2^(-1 : ℤ) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_five_halves_l3751_375148


namespace NUMINAMATH_CALUDE_expected_cards_theorem_l3751_375175

/-- A deck of cards with N cards, of which three are Aces -/
structure Deck :=
  (N : ℕ)
  (ace_count : Fin 3)

/-- The expected number of cards turned over until the second Ace appears -/
def expected_cards_until_second_ace (d : Deck) : ℚ :=
  (d.N + 1) / 2

/-- Theorem stating that the expected number of cards turned over until the second Ace appears is (N+1)/2 -/
theorem expected_cards_theorem (d : Deck) :
  expected_cards_until_second_ace d = (d.N + 1) / 2 := by
  sorry

#check expected_cards_theorem

end NUMINAMATH_CALUDE_expected_cards_theorem_l3751_375175


namespace NUMINAMATH_CALUDE_right_triangle_PR_length_l3751_375181

-- Define the triangle PQR
structure RightTriangle where
  PQ : ℝ
  PR : ℝ
  sinR : ℝ
  angle_Q_is_right : True  -- Represents ∠Q = 90°

-- State the theorem
theorem right_triangle_PR_length 
  (triangle : RightTriangle) 
  (h1 : triangle.PQ = 9) 
  (h2 : triangle.sinR = 3/5) : 
  triangle.PR = 15 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_PR_length_l3751_375181


namespace NUMINAMATH_CALUDE_bananas_left_in_jar_l3751_375104

theorem bananas_left_in_jar (original : ℕ) (removed : ℕ) (h1 : original = 46) (h2 : removed = 5) :
  original - removed = 41 := by
  sorry

end NUMINAMATH_CALUDE_bananas_left_in_jar_l3751_375104


namespace NUMINAMATH_CALUDE_pet_store_cats_l3751_375178

theorem pet_store_cats (initial_birds initial_puppies initial_spiders : ℕ)
  (sold_birds adopted_puppies loose_spiders : ℕ)
  (total_left : ℕ) :
  initial_birds = 12 →
  initial_puppies = 9 →
  initial_spiders = 15 →
  sold_birds = initial_birds / 2 →
  adopted_puppies = 3 →
  loose_spiders = 7 →
  total_left = 25 →
  ∃ initial_cats : ℕ,
    initial_cats = 5 ∧
    total_left = initial_birds - sold_birds +
                 initial_puppies - adopted_puppies +
                 initial_cats +
                 initial_spiders - loose_spiders :=
by sorry

end NUMINAMATH_CALUDE_pet_store_cats_l3751_375178


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l3751_375162

theorem gain_percent_calculation (C S : ℝ) (h : C > 0) :
  50 * C = 45 * S → (S - C) / C * 100 = 100 / 9 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l3751_375162


namespace NUMINAMATH_CALUDE_rectangle_area_l3751_375106

/-- Given a rectangle with length four times its width and perimeter 200 cm, 
    its area is 1600 square centimeters. -/
theorem rectangle_area (w : ℝ) (h₁ : w > 0) : 
  let l := 4 * w
  2 * l + 2 * w = 200 → l * w = 1600 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3751_375106


namespace NUMINAMATH_CALUDE_equation_solution_l3751_375103

theorem equation_solution (x : ℝ) : 9 / (1 + 4 / x) = 1 → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3751_375103


namespace NUMINAMATH_CALUDE_constant_age_difference_l3751_375192

/-- The age difference between two brothers remains constant over time -/
theorem constant_age_difference (a b x : ℕ) : (a + x) - (b + x) = a - b := by
  sorry

end NUMINAMATH_CALUDE_constant_age_difference_l3751_375192


namespace NUMINAMATH_CALUDE_ice_cream_theorem_ice_cream_distribution_count_l3751_375171

def ice_cream_distribution (n : ℕ) : ℕ :=
  (Nat.choose (n + 2) 2)

theorem ice_cream_theorem :
  ice_cream_distribution 62 = 2016 :=
by sorry

/-- Given:
    - 62 trainees choose from 5 ice cream flavors
    - Bubblegum flavor (r) at least as popular as tabasco (t)
    - Number of students choosing cactus flavor (a) is a multiple of 6
    - At most 5 students chose lemon basil flavor (b)
    - At most 1 student chose foie gras flavor (c)
    Prove: The number of possible distributions is 2016 -/
theorem ice_cream_distribution_count :
  ∃ (r t a b c : ℕ),
    r + t + a + b + c = 62 ∧
    r ≥ t ∧
    a % 6 = 0 ∧
    b ≤ 5 ∧
    c ≤ 1 ∧
    ice_cream_distribution 62 = 2016 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_theorem_ice_cream_distribution_count_l3751_375171


namespace NUMINAMATH_CALUDE_sticks_per_pack_l3751_375187

/-- Represents the number of packs in a carton -/
def packs_per_carton : ℕ := 5

/-- Represents the number of cartons in a brown box -/
def cartons_per_box : ℕ := 4

/-- Represents the total number of sticks in all brown boxes -/
def total_sticks : ℕ := 480

/-- Represents the total number of brown boxes -/
def total_boxes : ℕ := 8

/-- Theorem stating that the number of sticks in each pack is 3 -/
theorem sticks_per_pack : 
  total_sticks / (total_boxes * cartons_per_box * packs_per_carton) = 3 := by
  sorry


end NUMINAMATH_CALUDE_sticks_per_pack_l3751_375187


namespace NUMINAMATH_CALUDE_solution_set_exponential_inequality_l3751_375152

theorem solution_set_exponential_inequality :
  ∀ x : ℝ, (6 : ℝ) ^ (x - 2) < 1 ↔ x < 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_exponential_inequality_l3751_375152


namespace NUMINAMATH_CALUDE_shortest_path_length_on_cube_l3751_375119

/-- Represents a cube with a given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Represents a path on the surface of a cube -/
structure SurfacePath (c : Cube) where
  length : ℝ
  isOnSurface : Bool

/-- The shortest path on the surface of a cube from the center of one face to the center of the opposite face -/
def shortestPath (c : Cube) : SurfacePath c :=
  sorry

/-- Theorem stating that the shortest path on a cube with edge length 2 has length 3 -/
theorem shortest_path_length_on_cube :
  let c : Cube := { edgeLength := 2 }
  (shortestPath c).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_length_on_cube_l3751_375119


namespace NUMINAMATH_CALUDE_douyin_sales_and_profit_l3751_375142

/-- Represents an e-commerce platform selling a small commodity. -/
structure ECommercePlatform where
  cost_price : ℕ
  initial_price : ℕ
  initial_volume : ℕ
  price_decrease : ℕ
  volume_increase : ℕ

/-- Calculates the daily sales volume for a given selling price. -/
def daily_sales_volume (platform : ECommercePlatform) (selling_price : ℕ) : ℕ :=
  platform.initial_volume + 
    (platform.initial_price - selling_price) / platform.price_decrease * platform.volume_increase

/-- Calculates the daily profit for a given selling price. -/
def daily_profit (platform : ECommercePlatform) (selling_price : ℕ) : ℕ :=
  (selling_price - platform.cost_price) * daily_sales_volume platform selling_price

/-- The e-commerce platform with given conditions. -/
def douyin_platform : ECommercePlatform := {
  cost_price := 40
  initial_price := 60
  initial_volume := 20
  price_decrease := 5
  volume_increase := 10
}

theorem douyin_sales_and_profit :
  (daily_sales_volume douyin_platform 50 = 40) ∧
  (∃ (price : ℕ), daily_profit douyin_platform price = 448 ∧
    ∀ (p : ℕ), daily_profit douyin_platform p = 448 → p ≥ price) :=
by sorry

end NUMINAMATH_CALUDE_douyin_sales_and_profit_l3751_375142


namespace NUMINAMATH_CALUDE_corey_candy_count_l3751_375150

/-- Given that Tapanga and Corey have a total of 66 candies, and Tapanga has 8 more candies than Corey,
    prove that Corey has 29 candies. -/
theorem corey_candy_count (total : ℕ) (difference : ℕ) (corey : ℕ) : 
  total = 66 → difference = 8 → total = corey + (corey + difference) → corey = 29 := by
  sorry

end NUMINAMATH_CALUDE_corey_candy_count_l3751_375150


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l3751_375159

/-- A regular polygon with interior angles of 150 degrees has 12 sides. -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ, 
    n > 2 →
    (∀ angle : ℝ, angle = 150) →
    (180 * (n - 2) : ℝ) = (n * 150 : ℝ) →
    n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l3751_375159


namespace NUMINAMATH_CALUDE_parallel_vectors_x_values_l3751_375122

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_values :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (x - 1, 2)
  are_parallel a b → x = -1 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_values_l3751_375122


namespace NUMINAMATH_CALUDE_cake_piece_volume_and_icing_area_sum_l3751_375110

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Calculates the volume of a triangular prism -/
def triangularPrismVolume (base : ℝ) (height : ℝ) : ℝ := sorry

/-- Calculates the area of a rectangle -/
def rectangleArea (width : ℝ) (height : ℝ) : ℝ := sorry

/-- Main theorem: The sum of the volume and icing area of the cake piece is 19.8 -/
theorem cake_piece_volume_and_icing_area_sum :
  let a : ℝ := 3  -- edge length of the cube
  let p : Point3D := ⟨0, 0, 0⟩
  let q : Point3D := ⟨a, 0, 0⟩
  let r : Point3D := ⟨0, a, 0⟩
  let m : Point3D := ⟨a/3, a, 0⟩
  let triangleQMR_area : ℝ := triangleArea q m r
  let volume : ℝ := triangularPrismVolume triangleQMR_area a
  let icingArea : ℝ := triangleQMR_area + rectangleArea a a
  volume + icingArea = 19.8 := by sorry

end NUMINAMATH_CALUDE_cake_piece_volume_and_icing_area_sum_l3751_375110


namespace NUMINAMATH_CALUDE_circle_and_reflection_theorem_l3751_375165

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 ∧ 2*a - b - 4 = 0

-- Define the points A and B
def point_A : ℝ × ℝ := (3, 0)
def point_B : ℝ × ℝ := (5, 2)

-- Define the reflection line
def reflection_line (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the point M
def point_M : ℝ × ℝ := (-4, -3)

-- Define the theorem
theorem circle_and_reflection_theorem :
  (circle_C point_A.1 point_A.2) ∧
  (circle_C point_B.1 point_B.2) ∧
  (∃ (x y : ℝ), reflection_line x y ∧ 
    ∃ (t : ℝ), (1 - t) * point_M.1 + t * x = -4 ∧ (1 - t) * point_M.2 + t * y = -3 ∧
    circle_C x y) →
  (∀ (x y : ℝ), circle_C x y ↔ (x - 3)^2 + (y - 2)^2 = 4) ∧
  (∃ (k : ℝ), ∀ (x y : ℝ), (x = 1 ∨ 12*x - 5*y - 52 = 0) ↔ 
    (∃ (t : ℝ), x = (1 - t) * 1 + t * point_M.1 ∧ y = (1 - t) * (-8) + t * point_M.2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_reflection_theorem_l3751_375165


namespace NUMINAMATH_CALUDE_age_difference_l3751_375139

/-- The difference in ages between (x + y) and (y + z) is 12 years, given that z is 12 years younger than x -/
theorem age_difference (x y z : ℕ) (h : z = x - 12) :
  (x + y) - (y + z) = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3751_375139


namespace NUMINAMATH_CALUDE_wall_width_is_0_05_meters_l3751_375188

-- Define the brick dimensions in meters
def brick_length : Real := 0.21
def brick_width : Real := 0.10
def brick_height : Real := 0.08

-- Define the wall dimensions
def wall_length : Real := 9
def wall_height : Real := 18.5

-- Define the number of bricks
def num_bricks : Real := 4955.357142857142

-- Theorem to prove
theorem wall_width_is_0_05_meters :
  let brick_volume := brick_length * brick_width * brick_height
  let total_brick_volume := brick_volume * num_bricks
  let wall_width := total_brick_volume / (wall_length * wall_height)
  wall_width = 0.05 := by sorry

end NUMINAMATH_CALUDE_wall_width_is_0_05_meters_l3751_375188


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3751_375170

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℚ, (3 * x₁^2 - 5 * x₁ - 2 = 0 ∧ x₁ = 2) ∧
                (3 * x₂^2 - 5 * x₂ - 2 = 0 ∧ x₂ = -1/3)) ∧
  (∃ y₁ y₂ : ℚ, (3 * y₁ * (y₁ - 1) = 2 - 2 * y₁ ∧ y₁ = 1) ∧
                (3 * y₂ * (y₂ - 1) = 2 - 2 * y₂ ∧ y₂ = -2/3)) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3751_375170


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3751_375140

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 8*x + 16

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square inscribed in the region bound by the parabola and x-axis -/
structure InscribedSquare where
  E : Point
  F : Point
  G : Point
  H : Point
  -- E and F are on x-axis
  h1 : E.y = 0
  h2 : F.y = 0
  -- G is on the parabola
  h3 : G.y = parabola G.x
  -- EFGH forms a square
  h4 : (F.x - E.x)^2 + (G.y - F.y)^2 = (G.x - F.x)^2 + (G.y - F.y)^2

/-- The theorem stating that the area of the inscribed square is 16 -/
theorem inscribed_square_area (s : InscribedSquare) : (s.F.x - s.E.x)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3751_375140


namespace NUMINAMATH_CALUDE_caitlin_age_l3751_375169

/-- Prove that Caitlin's age is 29 years -/
theorem caitlin_age :
  let aunt_anna_age : ℕ := 54
  let brianna_age : ℕ := (2 * aunt_anna_age) / 3
  let caitlin_age : ℕ := brianna_age - 7
  caitlin_age = 29 := by
  sorry

end NUMINAMATH_CALUDE_caitlin_age_l3751_375169


namespace NUMINAMATH_CALUDE_cistern_water_depth_l3751_375111

/-- Proves that for a cistern with given dimensions and wet surface area, the water depth is 1.25 m -/
theorem cistern_water_depth
  (length : ℝ)
  (width : ℝ)
  (total_wet_area : ℝ)
  (h_length : length = 12)
  (h_width : width = 14)
  (h_total_wet_area : total_wet_area = 233) :
  let bottom_area := length * width
  let perimeter := 2 * (length + width)
  let water_depth := (total_wet_area - bottom_area) / perimeter
  water_depth = 1.25 := by
sorry

end NUMINAMATH_CALUDE_cistern_water_depth_l3751_375111


namespace NUMINAMATH_CALUDE_natural_number_representation_l3751_375156

theorem natural_number_representation (k : ℕ) : 
  ∃ n : ℕ, k = 3*n ∨ k = 3*n + 1 ∨ k = 3*n + 2 :=
sorry

end NUMINAMATH_CALUDE_natural_number_representation_l3751_375156


namespace NUMINAMATH_CALUDE_cube_surface_area_l3751_375157

/-- The surface area of a cube with edge length 11 centimeters is 726 square centimeters. -/
theorem cube_surface_area : 
  let edge_length : ℝ := 11
  let surface_area : ℝ := 6 * edge_length^2
  surface_area = 726 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3751_375157


namespace NUMINAMATH_CALUDE_field_trip_groups_l3751_375128

/-- Given the conditions for a field trip lunch preparation, prove the number of groups. -/
theorem field_trip_groups (
  sandwiches_per_student : ℕ)
  (bread_per_sandwich : ℕ)
  (students_per_group : ℕ)
  (total_bread : ℕ)
  (h1 : sandwiches_per_student = 2)
  (h2 : bread_per_sandwich = 2)
  (h3 : students_per_group = 6)
  (h4 : total_bread = 120) :
  total_bread / (bread_per_sandwich * sandwiches_per_student * students_per_group) = 5 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_groups_l3751_375128


namespace NUMINAMATH_CALUDE_f_negative_one_equals_negative_twelve_l3751_375144

def f (x : ℝ) : ℝ := sorry

theorem f_negative_one_equals_negative_twelve
  (h_odd : ∀ x, f x = -f (-x))
  (h_nonneg : ∀ x ≥ 0, ∃ a : ℝ, f x = a^(x+1) - 4) :
  f (-1) = -12 := by sorry

end NUMINAMATH_CALUDE_f_negative_one_equals_negative_twelve_l3751_375144


namespace NUMINAMATH_CALUDE_intersection_counts_theorem_l3751_375158

/-- Represents a line in 2D space -/
structure Line :=
  (a b c : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Represents the number of intersection points -/
inductive IntersectionCount : Type
  | Zero
  | One
  | Two
  | Three
  | Four

/-- Given two intersecting lines and a circle, this function returns the possible numbers of intersection points -/
def possibleIntersectionCounts (l1 l2 : Line) (c : Circle) : Set IntersectionCount :=
  sorry

/-- Theorem stating that the possible numbers of intersection points are 0, 1, 2, 3, and 4 -/
theorem intersection_counts_theorem (l1 l2 : Line) (c : Circle) :
  possibleIntersectionCounts l1 l2 c = {IntersectionCount.Zero, IntersectionCount.One, IntersectionCount.Two, IntersectionCount.Three, IntersectionCount.Four} :=
by sorry

end NUMINAMATH_CALUDE_intersection_counts_theorem_l3751_375158


namespace NUMINAMATH_CALUDE_triangle_properties_l3751_375193

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a + t.c = t.b * (Real.cos t.C + Real.sqrt 3 * Real.sin t.C))
  (h2 : t.b = 2) : 
  t.B = π / 3 ∧ 
  ∀ (s : Triangle), s.b = 2 → 
    Real.sqrt 3 / 4 * s.a * s.c * Real.sin s.B ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3751_375193


namespace NUMINAMATH_CALUDE_optimal_meeting_time_l3751_375149

/-- The optimal meeting time for a pedestrian and cyclist on a circular path -/
theorem optimal_meeting_time 
  (pedestrian_speed : ℝ) 
  (cyclist_speed : ℝ) 
  (path_length : ℝ) 
  (walked_distance : ℝ) 
  (remaining_distance : ℝ) 
  (h1 : pedestrian_speed = 6.5)
  (h2 : cyclist_speed = 20)
  (h3 : path_length = 4 * Real.pi)
  (h4 : walked_distance = 6.5)
  (h5 : remaining_distance = 4 * Real.pi - 6.5)
  (h6 : walked_distance = pedestrian_speed * 1) -- 1 hour of walking
  : ∃ (t : ℝ), t = (155 - 28 * Real.pi) / 172 ∧ 
    t = min (remaining_distance / (pedestrian_speed + cyclist_speed))
            ((path_length - walked_distance) / (pedestrian_speed + cyclist_speed)) := by
  sorry

end NUMINAMATH_CALUDE_optimal_meeting_time_l3751_375149


namespace NUMINAMATH_CALUDE_no_real_solutions_l3751_375112

theorem no_real_solutions :
  ¬∃ (x : ℝ), (6 * x) / (x^2 + 2 * x + 5) + (7 * x) / (x^2 - 7 * x + 5) = -3/2 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3751_375112


namespace NUMINAMATH_CALUDE_rabbit_cleaner_amount_l3751_375185

/-- The amount of cleaner used for a dog stain in ounces -/
def dog_cleaner : ℝ := 6

/-- The amount of cleaner used for a cat stain in ounces -/
def cat_cleaner : ℝ := 4

/-- The total amount of cleaner used in ounces -/
def total_cleaner : ℝ := 49

/-- The number of dogs -/
def num_dogs : ℕ := 6

/-- The number of cats -/
def num_cats : ℕ := 3

/-- The number of rabbits -/
def num_rabbits : ℕ := 1

/-- The amount of cleaner used for a rabbit stain in ounces -/
def rabbit_cleaner : ℝ := 1

theorem rabbit_cleaner_amount :
  dog_cleaner * num_dogs + cat_cleaner * num_cats + rabbit_cleaner * num_rabbits = total_cleaner := by
  sorry

end NUMINAMATH_CALUDE_rabbit_cleaner_amount_l3751_375185


namespace NUMINAMATH_CALUDE_remainder_problem_l3751_375133

theorem remainder_problem (g : ℕ) (h1 : g = 101) (h2 : 4351 % g = 8) :
  5161 % g = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3751_375133


namespace NUMINAMATH_CALUDE_males_count_l3751_375143

/-- Represents the population of a village -/
structure Village where
  total_population : ℕ
  num_groups : ℕ
  males_in_one_group : Bool

/-- Theorem: In a village with 520 people divided into 4 equal groups,
    if one group represents all males, then the number of males is 130 -/
theorem males_count (v : Village)
  (h1 : v.total_population = 520)
  (h2 : v.num_groups = 4)
  (h3 : v.males_in_one_group = true) :
  v.total_population / v.num_groups = 130 := by
  sorry

#check males_count

end NUMINAMATH_CALUDE_males_count_l3751_375143


namespace NUMINAMATH_CALUDE_candy_packing_problem_l3751_375127

theorem candy_packing_problem :
  ∃! n : ℕ, 11 ≤ n ∧ n ≤ 100 ∧ 
    6 ∣ n ∧ 9 ∣ n ∧ n % 7 = 1 ∧
    n = 36 := by sorry

end NUMINAMATH_CALUDE_candy_packing_problem_l3751_375127


namespace NUMINAMATH_CALUDE_expand_and_simplify_expression_l3751_375194

theorem expand_and_simplify_expression (x : ℝ) :
  2*x*(3*x^2 - 4*x + 5) - (x^2 - 3*x)*(4*x + 5) = 2*x^3 - x^2 + 25*x := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_expression_l3751_375194


namespace NUMINAMATH_CALUDE_solutions_of_quadratic_l3751_375183

theorem solutions_of_quadratic (x : ℝ) : x^2 = 16*x ↔ x = 0 ∨ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_solutions_of_quadratic_l3751_375183


namespace NUMINAMATH_CALUDE_roller_plate_acceleration_l3751_375160

noncomputable def plate_acceleration (R r : ℝ) (m : ℝ) (α : ℝ) (g : ℝ) : ℝ :=
  g * Real.sqrt ((1 - Real.cos α) / 2)

noncomputable def plate_direction (α : ℝ) : ℝ :=
  Real.arcsin (Real.sqrt ((1 - Real.cos α) / 2))

theorem roller_plate_acceleration 
  (R : ℝ) 
  (r : ℝ) 
  (m : ℝ) 
  (α : ℝ) 
  (g : ℝ) 
  (h_R : R = 1) 
  (h_r : r = 0.4) 
  (h_m : m = 150) 
  (h_α : α = Real.arccos 0.68) 
  (h_g : g = 10) :
  plate_acceleration R r m α g = 4 ∧ 
  plate_direction α = Real.arcsin 0.4 ∧
  plate_acceleration R r m α g = g * Real.sin (α / 2) :=
by
  sorry

#check roller_plate_acceleration

end NUMINAMATH_CALUDE_roller_plate_acceleration_l3751_375160


namespace NUMINAMATH_CALUDE_ellipse_focus_distance_l3751_375101

/-- An ellipse in the first quadrant tangent to both axes with foci at (4,8) and (d,8) -/
structure Ellipse where
  d : ℝ
  tangent_to_axes : Bool
  in_first_quadrant : Bool
  focus1 : ℝ × ℝ := (4, 8)
  focus2 : ℝ × ℝ := (d, 8)

/-- The theorem stating that d = 30 for the given ellipse -/
theorem ellipse_focus_distance (e : Ellipse) (h1 : e.tangent_to_axes) (h2 : e.in_first_quadrant) :
  e.d = 30 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_l3751_375101


namespace NUMINAMATH_CALUDE_convex_polygon_diagonal_triangles_l3751_375180

/-- Represents a convex polygon with diagonals drawn to create triangles -/
structure ConvexPolygonWithDiagonals where
  sides : ℕ
  triangles : ℕ
  diagonalTriangles : ℕ

/-- The property that needs to be proven -/
def impossibleHalfDiagonalTriangles (p : ConvexPolygonWithDiagonals) : Prop :=
  p.sides = 2016 ∧ p.triangles = 2014 → p.diagonalTriangles ≠ 1007

theorem convex_polygon_diagonal_triangles :
  ∀ p : ConvexPolygonWithDiagonals, impossibleHalfDiagonalTriangles p :=
sorry

end NUMINAMATH_CALUDE_convex_polygon_diagonal_triangles_l3751_375180


namespace NUMINAMATH_CALUDE_parallel_line_length_l3751_375186

/-- Given a triangle with base 36, prove that a line parallel to the base
    that divides the area into two equal parts has a length of 18√2. -/
theorem parallel_line_length (base : ℝ) (h_base : base = 36) :
  ∃ (line_length : ℝ), line_length = 18 * Real.sqrt 2 ∧
  ∀ (triangle_area smaller_area : ℝ),
    smaller_area = triangle_area / 2 →
    line_length ^ 2 / base ^ 2 = smaller_area / triangle_area :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_length_l3751_375186


namespace NUMINAMATH_CALUDE_fill_tank_theorem_l3751_375132

/-- Calculates the remaining water needed to fill a tank -/
def remaining_water (tank_capacity : ℕ) (pour_rate : ℚ) (pour_time : ℕ) : ℕ :=
  tank_capacity - (pour_time / 15 : ℕ)

/-- Theorem: Given a 150-gallon tank, pouring water at 1 gallon per 15 seconds for 525 seconds,
    the remaining water needed to fill the tank is 115 gallons -/
theorem fill_tank_theorem :
  remaining_water 150 (1/15 : ℚ) 525 = 115 := by
  sorry

end NUMINAMATH_CALUDE_fill_tank_theorem_l3751_375132


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l3751_375135

def U : Set Int := Set.univ
def M : Set Int := {1, 2}
def P : Set Int := {-2, -1, 0, 1, 2}

theorem intersection_complement_equals_set : P ∩ (U \ M) = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l3751_375135


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_of_roots_l3751_375126

/-- A quadratic function f(x) = 2x² + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := 2 * x^2 + b * x + c

theorem min_reciprocal_sum_of_roots (b c : ℝ) :
  (f b c (-10) = f b c 12) →
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ f b c x₁ = 0 ∧ f b c x₂ = 0) →
  (∃ m : ℝ, m = 2 ∧ ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f b c x₁ = 0 → f b c x₂ = 0 → 1/x₁ + 1/x₂ ≥ m) :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_of_roots_l3751_375126


namespace NUMINAMATH_CALUDE_ratio_equality_l3751_375123

theorem ratio_equality (x y z : ℝ) (h : x / 3 = y / 5 ∧ y / 5 = z / 7) :
  (x - y + z) / (x + y - z) = 5 := by sorry

end NUMINAMATH_CALUDE_ratio_equality_l3751_375123


namespace NUMINAMATH_CALUDE_spider_total_distance_l3751_375107

def spider_crawl (start end_1 end_2 end_3 : Int) : Int :=
  |end_1 - start| + |end_2 - end_1| + |end_3 - end_2|

theorem spider_total_distance :
  spider_crawl (-3) (-8) 0 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_spider_total_distance_l3751_375107


namespace NUMINAMATH_CALUDE_ellipse_product_l3751_375102

-- Define the ellipse C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 5 + p.2^2 = 1}

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (2, 0)
def F₂ : ℝ × ℝ := (-2, 0)

-- State the theorem
theorem ellipse_product (P : ℝ × ℝ) 
  (h_on_ellipse : P ∈ C) 
  (h_perpendicular : (P.1 - F₁.1, P.2 - F₁.2) • (P.1 - F₂.1, P.2 - F₂.2) = 0) :
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * 
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_product_l3751_375102


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3751_375117

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (2 - I) / (2 + I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l3751_375117


namespace NUMINAMATH_CALUDE_farmer_apples_l3751_375197

/-- The number of apples the farmer has after giving some away -/
def remaining_apples (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that the farmer has 39 apples after giving some away -/
theorem farmer_apples : remaining_apples 127 88 = 39 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l3751_375197


namespace NUMINAMATH_CALUDE_product_of_sines_l3751_375174

theorem product_of_sines (π : Real) : 
  (1 + Real.sin (π / 12)) * (1 + Real.sin (5 * π / 12)) * 
  (1 + Real.sin (7 * π / 12)) * (1 + Real.sin (11 * π / 12)) = 
  (17 / 16 + 2 * Real.sin (π / 12)) * (17 / 16 + 2 * Real.sin (5 * π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_sines_l3751_375174


namespace NUMINAMATH_CALUDE_certain_number_proof_l3751_375154

def w : ℕ := 132

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem certain_number_proof :
  ∃ (n : ℕ), 
    (is_factor (2^5) (n * w)) ∧ 
    (is_factor (3^3) (n * w)) ∧ 
    (is_factor (11^2) (n * w)) ∧
    (∀ (m : ℕ), m < w → ¬(is_factor (2^5) (n * m) ∧ is_factor (3^3) (n * m) ∧ is_factor (11^2) (n * m))) →
  n = 792 :=
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3751_375154


namespace NUMINAMATH_CALUDE_acid_percentage_proof_l3751_375198

/-- Given a solution with 1.4 litres of pure acid in 4 litres total volume,
    prove that the percentage of pure acid is 35%. -/
theorem acid_percentage_proof : 
  let pure_acid_volume : ℝ := 1.4
  let total_solution_volume : ℝ := 4
  let percentage_pure_acid : ℝ := (pure_acid_volume / total_solution_volume) * 100
  percentage_pure_acid = 35 := by
  sorry

end NUMINAMATH_CALUDE_acid_percentage_proof_l3751_375198
