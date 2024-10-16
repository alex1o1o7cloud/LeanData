import Mathlib

namespace NUMINAMATH_CALUDE_symmetric_sine_cosine_l919_91910

/-- A function f is symmetric about a line x = c if f(c + h) = f(c - h) for all h -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ h, f (c + h) = f (c - h)

/-- The main theorem -/
theorem symmetric_sine_cosine (a : ℝ) :
  SymmetricAbout (fun x ↦ Real.sin (2 * x) + a * Real.cos (2 * x)) (π / 8) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_sine_cosine_l919_91910


namespace NUMINAMATH_CALUDE_ladder_length_l919_91931

theorem ladder_length : ∃ L : ℝ, 
  L > 0 ∧ 
  (4/5 * L)^2 + 4^2 = L^2 ∧ 
  L = 20/3 :=
by sorry

end NUMINAMATH_CALUDE_ladder_length_l919_91931


namespace NUMINAMATH_CALUDE_rectangle_circle_intersection_area_l919_91914

/-- The area of intersection between a rectangle and a circle with shared center -/
theorem rectangle_circle_intersection_area :
  ∀ (rectangle_length rectangle_width circle_radius : ℝ),
  rectangle_length = 10 →
  rectangle_width = 2 * Real.sqrt 3 →
  circle_radius = 3 →
  ∃ (intersection_area : ℝ),
  intersection_area = (9 * Real.pi) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_circle_intersection_area_l919_91914


namespace NUMINAMATH_CALUDE_balloon_problem_solution_l919_91987

def balloon_problem (initial_balloons : ℕ) (given_to_girl : ℕ) (floated_away : ℕ) (given_away_later : ℕ) (final_balloons : ℕ) : ℕ :=
  final_balloons - (initial_balloons - given_to_girl - floated_away - given_away_later)

theorem balloon_problem_solution :
  balloon_problem 50 1 12 9 39 = 11 := by
  sorry

end NUMINAMATH_CALUDE_balloon_problem_solution_l919_91987


namespace NUMINAMATH_CALUDE_billie_baking_days_l919_91996

/-- The number of days Billie bakes pumpkin pies -/
def days_baking : ℕ := 11

/-- The number of pies Billie bakes per day -/
def pies_per_day : ℕ := 3

/-- The number of cans of whipped cream needed to cover one pie -/
def cans_per_pie : ℕ := 2

/-- The number of pies eaten -/
def pies_eaten : ℕ := 4

/-- The number of cans of whipped cream needed for the remaining pies -/
def cans_needed : ℕ := 58

theorem billie_baking_days :
  days_baking * pies_per_day - pies_eaten = cans_needed / cans_per_pie := by sorry

end NUMINAMATH_CALUDE_billie_baking_days_l919_91996


namespace NUMINAMATH_CALUDE_triangle_arithmetic_sides_cosine_identity_l919_91907

/-- In a triangle ABC where the sides form an arithmetic sequence, 
    5 cos A - 4 cos A cos C + 5 cos C equals 8 -/
theorem triangle_arithmetic_sides_cosine_identity 
  (A B C : ℝ) (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  A + B + C = π →
  2 * b = a + c →
  5 * Real.cos A - 4 * Real.cos A * Real.cos C + 5 * Real.cos C = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_arithmetic_sides_cosine_identity_l919_91907


namespace NUMINAMATH_CALUDE_rose_group_size_l919_91951

theorem rose_group_size (group_size : ℕ) : 
  (9 > 0) →
  (group_size > 0) →
  (Nat.lcm 9 group_size = 171) →
  (171 % group_size = 0) →
  (9 % group_size ≠ 0) →
  group_size = 19 := by
  sorry

end NUMINAMATH_CALUDE_rose_group_size_l919_91951


namespace NUMINAMATH_CALUDE_total_tickets_is_150_l919_91908

/-- The number of tickets Alan handed out -/
def alan_tickets : ℕ := 26

/-- The number of tickets Marcy handed out -/
def marcy_tickets : ℕ := 5 * alan_tickets - 6

/-- The total number of tickets handed out by Alan and Marcy -/
def total_tickets : ℕ := alan_tickets + marcy_tickets

/-- Theorem stating that the total number of tickets handed out is 150 -/
theorem total_tickets_is_150 : total_tickets = 150 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_is_150_l919_91908


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l919_91946

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Theorem stating that the complex fraction simplifies to the given result -/
theorem complex_fraction_simplification :
  (3 * (1 + i)^2) / (i - 1) = 3 - 3*i :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l919_91946


namespace NUMINAMATH_CALUDE_salary_comparison_l919_91930

theorem salary_comparison (a b : ℝ) (h : a = 0.8 * b) : b = 1.25 * a := by
  sorry

end NUMINAMATH_CALUDE_salary_comparison_l919_91930


namespace NUMINAMATH_CALUDE_b_is_positive_l919_91962

theorem b_is_positive (a b : ℝ) (h : ∀ x : ℝ, (x - a)^2 + b > 0) : b > 0 := by
  sorry

end NUMINAMATH_CALUDE_b_is_positive_l919_91962


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l919_91906

theorem product_of_three_numbers (a b c : ℚ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 6 * (b + c))
  (second_eq : b = 5 * c) : 
  a * b * c = 22500 / 343 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l919_91906


namespace NUMINAMATH_CALUDE_problem_solution_l919_91918

theorem problem_solution (n x y : ℝ) 
  (h1 : x = 4 * n)
  (h2 : y = x / 2)
  (h3 : 2 * n + 3 = 0.20 * 25)
  (h4 : y^3 - 4 = (1/3) * x) :
  y = (16/3)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l919_91918


namespace NUMINAMATH_CALUDE_road_distance_ratio_l919_91977

/-- Represents the distance between two cities --/
structure CityDistance where
  total : ℕ
  deriving Repr

/-- Represents a pole with distances to two cities --/
structure Pole where
  distanceA : ℕ
  distanceB : ℕ
  deriving Repr

/-- The configuration of poles between two cities --/
structure RoadConfiguration where
  distance : CityDistance
  pole1 : Pole
  pole2 : Pole
  pole3 : Pole
  deriving Repr

/-- The theorem to be proved --/
theorem road_distance_ratio 
  (config : RoadConfiguration) 
  (h1 : config.pole1.distanceB = 3 * config.pole1.distanceA)
  (h2 : config.pole2.distanceB = 3 * config.pole2.distanceA)
  (h3 : config.pole1.distanceA + config.pole1.distanceB = config.distance.total)
  (h4 : config.pole2.distanceA + config.pole2.distanceB = config.distance.total)
  (h5 : config.pole2.distanceA = config.pole1.distanceA + 40)
  (h6 : config.pole3.distanceA = config.pole2.distanceA + 10)
  (h7 : config.pole3.distanceB = config.pole2.distanceB - 10) :
  (max config.pole3.distanceA config.pole3.distanceB) / 
  (min config.pole3.distanceA config.pole3.distanceB) = 7 := by
  sorry

end NUMINAMATH_CALUDE_road_distance_ratio_l919_91977


namespace NUMINAMATH_CALUDE_abc_inequality_l919_91927

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c = 1/a + 1/b + 1/c) : a + b + c ≥ 3/(a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l919_91927


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l919_91935

theorem completing_square_quadratic (x : ℝ) : 
  x^2 + 8*x - 3 = 0 ↔ (x + 4)^2 = 19 :=
sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l919_91935


namespace NUMINAMATH_CALUDE_certain_number_problem_l919_91937

theorem certain_number_problem (x : ℝ) : 
  (10 + 20 + 60) / 3 = ((x + 40 + 25) / 3 + 5) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l919_91937


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l919_91942

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of the number we want to convert -/
def binary_number : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that the given binary number is equal to 51 in decimal -/
theorem binary_110011_equals_51 :
  binary_to_decimal binary_number = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l919_91942


namespace NUMINAMATH_CALUDE_probability_point_in_sphere_l919_91915

/-- The probability that a randomly selected point (x, y, z) in a cube with side length 2
    centered at the origin lies within a unit sphere centered at the origin. -/
theorem probability_point_in_sphere : 
  let cube_volume : ℝ := 8
  let sphere_volume : ℝ := (4 / 3) * Real.pi
  let prob : ℝ := sphere_volume / cube_volume
  prob = Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_probability_point_in_sphere_l919_91915


namespace NUMINAMATH_CALUDE_water_level_correct_water_level_rate_initial_water_level_l919_91923

/-- Represents the water level function in a reservoir -/
def water_level (x : ℝ) : ℝ := 6 + 0.3 * x

theorem water_level_correct (x : ℝ) (h : x ≥ 0) :
  water_level x = 6 + 0.3 * x :=
by sorry

/-- The water level rises at a constant rate of 0.3 meters per hour -/
theorem water_level_rate (x y : ℝ) (hx : x ≥ 0) (hy : y > x) :
  (water_level y - water_level x) / (y - x) = 0.3 :=
by sorry

/-- The initial water level is 6 meters -/
theorem initial_water_level : water_level 0 = 6 :=
by sorry

end NUMINAMATH_CALUDE_water_level_correct_water_level_rate_initial_water_level_l919_91923


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l919_91916

/-- Given a line intersecting y = x^2 at (x₁, x₁²) and (x₂, x₂²), and the x-axis at (x₃, 0),
    prove that 1/x₁ + 1/x₂ = 1/x₃ -/
theorem line_parabola_intersection (x₁ x₂ x₃ : ℝ) (h₁ : x₁ ≠ 0) (h₂ : x₂ ≠ 0) (h₃ : x₃ ≠ 0)
  (h_line : ∃ (k m : ℝ), x₁^2 = k * x₁ + m ∧ x₂^2 = k * x₂ + m ∧ 0 = k * x₃ + m) :
  1 / x₁ + 1 / x₂ = 1 / x₃ := by
  sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l919_91916


namespace NUMINAMATH_CALUDE_warehouse_inventory_equality_l919_91994

theorem warehouse_inventory_equality (x : ℝ) : 
  x > 0 → -- Ensure positive initial inventory
  (x * (8/15) - x * (8/15) * (1/10)) = (1200 * (2/3) + x * (8/15) * (1/10)) →
  x = 1875 := by
sorry

end NUMINAMATH_CALUDE_warehouse_inventory_equality_l919_91994


namespace NUMINAMATH_CALUDE_bottles_left_l919_91948

theorem bottles_left (initial : Float) (maria_drank : Float) (sister_drank : Float) :
  initial = 45.0 →
  maria_drank = 14.0 →
  sister_drank = 8.0 →
  initial - maria_drank - sister_drank = 23.0 :=
by sorry

end NUMINAMATH_CALUDE_bottles_left_l919_91948


namespace NUMINAMATH_CALUDE_peters_nickels_problem_l919_91959

theorem peters_nickels_problem :
  ∃! n : ℕ, 40 < n ∧ n < 400 ∧ 
    n % 4 = 2 ∧ n % 5 = 2 ∧ n % 7 = 2 ∧ n = 142 := by
  sorry

end NUMINAMATH_CALUDE_peters_nickels_problem_l919_91959


namespace NUMINAMATH_CALUDE_product_equality_implies_sum_l919_91925

theorem product_equality_implies_sum (m n : ℝ) : 
  (m^2 + 4*m + 5) * (n^2 - 2*n + 6) = 5 → 2*m + 3*n = -1 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_implies_sum_l919_91925


namespace NUMINAMATH_CALUDE_a_plus_b_equals_10_l919_91988

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem a_plus_b_equals_10 (a b : ℝ) 
  (ha : a + log10 a = 10) 
  (hb : b + 10^b = 10) : 
  a + b = 10 := by sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_10_l919_91988


namespace NUMINAMATH_CALUDE_sophomore_mean_is_94_l919_91967

/-- Represents the number of students and their scores in a math competition -/
structure MathCompetition where
  total_students : ℕ
  overall_mean : ℝ
  sophomores : ℕ
  juniors : ℕ
  sophomore_mean : ℝ
  junior_mean : ℝ

/-- The math competition satisfies the given conditions -/
def satisfies_conditions (mc : MathCompetition) : Prop :=
  mc.total_students = 150 ∧
  mc.overall_mean = 85 ∧
  mc.juniors = mc.sophomores - (mc.sophomores / 5) ∧
  mc.sophomore_mean = mc.junior_mean * 1.25

/-- Theorem stating that under the given conditions, the sophomore mean score is 94 -/
theorem sophomore_mean_is_94 (mc : MathCompetition) 
  (h : satisfies_conditions mc) : mc.sophomore_mean = 94 := by
  sorry

#check sophomore_mean_is_94

end NUMINAMATH_CALUDE_sophomore_mean_is_94_l919_91967


namespace NUMINAMATH_CALUDE_similar_walls_length_l919_91966

/-- Represents the work done to build a wall -/
structure WallWork where
  persons : ℕ
  days : ℕ
  length : ℝ

/-- The theorem stating the relationship between two similar walls -/
theorem similar_walls_length
  (wall1 : WallWork)
  (wall2 : WallWork)
  (h1 : wall1.persons = 18)
  (h2 : wall1.days = 42)
  (h3 : wall2.persons = 30)
  (h4 : wall2.days = 18)
  (h5 : wall2.length = 100)
  (h6 : (wall1.persons * wall1.days) / (wall2.persons * wall2.days) = wall1.length / wall2.length) :
  wall1.length = 140 := by
  sorry

#check similar_walls_length

end NUMINAMATH_CALUDE_similar_walls_length_l919_91966


namespace NUMINAMATH_CALUDE_profit_share_ratio_l919_91965

theorem profit_share_ratio (total_profit : ℕ) (difference : ℕ) : 
  total_profit = 700 → difference = 140 → 
  ∃ (x y : ℕ), x + y = total_profit ∧ x - y = difference ∧ 
  (y : ℚ) / total_profit = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l919_91965


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_not_necessary_condition_l919_91976

theorem sufficient_condition_for_inequality (x : ℝ) :
  (-1 < x ∧ x < 5) → (6 / (x + 1) ≥ 1) :=
by
  sorry

theorem not_necessary_condition (x : ℝ) :
  (6 / (x + 1) ≥ 1) → ¬(-1 < x ∧ x < 5) :=
by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_not_necessary_condition_l919_91976


namespace NUMINAMATH_CALUDE_greatest_n_perfect_square_l919_91981

/-- Sum of squares from 1 to n -/
def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Product of sum of squares -/
def product_sum_squares (n : ℕ) : ℕ :=
  (sum_squares n) * (sum_squares (2 * n) - sum_squares n)

/-- Check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Main theorem -/
theorem greatest_n_perfect_square :
  (∀ k : ℕ, k ≤ 2023 → is_perfect_square (product_sum_squares k) → k ≤ 1921) ∧
  is_perfect_square (product_sum_squares 1921) := by sorry

end NUMINAMATH_CALUDE_greatest_n_perfect_square_l919_91981


namespace NUMINAMATH_CALUDE_exists_diagonal_le_two_l919_91909

-- Define a convex hexagon
structure ConvexHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_convex : sorry -- Add convexity condition

-- Define the property that all sides have length ≤ 1
def all_sides_le_one (h : ConvexHexagon) : Prop :=
  ∀ i : Fin 6, dist (h.vertices i) (h.vertices ((i + 1) % 6)) ≤ 1

-- Define a diagonal of the hexagon
def diagonal (h : ConvexHexagon) (i j : Fin 6) : ℝ :=
  dist (h.vertices i) (h.vertices j)

-- Theorem statement
theorem exists_diagonal_le_two (h : ConvexHexagon) (h_sides : all_sides_le_one h) :
  ∃ (i j : Fin 6), i ≠ j ∧ diagonal h i j ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_diagonal_le_two_l919_91909


namespace NUMINAMATH_CALUDE_membership_change_theorem_l919_91903

/-- Calculates the overall percentage change in membership across four seasons -/
def overallPercentageChange (fall winter spring summer : ℝ) : ℝ :=
  let afterFall := 1 + fall
  let afterWinter := afterFall * (1 + winter)
  let afterSpring := afterWinter * (1 + spring)
  let afterSummer := afterSpring * (1 + summer)
  (afterSummer - 1) * 100

/-- Theorem stating the overall percentage change in membership -/
theorem membership_change_theorem :
  let fall := 0.06
  let winter := 0.10
  let spring := -0.19
  let summer := 0.05
  abs (overallPercentageChange fall winter spring summer + 0.832) < 0.001 := by
  sorry

#eval overallPercentageChange 0.06 0.10 (-0.19) 0.05

end NUMINAMATH_CALUDE_membership_change_theorem_l919_91903


namespace NUMINAMATH_CALUDE_remainder_problem_l919_91917

theorem remainder_problem (x : ℤ) : x % 84 = 25 → x % 14 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l919_91917


namespace NUMINAMATH_CALUDE_square_sum_identity_l919_91912

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l919_91912


namespace NUMINAMATH_CALUDE_unique_number_property_l919_91938

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_property_l919_91938


namespace NUMINAMATH_CALUDE_length_of_lm_l919_91972

/-- An isosceles triangle with given properties -/
structure IsoscelesTriangle where
  area : ℝ
  altitude : ℝ
  base : ℝ

/-- A line segment parallel to the base of the triangle -/
structure ParallelLine where
  length : ℝ

/-- The resulting trapezoid after cutting the triangle -/
structure Trapezoid where
  area : ℝ

/-- Theorem: Length of LM in the given isosceles triangle scenario -/
theorem length_of_lm (triangle : IsoscelesTriangle) (trapezoid : Trapezoid) 
    (h1 : triangle.area = 200)
    (h2 : triangle.altitude = 40)
    (h3 : trapezoid.area = 150)
    (h4 : triangle.base = 2 * triangle.area / triangle.altitude) :
  ∃ (lm : ParallelLine), lm.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_length_of_lm_l919_91972


namespace NUMINAMATH_CALUDE_girls_math_questions_l919_91945

def total_questions (fiona_per_hour shirley_per_hour kiana_per_hour : ℕ) (hours : ℕ) : ℕ :=
  (fiona_per_hour + shirley_per_hour + kiana_per_hour) * hours

theorem girls_math_questions :
  ∀ (fiona_per_hour : ℕ),
    fiona_per_hour = 36 →
    ∀ (shirley_per_hour : ℕ),
      shirley_per_hour = 2 * fiona_per_hour →
      ∀ (kiana_per_hour : ℕ),
        kiana_per_hour = (fiona_per_hour + shirley_per_hour) / 2 →
        total_questions fiona_per_hour shirley_per_hour kiana_per_hour 2 = 324 :=
by
  sorry

#eval total_questions 36 72 54 2

end NUMINAMATH_CALUDE_girls_math_questions_l919_91945


namespace NUMINAMATH_CALUDE_third_tea_price_l919_91971

/-- The price of the first variety of tea in Rs per kg -/
def price1 : ℝ := 126

/-- The price of the second variety of tea in Rs per kg -/
def price2 : ℝ := 135

/-- The price of the mixture in Rs per kg -/
def mixPrice : ℝ := 153

/-- The ratio of the first variety in the mixture -/
def ratio1 : ℝ := 1

/-- The ratio of the second variety in the mixture -/
def ratio2 : ℝ := 1

/-- The ratio of the third variety in the mixture -/
def ratio3 : ℝ := 2

/-- The theorem stating the price of the third variety of tea -/
theorem third_tea_price : 
  ∃ (price3 : ℝ), 
    (ratio1 * price1 + ratio2 * price2 + ratio3 * price3) / (ratio1 + ratio2 + ratio3) = mixPrice ∧ 
    price3 = 175.5 := by
  sorry

end NUMINAMATH_CALUDE_third_tea_price_l919_91971


namespace NUMINAMATH_CALUDE_cider_production_l919_91929

theorem cider_production (golden_per_pint pink_per_pint : ℕ)
  (num_farmhands work_hours : ℕ) (total_pints : ℕ) :
  golden_per_pint = 20 →
  pink_per_pint = 40 →
  num_farmhands = 6 →
  work_hours = 5 →
  total_pints = 120 →
  (∃ (apples_per_hour : ℕ),
    apples_per_hour * num_farmhands * work_hours = 
      (golden_per_pint + pink_per_pint) * total_pints ∧
    3 * (golden_per_pint * total_pints) = 
      (golden_per_pint + pink_per_pint) * total_pints ∧
    apples_per_hour = 240) :=
by sorry

end NUMINAMATH_CALUDE_cider_production_l919_91929


namespace NUMINAMATH_CALUDE_travel_theorem_l919_91989

/-- Represents the speeds of Butch, Sundance, and Sparky in miles per hour -/
structure Speeds where
  butch : ℝ
  sundance : ℝ
  sparky : ℝ

/-- Represents the distance traveled and time taken -/
structure TravelData where
  distance : ℕ  -- in miles
  time : ℕ      -- in minutes

/-- The main theorem representing the problem -/
theorem travel_theorem (speeds : Speeds) (h1 : speeds.butch = 4)
    (h2 : speeds.sundance = 2.5) (h3 : speeds.sparky = 6) : 
    ∃ (data : TravelData), data.distance = 19 ∧ data.time = 330 ∧ 
    data.distance + data.time = 349 := by
  sorry

#check travel_theorem

end NUMINAMATH_CALUDE_travel_theorem_l919_91989


namespace NUMINAMATH_CALUDE_same_color_pair_count_l919_91985

/-- The number of ways to choose a pair of socks of the same color -/
def choose_same_color_pair (white : Nat) (brown : Nat) (blue : Nat) : Nat :=
  Nat.choose white 2 + Nat.choose brown 2 + Nat.choose blue 2

/-- Theorem: The number of ways to choose a pair of socks of the same color
    from 4 white, 4 brown, and 2 blue socks is 13 -/
theorem same_color_pair_count :
  choose_same_color_pair 4 4 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_same_color_pair_count_l919_91985


namespace NUMINAMATH_CALUDE_theater_revenue_l919_91950

/-- Calculates the total revenue for a theater performance series -/
theorem theater_revenue (seats : ℕ) (capacity : ℚ) (ticket_price : ℕ) (days : ℕ) :
  seats = 400 →
  capacity = 4/5 →
  ticket_price = 30 →
  days = 3 →
  (seats : ℚ) * capacity * (ticket_price : ℚ) * (days : ℚ) = 28800 := by
sorry

end NUMINAMATH_CALUDE_theater_revenue_l919_91950


namespace NUMINAMATH_CALUDE_expand_expression_l919_91956

theorem expand_expression (x : ℝ) : 12 * (3 * x - 4) = 36 * x - 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l919_91956


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l919_91975

-- Define the set M
def M : Set ℝ := {x | 0 < x ∧ x < 27}

-- Define the set N
def N : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem statement
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = Set.Ioo 0 5 := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l919_91975


namespace NUMINAMATH_CALUDE_largest_three_digit_base5_l919_91933

-- Define a function to convert a three-digit base-5 number to base-10
def base5ToBase10 (a b c : Nat) : Nat :=
  a * 5^2 + b * 5^1 + c * 5^0

-- Theorem statement
theorem largest_three_digit_base5 : 
  base5ToBase10 4 4 4 = 124 := by sorry

end NUMINAMATH_CALUDE_largest_three_digit_base5_l919_91933


namespace NUMINAMATH_CALUDE_odometer_puzzle_l919_91984

theorem odometer_puzzle (a b c : ℕ) : 
  (a ≥ 1) →
  (a + b + c ≤ 9) →
  (100 * b + 10 * a + c - (100 * a + 10 * b + c)) % 60 = 0 →
  a^2 + b^2 + c^2 = 35 := by
sorry

end NUMINAMATH_CALUDE_odometer_puzzle_l919_91984


namespace NUMINAMATH_CALUDE_marlington_orchestra_max_members_l919_91983

theorem marlington_orchestra_max_members :
  ∀ n : ℕ,
  (∃ k : ℕ, 30 * n = 31 * k + 5) →
  30 * n < 1500 →
  30 * n ≤ 780 :=
by sorry

end NUMINAMATH_CALUDE_marlington_orchestra_max_members_l919_91983


namespace NUMINAMATH_CALUDE_fridge_cost_difference_l919_91999

theorem fridge_cost_difference (total_budget : ℕ) (tv_cost : ℕ) (computer_cost : ℕ) 
  (h1 : total_budget = 1600)
  (h2 : tv_cost = 600)
  (h3 : computer_cost = 250)
  (h4 : ∃ fridge_cost : ℕ, fridge_cost > computer_cost ∧ 
        fridge_cost + tv_cost + computer_cost = total_budget) :
  ∃ fridge_cost : ℕ, fridge_cost - computer_cost = 500 := by
sorry

end NUMINAMATH_CALUDE_fridge_cost_difference_l919_91999


namespace NUMINAMATH_CALUDE_nomogram_relationships_l919_91968

/-- A structure representing the scales in the nomogram -/
structure Scales where
  X : ℝ
  Y : ℝ
  Z : ℝ
  W : ℝ
  V : ℝ
  U : ℝ
  T : ℝ
  S : ℝ

/-- The theorem stating the relationships between the scales -/
theorem nomogram_relationships (scales : Scales) :
  scales.Z = (scales.X + scales.Y) / 2 ∧
  scales.W = scales.X + scales.Y ∧
  scales.Y = scales.W - scales.X ∧
  scales.V = 2 * (scales.X + scales.Z) ∧
  scales.X + scales.Z + 5 * scales.U = 0 ∧
  scales.T = (6 + scales.Y + scales.Z) / 2 ∧
  scales.Y + scales.Z + 4 * scales.S - 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_nomogram_relationships_l919_91968


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l919_91997

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a > b → a > b - 1) ∧ 
  (∃ a b, a > b - 1 ∧ ¬(a > b)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l919_91997


namespace NUMINAMATH_CALUDE_infinite_sum_convergence_l919_91902

theorem infinite_sum_convergence : 
  ∑' n : ℕ+, (n : ℝ) * Real.sin n / ((n : ℝ)^4 + 8*(n : ℝ)^2 + 16) = 0 := by sorry

end NUMINAMATH_CALUDE_infinite_sum_convergence_l919_91902


namespace NUMINAMATH_CALUDE_distance_to_nearest_town_l919_91957

theorem distance_to_nearest_town (d : ℝ) : 
  (¬ (d ≥ 8)) →
  (¬ (d ≤ 7)) →
  (¬ (d ≤ 6)) →
  (d ≠ 9) →
  7 < d ∧ d < 8 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_nearest_town_l919_91957


namespace NUMINAMATH_CALUDE_negation_equivalence_l919_91998

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 < 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l919_91998


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l919_91952

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m : Line) (α β : Plane) (h_diff : α ≠ β) :
  parallel m α → perpendicular m β → plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l919_91952


namespace NUMINAMATH_CALUDE_smallest_n_for_P_less_than_1000th_l919_91991

def P (n : ℕ) : ℚ :=
  (2^(n-1) * Nat.factorial (n-1)) / (Nat.factorial (2*n-1) * (2*n+1))

theorem smallest_n_for_P_less_than_1000th (n : ℕ) : n = 18 ↔ 
  (n > 0 ∧ P n < 1/1000 ∧ ∀ m : ℕ, m > 0 ∧ m < n → P m ≥ 1/1000) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_P_less_than_1000th_l919_91991


namespace NUMINAMATH_CALUDE_initial_tagged_fish_calculation_l919_91934

/-- Calculates the number of initially tagged fish in a pond -/
def initiallyTaggedFish (totalFish : ℕ) (secondCatchTotal : ℕ) (secondCatchTagged : ℕ) : ℕ :=
  (totalFish * secondCatchTagged) / secondCatchTotal

theorem initial_tagged_fish_calculation :
  initiallyTaggedFish 1250 50 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_tagged_fish_calculation_l919_91934


namespace NUMINAMATH_CALUDE_other_number_in_product_l919_91924

theorem other_number_in_product (P w n : ℕ) : 
  P % 2^4 = 0 →
  P % 3^3 = 0 →
  P % 13^3 = 0 →
  P = n * w →
  w > 0 →
  w = 468 →
  (∀ w' : ℕ, w' > 0 ∧ w' < w → ¬(P % w' = 0)) →
  n = 2028 := by
sorry

end NUMINAMATH_CALUDE_other_number_in_product_l919_91924


namespace NUMINAMATH_CALUDE_value_of_expression_l919_91928

theorem value_of_expression (x y : ℝ) 
  (h1 : x^2 + x*y = 3) 
  (h2 : x*y + y^2 = -2) : 
  2*x^2 - x*y - 3*y^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l919_91928


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l919_91913

theorem arithmetic_mean_of_fractions : 
  let a := 8 / 11
  let b := 5 / 6
  let c := 19 / 22
  b = (a + c) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l919_91913


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l919_91944

theorem divisibility_equivalence (n : ℕ+) :
  11 ∣ (n.val^5 + 5^n.val) ↔ 11 ∣ (n.val^5 * 5^n.val + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l919_91944


namespace NUMINAMATH_CALUDE_double_age_proof_l919_91936

/-- The number of years in the future when Richard will be twice as old as Scott -/
def years_until_double_age : ℕ := 8

theorem double_age_proof (david_current_age richard_current_age scott_current_age : ℕ) 
  (h1 : david_current_age = 14)
  (h2 : richard_current_age = david_current_age + 6)
  (h3 : david_current_age = scott_current_age + 8) :
  richard_current_age + years_until_double_age = 2 * (scott_current_age + years_until_double_age) := by
  sorry

#check double_age_proof

end NUMINAMATH_CALUDE_double_age_proof_l919_91936


namespace NUMINAMATH_CALUDE_polynomial_factorization_l919_91978

theorem polynomial_factorization (x y : ℝ) : 
  (x - 2*y) * (x - 2*y + 1) = x^2 - 4*x*y - 2*y + x + 4*y^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l919_91978


namespace NUMINAMATH_CALUDE_max_area_rectangle_perimeter_36_l919_91973

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Theorem: The maximum area of a rectangle with perimeter 36 is 81 -/
theorem max_area_rectangle_perimeter_36 :
  (∃ (r : Rectangle), perimeter r = 36 ∧ 
    ∀ (s : Rectangle), perimeter s = 36 → area s ≤ area r) ∧
  (∀ (r : Rectangle), perimeter r = 36 → area r ≤ 81) := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_perimeter_36_l919_91973


namespace NUMINAMATH_CALUDE_pams_apples_l919_91955

theorem pams_apples (pam_bags : ℕ) (gerald_apples_per_bag : ℕ) 
  (h1 : pam_bags = 10)
  (h2 : gerald_apples_per_bag = 40) :
  pam_bags * (3 * gerald_apples_per_bag) = 1200 := by
  sorry

end NUMINAMATH_CALUDE_pams_apples_l919_91955


namespace NUMINAMATH_CALUDE_range_of_m_for_equation_l919_91974

/-- Given that the equation e^(mx) = x^2 has two distinct real roots in the interval (0, 16),
    prove that the range of values for the real number m is (ln(2)/2, 2/e). -/
theorem range_of_m_for_equation (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 16 ∧ 
   Real.exp (m * x₁) = x₁^2 ∧ Real.exp (m * x₂) = x₂^2) →
  (Real.log 2 / 2 < m ∧ m < 2 / Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_equation_l919_91974


namespace NUMINAMATH_CALUDE_trapezoid_existence_l919_91943

-- Define the trapezoid structure
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ

-- Define the existence theorem
theorem trapezoid_existence (a b c α β : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) : 
  ∃ t : Trapezoid, 
    t.a = a ∧ t.b = b ∧ t.c = c ∧ t.α = α ∧ t.β = β :=
sorry


end NUMINAMATH_CALUDE_trapezoid_existence_l919_91943


namespace NUMINAMATH_CALUDE_first_load_theorem_l919_91941

/-- Calculates the number of pieces of clothing in the first load -/
def first_load_pieces (total_pieces : ℕ) (num_equal_loads : ℕ) (pieces_per_equal_load : ℕ) : ℕ :=
  total_pieces - (num_equal_loads * pieces_per_equal_load)

/-- Theorem stating that given 59 total pieces of clothing, with 9 equal loads of 3 pieces each,
    the number of pieces in the first load is 32. -/
theorem first_load_theorem :
  first_load_pieces 59 9 3 = 32 := by
  sorry

end NUMINAMATH_CALUDE_first_load_theorem_l919_91941


namespace NUMINAMATH_CALUDE_outfit_problem_l919_91993

/-- The number of possible outfits given shirts, pants, and restrictions -/
def num_outfits (shirts : ℕ) (pants : ℕ) (restricted_shirts : ℕ) (restricted_pants : ℕ) : ℕ :=
  (shirts - restricted_shirts) * pants + restricted_shirts * (pants - restricted_pants)

/-- Theorem stating the number of outfits for the given problem -/
theorem outfit_problem :
  num_outfits 5 4 2 1 = 18 := by
  sorry


end NUMINAMATH_CALUDE_outfit_problem_l919_91993


namespace NUMINAMATH_CALUDE_equation_system_solution_l919_91949

theorem equation_system_solution :
  ∀ (x y z : ℤ),
    (4 : ℝ) ^ (x^2 + 2*x*y + 1) = (z + 2 : ℝ) * 7^(|y| - 1) →
    Real.sin ((3 * Real.pi * ↑z) / 2) = 1 →
    ((x = 1 ∧ y = -1 ∧ z = -1) ∨ (x = -1 ∧ y = 1 ∧ z = -1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l919_91949


namespace NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l919_91940

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 2) * x - Real.log x

theorem f_monotonicity_and_zeros (a : ℝ) :
  (∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → f a x₁ > f a x₂) ∨
  (∃ c, 0 < c ∧ 
    (∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ ∧ x₂ < c → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, c < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
  (∃ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ 0 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_zeros_l919_91940


namespace NUMINAMATH_CALUDE_square_difference_divided_l919_91954

theorem square_difference_divided : (180^2 - 150^2) / 30 = 330 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_l919_91954


namespace NUMINAMATH_CALUDE_max_five_cent_coins_l919_91920

theorem max_five_cent_coins (x y z : ℕ) : 
  x + y + z = 25 →
  x + 2*y + 5*z = 60 →
  z ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_five_cent_coins_l919_91920


namespace NUMINAMATH_CALUDE_geometric_sequence_log_sum_l919_91986

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The logarithm function (base 10) -/
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- Theorem: In a geometric sequence where a₂ * a₅ * a₈ = 1, lg(a₄) + lg(a₆) = 0 -/
theorem geometric_sequence_log_sum (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_prod : a 2 * a 5 * a 8 = 1) : 
  lg (a 4) + lg (a 6) = 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_log_sum_l919_91986


namespace NUMINAMATH_CALUDE_schedule_arrangements_eq_192_l919_91979

/-- Represents the number of periods in a day -/
def num_periods : ℕ := 6

/-- Represents the number of subjects to be scheduled -/
def num_subjects : ℕ := 6

/-- Represents the number of morning periods -/
def num_morning_periods : ℕ := 4

/-- Represents the number of afternoon periods -/
def num_afternoon_periods : ℕ := 2

/-- Calculates the number of ways to arrange the schedule given the constraints -/
def schedule_arrangements : ℕ :=
  (num_morning_periods.choose 1) * (num_afternoon_periods.choose 1) * (num_subjects - 2).factorial

theorem schedule_arrangements_eq_192 : schedule_arrangements = 192 := by
  sorry

end NUMINAMATH_CALUDE_schedule_arrangements_eq_192_l919_91979


namespace NUMINAMATH_CALUDE_parallelogram_area_l919_91926

def a : Fin 3 → ℝ := ![2, -1, 1]
def b : Fin 3 → ℝ := ![1, 3, 1]

theorem parallelogram_area : Real.sqrt ((a 0 * b 1 - a 1 * b 0)^2 + 
                                        (a 1 * b 2 - a 2 * b 1)^2 + 
                                        (a 2 * b 0 - a 0 * b 2)^2) = Real.sqrt 66 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l919_91926


namespace NUMINAMATH_CALUDE_train_average_speed_l919_91958

/-- Proves that the average speed of a train including stoppages is 36 kmph,
    given its speed excluding stoppages and the duration of stoppages. -/
theorem train_average_speed
  (speed_without_stops : ℝ)
  (stop_duration : ℝ)
  (h1 : speed_without_stops = 60)
  (h2 : stop_duration = 24)
  : (speed_without_stops * (60 - stop_duration) / 60) = 36 := by
  sorry

end NUMINAMATH_CALUDE_train_average_speed_l919_91958


namespace NUMINAMATH_CALUDE_good_functions_count_l919_91963

/-- A function f: ℤ → {1, 2, ..., n} is good if it satisfies the given condition -/
def IsGoodFunction (n : ℕ) (f : ℤ → Fin n) : Prop :=
  n ≥ 2 ∧ ∀ k : Fin (n-1), ∃ j : ℤ, ∀ m : ℤ,
    (f (m + j) : ℤ) ≡ (f (m + k) : ℤ) - (f m : ℤ) [ZMOD (n+1)]

/-- The number of good functions for a given n -/
def NumberOfGoodFunctions (n : ℕ) : ℕ := sorry

theorem good_functions_count (n : ℕ) :
  (n ≥ 2 ∧ NumberOfGoodFunctions n = n * Nat.totient n) ↔ Nat.Prime (n+1) :=
sorry

end NUMINAMATH_CALUDE_good_functions_count_l919_91963


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l919_91904

theorem rectangle_dimensions (x : ℝ) : 
  (x + 1 > 0) → 
  (3*x - 4 > 0) → 
  (x + 1) * (3*x - 4) = 12*x - 19 → 
  x = (13 + Real.sqrt 349) / 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l919_91904


namespace NUMINAMATH_CALUDE_jerry_tickets_l919_91919

theorem jerry_tickets (initial_tickets spent_tickets later_won_tickets current_tickets : ℕ) :
  spent_tickets = 2 →
  later_won_tickets = 47 →
  current_tickets = 49 →
  initial_tickets = current_tickets - later_won_tickets + spent_tickets →
  initial_tickets = 4 := by
sorry

end NUMINAMATH_CALUDE_jerry_tickets_l919_91919


namespace NUMINAMATH_CALUDE_complement_intersection_l919_91939

-- Define the universe U
def U : Set ℕ := {x | x ≤ 5}

-- Define sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

-- Theorem statement
theorem complement_intersection :
  (U \ A) ∩ (U \ B) = {0, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_l919_91939


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l919_91982

def point_A : ℝ × ℝ := (-2, 1)
def point_B : ℝ × ℝ := (9, 3)
def point_C : ℝ × ℝ := (1, 7)

def circle_equation (x y : ℝ) : Prop :=
  (x - 7/2)^2 + (y - 2)^2 = 125/4

theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 :=
by sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l919_91982


namespace NUMINAMATH_CALUDE_october_visitors_l919_91970

theorem october_visitors (oct nov dec : ℕ) : 
  nov = oct * 115 / 100 →
  dec = nov + 15 →
  oct + nov + dec = 345 →
  oct = 100 := by
sorry

end NUMINAMATH_CALUDE_october_visitors_l919_91970


namespace NUMINAMATH_CALUDE_mechanism_efficiency_problem_l919_91947

theorem mechanism_efficiency_problem (t_combined t_partial t_remaining : ℝ) 
  (h_combined : t_combined = 30)
  (h_partial : t_partial = 6)
  (h_remaining : t_remaining = 40) :
  ∃ (t1 t2 : ℝ),
    t1 = 75 ∧ 
    t2 = 50 ∧ 
    (1 / t1 + 1 / t2 = 1 / t_combined) ∧
    (t_partial * (1 / t1 + 1 / t2) + t_remaining / t2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_mechanism_efficiency_problem_l919_91947


namespace NUMINAMATH_CALUDE_f_one_eq_zero_f_x_minus_one_lt_zero_iff_f_inequality_iff_l919_91961

/-- An increasing function f satisfying f(x/y) = f(x) - f(y) -/
class SpecialFunction (f : ℝ → ℝ) : Prop where
  domain : ∀ x, x > 0 → f x ≠ 0
  increasing : ∀ x y, x < y → f x < f y
  special_prop : ∀ x y, x > 0 → y > 0 → f (x / y) = f x - f y

variable (f : ℝ → ℝ) [SpecialFunction f]

/-- f(1) = 0 -/
theorem f_one_eq_zero : f 1 = 0 := by sorry

/-- f(x-1) < 0 iff x ∈ (1, 2) -/
theorem f_x_minus_one_lt_zero_iff (x : ℝ) : f (x - 1) < 0 ↔ 1 < x ∧ x < 2 := by sorry

/-- If f(2) = 1, then f(x+3) - f(1/x) < 2 iff x ∈ (0, 1) -/
theorem f_inequality_iff (h : f 2 = 1) (x : ℝ) : f (x + 3) - f (1 / x) < 2 ↔ 0 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_f_one_eq_zero_f_x_minus_one_lt_zero_iff_f_inequality_iff_l919_91961


namespace NUMINAMATH_CALUDE_nomogram_relations_l919_91990

-- Define the nomogram scales as real numbers
variables (x y z t r w v q s : ℝ)

-- Define y₁ as a function of y
def y₁ (y : ℝ) : ℝ := y

-- Theorem statement
theorem nomogram_relations :
  z = (x + 2 * y₁ y) / 3 ∧
  w = 2 * z ∧
  r = x - 2 ∧
  y + q = 6 ∧
  2 * s + t = 8 ∧
  3 * z - x - 2 * t + 6 = 0 ∧
  8 * z - 4 * t - v + 12 = 0 := by
sorry


end NUMINAMATH_CALUDE_nomogram_relations_l919_91990


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l919_91953

-- Problem 1
theorem simplify_expression_1 (x y : ℝ) :
  2 * x^2 - 3 * y^2 + 6 * x - x^2 + 3 * y^2 = x^2 + 6 * x :=
by sorry

-- Problem 2
theorem simplify_expression_2 (m : ℝ) :
  4 * m^2 + 1 + 2 * m - 3 * (2 + m - m^2) = 7 * m^2 - m - 5 :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l919_91953


namespace NUMINAMATH_CALUDE_intersection_M_N_l919_91932

-- Define set M
def M : Set ℝ := {x | x^2 + x - 6 < 0}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 2^x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l919_91932


namespace NUMINAMATH_CALUDE_cost_price_cloth_sale_l919_91901

/-- Given a cloth sale scenario, calculate the cost price per meter. -/
def cost_price_per_meter (total_meters : ℕ) (total_selling_price : ℚ) (profit_per_meter : ℚ) : ℚ :=
  (total_selling_price - (profit_per_meter * total_meters)) / total_meters

/-- Theorem stating the cost price per meter of cloth in the given scenario. -/
theorem cost_price_cloth_sale :
  cost_price_per_meter 85 8925 15 = 90 := by
  sorry


end NUMINAMATH_CALUDE_cost_price_cloth_sale_l919_91901


namespace NUMINAMATH_CALUDE_flute_cost_l919_91995

def total_spent : ℝ := 158.35
def music_stand_cost : ℝ := 8.89
def song_book_cost : ℝ := 7

theorem flute_cost : 
  ∃ (flute_cost : ℝ), 
    flute_cost + music_stand_cost + song_book_cost = total_spent ∧ 
    flute_cost = 142.46 := by
  sorry

end NUMINAMATH_CALUDE_flute_cost_l919_91995


namespace NUMINAMATH_CALUDE_tangent_parallel_points_tangent_parallel_result_point_coordinates_l919_91980

-- Define the function f(x) = x^3 + x - 2
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_parallel_points :
  ∀ x : ℝ, (f' x = 4) ↔ (x = 1 ∨ x = -1) :=
by sorry

theorem tangent_parallel_result :
  {x : ℝ | f' x = 4} = {1, -1} :=
by sorry

theorem point_coordinates :
  {p : ℝ × ℝ | ∃ x, p.1 = x ∧ p.2 = f x ∧ f' x = 4} = {(1, 0), (-1, -4)} :=
by sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_tangent_parallel_result_point_coordinates_l919_91980


namespace NUMINAMATH_CALUDE_unique_solution_condition_l919_91964

theorem unique_solution_condition (a b : ℝ) : 
  (∃! x : ℝ, 4 * x - 6 + a = (b + 1) * x + 2) ↔ b ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l919_91964


namespace NUMINAMATH_CALUDE_possible_distances_l919_91900

/-- Three points on a line -/
structure PointsOnLine where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The distance between two points on a line -/
def distance (x y : ℝ) : ℝ := |x - y|

theorem possible_distances (p : PointsOnLine) 
  (h1 : distance p.A p.B = 1)
  (h2 : distance p.B p.C = 3) :
  distance p.A p.C = 4 ∨ distance p.A p.C = 2 := by
  sorry

end NUMINAMATH_CALUDE_possible_distances_l919_91900


namespace NUMINAMATH_CALUDE_odd_increasing_function_inequality_l919_91905

-- Define the properties of function f
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

-- State the theorem
theorem odd_increasing_function_inequality (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_incr : monotone_increasing_on f (Set.Ici 0)) :
  {x : ℝ | f (2*x - 1) < f (x^2 - x + 1)} = 
  Set.union (Set.Iio 1) (Set.Ioi 2) := by sorry

end NUMINAMATH_CALUDE_odd_increasing_function_inequality_l919_91905


namespace NUMINAMATH_CALUDE_ship_cats_count_l919_91969

/-- Represents the passengers on the ship --/
structure ShipPassengers where
  cats : ℕ
  sailors : ℕ
  cook : ℕ
  captain : ℕ

/-- Calculates the total number of heads on the ship --/
def totalHeads (p : ShipPassengers) : ℕ :=
  p.cats + p.sailors + p.cook + p.captain

/-- Calculates the total number of legs on the ship --/
def totalLegs (p : ShipPassengers) : ℕ :=
  4 * p.cats + 2 * p.sailors + 2 * p.cook + p.captain

/-- Theorem stating that given the conditions, the number of cats is 7 --/
theorem ship_cats_count (p : ShipPassengers) 
  (h1 : p.cook = 1) 
  (h2 : p.captain = 1) 
  (h3 : totalHeads p = 16) 
  (h4 : totalLegs p = 45) : 
  p.cats = 7 := by
  sorry


end NUMINAMATH_CALUDE_ship_cats_count_l919_91969


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l919_91960

/-- The product of fractions in the sequence -/
def fraction_product : ℕ → ℚ
| 0 => 3 / 1
| n + 1 => fraction_product n * ((3 * (n + 1) + 6) / (3 * (n + 1)))

/-- The last term in the sequence -/
def last_term : ℚ := 3003 / 2997

/-- The number of terms in the sequence -/
def num_terms : ℕ := 999

theorem fraction_product_simplification :
  fraction_product num_terms * last_term = 1001 := by
  sorry


end NUMINAMATH_CALUDE_fraction_product_simplification_l919_91960


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l919_91922

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposites
    and their y-coordinates are equal -/
def symmetric_wrt_y_axis (a b : ℝ × ℝ) : Prop :=
  a.1 = -b.1 ∧ a.2 = b.2

theorem symmetric_points_sum (x y : ℝ) :
  let a : ℝ × ℝ := (3, 2)
  let b : ℝ × ℝ := (x - 4, 6 + y)
  symmetric_wrt_y_axis a b → x + y = -3 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l919_91922


namespace NUMINAMATH_CALUDE_intersection_A_B_l919_91921

def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 2}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l919_91921


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l919_91992

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 17 % 31 ∧
  ∀ (y : ℕ), y > 0 → (5 * y) % 31 = 17 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l919_91992


namespace NUMINAMATH_CALUDE_tan_graph_property_l919_91911

theorem tan_graph_property (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + π))) → 
  a * Real.tan (b * (π / 4)) = 3 → 
  a * b = 3 := by
sorry

end NUMINAMATH_CALUDE_tan_graph_property_l919_91911
