import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l3661_366102

theorem smallest_n_for_roots_of_unity : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∀ z : ℂ, z^4 - z^2 + 1 = 0 → z^n = 1) ∧
  (∀ m : ℕ, m > 0 → (∀ z : ℂ, z^4 - z^2 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l3661_366102


namespace NUMINAMATH_CALUDE_profit_share_difference_example_l3661_366144

/-- Given a total profit and a ratio of division between two parties, 
    calculate the difference between their shares. -/
def profit_share_difference (total_profit : ℚ) (ratio_x ratio_y : ℚ) : ℚ :=
  let total_ratio := ratio_x + ratio_y
  let share_x := (ratio_x / total_ratio) * total_profit
  let share_y := (ratio_y / total_ratio) * total_profit
  share_x - share_y

/-- Theorem stating that for a total profit of 500 and a ratio of 1/2 : 1/3, 
    the difference in profit shares is 100. -/
theorem profit_share_difference_example : 
  profit_share_difference 500 (1/2) (1/3) = 100 := by
  sorry

#eval profit_share_difference 500 (1/2) (1/3)

end NUMINAMATH_CALUDE_profit_share_difference_example_l3661_366144


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3661_366118

theorem fraction_to_decimal : (45 : ℚ) / (2^2 * 5^3) = (9 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3661_366118


namespace NUMINAMATH_CALUDE_A_l3661_366136

def A' : ℕ → ℕ → ℕ → ℕ
  | 0, n, k => n + k
  | m+1, 0, k => A' m k 1
  | m+1, n+1, k => A' m (A' (m+1) n k) k

theorem A'_3_2_2 : A' 3 2 2 = 17 := by sorry

end NUMINAMATH_CALUDE_A_l3661_366136


namespace NUMINAMATH_CALUDE_moving_trips_l3661_366160

theorem moving_trips (total_time : ℕ) (fill_time : ℕ) (drive_time : ℕ) : 
  total_time = 7 * 60 ∧ fill_time = 15 ∧ drive_time = 30 →
  (total_time / (fill_time + 2 * drive_time) : ℕ) = 5 := by
sorry

end NUMINAMATH_CALUDE_moving_trips_l3661_366160


namespace NUMINAMATH_CALUDE_emily_took_55_apples_l3661_366110

/-- The number of apples Ruby initially had -/
def initial_apples : ℕ := 63

/-- The number of apples Ruby has left -/
def remaining_apples : ℕ := 8

/-- The number of apples Emily took -/
def emily_took : ℕ := initial_apples - remaining_apples

/-- Theorem stating that Emily took 55 apples -/
theorem emily_took_55_apples : emily_took = 55 := by
  sorry

end NUMINAMATH_CALUDE_emily_took_55_apples_l3661_366110


namespace NUMINAMATH_CALUDE_triangle_special_condition_l3661_366141

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem triangle_special_condition (t : Triangle) :
  t.a^2 = 3*t.b^2 + 3*t.c^2 - 2*Real.sqrt 3*t.b*t.c*Real.sin t.A →
  t.C = π/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_condition_l3661_366141


namespace NUMINAMATH_CALUDE_clinton_school_earnings_l3661_366156

/-- Represents the total compensation for all students -/
def total_compensation : ℝ := 1456

/-- Represents the number of students from Arlington school -/
def arlington_students : ℕ := 8

/-- Represents the number of days Arlington students worked -/
def arlington_days : ℕ := 4

/-- Represents the number of students from Bradford school -/
def bradford_students : ℕ := 6

/-- Represents the number of days Bradford students worked -/
def bradford_days : ℕ := 7

/-- Represents the number of students from Clinton school -/
def clinton_students : ℕ := 7

/-- Represents the number of days Clinton students worked -/
def clinton_days : ℕ := 8

/-- Theorem stating that the total earnings for Clinton school students is 627.20 dollars -/
theorem clinton_school_earnings :
  let total_student_days := arlington_students * arlington_days + bradford_students * bradford_days + clinton_students * clinton_days
  let daily_wage := total_compensation / total_student_days
  clinton_students * clinton_days * daily_wage = 627.2 := by
  sorry

end NUMINAMATH_CALUDE_clinton_school_earnings_l3661_366156


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l3661_366191

theorem quadratic_equation_result (a : ℝ) (h : 2 * a^2 - 3 * a + 4 = 5) :
  7 + 6 * a - 4 * a^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l3661_366191


namespace NUMINAMATH_CALUDE_sarahs_mean_score_l3661_366132

def scores : List ℕ := [78, 80, 85, 87, 90, 95, 100]

theorem sarahs_mean_score 
  (john_score_count : ℕ) 
  (sarah_score_count : ℕ)
  (total_score_count : ℕ)
  (john_mean : ℚ)
  (h1 : john_score_count = 4)
  (h2 : sarah_score_count = 3)
  (h3 : total_score_count = john_score_count + sarah_score_count)
  (h4 : john_mean = 86)
  (h5 : (scores.sum : ℚ) = john_mean * john_score_count + sarah_score_count * sarah_mean) :
  sarah_mean = 90 + 1/3 := by
    sorry

#check sarahs_mean_score

end NUMINAMATH_CALUDE_sarahs_mean_score_l3661_366132


namespace NUMINAMATH_CALUDE_condition_relationship_l3661_366171

theorem condition_relationship :
  (∀ x : ℝ, x^2 - x - 2 < 0 → |x| < 2) ∧
  (∃ x : ℝ, |x| < 2 ∧ x^2 - x - 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l3661_366171


namespace NUMINAMATH_CALUDE_smallest_n_repeating_decimal_l3661_366173

/-- A number is a repeating decimal with period k if it can be expressed as m/(10^k - 1) for some integer m -/
def is_repeating_decimal (x : ℚ) (k : ℕ) : Prop :=
  ∃ m : ℤ, x = m / (10^k - 1)

/-- The smallest positive integer n < 1000 such that 1/n is a repeating decimal with period 3
    and 1/(n+6) is a repeating decimal with period 2 is 27 -/
theorem smallest_n_repeating_decimal : 
  ∃ n : ℕ, n < 1000 ∧ 
           is_repeating_decimal (1 / n) 3 ∧ 
           is_repeating_decimal (1 / (n + 6)) 2 ∧
           ∀ m : ℕ, m < n → ¬(is_repeating_decimal (1 / m) 3 ∧ is_repeating_decimal (1 / (m + 6)) 2) ∧
           n = 27 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_repeating_decimal_l3661_366173


namespace NUMINAMATH_CALUDE_shaded_cubes_count_l3661_366178

/-- Represents a 4x4x4 cube with a specific shading pattern -/
structure ShadedCube where
  /-- Total number of smaller cubes -/
  total_cubes : Nat
  /-- Number of cubes per face -/
  cubes_per_face : Nat
  /-- Number of shaded cubes on one face -/
  shaded_per_face : Nat
  /-- Number of corner cubes -/
  corner_cubes : Nat
  /-- Number of edge cubes -/
  edge_cubes : Nat
  /-- Condition: The cube is 4x4x4 -/
  is_4x4x4 : total_cubes = 64 ∧ cubes_per_face = 16
  /-- Condition: Shading pattern on one face -/
  shading_pattern : shaded_per_face = 9
  /-- Condition: Number of corners and edges -/
  cube_structure : corner_cubes = 8 ∧ edge_cubes = 12

/-- Theorem: The number of shaded cubes in the given 4x4x4 cube is 33 -/
theorem shaded_cubes_count (c : ShadedCube) : 
  c.corner_cubes + c.edge_cubes + (3 * c.shaded_per_face - c.corner_cubes - c.edge_cubes) = 33 := by
  sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_l3661_366178


namespace NUMINAMATH_CALUDE_division_problem_l3661_366101

theorem division_problem (R Q D : ℕ) : 
  D = 3 * Q ∧ 
  D = 3 * R + 3 ∧ 
  113 = D * Q + R → 
  R = 5 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3661_366101


namespace NUMINAMATH_CALUDE_yard_length_26_trees_l3661_366149

/-- The length of a yard with equally spaced trees -/
def yardLength (n : ℕ) (d : ℝ) : ℝ := (n - 1 : ℝ) * d

/-- Theorem: The length of a yard with 26 equally spaced trees, 
    one at each end, and 12 meters between consecutive trees, is 300 meters. -/
theorem yard_length_26_trees : yardLength 26 12 = 300 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_26_trees_l3661_366149


namespace NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_sqrt_16_l3661_366111

theorem cube_root_125_times_fourth_root_256_times_sqrt_16 : 
  (125 : ℝ) ^ (1/3) * (256 : ℝ) ^ (1/4) * (16 : ℝ) ^ (1/2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_125_times_fourth_root_256_times_sqrt_16_l3661_366111


namespace NUMINAMATH_CALUDE_correct_equation_transformation_l3661_366143

theorem correct_equation_transformation (x : ℝ) : 
  (x / 3 = 7) → (x = 21) :=
by sorry

end NUMINAMATH_CALUDE_correct_equation_transformation_l3661_366143


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_equals_l3661_366182

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 2) :
  ∀ z w : ℝ, z > 0 → w > 0 → 1/z + 1/w = 2 → x + 2*y ≤ z + 2*w :=
by sorry

theorem min_value_equals (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 2) :
  ∃ z w : ℝ, z > 0 ∧ w > 0 ∧ 1/z + 1/w = 2 ∧ z + 2*w = (3 + 2*Real.sqrt 2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_equals_l3661_366182


namespace NUMINAMATH_CALUDE_intersection_segment_length_l3661_366185

/-- Line l in Cartesian coordinates -/
def line_l (x y : ℝ) : Prop := x + y = 0

/-- Curve C in Cartesian coordinates -/
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

/-- The length of segment AB formed by the intersection of line l and curve C -/
theorem intersection_segment_length :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l3661_366185


namespace NUMINAMATH_CALUDE_silk_order_total_l3661_366120

/-- Calculates the total yards of silk dyed given the yards of each color and the percentage of red silk -/
def total_silk_dyed (green pink blue yellow : ℝ) (red_percent : ℝ) : ℝ :=
  let non_red := green + pink + blue + yellow
  let red := red_percent * non_red
  non_red + red

/-- Theorem stating the total yards of silk dyed for the given order -/
theorem silk_order_total :
  total_silk_dyed 61921 49500 75678 34874.5 0.1 = 245270.85 := by
  sorry

end NUMINAMATH_CALUDE_silk_order_total_l3661_366120


namespace NUMINAMATH_CALUDE_second_group_size_l3661_366174

theorem second_group_size (total : ℕ) (group1 group3 group4 : ℕ) 
  (h1 : total = 24)
  (h2 : group1 = 5)
  (h3 : group3 = 7)
  (h4 : group4 = 4) :
  total - (group1 + group3 + group4) = 8 := by
sorry

end NUMINAMATH_CALUDE_second_group_size_l3661_366174


namespace NUMINAMATH_CALUDE_symmetry_axis_property_l3661_366175

/-- Given a function f(x) = 3sin(x) + 4cos(x), if x = θ is an axis of symmetry
    for the curve y = f(x), then cos(2θ) + sin(θ)cos(θ) = 19/25 -/
theorem symmetry_axis_property (θ : ℝ) :
  (∀ x, 3 * Real.sin x + 4 * Real.cos x = 3 * Real.sin (2 * θ - x) + 4 * Real.cos (2 * θ - x)) →
  Real.cos (2 * θ) + Real.sin θ * Real.cos θ = 19 / 25 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_axis_property_l3661_366175


namespace NUMINAMATH_CALUDE_price_adjustment_l3661_366180

theorem price_adjustment (original_price : ℝ) (original_price_pos : 0 < original_price) : 
  let increased_price := original_price * (1 + 0.25)
  let decrease_percentage := (increased_price - original_price) / increased_price
  decrease_percentage = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_price_adjustment_l3661_366180


namespace NUMINAMATH_CALUDE_sum_divisors_cube_lt_n_fourth_l3661_366162

def S (n : ℕ) : ℕ := sorry

theorem sum_divisors_cube_lt_n_fourth {n : ℕ} (h_odd : Odd n) (h_gt_one : n > 1) :
  (S n)^3 < n^4 := by sorry

end NUMINAMATH_CALUDE_sum_divisors_cube_lt_n_fourth_l3661_366162


namespace NUMINAMATH_CALUDE_garden_area_difference_l3661_366176

-- Define the dimensions of the gardens
def karl_length : ℕ := 30
def karl_width : ℕ := 50
def makenna_length : ℕ := 35
def makenna_width : ℕ := 45

-- Define the areas of the gardens
def karl_area : ℕ := karl_length * karl_width
def makenna_area : ℕ := makenna_length * makenna_width

-- Theorem statement
theorem garden_area_difference :
  makenna_area - karl_area = 75 ∧ makenna_area > karl_area := by
  sorry

end NUMINAMATH_CALUDE_garden_area_difference_l3661_366176


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_solution_l3661_366186

theorem arithmetic_geometric_sequence_solution :
  ∀ a b c : ℝ,
  (b - a = c - b) →                      -- arithmetic sequence
  (a + b + c = 12) →                     -- sum is 12
  ((b + 2)^2 = (a + 2) * (c + 5)) →      -- geometric sequence condition
  ((a = 1 ∧ b = 4 ∧ c = 7) ∨ (a = 10 ∧ b = 4 ∧ c = -2)) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_solution_l3661_366186


namespace NUMINAMATH_CALUDE_alex_walking_distance_l3661_366129

/-- Represents the bike trip with given conditions -/
structure BikeTrip where
  total_distance : ℝ
  flat_time : ℝ
  flat_speed : ℝ
  uphill_time : ℝ
  uphill_speed : ℝ
  downhill_time : ℝ
  downhill_speed : ℝ

/-- Calculates the distance walked given a BikeTrip -/
def distance_walked (trip : BikeTrip) : ℝ :=
  trip.total_distance - (
    trip.flat_time * trip.flat_speed +
    trip.uphill_time * trip.uphill_speed +
    trip.downhill_time * trip.downhill_speed
  )

/-- Proves that Alex walked 8 miles given the conditions of the problem -/
theorem alex_walking_distance :
  let trip : BikeTrip := {
    total_distance := 164,
    flat_time := 4.5,
    flat_speed := 20,
    uphill_time := 2.5,
    uphill_speed := 12,
    downhill_time := 1.5,
    downhill_speed := 24
  }
  distance_walked trip = 8 := by
  sorry

end NUMINAMATH_CALUDE_alex_walking_distance_l3661_366129


namespace NUMINAMATH_CALUDE_total_students_is_600_l3661_366121

/-- Represents a school with boys and girls -/
structure School where
  numBoys : ℕ
  numGirls : ℕ
  avgAgeBoys : ℝ
  avgAgeGirls : ℝ
  avgAgeSchool : ℝ

/-- The conditions of the problem -/
def problemSchool : School :=
  { numBoys := 0,  -- We don't know this yet, so we set it to 0
    numGirls := 150,
    avgAgeBoys := 12,
    avgAgeGirls := 11,
    avgAgeSchool := 11.75 }

/-- The theorem stating that the total number of students is 600 -/
theorem total_students_is_600 (s : School) 
  (h1 : s.numGirls = problemSchool.numGirls)
  (h2 : s.avgAgeBoys = problemSchool.avgAgeBoys)
  (h3 : s.avgAgeGirls = problemSchool.avgAgeGirls)
  (h4 : s.avgAgeSchool = problemSchool.avgAgeSchool)
  (h5 : s.avgAgeSchool * (s.numBoys + s.numGirls) = 
        s.avgAgeBoys * s.numBoys + s.avgAgeGirls * s.numGirls) :
  s.numBoys + s.numGirls = 600 := by
  sorry

#check total_students_is_600

end NUMINAMATH_CALUDE_total_students_is_600_l3661_366121


namespace NUMINAMATH_CALUDE_probability_of_green_ball_l3661_366135

/-- The probability of drawing a green ball from a bag with specified conditions -/
theorem probability_of_green_ball (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h_total : total = 10)
  (h_red : red = 3)
  (h_blue : blue = 2) :
  (total - red - blue) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_green_ball_l3661_366135


namespace NUMINAMATH_CALUDE_officers_count_l3661_366151

/-- The number of ways to choose 5 distinct officers from a group of 12 people -/
def choose_officers : ℕ := 12 * 11 * 10 * 9 * 8

/-- Theorem stating that the number of ways to choose 5 distinct officers 
    from a group of 12 people is 95040 -/
theorem officers_count : choose_officers = 95040 := by
  sorry

end NUMINAMATH_CALUDE_officers_count_l3661_366151


namespace NUMINAMATH_CALUDE_club_size_l3661_366181

/-- The cost of a pair of socks in dollars -/
def sock_cost : ℕ := 3

/-- The cost of a jersey in dollars -/
def jersey_cost : ℕ := sock_cost + 7

/-- The cost of a warm-up jacket in dollars -/
def jacket_cost : ℕ := 2 * jersey_cost

/-- The total cost for one player's equipment in dollars -/
def player_cost : ℕ := 2 * (sock_cost + jersey_cost) + jacket_cost

/-- The total expenditure for the club in dollars -/
def total_expenditure : ℕ := 3276

/-- The number of players in the club -/
def num_players : ℕ := total_expenditure / player_cost

theorem club_size :
  num_players = 71 :=
sorry

end NUMINAMATH_CALUDE_club_size_l3661_366181


namespace NUMINAMATH_CALUDE_longest_boat_through_bend_l3661_366172

theorem longest_boat_through_bend (a : ℝ) (h : a > 0) :
  ∃ c : ℝ, c = 2 * a * Real.sqrt 2 ∧
  ∀ l : ℝ, l > c → ¬ (∃ θ : ℝ, 
    l * Real.cos θ ≤ a ∧ l * Real.sin θ ≤ a) := by
  sorry

end NUMINAMATH_CALUDE_longest_boat_through_bend_l3661_366172


namespace NUMINAMATH_CALUDE_stratified_sampling_results_l3661_366157

theorem stratified_sampling_results (total_sample : ℕ) (junior_students senior_students : ℕ) :
  total_sample = 60 ∧ junior_students = 400 ∧ senior_students = 200 →
  (Nat.choose junior_students ((total_sample * junior_students) / (junior_students + senior_students))) *
  (Nat.choose senior_students ((total_sample * senior_students) / (junior_students + senior_students))) =
  Nat.choose 400 40 * Nat.choose 200 20 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_results_l3661_366157


namespace NUMINAMATH_CALUDE_lines_intersect_at_point_l3661_366106

/-- The first line parameterized by t -/
def line1 (t : ℚ) : ℚ × ℚ := (3 - t, 2 + 4*t)

/-- The second line parameterized by u -/
def line2 (u : ℚ) : ℚ × ℚ := (-1 + 3*u, 3 + 5*u)

/-- The proposed intersection point -/
def intersection_point : ℚ × ℚ := (39/17, 74/17)

theorem lines_intersect_at_point :
  ∃! (t u : ℚ), line1 t = line2 u ∧ line1 t = intersection_point :=
sorry

end NUMINAMATH_CALUDE_lines_intersect_at_point_l3661_366106


namespace NUMINAMATH_CALUDE_class_presentation_periods_l3661_366155

/-- The number of periods required for all student presentations in a class --/
def periods_required (total_students : ℕ) (period_length : ℕ) (individual_presentation_length : ℕ) 
  (group_presentation_length : ℕ) (group_presentations : ℕ) : ℕ :=
  let individual_students := total_students - group_presentations
  let total_minutes := individual_students * individual_presentation_length + 
                       group_presentations * group_presentation_length
  (total_minutes + period_length - 1) / period_length

theorem class_presentation_periods :
  periods_required 32 40 8 12 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_class_presentation_periods_l3661_366155


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l3661_366198

theorem arithmetic_geometric_sequence_properties
  (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- a_n is arithmetic with common difference d
  (d ≠ 0) →  -- nonzero common difference
  (∀ n, b (n + 1) = q * b n) →  -- b_n is geometric with common ratio q
  (b 1 = a 1 ^ 2) →  -- b₁ = a₁²
  (b 2 = a 2 ^ 2) →  -- b₂ = a₂²
  (b 3 = a 3 ^ 2) →  -- b₃ = a₃²
  (a 2 = -1) →  -- a₂ = -1
  (a 1 < a 2) →  -- a₁ < a₂
  (q = 3 - 2 * Real.sqrt 2 ∧ d = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_properties_l3661_366198


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3661_366127

-- Define the logarithm function
noncomputable def lg (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Theorem 1
theorem problem_1 : (lg 2 2)^2 + (lg 2 2) * (lg 2 5) + (lg 2 5) = 1 := by sorry

-- Theorem 2
theorem problem_2 : (2^(1/3) * 3^(1/2))^6 - 8 * (16/49)^(-1/2) - 2^(1/4) * 8^0.25 - (-2016)^0 = 91 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3661_366127


namespace NUMINAMATH_CALUDE_sugar_price_correct_l3661_366199

/-- The price of a kilogram of sugar -/
def sugar_price : ℝ := 1.50

/-- The price of a kilogram of salt -/
noncomputable def salt_price : ℝ := 5 - 3 * sugar_price

theorem sugar_price_correct : sugar_price = 1.50 := by
  have h1 : 2 * sugar_price + 5 * salt_price = 5.50 := by sorry
  have h2 : 3 * sugar_price + salt_price = 5 := by sorry
  sorry

end NUMINAMATH_CALUDE_sugar_price_correct_l3661_366199


namespace NUMINAMATH_CALUDE_triangle_max_area_l3661_366117

/-- Given a triangle ABC with sides a, b, c and area S, where S = a² - (b-c)² 
    and the circumference of its circumcircle is 17π, 
    prove that the maximum value of S is 64. -/
theorem triangle_max_area (a b c S : ℝ) (h1 : S = a^2 - (b - c)^2) 
  (h2 : 2 * Real.pi * (a / (2 * Real.sin (Real.arcsin (8/17)))) = 17 * Real.pi) :
  S ≤ 64 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3661_366117


namespace NUMINAMATH_CALUDE_function_inequality_solution_l3661_366195

/-- Given a real number q with |q| < 1 and q ≠ 0, there exists a function f and a non-negative function g
    satisfying the given conditions. -/
theorem function_inequality_solution (q : ℝ) (hq1 : |q| < 1) (hq2 : q ≠ 0) :
  ∃ (f g : ℝ → ℝ) (a : ℕ → ℝ),
    (∀ x, g x ≥ 0) ∧
    (∀ x, f x = (1 - q * x) * f (q * x) + g x) ∧
    (∀ x, f x = ∑' i, a i * x^i) ∧
    (∀ k, k > 0 → a k = (a (k-1) * q^k - (1 / k.factorial) * (deriv^[k] g) 0) / (q^k - 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_solution_l3661_366195


namespace NUMINAMATH_CALUDE_scientific_notation_equiv_l3661_366163

theorem scientific_notation_equiv : 
  0.0000006 = 6 * 10^(-7) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equiv_l3661_366163


namespace NUMINAMATH_CALUDE_pizza_piece_cost_l3661_366189

/-- Given that 4 pizzas cost $80 in total, and each pizza is cut into 5 pieces,
    prove that the cost of each piece of pizza is $4. -/
theorem pizza_piece_cost : 
  (total_cost : ℝ) →
  (num_pizzas : ℕ) →
  (pieces_per_pizza : ℕ) →
  total_cost = 80 →
  num_pizzas = 4 →
  pieces_per_pizza = 5 →
  (total_cost / (num_pizzas * pieces_per_pizza : ℝ)) = 4 :=
by sorry

end NUMINAMATH_CALUDE_pizza_piece_cost_l3661_366189


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l3661_366116

/-- Represents the number of employees in each age group -/
structure EmployeeGroups where
  over50 : ℕ
  between35and49 : ℕ
  under35 : ℕ

/-- Represents the sampling results for each age group -/
structure SamplingResult where
  over50 : ℕ
  between35and49 : ℕ
  under35 : ℕ

/-- Calculates the correct stratified sampling for given employee groups and sample size -/
def stratifiedSampling (groups : EmployeeGroups) (sampleSize : ℕ) : SamplingResult :=
  sorry

/-- The theorem statement for the stratified sampling problem -/
theorem stratified_sampling_correct 
  (groups : EmployeeGroups)
  (h1 : groups.over50 = 15)
  (h2 : groups.between35and49 = 45)
  (h3 : groups.under35 = 90)
  (h4 : groups.over50 + groups.between35and49 + groups.under35 = 150)
  (sampleSize : ℕ)
  (h5 : sampleSize = 30) :
  stratifiedSampling groups sampleSize = SamplingResult.mk 3 9 18 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_l3661_366116


namespace NUMINAMATH_CALUDE_soccer_goals_proof_l3661_366123

def goals_first_6 : List Nat := [5, 2, 4, 3, 6, 2]

def total_goals_6 : Nat := goals_first_6.sum

theorem soccer_goals_proof (goals_7 goals_8 : Nat) : 
  goals_7 < 7 →
  goals_8 < 7 →
  (total_goals_6 + goals_7) % 7 = 0 →
  (total_goals_6 + goals_7 + goals_8) % 8 = 0 →
  goals_7 * goals_8 = 24 := by
  sorry

#eval total_goals_6

end NUMINAMATH_CALUDE_soccer_goals_proof_l3661_366123


namespace NUMINAMATH_CALUDE_line_translation_l3661_366139

/-- Represents a line in the 2D Cartesian plane -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- The vertical translation distance between two lines -/
def vertical_translation (l1 l2 : Line) : ℝ :=
  l2.y_intercept - l1.y_intercept

theorem line_translation (l1 l2 : Line) :
  l1.slope = -2 ∧ l1.y_intercept = -2 ∧ 
  l2.slope = -2 ∧ l2.y_intercept = 4 →
  vertical_translation l1 l2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_line_translation_l3661_366139


namespace NUMINAMATH_CALUDE_no_numbers_equal_seven_times_digit_sum_l3661_366187

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem no_numbers_equal_seven_times_digit_sum :
  ∀ n : ℕ, n > 0 ∧ n < 2000 → n ≠ 7 * (sum_of_digits n) :=
by
  sorry

end NUMINAMATH_CALUDE_no_numbers_equal_seven_times_digit_sum_l3661_366187


namespace NUMINAMATH_CALUDE_odd_guess_probability_l3661_366164

theorem odd_guess_probability (n : ℕ) (hn : n = 2002) :
  (n - n / 3 : ℚ) / n > 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_guess_probability_l3661_366164


namespace NUMINAMATH_CALUDE_formula_always_zero_l3661_366194

theorem formula_always_zero :
  ∃ (F : (ℝ → ℝ → ℝ) → (ℝ → ℝ → ℝ) → ℝ → ℝ), 
    ∀ (sub mul : ℝ → ℝ → ℝ) (a : ℝ), 
      (∀ x y, sub x y = x - y ∨ sub x y = x * y) →
      (∀ x y, mul x y = x * y ∨ mul x y = x - y) →
      F sub mul a = 0 :=
by sorry

end NUMINAMATH_CALUDE_formula_always_zero_l3661_366194


namespace NUMINAMATH_CALUDE_max_ratio_squared_l3661_366179

theorem max_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≥ b) :
  (∃ (ρ : ℝ), ∀ (x y : ℝ), 
    (0 ≤ x ∧ x < a) → 
    (0 ≤ y ∧ y < b) → 
    (a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a + x)^2 + (b - y)^2) →
    (a / b)^2 ≤ ρ^2 ∧
    ρ^2 = 4/3) :=
sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l3661_366179


namespace NUMINAMATH_CALUDE_binomial_8_5_l3661_366138

theorem binomial_8_5 : Nat.choose 8 5 = 56 := by sorry

end NUMINAMATH_CALUDE_binomial_8_5_l3661_366138


namespace NUMINAMATH_CALUDE_tinas_oranges_l3661_366197

/-- The number of oranges in Tina's bag -/
def oranges : ℕ := sorry

/-- The number of apples in Tina's bag -/
def apples : ℕ := 9

/-- The number of tangerines in Tina's bag -/
def tangerines : ℕ := 17

/-- The number of oranges removed -/
def oranges_removed : ℕ := 2

/-- The number of tangerines removed -/
def tangerines_removed : ℕ := 10

/-- Theorem stating that the number of oranges in Tina's bag is 5 -/
theorem tinas_oranges : oranges = 5 := by
  have h1 : tangerines - tangerines_removed = (oranges - oranges_removed) + 4 := by sorry
  sorry

end NUMINAMATH_CALUDE_tinas_oranges_l3661_366197


namespace NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l3661_366103

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The sum of the interior angles of a hexagon is 720 degrees -/
theorem hexagon_interior_angles_sum : 
  sum_interior_angles hexagon_sides = 720 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_interior_angles_sum_l3661_366103


namespace NUMINAMATH_CALUDE_min_area_APQB_l3661_366140

/-- Parabola Γ defined by y² = 8x -/
def Γ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- Focus of the parabola Γ -/
def F : ℝ × ℝ := (2, 0)

/-- Line l passing through F -/
def l (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = m * p.2 + 2}

/-- Points A and B are intersections of Γ and l -/
def A (m : ℝ) : ℝ × ℝ := sorry

def B (m : ℝ) : ℝ × ℝ := sorry

/-- Tangent line to Γ at point (x, y) -/
def tangentLine (x y : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 * y = 4 * (p.1 + x)}

/-- Point P is the intersection of tangent at A with y-axis -/
def P (m : ℝ) : ℝ := sorry

/-- Point Q is the intersection of tangent at B with y-axis -/
def Q (m : ℝ) : ℝ := sorry

/-- Area of quadrilateral APQB -/
def areaAPQB (m : ℝ) : ℝ := sorry

/-- The minimum area of quadrilateral APQB is 12 -/
theorem min_area_APQB : 
  ∀ m : ℝ, areaAPQB m ≥ 12 ∧ ∃ m₀ : ℝ, areaAPQB m₀ = 12 :=
sorry

end NUMINAMATH_CALUDE_min_area_APQB_l3661_366140


namespace NUMINAMATH_CALUDE_triangle_perimeter_in_divided_square_l3661_366108

/-- Given a square of side z divided into a smaller square of side w and four congruent triangles,
    the perimeter of one of these triangles is h + z, where h is the height of the triangle. -/
theorem triangle_perimeter_in_divided_square (z w h : ℝ) :
  z > 0 → w > 0 → h > 0 →
  h + (z - h) = z →  -- The height plus the base of the triangle equals the side of the larger square
  w^2 = h^2 + (z - h)^2 →  -- Pythagoras theorem for the triangle
  (h + z : ℝ) = 2 * h + (z - h) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_in_divided_square_l3661_366108


namespace NUMINAMATH_CALUDE_no_cracked_seashells_l3661_366146

theorem no_cracked_seashells (tim_shells sally_shells total_shells : ℕ) 
  (h1 : tim_shells = 37)
  (h2 : sally_shells = 13)
  (h3 : total_shells = 50)
  (h4 : tim_shells + sally_shells = total_shells) :
  total_shells - (tim_shells + sally_shells) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_cracked_seashells_l3661_366146


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l3661_366109

/-- Given a paint mixture ratio of 5:3:7 for yellow:blue:red,
    if 21 quarts of red paint are used, then 9 quarts of blue paint should be used. -/
theorem paint_mixture_ratio (yellow blue red : ℚ) (red_quarts : ℚ) :
  yellow = 5 →
  blue = 3 →
  red = 7 →
  red_quarts = 21 →
  (blue / red) * red_quarts = 9 := by
  sorry


end NUMINAMATH_CALUDE_paint_mixture_ratio_l3661_366109


namespace NUMINAMATH_CALUDE_decimal_to_binary_and_remainder_l3661_366100

def decimal_to_binary (n : ℕ) : List Bool :=
  sorry

def binary_to_decimal (b : List Bool) : ℕ :=
  sorry

def binary_division_remainder (dividend : List Bool) (divisor : List Bool) : List Bool :=
  sorry

theorem decimal_to_binary_and_remainder : 
  let binary_126 := decimal_to_binary 126
  let remainder := binary_division_remainder binary_126 [true, false, true, true]
  binary_126 = [true, true, true, true, true, true, false] ∧ 
  remainder = [true, false, false, true] :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_and_remainder_l3661_366100


namespace NUMINAMATH_CALUDE_expansion_contains_2017_l3661_366112

/-- The first term in the expansion of n^3 -/
def first_term (n : ℕ) : ℕ := n^2 - n + 1

/-- The last term in the expansion of n^3 -/
def last_term (n : ℕ) : ℕ := n^2 + n - 1

/-- The sum of n consecutive odd numbers starting from the first term -/
def sum_expansion (n : ℕ) : ℕ := n * (first_term n + last_term n) / 2

theorem expansion_contains_2017 :
  ∃ (n : ℕ), n = 45 ∧ 
  first_term n ≤ 2017 ∧ 
  2017 ≤ last_term n ∧ 
  sum_expansion n = n^3 :=
sorry

end NUMINAMATH_CALUDE_expansion_contains_2017_l3661_366112


namespace NUMINAMATH_CALUDE_closed_broken_line_existence_l3661_366145

/-- A closed broken line consisting of six segments with sides a, b, c, d, e, f (in that order)
    and opposite sides perpendicular to each other exists if and only if √(a² + d²), √(b² + e²),
    and √(c² + f²) satisfy the triangle inequality. -/
theorem closed_broken_line_existence (a b c d e f : ℝ) :
  (∃ (A B C D E F : ℝ × ℝ),
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = a^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = b^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = c^2 ∧
    (D.1 - E.1)^2 + (D.2 - E.2)^2 = d^2 ∧
    (E.1 - F.1)^2 + (E.2 - F.2)^2 = e^2 ∧
    (F.1 - A.1)^2 + (F.2 - A.2)^2 = f^2 ∧
    (A.1 - B.1) * (D.1 - E.1) + (A.2 - B.2) * (D.2 - E.2) = 0 ∧
    (B.1 - C.1) * (E.1 - F.1) + (B.2 - C.2) * (E.2 - F.2) = 0 ∧
    (C.1 - D.1) * (F.1 - A.1) + (C.2 - D.2) * (F.2 - A.2) = 0) ↔
  (Real.sqrt (a^2 + d^2) ≤ Real.sqrt (b^2 + e^2) + Real.sqrt (c^2 + f^2) ∧
   Real.sqrt (b^2 + e^2) ≤ Real.sqrt (a^2 + d^2) + Real.sqrt (c^2 + f^2) ∧
   Real.sqrt (c^2 + f^2) ≤ Real.sqrt (a^2 + d^2) + Real.sqrt (b^2 + e^2)) :=
by sorry


end NUMINAMATH_CALUDE_closed_broken_line_existence_l3661_366145


namespace NUMINAMATH_CALUDE_remaining_squares_l3661_366192

/-- A chocolate bar with rectangular shape -/
structure ChocolateBar where
  length : ℕ
  width : ℕ
  total_squares : ℕ
  h_width : width = 6
  h_length : length ≥ 9
  h_total : total_squares = length * width

/-- The number of squares removed by Irena and Jack -/
def squares_removed : ℕ := 12 + 9

/-- The theorem stating the number of remaining squares -/
theorem remaining_squares (bar : ChocolateBar) : 
  bar.total_squares - squares_removed = 45 := by
  sorry

#check remaining_squares

end NUMINAMATH_CALUDE_remaining_squares_l3661_366192


namespace NUMINAMATH_CALUDE_set_M_characterization_inequality_holds_complement_M_l3661_366167

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Define the set M
def M : Set ℝ := {x | f x > 2}

-- Theorem 1
theorem set_M_characterization : M = {x : ℝ | x < (1/2) ∨ x > (5/2)} := by sorry

-- Theorem 2
theorem inequality_holds_complement_M (a b x : ℝ) (ha : a ≠ 0) (hx : (1/2) ≤ x ∧ x ≤ (5/2)) :
  |a + b| + |a - b| ≥ |a| * (f x) := by sorry

end NUMINAMATH_CALUDE_set_M_characterization_inequality_holds_complement_M_l3661_366167


namespace NUMINAMATH_CALUDE_herd_division_l3661_366125

theorem herd_division (total : ℕ) (fourth_son : ℕ) : 
  (total : ℚ) / 3 + total / 5 + total / 6 + fourth_son = total ∧ 
  fourth_son = 19 → 
  total = 63 := by
sorry

end NUMINAMATH_CALUDE_herd_division_l3661_366125


namespace NUMINAMATH_CALUDE_square_tens_seven_units_six_l3661_366166

theorem square_tens_seven_units_six (n : ℤ) : 
  (n^2 % 100 ≥ 70) ∧ (n^2 % 100 < 80) → n^2 % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_tens_seven_units_six_l3661_366166


namespace NUMINAMATH_CALUDE_equation_solutions_l3661_366196

theorem equation_solutions :
  (∃ x : ℝ, 3 * x + 7 = 32 - 2 * x ∧ x = 5) ∧
  (∃ x : ℝ, (2 * x - 3) / 5 = (3 * x - 1) / 2 + 1 ∧ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3661_366196


namespace NUMINAMATH_CALUDE_max_min_ratio_l3661_366137

/-- The curve on which point P moves --/
def curve (x y : ℝ) : Prop := y = 3 * Real.sqrt (1 - x^2 / 4)

/-- The expression we're maximizing and minimizing --/
def expr (x y : ℝ) : ℝ := 2 * x - y

/-- Theorem stating the ratio of max to min values of the expression --/
theorem max_min_ratio :
  ∃ (max min : ℝ),
    (∀ x y : ℝ, curve x y → expr x y ≤ max) ∧
    (∃ x y : ℝ, curve x y ∧ expr x y = max) ∧
    (∀ x y : ℝ, curve x y → expr x y ≥ min) ∧
    (∃ x y : ℝ, curve x y ∧ expr x y = min) ∧
    max / min = -4 / 5 :=
by sorry

end NUMINAMATH_CALUDE_max_min_ratio_l3661_366137


namespace NUMINAMATH_CALUDE_two_greater_than_sqrt_three_l3661_366114

theorem two_greater_than_sqrt_three : 2 > Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_two_greater_than_sqrt_three_l3661_366114


namespace NUMINAMATH_CALUDE_line_increase_percentage_l3661_366107

/-- Given an increase of 450 lines resulting in a total of 1350 lines, 
    prove that the percentage increase is 50%. -/
theorem line_increase_percentage (increase : ℕ) (total : ℕ) : 
  increase = 450 → total = 1350 → (increase : ℚ) / ((total - increase) : ℚ) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_line_increase_percentage_l3661_366107


namespace NUMINAMATH_CALUDE_sara_cannot_have_two_l3661_366153

-- Define the set of cards
def Cards : Finset ℕ := {1, 2, 3, 4}

-- Define the players
inductive Player
| Ben
| Wendy
| Riley
| Sara

-- Define the distribution of cards
def Distribution := Player → ℕ

-- Define the conditions
def ValidDistribution (d : Distribution) : Prop :=
  (∀ p : Player, d p ∈ Cards) ∧
  (∀ p q : Player, p ≠ q → d p ≠ d q) ∧
  (d Player.Ben ≠ 1) ∧
  (d Player.Wendy = d Player.Riley + 1)

-- Theorem statement
theorem sara_cannot_have_two (d : Distribution) :
  ValidDistribution d → d Player.Sara ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_sara_cannot_have_two_l3661_366153


namespace NUMINAMATH_CALUDE_tyler_age_l3661_366142

/-- Represents the ages of Tyler and Clay -/
structure Ages where
  tyler : ℕ
  clay : ℕ

/-- The conditions of the problem -/
def validAges (ages : Ages) : Prop :=
  ages.tyler = 3 * ages.clay + 1 ∧ ages.tyler + ages.clay = 21

/-- The theorem to prove -/
theorem tyler_age (ages : Ages) (h : validAges ages) : ages.tyler = 16 := by
  sorry

end NUMINAMATH_CALUDE_tyler_age_l3661_366142


namespace NUMINAMATH_CALUDE_combined_cost_increase_percentage_l3661_366115

def bicycle_initial_cost : ℝ := 200
def skates_initial_cost : ℝ := 50
def bicycle_increase_rate : ℝ := 0.06
def skates_increase_rate : ℝ := 0.15

theorem combined_cost_increase_percentage :
  let bicycle_new_cost := bicycle_initial_cost * (1 + bicycle_increase_rate)
  let skates_new_cost := skates_initial_cost * (1 + skates_increase_rate)
  let initial_total_cost := bicycle_initial_cost + skates_initial_cost
  let new_total_cost := bicycle_new_cost + skates_new_cost
  (new_total_cost - initial_total_cost) / initial_total_cost = 0.078 := by
  sorry

end NUMINAMATH_CALUDE_combined_cost_increase_percentage_l3661_366115


namespace NUMINAMATH_CALUDE_relatively_prime_squares_l3661_366126

theorem relatively_prime_squares (a b c : ℤ) 
  (h_coprime : ∀ d : ℤ, d ∣ a ∧ d ∣ b ∧ d ∣ c → d = 1 ∨ d = -1)
  (h_eq : 1 / a + 1 / b = 1 / c) :
  ∃ (p q r : ℤ), (a + b = p^2) ∧ (a - c = q^2) ∧ (b - c = r^2) := by
  sorry

end NUMINAMATH_CALUDE_relatively_prime_squares_l3661_366126


namespace NUMINAMATH_CALUDE_dress_discount_price_l3661_366188

theorem dress_discount_price (d : ℝ) (h : d > 0) : 
  d * (1 - 0.45) * (1 - 0.4) = d * 0.33 := by
sorry

end NUMINAMATH_CALUDE_dress_discount_price_l3661_366188


namespace NUMINAMATH_CALUDE_evaluate_expression_l3661_366130

theorem evaluate_expression : 4^4 - 4 * 4^3 + 6 * 4^2 - 4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3661_366130


namespace NUMINAMATH_CALUDE_equation_solution_l3661_366133

open Real

theorem equation_solution (x : ℝ) :
  (sin x ≠ 0) →
  (cos x ≠ 0) →
  (sin x + cos x ≥ 0) →
  (Real.sqrt (1 + tan x) = sin x + cos x) ↔
  (∃ n : ℤ, (x = π/4 + 2*π*↑n) ∨ (x = -π/4 + 2*π*↑n) ∨ (x = 3*π/4 + 2*π*↑n)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3661_366133


namespace NUMINAMATH_CALUDE_loan_payment_period_l3661_366104

theorem loan_payment_period
  (house_cost : ℕ)
  (trailer_cost : ℕ)
  (monthly_payment_diff : ℕ)
  (h1 : house_cost = 480000)
  (h2 : trailer_cost = 120000)
  (h3 : monthly_payment_diff = 1500)
  : (house_cost - trailer_cost) / monthly_payment_diff / 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_loan_payment_period_l3661_366104


namespace NUMINAMATH_CALUDE_vector_problem_l3661_366122

theorem vector_problem (α : Real) 
  (h1 : α ∈ Set.Ioo (3*π/2) (2*π))
  (h2 : (3*Real.sin α)*(2*Real.sin α) + (Real.cos α)*(5*Real.sin α - 4*Real.cos α) = 0) :
  Real.tan α = -4/3 ∧ Real.cos (α/2 + π/3) = -(2*Real.sqrt 5 + Real.sqrt 15)/10 := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3661_366122


namespace NUMINAMATH_CALUDE_sqrt_five_power_calculation_l3661_366168

theorem sqrt_five_power_calculation : (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 78125 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_power_calculation_l3661_366168


namespace NUMINAMATH_CALUDE_function_symmetry_and_translation_l3661_366105

-- Define a function type
def RealFunction := ℝ → ℝ

-- Define the translation operation
def translate (f : RealFunction) (h : ℝ) : RealFunction :=
  λ x => f (x + h)

-- Define symmetry with respect to y-axis
def symmetricToYAxis (f g : RealFunction) : Prop :=
  ∀ x, f x = g (-x)

-- State the theorem
theorem function_symmetry_and_translation (f : RealFunction) :
  (symmetricToYAxis (translate f 1) (λ x => 2^x)) →
  (f = λ x => (1/2)^(x-1)) := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_and_translation_l3661_366105


namespace NUMINAMATH_CALUDE_percentage_difference_l3661_366150

theorem percentage_difference (A B y : ℝ) : 
  A > 0 → B > A → B = A * (1 + y / 100) → y = 100 * (B - A) / A := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3661_366150


namespace NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3661_366113

def f (x : ℝ) : ℝ := x^5 - 6*x^4 + 11*x^3 + 21*x^2 - 17*x + 10

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := by sorry

theorem polynomial_remainder : 
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 2) * q x + 84 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_polynomial_remainder_l3661_366113


namespace NUMINAMATH_CALUDE_positive_number_problem_l3661_366134

theorem positive_number_problem : ∃ (n : ℕ), n > 0 ∧ 3 * n + n^2 = 300 ∧ n = 16 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_problem_l3661_366134


namespace NUMINAMATH_CALUDE_multiplicative_inverse_484_mod_1123_l3661_366124

theorem multiplicative_inverse_484_mod_1123 :
  ∃ (n : ℤ), 0 ≤ n ∧ n < 1123 ∧ (484 * n) % 1123 = 1 :=
by
  use 535
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_484_mod_1123_l3661_366124


namespace NUMINAMATH_CALUDE_set_operations_and_subsets_l3661_366147

def U : Finset ℕ := {4, 5, 6, 7, 8, 9, 10, 11, 12}
def A : Finset ℕ := {6, 8, 10, 12}
def B : Finset ℕ := {1, 6, 8}

theorem set_operations_and_subsets :
  (A ∪ B = {1, 6, 8, 10, 12}) ∧
  (U \ A = {4, 5, 7, 9, 11}) ∧
  (Finset.powerset (A ∩ B)).card = 4 := by sorry

end NUMINAMATH_CALUDE_set_operations_and_subsets_l3661_366147


namespace NUMINAMATH_CALUDE_cookout_attendance_l3661_366119

theorem cookout_attendance (kids_2004 kids_2005 kids_2006 : ℕ) : 
  kids_2005 = kids_2004 / 2 →
  kids_2006 = (2 * kids_2005) / 3 →
  kids_2006 = 20 →
  kids_2004 = 60 := by
sorry

end NUMINAMATH_CALUDE_cookout_attendance_l3661_366119


namespace NUMINAMATH_CALUDE_division_theorem_l3661_366184

theorem division_theorem (dividend divisor remainder quotient : ℕ) : 
  dividend = divisor * quotient + remainder →
  dividend = 167 →
  divisor = 18 →
  remainder = 5 →
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l3661_366184


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3661_366193

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Add necessary conditions for a valid triangle
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  -- Add cosine law
  cos_law_c : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop := t.b = t.c ∨ t.a = t.c ∨ t.a = t.b

-- State the theorem
theorem sufficient_not_necessary_condition (t : Triangle) :
  (t.a = 2 * t.b * Real.cos t.C → isIsosceles t) ∧
  ∃ t' : Triangle, isIsosceles t' ∧ t'.a ≠ 2 * t'.b * Real.cos t'.C :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3661_366193


namespace NUMINAMATH_CALUDE_inequality_preserved_division_l3661_366148

theorem inequality_preserved_division (x y a : ℝ) (h : x > y) :
  x / (a^2 + 1) > y / (a^2 + 1) := by sorry

end NUMINAMATH_CALUDE_inequality_preserved_division_l3661_366148


namespace NUMINAMATH_CALUDE_unique_prime_solution_l3661_366154

theorem unique_prime_solution :
  ∀ p m : ℕ,
    p.Prime →
    m > 0 →
    p * (p + m) + p = (m + 1)^3 →
    p = 2 ∧ m = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l3661_366154


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3661_366169

theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25) (h4 : w = 7) : 
  let triangle_area := (1/2) * a * b
  let rectangle_length := triangle_area / w
  2 * (rectangle_length + w) = 38 := by
sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3661_366169


namespace NUMINAMATH_CALUDE_even_numbers_set_builder_notation_l3661_366170

-- Define the set of even numbers
def EvenNumbers : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}

-- State the theorem
theorem even_numbers_set_builder_notation : 
  EvenNumbers = {x : ℤ | ∃ n : ℤ, x = 2 * n} := by sorry

end NUMINAMATH_CALUDE_even_numbers_set_builder_notation_l3661_366170


namespace NUMINAMATH_CALUDE_equation_solutions_l3661_366152

theorem equation_solutions :
  (∀ x : ℝ, 4 * (x - 1)^2 = 100 ↔ x = 6 ∨ x = -4) ∧
  (∀ x : ℝ, (2*x - 1)^3 = -8 ↔ x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3661_366152


namespace NUMINAMATH_CALUDE_perpendicular_lines_slope_l3661_366183

theorem perpendicular_lines_slope (a : ℝ) : 
  (∃ (x y : ℝ), y = a * x - 2) ∧ 
  (∃ (x y : ℝ), y = (a + 2) * x + 1) ∧ 
  (a * (a + 2) = -1) → 
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_slope_l3661_366183


namespace NUMINAMATH_CALUDE_john_calorie_increase_l3661_366190

/-- Represents the daily calorie intake of John --/
structure DailyCalories where
  breakfast : ℕ
  lunch_percentage : ℕ
  shakes : ℕ
  total : ℕ

/-- Calculates the lunch calories based on breakfast and percentage increase --/
def lunch_calories (d : DailyCalories) : ℕ :=
  d.breakfast * (100 + d.lunch_percentage) / 100

/-- Calculates the dinner calories based on lunch calories --/
def dinner_calories (d : DailyCalories) : ℕ :=
  2 * lunch_calories d

/-- Theorem stating that given John's eating habits, the percentage increase from breakfast to lunch is 125% --/
theorem john_calorie_increase (d : DailyCalories) 
  (h1 : d.breakfast = 500)
  (h2 : d.shakes = 900)
  (h3 : d.total = 3275)
  (h4 : d.breakfast + lunch_calories d + dinner_calories d + d.shakes = d.total) :
  d.lunch_percentage = 125 := by
  sorry

end NUMINAMATH_CALUDE_john_calorie_increase_l3661_366190


namespace NUMINAMATH_CALUDE_circle_center_sum_l3661_366131

theorem circle_center_sum (x y : ℝ) : 
  (∀ a b : ℝ, (a - x)^2 + (b - y)^2 = (a^2 + b^2 - 12*a + 4*b - 10)) → 
  x + y = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l3661_366131


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3661_366159

/-- The eccentricity of an ellipse with equation x²/3 + y²/9 = 1 is √6/3 -/
theorem ellipse_eccentricity : 
  let a : ℝ := 3
  let b : ℝ := Real.sqrt 3
  let e : ℝ := Real.sqrt (a^2 - b^2) / a
  e = Real.sqrt 6 / 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3661_366159


namespace NUMINAMATH_CALUDE_joan_seashells_l3661_366128

/-- The number of seashells Joan has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Joan has 16 seashells after giving away 63 from her initial 79 -/
theorem joan_seashells : remaining_seashells 79 63 = 16 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l3661_366128


namespace NUMINAMATH_CALUDE_vendor_drink_problem_l3661_366161

theorem vendor_drink_problem (maaza sprite cans : ℕ) (pepsi : ℕ) : 
  maaza = 50 →
  sprite = 368 →
  cans = 281 →
  (maaza + sprite + pepsi) % cans = 0 →
  pepsi = 144 :=
by sorry

end NUMINAMATH_CALUDE_vendor_drink_problem_l3661_366161


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3661_366158

theorem parabola_line_intersection (a k b x₁ x₂ x₃ : ℝ) 
  (ha : a > 0)
  (h₁ : a * x₁^2 = k * x₁ + b)
  (h₂ : a * x₂^2 = k * x₂ + b)
  (h₃ : 0 = k * x₃ + b) :
  x₁ * x₂ = x₂ * x₃ + x₁ * x₃ := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3661_366158


namespace NUMINAMATH_CALUDE_sin_derivative_bound_and_inequality_range_l3661_366177

noncomputable def f (x : ℝ) : ℝ := Real.sin x

theorem sin_derivative_bound_and_inequality_range :
  (∀ x > 0, (deriv f) x > 1 - x^2 / 2) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Ioo 0 (Real.pi / 2), f x + f x / (deriv f) x > a * x) → a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_derivative_bound_and_inequality_range_l3661_366177


namespace NUMINAMATH_CALUDE_power_of_power_l3661_366165

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3661_366165
