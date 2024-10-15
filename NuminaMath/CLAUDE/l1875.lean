import Mathlib

namespace NUMINAMATH_CALUDE_m_neg_one_necessary_not_sufficient_l1875_187511

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := m * x + (2 * m - 1) * y + 1 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := 3 * x + m * y + 3 = 0

-- Define perpendicularity of two lines
def perpendicular (m : ℝ) : Prop := ∃ (k₁ k₂ : ℝ), k₁ * k₂ = -1 ∧
  (∀ (x y : ℝ), l₁ m x y → m * x + (2 * m - 1) * y = k₁) ∧
  (∀ (x y : ℝ), l₂ m x y → 3 * x + m * y = k₂)

-- State the theorem
theorem m_neg_one_necessary_not_sufficient :
  (∀ m : ℝ, m = -1 → perpendicular m) ∧
  ¬(∀ m : ℝ, perpendicular m → m = -1) :=
sorry

end NUMINAMATH_CALUDE_m_neg_one_necessary_not_sufficient_l1875_187511


namespace NUMINAMATH_CALUDE_min_apples_in_basket_l1875_187525

theorem min_apples_in_basket (N : ℕ) : 
  N ≥ 67 ∧ 
  N % 3 = 1 ∧ 
  N % 4 = 3 ∧ 
  N % 5 = 2 ∧
  (∀ m : ℕ, m < N → ¬(m % 3 = 1 ∧ m % 4 = 3 ∧ m % 5 = 2)) := by
  sorry

end NUMINAMATH_CALUDE_min_apples_in_basket_l1875_187525


namespace NUMINAMATH_CALUDE_traffic_light_statement_correct_l1875_187594

/-- A traffic light state can be either red or green -/
inductive TrafficLightState
  | Red
  | Green

/-- A traffic light intersection scenario -/
structure TrafficLightIntersection where
  state : TrafficLightState

/-- The statement about traffic light outcomes is correct -/
theorem traffic_light_statement_correct :
  ∀ (intersection : TrafficLightIntersection),
    (intersection.state = TrafficLightState.Red) ∨
    (intersection.state = TrafficLightState.Green) :=
by sorry

end NUMINAMATH_CALUDE_traffic_light_statement_correct_l1875_187594


namespace NUMINAMATH_CALUDE_video_game_sales_earnings_l1875_187565

theorem video_game_sales_earnings 
  (total_games : ℕ) 
  (non_working_games : ℕ) 
  (price_per_game : ℕ) : 
  total_games = 16 → 
  non_working_games = 8 → 
  price_per_game = 7 → 
  (total_games - non_working_games) * price_per_game = 56 := by
sorry

end NUMINAMATH_CALUDE_video_game_sales_earnings_l1875_187565


namespace NUMINAMATH_CALUDE_unique_m_value_l1875_187598

theorem unique_m_value (m : ℝ) : 
  let A : Set ℝ := {0, m, m^2 - 3*m + 2}
  2 ∈ A → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_value_l1875_187598


namespace NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l1875_187551

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : Prop := x^2 - 8*x + m = 0

-- Define an isosceles triangle
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

-- Define the triangle inequality
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

-- Main theorem
theorem isosceles_triangle_quadratic_roots (m : ℝ) : 
  (∃ x y : ℝ, 
    quadratic_equation x m ∧ 
    quadratic_equation y m ∧ 
    x ≠ y ∧
    is_isosceles_triangle 6 x y ∧
    satisfies_triangle_inequality 6 x y) ↔ 
  (m = 12 ∨ m = 16) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l1875_187551


namespace NUMINAMATH_CALUDE_divisors_of_72_l1875_187561

theorem divisors_of_72 : Finset.card ((Finset.range 73).filter (λ x => 72 % x = 0)) * 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_72_l1875_187561


namespace NUMINAMATH_CALUDE_quarter_circles_sum_approaches_circumference_l1875_187584

/-- The sum of quarter-circle arc lengths approaches the original circle's circumference as n approaches infinity -/
theorem quarter_circles_sum_approaches_circumference (R : ℝ) (h : R > 0) :
  let C := 2 * Real.pi * R
  let quarter_circle_sum (n : ℕ) := 2 * n * (Real.pi * C) / (2 * n)
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |quarter_circle_sum n - C| < ε :=
by sorry

end NUMINAMATH_CALUDE_quarter_circles_sum_approaches_circumference_l1875_187584


namespace NUMINAMATH_CALUDE_min_value_problem_l1875_187572

theorem min_value_problem (a b m n : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hn : 0 < n)
  (hab : a + b = 1) (hmn : m * n = 2) :
  (a * m + b * n) * (b * m + a * n) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l1875_187572


namespace NUMINAMATH_CALUDE_gcd_xyz_square_l1875_187583

theorem gcd_xyz_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, Nat.gcd x (Nat.gcd y z) * x * y * z = k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_gcd_xyz_square_l1875_187583


namespace NUMINAMATH_CALUDE_smallest_four_digit_mod_8_l1875_187516

theorem smallest_four_digit_mod_8 :
  ∀ n : ℕ, n ≥ 1000 ∧ n ≡ 5 [MOD 8] → n ≥ 1005 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_mod_8_l1875_187516


namespace NUMINAMATH_CALUDE_tray_height_l1875_187536

theorem tray_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) : 
  side_length = 120 →
  cut_distance = Real.sqrt 20 →
  cut_angle = π / 4 →
  ∃ (height : ℝ), height = Real.sqrt 10 ∧ 
    height = (cut_distance * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_tray_height_l1875_187536


namespace NUMINAMATH_CALUDE_sqrt_245_simplification_l1875_187505

theorem sqrt_245_simplification : Real.sqrt 245 = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_245_simplification_l1875_187505


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l1875_187566

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes (m₁ m₂ : ℝ) : 
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, (y = m₁ * x + b₁ ↔ y = m₂ * x + b₂)) ↔ m₁ = m₂

/-- The problem statement -/
theorem parallel_lines_k_value :
  (∃ b₁ b₂ : ℝ, ∀ x y : ℝ, (y = 5 * x - 3 ↔ y = 3 * k * x + 7)) → k = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l1875_187566


namespace NUMINAMATH_CALUDE_marble_distribution_l1875_187504

def is_valid_combination (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧ e > 1 ∧
  a + e ≥ 11 ∧
  c + a < 11 ∧
  b + c ≥ 11 ∧
  c + d ≥ 11 ∧
  a + b + c + d + e = 26

theorem marble_distribution :
  ∀ a b c d e : ℕ,
  is_valid_combination a b c d e ↔
  ((a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 9 ∧ e = 11) ∨
   (a = 1 ∧ b = 2 ∧ c = 4 ∧ d = 9 ∧ e = 10) ∨
   (a = 1 ∧ b = 3 ∧ c = 4 ∧ d = 8 ∧ e = 10) ∨
   (a = 2 ∧ b = 3 ∧ c = 4 ∧ d = 8 ∧ e = 9)) :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_l1875_187504


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l1875_187527

theorem range_of_a_minus_b (a b : ℝ) (ha : -1 < a ∧ a < 2) (hb : -2 < b ∧ b < 1) :
  ∃ x, -2 < x ∧ x < 4 ∧ ∃ a' b', -1 < a' ∧ a' < 2 ∧ -2 < b' ∧ b' < 1 ∧ x = a' - b' :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l1875_187527


namespace NUMINAMATH_CALUDE_tax_rate_ratio_l1875_187540

theorem tax_rate_ratio (mork_rate mindy_rate combined_rate : ℚ) 
  (h1 : mork_rate = 45/100)
  (h2 : mindy_rate = 15/100)
  (h3 : combined_rate = 21/100) :
  ∃ (m k : ℚ), m > 0 ∧ k > 0 ∧ 
    mindy_rate * m + mork_rate * k = combined_rate * (m + k) ∧
    m / k = 4 := by
  sorry

end NUMINAMATH_CALUDE_tax_rate_ratio_l1875_187540


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l1875_187529

theorem power_mod_seventeen : 7^2048 % 17 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l1875_187529


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1875_187548

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x - 48 = 0 → 
  (x = 16 ∨ x = -3) → 
  ∃ y : ℝ, y^2 - 13*y - 48 = 0 ∧ y ≠ x ∧ x ≤ y → x = 16 :=
by sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1875_187548


namespace NUMINAMATH_CALUDE_right_triangle_sets_set_a_not_right_triangle_l1875_187502

/-- A function that checks if three numbers can form a right triangle -/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that set A cannot form a right triangle while others can -/
theorem right_triangle_sets :
  ¬(is_right_triangle (Real.sqrt 3) (Real.sqrt 4) (Real.sqrt 5)) ∧
  (is_right_triangle 1 (Real.sqrt 2) (Real.sqrt 3)) ∧
  (is_right_triangle 6 8 10) ∧
  (is_right_triangle 3 4 5) := by
  sorry

/-- The specific theorem for set A -/
theorem set_a_not_right_triangle :
  ¬(is_right_triangle (Real.sqrt 3) (Real.sqrt 4) (Real.sqrt 5)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sets_set_a_not_right_triangle_l1875_187502


namespace NUMINAMATH_CALUDE_cuboid_volume_l1875_187576

/-- The volume of a cuboid with edges 2, 5, and 8 is 80 -/
theorem cuboid_volume : 
  let edge1 : ℝ := 2
  let edge2 : ℝ := 5
  let edge3 : ℝ := 8
  edge1 * edge2 * edge3 = 80 := by sorry

end NUMINAMATH_CALUDE_cuboid_volume_l1875_187576


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1875_187569

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The conditions of the problem -/
structure ProblemConditions (a : ℕ → ℝ) : Prop :=
  (geom_seq : geometric_sequence a)
  (sum_cond : a 4 + a 7 = 2)
  (prod_cond : a 2 * a 9 = -8)

/-- The theorem to prove -/
theorem geometric_sequence_sum 
  (a : ℕ → ℝ) 
  (h : ProblemConditions a) : 
  a 1 + a 10 = -7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1875_187569


namespace NUMINAMATH_CALUDE_room_width_calculation_l1875_187596

/-- Proves that given a rectangular room with length 5.5 meters, if the total cost of paving the floor at a rate of 1200 Rs per square meter is 24750 Rs, then the width of the room is 3.75 meters. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (width : ℝ) : 
  length = 5.5 → 
  cost_per_sqm = 1200 → 
  total_cost = 24750 → 
  width = total_cost / cost_per_sqm / length → 
  width = 3.75 := by
sorry


end NUMINAMATH_CALUDE_room_width_calculation_l1875_187596


namespace NUMINAMATH_CALUDE_sum_smallest_largest_fourdigit_l1875_187560

/-- A function that generates all four-digit numbers using the digits 0, 3, 4, and 8 -/
def fourDigitNumbers : List Nat := sorry

/-- The smallest four-digit number formed using 0, 3, 4, and 8 -/
def smallestNumber : Nat := sorry

/-- The largest four-digit number formed using 0, 3, 4, and 8 -/
def largestNumber : Nat := sorry

/-- Theorem stating that the sum of the smallest and largest four-digit numbers
    formed using 0, 3, 4, and 8 is 11478 -/
theorem sum_smallest_largest_fourdigit :
  smallestNumber + largestNumber = 11478 := by sorry

end NUMINAMATH_CALUDE_sum_smallest_largest_fourdigit_l1875_187560


namespace NUMINAMATH_CALUDE_max_value_zero_l1875_187579

theorem max_value_zero (a : Real) (h1 : 0 ≤ a) (h2 : a ≤ 1) :
  (∀ x : Real, x ≤ Real.sqrt (a * 1 * 0) + Real.sqrt ((1 - a) * (1 - 1) * (1 - 0))) →
  (Real.sqrt (a * 1 * 0) + Real.sqrt ((1 - a) * (1 - 1) * (1 - 0))) = 0 :=
by sorry

end NUMINAMATH_CALUDE_max_value_zero_l1875_187579


namespace NUMINAMATH_CALUDE_complex_number_location_l1875_187506

theorem complex_number_location :
  let z : ℂ := (1 + Complex.I) * (1 - 2 * Complex.I)
  (z.re > 0 ∧ z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l1875_187506


namespace NUMINAMATH_CALUDE_rectangle_properties_l1875_187534

/-- Rectangle with adjacent sides x and 4, and perimeter y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- The perimeter of the rectangle is related to x -/
axiom perimeter_relation (rect : Rectangle) : rect.y = 2 * rect.x + 8

theorem rectangle_properties :
  ∀ (rect : Rectangle),
  (rect.x = 10 → rect.y = 28) ∧
  (rect.y = 30 → rect.x = 11) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_properties_l1875_187534


namespace NUMINAMATH_CALUDE_ann_has_eight_bags_l1875_187523

/-- Represents the number of apples in one of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- Represents the total number of apples Pam has -/
def pams_total_apples : ℕ := 1800

/-- Represents the total number of apples Ann has -/
def anns_total_apples : ℕ := 1800

/-- Represents the number of apples in one of Pam's bags -/
def pams_bag_count : ℕ := 3 * geralds_bag_count

/-- Represents the number of apples in one of Ann's bags -/
def anns_bag_count : ℕ := 2 * pams_bag_count

/-- Theorem stating that Ann has 8 bags of apples -/
theorem ann_has_eight_bags : 
  anns_total_apples / anns_bag_count = 8 ∧ 
  anns_total_apples % anns_bag_count = 0 :=
by sorry

end NUMINAMATH_CALUDE_ann_has_eight_bags_l1875_187523


namespace NUMINAMATH_CALUDE_right_triangles_shared_hypotenuse_l1875_187542

/-- Given two right triangles ABC and ABD sharing hypotenuse AB, 
    where BC = 3, AC = a, and AD = 4, prove that BD = √(a² - 7) -/
theorem right_triangles_shared_hypotenuse 
  (a : ℝ) 
  (h : a ≥ Real.sqrt 7) : 
  ∃ (AB BC AC AD BD : ℝ),
    BC = 3 ∧ 
    AC = a ∧ 
    AD = 4 ∧
    AB ^ 2 = AC ^ 2 + BC ^ 2 ∧ 
    AB ^ 2 = AD ^ 2 + BD ^ 2 ∧
    BD = Real.sqrt (a ^ 2 - 7) := by
  sorry


end NUMINAMATH_CALUDE_right_triangles_shared_hypotenuse_l1875_187542


namespace NUMINAMATH_CALUDE_sports_conference_games_l1875_187538

/-- Calculates the total number of games in a sports conference season -/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  (total_teams * (intra_division_games * (teams_per_division - 1) + 
  inter_division_games * teams_per_division)) / 2

/-- Theorem stating the total number of games in the given sports conference -/
theorem sports_conference_games : 
  total_games 16 8 2 1 = 176 := by
  sorry

end NUMINAMATH_CALUDE_sports_conference_games_l1875_187538


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l1875_187573

/-- The number of different types of ice cream cones available. -/
def num_cone_types : ℕ := 2

/-- The number of different ice cream flavors available. -/
def num_flavors : ℕ := 4

/-- The total number of different ways to order ice cream. -/
def total_combinations : ℕ := num_cone_types * num_flavors

/-- Theorem stating that the total number of different ways to order ice cream is 8. -/
theorem ice_cream_combinations : total_combinations = 8 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l1875_187573


namespace NUMINAMATH_CALUDE_job_completion_time_l1875_187562

/-- Represents the workforce and time required to complete a job -/
structure JobInfo where
  initialWorkforce : ℕ
  initialDays : ℕ
  extraWorkers : ℕ
  joinInterval : ℕ

/-- Calculates the total time required to complete the job given the job information -/
def calculateTotalTime (job : JobInfo) : ℕ :=
  sorry

/-- Theorem stating that for the given job information, the total time is 12 days -/
theorem job_completion_time (job : JobInfo) 
  (h1 : job.initialWorkforce = 20)
  (h2 : job.initialDays = 15)
  (h3 : job.extraWorkers = 10)
  (h4 : job.joinInterval = 5) : 
  calculateTotalTime job = 12 :=
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l1875_187562


namespace NUMINAMATH_CALUDE_honey_harvest_increase_l1875_187530

def last_year_harvest : ℕ := 2479
def this_year_harvest : ℕ := 8564

theorem honey_harvest_increase :
  this_year_harvest - last_year_harvest = 6085 :=
by sorry

end NUMINAMATH_CALUDE_honey_harvest_increase_l1875_187530


namespace NUMINAMATH_CALUDE_x_value_from_ratios_l1875_187546

theorem x_value_from_ratios (x y z a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : x * y / (x + y) = a)
  (h2 : x * z / (x + z) = b)
  (h3 : y * z / (y + z) = c) :
  x = 2 * a * b * c / (a * c + b * c - a * b) := by
sorry

end NUMINAMATH_CALUDE_x_value_from_ratios_l1875_187546


namespace NUMINAMATH_CALUDE_smallest_integer_above_root_sum_power_l1875_187599

theorem smallest_integer_above_root_sum_power :
  ∃ n : ℕ, (n = 3323 ∧ (∀ m : ℕ, m < n → m ≤ (Real.sqrt 5 + Real.sqrt 3)^6) ∧
            (∀ k : ℕ, k > (Real.sqrt 5 + Real.sqrt 3)^6 → k ≥ n)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_above_root_sum_power_l1875_187599


namespace NUMINAMATH_CALUDE_odd_prime_square_root_l1875_187539

theorem odd_prime_square_root (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (k : ℕ), k > 0 ∧ ∃ (n : ℕ), n > 0 ∧ k - p * k = n^2 ∧ k = (p + 1)^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_square_root_l1875_187539


namespace NUMINAMATH_CALUDE_sum_of_roots_l1875_187588

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x : ℝ, x^2 - 8*a*x - 9*b = 0 ↔ (x = c ∨ x = d)) →
  (∀ x : ℝ, x^2 - 8*c*x - 9*d = 0 ↔ (x = a ∨ x = b)) →
  a + b + c + d = 648 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1875_187588


namespace NUMINAMATH_CALUDE_pool_volume_l1875_187520

/-- The volume of a circular pool with linearly varying depth -/
theorem pool_volume (diameter : ℝ) (min_depth max_depth : ℝ) :
  diameter = 20 →
  min_depth = 3 →
  max_depth = 6 →
  let radius := diameter / 2
  let avg_depth := (min_depth + max_depth) / 2
  let volume := π * radius^2 * avg_depth
  volume = 450 * π := by
  sorry

end NUMINAMATH_CALUDE_pool_volume_l1875_187520


namespace NUMINAMATH_CALUDE_square_bound_values_l1875_187558

theorem square_bound_values (k : ℤ) : 
  (∃ (s : Finset ℤ), (∀ x ∈ s, 121 < x^2 ∧ x^2 < 225) ∧ s.card ≤ 3 ∧ 
   (∀ y : ℤ, 121 < y^2 ∧ y^2 < 225 → y ∈ s)) :=
by sorry

end NUMINAMATH_CALUDE_square_bound_values_l1875_187558


namespace NUMINAMATH_CALUDE_square_difference_identity_l1875_187541

theorem square_difference_identity (x : ℝ) (c : ℝ) (hc : c > 0) :
  (x^2 + c)^2 - (x^2 - c)^2 = 4*x^2*c := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l1875_187541


namespace NUMINAMATH_CALUDE_not_always_divisible_l1875_187557

theorem not_always_divisible : ¬ ∀ n : ℕ, (5^n - 1) % (4^n - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_not_always_divisible_l1875_187557


namespace NUMINAMATH_CALUDE_max_candies_one_student_l1875_187580

/-- Given a class of students, proves the maximum number of candies one student could have taken -/
theorem max_candies_one_student 
  (n : ℕ) -- number of students
  (mean : ℕ) -- mean number of candies per student
  (min_candies : ℕ) -- minimum number of candies per student
  (h1 : n = 25) -- there are 25 students
  (h2 : mean = 6) -- the mean number of candies is 6
  (h3 : min_candies = 2) -- each student takes at least 2 candies
  : ∃ (max_candies : ℕ), max_candies = 102 ∧ 
    max_candies = n * mean - (n - 1) * min_candies :=
by sorry

end NUMINAMATH_CALUDE_max_candies_one_student_l1875_187580


namespace NUMINAMATH_CALUDE_helga_shoe_shopping_l1875_187586

/-- The number of pairs of shoes Helga tried on at the first store -/
def first_store : ℕ := 7

/-- The number of pairs of shoes Helga tried on at the second store -/
def second_store : ℕ := first_store + 2

/-- The number of pairs of shoes Helga tried on at the third store -/
def third_store : ℕ := 0

/-- The number of pairs of shoes Helga tried on at the fourth store -/
def fourth_store : ℕ := 2 * (first_store + second_store + third_store)

/-- The total number of pairs of shoes Helga tried on -/
def total_shoes : ℕ := first_store + second_store + third_store + fourth_store

theorem helga_shoe_shopping :
  total_shoes = 48 := by sorry

end NUMINAMATH_CALUDE_helga_shoe_shopping_l1875_187586


namespace NUMINAMATH_CALUDE_concert_attendance_l1875_187593

theorem concert_attendance (adults : ℕ) (children : ℕ) : 
  children = 3 * adults →
  7 * adults + 3 * children = 6000 →
  adults + children = 1500 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l1875_187593


namespace NUMINAMATH_CALUDE_remainder_problem_l1875_187592

theorem remainder_problem (m n : ℕ) (h1 : m % n = 2) (h2 : (3 * m) % n = 1) : n = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1875_187592


namespace NUMINAMATH_CALUDE_sum_of_logarithmic_equation_l1875_187509

theorem sum_of_logarithmic_equation : 
  ∃ (k m n : ℕ+), 
    (Nat.gcd k.val (Nat.gcd m.val n.val) = 1) ∧ 
    (k.val * Real.log 5 / Real.log 400 + m.val * Real.log 2 / Real.log 400 = n.val) ∧
    (k.val + m.val + n.val = 7) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_logarithmic_equation_l1875_187509


namespace NUMINAMATH_CALUDE_malcolm_facebook_followers_l1875_187568

/-- Represents the number of followers on different social media platforms -/
structure Followers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ

/-- Calculates the total number of followers across all platforms -/
def totalFollowers (f : Followers) : ℕ :=
  f.instagram + f.facebook + f.twitter + f.tiktok + f.youtube

/-- Theorem stating that given the conditions, Malcolm has 375 followers on Facebook -/
theorem malcolm_facebook_followers :
  ∃ (f : Followers),
    f.instagram = 240 ∧
    f.twitter = (f.instagram + f.facebook) / 2 ∧
    f.tiktok = 3 * f.twitter ∧
    f.youtube = f.tiktok + 510 ∧
    totalFollowers f = 3840 →
    f.facebook = 375 := by
  sorry

end NUMINAMATH_CALUDE_malcolm_facebook_followers_l1875_187568


namespace NUMINAMATH_CALUDE_investment_dividend_l1875_187581

/-- Calculates the total dividend received from an investment in shares -/
theorem investment_dividend (investment : ℝ) (share_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ) :
  investment = 14400 →
  share_value = 100 →
  premium_rate = 0.20 →
  dividend_rate = 0.06 →
  let share_cost := share_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := share_value * dividend_rate
  dividend_per_share * num_shares = 720 := by
  sorry

end NUMINAMATH_CALUDE_investment_dividend_l1875_187581


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_fractional_equation_solution_l1875_187512

-- Equation 1
theorem quadratic_equation_solution (x : ℝ) :
  2 * x^2 + 3 * x = 1 ↔ x = (-3 + Real.sqrt 17) / 4 ∨ x = (-3 - Real.sqrt 17) / 4 :=
sorry

-- Equation 2
theorem fractional_equation_solution (x : ℝ) (h : x ≠ 2) :
  3 / (x - 2) = 5 / (2 - x) - 1 ↔ x = -6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_fractional_equation_solution_l1875_187512


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l1875_187517

-- Define the set of numbers
def S : Set Nat := {n | 1 ≤ n ∧ n ≤ 9}

-- Define a type for a selection of three numbers
structure Selection :=
  (a b c : Nat)
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (inS : a ∈ S ∧ b ∈ S ∧ c ∈ S)

-- Define events
def allEven (s : Selection) : Prop := s.a % 2 = 0 ∧ s.b % 2 = 0 ∧ s.c % 2 = 0
def allOdd (s : Selection) : Prop := s.a % 2 = 1 ∧ s.b % 2 = 1 ∧ s.c % 2 = 1
def oneEvenTwoOdd (s : Selection) : Prop :=
  (s.a % 2 = 0 ∧ s.b % 2 = 1 ∧ s.c % 2 = 1) ∨
  (s.a % 2 = 1 ∧ s.b % 2 = 0 ∧ s.c % 2 = 1) ∨
  (s.a % 2 = 1 ∧ s.b % 2 = 1 ∧ s.c % 2 = 0)
def twoEvenOneOdd (s : Selection) : Prop :=
  (s.a % 2 = 0 ∧ s.b % 2 = 0 ∧ s.c % 2 = 1) ∨
  (s.a % 2 = 0 ∧ s.b % 2 = 1 ∧ s.c % 2 = 0) ∨
  (s.a % 2 = 1 ∧ s.b % 2 = 0 ∧ s.c % 2 = 0)

-- Define mutual exclusivity and complementarity
def mutuallyExclusive (e1 e2 : Selection → Prop) : Prop :=
  ∀ s : Selection, ¬(e1 s ∧ e2 s)

def complementary (e1 e2 : Selection → Prop) : Prop :=
  ∀ s : Selection, e1 s ∨ e2 s

-- Theorem statement
theorem events_mutually_exclusive_not_complementary :
  (mutuallyExclusive allEven allOdd ∧ ¬complementary allEven allOdd) ∧
  (mutuallyExclusive oneEvenTwoOdd twoEvenOneOdd ∧ ¬complementary oneEvenTwoOdd twoEvenOneOdd) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_complementary_l1875_187517


namespace NUMINAMATH_CALUDE_registered_number_scientific_notation_l1875_187521

/-- The number of people registered for the national college entrance examination in 2023 -/
def registered_number : ℝ := 12910000

/-- The scientific notation representation of the registered number -/
def scientific_notation : ℝ := 1.291 * (10 ^ 7)

/-- Theorem stating that the registered number is equal to its scientific notation representation -/
theorem registered_number_scientific_notation : registered_number = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_registered_number_scientific_notation_l1875_187521


namespace NUMINAMATH_CALUDE_right_pyramid_base_side_length_l1875_187501

/-- Represents a right pyramid with an equilateral triangular base -/
structure RightPyramid where
  base_side_length : ℝ
  slant_height : ℝ
  lateral_face_area : ℝ

/-- Theorem: If the area of one lateral face is 90 square meters and the slant height is 20 meters,
    then the side length of the base is 9 meters -/
theorem right_pyramid_base_side_length 
  (pyramid : RightPyramid) 
  (h1 : pyramid.lateral_face_area = 90) 
  (h2 : pyramid.slant_height = 20) : 
  pyramid.base_side_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_right_pyramid_base_side_length_l1875_187501


namespace NUMINAMATH_CALUDE_base8_digit_product_l1875_187552

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers --/
def productList (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 8 representation of 7254 (base 10) is 72 --/
theorem base8_digit_product :
  productList (toBase8 7254) = 72 := by
  sorry

end NUMINAMATH_CALUDE_base8_digit_product_l1875_187552


namespace NUMINAMATH_CALUDE_A_intersect_B_l1875_187574

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {x | x > 2}

theorem A_intersect_B : A ∩ B = {3, 4} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l1875_187574


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1875_187563

theorem polynomial_simplification (x : ℝ) :
  (3 * x^3 + 4 * x^2 + 6 * x - 5) - (2 * x^3 + 2 * x^2 + 3 * x - 8) = x^3 + 2 * x^2 + 3 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1875_187563


namespace NUMINAMATH_CALUDE_cylinder_dimensions_l1875_187554

/-- Proves that a cylinder with equal surface area to a sphere of radius 4 cm
    and with equal height and diameter has height and diameter of 8 cm. -/
theorem cylinder_dimensions (r : ℝ) (h : ℝ) :
  r = 4 →  -- radius of the sphere is 4 cm
  (4 * π * r^2 : ℝ) = 2 * π * r * h →  -- surface areas are equal
  h = 2 * r →  -- height equals diameter
  h = 8 ∧ (2 * r) = 8 :=  -- height and diameter are both 8 cm
by sorry

end NUMINAMATH_CALUDE_cylinder_dimensions_l1875_187554


namespace NUMINAMATH_CALUDE_all_ap_lines_pass_through_point_l1875_187582

/-- A line in the form ax + by = c where a, b, and c form an arithmetic progression -/
structure APLine where
  a : ℝ
  d : ℝ
  eq : ℝ × ℝ → Prop := fun (x, y) ↦ a * x + (a + d) * y = a + 2 * d

/-- The theorem stating that all APLines pass through the point (-1, 2) -/
theorem all_ap_lines_pass_through_point :
  ∀ (l : APLine), l.eq (-1, 2) :=
sorry

end NUMINAMATH_CALUDE_all_ap_lines_pass_through_point_l1875_187582


namespace NUMINAMATH_CALUDE_max_a_value_l1875_187508

theorem max_a_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = a * x^2 - a * x + 1) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f x| ≤ 1) →
  a ≤ 8 ∧ ∃ b : ℝ, b > 8 ∧ ∃ y : ℝ, 0 ≤ y ∧ y ≤ 1 ∧ |b * y^2 - b * y + 1| > 1 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l1875_187508


namespace NUMINAMATH_CALUDE_jill_second_bus_ride_time_l1875_187555

/-- The time Jill spends waiting for her first bus, in minutes -/
def first_bus_wait : ℕ := 12

/-- The time Jill spends riding on her first bus, in minutes -/
def first_bus_ride : ℕ := 30

/-- The total time Jill spends on her first bus (waiting and riding), in minutes -/
def first_bus_total : ℕ := first_bus_wait + first_bus_ride

/-- The time Jill spends on her second bus ride, in minutes -/
def second_bus_ride : ℕ := first_bus_total / 2

theorem jill_second_bus_ride_time :
  second_bus_ride = 21 := by sorry

end NUMINAMATH_CALUDE_jill_second_bus_ride_time_l1875_187555


namespace NUMINAMATH_CALUDE_inequality_solution_l1875_187513

theorem inequality_solution (x : ℝ) :
  (x * (x + 1)) / ((x - 5)^2) ≥ 15 ↔ (x > Real.sqrt (151 - Real.sqrt 1801) / 2 ∧ x < 5) ∨
                                    (x > 5 ∧ x < Real.sqrt (151 + Real.sqrt 1801) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1875_187513


namespace NUMINAMATH_CALUDE_initial_rope_length_correct_l1875_187587

/-- The initial length of rope before decorating trees -/
def initial_rope_length : ℝ := 8.9

/-- The length of string used to decorate one tree -/
def string_per_tree : ℝ := 0.84

/-- The number of trees decorated -/
def num_trees : ℕ := 10

/-- The length of rope remaining after decorating trees -/
def remaining_rope : ℝ := 0.5

/-- Theorem stating that the initial rope length is correct -/
theorem initial_rope_length_correct :
  initial_rope_length = string_per_tree * num_trees + remaining_rope :=
by sorry

end NUMINAMATH_CALUDE_initial_rope_length_correct_l1875_187587


namespace NUMINAMATH_CALUDE_triangle_inequalities_l1875_187514

/-- Given four collinear points E, F, G, H in order, with EF = a, EG = b, EH = c,
    if EF and GH are rotated to form a triangle with positive area,
    then a < c/3 and b < a + c/3 must be true, while b < c/3 is not necessarily true. -/
theorem triangle_inequalities (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hab : a < b) (hbc : b < c) 
  (h_triangle : a + (c - b) > b - a) : 
  (a < c / 3 ∧ b < a + c / 3) ∧ ¬(b < c / 3 → True) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l1875_187514


namespace NUMINAMATH_CALUDE_problem_1_l1875_187585

theorem problem_1 : Real.sin (30 * π / 180) + |(-1)| - (Real.sqrt 3 - Real.pi)^0 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1875_187585


namespace NUMINAMATH_CALUDE_system_1_solution_system_2_solution_l1875_187535

-- System 1
theorem system_1_solution :
  ∃ (x y : ℝ), y = 2 * x ∧ x + y = 12 ∧ x = 4 ∧ y = 8 := by sorry

-- System 2
theorem system_2_solution :
  ∃ (x y : ℝ), 3 * x + 5 * y = 21 ∧ 2 * x - 5 * y = -11 ∧ x = 2 ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_system_1_solution_system_2_solution_l1875_187535


namespace NUMINAMATH_CALUDE_expansion_properties_l1875_187531

-- Define the expansion of (x-m)^7
def expansion (x m : ℝ) (a : Fin 8 → ℝ) : Prop :=
  (x - m)^7 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7

-- State the theorem
theorem expansion_properties {m : ℝ} {a : Fin 8 → ℝ} 
  (h_expansion : ∀ x, expansion x m a)
  (h_coeff : a 4 = -35) :
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 1) ∧
  (a 1 + a 3 + a 5 + a 7 = 26) := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l1875_187531


namespace NUMINAMATH_CALUDE_smallest_k_for_equalization_l1875_187515

/-- Represents the state of gas cylinders -/
def CylinderState := List ℝ

/-- Represents a connection operation on cylinders -/
def Connection := List Nat

/-- Applies a single connection operation to a cylinder state -/
def applyConnection (state : CylinderState) (conn : Connection) : CylinderState :=
  sorry

/-- Checks if all pressures in a state are equal -/
def isEqualized (state : CylinderState) : Prop :=
  sorry

/-- Checks if a connection is valid (size ≤ k) -/
def isValidConnection (conn : Connection) (k : ℕ) : Prop :=
  sorry

/-- Represents a sequence of connection operations -/
def EqualizationProcess := List Connection

/-- Checks if an equalization process is valid for a given k -/
def isValidProcess (process : EqualizationProcess) (k : ℕ) : Prop :=
  sorry

/-- Applies an equalization process to a cylinder state -/
def applyProcess (state : CylinderState) (process : EqualizationProcess) : CylinderState :=
  sorry

/-- Main theorem: 5 is the smallest k that allows equalization -/
theorem smallest_k_for_equalization :
  (∀ (initial : CylinderState), initial.length = 40 →
    ∃ (process : EqualizationProcess), 
      isValidProcess process 5 ∧ 
      isEqualized (applyProcess initial process)) ∧
  (∀ k < 5, ∃ (initial : CylinderState), initial.length = 40 ∧
    ∀ (process : EqualizationProcess), 
      isValidProcess process k → 
      ¬isEqualized (applyProcess initial process)) :=
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_equalization_l1875_187515


namespace NUMINAMATH_CALUDE_base_n_representation_of_b_l1875_187564

/-- Represents a number in base n -/
def BaseNRepr (n : ℕ) (x : ℕ) : Prop :=
  ∃ (d₀ d₁ : ℕ), x = d₁ * n + d₀ ∧ d₁ < n ∧ d₀ < n

theorem base_n_representation_of_b
  (n : ℕ)
  (hn : n > 9)
  (a b : ℕ)
  (heq : n^2 - a*n + b = 0)
  (ha : BaseNRepr n a ∧ a = 19) :
  BaseNRepr n b ∧ b = 90 := by
  sorry

end NUMINAMATH_CALUDE_base_n_representation_of_b_l1875_187564


namespace NUMINAMATH_CALUDE_system_solution_range_l1875_187524

theorem system_solution_range (x y k : ℝ) : 
  (2 * x - 3 * y = 5) → 
  (2 * x - y = k) → 
  (x > y) → 
  (k > -5) :=
sorry

end NUMINAMATH_CALUDE_system_solution_range_l1875_187524


namespace NUMINAMATH_CALUDE_roberts_birth_year_l1875_187545

theorem roberts_birth_year (n : ℕ) : 
  (n + 1)^2 - n^2 = 89 → n^2 = 1936 := by
  sorry

end NUMINAMATH_CALUDE_roberts_birth_year_l1875_187545


namespace NUMINAMATH_CALUDE_coloring_count_l1875_187550

/-- The number of colors available -/
def num_colors : ℕ := 5

/-- The number of parts to be colored -/
def num_parts : ℕ := 3

/-- A function that calculates the number of coloring possibilities -/
def count_colorings : ℕ := num_colors * (num_colors - 1) * (num_colors - 2)

/-- Theorem stating that the number of valid colorings is 60 -/
theorem coloring_count : count_colorings = 60 := by
  sorry

end NUMINAMATH_CALUDE_coloring_count_l1875_187550


namespace NUMINAMATH_CALUDE_concatenated_evens_not_divisible_by_24_l1875_187570

def concatenated_evens : ℕ := 121416182022242628303234

theorem concatenated_evens_not_divisible_by_24 : ¬ (concatenated_evens % 24 = 0) := by
  sorry

end NUMINAMATH_CALUDE_concatenated_evens_not_divisible_by_24_l1875_187570


namespace NUMINAMATH_CALUDE_prime_square_mod_180_l1875_187547

theorem prime_square_mod_180 (p : ℕ) (hp : Nat.Prime p) (hp_gt_5 : p > 5) :
  ∃ r : Fin 180, p^2 % 180 = r.val ∧ (r.val = 1 ∨ r.val = 145) := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_180_l1875_187547


namespace NUMINAMATH_CALUDE_max_value_of_f_l1875_187597

def f (x : ℕ) : ℤ := 2 * x - 3

def S : Set ℕ := {x : ℕ | 1 ≤ x ∧ x ≤ 10/3}

theorem max_value_of_f :
  ∃ (m : ℤ), m = 3 ∧ ∀ (x : ℕ), x ∈ S → f x ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1875_187597


namespace NUMINAMATH_CALUDE_restaurant_bill_l1875_187577

theorem restaurant_bill (total_friends : ℕ) (contributing_friends : ℕ) (extra_payment : ℚ) (total_bill : ℚ) :
  total_friends = 10 →
  contributing_friends = 9 →
  extra_payment = 3 →
  total_bill = (contributing_friends * (total_bill / total_friends + extra_payment)) →
  total_bill = 270 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_l1875_187577


namespace NUMINAMATH_CALUDE_evaluate_expression_l1875_187500

theorem evaluate_expression : (2^3)^2 - (3^2)^3 = -665 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1875_187500


namespace NUMINAMATH_CALUDE_jungkook_has_fewest_erasers_l1875_187589

-- Define the number of erasers for each person
def jungkook_erasers : ℕ := 6
def jimin_erasers : ℕ := jungkook_erasers + 4
def seokjin_erasers : ℕ := jimin_erasers - 3

-- Theorem to prove Jungkook has the fewest erasers
theorem jungkook_has_fewest_erasers :
  jungkook_erasers < jimin_erasers ∧ jungkook_erasers < seokjin_erasers :=
by sorry

end NUMINAMATH_CALUDE_jungkook_has_fewest_erasers_l1875_187589


namespace NUMINAMATH_CALUDE_complement_of_union_l1875_187510

def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3, 4}
def B : Set ℕ := {3, 5}

theorem complement_of_union :
  (U \ (A ∪ B)) = {2} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l1875_187510


namespace NUMINAMATH_CALUDE_paint_mixture_problem_l1875_187537

/-- Given a paint mixture with ratio 7:2:1:1 for blue, red, white, and green,
    prove that if 140 oz of blue paint is used and the total mixture should not exceed 220 oz,
    then 20 oz of white paint is required. -/
theorem paint_mixture_problem (blue red white green : ℕ) 
  (ratio : blue = 7 ∧ red = 2 ∧ white = 1 ∧ green = 1) 
  (blue_amount : ℕ) (total_limit : ℕ)
  (h_blue_amount : blue_amount = 140)
  (h_total_limit : total_limit = 220) :
  let total_parts := blue + red + white + green
  let ounces_per_part := blue_amount / blue
  let white_amount := ounces_per_part * white
  white_amount = 20 ∧ white_amount ≤ total_limit - blue_amount :=
by sorry

end NUMINAMATH_CALUDE_paint_mixture_problem_l1875_187537


namespace NUMINAMATH_CALUDE_third_person_weight_l1875_187567

/-- The weight of the third person entering an elevator given specific average weight changes --/
theorem third_person_weight (initial_people : ℕ) (initial_avg : ℝ) 
  (avg_after_first : ℝ) (avg_after_second : ℝ) (avg_after_third : ℝ) :
  initial_people = 6 →
  initial_avg = 156 →
  avg_after_first = 159 →
  avg_after_second = 162 →
  avg_after_third = 161 →
  ∃ (w1 w2 w3 : ℝ),
    w1 = (initial_people + 1) * avg_after_first - initial_people * initial_avg ∧
    w2 = (initial_people + 2) * avg_after_second - (initial_people + 1) * avg_after_first ∧
    w3 = (initial_people + 3) * avg_after_third - (initial_people + 2) * avg_after_second ∧
    w3 = 163 :=
by sorry

end NUMINAMATH_CALUDE_third_person_weight_l1875_187567


namespace NUMINAMATH_CALUDE_oranges_per_child_l1875_187559

/-- Given 4 children and 12 oranges in total, prove that each child has 3 oranges. -/
theorem oranges_per_child (num_children : ℕ) (total_oranges : ℕ) 
  (h1 : num_children = 4) (h2 : total_oranges = 12) : 
  total_oranges / num_children = 3 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_child_l1875_187559


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1875_187553

theorem log_sum_equals_two : Real.log 0.01 / Real.log 10 + Real.log 16 / Real.log 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1875_187553


namespace NUMINAMATH_CALUDE_work_completion_time_l1875_187526

theorem work_completion_time (x : ℝ) : 
  (x > 0) →  -- A's completion time is positive
  (4 * (1 / x + 1 / 30) = 1 / 3) →  -- Equation from working together
  (x = 20) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1875_187526


namespace NUMINAMATH_CALUDE_decagon_interior_intersections_l1875_187518

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def interior_intersection_points (n : ℕ) : ℕ :=
  Nat.choose n 4

/-- A regular decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_interior_intersections :
  interior_intersection_points decagon_sides = 210 := by
  sorry

end NUMINAMATH_CALUDE_decagon_interior_intersections_l1875_187518


namespace NUMINAMATH_CALUDE_least_multiple_of_primes_l1875_187507

theorem least_multiple_of_primes : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 ∧ 3 ∣ m ∧ 5 ∣ m ∧ 7 ∣ m → n ≤ m) ∧ 
  3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ n = 105 := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_primes_l1875_187507


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1875_187590

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - 2 * Complex.I) : 
  z.im = -3/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1875_187590


namespace NUMINAMATH_CALUDE_socks_needed_to_triple_wardrobe_l1875_187591

/-- Represents the number of items in Jonas' wardrobe -/
structure Wardrobe where
  socks : ℕ
  shoes : ℕ
  pants : ℕ
  tshirts : ℕ
  hats : ℕ
  jackets : ℕ

/-- Calculates the total number of individual items in the wardrobe -/
def totalItems (w : Wardrobe) : ℕ :=
  w.socks * 2 + w.shoes * 2 + w.pants + w.tshirts + w.hats + w.jackets

/-- Jonas' current wardrobe -/
def jonasWardrobe : Wardrobe :=
  { socks := 20
    shoes := 5
    pants := 10
    tshirts := 10
    hats := 6
    jackets := 4 }

/-- Theorem: Jonas needs to buy 80 pairs of socks to triple his wardrobe -/
theorem socks_needed_to_triple_wardrobe :
  let current := totalItems jonasWardrobe
  let target := current * 3
  let difference := target - current
  difference / 2 = 80 := by sorry

end NUMINAMATH_CALUDE_socks_needed_to_triple_wardrobe_l1875_187591


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l1875_187533

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the basis vectors
variable (e₁ e₂ : V)

-- Define the points and vectors
variable (A B C D : V)
variable (k : ℝ)

-- State the theorem
theorem collinear_points_k_value
  (hAB : B - A = e₁ - e₂)
  (hBC : C - B = 3 • e₁ + 2 • e₂)
  (hCD : D - C = k • e₁ + 2 • e₂)
  (hCollinear : ∃ (t : ℝ), D - A = t • (C - A)) :
  k = 8 := by sorry

end NUMINAMATH_CALUDE_collinear_points_k_value_l1875_187533


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l1875_187503

theorem recipe_flour_amount (flour_added : ℕ) (flour_needed : ℕ) : 
  flour_added = 2 → flour_needed = 6 → flour_added + flour_needed = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l1875_187503


namespace NUMINAMATH_CALUDE_total_games_in_season_l1875_187595

/-- Calculate the number of games in a round-robin tournament -/
def num_games (n : ℕ) (r : ℕ) : ℕ :=
  (n * (n - 1) / 2) * r

/-- The number of teams in the league -/
def num_teams : ℕ := 14

/-- The number of times each team plays every other team -/
def num_rounds : ℕ := 5

theorem total_games_in_season :
  num_games num_teams num_rounds = 455 := by sorry

end NUMINAMATH_CALUDE_total_games_in_season_l1875_187595


namespace NUMINAMATH_CALUDE_circles_intersect_l1875_187549

/-- Circle C₁ with equation x² + y² + 2x + 8y - 8 = 0 -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 8*y - 8 = 0

/-- Circle C₂ with equation x² + y² - 4x - 5 = 0 -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 5 = 0

/-- The circles C₁ and C₂ intersect -/
theorem circles_intersect : ∃ (x y : ℝ), C₁ x y ∧ C₂ x y := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l1875_187549


namespace NUMINAMATH_CALUDE_tree_planting_around_lake_l1875_187519

theorem tree_planting_around_lake (circumference : ℕ) (willow_interval : ℕ) : 
  circumference = 1200 → willow_interval = 10 → 
  (circumference / willow_interval + circumference / willow_interval = 240) := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_around_lake_l1875_187519


namespace NUMINAMATH_CALUDE_single_intersection_l1875_187578

/-- The quadratic function representing the first graph -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + 2 * x + 3

/-- The linear function representing the second graph -/
def g (x : ℝ) : ℝ := 2 * x + 5

/-- The theorem stating the condition for a single intersection point -/
theorem single_intersection (k : ℝ) : 
  (∃! x, f k x = g x) ↔ k = -1/2 := by sorry

end NUMINAMATH_CALUDE_single_intersection_l1875_187578


namespace NUMINAMATH_CALUDE_kitchen_tiling_l1875_187522

def kitchen_length : ℕ := 20
def kitchen_width : ℕ := 15
def border_width : ℕ := 2
def border_tile_length : ℕ := 2
def border_tile_width : ℕ := 1
def inner_tile_size : ℕ := 3

def border_tiles_count : ℕ := 
  2 * (kitchen_length - 2 * border_width) / border_tile_length +
  2 * (kitchen_width - 2 * border_width) / border_tile_length

def inner_area : ℕ := (kitchen_length - 2 * border_width) * (kitchen_width - 2 * border_width)

def inner_tiles_count : ℕ := (inner_area + inner_tile_size^2 - 1) / inner_tile_size^2

def total_tiles : ℕ := border_tiles_count + inner_tiles_count

theorem kitchen_tiling :
  total_tiles = 48 :=
sorry

end NUMINAMATH_CALUDE_kitchen_tiling_l1875_187522


namespace NUMINAMATH_CALUDE_prob_green_marble_l1875_187543

/-- The probability of drawing a green marble from a box of 90 marbles -/
theorem prob_green_marble (total_marbles : ℕ) (prob_white : ℝ) (prob_red_or_blue : ℝ) :
  total_marbles = 90 →
  prob_white = 1 / 6 →
  prob_red_or_blue = 0.6333333333333333 →
  ∃ (prob_green : ℝ), prob_green = 0.2 ∧ prob_white + prob_red_or_blue + prob_green = 1 :=
by sorry

end NUMINAMATH_CALUDE_prob_green_marble_l1875_187543


namespace NUMINAMATH_CALUDE_product_consecutive_integers_even_l1875_187571

theorem product_consecutive_integers_even (n : ℤ) : ∃ k : ℤ, n * (n + 1) = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_product_consecutive_integers_even_l1875_187571


namespace NUMINAMATH_CALUDE_student_handshake_problem_l1875_187528

theorem student_handshake_problem (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  let total_handshakes := (12 + 10 * (m + n - 4) + 8 * (m - 2) * (n - 2)) / 2
  total_handshakes = 1020 → m * n = 280 := by
  sorry

end NUMINAMATH_CALUDE_student_handshake_problem_l1875_187528


namespace NUMINAMATH_CALUDE_james_score_l1875_187556

/-- Quiz bowl scoring system at Highridge High -/
structure QuizBowl where
  pointsPerCorrect : ℕ := 2
  bonusPoints : ℕ := 4
  numRounds : ℕ := 5
  questionsPerRound : ℕ := 5

/-- Calculate the total points scored by a student in the quiz bowl -/
def calculatePoints (qb : QuizBowl) (missedQuestions : ℕ) : ℕ :=
  let totalQuestions := qb.numRounds * qb.questionsPerRound
  let correctAnswers := totalQuestions - missedQuestions
  let pointsFromCorrect := correctAnswers * qb.pointsPerCorrect
  let fullRounds := qb.numRounds - (if missedQuestions > 0 then 1 else 0)
  let bonusPointsTotal := fullRounds * qb.bonusPoints
  pointsFromCorrect + bonusPointsTotal

/-- Theorem: James scored 64 points in the quiz bowl -/
theorem james_score (qb : QuizBowl) : calculatePoints qb 1 = 64 := by
  sorry

end NUMINAMATH_CALUDE_james_score_l1875_187556


namespace NUMINAMATH_CALUDE_ron_chocolate_cost_l1875_187575

/-- Calculates the cost of chocolate bars for a boy scout camp out -/
def chocolate_cost (chocolate_bar_price : ℚ) (sections_per_bar : ℕ) (num_scouts : ℕ) (smores_per_scout : ℕ) : ℚ :=
  let total_smores := num_scouts * smores_per_scout
  let bars_needed := (total_smores + sections_per_bar - 1) / sections_per_bar
  bars_needed * chocolate_bar_price

/-- Theorem: The cost of chocolate bars for Ron's boy scout camp out is $15.00 -/
theorem ron_chocolate_cost :
  chocolate_cost (3/2) 3 15 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_ron_chocolate_cost_l1875_187575


namespace NUMINAMATH_CALUDE_yang_final_floor_l1875_187532

/-- The number of floors in the building -/
def total_floors : ℕ := 36

/-- The floor Xiao Wu reaches in the initial observation -/
def wu_initial : ℕ := 6

/-- The floor Xiao Yang reaches in the initial observation -/
def yang_initial : ℕ := 5

/-- The starting floor for both climbers -/
def start_floor : ℕ := 1

/-- The floor Xiao Yang reaches when Xiao Wu reaches the top floor -/
def yang_final : ℕ := 29

theorem yang_final_floor :
  (wu_initial - start_floor) / (yang_initial - start_floor) =
  (total_floors - start_floor) / (yang_final - start_floor) :=
sorry

end NUMINAMATH_CALUDE_yang_final_floor_l1875_187532


namespace NUMINAMATH_CALUDE_angle_measure_from_cosine_l1875_187544

theorem angle_measure_from_cosine (A : Real) : 
  0 < A → A < Real.pi / 2 → -- A is acute
  Real.cos A = Real.sqrt 3 / 2 → -- cos A = √3/2
  A = Real.pi / 6 -- A = 30° (π/6 radians)
:= by sorry

end NUMINAMATH_CALUDE_angle_measure_from_cosine_l1875_187544
