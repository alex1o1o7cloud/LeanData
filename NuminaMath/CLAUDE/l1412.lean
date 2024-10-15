import Mathlib

namespace NUMINAMATH_CALUDE_max_a_value_l1412_141240

-- Define the quadratic polynomial f(x) = x^2 + ax + b
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem max_a_value (a b : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a b y = f a b x + y) →
  a ≤ (1/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1412_141240


namespace NUMINAMATH_CALUDE_parallel_segments_and_midpoint_l1412_141217

/-- Given four points on a Cartesian plane, if two line segments formed by these points are parallel,
    then we can determine the y-coordinate of one point and the midpoint of one segment. -/
theorem parallel_segments_and_midpoint
  (A B X Y : ℝ × ℝ)
  (hA : A = (-6, 2))
  (hB : B = (2, -6))
  (hX : X = (4, 16))
  (hY : Y = (20, k))
  (h_parallel : (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1)) :
  k = 0 ∧ ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2) = (12, 8) :=
by sorry

end NUMINAMATH_CALUDE_parallel_segments_and_midpoint_l1412_141217


namespace NUMINAMATH_CALUDE_complex_number_purely_imaginary_l1412_141247

theorem complex_number_purely_imaginary (a : ℝ) : 
  (a = -1) ↔ (∃ (t : ℝ), (1 + I) / (1 + a * I) = t * I) :=
sorry

end NUMINAMATH_CALUDE_complex_number_purely_imaginary_l1412_141247


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l1412_141202

theorem smallest_n_for_sqrt_inequality :
  ∀ n : ℕ, n > 0 → (Real.sqrt (5 * n) - Real.sqrt (5 * n - 4) < 0.01) ↔ n ≥ 8001 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l1412_141202


namespace NUMINAMATH_CALUDE_range_of_a_l1412_141219

/-- The range of real number a satisfying the given inequality -/
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (a * x + 1) * (Real.exp x - a * Real.exp 1 * x) ≥ 0) ↔ 
  (0 ≤ a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1412_141219


namespace NUMINAMATH_CALUDE_work_completion_time_l1412_141211

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 30
def work_rate_B : ℚ := 1 / 55
def work_rate_C : ℚ := 1 / 45

-- Define the combined work rate
def combined_work_rate : ℚ := work_rate_A + work_rate_B + work_rate_C

-- Define the time taken to complete the work together
def time_to_complete : ℚ := 1 / combined_work_rate

-- Theorem statement
theorem work_completion_time :
  time_to_complete = 55 / 4 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1412_141211


namespace NUMINAMATH_CALUDE_total_insects_count_l1412_141204

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The number of lacewings -/
def lacewings : ℕ := 23250

/-- The total number of insects on the fields -/
def total_insects : ℕ := ladybugs_with_spots + ladybugs_without_spots + lacewings

theorem total_insects_count : total_insects = 90332 := by
  sorry

end NUMINAMATH_CALUDE_total_insects_count_l1412_141204


namespace NUMINAMATH_CALUDE_compare_two_point_five_and_sqrt_six_l1412_141264

theorem compare_two_point_five_and_sqrt_six :
  2.5 > Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_compare_two_point_five_and_sqrt_six_l1412_141264


namespace NUMINAMATH_CALUDE_group_selection_problem_l1412_141296

theorem group_selection_problem (n : ℕ) (k : ℕ) : n = 30 ∧ k = 3 → Nat.choose n k = 4060 := by
  sorry

end NUMINAMATH_CALUDE_group_selection_problem_l1412_141296


namespace NUMINAMATH_CALUDE_fraction_value_l1412_141236

theorem fraction_value (a b : ℝ) (h1 : 1/a + 2/b = 1) (h2 : a ≠ -b) : (a*b - a) / (a + b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1412_141236


namespace NUMINAMATH_CALUDE_community_age_theorem_l1412_141225

/-- Represents the average age of a community given the ratio of women to men and their respective average ages -/
def community_average_age (women_ratio : ℚ) (men_ratio : ℚ) (women_avg_age : ℚ) (men_avg_age : ℚ) : ℚ :=
  (women_ratio * women_avg_age + men_ratio * men_avg_age) / (women_ratio + men_ratio)

/-- Theorem stating that for a community with a 3:2 ratio of women to men, where women's average age is 30 and men's is 35, the community's average age is 32 -/
theorem community_age_theorem :
  community_average_age (3/5) (2/5) 30 35 = 32 := by sorry

end NUMINAMATH_CALUDE_community_age_theorem_l1412_141225


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1412_141244

theorem unique_integer_solution :
  ∃! (x y z : ℤ), x^2 + y^2 + z^2 = x^2 * y^2 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1412_141244


namespace NUMINAMATH_CALUDE_acute_triangle_angle_sine_inequality_l1412_141218

theorem acute_triangle_angle_sine_inequality (A B C : Real) 
  (h1 : 0 < A ∧ A < π/2) 
  (h2 : 0 < B ∧ B < π/2) 
  (h3 : 0 < C ∧ C < π/2) 
  (h4 : A + B + C = π) 
  (h5 : A < B) 
  (h6 : B < C) : 
  Real.sin (2*A) > Real.sin (2*B) ∧ Real.sin (2*B) > Real.sin (2*C) := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_angle_sine_inequality_l1412_141218


namespace NUMINAMATH_CALUDE_system_solution_l1412_141207

theorem system_solution (x y : ℝ) (eq1 : 3 * x + y = 21) (eq2 : x + 3 * y = 1) : 2 * x + 2 * y = 11 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1412_141207


namespace NUMINAMATH_CALUDE_condition_for_f_sum_positive_l1412_141250

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x

-- State the theorem
theorem condition_for_f_sum_positive :
  ∀ (a b : ℝ), (a + b > 0 ↔ f a + f b > 0) := by sorry

end NUMINAMATH_CALUDE_condition_for_f_sum_positive_l1412_141250


namespace NUMINAMATH_CALUDE_fraction_equality_l1412_141272

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : a / b = (2 * a) / (2 * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1412_141272


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1412_141259

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 + 0.000012 = 3858 / 15625 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1412_141259


namespace NUMINAMATH_CALUDE_inequality_proof_l1412_141220

theorem inequality_proof (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (h : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (x + z) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1412_141220


namespace NUMINAMATH_CALUDE_sine_function_amplitude_l1412_141248

theorem sine_function_amplitude (a b : ℝ) (ha : a < 0) (hb : b > 0) :
  (∀ x, |a * Real.sin (b * x)| ≤ 3) ∧ (∃ x, |a * Real.sin (b * x)| = 3) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_amplitude_l1412_141248


namespace NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l1412_141287

theorem units_digit_of_7_power_2023 : 7^2023 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_power_2023_l1412_141287


namespace NUMINAMATH_CALUDE_shaded_cells_after_five_minutes_l1412_141239

/-- Represents the state of the grid at a given minute -/
def GridState := Nat → Nat → Bool

/-- The initial state of the grid with a 1 × 5 shaded rectangle -/
def initial_state : GridState := sorry

/-- The rule for shading cells in the next minute -/
def shade_rule (state : GridState) : GridState := sorry

/-- The state of the grid after n minutes -/
def state_after (n : Nat) : GridState := sorry

/-- Counts the number of shaded cells in a given state -/
def count_shaded (state : GridState) : Nat := sorry

/-- The main theorem: after 5 minutes, 105 cells are shaded -/
theorem shaded_cells_after_five_minutes :
  count_shaded (state_after 5) = 105 := by sorry

end NUMINAMATH_CALUDE_shaded_cells_after_five_minutes_l1412_141239


namespace NUMINAMATH_CALUDE_paperboy_delivery_ways_l1412_141251

/-- Represents the number of valid delivery sequences for n houses -/
def E : ℕ → ℕ
  | 0 => 0  -- No houses, no deliveries
  | 1 => 2  -- For one house, two options: deliver or not
  | 2 => 4  -- For two houses, all combinations are valid
  | 3 => 8  -- E_3 = E_2 + E_1 + 2
  | n + 4 => E (n + 3) + E (n + 2) + E (n + 1)

/-- The problem statement -/
theorem paperboy_delivery_ways : E 12 = 1854 := by
  sorry

end NUMINAMATH_CALUDE_paperboy_delivery_ways_l1412_141251


namespace NUMINAMATH_CALUDE_sin_225_degrees_l1412_141280

theorem sin_225_degrees : Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_225_degrees_l1412_141280


namespace NUMINAMATH_CALUDE_median_inequality_l1412_141275

/-- Given a triangle ABC with sides a, b, c and medians s_a, s_b, s_c,
    if a < (b+c)/2, then s_a > (s_b + s_c)/2 -/
theorem median_inequality (a b c s_a s_b s_c : ℝ) 
    (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
    (h_medians : s_a = (1/2) * Real.sqrt (2*b^2 + 2*c^2 - a^2) ∧
                 s_b = (1/2) * Real.sqrt (2*a^2 + 2*c^2 - b^2) ∧
                 s_c = (1/2) * Real.sqrt (2*a^2 + 2*b^2 - c^2))
    (h_cond : a < (b + c) / 2) :
  s_a > (s_b + s_c) / 2 := by
  sorry

end NUMINAMATH_CALUDE_median_inequality_l1412_141275


namespace NUMINAMATH_CALUDE_grocer_coffee_stock_l1412_141283

theorem grocer_coffee_stock (initial_stock : ℝ) : 
  initial_stock > 0 →
  0.30 * initial_stock + 0.60 * 100 = 0.36 * (initial_stock + 100) →
  initial_stock = 400 := by
sorry

end NUMINAMATH_CALUDE_grocer_coffee_stock_l1412_141283


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l1412_141245

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speedAgainstCurrent (speedWithCurrent : ℝ) (currentSpeed : ℝ) : ℝ :=
  speedWithCurrent - 2 * currentSpeed

/-- Theorem stating that given the specific speeds mentioned in the problem, 
    the man's speed against the current is 10 km/hr. -/
theorem mans_speed_against_current :
  speedAgainstCurrent 15 2.5 = 10 := by
  sorry

#eval speedAgainstCurrent 15 2.5

end NUMINAMATH_CALUDE_mans_speed_against_current_l1412_141245


namespace NUMINAMATH_CALUDE_largest_prime_with_special_form_l1412_141268

theorem largest_prime_with_special_form :
  ∀ p : ℕ, Prime p →
    (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ p = (b / 2 : ℚ) * Real.sqrt ((a - b : ℚ) / (a + b))) →
    p ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_with_special_form_l1412_141268


namespace NUMINAMATH_CALUDE_unique_solution_l1412_141290

/-- The product of all digits of a positive integer -/
def digit_product (n : ℕ+) : ℕ := sorry

/-- Theorem stating that 12 is the only positive integer solution -/
theorem unique_solution : 
  ∃! (x : ℕ+), digit_product x = x^2 - 10*x - 22 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1412_141290


namespace NUMINAMATH_CALUDE_part1_part2_l1412_141235

-- Part 1
def f (x : ℝ) : ℝ := |2*x - 2| + 2

theorem part1 : {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Part 2
def g (x : ℝ) : ℝ := |2*x - 1|

def h (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

theorem part2 : {a : ℝ | ∀ x : ℝ, h a x + g x ≥ 3} = {a : ℝ | 2 ≤ a} := by sorry

end NUMINAMATH_CALUDE_part1_part2_l1412_141235


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_55_l1412_141249

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_55 :
  ∀ n : ℕ, is_four_digit n → is_divisible_by n 55 → n ≥ 1100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_55_l1412_141249


namespace NUMINAMATH_CALUDE_power_sum_equality_l1412_141276

theorem power_sum_equality : (-1)^45 + 2^(3^2 + 5^2 - 4^2) = 262143 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l1412_141276


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l1412_141262

theorem discounted_price_calculation (original_price discount_percentage : ℝ) 
  (h1 : original_price = 975)
  (h2 : discount_percentage = 20) : 
  original_price * (1 - discount_percentage / 100) = 780 := by
  sorry

end NUMINAMATH_CALUDE_discounted_price_calculation_l1412_141262


namespace NUMINAMATH_CALUDE_range_of_a_l1412_141200

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ x : ℝ, x ≥ a ∧ |x - 1| ≥ 1) → 
  a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1412_141200


namespace NUMINAMATH_CALUDE_polygon_contains_center_l1412_141298

/-- A convex polygon type -/
structure ConvexPolygon where
  area : ℝ
  isConvex : Bool

/-- A circle type -/
structure Circle where
  radius : ℝ

/-- Predicate to check if a polygon is inside a circle -/
def isInside (p : ConvexPolygon) (c : Circle) : Prop :=
  sorry

/-- Predicate to check if a polygon contains the center of a circle -/
def containsCenter (p : ConvexPolygon) (c : Circle) : Prop :=
  sorry

/-- Theorem statement -/
theorem polygon_contains_center (p : ConvexPolygon) (c : Circle) :
  p.area = 7 ∧ p.isConvex = true ∧ c.radius = 2 ∧ isInside p c → containsCenter p c :=
sorry

end NUMINAMATH_CALUDE_polygon_contains_center_l1412_141298


namespace NUMINAMATH_CALUDE_bagel_cost_proof_l1412_141291

/-- The cost of a dozen bagels when bought together -/
def dozen_cost : ℝ := 24

/-- The amount saved per bagel when buying a dozen -/
def savings_per_bagel : ℝ := 0.25

/-- The number of bagels in a dozen -/
def dozen : ℕ := 12

/-- The individual cost of a bagel -/
def individual_cost : ℝ := 2.25

theorem bagel_cost_proof :
  individual_cost = (dozen_cost + dozen * savings_per_bagel) / dozen :=
by sorry

end NUMINAMATH_CALUDE_bagel_cost_proof_l1412_141291


namespace NUMINAMATH_CALUDE_remainder_theorem_l1412_141285

-- Define the polynomial f(r) = r^15 - 3
def f (r : ℝ) : ℝ := r^15 - 3

-- Theorem statement
theorem remainder_theorem (r : ℝ) : 
  (f r) % (r - 2) = 32765 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1412_141285


namespace NUMINAMATH_CALUDE_hare_wins_by_10_meters_l1412_141274

-- Define the race parameters
def race_duration : ℕ := 50
def hare_initial_speed : ℕ := 12
def hare_later_speed : ℕ := 1
def tortoise_speed : ℕ := 3

-- Define the function to calculate the hare's distance
def hare_distance (initial_time : ℕ) : ℕ :=
  (initial_time * hare_initial_speed) + ((race_duration - initial_time) * hare_later_speed)

-- Define the function to calculate the tortoise's distance
def tortoise_distance : ℕ := race_duration * tortoise_speed

-- Theorem statement
theorem hare_wins_by_10_meters :
  ∃ (initial_time : ℕ), initial_time < race_duration ∧ 
  hare_distance initial_time = tortoise_distance + 10 :=
sorry

end NUMINAMATH_CALUDE_hare_wins_by_10_meters_l1412_141274


namespace NUMINAMATH_CALUDE_no_roots_of_equation_l1412_141216

theorem no_roots_of_equation (x : ℝ) (h : x ≠ 4) :
  ¬∃x, x - 9 / (x - 4) = 4 - 9 / (x - 4) :=
sorry

end NUMINAMATH_CALUDE_no_roots_of_equation_l1412_141216


namespace NUMINAMATH_CALUDE_problem_figure_area_l1412_141252

/-- A figure composed of square segments -/
structure SegmentedFigure where
  /-- The number of segments along one side of the square -/
  segments_per_side : ℕ
  /-- The length of each segment in cm -/
  segment_length : ℝ

/-- The area of a SegmentedFigure in cm² -/
def area (figure : SegmentedFigure) : ℝ :=
  (figure.segments_per_side * figure.segment_length) ^ 2

/-- The specific figure from the problem -/
def problem_figure : SegmentedFigure :=
  { segments_per_side := 3
  , segment_length := 3 }

theorem problem_figure_area :
  area problem_figure = 81 := by sorry

end NUMINAMATH_CALUDE_problem_figure_area_l1412_141252


namespace NUMINAMATH_CALUDE_ticket_cost_correct_l1412_141223

/-- The cost of one ticket for Sebastian's art exhibit -/
def ticket_cost : ℝ := 44

/-- The number of tickets Sebastian bought -/
def num_tickets : ℕ := 3

/-- The service fee for the online transaction -/
def service_fee : ℝ := 18

/-- The total amount Sebastian paid -/
def total_paid : ℝ := 150

/-- Theorem stating that the ticket cost is correct given the conditions -/
theorem ticket_cost_correct : 
  ticket_cost * num_tickets + service_fee = total_paid :=
by sorry

end NUMINAMATH_CALUDE_ticket_cost_correct_l1412_141223


namespace NUMINAMATH_CALUDE_relationship_fg_l1412_141243

noncomputable def f (x : ℝ) := Real.exp x + x - 2
noncomputable def g (x : ℝ) := Real.log x + x^2 - 3

theorem relationship_fg (a b : ℝ) (h1 : f a = 0) (h2 : g b = 0) :
  g a < 0 ∧ 0 < f b := by sorry

end NUMINAMATH_CALUDE_relationship_fg_l1412_141243


namespace NUMINAMATH_CALUDE_A_in_second_quadrant_implies_x_gt_5_l1412_141205

/-- A point in the second quadrant of the rectangular coordinate system -/
structure SecondQuadrantPoint where
  x : ℝ
  y : ℝ
  x_neg : x < 0
  y_pos : y > 0

/-- The point A with coordinates (6-2x, x-5) -/
def A (x : ℝ) : ℝ × ℝ := (6 - 2*x, x - 5)

/-- Theorem: If A(6-2x, x-5) is in the second quadrant, then x > 5 -/
theorem A_in_second_quadrant_implies_x_gt_5 :
  ∀ x : ℝ, (∃ p : SecondQuadrantPoint, A x = (p.x, p.y)) → x > 5 := by
  sorry

end NUMINAMATH_CALUDE_A_in_second_quadrant_implies_x_gt_5_l1412_141205


namespace NUMINAMATH_CALUDE_temperature_difference_l1412_141265

def highest_temp : Int := 9
def lowest_temp : Int := -1

theorem temperature_difference : highest_temp - lowest_temp = 10 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l1412_141265


namespace NUMINAMATH_CALUDE_translated_minimum_point_l1412_141203

-- Define the original function
def f (x : ℝ) : ℝ := |x + 1| - 4

-- Define the translated function
def g (x : ℝ) : ℝ := f (x - 3) + 4

-- Theorem statement
theorem translated_minimum_point :
  ∃ (x_min : ℝ), (∀ (x : ℝ), g x_min ≤ g x) ∧ g x_min = 0 ∧ x_min = 2 := by
  sorry

end NUMINAMATH_CALUDE_translated_minimum_point_l1412_141203


namespace NUMINAMATH_CALUDE_sum_of_cubes_counterexample_l1412_141208

theorem sum_of_cubes_counterexample : ¬∀ a : ℝ, (a + 1) * (a^2 - a + 1) = a^3 + 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_counterexample_l1412_141208


namespace NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l1412_141286

def monday_fabric : ℕ := 20
def fabric_cost : ℕ := 2
def wednesday_ratio : ℚ := 1/4
def total_earnings : ℕ := 140

theorem tuesday_to_monday_ratio :
  ∃ (tuesday_fabric : ℕ),
    (monday_fabric * fabric_cost + 
     tuesday_fabric * fabric_cost + 
     (wednesday_ratio * tuesday_fabric) * fabric_cost = total_earnings) ∧
    (tuesday_fabric = monday_fabric) := by
  sorry

end NUMINAMATH_CALUDE_tuesday_to_monday_ratio_l1412_141286


namespace NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l1412_141299

theorem polar_to_cartesian_conversion :
  let r : ℝ := 2
  let θ : ℝ := 2 * Real.pi / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -1) ∧ (y = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_conversion_l1412_141299


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1412_141227

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_dot_product : m * 1 + 1 * (n - 1) = 0) :
  ∃ (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_dot : x * 1 + 1 * (y - 1) = 0), 
    (1 / m + 1 / n ≥ 1 / x + 1 / y) ∧ (1 / x + 1 / y = 4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1412_141227


namespace NUMINAMATH_CALUDE_sequence_characterization_l1412_141270

/-- An infinite sequence of positive integers -/
def Sequence := ℕ → ℕ

/-- The property that the sequence is strictly increasing -/
def StrictlyIncreasing (a : Sequence) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

/-- The property that no three terms in the sequence sum to another term -/
def NoThreeSum (a : Sequence) : Prop :=
  ∀ i j k : ℕ, a i + a j ≠ a k

/-- The property that infinitely many terms of the sequence are of the form 2k - 1 -/
def InfinitelyManyOdd (a : Sequence) : Prop :=
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ a k = 2 * k - 1

/-- The main theorem: any sequence satisfying the given properties must be aₙ = 2n - 1 -/
theorem sequence_characterization (a : Sequence)
  (h1 : StrictlyIncreasing a)
  (h2 : NoThreeSum a)
  (h3 : InfinitelyManyOdd a) :
  ∀ n : ℕ, a n = 2 * n - 1 :=
by sorry

end NUMINAMATH_CALUDE_sequence_characterization_l1412_141270


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l1412_141234

theorem polynomial_identity_sum (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) :
  (∀ x : ℝ, x^8 - x^6 + x^4 - x^2 + 1 = (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃) * (x^2 + 1)) →
  a₁*d₁ + a₂*d₂ + a₃*d₃ = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l1412_141234


namespace NUMINAMATH_CALUDE_sin_2x_minus_pi_4_increasing_l1412_141253

open Real

theorem sin_2x_minus_pi_4_increasing (k : ℤ) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → 
  x₁ ∈ Set.Ioo (- π/8 + k*π) (3*π/8 + k*π) → 
  x₂ ∈ Set.Ioo (- π/8 + k*π) (3*π/8 + k*π) → 
  sin (2*x₁ - π/4) < sin (2*x₂ - π/4) := by
sorry

end NUMINAMATH_CALUDE_sin_2x_minus_pi_4_increasing_l1412_141253


namespace NUMINAMATH_CALUDE_x_power_four_plus_reciprocal_l1412_141281

theorem x_power_four_plus_reciprocal (x : ℝ) (h : x ≠ 0) :
  x^2 + (1/x^2) = 2 → x^4 + (1/x^4) = 2 := by
sorry

end NUMINAMATH_CALUDE_x_power_four_plus_reciprocal_l1412_141281


namespace NUMINAMATH_CALUDE_sand_loss_l1412_141206

theorem sand_loss (initial_sand final_sand : ℝ) 
  (h_initial : initial_sand = 4.1)
  (h_final : final_sand = 1.7) : 
  initial_sand - final_sand = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_sand_loss_l1412_141206


namespace NUMINAMATH_CALUDE_jason_remaining_cards_l1412_141241

def initial_cards : ℕ := 3
def cards_bought : ℕ := 2

theorem jason_remaining_cards : initial_cards - cards_bought = 1 := by
  sorry

end NUMINAMATH_CALUDE_jason_remaining_cards_l1412_141241


namespace NUMINAMATH_CALUDE_integral_x_over_sqrt_x_squared_plus_one_l1412_141209

theorem integral_x_over_sqrt_x_squared_plus_one (x : ℝ) :
  deriv (λ x => Real.sqrt (x^2 + 1)) x = x / Real.sqrt (x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_integral_x_over_sqrt_x_squared_plus_one_l1412_141209


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l1412_141257

theorem cyclic_sum_inequality (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq_3 : x + y + z = 3) :
  x^2 * y^2 + y^2 * z^2 + z^2 * x^2 < 3 + x*y + y*z + z*x := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l1412_141257


namespace NUMINAMATH_CALUDE_polo_shirt_cost_l1412_141230

/-- Calculates the total cost of two discounted polo shirts with sales tax -/
theorem polo_shirt_cost : 
  let regular_price : ℝ := 50
  let discount1 : ℝ := 0.4
  let discount2 : ℝ := 0.3
  let sales_tax : ℝ := 0.08
  let discounted_price1 := regular_price * (1 - discount1)
  let discounted_price2 := regular_price * (1 - discount2)
  let total_before_tax := discounted_price1 + discounted_price2
  let total_with_tax := total_before_tax * (1 + sales_tax)
  total_with_tax = 70.20 := by sorry

end NUMINAMATH_CALUDE_polo_shirt_cost_l1412_141230


namespace NUMINAMATH_CALUDE_work_distribution_l1412_141278

theorem work_distribution (total_work : ℝ) (h1 : total_work > 0) : 
  let top_20_percent_work := 0.8 * total_work
  let remaining_work := total_work - top_20_percent_work
  let next_20_percent_work := 0.25 * remaining_work
  ∃ (work_40_percent : ℝ), work_40_percent ≥ top_20_percent_work + next_20_percent_work ∧ 
                            work_40_percent / total_work ≥ 0.85 := by
  sorry

end NUMINAMATH_CALUDE_work_distribution_l1412_141278


namespace NUMINAMATH_CALUDE_false_proposition_l1412_141288

-- Define proposition p
def p : Prop := ∃ x : ℝ, (Real.cos x)^2 - (Real.sin x)^2 = 7

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.exp x > 0

-- Theorem statement
theorem false_proposition : ¬(¬p ∧ ¬q) := by
  sorry

end NUMINAMATH_CALUDE_false_proposition_l1412_141288


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l1412_141232

/-- Proves that the cost of an adult ticket is $12 given the conditions of the problem -/
theorem adult_ticket_cost (total_tickets : ℕ) (total_receipts : ℕ) (adult_tickets : ℕ) (child_ticket_cost : ℕ) :
  total_tickets = 130 →
  total_receipts = 840 →
  adult_tickets = 40 →
  child_ticket_cost = 4 →
  ∃ (adult_ticket_cost : ℕ),
    adult_ticket_cost * adult_tickets + child_ticket_cost * (total_tickets - adult_tickets) = total_receipts ∧
    adult_ticket_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l1412_141232


namespace NUMINAMATH_CALUDE_gauss_candy_remaining_l1412_141214

/-- The number of lollipops that remain after packaging -/
def remaining_lollipops (total : ℕ) (per_package : ℕ) : ℕ :=
  total % per_package

/-- Theorem stating the number of remaining lollipops for the Gauss Candy Company problem -/
theorem gauss_candy_remaining : remaining_lollipops 8362 12 = 10 := by
  sorry

end NUMINAMATH_CALUDE_gauss_candy_remaining_l1412_141214


namespace NUMINAMATH_CALUDE_min_edges_after_operations_l1412_141238

/-- A complete graph with n vertices. -/
structure CompleteGraph (n : ℕ) where
  vertices : Finset (Fin n)
  edges : Finset (Fin n × Fin n)
  complete : ∀ i j : Fin n, i ≠ j → (i, j) ∈ edges

/-- An elementary operation on a graph. -/
def elementaryOperation (G : CompleteGraph n) : CompleteGraph n → Prop :=
  sorry

/-- The result of applying any number of elementary operations. -/
def resultGraph (G : CompleteGraph n) : CompleteGraph n → Prop :=
  sorry

/-- The number of edges in a graph. -/
def numEdges (G : CompleteGraph n) : ℕ :=
  G.edges.card

theorem min_edges_after_operations (n : ℕ) (G : CompleteGraph n) (H : CompleteGraph n) :
  resultGraph G H → numEdges H ≥ n :=
  sorry

end NUMINAMATH_CALUDE_min_edges_after_operations_l1412_141238


namespace NUMINAMATH_CALUDE_atomic_weight_X_is_13_l1412_141201

/-- The atomic weight of element X in the compound H3XCOOH -/
def atomic_weight_X : ℝ :=
  let atomic_weight_H : ℝ := 1
  let atomic_weight_C : ℝ := 12
  let atomic_weight_O : ℝ := 16
  let molecular_weight : ℝ := 60
  molecular_weight - (3 * atomic_weight_H + atomic_weight_C + 3 * atomic_weight_O)

/-- Theorem stating that the atomic weight of X is 13 -/
theorem atomic_weight_X_is_13 : atomic_weight_X = 13 := by
  sorry

end NUMINAMATH_CALUDE_atomic_weight_X_is_13_l1412_141201


namespace NUMINAMATH_CALUDE_packetB_height_day10_l1412_141269

/-- Represents the growth rate of sunflowers --/
structure GrowthRate where
  x : ℝ  -- number of days since planting
  y : ℝ  -- daily average sunlight exposure (hours)
  W : ℝ  -- combined effect of competition and weather (0-10 scale)

/-- Calculates the growth rate for Packet A sunflowers --/
def growthRateA (r : GrowthRate) : ℝ := 2 * r.x + r.y - 0.1 * r.W

/-- Calculates the growth rate for Packet B sunflowers --/
def growthRateB (r : GrowthRate) : ℝ := 3 * r.x - r.y + 0.2 * r.W

/-- Theorem stating the height of Packet B sunflowers on day 10 --/
theorem packetB_height_day10 (r : GrowthRate) 
  (h1 : r.x = 10)
  (h2 : r.y = 6)
  (h3 : r.W = 5)
  (h4 : ∃ (hA hB : ℝ), hA = 192 ∧ hA = 1.2 * hB) :
  ∃ (hB : ℝ), hB = 160 := by
  sorry


end NUMINAMATH_CALUDE_packetB_height_day10_l1412_141269


namespace NUMINAMATH_CALUDE_intersection_in_first_quadrant_l1412_141284

theorem intersection_in_first_quadrant (k : ℝ) : 
  (∃ x y : ℝ, 
    y = k * x + 2 * k + 1 ∧ 
    y = -1/2 * x + 2 ∧ 
    x > 0 ∧ 
    y > 0) ↔ 
  -1/6 < k ∧ k < 1/2 :=
sorry

end NUMINAMATH_CALUDE_intersection_in_first_quadrant_l1412_141284


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_in_specific_pyramid_l1412_141293

/-- A pyramid with a regular hexagonal base and isosceles triangular lateral faces -/
structure HexagonalPyramid where
  base_side_length : ℝ
  lateral_face_height : ℝ

/-- A cube inscribed in a hexagonal pyramid -/
structure InscribedCube where
  pyramid : HexagonalPyramid
  -- Each vertex of the cube is either on the base or touches a point on the lateral faces

/-- The volume of an inscribed cube in a hexagonal pyramid -/
def inscribed_cube_volume (cube : InscribedCube) : ℝ :=
  sorry

theorem inscribed_cube_volume_in_specific_pyramid :
  ∀ (cube : InscribedCube),
    cube.pyramid.base_side_length = 2 →
    cube.pyramid.lateral_face_height = 3 →
    inscribed_cube_volume cube = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_in_specific_pyramid_l1412_141293


namespace NUMINAMATH_CALUDE_investment_interest_proof_l1412_141294

/-- Calculates the interest earned on an investment with annual compounding -/
def interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years - principal

/-- Proves that the interest earned on a $500 investment at 2% annual rate for 3 years is approximately $30.60 -/
theorem investment_interest_proof :
  let principal := 500
  let rate := 0.02
  let years := 3
  abs (interest_earned principal rate years - 30.60) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_proof_l1412_141294


namespace NUMINAMATH_CALUDE_optimal_purchasing_plan_l1412_141229

theorem optimal_purchasing_plan :
  let total_price : ℝ := 12
  let bulb_cost : ℝ := 30
  let motor_cost : ℝ := 45
  let total_items : ℕ := 90
  let bulb_price : ℝ := 3
  let motor_price : ℝ := 9
  let optimal_bulbs : ℕ := 30
  let optimal_motors : ℕ := 60
  let optimal_cost : ℝ := 630

  (∀ x y : ℕ, 
    x = 2 * y → 
    x * bulb_price = bulb_cost ∧ 
    y * motor_price = motor_cost) ∧
  
  (∀ m : ℕ,
    m ≤ total_items ∧
    m ≤ (total_items - m) / 2 →
    3 * m + 9 * (total_items - m) ≥ optimal_cost) ∧
  
  optimal_bulbs * bulb_price + optimal_motors * motor_price = optimal_cost :=
by sorry

end NUMINAMATH_CALUDE_optimal_purchasing_plan_l1412_141229


namespace NUMINAMATH_CALUDE_chessboard_touching_squares_probability_l1412_141246

/-- Represents a square on the chessboard -/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Checks if two squares are touching -/
def are_touching (s1 s2 : Square) : Prop :=
  (s1.row = s2.row ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row = s2.row ∧ s1.col.val = s2.col.val + 1) ∨
  (s1.col = s2.col ∧ s1.row.val + 1 = s2.row.val) ∨
  (s1.col = s2.col ∧ s1.row.val = s2.row.val + 1) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row.val + 1 = s2.row.val ∧ s1.col.val = s2.col.val + 1) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.row.val = s2.row.val + 1 ∧ s1.col.val = s2.col.val + 1)

/-- Checks if two squares are the same color -/
def same_color (s1 s2 : Square) : Prop :=
  (s1.row.val + s1.col.val) % 2 = (s2.row.val + s2.col.val) % 2

theorem chessboard_touching_squares_probability :
  ∀ (s1 s2 : Square), s1 ≠ s2 → are_touching s1 s2 → ¬(same_color s1 s2) :=
by sorry

end NUMINAMATH_CALUDE_chessboard_touching_squares_probability_l1412_141246


namespace NUMINAMATH_CALUDE_solution_pairs_l1412_141224

theorem solution_pairs : 
  ∀ x y : ℝ, x - y = 10 ∧ x^2 + y^2 = 100 → (x = 0 ∧ y = -10) ∨ (x = 10 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l1412_141224


namespace NUMINAMATH_CALUDE_train_crossing_time_l1412_141210

/-- Represents the problem of a train crossing a stationary train -/
theorem train_crossing_time
  (train_speed : Real)
  (pole_passing_time : Real)
  (stationary_train_length : Real)
  (h1 : train_speed = 72 * 1000 / 3600) -- 72 km/h converted to m/s
  (h2 : pole_passing_time = 10)
  (h3 : stationary_train_length = 500)
  : (train_speed * pole_passing_time + stationary_train_length) / train_speed = 35 := by
  sorry


end NUMINAMATH_CALUDE_train_crossing_time_l1412_141210


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1412_141215

-- Define the complex function f(x) = x^2
def f (x : ℂ) : ℂ := x^2

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Theorem statement
theorem complex_number_in_second_quadrant :
  let z := f (1 + i) / (3 + i)
  (z.re < 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1412_141215


namespace NUMINAMATH_CALUDE_additive_function_is_linear_l1412_141254

theorem additive_function_is_linear (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = f x + f y) :
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x :=
sorry

end NUMINAMATH_CALUDE_additive_function_is_linear_l1412_141254


namespace NUMINAMATH_CALUDE_sqrt_neg_two_squared_l1412_141263

theorem sqrt_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_neg_two_squared_l1412_141263


namespace NUMINAMATH_CALUDE_consecutive_primes_square_sum_prime_l1412_141212

/-- Definition of consecutive primes -/
def ConsecutivePrimes (p q r : Nat) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  ∃ (x y : Nat), p < x ∧ ¬Nat.Prime x ∧ x < q ∧
                 q < y ∧ ¬Nat.Prime y ∧ y < r

theorem consecutive_primes_square_sum_prime :
  ∀ p q r : Nat,
    ConsecutivePrimes p q r →
    Nat.Prime (p^2 + q^2 + r^2) →
    p = 3 ∧ q = 5 ∧ r = 7 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_primes_square_sum_prime_l1412_141212


namespace NUMINAMATH_CALUDE_power_division_l1412_141273

theorem power_division (n : ℕ) : (16^3018) / 8 = 2^9032 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l1412_141273


namespace NUMINAMATH_CALUDE_infinitely_many_increasing_prime_divisors_l1412_141222

-- Define w(n) as the number of different prime divisors of n
def w (n : Nat) : Nat :=
  (Nat.factors n).toFinset.card

-- Theorem statement
theorem infinitely_many_increasing_prime_divisors :
  ∃ (S : Set Nat), Set.Infinite S ∧ ∀ n ∈ S, w n < w (n + 1) ∧ w (n + 1) < w (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_increasing_prime_divisors_l1412_141222


namespace NUMINAMATH_CALUDE_min_cuts_for_3inch_to_1inch_cube_l1412_141297

/-- Represents a three-dimensional cube -/
structure Cube where
  side_length : ℕ

/-- Represents a cut on a cube -/
inductive Cut
  | plane : Cut

/-- The minimum number of cuts required to divide a cube into smaller cubes -/
def min_cuts (original : Cube) (target : Cube) : ℕ := sorry

/-- The number of smaller cubes that can be created from a larger cube -/
def num_smaller_cubes (original : Cube) (target : Cube) : ℕ := 
  (original.side_length / target.side_length) ^ 3

theorem min_cuts_for_3inch_to_1inch_cube : 
  let original := Cube.mk 3
  let target := Cube.mk 1
  min_cuts original target = 6 ∧ 
  num_smaller_cubes original target = 27 := by sorry

end NUMINAMATH_CALUDE_min_cuts_for_3inch_to_1inch_cube_l1412_141297


namespace NUMINAMATH_CALUDE_prob_same_school_adjacent_l1412_141271

/-- The number of students from the first school -/
def students_school1 : ℕ := 2

/-- The number of students from the second school -/
def students_school2 : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := students_school1 + students_school2

/-- The probability that students from the same school will be standing next to each other -/
def probability_same_school_adjacent : ℚ := 4/5

/-- Theorem stating that the probability of students from the same school standing next to each other is 4/5 -/
theorem prob_same_school_adjacent :
  probability_same_school_adjacent = 4/5 := by sorry

end NUMINAMATH_CALUDE_prob_same_school_adjacent_l1412_141271


namespace NUMINAMATH_CALUDE_positive_number_equality_l1412_141282

theorem positive_number_equality (x : ℝ) (h1 : x > 0) : 
  (2 / 3) * x = (144 / 216) * (1 / x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_equality_l1412_141282


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l1412_141228

theorem quadratic_roots_expression (x₁ x₂ : ℝ) : 
  x₁^2 - x₁ - 2 = 0 → x₂^2 - x₂ - 2 = 0 → (1 + x₁) + x₂ * (1 - x₁) = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l1412_141228


namespace NUMINAMATH_CALUDE_rectangular_field_width_l1412_141261

/-- The width of a rectangular field, given its length and a relationship between length and width. -/
def field_width (length : ℝ) (length_width_relation : ℝ → ℝ → Prop) : ℝ :=
  13.5

/-- Theorem stating that the width of a rectangular field is 13.5 meters, given specific conditions. -/
theorem rectangular_field_width :
  let length := 24
  let length_width_relation := λ l w => l = 2 * w - 3
  field_width length length_width_relation = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l1412_141261


namespace NUMINAMATH_CALUDE_square_inequality_l1412_141226

theorem square_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_inequality_l1412_141226


namespace NUMINAMATH_CALUDE_coefficient_sum_equals_15625_l1412_141289

theorem coefficient_sum_equals_15625 (b₆ b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 15625 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_equals_15625_l1412_141289


namespace NUMINAMATH_CALUDE_max_cylinder_radius_in_crate_l1412_141233

/-- A rectangular crate with given dimensions. -/
structure Crate where
  length : ℝ
  width : ℝ
  height : ℝ

/-- A right circular cylinder. -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Check if a cylinder fits in a crate when placed upright. -/
def cylinderFitsInCrate (cyl : Cylinder) (crate : Crate) : Prop :=
  cyl.radius * 2 ≤ min crate.length crate.width ∧
  cyl.height ≤ max crate.length (max crate.width crate.height)

/-- The theorem stating the maximum radius of a cylinder that fits in the given crate. -/
theorem max_cylinder_radius_in_crate :
  let crate := Crate.mk 5 8 12
  ∃ (max_radius : ℝ),
    max_radius = 2.5 ∧
    (∀ (r : ℝ), r > max_radius → ∃ (h : ℝ),
      ¬cylinderFitsInCrate (Cylinder.mk r h) crate) ∧
    (∀ (r : ℝ), r ≤ max_radius → ∃ (h : ℝ),
      cylinderFitsInCrate (Cylinder.mk r h) crate) :=
by sorry

end NUMINAMATH_CALUDE_max_cylinder_radius_in_crate_l1412_141233


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1412_141266

-- Define the sets M and P
def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def P : Set ℝ := {x : ℝ | x ≤ -1}

-- Theorem statement
theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x ∈ M ∩ P → x ∈ M ∪ P) ∧
  ¬(∀ x : ℝ, x ∈ M ∪ P → x ∈ M ∩ P) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1412_141266


namespace NUMINAMATH_CALUDE_min_value_expression_l1412_141213

theorem min_value_expression (m n : ℝ) (h : m - n^2 = 1) :
  ∀ x y : ℝ, x - y^2 = 1 → m^2 + 2*n^2 + 4*m - 1 ≤ x^2 + 2*y^2 + 4*x - 1 ∧
  ∃ a b : ℝ, a - b^2 = 1 ∧ a^2 + 2*b^2 + 4*a - 1 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1412_141213


namespace NUMINAMATH_CALUDE_rational_equation_solution_l1412_141260

theorem rational_equation_solution :
  ∃ x : ℝ, x ≠ 2 ∧ x ≠ (4/5 : ℝ) ∧
  (x^2 - 11*x + 24)/(x - 2) + (5*x^2 + 20*x - 40)/(5*x - 4) = -5 ∧
  x = -3 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l1412_141260


namespace NUMINAMATH_CALUDE_david_is_seven_l1412_141267

/-- David's age in years -/
def david_age : ℕ := sorry

/-- Yuan's age in years -/
def yuan_age : ℕ := sorry

/-- Yuan is 7 years older than David -/
axiom yuan_older : yuan_age = david_age + 7

/-- Yuan is twice David's age -/
axiom yuan_twice : yuan_age = 2 * david_age

theorem david_is_seven : david_age = 7 := by sorry

end NUMINAMATH_CALUDE_david_is_seven_l1412_141267


namespace NUMINAMATH_CALUDE_janet_video_game_lives_l1412_141255

theorem janet_video_game_lives : ∀ initial_lives : ℕ,
  initial_lives - 23 + 46 = 70 → initial_lives = 47 :=
by
  sorry

end NUMINAMATH_CALUDE_janet_video_game_lives_l1412_141255


namespace NUMINAMATH_CALUDE_square_sum_implies_fourth_power_sum_l1412_141258

theorem square_sum_implies_fourth_power_sum (r : ℝ) (h : (r + 1/r)^2 = 7) : 
  r^4 + 1/r^4 = 23 := by
sorry

end NUMINAMATH_CALUDE_square_sum_implies_fourth_power_sum_l1412_141258


namespace NUMINAMATH_CALUDE_tranquility_essence_l1412_141231

/-- Represents the philosophical concepts in the problem --/
structure PhilosophicalConcept where
  opposingAndUnified : Bool  -- The sides of a contradiction are both opposing and unified
  struggleWithinUnity : Bool -- The nature of struggle is embedded within unity
  differencesBasedOnUnity : Bool -- Differences and opposition are based on unity
  motionCharacteristic : Bool -- Motion is the only characteristic of matter

/-- Represents a painting with its elements --/
structure Painting where
  hasWaterfall : Bool
  hasTree : Bool
  hasBirdNest : Bool
  hasSleepingBird : Bool

/-- Defines the essence of tranquility based on philosophical concepts --/
def essenceOfTranquility (p : Painting) (c : PhilosophicalConcept) : Prop :=
  p.hasWaterfall ∧ p.hasTree ∧ p.hasBirdNest ∧ p.hasSleepingBird ∧
  c.opposingAndUnified ∧ c.struggleWithinUnity ∧
  ¬c.differencesBasedOnUnity ∧ ¬c.motionCharacteristic

/-- The theorem to be proved --/
theorem tranquility_essence (p : Painting) (c : PhilosophicalConcept) :
  p.hasWaterfall ∧ p.hasTree ∧ p.hasBirdNest ∧ p.hasSleepingBird →
  c.opposingAndUnified ∧ c.struggleWithinUnity →
  essenceOfTranquility p c := by
  sorry


end NUMINAMATH_CALUDE_tranquility_essence_l1412_141231


namespace NUMINAMATH_CALUDE_max_profit_and_break_even_l1412_141295

/-- Revenue function (in ten thousand yuan) -/
def R (x : ℝ) : ℝ := 5 * x - x^2

/-- Cost function (in ten thousand yuan) -/
def C (x : ℝ) : ℝ := 0.5 + 0.25 * x

/-- Profit function (in ten thousand yuan) -/
def profit (x : ℝ) : ℝ := R x - C x

/-- Annual demand in hundreds of units -/
def annual_demand : ℝ := 5

theorem max_profit_and_break_even :
  ∃ (max_profit_units : ℝ) (break_even_lower break_even_upper : ℝ),
    (∀ x, 0 ≤ x → x ≤ annual_demand → profit x ≤ profit max_profit_units) ∧
    (max_profit_units = 4.75) ∧
    (break_even_lower = 0.1) ∧
    (break_even_upper = 48) ∧
    (∀ x, break_even_lower ≤ x → x ≤ break_even_upper → profit x ≥ 0) :=
  sorry

end NUMINAMATH_CALUDE_max_profit_and_break_even_l1412_141295


namespace NUMINAMATH_CALUDE_min_c_value_l1412_141242

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! (x y : ℝ), 2*x + y = 2019 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1010 :=
sorry

end NUMINAMATH_CALUDE_min_c_value_l1412_141242


namespace NUMINAMATH_CALUDE_cupcakes_left_l1412_141292

/-- The number of cupcakes in a dozen -/
def dozen : ℕ := 12

/-- The number of cupcakes Dani brings -/
def cupcakes_brought : ℕ := (5 * dozen) / 2

/-- The initial number of people in the class -/
def initial_class_size : ℕ := 27 + 1 + 1

/-- The number of students absent -/
def absent_students : ℕ := 3

/-- The actual number of people present in the class -/
def class_size : ℕ := initial_class_size - absent_students

theorem cupcakes_left : cupcakes_brought - class_size = 4 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_left_l1412_141292


namespace NUMINAMATH_CALUDE_solution_of_equation_l1412_141277

theorem solution_of_equation :
  let f (x : ℝ) := 
    8 / (Real.sqrt (x - 10) - 10) + 
    2 / (Real.sqrt (x - 10) - 5) + 
    9 / (Real.sqrt (x - 10) + 5) + 
    16 / (Real.sqrt (x - 10) + 10)
  ∀ x : ℝ, f x = 0 ↔ x = 1841 / 121 ∨ x = 190 / 9 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l1412_141277


namespace NUMINAMATH_CALUDE_lisa_walks_2100_meters_l1412_141221

/-- Calculates the total distance Lisa walks over two days given her usual pace and terrain conditions -/
def lisa_total_distance (usual_pace : ℝ) : ℝ :=
  let day1_morning := usual_pace * 60
  let day1_evening := (usual_pace * 0.7) * 60
  let day2_morning := (usual_pace * 1.2) * 60
  let day2_evening := (usual_pace * 0.6) * 60
  day1_morning + day1_evening + day2_morning + day2_evening

/-- Theorem stating that Lisa's total distance over two days is 2100 meters -/
theorem lisa_walks_2100_meters :
  lisa_total_distance 10 = 2100 := by
  sorry

#eval lisa_total_distance 10

end NUMINAMATH_CALUDE_lisa_walks_2100_meters_l1412_141221


namespace NUMINAMATH_CALUDE_triangle_division_theorem_l1412_141237

/-- Represents a triangle with two angles α and β --/
structure Triangle where
  α : ℝ
  β : ℝ
  angle_sum : α + β < π / 2

/-- Predicate to check if two triangles are similar --/
def similar (t1 t2 : Triangle) : Prop := sorry

/-- Predicate to check if a triangle can be divided into a list of triangles --/
def can_be_divided_into (t : Triangle) (ts : List Triangle) : Prop := sorry

/-- The main theorem --/
theorem triangle_division_theorem (n : ℕ) (h : n ≥ 2) :
  ∃ (ts : List Triangle),
    ts.length = n ∧
    (∀ i j, i ≠ j → ¬similar (ts.get i) (ts.get j)) ∧
    (∀ t ∈ ts, ∃ (subts : List Triangle),
      subts.length = n ∧
      can_be_divided_into t subts ∧
      (∀ i j, i ≠ j → ¬similar (subts.get i) (subts.get j)) ∧
      (∀ subt ∈ subts, ∃ t' ∈ ts, similar subt t')) :=
sorry

end NUMINAMATH_CALUDE_triangle_division_theorem_l1412_141237


namespace NUMINAMATH_CALUDE_probability_red_or_blue_specific_l1412_141279

/-- Represents the probability of drawing a red or blue marble after a previous draw -/
def probability_red_or_blue (red blue yellow : ℕ) : ℚ :=
  let total := red + blue + yellow
  let p_yellow := yellow / total
  let p_not_yellow := 1 - p_yellow
  let p_red_or_blue_after_yellow := (red + blue) / (total - 1)
  let p_red_or_blue_after_not_yellow := (red + blue) / total
  p_yellow * p_red_or_blue_after_yellow + p_not_yellow * p_red_or_blue_after_not_yellow

/-- Theorem stating the probability of drawing a red or blue marble
    after a previous draw from a bag with 4 red, 3 blue, and 6 yellow marbles -/
theorem probability_red_or_blue_specific :
  probability_red_or_blue 4 3 6 = 91 / 169 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_or_blue_specific_l1412_141279


namespace NUMINAMATH_CALUDE_fraction_addition_l1412_141256

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l1412_141256
