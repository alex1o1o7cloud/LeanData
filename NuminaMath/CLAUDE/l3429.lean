import Mathlib

namespace NUMINAMATH_CALUDE_count_pairs_eq_28_l3429_342979

def count_pairs : ℕ :=
  (Finset.range 7).sum (λ m =>
    (Finset.range (8 - m)).card)

theorem count_pairs_eq_28 : count_pairs = 28 := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_28_l3429_342979


namespace NUMINAMATH_CALUDE_intersection_point_l3429_342911

theorem intersection_point (a : ℝ) :
  (∃! p : ℝ × ℝ, (p.2 = a * p.1 + a ∧ p.2 = p.1 ∧ p.2 = 2 - 2 * a * p.1)) ↔ (a = 1/2 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l3429_342911


namespace NUMINAMATH_CALUDE_xyz_product_l3429_342954

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 168)
  (h2 : y * (z + x) = 186)
  (h3 : z * (x + y) = 194) :
  x * y * z = 860 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l3429_342954


namespace NUMINAMATH_CALUDE_log_properties_l3429_342938

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- Main theorem
theorem log_properties (b : ℝ) (x : ℝ) (y : ℝ) 
    (h1 : b > 1) 
    (h2 : y = log b (x^2)) :
  (x = 1 → y = 0) ∧ 
  (x = -b → y = 2) ∧ 
  (-1 < x ∧ x < 1 → y < 0) := by
  sorry

end NUMINAMATH_CALUDE_log_properties_l3429_342938


namespace NUMINAMATH_CALUDE_sum_gcd_lcm_6_15_30_l3429_342981

def gcd_three (a b c : ℕ) : ℕ := Nat.gcd a (Nat.gcd b c)

def lcm_three (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem sum_gcd_lcm_6_15_30 :
  gcd_three 6 15 30 + lcm_three 6 15 30 = 33 := by
  sorry

end NUMINAMATH_CALUDE_sum_gcd_lcm_6_15_30_l3429_342981


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3429_342986

theorem inequality_solution_set (x : ℝ) (h : x ≠ 1) :
  (1 / (x - 1) ≤ 1) ↔ (x < 1 ∨ x ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3429_342986


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l3429_342963

theorem factorization_cubic_minus_linear (a : ℝ) : 
  a^3 - 4*a = a*(a+2)*(a-2) := by sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l3429_342963


namespace NUMINAMATH_CALUDE_placard_distribution_l3429_342970

theorem placard_distribution (total_placards : ℕ) (total_people : ℕ) 
  (h1 : total_placards = 823) 
  (h2 : total_people = 412) :
  (total_placards : ℚ) / total_people = 2 := by
sorry

end NUMINAMATH_CALUDE_placard_distribution_l3429_342970


namespace NUMINAMATH_CALUDE_largest_base5_five_digit_to_base10_l3429_342928

/-- Converts a base-5 number to base-10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ (digits.length - 1 - i))) 0

/-- The largest five-digit number in base 5 --/
def largestBase5FiveDigit : List Nat := [4, 4, 4, 4, 4]

theorem largest_base5_five_digit_to_base10 :
  base5ToBase10 largestBase5FiveDigit = 3124 := by
  sorry

#eval base5ToBase10 largestBase5FiveDigit

end NUMINAMATH_CALUDE_largest_base5_five_digit_to_base10_l3429_342928


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l3429_342951

/-- Calculates the average speed of a round trip given the specified conditions -/
theorem round_trip_average_speed
  (outbound_distance : ℝ)
  (outbound_time : ℝ)
  (return_distance : ℝ)
  (return_speed : ℝ)
  (h1 : outbound_distance = 5)
  (h2 : outbound_time = 1)
  (h3 : return_distance = outbound_distance)
  (h4 : return_speed = 20)
  : (outbound_distance + return_distance) / (outbound_time + return_distance / return_speed) = 8 := by
  sorry

#check round_trip_average_speed

end NUMINAMATH_CALUDE_round_trip_average_speed_l3429_342951


namespace NUMINAMATH_CALUDE_light_bulbs_theorem_l3429_342993

/-- The number of light bulbs in the kitchen -/
def kitchen_bulbs : ℕ := 35

/-- The fraction of broken light bulbs in the kitchen -/
def kitchen_broken_fraction : ℚ := 3/5

/-- The number of broken light bulbs in the foyer -/
def foyer_broken : ℕ := 10

/-- The fraction of broken light bulbs in the foyer -/
def foyer_broken_fraction : ℚ := 1/3

/-- The total number of unbroken light bulbs in both the foyer and kitchen -/
def total_unbroken : ℕ := 34

theorem light_bulbs_theorem : 
  kitchen_bulbs * (1 - kitchen_broken_fraction) + 
  (foyer_broken / foyer_broken_fraction) * (1 - foyer_broken_fraction) = total_unbroken := by
sorry

end NUMINAMATH_CALUDE_light_bulbs_theorem_l3429_342993


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l3429_342964

theorem trig_expression_simplification :
  (Real.sin (20 * π / 180) * Real.cos (15 * π / 180) + Real.cos (160 * π / 180) * Real.cos (105 * π / 180)) /
  (Real.sin (25 * π / 180) * Real.cos (10 * π / 180) + Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) =
  Real.sin (5 * π / 180) / Real.sin (15 * π / 180) := by
sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l3429_342964


namespace NUMINAMATH_CALUDE_tony_puzzle_time_l3429_342942

/-- Calculates the total time spent solving puzzles given the time for a warm-up puzzle
    and the number and relative duration of additional puzzles. -/
def total_puzzle_time (warm_up_time : ℕ) (num_additional_puzzles : ℕ) (additional_puzzle_factor : ℕ) : ℕ :=
  warm_up_time + num_additional_puzzles * (warm_up_time * additional_puzzle_factor)

/-- Proves that given the specific conditions of Tony's puzzle-solving session,
    the total time spent is 70 minutes. -/
theorem tony_puzzle_time :
  total_puzzle_time 10 2 3 = 70 := by
  sorry

end NUMINAMATH_CALUDE_tony_puzzle_time_l3429_342942


namespace NUMINAMATH_CALUDE_square_root_problem_l3429_342921

theorem square_root_problem (m n : ℝ) 
  (h1 : (5*m - 2)^(1/3) = -3) 
  (h2 : Real.sqrt (3*m + 2*n - 1) = 4) : 
  Real.sqrt (2*m + n + 10) = 4 ∨ Real.sqrt (2*m + n + 10) = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l3429_342921


namespace NUMINAMATH_CALUDE_divisibility_by_eleven_l3429_342926

def seven_digit_number (n : ℕ) : ℕ := 7010000 + n * 1000 + 864

theorem divisibility_by_eleven (n : ℕ) :
  (seven_digit_number n) % 11 = 0 ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_eleven_l3429_342926


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3429_342907

/-- The quadratic function f(x) = 2(x-1)^2 - 3 -/
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 - 3

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := -3

/-- Theorem: The vertex of the quadratic function f(x) = 2(x-1)^2 - 3 is (1, -3) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3429_342907


namespace NUMINAMATH_CALUDE_estimate_shaded_area_l3429_342967

/-- Estimates the area of a shaded region within a square using Monte Carlo method. -/
theorem estimate_shaded_area (side_length : ℝ) (total_points : ℕ) (shaded_points : ℕ) : 
  side_length = 6 →
  total_points = 800 →
  shaded_points = 200 →
  (shaded_points : ℝ) / (total_points : ℝ) * side_length^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_estimate_shaded_area_l3429_342967


namespace NUMINAMATH_CALUDE_binomial_15_13_l3429_342944

theorem binomial_15_13 : Nat.choose 15 13 = 105 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_13_l3429_342944


namespace NUMINAMATH_CALUDE_planes_parallel_if_lines_perpendicular_and_parallel_l3429_342929

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_lines_perpendicular_and_parallel
  (m n : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : perpendicular n β)
  (h3 : parallel_lines m n) :
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_lines_perpendicular_and_parallel_l3429_342929


namespace NUMINAMATH_CALUDE_multiply_polynomial_equality_l3429_342991

theorem multiply_polynomial_equality (x : ℝ) :
  (x^6 + 27*x^3 + 729) * (x^3 - 27) = x^12 + 27*x^9 - 19683*x^3 - 531441 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomial_equality_l3429_342991


namespace NUMINAMATH_CALUDE_max_profit_at_upper_bound_l3429_342950

/-- Represents the profit function for a product given its cost price, initial selling price,
    initial daily sales, and the rate of sales decrease per yuan increase in price. -/
def profit_function (cost_price : ℝ) (initial_price : ℝ) (initial_sales : ℝ) (sales_decrease_rate : ℝ) (x : ℝ) : ℝ :=
  (x - cost_price) * (initial_sales - (x - initial_price) * sales_decrease_rate)

/-- Theorem stating the maximum profit and the price at which it occurs -/
theorem max_profit_at_upper_bound (cost_price : ℝ) (initial_price : ℝ) (initial_sales : ℝ) 
    (sales_decrease_rate : ℝ) (lower_bound : ℝ) (upper_bound : ℝ) :
    cost_price = 30 →
    initial_price = 40 →
    initial_sales = 600 →
    sales_decrease_rate = 10 →
    lower_bound = 40 →
    upper_bound = 60 →
    (∀ x, lower_bound ≤ x ∧ x ≤ upper_bound →
      profit_function cost_price initial_price initial_sales sales_decrease_rate x ≤
      profit_function cost_price initial_price initial_sales sales_decrease_rate upper_bound) ∧
    profit_function cost_price initial_price initial_sales sales_decrease_rate upper_bound = 12000 :=
  sorry

#check max_profit_at_upper_bound

end NUMINAMATH_CALUDE_max_profit_at_upper_bound_l3429_342950


namespace NUMINAMATH_CALUDE_arrangement_theorems_l3429_342925

/-- The number of men in the group -/
def num_men : ℕ := 6

/-- The number of women in the group -/
def num_women : ℕ := 4

/-- The total number of people in the group -/
def total_people : ℕ := num_men + num_women

/-- Calculate the number of arrangements with no two women next to each other -/
def arrangements_no_adjacent_women : ℕ := sorry

/-- Calculate the number of arrangements with Man A not first and Man B not last -/
def arrangements_a_not_first_b_not_last : ℕ := sorry

/-- Calculate the number of arrangements with fixed order of Men A, B, and C -/
def arrangements_fixed_abc : ℕ := sorry

/-- Calculate the number of arrangements with Man A to the left of Man B -/
def arrangements_a_left_of_b : ℕ := sorry

theorem arrangement_theorems :
  (arrangements_no_adjacent_women = num_men.factorial * (num_women.choose (num_men + 1))) ∧
  (arrangements_a_not_first_b_not_last = total_people.factorial - 2 * (total_people - 1).factorial + (total_people - 2).factorial) ∧
  (arrangements_fixed_abc = total_people.factorial / 6) ∧
  (arrangements_a_left_of_b = total_people.factorial / 2) := by sorry

end NUMINAMATH_CALUDE_arrangement_theorems_l3429_342925


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3429_342971

theorem inequality_system_solution (x : ℝ) :
  (1 / x < 3) ∧ (1 / x > -4) ∧ (x^2 - 3*x + 2 < 0) → (1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3429_342971


namespace NUMINAMATH_CALUDE_vertex_D_coordinates_l3429_342931

/-- A parallelogram with vertices A, B, C, and D in 2D space. -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- The given parallelogram ABCD with specified coordinates for A, B, and C. -/
def givenParallelogram : Parallelogram where
  A := (0, 0)
  B := (1, 2)
  C := (3, 1)
  D := (2, -1)  -- We include D here, but will prove it's correct

/-- Theorem stating that the coordinates of vertex D in the given parallelogram are (2, -1). -/
theorem vertex_D_coordinates (p : Parallelogram) (h : p = givenParallelogram) :
  p.D = (2, -1) := by
  sorry

end NUMINAMATH_CALUDE_vertex_D_coordinates_l3429_342931


namespace NUMINAMATH_CALUDE_triangle_third_angle_l3429_342937

theorem triangle_third_angle (a b c : ℝ) (ha : a = 70) (hb : b = 50) 
  (sum_of_angles : a + b + c = 180) : c = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_angle_l3429_342937


namespace NUMINAMATH_CALUDE_contrapositive_of_p_is_true_l3429_342961

theorem contrapositive_of_p_is_true :
  (∀ x : ℝ, x^2 - 2*x - 8 ≤ 0 → x ≥ -3) := by sorry

end NUMINAMATH_CALUDE_contrapositive_of_p_is_true_l3429_342961


namespace NUMINAMATH_CALUDE_ben_total_items_is_27_1_l3429_342943

/-- The number of new clothing items for Ben -/
def ben_total_items (alex_shirts alex_pants alex_shoes alex_hats alex_jackets : ℝ)
  (joe_shirts_diff joe_pants_diff joe_hats_diff joe_jackets_diff : ℝ)
  (ben_shirts_diff ben_pants_diff ben_shoes_diff ben_hats_diff ben_jackets_diff : ℝ) : ℝ :=
  let joe_shirts := alex_shirts + joe_shirts_diff
  let joe_pants := alex_pants + joe_pants_diff
  let joe_shoes := alex_shoes
  let joe_hats := alex_hats + joe_hats_diff
  let joe_jackets := alex_jackets + joe_jackets_diff
  let ben_shirts := joe_shirts + ben_shirts_diff
  let ben_pants := alex_pants + ben_pants_diff
  let ben_shoes := joe_shoes + ben_shoes_diff
  let ben_hats := alex_hats + ben_hats_diff
  let ben_jackets := joe_jackets + ben_jackets_diff
  ben_shirts + ben_pants + ben_shoes + ben_hats + ben_jackets

/-- Theorem stating that Ben has 27.1 total new clothing items -/
theorem ben_total_items_is_27_1 :
  ben_total_items 4.5 3 2.5 1.5 2 3.5 (-2.5) 0.3 (-1) 5.3 5.5 (-1.7) 0.5 1.5 = 27.1 := by
  sorry

end NUMINAMATH_CALUDE_ben_total_items_is_27_1_l3429_342943


namespace NUMINAMATH_CALUDE_open_box_volume_is_5120_l3429_342983

/-- The volume of an open box formed by cutting squares from a rectangular sheet. -/
def open_box_volume (sheet_length sheet_width cut_side : ℝ) : ℝ :=
  (sheet_length - 2 * cut_side) * (sheet_width - 2 * cut_side) * cut_side

/-- Theorem: The volume of the open box is 5120 m³ -/
theorem open_box_volume_is_5120 :
  open_box_volume 48 36 8 = 5120 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_is_5120_l3429_342983


namespace NUMINAMATH_CALUDE_marble_jar_count_l3429_342957

theorem marble_jar_count :
  ∀ (total blue red green yellow : ℕ),
    2 * blue = total →
    4 * red = total →
    green = 27 →
    yellow = 14 →
    blue + red + green + yellow = total →
    total = 164 := by
  sorry

end NUMINAMATH_CALUDE_marble_jar_count_l3429_342957


namespace NUMINAMATH_CALUDE_index_card_area_l3429_342923

theorem index_card_area (length width : ℝ) 
  (h1 : length = 5 ∧ width = 7)
  (h2 : ∃ (shortened_side : ℝ), 
    (shortened_side = length - 2 ∨ shortened_side = width - 2) ∧
    shortened_side * (if shortened_side = length - 2 then width else length) = 21) :
  length * (width - 1) = 30 :=
by sorry

end NUMINAMATH_CALUDE_index_card_area_l3429_342923


namespace NUMINAMATH_CALUDE_volunteer_selection_l3429_342978

/-- The number of ways to select 3 volunteers from 5, with at most one of A and B --/
def select_volunteers (total : ℕ) (to_select : ℕ) (special : ℕ) : ℕ :=
  Nat.choose total to_select - Nat.choose (total - special) (to_select - special)

/-- Theorem stating that selecting 3 from 5 with at most one of two special volunteers results in 7 ways --/
theorem volunteer_selection :
  select_volunteers 5 3 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_selection_l3429_342978


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3429_342989

theorem fraction_equivalence 
  (a b d k : ℝ) 
  (h1 : d ≠ 0) 
  (h2 : k ≠ 0) : 
  (∀ x, (a * (k * x) + b) / (a * (k * x) + d) = (b * (k * x)) / (d * (k * x))) ↔ b = d :=
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3429_342989


namespace NUMINAMATH_CALUDE_trapezoid_segment_ratio_l3429_342912

/-- A trapezoid with specific properties -/
structure Trapezoid where
  upperLength : ℝ
  lowerLength : ℝ
  smallSegment : ℝ
  largeSegment : ℝ
  upperEquation : 3 * smallSegment + largeSegment = upperLength
  lowerEquation : 2 * largeSegment + 6 * smallSegment = lowerLength

/-- The ratio of the largest to smallest segment in a specific trapezoid is 2 -/
theorem trapezoid_segment_ratio (t : Trapezoid) 
    (h1 : t.upperLength = 1) 
    (h2 : t.lowerLength = 2) : 
  t.largeSegment / t.smallSegment = 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_ratio_l3429_342912


namespace NUMINAMATH_CALUDE_inequality_region_l3429_342917

theorem inequality_region (x y : ℝ) : 
  x + 3*y - 1 < 0 → (x < 1 - 3*y) ∧ (y < (1 - x)/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_region_l3429_342917


namespace NUMINAMATH_CALUDE_complement_connected_if_not_connected_l3429_342946

-- Define a graph
def Graph := Type

-- Define the property of being connected
def is_connected (G : Graph) : Prop := sorry

-- Define the complement of a graph
def complement (G : Graph) : Graph := sorry

-- Theorem statement
theorem complement_connected_if_not_connected (G : Graph) :
  ¬(is_connected G) → is_connected (complement G) := by sorry

end NUMINAMATH_CALUDE_complement_connected_if_not_connected_l3429_342946


namespace NUMINAMATH_CALUDE_quadratic_roots_expression_l3429_342987

theorem quadratic_roots_expression (x₁ x₂ : ℝ) : 
  x₁^2 + 5*x₁ + 1 = 0 →
  x₂^2 + 5*x₂ + 1 = 0 →
  (x₁*Real.sqrt 6 / (1 + x₂))^2 + (x₂*Real.sqrt 6 / (1 + x₁))^2 = 220 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_expression_l3429_342987


namespace NUMINAMATH_CALUDE_intersection_counts_theorem_l3429_342914

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

end NUMINAMATH_CALUDE_intersection_counts_theorem_l3429_342914


namespace NUMINAMATH_CALUDE_min_abs_z_is_zero_l3429_342933

theorem min_abs_z_is_zero (z : ℂ) (h : Complex.abs (z + 2 - 3*I) + Complex.abs (z - 2*I) = 7) :
  ∃ (w : ℂ), ∀ (z : ℂ), Complex.abs (z + 2 - 3*I) + Complex.abs (z - 2*I) = 7 → Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 0 :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_is_zero_l3429_342933


namespace NUMINAMATH_CALUDE_jerry_current_average_l3429_342973

/-- Jerry's current average score on the first 3 tests -/
def current_average : ℝ := sorry

/-- Jerry's score on the fourth test -/
def fourth_test_score : ℝ := 93

/-- The increase in average score after the fourth test -/
def average_increase : ℝ := 2

theorem jerry_current_average : 
  (current_average * 3 + fourth_test_score) / 4 = current_average + average_increase ∧ 
  current_average = 85 := by sorry

end NUMINAMATH_CALUDE_jerry_current_average_l3429_342973


namespace NUMINAMATH_CALUDE_biology_group_specimen_exchange_l3429_342982

/-- Represents the number of specimens exchanged in a biology interest group --/
def specimens_exchanged (x : ℕ) : ℕ := x * (x - 1)

/-- Theorem stating that the equation x(x-1) = 110 correctly represents the situation --/
theorem biology_group_specimen_exchange (x : ℕ) :
  specimens_exchanged x = 110 ↔ x * (x - 1) = 110 := by
  sorry

end NUMINAMATH_CALUDE_biology_group_specimen_exchange_l3429_342982


namespace NUMINAMATH_CALUDE_five_twos_to_one_to_five_l3429_342930

theorem five_twos_to_one_to_five :
  ∃ (a b c d e : ℕ → ℕ → ℕ → ℕ → ℕ → ℚ),
    (∀ x y z w v, x = 2 ∧ y = 2 ∧ z = 2 ∧ w = 2 ∧ v = 2 →
      a x y z w v = 1 ∧
      b x y z w v = 2 ∧
      c x y z w v = 3 ∧
      d x y z w v = 4 ∧
      e x y z w v = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_five_twos_to_one_to_five_l3429_342930


namespace NUMINAMATH_CALUDE_circumradius_inequality_equality_condition_l3429_342948

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

def circumradius (t : Triangle) : ℝ := sorry

theorem circumradius_inequality (t : Triangle) :
  circumradius t ≥ (t.a^2 + t.b^2) / (2 * Real.sqrt (2 * t.a^2 + 2 * t.b^2 - t.c^2)) :=
sorry

theorem equality_condition (t : Triangle) :
  circumradius t = (t.a^2 + t.b^2) / (2 * Real.sqrt (2 * t.a^2 + 2 * t.b^2 - t.c^2)) ↔
  (t.a = t.b ∨ t.a^2 + t.b^2 = t.c^2) :=
sorry

end NUMINAMATH_CALUDE_circumradius_inequality_equality_condition_l3429_342948


namespace NUMINAMATH_CALUDE_sin_cos_sum_21_39_l3429_342910

theorem sin_cos_sum_21_39 : 
  Real.sin (21 * π / 180) * Real.cos (39 * π / 180) + 
  Real.cos (21 * π / 180) * Real.sin (39 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_21_39_l3429_342910


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_a_greater_than_one_l3429_342990

theorem intersection_nonempty_iff_a_greater_than_one (a : ℝ) :
  ({x : ℝ | x > 1} ∩ {x : ℝ | x ≤ a}).Nonempty ↔ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_a_greater_than_one_l3429_342990


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l3429_342977

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 36) : x + y ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l3429_342977


namespace NUMINAMATH_CALUDE_right_triangle_vector_relation_l3429_342945

/-- Given a right triangle ABC with ∠C = 90°, vector AB = (t, 1), and vector AC = (2, 3), prove that t = 5 -/
theorem right_triangle_vector_relation (t : ℝ) : 
  let A : ℝ × ℝ := (0, 0)  -- Assuming A is at the origin for simplicity
  let B : ℝ × ℝ := (t, 1)
  let C : ℝ × ℝ := (2, 3)
  let AB : ℝ × ℝ := (t - 0, 1 - 0)  -- Vector from A to B
  let AC : ℝ × ℝ := (2 - 0, 3 - 0)  -- Vector from A to C
  let BC : ℝ × ℝ := (2 - t, 3 - 1)  -- Vector from B to C
  (AC.1 * BC.1 + AC.2 * BC.2 = 0) →  -- Dot product of AC and BC is 0 (perpendicular)
  t = 5 := by
sorry


end NUMINAMATH_CALUDE_right_triangle_vector_relation_l3429_342945


namespace NUMINAMATH_CALUDE_f_lower_bound_a_range_l3429_342904

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a^2| + |x - a - 1|

-- Theorem 1: f(x) ≥ 3/4 for all x and a
theorem f_lower_bound (x a : ℝ) : f x a ≥ 3/4 := by
  sorry

-- Theorem 2: If f(4) < 13, then -2 < a < 3
theorem a_range (a : ℝ) : f 4 a < 13 → -2 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_f_lower_bound_a_range_l3429_342904


namespace NUMINAMATH_CALUDE_tournament_handshakes_count_l3429_342909

/-- The number of unique handshakes in a tournament with 4 teams of 2 players each,
    where each player shakes hands once with every other player except their partner. -/
def tournament_handshakes : ℕ :=
  let total_players : ℕ := 4 * 2
  let handshakes_per_player : ℕ := total_players - 2
  (total_players * handshakes_per_player) / 2

theorem tournament_handshakes_count : tournament_handshakes = 24 := by
  sorry

end NUMINAMATH_CALUDE_tournament_handshakes_count_l3429_342909


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3429_342984

theorem quadratic_inequality (x : ℝ) : -9*x^2 + 6*x + 15 > 0 ↔ -1 < x ∧ x < 5/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3429_342984


namespace NUMINAMATH_CALUDE_A_and_D_mutually_exclusive_but_not_complementary_l3429_342919

-- Define the sample space for a fair six-sided die
def DieOutcome := Fin 6

-- Define the events
def event_A (n : DieOutcome) : Prop := n.val % 2 = 1
def event_B (n : DieOutcome) : Prop := n.val % 2 = 0
def event_C (n : DieOutcome) : Prop := n.val % 2 = 0
def event_D (n : DieOutcome) : Prop := n.val = 2 ∨ n.val = 4

-- Define mutual exclusivity
def mutually_exclusive (e1 e2 : DieOutcome → Prop) : Prop :=
  ∀ n : DieOutcome, ¬(e1 n ∧ e2 n)

-- Define complementary events
def complementary (e1 e2 : DieOutcome → Prop) : Prop :=
  ∀ n : DieOutcome, e1 n ↔ ¬e2 n

-- Theorem to prove
theorem A_and_D_mutually_exclusive_but_not_complementary :
  mutually_exclusive event_A event_D ∧ ¬complementary event_A event_D :=
sorry

end NUMINAMATH_CALUDE_A_and_D_mutually_exclusive_but_not_complementary_l3429_342919


namespace NUMINAMATH_CALUDE_power_and_arithmetic_equality_l3429_342908

theorem power_and_arithmetic_equality : (-1)^100 * 5 + (-2)^3 / 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_and_arithmetic_equality_l3429_342908


namespace NUMINAMATH_CALUDE_problem_solution_l3429_342939

/-- The function f(x) defined in the problem -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^3 + 3 * (k - 1) * x^2 - k^2 + 1

/-- The derivative of f(x) with respect to x -/
def f_deriv (k : ℝ) (x : ℝ) : ℝ := 3 * k * x^2 + 6 * (k - 1) * x

theorem problem_solution (k : ℝ) (h : k > 0) :
  (∀ x ∈ Set.Ioo 0 4, f_deriv k x < 0) → k = 1/3 ∧
  (∀ x ∈ Set.Ioo 0 4, f_deriv k x ≤ 0) ↔ 0 < k ∧ k ≤ 1/3 :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3429_342939


namespace NUMINAMATH_CALUDE_point_transformation_l3429_342902

-- Define the transformations
def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def rotate_x_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -z, y)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

-- Define the sequence of transformations
def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  reflect_yz (rotate_x_90 (reflect_xy (rotate_z_90 p)))

-- Theorem statement
theorem point_transformation :
  transform (2, 3, 4) = (3, 4, 2) := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l3429_342902


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l3429_342952

/-- The number of distinct arrangements of n distinct beads on a bracelet,
    where arrangements are indistinguishable under rotation and reflection -/
def bracelet_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem: The number of distinct arrangements of 8 distinct beads on a bracelet,
    where arrangements are indistinguishable under rotation and reflection, is 2520 -/
theorem eight_bead_bracelet_arrangements :
  bracelet_arrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l3429_342952


namespace NUMINAMATH_CALUDE_neighboring_cells_difference_l3429_342924

/-- A type representing a cell in an n × n grid --/
structure Cell (n : ℕ) where
  row : Fin n
  col : Fin n

/-- A function that assigns values to cells in the grid --/
def GridAssignment (n : ℕ) := Cell n → Fin (n^2)

/-- Two cells are neighbors if they share at least one point --/
def IsNeighbor {n : ℕ} (c1 c2 : Cell n) : Prop :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col = c2.col) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col = c2.col) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col.val = c2.col.val + 1)

/-- The main theorem to be proved --/
theorem neighboring_cells_difference {n : ℕ} (h : n > 1) (g : GridAssignment n) :
  ∃ (c1 c2 : Cell n), IsNeighbor c1 c2 ∧ 
    (g c1).val ≥ (g c2).val + n + 1 ∨ (g c2).val ≥ (g c1).val + n + 1 :=
sorry

end NUMINAMATH_CALUDE_neighboring_cells_difference_l3429_342924


namespace NUMINAMATH_CALUDE_height_of_smaller_cone_is_9_l3429_342934

/-- The height of the smaller cone removed from a right circular cone to create a frustum -/
def height_of_smaller_cone (frustum_height : ℝ) (larger_base_area : ℝ) (smaller_base_area : ℝ) : ℝ :=
  sorry

theorem height_of_smaller_cone_is_9 :
  height_of_smaller_cone 18 (324 * Real.pi) (36 * Real.pi) = 9 := by
  sorry

end NUMINAMATH_CALUDE_height_of_smaller_cone_is_9_l3429_342934


namespace NUMINAMATH_CALUDE_james_candy_packs_l3429_342958

/-- Given the initial amount, change received, and cost per pack of candy,
    calculate the number of packs of candy bought. -/
def candyPacks (initialAmount change costPerPack : ℕ) : ℕ :=
  (initialAmount - change) / costPerPack

/-- Theorem stating that James bought 3 packs of candy -/
theorem james_candy_packs :
  candyPacks 20 11 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_james_candy_packs_l3429_342958


namespace NUMINAMATH_CALUDE_cat_age_proof_l3429_342920

theorem cat_age_proof (cat_age rabbit_age dog_age : ℕ) : 
  rabbit_age = cat_age / 2 →
  dog_age = 3 * rabbit_age →
  dog_age = 12 →
  cat_age = 8 := by
sorry

end NUMINAMATH_CALUDE_cat_age_proof_l3429_342920


namespace NUMINAMATH_CALUDE_total_area_approx_33_87_l3429_342995

/-- Converts feet and inches to meters -/
def to_meters (feet : ℕ) (inches : ℕ) : ℝ :=
  feet * 0.3048 + inches * 0.0254

/-- Calculates the area of a room in square meters -/
def room_area (length_feet : ℕ) (length_inches : ℕ) (width_feet : ℕ) (width_inches : ℕ) : ℝ :=
  to_meters length_feet length_inches * to_meters width_feet width_inches

/-- Theorem: The total area of three rooms is approximately 33.87 square meters -/
theorem total_area_approx_33_87 :
  let room_a := room_area 14 8 10 5
  let room_b := room_area 12 3 11 2
  let room_c := room_area 9 7 7 10
  let total_area := room_a + room_b + room_c
  ∃ ε > 0, |total_area - 33.87| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_total_area_approx_33_87_l3429_342995


namespace NUMINAMATH_CALUDE_find_number_l3429_342974

theorem find_number : ∃! x : ℝ, ((((x - 74) * 15) / 5) + 16) - 15 = 58 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3429_342974


namespace NUMINAMATH_CALUDE_circle_tangent_sum_radii_l3429_342900

theorem circle_tangent_sum_radii : 
  ∀ r : ℝ, 
  (r > 0) →
  ((r - 4)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = 12) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_sum_radii_l3429_342900


namespace NUMINAMATH_CALUDE_onion_root_tip_no_tetrads_l3429_342936

/-- Represents the type of cell division a plant tissue undergoes -/
inductive CellDivisionType
  | Mitosis
  | Meiosis

/-- Represents whether tetrads can be observed in a given tissue -/
def can_observe_tetrads (division_type : CellDivisionType) : Prop :=
  match division_type with
  | CellDivisionType.Meiosis => true
  | CellDivisionType.Mitosis => false

/-- The cell division type of onion root tips -/
def onion_root_tip_division : CellDivisionType := CellDivisionType.Mitosis

theorem onion_root_tip_no_tetrads :
  ¬(can_observe_tetrads onion_root_tip_division) :=
by sorry

end NUMINAMATH_CALUDE_onion_root_tip_no_tetrads_l3429_342936


namespace NUMINAMATH_CALUDE_race_probability_l3429_342947

theorem race_probability (total_cars : ℕ) (prob_X prob_Z prob_total : ℝ) : 
  total_cars = 12 →
  prob_X = 1/6 →
  prob_Z = 1/8 →
  prob_total = 0.39166666666666666 →
  ∃ (prob_Y : ℝ), prob_Y = prob_total - prob_X - prob_Z ∧ prob_Y = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_race_probability_l3429_342947


namespace NUMINAMATH_CALUDE_water_percentage_fresh_is_75_percent_l3429_342903

/-- The percentage of water in fresh grapes -/
def water_percentage_fresh : ℝ := 75

/-- The percentage of water in dried grapes -/
def water_percentage_dried : ℝ := 25

/-- The weight of fresh grapes in kg -/
def fresh_weight : ℝ := 200

/-- The weight of dried grapes in kg -/
def dried_weight : ℝ := 66.67

/-- Theorem stating that the percentage of water in fresh grapes is 75% -/
theorem water_percentage_fresh_is_75_percent :
  water_percentage_fresh = 75 := by sorry

end NUMINAMATH_CALUDE_water_percentage_fresh_is_75_percent_l3429_342903


namespace NUMINAMATH_CALUDE_fraction_simplification_l3429_342935

theorem fraction_simplification (a b x : ℝ) :
  (Real.sqrt (a^2 + b^2 + x^2) - (x^2 - a^2 - b^2) / Real.sqrt (a^2 + b^2 + x^2)) / (a^2 + b^2 + x^2) = 
  2 * (a^2 + b^2) / (a^2 + b^2 + x^2)^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3429_342935


namespace NUMINAMATH_CALUDE_possible_teams_count_l3429_342927

/-- Represents the number of players in each position in the squad --/
structure SquadComposition :=
  (goalkeepers : Nat)
  (defenders : Nat)
  (midfielders : Nat)
  (strikers : Nat)

/-- Represents the required composition of a team --/
structure TeamComposition :=
  (goalkeepers : Nat)
  (defenders : Nat)
  (midfielders : Nat)
  (strikers : Nat)

/-- Function to calculate the number of possible teams --/
def calculatePossibleTeams (squad : SquadComposition) (team : TeamComposition) : Nat :=
  let goalkeeperChoices := Nat.choose squad.goalkeepers team.goalkeepers
  let strikerChoices := Nat.choose squad.strikers team.strikers
  let midfielderChoices := Nat.choose squad.midfielders team.midfielders
  let defenderChoices := Nat.choose (squad.defenders + (squad.midfielders - team.midfielders)) team.defenders
  goalkeeperChoices * strikerChoices * midfielderChoices * defenderChoices

/-- Theorem stating the number of possible teams --/
theorem possible_teams_count (squad : SquadComposition) (team : TeamComposition) :
  squad.goalkeepers = 3 →
  squad.defenders = 5 →
  squad.midfielders = 5 →
  squad.strikers = 5 →
  team.goalkeepers = 1 →
  team.defenders = 4 →
  team.midfielders = 4 →
  team.strikers = 2 →
  calculatePossibleTeams squad team = 2250 := by
  sorry

end NUMINAMATH_CALUDE_possible_teams_count_l3429_342927


namespace NUMINAMATH_CALUDE_parallelogram_angle_c_l3429_342999

-- Define a parallelogram structure
structure Parallelogram :=
  (A B C D : ℝ × ℝ)

-- Define angle measure in degrees
def angle_measure (p : Parallelogram) (vertex : Char) : ℝ := sorry

-- State the theorem
theorem parallelogram_angle_c (p : Parallelogram) :
  angle_measure p 'A' + 40 = angle_measure p 'B' →
  angle_measure p 'C' = 70 := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_c_l3429_342999


namespace NUMINAMATH_CALUDE_total_volume_of_cubes_l3429_342969

/-- The volume of a cube with side length s -/
def cube_volume (s : ℕ) : ℕ := s^3

/-- The total volume of n cubes with side length s -/
def total_volume (n : ℕ) (s : ℕ) : ℕ := n * (cube_volume s)

/-- Carl's cubes -/
def carl_cubes : ℕ × ℕ := (3, 3)

/-- Kate's cubes -/
def kate_cubes : ℕ × ℕ := (4, 4)

theorem total_volume_of_cubes :
  total_volume carl_cubes.1 carl_cubes.2 + total_volume kate_cubes.1 kate_cubes.2 = 337 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_cubes_l3429_342969


namespace NUMINAMATH_CALUDE_function_property_l3429_342905

-- Define the function type
def FunctionQ := ℚ → ℚ

-- Define the property that the function must satisfy
def SatisfiesProperty (f : FunctionQ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1

-- Theorem statement
theorem function_property (f : FunctionQ) (h : SatisfiesProperty f) :
  ∀ x : ℚ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3429_342905


namespace NUMINAMATH_CALUDE_arcade_candy_cost_l3429_342966

theorem arcade_candy_cost (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candies : ℕ) :
  whack_a_mole_tickets = 8 →
  skee_ball_tickets = 7 →
  candies = 3 →
  (whack_a_mole_tickets + skee_ball_tickets) / candies = 5 :=
by sorry

end NUMINAMATH_CALUDE_arcade_candy_cost_l3429_342966


namespace NUMINAMATH_CALUDE_limit_exponential_arctangent_sine_l3429_342980

/-- The limit of (e^(4x) - e^(-2x)) / (2 arctan(x) - sin(x)) as x approaches 0 is 6 -/
theorem limit_exponential_arctangent_sine :
  let f : ℝ → ℝ := λ x => (Real.exp (4 * x) - Real.exp (-2 * x)) / (2 * Real.arctan x - Real.sin x)
  ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f x - 6| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_exponential_arctangent_sine_l3429_342980


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_parallel_l3429_342972

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_implies_parallel
  (a b c : Line) (α β γ : Plane)
  (h1 : perpendicular a α)
  (h2 : perpendicular b β)
  (h3 : parallel_lines a b) :
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_parallel_l3429_342972


namespace NUMINAMATH_CALUDE_average_thirteen_l3429_342968

theorem average_thirteen (x : ℝ) : 
  (6 + 16 + 8 + x) / 4 = 13 → x = 22 := by
sorry

end NUMINAMATH_CALUDE_average_thirteen_l3429_342968


namespace NUMINAMATH_CALUDE_angle_B_is_140_degrees_l3429_342916

/-- A quadrilateral with angles A, B, C, and D -/
structure Quadrilateral where
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  angleD : ℝ

/-- The theorem stating that if the sum of angles A, B, and C in a quadrilateral is 220°, 
    then angle B is 140° -/
theorem angle_B_is_140_degrees (q : Quadrilateral) 
    (h : q.angleA + q.angleB + q.angleC = 220) : q.angleB = 140 := by
  sorry


end NUMINAMATH_CALUDE_angle_B_is_140_degrees_l3429_342916


namespace NUMINAMATH_CALUDE_actual_height_is_191_l3429_342992

/-- Represents the height correction problem for a class of students. -/
structure HeightCorrectionProblem where
  num_students : ℕ
  initial_average : ℝ
  incorrect_height : ℝ
  actual_average : ℝ

/-- Calculates the actual height of the student with the incorrect measurement. -/
def calculate_actual_height (problem : HeightCorrectionProblem) : ℝ :=
  problem.num_students * (problem.initial_average - problem.actual_average) + problem.incorrect_height

/-- Theorem stating that the actual height of the student with the incorrect measurement is 191 cm. -/
theorem actual_height_is_191 (problem : HeightCorrectionProblem)
  (h1 : problem.num_students = 20)
  (h2 : problem.initial_average = 175)
  (h3 : problem.incorrect_height = 151)
  (h4 : problem.actual_average = 173) :
  calculate_actual_height problem = 191 := by
  sorry

end NUMINAMATH_CALUDE_actual_height_is_191_l3429_342992


namespace NUMINAMATH_CALUDE_evaluate_expression_l3429_342994

theorem evaluate_expression : 2 + 3 * 4 - 5 + 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3429_342994


namespace NUMINAMATH_CALUDE_geometric_progression_with_means_l3429_342940

theorem geometric_progression_with_means
  (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) :
  let q := (b / a) ^ (1 / (n + 1 : ℝ))
  ∀ k : ℕ, ∃ r : ℝ, a * q ^ k = a * (b / a) ^ (k / (n + 1 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_with_means_l3429_342940


namespace NUMINAMATH_CALUDE_divisibility_by_five_l3429_342997

theorem divisibility_by_five (m n : ℕ) : 
  (∃ k : ℕ, m * n = 5 * k) → (∃ j : ℕ, m = 5 * j) ∨ (∃ l : ℕ, n = 5 * l) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l3429_342997


namespace NUMINAMATH_CALUDE_number_puzzle_l3429_342985

theorem number_puzzle : ∃ x : ℝ, 22 * (x - 36) = 748 ∧ x = 70 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3429_342985


namespace NUMINAMATH_CALUDE_black_ball_from_red_bag_impossible_l3429_342956

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | Black

/-- Represents the contents of a bag -/
structure Bag where
  balls : List BallColor

/-- Defines an impossible event -/
def impossibleEvent (p : ℝ) : Prop := p = 0

/-- Theorem: Drawing a black ball from a bag with only red balls is an impossible event -/
theorem black_ball_from_red_bag_impossible (bag : Bag) 
    (h : ∀ b ∈ bag.balls, b = BallColor.Red) : 
  impossibleEvent (Nat.card {i | bag.balls.get? i = some BallColor.Black} / bag.balls.length) := by
  sorry

end NUMINAMATH_CALUDE_black_ball_from_red_bag_impossible_l3429_342956


namespace NUMINAMATH_CALUDE_product_of_real_parts_is_two_l3429_342959

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z : ℂ) : Prop :=
  z^2 + 3*z = -7 + 2*i

-- Theorem statement
theorem product_of_real_parts_is_two :
  ∃ (z₁ z₂ : ℂ), quadratic_equation z₁ ∧ quadratic_equation z₂ ∧
  z₁ ≠ z₂ ∧ (z₁.re * z₂.re = 2) :=
sorry

end NUMINAMATH_CALUDE_product_of_real_parts_is_two_l3429_342959


namespace NUMINAMATH_CALUDE_jerry_zinc_consumption_l3429_342976

/-- Calculates the total milligrams of zinc consumed from antacids -/
def total_zinc_mg (large_antacid_count : ℕ) (large_antacid_weight : ℝ) (large_antacid_zinc_percent : ℝ)
                  (small_antacid_count : ℕ) (small_antacid_weight : ℝ) (small_antacid_zinc_percent : ℝ) : ℝ :=
  ((large_antacid_count : ℝ) * large_antacid_weight * large_antacid_zinc_percent +
   (small_antacid_count : ℝ) * small_antacid_weight * small_antacid_zinc_percent) * 1000

/-- Theorem stating the total zinc consumed by Jerry -/
theorem jerry_zinc_consumption :
  total_zinc_mg 2 2 0.05 3 1 0.15 = 650 := by
  sorry

end NUMINAMATH_CALUDE_jerry_zinc_consumption_l3429_342976


namespace NUMINAMATH_CALUDE_shirts_per_minute_l3429_342901

/-- An industrial machine that makes shirts -/
structure ShirtMachine where
  /-- The number of shirts made in 6 minutes -/
  shirts_in_6_min : ℕ
  /-- The number of minutes (6) -/
  minutes : ℕ
  /-- Assumption that the machine made 36 shirts in 6 minutes -/
  h_shirts : shirts_in_6_min = 36
  /-- Assumption that the time period is 6 minutes -/
  h_minutes : minutes = 6

/-- Theorem stating that the machine makes 6 shirts per minute -/
theorem shirts_per_minute (machine : ShirtMachine) : 
  machine.shirts_in_6_min / machine.minutes = 6 := by
  sorry

#check shirts_per_minute

end NUMINAMATH_CALUDE_shirts_per_minute_l3429_342901


namespace NUMINAMATH_CALUDE_power_division_rule_l3429_342962

theorem power_division_rule (a : ℝ) : a^5 / a^2 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l3429_342962


namespace NUMINAMATH_CALUDE_range_of_a_l3429_342988

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, (a - a^2) * (x^2 + 1) + x ≤ 0) ↔ 
  a ∈ Set.Iic ((1 - Real.sqrt 3) / 2) ∪ Set.Ici ((1 + Real.sqrt 3) / 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3429_342988


namespace NUMINAMATH_CALUDE_blacksmith_shoeing_time_l3429_342996

/-- The minimum time required for a group of blacksmiths to shoe all horses. -/
def minimum_shoeing_time (num_blacksmiths : ℕ) (num_horses : ℕ) (time_per_shoe : ℕ) : ℕ :=
  let total_shoes := num_horses * 4
  let total_time := total_shoes * time_per_shoe
  total_time / num_blacksmiths

/-- Theorem stating that 48 blacksmiths can shoe 60 horses in 25 minutes when each horseshoe takes 5 minutes. -/
theorem blacksmith_shoeing_time :
  minimum_shoeing_time 48 60 5 = 25 := by
  sorry

#eval minimum_shoeing_time 48 60 5

end NUMINAMATH_CALUDE_blacksmith_shoeing_time_l3429_342996


namespace NUMINAMATH_CALUDE_complex_magnitude_one_l3429_342932

/-- Given a complex number z satisfying z⋅2i = |z|² + 1, prove that |z| = 1 -/
theorem complex_magnitude_one (z : ℂ) (h : z * (2 * Complex.I) = Complex.abs z ^ 2 + 1) :
  Complex.abs z = 1 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_one_l3429_342932


namespace NUMINAMATH_CALUDE_cube_angles_l3429_342965

/-- Represents a cube in 3D space -/
structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

/-- Calculates the angle between two skew lines in a cube -/
def angle_between_skew_lines (c : Cube) (l1 l2 : Fin 2 → Fin 8) : ℝ :=
  sorry

/-- Calculates the angle between a line and a plane in a cube -/
def angle_between_line_and_plane (c : Cube) (l : Fin 2 → Fin 8) (p : Fin 4 → Fin 8) : ℝ :=
  sorry

/-- Theorem stating the angles in a cube -/
theorem cube_angles (c : Cube) : 
  angle_between_skew_lines c ![7, 1] ![0, 2] = 60 ∧ 
  angle_between_line_and_plane c ![7, 1] ![7, 5, 2, 3] = 30 :=
sorry

end NUMINAMATH_CALUDE_cube_angles_l3429_342965


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3429_342918

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a where a_4 * a_6 = 5, prove that a_2 * a_3 * a_7 * a_8 = 25 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : IsGeometricSequence a) 
    (h_prod : a 4 * a 6 = 5) : a 2 * a 3 * a 7 * a 8 = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3429_342918


namespace NUMINAMATH_CALUDE_exists_vertex_reach_all_l3429_342955

/-- A directed graph where every pair of vertices is connected by a directed edge. -/
structure CompleteDigraph (V : Type) where
  edge : V → V → Prop
  complete : ∀ (u v : V), u ≠ v → edge u v ∨ edge v u

/-- A path of length at most 2 exists between two vertices. -/
def PathLengthAtMostTwo {V : Type} (G : CompleteDigraph V) (u v : V) : Prop :=
  G.edge u v ∨ ∃ w : V, G.edge u w ∧ G.edge w v

/-- There exists a vertex from which every other vertex can be reached by a path of length at most 2. -/
theorem exists_vertex_reach_all {V : Type} (G : CompleteDigraph V) [Finite V] [Nonempty V] :
  ∃ u : V, ∀ v : V, u ≠ v → PathLengthAtMostTwo G u v := by sorry

end NUMINAMATH_CALUDE_exists_vertex_reach_all_l3429_342955


namespace NUMINAMATH_CALUDE_remainder_3_pow_2003_mod_13_l3429_342949

theorem remainder_3_pow_2003_mod_13 :
  ∃ k : ℤ, 3^2003 = 13 * k + 9 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_2003_mod_13_l3429_342949


namespace NUMINAMATH_CALUDE_initial_value_proof_l3429_342922

theorem initial_value_proof : 
  ∃! x : ℕ, x ≥ 0 ∧ (∀ y : ℕ, y ≥ 0 → (y + 37) % 3 = 0 ∧ (y + 37) % 5 = 0 ∧ (y + 37) % 7 = 0 ∧ (y + 37) % 8 = 0 → x ≤ y) ∧
  (x + 37) % 3 = 0 ∧ (x + 37) % 5 = 0 ∧ (x + 37) % 7 = 0 ∧ (x + 37) % 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_initial_value_proof_l3429_342922


namespace NUMINAMATH_CALUDE_line_parameterization_l3429_342941

/-- Given a line y = 2x - 30 parameterized by (x, y) = (f(t), 20t - 10), prove that f(t) = 10t + 10 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 20 * t - 10 = 2 * f t - 30) → 
  (∀ t : ℝ, f t = 10 * t + 10) := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l3429_342941


namespace NUMINAMATH_CALUDE_roots_equal_magnitude_implies_real_ratio_l3429_342998

theorem roots_equal_magnitude_implies_real_ratio 
  (p q : ℂ) 
  (h_q_nonzero : q ≠ 0) 
  (h_roots_equal_magnitude : ∀ z₁ z₂ : ℂ, z₁^2 + p*z₁ + q^2 = 0 → z₂^2 + p*z₂ + q^2 = 0 → Complex.abs z₁ = Complex.abs z₂) :
  ∃ r : ℝ, p / q = r := by sorry

end NUMINAMATH_CALUDE_roots_equal_magnitude_implies_real_ratio_l3429_342998


namespace NUMINAMATH_CALUDE_pumpkin_patch_problem_l3429_342906

def pumpkin_pie_filling_cans (total_pumpkins : ℕ) (price_per_pumpkin : ℕ) (total_earnings : ℕ) (pumpkins_per_can : ℕ) : ℕ :=
  let pumpkins_sold := total_earnings / price_per_pumpkin
  let remaining_pumpkins := total_pumpkins - pumpkins_sold
  remaining_pumpkins / pumpkins_per_can

theorem pumpkin_patch_problem :
  pumpkin_pie_filling_cans 83 3 96 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_patch_problem_l3429_342906


namespace NUMINAMATH_CALUDE_fraction_of_fraction_two_ninths_of_three_fourths_l3429_342960

theorem fraction_of_fraction (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem two_ninths_of_three_fourths :
  (2 : ℚ) / 9 / ((3 : ℚ) / 4) = 8 / 27 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_two_ninths_of_three_fourths_l3429_342960


namespace NUMINAMATH_CALUDE_prism_21_edges_has_9_faces_l3429_342953

/-- A prism is a polyhedron with two congruent parallel faces (bases) and whose other faces (lateral faces) are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism -/
def Prism.faces (p : Prism) : ℕ := sorry

/-- Theorem: A prism with 21 edges has 9 faces -/
theorem prism_21_edges_has_9_faces (p : Prism) (h : p.edges = 21) : p.faces = 9 := by
  sorry

end NUMINAMATH_CALUDE_prism_21_edges_has_9_faces_l3429_342953


namespace NUMINAMATH_CALUDE_value_of_y_l3429_342913

theorem value_of_y (x y : ℝ) (h1 : x^2 - 2*x + 5 = y + 3) (h2 : x = -3) : y = 17 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l3429_342913


namespace NUMINAMATH_CALUDE_convex_polyhedron_same_sided_faces_l3429_342915

/-- A face of a polyhedron -/
structure Face where
  sides : ℕ

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : Set Face

/-- Theorem: Every convex polyhedron has at least two faces with the same number of sides -/
theorem convex_polyhedron_same_sided_faces (P : ConvexPolyhedron) :
  ∃ (f₁ f₂ : Face), f₁ ∈ P.faces ∧ f₂ ∈ P.faces ∧ f₁ ≠ f₂ ∧ f₁.sides = f₂.sides :=
sorry

end NUMINAMATH_CALUDE_convex_polyhedron_same_sided_faces_l3429_342915


namespace NUMINAMATH_CALUDE_S_bounds_l3429_342975

theorem S_bounds (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  let S := Real.sqrt (a * b / ((b + c) * (c + a))) +
           Real.sqrt (b * c / ((a + c) * (b + a))) +
           Real.sqrt (c * a / ((b + c) * (b + a)))
  1 ≤ S ∧ S ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_S_bounds_l3429_342975
