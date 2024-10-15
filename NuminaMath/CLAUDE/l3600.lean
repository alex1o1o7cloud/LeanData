import Mathlib

namespace NUMINAMATH_CALUDE_stating_acquaintance_group_relation_l3600_360075

/-- 
A group of people with specific acquaintance relationships.
-/
structure AcquaintanceGroup where
  n : ℕ  -- Total number of people
  k : ℕ  -- Number of acquaintances per person
  l : ℕ  -- Number of common acquaintances for acquainted pairs
  m : ℕ  -- Number of common acquaintances for non-acquainted pairs
  k_lt_n : k < n  -- Each person is acquainted with fewer than the total number of people

/-- 
Theorem stating the relationship between the parameters of an AcquaintanceGroup.
-/
theorem acquaintance_group_relation (g : AcquaintanceGroup) : 
  g.m * (g.n - g.k - 1) = g.k * (g.k - g.l - 1) := by
  sorry

end NUMINAMATH_CALUDE_stating_acquaintance_group_relation_l3600_360075


namespace NUMINAMATH_CALUDE_cone_volume_l3600_360045

/-- Given a cone with base circumference 2π and lateral area 2π, its volume is (√3 * π) / 3 -/
theorem cone_volume (r h l : ℝ) (h1 : 2 * π = 2 * π * r) (h2 : 2 * π = π * r * l) :
  (1 / 3) * π * r^2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l3600_360045


namespace NUMINAMATH_CALUDE_asteroid_speed_comparison_l3600_360009

/-- Asteroid observation and speed comparison -/
theorem asteroid_speed_comparison 
  (distance_X13 : ℝ) 
  (time_X13 : ℝ) 
  (speed_X13 : ℝ) 
  (speed_Y14 : ℝ) 
  (h1 : distance_X13 = 2000) 
  (h2 : speed_X13 = distance_X13 / time_X13) 
  (h3 : speed_Y14 = 3 * speed_X13) : 
  speed_Y14 - speed_X13 = speed_X13 := by
  sorry

end NUMINAMATH_CALUDE_asteroid_speed_comparison_l3600_360009


namespace NUMINAMATH_CALUDE_royal_family_children_count_l3600_360073

/-- Represents the royal family -/
structure RoyalFamily where
  king_age : ℕ
  queen_age : ℕ
  num_sons : ℕ
  num_daughters : ℕ
  children_total_age : ℕ

/-- The possible numbers of children for the royal family -/
def possible_children_numbers : Set ℕ := {7, 9}

theorem royal_family_children_count (family : RoyalFamily) 
  (h1 : family.king_age = 35)
  (h2 : family.queen_age = 35)
  (h3 : family.num_sons = 3)
  (h4 : family.num_daughters ≥ 1)
  (h5 : family.children_total_age = 35)
  (h6 : family.num_sons + family.num_daughters ≤ 20)
  (h7 : ∃ (n : ℕ), n > 0 ∧ family.king_age + n + family.queen_age + n = family.children_total_age + n * (family.num_sons + family.num_daughters)) :
  (family.num_sons + family.num_daughters) ∈ possible_children_numbers :=
sorry

end NUMINAMATH_CALUDE_royal_family_children_count_l3600_360073


namespace NUMINAMATH_CALUDE_faster_train_speed_is_72_l3600_360043

/-- The speed of the faster train given the conditions of the problem -/
def faster_train_speed (slower_train_speed : ℝ) (speed_difference : ℝ) 
  (crossing_time : ℝ) (train_length : ℝ) : ℝ :=
  slower_train_speed + speed_difference

/-- Theorem stating the speed of the faster train under the given conditions -/
theorem faster_train_speed_is_72 :
  faster_train_speed 36 36 20 200 = 72 := by
  sorry

end NUMINAMATH_CALUDE_faster_train_speed_is_72_l3600_360043


namespace NUMINAMATH_CALUDE_two_counterexamples_l3600_360094

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has any digit equal to 0 -/
def has_zero_digit (n : ℕ) : Bool := sorry

/-- The main theorem -/
theorem two_counterexamples : 
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, sum_of_digits n = 4 ∧ ¬has_zero_digit n ∧ ¬Nat.Prime n) ∧ 
    s.card = 2 := by sorry

end NUMINAMATH_CALUDE_two_counterexamples_l3600_360094


namespace NUMINAMATH_CALUDE_m_range_l3600_360060

/-- A function f parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 3 * x - m - 2

/-- The property that f has exactly one root in (0, 1) -/
def has_one_root_in_unit_interval (m : ℝ) : Prop :=
  ∃! x, 0 < x ∧ x < 1 ∧ f m x = 0

/-- The main theorem stating the range of m -/
theorem m_range :
  ∀ m : ℝ, has_one_root_in_unit_interval m ↔ m > -2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l3600_360060


namespace NUMINAMATH_CALUDE_sequence_ratio_theorem_l3600_360046

/-- An arithmetic sequence with a non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = q * b n

/-- Theorem: Given conditions on arithmetic and geometric sequences, prove the ratio -/
theorem sequence_ratio_theorem (a b : ℕ → ℝ) :
  ArithmeticSequence a →
  GeometricSequence b →
  a 1 = b 1 ∧ a 3 = b 2 ∧ a 7 = b 3 →
  (b 3 + b 4) / (b 4 + b 5) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_theorem_l3600_360046


namespace NUMINAMATH_CALUDE_bamboo_pole_sections_l3600_360063

theorem bamboo_pole_sections (n : ℕ) (a : ℕ → ℝ) : 
  (∀ i j, i < j → j ≤ n → a j - a i = (j - i) * (a 2 - a 1)) →  -- arithmetic sequence
  (a 1 = 10) →  -- top section length
  (a n + a (n-1) + a (n-2) = 114) →  -- last three sections total
  (a 6 ^ 2 = a 1 * a n) →  -- 6th section is geometric mean of first and last
  (n > 6) →
  n = 16 :=
by sorry

end NUMINAMATH_CALUDE_bamboo_pole_sections_l3600_360063


namespace NUMINAMATH_CALUDE_money_difference_l3600_360023

/-- Proves that Hoseok has 170,000 won more than Min-young after they both earn additional money -/
theorem money_difference (initial_amount : ℕ) (minyoung_earnings hoseok_earnings : ℕ) :
  initial_amount = 1500000 →
  minyoung_earnings = 320000 →
  hoseok_earnings = 490000 →
  (initial_amount + hoseok_earnings) - (initial_amount + minyoung_earnings) = 170000 :=
by
  sorry

end NUMINAMATH_CALUDE_money_difference_l3600_360023


namespace NUMINAMATH_CALUDE_log_5_12_equals_fraction_l3600_360062

-- Define the common logarithm (base 10) function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the logarithm with base 5
noncomputable def log_5 (x : ℝ) := Real.log x / Real.log 5

theorem log_5_12_equals_fraction (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) :
  log_5 12 = (2 * a + b) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_log_5_12_equals_fraction_l3600_360062


namespace NUMINAMATH_CALUDE_expression_factorization_l3600_360090

theorem expression_factorization (x : ℝ) :
  (9 * x^5 + 25 * x^3 - 4) - (x^5 - 3 * x^3 - 4) = 4 * x^3 * (2 * x^2 + 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3600_360090


namespace NUMINAMATH_CALUDE_tape_area_calculation_l3600_360001

theorem tape_area_calculation (width : ℝ) (length : ℝ) (num_pieces : ℕ) (overlap : ℝ) 
  (h_width : width = 9.4)
  (h_length : length = 3.7)
  (h_num_pieces : num_pieces = 15)
  (h_overlap : overlap = 0.6) :
  let single_area := width * length
  let total_area_no_overlap := num_pieces * single_area
  let overlap_area := overlap * length
  let total_overlap_area := (num_pieces - 1) * overlap_area
  let total_area := total_area_no_overlap - total_overlap_area
  total_area = 490.62 := by sorry

end NUMINAMATH_CALUDE_tape_area_calculation_l3600_360001


namespace NUMINAMATH_CALUDE_exist_three_equal_digit_sums_l3600_360025

-- Define the sum of decimal digits function
def S (n : ℕ) : ℕ := sorry

-- State the theorem
theorem exist_three_equal_digit_sums :
  ∃ a b c : ℕ, 1 ≤ a ∧ a < b ∧ b < c ∧ c ≤ 25 ∧
  S (a^6 + 2014) = S (b^6 + 2014) ∧ S (b^6 + 2014) = S (c^6 + 2014) := by sorry

end NUMINAMATH_CALUDE_exist_three_equal_digit_sums_l3600_360025


namespace NUMINAMATH_CALUDE_solution_set_proof_l3600_360069

theorem solution_set_proof (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h0 : f 0 = 2) (h1 : ∀ x : ℝ, f x + (deriv f) x > 1) :
  {x : ℝ | Real.exp x * f x > Real.exp x + 1} = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_proof_l3600_360069


namespace NUMINAMATH_CALUDE_ferry_tourist_count_l3600_360051

/-- Represents the ferry schedule and passenger count --/
structure FerrySchedule where
  start_time : Nat -- 10 AM represented as 0
  end_time : Nat   -- 3 PM represented as 10
  initial_passengers : Nat
  passenger_decrease : Nat

/-- Calculates the total number of tourists transported by the ferry --/
def total_tourists (schedule : FerrySchedule) : Nat :=
  let num_trips := schedule.end_time - schedule.start_time + 1
  let arithmetic_sum := num_trips * (2 * schedule.initial_passengers - (num_trips - 1) * schedule.passenger_decrease)
  arithmetic_sum / 2

/-- Theorem stating that the total number of tourists is 990 --/
theorem ferry_tourist_count :
  ∀ (schedule : FerrySchedule),
    schedule.start_time = 0 ∧
    schedule.end_time = 10 ∧
    schedule.initial_passengers = 100 ∧
    schedule.passenger_decrease = 2 →
    total_tourists schedule = 990 := by
  sorry

end NUMINAMATH_CALUDE_ferry_tourist_count_l3600_360051


namespace NUMINAMATH_CALUDE_mary_book_count_l3600_360004

/-- Represents the number of books Mary has at different stages --/
structure BookCount where
  initial : Nat
  afterReturningUnhelpful : Nat
  afterFirstCheckout : Nat
  beforeSecondCheckout : Nat
  final : Nat

/-- Represents the number of books Mary checks out or returns --/
structure BookTransactions where
  firstReturn : Nat
  firstCheckout : Nat
  secondReturn : Nat
  secondCheckout : Nat

theorem mary_book_count (b : BookCount) (t : BookTransactions) :
  b.initial = 5 →
  t.firstReturn = 3 →
  b.afterReturningUnhelpful = b.initial - t.firstReturn →
  b.afterFirstCheckout = b.afterReturningUnhelpful + t.firstCheckout →
  b.beforeSecondCheckout = b.afterFirstCheckout - t.secondReturn →
  t.secondReturn = 2 →
  t.secondCheckout = 7 →
  b.final = b.beforeSecondCheckout + t.secondCheckout →
  b.final = 12 →
  t.firstCheckout = 5 := by
sorry

end NUMINAMATH_CALUDE_mary_book_count_l3600_360004


namespace NUMINAMATH_CALUDE_choir_arrangement_theorem_l3600_360036

theorem choir_arrangement_theorem (m : ℕ) : 
  (∃ y : ℕ, m = y^2 + 11) ∧ 
  (∃ n : ℕ, m = n * (n + 5)) ∧ 
  (∀ k : ℕ, (∃ y : ℕ, k = y^2 + 11) ∧ (∃ n : ℕ, k = n * (n + 5)) → k ≤ m) → 
  m = 300 :=
by sorry

end NUMINAMATH_CALUDE_choir_arrangement_theorem_l3600_360036


namespace NUMINAMATH_CALUDE_mildred_weight_l3600_360021

/-- Given that Carol weighs 9 pounds and Mildred is 50 pounds heavier than Carol,
    prove that Mildred weighs 59 pounds. -/
theorem mildred_weight (carol_weight : ℕ) (weight_difference : ℕ) :
  carol_weight = 9 →
  weight_difference = 50 →
  carol_weight + weight_difference = 59 :=
by sorry

end NUMINAMATH_CALUDE_mildred_weight_l3600_360021


namespace NUMINAMATH_CALUDE_linear_equation_m_value_l3600_360076

theorem linear_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b, (4 - m) * x^(|m| - 3) - 16 = a * x + b) ∧ 
  (m - 4 ≠ 0) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_m_value_l3600_360076


namespace NUMINAMATH_CALUDE_nested_average_equals_seven_sixths_l3600_360098

-- Define the average of two numbers
def avg2 (a b : ℚ) : ℚ := (a + b) / 2

-- Define the average of three numbers
def avg3 (a b c : ℚ) : ℚ := (a + b + c) / 3

-- The main theorem
theorem nested_average_equals_seven_sixths :
  avg3 (avg3 2 1 0) (avg2 1 2) 1 = 7/6 := by sorry

end NUMINAMATH_CALUDE_nested_average_equals_seven_sixths_l3600_360098


namespace NUMINAMATH_CALUDE_square_root_divided_by_19_l3600_360044

theorem square_root_divided_by_19 : 
  Real.sqrt 5776 / 19 = 4 := by sorry

end NUMINAMATH_CALUDE_square_root_divided_by_19_l3600_360044


namespace NUMINAMATH_CALUDE_charlies_bus_ride_l3600_360086

theorem charlies_bus_ride (oscars_ride : ℝ) (difference : ℝ) :
  oscars_ride = 0.75 →
  oscars_ride = difference + charlies_ride →
  difference = 0.5 →
  charlies_ride = 0.25 :=
by
  sorry

end NUMINAMATH_CALUDE_charlies_bus_ride_l3600_360086


namespace NUMINAMATH_CALUDE_garden_perimeter_l3600_360016

/-- Given a square garden with a pond, if the pond area is 20 square meters
    and the remaining garden area is 124 square meters,
    then the perimeter of the garden is 48 meters. -/
theorem garden_perimeter (s : ℝ) : 
  s > 0 → 
  s^2 = 20 + 124 → 
  4 * s = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l3600_360016


namespace NUMINAMATH_CALUDE_cannot_form_triangle_l3600_360047

/-- Represents the lengths of three line segments -/
structure Triangle :=
  (a b c : ℝ)

/-- Checks if three line segments can form a triangle -/
def canFormTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

/-- Theorem: The set of line segments 2cm, 3cm, 6cm cannot form a triangle -/
theorem cannot_form_triangle :
  ¬ canFormTriangle ⟨2, 3, 6⟩ :=
sorry

end NUMINAMATH_CALUDE_cannot_form_triangle_l3600_360047


namespace NUMINAMATH_CALUDE_purple_jellybeans_count_l3600_360018

theorem purple_jellybeans_count (total : ℕ) (blue : ℕ) (orange : ℕ) (red : ℕ) 
  (h_total : total = 200)
  (h_blue : blue = 14)
  (h_orange : orange = 40)
  (h_red : red = 120) :
  total - (blue + orange + red) = 26 := by
  sorry

end NUMINAMATH_CALUDE_purple_jellybeans_count_l3600_360018


namespace NUMINAMATH_CALUDE_smallest_number_in_set_l3600_360005

theorem smallest_number_in_set (a b c d : ℕ+) : 
  (a + b + c + d : ℝ) / 4 = 30 →
  b = 28 →
  b < c →
  c < d →
  d = b + 7 →
  a < b →
  a = 27 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_in_set_l3600_360005


namespace NUMINAMATH_CALUDE_right_triangle_area_l3600_360066

theorem right_triangle_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_hypotenuse : c = 15) (h_side : a = 12) : (1/2) * a * b = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3600_360066


namespace NUMINAMATH_CALUDE_power_function_through_point_l3600_360092

/-- Given a power function f(x) = x^a that passes through the point (2, 4), prove that f(x) = x^2 -/
theorem power_function_through_point (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x ^ a) →
  f 2 = 4 →
  ∀ x, f x = x ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3600_360092


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3600_360074

theorem quadratic_solution_difference_squared :
  ∀ a b : ℝ,
  (2 * a^2 - 7 * a + 6 = 0) →
  (2 * b^2 - 7 * b + 6 = 0) →
  (a ≠ b) →
  (a - b)^2 = (1 : ℝ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l3600_360074


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l3600_360093

/-- The equation of a circle in the form x^2 + y^2 + ax + by + c = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

/-- Given a circle equation, returns its properties (center and radius) -/
def circle_properties (eq : CircleEquation) : CircleProperties :=
  sorry

theorem circle_center_and_radius 
  (eq : CircleEquation) 
  (h : eq = ⟨-6, 0, 0⟩) : 
  circle_properties eq = ⟨(3, 0), 3⟩ :=
sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l3600_360093


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l3600_360019

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬ ∀ x > 1, p x) ↔ (∃ x > 1, ¬ p x) :=
by sorry

theorem negation_of_specific_proposition : 
  (¬ ∀ x > 1, x^3 + 16 > 8*x) ↔ (∃ x > 1, x^3 + 16 ≤ 8*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l3600_360019


namespace NUMINAMATH_CALUDE_arthur_muffins_l3600_360017

theorem arthur_muffins (initial_muffins : ℕ) : 
  initial_muffins + 48 = 83 → initial_muffins = 35 := by
  sorry

end NUMINAMATH_CALUDE_arthur_muffins_l3600_360017


namespace NUMINAMATH_CALUDE_cistern_leak_empty_time_l3600_360087

/-- Given a cistern with normal fill time and leak-affected fill time, 
    calculate the time it takes for the leak to empty the full cistern. -/
theorem cistern_leak_empty_time 
  (normal_fill_time : ℝ) 
  (leak_fill_time : ℝ) 
  (h1 : normal_fill_time = 12) 
  (h2 : leak_fill_time = normal_fill_time + 2) : 
  (1 / ((1 / normal_fill_time) - (1 / leak_fill_time))) = 84 := by
  sorry

end NUMINAMATH_CALUDE_cistern_leak_empty_time_l3600_360087


namespace NUMINAMATH_CALUDE_west_side_denial_percentage_l3600_360026

theorem west_side_denial_percentage :
  let total_kids := 260
  let riverside_kids := 120
  let west_side_kids := 90
  let mountaintop_kids := 50
  let riverside_denied_percentage := 20
  let mountaintop_denied_percentage := 50
  let kids_admitted := 148
  
  let riverside_denied := riverside_kids * riverside_denied_percentage / 100
  let mountaintop_denied := mountaintop_kids * mountaintop_denied_percentage / 100
  let total_denied := total_kids - kids_admitted
  let west_side_denied := total_denied - riverside_denied - mountaintop_denied
  let west_side_denied_percentage := west_side_denied / west_side_kids * 100

  west_side_denied_percentage = 70 := by sorry

end NUMINAMATH_CALUDE_west_side_denial_percentage_l3600_360026


namespace NUMINAMATH_CALUDE_complex_simplification_l3600_360068

theorem complex_simplification :
  (4 - 3*Complex.I) - (7 + 5*Complex.I) + 2*(1 - 2*Complex.I) = -1 - 12*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l3600_360068


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_cyclic_polygon_l3600_360064

/-- A cyclic polygon is a polygon whose vertices all lie on a single circle. -/
structure CyclicPolygon where
  n : ℕ
  sides_ge_4 : n ≥ 4

/-- The sum of interior angles of a cyclic polygon. -/
def sum_of_interior_angles (p : CyclicPolygon) : ℝ :=
  (p.n - 2) * 180

/-- Theorem: The sum of interior angles of a cyclic polygon with n sides is (n-2) * 180°. -/
theorem sum_of_interior_angles_cyclic_polygon (p : CyclicPolygon) :
  sum_of_interior_angles p = (p.n - 2) * 180 := by
  sorry

#check sum_of_interior_angles_cyclic_polygon

end NUMINAMATH_CALUDE_sum_of_interior_angles_cyclic_polygon_l3600_360064


namespace NUMINAMATH_CALUDE_order_of_expressions_l3600_360000

theorem order_of_expressions : 3^(1/2) > Real.log (1/2) / Real.log (1/3) ∧ 
  Real.log (1/2) / Real.log (1/3) > Real.log (1/3) / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_order_of_expressions_l3600_360000


namespace NUMINAMATH_CALUDE_reciprocal_sum_l3600_360089

theorem reciprocal_sum (x y : ℝ) (h1 : x * y > 0) (h2 : 1 / (x * y) = 5) (h3 : (x + y) / 5 = 0.6) :
  1 / x + 1 / y = 15 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_l3600_360089


namespace NUMINAMATH_CALUDE_mans_upstream_rate_l3600_360022

/-- Given a man's rowing rates and current speed, calculate his upstream rate -/
theorem mans_upstream_rate
  (downstream_rate : ℝ)
  (still_water_rate : ℝ)
  (current_rate : ℝ)
  (h1 : downstream_rate = 24)
  (h2 : still_water_rate = 15.5)
  (h3 : current_rate = 8.5) :
  still_water_rate - current_rate = 7 := by
  sorry

end NUMINAMATH_CALUDE_mans_upstream_rate_l3600_360022


namespace NUMINAMATH_CALUDE_range_of_a_l3600_360082

theorem range_of_a (a : ℝ) : 
  (∀ x₀ : ℝ, ∀ x : ℝ, x + a * x₀ + 1 ≥ 0) → a ∈ Set.Icc (-2) 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3600_360082


namespace NUMINAMATH_CALUDE_parabolas_common_point_l3600_360013

/-- A parabola in the family y = -x^2 + px + q -/
structure Parabola where
  p : ℝ
  q : ℝ

/-- The y-coordinate of a point on a parabola given its x-coordinate -/
def Parabola.y_coord (para : Parabola) (x : ℝ) : ℝ :=
  -x^2 + para.p * x + para.q

/-- The condition that the vertex of a parabola lies on y = x^2 -/
def vertex_on_curve (para : Parabola) : Prop :=
  ∃ a : ℝ, para.y_coord a = a^2

theorem parabolas_common_point :
  ∀ p : ℝ, ∃ para : Parabola, 
    vertex_on_curve para ∧ 
    para.p = p ∧
    para.y_coord 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabolas_common_point_l3600_360013


namespace NUMINAMATH_CALUDE_alice_bob_earnings_l3600_360091

/-- Given the working hours and hourly rates of Alice and Bob, prove that the value of t that makes their earnings equal is 7.8 -/
theorem alice_bob_earnings (t : ℝ) : 
  (3 * t - 9) * (4 * t - 3) = (4 * t - 16) * (3 * t - 9) → t = 7.8 := by
  sorry

end NUMINAMATH_CALUDE_alice_bob_earnings_l3600_360091


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3600_360050

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b → b ≤ c → 
  b = 8 → 
  (a + b + c) / 3 = a + 12 →
  (a + b + c) / 3 = c - 20 →
  a + b + c = 48 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3600_360050


namespace NUMINAMATH_CALUDE_sum_three_squares_not_7_mod_8_l3600_360083

theorem sum_three_squares_not_7_mod_8 (a b c : ℤ) : (a^2 + b^2 + c^2) % 8 ≠ 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_three_squares_not_7_mod_8_l3600_360083


namespace NUMINAMATH_CALUDE_square_pyramid_dihedral_angle_cosine_l3600_360078

/-- A pyramid with a square base and specific properties -/
structure SquarePyramid where
  -- The length of the congruent edges
  edge_length : ℝ
  -- The measure of the dihedral angle between faces PQR and PRS
  dihedral_angle : ℝ
  -- Angle QPR is 45°
  angle_QPR_is_45 : angle_QPR = Real.pi / 4
  -- The base is square (implied by the problem setup)
  base_is_square : True

/-- The theorem statement -/
theorem square_pyramid_dihedral_angle_cosine 
  (P : SquarePyramid) 
  (a b : ℝ) 
  (h : Real.cos P.dihedral_angle = a + Real.sqrt b) : 
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_pyramid_dihedral_angle_cosine_l3600_360078


namespace NUMINAMATH_CALUDE_henry_bicycle_improvement_l3600_360024

/-- Henry's bicycle ride improvement --/
theorem henry_bicycle_improvement (initial_laps initial_time current_laps current_time : ℚ) 
  (h1 : initial_laps = 15)
  (h2 : initial_time = 45)
  (h3 : current_laps = 18)
  (h4 : current_time = 42) :
  (initial_time / initial_laps) - (current_time / current_laps) = 2/3 := by
  sorry

#eval (45 : ℚ) / 15 - (42 : ℚ) / 18

end NUMINAMATH_CALUDE_henry_bicycle_improvement_l3600_360024


namespace NUMINAMATH_CALUDE_president_vp_committee_selection_l3600_360038

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem president_vp_committee_selection (n : ℕ) (h : n = 10) : 
  n * (n - 1) * choose (n - 2) 2 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_president_vp_committee_selection_l3600_360038


namespace NUMINAMATH_CALUDE_absolute_value_plus_inverse_l3600_360072

theorem absolute_value_plus_inverse : |(-2 : ℝ)| + 3⁻¹ = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_plus_inverse_l3600_360072


namespace NUMINAMATH_CALUDE_triangle_side_sum_range_l3600_360053

theorem triangle_side_sum_range (a b c : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- positive side lengths
  (a + b > c ∧ b + c > a ∧ a + c > b) →  -- triangle inequality
  (∃ x : ℝ, x^2 - (a + b)*x + a*b = 0) →  -- a and b are roots of the quadratic equation
  (a < b) →  -- given condition
  (7/8 < a + b - c ∧ a + b - c < Real.sqrt 5 - 1) := by
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_range_l3600_360053


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3600_360041

theorem sum_of_three_numbers (S : Finset ℕ) (h1 : S.card = 10) (h2 : S.sum id > 144) :
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a + b + c ≥ 54 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3600_360041


namespace NUMINAMATH_CALUDE_parabola_shift_left_two_l3600_360054

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (h : ℝ) : Parabola :=
  { f := fun x => p.f (x + h) }

/-- The standard parabola y = x^2 -/
def standard_parabola : Parabola :=
  { f := fun x => x^2 }

theorem parabola_shift_left_two :
  (shift_parabola standard_parabola 2).f = fun x => (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_left_two_l3600_360054


namespace NUMINAMATH_CALUDE_x_plus_y_positive_l3600_360085

theorem x_plus_y_positive (x y : ℝ) (h1 : x * y < 0) (h2 : x > |y|) : x + y > 0 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_positive_l3600_360085


namespace NUMINAMATH_CALUDE_min_value_2a_minus_ab_l3600_360071

def is_valid (a b : ℕ) : Prop := 0 < a ∧ a < 8 ∧ 0 < b ∧ b < 8

theorem min_value_2a_minus_ab :
  ∃ (a₀ b₀ : ℕ), is_valid a₀ b₀ ∧
  (∀ (a b : ℕ), is_valid a b → (2 * a - a * b : ℤ) ≥ (2 * a₀ - a₀ * b₀ : ℤ)) ∧
  (2 * a₀ - a₀ * b₀ : ℤ) = -35 :=
sorry

end NUMINAMATH_CALUDE_min_value_2a_minus_ab_l3600_360071


namespace NUMINAMATH_CALUDE_milk_for_cookies_l3600_360095

/-- Given that 18 cookies require 3 quarts of milk, 1 quart equals 2 pints,
    prove that 9 cookies require 3 pints of milk. -/
theorem milk_for_cookies (cookies_large : ℕ) (milk_quarts : ℕ) (cookies_small : ℕ) :
  cookies_large = 18 →
  milk_quarts = 3 →
  cookies_small = 9 →
  cookies_small * 2 = cookies_large →
  ∃ (milk_pints : ℕ),
    milk_pints = milk_quarts * 2 ∧
    milk_pints / 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_milk_for_cookies_l3600_360095


namespace NUMINAMATH_CALUDE_train_exit_time_l3600_360096

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Represents a train passing through a tunnel -/
structure TrainPassage where
  trainSpeed : Real  -- in km/h
  trainLength : Real  -- in km
  tunnelLength : Real  -- in km
  entryTime : Time

/-- Calculates the exit time of a train passing through a tunnel -/
def calculateExitTime (passage : TrainPassage) : Time :=
  sorry  -- Proof omitted

/-- Theorem stating that the given train leaves the tunnel at 6:05:15 am -/
theorem train_exit_time (passage : TrainPassage) 
  (h1 : passage.trainSpeed = 80)
  (h2 : passage.trainLength = 1)
  (h3 : passage.tunnelLength = 70)
  (h4 : passage.entryTime = ⟨5, 12, 0⟩) :
  calculateExitTime passage = ⟨6, 5, 15⟩ :=
by sorry

end NUMINAMATH_CALUDE_train_exit_time_l3600_360096


namespace NUMINAMATH_CALUDE_larger_number_proof_l3600_360042

theorem larger_number_proof (L S : ℕ) (hL : L > S) : 
  L - S = 1365 → L = 6 * S + 15 → L = 1635 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3600_360042


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3600_360039

theorem min_value_quadratic (x : ℝ) :
  let z := 4 * x^2 + 8 * x + 16
  ∀ y : ℝ, z ≤ y → 12 ≤ y :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3600_360039


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3600_360003

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = 2007 ∧ ∀ x, 3 * x^2 - 12 * x + 2023 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3600_360003


namespace NUMINAMATH_CALUDE_intersection_M_N_l3600_360034

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3600_360034


namespace NUMINAMATH_CALUDE_no_solution_to_system_l3600_360099

theorem no_solution_to_system :
  ¬∃ (x : ℝ), 
    (|Real.log x / Real.log 2| + (4 * x^2 / 15) - (16/15) = 0) ∧ 
    (Real.log (x + 2/3) / Real.log 7 + 12*x - 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l3600_360099


namespace NUMINAMATH_CALUDE_alyssa_games_last_year_l3600_360032

/-- The number of soccer games Alyssa attended last year -/
def games_last_year : ℕ := sorry

/-- The number of soccer games Alyssa attended this year -/
def games_this_year : ℕ := 11

/-- The number of soccer games Alyssa plans to attend next year -/
def games_next_year : ℕ := 15

/-- The total number of soccer games Alyssa will have attended -/
def total_games : ℕ := 39

/-- Theorem stating that Alyssa attended 13 soccer games last year -/
theorem alyssa_games_last_year : 
  games_last_year + games_this_year + games_next_year = total_games ∧ 
  games_last_year = 13 := by sorry

end NUMINAMATH_CALUDE_alyssa_games_last_year_l3600_360032


namespace NUMINAMATH_CALUDE_greatest_common_divisor_420_90_under_60_l3600_360040

theorem greatest_common_divisor_420_90_under_60 : 
  ∃ (n : ℕ), n ∣ 420 ∧ n ∣ 90 ∧ n < 60 ∧ 
  ∀ (m : ℕ), m ∣ 420 ∧ m ∣ 90 ∧ m < 60 → m ≤ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_420_90_under_60_l3600_360040


namespace NUMINAMATH_CALUDE_unique_prime_seventh_power_l3600_360052

theorem unique_prime_seventh_power (p : ℕ) : 
  Prime p ∧ ∃ q : ℕ, Prime q ∧ p + 25 = q^7 ↔ p = 103 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_seventh_power_l3600_360052


namespace NUMINAMATH_CALUDE_initial_list_size_l3600_360031

theorem initial_list_size (l : List Int) (m : ℚ) : 
  (((l.sum + 20) / (l.length + 1) : ℚ) = m + 3) →
  (((l.sum + 25) / (l.length + 2) : ℚ) = m + 1) →
  l.length = 3 := by
sorry

end NUMINAMATH_CALUDE_initial_list_size_l3600_360031


namespace NUMINAMATH_CALUDE_animal_ages_l3600_360002

theorem animal_ages (x : ℝ) 
  (h1 : 7 * (x - 3) = 2.5 * x - 3) : x + 2.5 * x = 14 := by
  sorry

end NUMINAMATH_CALUDE_animal_ages_l3600_360002


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_4_range_of_a_for_solutions_f_min_value_f_min_condition_l3600_360077

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem for part I
theorem solution_set_f_less_than_4 :
  {x : ℝ | f x < 4} = Set.Ioo (-2) 2 := by sorry

-- Theorem for part II
theorem range_of_a_for_solutions :
  {a : ℝ | ∃ x, f x - |a - 1| < 0} = Set.Iio (-1) ∪ Set.Ioi 3 := by sorry

-- Helper theorem: Minimum value of f is 2
theorem f_min_value :
  ∀ x : ℝ, f x ≥ 2 := by sorry

-- Helper theorem: Condition for f to achieve its minimum value
theorem f_min_condition (x : ℝ) :
  f x = 2 ↔ (x + 1) * (x - 1) ≤ 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_4_range_of_a_for_solutions_f_min_value_f_min_condition_l3600_360077


namespace NUMINAMATH_CALUDE_least_multiple_and_digit_sum_l3600_360059

def least_multiple_of_17_gt_500 : ℕ := 510

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem least_multiple_and_digit_sum :
  (least_multiple_of_17_gt_500 % 17 = 0) ∧
  (least_multiple_of_17_gt_500 > 500) ∧
  (∀ m : ℕ, m % 17 = 0 ∧ m > 500 → m ≥ least_multiple_of_17_gt_500) ∧
  (sum_of_digits least_multiple_of_17_gt_500 = 6) :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_and_digit_sum_l3600_360059


namespace NUMINAMATH_CALUDE_half_plus_five_equals_eleven_l3600_360033

theorem half_plus_five_equals_eleven : 
  (12 / 2 : ℚ) + 5 = 11 := by sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_eleven_l3600_360033


namespace NUMINAMATH_CALUDE_function_existence_l3600_360035

theorem function_existence (A B : Type) [Fintype A] [Fintype B]
  (hA : Fintype.card A = 2011^2) (hB : Fintype.card B = 2010) :
  ∃ f : A × A → B,
    (∀ x y : A, f (x, y) = f (y, x)) ∧
    (∀ g : A → B, ∃ a₁ a₂ : A, a₁ ≠ a₂ ∧ g a₁ = f (a₁, a₂) ∧ f (a₁, a₂) = g a₂) := by
  sorry

end NUMINAMATH_CALUDE_function_existence_l3600_360035


namespace NUMINAMATH_CALUDE_number_comparisons_l3600_360037

theorem number_comparisons :
  (π > 3.14) ∧ (-Real.sqrt 3 < -Real.sqrt 2) ∧ (2 < Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_number_comparisons_l3600_360037


namespace NUMINAMATH_CALUDE_grace_marks_calculation_l3600_360027

theorem grace_marks_calculation (num_students : ℕ) (initial_avg : ℚ) (final_avg : ℚ) 
  (h1 : num_students = 35)
  (h2 : initial_avg = 37)
  (h3 : final_avg = 40) :
  (num_students * final_avg - num_students * initial_avg) / num_students = 3 := by
  sorry

end NUMINAMATH_CALUDE_grace_marks_calculation_l3600_360027


namespace NUMINAMATH_CALUDE_rectangular_plot_longer_side_l3600_360097

theorem rectangular_plot_longer_side 
  (width : ℝ) 
  (num_poles : ℕ) 
  (pole_distance : ℝ) :
  width = 40 ∧ 
  num_poles = 36 ∧ 
  pole_distance = 5 →
  ∃ length : ℝ, 
    length > width ∧
    2 * (length + width) = (num_poles - 1 : ℝ) * pole_distance ∧
    length = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_longer_side_l3600_360097


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3600_360055

/-- An isosceles triangle with one side length of 3 and perimeter of 7 has equal sides of length 3 or 2 -/
theorem isosceles_triangle_side_length (a b c : ℝ) : 
  a + b + c = 7 →  -- perimeter is 7
  (a = 3 ∨ b = 3 ∨ c = 3) →  -- one side length is 3
  ((a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (a = c ∧ a ≠ b)) →  -- isosceles condition
  (a = 3 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (b = 3 ∧ c = 3) ∨ (b = 2 ∧ c = 2) ∨ (a = 3 ∧ c = 3) ∨ (a = 2 ∧ c = 2) := by
sorry


end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l3600_360055


namespace NUMINAMATH_CALUDE_building_height_average_l3600_360070

def measurements : List ℝ := [79.4, 80.6, 80.8, 79.1, 80, 79.6, 80.5]

theorem building_height_average : 
  (measurements.sum / measurements.length : ℝ) = 80 := by sorry

end NUMINAMATH_CALUDE_building_height_average_l3600_360070


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l3600_360061

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The problem statement -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 5 * x + 3 ↔ y = (3 * k) * x + 7) → k = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l3600_360061


namespace NUMINAMATH_CALUDE_james_training_hours_l3600_360020

/-- Represents James' training schedule and conditions --/
structure TrainingSchedule where
  daysInYear : Nat
  weekdayTrainingHours : Nat
  vacationWeeks : Nat
  injuryDays : Nat
  competitionDays : Nat

/-- Calculates the total training hours for James in a non-leap year --/
def calculateTrainingHours (schedule : TrainingSchedule) : Nat :=
  let weekdays := schedule.daysInYear - (52 * 2)
  let trainingDays := weekdays - (schedule.vacationWeeks * 5) - schedule.injuryDays - schedule.competitionDays
  let trainingWeeks := trainingDays / 5
  trainingWeeks * (5 * schedule.weekdayTrainingHours)

/-- Theorem stating that James' total training hours in a non-leap year is 1904 --/
theorem james_training_hours :
  let schedule : TrainingSchedule := {
    daysInYear := 365,
    weekdayTrainingHours := 8,
    vacationWeeks := 2,
    injuryDays := 5,
    competitionDays := 8
  }
  calculateTrainingHours schedule = 1904 := by
  sorry

end NUMINAMATH_CALUDE_james_training_hours_l3600_360020


namespace NUMINAMATH_CALUDE_negation_of_existence_l3600_360029

theorem negation_of_existence (x : ℝ) : 
  ¬(∃ x ≥ 0, x^2 - 2*x - 3 = 0) ↔ ∀ x ≥ 0, x^2 - 2*x - 3 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l3600_360029


namespace NUMINAMATH_CALUDE_num_a_animals_l3600_360015

def total_animals : ℕ := 17
def num_b_animals : ℕ := 8

theorem num_a_animals : total_animals - num_b_animals = 9 := by
  sorry

end NUMINAMATH_CALUDE_num_a_animals_l3600_360015


namespace NUMINAMATH_CALUDE_lunchroom_tables_l3600_360006

theorem lunchroom_tables (students_per_table : ℕ) (total_students : ℕ) 
  (h1 : students_per_table = 6)
  (h2 : total_students = 204)
  (h3 : total_students % students_per_table = 0) :
  total_students / students_per_table = 34 := by
sorry

end NUMINAMATH_CALUDE_lunchroom_tables_l3600_360006


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3600_360028

theorem rectangle_dimension_change (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  let new_L := 1.25 * L
  let new_W := W * (1 / 1.25)
  new_L * new_W = L * W ∧ (1 - new_W / W) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3600_360028


namespace NUMINAMATH_CALUDE_pastry_sale_revenue_l3600_360084

/-- Calculates the total money made from selling discounted pastries. -/
theorem pastry_sale_revenue (cupcake_price cookie_price : ℚ)
  (cupcakes_sold cookies_sold : ℕ) : 
  cupcake_price = 3 ∧ cookie_price = 2 ∧ cupcakes_sold = 16 ∧ cookies_sold = 8 →
  (cupcake_price / 2 * cupcakes_sold + cookie_price / 2 * cookies_sold : ℚ) = 32 := by
  sorry

#check pastry_sale_revenue

end NUMINAMATH_CALUDE_pastry_sale_revenue_l3600_360084


namespace NUMINAMATH_CALUDE_father_age_twice_marika_l3600_360057

/-- The year when Marika's father's age will be twice Marika's age -/
def target_year : ℕ := 2036

/-- Marika's age in 2006 -/
def marika_age_2006 : ℕ := 10

/-- The year of reference -/
def reference_year : ℕ := 2006

/-- Father's age is five times Marika's age in 2006 -/
def father_age_2006 : ℕ := 5 * marika_age_2006

theorem father_age_twice_marika (y : ℕ) :
  y = target_year →
  father_age_2006 + (y - reference_year) = 2 * (marika_age_2006 + (y - reference_year)) :=
by sorry

end NUMINAMATH_CALUDE_father_age_twice_marika_l3600_360057


namespace NUMINAMATH_CALUDE_first_row_chairs_l3600_360049

/-- Given a sequence of chair counts in rows, prove that the first row has 14 chairs. -/
theorem first_row_chairs (chairs : ℕ → ℕ) : 
  chairs 2 = 23 →                    -- Second row has 23 chairs
  (∀ n ≥ 2, chairs (n + 1) = chairs n + 9) →  -- Each subsequent row increases by 9
  chairs 6 = 59 →                    -- Sixth row has 59 chairs
  chairs 1 = 14 :=                   -- First row has 14 chairs
by sorry

end NUMINAMATH_CALUDE_first_row_chairs_l3600_360049


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l3600_360030

theorem triangle_angle_theorem (a : ℝ) (x : ℝ) :
  (5 < a) → (a < 35) →
  (2 * a + 20) + (3 * a - 15) + x = 180 →
  x = 175 - 5 * a ∧
  ∃ (ε : ℝ), ε > 0 ∧ 35 - ε > a ∧
  max (2 * a + 20) (max (3 * a - 15) (175 - 5 * a)) = 88 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l3600_360030


namespace NUMINAMATH_CALUDE_vector_addition_l3600_360081

def vector_AB : ℝ × ℝ := (1, 2)
def vector_BC : ℝ × ℝ := (3, 4)

theorem vector_addition :
  let vector_AC := (vector_AB.1 + vector_BC.1, vector_AB.2 + vector_BC.2)
  vector_AC = (4, 6) := by sorry

end NUMINAMATH_CALUDE_vector_addition_l3600_360081


namespace NUMINAMATH_CALUDE_milk_cost_l3600_360079

/-- The cost of a gallon of milk given the following conditions:
  * 4 pounds of coffee beans and 2 gallons of milk were bought
  * A pound of coffee beans costs $2.50
  * The total cost is $17
-/
theorem milk_cost (coffee_pounds : ℕ) (milk_gallons : ℕ) 
  (coffee_price : ℚ) (total_cost : ℚ) :
  coffee_pounds = 4 →
  milk_gallons = 2 →
  coffee_price = 5/2 →
  total_cost = 17 →
  ∃ (milk_price : ℚ), 
    milk_price * milk_gallons + coffee_price * coffee_pounds = total_cost ∧
    milk_price = 7/2 :=
by sorry

end NUMINAMATH_CALUDE_milk_cost_l3600_360079


namespace NUMINAMATH_CALUDE_absolute_value_w_l3600_360008

theorem absolute_value_w (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 2 / w = s) : 
  Complex.abs w = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_w_l3600_360008


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l3600_360067

def satisfies_conditions (n : ℕ) : Prop :=
  ∀ d : ℕ, 2 ≤ d → d ≤ 10 → n % d = d - 1

theorem smallest_satisfying_number : 
  satisfies_conditions 2519 ∧ 
  ∀ m : ℕ, m < 2519 → ¬(satisfies_conditions m) :=
sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l3600_360067


namespace NUMINAMATH_CALUDE_problem_statement_l3600_360011

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 5) :
  b / (a + b) + c / (b + c) + a / (c + a) = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3600_360011


namespace NUMINAMATH_CALUDE_stratified_sampling_result_l3600_360088

def grade10_students : ℕ := 300
def grade11_students : ℕ := 200
def grade12_students : ℕ := 400
def total_selected : ℕ := 18

def total_students : ℕ := grade10_students + grade11_students + grade12_students

def stratified_sample (grade_students : ℕ) : ℕ :=
  (total_selected * grade_students) / total_students

theorem stratified_sampling_result :
  (stratified_sample grade10_students,
   stratified_sample grade11_students,
   stratified_sample grade12_students) = (6, 4, 8) := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_result_l3600_360088


namespace NUMINAMATH_CALUDE_solve_equation_l3600_360010

theorem solve_equation : ∀ x : ℝ, x + 1 = 2 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3600_360010


namespace NUMINAMATH_CALUDE_tangent_line_and_inequalities_l3600_360056

noncomputable def f (x : ℝ) := x - x^2 + 3 * Real.log x

theorem tangent_line_and_inequalities :
  (∃ x₀ : ℝ, x₀ > 0 ∧ (∀ x > 0, f x ≤ 2 * x - 2) ∧
   (∀ k < 2, ∃ x₁ > 1, ∀ x ∈ Set.Ioo 1 x₁, f x ≥ k * (x - 1))) ∧
  (∃ a b : ℝ, ∀ x > 0, f x = 2 * x - 2 → x = a ∧ f x = b) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequalities_l3600_360056


namespace NUMINAMATH_CALUDE_abigail_savings_l3600_360012

/-- Calculates the monthly savings given the total savings and number of months. -/
def monthly_savings (total_savings : ℕ) (num_months : ℕ) : ℕ :=
  total_savings / num_months

/-- Theorem stating that given a total savings of 48000 over 12 months, 
    the monthly savings is 4000. -/
theorem abigail_savings : monthly_savings 48000 12 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_abigail_savings_l3600_360012


namespace NUMINAMATH_CALUDE_roots_sum_reciprocals_l3600_360014

theorem roots_sum_reciprocals (a b : ℝ) : 
  (a^2 - 3*a - 5 = 0) → 
  (b^2 - 3*b - 5 = 0) → 
  (a ≠ 0) →
  (b ≠ 0) →
  (1/a + 1/b = -3/5) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocals_l3600_360014


namespace NUMINAMATH_CALUDE_tensor_identity_implies_unit_vector_l3600_360048

def Vector2D := ℝ × ℝ

def tensor_product (m n : Vector2D) : Vector2D :=
  let (a, b) := m
  let (c, d) := n
  (a * c + b * d, a * d + b * c)

theorem tensor_identity_implies_unit_vector (p : Vector2D) :
  (∀ m : Vector2D, tensor_product m p = m) → p = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_tensor_identity_implies_unit_vector_l3600_360048


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3600_360058

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin (2 * α) + 2 * Real.cos (2 * α) = 2) : 
  Real.tan α = 1/2 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3600_360058


namespace NUMINAMATH_CALUDE_project_hours_difference_l3600_360080

theorem project_hours_difference (total_hours : ℕ) 
  (h_total : total_hours = 189) 
  (h_pat_kate : ∃ k : ℕ, pat = 2 * k ∧ kate = k) 
  (h_pat_mark : ∃ m : ℕ, mark = 3 * pat ∧ pat = m) 
  (h_sum : pat + kate + mark = total_hours) :
  mark - kate = 105 :=
by sorry

end NUMINAMATH_CALUDE_project_hours_difference_l3600_360080


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l3600_360065

theorem opposite_of_negative_three : -(- 3) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l3600_360065


namespace NUMINAMATH_CALUDE_unique_solution_system_l3600_360007

theorem unique_solution_system (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃! (x y z : ℝ), x + a * y + a^2 * z = 0 ∧
                   x + b * y + b^2 * z = 0 ∧
                   x + c * y + c^2 * z = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l3600_360007
