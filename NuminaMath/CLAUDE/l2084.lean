import Mathlib

namespace NUMINAMATH_CALUDE_tickets_left_l2084_208477

/-- The number of tickets Tom won from 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 32

/-- The number of tickets Tom won from 'skee ball' -/
def skee_ball_tickets : ℕ := 25

/-- The number of tickets Tom spent on a hat -/
def spent_tickets : ℕ := 7

/-- Theorem: Given the conditions, Tom has 50 tickets left -/
theorem tickets_left : 
  whack_a_mole_tickets + skee_ball_tickets - spent_tickets = 50 := by
  sorry

end NUMINAMATH_CALUDE_tickets_left_l2084_208477


namespace NUMINAMATH_CALUDE_expression_percentage_of_y_l2084_208446

theorem expression_percentage_of_y (y z : ℝ) (hy : y > 0) :
  ((2 * y + z) / 10 + (3 * y - z) / 10) / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_percentage_of_y_l2084_208446


namespace NUMINAMATH_CALUDE_container_capacity_l2084_208489

theorem container_capacity (initial_fill : Real) (added_water : Real) (final_fill : Real) :
  initial_fill = 0.3 →
  added_water = 18 →
  final_fill = 0.75 →
  ∃ capacity : Real, 
    capacity * final_fill - capacity * initial_fill = added_water ∧
    capacity = 40 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l2084_208489


namespace NUMINAMATH_CALUDE_ending_number_of_range_problem_solution_l2084_208422

theorem ending_number_of_range (start : Nat) (count : Nat) (divisor : Nat) : Nat :=
  let first_multiple := ((start + divisor - 1) / divisor) * divisor
  first_multiple + (count - 1) * divisor

theorem problem_solution : 
  ending_number_of_range 49 3 11 = 77 := by
  sorry

end NUMINAMATH_CALUDE_ending_number_of_range_problem_solution_l2084_208422


namespace NUMINAMATH_CALUDE_ground_lines_perpendicular_l2084_208498

-- Define a type for lines
def Line : Type := ℝ × ℝ → Prop

-- Define a relation for parallel lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- Define a relation for perpendicular lines
def Perpendicular (l1 l2 : Line) : Prop := sorry

-- Define a set of lines on the ground
def GroundLines : Set Line := sorry

-- Define the ruler's orientation
def RulerOrientation : Line := sorry

-- Theorem statement
theorem ground_lines_perpendicular 
  (always_parallel : ∀ (r : Line), ∃ (g : Line), g ∈ GroundLines ∧ Parallel r g) :
  ∀ (l1 l2 : Line), l1 ∈ GroundLines → l2 ∈ GroundLines → l1 ≠ l2 → Perpendicular l1 l2 :=
sorry

end NUMINAMATH_CALUDE_ground_lines_perpendicular_l2084_208498


namespace NUMINAMATH_CALUDE_circle_radius_l2084_208491

theorem circle_radius (x y : ℝ) : 
  (∀ x y, x^2 - 6*x + y^2 + 2*y + 6 = 0) → 
  ∃ r : ℝ, r = 2 ∧ ∀ x y, (x - 3)^2 + (y + 1)^2 = r^2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_l2084_208491


namespace NUMINAMATH_CALUDE_gcd_282_470_l2084_208456

theorem gcd_282_470 : Nat.gcd 282 470 = 94 := by
  sorry

end NUMINAMATH_CALUDE_gcd_282_470_l2084_208456


namespace NUMINAMATH_CALUDE_work_left_fraction_l2084_208495

/-- The fraction of work left after two workers work together for a given number of days -/
def fraction_left (days_a : ℕ) (days_b : ℕ) (days_together : ℕ) : ℚ :=
  1 - (days_together : ℚ) * ((1 : ℚ) / days_a + (1 : ℚ) / days_b)

/-- Theorem: Given A can do a job in 15 days and B in 20 days, if they work together for 3 days, 
    the fraction of work left is 13/20 -/
theorem work_left_fraction : fraction_left 15 20 3 = 13 / 20 := by
  sorry

end NUMINAMATH_CALUDE_work_left_fraction_l2084_208495


namespace NUMINAMATH_CALUDE_remainder_sum_l2084_208493

theorem remainder_sum (c d : ℤ) :
  (∃ p : ℤ, c = 84 * p + 76) →
  (∃ q : ℤ, d = 126 * q + 117) →
  (c + d) % 42 = 25 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l2084_208493


namespace NUMINAMATH_CALUDE_unique_integer_product_digits_l2084_208452

/-- ProductOfDigits calculates the product of digits for a given natural number -/
def ProductOfDigits (n : ℕ) : ℕ := sorry

/-- The theorem states that 84 is the unique positive integer k such that 
    the product of its digits is equal to (11k/4) - 199 -/
theorem unique_integer_product_digits : 
  ∃! (k : ℕ), k > 0 ∧ ProductOfDigits k = (11 * k) / 4 - 199 := by sorry

end NUMINAMATH_CALUDE_unique_integer_product_digits_l2084_208452


namespace NUMINAMATH_CALUDE_regular_polygon_15_diagonals_l2084_208480

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 15 diagonals has 7 sides -/
theorem regular_polygon_15_diagonals :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 15 ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_15_diagonals_l2084_208480


namespace NUMINAMATH_CALUDE_largest_area_triangle_l2084_208419

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Checks if a point is an internal point of a line segment -/
def isInternalPoint (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := sorry

/-- Checks if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop := sorry

/-- Calculates the area of a triangle -/
def triangleArea (T : Triangle) : ℝ := sorry

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Represents an arc of a circle -/
structure Arc :=
  (circle : Circle)
  (startAngle endAngle : ℝ)

/-- Finds the intersection point of two circles -/
def circleIntersection (c1 c2 : Circle) : Option (ℝ × ℝ) := sorry

/-- Calculates the distance between two points -/
def distance (P1 P2 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem largest_area_triangle
  (A₀B₀C₀ : Triangle)
  (A'B'C' : Triangle)
  (k_a k_c : Circle)
  (i_a i_c : Arc) :
  ∀ (ABC : Triangle),
    isInternalPoint A₀B₀C₀.C ABC.A ABC.B →
    isInternalPoint A₀B₀C₀.A ABC.B ABC.C →
    isInternalPoint A₀B₀C₀.B ABC.C ABC.A →
    areSimilar ABC A'B'C' →
    (∃ (M : ℝ × ℝ), circleIntersection k_a k_c = some M) →
    (∀ (ABC' : Triangle),
      isInternalPoint A₀B₀C₀.C ABC'.A ABC'.B →
      isInternalPoint A₀B₀C₀.A ABC'.B ABC'.C →
      isInternalPoint A₀B₀C₀.B ABC'.C ABC'.A →
      areSimilar ABC' A'B'C' →
      (∃ (M' : ℝ × ℝ), circleIntersection k_a k_c = some M') →
      distance M ABC.C + distance M ABC.A ≥ distance M' ABC'.C + distance M' ABC'.A) →
    ∀ (ABC' : Triangle),
      isInternalPoint A₀B₀C₀.C ABC'.A ABC'.B →
      isInternalPoint A₀B₀C₀.A ABC'.B ABC'.C →
      isInternalPoint A₀B₀C₀.B ABC'.C ABC'.A →
      areSimilar ABC' A'B'C' →
      triangleArea ABC ≥ triangleArea ABC' :=
by
  sorry

end NUMINAMATH_CALUDE_largest_area_triangle_l2084_208419


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2084_208482

theorem polynomial_simplification (s : ℝ) : 
  (2 * s^2 + 5 * s - 3) - (2 * s^2 + 9 * s - 6) = -4 * s + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2084_208482


namespace NUMINAMATH_CALUDE_cup_arrangement_theorem_l2084_208450

/-- Represents the number of ways to arrange cups in a circular pattern -/
def circularArrangements (yellow blue red : ℕ) : ℕ := sorry

/-- Represents the number of ways to arrange cups in a circular pattern with adjacent red cups -/
def circularArrangementsAdjacentRed (yellow blue red : ℕ) : ℕ := sorry

/-- The main theorem stating the number of valid arrangements -/
theorem cup_arrangement_theorem :
  circularArrangements 4 3 2 - circularArrangementsAdjacentRed 4 3 2 = 105 := by
  sorry

end NUMINAMATH_CALUDE_cup_arrangement_theorem_l2084_208450


namespace NUMINAMATH_CALUDE_daily_forfeit_is_25_l2084_208429

/-- Calculates the daily forfeit amount for idle days given work conditions --/
def calculate_daily_forfeit (daily_pay : ℕ) (total_days : ℕ) (net_earnings : ℕ) (worked_days : ℕ) : ℕ :=
  let idle_days := total_days - worked_days
  let total_possible_earnings := daily_pay * total_days
  let total_forfeit := total_possible_earnings - net_earnings
  total_forfeit / idle_days

/-- Proves that the daily forfeit amount is 25 dollars given the specific work conditions --/
theorem daily_forfeit_is_25 :
  calculate_daily_forfeit 20 25 450 23 = 25 := by
  sorry

end NUMINAMATH_CALUDE_daily_forfeit_is_25_l2084_208429


namespace NUMINAMATH_CALUDE_factors_of_24_l2084_208404

theorem factors_of_24 : Finset.card (Nat.divisors 24) = 8 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_24_l2084_208404


namespace NUMINAMATH_CALUDE_symmetric_line_theorem_l2084_208476

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two points are symmetric with respect to a vertical line -/
def symmetric_points (x₁ y₁ x₂ y₂ x_sym : ℝ) : Prop :=
  x₁ + x₂ = 2 * x_sym ∧ y₁ = y₂

/-- Check if a point (x, y) lies on a given line -/
def point_on_line (x y : ℝ) (l : Line) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are symmetric with respect to a vertical line -/
def symmetric_lines (l₁ l₂ : Line) (x_sym : ℝ) : Prop :=
  ∀ x₁ y₁, point_on_line x₁ y₁ l₁ →
    ∃ x₂ y₂, point_on_line x₂ y₂ l₂ ∧ symmetric_points x₁ y₁ x₂ y₂ x_sym

theorem symmetric_line_theorem :
  let l₁ : Line := ⟨2, -1, 1⟩
  let l₂ : Line := ⟨2, 1, -5⟩
  let x_sym : ℝ := 1
  symmetric_lines l₁ l₂ x_sym := by sorry

end NUMINAMATH_CALUDE_symmetric_line_theorem_l2084_208476


namespace NUMINAMATH_CALUDE_phd_time_ratio_l2084_208459

/-- Represents the time spent in years for each phase of John's PhD journey -/
structure PhDTime where
  total : ℝ
  acclimation : ℝ
  basics : ℝ
  research : ℝ
  dissertation : ℝ

/-- Theorem stating the ratio of dissertation writing time to acclimation time -/
theorem phd_time_ratio (t : PhDTime) : 
  t.total = 7 ∧ 
  t.acclimation = 1 ∧ 
  t.basics = 2 ∧ 
  t.research = t.basics * 1.75 ∧
  t.total = t.acclimation + t.basics + t.research + t.dissertation →
  t.dissertation / t.acclimation = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_phd_time_ratio_l2084_208459


namespace NUMINAMATH_CALUDE_fraction_simplification_l2084_208468

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) :
  (x^2 - 1) / (x^2 - 2*x + 1) = (x + 1) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2084_208468


namespace NUMINAMATH_CALUDE_perpendicular_parallel_transitivity_l2084_208449

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_transitivity 
  (l : Line) (α β : Plane) :
  perp l α → para α β → perp l β :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_transitivity_l2084_208449


namespace NUMINAMATH_CALUDE_carpooling_distance_ratio_l2084_208432

/-- Proves that the ratio of the distance driven between the second friend's house and work
    to the total distance driven to the first and second friend's houses is 3:1 -/
theorem carpooling_distance_ratio :
  let distance_to_first : ℝ := 8
  let distance_to_second : ℝ := distance_to_first / 2
  let distance_to_work : ℝ := 36
  let total_distance_to_friends : ℝ := distance_to_first + distance_to_second
  (distance_to_work / total_distance_to_friends) = 3
  := by sorry

end NUMINAMATH_CALUDE_carpooling_distance_ratio_l2084_208432


namespace NUMINAMATH_CALUDE_distinct_weights_count_l2084_208403

def weights : List ℕ := [1, 2, 3, 4]

def possible_combinations (weights : List ℕ) : List (List ℕ) :=
  sorry

def distinct_weights (combinations : List (List ℕ)) : List ℕ :=
  sorry

theorem distinct_weights_count :
  weights.length = 4 →
  (distinct_weights (possible_combinations weights)).length = 10 :=
by sorry

end NUMINAMATH_CALUDE_distinct_weights_count_l2084_208403


namespace NUMINAMATH_CALUDE_bianca_received_30_dollars_l2084_208434

/-- The amount of money Bianca received for her birthday -/
def biancas_birthday_money (num_friends : ℕ) (dollars_per_friend : ℕ) : ℕ :=
  num_friends * dollars_per_friend

/-- Theorem stating that Bianca received 30 dollars for her birthday -/
theorem bianca_received_30_dollars :
  biancas_birthday_money 5 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_bianca_received_30_dollars_l2084_208434


namespace NUMINAMATH_CALUDE_even_product_probability_l2084_208423

-- Define the spinners
def spinner1 : List ℕ := [2, 4, 6, 8]
def spinner2 : List ℕ := [1, 3, 5, 7, 9]

-- Define a function to check if a number is even
def isEven (n : ℕ) : Bool := n % 2 = 0

-- Define the probability function
def probabilityEvenProduct (s1 s2 : List ℕ) : ℚ :=
  let totalOutcomes := (s1.length * s2.length : ℚ)
  let evenOutcomes := (s1.filter isEven).length * s2.length
  evenOutcomes / totalOutcomes

-- Theorem statement
theorem even_product_probability :
  probabilityEvenProduct spinner1 spinner2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_product_probability_l2084_208423


namespace NUMINAMATH_CALUDE_brendan_tax_payment_l2084_208445

/-- Calculates the weekly tax payment for a waiter named Brendan --/
def brendan_weekly_tax (hourly_wage : ℝ) (shift_hours : List ℝ) (tip_per_hour : ℝ) (tax_rate : ℝ) (tip_report_ratio : ℝ) : ℝ :=
  let total_hours := shift_hours.sum
  let wage_income := hourly_wage * total_hours
  let total_tips := tip_per_hour * total_hours
  let reported_tips := total_tips * tip_report_ratio
  let reported_income := wage_income + reported_tips
  reported_income * tax_rate

/-- Theorem stating that Brendan's weekly tax payment is $56 --/
theorem brendan_tax_payment :
  brendan_weekly_tax 6 [8, 8, 12] 12 0.2 (1/3) = 56 := by
  sorry

end NUMINAMATH_CALUDE_brendan_tax_payment_l2084_208445


namespace NUMINAMATH_CALUDE_students_in_diligence_before_transfer_l2084_208444

theorem students_in_diligence_before_transfer 
  (total_students : ℕ) 
  (transferred_students : ℕ) 
  (h1 : total_students = 50)
  (h2 : transferred_students = 2)
  (h3 : ∃ (x : ℕ), x + transferred_students = total_students - x) :
  ∃ (initial_diligence : ℕ), initial_diligence = (total_students / 2) - transferred_students :=
sorry

end NUMINAMATH_CALUDE_students_in_diligence_before_transfer_l2084_208444


namespace NUMINAMATH_CALUDE_intersection_point_l2084_208401

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = x + 3

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Theorem statement
theorem intersection_point :
  ∃ (x y : ℝ), line_equation x y ∧ y_axis x ∧ x = 0 ∧ y = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l2084_208401


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2084_208454

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 733 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l2084_208454


namespace NUMINAMATH_CALUDE_dual_expression_problem_l2084_208457

theorem dual_expression_problem (x : ℝ) :
  (Real.sqrt (20 - x) + Real.sqrt (4 - x) = 8) →
  (Real.sqrt (20 - x) - Real.sqrt (4 - x) = 2) ∧ (x = -5) := by
  sorry


end NUMINAMATH_CALUDE_dual_expression_problem_l2084_208457


namespace NUMINAMATH_CALUDE_solution_replacement_l2084_208488

theorem solution_replacement (initial_volume : ℝ) (initial_concentration : ℝ) 
  (replacement_concentration : ℝ) (final_concentration : ℝ) 
  (h1 : initial_volume = 100)
  (h2 : initial_concentration = 0.4)
  (h3 : replacement_concentration = 0.25)
  (h4 : final_concentration = 0.35) :
  ∃ (replaced_volume : ℝ), 
    replaced_volume / initial_volume = 1 / 3 ∧
    initial_volume * initial_concentration - replaced_volume * initial_concentration + 
    replaced_volume * replacement_concentration = 
    initial_volume * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_solution_replacement_l2084_208488


namespace NUMINAMATH_CALUDE_larry_jogging_time_l2084_208448

/-- Represents the number of days Larry jogs in the first week -/
def days_first_week : ℕ := 3

/-- Represents the number of days Larry jogs in the second week -/
def days_second_week : ℕ := 5

/-- Represents the total number of hours Larry jogs in two weeks -/
def total_hours : ℕ := 4

/-- Calculates the total number of days Larry jogs in two weeks -/
def total_days : ℕ := days_first_week + days_second_week

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℕ) : ℕ := hours * 60

/-- Represents the total jogging time in minutes -/
def total_minutes : ℕ := hours_to_minutes total_hours

/-- Theorem: Larry jogs for 30 minutes each day -/
theorem larry_jogging_time :
  total_minutes / total_days = 30 := by sorry

end NUMINAMATH_CALUDE_larry_jogging_time_l2084_208448


namespace NUMINAMATH_CALUDE_initial_crayons_l2084_208410

theorem initial_crayons (initial final added : ℕ) : 
  final = initial + added → 
  added = 6 → 
  final = 13 → 
  initial = 7 := by 
sorry

end NUMINAMATH_CALUDE_initial_crayons_l2084_208410


namespace NUMINAMATH_CALUDE_cow_count_is_six_l2084_208481

/-- Represents the number of animals in a group -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of heads in the group -/
def totalHeads (g : AnimalGroup) : ℕ := g.ducks + g.cows

/-- Calculates the total number of legs in the group -/
def totalLegs (g : AnimalGroup) : ℕ := 2 * g.ducks + 4 * g.cows

/-- Theorem: If the total number of legs is 12 more than twice the number of heads,
    then the number of cows is 6 -/
theorem cow_count_is_six (g : AnimalGroup) :
  totalLegs g = 2 * totalHeads g + 12 → g.cows = 6 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_is_six_l2084_208481


namespace NUMINAMATH_CALUDE_elder_son_toys_l2084_208406

theorem elder_son_toys (total : ℕ) (younger_ratio : ℕ) : 
  total = 240 → younger_ratio = 3 → 
  ∃ (elder : ℕ), elder * (1 + younger_ratio) = total ∧ elder = 60 := by
sorry

end NUMINAMATH_CALUDE_elder_son_toys_l2084_208406


namespace NUMINAMATH_CALUDE_expression_equality_l2084_208418

theorem expression_equality : (-3)^2 - Real.sqrt 4 + (1/2)⁻¹ = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2084_208418


namespace NUMINAMATH_CALUDE_problem_solution_l2084_208460

theorem problem_solution (k : ℕ) (y : ℚ) 
  (h1 : (1/2)^18 * (1/81)^k = y)
  (h2 : k = 9) : 
  y = 1 / (2^18 * 3^36) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2084_208460


namespace NUMINAMATH_CALUDE_bottles_left_l2084_208474

theorem bottles_left (initial : Real) (maria_drank : Real) (sister_drank : Real) 
  (h1 : initial = 45.0)
  (h2 : maria_drank = 14.0)
  (h3 : sister_drank = 8.0) :
  initial - maria_drank - sister_drank = 23.0 := by
  sorry

end NUMINAMATH_CALUDE_bottles_left_l2084_208474


namespace NUMINAMATH_CALUDE_domain_f_real_iff_a_gt_one_range_f_real_iff_a_between_zero_and_one_decreasing_interval_g_l2084_208463

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) := Real.log (a * x^2 + 2 * x + 1)
noncomputable def g (x : ℝ) := Real.log (x^2 - 4 * x - 5) / Real.log (1/2)

-- Theorem 1: Domain of f is ℝ iff a > 1
theorem domain_f_real_iff_a_gt_one (a : ℝ) :
  (∀ x, ∃ y, f a x = y) ↔ a > 1 :=
sorry

-- Theorem 2: Range of f is ℝ iff 0 ≤ a ≤ 1
theorem range_f_real_iff_a_between_zero_and_one (a : ℝ) :
  (∀ y, ∃ x, f a x = y) ↔ 0 ≤ a ∧ a ≤ 1 :=
sorry

-- Theorem 3: Decreasing interval of g is (5, +∞)
theorem decreasing_interval_g :
  ∀ x₁ x₂, x₁ > 5 → x₂ > 5 → x₁ < x₂ → g x₁ > g x₂ :=
sorry

end NUMINAMATH_CALUDE_domain_f_real_iff_a_gt_one_range_f_real_iff_a_between_zero_and_one_decreasing_interval_g_l2084_208463


namespace NUMINAMATH_CALUDE_train_passing_time_l2084_208464

/-- Proves that a train of given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 500 → 
  train_speed_kmh = 72 → 
  passing_time = 25 → 
  passing_time = train_length / (train_speed_kmh * (5/18)) := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l2084_208464


namespace NUMINAMATH_CALUDE_diamond_properties_l2084_208478

def diamond (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem diamond_properties :
  (∀ x y : ℝ, diamond x y = diamond y x) ∧
  (∃ x y : ℝ, 2 * (diamond x y) ≠ diamond (2*x) (2*y)) ∧
  (∀ x : ℝ, diamond x 0 = x^2) ∧
  (∀ x : ℝ, diamond x x = 0) ∧
  (∀ x y : ℝ, x = y → diamond x y = 0) :=
by sorry

end NUMINAMATH_CALUDE_diamond_properties_l2084_208478


namespace NUMINAMATH_CALUDE_bus_route_distance_bounds_l2084_208439

/-- Represents a bus route with n stops -/
structure BusRoute (n : ℕ) where
  distance_between_stops : ℝ
  (distance_positive : distance_between_stops > 0)

/-- Represents a vehicle's journey through all stops -/
def Journey (n : ℕ) := Fin n → Fin n

/-- Calculates the distance traveled in a journey -/
def distance_traveled (r : BusRoute n) (j : Journey n) : ℝ :=
  sorry

/-- Theorem stating the maximum and minimum distances for a 10-stop route -/
theorem bus_route_distance_bounds :
  ∀ (r : BusRoute 10),
    (∃ (j : Journey 10), distance_traveled r j = 50 * r.distance_between_stops) ∧
    (∃ (j : Journey 10), distance_traveled r j = 18 * r.distance_between_stops) ∧
    (∀ (j : Journey 10), 18 * r.distance_between_stops ≤ distance_traveled r j ∧ 
                         distance_traveled r j ≤ 50 * r.distance_between_stops) :=
sorry

end NUMINAMATH_CALUDE_bus_route_distance_bounds_l2084_208439


namespace NUMINAMATH_CALUDE_chi_square_relationship_certainty_l2084_208412

-- Define the Chi-square test result
def chi_square_result : ℝ := 6.825

-- Define the degrees of freedom for a 2x2 contingency table
def degrees_of_freedom : ℕ := 1

-- Define the critical value for 99% confidence level
def critical_value : ℝ := 6.635

-- Define the certainty level
def certainty_level : ℝ := 0.99

-- Theorem statement
theorem chi_square_relationship_certainty :
  chi_square_result > critical_value →
  certainty_level = 0.99 :=
sorry

end NUMINAMATH_CALUDE_chi_square_relationship_certainty_l2084_208412


namespace NUMINAMATH_CALUDE_jungkook_has_largest_number_l2084_208431

/-- Given the numbers collected by Yoongi, Yuna, and Jungkook, prove that Jungkook has the largest number. -/
theorem jungkook_has_largest_number (yoongi_number yuna_number : ℕ) : 
  yoongi_number = 4 → 
  yuna_number = 5 → 
  6 + 3 > yoongi_number ∧ 6 + 3 > yuna_number := by
  sorry

#check jungkook_has_largest_number

end NUMINAMATH_CALUDE_jungkook_has_largest_number_l2084_208431


namespace NUMINAMATH_CALUDE_power_equation_solution_l2084_208497

theorem power_equation_solution : ∃ k : ℕ, 3 * 2^2001 - 3 * 2^2000 - 2^1999 + 2^1998 = k * 2^1998 ∧ k = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2084_208497


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_relation_l2084_208494

/-- In an isosceles right triangle ABC with right angle at A, 
    if CB = CA = h, BM + MA = 2(BC + CA), and MB = x, then x = 7h/5 -/
theorem isosceles_right_triangle_relation (h x : ℝ) : 
  h > 0 → 
  x > 0 → 
  x + Real.sqrt ((x + h)^2 + h^2) = 4 * h → 
  x = (7 * h) / 5 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_relation_l2084_208494


namespace NUMINAMATH_CALUDE_smaller_cube_edge_length_l2084_208486

/-- Given a cube with volume 1000 cm³ divided into 8 equal smaller cubes,
    prove that the edge length of each smaller cube is 5 cm. -/
theorem smaller_cube_edge_length :
  ∀ (original_volume smaller_volume : ℝ) (original_edge smaller_edge : ℝ),
  original_volume = 1000 →
  smaller_volume = original_volume / 8 →
  original_volume = original_edge ^ 3 →
  smaller_volume = smaller_edge ^ 3 →
  smaller_edge = 5 := by
sorry

end NUMINAMATH_CALUDE_smaller_cube_edge_length_l2084_208486


namespace NUMINAMATH_CALUDE_cubic_inequality_false_l2084_208415

theorem cubic_inequality_false (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  ¬(a^3 > b^3) := by
sorry

end NUMINAMATH_CALUDE_cubic_inequality_false_l2084_208415


namespace NUMINAMATH_CALUDE_article_cost_price_l2084_208455

/-- Proves that the cost price of an article is 600, given the conditions of the problem -/
theorem article_cost_price : ∃ (C : ℝ), 
  (∃ (S : ℝ), S = 1.05 * C) ∧ 
  (1.05 * C - 3 = 1.10 * (0.95 * C)) ∧
  C = 600 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_price_l2084_208455


namespace NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l2084_208400

/-- A Pythagorean triple is a set of three positive integers (a, b, c) where a² + b² = c² -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The set (5, 12, 13) is a Pythagorean triple -/
theorem five_twelve_thirteen_pythagorean_triple : isPythagoreanTriple 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l2084_208400


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2084_208438

/-- Two circles with equations x^2 + y^2 + 2ax + a^2 - 9 = 0 and x^2 + y^2 - 4by - 1 + 4b^2 = 0 -/
def Circle1 (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*a*x + a^2 - 9 = 0
def Circle2 (b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

/-- The circles have three shared tangents -/
axiom three_shared_tangents (a b : ℝ) : ∃ (t1 t2 t3 : ℝ × ℝ → ℝ), 
  (∀ x y, Circle1 a x y → t1 (x, y) = 0) ∧
  (∀ x y, Circle2 b x y → t1 (x, y) = 0) ∧
  (∀ x y, Circle1 a x y → t2 (x, y) = 0) ∧
  (∀ x y, Circle2 b x y → t2 (x, y) = 0) ∧
  (∀ x y, Circle1 a x y → t3 (x, y) = 0) ∧
  (∀ x y, Circle2 b x y → t3 (x, y) = 0)

/-- The theorem to be proved -/
theorem min_value_of_expression (a b : ℝ) (h : a * b ≠ 0) : 
  (∃ (x y : ℝ), Circle1 a x y) → 
  (∃ (x y : ℝ), Circle2 b x y) → 
  (∀ c, c ≥ 1 → 4 / a^2 + 1 / b^2 ≥ c) ∧ 
  (∃ a0 b0, a0 * b0 ≠ 0 ∧ 4 / a0^2 + 1 / b0^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2084_208438


namespace NUMINAMATH_CALUDE_list_price_problem_l2084_208417

theorem list_price_problem (list_price : ℝ) : 
  (0.15 * (list_price - 15) = 0.30 * (list_price - 25)) → list_price = 35 := by
  sorry

end NUMINAMATH_CALUDE_list_price_problem_l2084_208417


namespace NUMINAMATH_CALUDE_max_surface_area_cuboid_in_sphere_l2084_208405

/-- The maximum surface area of a rectangular cuboid inscribed in a sphere -/
theorem max_surface_area_cuboid_in_sphere :
  ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  (a^2 + b^2 + c^2 : ℝ) = 25 →
  2 * (a * b + a * c + b * c) ≤ 50 :=
by sorry

end NUMINAMATH_CALUDE_max_surface_area_cuboid_in_sphere_l2084_208405


namespace NUMINAMATH_CALUDE_equilateral_triangle_in_ellipse_ratio_l2084_208496

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- An equilateral triangle -/
structure EquilateralTriangle where
  A : Point
  B : Point
  C : Point

/-- The foci of an ellipse -/
structure EllipseFoci where
  F1 : Point
  F2 : Point

/-- Theorem: Ratio of side length to focal distance in special equilateral triangle -/
theorem equilateral_triangle_in_ellipse_ratio 
  (e : Ellipse) 
  (t : EquilateralTriangle) 
  (f : EllipseFoci) :
  (t.B.x = 0 ∧ t.B.y = e.b) →  -- B is at (0, q)
  (t.A.y = t.C.y) →  -- AC is parallel to x-axis
  (f.F1.x - t.B.x) * (t.C.x - t.B.x) + (f.F1.y - t.B.y) * (t.C.y - t.B.y) = 0 →  -- F1 is on BC
  (f.F2.x - t.A.x) * (t.B.x - t.A.x) + (f.F2.y - t.A.y) * (t.B.y - t.A.y) = 0 →  -- F2 is on AB
  (f.F2.x - f.F1.x)^2 + (f.F2.y - f.F1.y)^2 = 16 →  -- F1F2 = 4
  (t.B.x - t.A.x)^2 + (t.B.y - t.A.y)^2 = 
    (2 * Real.sqrt 3 / 5)^2 * ((f.F2.x - f.F1.x)^2 + (f.F2.y - f.F1.y)^2) := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_in_ellipse_ratio_l2084_208496


namespace NUMINAMATH_CALUDE_expansion_coefficient_equation_l2084_208442

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^2 in the expansion of (√x + a)^6
def coefficient_x_squared (a : ℝ) : ℝ := binomial 6 2 * a^2

-- State the theorem
theorem expansion_coefficient_equation :
  ∃ a : ℝ, coefficient_x_squared a = 60 ∧ (a = 2 ∨ a = -2) :=
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_equation_l2084_208442


namespace NUMINAMATH_CALUDE_proposition_p_true_q_false_l2084_208490

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define a triangle
structure Triangle :=
(A B C : ℝ)
(angle_sum : A + B + C = π)
(positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

theorem proposition_p_true_q_false :
  (∀ x : ℝ, 0 < x → x < 1 → lg (x * (1 - x) + 1) > 0) ∧
  (∃ t : Triangle, t.A > t.B ∧ Real.cos (t.A / 2)^2 ≥ Real.cos (t.B / 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_true_q_false_l2084_208490


namespace NUMINAMATH_CALUDE_magazine_boxes_l2084_208461

theorem magazine_boxes (total_magazines : ℕ) (magazines_per_box : ℚ) : 
  total_magazines = 150 → magazines_per_box = 11.5 → 
  ⌈(total_magazines : ℚ) / magazines_per_box⌉ = 14 := by
  sorry

end NUMINAMATH_CALUDE_magazine_boxes_l2084_208461


namespace NUMINAMATH_CALUDE_probability_of_convex_quadrilateral_l2084_208427

/-- The number of points on the circle -/
def n : ℕ := 6

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords between n points -/
def total_chords : ℕ := n.choose 2

/-- The probability of four randomly selected chords forming a convex quadrilateral -/
def probability : ℚ := (n.choose k : ℚ) / (total_chords.choose k : ℚ)

/-- Theorem stating the probability is 1/91 -/
theorem probability_of_convex_quadrilateral : probability = 1 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_convex_quadrilateral_l2084_208427


namespace NUMINAMATH_CALUDE_student_volunteer_arrangements_l2084_208409

theorem student_volunteer_arrangements :
  let n : ℕ := 5  -- number of students
  let k : ℕ := 2  -- number of communities
  (2^n : ℕ) - k = 30 :=
by sorry

end NUMINAMATH_CALUDE_student_volunteer_arrangements_l2084_208409


namespace NUMINAMATH_CALUDE_remainder_comparison_l2084_208485

theorem remainder_comparison (P P' : ℕ) (h1 : P = P' + 10) (h2 : P % 10 = 0) (h3 : P' % 10 = 0) :
  (P^2 - P'^2) % 10 = 0 ∧ 0 % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_comparison_l2084_208485


namespace NUMINAMATH_CALUDE_bottles_per_case_is_13_l2084_208407

/-- The number of bottles of water a company produces per day -/
def daily_production : ℕ := 65000

/-- The number of cases required to hold the daily production -/
def cases_required : ℕ := 5000

/-- The number of bottles that a single case can hold -/
def bottles_per_case : ℕ := daily_production / cases_required

theorem bottles_per_case_is_13 : bottles_per_case = 13 := by
  sorry

end NUMINAMATH_CALUDE_bottles_per_case_is_13_l2084_208407


namespace NUMINAMATH_CALUDE_all_solutions_are_powers_l2084_208408

-- Define the equation (1) as a predicate
def is_solution (p q : ℤ) : Prop := sorry

-- Define the main theorem
theorem all_solutions_are_powers (p q : ℤ) :
  p ≥ 0 ∧ q ≥ 0 ∧ is_solution p q ↔ ∃ n : ℕ, p + q * Real.sqrt 5 = (9 + 4 * Real.sqrt 5) ^ n := by
  sorry

end NUMINAMATH_CALUDE_all_solutions_are_powers_l2084_208408


namespace NUMINAMATH_CALUDE_students_without_A_count_l2084_208425

/-- Represents the number of students who received an A in a specific combination of subjects -/
structure GradeDistribution where
  total : Nat
  history : Nat
  math : Nat
  computing : Nat
  historyAndMath : Nat
  historyAndComputing : Nat
  mathAndComputing : Nat
  allThree : Nat

/-- Calculates the number of students who didn't receive an A in any subject -/
def studentsWithoutA (g : GradeDistribution) : Nat :=
  g.total - (g.history + g.math + g.computing - g.historyAndMath - g.historyAndComputing - g.mathAndComputing + g.allThree)

theorem students_without_A_count (g : GradeDistribution) 
  (h_total : g.total = 40)
  (h_history : g.history = 10)
  (h_math : g.math = 18)
  (h_computing : g.computing = 9)
  (h_historyAndMath : g.historyAndMath = 5)
  (h_historyAndComputing : g.historyAndComputing = 3)
  (h_mathAndComputing : g.mathAndComputing = 4)
  (h_allThree : g.allThree = 2) :
  studentsWithoutA g = 13 := by
  sorry

end NUMINAMATH_CALUDE_students_without_A_count_l2084_208425


namespace NUMINAMATH_CALUDE_jacks_stamp_collection_value_l2084_208475

/-- Given a collection of stamps where all stamps have equal value, 
    calculate the total value of the collection. -/
def stamp_collection_value (total_stamps : ℕ) (sample_stamps : ℕ) (sample_value : ℕ) : ℕ :=
  total_stamps * (sample_value / sample_stamps)

/-- Prove that Jack's stamp collection is worth 80 dollars -/
theorem jacks_stamp_collection_value :
  stamp_collection_value 20 4 16 = 80 := by
  sorry

end NUMINAMATH_CALUDE_jacks_stamp_collection_value_l2084_208475


namespace NUMINAMATH_CALUDE_sally_lemonade_sales_l2084_208435

/-- Calculates the total number of lemonade cups sold over two weeks -/
def total_lemonade_sales (last_week : ℕ) (increase_percentage : ℕ) : ℕ :=
  let this_week := last_week + last_week * increase_percentage / 100
  last_week + this_week

/-- Proves that given the conditions, Sally sold 46 cups of lemonade in total -/
theorem sally_lemonade_sales : total_lemonade_sales 20 30 = 46 := by
  sorry

end NUMINAMATH_CALUDE_sally_lemonade_sales_l2084_208435


namespace NUMINAMATH_CALUDE_investment_interest_rate_l2084_208466

/-- Proves that the interest rate for the first part of an investment is 3% given the specified conditions --/
theorem investment_interest_rate : 
  ∀ (total_amount first_part second_part first_rate second_rate total_interest : ℚ),
  total_amount = 4000 →
  first_part = 2800 →
  second_part = total_amount - first_part →
  second_rate = 5 →
  (first_part * first_rate / 100 + second_part * second_rate / 100) = total_interest →
  total_interest = 144 →
  first_rate = 3 := by
sorry

end NUMINAMATH_CALUDE_investment_interest_rate_l2084_208466


namespace NUMINAMATH_CALUDE_difficult_vs_easy_problems_l2084_208402

/-- Represents the number of problems solved by different combinations of students -/
structure ProblemDistribution where
  x₁ : ℕ  -- problems solved only by student 1
  x₂ : ℕ  -- problems solved only by student 2
  x₃ : ℕ  -- problems solved only by student 3
  y₁₂ : ℕ -- problems solved only by students 1 and 2
  y₁₃ : ℕ -- problems solved only by students 1 and 3
  y₂₃ : ℕ -- problems solved only by students 2 and 3
  z : ℕ   -- problems solved by all three students

/-- The main theorem stating the relationship between difficult and easy problems -/
theorem difficult_vs_easy_problems (d : ProblemDistribution) :
  d.x₁ + d.x₂ + d.x₃ + d.y₁₂ + d.y₁₃ + d.y₂₃ + d.z = 100 →
  d.x₁ + d.y₁₂ + d.y₁₃ + d.z = 60 →
  d.x₂ + d.y₁₂ + d.y₂₃ + d.z = 60 →
  d.x₃ + d.y₁₃ + d.y₂₃ + d.z = 60 →
  d.x₁ + d.x₂ + d.x₃ - d.z = 20 :=
by sorry

end NUMINAMATH_CALUDE_difficult_vs_easy_problems_l2084_208402


namespace NUMINAMATH_CALUDE_polynomial_identity_l2084_208458

theorem polynomial_identity (P : ℝ → ℝ) 
  (h1 : P 0 = 0) 
  (h2 : ∀ x : ℝ, P (x^2 + 1) = (P x)^2 + 1) : 
  ∀ x : ℝ, P x = x := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2084_208458


namespace NUMINAMATH_CALUDE_robin_cupcakes_sold_l2084_208479

/-- Represents the number of cupcakes Robin initially made -/
def initial_cupcakes : ℕ := 42

/-- Represents the number of additional cupcakes Robin made -/
def additional_cupcakes : ℕ := 39

/-- Represents the final number of cupcakes Robin had -/
def final_cupcakes : ℕ := 59

/-- Represents the number of cupcakes Robin sold -/
def sold_cupcakes : ℕ := 22

theorem robin_cupcakes_sold :
  initial_cupcakes - sold_cupcakes + additional_cupcakes = final_cupcakes :=
by sorry

end NUMINAMATH_CALUDE_robin_cupcakes_sold_l2084_208479


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l2084_208492

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x - 2*x + 1

theorem tangent_line_at_one (x y : ℝ) :
  let f' : ℝ → ℝ := λ t => 2*t * Real.log t + t - 2
  (x + y = 0) ↔ (y - f 1 = f' 1 * (x - 1)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l2084_208492


namespace NUMINAMATH_CALUDE_balls_distribution_proof_l2084_208420

def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

theorem balls_distribution_proof :
  distribute_balls 10 4 = 84 := by
  sorry

end NUMINAMATH_CALUDE_balls_distribution_proof_l2084_208420


namespace NUMINAMATH_CALUDE_point_on_y_axis_has_zero_x_coordinate_l2084_208413

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on the y-axis -/
def lies_on_y_axis (p : Point) : Prop := p.x = 0

/-- Theorem: If a point lies on the y-axis, its x-coordinate is zero -/
theorem point_on_y_axis_has_zero_x_coordinate (m n : ℝ) :
  lies_on_y_axis (Point.mk m n) → m = 0 := by
  sorry


end NUMINAMATH_CALUDE_point_on_y_axis_has_zero_x_coordinate_l2084_208413


namespace NUMINAMATH_CALUDE_jake_jill_difference_l2084_208437

/-- The number of peaches each person has -/
structure Peaches where
  jill : ℕ
  steven : ℕ
  jake : ℕ

/-- Given conditions about peach quantities -/
def peach_conditions (p : Peaches) : Prop :=
  p.jill = 87 ∧
  p.steven = p.jill + 18 ∧
  p.jake = p.steven - 5

/-- Theorem stating the difference between Jake's and Jill's peaches -/
theorem jake_jill_difference (p : Peaches) :
  peach_conditions p → p.jake - p.jill = 13 := by
  sorry

end NUMINAMATH_CALUDE_jake_jill_difference_l2084_208437


namespace NUMINAMATH_CALUDE_square_side_length_l2084_208473

theorem square_side_length (x : ℝ) (triangle_side : ℝ) (square_side : ℝ) : 
  triangle_side = 2 * x →
  4 * square_side = 3 * triangle_side →
  x = 4 →
  square_side = 6 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l2084_208473


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2084_208472

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I)^2 / z = 1 + Complex.I) : 
  Complex.im z = -1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2084_208472


namespace NUMINAMATH_CALUDE_arrangement_count_proof_l2084_208416

/-- The number of ways to arrange 2 female students and 4 male students in a row,
    such that female student A is to the left of female student B. -/
def arrangement_count : ℕ := 360

/-- The total number of students -/
def total_students : ℕ := 6

/-- The number of female students -/
def female_students : ℕ := 2

/-- The number of male students -/
def male_students : ℕ := 4

theorem arrangement_count_proof :
  arrangement_count = (Nat.factorial total_students) / 2 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_proof_l2084_208416


namespace NUMINAMATH_CALUDE_roots_sum_l2084_208465

theorem roots_sum (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x : ℝ, x^2 - 12*a*x - 13*b = 0 ↔ (x = c ∨ x = d)) →
  (∀ x : ℝ, x^2 - 12*c*x - 13*d = 0 ↔ (x = a ∨ x = b)) →
  a + b + c + d = 1716 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_l2084_208465


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2084_208467

-- Problem 1
theorem problem_1 : (1 * (-12)) - 5 + (-14) - (-39) = 8 := by sorry

-- Problem 2
theorem problem_2 : (1 : ℚ) / 3 + (-3 / 4) + (-1 / 3) + (-1 / 4) + 18 / 19 = -1 / 19 := by sorry

-- Problem 3
theorem problem_3 : (10 + 1 / 3) + (-11.5) + (-(10 + 1 / 3)) - 4.5 = -16 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2084_208467


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2084_208462

theorem inequality_equivalence (x : ℝ) :
  abs (2 * x - 1) + abs (x + 1) ≥ x + 2 ↔ x ≤ 0 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2084_208462


namespace NUMINAMATH_CALUDE_max_pages_for_25_dollars_l2084_208411

-- Define the cost per page in cents
def cost_per_page : ℕ := 3

-- Define the total amount available in cents
def total_amount : ℕ := 2500

-- Define the function to calculate the maximum number of pages
def max_pages (cost : ℕ) (total : ℕ) : ℕ := 
  (total / cost : ℕ)

-- Theorem statement
theorem max_pages_for_25_dollars : 
  max_pages cost_per_page total_amount = 833 := by
  sorry

end NUMINAMATH_CALUDE_max_pages_for_25_dollars_l2084_208411


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2084_208426

-- Problem 1
theorem problem_1 (x : ℝ) (h : x = -1) :
  (4 - x) * (2 * x + 1) + 3 * x * (x - 3) = 7 := by sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (hx : x = 1) (hy : y = 1/2) :
  ((x + 2*y)^2 - (3*x + y)*(3*x - y) - 5*y^2) / (-1/2 * x) = 12 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2084_208426


namespace NUMINAMATH_CALUDE_product_quality_comparison_l2084_208487

structure MachineData where
  first_class : ℕ
  second_class : ℕ
  total : ℕ

def machine_a : MachineData := ⟨150, 50, 200⟩
def machine_b : MachineData := ⟨120, 80, 200⟩

def total_products : ℕ := 400

def k_squared (n a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem product_quality_comparison :
  (machine_a.first_class : ℚ) / machine_a.total = 3/4 ∧
  (machine_b.first_class : ℚ) / machine_b.total = 3/5 ∧
  6.635 < k_squared total_products machine_a.first_class machine_a.second_class
    machine_b.first_class machine_b.second_class ∧
  k_squared total_products machine_a.first_class machine_a.second_class
    machine_b.first_class machine_b.second_class < 10.828 := by
  sorry

end NUMINAMATH_CALUDE_product_quality_comparison_l2084_208487


namespace NUMINAMATH_CALUDE_original_number_is_75_l2084_208428

theorem original_number_is_75 (x : ℝ) : ((x / 2.5) - 10.5) * 0.3 = 5.85 → x = 75 := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_75_l2084_208428


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a9_l2084_208469

/-- An arithmetic sequence {aₙ} where a₂ = -3 and a₃ = -5 has a₉ = -17 -/
theorem arithmetic_sequence_a9 (a : ℕ → ℤ) :
  (∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n) →  -- arithmetic sequence condition
  a 2 = -3 →
  a 3 = -5 →
  a 9 = -17 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a9_l2084_208469


namespace NUMINAMATH_CALUDE_problem_1_l2084_208483

theorem problem_1 : (-16) + 28 + (-128) - (-66) = -50 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2084_208483


namespace NUMINAMATH_CALUDE_is_systematic_sampling_l2084_208499

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Stratified
  | Systematic

/-- Represents the auditorium setup and sampling process -/
structure AuditoriumSampling where
  rows : Nat
  seatsPerRow : Nat
  selectedSeatNumber : Nat

/-- Determines the sampling method based on the auditorium setup and sampling process -/
def determineSamplingMethod (setup : AuditoriumSampling) : SamplingMethod := sorry

/-- Theorem stating that the given sampling process is systematic sampling -/
theorem is_systematic_sampling (setup : AuditoriumSampling) 
  (h1 : setup.rows = 40)
  (h2 : setup.seatsPerRow = 25)
  (h3 : setup.selectedSeatNumber = 18) :
  determineSamplingMethod setup = SamplingMethod.Systematic := by sorry

end NUMINAMATH_CALUDE_is_systematic_sampling_l2084_208499


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l2084_208421

/-- The orthocenter of a triangle is the point where all three altitudes intersect. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (6, 4, 2)
  let C : ℝ × ℝ × ℝ := (4, 5, 6)
  orthocenter A B C = (1/2, 8, 1/2) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l2084_208421


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2084_208414

theorem complex_equation_solution (z : ℂ) (h : (3 + z) * Complex.I = 1) : z = -3 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2084_208414


namespace NUMINAMATH_CALUDE_max_pies_without_ingredients_l2084_208424

/-- Given a set of pies with specific ingredient distributions, 
    calculate the maximum number of pies without any of the specified ingredients -/
theorem max_pies_without_ingredients 
  (total_pies : ℕ) 
  (chocolate_pies : ℕ) 
  (blueberry_pies : ℕ) 
  (vanilla_pies : ℕ) 
  (almond_pies : ℕ) 
  (h_total : total_pies = 60)
  (h_chocolate : chocolate_pies ≥ 20)
  (h_blueberry : blueberry_pies = 45)
  (h_vanilla : vanilla_pies ≥ 24)
  (h_almond : almond_pies ≥ 6) :
  total_pies - blueberry_pies ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_max_pies_without_ingredients_l2084_208424


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l2084_208433

/-- For a rectangle with length to width ratio of 5:2 and diagonal d, 
    the area A can be expressed as A = (10/29)d^2 -/
theorem rectangle_area_diagonal (l w d : ℝ) (h_ratio : l / w = 5 / 2) 
    (h_diagonal : l^2 + w^2 = d^2) : l * w = (10/29) * d^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l2084_208433


namespace NUMINAMATH_CALUDE_percentage_calculation_l2084_208440

theorem percentage_calculation (n : ℝ) (h : n = 5200) : 0.15 * (0.30 * (0.50 * n)) = 117 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2084_208440


namespace NUMINAMATH_CALUDE_base7_523_equals_base10_262_l2084_208470

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (a b c : ℕ) : ℕ :=
  a * 7^2 + b * 7^1 + c * 7^0

/-- The theorem stating that 523 in base-7 is equal to 262 in base-10 --/
theorem base7_523_equals_base10_262 : base7ToBase10 5 2 3 = 262 := by
  sorry

end NUMINAMATH_CALUDE_base7_523_equals_base10_262_l2084_208470


namespace NUMINAMATH_CALUDE_triangle_area_l2084_208443

theorem triangle_area (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = Real.sqrt 19) :
  (1/2 : ℝ) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2) = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2084_208443


namespace NUMINAMATH_CALUDE_increasing_function_inequality_range_l2084_208471

/-- Given an increasing function f defined on [0,+∞), 
    prove that the range of x satisfying f(2x-1) < f(1/3) is [1/2, 2/3). -/
theorem increasing_function_inequality_range 
  (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_domain : ∀ x, x ∈ Set.Ici (0 : ℝ) → f x ∈ Set.univ) :
  {x : ℝ | f (2*x - 1) < f (1/3)} = Set.Icc (1/2 : ℝ) (2/3) := by
sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_range_l2084_208471


namespace NUMINAMATH_CALUDE_distance_between_buildings_eight_trees_nine_meters_l2084_208430

/-- Given two buildings with trees planted between them, calculate the distance between the buildings. -/
theorem distance_between_buildings (num_trees : ℕ) (tree_spacing : ℕ) : ℕ :=
  (num_trees + 1) * tree_spacing

/-- Prove that with 8 trees planted 1 meter apart, the distance between buildings is 9 meters. -/
theorem eight_trees_nine_meters :
  distance_between_buildings 8 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_buildings_eight_trees_nine_meters_l2084_208430


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2084_208436

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular decagon has 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2084_208436


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l2084_208453

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 156) (h2 : a*b + b*c + a*c = 50) :
  a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l2084_208453


namespace NUMINAMATH_CALUDE_total_clamps_is_92_l2084_208441

/-- Represents the number of bike clamps given per bicycle purchase -/
def clamps_per_bike : ℕ := 2

/-- Represents the number of bikes sold in the morning -/
def morning_sales : ℕ := 19

/-- Represents the number of bikes sold in the afternoon -/
def afternoon_sales : ℕ := 27

/-- Calculates the total number of bike clamps given away -/
def total_clamps : ℕ := clamps_per_bike * (morning_sales + afternoon_sales)

/-- Proves that the total number of bike clamps given away is 92 -/
theorem total_clamps_is_92 : total_clamps = 92 := by
  sorry

end NUMINAMATH_CALUDE_total_clamps_is_92_l2084_208441


namespace NUMINAMATH_CALUDE_scallop_cost_theorem_l2084_208447

def scallop_cost (people : ℕ) (scallops_per_person : ℕ) (scallops_per_pound : ℕ) 
  (price_per_pound : ℚ) (discount_rate : ℚ) (tax_rate : ℚ) : ℚ :=
  let total_scallops := people * scallops_per_person
  let pounds_needed := total_scallops / scallops_per_pound
  let initial_cost := pounds_needed * price_per_pound
  let discounted_cost := initial_cost * (1 - discount_rate)
  let final_cost := discounted_cost * (1 + tax_rate)
  final_cost

theorem scallop_cost_theorem :
  let result := scallop_cost 8 2 8 24 (1/10) (7/100)
  ⌊result * 100⌋ / 100 = 4622 / 100 := by sorry

end NUMINAMATH_CALUDE_scallop_cost_theorem_l2084_208447


namespace NUMINAMATH_CALUDE_select_specific_boy_and_girl_probability_l2084_208451

/-- The probability of selecting both boy A and girl B when randomly choosing 1 boy and 2 girls from a group of 8 boys and 3 girls -/
theorem select_specific_boy_and_girl_probability :
  let total_boys : ℕ := 8
  let total_girls : ℕ := 3
  let boys_to_select : ℕ := 1
  let girls_to_select : ℕ := 2
  let total_events : ℕ := (total_boys.choose boys_to_select) * (total_girls.choose girls_to_select)
  let favorable_events : ℕ := 2  -- Only 2 ways to select the other girl
  (favorable_events : ℚ) / total_events = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_select_specific_boy_and_girl_probability_l2084_208451


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2084_208484

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_discriminant :
  discriminant 5 (-9) (-7) = 221 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2084_208484
