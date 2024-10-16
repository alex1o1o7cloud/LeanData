import Mathlib

namespace NUMINAMATH_CALUDE_intersection_circles_angle_value_l2066_206623

-- Define the problem setup
def circle1_angle1 (x : ℝ) := 7 * x
def circle1_angle2 (x : ℝ) := 3 * x
def circle2_angle (x : ℝ) := 8 * x

-- State the theorem
theorem intersection_circles_angle_value :
  ∀ x : ℝ,
  (circle1_angle1 x + circle1_angle2 x = 360) →
  (circle2_angle x ≤ 360) →
  x = 36 := by
  sorry


end NUMINAMATH_CALUDE_intersection_circles_angle_value_l2066_206623


namespace NUMINAMATH_CALUDE_triangle_max_side_length_l2066_206647

/-- A triangle with three different integer side lengths and a perimeter of 24 units has a maximum side length of 11 units. -/
theorem triangle_max_side_length :
  ∀ a b c : ℕ,
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a + b + c = 24 →
  a ≤ 11 ∧ b ≤ 11 ∧ c ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_side_length_l2066_206647


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2066_206688

theorem perfect_square_trinomial (x : ℝ) : ∃ (a : ℝ), (x^2 + 4 + 4*x) = (x + a)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2066_206688


namespace NUMINAMATH_CALUDE_shaded_region_angle_l2066_206692

/-- Given two concentric circles with radii 1 and 2, if the area of the shaded region
    between them is three times smaller than the area of the larger circle,
    then the angle subtending this shaded region at the center is 8π/9 radians. -/
theorem shaded_region_angle (r₁ r₂ : ℝ) (A_shaded A_large : ℝ) (θ : ℝ) :
  r₁ = 1 →
  r₂ = 2 →
  A_large = π * r₂^2 →
  A_shaded = (1/3) * A_large →
  A_shaded = (θ / (2 * π)) * (π * r₂^2 - π * r₁^2) →
  θ = (8 * π) / 9 := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_angle_l2066_206692


namespace NUMINAMATH_CALUDE_larger_number_proof_l2066_206668

theorem larger_number_proof (x y : ℕ) 
  (h1 : y - x = 1365)
  (h2 : y = 4 * x + 15) : 
  y = 1815 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2066_206668


namespace NUMINAMATH_CALUDE_common_root_condition_l2066_206618

theorem common_root_condition (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + x + a = 0) ↔ (a = 1 ∨ a = -2) := by
  sorry

end NUMINAMATH_CALUDE_common_root_condition_l2066_206618


namespace NUMINAMATH_CALUDE_kitchen_width_l2066_206666

/-- Calculates the width of a rectangular kitchen given its dimensions and painting information. -/
theorem kitchen_width (length height : ℝ) (total_painted_area : ℝ) : 
  length = 12 ∧ 
  height = 10 ∧ 
  total_painted_area = 1680 → 
  (total_painted_area / 3) = 2 * (length * height + height * (total_painted_area / (3 * height) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_kitchen_width_l2066_206666


namespace NUMINAMATH_CALUDE_barbaras_savings_l2066_206611

/-- Calculates the current savings given the total cost, weekly allowance, and remaining weeks to save. -/
def currentSavings (totalCost : ℕ) (weeklyAllowance : ℕ) (remainingWeeks : ℕ) : ℕ :=
  totalCost - (weeklyAllowance * remainingWeeks)

/-- Proves that given the specific conditions, Barbara's current savings is $20. -/
theorem barbaras_savings :
  let watchCost : ℕ := 100
  let weeklyAllowance : ℕ := 5
  let remainingWeeks : ℕ := 16
  currentSavings watchCost weeklyAllowance remainingWeeks = 20 := by
  sorry

end NUMINAMATH_CALUDE_barbaras_savings_l2066_206611


namespace NUMINAMATH_CALUDE_product_of_roots_l2066_206684

theorem product_of_roots (x : ℝ) : (x - 1) * (x + 4) = 22 → ∃ y : ℝ, (y - 1) * (y + 4) = 22 ∧ x * y = -26 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2066_206684


namespace NUMINAMATH_CALUDE_officer_selection_count_l2066_206698

/-- The number of members in the club -/
def club_size : ℕ := 12

/-- The number of officer positions to be filled -/
def officer_positions : ℕ := 5

/-- The number of ways to select distinct officers from the club members -/
def officer_selection_ways : ℕ := club_size * (club_size - 1) * (club_size - 2) * (club_size - 3) * (club_size - 4)

/-- Theorem stating that the number of ways to select officers is 95,040 -/
theorem officer_selection_count :
  officer_selection_ways = 95040 :=
by sorry

end NUMINAMATH_CALUDE_officer_selection_count_l2066_206698


namespace NUMINAMATH_CALUDE_decagon_area_theorem_l2066_206676

/-- A rectangle with an inscribed decagon -/
structure DecagonInRectangle where
  perimeter : ℝ
  length_width_ratio : ℝ
  inscribed_decagon : Unit

/-- Calculate the area of the inscribed decagon -/
def area_of_inscribed_decagon (r : DecagonInRectangle) : ℝ :=
  sorry

/-- The theorem statement -/
theorem decagon_area_theorem (r : DecagonInRectangle) 
  (h_perimeter : r.perimeter = 160)
  (h_ratio : r.length_width_ratio = 3 / 2) :
  area_of_inscribed_decagon r = 1413.12 := by
  sorry

end NUMINAMATH_CALUDE_decagon_area_theorem_l2066_206676


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l2066_206646

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: The number of ways to distribute 6 distinguishable balls into 3 distinguishable boxes is 729 -/
theorem distribute_six_balls_three_boxes :
  distribute_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l2066_206646


namespace NUMINAMATH_CALUDE_tan_45_degrees_l2066_206602

theorem tan_45_degrees (Q : ℝ × ℝ) : 
  (Q.1 = 1 / Real.sqrt 2) → 
  (Q.2 = 1 / Real.sqrt 2) → 
  (Q.1^2 + Q.2^2 = 1) →
  Real.tan (π/4) = 1 := by
  sorry


end NUMINAMATH_CALUDE_tan_45_degrees_l2066_206602


namespace NUMINAMATH_CALUDE_cos105_cos45_plus_sin45_sin105_eq_half_l2066_206635

theorem cos105_cos45_plus_sin45_sin105_eq_half :
  Real.cos (105 * π / 180) * Real.cos (45 * π / 180) +
  Real.sin (45 * π / 180) * Real.sin (105 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos105_cos45_plus_sin45_sin105_eq_half_l2066_206635


namespace NUMINAMATH_CALUDE_expression_equality_l2066_206657

theorem expression_equality : 2 * (2^7 + 2^7 + 2^8)^(1/4) = 8 * 2^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2066_206657


namespace NUMINAMATH_CALUDE_waiter_tip_earnings_l2066_206694

theorem waiter_tip_earnings (total_customers : ℕ) (non_tipping_customers : ℕ) (tip_amount : ℕ) (total_earnings : ℕ) : 
  total_customers = 10 →
  non_tipping_customers = 5 →
  tip_amount = 3 →
  total_earnings = (total_customers - non_tipping_customers) * tip_amount →
  total_earnings = 15 :=
by sorry

end NUMINAMATH_CALUDE_waiter_tip_earnings_l2066_206694


namespace NUMINAMATH_CALUDE_jerky_order_fulfillment_l2066_206628

def days_to_fulfill_order (ordered_bags : ℕ) (existing_bags : ℕ) (bags_per_batch : ℕ) : ℕ :=
  ((ordered_bags - existing_bags) + bags_per_batch - 1) / bags_per_batch

theorem jerky_order_fulfillment :
  days_to_fulfill_order 60 20 10 = 4 :=
by sorry

end NUMINAMATH_CALUDE_jerky_order_fulfillment_l2066_206628


namespace NUMINAMATH_CALUDE_class_size_l2066_206620

theorem class_size (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : french = 41)
  (h2 : german = 22)
  (h3 : both = 9)
  (h4 : neither = 24) :
  french + german - both + neither = 78 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l2066_206620


namespace NUMINAMATH_CALUDE_right_triangle_complex_count_l2066_206663

/-- A complex number z satisfies the right triangle property if 0, z, and z^2 form a right triangle
    with the right angle at z. -/
def has_right_triangle_property (z : ℂ) : Prop :=
  z ≠ 0 ∧ 
  (0 : ℂ) ≠ z ∧ 
  z ≠ z^2 ∧
  (z - 0) * (z^2 - z) = 0

/-- There are exactly two complex numbers that satisfy the right triangle property. -/
theorem right_triangle_complex_count : 
  ∃! (s : Finset ℂ), (∀ z ∈ s, has_right_triangle_property z) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_right_triangle_complex_count_l2066_206663


namespace NUMINAMATH_CALUDE_spade_sum_equals_negative_sixteen_l2066_206681

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_sum_equals_negative_sixteen :
  (spade 2 3) + (spade 5 6) = -16 := by sorry

end NUMINAMATH_CALUDE_spade_sum_equals_negative_sixteen_l2066_206681


namespace NUMINAMATH_CALUDE_bus_trip_speed_l2066_206693

/-- Proves that for a trip of 420 miles, if increasing the average speed by 10 mph
    reduces the travel time by 2 hours, then the original average speed was 42 mph. -/
theorem bus_trip_speed (v : ℝ) (h : v > 0) :
  (420 / v) - (420 / (v + 10)) = 2 → v = 42 := by
sorry

end NUMINAMATH_CALUDE_bus_trip_speed_l2066_206693


namespace NUMINAMATH_CALUDE_gcf_of_72_90_120_l2066_206678

theorem gcf_of_72_90_120 : Nat.gcd 72 (Nat.gcd 90 120) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_72_90_120_l2066_206678


namespace NUMINAMATH_CALUDE_train_passing_time_l2066_206669

/-- Proves that a train with given length and speed takes the calculated time to pass a fixed point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 300 ∧ train_speed_kmh = 90 → 
  (train_length / (train_speed_kmh * (5/18))) = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l2066_206669


namespace NUMINAMATH_CALUDE_max_time_sum_of_digits_is_19_l2066_206654

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours ≤ 23
  minute_valid : minutes ≤ 59

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def timeSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits for any Time24 -/
def maxTimeSumOfDigits : Nat :=
  19

theorem max_time_sum_of_digits_is_19 :
  ∀ t : Time24, timeSumOfDigits t ≤ maxTimeSumOfDigits :=
by sorry

end NUMINAMATH_CALUDE_max_time_sum_of_digits_is_19_l2066_206654


namespace NUMINAMATH_CALUDE_fraction_product_cubes_evaluate_fraction_product_l2066_206610

theorem fraction_product_cubes (a b c d : ℚ) :
  (a / b) ^ 3 * (c / d) ^ 3 = (a * c / (b * d)) ^ 3 :=
by sorry

theorem evaluate_fraction_product :
  (8 / 9 : ℚ) ^ 3 * (3 / 4 : ℚ) ^ 3 = 8 / 27 :=
by sorry

end NUMINAMATH_CALUDE_fraction_product_cubes_evaluate_fraction_product_l2066_206610


namespace NUMINAMATH_CALUDE_common_chord_equation_l2066_206621

/-- The equation of the common chord of two circles -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 + 4*x - 6*y + 4 = 0) ∧ 
  (x^2 + y^2 + 2*x - 4*y - 3 = 0) →
  (2*x - 2*y + 7 = 0) :=
sorry

end NUMINAMATH_CALUDE_common_chord_equation_l2066_206621


namespace NUMINAMATH_CALUDE_expression_evaluation_l2066_206630

theorem expression_evaluation (a b : ℝ) (h1 : a = -1) (h2 : b = -4) :
  ((a - 2*b)^2 + (a - 2*b)*(a + 2*b) + 2*a*(2*a - b)) / (2*a) = 9 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2066_206630


namespace NUMINAMATH_CALUDE_largest_integer_divisible_by_all_less_than_cube_root_l2066_206600

theorem largest_integer_divisible_by_all_less_than_cube_root : ∃ (N : ℕ), 
  (N = 420) ∧ 
  (∀ (k : ℕ), k > 0 ∧ k ≤ ⌊(N : ℝ)^(1/3)⌋ → N % k = 0) ∧
  (∀ (M : ℕ), M > N → ∃ (m : ℕ), m > 0 ∧ m ≤ ⌊(M : ℝ)^(1/3)⌋ ∧ M % m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_divisible_by_all_less_than_cube_root_l2066_206600


namespace NUMINAMATH_CALUDE_first_division_meiosis_characteristics_l2066_206643

/-- Represents the behavior of chromosomes during cell division -/
inductive ChromosomeBehavior
  | separate
  | notSeparate

/-- Represents the behavior of centromeres during cell division -/
inductive CentromereBehavior
  | split
  | notSplit

/-- Represents the characteristics of a cell division -/
structure CellDivisionCharacteristics where
  chromosomeBehavior : ChromosomeBehavior
  centromereBehavior : CentromereBehavior

/-- Represents the first division of meiosis -/
def firstDivisionMeiosis : CellDivisionCharacteristics := sorry

/-- Theorem stating the characteristics of the first division of meiosis -/
theorem first_division_meiosis_characteristics :
  firstDivisionMeiosis.chromosomeBehavior = ChromosomeBehavior.separate ∧
  firstDivisionMeiosis.centromereBehavior = CentromereBehavior.notSplit :=
sorry

end NUMINAMATH_CALUDE_first_division_meiosis_characteristics_l2066_206643


namespace NUMINAMATH_CALUDE_prob_three_green_out_of_six_l2066_206682

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 7

/-- The number of purple marbles in the bag -/
def purple_marbles : ℕ := 3

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := green_marbles + purple_marbles

/-- The number of draws -/
def num_draws : ℕ := 6

/-- The number of green marbles we want to draw -/
def target_green : ℕ := 3

/-- The probability of drawing a green marble in a single draw -/
def prob_green : ℚ := green_marbles / total_marbles

/-- The probability of drawing a purple marble in a single draw -/
def prob_purple : ℚ := purple_marbles / total_marbles

/-- The probability of drawing exactly 3 green marbles out of 6 draws -/
theorem prob_three_green_out_of_six :
  (Nat.choose num_draws target_green : ℚ) * prob_green ^ target_green * prob_purple ^ (num_draws - target_green) =
  185220 / 1000000 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_green_out_of_six_l2066_206682


namespace NUMINAMATH_CALUDE_hypotenuse_angle_is_45_degrees_l2066_206686

/-- A right triangle with perimeter 2 units -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  perimeter_eq_two : a + b + c = 2
  pythagorean : a^2 + b^2 = c^2

/-- Point on the internal angle bisector of the right angle -/
structure BisectorPoint (t : RightTriangle) where
  distance_sqrt_two : ℝ
  is_sqrt_two : distance_sqrt_two = Real.sqrt 2

/-- The angle subtended by the hypotenuse from the bisector point -/
def hypotenuse_angle (t : RightTriangle) (p : BisectorPoint t) : ℝ := sorry

theorem hypotenuse_angle_is_45_degrees (t : RightTriangle) (p : BisectorPoint t) :
  hypotenuse_angle t p = 45 * π / 180 := by sorry

end NUMINAMATH_CALUDE_hypotenuse_angle_is_45_degrees_l2066_206686


namespace NUMINAMATH_CALUDE_handshake_theorem_l2066_206612

theorem handshake_theorem (n : ℕ) (total_handshakes : ℕ) :
  n = 10 →
  total_handshakes = 45 →
  total_handshakes = n * (n - 1) / 2 →
  ∀ boy : Fin n, (n - 1 : ℕ) = total_handshakes / n :=
by sorry

end NUMINAMATH_CALUDE_handshake_theorem_l2066_206612


namespace NUMINAMATH_CALUDE_impossibleToReachOpposite_l2066_206619

/-- Represents the color of a point -/
inductive Color
| White
| Black

/-- Represents a point on the circle -/
structure Point where
  position : Fin 2022
  color : Color

/-- The type of operation that can be performed -/
inductive Operation
| FlipAdjacent (i : Fin 2022)
| FlipWithGap (i : Fin 2022)

/-- The configuration of all points on the circle -/
def Configuration := Fin 2022 → Color

/-- Apply an operation to a configuration -/
def applyOperation (config : Configuration) (op : Operation) : Configuration :=
  sorry

/-- The initial configuration with one black point and others white -/
def initialConfig : Configuration :=
  sorry

/-- Check if a configuration is the opposite of the initial configuration -/
def isOppositeConfig (config : Configuration) : Prop :=
  sorry

/-- The main theorem stating that it's impossible to reach the opposite configuration -/
theorem impossibleToReachOpposite : 
  ∀ (ops : List Operation), 
    ¬(isOppositeConfig (ops.foldl applyOperation initialConfig)) :=
  sorry

end NUMINAMATH_CALUDE_impossibleToReachOpposite_l2066_206619


namespace NUMINAMATH_CALUDE_midpoint_locus_l2066_206631

-- Define the line l
def line_l (x y : ℝ) : Prop := x / 12 + y / 8 = 1

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 24 + y^2 / 16 = 1

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define tangent points A and B on ellipse C
def tangent_point (x y : ℝ) : Prop := ellipse_C x y

-- Define the midpoint M of AB
def midpoint_M (x y x1 y1 x2 y2 : ℝ) : Prop :=
  x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2

-- Theorem statement
theorem midpoint_locus
  (P_x P_y A_x A_y B_x B_y M_x M_y : ℝ)
  (h_P : point_P P_x P_y)
  (h_A : tangent_point A_x A_y)
  (h_B : tangent_point B_x B_y)
  (h_M : midpoint_M M_x M_y A_x A_y B_x B_y) :
  (M_x - 1)^2 / (5/2) + (M_y - 1)^2 / (5/3) = 1 :=
sorry

end NUMINAMATH_CALUDE_midpoint_locus_l2066_206631


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l2066_206644

theorem smallest_common_multiple_of_6_and_15 :
  ∃ b : ℕ+, (∀ n : ℕ+, 6 ∣ n ∧ 15 ∣ n → b ≤ n) ∧ 6 ∣ b ∧ 15 ∣ b ∧ b = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l2066_206644


namespace NUMINAMATH_CALUDE_certain_number_value_l2066_206659

/-- Custom operation # -/
def hash (a b : ℝ) : ℝ := a * b - b + b^2

/-- Theorem stating that if 3 # x = 48, then x = 6 -/
theorem certain_number_value (x : ℝ) : hash 3 x = 48 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l2066_206659


namespace NUMINAMATH_CALUDE_point_C_coordinates_l2066_206696

def A : ℝ × ℝ := (-2, 1)
def B : ℝ × ℝ := (4, 9)

theorem point_C_coordinates :
  ∀ C : ℝ × ℝ,
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C = (1 - t) • A + t • B) →  -- C lies on segment AB
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 16 * ((B.1 - C.1)^2 + (B.2 - C.2)^2) →  -- AC = 4CB
  C = (8/5, 14/5) :=
by sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l2066_206696


namespace NUMINAMATH_CALUDE_min_wednesday_birthdays_l2066_206606

/-- Given a company with 61 employees, where the number of employees with birthdays on Wednesday
    is greater than the number on any other day, and all other days have an equal number of birthdays,
    the minimum number of employees with birthdays on Wednesday is 13. -/
theorem min_wednesday_birthdays (total_employees : ℕ) (wednesday_birthdays : ℕ) 
  (other_day_birthdays : ℕ) : 
  total_employees = 61 →
  wednesday_birthdays > other_day_birthdays →
  total_employees = wednesday_birthdays + 6 * other_day_birthdays →
  wednesday_birthdays ≥ 13 :=
by sorry

end NUMINAMATH_CALUDE_min_wednesday_birthdays_l2066_206606


namespace NUMINAMATH_CALUDE_pencil_count_is_830_l2066_206699

/-- The final number of pencils in the drawer after a series of additions and removals. -/
def final_pencil_count (initial : ℕ) (nancy_adds : ℕ) (steven_adds : ℕ) (maria_adds : ℕ) (kim_removes : ℕ) (george_removes : ℕ) : ℕ :=
  initial + nancy_adds + steven_adds + maria_adds - kim_removes - george_removes

/-- Theorem stating that the final number of pencils in the drawer is 830. -/
theorem pencil_count_is_830 :
  final_pencil_count 200 375 150 250 85 60 = 830 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_is_830_l2066_206699


namespace NUMINAMATH_CALUDE_jakes_comic_books_l2066_206655

theorem jakes_comic_books (jake_books : ℕ) (brother_books : ℕ) : 
  brother_books = jake_books + 15 →
  jake_books + brother_books = 87 →
  jake_books = 36 := by
sorry

end NUMINAMATH_CALUDE_jakes_comic_books_l2066_206655


namespace NUMINAMATH_CALUDE_monkey_climbing_theorem_l2066_206671

/-- Represents the climbing problem of a monkey on a tree -/
structure ClimbingProblem where
  treeHeight : ℕ
  climbRate : ℕ
  slipRate : ℕ
  restPeriod : ℕ
  restDuration : ℕ

/-- Calculates the time taken for the monkey to reach the top of the tree -/
def timeTakenToClimb (problem : ClimbingProblem) : ℕ :=
  sorry

/-- The theorem stating the solution to the specific climbing problem -/
theorem monkey_climbing_theorem :
  let problem : ClimbingProblem := {
    treeHeight := 253,
    climbRate := 7,
    slipRate := 4,
    restPeriod := 4,
    restDuration := 1
  }
  timeTakenToClimb problem = 109 := by
  sorry

end NUMINAMATH_CALUDE_monkey_climbing_theorem_l2066_206671


namespace NUMINAMATH_CALUDE_odd_difference_of_even_and_odd_l2066_206656

theorem odd_difference_of_even_and_odd (a b : ℤ) 
  (ha : Even a) (hb : Odd b) : Odd (a - b) := by
  sorry

end NUMINAMATH_CALUDE_odd_difference_of_even_and_odd_l2066_206656


namespace NUMINAMATH_CALUDE_chord_length_no_intersection_tangent_two_intersections_one_intersection_l2066_206697

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 12*x

-- Define the line y = 2x - 6
def line1 (x y : ℝ) : Prop := y = 2*x - 6

-- Define the line y = kx + 1
def line2 (k x y : ℝ) : Prop := y = k*x + 1

-- Theorem for the chord length
theorem chord_length : ∃ (x1 y1 x2 y2 : ℝ),
  parabola x1 y1 ∧ parabola x2 y2 ∧ 
  line1 x1 y1 ∧ line1 x2 y2 ∧
  ((x2 - x1)^2 + (y2 - y1)^2)^(1/2 : ℝ) = 15 := by sorry

-- Theorems for the positional relationships
theorem no_intersection (k : ℝ) : 
  k > 3 → ¬∃ (x y : ℝ), parabola x y ∧ line2 k x y := by sorry

theorem tangent : 
  ∃! (x y : ℝ), parabola x y ∧ line2 3 x y := by sorry

theorem two_intersections (k : ℝ) : 
  k < 3 ∧ k ≠ 0 → ∃ (x1 y1 x2 y2 : ℝ), 
    x1 ≠ x2 ∧ parabola x1 y1 ∧ parabola x2 y2 ∧ 
    line2 k x1 y1 ∧ line2 k x2 y2 := by sorry

theorem one_intersection : 
  ∃! (x y : ℝ), parabola x y ∧ line2 0 x y := by sorry

end NUMINAMATH_CALUDE_chord_length_no_intersection_tangent_two_intersections_one_intersection_l2066_206697


namespace NUMINAMATH_CALUDE_exists_cutting_method_for_person_to_fit_l2066_206625

/-- Represents a sheet of paper -/
structure Sheet :=
  (length : ℝ)
  (width : ℝ)
  (thickness : ℝ)

/-- Represents a person -/
structure Person :=
  (height : ℝ)
  (width : ℝ)

/-- Represents a cutting method -/
structure CuttingMethod :=
  (cuts : List (ℝ × ℝ))  -- List of cut coordinates

/-- Represents the result of applying a cutting method to a sheet -/
def apply_cutting_method (s : Sheet) (cm : CuttingMethod) : ℝ := sorry

/-- Determines if a person can fit through an opening -/
def can_fit_through (p : Person) (opening_size : ℝ) : Prop := sorry

/-- Main theorem: There exists a cutting method that creates an opening large enough for a person -/
theorem exists_cutting_method_for_person_to_fit (s : Sheet) (p : Person) : 
  ∃ (cm : CuttingMethod), can_fit_through p (apply_cutting_method s cm) :=
sorry

end NUMINAMATH_CALUDE_exists_cutting_method_for_person_to_fit_l2066_206625


namespace NUMINAMATH_CALUDE_single_digit_number_trick_l2066_206687

theorem single_digit_number_trick (x : ℕ) (h : x < 10) : 
  (((2 * x + 3) * 5 + 7) % 10 + 18) / 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_single_digit_number_trick_l2066_206687


namespace NUMINAMATH_CALUDE_decagon_perimeter_l2066_206641

/-- The perimeter of a regular polygon with n sides of length s -/
def perimeter (n : ℕ) (s : ℝ) : ℝ := n * s

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- The side length of our specific decagon -/
def side_length : ℝ := 3

theorem decagon_perimeter : perimeter decagon_sides side_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_decagon_perimeter_l2066_206641


namespace NUMINAMATH_CALUDE_jogging_average_l2066_206667

theorem jogging_average (days_short : ℕ) (days_long : ℕ) (minutes_short : ℕ) (minutes_long : ℕ) 
  (target_average : ℕ) (total_days : ℕ) :
  days_short = 6 →
  days_long = 4 →
  minutes_short = 80 →
  minutes_long = 105 →
  target_average = 100 →
  total_days = 11 →
  (days_short * minutes_short + days_long * minutes_long + 
   (target_average * total_days - (days_short * minutes_short + days_long * minutes_long))) / total_days = target_average :=
by sorry

end NUMINAMATH_CALUDE_jogging_average_l2066_206667


namespace NUMINAMATH_CALUDE_leap_year_53_sundays_5_feb_sundays_probability_l2066_206651

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a leap year -/
structure LeapYear where
  days : Fin 366
  sundays : Nat
  februarySundays : Nat

/-- The probability of a specific configuration of extra days in a leap year -/
def extraDaysProbability : ℚ := 1 / 7

/-- The probability of a leap year having 53 Sundays -/
def prob53Sundays : ℚ := 2 / 7

/-- The probability of February in a leap year having 5 Sundays -/
def probFeb5Sundays : ℚ := 1 / 7

/-- 
Theorem: The probability of a randomly selected leap year having 53 Sundays, 
with exactly 5 of those Sundays falling in February, is 2/49.
-/
theorem leap_year_53_sundays_5_feb_sundays_probability : 
  prob53Sundays * probFeb5Sundays = 2 / 49 := by
  sorry

end NUMINAMATH_CALUDE_leap_year_53_sundays_5_feb_sundays_probability_l2066_206651


namespace NUMINAMATH_CALUDE_jason_omelet_eggs_l2066_206691

/-- The number of eggs Jason consumes in two weeks -/
def total_eggs : ℕ := 42

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- The number of eggs Jason uses for his omelet each morning -/
def eggs_per_day : ℚ := total_eggs / days_in_two_weeks

theorem jason_omelet_eggs : eggs_per_day = 3 := by
  sorry

end NUMINAMATH_CALUDE_jason_omelet_eggs_l2066_206691


namespace NUMINAMATH_CALUDE_second_meeting_time_is_12_minutes_l2066_206601

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  initialPosition : ℝ

/-- Represents the race scenario -/
structure RaceScenario where
  trackLength : ℝ
  firstMeetingDistance : ℝ
  firstMeetingTime : ℝ
  marie : Runner
  john : Runner

/-- Calculates the time of the second meeting given a race scenario -/
def secondMeetingTime (scenario : RaceScenario) : ℝ :=
  sorry

/-- Theorem stating that the second meeting occurs 12 minutes after the start -/
theorem second_meeting_time_is_12_minutes (scenario : RaceScenario) 
  (h1 : scenario.trackLength = 500)
  (h2 : scenario.firstMeetingDistance = 100)
  (h3 : scenario.firstMeetingTime = 2)
  (h4 : scenario.marie.initialPosition = 0)
  (h5 : scenario.john.initialPosition = 500)
  (h6 : scenario.marie.speed = scenario.firstMeetingDistance / scenario.firstMeetingTime)
  (h7 : scenario.john.speed = (scenario.trackLength - scenario.firstMeetingDistance) / scenario.firstMeetingTime) :
  secondMeetingTime scenario = 12 :=
sorry

end NUMINAMATH_CALUDE_second_meeting_time_is_12_minutes_l2066_206601


namespace NUMINAMATH_CALUDE_inequality_proof_l2066_206609

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2066_206609


namespace NUMINAMATH_CALUDE_fraction_equality_l2066_206677

theorem fraction_equality (m n : ℝ) (h : n ≠ 0) (h1 : m / n = 2 / 3) :
  m / (m + n) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2066_206677


namespace NUMINAMATH_CALUDE_fraction_simplification_l2066_206616

theorem fraction_simplification (x : ℝ) : (2*x - 3) / 4 + (3*x + 4) / 3 = (18*x + 7) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2066_206616


namespace NUMINAMATH_CALUDE_single_elimination_games_l2066_206661

/-- The number of games required to determine a champion in a single-elimination tournament -/
def gamesRequired (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with n players, 
    the number of games required to determine a champion is n - 1 -/
theorem single_elimination_games (n : ℕ) (h : n > 0) : 
  gamesRequired n = n - 1 := by sorry

end NUMINAMATH_CALUDE_single_elimination_games_l2066_206661


namespace NUMINAMATH_CALUDE_range_of_sqrt_function_l2066_206665

theorem range_of_sqrt_function :
  ∀ x : ℝ, (∃ y : ℝ, y = Real.sqrt (x + 2)) ↔ x ≥ -2 := by sorry

end NUMINAMATH_CALUDE_range_of_sqrt_function_l2066_206665


namespace NUMINAMATH_CALUDE_johns_total_distance_l2066_206675

/-- Calculates the total distance driven given the speed and time for each segment of a trip. -/
def total_distance (speed1 speed2 speed3 speed4 : ℝ) (time1 time2 time3 time4 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2 + speed3 * time3 + speed4 * time4

/-- Theorem stating that John's total distance driven is 470 miles. -/
theorem johns_total_distance :
  total_distance 45 55 60 50 2 3 1.5 2.5 = 470 := by
  sorry

#eval total_distance 45 55 60 50 2 3 1.5 2.5

end NUMINAMATH_CALUDE_johns_total_distance_l2066_206675


namespace NUMINAMATH_CALUDE_youngest_sibling_age_l2066_206629

theorem youngest_sibling_age (youngest_age : ℕ) : 
  (youngest_age + (youngest_age + 4) + (youngest_age + 5) + (youngest_age + 7)) / 4 = 21 →
  youngest_age = 17 := by
sorry

end NUMINAMATH_CALUDE_youngest_sibling_age_l2066_206629


namespace NUMINAMATH_CALUDE_shortest_tree_height_l2066_206642

/-- Proves that the height of the shortest tree is 50 feet given the conditions of the problem. -/
theorem shortest_tree_height (tallest middle shortest : ℝ) : 
  tallest = 150 ∧ 
  middle = 2/3 * tallest ∧ 
  shortest = 1/2 * middle →
  shortest = 50 := by
  sorry

end NUMINAMATH_CALUDE_shortest_tree_height_l2066_206642


namespace NUMINAMATH_CALUDE_circle_s_radius_l2066_206640

/-- Triangle XYZ with given side lengths -/
structure Triangle :=
  (xy : ℝ)
  (xz : ℝ)
  (yz : ℝ)

/-- Circle with given radius -/
structure Circle :=
  (radius : ℝ)

/-- Theorem stating the radius of circle S in the given triangle configuration -/
theorem circle_s_radius (t : Triangle) (r : Circle) (s : Circle) :
  t.xy = 120 →
  t.xz = 120 →
  t.yz = 80 →
  r.radius = 20 →
  -- Circle R is tangent to XZ and YZ
  -- Circle S is externally tangent to R and tangent to XY and YZ
  -- No point of circle S lies outside of triangle XYZ
  s.radius = 56 - 8 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_circle_s_radius_l2066_206640


namespace NUMINAMATH_CALUDE_det_special_matrix_l2066_206690

def matrix (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![x + 2, x, x;
     x, x + 2, x;
     x, x, x + 2]

theorem det_special_matrix (x : ℝ) :
  Matrix.det (matrix x) = 8 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l2066_206690


namespace NUMINAMATH_CALUDE_smallest_sum_with_gcd_conditions_l2066_206604

theorem smallest_sum_with_gcd_conditions (a b c : ℕ+) : 
  (Nat.gcd a.val (Nat.gcd b.val c.val) = 1) →
  (Nat.gcd a.val (b.val + c.val) > 1) →
  (Nat.gcd b.val (c.val + a.val) > 1) →
  (Nat.gcd c.val (a.val + b.val) > 1) →
  (∃ (x y z : ℕ+), x.val + y.val + z.val < a.val + b.val + c.val) →
  a.val + b.val + c.val ≥ 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_with_gcd_conditions_l2066_206604


namespace NUMINAMATH_CALUDE_tino_has_34_jellybeans_l2066_206637

/-- The number of jellybeans each person has -/
structure JellyBeans where
  arnold : ℕ
  lee : ℕ
  tino : ℕ

/-- The jellybean distribution satisfies the given conditions -/
def satisfies_conditions (jb : JellyBeans) : Prop :=
  jb.arnold = 5 ∧
  jb.arnold * 2 = jb.lee ∧
  jb.tino = jb.lee + 24

/-- Theorem: If the jellybean distribution satisfies the conditions, Tino has 34 jellybeans -/
theorem tino_has_34_jellybeans (jb : JellyBeans) (h : satisfies_conditions jb) : jb.tino = 34 := by
  sorry

end NUMINAMATH_CALUDE_tino_has_34_jellybeans_l2066_206637


namespace NUMINAMATH_CALUDE_speeding_statistics_l2066_206689

structure SpeedingCategory where
  name : Char
  percentMotorists : ℝ
  ticketRate : ℝ

def categoryA : SpeedingCategory := ⟨'A', 0.14, 0.25⟩
def categoryB : SpeedingCategory := ⟨'B', 0.07, 0.55⟩
def categoryC : SpeedingCategory := ⟨'C', 0.04, 0.80⟩
def categoryD : SpeedingCategory := ⟨'D', 0.02, 0.95⟩

def categories : List SpeedingCategory := [categoryA, categoryB, categoryC, categoryD]

theorem speeding_statistics :
  (List.sum (categories.map (λ c => c.percentMotorists)) = 0.27) ∧
  (categoryA.percentMotorists * categoryA.ticketRate = 0.035) ∧
  (categoryB.percentMotorists * categoryB.ticketRate = 0.0385) ∧
  (categoryC.percentMotorists * categoryC.ticketRate = 0.032) ∧
  (categoryD.percentMotorists * categoryD.ticketRate = 0.019) :=
by sorry

end NUMINAMATH_CALUDE_speeding_statistics_l2066_206689


namespace NUMINAMATH_CALUDE_horner_rule_v3_value_l2066_206662

def horner_v3 (a b c d e x : ℝ) : ℝ := (((x + a) * x + b) * x + c)

theorem horner_rule_v3_value :
  let f (x : ℝ) := x^4 + 2*x^3 + x^2 - 3*x - 1
  let x : ℝ := 2
  horner_v3 2 1 (-3) (-1) 0 x = 15 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_v3_value_l2066_206662


namespace NUMINAMATH_CALUDE_bob_sandwich_combinations_l2066_206650

/-- Represents the number of sandwich combinations Bob can order -/
def bobSandwichCombinations : ℕ :=
  let totalBreads : ℕ := 5
  let totalMeats : ℕ := 7
  let totalCheeses : ℕ := 6
  let turkeyCombos : ℕ := totalBreads -- Turkey/Swiss combinations
  let roastBeefRyeCombos : ℕ := totalCheeses -- Roast beef/Rye combinations
  let roastBeefSwissCombos : ℕ := totalBreads - 1 -- Roast beef/Swiss combinations (excluding Rye)
  let totalCombinations : ℕ := totalBreads * totalMeats * totalCheeses
  let forbiddenCombinations : ℕ := turkeyCombos + roastBeefRyeCombos + roastBeefSwissCombos
  totalCombinations - forbiddenCombinations

/-- Theorem stating that Bob can order exactly 194 different sandwiches -/
theorem bob_sandwich_combinations : bobSandwichCombinations = 194 := by
  sorry

end NUMINAMATH_CALUDE_bob_sandwich_combinations_l2066_206650


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l2066_206664

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (x y b : ℝ) : Prop := y = (1/2) * x + b

-- Theorem statement
theorem ellipse_and_line_intersection :
  -- Given conditions
  let left_focus : ℝ × ℝ := (-Real.sqrt 3, 0)
  let right_vertex : ℝ × ℝ := (2, 0)
  
  -- Prove the following
  -- 1. Standard equation of ellipse C
  ∀ x y : ℝ, ellipse_C x y ↔ x^2 / 4 + y^2 = 1
  
  -- 2. Maximum chord length and corresponding line equation
  ∧ ∃ max_length : ℝ,
    (max_length = Real.sqrt 10) ∧
    (∀ b : ℝ, ∃ A B : ℝ × ℝ,
      ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
      line_l A.1 A.2 b ∧ line_l B.1 B.2 b ∧
      (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ max_length)) ∧
    (∃ A B : ℝ × ℝ,
      ellipse_C A.1 A.2 ∧ ellipse_C B.1 B.2 ∧
      line_l A.1 A.2 0 ∧ line_l B.1 B.2 0 ∧
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = max_length) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l2066_206664


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l2066_206660

/-- Given vectors a and b in ℝ², prove that ‖a + 2b‖ = 2√3 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  (a.1 = 2 ∧ a.2 = 0) →  -- a = (2, 0)
  ‖b‖ = 1 →  -- ‖b‖ = 1
  (a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖) = 1/2 →  -- angle between a and b is π/3 (cos(π/3) = 1/2)
  ‖a + 2 • b‖ = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l2066_206660


namespace NUMINAMATH_CALUDE_square_perimeter_l2066_206658

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 144 →
  area = side ^ 2 →
  perimeter = 4 * side →
  perimeter = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2066_206658


namespace NUMINAMATH_CALUDE_complex_power_20_l2066_206695

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_20 : (1 + i) ^ 20 = -1024 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_20_l2066_206695


namespace NUMINAMATH_CALUDE_symmetric_log_value_of_a_l2066_206608

/-- Given a function f and a real number a, we say f is symmetric to log₂(x+a) with respect to y = x -/
def symmetric_to_log (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x = 2^x - a

theorem symmetric_log_value_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_sym : symmetric_to_log f a) (h_sum : f 2 + f 4 = 6) : a = 7 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_log_value_of_a_l2066_206608


namespace NUMINAMATH_CALUDE_max_sum_of_first_three_l2066_206670

theorem max_sum_of_first_three (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℕ) 
  (h_order : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧ x₄ < x₅ ∧ x₅ < x₆ ∧ x₆ < x₇)
  (h_sum : x₁ + x₂ + x₃ + x₄ + x₅ + x₆ + x₇ = 159) :
  (∀ y₁ y₂ y₃ : ℕ, y₁ < y₂ ∧ y₂ < y₃ ∧ 
    (∃ y₄ y₅ y₆ y₇ : ℕ, y₃ < y₄ ∧ y₄ < y₅ ∧ y₅ < y₆ ∧ y₆ < y₇ ∧
      y₁ + y₂ + y₃ + y₄ + y₅ + y₆ + y₇ = 159) →
    y₁ + y₂ + y₃ ≤ 61) ∧
  (x₁ + x₂ + x₃ = 61) := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_first_three_l2066_206670


namespace NUMINAMATH_CALUDE_solution_set_f_less_g_plus_a_range_of_a_for_f_plus_g_greater_a_squared_l2066_206679

-- Define the functions f and g
def f (x a : ℝ) : ℝ := |x - 2| + a
def g (x : ℝ) : ℝ := |x + 4|

-- Theorem for part I
theorem solution_set_f_less_g_plus_a (a : ℝ) :
  {x : ℝ | f x a < g x + a} = {x : ℝ | x > -1} :=
sorry

-- Theorem for part II
theorem range_of_a_for_f_plus_g_greater_a_squared :
  {a : ℝ | ∀ x, f x a + g x > a^2} = {a : ℝ | -2 < a ∧ a < 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_g_plus_a_range_of_a_for_f_plus_g_greater_a_squared_l2066_206679


namespace NUMINAMATH_CALUDE_expression_evaluation_l2066_206649

theorem expression_evaluation :
  let x : ℚ := 1/2
  let y : ℚ := -1/4
  ((3*x + 2*y) * (3*x - 2*y) - (3*x - 2*y)^2) / (4*y) = 2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2066_206649


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2066_206653

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the sum of specific terms in the arithmetic sequence -/
theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_roots : a 5 ^ 2 - 6 * a 5 - 1 = 0 ∧ a 13 ^ 2 - 6 * a 13 - 1 = 0) :
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2066_206653


namespace NUMINAMATH_CALUDE_points_five_units_from_negative_three_l2066_206638

theorem points_five_units_from_negative_three (x : ℝ) : 
  |x - (-3)| = 5 ↔ x = -8 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_points_five_units_from_negative_three_l2066_206638


namespace NUMINAMATH_CALUDE_complex_equation_unit_circle_l2066_206673

theorem complex_equation_unit_circle (z : ℂ) :
  11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0 →
  Complex.abs z = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_unit_circle_l2066_206673


namespace NUMINAMATH_CALUDE_smallest_multiple_l2066_206634

theorem smallest_multiple (n : ℕ) : n = 663 ↔ 
  n > 0 ∧ 
  n % 17 = 0 ∧ 
  (n - 6) % 73 = 0 ∧ 
  ∀ m : ℕ, m > 0 → m % 17 = 0 → (m - 6) % 73 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2066_206634


namespace NUMINAMATH_CALUDE_max_gcd_of_consecutive_bn_l2066_206605

theorem max_gcd_of_consecutive_bn (n : ℕ) : Nat.gcd (2^n - 1) (2^(n+1) - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_of_consecutive_bn_l2066_206605


namespace NUMINAMATH_CALUDE_polynomial_never_33_l2066_206617

theorem polynomial_never_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
sorry

end NUMINAMATH_CALUDE_polynomial_never_33_l2066_206617


namespace NUMINAMATH_CALUDE_assign_teachers_count_l2066_206632

/-- The number of ways to assign 6 teachers to 4 grades -/
def assign_teachers : ℕ :=
  let n_teachers : ℕ := 6
  let n_grades : ℕ := 4
  let two_specific_teachers : ℕ := 2
  -- Define the function to calculate the number of ways
  sorry

/-- Theorem stating that the number of ways to assign teachers is 240 -/
theorem assign_teachers_count : assign_teachers = 240 := by
  sorry

end NUMINAMATH_CALUDE_assign_teachers_count_l2066_206632


namespace NUMINAMATH_CALUDE_range_of_a_in_fourth_quadrant_l2066_206622

/-- A point in the fourth quadrant has a positive x-coordinate and a negative y-coordinate -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The range of a for a point P(a, a-2) in the fourth quadrant -/
theorem range_of_a_in_fourth_quadrant (a : ℝ) :
  fourth_quadrant a (a - 2) ↔ 0 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_in_fourth_quadrant_l2066_206622


namespace NUMINAMATH_CALUDE_total_cds_l2066_206672

/-- The number of CDs each person has -/
structure CDCounts where
  dawn : ℕ
  kristine : ℕ
  mark : ℕ
  alice : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (c : CDCounts) : Prop :=
  c.dawn = 10 ∧
  c.kristine = c.dawn + 7 ∧
  c.mark = 2 * c.kristine ∧
  c.alice = c.kristine + c.mark - 5

/-- The theorem to prove -/
theorem total_cds (c : CDCounts) (h : satisfiesConditions c) :
  c.dawn + c.kristine + c.mark + c.alice = 107 := by
  sorry

#check total_cds

end NUMINAMATH_CALUDE_total_cds_l2066_206672


namespace NUMINAMATH_CALUDE_two_quadrilaterals_nine_regions_l2066_206624

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral in the plane -/
structure Quadrilateral :=
  (p1 p2 p3 p4 : Point)

/-- The plane divided by quadrilaterals -/
def PlaneDivision :=
  List Quadrilateral

/-- Count the number of regions in a plane division -/
def countRegions (division : PlaneDivision) : ℕ :=
  sorry

/-- Theorem: There exists a plane division with two quadrilaterals that results in 9 regions -/
theorem two_quadrilaterals_nine_regions :
  ∃ (division : PlaneDivision),
    division.length = 2 ∧ countRegions division = 9 :=
  sorry

end NUMINAMATH_CALUDE_two_quadrilaterals_nine_regions_l2066_206624


namespace NUMINAMATH_CALUDE_bird_count_proof_l2066_206680

theorem bird_count_proof (cardinals : ℕ) (robins : ℕ) (blue_jays : ℕ) (sparrows : ℕ) (pigeons : ℕ) :
  cardinals = 3 ∧
  robins = 4 * cardinals ∧
  blue_jays = 2 * cardinals ∧
  sparrows = 3 * cardinals + 1 ∧
  pigeons = 3 * blue_jays →
  cardinals + robins + blue_jays + sparrows + pigeons = 49 := by
  sorry


end NUMINAMATH_CALUDE_bird_count_proof_l2066_206680


namespace NUMINAMATH_CALUDE_reciprocal_square_inequality_l2066_206613

theorem reciprocal_square_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≤ y) : 
  1 / y^2 ≤ 1 / x^2 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_square_inequality_l2066_206613


namespace NUMINAMATH_CALUDE_total_plums_picked_l2066_206607

theorem total_plums_picked (melanie_plums dan_plums sally_plums : ℕ) 
  (h1 : melanie_plums = 4)
  (h2 : dan_plums = 9)
  (h3 : sally_plums = 3) :
  melanie_plums + dan_plums + sally_plums = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_plums_picked_l2066_206607


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l2066_206636

/-- If the vertex of the parabola y = x^2 + 2x + c is on the x-axis, then c = 1 -/
theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + c = 0 ∧ ∀ t : ℝ, t^2 + 2*t + c ≥ x^2 + 2*x + c) → c = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l2066_206636


namespace NUMINAMATH_CALUDE_binary_multiplication_division_l2066_206645

def binary_to_nat (b : List Bool) : Nat :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_helper (n : Nat) : List Bool :=
    if n = 0 then [] else (n % 2 = 1) :: to_binary_helper (n / 2)
  to_binary_helper n

def binary_110010 : List Bool := [false, true, false, false, true, true]
def binary_1101 : List Bool := [true, false, true, true]
def binary_101 : List Bool := [true, false, true]
def binary_11110100 : List Bool := [false, false, true, false, true, true, true, true]

theorem binary_multiplication_division :
  (binary_to_nat binary_110010 * binary_to_nat binary_1101) / binary_to_nat binary_101 =
  binary_to_nat binary_11110100 := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_l2066_206645


namespace NUMINAMATH_CALUDE_square_vertex_C_l2066_206648

def square (A B C D : ℂ) : Prop :=
  (B - A) * Complex.I = C - B ∧
  (C - B) * Complex.I = D - C ∧
  (D - C) * Complex.I = A - D ∧
  (A - D) * Complex.I = B - A

theorem square_vertex_C (A B C D : ℂ) :
  square A B C D →
  A = 1 + 2*Complex.I →
  B = 3 - 5*Complex.I →
  C = 10 - 3*Complex.I :=
by
  sorry

#check square_vertex_C

end NUMINAMATH_CALUDE_square_vertex_C_l2066_206648


namespace NUMINAMATH_CALUDE_ellipse_m_range_l2066_206683

/-- Represents an ellipse with equation x^2/m^2 + y^2/(2+m) = 1 -/
structure Ellipse (m : ℝ) where
  equation : ∀ x y : ℝ, x^2/m^2 + y^2/(2+m) = 1

/-- Condition for foci on x-axis -/
def foci_on_x_axis (m : ℝ) : Prop := m^2 > 2 + m

/-- The range of m for which the ellipse is valid and has foci on x-axis -/
def valid_m_range (m : ℝ) : Prop := (m > 2 ∨ (-2 < m ∧ m < -1))

theorem ellipse_m_range (m : ℝ) (e : Ellipse m) :
  foci_on_x_axis m → valid_m_range m :=
by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l2066_206683


namespace NUMINAMATH_CALUDE_typists_productivity_l2066_206627

/-- Given that 20 typists can type 48 letters in 20 minutes, 
    prove that 30 typists can type 216 letters in 1 hour at the same rate. -/
theorem typists_productivity (typists_base : ℕ) (letters_base : ℕ) (minutes_base : ℕ)
  (typists_new : ℕ) (minutes_new : ℕ) :
  typists_base = 20 →
  letters_base = 48 →
  minutes_base = 20 →
  typists_new = 30 →
  minutes_new = 60 →
  (typists_new * letters_base * minutes_new) / (typists_base * minutes_base) = 216 :=
by sorry

end NUMINAMATH_CALUDE_typists_productivity_l2066_206627


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l2066_206626

theorem min_sum_of_squares (x y : ℝ) (h : (x + 8) * (y - 8) = 0) :
  ∃ (min : ℝ), min = 128 ∧ ∀ (a b : ℝ), (a + 8) * (b - 8) = 0 → a^2 + b^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l2066_206626


namespace NUMINAMATH_CALUDE_point_on_parametric_line_l2066_206639

/-- Given a line l defined by parametric equations x = 1 + t and y = 3 - at,
    prove that a = -1 if the point P(2,4) lies on this line. -/
theorem point_on_parametric_line (t : ℝ) (a : ℝ) : 
  (2 = 1 + t ∧ 4 = 3 - a * t) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_parametric_line_l2066_206639


namespace NUMINAMATH_CALUDE_complex_number_solution_l2066_206685

theorem complex_number_solution (z : ℂ) : 
  Complex.abs z = Real.sqrt 13 ∧ 
  ∃ (k : ℝ), (2 + 3*I)*z*I = k*I → 
  z = 3 + 2*I ∨ z = -3 - 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_solution_l2066_206685


namespace NUMINAMATH_CALUDE_binomial_parameters_determination_l2066_206633

/-- A random variable X following a Binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a Binomial distribution -/
def expectedValue (X : BinomialDistribution) : ℝ := X.n * X.p

/-- The variance of a Binomial distribution -/
def variance (X : BinomialDistribution) : ℝ := X.n * X.p * (1 - X.p)

/-- Theorem: For a Binomial distribution X with EX = 8 and DX = 1.6, n = 100 and p = 0.08 -/
theorem binomial_parameters_determination :
  ∀ X : BinomialDistribution, 
    expectedValue X = 8 → 
    variance X = 1.6 → 
    X.n = 100 ∧ X.p = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_binomial_parameters_determination_l2066_206633


namespace NUMINAMATH_CALUDE_total_length_of_objects_l2066_206674

/-- Given the lengths of various objects and their relationships, prove their total length. -/
theorem total_length_of_objects (pencil_length : ℝ) 
  (h1 : pencil_length = 12) 
  (h2 : ∃ pen_length rubber_length, 
    pen_length = rubber_length + 3 ∧ 
    pencil_length = pen_length + 2)
  (h3 : ∃ ruler_length, 
    ruler_length = 3 * rubber_length ∧ 
    ruler_length = pen_length * 1.2)
  (h4 : ∃ marker_length, marker_length = ruler_length / 2)
  (h5 : ∃ scissors_length, scissors_length = pencil_length * 0.75) :
  ∃ total_length, total_length = 69.5 ∧ 
    total_length = rubber_length + pen_length + pencil_length + 
                   marker_length + ruler_length + scissors_length :=
by sorry

end NUMINAMATH_CALUDE_total_length_of_objects_l2066_206674


namespace NUMINAMATH_CALUDE_blue_marbles_count_l2066_206614

theorem blue_marbles_count (total : ℕ) (red : ℕ) (prob_red_or_white : ℚ) 
  (h_total : total = 20)
  (h_red : red = 7)
  (h_prob : prob_red_or_white = 3/4) :
  ∃ blue : ℕ, blue = 5 ∧ 
    (red : ℚ) / total + (total - red - blue : ℚ) / total = prob_red_or_white :=
by sorry

end NUMINAMATH_CALUDE_blue_marbles_count_l2066_206614


namespace NUMINAMATH_CALUDE_base_prime_repr_360_l2066_206603

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- The number 360 -/
def n : ℕ := 360

theorem base_prime_repr_360 :
  base_prime_repr n = [3, 2, 1] :=
by
  sorry

end NUMINAMATH_CALUDE_base_prime_repr_360_l2066_206603


namespace NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_16385_l2066_206652

def greatest_prime_divisor (n : Nat) : Nat :=
  sorry

def sum_of_digits (n : Nat) : Nat :=
  sorry

theorem sum_of_digits_greatest_prime_divisor_16385 :
  sum_of_digits (greatest_prime_divisor 16385) = 19 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_greatest_prime_divisor_16385_l2066_206652


namespace NUMINAMATH_CALUDE_bethany_current_age_l2066_206615

/-- Bethany's current age -/
def bethany_age : ℕ := sorry

/-- Bethany's sister's current age -/
def sister_age : ℕ := sorry

/-- Bethany's brother's current age -/
def brother_age : ℕ := sorry

/-- Theorem stating Bethany's current age given the conditions -/
theorem bethany_current_age :
  (bethany_age - 3 = 2 * (sister_age - 3)) ∧
  (bethany_age - 3 = brother_age - 3 + 4) ∧
  (sister_age + 5 = 16) ∧
  (brother_age + 5 = 21) →
  bethany_age = 19 := by sorry

end NUMINAMATH_CALUDE_bethany_current_age_l2066_206615
