import Mathlib

namespace NUMINAMATH_CALUDE_pythagorean_triple_6_8_10_l2866_286606

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_6_8_10 :
  (is_pythagorean_triple 6 8 10) ∧
  ¬(is_pythagorean_triple 6 7 10) ∧
  ¬(is_pythagorean_triple 1 2 3) ∧
  ¬(is_pythagorean_triple 4 5 8) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_6_8_10_l2866_286606


namespace NUMINAMATH_CALUDE_license_plate_count_l2866_286681

def alphabet_size : ℕ := 26
def digit_count : ℕ := 10

def license_plate_combinations : ℕ :=
  -- Choose first repeated letter
  alphabet_size *
  -- Choose second repeated letter
  (alphabet_size - 1) *
  -- Choose two other unique letters
  (Nat.choose (alphabet_size - 2) 2) *
  -- Positions for first repeated letter
  (Nat.choose 6 2) *
  -- Positions for second repeated letter
  (Nat.choose 4 2) *
  -- Arrange two unique letters
  2 *
  -- Choose first digit
  digit_count *
  -- Choose second digit
  (digit_count - 1) *
  -- Choose third digit
  (digit_count - 2)

theorem license_plate_count :
  license_plate_combinations = 241164000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2866_286681


namespace NUMINAMATH_CALUDE_second_half_speed_l2866_286615

theorem second_half_speed (total_distance : ℝ) (first_half_speed : ℝ) (total_time : ℝ)
  (h1 : total_distance = 3600)
  (h2 : first_half_speed = 90)
  (h3 : total_time = 30) :
  (total_distance / 2) / (total_time - (total_distance / 2) / first_half_speed) = 180 :=
by sorry

end NUMINAMATH_CALUDE_second_half_speed_l2866_286615


namespace NUMINAMATH_CALUDE_monotonic_decrease_interval_l2866_286653

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * |2 * x - 4|

-- State the theorem
theorem monotonic_decrease_interval
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : f a 1 = 9) :
  ∃ (I : Set ℝ), StrictMonoOn (f a) (Set.Iic 2) ∧ I = Set.Iic 2 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decrease_interval_l2866_286653


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l2866_286678

/-- A geometric progression is defined by its first term and common ratio -/
structure GeometricProgression where
  b₁ : ℚ  -- First term
  q : ℚ   -- Common ratio

/-- The nth term of a geometric progression -/
def GeometricProgression.nthTerm (gp : GeometricProgression) (n : ℕ) : ℚ :=
  gp.b₁ * gp.q ^ (n - 1)

theorem geometric_progression_solution :
  ∃ (gp : GeometricProgression),
    gp.nthTerm 3 = -1 ∧
    gp.nthTerm 6 = 27/8 ∧
    gp.b₁ = -4/9 ∧
    gp.q = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l2866_286678


namespace NUMINAMATH_CALUDE_smallest_invertible_domain_l2866_286643

/-- The function g(x) = (x-3)^2 + 1 -/
def g (x : ℝ) : ℝ := (x - 3)^2 + 1

/-- g is invertible on [c,∞) -/
def invertible_on (c : ℝ) : Prop :=
  ∀ x y, x ≥ c → y ≥ c → g x = g y → x = y

/-- The smallest value of c for which g is invertible on [c,∞) -/
theorem smallest_invertible_domain : 
  (∃ c, invertible_on c ∧ ∀ c', c' < c → ¬invertible_on c') ∧ 
  (∀ c, invertible_on c → c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_invertible_domain_l2866_286643


namespace NUMINAMATH_CALUDE_lcm_14_21_l2866_286654

theorem lcm_14_21 : Nat.lcm 14 21 = 42 := by
  sorry

end NUMINAMATH_CALUDE_lcm_14_21_l2866_286654


namespace NUMINAMATH_CALUDE_two_tangent_lines_l2866_286614

/-- A line that passes through a point and intersects a parabola at only one point. -/
structure TangentLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The point through which the line passes -/
  point : ℝ × ℝ
  /-- The parabola equation in the form y^2 = ax -/
  parabola_coeff : ℝ

/-- The number of lines passing through a given point and tangent to a parabola -/
def count_tangent_lines (point : ℝ × ℝ) (parabola_coeff : ℝ) : ℕ :=
  sorry

/-- Theorem: There are exactly two lines that pass through point M(2, 4) 
    and intersect the parabola y^2 = 8x at only one point -/
theorem two_tangent_lines : count_tangent_lines (2, 4) 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_tangent_lines_l2866_286614


namespace NUMINAMATH_CALUDE_bug_travel_distance_l2866_286639

/-- The total distance traveled by a bug on a number line -/
def bugDistance (start end1 end2 end3 : ℤ) : ℝ :=
  |end1 - start| + |end2 - end1| + |end3 - end2|

/-- Theorem: The bug's total travel distance is 25 units -/
theorem bug_travel_distance :
  bugDistance (-3) (-7) 8 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_bug_travel_distance_l2866_286639


namespace NUMINAMATH_CALUDE_baseball_cards_count_l2866_286669

theorem baseball_cards_count (num_friends : ℕ) (cards_per_friend : ℕ) : 
  num_friends = 5 → cards_per_friend = 91 → num_friends * cards_per_friend = 455 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_count_l2866_286669


namespace NUMINAMATH_CALUDE_female_democrat_ratio_is_half_l2866_286680

/-- Represents the number of participants in a meeting with given conditions -/
structure Meeting where
  total : ℕ
  maleDemocratRatio : ℚ
  totalDemocratRatio : ℚ
  femaleDemocrats : ℕ
  male : ℕ
  female : ℕ

/-- The ratio of female democrats to total female participants -/
def femaleDemocratRatio (m : Meeting) : ℚ :=
  m.femaleDemocrats / m.female

theorem female_democrat_ratio_is_half (m : Meeting) 
  (h1 : m.total = 660)
  (h2 : m.maleDemocratRatio = 1/4)
  (h3 : m.totalDemocratRatio = 1/3)
  (h4 : m.femaleDemocrats = 110)
  (h5 : m.male + m.female = m.total)
  (h6 : m.maleDemocratRatio * m.male + m.femaleDemocrats = m.totalDemocratRatio * m.total) :
  femaleDemocratRatio m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_female_democrat_ratio_is_half_l2866_286680


namespace NUMINAMATH_CALUDE_original_average_weight_l2866_286619

theorem original_average_weight 
  (original_count : ℕ) 
  (new_boy_weight : ℝ) 
  (average_increase : ℝ) : 
  original_count = 5 →
  new_boy_weight = 40 →
  average_increase = 1 →
  (original_count : ℝ) * ((original_count : ℝ) * average_increase + new_boy_weight) / 
    (original_count + 1) - new_boy_weight = 34 := by
  sorry

end NUMINAMATH_CALUDE_original_average_weight_l2866_286619


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l2866_286688

theorem parabola_x_intercepts :
  ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 4 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l2866_286688


namespace NUMINAMATH_CALUDE_parallel_condition_necessary_not_sufficient_l2866_286640

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Define parallelism for two lines -/
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1 ≠ l2

/-- The first line: 2x + ay - 1 = 0 -/
def line1 (a : ℝ) : Line2D :=
  { a := 2, b := a, c := -1 }

/-- The second line: bx + 2y - 2 = 0 -/
def line2 (b : ℝ) : Line2D :=
  { a := b, b := 2, c := -2 }

/-- The main theorem -/
theorem parallel_condition_necessary_not_sufficient :
  (∀ a b : ℝ, parallel (line1 a) (line2 b) → a * b = 4) ∧
  ¬(∀ a b : ℝ, a * b = 4 → parallel (line1 a) (line2 b)) := by
  sorry

end NUMINAMATH_CALUDE_parallel_condition_necessary_not_sufficient_l2866_286640


namespace NUMINAMATH_CALUDE_problem_solution_l2866_286607

theorem problem_solution (x y : ℝ) 
  (h1 : x ≠ 0) 
  (h2 : x / 3 = y ^ 2) 
  (h3 : x / 5 = 5 * y) : 
  x = 625 / 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2866_286607


namespace NUMINAMATH_CALUDE_koolaid_percentage_is_four_percent_l2866_286617

/-- Calculates the percentage of koolaid powder in a mixture --/
def koolaid_percentage (initial_powder : ℚ) (initial_water : ℚ) (evaporated_water : ℚ) : ℚ :=
  let remaining_water := initial_water - evaporated_water
  let final_water := 4 * remaining_water
  let total_volume := final_water + initial_powder
  (initial_powder / total_volume) * 100

/-- Theorem stating that the percentage of koolaid powder is 4% given the initial conditions --/
theorem koolaid_percentage_is_four_percent :
  koolaid_percentage 2 16 4 = 4 := by sorry

end NUMINAMATH_CALUDE_koolaid_percentage_is_four_percent_l2866_286617


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2866_286699

theorem imaginary_part_of_complex_fraction : Complex.im ((2 + 4 * Complex.I) / (1 + Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2866_286699


namespace NUMINAMATH_CALUDE_expected_total_rolls_leap_year_l2866_286626

/-- Represents the outcome of rolling an eight-sided die -/
inductive DieRoll
| one | two | three | four | five | six | seven | eight

/-- Defines if a roll is a perfect square (1 or 4) -/
def isPerfectSquare (roll : DieRoll) : Prop :=
  roll = DieRoll.one ∨ roll = DieRoll.four

/-- Calculates the probability of rolling a perfect square -/
def probPerfectSquare : ℚ := 1/4

/-- Calculates the probability of not rolling a perfect square -/
def probNotPerfectSquare : ℚ := 3/4

/-- The number of days in a leap year -/
def daysInLeapYear : ℕ := 366

/-- The expected number of rolls per day -/
noncomputable def expectedRollsPerDay : ℚ := 4/3

/-- Theorem: The expected total number of rolls in a leap year is 488 -/
theorem expected_total_rolls_leap_year :
  (expectedRollsPerDay * daysInLeapYear : ℚ) = 488 := by
  sorry

end NUMINAMATH_CALUDE_expected_total_rolls_leap_year_l2866_286626


namespace NUMINAMATH_CALUDE_calculation_proof_l2866_286624

theorem calculation_proof : (2 - Real.pi) ^ 0 - 2⁻¹ + Real.cos (60 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2866_286624


namespace NUMINAMATH_CALUDE_divisibility_by_27_l2866_286631

theorem divisibility_by_27 (t : ℤ) : 
  27 ∣ (7 * (27 * t + 16)^4 + 19 * (27 * t + 16) + 25) := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_27_l2866_286631


namespace NUMINAMATH_CALUDE_calculate_expression_l2866_286616

theorem calculate_expression : -2^4 + 3 * (-1)^2010 - (-2)^2 = -17 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2866_286616


namespace NUMINAMATH_CALUDE_johns_allowance_spent_l2866_286666

theorem johns_allowance_spent (allowance : ℚ) (arcade_fraction : ℚ) (candy_spent : ℚ) 
  (h1 : allowance = 3.375)
  (h2 : arcade_fraction = 3/5)
  (h3 : candy_spent = 0.9) :
  let remaining := allowance - arcade_fraction * allowance
  let toy_spent := remaining - candy_spent
  toy_spent / remaining = 1/3 := by sorry

end NUMINAMATH_CALUDE_johns_allowance_spent_l2866_286666


namespace NUMINAMATH_CALUDE_f_2_equals_5_l2866_286658

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 1

theorem f_2_equals_5 : f 2 = 5 := by sorry

end NUMINAMATH_CALUDE_f_2_equals_5_l2866_286658


namespace NUMINAMATH_CALUDE_wednesday_to_tuesday_rainfall_ratio_l2866_286623

/-- Represents the rainfall data for a day -/
structure RainfallData where
  hours : ℝ
  rate : ℝ

/-- Calculates the total rainfall for a given day -/
def totalRainfall (data : RainfallData) : ℝ := data.hours * data.rate

theorem wednesday_to_tuesday_rainfall_ratio :
  let monday : RainfallData := { hours := 7, rate := 1 }
  let tuesday : RainfallData := { hours := 4, rate := 2 }
  let wednesday : RainfallData := { hours := 2, rate := ((23 : ℝ) - totalRainfall monday - totalRainfall tuesday) / 2 }
  wednesday.rate / tuesday.rate = 2 := by sorry

end NUMINAMATH_CALUDE_wednesday_to_tuesday_rainfall_ratio_l2866_286623


namespace NUMINAMATH_CALUDE_rectangle_length_l2866_286696

/-- Given a rectangle where the length is three times the breadth and the area is 6075 square meters,
    prove that the length of the rectangle is 135 meters. -/
theorem rectangle_length (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * breadth → 
  area = length * breadth → 
  area = 6075 → 
  length = 135 := by sorry

end NUMINAMATH_CALUDE_rectangle_length_l2866_286696


namespace NUMINAMATH_CALUDE_ninth_grade_classes_l2866_286693

theorem ninth_grade_classes (total_matches : ℕ) (h : total_matches = 28) :
  ∃ x : ℕ, x * (x - 1) / 2 = total_matches ∧ x = 8 :=
by sorry

end NUMINAMATH_CALUDE_ninth_grade_classes_l2866_286693


namespace NUMINAMATH_CALUDE_family_average_age_l2866_286635

theorem family_average_age (grandparents_avg : ℝ) (parents_avg : ℝ) (grandchildren_avg : ℝ)
  (h1 : grandparents_avg = 64)
  (h2 : parents_avg = 39)
  (h3 : grandchildren_avg = 6) :
  (2 * grandparents_avg + 2 * parents_avg + 3 * grandchildren_avg) / 7 = 32 := by
  sorry

end NUMINAMATH_CALUDE_family_average_age_l2866_286635


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l2866_286690

theorem largest_integer_satisfying_inequality :
  ∀ x : ℤ, x ≤ 3 ↔ 3 * x + 4 < 5 * x - 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l2866_286690


namespace NUMINAMATH_CALUDE_square_equation_solution_l2866_286641

theorem square_equation_solution (x : ℝ) : (x + 3)^2 = 121 ↔ x = 8 ∨ x = -14 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2866_286641


namespace NUMINAMATH_CALUDE_evaluate_expression_l2866_286618

theorem evaluate_expression : 6 - 9 * (1 / 2 - 3^3) * 2 = 483 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2866_286618


namespace NUMINAMATH_CALUDE_fencing_calculation_l2866_286697

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_calculation (area : ℝ) (uncovered_side : ℝ) : 
  area = 600 → uncovered_side = 30 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    2 * width + uncovered_side = 70 :=
by sorry

end NUMINAMATH_CALUDE_fencing_calculation_l2866_286697


namespace NUMINAMATH_CALUDE_atomic_weight_Al_l2866_286662

/-- The atomic weight of oxygen -/
def atomic_weight_O : ℝ := 16

/-- The molecular weight of Al2O3 -/
def molecular_weight_Al2O3 : ℝ := 102

/-- The number of aluminum atoms in Al2O3 -/
def num_Al_atoms : ℕ := 2

/-- The number of oxygen atoms in Al2O3 -/
def num_O_atoms : ℕ := 3

/-- Theorem stating that the atomic weight of Al is 27 -/
theorem atomic_weight_Al :
  (molecular_weight_Al2O3 - num_O_atoms * atomic_weight_O) / num_Al_atoms = 27 := by
  sorry

end NUMINAMATH_CALUDE_atomic_weight_Al_l2866_286662


namespace NUMINAMATH_CALUDE_function_equivalence_l2866_286602

theorem function_equivalence : ∀ x : ℝ, (3 * x)^3 = x := by
  sorry

end NUMINAMATH_CALUDE_function_equivalence_l2866_286602


namespace NUMINAMATH_CALUDE_equal_intercept_line_equations_equidistant_point_locus_l2866_286610

/-- A line passing through a point with equal intercepts on both axes -/
structure EqualInterceptLine where
  a : ℝ
  b : ℝ
  passes_through : a + b = 4
  equal_intercepts : a = b

/-- A point equidistant from two parallel lines -/
structure EquidistantPoint where
  x : ℝ
  y : ℝ
  equidistant : |4*x + 6*y - 10| = |4*x + 6*y + 8|

/-- Theorem for the equal intercept line -/
theorem equal_intercept_line_equations (l : EqualInterceptLine) :
  (∀ x y, y = 3*x) ∨ (∀ x y, y = -x + 4) :=
sorry

/-- Theorem for the locus of equidistant points -/
theorem equidistant_point_locus (p : EquidistantPoint) :
  4*p.x + 6*p.y - 9 = 0 :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equations_equidistant_point_locus_l2866_286610


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l2866_286611

/-- For z = 1 + i and a ∈ ℝ, if (1 - ai) / z is a pure imaginary number, then a = 1 -/
theorem complex_pure_imaginary_condition (a : ℝ) : 
  let z : ℂ := 1 + I
  (((1 : ℂ) - a * I) / z).re = 0 → (((1 : ℂ) - a * I) / z).im ≠ 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_condition_l2866_286611


namespace NUMINAMATH_CALUDE_min_value_of_function_l2866_286677

theorem min_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (∀ y : ℝ, y > 0 ∧ y < 1 → (4 / x + 1 / (1 - x)) ≤ (4 / y + 1 / (1 - y))) ∧
  (∃ z : ℝ, z > 0 ∧ z < 1 ∧ 4 / z + 1 / (1 - z) = 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2866_286677


namespace NUMINAMATH_CALUDE_total_bottles_l2866_286612

theorem total_bottles (regular : ℕ) (diet : ℕ) (lite : ℕ)
  (h1 : regular = 57)
  (h2 : diet = 26)
  (h3 : lite = 27) :
  regular + diet + lite = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_bottles_l2866_286612


namespace NUMINAMATH_CALUDE_shoes_sold_this_week_l2866_286674

def monthly_goal : ℕ := 80
def sold_last_week : ℕ := 27
def still_needed : ℕ := 41

theorem shoes_sold_this_week :
  monthly_goal - sold_last_week - still_needed = 12 := by
  sorry

end NUMINAMATH_CALUDE_shoes_sold_this_week_l2866_286674


namespace NUMINAMATH_CALUDE_external_tangent_circle_distance_l2866_286634

theorem external_tangent_circle_distance 
  (O P : ℝ × ℝ) 
  (r₁ r₂ : ℝ) 
  (Q : ℝ × ℝ) 
  (T : ℝ × ℝ) 
  (Z : ℝ × ℝ) :
  r₁ = 10 →
  r₂ = 3 →
  dist O P = r₁ + r₂ →
  dist O T = r₁ →
  dist P Z = r₂ →
  (T.1 - O.1) * (Z.1 - T.1) + (T.2 - O.2) * (Z.2 - T.2) = 0 →
  (Z.1 - P.1) * (Z.1 - T.1) + (Z.2 - P.2) * (Z.2 - T.2) = 0 →
  dist O Z = 2 * Real.sqrt 145 :=
by sorry

end NUMINAMATH_CALUDE_external_tangent_circle_distance_l2866_286634


namespace NUMINAMATH_CALUDE_joe_watching_schedule_l2866_286609

/-- The number of episodes Joe needs to watch per day to catch up with the season premiere. -/
def episodes_per_day (days_until_premiere : ℕ) (num_seasons : ℕ) (episodes_per_season : ℕ) : ℕ :=
  (num_seasons * episodes_per_season) / days_until_premiere

/-- Theorem stating that Joe needs to watch 6 episodes per day. -/
theorem joe_watching_schedule :
  episodes_per_day 10 4 15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_joe_watching_schedule_l2866_286609


namespace NUMINAMATH_CALUDE_train_distance_l2866_286689

/-- The distance between two trains after 30 seconds -/
theorem train_distance (speed1 speed2 : ℝ) (time : ℝ) : 
  speed1 = 36 →
  speed2 = 48 →
  time = 30 / 3600 →
  let d1 := speed1 * time * 1000
  let d2 := speed2 * time * 1000
  Real.sqrt (d1^2 + d2^2) = 500 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l2866_286689


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l2866_286645

/-- Given a rectangle with sides L and W, prove that if one side is measured 5% in excess
    and the calculated area has an error of 0.8%, the other side must be measured 4% in deficit. -/
theorem rectangle_measurement_error (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let L' := 1.05 * L
  let W' := W * (1 - p)
  let A := L * W
  let A' := L' * W'
  A' = 1.008 * A →
  p = 0.04
  := by sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l2866_286645


namespace NUMINAMATH_CALUDE_units_digit_sum_squares_odd_plus_7_1011_l2866_286613

/-- The units digit of the sum of squares of the first n odd positive integers plus 7 -/
def units_digit_sum_squares_odd_plus_7 (n : ℕ) : ℕ :=
  (((List.range n).map (fun i => (2 * i + 1) ^ 2)).sum + 7) % 10

/-- Theorem stating that the units digit of the sum of squares of the first 1011 odd positive integers plus 7 is 2 -/
theorem units_digit_sum_squares_odd_plus_7_1011 :
  units_digit_sum_squares_odd_plus_7 1011 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_squares_odd_plus_7_1011_l2866_286613


namespace NUMINAMATH_CALUDE_bug_return_probability_l2866_286621

/-- Probability of the bug being at the starting corner after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - Q n)

/-- The probability of the bug returning to its starting corner on its eighth move -/
theorem bug_return_probability : Q 8 = 547 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l2866_286621


namespace NUMINAMATH_CALUDE_symmetry_across_x_axis_l2866_286644

def point_symmetrical_to_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

theorem symmetry_across_x_axis :
  let M : ℝ × ℝ := (1, 3)
  let N : ℝ × ℝ := (1, -3)
  point_symmetrical_to_x_axis M N :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_across_x_axis_l2866_286644


namespace NUMINAMATH_CALUDE_petya_ran_less_than_two_minutes_l2866_286670

/-- Represents the race between Petya and Vasya -/
structure Race where
  distance : ℝ
  petyaSpeed : ℝ
  petyaTime : ℝ
  vasyaStartDelay : ℝ

/-- Conditions of the race -/
def raceConditions (r : Race) : Prop :=
  r.distance > 0 ∧
  r.petyaSpeed > 0 ∧
  r.petyaTime > 0 ∧
  r.vasyaStartDelay = 1 ∧
  r.distance = r.petyaSpeed * r.petyaTime ∧
  r.petyaTime < r.distance / (2 * r.petyaSpeed) + r.vasyaStartDelay

/-- Theorem: Under the given conditions, Petya ran the distance in less than two minutes -/
theorem petya_ran_less_than_two_minutes (r : Race) (h : raceConditions r) : r.petyaTime < 2 := by
  sorry

end NUMINAMATH_CALUDE_petya_ran_less_than_two_minutes_l2866_286670


namespace NUMINAMATH_CALUDE_absolute_value_sum_l2866_286665

theorem absolute_value_sum (a : ℝ) (h : 1 < a ∧ a < 2) : |a - 2| + |1 - a| = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l2866_286665


namespace NUMINAMATH_CALUDE_days_missed_difference_l2866_286687

/-- Represents the frequency histogram of days missed --/
structure FrequencyHistogram :=
  (days : List Nat)
  (frequencies : List Nat)
  (total_students : Nat)

/-- Calculate the median of the dataset --/
def median (h : FrequencyHistogram) : Rat :=
  sorry

/-- Calculate the mean of the dataset --/
def mean (h : FrequencyHistogram) : Rat :=
  sorry

/-- The main theorem --/
theorem days_missed_difference (h : FrequencyHistogram) 
  (h_days : h.days = [0, 1, 2, 3, 4, 5])
  (h_frequencies : h.frequencies = [4, 3, 6, 2, 3, 2])
  (h_total : h.total_students = 20) :
  mean h - median h = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_days_missed_difference_l2866_286687


namespace NUMINAMATH_CALUDE_c_wins_probability_l2866_286695

/-- Represents a player in the backgammon tournament -/
inductive Player := | A | B | C

/-- Represents the state of the tournament -/
structure TournamentState where
  lastWinner : Player
  lastLoser : Player

/-- The probability of a player winning a single game -/
def winProbability : ℚ := 1 / 2

/-- The probability of player C winning the tournament -/
def probCWins : ℚ := 2 / 7

/-- Theorem stating that the probability of player C winning the tournament is 2/7 -/
theorem c_wins_probability : 
  probCWins = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_c_wins_probability_l2866_286695


namespace NUMINAMATH_CALUDE_tree_spacing_l2866_286659

/-- Given 8 equally spaced trees along a straight road, where the distance between
    the first and fifth tree is 100 feet, the distance between the first and last
    tree is 175 feet. -/
theorem tree_spacing (n : ℕ) (d : ℝ) (h1 : n = 8) (h2 : d = 100) :
  (n - 1) * d / 4 = 175 :=
by sorry

end NUMINAMATH_CALUDE_tree_spacing_l2866_286659


namespace NUMINAMATH_CALUDE_prob_same_fee_prob_sum_fee_4_prob_sum_fee_6_l2866_286698

/-- Represents the rental time bracket for a bike rental -/
inductive RentalTime
  | WithinTwo
  | TwoToThree
  | ThreeToFour

/-- Calculates the rental fee based on the rental time -/
def rentalFee (time : RentalTime) : ℕ :=
  match time with
  | RentalTime.WithinTwo => 0
  | RentalTime.TwoToThree => 2
  | RentalTime.ThreeToFour => 4

/-- Represents the probability distribution for a person's rental time -/
structure RentalDistribution where
  withinTwo : ℚ
  twoToThree : ℚ
  threeToFour : ℚ
  sum_to_one : withinTwo + twoToThree + threeToFour = 1

/-- The rental distribution for person A -/
def distA : RentalDistribution :=
  { withinTwo := 1/4
  , twoToThree := 1/2
  , threeToFour := 1/4
  , sum_to_one := by norm_num }

/-- The rental distribution for person B -/
def distB : RentalDistribution :=
  { withinTwo := 1/2
  , twoToThree := 1/4
  , threeToFour := 1/4
  , sum_to_one := by norm_num }

/-- Theorem stating the probability that A and B pay the same fee -/
theorem prob_same_fee : 
  distA.withinTwo * distB.withinTwo + 
  distA.twoToThree * distB.twoToThree + 
  distA.threeToFour * distB.threeToFour = 5/16 := by sorry

/-- Theorem stating the probability that the sum of fees is 4 -/
theorem prob_sum_fee_4 :
  distA.withinTwo * distB.threeToFour + 
  distB.withinTwo * distA.threeToFour + 
  distA.twoToThree * distB.twoToThree = 5/16 := by sorry

/-- Theorem stating the probability that the sum of fees is 6 -/
theorem prob_sum_fee_6 :
  distA.twoToThree * distB.threeToFour + 
  distB.twoToThree * distA.threeToFour = 3/16 := by sorry

end NUMINAMATH_CALUDE_prob_same_fee_prob_sum_fee_4_prob_sum_fee_6_l2866_286698


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2866_286638

theorem inequality_solution_set (a : ℝ) (h : (4 : ℝ)^a = 2^(a + 2)) :
  {x : ℝ | a^(2*x + 1) > a^(x - 1)} = {x : ℝ | x > -2} :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2866_286638


namespace NUMINAMATH_CALUDE_binomial_square_constant_l2866_286629

theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 84*x + c = (x + a)^2) → c = 1764 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l2866_286629


namespace NUMINAMATH_CALUDE_f_g_properties_l2866_286628

/-- The absolute value function -/
def f (m : ℝ) (x : ℝ) : ℝ := |x - m|

/-- The function g defined in terms of f -/
def g (m : ℝ) (x : ℝ) : ℝ := 2 * f m x - f m (x + m)

/-- The theorem stating the properties of f and g -/
theorem f_g_properties :
  ∃ (m : ℝ), m > 0 ∧
  (∀ x, g m x ≥ -1) ∧
  (∃ x, g m x = -1) ∧
  m = 1 ∧
  ∀ (a b : ℝ), |a| < m → |b| < m → a ≠ 0 → f m (a * b) > |a| * f m (b / a) :=
sorry

end NUMINAMATH_CALUDE_f_g_properties_l2866_286628


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2866_286608

theorem solve_linear_equation (x : ℝ) : 5 * x + 3 = 10 * x - 17 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2866_286608


namespace NUMINAMATH_CALUDE_divisibility_problem_l2866_286692

/-- A number is a five-digit number if it's between 10000 and 99999 -/
def IsFiveDigit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- A number starts with 4 if its first digit is 4 -/
def StartsWithFour (n : ℕ) : Prop := ∃ k, n = 40000 + k ∧ k < 10000

/-- A number ends with 7 if its last digit is 7 -/
def EndsWithSeven (n : ℕ) : Prop := ∃ k, n = 10 * k + 7

/-- A number starts with 9 if its first digit is 9 -/
def StartsWithNine (n : ℕ) : Prop := ∃ k, n = 90000 + k ∧ k < 10000

/-- A number ends with 3 if its last digit is 3 -/
def EndsWithThree (n : ℕ) : Prop := ∃ k, n = 10 * k + 3

theorem divisibility_problem (x y z : ℕ) 
  (hx_five : IsFiveDigit x) (hy_five : IsFiveDigit y) (hz_five : IsFiveDigit z)
  (hx_start : StartsWithFour x) (hx_end : EndsWithSeven x)
  (hy_start : StartsWithNine y) (hy_end : EndsWithThree y)
  (hxz : z ∣ x) (hyz : z ∣ y) : 
  11 ∣ (2 * y - x) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l2866_286692


namespace NUMINAMATH_CALUDE_waiter_customers_count_waiter_problem_l2866_286636

theorem waiter_customers_count (non_tipping_customers : ℕ) (tip_per_customer : ℕ) (total_tips : ℕ) : ℕ :=
  let tipping_customers := total_tips / tip_per_customer
  non_tipping_customers + tipping_customers

#check waiter_customers_count 5 3 15 = 10

theorem waiter_problem :
  waiter_customers_count 5 3 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_count_waiter_problem_l2866_286636


namespace NUMINAMATH_CALUDE_sales_volume_correct_profit_at_95_max_profit_at_110_max_profit_value_l2866_286632

/-- Represents the weekly sales volume as a function of price -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 1500

/-- Represents the weekly profit as a function of price -/
def profit (x : ℝ) : ℝ := (x - 80) * sales_volume x

/-- The cost price of each shirt -/
def cost_price : ℝ := 80

/-- The minimum allowed selling price -/
def min_price : ℝ := 90

/-- The maximum allowed selling price -/
def max_price : ℝ := 110

theorem sales_volume_correct (x : ℝ) :
  sales_volume x = -10 * x + 1500 :=
sorry

theorem profit_at_95 :
  profit 95 = 8250 :=
sorry

theorem max_profit_at_110 :
  ∀ x, min_price ≤ x ∧ x ≤ max_price → profit x ≤ profit 110 :=
sorry

theorem max_profit_value :
  profit 110 = 12000 :=
sorry

end NUMINAMATH_CALUDE_sales_volume_correct_profit_at_95_max_profit_at_110_max_profit_value_l2866_286632


namespace NUMINAMATH_CALUDE_pencil_count_l2866_286622

theorem pencil_count (pens pencils : ℕ) 
  (h_ratio : pens * 6 = pencils * 5)
  (h_difference : pencils = pens + 4) :
  pencils = 24 := by
sorry

end NUMINAMATH_CALUDE_pencil_count_l2866_286622


namespace NUMINAMATH_CALUDE_find_set_C_l2866_286671

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 6 = 0}

theorem find_set_C : 
  ∃ C : Set ℝ, 
    (C = {0, 2, 3}) ∧ 
    (∀ a : ℝ, a ∈ C ↔ (A ∪ B a = A)) :=
by sorry

end NUMINAMATH_CALUDE_find_set_C_l2866_286671


namespace NUMINAMATH_CALUDE_triangle_determinant_l2866_286691

theorem triangle_determinant (A B C : Real) : 
  A + B + C = π → 
  A ≠ π/2 ∧ B ≠ π/2 ∧ C ≠ π/2 →
  Matrix.det !![Real.tan A, 1, 1; 1, Real.tan B, 1; 1, 1, Real.tan C] = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_determinant_l2866_286691


namespace NUMINAMATH_CALUDE_inverse_square_theorem_l2866_286605

def inverse_square_relation (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ y : ℝ, y ≠ 0 → f y = k / (y * y)

theorem inverse_square_theorem (f : ℝ → ℝ) 
  (h1 : inverse_square_relation f)
  (h2 : ∃ y : ℝ, f y = 1)
  (h3 : f 6 = 0.25) :
  f 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_square_theorem_l2866_286605


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l2866_286627

/-- Predicate for the equation x^2 - mx + n = 0 having two positive roots -/
def has_two_positive_roots (m n : ℝ) : Prop :=
  m^2 - 4*n ≥ 0 ∧ m > 0 ∧ n > 0

/-- Predicate for the curve mx^2 + ny^2 = 1 being an ellipse -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0

theorem neither_sufficient_nor_necessary :
  ¬(∀ m n : ℝ, has_two_positive_roots m n → is_ellipse m n) ∧
  ¬(∀ m n : ℝ, is_ellipse m n → has_two_positive_roots m n) :=
sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l2866_286627


namespace NUMINAMATH_CALUDE_max_ab_bisecting_line_l2866_286647

theorem max_ab_bisecting_line (a b : ℝ) : 
  (∀ x y : ℝ, 2*a*x - b*y + 2 = 0 → (x+1)^2 + (y-2)^2 = 4) → 
  (a * b ≤ 1/4) ∧ (∃ a₀ b₀ : ℝ, a₀ * b₀ = 1/4 ∧ 
    (∀ x y : ℝ, 2*a₀*x - b₀*y + 2 = 0 → (x+1)^2 + (y-2)^2 = 4)) :=
by sorry

end NUMINAMATH_CALUDE_max_ab_bisecting_line_l2866_286647


namespace NUMINAMATH_CALUDE_largest_whole_number_less_than_100_over_7_l2866_286685

theorem largest_whole_number_less_than_100_over_7 : 
  ∀ x : ℕ, x ≤ 14 ↔ 7 * x < 100 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_less_than_100_over_7_l2866_286685


namespace NUMINAMATH_CALUDE_distance_between_trees_l2866_286651

def yard_length : ℝ := 300
def num_trees : ℕ := 26

theorem distance_between_trees :
  let num_intervals : ℕ := num_trees - 1
  let distance : ℝ := yard_length / num_intervals
  distance = 12 := by sorry

end NUMINAMATH_CALUDE_distance_between_trees_l2866_286651


namespace NUMINAMATH_CALUDE_power_multiplication_l2866_286675

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2866_286675


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_l2866_286646

theorem sum_of_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℤ) :
  (∀ x : ℤ, (2*x + 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₂ + a₄ + a₆ = 364 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_l2866_286646


namespace NUMINAMATH_CALUDE_abs_opposite_of_one_l2866_286683

theorem abs_opposite_of_one (x : ℝ) (h : x = -1) : |x| = 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_opposite_of_one_l2866_286683


namespace NUMINAMATH_CALUDE_set_intersection_theorem_l2866_286603

def P : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x : ℝ | x^2 ≥ 4}

theorem set_intersection_theorem :
  P ∩ (Set.univ \ Q) = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_set_intersection_theorem_l2866_286603


namespace NUMINAMATH_CALUDE_product_evaluation_l2866_286652

theorem product_evaluation (n : ℕ) (h : n = 2) :
  (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l2866_286652


namespace NUMINAMATH_CALUDE_partition_positive_integers_l2866_286676

theorem partition_positive_integers : ∃ (A B : Set ℕ), 
  (∀ n : ℕ, n > 0 → (n ∈ A ∨ n ∈ B)) ∧ 
  (A ∩ B = ∅) ∧
  (∀ a b c : ℕ, a ∈ A → b ∈ A → c ∈ A → a < b → b < c → b - a ≠ c - b) ∧
  (∀ f : ℕ → ℕ, (∀ n : ℕ, f n ∈ B) → 
    ∃ i j k : ℕ, i < j ∧ j < k ∧ f j - f i ≠ f k - f j) :=
sorry

end NUMINAMATH_CALUDE_partition_positive_integers_l2866_286676


namespace NUMINAMATH_CALUDE_quadratic_properties_l2866_286601

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Theorem statement
theorem quadratic_properties :
  (∀ x, f x = (x - 2)^2 - 1) ∧
  (∀ x, f x ≥ f 2) ∧
  (f 2 = -1) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, ∀ y ∈ Set.Ioc 2 3, f x > f y) ∧
  (∀ y ∈ Set.Icc (-1 : ℝ) 8, ∃ x ∈ Set.Ico (-1 : ℝ) 3, f x = y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2866_286601


namespace NUMINAMATH_CALUDE_two_white_balls_possible_l2866_286633

/-- Represents the four types of ball replacements --/
inductive Replacement
  | ThreeBlackToOneBlack
  | TwoBlackOneWhiteToOneBlackOneWhite
  | OneBlackTwoWhiteToTwoWhite
  | ThreeWhiteToOneBlackOneWhite

/-- Represents the state of the box --/
structure BoxState :=
  (black : ℕ)
  (white : ℕ)

/-- Applies a single replacement to the box state --/
def applyReplacement (state : BoxState) (r : Replacement) : BoxState :=
  match r with
  | Replacement.ThreeBlackToOneBlack => 
      { black := state.black - 2, white := state.white }
  | Replacement.TwoBlackOneWhiteToOneBlackOneWhite => 
      { black := state.black - 1, white := state.white }
  | Replacement.OneBlackTwoWhiteToTwoWhite => 
      { black := state.black - 1, white := state.white - 1 }
  | Replacement.ThreeWhiteToOneBlackOneWhite => 
      { black := state.black + 1, white := state.white - 2 }

/-- Represents a sequence of replacements --/
def ReplacementSequence := List Replacement

/-- Applies a sequence of replacements to the initial box state --/
def applyReplacements (initial : BoxState) (seq : ReplacementSequence) : BoxState :=
  seq.foldl applyReplacement initial

/-- The theorem to be proved --/
theorem two_white_balls_possible : 
  ∃ (seq : ReplacementSequence), 
    (applyReplacements { black := 100, white := 100 } seq).white = 2 := by
  sorry


end NUMINAMATH_CALUDE_two_white_balls_possible_l2866_286633


namespace NUMINAMATH_CALUDE_min_value_on_circle_l2866_286668

theorem min_value_on_circle :
  ∃ (min : ℝ), min = -5 ∧
  (∀ x y : ℝ, x^2 + y^2 = 1 → 3*x + 4*y ≥ min) ∧
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ 3*x + 4*y = min) := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l2866_286668


namespace NUMINAMATH_CALUDE_cubic_root_equation_solutions_l2866_286661

theorem cubic_root_equation_solutions :
  ∀ x : ℝ, 
    (x^(1/3) - 4 / (x^(1/3) + 4) = 0) ↔ 
    (x = (-2 + 2 * Real.sqrt 2)^3 ∨ x = (-2 - 2 * Real.sqrt 2)^3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solutions_l2866_286661


namespace NUMINAMATH_CALUDE_inequality_proof_l2866_286656

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2) / (2 * a^5 * b^5) + 81 * a^2 * b^2 / 4 + 9 * a * b > 18 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2866_286656


namespace NUMINAMATH_CALUDE_unique_positive_integer_solution_l2866_286642

-- Define the new operation ※
def star_op (a b : ℝ) : ℝ := a * b - a + b - 2

-- Theorem statement
theorem unique_positive_integer_solution :
  ∃! (x : ℕ), x > 0 ∧ star_op 3 (x : ℝ) < 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_integer_solution_l2866_286642


namespace NUMINAMATH_CALUDE_students_in_two_classes_l2866_286672

theorem students_in_two_classes
  (total_students : ℕ)
  (history_students : ℕ)
  (math_students : ℕ)
  (english_students : ℕ)
  (all_three_classes : ℕ)
  (h_total : total_students = 68)
  (h_history : history_students = 19)
  (h_math : math_students = 14)
  (h_english : english_students = 26)
  (h_all_three : all_three_classes = 3)
  (h_at_least_one : total_students = history_students + math_students + english_students
    - (history_students + math_students - all_three_classes
    + history_students + english_students - all_three_classes
    + math_students + english_students - all_three_classes)
    + all_three_classes) :
  history_students + math_students - all_three_classes
  + history_students + english_students - all_three_classes
  + math_students + english_students - all_three_classes
  - 3 * all_three_classes = 6 :=
sorry

end NUMINAMATH_CALUDE_students_in_two_classes_l2866_286672


namespace NUMINAMATH_CALUDE_total_boxes_in_cases_l2866_286664

/-- The number of cases Jenny needs to deliver -/
def num_cases : ℕ := 3

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 8

/-- Theorem: The total number of boxes in the cases is 24 -/
theorem total_boxes_in_cases : num_cases * boxes_per_case = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_boxes_in_cases_l2866_286664


namespace NUMINAMATH_CALUDE_blue_ball_probability_l2866_286667

/-- Represents a container with red and blue balls -/
structure Container where
  red : ℕ
  blue : ℕ

/-- The probability of selecting a blue ball from a container -/
def blueProbability (c : Container) : ℚ :=
  c.blue / (c.red + c.blue)

/-- The containers X, Y, and Z -/
def X : Container := ⟨3, 7⟩
def Y : Container := ⟨5, 5⟩
def Z : Container := ⟨6, 4⟩

/-- The list of all containers -/
def containers : List Container := [X, Y, Z]

/-- The probability of selecting each container -/
def containerProbability : ℚ := 1 / containers.length

/-- The overall probability of selecting a blue ball -/
def overallBlueProbability : ℚ :=
  (containers.map blueProbability).sum / containers.length

theorem blue_ball_probability :
  overallBlueProbability = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_blue_ball_probability_l2866_286667


namespace NUMINAMATH_CALUDE_meeting_time_percentage_l2866_286625

def total_work_day : ℕ := 10 -- in hours
def lunch_break : ℕ := 1 -- in hours
def first_meeting : ℕ := 30 -- in minutes
def second_meeting : ℕ := 3 * first_meeting -- in minutes

def actual_work_minutes : ℕ := (total_work_day - lunch_break) * 60
def total_meeting_minutes : ℕ := first_meeting + second_meeting

def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (actual_work_minutes : ℚ) * 100

theorem meeting_time_percentage : 
  ∃ (ε : ℚ), abs (meeting_percentage - 22) < ε ∧ ε > 0 ∧ ε < 1 :=
sorry

end NUMINAMATH_CALUDE_meeting_time_percentage_l2866_286625


namespace NUMINAMATH_CALUDE_third_group_students_l2866_286682

/-- The number of tissues in each mini tissue box -/
def tissues_per_box : ℕ := 40

/-- The number of students in the first kindergartner group -/
def group1_students : ℕ := 9

/-- The number of students in the second kindergartner group -/
def group2_students : ℕ := 10

/-- The total number of tissues brought by all groups -/
def total_tissues : ℕ := 1200

/-- Theorem stating that the number of students in the third kindergartner group is 11 -/
theorem third_group_students :
  ∃ (x : ℕ), x = 11 ∧ 
  tissues_per_box * (group1_students + group2_students + x) = total_tissues :=
sorry

end NUMINAMATH_CALUDE_third_group_students_l2866_286682


namespace NUMINAMATH_CALUDE_atlanta_equals_boston_l2866_286650

/-- Two cyclists leave Cincinnati at the same time. One bikes to Boston, the other to Atlanta. -/
structure Cyclists where
  boston_distance : ℕ
  atlanta_distance : ℕ
  max_daily_distance : ℕ

/-- The conditions of the cycling problem -/
def cycling_problem (c : Cyclists) : Prop :=
  c.boston_distance = 840 ∧
  c.max_daily_distance = 40 ∧
  (c.boston_distance / c.max_daily_distance) * c.max_daily_distance = c.atlanta_distance

/-- The theorem stating that the distance to Atlanta is equal to the distance to Boston -/
theorem atlanta_equals_boston (c : Cyclists) (h : cycling_problem c) : 
  c.atlanta_distance = c.boston_distance :=
sorry

end NUMINAMATH_CALUDE_atlanta_equals_boston_l2866_286650


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2866_286684

open Set

theorem intersection_of_M_and_N :
  let M : Set ℝ := {x | x > 1}
  let N : Set ℝ := {x | x^2 - 2*x < 0}
  M ∩ N = {x | 1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2866_286684


namespace NUMINAMATH_CALUDE_constant_product_of_reciprocal_inputs_l2866_286648

theorem constant_product_of_reciprocal_inputs (a b : ℝ) (h : a * b ≠ 2) :
  let f : ℝ → ℝ := λ x => (b * x + 1) / (2 * x + a)
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → f x * f (1 / x) = k → k = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_constant_product_of_reciprocal_inputs_l2866_286648


namespace NUMINAMATH_CALUDE_largest_power_of_five_factor_l2866_286620

-- Define factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define the sum of factorials
def sum_of_factorials : ℕ := factorial 77 + factorial 78 + factorial 79

-- Define the function to count factors of 5
def count_factors_of_five (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc x => acc + if x % 5 = 0 then 1 else 0) 0

-- Theorem statement
theorem largest_power_of_five_factor :
  ∃ (n : ℕ), n = 18 ∧ 5^n ∣ sum_of_factorials ∧ ¬(5^(n+1) ∣ sum_of_factorials) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_five_factor_l2866_286620


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l2866_286679

/-- A monotonically decreasing geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 0 < q ∧ q < 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_sum : a 2 + a 3 + a 4 = 28)
  (h_mean : a 3 + 2 = (a 2 + a 4) / 2) :
  (∃ q : ℝ, ∀ n : ℕ, a n = (1/2)^(n - 6)) ∧
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = (1/2) * a n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l2866_286679


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2866_286637

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  equal_side : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.equal_side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * Real.sqrt (4 * (t.equal_side : ℝ)^2 - (t.base : ℝ)^2) / 4

/-- Theorem statement -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t1.base = 6 * t2.base ∧
    perimeter t1 = 399 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s1.base = 6 * s2.base →
      perimeter s1 ≥ 399) :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2866_286637


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_one_l2866_286600

/-- A polynomial in two variables that represents a^2005 + b^2005 -/
def P : (ℝ → ℝ → ℝ) → Prop :=
  λ p => ∀ a b : ℝ, p (a + b) (a * b) = a^2005 + b^2005

/-- The sum of coefficients of a polynomial in two variables -/
def sum_of_coefficients (p : ℝ → ℝ → ℝ) : ℝ := p 1 1

theorem sum_of_coefficients_is_one (p : ℝ → ℝ → ℝ) (h : P p) : 
  sum_of_coefficients p = 1 := by
  sorry

#check sum_of_coefficients_is_one

end NUMINAMATH_CALUDE_sum_of_coefficients_is_one_l2866_286600


namespace NUMINAMATH_CALUDE_triangle_count_is_48_l2866_286673

/-- Represents the configuration of the rectangle and its divisions -/
structure RectangleConfig where
  vertical_divisions : Nat
  horizontal_divisions : Nat
  additional_horizontal_divisions : Nat

/-- Calculates the number of triangles in the described figure -/
def count_triangles (config : RectangleConfig) : Nat :=
  let initial_rectangles := config.vertical_divisions * config.horizontal_divisions
  let initial_triangles := 2 * initial_rectangles
  let additional_rectangles := initial_rectangles * config.additional_horizontal_divisions
  let additional_triangles := 2 * additional_rectangles
  initial_triangles + additional_triangles

/-- The specific configuration described in the problem -/
def problem_config : RectangleConfig :=
  { vertical_divisions := 3
  , horizontal_divisions := 2
  , additional_horizontal_divisions := 2 }

/-- Theorem stating that the number of triangles in the described figure is 48 -/
theorem triangle_count_is_48 : count_triangles problem_config = 48 := by
  sorry


end NUMINAMATH_CALUDE_triangle_count_is_48_l2866_286673


namespace NUMINAMATH_CALUDE_equation_solution_l2866_286657

theorem equation_solution (x : ℝ) : 
  Real.sqrt (x + 15) - 9 / Real.sqrt (x + 15) = 3 → x = 18 * Real.sqrt 5 / 4 - 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2866_286657


namespace NUMINAMATH_CALUDE_fourth_number_is_eight_l2866_286649

/-- Given four numbers with an arithmetic mean of 20, where three of the numbers are 12, 24, and 36,
    and the fourth number is the square of another number, prove that the fourth number is 8. -/
theorem fourth_number_is_eight (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 20 →
  a = 12 →
  b = 24 →
  c = 36 →
  ∃ x, d = x^2 →
  d = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_is_eight_l2866_286649


namespace NUMINAMATH_CALUDE_custom_mul_ab_equals_nine_l2866_286660

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) (x y : ℝ) : ℝ := a * x + b * y - 1

/-- Theorem stating that under given conditions, a*b = 9 -/
theorem custom_mul_ab_equals_nine
  (a b : ℝ)
  (h1 : custom_mul a b 1 2 = 4)
  (h2 : custom_mul a b (-2) 3 = 10) :
  custom_mul a b a b = 9 :=
sorry

end NUMINAMATH_CALUDE_custom_mul_ab_equals_nine_l2866_286660


namespace NUMINAMATH_CALUDE_product_of_integers_l2866_286604

theorem product_of_integers (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 30)
  (h3 : 1 / p + 1 / q + 1 / r + 420 / (p * q * r) = 1) :
  p * q * r = 576 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l2866_286604


namespace NUMINAMATH_CALUDE_binomial_divisibility_implies_prime_l2866_286655

theorem binomial_divisibility_implies_prime (n : ℕ) (h : ∀ k : ℕ, 1 ≤ k → k < n → (n.choose k) % n = 0) : Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_binomial_divisibility_implies_prime_l2866_286655


namespace NUMINAMATH_CALUDE_fraction_equality_l2866_286686

theorem fraction_equality (a b c d : ℝ) (h : a / b = c / d) :
  (a * b) / (c * d) = ((a + b) / (c + d))^2 ∧ (a * b) / (c * d) = ((a - b) / (c - d))^2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2866_286686


namespace NUMINAMATH_CALUDE_max_distance_complex_l2866_286694

theorem max_distance_complex (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (max_dist : ℝ), max_dist = (36 * Real.sqrt 26) / 5 ∧
  ∀ w : ℂ, Complex.abs w = 3 → Complex.abs ((2 + Complex.I) * w^2 - w^4) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_complex_l2866_286694


namespace NUMINAMATH_CALUDE_fran_average_speed_l2866_286630

/-- Proves that given Joann's average speed and time, and Fran's riding time,
    Fran's required average speed to travel the same distance as Joann is 14 mph. -/
theorem fran_average_speed 
  (joann_speed : ℝ) 
  (joann_time : ℝ) 
  (fran_time : ℝ) 
  (h1 : joann_speed = 16) 
  (h2 : joann_time = 3.5) 
  (h3 : fran_time = 4) : 
  (joann_speed * joann_time) / fran_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_fran_average_speed_l2866_286630


namespace NUMINAMATH_CALUDE_yushu_donations_l2866_286663

/-- The number of matching combinations for backpacks and pencil cases -/
def matching_combinations (backpack_styles : ℕ) (pencil_case_styles : ℕ) : ℕ :=
  backpack_styles * pencil_case_styles

/-- Theorem: Given 2 backpack styles and 2 pencil case styles, there are 4 matching combinations -/
theorem yushu_donations : matching_combinations 2 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_yushu_donations_l2866_286663
