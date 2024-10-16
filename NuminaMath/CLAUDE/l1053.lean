import Mathlib

namespace NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l1053_105376

-- Problem 1
theorem calculation_proof (a b : ℝ) (h : a ≠ b ∧ a ≠ -b ∧ a ≠ 0) :
  (a - b) / (a + b) - (a^2 - 2*a*b + b^2) / (a^2 - b^2) / ((a - b) / a) = -b / (a + b) := by
  sorry

-- Problem 2
theorem inequality_system_solution (x : ℝ) :
  (x - 3*(x - 2) ≥ 4 ∧ (2*x - 1) / 5 > (x + 1) / 2) ↔ x < -7 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_inequality_system_solution_l1053_105376


namespace NUMINAMATH_CALUDE_solution_set_nonempty_implies_m_greater_than_five_l1053_105334

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 1|
def g (x m : ℝ) : ℝ := -|x + 4| + m

-- State the theorem
theorem solution_set_nonempty_implies_m_greater_than_five :
  (∃ x : ℝ, f x < g x m) → m > 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_nonempty_implies_m_greater_than_five_l1053_105334


namespace NUMINAMATH_CALUDE_counterexample_exists_l1053_105383

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def given_numbers : List ℕ := [6, 9, 10, 11, 15]

theorem counterexample_exists : ∃ n : ℕ, 
  n ∈ given_numbers ∧
  ¬(is_prime n) ∧ 
  is_prime (n - 2) ∧ 
  is_prime (n + 2) :=
sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1053_105383


namespace NUMINAMATH_CALUDE_taxi_charge_theorem_l1053_105370

/-- Calculates the total charge for a taxi trip given the initial fee, rate per increment, increment distance, and total distance. -/
def totalCharge (initialFee : ℚ) (ratePerIncrement : ℚ) (incrementDistance : ℚ) (totalDistance : ℚ) : ℚ :=
  initialFee + (totalDistance / incrementDistance).floor * ratePerIncrement

/-- Theorem stating that the total charge for a 3.6-mile trip with given fee structure is $3.60 -/
theorem taxi_charge_theorem :
  let initialFee : ℚ := 225/100
  let ratePerIncrement : ℚ := 15/100
  let incrementDistance : ℚ := 2/5
  let totalDistance : ℚ := 36/10
  totalCharge initialFee ratePerIncrement incrementDistance totalDistance = 360/100 := by
  sorry


end NUMINAMATH_CALUDE_taxi_charge_theorem_l1053_105370


namespace NUMINAMATH_CALUDE_highest_temperature_correct_l1053_105332

/-- The highest temperature reached during candy making --/
def highest_temperature (initial_temp final_temp : ℝ) (heating_rate cooling_rate : ℝ) (total_time : ℝ) : ℝ :=
  let T : ℝ := 240
  T

/-- Theorem stating that the highest temperature is correct --/
theorem highest_temperature_correct 
  (initial_temp : ℝ) (final_temp : ℝ) (heating_rate : ℝ) (cooling_rate : ℝ) (total_time : ℝ)
  (h1 : initial_temp = 60)
  (h2 : final_temp = 170)
  (h3 : heating_rate = 5)
  (h4 : cooling_rate = 7)
  (h5 : total_time = 46) :
  let T := highest_temperature initial_temp final_temp heating_rate cooling_rate total_time
  (T - initial_temp) / heating_rate + (T - final_temp) / cooling_rate = total_time :=
by
  sorry

#check highest_temperature_correct

end NUMINAMATH_CALUDE_highest_temperature_correct_l1053_105332


namespace NUMINAMATH_CALUDE_tan_product_seventh_pi_l1053_105378

theorem tan_product_seventh_pi : 
  Real.tan (π / 7) * Real.tan (2 * π / 7) * Real.tan (3 * π / 7) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_seventh_pi_l1053_105378


namespace NUMINAMATH_CALUDE_slope_product_sufficient_not_necessary_l1053_105314

/-- A line in a 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Two lines are perpendicular -/
def perpendicular (l₁ l₂ : Line) : Prop :=
  sorry

/-- The product of slopes of two lines is -1 -/
def slope_product_negative_one (l₁ l₂ : Line) : Prop :=
  l₁.slope * l₂.slope = -1

/-- The product of slopes being -1 is sufficient but not necessary for perpendicularity -/
theorem slope_product_sufficient_not_necessary :
  (∀ l₁ l₂ : Line, slope_product_negative_one l₁ l₂ → perpendicular l₁ l₂) ∧
  ¬(∀ l₁ l₂ : Line, perpendicular l₁ l₂ → slope_product_negative_one l₁ l₂) :=
sorry

end NUMINAMATH_CALUDE_slope_product_sufficient_not_necessary_l1053_105314


namespace NUMINAMATH_CALUDE_circumcircles_intersect_at_single_point_l1053_105333

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a triangle
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

-- Define a circle
structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define central symmetry
def centrally_symmetric (t1 t2 : Triangle) (center : Point) : Prop :=
  ∃ (O : Point),
    (t1.A.x + t2.A.x) / 2 = O.x ∧ (t1.A.y + t2.A.y) / 2 = O.y ∧
    (t1.B.x + t2.B.x) / 2 = O.x ∧ (t1.B.y + t2.B.y) / 2 = O.y ∧
    (t1.C.x + t2.C.x) / 2 = O.x ∧ (t1.C.y + t2.C.y) / 2 = O.y

-- Define circumcircle
def circumcircle (t : Triangle) : Circle :=
  sorry

-- Define intersection of circles
def intersect (c1 c2 : Circle) : Set Point :=
  sorry

theorem circumcircles_intersect_at_single_point
  (ABC A₁B₁C₁ : Triangle)
  (h : centrally_symmetric ABC A₁B₁C₁ (Point.mk 0 0)) :
  ∃ (S : Point),
    S ∈ intersect (circumcircle ABC) (circumcircle (Triangle.mk A₁B₁C₁.A ABC.B A₁B₁C₁.C)) ∧
    S ∈ intersect (circumcircle (Triangle.mk A₁B₁C₁.A A₁B₁C₁.B ABC.C)) (circumcircle (Triangle.mk ABC.A A₁B₁C₁.B A₁B₁C₁.C)) :=
sorry

end NUMINAMATH_CALUDE_circumcircles_intersect_at_single_point_l1053_105333


namespace NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l1053_105390

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nthNumberWithDigitSum13 (n : ℕ+) : ℕ+ := sorry

/-- The theorem stating that the 11th number with digit sum 13 is 166 -/
theorem eleventh_number_with_digit_sum_13 : 
  nthNumberWithDigitSum13 11 = 166 := by sorry

end NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l1053_105390


namespace NUMINAMATH_CALUDE_percent_relation_l1053_105305

theorem percent_relation (x y z : ℝ) 
  (h1 : x = 1.3 * y) 
  (h2 : y = 0.5 * z) : 
  x = 0.65 * z := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l1053_105305


namespace NUMINAMATH_CALUDE_line_graph_best_for_daily_income_fluctuations_l1053_105387

-- Define the types of statistical graphs
inductive StatGraph
| LineGraph
| BarGraph
| PieChart
| Histogram

-- Define a structure for daily income data
structure DailyIncomeData :=
  (days : Fin 7 → ℝ)

-- Define a property for showing fluctuations intuitively
def shows_fluctuations_intuitively (graph : StatGraph) : Prop :=
  match graph with
  | StatGraph.LineGraph => true
  | _ => false

-- Define the theorem
theorem line_graph_best_for_daily_income_fluctuations 
  (data : DailyIncomeData) :
  ∃ (best : StatGraph), 
    (shows_fluctuations_intuitively best ∧ 
     ∀ (g : StatGraph), shows_fluctuations_intuitively g → g = best) :=
sorry

end NUMINAMATH_CALUDE_line_graph_best_for_daily_income_fluctuations_l1053_105387


namespace NUMINAMATH_CALUDE_symmetric_distribution_theorem_l1053_105319

/-- A symmetric distribution with mean m and standard deviation d. -/
structure SymmetricDistribution where
  m : ℝ  -- mean
  d : ℝ  -- standard deviation
  symmetric : Bool
  within_one_std_dev : ℝ

/-- The percentage of the distribution less than m + d -/
def percent_less_than_m_plus_d (dist : SymmetricDistribution) : ℝ := sorry

theorem symmetric_distribution_theorem (dist : SymmetricDistribution) 
  (h_symmetric : dist.symmetric = true)
  (h_within_one_std_dev : dist.within_one_std_dev = 84) :
  percent_less_than_m_plus_d dist = 42 := by sorry

end NUMINAMATH_CALUDE_symmetric_distribution_theorem_l1053_105319


namespace NUMINAMATH_CALUDE_car_speed_proof_l1053_105396

/-- The speed of the first car in miles per hour -/
def speed1 : ℝ := 52

/-- The time traveled in hours -/
def time : ℝ := 3.5

/-- The total distance between the cars after the given time in miles -/
def total_distance : ℝ := 385

/-- The speed of the second car in miles per hour -/
def speed2 : ℝ := 58

theorem car_speed_proof :
  speed1 * time + speed2 * time = total_distance :=
by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l1053_105396


namespace NUMINAMATH_CALUDE_sandy_puppies_l1053_105302

def total_puppies (initial : ℝ) (additional : ℝ) : ℝ :=
  initial + additional

theorem sandy_puppies : total_puppies 8 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sandy_puppies_l1053_105302


namespace NUMINAMATH_CALUDE_ancient_tower_height_l1053_105308

/-- Proves that the height of an ancient tower is 14.4 meters given the conditions of Xiao Liang's height and shadow length, and the tower's shadow length. -/
theorem ancient_tower_height 
  (xiao_height : ℝ) 
  (xiao_shadow : ℝ) 
  (tower_shadow : ℝ) 
  (h1 : xiao_height = 1.6)
  (h2 : xiao_shadow = 2)
  (h3 : tower_shadow = 18) :
  (xiao_height / xiao_shadow) * tower_shadow = 14.4 :=
by sorry

end NUMINAMATH_CALUDE_ancient_tower_height_l1053_105308


namespace NUMINAMATH_CALUDE_mans_swimming_speed_l1053_105307

/-- The speed of a man in still water given his downstream and upstream swimming times and distances -/
theorem mans_swimming_speed (downstream_distance upstream_distance : ℝ) 
  (downstream_time upstream_time : ℝ) (h1 : downstream_distance = 51) 
  (h2 : upstream_distance = 18) (h3 : downstream_time = 3) (h4 : upstream_time = 3) :
  ∃ (v_m : ℝ), v_m = 11.5 ∧ 
    (downstream_distance / downstream_time + upstream_distance / upstream_time) / 2 = v_m :=
by sorry

end NUMINAMATH_CALUDE_mans_swimming_speed_l1053_105307


namespace NUMINAMATH_CALUDE_line_circle_intersection_k_range_l1053_105331

theorem line_circle_intersection_k_range 
  (k : ℝ) 
  (line : ℝ → ℝ → Prop)
  (circle : ℝ → ℝ → Prop)
  (M N : ℝ × ℝ) :
  (∀ x y, line x y ↔ y = k * x + 3) →
  (∀ x y, circle x y ↔ (x - 3)^2 + (y - 2)^2 = 4) →
  (line M.1 M.2 ∧ circle M.1 M.2) →
  (line N.1 N.2 ∧ circle N.1 N.2) →
  ((M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 12) →
  -3/4 ≤ k ∧ k ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_k_range_l1053_105331


namespace NUMINAMATH_CALUDE_second_difference_quadratic_l1053_105322

theorem second_difference_quadratic 
  (f : ℕ → ℝ) 
  (h : ∀ n : ℕ, f (n + 2) - 2 * f (n + 1) + f n = 1) : 
  ∃ a b : ℝ, ∀ n : ℕ, f n = (1/2) * n^2 + a * n + b := by
sorry

end NUMINAMATH_CALUDE_second_difference_quadratic_l1053_105322


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1053_105318

theorem quadratic_equation_properties :
  ∃ (p q : ℝ),
    (∀ (r : ℝ), (∀ (x : ℝ), x^2 - (r+7)*x + r + 87 = 0 →
      (∃ (y : ℝ), y ≠ x ∧ y^2 - (r+7)*y + r + 87 = 0) ∧
      x < 0) ↔ p < r ∧ r < q) ∧
    p^2 + q^2 = 8098 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1053_105318


namespace NUMINAMATH_CALUDE_crayons_left_l1053_105350

def initial_crayons : ℕ := 120

def kiley_fraction : ℚ := 3/8

def joe_fraction : ℚ := 5/9

theorem crayons_left : 
  let remaining_after_kiley := initial_crayons - (kiley_fraction * initial_crayons).floor
  let final_remaining := remaining_after_kiley - (joe_fraction * remaining_after_kiley).floor
  final_remaining = 33 := by sorry

end NUMINAMATH_CALUDE_crayons_left_l1053_105350


namespace NUMINAMATH_CALUDE_monotonic_exp_minus_mx_l1053_105364

/-- If f(x) = e^x - mx is monotonically increasing on [0, +∞), then m ≤ 1 -/
theorem monotonic_exp_minus_mx (m : ℝ) :
  (∀ x : ℝ, x ≥ 0 → Monotone (fun x : ℝ ↦ Real.exp x - m * x)) →
  m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_exp_minus_mx_l1053_105364


namespace NUMINAMATH_CALUDE_student_base_choices_l1053_105391

/-- The number of bases available for students to choose from -/
def num_bases : ℕ := 4

/-- The number of students choosing bases -/
def num_students : ℕ := 4

/-- The total number of ways for students to choose bases -/
def total_ways : ℕ := num_bases ^ num_students

theorem student_base_choices : total_ways = 256 := by
  sorry

end NUMINAMATH_CALUDE_student_base_choices_l1053_105391


namespace NUMINAMATH_CALUDE_instructors_next_meeting_l1053_105316

theorem instructors_next_meeting (f g h i j : ℕ) 
  (hf : f = 5) (hg : g = 3) (hh : h = 9) (hi : i = 2) (hj : j = 8) :
  Nat.lcm f (Nat.lcm g (Nat.lcm h (Nat.lcm i j))) = 360 :=
by sorry

end NUMINAMATH_CALUDE_instructors_next_meeting_l1053_105316


namespace NUMINAMATH_CALUDE_coin_denominations_exist_l1053_105373

theorem coin_denominations_exist : ∃ (S : Finset ℕ), 
  (Finset.card S = 12) ∧ 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 6543 → 
    ∃ (T : Finset ℕ), 
      (∀ m ∈ T, m ∈ S) ∧ 
      (Finset.card T ≤ 8) ∧ 
      (Finset.sum T id = n)) :=
sorry

end NUMINAMATH_CALUDE_coin_denominations_exist_l1053_105373


namespace NUMINAMATH_CALUDE_susan_cloth_bags_l1053_105363

/-- Calculates the number of cloth bags Susan brought to carry peaches. -/
def number_of_cloth_bags (total_peaches knapsack_peaches : ℕ) : ℕ :=
  let cloth_bag_peaches := 2 * knapsack_peaches
  (total_peaches - knapsack_peaches) / cloth_bag_peaches

/-- Proves that Susan brought 2 cloth bags given the problem conditions. -/
theorem susan_cloth_bags :
  number_of_cloth_bags (5 * 12) 12 = 2 := by
  sorry

end NUMINAMATH_CALUDE_susan_cloth_bags_l1053_105363


namespace NUMINAMATH_CALUDE_survey_result_l1053_105356

/-- The number of households that used neither brand E nor brand B soap -/
def neither : ℕ := 80

/-- The number of households that used only brand E soap -/
def only_E : ℕ := 60

/-- The number of households that used both brands of soap -/
def both : ℕ := 40

/-- The ratio of households that used only brand B soap to those that used both brands -/
def B_to_both_ratio : ℕ := 3

/-- The total number of households surveyed -/
def total_households : ℕ := neither + only_E + both + B_to_both_ratio * both

theorem survey_result : total_households = 300 := by
  sorry

end NUMINAMATH_CALUDE_survey_result_l1053_105356


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l1053_105346

def p (x : ℝ) : ℝ := 3 * (x^2 - 3*x + 3) - 5 * (x^3 - 2*x^2 + 4*x - 1)

theorem sum_of_squares_of_coefficients :
  ∃ (a b c d : ℝ),
    (∀ x, p x = a * x^3 + b * x^2 + c * x + d) ∧
    a^2 + b^2 + c^2 + d^2 = 1231 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l1053_105346


namespace NUMINAMATH_CALUDE_exactly_seven_numbers_satisfy_condition_l1053_105329

/-- A two-digit number is a natural number between 10 and 99 inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The tens digit of a natural number. -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- The units digit of a natural number. -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sum of digits of a two-digit number. -/
def digitSum (n : ℕ) : ℕ := tensDigit n + unitsDigit n

/-- The condition specified in the problem. -/
def satisfiesCondition (n : ℕ) : Prop :=
  TwoDigitNumber n ∧ unitsDigit (n - 2 * digitSum n) = 4

/-- The main theorem stating that exactly 7 two-digit numbers satisfy the condition. -/
theorem exactly_seven_numbers_satisfy_condition :
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfiesCondition n) ∧ s.card = 7 := by
  sorry

end NUMINAMATH_CALUDE_exactly_seven_numbers_satisfy_condition_l1053_105329


namespace NUMINAMATH_CALUDE_g_4_cubed_eq_16_l1053_105362

/-- Given two functions f and g satisfying certain conditions, prove that [g(4)]^3 = 16 -/
theorem g_4_cubed_eq_16
  (f g : ℝ → ℝ)
  (h1 : ∀ x ≥ 1, f (g x) = x^2)
  (h2 : ∀ x ≥ 1, g (f x) = x^3)
  (h3 : g 16 = 16) :
  (g 4)^3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_g_4_cubed_eq_16_l1053_105362


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l1053_105384

/-- Given a cylinder formed by rotating a square around one of its sides,
    if the volume of the cylinder is 27π cm³,
    then its lateral surface area is 18π cm². -/
theorem cylinder_lateral_surface_area 
  (side : ℝ) 
  (h_cylinder : side > 0) 
  (h_volume : π * side^2 * side = 27 * π) : 
  2 * π * side * side = 18 * π :=
sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l1053_105384


namespace NUMINAMATH_CALUDE_bucket_water_difference_l1053_105345

/-- Given two buckets with initial volumes and a water transfer between them,
    prove the resulting volume difference. -/
theorem bucket_water_difference 
  (large_initial small_initial transfer : ℕ)
  (h1 : large_initial = 7)
  (h2 : small_initial = 5)
  (h3 : transfer = 2)
  : large_initial + transfer - (small_initial - transfer) = 6 := by
  sorry

end NUMINAMATH_CALUDE_bucket_water_difference_l1053_105345


namespace NUMINAMATH_CALUDE_divisible_by_fifteen_l1053_105325

theorem divisible_by_fifteen (a : ℤ) : ∃ k : ℤ, 9 * a^5 - 5 * a^3 - 4 * a = 15 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_fifteen_l1053_105325


namespace NUMINAMATH_CALUDE_hexagon_walk_distance_hexagon_walk_distance_proof_l1053_105321

/-- The distance of a point from its starting position after moving 7 km along the perimeter of a regular hexagon with side length 3 km -/
theorem hexagon_walk_distance : ℝ :=
  let side_length : ℝ := 3
  let walk_distance : ℝ := 7
  let hexagon_angle : ℝ := 2 * Real.pi / 6
  let end_position : ℝ × ℝ := (1, Real.sqrt 3)
  2

theorem hexagon_walk_distance_proof :
  hexagon_walk_distance = 2 := by sorry

end NUMINAMATH_CALUDE_hexagon_walk_distance_hexagon_walk_distance_proof_l1053_105321


namespace NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l1053_105374

/-- Proves that for a rectangular plot with breadth 8 meters and length 10 meters more than its breadth, the ratio of its area to its breadth is 18:1. -/
theorem rectangle_area_breadth_ratio :
  let b : ℝ := 8  -- breadth
  let l : ℝ := b + 10  -- length
  let A : ℝ := l * b  -- area
  A / b = 18 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_breadth_ratio_l1053_105374


namespace NUMINAMATH_CALUDE_tiles_difference_7_6_l1053_105351

/-- The number of tiles in the n-th square of the sequence -/
def tiles_in_square (n : ℕ) : ℕ := n ^ 2

/-- The theorem stating the difference in tiles between the 7th and 6th squares -/
theorem tiles_difference_7_6 : tiles_in_square 7 - tiles_in_square 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_tiles_difference_7_6_l1053_105351


namespace NUMINAMATH_CALUDE_suitable_squares_are_1_4_9_49_l1053_105306

/-- The sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A number is suitable if it's the smallest among all natural numbers with the same digit sum -/
def is_suitable (n : ℕ) : Prop :=
  ∀ m : ℕ, digit_sum m = digit_sum n → n ≤ m

/-- The set of all suitable square numbers -/
def suitable_squares : Set ℕ :=
  {n : ℕ | is_suitable n ∧ ∃ k : ℕ, n = k^2}

/-- Theorem: The set of suitable square numbers is exactly {1, 4, 9, 49} -/
theorem suitable_squares_are_1_4_9_49 : suitable_squares = {1, 4, 9, 49} := by sorry

end NUMINAMATH_CALUDE_suitable_squares_are_1_4_9_49_l1053_105306


namespace NUMINAMATH_CALUDE_monotonicity_intervals_min_value_l1053_105358

-- Define the function f
def f (x a : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

-- Theorem for monotonicity intervals
theorem monotonicity_intervals (a : ℝ) :
  (∀ x < -1, (f' x) < 0) ∧
  (∀ x > 3, (f' x) < 0) ∧
  (∀ x ∈ Set.Ioo (-1) 3, (f' x) > 0) :=
sorry

-- Theorem for minimum value
theorem min_value (a : ℝ) :
  (∃ x ∈ Set.Icc (-2) 2, f x a = 20) →
  (∃ y ∈ Set.Icc (-2) 2, f y a = -7 ∧ ∀ z ∈ Set.Icc (-2) 2, f z a ≥ -7) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_intervals_min_value_l1053_105358


namespace NUMINAMATH_CALUDE_age_sum_theorem_l1053_105371

def mother_age : ℕ := 40

def daughter_age (m : ℕ) : ℕ := (70 - m) / 2

theorem age_sum_theorem (m : ℕ) (h : m = mother_age) : 
  2 * m + daughter_age m = 95 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_theorem_l1053_105371


namespace NUMINAMATH_CALUDE_article_cost_l1053_105357

/-- The cost of an article given specific selling conditions --/
theorem article_cost : ∃ (C : ℝ), 
  (450 - C = 1.1 * (380 - C)) ∧ 
  (C > 0) ∧ 
  (C = 320) := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l1053_105357


namespace NUMINAMATH_CALUDE_fish_population_calculation_l1053_105347

/-- Calculates the number of fish in a pond on April 1 based on sampling data --/
theorem fish_population_calculation (tagged_april : ℕ) (sample_august : ℕ) (tagged_in_sample : ℕ)
  (death_rate : ℚ) (birth_rate : ℚ) :
  tagged_april = 80 →
  sample_august = 100 →
  tagged_in_sample = 4 →
  death_rate = 30 / 100 →
  birth_rate = 50 / 100 →
  ∃ (total_fish : ℕ), total_fish = 1000 := by
  sorry

#check fish_population_calculation

end NUMINAMATH_CALUDE_fish_population_calculation_l1053_105347


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l1053_105348

theorem right_triangle_third_side_product (a b c d : ℝ) : 
  a = 6 → b = 8 → 
  ((a^2 + b^2 = c^2) ∨ (a^2 + d^2 = b^2)) →
  c * d = 20 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l1053_105348


namespace NUMINAMATH_CALUDE_courtyard_length_l1053_105311

theorem courtyard_length (stone_length : ℝ) (stone_width : ℝ) (courtyard_width : ℝ) (total_stones : ℕ) :
  stone_length = 2.5 →
  stone_width = 2 →
  courtyard_width = 16.5 →
  total_stones = 198 →
  ∃ courtyard_length : ℝ, courtyard_length = 60 ∧ 
    courtyard_length * courtyard_width = (stone_length * stone_width) * total_stones :=
by
  sorry

end NUMINAMATH_CALUDE_courtyard_length_l1053_105311


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1053_105320

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x ∈ Set.Ioo 0 2, P x) ↔ (∀ x ∈ Set.Ioo 0 2, ¬P x) :=
by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x ∈ Set.Ioo 0 2, x^2 + 2*x + 2 ≤ 0) ↔
  (∀ x ∈ Set.Ioo 0 2, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l1053_105320


namespace NUMINAMATH_CALUDE_solve_linear_system_l1053_105380

/-- Given a system of linear equations:
     a + b = c
     b + c = 7
     c - a = 2
    Prove that b = 2 -/
theorem solve_linear_system (a b c : ℤ) 
  (eq1 : a + b = c) 
  (eq2 : b + c = 7) 
  (eq3 : c - a = 2) : 
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_system_l1053_105380


namespace NUMINAMATH_CALUDE_medicine_supply_duration_l1053_105388

/-- Represents the number of pills in a bottle -/
def pills_per_bottle : ℕ := 90

/-- Represents the fraction of a pill taken per dose -/
def fraction_per_dose : ℚ := 1/3

/-- Represents the number of days between doses -/
def days_between_doses : ℕ := 3

/-- Represents the average number of days in a month -/
def days_per_month : ℕ := 30

/-- Proves that the supply of medicine lasts 27 months -/
theorem medicine_supply_duration :
  (pills_per_bottle : ℚ) * days_between_doses / fraction_per_dose / days_per_month = 27 := by
  sorry


end NUMINAMATH_CALUDE_medicine_supply_duration_l1053_105388


namespace NUMINAMATH_CALUDE_school_population_proof_l1053_105353

theorem school_population_proof (x : ℝ) (h1 : 162 = (x / 100) * (0.5 * x)) : x = 180 := by
  sorry

end NUMINAMATH_CALUDE_school_population_proof_l1053_105353


namespace NUMINAMATH_CALUDE_difference_of_numbers_l1053_105385

theorem difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares : x^2 - y^2 = 80) : 
  x - y = 8 := by
sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l1053_105385


namespace NUMINAMATH_CALUDE_grade_swap_possible_l1053_105394

/-- Represents a grade scaling system -/
structure GradeScale where
  upper_limit : ℕ
  round_up_half : Set ℕ

/-- Represents a grade within a scaling system -/
def Grade := { g : ℚ // 0 < g ∧ g < 1 }

/-- Function to rescale a grade -/
def rescale (g : Grade) (old_scale new_scale : GradeScale) : Grade :=
  sorry

/-- Theorem stating that any two grades can be swapped through a series of rescalings -/
theorem grade_swap_possible (a b : Grade) :
  ∃ (scales : List GradeScale), 
    let final_scale := scales.foldl (λ acc s => s) { upper_limit := 100, round_up_half := ∅ }
    let new_a := scales.foldl (λ acc s => rescale acc s final_scale) a
    let new_b := scales.foldl (λ acc s => rescale acc s final_scale) b
    new_a = b ∧ new_b = a :=
  sorry

end NUMINAMATH_CALUDE_grade_swap_possible_l1053_105394


namespace NUMINAMATH_CALUDE_prob_sum_seven_or_eleven_l1053_105354

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

/-- The number of ways to roll a sum of 7 -/
def waysToRollSeven : ℕ := 6

/-- The number of ways to roll a sum of 11 -/
def waysToRollEleven : ℕ := 2

/-- The total number of favorable outcomes (sum of 7 or 11) -/
def favorableOutcomes : ℕ := waysToRollSeven + waysToRollEleven

/-- The probability of rolling a sum of 7 or 11 with two fair six-sided dice -/
theorem prob_sum_seven_or_eleven : 
  (favorableOutcomes : ℚ) / totalOutcomes = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_seven_or_eleven_l1053_105354


namespace NUMINAMATH_CALUDE_zeros_properties_l1053_105379

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + Real.log x - 3

theorem zeros_properties (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : f a x₁ = 0) 
  (h₂ : f a x₂ = 0) 
  (h₃ : x₁ < x₂) : 
  (0 < a ∧ a < Real.exp 2) ∧ x₁ + x₂ > 2 * a := by
  sorry

end NUMINAMATH_CALUDE_zeros_properties_l1053_105379


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1053_105303

theorem complex_fraction_simplification :
  (1 + 2*Complex.I) / (3 - 4*Complex.I) = -1/5 + (2/5)*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1053_105303


namespace NUMINAMATH_CALUDE_ellipse_a_plus_k_equals_nine_l1053_105398

/-- An ellipse with given properties -/
structure Ellipse where
  foci1 : ℝ × ℝ
  foci2 : ℝ × ℝ
  point : ℝ × ℝ
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The ellipse satisfies the given conditions -/
def satisfies_conditions (e : Ellipse) : Prop :=
  e.foci1 = (1, 5) ∧
  e.foci2 = (1, 1) ∧
  e.point = (7, 3) ∧
  e.a > 0 ∧
  e.b > 0 ∧
  (e.point.1 - e.h)^2 / e.a^2 + (e.point.2 - e.k)^2 / e.b^2 = 1

/-- The theorem stating that a + k equals 9 for the given ellipse -/
theorem ellipse_a_plus_k_equals_nine (e : Ellipse) 
  (h : satisfies_conditions e) : e.a + e.k = 9 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_a_plus_k_equals_nine_l1053_105398


namespace NUMINAMATH_CALUDE_range_of_expression_l1053_105336

theorem range_of_expression (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : a > 0) (h3 : b > 0) :
  1 / a + a / b ≥ 1 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l1053_105336


namespace NUMINAMATH_CALUDE_complement_of_A_l1053_105367

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2}

theorem complement_of_A : Aᶜ = {3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l1053_105367


namespace NUMINAMATH_CALUDE_second_outlet_rate_calculation_l1053_105355

/-- Represents the rate of the second outlet pipe in cubic inches per minute -/
def second_outlet_rate : ℝ := 9

/-- Tank volume in cubic feet -/
def tank_volume : ℝ := 30

/-- Inlet pipe rate in cubic inches per minute -/
def inlet_rate : ℝ := 3

/-- First outlet pipe rate in cubic inches per minute -/
def first_outlet_rate : ℝ := 6

/-- Time to empty the tank when all pipes are open, in minutes -/
def emptying_time : ℝ := 4320

/-- Conversion factor from cubic feet to cubic inches -/
def cubic_feet_to_inches : ℝ := 12 ^ 3

theorem second_outlet_rate_calculation :
  second_outlet_rate = 
    (tank_volume * cubic_feet_to_inches - emptying_time * (inlet_rate - first_outlet_rate)) / 
    emptying_time := by
  sorry

end NUMINAMATH_CALUDE_second_outlet_rate_calculation_l1053_105355


namespace NUMINAMATH_CALUDE_t_range_theorem_max_radius_theorem_l1053_105361

-- Define the circle equation
def circle_equation (x y t : ℝ) : Prop :=
  x^2 + y^2 - 2*(t+3)*x + 2*(1-4*t^2)*y + 16*t^4 + 9 = 0

-- Define the range of t
def t_range (t : ℝ) : Prop :=
  -1/7 < t ∧ t < 1

-- Define the radius squared as a function of t
def radius_squared (t : ℝ) : ℝ :=
  -7*t^2 + 6*t + 1

-- Theorem for the range of t
theorem t_range_theorem :
  ∀ t : ℝ, (∃ x y : ℝ, circle_equation x y t) ↔ t_range t :=
sorry

-- Theorem for the maximum radius
theorem max_radius_theorem :
  ∃ t : ℝ, t_range t ∧ 
    ∀ t' : ℝ, t_range t' → radius_squared t ≥ radius_squared t' ∧
    t = 3/7 :=
sorry

end NUMINAMATH_CALUDE_t_range_theorem_max_radius_theorem_l1053_105361


namespace NUMINAMATH_CALUDE_new_ratio_is_two_to_three_l1053_105392

/-- Represents the ratio of two quantities -/
structure Ratio :=
  (numer : ℚ)
  (denom : ℚ)

/-- The initial ratio of acid to base -/
def initialRatio : Ratio := ⟨4, 1⟩

/-- The initial volume of acid in litres -/
def initialAcidVolume : ℚ := 16

/-- The volume of mixture taken out in litres -/
def volumeTakenOut : ℚ := 10

/-- The volume of base added in litres -/
def volumeBaseAdded : ℚ := 10

/-- Calculate the new ratio of acid to base after the replacement -/
def newRatio : Ratio :=
  let initialBaseVolume := initialAcidVolume / initialRatio.numer * initialRatio.denom
  let totalInitialVolume := initialAcidVolume + initialBaseVolume
  let acidRemoved := volumeTakenOut * (initialRatio.numer / (initialRatio.numer + initialRatio.denom))
  let baseRemoved := volumeTakenOut * (initialRatio.denom / (initialRatio.numer + initialRatio.denom))
  let remainingAcid := initialAcidVolume - acidRemoved
  let remainingBase := initialBaseVolume - baseRemoved + volumeBaseAdded
  ⟨remainingAcid, remainingBase⟩

theorem new_ratio_is_two_to_three :
  newRatio = ⟨2, 3⟩ := by sorry


end NUMINAMATH_CALUDE_new_ratio_is_two_to_three_l1053_105392


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1053_105395

/-- An isosceles triangle with sides 12cm and 24cm has a perimeter of 60cm -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 12 →
  b = 24 →
  c = 24 →
  a + b > c →
  a + c > b →
  b + c > a →
  a + b + c = 60 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1053_105395


namespace NUMINAMATH_CALUDE_jorge_total_goals_l1053_105382

/-- The total number of goals Jorge scored over two seasons -/
def total_goals (last_season_goals this_season_goals : ℕ) : ℕ :=
  last_season_goals + this_season_goals

/-- Theorem stating that Jorge's total goals over two seasons is 343 -/
theorem jorge_total_goals :
  total_goals 156 187 = 343 := by
  sorry

end NUMINAMATH_CALUDE_jorge_total_goals_l1053_105382


namespace NUMINAMATH_CALUDE_stream_speed_in_rowing_problem_l1053_105327

/-- Proves that the speed of the stream is 20 kmph given the conditions of the rowing problem. -/
theorem stream_speed_in_rowing_problem (boat_speed : ℝ) (stream_speed : ℝ) :
  boat_speed = 60 →
  (∀ d : ℝ, d > 0 → d / (boat_speed - stream_speed) = 2 * (d / (boat_speed + stream_speed))) →
  stream_speed = 20 :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_in_rowing_problem_l1053_105327


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l1053_105399

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  totalPopulation : ℕ
  totalSampleSize : ℕ
  stratumSize : ℕ
  stratumSampleSize : ℕ

/-- Checks if the stratified sample is proportional -/
def isProportional (s : StratifiedSample) : Prop :=
  s.stratumSampleSize * s.totalPopulation = s.totalSampleSize * s.stratumSize

theorem stratified_sample_theorem (s : StratifiedSample) 
  (h1 : s.totalPopulation = 2048)
  (h2 : s.totalSampleSize = 128)
  (h3 : s.stratumSize = 256)
  (h4 : isProportional s) :
  s.stratumSampleSize = 16 := by
  sorry

#check stratified_sample_theorem

end NUMINAMATH_CALUDE_stratified_sample_theorem_l1053_105399


namespace NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1053_105360

theorem perpendicular_lines_b_value (b : ℝ) : 
  (∀ x y : ℝ, 3 * y + 2 * x - 6 = 0 ∨ 4 * y + b * x + 8 = 0) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    3 * y₁ + 2 * x₁ - 6 = 0 ∧ 
    4 * y₂ + b * x₂ + 8 = 0 ∧
    (y₂ - y₁) * (x₂ - x₁) = 0) →
  b = -6 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_b_value_l1053_105360


namespace NUMINAMATH_CALUDE_focus_of_our_parabola_l1053_105342

/-- A parabola is defined by its equation and opening direction -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  opens_downward : Bool

/-- The focus of a parabola is a point in the plane -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Our specific parabola -/
def our_parabola : Parabola := {
  equation := fun x y => y = -1/4 * x^2,
  opens_downward := true
}

/-- Theorem stating that the focus of our parabola is at (0, -1) -/
theorem focus_of_our_parabola : focus our_parabola = (0, -1) := by sorry

end NUMINAMATH_CALUDE_focus_of_our_parabola_l1053_105342


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_determined_l1053_105359

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- Represents the division of a large rectangle into four smaller rectangles -/
structure DividedRectangle where
  large : Rectangle
  efgh : Rectangle
  efij : Rectangle
  ijkl : Rectangle
  ghkl : Rectangle
  h_division : 
    large.length = efgh.length + ijkl.length ∧ 
    large.width = efgh.width + efij.width

/-- Theorem stating that the area of the fourth rectangle (GHKL) is uniquely determined -/
theorem fourth_rectangle_area_determined (dr : DividedRectangle) : 
  ∃! a : ℝ, a = dr.ghkl.area ∧ 
    dr.large.area = dr.efgh.area + dr.efij.area + dr.ijkl.area + a :=
sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_determined_l1053_105359


namespace NUMINAMATH_CALUDE_probability_three_green_marbles_l1053_105300

theorem probability_three_green_marbles : 
  let total_marbles : ℕ := 15
  let green_marbles : ℕ := 8
  let purple_marbles : ℕ := 7
  let total_trials : ℕ := 7
  let green_trials : ℕ := 3
  
  let prob_green : ℚ := green_marbles / total_marbles
  let prob_purple : ℚ := purple_marbles / total_marbles
  
  let ways_to_choose_green : ℕ := Nat.choose total_trials green_trials
  let prob_specific_outcome : ℚ := prob_green ^ green_trials * prob_purple ^ (total_trials - green_trials)
  
  ways_to_choose_green * prob_specific_outcome = 43079680 / 170859375 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_three_green_marbles_l1053_105300


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1053_105369

theorem inequality_system_solution_set :
  let S := { x : ℝ | x - 1 < 7 ∧ 3 * x + 1 ≥ -2 }
  S = { x : ℝ | -1 ≤ x ∧ x < 8 } :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1053_105369


namespace NUMINAMATH_CALUDE_triangles_in_hexagon_with_center_l1053_105375

/-- The number of triangles formed by 7 points of a regular hexagon (including center) --/
def num_triangles_hexagon : ℕ :=
  Nat.choose 7 3 - 3

theorem triangles_in_hexagon_with_center :
  num_triangles_hexagon = 32 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_hexagon_with_center_l1053_105375


namespace NUMINAMATH_CALUDE_milk_consumption_l1053_105352

theorem milk_consumption (initial_milk : ℚ) (rachel_fraction : ℚ) (monica_fraction : ℚ) : 
  initial_milk = 3/4 →
  rachel_fraction = 1/2 →
  monica_fraction = 1/3 →
  let rachel_consumption := rachel_fraction * initial_milk
  let remaining_milk := initial_milk - rachel_consumption
  let monica_consumption := monica_fraction * remaining_milk
  rachel_consumption + monica_consumption = 1/2 := by
sorry

end NUMINAMATH_CALUDE_milk_consumption_l1053_105352


namespace NUMINAMATH_CALUDE_lottery_probability_l1053_105339

theorem lottery_probability (total_tickets winning_tickets people : ℕ) 
  (h1 : total_tickets = 10)
  (h2 : winning_tickets = 3)
  (h3 : people = 5) :
  let non_winning_tickets := total_tickets - winning_tickets
  1 - (Nat.choose non_winning_tickets people / Nat.choose total_tickets people : ℚ) = 11/12 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l1053_105339


namespace NUMINAMATH_CALUDE_flag_distribution_theorem_l1053_105317

structure FlagDistribution where
  total_flags : ℕ
  blue_percentage : ℚ
  red_percentage : ℚ
  green_percentage : ℚ

def children_with_both_blue_and_red (fd : FlagDistribution) : ℚ :=
  fd.blue_percentage + fd.red_percentage - 1

theorem flag_distribution_theorem (fd : FlagDistribution) 
  (h1 : Even fd.total_flags)
  (h2 : fd.blue_percentage = 1/2)
  (h3 : fd.red_percentage = 3/5)
  (h4 : fd.green_percentage = 2/5)
  (h5 : fd.blue_percentage + fd.red_percentage + fd.green_percentage = 3/2) :
  children_with_both_blue_and_red fd = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_flag_distribution_theorem_l1053_105317


namespace NUMINAMATH_CALUDE_consumption_increase_l1053_105381

theorem consumption_increase (original_tax original_consumption : ℝ) 
  (h1 : original_tax > 0) (h2 : original_consumption > 0) : 
  let new_tax := 0.65 * original_tax
  let new_revenue := 0.7475 * (original_tax * original_consumption)
  ∃ (new_consumption : ℝ), 
    new_revenue = new_tax * new_consumption ∧ 
    new_consumption = 1.15 * original_consumption :=
by
  sorry

end NUMINAMATH_CALUDE_consumption_increase_l1053_105381


namespace NUMINAMATH_CALUDE_solve_parking_problem_l1053_105389

def parking_problem (initial_balance : ℚ) (first_three_cost : ℚ) (fourth_cost_ratio : ℚ) (fifth_cost_ratio : ℚ) (roommate_payment_ratio : ℚ) : Prop :=
  let total_first_three := 3 * first_three_cost
  let fourth_ticket_cost := fourth_cost_ratio * first_three_cost
  let fifth_ticket_cost := fifth_cost_ratio * first_three_cost
  let total_cost := total_first_three + fourth_ticket_cost + fifth_ticket_cost
  let roommate_payment := roommate_payment_ratio * total_cost
  let james_payment := total_cost - roommate_payment
  let remaining_balance := initial_balance - james_payment
  remaining_balance = 871.88

theorem solve_parking_problem :
  parking_problem 1200 250 (1/4) (1/2) 0.65 :=
sorry

end NUMINAMATH_CALUDE_solve_parking_problem_l1053_105389


namespace NUMINAMATH_CALUDE_inclination_angle_range_l1053_105313

/-- A line passing through a point -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- A line segment defined by two endpoints -/
structure LineSegment where
  pointA : ℝ × ℝ
  pointB : ℝ × ℝ

/-- Checks if a line intersects a line segment -/
def intersects (l : Line) (seg : LineSegment) : Prop := sorry

/-- The inclination angle of a line -/
def inclinationAngle (l : Line) : ℝ := sorry

/-- The theorem statement -/
theorem inclination_angle_range 
  (l : Line) 
  (seg : LineSegment) :
  l.point = (0, -2) →
  seg.pointA = (1, -1) →
  seg.pointB = (2, -4) →
  intersects l seg →
  let α := inclinationAngle l
  (0 ≤ α ∧ α ≤ Real.pi / 4) ∨ (3 * Real.pi / 4 ≤ α ∧ α < Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_inclination_angle_range_l1053_105313


namespace NUMINAMATH_CALUDE_cos_15_cos_30_minus_sin_15_sin_150_l1053_105338

theorem cos_15_cos_30_minus_sin_15_sin_150 :
  Real.cos (15 * π / 180) * Real.cos (30 * π / 180) - 
  Real.sin (15 * π / 180) * Real.sin (150 * π / 180) = 
  Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_cos_15_cos_30_minus_sin_15_sin_150_l1053_105338


namespace NUMINAMATH_CALUDE_complex_equality_implies_ratio_l1053_105365

theorem complex_equality_implies_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b * Complex.I) ^ 4 = (a - b * Complex.I) ^ 4 →
  b / a = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_implies_ratio_l1053_105365


namespace NUMINAMATH_CALUDE_fraction_expansion_invariance_l1053_105349

theorem fraction_expansion_invariance (m n : ℝ) (h : m ≠ n) :
  (2 * (3 * m)) / ((3 * m) - (3 * n)) = (2 * m) / (m - n) := by
  sorry

end NUMINAMATH_CALUDE_fraction_expansion_invariance_l1053_105349


namespace NUMINAMATH_CALUDE_mms_per_pack_is_40_l1053_105366

/-- The number of sundaes made on Monday -/
def monday_sundaes : ℕ := 40

/-- The number of m&ms per sundae on Monday -/
def monday_mms_per_sundae : ℕ := 6

/-- The number of sundaes made on Tuesday -/
def tuesday_sundaes : ℕ := 20

/-- The number of m&ms per sundae on Tuesday -/
def tuesday_mms_per_sundae : ℕ := 10

/-- The total number of m&m packs used -/
def total_packs : ℕ := 11

/-- The number of m&ms in each pack -/
def mms_per_pack : ℕ := (monday_sundaes * monday_mms_per_sundae + tuesday_sundaes * tuesday_mms_per_sundae) / total_packs

theorem mms_per_pack_is_40 : mms_per_pack = 40 := by
  sorry

end NUMINAMATH_CALUDE_mms_per_pack_is_40_l1053_105366


namespace NUMINAMATH_CALUDE_sum_of_integers_l1053_105377

theorem sum_of_integers (x y z : ℕ+) 
  (h1 : x.val * y.val + z.val = y.val * z.val + x.val)
  (h2 : y.val * z.val + x.val = x.val * z.val + y.val)
  (h3 : x.val * z.val + y.val = 55)
  (h4 : Even x.val ∨ Even y.val ∨ Even z.val) :
  x.val + y.val + z.val = 56 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1053_105377


namespace NUMINAMATH_CALUDE_z_coordinate_is_zero_l1053_105315

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- A point on a line with a specific x-coordinate -/
def point_on_line (l : Line3D) (x : ℝ) : ℝ × ℝ × ℝ := sorry

theorem z_coordinate_is_zero : 
  let l : Line3D := { point1 := (1, 3, 2), point2 := (4, 2, -1) }
  let p := point_on_line l 3
  p.2.2 = 0 := by sorry

end NUMINAMATH_CALUDE_z_coordinate_is_zero_l1053_105315


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_necessary_not_sufficient_condition_l1053_105341

def M : Set ℝ := {x | (x + 3) * (x - 5) > 0}

def P (a : ℝ) : Set ℝ := {x | x^2 + (a - 8) * x - 8 * a ≤ 0}

def target_set : Set ℝ := {x | 5 < x ∧ x ≤ 8}

theorem sufficient_not_necessary_condition :
  (∀ a, a = 0 → M ∩ P a = target_set) ∧
  ¬(∀ a, M ∩ P a = target_set → a = 0) := by sorry

theorem necessary_not_sufficient_condition :
  (∀ a, M ∩ P a = target_set → a ≤ 3) ∧
  ¬(∀ a, a ≤ 3 → M ∩ P a = target_set) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_necessary_not_sufficient_condition_l1053_105341


namespace NUMINAMATH_CALUDE_david_spent_half_ben_spent_more_total_spent_is_48_l1053_105310

/-- The amount Ben spent at the bagel store -/
def ben_spent : ℝ := 32

/-- The amount David spent at the bagel store -/
def david_spent : ℝ := 16

/-- For every dollar Ben spent, David spent 50 cents less -/
theorem david_spent_half : david_spent = ben_spent / 2 := by sorry

/-- Ben paid $16.00 more than David -/
theorem ben_spent_more : ben_spent = david_spent + 16 := by sorry

/-- The total amount spent by Ben and David -/
def total_spent : ℝ := ben_spent + david_spent

theorem total_spent_is_48 : total_spent = 48 := by sorry

end NUMINAMATH_CALUDE_david_spent_half_ben_spent_more_total_spent_is_48_l1053_105310


namespace NUMINAMATH_CALUDE_base_conversion_proof_l1053_105304

/-- 
Given a positive integer n with the following properties:
1. Its base 9 representation is AB
2. Its base 7 representation is BA
3. A and B are single digits in their respective bases

This theorem proves that n = 31 in base 10.
-/
theorem base_conversion_proof (n : ℕ) (A B : ℕ) 
  (h1 : n = 9 * A + B)
  (h2 : n = 7 * B + A)
  (h3 : A < 9 ∧ B < 9)
  (h4 : A < 7 ∧ B < 7)
  (h5 : n > 0) :
  n = 31 := by
  sorry


end NUMINAMATH_CALUDE_base_conversion_proof_l1053_105304


namespace NUMINAMATH_CALUDE_derivative_of_even_function_l1053_105335

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the derivative of f as g
variable (g : ℝ → ℝ)

-- State the theorem
theorem derivative_of_even_function 
  (h1 : ∀ x, f (-x) = f x)  -- f is an even function
  (h2 : ∀ x, HasDerivAt f (g x) x)  -- g is the derivative of f
  : ∀ x, g (-x) = -g x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_even_function_l1053_105335


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1053_105397

theorem cubic_root_sum (a b c d : ℝ) (ha : a ≠ 0) 
  (h1 : a * 5^3 + b * 5^2 + c * 5 + d = 0)
  (h2 : a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) :
  (b + c) / a = -19 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1053_105397


namespace NUMINAMATH_CALUDE_office_chairs_probability_l1053_105372

theorem office_chairs_probability (black_chairs brown_chairs : ℕ) 
  (h1 : black_chairs = 15)
  (h2 : brown_chairs = 18) :
  let total_chairs := black_chairs + brown_chairs
  let prob_same_color := (black_chairs * (black_chairs - 1) + brown_chairs * (brown_chairs - 1)) / (total_chairs * (total_chairs - 1))
  prob_same_color = 43 / 88 := by
sorry

end NUMINAMATH_CALUDE_office_chairs_probability_l1053_105372


namespace NUMINAMATH_CALUDE_system_solution_l1053_105323

theorem system_solution : 
  ∀ x y z : ℕ, 
    (2 * x^2 + 30 * y^2 + 3 * z^2 + 12 * x * y + 12 * y * z = 308 ∧
     2 * x^2 + 6 * y^2 - 3 * z^2 + 12 * x * y - 12 * y * z = 92) →
    ((x = 7 ∧ y = 1 ∧ z = 4) ∨ (x = 4 ∧ y = 2 ∧ z = 2)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1053_105323


namespace NUMINAMATH_CALUDE_price_increase_percentage_l1053_105330

theorem price_increase_percentage (original_price new_price : ℝ) 
  (h1 : original_price = 300)
  (h2 : new_price = 420) :
  (new_price - original_price) / original_price * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l1053_105330


namespace NUMINAMATH_CALUDE_savings_percentage_l1053_105343

theorem savings_percentage (income : ℝ) (savings_rate : ℝ) : 
  savings_rate = 0.35 →
  (2 : ℝ) * (income * (1 - savings_rate)) = 
    income * (1 - savings_rate) + income * (1 - 2 * savings_rate) →
  savings_rate = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_savings_percentage_l1053_105343


namespace NUMINAMATH_CALUDE_construction_time_difference_l1053_105386

/-- Represents the work rate of one person per day -/
def work_rate : ℝ := 1

/-- Calculates the total work done given the number of workers, days, and work rate -/
def total_work (workers : ℕ) (days : ℕ) (rate : ℝ) : ℝ :=
  (workers : ℝ) * (days : ℝ) * rate

/-- Theorem: If 100 men work for 50 days and then 200 men work for another 50 days
    to complete a project in 100 days, it would take 150 days for 100 men to
    complete the same project working at the same rate. -/
theorem construction_time_difference :
  let initial_workers : ℕ := 100
  let additional_workers : ℕ := 100
  let initial_days : ℕ := 50
  let total_days : ℕ := 100
  let work_done_first_half := total_work initial_workers initial_days work_rate
  let work_done_second_half := total_work (initial_workers + additional_workers) initial_days work_rate
  let total_work_done := work_done_first_half + work_done_second_half
  total_work initial_workers 150 work_rate = total_work_done :=
by
  sorry

end NUMINAMATH_CALUDE_construction_time_difference_l1053_105386


namespace NUMINAMATH_CALUDE_f_has_maximum_for_negative_x_l1053_105312

/-- The function f(x) = 2x + 1/x - 1 has a maximum value when x < 0 -/
theorem f_has_maximum_for_negative_x :
  ∃ (M : ℝ), ∀ (x : ℝ), x < 0 → (2 * x + 1 / x - 1 : ℝ) ≤ M := by
  sorry

end NUMINAMATH_CALUDE_f_has_maximum_for_negative_x_l1053_105312


namespace NUMINAMATH_CALUDE_parallelogram_side_length_l1053_105340

/-- Given a parallelogram with adjacent sides of lengths 3s and 4s units forming a 30-degree angle,
    if the area is 18√3 square units, then s = 3^(3/4). -/
theorem parallelogram_side_length (s : ℝ) : 
  s > 0 →  -- Ensuring s is positive for physical meaning
  (3 * s) * (4 * s) * Real.sin (π / 6) = 18 * Real.sqrt 3 →
  s = 3 ^ (3 / 4) := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_side_length_l1053_105340


namespace NUMINAMATH_CALUDE_parabola_one_x_intercept_l1053_105326

/-- The parabola defined by x = -2y^2 + y + 1 has exactly one x-intercept -/
theorem parabola_one_x_intercept :
  ∃! x : ℝ, ∃ y : ℝ, x = -2 * y^2 + y + 1 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_parabola_one_x_intercept_l1053_105326


namespace NUMINAMATH_CALUDE_rental_car_cost_sharing_l1053_105368

theorem rental_car_cost_sharing (n : ℕ) (C : ℝ) (h : n > 1) :
  (C / (n - 1 : ℝ) - C / n = 0.125) → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_rental_car_cost_sharing_l1053_105368


namespace NUMINAMATH_CALUDE_grass_seed_cost_l1053_105393

/-- The cost of a 5-pound bag of grass seed -/
def cost_5lb : ℝ := 13.80

/-- The cost of a 10-pound bag of grass seed -/
def cost_10lb : ℝ := 20.43

/-- The cost of a 25-pound bag of grass seed -/
def cost_25lb : ℝ := 32.25

/-- The minimum amount of grass seed the customer must buy (in pounds) -/
def min_amount : ℝ := 65

/-- The maximum amount of grass seed the customer can buy (in pounds) -/
def max_amount : ℝ := 80

/-- The least possible cost for the customer -/
def least_cost : ℝ := 98.73

theorem grass_seed_cost : 
  2 * cost_25lb + cost_10lb + cost_5lb = least_cost ∧ 
  2 * 25 + 10 + 5 ≥ min_amount ∧
  2 * 25 + 10 + 5 ≤ max_amount := by
  sorry

end NUMINAMATH_CALUDE_grass_seed_cost_l1053_105393


namespace NUMINAMATH_CALUDE_m_value_l1053_105301

theorem m_value (m : ℝ) (M : Set ℝ) : M = {3, m + 1} → 4 ∈ M → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_m_value_l1053_105301


namespace NUMINAMATH_CALUDE_circle_symmetry_theorem_l1053_105309

/-- The equation of the circle C -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + 4*x + a*y - 5 = 0

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop :=
  2*x + y - 1 = 0

/-- The theorem stating the relationship between the circle and the line -/
theorem circle_symmetry_theorem (a : ℝ) : 
  (∀ x y : ℝ, circle_equation x y a → 
    ∃ x' y' : ℝ, circle_equation x' y' a ∧ 
    ((x + x')/2, (y + y')/2) ∈ {(x, y) | line_equation x y}) →
  a = -10 := by
  sorry


end NUMINAMATH_CALUDE_circle_symmetry_theorem_l1053_105309


namespace NUMINAMATH_CALUDE_integer_fractional_parts_theorem_l1053_105324

theorem integer_fractional_parts_theorem : ∃ (x y : ℝ), 
  (x = ⌊8 - Real.sqrt 11⌋) ∧ 
  (y = 8 - Real.sqrt 11 - ⌊8 - Real.sqrt 11⌋) ∧ 
  (2 * x * y - y^2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_integer_fractional_parts_theorem_l1053_105324


namespace NUMINAMATH_CALUDE_line_relationships_l1053_105337

/-- Definition of parallel lines based on slopes -/
def parallel (m1 m2 : ℚ) : Prop := m1 = m2

/-- Definition of perpendicular lines based on slopes -/
def perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

/-- The main theorem -/
theorem line_relationships :
  let slopes : List ℚ := [2, -3, 3, 4, -3/2]
  ∃! (pair : (ℚ × ℚ)), pair ∈ (slopes.product slopes) ∧
    (parallel pair.1 pair.2 ∨ perpendicular pair.1 pair.2) ∧
    pair.1 ≠ pair.2 :=
by sorry

end NUMINAMATH_CALUDE_line_relationships_l1053_105337


namespace NUMINAMATH_CALUDE_correct_score_is_even_l1053_105328

/-- Represents the scoring system for a math competition -/
structure ScoringSystem where
  correct : Int
  unanswered : Int
  incorrect : Int

/-- Represents the results of a class in the math competition -/
structure CompetitionResult where
  total_questions : Nat
  scoring : ScoringSystem
  first_calculation : Int
  second_calculation : Int

/-- Theorem stating that the correct total score must be even -/
theorem correct_score_is_even (result : CompetitionResult) 
  (h1 : result.scoring.correct = 3)
  (h2 : result.scoring.unanswered = 1)
  (h3 : result.scoring.incorrect = -1)
  (h4 : result.total_questions = 50)
  (h5 : result.first_calculation = 5734)
  (h6 : result.second_calculation = 5735)
  (h7 : result.first_calculation = 5734 ∨ result.second_calculation = 5734) :
  ∃ (n : Int), 2 * n = 5734 ∧ (result.first_calculation = 5734 ∨ result.second_calculation = 5734) :=
sorry

end NUMINAMATH_CALUDE_correct_score_is_even_l1053_105328


namespace NUMINAMATH_CALUDE_problem_solution_l1053_105344

theorem problem_solution (a : ℝ) (h : a = 2 / (3 - Real.sqrt 7)) :
  -2 * a^2 + 12 * a + 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1053_105344
