import Mathlib

namespace NUMINAMATH_CALUDE_tristan_saturday_study_time_l3354_335449

/-- Calculates Tristan's study hours on Saturday given his study schedule --/
def tristanSaturdayStudyHours (mondayHours : ℕ) (weekdayHours : ℕ) (totalWeekHours : ℕ) : ℕ :=
  let tuesdayHours := 2 * mondayHours
  let wednesdayToFridayHours := 3 * weekdayHours
  let mondayToFridayHours := mondayHours + tuesdayHours + wednesdayToFridayHours
  let remainingHours := totalWeekHours - mondayToFridayHours
  remainingHours / 2

/-- Theorem: Given Tristan's study schedule, he studies for 2 hours on Saturday --/
theorem tristan_saturday_study_time :
  tristanSaturdayStudyHours 4 3 25 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tristan_saturday_study_time_l3354_335449


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3354_335474

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 11) * (Real.sqrt 6 / Real.sqrt 8) =
  (3 * Real.sqrt 385) / 154 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3354_335474


namespace NUMINAMATH_CALUDE_thermometer_price_is_two_l3354_335473

/-- Represents the sales data for thermometers and hot-water bottles --/
structure SalesData where
  thermometer_price : ℝ
  hotwater_bottle_price : ℝ
  total_sales : ℝ
  thermometer_to_bottle_ratio : ℕ
  bottles_sold : ℕ

/-- Theorem stating that the thermometer price is 2 dollars given the sales data --/
theorem thermometer_price_is_two (data : SalesData)
  (h1 : data.hotwater_bottle_price = 6)
  (h2 : data.total_sales = 1200)
  (h3 : data.thermometer_to_bottle_ratio = 7)
  (h4 : data.bottles_sold = 60)
  : data.thermometer_price = 2 := by
  sorry


end NUMINAMATH_CALUDE_thermometer_price_is_two_l3354_335473


namespace NUMINAMATH_CALUDE_equal_count_for_any_number_l3354_335450

/-- A function that represents the number of n-digit numbers from which 
    a k-digit number composed of only 1 and 2 can be obtained by erasing digits -/
def F (k n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number is composed only of digits 1 and 2 -/
def OnlyOneTwo (x : ℕ) : Prop := sorry

theorem equal_count_for_any_number (k n : ℕ) (X Y : ℕ) (h1 : k > 0) (h2 : n ≥ k) 
  (hX : OnlyOneTwo X) (hY : OnlyOneTwo Y) 
  (hXdigits : X < 10^k) (hYdigits : Y < 10^k) : F k n = F k n := by
  sorry

end NUMINAMATH_CALUDE_equal_count_for_any_number_l3354_335450


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3354_335471

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_ratio
  (a : ℕ → ℝ) (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_sum_24 : a 2 + a 4 = 20)
  (h_sum_35 : a 3 + a 5 = 40) :
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3354_335471


namespace NUMINAMATH_CALUDE_square_fraction_integers_l3354_335444

theorem square_fraction_integers (n : ℕ) : n > 1 ∧ ∃ (k : ℕ), k > 0 ∧ (n^2 + 7*n + 136) / (n - 1) = k^2 ↔ n = 5 ∨ n = 37 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_integers_l3354_335444


namespace NUMINAMATH_CALUDE_line_equation_solution_l3354_335402

/-- The line x = k intersects y = x^2 + 4x + 4 and y = mx + b at two points 4 units apart -/
def intersectionCondition (m b k : ℝ) : Prop :=
  ∃ k, |k^2 + 4*k + 4 - (m*k + b)| = 4

/-- The line y = mx + b passes through the point (2, 8) -/
def passesThroughPoint (m b : ℝ) : Prop :=
  8 = 2*m + b

/-- b is not equal to 0 -/
def bNonZero (b : ℝ) : Prop :=
  b ≠ 0

/-- The theorem stating that given the conditions, the unique solution for the line equation is y = 8x - 8 -/
theorem line_equation_solution (m b : ℝ) :
  (∃ k, intersectionCondition m b k) →
  passesThroughPoint m b →
  bNonZero b →
  m = 8 ∧ b = -8 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_solution_l3354_335402


namespace NUMINAMATH_CALUDE_two_point_six_million_scientific_notation_l3354_335414

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem two_point_six_million_scientific_notation :
  toScientificNotation 2600000 = ScientificNotation.mk 2.6 6 sorry := by
  sorry

end NUMINAMATH_CALUDE_two_point_six_million_scientific_notation_l3354_335414


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l3354_335478

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l3354_335478


namespace NUMINAMATH_CALUDE_sandy_change_is_three_l3354_335431

/-- Represents the cost and quantity of a drink order -/
structure DrinkOrder where
  name : String
  price : ℚ
  quantity : ℕ

/-- Calculates the total cost of a drink order -/
def orderCost (order : DrinkOrder) : ℚ :=
  order.price * order.quantity

/-- Calculates the total cost of multiple drink orders -/
def totalCost (orders : List DrinkOrder) : ℚ :=
  orders.map orderCost |>.sum

/-- Calculates the change from a given amount -/
def calculateChange (paid : ℚ) (cost : ℚ) : ℚ :=
  paid - cost

theorem sandy_change_is_three :
  let orders : List DrinkOrder := [
    { name := "Cappuccino", price := 2, quantity := 3 },
    { name := "Iced Tea", price := 3, quantity := 2 },
    { name := "Cafe Latte", price := 1.5, quantity := 2 },
    { name := "Espresso", price := 1, quantity := 2 }
  ]
  let total := totalCost orders
  let paid := 20
  calculateChange paid total = 3 := by sorry

end NUMINAMATH_CALUDE_sandy_change_is_three_l3354_335431


namespace NUMINAMATH_CALUDE_initial_fish_count_l3354_335429

theorem initial_fish_count (x : ℕ) : 
  x - 50 - (x - 50) / 3 + 200 = 300 → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_initial_fish_count_l3354_335429


namespace NUMINAMATH_CALUDE_min_value_expression_l3354_335413

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 2) :
  (1/a + 1/(2*b) + 4*a*b) ≥ 4 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 2 ∧ (1/a + 1/(2*b) + 4*a*b) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3354_335413


namespace NUMINAMATH_CALUDE_perspective_properties_l3354_335440

-- Define a type for perspective drawings
def PerspectiveDrawing : Type := sorry

-- Define a function to represent the perspective drawing of a square
def perspectiveSquare : PerspectiveDrawing → Bool := sorry

-- Define a function to represent the perspective drawing of intersecting lines
def perspectiveIntersectingLines : PerspectiveDrawing → Bool := sorry

-- Define a function to represent the perspective drawing of perpendicular lines
def perspectivePerpendicularLines : PerspectiveDrawing → Bool := sorry

theorem perspective_properties :
  ∃ (d : PerspectiveDrawing),
    (¬ perspectiveSquare d) ∧
    (¬ perspectiveIntersectingLines d) ∧
    (¬ perspectivePerpendicularLines d) := by
  sorry

end NUMINAMATH_CALUDE_perspective_properties_l3354_335440


namespace NUMINAMATH_CALUDE_x_value_theorem_l3354_335454

theorem x_value_theorem (x y z a b c : ℝ) 
  (h1 : x * y / (x + y) = a)
  (h2 : x * z / (x + z) = b)
  (h3 : y * z / (y + z) = c)
  (h4 : a ≠ 0)
  (h5 : b ≠ 0)
  (h6 : c ≠ 0)
  (h7 : x + y + z = a * b * c) :
  x = 2 * a * b * c / (a * b + b * c + a * c) := by
  sorry

end NUMINAMATH_CALUDE_x_value_theorem_l3354_335454


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3354_335485

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y : ℝ, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3354_335485


namespace NUMINAMATH_CALUDE_total_population_l3354_335418

/-- Represents the population of a school -/
structure SchoolPopulation where
  boys : ℕ
  girls : ℕ
  teachers : ℕ
  staff : ℕ

/-- The ratios in the school population -/
def school_ratios (p : SchoolPopulation) : Prop :=
  p.boys = 4 * p.girls ∧ 
  p.girls = 8 * p.teachers ∧ 
  p.staff = 2 * p.teachers

theorem total_population (p : SchoolPopulation) 
  (h : school_ratios p) : 
  p.boys + p.girls + p.teachers + p.staff = (43 * p.boys) / 32 := by
  sorry

end NUMINAMATH_CALUDE_total_population_l3354_335418


namespace NUMINAMATH_CALUDE_pages_read_later_l3354_335488

/-- Given that Jake initially read some pages of a book and then read more later,
    prove that the number of pages he read later is the difference between
    the total pages read and the initial pages read. -/
theorem pages_read_later (initial_pages total_pages pages_read_later : ℕ) :
  initial_pages + pages_read_later = total_pages →
  pages_read_later = total_pages - initial_pages := by
  sorry

#check pages_read_later

end NUMINAMATH_CALUDE_pages_read_later_l3354_335488


namespace NUMINAMATH_CALUDE_expression_simplification_l3354_335427

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 3 / y) :
  (3 * x - 3 / x) * (3 * y + 3 / y) = 9 * x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3354_335427


namespace NUMINAMATH_CALUDE_contractor_absent_days_l3354_335400

/-- Represents the contract details and outcome -/
structure ContractDetails where
  totalDays : ℕ
  paymentPerDay : ℚ
  finePerDay : ℚ
  totalReceived : ℚ

/-- Calculates the number of absent days given the contract details -/
def absentDays (c : ContractDetails) : ℚ :=
  (c.totalDays * c.paymentPerDay - c.totalReceived) / (c.paymentPerDay + c.finePerDay)

/-- Theorem stating that given the specific contract details, the number of absent days is 10 -/
theorem contractor_absent_days :
  let c : ContractDetails := {
    totalDays := 30,
    paymentPerDay := 25,
    finePerDay := 15/2,
    totalReceived := 425
  }
  absentDays c = 10 := by sorry

end NUMINAMATH_CALUDE_contractor_absent_days_l3354_335400


namespace NUMINAMATH_CALUDE_garden_fence_length_l3354_335420

/-- The total length of a fence surrounding a sector-shaped garden -/
def fence_length (radius : ℝ) (central_angle : ℝ) : ℝ :=
  radius * central_angle + 2 * radius

/-- Proof that the fence length for a garden with radius 30m and central angle 120° is 20π + 60m -/
theorem garden_fence_length :
  fence_length 30 (2 * Real.pi / 3) = 20 * Real.pi + 60 := by
  sorry

end NUMINAMATH_CALUDE_garden_fence_length_l3354_335420


namespace NUMINAMATH_CALUDE_carnival_wait_time_l3354_335462

/-- Proves that the wait time for the roller coaster is 30 minutes given the carnival conditions --/
theorem carnival_wait_time (total_time : ℕ) (tilt_a_whirl_wait : ℕ) (giant_slide_wait : ℕ)
  (roller_coaster_rides : ℕ) (tilt_a_whirl_rides : ℕ) (giant_slide_rides : ℕ) :
  total_time = 4 * 60 ∧
  tilt_a_whirl_wait = 60 ∧
  giant_slide_wait = 15 ∧
  roller_coaster_rides = 4 ∧
  tilt_a_whirl_rides = 1 ∧
  giant_slide_rides = 4 →
  ∃ (roller_coaster_wait : ℕ),
    roller_coaster_wait = 30 ∧
    total_time = roller_coaster_rides * roller_coaster_wait +
                 tilt_a_whirl_rides * tilt_a_whirl_wait +
                 giant_slide_rides * giant_slide_wait :=
by
  sorry

end NUMINAMATH_CALUDE_carnival_wait_time_l3354_335462


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3354_335483

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (1 / x) + (9 / y) = 1) : 
  x + y ≥ 16 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (1 / x) + (9 / y) = 1 ∧ x + y = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3354_335483


namespace NUMINAMATH_CALUDE_min_overlap_percentage_l3354_335456

theorem min_overlap_percentage (math_pref science_pref : ℝ) 
  (h1 : math_pref = 0.90)
  (h2 : science_pref = 0.85) :
  let overlap := math_pref + science_pref - 1
  overlap ≥ 0.75 ∧ 
  ∀ x, x ≥ 0 ∧ x < overlap → 
    ∃ total_pref, total_pref ≤ 1 ∧ 
      total_pref = math_pref + science_pref - x :=
by sorry

end NUMINAMATH_CALUDE_min_overlap_percentage_l3354_335456


namespace NUMINAMATH_CALUDE_justin_run_time_l3354_335461

/-- Represents Justin's running speed and route information -/
structure RunningInfo where
  flat_speed : ℚ  -- blocks per minute on flat ground
  uphill_speed : ℚ  -- blocks per minute uphill
  total_distance : ℕ  -- total blocks to home
  uphill_distance : ℕ  -- blocks that are uphill

/-- Calculates the total time Justin needs to run home -/
def time_to_run_home (info : RunningInfo) : ℚ :=
  let flat_distance := info.total_distance - info.uphill_distance
  let flat_time := flat_distance / info.flat_speed
  let uphill_time := info.uphill_distance / info.uphill_speed
  flat_time + uphill_time

/-- Theorem stating that Justin will take 13 minutes to run home -/
theorem justin_run_time :
  let info : RunningInfo := {
    flat_speed := 1,  -- 2 blocks / 2 minutes
    uphill_speed := 2/3,  -- 2 blocks / 3 minutes
    total_distance := 10,
    uphill_distance := 6
  }
  time_to_run_home info = 13 := by
  sorry


end NUMINAMATH_CALUDE_justin_run_time_l3354_335461


namespace NUMINAMATH_CALUDE_no_constant_term_in_expansion_l3354_335465

theorem no_constant_term_in_expansion :
  ∀ k : ℕ, k ≤ 12 →
    (12 - k : ℚ) / 2 - 2 * k ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_constant_term_in_expansion_l3354_335465


namespace NUMINAMATH_CALUDE_phone_number_probability_l3354_335499

theorem phone_number_probability : 
  ∀ (n : ℕ) (p : ℚ),
    n = 10 →  -- There are 10 possible digits
    p = 1 / n →  -- Probability of correct guess on each attempt
    (p + p + p) = 3 / 10  -- Probability of success in no more than 3 attempts
    := by sorry

end NUMINAMATH_CALUDE_phone_number_probability_l3354_335499


namespace NUMINAMATH_CALUDE_ashley_tablet_battery_life_l3354_335489

/-- Represents the battery life of Ashley's tablet -/
structure TabletBattery where
  fullLifeIdle : ℝ  -- Battery life in hours when idle
  fullLifeActive : ℝ  -- Battery life in hours when active
  usedTime : ℝ  -- Total time used since last charge
  activeTime : ℝ  -- Time spent actively using the tablet

/-- Calculates the remaining battery life of Ashley's tablet -/
def remainingBatteryLife (tb : TabletBattery) : ℝ :=
  sorry

/-- Theorem stating that Ashley's tablet will last 8 more hours -/
theorem ashley_tablet_battery_life :
  ∀ (tb : TabletBattery),
    tb.fullLifeIdle = 36 ∧
    tb.fullLifeActive = 4 ∧
    tb.usedTime = 12 ∧
    tb.activeTime = 2 →
    remainingBatteryLife tb = 8 :=
  sorry

end NUMINAMATH_CALUDE_ashley_tablet_battery_life_l3354_335489


namespace NUMINAMATH_CALUDE_like_terms_example_l3354_335447

/-- Two monomials are like terms if they have the same variables raised to the same powers. -/
def are_like_terms (expr1 expr2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (x y : ℕ), expr1 x y ≠ 0 ∧ expr2 x y ≠ 0 → 
    (∃ (c1 c2 : ℚ), expr1 x y = c1 * x^5 * y^4 ∧ expr2 x y = c2 * x^5 * y^4)

theorem like_terms_example (a b : ℕ) (h1 : a = 2) (h2 : b = 3) :
  are_like_terms (λ x y => b * x^(2*a+1) * y^4) (λ x y => a * x^5 * y^(b+1)) :=
by
  sorry

end NUMINAMATH_CALUDE_like_terms_example_l3354_335447


namespace NUMINAMATH_CALUDE_like_terms_exponents_l3354_335434

theorem like_terms_exponents (a b x y : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ 5 * a^(|x|) * b^2 = k * (-0.2 * a^3 * b^(|y|))) → 
  |x| = 3 ∧ |y| = 2 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l3354_335434


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l3354_335421

theorem min_value_of_sum_of_ratios (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b) + (b / c) + (c / a) + (a / c) ≥ 4 ∧
  ((a / b) + (b / c) + (c / a) + (a / c) = 4 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l3354_335421


namespace NUMINAMATH_CALUDE_race_speed_ratio_l3354_335458

/-- Given two racers a and b, where a's speed is some multiple of b's speed,
    and a gives b a 0.2 part of the race length as a head start resulting in a dead heat,
    prove that the ratio of a's speed to b's speed is 5:4 -/
theorem race_speed_ratio (L : ℝ) (v_a v_b : ℝ) (h1 : v_a > 0) (h2 : v_b > 0) 
    (h3 : ∃ k : ℝ, v_a = k * v_b) 
    (h4 : L / v_a = (0.8 * L) / v_b) : 
  v_a / v_b = 5 / 4 := by
sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l3354_335458


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l3354_335463

theorem complex_purely_imaginary (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 + x - 2) (x^2 + 3*x + 2)
  (z.re = 0 ∧ z.im ≠ 0) → x = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l3354_335463


namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_solution_for_27461_answer_is_seven_l3354_335446

theorem smallest_addition_for_divisibility (n : ℕ) : 
  (∃ (k : ℕ), k < 9 ∧ (n + k) % 9 = 0) → 
  (∃ (m : ℕ), m < 9 ∧ (n + m) % 9 = 0 ∧ ∀ (l : ℕ), l < m → (n + l) % 9 ≠ 0) :=
by sorry

theorem solution_for_27461 : 
  ∃ (k : ℕ), k < 9 ∧ (27461 + k) % 9 = 0 ∧ ∀ (l : ℕ), l < k → (27461 + l) % 9 ≠ 0 :=
by sorry

theorem answer_is_seven : 
  ∃ (k : ℕ), k = 7 ∧ (27461 + k) % 9 = 0 ∧ ∀ (l : ℕ), l < k → (27461 + l) % 9 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_solution_for_27461_answer_is_seven_l3354_335446


namespace NUMINAMATH_CALUDE_square_b_minus_d_l3354_335451

theorem square_b_minus_d (a b c d : ℤ) 
  (eq1 : a - b - c + d = 18) 
  (eq2 : a + b - c - d = 6) : 
  (b - d)^2 = 36 := by
sorry

end NUMINAMATH_CALUDE_square_b_minus_d_l3354_335451


namespace NUMINAMATH_CALUDE_yoongi_initial_books_l3354_335468

/-- Represents the number of books each person has -/
structure BookCount where
  yoongi : ℕ
  eunji : ℕ
  yuna : ℕ

/-- Represents the book exchange described in the problem -/
def exchange (initial : BookCount) : BookCount :=
  { yoongi := initial.yoongi - 5 + 15,
    eunji := initial.eunji + 5 - 10,
    yuna := initial.yuna + 10 - 15 }

/-- Theorem stating that if after the exchange all have 45 books, 
    Yoongi must have started with 35 books -/
theorem yoongi_initial_books 
  (initial : BookCount) 
  (h : exchange initial = {yoongi := 45, eunji := 45, yuna := 45}) : 
  initial.yoongi = 35 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_initial_books_l3354_335468


namespace NUMINAMATH_CALUDE_adoption_time_proof_l3354_335415

/-- The number of days required to adopt all puppies in a pet shelter -/
def adoptionDays (initialPuppies additionalPuppies adoptionRate : ℕ) : ℕ :=
  (initialPuppies + additionalPuppies) / adoptionRate

/-- Theorem: It takes 7 days to adopt all puppies given the specified conditions -/
theorem adoption_time_proof :
  adoptionDays 9 12 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_adoption_time_proof_l3354_335415


namespace NUMINAMATH_CALUDE_old_pump_fills_in_600_seconds_l3354_335457

/-- Represents the time (in seconds) taken by the old pump to fill the trough alone. -/
def old_pump_time : ℝ := 600

/-- Represents the time (in seconds) taken by the new pump to fill the trough alone. -/
def new_pump_time : ℝ := 200

/-- Represents the time (in seconds) taken by both pumps working together to fill the trough. -/
def combined_time : ℝ := 150

/-- 
Proves that the old pump takes 600 seconds to fill the trough alone, given the times for the new pump
and both pumps working together.
-/
theorem old_pump_fills_in_600_seconds :
  (1 / old_pump_time) + (1 / new_pump_time) = (1 / combined_time) :=
by sorry

end NUMINAMATH_CALUDE_old_pump_fills_in_600_seconds_l3354_335457


namespace NUMINAMATH_CALUDE_west_distance_negative_l3354_335423

-- Define a type for direction
inductive Direction
  | East
  | West

-- Define a function to record distance based on direction
def recordDistance (distance : ℝ) (direction : Direction) : ℝ :=
  match direction with
  | Direction.East => distance
  | Direction.West => -distance

-- Theorem statement
theorem west_distance_negative (d : ℝ) :
  d > 0 → recordDistance d Direction.East = d → recordDistance d Direction.West = -d :=
by
  sorry

end NUMINAMATH_CALUDE_west_distance_negative_l3354_335423


namespace NUMINAMATH_CALUDE_cross_section_area_l3354_335479

/-- Regular hexagonal pyramid with square lateral sides -/
structure HexagonalPyramid where
  /-- Side length of the base -/
  a : ℝ
  /-- Assumption that a is positive -/
  a_pos : 0 < a

/-- Cross-section of the hexagonal pyramid -/
def cross_section (pyramid : HexagonalPyramid) : Set (Fin 3 → ℝ) :=
  sorry

/-- Area of a set in ℝ² -/
noncomputable def area (s : Set (Fin 3 → ℝ)) : ℝ :=
  sorry

/-- Theorem: The area of the cross-section is 3a² -/
theorem cross_section_area (pyramid : HexagonalPyramid) :
    area (cross_section pyramid) = 3 * pyramid.a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_area_l3354_335479


namespace NUMINAMATH_CALUDE_sum_of_squares_fourth_degree_equation_l3354_335497

-- Part 1
theorem sum_of_squares (x y : ℝ) :
  (x^2 + y^2 - 4) * (x^2 + y^2 + 2) = 7 → x^2 + y^2 = 5 :=
by sorry

-- Part 2
theorem fourth_degree_equation (x : ℝ) :
  x^4 - 6*x^2 + 8 = 0 → x = Real.sqrt 2 ∨ x = -Real.sqrt 2 ∨ x = 2 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_fourth_degree_equation_l3354_335497


namespace NUMINAMATH_CALUDE_tangent_ball_prism_area_relation_l3354_335428

/-- A quadrangular prism with a small ball tangent to each face -/
structure TangentBallPrism where
  S₁ : ℝ  -- Area of the upper base
  S₂ : ℝ  -- Area of the lower base
  S : ℝ   -- Lateral surface area
  h₁ : 0 < S₁  -- S₁ is positive
  h₂ : 0 < S₂  -- S₂ is positive
  h₃ : 0 < S   -- S is positive

/-- The relationship between the lateral surface area and the base areas -/
theorem tangent_ball_prism_area_relation (p : TangentBallPrism) : 
  Real.sqrt p.S = Real.sqrt p.S₁ + Real.sqrt p.S₂ := by
  sorry

end NUMINAMATH_CALUDE_tangent_ball_prism_area_relation_l3354_335428


namespace NUMINAMATH_CALUDE_unique_functional_equation_l3354_335442

theorem unique_functional_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (2 * f x + f y) = 2 * x + f y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_functional_equation_l3354_335442


namespace NUMINAMATH_CALUDE_minimum_a_value_l3354_335425

def set_A : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 2)^2 ≤ 4/5}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p | |p.1 - 1| + 2*|p.2 - 2| ≤ a}

theorem minimum_a_value (a : ℝ) (h : set_A ⊆ set_B a) : a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_minimum_a_value_l3354_335425


namespace NUMINAMATH_CALUDE_peanut_mixture_relation_l3354_335466

/-- Represents the peanut mixture problem -/
def PeanutMixture (S T : ℝ) : Prop :=
  let virginiaWeight : ℝ := 10
  let virginiaCost : ℝ := 3.50
  let spanishCost : ℝ := 3.00
  let texanCost : ℝ := 4.00
  let mixtureCost : ℝ := 3.60
  let totalWeight : ℝ := virginiaWeight + S + T
  let totalCost : ℝ := virginiaWeight * virginiaCost + S * spanishCost + T * texanCost
  (totalCost / totalWeight = mixtureCost) ∧ (0.40 * T - 0.60 * S = 1)

/-- Theorem stating the relationship between Spanish and Texan peanut weights -/
theorem peanut_mixture_relation (S T : ℝ) :
  PeanutMixture S T → (0.40 * T - 0.60 * S = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_peanut_mixture_relation_l3354_335466


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3354_335486

theorem max_sum_of_factors (clubsuit heartsuit : ℕ) : 
  clubsuit * heartsuit = 48 → 
  Even clubsuit → 
  ∃ (a b : ℕ), a * b = 48 ∧ Even a ∧ a + b ≤ clubsuit + heartsuit ∧ a + b = 26 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3354_335486


namespace NUMINAMATH_CALUDE_m_range_correct_l3354_335491

/-- Statement p: For all x in ℝ, x^2 + mx + m/2 + 2 ≥ 0 always holds true -/
def statement_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m*x + m/2 + 2 ≥ 0

/-- Statement q: The distance from the focus of the parabola y^2 = 2mx (where m > 0) to its directrix is greater than 1 -/
def statement_q (m : ℝ) : Prop :=
  m > 0 ∧ m/2 > 1

/-- The range of m that satisfies all conditions -/
def m_range : Set ℝ :=
  {m : ℝ | m > 4}

theorem m_range_correct :
  ∀ m : ℝ, (statement_p m ∨ statement_q m) ∧ ¬(statement_p m ∧ statement_q m) ↔ m ∈ m_range := by
  sorry

end NUMINAMATH_CALUDE_m_range_correct_l3354_335491


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3354_335470

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_sum : ∀ n : ℕ, S n = (a 0) * (1 - (a 1 / a 0)^n) / (1 - (a 1 / a 0))) 
  (h_a2 : a 2 = 1/4) 
  (h_S3 : S 3 = 7/8) :
  (a 1 / a 0 = 2) ∨ (a 1 / a 0 = 1/2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3354_335470


namespace NUMINAMATH_CALUDE_right_angled_triangle_l3354_335477

theorem right_angled_triangle (A B C : ℝ) (h : Real.sin A + Real.sin B = Real.sin C * (Real.cos A + Real.cos B)) :
  Real.cos C = 0 :=
sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l3354_335477


namespace NUMINAMATH_CALUDE_square_area_equals_triangle_perimeter_l3354_335498

/-- Given a right-angled triangle with sides 6 cm and 8 cm, 
    a square with the same perimeter as this triangle has an area of 36 cm². -/
theorem square_area_equals_triangle_perimeter : 
  ∃ (triangle_hypotenuse : ℝ) (square_side : ℝ),
    triangle_hypotenuse^2 = 6^2 + 8^2 ∧ 
    6 + 8 + triangle_hypotenuse = 4 * square_side ∧
    square_side^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equals_triangle_perimeter_l3354_335498


namespace NUMINAMATH_CALUDE_chess_tournament_games_l3354_335459

/-- Calculate the number of games in a chess tournament --/
def tournament_games (n : ℕ) : ℕ :=
  n * (n - 1)

/-- The number of players in the tournament --/
def num_players : ℕ := 12

/-- The number of times each pair of players compete --/
def games_per_pair : ℕ := 2

theorem chess_tournament_games :
  tournament_games num_players * games_per_pair = 264 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l3354_335459


namespace NUMINAMATH_CALUDE_min_difference_unit_complex_l3354_335416

theorem min_difference_unit_complex (z w : ℂ) 
  (hz : Complex.abs z = 1) 
  (hw : Complex.abs w = 1) 
  (h_sum : 1 ≤ Complex.abs (z + w) ∧ Complex.abs (z + w) ≤ Real.sqrt 2) : 
  Real.sqrt 2 ≤ Complex.abs (z - w) := by
  sorry

end NUMINAMATH_CALUDE_min_difference_unit_complex_l3354_335416


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3354_335467

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a) 
  (h_product : a 1 * a 5 = 16) : 
  a 3 = 4 ∨ a 3 = -4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3354_335467


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l3354_335495

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def is_in_first_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The given point (3,5) -/
def given_point : Point :=
  { x := 3, y := 5 }

/-- Theorem: The given point (3,5) is in the first quadrant -/
theorem point_in_first_quadrant :
  is_in_first_quadrant given_point := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l3354_335495


namespace NUMINAMATH_CALUDE_suzy_twice_mary_age_l3354_335438

/-- The number of years in the future when Suzy will be twice Mary's age -/
def future_years : ℕ := 4

/-- Suzy's current age -/
def suzy_age : ℕ := 20

/-- Mary's current age -/
def mary_age : ℕ := 8

/-- Theorem stating that in 'future_years', Suzy will be twice Mary's age -/
theorem suzy_twice_mary_age : 
  suzy_age + future_years = 2 * (mary_age + future_years) := by sorry

end NUMINAMATH_CALUDE_suzy_twice_mary_age_l3354_335438


namespace NUMINAMATH_CALUDE_trig_identity_l3354_335481

theorem trig_identity (α : ℝ) : 
  4.3 * Real.sin (4 * α) - Real.sin (5 * α) - Real.sin (6 * α) + Real.sin (7 * α) = 
  -4 * Real.sin (α / 2) * Real.sin α * Real.sin (11 * α / 2) := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3354_335481


namespace NUMINAMATH_CALUDE_power_comparison_l3354_335412

theorem power_comparison : 4^15 = 8^10 ∧ 8^10 < 2^31 := by sorry

end NUMINAMATH_CALUDE_power_comparison_l3354_335412


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3354_335430

theorem trigonometric_equation_solution (t : ℝ) : 
  (Real.sin (2 * t))^6 + (Real.cos (2 * t))^6 = 
    3/2 * ((Real.sin (2 * t))^4 + (Real.cos (2 * t))^4) + 1/2 * (Real.sin t + Real.cos t) ↔ 
  (∃ k : ℤ, t = π * (2 * k + 1)) ∨ 
  (∃ n : ℤ, t = π/2 * (4 * n - 1)) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3354_335430


namespace NUMINAMATH_CALUDE_expression_evaluation_l3354_335436

theorem expression_evaluation :
  let a : ℚ := 1/2
  let b : ℚ := -1/3
  b^2 - a^2 + 2*(a^2 + a*b) - (a^2 + b^2) = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3354_335436


namespace NUMINAMATH_CALUDE_class_group_division_l3354_335435

theorem class_group_division (total_students : ℕ) (students_per_group : ℕ) (h1 : total_students = 32) (h2 : students_per_group = 6) :
  total_students / students_per_group = 5 :=
by sorry

end NUMINAMATH_CALUDE_class_group_division_l3354_335435


namespace NUMINAMATH_CALUDE_sum_abcd_equals_negative_eleven_l3354_335469

theorem sum_abcd_equals_negative_eleven (a b c d : ℚ) 
  (h : 2*a + 3 = 2*b + 4 ∧ 2*a + 3 = 2*c + 5 ∧ 2*a + 3 = 2*d + 6 ∧ 2*a + 3 = a + b + c + d + 10) : 
  a + b + c + d = -11 := by
sorry

end NUMINAMATH_CALUDE_sum_abcd_equals_negative_eleven_l3354_335469


namespace NUMINAMATH_CALUDE_physical_fitness_test_probability_l3354_335445

theorem physical_fitness_test_probability 
  (total_students : ℕ) 
  (male_students : ℕ) 
  (female_students : ℕ) 
  (selected_students : ℕ) :
  total_students = male_students + female_students →
  male_students = 3 →
  female_students = 2 →
  selected_students = 2 →
  (Nat.choose male_students 1 * Nat.choose female_students 1) / 
  Nat.choose total_students selected_students = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_physical_fitness_test_probability_l3354_335445


namespace NUMINAMATH_CALUDE_rhombus_area_fraction_l3354_335432

/-- Represents a point on a 2D grid -/
structure Point where
  x : Int
  y : Int

/-- Represents a rhombus on a 2D grid -/
structure Rhombus where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Calculates the area of a rhombus given its vertices -/
def rhombusArea (r : Rhombus) : ℚ :=
  1 -- placeholder for the actual calculation

/-- Calculates the area of a square grid -/
def gridArea (side : ℕ) : ℕ :=
  side * side

/-- The main theorem to prove -/
theorem rhombus_area_fraction :
  let r : Rhombus := {
    v1 := { x := 3, y := 2 },
    v2 := { x := 4, y := 3 },
    v3 := { x := 3, y := 4 },
    v4 := { x := 2, y := 3 }
  }
  let gridSide : ℕ := 6
  rhombusArea r / gridArea gridSide = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_fraction_l3354_335432


namespace NUMINAMATH_CALUDE_total_candies_l3354_335439

/-- The total number of candies in a jar, given the number of red and blue candies -/
theorem total_candies (red : ℕ) (blue : ℕ) (h1 : red = 145) (h2 : blue = 3264) : 
  red + blue = 3409 :=
by sorry

end NUMINAMATH_CALUDE_total_candies_l3354_335439


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3354_335403

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 - 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 - 1 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3354_335403


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3354_335417

/-- Represents a tetrahedron with vertices P, Q, R, and S. -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths. -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 3√2. -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := 4,
    RS := (15/4) * Real.sqrt 2
  }
  tetrahedronVolume t = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3354_335417


namespace NUMINAMATH_CALUDE_product_mod_twenty_l3354_335410

theorem product_mod_twenty : (53 * 76 * 91) % 20 = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_twenty_l3354_335410


namespace NUMINAMATH_CALUDE_center_numbers_l3354_335487

/-- Represents a 4x4 grid of integers -/
def Grid := Fin 4 → Fin 4 → ℕ

/-- Checks if two positions in the grid share an edge -/
def sharesEdge (p1 p2 : Fin 4 × Fin 4) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Checks if a grid satisfies the conditions of the problem -/
def validGrid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 17) ∧
  (∀ n : ℕ, n ∈ Finset.range 16 → ∃ p1 p2, g p1.1 p1.2 = n ∧ g p2.1 p2.2 = n + 1 ∧ sharesEdge p1 p2) ∧
  (g 0 0 + g 0 3 + g 3 0 + g 3 3 = 34)

/-- The center 2x2 grid -/
def centerGrid (g : Grid) : Finset ℕ :=
  {g 1 1, g 1 2, g 2 1, g 2 2}

theorem center_numbers (g : Grid) (h : validGrid g) :
  centerGrid g = {9, 10, 11, 12} :=
sorry

end NUMINAMATH_CALUDE_center_numbers_l3354_335487


namespace NUMINAMATH_CALUDE_vaccine_cost_l3354_335419

theorem vaccine_cost (num_vaccines : ℕ) (doctor_visit_cost : ℝ) 
  (insurance_coverage : ℝ) (trip_cost : ℝ) (total_payment : ℝ) :
  num_vaccines = 10 ∧ 
  doctor_visit_cost = 250 ∧ 
  insurance_coverage = 0.8 ∧ 
  trip_cost = 1200 ∧ 
  total_payment = 1340 →
  (total_payment - trip_cost - (1 - insurance_coverage) * doctor_visit_cost) / 
  ((1 - insurance_coverage) * num_vaccines) = 45 := by
  sorry

end NUMINAMATH_CALUDE_vaccine_cost_l3354_335419


namespace NUMINAMATH_CALUDE_min_value_a_l3354_335490

theorem min_value_a (p : Prop) (h : ¬∀ x > 0, a < x + 1/x) : 
  ∃ a : ℝ, (∀ b : ℝ, (∃ x > 0, b ≥ x + 1/x) → a ≤ b) ∧ (∃ x > 0, a ≥ x + 1/x) ∧ a = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l3354_335490


namespace NUMINAMATH_CALUDE_phone_profit_optimization_l3354_335437

/-- Represents the profit calculation and optimization problem for two types of phones. -/
theorem phone_profit_optimization
  (profit_A_B : ℕ → ℕ → ℝ)
  (total_phones : ℕ)
  (h1 : profit_A_B 1 1 = 600)
  (h2 : profit_A_B 3 2 = 1400)
  (h3 : total_phones = 20)
  (h4 : ∀ x y, x + y = total_phones → y ≤ 2 / 3 * x) :
  ∃ (x y : ℕ),
    x + y = total_phones ∧
    y ≤ 2 / 3 * x ∧
    ∀ (a b : ℕ), a + b = total_phones → a ≥ 0 → b ≥ 0 →
      profit_A_B x y ≥ profit_A_B a b ∧
      profit_A_B x y = 5600 :=
by sorry

end NUMINAMATH_CALUDE_phone_profit_optimization_l3354_335437


namespace NUMINAMATH_CALUDE_a_3_equals_zero_l3354_335404

theorem a_3_equals_zero (a : ℕ → ℝ) (h : ∀ n, a n = Real.sin (n * π / 3)) : a 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_3_equals_zero_l3354_335404


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3354_335452

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = 2024*x ↔ x = 0 ∨ x = 2024 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3354_335452


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3354_335406

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 4 * x + 2 = 0) ↔ k ≤ 2 ∧ k ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3354_335406


namespace NUMINAMATH_CALUDE_pentagon_arrangement_exists_l3354_335453

/-- Represents a pentagon arrangement of natural numbers -/
def PentagonArrangement := Fin 5 → ℕ

/-- Checks if two numbers are coprime -/
def are_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- Checks if two numbers have a common divisor greater than 1 -/
def have_common_divisor (a b : ℕ) : Prop := ∃ d : ℕ, d > 1 ∧ d ∣ a ∧ d ∣ b

/-- Checks if the given arrangement satisfies the conditions -/
def is_valid_arrangement (arr : PentagonArrangement) : Prop :=
  (∀ i : Fin 5, are_coprime (arr i) (arr ((i + 1) % 5))) ∧
  (∀ i : Fin 5, have_common_divisor (arr i) (arr ((i + 2) % 5)))

/-- The main theorem: there exists a valid pentagon arrangement -/
theorem pentagon_arrangement_exists : ∃ arr : PentagonArrangement, is_valid_arrangement arr :=
sorry

end NUMINAMATH_CALUDE_pentagon_arrangement_exists_l3354_335453


namespace NUMINAMATH_CALUDE_swim_time_calculation_l3354_335460

/-- 
Given a person's swimming speed in still water, the speed of the water current,
and the time taken to swim with the current for a certain distance,
calculate the time taken to swim back against the current for the same distance.
-/
theorem swim_time_calculation (still_speed water_speed with_current_time : ℝ) 
  (still_speed_pos : still_speed > 0)
  (water_speed_pos : water_speed > 0)
  (with_current_time_pos : with_current_time > 0)
  (h_still_speed : still_speed = 16)
  (h_water_speed : water_speed = 8)
  (h_with_current_time : with_current_time = 1.5) :
  let against_current_speed := still_speed - water_speed
  let with_current_speed := still_speed + water_speed
  let distance := with_current_speed * with_current_time
  let against_current_time := distance / against_current_speed
  against_current_time = 4.5 := by
sorry

end NUMINAMATH_CALUDE_swim_time_calculation_l3354_335460


namespace NUMINAMATH_CALUDE_sum_distinct_prime_factors_of_7_power_difference_l3354_335441

theorem sum_distinct_prime_factors_of_7_power_difference : 
  (Finset.sum (Finset.filter (Nat.Prime) (Finset.range ((7^7 - 7^4).factors.toFinset.card + 1)))
    (λ p => if p ∈ (7^7 - 7^4).factors.toFinset then p else 0)) = 31 := by sorry

end NUMINAMATH_CALUDE_sum_distinct_prime_factors_of_7_power_difference_l3354_335441


namespace NUMINAMATH_CALUDE_alloy_combination_theorem_l3354_335476

/-- Represents the composition of an alloy --/
structure AlloyComposition where
  copper : ℝ
  zinc : ℝ

/-- The first alloy composition --/
def firstAlloy : AlloyComposition :=
  { copper := 2, zinc := 1 }

/-- The second alloy composition --/
def secondAlloy : AlloyComposition :=
  { copper := 1, zinc := 5 }

/-- Combines two alloys in a given ratio --/
def combineAlloys (a1 a2 : AlloyComposition) (r1 r2 : ℝ) : AlloyComposition :=
  { copper := r1 * a1.copper + r2 * a2.copper
  , zinc := r1 * a1.zinc + r2 * a2.zinc }

/-- The theorem to be proved --/
theorem alloy_combination_theorem :
  let combinedAlloy := combineAlloys firstAlloy secondAlloy 1 2
  combinedAlloy.zinc = 2 * combinedAlloy.copper := by
  sorry

end NUMINAMATH_CALUDE_alloy_combination_theorem_l3354_335476


namespace NUMINAMATH_CALUDE_bound_step_difference_is_10_l3354_335405

/-- The number of steps Martha takes between consecutive lamp posts -/
def martha_steps : ℕ := 50

/-- The number of bounds Percy takes between consecutive lamp posts -/
def percy_bounds : ℕ := 15

/-- The total number of lamp posts -/
def total_posts : ℕ := 51

/-- The distance between the first and last lamp post in feet -/
def total_distance : ℕ := 10560

/-- The difference between Percy's bound length and Martha's step length in feet -/
def bound_step_difference : ℚ := 10

theorem bound_step_difference_is_10 :
  (total_distance : ℚ) / ((total_posts - 1) * percy_bounds) -
  (total_distance : ℚ) / ((total_posts - 1) * martha_steps) =
  bound_step_difference := by sorry

end NUMINAMATH_CALUDE_bound_step_difference_is_10_l3354_335405


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_eight_l3354_335409

theorem reciprocal_of_negative_eight :
  ∃ x : ℚ, x * (-8) = 1 ∧ x = -1/8 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_eight_l3354_335409


namespace NUMINAMATH_CALUDE_pencil_distribution_l3354_335407

/-- Given:
  - total_pencils: The total number of pencils
  - original_classes: The original number of classes
  - remaining_pencils: The number of pencils remaining after distribution
  - pencil_difference: The difference in pencils per class compared to the original plan
  This theorem proves that the actual number of classes is 11
-/
theorem pencil_distribution
  (total_pencils : ℕ)
  (original_classes : ℕ)
  (remaining_pencils : ℕ)
  (pencil_difference : ℕ)
  (h1 : total_pencils = 172)
  (h2 : original_classes = 4)
  (h3 : remaining_pencils = 7)
  (h4 : pencil_difference = 28)
  : ∃ (actual_classes : ℕ),
    actual_classes > original_classes ∧
    (total_pencils - remaining_pencils) / actual_classes + pencil_difference = total_pencils / original_classes ∧
    actual_classes = 11 :=
sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3354_335407


namespace NUMINAMATH_CALUDE_power_nap_duration_l3354_335496

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

/-- Represents one fourth of an hour -/
def quarter_hour : ℚ := 1 / 4

theorem power_nap_duration :
  hours_to_minutes quarter_hour = 15 := by sorry

end NUMINAMATH_CALUDE_power_nap_duration_l3354_335496


namespace NUMINAMATH_CALUDE_convex_pentagon_probability_l3354_335422

/-- Given seven points on a circle -/
def num_points : ℕ := 7

/-- Number of chords that can be formed from seven points -/
def total_chords : ℕ := num_points.choose 2

/-- Number of chords selected -/
def selected_chords : ℕ := 5

/-- The probability of forming a convex pentagon -/
def probability : ℚ := (num_points.choose selected_chords) / (total_chords.choose selected_chords)

/-- Theorem: The probability of forming a convex pentagon by randomly selecting
    five chords from seven points on a circle is 1/969 -/
theorem convex_pentagon_probability : probability = 1 / 969 := by
  sorry

end NUMINAMATH_CALUDE_convex_pentagon_probability_l3354_335422


namespace NUMINAMATH_CALUDE_johns_age_difference_l3354_335492

theorem johns_age_difference (brother_age : ℕ) (john_age : ℕ) : 
  brother_age = 8 → 
  john_age + brother_age = 10 → 
  6 * brother_age - john_age = 46 := by
sorry

end NUMINAMATH_CALUDE_johns_age_difference_l3354_335492


namespace NUMINAMATH_CALUDE_prob_three_same_suit_standard_deck_l3354_335401

/-- A standard deck of cards --/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)
  (black_suits : Nat)
  (red_suits : Nat)

/-- Standard 52-card deck --/
def standard_deck : Deck :=
  { cards := 52,
    ranks := 13,
    suits := 4,
    black_suits := 2,
    red_suits := 2 }

/-- Probability of drawing three cards of the same specific suit --/
def prob_three_same_suit (d : Deck) : Rat :=
  (d.ranks * (d.ranks - 1) * (d.ranks - 2)) / (d.cards * (d.cards - 1) * (d.cards - 2))

/-- Theorem stating the probability of drawing three cards of the same specific suit from a standard deck --/
theorem prob_three_same_suit_standard_deck :
  prob_three_same_suit standard_deck = 11 / 850 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_same_suit_standard_deck_l3354_335401


namespace NUMINAMATH_CALUDE_tangent_line_implies_b_equals_3_l3354_335484

/-- A line y = kx + 1 is tangent to the curve y = x^3 + ax + b at the point (1, 3) -/
def is_tangent (k a b : ℝ) : Prop :=
  -- The point (1, 3) lies on both the line and the curve
  3 = k * 1 + 1 ∧ 3 = 1^3 + a * 1 + b ∧
  -- The slopes of the line and the curve are equal at (1, 3)
  k = 3 * 1^2 + a

theorem tangent_line_implies_b_equals_3 (k a b : ℝ) :
  is_tangent k a b → b = 3 := by
  sorry

#check tangent_line_implies_b_equals_3

end NUMINAMATH_CALUDE_tangent_line_implies_b_equals_3_l3354_335484


namespace NUMINAMATH_CALUDE_mike_initial_cards_l3354_335493

/-- The number of baseball cards Mike has initially -/
def initial_cards : ℕ := sorry

/-- The number of baseball cards Sam gave to Mike -/
def cards_from_sam : ℕ := 13

/-- The total number of baseball cards Mike has after receiving cards from Sam -/
def total_cards : ℕ := 100

/-- Theorem stating that Mike initially had 87 baseball cards -/
theorem mike_initial_cards : initial_cards = 87 := by
  sorry

end NUMINAMATH_CALUDE_mike_initial_cards_l3354_335493


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l3354_335472

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 100) 
  (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l3354_335472


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3354_335480

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = -2) :
  (1 + 1 / (a - 1)) / (2 * a / (a^2 - 1)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3354_335480


namespace NUMINAMATH_CALUDE_license_plate_increase_l3354_335411

def old_plates : ℕ := 26^3 * 10^3
def new_plates_a : ℕ := 26^2 * 10^4
def new_plates_b : ℕ := 26^4 * 10^2
def avg_new_plates : ℚ := (new_plates_a + new_plates_b) / 2

theorem license_plate_increase : 
  (avg_new_plates : ℚ) / old_plates = 468 / 10 := by sorry

end NUMINAMATH_CALUDE_license_plate_increase_l3354_335411


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l3354_335464

def f (x : ℝ) := 4 * x - x^3

theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (0 : ℝ) 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (0 : ℝ) 2, f x = max) ∧
    (∀ x ∈ Set.Icc (0 : ℝ) 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (0 : ℝ) 2, f x = min) ∧
    max = 16 * Real.sqrt 3 / 9 ∧
    min = 0 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l3354_335464


namespace NUMINAMATH_CALUDE_sum_of_roots_l3354_335424

theorem sum_of_roots (k c : ℝ) (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂)
  (h₁ : 6 * x₁^2 - k * x₁ = c)
  (h₂ : 6 * x₂^2 - k * x₂ = c) :
  x₁ + x₂ = k / 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3354_335424


namespace NUMINAMATH_CALUDE_not_a_gt_b_l3354_335408

theorem not_a_gt_b (A B : ℕ) (hA : A ≠ 0) (hB : B ≠ 0) 
  (h : A / (1 / 5) = B * (1 / 4)) : A ≤ B := by
  sorry

end NUMINAMATH_CALUDE_not_a_gt_b_l3354_335408


namespace NUMINAMATH_CALUDE_class_average_problem_l3354_335426

theorem class_average_problem (n : ℝ) (h1 : n > 0) :
  let total_average : ℝ := 80
  let quarter_average : ℝ := 92
  let quarter_sum : ℝ := quarter_average * (n / 4)
  let total_sum : ℝ := total_average * n
  let rest_sum : ℝ := total_sum - quarter_sum
  let rest_average : ℝ := rest_sum / (3 * n / 4)
  rest_average = 76 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l3354_335426


namespace NUMINAMATH_CALUDE_determinant_zero_l3354_335433

theorem determinant_zero (α β : Real) : 
  let M : Matrix (Fin 3) (Fin 3) Real := ![![0, Real.cos α, -Real.sin α],
                                           ![-Real.cos α, 0, Real.cos β],
                                           ![Real.sin α, -Real.cos β, 0]]
  Matrix.det M = 0 := by
sorry

end NUMINAMATH_CALUDE_determinant_zero_l3354_335433


namespace NUMINAMATH_CALUDE_stating_three_plane_division_l3354_335494

/-- Represents the possible numbers of parts that three planes can divide 3D space into -/
inductive PlaneDivision : Type
  | four : PlaneDivision
  | six : PlaneDivision
  | seven : PlaneDivision
  | eight : PlaneDivision

/-- Represents a configuration of three planes in 3D space -/
structure ThreePlaneConfiguration where
  -- Add necessary fields to represent the configuration

/-- 
Given a configuration of three planes in 3D space, 
returns the number of parts the space is divided into
-/
def countParts (config : ThreePlaneConfiguration) : PlaneDivision :=
  sorry

/-- 
Theorem stating that three planes can only divide 3D space into 4, 6, 7, or 8 parts,
and all these cases are possible
-/
theorem three_plane_division :
  (∀ config : ThreePlaneConfiguration, ∃ n : PlaneDivision, countParts config = n) ∧
  (∀ n : PlaneDivision, ∃ config : ThreePlaneConfiguration, countParts config = n) :=
sorry

end NUMINAMATH_CALUDE_stating_three_plane_division_l3354_335494


namespace NUMINAMATH_CALUDE_grid_completion_count_l3354_335448

/-- Represents a 2x3 grid with one fixed R and 5 remaining squares --/
def Grid := Fin 2 → Fin 3 → Fin 3

/-- Checks if two adjacent cells in the grid have the same value --/
def has_adjacent_match (g : Grid) : Prop :=
  ∃ i j, (g i j = g i (j + 1)) ∨ 
         (g i j = g (i + 1) j)

/-- The number of ways to fill the grid --/
def total_configurations : ℕ := 3^5

/-- The number of valid configurations without adjacent matches --/
def valid_configurations : ℕ := 18

theorem grid_completion_count :
  (total_configurations - valid_configurations : ℕ) = 225 :=
sorry

end NUMINAMATH_CALUDE_grid_completion_count_l3354_335448


namespace NUMINAMATH_CALUDE_M_closed_under_multiplication_l3354_335443

def M : Set ℕ := {n | ∃ m : ℕ, m > 0 ∧ n = m^2}

theorem M_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ M → b ∈ M → (a * b) ∈ M :=
by
  sorry

end NUMINAMATH_CALUDE_M_closed_under_multiplication_l3354_335443


namespace NUMINAMATH_CALUDE_sugar_box_surface_area_l3354_335482

theorem sugar_box_surface_area :
  ∀ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 →
    a * b * c = 280 →
    2 * (a * b + b * c + c * a) = 262 :=
by
  sorry

end NUMINAMATH_CALUDE_sugar_box_surface_area_l3354_335482


namespace NUMINAMATH_CALUDE_train_overtake_time_l3354_335475

/-- The time it takes for a train to overtake a motorbike -/
theorem train_overtake_time (train_speed motorbike_speed train_length : ℝ) 
  (h1 : train_speed = 100)
  (h2 : motorbike_speed = 64)
  (h3 : train_length = 850.068) : 
  (train_length / ((train_speed - motorbike_speed) * (1000 / 3600))) = 85.0068 := by
  sorry

end NUMINAMATH_CALUDE_train_overtake_time_l3354_335475


namespace NUMINAMATH_CALUDE_monogram_count_l3354_335455

/-- The number of letters in the alphabet before 'G' -/
def letters_before_g : Nat := 6

/-- The number of letters in the alphabet after 'G' -/
def letters_after_g : Nat := 18

/-- The total number of possible monograms with 'G' as the middle initial,
    and the other initials different and in alphabetical order -/
def total_monograms : Nat := letters_before_g * letters_after_g

theorem monogram_count :
  total_monograms = 108 := by
  sorry

end NUMINAMATH_CALUDE_monogram_count_l3354_335455
