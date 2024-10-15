import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_cos_a_l2774_277467

theorem right_triangle_cos_a (A B C : Real) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) (h3 : 0 < C ∧ C < π/2) : 
  B = π/2 → 3 * Real.tan A = 4 * Real.sin A → Real.cos A = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cos_a_l2774_277467


namespace NUMINAMATH_CALUDE_commission_percentage_is_21_875_l2774_277457

/-- Calculates the commission percentage for a sale with given rates and total amount -/
def commission_percentage (rate_below_500 : ℚ) (rate_above_500 : ℚ) (total_amount : ℚ) : ℚ :=
  let commission_below_500 := min total_amount 500 * rate_below_500
  let commission_above_500 := max (total_amount - 500) 0 * rate_above_500
  let total_commission := commission_below_500 + commission_above_500
  (total_commission / total_amount) * 100

/-- Theorem stating that the commission percentage for the given problem is 21.875% -/
theorem commission_percentage_is_21_875 :
  commission_percentage (20 / 100) (25 / 100) 800 = 21875 / 1000 := by sorry

end NUMINAMATH_CALUDE_commission_percentage_is_21_875_l2774_277457


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2774_277453

def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}
def B : Set ℤ := {x | ∃ k : ℕ, x = 2 * k + 1 ∧ k < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2774_277453


namespace NUMINAMATH_CALUDE_percentage_of_boys_from_school_A_l2774_277409

theorem percentage_of_boys_from_school_A (total_boys : ℕ) 
  (boys_A_not_science : ℕ) (science_percentage : ℚ) :
  total_boys = 400 →
  boys_A_not_science = 56 →
  science_percentage = 30 / 100 →
  (boys_A_not_science : ℚ) / ((1 - science_percentage) * total_boys) = 20 / 100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_boys_from_school_A_l2774_277409


namespace NUMINAMATH_CALUDE_relation_xyz_l2774_277459

theorem relation_xyz (x y z t : ℝ) (h : x / Real.sin t = y / Real.sin (2 * t) ∧ 
                                        x / Real.sin t = z / Real.sin (3 * t)) : 
  x^2 - y^2 + x*z = 0 := by
sorry

end NUMINAMATH_CALUDE_relation_xyz_l2774_277459


namespace NUMINAMATH_CALUDE_twelfth_day_is_monday_l2774_277432

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific conditions -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  fridayCount : Nat
  dayCount : Nat
  firstNotFriday : firstDay ≠ DayOfWeek.Friday
  lastNotFriday : lastDay ≠ DayOfWeek.Friday
  exactlyFiveFridays : fridayCount = 5
  validDayCount : dayCount ∈ [28, 29, 30, 31]

/-- Function to determine the day of week for a given day number -/
def dayOfWeekForDay (m : Month) (day : Nat) : DayOfWeek :=
  sorry

theorem twelfth_day_is_monday (m : Month) : 
  dayOfWeekForDay m 12 = DayOfWeek.Monday :=
sorry

end NUMINAMATH_CALUDE_twelfth_day_is_monday_l2774_277432


namespace NUMINAMATH_CALUDE_sum_last_three_coefficients_eq_21_l2774_277458

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function that calculates the sum of the last three coefficients
def sum_last_three_coefficients (a : ℝ) : ℝ :=
  binomial 8 0 * 1 + binomial 8 1 * (-1) + binomial 8 2 * 1

-- Theorem statement
theorem sum_last_three_coefficients_eq_21 :
  ∀ a : ℝ, sum_last_three_coefficients a = 21 := by sorry

end NUMINAMATH_CALUDE_sum_last_three_coefficients_eq_21_l2774_277458


namespace NUMINAMATH_CALUDE_tangent_line_slope_l2774_277494

/-- The slope of a line tangent to a circle --/
theorem tangent_line_slope (k : ℝ) : 
  (∀ x y : ℝ, y = k * (x - 2) + 2 → x^2 + y^2 - 2*x - 2*y = 0 → 
   ∃! x y : ℝ, y = k * (x - 2) + 2 ∧ x^2 + y^2 - 2*x - 2*y = 0) → 
  k = -1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_l2774_277494


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2774_277462

open Real

theorem trigonometric_identity (α β : ℝ) 
  (h1 : sin (π - α) - 2 * sin ((π / 2) + α) = 0) 
  (h2 : tan (α + β) = -1) : 
  (sin α * cos α + sin α ^ 2 = 6 / 5) ∧ 
  (tan β = 3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2774_277462


namespace NUMINAMATH_CALUDE_find_number_l2774_277422

theorem find_number : ∃ x : ℝ, x / 2 = 9 ∧ x = 18 := by sorry

end NUMINAMATH_CALUDE_find_number_l2774_277422


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l2774_277491

def quadratic_function (x : ℝ) : ℝ := (x - 1)^2

theorem y1_less_than_y2 (y₁ y₂ : ℝ) :
  quadratic_function (-1) = y₁ →
  quadratic_function 4 = y₂ →
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l2774_277491


namespace NUMINAMATH_CALUDE_benches_around_circular_track_l2774_277440

/-- The radius of the circular walking track in feet -/
def radius : ℝ := 15

/-- The spacing between benches in feet -/
def bench_spacing : ℝ := 3

/-- The number of benches needed for the circular track -/
def num_benches : ℕ := 31

/-- Theorem stating that the number of benches needed is approximately 31 -/
theorem benches_around_circular_track :
  Int.floor ((2 * Real.pi * radius) / bench_spacing) = num_benches := by
  sorry

end NUMINAMATH_CALUDE_benches_around_circular_track_l2774_277440


namespace NUMINAMATH_CALUDE_average_salary_non_officers_l2774_277482

/-- Proof of the average salary of non-officers in an office --/
theorem average_salary_non_officers
  (total_avg : ℝ)
  (officer_avg : ℝ)
  (officer_count : ℕ)
  (non_officer_count : ℕ)
  (h_total_avg : total_avg = 120)
  (h_officer_avg : officer_avg = 430)
  (h_officer_count : officer_count = 15)
  (h_non_officer_count : non_officer_count = 465) :
  let non_officer_avg := (((officer_count + non_officer_count) * total_avg) - (officer_count * officer_avg)) / non_officer_count
  non_officer_avg = 110 := by
sorry


end NUMINAMATH_CALUDE_average_salary_non_officers_l2774_277482


namespace NUMINAMATH_CALUDE_negation_of_existence_l2774_277423

theorem negation_of_existence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l2774_277423


namespace NUMINAMATH_CALUDE_rug_inner_length_is_four_l2774_277424

/-- Represents the dimensions of a rectangular region -/
structure RectDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular region -/
def area (dim : RectDimensions) : ℝ := dim.length * dim.width

/-- Represents the three regions of the rug -/
structure RugRegions where
  inner : RectDimensions
  middle : RectDimensions
  outer : RectDimensions

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem rug_inner_length_is_four
  (rug : RugRegions)
  (inner_width_two : rug.inner.width = 2)
  (middle_wider_by_two : rug.middle.length = rug.inner.length + 4 ∧ rug.middle.width = rug.inner.width + 4)
  (outer_wider_by_two : rug.outer.length = rug.middle.length + 4 ∧ rug.outer.width = rug.middle.width + 4)
  (areas_in_arithmetic_progression : isArithmeticProgression (area rug.inner) (area rug.middle) (area rug.outer)) :
  rug.inner.length = 4 := by
  sorry

end NUMINAMATH_CALUDE_rug_inner_length_is_four_l2774_277424


namespace NUMINAMATH_CALUDE_max_value_theorem_l2774_277490

theorem max_value_theorem (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 25) :
  (Real.sqrt (x + 64) + 2 * Real.sqrt (25 - x) + Real.sqrt x) ≤ Real.sqrt 328 ∧
  ∃ x₀, 0 ≤ x₀ ∧ x₀ ≤ 25 ∧ Real.sqrt (x₀ + 64) + 2 * Real.sqrt (25 - x₀) + Real.sqrt x₀ = Real.sqrt 328 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2774_277490


namespace NUMINAMATH_CALUDE_legs_more_than_twice_heads_l2774_277477

-- Define the group of animals
structure AnimalGroup where
  donkeys : ℕ
  pigs : ℕ

-- Define the properties of the group
def AnimalGroup.heads (g : AnimalGroup) : ℕ := g.donkeys + g.pigs
def AnimalGroup.legs (g : AnimalGroup) : ℕ := 4 * g.donkeys + 4 * g.pigs

-- Theorem statement
theorem legs_more_than_twice_heads (g : AnimalGroup) (h : g.donkeys = 8) :
  g.legs ≥ 2 * g.heads + 16 := by
  sorry

end NUMINAMATH_CALUDE_legs_more_than_twice_heads_l2774_277477


namespace NUMINAMATH_CALUDE_min_a_for_parabola_l2774_277404

/-- Given a parabola y = ax^2 + bx + c with vertex at (1/4, -9/8), 
    where a > 0 and a + b + c is an integer, 
    the minimum possible value of a is 2/9 -/
theorem min_a_for_parabola (a b c : ℝ) : 
  a > 0 ∧ 
  (∃ k : ℤ, a + b + c = k) ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = a * (x - 1/4)^2 - 9/8) → 
  (∀ a' : ℝ, a' > 0 ∧ 
    (∃ b' c' : ℝ, (∃ k : ℤ, a' + b' + c' = k) ∧ 
      (∀ x : ℝ, a' * x^2 + b' * x + c' = a' * (x - 1/4)^2 - 9/8)) → 
    a' ≥ 2/9) ∧ 
  a = 2/9 := by
sorry

end NUMINAMATH_CALUDE_min_a_for_parabola_l2774_277404


namespace NUMINAMATH_CALUDE_lizas_paycheck_amount_l2774_277443

/-- Calculates the amount of Liza's paycheck given her initial balance, expenses, and final balance -/
def calculate_paycheck (initial_balance rent electricity internet phone final_balance : ℕ) : ℕ :=
  final_balance + rent + electricity + internet + phone - initial_balance

/-- Theorem stating that Liza's paycheck is $1563 given the provided financial information -/
theorem lizas_paycheck_amount :
  calculate_paycheck 800 450 117 100 70 1563 = 1563 := by
  sorry

end NUMINAMATH_CALUDE_lizas_paycheck_amount_l2774_277443


namespace NUMINAMATH_CALUDE_scale_drawing_conversion_l2774_277486

/-- Given a scale where 1 inch represents 1000 feet, 
    a line segment of 3.6 inches represents 3600 feet. -/
theorem scale_drawing_conversion (scale : ℝ) (drawing_length : ℝ) :
  scale = 1000 →
  drawing_length = 3.6 →
  drawing_length * scale = 3600 := by
  sorry

end NUMINAMATH_CALUDE_scale_drawing_conversion_l2774_277486


namespace NUMINAMATH_CALUDE_intersection_A_B_l2774_277450

def A : Set ℝ := {-3, -1, 1, 2}
def B : Set ℝ := { x : ℝ | x ≥ 0 }

theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2774_277450


namespace NUMINAMATH_CALUDE_victor_books_left_l2774_277448

def book_count (initial bought gifted donated : ℕ) : ℕ :=
  initial + bought - gifted - donated

theorem victor_books_left : book_count 25 12 7 15 = 15 := by
  sorry

end NUMINAMATH_CALUDE_victor_books_left_l2774_277448


namespace NUMINAMATH_CALUDE_square_side_length_from_circle_area_l2774_277451

/-- Given a square from which a circle is described, if the area of the circle is 78.53981633974483 square inches, then the side length of the square is 10 inches. -/
theorem square_side_length_from_circle_area (circle_area : ℝ) (square_side : ℝ) : 
  circle_area = 78.53981633974483 →
  circle_area = Real.pi * (square_side / 2)^2 →
  square_side = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_from_circle_area_l2774_277451


namespace NUMINAMATH_CALUDE_license_plate_theorem_l2774_277445

def license_plate_combinations : ℕ :=
  let alphabet_size : ℕ := 26
  let digit_size : ℕ := 10
  let letter_positions : ℕ := 4
  let digit_positions : ℕ := 2

  let choose_repeated_letter := alphabet_size
  let choose_distinct_letters := Nat.choose (alphabet_size - 1) 2
  let place_repeated_letter := Nat.choose letter_positions 2
  let arrange_nonrepeated_letters := 2

  let letter_combinations := 
    choose_repeated_letter * choose_distinct_letters * place_repeated_letter * arrange_nonrepeated_letters

  let digit_combinations := digit_size ^ digit_positions

  letter_combinations * digit_combinations

theorem license_plate_theorem : license_plate_combinations = 936000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l2774_277445


namespace NUMINAMATH_CALUDE_complex_sum_of_parts_l2774_277489

theorem complex_sum_of_parts (z : ℂ) (h : (1 + 2*I)*z = 4 + 3*I) :
  z.re + z.im = 1 := by sorry

end NUMINAMATH_CALUDE_complex_sum_of_parts_l2774_277489


namespace NUMINAMATH_CALUDE_mersenne_primes_less_than_1000_are_3_7_31_127_l2774_277483

def is_mersenne_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ n : ℕ, Nat.Prime n ∧ p = 2^n - 1

def mersenne_primes_less_than_1000 : Set ℕ :=
  {p : ℕ | is_mersenne_prime p ∧ p < 1000}

theorem mersenne_primes_less_than_1000_are_3_7_31_127 :
  mersenne_primes_less_than_1000 = {3, 7, 31, 127} :=
by sorry

end NUMINAMATH_CALUDE_mersenne_primes_less_than_1000_are_3_7_31_127_l2774_277483


namespace NUMINAMATH_CALUDE_periodic_function_property_l2774_277487

/-- Given a function f(x) = a*sin(π*x + α) + b*cos(π*x + β) + 4,
    where a, b, α, β are non-zero real numbers,
    if f(2011) = 5, then f(2012) = 3 -/
theorem periodic_function_property (a b α β : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hα : α ≠ 0) (hβ : β ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4
  (f 2011 = 5) → (f 2012 = 3) := by sorry

end NUMINAMATH_CALUDE_periodic_function_property_l2774_277487


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l2774_277402

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h : is_increasing f) : f (a^2 + 1) > f a := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l2774_277402


namespace NUMINAMATH_CALUDE_max_automobile_weight_l2774_277428

/-- Represents the capacity of the ferry in tons -/
def ferry_capacity : ℝ := 50

/-- Represents the maximum number of automobiles the ferry can carry -/
def max_automobiles : ℝ := 62.5

/-- Represents the conversion factor from tons to pounds -/
def tons_to_pounds : ℝ := 2000

/-- Theorem stating that the maximum weight of an automobile is 1600 pounds -/
theorem max_automobile_weight :
  (ferry_capacity * tons_to_pounds) / max_automobiles = 1600 := by
  sorry

end NUMINAMATH_CALUDE_max_automobile_weight_l2774_277428


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2774_277401

theorem quadratic_equation_result (a : ℝ) (h : a^2 + 3*a - 4 = 0) : 2*a^2 + 6*a - 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2774_277401


namespace NUMINAMATH_CALUDE_probability_factor_90_less_than_8_l2774_277498

def positive_factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => x > 0 ∧ n % x = 0) (Finset.range (n + 1))

def factors_less_than (n k : ℕ) : Finset ℕ :=
  Finset.filter (λ x => x < k) (positive_factors n)

theorem probability_factor_90_less_than_8 :
  (Finset.card (factors_less_than 90 8) : ℚ) / (Finset.card (positive_factors 90) : ℚ) = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_factor_90_less_than_8_l2774_277498


namespace NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l2774_277430

/-- A line in the 2D plane represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line. -/
def yIntercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Two lines are parallel if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line (b : Line) :
  parallel b { slope := 3, point := (0, -6) } →
  b.point = (1, 2) →
  yIntercept b = -1 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_parallel_line_l2774_277430


namespace NUMINAMATH_CALUDE_complement_of_M_l2774_277488

def M : Set ℝ := {x | (1 + x) / (1 - x) > 0}

theorem complement_of_M : 
  (Set.univ : Set ℝ) \ M = {x | x ≤ -1 ∨ x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l2774_277488


namespace NUMINAMATH_CALUDE_standard_of_living_purchasing_power_correlated_l2774_277484

/-- Represents a person's standard of living -/
def StandardOfLiving : Type := ℝ

/-- Represents a person's purchasing power -/
def PurchasingPower : Type := ℝ

/-- Definition of correlation as a statistical relationship between two random variables -/
def Correlated (X Y : Type) : Prop := sorry

/-- Theorem stating that standard of living and purchasing power are correlated -/
theorem standard_of_living_purchasing_power_correlated :
  Correlated StandardOfLiving PurchasingPower :=
sorry

end NUMINAMATH_CALUDE_standard_of_living_purchasing_power_correlated_l2774_277484


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2774_277480

theorem consecutive_integers_sum (x : ℤ) :
  x * (x + 1) * (x + 2) = 2730 → x + (x + 1) + (x + 2) = 42 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2774_277480


namespace NUMINAMATH_CALUDE_exponent_and_logarithm_equalities_l2774_277425

theorem exponent_and_logarithm_equalities :
  (3 : ℝ) ^ 64 = 4 ∧ (4 : ℝ) ^ (Real.log 3 / Real.log 2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_exponent_and_logarithm_equalities_l2774_277425


namespace NUMINAMATH_CALUDE_circle_P_radius_l2774_277475

-- Define the circles and points
def circle_N (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 15 = 0
def point_M : ℝ × ℝ := (-1, 0)

-- Define the curve τ
def curve_τ (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*(x + 1) ∧ k > 0

-- Define the theorem
theorem circle_P_radius : 
  ∃ (x_P y_P r : ℝ),
  -- Circle P passes through M
  (x_P + 1)^2 + y_P^2 = r^2 ∧
  -- Circle P is internally tangent to N
  ∃ (x_N y_N : ℝ), circle_N x_N y_N ∧ ((x_P - x_N)^2 + (y_P - y_N)^2 = (4 - r)^2) ∧
  -- Center of P is on curve τ
  curve_τ x_P y_P ∧
  -- Line l is tangent to P and intersects τ
  ∃ (k x_A y_A x_B y_B : ℝ),
    line_l k x_A y_A ∧ line_l k x_B y_B ∧
    curve_τ x_A y_A ∧ curve_τ x_B y_B ∧
    -- Q is midpoint of AB with abscissa -4/13
    (x_A + x_B)/2 = -4/13 ∧
  -- One possible radius of P is 6/5
  r = 6/5 :=
sorry

end NUMINAMATH_CALUDE_circle_P_radius_l2774_277475


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2774_277437

theorem quadratic_inequality (x : ℝ) : x^2 + 7*x + 6 < 0 ↔ -6 < x ∧ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2774_277437


namespace NUMINAMATH_CALUDE_ice_cube_water_cost_l2774_277426

/-- The cost of 1 ounce of water in Pauly's ice cube production --/
theorem ice_cube_water_cost : 
  let pounds_needed : ℝ := 10
  let ounces_per_cube : ℝ := 2
  let pound_per_cube : ℝ := 1/16
  let cubes_per_hour : ℝ := 10
  let cost_per_hour : ℝ := 1.5
  let total_cost : ℝ := 56
  
  let num_cubes : ℝ := pounds_needed / pound_per_cube
  let hours_needed : ℝ := num_cubes / cubes_per_hour
  let ice_maker_cost : ℝ := hours_needed * cost_per_hour
  let water_cost : ℝ := total_cost - ice_maker_cost
  let total_ounces : ℝ := num_cubes * ounces_per_cube
  let cost_per_ounce : ℝ := water_cost / total_ounces
  
  cost_per_ounce = 0.1 := by sorry

end NUMINAMATH_CALUDE_ice_cube_water_cost_l2774_277426


namespace NUMINAMATH_CALUDE_mary_nickels_l2774_277481

theorem mary_nickels (initial_nickels : ℕ) (dad_gave_nickels : ℕ) 
  (h1 : initial_nickels = 7)
  (h2 : dad_gave_nickels = 5) : 
  initial_nickels + dad_gave_nickels = 12 := by
sorry

end NUMINAMATH_CALUDE_mary_nickels_l2774_277481


namespace NUMINAMATH_CALUDE_coffee_shop_revenue_l2774_277431

theorem coffee_shop_revenue : 
  let coffee_orders : ℕ := 7
  let tea_orders : ℕ := 8
  let coffee_price : ℕ := 5
  let tea_price : ℕ := 4
  let total_revenue := coffee_orders * coffee_price + tea_orders * tea_price
  total_revenue = 67 := by sorry

end NUMINAMATH_CALUDE_coffee_shop_revenue_l2774_277431


namespace NUMINAMATH_CALUDE_parallel_line_a_value_l2774_277400

/-- Given two points A and B on a line parallel to 2x - y + 1 = 0, prove a = 2 -/
theorem parallel_line_a_value (a : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    A = (-1, a) ∧ 
    B = (a, 8) ∧ 
    (∃ (m : ℝ), (B.2 - A.2) = m * (B.1 - A.1) ∧ m = 2)) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_a_value_l2774_277400


namespace NUMINAMATH_CALUDE_triangle_area_proof_l2774_277438

open Real

theorem triangle_area_proof (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  -- a * tan(B) = 2b * sin(A)
  a * tan B = 2 * b * sin A →
  -- b = √3
  b = sqrt 3 →
  -- A = 5π/12
  A = 5 * π / 12 →
  -- The area of triangle ABC is (3 + √3) / 4
  (1 / 2) * b * c * sin A = (3 + sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l2774_277438


namespace NUMINAMATH_CALUDE_sqrt_sum_eq_two_l2774_277420

theorem sqrt_sum_eq_two (x₁ x₂ : ℝ) 
  (h1 : x₁ ≥ x₂) 
  (h2 : x₂ ≥ 0) 
  (h3 : x₁ + x₂ = 2) : 
  Real.sqrt (x₁ + Real.sqrt (x₁^2 - x₂^2)) + Real.sqrt (x₁ - Real.sqrt (x₁^2 - x₂^2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_eq_two_l2774_277420


namespace NUMINAMATH_CALUDE_cabinet_price_l2774_277418

theorem cabinet_price (P : ℝ) (discount_rate : ℝ) (discounted_price : ℝ) : 
  discount_rate = 0.15 →
  discounted_price = 1020 →
  discounted_price = P * (1 - discount_rate) →
  P = 1200 := by
sorry

end NUMINAMATH_CALUDE_cabinet_price_l2774_277418


namespace NUMINAMATH_CALUDE_age_difference_proof_l2774_277419

theorem age_difference_proof (man_age son_age : ℕ) : 
  man_age > son_age →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 26 →
  man_age - son_age = 28 := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2774_277419


namespace NUMINAMATH_CALUDE_unique_solution_for_x_l2774_277441

theorem unique_solution_for_x : ∃! x : ℝ, 
  (x ≠ 2) ∧ 
  ((x^3 - 8) / (x - 2) = 3 * x^2) ∧ 
  (x = -1) := by
sorry

end NUMINAMATH_CALUDE_unique_solution_for_x_l2774_277441


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2774_277468

theorem condition_sufficient_not_necessary : 
  (∃ (S T : Set ℝ), 
    (S = {x : ℝ | x - 1 > 0}) ∧ 
    (T = {x : ℝ | x^2 - 1 > 0}) ∧ 
    (S ⊂ T) ∧ 
    (∃ x, x ∈ T ∧ x ∉ S)) := by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2774_277468


namespace NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l2774_277421

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- An octagon formed by connecting midpoints of another octagon's sides -/
def MidpointOctagon (o : RegularOctagon) : RegularOctagon :=
  sorry

/-- The area of an octagon -/
def area (o : RegularOctagon) : ℝ :=
  sorry

/-- Theorem: The ratio of the area of the midpoint octagon to the area of the original regular octagon is 1/2 -/
theorem midpoint_octagon_area_ratio (o : RegularOctagon) :
  area (MidpointOctagon o) / area o = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_midpoint_octagon_area_ratio_l2774_277421


namespace NUMINAMATH_CALUDE_vovochka_max_candies_l2774_277454

/-- Represents the problem of distributing candies among classmates -/
structure CandyDistribution where
  total_candies : ℕ
  num_classmates : ℕ
  min_group_size : ℕ
  min_group_candies : ℕ

/-- Calculates the maximum number of candies Vovochka can keep -/
def max_candies_kept (cd : CandyDistribution) : ℕ :=
  cd.total_candies - (cd.num_classmates - 1) * (cd.min_group_candies / cd.min_group_size)

/-- Theorem stating the maximum number of candies Vovochka can keep -/
theorem vovochka_max_candies :
  let cd : CandyDistribution := {
    total_candies := 200,
    num_classmates := 25,
    min_group_size := 16,
    min_group_candies := 100
  }
  max_candies_kept cd = 37 := by
  sorry

end NUMINAMATH_CALUDE_vovochka_max_candies_l2774_277454


namespace NUMINAMATH_CALUDE_area_enclosed_by_four_circles_l2774_277442

/-- The area of the figure enclosed by four identical circles inscribed in a larger circle -/
theorem area_enclosed_by_four_circles (R : ℝ) : 
  ∃ (area : ℝ), 
    area = R^2 * (4 - π) * (3 - 2 * Real.sqrt 2) ∧
    (∀ (r : ℝ), 
      r = R * (Real.sqrt 2 - 1) →
      area = 4 * r^2 - π * r^2 ∧
      (∃ (O₁ O₂ O₃ O₄ : ℝ × ℝ),
        -- Four circles with centers O₁, O₂, O₃, O₄ and radius r
        -- Each touching two others and the larger circle with radius R
        True)) :=
by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_four_circles_l2774_277442


namespace NUMINAMATH_CALUDE_sqrt_3_times_612_times_3_and_half_l2774_277493

theorem sqrt_3_times_612_times_3_and_half : Real.sqrt 3 * 612 * (3 + 3/2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_612_times_3_and_half_l2774_277493


namespace NUMINAMATH_CALUDE_subtract_fractions_l2774_277461

theorem subtract_fractions : (5 : ℚ) / 9 - (1 : ℚ) / 6 = (7 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l2774_277461


namespace NUMINAMATH_CALUDE_heather_start_time_l2774_277455

/-- Proves that Heather started her journey 24 minutes after Stacy given the problem conditions -/
theorem heather_start_time (total_distance : ℝ) (heather_speed : ℝ) (stacy_speed : ℝ) 
  (heather_distance : ℝ) :
  total_distance = 10 →
  heather_speed = 5 →
  stacy_speed = heather_speed + 1 →
  heather_distance = 3.4545454545454546 →
  (total_distance - heather_distance) / stacy_speed - heather_distance / heather_speed = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_heather_start_time_l2774_277455


namespace NUMINAMATH_CALUDE_items_distribution_count_l2774_277479

-- Define the number of items and bags
def num_items : ℕ := 5
def num_bags : ℕ := 4

-- Define a function to calculate the number of ways to distribute items
def distribute_items (items : ℕ) (bags : ℕ) : ℕ :=
  sorry

-- Theorem statement
theorem items_distribution_count :
  distribute_items num_items num_bags = 52 := by
  sorry

end NUMINAMATH_CALUDE_items_distribution_count_l2774_277479


namespace NUMINAMATH_CALUDE_equation_solutions_l2774_277408

theorem equation_solutions : 
  (∃ (x : ℝ), (x + 8) * (x + 1) = -12 ↔ (x = -4 ∨ x = -5)) ∧
  (∃ (x : ℝ), (2 * x - 3)^2 = 5 * (2 * x - 3) ↔ (x = 3/2 ∨ x = 4)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2774_277408


namespace NUMINAMATH_CALUDE_sufficient_condition_range_l2774_277497

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x : ℝ, |x + 1| ≤ 2 → x ≤ a) ∧ 
  (∃ x : ℝ, x ≤ a ∧ |x + 1| > 2) → 
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sufficient_condition_range_l2774_277497


namespace NUMINAMATH_CALUDE_sin_minus_cos_value_l2774_277495

theorem sin_minus_cos_value (θ : Real) 
  (h1 : π / 4 < θ) (h2 : θ < π / 2) 
  (h3 : Real.sin θ + Real.cos θ = 5 / 4) : 
  Real.sin θ - Real.cos θ = Real.sqrt 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_value_l2774_277495


namespace NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l2774_277433

/-- A color type representing red, green, or blue -/
inductive Color
  | Red
  | Green
  | Blue

/-- A type representing a 4 x 82 grid where each point is colored -/
def ColoredGrid := Fin 4 → Fin 82 → Color

/-- A function to check if four points form a rectangle with the same color -/
def isMonochromaticRectangle (grid : ColoredGrid) (i j p q : Nat) : Prop :=
  i < j ∧ p < q ∧
  grid ⟨i, by sorry⟩ ⟨p, by sorry⟩ = grid ⟨i, by sorry⟩ ⟨q, by sorry⟩ ∧
  grid ⟨i, by sorry⟩ ⟨p, by sorry⟩ = grid ⟨j, by sorry⟩ ⟨p, by sorry⟩ ∧
  grid ⟨i, by sorry⟩ ⟨p, by sorry⟩ = grid ⟨j, by sorry⟩ ⟨q, by sorry⟩

/-- The main theorem stating that any 4 x 82 grid colored with three colors
    contains a monochromatic rectangle -/
theorem monochromatic_rectangle_exists (grid : ColoredGrid) :
  ∃ i j p q, isMonochromaticRectangle grid i j p q :=
sorry

end NUMINAMATH_CALUDE_monochromatic_rectangle_exists_l2774_277433


namespace NUMINAMATH_CALUDE_smallest_geometric_sequence_number_l2774_277474

/-- A function that checks if a three-digit number's digits form a geometric sequence -/
def is_geometric_sequence (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b * b = a * c

/-- A function that checks if a three-digit number has distinct digits -/
def has_distinct_digits (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_geometric_sequence_number :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 →
    is_geometric_sequence n ∧ has_distinct_digits n →
    124 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_geometric_sequence_number_l2774_277474


namespace NUMINAMATH_CALUDE_cloth_profit_per_meter_l2774_277476

/-- Calculates the profit per meter of cloth given the total selling price, 
    number of meters sold, and cost price per meter. -/
def profit_per_meter (selling_price total_meters cost_price_per_meter : ℕ) : ℕ :=
  ((selling_price - (cost_price_per_meter * total_meters)) / total_meters)

/-- Proves that the profit per meter of cloth is 15 rupees given the specified conditions. -/
theorem cloth_profit_per_meter :
  profit_per_meter 8500 85 85 = 15 := by
  sorry


end NUMINAMATH_CALUDE_cloth_profit_per_meter_l2774_277476


namespace NUMINAMATH_CALUDE_x_range_for_negative_f_l2774_277449

def f (a x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + a + 1

theorem x_range_for_negative_f :
  (∀ x : ℝ, ∀ a ∈ Set.Icc (-1 : ℝ) 1, f a x < 0) →
  (∀ x : ℝ, f (-1) x < 0 ∧ f 1 x < 0 → x ∈ Set.Ioo 1 2) :=
sorry

end NUMINAMATH_CALUDE_x_range_for_negative_f_l2774_277449


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2774_277413

theorem inequality_system_solution (x : ℝ) :
  (2 * (x - 1) < x + 2) → ((x + 1) / 2 < x) → (1 < x ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2774_277413


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l2774_277472

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℕ)

-- Define the condition for integer side lengths and no two sides being equal
def validTriangle (t : Triangle) : Prop :=
  t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.c ≠ t.a

-- Define the semiperimeter
def semiperimeter (t : Triangle) : ℚ :=
  (t.a + t.b + t.c) / 2

-- Define the inradius
def inradius (t : Triangle) : ℚ :=
  let s := semiperimeter t
  (s - t.a) * (s - t.b) * (s - t.c) / s

-- Define the excircle radii
def excircleRadius (t : Triangle) (side : ℕ) : ℚ :=
  let s := semiperimeter t
  let r := inradius t
  r * s / (s - side)

-- Define the tangency conditions
def tangencyConditions (t : Triangle) : Prop :=
  let r := inradius t
  let rA := excircleRadius t t.a
  let rB := excircleRadius t t.b
  let rC := excircleRadius t t.c
  r + rA = rB ∧ r + rA = rC

-- Theorem statement
theorem min_perimeter_triangle (t : Triangle) :
  validTriangle t → tangencyConditions t → t.a + t.b + t.c ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l2774_277472


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2774_277470

theorem cubic_root_sum (α β γ : ℂ) : 
  (α^3 - 7*α^2 + 11*α - 13 = 0) →
  (β^3 - 7*β^2 + 11*β - 13 = 0) →
  (γ^3 - 7*γ^2 + 11*γ - 13 = 0) →
  (α*β/γ + β*γ/α + γ*α/β = -61/13) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2774_277470


namespace NUMINAMATH_CALUDE_vote_combinations_l2774_277492

/-- The number of ways to select k items from n distinct items with replacement,
    where order doesn't matter (combinations with repetition) -/
def combinations_with_repetition (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) k

/-- Theorem: There are 6 ways to select 2 items from 3 items with replacement,
    where order doesn't matter -/
theorem vote_combinations : combinations_with_repetition 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_vote_combinations_l2774_277492


namespace NUMINAMATH_CALUDE_line_AB_equation_min_area_and_B_coords_l2774_277446

-- Define the line l: y = 4x
def line_l (x y : ℝ) : Prop := y = 4 * x

-- Define point P
def point_P : ℝ × ℝ := (6, 4)

-- Define that point A is in the first quadrant and lies on line l
def point_A_condition (A : ℝ × ℝ) : Prop :=
  A.1 > 0 ∧ A.2 > 0 ∧ line_l A.1 A.2

-- Define that line PA intersects the positive half of the x-axis at point B
def point_B_condition (A B : ℝ × ℝ) : Prop :=
  B.2 = 0 ∧ B.1 > 0 ∧ ∃ t : ℝ, 0 < t ∧ t < 1 ∧
  B.1 = t * A.1 + (1 - t) * point_P.1 ∧
  B.2 = t * A.2 + (1 - t) * point_P.2

-- Theorem for part (1)
theorem line_AB_equation (A B : ℝ × ℝ) 
  (h1 : point_A_condition A) 
  (h2 : point_B_condition A B) 
  (h3 : (A.2 - point_P.2) * (B.1 - point_P.1) = -(A.1 - point_P.1) * (B.2 - point_P.2)) :
  ∃ k c : ℝ, k = -3/2 ∧ c = 13 ∧ ∀ x y : ℝ, y = k * x + c ↔ 3 * x + 2 * y - 26 = 0 :=
sorry

-- Theorem for part (2)
theorem min_area_and_B_coords (A B : ℝ × ℝ) 
  (h1 : point_A_condition A) 
  (h2 : point_B_condition A B) :
  ∃ S_min : ℝ, S_min = 40 ∧
  (∀ A' B' : ℝ × ℝ, point_A_condition A' → point_B_condition A' B' →
    1/2 * A'.1 * B'.2 - 1/2 * A'.2 * B'.1 ≥ S_min) ∧
  (∃ A_min B_min : ℝ × ℝ, 
    point_A_condition A_min ∧ 
    point_B_condition A_min B_min ∧
    1/2 * A_min.1 * B_min.2 - 1/2 * A_min.2 * B_min.1 = S_min ∧
    B_min = (10, 0)) :=
sorry

end NUMINAMATH_CALUDE_line_AB_equation_min_area_and_B_coords_l2774_277446


namespace NUMINAMATH_CALUDE_toms_dog_age_l2774_277473

/-- Given the ages of Tom's pets, prove the age of his dog. -/
theorem toms_dog_age (cat_age : ℕ) (rabbit_age : ℕ) (dog_age : ℕ)
  (h1 : cat_age = 8)
  (h2 : rabbit_age = cat_age / 2)
  (h3 : dog_age = rabbit_age * 3) :
  dog_age = 12 :=
by sorry

end NUMINAMATH_CALUDE_toms_dog_age_l2774_277473


namespace NUMINAMATH_CALUDE_min_distance_complex_circle_l2774_277416

open Complex

theorem min_distance_complex_circle (Z : ℂ) (h : abs (Z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 3 ∧ ∀ (W : ℂ), abs (W + 2 - 2*I) = 1 → abs (W - 2 - 2*I) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_distance_complex_circle_l2774_277416


namespace NUMINAMATH_CALUDE_roof_area_is_400_l2774_277444

/-- Represents a rectangular roof with given properties -/
structure RectangularRoof where
  width : ℝ
  length : ℝ
  length_is_triple_width : length = 3 * width
  length_width_difference : length - width = 30

/-- The area of a rectangular roof -/
def roof_area (roof : RectangularRoof) : ℝ :=
  roof.length * roof.width

/-- Theorem stating that a roof with the given properties has an area of 400 square feet -/
theorem roof_area_is_400 (roof : RectangularRoof) : roof_area roof = 400 := by
  sorry

end NUMINAMATH_CALUDE_roof_area_is_400_l2774_277444


namespace NUMINAMATH_CALUDE_rachel_apple_tree_l2774_277406

/-- The number of apples on Rachel's tree after picking some and new ones growing. -/
def final_apples (initial : ℕ) (picked : ℕ) (new_grown : ℕ) : ℕ :=
  initial - picked + new_grown

/-- Theorem stating that the final number of apples is correct. -/
theorem rachel_apple_tree (initial : ℕ) (picked : ℕ) (new_grown : ℕ) 
    (h1 : initial = 4) (h2 : picked = 2) (h3 : new_grown = 3) : 
  final_apples initial picked new_grown = 5 := by
  sorry

end NUMINAMATH_CALUDE_rachel_apple_tree_l2774_277406


namespace NUMINAMATH_CALUDE_sum_of_angles_in_divided_hexagon_l2774_277485

-- Define a hexagon divided into two quadrilaterals
structure DividedHexagon where
  quad1 : Finset ℕ
  quad2 : Finset ℕ
  h1 : quad1.card = 4
  h2 : quad2.card = 4
  h3 : quad1 ∩ quad2 = ∅
  h4 : quad1 ∪ quad2 = Finset.range 8

-- Define the sum of angles in a quadrilateral (in degrees)
def quadrilateralAngleSum : ℕ := 360

-- Theorem statement
theorem sum_of_angles_in_divided_hexagon (h : DividedHexagon) :
  (h.quad1.sum (λ i => quadrilateralAngleSum / 4)) +
  (h.quad2.sum (λ i => quadrilateralAngleSum / 4)) = 720 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_angles_in_divided_hexagon_l2774_277485


namespace NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_l2774_277435

theorem integral_sqrt_one_minus_x_squared_plus_x (f : ℝ → ℝ) :
  (∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x)) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_one_minus_x_squared_plus_x_l2774_277435


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2774_277464

theorem perpendicular_lines_a_values (a : ℝ) : 
  let l1 := {(x, y) : ℝ × ℝ | a * x + 2 * y + 1 = 0}
  let l2 := {(x, y) : ℝ × ℝ | (3 - a) * x - y + a = 0}
  let slope1 := -a / 2
  let slope2 := 3 - a
  (slope1 * slope2 = -1) → (a = 1 ∨ a = 2) := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2774_277464


namespace NUMINAMATH_CALUDE_dans_eggs_l2774_277412

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The number of dozens Dan bought -/
def dans_dozens : ℕ := 9

/-- Theorem: Dan bought 108 eggs -/
theorem dans_eggs : dans_dozens * eggs_per_dozen = 108 := by
  sorry

end NUMINAMATH_CALUDE_dans_eggs_l2774_277412


namespace NUMINAMATH_CALUDE_triangle_side_length_l2774_277460

/-- Given a triangle XYZ with sides x, y, and z, where y = 7, z = 6, and cos(Y - Z) = 47/64,
    prove that x = √63.75 -/
theorem triangle_side_length (x y z : ℝ) (Y Z : ℝ) :
  y = 7 →
  z = 6 →
  Real.cos (Y - Z) = 47 / 64 →
  x = Real.sqrt 63.75 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2774_277460


namespace NUMINAMATH_CALUDE_midpoint_locus_l2774_277452

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 8*x = 0

/-- The point M is the midpoint of OA -/
def is_midpoint (x y : ℝ) : Prop := ∃ (ax ay : ℝ), circle_equation ax ay ∧ x = ax/2 ∧ y = ay/2

/-- The locus equation -/
def locus_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

theorem midpoint_locus : ∀ (x y : ℝ), is_midpoint x y → locus_equation x y :=
by sorry

end NUMINAMATH_CALUDE_midpoint_locus_l2774_277452


namespace NUMINAMATH_CALUDE_students_passing_both_tests_l2774_277414

theorem students_passing_both_tests 
  (total : ℕ) 
  (passed_long_jump : ℕ) 
  (passed_shot_put : ℕ) 
  (failed_both : ℕ) 
  (h1 : total = 50) 
  (h2 : passed_long_jump = 40) 
  (h3 : passed_shot_put = 31) 
  (h4 : failed_both = 4) :
  total - failed_both = passed_long_jump + passed_shot_put - 25 := by
sorry

end NUMINAMATH_CALUDE_students_passing_both_tests_l2774_277414


namespace NUMINAMATH_CALUDE_be_length_is_fourth_root_three_l2774_277496

/-- A rhombus with specific properties and internal rectangles -/
structure SpecialRhombus where
  -- The side length of the rhombus
  side_length : ℝ
  -- The length of one diagonal of the rhombus
  diagonal_length : ℝ
  -- The length of the side BE of the internal rectangle EBCF
  be_length : ℝ
  -- The side length is 2
  side_length_eq : side_length = 2
  -- One diagonal measures 2√3
  diagonal_eq : diagonal_length = 2 * Real.sqrt 3
  -- EBCF is a square (implied by equal sides along BC and BE)
  ebcf_square : be_length * be_length = be_length * be_length
  -- Area of EBCF + Area of JKHG = Area of rhombus (since they are congruent and fit within the rhombus)
  area_eq : 2 * (be_length * be_length) = (1 / 2) * diagonal_length * side_length

/-- The length of BE in the special rhombus is ∜3 -/
theorem be_length_is_fourth_root_three (r : SpecialRhombus) : r.be_length = Real.sqrt (Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_be_length_is_fourth_root_three_l2774_277496


namespace NUMINAMATH_CALUDE_mean_proportional_81_100_l2774_277499

theorem mean_proportional_81_100 : 
  Real.sqrt (81 * 100) = 90 := by sorry

end NUMINAMATH_CALUDE_mean_proportional_81_100_l2774_277499


namespace NUMINAMATH_CALUDE_equipment_prices_l2774_277447

theorem equipment_prices (price_A price_B : ℝ) 
  (h1 : price_A = price_B + 25)
  (h2 : 2000 / price_A = 2 * (750 / price_B)) :
  price_A = 100 ∧ price_B = 75 := by
  sorry

end NUMINAMATH_CALUDE_equipment_prices_l2774_277447


namespace NUMINAMATH_CALUDE_total_difference_across_age_groups_l2774_277405

/-- Represents the number of children in each category for an age group -/
structure AgeGroup where
  camp : ℕ
  home : ℕ

/-- Calculates the difference between camp and home for an age group -/
def difference (group : AgeGroup) : ℤ :=
  group.camp - group.home

/-- The given data for each age group -/
def group_5_10 : AgeGroup := ⟨245785, 197680⟩
def group_11_15 : AgeGroup := ⟨287279, 253425⟩
def group_16_18 : AgeGroup := ⟨285994, 217173⟩

/-- The theorem to be proved -/
theorem total_difference_across_age_groups :
  difference group_5_10 + difference group_11_15 + difference group_16_18 = 150780 := by
  sorry

end NUMINAMATH_CALUDE_total_difference_across_age_groups_l2774_277405


namespace NUMINAMATH_CALUDE_polynomial_division_l2774_277434

theorem polynomial_division (z : ℝ) :
  6 * z^5 - 5 * z^4 + 2 * z^3 - 8 * z^2 + 7 * z - 3 = 
  (z^2 - 1) * (6 * z^3 + z^2 + 3 * z) + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l2774_277434


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l2774_277471

theorem quadratic_equation_from_means (α β : ℝ) : 
  (α + β) / 2 = 8 → α * β = 144 → 
  ∀ x, x^2 - 16*x + 144 = 0 ↔ (x = α ∨ x = β) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l2774_277471


namespace NUMINAMATH_CALUDE_teenager_age_problem_l2774_277466

theorem teenager_age_problem (a b : ℕ) (h1 : a > b) (h2 : a^2 - b^2 = 4*(a + b)) (h3 : a + b = 8*(a - b)) : a = 18 := by
  sorry

end NUMINAMATH_CALUDE_teenager_age_problem_l2774_277466


namespace NUMINAMATH_CALUDE_min_abc_value_l2774_277456

/-- Given prime numbers a, b, c where a^5 divides (b^2 - c) and b + c is a perfect square,
    the minimum value of abc is 1958. -/
theorem min_abc_value (a b c : ℕ) : Prime a → Prime b → Prime c →
  (a^5 ∣ (b^2 - c)) → ∃ (n : ℕ), b + c = n^2 → (∀ x y z : ℕ, Prime x → Prime y → Prime z →
  (x^5 ∣ (y^2 - z)) → ∃ (m : ℕ), y + z = m^2 → x*y*z ≥ a*b*c) → a*b*c = 1958 := by
  sorry

end NUMINAMATH_CALUDE_min_abc_value_l2774_277456


namespace NUMINAMATH_CALUDE_new_tv_width_l2774_277436

theorem new_tv_width (first_tv_width : ℝ) (first_tv_height : ℝ) (first_tv_cost : ℝ)
                     (new_tv_height : ℝ) (new_tv_cost : ℝ) :
  first_tv_width = 24 →
  first_tv_height = 16 →
  first_tv_cost = 672 →
  new_tv_height = 32 →
  new_tv_cost = 1152 →
  (first_tv_cost / (first_tv_width * first_tv_height)) =
    (new_tv_cost / (new_tv_height * (new_tv_cost / (new_tv_height * (first_tv_cost / (first_tv_width * first_tv_height) - 1))))) + 1 →
  new_tv_cost / (new_tv_height * (first_tv_cost / (first_tv_width * first_tv_height) - 1)) = 48 :=
by
  sorry

#check new_tv_width

end NUMINAMATH_CALUDE_new_tv_width_l2774_277436


namespace NUMINAMATH_CALUDE_prime_square_mod_180_l2774_277469

theorem prime_square_mod_180 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_5 : p > 5) :
  ∃ (s : Finset ℕ), (∀ x ∈ s, x < 180) ∧ (Finset.card s = 2) ∧
  (∀ q : ℕ, Nat.Prime q → q > 5 → (q^2 % 180) ∈ s) :=
sorry

end NUMINAMATH_CALUDE_prime_square_mod_180_l2774_277469


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2774_277465

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + x + 3 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2774_277465


namespace NUMINAMATH_CALUDE_sara_golf_balls_l2774_277407

def dozen : ℕ := 12

theorem sara_golf_balls (total_balls : ℕ) (h : total_balls = 192) :
  total_balls / dozen = 16 := by
  sorry

end NUMINAMATH_CALUDE_sara_golf_balls_l2774_277407


namespace NUMINAMATH_CALUDE_equivalent_operations_l2774_277415

theorem equivalent_operations (x : ℝ) : (x * (4/5)) / (4/7) = x * (7/5) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_operations_l2774_277415


namespace NUMINAMATH_CALUDE_sum_of_cubes_divisible_by_nine_l2774_277439

theorem sum_of_cubes_divisible_by_nine (n : ℤ) : 
  ∃ k : ℤ, (n - 1)^3 + n^3 + (n + 1)^3 = 9 * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_divisible_by_nine_l2774_277439


namespace NUMINAMATH_CALUDE_exam_max_marks_l2774_277429

theorem exam_max_marks (percentage : ℝ) (scored_marks : ℝ) (max_marks : ℝ) : 
  percentage = 0.95 → scored_marks = 285 → percentage * max_marks = scored_marks → max_marks = 300 := by
  sorry

end NUMINAMATH_CALUDE_exam_max_marks_l2774_277429


namespace NUMINAMATH_CALUDE_sum_of_perimeters_equals_expected_l2774_277403

/-- Calculates the sum of perimeters of triangles formed by repeatedly
    connecting points 1/3 of the distance along each side of an initial
    equilateral triangle, for a given number of iterations. -/
def sumOfPerimeters (initialSideLength : ℚ) (iterations : ℕ) : ℚ :=
  let rec perimeter (sideLength : ℚ) (n : ℕ) : ℚ :=
    if n = 0 then 0
    else 3 * sideLength + perimeter (sideLength / 3) (n - 1)
  perimeter initialSideLength (iterations + 1)

/-- Theorem stating that the sum of perimeters of triangles formed by
    repeatedly connecting points 1/3 of the distance along each side of
    an initial equilateral triangle with side length 18 units, for 4
    iterations, is equal to 80 2/3 units. -/
theorem sum_of_perimeters_equals_expected :
  sumOfPerimeters 18 4 = 80 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_perimeters_equals_expected_l2774_277403


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2774_277463

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define set A (domain of f(x))
def A : Set ℝ := {x | x ≥ Real.exp 1}

-- Theorem statement
theorem complement_of_A_in_U : 
  (U \ A) = Set.Ioo 0 (Real.exp 1) := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2774_277463


namespace NUMINAMATH_CALUDE_unique_modulus_for_xy_plus_one_implies_x_plus_y_l2774_277478

theorem unique_modulus_for_xy_plus_one_implies_x_plus_y (n : ℕ+) : 
  (∀ x y : ℤ, (x * y + 1) % n = 0 → (x + y) % n = 0) ↔ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_modulus_for_xy_plus_one_implies_x_plus_y_l2774_277478


namespace NUMINAMATH_CALUDE_binomial_18_6_l2774_277410

theorem binomial_18_6 : Nat.choose 18 6 = 18564 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_6_l2774_277410


namespace NUMINAMATH_CALUDE_first_set_video_count_l2774_277417

/-- The cost of a video cassette in rupees -/
def video_cost : ℕ := 300

/-- The total cost of the first set in rupees -/
def first_set_cost : ℕ := 1110

/-- The total cost of the second set in rupees -/
def second_set_cost : ℕ := 1350

/-- The number of audio cassettes in the first set -/
def first_set_audio : ℕ := 7

/-- The number of audio cassettes in the second set -/
def second_set_audio : ℕ := 5

/-- The number of video cassettes in the second set -/
def second_set_video : ℕ := 4

/-- Theorem stating that the number of video cassettes in the first set is 3 -/
theorem first_set_video_count : 
  ∃ (audio_cost : ℕ) (first_set_video : ℕ),
    first_set_audio * audio_cost + first_set_video * video_cost = first_set_cost ∧
    second_set_audio * audio_cost + second_set_video * video_cost = second_set_cost ∧
    first_set_video = 3 :=
by sorry

end NUMINAMATH_CALUDE_first_set_video_count_l2774_277417


namespace NUMINAMATH_CALUDE_square_with_trees_theorem_l2774_277411

/-- Represents a square with trees at its vertices -/
structure SquareWithTrees where
  side_length : ℝ
  height_A : ℝ
  height_B : ℝ
  height_C : ℝ
  height_D : ℝ

/-- Checks if there exists a point equidistant from all tree tops -/
def has_equidistant_point (s : SquareWithTrees) : Prop :=
  ∃ (x y : ℝ), 0 < x ∧ x < s.side_length ∧ 0 < y ∧ y < s.side_length ∧
  (s.height_A^2 + x^2 + y^2 = s.height_B^2 + (s.side_length - x)^2 + y^2) ∧
  (s.height_A^2 + x^2 + y^2 = s.height_C^2 + (s.side_length - x)^2 + (s.side_length - y)^2) ∧
  (s.height_A^2 + x^2 + y^2 = s.height_D^2 + x^2 + (s.side_length - y)^2)

/-- The main theorem about the square with trees -/
theorem square_with_trees_theorem (s : SquareWithTrees) 
  (h1 : s.height_A = 7)
  (h2 : s.height_B = 13)
  (h3 : s.height_C = 17)
  (h4 : has_equidistant_point s) :
  s.side_length > Real.sqrt 120 ∧ s.height_D = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_with_trees_theorem_l2774_277411


namespace NUMINAMATH_CALUDE_ralphSockPurchase_l2774_277427

/-- Represents the number of socks bought at each price point -/
structure SockPurchase where
  oneDollar : Nat
  twoDollar : Nat
  fourDollar : Nat

/-- Checks if the SockPurchase satisfies the problem conditions -/
def isValidPurchase (p : SockPurchase) : Prop :=
  p.oneDollar + p.twoDollar + p.fourDollar = 10 ∧
  p.oneDollar + 2 * p.twoDollar + 4 * p.fourDollar = 30 ∧
  p.oneDollar ≥ 1 ∧ p.twoDollar ≥ 1 ∧ p.fourDollar ≥ 1

/-- Theorem stating that the only valid purchase has 2 pairs of $1 socks -/
theorem ralphSockPurchase :
  ∀ p : SockPurchase, isValidPurchase p → p.oneDollar = 2 :=
by sorry

end NUMINAMATH_CALUDE_ralphSockPurchase_l2774_277427
