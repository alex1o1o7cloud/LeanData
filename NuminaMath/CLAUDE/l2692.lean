import Mathlib

namespace NUMINAMATH_CALUDE_common_root_of_quadratic_equations_l2692_269291

theorem common_root_of_quadratic_equations (a b x : ℝ) :
  (x^2 + 2019*a*x + b = 0) ∧
  (x^2 + 2019*b*x + a = 0) ∧
  (a ≠ b) →
  x = 1/2019 :=
by sorry

end NUMINAMATH_CALUDE_common_root_of_quadratic_equations_l2692_269291


namespace NUMINAMATH_CALUDE_rectangle_exists_l2692_269277

/-- A list of the given square side lengths -/
def square_sides : List ℕ := [2, 5, 7, 9, 16, 25, 28, 33, 36]

/-- The total area covered by all squares -/
def total_area : ℕ := (square_sides.map (λ x => x * x)).sum

/-- Proposition: There exists a rectangle with integer dimensions that can be tiled by the given squares -/
theorem rectangle_exists : ∃ (length width : ℕ), 
  length * width = total_area ∧ 
  length > 0 ∧ 
  width > 0 :=
sorry

end NUMINAMATH_CALUDE_rectangle_exists_l2692_269277


namespace NUMINAMATH_CALUDE_equation_solution_l2692_269253

theorem equation_solution : ∃! x : ℚ, (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) ∧ x = -19/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2692_269253


namespace NUMINAMATH_CALUDE_clock_hands_coincide_l2692_269239

/-- The rate at which the hour hand moves, in degrees per minute -/
def hour_hand_rate : ℝ := 0.5

/-- The rate at which the minute hand moves, in degrees per minute -/
def minute_hand_rate : ℝ := 6

/-- The position of the hour hand at 7:00, in degrees -/
def initial_hour_hand_position : ℝ := 210

/-- The time interval in which we're checking for coincidence -/
def time_interval : Set ℝ := {t | 30 ≤ t ∧ t ≤ 45}

/-- The theorem stating that the clock hands coincide once in the given interval -/
theorem clock_hands_coincide : ∃ t ∈ time_interval, 
  initial_hour_hand_position + hour_hand_rate * t = minute_hand_rate * t :=
sorry

end NUMINAMATH_CALUDE_clock_hands_coincide_l2692_269239


namespace NUMINAMATH_CALUDE_median_of_consecutive_integers_with_sum_property_l2692_269232

-- Define a set of consecutive integers
def ConsecutiveIntegers (a : ℤ) (n : ℕ) := {i : ℤ | ∃ k : ℕ, k < n ∧ i = a + k}

-- Define the property of sum of nth from beginning and end being 200
def SumProperty (s : Set ℤ) : Prop :=
  ∀ a n, s = ConsecutiveIntegers a n →
    ∀ k, k < n → (a + k) + (a + (n - 1 - k)) = 200

-- Theorem statement
theorem median_of_consecutive_integers_with_sum_property (s : Set ℤ) :
  SumProperty s → ∃ a n, s = ConsecutiveIntegers a n ∧ n % 2 = 1 ∧ 
  (∃ m : ℤ, m ∈ s ∧ (∀ x ∈ s, 2 * (x - m) ≤ n - 1 ∧ 2 * (m - x) ≤ n - 1) ∧ m = 100) :=
sorry

end NUMINAMATH_CALUDE_median_of_consecutive_integers_with_sum_property_l2692_269232


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_product_l2692_269229

/-- Given an ellipse and a hyperbola with specific foci, prove the product of their semi-axes lengths -/
theorem ellipse_hyperbola_product (a b : ℝ) : 
  (∀ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 → (x = 0 ∧ (y = 5 ∨ y = -5))) → 
  (∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 → (y = 0 ∧ (x = 7 ∨ x = -7))) → 
  |a * b| = 2 * Real.sqrt 111 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_product_l2692_269229


namespace NUMINAMATH_CALUDE_men_in_business_class_l2692_269271

def total_passengers : ℕ := 160
def men_percentage : ℚ := 3/4
def business_class_percentage : ℚ := 1/4

theorem men_in_business_class : 
  ⌊(total_passengers : ℚ) * men_percentage * business_class_percentage⌋ = 30 := by
  sorry

end NUMINAMATH_CALUDE_men_in_business_class_l2692_269271


namespace NUMINAMATH_CALUDE_repeating_decimal_seven_three_five_equals_fraction_l2692_269275

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- Converts a RepeatingDecimal to a rational number -/
def repeatingDecimalToRational (x : RepeatingDecimal) : ℚ :=
  sorry

theorem repeating_decimal_seven_three_five_equals_fraction : 
  repeatingDecimalToRational ⟨7, 35⟩ = 728 / 99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_seven_three_five_equals_fraction_l2692_269275


namespace NUMINAMATH_CALUDE_proportion_equality_l2692_269255

theorem proportion_equality (x : ℝ) : (x / 5 = 1.2 / 8) → x = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l2692_269255


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2692_269245

theorem sufficient_not_necessary :
  (∀ x y : ℝ, x + y = 1 → x * y ≤ 1/4) ∧
  (∃ x y : ℝ, x * y ≤ 1/4 ∧ x + y ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2692_269245


namespace NUMINAMATH_CALUDE_mary_money_left_l2692_269265

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let initial_money := 50
  let drink_cost := p
  let medium_pizza_cost := 3 * p
  let large_pizza_cost := 5 * p
  let num_drinks := 4
  let num_medium_pizzas := 3
  let num_large_pizzas := 2
  initial_money - (num_drinks * drink_cost + num_medium_pizzas * medium_pizza_cost + num_large_pizzas * large_pizza_cost)

/-- Theorem stating that Mary has 50 - 23p dollars left after her purchases -/
theorem mary_money_left (p : ℝ) : money_left p = 50 - 23 * p := by
  sorry

end NUMINAMATH_CALUDE_mary_money_left_l2692_269265


namespace NUMINAMATH_CALUDE_combustion_reaction_result_l2692_269261

-- Define the thermochemical equations
def nitrobenzene_combustion (x : ℝ) : ℝ := 3094.88 * x
def aniline_combustion (y : ℝ) : ℝ := 3392.15 * y
def ethanol_combustion (z : ℝ) : ℝ := 1370 * z

-- Define the relationship between x and y based on nitrogen production
def nitrogen_production (x y : ℝ) : Prop := 0.5 * x + 0.5 * y = 0.15

-- Define the total heat released
def total_heat_released (x y z : ℝ) : Prop :=
  nitrobenzene_combustion x + aniline_combustion y + ethanol_combustion z = 1467.4

-- Define the mass of the solution
def solution_mass (x : ℝ) : ℝ := 470 * x

-- Define the theorem
theorem combustion_reaction_result :
  ∃ (x y z : ℝ),
    nitrogen_production x y ∧
    total_heat_released x y z ∧
    x = 0.1 ∧
    solution_mass x = 47 := by
  sorry

end NUMINAMATH_CALUDE_combustion_reaction_result_l2692_269261


namespace NUMINAMATH_CALUDE_student_count_problem_l2692_269224

theorem student_count_problem : ∃! n : ℕ, n < 500 ∧ 
  n % 17 = 15 ∧ 
  n % 19 = 18 ∧ 
  n % 16 = 7 ∧ 
  n = 417 := by
sorry

end NUMINAMATH_CALUDE_student_count_problem_l2692_269224


namespace NUMINAMATH_CALUDE_log_inequality_domain_l2692_269279

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_inequality_domain (x : ℝ) :
  log 2 (log 1 (log 5 x)) > 0 ↔ 1 < x ∧ x < Real.rpow 5 (1/3) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_domain_l2692_269279


namespace NUMINAMATH_CALUDE_plane_existence_and_uniqueness_l2692_269249

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The first bisector plane -/
def firstBisectorPlane : Plane3D := sorry

/-- Check if a point lies in a plane -/
def pointInPlane (p : Point3D) (plane : Plane3D) : Prop := sorry

/-- First angle of projection of a plane -/
def firstProjectionAngle (plane : Plane3D) : ℝ := sorry

/-- Angle between first and second trace lines of a plane -/
def traceLinesAngle (plane : Plane3D) : ℝ := sorry

/-- Theorem: Existence and uniqueness of a plane with given properties -/
theorem plane_existence_and_uniqueness 
  (P : Point3D) 
  (α β : ℝ) 
  (h_P : pointInPlane P firstBisectorPlane) :
  ∃! s : Plane3D, 
    pointInPlane P s ∧ 
    firstProjectionAngle s = α ∧ 
    traceLinesAngle s = β := by
  sorry

end NUMINAMATH_CALUDE_plane_existence_and_uniqueness_l2692_269249


namespace NUMINAMATH_CALUDE_special_numbers_exist_l2692_269206

theorem special_numbers_exist : ∃ (a b c d e : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
  (¬(3 ∣ a) ∧ ¬(4 ∣ a) ∧ ¬(5 ∣ a)) ∧
  (¬(3 ∣ b) ∧ ¬(4 ∣ b) ∧ ¬(5 ∣ b)) ∧
  (¬(3 ∣ c) ∧ ¬(4 ∣ c) ∧ ¬(5 ∣ c)) ∧
  (¬(3 ∣ d) ∧ ¬(4 ∣ d) ∧ ¬(5 ∣ d)) ∧
  (¬(3 ∣ e) ∧ ¬(4 ∣ e) ∧ ¬(5 ∣ e)) ∧
  (3 ∣ (a + b + c)) ∧ (3 ∣ (a + b + d)) ∧ (3 ∣ (a + b + e)) ∧
  (3 ∣ (a + c + d)) ∧ (3 ∣ (a + c + e)) ∧ (3 ∣ (a + d + e)) ∧
  (3 ∣ (b + c + d)) ∧ (3 ∣ (b + c + e)) ∧ (3 ∣ (b + d + e)) ∧
  (3 ∣ (c + d + e)) ∧
  (4 ∣ (a + b + c + d)) ∧ (4 ∣ (a + b + c + e)) ∧
  (4 ∣ (a + b + d + e)) ∧ (4 ∣ (a + c + d + e)) ∧
  (4 ∣ (b + c + d + e)) ∧
  (5 ∣ (a + b + c + d + e)) := by
sorry

end NUMINAMATH_CALUDE_special_numbers_exist_l2692_269206


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2692_269272

/-- Represents a hyperbola with center O and focus F -/
structure Hyperbola where
  O : ℝ × ℝ  -- Center of the hyperbola
  F : ℝ × ℝ  -- Focus of the hyperbola

/-- Represents a point on the asymptote of the hyperbola -/
def AsymptoticPoint (h : Hyperbola) := ℝ × ℝ

/-- Checks if a triangle is isosceles right -/
def IsIsoscelesRight (A B C : ℝ × ℝ) : Prop := sorry

/-- Calculates the eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Main theorem: If a point P on the asymptote of a hyperbola forms an isosceles right triangle
    with the center O and focus F, then the eccentricity of the hyperbola is √2 -/
theorem hyperbola_eccentricity 
  (h : Hyperbola) 
  (P : AsymptoticPoint h) 
  (h_isosceles : IsIsoscelesRight h.O h.F P) : 
  eccentricity h = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2692_269272


namespace NUMINAMATH_CALUDE_expression_value_l2692_269230

theorem expression_value : (19 + 43 / 151) * 151 = 2912 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2692_269230


namespace NUMINAMATH_CALUDE_arithmetic_iff_straight_line_l2692_269209

/-- A sequence of real numbers -/
def Sequence := ℕ+ → ℝ

/-- A sequence of points in 2D space -/
def PointSequence := ℕ+ → ℝ × ℝ

/-- Predicate for arithmetic sequences -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

/-- Predicate for points lying on a straight line -/
def on_straight_line (P : PointSequence) : Prop :=
  ∃ m b : ℝ, ∀ n : ℕ+, (P n).2 = m * (P n).1 + b

/-- Main theorem: equivalence between arithmetic sequence and points on a straight line -/
theorem arithmetic_iff_straight_line (a : Sequence) (P : PointSequence) :
  is_arithmetic a ↔ on_straight_line P :=
sorry

end NUMINAMATH_CALUDE_arithmetic_iff_straight_line_l2692_269209


namespace NUMINAMATH_CALUDE_product_purely_imaginary_l2692_269238

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the product function
def product (x : ℝ) : ℂ := (x - i) * ((x + 2) - i) * ((x + 4) - i)

-- Define the property of being purely imaginary
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

-- State the theorem
theorem product_purely_imaginary (x : ℝ) : 
  isPurelyImaginary (product x) ↔ 
  (x = -3 ∨ x = (-3 + Real.sqrt 13) / 2 ∨ x = (-3 - Real.sqrt 13) / 2) :=
sorry

end NUMINAMATH_CALUDE_product_purely_imaginary_l2692_269238


namespace NUMINAMATH_CALUDE_value_of_z_l2692_269284

theorem value_of_z (x y z : ℝ) 
  (h1 : x = (1/3) * y) 
  (h2 : y = (1/4) * z) 
  (h3 : x + y = 16) : 
  z = 48 := by
sorry

end NUMINAMATH_CALUDE_value_of_z_l2692_269284


namespace NUMINAMATH_CALUDE_billy_ticket_usage_l2692_269252

/-- The number of times Billy rode the ferris wheel -/
def ferris_rides : ℕ := 7

/-- The number of times Billy rode the bumper cars -/
def bumper_rides : ℕ := 3

/-- The cost in tickets for each ride -/
def tickets_per_ride : ℕ := 5

/-- The total number of tickets Billy used -/
def total_tickets : ℕ := (ferris_rides + bumper_rides) * tickets_per_ride

theorem billy_ticket_usage : total_tickets = 50 := by
  sorry

end NUMINAMATH_CALUDE_billy_ticket_usage_l2692_269252


namespace NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l2692_269219

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem max_gcd_consecutive_terms : 
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (b n) (b (n + 1)) ≤ k ∧ 
  ∃ (m : ℕ), Nat.gcd (b m) (b (m + 1)) = k :=
sorry

end NUMINAMATH_CALUDE_max_gcd_consecutive_terms_l2692_269219


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l2692_269296

/-- The y-intercept of the line 3x - 5y = 7 is -7/5 -/
theorem y_intercept_of_line (x y : ℝ) :
  3 * x - 5 * y = 7 → x = 0 → y = -7/5 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l2692_269296


namespace NUMINAMATH_CALUDE_cookie_sheet_perimeter_l2692_269248

/-- The perimeter of a rectangular cookie sheet -/
theorem cookie_sheet_perimeter (width : ℝ) (length : ℝ) (inch_to_cm : ℝ) : 
  width = 15.2 ∧ length = 3.7 ∧ inch_to_cm = 2.54 →
  2 * (width * inch_to_cm + length * inch_to_cm) = 96.012 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sheet_perimeter_l2692_269248


namespace NUMINAMATH_CALUDE_function_inequality_l2692_269208

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -2 * Real.log x + a / x^2 + 1

theorem function_inequality (a : ℝ) (x₁ x₂ x₀ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₀ : x₀ > 0)
  (hz₁ : f a x₁ = 0) (hz₂ : f a x₂ = 0) (hx₁₂ : x₁ ≠ x₂)
  (hextremum : ∀ x > 0, f a x₀ ≥ f a x) :
  1 / x₁^2 + 1 / x₂^2 > 2 * f a x₀ :=
sorry

end NUMINAMATH_CALUDE_function_inequality_l2692_269208


namespace NUMINAMATH_CALUDE_message_encoding_l2692_269201

-- Define the encoding functions
def oldEncode (s : String) : String := sorry

def newEncode (s : String) : String := sorry

-- Define the decoding function
def decode (s : String) : String := sorry

-- Theorem statement
theorem message_encoding :
  let originalMessage := "011011010011"
  let decodedMessage := decode originalMessage
  newEncode decodedMessage = "211221121" := by sorry

end NUMINAMATH_CALUDE_message_encoding_l2692_269201


namespace NUMINAMATH_CALUDE_trig_identities_l2692_269234

/-- Proof of trigonometric identities -/
theorem trig_identities :
  (Real.cos (π / 3) + Real.sin (π / 4) - Real.tan (π / 4) = (-1 + Real.sqrt 2) / 2) ∧
  (6 * (Real.tan (π / 6))^2 - Real.sqrt 3 * Real.sin (π / 3) - 2 * Real.cos (π / 4) = 1 / 2 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_trig_identities_l2692_269234


namespace NUMINAMATH_CALUDE_pears_juice_calculation_l2692_269254

/-- The amount of pears processed into juice given a total harvest and export percentage -/
def pears_processed_into_juice (total_harvest : ℝ) (export_percentage : ℝ) (juice_percentage : ℝ) : ℝ :=
  total_harvest * (1 - export_percentage) * juice_percentage

theorem pears_juice_calculation (total_harvest : ℝ) (export_percentage : ℝ) (juice_percentage : ℝ) 
  (h1 : total_harvest = 8.5)
  (h2 : export_percentage = 0.3)
  (h3 : juice_percentage = 0.6) :
  pears_processed_into_juice total_harvest export_percentage juice_percentage = 3.57 := by
  sorry

#eval pears_processed_into_juice 8.5 0.3 0.6

end NUMINAMATH_CALUDE_pears_juice_calculation_l2692_269254


namespace NUMINAMATH_CALUDE_carries_profit_l2692_269251

/-- Carrie's profit from making and decorating a wedding cake -/
theorem carries_profit (hours_per_day : ℕ) (days_worked : ℕ) (hourly_rate : ℕ) (supply_cost : ℕ) : 
  hours_per_day = 2 →
  days_worked = 4 →
  hourly_rate = 22 →
  supply_cost = 54 →
  (hours_per_day * days_worked * hourly_rate - supply_cost : ℕ) = 122 := by
  sorry

end NUMINAMATH_CALUDE_carries_profit_l2692_269251


namespace NUMINAMATH_CALUDE_green_blue_difference_l2692_269235

/-- Represents the color of a disk -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ
  color_sum : blue + yellow + green = total
  ratio : ∃ (k : ℕ), blue = 3 * k ∧ yellow = 7 * k ∧ green = 8 * k

/-- The main theorem to prove -/
theorem green_blue_difference (bag : DiskBag) 
  (h_total : bag.total = 108) :
  bag.green - bag.blue = 30 := by
  sorry

#check green_blue_difference

end NUMINAMATH_CALUDE_green_blue_difference_l2692_269235


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l2692_269256

/-- Given that 2x^2 - 8x + 1 can be expressed as a(x-h)^2 + k, prove that a + h + k = -3 -/
theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 2*x^2 - 8*x + 1 = a*(x-h)^2 + k) → a + h + k = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l2692_269256


namespace NUMINAMATH_CALUDE_lowest_price_correct_l2692_269202

/-- Calculates the lowest price per unit to sell electronic components without making a loss. -/
def lowest_price_per_unit (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (num_units : ℕ) : ℚ :=
  (production_cost + shipping_cost + fixed_costs / num_units)

theorem lowest_price_correct (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (num_units : ℕ) :
  lowest_price_per_unit production_cost shipping_cost fixed_costs num_units =
  (production_cost * num_units + shipping_cost * num_units + fixed_costs) / num_units :=
by sorry

#eval lowest_price_per_unit 120 10 25000 100

end NUMINAMATH_CALUDE_lowest_price_correct_l2692_269202


namespace NUMINAMATH_CALUDE_bus_passengers_after_four_stops_l2692_269204

/-- Represents the change in passengers at a bus stop -/
structure StopChange where
  boarding : Int
  alighting : Int

/-- Calculates the final number of passengers on a bus after a series of stops -/
def finalPassengers (initial : Int) (changes : List StopChange) : Int :=
  changes.foldl (fun acc stop => acc + stop.boarding - stop.alighting) initial

/-- Theorem stating the final number of passengers after 4 stops -/
theorem bus_passengers_after_four_stops :
  let initial := 22
  let changes := [
    { boarding := 3, alighting := 6 },
    { boarding := 8, alighting := 5 },
    { boarding := 2, alighting := 4 },
    { boarding := 1, alighting := 8 }
  ]
  finalPassengers initial changes = 13 := by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_after_four_stops_l2692_269204


namespace NUMINAMATH_CALUDE_simplify_expression_l2692_269263

theorem simplify_expression (b c : ℝ) :
  (1 : ℝ) * (-2 * b) * (3 * b^2) * (-4 * c^3) * (5 * c^4) = -120 * b^3 * c^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2692_269263


namespace NUMINAMATH_CALUDE_total_exhibit_time_l2692_269286

-- Define the total number of students
def totalStudents : ℕ := 30

-- Define the number of groups
def numGroups : ℕ := 5

-- Define the time taken by each group (in minutes per student)
def groupTimes : List ℕ := [4, 5, 6, 7, 8]

-- Function to calculate the time taken by each group
def groupTime (studentsPerGroup : ℕ) (timePerStudent : ℕ) : ℕ :=
  studentsPerGroup * timePerStudent

-- Theorem stating the total time for all groups
theorem total_exhibit_time :
  let studentsPerGroup := totalStudents / numGroups
  (List.sum (List.map (groupTime studentsPerGroup) groupTimes)) = 180 := by
  sorry


end NUMINAMATH_CALUDE_total_exhibit_time_l2692_269286


namespace NUMINAMATH_CALUDE_distance_between_points_l2692_269262

/-- Two cars traveling towards each other -/
structure CarProblem where
  /-- Speed of Car A in km/h -/
  speed_a : ℝ
  /-- Speed of Car B in km/h -/
  speed_b : ℝ
  /-- Time in hours until cars meet -/
  time_to_meet : ℝ
  /-- Additional time for Car A to reach point B after meeting -/
  additional_time : ℝ

/-- The theorem stating the distance between points A and B -/
theorem distance_between_points (p : CarProblem)
  (h1 : p.speed_a = p.speed_b + 20)
  (h2 : p.time_to_meet = 4)
  (h3 : p.additional_time = 3) :
  p.speed_a * p.time_to_meet + p.speed_b * p.time_to_meet = 240 := by
  sorry

#check distance_between_points

end NUMINAMATH_CALUDE_distance_between_points_l2692_269262


namespace NUMINAMATH_CALUDE_green_peaches_per_basket_l2692_269264

/-- Given 7 baskets with a total of 14 green peaches evenly distributed,
    prove that each basket contains 2 green peaches. -/
theorem green_peaches_per_basket :
  ∀ (num_baskets : ℕ) (total_green : ℕ) (green_per_basket : ℕ),
    num_baskets = 7 →
    total_green = 14 →
    total_green = num_baskets * green_per_basket →
    green_per_basket = 2 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_per_basket_l2692_269264


namespace NUMINAMATH_CALUDE_intersection_when_a_is_4_union_condition_l2692_269223

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x > 5 ∨ x < -1}

-- Theorem 1: When a = 4, A ∩ B = {x | 6 < x ≤ 7}
theorem intersection_when_a_is_4 :
  A 4 ∩ B = {x | 6 < x ∧ x ≤ 7} := by sorry

-- Theorem 2: A ∪ B = B if and only if a < -4 or a > 5
theorem union_condition (a : ℝ) :
  A a ∪ B = B ↔ a < -4 ∨ a > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_4_union_condition_l2692_269223


namespace NUMINAMATH_CALUDE_pens_sold_is_226_l2692_269294

/-- Represents the profit and cost structure of a store promotion -/
structure StorePromotion where
  penProfit : ℕ        -- Profit from selling one pen (in yuan)
  bearCost : ℕ         -- Cost of one teddy bear (in yuan)
  pensPerBundle : ℕ    -- Number of pens in a promotion bundle
  totalProfit : ℕ      -- Total profit from the promotion (in yuan)

/-- Calculates the number of pens sold during a store promotion -/
def pensSold (promo : StorePromotion) : ℕ :=
  -- Implementation details are omitted as per instructions
  sorry

/-- Theorem stating that the number of pens sold is 226 for the given promotion -/
theorem pens_sold_is_226 (promo : StorePromotion) 
  (h1 : promo.penProfit = 9)
  (h2 : promo.bearCost = 2)
  (h3 : promo.pensPerBundle = 4)
  (h4 : promo.totalProfit = 1922) : 
  pensSold promo = 226 := by
  sorry

end NUMINAMATH_CALUDE_pens_sold_is_226_l2692_269294


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2692_269258

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number consists of all identical digits -/
def all_identical_digits (n : ℕ) : Prop := sorry

/-- The main theorem -/
theorem unique_three_digit_number : 
  ∃! (N : ℕ), 100 ≤ N ∧ N < 1000 ∧ 
  (all_identical_digits (N + digit_sum N)) ∧ 
  (all_identical_digits (N - digit_sum N)) ∧
  N = 105 := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2692_269258


namespace NUMINAMATH_CALUDE_sweater_shirt_price_difference_l2692_269290

/-- Given the total price and quantity of shirts and sweaters, 
    prove that the average price of a sweater exceeds that of a shirt by $4 -/
theorem sweater_shirt_price_difference 
  (shirt_quantity : ℕ) 
  (shirt_total_price : ℚ)
  (sweater_quantity : ℕ)
  (sweater_total_price : ℚ)
  (h_shirt_quantity : shirt_quantity = 25)
  (h_shirt_price : shirt_total_price = 400)
  (h_sweater_quantity : sweater_quantity = 75)
  (h_sweater_price : sweater_total_price = 1500) :
  sweater_total_price / sweater_quantity - shirt_total_price / shirt_quantity = 4 := by
sorry

end NUMINAMATH_CALUDE_sweater_shirt_price_difference_l2692_269290


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2692_269267

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | (x + 1) * (x - 4) > 0}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2692_269267


namespace NUMINAMATH_CALUDE_star_power_equality_l2692_269231

/-- The k-th smallest positive integer not in X -/
def f_X (X : Finset ℕ+) (k : ℕ+) : ℕ+ := sorry

/-- The * operation on finite sets of positive integers -/
def star (X Y : Finset ℕ+) : Finset ℕ+ :=
  X ∪ (Y.image (f_X X))

/-- Repeated application of star operation n times -/
def star_power (X : Finset ℕ+) : ℕ → Finset ℕ+
  | 0 => X
  | n + 1 => star X (star_power X n)

theorem star_power_equality {A B : Finset ℕ+} (hA : A.Nonempty) (hB : B.Nonempty)
    (h : star A B = star B A) :
    star_power A B.card = star_power B A.card := by sorry

end NUMINAMATH_CALUDE_star_power_equality_l2692_269231


namespace NUMINAMATH_CALUDE_division_remainder_problem_l2692_269283

theorem division_remainder_problem (smaller : ℕ) : 
  1614 - smaller = 1360 →
  1614 / smaller = 6 →
  1614 % smaller = 90 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l2692_269283


namespace NUMINAMATH_CALUDE_orange_seller_gain_l2692_269200

/-- The percentage gain a man wants to achieve when selling oranges -/
def desired_gain (initial_rate : ℚ) (loss_percent : ℚ) (new_rate : ℚ) : ℚ :=
  let cost_price := 1 / (initial_rate * (1 - loss_percent / 100))
  let new_price := 1 / new_rate
  (new_price / cost_price - 1) * 100

/-- Theorem stating the desired gain for specific selling rates and loss percentage -/
theorem orange_seller_gain :
  desired_gain 18 8 (11420689655172414 / 1000000000000000) = 45 := by
  sorry

end NUMINAMATH_CALUDE_orange_seller_gain_l2692_269200


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2692_269280

/-- A sufficient condition for a line and circle to have two distinct intersection points -/
theorem line_circle_intersection (k : ℝ) :
  0 < k → k < 3 →
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ ∧
    x₁ - y₁ - k = 0 ∧
    (x₁ - 1)^2 + y₁^2 = 2 ∧
    x₂ - y₂ - k = 0 ∧
    (x₂ - 1)^2 + y₂^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2692_269280


namespace NUMINAMATH_CALUDE_integer_factorization_l2692_269220

theorem integer_factorization (a b c d : ℤ) (h : a * b = c * d) :
  ∃ (w x y z : ℤ), a = w * x ∧ b = y * z ∧ c = w * y ∧ d = x * z := by
  sorry

end NUMINAMATH_CALUDE_integer_factorization_l2692_269220


namespace NUMINAMATH_CALUDE_bus_cyclist_speeds_l2692_269214

/-- The speed of the buses in km/h -/
def bus_speed : ℝ := 42

/-- The speed of the cyclist in km/h -/
def cyclist_speed : ℝ := 18

/-- The distance between points A and B in km -/
def distance : ℝ := 37

/-- The time in minutes from the start of the first bus to meeting the cyclist -/
def time_bus1_to_meeting : ℝ := 40

/-- The time in minutes from the start of the second bus to meeting the cyclist -/
def time_bus2_to_meeting : ℝ := 31

/-- The time in minutes from the start of the cyclist to meeting the first bus -/
def time_cyclist_to_bus1 : ℝ := 30

/-- The time in minutes from the start of the cyclist to meeting the second bus -/
def time_cyclist_to_bus2 : ℝ := 51

theorem bus_cyclist_speeds : 
  bus_speed * (time_bus1_to_meeting / 60) + cyclist_speed * (time_cyclist_to_bus1 / 60) = distance ∧
  bus_speed * (time_bus2_to_meeting / 60) + cyclist_speed * (time_cyclist_to_bus2 / 60) = distance :=
by sorry

end NUMINAMATH_CALUDE_bus_cyclist_speeds_l2692_269214


namespace NUMINAMATH_CALUDE_fruit_sales_problem_l2692_269281

/-- Fruit sales problem -/
theorem fruit_sales_problem 
  (cost_price : ℝ) 
  (base_price : ℝ) 
  (base_sales : ℝ) 
  (price_increment : ℝ) 
  (sales_decrement : ℝ) 
  (min_sales : ℝ) 
  (max_price : ℝ) :
  cost_price = 8 →
  base_price = 10 →
  base_sales = 300 →
  price_increment = 1 →
  sales_decrement = 50 →
  min_sales = 250 →
  max_price = 13 →
  ∃ (sales_function : ℝ → ℝ) (max_profit : ℝ) (donation_range : Set ℝ),
    -- 1. Sales function
    (∀ x, sales_function x = -50 * x + 800) ∧
    -- 2. Maximum profit
    max_profit = 750 ∧
    -- 3. Donation range
    donation_range = {a : ℝ | 2 ≤ a ∧ a ≤ 2.5} :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_sales_problem_l2692_269281


namespace NUMINAMATH_CALUDE_water_channel_length_l2692_269268

theorem water_channel_length : ∀ L : ℝ,
  L > 0 →
  (3/4 * L - 5/28 * L) = 4/7 * L →
  (4/7 * L - 2/7 * L) = 2/7 * L →
  2/7 * L = 100 →
  L = 350 := by
sorry

end NUMINAMATH_CALUDE_water_channel_length_l2692_269268


namespace NUMINAMATH_CALUDE_total_cars_count_l2692_269287

/-- The number of cars owned by Cathy, Lindsey, Carol, and Susan -/
def total_cars (cathy lindsey carol susan : ℕ) : ℕ :=
  cathy + lindsey + carol + susan

/-- Theorem stating the total number of cars owned by all four people -/
theorem total_cars_count :
  ∀ (cathy lindsey carol susan : ℕ),
    cathy = 5 →
    lindsey = cathy + 4 →
    carol = 2 * cathy →
    susan = carol - 2 →
    total_cars cathy lindsey carol susan = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_count_l2692_269287


namespace NUMINAMATH_CALUDE_cricket_run_rate_theorem_l2692_269212

/-- Represents a cricket game with given parameters -/
structure CricketGame where
  total_overs : ℕ
  first_part_overs : ℕ
  first_part_run_rate : ℚ
  target_runs : ℕ

/-- Calculates the required run rate for the remaining overs -/
def required_run_rate (game : CricketGame) : ℚ :=
  let remaining_overs := game.total_overs - game.first_part_overs
  let first_part_runs := game.first_part_run_rate * game.first_part_overs
  let remaining_runs := game.target_runs - first_part_runs
  remaining_runs / remaining_overs

/-- The main theorem stating the required run rate for the given game parameters -/
theorem cricket_run_rate_theorem (game : CricketGame) 
    (h_total_overs : game.total_overs = 50)
    (h_first_part_overs : game.first_part_overs = 10)
    (h_first_part_run_rate : game.first_part_run_rate = 3.2)
    (h_target_runs : game.target_runs = 242) :
    required_run_rate game = 5.25 := by
  sorry

#eval required_run_rate {
  total_overs := 50,
  first_part_overs := 10,
  first_part_run_rate := 3.2,
  target_runs := 242
}

end NUMINAMATH_CALUDE_cricket_run_rate_theorem_l2692_269212


namespace NUMINAMATH_CALUDE_gary_initial_amount_l2692_269250

/-- Gary's initial amount of money -/
def initial_amount : ℕ := sorry

/-- Amount Gary spent on the snake -/
def spent_amount : ℕ := 55

/-- Amount Gary has left -/
def remaining_amount : ℕ := 18

/-- Theorem: Gary's initial amount equals the sum of spent and remaining amounts -/
theorem gary_initial_amount : initial_amount = spent_amount + remaining_amount := by sorry

end NUMINAMATH_CALUDE_gary_initial_amount_l2692_269250


namespace NUMINAMATH_CALUDE_range_of_a_l2692_269297

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period f 3)
  (h_f1 : f 1 > 1)
  (h_f2015 : f 2015 = (2 * a - 3) / (a + 1)) :
  -1 < a ∧ a < 2/3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2692_269297


namespace NUMINAMATH_CALUDE_ellipse_iff_k_range_l2692_269225

/-- The curve equation -/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 / (4 - k) + y^2 / (k - 1) = 1

/-- Conditions for the curve to be an ellipse -/
def is_ellipse (k : ℝ) : Prop :=
  4 - k > 0 ∧ k - 1 > 0 ∧ 4 - k ≠ k - 1

/-- The range of k for which the curve is an ellipse -/
def k_range (k : ℝ) : Prop :=
  1 < k ∧ k < 4 ∧ k ≠ 5/2

/-- Theorem: The curve is an ellipse if and only if k is in the specified range -/
theorem ellipse_iff_k_range (k : ℝ) :
  is_ellipse k ↔ k_range k :=
sorry

end NUMINAMATH_CALUDE_ellipse_iff_k_range_l2692_269225


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l2692_269205

/-- Represents a hyperbola with the given equation and foci on the y-axis -/
structure Hyperbola (k : ℝ) :=
  (equation : ∀ x y : ℝ, x^2 / (k - 3) + y^2 / (k + 3) = 1)
  (foci_on_y_axis : True)  -- We can't directly represent this condition, so we use a placeholder

/-- The range of k for a hyperbola with the given properties -/
def k_range (h : Hyperbola k) : Set ℝ :=
  {k | -3 < k ∧ k < 3}

/-- Theorem stating that for any hyperbola satisfying the given conditions, k is in the range (-3, 3) -/
theorem hyperbola_k_range (k : ℝ) (h : Hyperbola k) : k ∈ k_range h := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l2692_269205


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2692_269240

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define set A
def A : Set Nat := {2, 3, 5, 6}

-- Define set B
def B : Set Nat := {1, 3, 4, 6, 7}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2692_269240


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_250_125_l2692_269269

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def is_two_digit_prime (p : ℕ) : Prop := 10 ≤ p ∧ p < 100 ∧ Nat.Prime p

theorem largest_two_digit_prime_factor_of_binom_250_125 :
  ∃ (p : ℕ), is_two_digit_prime p ∧
             p ∣ binomial_coefficient 250 125 ∧
             ∀ (q : ℕ), is_two_digit_prime q ∧ q ∣ binomial_coefficient 250 125 → q ≤ p ∧
             p = 83 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binom_250_125_l2692_269269


namespace NUMINAMATH_CALUDE_optimal_chair_removal_l2692_269270

theorem optimal_chair_removal :
  let chairs_per_row : ℕ := 15
  let initial_chairs : ℕ := 150
  let expected_attendees : ℕ := 125
  let removed_chairs : ℕ := 45

  -- All rows are complete
  (initial_chairs - removed_chairs) % chairs_per_row = 0 ∧
  -- At least one row is empty
  (initial_chairs - removed_chairs) / chairs_per_row < initial_chairs / chairs_per_row ∧
  -- Remaining chairs are sufficient for attendees
  initial_chairs - removed_chairs ≥ expected_attendees ∧
  -- Minimizes empty seats
  ∀ (x : ℕ), x < removed_chairs →
    (initial_chairs - x) % chairs_per_row ≠ 0 ∨
    (initial_chairs - x) / chairs_per_row ≥ initial_chairs / chairs_per_row ∨
    initial_chairs - x < expected_attendees :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_chair_removal_l2692_269270


namespace NUMINAMATH_CALUDE_square_root_meaningful_range_l2692_269213

theorem square_root_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 6 * x + 12) ↔ x ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_meaningful_range_l2692_269213


namespace NUMINAMATH_CALUDE_f_properties_l2692_269295

/-- The function f(x) defined as 2 / (2^x + 1) + m -/
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 / (2^x + 1) + m

/-- f is an odd function -/
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- f is decreasing on ℝ -/
def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

theorem f_properties (m : ℝ) :
  (is_odd (f · m) → m = -1) ∧
  is_decreasing (f · m) ∧
  (∀ x ≤ 1, f x m ≥ f 1 m) ∧
  f (-1) m = 4/3 + m :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2692_269295


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2692_269292

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2692_269292


namespace NUMINAMATH_CALUDE_olivia_spent_25_dollars_l2692_269233

/-- The amount Olivia spent at the supermarket -/
def amount_spent (initial_amount remaining_amount : ℕ) : ℕ :=
  initial_amount - remaining_amount

/-- Theorem stating that Olivia spent 25 dollars -/
theorem olivia_spent_25_dollars (initial_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 54)
  (h2 : remaining_amount = 29) : 
  amount_spent initial_amount remaining_amount = 25 := by
  sorry

end NUMINAMATH_CALUDE_olivia_spent_25_dollars_l2692_269233


namespace NUMINAMATH_CALUDE_sequence_a_2006_bounds_l2692_269266

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => sequence_a n + (1/2006) * (sequence_a n)^2

theorem sequence_a_2006_bounds : 
  1 - 1/2008 < sequence_a 2006 ∧ sequence_a 2006 < 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_2006_bounds_l2692_269266


namespace NUMINAMATH_CALUDE_class_ratio_proof_l2692_269243

theorem class_ratio_proof (B G : ℝ) 
  (h1 : B > 0) 
  (h2 : G > 0) 
  (h3 : 0.80 * B + 0.75 * G = 0.78 * (B + G)) : 
  B / G = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_class_ratio_proof_l2692_269243


namespace NUMINAMATH_CALUDE_surface_area_increase_after_cube_removal_l2692_269278

/-- Represents the dimensions of a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.height + solid.width * solid.height)

/-- Represents the dimensions of a cube -/
structure Cube where
  side : ℝ

/-- Theorem: Removing a 1-foot cube from a 4×3×5 feet rectangular solid increases surface area by 2 sq ft -/
theorem surface_area_increase_after_cube_removal 
  (original : RectangularSolid) 
  (removed : Cube) 
  (h1 : original.length = 4)
  (h2 : original.width = 3)
  (h3 : original.height = 5)
  (h4 : removed.side = 1)
  (h5 : removed.side < original.length ∧ removed.side < original.width ∧ removed.side < original.height) :
  surfaceArea original + 2 = surfaceArea original + 
    (removed.side * removed.side + 2 * removed.side * removed.side) - removed.side * removed.side := by
  sorry

end NUMINAMATH_CALUDE_surface_area_increase_after_cube_removal_l2692_269278


namespace NUMINAMATH_CALUDE_cookie_theorem_l2692_269293

def cookie_problem (initial_cookies eaten_cookies bought_cookies : ℕ) : Prop :=
  eaten_cookies - bought_cookies = 2

theorem cookie_theorem (initial_cookies : ℕ) : 
  cookie_problem initial_cookies 5 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_theorem_l2692_269293


namespace NUMINAMATH_CALUDE_square_perimeter_l2692_269257

/-- Given a square with area 720 square meters, its perimeter is 48√5 meters. -/
theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 720 → 
  area = side ^ 2 → 
  perimeter = 4 * side → 
  perimeter = 48 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2692_269257


namespace NUMINAMATH_CALUDE_cos_cubed_minus_sin_cubed_l2692_269207

theorem cos_cubed_minus_sin_cubed (θ : ℝ) :
  Real.cos θ ^ 3 - Real.sin θ ^ 3 = (Real.cos θ - Real.sin θ) * (1 + Real.cos θ * Real.sin θ) := by
  sorry

end NUMINAMATH_CALUDE_cos_cubed_minus_sin_cubed_l2692_269207


namespace NUMINAMATH_CALUDE_intersection_M_N_l2692_269285

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def N : Set ℝ := {y | ∃ x : ℝ, x^2 + y^2 = 1}

theorem intersection_M_N : M ∩ N = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2692_269285


namespace NUMINAMATH_CALUDE_same_grade_percentage_l2692_269221

-- Define the total number of students
def total_students : ℕ := 50

-- Define the number of students who got the same grade on both tests
def same_grade_A : ℕ := 3
def same_grade_B : ℕ := 6
def same_grade_C : ℕ := 7
def same_grade_D : ℕ := 2

-- Define the total number of students who got the same grade on both tests
def total_same_grade : ℕ := same_grade_A + same_grade_B + same_grade_C + same_grade_D

-- Define the percentage of students who got the same grade on both tests
def percentage_same_grade : ℚ := (total_same_grade : ℚ) / (total_students : ℚ) * 100

-- Theorem to prove
theorem same_grade_percentage :
  percentage_same_grade = 36 := by
  sorry

end NUMINAMATH_CALUDE_same_grade_percentage_l2692_269221


namespace NUMINAMATH_CALUDE_isabel_sold_three_bead_necklaces_total_cost_equals_earnings_l2692_269242

/-- The number of bead necklaces sold by Isabel -/
def bead_necklaces : ℕ := sorry

/-- The number of gem stone necklaces sold by Isabel -/
def gem_necklaces : ℕ := 3

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 6

/-- The total earnings from all necklaces in dollars -/
def total_earnings : ℕ := 36

/-- Theorem stating that Isabel sold 3 bead necklaces -/
theorem isabel_sold_three_bead_necklaces :
  bead_necklaces = 3 :=
by
  sorry

/-- The total number of necklaces sold -/
def total_necklaces : ℕ := bead_necklaces + gem_necklaces

/-- The total cost of all necklaces sold -/
def total_cost : ℕ := total_necklaces * necklace_cost

/-- Assertion that the total cost equals the total earnings -/
theorem total_cost_equals_earnings :
  total_cost = total_earnings :=
by
  sorry

end NUMINAMATH_CALUDE_isabel_sold_three_bead_necklaces_total_cost_equals_earnings_l2692_269242


namespace NUMINAMATH_CALUDE_isabel_math_homework_pages_l2692_269218

/-- Proves that Isabel had 2 pages of math homework given the problem conditions -/
theorem isabel_math_homework_pages :
  ∀ (total_pages math_pages reading_pages : ℕ) 
    (problems_per_page total_problems : ℕ),
  reading_pages = 4 →
  problems_per_page = 5 →
  total_problems = 30 →
  total_pages = math_pages + reading_pages →
  total_problems = total_pages * problems_per_page →
  math_pages = 2 := by
sorry


end NUMINAMATH_CALUDE_isabel_math_homework_pages_l2692_269218


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l2692_269222

theorem fraction_sum_equals_decimal : 
  (3 : ℚ) / 10 + (5 : ℚ) / 100 + (7 : ℚ) / 1000 + (1 : ℚ) / 1000 = (358 : ℚ) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l2692_269222


namespace NUMINAMATH_CALUDE_gilda_marbles_l2692_269237

theorem gilda_marbles (M : ℝ) (h : M > 0) : 
  let remaining_after_pedro : ℝ := M * (1 - 0.3)
  let remaining_after_ebony : ℝ := remaining_after_pedro * (1 - 0.1)
  let remaining_after_lisa : ℝ := remaining_after_ebony * (1 - 0.4)
  remaining_after_lisa / M = 0.378 := by
  sorry

end NUMINAMATH_CALUDE_gilda_marbles_l2692_269237


namespace NUMINAMATH_CALUDE_convex_pentagon_side_comparison_l2692_269203

/-- A circle in which pentagons are inscribed -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A convex pentagon inscribed in a circle -/
structure ConvexPentagon (c : Circle) where
  vertices : Fin 5 → ℝ × ℝ
  inscribed : ∀ i, (vertices i).1^2 + (vertices i).2^2 = c.radius^2
  convex : sorry  -- Additional condition to ensure convexity

/-- The side length of a regular pentagon inscribed in a circle -/
def regularPentagonSideLength (c : Circle) : ℝ := sorry

/-- The side lengths of a convex pentagon -/
def pentagonSideLengths (c : Circle) (p : ConvexPentagon c) : Fin 5 → ℝ := sorry

theorem convex_pentagon_side_comparison (c : Circle) (p : ConvexPentagon c) :
  ∃ i : Fin 5, pentagonSideLengths c p i ≤ regularPentagonSideLength c := by sorry

end NUMINAMATH_CALUDE_convex_pentagon_side_comparison_l2692_269203


namespace NUMINAMATH_CALUDE_desired_circle_satisfies_conditions_l2692_269210

/-- The equation of the first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0

/-- The equation of the second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 5

/-- The equation of the line on which the center of the desired circle lies -/
def centerLine (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0

/-- The equation of the desired circle -/
def desiredCircle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 11 = 0

/-- Theorem stating that the desired circle satisfies all conditions -/
theorem desired_circle_satisfies_conditions :
  ∀ (x y : ℝ),
    (circle1 x y ∧ circle2 x y → desiredCircle x y) ∧
    (∃ (h k : ℝ), centerLine h k ∧ 
      ∀ (x y : ℝ), desiredCircle x y ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 11)) :=
sorry

end NUMINAMATH_CALUDE_desired_circle_satisfies_conditions_l2692_269210


namespace NUMINAMATH_CALUDE_map_distance_conversion_l2692_269244

/-- Calculates the actual distance given map distance and scale --/
def actual_distance (map_distance : ℝ) (map_scale : ℝ) : ℝ :=
  map_distance * map_scale

/-- Theorem: Given a map scale where 312 inches represent 136 km,
    a distance of 25 inches on the map corresponds to approximately 10.897425 km
    in actual distance. --/
theorem map_distance_conversion (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (actual_dist : ℝ),
    abs (actual_distance 25 (136 / 312) - actual_dist) < ε ∧
    abs (actual_dist - 10.897425) < ε :=
by sorry

end NUMINAMATH_CALUDE_map_distance_conversion_l2692_269244


namespace NUMINAMATH_CALUDE_xiao_liang_score_l2692_269236

/-- Calculates the comprehensive score for a speech contest given the weights and scores for each aspect. -/
def comprehensive_score (content_weight delivery_weight effectiveness_weight : ℚ)
                        (content_score delivery_score effectiveness_score : ℚ) : ℚ :=
  content_weight * content_score + delivery_weight * delivery_score + effectiveness_weight * effectiveness_score

/-- Theorem stating that Xiao Liang's comprehensive score is 91 points. -/
theorem xiao_liang_score :
  let content_weight : ℚ := 1/2
  let delivery_weight : ℚ := 2/5
  let effectiveness_weight : ℚ := 1/10
  let content_score : ℚ := 88
  let delivery_score : ℚ := 95
  let effectiveness_score : ℚ := 90
  comprehensive_score content_weight delivery_weight effectiveness_weight
                      content_score delivery_score effectiveness_score = 91 := by
  sorry


end NUMINAMATH_CALUDE_xiao_liang_score_l2692_269236


namespace NUMINAMATH_CALUDE_largest_when_third_digit_changed_l2692_269288

def original_number : ℚ := 0.08765

def change_third_digit : ℚ := 0.08865
def change_fourth_digit : ℚ := 0.08785
def change_fifth_digit : ℚ := 0.08768

theorem largest_when_third_digit_changed :
  change_third_digit > change_fourth_digit ∧
  change_third_digit > change_fifth_digit :=
by sorry

end NUMINAMATH_CALUDE_largest_when_third_digit_changed_l2692_269288


namespace NUMINAMATH_CALUDE_all_nonnegative_possible_l2692_269228

theorem all_nonnegative_possible (nums : List ℝ) (h1 : nums.length = 10) 
  (h2 : nums.sum / nums.length = 0) : 
  ∃ (nonneg_nums : List ℝ), nonneg_nums.length = 10 ∧ 
    nonneg_nums.sum / nonneg_nums.length = 0 ∧
    ∀ x ∈ nonneg_nums, x ≥ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_all_nonnegative_possible_l2692_269228


namespace NUMINAMATH_CALUDE_ages_when_john_is_50_l2692_269246

/- Define the initial ages and relationships -/
def john_initial_age : ℕ := 10
def alice_initial_age : ℕ := 2 * john_initial_age
def mike_initial_age : ℕ := alice_initial_age - 4

/- Define John's future age -/
def john_future_age : ℕ := 50

/- Define the theorem to prove -/
theorem ages_when_john_is_50 :
  (john_future_age + (alice_initial_age - john_initial_age) = 60) ∧
  (john_future_age + (mike_initial_age - john_initial_age) = 56) := by
  sorry

end NUMINAMATH_CALUDE_ages_when_john_is_50_l2692_269246


namespace NUMINAMATH_CALUDE_power_gt_one_iff_diff_times_b_gt_zero_l2692_269226

theorem power_gt_one_iff_diff_times_b_gt_zero
  (a b : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  a^b > 1 ↔ (a - 1) * b > 0 := by
  sorry

end NUMINAMATH_CALUDE_power_gt_one_iff_diff_times_b_gt_zero_l2692_269226


namespace NUMINAMATH_CALUDE_intersection_point_distance_to_line_l2692_269227

-- Define the lines
def l1 (x y : ℝ) : Prop := x - y + 2 = 0
def l2 (x y : ℝ) : Prop := x - 2*y + 3 = 0
def l (x y : ℝ) : Prop := 3*x + 4*y - 10 = 0

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Theorem for the intersection point
theorem intersection_point : ∃ (x y : ℝ), l1 x y ∧ l2 x y ∧ x = -1 ∧ y = 1 := by sorry

-- Theorem for the distance
theorem distance_to_line : 
  let d := |3 * P.1 + 4 * P.2 - 10| / Real.sqrt (3^2 + 4^2)
  d = 3 := by sorry

end NUMINAMATH_CALUDE_intersection_point_distance_to_line_l2692_269227


namespace NUMINAMATH_CALUDE_theater_attendance_l2692_269217

theorem theater_attendance (adult_price child_price total_people total_revenue : ℕ) 
  (h1 : adult_price = 8)
  (h2 : child_price = 1)
  (h3 : total_people = 22)
  (h4 : total_revenue = 50) : 
  ∃ (num_children : ℕ), 
    num_children ≤ total_people ∧ 
    adult_price * (total_people - num_children) + child_price * num_children = total_revenue ∧
    num_children = 18 := by
  sorry

#check theater_attendance

end NUMINAMATH_CALUDE_theater_attendance_l2692_269217


namespace NUMINAMATH_CALUDE_chord_length_polar_l2692_269259

/-- Chord length intercepted by a line on a circle in polar coordinates -/
theorem chord_length_polar (ρ θ : ℝ) (h1 : ρ = 4 * Real.sin θ) (h2 : Real.tan θ = 1/2) :
  ρ = 4 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_polar_l2692_269259


namespace NUMINAMATH_CALUDE_min_value_theorem_l2692_269247

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 6*x + 100/x^3 ≥ 3 * 50^(2/5) + 6 * 50^(1/5) ∧
  ∃ y > 0, y^2 + 6*y + 100/y^3 = 3 * 50^(2/5) + 6 * 50^(1/5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2692_269247


namespace NUMINAMATH_CALUDE_chord_intersection_l2692_269289

theorem chord_intersection (a : ℝ) : 
  let line := {(x, y) : ℝ × ℝ | x + y - a - 1 = 0}
  let circle := {(x, y) : ℝ × ℝ | (x - 2)^2 + (y - 2)^2 = 4}
  let chord_length := 2 * Real.sqrt 2
  (∃ (p q : ℝ × ℝ), p ∈ line ∧ p ∈ circle ∧ q ∈ line ∧ q ∈ circle ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) → 
  (a = 1 ∨ a = 5) :=
by sorry

end NUMINAMATH_CALUDE_chord_intersection_l2692_269289


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l2692_269273

/-- Given an arithmetic sequence of 20 terms with first term 7 and last term 67,
    the 10th term is equal to 673 / 19. -/
theorem tenth_term_of_arithmetic_sequence : 
  ∀ (a : ℕ → ℚ), 
    (∀ n : ℕ, n < 19 → a (n + 1) - a n = a 1 - a 0) →  -- arithmetic sequence condition
    a 0 = 7 →                                         -- first term
    a 19 = 67 →                                       -- last term
    a 9 = 673 / 19 :=                                 -- 10th term (index 9)
by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l2692_269273


namespace NUMINAMATH_CALUDE_brooks_theorem_l2692_269276

/-- A graph represented by its vertex set and an adjacency relation -/
structure Graph (V : Type*) where
  adj : V → V → Prop

/-- The maximum degree of a graph -/
def maxDegree {V : Type*} (G : Graph V) : ℕ :=
  sorry

/-- The chromatic number of a graph -/
def chromaticNumber {V : Type*} (G : Graph V) : ℕ :=
  sorry

/-- Brooks' theorem: The chromatic number of a graph is at most one more than its maximum degree -/
theorem brooks_theorem {V : Type*} (G : Graph V) :
  chromaticNumber G ≤ maxDegree G + 1 :=
sorry

end NUMINAMATH_CALUDE_brooks_theorem_l2692_269276


namespace NUMINAMATH_CALUDE_constant_distance_special_points_min_distance_to_origin_euclidean_vs_orthogonal_distance_l2692_269211

-- Define orthogonal distance
def orthogonal_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

-- Proposition 1
theorem constant_distance_special_points :
  ∀ α : ℝ, orthogonal_distance 2 3 (Real.sin α ^ 2) (Real.cos α ^ 2) = 4 :=
sorry

-- Proposition 2 (negation)
theorem min_distance_to_origin :
  ∃ x y : ℝ, x - y + 1 = 0 ∧ |x| + |y| < 1 :=
sorry

-- Proposition 3
theorem euclidean_vs_orthogonal_distance :
  ∀ x₁ y₁ x₂ y₂ : ℝ,
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ (Real.sqrt 2 / 2) * (|x₁ - x₂| + |y₁ - y₂|) :=
sorry

end NUMINAMATH_CALUDE_constant_distance_special_points_min_distance_to_origin_euclidean_vs_orthogonal_distance_l2692_269211


namespace NUMINAMATH_CALUDE_sqrt_four_minus_one_l2692_269215

theorem sqrt_four_minus_one : Real.sqrt 4 - 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_minus_one_l2692_269215


namespace NUMINAMATH_CALUDE_housing_price_growth_equation_l2692_269298

/-- Proves that the equation for average annual growth rate of housing prices is correct -/
theorem housing_price_growth_equation (initial_price final_price : ℝ) (growth_rate : ℝ) 
  (h1 : initial_price = 8100)
  (h2 : final_price = 12500)
  (h3 : growth_rate ≥ 0)
  (h4 : growth_rate < 1) :
  initial_price * (1 + growth_rate)^2 = final_price := by
  sorry

end NUMINAMATH_CALUDE_housing_price_growth_equation_l2692_269298


namespace NUMINAMATH_CALUDE_no_increasing_function_with_properties_l2692_269260

theorem no_increasing_function_with_properties :
  ¬ ∃ (f : ℕ → ℕ),
    (∀ (a b : ℕ), a < b → f a < f b) ∧
    (f 2 = 2) ∧
    (∀ (n m : ℕ), f (n * m) = f n + f m) := by
  sorry

end NUMINAMATH_CALUDE_no_increasing_function_with_properties_l2692_269260


namespace NUMINAMATH_CALUDE_will_picked_up_38_sticks_l2692_269274

/-- The number of sticks originally in the yard -/
def original_sticks : ℕ := 99

/-- The number of sticks left after Will picked some up -/
def remaining_sticks : ℕ := 61

/-- The number of sticks Will picked up -/
def picked_up_sticks : ℕ := original_sticks - remaining_sticks

theorem will_picked_up_38_sticks : picked_up_sticks = 38 := by
  sorry

end NUMINAMATH_CALUDE_will_picked_up_38_sticks_l2692_269274


namespace NUMINAMATH_CALUDE_square_diagonal_point_theorem_l2692_269241

/-- Square with side length 12 -/
structure Square :=
  (side : ℝ)
  (is_twelve : side = 12)

/-- Point on the diagonal of the square -/
structure DiagonalPoint (s : Square) :=
  (x : ℝ)
  (y : ℝ)
  (on_diagonal : y = x)
  (in_square : 0 < x ∧ x < s.side)

/-- Circumcenter of a right triangle -/
def circumcenter (a b c : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Angle between three points -/
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem square_diagonal_point_theorem (s : Square) (p : DiagonalPoint s)
  (o1 : ℝ × ℝ) (o2 : ℝ × ℝ)
  (h1 : o1 = circumcenter (0, 0) (s.side, 0) (p.x, p.y))
  (h2 : o2 = circumcenter (s.side, s.side) (0, s.side) (p.x, p.y))
  (h3 : angle o1 (p.x, p.y) o2 = 120)
  : ∃ (a b : ℕ), (p.x : ℝ) = Real.sqrt a + Real.sqrt b ∧ a + b = 96 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_point_theorem_l2692_269241


namespace NUMINAMATH_CALUDE_no_simultaneous_properties_l2692_269282

theorem no_simultaneous_properties : ¬∃ (star : ℤ → ℤ → ℤ),
  (∀ Z : ℤ, ∃ X Y : ℤ, star X Y = Z) ∧
  (∀ A B : ℤ, star A B = -(star B A)) ∧
  (∀ A B C : ℤ, star (star A B) C = star A (star B C)) :=
by sorry

end NUMINAMATH_CALUDE_no_simultaneous_properties_l2692_269282


namespace NUMINAMATH_CALUDE_binomial_1000_500_not_divisible_by_7_l2692_269299

theorem binomial_1000_500_not_divisible_by_7 : ¬ (7 ∣ Nat.choose 1000 500) := by
  sorry

end NUMINAMATH_CALUDE_binomial_1000_500_not_divisible_by_7_l2692_269299


namespace NUMINAMATH_CALUDE_min_coefficient_value_l2692_269216

theorem min_coefficient_value (a b : ℤ) (box : ℤ) : 
  (∀ x, (a * x + b) * (b * x + a) = 15 * x^2 + box * x + 15) →
  a ≠ b ∧ a ≠ box ∧ b ≠ box →
  ∃ min_box : ℤ, (min_box = 34 ∧ box ≥ min_box) := by
  sorry

end NUMINAMATH_CALUDE_min_coefficient_value_l2692_269216
