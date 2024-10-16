import Mathlib

namespace NUMINAMATH_CALUDE_no_prime_perfect_square_l21_2127

theorem no_prime_perfect_square : ¬∃ (p : ℕ), Prime p ∧ ∃ (a : ℕ), 7 * p + 3^p - 4 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_perfect_square_l21_2127


namespace NUMINAMATH_CALUDE_building_floor_ratio_l21_2157

/-- Given three buildings A, B, and C, where:
  * Building A has 4 floors
  * Building B has 9 more floors than Building A
  * Building C has 59 floors
Prove that the ratio of floors in Building C to Building B is 59/13 -/
theorem building_floor_ratio : 
  (floors_A : ℕ) → 
  (floors_B : ℕ) → 
  (floors_C : ℕ) → 
  floors_A = 4 →
  floors_B = floors_A + 9 →
  floors_C = 59 →
  (floors_C : ℚ) / floors_B = 59 / 13 := by
sorry

end NUMINAMATH_CALUDE_building_floor_ratio_l21_2157


namespace NUMINAMATH_CALUDE_function_ordering_l21_2181

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define the property of being monotonically decreasing on an interval
def is_monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

-- State the theorem
theorem function_ordering (h1 : is_even f) (h2 : is_monotone_decreasing_on (fun x ↦ f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 :=
sorry

end NUMINAMATH_CALUDE_function_ordering_l21_2181


namespace NUMINAMATH_CALUDE_unique_solution_l21_2171

theorem unique_solution : ∃! x : ℕ, 
  (∃ k : ℕ, x = 7 * k) ∧ 
  x^3 < 8000 ∧ 
  10 < x ∧ x < 30 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l21_2171


namespace NUMINAMATH_CALUDE_f_expression_l21_2172

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_expression : 
  (∀ x : ℝ, f (x + 1) = 3 * x + 2) → 
  (∀ x : ℝ, f x = 3 * x - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_f_expression_l21_2172


namespace NUMINAMATH_CALUDE_complement_of_alpha_l21_2162

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the given angle
def alpha : Angle := ⟨75, 12⟩

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem complement_of_alpha :
  complement alpha = ⟨14, 48⟩ := by
  sorry

end NUMINAMATH_CALUDE_complement_of_alpha_l21_2162


namespace NUMINAMATH_CALUDE_units_digit_p_plus_4_l21_2169

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Definition of a positive even integer with a positive units digit -/
def isPositiveEvenWithPositiveUnitsDigit (p : ℕ) : Prop :=
  p > 0 ∧ p % 2 = 0 ∧ unitsDigit p > 0

/-- The main theorem -/
theorem units_digit_p_plus_4 (p : ℕ) 
  (h1 : isPositiveEvenWithPositiveUnitsDigit p)
  (h2 : unitsDigit (p^3) - unitsDigit (p^2) = 0) :
  unitsDigit (p + 4) = 0 := by
sorry

end NUMINAMATH_CALUDE_units_digit_p_plus_4_l21_2169


namespace NUMINAMATH_CALUDE_correct_remaining_leaves_l21_2185

/-- Calculates the number of remaining leaves on a tree at the end of summer --/
def remaining_leaves (branches : ℕ) (twigs_per_branch : ℕ) 
  (spring_3_leaf_percent : ℚ) (spring_4_leaf_percent : ℚ) (spring_5_leaf_percent : ℚ)
  (summer_leaf_increase : ℕ) (caterpillar_eaten_percent : ℚ) : ℕ :=
  sorry

/-- Theorem stating the correct number of remaining leaves --/
theorem correct_remaining_leaves :
  remaining_leaves 100 150 (20/100) (30/100) (50/100) 2 (10/100) = 85050 :=
sorry

end NUMINAMATH_CALUDE_correct_remaining_leaves_l21_2185


namespace NUMINAMATH_CALUDE_max_citizens_for_minister_l21_2110

theorem max_citizens_for_minister (n : ℕ) : 
  (∀ m : ℕ, m > n → Nat.choose m 4 ≥ Nat.choose m 2) ↔ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_max_citizens_for_minister_l21_2110


namespace NUMINAMATH_CALUDE_equation_solution_l21_2177

theorem equation_solution : ∃ x : ℝ, x > 0 ∧ 4 * x^(1/3) - 2 * (x / x^(2/3)) = 7 + x^(1/3) ∧ x = 343 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l21_2177


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l21_2126

/-- Simple interest calculation -/
theorem simple_interest_calculation (principal rate time : ℝ) :
  principal = 400 →
  rate = 22.5 →
  time = 2 →
  (principal * rate * time) / 100 = 90 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l21_2126


namespace NUMINAMATH_CALUDE_ellipse_theorem_l21_2165

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line y = kx + m -/
structure Line where
  k : ℝ
  m : ℝ

/-- The main theorem about the ellipse and the range of |OP| -/
theorem ellipse_theorem (C : Ellipse) (l : Line) : 
  C.a^2 = 4 ∧ C.b^2 = 2 ∧ abs l.k ≤ Real.sqrt 2 / 2 →
  (∀ x y : ℝ, x^2 / C.a^2 + y^2 / C.b^2 = 1 ↔ x^2 / 4 + y^2 / 2 = 1) ∧
  (∀ P : ℝ × ℝ, P.1^2 / 4 + P.2^2 / 2 = 1 → 
    Real.sqrt 2 ≤ Real.sqrt (P.1^2 + P.2^2) ∧ 
    Real.sqrt (P.1^2 + P.2^2) ≤ Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_theorem_l21_2165


namespace NUMINAMATH_CALUDE_adrian_holidays_in_year_l21_2164

/-- The number of holidays Adrian took in a year -/
def adriansHolidays (daysOffPerMonth : ℕ) (monthsInYear : ℕ) : ℕ :=
  daysOffPerMonth * monthsInYear

/-- Theorem: Adrian took 48 holidays in the entire year -/
theorem adrian_holidays_in_year : 
  adriansHolidays 4 12 = 48 := by
  sorry

end NUMINAMATH_CALUDE_adrian_holidays_in_year_l21_2164


namespace NUMINAMATH_CALUDE_annual_population_increase_rounded_l21_2130

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of hours between births -/
def hours_per_birth : ℕ := 6

/-- The number of hours between deaths -/
def hours_per_death : ℕ := 10

/-- The number of days in a year -/
def days_per_year : ℕ := 365

/-- Calculate the annual population increase -/
def annual_population_increase : ℕ :=
  (hours_per_day / hours_per_birth - hours_per_day / hours_per_death) * days_per_year

/-- Round to the nearest hundred -/
def round_to_hundred (n : ℕ) : ℕ :=
  ((n + 50) / 100) * 100

/-- Theorem stating the annual population increase rounded to the nearest hundred -/
theorem annual_population_increase_rounded :
  round_to_hundred annual_population_increase = 700 := by
  sorry

end NUMINAMATH_CALUDE_annual_population_increase_rounded_l21_2130


namespace NUMINAMATH_CALUDE_z_value_z_value_proof_l21_2104

theorem z_value : ℝ → ℝ → ℝ → Prop :=
  fun x y z =>
    x = 40 * (1 + 0.2) →
    y = x * (1 - 0.35) →
    z = (x + y) / 2 →
    z = 39.6

-- Proof
theorem z_value_proof : ∃ x y z : ℝ, z_value x y z := by
  sorry

end NUMINAMATH_CALUDE_z_value_z_value_proof_l21_2104


namespace NUMINAMATH_CALUDE_exponent_multiplication_l21_2119

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l21_2119


namespace NUMINAMATH_CALUDE_base8_calculation_l21_2146

-- Define a function to convert base 8 to natural numbers
def base8ToNat (x : ℕ) : ℕ := sorry

-- Define a function to convert natural numbers to base 8
def natToBase8 (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem base8_calculation : 
  natToBase8 ((base8ToNat 452 - base8ToNat 126) + base8ToNat 237) = 603 := by
  sorry

end NUMINAMATH_CALUDE_base8_calculation_l21_2146


namespace NUMINAMATH_CALUDE_central_cell_value_l21_2158

theorem central_cell_value (a b c d e f g h i : ℝ) 
  (row_prod : a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10)
  (col_prod : a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10)
  (square_prod : a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3) :
  e = 0.00081 := by
sorry

end NUMINAMATH_CALUDE_central_cell_value_l21_2158


namespace NUMINAMATH_CALUDE_circular_garden_radius_l21_2129

/-- 
Given a circular garden with radius r, if the length of the fence (circumference) 
is 1/8 of the area of the garden, then r = 16.
-/
theorem circular_garden_radius (r : ℝ) (h : r > 0) : 
  2 * π * r = (1 / 8) * π * r^2 → r = 16 := by sorry

end NUMINAMATH_CALUDE_circular_garden_radius_l21_2129


namespace NUMINAMATH_CALUDE_rectangle_length_l21_2144

/-- 
Given a rectangular garden with perimeter 950 meters and breadth 100 meters, 
this theorem proves that its length is 375 meters.
-/
theorem rectangle_length (perimeter breadth : ℝ) 
  (h_perimeter : perimeter = 950)
  (h_breadth : breadth = 100) :
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter :=
by
  sorry

#check rectangle_length

end NUMINAMATH_CALUDE_rectangle_length_l21_2144


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_six_l21_2106

theorem arctan_sum_equals_pi_over_six (b : ℝ) :
  (4/3 : ℝ) * (b + 1) = 3/2 →
  Real.arctan (1/3) + Real.arctan b = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_over_six_l21_2106


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l21_2114

theorem cubic_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
  a^3 + b^3 + c^3 = -36 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l21_2114


namespace NUMINAMATH_CALUDE_min_keys_required_l21_2154

/-- Represents the hotel key distribution problem -/
structure HotelKeyProblem where
  rooms : ℕ
  guests : ℕ
  returningGuests : ℕ
  keys : ℕ

/-- Predicate to check if a key distribution is valid -/
def isValidDistribution (p : HotelKeyProblem) : Prop :=
  p.rooms > 0 ∧ 
  p.guests > p.rooms ∧ 
  p.returningGuests = p.rooms ∧
  ∀ (subset : Finset ℕ), subset.card = p.returningGuests → 
    ∃ (f : subset → Fin p.rooms), Function.Injective f

/-- Theorem stating the minimum number of keys required -/
theorem min_keys_required (p : HotelKeyProblem) 
  (h : isValidDistribution p) : p.keys ≥ p.rooms * (p.guests - p.rooms + 1) :=
sorry

end NUMINAMATH_CALUDE_min_keys_required_l21_2154


namespace NUMINAMATH_CALUDE_inequality_solution_l21_2193

theorem inequality_solution (x : ℝ) : 
  (x + 1 ≠ 0) → ((2 - x) / (x + 1) ≥ 0 ↔ -1 < x ∧ x ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l21_2193


namespace NUMINAMATH_CALUDE_ceiling_square_minus_fraction_l21_2123

theorem ceiling_square_minus_fraction : ⌈((-7/4)^2 - 1/8)⌉ = 3 := by sorry

end NUMINAMATH_CALUDE_ceiling_square_minus_fraction_l21_2123


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l21_2175

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^2 ≥ 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l21_2175


namespace NUMINAMATH_CALUDE_tempo_value_calculation_l21_2151

/-- The original value of a tempo given insurance and premium information -/
def tempoOriginalValue (insuredFraction : ℚ) (premiumRate : ℚ) (premiumAmount : ℚ) : ℚ :=
  premiumAmount / (premiumRate * insuredFraction)

/-- Theorem stating the original value of the tempo given the problem conditions -/
theorem tempo_value_calculation :
  let insuredFraction : ℚ := 4 / 5
  let premiumRate : ℚ := 13 / 1000
  let premiumAmount : ℚ := 910
  tempoOriginalValue insuredFraction premiumRate premiumAmount = 87500 := by
  sorry

#eval tempoOriginalValue (4/5) (13/1000) 910

end NUMINAMATH_CALUDE_tempo_value_calculation_l21_2151


namespace NUMINAMATH_CALUDE_min_length_for_prob_threshold_l21_2161

/-- The probability that a random sequence of length n using digits 0, 1, and 2 does not contain all three digits -/
def prob_not_all_digits (n : ℕ) : ℚ :=
  (2^n - 1) / 3^(n-1)

/-- The probability that a random sequence of length n using digits 0, 1, and 2 contains all three digits -/
def prob_all_digits (n : ℕ) : ℚ :=
  1 - prob_not_all_digits n

theorem min_length_for_prob_threshold :
  prob_all_digits 5 ≥ 61/100 ∧
  ∀ k < 5, prob_all_digits k < 61/100 :=
sorry

end NUMINAMATH_CALUDE_min_length_for_prob_threshold_l21_2161


namespace NUMINAMATH_CALUDE_second_year_sample_size_l21_2155

/-- Represents the ratio of students in first, second, and third grades -/
structure GradeRatio where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the number of students in a specific grade for a stratified sample -/
def stratifiedSampleSize (ratio : GradeRatio) (totalSample : ℕ) (grade : ℕ) : ℕ :=
  (totalSample * grade) / (ratio.first + ratio.second + ratio.third)

/-- Theorem: In a stratified sample of 240 students with a grade ratio of 5:4:3,
    the number of second-year students in the sample is 80 -/
theorem second_year_sample_size :
  let ratio : GradeRatio := { first := 5, second := 4, third := 3 }
  stratifiedSampleSize ratio 240 ratio.second = 80 := by
  sorry

end NUMINAMATH_CALUDE_second_year_sample_size_l21_2155


namespace NUMINAMATH_CALUDE_tangent_line_to_sine_curve_l21_2150

theorem tangent_line_to_sine_curve (x y : ℝ) :
  let f : ℝ → ℝ := λ t => Real.sin (t + Real.pi / 3)
  let point : ℝ × ℝ := (0, Real.sqrt 3 / 2)
  let tangent_equation : ℝ → ℝ → Prop := λ x y => x - 2 * y + Real.sqrt 3 = 0
  (∀ t, f t = Real.sin (t + Real.pi / 3)) →
  (point.1 = 0 ∧ point.2 = Real.sqrt 3 / 2) →
  (∃ k, ∀ x, tangent_equation x (k * x + point.2)) →
  tangent_equation x y = (x - 2 * y + Real.sqrt 3 = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_tangent_line_to_sine_curve_l21_2150


namespace NUMINAMATH_CALUDE_AP_coordinates_l21_2152

-- Define the vectors OA and OB
def OA : ℝ × ℝ × ℝ := (1, -1, 1)
def OB : ℝ × ℝ × ℝ := (2, 0, -1)

-- Define point P on line segment AB
def P : ℝ × ℝ × ℝ := sorry

-- Define the condition AP = 2PB
def AP_eq_2PB : ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ P = (1 - t) • OA + t • OB ∧ t = 2/3 := sorry

-- Theorem to prove
theorem AP_coordinates : 
  let AP := P - OA
  AP = (2/3, 2/3, -4/3) :=
sorry

end NUMINAMATH_CALUDE_AP_coordinates_l21_2152


namespace NUMINAMATH_CALUDE_smallest_number_with_divisibility_properties_l21_2124

theorem smallest_number_with_divisibility_properties : ∃ n : ℕ, 
  (∀ m : ℕ, m ≥ n → (m % 8 = 0 ∧ (∀ k : ℕ, k ≥ 3 ∧ k ≤ 7 → m % k = 2)) → m ≥ 1682) ∧
  1682 % 8 = 0 ∧
  (∀ k : ℕ, k ≥ 3 ∧ k ≤ 7 → 1682 % k = 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_divisibility_properties_l21_2124


namespace NUMINAMATH_CALUDE_smallest_n_with_conditions_n_9000_satisfies_conditions_l21_2168

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def contains_digit_9 (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∈ n.digits 10 ∧ d = 9

theorem smallest_n_with_conditions :
  ∀ n : ℕ, n > 0 →
    (is_terminating_decimal n ∧ contains_digit_9 n ∧ n % 3 = 0) →
    n ≥ 9000 :=
by sorry

theorem n_9000_satisfies_conditions :
  is_terminating_decimal 9000 ∧ contains_digit_9 9000 ∧ 9000 % 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_conditions_n_9000_satisfies_conditions_l21_2168


namespace NUMINAMATH_CALUDE_same_color_probability_l21_2159

/-- The probability of drawing two balls of the same color from a box containing
    4 white balls and 2 black balls when drawing two balls at once. -/
theorem same_color_probability (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ)
    (h_total : total_balls = white_balls + black_balls)
    (h_white : white_balls = 4)
    (h_black : black_balls = 2) :
    (Nat.choose white_balls 2 + Nat.choose black_balls 2) / Nat.choose total_balls 2 = 7 / 15 :=
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l21_2159


namespace NUMINAMATH_CALUDE_circle_no_intersection_probability_l21_2105

/-- A rectangle with width 15 and height 36 -/
structure Rectangle where
  width : ℝ := 15
  height : ℝ := 36

/-- A circle with radius 1 -/
structure Circle where
  radius : ℝ := 1

/-- The probability that a circle doesn't intersect the diagonal of a rectangle -/
def probability_no_intersection (r : Rectangle) (c : Circle) : ℚ :=
  375 / 442

/-- Theorem stating the probability of no intersection -/
theorem circle_no_intersection_probability (r : Rectangle) (c : Circle) :
  probability_no_intersection r c = 375 / 442 := by
  sorry

#check circle_no_intersection_probability

end NUMINAMATH_CALUDE_circle_no_intersection_probability_l21_2105


namespace NUMINAMATH_CALUDE_marble_distribution_l21_2122

theorem marble_distribution (n : ℕ) (hn : n = 480) :
  (Finset.filter (fun m => m > 1 ∧ m < n) (Finset.range (n + 1))).card = 22 :=
by sorry

end NUMINAMATH_CALUDE_marble_distribution_l21_2122


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_eleven_l21_2147

-- Define the equation that x satisfies
def satisfies_equation (x : ℝ) : Prop :=
  x^2 + 4*x + 4/x + 1/x^2 = 34

-- Define the condition that x can be written as a + √b
def can_be_written_as_a_plus_sqrt_b (x : ℝ) : Prop :=
  ∃ (a b : ℕ), x = a + Real.sqrt b ∧ a > 0 ∧ b > 0

-- State the theorem
theorem sum_of_a_and_b_is_eleven :
  ∀ x : ℝ, satisfies_equation x → can_be_written_as_a_plus_sqrt_b x →
  ∃ (a b : ℕ), x = a + Real.sqrt b ∧ a > 0 ∧ b > 0 ∧ a + b = 11 :=
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_eleven_l21_2147


namespace NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l21_2187

theorem sum_and_ratio_to_difference (m n : ℝ) 
  (sum_eq : m + n = 490)
  (ratio_eq : m / n = 1.2) : 
  ∃ (diff : ℝ), abs (m - n - diff) < 0.5 ∧ diff = 45 :=
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l21_2187


namespace NUMINAMATH_CALUDE_inscribed_half_area_rhombus_l21_2145

/-- A centrally symmetric convex polygon -/
structure CentrallySymmetricConvexPolygon where
  -- Add necessary fields and properties
  area : ℝ
  centrally_symmetric : Bool
  convex : Bool

/-- A rhombus -/
structure Rhombus where
  -- Add necessary fields
  area : ℝ

/-- A rhombus is inscribed in a polygon -/
def is_inscribed (r : Rhombus) (p : CentrallySymmetricConvexPolygon) : Prop :=
  sorry

/-- Main theorem: For any centrally symmetric convex polygon, 
    there exists an inscribed rhombus with half the area of the polygon -/
theorem inscribed_half_area_rhombus (p : CentrallySymmetricConvexPolygon) :
  ∃ r : Rhombus, is_inscribed r p ∧ r.area = p.area / 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_half_area_rhombus_l21_2145


namespace NUMINAMATH_CALUDE_at_least_two_primes_of_form_l21_2156

theorem at_least_two_primes_of_form (n : ℕ) : ∃ (a b : ℕ), 2 ≤ a ∧ 2 ≤ b ∧ a ≠ b ∧ 
  Nat.Prime (a^3 + a^2 + 1) ∧ Nat.Prime (b^3 + b^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_primes_of_form_l21_2156


namespace NUMINAMATH_CALUDE_tile_arrangements_l21_2136

def brown_tiles : ℕ := 1
def purple_tiles : ℕ := 2
def green_tiles : ℕ := 3
def yellow_tiles : ℕ := 2

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem tile_arrangements :
  (Nat.factorial total_tiles) / 
  (Nat.factorial brown_tiles * Nat.factorial purple_tiles * 
   Nat.factorial green_tiles * Nat.factorial yellow_tiles) = 1680 := by
  sorry

end NUMINAMATH_CALUDE_tile_arrangements_l21_2136


namespace NUMINAMATH_CALUDE_circle_equation_coefficients_l21_2125

theorem circle_equation_coefficients (D E F : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + D*x + E*y + F = 0 ↔ (x + 2)^2 + (y - 3)^2 = 4^2) →
  D = 4 ∧ E = -6 ∧ F = -3 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_coefficients_l21_2125


namespace NUMINAMATH_CALUDE_system_solution_l21_2179

/-- Given a system of equations, prove that t = 24 -/
theorem system_solution (p t j x y a b c : ℝ) 
  (h1 : j = 0.75 * p)
  (h2 : j = 0.8 * t)
  (h3 : t = p - (t / 100) * p)
  (h4 : x = 0.1 * t)
  (h5 : y = 0.5 * j)
  (h6 : x + y = 12)
  (h7 : a = x + y)
  (h8 : b = 0.15 * a)
  (h9 : c = 2 * b) :
  t = 24 := by sorry

end NUMINAMATH_CALUDE_system_solution_l21_2179


namespace NUMINAMATH_CALUDE_garrison_provision_days_l21_2148

/-- The number of days provisions last for a garrison --/
def provisionDays (initialMen : ℕ) (reinforcementMen : ℕ) (daysBeforeReinforcement : ℕ) (daysAfterReinforcement : ℕ) : ℕ :=
  (initialMen * daysBeforeReinforcement + (initialMen + reinforcementMen) * daysAfterReinforcement) / initialMen

theorem garrison_provision_days :
  provisionDays 2000 2700 15 20 = 62 := by
  sorry

#eval provisionDays 2000 2700 15 20

end NUMINAMATH_CALUDE_garrison_provision_days_l21_2148


namespace NUMINAMATH_CALUDE_abc_def_ratio_l21_2182

theorem abc_def_ratio (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 2) :
  a * b * c / (d * e * f) = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_abc_def_ratio_l21_2182


namespace NUMINAMATH_CALUDE_percentage_difference_l21_2102

theorem percentage_difference (A B C x : ℝ) : 
  A > 0 → B > 0 → C > 0 → 
  A > C → C > B → 
  A = B * (1 + x / 100) → 
  C = 0.75 * A → 
  x > 100 / 3 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l21_2102


namespace NUMINAMATH_CALUDE_square_difference_division_l21_2199

theorem square_difference_division (a b c : ℕ) (h : (a^2 - b^2) / c = 488) :
  (144^2 - 100^2) / 22 = 488 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_division_l21_2199


namespace NUMINAMATH_CALUDE_sum_of_coefficients_bounds_l21_2163

/-- A quadratic function with vertex in the first quadrant passing through (0,1) and (-1,0) -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  vertex_first_quadrant : -b / (2 * a) > 0
  passes_through_0_1 : c = 1
  passes_through_neg1_0 : a - b + c = 0

/-- The sum of coefficients of a quadratic function -/
def S (f : QuadraticFunction) : ℝ := f.a + f.b + f.c

/-- Theorem: The sum of coefficients S is between 0 and 2 -/
theorem sum_of_coefficients_bounds (f : QuadraticFunction) : 0 < S f ∧ S f < 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_bounds_l21_2163


namespace NUMINAMATH_CALUDE_parabola_C_passes_through_origin_l21_2170

-- Define the parabolas
def parabola_A (x : ℝ) : ℝ := x^2 + 1
def parabola_B (x : ℝ) : ℝ := (x + 1)^2
def parabola_C (x : ℝ) : ℝ := x^2 + 2*x
def parabola_D (x : ℝ) : ℝ := x^2 - x + 1

-- Define what it means for a parabola to pass through the origin
def passes_through_origin (f : ℝ → ℝ) : Prop := f 0 = 0

-- Theorem stating that parabola C passes through the origin while others do not
theorem parabola_C_passes_through_origin :
  passes_through_origin parabola_C ∧
  ¬passes_through_origin parabola_A ∧
  ¬passes_through_origin parabola_B ∧
  ¬passes_through_origin parabola_D :=
by sorry

end NUMINAMATH_CALUDE_parabola_C_passes_through_origin_l21_2170


namespace NUMINAMATH_CALUDE_quadratic_equation_translation_l21_2153

/-- Quadratic form in two variables -/
def Q (a b c : ℝ) (x y : ℝ) : ℝ := a * x^2 + 2 * b * x * y + c * y^2

/-- Theorem: Transformation of quadratic equation using parallel translation -/
theorem quadratic_equation_translation
  (a b c d e f x₀ y₀ : ℝ)
  (h : a * c - b^2 ≠ 0) :
  ∃ f' : ℝ,
    (∀ x y, Q a b c x y + 2 * d * x + 2 * e * y = f) ↔
    (∀ x' y', Q a b c x' y' = f' ∧
      x' = x + x₀ ∧
      y' = y + y₀ ∧
      f' = f - Q a b c x₀ y₀ + 2 * (d * x₀ + e * y₀)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_translation_l21_2153


namespace NUMINAMATH_CALUDE_pen_price_l21_2120

theorem pen_price (total_pens : ℕ) (total_cost : ℚ) (regular_price : ℚ) : 
  total_pens = 20 ∧ total_cost = 30 ∧ 
  (regular_price * (total_pens / 2) + (regular_price / 2) * (total_pens / 2) = total_cost) →
  regular_price = 2 := by
sorry

end NUMINAMATH_CALUDE_pen_price_l21_2120


namespace NUMINAMATH_CALUDE_checking_account_theorem_l21_2190

/-- Given the total amount of yen and the amount in the savings account,
    calculate the amount in the checking account. -/
def checking_account_balance (total : ℕ) (savings : ℕ) : ℕ :=
  total - savings

/-- Theorem stating that given the total amount and savings account balance,
    the checking account balance is correctly calculated. -/
theorem checking_account_theorem (total savings : ℕ) 
  (h1 : total = 9844)
  (h2 : savings = 3485) :
  checking_account_balance total savings = 6359 := by
  sorry

end NUMINAMATH_CALUDE_checking_account_theorem_l21_2190


namespace NUMINAMATH_CALUDE_increase_by_percentage_l21_2108

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 80 → percentage = 150 → result = initial * (1 + percentage / 100) → result = 200 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l21_2108


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l21_2141

/-- Proves the number of adult tickets sold given ticket prices and total sales information -/
theorem adult_tickets_sold
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_revenue : ℕ)
  (total_tickets : ℕ)
  (h1 : adult_price = 8)
  (h2 : child_price = 5)
  (h3 : total_revenue = 236)
  (h4 : total_tickets = 34)
  : ∃ (adult_tickets : ℕ) (child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_tickets * adult_price + child_tickets * child_price = total_revenue ∧
    adult_tickets = 22 := by
  sorry


end NUMINAMATH_CALUDE_adult_tickets_sold_l21_2141


namespace NUMINAMATH_CALUDE_cody_additional_tickets_l21_2116

/-- Calculates the number of additional tickets won given initial tickets, tickets spent, and final tickets. -/
def additional_tickets_won (initial_tickets spent_tickets final_tickets : ℕ) : ℕ :=
  final_tickets - (initial_tickets - spent_tickets)

/-- Proves that Cody won 6 additional tickets given the problem conditions. -/
theorem cody_additional_tickets :
  let initial_tickets := 49
  let spent_tickets := 25
  let final_tickets := 30
  additional_tickets_won initial_tickets spent_tickets final_tickets = 6 := by
  sorry

end NUMINAMATH_CALUDE_cody_additional_tickets_l21_2116


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l21_2107

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x + 1)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (2 * x - 1)

theorem min_distance_between_curves :
  ∃ (min_dist : ℝ), 
    (∀ (x₁ x₂ : ℝ), f x₁ = g x₂ → |x₂ - x₁| ≥ min_dist) ∧
    (∃ (x₁ x₂ : ℝ), f x₁ = g x₂ ∧ |x₂ - x₁| = min_dist) ∧
    min_dist = (5 + Real.log 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l21_2107


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l21_2111

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x^2 - x < 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l21_2111


namespace NUMINAMATH_CALUDE_pythagorean_inequality_l21_2196

theorem pythagorean_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h5 : a^2 + b^2 = c^2 + a*b) :
  c^2 + a*b < a*c + b*c := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_inequality_l21_2196


namespace NUMINAMATH_CALUDE_milk_water_ratio_after_addition_l21_2195

theorem milk_water_ratio_after_addition 
  (initial_volume : ℝ) 
  (initial_milk_ratio : ℝ) 
  (initial_water_ratio : ℝ) 
  (added_water : ℝ) :
  initial_volume = 45 ∧ 
  initial_milk_ratio = 4 ∧ 
  initial_water_ratio = 1 ∧ 
  added_water = 3 →
  let initial_total_ratio := initial_milk_ratio + initial_water_ratio
  let initial_milk_volume := (initial_milk_ratio / initial_total_ratio) * initial_volume
  let initial_water_volume := (initial_water_ratio / initial_total_ratio) * initial_volume
  let final_milk_volume := initial_milk_volume
  let final_water_volume := initial_water_volume + added_water
  let final_ratio := final_milk_volume / final_water_volume
  final_ratio = 3 := by
sorry

end NUMINAMATH_CALUDE_milk_water_ratio_after_addition_l21_2195


namespace NUMINAMATH_CALUDE_jose_wandering_time_l21_2112

/-- Proves that Jose's wandering time is 10 hours given his distance and speed -/
theorem jose_wandering_time : 
  ∀ (distance : ℝ) (speed : ℝ),
  distance = 15 →
  speed = 1.5 →
  distance / speed = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jose_wandering_time_l21_2112


namespace NUMINAMATH_CALUDE_remainder_of_a_83_mod_49_l21_2103

theorem remainder_of_a_83_mod_49 : (6^83 + 8^83) % 49 = 35 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_a_83_mod_49_l21_2103


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achievable_l21_2176

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_one : x + y + z = 1) : 
  1/x + 4/y + 9/z ≥ 36 :=
by
  sorry

theorem min_value_achievable : 
  ∃ (x y z : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ x + y + z = 1 ∧ 1/x + 4/y + 9/z = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_min_value_achievable_l21_2176


namespace NUMINAMATH_CALUDE_sin_has_property_T_l21_2183

-- Define property T
def has_property_T (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2 : ℝ, (deriv f x1) * (deriv f x2) = -1

-- State the theorem
theorem sin_has_property_T : has_property_T Real.sin := by
  sorry


end NUMINAMATH_CALUDE_sin_has_property_T_l21_2183


namespace NUMINAMATH_CALUDE_total_points_after_perfect_games_l21_2143

/-- A perfect score in a game -/
def perfect_score : ℕ := 21

/-- The number of perfect games played -/
def games_played : ℕ := 3

/-- Theorem: The total points after 3 perfect games is 63 -/
theorem total_points_after_perfect_games : 
  perfect_score * games_played = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_points_after_perfect_games_l21_2143


namespace NUMINAMATH_CALUDE_jake_buys_three_packages_l21_2194

/-- Represents the number of sausage packages Jake buys -/
def num_packages : ℕ := 3

/-- Represents the weight of each sausage package in pounds -/
def package_weight : ℕ := 2

/-- Represents the price per pound of sausages in dollars -/
def price_per_pound : ℕ := 4

/-- Represents the total amount Jake pays in dollars -/
def total_paid : ℕ := 24

/-- Theorem stating that Jake buys 3 packages of sausages -/
theorem jake_buys_three_packages : 
  num_packages * package_weight * price_per_pound = total_paid :=
by sorry

end NUMINAMATH_CALUDE_jake_buys_three_packages_l21_2194


namespace NUMINAMATH_CALUDE_tan_expression_equals_neg_sqrt_three_l21_2131

/-- A sequence is a geometric progression -/
def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- A sequence is an arithmetic progression -/
def is_arithmetic_progression (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

/-- Main theorem -/
theorem tan_expression_equals_neg_sqrt_three
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_geom : is_geometric_progression a)
  (h_arith : is_arithmetic_progression b)
  (h_prod : a 1 * a 6 * a 11 = -3 * Real.sqrt 3)
  (h_sum : b 1 + b 6 + b 11 = 7 * Real.pi) :
  Real.tan ((b 3 + b 9) / (1 - a 4 * a 8)) = -Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_tan_expression_equals_neg_sqrt_three_l21_2131


namespace NUMINAMATH_CALUDE_sin_830_equality_l21_2189

theorem sin_830_equality (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.sin (830 * π / 180) → 
  n = 70 ∨ n = 110 := by
  sorry

end NUMINAMATH_CALUDE_sin_830_equality_l21_2189


namespace NUMINAMATH_CALUDE_arc_length_for_unit_angle_l21_2160

/-- Given a circle where the chord length corresponding to a central angle of 1 radian is 2,
    prove that the arc length corresponding to this central angle is 1/sin(1/2). -/
theorem arc_length_for_unit_angle (r : ℝ) : 
  (2 * r * Real.sin (1 / 2) = 2) → 
  (r * 1 = 1 / Real.sin (1 / 2)) := by
  sorry

end NUMINAMATH_CALUDE_arc_length_for_unit_angle_l21_2160


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l21_2197

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * |x|

-- Statement to prove
theorem f_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l21_2197


namespace NUMINAMATH_CALUDE_people_left_at_table_l21_2192

theorem people_left_at_table (initial_people : ℕ) (people_who_left : ℕ) : 
  initial_people = 11 → people_who_left = 6 → initial_people - people_who_left = 5 :=
by sorry

end NUMINAMATH_CALUDE_people_left_at_table_l21_2192


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l21_2133

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 3 * Nat.factorial 3 + Nat.factorial 3 = 35904 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l21_2133


namespace NUMINAMATH_CALUDE_min_intersection_points_2000_l21_2178

/-- Represents a collection of congruent circles on a plane -/
structure CircleCollection where
  n : ℕ
  no_tangent : Bool
  meets_two : Bool

/-- The minimum number of intersection points for a given collection of circles -/
def min_intersection_points (c : CircleCollection) : ℕ :=
  2 * (c.n - 2) + 1

/-- Theorem: For 2000 circles satisfying the given conditions, 
    the minimum number of intersection points is 3997 -/
theorem min_intersection_points_2000 :
  ∀ (c : CircleCollection), 
    c.n = 2000 ∧ c.no_tangent ∧ c.meets_two → 
    min_intersection_points c = 3997 := by
  sorry

#eval min_intersection_points ⟨2000, true, true⟩

end NUMINAMATH_CALUDE_min_intersection_points_2000_l21_2178


namespace NUMINAMATH_CALUDE_meaningful_fraction_range_l21_2117

theorem meaningful_fraction_range (x : ℝ) :
  (∃ y : ℝ, y = (1 / (x - 1)) + 1) → x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_fraction_range_l21_2117


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_problem_solution_l21_2132

theorem least_subtrahend_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  let r := n % d
  r = (n - (n - r)) % d ∧
  r ≤ d - 1 ∧
  (∀ m : ℕ, m < r → (n - m) % d ≠ 0) :=
by sorry

theorem problem_solution :
  let n : ℕ := 3674958423
  let d : ℕ := 47
  let r : ℕ := n % d
  r = 30 ∧
  (n - r) % d = 0 ∧
  (∀ m : ℕ, m < r → (n - m) % d ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_problem_solution_l21_2132


namespace NUMINAMATH_CALUDE_monkeys_count_l21_2180

theorem monkeys_count (termites : ℕ) (total_workers : ℕ) (h1 : termites = 622) (h2 : total_workers = 861) :
  total_workers - termites = 239 := by
  sorry

end NUMINAMATH_CALUDE_monkeys_count_l21_2180


namespace NUMINAMATH_CALUDE_power_multiplication_calculate_expression_l21_2138

theorem power_multiplication (a : ℕ) (m n : ℕ) :
  a * (a ^ n) = a ^ (n + 1) :=
by
  sorry

theorem calculate_expression : 
  3000 * (3000 ^ 1500) = 3000 ^ 1501 :=
by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_calculate_expression_l21_2138


namespace NUMINAMATH_CALUDE_maximum_assignment_x_plus_v_l21_2128

def Values : Finset ℕ := {2, 3, 4, 5}

structure Assignment where
  V : ℕ
  W : ℕ
  X : ℕ
  Y : ℕ
  h1 : V ∈ Values
  h2 : W ∈ Values
  h3 : X ∈ Values
  h4 : Y ∈ Values
  h5 : V ≠ W ∧ V ≠ X ∧ V ≠ Y ∧ W ≠ X ∧ W ≠ Y ∧ X ≠ Y

def ExpressionValue (a : Assignment) : ℕ := a.Y^a.X - a.W^a.V

def MaximumAssignment : Assignment → Prop := λ a => 
  ∀ b : Assignment, ExpressionValue a ≥ ExpressionValue b

theorem maximum_assignment_x_plus_v (a : Assignment) 
  (h : MaximumAssignment a) : a.X + a.V = 8 := by
  sorry

end NUMINAMATH_CALUDE_maximum_assignment_x_plus_v_l21_2128


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l21_2166

-- Define sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l21_2166


namespace NUMINAMATH_CALUDE_debt_average_payment_l21_2115

/-- Calculates the average payment for a debt paid in installments over a year. -/
theorem debt_average_payment
  (total_installments : ℕ)
  (first_payment_count : ℕ)
  (first_payment_amount : ℚ)
  (payment_increase : ℚ)
  (h1 : total_installments = 104)
  (h2 : first_payment_count = 24)
  (h3 : first_payment_amount = 520)
  (h4 : payment_increase = 95) :
  let remaining_payment_count := total_installments - first_payment_count
  let remaining_payment_amount := first_payment_amount + payment_increase
  let total_amount := first_payment_count * first_payment_amount +
                      remaining_payment_count * remaining_payment_amount
  total_amount / total_installments = 593.08 := by
sorry


end NUMINAMATH_CALUDE_debt_average_payment_l21_2115


namespace NUMINAMATH_CALUDE_color_selection_problem_l21_2113

/-- The number of ways to select k distinct items from a set of n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of colors available -/
def total_colors : ℕ := 9

/-- The number of colors to be selected -/
def colors_to_select : ℕ := 3

theorem color_selection_problem :
  choose total_colors colors_to_select = 84 := by
  sorry

end NUMINAMATH_CALUDE_color_selection_problem_l21_2113


namespace NUMINAMATH_CALUDE_museum_visitor_ratio_l21_2173

/-- Represents the number of adults and children visiting a museum. -/
structure MuseumVisitors where
  adults : ℕ
  children : ℕ

/-- Calculates the total admission fee for a given number of adults and children. -/
def admissionFee (visitors : MuseumVisitors) : ℕ :=
  30 * visitors.adults + 15 * visitors.children

/-- Checks if the number of visitors satisfies the minimum requirement. -/
def satisfiesMinimum (visitors : MuseumVisitors) : Prop :=
  visitors.adults ≥ 2 ∧ visitors.children ≥ 2

/-- Calculates the ratio of adults to children. -/
def visitorRatio (visitors : MuseumVisitors) : ℚ :=
  visitors.adults / visitors.children

theorem museum_visitor_ratio :
  ∃ (visitors : MuseumVisitors),
    satisfiesMinimum visitors ∧
    admissionFee visitors = 2700 ∧
    visitorRatio visitors = 2 ∧
    (∀ (other : MuseumVisitors),
      satisfiesMinimum other →
      admissionFee other = 2700 →
      |visitorRatio other - 2| ≥ |visitorRatio visitors - 2|) := by
  sorry

end NUMINAMATH_CALUDE_museum_visitor_ratio_l21_2173


namespace NUMINAMATH_CALUDE_number_fraction_proof_l21_2186

theorem number_fraction_proof (N : ℝ) (h : (3/10) * N - 8 = 12) : (1/5) * N = 40/3 := by
  sorry

end NUMINAMATH_CALUDE_number_fraction_proof_l21_2186


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l21_2101

def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | -3 < x ∧ x < 2}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -3 < x ∧ x < 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l21_2101


namespace NUMINAMATH_CALUDE_field_length_width_ratio_l21_2118

/-- Proves that the ratio of length to width of a rectangular field is 2:1 given specific conditions --/
theorem field_length_width_ratio :
  ∀ (field_length field_width pond_side : ℝ),
    pond_side = 8 →
    field_length = 112 →
    pond_side^2 = (1/98) * (field_length * field_width) →
    field_length / field_width = 2 := by
  sorry

end NUMINAMATH_CALUDE_field_length_width_ratio_l21_2118


namespace NUMINAMATH_CALUDE_express_y_in_terms_of_x_l21_2188

theorem express_y_in_terms_of_x (x y : ℝ) (h : 2 * x + y = 5) : y = 5 - 2 * x := by
  sorry

end NUMINAMATH_CALUDE_express_y_in_terms_of_x_l21_2188


namespace NUMINAMATH_CALUDE_blanket_warmth_l21_2142

theorem blanket_warmth (total_blankets : ℕ) (used_fraction : ℚ) (total_warmth : ℕ) : 
  total_blankets = 14 →
  used_fraction = 1/2 →
  total_warmth = 21 →
  (total_warmth : ℚ) / (used_fraction * total_blankets) = 3 := by
  sorry

end NUMINAMATH_CALUDE_blanket_warmth_l21_2142


namespace NUMINAMATH_CALUDE_point_not_in_third_quadrant_l21_2100

/-- A point P(x, y) on the line y = -x + 1 cannot be in the third quadrant -/
theorem point_not_in_third_quadrant (x y : ℝ) (h : y = -x + 1) :
  ¬(x < 0 ∧ y < 0) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_third_quadrant_l21_2100


namespace NUMINAMATH_CALUDE_dans_car_mpg_l21_2134

/-- Given the cost of gas and the distance a car can travel on a certain amount of gas,
    calculate the miles per gallon of the car. -/
theorem dans_car_mpg (gas_cost : ℝ) (miles : ℝ) (gas_expense : ℝ) (mpg : ℝ) : 
  gas_cost = 4 →
  miles = 464 →
  gas_expense = 58 →
  mpg = miles / (gas_expense / gas_cost) →
  mpg = 32 := by
sorry

end NUMINAMATH_CALUDE_dans_car_mpg_l21_2134


namespace NUMINAMATH_CALUDE_abs_inequality_l21_2140

theorem abs_inequality (a_n e : ℝ) (h : |a_n - e| < 1) : |a_n| < |e| + 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_inequality_l21_2140


namespace NUMINAMATH_CALUDE_bob_corn_harvest_l21_2121

/-- Calculates the number of whole bushels of corn harvested given the number of rows, stalks per row, and stalks per bushel. -/
def cornHarvest (rows : ℕ) (stalksPerRow : ℕ) (stalksPerBushel : ℕ) : ℕ :=
  (rows * stalksPerRow) / stalksPerBushel

theorem bob_corn_harvest :
  cornHarvest 7 92 9 = 71 := by
  sorry

end NUMINAMATH_CALUDE_bob_corn_harvest_l21_2121


namespace NUMINAMATH_CALUDE_hundredth_term_is_401_l21_2135

/-- 
Represents a sequence of toothpicks where:
- The first term is 5
- Each subsequent term increases by 4
-/
def toothpick_sequence (n : ℕ) : ℕ := 5 + 4 * (n - 1)

/-- 
Theorem: The 100th term of the toothpick sequence is 401
-/
theorem hundredth_term_is_401 : toothpick_sequence 100 = 401 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_term_is_401_l21_2135


namespace NUMINAMATH_CALUDE_parabolas_intersection_k_l21_2191

/-- Two different parabolas that intersect on the x-axis -/
def intersecting_parabolas (k : ℝ) : Prop :=
  ∃ x : ℝ, 
    (x^2 + k*x + 1 = 0) ∧ 
    (x^2 - x - k = 0) ∧
    (x^2 + k*x + 1 ≠ x^2 - x - k)

/-- The value of k for which the parabolas intersect on the x-axis -/
theorem parabolas_intersection_k : 
  ∃! k : ℝ, intersecting_parabolas k ∧ k = 2 :=
sorry

end NUMINAMATH_CALUDE_parabolas_intersection_k_l21_2191


namespace NUMINAMATH_CALUDE_rice_distribution_l21_2174

theorem rice_distribution (total_rice : ℝ) (difference : ℝ) (fraction : ℝ) : 
  total_rice = 50 →
  difference = 20 →
  fraction * total_rice = (1 - fraction) * total_rice + difference →
  fraction = 7/10 := by
sorry

end NUMINAMATH_CALUDE_rice_distribution_l21_2174


namespace NUMINAMATH_CALUDE_prob_five_three_l21_2198

/-- Represents the probability of reaching (0,0) from a given point (x,y) -/
def P (x y : ℕ) : ℚ :=
  sorry

/-- The probability of reaching (0,0) from any point on the x-axis (except origin) is 0 -/
axiom P_x_axis (x : ℕ) : x > 0 → P x 0 = 0

/-- The probability of reaching (0,0) from any point on the y-axis (except origin) is 0 -/
axiom P_y_axis (y : ℕ) : y > 0 → P 0 y = 0

/-- The probability at the origin is 1 -/
axiom P_origin : P 0 0 = 1

/-- The recursive relation for the probability function -/
axiom P_recursive (x y : ℕ) : x > 0 → y > 0 → 
  P x y = (1/3 : ℚ) * (P (x-1) y + P x (y-1) + P (x-1) (y-1))

/-- The main theorem: probability of reaching (0,0) from (5,3) is 121/729 -/
theorem prob_five_three : P 5 3 = 121 / 729 :=
  sorry

end NUMINAMATH_CALUDE_prob_five_three_l21_2198


namespace NUMINAMATH_CALUDE_sell_all_cars_in_five_months_l21_2137

/-- Calculates the number of months needed to sell all cars -/
def months_to_sell_cars (total_cars : ℕ) (num_salespeople : ℕ) (cars_per_salesperson_per_month : ℕ) : ℕ :=
  total_cars / (num_salespeople * cars_per_salesperson_per_month)

/-- Proves that it takes 5 months to sell all cars under given conditions -/
theorem sell_all_cars_in_five_months : 
  months_to_sell_cars 500 10 10 = 5 := by
  sorry

#eval months_to_sell_cars 500 10 10

end NUMINAMATH_CALUDE_sell_all_cars_in_five_months_l21_2137


namespace NUMINAMATH_CALUDE_puppy_adoption_cost_puppy_adoption_cost_proof_l21_2167

/-- The cost to get each puppy ready for adoption, given:
  * Cost for cats is $50 per cat
  * Cost for adult dogs is $100 per dog
  * 2 cats, 3 adult dogs, and 2 puppies were adopted
  * Total cost for all animals is $700
-/
theorem puppy_adoption_cost : ℝ :=
  let cat_cost : ℝ := 50
  let dog_cost : ℝ := 100
  let num_cats : ℕ := 2
  let num_dogs : ℕ := 3
  let num_puppies : ℕ := 2
  let total_cost : ℝ := 700
  150

theorem puppy_adoption_cost_proof (cat_cost dog_cost total_cost : ℝ) (num_cats num_dogs num_puppies : ℕ) 
  (h_cat_cost : cat_cost = 50)
  (h_dog_cost : dog_cost = 100)
  (h_num_cats : num_cats = 2)
  (h_num_dogs : num_dogs = 3)
  (h_num_puppies : num_puppies = 2)
  (h_total_cost : total_cost = 700)
  : puppy_adoption_cost = (total_cost - (↑num_cats * cat_cost + ↑num_dogs * dog_cost)) / ↑num_puppies :=
by sorry

end NUMINAMATH_CALUDE_puppy_adoption_cost_puppy_adoption_cost_proof_l21_2167


namespace NUMINAMATH_CALUDE_paper_side_length_l21_2139

theorem paper_side_length (cube_side: ℝ) (num_pieces: ℕ) (paper_side: ℝ)
  (h1: cube_side = 12)
  (h2: num_pieces = 54)
  (h3: (6 * cube_side^2) = (num_pieces * paper_side^2)) :
  paper_side = 4 := by
  sorry

end NUMINAMATH_CALUDE_paper_side_length_l21_2139


namespace NUMINAMATH_CALUDE_show_receipts_l21_2184

/-- Calculates the total receipts for a show given ticket prices and attendance. -/
def totalReceipts (adultPrice childPrice : ℚ) (numAdults : ℕ) : ℚ :=
  let numChildren := numAdults / 2
  adultPrice * numAdults + childPrice * numChildren

/-- Theorem stating that the total receipts for the show are 1026 dollars. -/
theorem show_receipts :
  totalReceipts (5.5) (2.5) 152 = 1026 := by
  sorry

#eval totalReceipts (5.5) (2.5) 152

end NUMINAMATH_CALUDE_show_receipts_l21_2184


namespace NUMINAMATH_CALUDE_translation_theorem_l21_2109

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translation of a point in 2D space -/
def translate (p : Point2D) (dx dy : ℝ) : Point2D :=
  ⟨p.x + dx, p.y + dy⟩

/-- Theorem: Given a translation of line segment AB to A'B', 
    if A(-2,3) corresponds to A'(3,2) and B corresponds to B'(4,0), 
    then the coordinates of B are (-1,1) -/
theorem translation_theorem 
  (A : Point2D) (A' : Point2D) (B' : Point2D)
  (h1 : A = ⟨-2, 3⟩)
  (h2 : A' = ⟨3, 2⟩)
  (h3 : B' = ⟨4, 0⟩)
  (h4 : ∃ (dx dy : ℝ), A' = translate A dx dy ∧ B' = translate ⟨-1, 1⟩ dx dy) :
  ∃ (B : Point2D), B = ⟨-1, 1⟩ ∧ B' = translate B (A'.x - A.x) (A'.y - A.y) :=
sorry

end NUMINAMATH_CALUDE_translation_theorem_l21_2109


namespace NUMINAMATH_CALUDE_hawk_breeding_theorem_l21_2149

/-- Given information about hawk breeding --/
structure HawkBreeding where
  num_kettles : ℕ
  pregnancies_per_kettle : ℕ
  survival_rate : ℚ
  expected_babies : ℕ

/-- Calculate the number of babies yielded per batch before loss --/
def babies_per_batch (h : HawkBreeding) : ℚ :=
  (h.expected_babies : ℚ) / h.survival_rate / ((h.num_kettles * h.pregnancies_per_kettle) : ℚ)

/-- Theorem stating the number of babies yielded per batch --/
theorem hawk_breeding_theorem (h : HawkBreeding) 
  (h_kettles : h.num_kettles = 6)
  (h_pregnancies : h.pregnancies_per_kettle = 15)
  (h_survival : h.survival_rate = 3/4)
  (h_expected : h.expected_babies = 270) :
  babies_per_batch h = 4 := by
  sorry


end NUMINAMATH_CALUDE_hawk_breeding_theorem_l21_2149
