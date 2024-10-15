import Mathlib

namespace NUMINAMATH_CALUDE_proof_uses_synthetic_method_l1701_170197

-- Define the proof process as a string
def proofProcess : String := "cos 4θ - sin 4θ = (cos 2θ + sin 2θ) ⋅ (cos 2θ - sin 2θ) = cos 2θ - sin 2θ = cos 2θ"

-- Define the possible proof methods
inductive ProofMethod
| Analytical
| Synthetic
| Combined
| Indirect

-- Define a function to determine the proof method
def determineProofMethod (process : String) : ProofMethod := sorry

-- Theorem stating that the proof process uses the Synthetic Method
theorem proof_uses_synthetic_method : 
  determineProofMethod proofProcess = ProofMethod.Synthetic := sorry

end NUMINAMATH_CALUDE_proof_uses_synthetic_method_l1701_170197


namespace NUMINAMATH_CALUDE_joe_needs_twelve_more_cars_l1701_170199

/-- Given that Joe has 50 toy cars initially and will have 62 cars after getting more,
    prove that he needs to get 12 more toy cars. -/
theorem joe_needs_twelve_more_cars 
  (initial_cars : ℕ) 
  (final_cars : ℕ) 
  (h1 : initial_cars = 50) 
  (h2 : final_cars = 62) : 
  final_cars - initial_cars = 12 := by
  sorry

end NUMINAMATH_CALUDE_joe_needs_twelve_more_cars_l1701_170199


namespace NUMINAMATH_CALUDE_initial_value_proof_l1701_170120

-- Define the property tax rate
def tax_rate : ℝ := 0.10

-- Define the new assessed value
def new_value : ℝ := 28000

-- Define the property tax increase
def tax_increase : ℝ := 800

-- Theorem statement
theorem initial_value_proof :
  ∃ (initial_value : ℝ),
    initial_value * tax_rate + tax_increase = new_value * tax_rate ∧
    initial_value = 20000 :=
by sorry

end NUMINAMATH_CALUDE_initial_value_proof_l1701_170120


namespace NUMINAMATH_CALUDE_shyne_eggplant_packets_l1701_170171

/-- The number of eggplants that can be grown from one seed packet -/
def eggplants_per_packet : ℕ := 14

/-- The number of sunflowers that can be grown from one seed packet -/
def sunflowers_per_packet : ℕ := 10

/-- The number of sunflower seed packets Shyne bought -/
def sunflower_packets : ℕ := 6

/-- The total number of plants Shyne can grow in her backyard -/
def total_plants : ℕ := 116

/-- The number of eggplant seed packets Shyne bought -/
def eggplant_packets : ℕ := 4

theorem shyne_eggplant_packets : 
  eggplant_packets * eggplants_per_packet + sunflower_packets * sunflowers_per_packet = total_plants :=
by sorry

end NUMINAMATH_CALUDE_shyne_eggplant_packets_l1701_170171


namespace NUMINAMATH_CALUDE_polynomial_independent_of_x_l1701_170149

-- Define the polynomial
def polynomial (x y a b : ℝ) : ℝ := 9*x^3 + y^2 + a*x - b*x^3 + x + 5

-- State the theorem
theorem polynomial_independent_of_x (y a b : ℝ) :
  (∀ x₁ x₂ : ℝ, polynomial x₁ y a b = polynomial x₂ y a b) →
  a - b = -10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_independent_of_x_l1701_170149


namespace NUMINAMATH_CALUDE_divisibility_problem_l1701_170147

theorem divisibility_problem (m n : ℕ) (hm : m > 0) (hn : n > 0) 
  (h_div : (5 * m + n) ∣ (5 * n + m)) : m ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1701_170147


namespace NUMINAMATH_CALUDE_sin_increasing_in_interval_l1701_170161

theorem sin_increasing_in_interval :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x - π / 6)
  ∀ x y, -π/6 < x ∧ x < y ∧ y < π/3 → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_sin_increasing_in_interval_l1701_170161


namespace NUMINAMATH_CALUDE_hound_catches_hare_l1701_170106

/-- The number of jumps required for a hound to catch a hare -/
def catchHare (initialSeparation : ℕ) (hareJump : ℕ) (houndJump : ℕ) : ℕ :=
  initialSeparation / (houndJump - hareJump)

/-- Theorem stating that given the specific conditions, the hound catches the hare in 75 jumps -/
theorem hound_catches_hare :
  catchHare 150 7 9 = 75 := by
  sorry

#eval catchHare 150 7 9

end NUMINAMATH_CALUDE_hound_catches_hare_l1701_170106


namespace NUMINAMATH_CALUDE_headcount_averages_l1701_170196

def spring_headcounts : List ℕ := [10900, 10500, 10700, 11300]
def fall_headcounts : List ℕ := [11700, 11500, 11600, 11300]

theorem headcount_averages :
  (spring_headcounts.sum / spring_headcounts.length : ℚ) = 10850 ∧
  (fall_headcounts.sum / fall_headcounts.length : ℚ) = 11525 := by
  sorry

end NUMINAMATH_CALUDE_headcount_averages_l1701_170196


namespace NUMINAMATH_CALUDE_circle_condition_l1701_170119

/-- A circle in the xy-plane can be represented by the equation x^2 + y^2 - x + y + m = 0,
    where m is a real number. This theorem states that for the equation to represent a circle,
    the value of m must be less than 1/4. -/
theorem circle_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ∧ 
   ∀ (a b : ℝ), (a - (1/2))^2 + (b - (1/2))^2 = ((1/2)^2 + (1/2)^2 - m)) →
  m < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l1701_170119


namespace NUMINAMATH_CALUDE_quadrilateral_classification_l1701_170115

/-
  Definitions:
  - a, b, c, d: vectors representing sides AB, BC, CD, DA of quadrilateral ABCD
  - m, n: real numbers
-/

variable (a b c d : ℝ × ℝ)
variable (m n : ℝ)

/-- A quadrilateral is a rectangle if its adjacent sides are perpendicular and opposite sides are equal -/
def is_rectangle (a b c d : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0 ∧
  b.1 * c.1 + b.2 * c.2 = 0 ∧
  c.1 * d.1 + c.2 * d.2 = 0 ∧
  d.1 * a.1 + d.2 * a.2 = 0 ∧
  a.1^2 + a.2^2 = c.1^2 + c.2^2 ∧
  b.1^2 + b.2^2 = d.1^2 + d.2^2

/-- A quadrilateral is an isosceles trapezoid if it has one pair of parallel sides and the other pair of equal length -/
def is_isosceles_trapezoid (a b c d : ℝ × ℝ) : Prop :=
  (a.1 * d.2 - a.2 * d.1 = b.1 * c.2 - b.2 * c.1) ∧
  (a.1^2 + a.2^2 = c.1^2 + c.2^2) ∧
  (a.1 * d.2 - a.2 * d.1 ≠ 0 ∨ b.1 * c.2 - b.2 * c.1 ≠ 0)

theorem quadrilateral_classification (h1 : a.1 * b.1 + a.2 * b.2 = m) 
                                     (h2 : b.1 * c.1 + b.2 * c.2 = m)
                                     (h3 : c.1 * d.1 + c.2 * d.2 = n)
                                     (h4 : d.1 * a.1 + d.2 * a.2 = n) :
  (m = n → is_rectangle a b c d) ∧
  (m ≠ n → is_isosceles_trapezoid a b c d) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_classification_l1701_170115


namespace NUMINAMATH_CALUDE_binomial_coefficient_equation_unique_solution_l1701_170132

theorem binomial_coefficient_equation_unique_solution :
  ∃! n : ℕ, (Nat.choose 25 n + Nat.choose 25 12 = Nat.choose 26 13) ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equation_unique_solution_l1701_170132


namespace NUMINAMATH_CALUDE_ball_cost_price_l1701_170121

/-- The cost price of a single ball -/
def cost_price : ℕ := sorry

/-- The selling price of 20 balls -/
def selling_price : ℕ := 720

/-- The number of balls sold -/
def balls_sold : ℕ := 20

/-- The number of balls whose cost equals the loss -/
def balls_loss : ℕ := 5

theorem ball_cost_price : 
  cost_price = 48 ∧ 
  selling_price = balls_sold * cost_price - balls_loss * cost_price :=
sorry

end NUMINAMATH_CALUDE_ball_cost_price_l1701_170121


namespace NUMINAMATH_CALUDE_conjunction_false_implies_one_false_l1701_170135

theorem conjunction_false_implies_one_false (p q : Prop) :
  (p ∧ q) = False → (p = False ∨ q = False) :=
by sorry

end NUMINAMATH_CALUDE_conjunction_false_implies_one_false_l1701_170135


namespace NUMINAMATH_CALUDE_place_balls_in_boxes_theorem_l1701_170173

/-- The number of ways to place 4 distinct balls into 4 distinct boxes such that exactly two boxes remain empty -/
def place_balls_in_boxes : ℕ :=
  let n_balls : ℕ := 4
  let n_boxes : ℕ := 4
  let n_empty_boxes : ℕ := 2
  -- The actual calculation is not implemented here
  84

/-- Theorem stating that the number of ways to place 4 distinct balls into 4 distinct boxes
    such that exactly two boxes remain empty is 84 -/
theorem place_balls_in_boxes_theorem :
  place_balls_in_boxes = 84 := by
  sorry

end NUMINAMATH_CALUDE_place_balls_in_boxes_theorem_l1701_170173


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l1701_170125

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 12 * x + c = 0) →  -- exactly one solution
  (a + c = 14) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 7 - Real.sqrt 31 ∧ c = 7 + Real.sqrt 31) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l1701_170125


namespace NUMINAMATH_CALUDE_hall_area_l1701_170192

/-- The area of a rectangular hall with specific proportions -/
theorem hall_area : 
  ∀ (length width : ℝ),
  width = (1/2) * length →
  length - width = 20 →
  length * width = 800 := by
sorry

end NUMINAMATH_CALUDE_hall_area_l1701_170192


namespace NUMINAMATH_CALUDE_smallest_self_repeating_square_end_l1701_170107

/-- A function that returns the last n digits of a natural number in base 10 -/
def lastNDigits (n : ℕ) (digits : ℕ) : ℕ :=
  n % (10 ^ digits)

/-- The theorem stating that 40625 is the smallest positive integer N such that
    N and N^2 end in the same sequence of five digits in base 10,
    with the first of these five digits being non-zero -/
theorem smallest_self_repeating_square_end : ∀ N : ℕ,
  N > 0 ∧ 
  lastNDigits N 5 = lastNDigits (N^2) 5 ∧
  N ≥ 10000 →
  N ≥ 40625 := by
  sorry

end NUMINAMATH_CALUDE_smallest_self_repeating_square_end_l1701_170107


namespace NUMINAMATH_CALUDE_exterior_angle_measure_l1701_170169

theorem exterior_angle_measure (a b : ℝ) (ha : a = 70) (hb : b = 40) :
  180 - a = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_measure_l1701_170169


namespace NUMINAMATH_CALUDE_carls_dad_contribution_l1701_170137

def weekly_savings : ℕ := 25
def weeks_saved : ℕ := 6
def coat_cost : ℕ := 170
def bill_fraction : ℚ := 1/3

theorem carls_dad_contribution :
  let total_savings := weekly_savings * weeks_saved
  let remaining_savings := total_savings - (bill_fraction * total_savings).floor
  coat_cost - remaining_savings = 70 := by
  sorry

end NUMINAMATH_CALUDE_carls_dad_contribution_l1701_170137


namespace NUMINAMATH_CALUDE_work_earnings_equality_l1701_170164

theorem work_earnings_equality (t : ℝ) : 
  (t + 2) * (4 * t - 2) = (4 * t - 7) * (t + 3) + 3 → t = 14 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_equality_l1701_170164


namespace NUMINAMATH_CALUDE_log_sum_equals_six_l1701_170129

theorem log_sum_equals_six :
  2 * (Real.log 10 / Real.log 5) + (Real.log 0.25 / Real.log 5) + 8^(2/3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_six_l1701_170129


namespace NUMINAMATH_CALUDE_tims_garden_fence_length_l1701_170124

/-- The perimeter of an irregular pentagon with given side lengths -/
def pentagon_perimeter (a b c d e : ℝ) : ℝ := a + b + c + d + e

/-- Theorem: The perimeter of Tim's garden fence -/
theorem tims_garden_fence_length :
  pentagon_perimeter 28 32 25 35 39 = 159 := by
  sorry

end NUMINAMATH_CALUDE_tims_garden_fence_length_l1701_170124


namespace NUMINAMATH_CALUDE_polynomial_equation_sum_l1701_170105

theorem polynomial_equation_sum (a b c d : ℤ) :
  (∀ x : ℤ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 + x^2 + 8*x - 12) →
  a + b + c + d = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_sum_l1701_170105


namespace NUMINAMATH_CALUDE_roots_nature_l1701_170101

/-- The quadratic equation x^2 + 2x + m = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 + 2*x + m = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  4 - 4*m

/-- The nature of the roots is determined by the value of m -/
theorem roots_nature (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation x m ∧ quadratic_equation y m) ∨
  (∃ x : ℝ, quadratic_equation x m ∧ ∀ y : ℝ, quadratic_equation y m → y = x) ∨
  (∀ x : ℝ, ¬quadratic_equation x m) :=
sorry

end NUMINAMATH_CALUDE_roots_nature_l1701_170101


namespace NUMINAMATH_CALUDE_stretches_per_meter_l1701_170152

/-- Given the following conversions between paces, stretches, leaps, and meters:
    p paces equals q stretches,
    r leaps equals s stretches,
    t leaps equals u meters,
    prove that the number of stretches in one meter is ts/ur. -/
theorem stretches_per_meter
  (p q r s t u : ℝ)
  (h1 : p * q⁻¹ = 1)  -- p paces equals q stretches
  (h2 : r * s⁻¹ = 1)  -- r leaps equals s stretches
  (h3 : t * u⁻¹ = 1)  -- t leaps equals u meters
  : 1 = t * s * (u * r)⁻¹ :=
sorry

end NUMINAMATH_CALUDE_stretches_per_meter_l1701_170152


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l1701_170177

theorem least_integer_satisfying_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), 3 * |2 * y - 1| + 6 < 24 → x ≤ y) ∧ (3 * |2 * x - 1| + 6 < 24) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_l1701_170177


namespace NUMINAMATH_CALUDE_bee_speed_difference_l1701_170151

/-- Proves the difference in bee's speed between two flight segments -/
theorem bee_speed_difference (time_daisy_rose time_rose_poppy : ℝ)
  (distance_difference : ℝ) (speed_daisy_rose : ℝ)
  (h1 : time_daisy_rose = 10)
  (h2 : time_rose_poppy = 6)
  (h3 : distance_difference = 8)
  (h4 : speed_daisy_rose = 2.6) :
  speed_daisy_rose * time_daisy_rose - distance_difference = 
  (speed_daisy_rose + 0.4) * time_rose_poppy := by
  sorry

end NUMINAMATH_CALUDE_bee_speed_difference_l1701_170151


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l1701_170126

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) :
  |x| + |y| ≤ 2 * Real.sqrt 2 ∧ ∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l1701_170126


namespace NUMINAMATH_CALUDE_skittles_eaten_l1701_170198

/-- Proves that the number of Skittles eaten is the difference between initial and final amounts --/
theorem skittles_eaten (initial_skittles final_skittles : ℝ) (oranges_bought : ℝ) :
  initial_skittles = 7 →
  final_skittles = 2 →
  oranges_bought = 18 →
  initial_skittles - final_skittles = 5 := by
  sorry

end NUMINAMATH_CALUDE_skittles_eaten_l1701_170198


namespace NUMINAMATH_CALUDE_belt_cost_calculation_l1701_170162

def initial_budget : ℕ := 200
def shirt_cost : ℕ := 30
def pants_cost : ℕ := 46
def coat_cost : ℕ := 38
def socks_cost : ℕ := 11
def shoes_cost : ℕ := 41
def amount_left : ℕ := 16

theorem belt_cost_calculation : 
  initial_budget - (shirt_cost + pants_cost + coat_cost + socks_cost + shoes_cost + amount_left) = 18 := by
  sorry

end NUMINAMATH_CALUDE_belt_cost_calculation_l1701_170162


namespace NUMINAMATH_CALUDE_electricity_billing_theorem_l1701_170136

/-- Represents the tariff rates for different zones --/
structure TariffRates where
  peak : ℝ
  night : ℝ
  half_peak : ℝ

/-- Represents the meter readings --/
structure MeterReadings where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Calculates the maximum possible additional payment --/
def max_additional_payment (rates : TariffRates) (readings : MeterReadings) (paid_amount : ℝ) : ℝ :=
  sorry

/-- Calculates the expected difference between company's calculation and customer's payment --/
def expected_difference (rates : TariffRates) (readings : MeterReadings) (paid_amount : ℝ) : ℝ :=
  sorry

/-- Main theorem stating the correct results for the given problem --/
theorem electricity_billing_theorem (rates : TariffRates) (readings : MeterReadings) (paid_amount : ℝ) :
  rates.peak = 4.03 ∧ rates.night = 1.01 ∧ rates.half_peak = 3.39 ∧
  readings.a = 1214 ∧ readings.b = 1270 ∧ readings.c = 1298 ∧
  readings.d = 1337 ∧ readings.e = 1347 ∧ readings.f = 1402 ∧
  paid_amount = 660.72 →
  max_additional_payment rates readings paid_amount = 397.34 ∧
  expected_difference rates readings paid_amount = 19.30 :=
by sorry

end NUMINAMATH_CALUDE_electricity_billing_theorem_l1701_170136


namespace NUMINAMATH_CALUDE_correct_dot_counts_l1701_170167

/-- Represents a single die face -/
inductive DieFace
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the four visible faces of the dice configuration -/
structure VisibleFaces :=
  (A : DieFace)
  (B : DieFace)
  (C : DieFace)
  (D : DieFace)

/-- Counts the number of dots on a die face -/
def dotCount (face : DieFace) : Nat :=
  match face with
  | DieFace.one => 1
  | DieFace.two => 2
  | DieFace.three => 3
  | DieFace.four => 4
  | DieFace.five => 5
  | DieFace.six => 6

/-- The configuration of dice as described in the problem -/
def diceConfiguration : VisibleFaces :=
  { A := DieFace.three
  , B := DieFace.five
  , C := DieFace.six
  , D := DieFace.five }

/-- Theorem stating the correct number of dots on each visible face -/
theorem correct_dot_counts :
  dotCount diceConfiguration.A = 3 ∧
  dotCount diceConfiguration.B = 5 ∧
  dotCount diceConfiguration.C = 6 ∧
  dotCount diceConfiguration.D = 5 :=
sorry

end NUMINAMATH_CALUDE_correct_dot_counts_l1701_170167


namespace NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l1701_170109

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  /-- The base angle of the isosceles triangle -/
  baseAngle : ℝ
  /-- The altitude to the base of the isosceles triangle -/
  altitude : ℝ
  /-- The base of the isosceles triangle -/
  base : ℝ

/-- Theorem stating that an isosceles triangle is not uniquely determined by one angle and the altitude to one of its sides -/
theorem isosceles_triangle_not_unique (α : ℝ) (h : ℝ) : 
  ∃ t1 t2 : IsoscelesTriangle, t1.baseAngle = α ∧ t1.altitude = h ∧ 
  t2.baseAngle = α ∧ t2.altitude = h ∧ t1 ≠ t2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_not_unique_l1701_170109


namespace NUMINAMATH_CALUDE_min_value_of_f_l1701_170150

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = -Real.exp 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1701_170150


namespace NUMINAMATH_CALUDE_ahn_max_number_l1701_170113

theorem ahn_max_number : ∃ (max : ℕ), max = 650 ∧ 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 3 * (300 - n) + 50 ≤ max :=
by sorry

end NUMINAMATH_CALUDE_ahn_max_number_l1701_170113


namespace NUMINAMATH_CALUDE_polynomial_division_l1701_170117

theorem polynomial_division (x : ℤ) : 
  ∃ (p : ℤ → ℤ), x^13 + 2*x + 180 = (x^2 - x + 3) * p x := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_l1701_170117


namespace NUMINAMATH_CALUDE_total_protest_days_l1701_170122

theorem total_protest_days (first_protest : ℕ) (second_protest_percentage : ℚ) : 
  first_protest = 4 →
  second_protest_percentage = 25 / 100 →
  first_protest + (first_protest + first_protest * second_protest_percentage) = 9 := by
sorry

end NUMINAMATH_CALUDE_total_protest_days_l1701_170122


namespace NUMINAMATH_CALUDE_pencils_taken_l1701_170140

theorem pencils_taken (initial_pencils remaining_pencils : ℕ) 
  (h1 : initial_pencils = 79)
  (h2 : remaining_pencils = 75) :
  initial_pencils - remaining_pencils = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencils_taken_l1701_170140


namespace NUMINAMATH_CALUDE_bake_sale_donation_percentage_l1701_170165

/-- Proves that the percentage of bake sale proceeds donated to the shelter is 75% --/
theorem bake_sale_donation_percentage :
  ∀ (carwash_earnings bake_sale_earnings lawn_mowing_earnings total_donation : ℚ),
  carwash_earnings = 100 →
  bake_sale_earnings = 80 →
  lawn_mowing_earnings = 50 →
  total_donation = 200 →
  0.9 * carwash_earnings + 1 * lawn_mowing_earnings + 
    (total_donation - (0.9 * carwash_earnings + 1 * lawn_mowing_earnings)) = total_donation →
  (total_donation - (0.9 * carwash_earnings + 1 * lawn_mowing_earnings)) / bake_sale_earnings = 0.75 :=
by sorry

end NUMINAMATH_CALUDE_bake_sale_donation_percentage_l1701_170165


namespace NUMINAMATH_CALUDE_isabellas_hourly_rate_l1701_170111

/-- Calculates Isabella's hourly rate given her work schedule and total earnings -/
theorem isabellas_hourly_rate 
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (total_weeks : ℕ)
  (total_earnings : ℕ)
  (h1 : hours_per_day = 5)
  (h2 : days_per_week = 6)
  (h3 : total_weeks = 7)
  (h4 : total_earnings = 1050) :
  total_earnings / (hours_per_day * days_per_week * total_weeks) = 5 := by
sorry

end NUMINAMATH_CALUDE_isabellas_hourly_rate_l1701_170111


namespace NUMINAMATH_CALUDE_calculation_part1_calculation_part2_l1701_170127

-- Part 1
theorem calculation_part1 : 
  (1/8)^(-(2/3)) - 4*(-3)^4 + (2 + 1/4)^(1/2) - (1.5)^2 = -320.75 := by sorry

-- Part 2
theorem calculation_part2 : 
  (Real.log 5 / Real.log 10)^2 + (Real.log 2 / Real.log 10) * (Real.log 50 / Real.log 10) - 
  (Real.log 8 / Real.log (1/2)) + (Real.log (427/3) / Real.log 3) = 
  1 - (Real.log 5 / Real.log 10)^2 + (Real.log 5 / Real.log 10) + (Real.log 2 / Real.log 10) + 2 := by sorry

end NUMINAMATH_CALUDE_calculation_part1_calculation_part2_l1701_170127


namespace NUMINAMATH_CALUDE_y_divisibility_l1701_170130

def y : ℕ := 64 + 96 + 192 + 256 + 352 + 480 + 4096 + 8192

theorem y_divisibility : 
  (∃ k : ℕ, y = 32 * k) ∧ ¬(∃ m : ℕ, y = 64 * m) :=
by sorry

end NUMINAMATH_CALUDE_y_divisibility_l1701_170130


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1701_170143

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 17) → (∃ m : ℤ, N = 13 * m + 4) :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1701_170143


namespace NUMINAMATH_CALUDE_janes_age_l1701_170114

/-- Jane's babysitting problem -/
theorem janes_age :
  ∀ (jane_start_age : ℕ) 
    (years_since_stopped : ℕ) 
    (oldest_babysat_age : ℕ),
  jane_start_age = 16 →
  years_since_stopped = 10 →
  oldest_babysat_age = 24 →
  ∃ (jane_current_age : ℕ),
    jane_current_age = 38 ∧
    (∀ (child_age : ℕ),
      child_age ≤ oldest_babysat_age →
      child_age ≤ (jane_current_age - years_since_stopped) / 2) :=
by sorry

end NUMINAMATH_CALUDE_janes_age_l1701_170114


namespace NUMINAMATH_CALUDE_equation_solutions_l1701_170159

theorem equation_solutions : 
  -- Equation 1
  (∃ x : ℝ, 4 * (x - 1)^2 - 8 = 0) ∧
  (∀ x : ℝ, 4 * (x - 1)^2 - 8 = 0 → (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2)) ∧
  -- Equation 2
  (∃ x : ℝ, 2 * x * (x - 3) = x - 3) ∧
  (∀ x : ℝ, 2 * x * (x - 3) = x - 3 → (x = 3 ∨ x = 1/2)) ∧
  -- Equation 3
  (∃ x : ℝ, x^2 - 10*x + 16 = 0) ∧
  (∀ x : ℝ, x^2 - 10*x + 16 = 0 → (x = 8 ∨ x = 2)) ∧
  -- Equation 4
  (∃ x : ℝ, 2*x^2 + 3*x - 1 = 0) ∧
  (∀ x : ℝ, 2*x^2 + 3*x - 1 = 0 → (x = (Real.sqrt 17 - 3) / 4 ∨ x = -(Real.sqrt 17 + 3) / 4)) := by
  sorry


end NUMINAMATH_CALUDE_equation_solutions_l1701_170159


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1701_170189

/-- Given a quadratic inequality ax^2 + (ab+1)x + b > 0 with solution set {x | 1 < x < 3},
    prove that a + b = -4 or a + b = -4/3 -/
theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + (a*b + 1)*x + b > 0 ↔ 1 < x ∧ x < 3) →
  (a + b = -4 ∨ a + b = -4/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1701_170189


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_l1701_170186

theorem log_equality_implies_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (Real.log a / Real.log 9) = (Real.log b / Real.log 12) ∧ 
       (Real.log a / Real.log 9) = (Real.log (2 * (a + b)) / Real.log 16)) : 
  b / a = Real.sqrt 3 + 1 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_l1701_170186


namespace NUMINAMATH_CALUDE_winnie_lollipops_l1701_170131

/-- The number of lollipops Winnie keeps for herself -/
def lollipops_kept (total : ℕ) (friends : ℕ) : ℕ :=
  total % friends

theorem winnie_lollipops :
  lollipops_kept 432 14 = 12 := by sorry

end NUMINAMATH_CALUDE_winnie_lollipops_l1701_170131


namespace NUMINAMATH_CALUDE_train_length_l1701_170123

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 5 → ∃ (length : ℝ), abs (length - 83.35) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1701_170123


namespace NUMINAMATH_CALUDE_zero_in_A_l1701_170195

def A : Set ℝ := {x : ℝ | x * (x - 2) = 0}

theorem zero_in_A : (0 : ℝ) ∈ A := by sorry

end NUMINAMATH_CALUDE_zero_in_A_l1701_170195


namespace NUMINAMATH_CALUDE_fraction_power_equality_l1701_170183

theorem fraction_power_equality : (72000 ^ 5) / (18000 ^ 5) = 1024 := by sorry

end NUMINAMATH_CALUDE_fraction_power_equality_l1701_170183


namespace NUMINAMATH_CALUDE_factorial_minus_one_mod_930_l1701_170174

theorem factorial_minus_one_mod_930 : (Nat.factorial 30 - 1) % 930 = 29 := by
  sorry

end NUMINAMATH_CALUDE_factorial_minus_one_mod_930_l1701_170174


namespace NUMINAMATH_CALUDE_max_garden_area_l1701_170178

/-- Represents the dimensions of a rectangular garden. -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden given its dimensions. -/
def gardenArea (d : GardenDimensions) : ℝ :=
  d.length * d.width

/-- Calculates the perimeter of a rectangular garden given its dimensions. -/
def gardenPerimeter (d : GardenDimensions) : ℝ :=
  2 * (d.length + d.width)

/-- Theorem stating the maximum area of a garden with given constraints. -/
theorem max_garden_area :
  ∀ d : GardenDimensions,
    d.length ≥ 100 →
    d.width ≥ 60 →
    gardenPerimeter d = 360 →
    gardenArea d ≤ 8000 :=
by sorry

end NUMINAMATH_CALUDE_max_garden_area_l1701_170178


namespace NUMINAMATH_CALUDE_sum_range_l1701_170175

theorem sum_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y + 4*y^2 = 1) :
  1/2 < x + y ∧ x + y < 1 := by
sorry

end NUMINAMATH_CALUDE_sum_range_l1701_170175


namespace NUMINAMATH_CALUDE_prime_between_squares_l1701_170188

/-- A number is a perfect square if it's the square of some integer. -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- A number is prime if it's greater than 1 and its only divisors are 1 and itself. -/
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

theorem prime_between_squares : 
  ∃! p : ℕ, is_prime p ∧ 
    (∃ n : ℕ, is_perfect_square n ∧ p = n + 12) ∧
    (∃ m : ℕ, is_perfect_square m ∧ p + 9 = m) :=
sorry

end NUMINAMATH_CALUDE_prime_between_squares_l1701_170188


namespace NUMINAMATH_CALUDE_flu_virus_diameter_scientific_notation_l1701_170153

theorem flu_virus_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000823 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8.23 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_flu_virus_diameter_scientific_notation_l1701_170153


namespace NUMINAMATH_CALUDE_percentage_error_calculation_l1701_170194

theorem percentage_error_calculation : 
  let incorrect_factor : ℚ := 3 / 5
  let correct_factor : ℚ := 5 / 3
  let ratio := incorrect_factor / correct_factor
  let error_percentage := (1 - ratio) * 100
  error_percentage = 64 := by
sorry

end NUMINAMATH_CALUDE_percentage_error_calculation_l1701_170194


namespace NUMINAMATH_CALUDE_f_inequality_l1701_170179

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.log (x + 1) - 1 / (1 + x^2)
  else Real.log (-x + 1) - 1 / (1 + x^2)

theorem f_inequality (a : ℝ) :
  f (a - 2) < f (4 - a^2) ↔ a > 2 ∨ a < -3 ∨ (-1 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_f_inequality_l1701_170179


namespace NUMINAMATH_CALUDE_ball_weights_l1701_170156

/-- The weight of a red ball in grams -/
def red_weight : ℝ := sorry

/-- The weight of a yellow ball in grams -/
def yellow_weight : ℝ := sorry

/-- The total weight of 5 red balls and 3 yellow balls in grams -/
def total_weight_1 : ℝ := 5 * red_weight + 3 * yellow_weight

/-- The total weight of 5 yellow balls and 3 red balls in grams -/
def total_weight_2 : ℝ := 5 * yellow_weight + 3 * red_weight

theorem ball_weights :
  total_weight_1 = 42 ∧ total_weight_2 = 38 → red_weight = 6 ∧ yellow_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_ball_weights_l1701_170156


namespace NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l1701_170148

theorem arithmetic_progression_of_primes (p d : ℕ) : 
  p ≠ 3 →
  Prime (p - d) →
  Prime p →
  Prime (p + d) →
  ∃ k : ℕ, d = 6 * k :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_of_primes_l1701_170148


namespace NUMINAMATH_CALUDE_jake_weight_loss_l1701_170134

theorem jake_weight_loss (total_weight jake_weight : ℝ) 
  (h1 : total_weight = 290)
  (h2 : jake_weight = 196) : 
  jake_weight - 2 * (total_weight - jake_weight) = 8 :=
sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l1701_170134


namespace NUMINAMATH_CALUDE_geometric_sequence_range_l1701_170108

theorem geometric_sequence_range (a₁ a₂ a₃ a₄ : ℝ) :
  (∃ q : ℝ, a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q) →
  (0 < a₁ ∧ a₁ < 1) →
  (1 < a₂ ∧ a₂ < 2) →
  (2 < a₃ ∧ a₃ < 4) →
  (2 * Real.sqrt 2 < a₄ ∧ a₄ < 16) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_range_l1701_170108


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1701_170139

/-- Two real numbers are inversely proportional -/
def InverselyProportional (a b : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a * b = k

theorem inverse_proportion_problem (a₁ a₂ b₁ b₂ : ℝ) 
  (h_inverse : InverselyProportional a₁ b₁) 
  (h_initial : a₁ = 40 ∧ b₁ = 8) 
  (h_final : b₂ = 10) : 
  a₂ = 32 ∧ InverselyProportional a₂ b₂ :=
sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1701_170139


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1701_170100

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 1 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1701_170100


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l1701_170176

theorem least_k_for_inequality (k : ℤ) : 
  (∀ m : ℤ, m < k → (0.000010101 * (10 : ℝ)^m ≤ 10000)) →
  (0.000010101 * (10 : ℝ)^k > 10000) →
  k = 9 := by
sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l1701_170176


namespace NUMINAMATH_CALUDE_failing_marks_difference_l1701_170185

/-- The number of marks needed to pass the exam -/
def passing_marks : ℝ := 199.99999999999997

/-- The percentage of marks obtained by the failing candidate -/
def failing_percentage : ℝ := 0.30

/-- The percentage of marks obtained by the passing candidate -/
def passing_percentage : ℝ := 0.45

/-- The number of marks the passing candidate gets above the passing mark -/
def marks_above_passing : ℝ := 25

/-- Theorem stating the number of marks by which the failing candidate fails -/
theorem failing_marks_difference : 
  let total_marks := (passing_marks + marks_above_passing) / passing_percentage
  passing_marks - (failing_percentage * total_marks) = 50 := by
sorry

end NUMINAMATH_CALUDE_failing_marks_difference_l1701_170185


namespace NUMINAMATH_CALUDE_perpendicular_vectors_implies_m_equals_three_l1701_170158

/-- Given vectors a and b in ℝ², if a is perpendicular to (a - b), then the second component of b is -1 and m = 3. -/
theorem perpendicular_vectors_implies_m_equals_three (a b : ℝ × ℝ) (m : ℝ) 
    (h1 : a = (2, 1))
    (h2 : b = (m, -1))
    (h3 : a • (a - b) = 0) :
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_implies_m_equals_three_l1701_170158


namespace NUMINAMATH_CALUDE_sasha_train_journey_l1701_170184

/-- Represents a day of the week -/
inductive DayOfWeek
  | Saturday
  | Sunday
  | Monday

/-- Represents the train journey -/
structure TrainJourney where
  departureDay : DayOfWeek
  arrivalDay : DayOfWeek
  journeyDuration : Nat
  departureDateNumber : Nat
  arrivalDateNumber : Nat
  trainCarNumber : Nat
  seatNumber : Nat

/-- The conditions of Sasha's train journey -/
def sashasJourney : TrainJourney :=
  { departureDay := DayOfWeek.Saturday
  , arrivalDay := DayOfWeek.Monday
  , journeyDuration := 50
  , departureDateNumber := 31  -- Assuming end of month
  , arrivalDateNumber := 2     -- Assuming start of next month
  , trainCarNumber := 2
  , seatNumber := 1
  }

theorem sasha_train_journey :
  ∀ (journey : TrainJourney),
    journey.departureDay = DayOfWeek.Saturday →
    journey.arrivalDay = DayOfWeek.Monday →
    journey.journeyDuration = 50 →
    journey.arrivalDateNumber = journey.trainCarNumber →
    journey.seatNumber < journey.trainCarNumber →
    journey.departureDateNumber > journey.trainCarNumber →
    journey.trainCarNumber = 2 ∧ journey.seatNumber = 1 := by
  sorry

#check sasha_train_journey

end NUMINAMATH_CALUDE_sasha_train_journey_l1701_170184


namespace NUMINAMATH_CALUDE_odd_prime_and_odd_natural_not_divide_l1701_170182

theorem odd_prime_and_odd_natural_not_divide (p n : ℕ) : 
  Nat.Prime p → Odd p → Odd n → ¬(p * n + 1 ∣ p^p - 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_prime_and_odd_natural_not_divide_l1701_170182


namespace NUMINAMATH_CALUDE_equation_solution_l1701_170154

theorem equation_solution : 
  ∀ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ↔ (x = 4 + Real.sqrt 2 ∨ x = 4 - Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1701_170154


namespace NUMINAMATH_CALUDE_sum_of_possible_N_values_l1701_170190

-- Define the set of expressions
def S (x y : ℝ) : Set ℝ := {(x + y)^2, (x - y)^2, x * y, x / y}

-- Define the given set of values
def T (N : ℝ) : Set ℝ := {4, 12.8, 28.8, N}

-- Theorem statement
theorem sum_of_possible_N_values (x y N : ℝ) (hy : y ≠ 0) 
  (h_equal : S x y = T N) : 
  ∃ (N₁ N₂ N₃ : ℝ), 
    (S x y = T N₁) ∧ 
    (S x y = T N₂) ∧ 
    (S x y = T N₃) ∧ 
    N₁ + N₂ + N₃ = 85.2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_possible_N_values_l1701_170190


namespace NUMINAMATH_CALUDE_coin_trick_theorem_l1701_170191

/-- Represents the state of a coin (Heads or Tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a sequence of coins -/
def CoinSequence (n : ℕ) := Fin n → CoinState

/-- Represents the strategy for the assistant and magician -/
structure Strategy (n : ℕ) where
  encode : CoinSequence n → Fin n → Fin n
  decode : CoinSequence n → Fin n

/-- Defines when a strategy is valid -/
def is_valid_strategy (n : ℕ) (s : Strategy n) : Prop :=
  ∀ (seq : CoinSequence n) (chosen : Fin n),
    ∃ (flipped : Fin n),
      s.decode (Function.update seq flipped (CoinState.Tails)) = chosen

/-- The main theorem: the trick is possible iff n is a power of 2 -/
theorem coin_trick_theorem (n : ℕ) :
  (∃ (s : Strategy n), is_valid_strategy n s) ↔ ∃ (k : ℕ), n = 2^k :=
sorry

end NUMINAMATH_CALUDE_coin_trick_theorem_l1701_170191


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1701_170104

/-- Given a geometric sequence {a_n}, prove that if a_1 + a_3 = 20 and a_2 + a_4 = 40, then a_3 + a_5 = 80 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_sum1 : a 1 + a 3 = 20) (h_sum2 : a 2 + a 4 = 40) : 
  a 3 + a 5 = 80 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1701_170104


namespace NUMINAMATH_CALUDE_line_intersects_segment_m_range_l1701_170145

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by the equation x + my + m = 0 -/
structure Line where
  m : ℝ

def intersectsSegment (l : Line) (a b : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    (1 - t) * a.x + t * b.x + l.m * ((1 - t) * a.y + t * b.y) + l.m = 0

theorem line_intersects_segment_m_range (l : Line) :
  let a : Point := ⟨-1, 1⟩
  let b : Point := ⟨2, -2⟩
  intersectsSegment l a b → 1/2 ≤ l.m ∧ l.m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_line_intersects_segment_m_range_l1701_170145


namespace NUMINAMATH_CALUDE_count_seating_arrangements_l1701_170133

/-- Represents a seating arrangement in a 5x5 classroom -/
def SeatingArrangement := Fin 5 → Fin 5 → Bool

/-- A seating arrangement is valid if for each occupied desk, either its row or column is full -/
def is_valid (arrangement : SeatingArrangement) : Prop :=
  ∀ i j, arrangement i j → 
    (∀ k, arrangement i k) ∨ (∀ k, arrangement k j)

/-- The total number of valid seating arrangements -/
def total_arrangements : ℕ := sorry

theorem count_seating_arrangements :
  total_arrangements = 962 := by sorry

end NUMINAMATH_CALUDE_count_seating_arrangements_l1701_170133


namespace NUMINAMATH_CALUDE_unique_coprime_solution_l1701_170168

theorem unique_coprime_solution (n : ℕ+) :
  ∀ p q : ℤ,
  p > 0 ∧ q > 0 ∧
  Int.gcd p q = 1 ∧
  p + q^2 = (n.val^2 + 1) * p^2 + q →
  p = n.val + 1 ∧ q = n.val^2 + n.val + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_coprime_solution_l1701_170168


namespace NUMINAMATH_CALUDE_no_zero_points_when_k_is_one_exactly_one_zero_point_when_k_is_negative_exists_k_with_two_zero_points_l1701_170187

-- Define the piecewise function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp x - k * x else k * x^2 - x + 1

-- Statement 1: When k = 1, f(x) has no zero points
theorem no_zero_points_when_k_is_one :
  ∀ x : ℝ, f 1 x ≠ 0 := by sorry

-- Statement 2: When k < 0, f(x) has exactly one zero point
theorem exactly_one_zero_point_when_k_is_negative :
  ∀ k : ℝ, k < 0 → ∃! x : ℝ, f k x = 0 := by sorry

-- Statement 3: There exists a k such that f(x) has two zero points
theorem exists_k_with_two_zero_points :
  ∃ k : ℝ, ∃ x y : ℝ, x ≠ y ∧ f k x = 0 ∧ f k y = 0 := by sorry

end NUMINAMATH_CALUDE_no_zero_points_when_k_is_one_exactly_one_zero_point_when_k_is_negative_exists_k_with_two_zero_points_l1701_170187


namespace NUMINAMATH_CALUDE_number_of_bs_l1701_170110

/-- Represents the number of students earning each grade in a philosophy class. -/
structure GradeDistribution where
  total : ℕ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the grade distribution satisfies the given conditions. -/
def isValidDistribution (g : GradeDistribution) : Prop :=
  g.total = 40 ∧
  g.a = 0.5 * g.b ∧
  g.c = 2 * g.b ∧
  g.a + g.b + g.c = g.total

/-- Theorem stating the number of B's in the class. -/
theorem number_of_bs (g : GradeDistribution) 
  (h : isValidDistribution g) : g.b = 40 / 3.5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_bs_l1701_170110


namespace NUMINAMATH_CALUDE_hotel_floors_l1701_170170

theorem hotel_floors (available_rooms : ℕ) (rooms_per_floor : ℕ) (unavailable_floors : ℕ) : 
  available_rooms = 90 → rooms_per_floor = 10 → unavailable_floors = 1 →
  (available_rooms / rooms_per_floor + unavailable_floors = 10) := by
sorry

end NUMINAMATH_CALUDE_hotel_floors_l1701_170170


namespace NUMINAMATH_CALUDE_emily_songs_l1701_170112

theorem emily_songs (x : ℕ) : x + 7 = 13 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_emily_songs_l1701_170112


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1701_170138

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Theorem statement
theorem circle_center_and_radius :
  ∃ (center_x center_y radius : ℝ),
    (center_x = 2 ∧ center_y = 0 ∧ radius = 2) ∧
    (∀ (x y : ℝ), circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1701_170138


namespace NUMINAMATH_CALUDE_mikes_remaining_cards_l1701_170103

/-- Given Mike's initial number of baseball cards and the number of cards Sam bought,
    prove that Mike's remaining number of cards is the difference between his initial number
    and the number Sam bought. -/
theorem mikes_remaining_cards (initial_cards sam_bought : ℕ) :
  initial_cards - sam_bought = initial_cards - sam_bought :=
by sorry

/-- Mike's initial number of baseball cards -/
def mike_initial_cards : ℕ := 87

/-- Number of cards Sam bought from Mike -/
def sam_bought_cards : ℕ := 13

/-- Mike's remaining number of cards -/
def mike_remaining_cards : ℕ := mike_initial_cards - sam_bought_cards

#eval mike_remaining_cards  -- Should output 74

end NUMINAMATH_CALUDE_mikes_remaining_cards_l1701_170103


namespace NUMINAMATH_CALUDE_vector_equality_l1701_170128

def a : ℝ × ℝ := (4, 2)
def b (k : ℝ) : ℝ × ℝ := (2 - k, k - 1)

theorem vector_equality (k : ℝ) :
  ‖a + b k‖ = ‖a - b k‖ → k = 3 := by sorry

end NUMINAMATH_CALUDE_vector_equality_l1701_170128


namespace NUMINAMATH_CALUDE_mary_saw_90_snakes_l1701_170166

/-- The number of breeding balls -/
def num_breeding_balls : ℕ := 5

/-- The number of snakes in each breeding ball -/
def snakes_per_ball : ℕ := 12

/-- The number of additional pairs of snakes -/
def num_additional_pairs : ℕ := 15

/-- The total number of snakes Mary saw -/
def total_snakes : ℕ := num_breeding_balls * snakes_per_ball + 2 * num_additional_pairs

theorem mary_saw_90_snakes : total_snakes = 90 := by
  sorry

end NUMINAMATH_CALUDE_mary_saw_90_snakes_l1701_170166


namespace NUMINAMATH_CALUDE_B_power_150_is_identity_l1701_170193

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150_is_identity :
  B^150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  sorry

end NUMINAMATH_CALUDE_B_power_150_is_identity_l1701_170193


namespace NUMINAMATH_CALUDE_trapezoid_height_theorem_l1701_170180

/-- Represents a trapezoid with given diagonal lengths and midsegment length -/
structure Trapezoid where
  diag1 : ℝ
  diag2 : ℝ
  midsegment : ℝ

/-- Calculates the height of a trapezoid given its properties -/
def trapezoidHeight (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating that a trapezoid with diagonals 6 and 8 and midsegment 5 has height 4.8 -/
theorem trapezoid_height_theorem (t : Trapezoid) 
  (h1 : t.diag1 = 6) 
  (h2 : t.diag2 = 8) 
  (h3 : t.midsegment = 5) : 
  trapezoidHeight t = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_height_theorem_l1701_170180


namespace NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l1701_170172

/-- The number of sides in the regular polygon -/
def n : ℕ := 17

/-- The number of lines of symmetry in a regular n-gon -/
def L (n : ℕ) : ℕ := n

/-- The smallest positive angle of rotational symmetry in degrees for a regular n-gon -/
def R (n : ℕ) : ℚ := 360 / n

/-- Theorem: For a regular 17-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees)
    is equal to 17 + (360 / 17) -/
theorem regular_17gon_symmetry_sum :
  (L n : ℚ) + R n = 17 + 360 / 17 := by sorry

end NUMINAMATH_CALUDE_regular_17gon_symmetry_sum_l1701_170172


namespace NUMINAMATH_CALUDE_shopping_spree_cost_equalization_l1701_170142

/-- Given the spending amounts and agreement to equally share costs, 
    prove that the difference between what Charlie gives to Bob and 
    what Alice gives to Bob is 30. -/
theorem shopping_spree_cost_equalization 
  (charlie_spent : ℝ) 
  (alice_spent : ℝ) 
  (bob_spent : ℝ) 
  (h1 : charlie_spent = 150)
  (h2 : alice_spent = 180)
  (h3 : bob_spent = 210)
  (c : ℝ)  -- amount Charlie gives to Bob
  (a : ℝ)  -- amount Alice gives to Bob
  (h4 : c = (charlie_spent + alice_spent + bob_spent) / 3 - charlie_spent)
  (h5 : a = (charlie_spent + alice_spent + bob_spent) / 3 - alice_spent) :
  c - a = 30 := by
sorry


end NUMINAMATH_CALUDE_shopping_spree_cost_equalization_l1701_170142


namespace NUMINAMATH_CALUDE_kitchen_renovation_rate_l1701_170163

/-- The hourly rate for professionals renovating Kamil's kitchen -/
def hourly_rate (professionals : ℕ) (hours_per_day : ℕ) (days : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (professionals * hours_per_day * days)

/-- Theorem stating the hourly rate for the kitchen renovation professionals -/
theorem kitchen_renovation_rate : 
  hourly_rate 2 6 7 1260 = 15 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_renovation_rate_l1701_170163


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1701_170116

theorem quadratic_roots_problem (a b p q : ℝ) : 
  p ≠ q ∧ p ≠ 0 ∧ q ≠ 0 ∧
  a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧
  p^2 - a*p + b = 0 ∧
  q^2 - a*q + b = 0 ∧
  a^2 - p*a - q = 0 ∧
  b^2 - p*b - q = 0 →
  a = 1 ∧ b = -2 ∧ p = -1 ∧ q = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1701_170116


namespace NUMINAMATH_CALUDE_special_sum_of_squares_l1701_170146

theorem special_sum_of_squares (n : ℕ) (a b : ℕ) : 
  n ≥ 2 →
  n = a^2 + b^2 →
  (∀ d : ℕ, d > 1 ∧ d ∣ n → a ≤ d) →
  a ∣ n →
  b ∣ n →
  n = 8 ∨ n = 20 :=
by sorry

end NUMINAMATH_CALUDE_special_sum_of_squares_l1701_170146


namespace NUMINAMATH_CALUDE_train_speed_problem_l1701_170155

theorem train_speed_problem (length_train1 length_train2 distance_between speed_train2 time_to_cross : ℝ)
  (h1 : length_train1 = 100)
  (h2 : length_train2 = 150)
  (h3 : distance_between = 50)
  (h4 : speed_train2 = 15)
  (h5 : time_to_cross = 60)
  : ∃ speed_train1 : ℝ,
    speed_train1 = 10 ∧
    (length_train1 + length_train2 + distance_between) / time_to_cross = speed_train2 - speed_train1 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1701_170155


namespace NUMINAMATH_CALUDE_student_multiplication_problem_l1701_170181

theorem student_multiplication_problem (chosen_number : ℕ) (final_result : ℕ) (subtracted_amount : ℕ) :
  chosen_number = 125 →
  final_result = 112 →
  subtracted_amount = 138 →
  ∃ x : ℚ, chosen_number * x - subtracted_amount = final_result ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_student_multiplication_problem_l1701_170181


namespace NUMINAMATH_CALUDE_f_negative_2014_l1701_170118

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_negative_2014 (h1 : ∀ x, f x = -f (-x))  -- f is odd
                        (h2 : ∀ x, f (x + 3) = f x)  -- f has period 3
                        (h3 : ∀ x ∈ Set.Icc 0 1, f x = x^2 - x + 2)  -- f on [0,1]
                        : f (-2014) = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_2014_l1701_170118


namespace NUMINAMATH_CALUDE_odd_periodic_monotone_increasing_l1701_170144

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y < b → f x < f y

theorem odd_periodic_monotone_increasing (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : is_periodic f 4)
  (h_monotone : monotone_increasing_on f 0 2) :
  f 3 < 0 ∧ 0 < f 1 := by sorry

end NUMINAMATH_CALUDE_odd_periodic_monotone_increasing_l1701_170144


namespace NUMINAMATH_CALUDE_grocer_purchase_price_l1701_170160

/-- Represents the price at which the grocer purchased 3 pounds of bananas -/
def purchase_price : ℝ := sorry

/-- Represents the total quantity of bananas purchased in pounds -/
def total_quantity : ℝ := 72

/-- Represents the profit made by the grocer -/
def profit : ℝ := 6

/-- Represents the selling price of 4 pounds of bananas -/
def selling_price : ℝ := 1

/-- Theorem stating that the purchase price for 3 pounds of bananas is $0.50 -/
theorem grocer_purchase_price : purchase_price = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_grocer_purchase_price_l1701_170160


namespace NUMINAMATH_CALUDE_rectangle_from_triangles_l1701_170141

/-- Represents a right-angled triangle tile with integer side lengths -/
structure Triangle :=
  (a b c : ℕ)

/-- Represents a rectangle with integer side lengths -/
structure Rectangle :=
  (width height : ℕ)

/-- Checks if a triangle is valid (right-angled and positive sides) -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧ t.a^2 + t.b^2 = t.c^2

/-- Checks if a rectangle can be formed from a given number of triangles -/
def canFormRectangle (r : Rectangle) (t : Triangle) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ 2 * n * t.a * t.b = r.width * r.height

theorem rectangle_from_triangles 
  (jackTile : Triangle)
  (targetRect : Rectangle)
  (h1 : isValidTriangle jackTile)
  (h2 : jackTile.a = 3 ∧ jackTile.b = 4 ∧ jackTile.c = 5)
  (h3 : targetRect.width = 2016 ∧ targetRect.height = 2021) :
  canFormRectangle targetRect jackTile :=
sorry

end NUMINAMATH_CALUDE_rectangle_from_triangles_l1701_170141


namespace NUMINAMATH_CALUDE_total_interest_is_1380_l1701_170157

def total_investment : ℝ := 17000
def low_rate_investment : ℝ := 12000
def low_rate : ℝ := 0.04
def high_rate : ℝ := 0.18

def calculate_total_interest : ℝ := 
  let high_rate_investment := total_investment - low_rate_investment
  let low_rate_interest := low_rate_investment * low_rate
  let high_rate_interest := high_rate_investment * high_rate
  low_rate_interest + high_rate_interest

theorem total_interest_is_1380 : 
  calculate_total_interest = 1380 := by sorry

end NUMINAMATH_CALUDE_total_interest_is_1380_l1701_170157


namespace NUMINAMATH_CALUDE_tens_digit_of_9_to_1503_l1701_170102

theorem tens_digit_of_9_to_1503 : ∃ n : ℕ, n ≥ 0 ∧ n < 10 ∧ 9^1503 ≡ 20 + n [ZMOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_9_to_1503_l1701_170102
