import Mathlib

namespace NUMINAMATH_CALUDE_probability_king_or_queen_l1502_150205

-- Define the structure of a standard deck
structure StandardDeck :=
  (total_cards : ℕ)
  (num_ranks : ℕ)
  (num_suits : ℕ)
  (num_kings : ℕ)
  (num_queens : ℕ)

-- Define the properties of a standard deck
def is_standard_deck (d : StandardDeck) : Prop :=
  d.total_cards = 52 ∧
  d.num_ranks = 13 ∧
  d.num_suits = 4 ∧
  d.num_kings = 4 ∧
  d.num_queens = 4

-- Theorem statement
theorem probability_king_or_queen (d : StandardDeck) 
  (h : is_standard_deck d) : 
  (d.num_kings + d.num_queens : ℚ) / d.total_cards = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_king_or_queen_l1502_150205


namespace NUMINAMATH_CALUDE_prob_less_than_two_defective_l1502_150272

/-- The probability of selecting fewer than 2 defective products -/
theorem prob_less_than_two_defective (total : Nat) (defective : Nat) (selected : Nat) 
  (h1 : total = 10) (h2 : defective = 3) (h3 : selected = 2) : 
  (Nat.choose (total - defective) selected + 
   Nat.choose (total - defective) (selected - 1) * Nat.choose defective 1) / 
  Nat.choose total selected = 14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_two_defective_l1502_150272


namespace NUMINAMATH_CALUDE_queen_of_hearts_favorites_l1502_150274

/-- The Queen of Hearts' pretzel distribution problem -/
theorem queen_of_hearts_favorites (total_guests : ℕ) (total_pretzels : ℕ) 
  (favorite_pretzels : ℕ) (non_favorite_pretzels : ℕ) (favorite_guests : ℕ) :
  total_guests = 30 →
  total_pretzels = 100 →
  favorite_pretzels = 4 →
  non_favorite_pretzels = 3 →
  favorite_guests * favorite_pretzels + (total_guests - favorite_guests) * non_favorite_pretzels = total_pretzels →
  favorite_guests = 10 := by
sorry

end NUMINAMATH_CALUDE_queen_of_hearts_favorites_l1502_150274


namespace NUMINAMATH_CALUDE_x_intercept_implies_a_value_l1502_150260

-- Define the line equation
def line_equation (a x y : ℝ) : Prop := a * x + y + 2 = 0

-- Define the x-intercept
def x_intercept (a : ℝ) : ℝ := 2

-- Theorem statement
theorem x_intercept_implies_a_value :
  ∀ a : ℝ, (∃ y : ℝ, line_equation a (x_intercept a) y) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_implies_a_value_l1502_150260


namespace NUMINAMATH_CALUDE_new_average_age_l1502_150249

def initial_people : ℕ := 8
def initial_average_age : ℚ := 35
def leaving_person_age : ℕ := 25
def remaining_people : ℕ := 7

theorem new_average_age :
  let total_age : ℚ := initial_people * initial_average_age
  let remaining_age : ℚ := total_age - leaving_person_age
  remaining_age / remaining_people = 36.42857 := by
  sorry

end NUMINAMATH_CALUDE_new_average_age_l1502_150249


namespace NUMINAMATH_CALUDE_mobile_phone_cost_mobile_phone_cost_is_8000_l1502_150294

/-- Proves that the cost of the mobile phone is 8000, given the conditions of the problem -/
theorem mobile_phone_cost : ℕ → Prop :=
  fun cost_mobile =>
    let cost_refrigerator : ℕ := 15000
    let loss_rate_refrigerator : ℚ := 2 / 100
    let profit_rate_mobile : ℚ := 10 / 100
    let overall_profit : ℕ := 500
    let selling_price_refrigerator : ℚ := cost_refrigerator * (1 - loss_rate_refrigerator)
    let selling_price_mobile : ℚ := cost_mobile * (1 + profit_rate_mobile)
    selling_price_refrigerator + selling_price_mobile - (cost_refrigerator + cost_mobile) = overall_profit →
    cost_mobile = 8000

/-- The cost of the mobile phone is 8000 -/
theorem mobile_phone_cost_is_8000 : mobile_phone_cost 8000 := by
  sorry

end NUMINAMATH_CALUDE_mobile_phone_cost_mobile_phone_cost_is_8000_l1502_150294


namespace NUMINAMATH_CALUDE_beach_probability_l1502_150215

theorem beach_probability (total : ℕ) (sunglasses : ℕ) (caps : ℕ) (cap_and_sunglasses_prob : ℚ) :
  total = 100 →
  sunglasses = 70 →
  caps = 60 →
  cap_and_sunglasses_prob = 2/3 →
  (cap_and_sunglasses_prob * caps : ℚ) / sunglasses = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_beach_probability_l1502_150215


namespace NUMINAMATH_CALUDE_first_triangle_is_isosceles_l1502_150220

-- Define a triangle structure
structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_eq_180 : α + β + γ = 180

-- Define the theorem
theorem first_triangle_is_isosceles 
  (t1 t2 : Triangle) 
  (h1 : ∃ (θ : Real), t1.α + t1.β = θ ∧ θ ≤ 180) 
  (h2 : ∃ (φ : Real), t1.β + t1.γ = φ ∧ φ ≤ 180) : 
  t1.α = t1.γ ∨ t1.α = t1.β ∨ t1.β = t1.γ :=
sorry

end NUMINAMATH_CALUDE_first_triangle_is_isosceles_l1502_150220


namespace NUMINAMATH_CALUDE_diagonal_intersection_point_l1502_150231

-- Define the four lines
def line1 (k b x : ℝ) : ℝ := k * x + b
def line2 (k b x : ℝ) : ℝ := k * x - b
def line3 (m b x : ℝ) : ℝ := m * x + b
def line4 (m b x : ℝ) : ℝ := m * x - b

-- Define the intersection point of diagonals
def intersection_point (k m b : ℝ) : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem diagonal_intersection_point (k m b : ℝ) (h : k ≠ m) :
  let vertices := [
    (0, b),
    ((-2 * b) / (k - m), (b * k - b * m) / (k - m)),
    (0, -b),
    ((2 * b) / (k - m), (-b * k + b * m) / (k - m))
  ]
  intersection_point k m b = (0, 0) := by sorry

end NUMINAMATH_CALUDE_diagonal_intersection_point_l1502_150231


namespace NUMINAMATH_CALUDE_expanded_binomial_equals_x_power_minus_one_l1502_150273

theorem expanded_binomial_equals_x_power_minus_one (x : ℝ) :
  (x - 1)^5 + 5*(x - 1)^4 + 10*(x - 1)^3 + 10*(x - 1)^2 + 5*(x - 1) = x^5 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expanded_binomial_equals_x_power_minus_one_l1502_150273


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l1502_150232

theorem arithmetic_sequence_length (a d last : ℕ) (h : last = a + (n - 1) * d) : 
  a = 2 → d = 5 → last = 2507 → n = 502 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l1502_150232


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l1502_150219

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 4620 → 
  Nat.gcd a b = 22 → 
  a = 220 → 
  b = 462 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l1502_150219


namespace NUMINAMATH_CALUDE_complex_exponential_form_l1502_150238

/-- For the complex number z = 1 + i√3, when expressed in the form re^(iθ), θ = π/3 -/
theorem complex_exponential_form (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_form_l1502_150238


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1502_150250

/-- The perimeter of an equilateral triangle with side length 23 centimeters is 69 centimeters. -/
theorem equilateral_triangle_perimeter :
  ∀ (triangle : Set ℝ × Set ℝ),
    (∀ side : ℝ, side ∈ (triangle.1 ∪ triangle.2) → side = 23) →
    (∃ (a b c : ℝ), a ∈ triangle.1 ∧ b ∈ triangle.1 ∧ c ∈ triangle.2 ∧
      a + b + c = 69) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1502_150250


namespace NUMINAMATH_CALUDE_intersection_range_l1502_150292

def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

def ellipse (m : ℝ) (x y : ℝ) : Prop := x^2 / 5 + y^2 / m = 1

theorem intersection_range (k : ℝ) (m : ℝ) :
  (∀ x y, line k x = y ∧ ellipse m x y → 
    ∃ x' y', x ≠ x' ∧ line k x' = y' ∧ ellipse m x' y') →
  m ∈ Set.Ioo 1 5 ∪ Set.Ioi 5 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l1502_150292


namespace NUMINAMATH_CALUDE_arc_length_parametric_curve_l1502_150251

open Real MeasureTheory

/-- The arc length of the curve given by the parametric equations
    x = e^t(cos t + sin t) and y = e^t(cos t - sin t) for 0 ≤ t ≤ 2π -/
theorem arc_length_parametric_curve :
  let x : ℝ → ℝ := fun t ↦ exp t * (cos t + sin t)
  let y : ℝ → ℝ := fun t ↦ exp t * (cos t - sin t)
  let curve_length := ∫ t in Set.Icc 0 (2 * π), sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2)
  curve_length = 2 * (exp (2 * π) - 1) := by
sorry

end NUMINAMATH_CALUDE_arc_length_parametric_curve_l1502_150251


namespace NUMINAMATH_CALUDE_base_h_solution_l1502_150296

/-- Represents a digit in base h --/
def Digit (h : ℕ) := {d : ℕ // d < h}

/-- Converts a natural number to its representation in base h --/
def toBaseH (n h : ℕ) : List (Digit h) :=
  sorry

/-- Performs addition in base h --/
def addBaseH (a b : List (Digit h)) : List (Digit h) :=
  sorry

/-- The given addition problem --/
def additionProblem (h : ℕ) : Prop :=
  let a := toBaseH 5342 h
  let b := toBaseH 6421 h
  let result := toBaseH 14263 h
  addBaseH a b = result

theorem base_h_solution :
  ∃ h : ℕ, h > 0 ∧ additionProblem h ∧ h = 8 :=
sorry

end NUMINAMATH_CALUDE_base_h_solution_l1502_150296


namespace NUMINAMATH_CALUDE_nail_trimming_customers_l1502_150218

/-- The number of nails per customer -/
def nails_per_customer : ℕ := 20

/-- The total number of sounds produced by the nail cutter -/
def total_sounds : ℕ := 120

/-- The number of customers -/
def num_customers : ℕ := total_sounds / nails_per_customer

theorem nail_trimming_customers :
  num_customers = 6 :=
by sorry

end NUMINAMATH_CALUDE_nail_trimming_customers_l1502_150218


namespace NUMINAMATH_CALUDE_geometric_series_equality_l1502_150258

/-- Given real numbers a and b satisfying an infinite geometric series equation,
    prove that another related infinite geometric series equals 3/4. -/
theorem geometric_series_equality (a b : ℝ) 
  (h : (a / (2 * b)) / (1 - 1 / (2 * b)) = 6) :
  (a / (a + 2 * b)) / (1 - 1 / (a + 2 * b)) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l1502_150258


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1502_150275

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 4}

theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1502_150275


namespace NUMINAMATH_CALUDE_circle_circumference_l1502_150211

theorem circle_circumference (r : ℝ) (h : π * r^2 = 4 * π) : 2 * π * r = 4 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_l1502_150211


namespace NUMINAMATH_CALUDE_base_of_power_l1502_150245

theorem base_of_power (b : ℝ) (x y : ℤ) 
  (h1 : b^x * 4^y = 531441)
  (h2 : x - y = 12)
  (h3 : x = 12) : 
  b = 3 := by sorry

end NUMINAMATH_CALUDE_base_of_power_l1502_150245


namespace NUMINAMATH_CALUDE_f_divides_m_minus_n_prime_condition_l1502_150280

def f (x : ℚ) : ℕ :=
  let p := x.num.natAbs
  let q := x.den
  p + q

theorem f_divides_m_minus_n (x : ℚ) (m n : ℕ) (h : m > 0) (k : n > 0) (hx : x > 0) :
  f x = f ((m : ℚ) * x / n) → (f x : ℤ) ∣ |m - n| :=
by sorry

theorem prime_condition (n : ℕ) (hn : n > 1) :
  (∀ x : ℚ, x > 0 → f x = f ((2^n : ℚ) * x) → f x = 2^n - 1) ↔ Nat.Prime n :=
by sorry

end NUMINAMATH_CALUDE_f_divides_m_minus_n_prime_condition_l1502_150280


namespace NUMINAMATH_CALUDE_stating_min_mozart_and_bach_not_beethoven_is_ten_l1502_150213

/-- Represents the preferences of a group of people for classical composers -/
structure ComposerPreferences where
  total : ℕ
  likes_mozart : ℕ
  likes_bach : ℕ
  likes_beethoven : ℕ

/-- 
Calculates the minimum number of people who like both Mozart and Bach but not Beethoven
given the preferences of a group of people for classical composers.
-/
def min_mozart_and_bach_not_beethoven (prefs : ComposerPreferences) : ℕ :=
  max 0 (prefs.likes_mozart + prefs.likes_bach - prefs.likes_beethoven - prefs.total)

/-- 
Theorem stating that for a group of 200 people where 160 like Mozart, 120 like Bach, 
and 90 like Beethoven, the minimum number of people who like both Mozart and Bach 
but not Beethoven is 10.
-/
theorem min_mozart_and_bach_not_beethoven_is_ten :
  min_mozart_and_bach_not_beethoven 
    { total := 200
    , likes_mozart := 160
    , likes_bach := 120
    , likes_beethoven := 90 } = 10 := by
  sorry

end NUMINAMATH_CALUDE_stating_min_mozart_and_bach_not_beethoven_is_ten_l1502_150213


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l1502_150288

theorem quadratic_solution_property (k : ℝ) : 
  (∃ a b : ℝ, 6 * a^2 + 5 * a + k = 0 ∧ 
              6 * b^2 + 5 * b + k = 0 ∧ 
              a ≠ b ∧
              |a - b| = 3 * (a^2 + b^2)) ↔ 
  (k = 1 ∨ k = -17900 / 864) :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l1502_150288


namespace NUMINAMATH_CALUDE_least_three_digit_product_6_l1502_150277

/-- A function that returns the product of the digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is three-digit -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_product_6 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 6 → 116 ≤ n := by sorry

end NUMINAMATH_CALUDE_least_three_digit_product_6_l1502_150277


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1502_150244

theorem unique_solution_condition (A B : ℝ) :
  (∀ x y : ℝ, A * x + B * ⌊x⌋ = A * y + B * ⌊y⌋ → x = y) ↔ 
  (A = 0 ∨ -2 < B / A ∧ B / A < 0) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1502_150244


namespace NUMINAMATH_CALUDE_license_plate_difference_l1502_150237

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits -/
def digit_size : ℕ := 10

/-- The number of letters in a California license plate -/
def california_letters : ℕ := 4

/-- The number of digits in a California license plate -/
def california_digits : ℕ := 3

/-- The number of letters in a Texas license plate -/
def texas_letters : ℕ := 3

/-- The number of digits in a Texas license plate -/
def texas_digits : ℕ := 4

/-- The number of possible California license plates -/
def california_plates : ℕ := alphabet_size ^ california_letters * digit_size ^ california_digits

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := alphabet_size ^ texas_letters * digit_size ^ texas_digits

/-- The difference in the number of possible license plates between California and Texas -/
theorem license_plate_difference : california_plates - texas_plates = 281216000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_difference_l1502_150237


namespace NUMINAMATH_CALUDE_abs_ratio_greater_than_one_l1502_150299

theorem abs_ratio_greater_than_one {a b : ℝ} (h1 : a < b) (h2 : b < 0) : |a| / |b| > 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_ratio_greater_than_one_l1502_150299


namespace NUMINAMATH_CALUDE_angle_relationship_l1502_150265

theorem angle_relationship (larger_angle smaller_angle : ℝ) : 
  larger_angle = 99 ∧ smaller_angle = 81 → larger_angle - smaller_angle = 18 := by
  sorry

end NUMINAMATH_CALUDE_angle_relationship_l1502_150265


namespace NUMINAMATH_CALUDE_parabola_translation_l1502_150201

/-- Represents a parabola of the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h - dx, k := p.k + dy }

theorem parabola_translation (p : Parabola) (dx dy : ℝ) :
  p.a = 2 ∧ p.h = 4 ∧ p.k = 3 ∧ dx = 4 ∧ dy = 3 →
  let p' := translate p dx dy
  p'.a = 2 ∧ p'.h = 0 ∧ p'.k = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l1502_150201


namespace NUMINAMATH_CALUDE_fraction_denominator_value_l1502_150283

theorem fraction_denominator_value (p q : ℚ) (x : ℚ) 
  (h1 : p / q = 4 / 5)
  (h2 : 11 / 7 + (2 * q - p) / x = 2) : x = 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_value_l1502_150283


namespace NUMINAMATH_CALUDE_two_true_propositions_l1502_150239

theorem two_true_propositions (a b c : ℝ) : 
  (∃! n : ℕ, n = 2 ∧ 
    n = (if (a > b → a*c^2 > b*c^2) then 1 else 0) +
        (if (a*c^2 > b*c^2 → a > b) then 1 else 0) +
        (if (a ≤ b → a*c^2 ≤ b*c^2) then 1 else 0) +
        (if (a*c^2 ≤ b*c^2 → a ≤ b) then 1 else 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_two_true_propositions_l1502_150239


namespace NUMINAMATH_CALUDE_vivienne_phone_count_l1502_150285

theorem vivienne_phone_count : ∀ (v : ℕ), 
  (400 * v + 400 * (v + 10) = 36000) → v = 40 := by
  sorry

end NUMINAMATH_CALUDE_vivienne_phone_count_l1502_150285


namespace NUMINAMATH_CALUDE_triple_base_and_exponent_l1502_150217

variable (a b x : ℝ)
variable (r : ℝ)

theorem triple_base_and_exponent (h1 : b ≠ 0) (h2 : r = (3 * a) ^ (3 * b)) (h3 : r = a ^ b * x ^ b) : x = 27 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triple_base_and_exponent_l1502_150217


namespace NUMINAMATH_CALUDE_wire_cutting_l1502_150224

theorem wire_cutting (total_length : ℝ) (difference : ℝ) (shorter_part : ℝ) : 
  total_length = 180 ∧ difference = 32 → 
  shorter_part + (shorter_part + difference) = total_length →
  shorter_part = 74 := by sorry

end NUMINAMATH_CALUDE_wire_cutting_l1502_150224


namespace NUMINAMATH_CALUDE_surface_area_of_problem_structure_l1502_150225

/-- Represents a solid formed by unit cubes -/
structure CubeStructure where
  base_layer : Nat
  middle_layer : Nat
  top_layer : Nat
  base_width : Nat
  base_length : Nat

/-- Calculates the surface area of the cube structure -/
def surface_area (c : CubeStructure) : Nat :=
  sorry

/-- The specific cube structure described in the problem -/
def problem_structure : CubeStructure :=
  { base_layer := 6
  , middle_layer := 4
  , top_layer := 2
  , base_width := 2
  , base_length := 3 }

theorem surface_area_of_problem_structure :
  surface_area problem_structure = 36 :=
sorry

end NUMINAMATH_CALUDE_surface_area_of_problem_structure_l1502_150225


namespace NUMINAMATH_CALUDE_train_cars_problem_l1502_150226

theorem train_cars_problem (passenger_cars cargo_cars : ℕ) : 
  cargo_cars = passenger_cars / 2 + 3 →
  passenger_cars + cargo_cars + 2 = 71 →
  passenger_cars = 44 := by
  sorry

end NUMINAMATH_CALUDE_train_cars_problem_l1502_150226


namespace NUMINAMATH_CALUDE_part_one_part_two_l1502_150284

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part 1
theorem part_one (m : ℝ) (h_m : m > 0) 
  (h_set : Set.Icc (-3/2) (1/2) = {x | f (x + 1) ≤ 2 * m}) : 
  m = 1 := by sorry

-- Part 2
theorem part_two : 
  (∃ n : ℝ, (∀ x y : ℝ, f (x + 1) ≤ 2018^y + n / 2018^y + |2*x - 1|) ∧ 
  (∀ n' : ℝ, (∀ x y : ℝ, f (x + 1) ≤ 2018^y + n' / 2018^y + |2*x - 1|) → n ≤ n')) ∧
  (∀ n : ℝ, (∀ x y : ℝ, f (x + 1) ≤ 2018^y + n / 2018^y + |2*x - 1|) → n ≥ 1) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1502_150284


namespace NUMINAMATH_CALUDE_danny_fish_tank_theorem_l1502_150289

/-- Represents the fish tank contents and sales --/
structure FishTank where
  initialGuppies : Nat
  initialAngelfish : Nat
  initialTigerSharks : Nat
  initialOscarFish : Nat
  soldGuppies : Nat
  soldAngelfish : Nat
  soldTigerSharks : Nat
  soldOscarFish : Nat

/-- Calculates the remaining fish in the tank --/
def remainingFish (tank : FishTank) : Nat :=
  (tank.initialGuppies + tank.initialAngelfish + tank.initialTigerSharks + tank.initialOscarFish) -
  (tank.soldGuppies + tank.soldAngelfish + tank.soldTigerSharks + tank.soldOscarFish)

/-- Theorem stating that the remaining fish in Danny's tank is 198 --/
theorem danny_fish_tank_theorem (tank : FishTank) 
  (h1 : tank.initialGuppies = 94)
  (h2 : tank.initialAngelfish = 76)
  (h3 : tank.initialTigerSharks = 89)
  (h4 : tank.initialOscarFish = 58)
  (h5 : tank.soldGuppies = 30)
  (h6 : tank.soldAngelfish = 48)
  (h7 : tank.soldTigerSharks = 17)
  (h8 : tank.soldOscarFish = 24) :
  remainingFish tank = 198 := by
  sorry

end NUMINAMATH_CALUDE_danny_fish_tank_theorem_l1502_150289


namespace NUMINAMATH_CALUDE_sequence_inequality_l1502_150281

theorem sequence_inequality (a : ℕ → ℝ) 
  (h1 : a 1 = π / 3)
  (h2 : ∀ n, 0 < a n ∧ a n < π / 3)
  (h3 : ∀ n ≥ 2, Real.sin (a (n + 1)) ≤ (1 / 3) * Real.sin (3 * a n)) :
  ∀ n, Real.sin (a n) < 1 / Real.sqrt n := by
sorry

end NUMINAMATH_CALUDE_sequence_inequality_l1502_150281


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_l1502_150287

/-- The number of factors of 18000 that are perfect squares -/
def num_perfect_square_factors : ℕ := 8

/-- The prime factorization of 18000 -/
def factorization_18000 : List (ℕ × ℕ) := [(2, 3), (3, 2), (5, 3)]

/-- Theorem: The number of factors of 18000 that are perfect squares is 8 -/
theorem count_perfect_square_factors :
  (List.prod (factorization_18000.map (fun (p, e) => e + 1)) / 8 : ℚ).num = num_perfect_square_factors := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_l1502_150287


namespace NUMINAMATH_CALUDE_mary_carrots_proof_l1502_150261

/-- The number of carrots Sandy grew -/
def sandys_carrots : ℕ := 8

/-- The total number of carrots grown by Sandy and Mary -/
def total_carrots : ℕ := 14

/-- The number of carrots Mary grew -/
def marys_carrots : ℕ := total_carrots - sandys_carrots

theorem mary_carrots_proof : 
  marys_carrots = total_carrots - sandys_carrots :=
by sorry

end NUMINAMATH_CALUDE_mary_carrots_proof_l1502_150261


namespace NUMINAMATH_CALUDE_line_points_k_value_l1502_150246

theorem line_points_k_value (m n k : ℝ) : 
  (m = 2 * n + 5) → 
  (m + 4 = 2 * (n + k) + 5) → 
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_line_points_k_value_l1502_150246


namespace NUMINAMATH_CALUDE_bridget_sarah_cents_difference_bridget_sarah_solution_l1502_150291

theorem bridget_sarah_cents_difference : ℕ → ℕ → ℕ → Prop :=
  fun total sarah_cents difference =>
    total = 300 ∧
    sarah_cents = 125 ∧
    difference = total - 2 * sarah_cents

theorem bridget_sarah_solution :
  ∃ (difference : ℕ), bridget_sarah_cents_difference 300 125 difference ∧ difference = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_bridget_sarah_cents_difference_bridget_sarah_solution_l1502_150291


namespace NUMINAMATH_CALUDE_complex_magnitude_l1502_150248

theorem complex_magnitude (z : ℂ) (h : (1 + Complex.I) * z = 1 - 7 * Complex.I) : 
  Complex.abs z = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1502_150248


namespace NUMINAMATH_CALUDE_tan_315_and_radian_conversion_l1502_150279

theorem tan_315_and_radian_conversion :
  Real.tan (315 * π / 180) = -1 ∧ 315 * π / 180 = 7 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_315_and_radian_conversion_l1502_150279


namespace NUMINAMATH_CALUDE_min_value_f_when_a_1_range_of_a_for_inequality_l1502_150257

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |x + a|

-- Theorem for the minimum value of f when a = 1
theorem min_value_f_when_a_1 :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x 1 ≥ f x_min 1 ∧ f x_min 1 = 3/2 :=
sorry

-- Theorem for the range of a
theorem range_of_a_for_inequality (a : ℝ) :
  (a > 0 ∧ ∃ (x : ℝ), x ∈ [1, 2] ∧ f x a < 5/x + a) ↔ 0 < a ∧ a < 6 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_when_a_1_range_of_a_for_inequality_l1502_150257


namespace NUMINAMATH_CALUDE_unique_P_value_l1502_150221

theorem unique_P_value (x y P : ℤ) : 
  x > 0 → y > 0 → x + y = P → 3 * x + 5 * y = 13 → P = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_P_value_l1502_150221


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1502_150252

theorem arithmetic_calculations : 
  (2 - 7 * (-3) + 10 + (-2) = 31) ∧ 
  (-1^2022 + 24 + (-2)^3 - 3^2 * (-1/3)^2 = 14) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1502_150252


namespace NUMINAMATH_CALUDE_shirt_discount_percentage_l1502_150203

/-- Calculates the discount percentage on a shirt given the original prices,
    total paid, and discount on the jacket. -/
theorem shirt_discount_percentage
  (jacket_price : ℝ)
  (shirt_price : ℝ)
  (total_paid : ℝ)
  (jacket_discount : ℝ)
  (h1 : jacket_price = 100)
  (h2 : shirt_price = 60)
  (h3 : total_paid = 110)
  (h4 : jacket_discount = 0.3)
  : (1 - (total_paid - jacket_price * (1 - jacket_discount)) / shirt_price) * 100 = 100 / 3 := by
  sorry

#eval (1 - (110 - 100 * (1 - 0.3)) / 60) * 100

end NUMINAMATH_CALUDE_shirt_discount_percentage_l1502_150203


namespace NUMINAMATH_CALUDE_seventeen_minus_fifteen_factorial_prime_divisors_l1502_150293

-- Define factorial function
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define the number of prime divisors function
def num_prime_divisors (n : ℕ) : ℕ := (Nat.factorization n).support.card

-- Theorem statement
theorem seventeen_minus_fifteen_factorial_prime_divisors :
  num_prime_divisors (factorial 17 - factorial 15) = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_minus_fifteen_factorial_prime_divisors_l1502_150293


namespace NUMINAMATH_CALUDE_square_8y_minus_5_l1502_150266

theorem square_8y_minus_5 (y : ℝ) (h : 4 * y^2 + 7 = 2 * y + 14) : (8 * y - 5)^2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_square_8y_minus_5_l1502_150266


namespace NUMINAMATH_CALUDE_lcm_fraction_even_l1502_150298

theorem lcm_fraction_even (n : ℕ) : 
  (n > 0) → (∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    n = (Nat.lcm x y + Nat.lcm y z) / Nat.lcm x z) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_lcm_fraction_even_l1502_150298


namespace NUMINAMATH_CALUDE_pairing_theorem_l1502_150262

/-- The number of ways to pair 2n points on a circle with n non-intersecting chords -/
def pairings (n : ℕ) : ℚ :=
  1 / (n + 1 : ℚ) * (Nat.choose (2 * n) n : ℚ)

/-- Theorem stating that the number of ways to pair 2n points on a circle
    with n non-intersecting chords is equal to (1 / (n+1)) * binomial(2n, n) -/
theorem pairing_theorem (n : ℕ) (h : n ≥ 1) :
  pairings n = 1 / (n + 1 : ℚ) * (Nat.choose (2 * n) n : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_pairing_theorem_l1502_150262


namespace NUMINAMATH_CALUDE_b_join_time_l1502_150242

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- Represents A's initial investment in Rupees -/
def aInvestment : ℕ := 45000

/-- Represents B's initial investment in Rupees -/
def bInvestment : ℕ := 27000

/-- Represents the ratio of profit sharing between A and B -/
def profitRatio : ℚ := 2 / 1

/-- 
Proves that B joined 2 months after A started the business, given the initial investments
and profit ratio.
-/
theorem b_join_time : 
  ∀ x : ℕ, 
  (aInvestment * monthsInYear) / (bInvestment * (monthsInYear - x)) = profitRatio → 
  x = 2 :=
by sorry

end NUMINAMATH_CALUDE_b_join_time_l1502_150242


namespace NUMINAMATH_CALUDE_abs_p_minus_q_equals_five_l1502_150235

theorem abs_p_minus_q_equals_five (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_p_minus_q_equals_five_l1502_150235


namespace NUMINAMATH_CALUDE_work_completion_time_l1502_150209

-- Define the work rates for a, b, and c
def work_rate_a : ℚ := 1 / 24
def work_rate_b : ℚ := 1 / 30
def work_rate_c : ℚ := 1 / 40

-- Define the combined work rate of a, b, and c
def combined_rate : ℚ := work_rate_a + work_rate_b + work_rate_c

-- Define the combined work rate of a and b
def combined_rate_ab : ℚ := work_rate_a + work_rate_b

-- Define the total days to complete the work
def total_days : ℚ := 11

-- Theorem statement
theorem work_completion_time :
  (total_days - 4) * combined_rate + 4 * combined_rate_ab = 1 :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1502_150209


namespace NUMINAMATH_CALUDE_fraction_value_l1502_150202

theorem fraction_value (a b : ℝ) (h1 : a * b ≠ 0) (h2 : 3 * a + 2 * b = 0) :
  (2 * a + b) / b = -1/3 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l1502_150202


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_fourth_l1502_150234

theorem opposite_of_negative_one_fourth : 
  (-(-(1/4 : ℚ))) = (1/4 : ℚ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_fourth_l1502_150234


namespace NUMINAMATH_CALUDE_expected_cards_theorem_l1502_150247

/-- A deck of cards with N cards, of which three are Aces -/
structure Deck :=
  (N : ℕ)
  (ace_count : Fin 3)

/-- The expected number of cards turned over until the second Ace appears -/
def expected_cards_until_second_ace (d : Deck) : ℚ :=
  (d.N + 1) / 2

/-- Theorem stating that the expected number of cards turned over until the second Ace appears is (N+1)/2 -/
theorem expected_cards_theorem (d : Deck) :
  expected_cards_until_second_ace d = (d.N + 1) / 2 := by
  sorry

#check expected_cards_theorem

end NUMINAMATH_CALUDE_expected_cards_theorem_l1502_150247


namespace NUMINAMATH_CALUDE_paul_pencil_sales_l1502_150282

def pencils_sold (daily_production : ℕ) (work_days : ℕ) (starting_stock : ℕ) (ending_stock : ℕ) : ℕ :=
  daily_production * work_days + starting_stock - ending_stock

theorem paul_pencil_sales : pencils_sold 100 5 80 230 = 350 := by
  sorry

end NUMINAMATH_CALUDE_paul_pencil_sales_l1502_150282


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_is_8_5_l1502_150271

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : 0 < std_dev

/-- The value that is exactly n standard deviations less than the mean -/
def value_n_std_dev_below_mean (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 10.5 and standard deviation 1,
    the value 2 standard deviations below the mean is 8.5 -/
theorem two_std_dev_below_mean_is_8_5 :
  let d : NormalDistribution := ⟨10.5, 1, by norm_num⟩
  value_n_std_dev_below_mean d 2 = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_is_8_5_l1502_150271


namespace NUMINAMATH_CALUDE_evaluate_expression_l1502_150297

theorem evaluate_expression : 2 - (-3) - 4 * (-5) - 6 - (-7) - 8 * (-9) + 10 = 108 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1502_150297


namespace NUMINAMATH_CALUDE_longest_interval_l1502_150276

-- Define the conversion factors
def minutes_per_hour : ℕ := 60
def hours_per_day : ℕ := 24

-- Define the time intervals
def interval_a : ℕ := 1500  -- in minutes
def interval_b : ℕ := 10    -- in hours
def interval_c : ℕ := 1     -- in days

-- Theorem to prove
theorem longest_interval :
  (interval_a : ℝ) > (interval_b * minutes_per_hour : ℝ) ∧
  (interval_a : ℝ) > (interval_c * hours_per_day * minutes_per_hour : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_longest_interval_l1502_150276


namespace NUMINAMATH_CALUDE_y_decreases_as_x_increases_l1502_150236

def linear_function (x : ℝ) : ℝ := -3 * x + 6

theorem y_decreases_as_x_increases :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → linear_function x₁ > linear_function x₂ := by
  sorry

end NUMINAMATH_CALUDE_y_decreases_as_x_increases_l1502_150236


namespace NUMINAMATH_CALUDE_instrument_probability_l1502_150212

theorem instrument_probability (total : ℕ) (at_least_one : ℕ) (two_or_more : ℕ) :
  total = 800 →
  at_least_one = total / 5 →
  two_or_more = 128 →
  (at_least_one - two_or_more : ℚ) / total = 1 / 25 := by
  sorry

end NUMINAMATH_CALUDE_instrument_probability_l1502_150212


namespace NUMINAMATH_CALUDE_third_generation_tail_length_l1502_150270

/-- The tail length of a generation of kittens -/
def tail_length (n : ℕ) : ℝ :=
  if n = 0 then 16
  else tail_length (n - 1) * 1.25

/-- The theorem stating that the third generation's tail length is 25 cm -/
theorem third_generation_tail_length :
  tail_length 2 = 25 := by sorry

end NUMINAMATH_CALUDE_third_generation_tail_length_l1502_150270


namespace NUMINAMATH_CALUDE_smallest_three_digit_number_satisfying_conditions_l1502_150230

theorem smallest_three_digit_number_satisfying_conditions : ∃ n : ℕ,
  (100 ≤ n ∧ n ≤ 999) ∧
  (∃ k : ℕ, n + 7 = 9 * k) ∧
  (∃ m : ℕ, n - 6 = 7 * m) ∧
  (∀ x : ℕ, (100 ≤ x ∧ x < n) → ¬((∃ k : ℕ, x + 7 = 9 * k) ∧ (∃ m : ℕ, x - 6 = 7 * m))) ∧
  n = 116 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_number_satisfying_conditions_l1502_150230


namespace NUMINAMATH_CALUDE_video_subscription_duration_l1502_150204

theorem video_subscription_duration (monthly_cost : ℚ) (total_paid : ℚ) : 
  monthly_cost = 14 →
  total_paid = 84 →
  (total_paid / (monthly_cost / 2)) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_video_subscription_duration_l1502_150204


namespace NUMINAMATH_CALUDE_range_of_t_when_p_range_of_t_when_p_xor_q_l1502_150214

-- Define the propositions
def p (t : ℝ) : Prop := ∀ x, x^2 + 2*x + 2*t - 4 ≠ 0

def q (t : ℝ) : Prop := 
  t ≠ 4 ∧ t ≠ 2 ∧ 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
  ∀ x y, x^2 / (4 - t) + y^2 / (t - 2) = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1

-- Theorem 1
theorem range_of_t_when_p (t : ℝ) : p t → t > 5/2 := by sorry

-- Theorem 2
theorem range_of_t_when_p_xor_q (t : ℝ) : 
  (p t ∨ q t) ∧ ¬(p t ∧ q t) → (2 < t ∧ t ≤ 5/2) ∨ t ≥ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_t_when_p_range_of_t_when_p_xor_q_l1502_150214


namespace NUMINAMATH_CALUDE_jack_quarantine_days_l1502_150241

/-- Calculates the number of days spent in quarantine given the total wait time and customs time. -/
def quarantine_days (total_hours : ℕ) (customs_hours : ℕ) : ℕ :=
  (total_hours - customs_hours) / 24

/-- Theorem stating that given a total wait time of 356 hours, including 20 hours for customs,
    the number of days spent in quarantine is 14. -/
theorem jack_quarantine_days :
  quarantine_days 356 20 = 14 := by
  sorry

end NUMINAMATH_CALUDE_jack_quarantine_days_l1502_150241


namespace NUMINAMATH_CALUDE_fgh_supermarkets_count_l1502_150216

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 49

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 14

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 84

theorem fgh_supermarkets_count :
  us_supermarkets = 49 ∧
  us_supermarkets + canada_supermarkets = total_supermarkets ∧
  us_supermarkets = canada_supermarkets + 14 :=
by sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_count_l1502_150216


namespace NUMINAMATH_CALUDE_value_of_X_l1502_150227

theorem value_of_X : ∃ X : ℚ, (1/3 : ℚ) * (1/4 : ℚ) * X = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ X = 60 := by
  sorry

end NUMINAMATH_CALUDE_value_of_X_l1502_150227


namespace NUMINAMATH_CALUDE_total_students_presentation_l1502_150243

/-- The total number of students presenting given Eunjeong's position and students after her -/
def total_students (eunjeong_position : Nat) (students_after : Nat) : Nat :=
  (eunjeong_position - 1) + 1 + students_after

/-- Theorem stating the total number of students presenting -/
theorem total_students_presentation : total_students 6 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_students_presentation_l1502_150243


namespace NUMINAMATH_CALUDE_parabola_midpoint_distance_squared_l1502_150206

/-- Given a parabola y = 3x^2 + 6x - 2 and two points C and D on it with the origin as their midpoint,
    the square of the distance between C and D is 740/3. -/
theorem parabola_midpoint_distance_squared :
  ∀ (C D : ℝ × ℝ),
  (∃ (x y : ℝ), C = (x, y) ∧ y = 3 * x^2 + 6 * x - 2) →
  (∃ (x y : ℝ), D = (x, y) ∧ y = 3 * x^2 + 6 * x - 2) →
  (0 : ℝ × ℝ) = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = 740 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_midpoint_distance_squared_l1502_150206


namespace NUMINAMATH_CALUDE_salad_price_proof_l1502_150210

/-- Proves that the price of each salad is $2.50 given the problem conditions --/
theorem salad_price_proof (hot_dog_price : ℝ) (hot_dog_count : ℕ) (salad_count : ℕ) 
  (payment : ℝ) (change : ℝ) :
  hot_dog_price = 1.5 →
  hot_dog_count = 5 →
  salad_count = 3 →
  payment = 20 →
  change = 5 →
  (payment - change - hot_dog_price * hot_dog_count) / salad_count = 2.5 := by
sorry

#eval (20 - 5 - 1.5 * 5) / 3

end NUMINAMATH_CALUDE_salad_price_proof_l1502_150210


namespace NUMINAMATH_CALUDE_largest_equal_cost_number_l1502_150255

/-- Calculates the cost of transmitting a number using Option 1 (decimal representation) -/
def option1Cost (n : Nat) : Nat :=
  sorry

/-- Calculates the cost of transmitting a number using Option 2 (binary representation) -/
def option2Cost (n : Nat) : Nat :=
  sorry

/-- Checks if the costs are equal for both options -/
def costsEqual (n : Nat) : Bool :=
  option1Cost n = option2Cost n

/-- Theorem stating that 1118 is the largest number less than 2000 with equal costs -/
theorem largest_equal_cost_number :
  (∀ n : Nat, n < 2000 → costsEqual n → n ≤ 1118) ∧ costsEqual 1118 := by
  sorry

end NUMINAMATH_CALUDE_largest_equal_cost_number_l1502_150255


namespace NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_sixth_l1502_150254

theorem arcsin_one_half_equals_pi_sixth : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_equals_pi_sixth_l1502_150254


namespace NUMINAMATH_CALUDE_rectangle_has_equal_diagonals_l1502_150240

-- Define a rectangle
def isRectangle (ABCD : Quadrilateral) : Prop := sorry

-- Define equal diagonals
def hasEqualDiagonals (ABCD : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem rectangle_has_equal_diagonals (ABCD : Quadrilateral) :
  isRectangle ABCD → hasEqualDiagonals ABCD := by sorry

end NUMINAMATH_CALUDE_rectangle_has_equal_diagonals_l1502_150240


namespace NUMINAMATH_CALUDE_hillary_climbing_rate_l1502_150286

/-- Represents the climbing scenario of Hillary and Eddy on Mt. Everest -/
structure ClimbingScenario where
  hillary_rate : ℝ
  eddy_rate : ℝ
  total_distance : ℝ
  hillary_stop_distance : ℝ
  hillary_descent_rate : ℝ
  total_time : ℝ

/-- The theorem stating that Hillary's climbing rate is 800 ft/hr -/
theorem hillary_climbing_rate 
  (scenario : ClimbingScenario)
  (h1 : scenario.total_distance = 5000)
  (h2 : scenario.eddy_rate = scenario.hillary_rate - 500)
  (h3 : scenario.hillary_stop_distance = 1000)
  (h4 : scenario.hillary_descent_rate = 1000)
  (h5 : scenario.total_time = 6)
  : scenario.hillary_rate = 800 := by
  sorry

end NUMINAMATH_CALUDE_hillary_climbing_rate_l1502_150286


namespace NUMINAMATH_CALUDE_charlie_has_31_pennies_l1502_150267

/-- The number of pennies Charlie has -/
def charlie_pennies : ℕ := 31

/-- The number of pennies Alex has -/
def alex_pennies : ℕ := 9

/-- Condition 1: If Alex gives Charlie a penny, Charlie will have four times as many pennies as Alex has -/
axiom condition1 : charlie_pennies + 1 = 4 * (alex_pennies - 1)

/-- Condition 2: If Charlie gives Alex a penny, Charlie will have three times as many pennies as Alex has -/
axiom condition2 : charlie_pennies - 1 = 3 * (alex_pennies + 1)

theorem charlie_has_31_pennies : charlie_pennies = 31 := by
  sorry

end NUMINAMATH_CALUDE_charlie_has_31_pennies_l1502_150267


namespace NUMINAMATH_CALUDE_max_value_polynomial_l1502_150256

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (max : ℝ), ∀ (a b : ℝ), a + b = 5 → x^3*y + x^2*y + x*y + x*y^2 + x*y^3 ≤ max) ∧
  (x^3*y + x^2*y + x*y + x*y^2 + x*y^3 ≤ 961/8) :=
sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l1502_150256


namespace NUMINAMATH_CALUDE_common_tangent_lower_bound_l1502_150253

/-- Given two curves C₁: y = ax² (a > 0) and C₂: y = e^x, if they have a common tangent line,
    then a ≥ e²/4 -/
theorem common_tangent_lower_bound (a : ℝ) (h_pos : a > 0) :
  (∃ x₁ x₂ : ℝ, (2 * a * x₁ = Real.exp x₂) ∧ 
                (a * x₁^2 - Real.exp x₂ = 2 * a * x₁ * (x₁ - x₂))) →
  a ≥ Real.exp 2 / 4 := by
  sorry


end NUMINAMATH_CALUDE_common_tangent_lower_bound_l1502_150253


namespace NUMINAMATH_CALUDE_multiple_of_8_in_second_column_thousand_in_second_column_l1502_150278

/-- Represents the column number in the arrangement -/
inductive Column
| First
| Second
| Third
| Fourth
| Fifth

/-- Represents the row type in the arrangement -/
inductive RowType
| Odd
| Even

/-- Function to determine the column of a given integer in the arrangement -/
def column_of_integer (n : ℕ) : Column :=
  sorry

/-- Function to determine the row type of a given integer in the arrangement -/
def row_type_of_integer (n : ℕ) : RowType :=
  sorry

/-- Theorem stating that any multiple of 8 appears in the second column -/
theorem multiple_of_8_in_second_column (n : ℕ) (h : 8 ∣ n) : column_of_integer n = Column.Second :=
  sorry

/-- Corollary: 1000 appears in the second column -/
theorem thousand_in_second_column : column_of_integer 1000 = Column.Second :=
  sorry

end NUMINAMATH_CALUDE_multiple_of_8_in_second_column_thousand_in_second_column_l1502_150278


namespace NUMINAMATH_CALUDE_cricketer_bowling_runs_cricketer_last_match_runs_l1502_150229

theorem cricketer_bowling_runs (initial_average : ℝ) (initial_wickets : ℕ) 
  (last_match_wickets : ℕ) (average_decrease : ℝ) : ℝ :=
  let final_average := initial_average - average_decrease
  let total_wickets := initial_wickets + last_match_wickets
  let initial_runs := initial_average * initial_wickets
  let final_runs := final_average * total_wickets
  final_runs - initial_runs

theorem cricketer_last_match_runs : 
  cricketer_bowling_runs 12.4 85 5 0.4 = 26 := by sorry

end NUMINAMATH_CALUDE_cricketer_bowling_runs_cricketer_last_match_runs_l1502_150229


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1502_150208

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies a horizontal shift to a parabola -/
def horizontal_shift (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := 2 * p.a * h + p.b,
    c := p.a * h^2 + p.b * h + p.c }

/-- Applies a vertical shift to a parabola -/
def vertical_shift (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a,
    b := p.b,
    c := p.c + v }

theorem parabola_shift_theorem (p : Parabola) :
  let p1 := horizontal_shift p (-1)
  let p2 := vertical_shift p1 (-3)
  p.a = -2 ∧ p.b = 0 ∧ p.c = 0 →
  p2.a = -2 ∧ p2.b = -4 ∧ p2.c = -5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1502_150208


namespace NUMINAMATH_CALUDE_two_distinct_roots_l1502_150228

theorem two_distinct_roots
  (a b c d : ℝ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (f_no_roots : ∀ x : ℝ, x^2 + b*x + a ≠ 0)
  (g_condition1 : a^2 + c*a + d = b)
  (g_condition2 : b^2 + c*b + d = a) :
  ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l1502_150228


namespace NUMINAMATH_CALUDE_units_digit_of_n_l1502_150207

/-- Given two natural numbers m and n, returns true if m has a units digit of 3 -/
def has_units_digit_3 (m : ℕ) : Prop :=
  m % 10 = 3

/-- Given a natural number n, returns its units digit -/
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 14^8) (h2 : has_units_digit_3 m) :
  units_digit n = 2 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l1502_150207


namespace NUMINAMATH_CALUDE_christmas_play_volunteers_l1502_150200

/-- Calculates the number of additional volunteers needed for the school Christmas play -/
def additional_volunteers_needed (total_needed : ℕ) (math_classes : ℕ) (students_per_class : ℕ) (teachers : ℕ) : ℕ :=
  total_needed - (math_classes * students_per_class + teachers)

/-- Proves that given the specified conditions, 50 additional volunteers are needed -/
theorem christmas_play_volunteers : 
  additional_volunteers_needed 80 5 4 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_christmas_play_volunteers_l1502_150200


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l1502_150268

/-- Represents a batsman's performance -/
structure Batsman where
  initialAverage : ℝ  -- Average before the 17th inning
  runsScored : ℕ      -- Runs scored in the 17th inning
  averageIncrease : ℝ -- Increase in average after the 17th inning

/-- Calculates the new average after the 17th inning -/
def newAverage (b : Batsman) : ℝ :=
  b.initialAverage + b.averageIncrease

/-- Theorem: The batsman's average after the 17th inning is 140 runs -/
theorem batsman_average_after_17th_inning (b : Batsman)
  (h1 : b.runsScored = 300)
  (h2 : b.averageIncrease = 10)
  : newAverage b = 140 := by
  sorry

#check batsman_average_after_17th_inning

end NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l1502_150268


namespace NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l1502_150269

/-- For an infinite geometric series with first term a and sum S, 
    the common ratio r can be calculated. -/
theorem infinite_geometric_series_ratio 
  (a : ℝ) (S : ℝ) (h1 : a = 400) (h2 : S = 2500) :
  let r := 1 - a / S
  r = 21 / 25 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_ratio_l1502_150269


namespace NUMINAMATH_CALUDE_governor_addresses_l1502_150264

theorem governor_addresses (sandoval hawkins sloan : ℕ) : 
  hawkins = sandoval / 2 →
  sloan = sandoval + 10 →
  sandoval + hawkins + sloan = 40 →
  sandoval = 12 := by
sorry

end NUMINAMATH_CALUDE_governor_addresses_l1502_150264


namespace NUMINAMATH_CALUDE_interval_necessary_not_sufficient_l1502_150263

theorem interval_necessary_not_sufficient :
  ¬(∀ x : ℝ, -1 ≤ x ∧ x ≤ 5 ↔ (x - 5) * (x + 1) < 0) ∧
  (∀ x : ℝ, (x - 5) * (x + 1) < 0 → -1 ≤ x ∧ x ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_interval_necessary_not_sufficient_l1502_150263


namespace NUMINAMATH_CALUDE_divisible_by_64_l1502_150290

theorem divisible_by_64 (n : ℕ+) : ∃ k : ℤ, (5 : ℤ)^n.val - 8*n.val^2 + 4*n.val - 1 = 64*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_64_l1502_150290


namespace NUMINAMATH_CALUDE_origin_outside_circle_l1502_150259

theorem origin_outside_circle (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + 2*a*x + 2*y + (a-1)^2 = 0}
  (0, 0) ∉ circle ∧ ∃ (p : ℝ × ℝ), p ∈ circle ∧ dist p (0, 0) < dist (0, 0) p := by
  sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l1502_150259


namespace NUMINAMATH_CALUDE_concentric_circles_radii_inequality_l1502_150222

theorem concentric_circles_radii_inequality (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a < b) (h5 : b < c) : 
  b + a ≠ c + b := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_inequality_l1502_150222


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l1502_150295

/-- Represents a rectangular field -/
structure RectangularField where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular field -/
def area (field : RectangularField) : ℝ :=
  field.width * field.length

/-- Calculates the perimeter of a rectangular field -/
def perimeter (field : RectangularField) : ℝ :=
  2 * (field.width + field.length)

theorem rectangular_field_perimeter 
  (field : RectangularField) 
  (h_area : area field = 50) 
  (h_width : field.width = 5) : 
  perimeter field = 30 := by
  sorry

#check rectangular_field_perimeter

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l1502_150295


namespace NUMINAMATH_CALUDE_x_over_y_value_l1502_150223

theorem x_over_y_value (x y : ℝ) (h1 : x ≠ y) 
  (h2 : x / y + (x + 5 * y) / (y + 5 * x) = 2) : 
  x / y = 0.6 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_value_l1502_150223


namespace NUMINAMATH_CALUDE_number_problem_l1502_150233

theorem number_problem (x : ℚ) :
  (35 / 100) * x = (25 / 100) * 40 → x = 200 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1502_150233
