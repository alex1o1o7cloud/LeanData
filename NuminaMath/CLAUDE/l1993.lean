import Mathlib

namespace NUMINAMATH_CALUDE_perfect_square_condition_l1993_199358

theorem perfect_square_condition (k : ℝ) :
  (∀ x y : ℝ, ∃ z : ℝ, 4 * x^2 - (k - 1) * x * y + 9 * y^2 = z^2) →
  k = 13 ∨ k = -11 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1993_199358


namespace NUMINAMATH_CALUDE_total_savings_ten_sets_l1993_199379

/-- The cost of 2 packs of milk -/
def cost_two_packs : ℚ := 2.50

/-- The cost of an individual pack of milk -/
def cost_individual : ℚ := 1.30

/-- The number of sets being purchased -/
def num_sets : ℕ := 10

/-- The number of packs in each set -/
def packs_per_set : ℕ := 2

/-- Theorem stating the total savings from buying ten sets of 2 packs of milk -/
theorem total_savings_ten_sets : 
  (num_sets * packs_per_set) * (cost_individual - cost_two_packs / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_ten_sets_l1993_199379


namespace NUMINAMATH_CALUDE_square_field_diagonal_l1993_199301

theorem square_field_diagonal (area : ℝ) (diagonal : ℝ) : 
  area = 450 → diagonal = 30 → diagonal^2 = 2 * area :=
by
  sorry

end NUMINAMATH_CALUDE_square_field_diagonal_l1993_199301


namespace NUMINAMATH_CALUDE_areas_product_eq_volume_squared_l1993_199368

/-- A rectangular box with specific proportions -/
structure Box where
  width : ℝ
  length : ℝ
  height : ℝ
  length_eq : length = 2 * width
  height_eq : height = 3 * width

/-- The volume of the box -/
def volume (b : Box) : ℝ := b.length * b.width * b.height

/-- The area of the bottom of the box -/
def bottomArea (b : Box) : ℝ := b.length * b.width

/-- The area of the side of the box -/
def sideArea (b : Box) : ℝ := b.width * b.height

/-- The area of the front of the box -/
def frontArea (b : Box) : ℝ := b.length * b.height

/-- Theorem: The product of the areas equals the square of the volume -/
theorem areas_product_eq_volume_squared (b : Box) :
  bottomArea b * sideArea b * frontArea b = (volume b) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_areas_product_eq_volume_squared_l1993_199368


namespace NUMINAMATH_CALUDE_money_distribution_l1993_199364

/-- Given three people with a total of $4000, where one person has two-thirds of the amount
    the other two have combined, prove that this person has $1600. -/
theorem money_distribution (total : ℚ) (r_share : ℚ) : 
  total = 4000 →
  r_share = (2/3) * (total - r_share) →
  r_share = 1600 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l1993_199364


namespace NUMINAMATH_CALUDE_aquarium_visitors_l1993_199377

/-- Calculates the number of healthy visitors given total visitors and ill percentage --/
def healthyVisitors (total : ℕ) (illPercentage : ℕ) : ℕ :=
  total - (total * illPercentage) / 100

/-- Properties of the aquarium visits over three days --/
theorem aquarium_visitors :
  let mondayTotal := 300
  let mondayIllPercentage := 15
  let tuesdayTotal := 500
  let tuesdayIllPercentage := 30
  let wednesdayTotal := 400
  let wednesdayIllPercentage := 20
  
  (healthyVisitors mondayTotal mondayIllPercentage +
   healthyVisitors tuesdayTotal tuesdayIllPercentage +
   healthyVisitors wednesdayTotal wednesdayIllPercentage) = 925 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_visitors_l1993_199377


namespace NUMINAMATH_CALUDE_total_owed_correct_l1993_199305

/-- Calculates the total amount owed after one year given monthly charges and interest rates -/
def totalOwed (jan_charge feb_charge mar_charge apr_charge : ℝ)
              (jan_rate feb_rate mar_rate apr_rate : ℝ) : ℝ :=
  let jan_total := jan_charge * (1 + jan_rate)
  let feb_total := feb_charge * (1 + feb_rate)
  let mar_total := mar_charge * (1 + mar_rate)
  let apr_total := apr_charge * (1 + apr_rate)
  jan_total + feb_total + mar_total + apr_total

/-- The theorem stating the total amount owed after one year -/
theorem total_owed_correct :
  totalOwed 35 45 55 25 0.05 0.07 0.04 0.06 = 168.60 := by
  sorry

end NUMINAMATH_CALUDE_total_owed_correct_l1993_199305


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1993_199313

/-- The discriminant of the quadratic equation 2x^2 + (2 + 1/2)x + 1/2 is 9/4 -/
theorem quadratic_discriminant : 
  let a : ℚ := 2
  let b : ℚ := 2 + 1/2
  let c : ℚ := 1/2
  (b^2 - 4*a*c) = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1993_199313


namespace NUMINAMATH_CALUDE_quadratic_through_origin_l1993_199361

/-- Given a quadratic function f(x) = ax^2 + x + a(a-2) that passes through the origin,
    prove that a = 2 -/
theorem quadratic_through_origin (a : ℝ) (h1 : a ≠ 0) :
  (∀ x, a*x^2 + x + a*(a-2) = 0 → x = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_through_origin_l1993_199361


namespace NUMINAMATH_CALUDE_cows_and_sheep_bushels_l1993_199342

/-- Represents the farm animals and their food consumption --/
structure Farm where
  cows : ℕ
  sheep : ℕ
  chickens : ℕ
  chicken_bushels : ℕ
  total_bushels : ℕ

/-- Calculates the bushels eaten by cows and sheep --/
def bushels_for_cows_and_sheep (farm : Farm) : ℕ :=
  farm.total_bushels - (farm.chickens * farm.chicken_bushels)

/-- Theorem stating that the bushels eaten by cows and sheep is 14 --/
theorem cows_and_sheep_bushels (farm : Farm) 
  (h1 : farm.cows = 4)
  (h2 : farm.sheep = 3)
  (h3 : farm.chickens = 7)
  (h4 : farm.chicken_bushels = 3)
  (h5 : farm.total_bushels = 35) :
  bushels_for_cows_and_sheep farm = 14 := by
  sorry

end NUMINAMATH_CALUDE_cows_and_sheep_bushels_l1993_199342


namespace NUMINAMATH_CALUDE_pyramid_division_theorem_l1993_199399

/-- A structure representing a pyramid divided by planes parallel to its base -/
structure DividedPyramid (n : ℕ) where
  volumePlanes : Fin (n + 1) → ℝ
  surfacePlanes : Fin (n + 1) → ℝ

/-- The condition for a common plane between volume and surface divisions -/
def hasCommonPlane (n : ℕ) : Prop :=
  ∃ (i k : ℕ), 1 ≤ i ∧ i ≤ n ∧ 1 ≤ k ∧ k ≤ n ∧ (n + 1) * i^2 = k^3

/-- The list of n values up to 100 that satisfy the common plane condition -/
def validNValues : List ℕ :=
  [7, 15, 23, 26, 31, 39, 47, 53, 55, 63, 71, 79, 80, 87, 95]

/-- The condition for multiple common planes -/
def hasMultipleCommonPlanes (n : ℕ) : Prop :=
  ∃ (i₁ k₁ i₂ k₂ : ℕ),
    1 ≤ i₁ ∧ i₁ ≤ n ∧ 1 ≤ k₁ ∧ k₁ ≤ n ∧
    1 ≤ i₂ ∧ i₂ ≤ n ∧ 1 ≤ k₂ ∧ k₂ ≤ n ∧
    (n + 1) * i₁^2 = k₁^3 ∧ (n + 1) * i₂^2 = k₂^3 ∧
    (i₁ ≠ i₂ ∨ k₁ ≠ k₂)

theorem pyramid_division_theorem :
  (∀ n ∈ validNValues, hasCommonPlane n) ∧
  (∀ n ∈ validNValues, n ≠ 63 → ¬hasMultipleCommonPlanes n) ∧
  hasMultipleCommonPlanes 63 :=
sorry

end NUMINAMATH_CALUDE_pyramid_division_theorem_l1993_199399


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l1993_199314

theorem gcd_digits_bound (a b : ℕ) (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7) 
  (hlcm : 10^10 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^11) : 
  Nat.gcd a b < 10^4 := by
sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l1993_199314


namespace NUMINAMATH_CALUDE_roots_and_inequality_solution_set_l1993_199350

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem roots_and_inequality_solution_set 
  (a b : ℝ) 
  (h1 : f a b (-1) = 0) 
  (h2 : f a b 2 = 0) :
  {x : ℝ | a * f a b (-2*x) > 0} = Set.Ioo (-1 : ℝ) (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_roots_and_inequality_solution_set_l1993_199350


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1993_199363

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 6 →
  b = Real.sqrt 3 →
  b + a * (Real.sin C - Real.cos C) = 0 →
  A + B + C = π →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  B = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1993_199363


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1993_199396

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := p.b - 2 * p.a * h, c := p.c + p.a * h^2 - p.b * h }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

theorem parabola_shift_theorem :
  let original := Parabola.mk 1 0 1  -- y = x^2 + 1
  let shifted_left := shift_horizontal original 2
  let final := shift_vertical shifted_left (-3)
  final = Parabola.mk 1 (-4) (-2)  -- y = (x + 2)^2 - 2
  := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1993_199396


namespace NUMINAMATH_CALUDE_expression_simplification_l1993_199343

theorem expression_simplification (p q r : ℝ) 
  (hp : p ≠ 2) (hq : q ≠ 3) (hr : r ≠ 4) : 
  (p - 2) / (4 - r) * (q - 3) / (2 - p) * (r - 4) / (3 - q) * (-2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1993_199343


namespace NUMINAMATH_CALUDE_second_article_loss_percentage_l1993_199353

/-- Proves that the loss percentage on the second article is 10% given the specified conditions --/
theorem second_article_loss_percentage
  (cost_price : ℝ)
  (profit_percent_first : ℝ)
  (net_profit_loss_percent : ℝ)
  (h1 : cost_price = 1000)
  (h2 : profit_percent_first = 10)
  (h3 : net_profit_loss_percent = 99.99999999999946) :
  let selling_price_first := cost_price * (1 + profit_percent_first / 100)
  let total_selling_price := 2 * cost_price * (1 + net_profit_loss_percent / 100)
  let selling_price_second := total_selling_price - selling_price_first
  let loss_second := cost_price - selling_price_second
  loss_second / cost_price * 100 = 10 := by
sorry


end NUMINAMATH_CALUDE_second_article_loss_percentage_l1993_199353


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1993_199340

theorem modulus_of_complex_fraction : 
  let z : ℂ := (1 + Complex.I * Real.sqrt 3) / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1993_199340


namespace NUMINAMATH_CALUDE_graduating_class_size_l1993_199330

/-- Given a graduating class where there are 208 boys and 69 more girls than boys,
    prove that the total number of students is 485. -/
theorem graduating_class_size :
  ∀ (boys girls total : ℕ),
  boys = 208 →
  girls = boys + 69 →
  total = boys + girls →
  total = 485 := by
sorry

end NUMINAMATH_CALUDE_graduating_class_size_l1993_199330


namespace NUMINAMATH_CALUDE_cos_equality_for_specific_angles_l1993_199370

theorem cos_equality_for_specific_angles :
  ∀ n : ℤ, 0 ≤ n ∧ n ≤ 360 →
    (Real.cos (n * π / 180) = Real.cos (321 * π / 180) ↔ n = 39 ∨ n = 321) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_for_specific_angles_l1993_199370


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1993_199326

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Converts a ThreeDigitNumber to its numerical value -/
def to_nat (n : ThreeDigitNumber) : Nat :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- Checks if the digits of a ThreeDigitNumber are distinct -/
def has_distinct_digits (n : ThreeDigitNumber) : Prop :=
  n.1 ≠ n.2.1 ∧ n.1 ≠ n.2.2 ∧ n.2.1 ≠ n.2.2

/-- The main theorem stating that 156 is the only number satisfying the conditions -/
theorem unique_three_digit_number : 
  ∀ n : ThreeDigitNumber, 
    has_distinct_digits n → 
    (100 ≤ to_nat n) ∧ (to_nat n ≤ 999) → 
    (to_nat n = (n.1 + n.2.1 + n.2.2) * (n.1 + n.2.1 + n.2.2 + 1)) → 
    n = (1, 5, 6) :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1993_199326


namespace NUMINAMATH_CALUDE_triangle_side_length_l1993_199387

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  C = 2 * A ∧
  Real.cos A = 3/4 ∧
  a * c * Real.cos B = 27/2 →
  b = 5 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1993_199387


namespace NUMINAMATH_CALUDE_saree_price_theorem_l1993_199315

/-- The original price of sarees before discounts -/
def original_price : ℝ := 550

/-- The first discount rate -/
def discount1 : ℝ := 0.18

/-- The second discount rate -/
def discount2 : ℝ := 0.12

/-- The final sale price after both discounts -/
def final_price : ℝ := 396.88

/-- Theorem stating that the original price of sarees is approximately 550,
    given the final price after two successive discounts -/
theorem saree_price_theorem :
  ∃ ε > 0, abs (original_price - (final_price / ((1 - discount1) * (1 - discount2)))) < ε :=
sorry

end NUMINAMATH_CALUDE_saree_price_theorem_l1993_199315


namespace NUMINAMATH_CALUDE_senate_committee_seating_arrangements_l1993_199391

def circular_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

theorem senate_committee_seating_arrangements :
  circular_arrangements 10 = 362880 := by
  sorry

end NUMINAMATH_CALUDE_senate_committee_seating_arrangements_l1993_199391


namespace NUMINAMATH_CALUDE_length_of_AB_prime_l1993_199346

/-- Given points A, B, and C in the plane, with A' and B' on the line y = x,
    and lines AA' and BB' intersecting at C, prove that the length of A'B' is 120√2/11 -/
theorem length_of_AB_prime (A B C A' B' : ℝ × ℝ) : 
  A = (0, 5) →
  B = (0, 15) →
  C = (3, 7) →
  A'.1 = A'.2 →
  B'.1 = B'.2 →
  (∃ t : ℝ, A' = (1 - t) • A + t • C) →
  (∃ s : ℝ, B' = (1 - s) • B + s • C) →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 120 * Real.sqrt 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AB_prime_l1993_199346


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1993_199318

/-- Quadratic function y = x^2 - 2tx + 3 -/
def f (t x : ℝ) : ℝ := x^2 - 2*t*x + 3

theorem quadratic_function_properties (t : ℝ) (h_t : t > 0) :
  (f t 2 = 1 → t = 3/2) ∧
  (∃ (x_min : ℝ), x_min ∈ Set.Icc 0 3 ∧ 
    (∀ x, x ∈ Set.Icc 0 3 → f t x ≥ f t x_min) ∧ 
    f t x_min = -2 → t = Real.sqrt 5) ∧
  (∀ (m a b : ℝ), 
    f t (m-2) = a ∧ f t 4 = b ∧ f t m = a ∧ a < b ∧ b < 3 → 
    (3 < m ∧ m < 4) ∨ m > 6) := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1993_199318


namespace NUMINAMATH_CALUDE_max_value_a_plus_b_l1993_199322

/-- Given that -1/4 * x^2 ≤ ax + b ≤ e^x for all x ∈ ℝ, the maximum value of a + b is 2 -/
theorem max_value_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, -1/4 * x^2 ≤ a * x + b ∧ a * x + b ≤ Real.exp x) → 
  (∀ c d : ℝ, (∀ x : ℝ, -1/4 * x^2 ≤ c * x + d ∧ c * x + d ≤ Real.exp x) → a + b ≥ c + d) →
  a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_a_plus_b_l1993_199322


namespace NUMINAMATH_CALUDE_kozlov_inequality_l1993_199341

theorem kozlov_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_kozlov_inequality_l1993_199341


namespace NUMINAMATH_CALUDE_proposition_and_converse_l1993_199366

theorem proposition_and_converse : 
  (∀ a b : ℝ, a + b ≥ 2 → (a ≥ 1 ∨ b ≥ 1)) ∧ 
  ¬(∀ a b : ℝ, (a ≥ 1 ∨ b ≥ 1) → a + b ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_proposition_and_converse_l1993_199366


namespace NUMINAMATH_CALUDE_triangle_angle_difference_l1993_199338

-- Define the triangle XYZ
structure Triangle where
  X : Real
  Y : Real
  Z : Real

-- Define the theorem
theorem triangle_angle_difference (t : Triangle) (h1 : t.Y = 2 * t.X) (h2 : t.X = 30) 
  (Z₁ Z₂ : Real) (h3 : Z₁ + Z₂ = t.Z) : Z₁ - Z₂ = 30 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_difference_l1993_199338


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l1993_199365

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 92 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 92 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l1993_199365


namespace NUMINAMATH_CALUDE_cannot_form_right_triangle_l1993_199393

theorem cannot_form_right_triangle : ¬ (9^2 + 16^2 = 25^2) := by
  sorry

end NUMINAMATH_CALUDE_cannot_form_right_triangle_l1993_199393


namespace NUMINAMATH_CALUDE_crackers_duration_l1993_199312

theorem crackers_duration (crackers_per_sandwich : ℕ) (sandwiches_per_night : ℕ)
  (sleeves_per_box : ℕ) (crackers_per_sleeve : ℕ) (num_boxes : ℕ)
  (h1 : crackers_per_sandwich = 2)
  (h2 : sandwiches_per_night = 5)
  (h3 : sleeves_per_box = 4)
  (h4 : crackers_per_sleeve = 28)
  (h5 : num_boxes = 5) :
  (num_boxes * sleeves_per_box * crackers_per_sleeve) / (crackers_per_sandwich * sandwiches_per_night) = 56 := by
  sorry

end NUMINAMATH_CALUDE_crackers_duration_l1993_199312


namespace NUMINAMATH_CALUDE_smallest_sum_A_plus_b_l1993_199328

theorem smallest_sum_A_plus_b : 
  ∀ (A : ℕ) (b : ℕ),
    A < 4 →
    A > 0 →
    b > 5 →
    21 * A = 3 * b + 3 →
    ∀ (A' : ℕ) (b' : ℕ),
      A' < 4 →
      A' > 0 →
      b' > 5 →
      21 * A' = 3 * b' + 3 →
      A + b ≤ A' + b' :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_A_plus_b_l1993_199328


namespace NUMINAMATH_CALUDE_alex_singing_probability_alex_singing_probability_proof_l1993_199360

theorem alex_singing_probability (p_sat : ℝ) 
  (h1 : ℝ → Prop) (h2 : ℝ → Prop) (h3 : ℝ → Prop) : Prop :=
  (h1 p_sat → (1 - p_sat) * 0.7 = 0.5) →
  (h2 p_sat → p_sat * 0 + (1 - p_sat) * 0.7 = 0.5) →
  (h3 p_sat → p_sat = 2 / 7) →
  p_sat = 2 / 7

-- The proof is omitted
theorem alex_singing_probability_proof : 
  ∃ (p_sat : ℝ) (h1 h2 h3 : ℝ → Prop), 
  alex_singing_probability p_sat h1 h2 h3 := by
  sorry

end NUMINAMATH_CALUDE_alex_singing_probability_alex_singing_probability_proof_l1993_199360


namespace NUMINAMATH_CALUDE_fraction_simplification_l1993_199347

theorem fraction_simplification : (3/7 + 5/8) / (5/12 + 2/9) = 531/322 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1993_199347


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_6_not_4_l1993_199362

theorem smallest_two_digit_multiple_of_6_not_4 :
  ∃ n : ℕ, 
    n ≥ 10 ∧ n < 100 ∧  -- two-digit positive integer
    n % 6 = 0 ∧         -- multiple of 6
    n % 4 ≠ 0 ∧         -- not a multiple of 4
    (∀ m : ℕ, m ≥ 10 ∧ m < 100 ∧ m % 6 = 0 ∧ m % 4 ≠ 0 → n ≤ m) ∧  -- smallest such number
    n = 18 :=           -- the number is 18
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_6_not_4_l1993_199362


namespace NUMINAMATH_CALUDE_integer_part_of_sqrt18_minus_2_l1993_199344

theorem integer_part_of_sqrt18_minus_2 :
  ⌊Real.sqrt 18 - 2⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_integer_part_of_sqrt18_minus_2_l1993_199344


namespace NUMINAMATH_CALUDE_exists_shorter_representation_l1993_199374

def repeatedSevens (n : ℕ) : ℕ := 
  (7 * (10^n - 1)) / 9

def validExpression (expr : ℕ → ℕ) : Prop :=
  ∃ (k : ℕ), expr k = repeatedSevens k ∧ 
  (∀ m : ℕ, m ≤ k → expr m ≠ repeatedSevens m)

theorem exists_shorter_representation : 
  ∃ (n : ℕ) (expr : ℕ → ℕ), n > 2 ∧ validExpression expr ∧ 
  (∀ k : ℕ, k ≥ n → expr k < repeatedSevens k) :=
sorry

end NUMINAMATH_CALUDE_exists_shorter_representation_l1993_199374


namespace NUMINAMATH_CALUDE_priya_speed_calculation_l1993_199331

/-- Priya's speed in km/h -/
def priya_speed : ℝ := 30

/-- Riya's speed in km/h -/
def riya_speed : ℝ := 20

/-- Time traveled in hours -/
def time : ℝ := 0.5

/-- Distance between Riya and Priya after traveling -/
def distance : ℝ := 25

theorem priya_speed_calculation :
  (riya_speed + priya_speed) * time = distance :=
by sorry

end NUMINAMATH_CALUDE_priya_speed_calculation_l1993_199331


namespace NUMINAMATH_CALUDE_problem_solution_l1993_199352

def f (x : ℝ) : ℝ := |x - 1|

theorem problem_solution :
  (∃ (m : ℝ), m > 0 ∧
    (∀ x, f (x + 5) ≤ 3 * m ↔ -7 ≤ x ∧ x ≤ -1)) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 2 * a^2 + b^2 = 3 →
    2 * a * Real.sqrt (1 + b^2) ≤ 2 * Real.sqrt 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a^2 + b^2 = 3 ∧
    2 * a * Real.sqrt (1 + b^2) = 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1993_199352


namespace NUMINAMATH_CALUDE_range_of_b_l1993_199302

def A : Set ℝ := {x | Real.log (x + 2) / Real.log (1/2) < 0}
def B (a b : ℝ) : Set ℝ := {x | (x - a) * (x - b) < 0}

theorem range_of_b (a : ℝ) (h : a = -3) :
  (∀ b : ℝ, (A ∩ B a b).Nonempty) → ∀ b : ℝ, b > -1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_l1993_199302


namespace NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l1993_199375

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 3

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements in the arrangement -/
def total_elements : ℕ := num_ones + num_zeros

/-- The probability that two zeros are not adjacent when randomly arranged with three ones -/
def prob_zeros_not_adjacent : ℚ := 3/5

theorem zeros_not_adjacent_probability :
  prob_zeros_not_adjacent = 3/5 := by sorry

end NUMINAMATH_CALUDE_zeros_not_adjacent_probability_l1993_199375


namespace NUMINAMATH_CALUDE_range_of_m_when_p_or_q_false_l1993_199304

theorem range_of_m_when_p_or_q_false (m : ℝ) : 
  (¬(∃ x : ℝ, m * x^2 + 1 ≤ 0) ∧ ¬(∀ x : ℝ, x^2 + m * x + 1 > 0)) → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_when_p_or_q_false_l1993_199304


namespace NUMINAMATH_CALUDE_average_score_theorem_l1993_199323

/-- The average score of a class given the proportions of students scoring different points -/
theorem average_score_theorem (p3 p2 p1 p0 : ℝ) 
  (h_p3 : p3 = 0.3) 
  (h_p2 : p2 = 0.5) 
  (h_p1 : p1 = 0.1) 
  (h_p0 : p0 = 0.1)
  (h_sum : p3 + p2 + p1 + p0 = 1) : 
  3 * p3 + 2 * p2 + 1 * p1 + 0 * p0 = 2 := by
  sorry

#check average_score_theorem

end NUMINAMATH_CALUDE_average_score_theorem_l1993_199323


namespace NUMINAMATH_CALUDE_parabola_focus_l1993_199395

/-- The parabola equation: y^2 + 4x = 0 -/
def parabola_eq (x y : ℝ) : Prop := y^2 + 4*x = 0

/-- The focus of a parabola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- The theorem stating that the focus of the parabola y^2 + 4x = 0 is at (-1, 0) -/
theorem parabola_focus :
  ∃ (f : Focus), (f.x = -1 ∧ f.y = 0) ∧
  ∀ (x y : ℝ), parabola_eq x y → 
    (y^2 = 4 * (f.x - x) ∧ f.y = 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l1993_199395


namespace NUMINAMATH_CALUDE_area_original_triangle_l1993_199355

/-- Given a triangle ABC and its oblique dimetric projection A''B''C'',
    where A''B''C'' is an equilateral triangle with side length a,
    prove that the area of ABC is (√6 * a^2) / 2. -/
theorem area_original_triangle (a : ℝ) (h : a > 0) :
  let s_projection := (Real.sqrt 3 * a^2) / 4
  let ratio := Real.sqrt 2 / 4
  s_projection / ratio = (Real.sqrt 6 * a^2) / 2 := by
sorry

end NUMINAMATH_CALUDE_area_original_triangle_l1993_199355


namespace NUMINAMATH_CALUDE_ellipse_iff_k_in_range_l1993_199351

/-- The equation of an ellipse in the form (x^2 / (3+k)) + (y^2 / (2-k)) = 1 -/
def is_ellipse (k : ℝ) : Prop :=
  (3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k)

/-- The range of k for which the equation represents an ellipse -/
def k_range : Set ℝ :=
  {k | k ∈ (Set.Ioo (-3) (-1/2) ∪ Set.Ioo (-1/2) 2)}

/-- Theorem stating that the equation represents an ellipse if and only if k is in the specified range -/
theorem ellipse_iff_k_in_range :
  ∀ k : ℝ, is_ellipse k ↔ k ∈ k_range :=
sorry

end NUMINAMATH_CALUDE_ellipse_iff_k_in_range_l1993_199351


namespace NUMINAMATH_CALUDE_ball_338_in_cup_360_l1993_199336

/-- The number of cups in the circle. -/
def n : ℕ := 1000

/-- The step size for placing balls. -/
def step : ℕ := 7

/-- The index of the ball we're interested in. -/
def ball_index : ℕ := 338

/-- Function to calculate the cup number for a given ball index. -/
def cup_number (k : ℕ) : ℕ :=
  (1 + step * (k - 1)) % n

theorem ball_338_in_cup_360 : cup_number ball_index = 360 := by
  sorry

end NUMINAMATH_CALUDE_ball_338_in_cup_360_l1993_199336


namespace NUMINAMATH_CALUDE_stating_largest_valid_n_l1993_199384

/-- 
Given a positive integer n, this function checks if n! can be expressed as 
the product of n - 4 consecutive positive integers.
-/
def is_valid (n : ℕ) : Prop :=
  ∃ (b : ℕ), b ≥ 4 ∧ n.factorial = ((n - 4 + b).factorial / b.factorial)

/-- 
Theorem stating that 119 is the largest positive integer n for which n! 
can be expressed as the product of n - 4 consecutive positive integers.
-/
theorem largest_valid_n : 
  (is_valid 119) ∧ (∀ m : ℕ, m > 119 → ¬(is_valid m)) :=
sorry

end NUMINAMATH_CALUDE_stating_largest_valid_n_l1993_199384


namespace NUMINAMATH_CALUDE_todds_initial_gum_l1993_199316

theorem todds_initial_gum (initial : ℕ) : 
  (∃ (after_steve after_emily : ℕ),
    after_steve = initial + 16 ∧
    after_emily = after_steve - 12 ∧
    after_emily = 54) →
  initial = 50 := by
sorry

end NUMINAMATH_CALUDE_todds_initial_gum_l1993_199316


namespace NUMINAMATH_CALUDE_angle_A_in_triangle_l1993_199348

-- Define the triangle ABC
structure Triangle where
  A : Real
  b : Real
  c : Real
  S : Real

-- State the theorem
theorem angle_A_in_triangle (abc : Triangle) (h1 : abc.b = 8) (h2 : abc.c = 8 * Real.sqrt 3) 
  (h3 : abc.S = 16 * Real.sqrt 3) : 
  abc.A = π / 6 ∨ abc.A = 5 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_in_triangle_l1993_199348


namespace NUMINAMATH_CALUDE_chris_babysitting_hours_l1993_199325

/-- The number of hours Chris worked babysitting -/
def hours_worked : ℕ := 9

/-- The cost of the video game in dollars -/
def video_game_cost : ℕ := 60

/-- The cost of the candy in dollars -/
def candy_cost : ℕ := 5

/-- Chris's hourly rate for babysitting in dollars -/
def hourly_rate : ℕ := 8

/-- The amount of money Chris had left over after purchases -/
def money_left : ℕ := 7

theorem chris_babysitting_hours :
  hours_worked * hourly_rate = video_game_cost + candy_cost + money_left :=
by sorry

end NUMINAMATH_CALUDE_chris_babysitting_hours_l1993_199325


namespace NUMINAMATH_CALUDE_flowers_per_bouquet_l1993_199337

theorem flowers_per_bouquet (total_flowers : ℕ) (wilted_flowers : ℕ) (num_bouquets : ℕ) :
  total_flowers = 88 →
  wilted_flowers = 48 →
  num_bouquets = 8 →
  (total_flowers - wilted_flowers) / num_bouquets = 5 :=
by sorry

end NUMINAMATH_CALUDE_flowers_per_bouquet_l1993_199337


namespace NUMINAMATH_CALUDE_subset_condition_l1993_199386

-- Define the sets A and B
def A : Set ℝ := {x | (x - 3) / (x - 4) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 5) > 0}

-- State the theorem
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ 4 ≤ a ∧ a < 5 := by sorry

end NUMINAMATH_CALUDE_subset_condition_l1993_199386


namespace NUMINAMATH_CALUDE_drunkard_theorem_l1993_199324

structure PubSystem where
  states : Finset (Fin 4)
  transition : Fin 4 → Fin 4 → ℚ
  start_state : Fin 4

def drunkard_walk (ps : PubSystem) (n : ℕ) : Fin 4 → ℚ :=
  sorry

theorem drunkard_theorem (ps : PubSystem) :
  (ps.states = {0, 1, 2, 3}) →
  (ps.transition 0 1 = 1/3) →
  (ps.transition 0 2 = 1/3) →
  (ps.transition 0 3 = 0) →
  (ps.transition 1 0 = 1/2) →
  (ps.transition 1 2 = 1/3) →
  (ps.transition 1 3 = 1/2) →
  (ps.transition 2 0 = 1/2) →
  (ps.transition 2 1 = 1/3) →
  (ps.transition 2 3 = 1/2) →
  (ps.transition 3 1 = 1/3) →
  (ps.transition 3 2 = 1/3) →
  (ps.transition 3 0 = 0) →
  (ps.start_state = 0) →
  ((drunkard_walk ps 5) 2 = 55/162) ∧
  (∀ n > 5, (drunkard_walk ps n) 1 > (drunkard_walk ps n) 0 ∧
            (drunkard_walk ps n) 2 > (drunkard_walk ps n) 0 ∧
            (drunkard_walk ps n) 1 > (drunkard_walk ps n) 3 ∧
            (drunkard_walk ps n) 2 > (drunkard_walk ps n) 3) :=
by sorry

end NUMINAMATH_CALUDE_drunkard_theorem_l1993_199324


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1993_199307

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 16} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1993_199307


namespace NUMINAMATH_CALUDE_limit_ratio_sevens_to_total_l1993_199390

/-- Count of digit 7 occurrences in decimal representation of numbers from 1 to n -/
def count_sevens (n : ℕ) : ℕ := sorry

/-- Total count of digits in decimal representation of numbers from 1 to n -/
def total_digits (n : ℕ) : ℕ := sorry

/-- The theorem stating that the limit of the ratio of 7's to total digits is 1/10 -/
theorem limit_ratio_sevens_to_total (ε : ℝ) (hε : ε > 0) : 
  ∃ N : ℕ, ∀ n ≥ N, |((count_sevens n : ℝ) / (total_digits n : ℝ)) - (1 / 10)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_ratio_sevens_to_total_l1993_199390


namespace NUMINAMATH_CALUDE_max_a_value_l1993_199306

/-- The function f(x) = x^2 + 2ax - 1 -/
def f (a : ℝ) (x : ℝ) := x^2 + 2*a*x - 1

/-- The theorem stating the maximum value of a -/
theorem max_a_value :
  ∀ a : ℝ,
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ici 1 → x₂ ∈ Set.Ici 1 → x₁ < x₂ →
    x₂ * (f a x₁) - x₁ * (f a x₂) < a * (x₁ - x₂)) ↔
  a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1993_199306


namespace NUMINAMATH_CALUDE_percentage_calculation_l1993_199308

theorem percentage_calculation (x : ℝ) (h : 0.4 * x = 160) : 0.5 * x = 200 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1993_199308


namespace NUMINAMATH_CALUDE_coefficient_of_monomial_l1993_199372

def monomial : ℚ × (ℕ × ℕ × ℕ) := (-2/9, (1, 4, 2))

theorem coefficient_of_monomial :
  (monomial.fst : ℚ) = -2/9 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_monomial_l1993_199372


namespace NUMINAMATH_CALUDE_three_quantities_problem_l1993_199310

theorem three_quantities_problem (x y z : ℕ) : 
  y = x + 8 →
  z = 3 * x →
  x + y + z = 108 →
  (x = 20 ∧ y = 28 ∧ z = 60) := by
  sorry

end NUMINAMATH_CALUDE_three_quantities_problem_l1993_199310


namespace NUMINAMATH_CALUDE_jerry_read_30_pages_saturday_l1993_199332

/-- The number of pages Jerry read on Saturday -/
def pages_read_saturday (total_pages : ℕ) (pages_read_sunday : ℕ) (pages_remaining : ℕ) : ℕ :=
  total_pages - (pages_remaining + pages_read_sunday)

theorem jerry_read_30_pages_saturday :
  pages_read_saturday 93 20 43 = 30 := by
  sorry

end NUMINAMATH_CALUDE_jerry_read_30_pages_saturday_l1993_199332


namespace NUMINAMATH_CALUDE_village_population_l1993_199381

theorem village_population (population : ℕ) : 
  (60 : ℕ) * population = 23040 * 100 → population = 38400 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1993_199381


namespace NUMINAMATH_CALUDE_meters_to_cm_conversion_l1993_199303

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

/-- Proves that 3.5 meters is equal to 350 centimeters -/
theorem meters_to_cm_conversion : 3.5 * meters_to_cm = 350 := by
  sorry

end NUMINAMATH_CALUDE_meters_to_cm_conversion_l1993_199303


namespace NUMINAMATH_CALUDE_range_of_m_l1993_199389

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 - 3*m*x + 9 ≥ 0) → m ∈ Set.Icc (-2) 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1993_199389


namespace NUMINAMATH_CALUDE_tan_equality_345_degrees_l1993_199354

theorem tan_equality_345_degrees (n : ℤ) :
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (345 * π / 180) → n = -15 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_345_degrees_l1993_199354


namespace NUMINAMATH_CALUDE_sequence_properties_l1993_199380

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

-- Define the geometric sequence
def geometric_sequence (b : ℕ → ℝ) := ∀ n m, b (n + m) = b n * b m

theorem sequence_properties 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_a_cond : 2 * a 5 - a 3 = 3)
  (h_b_2 : b 2 = 1)
  (h_b_4 : b 4 = 4) :
  (a 7 = 3) ∧ 
  ((b 3 = 2 ∨ b 3 = -2) ∧ b 6 = 16) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l1993_199380


namespace NUMINAMATH_CALUDE_counterexample_exists_l1993_199376

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1993_199376


namespace NUMINAMATH_CALUDE_complex_power_210_deg_60_l1993_199317

theorem complex_power_210_deg_60 :
  (Complex.exp (210 * π / 180 * I)) ^ 60 = -1/2 + Complex.I * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_210_deg_60_l1993_199317


namespace NUMINAMATH_CALUDE_intersection_point_property_l1993_199311

/-- The x-coordinate of the intersection point of y = 1/x and y = x + 2 -/
def a : ℝ := by sorry

/-- The y-coordinate of the intersection point of y = 1/x and y = x + 2 -/
def b : ℝ := by sorry

/-- The intersection point satisfies the equation of y = 1/x -/
axiom inverse_prop : b = 1 / a

/-- The intersection point satisfies the equation of y = x + 2 -/
axiom linear : b = a + 2

theorem intersection_point_property : a - a * b - b = -3 := by sorry

end NUMINAMATH_CALUDE_intersection_point_property_l1993_199311


namespace NUMINAMATH_CALUDE_special_function_unique_l1993_199309

/-- A function satisfying the given properties -/
def special_function (g : ℝ → ℝ) : Prop :=
  g 2 = 2 ∧ ∀ x y : ℝ, g (x * y + g x) = x * g y + g x

theorem special_function_unique (g : ℝ → ℝ) (h : special_function g) :
  ∀ x : ℝ, g x = 2 * x :=
sorry

end NUMINAMATH_CALUDE_special_function_unique_l1993_199309


namespace NUMINAMATH_CALUDE_cut_pyramid_volume_l1993_199320

/-- The volume of a smaller pyramid cut from a right square pyramid -/
theorem cut_pyramid_volume (base_edge original_height slant_edge cut_height : ℝ) : 
  base_edge = 12 * Real.sqrt 2 →
  slant_edge = 15 →
  original_height = Real.sqrt (slant_edge^2 - (base_edge/2)^2) →
  cut_height = 5 →
  cut_height < original_height →
  (1/3) * (base_edge * (original_height - cut_height) / original_height)^2 * (original_height - cut_height) = 2048/27 :=
by sorry

end NUMINAMATH_CALUDE_cut_pyramid_volume_l1993_199320


namespace NUMINAMATH_CALUDE_min_triangle_area_l1993_199378

/-- Triangle ABC with A at origin, B at (30, 18), and C with integer coordinates -/
structure Triangle :=
  (p : ℤ)
  (q : ℤ)

/-- Calculate the area of the triangle using the Shoelace formula -/
def triangleArea (t : Triangle) : ℚ :=
  (1 / 2 : ℚ) * |30 * t.q - 18 * t.p|

/-- Theorem: The minimum area of triangle ABC is 3 -/
theorem min_triangle_area :
  ∃ (t : Triangle), ∀ (t' : Triangle), triangleArea t ≤ triangleArea t' ∧ triangleArea t = 3 :=
sorry

end NUMINAMATH_CALUDE_min_triangle_area_l1993_199378


namespace NUMINAMATH_CALUDE_value_of_expression_l1993_199369

theorem value_of_expression (x : ℝ) (h : x = -2) : (3 * x + 4)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1993_199369


namespace NUMINAMATH_CALUDE_closest_to_half_at_seven_dips_l1993_199394

/-- The number of unit cubes --/
def num_cubes : ℕ := 1729

/-- The number of faces per cube --/
def faces_per_cube : ℕ := 6

/-- The total number of faces --/
def total_faces : ℕ := num_cubes * faces_per_cube

/-- The expected number of painted faces per dip --/
def painted_per_dip : ℚ := 978

/-- The recurrence relation for painted faces --/
def painted_faces (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | n+1 => painted_faces n * (1566 / 1729) + painted_per_dip

/-- The theorem to prove --/
theorem closest_to_half_at_seven_dips :
  ∀ k : ℕ, k ≠ 7 →
  |painted_faces 7 - (total_faces / 2)| < |painted_faces k - (total_faces / 2)| :=
sorry

end NUMINAMATH_CALUDE_closest_to_half_at_seven_dips_l1993_199394


namespace NUMINAMATH_CALUDE_min_value_theorem_l1993_199383

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 + 2*x)/(y - 2) + (y^2 + 2*y)/(x - 2) ≥ 22 ∧
  ((x^2 + 2*x)/(y - 2) + (y^2 + 2*y)/(x - 2) = 22 ↔ x = 3 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1993_199383


namespace NUMINAMATH_CALUDE_smallest_n_with_common_factor_l1993_199398

def has_common_factor_greater_than_one (a b : ℤ) : Prop :=
  ∃ (k : ℤ), k > 1 ∧ k ∣ a ∧ k ∣ b

theorem smallest_n_with_common_factor : 
  (∀ n : ℕ, n > 0 ∧ n < 19 → ¬(has_common_factor_greater_than_one (11*n - 3) (8*n + 2))) ∧ 
  (has_common_factor_greater_than_one (11*19 - 3) (8*19 + 2)) := by
  sorry

#check smallest_n_with_common_factor

end NUMINAMATH_CALUDE_smallest_n_with_common_factor_l1993_199398


namespace NUMINAMATH_CALUDE_sequence_general_term_l1993_199327

/-- Given sequences {a_n} and {b_n} with initial conditions and recurrence relations,
    prove the general term formula for {b_n}. -/
theorem sequence_general_term
  (p q r : ℝ)
  (h_q_pos : q > 0)
  (h_p_gt_r : p > r)
  (h_r_pos : r > 0)
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_a_init : a 1 = p)
  (h_b_init : b 1 = q)
  (h_a_rec : ∀ n : ℕ, n ≥ 2 → a n = p * a (n - 1))
  (h_b_rec : ∀ n : ℕ, n ≥ 2 → b n = q * a (n - 1) + r * b (n - 1)) :
  ∀ n : ℕ, n ≥ 1 → b n = (q * (p^n - r^n)) / (p - r) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1993_199327


namespace NUMINAMATH_CALUDE_m_range_l1993_199345

def p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x + y - m = 0 ∧ (x - 1)^2 + y^2 = 1

def q (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧
    m * x₁^2 - x₁ + m - 4 = 0 ∧
    m * x₂^2 - x₂ + m - 4 = 0

theorem m_range :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m) → Real.sqrt 2 + 1 ≤ m ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1993_199345


namespace NUMINAMATH_CALUDE_tan_150_degrees_l1993_199334

theorem tan_150_degrees : Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l1993_199334


namespace NUMINAMATH_CALUDE_carbon_processing_optimization_l1993_199367

-- Define the processing volume range
def ProcessingRange : Set ℝ := {x : ℝ | 300 ≤ x ∧ x ≤ 600}

-- Define the cost function
def CostFunction (x : ℝ) : ℝ := 0.5 * x^2 - 200 * x + 45000

-- Define the revenue function
def RevenueFunction (x : ℝ) : ℝ := 200 * x

-- Define the profit function
def ProfitFunction (x : ℝ) : ℝ := RevenueFunction x - CostFunction x

-- Theorem statement
theorem carbon_processing_optimization :
  ∃ (x_min : ℝ) (max_profit : ℝ),
    x_min ∈ ProcessingRange ∧
    (∀ x ∈ ProcessingRange, CostFunction x_min / x_min ≤ CostFunction x / x) ∧
    x_min = 300 ∧
    (∀ x ∈ ProcessingRange, ProfitFunction x > 0) ∧
    (∀ x ∈ ProcessingRange, ProfitFunction x ≤ max_profit) ∧
    max_profit = 35000 := by
  sorry

end NUMINAMATH_CALUDE_carbon_processing_optimization_l1993_199367


namespace NUMINAMATH_CALUDE_only_14_and_28_satisfy_l1993_199321

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (x y : ℕ), n = 10 * x + y ∧ y ≤ 9 ∧ n = 14 * x

theorem only_14_and_28_satisfy :
  ∀ n : ℕ, satisfies_condition n ↔ n = 14 ∨ n = 28 := by
  sorry

end NUMINAMATH_CALUDE_only_14_and_28_satisfy_l1993_199321


namespace NUMINAMATH_CALUDE_pentagonal_base_monochromatic_l1993_199339

-- Define the vertices of the prism
inductive Vertex : Type
| A : Fin 5 → Vertex
| B : Fin 5 → Vertex

-- Define the color of an edge
inductive Color : Type
| Red
| Blue

-- Define the edge coloring function
def edge_color : Vertex → Vertex → Color := sorry

-- No triangle has all edges of the same color
axiom no_monochromatic_triangle :
  ∀ (v1 v2 v3 : Vertex),
    v1 ≠ v2 → v2 ≠ v3 → v3 ≠ v1 →
    ¬(edge_color v1 v2 = edge_color v2 v3 ∧ edge_color v2 v3 = edge_color v3 v1)

-- Theorem: All edges of each pentagonal base are the same color
theorem pentagonal_base_monochromatic :
  (∀ (i j : Fin 5), edge_color (Vertex.A i) (Vertex.A j) = edge_color (Vertex.A 0) (Vertex.A 1)) ∧
  (∀ (i j : Fin 5), edge_color (Vertex.B i) (Vertex.B j) = edge_color (Vertex.B 0) (Vertex.B 1)) :=
sorry

end NUMINAMATH_CALUDE_pentagonal_base_monochromatic_l1993_199339


namespace NUMINAMATH_CALUDE_shaded_area_of_circumscribed_circles_shaded_area_equals_135π_l1993_199371

/-- The area of the shaded region between a circle circumscribing two externally tangent circles with radii 3 and 5 -/
theorem shaded_area_of_circumscribed_circles (π : ℝ) : ℝ := by
  -- Define the radii of the two smaller circles
  let r₁ : ℝ := 3
  let r₂ : ℝ := 5

  -- Define the radius of the larger circumscribing circle
  let R : ℝ := r₁ + r₂ + r₂

  -- Define the areas of the circles
  let A₁ : ℝ := π * r₁^2
  let A₂ : ℝ := π * r₂^2
  let A_large : ℝ := π * R^2

  -- Define the shaded area
  let shaded_area : ℝ := A_large - A₁ - A₂

  -- Prove that the shaded area equals 135π
  sorry

/-- The main theorem stating that the shaded area is equal to 135π -/
theorem shaded_area_equals_135π (π : ℝ) : shaded_area_of_circumscribed_circles π = 135 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_circumscribed_circles_shaded_area_equals_135π_l1993_199371


namespace NUMINAMATH_CALUDE_inequality_proof_l1993_199329

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_min : min (a + b) (min (b + c) (c + a)) > Real.sqrt 2)
  (h_sum : a^2 + b^2 + c^2 = 3) : 
  a / (b + c - a)^2 + b / (c + a - b)^2 + c / (a + b - c)^2 ≥ 3 / (a * b * c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1993_199329


namespace NUMINAMATH_CALUDE_clare_remaining_money_l1993_199357

/-- Given Clare's initial money and her purchases, calculate the remaining money. -/
def remaining_money (initial_money bread_price milk_price bread_quantity milk_quantity : ℕ) : ℕ :=
  initial_money - (bread_price * bread_quantity + milk_price * milk_quantity)

/-- Theorem: Clare has $35 left after her purchases. -/
theorem clare_remaining_money :
  remaining_money 47 2 2 4 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_clare_remaining_money_l1993_199357


namespace NUMINAMATH_CALUDE_geometry_textbook_weight_l1993_199356

theorem geometry_textbook_weight 
  (chemistry_weight : Real) 
  (weight_difference : Real) 
  (h1 : chemistry_weight = 7.12)
  (h2 : weight_difference = 6.5)
  (h3 : chemistry_weight = geometry_weight + weight_difference) :
  geometry_weight = 0.62 :=
by
  sorry

end NUMINAMATH_CALUDE_geometry_textbook_weight_l1993_199356


namespace NUMINAMATH_CALUDE_a_minus_b_equals_two_l1993_199392

theorem a_minus_b_equals_two (a b : ℝ) 
  (h1 : |a| = 1) 
  (h2 : |b - 1| = 2) 
  (h3 : a > b) : 
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_two_l1993_199392


namespace NUMINAMATH_CALUDE_cylinder_volume_in_sphere_l1993_199359

-- Define the sphere
def sphere_diameter : ℝ := 2

-- Define the cylinder
def cylinder_height : ℝ := 1

-- Theorem to prove
theorem cylinder_volume_in_sphere :
  let sphere_radius : ℝ := sphere_diameter / 2
  let base_radius : ℝ := sphere_radius
  let cylinder_volume : ℝ := Real.pi * base_radius^2 * (cylinder_height / 2)
  cylinder_volume = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_in_sphere_l1993_199359


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_7_pow_1024_l1993_199397

/-- The sum of the tens digit and the units digit in the decimal representation of 7^1024 is 17. -/
theorem sum_of_last_two_digits_of_7_pow_1024 :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ (7^1024 : ℕ) % 100 = 10 * a + b ∧ a + b = 17 := by
sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_7_pow_1024_l1993_199397


namespace NUMINAMATH_CALUDE_range_of_m_l1993_199385

theorem range_of_m (m : ℝ) : m ≥ 3 ↔ 
  (∀ x : ℝ, (|2*x + 1| ≤ 3 → x^2 - 2*x + 1 - m^2 ≤ 0) ∧ 
  (∃ x : ℝ, |2*x + 1| > 3 ∧ x^2 - 2*x + 1 - m^2 > 0)) ∧ 
  m > 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1993_199385


namespace NUMINAMATH_CALUDE_function_properties_l1993_199388

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - 2*a*x^2 + b*x

-- Define the derivative of f(x)
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 - 4*a*x + b

theorem function_properties :
  ∃ (a b : ℝ),
    -- Condition: f(1) = 3
    f a b 1 = 3 ∧
    -- Condition: f'(1) = 1 (slope of tangent line at x=1)
    f_derivative a b 1 = 1 ∧
    -- Prove: a = 2 and b = 6
    a = 2 ∧ b = 6 ∧
    -- Prove: Range of f(x) on [-1, 4] is [-11, 24]
    (∀ x, -1 ≤ x ∧ x ≤ 4 → -11 ≤ f a b x ∧ f a b x ≤ 24) ∧
    f a b (-1) = -11 ∧ f a b 4 = 24 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1993_199388


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l1993_199349

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 48 → (∀ x y : ℕ, x * y = 48 → x + y ≤ heart + club) → heart + club = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l1993_199349


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l1993_199333

/-- The common ratio of the infinite geometric series 7/8 - 14/32 + 56/256 - ... is -1/2 -/
theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -14/32
  let a₃ : ℚ := 56/256
  let r : ℚ := a₂ / a₁
  (r = -1/2) ∧ (a₃ / a₂ = r) := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l1993_199333


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l1993_199300

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Represents the ball placement process -/
def ballPlacement (step : ℕ) : ℕ :=
  sorry

theorem ball_placement_theorem (step : ℕ) :
  step = 1024 →
  ballPlacement step = sumDigits (toBase7 step) :=
sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l1993_199300


namespace NUMINAMATH_CALUDE_product_of_sums_l1993_199319

theorem product_of_sums (a b c d : ℚ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : (a + c) * (a + d) = 1)
  (h2 : (b + c) * (b + d) = 1) :
  (a + c) * (b + c) = -1 := by
sorry

end NUMINAMATH_CALUDE_product_of_sums_l1993_199319


namespace NUMINAMATH_CALUDE_square_difference_formula_l1993_199373

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 9/17) (h2 : x - y = 1/19) : x^2 - y^2 = 9/323 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l1993_199373


namespace NUMINAMATH_CALUDE_universal_set_determination_l1993_199382

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 5}
def complementA : Set Nat := {2, 4, 6}

theorem universal_set_determination :
  (A ⊆ U) ∧ (complementA ⊆ U) ∧ (A ∪ complementA = U) ∧ (A ∩ complementA = ∅) →
  U = {1, 2, 3, 4, 5, 6} :=
by sorry

end NUMINAMATH_CALUDE_universal_set_determination_l1993_199382


namespace NUMINAMATH_CALUDE_integer_solutions_inequality_l1993_199335

theorem integer_solutions_inequality :
  ∀ x y z : ℤ,
  x^2 * y^2 + y^2 * z^2 + x^2 + z^2 - 38*(x*y + z) - 40*(y*z + x) + 4*x*y*z + 761 ≤ 0 →
  ((x = 6 ∧ y = 2 ∧ z = 7) ∨ (x = 20 ∧ y = 0 ∧ z = 19)) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_inequality_l1993_199335
