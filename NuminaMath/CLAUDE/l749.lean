import Mathlib

namespace NUMINAMATH_CALUDE_h_function_iff_increasing_or_constant_l749_74963

/-- Definition of an "H function" -/
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ ≥ x₁ * f x₂ + x₂ * f x₁

/-- A function is increasing -/
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- A function is constant -/
def is_constant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = f y

theorem h_function_iff_increasing_or_constant (f : ℝ → ℝ) :
  is_h_function f ↔ is_increasing f ∨ is_constant f :=
sorry

end NUMINAMATH_CALUDE_h_function_iff_increasing_or_constant_l749_74963


namespace NUMINAMATH_CALUDE_quadratic_touch_existence_l749_74954

theorem quadratic_touch_existence (p q : ℤ) (h : p^2 = 4*q) :
  ∃ (a b : ℤ), b = a^2 + p*a + q ∧ a^2 = 4*b :=
sorry

end NUMINAMATH_CALUDE_quadratic_touch_existence_l749_74954


namespace NUMINAMATH_CALUDE_ninth_minus_eighth_rectangle_tiles_l749_74942

/-- The number of tiles in the nth rectangle of the sequence -/
def tiles (n : ℕ) : ℕ := 2 * n * n

/-- The difference in tiles between the 9th and 8th rectangles -/
def tile_difference : ℕ := tiles 9 - tiles 8

theorem ninth_minus_eighth_rectangle_tiles : tile_difference = 34 := by
  sorry

end NUMINAMATH_CALUDE_ninth_minus_eighth_rectangle_tiles_l749_74942


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_16_l749_74971

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  let rec aux (m : ℕ) (acc : ℕ) : ℕ :=
    if m = 0 then acc
    else aux (m / 10) (acc + m % 10)
  aux n 0

-- Define the property for a number to be a four-digit number
def isFourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

-- Theorem statement
theorem largest_four_digit_sum_16 :
  ∀ n : ℕ, isFourDigitNumber n → sumOfDigits n = 16 → n ≤ 9700 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_16_l749_74971


namespace NUMINAMATH_CALUDE_perpendicular_from_point_to_line_l749_74978

-- Define the plane
variable (Plane : Type)

-- Define points and lines
variable (Point : Plane → Type)
variable (Line : Plane → Type)

-- Define the relation of a point being on a line
variable (on_line : ∀ {p : Plane}, Point p → Line p → Prop)

-- Define perpendicularity of lines
variable (perpendicular : ∀ {p : Plane}, Line p → Line p → Prop)

-- Define the operation of drawing a line through two points
variable (line_through : ∀ {p : Plane}, Point p → Point p → Line p)

-- Define the operation of erecting a perpendicular to a line at a point
variable (erect_perpendicular : ∀ {p : Plane}, Line p → Point p → Line p)

-- Theorem statement
theorem perpendicular_from_point_to_line 
  {p : Plane} (A : Point p) (L : Line p) 
  (h : ¬ on_line A L) : 
  ∃ (M : Line p), perpendicular M L ∧ on_line A M := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_from_point_to_line_l749_74978


namespace NUMINAMATH_CALUDE_sequence_divisibility_l749_74981

theorem sequence_divisibility (k : ℕ+) 
  (a : ℕ → ℤ)
  (h : ∀ n : ℕ, n ≥ 1 → a n = (a (n - 1) + n^(k : ℕ)) / n) :
  3 ∣ (k : ℤ) - 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_divisibility_l749_74981


namespace NUMINAMATH_CALUDE_prob_three_two_digit_out_of_five_l749_74943

/-- A 12-sided die with numbers from 1 to 12 -/
def TwelveSidedDie : Type := Fin 12

/-- The probability of rolling a two-digit number on a single 12-sided die -/
def prob_two_digit : ℚ := 1 / 4

/-- The probability of rolling a one-digit number on a single 12-sided die -/
def prob_one_digit : ℚ := 3 / 4

/-- The number of 12-sided dice rolled -/
def num_dice : ℕ := 5

/-- The number of dice required to show a two-digit number -/
def required_two_digit : ℕ := 3

/-- Theorem stating the probability of exactly 3 out of 5 12-sided dice showing a two-digit number -/
theorem prob_three_two_digit_out_of_five :
  (Nat.choose num_dice required_two_digit : ℚ) *
  (prob_two_digit ^ required_two_digit) *
  (prob_one_digit ^ (num_dice - required_two_digit)) = 45 / 512 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_two_digit_out_of_five_l749_74943


namespace NUMINAMATH_CALUDE_vector_properties_l749_74967

/-- Given vectors in R², prove properties about their relationships -/
theorem vector_properties (a b : ℝ) :
  let m : Fin 2 → ℝ := ![a, b^2 - b + 7/3]
  let n : Fin 2 → ℝ := ![a + b + 2, 1]
  let μ : Fin 2 → ℝ := ![2, 1]
  (∃ (k : ℝ), m = k • μ) →
  (∃ (a_min : ℝ), a_min = 25/6 ∧ ∀ (a' : ℝ), (∃ (k : ℝ), ![a', b^2 - b + 7/3] = k • μ) → a' ≥ a_min) ∧
  (m • n ≥ 0) := by
sorry


end NUMINAMATH_CALUDE_vector_properties_l749_74967


namespace NUMINAMATH_CALUDE_garden_area_l749_74953

theorem garden_area (width length : ℝ) (h1 : length = 3 * width + 30) 
  (h2 : 2 * (length + width) = 800) : width * length = 28443.75 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l749_74953


namespace NUMINAMATH_CALUDE_ln_x_over_x_decreasing_l749_74925

theorem ln_x_over_x_decreasing (a b c : ℝ) : 
  a = (Real.log 3) / 3 → 
  b = (Real.log 5) / 5 → 
  c = (Real.log 6) / 6 → 
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ln_x_over_x_decreasing_l749_74925


namespace NUMINAMATH_CALUDE_nabla_calculation_l749_74912

def nabla (a b : ℕ) : ℕ := 3 + a^b

theorem nabla_calculation : nabla (nabla 2 3) 2 = 124 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l749_74912


namespace NUMINAMATH_CALUDE_f_properties_l749_74964

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := -1/a + 2/x

theorem f_properties :
  (∀ x > 0, ∀ y > 0, x < y → f a x > f a y) ∧
  (a < 0 → ∀ x > 0, f a x > 0) ∧
  (a > 0 → ∀ x ∈ Set.Ioo 0 (2*a), f a x > 0) ∧
  (a > 0 → ∀ x > 2*a, f a x ≤ 0) ∧
  (a < 0 ∨ a ≥ 1/4 ↔ ∀ x > 0, f a x + 2*x ≥ 0) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l749_74964


namespace NUMINAMATH_CALUDE_gcd_840_1764_l749_74987

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l749_74987


namespace NUMINAMATH_CALUDE_nth_term_is_3012_l749_74937

def arithmetic_sequence (a₁ a₂ a₃ : ℚ) : ℕ → ℚ :=
  λ n => a₁ + (n - 1) * (a₂ - a₁)

theorem nth_term_is_3012 (x : ℚ) :
  let a₁ := 3 * x - 4
  let a₂ := 7 * x - 14
  let a₃ := 4 * x + 6
  (∃ n : ℕ, arithmetic_sequence a₁ a₂ a₃ n = 3012) →
  (∃ n : ℕ, n = 392 ∧ arithmetic_sequence a₁ a₂ a₃ n = 3012) :=
by
  sorry

#check nth_term_is_3012

end NUMINAMATH_CALUDE_nth_term_is_3012_l749_74937


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l749_74947

theorem sqrt_product_equality : (Real.sqrt 8 + Real.sqrt 3) * Real.sqrt 6 = 4 * Real.sqrt 3 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l749_74947


namespace NUMINAMATH_CALUDE_cookie_cost_l749_74973

def total_spent : ℕ := 53
def candy_cost : ℕ := 14

theorem cookie_cost : total_spent - candy_cost = 39 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cost_l749_74973


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l749_74913

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_products_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 648 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l749_74913


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_change_l749_74931

theorem rectangular_prism_volume_change (V l w h : ℝ) (h1 : V = l * w * h) :
  2 * l * (3 * w) * (h / 4) = 1.5 * V := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_change_l749_74931


namespace NUMINAMATH_CALUDE_prime_sequence_divisibility_l749_74970

theorem prime_sequence_divisibility (p d : ℕ+) 
  (h1 : Nat.Prime p)
  (h2 : Nat.Prime (p + d))
  (h3 : Nat.Prime (p + 2*d))
  (h4 : Nat.Prime (p + 3*d))
  (h5 : Nat.Prime (p + 4*d))
  (h6 : Nat.Prime (p + 5*d)) :
  2 ∣ d ∧ 3 ∣ d ∧ 5 ∣ d := by
  sorry

end NUMINAMATH_CALUDE_prime_sequence_divisibility_l749_74970


namespace NUMINAMATH_CALUDE_student_arrangement_equality_l749_74926

theorem student_arrangement_equality (n : ℕ) : 
  n = 48 → 
  (Nat.factorial n) = (Nat.factorial n) :=
by
  sorry

end NUMINAMATH_CALUDE_student_arrangement_equality_l749_74926


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l749_74986

theorem complex_modulus_problem (z : ℂ) (h : (8 + 6*I)*z = 5 + 12*I) : 
  Complex.abs z = 13/10 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l749_74986


namespace NUMINAMATH_CALUDE_complex_symmetry_ratio_imag_part_l749_74961

theorem complex_symmetry_ratio_imag_part (z₁ z₂ : ℂ) :
  z₁ = 1 - 2*I →
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) →
  (z₂ / z₁).im = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_symmetry_ratio_imag_part_l749_74961


namespace NUMINAMATH_CALUDE_find_other_number_l749_74910

theorem find_other_number (a b : ℤ) (h1 : 3 * a + 2 * b = 120) (h2 : a = 26 ∨ b = 26) : 
  (a ≠ 26 → a = 21) ∧ (b ≠ 26 → b = 21) :=
sorry

end NUMINAMATH_CALUDE_find_other_number_l749_74910


namespace NUMINAMATH_CALUDE_greatest_number_with_odd_factors_under_200_l749_74928

theorem greatest_number_with_odd_factors_under_200 : 
  ∃ n : ℕ, n = 196 ∧ 
  n < 200 ∧ 
  (∃ k : ℕ, n = k^2) ∧ 
  (∀ m : ℕ, m < 200 → (∃ j : ℕ, m = j^2) → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_with_odd_factors_under_200_l749_74928


namespace NUMINAMATH_CALUDE_max_consecutive_sum_of_5_to_7_l749_74933

theorem max_consecutive_sum_of_5_to_7 :
  ∀ p : ℕ+, 
    (∃ a : ℕ+, (Finset.range p).sum (λ i => a + i) = 5^7) →
    p ≤ 125 :=
by sorry

end NUMINAMATH_CALUDE_max_consecutive_sum_of_5_to_7_l749_74933


namespace NUMINAMATH_CALUDE_tims_car_initial_price_l749_74984

/-- The initial price of a car, given its depreciation rate and value after a certain time -/
def initial_price (depreciation_rate : ℕ → ℚ) (years : ℕ) (final_value : ℚ) : ℚ :=
  final_value + (years : ℚ) * depreciation_rate years

/-- Theorem: The initial price of Tim's car is $20,000 -/
theorem tims_car_initial_price :
  let depreciation_rate : ℕ → ℚ := λ _ => 1000
  let years : ℕ := 6
  let final_value : ℚ := 14000
  initial_price depreciation_rate years final_value = 20000 := by
  sorry

end NUMINAMATH_CALUDE_tims_car_initial_price_l749_74984


namespace NUMINAMATH_CALUDE_pentagonal_to_triangular_prism_l749_74909

/-- The number of cans in a pentagonal pyramid with l layers -/
def T (l : ℕ) : ℕ := l * (3 * l^2 - l) / 2

/-- A triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem pentagonal_to_triangular_prism (l : ℕ) (h : l ≥ 2) :
  ∃ n : ℕ, T l = l * triangular_number n :=
by
  sorry

end NUMINAMATH_CALUDE_pentagonal_to_triangular_prism_l749_74909


namespace NUMINAMATH_CALUDE_vector_dot_product_cosine_l749_74983

theorem vector_dot_product_cosine (x : ℝ) : 
  let a : ℝ × ℝ := (Real.cos x, Real.sin x)
  let b : ℝ × ℝ := (Real.sqrt 2, Real.sqrt 2)
  (a.1 * b.1 + a.2 * b.2 = 8/5) → Real.cos (x - π/4) = 4/5 := by
sorry

end NUMINAMATH_CALUDE_vector_dot_product_cosine_l749_74983


namespace NUMINAMATH_CALUDE_division_remainder_problem_l749_74989

theorem division_remainder_problem (a b : ℕ) (h1 : a - b = 1000) (h2 : ∃ q r, a = b * q + r ∧ q = 10) (h3 : a = 1100) : 
  ∃ r, a = b * 10 + r ∧ r = 100 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l749_74989


namespace NUMINAMATH_CALUDE_value_of_b_l749_74908

theorem value_of_b (a b c : ℝ) 
  (h1 : a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1))
  (h2 : 6 * b * 2 = 4) : 
  b = 15 := by
sorry

end NUMINAMATH_CALUDE_value_of_b_l749_74908


namespace NUMINAMATH_CALUDE_offices_assignment_equals_factorial4_l749_74979

/-- The number of ways to assign 4 distinct offices to 4 distinct people -/
def assignOffices : ℕ := 24

/-- The factorial of 4 -/
def factorial4 : ℕ := 4 * 3 * 2 * 1

/-- Proof that the number of ways to assign 4 distinct offices to 4 distinct people
    is equal to 4 factorial -/
theorem offices_assignment_equals_factorial4 : assignOffices = factorial4 := by
  sorry

end NUMINAMATH_CALUDE_offices_assignment_equals_factorial4_l749_74979


namespace NUMINAMATH_CALUDE_f_f_one_eq_one_l749_74959

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x - 1 else -x^2 - 2*x

theorem f_f_one_eq_one : f (f 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_f_one_eq_one_l749_74959


namespace NUMINAMATH_CALUDE_carpet_price_not_152_l749_74948

/-- Represents the price of a flying carpet over time -/
structure CarpetPrice where
  /-- The initial price of the carpet in dinars -/
  initial : ℕ
  /-- The number of years the price increases -/
  years : ℕ
  /-- The year in which the price triples (1-indexed) -/
  tripleYear : ℕ

/-- Calculates the final price of the carpet given the initial conditions -/
def finalPrice (c : CarpetPrice) : ℕ :=
  let priceBeforeTriple := c.initial + c.tripleYear - 1
  let priceAfterTriple := 3 * priceBeforeTriple
  priceAfterTriple + (c.years - c.tripleYear)

/-- Theorem stating that the final price cannot be 152 dinars given the conditions -/
theorem carpet_price_not_152 (c : CarpetPrice) 
  (h1 : c.initial = 1)
  (h2 : c.years = 99)
  (h3 : c.tripleYear > 0)
  (h4 : c.tripleYear ≤ c.years) :
  finalPrice c ≠ 152 := by
  sorry

#eval finalPrice { initial := 1, years := 99, tripleYear := 27 }
#eval finalPrice { initial := 1, years := 99, tripleYear := 26 }

end NUMINAMATH_CALUDE_carpet_price_not_152_l749_74948


namespace NUMINAMATH_CALUDE_balloon_count_l749_74968

-- Define the number of balloons for each person
def alyssa_balloons : ℕ := 37
def sandy_balloons : ℕ := 28
def sally_balloons : ℕ := 39

-- Define the total number of balloons
def total_balloons : ℕ := alyssa_balloons + sandy_balloons + sally_balloons

-- Theorem to prove
theorem balloon_count : total_balloons = 104 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l749_74968


namespace NUMINAMATH_CALUDE_f_minimum_value_l749_74952

noncomputable def f (x : ℝ) : ℝ := |2 * Real.sqrt x * (Real.log (2 * x) / Real.log (Real.sqrt 2))|

theorem f_minimum_value :
  (∀ x > 0, f x ≥ 0) ∧ (∃ x > 0, f x = 0) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_l749_74952


namespace NUMINAMATH_CALUDE_angle_theorem_l749_74988

theorem angle_theorem (α β θ : Real) 
  (h1 : 0 < α ∧ α < 60)
  (h2 : 0 < β ∧ β < 60)
  (h3 : 0 < θ ∧ θ < 60)
  (h4 : α + β = 2 * θ)
  (h5 : Real.sin α * Real.sin β * Real.sin θ = 
        Real.sin (60 - α) * Real.sin (60 - β) * Real.sin (60 - θ)) :
  θ = 30 := by sorry

end NUMINAMATH_CALUDE_angle_theorem_l749_74988


namespace NUMINAMATH_CALUDE_cathy_doughnuts_l749_74940

/-- Prove that Cathy bought 3 dozen doughnuts given the conditions of the problem -/
theorem cathy_doughnuts : 
  ∀ (samuel_dozens cathy_dozens : ℕ),
  samuel_dozens = 2 →
  (samuel_dozens * 12 + cathy_dozens * 12 = (8 + 2) * 6) →
  cathy_dozens = 3 := by sorry

end NUMINAMATH_CALUDE_cathy_doughnuts_l749_74940


namespace NUMINAMATH_CALUDE_intersection_circle_line_l749_74956

/-- Given a line and a circle that intersect at two points, prove that the radius of the circle
    has a specific value when the line from the origin to one intersection point is perpendicular
    to the line from the origin to the other intersection point. -/
theorem intersection_circle_line (r : ℝ) (A B : ℝ × ℝ) : r > 0 →
  (3 * A.1 - 4 * A.2 - 1 = 0) →
  (3 * B.1 - 4 * B.2 - 1 = 0) →
  (A.1^2 + A.2^2 = r^2) →
  (B.1^2 + B.2^2 = r^2) →
  (A.1 * B.1 + A.2 * B.2 = 0) →
  r = Real.sqrt 2 / 5 := by
  sorry

#check intersection_circle_line

end NUMINAMATH_CALUDE_intersection_circle_line_l749_74956


namespace NUMINAMATH_CALUDE_point_B_in_fourth_quadrant_l749_74939

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The specific point we're considering -/
def point_B : Point2D :=
  { x := 3, y := -7 }

/-- Theorem stating that point_B is in the fourth quadrant -/
theorem point_B_in_fourth_quadrant : in_fourth_quadrant point_B := by
  sorry

end NUMINAMATH_CALUDE_point_B_in_fourth_quadrant_l749_74939


namespace NUMINAMATH_CALUDE_smallest_difference_of_valid_units_digits_l749_74997

def is_multiple_of_five (n : ℕ) : Prop := ∃ k, n = 5 * k

def valid_units_digit (x : ℕ) : Prop :=
  x < 10 ∧ is_multiple_of_five (520 + x)

theorem smallest_difference_of_valid_units_digits :
  ∃ (a b : ℕ), valid_units_digit a ∧ valid_units_digit b ∧
  (∀ (c d : ℕ), valid_units_digit c → valid_units_digit d →
    a - b ≤ c - d ∨ b - a ≤ c - d) ∧
  a - b = 5 ∨ b - a = 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_of_valid_units_digits_l749_74997


namespace NUMINAMATH_CALUDE_triangle_problem_l749_74903

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < C ∧ C < π / 2 →
  a * Real.sin A = b * Real.sin B * Real.sin C →
  b = Real.sqrt 2 * a →
  C = π / 6 ∧ c^2 / a^2 = 3 - Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l749_74903


namespace NUMINAMATH_CALUDE_squirrel_acorns_l749_74955

theorem squirrel_acorns (num_squirrels : ℕ) (total_acorns : ℕ) (acorns_needed : ℕ) 
  (h1 : num_squirrels = 20)
  (h2 : total_acorns = 4500)
  (h3 : acorns_needed = 300) :
  acorns_needed - (total_acorns / num_squirrels) = 75 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l749_74955


namespace NUMINAMATH_CALUDE_fractional_exponent_simplification_l749_74958

theorem fractional_exponent_simplification :
  (2^2024 + 2^2020) / (2^2024 - 2^2020) = 17 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fractional_exponent_simplification_l749_74958


namespace NUMINAMATH_CALUDE_order_of_products_and_square_l749_74906

theorem order_of_products_and_square (x a b : ℝ) 
  (h1 : x < a) (h2 : a < b) (h3 : b < 0) : 
  b * x > a * x ∧ a * x > a ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_order_of_products_and_square_l749_74906


namespace NUMINAMATH_CALUDE_first_group_size_l749_74974

/-- Represents the work done by a group of workers --/
def work (persons : ℕ) (days : ℕ) (hours : ℕ) : ℕ := persons * days * hours

/-- Proves that the number of persons in the first group is 45 --/
theorem first_group_size :
  ∃ (P : ℕ), work P 12 5 = work 30 15 6 ∧ P = 45 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l749_74974


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l749_74916

/-- Systematic sampling function -/
def systematicSample (populationSize : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  let interval := populationSize / sampleSize
  List.range sampleSize |>.map (fun i => start + i * interval)

/-- Theorem: In a systematic sample of size 4 from 56 items, if 6, 20, and 48 are in the sample, then 34 is the fourth number -/
theorem systematic_sample_theorem :
  ∀ (sample : List ℕ),
    sample = systematicSample 56 4 6 →
    sample.length = 4 →
    6 ∈ sample →
    20 ∈ sample →
    48 ∈ sample →
    34 ∈ sample := by
  sorry

#eval systematicSample 56 4 6

end NUMINAMATH_CALUDE_systematic_sample_theorem_l749_74916


namespace NUMINAMATH_CALUDE_vincents_earnings_l749_74972

def fantasy_book_price : ℚ := 4
def literature_book_price : ℚ := fantasy_book_price / 2
def fantasy_books_sold_per_day : ℕ := 5
def literature_books_sold_per_day : ℕ := 8
def days : ℕ := 5

theorem vincents_earnings :
  (fantasy_book_price * fantasy_books_sold_per_day +
   literature_book_price * literature_books_sold_per_day) * days = 180 := by
  sorry

end NUMINAMATH_CALUDE_vincents_earnings_l749_74972


namespace NUMINAMATH_CALUDE_problem_statement_l749_74904

theorem problem_statement (h : 125 = 5^3) : (125 : ℝ)^(2/3) * 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l749_74904


namespace NUMINAMATH_CALUDE_invariant_preserved_not_all_blue_l749_74991

/-- Represents the state of chameleons on the island -/
structure ChameleonState where
  red : ℕ
  green : ℕ
  blue : ℕ

/-- The initial state of chameleons -/
def initial_state : ChameleonState :=
  { red := 25, green := 12, blue := 8 }

/-- Represents a single interaction between chameleons -/
inductive Interaction
  | SameColor : Interaction
  | DifferentColor : Interaction

/-- Applies an interaction to the current state -/
def apply_interaction (state : ChameleonState) (interaction : Interaction) : ChameleonState :=
  sorry

/-- The invariant that remains constant after each interaction -/
def invariant (state : ChameleonState) : ℕ :=
  (state.red - state.green) % 3

/-- Theorem stating that the invariant remains constant after any interaction -/
theorem invariant_preserved (state : ChameleonState) (interaction : Interaction) :
  invariant state = invariant (apply_interaction state interaction) :=
  sorry

/-- Theorem stating that it's impossible for all chameleons to be blue -/
theorem not_all_blue (state : ChameleonState) :
  (∃ n : ℕ, (state.red = 0 ∧ state.green = 0 ∧ state.blue = n)) →
  state ≠ initial_state ∧ 
  ¬∃ (interactions : List Interaction), 
    state = List.foldl apply_interaction initial_state interactions :=
  sorry

end NUMINAMATH_CALUDE_invariant_preserved_not_all_blue_l749_74991


namespace NUMINAMATH_CALUDE_hyperbola_b_value_l749_74901

/-- The value of b for a hyperbola with given equation and asymptote -/
theorem hyperbola_b_value (b : ℝ) (h1 : b > 0) : 
  (∀ x y : ℝ, x^2 / 4 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, 3*x - 2*y = 0 ∧ x^2 / 4 - y^2 / b^2 = 1) →
  b = 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_b_value_l749_74901


namespace NUMINAMATH_CALUDE_oliver_earnings_l749_74994

def laundry_shop_earnings (price_per_kilo : ℕ) (day1_kilos : ℕ) : ℕ :=
  let day2_kilos := day1_kilos + 5
  let day3_kilos := 2 * day2_kilos
  price_per_kilo * (day1_kilos + day2_kilos + day3_kilos)

theorem oliver_earnings :
  laundry_shop_earnings 2 5 = 70 :=
by sorry

end NUMINAMATH_CALUDE_oliver_earnings_l749_74994


namespace NUMINAMATH_CALUDE_quadratic_inequality_iff_abs_a_leq_two_l749_74930

theorem quadratic_inequality_iff_abs_a_leq_two (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ abs a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_iff_abs_a_leq_two_l749_74930


namespace NUMINAMATH_CALUDE_max_valid_arrangement_l749_74962

/-- A type representing the cards with numbers 1 to 9 -/
inductive Card : Type
  | one | two | three | four | five | six | seven | eight | nine

/-- A function that returns the numerical value of a card -/
def cardValue : Card → Nat
  | Card.one => 1
  | Card.two => 2
  | Card.three => 3
  | Card.four => 4
  | Card.five => 5
  | Card.six => 6
  | Card.seven => 7
  | Card.eight => 8
  | Card.nine => 9

/-- A predicate that checks if two cards satisfy the adjacency condition -/
def validAdjacent (c1 c2 : Card) : Prop :=
  (cardValue c1 ∣ cardValue c2) ∨ (cardValue c2 ∣ cardValue c1)

/-- A type representing a valid arrangement of cards -/
def ValidArrangement := List Card

/-- A predicate that checks if an arrangement is valid -/
def isValidArrangement : ValidArrangement → Prop
  | [] => True
  | [_] => True
  | (c1 :: c2 :: rest) => validAdjacent c1 c2 ∧ isValidArrangement (c2 :: rest)

/-- The main theorem stating that the maximum number of cards in a valid arrangement is 8 -/
theorem max_valid_arrangement :
  (∃ (arr : ValidArrangement), isValidArrangement arr ∧ arr.length = 8) ∧
  (∀ (arr : ValidArrangement), isValidArrangement arr → arr.length ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_max_valid_arrangement_l749_74962


namespace NUMINAMATH_CALUDE_solve_equation_l749_74998

theorem solve_equation : ∃ x : ℝ, 2*x + 3*x = 600 - (4*x + 6*x) ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l749_74998


namespace NUMINAMATH_CALUDE_metro_ticket_sales_l749_74995

/-- Proves that the average number of tickets sold per minute is 5,
    given the cost per ticket and total earnings over 6 minutes. -/
theorem metro_ticket_sales
  (ticket_cost : ℝ)
  (total_earnings : ℝ)
  (duration : ℕ)
  (h1 : ticket_cost = 3)
  (h2 : total_earnings = 90)
  (h3 : duration = 6) :
  total_earnings / (ticket_cost * duration) = 5 := by
  sorry

end NUMINAMATH_CALUDE_metro_ticket_sales_l749_74995


namespace NUMINAMATH_CALUDE_min_cookies_eaten_is_five_l749_74927

/-- Represents the number of cookies Paco had, ate, and bought -/
structure CookieCount where
  initial : ℕ
  eaten_first : ℕ
  bought : ℕ
  eaten_second : ℕ

/-- The conditions of the cookie problem -/
def cookie_problem (c : CookieCount) : Prop :=
  c.initial = 25 ∧
  c.bought = 3 ∧
  c.eaten_second = c.bought + 2

/-- The minimum number of cookies Paco ate -/
def min_cookies_eaten (c : CookieCount) : ℕ :=
  c.eaten_second

/-- Theorem stating that the minimum number of cookies Paco ate is 5 -/
theorem min_cookies_eaten_is_five :
  ∀ c : CookieCount, cookie_problem c → min_cookies_eaten c = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_min_cookies_eaten_is_five_l749_74927


namespace NUMINAMATH_CALUDE_cone_volume_l749_74993

/-- Given a cone whose lateral surface unfolds into a sector with radius 3 and central angle 2π/3,
    the volume of the cone is 2√2π/3 -/
theorem cone_volume (r l : ℝ) (h : ℝ) : 
  r > 0 → l > 0 → h > 0 →
  l = 3 →
  2 * π * r = 2 * π / 3 * l →
  h^2 + r^2 = l^2 →
  (1/3) * π * r^2 * h = (2 * Real.sqrt 2 / 3) * π := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l749_74993


namespace NUMINAMATH_CALUDE_student_failed_by_89_marks_l749_74990

def total_marks : ℕ := 800
def passing_percentage : ℚ := 33 / 100
def student_marks : ℕ := 175

theorem student_failed_by_89_marks :
  ⌈(passing_percentage * total_marks : ℚ)⌉ - student_marks = 89 :=
sorry

end NUMINAMATH_CALUDE_student_failed_by_89_marks_l749_74990


namespace NUMINAMATH_CALUDE_sum_reciprocal_equality_l749_74980

theorem sum_reciprocal_equality (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c ≠ 0) (h5 : (a + b) / (a * b) + 1 / c = 1 / (a + b + c)) :
  (∀ n : ℕ, 1 / a^n + 1 / b^n + 1 / c^n = 1 / (a^n + b^n + c^n)) := by
  sorry

#check sum_reciprocal_equality

end NUMINAMATH_CALUDE_sum_reciprocal_equality_l749_74980


namespace NUMINAMATH_CALUDE_shortest_player_height_l749_74951

theorem shortest_player_height 
  (tallest_height : ℝ) 
  (height_difference : ℝ) 
  (h1 : tallest_height = 77.75)
  (h2 : height_difference = 9.5) : 
  tallest_height - height_difference = 68.25 := by
sorry

end NUMINAMATH_CALUDE_shortest_player_height_l749_74951


namespace NUMINAMATH_CALUDE_prob_same_color_specific_l749_74950

/-- The probability of selecting two plates of the same color -/
def prob_same_color (red blue green : ℕ) : ℚ :=
  let total := red + blue + green
  let same_color := (red.choose 2) + (blue.choose 2) + (green.choose 2)
  same_color / total.choose 2

/-- Theorem: The probability of selecting two plates of the same color
    given 6 red, 5 blue, and 3 green plates is 28/91 -/
theorem prob_same_color_specific : prob_same_color 6 5 3 = 28 / 91 := by
  sorry

#eval prob_same_color 6 5 3

end NUMINAMATH_CALUDE_prob_same_color_specific_l749_74950


namespace NUMINAMATH_CALUDE_roots_of_equation_l749_74924

/-- The equation for which we need to find roots -/
def equation (x : ℝ) : Prop :=
  15 / (x^2 - 4) - 2 / (x - 2) = 1

/-- Theorem stating that -3 and 5 are the roots of the equation -/
theorem roots_of_equation :
  equation (-3) ∧ equation 5 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l749_74924


namespace NUMINAMATH_CALUDE_min_face_sum_is_16_l749_74919

/-- Represents the arrangement of numbers on a cube's vertices -/
def CubeArrangement := Fin 8 → Fin 8

/-- Check if a given arrangement satisfies the condition that the sum of any three vertices on a face is at least 10 -/
def ValidArrangement (arr : CubeArrangement) : Prop :=
  ∀ (face : Fin 6) (v1 v2 v3 : Fin 4), v1 ≠ v2 ∧ v1 ≠ v3 ∧ v2 ≠ v3 →
    (arr (face * 4 + v1) + arr (face * 4 + v2) + arr (face * 4 + v3) : ℕ) ≥ 10

/-- Calculate the sum of numbers on a given face -/
def FaceSum (arr : CubeArrangement) (face : Fin 6) : ℕ :=
  (arr (face * 4) : ℕ) + (arr (face * 4 + 1) : ℕ) + (arr (face * 4 + 2) : ℕ) + (arr (face * 4 + 3) : ℕ)

/-- The main theorem stating that the minimal possible sum on any face is 16 -/
theorem min_face_sum_is_16 :
  ∃ (arr : CubeArrangement), ValidArrangement arr ∧
    (∀ (arr' : CubeArrangement), ValidArrangement arr' →
      ∀ (face : Fin 6), FaceSum arr face ≤ FaceSum arr' face) ∧
    (∃ (face : Fin 6), FaceSum arr face = 16) :=
  sorry

end NUMINAMATH_CALUDE_min_face_sum_is_16_l749_74919


namespace NUMINAMATH_CALUDE_ramanujan_number_l749_74935

theorem ramanujan_number (hardy_number ramanujan_number : ℂ) : 
  hardy_number * ramanujan_number = 48 - 24 * I ∧ 
  hardy_number = 6 + I → 
  ramanujan_number = (312 - 432 * I) / 37 := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_number_l749_74935


namespace NUMINAMATH_CALUDE_circle_area_increase_l749_74996

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_increase_l749_74996


namespace NUMINAMATH_CALUDE_x_power_y_equals_243_l749_74922

theorem x_power_y_equals_243 (x y : ℝ) : 
  y = Real.sqrt (x - 3) + Real.sqrt (3 - x) + 5 → x^y = 243 := by sorry

end NUMINAMATH_CALUDE_x_power_y_equals_243_l749_74922


namespace NUMINAMATH_CALUDE_ram_ravi_selection_probability_l749_74917

theorem ram_ravi_selection_probability :
  let p_ram : ℝ := 5/7
  let p_both : ℝ := 0.14285714285714288
  let p_ravi : ℝ := p_both / p_ram
  p_ravi = 0.2 := by sorry

end NUMINAMATH_CALUDE_ram_ravi_selection_probability_l749_74917


namespace NUMINAMATH_CALUDE_candy_theorem_l749_74934

def candy_problem (corey_candies tapanga_candies total_candies : ℕ) : Prop :=
  (tapanga_candies = corey_candies + 8) ∧
  (corey_candies = 29) ∧
  (total_candies = corey_candies + tapanga_candies)

theorem candy_theorem : ∃ (corey_candies tapanga_candies total_candies : ℕ),
  candy_problem corey_candies tapanga_candies total_candies ∧ total_candies = 66 := by
  sorry

end NUMINAMATH_CALUDE_candy_theorem_l749_74934


namespace NUMINAMATH_CALUDE_new_lift_count_correct_l749_74929

/-- The number of times Terrell must lift the new weight configuration to match the total weight of the original configuration -/
def new_lift_count : ℕ := 12

/-- The weight of each item in the original configuration -/
def original_weight : ℕ := 12

/-- The number of weights in the original configuration -/
def original_count : ℕ := 3

/-- The number of times Terrell lifts the original configuration -/
def original_lifts : ℕ := 20

/-- The weights in the new configuration -/
def new_weights : List ℕ := [18, 18, 24]

theorem new_lift_count_correct :
  new_lift_count * (new_weights.sum) = original_weight * original_count * original_lifts :=
by sorry

end NUMINAMATH_CALUDE_new_lift_count_correct_l749_74929


namespace NUMINAMATH_CALUDE_total_minutes_worked_l749_74945

/-- Calculates the total minutes worked by three people given specific conditions -/
theorem total_minutes_worked (bianca_hours : ℝ) : 
  bianca_hours = 12.5 → 
  (3 * bianca_hours + bianca_hours - 8.5) * 60 = 3240 := by
  sorry

#check total_minutes_worked

end NUMINAMATH_CALUDE_total_minutes_worked_l749_74945


namespace NUMINAMATH_CALUDE_rod_and_rope_problem_l749_74999

/-- 
Given a rod and a rope with the following properties:
1. The rope is 5 feet longer than the rod
2. When the rope is folded in half, it is 5 feet shorter than the rod

Prove that the system of equations x = y + 5 and 1/2 * x = y - 5 holds true,
where x is the length of the rope in feet and y is the length of the rod in feet.
-/
theorem rod_and_rope_problem (x y : ℝ) 
  (h1 : x = y + 5)
  (h2 : x / 2 = y - 5) : 
  x = y + 5 ∧ x / 2 = y - 5 := by
  sorry

end NUMINAMATH_CALUDE_rod_and_rope_problem_l749_74999


namespace NUMINAMATH_CALUDE_exponential_logarithmic_sum_implies_cosine_sum_l749_74992

theorem exponential_logarithmic_sum_implies_cosine_sum :
  ∃ (x y z : ℝ),
    (Real.exp x + Real.exp y + Real.exp z = 3) ∧
    (Real.log (1 + x^2) + Real.log (1 + y^2) + Real.log (1 + z^2) = 3) ∧
    (Real.cos (2*x) + Real.cos (2*y) + Real.cos (2*z) = 3) := by
  sorry

end NUMINAMATH_CALUDE_exponential_logarithmic_sum_implies_cosine_sum_l749_74992


namespace NUMINAMATH_CALUDE_total_fish_count_l749_74946

/-- The number of fish owned by Billy, Tony, Sarah, and Bobby -/
def fish_count (billy tony sarah bobby : ℕ) : Prop :=
  (tony = 3 * billy) ∧
  (sarah = tony + 5) ∧
  (bobby = 2 * sarah) ∧
  (billy = 10)

/-- The total number of fish owned by all four people -/
def total_fish (billy tony sarah bobby : ℕ) : ℕ :=
  billy + tony + sarah + bobby

/-- Theorem stating that the total number of fish is 145 -/
theorem total_fish_count :
  ∀ billy tony sarah bobby : ℕ,
  fish_count billy tony sarah bobby →
  total_fish billy tony sarah bobby = 145 :=
by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l749_74946


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l749_74944

theorem modulus_of_complex_number (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l749_74944


namespace NUMINAMATH_CALUDE_only_parallelogram_centrally_symmetric_l749_74960

-- Define the shapes
inductive Shape
  | EquilateralTriangle
  | Parallelogram
  | RegularPentagon
  | RightTriangle

-- Define central symmetry
def is_centrally_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => True
  | _ => False

-- Theorem statement
theorem only_parallelogram_centrally_symmetric :
  ∀ s : Shape, is_centrally_symmetric s ↔ s = Shape.Parallelogram :=
by
  sorry

end NUMINAMATH_CALUDE_only_parallelogram_centrally_symmetric_l749_74960


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_l749_74938

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_l749_74938


namespace NUMINAMATH_CALUDE_truncated_pyramid_edge_count_l749_74957

/-- A square-based pyramid with truncated vertices -/
structure TruncatedPyramid where
  /-- The number of vertices in the original square-based pyramid -/
  original_vertices : Nat
  /-- The number of edges in the original square-based pyramid -/
  original_edges : Nat
  /-- The number of new edges created by each truncation -/
  new_edges_per_truncation : Nat
  /-- Assertion that the original shape is a square-based pyramid -/
  is_square_based_pyramid : original_vertices = 5 ∧ original_edges = 8
  /-- Assertion that each truncation creates a triangular face -/
  truncation_creates_triangle : new_edges_per_truncation = 3

/-- Theorem stating that a truncated square-based pyramid has 23 edges -/
theorem truncated_pyramid_edge_count (p : TruncatedPyramid) :
  p.original_edges + p.original_vertices * p.new_edges_per_truncation = 23 :=
by sorry

end NUMINAMATH_CALUDE_truncated_pyramid_edge_count_l749_74957


namespace NUMINAMATH_CALUDE_negation_equivalence_l749_74911

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ∈ Set.Ici (0 : ℝ) ∧ x^3 + x < 0) ↔
  (∀ x : ℝ, x ∈ Set.Ici (0 : ℝ) → x^3 + x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l749_74911


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_l749_74969

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x^2 - 2*x ≥ 0}
def Q : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_P_intersect_Q : (P.compl ∩ Q) = Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_l749_74969


namespace NUMINAMATH_CALUDE_equation_solution_l749_74918

theorem equation_solution (y : ℚ) : 
  (1 : ℚ) / 3 + 1 / y = 7 / 9 → y = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l749_74918


namespace NUMINAMATH_CALUDE_dvd_sales_l749_74985

theorem dvd_sales (dvd cd : ℕ) : 
  dvd = (1.6 : ℝ) * cd →
  dvd + cd = 273 →
  dvd = 168 := by
sorry

end NUMINAMATH_CALUDE_dvd_sales_l749_74985


namespace NUMINAMATH_CALUDE_ace_then_diamond_probability_l749_74907

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : ℕ := 52

/-- Number of Aces in a standard deck -/
def NumberOfAces : ℕ := 4

/-- Number of diamonds in a standard deck -/
def NumberOfDiamonds : ℕ := 13

/-- Probability of drawing an Ace as the first card and a diamond as the second card -/
def ProbabilityAceThenDiamond : ℚ := 1 / StandardDeck

theorem ace_then_diamond_probability :
  ProbabilityAceThenDiamond = 1 / StandardDeck := by
  sorry

end NUMINAMATH_CALUDE_ace_then_diamond_probability_l749_74907


namespace NUMINAMATH_CALUDE_no_natural_solution_for_equation_l749_74932

theorem no_natural_solution_for_equation : ∀ m n : ℕ, m^2 ≠ n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solution_for_equation_l749_74932


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l749_74965

theorem termite_ridden_not_collapsing 
  (total_homes : ℕ) 
  (termite_ridden : ℕ) 
  (collapsing : ℕ) 
  (h1 : termite_ridden = total_homes / 3)
  (h2 : collapsing = (termite_ridden * 4) / 7) :
  (termite_ridden - collapsing : ℚ) / total_homes = 3 / 21 :=
by sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l749_74965


namespace NUMINAMATH_CALUDE_exists_number_with_sum_and_count_of_factors_l749_74936

open Nat

def sumOfDivisors (n : ℕ) : ℕ := sorry

def numberOfDivisors (n : ℕ) : ℕ := sorry

theorem exists_number_with_sum_and_count_of_factors :
  ∃ n : ℕ, n > 0 ∧ sumOfDivisors n + numberOfDivisors n = 1767 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_sum_and_count_of_factors_l749_74936


namespace NUMINAMATH_CALUDE_first_floor_units_count_l749_74923

/-- A building with a specified number of floors and apartments -/
structure Building where
  floors : ℕ
  firstFloorUnits : ℕ
  otherFloorUnits : ℕ

/-- The total number of apartment units in a building -/
def totalUnits (b : Building) : ℕ :=
  b.firstFloorUnits + (b.floors - 1) * b.otherFloorUnits

theorem first_floor_units_count (b1 b2 : Building) :
  b1 = b2 ∧ 
  b1.floors = 4 ∧ 
  b1.otherFloorUnits = 5 ∧ 
  totalUnits b1 + totalUnits b2 = 34 →
  b1.firstFloorUnits = 2 :=
sorry

end NUMINAMATH_CALUDE_first_floor_units_count_l749_74923


namespace NUMINAMATH_CALUDE_least_value_x_minus_y_minus_z_l749_74976

theorem least_value_x_minus_y_minus_z (x y z : ℕ+) 
  (h1 : x = 4 * y) (h2 : y = 7 * z) : 
  (x - y - z : ℤ) ≥ 19 ∧ ∃ (x₀ y₀ z₀ : ℕ+), 
    x₀ = 4 * y₀ ∧ y₀ = 7 * z₀ ∧ (x₀ - y₀ - z₀ : ℤ) = 19 := by
  sorry

end NUMINAMATH_CALUDE_least_value_x_minus_y_minus_z_l749_74976


namespace NUMINAMATH_CALUDE_inequality_solution_l749_74966

theorem inequality_solution (x : ℝ) :
  x ≠ 3 →
  (x * (x + 2) / (x - 3)^2 ≥ 8 ↔ x ∈ Set.Iic (18/7) ∪ Set.Ioi 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l749_74966


namespace NUMINAMATH_CALUDE_markus_family_ages_l749_74900

/-- Given a family where:
  * Markus is twice the age of his son
  * Markus's son is twice the age of Markus's grandson
  * Markus's grandson is three times the age of Markus's great-grandson
  * The sum of their ages is 140 years
Prove that Markus's great-grandson's age is 140/22 years. -/
theorem markus_family_ages (markus son grandson great_grandson : ℚ)
  (h1 : markus = 2 * son)
  (h2 : son = 2 * grandson)
  (h3 : grandson = 3 * great_grandson)
  (h4 : markus + son + grandson + great_grandson = 140) :
  great_grandson = 140 / 22 := by
  sorry

end NUMINAMATH_CALUDE_markus_family_ages_l749_74900


namespace NUMINAMATH_CALUDE_maria_car_trip_l749_74914

theorem maria_car_trip (total_distance : ℝ) (first_stop_fraction : ℝ) (second_stop_fraction : ℝ) :
  total_distance = 560 ∧ 
  first_stop_fraction = 1/2 ∧ 
  second_stop_fraction = 1/4 →
  total_distance - (first_stop_fraction * total_distance) - 
    (second_stop_fraction * (total_distance - first_stop_fraction * total_distance)) = 210 := by
  sorry

end NUMINAMATH_CALUDE_maria_car_trip_l749_74914


namespace NUMINAMATH_CALUDE_solution_range_l749_74975

-- Define the equation
def equation (x a : ℝ) : Prop :=
  1 / (x - 2) + (a - 2) / (2 - x) = 1

-- Define the solution function
def solution (a : ℝ) : ℝ := 5 - a

-- Theorem statement
theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ equation x a) ↔ (a < 5 ∧ a ≠ 3) :=
sorry

end NUMINAMATH_CALUDE_solution_range_l749_74975


namespace NUMINAMATH_CALUDE_base_conversion_1623_to_base7_l749_74982

/-- Converts a base-7 number to base 10 --/
def base7ToBase10 (a b c d : Nat) : Nat :=
  a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

/-- Theorem: 1623 in base 10 is equal to 4506 in base 7 --/
theorem base_conversion_1623_to_base7 : 
  1623 = base7ToBase10 4 5 0 6 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1623_to_base7_l749_74982


namespace NUMINAMATH_CALUDE_quadratic_function_range_l749_74915

theorem quadratic_function_range (a b : ℝ) :
  let f := fun x => a * x^2 + b * x
  (1 ≤ f (-1) ∧ f (-1) ≤ 2) →
  (2 ≤ f 1 ∧ f 1 ≤ 4) →
  (5 ≤ f (-2) ∧ f (-2) ≤ 10) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l749_74915


namespace NUMINAMATH_CALUDE_gear_q_revolutions_per_minute_l749_74977

/-- The number of revolutions per minute for gear p -/
def p_rev_per_min : ℚ := 10

/-- The number of seconds in the given time interval -/
def time_interval : ℚ := 10

/-- The additional revolutions gear q makes compared to gear p in the given time interval -/
def additional_rev : ℚ := 5

/-- The number of seconds in a minute -/
def seconds_per_minute : ℚ := 60

theorem gear_q_revolutions_per_minute :
  let p_rev_in_interval := p_rev_per_min * time_interval / seconds_per_minute
  let q_rev_in_interval := p_rev_in_interval + additional_rev
  let q_rev_per_min := q_rev_in_interval * seconds_per_minute / time_interval
  q_rev_per_min = 40 := by
  sorry

end NUMINAMATH_CALUDE_gear_q_revolutions_per_minute_l749_74977


namespace NUMINAMATH_CALUDE_correct_total_distance_l749_74941

/-- Converts kilometers to meters -/
def km_to_m (km : ℝ) : ℝ := km * 1000

/-- Calculates the total distance in meters -/
def total_distance (initial_km : ℝ) (additional_m : ℝ) : ℝ :=
  km_to_m initial_km + additional_m

/-- Theorem: The correct total distance is 3700 meters -/
theorem correct_total_distance :
  total_distance 3.5 200 = 3700 := by sorry

end NUMINAMATH_CALUDE_correct_total_distance_l749_74941


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l749_74949

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of parallel lines -/
def parallel (l₁ l₂ : Line) : Prop :=
  l₁.a * l₂.b = l₂.a * l₁.b

/-- The problem statement -/
theorem parallel_lines_m_value (m : ℝ) :
  let l₁ : Line := ⟨m - 2, -1, 5⟩
  let l₂ : Line := ⟨m - 2, 3 - m, 2⟩
  parallel l₁ l₂ → m = 2 ∨ m = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_lines_m_value_l749_74949


namespace NUMINAMATH_CALUDE_trader_donations_l749_74921

theorem trader_donations (total_profit : ℝ) (goal_amount : ℝ) (above_goal : ℝ) : 
  total_profit = 960 → 
  goal_amount = 610 → 
  above_goal = 180 → 
  (goal_amount + above_goal) - (total_profit / 2) = 310 := by
sorry

end NUMINAMATH_CALUDE_trader_donations_l749_74921


namespace NUMINAMATH_CALUDE_only_rectangle_both_symmetric_l749_74920

-- Define the shape type
inductive Shape
  | EquilateralTriangle
  | Angle
  | Rectangle
  | Parallelogram

-- Define axisymmetry property
def isAxisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => true
  | Shape.Angle => true
  | Shape.Rectangle => true
  | Shape.Parallelogram => false

-- Define central symmetry property
def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.EquilateralTriangle => false
  | Shape.Angle => false
  | Shape.Rectangle => true
  | Shape.Parallelogram => true

-- Theorem stating that only Rectangle is both axisymmetric and centrally symmetric
theorem only_rectangle_both_symmetric :
  ∀ s : Shape, isAxisymmetric s ∧ isCentrallySymmetric s ↔ s = Shape.Rectangle :=
by sorry

end NUMINAMATH_CALUDE_only_rectangle_both_symmetric_l749_74920


namespace NUMINAMATH_CALUDE_smallest_area_special_square_l749_74905

/-- A square with two vertices on a line and two on a parabola -/
structure SpecialSquare where
  /-- The y-intercept of the line containing two vertices of the square -/
  k : ℝ
  /-- The side length of the square -/
  s : ℝ
  /-- Two vertices of the square lie on the line y = 3x - 5 -/
  line_constraint : ∃ (x₁ x₂ : ℝ), y = 3 * x₁ - 5 ∧ y = 3 * x₂ - 5
  /-- Two vertices of the square lie on the parabola y = x^2 + 4 -/
  parabola_constraint : ∃ (x₁ x₂ : ℝ), y = x₁^2 + 4 ∧ y = x₂^2 + 4
  /-- The square's sides are parallel/perpendicular to coordinate axes -/
  axis_aligned : True
  /-- The area of the square is s^2 -/
  area_eq : s^2 = 10 * (25 + 4 * k)

/-- The theorem stating the smallest possible area of the special square -/
theorem smallest_area_special_square :
  ∀ (sq : SpecialSquare), sq.s^2 ≥ 200 :=
sorry

end NUMINAMATH_CALUDE_smallest_area_special_square_l749_74905


namespace NUMINAMATH_CALUDE_unique_train_journey_l749_74902

/-- Represents a day of the week -/
inductive DayOfWeek
| Saturday
| Sunday
| Monday

/-- Represents the train journey details -/
structure TrainJourney where
  carNumber : Nat
  seatNumber : Nat
  saturdayDate : Nat
  mondayDate : Nat

/-- Checks if the journey satisfies all given conditions -/
def isValidJourney (journey : TrainJourney) : Prop :=
  journey.seatNumber < journey.carNumber ∧
  journey.saturdayDate > journey.carNumber ∧
  journey.mondayDate = journey.carNumber

theorem unique_train_journey : 
  ∀ (journey : TrainJourney), 
    isValidJourney journey → 
    journey.carNumber = 2 ∧ journey.seatNumber = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_train_journey_l749_74902
