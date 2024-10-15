import Mathlib

namespace NUMINAMATH_CALUDE_inequality_equivalence_l1632_163263

theorem inequality_equivalence (x : ℝ) : 2 - x > 3 + x ↔ x < -1/2 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1632_163263


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_three_pi_eighths_l1632_163203

theorem cos_squared_minus_sin_squared_three_pi_eighths :
  Real.cos (3 * Real.pi / 8) ^ 2 - Real.sin (3 * Real.pi / 8) ^ 2 = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_three_pi_eighths_l1632_163203


namespace NUMINAMATH_CALUDE_at_least_one_acute_angle_leq_45_l1632_163254

-- Define a right triangle
structure RightTriangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real
  is_right_triangle : angle1 + angle2 + angle3 = 180
  has_right_angle : angle3 = 90

-- Theorem statement
theorem at_least_one_acute_angle_leq_45 (t : RightTriangle) :
  t.angle1 ≤ 45 ∨ t.angle2 ≤ 45 :=
sorry

end NUMINAMATH_CALUDE_at_least_one_acute_angle_leq_45_l1632_163254


namespace NUMINAMATH_CALUDE_identical_coordinate_point_exists_l1632_163204

/-- Represents a 2D rectangular coordinate system -/
structure CoordinateSystem :=
  (origin : ℝ × ℝ)
  (xAxis : ℝ × ℝ)
  (yAxis : ℝ × ℝ)
  (unitLength : ℝ)

/-- Theorem: Existence of a point with identical coordinates in two different coordinate systems -/
theorem identical_coordinate_point_exists 
  (cs1 cs2 : CoordinateSystem) 
  (h1 : cs1.origin ≠ cs2.origin) 
  (h2 : ¬ (∃ k : ℝ, cs1.xAxis = k • cs2.xAxis)) 
  (h3 : cs1.unitLength ≠ cs2.unitLength) : 
  ∃ p : ℝ × ℝ, ∃ x y : ℝ, 
    (x, y) = p ∧ 
    (∃ x' y' : ℝ, (x', y') = p ∧ x = x' ∧ y = y') :=
sorry

end NUMINAMATH_CALUDE_identical_coordinate_point_exists_l1632_163204


namespace NUMINAMATH_CALUDE_shaded_region_area_l1632_163207

/-- The number of congruent squares in the shaded region -/
def total_squares : ℕ := 20

/-- The number of shaded squares in the larger square -/
def squares_in_larger : ℕ := 4

/-- The length of the diagonal of the larger square in cm -/
def diagonal_length : ℝ := 10

/-- The area of the entire shaded region in square cm -/
def shaded_area : ℝ := 250

theorem shaded_region_area :
  ∀ (total_squares squares_in_larger : ℕ) (diagonal_length shaded_area : ℝ),
  total_squares = 20 →
  squares_in_larger = 4 →
  diagonal_length = 10 →
  shaded_area = total_squares * (diagonal_length / (2 * Real.sqrt 2))^2 →
  shaded_area = 250 := by
sorry

end NUMINAMATH_CALUDE_shaded_region_area_l1632_163207


namespace NUMINAMATH_CALUDE_ordered_pairs_satisfying_equation_l1632_163253

theorem ordered_pairs_satisfying_equation : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      let a := p.1
      let b := p.2
      a > 0 ∧ b > 0 ∧ 
      a * b + 83 = 24 * Nat.lcm a b + 17 * Nat.gcd a b)
    (Finset.product (Finset.range 1000) (Finset.range 1000))).card ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_ordered_pairs_satisfying_equation_l1632_163253


namespace NUMINAMATH_CALUDE_nicolai_ate_six_pounds_l1632_163291

/-- The amount of fruit eaten by Nicolai given the total fruit eaten and the amounts eaten by Mario and Lydia. -/
def nicolai_fruit (total_fruit : ℚ) (mario_fruit : ℚ) (lydia_fruit : ℚ) : ℚ :=
  total_fruit - (mario_fruit + lydia_fruit)

/-- Converts ounces to pounds -/
def ounces_to_pounds (ounces : ℚ) : ℚ :=
  ounces / 16

theorem nicolai_ate_six_pounds 
  (total_fruit : ℚ)
  (mario_ounces : ℚ)
  (lydia_ounces : ℚ)
  (h_total : total_fruit = 8)
  (h_mario : mario_ounces = 8)
  (h_lydia : lydia_ounces = 24) :
  nicolai_fruit total_fruit (ounces_to_pounds mario_ounces) (ounces_to_pounds lydia_ounces) = 6 := by
  sorry

#check nicolai_ate_six_pounds

end NUMINAMATH_CALUDE_nicolai_ate_six_pounds_l1632_163291


namespace NUMINAMATH_CALUDE_four_variable_inequality_l1632_163234

theorem four_variable_inequality (a b c d : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : 0 ≤ d) 
  (h5 : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d)) + (b^3 / (c + d + a)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_four_variable_inequality_l1632_163234


namespace NUMINAMATH_CALUDE_division_4512_by_32_l1632_163212

theorem division_4512_by_32 : ∃ (q r : ℕ), 4512 = 32 * q + r ∧ r < 32 ∧ q = 141 ∧ r = 0 := by
  sorry

end NUMINAMATH_CALUDE_division_4512_by_32_l1632_163212


namespace NUMINAMATH_CALUDE_max_value_abc_l1632_163249

theorem max_value_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) : 
  ∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z → x^2 + y^2 + z^2 = 1 → 
  a * b * Real.sqrt 3 + 2 * a * c ≤ Real.sqrt 3 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ a₀^2 + b₀^2 + c₀^2 = 1 ∧ 
  a₀ * b₀ * Real.sqrt 3 + 2 * a₀ * c₀ = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_abc_l1632_163249


namespace NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l1632_163213

def eggplant_pounds : ℝ := 5
def eggplant_price : ℝ := 2
def zucchini_pounds : ℝ := 4
def zucchini_price : ℝ := 2
def tomato_pounds : ℝ := 4
def tomato_price : ℝ := 3.5
def onion_pounds : ℝ := 3
def onion_price : ℝ := 1
def basil_pounds : ℝ := 1
def basil_price : ℝ := 2.5
def basil_unit : ℝ := 0.5
def yield_quarts : ℝ := 4

theorem ratatouille_cost_per_quart :
  let total_cost := eggplant_pounds * eggplant_price +
                    zucchini_pounds * zucchini_price +
                    tomato_pounds * tomato_price +
                    onion_pounds * onion_price +
                    basil_pounds / basil_unit * basil_price
  total_cost / yield_quarts = 10 := by sorry

end NUMINAMATH_CALUDE_ratatouille_cost_per_quart_l1632_163213


namespace NUMINAMATH_CALUDE_factor_expression_l1632_163248

theorem factor_expression (x : ℝ) : x * (x + 2) + (x + 2) = (x + 1) * (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1632_163248


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l1632_163289

theorem jacket_price_reduction (P : ℝ) (x : ℝ) : 
  P > 0 →
  (1 - x) * 0.75 * P * 1.5686274509803921 = P →
  x = 0.15 :=
by sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l1632_163289


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1632_163205

theorem inequality_solution_range (m : ℝ) : 
  (∀ x : ℕ+, (x = 1 ∨ x = 2 ∨ x = 3) ↔ 3 * (x : ℝ) - m ≤ 0) →
  (9 ≤ m ∧ m < 12) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1632_163205


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1632_163217

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + 2*I) = 5) : z.im = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1632_163217


namespace NUMINAMATH_CALUDE_max_value_of_five_integers_with_mean_eleven_l1632_163214

theorem max_value_of_five_integers_with_mean_eleven (a b c d e : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  (a + b + c + d + e : ℚ) / 5 = 11 →
  max a (max b (max c (max d e))) ≤ 45 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_five_integers_with_mean_eleven_l1632_163214


namespace NUMINAMATH_CALUDE_sum_of_digits_equals_five_l1632_163220

/-- S(n) is the sum of digits in the decimal representation of 2^n -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating that S(n) = 5 if and only if n = 5 -/
theorem sum_of_digits_equals_five (n : ℕ) : S n = 5 ↔ n = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_equals_five_l1632_163220


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1632_163225

theorem hyperbola_eccentricity (m : ℝ) :
  (∃ e : ℝ, e > Real.sqrt 2 ∧
    ∀ x y : ℝ, x^2 - y^2/m = 1 → e = Real.sqrt (1 + m)) ↔
  m > 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1632_163225


namespace NUMINAMATH_CALUDE_unique_n_with_divisor_sum_property_l1632_163237

theorem unique_n_with_divisor_sum_property (n : ℕ+) 
  (h1 : ∃ d₁ d₂ d₃ d₄ : ℕ+, d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ 
        (∀ m : ℕ+, m ∣ n → m ≥ d₁) ∧
        (∀ m : ℕ+, m ∣ n → m = d₁ ∨ m ≥ d₂) ∧
        (∀ m : ℕ+, m ∣ n → m = d₁ ∨ m = d₂ ∨ m ≥ d₃) ∧
        (∀ m : ℕ+, m ∣ n → m = d₁ ∨ m = d₂ ∨ m = d₃ ∨ m ≥ d₄))
  (h2 : ∃ d₁ d₂ d₃ d₄ : ℕ+, d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ 
        d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₄ ∣ n ∧
        d₁^2 + d₂^2 + d₃^2 + d₄^2 = n) :
  n = 130 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_with_divisor_sum_property_l1632_163237


namespace NUMINAMATH_CALUDE_calculation_proof_l1632_163294

theorem calculation_proof : (12 * 0.5 * 3 * 0.0625 - 1.5) = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1632_163294


namespace NUMINAMATH_CALUDE_volume_of_cube_with_triple_surface_area_l1632_163269

noncomputable def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

noncomputable def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length ^ 2

theorem volume_of_cube_with_triple_surface_area (v₁ : ℝ) (h₁ : v₁ = 64) :
  ∃ v₂ : ℝ, 
    (∃ s₁ s₂ : ℝ, 
      cube_volume s₁ = v₁ ∧ 
      cube_surface_area s₂ = 3 * cube_surface_area s₁ ∧ 
      cube_volume s₂ = v₂) ∧ 
    v₂ = 192 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_cube_with_triple_surface_area_l1632_163269


namespace NUMINAMATH_CALUDE_derivative_of_exp_plus_x_l1632_163283

open Real

theorem derivative_of_exp_plus_x (x : ℝ) :
  deriv (fun x => exp x + x) x = exp x + 1 := by sorry

end NUMINAMATH_CALUDE_derivative_of_exp_plus_x_l1632_163283


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1632_163298

theorem absolute_value_inequality (x : ℝ) :
  |x^2 - 5| < 9 ↔ -Real.sqrt 14 < x ∧ x < Real.sqrt 14 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1632_163298


namespace NUMINAMATH_CALUDE_perp_necessary_not_sufficient_l1632_163208

-- Define the plane α
variable (α : Plane)

-- Define lines l, m, and n
variable (l m n : Line)

-- Define the property that a line is in a plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- Define perpendicularity between a line and a plane
def line_perp_plane (l : Line) (p : Plane) : Prop := sorry

-- Define perpendicularity between two lines
def line_perp_line (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perp_necessary_not_sufficient :
  (line_in_plane m α ∧ line_in_plane n α) →
  (∀ l, line_perp_plane l α → (line_perp_line l m ∧ line_perp_line l n)) ∧
  (∃ l, line_perp_line l m ∧ line_perp_line l n ∧ ¬line_perp_plane l α) := by
  sorry

end NUMINAMATH_CALUDE_perp_necessary_not_sufficient_l1632_163208


namespace NUMINAMATH_CALUDE_evan_book_difference_l1632_163251

/-- Represents the number of books Evan owns at different points in time -/
structure EvanBooks where
  twoYearsAgo : ℕ
  current : ℕ
  inFiveYears : ℕ

/-- The conditions of Evan's book collection -/
def evanBookConditions (books : EvanBooks) : Prop :=
  books.twoYearsAgo = 200 ∧
  books.current = books.twoYearsAgo - 40 ∧
  books.inFiveYears = 860

/-- The theorem stating the difference between Evan's books in five years
    and five times his current number of books -/
theorem evan_book_difference (books : EvanBooks) 
  (h : evanBookConditions books) : 
  books.inFiveYears - (5 * books.current) = 60 := by
  sorry

end NUMINAMATH_CALUDE_evan_book_difference_l1632_163251


namespace NUMINAMATH_CALUDE_min_five_dollar_frisbees_l1632_163244

/-- Given a total of 115 frisbees sold for $450, with prices of $3, $4, and $5,
    the minimum number of $5 frisbees sold is 1. -/
theorem min_five_dollar_frisbees :
  ∀ (x y z : ℕ),
    x + y + z = 115 →
    3 * x + 4 * y + 5 * z = 450 →
    z ≥ 1 ∧
    ∀ (a b c : ℕ),
      a + b + c = 115 →
      3 * a + 4 * b + 5 * c = 450 →
      c ≥ z :=
by sorry

end NUMINAMATH_CALUDE_min_five_dollar_frisbees_l1632_163244


namespace NUMINAMATH_CALUDE_card_combination_proof_l1632_163290

theorem card_combination_proof : Nat.choose 60 13 = 7446680748480 := by
  sorry

end NUMINAMATH_CALUDE_card_combination_proof_l1632_163290


namespace NUMINAMATH_CALUDE_binomial_13_11_times_2_l1632_163282

theorem binomial_13_11_times_2 : 2 * Nat.choose 13 11 = 156 := by
  sorry

end NUMINAMATH_CALUDE_binomial_13_11_times_2_l1632_163282


namespace NUMINAMATH_CALUDE_arithmetic_operation_l1632_163286

theorem arithmetic_operation : 3 * 14 + 3 * 15 + 3 * 18 + 11 = 152 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_operation_l1632_163286


namespace NUMINAMATH_CALUDE_angle_triple_supplement_l1632_163279

theorem angle_triple_supplement (x : ℝ) : 
  (x = 3 * (180 - x)) → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_supplement_l1632_163279


namespace NUMINAMATH_CALUDE_jane_change_l1632_163261

/-- Calculates the change received after a purchase -/
def calculate_change (num_skirts : ℕ) (price_skirt : ℕ) (num_blouses : ℕ) (price_blouse : ℕ) (amount_paid : ℕ) : ℕ :=
  amount_paid - (num_skirts * price_skirt + num_blouses * price_blouse)

/-- Proves that Jane received $56 in change -/
theorem jane_change : calculate_change 2 13 3 6 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_jane_change_l1632_163261


namespace NUMINAMATH_CALUDE_min_value_of_f_l1632_163232

theorem min_value_of_f (a₁ a₂ a₃ a₄ : ℝ) (h : a₁ * a₄ - a₂ * a₃ = 1) : 
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ ∀ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ * x₄ - x₂ * x₃ = 1 → 
    x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₁*x₃ + x₂*x₄ ≥ m ∧
    ∃ (y₁ y₂ y₃ y₄ : ℝ), y₁ * y₄ - y₂ * y₃ = 1 ∧
      y₁^2 + y₂^2 + y₃^2 + y₄^2 + y₁*y₃ + y₂*y₄ = m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1632_163232


namespace NUMINAMATH_CALUDE_base_difference_theorem_l1632_163202

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

-- Define the given numbers
def num1 : List Nat := [3, 2, 7]
def base1 : Nat := 9
def num2 : List Nat := [2, 5, 3]
def base2 : Nat := 8

-- State the theorem
theorem base_difference_theorem : 
  to_base_10 num1 base1 - to_base_10 num2 base2 = 97 := by
  sorry

end NUMINAMATH_CALUDE_base_difference_theorem_l1632_163202


namespace NUMINAMATH_CALUDE_digit_1983_is_7_l1632_163281

/-- Represents the decimal number formed by concatenating numbers from 1 to 999 -/
def x : ℚ := sorry

/-- Returns the nth digit after the decimal point in x -/
def nthDigit (n : ℕ) : ℕ := sorry

theorem digit_1983_is_7 : nthDigit 1983 = 7 := by sorry

end NUMINAMATH_CALUDE_digit_1983_is_7_l1632_163281


namespace NUMINAMATH_CALUDE_circuit_board_count_l1632_163215

theorem circuit_board_count (T P : ℕ) : 
  (64 + P = T) →  -- Total boards = Failed + Passed
  (64 + P / 8 = 456) →  -- Total faulty boards
  T = 3200 := by
  sorry

end NUMINAMATH_CALUDE_circuit_board_count_l1632_163215


namespace NUMINAMATH_CALUDE_d_is_nonzero_l1632_163201

/-- A polynomial of degree 5 with six distinct x-intercepts, including 0 and -1 -/
def Q (a b c d : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x

/-- The property that Q has six distinct x-intercepts, including 0 and -1 -/
def has_six_distinct_intercepts (a b c d : ℝ) : Prop :=
  ∃ p q r s : ℝ, p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
              p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 ∧
              p ≠ -1 ∧ q ≠ -1 ∧ r ≠ -1 ∧ s ≠ -1 ∧
              ∀ x : ℝ, Q a b c d x = 0 ↔ x = 0 ∨ x = -1 ∨ x = p ∨ x = q ∨ x = r ∨ x = s

theorem d_is_nonzero (a b c d : ℝ) (h : has_six_distinct_intercepts a b c d) : d ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_d_is_nonzero_l1632_163201


namespace NUMINAMATH_CALUDE_salary_increase_proof_l1632_163257

theorem salary_increase_proof (S : ℝ) (P : ℝ) : 
  S > 0 →
  0.06 * S > 0 →
  0.10 * (S * (1 + P / 100)) = 1.8333333333333331 * (0.06 * S) →
  P = 10 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l1632_163257


namespace NUMINAMATH_CALUDE_car_distance_proof_l1632_163228

/-- Proves that the distance covered by a car is 540 kilometers under given conditions -/
theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) :
  initial_time = 8 →
  speed = 45 →
  (3 / 2 : ℝ) * initial_time * speed = 540 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_proof_l1632_163228


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1632_163280

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The condition c < 0 is sufficient but not necessary for f(x) < 0 -/
theorem sufficient_not_necessary (b c : ℝ) :
  (c < 0 → ∃ x, f b c x < 0) ∧
  ∃ b' c' x', c' ≥ 0 ∧ f b' c' x' < 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1632_163280


namespace NUMINAMATH_CALUDE_smallest_a_is_54_l1632_163247

/-- A polynomial with three positive integer roots -/
structure PolynomialWithIntegerRoots where
  a : ℤ
  b : ℤ
  roots : Fin 3 → ℤ
  roots_positive : ∀ i, 0 < roots i
  polynomial_property : ∀ x, x^3 - a*x^2 + b*x - 30030 = (x - roots 0) * (x - roots 1) * (x - roots 2)

/-- The smallest possible value of a for a polynomial with three positive integer roots -/
def smallest_a : ℤ := 54

/-- Theorem stating that 54 is the smallest possible value of a -/
theorem smallest_a_is_54 (p : PolynomialWithIntegerRoots) : 
  p.a ≥ smallest_a ∧ ∃ (q : PolynomialWithIntegerRoots), q.a = smallest_a :=
sorry

end NUMINAMATH_CALUDE_smallest_a_is_54_l1632_163247


namespace NUMINAMATH_CALUDE_caitlin_age_caitlin_age_proof_l1632_163210

/-- Proves that Caitlin is 54 years old given the conditions in the problem -/
theorem caitlin_age : ℕ → ℕ → ℕ → Prop :=
  λ anna_age brianna_age caitlin_age =>
    anna_age = 48 ∧
    brianna_age = 2 * (anna_age - 18) ∧
    caitlin_age = brianna_age - 6 →
    caitlin_age = 54

/-- Proof of the theorem -/
theorem caitlin_age_proof : caitlin_age 48 60 54 := by
  sorry

end NUMINAMATH_CALUDE_caitlin_age_caitlin_age_proof_l1632_163210


namespace NUMINAMATH_CALUDE_clock_angle_at_7_l1632_163222

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The time in hours -/
def time : ℕ := 7

/-- The angle between each hour mark on the clock -/
def angle_per_hour : ℚ := 360 / clock_hours

/-- The position of the hour hand in degrees -/
def hour_hand_position : ℚ := time * angle_per_hour

/-- The smaller angle between the hour and minute hands at the given time -/
def smaller_angle : ℚ := min hour_hand_position (360 - hour_hand_position)

/-- Theorem stating that the smaller angle between clock hands at 7 o'clock is 150 degrees -/
theorem clock_angle_at_7 : smaller_angle = 150 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_7_l1632_163222


namespace NUMINAMATH_CALUDE_vanilla_percentage_is_30_percent_l1632_163293

def chocolate : ℕ := 70
def vanilla : ℕ := 90
def strawberry : ℕ := 50
def mint : ℕ := 30
def cookieDough : ℕ := 60

def totalResponses : ℕ := chocolate + vanilla + strawberry + mint + cookieDough

theorem vanilla_percentage_is_30_percent :
  (vanilla : ℚ) / (totalResponses : ℚ) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_vanilla_percentage_is_30_percent_l1632_163293


namespace NUMINAMATH_CALUDE_frac_two_x_gt_one_sufficient_not_necessary_for_x_lt_two_l1632_163262

theorem frac_two_x_gt_one_sufficient_not_necessary_for_x_lt_two :
  (∃ x : ℝ, 2 / x > 1 ∧ x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ ¬(2 / x > 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_frac_two_x_gt_one_sufficient_not_necessary_for_x_lt_two_l1632_163262


namespace NUMINAMATH_CALUDE_box_dimensions_sum_l1632_163227

theorem box_dimensions_sum (X Y Z : ℝ) : 
  X > 0 ∧ Y > 0 ∧ Z > 0 →
  X * Y = 24 →
  X * Z = 48 →
  Y * Z = 72 →
  X + Y + Z = 22 := by
sorry

end NUMINAMATH_CALUDE_box_dimensions_sum_l1632_163227


namespace NUMINAMATH_CALUDE_power_three_minus_two_plus_three_l1632_163268

theorem power_three_minus_two_plus_three : 2^3 - 2 + 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_three_minus_two_plus_three_l1632_163268


namespace NUMINAMATH_CALUDE_base3_addition_proof_l1632_163295

/-- Represents a single digit in base 3 -/
def Base3Digit := Fin 3

/-- Represents a three-digit number in base 3 -/
def Base3Number := Fin 27

def toBase3 (n : ℕ) : Base3Number :=
  Fin.ofNat (n % 27)

def fromBase3 (n : Base3Number) : ℕ :=
  n.val

def addBase3 (a b c : Base3Number) : Base3Number :=
  toBase3 (fromBase3 a + fromBase3 b + fromBase3 c)

theorem base3_addition_proof (C D : ℕ) 
  (h1 : C < 10 ∧ D < 10)
  (h2 : addBase3 (toBase3 (D * 10 + D)) (toBase3 (3 * 10 + 2)) (toBase3 (C * 100 + 2 * 10 + 4)) = 
        toBase3 (C * 100 + 2 * 10 + 4 + 1)) :
  toBase3 (if D > C then D - C else C - D) = toBase3 1 := by
  sorry

end NUMINAMATH_CALUDE_base3_addition_proof_l1632_163295


namespace NUMINAMATH_CALUDE_price_decrease_approx_16_67_percent_l1632_163246

/-- Calculates the percent decrease between two prices -/
def percentDecrease (oldPrice newPrice : ℚ) : ℚ :=
  (oldPrice - newPrice) / oldPrice * 100

/-- The original price per pack -/
def originalPricePerPack : ℚ := 9 / 6

/-- The promotional price per pack -/
def promotionalPricePerPack : ℚ := 10 / 8

/-- Theorem stating that the percent decrease in price per pack is approximately 16.67% -/
theorem price_decrease_approx_16_67_percent :
  abs (percentDecrease originalPricePerPack promotionalPricePerPack - 100 * (1 / 6)) < 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_price_decrease_approx_16_67_percent_l1632_163246


namespace NUMINAMATH_CALUDE_points_per_recycled_bag_l1632_163240

theorem points_per_recycled_bag 
  (total_bags : ℕ) 
  (unrecycled_bags : ℕ) 
  (total_points : ℕ) 
  (h1 : total_bags = 11) 
  (h2 : unrecycled_bags = 2) 
  (h3 : total_points = 45) :
  total_points / (total_bags - unrecycled_bags) = 5 := by
  sorry

end NUMINAMATH_CALUDE_points_per_recycled_bag_l1632_163240


namespace NUMINAMATH_CALUDE_rotational_cipher_key_l1632_163267

/-- Represents the encoding function for a rotational cipher --/
def encode (key : ℕ) (letter : ℕ) : ℕ :=
  ((letter + key - 1) % 26) + 1

/-- Theorem: If the sum of encoded values for A, B, and C is 52, the key is 25 --/
theorem rotational_cipher_key (key : ℕ) 
  (h1 : 1 ≤ key ∧ key ≤ 26) 
  (h2 : encode key 1 + encode key 2 + encode key 3 = 52) : 
  key = 25 := by
  sorry

#check rotational_cipher_key

end NUMINAMATH_CALUDE_rotational_cipher_key_l1632_163267


namespace NUMINAMATH_CALUDE_two_talent_students_l1632_163274

theorem two_talent_students (total : ℕ) (cant_sing cant_dance cant_act : ℕ) :
  total = 50 ∧
  cant_sing = 20 ∧
  cant_dance = 35 ∧
  cant_act = 15 →
  ∃ (two_talents : ℕ),
    two_talents = 30 ∧
    two_talents = (total - cant_sing) + (total - cant_dance) + (total - cant_act) - total :=
by sorry

end NUMINAMATH_CALUDE_two_talent_students_l1632_163274


namespace NUMINAMATH_CALUDE_complex_cube_theorem_l1632_163236

theorem complex_cube_theorem : 
  let i : ℂ := Complex.I
  let z : ℂ := (2 + i) / (1 - 2*i)
  z^3 = -i := by sorry

end NUMINAMATH_CALUDE_complex_cube_theorem_l1632_163236


namespace NUMINAMATH_CALUDE_maria_boxes_count_l1632_163224

def eggs_per_box : ℕ := 7
def total_eggs : ℕ := 21

theorem maria_boxes_count : 
  total_eggs / eggs_per_box = 3 := by sorry

end NUMINAMATH_CALUDE_maria_boxes_count_l1632_163224


namespace NUMINAMATH_CALUDE_bananas_bought_l1632_163239

/-- The number of times Ronald went to the store last month -/
def store_visits : ℕ := 2

/-- The number of bananas Ronald buys each time he goes to the store -/
def bananas_per_visit : ℕ := 10

/-- The total number of bananas Ronald bought last month -/
def total_bananas : ℕ := store_visits * bananas_per_visit

theorem bananas_bought : total_bananas = 20 := by
  sorry

end NUMINAMATH_CALUDE_bananas_bought_l1632_163239


namespace NUMINAMATH_CALUDE_cube_root_7200_simplification_l1632_163235

theorem cube_root_7200_simplification : 
  ∃ (c d : ℕ+), (c.val : ℝ) * (d.val : ℝ)^(1/3) = 7200^(1/3) ∧ 
  (∀ (c' d' : ℕ+), (c'.val : ℝ) * (d'.val : ℝ)^(1/3) = 7200^(1/3) → d'.val ≤ d.val) →
  c.val + d.val = 452 := by
sorry

end NUMINAMATH_CALUDE_cube_root_7200_simplification_l1632_163235


namespace NUMINAMATH_CALUDE_arcsin_arccos_inequality_l1632_163221

theorem arcsin_arccos_inequality (x : ℝ) : 
  Real.arcsin ((5 / (2 * Real.pi)) * Real.arccos x) > Real.arccos ((10 / (3 * Real.pi)) * Real.arcsin x) ↔ 
  (x ∈ Set.Icc (Real.cos (2 * Real.pi / 5)) (Real.cos (8 * Real.pi / 25)) ∪ 
   Set.Ioo (Real.cos (8 * Real.pi / 25)) (Real.cos (Real.pi / 5))) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_arccos_inequality_l1632_163221


namespace NUMINAMATH_CALUDE_tangent_sphere_radius_l1632_163250

/-- A truncated cone with horizontal bases of radii 12 and 4, and height 15 -/
structure TruncatedCone where
  largeRadius : ℝ := 12
  smallRadius : ℝ := 4
  height : ℝ := 15

/-- A sphere tangent to the inside surfaces of a truncated cone -/
structure TangentSphere (cone : TruncatedCone) where
  radius : ℝ

/-- The radius of the tangent sphere is √161/2 -/
theorem tangent_sphere_radius (cone : TruncatedCone) (sphere : TangentSphere cone) :
  sphere.radius = Real.sqrt 161 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sphere_radius_l1632_163250


namespace NUMINAMATH_CALUDE_fraction_irreducible_l1632_163278

theorem fraction_irreducible (n m : ℕ) : 
  Nat.gcd (m * (n + 1) + 1) (m * (n + 1) - n) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l1632_163278


namespace NUMINAMATH_CALUDE_f_smallest_positive_period_l1632_163255

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x) + 2^(|Real.sin (2 * x)|^2) + 5 * |Real.sin (2 * x)|

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬is_periodic f q

theorem f_smallest_positive_period :
  is_smallest_positive_period f (Real.pi / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_smallest_positive_period_l1632_163255


namespace NUMINAMATH_CALUDE_alex_savings_dimes_l1632_163245

/-- Proves that given $6.35 in dimes and quarters, with 5 more dimes than quarters, the number of dimes is 22 -/
theorem alex_savings_dimes : 
  ∀ (d q : ℕ), 
    (d : ℚ) * 0.1 + (q : ℚ) * 0.25 = 6.35 → -- Total value in dollars
    d = q + 5 → -- 5 more dimes than quarters
    d = 22 := by sorry

end NUMINAMATH_CALUDE_alex_savings_dimes_l1632_163245


namespace NUMINAMATH_CALUDE_hobbit_burrow_assignment_l1632_163216

-- Define the burrows
inductive Burrow
| A | B | C | D | E | F

-- Define the hobbits
inductive Hobbit
| Frodo | Sam | Merry | Pippin

-- Define the concept of distance between burrows
def closer_to (b1 b2 b3 : Burrow) : Prop := sorry

-- Define the concept of closeness to river and forest
def closer_to_river (b1 b2 : Burrow) : Prop := sorry
def farther_from_forest (b1 b2 : Burrow) : Prop := sorry

-- Define the assignment of hobbits to burrows
def assignment : Hobbit → Burrow
| Hobbit.Frodo => Burrow.E
| Hobbit.Sam => Burrow.A
| Hobbit.Merry => Burrow.C
| Hobbit.Pippin => Burrow.F

-- Theorem statement
theorem hobbit_burrow_assignment :
  (∀ h1 h2 : Hobbit, h1 ≠ h2 → assignment h1 ≠ assignment h2) ∧
  (∀ b : Burrow, b ≠ Burrow.B ∧ b ≠ Burrow.D → ∃ h : Hobbit, assignment h = b) ∧
  (closer_to Burrow.B Burrow.A Burrow.E) ∧
  (closer_to Burrow.D Burrow.A Burrow.E) ∧
  (closer_to_river Burrow.E Burrow.C) ∧
  (farther_from_forest Burrow.E Burrow.F) :=
by sorry

end NUMINAMATH_CALUDE_hobbit_burrow_assignment_l1632_163216


namespace NUMINAMATH_CALUDE_coin_split_sum_l1632_163275

/-- Represents the sum of recorded products when splitting coins into piles -/
def recordedSum (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that for 25 coins, the sum of recorded products is 300 -/
theorem coin_split_sum :
  recordedSum 25 = 300 := by
  sorry

end NUMINAMATH_CALUDE_coin_split_sum_l1632_163275


namespace NUMINAMATH_CALUDE_rectangle_area_l1632_163242

/-- Given a rectangle with width 42 inches, where ten such rectangles placed end to end
    reach a length of 390 inches, prove that its area is 1638 square inches. -/
theorem rectangle_area (width : ℝ) (total_length : ℝ) (h1 : width = 42)
    (h2 : total_length = 390) : width * (total_length / 10) = 1638 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1632_163242


namespace NUMINAMATH_CALUDE_pencil_boxes_filled_l1632_163223

theorem pencil_boxes_filled (total_pencils : ℕ) (pencils_per_box : ℕ) (h1 : total_pencils = 7344) (h2 : pencils_per_box = 7) : 
  total_pencils / pencils_per_box = 1049 := by
  sorry

end NUMINAMATH_CALUDE_pencil_boxes_filled_l1632_163223


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1632_163252

/-- Given two lines in the xy-plane, this function returns true if they are parallel --/
def are_parallel (m1 : ℝ) (m2 : ℝ) : Prop := m1 = m2

/-- Given a point (x, y) and a line equation y = mx + b, this function returns true if the point lies on the line --/
def point_on_line (x : ℝ) (y : ℝ) (m : ℝ) (b : ℝ) : Prop := y = m * x + b

theorem parallel_line_through_point (x0 y0 : ℝ) : 
  ∃ (m b : ℝ), 
    are_parallel m 2 ∧ 
    point_on_line x0 y0 m b ∧ 
    m = 2 ∧ 
    b = -5 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1632_163252


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1632_163229

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h : IsGeometric a) 
    (h3 : a 3 = 3) (h10 : a 10 = 384) : 
  ∀ n : ℕ, a n = 3 * 2^(n - 3) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1632_163229


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1632_163233

def M : Set ℝ := {x : ℝ | x^2 + x - 6 = 0}

def N (a : ℝ) : Set ℝ := {x : ℝ | a * x + 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (N a ⊆ M) ↔ (a = -1 ∨ a = 0 ∨ a = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1632_163233


namespace NUMINAMATH_CALUDE_peters_socks_l1632_163230

theorem peters_socks (x y z : ℕ) : 
  x + y + z = 15 →
  2*x + 3*y + 5*z = 45 →
  x ≥ 1 →
  y ≥ 1 →
  z ≥ 1 →
  x = 6 :=
by sorry

end NUMINAMATH_CALUDE_peters_socks_l1632_163230


namespace NUMINAMATH_CALUDE_functional_equation_problem_l1632_163243

/-- A function satisfying f(a+b) = f(a)f(b) for all a and b -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b, f (a + b) = f a * f b

theorem functional_equation_problem (f : ℝ → ℝ) 
  (h1 : FunctionalEquation f) 
  (h2 : f 1 = 2) : 
  (f 1)^2 / f 1 + f 2 / f 1 + 
  (f 2)^2 / f 3 + f 4 / f 3 + 
  (f 3)^2 / f 5 + f 6 / f 5 + 
  (f 4)^2 / f 7 + f 8 / f 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l1632_163243


namespace NUMINAMATH_CALUDE_simplify_expression_l1632_163238

theorem simplify_expression (c : ℝ) : ((3 * c + 6) - 6 * c) / 3 = -c + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1632_163238


namespace NUMINAMATH_CALUDE_cindy_calculation_l1632_163219

theorem cindy_calculation (x : ℚ) : (x - 7) / 5 = 53 → (x - 5) / 7 = 38 := by
  sorry

end NUMINAMATH_CALUDE_cindy_calculation_l1632_163219


namespace NUMINAMATH_CALUDE_total_time_calculation_l1632_163265

/-- Calculates the total time to complete an assignment and clean sticky keys. -/
theorem total_time_calculation (assignment_time : ℕ) (num_keys : ℕ) (time_per_key : ℕ) :
  assignment_time = 10 ∧ num_keys = 14 ∧ time_per_key = 3 →
  assignment_time + num_keys * time_per_key = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_time_calculation_l1632_163265


namespace NUMINAMATH_CALUDE_division_of_eleven_by_five_l1632_163284

theorem division_of_eleven_by_five :
  ∃ (A B : ℕ), 11 = 5 * A + B ∧ B < 5 ∧ A = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_eleven_by_five_l1632_163284


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_sum_l1632_163260

theorem geometric_arithmetic_sequence_ratio_sum :
  ∀ (x y z : ℝ),
  (x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) →
  (∃ (r : ℝ), r ≠ 0 ∧ 4*y = 3*x*r ∧ 5*z = 4*y*r) →
  (∃ (d : ℝ), 1/y - 1/x = d ∧ 1/z - 1/y = d) →
  x/z + z/x = 34/15 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_ratio_sum_l1632_163260


namespace NUMINAMATH_CALUDE_bryan_pushups_l1632_163258

/-- The number of push-ups Bryan did in total -/
def total_pushups (sets : ℕ) (pushups_per_set : ℕ) (reduced_pushups : ℕ) : ℕ :=
  (sets - 1) * pushups_per_set + (pushups_per_set - reduced_pushups)

/-- Theorem stating that Bryan did 100 push-ups in total -/
theorem bryan_pushups :
  total_pushups 9 12 8 = 100 := by
  sorry

end NUMINAMATH_CALUDE_bryan_pushups_l1632_163258


namespace NUMINAMATH_CALUDE_unique_divisible_by_33_l1632_163299

/-- Represents a five-digit number in the form 7n742 where n is a single digit -/
def number (n : ℕ) : ℕ := 70000 + n * 1000 + 742

/-- Checks if a number is divisible by another number -/
def isDivisibleBy (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem unique_divisible_by_33 :
  (isDivisibleBy (number 1) 33) ∧
  (∀ n : ℕ, n ≤ 9 → n ≠ 1 → ¬(isDivisibleBy (number n) 33)) :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_by_33_l1632_163299


namespace NUMINAMATH_CALUDE_mother_pies_per_day_l1632_163287

/-- The number of pies Eddie's mother can bake per day -/
def mother_pies : ℕ := 8

/-- The number of pies Eddie can bake per day -/
def eddie_pies : ℕ := 3

/-- The number of pies Eddie's sister can bake per day -/
def sister_pies : ℕ := 6

/-- The number of days they bake pies -/
def days : ℕ := 7

/-- The total number of pies they can bake in the given days -/
def total_pies : ℕ := 119

theorem mother_pies_per_day :
  eddie_pies * days + sister_pies * days + mother_pies * days = total_pies :=
by sorry

end NUMINAMATH_CALUDE_mother_pies_per_day_l1632_163287


namespace NUMINAMATH_CALUDE_journey_distance_l1632_163218

/-- Proves that given a journey of 9 hours, partly on foot at 4 km/hr for 16 km,
    and partly on bicycle at 9 km/hr, the total distance traveled is 61 km. -/
theorem journey_distance (total_time foot_speed bike_speed foot_distance : ℝ) :
  total_time = 9 ∧
  foot_speed = 4 ∧
  bike_speed = 9 ∧
  foot_distance = 16 →
  ∃ (bike_distance : ℝ),
    foot_distance / foot_speed + bike_distance / bike_speed = total_time ∧
    foot_distance + bike_distance = 61 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_l1632_163218


namespace NUMINAMATH_CALUDE_remainder_problem_l1632_163266

theorem remainder_problem (d : ℕ) (r : ℕ) (h1 : d > 1) 
  (h2 : 1059 % d = r)
  (h3 : 1482 % d = r)
  (h4 : 2340 % d = r) :
  2 * d - r = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1632_163266


namespace NUMINAMATH_CALUDE_intersection_count_l1632_163272

/-- Calculates the number of intersections between two regular polygons inscribed in a circle -/
def intersections (n m : ℕ) : ℕ := 2 * n * m

/-- The set of regular polygons inscribed in the circle -/
def polygons : Finset ℕ := {4, 6, 8, 10}

/-- The set of all pairs of polygons -/
def polygon_pairs : Finset (ℕ × ℕ) := 
  {(4, 6), (4, 8), (4, 10), (6, 8), (6, 10), (8, 10)}

theorem intersection_count :
  (polygon_pairs.sum (fun (p : ℕ × ℕ) => intersections p.1 p.2)) = 568 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l1632_163272


namespace NUMINAMATH_CALUDE_age_difference_l1632_163297

-- Define variables for ages
variable (a b c : ℕ)

-- Define the condition from the problem
def age_condition (a b c : ℕ) : Prop := a + b = b + c + 12

-- Theorem to prove
theorem age_difference (h : age_condition a b c) : a = c + 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1632_163297


namespace NUMINAMATH_CALUDE_percentage_equality_l1632_163226

theorem percentage_equality : (10 : ℚ) / 100 * 200 = (20 : ℚ) / 100 * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l1632_163226


namespace NUMINAMATH_CALUDE_even_monotone_increasing_neg_implies_f1_gt_fneg2_l1632_163259

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_increasing_on_neg (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ 0 → f x < f y

-- Theorem statement
theorem even_monotone_increasing_neg_implies_f1_gt_fneg2
  (h_even : is_even f)
  (h_mono : monotone_increasing_on_neg f) :
  f 1 > f (-2) :=
sorry

end NUMINAMATH_CALUDE_even_monotone_increasing_neg_implies_f1_gt_fneg2_l1632_163259


namespace NUMINAMATH_CALUDE_remainder_3079_div_67_l1632_163273

theorem remainder_3079_div_67 : 3079 % 67 = 64 := by sorry

end NUMINAMATH_CALUDE_remainder_3079_div_67_l1632_163273


namespace NUMINAMATH_CALUDE_quadratic_other_intercept_l1632_163209

/-- For a quadratic function f(x) = ax^2 + bx + c with vertex (2, 10) and one x-intercept at (1, 0),
    the x-coordinate of the other x-intercept is 3. -/
theorem quadratic_other_intercept 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 2 = 10 ∧ (∀ x, f x ≤ 10)) 
  (h3 : f 1 = 0) : 
  ∃ x, x ≠ 1 ∧ f x = 0 ∧ x = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_other_intercept_l1632_163209


namespace NUMINAMATH_CALUDE_distance_between_x_and_y_l1632_163292

-- Define the walking speeds
def yolanda_speed : ℝ := 2
def bob_speed : ℝ := 4

-- Define the time difference in starting
def time_difference : ℝ := 1

-- Define Bob's distance walked when they meet
def bob_distance : ℝ := 25.333333333333332

-- Define the total distance between X and Y
def total_distance : ℝ := 40

-- Theorem statement
theorem distance_between_x_and_y :
  let time_bob_walked := bob_distance / bob_speed
  let yolanda_distance := yolanda_speed * (time_bob_walked + time_difference)
  yolanda_distance + bob_distance = total_distance := by
  sorry

end NUMINAMATH_CALUDE_distance_between_x_and_y_l1632_163292


namespace NUMINAMATH_CALUDE_male_contestants_count_l1632_163264

theorem male_contestants_count (total : ℕ) (female_ratio : ℚ) : 
  total = 18 → female_ratio = 1/3 → (total : ℚ) * (1 - female_ratio) = 12 := by
  sorry

end NUMINAMATH_CALUDE_male_contestants_count_l1632_163264


namespace NUMINAMATH_CALUDE_units_digit_of_3_pow_5_times_2_pow_3_l1632_163211

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of 3^5 × 2^3 is 4 -/
theorem units_digit_of_3_pow_5_times_2_pow_3 :
  unitsDigit (3^5 * 2^3) = 4 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_3_pow_5_times_2_pow_3_l1632_163211


namespace NUMINAMATH_CALUDE_system_solution_iff_conditions_l1632_163200

-- Define the system of equations
def has_solution (n p x y z : ℕ) : Prop :=
  x + p * y = n ∧ x + y = p^z

-- Define the conditions
def conditions (n p : ℕ) : Prop :=
  p > 1 ∧ (n - 1) % (p - 1) = 0 ∧ ∀ k : ℕ, n ≠ p^k

-- Theorem statement
theorem system_solution_iff_conditions (n p : ℕ) :
  (∃! x y z : ℕ, has_solution n p x y z) ↔ conditions n p :=
sorry

end NUMINAMATH_CALUDE_system_solution_iff_conditions_l1632_163200


namespace NUMINAMATH_CALUDE_p_or_q_false_sufficient_not_necessary_l1632_163276

-- Define propositions p and q
variable (p q : Prop)

-- Define the statement "p or q is false"
def p_or_q_false : Prop := ¬(p ∨ q)

-- Define the statement "not p is true"
def not_p_true : Prop := ¬p

-- Theorem stating that "p or q is false" is sufficient but not necessary for "not p is true"
theorem p_or_q_false_sufficient_not_necessary :
  (p_or_q_false p q → not_p_true p) ∧
  ¬(not_p_true p → p_or_q_false p q) :=
sorry

end NUMINAMATH_CALUDE_p_or_q_false_sufficient_not_necessary_l1632_163276


namespace NUMINAMATH_CALUDE_line_proof_l1632_163271

-- Define the three given lines
def line1 (x y : ℝ) : Prop := 3 * x + y = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := 2 * x + y + 3 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := x - 2 * y + 7 = 0

-- Theorem statement
theorem line_proof :
  ∃ (x₀ y₀ : ℝ),
    (line1 x₀ y₀ ∧ line2 x₀ y₀) ∧  -- Intersection point
    (∀ (x y : ℝ), result_line x y → 
      ((y - y₀) = -(1 / (2 : ℝ)) * (x - x₀)) ∧  -- Slope is perpendicular
      (result_line x₀ y₀))  -- Result line passes through intersection
:= by sorry

end NUMINAMATH_CALUDE_line_proof_l1632_163271


namespace NUMINAMATH_CALUDE_cube_averaging_solution_l1632_163277

/-- Represents a cube with real numbers on its vertices -/
structure Cube where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ
  G : ℝ
  H : ℝ

/-- Checks if the cube satisfies the averaging condition -/
def satisfiesAveraging (c : Cube) : Prop :=
  (c.D + c.E + c.B) / 3 = 6 ∧
  (c.A + c.F + c.C) / 3 = 3 ∧
  (c.D + c.G + c.B) / 3 = 6 ∧
  (c.A + c.C + c.H) / 3 = 4 ∧
  (c.A + c.H + c.F) / 3 = 3 ∧
  (c.E + c.G + c.B) / 3 = 6 ∧
  (c.H + c.F + c.C) / 3 = 5 ∧
  (c.D + c.G + c.E) / 3 = 3

/-- The theorem stating that the given solution is the only one satisfying the averaging condition -/
theorem cube_averaging_solution :
  ∀ c : Cube, satisfiesAveraging c →
    c.A = 0 ∧ c.B = 12 ∧ c.C = 6 ∧ c.D = 3 ∧ c.E = 3 ∧ c.F = 3 ∧ c.G = 3 ∧ c.H = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_averaging_solution_l1632_163277


namespace NUMINAMATH_CALUDE_yeast_population_after_30_minutes_l1632_163270

/-- The population of yeast cells after a given time period. -/
def yeast_population (initial_population : ℕ) (time_minutes : ℕ) : ℕ :=
  initial_population * (3 ^ (time_minutes / 5))

/-- Theorem: The yeast population after 30 minutes is 36450 cells. -/
theorem yeast_population_after_30_minutes :
  yeast_population 50 30 = 36450 := by
  sorry

end NUMINAMATH_CALUDE_yeast_population_after_30_minutes_l1632_163270


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l1632_163285

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem for the first part of the problem
theorem solution_set_of_inequality (x : ℝ) :
  f x ≤ x + 5 ↔ 0 ≤ x ∧ x ≤ 4 := by sorry

-- Theorem for the second part of the problem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a^2 + 4*a) → -5 ≤ a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l1632_163285


namespace NUMINAMATH_CALUDE_min_equation_solution_l1632_163296

theorem min_equation_solution (x : ℝ) : min (1/2 + x) (x^2) = 1 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_min_equation_solution_l1632_163296


namespace NUMINAMATH_CALUDE_lateral_surface_area_theorem_l1632_163206

/-- Represents a regular quadrilateral pyramid -/
structure RegularQuadrilateralPyramid where
  /-- The dihedral angle at the lateral edge -/
  dihedral_angle : ℝ
  /-- The area of the diagonal section -/
  diagonal_section_area : ℝ

/-- The lateral surface area of a regular quadrilateral pyramid -/
noncomputable def lateral_surface_area (p : RegularQuadrilateralPyramid) : ℝ :=
  4 * p.diagonal_section_area

/-- Theorem: The lateral surface area of a regular quadrilateral pyramid is 4S,
    where S is the area of its diagonal section, given that the dihedral angle
    at the lateral edge is 120° -/
theorem lateral_surface_area_theorem (p : RegularQuadrilateralPyramid) 
  (h : p.dihedral_angle = 120) : 
  lateral_surface_area p = 4 * p.diagonal_section_area := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_theorem_l1632_163206


namespace NUMINAMATH_CALUDE_f_max_at_two_l1632_163241

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x + 16

/-- Theorem stating that f has a maximum value of 24 at x = 2 -/
theorem f_max_at_two :
  ∃ (x_max : ℝ), x_max = 2 ∧ f x_max = 24 ∧ ∀ (x : ℝ), f x ≤ f x_max :=
sorry

end NUMINAMATH_CALUDE_f_max_at_two_l1632_163241


namespace NUMINAMATH_CALUDE_compare_large_powers_l1632_163231

theorem compare_large_powers : 100^100 > 50^50 * 150^50 := by
  sorry

end NUMINAMATH_CALUDE_compare_large_powers_l1632_163231


namespace NUMINAMATH_CALUDE_x_squared_over_x_fourth_plus_x_squared_plus_one_l1632_163256

theorem x_squared_over_x_fourth_plus_x_squared_plus_one (x : ℝ) 
  (h1 : x^2 - 3*x - 1 = 0) (h2 : x ≠ 0) : x^2 / (x^4 + x^2 + 1) = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_over_x_fourth_plus_x_squared_plus_one_l1632_163256


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1632_163288

/-- The eccentricity of the hyperbola x² - 4y² = 1 is √5/2 -/
theorem hyperbola_eccentricity : 
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - 4*y^2 = 1
  ∃ e : ℝ, e = Real.sqrt 5 / 2 ∧ 
    ∀ x y : ℝ, h x y → 
      ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
        x^2 / a^2 - y^2 / b^2 = 1 ∧
        c^2 = a^2 + b^2 ∧
        e = c / a :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1632_163288
