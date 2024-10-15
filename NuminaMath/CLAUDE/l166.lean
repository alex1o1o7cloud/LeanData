import Mathlib

namespace NUMINAMATH_CALUDE_supremum_inequality_l166_16655

theorem supremum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  - (1 / (2 * a)) - (2 / b) ≤ - (9 / 2) ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ - (1 / (2 * a₀)) - (2 / b₀) = - (9 / 2) :=
sorry

end NUMINAMATH_CALUDE_supremum_inequality_l166_16655


namespace NUMINAMATH_CALUDE_problem_statement_l166_16690

theorem problem_statement : 
  ∃ d : ℝ, 5^(Real.log 30) * (1/3)^(Real.log 0.5) = d ∧ d = 30 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l166_16690


namespace NUMINAMATH_CALUDE_book_distribution_l166_16640

theorem book_distribution (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 6 →
  k = 3 →
  m = 2 →
  (Nat.choose n m) * (Nat.choose (n - m) m) * (Nat.choose (n - 2*m) m) = 90 :=
by sorry

end NUMINAMATH_CALUDE_book_distribution_l166_16640


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_max_l166_16601

theorem triangle_cosine_sum_max (A B C : ℝ) : 
  0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ A + B + C = π →
  Real.cos A + Real.cos B * Real.cos C ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_max_l166_16601


namespace NUMINAMATH_CALUDE_jills_age_l166_16649

theorem jills_age (henry_age jill_age : ℕ) : 
  (henry_age + jill_age = 48) →
  (henry_age - 9 = 2 * (jill_age - 9)) →
  jill_age = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_jills_age_l166_16649


namespace NUMINAMATH_CALUDE_bacteria_growth_days_l166_16636

def initial_bacteria : ℕ := 5
def growth_rate : ℕ := 3
def target_bacteria : ℕ := 200

def bacteria_count (days : ℕ) : ℕ :=
  initial_bacteria * growth_rate ^ days

theorem bacteria_growth_days :
  (∀ k : ℕ, k < 4 → bacteria_count k ≤ target_bacteria) ∧
  bacteria_count 4 > target_bacteria :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_days_l166_16636


namespace NUMINAMATH_CALUDE_trinomial_square_l166_16692

theorem trinomial_square (a : ℚ) : 
  (∃ b : ℚ, ∀ x : ℚ, 9*x^2 + 21*x + a = (3*x + b)^2) → a = 49/4 := by
  sorry

end NUMINAMATH_CALUDE_trinomial_square_l166_16692


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l166_16610

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l166_16610


namespace NUMINAMATH_CALUDE_max_candy_count_l166_16648

/-- Represents the state of the board and candy count -/
structure BoardState where
  numbers : List Nat
  candy_count : Nat

/-- Combines two numbers on the board and updates the candy count -/
def combine_numbers (state : BoardState) (i j : Nat) : BoardState :=
  { numbers := (state.numbers.removeNth i).removeNth j ++ [state.numbers[i]! + state.numbers[j]!],
    candy_count := state.candy_count + state.numbers[i]! * state.numbers[j]! }

/-- Theorem: The maximum number of candies Karlson can eat is 300 -/
theorem max_candy_count :
  ∃ (final_state : BoardState),
    (final_state.numbers.length = 1) ∧
    (final_state.candy_count = 300) ∧
    (∃ (initial_state : BoardState),
      (initial_state.numbers = List.replicate 25 1) ∧
      (∃ (moves : List (Nat × Nat)),
        moves.length = 24 ∧
        final_state = moves.foldl (fun state (i, j) => combine_numbers state i j) initial_state)) :=
by
  sorry

#check max_candy_count

end NUMINAMATH_CALUDE_max_candy_count_l166_16648


namespace NUMINAMATH_CALUDE_symmetric_points_ab_power_l166_16600

/-- Given two points M(2a, 2) and N(-8, a+b) that are symmetric with respect to the y-axis,
    prove that a^b = 1/16 -/
theorem symmetric_points_ab_power (a b : ℝ) : 
  (∃ (M N : ℝ × ℝ), 
    M = (2*a, 2) ∧ 
    N = (-8, a+b) ∧ 
    (M.1 = -N.1) ∧  -- x-coordinates are opposite
    (M.2 = N.2))    -- y-coordinates are equal
  → a^b = 1/16 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_ab_power_l166_16600


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l166_16682

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 3490) % 15 = 2801 % 15 ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 3490) % 15 = 2801 % 15 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l166_16682


namespace NUMINAMATH_CALUDE_cylinder_to_cone_volume_l166_16698

/-- Given a cylindrical block carved into the largest possible cone, 
    if the volume of the part removed is 25.12 cubic centimeters, 
    then the volume of the original cylindrical block is 37.68 cubic centimeters 
    and the volume of the cone-shaped block is 12.56 cubic centimeters. -/
theorem cylinder_to_cone_volume (removed_volume : ℝ) 
  (h : removed_volume = 25.12) : 
  ∃ (cylinder_volume cone_volume : ℝ),
    cylinder_volume = 37.68 ∧ 
    cone_volume = 12.56 ∧
    removed_volume = cylinder_volume - cone_volume := by
  sorry

end NUMINAMATH_CALUDE_cylinder_to_cone_volume_l166_16698


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l166_16630

def B : Set ℕ := {n | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l166_16630


namespace NUMINAMATH_CALUDE_rational_combination_equals_24_l166_16646

theorem rational_combination_equals_24 :
  ∃ (f : List ℚ → ℚ),
    f [-1, -2, -3, -4] = 24 ∧
    (∀ x y z w, f [x, y, z, w] = ((x + y + z) * w) ∨
                f [x, y, z, w] = ((x + y + z) / w) ∨
                f [x, y, z, w] = ((x + y - z) * w) ∨
                f [x, y, z, w] = ((x + y - z) / w) ∨
                f [x, y, z, w] = ((x - y + z) * w) ∨
                f [x, y, z, w] = ((x - y + z) / w) ∨
                f [x, y, z, w] = ((x - y - z) * w) ∨
                f [x, y, z, w] = ((x - y - z) / w)) :=
by
  sorry

end NUMINAMATH_CALUDE_rational_combination_equals_24_l166_16646


namespace NUMINAMATH_CALUDE_gcd_97_power_plus_one_l166_16615

theorem gcd_97_power_plus_one (p : Nat) (h_prime : Nat.Prime p) (h_p : p = 97) :
  Nat.gcd (p^7 + 1) (p^7 + p^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_97_power_plus_one_l166_16615


namespace NUMINAMATH_CALUDE_y_derivative_l166_16696

noncomputable def y (x : ℝ) : ℝ := 
  (Real.cos (Real.tan (1/3)) * (Real.sin (15*x))^2) / (15 * Real.cos (30*x))

theorem y_derivative (x : ℝ) : 
  deriv y x = (Real.cos (Real.tan (1/3)) * Real.tan (30*x)) / Real.cos (30*x) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l166_16696


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l166_16664

theorem condition_necessary_not_sufficient :
  (∀ x y : ℝ, x = 1 ∧ y = 2 → x + y = 3) ∧
  (∃ x y : ℝ, x + y = 3 ∧ (x ≠ 1 ∨ y ≠ 2)) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l166_16664


namespace NUMINAMATH_CALUDE_quadratic_sum_l166_16617

theorem quadratic_sum (x : ℝ) : 
  let f : ℝ → ℝ := λ x => -3 * x^2 + 27 * x - 153
  ∃ (a b c : ℝ), (∀ x, f x = a * (x + b)^2 + c) ∧ (a + b + c = -99.75) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l166_16617


namespace NUMINAMATH_CALUDE_find_q_l166_16699

-- Define the polynomial g(x)
def g (p q r s t : ℝ) (x : ℝ) : ℝ := p*x^4 + q*x^3 + r*x^2 + s*x + t

-- State the theorem
theorem find_q :
  ∀ p q r s t : ℝ,
  (∀ x : ℝ, g p q r s t x = 0 ↔ x = -2 ∨ x = 0 ∨ x = 1 ∨ x = 3) →
  g p q r s t 2 = -24 →
  q = 12 := by
sorry


end NUMINAMATH_CALUDE_find_q_l166_16699


namespace NUMINAMATH_CALUDE_exp_sum_rule_l166_16652

theorem exp_sum_rule (a b : ℝ) : Real.exp a * Real.exp b = Real.exp (a + b) := by
  sorry

end NUMINAMATH_CALUDE_exp_sum_rule_l166_16652


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l166_16651

/-- The perimeter of a semicircle with radius 10 is approximately 51.4 -/
theorem semicircle_perimeter_approx : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |((10 : ℝ) * Real.pi + 20) - 51.4| < ε := by
  sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l166_16651


namespace NUMINAMATH_CALUDE_stock_price_example_l166_16663

/-- Given a stock with income, dividend rate, and investment amount, calculate its price. -/
def stock_price (income : ℚ) (dividend_rate : ℚ) (investment : ℚ) : ℚ :=
  let face_value := (income * 100) / dividend_rate
  (investment / face_value) * 100

/-- Theorem: The price of a stock with income Rs. 650, 10% dividend rate, and Rs. 6240 investment is Rs. 96. -/
theorem stock_price_example : stock_price 650 10 6240 = 96 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_example_l166_16663


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l166_16607

/-- Represents a six-digit number -/
def SixDigitNumber := { n : ℕ // 100000 ≤ n ∧ n ≤ 999999 }

/-- Represents a five-digit number -/
def FiveDigitNumber := { n : ℕ // 10000 ≤ n ∧ n ≤ 99999 }

/-- Function that removes one digit from a six-digit number to form a five-digit number -/
def removeOneDigit (n : SixDigitNumber) : FiveDigitNumber :=
  sorry

/-- The problem statement -/
theorem unique_six_digit_number :
  ∃! (n : SixDigitNumber), 
    ∀ (m : FiveDigitNumber), 
      (m = removeOneDigit n) → (n.val - m.val = 654321) := by
  sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l166_16607


namespace NUMINAMATH_CALUDE_four_even_cards_different_suits_count_l166_16681

/-- Represents a standard playing card suit -/
inductive Suit
| hearts
| diamonds
| clubs
| spades

/-- Represents an even-numbered card (including face cards) -/
inductive EvenCard
| two
| four
| six
| eight
| ten
| queen

/-- The number of suits in a standard deck -/
def number_of_suits : Nat := 4

/-- The number of even-numbered cards in each suit -/
def even_cards_per_suit : Nat := 6

/-- A function to calculate the number of ways to choose 4 cards from a standard deck
    under the given conditions -/
def choose_four_even_cards_different_suits : Nat :=
  number_of_suits * even_cards_per_suit ^ 4

/-- The theorem stating that the number of ways to choose 4 cards from a standard deck,
    where all four cards are of different suits, each card is even-numbered,
    and the order doesn't matter, is equal to 1296 -/
theorem four_even_cards_different_suits_count :
  choose_four_even_cards_different_suits = 1296 := by
  sorry


end NUMINAMATH_CALUDE_four_even_cards_different_suits_count_l166_16681


namespace NUMINAMATH_CALUDE_one_real_root_condition_l166_16650

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (k : ℝ) (x : ℝ) : Prop := lg (k * x) = 2 * lg (x + 1)

-- Theorem statement
theorem one_real_root_condition (k : ℝ) : 
  (∃! x : ℝ, equation k x) ↔ (k = 4 ∨ k < 0) :=
sorry

end NUMINAMATH_CALUDE_one_real_root_condition_l166_16650


namespace NUMINAMATH_CALUDE_polynomial_decrease_l166_16694

theorem polynomial_decrease (b : ℝ) :
  let P : ℝ → ℝ := fun x ↦ -2 * x + b
  ∀ x : ℝ, P (x + 1) = P x - 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_decrease_l166_16694


namespace NUMINAMATH_CALUDE_years_until_26_l166_16608

/-- Kiril's current age -/
def current_age : ℕ := sorry

/-- Kiril's target age -/
def target_age : ℕ := 26

/-- Condition that current age is a multiple of 5 -/
axiom current_age_multiple_of_5 : ∃ k : ℕ, current_age = 5 * k

/-- Condition that last year's age was a multiple of 7 -/
axiom last_year_age_multiple_of_7 : ∃ m : ℕ, current_age - 1 = 7 * m

/-- Theorem stating the number of years until Kiril is 26 -/
theorem years_until_26 : target_age - current_age = 11 := by sorry

end NUMINAMATH_CALUDE_years_until_26_l166_16608


namespace NUMINAMATH_CALUDE_expression_factorization_l166_16653

theorem expression_factorization (x y z : ℝ) :
  ((x^2 - y^2)^3 + (y^2 - z^2)^3 + (z^2 - x^2)^3) / 
  ((x - y)^3 + (y - z)^3 + (z - x)^3) = 
  (x + y) * (y + z) * (z + x) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l166_16653


namespace NUMINAMATH_CALUDE_afternoon_fish_count_l166_16632

/-- Proves that the number of fish caught in the afternoon is 3 --/
theorem afternoon_fish_count (morning_a : ℕ) (morning_b : ℕ) (total : ℕ)
  (h1 : morning_a = 4)
  (h2 : morning_b = 3)
  (h3 : total = 10) :
  total - (morning_a + morning_b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_fish_count_l166_16632


namespace NUMINAMATH_CALUDE_square_area_l166_16668

/-- The parabola function -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 4

/-- The line function -/
def g (x : ℝ) : ℝ := 3

/-- The theorem stating the area of the square -/
theorem square_area : 
  ∃ (x₁ x₂ : ℝ), 
    f x₁ = g x₁ ∧ 
    f x₂ = g x₂ ∧ 
    x₁ ≠ x₂ ∧
    (x₂ - x₁)^2 = 8 :=
sorry

end NUMINAMATH_CALUDE_square_area_l166_16668


namespace NUMINAMATH_CALUDE_gcd_6273_14593_l166_16688

theorem gcd_6273_14593 : Nat.gcd 6273 14593 = 3 := by sorry

end NUMINAMATH_CALUDE_gcd_6273_14593_l166_16688


namespace NUMINAMATH_CALUDE_solve_linear_equation_l166_16666

theorem solve_linear_equation (x : ℝ) (h : 5*x - 7 = 15*x + 13) : 3*(x+10) = 24 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l166_16666


namespace NUMINAMATH_CALUDE_l_shape_surface_area_l166_16637

/-- Represents the 'L' shaped solid described in the problem -/
structure LShape where
  base_cubes : Nat
  column_cubes : Nat
  base_length : Nat
  base_width : Nat
  extension_length : Nat

/-- Calculates the surface area of the 'L' shaped solid -/
def surface_area (shape : LShape) : Nat :=
  let base_area := shape.base_cubes
  let top_exposed := shape.base_cubes - 1
  let column_sides := 4 * shape.column_cubes
  let column_top := 1
  let base_perimeter := 2 * (shape.base_length + shape.base_width + 2 * shape.extension_length)
  top_exposed + column_sides + column_top + base_perimeter

/-- The specific 'L' shape described in the problem -/
def problem_shape : LShape := {
  base_cubes := 8
  column_cubes := 7
  base_length := 3
  base_width := 2
  extension_length := 2
}

theorem l_shape_surface_area :
  surface_area problem_shape = 58 := by sorry

end NUMINAMATH_CALUDE_l_shape_surface_area_l166_16637


namespace NUMINAMATH_CALUDE_special_triangle_sides_l166_16680

/-- A triangle with sides a, b, and c satisfying specific conditions -/
structure SpecialTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  perimeter_eq : a + b + c = 18
  sum_eq_double_c : a + b = 2 * c
  b_eq_double_a : b = 2 * a

/-- Theorem stating the unique side lengths of the special triangle -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 4 ∧ t.b = 8 ∧ t.c = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sides_l166_16680


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_radius_of_circle_l166_16605

-- Define the circle M
def circle_M (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*x + a = 0

-- Define the point P
def point_P : ℝ × ℝ := (4, 5)

-- Define the tangent line equations
def tangent_line_1 (x : ℝ) : Prop := x = 4
def tangent_line_2 (x y : ℝ) : Prop := 8*x - 15*y + 43 = 0

-- Theorem for part (1)
theorem tangent_lines_to_circle (x y : ℝ) :
  circle_M (-8) x y →
  (∃ (t : ℝ), (x = t * (point_P.1 - x) + x ∧ y = t * (point_P.2 - y) + y)) →
  (tangent_line_1 x ∨ tangent_line_2 x y) :=
sorry

-- Theorem for part (2)
theorem radius_of_circle (a : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  circle_M a x₁ y₁ →
  circle_M a x₂ y₂ →
  x₁ * x₂ + y₁ * y₂ = -6 →
  ∃ (r : ℝ), r^2 = 7 ∧ 
    ∀ (x y : ℝ), circle_M a x y → (x - 1)^2 + y^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_radius_of_circle_l166_16605


namespace NUMINAMATH_CALUDE_total_pups_is_91_l166_16685

/-- Represents the number of pups for each dog breed --/
structure DogBreedPups where
  huskies : Nat
  pitbulls : Nat
  goldenRetrievers : Nat
  germanShepherds : Nat
  bulldogs : Nat
  poodles : Nat

/-- Calculates the total number of pups from all dog breeds --/
def totalPups (d : DogBreedPups) : Nat :=
  d.huskies + d.pitbulls + d.goldenRetrievers + d.germanShepherds + d.bulldogs + d.poodles

/-- Theorem stating that the total number of pups is 91 --/
theorem total_pups_is_91 :
  let numHuskies := 5
  let numPitbulls := 2
  let numGoldenRetrievers := 4
  let numGermanShepherds := 3
  let numBulldogs := 2
  let numPoodles := 3
  let huskiePups := 4
  let pitbullPups := 3
  let goldenRetrieverPups := huskiePups + 2
  let germanShepherdPups := pitbullPups + 3
  let bulldogPups := 4
  let poodlePups := bulldogPups + 1
  let d := DogBreedPups.mk
    (numHuskies * huskiePups)
    (numPitbulls * pitbullPups)
    (numGoldenRetrievers * goldenRetrieverPups)
    (numGermanShepherds * germanShepherdPups)
    (numBulldogs * bulldogPups)
    (numPoodles * poodlePups)
  totalPups d = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_pups_is_91_l166_16685


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_pow_2017_l166_16662

-- Define the pattern of last two digits
def lastTwoDigitsPattern : Fin 4 → Nat
  | 0 => 49
  | 1 => 43
  | 2 => 01
  | 3 => 07

-- Define the function to get the last two digits of 7^n
def lastTwoDigits (n : Nat) : Nat :=
  lastTwoDigitsPattern ((n - 2) % 4)

-- Theorem statement
theorem last_two_digits_of_7_pow_2017 :
  lastTwoDigits 2017 = 07 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_7_pow_2017_l166_16662


namespace NUMINAMATH_CALUDE_pasta_sauce_cost_l166_16612

/-- The cost of pasta sauce given grocery shopping conditions -/
theorem pasta_sauce_cost 
  (mustard_oil_quantity : ℝ) 
  (mustard_oil_price : ℝ) 
  (pasta_quantity : ℝ) 
  (pasta_price : ℝ) 
  (pasta_sauce_quantity : ℝ) 
  (initial_money : ℝ) 
  (money_left : ℝ) 
  (h1 : mustard_oil_quantity = 2) 
  (h2 : mustard_oil_price = 13) 
  (h3 : pasta_quantity = 3) 
  (h4 : pasta_price = 4) 
  (h5 : pasta_sauce_quantity = 1) 
  (h6 : initial_money = 50) 
  (h7 : money_left = 7) : 
  (initial_money - money_left - (mustard_oil_quantity * mustard_oil_price + pasta_quantity * pasta_price)) / pasta_sauce_quantity = 5 := by
sorry

end NUMINAMATH_CALUDE_pasta_sauce_cost_l166_16612


namespace NUMINAMATH_CALUDE_x_value_proof_l166_16611

def star_operation (a b : ℝ) : ℝ := a * b + a + b

theorem x_value_proof :
  ∀ x : ℝ, star_operation 3 x = 27 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l166_16611


namespace NUMINAMATH_CALUDE_intersection_M_N_l166_16603

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≠ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l166_16603


namespace NUMINAMATH_CALUDE_sams_initial_dimes_l166_16656

/-- The problem of determining Sam's initial number of dimes -/
theorem sams_initial_dimes :
  ∀ (initial_dimes current_dimes : ℕ),
    initial_dimes - 4 = current_dimes →
    current_dimes = 4 →
    initial_dimes = 8 := by
  sorry

end NUMINAMATH_CALUDE_sams_initial_dimes_l166_16656


namespace NUMINAMATH_CALUDE_mary_fruit_change_l166_16627

/-- The change Mary received after buying fruits -/
theorem mary_fruit_change (berries_cost peaches_cost payment : ℚ) 
  (h1 : berries_cost = 719 / 100)
  (h2 : peaches_cost = 683 / 100)
  (h3 : payment = 20) :
  payment - (berries_cost + peaches_cost) = 598 / 100 := by
  sorry

end NUMINAMATH_CALUDE_mary_fruit_change_l166_16627


namespace NUMINAMATH_CALUDE_constant_term_when_sum_is_64_l166_16609

-- Define the sum of binomial coefficients
def sum_binomial_coeffs (n : ℕ) : ℕ := 2^n

-- Define the constant term in the expansion
def constant_term (n : ℕ) : ℤ :=
  (-1)^(n/2) * (n.choose (n/2))

-- Theorem statement
theorem constant_term_when_sum_is_64 :
  ∃ n : ℕ, sum_binomial_coeffs n = 64 ∧ constant_term n = 15 :=
sorry

end NUMINAMATH_CALUDE_constant_term_when_sum_is_64_l166_16609


namespace NUMINAMATH_CALUDE_smallest_k_is_2010_l166_16621

/-- A sequence of natural numbers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧
  (∀ n, 1005 ∣ a n ∨ 1006 ∣ a n) ∧
  (∀ n, ¬(97 ∣ a n))

/-- The difference between consecutive terms is at most k -/
def BoundedDifference (a : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n, a (n + 1) - a n ≤ k

/-- The theorem stating the smallest possible k -/
theorem smallest_k_is_2010 :
  (∃ a, ValidSequence a ∧ BoundedDifference a 2010) ∧
  (∀ k < 2010, ¬∃ a, ValidSequence a ∧ BoundedDifference a k) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_is_2010_l166_16621


namespace NUMINAMATH_CALUDE_ramu_car_profit_percent_l166_16619

/-- Calculates the profit percent given the purchase price, repair cost, and selling price of a car. -/
def profit_percent (purchase_price repair_cost selling_price : ℚ) : ℚ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

/-- Theorem stating that the profit percent for Ramu's car transaction is 29.8% -/
theorem ramu_car_profit_percent :
  profit_percent 42000 8000 64900 = 29.8 :=
by sorry

end NUMINAMATH_CALUDE_ramu_car_profit_percent_l166_16619


namespace NUMINAMATH_CALUDE_part1_part2_l166_16626

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + a*x + 3

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x ∈ Set.Icc (-2) 2, f a x ≥ a) ↔ a ∈ Set.Icc (-7) 2 :=
sorry

-- Part 2
theorem part2 (x : ℝ) :
  (∀ a ∈ Set.Icc 4 6, f a x ≥ 0) ↔ 
  x ∈ Set.Iic (-3 - Real.sqrt 6) ∪ Set.Ici (-3 + Real.sqrt 6) :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l166_16626


namespace NUMINAMATH_CALUDE_johns_age_l166_16665

/-- Given that John is 30 years younger than his dad and the sum of their ages is 80 years, 
    prove that John is 25 years old. -/
theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l166_16665


namespace NUMINAMATH_CALUDE_next_perfect_square_sum_l166_16624

def children_ages : List ℕ := [1, 3, 5, 7, 9, 11, 13]

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def sum_ages (years_later : ℕ) : ℕ :=
  List.sum (List.map (· + years_later) children_ages)

theorem next_perfect_square_sum :
  (∃ (x : ℕ), x > 0 ∧ 
    is_perfect_square (sum_ages x) ∧
    (∀ y : ℕ, 0 < y ∧ y < x → ¬is_perfect_square (sum_ages y))) →
  (∃ (x : ℕ), x = 21 ∧ 
    is_perfect_square (sum_ages x) ∧
    (List.head! children_ages + x) + sum_ages x = 218) :=
by sorry

end NUMINAMATH_CALUDE_next_perfect_square_sum_l166_16624


namespace NUMINAMATH_CALUDE_well_digging_payment_l166_16641

/-- Calculates the total payment for a group of workers given their daily work hours and hourly rate -/
def totalPayment (numWorkers : ℕ) (dailyHours : List ℕ) (hourlyRate : ℕ) : ℕ :=
  numWorkers * (dailyHours.sum * hourlyRate)

/-- Proves that the total payment for 3 workers working 12, 10, 8, and 14 hours on four days at $15 per hour is $1980 -/
theorem well_digging_payment :
  totalPayment 3 [12, 10, 8, 14] 15 = 1980 := by
  sorry

end NUMINAMATH_CALUDE_well_digging_payment_l166_16641


namespace NUMINAMATH_CALUDE_room_width_l166_16691

/-- Proves that a rectangular room with given volume, length, and height has a specific width -/
theorem room_width (volume : ℝ) (length : ℝ) (height : ℝ) (width : ℝ) 
  (h_volume : volume = 10000)
  (h_length : length = 100)
  (h_height : height = 10)
  (h_relation : volume = length * width * height) :
  width = 10 := by
  sorry

end NUMINAMATH_CALUDE_room_width_l166_16691


namespace NUMINAMATH_CALUDE_square_division_impossible_l166_16659

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square in a 2D plane -/
structure Square where
  sideLength : ℝ
  center : Point

/-- Represents a division of a square by two internal points -/
structure SquareDivision where
  square : Square
  point1 : Point
  point2 : Point

/-- Checks if a point is inside a square -/
def isPointInsideSquare (s : Square) (p : Point) : Prop :=
  abs (p.x - s.center.x) ≤ s.sideLength / 2 ∧ abs (p.y - s.center.y) ≤ s.sideLength / 2

/-- Checks if a square division results in 9 equal parts -/
def isDividedIntoNineEqualParts (sd : SquareDivision) : Prop :=
  ∃ (areas : Finset ℝ), areas.card = 9 ∧ 
  (∀ a ∈ areas, a = sd.square.sideLength^2 / 9) ∧
  (isPointInsideSquare sd.square sd.point1) ∧
  (isPointInsideSquare sd.square sd.point2)

/-- Theorem stating that it's impossible to divide a square into 9 equal parts
    by connecting two internal points to its vertices -/
theorem square_division_impossible :
  ¬ ∃ (sd : SquareDivision), isDividedIntoNineEqualParts sd :=
sorry

end NUMINAMATH_CALUDE_square_division_impossible_l166_16659


namespace NUMINAMATH_CALUDE_morning_fliers_fraction_l166_16643

theorem morning_fliers_fraction (total : ℕ) (remaining : ℕ) : 
  total = 2500 → remaining = 1500 → 
  ∃ x : ℚ, x > 0 ∧ x < 1 ∧ 
  (1 - x) * total - (1 - x) * total / 4 = remaining ∧
  x = 1/5 := by
sorry

end NUMINAMATH_CALUDE_morning_fliers_fraction_l166_16643


namespace NUMINAMATH_CALUDE_price_reduction_problem_l166_16695

theorem price_reduction_problem (x : ℝ) : 
  (∀ (P : ℝ), P > 0 → 
    P * (1 - x / 100) * (1 - 20 / 100) = P * (1 - 40 / 100)) → 
  x = 25 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_problem_l166_16695


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l166_16604

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + x + 1 ≥ 0) ↔ a ≥ (1/4 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l166_16604


namespace NUMINAMATH_CALUDE_min_balls_to_draw_theorem_l166_16683

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The minimum number of balls to draw to ensure at least 15 of one color -/
def minBallsToDraw (counts : BallCounts) : Nat :=
  sorry

/-- The theorem stating the minimum number of balls to draw -/
theorem min_balls_to_draw_theorem (counts : BallCounts) 
  (h1 : counts.red = 28)
  (h2 : counts.green = 20)
  (h3 : counts.yellow = 13)
  (h4 : counts.blue = 19)
  (h5 : counts.white = 11)
  (h6 : counts.black = 9)
  (h_total : counts.red + counts.green + counts.yellow + counts.blue + counts.white + counts.black = 100) :
  minBallsToDraw counts = 76 :=
sorry

end NUMINAMATH_CALUDE_min_balls_to_draw_theorem_l166_16683


namespace NUMINAMATH_CALUDE_provisions_problem_l166_16638

/-- The initial number of men given the conditions of the problem -/
def initial_men : ℕ := 1000

/-- The number of days the provisions last for the initial group -/
def initial_days : ℕ := 20

/-- The number of additional men that join the group -/
def additional_men : ℕ := 650

/-- The number of days the provisions last after additional men join -/
def final_days : ℚ := 12121212121212121 / 1000000000000000

theorem provisions_problem :
  initial_men * initial_days = (initial_men + additional_men) * final_days :=
sorry

end NUMINAMATH_CALUDE_provisions_problem_l166_16638


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_81_l166_16667

theorem factor_x_squared_minus_81 (x : ℝ) : x^2 - 81 = (x - 9) * (x + 9) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_81_l166_16667


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_l166_16602

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_perpendicular_implication 
  (l m : Line) (α : Plane) :
  parallel m l → perpendicular m α → perpendicular l α :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_l166_16602


namespace NUMINAMATH_CALUDE_point_Q_in_first_quadrant_l166_16673

-- Define the conditions for point P
def fourth_quadrant (a b : ℝ) : Prop := a > 0 ∧ b < 0

-- Define the condition |a| > |b|
def magnitude_condition (a b : ℝ) : Prop := abs a > abs b

-- Define what it means for a point to be in the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Theorem statement
theorem point_Q_in_first_quadrant (a b : ℝ) 
  (h1 : fourth_quadrant a b) (h2 : magnitude_condition a b) : 
  first_quadrant (a + b) (a - b) := by
  sorry

end NUMINAMATH_CALUDE_point_Q_in_first_quadrant_l166_16673


namespace NUMINAMATH_CALUDE_factoring_expression_l166_16657

theorem factoring_expression (y : ℝ) : 5 * y * (y + 2) + 9 * (y + 2) = (5 * y + 9) * (y + 2) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l166_16657


namespace NUMINAMATH_CALUDE_nine_integer_chords_l166_16672

/-- Represents a circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distance_to_p : ℝ

/-- Counts the number of integer-length chords through P -/
def count_integer_chords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem -/
theorem nine_integer_chords :
  let c := CircleWithPoint.mk 20 12
  count_integer_chords c = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_integer_chords_l166_16672


namespace NUMINAMATH_CALUDE_polynomial_factorization_l166_16639

theorem polynomial_factorization (x : ℝ) : 
  x^8 - 8*x^6 + 24*x^4 - 32*x^2 + 16 = (x - Real.sqrt 2)^4 * (x + Real.sqrt 2)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l166_16639


namespace NUMINAMATH_CALUDE_relationship_between_exponents_l166_16675

theorem relationship_between_exponents 
  (a b c : ℝ) (x y q z : ℝ) 
  (h1 : a^x = c^q) (h2 : a^x = b^2) (h3 : c^q = b^2)
  (h4 : c^y = a^z) (h5 : c^y = b^3) (h6 : a^z = b^3) :
  x * q = y * z := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_exponents_l166_16675


namespace NUMINAMATH_CALUDE_d_value_for_four_roots_l166_16654

/-- The polynomial Q(x) -/
def Q (d : ℝ) (x : ℝ) : ℝ := (x^2 - 3*x + 5) * (x^2 - d*x + 7) * (x^2 - 6*x + 18)

/-- The number of distinct roots of Q(x) -/
def distinctRoots (d : ℝ) : ℕ := sorry

/-- Theorem stating that |d| = 9 when Q(x) has exactly 4 distinct roots -/
theorem d_value_for_four_roots :
  ∃ d : ℝ, distinctRoots d = 4 ∧ |d| = 9 := by sorry

end NUMINAMATH_CALUDE_d_value_for_four_roots_l166_16654


namespace NUMINAMATH_CALUDE_hoseok_persimmons_l166_16661

theorem hoseok_persimmons (jungkook_persimmons hoseok_persimmons : ℕ) : 
  jungkook_persimmons = 25 → 
  3 * hoseok_persimmons = jungkook_persimmons - 4 →
  hoseok_persimmons = 7 := by
sorry

end NUMINAMATH_CALUDE_hoseok_persimmons_l166_16661


namespace NUMINAMATH_CALUDE_linear_function_properties_l166_16614

def f (x : ℝ) : ℝ := -2 * x - 4

theorem linear_function_properties :
  (f (-1) = -2) ∧
  (f 0 ≠ -2) ∧
  (∀ x, x < -2 → f x > 0) ∧
  (∀ x y, x ≥ 0 → y ≥ 0 → f x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l166_16614


namespace NUMINAMATH_CALUDE_modulus_of_z_l166_16625

theorem modulus_of_z (z : ℂ) (h : z * (4 - 3*I) = 1) : Complex.abs z = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l166_16625


namespace NUMINAMATH_CALUDE_problem_solution_l166_16629

/-- The function f(x) as defined in the problem -/
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := (1/2) * (t * Real.log (x + 2) - Real.log (x - 2))

/-- The function F(x) as defined in the problem -/
noncomputable def F (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := a * Real.log (x - 1) - f t x

/-- Theorem stating the main results of the problem -/
theorem problem_solution :
  ∃ (t : ℝ),
    (∀ x : ℝ, f t x ≥ f t 4) ∧
    (t = 3) ∧
    (∀ x ∈ Set.Icc 3 7, f t x ≤ f t 7) ∧
    (∀ a : ℝ, (∀ x > 2, Monotone (F a t)) ↔ a ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l166_16629


namespace NUMINAMATH_CALUDE_total_people_in_park_l166_16686

/-- The number of lines formed by people in the park -/
def num_lines : ℕ := 4

/-- The number of people in each line -/
def people_per_line : ℕ := 8

/-- The total number of people doing gymnastics in the park -/
def total_people : ℕ := num_lines * people_per_line

theorem total_people_in_park : total_people = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_park_l166_16686


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l166_16644

theorem roots_sum_of_squares (a b : ℝ) : 
  (∀ x, x^2 - 8*x + 8 = 0 ↔ x = a ∨ x = b) → a^2 + b^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l166_16644


namespace NUMINAMATH_CALUDE_max_root_sum_l166_16658

theorem max_root_sum (a b c : ℝ) : 
  (a^3 - 4 * Real.sqrt 3 * a^2 + 13 * a - 2 * Real.sqrt 3 = 0) →
  (b^3 - 4 * Real.sqrt 3 * b^2 + 13 * b - 2 * Real.sqrt 3 = 0) →
  (c^3 - 4 * Real.sqrt 3 * c^2 + 13 * c - 2 * Real.sqrt 3 = 0) →
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  max (a + b - c) (max (a - b + c) (-a + b + c)) = 2 * Real.sqrt 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_root_sum_l166_16658


namespace NUMINAMATH_CALUDE_first_equation_is_double_root_second_equation_double_root_condition_l166_16687

/-- Definition of a double root equation -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ (a * x^2 + b * x + c = 0) ∧ (a * y^2 + b * y + c = 0) ∧ (x = 2 * y ∨ y = 2 * x)

/-- Theorem for the first part of the problem -/
theorem first_equation_is_double_root : is_double_root_equation 1 (-6) 8 := by sorry

/-- Theorem for the second part of the problem -/
theorem second_equation_double_root_condition (n : ℝ) : 
  is_double_root_equation 1 (-8 - n) (8 * n) → n = 4 ∨ n = 16 := by sorry

end NUMINAMATH_CALUDE_first_equation_is_double_root_second_equation_double_root_condition_l166_16687


namespace NUMINAMATH_CALUDE_symmetrical_circle_l166_16674

/-- Given a circle with equation x² + y² + 2x = 0, 
    its symmetrical circle with respect to the y-axis 
    has the equation x² + y² - 2x = 0 -/
theorem symmetrical_circle (x y : ℝ) : 
  (x^2 + y^2 + 2*x = 0) → 
  ∃ (x' y' : ℝ), (x'^2 + y'^2 - 2*x' = 0 ∧ 
                  x' = -x ∧ 
                  y' = y) :=
by sorry

end NUMINAMATH_CALUDE_symmetrical_circle_l166_16674


namespace NUMINAMATH_CALUDE_sqrt_x_minus_2_real_iff_x_geq_2_l166_16628

theorem sqrt_x_minus_2_real_iff_x_geq_2 (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_2_real_iff_x_geq_2_l166_16628


namespace NUMINAMATH_CALUDE_total_arrangements_l166_16613

/-- Represents the three elective math courses -/
inductive Course
| MatrixTransformation
| InfoSecCrypto
| SwitchCircuits

/-- Represents a teacher and their teaching capabilities -/
structure Teacher where
  id : Nat
  canTeach : Course → Bool

/-- The pool of available teachers -/
def teacherPool : Finset Teacher := sorry

/-- Teachers who can teach only Matrix and Transformation -/
def matrixOnlyTeachers : Finset Teacher := sorry

/-- Teachers who can teach only Information Security and Cryptography -/
def cryptoOnlyTeachers : Finset Teacher := sorry

/-- Teachers who can teach only Switch Circuits and Boolean Algebra -/
def switchOnlyTeachers : Finset Teacher := sorry

/-- Teachers who can teach all three courses -/
def versatileTeachers : Finset Teacher := sorry

/-- A valid selection of teachers for the courses -/
def isValidSelection (selection : Finset Teacher) : Prop := sorry

/-- The number of different valid arrangements -/
def numArrangements : Nat := sorry

theorem total_arrangements :
  (Finset.card teacherPool = 10) →
  (Finset.card matrixOnlyTeachers = 3) →
  (Finset.card cryptoOnlyTeachers = 2) →
  (Finset.card switchOnlyTeachers = 3) →
  (Finset.card versatileTeachers = 2) →
  (∀ s : Finset Teacher, isValidSelection s → Finset.card s = 9) →
  (∀ c : Course, ∀ s : Finset Teacher, isValidSelection s →
    Finset.card (s.filter (fun t => t.canTeach c)) = 3) →
  numArrangements = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_arrangements_l166_16613


namespace NUMINAMATH_CALUDE_jake_has_eleven_apples_l166_16679

-- Define the number of peaches and apples for Steven
def steven_peaches : ℕ := 9
def steven_apples : ℕ := 8

-- Define Jake's peaches and apples in relation to Steven's
def jake_peaches : ℕ := steven_peaches - 13
def jake_apples : ℕ := steven_apples + 3

-- Theorem to prove
theorem jake_has_eleven_apples : jake_apples = 11 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_eleven_apples_l166_16679


namespace NUMINAMATH_CALUDE_roots_of_equation_l166_16670

theorem roots_of_equation : 
  ∃ (x₁ x₂ : ℝ), (x₁ = -1 ∧ x₂ = 0) ∧ 
  (∀ x : ℝ, (x + 1) * x = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l166_16670


namespace NUMINAMATH_CALUDE_orange_count_indeterminate_l166_16678

/-- Represents Philip's fruit collection -/
structure FruitCollection where
  banana_count : ℕ
  banana_groups : ℕ
  bananas_per_group : ℕ
  orange_groups : ℕ

/-- Predicate to check if the banana count is consistent with the groups and bananas per group -/
def banana_count_consistent (collection : FruitCollection) : Prop :=
  collection.banana_count = collection.banana_groups * collection.bananas_per_group

/-- Theorem stating that the number of oranges cannot be determined -/
theorem orange_count_indeterminate (collection : FruitCollection)
  (h1 : collection.banana_count = 290)
  (h2 : collection.banana_groups = 2)
  (h3 : collection.bananas_per_group = 145)
  (h4 : collection.orange_groups = 93)
  (h5 : banana_count_consistent collection) :
  ¬∃ (orange_count : ℕ), ∀ (other_collection : FruitCollection),
    collection.banana_count = other_collection.banana_count ∧
    collection.banana_groups = other_collection.banana_groups ∧
    collection.bananas_per_group = other_collection.bananas_per_group ∧
    collection.orange_groups = other_collection.orange_groups →
    orange_count = (other_collection.orange_groups : ℕ) * (orange_count / other_collection.orange_groups) :=
sorry

end NUMINAMATH_CALUDE_orange_count_indeterminate_l166_16678


namespace NUMINAMATH_CALUDE_maria_towels_problem_l166_16634

theorem maria_towels_problem (green_towels white_towels given_to_mother : ℝ) 
  (h1 : green_towels = 124.5)
  (h2 : white_towels = 67.7)
  (h3 : given_to_mother = 85.35) :
  green_towels + white_towels - given_to_mother = 106.85 := by
sorry

end NUMINAMATH_CALUDE_maria_towels_problem_l166_16634


namespace NUMINAMATH_CALUDE_sum_x_y_equals_two_l166_16642

theorem sum_x_y_equals_two (x y : ℝ) 
  (h1 : (4 : ℝ) ^ x = 16 ^ (y + 1))
  (h2 : (5 : ℝ) ^ (2 * y) = 25 ^ (x - 2)) : 
  x + y = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_two_l166_16642


namespace NUMINAMATH_CALUDE_backpack_cost_is_fifteen_l166_16689

def total_spent : ℝ := 32
def pens_cost : ℝ := 1
def pencils_cost : ℝ := 1
def notebook_cost : ℝ := 3
def notebook_count : ℕ := 5

def backpack_cost : ℝ := total_spent - (pens_cost + pencils_cost + notebook_cost * notebook_count)

theorem backpack_cost_is_fifteen : backpack_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_backpack_cost_is_fifteen_l166_16689


namespace NUMINAMATH_CALUDE_hash_five_neg_one_l166_16693

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * (y + 2) + x * y

-- Theorem statement
theorem hash_five_neg_one : hash 5 (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_hash_five_neg_one_l166_16693


namespace NUMINAMATH_CALUDE_f_satisfies_equation_l166_16660

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x + 1

-- State the theorem
theorem f_satisfies_equation : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_equation_l166_16660


namespace NUMINAMATH_CALUDE_alpha_necessary_not_sufficient_l166_16631

-- Define the conditions
def α (x : ℝ) : Prop := x^2 = 4
def β (x : ℝ) : Prop := x = 2

-- State the theorem
theorem alpha_necessary_not_sufficient :
  (∀ x, β x → α x) ∧ (∃ x, α x ∧ ¬β x) := by sorry

end NUMINAMATH_CALUDE_alpha_necessary_not_sufficient_l166_16631


namespace NUMINAMATH_CALUDE_alpha_beta_equivalence_l166_16633

theorem alpha_beta_equivalence (α β : ℝ) :
  (α > β) ↔ (α + Real.sin α * Real.cos β > β + Real.sin β * Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_equivalence_l166_16633


namespace NUMINAMATH_CALUDE_parabola_through_fixed_point_l166_16647

-- Define the line equation as a function of a
def line_equation (a x y : ℝ) : Prop := (a - 1) * x - y + 2 * a + 1 = 0

-- Define the fixed point P
def fixed_point : ℝ × ℝ := (-2, 3)

-- Define the two possible parabola equations
def parabola1 (x y : ℝ) : Prop := y^2 = -9/2 * x
def parabola2 (x y : ℝ) : Prop := x^2 = 4/3 * y

-- State the theorem
theorem parabola_through_fixed_point :
  (∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2)) →
  (parabola1 (fixed_point.1) (fixed_point.2) ∨ parabola2 (fixed_point.1) (fixed_point.2)) :=
sorry

end NUMINAMATH_CALUDE_parabola_through_fixed_point_l166_16647


namespace NUMINAMATH_CALUDE_total_age_is_877_l166_16697

def family_gathering (T : ℕ) : Prop :=
  ∃ (father mother brother sister elder_cousin younger_cousin grandmother uncle aunt kaydence : ℕ),
    father = 60 ∧
    mother = father - 2 ∧
    brother = father / 2 ∧
    sister = 40 ∧
    elder_cousin = brother + 2 * sister ∧
    younger_cousin = elder_cousin / 2 + 3 ∧
    grandmother = 3 * mother - 5 ∧
    uncle = 5 * younger_cousin - 10 ∧
    aunt = 2 * mother + 7 ∧
    5 * kaydence = 2 * aunt ∧
    T = father + mother + brother + sister + elder_cousin + younger_cousin + grandmother + uncle + aunt + kaydence

theorem total_age_is_877 : family_gathering 877 := by
  sorry

end NUMINAMATH_CALUDE_total_age_is_877_l166_16697


namespace NUMINAMATH_CALUDE_n_in_interval_l166_16671

def is_repeating_decimal (d : ℚ) (period : ℕ) : Prop :=
  ∃ (k : ℕ), d * 10^period - d.floor = k / (10^period - 1)

theorem n_in_interval (n : ℕ) (hn : n < 1000) 
  (h1 : is_repeating_decimal (1 / n) 3)
  (h2 : is_repeating_decimal (1 / (n + 4)) 6) :
  n ∈ Set.Icc 1 150 := by
  sorry

end NUMINAMATH_CALUDE_n_in_interval_l166_16671


namespace NUMINAMATH_CALUDE_star_commutative_star_not_distributive_l166_16676

/-- Binary operation ⋆ -/
def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

/-- Commutativity of ⋆ -/
theorem star_commutative : ∀ x y : ℝ, star x y = star y x := by sorry

/-- Non-distributivity of ⋆ over addition -/
theorem star_not_distributive : ¬(∀ x y z : ℝ, star x (y + z) = star x y + star x z) := by sorry

end NUMINAMATH_CALUDE_star_commutative_star_not_distributive_l166_16676


namespace NUMINAMATH_CALUDE_max_a_for_monotonic_f_l166_16618

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x

-- State the theorem
theorem max_a_for_monotonic_f :
  ∀ a : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x ≤ y → f a x ≤ f a y) →
  a ≤ 0 ∧ ∀ b : ℝ, (∀ x y : ℝ, 0 ≤ x ∧ x ≤ y → f b x ≤ f b y) → b ≤ a :=
sorry

end NUMINAMATH_CALUDE_max_a_for_monotonic_f_l166_16618


namespace NUMINAMATH_CALUDE_profit_increase_l166_16669

theorem profit_increase (m : ℝ) : 
  (m + 8) / 0.92 = m + 10 → m = 15 := by
  sorry

end NUMINAMATH_CALUDE_profit_increase_l166_16669


namespace NUMINAMATH_CALUDE_absent_workers_l166_16635

theorem absent_workers (total_workers : ℕ) (original_days : ℕ) (actual_days : ℕ) 
  (h1 : total_workers = 42)
  (h2 : original_days = 12)
  (h3 : actual_days = 14) :
  ∃ (absent : ℕ), 
    absent = 6 ∧ 
    (total_workers * original_days = (total_workers - absent) * actual_days) :=
by sorry

end NUMINAMATH_CALUDE_absent_workers_l166_16635


namespace NUMINAMATH_CALUDE_prob_one_sunny_day_l166_16684

/-- The probability of exactly one sunny day in a three-day festival --/
theorem prob_one_sunny_day (p_sunny : ℝ) (p_not_sunny : ℝ) :
  p_sunny = 0.1 →
  p_not_sunny = 0.9 →
  3 * (p_sunny * p_not_sunny * p_not_sunny) = 0.243 :=
by sorry

end NUMINAMATH_CALUDE_prob_one_sunny_day_l166_16684


namespace NUMINAMATH_CALUDE_degree_of_specific_monomial_l166_16606

/-- The degree of a monomial is the sum of the exponents of its variables -/
def degree_of_monomial (m : Polynomial ℚ) : ℕ :=
  sorry

/-- The monomial 2/3 * a^3 * b -/
def monomial : Polynomial ℚ :=
  sorry

theorem degree_of_specific_monomial :
  degree_of_monomial monomial = 4 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_specific_monomial_l166_16606


namespace NUMINAMATH_CALUDE_joseph_cards_percentage_l166_16622

theorem joseph_cards_percentage (initial_cards : ℕ) 
  (brother_fraction : ℚ) (friend_cards : ℕ) : 
  initial_cards = 16 →
  brother_fraction = 3/8 →
  friend_cards = 2 →
  (initial_cards - (initial_cards * brother_fraction).floor - friend_cards) / initial_cards * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_joseph_cards_percentage_l166_16622


namespace NUMINAMATH_CALUDE_tech_ownership_1995_l166_16623

/-- The percentage of families owning computers, tablets, and smartphones in City X in 1995 -/
def tech_ownership (pc_1992 : ℝ) (pc_increase_1993 : ℝ) (family_increase_1993 : ℝ)
                   (tablet_adoption_1994 : ℝ) (smartphone_adoption_1995 : ℝ) : ℝ :=
  let pc_1993 := pc_1992 * (1 + pc_increase_1993)
  let pc_tablet_1994 := pc_1993 * tablet_adoption_1994
  pc_tablet_1994 * smartphone_adoption_1995

theorem tech_ownership_1995 :
  tech_ownership 0.6 0.5 0.03 0.4 0.3 = 0.108 := by
  sorry

end NUMINAMATH_CALUDE_tech_ownership_1995_l166_16623


namespace NUMINAMATH_CALUDE_x_value_possibilities_l166_16616

theorem x_value_possibilities (x y p q : ℝ) (h1 : y ≠ 0) (h2 : q ≠ 0) 
  (h3 : |x / y| < |p| / q^2) :
  ∃ (x_neg x_zero x_pos : ℝ), 
    (x_neg < 0 ∧ |x_neg / y| < |p| / q^2) ∧
    (x_zero = 0 ∧ |x_zero / y| < |p| / q^2) ∧
    (x_pos > 0 ∧ |x_pos / y| < |p| / q^2) :=
by sorry

end NUMINAMATH_CALUDE_x_value_possibilities_l166_16616


namespace NUMINAMATH_CALUDE_lucas_raspberry_candies_l166_16677

-- Define the variables
def original_raspberry : ℕ := sorry
def original_lemon : ℕ := sorry

-- Define the conditions
axiom initial_ratio : original_raspberry = 3 * original_lemon
axiom after_giving_away : original_raspberry - 5 = 4 * (original_lemon - 5)

-- Theorem to prove
theorem lucas_raspberry_candies : original_raspberry = 45 := by
  sorry

end NUMINAMATH_CALUDE_lucas_raspberry_candies_l166_16677


namespace NUMINAMATH_CALUDE_special_sequence_a6_l166_16645

/-- A sequence where a₂ = 3, a₄ = 15, and {aₙ + 1} is a geometric sequence -/
def special_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 3 ∧ a 4 = 15 ∧ ∃ q : ℝ, ∀ n : ℕ, (a (n + 1) + 1) = (a n + 1) * q

theorem special_sequence_a6 (a : ℕ → ℝ) (h : special_sequence a) : a 6 = 63 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_a6_l166_16645


namespace NUMINAMATH_CALUDE_total_savings_three_months_l166_16620

def savings (n : ℕ) : ℕ := 10 + 30 * n

theorem total_savings_three_months :
  savings 0 + savings 1 + savings 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_three_months_l166_16620
