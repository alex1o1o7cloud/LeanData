import Mathlib

namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l2417_241773

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 12 n = Nat.choose 12 (2*n - 3)) → (n = 3 ∨ n = 5) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l2417_241773


namespace NUMINAMATH_CALUDE_solve_equation_y_l2417_241741

theorem solve_equation_y (y : ℝ) (hy : y ≠ 0) :
  (7 * y)^4 = (14 * y)^3 ↔ y = 8 / 7 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_y_l2417_241741


namespace NUMINAMATH_CALUDE_square_of_sum_l2417_241736

theorem square_of_sum (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_l2417_241736


namespace NUMINAMATH_CALUDE_polynomial_constant_term_l2417_241737

/-- A polynomial of degree 4 with integer coefficients -/
structure Polynomial4 where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ

/-- The polynomial g(x) = x^4 + px^3 + qx^2 + rx + s -/
def g (poly : Polynomial4) (x : ℤ) : ℤ :=
  x^4 + poly.p * x^3 + poly.q * x^2 + poly.r * x + poly.s

/-- A polynomial has all negative integer roots -/
def has_all_negative_integer_roots (poly : Polynomial4) : Prop :=
  ∃ (a b c d : ℤ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    ∀ (x : ℤ), g poly x = (x + a) * (x + b) * (x + c) * (x + d)

theorem polynomial_constant_term (poly : Polynomial4) :
  has_all_negative_integer_roots poly →
  poly.p + poly.q + poly.r + poly.s = 8091 →
  poly.s = 8064 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_constant_term_l2417_241737


namespace NUMINAMATH_CALUDE_boxes_with_neither_pens_nor_pencils_l2417_241771

theorem boxes_with_neither_pens_nor_pencils 
  (total_boxes : ℕ) 
  (boxes_with_pencils : ℕ) 
  (boxes_with_pens : ℕ) 
  (boxes_with_both : ℕ) 
  (h1 : total_boxes = 15)
  (h2 : boxes_with_pencils = 7)
  (h3 : boxes_with_pens = 4)
  (h4 : boxes_with_both = 3) :
  total_boxes - (boxes_with_pencils + boxes_with_pens - boxes_with_both) = 7 :=
by sorry

end NUMINAMATH_CALUDE_boxes_with_neither_pens_nor_pencils_l2417_241771


namespace NUMINAMATH_CALUDE_kims_weekly_production_l2417_241714

/-- Represents Kim's daily sweater production for a week --/
structure WeeklyKnitting where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total number of sweaters knit in a week --/
def totalSweaters (week : WeeklyKnitting) : ℕ :=
  week.monday + week.tuesday + week.wednesday + week.thursday + week.friday

/-- Theorem stating that Kim's total sweater production for the given week is 34 --/
theorem kims_weekly_production :
  ∃ (week : WeeklyKnitting),
    week.monday = 8 ∧
    week.tuesday = week.monday + 2 ∧
    week.wednesday = week.tuesday - 4 ∧
    week.thursday = week.tuesday - 4 ∧
    week.friday = week.monday / 2 ∧
    totalSweaters week = 34 := by
  sorry


end NUMINAMATH_CALUDE_kims_weekly_production_l2417_241714


namespace NUMINAMATH_CALUDE_prime_square_mod_24_l2417_241749

theorem prime_square_mod_24 (p : ℕ) (hp : Prime p) (hp_gt_3 : p > 3) :
  p^2 % 24 = 1 :=
sorry

end NUMINAMATH_CALUDE_prime_square_mod_24_l2417_241749


namespace NUMINAMATH_CALUDE_cat_weight_problem_l2417_241758

theorem cat_weight_problem (total_weight : ℕ) (cat1_weight : ℕ) (cat2_weight : ℕ) 
  (h1 : total_weight = 13)
  (h2 : cat1_weight = 2)
  (h3 : cat2_weight = 7) :
  total_weight - cat1_weight - cat2_weight = 4 := by
  sorry

end NUMINAMATH_CALUDE_cat_weight_problem_l2417_241758


namespace NUMINAMATH_CALUDE_decimal_to_binary_87_l2417_241718

theorem decimal_to_binary_87 : 
  (87 : ℕ).digits 2 = [1, 1, 1, 0, 1, 0, 1] :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_87_l2417_241718


namespace NUMINAMATH_CALUDE_power_product_eq_product_of_powers_l2417_241720

theorem power_product_eq_product_of_powers (a b : ℝ) : (a * b)^2 = a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_eq_product_of_powers_l2417_241720


namespace NUMINAMATH_CALUDE_alpha_squared_greater_than_beta_squared_l2417_241750

theorem alpha_squared_greater_than_beta_squared
  (α β : ℝ)
  (h1 : α ∈ Set.Icc (-π/2) (π/2))
  (h2 : β ∈ Set.Icc (-π/2) (π/2))
  (h3 : α * Real.sin α - β * Real.sin β > 0) :
  α^2 > β^2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_squared_greater_than_beta_squared_l2417_241750


namespace NUMINAMATH_CALUDE_remainder_equality_l2417_241764

/-- Represents a natural number as a list of its digits in reverse order -/
def DigitList := List Nat

/-- Converts a natural number to its digit list representation -/
def toDigitList (n : Nat) : DigitList :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : DigitList) : DigitList :=
      if m = 0 then acc
      else aux (m / 10) ((m % 10) :: acc)
    aux n []

/-- Pairs digits from right to left, allowing the leftmost pair to be a single digit -/
def pairDigits (dl : DigitList) : List Nat :=
  match dl with
  | [] => []
  | [x] => [x]
  | x :: y :: rest => (x + 10 * y) :: pairDigits rest

/-- Sums a list of natural numbers -/
def sumList (l : List Nat) : Nat := l.foldl (·+·) 0

/-- The main theorem statement -/
theorem remainder_equality (n : Nat) :
  n % 99 = (sumList (pairDigits (toDigitList n))) % 99 := by
  sorry

end NUMINAMATH_CALUDE_remainder_equality_l2417_241764


namespace NUMINAMATH_CALUDE_complex_modulus_of_iz_eq_one_l2417_241794

theorem complex_modulus_of_iz_eq_one (z : ℂ) (h : Complex.I * z = 1) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_of_iz_eq_one_l2417_241794


namespace NUMINAMATH_CALUDE_garden_area_l2417_241734

theorem garden_area (perimeter : ℝ) (length width : ℝ) : 
  perimeter = 2 * (length + width) →
  length = 3 * width →
  perimeter = 84 →
  length * width = 330.75 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l2417_241734


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2417_241772

theorem max_value_of_expression (x : ℝ) :
  x^4 / (x^8 + 4*x^6 - 8*x^4 + 16*x^2 + 64) ≤ 1/24 ∧
  ∃ y : ℝ, y^4 / (y^8 + 4*y^6 - 8*y^4 + 16*y^2 + 64) = 1/24 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2417_241772


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2417_241774

theorem arithmetic_sequence_sum (n : ℕ) : 
  (Finset.range (n + 3)).sum (fun i => 2 * i + 3) = n^2 + 8*n + 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2417_241774


namespace NUMINAMATH_CALUDE_sqrt_four_fourth_powers_sum_l2417_241728

theorem sqrt_four_fourth_powers_sum : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_fourth_powers_sum_l2417_241728


namespace NUMINAMATH_CALUDE_sum_sqrt_squared_pairs_geq_sqrt2_l2417_241783

theorem sum_sqrt_squared_pairs_geq_sqrt2 (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) : 
  Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + a^2) ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_sqrt_squared_pairs_geq_sqrt2_l2417_241783


namespace NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_sufficient_not_necessary_l2417_241787

theorem ac_squared_gt_bc_squared_sufficient_not_necessary
  (a b c : ℝ) (h : c ≠ 0) :
  (∀ a b, a * c^2 > b * c^2 → a > b) ∧
  ¬(∀ a b, a > b → a * c^2 > b * c^2) :=
sorry

end NUMINAMATH_CALUDE_ac_squared_gt_bc_squared_sufficient_not_necessary_l2417_241787


namespace NUMINAMATH_CALUDE_range_of_a_l2417_241789

open Set Real

theorem range_of_a (p q : Prop) (h : p ∧ q) : 
  (∀ x ∈ Icc 1 2, x^2 ≥ a) ∧ (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2417_241789


namespace NUMINAMATH_CALUDE_transformed_area_doubled_l2417_241795

-- Define a function representing the area under a curve
noncomputable def areaUnderCurve (f : ℝ → ℝ) (a b : ℝ) : ℝ := sorry

-- Define the original function g
variable (g : ℝ → ℝ)

-- Define the interval [a, b] over which we're measuring the area
variable (a b : ℝ)

-- Theorem statement
theorem transformed_area_doubled 
  (h : areaUnderCurve g a b = 15) : 
  areaUnderCurve (fun x ↦ 2 * g (x + 3)) a b = 30 := by
  sorry

end NUMINAMATH_CALUDE_transformed_area_doubled_l2417_241795


namespace NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l2417_241751

theorem smallest_lcm_with_gcd_5 :
  ∃ (m n : ℕ), 
    1000 ≤ m ∧ m < 10000 ∧
    1000 ≤ n ∧ n < 10000 ∧
    Nat.gcd m n = 5 ∧
    Nat.lcm m n = 201000 ∧
    ∀ (p q : ℕ), 
      1000 ≤ p ∧ p < 10000 ∧
      1000 ≤ q ∧ q < 10000 ∧
      Nat.gcd p q = 5 →
      Nat.lcm p q ≥ 201000 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_lcm_with_gcd_5_l2417_241751


namespace NUMINAMATH_CALUDE_plane_distance_ratio_l2417_241715

theorem plane_distance_ratio (total_distance bus_distance : ℝ) 
  (h1 : total_distance = 1800)
  (h2 : bus_distance = 720)
  : (total_distance - (2/3 * bus_distance + bus_distance)) / total_distance = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_plane_distance_ratio_l2417_241715


namespace NUMINAMATH_CALUDE_inequality_proof_l2417_241786

theorem inequality_proof (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > -a*b ∧ -a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2417_241786


namespace NUMINAMATH_CALUDE_mr_martin_bagels_l2417_241755

/-- Represents the purchase of coffee and bagels -/
structure Purchase where
  coffee : ℕ
  bagels : ℕ
  total : ℚ

/-- Represents the cost of items -/
structure Prices where
  coffee : ℚ
  bagel : ℚ

def mrs_martin : Purchase := { coffee := 3, bagels := 2, total := 12.75 }
def mr_martin (x : ℕ) : Purchase := { coffee := 2, bagels := x, total := 14 }

def prices : Prices := { coffee := 3.25, bagel := 1.5 }

theorem mr_martin_bagels :
  ∃ x : ℕ, 
    (mr_martin x).total = (mr_martin x).coffee • prices.coffee + (mr_martin x).bagels • prices.bagel ∧
    mrs_martin.total = mrs_martin.coffee • prices.coffee + mrs_martin.bagels • prices.bagel ∧
    x = 5 := by
  sorry

#check mr_martin_bagels

end NUMINAMATH_CALUDE_mr_martin_bagels_l2417_241755


namespace NUMINAMATH_CALUDE_clarissa_photos_eq_14_l2417_241765

/-- The number of slots in the photo album -/
def total_slots : ℕ := 40

/-- The number of photos Cristina brings -/
def cristina_photos : ℕ := 7

/-- The number of photos John brings -/
def john_photos : ℕ := 10

/-- The number of photos Sarah brings -/
def sarah_photos : ℕ := 9

/-- The number of photos Clarissa needs to bring -/
def clarissa_photos : ℕ := total_slots - (cristina_photos + john_photos + sarah_photos)

theorem clarissa_photos_eq_14 : clarissa_photos = 14 := by
  sorry

end NUMINAMATH_CALUDE_clarissa_photos_eq_14_l2417_241765


namespace NUMINAMATH_CALUDE_polynomial_root_sum_squares_l2417_241759

theorem polynomial_root_sum_squares (a b c t : ℝ) : 
  (∀ x : ℝ, x^3 - 6*x^2 + 8*x - 2 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  t = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  t^4 - 12*t^2 - 4*t = -4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_squares_l2417_241759


namespace NUMINAMATH_CALUDE_investment_interest_calculation_l2417_241762

/-- Calculates the total interest earned on an investment -/
def total_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- The problem statement -/
theorem investment_interest_calculation :
  let principal : ℝ := 2000
  let rate : ℝ := 0.08
  let time : ℕ := 5
  abs (total_interest principal rate time - 938.66) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_calculation_l2417_241762


namespace NUMINAMATH_CALUDE_sin_sum_simplification_l2417_241716

theorem sin_sum_simplification :
  Real.sin (119 * π / 180) * Real.sin (181 * π / 180) - 
  Real.sin (91 * π / 180) * Real.sin (29 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_simplification_l2417_241716


namespace NUMINAMATH_CALUDE_negative_y_ceil_floor_product_l2417_241742

theorem negative_y_ceil_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 :=
by sorry

end NUMINAMATH_CALUDE_negative_y_ceil_floor_product_l2417_241742


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l2417_241746

theorem equidistant_point_x_coordinate : 
  ∃ (x : ℝ), 
    (x^2 + 6*x + 9 = x^2 + 25) ∧ 
    (∀ (y : ℝ), (y^2 + 6*y + 9 = y^2 + 25) → y = x) ∧
    x = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l2417_241746


namespace NUMINAMATH_CALUDE_route_number_theorem_l2417_241780

/-- Represents a digit on a seven-segment display -/
inductive Digit
| Zero | One | Two | Three | Four | Five | Six | Seven | Eight | Nine

/-- Represents a three-digit number -/
structure ThreeDigitNumber :=
  (hundreds : Digit)
  (tens : Digit)
  (units : Digit)

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  match n.hundreds, n.tens, n.units with
  | Digit.Three, Digit.Five, Digit.One => 351
  | Digit.Three, Digit.Five, Digit.Four => 354
  | Digit.Three, Digit.Five, Digit.Seven => 357
  | Digit.Three, Digit.Six, Digit.One => 361
  | Digit.Three, Digit.Six, Digit.Seven => 367
  | Digit.Three, Digit.Eight, Digit.One => 381
  | Digit.Three, Digit.Nine, Digit.One => 391
  | Digit.Three, Digit.Nine, Digit.Seven => 397
  | Digit.Eight, Digit.Five, Digit.One => 851
  | Digit.Nine, Digit.Five, Digit.One => 951
  | Digit.Nine, Digit.Five, Digit.Seven => 957
  | Digit.Nine, Digit.Six, Digit.One => 961
  | Digit.Nine, Digit.Nine, Digit.One => 991
  | _, _, _ => 0  -- Default case, should not occur in our problem

/-- The set of possible route numbers -/
def possibleRouteNumbers : Set Nat :=
  {351, 354, 357, 361, 367, 381, 391, 397, 851, 951, 957, 961, 991}

/-- The theorem stating that the displayed number 351 with two non-working segments
    can only result in the numbers in the possibleRouteNumbers set -/
theorem route_number_theorem (n : ThreeDigitNumber) 
    (h : n.toNat ∈ possibleRouteNumbers) : 
    ∃ (broken_segments : Nat), broken_segments ≤ 2 ∧ 
    n.toNat ∈ possibleRouteNumbers :=
  sorry

end NUMINAMATH_CALUDE_route_number_theorem_l2417_241780


namespace NUMINAMATH_CALUDE_farm_animals_l2417_241744

/-- The number of chickens on the farm -/
def num_chickens : ℕ := 49

/-- The number of ducks on the farm -/
def num_ducks : ℕ := 37

/-- The number of rabbits on the farm -/
def num_rabbits : ℕ := 21

theorem farm_animals :
  (num_ducks + num_rabbits = num_chickens + 9) →
  num_rabbits = 21 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l2417_241744


namespace NUMINAMATH_CALUDE_symmetry_line_equation_l2417_241763

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two points are symmetric about a line -/
def symmetric_about (P Q : Point) (l : Line) : Prop :=
  -- Definition of symmetry (to be implemented)
  sorry

/-- The problem statement -/
theorem symmetry_line_equation :
  let P : Point := ⟨3, 2⟩
  let Q : Point := ⟨1, 4⟩
  let l : Line := ⟨1, -1, 1⟩  -- Represents x - y + 1 = 0
  symmetric_about P Q l → l = ⟨1, -1, 1⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetry_line_equation_l2417_241763


namespace NUMINAMATH_CALUDE_function_equation_solution_l2417_241719

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, (f x + x * y) * f (x - 3 * y) + (f y + x * y) * f (3 * x - y) = (f (x + y))^2) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l2417_241719


namespace NUMINAMATH_CALUDE_emily_total_score_l2417_241752

/-- A game with five rounds and specific scoring rules. -/
structure Game where
  round1_score : ℤ
  round2_score : ℤ
  round3_score : ℤ
  round4_score : ℤ
  round5_score : ℤ

/-- Emily's game performance -/
def emily_game : Game where
  round1_score := 16
  round2_score := 32
  round3_score := -27
  round4_score := 46
  round5_score := 12

/-- Calculate the total score for a game -/
def total_score (g : Game) : ℤ :=
  g.round1_score + g.round2_score + g.round3_score + (g.round4_score * 2) + (g.round5_score / 3)

/-- Theorem stating that Emily's total score is 117 -/
theorem emily_total_score : total_score emily_game = 117 := by
  sorry


end NUMINAMATH_CALUDE_emily_total_score_l2417_241752


namespace NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l2417_241733

theorem least_integer_greater_than_sqrt_500 : 
  ∃ n : ℕ, (n : ℝ) > Real.sqrt 500 ∧ ∀ m : ℕ, (m : ℝ) > Real.sqrt 500 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_integer_greater_than_sqrt_500_l2417_241733


namespace NUMINAMATH_CALUDE_t_is_perfect_square_l2417_241711

theorem t_is_perfect_square (n : ℕ+) (t : ℕ+) (h : t = 2 + 2 * Real.sqrt (1 + 12 * n.val ^ 2)) :
  ∃ (x : ℕ), t = x ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_t_is_perfect_square_l2417_241711


namespace NUMINAMATH_CALUDE_fraction_comparison_l2417_241784

theorem fraction_comparison : -8/21 > -3/7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2417_241784


namespace NUMINAMATH_CALUDE_fraction_simplification_l2417_241757

theorem fraction_simplification (y : ℝ) (h : y = 3) : 
  (y^8 + 18*y^4 + 81) / (y^4 + 9) = 90 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2417_241757


namespace NUMINAMATH_CALUDE_dividend_calculation_l2417_241706

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 50)
  (h2 : quotient = 70)
  (h3 : remainder = 20) :
  divisor * quotient + remainder = 3520 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2417_241706


namespace NUMINAMATH_CALUDE_square_of_six_y_minus_two_l2417_241792

theorem square_of_six_y_minus_two (y : ℝ) (h : 3 * y^2 + 6 = 2 * y + 10) : (6 * y - 2)^2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_square_of_six_y_minus_two_l2417_241792


namespace NUMINAMATH_CALUDE_sum_of_e_values_l2417_241770

theorem sum_of_e_values (e : ℝ) : (|2 - e| = 5) → (∃ (e₁ e₂ : ℝ), (|2 - e₁| = 5 ∧ |2 - e₂| = 5 ∧ e₁ + e₂ = 4)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_e_values_l2417_241770


namespace NUMINAMATH_CALUDE_triangle_proof_l2417_241793

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given equation for the triangle -/
def triangle_equation (t : Triangle) : Prop :=
  t.b^2 - (2 * Real.sqrt 3 / 3) * t.b * t.c * Real.sin t.A + t.c^2 = t.a^2

theorem triangle_proof (t : Triangle) 
  (h_eq : triangle_equation t) 
  (h_b : t.b = 2) 
  (h_c : t.c = 3) :
  t.A = π/3 ∧ t.a = Real.sqrt 7 ∧ Real.sin (2*t.B - t.A) = 3*Real.sqrt 3/14 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_proof_l2417_241793


namespace NUMINAMATH_CALUDE_sum_of_digits_of_square_ones_l2417_241748

def ones (n : ℕ) : ℕ := 
  (10^n - 1) / 9

def sum_of_digits (m : ℕ) : ℕ :=
  if m = 0 then 0 else m % 10 + sum_of_digits (m / 10)

theorem sum_of_digits_of_square_ones (n : ℕ) : 
  sum_of_digits ((ones n)^2) = n^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_square_ones_l2417_241748


namespace NUMINAMATH_CALUDE_power_of_three_mod_eight_l2417_241777

theorem power_of_three_mod_eight : 3^2028 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eight_l2417_241777


namespace NUMINAMATH_CALUDE_flowers_per_bouquet_l2417_241766

theorem flowers_per_bouquet 
  (total_flowers : ℕ) 
  (wilted_flowers : ℕ) 
  (num_bouquets : ℕ) 
  (h1 : total_flowers = 45) 
  (h2 : wilted_flowers = 35) 
  (h3 : num_bouquets = 2) : 
  (total_flowers - wilted_flowers) / num_bouquets = 5 := by
sorry

end NUMINAMATH_CALUDE_flowers_per_bouquet_l2417_241766


namespace NUMINAMATH_CALUDE_jackson_percentage_difference_l2417_241732

/-- Represents the count of birds seen by a person -/
structure BirdCount where
  robins : ℕ
  cardinals : ℕ
  blueJays : ℕ
  goldfinches : ℕ
  starlings : ℕ

/-- Calculates the total number of birds seen by a person -/
def totalBirds (count : BirdCount) : ℕ :=
  count.robins + count.cardinals + count.blueJays + count.goldfinches + count.starlings

/-- Calculates the percentage difference from the average -/
def percentageDifference (individual : ℕ) (average : ℚ) : ℚ :=
  ((individual : ℚ) - average) / average * 100

theorem jackson_percentage_difference :
  let gabrielle := BirdCount.mk 7 5 4 3 6
  let chase := BirdCount.mk 4 3 4 2 1
  let maria := BirdCount.mk 5 3 2 4 7
  let jackson := BirdCount.mk 6 2 3 5 2
  let total := totalBirds gabrielle + totalBirds chase + totalBirds maria + totalBirds jackson
  let average : ℚ := (total : ℚ) / 4
  abs (percentageDifference (totalBirds jackson) average - (-7.69)) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_jackson_percentage_difference_l2417_241732


namespace NUMINAMATH_CALUDE_cameron_questions_total_l2417_241775

/-- Represents a tour group with a number of regular tourists and inquisitive tourists -/
structure TourGroup where
  regular_tourists : ℕ
  inquisitive_tourists : ℕ

/-- Calculates the number of questions answered for a tour group -/
def questions_answered (group : TourGroup) (questions_per_tourist : ℕ) : ℕ :=
  group.regular_tourists * questions_per_tourist + 
  group.inquisitive_tourists * (questions_per_tourist * 3)

theorem cameron_questions_total : 
  let questions_per_tourist := 2
  let tour1 := TourGroup.mk 6 0
  let tour2 := TourGroup.mk 11 0
  let tour3 := TourGroup.mk 7 1
  let tour4 := TourGroup.mk 7 0
  questions_answered tour1 questions_per_tourist +
  questions_answered tour2 questions_per_tourist +
  questions_answered tour3 questions_per_tourist +
  questions_answered tour4 questions_per_tourist = 68 := by
  sorry

end NUMINAMATH_CALUDE_cameron_questions_total_l2417_241775


namespace NUMINAMATH_CALUDE_negation_equivalence_l2417_241721

theorem negation_equivalence :
  (¬ ∃ x : ℝ, (x < 1 ∨ x^2 ≥ 4)) ↔ (∀ x : ℝ, (x ≥ 1 ∧ x^2 < 4)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2417_241721


namespace NUMINAMATH_CALUDE_fraction_equality_l2417_241713

theorem fraction_equality : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5000 = 750.0000000000001 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2417_241713


namespace NUMINAMATH_CALUDE_factory_works_ten_hours_per_day_l2417_241745

/-- Represents a chocolate factory with its production parameters -/
structure ChocolateFactory where
  production_rate : ℕ  -- candies per hour
  order_size : ℕ       -- total candies to produce
  days_to_complete : ℕ -- number of days to complete the order

/-- Calculates the number of hours the factory works each day -/
def hours_per_day (factory : ChocolateFactory) : ℚ :=
  (factory.order_size / factory.production_rate : ℚ) / factory.days_to_complete

/-- Theorem stating that for the given parameters, the factory works 10 hours per day -/
theorem factory_works_ten_hours_per_day :
  let factory := ChocolateFactory.mk 50 4000 8
  hours_per_day factory = 10 := by
  sorry

end NUMINAMATH_CALUDE_factory_works_ten_hours_per_day_l2417_241745


namespace NUMINAMATH_CALUDE_binary_expression_equals_expected_result_l2417_241704

/-- Converts a list of binary digits to a natural number. -/
def binary_to_nat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 2 * acc + d) 0

/-- Calculates the result of the given binary expression. -/
def binary_expression_result : Nat :=
  let a := binary_to_nat [1, 0, 1, 1, 0]
  let b := binary_to_nat [1, 0, 1, 0]
  let c := binary_to_nat [1, 1, 1, 0, 0]
  let d := binary_to_nat [1, 1, 1, 0]
  a + b - c + d

/-- The expected result in binary. -/
def expected_result : Nat :=
  binary_to_nat [0, 1, 1, 1, 0]

theorem binary_expression_equals_expected_result :
  binary_expression_result = expected_result := by
  sorry

end NUMINAMATH_CALUDE_binary_expression_equals_expected_result_l2417_241704


namespace NUMINAMATH_CALUDE_arithmetic_computation_l2417_241769

theorem arithmetic_computation : 6^2 + 2*(5) - 4^2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l2417_241769


namespace NUMINAMATH_CALUDE_election_majority_proof_l2417_241790

theorem election_majority_proof (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 455 → 
  winning_percentage = 70 / 100 → 
  ⌊(2 * winning_percentage - 1) * total_votes⌋ = 182 := by
  sorry

end NUMINAMATH_CALUDE_election_majority_proof_l2417_241790


namespace NUMINAMATH_CALUDE_parallelogram_xy_product_l2417_241730

/-- A parallelogram with side lengths given in terms of x and y -/
structure Parallelogram (x y : ℝ) :=
  (ef : ℝ)
  (fg : ℝ)
  (gh : ℝ)
  (he : ℝ)
  (ef_eq : ef = 42)
  (fg_eq : fg = 4 * y^3)
  (gh_eq : gh = 2 * x + 10)
  (he_eq : he = 32)
  (opposite_sides_equal : ef = gh ∧ fg = he)

/-- The product of x and y in the given parallelogram is 32 -/
theorem parallelogram_xy_product (x y : ℝ) (p : Parallelogram x y) :
  x * y = 32 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_xy_product_l2417_241730


namespace NUMINAMATH_CALUDE_two_red_cards_probability_l2417_241785

/-- The probability of drawing two red cards in succession from a deck of 100 cards
    containing 50 red cards and 50 black cards, without replacement. -/
theorem two_red_cards_probability (total_cards : ℕ) (red_cards : ℕ) (black_cards : ℕ) 
    (h1 : total_cards = 100)
    (h2 : red_cards = 50)
    (h3 : black_cards = 50)
    (h4 : total_cards = red_cards + black_cards) :
    (red_cards : ℚ) / total_cards * ((red_cards - 1) : ℚ) / (total_cards - 1) = 49 / 198 := by
  sorry

end NUMINAMATH_CALUDE_two_red_cards_probability_l2417_241785


namespace NUMINAMATH_CALUDE_fiftieth_term_of_specific_sequence_l2417_241781

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 50th term of the arithmetic sequence with first term 3 and common difference 7 is 346 -/
theorem fiftieth_term_of_specific_sequence : 
  arithmeticSequenceTerm 3 7 50 = 346 := by
  sorry

#check fiftieth_term_of_specific_sequence

end NUMINAMATH_CALUDE_fiftieth_term_of_specific_sequence_l2417_241781


namespace NUMINAMATH_CALUDE_water_purifier_filtration_layers_l2417_241799

theorem water_purifier_filtration_layers (initial_impurities : ℝ) (target_impurities : ℝ) 
  (filter_efficiency : ℝ) (h1 : initial_impurities = 80) (h2 : target_impurities = 2) 
  (h3 : filter_efficiency = 1/3) : 
  ∃ n : ℕ, (initial_impurities * (1 - filter_efficiency)^n ≤ target_impurities ∧ 
  ∀ m : ℕ, m < n → initial_impurities * (1 - filter_efficiency)^m > target_impurities) :=
sorry

end NUMINAMATH_CALUDE_water_purifier_filtration_layers_l2417_241799


namespace NUMINAMATH_CALUDE_unique_solution_power_equation_l2417_241717

theorem unique_solution_power_equation (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃! x : ℝ, a^x + b^x = c^x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_equation_l2417_241717


namespace NUMINAMATH_CALUDE_integral_proof_l2417_241722

open Real

noncomputable def f (x : ℝ) : ℝ := 
  -20/27 * ((1 + x^(3/4))^(1/5) / x^(3/20))^9

theorem integral_proof (x : ℝ) (h : x > 0) : 
  deriv f x = (((1 + x^(3/4))^4)^(1/5)) / (x^2 * x^(7/20)) :=
by sorry

end NUMINAMATH_CALUDE_integral_proof_l2417_241722


namespace NUMINAMATH_CALUDE_allocation_schemes_eq_36_l2417_241702

/-- Represents the number of teachers --/
def num_teachers : ℕ := 4

/-- Represents the number of schools --/
def num_schools : ℕ := 3

/-- Represents the condition that each school must receive at least one teacher --/
def min_teachers_per_school : ℕ := 1

/-- Calculates the number of ways to allocate teachers to schools --/
def allocation_schemes (n_teachers : ℕ) (n_schools : ℕ) (min_per_school : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of allocation schemes is 36 --/
theorem allocation_schemes_eq_36 : 
  allocation_schemes num_teachers num_schools min_teachers_per_school = 36 := by
  sorry

end NUMINAMATH_CALUDE_allocation_schemes_eq_36_l2417_241702


namespace NUMINAMATH_CALUDE_paint_project_total_l2417_241740

/-- The total amount of paint needed for a project, given the amount left from a previous project and the amount that needs to be bought. -/
def total_paint (left_over : ℕ) (to_buy : ℕ) : ℕ :=
  left_over + to_buy

/-- Theorem stating that the total amount of paint needed is 333 liters. -/
theorem paint_project_total :
  total_paint 157 176 = 333 := by
  sorry

end NUMINAMATH_CALUDE_paint_project_total_l2417_241740


namespace NUMINAMATH_CALUDE_right_triangle_sin_value_l2417_241735

theorem right_triangle_sin_value (A B C : ℝ) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) :
  B = π/2 → 2 * Real.sin A = 3 * Real.cos A → Real.sin A = 3 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_value_l2417_241735


namespace NUMINAMATH_CALUDE_factor_of_quadratic_l2417_241753

theorem factor_of_quadratic (m x : ℤ) : 
  (∃ k : ℤ, (m - x) * k = m^2 - 5*m - 24) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_factor_of_quadratic_l2417_241753


namespace NUMINAMATH_CALUDE_number_of_female_employees_l2417_241768

/-- Given a company with employees, prove that the number of female employees is 500 -/
theorem number_of_female_employees 
  (E : ℕ) -- Total number of employees
  (F : ℕ) -- Number of female employees
  (M : ℕ) -- Number of male employees
  (h1 : E = F + M) -- Total employees is sum of female and male employees
  (h2 : (2 : ℚ) / 5 * E = 200 + (2 : ℚ) / 5 * M) -- Equation for total managers
  (h3 : (200 : ℚ) = F - (2 : ℚ) / 5 * M) -- Equation for female managers
  : F = 500 := by
  sorry

end NUMINAMATH_CALUDE_number_of_female_employees_l2417_241768


namespace NUMINAMATH_CALUDE_fourth_drawn_is_92_l2417_241701

/-- Systematic sampling function -/
def systematicSample (populationSize : ℕ) (sampleSize : ℕ) (firstDrawn : ℕ) (groupNumber : ℕ) : ℕ :=
  firstDrawn + (groupNumber - 1) * (populationSize / sampleSize)

/-- Theorem: The fourth drawn number in the given systematic sampling scenario is 92 -/
theorem fourth_drawn_is_92 :
  systematicSample 600 20 2 4 = 92 := by
  sorry

#eval systematicSample 600 20 2 4

end NUMINAMATH_CALUDE_fourth_drawn_is_92_l2417_241701


namespace NUMINAMATH_CALUDE_price_increase_percentage_l2417_241710

theorem price_increase_percentage (initial_price : ℝ) : 
  initial_price > 0 →
  let new_egg_price := initial_price * 1.1
  let new_apple_price := initial_price * 1.02
  let initial_total := initial_price * 2
  let new_total := new_egg_price + new_apple_price
  (new_total - initial_total) / initial_total = 0.04 :=
by
  sorry

#check price_increase_percentage

end NUMINAMATH_CALUDE_price_increase_percentage_l2417_241710


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l2417_241729

theorem sufficient_condition_for_inequality (a : ℝ) : 
  (a ≥ 5) → (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) ∧ 
  ∃ b : ℝ, b < 5 ∧ (∀ x ∈ Set.Icc 1 2, x^2 - b ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l2417_241729


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2417_241723

theorem complex_equation_solution (z : ℂ) :
  z + Complex.abs z = 2 + Complex.I → z = 3/4 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2417_241723


namespace NUMINAMATH_CALUDE_lucy_money_theorem_l2417_241778

def lucy_money_problem (initial_amount : ℚ) : ℚ :=
  let remaining_after_loss := initial_amount * (1 - 1/3)
  let spent := remaining_after_loss * (1/4)
  remaining_after_loss - spent

theorem lucy_money_theorem :
  lucy_money_problem 30 = 15 := by sorry

end NUMINAMATH_CALUDE_lucy_money_theorem_l2417_241778


namespace NUMINAMATH_CALUDE_expression_equality_l2417_241798

theorem expression_equality : (1/4)⁻¹ - |Real.sqrt 3 - 2| + 2 * (-Real.sqrt 3) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2417_241798


namespace NUMINAMATH_CALUDE_bus_seating_capacity_l2417_241726

/-- Represents the seating capacity of a bus with specific seat arrangements. -/
structure BusSeating where
  left_seats : Nat
  right_seats : Nat
  people_per_seat : Nat
  back_seat_capacity : Nat

/-- Calculates the total number of people who can sit in the bus. -/
def total_seating_capacity (bus : BusSeating) : Nat :=
  (bus.left_seats + bus.right_seats) * bus.people_per_seat + bus.back_seat_capacity

/-- Theorem stating the total seating capacity of the bus with given conditions. -/
theorem bus_seating_capacity : 
  ∀ (bus : BusSeating), 
    bus.left_seats = 15 → 
    bus.right_seats = bus.left_seats - 3 →
    bus.people_per_seat = 3 →
    bus.back_seat_capacity = 11 →
    total_seating_capacity bus = 92 :=
by
  sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_l2417_241726


namespace NUMINAMATH_CALUDE_chemistry_mean_marks_l2417_241779

/-- Proves that the mean mark in the second section is 60 given the conditions of the problem -/
theorem chemistry_mean_marks (n₁ n₂ n₃ n₄ : ℕ) (m₁ m₃ m₄ : ℚ) (overall_avg : ℚ) :
  n₁ = 60 →
  n₂ = 35 →
  n₃ = 45 →
  n₄ = 42 →
  m₁ = 50 →
  m₃ = 55 →
  m₄ = 45 →
  overall_avg = 52005494505494504/1000000000000000 →
  ∃ m₂ : ℚ, m₂ = 60 ∧ 
    overall_avg * (n₁ + n₂ + n₃ + n₄ : ℚ) = n₁ * m₁ + n₂ * m₂ + n₃ * m₃ + n₄ * m₄ :=
by sorry


end NUMINAMATH_CALUDE_chemistry_mean_marks_l2417_241779


namespace NUMINAMATH_CALUDE_sum_of_max_marks_is_1300_l2417_241731

/-- Given the conditions for three tests (Math, Science, and English) in an examination,
    this theorem proves that the sum of maximum marks for all three tests is 1300. -/
theorem sum_of_max_marks_is_1300 
  (math_pass_percent : ℝ) 
  (math_marks_obtained : ℕ) 
  (math_marks_failed_by : ℕ)
  (science_pass_percent : ℝ) 
  (science_marks_obtained : ℕ) 
  (science_marks_failed_by : ℕ)
  (english_pass_percent : ℝ) 
  (english_marks_obtained : ℕ) 
  (english_marks_failed_by : ℕ)
  (h_math_percent : math_pass_percent = 0.3)
  (h_science_percent : science_pass_percent = 0.5)
  (h_english_percent : english_pass_percent = 0.4)
  (h_math_marks : math_marks_obtained = 80 ∧ math_marks_failed_by = 100)
  (h_science_marks : science_marks_obtained = 120 ∧ science_marks_failed_by = 80)
  (h_english_marks : english_marks_obtained = 60 ∧ english_marks_failed_by = 60) :
  ↑((math_marks_obtained + math_marks_failed_by) / math_pass_percent +
    (science_marks_obtained + science_marks_failed_by) / science_pass_percent +
    (english_marks_obtained + english_marks_failed_by) / english_pass_percent) = 1300 :=
by sorry


end NUMINAMATH_CALUDE_sum_of_max_marks_is_1300_l2417_241731


namespace NUMINAMATH_CALUDE_length_of_EF_l2417_241707

/-- A rectangle intersecting a circle -/
structure RectangleIntersectingCircle where
  /-- Length of AB -/
  AB : ℝ
  /-- Length of BC -/
  BC : ℝ
  /-- Length of DE -/
  DE : ℝ
  /-- Length of EF -/
  EF : ℝ

/-- Theorem stating the length of EF in the given configuration -/
theorem length_of_EF (r : RectangleIntersectingCircle) 
  (h1 : r.AB = 4)
  (h2 : r.BC = 5)
  (h3 : r.DE = 3) :
  r.EF = 7 := by
  sorry

#check length_of_EF

end NUMINAMATH_CALUDE_length_of_EF_l2417_241707


namespace NUMINAMATH_CALUDE_share_multiple_l2417_241760

theorem share_multiple (total : ℕ) (c_share : ℕ) (k : ℕ) : 
  total = 880 → c_share = 160 → 
  (∃ (a_share b_share : ℕ), 
    a_share + b_share + c_share = total ∧ 
    4 * a_share = k * b_share ∧ 
    k * b_share = 10 * c_share) → 
  k = 5 := by
sorry

end NUMINAMATH_CALUDE_share_multiple_l2417_241760


namespace NUMINAMATH_CALUDE_selling_price_is_twenty_l2417_241782

/-- Calculates the selling price per phone given the total number of phones, 
    total cost, and desired profit ratio. -/
def selling_price_per_phone (total_phones : ℕ) (total_cost : ℚ) (profit_ratio : ℚ) : ℚ :=
  let cost_per_phone := total_cost / total_phones
  let profit_per_phone := (total_cost * profit_ratio) / total_phones
  cost_per_phone + profit_per_phone

/-- Theorem stating that the selling price per phone is $20 given the problem conditions. -/
theorem selling_price_is_twenty :
  selling_price_per_phone 200 3000 (1/3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_is_twenty_l2417_241782


namespace NUMINAMATH_CALUDE_divisible_by_four_sum_consecutive_odds_l2417_241727

theorem divisible_by_four_sum_consecutive_odds (a : ℤ) : ∃ (x y : ℤ), 
  4 * a = x + y ∧ Odd x ∧ Odd y ∧ y = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_four_sum_consecutive_odds_l2417_241727


namespace NUMINAMATH_CALUDE_adult_attraction_cost_is_four_l2417_241703

/-- Represents the cost structure and family composition for a park visit -/
structure ParkVisit where
  entrance_fee : ℕ
  child_attraction_fee : ℕ
  num_children : ℕ
  num_parents : ℕ
  num_grandparents : ℕ
  total_cost : ℕ

/-- Calculates the cost of an adult attraction ticket given the park visit details -/
def adult_attraction_cost (visit : ParkVisit) : ℕ :=
  let total_people := visit.num_children + visit.num_parents + visit.num_grandparents
  let entrance_cost := total_people * visit.entrance_fee
  let children_attraction_cost := visit.num_children * visit.child_attraction_fee
  let adult_attraction_total := visit.total_cost - entrance_cost - children_attraction_cost
  let num_adults := visit.num_parents + visit.num_grandparents
  adult_attraction_total / num_adults

theorem adult_attraction_cost_is_four : 
  adult_attraction_cost ⟨5, 2, 4, 2, 1, 55⟩ = 4 := by
  sorry

end NUMINAMATH_CALUDE_adult_attraction_cost_is_four_l2417_241703


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l2417_241776

theorem sum_of_A_and_B (A B : ℕ) (h1 : 3 * 7 = 7 * A) (h2 : 7 * A = B) : A + B = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l2417_241776


namespace NUMINAMATH_CALUDE_sandys_phone_bill_l2417_241761

theorem sandys_phone_bill (kim_age : ℕ) (sandy_age : ℕ) (sandy_bill : ℕ) :
  kim_age = 10 →
  sandy_age + 2 = 3 * (kim_age + 2) →
  sandy_bill = 10 * sandy_age →
  sandy_bill = 340 := by
  sorry

end NUMINAMATH_CALUDE_sandys_phone_bill_l2417_241761


namespace NUMINAMATH_CALUDE_desk_chair_relationship_l2417_241708

def chair_heights : List ℝ := [37.0, 40.0, 42.0, 45.0]
def desk_heights : List ℝ := [70.0, 74.8, 78.0, 82.8]

def linear_function (x : ℝ) : ℝ := 1.6 * x + 10.8

theorem desk_chair_relationship :
  ∀ (i : Fin 4),
    linear_function (chair_heights.get i) = desk_heights.get i :=
by sorry

end NUMINAMATH_CALUDE_desk_chair_relationship_l2417_241708


namespace NUMINAMATH_CALUDE_train_crossing_time_l2417_241739

def train_length : ℝ := 450
def train_speed_kmh : ℝ := 54

theorem train_crossing_time : 
  ∀ (platform_length : ℝ) (train_speed_ms : ℝ),
    platform_length = train_length →
    train_speed_ms = train_speed_kmh * (1000 / 3600) →
    (2 * train_length) / train_speed_ms = 60 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2417_241739


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l2417_241791

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l2417_241791


namespace NUMINAMATH_CALUDE_shekars_math_marks_l2417_241747

def science_marks : ℕ := 65
def social_studies_marks : ℕ := 82
def english_marks : ℕ := 67
def biology_marks : ℕ := 85
def average_marks : ℕ := 75
def total_subjects : ℕ := 5

theorem shekars_math_marks :
  ∃ math_marks : ℕ,
    math_marks = average_marks * total_subjects - (science_marks + social_studies_marks + english_marks + biology_marks) :=
by
  sorry

end NUMINAMATH_CALUDE_shekars_math_marks_l2417_241747


namespace NUMINAMATH_CALUDE_f_g_properties_l2417_241725

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x + a * Real.log x

def g (a b : ℝ) (x : ℝ) : ℝ := f a x + 1/2 * x^2 - b * x

def tangent_perpendicular (a : ℝ) : Prop :=
  (deriv (f a) 1) * (-1/2) = -1

def has_decreasing_interval (a b : ℝ) : Prop :=
  ∃ x y, x < y ∧ ∀ z ∈ Set.Ioo x y, (deriv (g a b) z) < 0

def extreme_points (a b : ℝ) (x₁ x₂ : ℝ) : Prop :=
  x₁ < x₂ ∧ (deriv (g a b) x₁) = 0 ∧ (deriv (g a b) x₂) = 0

theorem f_g_properties (a b : ℝ) (x₁ x₂ : ℝ) 
  (h1 : tangent_perpendicular a)
  (h2 : extreme_points a b x₁ x₂)
  (h3 : b ≥ 7/2) :
  a = 1 ∧ 
  (has_decreasing_interval a b → b > 3) ∧
  (g a b x₁ - g a b x₂ ≥ 15/8 - 2 * Real.log 2) :=
sorry

end NUMINAMATH_CALUDE_f_g_properties_l2417_241725


namespace NUMINAMATH_CALUDE_selling_price_calculation_l2417_241754

theorem selling_price_calculation (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  cost_price = 22500 →
  discount_rate = 0.1 →
  profit_rate = 0.08 →
  (1 - discount_rate) * (cost_price * (1 + profit_rate)) = 24300 →
  cost_price * (1 + profit_rate) / (1 - discount_rate) = 27000 := by
  sorry

#check selling_price_calculation

end NUMINAMATH_CALUDE_selling_price_calculation_l2417_241754


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2417_241797

theorem quadratic_inequality_range : 
  ∀ x : ℝ, x^2 - 7*x + 12 < 0 → 
  ∃ y : ℝ, y ∈ Set.Ioo 42 56 ∧ y = x^2 + 7*x + 12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2417_241797


namespace NUMINAMATH_CALUDE_valid_fractions_characterization_l2417_241743

def is_valid_fraction (n d : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ d ≥ 10 ∧ d < 100 ∧
  (∃ (a b c : Nat), (a > 0 ∧ b > 0 ∧ c > 0) ∧
    ((n = 10 * a + b ∧ d = 10 * a + c ∧ n * c = d * b) ∨
     (n = 10 * a + b ∧ d = 10 * c + b ∧ n * c = d * a) ∨
     (n = 10 * a + c ∧ d = 10 * b + c ∧ n * b = d * a)))

theorem valid_fractions_characterization :
  {p : Nat × Nat | is_valid_fraction p.1 p.2} =
  {(26, 65), (16, 64), (19, 95), (49, 98)} := by
  sorry

end NUMINAMATH_CALUDE_valid_fractions_characterization_l2417_241743


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l2417_241712

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l2417_241712


namespace NUMINAMATH_CALUDE_possible_sum_BC_ge_90_l2417_241709

/-- Represents an acute triangle with angles A, B, and C --/
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute : A < 90 ∧ B < 90 ∧ C < 90
  sum_180 : A + B + C = 180
  ordered : A > B ∧ B > C

/-- 
Theorem: In an acute triangle with angles A > B > C, 
it's possible for the sum of B and C to be greater than or equal to 90°
--/
theorem possible_sum_BC_ge_90 (t : AcuteTriangle) : 
  ∃ (x y z : Real), x > y ∧ y > z ∧ x < 90 ∧ y < 90 ∧ z < 90 ∧ x + y + z = 180 ∧ y + z ≥ 90 := by
  sorry

end NUMINAMATH_CALUDE_possible_sum_BC_ge_90_l2417_241709


namespace NUMINAMATH_CALUDE_negative_of_difference_l2417_241767

theorem negative_of_difference (a b : ℝ) : -(a - b) = -a + b := by sorry

end NUMINAMATH_CALUDE_negative_of_difference_l2417_241767


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2417_241724

theorem circle_area_ratio (R_A R_B : ℝ) (h : R_A > 0 ∧ R_B > 0) : 
  (60 : ℝ) / 360 * (2 * Real.pi * R_A) = (40 : ℝ) / 360 * (2 * Real.pi * R_B) → 
  (R_A^2 * Real.pi) / (R_B^2 * Real.pi) = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2417_241724


namespace NUMINAMATH_CALUDE_power_calculation_l2417_241796

theorem power_calculation : (2 : ℝ)^2021 * (-1/2 : ℝ)^2022 = 1/2 := by sorry

end NUMINAMATH_CALUDE_power_calculation_l2417_241796


namespace NUMINAMATH_CALUDE_max_workers_l2417_241738

/-- Represents the number of workers on the small field -/
def n : ℕ := sorry

/-- The total number of workers in the crew -/
def total_workers : ℕ := 2 * n + 4

/-- The area of the small field -/
def small_area : ℝ := sorry

/-- The area of the large field -/
def large_area : ℝ := 2 * small_area

/-- The time taken to complete work on the small field -/
def small_field_time : ℝ := sorry

/-- The time taken to complete work on the large field -/
def large_field_time : ℝ := sorry

/-- The condition that the small field is still being worked on when the large field is finished -/
axiom work_condition : small_field_time > large_field_time

/-- The theorem stating the maximum number of workers in the crew -/
theorem max_workers : total_workers ≤ 10 := by sorry

end NUMINAMATH_CALUDE_max_workers_l2417_241738


namespace NUMINAMATH_CALUDE_geometric_sequence_proof_l2417_241788

theorem geometric_sequence_proof (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  a 1 + a 3 = 10 →                                  -- first given condition
  a 2 + a 4 = 5 →                                   -- second given condition
  ∀ n, a n = 2^(4 - n) :=                           -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_proof_l2417_241788


namespace NUMINAMATH_CALUDE_import_tax_problem_l2417_241756

/-- The import tax rate as a decimal -/
def tax_rate : ℝ := 0.07

/-- The threshold above which the tax is applied -/
def tax_threshold : ℝ := 1000

/-- The amount of tax paid -/
def tax_paid : ℝ := 112.70

/-- The total value of the item -/
def total_value : ℝ := 2610

theorem import_tax_problem :
  tax_rate * (total_value - tax_threshold) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_import_tax_problem_l2417_241756


namespace NUMINAMATH_CALUDE_smallest_number_proof_l2417_241700

def smallest_number : ℕ := 271562

theorem smallest_number_proof :
  smallest_number = 271562 ∧
  ∃ k : ℕ, (smallest_number - 18) = k * lcm 14 (lcm 26 28) ∧
  k = 746 ∧
  ∀ y : ℕ, y < smallest_number →
    ¬(∃ m : ℕ, (y - 18) = m * lcm 14 (lcm 26 28) ∧ m = 746) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l2417_241700


namespace NUMINAMATH_CALUDE_water_evaporation_proof_l2417_241705

-- Define the initial composition of solution y
def solution_y_composition : ℝ := 0.3

-- Define the initial amount of solution y
def initial_amount : ℝ := 6

-- Define the amount of solution y added after evaporation
def amount_added : ℝ := 2

-- Define the amount remaining after evaporation
def amount_remaining : ℝ := 4

-- Define the new composition of the solution
def new_composition : ℝ := 0.4

-- Define the amount of water evaporated
def water_evaporated : ℝ := 2

-- Theorem statement
theorem water_evaporation_proof :
  let initial_liquid_x := solution_y_composition * initial_amount
  let added_liquid_x := solution_y_composition * amount_added
  let total_liquid_x := initial_liquid_x + added_liquid_x
  let new_total_amount := total_liquid_x / new_composition
  new_total_amount = amount_remaining + amount_added →
  water_evaporated = amount_added :=
by sorry

end NUMINAMATH_CALUDE_water_evaporation_proof_l2417_241705
