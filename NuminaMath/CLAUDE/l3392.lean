import Mathlib

namespace NUMINAMATH_CALUDE_smallest_x_satisfying_equations_l3392_339298

theorem smallest_x_satisfying_equations : 
  ∃ x : ℝ, x = -12 ∧ 
    abs (x - 3) = 15 ∧ 
    abs (x + 2) = 10 ∧ 
    ∀ y : ℝ, (abs (y - 3) = 15 ∧ abs (y + 2) = 10) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_equations_l3392_339298


namespace NUMINAMATH_CALUDE_triangle_area_l3392_339278

/-- The area of a triangle with vertices at (2, -3), (8, 1), and (2, 3) is 18 square units. -/
theorem triangle_area : Real := by
  -- Define the vertices of the triangle
  let A : (ℝ × ℝ) := (2, -3)
  let B : (ℝ × ℝ) := (8, 1)
  let C : (ℝ × ℝ) := (2, 3)

  -- Calculate the area of the triangle
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)

  -- Prove that the area is equal to 18
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3392_339278


namespace NUMINAMATH_CALUDE_purchase_price_calculation_l3392_339267

theorem purchase_price_calculation (markup : ℝ) (overhead_percentage : ℝ) (net_profit : ℝ) :
  markup = 45 ∧ 
  overhead_percentage = 0.20 ∧ 
  net_profit = 12 →
  ∃ purchase_price : ℝ, 
    markup = overhead_percentage * purchase_price + net_profit ∧
    purchase_price = 165 := by
  sorry

end NUMINAMATH_CALUDE_purchase_price_calculation_l3392_339267


namespace NUMINAMATH_CALUDE_parabola_normal_intersection_l3392_339270

/-- Given a parabola y = x^2, for any point (x₀, y₀) on the parabola,
    if the normal line at this point intersects the y-axis at (0, y₁),
    then y₁ - y₀ = 1/2 -/
theorem parabola_normal_intersection (x₀ y₀ y₁ : ℝ) : 
  y₀ = x₀^2 →  -- point (x₀, y₀) is on the parabola
  (∃ k : ℝ, k * (x - x₀) = y - y₀ ∧  -- equation of the normal line
            k = -(2 * x₀)⁻¹ ∧        -- slope of the normal line
            y₁ = k * (-x₀) + y₀) →   -- y₁ is the y-intercept of the normal line
  y₁ - y₀ = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_normal_intersection_l3392_339270


namespace NUMINAMATH_CALUDE_lisa_savings_analysis_l3392_339241

def lisa_savings : Fin 6 → ℝ
  | 0 => 100  -- January
  | 1 => 300  -- February
  | 2 => 200  -- March
  | 3 => 200  -- April
  | 4 => 100  -- May
  | 5 => 100  -- June

theorem lisa_savings_analysis :
  let total_average := (lisa_savings 0 + lisa_savings 1 + lisa_savings 2 + 
                        lisa_savings 3 + lisa_savings 4 + lisa_savings 5) / 6
  let first_trimester_average := (lisa_savings 0 + lisa_savings 1 + lisa_savings 2) / 3
  let second_trimester_average := (lisa_savings 3 + lisa_savings 4 + lisa_savings 5) / 3
  (total_average = 1000 / 6) ∧
  (first_trimester_average = 200) ∧
  (second_trimester_average = 400 / 3) ∧
  (first_trimester_average - second_trimester_average = 200 / 3) :=
by sorry

end NUMINAMATH_CALUDE_lisa_savings_analysis_l3392_339241


namespace NUMINAMATH_CALUDE_integers_starting_with_6_divisible_by_25_no_integers_divisible_by_35_without_first_digit_l3392_339260

def starts_with_6 (x : ℕ) : Prop :=
  ∃ n : ℕ, x = 6 * 10^n + (x % 10^n)

def divisible_by_25_without_first_digit (x : ℕ) : Prop :=
  ∃ n : ℕ, (x % 10^n) % 25 = 0

def divisible_by_35_without_first_digit (x : ℕ) : Prop :=
  ∃ n : ℕ, (x % 10^n) % 35 = 0

theorem integers_starting_with_6_divisible_by_25 :
  ∀ x : ℕ, starts_with_6 x ∧ divisible_by_25_without_first_digit x →
    ∃ k : ℕ, x = 625 * 10^k :=
sorry

theorem no_integers_divisible_by_35_without_first_digit :
  ¬ ∃ x : ℕ, divisible_by_35_without_first_digit x :=
sorry

end NUMINAMATH_CALUDE_integers_starting_with_6_divisible_by_25_no_integers_divisible_by_35_without_first_digit_l3392_339260


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3392_339290

theorem min_value_trig_expression (θ : Real) (h : θ ∈ Set.Ioo 0 (π / 2)) :
  1 / (Real.sin θ)^2 + 4 / (Real.cos θ)^2 ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3392_339290


namespace NUMINAMATH_CALUDE_remainder_problem_l3392_339281

theorem remainder_problem (N : ℤ) : ∃ (k : ℤ), N = 296 * k + 75 → ∃ (m : ℤ), N = 37 * m + 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3392_339281


namespace NUMINAMATH_CALUDE_point_difference_l3392_339277

/-- The value of a touchdown in points -/
def touchdown_value : ℕ := 7

/-- The number of touchdowns scored by Brayden and Gavin -/
def brayden_gavin_touchdowns : ℕ := 7

/-- The number of touchdowns scored by Cole and Freddy -/
def cole_freddy_touchdowns : ℕ := 9

/-- The point difference between Cole and Freddy's team and Brayden and Gavin's team -/
theorem point_difference : 
  cole_freddy_touchdowns * touchdown_value - brayden_gavin_touchdowns * touchdown_value = 14 :=
by sorry

end NUMINAMATH_CALUDE_point_difference_l3392_339277


namespace NUMINAMATH_CALUDE_probability_three_primes_is_correct_l3392_339243

def num_dice : ℕ := 7
def faces_per_die : ℕ := 10
def num_primes_per_die : ℕ := 4

def probability_exactly_three_primes : ℚ :=
  (num_dice.choose 3) *
  (num_primes_per_die / faces_per_die) ^ 3 *
  ((faces_per_die - num_primes_per_die) / faces_per_die) ^ (num_dice - 3)

theorem probability_three_primes_is_correct :
  probability_exactly_three_primes = 9072 / 31250 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_primes_is_correct_l3392_339243


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l3392_339282

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := p.b - 2 * p.a * h
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 1 ∧ p.b = -4 ∧ p.c = -4 →
  let p' := shift p 3 3
  p'.a = 1 ∧ p'.b = 2 ∧ p'.c = -5 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l3392_339282


namespace NUMINAMATH_CALUDE__l3392_339238

def main_theorem (f : ℝ → ℝ) (h1 : ∀ p q, f (p + q) = f p * f q) (h2 : f 1 = 3) : 
  (f 1 ^ 2 + f 2) / f 1 + (f 2 ^ 2 + f 4) / f 3 + (f 3 ^ 2 + f 6) / f 5 + 
  (f 4 ^ 2 + f 8) / f 7 + (f 5 ^ 2 + f 10) / f 9 = 30 := by
  sorry

end NUMINAMATH_CALUDE__l3392_339238


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l3392_339257

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l3392_339257


namespace NUMINAMATH_CALUDE_focal_length_determination_l3392_339253

/-- Represents a converging lens with a right isosceles triangle -/
structure LensSystem where
  focalLength : ℝ
  triangleArea : ℝ
  imageArea : ℝ

/-- The conditions of the lens system -/
def validLensSystem (s : LensSystem) : Prop :=
  s.triangleArea = 8 ∧ s.imageArea = s.triangleArea / 2

/-- The theorem statement -/
theorem focal_length_determination (s : LensSystem) 
  (h : validLensSystem s) : s.focalLength = 2 := by
  sorry

end NUMINAMATH_CALUDE_focal_length_determination_l3392_339253


namespace NUMINAMATH_CALUDE_probability_of_rolling_seven_l3392_339274

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := sides * sides

/-- The number of ways to roll a sum of 7 with two dice -/
def waysToRollSeven : ℕ := 6

/-- The probability of rolling a sum of 7 with two fair 6-sided dice -/
theorem probability_of_rolling_seven :
  (waysToRollSeven : ℚ) / totalOutcomes = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_probability_of_rolling_seven_l3392_339274


namespace NUMINAMATH_CALUDE_truth_telling_probability_l3392_339259

/-- The probability of two independent events occurring simultaneously -/
def simultaneous_probability (p_a p_b : ℝ) : ℝ := p_a * p_b

/-- Proof that given A speaks the truth 55% of the times and B speaks the truth 60% of the times, 
    the probability that they both tell the truth simultaneously is 0.33 -/
theorem truth_telling_probability : 
  let p_a : ℝ := 0.55
  let p_b : ℝ := 0.60
  simultaneous_probability p_a p_b = 0.33 := by
sorry

end NUMINAMATH_CALUDE_truth_telling_probability_l3392_339259


namespace NUMINAMATH_CALUDE_magician_payment_calculation_l3392_339218

/-- The total amount paid to a magician given their hourly rate, daily hours, and number of weeks worked -/
def magician_payment (hourly_rate : ℕ) (daily_hours : ℕ) (weeks : ℕ) : ℕ :=
  hourly_rate * daily_hours * 7 * weeks

/-- Theorem stating that a magician charging $60 per hour, working 3 hours daily for 2 weeks, earns $2520 -/
theorem magician_payment_calculation :
  magician_payment 60 3 2 = 2520 := by
  sorry

#eval magician_payment 60 3 2

end NUMINAMATH_CALUDE_magician_payment_calculation_l3392_339218


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3392_339288

/-- Given a quadratic function f(x) = 2x^2 + bx + c with solution set (0, 2) for f(x) < 0,
    if f(x) + t ≥ 2 holds for all real x, then t ≥ 4 -/
theorem quadratic_inequality_range (b c t : ℝ) : 
  (∀ x, x ∈ Set.Ioo 0 2 ↔ 2*x^2 + b*x + c < 0) →
  (∀ x, 2*x^2 + b*x + c + t ≥ 2) →
  t ≥ 4 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_range_l3392_339288


namespace NUMINAMATH_CALUDE_angle_is_2pi_3_l3392_339291

open Real

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_is_2pi_3 (a b : ℝ × ℝ) :
  b.1 * (a.1 + b.1) + b.2 * (a.2 + b.2) = 3 →
  a.1^2 + a.2^2 = 1 →
  b.1^2 + b.2^2 = 4 →
  angle_between_vectors a b = 2 * π / 3 := by sorry

end NUMINAMATH_CALUDE_angle_is_2pi_3_l3392_339291


namespace NUMINAMATH_CALUDE_arithmetic_progression_with_squares_is_integer_l3392_339255

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_progression (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A sequence contains the squares of its first three terms. -/
def contains_first_three_squares (a : ℕ → ℚ) : Prop :=
  ∃ k₁ k₂ k₃ : ℕ, a k₁ = (a 1)^2 ∧ a k₂ = (a 2)^2 ∧ a k₃ = (a 3)^2

/-- If an arithmetic progression contains the squares of its first three terms,
    then all terms in the progression are integers. -/
theorem arithmetic_progression_with_squares_is_integer
  (a : ℕ → ℚ)
  (h₁ : is_arithmetic_progression a)
  (h₂ : contains_first_three_squares a) :
  ∀ n : ℕ, ∃ k : ℤ, a n = k :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_with_squares_is_integer_l3392_339255


namespace NUMINAMATH_CALUDE_multiplication_of_monomials_l3392_339273

theorem multiplication_of_monomials (a : ℝ) : 3 * a * (4 * a^2) = 12 * a^3 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_of_monomials_l3392_339273


namespace NUMINAMATH_CALUDE_sum_of_digits_is_nine_l3392_339251

/-- The sum of the tens digit and the units digit in the decimal representation of 7^1974 -/
def sum_of_digits : ℕ :=
  let n := 7^1974
  let tens_digit := (n / 10) % 10
  let units_digit := n % 10
  tens_digit + units_digit

/-- The sum of the tens digit and the units digit in the decimal representation of 7^1974 is 9 -/
theorem sum_of_digits_is_nine : sum_of_digits = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_is_nine_l3392_339251


namespace NUMINAMATH_CALUDE_k_value_l3392_339213

def length (k : ℕ) : ℕ :=
  (Nat.factors k).length

theorem k_value (k : ℕ) (h1 : k > 1) (h2 : length k = 4) (h3 : k = 2 * 2 * 2 * 3) :
  k = 24 := by
  sorry

end NUMINAMATH_CALUDE_k_value_l3392_339213


namespace NUMINAMATH_CALUDE_nine_qualified_products_possible_l3392_339279

/-- The probability of success (pass rate) -/
def p : ℝ := 0.9

/-- The number of trials (products inspected) -/
def n : ℕ := 10

/-- The number of successes (qualified products) we're interested in -/
def k : ℕ := 9

/-- The binomial probability of k successes in n trials with probability p -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

theorem nine_qualified_products_possible : binomialProbability n k p > 0 := by
  sorry

end NUMINAMATH_CALUDE_nine_qualified_products_possible_l3392_339279


namespace NUMINAMATH_CALUDE_min_value_expression_l3392_339204

theorem min_value_expression (u v : ℝ) : 
  (u - v)^2 + (Real.sqrt (4 - u^2) - 2*v - 5)^2 ≥ 9 - 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3392_339204


namespace NUMINAMATH_CALUDE_total_coins_l3392_339295

theorem total_coins (quarters_piles : Nat) (quarters_per_pile : Nat)
                    (dimes_piles : Nat) (dimes_per_pile : Nat)
                    (nickels_piles : Nat) (nickels_per_pile : Nat)
                    (pennies_piles : Nat) (pennies_per_pile : Nat)
                    (h1 : quarters_piles = 8) (h2 : quarters_per_pile = 5)
                    (h3 : dimes_piles = 6) (h4 : dimes_per_pile = 7)
                    (h5 : nickels_piles = 4) (h6 : nickels_per_pile = 4)
                    (h7 : pennies_piles = 3) (h8 : pennies_per_pile = 6) :
  quarters_piles * quarters_per_pile +
  dimes_piles * dimes_per_pile +
  nickels_piles * nickels_per_pile +
  pennies_piles * pennies_per_pile = 116 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_l3392_339295


namespace NUMINAMATH_CALUDE_households_without_car_or_bike_l3392_339296

theorem households_without_car_or_bike
  (total : ℕ)
  (both : ℕ)
  (car : ℕ)
  (bike_only : ℕ)
  (h_total : total = 90)
  (h_both : both = 14)
  (h_car : car = 44)
  (h_bike_only : bike_only = 35) :
  total - (car + bike_only + both) = 11 :=
by sorry

end NUMINAMATH_CALUDE_households_without_car_or_bike_l3392_339296


namespace NUMINAMATH_CALUDE_four_books_equals_one_kg_l3392_339224

/-- Proves that 4 books weighing 250 grams each is equal to 1 kilogram -/
theorem four_books_equals_one_kg (book_weight : ℕ) (kg_in_grams : ℕ) : 
  book_weight = 250 → kg_in_grams = 1000 → 4 * book_weight = kg_in_grams := by
  sorry

end NUMINAMATH_CALUDE_four_books_equals_one_kg_l3392_339224


namespace NUMINAMATH_CALUDE_quadratic_residue_criterion_l3392_339232

theorem quadratic_residue_criterion (p a : ℕ) (hp : Prime p) (hp2 : p ≠ 2) (ha : a ≠ 0) :
  (∃ x, x^2 ≡ a [ZMOD p]) → a^((p-1)/2) ≡ 1 [ZMOD p] ∧
  (¬∃ x, x^2 ≡ a [ZMOD p]) → a^((p-1)/2) ≡ -1 [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_quadratic_residue_criterion_l3392_339232


namespace NUMINAMATH_CALUDE_amcb_paths_count_l3392_339254

/-- Represents the number of paths from one letter to the next -/
structure PathCount where
  a_to_m : Nat
  m_to_c : Nat
  c_to_b : Nat

/-- The configuration of the letter arrangement -/
structure LetterArrangement where
  central_a : Nat
  m_adjacent_to_a : Nat
  c_adjacent_to_m : Nat
  b_adjacent_to_c : Nat

/-- Calculates the total number of paths spelling "AMCB" -/
def total_paths (arrangement : LetterArrangement) : Nat :=
  arrangement.central_a * arrangement.m_adjacent_to_a * arrangement.c_adjacent_to_m * arrangement.b_adjacent_to_c

/-- The specific arrangement for this problem -/
def amcb_arrangement : LetterArrangement :=
  { central_a := 1
  , m_adjacent_to_a := 4
  , c_adjacent_to_m := 2
  , b_adjacent_to_c := 3 }

theorem amcb_paths_count :
  total_paths amcb_arrangement = 24 :=
sorry

end NUMINAMATH_CALUDE_amcb_paths_count_l3392_339254


namespace NUMINAMATH_CALUDE_minimum_bookmarks_l3392_339233

def is_divisible_by (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

theorem minimum_bookmarks : 
  ∀ (n : ℕ), n > 0 → 
  (is_divisible_by n 3 ∧ 
   is_divisible_by n 4 ∧ 
   is_divisible_by n 5 ∧ 
   is_divisible_by n 7 ∧ 
   is_divisible_by n 8) → 
  n ≥ 840 :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_bookmarks_l3392_339233


namespace NUMINAMATH_CALUDE_angle_equation_l3392_339271

theorem angle_equation (α β : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : (Real.sin α + Real.cos α) * (Real.sin β + Real.cos β) = 2) :
  (Real.sin (2 * α) + Real.cos (3 * β))^2 + (Real.sin (2 * β) + Real.cos (3 * α))^2 = 3 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_equation_l3392_339271


namespace NUMINAMATH_CALUDE_alpha_range_l3392_339215

theorem alpha_range (α : Real) (k : Int) 
  (h1 : Real.sin α > 0)
  (h2 : Real.cos α < 0)
  (h3 : Real.sin (α / 3) > Real.cos (α / 3)) :
  ∃ k, (α / 3 ∈ Set.Ioo (2 * k * Real.pi + Real.pi / 4) (2 * k * Real.pi + Real.pi / 3)) ∨
       (α / 3 ∈ Set.Ioo (2 * k * Real.pi + 5 * Real.pi / 6) (2 * k * Real.pi + Real.pi)) :=
by sorry

end NUMINAMATH_CALUDE_alpha_range_l3392_339215


namespace NUMINAMATH_CALUDE_classroom_desks_proof_l3392_339247

/-- The number of rows in the classroom -/
def num_rows : ℕ := 8

/-- The number of desks in the first row -/
def first_row_desks : ℕ := 10

/-- The maximum number of students that can be seated -/
def max_students : ℕ := 136

/-- The number of additional desks in each subsequent row -/
def additional_desks : ℕ := 2

/-- Calculates the total number of desks in the classroom -/
def total_desks (n : ℕ) : ℕ :=
  num_rows * first_row_desks + (num_rows - 1) * num_rows * n / 2

theorem classroom_desks_proof :
  total_desks additional_desks = max_students :=
sorry

end NUMINAMATH_CALUDE_classroom_desks_proof_l3392_339247


namespace NUMINAMATH_CALUDE_expression_equality_l3392_339229

theorem expression_equality : (481 * 7 + 426 * 5)^3 - 4 * (481 * 7) * (426 * 5) = 166021128033 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3392_339229


namespace NUMINAMATH_CALUDE_f_properties_l3392_339210

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x + a else -x - 2 * a

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  (f (-3) 10 = -4 ∧ f (-3) (f (-3) 10) = -11) ∧
  (∀ b : ℝ, b ≠ 0 → (f b (1 - b) = f b (1 + b) ↔ b = -3/4)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3392_339210


namespace NUMINAMATH_CALUDE_beach_trip_ratio_l3392_339223

theorem beach_trip_ratio (total : ℕ) (remaining : ℕ) (beach : ℕ) :
  total = 1000 →
  remaining = (total - beach) / 2 →
  remaining = 250 →
  beach / total = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_beach_trip_ratio_l3392_339223


namespace NUMINAMATH_CALUDE_angle_AFE_measure_l3392_339286

-- Define the points
variable (A B C D E F : Point)

-- Define the rectangle ABCD
def is_rectangle (A B C D : Point) : Prop := sorry

-- Define the relationship AB = 2BC
def AB_twice_BC (A B C : Point) : Prop := sorry

-- Define E on the opposite half-plane from A with respect to CD
def E_opposite_halfplane (A C D E : Point) : Prop := sorry

-- Define angle CDE = 120°
def angle_CDE_120 (C D E : Point) : Prop := sorry

-- Define F as midpoint of AD
def F_midpoint_AD (A D F : Point) : Prop := sorry

-- Define the measure of an angle
def angle_measure (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem angle_AFE_measure
  (h_rect : is_rectangle A B C D)
  (h_AB_BC : AB_twice_BC A B C)
  (h_E_opp : E_opposite_halfplane A C D E)
  (h_CDE : angle_CDE_120 C D E)
  (h_F_mid : F_midpoint_AD A D F) :
  angle_measure A F E = 150 := by sorry

end NUMINAMATH_CALUDE_angle_AFE_measure_l3392_339286


namespace NUMINAMATH_CALUDE_root_equation_value_l3392_339276

theorem root_equation_value (a b m : ℝ) : 
  a * m^2 + b * m + 5 = 0 → a * m^2 + b * m - 7 = -12 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l3392_339276


namespace NUMINAMATH_CALUDE_circle_passes_through_fixed_point_l3392_339283

-- Define the parabola
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

-- Define the line x = -1
def line (x : ℝ) : Prop := x = -1

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define when a circle is tangent to a line
def is_tangent_to_line (c : Circle) (l : ℝ → Prop) : Prop :=
  abs (c.center.1 - (-1)) = c.radius

-- Define when a point is on a circle
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- The main theorem
theorem circle_passes_through_fixed_point :
  ∀ c : Circle,
    parabola c.center →
    is_tangent_to_line c line →
    point_on_circle (1, 0) c :=
sorry

end NUMINAMATH_CALUDE_circle_passes_through_fixed_point_l3392_339283


namespace NUMINAMATH_CALUDE_sum_a_b_equals_negative_one_l3392_339272

theorem sum_a_b_equals_negative_one (a b : ℝ) (h : |a + 3| + (b - 2)^2 = 0) : a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_negative_one_l3392_339272


namespace NUMINAMATH_CALUDE_mendel_pea_experiment_l3392_339292

/-- Represents the genotype of a pea plant -/
inductive Genotype
| DD
| Dd
| dd

/-- Represents a generation of pea plants -/
structure Generation where
  DD_ratio : ℚ
  Dd_ratio : ℚ
  dd_ratio : ℚ
  sum_to_one : DD_ratio + Dd_ratio + dd_ratio = 1

/-- First generation with all Dd genotype -/
def first_gen : Generation where
  DD_ratio := 0
  Dd_ratio := 1
  dd_ratio := 0
  sum_to_one := by norm_num

/-- Function to calculate the next generation's ratios -/
def next_gen (g : Generation) : Generation where
  DD_ratio := g.DD_ratio^2 + g.DD_ratio * g.Dd_ratio + (g.Dd_ratio^2) / 4
  Dd_ratio := g.DD_ratio * g.Dd_ratio + g.Dd_ratio * g.dd_ratio + (g.Dd_ratio^2) / 2
  dd_ratio := g.dd_ratio^2 + g.dd_ratio * g.Dd_ratio + (g.Dd_ratio^2) / 4
  sum_to_one := by sorry

/-- Second generation -/
def second_gen : Generation := next_gen first_gen

/-- Third generation -/
def third_gen : Generation := next_gen second_gen

/-- Probability of dominant trait in a generation -/
def prob_dominant (g : Generation) : ℚ := g.DD_ratio + g.Dd_ratio

theorem mendel_pea_experiment :
  (third_gen.dd_ratio = 1/4) ∧
  (3 * (prob_dominant third_gen)^2 * (1 - prob_dominant third_gen) = 27/64) := by sorry

end NUMINAMATH_CALUDE_mendel_pea_experiment_l3392_339292


namespace NUMINAMATH_CALUDE_pauls_tips_amount_l3392_339235

def pauls_tips : ℕ := 14
def vinnies_earnings : ℕ := 30

theorem pauls_tips_amount :
  (∃ (p : ℕ), p = pauls_tips ∧ vinnies_earnings = p + 16) →
  pauls_tips = 14 := by
  sorry

end NUMINAMATH_CALUDE_pauls_tips_amount_l3392_339235


namespace NUMINAMATH_CALUDE_remaining_pokemon_cards_l3392_339231

/-- Theorem: Calculating remaining Pokemon cards after a sale --/
theorem remaining_pokemon_cards 
  (initial_cards : ℕ) 
  (sold_cards : ℕ) 
  (h1 : initial_cards = 676)
  (h2 : sold_cards = 224) :
  initial_cards - sold_cards = 452 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_pokemon_cards_l3392_339231


namespace NUMINAMATH_CALUDE_probability_at_least_one_boy_one_girl_l3392_339258

theorem probability_at_least_one_boy_one_girl :
  let p_boy : ℝ := 1 / 2
  let p_girl : ℝ := 1 - p_boy
  let num_children : ℕ := 4
  let p_all_boys : ℝ := p_boy ^ num_children
  let p_all_girls : ℝ := p_girl ^ num_children
  p_all_boys + p_all_girls = 1 / 8 →
  1 - (p_all_boys + p_all_girls) = 7 / 8 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_boy_one_girl_l3392_339258


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l3392_339214

/-- Given a line passing through points (-2, 3) and (3, -2), 
    the product of the square of its slope and its y-intercept equals 1 -/
theorem line_slope_intercept_product : 
  ∀ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b) →  -- Line equation
    (3 = m * (-2) + b) →          -- Point (-2, 3) satisfies the equation
    (-2 = m * 3 + b) →            -- Point (3, -2) satisfies the equation
    m^2 * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l3392_339214


namespace NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l3392_339249

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largestPerfectSquareDivisor (n : ℕ) : ℕ :=
  sorry

def sumOfExponents (n : ℕ) : ℕ :=
  sorry

theorem sum_of_exponents_15_factorial :
  sumOfExponents (largestPerfectSquareDivisor (factorial 15).sqrt) = 10 :=
sorry

end NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l3392_339249


namespace NUMINAMATH_CALUDE_root_product_l3392_339203

theorem root_product (x₁ x₂ : ℝ) (h₁ : x₁ * Real.log x₁ = 2006) (h₂ : x₂ * Real.exp x₂ = 2006) : 
  x₁ * x₂ = 2006 := by
sorry

end NUMINAMATH_CALUDE_root_product_l3392_339203


namespace NUMINAMATH_CALUDE_symmetric_complex_numbers_l3392_339240

/-- Two complex numbers are symmetric with respect to the imaginary axis if their real parts are negatives of each other and their imaginary parts are equal. -/
def symmetric_wrt_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

/-- If z₁ and z₂ are complex numbers symmetric with respect to the imaginary axis,
    and z₁ = 2 + i, then z₂ = -2 + i. -/
theorem symmetric_complex_numbers (z₁ z₂ : ℂ) 
    (h_sym : symmetric_wrt_imaginary_axis z₁ z₂) 
    (h_z₁ : z₁ = 2 + I) : 
    z₂ = -2 + I := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_numbers_l3392_339240


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l3392_339211

theorem sum_of_x_and_y_is_two (x y : ℝ) (h : x^2 + y^2 = 10*x - 6*y - 34) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l3392_339211


namespace NUMINAMATH_CALUDE_subway_length_l3392_339245

/-- The length of a subway given its speed, time to cross a bridge, and bridge length. -/
theorem subway_length
  (speed : ℝ)  -- Speed of the subway in km/min
  (time : ℝ)   -- Time to cross the bridge in minutes
  (bridge_length : ℝ)  -- Length of the bridge in km
  (h1 : speed = 1.6)  -- The subway speed is 1.6 km/min
  (h2 : time = 3.25)  -- The time to cross the bridge is 3 min and 15 sec (3.25 min)
  (h3 : bridge_length = 4.85)  -- The bridge length is 4.85 km
  : (speed * time - bridge_length) * 1000 = 350 :=
by sorry

end NUMINAMATH_CALUDE_subway_length_l3392_339245


namespace NUMINAMATH_CALUDE_f_ln_2_equals_neg_1_l3392_339219

-- Define the base of natural logarithm
noncomputable def e : ℝ := Real.exp 1

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_ln_2_equals_neg_1 
  (h_monotonic : Monotone f)
  (h_condition : ∀ x : ℝ, f (f x + Real.exp x) = 1 - e) :
  f (Real.log 2) = -1 := by sorry

end NUMINAMATH_CALUDE_f_ln_2_equals_neg_1_l3392_339219


namespace NUMINAMATH_CALUDE_music_festival_audience_count_l3392_339297

/-- Represents the distribution of audience for a band -/
structure BandDistribution where
  underThirtyMale : ℝ
  underThirtyFemale : ℝ
  thirtyToFiftyMale : ℝ
  thirtyToFiftyFemale : ℝ
  overFiftyMale : ℝ
  overFiftyFemale : ℝ

/-- The music festival with its audience distribution -/
def MusicFestival : List BandDistribution :=
  [
    { underThirtyMale := 0.04, underThirtyFemale := 0.0266667, thirtyToFiftyMale := 0.0375, thirtyToFiftyFemale := 0.0458333, overFiftyMale := 0.00833333, overFiftyFemale := 0.00833333 },
    { underThirtyMale := 0.03, underThirtyFemale := 0.07, thirtyToFiftyMale := 0.02, thirtyToFiftyFemale := 0.03, overFiftyMale := 0.00833333, overFiftyFemale := 0.00833333 },
    { underThirtyMale := 0.02, underThirtyFemale := 0.03, thirtyToFiftyMale := 0.0416667, thirtyToFiftyFemale := 0.0416667, overFiftyMale := 0.0133333, overFiftyFemale := 0.02 },
    { underThirtyMale := 0.0458333, underThirtyFemale := 0.0375, thirtyToFiftyMale := 0.03, thirtyToFiftyFemale := 0.0366667, overFiftyMale := 0.01, overFiftyFemale := 0.00666667 },
    { underThirtyMale := 0.015, underThirtyFemale := 0.0183333, thirtyToFiftyMale := 0.0333333, thirtyToFiftyFemale := 0.0333333, overFiftyMale := 0.03, overFiftyFemale := 0.0366667 },
    { underThirtyMale := 0.0583333, underThirtyFemale := 0.025, thirtyToFiftyMale := 0.03, thirtyToFiftyFemale := 0.0366667, overFiftyMale := 0.00916667, overFiftyFemale := 0.00750 }
  ]

theorem music_festival_audience_count : 
  let totalMaleUnder30 := (MusicFestival.map (λ b => b.underThirtyMale)).sum
  ∃ n : ℕ, n ≥ 431 ∧ n < 432 ∧ (90 : ℝ) / totalMaleUnder30 = n := by
  sorry

end NUMINAMATH_CALUDE_music_festival_audience_count_l3392_339297


namespace NUMINAMATH_CALUDE_loss_recording_l3392_339239

/-- Records a financial transaction as a number, where profits are positive and losses are negative. -/
def recordTransaction (amount : ℤ) : ℤ := amount

/-- Given that a profit of 100 yuan is recorded as +100, prove that a loss of 50 yuan is recorded as -50. -/
theorem loss_recording (h : recordTransaction 100 = 100) : recordTransaction (-50) = -50 := by
  sorry

end NUMINAMATH_CALUDE_loss_recording_l3392_339239


namespace NUMINAMATH_CALUDE_no_sequence_satisfying_inequality_l3392_339264

theorem no_sequence_satisfying_inequality :
  ¬ ∃ (α : ℝ) (a : ℕ → ℝ), 
    (0 < α ∧ α < 1) ∧
    (∀ n, 0 < a n) ∧
    (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) :=
by sorry

end NUMINAMATH_CALUDE_no_sequence_satisfying_inequality_l3392_339264


namespace NUMINAMATH_CALUDE_lcm_of_165_and_396_l3392_339236

theorem lcm_of_165_and_396 : Nat.lcm 165 396 = 1980 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_165_and_396_l3392_339236


namespace NUMINAMATH_CALUDE_sequence_problem_l3392_339289

def sequence_rule (x y z : ℕ) : Prop := z = 2 * (x + y)

theorem sequence_problem : 
  ∀ (a b c : ℕ), 
  sequence_rule 10 a 30 → 
  sequence_rule a 30 b → 
  sequence_rule 30 b c → 
  c = 200 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l3392_339289


namespace NUMINAMATH_CALUDE_cornbread_pieces_count_l3392_339242

/-- Represents the dimensions of a rectangular pan --/
structure PanDimensions where
  length : ℕ
  width : ℕ

/-- Represents the dimensions of a piece of cornbread --/
structure PieceDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of whole pieces that can be cut from a pan --/
def maxPieces (pan : PanDimensions) (piece : PieceDimensions) (margin : ℕ) : ℕ :=
  ((pan.length / piece.length) * ((pan.width - margin) / piece.width))

/-- Theorem stating that the maximum number of pieces for the given dimensions is 72 --/
theorem cornbread_pieces_count :
  let pan := PanDimensions.mk 24 20
  let piece := PieceDimensions.mk 3 2
  let margin := 1
  maxPieces pan piece margin = 72 := by
  sorry


end NUMINAMATH_CALUDE_cornbread_pieces_count_l3392_339242


namespace NUMINAMATH_CALUDE_factorization_proof_l3392_339294

theorem factorization_proof (a : ℝ) : 2*a - 2*a^3 = 2*a*(1+a)*(1-a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3392_339294


namespace NUMINAMATH_CALUDE_speed_equivalence_l3392_339209

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- The given speed in meters per second -/
def speed_mps : ℝ := 115.00919999999999

/-- The calculated speed in kilometers per hour -/
def speed_kmph : ℝ := 414.03312

/-- Theorem stating that the given speed in m/s is equivalent to the calculated speed in km/h -/
theorem speed_equivalence : speed_mps * mps_to_kmph = speed_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_equivalence_l3392_339209


namespace NUMINAMATH_CALUDE_fraction_equality_implies_c_geq_one_l3392_339226

theorem fraction_equality_implies_c_geq_one 
  (a b : ℕ+) 
  (c : ℝ) 
  (h_c_pos : c > 0) 
  (h_eq : (a + 1 : ℝ) / (b + c) = (b : ℝ) / a) : 
  c ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_c_geq_one_l3392_339226


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l3392_339216

theorem tan_product_pi_ninths : Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l3392_339216


namespace NUMINAMATH_CALUDE_unique_prime_pair_l3392_339250

theorem unique_prime_pair : ∃! (p q : ℕ), 
  Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 3 ∧ q ≠ 3 ∧
  (∀ α : ℤ, (α ^ (3 * p * q) - α) % (3 * p * q) = 0) ∧
  p = 11 ∧ q = 17 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l3392_339250


namespace NUMINAMATH_CALUDE_family_park_cost_l3392_339202

/-- Calculates the total cost for a family to visit a park and one attraction -/
def total_cost (park_fee : ℕ) (child_attraction_fee : ℕ) (adult_attraction_fee : ℕ) 
                (num_children : ℕ) (num_parents : ℕ) (num_grandparents : ℕ) : ℕ :=
  let total_family_members := num_children + num_parents + num_grandparents
  let total_adults := num_parents + num_grandparents
  let park_cost := total_family_members * park_fee
  let children_attraction_cost := num_children * child_attraction_fee
  let adult_attraction_cost := total_adults * adult_attraction_fee
  park_cost + children_attraction_cost + adult_attraction_cost

/-- Theorem: The total cost for the specified family composition is $55 -/
theorem family_park_cost : 
  total_cost 5 2 4 4 2 1 = 55 := by
  sorry

end NUMINAMATH_CALUDE_family_park_cost_l3392_339202


namespace NUMINAMATH_CALUDE_range_of_power_function_l3392_339227

theorem range_of_power_function (k : ℝ) (h : k > 0) :
  Set.range (fun x : ℝ => x ^ k) ∩ Set.Ici 1 = Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_range_of_power_function_l3392_339227


namespace NUMINAMATH_CALUDE_intersection_line_property_l3392_339228

-- Define the circles
def circle_ω₁ : Set (ℝ × ℝ) := {p | (p.1^2 + (p.2 - 6)^2) = 900}
def circle_ω₂ : Set (ℝ × ℝ) := {p | ((p.1 - 20)^2 + p.2^2) = 900}

-- Define the intersection points
def X : ℝ × ℝ := sorry
def Y : ℝ × ℝ := sorry

-- Define the line passing through X and Y
def m : ℝ := sorry
def b : ℝ := sorry

-- Theorem statement
theorem intersection_line_property :
  X ∈ circle_ω₁ ∧ X ∈ circle_ω₂ ∧
  Y ∈ circle_ω₁ ∧ Y ∈ circle_ω₂ ∧
  (∀ p : ℝ × ℝ, p.2 = m * p.1 + b ↔ (p = X ∨ p = Y)) →
  100 * m + b = 303 := by sorry

end NUMINAMATH_CALUDE_intersection_line_property_l3392_339228


namespace NUMINAMATH_CALUDE_friends_earrings_count_l3392_339248

def total_earrings (bella_earrings monica_earrings rachel_earrings olivia_earrings : ℕ) : ℕ :=
  bella_earrings + monica_earrings + rachel_earrings + olivia_earrings

theorem friends_earrings_count :
  ∀ (bella_earrings monica_earrings rachel_earrings olivia_earrings : ℕ),
    bella_earrings = 10 →
    bella_earrings = monica_earrings / 4 →
    monica_earrings = 2 * rachel_earrings →
    olivia_earrings = bella_earrings + monica_earrings + rachel_earrings + 5 →
    total_earrings bella_earrings monica_earrings rachel_earrings olivia_earrings = 145 :=
by
  sorry

#check friends_earrings_count

end NUMINAMATH_CALUDE_friends_earrings_count_l3392_339248


namespace NUMINAMATH_CALUDE_fraction_equality_l3392_339285

theorem fraction_equality (a b c d : ℝ) : 
  (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 4 →
  (b - a) * (d - c) / ((c - b) * (a - d)) = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3392_339285


namespace NUMINAMATH_CALUDE_ratio_difference_l3392_339284

theorem ratio_difference (a b c : ℕ) (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : ∃ (x : ℕ), a = 3 * x ∧ b = 5 * x ∧ c = 7 * x) (h3 : c = 70) :
  c - a = 40 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_l3392_339284


namespace NUMINAMATH_CALUDE_equation_solution_l3392_339222

theorem equation_solution (a b c : ℤ) : 
  (∀ x : ℤ, (x - a) * (x - 12) + 1 = (x + b) * (x + c)) ↔ 
  ((a = 10 ∧ b = -11 ∧ c = -11) ∨ (a = 14 ∧ b = -13 ∧ c = -13)) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3392_339222


namespace NUMINAMATH_CALUDE_output_for_15_l3392_339261

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 40 then
    step1 + 10
  else
    step1 - 7

theorem output_for_15 : function_machine 15 = 38 := by
  sorry

end NUMINAMATH_CALUDE_output_for_15_l3392_339261


namespace NUMINAMATH_CALUDE_floors_per_house_l3392_339208

/-- The number of floors in each house given the building conditions -/
theorem floors_per_house 
  (builders_per_floor : ℕ)
  (days_per_floor : ℕ)
  (daily_wage : ℕ)
  (total_builders : ℕ)
  (total_houses : ℕ)
  (total_cost : ℕ)
  (h1 : builders_per_floor = 3)
  (h2 : days_per_floor = 30)
  (h3 : daily_wage = 100)
  (h4 : total_builders = 6)
  (h5 : total_houses = 5)
  (h6 : total_cost = 270000) :
  total_cost / (total_builders * daily_wage * days_per_floor * total_houses) = 3 :=
sorry

end NUMINAMATH_CALUDE_floors_per_house_l3392_339208


namespace NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l3392_339205

theorem midpoint_of_complex_line_segment :
  let z₁ : ℂ := -7 + 5*I
  let z₂ : ℂ := 5 - 3*I
  let midpoint := (z₁ + z₂) / 2
  midpoint = -1 + I :=
by sorry

end NUMINAMATH_CALUDE_midpoint_of_complex_line_segment_l3392_339205


namespace NUMINAMATH_CALUDE_f_properties_l3392_339212

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (-2^x + a) / (2^x + 1)

theorem f_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 1 ∧
   (∀ x y, x < y → f a y < f a x) ∧
   (∀ k, (∀ t, 1 ≤ t ∧ t ≤ 3 → f a (t^2 - 2*t) + f a (2*t^2 - k) < 0) → k < -1/3)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3392_339212


namespace NUMINAMATH_CALUDE_boat_travel_distance_l3392_339201

/-- Proves that a boat traveling upstream and downstream with given conditions travels 91.25 miles -/
theorem boat_travel_distance (v : ℝ) (d : ℝ) : 
  d / (v - 3) = d / (v + 3) + 0.5 →
  d / (v + 3) = 2.5191640969412834 →
  d = 91.25 := by
  sorry

end NUMINAMATH_CALUDE_boat_travel_distance_l3392_339201


namespace NUMINAMATH_CALUDE_exists_subset_with_unique_sum_representation_l3392_339225

theorem exists_subset_with_unique_sum_representation : 
  ∃ (X : Set ℤ), ∀ (n : ℤ), ∃! (p : ℤ × ℤ), p.1 ∈ X ∧ p.2 ∈ X ∧ p.1 + 2 * p.2 = n :=
sorry

end NUMINAMATH_CALUDE_exists_subset_with_unique_sum_representation_l3392_339225


namespace NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l3392_339200

/-- The coefficient of x^5 in the expansion of (x^3 + 1/x)^7 is 35 -/
theorem coefficient_x5_in_expansion : Nat := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x5_in_expansion_l3392_339200


namespace NUMINAMATH_CALUDE_system_solution_l3392_339268

theorem system_solution : ∃ (x y : ℚ), 
  (7 * x - 14 * y = 3) ∧ 
  (3 * y - x = 5) ∧ 
  (x = 79/7) ∧ 
  (y = 38/7) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3392_339268


namespace NUMINAMATH_CALUDE_quiz_competition_order_l3392_339287

theorem quiz_competition_order (A B C : ℕ) 
  (h1 : A + B = 2 * C)
  (h2 : 3 * A > 3 * B + 3 * C + 10)
  (h3 : 3 * B = 3 * C + 5)
  (h4 : A > 0 ∧ B > 0 ∧ C > 0) :
  C > A ∧ A > B := by
  sorry

end NUMINAMATH_CALUDE_quiz_competition_order_l3392_339287


namespace NUMINAMATH_CALUDE_circle_and_max_distance_l3392_339293

-- Define the circle C
def Circle (a b r : ℝ) := {(x, y) : ℝ × ℝ | (x - a)^2 + (y - b)^2 = r^2}

-- Define the conditions for the circle
def CircleConditions (a b r : ℝ) : Prop :=
  3 * a - b = 0 ∧ 
  a ≥ 0 ∧ 
  |a - 4| = r ∧ 
  ((3 * a + 4 * b + 10)^2 / 25 + 12 = r^2)

-- Define points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-2, 0)

-- Define the distance squared function
def DistanceSquared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Theorem statement
theorem circle_and_max_distance :
  ∃ a b r : ℝ, 
    CircleConditions a b r → 
    (Circle a b r = Circle 0 0 4) ∧
    (∀ p ∈ Circle 0 0 4, DistanceSquared p A + DistanceSquared p B ≤ 38 + 8 * Real.sqrt 2) ∧
    (∃ p ∈ Circle 0 0 4, DistanceSquared p A + DistanceSquared p B = 38 + 8 * Real.sqrt 2) :=
  sorry

end NUMINAMATH_CALUDE_circle_and_max_distance_l3392_339293


namespace NUMINAMATH_CALUDE_smallest_mustang_is_12_inches_l3392_339220

/-- The length of the smallest Mustang model given the full-size and scaling factors -/
def smallest_mustang_length (full_size : ℝ) (mid_size_factor : ℝ) (smallest_factor : ℝ) : ℝ :=
  full_size * mid_size_factor * smallest_factor

/-- Theorem stating that the smallest Mustang model is 12 inches long -/
theorem smallest_mustang_is_12_inches :
  smallest_mustang_length 240 (1/10) (1/2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_mustang_is_12_inches_l3392_339220


namespace NUMINAMATH_CALUDE_geometric_sum_example_l3392_339230

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Proof that the sum of the first 8 terms of the geometric sequence
    with first term 1/3 and common ratio 1/3 is 9840/19683 -/
theorem geometric_sum_example :
  geometric_sum (1/3) (1/3) 8 = 9840/19683 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_example_l3392_339230


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3392_339207

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 12 is √3 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 12
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l3392_339207


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3392_339280

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_a1a3 : a 1 * a 3 = 36)
  (h_a4 : a 4 = 54) :
  ∃ q : ℝ, q = 3 ∧ ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3392_339280


namespace NUMINAMATH_CALUDE_equation_solution_l3392_339299

theorem equation_solution : 
  ∃! x : ℝ, x ≠ 1 ∧ (2 * x) / (x - 1) - 1 = 4 / (1 - x) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3392_339299


namespace NUMINAMATH_CALUDE_cubic_not_always_square_l3392_339221

theorem cubic_not_always_square (a b c : ℤ) : ∃ n : ℕ+, ¬∃ m : ℤ, (n : ℤ)^3 + a*(n : ℤ)^2 + b*(n : ℤ) + c = m^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_not_always_square_l3392_339221


namespace NUMINAMATH_CALUDE_stone_reduction_moves_l3392_339206

theorem stone_reduction_moves (n : ℕ) (h : n = 2005) : 
  ∃ (m : ℕ), m = 11 ∧ 
  (∀ (k : ℕ), k < m → 2^(m - k) ≥ n) ∧
  (2^(m - m) < n) :=
sorry

end NUMINAMATH_CALUDE_stone_reduction_moves_l3392_339206


namespace NUMINAMATH_CALUDE_circles_intersect_and_common_chord_l3392_339237

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 6 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 6 = 0

-- Define the intersection of the circles
def intersect : Prop := ∃ x y : ℝ, circle1 x y ∧ circle2 x y

-- Define the common chord equation
def commonChord (x y : ℝ) : Prop := 3*x - 2*y = 0

theorem circles_intersect_and_common_chord :
  intersect ∧ (∀ x y : ℝ, circle1 x y ∧ circle2 x y → commonChord x y) :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_and_common_chord_l3392_339237


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3392_339217

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_diff_1 : a 2 - a 1 = 1)
  (h_diff_2 : a 5 - a 4 = 8) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3392_339217


namespace NUMINAMATH_CALUDE_patients_arrangement_exists_l3392_339265

theorem patients_arrangement_exists :
  ∃ (cow she_wolf beetle worm : ℝ),
    0 ≤ cow ∧ cow < she_wolf ∧ she_wolf < beetle ∧ beetle < worm ∧ worm = 6 ∧
    she_wolf - cow = 1 ∧
    beetle - cow = 2 ∧
    (she_wolf - cow) + (beetle - she_wolf) + (worm - beetle) = 7 ∧
    (beetle - cow) + (worm - beetle) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_patients_arrangement_exists_l3392_339265


namespace NUMINAMATH_CALUDE_tangent_line_m_squared_l3392_339256

/-- A line that intersects an ellipse and a circle exactly once -/
structure TangentLine where
  m : ℝ
  -- Line equation: y = mx + 2
  line : ℝ → ℝ := fun x => m * x + 2
  -- Ellipse equation: x^2 + 9y^2 = 9
  ellipse : ℝ × ℝ → Prop := fun (x, y) => x^2 + 9 * y^2 = 9
  -- Circle equation: x^2 + y^2 = 4
  circle : ℝ × ℝ → Prop := fun (x, y) => x^2 + y^2 = 4
  -- The line intersects both the ellipse and the circle exactly once
  h_tangent_ellipse : ∃! x, ellipse (x, line x)
  h_tangent_circle : ∃! x, circle (x, line x)

/-- The theorem stating that m^2 = 1/3 for a line tangent to both the ellipse and circle -/
theorem tangent_line_m_squared (l : TangentLine) : l.m^2 = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_m_squared_l3392_339256


namespace NUMINAMATH_CALUDE_possible_sum_less_than_100_l3392_339269

/-- Represents a team in the tournament -/
structure Team :=
  (id : Nat)
  (score : Nat)

/-- Represents the tournament -/
structure Tournament :=
  (teams : List Team)
  (num_teams : Nat)
  (num_games : Nat)

/-- The scoring system for the tournament -/
def scoring_system (winner_rank : Nat) (loser_rank : Nat) : Nat :=
  if winner_rank ≤ 5 then 3 else 2

/-- The theorem stating that it's possible for the sum of scores to be less than 100 -/
theorem possible_sum_less_than_100 (t : Tournament) :
  t.num_teams = 10 →
  t.num_games = (t.num_teams * (t.num_teams - 1)) / 2 →
  ∃ (scores : List Nat), 
    scores.length = t.num_teams ∧ 
    scores.sum < 100 ∧
    (∀ (i j : Nat), i < j → j < t.num_teams → 
      ∃ (points : Nat), points ≤ (scoring_system (i + 1) (j + 1)) ∧
        (scores.get! i + scores.get! j = points)) :=
sorry

end NUMINAMATH_CALUDE_possible_sum_less_than_100_l3392_339269


namespace NUMINAMATH_CALUDE_profit_maximization_l3392_339263

def profit_function (x : ℝ) : ℝ := -2 * x^2 + 200 * x - 3200

theorem profit_maximization (x : ℝ) :
  35 ≤ x ∧ x ≤ 45 →
  (∀ y : ℝ, 35 ≤ y ∧ y ≤ 45 → profit_function y ≤ profit_function 45) ∧
  profit_function 45 = 1750 ∧
  (∀ z : ℝ, 35 ≤ z ∧ z ≤ 45 ∧ profit_function z ≥ 1600 → 40 ≤ z ∧ z ≤ 45) :=
by sorry

end NUMINAMATH_CALUDE_profit_maximization_l3392_339263


namespace NUMINAMATH_CALUDE_car_dealership_silver_percentage_l3392_339246

theorem car_dealership_silver_percentage
  (initial_cars : ℕ)
  (initial_silver_percentage : ℚ)
  (new_shipment : ℕ)
  (new_non_silver_percentage : ℚ)
  (h1 : initial_cars = 40)
  (h2 : initial_silver_percentage = 1/5)
  (h3 : new_shipment = 80)
  (h4 : new_non_silver_percentage = 7/20)
  : (initial_silver_percentage * initial_cars + (1 - new_non_silver_percentage) * new_shipment) / (initial_cars + new_shipment) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_car_dealership_silver_percentage_l3392_339246


namespace NUMINAMATH_CALUDE_distance_to_center_is_five_l3392_339252

/-- A square with side length 10 and a circle passing through two opposite vertices
    and tangent to one side -/
structure SquareWithCircle where
  /-- The side length of the square -/
  sideLength : ℝ
  /-- The circle passes through two opposite vertices -/
  circlePassesThroughOppositeVertices : Bool
  /-- The circle is tangent to one side -/
  circleTangentToSide : Bool

/-- The distance from the center of the circle to a vertex of the square -/
def distanceToCenterFromVertex (s : SquareWithCircle) : ℝ := sorry

/-- Theorem stating that the distance from the center of the circle to a vertex is 5 -/
theorem distance_to_center_is_five (s : SquareWithCircle) 
  (h1 : s.sideLength = 10)
  (h2 : s.circlePassesThroughOppositeVertices = true)
  (h3 : s.circleTangentToSide = true) : 
  distanceToCenterFromVertex s = 5 := by sorry

end NUMINAMATH_CALUDE_distance_to_center_is_five_l3392_339252


namespace NUMINAMATH_CALUDE_car_speed_comparison_l3392_339262

theorem car_speed_comparison (u v : ℝ) (hu : u > 0) (hv : v > 0) :
  let x := 2 / (1 / u + 1 / v)
  let y := (u + v) / 2
  x ≤ y := by
sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l3392_339262


namespace NUMINAMATH_CALUDE_approx_value_of_625_power_l3392_339275

theorem approx_value_of_625_power (ε : Real) (hε : ε > 0) :
  ∃ (x : Real), abs (x - ((625 : Real)^(0.2 : Real) * (625 : Real)^(0.12 : Real))) < ε ∧
                 abs (x - 17.15) < ε :=
sorry

end NUMINAMATH_CALUDE_approx_value_of_625_power_l3392_339275


namespace NUMINAMATH_CALUDE_tea_trader_profit_percentage_tea_trader_profit_is_35_percent_l3392_339244

/-- Calculates the profit percentage for a tea trader --/
theorem tea_trader_profit_percentage 
  (tea1_weight : ℝ) (tea1_cost : ℝ) 
  (tea2_weight : ℝ) (tea2_cost : ℝ) 
  (sale_price : ℝ) : ℝ :=
  let total_weight := tea1_weight + tea2_weight
  let total_cost := tea1_weight * tea1_cost + tea2_weight * tea2_cost
  let total_sale := total_weight * sale_price
  let profit := total_sale - total_cost
  let profit_percentage := (profit / total_cost) * 100
  profit_percentage

/-- Proves that the profit percentage is 35% for the given scenario --/
theorem tea_trader_profit_is_35_percent : 
  tea_trader_profit_percentage 80 15 20 20 21.6 = 35 := by
  sorry

end NUMINAMATH_CALUDE_tea_trader_profit_percentage_tea_trader_profit_is_35_percent_l3392_339244


namespace NUMINAMATH_CALUDE_interest_difference_approximation_l3392_339266

/-- Calculates the balance after compound interest --/
def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- Calculates the balance difference between two accounts --/
def balance_difference (
  principal : ℝ)
  (rate1 : ℝ) (periods1 : ℕ)
  (rate2 : ℝ) (periods2 : ℕ) : ℝ :=
  compound_interest principal rate1 periods1 - compound_interest principal rate2 periods2

theorem interest_difference_approximation :
  let principal := 10000
  let rate_alice := 0.03  -- 6% / 2 for semiannual compounding
  let periods_alice := 20  -- 2 * 10 years
  let rate_bob := 0.04
  let periods_bob := 10
  abs (balance_difference principal rate_alice periods_alice rate_bob periods_bob - 3259) < 1 := by
  sorry

#eval balance_difference 10000 0.03 20 0.04 10

end NUMINAMATH_CALUDE_interest_difference_approximation_l3392_339266


namespace NUMINAMATH_CALUDE_base_three_sum_l3392_339234

/-- Represents a number in base 3 --/
def BaseThree : Type := List Nat

/-- Converts a base 3 number to its decimal representation --/
def toDecimal (n : BaseThree) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The theorem to prove --/
theorem base_three_sum :
  let a : BaseThree := [2]
  let b : BaseThree := [0, 2, 1]
  let c : BaseThree := [1, 2, 0, 2]
  let d : BaseThree := [2, 0, 1, 1]
  let result : BaseThree := [2, 2, 2, 2]
  toDecimal a + toDecimal b + toDecimal c + toDecimal d = toDecimal result :=
by sorry

end NUMINAMATH_CALUDE_base_three_sum_l3392_339234
