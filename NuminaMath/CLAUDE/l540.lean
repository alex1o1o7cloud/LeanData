import Mathlib

namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l540_54006

/-- A function that checks if a number is prime --/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers form a scalene triangle --/
def isScaleneTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ b + c > a ∧ a + c > b

/-- The main theorem --/
theorem smallest_prime_perimeter_scalene_triangle :
  ∃ (a b c : ℕ),
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    a > 3 ∧ b > 3 ∧ c > 3 ∧
    isScaleneTriangle a b c ∧
    isPrime (a + b + c) ∧
    (a + b + c = 23) ∧
    (∀ (x y z : ℕ),
      isPrime x ∧ isPrime y ∧ isPrime z ∧
      x > 3 ∧ y > 3 ∧ z > 3 ∧
      isScaleneTriangle x y z ∧
      isPrime (x + y + z) →
      x + y + z ≥ 23) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l540_54006


namespace NUMINAMATH_CALUDE_f_minimum_value_tangent_line_equation_l540_54098

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - x - 2 * Real.log x + 1/2

-- Theorem for the minimum value of f(x)
theorem f_minimum_value :
  ∃ (x_min : ℝ), x_min > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≥ f x_min ∧ f x_min = -2 * Real.log 2 + 1/2 :=
sorry

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (x₀ : ℝ), x₀ > 0 ∧
  (2 * x₀ + f x₀ - 2 = 0) ∧
  ∀ (x : ℝ), 2 * x + f x₀ + (x - x₀) * (x₀ - 1 - 2 / x₀) - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_tangent_line_equation_l540_54098


namespace NUMINAMATH_CALUDE_card_58_is_6_l540_54047

/-- The sequence of playing cards -/
def card_sequence : ℕ → ℕ :=
  fun n => (n - 1) % 13 + 1

/-- The 58th card in the sequence -/
def card_58 : ℕ := card_sequence 58

theorem card_58_is_6 : card_58 = 6 := by
  sorry

end NUMINAMATH_CALUDE_card_58_is_6_l540_54047


namespace NUMINAMATH_CALUDE_final_algae_count_l540_54057

/-- The number of algae plants in Milford Lake -/
def algae_count : ℕ → ℕ
| 0 => 809  -- Original count
| (n + 1) => algae_count n + 2454  -- Increase

theorem final_algae_count : algae_count 1 = 3263 := by
  sorry

end NUMINAMATH_CALUDE_final_algae_count_l540_54057


namespace NUMINAMATH_CALUDE_number_of_cut_cubes_l540_54017

/-- Represents a cube with integer edge length -/
structure Cube where
  edge : ℕ

/-- Represents a set of cubes resulting from cutting a larger cube -/
structure CutCube where
  original : Cube
  pieces : List Cube
  all_same_size : Bool

/-- The volume of a cube -/
def volume (c : Cube) : ℕ := c.edge ^ 3

/-- The total volume of a list of cubes -/
def total_volume (cubes : List Cube) : ℕ :=
  cubes.map volume |>.sum

/-- Theorem: The number of smaller cubes obtained by cutting a 4cm cube is 57 -/
theorem number_of_cut_cubes : ∃ (cut : CutCube), 
  cut.original.edge = 4 ∧ 
  cut.all_same_size = false ∧
  (∀ c ∈ cut.pieces, c.edge > 0) ∧
  total_volume cut.pieces = volume cut.original ∧
  cut.pieces.length = 57 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cut_cubes_l540_54017


namespace NUMINAMATH_CALUDE_total_money_l540_54042

/-- Given that r has two-thirds of the total amount and r has $2800, 
    prove that the total amount of money p, q, and r have among themselves is $4200. -/
theorem total_money (r_share : ℚ) (r_amount : ℕ) (total : ℕ) : 
  r_share = 2/3 → r_amount = 2800 → total = r_amount * 3/2 → total = 4200 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l540_54042


namespace NUMINAMATH_CALUDE_fourth_root_equation_l540_54008

theorem fourth_root_equation (m : ℝ) : (m^4)^(1/4) = 2 → m = 2 ∨ m = -2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_l540_54008


namespace NUMINAMATH_CALUDE_money_sharing_l540_54048

theorem money_sharing (john_share : ℕ) (jose_share : ℕ) (binoy_share : ℕ) 
  (h1 : john_share = 1400)
  (h2 : jose_share = 2 * john_share)
  (h3 : binoy_share = 3 * john_share) :
  john_share + jose_share + binoy_share = 8400 := by
  sorry

end NUMINAMATH_CALUDE_money_sharing_l540_54048


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l540_54007

theorem fraction_to_decimal : (13 : ℚ) / (2 * 5^8) = 0.00001664 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l540_54007


namespace NUMINAMATH_CALUDE_angle_bisector_length_l540_54019

/-- The length of an angle bisector in a triangle -/
theorem angle_bisector_length (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (htri : a < b + c ∧ b < a + c ∧ c < a + b) :
  let p := (a + b + c) / 2
  ∃ l_a : ℝ, l_a = (2 / (b + c)) * Real.sqrt (b * c * p * (p - a)) := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l540_54019


namespace NUMINAMATH_CALUDE_path_combinations_l540_54079

theorem path_combinations (ways_AB ways_BC : ℕ) (h1 : ways_AB = 2) (h2 : ways_BC = 3) :
  ways_AB * ways_BC = 6 := by
sorry

end NUMINAMATH_CALUDE_path_combinations_l540_54079


namespace NUMINAMATH_CALUDE_quadratic_properties_l540_54013

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  pass_through_minus_one : a * (-1)^2 + b * (-1) + c = 0
  pass_through_zero : c = -1.5
  pass_through_one : a + b + c = -2
  pass_through_two : 4 * a + 2 * b + c = -1.5

theorem quadratic_properties (f : QuadraticFunction) :
  (∃ a' : ℝ, ∀ x, f.a * x^2 + f.b * x + f.c = a' * (x - 1)^2 - 2) ∧
  (f.a * 0^2 + f.b * 0 + f.c + 1.5 = 0 ∧ f.a * 2^2 + f.b * 2 + f.c + 1.5 = 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l540_54013


namespace NUMINAMATH_CALUDE_sphere_volume_in_cube_l540_54046

/-- The volume of a sphere inscribed in a cube with surface area 24 cm² is (4/3)π cm³ -/
theorem sphere_volume_in_cube (cube_surface_area : ℝ) (sphere_volume : ℝ) : 
  cube_surface_area = 24 →
  sphere_volume = (4/3) * Real.pi := by
  sorry

#check sphere_volume_in_cube

end NUMINAMATH_CALUDE_sphere_volume_in_cube_l540_54046


namespace NUMINAMATH_CALUDE_prob_five_odd_in_seven_rolls_l540_54082

/-- The probability of getting an odd number on a single roll of a fair 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of rolls -/
def num_rolls : ℕ := 7

/-- The number of successful rolls (odd numbers) we want -/
def num_success : ℕ := 5

/-- The probability of getting exactly 5 odd numbers in 7 rolls of a fair 6-sided die -/
theorem prob_five_odd_in_seven_rolls :
  (Nat.choose num_rolls num_success : ℚ) * prob_odd ^ num_success * (1 - prob_odd) ^ (num_rolls - num_success) = 21/128 :=
sorry

end NUMINAMATH_CALUDE_prob_five_odd_in_seven_rolls_l540_54082


namespace NUMINAMATH_CALUDE_bong_paint_time_l540_54061

def jay_time : ℝ := 2
def combined_time : ℝ := 1.2

theorem bong_paint_time :
  ∀ bong_time : ℝ,
  (1 / jay_time + 1 / bong_time = 1 / combined_time) →
  bong_time = 3 := by
sorry

end NUMINAMATH_CALUDE_bong_paint_time_l540_54061


namespace NUMINAMATH_CALUDE_two_digit_product_1365_l540_54081

/-- Represents a two-digit number --/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : 1 ≤ tens ∧ tens ≤ 9 ∧ 0 ≤ ones ∧ ones ≤ 9

/-- Converts a TwoDigitNumber to a natural number --/
def TwoDigitNumber.toNat (n : TwoDigitNumber) : Nat :=
  10 * n.tens + n.ones

theorem two_digit_product_1365 :
  ∀ (ab cd : TwoDigitNumber),
    ab.toNat * cd.toNat = 1365 →
    ab.tens ≠ ab.ones →
    cd.tens ≠ cd.ones →
    ab.tens ≠ cd.tens →
    ab.tens ≠ cd.ones →
    ab.ones ≠ cd.tens →
    ab.ones ≠ cd.ones →
    ((ab.tens = 2 ∧ ab.ones = 1) ∧ (cd.tens = 6 ∧ cd.ones = 5)) ∨
    ((ab.tens = 6 ∧ ab.ones = 5) ∧ (cd.tens = 2 ∧ cd.ones = 1)) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_product_1365_l540_54081


namespace NUMINAMATH_CALUDE_chess_team_boys_l540_54056

/-- Represents the number of boys on a chess team --/
def num_boys (total : ℕ) (attendees : ℕ) : ℕ :=
  total - 2 * (total - attendees)

/-- Theorem stating the number of boys on the chess team --/
theorem chess_team_boys (total : ℕ) (attendees : ℕ) 
  (h_total : total = 30)
  (h_attendees : attendees = 18)
  (h_attendance : ∃ (girls : ℕ), girls + (total - girls) = total ∧ 
                                  girls / 3 + (total - girls) = attendees) :
  num_boys total attendees = 12 := by
sorry

#eval num_boys 30 18  -- Should output 12

end NUMINAMATH_CALUDE_chess_team_boys_l540_54056


namespace NUMINAMATH_CALUDE_cloth_sale_profit_per_meter_l540_54053

/-- Calculates the profit per meter of cloth given the total length sold,
    total selling price, and cost price per meter. -/
def profit_per_meter (total_length : ℕ) (total_selling_price : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  (total_selling_price - total_length * cost_price_per_meter) / total_length

/-- Proves that for the given cloth sale, the profit per meter is 25 rupees. -/
theorem cloth_sale_profit_per_meter :
  profit_per_meter 85 8925 80 = 25 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_profit_per_meter_l540_54053


namespace NUMINAMATH_CALUDE_find_B_value_l540_54051

theorem find_B_value (A B : ℕ) : 
  (100 ≤ 6 * 100 + A * 10 + 5) ∧ (6 * 100 + A * 10 + 5 < 1000) ∧ 
  (100 ≤ 1 * 100 + 0 * 10 + B) ∧ (1 * 100 + 0 * 10 + B < 1000) ∧
  (6 * 100 + A * 10 + 5 + 1 * 100 + 0 * 10 + B = 748) →
  B = 3 := by sorry

end NUMINAMATH_CALUDE_find_B_value_l540_54051


namespace NUMINAMATH_CALUDE_two_valid_m_values_l540_54024

/-- A right triangle in the coordinate plane with legs parallel to the axes -/
structure RightTriangle where
  a : ℝ  -- x-coordinate of the point on the x-axis
  b : ℝ  -- y-coordinate of the point on the y-axis

/-- Check if the given m value satisfies the conditions for the right triangle -/
def satisfiesConditions (t : RightTriangle) (m : ℝ) : Prop :=
  3 * (t.a / 2) + 1 = 0 ∧  -- Condition for the line y = 3x + 1
  t.b / 2 = 2 ∧           -- Condition for the line y = mx + 2
  (t.b / 2) / (t.a / 2) = 4  -- Condition for the ratio of slopes

/-- The theorem stating that there are exactly two values of m that satisfy the conditions -/
theorem two_valid_m_values :
  ∃ m₁ m₂ : ℝ,
    m₁ ≠ m₂ ∧
    (∃ t : RightTriangle, satisfiesConditions t m₁) ∧
    (∃ t : RightTriangle, satisfiesConditions t m₂) ∧
    (∀ m : ℝ, (∃ t : RightTriangle, satisfiesConditions t m) → m = m₁ ∨ m = m₂) :=
  sorry

end NUMINAMATH_CALUDE_two_valid_m_values_l540_54024


namespace NUMINAMATH_CALUDE_S_is_valid_set_l540_54064

-- Define the set of non-negative integers not exceeding 10
def S : Set ℕ := {n : ℕ | n ≤ 10}

-- Theorem stating that S is a valid set
theorem S_is_valid_set :
  -- S has definite elements
  (∀ n : ℕ, n ∈ S ↔ n ≤ 10) ∧
  -- S has disordered elements (always true for sets)
  True ∧
  -- S has distinct elements (follows from the definition of ℕ)
  (∀ a b : ℕ, a ∈ S → b ∈ S → a = b → a = b) :=
sorry

end NUMINAMATH_CALUDE_S_is_valid_set_l540_54064


namespace NUMINAMATH_CALUDE_books_on_shelf_l540_54090

/-- The number of books on a shelf after adding more books is equal to the sum of the initial number of books and the number of books added. -/
theorem books_on_shelf (initial_books additional_books : ℕ) :
  initial_books + additional_books = initial_books + additional_books :=
by sorry

end NUMINAMATH_CALUDE_books_on_shelf_l540_54090


namespace NUMINAMATH_CALUDE_adjacent_supplementary_not_always_complementary_l540_54012

-- Define supplementary angles
def supplementary (α β : Real) : Prop := α + β = 180

-- Define complementary angles
def complementary (α β : Real) : Prop := α + β = 90

-- Theorem statement
theorem adjacent_supplementary_not_always_complementary :
  ∃ α β : Real, supplementary α β ∧ ¬complementary α β :=
sorry

end NUMINAMATH_CALUDE_adjacent_supplementary_not_always_complementary_l540_54012


namespace NUMINAMATH_CALUDE_gcd_840_1764_l540_54025

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l540_54025


namespace NUMINAMATH_CALUDE_kayak_rental_cost_l540_54062

/-- Represents the daily rental cost of a kayak -/
def kayak_cost : ℝ := 18

/-- Represents the daily rental cost of a canoe -/
def canoe_cost : ℝ := 15

/-- Represents the number of kayaks rented -/
def num_kayaks : ℕ := 10

/-- Represents the number of canoes rented -/
def num_canoes : ℕ := 15

/-- Represents the total revenue for one day -/
def total_revenue : ℝ := 405

theorem kayak_rental_cost :
  (kayak_cost * num_kayaks + canoe_cost * num_canoes = total_revenue) ∧
  (num_canoes = num_kayaks + 5) ∧
  (3 * num_kayaks = 2 * num_canoes) :=
by sorry

end NUMINAMATH_CALUDE_kayak_rental_cost_l540_54062


namespace NUMINAMATH_CALUDE_secondPlayerCanEnsureDivisibilityFor60_secondPlayerCannotEnsureDivisibilityFor14_l540_54089

/-- Represents a strategy for the second player to choose digits -/
def Strategy := Nat → Nat → Nat

/-- Checks if a list of digits is divisible by 9 -/
def isDivisibleBy9 (digits : List Nat) : Prop :=
  (digits.sum % 9) = 0

/-- Generates all possible sequences of digits for the first player -/
def firstPlayerSequences (n : Nat) : List (List Nat) :=
  sorry

/-- Applies the second player's strategy to the first player's sequence -/
def applyStrategy (firstPlayerSeq : List Nat) (strategy : Strategy) : List Nat :=
  sorry

theorem secondPlayerCanEnsureDivisibilityFor60 :
  ∃ (strategy : Strategy),
    ∀ (firstPlayerSeq : List Nat),
      firstPlayerSeq.length = 30 →
      firstPlayerSeq.all (λ d => d ≥ 1 ∧ d ≤ 5) →
      isDivisibleBy9 (applyStrategy firstPlayerSeq strategy) :=
sorry

theorem secondPlayerCannotEnsureDivisibilityFor14 :
  ∀ (strategy : Strategy),
    ∃ (firstPlayerSeq : List Nat),
      firstPlayerSeq.length = 7 →
      firstPlayerSeq.all (λ d => d ≥ 1 ∧ d ≤ 5) →
      ¬isDivisibleBy9 (applyStrategy firstPlayerSeq strategy) :=
sorry

end NUMINAMATH_CALUDE_secondPlayerCanEnsureDivisibilityFor60_secondPlayerCannotEnsureDivisibilityFor14_l540_54089


namespace NUMINAMATH_CALUDE_moscow_olympiad_1975_l540_54092

theorem moscow_olympiad_1975 (a b c p q r : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r →
  p = b^c + a →
  q = a^b + c →
  r = c^a + b →
  q = r := by
sorry

end NUMINAMATH_CALUDE_moscow_olympiad_1975_l540_54092


namespace NUMINAMATH_CALUDE_same_parity_min_max_l540_54052

/-- A set with elements related to positioning in a function or polynomial -/
def A_P : Set ℤ := sorry

/-- The smallest element of A_P -/
def smallest (A : Set ℤ) : ℤ := sorry

/-- The largest element of A_P -/
def largest (A : Set ℤ) : ℤ := sorry

/-- A function to determine if a number is even -/
def isEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem same_parity_min_max : isEven (smallest A_P) ↔ isEven (largest A_P) := by sorry

end NUMINAMATH_CALUDE_same_parity_min_max_l540_54052


namespace NUMINAMATH_CALUDE_min_value_of_f_l540_54060

noncomputable def f (a : ℝ) : ℝ := a/2 - 1/4 + (Real.exp (-2*a))/2

theorem min_value_of_f :
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ (b : ℝ), b > 0 → f a ≤ f b) ∧
  a = Real.log 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l540_54060


namespace NUMINAMATH_CALUDE_subtract_fraction_from_decimal_l540_54055

theorem subtract_fraction_from_decimal : 7.31 - (1 / 5 : ℚ) = 7.11 := by sorry

end NUMINAMATH_CALUDE_subtract_fraction_from_decimal_l540_54055


namespace NUMINAMATH_CALUDE_S_max_at_9_l540_54095

/-- An arithmetic sequence -/
def arithmetic_sequence : ℕ → ℝ := sorry

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- Conditions of the problem -/
axiom S_18_positive : S 18 > 0
axiom S_19_negative : S 19 < 0

/-- Theorem: S_n is maximum when n = 9 -/
theorem S_max_at_9 : ∀ k : ℕ, S 9 ≥ S k := by sorry

end NUMINAMATH_CALUDE_S_max_at_9_l540_54095


namespace NUMINAMATH_CALUDE_negation_inverse_implies_contrapositive_l540_54088

-- Define propositions as functions from some universe U to Prop
variable {U : Type}
variable (p q r : U → Prop)

-- Define the negation relation
def is_negation (p q : U → Prop) : Prop :=
  ∀ x, q x ↔ ¬(p x)

-- Define the inverse relation
def is_inverse (q r : U → Prop) : Prop :=
  ∀ x, r x ↔ (¬q x)

-- Define the contrapositive relation
def is_contrapositive (p r : U → Prop) : Prop :=
  ∀ x y, (p x → p y) ↔ (¬p y → ¬p x)

-- The main theorem
theorem negation_inverse_implies_contrapositive (p q r : U → Prop) :
  is_negation p q → is_inverse q r → is_contrapositive p r :=
sorry

end NUMINAMATH_CALUDE_negation_inverse_implies_contrapositive_l540_54088


namespace NUMINAMATH_CALUDE_line_through_points_with_slope_l540_54000

theorem line_through_points_with_slope (k : ℝ) : 
  (∃ (m : ℝ), m = (3 * k - (-9)) / (7 - k) ∧ m = 2 * k) → 
  k = 9 / 2 ∨ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_with_slope_l540_54000


namespace NUMINAMATH_CALUDE_regular_octagon_interior_angle_l540_54078

/-- The measure of each interior angle in a regular octagon is 135 degrees. -/
theorem regular_octagon_interior_angle : ℝ := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_interior_angle_l540_54078


namespace NUMINAMATH_CALUDE_subtraction_absolute_value_l540_54004

theorem subtraction_absolute_value : ∃ (x y : ℝ), 
  (|9 - 4| - |x - y| = 3) ∧ (|x - y| = 2) :=
by sorry

end NUMINAMATH_CALUDE_subtraction_absolute_value_l540_54004


namespace NUMINAMATH_CALUDE_no_real_solutions_l540_54037

theorem no_real_solutions : 
  ¬∃ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l540_54037


namespace NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l540_54029

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 723 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l540_54029


namespace NUMINAMATH_CALUDE_bike_ride_percentage_increase_l540_54026

theorem bike_ride_percentage_increase (d1 d2 d3 : ℝ) : 
  d2 = 12 →                   -- Second hour distance is 12 miles
  d2 = 1.2 * d1 →             -- Second hour is 20% farther than first hour
  d1 + d2 + d3 = 37 →         -- Total distance is 37 miles
  (d3 - d2) / d2 * 100 = 25   -- Percentage increase from second to third hour is 25%
  := by sorry

end NUMINAMATH_CALUDE_bike_ride_percentage_increase_l540_54026


namespace NUMINAMATH_CALUDE_blackjack_payout_40_dollars_l540_54002

/-- Calculates the total amount received for a blackjack bet -/
def blackjack_payout (bet : ℚ) (payout_ratio : ℚ × ℚ) : ℚ :=
  bet + bet * (payout_ratio.1 / payout_ratio.2)

/-- Theorem: The total amount received for a $40 blackjack bet with 3:2 payout is $100 -/
theorem blackjack_payout_40_dollars :
  blackjack_payout 40 (3, 2) = 100 := by
  sorry

end NUMINAMATH_CALUDE_blackjack_payout_40_dollars_l540_54002


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l540_54066

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), (4 * a) % 65 = 1 ∧ 
                (13 * b) % 65 = 1 ∧ 
                (3 * a + 12 * b) % 65 = 42 :=
by sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l540_54066


namespace NUMINAMATH_CALUDE_tangent_line_equation_l540_54075

theorem tangent_line_equation (x y : ℝ) :
  x < 0 ∧ y > 0 ∧  -- P is in the second quadrant
  y = x^3 - 10*x + 3 ∧  -- P is on the curve
  3*x^2 - 10 = 2  -- Slope of tangent line is 2
  →
  ∃ (a b : ℝ), a = 2 ∧ b = 19 ∧ ∀ (x' y' : ℝ), y' = a*x' + b  -- Equation of tangent line
  :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l540_54075


namespace NUMINAMATH_CALUDE_train_bus_cost_difference_proof_l540_54099

def train_bus_cost_difference (train_cost bus_cost : ℝ) : Prop :=
  (train_cost > bus_cost) ∧
  (train_cost + bus_cost = 9.65) ∧
  (bus_cost = 1.40) ∧
  (train_cost - bus_cost = 6.85)

theorem train_bus_cost_difference_proof :
  ∃ (train_cost bus_cost : ℝ), train_bus_cost_difference train_cost bus_cost :=
by sorry

end NUMINAMATH_CALUDE_train_bus_cost_difference_proof_l540_54099


namespace NUMINAMATH_CALUDE_distance_between_points_l540_54001

def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (8, 9)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.1)^2) = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l540_54001


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l540_54068

/-- The area of the shaded region formed by a sector of a circle and an equilateral triangle -/
theorem shaded_area_calculation (r : ℝ) (θ : ℝ) (a : ℝ) (h1 : r = 12) (h2 : θ = 112) (h3 : a = 12) :
  let sector_area := (θ / 360) * π * r^2
  let triangle_area := (Real.sqrt 3 / 4) * a^2
  abs ((sector_area - triangle_area) - 78.0211) < 0.0001 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l540_54068


namespace NUMINAMATH_CALUDE_tangent_y_intercept_l540_54005

-- Define the circles
def circle1_center : ℝ × ℝ := (3, 0)
def circle1_radius : ℝ := 3
def circle2_center : ℝ × ℝ := (8, 0)
def circle2_radius : ℝ := 2

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) := sorry

-- Define the property of being tangent to a circle in the fourth quadrant
def is_tangent_in_fourth_quadrant (line : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ) : Prop := sorry

-- Theorem statement
theorem tangent_y_intercept :
  is_tangent_in_fourth_quadrant tangent_line circle1_center circle1_radius ∧
  is_tangent_in_fourth_quadrant tangent_line circle2_center circle2_radius →
  ∃ (y : ℝ), y = 6/5 ∧ (0, y) ∈ tangent_line :=
sorry

end NUMINAMATH_CALUDE_tangent_y_intercept_l540_54005


namespace NUMINAMATH_CALUDE_exponent_equality_l540_54010

theorem exponent_equality : 
  ((-2 : ℤ)^3 ≠ (-3 : ℤ)^2) ∧ 
  (-(3 : ℤ)^2 ≠ (-3 : ℤ)^2) ∧ 
  (-(3 : ℤ)^3 = (-3 : ℤ)^3) ∧ 
  (-(3 : ℤ) * (2 : ℤ)^3 ≠ (-3 * 2 : ℤ)^3) :=
by sorry

end NUMINAMATH_CALUDE_exponent_equality_l540_54010


namespace NUMINAMATH_CALUDE_g_composition_value_l540_54018

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^2 - 6

-- State the theorem
theorem g_composition_value : g (g 2) = 394 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_value_l540_54018


namespace NUMINAMATH_CALUDE_total_students_calculation_l540_54058

theorem total_students_calculation (short_ratio : Rat) (tall_count : Nat) (average_count : Nat) :
  short_ratio = 2/5 →
  tall_count = 90 →
  average_count = 150 →
  ∃ (total : Nat), total = (tall_count + average_count) / (1 - short_ratio) ∧ total = 400 :=
by sorry

end NUMINAMATH_CALUDE_total_students_calculation_l540_54058


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l540_54086

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l540_54086


namespace NUMINAMATH_CALUDE_smallest_with_20_divisors_l540_54016

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Check if a number has exactly 20 positive divisors -/
def has_20_divisors (n : ℕ+) : Prop := num_divisors n = 20

theorem smallest_with_20_divisors :
  ∃ (n : ℕ+), has_20_divisors n ∧ ∀ (m : ℕ+), has_20_divisors m → n ≤ m :=
  sorry

end NUMINAMATH_CALUDE_smallest_with_20_divisors_l540_54016


namespace NUMINAMATH_CALUDE_min_balls_theorem_l540_54027

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The total number of balls in the box -/
def total_balls (counts : BallCounts) : Nat :=
  counts.red + counts.green + counts.yellow + counts.blue + counts.white + counts.black

/-- The minimum number of balls to draw to ensure at least n are of the same color -/
def min_balls_to_draw (counts : BallCounts) (n : Nat) : Nat :=
  sorry

theorem min_balls_theorem (counts : BallCounts) (n : Nat) :
  counts.red = 28 →
  counts.green = 20 →
  counts.yellow = 12 →
  counts.blue = 20 →
  counts.white = 10 →
  counts.black = 10 →
  total_balls counts = 100 →
  min_balls_to_draw counts 15 = 75 :=
by sorry

end NUMINAMATH_CALUDE_min_balls_theorem_l540_54027


namespace NUMINAMATH_CALUDE_winnie_the_pooh_escalator_steps_l540_54038

theorem winnie_the_pooh_escalator_steps :
  ∀ (u v L : ℝ),
    u > 0 →
    v > 0 →
    L > 0 →
    (L * u) / (u + v) = 55 →
    (L * u) / (u - v) = 1155 →
    L = 105 :=
by
  sorry

end NUMINAMATH_CALUDE_winnie_the_pooh_escalator_steps_l540_54038


namespace NUMINAMATH_CALUDE_total_trees_cut_is_1021_l540_54020

/-- Calculates the total number of trees cut down by James and his helpers. -/
def total_trees_cut (james_rate : ℕ) (brother_rate : ℕ) (cousin_rate : ℕ) (professional_rate : ℕ) : ℕ :=
  let james_alone := 2 * james_rate
  let with_brothers := 3 * (james_rate + 2 * brother_rate)
  let with_cousin := 4 * (james_rate + 2 * brother_rate + cousin_rate)
  let all_together := 5 * (james_rate + 2 * brother_rate + cousin_rate + professional_rate)
  james_alone + with_brothers + with_cousin + all_together

/-- The theorem states that the total number of trees cut down is 1021. -/
theorem total_trees_cut_is_1021 :
  total_trees_cut 20 16 23 30 = 1021 := by
  sorry

#eval total_trees_cut 20 16 23 30

end NUMINAMATH_CALUDE_total_trees_cut_is_1021_l540_54020


namespace NUMINAMATH_CALUDE_boys_who_love_marbles_l540_54093

/-- The number of marbles Haley has -/
def total_marbles : ℕ := 20

/-- The number of marbles each boy receives -/
def marbles_per_boy : ℕ := 10

/-- The number of boys who love to play marbles -/
def num_boys : ℕ := total_marbles / marbles_per_boy

theorem boys_who_love_marbles : num_boys = 2 := by
  sorry

end NUMINAMATH_CALUDE_boys_who_love_marbles_l540_54093


namespace NUMINAMATH_CALUDE_equation_solution_l540_54059

theorem equation_solution :
  ∀ x : ℝ, x^6 - 19*x^3 = 216 ↔ x = 3 ∨ x = -2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l540_54059


namespace NUMINAMATH_CALUDE_laura_savings_l540_54022

def original_price : ℝ := 3.00
def discount_rate : ℝ := 0.30
def num_notebooks : ℕ := 7

theorem laura_savings : 
  (num_notebooks : ℝ) * original_price * discount_rate = 6.30 := by
  sorry

end NUMINAMATH_CALUDE_laura_savings_l540_54022


namespace NUMINAMATH_CALUDE_intersection_when_a_neg_two_subset_condition_l540_54069

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | (x - (a + 5)) / (x - a) > 0}

-- Theorem for part 1
theorem intersection_when_a_neg_two :
  A ∩ B (-2) = {x | 3 < x ∧ x ≤ 4} := by sorry

-- Theorem for part 2
theorem subset_condition (a : ℝ) :
  A ⊆ B a ↔ a < -6 ∨ a > 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_neg_two_subset_condition_l540_54069


namespace NUMINAMATH_CALUDE_dot_product_range_l540_54084

-- Define the points A and B
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, -1)

-- Define the curve y = √(1-x^2)
def on_curve (P : ℝ × ℝ) : Prop :=
  P.2 = Real.sqrt (1 - P.1^2)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Define the vector from B to P
def BP (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1 - B.1, P.2 - B.2)

-- Define the vector from B to A
def BA : ℝ × ℝ :=
  (A.1 - B.1, A.2 - B.2)

-- The main theorem
theorem dot_product_range :
  ∀ P : ℝ × ℝ, on_curve P →
  0 ≤ dot_product (BP P) BA ∧ dot_product (BP P) BA ≤ 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l540_54084


namespace NUMINAMATH_CALUDE_Q_zeros_count_l540_54028

noncomputable def Q (x : ℝ) : ℂ :=
  2 + Complex.exp (Complex.I * x) - 2 * Complex.exp (2 * Complex.I * x) + Complex.exp (3 * Complex.I * x)

theorem Q_zeros_count : ∃! (s : Finset ℝ), s.card = 2 ∧ (∀ x ∈ s, 0 ≤ x ∧ x < 4 * Real.pi ∧ Q x = 0) ∧ (∀ x, 0 ≤ x → x < 4 * Real.pi → Q x = 0 → x ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_Q_zeros_count_l540_54028


namespace NUMINAMATH_CALUDE_prime_iff_sum_four_integers_l540_54096

theorem prime_iff_sum_four_integers (n : ℕ) (h : n ≥ 5) :
  Nat.Prime n ↔ ∀ (a b c d : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → n = a + b + c + d → a * b ≠ c * d := by
  sorry

end NUMINAMATH_CALUDE_prime_iff_sum_four_integers_l540_54096


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_28_l540_54076

/-- Represents a triangle partitioned into three triangles and a quadrilateral -/
structure PartitionedTriangle where
  /-- Area of the first small triangle -/
  area1 : ℝ
  /-- Area of the second small triangle -/
  area2 : ℝ
  /-- Area of the third small triangle -/
  area3 : ℝ
  /-- Area of the quadrilateral -/
  areaQuad : ℝ

/-- The theorem stating that if the areas of the three triangles are 4, 8, and 8,
    then the area of the quadrilateral is 28 -/
theorem quadrilateral_area_is_28 (t : PartitionedTriangle) 
  (h1 : t.area1 = 4) 
  (h2 : t.area2 = 8) 
  (h3 : t.area3 = 8) : 
  t.areaQuad = 28 := by
  sorry


end NUMINAMATH_CALUDE_quadrilateral_area_is_28_l540_54076


namespace NUMINAMATH_CALUDE_cost_price_proof_l540_54040

/-- The cost price of an article satisfying the given profit and loss conditions. -/
def cost_price : ℝ := 49

/-- The selling price that results in a profit. -/
def profit_price : ℝ := 56

/-- The selling price that results in a loss. -/
def loss_price : ℝ := 42

theorem cost_price_proof :
  (profit_price - cost_price = cost_price - loss_price) →
  cost_price = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_cost_price_proof_l540_54040


namespace NUMINAMATH_CALUDE_missing_number_proof_l540_54035

theorem missing_number_proof (numbers : List ℕ) (h_count : numbers.length = 9) 
  (h_sum : numbers.sum = 744 + 745 + 747 + 749 + 752 + 752 + 753 + 755 + 755) 
  (h_avg : (numbers.sum + missing) / 10 = 750) : missing = 1748 := by
  sorry

#check missing_number_proof

end NUMINAMATH_CALUDE_missing_number_proof_l540_54035


namespace NUMINAMATH_CALUDE_trapezoid_height_l540_54083

/-- Given S = (1/2)(a+b)h and a+b ≠ 0, prove that h = 2S / (a+b) -/
theorem trapezoid_height (a b S h : ℝ) (h_eq : S = (1/2) * (a + b) * h) (h_ne_zero : a + b ≠ 0) :
  h = 2 * S / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_height_l540_54083


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l540_54044

theorem cubic_equation_solution : 
  ∃! x : ℝ, x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3 ∧ x^2 + 4*x + 4 > 0 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l540_54044


namespace NUMINAMATH_CALUDE_min_toothpicks_to_remove_correct_l540_54094

/-- Represents a figure made of toothpicks and triangles -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  upward_triangles : ℕ
  downward_triangles : ℕ
  upward_side_length : ℕ
  downward_side_length : ℕ

/-- The minimum number of toothpicks to remove to eliminate all triangles -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ :=
  figure.upward_triangles * figure.upward_side_length

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_to_remove_correct (figure : ToothpickFigure) 
  (h1 : figure.total_toothpicks = 40)
  (h2 : figure.upward_triangles = 10)
  (h3 : figure.downward_triangles = 8)
  (h4 : figure.upward_side_length = 2)
  (h5 : figure.downward_side_length = 1) :
  min_toothpicks_to_remove figure = 20 := by
  sorry

#eval min_toothpicks_to_remove {
  total_toothpicks := 40,
  upward_triangles := 10,
  downward_triangles := 8,
  upward_side_length := 2,
  downward_side_length := 1
}

end NUMINAMATH_CALUDE_min_toothpicks_to_remove_correct_l540_54094


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l540_54034

theorem polynomial_division_theorem (x : ℝ) : 
  (4 * x^3 + x^2 + 2 * x + 3) * (3 * x - 2) + 11 = 
  12 * x^4 - 9 * x^3 + 6 * x^2 + 11 * x - 3 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l540_54034


namespace NUMINAMATH_CALUDE_movie_profit_calculation_l540_54080

def movie_profit (actor_cost food_cost_per_person num_people equipment_rental_factor selling_price : ℚ) : ℚ :=
  let food_cost := food_cost_per_person * num_people
  let total_food_and_actors := actor_cost + food_cost
  let equipment_cost := equipment_rental_factor * total_food_and_actors
  let total_cost := actor_cost + food_cost + equipment_cost
  selling_price - total_cost

theorem movie_profit_calculation :
  movie_profit 1200 3 50 2 10000 = 5950 :=
by sorry

end NUMINAMATH_CALUDE_movie_profit_calculation_l540_54080


namespace NUMINAMATH_CALUDE_periodic_scaled_function_l540_54085

-- Define a real-valued function with period T
def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

-- Define F(x) = f(αx)
def F (f : ℝ → ℝ) (α : ℝ) (x : ℝ) : ℝ := f (α * x)

-- Theorem statement
theorem periodic_scaled_function
  (f : ℝ → ℝ) (T α : ℝ) (h_periodic : is_periodic f T) (h_pos : α > 0) :
  is_periodic (F f α) (T / α) :=
sorry

end NUMINAMATH_CALUDE_periodic_scaled_function_l540_54085


namespace NUMINAMATH_CALUDE_fraction_sum_difference_l540_54009

theorem fraction_sum_difference : 2/5 + 3/8 - 1/10 = 27/40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_l540_54009


namespace NUMINAMATH_CALUDE_three_intersection_points_l540_54050

-- Define the four lines
def line1 (x y : ℝ) : Prop := 3 * y - 2 * x = 1
def line2 (x y : ℝ) : Prop := x + 2 * y = 2
def line3 (x y : ℝ) : Prop := 4 * x - 6 * y = 5
def line4 (x y : ℝ) : Prop := 2 * x - 3 * y = 4

-- Define a function to check if a point lies on a line
def point_on_line (x y : ℝ) (line : ℝ → ℝ → Prop) : Prop := line x y

-- Define a function to check if a point is an intersection of at least two lines
def is_intersection (x y : ℝ) : Prop :=
  (point_on_line x y line1 ∧ point_on_line x y line2) ∨
  (point_on_line x y line1 ∧ point_on_line x y line3) ∨
  (point_on_line x y line1 ∧ point_on_line x y line4) ∨
  (point_on_line x y line2 ∧ point_on_line x y line3) ∨
  (point_on_line x y line2 ∧ point_on_line x y line4) ∨
  (point_on_line x y line3 ∧ point_on_line x y line4)

-- Theorem stating that there are exactly 3 distinct intersection points
theorem three_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    is_intersection p3.1 p3.2 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 ∨ (x, y) = p3 :=
by sorry


end NUMINAMATH_CALUDE_three_intersection_points_l540_54050


namespace NUMINAMATH_CALUDE_factor_condition_l540_54003

theorem factor_condition (t : ℚ) :
  (∃ k : ℚ, ∀ x, 4*x^2 + 11*x - 3 = (x - t) * k) ↔ (t = 1/4 ∨ t = -3) := by
sorry

end NUMINAMATH_CALUDE_factor_condition_l540_54003


namespace NUMINAMATH_CALUDE_at_most_one_tiling_l540_54032

/-- Represents a polyomino -/
structure Polyomino where
  squares : Set (ℕ × ℕ)
  nonempty : squares.Nonempty

/-- An L-shaped polyomino consisting of three squares -/
def l_shape : Polyomino := {
  squares := {(0,0), (0,1), (1,0)}
  nonempty := by simp
}

/-- Another polyomino with at least two squares -/
def other_polyomino : Polyomino := {
  squares := {(0,0), (0,1)}  -- Minimal example with two squares
  nonempty := by simp
}

/-- Represents a tiling of a board -/
def Tiling (n : ℕ) (p1 p2 : Polyomino) :=
  ∃ (t : Set (ℕ × ℕ × ℕ × ℕ)), 
    (∀ x y, x < n ∧ y < n → ∃ a b dx dy, (a, b, dx, dy) ∈ t ∧
      ((dx, dy) ∈ p1.squares ∨ (dx, dy) ∈ p2.squares) ∧
      (a + dx = x ∧ b + dy = y)) ∧
    (∀ (a b dx dy : ℕ), (a, b, dx, dy) ∈ t →
      (dx, dy) ∈ p1.squares ∨ (dx, dy) ∈ p2.squares)

/-- The main theorem -/
theorem at_most_one_tiling (n m : ℕ) (h : Nat.Coprime n m) :
  ¬(Tiling n l_shape other_polyomino ∧ Tiling m l_shape other_polyomino) := by
  sorry

end NUMINAMATH_CALUDE_at_most_one_tiling_l540_54032


namespace NUMINAMATH_CALUDE_systematic_sampling_groups_for_56_and_8_l540_54041

/-- Calculates the number of groups formed in systematic sampling -/
def systematicSamplingGroups (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  populationSize / sampleSize

theorem systematic_sampling_groups_for_56_and_8 :
  systematicSamplingGroups 56 8 = 8 := by
  sorry

#eval systematicSamplingGroups 56 8

end NUMINAMATH_CALUDE_systematic_sampling_groups_for_56_and_8_l540_54041


namespace NUMINAMATH_CALUDE_average_side_lengths_of_squares_l540_54065

theorem average_side_lengths_of_squares (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ = 25) (h₂ : a₂ = 36) (h₃ : a₃ = 64) (h₄ : a₄ = 144) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃ + Real.sqrt a₄) / 4 = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_average_side_lengths_of_squares_l540_54065


namespace NUMINAMATH_CALUDE_expression_evaluation_l540_54067

theorem expression_evaluation : 
  let x : ℚ := 1/2
  (((x^2 - 2*x + 1) / (x^2 - 1) - 1 / (x + 1)) / ((2*x - 4) / (x^2 + x))) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l540_54067


namespace NUMINAMATH_CALUDE_retailer_profit_percentage_l540_54063

/-- Calculates the profit percentage for a retailer selling a machine --/
theorem retailer_profit_percentage
  (wholesale_price : ℝ)
  (retail_price : ℝ)
  (discount_rate : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : retail_price = 120)
  (h3 : discount_rate = 0.1)
  : (((retail_price * (1 - discount_rate) - wholesale_price) / wholesale_price) * 100 = 20) := by
  sorry

#check retailer_profit_percentage

end NUMINAMATH_CALUDE_retailer_profit_percentage_l540_54063


namespace NUMINAMATH_CALUDE_odd_even_sum_difference_l540_54072

/-- The sum of the first n odd natural numbers -/
def sum_odd (n : ℕ) : ℕ := n^2

/-- The sum of the first n even natural numbers -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- The number of odd terms from 1 to 2023 -/
def n_odd : ℕ := (2023 - 1) / 2 + 1

/-- The number of even terms from 2 to 2022 -/
def n_even : ℕ := (2022 - 2) / 2 + 1

theorem odd_even_sum_difference : 
  sum_odd n_odd - sum_even n_even = 22 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_sum_difference_l540_54072


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l540_54023

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_prime_divisor_of_factorial_sum : 
  (Nat.factors (factorial 13 + factorial 14)).maximum? = some 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l540_54023


namespace NUMINAMATH_CALUDE_problem_statement_l540_54015

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5 * a * b) :
  (∃ (x : ℝ), x ≥ a + b ∧ x ≥ 4/5) ∧
  (∀ (x : ℝ), x * a * b ≤ b^2 + 5*a → x ≤ 9) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l540_54015


namespace NUMINAMATH_CALUDE_x_range_when_f_lg_x_gt_f_1_l540_54033

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x < y → f y < f x

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem x_range_when_f_lg_x_gt_f_1 (heven : is_even f) (hdec : is_decreasing_on_nonneg f) :
  (∀ x, f (lg x) > f 1) → (∀ x, x > (1/10) ∧ x < 10) :=
sorry

end NUMINAMATH_CALUDE_x_range_when_f_lg_x_gt_f_1_l540_54033


namespace NUMINAMATH_CALUDE_vector_sum_inequality_l540_54021

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable (a b c d : V)

theorem vector_sum_inequality (h : a + b + c + d = 0) :
  ‖a‖ + ‖b‖ + ‖c‖ + ‖d‖ ≥ ‖a + d‖ + ‖b + d‖ + ‖c + d‖ := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_inequality_l540_54021


namespace NUMINAMATH_CALUDE_positive_A_value_l540_54091

def hash (k : ℝ) (A B : ℝ) : ℝ := A^2 + k * B^2

theorem positive_A_value (k : ℝ) (A : ℝ) :
  k = 3 →
  hash k A 7 = 196 →
  A > 0 →
  A = 7 := by
sorry

end NUMINAMATH_CALUDE_positive_A_value_l540_54091


namespace NUMINAMATH_CALUDE_vector_computation_l540_54074

def c : Fin 3 → ℝ := ![(-3), 5, 2]
def d : Fin 3 → ℝ := ![5, (-1), 3]

theorem vector_computation :
  (2 • c - 5 • d + c) = ![(-34), 20, (-9)] := by sorry

end NUMINAMATH_CALUDE_vector_computation_l540_54074


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l540_54043

theorem negation_of_existential_proposition :
  (¬ ∃ n : ℕ, 2^n < 1000) ↔ (∀ n : ℕ, 2^n ≥ 1000) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l540_54043


namespace NUMINAMATH_CALUDE_picture_frame_length_l540_54014

/-- Given a rectangular picture frame with height 12 inches and perimeter 44 inches, 
    prove that its length is 10 inches. -/
theorem picture_frame_length (height : ℝ) (perimeter : ℝ) (length : ℝ) : 
  height = 12 → perimeter = 44 → perimeter = 2 * (length + height) → length = 10 := by
  sorry

end NUMINAMATH_CALUDE_picture_frame_length_l540_54014


namespace NUMINAMATH_CALUDE_max_y_value_l540_54087

theorem max_y_value (x y : ℝ) (h : x^2 + y^2 = 10*x + 60*y) :
  y ≤ 30 + 5 * Real.sqrt 37 ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 10*x₀ + 60*y₀ ∧ y₀ = 30 + 5 * Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l540_54087


namespace NUMINAMATH_CALUDE_principal_amount_calculation_l540_54030

def simple_interest : ℝ := 4016.25
def interest_rate : ℝ := 0.09
def time_period : ℝ := 5

theorem principal_amount_calculation :
  simple_interest / (interest_rate * time_period) = 8925 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_calculation_l540_54030


namespace NUMINAMATH_CALUDE_exactly_one_of_each_survives_l540_54071

-- Define the number of trees of each type
def num_trees_A : ℕ := 2
def num_trees_B : ℕ := 2

-- Define the survival rates
def survival_rate_A : ℚ := 2/3
def survival_rate_B : ℚ := 1/2

-- Define the probability of exactly one tree of type A surviving
def prob_one_A_survives : ℚ := 
  (num_trees_A.choose 1 : ℚ) * survival_rate_A * (1 - survival_rate_A)

-- Define the probability of exactly one tree of type B surviving
def prob_one_B_survives : ℚ := 
  (num_trees_B.choose 1 : ℚ) * survival_rate_B * (1 - survival_rate_B)

-- State the theorem
theorem exactly_one_of_each_survives : 
  prob_one_A_survives * prob_one_B_survives = 2/9 := by sorry

end NUMINAMATH_CALUDE_exactly_one_of_each_survives_l540_54071


namespace NUMINAMATH_CALUDE_orthic_triangle_right_angled_iff_45_or_135_angle_l540_54039

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the orthic triangle of a given triangle
def orthicTriangle (t : Triangle) : Triangle := sorry

-- Define a right-angled triangle
def isRightAngled (t : Triangle) : Prop := sorry

-- Define the condition for an angle to be 45° or 135°
def has45or135Angle (t : Triangle) : Prop := sorry

-- Theorem statement
theorem orthic_triangle_right_angled_iff_45_or_135_angle (t : Triangle) :
  has45or135Angle t ↔ isRightAngled (orthicTriangle t) := by sorry

end NUMINAMATH_CALUDE_orthic_triangle_right_angled_iff_45_or_135_angle_l540_54039


namespace NUMINAMATH_CALUDE_fencing_cost_l540_54073

/-- Given a rectangular field with sides in ratio 3:4 and area 9408 sq. m,
    prove that the cost of fencing at 25 paise per metre is 98 rupees. -/
theorem fencing_cost (length width : ℝ) (area perimeter cost_per_metre total_cost : ℝ) : 
  length / width = 3 / 4 →
  area = 9408 →
  area = length * width →
  perimeter = 2 * (length + width) →
  cost_per_metre = 25 / 100 →
  total_cost = perimeter * cost_per_metre →
  total_cost = 98 := by
sorry

end NUMINAMATH_CALUDE_fencing_cost_l540_54073


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l540_54036

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x)^(1/3 : ℝ) = 4 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l540_54036


namespace NUMINAMATH_CALUDE_science_score_calculation_l540_54077

def average_score : ℝ := 95
def chinese_score : ℝ := 90
def math_score : ℝ := 98

theorem science_score_calculation :
  ∃ (science_score : ℝ),
    (chinese_score + math_score + science_score) / 3 = average_score ∧
    science_score = 97 := by sorry

end NUMINAMATH_CALUDE_science_score_calculation_l540_54077


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l540_54097

/-- Proves that for a rectangular field with sides in the ratio 3:4, area of 7500 sq. m,
    and a total fencing cost of 87.5, the cost per metre of fencing is 0.25. -/
theorem fencing_cost_per_meter (length width : ℝ) (h1 : width / length = 4 / 3)
    (h2 : length * width = 7500) (h3 : 87.5 = 2 * (length + width) * cost_per_meter) :
  cost_per_meter = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l540_54097


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l540_54045

/-- Given an arithmetic sequence {aₙ} where a₁ = 1 and the common difference d = 2,
    prove that a₈ = 15. -/
theorem arithmetic_sequence_eighth_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = 2) →  -- Common difference is 2
    a 1 = 1 →                    -- First term is 1
    a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l540_54045


namespace NUMINAMATH_CALUDE_parabola_vertex_l540_54054

/-- A parabola is defined by the equation y = 3(x-7)^2 + 5. -/
def parabola (x y : ℝ) : Prop := y = 3 * (x - 7)^2 + 5

/-- The vertex of a parabola is the point where it reaches its minimum or maximum. -/
def is_vertex (x y : ℝ) : Prop := parabola x y ∧ ∀ x' y', parabola x' y' → y ≤ y'

/-- The vertex of the parabola y = 3(x-7)^2 + 5 has coordinates (7, 5). -/
theorem parabola_vertex : is_vertex 7 5 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l540_54054


namespace NUMINAMATH_CALUDE_four_digit_integer_problem_l540_54049

theorem four_digit_integer_problem (n : ℕ) (a b c d : ℕ) :
  n = a * 1000 + b * 100 + c * 10 + d →
  a ≥ 1 →
  a ≤ 9 →
  b ≤ 9 →
  c ≤ 9 →
  d ≤ 9 →
  a + b + c + d = 16 →
  b + c = 10 →
  a - d = 2 →
  n % 11 = 0 →
  n = 4462 := by
sorry

end NUMINAMATH_CALUDE_four_digit_integer_problem_l540_54049


namespace NUMINAMATH_CALUDE_sharp_composition_10_l540_54031

def sharp (N : ℕ) : ℕ := N^2 - N + 2

theorem sharp_composition_10 : sharp (sharp (sharp 10)) = 70123304 := by
  sorry

end NUMINAMATH_CALUDE_sharp_composition_10_l540_54031


namespace NUMINAMATH_CALUDE_total_people_in_tribes_l540_54011

/-- Proves that the total number of people in two tribes is 378, given specific conditions about the number of women, men, and cannoneers. -/
theorem total_people_in_tribes (cannoneers : ℕ) (women : ℕ) (men : ℕ) : 
  cannoneers = 63 →
  women = 2 * cannoneers →
  men = 2 * women →
  cannoneers + women + men = 378 :=
by sorry

end NUMINAMATH_CALUDE_total_people_in_tribes_l540_54011


namespace NUMINAMATH_CALUDE_max_x_minus_y_l540_54070

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 6*y + 12 = 0) :
  ∃ (max : ℝ), max = 1 + Real.sqrt 2 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 6*y' + 12 = 0 → x' - y' ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l540_54070
