import Mathlib

namespace NUMINAMATH_CALUDE_patanjali_speed_l600_60024

/-- Represents Patanjali's walking data over three days -/
structure WalkingData where
  speed_day1 : ℝ
  hours_day1 : ℝ
  total_distance : ℝ

/-- Conditions for Patanjali's walking problem -/
def walking_conditions (data : WalkingData) : Prop :=
  data.speed_day1 * data.hours_day1 = 18 ∧
  (data.speed_day1 + 1) * (data.hours_day1 - 1) + (data.speed_day1 + 1) * data.hours_day1 = data.total_distance - 18 ∧
  data.total_distance = 62

/-- Theorem stating that Patanjali's speed on the first day was 9 miles per hour -/
theorem patanjali_speed (data : WalkingData) 
  (h : walking_conditions data) : data.speed_day1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_patanjali_speed_l600_60024


namespace NUMINAMATH_CALUDE_area_of_four_isosceles_triangles_l600_60089

/-- The area of a figure composed of four isosceles triangles -/
theorem area_of_four_isosceles_triangles :
  ∀ (s : ℝ) (θ : ℝ),
  s = 1 →
  θ = 75 * π / 180 →
  2 * s^2 * Real.sin θ = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_four_isosceles_triangles_l600_60089


namespace NUMINAMATH_CALUDE_larger_number_of_pair_l600_60086

theorem larger_number_of_pair (a b : ℝ) (h1 : a - b = 5) (h2 : a + b = 37) :
  max a b = 21 := by sorry

end NUMINAMATH_CALUDE_larger_number_of_pair_l600_60086


namespace NUMINAMATH_CALUDE_modulus_of_z_l600_60023

theorem modulus_of_z (z : ℂ) (h : z * (1 - Complex.I) = -1 - Complex.I) : 
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l600_60023


namespace NUMINAMATH_CALUDE_fair_coin_tosses_l600_60003

theorem fair_coin_tosses (n : ℕ) : 
  (1 / 2 : ℝ) ^ n = (1 / 16 : ℝ) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_fair_coin_tosses_l600_60003


namespace NUMINAMATH_CALUDE_smallest_integer_sqrt_difference_l600_60045

theorem smallest_integer_sqrt_difference (n : ℕ) : 
  (∀ m : ℕ, m > 0 → m < 250001 → Real.sqrt m - Real.sqrt (m - 1) ≥ (1 : ℝ) / 1000) ∧ 
  (Real.sqrt 250001 - Real.sqrt 250000 < (1 : ℝ) / 1000) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_sqrt_difference_l600_60045


namespace NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l600_60006

-- Define the complex numbers p, q, r
variable (p q r : ℂ)

-- Define the conditions
def is_equilateral_triangle (p q r : ℂ) : Prop :=
  Complex.abs (q - p) = 24 ∧ Complex.abs (r - q) = 24 ∧ Complex.abs (p - r) = 24

-- State the theorem
theorem equilateral_triangle_sum_product (h1 : is_equilateral_triangle p q r) 
  (h2 : Complex.abs (p + q + r) = 48) : 
  Complex.abs (p*q + p*r + q*r) = 768 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_sum_product_l600_60006


namespace NUMINAMATH_CALUDE_unique_solution_equation_l600_60002

theorem unique_solution_equation (x : ℝ) :
  x ≥ 0 ∧ (2021 * (x^2020)^(1/202) - 1 = 2020 * x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l600_60002


namespace NUMINAMATH_CALUDE_larger_number_of_pair_l600_60079

theorem larger_number_of_pair (x y : ℝ) : 
  x - y = 7 → x + y = 47 → max x y = 27 := by sorry

end NUMINAMATH_CALUDE_larger_number_of_pair_l600_60079


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l600_60033

def U : Set ℕ := {2011, 2012, 2013, 2014, 2015}
def M : Set ℕ := {2011, 2012, 2013}

theorem complement_of_M_in_U : U \ M = {2014, 2015} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l600_60033


namespace NUMINAMATH_CALUDE_four_Z_one_l600_60066

/-- Define the Z operation -/
def Z (a b : ℝ) : ℝ := a^3 - 3*a^2*b + 3*a*b^2 - b^3

/-- Theorem: The value of 4 Z 1 is 27 -/
theorem four_Z_one : Z 4 1 = 27 := by sorry

end NUMINAMATH_CALUDE_four_Z_one_l600_60066


namespace NUMINAMATH_CALUDE_banana_purchase_cost_l600_60074

/-- The cost of bananas in dollars per three pounds -/
def banana_rate : ℚ := 3

/-- The amount of bananas in pounds to be purchased -/
def banana_amount : ℚ := 18

/-- The cost of purchasing the given amount of bananas -/
def banana_cost : ℚ := banana_amount * (banana_rate / 3)

theorem banana_purchase_cost : banana_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_banana_purchase_cost_l600_60074


namespace NUMINAMATH_CALUDE_remainder_of_n_l600_60097

theorem remainder_of_n (n : ℕ) 
  (h1 : n^2 % 7 = 3) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_n_l600_60097


namespace NUMINAMATH_CALUDE_problem_solution_l600_60069

def quadratic_equation (x : ℝ) : Prop := x^2 - 5*x + 1 = 0

def inequality_system (x : ℝ) : Prop :=
  x + 8 < 4*x - 1 ∧ (1/2)*x ≤ 8 - (3/2)*x

theorem problem_solution :
  (∃ x₁ x₂ : ℝ, x₁ = (5 + Real.sqrt 21) / 2 ∧
                x₂ = (5 - Real.sqrt 21) / 2 ∧
                quadratic_equation x₁ ∧
                quadratic_equation x₂) ∧
  (∀ x : ℝ, inequality_system x ↔ 3 < x ∧ x ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l600_60069


namespace NUMINAMATH_CALUDE_probability_of_dime_l600_60094

/-- Represents the types of coins in the jar -/
inductive Coin
  | Dime
  | Nickel
  | Penny

/-- The value of each coin type in cents -/
def coinValue : Coin → ℚ
  | Coin.Dime => 10
  | Coin.Nickel => 5
  | Coin.Penny => 1

/-- The total value of each coin type in cents -/
def totalValue : Coin → ℚ
  | Coin.Dime => 800
  | Coin.Nickel => 700
  | Coin.Penny => 500

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℚ :=
  totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℚ :=
  coinCount Coin.Dime + coinCount Coin.Nickel + coinCount Coin.Penny

/-- The probability of randomly selecting a dime from the jar -/
theorem probability_of_dime : 
  coinCount Coin.Dime / totalCoins = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_probability_of_dime_l600_60094


namespace NUMINAMATH_CALUDE_min_segments_for_perimeter_is_three_l600_60092

/-- Represents an octagon formed by cutting a smaller rectangle from a larger rectangle -/
structure CutOutOctagon where
  /-- The length of the larger rectangle -/
  outer_length : ℝ
  /-- The width of the larger rectangle -/
  outer_width : ℝ
  /-- The length of the smaller cut-out rectangle -/
  inner_length : ℝ
  /-- The width of the smaller cut-out rectangle -/
  inner_width : ℝ
  /-- Ensures the inner rectangle fits inside the outer rectangle -/
  h_inner_fits : inner_length < outer_length ∧ inner_width < outer_width

/-- The minimum number of line segment lengths required to calculate the perimeter of a CutOutOctagon -/
def min_segments_for_perimeter (oct : CutOutOctagon) : ℕ := 3

/-- Theorem stating that the minimum number of line segment lengths required to calculate
    the perimeter of a CutOutOctagon is always 3 -/
theorem min_segments_for_perimeter_is_three (oct : CutOutOctagon) :
  min_segments_for_perimeter oct = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_segments_for_perimeter_is_three_l600_60092


namespace NUMINAMATH_CALUDE_circle_diameter_l600_60078

/-- The diameter of a circle is twice its radius -/
theorem circle_diameter (r : ℝ) (d : ℝ) (h : r = 7) : d = 14 ↔ d = 2 * r := by sorry

end NUMINAMATH_CALUDE_circle_diameter_l600_60078


namespace NUMINAMATH_CALUDE_wine_card_probability_l600_60019

theorem wine_card_probability : 
  let n_card_types : ℕ := 3
  let n_bottles : ℕ := 5
  let total_outcomes : ℕ := n_card_types^n_bottles
  let two_type_outcomes : ℕ := Nat.choose n_card_types 2 * 2^n_bottles
  let one_type_outcomes : ℕ := n_card_types
  let favorable_outcomes : ℕ := total_outcomes - (two_type_outcomes - one_type_outcomes)
  (favorable_outcomes : ℚ) / total_outcomes = 50 / 81 :=
by sorry

end NUMINAMATH_CALUDE_wine_card_probability_l600_60019


namespace NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_l600_60010

-- First expression
theorem simplify_first_expression (a b : ℝ) :
  2*a + 3*b + 6*a + 9*b - 8*a - 5 = 12*b - 5 := by sorry

-- Second expression
theorem simplify_second_expression (x : ℝ) :
  2*(3*x + 1) - (4 - x - x^2) = x^2 + 7*x - 2 := by sorry

end NUMINAMATH_CALUDE_simplify_first_expression_simplify_second_expression_l600_60010


namespace NUMINAMATH_CALUDE_cory_fruit_sequences_l600_60084

/-- The number of distinct sequences for eating fruits -/
def fruitSequences (apples oranges bananas pears : ℕ) : ℕ :=
  let total := apples + oranges + bananas + pears
  Nat.factorial total / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas * Nat.factorial pears)

/-- Theorem: The number of distinct sequences for eating 4 apples, 2 oranges, 1 banana, and 2 pears over 8 days is 420 -/
theorem cory_fruit_sequences :
  fruitSequences 4 2 1 2 = 420 := by
  sorry

#eval fruitSequences 4 2 1 2

end NUMINAMATH_CALUDE_cory_fruit_sequences_l600_60084


namespace NUMINAMATH_CALUDE_ellipse_tangent_line_l600_60022

/-- Given an ellipse x^2/a^2 + y^2/b^2 = 1, the tangent line at point P(x₀, y₀) 
    has the equation x₀x/a^2 + y₀y/b^2 = 1 -/
theorem ellipse_tangent_line (a b x₀ y₀ : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
    (h : x₀^2 / a^2 + y₀^2 / b^2 = 1) :
  ∀ x y, (x₀ * x) / a^2 + (y₀ * y) / b^2 = 1 ↔ 
    (∃ t : ℝ, x = x₀ + t * (-2 * x₀ / a^2) ∧ y = y₀ + t * (-2 * y₀ / b^2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_tangent_line_l600_60022


namespace NUMINAMATH_CALUDE_cube_root_unity_sum_l600_60063

def N (p q r : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![p, q, r],
    ![q, r, p],
    ![r, p, q]]

theorem cube_root_unity_sum (p q r : ℂ) :
  N p q r ^ 3 = 1 →
  p * q * r = -1 →
  p^3 + q^3 + r^3 = -2 ∨ p^3 + q^3 + r^3 = -4 := by
sorry

end NUMINAMATH_CALUDE_cube_root_unity_sum_l600_60063


namespace NUMINAMATH_CALUDE_chess_sets_problem_l600_60029

theorem chess_sets_problem (x : ℕ) (y : ℕ) : 
  (x > 0) →
  (y > 0) →
  (16 * x = y * ((16 * x) / y)) →
  ((16 * x) / y + 2) * (y - 10) = 16 * x →
  ((16 * x) / y + 4) * (y - 16) = 16 * x →
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_chess_sets_problem_l600_60029


namespace NUMINAMATH_CALUDE_problem_statement_l600_60018

theorem problem_statement : 2 * Real.sin (π / 3) + (-1/2)⁻¹ + |2 - Real.sqrt 3| = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l600_60018


namespace NUMINAMATH_CALUDE_girls_minus_boys_l600_60000

/-- The number of boys in Grade 7 Class 1 -/
def num_boys (a b : ℤ) : ℤ := 2*a - b

/-- The number of girls in Grade 7 Class 1 -/
def num_girls (a b : ℤ) : ℤ := 3*a + b

/-- The theorem stating the difference between the number of girls and boys -/
theorem girls_minus_boys (a b : ℤ) : 
  num_girls a b - num_boys a b = a + 2*b := by
  sorry

end NUMINAMATH_CALUDE_girls_minus_boys_l600_60000


namespace NUMINAMATH_CALUDE_solution_part1_solution_part2_l600_60085

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |2*x - 1|

-- Theorem for part (1)
theorem solution_part1 : {x : ℝ | f x > 3 - 4*x} = {x : ℝ | x > 3/5} := by sorry

-- Theorem for part (2)
theorem solution_part2 : 
  (∀ x : ℝ, f x + |1 - x| ≥ 6*m^2 - 5*m) → 
  -1/6 ≤ m ∧ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_part1_solution_part2_l600_60085


namespace NUMINAMATH_CALUDE_sum_of_fractions_l600_60096

theorem sum_of_fractions (a : ℝ) (h : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l600_60096


namespace NUMINAMATH_CALUDE_infinitely_many_terms_greater_than_position_l600_60030

/-- A sequence of natural numbers excluding 1 -/
def NatSequenceExcluding1 := ℕ → {n : ℕ // n ≠ 1}

/-- The proposition that for any sequence of natural numbers excluding 1,
    there are infinitely many terms greater than their positions -/
theorem infinitely_many_terms_greater_than_position
  (seq : NatSequenceExcluding1) :
  ∀ N : ℕ, ∃ n > N, (seq n).val > n := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_terms_greater_than_position_l600_60030


namespace NUMINAMATH_CALUDE_log_28_5_l600_60099

theorem log_28_5 (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 7 = b) :
  (Real.log 5) / (Real.log 28) = (1 - a) / (2 * a + b) := by
  sorry

end NUMINAMATH_CALUDE_log_28_5_l600_60099


namespace NUMINAMATH_CALUDE_baker_cakes_sold_l600_60037

/-- Proves the number of cakes sold by a baker given certain conditions -/
theorem baker_cakes_sold (cake_price : ℕ) (pie_price : ℕ) (pies_sold : ℕ) (total_revenue : ℕ) :
  cake_price = 12 →
  pie_price = 7 →
  pies_sold = 126 →
  total_revenue = 6318 →
  ∃ cakes_sold : ℕ, cakes_sold * cake_price + pies_sold * pie_price = total_revenue ∧ cakes_sold = 453 :=
by sorry

end NUMINAMATH_CALUDE_baker_cakes_sold_l600_60037


namespace NUMINAMATH_CALUDE_cars_in_north_america_l600_60020

def total_cars : ℕ := 6755
def cars_in_europe : ℕ := 2871

theorem cars_in_north_america : total_cars - cars_in_europe = 3884 := by
  sorry

end NUMINAMATH_CALUDE_cars_in_north_america_l600_60020


namespace NUMINAMATH_CALUDE_circle_inscribed_line_intersection_l600_60048

-- Define the angle
variable (angle : Angle)

-- Define the circles
variable (ω Ω : Circle)

-- Define the line
variable (l : Line)

-- Define the points
variable (A B C D E F : Point)

-- Define the inscribed property
def inscribed (c : Circle) (α : Angle) : Prop := sorry

-- Define the intersection property
def intersects (l : Line) (c : Circle) (p q : Point) : Prop := sorry

-- Define the order of points on a line
def ordered_on_line (l : Line) (p₁ p₂ p₃ p₄ p₅ p₆ : Point) : Prop := sorry

-- Define the equality of line segments
def segment_eq (p₁ p₂ q₁ q₂ : Point) : Prop := sorry

theorem circle_inscribed_line_intersection 
  (h₁ : inscribed ω angle)
  (h₂ : inscribed Ω angle)
  (h₃ : intersects l angle A F)
  (h₄ : intersects l ω B C)
  (h₅ : intersects l Ω D E)
  (h₆ : ordered_on_line l A B C D E F)
  (h₇ : segment_eq B C D E) :
  segment_eq A B E F := by sorry

end NUMINAMATH_CALUDE_circle_inscribed_line_intersection_l600_60048


namespace NUMINAMATH_CALUDE_meaningful_fraction_l600_60038

theorem meaningful_fraction (x : ℝ) : (x - 5)⁻¹ ≠ 0 ↔ x ≠ 5 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l600_60038


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l600_60016

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem smallest_n_satisfying_conditions :
  ∃! n : ℕ, n ≥ 10 ∧ 
            is_prime (n + 6) ∧ 
            is_perfect_square (9*n + 7) ∧
            ∀ m : ℕ, m ≥ 10 → 
                     is_prime (m + 6) → 
                     is_perfect_square (9*m + 7) → 
                     n ≤ m ∧
            n = 53 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l600_60016


namespace NUMINAMATH_CALUDE_pie_selection_theorem_l600_60014

/-- Represents the types of pie packets -/
inductive PiePacket
  | CabbageCabbage
  | CherryCherry
  | CabbageCherry

/-- Represents the possible fillings of a pie -/
inductive PieFilling
  | Cabbage
  | Cherry

/-- Represents the state of a pie -/
inductive PieState
  | Whole
  | Broken

/-- Represents a strategy for selecting a pie -/
def Strategy := PiePacket → PieFilling → PieState

/-- The probability of giving a whole cherry pie given a strategy -/
def probability_whole_cherry (s : Strategy) : ℚ := sorry

/-- The simple strategy described in part (a) -/
def simple_strategy : Strategy := sorry

/-- The improved strategy described in part (b) -/
def improved_strategy : Strategy := sorry

theorem pie_selection_theorem :
  (probability_whole_cherry simple_strategy = 2/3) ∧
  (probability_whole_cherry improved_strategy > 2/3) := by
  sorry


end NUMINAMATH_CALUDE_pie_selection_theorem_l600_60014


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_largest_n_not_exceeding_500_l600_60017

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 31 ↔ n * (n + 1) ≤ 1000 := by sorry

theorem largest_n_not_exceeding_500 : ∃ n : ℕ, n * (n + 1) ≤ 1000 ∧ ∀ m : ℕ, m > n → m * (m + 1) > 1000 := by sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_largest_n_not_exceeding_500_l600_60017


namespace NUMINAMATH_CALUDE_josh_gummy_bears_l600_60090

theorem josh_gummy_bears (initial_candies : ℕ) : 
  (∃ (remaining_after_siblings : ℕ) (remaining_after_friend : ℕ),
    initial_candies = 3 * 10 + remaining_after_siblings ∧
    remaining_after_siblings = 2 * remaining_after_friend ∧
    remaining_after_friend = 16 + 19) →
  initial_candies = 100 := by
sorry

end NUMINAMATH_CALUDE_josh_gummy_bears_l600_60090


namespace NUMINAMATH_CALUDE_shares_owned_problem_solution_l600_60015

/-- A function that calculates the dividend per share based on actual earnings --/
def dividend_per_share (expected_earnings : ℚ) (actual_earnings : ℚ) : ℚ :=
  let base_dividend := expected_earnings / 2
  let additional_earnings := max (actual_earnings - expected_earnings) 0
  let additional_dividend := (additional_earnings / (1/10)) * (4/100)
  base_dividend + additional_dividend

theorem shares_owned (expected_earnings actual_earnings total_dividend : ℚ) : ℚ :=
  total_dividend / (dividend_per_share expected_earnings actual_earnings)

/-- Proves the number of shares owned given the problem conditions --/
theorem problem_solution :
  let expected_earnings : ℚ := 80/100
  let actual_earnings : ℚ := 110/100
  let total_dividend : ℚ := 260
  shares_owned expected_earnings actual_earnings total_dividend = 500 := by
  sorry

end NUMINAMATH_CALUDE_shares_owned_problem_solution_l600_60015


namespace NUMINAMATH_CALUDE_remainder_2345678_div_5_l600_60068

theorem remainder_2345678_div_5 : 2345678 % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2345678_div_5_l600_60068


namespace NUMINAMATH_CALUDE_track_length_track_length_is_200_l600_60012

/-- The length of a circular track given specific meeting conditions of two runners -/
theorem track_length : ℝ → Prop :=
  fun track_length =>
    ∀ (brenda_speed sally_speed : ℝ),
      brenda_speed > 0 ∧ sally_speed > 0 →
      ∃ (first_meeting_time second_meeting_time : ℝ),
        first_meeting_time > 0 ∧ second_meeting_time > first_meeting_time ∧
        brenda_speed * first_meeting_time = 120 ∧
        brenda_speed * (second_meeting_time - first_meeting_time) = 200 ∧
        (brenda_speed * first_meeting_time + sally_speed * first_meeting_time = track_length / 2) ∧
        (brenda_speed * second_meeting_time + sally_speed * second_meeting_time = 
          track_length + track_length / 2) →
        track_length = 200

/-- The track length is 200 meters -/
theorem track_length_is_200 : track_length 200 := by
  sorry

end NUMINAMATH_CALUDE_track_length_track_length_is_200_l600_60012


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l600_60027

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 = ((37 * x) / (73 * y)) * ((1/5)^2 + (1/6)^2)) :
  Real.sqrt x / Real.sqrt y = 1281 / 94 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l600_60027


namespace NUMINAMATH_CALUDE_integer_fraction_problem_l600_60093

theorem integer_fraction_problem (a b : ℕ+) :
  (a.val > 0) →
  (b.val > 0) →
  (∃ k : ℤ, (a.val^3 * b.val - 1) = k * (a.val + 1)) →
  (∃ m : ℤ, (b.val^3 * a.val + 1) = m * (b.val - 1)) →
  ((a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3)) := by
sorry

end NUMINAMATH_CALUDE_integer_fraction_problem_l600_60093


namespace NUMINAMATH_CALUDE_problem_solution_l600_60041

def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x < 4 ↔ x ∈ Set.Ioo (-2) 2) ∧
  (∃ x : ℝ, f x - |a - 1| < 0) ↔ a ∈ Set.Iio (-1) ∪ Set.Ioi 3 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l600_60041


namespace NUMINAMATH_CALUDE_male_associate_or_full_tenured_percentage_l600_60047

structure University where
  total_professors : ℕ
  women_professors : ℕ
  tenured_professors : ℕ
  associate_or_full_professors : ℕ
  women_or_tenured_professors : ℕ
  male_associate_or_full_professors : ℕ

def University.valid (u : University) : Prop :=
  u.women_professors = (70 * u.total_professors) / 100 ∧
  u.tenured_professors = (70 * u.total_professors) / 100 ∧
  u.associate_or_full_professors = (50 * u.total_professors) / 100 ∧
  u.women_or_tenured_professors = (90 * u.total_professors) / 100 ∧
  u.male_associate_or_full_professors = (80 * u.associate_or_full_professors) / 100

theorem male_associate_or_full_tenured_percentage (u : University) (h : u.valid) :
  (u.tenured_professors - u.women_professors + (u.women_or_tenured_professors - u.total_professors)) * 100 / u.male_associate_or_full_professors = 50 := by
  sorry

end NUMINAMATH_CALUDE_male_associate_or_full_tenured_percentage_l600_60047


namespace NUMINAMATH_CALUDE_sum_sqrt_inequality_l600_60007

theorem sum_sqrt_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  Real.sqrt (x / (1 - x)) + Real.sqrt (y / (1 - y)) + Real.sqrt (z / (1 - z)) > 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_sqrt_inequality_l600_60007


namespace NUMINAMATH_CALUDE_village_population_l600_60013

def initial_population : ℝ → ℝ → ℝ → Prop :=
  fun P rate years =>
    P * (1 - rate)^years = 4860

theorem village_population :
  ∃ P : ℝ, initial_population P 0.1 2 ∧ P = 6000 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l600_60013


namespace NUMINAMATH_CALUDE_teresas_current_age_l600_60043

/-- Given the ages of family members at different points in time, 
    prove Teresa's current age. -/
theorem teresas_current_age 
  (morio_current_age : ℕ)
  (morio_age_at_birth : ℕ)
  (teresa_age_at_birth : ℕ)
  (h1 : morio_current_age = 71)
  (h2 : morio_age_at_birth = 38)
  (h3 : teresa_age_at_birth = 26) :
  teresa_age_at_birth + (morio_current_age - morio_age_at_birth) = 59 :=
by sorry

end NUMINAMATH_CALUDE_teresas_current_age_l600_60043


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l600_60036

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 104) 
  (h2 : x * y = 35) : 
  (x + y ≤ Real.sqrt 174) ∧ (∃ (a b : ℝ), a^2 + b^2 = 104 ∧ a * b = 35 ∧ a + b = Real.sqrt 174) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l600_60036


namespace NUMINAMATH_CALUDE_notebook_difference_proof_l600_60061

/-- The price of notebooks Jeremy bought -/
def jeremy_total : ℚ := 180 / 100

/-- The price of notebooks Tina bought -/
def tina_total : ℚ := 300 / 100

/-- The difference in the number of notebooks bought by Tina and Jeremy -/
def notebook_difference : ℕ := 4

/-- The price of a single notebook -/
def notebook_price : ℚ := 30 / 100

theorem notebook_difference_proof :
  ∃ (jeremy_count tina_count : ℕ),
    jeremy_count * notebook_price = jeremy_total ∧
    tina_count * notebook_price = tina_total ∧
    tina_count - jeremy_count = notebook_difference :=
by sorry

end NUMINAMATH_CALUDE_notebook_difference_proof_l600_60061


namespace NUMINAMATH_CALUDE_prime_squared_minus_one_divisible_by_thirty_l600_60077

theorem prime_squared_minus_one_divisible_by_thirty
  (p : ℕ) (hp : Nat.Prime p) (hp_ge_seven : p ≥ 7) :
  30 ∣ p^2 - 1 := by
sorry

end NUMINAMATH_CALUDE_prime_squared_minus_one_divisible_by_thirty_l600_60077


namespace NUMINAMATH_CALUDE_nested_expression_value_l600_60065

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l600_60065


namespace NUMINAMATH_CALUDE_min_distance_to_line_min_distance_achievable_l600_60035

/-- The minimum distance from the origin to a point on the line x + y - 4 = 0 is 2√2 -/
theorem min_distance_to_line : ∀ m n : ℝ, m + n = 4 → Real.sqrt (m^2 + n^2) ≥ 2 * Real.sqrt 2 := by
  sorry

/-- The minimum distance 2√2 is achievable -/
theorem min_distance_achievable : ∃ m n : ℝ, m + n = 4 ∧ Real.sqrt (m^2 + n^2) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_line_min_distance_achievable_l600_60035


namespace NUMINAMATH_CALUDE_complex_point_equivalence_l600_60049

theorem complex_point_equivalence : 
  let z : ℂ := (Complex.I) / (1 + 3 * Complex.I)
  z = (3 : ℝ) / 10 + ((1 : ℝ) / 10) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_point_equivalence_l600_60049


namespace NUMINAMATH_CALUDE_integers_between_cubes_l600_60055

theorem integers_between_cubes : ∃ n : ℕ, n = (⌊(9.5 : ℝ)^3⌋ - ⌈(9.4 : ℝ)^3⌉ + 1) ∧ n = 27 := by
  sorry

end NUMINAMATH_CALUDE_integers_between_cubes_l600_60055


namespace NUMINAMATH_CALUDE_factorization_equality_l600_60082

theorem factorization_equality (a : ℝ) : 
  (2 / 9) * a^2 - (4 / 3) * a + 2 = (2 / 9) * (a - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l600_60082


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l600_60053

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (2, x + 2)
  parallel a b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l600_60053


namespace NUMINAMATH_CALUDE_notebook_price_l600_60091

theorem notebook_price (notebook_count : ℕ) (pencil_price pen_price total_spent : ℚ) : 
  notebook_count = 3 →
  pencil_price = 1.5 →
  pen_price = 1.7 →
  total_spent = 6.8 →
  ∃ (notebook_price : ℚ), 
    notebook_count * notebook_price + pencil_price + pen_price = total_spent ∧
    notebook_price = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_notebook_price_l600_60091


namespace NUMINAMATH_CALUDE_antenna_spire_height_l600_60034

/-- The height of the Empire State Building's antenna spire -/
theorem antenna_spire_height :
  let total_height : ℕ := 1454
  let top_floor_height : ℕ := 1250
  let antenna_height := total_height - top_floor_height
  antenna_height = 204 :=
by sorry

end NUMINAMATH_CALUDE_antenna_spire_height_l600_60034


namespace NUMINAMATH_CALUDE_initial_average_is_16_l600_60040

def initial_average_problem (A : ℝ) : Prop :=
  -- Define the sum of 6 initial observations
  let initial_sum := 6 * A
  -- Define the sum of 7 observations after adding the new one
  let new_sum := initial_sum + 9
  -- The new average is A - 1
  new_sum / 7 = A - 1

theorem initial_average_is_16 :
  ∃ A : ℝ, initial_average_problem A ∧ A = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_is_16_l600_60040


namespace NUMINAMATH_CALUDE_increasing_function_bounds_l600_60081

theorem increasing_function_bounds (k : ℕ) (f : ℕ → ℕ) 
  (h_increasing : ∀ m n, m < n → f m < f n) 
  (h_functional : ∀ n, f (f n) = k * n) :
  ∀ n, (2 * k * n) / (k + 1) ≤ f n ∧ f n ≤ ((k + 1) * n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_bounds_l600_60081


namespace NUMINAMATH_CALUDE_correct_calculation_l600_60004

theorem correct_calculation (x : ℚ) : x * 15 = 45 → x * 5 * 10 = 150 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l600_60004


namespace NUMINAMATH_CALUDE_midsegment_inequality_l600_60056

/-- Midsegment theorem for triangles -/
theorem midsegment_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  let perimeter := a + b + c
  let midsegment_sum := (b + c) / 2 + (a + c) / 2 + (a + b) / 2
  midsegment_sum < perimeter ∧ midsegment_sum > 3 / 4 * perimeter :=
by sorry

end NUMINAMATH_CALUDE_midsegment_inequality_l600_60056


namespace NUMINAMATH_CALUDE_intersection_when_a_is_two_empty_intersection_iff_a_in_range_l600_60054

def A (a : ℝ) : Set ℝ := {x : ℝ | |x - 1| ≤ a ∧ a > 0}

def B : Set ℝ := {x : ℝ | x > 2 ∨ x < -2}

theorem intersection_when_a_is_two :
  A 2 ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

theorem empty_intersection_iff_a_in_range :
  ∀ a : ℝ, A a ∩ B = ∅ ↔ 0 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_two_empty_intersection_iff_a_in_range_l600_60054


namespace NUMINAMATH_CALUDE_difference_of_squares_simplification_l600_60001

theorem difference_of_squares_simplification : (365^2 - 349^2) / 16 = 714 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_simplification_l600_60001


namespace NUMINAMATH_CALUDE_infinite_primes_l600_60076

theorem infinite_primes : ∀ (S : Finset Nat), (∀ p ∈ S, Nat.Prime p) → 
  ∃ q, Nat.Prime q ∧ q ∉ S := by
  sorry

end NUMINAMATH_CALUDE_infinite_primes_l600_60076


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l600_60070

/-- If the quadratic equation 2x^2 - 4x + m = 0 has two equal real roots, then m = 2 -/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - 4 * x + m = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - 4 * y + m = 0 → y = x) → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l600_60070


namespace NUMINAMATH_CALUDE_factorization_2m_squared_minus_8_factorization_perfect_square_trinomial_l600_60073

-- Part 1
theorem factorization_2m_squared_minus_8 (m : ℝ) : 
  2 * m^2 - 8 = 2 * (m + 2) * (m - 2) := by sorry

-- Part 2
theorem factorization_perfect_square_trinomial (x y : ℝ) : 
  (x + y)^2 - 4 * (x + y) + 4 = (x + y - 2)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_2m_squared_minus_8_factorization_perfect_square_trinomial_l600_60073


namespace NUMINAMATH_CALUDE_abc_inequality_l600_60046

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 / (1 + a^2) + b^2 / (1 + b^2) + c^2 / (1 + c^2) = 1) :
  a * b * c ≤ Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_abc_inequality_l600_60046


namespace NUMINAMATH_CALUDE_inequality_solution_set_l600_60060

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x : ℝ, ax^2 - 5*x + b > 0 ↔ x < -1/3 ∨ x > 1/2) →
  (∀ x : ℝ, bx^2 - 5*x + a > 0 ↔ -3 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l600_60060


namespace NUMINAMATH_CALUDE_students_playing_neither_l600_60039

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) 
  (h1 : total = 35)
  (h2 : football = 26)
  (h3 : tennis = 20)
  (h4 : both = 17) :
  total - (football + tennis - both) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_students_playing_neither_l600_60039


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l600_60083

def a (n : ℕ) := 2 * (n + 1) + 3

theorem arithmetic_sequence_proof :
  ∀ n : ℕ, a (n + 1) - a n = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l600_60083


namespace NUMINAMATH_CALUDE_sum_of_coefficients_after_shift_l600_60051

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a quadratic function horizontally -/
def shift_left (f : QuadraticFunction) (units : ℝ) : QuadraticFunction :=
  QuadraticFunction.mk
    f.a
    (f.b + 2 * f.a * units)
    (f.a * units^2 + f.b * units + f.c)

/-- The original quadratic function y = 3x^2 + 2x - 5 -/
def original : QuadraticFunction :=
  QuadraticFunction.mk 3 2 (-5)

/-- The shifted quadratic function -/
def shifted : QuadraticFunction :=
  shift_left original 6

/-- Theorem stating that the sum of coefficients of the shifted function is 156 -/
theorem sum_of_coefficients_after_shift :
  shifted.a + shifted.b + shifted.c = 156 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_after_shift_l600_60051


namespace NUMINAMATH_CALUDE_replaced_student_weight_is_96_l600_60021

/-- The weight of the replaced student given the conditions of the problem -/
def replaced_student_weight (initial_students : ℕ) (new_student_weight : ℝ) (average_decrease : ℝ) : ℝ :=
  let total_weight_decrease := initial_students * average_decrease
  let weight_difference := total_weight_decrease + new_student_weight
  weight_difference

/-- Theorem stating that under the given conditions, the replaced student's weight is 96 kg -/
theorem replaced_student_weight_is_96 :
  replaced_student_weight 4 64 8 = 96 := by
  sorry

end NUMINAMATH_CALUDE_replaced_student_weight_is_96_l600_60021


namespace NUMINAMATH_CALUDE_green_peaches_per_basket_l600_60032

/-- Proves the number of green peaches in each basket -/
theorem green_peaches_per_basket 
  (num_baskets : ℕ) 
  (red_per_basket : ℕ) 
  (total_peaches : ℕ) 
  (h1 : num_baskets = 2)
  (h2 : red_per_basket = 4)
  (h3 : total_peaches = 12) :
  (total_peaches - num_baskets * red_per_basket) / num_baskets = 2 := by
sorry

end NUMINAMATH_CALUDE_green_peaches_per_basket_l600_60032


namespace NUMINAMATH_CALUDE_positive_slope_implies_positive_correlation_l600_60075

/-- A linear regression model relating variables x and y. -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope
  equation : ∀ x y : ℝ, y = a + b * x

/-- Definition of positive linear correlation between two variables. -/
def positively_correlated (x y : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → y x₁ < y x₂

/-- Theorem stating that a linear regression with positive slope implies positive correlation. -/
theorem positive_slope_implies_positive_correlation
  (model : LinearRegression)
  (h_positive_slope : model.b > 0) :
  positively_correlated (λ x => x) (λ x => model.a + model.b * x) :=
by
  sorry


end NUMINAMATH_CALUDE_positive_slope_implies_positive_correlation_l600_60075


namespace NUMINAMATH_CALUDE_adam_caramel_boxes_l600_60026

/-- The number of boxes of caramel candy Adam bought -/
def caramel_boxes (chocolate_boxes : ℕ) (pieces_per_box : ℕ) (total_candies : ℕ) : ℕ :=
  (total_candies - chocolate_boxes * pieces_per_box) / pieces_per_box

/-- Proof that Adam bought 5 boxes of caramel candy -/
theorem adam_caramel_boxes : 
  caramel_boxes 2 4 28 = 5 := by
  sorry

end NUMINAMATH_CALUDE_adam_caramel_boxes_l600_60026


namespace NUMINAMATH_CALUDE_inverse_proportion_change_l600_60005

/-- Given positive numbers x and y that are inversely proportional, prove that when x doubles, y decreases by 50% -/
theorem inverse_proportion_change (x y x' y' k : ℝ) :
  x > 0 →
  y > 0 →
  x * y = k →
  x' = 2 * x →
  x' * y' = k →
  y' / y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_change_l600_60005


namespace NUMINAMATH_CALUDE_gear_alignment_l600_60052

theorem gear_alignment (n : ℕ) (h1 : n = 6) :
  ∃ (rotation : Fin 32), ∀ (i : Fin n),
    (i.val + rotation : Fin 32) ∉ {j : Fin 32 | j.val < n} :=
sorry

end NUMINAMATH_CALUDE_gear_alignment_l600_60052


namespace NUMINAMATH_CALUDE_no_cube_sum_4099_l600_60080

theorem no_cube_sum_4099 : 
  ∀ a b : ℤ, a^3 + b^3 ≠ 4099 :=
by
  sorry

#check no_cube_sum_4099

end NUMINAMATH_CALUDE_no_cube_sum_4099_l600_60080


namespace NUMINAMATH_CALUDE_not_power_of_two_concat_l600_60011

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def concat_numbers (nums : List ℕ) : ℕ := sorry

theorem not_power_of_two_concat :
  ∀ (perm : List ℕ),
    (∀ n ∈ perm, is_five_digit n) →
    (perm.length = 88889) →
    (∀ n, is_five_digit n → n ∈ perm) →
    ¬ ∃ k : ℕ, concat_numbers perm = 2^k :=
by sorry

end NUMINAMATH_CALUDE_not_power_of_two_concat_l600_60011


namespace NUMINAMATH_CALUDE_min_discriminant_quadratic_trinomial_l600_60072

theorem min_discriminant_quadratic_trinomial (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c ≥ 0) →
  (∀ x, abs x < 1 → a * x^2 + b * x + c ≤ 1 / Real.sqrt (1 - x^2)) →
  b^2 - 4*a*c ≥ -4 ∧ ∃ a' b' c', b'^2 - 4*a'*c' = -4 :=
by sorry

end NUMINAMATH_CALUDE_min_discriminant_quadratic_trinomial_l600_60072


namespace NUMINAMATH_CALUDE_triangle_right_angle_l600_60025

theorem triangle_right_angle (A B C : ℝ) (h : A = B - C) : B = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_right_angle_l600_60025


namespace NUMINAMATH_CALUDE_f_zeros_f_min_max_on_interval_l600_60057

def f (x : ℝ) : ℝ := x^2 + x - 2

theorem f_zeros (x : ℝ) : f x = 0 ↔ x = 1 ∨ x = -2 := by sorry

theorem f_min_max_on_interval :
  let a : ℝ := -1
  let b : ℝ := 1
  (∀ x ∈ Set.Icc a b, f x ≥ -9/4) ∧
  (∃ x ∈ Set.Icc a b, f x = -9/4) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 0) ∧
  (∃ x ∈ Set.Icc a b, f x = 0) := by sorry

end NUMINAMATH_CALUDE_f_zeros_f_min_max_on_interval_l600_60057


namespace NUMINAMATH_CALUDE_cubic_inequality_range_l600_60095

theorem cubic_inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, x^3 - a*x + 1 ≥ 0) → 
  0 ≤ a ∧ a ≤ (3 * 2^(1/3)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_inequality_range_l600_60095


namespace NUMINAMATH_CALUDE_catenary_properties_l600_60008

noncomputable def f (a b x : ℝ) : ℝ := a * Real.exp x + b * Real.exp (-x)

theorem catenary_properties (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ x, f a b x = f a b (-x) ↔ a = b) ∧
  (∀ x, f a b x = -f a b (-x) ↔ a = -b) ∧
  (a * b < 0 → ∀ x y, x < y → f a b x < f a b y ∨ ∀ x y, x < y → f a b x > f a b y) ∧
  (a * b > 0 → ∃ x, (∀ y, f a b y ≥ f a b x) ∨ (∀ y, f a b y ≤ f a b x)) :=
sorry

end NUMINAMATH_CALUDE_catenary_properties_l600_60008


namespace NUMINAMATH_CALUDE_annie_initial_money_l600_60064

/-- Annie's hamburger and milkshake purchase problem -/
theorem annie_initial_money :
  let hamburger_price : ℕ := 4
  let milkshake_price : ℕ := 3
  let hamburgers_bought : ℕ := 8
  let milkshakes_bought : ℕ := 6
  let money_left : ℕ := 70
  let initial_money : ℕ := hamburger_price * hamburgers_bought + milkshake_price * milkshakes_bought + money_left
  initial_money = 120 := by sorry

end NUMINAMATH_CALUDE_annie_initial_money_l600_60064


namespace NUMINAMATH_CALUDE_simplify_expression_l600_60028

theorem simplify_expression (a b : ℝ) :
  (35 * a + 70 * b + 15) + (15 * a + 54 * b + 5) - (20 * a + 85 * b + 10) = 30 * a + 39 * b + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l600_60028


namespace NUMINAMATH_CALUDE_total_price_calculation_l600_60067

/-- Calculates the total price of an order of ice-cream bars and sundaes -/
theorem total_price_calculation (ice_cream_bars sundaes : ℕ) (ice_cream_price sundae_price : ℚ) :
  ice_cream_bars = 125 →
  sundaes = 125 →
  ice_cream_price = 0.60 →
  sundae_price = 1.40 →
  ice_cream_bars * ice_cream_price + sundaes * sundae_price = 250 :=
by
  sorry

#check total_price_calculation

end NUMINAMATH_CALUDE_total_price_calculation_l600_60067


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l600_60009

theorem quadratic_equation_solution : 
  let x₁ : ℝ := (2 + Real.sqrt 2) / 2
  let x₂ : ℝ := (2 - Real.sqrt 2) / 2
  2 * x₁^2 = 4 * x₁ - 1 ∧ 2 * x₂^2 = 4 * x₂ - 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l600_60009


namespace NUMINAMATH_CALUDE_no_food_left_for_dog_l600_60044

theorem no_food_left_for_dog (N : ℕ) (prepared_food : ℝ) : 
  let stayed := N / 3
  let excursion := 2 * N / 3
  let lunch_portion := prepared_food / 4
  let excursion_portion := 1.5 * lunch_portion
  stayed * lunch_portion + excursion * excursion_portion = prepared_food :=
by sorry

end NUMINAMATH_CALUDE_no_food_left_for_dog_l600_60044


namespace NUMINAMATH_CALUDE_solve_fraction_equation_l600_60062

theorem solve_fraction_equation (y : ℚ) (h : (1:ℚ)/3 - (1:ℚ)/4 = 1/y) : y = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_fraction_equation_l600_60062


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l600_60098

/-- The imaginary part of 2i / (2 + i^3) is equal to 4/5 -/
theorem imaginary_part_of_complex_fraction :
  Complex.im (2 * Complex.I / (2 + Complex.I ^ 3)) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l600_60098


namespace NUMINAMATH_CALUDE_expected_value_of_marbles_l600_60042

/-- The set of marble numbers -/
def MarbleSet : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The number of marbles to draw -/
def DrawCount : ℕ := 3

/-- The sum of a combination of marbles -/
def CombinationSum (c : Finset ℕ) : ℕ := c.sum id

/-- The expected value of the sum of drawn marbles -/
noncomputable def ExpectedValue : ℚ :=
  (MarbleSet.powerset.filter (λ s => s.card = DrawCount)).sum CombinationSum /
   (MarbleSet.powerset.filter (λ s => s.card = DrawCount)).card

/-- Theorem: The expected value of the sum of three randomly drawn marbles is 10.5 -/
theorem expected_value_of_marbles : ExpectedValue = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_marbles_l600_60042


namespace NUMINAMATH_CALUDE_expressions_equality_l600_60059

variable (a b c : ℝ)

theorem expressions_equality :
  (a - (b + c) = a - b - c) ∧
  (a + (-b - c) = a - b - c) ∧
  (a - (b - c) ≠ a - b - c) ∧
  ((-c) + (a - b) = a - b - c) := by
  sorry

end NUMINAMATH_CALUDE_expressions_equality_l600_60059


namespace NUMINAMATH_CALUDE_trout_weight_l600_60087

theorem trout_weight (num_trout num_catfish num_bluegill : ℕ) 
                     (weight_catfish weight_bluegill total_weight : ℚ) :
  num_trout = 4 →
  num_catfish = 3 →
  num_bluegill = 5 →
  weight_catfish = 3/2 →
  weight_bluegill = 5/2 →
  total_weight = 25 →
  ∃ weight_trout : ℚ,
    weight_trout * num_trout + weight_catfish * num_catfish + weight_bluegill * num_bluegill = total_weight ∧
    weight_trout = 2 :=
by sorry

end NUMINAMATH_CALUDE_trout_weight_l600_60087


namespace NUMINAMATH_CALUDE_tim_took_eleven_rulers_l600_60058

/-- The number of rulers initially in the drawer -/
def initial_rulers : ℕ := 14

/-- The number of rulers left in the drawer after Tim took some out -/
def remaining_rulers : ℕ := 3

/-- The number of rulers Tim took out -/
def rulers_taken : ℕ := initial_rulers - remaining_rulers

theorem tim_took_eleven_rulers : rulers_taken = 11 := by
  sorry

end NUMINAMATH_CALUDE_tim_took_eleven_rulers_l600_60058


namespace NUMINAMATH_CALUDE_exists_double_application_square_l600_60088

theorem exists_double_application_square :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_double_application_square_l600_60088


namespace NUMINAMATH_CALUDE_area_of_square_B_l600_60071

/-- Given a square A with diagonal x and a square B with diagonal 3x, 
    the area of square B is 9x^2/2 -/
theorem area_of_square_B (x : ℝ) :
  let diag_A := x
  let diag_B := 3 * diag_A
  let area_B := (diag_B^2) / 4
  area_B = 9 * x^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_square_B_l600_60071


namespace NUMINAMATH_CALUDE_unique_prime_pair_l600_60050

def isPrime (n : ℕ) : Prop := sorry

def nthPrime (n : ℕ) : ℕ := sorry

theorem unique_prime_pair :
  ∀ a b : ℕ, 
    a > 0 → b > 0 → 
    a - b ≥ 2 → 
    (nthPrime a - nthPrime b) ∣ (2 * (a - b)) → 
    a = 4 ∧ b = 2 := by sorry

end NUMINAMATH_CALUDE_unique_prime_pair_l600_60050


namespace NUMINAMATH_CALUDE_furniture_sale_price_l600_60031

theorem furniture_sale_price (wholesale_price : ℝ) 
  (sticker_price : ℝ) (sale_price : ℝ) :
  sticker_price = wholesale_price * 1.4 →
  sale_price = sticker_price * 0.65 →
  sale_price = wholesale_price * 0.91 := by
sorry

end NUMINAMATH_CALUDE_furniture_sale_price_l600_60031
