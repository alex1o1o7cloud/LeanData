import Mathlib

namespace NUMINAMATH_CALUDE_wilson_sledding_l1040_104013

theorem wilson_sledding (tall_hills small_hills tall_runs small_runs : ℕ) 
  (h1 : tall_hills = 2)
  (h2 : small_hills = 3)
  (h3 : tall_runs = 4)
  (h4 : small_runs = tall_runs / 2)
  : tall_hills * tall_runs + small_hills * small_runs = 14 := by
  sorry

end NUMINAMATH_CALUDE_wilson_sledding_l1040_104013


namespace NUMINAMATH_CALUDE_tangent_and_common_point_l1040_104005

/-- The line l: y = kx - 3k + 2 -/
def line (k : ℝ) (x : ℝ) : ℝ := k * x - 3 * k + 2

/-- The curve C: (x-1)² + (y+1)² = 4 where -1 ≤ x ≤ 1 -/
def curve (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 1)^2 = 4 ∧ -1 ≤ x ∧ x ≤ 1

theorem tangent_and_common_point (k : ℝ) :
  (∃ x, curve x (line k x) ∧
    ∀ x', x' ≠ x → ¬ curve x' (line k x')) ↔
  k = 5/12 ∨ (1/2 < k ∧ k ≤ 5/2) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_common_point_l1040_104005


namespace NUMINAMATH_CALUDE_renata_lottery_winnings_l1040_104080

/-- Represents the financial transactions of Renata --/
structure RenataMoney where
  initial : ℕ
  donation : ℕ
  charityWin : ℕ
  waterCost : ℕ
  lotteryCost : ℕ
  final : ℕ

/-- Calculates the lottery winnings based on Renata's transactions --/
def lotteryWinnings (r : RenataMoney) : ℕ :=
  r.final + r.donation + r.waterCost + r.lotteryCost - r.initial - r.charityWin

/-- Theorem stating that Renata's lottery winnings were $2 --/
theorem renata_lottery_winnings :
  let r : RenataMoney := {
    initial := 10,
    donation := 4,
    charityWin := 90,
    waterCost := 1,
    lotteryCost := 1,
    final := 94
  }
  lotteryWinnings r = 2 := by sorry

end NUMINAMATH_CALUDE_renata_lottery_winnings_l1040_104080


namespace NUMINAMATH_CALUDE_first_term_of_geometric_series_l1040_104056

/-- Given an infinite geometric series with common ratio -1/3 and sum 18,
    the first term of the series is 24. -/
theorem first_term_of_geometric_series :
  ∀ (a : ℝ), 
    (∃ (S : ℝ), S = 18 ∧ S = a / (1 - (-1/3))) →
    a = 24 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_series_l1040_104056


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_6_l1040_104000

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Define the property we want to prove
def is_smallest_divisible_by_5_and_6 (n : ℕ) : Prop :=
  is_perfect_square n ∧ 
  n % 5 = 0 ∧ 
  n % 6 = 0 ∧
  ∀ m : ℕ, m < n → ¬(is_perfect_square m ∧ m % 5 = 0 ∧ m % 6 = 0)

-- State the theorem
theorem smallest_perfect_square_divisible_by_5_and_6 :
  is_smallest_divisible_by_5_and_6 900 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_6_l1040_104000


namespace NUMINAMATH_CALUDE_cubic_roots_eighth_power_sum_l1040_104077

theorem cubic_roots_eighth_power_sum (r s : ℂ) : 
  (r^3 - r^2 * Real.sqrt 5 - r + 1 = 0) → 
  (s^3 - s^2 * Real.sqrt 5 - s + 1 = 0) → 
  r^8 + s^8 = 47 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_eighth_power_sum_l1040_104077


namespace NUMINAMATH_CALUDE_selling_price_calculation_l1040_104055

theorem selling_price_calculation (cost_price : ℝ) (profit_percentage : ℝ) : 
  cost_price = 240 → profit_percentage = 20 → 
  cost_price * (1 + profit_percentage / 100) = 288 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_calculation_l1040_104055


namespace NUMINAMATH_CALUDE_sharon_drive_distance_l1040_104017

theorem sharon_drive_distance :
  let usual_time : ℝ := 180
  let snowstorm_time : ℝ := 300
  let speed_decrease : ℝ := 30
  let distance : ℝ := 157.5
  let usual_speed : ℝ := distance / usual_time
  let snowstorm_speed : ℝ := usual_speed - speed_decrease / 60
  (distance / 2) / usual_speed + (distance / 2) / snowstorm_speed = snowstorm_time :=
by sorry

end NUMINAMATH_CALUDE_sharon_drive_distance_l1040_104017


namespace NUMINAMATH_CALUDE_golden_ratio_less_than_one_l1040_104022

theorem golden_ratio_less_than_one : (Real.sqrt 5 - 1) / 2 < 1 := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_less_than_one_l1040_104022


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l1040_104072

/-- A quadratic function with a vertex at (-1, -3) -/
def f (b c : ℝ) (x : ℝ) : ℝ := -x^2 + b*x + c

/-- The vertex of the quadratic function is at (-1, -3) -/
def has_vertex (b c : ℝ) : Prop :=
  (∀ x, f b c x ≤ f b c (-1)) ∧ (f b c (-1) = -3)

/-- Theorem stating that b = -2 and c = -4 for the given quadratic function -/
theorem quadratic_coefficients :
  ∃ b c : ℝ, has_vertex b c ∧ b = -2 ∧ c = -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l1040_104072


namespace NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l1040_104046

theorem integer_solutions_quadratic_equation :
  ∀ a b : ℤ, 7 * a + 14 * b = 5 * a^2 + 5 * a * b + 5 * b^2 ↔
    (a = -1 ∧ b = 3) ∨ (a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 2) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l1040_104046


namespace NUMINAMATH_CALUDE_power_of_two_equality_l1040_104087

theorem power_of_two_equality : (16^3) * (4^4) * (32^2) = 2^30 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l1040_104087


namespace NUMINAMATH_CALUDE_winner_ate_15_ounces_l1040_104016

-- Define the weights of each ravioli type
def meat_ravioli_weight : ℝ := 1.5
def pumpkin_ravioli_weight : ℝ := 1.25
def cheese_ravioli_weight : ℝ := 1

-- Define the quantities eaten by Javier
def javier_meat_count : ℕ := 5
def javier_pumpkin_count : ℕ := 2
def javier_cheese_count : ℕ := 4

-- Define the quantity eaten by Javier's brother
def brother_pumpkin_count : ℕ := 12

-- Calculate total weight eaten by Javier
def javier_total_weight : ℝ := 
  meat_ravioli_weight * javier_meat_count +
  pumpkin_ravioli_weight * javier_pumpkin_count +
  cheese_ravioli_weight * javier_cheese_count

-- Calculate total weight eaten by Javier's brother
def brother_total_weight : ℝ := pumpkin_ravioli_weight * brother_pumpkin_count

-- Theorem: The winner ate 15 ounces of ravioli
theorem winner_ate_15_ounces : 
  max javier_total_weight brother_total_weight = 15 := by sorry

end NUMINAMATH_CALUDE_winner_ate_15_ounces_l1040_104016


namespace NUMINAMATH_CALUDE_mike_gave_ten_books_l1040_104010

/-- The number of books Mike gave to Lily -/
def books_from_mike : ℕ := sorry

/-- The number of books Corey gave to Lily -/
def books_from_corey : ℕ := sorry

/-- The total number of books Lily received -/
def total_books : ℕ := 35

theorem mike_gave_ten_books :
  (books_from_mike = 10) ∧
  (books_from_corey = books_from_mike + 15) ∧
  (books_from_mike + books_from_corey = total_books) :=
sorry

end NUMINAMATH_CALUDE_mike_gave_ten_books_l1040_104010


namespace NUMINAMATH_CALUDE_roommate_difference_l1040_104054

theorem roommate_difference (bob_roommates john_roommates : ℕ) 
  (h1 : bob_roommates = 10) 
  (h2 : john_roommates = 25) : 
  john_roommates - 2 * bob_roommates = 5 := by
  sorry

end NUMINAMATH_CALUDE_roommate_difference_l1040_104054


namespace NUMINAMATH_CALUDE_exists_interior_rectangle_l1040_104059

/-- A rectangle in a square partition -/
structure Rectangle where
  left : ℝ
  right : ℝ
  bottom : ℝ
  top : ℝ
  left_lt_right : left < right
  bottom_lt_top : bottom < top

/-- A partition of a square into rectangles -/
structure SquarePartition where
  rectangles : List Rectangle
  n_gt_one : rectangles.length > 1
  covers_square : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 →
    ∃ r ∈ rectangles, r.left ≤ x ∧ x ≤ r.right ∧ r.bottom ≤ y ∧ y ≤ r.top
  intersects_line : ∀ l : ℝ, 0 < l ∧ l < 1 →
    (∃ r ∈ rectangles, r.left < l ∧ l < r.right) ∧
    (∃ r ∈ rectangles, r.bottom < l ∧ l < r.top)

/-- A rectangle touches the side of the square if any of its sides coincide with the square's sides -/
def touches_side (r : Rectangle) : Prop :=
  r.left = 0 ∨ r.right = 1 ∨ r.bottom = 0 ∨ r.top = 1

/-- Main theorem: There exists a rectangle that doesn't touch the sides of the square -/
theorem exists_interior_rectangle (p : SquarePartition) :
  ∃ r ∈ p.rectangles, ¬touches_side r := by
  sorry

end NUMINAMATH_CALUDE_exists_interior_rectangle_l1040_104059


namespace NUMINAMATH_CALUDE_y_investment_is_75000_l1040_104028

/-- Represents the investment and profit scenario of a business --/
structure BusinessScenario where
  x_investment : ℕ
  z_investment : ℕ
  z_join_time : ℕ
  z_profit_share : ℕ
  total_profit : ℕ
  total_duration : ℕ

/-- Calculates Y's investment given a business scenario --/
def calculate_y_investment (scenario : BusinessScenario) : ℕ :=
  sorry

/-- Theorem stating that Y's investment is 75000 for the given scenario --/
theorem y_investment_is_75000 (scenario : BusinessScenario) 
  (h1 : scenario.x_investment = 36000)
  (h2 : scenario.z_investment = 48000)
  (h3 : scenario.z_join_time = 4)
  (h4 : scenario.z_profit_share = 4064)
  (h5 : scenario.total_profit = 13970)
  (h6 : scenario.total_duration = 12) :
  calculate_y_investment scenario = 75000 :=
sorry

end NUMINAMATH_CALUDE_y_investment_is_75000_l1040_104028


namespace NUMINAMATH_CALUDE_two_distinct_values_of_T_l1040_104033

theorem two_distinct_values_of_T (n : ℤ) : 
  let i : ℂ := Complex.I
  let T : ℂ := i^(2*n) + i^(-2*n) + Real.cos (n * Real.pi)
  ∃ (a b : ℂ), ∀ (m : ℤ), 
    (let T_m : ℂ := i^(2*m) + i^(-2*m) + Real.cos (m * Real.pi)
     T_m = a ∨ T_m = b) ∧ a ≠ b :=
sorry

end NUMINAMATH_CALUDE_two_distinct_values_of_T_l1040_104033


namespace NUMINAMATH_CALUDE_shirt_sale_problem_l1040_104073

/-- Shirt sale problem -/
theorem shirt_sale_problem 
  (total_shirts : ℕ) 
  (total_cost : ℕ) 
  (black_wholesale : ℕ) 
  (black_retail : ℕ) 
  (white_wholesale : ℕ) 
  (white_retail : ℕ) 
  (h1 : total_shirts = 200)
  (h2 : total_cost = 3500)
  (h3 : black_wholesale = 25)
  (h4 : black_retail = 50)
  (h5 : white_wholesale = 15)
  (h6 : white_retail = 35) :
  ∃ (black_count white_count : ℕ),
    black_count + white_count = total_shirts ∧
    black_count * black_wholesale + white_count * white_wholesale = total_cost ∧
    black_count = 50 ∧
    white_count = 150 ∧
    (black_count * (black_retail - black_wholesale) + 
     white_count * (white_retail - white_wholesale)) = 4250 :=
by sorry

end NUMINAMATH_CALUDE_shirt_sale_problem_l1040_104073


namespace NUMINAMATH_CALUDE_master_craftsman_production_l1040_104095

/-- The number of parts manufactured by a master craftsman during a shift -/
def total_parts : ℕ := 210

/-- The number of parts manufactured in the first hour -/
def first_hour_parts : ℕ := 35

/-- The increase in production rate (parts per hour) -/
def rate_increase : ℕ := 15

/-- The time saved by increasing the production rate (in hours) -/
def time_saved : ℚ := 1.5

theorem master_craftsman_production :
  ∃ (N : ℕ),
    (N : ℚ) / first_hour_parts - (N : ℚ) / (first_hour_parts + rate_increase) = time_saved ∧
    total_parts = first_hour_parts + N :=
  sorry

end NUMINAMATH_CALUDE_master_craftsman_production_l1040_104095


namespace NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l1040_104065

theorem geometric_sequence_eighth_term 
  (a : ℝ) (r : ℝ) 
  (positive_sequence : ∀ n : ℕ, a * r^n > 0)
  (fourth_term : a * r^3 = 12)
  (twelfth_term : a * r^11 = 3) :
  a * r^7 = 6 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_eighth_term_l1040_104065


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1040_104040

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h : ∀ n, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence with a₂ = 3 and a₅ = 6 is 1. -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h₂ : seq.a 2 = 3)
  (h₅ : seq.a 5 = 6) :
  seq.d = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1040_104040


namespace NUMINAMATH_CALUDE_horse_speed_around_square_field_l1040_104096

/-- Given a square field with area 900 km² and a horse that takes 10 hours to run around it,
    prove that the horse's speed is 12 km/h. -/
theorem horse_speed_around_square_field : 
  let field_area : ℝ := 900
  let time_to_run_around : ℝ := 10
  let horse_speed : ℝ := 4 * Real.sqrt field_area / time_to_run_around
  horse_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_horse_speed_around_square_field_l1040_104096


namespace NUMINAMATH_CALUDE_expression_factorization_l1040_104060

theorem expression_factorization (b : ℝ) :
  (4 * b^3 - 84 * b^2 - 12 * b) - (-3 * b^3 - 9 * b^2 + 3 * b) = b * (7 * b + 3) * (b - 5) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1040_104060


namespace NUMINAMATH_CALUDE_right_to_left_grouping_equivalence_l1040_104020

/-- Right-to-left grouping evaluation function -/
noncomputable def rightToLeftEval (a b c d : ℝ) : ℝ := a - b * (c + d)

/-- Standard algebraic notation evaluation function -/
noncomputable def standardEval (a b c d : ℝ) : ℝ := a - b * (c + d)

/-- Theorem stating that right-to-left grouping of a - b × c + d
    is equivalent to a - b(c + d) in standard algebraic notation -/
theorem right_to_left_grouping_equivalence (a b c d : ℝ) :
  rightToLeftEval a b c d = standardEval a b c d := by
  sorry

end NUMINAMATH_CALUDE_right_to_left_grouping_equivalence_l1040_104020


namespace NUMINAMATH_CALUDE_mile_to_yard_l1040_104082

-- Define the units
def mile : ℝ := 1
def furlong : ℝ := 1
def yard : ℝ := 1

-- Define the conversion factors
axiom mile_to_furlong : mile = 8 * furlong
axiom furlong_to_yard : furlong = 220 * yard

-- Theorem to prove
theorem mile_to_yard : mile = 1760 * yard := by
  sorry

end NUMINAMATH_CALUDE_mile_to_yard_l1040_104082


namespace NUMINAMATH_CALUDE_machine_value_depletion_rate_l1040_104004

/-- Proves that the annual value depletion rate is 0.1 for a machine with given initial and final values over 2 years -/
theorem machine_value_depletion_rate
  (initial_value : ℝ)
  (final_value : ℝ)
  (time_period : ℝ)
  (h1 : initial_value = 900)
  (h2 : final_value = 729)
  (h3 : time_period = 2)
  : ∃ (rate : ℝ), rate = 0.1 ∧ final_value = initial_value * (1 - rate) ^ time_period :=
sorry

end NUMINAMATH_CALUDE_machine_value_depletion_rate_l1040_104004


namespace NUMINAMATH_CALUDE_function_and_value_proof_l1040_104036

noncomputable section

-- Define the function f
def f (A : ℝ) (φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (x + φ)

-- State the theorem
theorem function_and_value_proof 
  (A : ℝ) (φ : ℝ) (α β : ℝ) 
  (h1 : A > 0) 
  (h2 : 0 < φ) (h3 : φ < π) 
  (h4 : ∀ x, f A φ x ≤ 1) 
  (h5 : f A φ (π/3) = 1/2) 
  (h6 : 0 < α) (h7 : α < π/2) 
  (h8 : 0 < β) (h9 : β < π/2) 
  (h10 : f A φ α = 3/5) 
  (h11 : f A φ β = 12/13) :
  (∀ x, f A φ x = Real.cos x) ∧ (f A φ (α - β) = 56/65) := by
  sorry

end

end NUMINAMATH_CALUDE_function_and_value_proof_l1040_104036


namespace NUMINAMATH_CALUDE_ellipse_tangent_min_length_l1040_104050

/-
  Define the ellipse C₁: x²/a² + y² = 1 (a > 1)
  where |F₁F₂|² is the arithmetic mean of |A₁A₂|² and |B₁B₂|²
-/
def C₁ (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 = 1 ∧ a > 1 ∧ 
  ∃ (b c : ℝ), 2 * (2*c)^2 = (2*a)^2 + (2*b)^2 ∧ b = 1

-- Define the curve C₂
def C₂ (t x y : ℝ) : Prop :=
  (x - t)^2 + y^2 = (t^2 + Real.sqrt 3 * t)^2 ∧ 0 < t ∧ t ≤ Real.sqrt 2 / 2

-- Define the tangent line l passing through the left vertex of C₁
def tangent_line (a t k : ℝ) : Prop :=
  ∃ (x y : ℝ), C₂ t x y ∧ y = k * (x + Real.sqrt 3)

-- Theorem statement
theorem ellipse_tangent_min_length :
  ∃ (a : ℝ), C₁ a (-Real.sqrt 3) 0 ∧
  (∀ x y, C₁ a x y ↔ x^2 / 3 + y^2 = 1) ∧
  (∀ t k, tangent_line a t k →
    ∃ (x y : ℝ), C₁ a x y ∧ y = k * (x + Real.sqrt 3) ∧
    ∀ (x' y' : ℝ), C₁ a x' y' ∧ y' = k * (x' + Real.sqrt 3) →
      (x - (-Real.sqrt 3))^2 + y^2 ≥ 3/2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_tangent_min_length_l1040_104050


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l1040_104045

theorem average_of_six_numbers (numbers : Fin 6 → ℝ) 
  (h1 : (numbers 0 + numbers 1) / 2 = 1.1)
  (h2 : (numbers 2 + numbers 3) / 2 = 1.4)
  (h3 : (numbers 4 + numbers 5) / 2 = 5) :
  (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l1040_104045


namespace NUMINAMATH_CALUDE_no_matrix_satisfies_condition_l1040_104063

theorem no_matrix_satisfies_condition : 
  ¬∃ (N : Matrix (Fin 2) (Fin 2) ℝ), 
    ∀ (x y z w : ℝ), 
      N * !![x, y; z, w] = !![2*x, 3*y; 4*z, 5*w] := by
sorry

end NUMINAMATH_CALUDE_no_matrix_satisfies_condition_l1040_104063


namespace NUMINAMATH_CALUDE_joan_can_buy_5_apples_l1040_104026

/-- Represents the grocery shopping problem --/
def grocery_problem (total_money : ℕ) (hummus_price : ℕ) (hummus_quantity : ℕ)
  (chicken_price : ℕ) (bacon_price : ℕ) (vegetable_price : ℕ) (apple_price : ℕ) : Prop :=
  let remaining_money := total_money - (hummus_price * hummus_quantity + chicken_price + bacon_price + vegetable_price)
  remaining_money / apple_price = 5

/-- Theorem stating that Joan can buy 5 apples with her remaining money --/
theorem joan_can_buy_5_apples :
  grocery_problem 60 5 2 20 10 10 2 := by
  sorry

#check joan_can_buy_5_apples

end NUMINAMATH_CALUDE_joan_can_buy_5_apples_l1040_104026


namespace NUMINAMATH_CALUDE_isabel_songs_proof_l1040_104043

/-- The number of songs Isabel bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Proof that Isabel bought 72 songs -/
theorem isabel_songs_proof :
  total_songs 6 2 9 = 72 := by
  sorry

end NUMINAMATH_CALUDE_isabel_songs_proof_l1040_104043


namespace NUMINAMATH_CALUDE_range_of_p_l1040_104091

def h (x : ℝ) : ℝ := 2 * x + 3

def p (x : ℝ) : ℝ := h (h (h (h x)))

theorem range_of_p :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3, 29 ≤ p x ∧ p x ≤ 93 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_p_l1040_104091


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l1040_104002

theorem junk_mail_distribution (total_mail : ℕ) (num_blocks : ℕ) 
  (h1 : total_mail = 192) 
  (h2 : num_blocks = 4) :
  total_mail / num_blocks = 48 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l1040_104002


namespace NUMINAMATH_CALUDE_race_heartbeats_l1040_104037

/-- Calculates the total number of heartbeats during a race -/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proves that the total number of heartbeats during the specified race is 28800 -/
theorem race_heartbeats :
  total_heartbeats 160 30 6 = 28800 := by
  sorry

#eval total_heartbeats 160 30 6

end NUMINAMATH_CALUDE_race_heartbeats_l1040_104037


namespace NUMINAMATH_CALUDE_seven_digit_palindromes_count_l1040_104023

/-- A function that counts the number of seven-digit palindromes with leading digit 1 or 2 -/
def count_seven_digit_palindromes : ℕ :=
  let leading_digits := 2  -- Number of choices for the leading digit (1 or 2)
  let middle_digits := 10 * 10 * 10  -- Number of choices for the middle three digits
  leading_digits * middle_digits

/-- Theorem stating that the number of seven-digit palindromes with leading digit 1 or 2 is 2000 -/
theorem seven_digit_palindromes_count : count_seven_digit_palindromes = 2000 := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_palindromes_count_l1040_104023


namespace NUMINAMATH_CALUDE_sunrise_is_certain_event_l1040_104061

-- Define the type for events
inductive Event
| TV : Event
| Dice : Event
| Sunrise : Event
| SeedGermination : Event

-- Define the property of being a certain event
def isCertainEvent (e : Event) : Prop :=
  match e with
  | Event.TV => False
  | Event.Dice => False
  | Event.Sunrise => True
  | Event.SeedGermination => False

-- Theorem statement
theorem sunrise_is_certain_event : isCertainEvent Event.Sunrise := by
  sorry

end NUMINAMATH_CALUDE_sunrise_is_certain_event_l1040_104061


namespace NUMINAMATH_CALUDE_conference_handshakes_l1040_104086

def number_of_attendees : ℕ := 10

def handshake (a b : ℕ) : Prop := a ≠ b

def total_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

theorem conference_handshakes :
  total_handshakes number_of_attendees = 45 :=
sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1040_104086


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1040_104081

theorem possible_values_of_a (a : ℝ) : 
  let P : Set ℝ := {-1, 2*a+1, a^2-1}
  0 ∈ P → a = -1/2 ∨ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1040_104081


namespace NUMINAMATH_CALUDE_sandwich_count_l1040_104075

def num_bread_types : ℕ := 12
def num_spread_types : ℕ := 10

def sandwich_combinations : ℕ := num_bread_types * (num_spread_types.choose 2)

theorem sandwich_count : sandwich_combinations = 540 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_count_l1040_104075


namespace NUMINAMATH_CALUDE_group_value_l1040_104049

theorem group_value (a : ℝ) (h : 21 ≤ a ∧ a < 41) : (21 + 41) / 2 = 31 := by
  sorry

#check group_value

end NUMINAMATH_CALUDE_group_value_l1040_104049


namespace NUMINAMATH_CALUDE_sqrt_2_2801_eq_1_51_square_diff_16_2_16_1_square_diff_less_than_3_01_l1040_104001

-- Define the square function
def square (x : ℝ) : ℝ := x * x

-- Statement 1: √2.2801 = 1.51
theorem sqrt_2_2801_eq_1_51 : Real.sqrt 2.2801 = 1.51 := by sorry

-- Statement 2: 16.2² - 16.1² = 3.23
theorem square_diff_16_2_16_1 : square 16.2 - square 16.1 = 3.23 := by sorry

-- Statement 3: For any x where 0 < x < 15, (x + 0.1)² - x² < 3.01
theorem square_diff_less_than_3_01 (x : ℝ) (h1 : 0 < x) (h2 : x < 15) :
  square (x + 0.1) - square x < 3.01 := by sorry

end NUMINAMATH_CALUDE_sqrt_2_2801_eq_1_51_square_diff_16_2_16_1_square_diff_less_than_3_01_l1040_104001


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l1040_104071

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 0 → m < n → n % m ≠ 0

/-- The roots of a quadratic equation ax² + bx + c = 0 are given by the quadratic formula:
    x = (-b ± √(b² - 4ac)) / (2a) -/
def isRootOfQuadratic (x k : ℝ) : Prop := x^2 - 72*x + k = 0

theorem unique_k_for_prime_roots : 
  ∃! k : ℝ, ∃ p q : ℕ, 
    isPrime p ∧ 
    isPrime q ∧ 
    isRootOfQuadratic p k ∧ 
    isRootOfQuadratic q k ∧
    k = 335 := by sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l1040_104071


namespace NUMINAMATH_CALUDE_nine_times_reverse_is_9801_l1040_104074

def reverse_number (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d => acc * 10 + d) 0

theorem nine_times_reverse_is_9801 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 → (n = 9 * reverse_number n) → n = 9801 :=
by sorry

end NUMINAMATH_CALUDE_nine_times_reverse_is_9801_l1040_104074


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l1040_104058

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | x > a}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∪ B = {x | 1 < x ≤ 8}
theorem union_A_B : A ∪ B = {x | 1 < x ∧ x ≤ 8} := by sorry

-- Theorem 2: (∁ᵤA) ∩ B = {x | 1 < x < 2}
theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = {x | 1 < x ∧ x < 2} := by sorry

-- Theorem 3: If A ∩ C ≠ ∅, then a < 8
theorem intersection_A_C_nonempty (a : ℝ) : (A ∩ C a).Nonempty → a < 8 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_l1040_104058


namespace NUMINAMATH_CALUDE_inequality_problem_l1040_104021

theorem inequality_problem (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0 → 2*x^2 - 9*x + a < 0) → 
  a ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l1040_104021


namespace NUMINAMATH_CALUDE_max_teams_tied_for_most_wins_l1040_104069

/-- Represents a round-robin tournament --/
structure Tournament :=
  (num_teams : ℕ)
  (wins : Fin num_teams → ℕ)

/-- The total number of games in a round-robin tournament --/
def total_games (t : Tournament) : ℕ :=
  t.num_teams * (t.num_teams - 1) / 2

/-- The maximum number of wins for any team in the tournament --/
def max_wins (t : Tournament) : ℕ :=
  Finset.sup Finset.univ t.wins

/-- The number of teams tied for the maximum number of wins --/
def num_teams_with_max_wins (t : Tournament) : ℕ :=
  Finset.card (Finset.filter (λ i => t.wins i = max_wins t) Finset.univ)

/-- The main theorem --/
theorem max_teams_tied_for_most_wins :
  ∃ (t : Tournament), t.num_teams = 8 ∧
  (∀ (t' : Tournament), t'.num_teams = 8 →
    num_teams_with_max_wins t' ≤ num_teams_with_max_wins t) ∧
  num_teams_with_max_wins t = 7 :=
sorry

end NUMINAMATH_CALUDE_max_teams_tied_for_most_wins_l1040_104069


namespace NUMINAMATH_CALUDE_triangle_properties_l1040_104099

theorem triangle_properties (a b c : ℝ) (A : ℝ) (area : ℝ) :
  (a + b + c) * (b + c - a) = 3 * b * c →
  a = 2 →
  area = Real.sqrt 3 →
  A = π / 3 ∧ b = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1040_104099


namespace NUMINAMATH_CALUDE_triangle_cosine_value_l1040_104032

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b - c = 1/4 * a and 2 * sin B = 3 * sin C, then cos A = -1/4 -/
theorem triangle_cosine_value (a b c A B C : ℝ) :
  b - c = (1/4) * a →
  2 * Real.sin B = 3 * Real.sin C →
  Real.cos A = -(1/4) :=
by sorry

end NUMINAMATH_CALUDE_triangle_cosine_value_l1040_104032


namespace NUMINAMATH_CALUDE_expression_value_l1040_104011

theorem expression_value (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + d^2 - a*d = b^2 + c^2 + b*c) (h2 : a^2 + b^2 = c^2 + d^2) :
  (a*b + c*d) / (a*d + b*c) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_expression_value_l1040_104011


namespace NUMINAMATH_CALUDE_sequence_property_l1040_104039

theorem sequence_property (x y z : ℝ) 
  (h1 : (4 * y) ^ 2 = (3 * x) * (5 * z))  -- Geometric sequence condition
  (h2 : 2 / y = 1 / x + 1 / z)            -- Arithmetic sequence condition
  : x / z + z / x = 34 / 15 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l1040_104039


namespace NUMINAMATH_CALUDE_candy_mixture_l1040_104031

theorem candy_mixture :
  ∀ (x y : ℝ),
  x + y = 100 →
  18 * x + 10 * y = 15 * 100 →
  x = 62.5 ∧ y = 37.5 :=
by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_l1040_104031


namespace NUMINAMATH_CALUDE_point_P_in_second_quadrant_l1040_104027

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : CartesianPoint) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: Point P(-3, 2) is in the second quadrant -/
theorem point_P_in_second_quadrant :
  let P : CartesianPoint := ⟨-3, 2⟩
  is_in_second_quadrant P := by
  sorry

end NUMINAMATH_CALUDE_point_P_in_second_quadrant_l1040_104027


namespace NUMINAMATH_CALUDE_layer_cake_frosting_usage_l1040_104014

/-- Represents the amount of frosting in cans used for different types of baked goods. -/
structure FrostingUsage where
  single_cake : ℚ
  pan_brownies : ℚ
  dozen_cupcakes : ℚ
  layer_cake : ℚ

/-- Represents the quantities of different baked goods to be frosted. -/
structure BakedGoods where
  layer_cakes : ℕ
  dozen_cupcakes : ℕ
  single_cakes : ℕ
  pans_brownies : ℕ

/-- Calculates the total number of cans of frosting needed for a given set of baked goods and frosting usage. -/
def total_frosting (usage : FrostingUsage) (goods : BakedGoods) : ℚ :=
  usage.layer_cake * goods.layer_cakes +
  usage.dozen_cupcakes * goods.dozen_cupcakes +
  usage.single_cake * goods.single_cakes +
  usage.pan_brownies * goods.pans_brownies

/-- The main theorem stating that the amount of frosting used for a layer cake is 1 can. -/
theorem layer_cake_frosting_usage
  (usage : FrostingUsage)
  (goods : BakedGoods)
  (h1 : usage.single_cake = 1/2)
  (h2 : usage.pan_brownies = 1/2)
  (h3 : usage.dozen_cupcakes = 1/2)
  (h4 : goods.layer_cakes = 3)
  (h5 : goods.dozen_cupcakes = 6)
  (h6 : goods.single_cakes = 12)
  (h7 : goods.pans_brownies = 18)
  (h8 : total_frosting usage goods = 21)
  : usage.layer_cake = 1 := by
  sorry

end NUMINAMATH_CALUDE_layer_cake_frosting_usage_l1040_104014


namespace NUMINAMATH_CALUDE_final_produce_theorem_l1040_104044

/-- Represents the quantity of produce -/
structure Produce where
  potatoes : ℕ
  cantaloupes : ℕ
  cucumbers : ℕ

/-- Calculates the final quantity of produce after various events -/
def finalProduce (initial : Produce) : Produce :=
  let potatoesAfterRabbits := initial.potatoes - initial.potatoes / 2
  let cantaloupesAfterSquirrels := initial.cantaloupes - initial.cantaloupes / 4
  let cantaloupesAfterGift := cantaloupesAfterSquirrels + initial.cantaloupes / 2
  let cucumbersAfterRabbits := initial.cucumbers - 2
  let cucumbersAfterHarvest := cucumbersAfterRabbits - (cucumbersAfterRabbits * 3) / 4
  { potatoes := potatoesAfterRabbits,
    cantaloupes := cantaloupesAfterGift,
    cucumbers := cucumbersAfterHarvest }

theorem final_produce_theorem (initial : Produce) :
  initial.potatoes = 7 ∧ initial.cantaloupes = 4 ∧ initial.cucumbers = 5 →
  finalProduce initial = { potatoes := 4, cantaloupes := 5, cucumbers := 1 } :=
by sorry

end NUMINAMATH_CALUDE_final_produce_theorem_l1040_104044


namespace NUMINAMATH_CALUDE_train_crossing_time_l1040_104094

/-- Given a train and a platform with specific dimensions and time to pass,
    calculate the time it takes for the train to cross a tree. -/
theorem train_crossing_time (train_length platform_length time_to_pass_platform : ℝ)
  (h1 : train_length = 1200)
  (h2 : platform_length = 1100)
  (h3 : time_to_pass_platform = 230) :
  let total_distance := train_length + platform_length
  let train_speed := total_distance / time_to_pass_platform
  train_length / train_speed = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1040_104094


namespace NUMINAMATH_CALUDE_product_greater_than_sum_l1040_104030

theorem product_greater_than_sum (a b : ℝ) (ha : a > 2) (hb : b > 2) : a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_l1040_104030


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l1040_104051

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define axis of symmetry for a function
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem statement
theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) :
  EvenFunction f → AxisOfSymmetry (fun x ↦ f (x + 1)) (-1) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l1040_104051


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1040_104038

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) :
  (∃ (x y z : ℝ), x^2 + y^2 + z^2 = 9 ∧ 
    (x - y)^2 + (y - z)^2 + (z - x)^2 ≥ (a - b)^2 + (b - c)^2 + (c - a)^2) ∧
  (∀ (x y z : ℝ), x^2 + y^2 + z^2 = 9 → 
    (x - y)^2 + (y - z)^2 + (z - x)^2 ≤ 27) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1040_104038


namespace NUMINAMATH_CALUDE_infinite_geometric_sum_l1040_104097

/-- The sum of an infinite geometric sequence with first term 1 and common ratio -1/2 is 2/3 -/
theorem infinite_geometric_sum : 
  ∀ (a : ℕ → ℚ), 
  (a 0 = 1) → 
  (∀ n : ℕ, a (n + 1) = a n * (-1/2)) → 
  (∑' n, a n) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_infinite_geometric_sum_l1040_104097


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_not_three_l1040_104079

theorem negation_of_absolute_value_not_three :
  (¬ ∀ x : ℤ, abs x ≠ 3) ↔ (∃ x : ℤ, abs x = 3) := by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_not_three_l1040_104079


namespace NUMINAMATH_CALUDE_trees_planted_l1040_104083

theorem trees_planted (initial_trees final_trees : ℕ) (h1 : initial_trees = 13) (h2 : final_trees = 25) :
  final_trees - initial_trees = 12 := by
  sorry

end NUMINAMATH_CALUDE_trees_planted_l1040_104083


namespace NUMINAMATH_CALUDE_bobbys_paycheck_l1040_104064

/-- Calculates Bobby's final paycheck amount --/
def calculate_paycheck (salary : ℝ) (performance_rate : ℝ) (federal_tax_rate : ℝ) 
  (state_tax_rate : ℝ) (local_tax_rate : ℝ) (health_insurance : ℝ) (life_insurance : ℝ) 
  (parking_fee : ℝ) (retirement_rate : ℝ) : ℝ :=
  let bonus := salary * performance_rate
  let total_income := salary + bonus
  let federal_tax := total_income * federal_tax_rate
  let state_tax := total_income * state_tax_rate
  let local_tax := total_income * local_tax_rate
  let total_taxes := federal_tax + state_tax + local_tax
  let other_deductions := health_insurance + life_insurance + parking_fee
  let retirement_contribution := salary * retirement_rate
  total_income - total_taxes - other_deductions - retirement_contribution

/-- Theorem stating that Bobby's final paycheck amount is $176.98 --/
theorem bobbys_paycheck : 
  calculate_paycheck 450 0.12 (1/3) 0.08 0.05 50 20 10 0.03 = 176.98 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_paycheck_l1040_104064


namespace NUMINAMATH_CALUDE_middle_number_proof_l1040_104042

theorem middle_number_proof (x y z : ℕ) 
  (sum_xy : x + y = 20)
  (sum_xz : x + z = 26)
  (sum_yz : y + z = 30) :
  y = 12 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l1040_104042


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1040_104025

def inequality (a x : ℝ) : Prop := (a + 1) * x - 3 < x - 1

def solution_set (a : ℝ) : Set ℝ :=
  if a < 0 then {x | x < 2/a ∨ x > 1}
  else if a = 0 then {x | x > 1}
  else if 0 < a ∧ a < 2 then {x | 1 < x ∧ x < 2/a}
  else if a = 2 then ∅
  else {x | 2/a < x ∧ x < 1}

theorem inequality_solution_set (a : ℝ) :
  {x : ℝ | inequality a x} = solution_set a :=
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1040_104025


namespace NUMINAMATH_CALUDE_percentage_of_value_in_quarters_l1040_104024

theorem percentage_of_value_in_quarters : 
  let num_dimes : ℕ := 40
  let num_quarters : ℕ := 30
  let num_nickels : ℕ := 10
  let value_dime : ℕ := 10
  let value_quarter : ℕ := 25
  let value_nickel : ℕ := 5
  let total_value := num_dimes * value_dime + num_quarters * value_quarter + num_nickels * value_nickel
  let value_in_quarters := num_quarters * value_quarter
  (value_in_quarters : ℚ) / total_value * 100 = 62.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_value_in_quarters_l1040_104024


namespace NUMINAMATH_CALUDE_annual_cost_difference_l1040_104008

/-- Calculates the annual cost difference between combined piano, violin, and singing lessons
    and clarinet lessons, given the hourly rates and weekly hours for each lesson type. -/
theorem annual_cost_difference
  (clarinet_rate : ℕ) (clarinet_hours : ℕ)
  (piano_rate : ℕ) (piano_hours : ℕ)
  (violin_rate : ℕ) (violin_hours : ℕ)
  (singing_rate : ℕ) (singing_hours : ℕ)
  (h1 : clarinet_rate = 40)
  (h2 : clarinet_hours = 3)
  (h3 : piano_rate = 28)
  (h4 : piano_hours = 5)
  (h5 : violin_rate = 35)
  (h6 : violin_hours = 2)
  (h7 : singing_rate = 45)
  (h8 : singing_hours = 1)
  : (piano_rate * piano_hours + violin_rate * violin_hours + singing_rate * singing_hours) * 52 -
    (clarinet_rate * clarinet_hours) * 52 = 7020 := by
  sorry

#eval (28 * 5 + 35 * 2 + 45 * 1) * 52 - (40 * 3) * 52

end NUMINAMATH_CALUDE_annual_cost_difference_l1040_104008


namespace NUMINAMATH_CALUDE_mars_other_elements_weight_l1040_104084

/-- The weight of the moon in tons -/
def moon_weight : ℝ := 250

/-- The ratio of iron in the moon's composition -/
def iron_ratio : ℝ := 0.5

/-- The ratio of carbon in the moon's composition -/
def carbon_ratio : ℝ := 0.2

/-- The ratio of Mars' weight to the moon's weight -/
def mars_moon_weight_ratio : ℝ := 2

theorem mars_other_elements_weight :
  let other_ratio : ℝ := 1 - iron_ratio - carbon_ratio
  let moon_other_weight : ℝ := other_ratio * moon_weight
  let mars_other_weight : ℝ := mars_moon_weight_ratio * moon_other_weight
  mars_other_weight = 150 := by sorry

end NUMINAMATH_CALUDE_mars_other_elements_weight_l1040_104084


namespace NUMINAMATH_CALUDE_min_value_expression_l1040_104068

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x * y) ≥ 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ (((x₀^2 + y₀^2) * (4*x₀^2 + y₀^2)).sqrt) / (x₀ * y₀) = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1040_104068


namespace NUMINAMATH_CALUDE_kellys_vacation_duration_l1040_104067

/-- Kelly's vacation duration calculation -/
theorem kellys_vacation_duration :
  let travel_days : ℕ := 1 + 1 + 2 + 2  -- Sum of all travel days
  let stay_days : ℕ := 5 + 5 + 5        -- Sum of all stay days
  let total_days : ℕ := travel_days + stay_days
  let days_per_week : ℕ := 7
  (total_days / days_per_week : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_kellys_vacation_duration_l1040_104067


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l1040_104089

theorem geometric_progression_solution : 
  ∃! x : ℝ, (30 + x)^2 = (15 + x) * (60 + x) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l1040_104089


namespace NUMINAMATH_CALUDE_total_lemons_picked_l1040_104048

theorem total_lemons_picked (sally_lemons mary_lemons : ℕ) 
  (h1 : sally_lemons = 7)
  (h2 : mary_lemons = 9) :
  sally_lemons + mary_lemons = 16 := by
sorry

end NUMINAMATH_CALUDE_total_lemons_picked_l1040_104048


namespace NUMINAMATH_CALUDE_chloe_dimes_needed_l1040_104003

/-- Represents the minimum number of dimes needed to purchase a hoodie -/
def min_dimes_needed (hoodie_cost : ℚ) (ten_dollar_bills : ℕ) (quarters : ℕ) (one_dollar_coins : ℕ) : ℕ :=
  let current_money : ℚ := 10 * ten_dollar_bills + 0.25 * quarters + one_dollar_coins
  ⌈(hoodie_cost - current_money) / 0.1⌉₊

/-- Theorem stating that Chloe needs 0 additional dimes to buy the hoodie -/
theorem chloe_dimes_needed : 
  min_dimes_needed 45.50 4 10 3 = 0 := by
  sorry

#eval min_dimes_needed 45.50 4 10 3

end NUMINAMATH_CALUDE_chloe_dimes_needed_l1040_104003


namespace NUMINAMATH_CALUDE_tangent_power_equality_l1040_104007

open Complex

theorem tangent_power_equality (α : ℝ) (n : ℕ) :
  ((1 + I * Real.tan α) / (1 - I * Real.tan α)) ^ n = 
  (1 + I * Real.tan (n * α)) / (1 - I * Real.tan (n * α)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_power_equality_l1040_104007


namespace NUMINAMATH_CALUDE_pump_x_portion_l1040_104093

/-- Represents the pumping scenario with two pumps -/
structure PumpingScenario where
  total_water : ℝ
  pump_x_rate : ℝ
  pump_y_rate : ℝ

/-- The conditions of the pumping scenario -/
def pumping_conditions (s : PumpingScenario) : Prop :=
  s.pump_x_rate > 0 ∧
  s.pump_y_rate > 0 ∧
  3 * s.pump_x_rate + 3 * (s.pump_x_rate + s.pump_y_rate) = s.total_water ∧
  20 * s.pump_y_rate = s.total_water

/-- The theorem stating that Pump X pumps out 17/40 of the total water in the first 3 hours -/
theorem pump_x_portion (s : PumpingScenario) 
  (h : pumping_conditions s) : 
  3 * s.pump_x_rate = (17 / 40) * s.total_water := by
  sorry

end NUMINAMATH_CALUDE_pump_x_portion_l1040_104093


namespace NUMINAMATH_CALUDE_subset_proof_l1040_104029

def M : Set ℕ := {1}
def N : Set ℕ := {1, 2, 3}

theorem subset_proof : M ⊆ N := by sorry

end NUMINAMATH_CALUDE_subset_proof_l1040_104029


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l1040_104009

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ

/-- Calculates the number of sampled elements within the given interval -/
def elements_in_interval (s : SystematicSample) : ℕ :=
  ((s.interval_end - s.interval_start + 1) * s.sample_size + s.population - 1) / s.population

/-- The main theorem stating that for the given scenario, 12 people fall within the interval -/
theorem systematic_sample_theorem (s : SystematicSample) 
  (h1 : s.population = 840)
  (h2 : s.sample_size = 42)
  (h3 : s.interval_start = 481)
  (h4 : s.interval_end = 720) :
  elements_in_interval s = 12 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l1040_104009


namespace NUMINAMATH_CALUDE_factorization_equality_l1040_104015

theorem factorization_equality (x y z : ℝ) :
  x^2 - 4*y^2 - z^2 + 4*y*z = (x + 2*y - z) * (x - 2*y + z) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1040_104015


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l1040_104062

theorem quadratic_function_minimum (a b c : ℝ) (h_a : a ≠ 0) :
  let f := fun x => a * x^2 + b * x + c
  let f' := fun x => 2 * a * x + b
  (f' 0 > 0) →
  (∀ x, f x ≥ 0) →
  ∀ ε > 0, ∃ x, f x / f' 0 < 2 + ε :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l1040_104062


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l1040_104085

theorem geometric_series_ratio (a r : ℝ) (h : a ≠ 0) (hr : abs r < 1) :
  (a / (1 - r)) = 8 * (a * r^2 / (1 - r)) →
  r = 1 / (2 * Real.sqrt 2) ∨ r = -1 / (2 * Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l1040_104085


namespace NUMINAMATH_CALUDE_no_perfect_square_133_base_n_l1040_104092

/-- Represents a number in base n -/
def base_n (digits : List Nat) (n : Nat) : Nat :=
  digits.foldr (fun d acc => d + n * acc) 0

/-- Checks if a number is a perfect square -/
def is_perfect_square (m : Nat) : Prop :=
  ∃ k : Nat, k * k = m

theorem no_perfect_square_133_base_n :
  ¬∃ n : Nat, 5 ≤ n ∧ n ≤ 15 ∧ is_perfect_square (base_n [1, 3, 3] n) := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_133_base_n_l1040_104092


namespace NUMINAMATH_CALUDE_smallest_common_multiple_l1040_104090

theorem smallest_common_multiple : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n % 6 = 0 ∧ n % 5 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0) ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ m % 6 = 0 ∧ m % 5 = 0 ∧ m % 8 = 0 ∧ m % 9 = 0 → m ≥ n) ∧
  n = 360 :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_l1040_104090


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1040_104034

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 16 = 0 ↔ x = 4 ∨ x = -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1040_104034


namespace NUMINAMATH_CALUDE_equality_from_cubic_relations_equality_from_mixed_cubic_relations_l1040_104018

theorem equality_from_cubic_relations (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * (b^3 + c^3) = b * (c^3 + a^3) ∧ b * (c^3 + a^3) = c * (a^3 + b^3)) → 
  (a = b ∧ b = c) :=
by sorry

theorem equality_from_mixed_cubic_relations (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * (a^3 + b^3) = b * (b^3 + c^3) ∧ b * (b^3 + c^3) = c * (c^3 + a^3)) → 
  (a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_equality_from_cubic_relations_equality_from_mixed_cubic_relations_l1040_104018


namespace NUMINAMATH_CALUDE_percent_calculation_l1040_104066

theorem percent_calculation (x : ℝ) (h : 0.4 * x = 160) : 0.2 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_percent_calculation_l1040_104066


namespace NUMINAMATH_CALUDE_square_root_sum_implies_product_l1040_104057

theorem square_root_sum_implies_product (x : ℝ) :
  (Real.sqrt (7 + x) + Real.sqrt (25 - x) = 8) →
  ((7 + x) * (25 - x) = 256) := by
sorry

end NUMINAMATH_CALUDE_square_root_sum_implies_product_l1040_104057


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1040_104019

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 8 → b = 15 → c^2 = a^2 + b^2 → c = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1040_104019


namespace NUMINAMATH_CALUDE_u_equals_fib_l1040_104078

/-- Array I as defined in the problem -/
def array_I (n : ℕ) : Fin n → Fin 3 → ℕ :=
  λ i j => match j with
    | 0 => i + 1
    | 1 => i + 2
    | 2 => i + 3

/-- Number of SDRs for array I -/
def u (n : ℕ) : ℕ := sorry

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Theorem stating that u_n is equal to the (n+1)th Fibonacci number for n ≥ 2 -/
theorem u_equals_fib (n : ℕ) (h : n ≥ 2) : u n = fib (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_u_equals_fib_l1040_104078


namespace NUMINAMATH_CALUDE_rock_volume_l1040_104047

/-- Calculates the volume of a rock based on the water level rise in a rectangular tank. -/
theorem rock_volume (tank_length tank_width water_rise : ℝ) 
  (h1 : tank_length = 30)
  (h2 : tank_width = 20)
  (h3 : water_rise = 4) :
  tank_length * tank_width * water_rise = 2400 := by
  sorry

#check rock_volume

end NUMINAMATH_CALUDE_rock_volume_l1040_104047


namespace NUMINAMATH_CALUDE_movie_of_the_year_fraction_l1040_104076

theorem movie_of_the_year_fraction (total_members : ℕ) (min_appearances : ℚ) : 
  total_members = 795 → min_appearances = 198.75 → min_appearances / total_members = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_movie_of_the_year_fraction_l1040_104076


namespace NUMINAMATH_CALUDE_hannah_dog_food_theorem_l1040_104088

/-- The amount of dog food Hannah needs to prepare daily for her three dogs -/
def total_dog_food (first_dog_food : ℝ) (second_dog_multiplier : ℝ) (third_dog_additional : ℝ) : ℝ :=
  first_dog_food + 
  (first_dog_food * second_dog_multiplier) + 
  (first_dog_food * second_dog_multiplier + third_dog_additional)

/-- Theorem stating that Hannah needs to prepare 10 cups of dog food daily -/
theorem hannah_dog_food_theorem : 
  total_dog_food 1.5 2 2.5 = 10 := by
  sorry

#eval total_dog_food 1.5 2 2.5

end NUMINAMATH_CALUDE_hannah_dog_food_theorem_l1040_104088


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1040_104053

theorem imaginary_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (4 - 3*i) / i
  Complex.im z = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1040_104053


namespace NUMINAMATH_CALUDE_initial_markup_percentage_l1040_104012

theorem initial_markup_percentage (C : ℝ) (M : ℝ) : 
  C > 0 →
  let S₁ := C * (1 + M)
  let S₂ := S₁ * 1.25
  let S₃ := S₂ * 0.94
  S₃ = C * 1.41 →
  M = 0.2 := by sorry

end NUMINAMATH_CALUDE_initial_markup_percentage_l1040_104012


namespace NUMINAMATH_CALUDE_missing_number_in_set_l1040_104006

theorem missing_number_in_set (x : ℝ) (a : ℝ) : 
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (a + 255 + 511 + 1023 + x) / 5 = 398.2 →
  a = 128 := by
sorry

end NUMINAMATH_CALUDE_missing_number_in_set_l1040_104006


namespace NUMINAMATH_CALUDE_prop_range_m_l1040_104098

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- State the theorem
theorem prop_range_m : 
  ∀ m : ℝ, ¬(p m ∨ q m) → m ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_prop_range_m_l1040_104098


namespace NUMINAMATH_CALUDE_sandwich_cost_l1040_104041

theorem sandwich_cost (num_sandwiches num_drinks drink_cost total_cost : ℕ) 
  (h1 : num_sandwiches = 3)
  (h2 : num_drinks = 2)
  (h3 : drink_cost = 4)
  (h4 : total_cost = 26) :
  ∃ (sandwich_cost : ℕ), 
    sandwich_cost = 6 ∧ 
    num_sandwiches * sandwich_cost + num_drinks * drink_cost = total_cost := by
  sorry

end NUMINAMATH_CALUDE_sandwich_cost_l1040_104041


namespace NUMINAMATH_CALUDE_average_diff_100_400_50_250_l1040_104035

def average_difference : ℤ → ℤ → ℤ → ℤ → ℤ :=
  fun a b c d => ((b + a) / 2) - ((d + c) / 2)

theorem average_diff_100_400_50_250 :
  average_difference 100 400 50 250 = 100 := by
  sorry

end NUMINAMATH_CALUDE_average_diff_100_400_50_250_l1040_104035


namespace NUMINAMATH_CALUDE_equation_equivalence_l1040_104070

theorem equation_equivalence (x : ℝ) : x * (2 * x - 1) = 5 * (x + 3) ↔ 2 * x^2 - 6 * x - 15 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1040_104070


namespace NUMINAMATH_CALUDE_exam_candidates_count_l1040_104052

theorem exam_candidates_count :
  ∀ (N : ℕ) (total_avg marks_11th : ℝ) (avg_first_10 avg_last_11 : ℝ),
    total_avg = 48 →
    avg_first_10 = 55 →
    avg_last_11 = 40 →
    marks_11th = 66 →
    N * total_avg = 10 * avg_first_10 + 11 * avg_last_11 - marks_11th →
    N = 21 := by
  sorry

end NUMINAMATH_CALUDE_exam_candidates_count_l1040_104052
