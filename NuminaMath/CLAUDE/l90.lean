import Mathlib

namespace NUMINAMATH_CALUDE_number_list_difference_l90_9072

theorem number_list_difference (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h1 : (x₁ + x₂ + x₃) / 3 = -3)
  (h2 : (x₁ + x₂ + x₃ + x₄) / 4 = 4)
  (h3 : (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = -5) :
  x₄ - x₅ = 66 := by
sorry

end NUMINAMATH_CALUDE_number_list_difference_l90_9072


namespace NUMINAMATH_CALUDE_rod_division_theorem_l90_9080

/-- Represents a rod divided into equal parts -/
structure DividedRod where
  length : ℕ
  divisions : List ℕ

/-- Calculates the total number of segments in a divided rod -/
def totalSegments (rod : DividedRod) : ℕ := sorry

/-- Calculates the length of the shortest segment in a divided rod -/
def shortestSegment (rod : DividedRod) : ℚ := sorry

/-- Theorem about a specific rod division -/
theorem rod_division_theorem (k : ℕ) :
  let rod := DividedRod.mk (72 * k) [8, 12, 18]
  totalSegments rod = 28 ∧ shortestSegment rod = 1 / 72 := by sorry

end NUMINAMATH_CALUDE_rod_division_theorem_l90_9080


namespace NUMINAMATH_CALUDE_chicken_count_l90_9008

/-- Given a farm with chickens and buffalos, prove the number of chickens. -/
theorem chicken_count (total_animals : ℕ) (total_legs : ℕ) (chickens : ℕ) (buffalos : ℕ) : 
  total_animals = 9 →
  total_legs = 26 →
  chickens + buffalos = total_animals →
  2 * chickens + 4 * buffalos = total_legs →
  chickens = 5 := by
sorry

end NUMINAMATH_CALUDE_chicken_count_l90_9008


namespace NUMINAMATH_CALUDE_david_pushups_difference_l90_9043

/-- Proves that David did 17 more push-ups than Zachary given the conditions in the problem -/
theorem david_pushups_difference (david_crunches zachary_pushups zachary_crunches : ℕ) 
  (h1 : david_crunches = 45)
  (h2 : zachary_pushups = 34)
  (h3 : zachary_crunches = 62)
  (h4 : david_crunches + 17 = zachary_crunches) : 
  ∃ (david_pushups : ℕ), david_pushups - zachary_pushups = 17 := by
sorry

end NUMINAMATH_CALUDE_david_pushups_difference_l90_9043


namespace NUMINAMATH_CALUDE_simplify_expression_l90_9017

theorem simplify_expression (r : ℝ) : 100*r - 48*r + 10 = 52*r + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l90_9017


namespace NUMINAMATH_CALUDE_calculation_difference_l90_9016

theorem calculation_difference : 
  let correct_calc := 8 - (2 + 5)
  let incorrect_calc := 8 - 2 + 5
  correct_calc - incorrect_calc = -10 := by
sorry

end NUMINAMATH_CALUDE_calculation_difference_l90_9016


namespace NUMINAMATH_CALUDE_expression_equals_seven_l90_9097

theorem expression_equals_seven :
  (-2023)^0 + Real.sqrt 4 - 2 * Real.sin (30 * π / 180) + abs (-5) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_seven_l90_9097


namespace NUMINAMATH_CALUDE_rectangular_field_area_decrease_l90_9084

theorem rectangular_field_area_decrease :
  ∀ (L W : ℝ),
  L > 0 → W > 0 →
  let original_area := L * W
  let new_length := L * (1 - 0.4)
  let new_width := W * (1 - 0.4)
  let new_area := new_length * new_width
  (original_area - new_area) / original_area = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_decrease_l90_9084


namespace NUMINAMATH_CALUDE_largest_solution_quadratic_l90_9076

theorem largest_solution_quadratic (x : ℝ) : 
  (9 * x^2 - 51 * x + 70 = 0) → x ≤ 70/9 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_quadratic_l90_9076


namespace NUMINAMATH_CALUDE_square_difference_l90_9021

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l90_9021


namespace NUMINAMATH_CALUDE_sprint_competition_races_l90_9050

/-- Calculates the number of races needed in a sprint competition. -/
def calculate_races (total_sprinters : ℕ) (byes : ℕ) (first_round_lanes : ℕ) (subsequent_lanes : ℕ) : ℕ :=
  let first_round_races := ((total_sprinters - byes + first_round_lanes - 1) / first_round_lanes : ℕ)
  let first_round_winners := first_round_races
  let second_round_competitors := first_round_winners + byes
  let second_round_races := ((second_round_competitors + subsequent_lanes - 1) / subsequent_lanes : ℕ)
  let third_round_races := ((second_round_races + subsequent_lanes - 1) / subsequent_lanes : ℕ)
  let final_race := 1
  first_round_races + second_round_races + third_round_races + final_race

/-- The sprint competition theorem. -/
theorem sprint_competition_races :
  calculate_races 300 16 8 6 = 48 :=
by sorry

end NUMINAMATH_CALUDE_sprint_competition_races_l90_9050


namespace NUMINAMATH_CALUDE_quadratic_inequality_sufficient_conditions_quadratic_inequality_not_necessary_l90_9066

theorem quadratic_inequality_sufficient_conditions 
  (k : ℝ) (h : k = 0 ∨ (-3 < k ∧ k < 0) ∨ (-3 < k ∧ k < -1)) :
  ∀ x : ℝ, 2*k*x^2 + k*x - 3/8 < 0 :=
by sorry

theorem quadratic_inequality_not_necessary 
  (k : ℝ) (h : ∀ x : ℝ, 2*k*x^2 + k*x - 3/8 < 0) :
  ¬(k = 0 ∧ (-3 < k ∧ k < 0) ∧ (-3 < k ∧ k < -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_sufficient_conditions_quadratic_inequality_not_necessary_l90_9066


namespace NUMINAMATH_CALUDE_unique_triangle_side_l90_9051

/-- A function that checks if a triangle with sides a, b, and c can exist -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that 3 is the only positive integer value of x for which
    a triangle with sides 5, x + 1, and x^3 can exist -/
theorem unique_triangle_side : ∀ x : ℕ+, 
  (is_valid_triangle 5 (x + 1) (x^3) ↔ x = 3) := by sorry

end NUMINAMATH_CALUDE_unique_triangle_side_l90_9051


namespace NUMINAMATH_CALUDE_revenue_decrease_percent_l90_9002

/-- Calculates the percentage decrease in revenue when tax is reduced and consumption is increased -/
theorem revenue_decrease_percent (tax_reduction : Real) (consumption_increase : Real)
  (h1 : tax_reduction = 0.20)
  (h2 : consumption_increase = 0.05) :
  1 - (1 - tax_reduction) * (1 + consumption_increase) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_percent_l90_9002


namespace NUMINAMATH_CALUDE_family_savings_by_end_of_2019_l90_9026

/-- Represents the family's financial situation --/
structure FinancialSituation where
  initial_savings : ℕ
  income : ℕ
  expenses : ℕ

/-- Calculates the final savings given a financial situation --/
def calculate_final_savings (fs : FinancialSituation) : ℕ :=
  fs.initial_savings + fs.income - fs.expenses

/-- Theorem: The family's savings by 31.12.2019 will be 1340840 rubles --/
theorem family_savings_by_end_of_2019 :
  let fs : FinancialSituation := {
    initial_savings := 1147240,
    income := 509600,
    expenses := 276000
  }
  calculate_final_savings fs = 1340840 := by
  sorry

#eval calculate_final_savings {
  initial_savings := 1147240,
  income := 509600,
  expenses := 276000
}

end NUMINAMATH_CALUDE_family_savings_by_end_of_2019_l90_9026


namespace NUMINAMATH_CALUDE_translated_min_point_l90_9064

/-- The original function before translation -/
def f (x : ℝ) : ℝ := |x + 1| - 4

/-- The translated function -/
def g (x : ℝ) : ℝ := f (x - 3) - 4

/-- The minimum point of the translated function -/
def min_point : ℝ × ℝ := (2, -8)

theorem translated_min_point :
  (∀ x : ℝ, g x ≥ g (min_point.1)) ∧
  g (min_point.1) = min_point.2 :=
sorry

end NUMINAMATH_CALUDE_translated_min_point_l90_9064


namespace NUMINAMATH_CALUDE_smallest_valid_number_l90_9073

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    n = 1000 + 100 * a + b ∧
    n = (10 * a + b) ^ 2 ∧
    0 ≤ a ∧ a ≤ 99 ∧
    0 ≤ b ∧ b ≤ 99

theorem smallest_valid_number :
  is_valid_number 2025 ∧ ∀ n, is_valid_number n → n ≥ 2025 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l90_9073


namespace NUMINAMATH_CALUDE_coin_distribution_l90_9061

theorem coin_distribution (x y z : ℕ) : 
  x + 2*y + 5*z = 71 →  -- total value is 71 kopecks
  x = y →  -- number of 1-kopeck coins equals number of 2-kopeck coins
  x + y + z = 31 →  -- total number of coins is 31
  (x = 12 ∧ y = 12 ∧ z = 7) :=
by sorry

end NUMINAMATH_CALUDE_coin_distribution_l90_9061


namespace NUMINAMATH_CALUDE_sample_size_is_fifteen_l90_9088

/-- Represents a company with employees and a sampling method -/
structure Company where
  total_employees : ℕ
  young_employees : ℕ
  sample_young : ℕ
  stratified_sampling : Bool

/-- Calculates the sample size for a given company using stratified sampling -/
def calculate_sample_size (c : Company) : ℕ :=
  (c.sample_young * c.total_employees) / c.young_employees

/-- Theorem stating that for the given company, the sample size is 15 -/
theorem sample_size_is_fifteen (c : Company)
  (h1 : c.total_employees = 750)
  (h2 : c.young_employees = 350)
  (h3 : c.sample_young = 7)
  (h4 : c.stratified_sampling = true) :
  calculate_sample_size c = 15 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_fifteen_l90_9088


namespace NUMINAMATH_CALUDE_josies_initial_money_l90_9099

/-- The amount of money Josie's mom gave her initially --/
def initial_money (milk_price bread_price detergent_price banana_price_per_pound : ℚ)
  (milk_discount detergent_discount : ℚ) (banana_pounds leftover_money : ℚ) : ℚ :=
  (milk_price * (1 - milk_discount) + bread_price + 
   (detergent_price - detergent_discount) + 
   (banana_price_per_pound * banana_pounds) + leftover_money)

/-- Theorem stating that Josie's mom gave her $20.00 initially --/
theorem josies_initial_money :
  initial_money 4 3.5 10.25 0.75 0.5 1.25 2 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_josies_initial_money_l90_9099


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l90_9081

theorem sum_remainder_mod_seven : 
  (51730 % 7 + 51731 % 7 + 51732 % 7 + 51733 % 7 + 51734 % 7 + 51735 % 7) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l90_9081


namespace NUMINAMATH_CALUDE_count_squares_five_by_five_l90_9058

/-- Represents a square grid with a dot in the center -/
structure CenteredGrid (n : ℕ) :=
  (size : ℕ)
  (has_center_dot : size % 2 = 1)

/-- Counts the number of squares in a grid that contain the center dot -/
def count_squares_with_dot (grid : CenteredGrid 5) : ℕ :=
  let center := grid.size / 2
  let count_for_size (k : ℕ) : ℕ :=
    if k ≤ grid.size
    then (min (center + 1) (grid.size - k + 1))^2
    else 0
  (List.range grid.size).map count_for_size |> List.sum

/-- The main theorem to prove -/
theorem count_squares_five_by_five :
  ∀ (grid : CenteredGrid 5), count_squares_with_dot grid = 19 :=
sorry

end NUMINAMATH_CALUDE_count_squares_five_by_five_l90_9058


namespace NUMINAMATH_CALUDE_power_sum_ratio_l90_9032

theorem power_sum_ratio (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (sum_zero : a + b + c = 0) :
  (a^7 + b^7 + c^7)^2 / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49/60 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_ratio_l90_9032


namespace NUMINAMATH_CALUDE_pizza_cost_l90_9036

/-- The cost of purchasing pizzas with special pricing -/
theorem pizza_cost (standard_price : ℕ) (triple_cheese_count : ℕ) (meat_lovers_count : ℕ) :
  standard_price = 5 →
  triple_cheese_count = 10 →
  meat_lovers_count = 9 →
  (standard_price * (triple_cheese_count / 2 + 2 * meat_lovers_count / 3) : ℕ) = 55 := by
  sorry

#check pizza_cost

end NUMINAMATH_CALUDE_pizza_cost_l90_9036


namespace NUMINAMATH_CALUDE_direction_vector_b_value_l90_9094

/-- Given a line passing through two points, prove that its direction vector
    in the form (b, -1) has b = 1. -/
theorem direction_vector_b_value 
  (p1 p2 : ℝ × ℝ) 
  (h1 : p1 = (-3, 2)) 
  (h2 : p2 = (2, -3)) 
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ (p2.1 - p1.1, p2.2 - p1.2) = (k * b, k * (-1))) : 
  b = 1 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_b_value_l90_9094


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l90_9005

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1
  a₁_eq_1 : a 1 = 1
  geometric_subseq : (a 3)^2 = a 1 * a 9

/-- The general term of the arithmetic sequence is either n or 1 -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = n) ∨ (∀ n : ℕ, seq.a n = 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l90_9005


namespace NUMINAMATH_CALUDE_circle_center_l90_9065

/-- The center of the circle with equation x^2 + y^2 - x + 2y = 0 has coordinates (1/2, -1) -/
theorem circle_center (x y : ℝ) : 
  x^2 + y^2 - x + 2*y = 0 → (x - 1/2)^2 + (y + 1)^2 = 5/4 := by
sorry

end NUMINAMATH_CALUDE_circle_center_l90_9065


namespace NUMINAMATH_CALUDE_apple_price_proof_l90_9067

def grocery_problem (total_spent milk_price cereal_price banana_price cookie_multiplier
                     milk_qty cereal_qty banana_qty cookie_qty apple_qty : ℚ) : Prop :=
  let cereal_total := cereal_price * cereal_qty
  let banana_total := banana_price * banana_qty
  let cookie_price := milk_price * cookie_multiplier
  let cookie_total := cookie_price * cookie_qty
  let known_items_total := milk_price * milk_qty + cereal_total + banana_total + cookie_total
  let apple_total := total_spent - known_items_total
  let apple_price := apple_total / apple_qty
  apple_price = 0.5

theorem apple_price_proof :
  grocery_problem 25 3 3.5 0.25 2 1 2 4 2 4 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_proof_l90_9067


namespace NUMINAMATH_CALUDE_matrix_square_result_l90_9068

theorem matrix_square_result (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : M.mulVec ![1, 0] = ![1, 0])
  (h2 : M.mulVec ![1, 1] = ![2, 2]) :
  (M ^ 2).mulVec ![1, -1] = ![-2, -4] := by
  sorry

end NUMINAMATH_CALUDE_matrix_square_result_l90_9068


namespace NUMINAMATH_CALUDE_regular_octagon_perimeter_l90_9039

/-- A regular octagon is a polygon with 8 equal sides -/
def RegularOctagon : Type := Unit

/-- The side length of the regular octagon -/
def side_length : ℝ := 3

/-- The number of sides in an octagon -/
def num_sides : ℕ := 8

/-- The perimeter of a regular octagon is the product of its number of sides and side length -/
def perimeter (o : RegularOctagon) : ℝ := num_sides * side_length

theorem regular_octagon_perimeter : 
  ∀ (o : RegularOctagon), perimeter o = 24 := by sorry

end NUMINAMATH_CALUDE_regular_octagon_perimeter_l90_9039


namespace NUMINAMATH_CALUDE_smallest_value_for_y_between_zero_and_one_l90_9052

theorem smallest_value_for_y_between_zero_and_one
  (y : ℝ) (h : 0 < y ∧ y < 1) :
  y^3 < y ∧ y^3 < 3*y ∧ y^3 < y^(1/3) ∧ y^3 < 1/y :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_for_y_between_zero_and_one_l90_9052


namespace NUMINAMATH_CALUDE_legs_on_ground_for_ten_horses_l90_9044

/-- Represents the number of legs walking on the ground given the conditions of the problem --/
def legs_on_ground (num_horses : ℕ) : ℕ :=
  let num_men := num_horses
  let num_walking_men := num_men / 2
  let men_legs := num_walking_men * 2
  let horse_legs := num_horses * 4
  men_legs + horse_legs

/-- Theorem stating that with 10 horses, there are 50 legs walking on the ground --/
theorem legs_on_ground_for_ten_horses :
  legs_on_ground 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_legs_on_ground_for_ten_horses_l90_9044


namespace NUMINAMATH_CALUDE_trigonometric_expression_simplification_l90_9091

open Real

theorem trigonometric_expression_simplification (α : ℝ) :
  (sin (2 * π - α) * cos (π + α) * cos (π / 2 + α) * cos (11 * π / 2 - α)) /
  (cos (π - α) * sin (3 * π - α) * sin (-π - α) * sin (9 * π / 2 + α)) = -tan α :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_simplification_l90_9091


namespace NUMINAMATH_CALUDE_walter_school_allocation_is_correct_l90_9012

/-- Calculates Walter's school expenses allocation based on his work schedule and earnings --/
def walter_school_allocation (
  fast_food_weekday_hours : ℕ)
  (fast_food_weekend_hours : ℕ)
  (fast_food_hourly_rate : ℚ)
  (fast_food_weekday_days : ℕ)
  (fast_food_weekend_days : ℕ)
  (convenience_store_hours : ℕ)
  (convenience_store_hourly_rate : ℚ)
  (school_allocation_fraction : ℚ) : ℚ :=
let fast_food_weekday_earnings := fast_food_weekday_hours * fast_food_weekday_days * fast_food_hourly_rate
let fast_food_weekend_earnings := fast_food_weekend_hours * fast_food_weekend_days * fast_food_hourly_rate
let convenience_store_earnings := convenience_store_hours * convenience_store_hourly_rate
let total_earnings := fast_food_weekday_earnings + fast_food_weekend_earnings + convenience_store_earnings
school_allocation_fraction * total_earnings

/-- Theorem stating that Walter's school expenses allocation is $146.25 --/
theorem walter_school_allocation_is_correct : 
  walter_school_allocation 4 6 5 5 2 5 7 (3/4) = 146.25 := by
  sorry

end NUMINAMATH_CALUDE_walter_school_allocation_is_correct_l90_9012


namespace NUMINAMATH_CALUDE_cross_section_area_l90_9031

-- Define the rectangular parallelepiped
structure RectangularParallelepiped where
  AB : ℝ
  AD : ℝ
  BD : ℝ
  AA₁ : ℝ

-- Define the theorem
theorem cross_section_area (rp : RectangularParallelepiped)
  (h1 : rp.AB = 29)
  (h2 : rp.AD = 36)
  (h3 : rp.BD = 25)
  (h4 : rp.AA₁ = 48) :
  ∃ (area : ℝ), area = 1872 ∧ area = rp.AD * Real.sqrt (rp.AA₁^2 + (Real.sqrt (rp.AD^2 + rp.AB^2 - rp.BD^2))^2) :=
by sorry

end NUMINAMATH_CALUDE_cross_section_area_l90_9031


namespace NUMINAMATH_CALUDE_weights_not_divisible_by_three_l90_9010

theorem weights_not_divisible_by_three :
  ¬ (∃ k : ℕ, 3 * k = (67 * 68) / 2) := by
  sorry

end NUMINAMATH_CALUDE_weights_not_divisible_by_three_l90_9010


namespace NUMINAMATH_CALUDE_rational_triplet_problem_l90_9045

theorem rational_triplet_problem (m n p : ℚ) : 
  m > 0 ∧ n > 0 ∧ p > 0 →
  (∃ (a b c : ℤ), m + 1 / (n * p) = a ∧ n + 1 / (p * m) = b ∧ p + 1 / (m * n) = c) →
  ((m = 1/2 ∧ n = 1/2 ∧ p = 4) ∨ 
   (m = 1/2 ∧ n = 1 ∧ p = 2) ∨ 
   (m = 1 ∧ n = 1 ∧ p = 1) ∨
   (m = 1/2 ∧ n = 4 ∧ p = 1/2) ∨
   (m = 1 ∧ n = 2 ∧ p = 1/2) ∨
   (m = 4 ∧ n = 1/2 ∧ p = 1/2) ∨
   (m = 2 ∧ n = 1/2 ∧ p = 1) ∨
   (m = 2 ∧ n = 1 ∧ p = 1/2) ∨
   (m = 1/2 ∧ n = 2 ∧ p = 1)) :=
by sorry

end NUMINAMATH_CALUDE_rational_triplet_problem_l90_9045


namespace NUMINAMATH_CALUDE_probability_diamond_spade_standard_deck_l90_9082

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (diamonds : ℕ)
  (spades : ℕ)

/-- The probability of drawing a diamond first and then a spade from a standard deck -/
def probability_diamond_then_spade (d : Deck) : ℚ :=
  (d.diamonds : ℚ) / d.total_cards * d.spades / (d.total_cards - 1)

/-- Theorem stating the probability of drawing a diamond first and then a spade from a standard deck -/
theorem probability_diamond_spade_standard_deck :
  ∃ d : Deck, d.total_cards = 52 ∧ d.diamonds = 13 ∧ d.spades = 13 ∧
  probability_diamond_then_spade d = 13 / 204 := by
  sorry

#check probability_diamond_spade_standard_deck

end NUMINAMATH_CALUDE_probability_diamond_spade_standard_deck_l90_9082


namespace NUMINAMATH_CALUDE_f_properties_l90_9060

-- Define the function f(x) = x³ - 3x² + 3
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

theorem f_properties :
  -- 1. The tangent line at (1, f(1)) is 3x + y - 4 = 0
  (∀ x y : ℝ, y = f' 1 * (x - 1) + f 1 ↔ 3*x + y - 4 = 0) ∧
  -- 2. The function has exactly 3 zeros
  (∃! (a b c : ℝ), a < b ∧ b < c ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) ∧
  -- 3. The function is symmetric about the point (1, 1)
  (∀ x : ℝ, f (1 + x) - 1 = -(f (1 - x) - 1)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l90_9060


namespace NUMINAMATH_CALUDE_orlies_age_l90_9083

/-- Proves Orlie's age given the conditions about Ruffy and Orlie's ages -/
theorem orlies_age (ruffy_age orlie_age : ℕ) : 
  ruffy_age = 9 →
  ruffy_age = (3 / 4) * orlie_age →
  ruffy_age - 4 = (1 / 2) * (orlie_age - 4) + 1 →
  orlie_age = 12 := by
  sorry

#check orlies_age

end NUMINAMATH_CALUDE_orlies_age_l90_9083


namespace NUMINAMATH_CALUDE_intersection_point_sum_l90_9007

/-- Given two lines that intersect at (3, 5), prove that a + b = 86/15 -/
theorem intersection_point_sum (a b : ℚ) : 
  (3 = (1/3) * 5 + a) → 
  (5 = (1/5) * 3 + b) → 
  a + b = 86/15 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l90_9007


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l90_9086

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem right_triangle_perimeter : ∃ (a b c : ℕ),
  a = 11 ∧
  is_pythagorean_triple a b c ∧
  a + b + c = 132 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l90_9086


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l90_9069

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 9*x + 20 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 11*x - 60 = (x + b)*(x - c)) →
  a + b + c = 23 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l90_9069


namespace NUMINAMATH_CALUDE_prob_not_all_even_l90_9089

/-- The number of sides on a fair die -/
def sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of even outcomes on a single die -/
def even_outcomes : ℕ := 3

/-- The probability that not all dice show an even number when rolling five fair 6-sided dice -/
theorem prob_not_all_even : 
  1 - (even_outcomes : ℚ) ^ num_dice / sides ^ num_dice = 7533 / 7776 := by
sorry

end NUMINAMATH_CALUDE_prob_not_all_even_l90_9089


namespace NUMINAMATH_CALUDE_incorrect_expression_l90_9028

theorem incorrect_expression (x y : ℚ) (h : x / y = 4 / 5) : 
  (x + 2 * y) / y = 14 / 5 ∧ 
  y / (2 * x - y) = 5 / 3 ∧ 
  (4 * x - y) / y = 11 / 5 ∧ 
  x / (3 * y) = 4 / 15 ∧ 
  (2 * x - y) / x ≠ 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l90_9028


namespace NUMINAMATH_CALUDE_polynomial_value_l90_9096

def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

theorem polynomial_value : f 4 = 1559 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_l90_9096


namespace NUMINAMATH_CALUDE_no_arithmetic_progression_roots_l90_9079

theorem no_arithmetic_progression_roots (a : ℝ) : 
  ¬ ∃ (x d : ℝ), 
    (∀ k : Fin 4, 16 * (x + k * d)^4 - a * (x + k * d)^3 + (2*a + 17) * (x + k * d)^2 - a * (x + k * d) + 16 = 0) ∧
    (d ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_arithmetic_progression_roots_l90_9079


namespace NUMINAMATH_CALUDE_ice_cream_volume_l90_9041

/-- The volume of ice cream in a cone and hemisphere -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1/3) * π * r^2 * h
  let hemisphere_volume := (1/2) * (4/3) * π * r^3
  h = 8 ∧ r = 2 → cone_volume + hemisphere_volume = 16 * π := by sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l90_9041


namespace NUMINAMATH_CALUDE_sector_arc_length_l90_9035

/-- Given a circular sector with area 24π cm² and central angle 216°, 
    its arc length is (12√10π)/5 cm. -/
theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) : 
  area = 24 * Real.pi ∧ 
  angle = 216 →
  arc_length = (12 * Real.sqrt 10 * Real.pi) / 5 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l90_9035


namespace NUMINAMATH_CALUDE_gummy_worm_fraction_l90_9047

theorem gummy_worm_fraction (initial_count : ℕ) (days : ℕ) (final_count : ℕ) (f : ℚ) :
  initial_count = 64 →
  days = 4 →
  final_count = 4 →
  0 < f →
  f < 1 →
  (1 - f) ^ days * initial_count = final_count →
  f = 1/2 := by
sorry

end NUMINAMATH_CALUDE_gummy_worm_fraction_l90_9047


namespace NUMINAMATH_CALUDE_grading_ratio_l90_9023

/-- A grading method for a test with 100 questions. -/
structure GradingMethod where
  total_questions : Nat
  score : Nat
  correct_answers : Nat

/-- Theorem stating the ratio of points subtracted per incorrect answer
    to points given per correct answer is 2:1 -/
theorem grading_ratio (g : GradingMethod)
  (h1 : g.total_questions = 100)
  (h2 : g.score = 73)
  (h3 : g.correct_answers = 91) :
  (g.correct_answers - g.score) / (g.total_questions - g.correct_answers) = 2 := by
  sorry


end NUMINAMATH_CALUDE_grading_ratio_l90_9023


namespace NUMINAMATH_CALUDE_company_shares_l90_9020

theorem company_shares (K : ℝ) (P V S I : ℝ) 
  (h1 : P + V + S + I = K)
  (h2 : K + P = 1.25 * K)
  (h3 : K + V = 1.35 * K)
  (h4 : K + 2 * S = 1.4 * K)
  (h5 : I > 0) :
  ∃ x : ℝ, x > 2.5 ∧ x * I > 0.5 * K := by
  sorry

end NUMINAMATH_CALUDE_company_shares_l90_9020


namespace NUMINAMATH_CALUDE_sandy_payment_l90_9074

/-- Represents the cost and quantity of a coffee shop item -/
structure Item where
  price : ℚ
  quantity : ℕ

/-- Calculates the total cost of an order -/
def orderTotal (items : List Item) : ℚ :=
  items.foldl (fun acc item => acc + item.price * item.quantity) 0

/-- Proves that Sandy paid $20 given the order details and change received -/
theorem sandy_payment (cappuccino iced_tea cafe_latte espresso : Item)
    (change : ℚ) :
    cappuccino.price = 2 →
    iced_tea.price = 3 →
    cafe_latte.price = 3/2 →
    espresso.price = 1 →
    cappuccino.quantity = 3 →
    iced_tea.quantity = 2 →
    cafe_latte.quantity = 2 →
    espresso.quantity = 2 →
    change = 3 →
    orderTotal [cappuccino, iced_tea, cafe_latte, espresso] + change = 20 := by
  sorry


end NUMINAMATH_CALUDE_sandy_payment_l90_9074


namespace NUMINAMATH_CALUDE_figure_b_impossible_l90_9037

-- Define the shape of a square
structure Square :=
  (side : ℝ)
  (area : ℝ := side * side)

-- Define the set of available squares
def available_squares : Finset Square := sorry

-- Define the shapes of the five figures
inductive Figure
| A
| B
| C
| D
| E

-- Function to check if a figure can be formed from the available squares
def can_form_figure (f : Figure) (squares : Finset Square) : Prop := sorry

-- Theorem stating that figure B cannot be formed while others can
theorem figure_b_impossible :
  (∀ s ∈ available_squares, s.side = 1) →
  (available_squares.card = 17) →
  (¬ can_form_figure Figure.B available_squares) ∧
  (can_form_figure Figure.A available_squares) ∧
  (can_form_figure Figure.C available_squares) ∧
  (can_form_figure Figure.D available_squares) ∧
  (can_form_figure Figure.E available_squares) :=
by sorry

end NUMINAMATH_CALUDE_figure_b_impossible_l90_9037


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_half_l90_9070

theorem reciprocal_of_negative_half : ((-1/2)⁻¹ : ℚ) = -2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_half_l90_9070


namespace NUMINAMATH_CALUDE_distance_difference_l90_9078

/-- The rate at which Bjorn bikes in miles per hour -/
def bjorn_rate : ℝ := 12

/-- The rate at which Alberto bikes in miles per hour -/
def alberto_rate : ℝ := 15

/-- The duration of the biking trip in hours -/
def trip_duration : ℝ := 6

/-- The theorem stating the difference in distance traveled between Alberto and Bjorn -/
theorem distance_difference : 
  alberto_rate * trip_duration - bjorn_rate * trip_duration = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l90_9078


namespace NUMINAMATH_CALUDE_regular_polygon_20_sides_l90_9054

-- Define a regular polygon with exterior angle of 18 degrees
structure RegularPolygon where
  sides : ℕ
  exteriorAngle : ℝ
  regular : exteriorAngle = 18

-- Theorem: A regular polygon with exterior angle of 18 degrees has 20 sides
theorem regular_polygon_20_sides (p : RegularPolygon) : p.sides = 20 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_20_sides_l90_9054


namespace NUMINAMATH_CALUDE_work_completion_time_l90_9075

/-- Given a work that can be completed by two workers A and B, this theorem proves
    the number of days B needs to complete the work alone, given certain conditions. -/
theorem work_completion_time (W : ℝ) (h_pos : W > 0) : 
  (∃ (work_A work_B : ℝ),
    -- A can finish the work in 21 days
    21 * work_A = W ∧
    -- B worked for 10 days
    10 * work_B + 
    -- A finished the remaining work in 7 days
    7 * work_A = W) →
  -- B can finish the work in 15 days
  15 * work_B = W :=
by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l90_9075


namespace NUMINAMATH_CALUDE_max_candy_eaten_l90_9042

def board_operation (board : List Nat) : Nat → Nat → List Nat :=
  fun i j => (board.removeNth i).removeNth j ++ [board[i]! + board[j]!]

def candy_eaten (board : List Nat) : Nat → Nat → Nat :=
  fun i j => board[i]! * board[j]!

theorem max_candy_eaten :
  ∃ (operations : List (Nat × Nat)),
    operations.length = 33 ∧
    (operations.foldl
      (fun (acc : List Nat × Nat) (op : Nat × Nat) =>
        (board_operation acc.1 op.1 op.2, acc.2 + candy_eaten acc.1 op.1 op.2))
      (List.replicate 34 1, 0)).2 = 561 :=
sorry

end NUMINAMATH_CALUDE_max_candy_eaten_l90_9042


namespace NUMINAMATH_CALUDE_train_length_calculation_l90_9018

theorem train_length_calculation (speed_kmh : ℝ) (crossing_time : ℝ) : 
  speed_kmh = 18 → crossing_time = 20 → speed_kmh * (1000 / 3600) * crossing_time = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l90_9018


namespace NUMINAMATH_CALUDE_remainder_of_12345678_div_10_l90_9046

theorem remainder_of_12345678_div_10 :
  ∃ q : ℕ, 12345678 = 10 * q + 8 ∧ 8 < 10 := by sorry

end NUMINAMATH_CALUDE_remainder_of_12345678_div_10_l90_9046


namespace NUMINAMATH_CALUDE_square_floor_tiles_l90_9062

theorem square_floor_tiles (n : ℕ) (h : 2 * n - 1 = 25) : n ^ 2 = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l90_9062


namespace NUMINAMATH_CALUDE_faye_earnings_l90_9025

/-- Calculates the total amount earned from selling necklaces -/
def total_earned (bead_count gemstone_count pearl_count crystal_count : ℕ) 
                 (bead_price gemstone_price pearl_price crystal_price : ℕ) : ℕ :=
  bead_count * bead_price + 
  gemstone_count * gemstone_price + 
  pearl_count * pearl_price + 
  crystal_count * crystal_price

/-- Theorem: The total amount Faye earned is $190 -/
theorem faye_earnings : 
  total_earned 3 7 2 5 7 10 12 15 = 190 := by
  sorry

end NUMINAMATH_CALUDE_faye_earnings_l90_9025


namespace NUMINAMATH_CALUDE_loan_principal_is_1200_l90_9063

/-- Calculates the principal amount of a loan given the interest rate, time period, and total interest paid. -/
def calculate_principal (rate : ℚ) (time : ℚ) (interest : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem stating that under the given conditions, the loan principal is $1200. -/
theorem loan_principal_is_1200 :
  let rate : ℚ := 4
  let time : ℚ := rate
  let interest : ℚ := 192
  calculate_principal rate time interest = 1200 := by sorry

end NUMINAMATH_CALUDE_loan_principal_is_1200_l90_9063


namespace NUMINAMATH_CALUDE_exists_checkered_square_l90_9034

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 100 x 100 board -/
def Board := Fin 100 → Fin 100 → Color

/-- Checks if a cell is adjacent to the border -/
def is_border_adjacent (i j : Fin 100) : Prop :=
  i = 0 ∨ i = 99 ∨ j = 0 ∨ j = 99

/-- Checks if a 2x2 square starting at (i, j) is monochrome -/
def is_monochrome (board : Board) (i j : Fin 100) : Prop :=
  ∃ c : Color,
    board i j = c ∧
    board i (j + 1) = c ∧
    board (i + 1) j = c ∧
    board (i + 1) (j + 1) = c

/-- Checks if a 2x2 square starting at (i, j) is checkered -/
def is_checkered (board : Board) (i j : Fin 100) : Prop :=
  (board i j = board (i + 1) (j + 1) ∧
   board i (j + 1) = board (i + 1) j ∧
   board i j ≠ board i (j + 1))

/-- Main theorem -/
theorem exists_checkered_square (board : Board) 
  (h1 : ∀ i j : Fin 100, is_border_adjacent i j → board i j = Color.Black)
  (h2 : ∀ i j : Fin 100, ¬is_monochrome board i j) :
  ∃ i j : Fin 100, is_checkered board i j :=
sorry

end NUMINAMATH_CALUDE_exists_checkered_square_l90_9034


namespace NUMINAMATH_CALUDE_math_exam_total_points_math_exam_points_is_35_l90_9027

/-- The total number of points in a math exam given the scores of three students and the number of mistakes made by one of them. -/
theorem math_exam_total_points (bryan_score : ℕ) (jen_score_diff : ℕ) (sammy_score_diff : ℕ) (sammy_mistakes : ℕ) : ℕ :=
  let jen_score := bryan_score + jen_score_diff
  let sammy_score := jen_score - sammy_score_diff
  sammy_score + sammy_mistakes

/-- Proof that the total number of points in the math exam is 35. -/
theorem math_exam_points_is_35 :
  math_exam_total_points 20 10 2 7 = 35 := by
  sorry

end NUMINAMATH_CALUDE_math_exam_total_points_math_exam_points_is_35_l90_9027


namespace NUMINAMATH_CALUDE_sqrt_16_equals_4_l90_9098

theorem sqrt_16_equals_4 : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_16_equals_4_l90_9098


namespace NUMINAMATH_CALUDE_evaluate_expression_l90_9038

theorem evaluate_expression : (3^2)^4 * 2^3 = 52488 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l90_9038


namespace NUMINAMATH_CALUDE_nba_conference_impossibility_l90_9013

/-- Represents the number of teams in the NBA -/
def total_teams : ℕ := 30

/-- Represents the number of games each team plays in a regular season -/
def games_per_team : ℕ := 82

/-- Represents a potential division of teams into conferences -/
structure Conference_Division where
  eastern : ℕ
  western : ℕ
  sum_teams : eastern + western = total_teams

/-- Represents the condition for inter-conference games -/
def valid_inter_conference_games (d : Conference_Division) : Prop :=
  ∃ (inter_games : ℕ), 
    inter_games * 2 = games_per_team ∧
    d.eastern * inter_games = d.western * inter_games

theorem nba_conference_impossibility : 
  ¬ ∃ (d : Conference_Division), valid_inter_conference_games d :=
sorry

end NUMINAMATH_CALUDE_nba_conference_impossibility_l90_9013


namespace NUMINAMATH_CALUDE_sum_xy_value_l90_9004

theorem sum_xy_value (x y : ℝ) (h1 : x + 2*y = 5) (h2 : (x + y) / 3 = 1.222222222222222) :
  x + y = 3.666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_sum_xy_value_l90_9004


namespace NUMINAMATH_CALUDE_max_cars_in_parking_lot_l90_9014

/-- Represents a parking lot configuration --/
structure ParkingLot :=
  (grid : Fin 7 → Fin 7 → Bool)
  (gate : Fin 7 × Fin 7)

/-- Checks if a car can exit from its position --/
def canExit (lot : ParkingLot) (pos : Fin 7 × Fin 7) : Prop :=
  sorry

/-- Counts the number of cars in the parking lot --/
def carCount (lot : ParkingLot) : Nat :=
  sorry

/-- Checks if the parking lot configuration is valid --/
def isValidConfig (lot : ParkingLot) : Prop :=
  ∀ pos, lot.grid pos.1 pos.2 → canExit lot pos

/-- The main theorem --/
theorem max_cars_in_parking_lot :
  ∃ (lot : ParkingLot),
    isValidConfig lot ∧
    carCount lot = 28 ∧
    ∀ (other : ParkingLot), isValidConfig other → carCount other ≤ 28 :=
  sorry

end NUMINAMATH_CALUDE_max_cars_in_parking_lot_l90_9014


namespace NUMINAMATH_CALUDE_benny_comic_books_l90_9077

theorem benny_comic_books (initial : ℕ) : 
  (initial / 2 + 6 = 17) → initial = 22 := by
  sorry

end NUMINAMATH_CALUDE_benny_comic_books_l90_9077


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l90_9009

theorem complex_magnitude_product : Complex.abs (5 - 3*I) * Complex.abs (5 + 3*I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l90_9009


namespace NUMINAMATH_CALUDE_root_transformation_l90_9033

theorem root_transformation (a₁ a₂ a₃ b c₁ c₂ c₃ : ℝ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₂ ≠ a₃)
  (h_roots : ∀ x, (x - a₁) * (x - a₂) * (x - a₃) = b ↔ x = c₁ ∨ x = c₂ ∨ x = c₃)
  (h_distinct_roots : c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₂ ≠ c₃) :
  ∀ x, (x + c₁) * (x + c₂) * (x + c₃) = b ↔ x = -a₁ ∨ x = -a₂ ∨ x = -a₃ :=
sorry

end NUMINAMATH_CALUDE_root_transformation_l90_9033


namespace NUMINAMATH_CALUDE_power_sum_equality_l90_9055

theorem power_sum_equality : 3 * 3^3 + 9^61 / 9^59 = 162 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l90_9055


namespace NUMINAMATH_CALUDE_marble_probability_l90_9095

theorem marble_probability (red blue green : ℕ) 
  (h_red : red = 4) 
  (h_blue : blue = 3) 
  (h_green : green = 6) : 
  (red + blue : ℚ) / (red + blue + green) = 7 / 13 := by
sorry

end NUMINAMATH_CALUDE_marble_probability_l90_9095


namespace NUMINAMATH_CALUDE_larger_integer_is_48_l90_9022

theorem larger_integer_is_48 (x y : ℤ) : 
  y = 4 * x →                           -- Two integers are in the ratio of 1 to 4
  (x + 12) * 2 = y →                    -- If 12 is added to the smaller number, the ratio becomes 1 to 2
  y = 48 :=                             -- The larger integer is 48
by
  sorry


end NUMINAMATH_CALUDE_larger_integer_is_48_l90_9022


namespace NUMINAMATH_CALUDE_time_saved_weekly_l90_9006

/-- Time saved weekly by eliminating a daily habit -/
theorem time_saved_weekly (search_time complain_time : ℕ) (days_per_week : ℕ) : 
  search_time = 8 → complain_time = 3 → days_per_week = 7 →
  (search_time + complain_time) * days_per_week = 77 :=
by sorry

end NUMINAMATH_CALUDE_time_saved_weekly_l90_9006


namespace NUMINAMATH_CALUDE_cow_count_theorem_l90_9056

/-- The number of cows on a dairy farm -/
def number_of_cows : ℕ := 20

/-- The number of bags of husk eaten by some cows in 20 days -/
def total_bags_eaten : ℕ := 20

/-- The number of bags of husk eaten by one cow in 20 days -/
def bags_per_cow : ℕ := 1

/-- Theorem stating that the number of cows is equal to the total bags eaten divided by the bags eaten per cow -/
theorem cow_count_theorem : number_of_cows = total_bags_eaten / bags_per_cow := by
  sorry

end NUMINAMATH_CALUDE_cow_count_theorem_l90_9056


namespace NUMINAMATH_CALUDE_vector_simplification_l90_9053

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification (A B C : V) : 
  (B - A) - (C - A) + (C - B) = (0 : V) := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l90_9053


namespace NUMINAMATH_CALUDE_tetrahedron_angle_difference_l90_9003

open Real

/-- Represents a tetrahedron -/
structure Tetrahedron where
  /-- The sum of all dihedral angles in the tetrahedron -/
  dihedral_sum : ℝ
  /-- The sum of all trihedral angles in the tetrahedron -/
  trihedral_sum : ℝ

/-- 
Theorem: For any tetrahedron, the difference between the sum of its dihedral angles 
and the sum of its trihedral angles is equal to 4π.
-/
theorem tetrahedron_angle_difference (t : Tetrahedron) : 
  t.dihedral_sum - t.trihedral_sum = 4 * π :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_angle_difference_l90_9003


namespace NUMINAMATH_CALUDE_part_one_part_two_l90_9048

/-- Set A defined as {x | -2 ≤ x ≤ 2} -/
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

/-- Set B defined as {x | 1-m ≤ x ≤ 2m-2} where m is a real number -/
def B (m : ℝ) : Set ℝ := {x | 1-m ≤ x ∧ x ≤ 2*m-2}

/-- Theorem for part (1) -/
theorem part_one (m : ℝ) : A ⊆ B m ∧ A ≠ B m → m ∈ Set.Ici 3 := by sorry

/-- Theorem for part (2) -/
theorem part_two (m : ℝ) : A ∩ B m = B m → m ∈ Set.Iic 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l90_9048


namespace NUMINAMATH_CALUDE_jamies_liquid_limit_l90_9000

/-- Jamie's liquid consumption limit problem -/
theorem jamies_liquid_limit :
  let cup_oz : ℕ := 8  -- A cup is 8 ounces
  let pint_oz : ℕ := 16  -- A pint is 16 ounces
  let milk_consumed : ℕ := cup_oz  -- Jamie had a cup of milk
  let juice_consumed : ℕ := pint_oz  -- Jamie had a pint of grape juice
  let water_limit : ℕ := 8  -- Jamie can drink 8 more ounces before needing the bathroom
  milk_consumed + juice_consumed + water_limit = 32  -- Jamie's total liquid limit
  := by sorry

end NUMINAMATH_CALUDE_jamies_liquid_limit_l90_9000


namespace NUMINAMATH_CALUDE_f_three_zeros_range_l90_9015

/-- The function f(x) = x^2 * exp(x) - a -/
noncomputable def f (x a : ℝ) : ℝ := x^2 * Real.exp x - a

/-- The statement that f has exactly three zeros -/
def has_exactly_three_zeros (a : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧ 
    (f x₁ a = 0 ∧ f x₂ a = 0 ∧ f x₃ a = 0) ∧
    (∀ x : ℝ, f x a = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)

/-- The theorem stating the range of 'a' for which f has exactly three zeros -/
theorem f_three_zeros_range :
  ∀ a : ℝ, has_exactly_three_zeros a ↔ 0 < a ∧ a < 4 / Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_f_three_zeros_range_l90_9015


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l90_9019

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l90_9019


namespace NUMINAMATH_CALUDE_n_balanced_max_size_l90_9085

/-- A set S is n-balanced if:
    1) Any subset of 3 elements contains at least 2 that are connected
    2) Any subset of n elements contains at least 2 that are not connected -/
def IsNBalanced (n : ℕ) (S : Set α) (connected : α → α → Prop) : Prop :=
  n ≠ 0 ∧
  (∀ (a b c : α), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c →
    connected a b ∨ connected b c ∨ connected a c) ∧
  (∀ (T : Set α), T ⊆ S → T.ncard = n →
    ∃ (x y : α), x ∈ T ∧ y ∈ T ∧ x ≠ y ∧ ¬connected x y)

theorem n_balanced_max_size (n : ℕ) (S : Set α) (connected : α → α → Prop) :
  IsNBalanced n S connected → S.ncard ≤ (n - 1) * (n + 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_n_balanced_max_size_l90_9085


namespace NUMINAMATH_CALUDE_largest_package_size_l90_9011

theorem largest_package_size (john_markers alex_markers : ℕ) 
  (h1 : john_markers = 36) (h2 : alex_markers = 60) : 
  Nat.gcd john_markers alex_markers = 12 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l90_9011


namespace NUMINAMATH_CALUDE_set_operations_l90_9071

def M : Set ℝ := {x | 4 * x^2 - 4 * x - 15 > 0}

def N : Set ℝ := {x | (x + 1) / (6 - x) < 0}

theorem set_operations (M N : Set ℝ) :
  (M = {x | 4 * x^2 - 4 * x - 15 > 0}) →
  (N = {x | (x + 1) / (6 - x) < 0}) →
  (M ∪ N = {x | x < -1 ∨ x ≥ 5/2}) ∧
  ((Set.univ \ M) ∩ (Set.univ \ N) = {x | -1 ≤ x ∧ x < 5/2}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l90_9071


namespace NUMINAMATH_CALUDE_road_trip_gas_usage_l90_9029

/-- Calculates the total gallons of gas used on a road trip --/
theorem road_trip_gas_usage
  (highway_miles : ℝ)
  (highway_efficiency : ℝ)
  (city_miles : ℝ)
  (city_efficiency : ℝ)
  (h1 : highway_miles = 210)
  (h2 : highway_efficiency = 35)
  (h3 : city_miles = 54)
  (h4 : city_efficiency = 18) :
  highway_miles / highway_efficiency + city_miles / city_efficiency = 9 :=
by
  sorry

#check road_trip_gas_usage

end NUMINAMATH_CALUDE_road_trip_gas_usage_l90_9029


namespace NUMINAMATH_CALUDE_french_only_students_l90_9093

/-- Given a group of students with the following properties:
  * There are 28 students in total
  * Some students take French
  * 10 students take Spanish
  * 4 students take both French and Spanish
  * 13 students take neither French nor Spanish
  * Students taking both languages are not counted with those taking only French or only Spanish
This theorem proves that exactly 1 student is taking only French. -/
theorem french_only_students (total : ℕ) (spanish : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 28)
  (h_spanish : spanish = 10)
  (h_both : both = 4)
  (h_neither : neither = 13) :
  total - spanish - both - neither = 1 := by
sorry

end NUMINAMATH_CALUDE_french_only_students_l90_9093


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l90_9049

/-- The repeating decimal 0.565656... -/
def repeating_decimal : ℚ :=
  0 + (56 / 100) * (1 / (1 - 1/100))

/-- The target fraction 56/99 -/
def target_fraction : ℚ := 56 / 99

/-- Theorem stating that the repeating decimal 0.565656... is equal to 56/99 -/
theorem repeating_decimal_equals_fraction :
  repeating_decimal = target_fraction := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l90_9049


namespace NUMINAMATH_CALUDE_fraction_problem_l90_9059

theorem fraction_problem (p q x y : ℚ) : 
  p / q = 4 / 5 →
  x / y + (2 * q - p) / (2 * q + p) = 1 →
  x / y = 4 / 7 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l90_9059


namespace NUMINAMATH_CALUDE_min_value_inequality_sum_squared_ratio_inequality_l90_9092

-- Part 1
theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  1 / a + 4 / b ≥ 9 := by sorry

-- Part 2
theorem sum_squared_ratio_inequality (a b c m : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (habc : a + b + c = m) :
  a^2 / b + b^2 / c + c^2 / a ≥ m := by sorry

end NUMINAMATH_CALUDE_min_value_inequality_sum_squared_ratio_inequality_l90_9092


namespace NUMINAMATH_CALUDE_toys_per_day_l90_9030

/-- A factory produces toys according to the following conditions:
  1. The factory produces 4560 toys per week.
  2. The workers work 4 days a week.
  3. The same number of toys is made every day.
-/
def factory_production (toys_per_day : ℕ) : Prop :=
  toys_per_day * 4 = 4560 ∧ toys_per_day > 0

/-- The number of toys produced each day is 1140. -/
theorem toys_per_day : ∃ (n : ℕ), factory_production n ∧ n = 1140 :=
  sorry

end NUMINAMATH_CALUDE_toys_per_day_l90_9030


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l90_9087

theorem quadratic_function_max_value (a b c : ℝ) : 
  (∃ a' ∈ Set.Icc 1 2, ∀ x ∈ Set.Icc 1 2, a' * x^2 + b * x + c ≤ 1) →
  (∀ m : ℝ, 7 * b + 5 * c ≤ m → m ≥ -6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l90_9087


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l90_9024

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 39) : 
  a * b = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l90_9024


namespace NUMINAMATH_CALUDE_function_composition_l90_9090

/-- Given a function f where f(3x) = 3 / (3 + x) for all x > 0, prove that 3f(x) = 27 / (9 + x) -/
theorem function_composition (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 3 / (3 + x)) :
  ∀ x > 0, 3 * f x = 27 / (9 + x) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l90_9090


namespace NUMINAMATH_CALUDE_inscribed_ngon_iff_three_or_four_l90_9057

/-- An ellipse that is not a circle -/
structure NonCircularEllipse where
  -- Add necessary fields to define a non-circular ellipse
  is_not_circle : Bool

/-- A regular n-gon -/
structure RegularNGon (n : ℕ) where
  -- Add necessary fields to define a regular n-gon
  vertices : Fin n → ℝ × ℝ

/-- Predicate to check if a regular n-gon is inscribed in an ellipse -/
def is_inscribed (E : NonCircularEllipse) (n : ℕ) (polygon : RegularNGon n) : Prop :=
  sorry

/-- Theorem: A regular n-gon can be inscribed in a non-circular ellipse if and only if n = 3 or n = 4 -/
theorem inscribed_ngon_iff_three_or_four (E : NonCircularEllipse) (n : ℕ) :
    (∃ (polygon : RegularNGon n), is_inscribed E n polygon) ↔ (n = 3 ∨ n = 4) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_ngon_iff_three_or_four_l90_9057


namespace NUMINAMATH_CALUDE_symmetric_line_y_axis_correct_l90_9001

/-- Given a line with equation ax + by + c = 0, return the equation of the line symmetric to it with respect to the y-axis -/
def symmetricLineYAxis (a b c : ℝ) : ℝ × ℝ × ℝ := (-a, b, c)

theorem symmetric_line_y_axis_correct :
  let original_line := (2, 1, -4)
  let symmetric_line := symmetricLineYAxis 2 1 (-4)
  symmetric_line = (-2, 1, -4) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_y_axis_correct_l90_9001


namespace NUMINAMATH_CALUDE_line_chart_best_for_temperature_l90_9040

/-- Represents different types of charts --/
inductive ChartType
| BarChart
| LineChart
| PieChart

/-- Represents the characteristics a chart can show --/
structure ChartCharacteristics where
  showsAmount : Bool
  showsChangeOverTime : Bool
  showsPartToWhole : Bool

/-- Defines the characteristics of different chart types --/
def chartTypeCharacteristics : ChartType → ChartCharacteristics
| ChartType.BarChart => ⟨true, false, false⟩
| ChartType.LineChart => ⟨true, true, false⟩
| ChartType.PieChart => ⟨false, false, true⟩

/-- Defines what characteristics are needed for temperature representation --/
def temperatureRepresentationNeeds : ChartCharacteristics :=
  ⟨true, true, false⟩

/-- Theorem: Line chart is the most appropriate for representing temperature changes --/
theorem line_chart_best_for_temperature : 
  ∀ (ct : ChartType), 
    (chartTypeCharacteristics ct = temperatureRepresentationNeeds) → 
    (ct = ChartType.LineChart) :=
by sorry

end NUMINAMATH_CALUDE_line_chart_best_for_temperature_l90_9040
