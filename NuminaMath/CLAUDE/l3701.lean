import Mathlib

namespace candy_bar_multiple_l3701_370152

def fred_candy_bars : ℕ := 12
def uncle_bob_extra_candy_bars : ℕ := 6
def jacqueline_percentage : ℚ := 40 / 100
def jacqueline_percentage_amount : ℕ := 120

theorem candy_bar_multiple :
  let uncle_bob_candy_bars := fred_candy_bars + uncle_bob_extra_candy_bars
  let total_fred_uncle_bob := fred_candy_bars + uncle_bob_candy_bars
  let jacqueline_candy_bars := jacqueline_percentage_amount / jacqueline_percentage
  jacqueline_candy_bars / total_fred_uncle_bob = 10 := by
sorry

end candy_bar_multiple_l3701_370152


namespace probability_of_specific_arrangement_l3701_370175

def total_tiles : ℕ := 5
def x_tiles : ℕ := 3
def o_tiles : ℕ := 2

def specific_arrangement : List Char := ['X', 'O', 'X', 'O', 'X']

def probability_of_arrangement : ℚ :=
  1 / (total_tiles.factorial / (x_tiles.factorial * o_tiles.factorial))

theorem probability_of_specific_arrangement :
  probability_of_arrangement = 1 / 10 := by
  sorry

end probability_of_specific_arrangement_l3701_370175


namespace num_arrangements_eq_360_l3701_370168

/-- The number of volunteers --/
def num_volunteers : ℕ := 6

/-- The number of people to be selected --/
def num_selected : ℕ := 4

/-- The number of distinct tasks --/
def num_tasks : ℕ := 4

/-- Theorem stating the number of arrangements --/
theorem num_arrangements_eq_360 : 
  (num_volunteers.factorial) / ((num_volunteers - num_selected).factorial) = 360 :=
sorry

end num_arrangements_eq_360_l3701_370168


namespace function_passes_through_point_l3701_370129

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by sorry

end function_passes_through_point_l3701_370129


namespace measure_15_minutes_with_7_and_11_l3701_370143

/-- Represents an hourglass that measures a specific duration. -/
structure Hourglass where
  duration : ℕ

/-- Represents the state of measuring time with two hourglasses. -/
structure MeasurementState where
  time_elapsed : ℕ
  hourglass1 : Hourglass
  hourglass2 : Hourglass

/-- Checks if it's possible to measure the target time with given hourglasses. -/
def can_measure_time (target : ℕ) (h1 h2 : Hourglass) : Prop :=
  ∃ (state : MeasurementState), state.time_elapsed = target ∧
    state.hourglass1 = h1 ∧ state.hourglass2 = h2

/-- Theorem stating that 15 minutes can be measured using 7-minute and 11-minute hourglasses. -/
theorem measure_15_minutes_with_7_and_11 :
  can_measure_time 15 (Hourglass.mk 7) (Hourglass.mk 11) := by
  sorry


end measure_15_minutes_with_7_and_11_l3701_370143


namespace ribbon_leftover_l3701_370103

theorem ribbon_leftover (total_ribbon : ℕ) (num_gifts : ℕ) (ribbon_per_gift : ℕ) : 
  total_ribbon = 18 → num_gifts = 6 → ribbon_per_gift = 2 →
  total_ribbon - (num_gifts * ribbon_per_gift) = 6 := by
  sorry

end ribbon_leftover_l3701_370103


namespace wire_ratio_proof_l3701_370186

theorem wire_ratio_proof (total_length shorter_length : ℕ) 
  (h1 : total_length = 49)
  (h2 : shorter_length = 14)
  (h3 : shorter_length < total_length) :
  let longer_length := total_length - shorter_length
  (shorter_length : ℚ) / longer_length = 2 / 5 := by
  sorry

end wire_ratio_proof_l3701_370186


namespace delivery_pay_calculation_l3701_370184

/-- The amount paid per delivery for Oula and Tona --/
def amount_per_delivery : ℝ := sorry

/-- The number of deliveries made by Oula --/
def oula_deliveries : ℕ := 96

/-- The number of deliveries made by Tona --/
def tona_deliveries : ℕ := (3 * oula_deliveries) / 4

/-- The difference in pay between Oula and Tona --/
def pay_difference : ℝ := 2400

theorem delivery_pay_calculation :
  amount_per_delivery * (oula_deliveries - tona_deliveries : ℝ) = pay_difference ∧
  amount_per_delivery = 100 := by sorry

end delivery_pay_calculation_l3701_370184


namespace max_full_pikes_l3701_370167

/-- The maximum number of full pikes given initial conditions -/
theorem max_full_pikes (initial_pikes : ℕ) (full_requirement : ℕ) 
  (h1 : initial_pikes = 30)
  (h2 : full_requirement = 3) : 
  ∃ (max_full : ℕ), max_full = 9 ∧ 
  (∀ (n : ℕ), n ≤ initial_pikes - 1 → n * full_requirement ≤ initial_pikes - 1 ↔ n ≤ max_full) :=
sorry

end max_full_pikes_l3701_370167


namespace complete_square_problems_l3701_370176

theorem complete_square_problems :
  (∀ a b : ℝ, a + b = 5 ∧ a * b = 2 → a^2 + b^2 = 21) ∧
  (∀ a b : ℝ, a + b = 10 ∧ a^2 + b^2 = 50^2 → a * b = -1200) :=
by sorry

end complete_square_problems_l3701_370176


namespace square_plus_reciprocal_squared_l3701_370199

theorem square_plus_reciprocal_squared (x : ℝ) (h : x^2 + 1/x^2 = 2) :
  x^4 + 1/x^4 = 2 := by
  sorry

end square_plus_reciprocal_squared_l3701_370199


namespace line_circle_distance_range_l3701_370106

/-- The range of k for which a line y = k(x+2) has at least three points on the circle x^2 + y^2 = 4 at distance 1 from it -/
theorem line_circle_distance_range :
  ∀ k : ℝ,
  (∃ (A B C : ℝ × ℝ),
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    (A.1^2 + A.2^2 = 4) ∧ (B.1^2 + B.2^2 = 4) ∧ (C.1^2 + C.2^2 = 4) ∧
    (|k * (A.1 + 2) - A.2| / Real.sqrt (k^2 + 1) = 1) ∧
    (|k * (B.1 + 2) - B.2| / Real.sqrt (k^2 + 1) = 1) ∧
    (|k * (C.1 + 2) - C.2| / Real.sqrt (k^2 + 1) = 1))
  ↔
  -Real.sqrt 3 / 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3 :=
by sorry

end line_circle_distance_range_l3701_370106


namespace andrew_game_preparation_time_l3701_370120

/-- Represents the time in minutes required to prepare each type of game -/
structure GamePreparationTime where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Represents the number of games of each type to be prepared -/
structure GameCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Calculates the total preparation time for all games -/
def totalPreparationTime (prep : GamePreparationTime) (counts : GameCounts) : ℕ :=
  prep.typeA * counts.typeA + prep.typeB * counts.typeB + prep.typeC * counts.typeC

/-- Theorem: Given the specific game preparation times and counts, the total preparation time is 350 minutes -/
theorem andrew_game_preparation_time :
  let prep : GamePreparationTime := { typeA := 15, typeB := 25, typeC := 30 }
  let counts : GameCounts := { typeA := 5, typeB := 5, typeC := 5 }
  totalPreparationTime prep counts = 350 := by
  sorry

end andrew_game_preparation_time_l3701_370120


namespace min_value_of_F_l3701_370192

/-- The feasible region defined by the given constraints -/
def FeasibleRegion (x₁ x₂ : ℝ) : Prop :=
  2 - 2*x₁ - x₂ ≥ 0 ∧
  2 - x₁ + x₂ ≥ 0 ∧
  5 - x₁ - x₂ ≥ 0 ∧
  x₁ ≥ 0 ∧
  x₂ ≥ 0

/-- The objective function to be minimized -/
def F (x₁ x₂ : ℝ) : ℝ := x₂ - x₁

/-- Theorem stating that the minimum value of F in the feasible region is -2 -/
theorem min_value_of_F :
  ∀ x₁ x₂ : ℝ, FeasibleRegion x₁ x₂ → F x₁ x₂ ≥ -2 :=
by sorry

end min_value_of_F_l3701_370192


namespace perfect_square_trinomial_l3701_370161

theorem perfect_square_trinomial (a b : ℝ) : 9*a^2 - 24*a*b + 16*b^2 = (3*a + 4*b)^2 := by
  sorry

end perfect_square_trinomial_l3701_370161


namespace marys_candy_count_l3701_370157

/-- Given that Megan has 5 pieces of candy, and Mary has 3 times as much candy as Megan
    plus an additional 10 pieces, prove that Mary's total candy is 25 pieces. -/
theorem marys_candy_count (megan_candy : ℕ) (mary_initial_multiplier : ℕ) (mary_additional_candy : ℕ)
  (h1 : megan_candy = 5)
  (h2 : mary_initial_multiplier = 3)
  (h3 : mary_additional_candy = 10) :
  megan_candy * mary_initial_multiplier + mary_additional_candy = 25 := by
  sorry

end marys_candy_count_l3701_370157


namespace missing_number_proof_l3701_370179

theorem missing_number_proof (a b x : ℕ) (h1 : a = 105) (h2 : b = 147) 
  (h3 : a^3 = 21 * x * 15 * b) : x = 25 := by
  sorry

end missing_number_proof_l3701_370179


namespace rectangle_square_probability_l3701_370104

theorem rectangle_square_probability (rectangle_A_area square_B_perimeter : ℝ) :
  rectangle_A_area = 30 →
  square_B_perimeter = 16 →
  let square_B_side := square_B_perimeter / 4
  let square_B_area := square_B_side ^ 2
  let area_difference := rectangle_A_area - square_B_area
  (area_difference / rectangle_A_area) = 7 / 15 :=
by sorry

end rectangle_square_probability_l3701_370104


namespace rectangular_prism_sum_l3701_370182

/-- A rectangular prism with dimensions 3, 4, and 5 units -/
structure RectangularPrism where
  length : ℕ := 3
  width : ℕ := 4
  height : ℕ := 5

/-- The number of edges in a rectangular prism -/
def num_edges (p : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (p : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (p : RectangularPrism) : ℕ := 6

theorem rectangular_prism_sum (p : RectangularPrism) :
  num_edges p + num_vertices p + num_faces p = 26 := by
  sorry

end rectangular_prism_sum_l3701_370182


namespace quadratic_product_property_l3701_370163

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0

/-- The discriminant of a quadratic polynomial -/
def discriminant (p : QuadraticPolynomial) : ℤ :=
  p.b^2 - 4 * p.a * p.c

/-- Predicate for a quadratic polynomial having distinct roots -/
def has_distinct_roots (p : QuadraticPolynomial) : Prop :=
  discriminant p ≠ 0

/-- The product of the roots of a quadratic polynomial -/
def root_product (p : QuadraticPolynomial) : ℚ :=
  (p.c : ℚ) / (p.a : ℚ)

/-- The product of the coefficients of a quadratic polynomial -/
def coeff_product (p : QuadraticPolynomial) : ℤ :=
  p.a * p.b * p.c

theorem quadratic_product_property (p : QuadraticPolynomial) 
  (h_distinct : has_distinct_roots p)
  (h_product : (coeff_product p : ℚ) = root_product p) :
  ∃ (n : ℤ), n < 0 ∧ coeff_product p = n :=
by sorry

end quadratic_product_property_l3701_370163


namespace equation_solution_l3701_370140

theorem equation_solution : ∃ x : ℝ, 
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ∧ 
  x = -9 := by
sorry

end equation_solution_l3701_370140


namespace f_min_value_when_a_is_one_f_inequality_solution_range_l3701_370177

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |x + a|

-- Theorem 1: Minimum value of f when a = 1
theorem f_min_value_when_a_is_one :
  ∃ (min : ℝ), min = 3/2 ∧ ∀ (x : ℝ), f x 1 ≥ min :=
sorry

-- Theorem 2: Range of a for which f(x) < 5/x + a has a solution in [1, 2]
theorem f_inequality_solution_range :
  ∀ (a : ℝ), a > 0 →
    (∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ f x a < 5/x + a) ↔ (11/2 < a ∧ a < 9/2) :=
sorry

end f_min_value_when_a_is_one_f_inequality_solution_range_l3701_370177


namespace max_x_value_l3701_370197

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 6) (sum_prod_eq : x*y + x*z + y*z = 10) :
  x ≤ 2 ∧ ∃ y z, x = 2 ∧ y + z = 4 ∧ x + y + z = 6 ∧ x*y + x*z + y*z = 10 :=
by sorry

end max_x_value_l3701_370197


namespace tony_drives_five_days_a_week_l3701_370181

/-- Represents the problem of determining Tony's work commute frequency --/
def TonysDriving (car_efficiency : ℝ) (round_trip : ℝ) (tank_capacity : ℝ) (gas_price : ℝ) (total_spent : ℝ) (weeks : ℕ) : Prop :=
  let gallons_per_day := round_trip / car_efficiency
  let total_gallons := total_spent / gas_price
  let gallons_per_week := total_gallons / weeks
  gallons_per_week / gallons_per_day = 5

/-- Theorem stating that given the problem conditions, Tony drives to work 5 days a week --/
theorem tony_drives_five_days_a_week :
  TonysDriving 25 50 10 2 80 4 := by
  sorry

end tony_drives_five_days_a_week_l3701_370181


namespace macaroon_weight_l3701_370119

theorem macaroon_weight (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (num_bags : ℕ) :
  total_macaroons = 12 →
  weight_per_macaroon = 5 →
  num_bags = 4 →
  total_macaroons % num_bags = 0 →
  (total_macaroons - total_macaroons / num_bags) * weight_per_macaroon = 45 := by
  sorry

end macaroon_weight_l3701_370119


namespace sum_of_six_l3701_370125

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Properties of the specific arithmetic sequence -/
def special_sequence (seq : ArithmeticSequence) : Prop :=
  seq.a 1 = 2 ∧ seq.S 4 = 20

theorem sum_of_six (seq : ArithmeticSequence) (h : special_sequence seq) : 
  seq.S 6 = 42 := by
  sorry


end sum_of_six_l3701_370125


namespace parabola_with_directrix_y_2_l3701_370165

/-- Represents a parabola in 2D space -/
structure Parabola where
  /-- The equation of the parabola in the form x² = ky, where k is a non-zero real number -/
  equation : ℝ → ℝ → Prop

/-- Represents the directrix of a parabola -/
structure Directrix where
  /-- The y-coordinate of the horizontal directrix -/
  y : ℝ

/-- 
Given a parabola with a horizontal directrix y = 2, 
prove that its standard equation is x² = -8y 
-/
theorem parabola_with_directrix_y_2 (p : Parabola) (d : Directrix) :
  d.y = 2 → p.equation = fun x y ↦ x^2 = -8*y := by
  sorry

end parabola_with_directrix_y_2_l3701_370165


namespace quadratic_symmetry_and_value_l3701_370142

/-- A quadratic function with symmetry around x = 5.5 and p(0) = -4 -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry_and_value (a b c : ℝ) :
  (∀ x, p a b c (5.5 - x) = p a b c (5.5 + x)) →  -- Symmetry around x = 5.5
  p a b c 0 = -4 →                                -- p(0) = -4
  p a b c 11 = -4 :=                              -- Conclusion: p(11) = -4
by sorry

end quadratic_symmetry_and_value_l3701_370142


namespace candidate_probability_l3701_370101

/-- Represents the probability space of job candidates -/
structure CandidateSpace where
  /-- Probability of having intermediate or advanced Excel skills -/
  excel_skills : ℝ
  /-- Probability of having intermediate Excel skills -/
  intermediate_excel : ℝ
  /-- Probability of having advanced Excel skills -/
  advanced_excel : ℝ
  /-- Probability of being willing to work night shifts among those with Excel skills -/
  night_shift_willing : ℝ
  /-- Probability of not being willing to work weekends among those willing to work night shifts -/
  weekend_unwilling : ℝ
  /-- Ensure probabilities are valid -/
  excel_skills_valid : excel_skills = intermediate_excel + advanced_excel
  excel_skills_prob : excel_skills = 0.45
  intermediate_excel_prob : intermediate_excel = 0.25
  advanced_excel_prob : advanced_excel = 0.20
  night_shift_willing_prob : night_shift_willing = 0.32
  weekend_unwilling_prob : weekend_unwilling = 0.60

/-- The main theorem to prove -/
theorem candidate_probability (cs : CandidateSpace) :
  cs.excel_skills * cs.night_shift_willing * cs.weekend_unwilling = 0.0864 := by
  sorry

end candidate_probability_l3701_370101


namespace academy_league_games_l3701_370132

/-- The number of teams in the Academy League -/
def num_teams : ℕ := 8

/-- The number of non-conference games each team plays -/
def non_conference_games : ℕ := 6

/-- Calculates the total number of games in a season for the Academy League -/
def total_games (n : ℕ) (nc : ℕ) : ℕ :=
  (n * (n - 1)) + (n * nc)

/-- Theorem stating that the total number of games in the Academy League season is 104 -/
theorem academy_league_games :
  total_games num_teams non_conference_games = 104 := by
  sorry

end academy_league_games_l3701_370132


namespace ratio_six_three_percent_l3701_370107

/-- Expresses a ratio as a percentage -/
def ratioToPercent (a b : ℕ) : ℚ :=
  (a : ℚ) / (b : ℚ) * 100

/-- The ratio 6:3 expressed as a percent is 200% -/
theorem ratio_six_three_percent : ratioToPercent 6 3 = 200 := by
  sorry

end ratio_six_three_percent_l3701_370107


namespace bicycle_price_after_discounts_l3701_370193

def original_price : ℝ := 200
def first_discount : ℝ := 0.4
def second_discount : ℝ := 0.25

theorem bicycle_price_after_discounts :
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price = 90 := by sorry

end bicycle_price_after_discounts_l3701_370193


namespace cubic_greater_than_quadratic_plus_one_l3701_370115

theorem cubic_greater_than_quadratic_plus_one (x : ℝ) (h : x > 1) : 2 * x^3 > x^2 + 1 := by
  sorry

end cubic_greater_than_quadratic_plus_one_l3701_370115


namespace arithmetic_sequence_common_difference_l3701_370172

/-- An arithmetic sequence with first term 2 and the sum of the 3rd and 5th terms equal to 10 has a common difference of 1. -/
theorem arithmetic_sequence_common_difference : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  a 1 = 2 →                            -- first term is 2
  a 3 + a 5 = 10 →                     -- sum of 3rd and 5th terms is 10
  a 2 - a 1 = 1 :=                     -- common difference is 1
by
  sorry

end arithmetic_sequence_common_difference_l3701_370172


namespace trigonometric_identity_l3701_370154

open Real

theorem trigonometric_identity (x : ℝ) : 
  sin x * cos x + sin x^3 * cos x + sin x^5 * (1 / cos x) = tan x := by
  sorry

end trigonometric_identity_l3701_370154


namespace system_solution_l3701_370137

theorem system_solution (k : ℚ) (x y z : ℚ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + k*y + 4*z = 0 →
  4*x + k*y - 3*z = 0 →
  3*x + 5*y - 4*z = 0 →
  x*z / (y^2) = 147/28 := by
sorry

end system_solution_l3701_370137


namespace min_shift_for_symmetry_l3701_370198

open Real

theorem min_shift_for_symmetry (φ : ℝ) : 
  φ > 0 ∧ 
  (∀ x, sin (2 * (x - φ)) = sin (2 * (π / 3 - x))) →
  φ ≥ 5 * π / 12 :=
sorry

end min_shift_for_symmetry_l3701_370198


namespace total_leaves_l3701_370145

theorem total_leaves (initial_leaves additional_leaves : ℝ) :
  initial_leaves + additional_leaves = initial_leaves + additional_leaves :=
by sorry

end total_leaves_l3701_370145


namespace pool_filling_rates_l3701_370144

theorem pool_filling_rates (r₁ r₂ r₃ : ℝ) 
  (h1 : r₁ + r₂ = 1 / 70)
  (h2 : r₁ + r₃ = 1 / 84)
  (h3 : r₂ + r₃ = 1 / 140) :
  r₁ = 1 / 105 ∧ r₂ = 1 / 210 ∧ r₃ = 1 / 420 ∧ r₁ + r₂ + r₃ = 1 / 60 := by
  sorry

end pool_filling_rates_l3701_370144


namespace no_error_in_calculation_l3701_370169

theorem no_error_in_calculation : 
  (7 * 4) / (5/3) = (7 * 4) * (3/5) := by
  sorry

#eval (7 * 4) / (5/3) -- To verify the result
#eval (7 * 4) * (3/5) -- To verify the result

end no_error_in_calculation_l3701_370169


namespace cubic_factorization_l3701_370114

theorem cubic_factorization (a : ℝ) : a^3 - 4*a = a*(a+2)*(a-2) := by
  sorry

end cubic_factorization_l3701_370114


namespace spider_web_paths_spider_web_problem_l3701_370194

theorem spider_web_paths : Nat → Nat → Nat
  | m, n => Nat.choose (m + n) m

theorem spider_web_problem : spider_web_paths 5 6 = 462 := by
  sorry

end spider_web_paths_spider_web_problem_l3701_370194


namespace total_three_digit_numbers_l3701_370108

/-- Represents a card with two numbers -/
structure Card where
  side1 : Nat
  side2 : Nat
  different : side1 ≠ side2

/-- The set of cards given in the problem -/
def problemCards : Finset Card := sorry

/-- The number of ways to arrange 3 cards -/
def cardArrangements : Nat := sorry

/-- The number of ways to choose sides for 3 cards -/
def sideChoices : Nat := sorry

/-- Theorem stating the total number of different three-digit numbers -/
theorem total_three_digit_numbers : 
  cardArrangements * sideChoices = 48 := by sorry

end total_three_digit_numbers_l3701_370108


namespace max_small_packages_with_nine_large_l3701_370188

/-- Represents the weight capacity of a service lift -/
structure LiftCapacity where
  large_packages : ℕ
  small_packages : ℕ

/-- Calculates the maximum number of small packages that can be carried alongside a given number of large packages -/
def max_small_packages (capacity : LiftCapacity) (large_count : ℕ) : ℕ :=
  let large_weight := capacity.small_packages / capacity.large_packages
  let remaining_capacity := capacity.small_packages - large_count * large_weight
  remaining_capacity

/-- Theorem: Given a lift with capacity of 12 large packages or 20 small packages,
    the maximum number of small packages that can be carried alongside 9 large packages is 5 -/
theorem max_small_packages_with_nine_large :
  let capacity := LiftCapacity.mk 12 20
  max_small_packages capacity 9 = 5 := by
  sorry

end max_small_packages_with_nine_large_l3701_370188


namespace tangent_line_slope_l3701_370126

/-- The curve function f(x) = x³ + x + 16 -/
def f (x : ℝ) : ℝ := x^3 + x + 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_line_slope :
  ∃ a : ℝ,
    (f a = (f' a) * a) ∧  -- Point (a, f(a)) lies on the tangent line
    (f' a = 13) -- The slope of the tangent line is 13
  := by sorry

end tangent_line_slope_l3701_370126


namespace ice_cream_scoops_prove_ice_cream_scoops_l3701_370123

-- Define the given conditions
def aaron_savings : ℚ := 40
def carson_savings : ℚ := 40
def total_savings : ℚ := aaron_savings + carson_savings
def dinner_bill_ratio : ℚ := 3 / 4
def scoop_cost : ℚ := 3 / 2
def change_per_person : ℚ := 1

-- Define the theorem
theorem ice_cream_scoops : ℚ :=
  let dinner_bill := dinner_bill_ratio * total_savings
  let remaining_after_dinner := total_savings - dinner_bill
  let ice_cream_spending := remaining_after_dinner - 2 * change_per_person
  let total_scoops := ice_cream_spending / scoop_cost
  total_scoops / 2

-- The theorem to prove
theorem prove_ice_cream_scoops : ice_cream_scoops = 6 := by
  sorry

end ice_cream_scoops_prove_ice_cream_scoops_l3701_370123


namespace ternary_121_eq_decimal_16_l3701_370111

/-- Converts a ternary (base 3) number to decimal (base 10) --/
def ternary_to_decimal (t₂ t₁ t₀ : ℕ) : ℕ :=
  t₂ * 3^2 + t₁ * 3^1 + t₀ * 3^0

/-- Proves that the ternary number 121₃ is equal to the decimal number 16 --/
theorem ternary_121_eq_decimal_16 : ternary_to_decimal 1 2 1 = 16 := by
  sorry

end ternary_121_eq_decimal_16_l3701_370111


namespace bus_seats_l3701_370105

theorem bus_seats (west_lake : Nat) (east_lake : Nat)
  (h1 : west_lake = 138)
  (h2 : east_lake = 115)
  (h3 : ∀ x : Nat, x > 1 ∧ x ∣ west_lake ∧ x ∣ east_lake → x ≤ 23) :
  23 > 1 ∧ 23 ∣ west_lake ∧ 23 ∣ east_lake :=
by sorry

#check bus_seats

end bus_seats_l3701_370105


namespace union_and_intersection_when_m_2_intersection_empty_iff_l3701_370112

def A : Set ℝ := {x | -3 < x ∧ x < 4}
def B (m : ℝ) : Set ℝ := {x | m - 1 < x ∧ x < 3 * m + 3}

theorem union_and_intersection_when_m_2 :
  (A ∪ B 2 = {x | -3 < x ∧ x < 9}) ∧
  (A ∩ (Set.univ \ B 2) = {x | -3 < x ∧ x ≤ 1}) := by sorry

theorem intersection_empty_iff :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m ≥ 5 ∨ m ≤ -2 := by sorry

end union_and_intersection_when_m_2_intersection_empty_iff_l3701_370112


namespace grade_10_sample_size_l3701_370130

/-- Represents the ratio of students in grades 10, 11, and 12 -/
def grade_ratio : Fin 3 → ℕ
  | 0 => 2  -- Grade 10
  | 1 => 2  -- Grade 11
  | 2 => 1  -- Grade 12

/-- Total sample size -/
def sample_size : ℕ := 45

/-- Calculates the number of students sampled from a specific grade -/
def students_sampled (grade : Fin 3) : ℕ :=
  (sample_size * grade_ratio grade) / (grade_ratio 0 + grade_ratio 1 + grade_ratio 2)

/-- Theorem stating that the number of grade 10 students in the sample is 18 -/
theorem grade_10_sample_size :
  students_sampled 0 = 18 := by sorry

end grade_10_sample_size_l3701_370130


namespace triangle_properties_l3701_370100

theorem triangle_properties (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Acute triangle condition
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C) →  -- Sine rule
  (Real.sin A + Real.sin B)^2 = (2 * Real.sin B + Real.sin C) * Real.sin C →  -- Given equation
  Real.sin A > Real.sqrt 3 / 3 →  -- Given inequality
  (c - a = a * Real.cos C) ∧ (c > a) ∧ (C > π / 3) := by
  sorry

end triangle_properties_l3701_370100


namespace initial_bacteria_count_l3701_370124

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The doubling period of bacteria in seconds -/
def doubling_period : ℕ := 30

/-- The duration of the experiment in minutes -/
def experiment_duration : ℕ := 4

/-- The number of bacteria after the experiment -/
def final_bacteria_count : ℕ := 65536

/-- The number of doubling periods in the experiment -/
def doubling_periods : ℕ := (experiment_duration * seconds_per_minute) / doubling_period

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := final_bacteria_count / (2 ^ doubling_periods)

theorem initial_bacteria_count : initial_bacteria = 256 := by
  sorry

end initial_bacteria_count_l3701_370124


namespace candy_mixture_price_l3701_370173

/-- Calculates the selling price per pound of a candy mixture -/
theorem candy_mixture_price (total_weight : ℝ) (cheap_weight : ℝ) (cheap_price : ℝ) (expensive_price : ℝ)
  (h1 : total_weight = 80)
  (h2 : cheap_weight = 64)
  (h3 : cheap_price = 2)
  (h4 : expensive_price = 3)
  : (cheap_weight * cheap_price + (total_weight - cheap_weight) * expensive_price) / total_weight = 2.20 := by
  sorry

end candy_mixture_price_l3701_370173


namespace second_half_speed_l3701_370102

theorem second_half_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (first_half_speed : ℝ) 
  (h1 : total_distance = 400) 
  (h2 : total_time = 30) 
  (h3 : first_half_speed = 20) : 
  (total_distance / 2) / (total_time - (total_distance / 2) / first_half_speed) = 10 :=
sorry

end second_half_speed_l3701_370102


namespace rationalize_denominator_l3701_370139

theorem rationalize_denominator (x : ℝ) (h : x^4 = 81) :
  1 / (x + x^(1/4)) = 1 / 27 :=
sorry

end rationalize_denominator_l3701_370139


namespace bicycling_time_l3701_370147

-- Define the distance in kilometers
def distance : ℝ := 96

-- Define the speed in kilometers per hour
def speed : ℝ := 6

-- Theorem: The time taken is 16 hours
theorem bicycling_time : distance / speed = 16 := by
  sorry

end bicycling_time_l3701_370147


namespace factorization_sum_l3701_370151

def P (y : ℤ) : ℤ := y^6 - y^3 - 2*y - 2

def is_irreducible_factor (q : ℤ → ℤ) : Prop :=
  (∀ y, q y ∣ P y) ∧ 
  (∀ f g : ℤ → ℤ, (∀ y, q y = f y * g y) → (∀ y, f y = 1 ∨ g y = 1))

theorem factorization_sum (q₁ q₂ q₃ q₄ : ℤ → ℤ) :
  is_irreducible_factor q₁ ∧
  is_irreducible_factor q₂ ∧
  is_irreducible_factor q₃ ∧
  is_irreducible_factor q₄ ∧
  (∀ y, P y = q₁ y * q₂ y * q₃ y * q₄ y) →
  q₁ 3 + q₂ 3 + q₃ 3 + q₄ 3 = 30 := by
  sorry

end factorization_sum_l3701_370151


namespace cannot_determine_f_triple_prime_l3701_370121

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- State the theorem
theorem cannot_determine_f_triple_prime (a b c : ℝ) :
  (∃ x, f a b c x = a * x^4 + b * x^2 + c) →
  ((12 * a + 2 * b) = 2) →
  ¬ (∃! y, (24 * a * (-1) = y)) :=
by sorry

end cannot_determine_f_triple_prime_l3701_370121


namespace reciprocal_and_fraction_operations_l3701_370170

theorem reciprocal_and_fraction_operations :
  (∀ a b c : ℚ, (a + b) / c = -2 → c / (a + b) = -1/2) ∧
  (5/12 - 1/9 + 2/3) / (1/36) = 35 ∧
  (-1/36) / (5/12 - 1/9 + 2/3) = -1/35 := by
  sorry

end reciprocal_and_fraction_operations_l3701_370170


namespace line_equation_through_point_with_slope_l3701_370187

/-- A line passing through the point (2, 1) with a slope of 2 has the equation 2x - y - 3 = 0. -/
theorem line_equation_through_point_with_slope (x y : ℝ) :
  (2 : ℝ) * x - y - 3 = 0 ↔ (y - 1 = 2 * (x - 2)) := by
  sorry

end line_equation_through_point_with_slope_l3701_370187


namespace investment_solution_l3701_370135

def investment_problem (x y r1 r2 total_investment desired_interest : ℝ) : Prop :=
  x + y = total_investment ∧
  r1 * x + r2 * y = desired_interest

theorem investment_solution :
  investment_problem 6000 4000 0.09 0.11 10000 980 := by
  sorry

end investment_solution_l3701_370135


namespace ellipse_right_triangle_distance_l3701_370131

def Ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

def LeftFocus (x y : ℝ) : Prop := x = -Real.sqrt 7 ∧ y = 0
def RightFocus (x y : ℝ) : Prop := x = Real.sqrt 7 ∧ y = 0

def RightTriangle (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (x₂ - x₁) * (x₃ - x₁) + (y₂ - y₁) * (y₃ - y₁) = 0 ∨
  (x₁ - x₂) * (x₃ - x₂) + (y₁ - y₂) * (y₃ - y₂) = 0 ∨
  (x₁ - x₃) * (x₂ - x₃) + (y₁ - y₃) * (y₂ - y₃) = 0

theorem ellipse_right_triangle_distance (x y xf₁ yf₁ xf₂ yf₂ : ℝ) :
  Ellipse x y →
  LeftFocus xf₁ yf₁ →
  RightFocus xf₂ yf₂ →
  RightTriangle x y xf₁ yf₁ xf₂ yf₂ →
  |y| = 9/4 :=
sorry

end ellipse_right_triangle_distance_l3701_370131


namespace negative_two_classification_l3701_370110

theorem negative_two_classification :
  (∃ (n : ℤ), n = -2) →  -- -2 is an integer
  (∃ (q : ℚ), q = -2 ∧ q < 0)  -- -2 is a negative rational number
:= by sorry

end negative_two_classification_l3701_370110


namespace garden_length_l3701_370153

theorem garden_length (width : ℝ) (length : ℝ) : 
  width > 0 →
  length = 2 * width →
  2 * length + 2 * width = 240 →
  length = 80 :=
by
  sorry

end garden_length_l3701_370153


namespace pencil_count_l3701_370195

theorem pencil_count (num_pens : ℕ) (max_students : ℕ) (num_pencils : ℕ) :
  num_pens = 1001 →
  max_students = 91 →
  (∃ (s : ℕ), s ≤ max_students ∧ num_pens % s = 0 ∧ num_pencils % s = 0) →
  ∃ (k : ℕ), num_pencils = 91 * k :=
by sorry

end pencil_count_l3701_370195


namespace combined_tax_rate_l3701_370116

/-- Given two individuals with different tax rates and income levels, 
    calculate their combined tax rate -/
theorem combined_tax_rate 
  (mork_rate : ℚ) 
  (mindy_rate : ℚ) 
  (income_ratio : ℚ) : 
  mork_rate = 45/100 → 
  mindy_rate = 25/100 → 
  income_ratio = 4 → 
  (mork_rate + income_ratio * mindy_rate) / (1 + income_ratio) = 29/100 :=
by sorry

end combined_tax_rate_l3701_370116


namespace angle_DEB_value_l3701_370190

-- Define the geometric configuration
structure GeometricConfig where
  -- Triangle ABC
  angleABC : ℝ
  angleACB : ℝ
  -- Other angles
  angleCDE : ℝ
  -- Straight line and angle conditions
  angleADC_straight : angleADC = 180
  angleAEB_straight : angleAEB = 180
  -- Given conditions
  h1 : angleABC = 72
  h2 : angleACB = 90
  h3 : angleCDE = 36

-- Theorem statement
theorem angle_DEB_value (config : GeometricConfig) : ∃ (angleDEB : ℝ), angleDEB = 162 := by
  sorry


end angle_DEB_value_l3701_370190


namespace arithmetic_sequence_300th_term_l3701_370133

/-- 
Given an arithmetic sequence where:
- The first term is 6
- The common difference is 4
Prove that the 300th term is equal to 1202
-/
theorem arithmetic_sequence_300th_term : 
  let a : ℕ → ℕ := λ n => 6 + (n - 1) * 4
  a 300 = 1202 := by
  sorry

end arithmetic_sequence_300th_term_l3701_370133


namespace interest_calculation_l3701_370122

theorem interest_calculation (initial_investment second_investment second_interest : ℝ) 
  (h1 : initial_investment = 5000)
  (h2 : second_investment = 20000)
  (h3 : second_interest = 1000)
  (h4 : second_interest = second_investment * (second_interest / second_investment))
  (h5 : initial_investment > 0)
  (h6 : second_investment > 0) :
  initial_investment * (second_interest / second_investment) = 250 := by
sorry

end interest_calculation_l3701_370122


namespace second_stop_count_l3701_370150

/-- The number of students who got on the bus during the first stop -/
def first_stop_students : ℕ := 39

/-- The total number of students on the bus after the second stop -/
def total_students : ℕ := 68

/-- The number of students who got on the bus during the second stop -/
def second_stop_students : ℕ := total_students - first_stop_students

theorem second_stop_count : second_stop_students = 29 := by
  sorry

end second_stop_count_l3701_370150


namespace lisa_candy_weeks_l3701_370149

/-- The number of weeks it takes Lisa to eat all her candies -/
def weeks_to_eat_candies (total_candies : ℕ) (candies_mon_wed : ℕ) (candies_other_days : ℕ) : ℕ :=
  total_candies / (2 * candies_mon_wed + 5 * candies_other_days)

/-- Theorem stating that it takes 4 weeks for Lisa to eat all her candies -/
theorem lisa_candy_weeks : weeks_to_eat_candies 36 2 1 = 4 := by
  sorry

end lisa_candy_weeks_l3701_370149


namespace letter_lock_unsuccessful_attempts_l3701_370141

/-- Represents a letter lock with a given number of rings and letters per ring -/
structure LetterLock where
  num_rings : ℕ
  letters_per_ring : ℕ

/-- Calculates the maximum number of unsuccessful attempts for a given lock -/
def max_unsuccessful_attempts (lock : LetterLock) : ℕ :=
  lock.letters_per_ring ^ lock.num_rings - 1

/-- Theorem stating that a lock with 5 rings and 10 letters per ring has 99,999 unsuccessful attempts -/
theorem letter_lock_unsuccessful_attempts :
  let lock : LetterLock := { num_rings := 5, letters_per_ring := 10 }
  max_unsuccessful_attempts lock = 99999 := by
  sorry

#eval max_unsuccessful_attempts { num_rings := 5, letters_per_ring := 10 }

end letter_lock_unsuccessful_attempts_l3701_370141


namespace simplify_expression_l3701_370156

theorem simplify_expression (a b : ℝ) : 2*a*(2*a^2 + a*b) - a^2*b = 4*a^3 + a^2*b := by
  sorry

end simplify_expression_l3701_370156


namespace angle_inequality_l3701_370128

theorem angle_inequality (θ : Real) : 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.sin θ - x * (1 - x) + (1 - x)^2 * Real.cos θ > 0) ↔ 
  (π / 12 < θ ∧ θ < 5 * π / 12) := by
sorry

end angle_inequality_l3701_370128


namespace last_remaining_is_c_implies_start_is_f_l3701_370171

/-- Represents the children in the circle -/
inductive Child : Type
| a | b | c | d | e | f

/-- The number of children in the circle -/
def numChildren : Nat := 6

/-- The number of words in the song -/
def songWords : Nat := 9

/-- Function to determine the last remaining child given a starting position -/
def lastRemaining (start : Child) : Child :=
  sorry

/-- Theorem stating that if c is the last remaining child, the starting position must be f -/
theorem last_remaining_is_c_implies_start_is_f :
  lastRemaining Child.f = Child.c :=
sorry

end last_remaining_is_c_implies_start_is_f_l3701_370171


namespace percentage_relationship_l3701_370178

theorem percentage_relationship (x y : ℝ) (h : x = y * (1 - 0.4117647058823529)) :
  y = x * (1 + 0.7) := by
sorry

end percentage_relationship_l3701_370178


namespace min_workers_team_a_l3701_370127

theorem min_workers_team_a (a b : ℕ) : 
  (∃ c : ℕ, c > 0 ∧ 2 * (a - 90) = b + 90 ∧ a + c = 6 * (b - c)) →
  a ≥ 153 :=
by sorry

end min_workers_team_a_l3701_370127


namespace grandma_olga_grandchildren_l3701_370191

-- Define the number of daughters and sons
def num_daughters : ℕ := 3
def num_sons : ℕ := 3

-- Define the number of children for each daughter and son
def sons_per_daughter : ℕ := 6
def daughters_per_son : ℕ := 5

-- Define the total number of grandchildren
def total_grandchildren : ℕ := num_daughters * sons_per_daughter + num_sons * daughters_per_son

-- Theorem statement
theorem grandma_olga_grandchildren : total_grandchildren = 33 := by
  sorry

end grandma_olga_grandchildren_l3701_370191


namespace left_handed_or_throwers_count_l3701_370160

/-- Represents a football team with specific player distributions -/
structure FootballTeam where
  total_players : Nat
  throwers : Nat
  left_handed : Nat
  right_handed : Nat

/-- Calculates the number of players who are either left-handed or throwers -/
def left_handed_or_throwers (team : FootballTeam) : Nat :=
  team.left_handed + team.throwers

/-- Theorem stating the number of players who are either left-handed or throwers in the given scenario -/
theorem left_handed_or_throwers_count (team : FootballTeam) :
  team.total_players = 70 →
  team.throwers = 34 →
  team.left_handed = (team.total_players - team.throwers) / 3 →
  team.right_handed = team.total_players - team.throwers - team.left_handed + team.throwers →
  left_handed_or_throwers team = 46 := by
  sorry

end left_handed_or_throwers_count_l3701_370160


namespace angle_A_value_l3701_370196

theorem angle_A_value (A : Real) (h1 : 0 < A ∧ A < π / 2) (h2 : Real.tan A = 1) : A = π / 4 := by
  sorry

end angle_A_value_l3701_370196


namespace inequality_equivalence_l3701_370162

theorem inequality_equivalence (x : ℝ) : (x + 1) / 2 ≥ x / 3 ↔ x ≥ -3 := by
  sorry

end inequality_equivalence_l3701_370162


namespace intersection_point_unique_intersection_point_correct_l3701_370117

/-- The line equation -/
def line (x y z : ℝ) : Prop :=
  (x - 7) / 3 = (y - 3) / 1 ∧ (y - 3) / 1 = (z + 1) / (-2)

/-- The plane equation -/
def plane (x y z : ℝ) : Prop :=
  2 * x + y + 7 * z - 3 = 0

/-- The intersection point -/
def intersection_point : ℝ × ℝ × ℝ := (10, 4, -3)

theorem intersection_point_unique :
  ∃! p : ℝ × ℝ × ℝ, line p.1 p.2.1 p.2.2 ∧ plane p.1 p.2.1 p.2.2 :=
by
  sorry

theorem intersection_point_correct :
  line intersection_point.1 intersection_point.2.1 intersection_point.2.2 ∧
  plane intersection_point.1 intersection_point.2.1 intersection_point.2.2 :=
by
  sorry

end intersection_point_unique_intersection_point_correct_l3701_370117


namespace einstein_fundraising_l3701_370155

/-- Einstein's fundraising problem -/
theorem einstein_fundraising 
  (goal : ℕ)
  (pizza_price potato_price soda_price : ℚ)
  (pizza_sold potato_sold soda_sold : ℕ) :
  goal = 500 ∧ 
  pizza_price = 12 ∧ 
  potato_price = 3/10 ∧ 
  soda_price = 2 ∧
  pizza_sold = 15 ∧ 
  potato_sold = 40 ∧ 
  soda_sold = 25 →
  (goal : ℚ) - (pizza_price * pizza_sold + potato_price * potato_sold + soda_price * soda_sold) = 258 :=
by sorry


end einstein_fundraising_l3701_370155


namespace arrangements_part1_arrangements_part2_arrangements_part3_arrangements_part4_arrangements_part5_arrangements_part6_l3701_370138

/- Given: 3 male students and 4 female students -/
def num_male : ℕ := 3
def num_female : ℕ := 4
def total_students : ℕ := num_male + num_female

/- Part 1: Select 5 people and arrange them in a row -/
theorem arrangements_part1 : (Nat.choose total_students 5) * (Nat.factorial 5) = 2520 := by sorry

/- Part 2: Arrange them in two rows, with 3 in the front row and 4 in the back row -/
theorem arrangements_part2 : (Nat.factorial 7) * (Nat.factorial 6) * (Nat.factorial 5) = 5040 := by sorry

/- Part 3: Arrange all of them in a row, with a specific person not standing at the head or tail of the row -/
theorem arrangements_part3 : (Nat.factorial 6) * 5 = 3600 := by sorry

/- Part 4: Arrange all of them in a row, with all female students standing together -/
theorem arrangements_part4 : (Nat.factorial 4) * (Nat.factorial 4) = 576 := by sorry

/- Part 5: Arrange all of them in a row, with male students not standing next to each other -/
theorem arrangements_part5 : (Nat.factorial 4) * (Nat.factorial 5) * (Nat.factorial 3) = 1440 := by sorry

/- Part 6: Arrange all of them in a row, with exactly 3 people between person A and person B -/
theorem arrangements_part6 : (Nat.factorial 5) * 2 * (Nat.factorial 2) = 720 := by sorry

end arrangements_part1_arrangements_part2_arrangements_part3_arrangements_part4_arrangements_part5_arrangements_part6_l3701_370138


namespace max_cards_saved_is_34_l3701_370134

/-- The set of digits that remain valid when flipped upside down -/
def valid_digits : Finset Nat := {1, 6, 8, 9}

/-- The set of digits that can be used in the tens place -/
def tens_digits : Finset Nat := {0, 1, 6, 8, 9}

/-- The total number of three-digit numbers -/
def total_numbers : Nat := 900

/-- The number of valid reversible three-digit numbers -/
def reversible_numbers : Nat := valid_digits.card * tens_digits.card * valid_digits.card

/-- The number of palindromic reversible numbers -/
def palindromic_numbers : Nat := valid_digits.card * 3

/-- The maximum number of cards that can be saved -/
def max_cards_saved : Nat := (reversible_numbers - palindromic_numbers) / 2

theorem max_cards_saved_is_34 : max_cards_saved = 34 := by
  sorry

#eval max_cards_saved

end max_cards_saved_is_34_l3701_370134


namespace log_expression_equals_two_l3701_370113

-- Define the common logarithm (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  (log10 5)^2 + log10 2 * log10 5 + log10 20 = 2 := by sorry

end log_expression_equals_two_l3701_370113


namespace teachers_not_picking_square_l3701_370180

theorem teachers_not_picking_square (total_teachers : ℕ) (square_teachers : ℕ) 
  (h1 : total_teachers = 20) 
  (h2 : square_teachers = 7) : 
  total_teachers - square_teachers = 13 := by
  sorry

end teachers_not_picking_square_l3701_370180


namespace oblique_drawing_parallelogram_oblique_drawing_other_shapes_l3701_370189

/-- Represents a shape in 2D space -/
inductive Shape
  | Triangle
  | Parallelogram
  | Square
  | Rhombus

/-- Represents the result of applying the oblique drawing method to a shape -/
def obliqueDrawing (s : Shape) : Shape :=
  match s with
  | Shape.Parallelogram => Shape.Parallelogram
  | _ => Shape.Parallelogram  -- Simplified for this problem

/-- Theorem stating that the oblique drawing of a parallelogram is always a parallelogram -/
theorem oblique_drawing_parallelogram :
  ∀ s : Shape, s = Shape.Parallelogram → obliqueDrawing s = Shape.Parallelogram :=
by sorry

/-- Theorem stating that the oblique drawing of non-parallelogram shapes may not preserve the original shape -/
theorem oblique_drawing_other_shapes :
  ∃ s : Shape, s ≠ Shape.Parallelogram ∧ obliqueDrawing s ≠ s :=
by sorry

end oblique_drawing_parallelogram_oblique_drawing_other_shapes_l3701_370189


namespace exact_fare_payment_l3701_370166

/-- The bus fare in kopecks -/
def busFare : ℕ := 5

/-- The smallest coin denomination in kopecks -/
def smallestCoin : ℕ := 10

/-- The number of passengers is always a multiple of 4 -/
def numPassengers (k : ℕ) : ℕ := 4 * k

/-- The minimum number of coins required for exact fare payment -/
def minCoins (k : ℕ) : ℕ := 5 * k

theorem exact_fare_payment (k : ℕ) (h : k > 0) :
  ∀ (n : ℕ), n < minCoins k → ¬∃ (coins : List ℕ),
    (∀ c ∈ coins, c ≥ smallestCoin) ∧
    coins.length = n ∧
    coins.sum = busFare * numPassengers k :=
  sorry

#check exact_fare_payment

end exact_fare_payment_l3701_370166


namespace max_probability_at_twenty_l3701_370185

-- Define the total number of bulbs
def total_bulbs : ℕ := 100

-- Define the number of bulbs picked
def bulbs_picked : ℕ := 10

-- Define the number of defective bulbs in the picked sample
def defective_in_sample : ℕ := 2

-- Define the probability function f(n)
def f (n : ℕ) : ℚ :=
  (Nat.choose n defective_in_sample * Nat.choose (total_bulbs - n) (bulbs_picked - defective_in_sample)) /
  Nat.choose total_bulbs bulbs_picked

-- State the theorem
theorem max_probability_at_twenty {n : ℕ} (h1 : 2 ≤ n) (h2 : n ≤ 92) :
  ∀ m : ℕ, 2 ≤ m ∧ m ≤ 92 → f n ≤ f 20 :=
sorry

end max_probability_at_twenty_l3701_370185


namespace parallel_perpendicular_relation_l3701_370158

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (line_perpendicular : Line → Line → Prop)

-- Theorem statement
theorem parallel_perpendicular_relation 
  (m n : Line) (α β : Plane) :
  (parallel m α ∧ parallel n β ∧ perpendicular α β) → 
  (line_perpendicular m n ∨ line_parallel m n) = False := by
sorry

end parallel_perpendicular_relation_l3701_370158


namespace right_triangle_leg_l3701_370183

theorem right_triangle_leg (h : Real) (angle : Real) :
  angle = Real.pi / 4 →
  h = 10 * Real.sqrt 2 →
  h * Real.sin angle = 10 := by
  sorry

end right_triangle_leg_l3701_370183


namespace max_value_polynomial_l3701_370159

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  (∃ (max : ℝ), ∀ (a b : ℝ), a + b = 5 →
    a^5*b + a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 + a*b^5 ≤ max) ∧
  (x^5*y + x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 + x*y^5 ≤ 22884) ∧
  (∃ (x₀ y₀ : ℝ), x₀ + y₀ = 5 ∧
    x₀^5*y₀ + x₀^4*y₀ + x₀^3*y₀ + x₀^2*y₀ + x₀*y₀ + x₀*y₀^2 + x₀*y₀^3 + x₀*y₀^4 + x₀*y₀^5 = 22884) :=
by sorry

end max_value_polynomial_l3701_370159


namespace gumballs_remaining_is_sixty_l3701_370146

/-- The number of gumballs remaining in the bowl after Pedro takes out 40% -/
def remaining_gumballs (alicia_gumballs : ℕ) : ℕ :=
  let pedro_gumballs := alicia_gumballs + 3 * alicia_gumballs
  let total_gumballs := alicia_gumballs + pedro_gumballs
  let taken_out := (40 * total_gumballs) / 100
  total_gumballs - taken_out

/-- Theorem stating that given Alicia has 20 gumballs, the number of gumballs
    remaining in the bowl after Pedro takes out 40% is 60 -/
theorem gumballs_remaining_is_sixty :
  remaining_gumballs 20 = 60 := by
  sorry

end gumballs_remaining_is_sixty_l3701_370146


namespace max_difference_correct_l3701_370136

/-- Represents a convex N-gon divided into triangles by non-intersecting diagonals --/
structure ConvexNgon (N : ℕ) where
  triangles : ℕ
  diagonals : ℕ
  is_valid : triangles = N - 2 ∧ diagonals = N - 3

/-- Represents a coloring of the triangles in the N-gon --/
structure Coloring (N : ℕ) where
  ngon : ConvexNgon N
  white_triangles : ℕ
  black_triangles : ℕ
  is_valid : white_triangles + black_triangles = ngon.triangles
  adjacent_different : True  -- Represents the condition that adjacent triangles have different colors

/-- The maximum difference between white and black triangles for a given N --/
def max_difference (N : ℕ) : ℕ :=
  if N % 3 = 1 then
    N / 3 - 1
  else
    N / 3

/-- The theorem stating the maximum difference between white and black triangles --/
theorem max_difference_correct (N : ℕ) (c : Coloring N) :
  (c.white_triangles : ℤ) - (c.black_triangles : ℤ) ≤ max_difference N := by
  sorry

end max_difference_correct_l3701_370136


namespace remaining_for_coffee_l3701_370118

def initial_amount : ℝ := 60
def celery_cost : ℝ := 5
def cereal_original_price : ℝ := 12
def cereal_discount : ℝ := 0.5
def bread_cost : ℝ := 8
def milk_original_price : ℝ := 10
def milk_discount : ℝ := 0.1
def potato_cost : ℝ := 1
def potato_quantity : ℕ := 6

def total_spent : ℝ :=
  celery_cost +
  (cereal_original_price * (1 - cereal_discount)) +
  bread_cost +
  (milk_original_price * (1 - milk_discount)) +
  (potato_cost * potato_quantity)

theorem remaining_for_coffee :
  initial_amount - total_spent = 26 :=
sorry

end remaining_for_coffee_l3701_370118


namespace starting_number_proof_l3701_370164

theorem starting_number_proof (n : ℕ) (h1 : n > 0) (h2 : n ≤ 79) (h3 : n % 11 = 0)
  (h4 : ∀ k, k ∈ Finset.range 6 → (n - k * 11) % 11 = 0)
  (h5 : ∀ m, m < n - 5 * 11 → ¬(∃ l, l ∈ Finset.range 6 ∧ m = n - l * 11)) :
  n - 5 * 11 = 22 := by
sorry

end starting_number_proof_l3701_370164


namespace boatman_downstream_distance_l3701_370109

/-- Represents the speed of a boat in various conditions -/
structure BoatSpeed where
  stationary : ℝ
  upstream : ℝ
  current : ℝ
  downstream : ℝ

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem stating the distance traveled by the boatman along the current -/
theorem boatman_downstream_distance 
  (speed : BoatSpeed)
  (h1 : distance speed.upstream 3 = 3) -- 3 km against current in 3 hours
  (h2 : distance speed.stationary 2 = 3) -- 3 km in stationary water in 2 hours
  (h3 : speed.current = speed.stationary - speed.upstream)
  (h4 : speed.downstream = speed.stationary + speed.current) :
  distance speed.downstream 0.5 = 1 := by
  sorry

#check boatman_downstream_distance

end boatman_downstream_distance_l3701_370109


namespace ball_color_probability_l3701_370174

theorem ball_color_probability : 
  let n : ℕ := 8
  let p : ℝ := 1 / 2
  let k : ℕ := 4
  Nat.choose n k * p^n = 35 / 128 := by sorry

end ball_color_probability_l3701_370174


namespace imaginary_part_of_z_l3701_370148

def complexI : ℂ := Complex.I

theorem imaginary_part_of_z (z : ℂ) (h : z / (1 + complexI) = 2 - 3 * complexI) :
  z.im = -1 := by sorry

end imaginary_part_of_z_l3701_370148
