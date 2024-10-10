import Mathlib

namespace video_cassettes_in_second_set_l2411_241167

-- Define the cost of a video cassette
def video_cassette_cost : ℕ := 300

-- Define the equations from the problem
def equation1 (audio_cost video_count : ℕ) : Prop :=
  5 * audio_cost + video_count * video_cassette_cost = 1350

def equation2 (audio_cost : ℕ) : Prop :=
  7 * audio_cost + 3 * video_cassette_cost = 1110

-- Theorem to prove
theorem video_cassettes_in_second_set :
  ∃ (audio_cost video_count : ℕ),
    equation1 audio_cost video_count ∧
    equation2 audio_cost →
    3 = 3 :=
sorry

end video_cassettes_in_second_set_l2411_241167


namespace cubic_quartic_relation_l2411_241191

theorem cubic_quartic_relation (x y : ℝ) 
  (h1 : x^3 + y^3 + 1 / (x^3 + y^3) = 3) 
  (h2 : x + y = 2) : 
  x^4 + y^4 + 1 / (x^4 + y^4) = 257/16 := by
  sorry

end cubic_quartic_relation_l2411_241191


namespace unique_non_representable_expression_l2411_241147

/-- Represents an algebraic expression that may or may not be
    representable as a square of a binomial or difference of squares. -/
inductive BinomialExpression
  | Representable (a b : ℤ) : BinomialExpression
  | NotRepresentable (a b : ℤ) : BinomialExpression

/-- Determines if a given expression can be represented as a 
    square of a binomial or difference of squares. -/
def is_representable (expr : BinomialExpression) : Prop :=
  match expr with
  | BinomialExpression.Representable _ _ => True
  | BinomialExpression.NotRepresentable _ _ => False

/-- The four expressions from the original problem. -/
def expr1 : BinomialExpression := BinomialExpression.Representable 1 (-2)
def expr2 : BinomialExpression := BinomialExpression.Representable 1 (-2)
def expr3 : BinomialExpression := BinomialExpression.Representable 2 (-1)
def expr4 : BinomialExpression := BinomialExpression.NotRepresentable 1 2

theorem unique_non_representable_expression :
  is_representable expr1 ∧
  is_representable expr2 ∧
  is_representable expr3 ∧
  ¬is_representable expr4 :=
sorry

end unique_non_representable_expression_l2411_241147


namespace shaded_area_fraction_l2411_241174

theorem shaded_area_fraction (s : ℝ) (h : s > 0) : 
  let square_area := s^2
  let triangle_area := (1/2) * (s/2) * (s/2)
  let shaded_area := 2 * triangle_area
  shaded_area / square_area = (1 : ℝ) / 4 := by
sorry

end shaded_area_fraction_l2411_241174


namespace smallest_block_size_l2411_241180

/-- Given a rectangular block made of N identical 1-cm cubes, where 378 cubes are not visible
    when three faces are visible, the smallest possible value of N is 560. -/
theorem smallest_block_size (N : ℕ) : 
  (∃ l m n : ℕ, (l - 1) * (m - 1) * (n - 1) = 378 ∧ N = l * m * n) →
  (∀ N' : ℕ, (∃ l' m' n' : ℕ, (l' - 1) * (m' - 1) * (n' - 1) = 378 ∧ N' = l' * m' * n') → N' ≥ N) →
  N = 560 := by
  sorry

end smallest_block_size_l2411_241180


namespace exists_phi_sin_2x_plus_phi_even_l2411_241105

theorem exists_phi_sin_2x_plus_phi_even : ∃ φ : ℝ, ∀ x : ℝ, 
  Real.sin (2 * x + φ) = Real.sin (2 * (-x) + φ) := by
  sorry

end exists_phi_sin_2x_plus_phi_even_l2411_241105


namespace jack_baseball_cards_l2411_241184

theorem jack_baseball_cards :
  ∀ (total_cards baseball_cards football_cards : ℕ),
    total_cards = 125 →
    baseball_cards = 3 * football_cards + 5 →
    total_cards = baseball_cards + football_cards →
    baseball_cards = 95 := by
  sorry

end jack_baseball_cards_l2411_241184


namespace star_equality_implies_y_value_l2411_241173

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := fun (a, b) (c, d) ↦ (a - c, b + d)

/-- Theorem stating that if (5, 0) ★ (2, -2) = (x, y) ★ (0, 3), then y = -5 -/
theorem star_equality_implies_y_value (x y : ℤ) :
  star (5, 0) (2, -2) = star (x, y) (0, 3) → y = -5 := by
  sorry

end star_equality_implies_y_value_l2411_241173


namespace min_sum_of_squares_l2411_241146

/-- Given that a + 2b + 3c + 4d = 12, prove that a^2 + b^2 + c^2 + d^2 ≥ 24/5 -/
theorem min_sum_of_squares (a b c d : ℝ) (h : a + 2*b + 3*c + 4*d = 12) :
  a^2 + b^2 + c^2 + d^2 ≥ 24/5 := by
  sorry

end min_sum_of_squares_l2411_241146


namespace perfume_dilution_l2411_241150

/-- Proves that adding 7.2 ounces of water to 12 ounces of a 40% alcohol solution
    results in a 25% alcohol solution -/
theorem perfume_dilution (initial_volume : ℝ) (initial_concentration : ℝ)
                         (target_concentration : ℝ) (water_added : ℝ) :
  initial_volume = 12 →
  initial_concentration = 0.4 →
  target_concentration = 0.25 →
  water_added = 7.2 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = target_concentration :=
by
  sorry

#check perfume_dilution

end perfume_dilution_l2411_241150


namespace greatest_prime_factor_of_154_l2411_241151

theorem greatest_prime_factor_of_154 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 154 ∧ ∀ q, Nat.Prime q → q ∣ 154 → q ≤ p ∧ p = 11 :=
by sorry

end greatest_prime_factor_of_154_l2411_241151


namespace six_digit_multiply_rearrange_l2411_241189

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧ n / 100000 = 2

def rearranged (n m : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    n = 200000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    m = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + 2

def digit_sum (n : ℕ) : ℕ :=
  (n / 100000) + ((n / 10000) % 10) + ((n / 1000) % 10) +
  ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem six_digit_multiply_rearrange (n : ℕ) :
  is_valid_number n → rearranged n (3 * n) → digit_sum n = 27 :=
by sorry

end six_digit_multiply_rearrange_l2411_241189


namespace urn_problem_l2411_241164

theorem urn_problem (M : ℕ) : M = 111 ↔ 
  (5 : ℝ) / 12 * 20 / (20 + M) + 7 / 12 * M / (20 + M) = 0.62 := by
  sorry

end urn_problem_l2411_241164


namespace altitude_segment_theorem_l2411_241169

-- Define the triangle and its properties
structure AcuteTriangle where
  -- We don't need to explicitly define the vertices, just the properties we need
  altitude1_segment1 : ℝ
  altitude1_segment2 : ℝ
  altitude2_segment1 : ℝ
  altitude2_segment2 : ℝ
  acute : Bool
  h_acute : acute = true
  h_altitude1 : altitude1_segment1 = 6 ∧ altitude1_segment2 = 4
  h_altitude2 : altitude2_segment1 = 3

-- State the theorem
theorem altitude_segment_theorem (t : AcuteTriangle) : t.altitude2_segment2 = 31/3 := by
  sorry

end altitude_segment_theorem_l2411_241169


namespace jack_hunting_frequency_l2411_241127

/-- Represents the hunting scenario for Jack --/
structure HuntingScenario where
  seasonLength : ℚ  -- Length of hunting season in quarters of a year
  deersPerTrip : ℕ  -- Number of deers caught per hunting trip
  deerWeight : ℕ    -- Weight of each deer in pounds
  keepRatio : ℚ     -- Ratio of deer weight kept per year
  keptWeight : ℕ    -- Total weight of deer kept in pounds

/-- Calculates the number of hunting trips per month --/
def tripsPerMonth (scenario : HuntingScenario) : ℚ :=
  let totalWeight := scenario.keptWeight / scenario.keepRatio
  let weightPerTrip := scenario.deersPerTrip * scenario.deerWeight
  let tripsPerYear := totalWeight / weightPerTrip
  let monthsInSeason := scenario.seasonLength * 12
  tripsPerYear / monthsInSeason

/-- Theorem stating that Jack goes hunting 6 times per month --/
theorem jack_hunting_frequency :
  let scenario : HuntingScenario := {
    seasonLength := 1/4,
    deersPerTrip := 2,
    deerWeight := 600,
    keepRatio := 1/2,
    keptWeight := 10800
  }
  tripsPerMonth scenario = 6 := by sorry

end jack_hunting_frequency_l2411_241127


namespace nested_expression_value_l2411_241130

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end nested_expression_value_l2411_241130


namespace circle_area_ratio_l2411_241136

theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0)
  (h_arc : (60 / 360) * (2 * Real.pi * C) = (40 / 360) * (2 * Real.pi * D)) :
  (Real.pi * C^2) / (Real.pi * D^2) = 9/4 := by
sorry

end circle_area_ratio_l2411_241136


namespace concentric_circles_area_ratio_l2411_241117

/-- Given two concentric circles D and C, where C is inside D, 
    this theorem proves the diameter of C when the ratio of 
    the area between the circles to the area of C is 4:1 -/
theorem concentric_circles_area_ratio (d_diameter : ℝ) 
  (h_d_diameter : d_diameter = 24) 
  (c_diameter : ℝ) 
  (h_inside : c_diameter < d_diameter) 
  (h_ratio : (π * (d_diameter/2)^2 - π * (c_diameter/2)^2) / (π * (c_diameter/2)^2) = 4) :
  c_diameter = 24 * Real.sqrt 5 / 5 := by
  sorry

#check concentric_circles_area_ratio

end concentric_circles_area_ratio_l2411_241117


namespace red_marbles_taken_away_l2411_241148

/-- Proves that the number of red marbles taken away is 3 --/
theorem red_marbles_taken_away :
  let initial_red : ℕ := 20
  let initial_blue : ℕ := 30
  let total_left : ℕ := 35
  ∃ (red_taken : ℕ),
    (initial_red - red_taken) + (initial_blue - 4 * red_taken) = total_left ∧
    red_taken = 3 :=
by sorry

end red_marbles_taken_away_l2411_241148


namespace unique_palindrome_square_l2411_241115

/-- A function that returns true if a number is a three-digit palindrome with an even middle digit -/
def is_valid_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  (n / 100 = n % 10) ∧  -- first and last digits are the same
  (n / 10 % 10) % 2 = 0  -- middle digit is even

/-- The main theorem stating that there is exactly one number satisfying the conditions -/
theorem unique_palindrome_square : ∃! n : ℕ, 
  is_valid_palindrome n ∧ ∃ m : ℕ, n = m^2 :=
sorry

end unique_palindrome_square_l2411_241115


namespace unique_solution_l2411_241110

/-- Given a real number a, returns the sum of coefficients of odd powers of x 
    in the expansion of (1+ax)^2(1-x)^5 -/
def oddPowerSum (a : ℝ) : ℝ := sorry

theorem unique_solution : 
  ∃! (a : ℝ), a > 0 ∧ oddPowerSum a = -64 :=
by
  use 3
  sorry

end unique_solution_l2411_241110


namespace common_factor_of_2a2_and_4ab_l2411_241183

theorem common_factor_of_2a2_and_4ab :
  ∀ (a b : ℤ), ∃ (k₁ k₂ : ℤ), 2 * a^2 = (2 * a) * k₁ ∧ 4 * a * b = (2 * a) * k₂ ∧
  (∀ (d : ℤ), (∃ (m₁ m₂ : ℤ), 2 * a^2 = d * m₁ ∧ 4 * a * b = d * m₂) → d ∣ (2 * a)) :=
by sorry

end common_factor_of_2a2_and_4ab_l2411_241183


namespace opposite_of_abs_one_over_2023_l2411_241121

theorem opposite_of_abs_one_over_2023 :
  -(|1 / 2023|) = -1 / 2023 := by sorry

end opposite_of_abs_one_over_2023_l2411_241121


namespace gcd_245_1001_l2411_241126

theorem gcd_245_1001 : Nat.gcd 245 1001 = 7 := by sorry

end gcd_245_1001_l2411_241126


namespace adams_lawn_mowing_l2411_241109

/-- Given that Adam earns 9 dollars per lawn, forgot to mow 8 lawns, and actually earned 36 dollars,
    prove that the total number of lawns he had to mow is 12. -/
theorem adams_lawn_mowing (dollars_per_lawn : ℕ) (forgotten_lawns : ℕ) (actual_earnings : ℕ) :
  dollars_per_lawn = 9 →
  forgotten_lawns = 8 →
  actual_earnings = 36 →
  (actual_earnings / dollars_per_lawn) + forgotten_lawns = 12 :=
by sorry

end adams_lawn_mowing_l2411_241109


namespace exists_function_with_properties_l2411_241153

theorem exists_function_with_properties : ∃ f : ℝ → ℝ, 
  (∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1) ∧ 
  (∀ x : ℝ, f x = f (-x)) := by
  sorry

end exists_function_with_properties_l2411_241153


namespace painted_cube_probability_l2411_241179

/-- Represents a cube with side length 5 and two adjacent faces painted --/
structure PaintedCube :=
  (side_length : ℕ)
  (painted_faces : ℕ)

/-- Calculates the total number of unit cubes in the large cube --/
def total_cubes (c : PaintedCube) : ℕ :=
  c.side_length ^ 3

/-- Calculates the number of unit cubes with exactly two painted faces --/
def two_painted_faces (c : PaintedCube) : ℕ :=
  (c.side_length - 2) ^ 2

/-- Calculates the number of unit cubes with no painted faces --/
def no_painted_faces (c : PaintedCube) : ℕ :=
  total_cubes c - (2 * c.side_length ^ 2 - c.side_length)

/-- Calculates the probability of selecting one cube with two painted faces
    and one cube with no painted faces --/
def probability (c : PaintedCube) : ℚ :=
  (two_painted_faces c * no_painted_faces c : ℚ) /
  (total_cubes c * (total_cubes c - 1) / 2 : ℚ)

/-- The main theorem stating the probability for a 5x5x5 cube with two painted faces --/
theorem painted_cube_probability :
  let c := PaintedCube.mk 5 2
  probability c = 24 / 258 := by
  sorry


end painted_cube_probability_l2411_241179


namespace power_function_sum_l2411_241195

/-- A function f is a power function if it can be written as f(x) = k * x^c, 
    where k and c are constants, and c is not zero. -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (k c : ℝ), c ≠ 0 ∧ ∀ x, f x = k * x^c

/-- Given that f(x) = a*x^(2a+1) - b + 1 is a power function, prove that a + b = 2 -/
theorem power_function_sum (a b : ℝ) :
  isPowerFunction (fun x ↦ a * x^(2*a+1) - b + 1) → a + b = 2 := by
  sorry

end power_function_sum_l2411_241195


namespace sale_ratio_l2411_241107

def floral_shop_sales (monday_sales : ℕ) (total_sales : ℕ) : Prop :=
  let tuesday_sales := 3 * monday_sales
  let wednesday_sales := total_sales - (monday_sales + tuesday_sales)
  (wednesday_sales : ℚ) / tuesday_sales = 1 / 3

theorem sale_ratio : floral_shop_sales 12 60 := by
  sorry

end sale_ratio_l2411_241107


namespace vector_magnitude_condition_l2411_241170

theorem vector_magnitude_condition (n : Type*) [NormedAddCommGroup n] :
  ∃ (a b : n),
    (‖a‖ = ‖b‖ ∧ ‖a + b‖ ≠ ‖a - b‖) ∧
    (‖a‖ ≠ ‖b‖ ∧ ‖a + b‖ = ‖a - b‖) :=
by sorry

end vector_magnitude_condition_l2411_241170


namespace triangle_problem_l2411_241188

open Real

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a * cos B + b * cos A = 2 * c * cos C →
  c = 2 * Real.sqrt 3 →
  C = π / 3 ∧
  (∃ (area : ℝ), area ≤ 3 * Real.sqrt 3 ∧
    ∀ (area' : ℝ), area' = 1/2 * a * b * sin C → area' ≤ area) :=
by sorry

end triangle_problem_l2411_241188


namespace expected_hypertension_cases_l2411_241199

/-- Given a population where 1 out of 3 individuals has a condition,
    prove that the expected number of individuals with the condition
    in a sample of 450 is 150. -/
theorem expected_hypertension_cases (
  total_sample : ℕ
  ) (h1 : total_sample = 450)
  (probability : ℚ)
  (h2 : probability = 1 / 3) :
  ↑total_sample * probability = 150 :=
sorry

end expected_hypertension_cases_l2411_241199


namespace volunteer_selection_probabilities_l2411_241129

/-- Represents the number of calligraphy competition winners -/
def calligraphy_winners : ℕ := 4

/-- Represents the number of painting competition winners -/
def painting_winners : ℕ := 2

/-- Represents the total number of winners -/
def total_winners : ℕ := calligraphy_winners + painting_winners

/-- Represents the number of volunteers to be selected -/
def volunteers_needed : ℕ := 2

/-- The probability of selecting both volunteers from calligraphy winners -/
def prob_both_calligraphy : ℚ := 2 / 5

/-- The probability of selecting one volunteer from each competition -/
def prob_one_each : ℚ := 8 / 15

theorem volunteer_selection_probabilities :
  (Nat.choose calligraphy_winners volunteers_needed) / (Nat.choose total_winners volunteers_needed) = prob_both_calligraphy ∧
  (calligraphy_winners * painting_winners) / (Nat.choose total_winners volunteers_needed) = prob_one_each :=
sorry

end volunteer_selection_probabilities_l2411_241129


namespace point_coordinates_l2411_241197

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the distance from a point to the x-axis
def distToXAxis (p : Point) : ℝ := |p.2|

-- Define the distance from a point to the y-axis
def distToYAxis (p : Point) : ℝ := |p.1|

-- Theorem statement
theorem point_coordinates (P : Point) :
  P.2 > 0 →  -- P is above the x-axis
  P.1 < 0 →  -- P is to the left of the y-axis
  distToXAxis P = 4 →  -- P is 4 units away from x-axis
  distToYAxis P = 4 →  -- P is 4 units away from y-axis
  P = (-4, 4) := by
sorry

end point_coordinates_l2411_241197


namespace count_odd_integers_between_fractions_l2411_241158

theorem count_odd_integers_between_fractions :
  let lower_bound : ℚ := 17 / 4
  let upper_bound : ℚ := 35 / 2
  (Finset.filter (fun n => n % 2 = 1)
    (Finset.Icc (Int.ceil lower_bound) (Int.floor upper_bound))).card = 7 := by
  sorry

end count_odd_integers_between_fractions_l2411_241158


namespace ellipse_eccentricity_equilateral_l2411_241133

/-- An ellipse with a vertex and foci forming an equilateral triangle has eccentricity 1/2 -/
theorem ellipse_eccentricity_equilateral (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive semi-axes and focal distance
  a^2 = b^2 + c^2 →        -- Relationship between semi-axes and focal distance
  b = Real.sqrt 3 * c →    -- Condition for equilateral triangle
  c / a = 1 / 2 :=         -- Eccentricity definition and target value
by sorry

end ellipse_eccentricity_equilateral_l2411_241133


namespace adjacent_knights_probability_l2411_241120

def number_of_knights : ℕ := 30
def chosen_knights : ℕ := 4

def probability_adjacent_knights : ℚ :=
  1 - (Nat.choose (number_of_knights - chosen_knights) chosen_knights : ℚ) /
      (Nat.choose number_of_knights chosen_knights : ℚ)

theorem adjacent_knights_probability :
  probability_adjacent_knights = 250 / 549 := by sorry

end adjacent_knights_probability_l2411_241120


namespace k_value_l2411_241172

theorem k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - 2*k)*(x + 3*k) = x^3 + 3*k*(x^2 - x - 4)) →
  k = 2 := by
sorry

end k_value_l2411_241172


namespace interior_angles_sum_l2411_241131

theorem interior_angles_sum (n : ℕ) (h : 180 * (n - 2) = 2340) :
  180 * ((n + 3) - 2) = 2880 := by
  sorry

end interior_angles_sum_l2411_241131


namespace capital_after_18_years_l2411_241193

def initial_investment : ℝ := 2000
def increase_rate : ℝ := 0.5
def years_per_period : ℕ := 3
def total_years : ℕ := 18

theorem capital_after_18_years :
  let periods : ℕ := total_years / years_per_period
  let growth_factor : ℝ := 1 + increase_rate
  let final_capital : ℝ := initial_investment * growth_factor ^ periods
  final_capital = 22781.25 := by sorry

end capital_after_18_years_l2411_241193


namespace max_candy_consumption_l2411_241186

theorem max_candy_consumption (n : ℕ) (h : n = 45) : 
  (n * (n - 1)) / 2 = 990 :=
by sorry

end max_candy_consumption_l2411_241186


namespace fifth_term_of_arithmetic_sequence_l2411_241116

/-- 
Given an arithmetic sequence starting with 3, 7, 11, ..., 
prove that its fifth term is 19.
-/
theorem fifth_term_of_arithmetic_sequence : 
  ∀ (a : ℕ → ℕ), 
  (a 0 = 3) → 
  (a 1 = 7) → 
  (a 2 = 11) → 
  (∀ n, a (n + 1) - a n = a 1 - a 0) → 
  a 4 = 19 := by
sorry

end fifth_term_of_arithmetic_sequence_l2411_241116


namespace division_remainder_l2411_241108

theorem division_remainder : ∃ (A : ℕ), 17 = 6 * 2 + A ∧ A < 6 := by
  sorry

end division_remainder_l2411_241108


namespace cat_in_bag_change_l2411_241178

theorem cat_in_bag_change (p : ℕ) (h : 0 < p ∧ p ≤ 1000) : 
  ∃ (change : ℕ), change = 1000 - p := by
  sorry

end cat_in_bag_change_l2411_241178


namespace x_coordinate_at_y_3_l2411_241154

-- Define the line
def line (x y : ℝ) : Prop :=
  y + 3 = (1/2) * (x + 2)

-- Define the point (-2, -3) on the line
axiom point_on_line : line (-2) (-3)

-- Define the x-intercept
axiom x_intercept : line 4 0

-- Theorem to prove
theorem x_coordinate_at_y_3 :
  ∃ (x : ℝ), line x 3 ∧ x = 10 :=
sorry

end x_coordinate_at_y_3_l2411_241154


namespace oatmeal_raisin_cookies_l2411_241101

/-- Given a class of students and cookie preferences, calculate the number of oatmeal raisin cookies to be made. -/
theorem oatmeal_raisin_cookies (total_students : ℕ) (cookies_per_student : ℕ) (oatmeal_raisin_percentage : ℚ) : 
  total_students = 40 → 
  cookies_per_student = 2 → 
  oatmeal_raisin_percentage = 1/10 →
  (total_students : ℚ) * oatmeal_raisin_percentage * cookies_per_student = 8 := by
  sorry

#check oatmeal_raisin_cookies

end oatmeal_raisin_cookies_l2411_241101


namespace consecutive_squares_difference_l2411_241103

theorem consecutive_squares_difference (n : ℕ) : (n + 1)^2 - n^2 = 2*n + 1 := by
  sorry

end consecutive_squares_difference_l2411_241103


namespace min_value_of_reciprocal_sum_l2411_241106

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 9/y ≥ 1/a + 9/b) →
  1/a + 9/b = 16 :=
by sorry

end min_value_of_reciprocal_sum_l2411_241106


namespace cambridge_population_l2411_241182

-- Define the number of people in Cambridge
variable (n : ℕ)

-- Define the total amount of water and apple juice consumed
variable (W A : ℝ)

-- Define the mayor's drink
variable (L : ℝ)

-- Each person drinks 12 ounces
axiom total_drink : W + A = 12 * n

-- The mayor's drink is 12 ounces
axiom mayor_drink : L = 12

-- The mayor drinks 1/6 of total water and 1/8 of total apple juice
axiom mayor_portions : L = (1/6) * W + (1/8) * A

-- All drinks have positive amounts of both liquids
axiom positive_amounts : W > 0 ∧ A > 0

-- Theorem: The number of people in Cambridge is 7
theorem cambridge_population : n = 7 :=
sorry

end cambridge_population_l2411_241182


namespace triangle_area_l2411_241100

theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.cos (B - C) + a * Real.cos A = 2 * Real.sqrt 3 * c * Real.sin B * Real.cos A →
  b^2 + c^2 - a^2 = 2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 := by
sorry

end triangle_area_l2411_241100


namespace intersection_slope_range_l2411_241132

/-- Given two points P and Q in the Cartesian plane, and a linear function y = kx - 1
    that intersects the extension of line segment PQ (excluding Q),
    prove that the range of k is between 1/3 and 3/2 (exclusive). -/
theorem intersection_slope_range (P Q : ℝ × ℝ) (k : ℝ) : 
  P = (-1, 1) →
  Q = (2, 2) →
  (∃ x y : ℝ, y = k * x - 1 ∧ 
              (y - 1) / (x + 1) = (2 - 1) / (2 + 1) ∧
              (x, y) ≠ Q) →
  1/3 < k ∧ k < 3/2 :=
by sorry

end intersection_slope_range_l2411_241132


namespace sqrt_three_irrational_l2411_241102

theorem sqrt_three_irrational :
  (∃ (q : ℚ), (1 : ℝ) / 3 = ↑q) ∧ 
  (∃ (q : ℚ), (3.14 : ℝ) = ↑q) ∧ 
  (∃ (q : ℚ), Real.sqrt 9 = ↑q) →
  ¬ ∃ (q : ℚ), Real.sqrt 3 = ↑q :=
by sorry

end sqrt_three_irrational_l2411_241102


namespace initial_friends_count_l2411_241143

theorem initial_friends_count (initial_group : ℕ) (additional_friends : ℕ) (total_people : ℕ) : 
  initial_group + additional_friends = total_people ∧ 
  additional_friends = 3 ∧ 
  total_people = 7 → 
  initial_group = 4 := by
  sorry

end initial_friends_count_l2411_241143


namespace commodity_consumption_increase_l2411_241145

theorem commodity_consumption_increase
  (original_tax : ℝ)
  (original_consumption : ℝ)
  (h_tax_positive : original_tax > 0)
  (h_consumption_positive : original_consumption > 0)
  (h_tax_reduction : ℝ)
  (h_revenue_decrease : ℝ)
  (h_consumption_increase : ℝ)
  (h_tax_reduction_eq : h_tax_reduction = 0.20)
  (h_revenue_decrease_eq : h_revenue_decrease = 0.16)
  (h_new_tax : ℝ := original_tax * (1 - h_tax_reduction))
  (h_new_consumption : ℝ := original_consumption * (1 + h_consumption_increase))
  (h_new_revenue : ℝ := h_new_tax * h_new_consumption)
  (h_original_revenue : ℝ := original_tax * original_consumption)
  (h_revenue_equation : h_new_revenue = h_original_revenue * (1 - h_revenue_decrease)) :
  h_consumption_increase = 0.05 := by sorry

end commodity_consumption_increase_l2411_241145


namespace sum_mod_eleven_l2411_241196

theorem sum_mod_eleven : (10555 + 10556 + 10557 + 10558 + 10559) % 11 = 4 := by
  sorry

end sum_mod_eleven_l2411_241196


namespace fraction_simplification_l2411_241112

theorem fraction_simplification :
  (1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25 := by
  sorry

end fraction_simplification_l2411_241112


namespace cube_opposite_face_l2411_241118

/-- Represents a face of the cube -/
inductive Face : Type
| A | B | C | D | E | F

/-- Represents the adjacency relation between faces -/
def adjacent : Face → Face → Prop := sorry

/-- Represents the opposite relation between faces -/
def opposite : Face → Face → Prop := sorry

/-- The theorem stating that F is opposite to A in the given cube configuration -/
theorem cube_opposite_face :
  (adjacent Face.A Face.B) →
  (adjacent Face.A Face.C) →
  (adjacent Face.B Face.D) →
  (opposite Face.A Face.F) := by sorry

end cube_opposite_face_l2411_241118


namespace divide_by_repeating_decimal_l2411_241124

def repeating_decimal_to_fraction (a b : ℕ) : ℚ :=
  (a : ℚ) / (99 : ℚ)

theorem divide_by_repeating_decimal (a b : ℕ) :
  (7 : ℚ) / (repeating_decimal_to_fraction a b) = 38.5 :=
sorry

end divide_by_repeating_decimal_l2411_241124


namespace parallelogram_side_sum_l2411_241157

/-- 
A parallelogram has side lengths of 10, 12, 10y-2, and 4x+6. 
This theorem proves that x+y = 2.7.
-/
theorem parallelogram_side_sum (x y : ℝ) : 
  (4*x + 6 = 12) → (10*y - 2 = 10) → x + y = 2.7 := by sorry

end parallelogram_side_sum_l2411_241157


namespace function_always_negative_m_range_l2411_241166

theorem function_always_negative_m_range
  (f : ℝ → ℝ)
  (m : ℝ)
  (h1 : ∀ x, f x = m * x^2 - m * x - 1)
  (h2 : ∀ x, f x < 0) :
  -4 < m ∧ m ≤ 0 :=
by sorry

end function_always_negative_m_range_l2411_241166


namespace two_integer_solutions_l2411_241113

theorem two_integer_solutions :
  ∃! (s : Finset ℤ), (∀ x ∈ s, |3*x - 4| + |3*x + 2| = 6) ∧ s.card = 2 :=
by sorry

end two_integer_solutions_l2411_241113


namespace square_side_length_l2411_241194

theorem square_side_length (rectangle_width rectangle_length : ℝ) 
  (h1 : rectangle_width = 4)
  (h2 : rectangle_length = 9)
  (h3 : rectangle_width > 0)
  (h4 : rectangle_length > 0) :
  ∃ (square_side : ℝ), 
    square_side * square_side = rectangle_width * rectangle_length ∧ 
    square_side = 6 := by
  sorry

end square_side_length_l2411_241194


namespace subtracted_value_l2411_241111

theorem subtracted_value (x y : ℤ) : x = 60 ∧ 4 * x - y = 102 → y = 138 := by
  sorry

end subtracted_value_l2411_241111


namespace triangle_angle_weighted_average_bounds_l2411_241162

theorem triangle_angle_weighted_average_bounds 
  (A B C a b c : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) : 
  π / 3 ≤ (a * A + b * B + c * C) / (a + b + c) ∧ 
  (a * A + b * B + c * C) / (a + b + c) < π / 2 := by
sorry

end triangle_angle_weighted_average_bounds_l2411_241162


namespace fraction_representation_l2411_241152

theorem fraction_representation (n : ℕ) : ∃ x y : ℕ, n = x^2 / y^3 := by
  sorry

end fraction_representation_l2411_241152


namespace max_S_value_l2411_241104

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  -- Triangle inequality constraints
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  ineq_ab : a + b > c
  ineq_bc : b + c > a
  ineq_ca : c + a > b

-- Define the area function S
def S (t : Triangle) : ℝ := (t.a - t.b + t.c) * (t.a + t.b - t.c)

-- Theorem statement
theorem max_S_value (t : Triangle) (h : t.b + t.c = 8) :
  S t ≤ 64 / 17 :=
sorry

end max_S_value_l2411_241104


namespace exponent_division_l2411_241114

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^2 / x^5 = 1 / x^3 := by
  sorry

end exponent_division_l2411_241114


namespace perfect_cube_values_l2411_241139

theorem perfect_cube_values (Z K : ℤ) (h1 : 600 < Z) (h2 : Z < 2000) (h3 : K > 1) (h4 : Z = K^3) :
  K = 9 ∨ K = 10 ∨ K = 11 ∨ K = 12 := by
  sorry

end perfect_cube_values_l2411_241139


namespace inscribed_circle_radius_l2411_241198

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 5) (hb : b = 12) (hc : c = 15) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 20 / 19 := by sorry

end inscribed_circle_radius_l2411_241198


namespace fourth_intersection_point_l2411_241135

def curve (x y : ℝ) : Prop := x * y = 2

theorem fourth_intersection_point (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (h₁ : curve x₁ y₁) (h₂ : curve x₂ y₂) (h₃ : curve x₃ y₃) (h₄ : curve x₄ y₄)
  (p₁ : x₁ = 4 ∧ y₁ = 1/2) (p₂ : x₂ = -2 ∧ y₂ = -1) (p₃ : x₃ = 1/4 ∧ y₃ = 8)
  (distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) :
  x₄ = 1 ∧ y₄ = 2 := by
  sorry

end fourth_intersection_point_l2411_241135


namespace rotate_right_triangle_surface_area_l2411_241149

/-- The surface area of a solid formed by rotating a right triangle with sides 3, 4, and 5 around its shortest side -/
theorem rotate_right_triangle_surface_area :
  let triangle : Fin 3 → ℝ := ![3, 4, 5]
  let shortest_side := triangle 0
  let hypotenuse := triangle 2
  let height := triangle 1
  let base_area := π * height ^ 2
  let lateral_area := π * height * hypotenuse
  base_area + lateral_area = 36 * π :=
by sorry

end rotate_right_triangle_surface_area_l2411_241149


namespace tan_37_5_deg_identity_l2411_241185

theorem tan_37_5_deg_identity : 
  (Real.tan (37.5 * π / 180)) / (1 - (Real.tan (37.5 * π / 180))^2) = 1 + (Real.sqrt 3) / 2 := by
  sorry

end tan_37_5_deg_identity_l2411_241185


namespace min_value_for_four_digit_product_l2411_241165

theorem min_value_for_four_digit_product (n : ℕ) : 
  (341 * n ≥ 1000 ∧ ∀ m < n, 341 * m < 1000) → n = 3 := by
  sorry

end min_value_for_four_digit_product_l2411_241165


namespace root_product_equals_sixteen_l2411_241163

theorem root_product_equals_sixteen :
  (16 : ℝ)^(1/4) * (64 : ℝ)^(1/3) * (4 : ℝ)^(1/2) = 16 := by sorry

end root_product_equals_sixteen_l2411_241163


namespace opposite_of_2021_l2411_241176

theorem opposite_of_2021 : -(2021 : ℤ) = -2021 := by
  sorry

end opposite_of_2021_l2411_241176


namespace best_fit_r_squared_l2411_241175

def r_squared_values : List ℝ := [0.27, 0.85, 0.96, 0.5]

theorem best_fit_r_squared (best_fit : ℝ) (h : best_fit ∈ r_squared_values) :
  (∀ x ∈ r_squared_values, x ≤ best_fit) ∧ best_fit = 0.96 := by
  sorry

end best_fit_r_squared_l2411_241175


namespace distinct_divisors_lower_bound_l2411_241128

theorem distinct_divisors_lower_bound (n : ℕ) (A : ℕ) (factors : Finset ℕ) 
  (h1 : factors.card = n)
  (h2 : ∀ x ∈ factors, x > 1)
  (h3 : A = factors.prod id) :
  (Finset.filter (· ∣ A) (Finset.range (A + 1))).card ≥ n * (n - 1) / 2 + 1 := by
  sorry

end distinct_divisors_lower_bound_l2411_241128


namespace equation_solution_l2411_241122

theorem equation_solution (x : ℝ) : (2*x - 3)^(x + 3) = 1 ↔ x = -3 ∨ x = 2 ∨ x = -1 := by
  sorry

end equation_solution_l2411_241122


namespace extended_quad_ratio_gt_one_ratio_always_gt_one_l2411_241161

/-- Represents a convex quadrilateral ABCD with an extended construction --/
structure ExtendedQuadrilateral where
  /-- The sum of all internal angles of the quadrilateral --/
  internal_sum : ℝ
  /-- The sum of angles BAD and ABC --/
  partial_sum : ℝ
  /-- Assumption that the quadrilateral is convex --/
  convex : 0 < partial_sum ∧ partial_sum < internal_sum

/-- The ratio of external angle sum to partial internal angle sum is greater than 1 --/
theorem extended_quad_ratio_gt_one (q : ExtendedQuadrilateral) : 
  q.internal_sum / q.partial_sum > 1 := by
  sorry

/-- Main theorem: For any convex quadrilateral with the given construction, 
    the ratio r is always greater than 1 --/
theorem ratio_always_gt_one : 
  ∀ q : ExtendedQuadrilateral, q.internal_sum / q.partial_sum > 1 := by
  sorry

end extended_quad_ratio_gt_one_ratio_always_gt_one_l2411_241161


namespace min_distance_circle_line_l2411_241190

theorem min_distance_circle_line : 
  ∃ (d : ℝ), d = 2 ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), 
    ((x₁ - Real.sqrt 3 / 2)^2 + (y₁ - 1/2)^2 = 1) →
    (Real.sqrt 3 * x₂ + y₂ = 8) →
    d ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) ∧
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    ((x₁ - Real.sqrt 3 / 2)^2 + (y₁ - 1/2)^2 = 1) ∧
    (Real.sqrt 3 * x₂ + y₂ = 8) ∧
    d = Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)) :=
by
  sorry

end min_distance_circle_line_l2411_241190


namespace fraction_product_simplification_l2411_241155

theorem fraction_product_simplification :
  (18 : ℚ) / 17 * 13 / 24 * 68 / 39 = 1 := by sorry

end fraction_product_simplification_l2411_241155


namespace min_yellow_surface_fraction_l2411_241156

/-- Represents a 4x4x4 cube constructed from smaller 1-inch cubes -/
structure LargeCube where
  small_cubes : Fin 64 → Color
  blue_count : Nat
  yellow_count : Nat
  h_blue_count : blue_count = 32
  h_yellow_count : yellow_count = 32
  h_total_count : blue_count + yellow_count = 64

inductive Color
  | Blue
  | Yellow

/-- Calculates the surface area of the large cube -/
def surface_area : Nat := 6 * 4 * 4

/-- Calculates the minimum yellow surface area possible -/
def min_yellow_surface_area (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating that the minimum fraction of yellow surface area is 1/4 -/
theorem min_yellow_surface_fraction (cube : LargeCube) :
  (min_yellow_surface_area cube : ℚ) / surface_area = 1 / 4 := by
  sorry

end min_yellow_surface_fraction_l2411_241156


namespace keiko_text_messages_l2411_241160

/-- The number of text messages Keiko sent in the first week -/
def first_week : ℕ := 111

/-- The number of text messages Keiko sent in the second week -/
def second_week : ℕ := 2 * first_week - 50

/-- The number of text messages Keiko sent in the third week -/
def third_week : ℕ := second_week + (second_week / 4)

/-- The total number of text messages Keiko sent over three weeks -/
def total_messages : ℕ := first_week + second_week + third_week

theorem keiko_text_messages : total_messages = 498 := by
  sorry

end keiko_text_messages_l2411_241160


namespace triangle_angle_sum_l2411_241134

theorem triangle_angle_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a + b + c = 180) (h5 : a = 37) (h6 : b = 53) : c = 90 := by
  sorry

end triangle_angle_sum_l2411_241134


namespace golden_ratio_from_logarithms_l2411_241137

theorem golden_ratio_from_logarithms (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.log a / Real.log 4 = Real.log b / Real.log 18) ∧ 
  (Real.log b / Real.log 18 = Real.log (a + b) / Real.log 32) →
  b / a = (1 + Real.sqrt 5) / 2 := by
  sorry

end golden_ratio_from_logarithms_l2411_241137


namespace exists_zero_implies_a_range_l2411_241159

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * abs x - 3 * a - 1

-- State the theorem
theorem exists_zero_implies_a_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-1) 1 ∧ f a x₀ = 0) →
  a ∈ Set.Icc (-1/2) (-1/3) :=
by sorry

end exists_zero_implies_a_range_l2411_241159


namespace optimal_sampling_methods_l2411_241119

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a box of ping-pong balls -/
structure Box where
  color : String
  count : Nat

/-- Represents the ping-pong ball problem -/
structure PingPongProblem where
  totalBalls : Nat
  boxes : List Box
  sampleSize : Nat

/-- Represents the student selection problem -/
structure StudentProblem where
  totalStudents : Nat
  selectCount : Nat

/-- Determines the optimal sampling method for a given problem -/
def optimalSamplingMethod (p : PingPongProblem ⊕ StudentProblem) : SamplingMethod :=
  match p with
  | .inl _ => SamplingMethod.Stratified
  | .inr _ => SamplingMethod.SimpleRandom

/-- The main theorem stating the optimal sampling methods for both problems -/
theorem optimal_sampling_methods 
  (pingPong : PingPongProblem)
  (student : StudentProblem)
  (h1 : pingPong.totalBalls = 1000)
  (h2 : pingPong.boxes = [
    { color := "red", count := 500 },
    { color := "blue", count := 200 },
    { color := "yellow", count := 300 }
  ])
  (h3 : pingPong.sampleSize = 100)
  (h4 : student.totalStudents = 20)
  (h5 : student.selectCount = 3) :
  (optimalSamplingMethod (.inl pingPong) = SamplingMethod.Stratified) ∧
  (optimalSamplingMethod (.inr student) = SamplingMethod.SimpleRandom) :=
sorry


end optimal_sampling_methods_l2411_241119


namespace battery_price_l2411_241142

theorem battery_price (total_cost tire_cost : ℕ) (h1 : total_cost = 224) (h2 : tire_cost = 42) :
  total_cost - 4 * tire_cost = 56 := by
  sorry

end battery_price_l2411_241142


namespace probability_between_lines_in_first_quadrant_l2411_241192

/-- Line represented by a linear equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def Line.eval (l : Line) (x : ℝ) : ℝ := l.m * x + l.b

def is_below (p : Point) (l : Line) : Prop := p.y ≤ l.eval p.x

def is_in_first_quadrant (p : Point) : Prop := p.x ≥ 0 ∧ p.y ≥ 0

def is_between_lines (p : Point) (l1 l2 : Line) : Prop :=
  is_below p l1 ∧ ¬is_below p l2

theorem probability_between_lines_in_first_quadrant
  (l m : Line)
  (h1 : l.m = -3 ∧ l.b = 9)
  (h2 : m.m = -1 ∧ m.b = 3)
  (h3 : ∀ (p : Point), is_in_first_quadrant p → is_below p l → is_below p m) :
  (∀ (p : Point), is_in_first_quadrant p → is_below p l → is_between_lines p l m) :=
sorry

end probability_between_lines_in_first_quadrant_l2411_241192


namespace madeline_and_brother_total_money_l2411_241141

def madeline_money : ℕ := 48

theorem madeline_and_brother_total_money :
  madeline_money + (madeline_money / 2) = 72 := by
  sorry

end madeline_and_brother_total_money_l2411_241141


namespace kids_difference_l2411_241123

/-- The number of kids Julia played with on each day of the week. -/
structure WeeklyKids where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Theorem stating the difference in the number of kids played with on specific days. -/
theorem kids_difference (w : WeeklyKids)
    (h1 : w.monday = 6)
    (h2 : w.tuesday = 17)
    (h3 : w.wednesday = 4)
    (h4 : w.thursday = 12)
    (h5 : w.friday = 10)
    (h6 : w.saturday = 15)
    (h7 : w.sunday = 9) :
    (w.tuesday + w.thursday) - (w.monday + w.wednesday + w.sunday) = 10 := by
  sorry


end kids_difference_l2411_241123


namespace angle_is_135_degrees_l2411_241140

def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_is_135_degrees (a b : ℝ × ℝ) 
  (sum_condition : a.1 + b.1 = 2 ∧ a.2 + b.2 = -1)
  (a_condition : a = (1, 2)) :
  angle_between_vectors a b = 135 * (π / 180) := by sorry

end angle_is_135_degrees_l2411_241140


namespace necessary_but_not_sufficient_condition_l2411_241144

theorem necessary_but_not_sufficient_condition (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) := by
  sorry

end necessary_but_not_sufficient_condition_l2411_241144


namespace displeased_polynomial_at_one_is_zero_l2411_241187

-- Define a polynomial p(x) = x^2 - (m+n)x + mn
def p (m n : ℝ) (x : ℝ) : ℝ := x^2 - (m + n) * x + m * n

-- Define what it means for a polynomial to be displeased
def isDispleased (m n : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
  (∀ x : ℝ, p m n (p m n x) = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

-- Define the theorem
theorem displeased_polynomial_at_one_is_zero :
  ∃! (a : ℝ), isDispleased a a ∧
  (∀ m n : ℝ, isDispleased m n → m * n ≤ a * a) ∧
  p a a 1 = 0 :=
sorry

end displeased_polynomial_at_one_is_zero_l2411_241187


namespace distance_from_origin_implies_k_range_l2411_241181

theorem distance_from_origin_implies_k_range (k : ℝ) (h1 : k > 0) :
  (∃ x : ℝ, x ≠ 0 ∧ x^2 + (k/x)^2 = 1) → 0 < k ∧ k ≤ 1/2 := by
  sorry

end distance_from_origin_implies_k_range_l2411_241181


namespace negative_reciprocal_positive_l2411_241168

theorem negative_reciprocal_positive (x : ℝ) (h : x < 0) : -x⁻¹ > 0 := by
  sorry

end negative_reciprocal_positive_l2411_241168


namespace rahul_salary_calculation_l2411_241177

def calculate_remaining_salary (initial_salary : ℕ) : ℕ :=
  let after_rent := initial_salary - initial_salary * 20 / 100
  let after_education := after_rent - after_rent * 10 / 100
  let after_clothes := after_education - after_education * 10 / 100
  after_clothes

theorem rahul_salary_calculation :
  calculate_remaining_salary 2125 = 1377 := by
  sorry

end rahul_salary_calculation_l2411_241177


namespace stream_speed_l2411_241125

/-- Represents the speed of a boat in a stream -/
structure BoatSpeed where
  boatStillWater : ℝ  -- Speed of the boat in still water
  stream : ℝ          -- Speed of the stream

/-- Calculates the effective speed of the boat -/
def effectiveSpeed (b : BoatSpeed) (downstream : Bool) : ℝ :=
  if downstream then b.boatStillWater + b.stream else b.boatStillWater - b.stream

/-- Theorem: Given the conditions, the speed of the stream is 3 km/h -/
theorem stream_speed (b : BoatSpeed) 
  (h1 : effectiveSpeed b true * 4 = 84)  -- Downstream condition
  (h2 : effectiveSpeed b false * 4 = 60) -- Upstream condition
  : b.stream = 3 := by
  sorry


end stream_speed_l2411_241125


namespace light_travel_distance_l2411_241171

/-- The distance light travels in one year in kilometers -/
def light_year_distance : ℝ := 9460800000000

/-- The number of years we're calculating for -/
def years : ℝ := 70

/-- The expected distance light travels in the given number of years -/
def expected_distance : ℝ := 6.62256 * (10 ^ 14)

/-- Theorem stating that the distance light travels in the given number of years
    is equal to the expected distance -/
theorem light_travel_distance : light_year_distance * years = expected_distance := by
  sorry

end light_travel_distance_l2411_241171


namespace golden_ratio_solution_l2411_241138

theorem golden_ratio_solution (x : ℝ) :
  x > 0 ∧ x = Real.sqrt (x - 1 / x) + Real.sqrt (1 - 1 / x) ↔ x = (1 + Real.sqrt 5) / 2 := by
  sorry

end golden_ratio_solution_l2411_241138
