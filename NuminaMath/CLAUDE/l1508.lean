import Mathlib

namespace NUMINAMATH_CALUDE_min_probability_is_601_1225_l1508_150804

/-- The number of cards in the deck -/
def num_cards : ℕ := 52

/-- The probability that Charlie and Jane are on the same team, given that they draw cards a and a+11 -/
def p (a : ℕ) : ℚ :=
  let remaining_combinations := (num_cards - 2).choose 2
  let lower_team_combinations := (a - 1).choose 2
  let higher_team_combinations := (num_cards - (a + 11) - 1).choose 2
  (lower_team_combinations + higher_team_combinations : ℚ) / remaining_combinations

/-- The minimum value of a for which p(a) is at least 1/2 -/
def min_a : ℕ := 36

theorem min_probability_is_601_1225 :
  p min_a = 601 / 1225 ∧ ∀ a : ℕ, 1 ≤ a ∧ a ≤ num_cards - 11 → p a ≥ 1 / 2 → p a ≥ p min_a :=
sorry

end NUMINAMATH_CALUDE_min_probability_is_601_1225_l1508_150804


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1508_150807

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = -2 ∧ p.b = 0 ∧ p.c = 0 →
  (shift_parabola (shift_parabola p 1 0) 0 (-3)) = { a := -2, b := -4, c := -5 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1508_150807


namespace NUMINAMATH_CALUDE_ball_probabilities_l1508_150845

/-- Represents the color of a ball -/
inductive BallColor
  | Yellow
  | White

/-- Represents the bag with balls -/
structure Bag :=
  (yellow : ℕ)
  (white : ℕ)

/-- The probability of drawing a white ball from the bag -/
def prob_white (bag : Bag) : ℚ :=
  bag.white / (bag.yellow + bag.white)

/-- The probability of drawing a yellow ball from the bag -/
def prob_yellow (bag : Bag) : ℚ :=
  bag.yellow / (bag.yellow + bag.white)

/-- The probability that two drawn balls have the same color -/
def prob_same_color (bag : Bag) : ℚ :=
  (prob_yellow bag)^2 + (prob_white bag)^2

theorem ball_probabilities (bag : Bag) 
  (h1 : bag.yellow = 1) 
  (h2 : bag.white = 2) : 
  prob_white bag = 2/3 ∧ prob_same_color bag = 5/9 := by
  sorry

#check ball_probabilities

end NUMINAMATH_CALUDE_ball_probabilities_l1508_150845


namespace NUMINAMATH_CALUDE_smallest_positive_b_squared_l1508_150821

-- Define the circles w₁ and w₂
def w₁ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 6*y - 23 = 0
def w₂ (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 6*y + 41 = 0

-- Define the condition for a point (x, y) to be on the line y = bx
def on_line (x y b : ℝ) : Prop := y = b * x

-- Define the condition for a circle to be externally tangent to w₂ and internally tangent to w₁
def tangent_condition (x y r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    w₁ x₁ y₁ ∧ w₂ x₂ y₂ ∧
    (x - x₂)^2 + (y - y₂)^2 = (r + Real.sqrt 10)^2 ∧
    (x - x₁)^2 + (y - y₁)^2 = (Real.sqrt 50 - r)^2

-- Main theorem
theorem smallest_positive_b_squared (b : ℝ) :
  (∀ b' : ℝ, b' > 0 ∧ b' < b →
    ¬∃ (x y r : ℝ), on_line x y b' ∧ tangent_condition x y r) →
  (∃ (x y r : ℝ), on_line x y b ∧ tangent_condition x y r) →
  b^2 = 21/16 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_b_squared_l1508_150821


namespace NUMINAMATH_CALUDE_prob_non_first_class_l1508_150867

theorem prob_non_first_class (A B C : ℝ) 
  (hA : A = 0.65) 
  (hB : B = 0.2) 
  (hC : C = 0.1) : 
  1 - A = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_prob_non_first_class_l1508_150867


namespace NUMINAMATH_CALUDE_best_fitting_model_l1508_150839

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  r_squared : ℝ
  h_r_squared_nonneg : 0 ≤ r_squared
  h_r_squared_le_one : r_squared ≤ 1

/-- Given four regression models, proves that the model with the highest R² has the best fitting effect -/
theorem best_fitting_model
  (model1 model2 model3 model4 : RegressionModel)
  (h1 : model1.r_squared = 0.98)
  (h2 : model2.r_squared = 0.80)
  (h3 : model3.r_squared = 0.50)
  (h4 : model4.r_squared = 0.25) :
  model1.r_squared = max model1.r_squared (max model2.r_squared (max model3.r_squared model4.r_squared)) :=
sorry

end NUMINAMATH_CALUDE_best_fitting_model_l1508_150839


namespace NUMINAMATH_CALUDE_square_of_sum_23_2_l1508_150859

theorem square_of_sum_23_2 : 23^2 + 2*(23*2) + 2^2 = 625 := by sorry

end NUMINAMATH_CALUDE_square_of_sum_23_2_l1508_150859


namespace NUMINAMATH_CALUDE_book_weight_is_205_l1508_150863

/-- Calculates the weight of a single book given the following conditions:
  * 6 books in each small box
  * Small box weighs 220 grams
  * 9 small boxes in a large box
  * Large box weighs 250 grams
  * Total weight is 13.3 kilograms
  * All books weigh the same
-/
def bookWeight (booksPerSmallBox : ℕ) (smallBoxWeight : ℕ) (smallBoxCount : ℕ) 
                (largeBoxWeight : ℕ) (totalWeightKg : ℚ) : ℚ :=
  let totalWeightG : ℚ := totalWeightKg * 1000
  let smallBoxesWeight : ℚ := smallBoxWeight * smallBoxCount
  let booksWeight : ℚ := totalWeightG - largeBoxWeight - smallBoxesWeight
  let totalBooks : ℕ := booksPerSmallBox * smallBoxCount
  booksWeight / totalBooks

theorem book_weight_is_205 :
  bookWeight 6 220 9 250 (13.3 : ℚ) = 205 := by
  sorry

#eval bookWeight 6 220 9 250 (13.3 : ℚ)

end NUMINAMATH_CALUDE_book_weight_is_205_l1508_150863


namespace NUMINAMATH_CALUDE_marble_probability_l1508_150887

/-- Given a box of marbles with the following properties:
  - There are 120 marbles in total
  - Each marble is either red, green, blue, or white
  - The probability of drawing a white marble is 1/4
  - The probability of drawing a green marble is 1/3
  This theorem proves that the probability of drawing either a red or blue marble is 5/12. -/
theorem marble_probability (total_marbles : ℕ) (p_white p_green : ℚ)
  (h_total : total_marbles = 120)
  (h_white : p_white = 1/4)
  (h_green : p_green = 1/3) :
  1 - (p_white + p_green) = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l1508_150887


namespace NUMINAMATH_CALUDE_angle_halving_l1508_150830

/-- An angle is in the third quadrant if it's between π and 3π/2 (modulo 2π) -/
def is_third_quadrant (α : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2

/-- An angle is in the second or fourth quadrant if it's between π/2 and 3π/4 or between 3π/2 and 7π/4 (modulo 2π) -/
def is_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, (k * Real.pi + Real.pi / 2 < α ∧ α < k * Real.pi + 3 * Real.pi / 4) ∨
             (k * Real.pi + 3 * Real.pi / 2 < α ∧ α < k * Real.pi + 7 * Real.pi / 4)

theorem angle_halving (α : Real) :
  is_third_quadrant α → is_second_or_fourth_quadrant (α / 2) := by
  sorry

end NUMINAMATH_CALUDE_angle_halving_l1508_150830


namespace NUMINAMATH_CALUDE_custom_mult_three_two_l1508_150878

/-- Custom multiplication operation -/
def custom_mult (a b : ℤ) : ℤ := a^2 + a*b - b^2

/-- Theorem stating that 3*2 equals 11 under the custom multiplication -/
theorem custom_mult_three_two : custom_mult 3 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_three_two_l1508_150878


namespace NUMINAMATH_CALUDE_protest_days_calculation_l1508_150892

/-- Calculates the number of days of protest given the conditions of the problem. -/
def daysOfProtest (
  numCities : ℕ)
  (arrestsPerDay : ℕ)
  (preTrialDays : ℕ)
  (sentenceDays : ℕ)
  (totalJailWeeks : ℕ) : ℕ :=
  let totalJailDays := totalJailWeeks * 7
  let daysPerPerson := preTrialDays + sentenceDays / 2
  let totalArrests := totalJailDays / daysPerPerson
  let totalProtestDays := totalArrests / arrestsPerDay
  totalProtestDays / numCities

/-- Theorem stating that given the conditions of the problem, there were 30 days of protest. -/
theorem protest_days_calculation :
  daysOfProtest 21 10 4 14 9900 = 30 := by
  sorry

end NUMINAMATH_CALUDE_protest_days_calculation_l1508_150892


namespace NUMINAMATH_CALUDE_abs_sum_equals_two_l1508_150865

theorem abs_sum_equals_two (a b c : ℤ) 
  (h : |a - b|^19 + |c - a|^2010 = 1) : 
  |a - b| + |b - c| + |c - a| = 2 := by
sorry

end NUMINAMATH_CALUDE_abs_sum_equals_two_l1508_150865


namespace NUMINAMATH_CALUDE_min_value_sum_squares_over_sum_l1508_150893

theorem min_value_sum_squares_over_sum (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → a + b + c = 9 →
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) ≥ 9 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_over_sum_l1508_150893


namespace NUMINAMATH_CALUDE_ladder_length_l1508_150800

/-- The length of a ladder given specific conditions --/
theorem ladder_length : ∃ (L : ℝ), 
  (∀ (H : ℝ), L^2 = H^2 + 5^2) ∧ 
  (∀ (H : ℝ), L^2 = (H - 4)^2 + 10.658966865741546^2) ∧
  (abs (L - 14.04) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ladder_length_l1508_150800


namespace NUMINAMATH_CALUDE_sum_first_four_eq_40_l1508_150829

/-- A geometric sequence with a_2 = 6 and a_3 = -18 -/
def geometric_sequence (n : ℕ) : ℝ :=
  let q := -3  -- common ratio
  let a1 := -2 -- first term
  a1 * q^(n-1)

/-- The sum of the first four terms of the geometric sequence -/
def sum_first_four : ℝ :=
  (geometric_sequence 1) + (geometric_sequence 2) + (geometric_sequence 3) + (geometric_sequence 4)

/-- Theorem stating that the sum of the first four terms equals 40 -/
theorem sum_first_four_eq_40 : sum_first_four = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_four_eq_40_l1508_150829


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_sector_central_angle_l1508_150872

-- Problem 1
theorem simplify_trigonometric_expression (x : ℝ) :
  (1 + Real.sin x) / Real.cos x * Real.sin (2 * x) / (2 * (Real.cos (π / 4 - x / 2))^2) = 2 * Real.sin x :=
sorry

-- Problem 2
theorem sector_central_angle (r α : ℝ) (h1 : 2 * r + α * r = 4) (h2 : 1/2 * α * r^2 = 1) :
  α = 2 :=
sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_sector_central_angle_l1508_150872


namespace NUMINAMATH_CALUDE_dried_mushroom_mass_dried_mushroom_mass_44kg_l1508_150818

/-- Given fresh mushrooms with 90% water content and dried mushrooms with 12% water content,
    calculate the mass of dried mushrooms obtained from a given mass of fresh mushrooms. -/
theorem dried_mushroom_mass (fresh_mass : ℝ) : 
  fresh_mass > 0 →
  (fresh_mass * (1 - 0.9)) / (1 - 0.12) = 5 →
  fresh_mass = 44 := by
sorry

/-- The mass of dried mushrooms obtained from 44 kg of fresh mushrooms is 5 kg. -/
theorem dried_mushroom_mass_44kg : 
  (44 * (1 - 0.9)) / (1 - 0.12) = 5 := by
sorry

end NUMINAMATH_CALUDE_dried_mushroom_mass_dried_mushroom_mass_44kg_l1508_150818


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l1508_150896

theorem fourth_rectangle_area (P Q R S : ℝ × ℝ) : 
  (R.1 - P.1)^2 + (R.2 - P.2)^2 = 25 →
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 49 →
  (S.1 - R.1)^2 + (S.2 - R.2)^2 = 64 →
  (Q.2 - P.2) * (R.1 - P.1) = (Q.1 - P.1) * (R.2 - P.2) →
  (S.2 - P.2) * (R.1 - P.1) = (S.1 - P.1) * (R.2 - P.2) →
  (S.1 - P.1)^2 + (S.2 - P.2)^2 = 89 := by
sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_l1508_150896


namespace NUMINAMATH_CALUDE_town_employment_theorem_l1508_150810

/-- Represents the employment statistics of town X -/
structure TownEmployment where
  total_population : ℝ
  employed_percentage : ℝ
  employed_male_percentage : ℝ
  employed_female_percentage : ℝ

/-- The employment theorem for town X -/
theorem town_employment_theorem (stats : TownEmployment) 
  (h1 : stats.employed_male_percentage = 24)
  (h2 : stats.employed_female_percentage = 75) :
  stats.employed_percentage = 96 := by
  sorry

end NUMINAMATH_CALUDE_town_employment_theorem_l1508_150810


namespace NUMINAMATH_CALUDE_circle_properties_1_circle_properties_2_l1508_150879

/-- Definition of a circle in the xy-plane -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  eq : ∀ x y : ℝ, x^2 + y^2 + a*x + b*y + c = 0

/-- The center and radius of a circle -/
structure CircleProperties where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem for the first circle -/
theorem circle_properties_1 :
  let C : Circle := {
    a := 2
    b := -4
    c := -3
    d := 1
    e := 1
    eq := by sorry
  }
  ∃ (props : CircleProperties), props.center = (-1, 2) ∧ props.radius = 2 * Real.sqrt 2 := by sorry

/-- Theorem for the second circle -/
theorem circle_properties_2 (m : ℝ) :
  let C : Circle := {
    a := 2*m
    b := 0
    c := 0
    d := 1
    e := 1
    eq := by sorry
  }
  ∃ (props : CircleProperties), props.center = (-m, 0) ∧ props.radius = |m| := by sorry

end NUMINAMATH_CALUDE_circle_properties_1_circle_properties_2_l1508_150879


namespace NUMINAMATH_CALUDE_bacteria_growth_proof_l1508_150861

/-- The growth factor of the bacteria colony per day -/
def growth_factor : ℕ := 3

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := 5

/-- The threshold number of bacteria -/
def threshold : ℕ := 200

/-- The number of bacteria after n days -/
def bacteria_count (n : ℕ) : ℕ := initial_bacteria * growth_factor ^ n

/-- The smallest number of days for the bacteria count to exceed the threshold -/
def days_to_exceed_threshold : ℕ := 4

theorem bacteria_growth_proof :
  (∀ k : ℕ, k < days_to_exceed_threshold → bacteria_count k ≤ threshold) ∧
  bacteria_count days_to_exceed_threshold > threshold :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_proof_l1508_150861


namespace NUMINAMATH_CALUDE_warehouse_chocolate_count_l1508_150884

/-- The number of large boxes in the warehouse -/
def num_large_boxes : ℕ := 150

/-- The number of small boxes in each large box -/
def small_boxes_per_large : ℕ := 45

/-- The number of chocolate bars in each small box -/
def chocolates_per_small : ℕ := 35

/-- The total number of chocolate bars in the warehouse -/
def total_chocolates : ℕ := num_large_boxes * small_boxes_per_large * chocolates_per_small

theorem warehouse_chocolate_count :
  total_chocolates = 236250 :=
by sorry

end NUMINAMATH_CALUDE_warehouse_chocolate_count_l1508_150884


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l1508_150802

/-- Simple interest calculation -/
theorem simple_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℝ) 
  (h1 : principal = 10000)
  (h2 : rate = 0.04)
  (h3 : time = 1) :
  principal * rate * time = 400 := by
  sorry

#check simple_interest_calculation

end NUMINAMATH_CALUDE_simple_interest_calculation_l1508_150802


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1508_150860

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 4}
def B : Set Nat := {2, 3}

theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {2} :=
sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1508_150860


namespace NUMINAMATH_CALUDE_min_sum_with_product_144_l1508_150868

theorem min_sum_with_product_144 :
  (∃ (a b : ℤ), a * b = 144 ∧ a + b = -145) ∧
  (∀ (a b : ℤ), a * b = 144 → a + b ≥ -145) := by
sorry

end NUMINAMATH_CALUDE_min_sum_with_product_144_l1508_150868


namespace NUMINAMATH_CALUDE_smallest_prime_cube_sum_fourth_power_l1508_150849

theorem smallest_prime_cube_sum_fourth_power :
  ∃ (p : ℕ), Prime p ∧ 
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a^2 + p^3 = b^4) ∧
  (∀ (q : ℕ), Prime q → q < p → 
    ¬∃ (c d : ℕ), c > 0 ∧ d > 0 ∧ c^2 + q^3 = d^4) ∧
  p = 23 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_cube_sum_fourth_power_l1508_150849


namespace NUMINAMATH_CALUDE_mt_everest_summit_distance_l1508_150840

/-- The distance from the base camp to the summit of Mt. Everest --/
def summit_distance : ℝ := 5800

/-- Hillary's climbing rate in feet per hour --/
def hillary_climb_rate : ℝ := 800

/-- Eddy's climbing rate in feet per hour --/
def eddy_climb_rate : ℝ := 500

/-- Hillary's descent rate in feet per hour --/
def hillary_descent_rate : ℝ := 1000

/-- The distance in feet that Hillary stops short of the summit --/
def hillary_stop_distance : ℝ := 1000

/-- The time in hours from start until Hillary and Eddy pass each other --/
def time_until_pass : ℝ := 6

theorem mt_everest_summit_distance :
  summit_distance = 
    hillary_climb_rate * time_until_pass + hillary_stop_distance ∧
  summit_distance = 
    eddy_climb_rate * time_until_pass + 
    hillary_descent_rate * (time_until_pass - hillary_climb_rate * time_until_pass / hillary_descent_rate) :=
by sorry

end NUMINAMATH_CALUDE_mt_everest_summit_distance_l1508_150840


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1508_150805

/-- Given a hyperbola with equation x²/2 - y² = 1, prove its asymptotes and eccentricity -/
theorem hyperbola_properties :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 / 2 - y^2 = 1
  ∃ (a b c : ℝ),
    a^2 = 2 ∧ 
    b^2 = 1 ∧
    c^2 = a^2 + b^2 ∧
    (∀ x y, h x y ↔ x^2 / a^2 - y^2 / b^2 = 1) ∧
    (∀ x, (h x (x * (b / a)) ∨ h x (-x * (b / a))) ↔ x ≠ 0) ∧
    c / a = Real.sqrt 6 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1508_150805


namespace NUMINAMATH_CALUDE_product_of_fractions_l1508_150815

theorem product_of_fractions : (1 : ℚ) / 3 * 4 / 7 * 9 / 11 = 12 / 77 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l1508_150815


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_31_l1508_150888

/-- Represents a quadrilateral with vertices A, B, C, D and intersection point O of diagonals -/
structure Quadrilateral :=
  (A B C D O : ℝ × ℝ)

/-- The area of a quadrilateral given its side lengths and angle between diagonals -/
def area_quadrilateral (q : Quadrilateral) (AB BC CD DA : ℝ) (angle_COB : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the area of the given quadrilateral is 31 -/
theorem quadrilateral_area_is_31 (q : Quadrilateral) 
  (h1 : area_quadrilateral q 10 6 8 2 (π/4) = 31) : 
  area_quadrilateral q 10 6 8 2 (π/4) = 31 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_31_l1508_150888


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_sum_l1508_150876

/-- A monic cubic polynomial -/
def monicCubicPolynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, p x = x^3 + a*x^2 + b*x + c

/-- The main theorem -/
theorem monic_cubic_polynomial_sum (p : ℝ → ℝ) 
  (h_monic : monicCubicPolynomial p)
  (h1 : p 1 = 10)
  (h2 : p 2 = 20)
  (h3 : p 3 = 30) :
  p 0 + p 5 = 68 := by sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_sum_l1508_150876


namespace NUMINAMATH_CALUDE_donation_amount_l1508_150809

def barbara_stuffed_animals : ℕ := 9
def trish_stuffed_animals : ℕ := 2 * barbara_stuffed_animals
def sam_stuffed_animals : ℕ := barbara_stuffed_animals + 5

def barbara_price : ℚ := 2
def trish_price : ℚ := (3 : ℚ) / 2
def sam_price : ℚ := (5 : ℚ) / 2

def total_donation : ℚ := 
  barbara_stuffed_animals * barbara_price + 
  trish_stuffed_animals * trish_price + 
  sam_stuffed_animals * sam_price

theorem donation_amount : total_donation = 80 := by
  sorry

end NUMINAMATH_CALUDE_donation_amount_l1508_150809


namespace NUMINAMATH_CALUDE_min_movements_for_ten_l1508_150862

/-- Represents a circular arrangement of n distinct elements -/
def CircularArrangement (n : ℕ) := Fin n → ℕ

/-- A single movement in the circular arrangement -/
def Movement (n : ℕ) (arr : CircularArrangement n) (i j : Fin n) : CircularArrangement n :=
  sorry

/-- Checks if the circular arrangement is sorted in ascending order clockwise -/
def IsSorted (n : ℕ) (arr : CircularArrangement n) : Prop :=
  sorry

/-- The minimum number of movements required to sort the arrangement -/
def MinMovements (n : ℕ) (arr : CircularArrangement n) : ℕ :=
  sorry

/-- Theorem: For 10 distinct elements, 8 movements are always sufficient and necessary -/
theorem min_movements_for_ten :
  ∀ (arr : CircularArrangement 10),
    (∀ i j : Fin 10, i ≠ j → arr i ≠ arr j) →
    MinMovements 10 arr = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_movements_for_ten_l1508_150862


namespace NUMINAMATH_CALUDE_parallel_iff_parallel_sum_l1508_150877

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def IsParallel (u v : V) : Prop :=
  ∃ (k : ℝ), v = k • u ∨ u = k • v

theorem parallel_iff_parallel_sum {a b : V} (ha : a ≠ 0) (hb : b ≠ 0) :
  IsParallel a b ↔ IsParallel a (a + b) :=
sorry

end NUMINAMATH_CALUDE_parallel_iff_parallel_sum_l1508_150877


namespace NUMINAMATH_CALUDE_perfect_pairing_S8_exists_no_perfect_pairing_S5_l1508_150822

def Sn (n : ℕ) : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2*n}

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_pairing (n : ℕ) (pairing : List (ℕ × ℕ)) : Prop :=
  (∀ (pair : ℕ × ℕ), pair ∈ pairing → pair.1 ∈ Sn n ∧ pair.2 ∈ Sn n) ∧
  (∀ x ∈ Sn n, ∃ pair ∈ pairing, x = pair.1 ∨ x = pair.2) ∧
  (∀ pair ∈ pairing, is_perfect_square (pair.1 + pair.2)) ∧
  pairing.length = n

theorem perfect_pairing_S8_exists : ∃ pairing : List (ℕ × ℕ), is_perfect_pairing 8 pairing :=
sorry

theorem no_perfect_pairing_S5 : ¬∃ pairing : List (ℕ × ℕ), is_perfect_pairing 5 pairing :=
sorry

end NUMINAMATH_CALUDE_perfect_pairing_S8_exists_no_perfect_pairing_S5_l1508_150822


namespace NUMINAMATH_CALUDE_kayla_apples_l1508_150816

theorem kayla_apples (total : ℕ) (kylie : ℕ) (kayla : ℕ) : 
  total = 340 →
  kayla = 4 * kylie + 10 →
  total = kylie + kayla →
  kayla = 274 := by
sorry

end NUMINAMATH_CALUDE_kayla_apples_l1508_150816


namespace NUMINAMATH_CALUDE_ryan_coin_value_l1508_150880

/-- Represents the types of coins Ryan has --/
inductive Coin
| Penny
| Nickel

/-- The value of a coin in cents --/
def coinValue : Coin → Nat
| Coin.Penny => 1
| Coin.Nickel => 5

/-- Ryan's coin collection --/
structure CoinCollection where
  pennies : Nat
  nickels : Nat
  total_coins : pennies + nickels = 17
  equal_count : pennies = nickels

theorem ryan_coin_value (c : CoinCollection) : 
  c.pennies * coinValue Coin.Penny + c.nickels * coinValue Coin.Nickel = 49 := by
  sorry

#check ryan_coin_value

end NUMINAMATH_CALUDE_ryan_coin_value_l1508_150880


namespace NUMINAMATH_CALUDE_coronavirus_cases_l1508_150889

theorem coronavirus_cases (initial_cases : ℕ) : 
  initial_cases > 0 →
  initial_cases + 450 + 1300 = 3750 →
  initial_cases = 2000 := by
sorry

end NUMINAMATH_CALUDE_coronavirus_cases_l1508_150889


namespace NUMINAMATH_CALUDE_probability_is_one_twelfth_l1508_150846

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- The probability of a point satisfying certain conditions within a rectangle --/
def probability_in_rectangle (R : Rectangle) (P : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The specific rectangle in the problem --/
def problem_rectangle : Rectangle := {
  x_min := 0
  x_max := 6
  y_min := 0
  y_max := 2
  h_x := by norm_num
  h_y := by norm_num
}

/-- The condition that needs to be satisfied --/
def condition (p : ℝ × ℝ) : Prop :=
  p.1 < p.2 ∧ p.1 + p.2 < 2

/-- The main theorem --/
theorem probability_is_one_twelfth :
  probability_in_rectangle problem_rectangle condition = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_twelfth_l1508_150846


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_ages_l1508_150823

theorem arithmetic_mean_of_ages : 
  let ages : List ℝ := [18, 27, 35, 46]
  (ages.sum / ages.length : ℝ) = 31.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_ages_l1508_150823


namespace NUMINAMATH_CALUDE_pink_yards_calculation_l1508_150852

/-- The total number of yards dyed for the order -/
def total_yards : ℕ := 111421

/-- The number of yards dyed green -/
def green_yards : ℕ := 61921

/-- The number of yards dyed pink -/
def pink_yards : ℕ := total_yards - green_yards

theorem pink_yards_calculation : pink_yards = 49500 := by
  sorry

end NUMINAMATH_CALUDE_pink_yards_calculation_l1508_150852


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_threes_l1508_150857

/-- Given a natural number n, this function returns the number composed of 2n digits of 1 -/
def two_n_ones (n : ℕ) : ℕ := (10^(2*n) - 1) / 9

/-- Given a natural number n, this function returns the number composed of n digits of 2 -/
def n_twos (n : ℕ) : ℕ := 2 * ((10^n - 1) / 9)

/-- Given a natural number n, this function returns the number composed of n digits of 3 -/
def n_threes (n : ℕ) : ℕ := (10^n - 1) / 3

theorem sqrt_difference_equals_threes (n : ℕ) : 
  Real.sqrt (two_n_ones n - n_twos n) = n_threes n := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_threes_l1508_150857


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1508_150869

def a (m : ℝ) : ℝ × ℝ := (1, m + 2)
def b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem parallel_vectors_magnitude (m : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ a m = k • b m) →
  Real.sqrt ((b m).1^2 + (b m).2^2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l1508_150869


namespace NUMINAMATH_CALUDE_range_of_a_l1508_150801

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + 4 * x + a > 0) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1508_150801


namespace NUMINAMATH_CALUDE_tangent_difference_l1508_150828

noncomputable section

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := a * x + 2 * Real.log (abs x)

-- Define the tangent line
def tangent_line (k : ℝ) (x : ℝ) : ℝ := k * x

-- Define the property of being a tangent line to the curve
def is_tangent (a k : ℝ) : Prop :=
  ∃ x : ℝ, x ≠ 0 ∧ curve a x = tangent_line k x ∧
    (∀ y : ℝ, y ≠ x → curve a y ≠ tangent_line k y)

-- Theorem statement
theorem tangent_difference (a k₁ k₂ : ℝ) :
  is_tangent a k₁ → is_tangent a k₂ → k₁ > k₂ → k₁ - k₂ = 4 / Real.exp 1 := by
  sorry

end

end NUMINAMATH_CALUDE_tangent_difference_l1508_150828


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_over_one_minus_i_l1508_150881

theorem imaginary_part_of_one_over_one_minus_i :
  Complex.im (1 / (1 - Complex.I)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_over_one_minus_i_l1508_150881


namespace NUMINAMATH_CALUDE_expression_value_l1508_150897

theorem expression_value : 
  2 * Real.tan (60 * π / 180) - (1/3)⁻¹ + (-2)^2 * (2017 - Real.sin (45 * π / 180))^0 - |-(12: ℝ).sqrt| = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1508_150897


namespace NUMINAMATH_CALUDE_extended_line_point_l1508_150850

-- Define points A and B
def A : ℝ × ℝ := (3, 1)
def B : ℝ × ℝ := (17, 7)

-- Define the ratio of BC to AB
def ratio : ℚ := 2 / 5

-- Define point C
def C : ℝ × ℝ := (22.6, 9.4)

-- Theorem statement
theorem extended_line_point : 
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let BC : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
  BC.1 = ratio * AB.1 ∧ BC.2 = ratio * AB.2 := by sorry

end NUMINAMATH_CALUDE_extended_line_point_l1508_150850


namespace NUMINAMATH_CALUDE_special_rhombus_center_distance_l1508_150853

/-- A rhombus with a specific acute angle and projection length. -/
structure SpecialRhombus where
  /-- The acute angle of the rhombus in degrees -/
  acute_angle : ℝ
  /-- The length of the projection of side AB onto side AD -/
  projection_length : ℝ
  /-- The acute angle is 45 degrees -/
  angle_is_45 : acute_angle = 45
  /-- The projection length is 12 -/
  projection_is_12 : projection_length = 12

/-- The distance from the center of the rhombus to any side -/
def center_to_side_distance (r : SpecialRhombus) : ℝ := 6

/-- 
Theorem: In a rhombus where the acute angle is 45° and the projection of one side 
onto an adjacent side is 12, the distance from the center to any side is 6.
-/
theorem special_rhombus_center_distance (r : SpecialRhombus) : 
  center_to_side_distance r = 6 := by
  sorry

end NUMINAMATH_CALUDE_special_rhombus_center_distance_l1508_150853


namespace NUMINAMATH_CALUDE_die_roll_probability_l1508_150812

/-- The probability of rolling a 5 on a standard die -/
def prob_five : ℚ := 1/6

/-- The number of rolls -/
def num_rolls : ℕ := 8

/-- The probability of not rolling a 5 in a single roll -/
def prob_not_five : ℚ := 1 - prob_five

theorem die_roll_probability : 
  1 - prob_not_five ^ num_rolls = 1288991/1679616 := by
sorry

end NUMINAMATH_CALUDE_die_roll_probability_l1508_150812


namespace NUMINAMATH_CALUDE_system_solution_l1508_150831

theorem system_solution :
  ∃ (x y : ℚ), 
    (12 * x^2 + 4 * x * y + 3 * y^2 + 16 * x = -6) ∧
    (4 * x^2 - 12 * x * y + y^2 + 12 * x - 10 * y = -7) ∧
    (x = -3/4) ∧ (y = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1508_150831


namespace NUMINAMATH_CALUDE_min_value_theorem_l1508_150806

theorem min_value_theorem (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) :
  ∃ (min : ℝ), min = 1 ∧ ∀ (z : ℝ), z = (1 / (x + y)^2) + (1 / (x - y)^2) → z ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1508_150806


namespace NUMINAMATH_CALUDE_target_has_six_more_tools_l1508_150851

/-- The number of tools in the Walmart multitool -/
def walmart_tools : ℕ := 1 + 3 + 2

/-- The number of tools in the Target multitool -/
def target_tools : ℕ := 1 + (2 * 3) + 3 + 1

/-- The difference in the number of tools between Target and Walmart multitools -/
def tool_difference : ℕ := target_tools - walmart_tools

theorem target_has_six_more_tools : tool_difference = 6 := by
  sorry

end NUMINAMATH_CALUDE_target_has_six_more_tools_l1508_150851


namespace NUMINAMATH_CALUDE_eighth_term_value_l1508_150891

theorem eighth_term_value (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h : ∀ n : ℕ, S n = n^2) : a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_value_l1508_150891


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_40_l1508_150855

theorem consecutive_integers_around_sqrt_40 (a b : ℤ) : 
  (a + 1 = b) → (a < Real.sqrt 40) → (Real.sqrt 40 < b) → (a + b = 13) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_40_l1508_150855


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l1508_150808

theorem imaginary_part_of_one_minus_i_squared :
  Complex.im ((1 - Complex.I) ^ 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l1508_150808


namespace NUMINAMATH_CALUDE_jason_borrowed_amount_l1508_150890

/-- Calculates the payment for a given hour based on the repeating pattern -/
def hourly_payment (hour : ℕ) : ℕ :=
  (hour - 1) % 6 + 1

/-- Calculates the total payment for a given number of hours -/
def total_payment (hours : ℕ) : ℕ :=
  (List.range hours).map hourly_payment |>.sum

/-- The problem statement -/
theorem jason_borrowed_amount :
  total_payment 39 = 132 := by
  sorry

end NUMINAMATH_CALUDE_jason_borrowed_amount_l1508_150890


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1508_150819

/-- For an infinite geometric series with common ratio 1/3 and sum 18, the first term is 12. -/
theorem infinite_geometric_series_first_term 
  (r : ℝ) 
  (S : ℝ) 
  (h1 : r = 1/3) 
  (h2 : S = 18) 
  (h3 : S = a / (1 - r)) 
  (a : ℝ) : 
  a = 12 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1508_150819


namespace NUMINAMATH_CALUDE_oil_bill_problem_l1508_150864

/-- The oil bill problem -/
theorem oil_bill_problem (january_bill : ℝ) (february_bill : ℝ) (additional_amount : ℝ) :
  january_bill = 180 →
  february_bill / january_bill = 5 / 4 →
  (february_bill + additional_amount) / january_bill = 3 / 2 →
  additional_amount = 45 := by
  sorry

end NUMINAMATH_CALUDE_oil_bill_problem_l1508_150864


namespace NUMINAMATH_CALUDE_number_greater_than_fifteen_l1508_150848

theorem number_greater_than_fifteen (x : ℝ) : 0.4 * x > 0.8 * 5 + 2 → x > 15 := by
  sorry

end NUMINAMATH_CALUDE_number_greater_than_fifteen_l1508_150848


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l1508_150814

theorem simplify_sqrt_sum : 
  Real.sqrt (12 + 6 * Real.sqrt 3) + Real.sqrt (12 - 6 * Real.sqrt 3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l1508_150814


namespace NUMINAMATH_CALUDE_root_product_sum_l1508_150842

theorem root_product_sum (a b c : ℂ) : 
  (5 * a^3 - 4 * a^2 + 15 * a - 12 = 0) →
  (5 * b^3 - 4 * b^2 + 15 * b - 12 = 0) →
  (5 * c^3 - 4 * c^2 + 15 * c - 12 = 0) →
  a * b + a * c + b * c = -3 := by
sorry

end NUMINAMATH_CALUDE_root_product_sum_l1508_150842


namespace NUMINAMATH_CALUDE_power_six_mod_72_l1508_150837

theorem power_six_mod_72 : 6^700 % 72 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_six_mod_72_l1508_150837


namespace NUMINAMATH_CALUDE_table_covered_area_l1508_150838

/-- Represents a rectangular paper strip -/
structure Strip where
  length : ℕ
  width : ℕ

/-- Calculates the area of a strip -/
def stripArea (s : Strip) : ℕ := s.length * s.width

/-- Represents the overlap between two strips -/
structure Overlap where
  width : ℕ
  length : ℕ

/-- Calculates the area of an overlap -/
def overlapArea (o : Overlap) : ℕ := o.width * o.length

theorem table_covered_area (strip1 strip2 strip3 : Strip)
  (overlap12 overlap13 overlap23 : Overlap)
  (h1 : strip1.length = 12)
  (h2 : strip2.length = 15)
  (h3 : strip3.length = 9)
  (h4 : strip1.width = 2)
  (h5 : strip2.width = 2)
  (h6 : strip3.width = 2)
  (h7 : overlap12.width = 2)
  (h8 : overlap12.length = 2)
  (h9 : overlap13.width = 1)
  (h10 : overlap13.length = 2)
  (h11 : overlap23.width = 1)
  (h12 : overlap23.length = 2) :
  stripArea strip1 + stripArea strip2 + stripArea strip3 -
  (overlapArea overlap12 + overlapArea overlap13 + overlapArea overlap23) = 64 := by
  sorry

end NUMINAMATH_CALUDE_table_covered_area_l1508_150838


namespace NUMINAMATH_CALUDE_point_move_upward_l1508_150841

def Point := ℝ × ℝ

def move_upward (p : Point) (units : ℝ) : Point :=
  (p.1, p.2 + units)

theorem point_move_upward (A B : Point) (h : ℝ) :
  A = (1, -2) →
  h = 1 →
  B = move_upward A h →
  B = (1, -1) := by
  sorry

end NUMINAMATH_CALUDE_point_move_upward_l1508_150841


namespace NUMINAMATH_CALUDE_tom_dance_lesson_payment_l1508_150843

/-- The amount Tom pays for dance lessons -/
def tom_payment (total_lessons : ℕ) (cost_per_lesson : ℕ) (free_lessons : ℕ) : ℕ :=
  (total_lessons - free_lessons) * cost_per_lesson

/-- Proof that Tom pays $80 for dance lessons -/
theorem tom_dance_lesson_payment :
  tom_payment 10 10 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_tom_dance_lesson_payment_l1508_150843


namespace NUMINAMATH_CALUDE_distance_P_to_y_axis_l1508_150827

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate. -/
def distance_to_y_axis (x y : ℝ) : ℝ := |x|

/-- The point P has coordinates (3, -5). -/
def P : ℝ × ℝ := (3, -5)

/-- Theorem: The distance from point P(3, -5) to the y-axis is 3. -/
theorem distance_P_to_y_axis : distance_to_y_axis P.1 P.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_P_to_y_axis_l1508_150827


namespace NUMINAMATH_CALUDE_harold_wrapping_cost_l1508_150856

/-- Represents the number of shirt boxes that can be wrapped with one roll of paper -/
def shirt_boxes_per_roll : ℕ := 5

/-- Represents the number of XL boxes that can be wrapped with one roll of paper -/
def xl_boxes_per_roll : ℕ := 3

/-- Represents the number of shirt boxes Harold needs to wrap -/
def harold_shirt_boxes : ℕ := 20

/-- Represents the number of XL boxes Harold needs to wrap -/
def harold_xl_boxes : ℕ := 12

/-- Represents the cost of one roll of wrapping paper in cents -/
def cost_per_roll : ℕ := 400

/-- Theorem stating that Harold will spend $32.00 to wrap all boxes -/
theorem harold_wrapping_cost : 
  (((harold_shirt_boxes + shirt_boxes_per_roll - 1) / shirt_boxes_per_roll) + 
   ((harold_xl_boxes + xl_boxes_per_roll - 1) / xl_boxes_per_roll)) * 
  cost_per_roll = 3200 := by
  sorry

end NUMINAMATH_CALUDE_harold_wrapping_cost_l1508_150856


namespace NUMINAMATH_CALUDE_min_circles_cover_square_l1508_150885

/-- A circle with radius 1 -/
structure UnitCircle where
  center : ℝ × ℝ

/-- A square with side length 2 -/
structure TwoSquare where
  bottomLeft : ℝ × ℝ

/-- A covering of a TwoSquare by UnitCircles -/
structure Covering where
  circles : List UnitCircle
  square : TwoSquare
  covers : ∀ (x y : ℝ), 
    (x - square.bottomLeft.1 ∈ Set.Icc 0 2 ∧ 
     y - square.bottomLeft.2 ∈ Set.Icc 0 2) → 
    ∃ (c : UnitCircle), c ∈ circles ∧ 
      (x - c.center.1)^2 + (y - c.center.2)^2 ≤ 1

/-- The theorem stating that the minimum number of unit circles 
    needed to cover a 2x2 square is 4 -/
theorem min_circles_cover_square :
  ∀ (cov : Covering), cov.circles.length ≥ 4 ∧ 
  ∃ (cov' : Covering), cov'.circles.length = 4 := by
  sorry


end NUMINAMATH_CALUDE_min_circles_cover_square_l1508_150885


namespace NUMINAMATH_CALUDE_next_simultaneous_event_l1508_150854

/-- Represents the number of minutes between events for a clock -/
structure ClockEvents where
  lightup : ℕ  -- Number of minutes between light-ups
  ring : ℕ     -- Number of minutes between rings

/-- Calculates the time until the next simultaneous light-up and ring -/
def timeToNextSimultaneousEvent (c : ClockEvents) : ℕ :=
  Nat.lcm c.lightup c.ring

/-- The theorem stating that for a clock that lights up every 9 minutes
    and rings every 60 minutes, the next simultaneous event occurs after 180 minutes -/
theorem next_simultaneous_event :
  let c := ClockEvents.mk 9 60
  timeToNextSimultaneousEvent c = 180 := by
  sorry

end NUMINAMATH_CALUDE_next_simultaneous_event_l1508_150854


namespace NUMINAMATH_CALUDE_manuscript_revision_l1508_150811

/-- Proves that the number of pages revised twice is 15, given the manuscript typing conditions --/
theorem manuscript_revision (total_pages : ℕ) (revised_once : ℕ) (total_cost : ℕ) 
  (first_typing_cost : ℕ) (revision_cost : ℕ) :
  total_pages = 100 →
  revised_once = 35 →
  total_cost = 860 →
  first_typing_cost = 6 →
  revision_cost = 4 →
  ∃ (revised_twice : ℕ),
    revised_twice = 15 ∧
    total_cost = (total_pages - revised_once - revised_twice) * first_typing_cost +
                 revised_once * (first_typing_cost + revision_cost) +
                 revised_twice * (first_typing_cost + 2 * revision_cost) :=
by sorry


end NUMINAMATH_CALUDE_manuscript_revision_l1508_150811


namespace NUMINAMATH_CALUDE_multiply_72515_9999_l1508_150826

theorem multiply_72515_9999 : 72515 * 9999 = 725077485 := by sorry

end NUMINAMATH_CALUDE_multiply_72515_9999_l1508_150826


namespace NUMINAMATH_CALUDE_square_root_of_1024_l1508_150847

theorem square_root_of_1024 (y : ℝ) (h1 : y > 0) (h2 : y^2 = 1024) : y = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_1024_l1508_150847


namespace NUMINAMATH_CALUDE_intersection_point_on_fixed_line_l1508_150833

-- Define the hyperbola C
structure Hyperbola where
  center : ℝ × ℝ
  left_focus : ℝ × ℝ
  eccentricity : ℝ
  left_vertex : ℝ × ℝ
  right_vertex : ℝ × ℝ

-- Define the intersection points M and N
structure IntersectionPoints where
  M : ℝ × ℝ
  N : ℝ × ℝ

-- Define the point P
def P (h : Hyperbola) (i : IntersectionPoints) : ℝ × ℝ := sorry

-- Theorem statement
theorem intersection_point_on_fixed_line 
  (h : Hyperbola) 
  (i : IntersectionPoints) :
  h.center = (0, 0) →
  h.left_focus = (-2 * Real.sqrt 5, 0) →
  h.eccentricity = Real.sqrt 5 →
  h.left_vertex = (-2, 0) →
  h.right_vertex = (2, 0) →
  (∃ (m : ℝ), i.M.1 = m * i.M.2 - 4 ∧ i.N.1 = m * i.N.2 - 4) →
  i.M.2 > 0 →
  (P h i).1 = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_point_on_fixed_line_l1508_150833


namespace NUMINAMATH_CALUDE_no_specific_m_value_l1508_150858

theorem no_specific_m_value (m : ℝ) (z₁ z₂ : ℂ) 
  (h₁ : z₁ = m + 2*I) 
  (h₂ : z₂ = 3 - 4*I) : 
  ∀ (n : ℝ), ∃ (m' : ℝ), m' ≠ n ∧ z₁ = m' + 2*I :=
sorry

end NUMINAMATH_CALUDE_no_specific_m_value_l1508_150858


namespace NUMINAMATH_CALUDE_guess_who_i_am_l1508_150886

theorem guess_who_i_am : ∃ x y : ℕ,
  120 = 4 * x ∧
  87 = y - 40 ∧
  x = 30 ∧
  y = 127 := by
sorry

end NUMINAMATH_CALUDE_guess_who_i_am_l1508_150886


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l1508_150832

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

-- Define an increasing sequence
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) > a n

-- Define the condition 8a_2 - a_5 = 0
def condition (a : ℕ → ℝ) : Prop :=
  8 * a 2 - a 5 = 0

-- Theorem stating that the condition is neither sufficient nor necessary
theorem condition_neither_sufficient_nor_necessary :
  ¬(∀ (a : ℕ → ℝ), geometric_sequence a → condition a → increasing_sequence a) ∧
  ¬(∀ (a : ℕ → ℝ), geometric_sequence a → increasing_sequence a → condition a) :=
sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l1508_150832


namespace NUMINAMATH_CALUDE_x_prime_condition_x_divisibility_l1508_150895

def x : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => x (n + 1) + x n

theorem x_prime_condition (n : ℕ) (h : n ≥ 1) :
  Nat.Prime (x n) → Nat.Prime n ∨ (∀ p, Nat.Prime p → p > 2 → ¬p ∣ n) :=
sorry

theorem x_divisibility (m n : ℕ) :
  x m ∣ x n ↔ (∃ k, (m = 0 ∧ n = 3 * k) ∨ (m = 1 ∧ n = k) ∨ (∃ t, m = n ∧ n = (2 * t + 1) * n)) :=
sorry

end NUMINAMATH_CALUDE_x_prime_condition_x_divisibility_l1508_150895


namespace NUMINAMATH_CALUDE_triangle_sequence_solution_l1508_150817

theorem triangle_sequence_solution (b d c k : ℤ) 
  (h1 : b % d = 0)
  (h2 : c % k = 0)
  (h3 : b^2 + (b+2*d)^2 = (c+6*k)^2) :
  ∃ (b d c k : ℤ), c = 0 ∧ 
    b % d = 0 ∧ 
    c % k = 0 ∧ 
    b^2 + (b+2*d)^2 = (c+6*k)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sequence_solution_l1508_150817


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1508_150875

/-- A quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The function is decreasing on the interval (-∞, 4] -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 4 → f a x > f a y

/-- The range of values for a -/
def a_range (a : ℝ) : Prop := a ≤ -3

theorem quadratic_function_theorem (a : ℝ) :
  is_decreasing_on_interval a → a_range a :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1508_150875


namespace NUMINAMATH_CALUDE_probability_two_girls_l1508_150866

def total_students : ℕ := 5
def num_boys : ℕ := 2
def num_girls : ℕ := 3
def students_selected : ℕ := 2

theorem probability_two_girls :
  (Nat.choose num_girls students_selected) / (Nat.choose total_students students_selected) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_girls_l1508_150866


namespace NUMINAMATH_CALUDE_banana_cantaloupe_cost_l1508_150824

/-- Represents the prices of fruits in dollars -/
structure FruitPrices where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ

/-- The conditions of the fruit purchase problem -/
def fruitPurchaseConditions (p : FruitPrices) : Prop :=
  p.apples + p.bananas + p.cantaloupe + p.dates = 30 ∧
  p.dates = 3 * p.apples ∧
  p.cantaloupe = p.apples - p.bananas

/-- The theorem stating the cost of bananas and cantaloupe -/
theorem banana_cantaloupe_cost (p : FruitPrices) 
  (h : fruitPurchaseConditions p) : 
  p.bananas + p.cantaloupe = 6 := by
  sorry


end NUMINAMATH_CALUDE_banana_cantaloupe_cost_l1508_150824


namespace NUMINAMATH_CALUDE_constant_term_is_99_l1508_150894

-- Define the function q'
def q' (q : ℝ) (c : ℝ) : ℝ := 3 * q - 3 + c

-- Define the condition that (5')' = 132
axiom condition : q' (q' 5 99) 99 = 132

-- Theorem to prove
theorem constant_term_is_99 : ∃ c : ℝ, q' (q' 5 c) c = 132 ∧ c = 99 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_is_99_l1508_150894


namespace NUMINAMATH_CALUDE_sum_fourth_powers_of_roots_l1508_150813

/-- Given a cubic polynomial x^3 - x^2 + x - 3 = 0 with roots p, q, and r,
    prove that p^4 + q^4 + r^4 = 11 -/
theorem sum_fourth_powers_of_roots (p q r : ℂ) : 
  p^3 - p^2 + p - 3 = 0 → 
  q^3 - q^2 + q - 3 = 0 → 
  r^3 - r^2 + r - 3 = 0 → 
  p^4 + q^4 + r^4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_fourth_powers_of_roots_l1508_150813


namespace NUMINAMATH_CALUDE_smallest_side_of_triangle_l1508_150899

/-- Given a triangle ABC with ∠A = 60°, ∠C = 45°, and side b = 4,
    prove that the smallest side of the triangle is 4√3 - 4. -/
theorem smallest_side_of_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  A = 60 * π / 180 →
  C = 45 * π / 180 →
  b = 4 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  A + B + C = π →
  c / Real.sin C = b / Real.sin B →
  c = 4 * Real.sqrt 3 - 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_side_of_triangle_l1508_150899


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_294_490_l1508_150835

theorem lcm_gcf_ratio_294_490 : 
  (Nat.lcm 294 490) / (Nat.gcd 294 490) = 15 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_294_490_l1508_150835


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1508_150874

theorem sufficient_not_necessary (x : ℝ) : 
  (|x - 1| < 2 → x < 3) ∧ ¬(x < 3 → |x - 1| < 2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1508_150874


namespace NUMINAMATH_CALUDE_rectangle_area_l1508_150883

/-- Proves that a rectangle with width 4 inches and perimeter 30 inches has an area of 44 square inches -/
theorem rectangle_area (width : ℝ) (perimeter : ℝ) (height : ℝ) (area : ℝ) : 
  width = 4 →
  perimeter = 30 →
  perimeter = 2 * (width + height) →
  area = width * height →
  area = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1508_150883


namespace NUMINAMATH_CALUDE_no_simultaneous_divisibility_l1508_150825

theorem no_simultaneous_divisibility (k n : ℕ) (hk : k > 1) (hn : n > 1)
  (hk_odd : Odd k) (hn_odd : Odd n)
  (h_exists : ∃ a : ℕ, k ∣ 2^a + 1 ∧ n ∣ 2^a - 1) :
  ¬∃ b : ℕ, k ∣ 2^b - 1 ∧ n ∣ 2^b + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_divisibility_l1508_150825


namespace NUMINAMATH_CALUDE_distance_between_points_l1508_150803

theorem distance_between_points (A B : ℝ) : 
  (|A| = 2 ∧ |B| = 7) → (|A - B| = 5 ∨ |A - B| = 9) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1508_150803


namespace NUMINAMATH_CALUDE_find_V_l1508_150870

-- Define the relationship between U, V, and W
def relationship (k : ℝ) (U V W : ℝ) : Prop :=
  U = k * (V / W)

-- Define the theorem
theorem find_V (k : ℝ) :
  relationship k 16 2 (1/4) →
  relationship k 25 (5/2) (1/5) :=
by sorry

end NUMINAMATH_CALUDE_find_V_l1508_150870


namespace NUMINAMATH_CALUDE_john_climbs_nine_flights_l1508_150898

/-- The number of flights climbed given step height, flight height, and number of steps -/
def flights_climbed (step_height_inches : ℚ) (flight_height_feet : ℚ) (num_steps : ℕ) : ℚ :=
  (step_height_inches / 12 * num_steps) / flight_height_feet

/-- Theorem: John climbs 9 flights of stairs -/
theorem john_climbs_nine_flights :
  flights_climbed 18 10 60 = 9 := by
  sorry

end NUMINAMATH_CALUDE_john_climbs_nine_flights_l1508_150898


namespace NUMINAMATH_CALUDE_f_sum_opposite_l1508_150844

def f (x : ℝ) : ℝ := 5 * x^3

theorem f_sum_opposite : f 2012 + f (-2012) = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_opposite_l1508_150844


namespace NUMINAMATH_CALUDE_evaluate_trigonometric_expression_l1508_150873

theorem evaluate_trigonometric_expression :
  let angle_27 : Real := 27 * Real.pi / 180
  let angle_18 : Real := 18 * Real.pi / 180
  let angle_63 : Real := 63 * Real.pi / 180
  (Real.cos angle_63 = Real.sin angle_27) →
  (angle_27 = 45 * Real.pi / 180 - angle_18) →
  (Real.cos angle_27 - Real.sqrt 2 * Real.sin angle_18) / Real.cos angle_63 = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_trigonometric_expression_l1508_150873


namespace NUMINAMATH_CALUDE_painted_cubes_count_l1508_150820

/-- Given a cube with side length 4, composed of unit cubes, where the interior 2x2x2 cube is unpainted,
    the number of unit cubes with at least one face painted is 56. -/
theorem painted_cubes_count (n : ℕ) (h1 : n = 4) : 
  n^3 - (n - 2)^3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_count_l1508_150820


namespace NUMINAMATH_CALUDE_binary_101_equals_5_l1508_150882

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101_equals_5 :
  binary_to_decimal [true, false, true] = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_equals_5_l1508_150882


namespace NUMINAMATH_CALUDE_slope_of_AB_l1508_150834

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  P.1 = 1 ∧ P.2 = Real.sqrt 2 ∧ parabola P.1 P.2

-- Define complementary inclination angles
def complementary_angles (k : ℝ) (PA PB : ℝ → ℝ) : Prop :=
  (∀ x, PA x = k*(x - 1) + Real.sqrt 2) ∧
  (∀ x, PB x = -k*(x - 1) + Real.sqrt 2)

-- Define intersection points
def intersection_points (A B : ℝ × ℝ) (PA PB : ℝ → ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  A.2 = PA A.1 ∧ B.2 = PB B.1

-- Theorem statement
theorem slope_of_AB (P A B : ℝ × ℝ) (k : ℝ) (PA PB : ℝ → ℝ) :
  point_on_parabola P →
  complementary_angles k PA PB →
  intersection_points A B PA PB →
  (B.2 - A.2) / (B.1 - A.1) = -2 - 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_slope_of_AB_l1508_150834


namespace NUMINAMATH_CALUDE_water_overflow_l1508_150836

/-- Given a tap producing water at a constant rate and a water tank with a fixed capacity,
    calculate the amount of water that overflows after a certain time. -/
theorem water_overflow (flow_rate : ℕ) (time : ℕ) (tank_capacity : ℕ) : 
  flow_rate = 200 → time = 24 → tank_capacity = 4000 → 
  flow_rate * time - tank_capacity = 800 := by
  sorry

end NUMINAMATH_CALUDE_water_overflow_l1508_150836


namespace NUMINAMATH_CALUDE_greatest_b_value_l1508_150871

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 9*x - 18 ≥ 0 → x ≤ 6) ∧ 
  (-6^2 + 9*6 - 18 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l1508_150871
