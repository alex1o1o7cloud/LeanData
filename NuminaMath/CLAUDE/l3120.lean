import Mathlib

namespace NUMINAMATH_CALUDE_brennans_pepper_theorem_l3120_312047

/-- The amount of pepper remaining after using some from an initial amount -/
def pepper_remaining (initial : ℝ) (used : ℝ) : ℝ :=
  initial - used

/-- Theorem: Given 0.25 grams of pepper initially and using 0.16 grams, 
    the remaining amount is 0.09 grams -/
theorem brennans_pepper_theorem :
  pepper_remaining 0.25 0.16 = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_brennans_pepper_theorem_l3120_312047


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l3120_312050

theorem quadratic_real_root_condition (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l3120_312050


namespace NUMINAMATH_CALUDE_douglas_county_y_votes_l3120_312017

/-- Represents the percentage of votes Douglas won in county Y -/
def douglas_county_y_percentage : ℝ := 46

/-- Represents the total percentage of votes Douglas won in both counties -/
def total_percentage : ℝ := 58

/-- Represents the percentage of votes Douglas won in county X -/
def douglas_county_x_percentage : ℝ := 64

/-- Represents the ratio of voters in county X to county Y -/
def county_ratio : ℚ := 2 / 1

theorem douglas_county_y_votes :
  douglas_county_y_percentage = 
    (3 * total_percentage - 2 * douglas_county_x_percentage) := by sorry

end NUMINAMATH_CALUDE_douglas_county_y_votes_l3120_312017


namespace NUMINAMATH_CALUDE_mothers_age_l3120_312085

-- Define variables for current ages
variable (A : ℕ) -- Allen's current age
variable (M : ℕ) -- Mother's current age
variable (S : ℕ) -- Sister's current age

-- Define the conditions
axiom allen_younger : A = M - 30
axiom sister_older : S = A + 5
axiom future_sum : (A + 7) + (M + 7) + (S + 7) = 110
axiom mother_sister_diff : M - S = 25

-- Theorem to prove
theorem mothers_age : M = 48 := by
  sorry

end NUMINAMATH_CALUDE_mothers_age_l3120_312085


namespace NUMINAMATH_CALUDE_estimate_value_l3120_312065

theorem estimate_value : 6 < (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 ∧ 
                         (2 * Real.sqrt 2 + Real.sqrt 3) * Real.sqrt 2 < 7 := by
  sorry

end NUMINAMATH_CALUDE_estimate_value_l3120_312065


namespace NUMINAMATH_CALUDE_art_gallery_pieces_l3120_312078

theorem art_gallery_pieces (total : ℕ) 
  (displayed : ℕ) (sculptures_displayed : ℕ) 
  (paintings_not_displayed : ℕ) (sculptures_not_displayed : ℕ) :
  displayed = total / 3 →
  sculptures_displayed = displayed / 6 →
  paintings_not_displayed = (total - displayed) / 3 →
  sculptures_not_displayed = 1400 →
  total = 3150 := by
sorry

end NUMINAMATH_CALUDE_art_gallery_pieces_l3120_312078


namespace NUMINAMATH_CALUDE_islet_cell_transplant_indicators_l3120_312055

/-- Represents the type of transplantation performed -/
inductive TransplantationType
| IsletCell

/-- Represents the possible indicators for determining cure and medication needed -/
inductive Indicator
| UrineSugar
| Insulin
| Antiallergics
| BloodSugar
| Immunosuppressants

/-- Represents a pair of indicators -/
structure IndicatorPair :=
  (first second : Indicator)

/-- Function to determine the correct indicators based on transplantation type -/
def correctIndicators (transplantType : TransplantationType) : IndicatorPair :=
  match transplantType with
  | TransplantationType.IsletCell => ⟨Indicator.BloodSugar, Indicator.Immunosuppressants⟩

/-- Theorem stating that for islet cell transplantation, the correct indicators are blood sugar and immunosuppressants -/
theorem islet_cell_transplant_indicators :
  correctIndicators TransplantationType.IsletCell = ⟨Indicator.BloodSugar, Indicator.Immunosuppressants⟩ :=
by sorry

end NUMINAMATH_CALUDE_islet_cell_transplant_indicators_l3120_312055


namespace NUMINAMATH_CALUDE_adult_dogs_adopted_l3120_312084

/-- The number of adult dogs adopted given the costs and number of other animals -/
def num_adult_dogs (cat_cost puppy_cost adult_dog_cost total_cost : ℕ) 
                   (num_cats num_puppies : ℕ) : ℕ :=
  (total_cost - cat_cost * num_cats - puppy_cost * num_puppies) / adult_dog_cost

theorem adult_dogs_adopted :
  num_adult_dogs 50 150 100 700 2 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_adult_dogs_adopted_l3120_312084


namespace NUMINAMATH_CALUDE_tan_neg_seven_pi_sixths_l3120_312059

theorem tan_neg_seven_pi_sixths : 
  Real.tan (-7 * π / 6) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_neg_seven_pi_sixths_l3120_312059


namespace NUMINAMATH_CALUDE_shop_profit_calculation_l3120_312079

/-- The amount the shop makes off each jersey -/
def jersey_profit : ℝ := 34

/-- The amount the shop makes off each t-shirt -/
def tshirt_profit : ℝ := 192

/-- The difference in cost between a t-shirt and a jersey -/
def cost_difference : ℝ := 158

theorem shop_profit_calculation :
  jersey_profit = tshirt_profit - cost_difference :=
by sorry

end NUMINAMATH_CALUDE_shop_profit_calculation_l3120_312079


namespace NUMINAMATH_CALUDE_binomial_coefficient_2000_3_l3120_312014

theorem binomial_coefficient_2000_3 : Nat.choose 2000 3 = 1331000333 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_2000_3_l3120_312014


namespace NUMINAMATH_CALUDE_boat_purchase_payment_l3120_312008

theorem boat_purchase_payment (w x y z : ℝ) : 
  w + x + y + z = 60 ∧
  w = (1/2) * (x + y + z) ∧
  x = (1/3) * (w + y + z) ∧
  y = (1/4) * (w + x + z) →
  z = 13 := by sorry

end NUMINAMATH_CALUDE_boat_purchase_payment_l3120_312008


namespace NUMINAMATH_CALUDE_prime_cube_plus_one_l3120_312056

theorem prime_cube_plus_one (p : ℕ) (x y : ℕ+) (h_prime : Nat.Prime p) 
  (h_eq : p ^ x.val = y.val ^ 3 + 1) :
  ((p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2)) := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_plus_one_l3120_312056


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l3120_312089

theorem factorization_cubic_minus_linear (a x : ℝ) : 
  a * x^3 - 16 * a * x = a * x * (x + 4) * (x - 4) := by
sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l3120_312089


namespace NUMINAMATH_CALUDE_max_diagonal_area_ratio_l3120_312092

/-- A triangle with an inscribed rectangle -/
structure TriangleWithInscribedRectangle where
  /-- The area of the triangle -/
  area : ℝ
  /-- The length of the shortest diagonal of any inscribed rectangle -/
  shortest_diagonal : ℝ
  /-- The area is positive -/
  area_pos : 0 < area

/-- The theorem statement -/
theorem max_diagonal_area_ratio (T : TriangleWithInscribedRectangle) :
  T.shortest_diagonal ^ 2 / T.area ≤ 4 * Real.sqrt 3 / 7 := by
  sorry


end NUMINAMATH_CALUDE_max_diagonal_area_ratio_l3120_312092


namespace NUMINAMATH_CALUDE_ant_distance_theorem_l3120_312045

theorem ant_distance_theorem (n : ℕ) (points : Fin n → ℝ × ℝ) :
  n = 1390 →
  (∀ i, abs (points i).2 < 1) →
  (∀ i j, i ≠ j → dist (points i) (points j) > 2) →
  ∃ i j, dist (points i) (points j) ≥ 1000 :=
by sorry

#check ant_distance_theorem

end NUMINAMATH_CALUDE_ant_distance_theorem_l3120_312045


namespace NUMINAMATH_CALUDE_trailing_zeros_count_l3120_312015

def N : ℕ := 10^2018 + 1

theorem trailing_zeros_count (n : ℕ) : 
  ∃ k : ℕ, (N^2017 - 1) % 10^2018 = 0 ∧ (N^2017 - 1) % 10^2019 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_count_l3120_312015


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2023_l3120_312070

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_2023 (seq : ArithmeticSequence)
  (h1 : seq.a 2 + seq.a 7 = seq.a 8 + 1)
  (h2 : ∃ r : ℝ, r ≠ 0 ∧ seq.a 4 = r * seq.a 2 ∧ seq.a 8 = r * seq.a 4) :
  seq.a 2023 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2023_l3120_312070


namespace NUMINAMATH_CALUDE_mean_of_three_numbers_l3120_312011

theorem mean_of_three_numbers (x y z : ℝ) 
  (h1 : (x + y) / 2 = 5)
  (h2 : (y + z) / 2 = 9)
  (h3 : (z + x) / 2 = 10) :
  (x + y + z) / 3 = 8 := by sorry

end NUMINAMATH_CALUDE_mean_of_three_numbers_l3120_312011


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_l3120_312087

/-- The coefficient of x^3 in the expansion of (3x^3 + 2x^2 + 5x + 3)(4x^3 + 5x^2 + 6x + 8) is 61 -/
theorem coefficient_x_cubed (x : ℝ) : 
  let p₁ : Polynomial ℝ := 3 * X^3 + 2 * X^2 + 5 * X + 3
  let p₂ : Polynomial ℝ := 4 * X^3 + 5 * X^2 + 6 * X + 8
  (p₁ * p₂).coeff 3 = 61 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_l3120_312087


namespace NUMINAMATH_CALUDE_shorter_diagonal_length_l3120_312074

theorem shorter_diagonal_length (a b : ℝ × ℝ) :
  ‖a‖ = 2 →
  ‖b‖ = 4 →
  a • b = 4 →
  ‖a - b‖ = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_shorter_diagonal_length_l3120_312074


namespace NUMINAMATH_CALUDE_unique_geometric_progression_pair_l3120_312004

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricProgression (x y z w : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ y = x * r ∧ z = y * r ∧ w = z * r

/-- There exists exactly one pair of real numbers (a, b) such that 12, a, b, ab form a geometric progression. -/
theorem unique_geometric_progression_pair :
  ∃! (a b : ℝ), IsGeometricProgression 12 a b (a * b) := by
  sorry

#check unique_geometric_progression_pair

end NUMINAMATH_CALUDE_unique_geometric_progression_pair_l3120_312004


namespace NUMINAMATH_CALUDE_jeans_bought_l3120_312018

/-- Given a clothing sale with specific prices and quantities, prove the number of jeans bought. -/
theorem jeans_bought (shirt_price hat_price jeans_price total_cost : ℕ) 
  (shirts_bought hats_bought : ℕ) : 
  shirt_price = 5 →
  hat_price = 4 →
  jeans_price = 10 →
  total_cost = 51 →
  shirts_bought = 3 →
  hats_bought = 4 →
  ∃ (jeans_bought : ℕ), 
    jeans_bought = 2 ∧ 
    total_cost = shirt_price * shirts_bought + hat_price * hats_bought + jeans_price * jeans_bought :=
by sorry

end NUMINAMATH_CALUDE_jeans_bought_l3120_312018


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l3120_312030

theorem half_abs_diff_squares_20_15 : (1 / 2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l3120_312030


namespace NUMINAMATH_CALUDE_rectangle_area_l3120_312049

def square_side : ℝ := 15
def rectangle_length : ℝ := 18

theorem rectangle_area (rectangle_width : ℝ) :
  (4 * square_side = 2 * (rectangle_length + rectangle_width)) →
  (rectangle_length * rectangle_width = 216) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3120_312049


namespace NUMINAMATH_CALUDE_intersection_and_midpoint_trajectory_l3120_312025

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y-1)^2 = 5

-- Define the line l
def line_l (m x y : ℝ) : Prop := m*x - y + 1 - m = 0

-- Define the trajectory of midpoint M
def trajectory_M (x y : ℝ) : Prop := (x - 1/2)^2 + (y-1)^2 = 1/4

theorem intersection_and_midpoint_trajectory :
  ∀ m : ℝ,
  (∃ A B : ℝ × ℝ, A ≠ B ∧ circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l m A.1 A.2 ∧ line_l m B.1 B.2) ∧
  (∀ x y : ℝ, (∃ A B : ℝ × ℝ, A ≠ B ∧ circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ line_l m A.1 A.2 ∧ line_l m B.1 B.2 ∧
    x = (A.1 + B.1) / 2 ∧ y = (A.2 + B.2) / 2) → trajectory_M x y) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_midpoint_trajectory_l3120_312025


namespace NUMINAMATH_CALUDE_complex_addition_l3120_312053

theorem complex_addition : (6 - 5*Complex.I) + (3 + 2*Complex.I) = 9 - 3*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_addition_l3120_312053


namespace NUMINAMATH_CALUDE_area_between_curves_l3120_312027

-- Define the two curves
def curve1 (x : ℝ) : ℝ := x^3 - x
def curve2 (a x : ℝ) : ℝ := x^2 - a

-- Define the derivatives of the curves
def curve1_derivative (x : ℝ) : ℝ := 3 * x^2 - 1
def curve2_derivative (x : ℝ) : ℝ := 2 * x

-- Theorem statement
theorem area_between_curves :
  ∃ (a : ℝ) (P : ℝ × ℝ),
    -- Conditions:
    -- 1. P lies on both curves
    curve1 P.1 = P.2 ∧
    curve2 a P.1 = P.2 ∧
    -- 2. The curves have a common tangent at P
    curve1_derivative P.1 = curve2_derivative P.1 →
    -- Conclusion:
    -- The area between the curves is 13/12
    (∫ x in (Real.sqrt 5 / 2 - 1 / 6)..(1 / 6 + Real.sqrt 5 / 2), |curve1 x - curve2 a x|) = 13 / 12 :=
by
  sorry

end NUMINAMATH_CALUDE_area_between_curves_l3120_312027


namespace NUMINAMATH_CALUDE_watch_cost_price_l3120_312058

/-- The cost price of a watch satisfying certain selling conditions -/
theorem watch_cost_price : ∃ (cp : ℚ), 
  (cp * (1 - 1/10) = cp * 0.9) ∧ 
  (cp * (1 + 1/10) = cp * 1.1) ∧ 
  (cp * 1.1 - cp * 0.9 = 500) ∧ 
  cp = 2500 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l3120_312058


namespace NUMINAMATH_CALUDE_pi_is_irrational_l3120_312028

-- Define what it means for a number to be rational
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define π (since it's not a built-in constant in Lean)
noncomputable def π : ℝ := Real.pi

-- Theorem statement
theorem pi_is_irrational : ¬ IsRational π := by
  sorry


end NUMINAMATH_CALUDE_pi_is_irrational_l3120_312028


namespace NUMINAMATH_CALUDE_next_perfect_square_l3120_312035

theorem next_perfect_square (n : ℤ) (x : ℤ) (h1 : Even n) (h2 : x = n^2) :
  (n + 1)^2 = x + 2*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_next_perfect_square_l3120_312035


namespace NUMINAMATH_CALUDE_fraction_difference_l3120_312066

theorem fraction_difference (p q : ℝ) (hp : 3 ≤ p ∧ p ≤ 10) (hq : 12 ≤ q ∧ q ≤ 21) :
  (10 / 12 : ℝ) - (3 / 21 : ℝ) = 29 / 42 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l3120_312066


namespace NUMINAMATH_CALUDE_independence_test_problems_l3120_312073

/-- A real-world problem that may or may not be solvable by independence tests. -/
inductive Problem
| DrugCureRate
| DrugRelation
| SmokingLungDisease
| SmokingGenderRelation
| InternetCrimeRate

/-- Determines if a problem involves examining the relationship between two categorical variables. -/
def involves_categorical_relationship (p : Problem) : Prop :=
  match p with
  | Problem.DrugRelation => True
  | Problem.SmokingGenderRelation => True
  | Problem.InternetCrimeRate => True
  | _ => False

/-- The definition of an independence test. -/
def is_independence_test (test : Problem → Prop) : Prop :=
  ∀ p, test p ↔ involves_categorical_relationship p

/-- The theorem stating which problems can be solved using independence tests. -/
theorem independence_test_problems (test : Problem → Prop) 
  (h : is_independence_test test) : 
  (test Problem.DrugRelation ∧ 
   test Problem.SmokingGenderRelation ∧ 
   test Problem.InternetCrimeRate) ∧
  (¬ test Problem.DrugCureRate ∧ 
   ¬ test Problem.SmokingLungDisease) :=
by sorry

end NUMINAMATH_CALUDE_independence_test_problems_l3120_312073


namespace NUMINAMATH_CALUDE_ceiling_floor_calculation_l3120_312021

theorem ceiling_floor_calculation : 
  ⌈(15 : ℚ) / 8 * (-34 : ℚ) / 4⌉ - ⌊(15 : ℚ) / 8 * ⌊(-34 : ℚ) / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_calculation_l3120_312021


namespace NUMINAMATH_CALUDE_max_distance_in_parallelepiped_l3120_312048

/-- The maximum distance between two points in a 3x4x2 rectangular parallelepiped --/
theorem max_distance_in_parallelepiped :
  let a : ℝ := 3
  let b : ℝ := 4
  let c : ℝ := 2
  ∃ (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ),
    0 ≤ x₁ ∧ x₁ ≤ a ∧
    0 ≤ y₁ ∧ y₁ ≤ b ∧
    0 ≤ z₁ ∧ z₁ ≤ c ∧
    0 ≤ x₂ ∧ x₂ ≤ a ∧
    0 ≤ y₂ ∧ y₂ ≤ b ∧
    0 ≤ z₂ ∧ z₂ ≤ c ∧
    ∀ (x₃ y₃ z₃ x₄ y₄ z₄ : ℝ),
      0 ≤ x₃ ∧ x₃ ≤ a ∧
      0 ≤ y₃ ∧ y₃ ≤ b ∧
      0 ≤ z₃ ∧ z₃ ≤ c ∧
      0 ≤ x₄ ∧ x₄ ≤ a ∧
      0 ≤ y₄ ∧ y₄ ≤ b ∧
      0 ≤ z₄ ∧ z₄ ≤ c →
      (x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2 ≥ (x₃ - x₄)^2 + (y₃ - y₄)^2 + (z₃ - z₄)^2 ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_in_parallelepiped_l3120_312048


namespace NUMINAMATH_CALUDE_cake_cost_calculation_l3120_312003

/-- The cost of a cake given initial money and remaining money after purchase -/
def cake_cost (initial_money remaining_money : ℚ) : ℚ :=
  initial_money - remaining_money

theorem cake_cost_calculation (initial_money remaining_money : ℚ) 
  (h1 : initial_money = 59.5)
  (h2 : remaining_money = 42) : 
  cake_cost initial_money remaining_money = 17.5 := by
  sorry

#eval cake_cost 59.5 42

end NUMINAMATH_CALUDE_cake_cost_calculation_l3120_312003


namespace NUMINAMATH_CALUDE_y_range_l3120_312075

theorem y_range (a b y : ℝ) (h1 : a + b = 2) (h2 : b ≤ 2) (h3 : y - a^2 - 2*a + 2 = 0) :
  y ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_y_range_l3120_312075


namespace NUMINAMATH_CALUDE_product_of_symmetric_complex_numbers_l3120_312094

def symmetric_about_imaginary_axis (z₁ z₂ : ℂ) : Prop :=
  z₁.re = -z₂.re ∧ z₁.im = z₂.im

theorem product_of_symmetric_complex_numbers :
  ∀ z₁ z₂ : ℂ, 
    symmetric_about_imaginary_axis z₁ z₂ → 
    z₁ = 1 + 2*I → 
    z₁ * z₂ = -5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_symmetric_complex_numbers_l3120_312094


namespace NUMINAMATH_CALUDE_emma_age_l3120_312026

def guesses : List Nat := [26, 29, 31, 33, 35, 39, 42, 44, 47, 50]

def is_prime (n : Nat) : Prop := Nat.Prime n

def off_by_one (guess : Nat) (age : Nat) : Prop :=
  guess = age - 1 ∨ guess = age + 1

def count_lower_guesses (age : Nat) : Nat :=
  guesses.filter (· < age) |>.length

theorem emma_age : ∃ (age : Nat),
  age ∈ guesses ∧
  is_prime age ∧
  (count_lower_guesses age : Rat) / guesses.length ≥ 6/10 ∧
  (∃ (g1 g2 : Nat), g1 ∈ guesses ∧ g2 ∈ guesses ∧ g1 ≠ g2 ∧ 
    off_by_one g1 age ∧ off_by_one g2 age) ∧
  age = 43 := by
  sorry

end NUMINAMATH_CALUDE_emma_age_l3120_312026


namespace NUMINAMATH_CALUDE_inscribed_circle_area_isosceles_trapezoid_l3120_312051

/-- The area of a circle inscribed in an isosceles trapezoid -/
theorem inscribed_circle_area_isosceles_trapezoid 
  (a : ℝ) 
  (h_positive : a > 0) 
  (h_isosceles : IsoscelesTrapezoid) 
  (h_angle : AngleAtSmallerBase = 120) : 
  AreaOfInscribedCircle = π * a^2 / 12 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_isosceles_trapezoid_l3120_312051


namespace NUMINAMATH_CALUDE_sum_after_removal_is_perfect_square_l3120_312032

-- Define the set M
def M : Set Nat := {n | 1 ≤ n ∧ n ≤ 2017}

-- Define the sum of all elements in M
def sum_M : Nat := (2017 * 2018) / 2

-- Define the element to be removed
def removed_element : Nat := 1677

-- Theorem to prove
theorem sum_after_removal_is_perfect_square :
  ∃ k : Nat, sum_M - removed_element = k^2 ∧ removed_element ∈ M :=
sorry

end NUMINAMATH_CALUDE_sum_after_removal_is_perfect_square_l3120_312032


namespace NUMINAMATH_CALUDE_quadrilateral_area_l3120_312036

/-- The area of a quadrilateral with one diagonal of length 50 cm and offsets of 10 cm and 8 cm is 450 cm². -/
theorem quadrilateral_area (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) 
  (h1 : diagonal = 50) 
  (h2 : offset1 = 10) 
  (h3 : offset2 = 8) : 
  (1/2 * diagonal * offset1) + (1/2 * diagonal * offset2) = 450 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l3120_312036


namespace NUMINAMATH_CALUDE_cricket_bat_weight_proof_l3120_312023

/-- The weight of one cricket bat in pounds -/
def cricket_bat_weight : ℝ := 18

/-- The weight of one basketball in pounds -/
def basketball_weight : ℝ := 36

/-- The number of cricket bats -/
def num_cricket_bats : ℕ := 8

/-- The number of basketballs -/
def num_basketballs : ℕ := 4

theorem cricket_bat_weight_proof :
  cricket_bat_weight * num_cricket_bats = basketball_weight * num_basketballs :=
by sorry

end NUMINAMATH_CALUDE_cricket_bat_weight_proof_l3120_312023


namespace NUMINAMATH_CALUDE_quarter_circle_sum_limit_l3120_312069

theorem quarter_circle_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |n * (π * D / (4 * n)) - (π * D / 4)| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circle_sum_limit_l3120_312069


namespace NUMINAMATH_CALUDE_beam_max_strength_l3120_312095

/-- The strength of a rectangular beam cut from a circular log is maximized when its width is 2R/√3 and its height is 2R√2/√3, where R is the radius of the log. -/
theorem beam_max_strength (R : ℝ) (R_pos : R > 0) :
  let strength (x y : ℝ) := x * y^2
  let constraint (x y : ℝ) := x^2 + y^2 = 4 * R^2
  ∃ (k : ℝ), k > 0 ∧
    ∀ (x y : ℝ), constraint x y →
      strength x y ≤ k * strength (2*R/Real.sqrt 3) (2*R*Real.sqrt 2/Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_beam_max_strength_l3120_312095


namespace NUMINAMATH_CALUDE_apple_arrangements_l3120_312099

/-- The number of distinct arrangements of letters in a word with repeated letters -/
def distinctArrangements (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The word "APPLE" has 5 letters with 'P' repeating twice -/
def appleWord : (ℕ × List ℕ) := (5, [2])

theorem apple_arrangements :
  distinctArrangements appleWord.1 appleWord.2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_apple_arrangements_l3120_312099


namespace NUMINAMATH_CALUDE_inverse_inequality_conditions_l3120_312076

theorem inverse_inequality_conditions (a b : ℝ) :
  (1 / a < 1 / b) ↔ (b > 0 ∧ 0 > a) ∨ (0 > a ∧ a > b) ∨ (a > b ∧ b > 0) :=
sorry

end NUMINAMATH_CALUDE_inverse_inequality_conditions_l3120_312076


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l3120_312041

theorem complex_equation_solutions (z : ℂ) : 
  z^3 + z = 2 * Complex.abs z^2 → 
  z = 0 ∨ z = 1 ∨ z = -1 + 2*Complex.I ∨ z = -1 - 2*Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l3120_312041


namespace NUMINAMATH_CALUDE_toris_growth_l3120_312088

theorem toris_growth (original_height current_height : Real) 
  (h1 : original_height = 4.4)
  (h2 : current_height = 7.26) :
  current_height - original_height = 2.86 := by
  sorry

end NUMINAMATH_CALUDE_toris_growth_l3120_312088


namespace NUMINAMATH_CALUDE_ratio_x_to_y_l3120_312068

theorem ratio_x_to_y (x y : ℚ) (h : (8 * x + 5 * y) / (10 * x + 3 * y) = 4 / 7) :
  x / y = -23 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ratio_x_to_y_l3120_312068


namespace NUMINAMATH_CALUDE_gym_spending_l3120_312054

theorem gym_spending (total_spent adidas_cost nike_cost skechers_cost clothes_cost : ℝ) : 
  total_spent = 8000 →
  nike_cost = 3 * adidas_cost →
  adidas_cost = (1 / 5) * skechers_cost →
  adidas_cost = 600 →
  total_spent = adidas_cost + nike_cost + skechers_cost + clothes_cost →
  clothes_cost = 2600 := by
sorry

end NUMINAMATH_CALUDE_gym_spending_l3120_312054


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_l3120_312040

/-- Represents a cone with given base radius and height -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere with given radius -/
structure Sphere where
  radius : ℝ

/-- Checks if a sphere is inscribed in a cone -/
def isInscribed (c : Cone) (s : Sphere) : Prop :=
  -- This is a placeholder for the actual geometric condition
  True

theorem inscribed_sphere_radius (c : Cone) (s : Sphere) 
  (h1 : c.baseRadius = 15)
  (h2 : c.height = 30)
  (h3 : isInscribed c s) :
  s.radius = 7.5 * Real.sqrt 5 - 7.5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_l3120_312040


namespace NUMINAMATH_CALUDE_optimal_solution_l3120_312062

-- Define the normal distribution parameters
def μ : ℝ := 800
def σ : ℝ := 50

-- Define the probability p₀
def p₀ : ℝ := 0.9772

-- Define vehicle capacities and costs
def capacity_A : ℕ := 36
def capacity_B : ℕ := 60
def cost_A : ℕ := 1600
def cost_B : ℕ := 2400

-- Define the optimization problem
def optimal_fleet (a b : ℕ) : Prop :=
  -- Total vehicles constraint
  a + b ≤ 21 ∧
  -- Type B vehicles constraint
  b ≤ a + 7 ∧
  -- Probability constraint (simplified)
  (a * capacity_A + b * capacity_B : ℝ) ≥ μ + σ * 2 ∧
  -- Minimizes cost
  ∀ a' b' : ℕ,
    (a' * capacity_A + b' * capacity_B : ℝ) ≥ μ + σ * 2 →
    a' + b' ≤ 21 →
    b' ≤ a' + 7 →
    a * cost_A + b * cost_B ≤ a' * cost_A + b' * cost_B

-- Theorem statement
theorem optimal_solution :
  optimal_fleet 5 12 :=
sorry

end NUMINAMATH_CALUDE_optimal_solution_l3120_312062


namespace NUMINAMATH_CALUDE_count_words_theorem_l3120_312077

/-- The set of all available letters -/
def Letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

/-- The set of consonants -/
def Consonants : Finset Char := {'B', 'C', 'D', 'F'}

/-- The set of vowels -/
def Vowels : Finset Char := {'A', 'E'}

/-- The length of words we're considering -/
def WordLength : Nat := 5

/-- Function to count the number of 5-letter words with at least two consonants -/
def count_words_with_at_least_two_consonants : Nat :=
  sorry

/-- Theorem stating that the number of 5-letter words with at least two consonants is 7424 -/
theorem count_words_theorem : count_words_with_at_least_two_consonants = 7424 := by
  sorry

end NUMINAMATH_CALUDE_count_words_theorem_l3120_312077


namespace NUMINAMATH_CALUDE_expression_evaluation_l3120_312071

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(2*y) * y^(3*x)) / (y^(2*y) * x^(3*x)) = x^(2*y - 3*x) * y^(3*x - 2*y) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3120_312071


namespace NUMINAMATH_CALUDE_similar_triangles_side_length_l3120_312033

/-- Given two similar triangles with areas A₁ and A₂, where A₁ > A₂,
    prove that the corresponding side of the larger triangle is 12 feet. -/
theorem similar_triangles_side_length 
  (A₁ A₂ : ℝ) 
  (h_positive : A₁ > A₂) 
  (h_diff : A₁ - A₂ = 27) 
  (h_ratio : A₁ / A₂ = 9) 
  (h_small_side : ∃ (s : ℝ), s = 4 ∧ s * s / 2 ≤ A₂) : 
  ∃ (S : ℝ), S = 12 ∧ S * S / 2 ≤ A₁ := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_side_length_l3120_312033


namespace NUMINAMATH_CALUDE_problem_solution_l3120_312043

theorem problem_solution (x : ℝ) :
  x + Real.sqrt (x^2 + 2) + 1 / (x - Real.sqrt (x^2 + 2)) = 15 →
  x^2 + Real.sqrt (x^4 + 2) + 1 / (x^2 + Real.sqrt (x^4 + 2)) = 47089 / 1800 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3120_312043


namespace NUMINAMATH_CALUDE_average_candies_sigyeong_group_l3120_312039

def sigyeong_group : List Nat := [16, 22, 30, 26, 18, 20]

theorem average_candies_sigyeong_group : 
  (sigyeong_group.sum / sigyeong_group.length : ℚ) = 22 := by
  sorry

end NUMINAMATH_CALUDE_average_candies_sigyeong_group_l3120_312039


namespace NUMINAMATH_CALUDE_cafeteria_extra_fruits_l3120_312057

/-- The number of extra fruits ordered by the cafeteria -/
def extra_fruits (total_fruits students max_per_student : ℕ) : ℕ :=
  total_fruits - (students * max_per_student)

/-- Theorem stating that the cafeteria ordered 43 extra fruits -/
theorem cafeteria_extra_fruits :
  extra_fruits 85 21 2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_extra_fruits_l3120_312057


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l3120_312000

theorem abs_neg_three_eq_three : |(-3 : ℤ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l3120_312000


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_negative_l3120_312007

theorem sum_of_reciprocals_negative (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (product_eight : a * b * c = 8) : 
  1 / a + 1 / b + 1 / c < 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_negative_l3120_312007


namespace NUMINAMATH_CALUDE_tshirts_per_package_l3120_312024

theorem tshirts_per_package (total_tshirts : ℕ) (num_packages : ℕ) 
  (h1 : total_tshirts = 70) 
  (h2 : num_packages = 14) : 
  total_tshirts / num_packages = 5 := by
  sorry

end NUMINAMATH_CALUDE_tshirts_per_package_l3120_312024


namespace NUMINAMATH_CALUDE_fraction_inequality_not_sufficient_nor_necessary_sufficient_condition_implies_subset_l3120_312093

-- Statement B
theorem fraction_inequality_not_sufficient_nor_necessary :
  ¬(∀ a b : ℝ, (1 / a > 1 / b → a < b) ∧ (a < b → 1 / a > 1 / b)) := by sorry

-- Statement C
theorem sufficient_condition_implies_subset (A B : Set α) :
  (∀ x, x ∈ A → x ∈ B) → A ⊆ B := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_not_sufficient_nor_necessary_sufficient_condition_implies_subset_l3120_312093


namespace NUMINAMATH_CALUDE_quadratic_conditions_l3120_312001

/-- The quadratic function f(x) = x^2 - 4x - 3 + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x - 3 + a

/-- Theorem stating the conditions for the quadratic function -/
theorem quadratic_conditions :
  (∃ a : ℝ, f a 0 = 1 ∧ a = 4) ∧
  (∃ a : ℝ, (∀ x : ℝ, f a x = 0 → x = 0 ∨ x ≠ 0) ∧ (a = 3 ∨ a = 7)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_conditions_l3120_312001


namespace NUMINAMATH_CALUDE_power_product_simplification_l3120_312044

theorem power_product_simplification (a : ℝ) : (3 * a)^2 * a^5 = 9 * a^7 := by
  sorry

end NUMINAMATH_CALUDE_power_product_simplification_l3120_312044


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l3120_312081

/-- Given two vectors a and b in ℝ³, where a = (2, 4, 5) and b = (3, x, y),
    if a is parallel to b, then x + y = 27/2 -/
theorem parallel_vectors_sum (x y : ℝ) :
  let a : Fin 3 → ℝ := ![2, 4, 5]
  let b : Fin 3 → ℝ := ![3, x, y]
  (∃ (k : ℝ), ∀ i, a i = k * b i) →
  x + y = 27/2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l3120_312081


namespace NUMINAMATH_CALUDE_percentage_of_part_to_whole_l3120_312091

theorem percentage_of_part_to_whole (part whole : ℝ) (h : whole ≠ 0) :
  (part / whole) * 100 = 25 →
  part = 70 ∧ whole = 280 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_part_to_whole_l3120_312091


namespace NUMINAMATH_CALUDE_parabola_line_slope_l3120_312016

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through a point with a given slope
def line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define a point on the latus rectum
def on_latus_rectum (x y : ℝ) : Prop := x = -1

-- Define a point in the first quadrant
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the midpoint condition
def is_midpoint (x1 y1 x2 y2 x3 y3 : ℝ) : Prop := 
  x2 = (x1 + x3) / 2 ∧ y2 = (y1 + y3) / 2

theorem parabola_line_slope (k : ℝ) (x1 y1 x2 y2 x3 y3 : ℝ) : 
  parabola x1 y1 →
  parabola x2 y2 →
  line k x1 y1 →
  line k x2 y2 →
  line k x3 y3 →
  on_latus_rectum x3 y3 →
  in_first_quadrant x1 y1 →
  is_midpoint x1 y1 x2 y2 x3 y3 →
  k = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_line_slope_l3120_312016


namespace NUMINAMATH_CALUDE_one_of_each_color_probability_l3120_312063

/-- Probability of selecting one marble of each color -/
theorem one_of_each_color_probability
  (total_marbles : Nat)
  (red_marbles blue_marbles green_marbles : Nat)
  (h1 : total_marbles = red_marbles + blue_marbles + green_marbles)
  (h2 : red_marbles = 3)
  (h3 : blue_marbles = 3)
  (h4 : green_marbles = 2)
  (h5 : total_marbles = 8) :
  (red_marbles * blue_marbles * green_marbles : Rat) /
  (Nat.choose total_marbles 3 : Rat) = 9 / 28 := by
  sorry

end NUMINAMATH_CALUDE_one_of_each_color_probability_l3120_312063


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l3120_312006

theorem inscribed_cube_volume (large_cube_edge : ℝ) (sphere_diameter : ℝ) (small_cube_edge : ℝ) :
  large_cube_edge = 12 →
  sphere_diameter = large_cube_edge →
  small_cube_edge * Real.sqrt 3 = sphere_diameter →
  small_cube_edge ^ 3 = 192 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l3120_312006


namespace NUMINAMATH_CALUDE_ellipse_equation_l3120_312029

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and eccentricity e,
    if a triangle formed by the intersection of a line through one focus with the ellipse
    has perimeter p, then the standard equation of the ellipse is x²/3 + y²/2 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (e : ℝ) (he : e = Real.sqrt 3 / 3)
  (p : ℝ) (hp : p = 4 * Real.sqrt 3) :
  ∃ (x y : ℝ), x^2 / 3 + y^2 / 2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3120_312029


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3120_312061

theorem trigonometric_identity (α : ℝ) : 
  (Real.sin (7 * α) / Real.sin α) - 2 * (Real.cos (2 * α) + Real.cos (4 * α) + Real.cos (6 * α)) - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3120_312061


namespace NUMINAMATH_CALUDE_sin_double_angle_l3120_312012

theorem sin_double_angle (x : Real) (h : Real.sin (x + π/4) = 3/5) : 
  Real.sin (2*x) = 8*Real.sqrt 2/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_l3120_312012


namespace NUMINAMATH_CALUDE_man_swimming_speed_l3120_312020

/-- The speed of a man in still water given his downstream and upstream swimming times and distances -/
theorem man_swimming_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (h_downstream : downstream_distance = 36) 
  (h_upstream : upstream_distance = 48) 
  (h_time : time = 6) : 
  ∃ (v_man : ℝ) (v_stream : ℝ), 
    v_man + v_stream = downstream_distance / time ∧ 
    v_man - v_stream = upstream_distance / time ∧ 
    v_man = 7 := by
  sorry

#check man_swimming_speed

end NUMINAMATH_CALUDE_man_swimming_speed_l3120_312020


namespace NUMINAMATH_CALUDE_inverse_proportion_m_value_l3120_312083

-- Define the function y as an inverse proportion function
def is_inverse_proportion (m : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, x ≠ 0 → (m - 2) * x^(m^2 - 5) = k / x

-- State the theorem
theorem inverse_proportion_m_value :
  ∀ m : ℝ, is_inverse_proportion m → m - 2 ≠ 0 → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_m_value_l3120_312083


namespace NUMINAMATH_CALUDE_half_power_decreasing_l3120_312097

theorem half_power_decreasing (a b : ℝ) (h : a > b) : (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_half_power_decreasing_l3120_312097


namespace NUMINAMATH_CALUDE_perimeter_quadrilateral_l3120_312082

/-- The perimeter of a quadrilateral PQRS with given coordinates can be expressed as x√3 + y√10, where x + y = 12 -/
theorem perimeter_quadrilateral (P Q R S : ℝ × ℝ) : 
  P = (1, 2) → Q = (3, 6) → R = (6, 3) → S = (8, 1) →
  ∃ (x y : ℤ), 
    (Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) +
     Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) +
     Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) +
     Real.sqrt ((S.1 - P.1)^2 + (S.2 - P.2)^2) =
     x * Real.sqrt 3 + y * Real.sqrt 10) ∧
    x + y = 12 :=
by sorry

end NUMINAMATH_CALUDE_perimeter_quadrilateral_l3120_312082


namespace NUMINAMATH_CALUDE_min_abs_z_plus_i_l3120_312034

theorem min_abs_z_plus_i (z : ℂ) (h : Complex.abs (z^2 + 16) = Complex.abs (z * (z + 4*I))) :
  ∃ (w : ℂ), Complex.abs (w + I) = 3 ∧ ∀ (z : ℂ), Complex.abs (z^2 + 16) = Complex.abs (z * (z + 4*I)) → Complex.abs (z + I) ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_plus_i_l3120_312034


namespace NUMINAMATH_CALUDE_percentage_calculation_l3120_312013

theorem percentage_calculation (n : ℝ) (h : n = 6000) :
  (0.1 * (0.3 * (0.5 * n))) = 90 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3120_312013


namespace NUMINAMATH_CALUDE_negation_equivalence_l3120_312038

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ ≤ 0 ∧ x₀^2 ≥ 0) ↔ (∀ x : ℝ, x ≤ 0 → x^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3120_312038


namespace NUMINAMATH_CALUDE_atomic_number_difference_l3120_312052

/-- Represents an element in the periodic table -/
structure Element where
  atomicNumber : ℕ

/-- Represents a main group in the periodic table -/
structure MainGroup where
  elements : Set Element

/-- 
  Given two elements A and B in the same main group of the periodic table, 
  where the atomic number of A is x, the atomic number of B cannot be x+4.
-/
theorem atomic_number_difference (g : MainGroup) (A B : Element) (x : ℕ) :
  A ∈ g.elements → B ∈ g.elements → A.atomicNumber = x → 
  B.atomicNumber ≠ x + 4 := by
  sorry

end NUMINAMATH_CALUDE_atomic_number_difference_l3120_312052


namespace NUMINAMATH_CALUDE_zero_is_monomial_l3120_312072

/-- Definition of a monomial -/
def is_monomial (expr : ℕ → ℚ) : Prop :=
  ∃ (c : ℚ) (n : ℕ), ∀ (k : ℕ), expr k = if k = n then c else 0

/-- Theorem: 0 is a monomial -/
theorem zero_is_monomial : is_monomial (λ _ => 0) := by
  sorry

end NUMINAMATH_CALUDE_zero_is_monomial_l3120_312072


namespace NUMINAMATH_CALUDE_correct_assignment_count_l3120_312096

/-- The number of ways to assign volunteers to pavilions. -/
def assign_volunteers (total_volunteers : ℕ) (female_volunteers : ℕ) (male_volunteers : ℕ) (pavilions : ℕ) : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem stating the correct number of ways to assign volunteers. -/
theorem correct_assignment_count :
  assign_volunteers 8 3 5 3 = 180 :=
sorry

end NUMINAMATH_CALUDE_correct_assignment_count_l3120_312096


namespace NUMINAMATH_CALUDE_perfect_square_preserver_iff_square_multiple_l3120_312086

/-- A function is a perfect square preserver if it preserves the property of
    the sum of three distinct positive integers being a perfect square. -/
def IsPerfectSquarePreserver (f : ℕ → ℕ) : Prop :=
  ∀ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z →
    (∃ n : ℕ, x + y + z = n^2) ↔ (∃ m : ℕ, f x + f y + f z = m^2)

/-- A function is a square multiple if it's of the form f(x) = k²x for some k ∈ ℕ. -/
def IsSquareMultiple (f : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, ∀ x : ℕ, f x = k^2 * x

theorem perfect_square_preserver_iff_square_multiple (f : ℕ → ℕ) :
  IsPerfectSquarePreserver f ↔ IsSquareMultiple f := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_preserver_iff_square_multiple_l3120_312086


namespace NUMINAMATH_CALUDE_smallest_integer_square_triple_l3120_312046

theorem smallest_integer_square_triple (x : ℤ) : x^2 = 3*x + 75 → x ≥ -5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_square_triple_l3120_312046


namespace NUMINAMATH_CALUDE_trig_inequality_l3120_312064

theorem trig_inequality (x y : Real) 
  (hx : 0 < x ∧ x < Real.pi / 2)
  (hy : 0 < y ∧ y < Real.pi / 2)
  (h_eq : Real.sin x = x * Real.cos y) : 
  x / 2 < y ∧ y < x :=
by sorry

end NUMINAMATH_CALUDE_trig_inequality_l3120_312064


namespace NUMINAMATH_CALUDE_max_fraction_sum_l3120_312080

theorem max_fraction_sum (A B C D : ℕ) : 
  A ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) → 
  B ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) → 
  C ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) → 
  D ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) → 
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  B ≠ 0 → D ≠ 0 →
  (A : ℚ) / B + (C : ℚ) / D ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_fraction_sum_l3120_312080


namespace NUMINAMATH_CALUDE_problem_solution_l3120_312067

-- Define proposition p
def p : Prop := ∀ x : ℝ, 2^x > x^2

-- Define proposition q
def q : Prop := ∃ x₀ : ℝ, x₀ - 2 > 0

-- Theorem to prove
theorem problem_solution : ¬p ∧ q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3120_312067


namespace NUMINAMATH_CALUDE_group_distribution_theorem_l3120_312090

def number_of_ways (n_men n_women : ℕ) (group_sizes : List ℕ) : ℕ :=
  sorry

theorem group_distribution_theorem :
  let n_men := 4
  let n_women := 5
  let group_sizes := [3, 3, 3]
  number_of_ways n_men n_women group_sizes = 1440 :=
by sorry

end NUMINAMATH_CALUDE_group_distribution_theorem_l3120_312090


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3120_312037

def A : Set ℕ := {2, 4, 6, 8}
def B : Set ℕ := {1, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3120_312037


namespace NUMINAMATH_CALUDE_angle_C_value_l3120_312019

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin x + (Real.sqrt 3/2) * Real.cos x

theorem angle_C_value (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  f A = Real.sqrt 3 / 2 ∧
  a = (Real.sqrt 3 / 2) * b ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C →
  C = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_value_l3120_312019


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3120_312005

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 + x + 1 ≥ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x ≥ 3}

-- Define the complement of B with respect to U
def C_U_B : Set ℝ := {x : ℝ | x ∉ B}

-- Theorem statement
theorem intersection_A_complement_B :
  A ∩ C_U_B = {x : ℝ | x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3120_312005


namespace NUMINAMATH_CALUDE_company_kw_price_percentage_l3120_312010

theorem company_kw_price_percentage (price kw : ℝ) (assets_a assets_b : ℝ) 
  (h1 : price = 2 * assets_b)
  (h2 : price = 0.75 * (assets_a + assets_b))
  (h3 : ∃ x : ℝ, price = assets_a * (1 + x / 100)) :
  ∃ x : ℝ, x = 20 ∧ price = assets_a * (1 + x / 100) :=
sorry

end NUMINAMATH_CALUDE_company_kw_price_percentage_l3120_312010


namespace NUMINAMATH_CALUDE_painting_payment_l3120_312098

theorem painting_payment (rate : ℚ) (rooms : ℚ) (h1 : rate = 13 / 3) (h2 : rooms = 8 / 5) :
  rate * rooms = 104 / 15 := by
sorry

end NUMINAMATH_CALUDE_painting_payment_l3120_312098


namespace NUMINAMATH_CALUDE_function_expressions_and_minimum_l3120_312060

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x * (x + 2)
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b * x + 2

def has_same_tangent_at_zero (f g : ℝ → ℝ) : Prop :=
  (deriv f) 0 = (deriv g) 0 ∧ f 0 = g 0

theorem function_expressions_and_minimum (a b : ℝ) (t : ℝ) 
  (h1 : has_same_tangent_at_zero (f a) (g b))
  (h2 : t > -4) :
  (∃ (a' b' : ℝ), f a' = f 1 ∧ g b' = g 3) ∧
  (∀ x ∈ Set.Icc t (t + 1),
    (t < -3 → f 1 x ≥ -Real.exp (-3)) ∧
    (t ≥ -3 → f 1 x ≥ Real.exp t * (t + 2))) :=
by sorry

end NUMINAMATH_CALUDE_function_expressions_and_minimum_l3120_312060


namespace NUMINAMATH_CALUDE_family_of_lines_fixed_point_l3120_312022

/-- The point that all lines in the family kx+y+2k+1=0 pass through -/
theorem family_of_lines_fixed_point (k : ℝ) : 
  k * (-2) + (-1) + 2 * k + 1 = 0 := by
  sorry

#check family_of_lines_fixed_point

end NUMINAMATH_CALUDE_family_of_lines_fixed_point_l3120_312022


namespace NUMINAMATH_CALUDE_interior_nodes_theorem_l3120_312009

/-- A point with integer coordinates -/
structure Node where
  x : ℤ
  y : ℤ

/-- A triangle with vertices at nodes -/
structure Triangle where
  a : Node
  b : Node
  c : Node

/-- Checks if a node is inside a triangle -/
def Node.isInside (n : Node) (t : Triangle) : Prop := sorry

/-- Checks if a line through two nodes contains a vertex of the triangle -/
def Line.containsVertex (p q : Node) (t : Triangle) : Prop := sorry

/-- Checks if a line through two nodes is parallel to a side of the triangle -/
def Line.isParallelToSide (p q : Node) (t : Triangle) : Prop := sorry

/-- The main theorem -/
theorem interior_nodes_theorem (t : Triangle) 
  (h : ∃ (p q : Node), p.isInside t ∧ q.isInside t ∧ p ≠ q) :
  ∃ (x y : Node), 
    x.isInside t ∧ 
    y.isInside t ∧ 
    x ≠ y ∧
    (Line.containsVertex x y t ∨ Line.isParallelToSide x y t) := by
  sorry

end NUMINAMATH_CALUDE_interior_nodes_theorem_l3120_312009


namespace NUMINAMATH_CALUDE_decimal_to_binary_89_l3120_312002

theorem decimal_to_binary_89 :
  ∃ (b : List Bool),
    b.reverse.map (λ x => if x then 1 else 0) = [1, 0, 1, 1, 0, 0, 1] ∧
    b.foldr (λ x acc => 2 * acc + if x then 1 else 0) 0 = 89 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_89_l3120_312002


namespace NUMINAMATH_CALUDE_dvd_average_price_l3120_312042

/-- Calculates the average price of DVDs bought from different boxes -/
theorem dvd_average_price (box1_count box1_price box2_count box2_price box3_count box3_price : ℚ) :
  box1_count = 10 →
  box1_price = 2 →
  box2_count = 5 →
  box2_price = 5 →
  box3_count = 3 →
  box3_price = 7 →
  (box1_count * box1_price + box2_count * box2_price + box3_count * box3_price) / 
  (box1_count + box2_count + box3_count) = 367/100 := by
  sorry

#eval (10 * 2 + 5 * 5 + 3 * 7) / (10 + 5 + 3)

end NUMINAMATH_CALUDE_dvd_average_price_l3120_312042


namespace NUMINAMATH_CALUDE_incorrect_inequality_transformation_l3120_312031

theorem incorrect_inequality_transformation :
  ¬(∀ (a b c : ℝ), a * c > b * c → a > b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_transformation_l3120_312031
