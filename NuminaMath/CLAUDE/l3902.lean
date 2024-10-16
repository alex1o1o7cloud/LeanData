import Mathlib

namespace NUMINAMATH_CALUDE_triangle_inradius_l3902_390259

/-- Given a triangle with perimeter 39 cm and area 29.25 cm², its inradius is 1.5 cm -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h1 : p = 39) 
  (h2 : A = 29.25) 
  (h3 : A = r * p / 2) : 
  r = 1.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_inradius_l3902_390259


namespace NUMINAMATH_CALUDE_all_are_siblings_l3902_390245

-- Define a finite type with 7 elements to represent the boys
inductive Boy : Type
  | B1 | B2 | B3 | B4 | B5 | B6 | B7

-- Define the sibling relation
def is_sibling : Boy → Boy → Prop := sorry

-- State the theorem
theorem all_are_siblings :
  (∀ b : Boy, ∃ (s : Finset Boy), s.card ≥ 3 ∧ ∀ s' ∈ s, is_sibling b s') →
  (∀ b1 b2 : Boy, is_sibling b1 b2) :=
sorry

end NUMINAMATH_CALUDE_all_are_siblings_l3902_390245


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l3902_390252

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 552) : x + (x + 1) = 47 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l3902_390252


namespace NUMINAMATH_CALUDE_correct_sunset_time_l3902_390294

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define addition for Time
def Time.add (t1 t2 : Time) : Time :=
  let totalMinutes := t1.hours * 60 + t1.minutes + t2.hours * 60 + t2.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

-- Convert 24-hour format to 12-hour format
def to12HourFormat (t : Time) : Time :=
  if t.hours ≥ 12 then
    { hours := if t.hours = 12 then 12 else t.hours - 12, minutes := t.minutes }
  else
    { hours := if t.hours = 0 then 12 else t.hours, minutes := t.minutes }

theorem correct_sunset_time :
  let sunrise : Time := { hours := 6, minutes := 57 }
  let daylight : Time := { hours := 10, minutes := 24 }
  let sunset := to12HourFormat (Time.add sunrise daylight)
  sunset = { hours := 5, minutes := 21 } := by sorry

end NUMINAMATH_CALUDE_correct_sunset_time_l3902_390294


namespace NUMINAMATH_CALUDE_limit_of_a_is_one_l3902_390217

def a : ℕ+ → ℚ
  | n => if n < 10000 then (2^(n.val+1)) / (2^n.val+1) else ((n.val+1)^2) / (n.val^2+1)

theorem limit_of_a_is_one :
  ∀ ε > 0, ∃ N : ℕ+, ∀ n ≥ N, |a n - 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_a_is_one_l3902_390217


namespace NUMINAMATH_CALUDE_probability_of_pair_l3902_390274

/-- Represents a standard deck of cards -/
def StandardDeck := 52

/-- Represents the number of cards of each rank in a standard deck -/
def CardsPerRank := 4

/-- Represents the number of ranks in a standard deck -/
def NumRanks := 13

/-- Represents the number of cards remaining after removing a pair -/
def RemainingCards := StandardDeck - 2

/-- Represents the number of ways to choose 2 cards from the remaining deck -/
def TotalChoices := (RemainingCards.choose 2)

/-- Represents the number of ranks with 4 cards after removing a pair -/
def FullRanks := NumRanks - 1

/-- Represents the number of ways to form a pair from ranks with 4 cards -/
def PairsFromFullRanks := FullRanks * (CardsPerRank.choose 2)

/-- Represents the number of ways to form a pair from the rank with 2 cards -/
def PairsFromReducedRank := 1

/-- Represents the total number of ways to form a pair -/
def TotalPairs := PairsFromFullRanks + PairsFromReducedRank

/-- The main theorem stating the probability of forming a pair -/
theorem probability_of_pair : 
  (TotalPairs : ℚ) / TotalChoices = 73 / 1225 := by sorry

end NUMINAMATH_CALUDE_probability_of_pair_l3902_390274


namespace NUMINAMATH_CALUDE_sqrt_81_div_3_equals_3_l3902_390235

theorem sqrt_81_div_3_equals_3 : Real.sqrt 81 / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_81_div_3_equals_3_l3902_390235


namespace NUMINAMATH_CALUDE_studentG_score_l3902_390263

-- Define the answer types
inductive Answer
| Correct
| Incorrect
| Unanswered

-- Define the scoring function
def score (a : Answer) : Nat :=
  match a with
  | Answer.Correct => 2
  | Answer.Incorrect => 0
  | Answer.Unanswered => 1

-- Define Student G's answer pattern
def studentG_answers : List Answer :=
  [Answer.Correct, Answer.Incorrect, Answer.Correct, Answer.Correct, Answer.Incorrect, Answer.Correct]

-- Theorem: Student G's total score is 8 points
theorem studentG_score :
  (studentG_answers.map score).sum = 8 := by
  sorry

end NUMINAMATH_CALUDE_studentG_score_l3902_390263


namespace NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l3902_390292

/-- A rectangular prism with specific proportions -/
structure RectangularPrism where
  width : ℝ
  length : ℝ
  height : ℝ
  length_eq : length = 2 * width
  height_eq : height = 3 * width

/-- The number of pairs of parallel edges in a rectangular prism -/
def parallel_edge_pairs (prism : RectangularPrism) : ℕ := 8

/-- Theorem stating that a rectangular prism with the given proportions has 8 pairs of parallel edges -/
theorem rectangular_prism_parallel_edges (prism : RectangularPrism) : 
  parallel_edge_pairs prism = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_parallel_edges_l3902_390292


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3902_390214

theorem cubic_equation_solution : 27^3 + 27^3 + 27^3 = 3^10 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3902_390214


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3902_390277

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 4 = 0) → 
  (x₂^2 + 2*x₂ - 4 = 0) → 
  (x₁ + x₂ = -2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3902_390277


namespace NUMINAMATH_CALUDE_set_a_range_l3902_390299

theorem set_a_range (a : ℝ) : 
  let A : Set ℝ := {x | x^2 - 2*x + a > 0}
  1 ∉ A → a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_set_a_range_l3902_390299


namespace NUMINAMATH_CALUDE_compare_negative_two_and_three_l3902_390211

theorem compare_negative_two_and_three : -2 > -3 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_two_and_three_l3902_390211


namespace NUMINAMATH_CALUDE_wage_difference_l3902_390255

theorem wage_difference (w1 w2 : ℝ) 
  (h1 : w1 > 0) 
  (h2 : w2 > 0) 
  (h3 : 0.4 * w2 = 1.6 * (0.2 * w1)) : 
  (w1 - w2) / w1 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_wage_difference_l3902_390255


namespace NUMINAMATH_CALUDE_number_difference_l3902_390244

theorem number_difference (A B : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : 0.075 * A = 0.125 * B) (h4 : A = 2430) : A - B = 972 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3902_390244


namespace NUMINAMATH_CALUDE_polynomial_coefficients_l3902_390268

theorem polynomial_coefficients 
  (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) : 
  (∀ x : ℝ, (x + 2) * (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  (a₁ - a₂ + a₃ - a₄ + a₅ - a₆ = 241 ∧ a₂ = -70) := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_l3902_390268


namespace NUMINAMATH_CALUDE_last_element_proof_l3902_390210

def first_row (n : ℕ) : ℕ := 2*n - 1

def third_row (n : ℕ) : ℕ := (first_row n) * (first_row n)^2 - (first_row n)

theorem last_element_proof : third_row 5 = 720 := by
  sorry

end NUMINAMATH_CALUDE_last_element_proof_l3902_390210


namespace NUMINAMATH_CALUDE_chickens_and_rabbits_l3902_390226

theorem chickens_and_rabbits (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 40) 
  (h2 : total_legs = 108) : 
  ∃ (chickens rabbits : ℕ), 
    chickens + rabbits = total_animals ∧ 
    2 * chickens + 4 * rabbits = total_legs ∧ 
    chickens = 26 ∧ 
    rabbits = 14 := by
  sorry

end NUMINAMATH_CALUDE_chickens_and_rabbits_l3902_390226


namespace NUMINAMATH_CALUDE_pet_store_snake_distribution_l3902_390265

/-- Given a total number of snakes and cages, calculate the number of snakes per cage -/
def snakesPerCage (totalSnakes : ℕ) (totalCages : ℕ) : ℕ :=
  totalSnakes / totalCages

theorem pet_store_snake_distribution :
  let totalSnakes : ℕ := 4
  let totalCages : ℕ := 2
  snakesPerCage totalSnakes totalCages = 2 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_snake_distribution_l3902_390265


namespace NUMINAMATH_CALUDE_x_intercept_distance_l3902_390205

/-- Given two lines intersecting at (8, 20) with slopes 4 and -2,
    the distance between their x-intercepts is 15. -/
theorem x_intercept_distance (line1 line2 : ℝ → ℝ) : 
  (∀ x, line1 x = 4 * x - 12) →
  (∀ x, line2 x = -2 * x + 36) →
  line1 8 = 20 →
  line2 8 = 20 →
  |line1⁻¹ 0 - line2⁻¹ 0| = 15 :=
sorry

end NUMINAMATH_CALUDE_x_intercept_distance_l3902_390205


namespace NUMINAMATH_CALUDE_absolute_value_squared_l3902_390282

theorem absolute_value_squared (a b : ℝ) : |a| > |b| → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_squared_l3902_390282


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3902_390242

theorem quadratic_inequality (x : ℝ) : x ≥ 1 → x^2 + 3*x - 2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3902_390242


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l3902_390281

theorem merchant_profit_percentage
  (markup_rate : ℝ)
  (discount_rate : ℝ)
  (h_markup : markup_rate = 0.40)
  (h_discount : discount_rate = 0.15) :
  let marked_price := 1 + markup_rate
  let selling_price := marked_price * (1 - discount_rate)
  let profit_percentage := (selling_price - 1) * 100
  profit_percentage = 19 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_percentage_l3902_390281


namespace NUMINAMATH_CALUDE_distance_on_parametric_line_l3902_390230

/-- The distance between two points on a parametric line --/
theorem distance_on_parametric_line :
  let line : ℝ → ℝ × ℝ := λ t ↦ (2 + 3*t, -1 + t)
  let point1 := line 0
  let point2 := line 1
  (point1.1 - point2.1)^2 + (point1.2 - point2.2)^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_on_parametric_line_l3902_390230


namespace NUMINAMATH_CALUDE_unique_g₅₀_equals_18_l3902_390202

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range n)).card

-- Define g₁
def g₁ (n : ℕ) : ℕ := 3 * num_divisors n

-- Define gⱼ recursively
def g (j n : ℕ) : ℕ :=
  match j with
  | 0 => n
  | j+1 => g₁ (g j n)

-- State the theorem
theorem unique_g₅₀_equals_18 :
  ∃! n : ℕ, n ≤ 25 ∧ g 50 n = 18 :=
sorry

end NUMINAMATH_CALUDE_unique_g₅₀_equals_18_l3902_390202


namespace NUMINAMATH_CALUDE_aunt_gemma_dog_food_duration_l3902_390267

/-- Calculates the number of days dog food will last given the number of dogs, 
    feeding frequency, food consumption per meal, and amount of food bought. -/
def dogFoodDuration (numDogs : ℕ) (feedingsPerDay : ℕ) (gramsPerMeal : ℕ) 
                    (numSacks : ℕ) (kgPerSack : ℕ) : ℕ :=
  let dailyConsumptionGrams := numDogs * feedingsPerDay * gramsPerMeal
  let totalFoodKg := numSacks * kgPerSack
  totalFoodKg * 1000 / dailyConsumptionGrams

theorem aunt_gemma_dog_food_duration :
  dogFoodDuration 4 2 250 2 50 = 50 := by
  sorry

#eval dogFoodDuration 4 2 250 2 50

end NUMINAMATH_CALUDE_aunt_gemma_dog_food_duration_l3902_390267


namespace NUMINAMATH_CALUDE_ellipse_equation_l3902_390218

/-- Given an ellipse centered at the origin with eccentricity e = 1/2, 
    and one of its foci coinciding with the focus of the parabola y^2 = -4x,
    prove that the equation of this ellipse is x^2/4 + y^2/3 = 1 -/
theorem ellipse_equation (e : ℝ) (f : ℝ × ℝ) :
  e = (1 : ℝ) / 2 →
  f = (-1, 0) →
  ∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1) ↔ 
    (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
      x^2 / a^2 + y^2 / b^2 = 1 ∧
      e = (f.1^2 + f.2^2).sqrt / a) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3902_390218


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3902_390200

/-- Given four points in 3D space, prove that the intersection of lines AB and CD is at the specified point. -/
theorem intersection_of_lines (A B C D : ℝ × ℝ × ℝ) : 
  A = (5, -3, 2) →
  B = (15, -13, 7) →
  C = (2, 4, -5) →
  D = (4, -1, 15) →
  ∃ (t s : ℝ), 
    (5 + 10*t, -3 - 10*t, 2 + 5*t) = (2 + 2*s, 4 - 5*s, -5 + 20*s) ∧
    (5 + 10*t, -3 - 10*t, 2 + 5*t) = (23/3, -19/3, 7/3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3902_390200


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l3902_390246

theorem chinese_remainder_theorem_example :
  ∃ x : ℤ, (x ≡ 1 [ZMOD 3] ∧
             x ≡ -1 [ZMOD 5] ∧
             x ≡ 2 [ZMOD 7] ∧
             x ≡ -2 [ZMOD 11]) ↔
            x ≡ 394 [ZMOD 1155] := by
  sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l3902_390246


namespace NUMINAMATH_CALUDE_sum_of_cubes_nonnegative_l3902_390296

theorem sum_of_cubes_nonnegative (n : ℤ) (a b : ℚ) 
  (h1 : n > 1) 
  (h2 : n = a^3 + b^3) : 
  ∃ (x y : ℚ), x ≥ 0 ∧ y ≥ 0 ∧ n = x^3 + y^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_nonnegative_l3902_390296


namespace NUMINAMATH_CALUDE_sin_cos_difference_sin_negative_main_theorem_l3902_390248

theorem sin_cos_difference (x y : Real) : 
  Real.sin (x * π / 180) * Real.cos (y * π / 180) - Real.cos (x * π / 180) * Real.sin (y * π / 180) = 
  Real.sin ((x - y) * π / 180) :=
sorry

theorem sin_negative (x : Real) : Real.sin (-x) = -Real.sin x :=
sorry

theorem main_theorem : 
  Real.sin (24 * π / 180) * Real.cos (54 * π / 180) - Real.cos (24 * π / 180) * Real.sin (54 * π / 180) = -1/2 :=
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_sin_negative_main_theorem_l3902_390248


namespace NUMINAMATH_CALUDE_remainder_mod_five_l3902_390256

theorem remainder_mod_five : (1234 * 1456 * 1789 * 2005 + 123) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_mod_five_l3902_390256


namespace NUMINAMATH_CALUDE_fixed_point_on_all_lines_fixed_point_unique_l3902_390283

/-- The fixed point through which all lines of the form ax + y + 1 = 0 pass -/
def fixed_point : ℝ × ℝ := (0, -1)

/-- The equation of the line ax + y + 1 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x + y + 1 = 0

theorem fixed_point_on_all_lines :
  ∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2) :=
by sorry

theorem fixed_point_unique :
  ∀ x y : ℝ, (∀ a : ℝ, line_equation a x y) → (x, y) = fixed_point :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_on_all_lines_fixed_point_unique_l3902_390283


namespace NUMINAMATH_CALUDE_increasing_function_condition_l3902_390227

theorem increasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → ((a - 1) * x + 2) < ((a - 1) * y + 2)) →
  a > 1 := by sorry

end NUMINAMATH_CALUDE_increasing_function_condition_l3902_390227


namespace NUMINAMATH_CALUDE_f_of_2_equals_5_l3902_390233

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x - 1

-- State the theorem
theorem f_of_2_equals_5 : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_of_2_equals_5_l3902_390233


namespace NUMINAMATH_CALUDE_triangle_ratio_l3902_390207

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  A = π / 3 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A →
  b / Real.sin B = c / Real.sin C →
  b / Real.sin B = a / Real.sin A →
  (a + 2 * b - 3 * c) / (Real.sin A + 2 * Real.sin B - 3 * Real.sin C) = 2 * Real.sqrt 39 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3902_390207


namespace NUMINAMATH_CALUDE_system_solution_ratio_l3902_390250

theorem system_solution_ratio (x y c d : ℝ) (h1 : 4*x - 2*y = c) (h2 : 6*y - 12*x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l3902_390250


namespace NUMINAMATH_CALUDE_prob_not_losing_l3902_390257

/-- The probability of Hou Yifan winning a chess match against a computer -/
def prob_win : ℝ := 0.65

/-- The probability of a draw in a chess match between Hou Yifan and a computer -/
def prob_draw : ℝ := 0.25

/-- Theorem: The probability of Hou Yifan not losing is 0.9 -/
theorem prob_not_losing : prob_win + prob_draw = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_losing_l3902_390257


namespace NUMINAMATH_CALUDE_die_roll_probability_l3902_390291

theorem die_roll_probability : 
  let p : ℚ := 1/3  -- probability of rolling a number divisible by 3
  let n : ℕ := 8    -- number of rolls
  1 - (1 - p)^n = 6305/6561 := by sorry

end NUMINAMATH_CALUDE_die_roll_probability_l3902_390291


namespace NUMINAMATH_CALUDE_mary_remaining_money_l3902_390251

-- Define the initial amount Mary received
def initial_amount : ℚ := 150

-- Define the original price of the video game
def game_price : ℚ := 60

-- Define the discount rate for the video game
def game_discount_rate : ℚ := 0.15

-- Define the percentage spent on goggles
def goggles_spend_rate : ℚ := 0.20

-- Define the sales tax rate for the goggles
def goggles_tax_rate : ℚ := 0.08

-- Function to calculate the discounted price of the video game
def discounted_game_price : ℚ :=
  game_price * (1 - game_discount_rate)

-- Function to calculate the amount left after buying the video game
def amount_after_game : ℚ :=
  initial_amount - discounted_game_price

-- Function to calculate the price of the goggles before tax
def goggles_price_before_tax : ℚ :=
  amount_after_game * goggles_spend_rate

-- Function to calculate the total price of the goggles including tax
def goggles_total_price : ℚ :=
  goggles_price_before_tax * (1 + goggles_tax_rate)

-- Theorem stating that Mary has $77.62 left after her shopping trip
theorem mary_remaining_money :
  initial_amount - discounted_game_price - goggles_total_price = 77.62 := by
  sorry

end NUMINAMATH_CALUDE_mary_remaining_money_l3902_390251


namespace NUMINAMATH_CALUDE_lcm_gcf_product_20_90_l3902_390212

theorem lcm_gcf_product_20_90 : Nat.lcm 20 90 * Nat.gcd 20 90 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_product_20_90_l3902_390212


namespace NUMINAMATH_CALUDE_valid_sequences_count_l3902_390223

/-- A transformation on a regular hexagon -/
inductive HexagonTransform
| T1  -- 60° clockwise rotation
| T2  -- 60° counterclockwise rotation
| T3  -- reflection across x-axis
| T4  -- reflection across y-axis

/-- A sequence of transformations -/
def TransformSequence := List HexagonTransform

/-- The identity transformation -/
def identity : TransformSequence := []

/-- Applies a single transformation to a sequence -/
def applyTransform (t : HexagonTransform) (s : TransformSequence) : TransformSequence :=
  t :: s

/-- Checks if a sequence of transformations results in the identity transformation -/
def isIdentity (s : TransformSequence) : Bool :=
  sorry

/-- Counts the number of valid 18-transformation sequences -/
def countValidSequences : Nat :=
  sorry

/-- Main theorem: There are 286 valid sequences of 18 transformations -/
theorem valid_sequences_count : countValidSequences = 286 := by
  sorry

end NUMINAMATH_CALUDE_valid_sequences_count_l3902_390223


namespace NUMINAMATH_CALUDE_remainder_proof_l3902_390222

theorem remainder_proof (a b c : ℕ) : 
  a < 10 → b < 10 → c < 10 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 10 = 2 →
  (7 * c) % 10 = 3 →
  (8 * b) % 10 = (4 + b) % 10 →
  (2 * a + b + 3 * c) % 10 = 1 :=
by sorry

end NUMINAMATH_CALUDE_remainder_proof_l3902_390222


namespace NUMINAMATH_CALUDE_first_class_size_l3902_390280

/-- The number of students in the second class -/
def second_class_students : ℕ := 48

/-- The average marks of the first class -/
def first_class_average : ℚ := 60

/-- The average marks of the second class -/
def second_class_average : ℚ := 58

/-- The average marks of all students -/
def total_average : ℚ := 59067961165048544 / 1000000000000000

/-- The number of students in the first class -/
def first_class_students : ℕ := 55

theorem first_class_size :
  (first_class_students * first_class_average + second_class_students * second_class_average) / 
  (first_class_students + second_class_students) = total_average :=
sorry

end NUMINAMATH_CALUDE_first_class_size_l3902_390280


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l3902_390239

theorem abs_inequality_solution_set :
  {x : ℝ | |2*x + 1| < 3} = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l3902_390239


namespace NUMINAMATH_CALUDE_bigger_part_of_52_l3902_390298

theorem bigger_part_of_52 (x y : ℕ) (h1 : x + y = 52) (h2 : 10 * x + 22 * y = 780) :
  max x y = 30 := by sorry

end NUMINAMATH_CALUDE_bigger_part_of_52_l3902_390298


namespace NUMINAMATH_CALUDE_total_wrapping_paper_l3902_390206

/-- The amount of wrapping paper needed for three presents -/
def wrapping_paper_needed (first_present second_present third_present : ℚ) : ℚ :=
  first_present + second_present + third_present

/-- Theorem stating the total amount of wrapping paper needed for three presents
    given specific conditions -/
theorem total_wrapping_paper :
  let first_present : ℚ := 2
  let second_present : ℚ := 3/4 * first_present
  let third_present : ℚ := first_present + second_present
  wrapping_paper_needed first_present second_present third_present = 7 := by
sorry

end NUMINAMATH_CALUDE_total_wrapping_paper_l3902_390206


namespace NUMINAMATH_CALUDE_lecture_schedules_count_l3902_390249

/-- Represents the number of lecturers --/
def num_lecturers : ℕ := 8

/-- Represents the number of lecturer pairs with order requirements --/
def num_ordered_pairs : ℕ := 2

/-- Calculates the number of valid lecture schedules --/
def num_valid_schedules : ℕ := (Nat.factorial num_lecturers) / (2^num_ordered_pairs)

/-- Theorem stating the number of valid lecture schedules --/
theorem lecture_schedules_count : num_valid_schedules = 10080 := by
  sorry

end NUMINAMATH_CALUDE_lecture_schedules_count_l3902_390249


namespace NUMINAMATH_CALUDE_base_of_first_term_l3902_390236

theorem base_of_first_term (base x y : ℕ) : 
  base ^ x * 4 ^ y = 19683 → 
  x - y = 9 → 
  x = 9 → 
  base = 3 := by
sorry

end NUMINAMATH_CALUDE_base_of_first_term_l3902_390236


namespace NUMINAMATH_CALUDE_partnership_profit_l3902_390293

/-- Represents a partnership between two individuals -/
structure Partnership where
  investment_ratio : ℕ  -- Ratio of investments (larger : smaller)
  time_ratio : ℕ        -- Ratio of investment periods (longer : shorter)
  smaller_profit : ℕ    -- Profit of the partner with smaller investment

/-- Calculates the total profit of a partnership -/
def total_profit (p : Partnership) : ℕ :=
  let profit_ratio := p.investment_ratio * p.time_ratio + 1
  profit_ratio * p.smaller_profit

/-- Theorem: For a partnership where one partner's investment is triple and 
    investment period is double that of the other, if the partner with 
    smaller investment receives 7000, the total profit is 49000 -/
theorem partnership_profit : 
  ∀ (p : Partnership), 
    p.investment_ratio = 3 → 
    p.time_ratio = 2 → 
    p.smaller_profit = 7000 → 
    total_profit p = 49000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l3902_390293


namespace NUMINAMATH_CALUDE_parity_of_D_2021_2022_2023_l3902_390295

def D : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | n+3 => D (n+2) + D n

theorem parity_of_D_2021_2022_2023 :
  (D 2021 % 2 = 0) ∧ (D 2022 % 2 = 1) ∧ (D 2023 % 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_parity_of_D_2021_2022_2023_l3902_390295


namespace NUMINAMATH_CALUDE_line_equation_from_slope_and_intercept_l3902_390243

/-- Given a line with slope 6 and y-intercept -4, its equation is 6x - y - 4 = 0 -/
theorem line_equation_from_slope_and_intercept :
  ∀ (f : ℝ → ℝ),
  (∀ x y : ℝ, f y - f x = 6 * (y - x)) →  -- slope is 6
  (f 0 = -4) →                           -- y-intercept is -4
  ∀ x : ℝ, 6 * x - f x - 4 = 0 :=
by sorry


end NUMINAMATH_CALUDE_line_equation_from_slope_and_intercept_l3902_390243


namespace NUMINAMATH_CALUDE_expression_undefined_iff_x_eq_11_l3902_390270

theorem expression_undefined_iff_x_eq_11 (x : ℝ) :
  ¬ (∃ y : ℝ, y = (3 * x^3 + 4) / (x^2 - 22*x + 121)) ↔ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_undefined_iff_x_eq_11_l3902_390270


namespace NUMINAMATH_CALUDE_f_of_10_l3902_390247

theorem f_of_10 (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f x + f (2*x + y) + 5*x*y = f (3*x - y) + 2*x^2 + 1) :
  f 10 = -49 := by
sorry

end NUMINAMATH_CALUDE_f_of_10_l3902_390247


namespace NUMINAMATH_CALUDE_initial_white_cookies_l3902_390220

def cookie_problem (w : ℕ) : Prop :=
  let b := w + 50
  let remaining_black := b / 2
  let remaining_white := w / 4
  remaining_black + remaining_white = 85

theorem initial_white_cookies : ∃ w : ℕ, cookie_problem w ∧ w = 80 := by
  sorry

end NUMINAMATH_CALUDE_initial_white_cookies_l3902_390220


namespace NUMINAMATH_CALUDE_percentage_increase_l3902_390219

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 1500 → final = 1800 → (final - initial) / initial * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3902_390219


namespace NUMINAMATH_CALUDE_isabellas_paintable_area_l3902_390241

/-- Calculates the total paintable area of walls in multiple bedrooms. -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (length * height + width * height)
  let paintable_area := wall_area - unpaintable_area
  num_bedrooms * paintable_area

/-- Proves that the total paintable area for Isabella's bedrooms is 1552 square feet. -/
theorem isabellas_paintable_area :
  total_paintable_area 4 14 12 9 80 = 1552 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_paintable_area_l3902_390241


namespace NUMINAMATH_CALUDE_gcd_of_45_and_75_l3902_390275

theorem gcd_of_45_and_75 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_45_and_75_l3902_390275


namespace NUMINAMATH_CALUDE_triangle_angle_range_l3902_390231

theorem triangle_angle_range (α β γ : Real) : 
  (α + β + γ = 180) →  -- Sum of angles in a triangle
  (α ≥ β) → 
  (β ≥ γ) → 
  (α = 2 * γ) → 
  (45 ≤ β) ∧ (β ≤ 72) := by
  sorry

#check triangle_angle_range

end NUMINAMATH_CALUDE_triangle_angle_range_l3902_390231


namespace NUMINAMATH_CALUDE_intersection_M_N_l3902_390225

def M : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def N : Set ℝ := {-3, -2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3902_390225


namespace NUMINAMATH_CALUDE_negation_of_some_primes_even_l3902_390232

theorem negation_of_some_primes_even :
  (¬ ∃ p, Nat.Prime p ∧ Even p) ↔ (∀ p, Nat.Prime p → Odd p) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_some_primes_even_l3902_390232


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3902_390238

/-- An arithmetic sequence with given first term and 17th term -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ a 17 = 66 ∧ ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_properties (a : ℕ → ℤ) 
  (h : arithmetic_sequence a) : 
  (∀ n : ℕ, a n = 4 * n - 2) ∧ 
  (¬ ∃ n : ℕ, a n = 88) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3902_390238


namespace NUMINAMATH_CALUDE_train_journey_speed_l3902_390266

/-- Given a train journey with the following conditions:
  - The total distance is 5x km
  - The first part of the journey is x km at 40 kmph
  - The second part of the journey is 2x km at speed v
  - The average speed for the entire journey is 40 kmph
  Prove that the speed v during the second part of the journey is 20 kmph -/
theorem train_journey_speed (x : ℝ) (v : ℝ) 
  (h1 : x > 0) 
  (h2 : x / 40 + 2 * x / v = 5 * x / 40) : v = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_speed_l3902_390266


namespace NUMINAMATH_CALUDE_trailing_zeros_remainder_l3902_390278

/-- Calculate the number of trailing zeros in the product of factorials from 1 to n -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- The remainder when the number of trailing zeros in 1!2!3!...50! is divided by 500 -/
theorem trailing_zeros_remainder : trailingZeros 50 % 500 = 12 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_remainder_l3902_390278


namespace NUMINAMATH_CALUDE_expression_evaluation_l3902_390209

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -1
  (2*x + y) * (2*x - y) - 3*(2*x^2 - x*y) + y^2 = -14 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3902_390209


namespace NUMINAMATH_CALUDE_range_of_k_l3902_390297

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < k + 1}

-- Define the complement of A in ℝ
def C_ℝA : Set ℝ := {x | 1 < x ∧ x < 3}

-- Theorem statement
theorem range_of_k (k : ℝ) :
  (C_ℝA ∩ B k = ∅) → (k ≤ 0 ∨ k ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_l3902_390297


namespace NUMINAMATH_CALUDE_square_sum_from_conditions_l3902_390213

theorem square_sum_from_conditions (x y : ℝ) 
  (h1 : (x + y)^2 = 36) 
  (h2 : x * y = 8) : 
  x^2 + y^2 = 20 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_conditions_l3902_390213


namespace NUMINAMATH_CALUDE_age_difference_proof_l3902_390289

def age_difference (a b c : ℕ) : ℕ := (a + b) - (b + c)

theorem age_difference_proof (a b c : ℕ) (h : c = a - 11) :
  age_difference a b c = 11 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3902_390289


namespace NUMINAMATH_CALUDE_job_completion_time_l3902_390264

/-- The time taken for machines to complete a job given specific conditions -/
theorem job_completion_time : 
  -- Machine R completion time
  let r_time : ℝ := 36
  -- Machine S completion time
  let s_time : ℝ := 2
  -- Number of each type of machine used
  let n : ℝ := 0.9473684210526315
  -- Total rate of job completion
  let total_rate : ℝ := n * (1 / r_time) + n * (1 / s_time)
  -- Time taken to complete the job
  let completion_time : ℝ := 1 / total_rate
  -- Proof that the completion time is 2 hours
  completion_time = 2 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l3902_390264


namespace NUMINAMATH_CALUDE_six_containing_triangles_l3902_390216

/-- Represents a quadrilateral composed of small equilateral triangles -/
structure TriangleQuadrilateral where
  /-- The total number of small equilateral triangles in the quadrilateral -/
  total_triangles : ℕ
  /-- The number of small triangles per side of the largest equilateral triangle -/
  max_side_length : ℕ
  /-- Assertion that the total number of triangles is 18 -/
  h_total : total_triangles = 18

/-- Counts the number of equilateral triangles containing a marked triangle -/
def count_containing_triangles (q : TriangleQuadrilateral) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 6 equilateral triangles containing the marked triangle -/
theorem six_containing_triangles (q : TriangleQuadrilateral) :
  count_containing_triangles q = 6 :=
sorry

end NUMINAMATH_CALUDE_six_containing_triangles_l3902_390216


namespace NUMINAMATH_CALUDE_equal_roots_implies_m_l3902_390290

/-- The polynomial function in question -/
def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^3 + 9 * x^2 - 135 * x + m

/-- Predicate to check if a polynomial has a repeated root -/
def has_repeated_root (p : ℝ → ℝ) : Prop :=
  ∃ r : ℝ, p r = 0 ∧ (deriv p) r = 0

theorem equal_roots_implies_m (m : ℝ) :
  has_repeated_root (f m) → m > 0 → m = 22275 := by
  sorry

#check equal_roots_implies_m

end NUMINAMATH_CALUDE_equal_roots_implies_m_l3902_390290


namespace NUMINAMATH_CALUDE_document_word_count_l3902_390237

/-- Given Barbara's typing speeds and time, calculate the number of words in the document -/
theorem document_word_count 
  (original_speed : ℕ) 
  (speed_reduction : ℕ) 
  (typing_time : ℕ) 
  (h1 : original_speed = 212)
  (h2 : speed_reduction = 40)
  (h3 : typing_time = 20) : 
  (original_speed - speed_reduction) * typing_time = 3440 :=
by sorry

end NUMINAMATH_CALUDE_document_word_count_l3902_390237


namespace NUMINAMATH_CALUDE_mary_has_five_candies_l3902_390276

/-- The number of candies Mary has on Halloween -/
def marys_candies (bob_candies sue_candies john_candies sam_candies total_candies : ℕ) : ℕ :=
  total_candies - (bob_candies + sue_candies + john_candies + sam_candies)

/-- Theorem: Mary has 5 candies given the Halloween candy distribution -/
theorem mary_has_five_candies :
  marys_candies 10 20 5 10 50 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mary_has_five_candies_l3902_390276


namespace NUMINAMATH_CALUDE_roger_birthday_money_l3902_390288

/-- Calculates the amount of birthday money Roger received -/
def birthday_money (initial_amount spent_amount final_amount : ℤ) : ℤ :=
  final_amount - initial_amount + spent_amount

/-- Proves that Roger received 28 dollars for his birthday -/
theorem roger_birthday_money :
  birthday_money 16 25 19 = 28 := by
  sorry

end NUMINAMATH_CALUDE_roger_birthday_money_l3902_390288


namespace NUMINAMATH_CALUDE_fundraiser_item_price_l3902_390287

theorem fundraiser_item_price 
  (num_brownie_students : ℕ) 
  (num_cookie_students : ℕ) 
  (num_donut_students : ℕ) 
  (brownies_per_student : ℕ) 
  (cookies_per_student : ℕ) 
  (donuts_per_student : ℕ) 
  (total_amount_raised : ℚ) : 
  num_brownie_students = 30 →
  num_cookie_students = 20 →
  num_donut_students = 15 →
  brownies_per_student = 12 →
  cookies_per_student = 24 →
  donuts_per_student = 12 →
  total_amount_raised = 2040 →
  (total_amount_raised / (num_brownie_students * brownies_per_student + 
                          num_cookie_students * cookies_per_student + 
                          num_donut_students * donuts_per_student) : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_item_price_l3902_390287


namespace NUMINAMATH_CALUDE_last_score_must_be_86_l3902_390269

def scores : List ℕ := [73, 78, 84, 86, 97]

def is_integer (n : ℚ) : Prop := ∃ m : ℤ, n = m

def average_is_integer (entered_scores : List ℕ) : Prop :=
  ∀ k : ℕ, k > 0 → k ≤ entered_scores.length →
    is_integer ((entered_scores.take k).sum / k)

theorem last_score_must_be_86 :
  ∀ perm : List ℕ, perm.length = 5 →
  perm.toFinset = scores.toFinset →
  average_is_integer perm →
  perm.getLast? = some 86 :=
sorry

end NUMINAMATH_CALUDE_last_score_must_be_86_l3902_390269


namespace NUMINAMATH_CALUDE_blocks_differing_in_two_ways_l3902_390201

/-- Represents the number of options for each property of a block -/
structure BlockOptions :=
  (material : ℕ)
  (size : ℕ)
  (color : ℕ)
  (shape : ℕ)

/-- Calculates the number of blocks that differ from a specific block in exactly k ways -/
def blocksWithExactDifferences (options : BlockOptions) (k : ℕ) : ℕ :=
  sorry

/-- The specific block options for our problem -/
def ourBlockOptions : BlockOptions :=
  { material := 2, size := 4, color := 4, shape := 4 }

/-- The main theorem: prove that there are 30 blocks differing in exactly 2 ways -/
theorem blocks_differing_in_two_ways :
  blocksWithExactDifferences ourBlockOptions 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_blocks_differing_in_two_ways_l3902_390201


namespace NUMINAMATH_CALUDE_find_number_l3902_390240

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 63 :=
  sorry

end NUMINAMATH_CALUDE_find_number_l3902_390240


namespace NUMINAMATH_CALUDE_simplify_expression_l3902_390229

theorem simplify_expression : (2^5 + 7^3) * (2^3 - (-2)^2)^8 = 24576000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3902_390229


namespace NUMINAMATH_CALUDE_tea_cost_price_l3902_390273

/-- The cost price of 80 kg of tea per kg -/
def C : ℝ := 15

/-- The theorem stating the cost price of 80 kg of tea per kg -/
theorem tea_cost_price :
  -- 80 kg of tea is mixed with 20 kg of tea at cost price of 20 per kg
  -- The sale price of the mixed tea is 20 per kg
  -- The trader wants to earn a profit of 25%
  (80 * C + 20 * 20) * 1.25 = 100 * 20 :=
by
  sorry

end NUMINAMATH_CALUDE_tea_cost_price_l3902_390273


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3902_390254

def sachin_age : ℕ := 28
def age_difference : ℕ := 8

def rahul_age : ℕ := sachin_age + age_difference

theorem age_ratio_proof : 
  (sachin_age : ℚ) / (rahul_age : ℚ) = 7 / 9 := by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3902_390254


namespace NUMINAMATH_CALUDE_right_focus_coordinates_l3902_390258

/-- The coordinates of the right focus of a hyperbola with equation x^2 - 2y^2 = 1 -/
theorem right_focus_coordinates :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 - 2*y^2 = 1}
  ∃ (f : ℝ × ℝ), f ∈ hyperbola ∧ f.1 > 0 ∧ f.2 = 0 ∧ 
    ∀ (p : ℝ × ℝ), p ∈ hyperbola ∧ p.1 > 0 ∧ p.2 = 0 → p = f ∧
    f = (Real.sqrt (3/2), 0) :=
by sorry

end NUMINAMATH_CALUDE_right_focus_coordinates_l3902_390258


namespace NUMINAMATH_CALUDE_total_age_in_three_years_l3902_390221

def age_problem (sam sue kendra : ℕ) : Prop :=
  sam = 2 * sue ∧ 
  kendra = 3 * sam ∧ 
  kendra = 18

theorem total_age_in_three_years 
  (sam sue kendra : ℕ) 
  (h : age_problem sam sue kendra) : 
  (sue + 3) + (sam + 3) + (kendra + 3) = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_age_in_three_years_l3902_390221


namespace NUMINAMATH_CALUDE_triangle_theorem_l3902_390224

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle. -/
theorem triangle_theorem (t : Triangle) 
  (h1 : 3 * t.a * Real.cos t.A - t.c * Real.cos t.B + t.b * Real.cos t.C = 0)
  (h2 : t.a = 2 * Real.sqrt 3)
  (h3 : Real.cos t.B + Real.cos t.C = 2 * Real.sqrt 3 / 3) :
  Real.cos t.A = 1/3 ∧ t.c = 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3902_390224


namespace NUMINAMATH_CALUDE_shiela_animal_drawings_l3902_390284

/-- Proves that each neighbor receives 8 animal drawings when Shiela distributes
    96 drawings equally among 12 neighbors. -/
theorem shiela_animal_drawings (neighbors : ℕ) (drawings : ℕ) (h1 : neighbors = 12) (h2 : drawings = 96) :
  drawings / neighbors = 8 := by
  sorry

end NUMINAMATH_CALUDE_shiela_animal_drawings_l3902_390284


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3902_390204

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes : 
  distribute_balls 5 4 = 56 := by
  sorry

#eval distribute_balls 5 4

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3902_390204


namespace NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l3902_390261

/-- Represents the daily sales and profit of a clothing store --/
structure ClothingStore where
  initialSales : ℕ
  initialProfit : ℝ
  priceReductionEffect : ℝ → ℕ
  targetProfit : ℝ

/-- Calculates the daily profit based on price reduction --/
def dailyProfit (store : ClothingStore) (priceReduction : ℝ) : ℝ :=
  let newSales := store.initialSales + store.priceReductionEffect priceReduction
  let newProfit := store.initialProfit - priceReduction
  newSales * newProfit

/-- Theorem stating that a $20 price reduction achieves the target profit --/
theorem price_reduction_achieves_target_profit (store : ClothingStore) 
  (h1 : store.initialSales = 20)
  (h2 : store.initialProfit = 40)
  (h3 : ∀ x, store.priceReductionEffect x = (8 / 4 : ℝ) * x)
  (h4 : store.targetProfit = 1200) :
  dailyProfit store 20 = store.targetProfit := by
  sorry


end NUMINAMATH_CALUDE_price_reduction_achieves_target_profit_l3902_390261


namespace NUMINAMATH_CALUDE_heptagon_angle_sum_l3902_390279

/-- A polygon with vertices A, B, C, D, E, F, G -/
structure Heptagon :=
  (A B C D E F G : ℝ × ℝ)

/-- The angle between three points -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

/-- The sum of angles FAD, GBC, BCE, ADG, CEF, AFE, DGB -/
def angle_sum (h : Heptagon) : ℝ :=
  angle h.F h.A h.D +
  angle h.G h.B h.C +
  angle h.B h.C h.E +
  angle h.A h.D h.G +
  angle h.C h.E h.F +
  angle h.A h.F h.E +
  angle h.D h.G h.B

theorem heptagon_angle_sum (h : Heptagon) : angle_sum h = 540 := by sorry

end NUMINAMATH_CALUDE_heptagon_angle_sum_l3902_390279


namespace NUMINAMATH_CALUDE_max_popsicles_for_budget_l3902_390208

/-- Represents the number of popsicles in a pack -/
inductive PackSize
  | single
  | pack4
  | pack6

/-- The cost of a pack of popsicles -/
def packCost : PackSize → ℕ
  | PackSize.single => 1
  | PackSize.pack4 => 3
  | PackSize.pack6 => 4

/-- The number of popsicles in a pack -/
def popsiclesInPack : PackSize → ℕ
  | PackSize.single => 1
  | PackSize.pack4 => 4
  | PackSize.pack6 => 6

/-- A purchase is a list of packs bought -/
def Purchase := List PackSize

/-- The total cost of a purchase -/
def totalCost (purchase : Purchase) : ℕ :=
  purchase.map packCost |>.sum

/-- The total number of popsicles in a purchase -/
def totalPopsicles (purchase : Purchase) : ℕ :=
  purchase.map popsiclesInPack |>.sum

def budget : ℕ := 10

theorem max_popsicles_for_budget :
  ∃ (maxPurchase : Purchase),
    totalCost maxPurchase ≤ budget ∧
    totalPopsicles maxPurchase = 14 ∧
    ∀ (p : Purchase), totalCost p ≤ budget → totalPopsicles p ≤ 14 :=
  sorry


end NUMINAMATH_CALUDE_max_popsicles_for_budget_l3902_390208


namespace NUMINAMATH_CALUDE_x_power_4095_minus_reciprocal_l3902_390272

theorem x_power_4095_minus_reciprocal (x : ℝ) (h : x - 1/x = Real.sqrt 2) :
  x^4095 - 1/x^4095 = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_x_power_4095_minus_reciprocal_l3902_390272


namespace NUMINAMATH_CALUDE_lamppost_combinations_lamppost_problem_l3902_390228

theorem lamppost_combinations : Nat → Nat → Nat
| n, k => Nat.choose n k

theorem lamppost_problem :
  let total_posts : Nat := 11
  let posts_to_turn_off : Nat := 3
  let available_positions : Nat := total_posts - 4  -- Subtracting 2 for each end and 2 for adjacent positions
  lamppost_combinations available_positions posts_to_turn_off = 35 := by
  sorry

end NUMINAMATH_CALUDE_lamppost_combinations_lamppost_problem_l3902_390228


namespace NUMINAMATH_CALUDE_paperboy_delivery_ways_l3902_390253

/-- Represents the number of valid delivery sequences for n houses --/
def P : ℕ → ℕ
| 0 => 1  -- Base case for 0 houses
| 1 => 2  -- Base case for 1 house
| 2 => 4  -- Base case for 2 houses
| 3 => 8  -- Base case for 3 houses
| 4 => 15 -- Base case for 4 houses
| n + 5 => P (n + 4) + P (n + 3) + P (n + 2) + P (n + 1)

/-- The number of houses the paperboy delivers to --/
def num_houses : ℕ := 12

/-- Theorem stating the number of ways to deliver newspapers to 12 houses --/
theorem paperboy_delivery_ways : P num_houses = 2873 := by
  sorry

end NUMINAMATH_CALUDE_paperboy_delivery_ways_l3902_390253


namespace NUMINAMATH_CALUDE_sum_lent_is_1000_l3902_390215

/-- Proves that the sum lent is $1000 given the specified conditions --/
theorem sum_lent_is_1000 (annual_rate : ℝ) (duration : ℝ) (interest_difference : ℝ) :
  annual_rate = 0.06 →
  duration = 8 →
  interest_difference = 520 →
  ∃ (P : ℝ), P * annual_rate * duration = P - interest_difference ∧ P = 1000 := by
  sorry

#check sum_lent_is_1000

end NUMINAMATH_CALUDE_sum_lent_is_1000_l3902_390215


namespace NUMINAMATH_CALUDE_toms_candy_problem_l3902_390203

/-- Tom's candy problem -/
theorem toms_candy_problem (initial : Nat) (bought : Nat) (total : Nat) (friend_gave : Nat) : 
  initial = 2 → 
  bought = 10 → 
  total = 19 → 
  initial + bought + friend_gave = total → 
  friend_gave = 7 := by
  sorry

#check toms_candy_problem

end NUMINAMATH_CALUDE_toms_candy_problem_l3902_390203


namespace NUMINAMATH_CALUDE_divisibility_by_six_l3902_390262

theorem divisibility_by_six (n : ℕ) : ∃ k : ℤ, (17 : ℤ)^n - (11 : ℤ)^n = 6 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_six_l3902_390262


namespace NUMINAMATH_CALUDE_distance_after_pie_is_18_l3902_390234

/-- Calculates the distance driven after buying pie and before stopping for gas -/
def distance_after_pie (total_distance : ℕ) (distance_before_pie : ℕ) (remaining_distance : ℕ) : ℕ :=
  total_distance - distance_before_pie - remaining_distance

/-- Proves that the distance driven after buying pie and before stopping for gas is 18 miles -/
theorem distance_after_pie_is_18 :
  distance_after_pie 78 35 25 = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_pie_is_18_l3902_390234


namespace NUMINAMATH_CALUDE_max_log_function_l3902_390260

theorem max_log_function (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + 2*y = 1/2) :
  ∃ (max_u : ℝ), max_u = 0 ∧ 
  ∀ (u : ℝ), u = Real.log (8*x*y + 4*y^2 + 1) / Real.log (1/2) → u ≤ max_u :=
sorry

end NUMINAMATH_CALUDE_max_log_function_l3902_390260


namespace NUMINAMATH_CALUDE_solution_count_is_correct_l3902_390285

/-- The number of groups of integer solutions for the equation xyz = 2009 -/
def solution_count : ℕ := 72

/-- A function that counts the number of groups of integer solutions for xyz = 2009 -/
noncomputable def count_solutions : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the number of groups of integer solutions for xyz = 2009 is 72 -/
theorem solution_count_is_correct : count_solutions = solution_count := by
  sorry

end NUMINAMATH_CALUDE_solution_count_is_correct_l3902_390285


namespace NUMINAMATH_CALUDE_range_of_f_l3902_390271

-- Define the function f
def f (x : ℝ) : ℝ := 2*x - x^2

-- Define the domain
def domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 3

-- Theorem statement
theorem range_of_f :
  ∃ (y : ℝ), (∃ (x : ℝ), domain x ∧ f x = y) ↔ -3 ≤ y ∧ y ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3902_390271


namespace NUMINAMATH_CALUDE_last_integer_before_100_l3902_390286

def sequence_term (n : ℕ) : ℕ := (16777216 : ℕ) / 2^n

theorem last_integer_before_100 :
  ∃ n : ℕ, sequence_term n = 64 ∧ sequence_term (n + 1) < 100 :=
sorry

end NUMINAMATH_CALUDE_last_integer_before_100_l3902_390286
