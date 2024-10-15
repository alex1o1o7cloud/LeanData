import Mathlib

namespace NUMINAMATH_CALUDE_hotpot_expenditure_theorem_l2366_236679

/-- Represents the expenditure of three people on hotpot base materials. -/
structure HotpotExpenditure where
  a : ℕ  -- number of brands
  m : ℕ  -- price of clear soup flavor
  n : ℕ  -- price of mushroom soup flavor
  spicy_price : ℕ := 25  -- price of spicy flavor

/-- Conditions for the hotpot expenditure problem -/
def valid_expenditure (h : HotpotExpenditure) : Prop :=
  h.a * (h.spicy_price + h.m + h.n) = 1900 ∧
  33 ≤ h.m ∧ h.m < h.n ∧ h.n ≤ 37

/-- The maximum amount Xiao Li could have spent on clear soup and mushroom soup flavors -/
def max_non_spicy_expenditure (h : HotpotExpenditure) : ℕ :=
  700 - h.spicy_price

/-- The main theorem stating the maximum amount Xiao Li could have spent on non-spicy flavors -/
theorem hotpot_expenditure_theorem (h : HotpotExpenditure) 
  (h_valid : valid_expenditure h) : 
  max_non_spicy_expenditure h = 675 := by
  sorry

end NUMINAMATH_CALUDE_hotpot_expenditure_theorem_l2366_236679


namespace NUMINAMATH_CALUDE_factorization_proof_l2366_236686

theorem factorization_proof (a b x y : ℝ) : 
  (4 * a^2 * b - 6 * a * b^2 = 2 * a * b * (2 * a - 3 * b)) ∧ 
  (25 * x^2 - 9 * y^2 = (5 * x + 3 * y) * (5 * x - 3 * y)) ∧ 
  (2 * a^2 * b - 8 * a * b^2 + 8 * b^3 = 2 * b * (a - 2 * b)^2) ∧ 
  ((x + 2) * (x - 8) + 25 = (x - 3)^2) :=
by sorry


end NUMINAMATH_CALUDE_factorization_proof_l2366_236686


namespace NUMINAMATH_CALUDE_representatives_count_l2366_236617

/-- The number of ways to select representatives from male and female students -/
def selectRepresentatives (numMale numFemale numReps : ℕ) (minMale minFemale : ℕ) : ℕ :=
  (numMale.choose (numReps - minFemale) * numFemale.choose minFemale) +
  (numMale.choose minMale * numFemale.choose (numReps - minMale))

/-- Theorem stating the number of ways to select representatives -/
theorem representatives_count :
  selectRepresentatives 5 4 4 2 1 = 100 := by
  sorry

#eval selectRepresentatives 5 4 4 2 1

end NUMINAMATH_CALUDE_representatives_count_l2366_236617


namespace NUMINAMATH_CALUDE_distinct_numbers_probability_l2366_236682

/-- The number of sides on a standard die -/
def numSides : ℕ := 6

/-- The number of dice being rolled -/
def numDice : ℕ := 5

/-- The probability of rolling five standard, six-sided dice and getting five distinct numbers -/
def probabilityDistinctNumbers : ℚ := 5 / 54

theorem distinct_numbers_probability :
  (numSides.factorial / (numSides - numDice).factorial) / numSides ^ numDice = probabilityDistinctNumbers :=
sorry

end NUMINAMATH_CALUDE_distinct_numbers_probability_l2366_236682


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2366_236632

theorem quadratic_equation_solution :
  let x₁ : ℝ := -1 + Real.sqrt 6 / 2
  let x₂ : ℝ := -1 - Real.sqrt 6 / 2
  2 * x₁^2 + 4 * x₁ - 1 = 0 ∧ 2 * x₂^2 + 4 * x₂ - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2366_236632


namespace NUMINAMATH_CALUDE_geoffrey_games_l2366_236622

/-- The number of games Geoffrey bought -/
def num_games : ℕ := sorry

/-- The amount of money Geoffrey had before his birthday -/
def initial_money : ℕ := sorry

/-- The cost of each game -/
def game_cost : ℕ := 35

/-- The amount of money Geoffrey received from his grandmother -/
def grandmother_gift : ℕ := 20

/-- The amount of money Geoffrey received from his aunt -/
def aunt_gift : ℕ := 25

/-- The amount of money Geoffrey received from his uncle -/
def uncle_gift : ℕ := 30

/-- The total amount of money Geoffrey has after receiving gifts -/
def total_money : ℕ := 125

/-- The amount of money Geoffrey has left after buying games -/
def money_left : ℕ := 20

theorem geoffrey_games :
  num_games = 3 ∧
  initial_money + grandmother_gift + aunt_gift + uncle_gift = total_money ∧
  total_money - money_left = num_games * game_cost :=
sorry

end NUMINAMATH_CALUDE_geoffrey_games_l2366_236622


namespace NUMINAMATH_CALUDE_cherry_pies_count_l2366_236643

/-- Represents the types of pies --/
inductive PieType
  | Apple
  | Blueberry
  | Cherry

/-- Calculates the number of cherry pies given the total number of pies and the ratio --/
def cherry_pies (total : ℕ) (apple_ratio : ℕ) (blueberry_ratio : ℕ) (cherry_ratio : ℕ) : ℕ :=
  let ratio_sum := apple_ratio + blueberry_ratio + cherry_ratio
  let pies_per_ratio := total / ratio_sum
  cherry_ratio * pies_per_ratio

/-- Theorem stating that given 30 total pies and a 1:5:4 ratio, there are 12 cherry pies --/
theorem cherry_pies_count : cherry_pies 30 1 5 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pies_count_l2366_236643


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_less_than_1000_l2366_236675

theorem greatest_multiple_of_5_and_7_less_than_1000 :
  ∀ n : ℕ, n < 1000 → n % 5 = 0 → n % 7 = 0 → n ≤ 980 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_less_than_1000_l2366_236675


namespace NUMINAMATH_CALUDE_area_inequality_l2366_236625

/-- A convex n-gon with circumscribed and inscribed circles -/
class ConvexNGon (n : ℕ) where
  /-- The area of the n-gon -/
  area : ℝ
  /-- The area of the circumscribed circle -/
  circumArea : ℝ
  /-- The area of the inscribed circle -/
  inscribedArea : ℝ
  /-- The n-gon is convex -/
  convex : Prop
  /-- The n-gon has a circumscribed circle -/
  hasCircumscribed : Prop
  /-- The n-gon has an inscribed circle -/
  hasInscribed : Prop

/-- Theorem: For a convex n-gon with circumscribed and inscribed circles,
    twice the area of the n-gon is less than the sum of the areas of the circumscribed and inscribed circles -/
theorem area_inequality {n : ℕ} (ngon : ConvexNGon n) :
  2 * ngon.area < ngon.circumArea + ngon.inscribedArea :=
sorry

end NUMINAMATH_CALUDE_area_inequality_l2366_236625


namespace NUMINAMATH_CALUDE_bella_age_is_five_l2366_236676

/-- Bella's age in years -/
def bella_age : ℕ := sorry

/-- Bella's brother's age in years -/
def brother_age : ℕ := sorry

/-- Theorem stating Bella's age given the conditions -/
theorem bella_age_is_five :
  (brother_age = bella_age + 9) →  -- Brother is 9 years older
  (bella_age + brother_age = 19) →  -- Ages add up to 19
  bella_age = 5 := by sorry

end NUMINAMATH_CALUDE_bella_age_is_five_l2366_236676


namespace NUMINAMATH_CALUDE_max_m_value_inequality_proof_l2366_236614

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| + |x + 2|

-- Theorem for part (1)
theorem max_m_value (M : ℝ) : (∀ x, f x ≥ |M + 1|) → M ≤ 4 :=
sorry

-- Theorem for part (2)
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + 2*b + c = 4) : 1 / (a + b) + 1 / (b + c) ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_inequality_proof_l2366_236614


namespace NUMINAMATH_CALUDE_slope_angle_of_parametric_line_l2366_236634

/-- The slope angle of a line given by parametric equations -/
theorem slope_angle_of_parametric_line :
  let x : ℝ → ℝ := λ t ↦ 5 - 3 * t
  let y : ℝ → ℝ := λ t ↦ 3 + Real.sqrt 3 * t
  (∃ α : ℝ, α = 150 * π / 180 ∧
    ∀ t : ℝ, (y t - y 0) / (x t - x 0) = Real.tan α) :=
by sorry

end NUMINAMATH_CALUDE_slope_angle_of_parametric_line_l2366_236634


namespace NUMINAMATH_CALUDE_solve_for_x_l2366_236687

/-- The operation defined for real numbers a, b, c, d -/
def operation (a b c d : ℝ) : ℝ := a * d - b * c

/-- The theorem stating that if the operation on the given matrix equals 2023, then x = 2018 -/
theorem solve_for_x (x : ℝ) : operation (x + 1) (x + 2) (x - 3) (x - 1) = 2023 → x = 2018 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2366_236687


namespace NUMINAMATH_CALUDE_intersection_A_B_l2366_236648

def A : Set ℝ := {0, 1, 2}
def B : Set ℝ := {y | ∃ x, y = Real.exp x}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2366_236648


namespace NUMINAMATH_CALUDE_apple_problem_l2366_236692

theorem apple_problem (older younger : ℕ) 
  (h1 : older - 1 = younger + 1)
  (h2 : older + 1 = 2 * (younger - 1)) :
  older + younger = 12 := by
  sorry

end NUMINAMATH_CALUDE_apple_problem_l2366_236692


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l2366_236691

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- There are 5 balls -/
def num_balls : ℕ := 5

/-- There are 4 boxes -/
def num_boxes : ℕ := 4

/-- The theorem stating that there are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes : 
  distribute_balls num_balls num_boxes = 56 := by sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l2366_236691


namespace NUMINAMATH_CALUDE_parabola_x_intercepts_l2366_236661

/-- The number of x-intercepts of the parabola y = 3x^2 - 4x + 1 -/
theorem parabola_x_intercepts :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 1
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end NUMINAMATH_CALUDE_parabola_x_intercepts_l2366_236661


namespace NUMINAMATH_CALUDE_sum_of_squares_divisibility_l2366_236665

theorem sum_of_squares_divisibility (n : ℕ) : 
  (∃ k : ℕ, n = 6 * k - 1 ∨ n = 6 * k + 1) ↔ 
  (∃ m : ℕ, n * (n + 1) * (2 * n + 1) = 6 * m) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisibility_l2366_236665


namespace NUMINAMATH_CALUDE_end_on_multiple_of_four_probability_l2366_236642

def num_cards : ℕ := 12
def move_right_prob : ℚ := 1/2
def move_left_prob : ℚ := 1/4
def move_two_right_prob : ℚ := 1/4

def is_multiple_of_four (n : ℕ) : Prop := ∃ k, n = 4 * k

theorem end_on_multiple_of_four_probability :
  let total_outcomes := num_cards * 4 * 4  -- 12 cards * 4 spinner outcomes * 4 spinner outcomes
  let favorable_outcomes := 21  -- This is derived from the problem constraints
  (favorable_outcomes : ℚ) / total_outcomes = 21 / 192 := by sorry

end NUMINAMATH_CALUDE_end_on_multiple_of_four_probability_l2366_236642


namespace NUMINAMATH_CALUDE_bad_carrots_count_l2366_236603

theorem bad_carrots_count (olivia_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) 
  (h1 : olivia_carrots = 20)
  (h2 : mom_carrots = 14)
  (h3 : good_carrots = 19) :
  olivia_carrots + mom_carrots - good_carrots = 15 :=
by sorry

end NUMINAMATH_CALUDE_bad_carrots_count_l2366_236603


namespace NUMINAMATH_CALUDE_range_of_a_l2366_236690

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 4}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statement
theorem range_of_a (a : ℝ) : C a ⊆ (A ∩ B) → 0 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2366_236690


namespace NUMINAMATH_CALUDE_range_of_a_l2366_236636

/-- An odd function with period 3 -/
def OddPeriodic3 (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 3) = f x)

/-- The main theorem -/
theorem range_of_a (a : ℝ) :
  (∃ f : ℝ → ℝ, OddPeriodic3 f ∧ f 2 > 1 ∧ f 2014 = (2 * a - 3) / (a + 1)) →
  -1 < a ∧ a < 2/3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l2366_236636


namespace NUMINAMATH_CALUDE_complex_number_equality_l2366_236627

theorem complex_number_equality : ((-1 + Complex.I * Real.sqrt 3) ^ 5) / (1 + Complex.I * Real.sqrt 3) = -16 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2366_236627


namespace NUMINAMATH_CALUDE_g_value_at_3_l2366_236696

def g (x : ℝ) : ℝ := 2 * x^5 - 3 * x^4 + 4 * x^2 - 3 * x + 6

theorem g_value_at_3 (h : g (-3) = 2) : g 3 = -20 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_3_l2366_236696


namespace NUMINAMATH_CALUDE_greatest_base_eight_digit_sum_l2366_236653

/-- Represents a positive integer in base 8 --/
def BaseEightRepresentation (n : ℕ+) : List ℕ := sorry

/-- Calculates the sum of digits in a base 8 representation --/
def sumOfDigits (digits : List ℕ) : ℕ := sorry

/-- Theorem stating that the greatest possible sum of digits in base 8 for numbers less than 1728 is 23 --/
theorem greatest_base_eight_digit_sum :
  ∃ (n : ℕ+), n < 1728 ∧
  sumOfDigits (BaseEightRepresentation n) = 23 ∧
  ∀ (m : ℕ+), m < 1728 →
    sumOfDigits (BaseEightRepresentation m) ≤ 23 := by
  sorry

end NUMINAMATH_CALUDE_greatest_base_eight_digit_sum_l2366_236653


namespace NUMINAMATH_CALUDE_range_of_a_l2366_236616

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

-- Define the property that the solution set is not empty
def has_solution (a : ℝ) : Prop := ∃ x : ℝ, f a x < 0

-- Theorem statement
theorem range_of_a (a : ℝ) : has_solution a ↔ a < -4 ∨ a > 4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2366_236616


namespace NUMINAMATH_CALUDE_shekar_average_proof_l2366_236656

def shekar_average_marks (math science social_studies english biology : ℕ) : ℚ :=
  (math + science + social_studies + english + biology : ℚ) / 5

theorem shekar_average_proof :
  shekar_average_marks 76 65 82 67 75 = 73 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_proof_l2366_236656


namespace NUMINAMATH_CALUDE_trumpington_band_max_size_l2366_236651

theorem trumpington_band_max_size :
  ∃ m : ℕ,
    (∀ k : ℕ, 24 * k < 1000 → 24 * k ≤ 24 * m) ∧
    (24 * m < 1000) ∧
    (24 * m % 30 = 6) ∧
    (24 * m = 936) := by
  sorry

end NUMINAMATH_CALUDE_trumpington_band_max_size_l2366_236651


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2366_236606

theorem polynomial_factorization (x : ℝ) :
  let P : ℝ → ℝ := λ x => x^8 + x^4 + 1
  (P x = (x^4 + x^2 + 1) * (x^4 - x^2 + 1)) ∧
  (P x = (x^2 + x + 1) * (x^2 - x + 1) * (x^2 + Real.sqrt 3 * x + 1) * (x^2 - Real.sqrt 3 * x + 1)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2366_236606


namespace NUMINAMATH_CALUDE_correct_calculation_l2366_236650

theorem correct_calculation (x : ℝ) : (4 * x + 16 = 32) → (x / 4 + 16 = 17) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2366_236650


namespace NUMINAMATH_CALUDE_print_shop_charge_difference_l2366_236609

/-- The charge difference between two print shops for a given number of copies -/
def charge_difference (price_x price_y : ℚ) (num_copies : ℕ) : ℚ :=
  num_copies * (price_y - price_x)

/-- The price per copy at print shop X -/
def price_x : ℚ := 1.25

/-- The price per copy at print shop Y -/
def price_y : ℚ := 2.75

/-- The number of copies to be printed -/
def num_copies : ℕ := 60

theorem print_shop_charge_difference :
  charge_difference price_x price_y num_copies = 90 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_charge_difference_l2366_236609


namespace NUMINAMATH_CALUDE_max_a_value_l2366_236646

/-- The quadratic function f(x) = ax^2 - ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x + 1

/-- The maximum possible value of a for the quadratic function f(x) = ax^2 - ax + 1
    such that |f(x)| ≤ 1 for all x in [0, 1] is 8 -/
theorem max_a_value :
  ∃ (a_max : ℝ), a_max = 8 ∧
  (∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |f a x| ≤ 1) →
               a ≤ a_max) ∧
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |f a_max x| ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l2366_236646


namespace NUMINAMATH_CALUDE_safari_count_l2366_236602

theorem safari_count (total_heads : ℕ) (total_legs : ℕ) 
  (h1 : total_heads = 300) 
  (h2 : total_legs = 710) : ∃ (birds mammals tripeds : ℕ),
  birds + mammals + tripeds = total_heads ∧
  2 * birds + 4 * mammals + 3 * tripeds = total_legs ∧
  birds = 139 := by
  sorry

end NUMINAMATH_CALUDE_safari_count_l2366_236602


namespace NUMINAMATH_CALUDE_thirty_blocks_placeable_l2366_236680

/-- Represents a chessboard with two opposite corners removed -/
structure ModifiedChessboard :=
  (size : Nat)
  (cornersRemoved : Nat)

/-- Represents a rectangular block -/
structure Block :=
  (length : Nat)
  (width : Nat)

/-- Calculates the number of blocks that can be placed on the modified chessboard -/
def countPlaceableBlocks (board : ModifiedChessboard) (block : Block) : Nat :=
  sorry

/-- Theorem stating that 30 blocks can be placed on the modified 8x8 chessboard -/
theorem thirty_blocks_placeable :
  ∀ (board : ModifiedChessboard) (block : Block),
    board.size = 8 ∧ 
    board.cornersRemoved = 2 ∧ 
    block.length = 2 ∧ 
    block.width = 1 →
    countPlaceableBlocks board block = 30 :=
  sorry

end NUMINAMATH_CALUDE_thirty_blocks_placeable_l2366_236680


namespace NUMINAMATH_CALUDE_sum_geq_three_over_product_l2366_236663

theorem sum_geq_three_over_product {a b c : ℝ} 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : a + b + c > 1/a + 1/b + 1/c) : 
  a + b + c ≥ 3/(a*b*c) := by
  sorry

end NUMINAMATH_CALUDE_sum_geq_three_over_product_l2366_236663


namespace NUMINAMATH_CALUDE_alice_bob_meet_l2366_236623

/-- Represents the number of points on the circle -/
def numPoints : ℕ := 18

/-- Represents Alice's movement per turn (clockwise) -/
def aliceMove : ℕ := 7

/-- Represents Bob's movement per turn (counterclockwise) -/
def bobMove : ℕ := 13

/-- Calculates the effective clockwise movement of a player given their movement -/
def effectiveMove (move : ℕ) : ℕ :=
  move % numPoints

/-- Calculates the relative movement between Alice and Bob in one turn -/
def relativeMove : ℤ :=
  (effectiveMove aliceMove : ℤ) - (effectiveMove (numPoints - bobMove) : ℤ)

/-- The number of turns it takes for Alice and Bob to meet -/
def numTurns : ℕ := 9

theorem alice_bob_meet :
  (numTurns : ℤ) * relativeMove % (numPoints : ℤ) = 0 :=
sorry

end NUMINAMATH_CALUDE_alice_bob_meet_l2366_236623


namespace NUMINAMATH_CALUDE_eighteen_player_tournament_l2366_236640

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A round-robin tournament with 18 players has 153 games -/
theorem eighteen_player_tournament : num_games 18 = 153 := by
  sorry

end NUMINAMATH_CALUDE_eighteen_player_tournament_l2366_236640


namespace NUMINAMATH_CALUDE_z3_magnitude_range_l2366_236664

open Complex

theorem z3_magnitude_range (z₁ z₂ z₃ : ℂ) 
  (h1 : abs z₁ = Real.sqrt 2)
  (h2 : abs z₂ = Real.sqrt 2)
  (h3 : (z₁.re * z₂.re + z₁.im * z₂.im) = 0)
  (h4 : abs (z₁ + z₂ - z₃) = 2) :
  ∃ (r : ℝ), r ∈ Set.Icc 0 4 ∧ abs z₃ = r :=
by sorry

end NUMINAMATH_CALUDE_z3_magnitude_range_l2366_236664


namespace NUMINAMATH_CALUDE_three_times_more_plus_constant_problem_solution_l2366_236659

theorem three_times_more_plus_constant (base : ℝ) (more : ℕ) (constant : ℝ) :
  (base * (1 + more : ℝ) + constant = base * (more + 1 : ℝ) + constant) := by sorry

theorem problem_solution : 
  (608 : ℝ) * (1 + 3 : ℝ) + 12.8 = 2444.8 := by sorry

end NUMINAMATH_CALUDE_three_times_more_plus_constant_problem_solution_l2366_236659


namespace NUMINAMATH_CALUDE_quadratic_completing_square_sum_l2366_236662

theorem quadratic_completing_square_sum (x q t : ℝ) : 
  (9 * x^2 - 54 * x - 36 = 0) →
  ((x + q)^2 = t) →
  (9 * (x + q)^2 = 9 * t) →
  (q + t = 10) := by
sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_sum_l2366_236662


namespace NUMINAMATH_CALUDE_projection_problem_l2366_236619

def v (z : ℝ) : Fin 3 → ℝ := ![4, -1, z]
def u : Fin 3 → ℝ := ![6, -2, 3]

theorem projection_problem (z : ℝ) : 
  (v z • u) / (u • u) = 20 / 49 → z = -2 := by sorry

end NUMINAMATH_CALUDE_projection_problem_l2366_236619


namespace NUMINAMATH_CALUDE_largest_fraction_l2366_236601

theorem largest_fraction : 
  let f1 := 5 / 11
  let f2 := 6 / 13
  let f3 := 18 / 37
  let f4 := 101 / 202
  let f5 := 200 / 399
  f5 > f1 ∧ f5 > f2 ∧ f5 > f3 ∧ f5 > f4 := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l2366_236601


namespace NUMINAMATH_CALUDE_tens_digit_of_expression_l2366_236658

-- Define the expression
def expression : ℤ := 2027^2028 - 2029

-- Theorem statement
theorem tens_digit_of_expression :
  (expression / 10) % 10 = 1 :=
by sorry

end NUMINAMATH_CALUDE_tens_digit_of_expression_l2366_236658


namespace NUMINAMATH_CALUDE_fraction_of_three_fourths_that_is_one_fifth_l2366_236630

theorem fraction_of_three_fourths_that_is_one_fifth (x : ℚ) : x * (3/4 : ℚ) = (1/5 : ℚ) ↔ x = (4/15 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_three_fourths_that_is_one_fifth_l2366_236630


namespace NUMINAMATH_CALUDE_triangle_inequality_l2366_236670

theorem triangle_inequality (R r p : ℝ) (a b c m_a m_b m_c : ℝ) 
  (h1 : R * r = a * b * c / (4 * p))
  (h2 : a * b * c ≤ 8 * p^3)
  (h3 : p^2 ≤ (m_a^2 + m_b^2 + m_c^2) / 4)
  (h4 : m_a^2 + m_b^2 + m_c^2 ≤ 27 * R^2 / 4) :
  27 * R * r ≤ 2 * p^2 ∧ 2 * p^2 ≤ 27 * R^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2366_236670


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2366_236610

theorem trigonometric_identity : 
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) + 
  Real.cos (225 * π / 180) * Real.sin (165 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2366_236610


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2366_236655

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x - 3 < 0}

theorem intersection_of_A_and_B :
  ∀ x : ℝ, x ∈ A ∩ B ↔ -1 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2366_236655


namespace NUMINAMATH_CALUDE_equivalent_statements_l2366_236600

theorem equivalent_statements (P Q : Prop) : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_statements_l2366_236600


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_equals_two_l2366_236637

-- Define the curve and tangent line
def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 1
def tangent_line (x : ℝ) : ℝ := 2*x + 1

-- Theorem statement
theorem tangent_line_implies_a_equals_two (a : ℝ) :
  (∃ x₀ : ℝ, curve a x₀ = tangent_line x₀ ∧ 
    ∀ x : ℝ, x ≠ x₀ → curve a x ≠ tangent_line x) →
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_equals_two_l2366_236637


namespace NUMINAMATH_CALUDE_fraction_equality_l2366_236685

theorem fraction_equality (a b c d : ℚ) 
  (h1 : a / b = 20)
  (h2 : c / b = 5)
  (h3 : c / d = 1 / 15) :
  a / d = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2366_236685


namespace NUMINAMATH_CALUDE_highest_probability_l2366_236684

-- Define the sample space
variable (Ω : Type)

-- Define the probability measure
variable (P : Set Ω → ℝ)

-- Define events A, B, and C
variable (A B C : Set Ω)

-- State the theorem
theorem highest_probability :
  C ⊆ B → B ⊆ A → P A ≥ P B ∧ P A ≥ P C := by
  sorry

end NUMINAMATH_CALUDE_highest_probability_l2366_236684


namespace NUMINAMATH_CALUDE_tank_capacity_is_640_l2366_236605

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 640

/-- The time in hours it takes to empty the tank with only the outlet pipe open -/
def outlet_time : ℝ := 10

/-- The rate at which the inlet pipe adds water in liters per minute -/
def inlet_rate : ℝ := 4

/-- The time in hours it takes to empty the tank with both pipes open -/
def both_pipes_time : ℝ := 16

/-- Theorem stating that the tank capacity is 640 liters given the conditions -/
theorem tank_capacity_is_640 :
  tank_capacity = outlet_time * (inlet_rate * 60) * both_pipes_time / (both_pipes_time - outlet_time) :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_is_640_l2366_236605


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2366_236660

theorem smallest_n_square_and_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 3 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 5 * x = y^2) → (∃ (z : ℕ), 3 * x = z^3) → x ≥ n) ∧
  n = 45 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l2366_236660


namespace NUMINAMATH_CALUDE_quadratic_properties_l2366_236693

/-- Quadratic function f(x) = 2x^2 + 4x - 6 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 6

/-- Vertex form of f(x) -/
def vertex_form (x : ℝ) : ℝ := 2 * (x + 1)^2 - 8

/-- Axis of symmetry -/
def axis_of_symmetry : ℝ := -1

/-- Vertex coordinates -/
def vertex : ℝ × ℝ := (-1, -8)

theorem quadratic_properties :
  (∀ x, f x = vertex_form x) ∧
  (axis_of_symmetry = -1) ∧
  (vertex = (-1, -8)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2366_236693


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2366_236639

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) (h5 : a 5 = 15) :
  a 3 + a 4 + a 7 + a 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2366_236639


namespace NUMINAMATH_CALUDE_square_value_when_product_zero_l2366_236654

theorem square_value_when_product_zero (a : ℝ) :
  (a^2 - 3) * (a^2 + 1) = 0 → a^2 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_square_value_when_product_zero_l2366_236654


namespace NUMINAMATH_CALUDE_probability_two_heads_in_four_flips_l2366_236666

def coin_flip_probability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

theorem probability_two_heads_in_four_flips :
  coin_flip_probability 4 2 = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_heads_in_four_flips_l2366_236666


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l2366_236647

theorem sin_sum_of_complex_exponentials (θ φ : ℝ) :
  Complex.exp (θ * Complex.I) = 4/5 + 3/5 * Complex.I →
  Complex.exp (φ * Complex.I) = -5/13 + 12/13 * Complex.I →
  Real.sin (θ + φ) = 84/65 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l2366_236647


namespace NUMINAMATH_CALUDE_highest_score_is_174_l2366_236615

/-- Represents a batsman's cricket statistics -/
structure BatsmanStats where
  total_innings : ℕ
  total_runs : ℕ
  highest_score : ℕ
  lowest_score : ℕ

/-- Calculates the average score for a batsman -/
def average_score (stats : BatsmanStats) : ℚ :=
  stats.total_runs / stats.total_innings

/-- Calculates the average score excluding highest and lowest scores -/
def average_score_excluding_extremes (stats : BatsmanStats) : ℚ :=
  (stats.total_runs - stats.highest_score - stats.lowest_score) / (stats.total_innings - 2)

/-- Theorem: Given the conditions, the batsman's highest score is 174 runs -/
theorem highest_score_is_174 (stats : BatsmanStats) :
  stats.total_innings = 40 ∧
  average_score stats = 50 ∧
  stats.highest_score = stats.lowest_score + 172 ∧
  average_score_excluding_extremes stats = 48 →
  stats.highest_score = 174 := by
  sorry

#check highest_score_is_174

end NUMINAMATH_CALUDE_highest_score_is_174_l2366_236615


namespace NUMINAMATH_CALUDE_exists_a_b_counterexample_l2366_236645

theorem exists_a_b_counterexample : ∃ (a b : ℝ), a > b ∧ a^2 ≤ b^2 := by sorry

end NUMINAMATH_CALUDE_exists_a_b_counterexample_l2366_236645


namespace NUMINAMATH_CALUDE_roots_of_polynomial_l2366_236689

def p (x : ℝ) : ℝ := x^3 - 6*x^2 + 11*x - 6

theorem roots_of_polynomial :
  (∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) := by sorry

end NUMINAMATH_CALUDE_roots_of_polynomial_l2366_236689


namespace NUMINAMATH_CALUDE_max_tuesdays_in_80_days_l2366_236608

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Counts the number of Tuesdays in the first n days of a year -/
def countTuesdays (startDay : DayOfWeek) (n : ℕ) : ℕ :=
  sorry

/-- The maximum number of Tuesdays in the first 80 days of a year is 12 -/
theorem max_tuesdays_in_80_days :
  ∃ (startDay : DayOfWeek), countTuesdays startDay 80 = 12 ∧
  ∀ (d : DayOfWeek), countTuesdays d 80 ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_max_tuesdays_in_80_days_l2366_236608


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2366_236607

theorem arithmetic_calculation : 2 + 3 * 4 - 5 + 6 = 15 := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2366_236607


namespace NUMINAMATH_CALUDE_horner_method_polynomial_evaluation_l2366_236620

def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 6*x + 7

theorem horner_method_polynomial_evaluation :
  f 5 = 2677 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_polynomial_evaluation_l2366_236620


namespace NUMINAMATH_CALUDE_arctan_equation_equivalence_l2366_236681

theorem arctan_equation_equivalence (x : ℝ) :
  Real.arctan (1 / x) + Real.arctan (1 / x^5) = π / 6 →
  x^6 - Real.sqrt 3 * x^5 - Real.sqrt 3 * x - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_equivalence_l2366_236681


namespace NUMINAMATH_CALUDE_holly_401k_contribution_l2366_236644

/-- Calculates the total contribution to Holly's 401k after 1 year -/
def total_contribution (paychecks_per_year : ℕ) (contribution_per_paycheck : ℚ) (company_match_percentage : ℚ) : ℚ :=
  let employee_contribution := paychecks_per_year * contribution_per_paycheck
  let company_contribution := employee_contribution * company_match_percentage
  employee_contribution + company_contribution

/-- Theorem stating that Holly's total 401k contribution after 1 year is $2,756.00 -/
theorem holly_401k_contribution :
  total_contribution 26 100 (6 / 100) = 2756 :=
by sorry

end NUMINAMATH_CALUDE_holly_401k_contribution_l2366_236644


namespace NUMINAMATH_CALUDE_no_inscribed_circle_pentagon_l2366_236629

/-- A pentagon with side lengths a, b, c, d, e has an inscribed circle if and only if
    there exists a positive real number r such that
    2(a + b + c + d + e) = (a + b - c - d + e)(a - b + c - d + e)(-a + b + c - d + e)(-a - b + c + d + e)/r^2 -/
def has_inscribed_circle (a b c d e : ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ 2*(a + b + c + d + e) = (a + b - c - d + e)*(a - b + c - d + e)*(-a + b + c - d + e)*(-a - b + c + d + e)/(r^2)

/-- Theorem: There does not exist a pentagon with side lengths 3, 4, 9, 11, and 13 cm
    that has an inscribed circle -/
theorem no_inscribed_circle_pentagon : ¬ has_inscribed_circle 3 4 9 11 13 := by
  sorry

end NUMINAMATH_CALUDE_no_inscribed_circle_pentagon_l2366_236629


namespace NUMINAMATH_CALUDE_problem_statement_l2366_236688

theorem problem_statement (a x y : ℝ) (h1 : a ≠ x) (h2 : a ≠ y) (h3 : x ≠ y)
  (h4 : Real.sqrt (a * (x - a)) + Real.sqrt (a * (y - a)) = Real.sqrt (x - a) - Real.sqrt (a - y)) :
  (3 * x^2 + x * y - y^2) / (x^2 - x * y + y^2) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2366_236688


namespace NUMINAMATH_CALUDE_initial_red_marbles_l2366_236695

theorem initial_red_marbles (r g : ℕ) : 
  (r : ℚ) / g = 5 / 3 →
  ((r - 15) : ℚ) / (g + 18) = 1 / 2 →
  r = 34 :=
by sorry

end NUMINAMATH_CALUDE_initial_red_marbles_l2366_236695


namespace NUMINAMATH_CALUDE_equilateral_triangle_cd_l2366_236672

/-- An equilateral triangle with vertices at (0,0), (c,14), and (d,41) has cd = -2208 -/
theorem equilateral_triangle_cd (c d : ℝ) : 
  (Complex.abs (Complex.I * 14 - c - Complex.I * 14) = Complex.abs (Complex.I * 41 - c - Complex.I * 14)) ∧
  (Complex.abs (Complex.I * 41 - 0) = Complex.abs (c + Complex.I * 14 - 0)) ∧
  (Complex.abs (c + Complex.I * 14 - 0) = Complex.abs (Complex.I * 14 - 0)) →
  c * d = -2208 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_cd_l2366_236672


namespace NUMINAMATH_CALUDE_lisa_marbles_problem_l2366_236633

/-- The minimum number of additional marbles needed for Lisa to distribute to her friends -/
def minimum_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

theorem lisa_marbles_problem (num_friends : ℕ) (initial_marbles : ℕ) 
  (h1 : num_friends = 12) (h2 : initial_marbles = 50) :
  minimum_additional_marbles num_friends initial_marbles = 28 := by
  sorry

end NUMINAMATH_CALUDE_lisa_marbles_problem_l2366_236633


namespace NUMINAMATH_CALUDE_otimes_calculation_l2366_236674

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a^3 - b

-- Theorem statement
theorem otimes_calculation (a : ℝ) : otimes a (otimes a (otimes a a)) = a^3 - a := by
  sorry

end NUMINAMATH_CALUDE_otimes_calculation_l2366_236674


namespace NUMINAMATH_CALUDE_at_least_one_first_class_l2366_236631

theorem at_least_one_first_class (n m k : ℕ) (h1 : n = 20) (h2 : m = 16) (h3 : k = 3) :
  (Nat.choose m 1 * Nat.choose (n - m) 2) +
  (Nat.choose m 2 * Nat.choose (n - m) 1) +
  (Nat.choose m 3) = 1136 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_first_class_l2366_236631


namespace NUMINAMATH_CALUDE_min_diff_is_one_l2366_236673

-- Define the functions
def f (x : ℤ) : ℝ := 2 * (abs x)
def g (x : ℤ) : ℝ := -(x^2) - 4*x - 1

-- Define the difference function
def diff (x : ℤ) : ℝ := f x - g x

-- Theorem statement
theorem min_diff_is_one :
  ∃ (x : ℤ), diff x = 1 ∧ ∀ (y : ℤ), diff y ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_min_diff_is_one_l2366_236673


namespace NUMINAMATH_CALUDE_simultaneous_integer_fractions_l2366_236657

theorem simultaneous_integer_fractions (x : ℤ) :
  (∃ y z : ℤ, (x - 3) / 7 = y ∧ (x - 2) / 5 = z) ↔ 
  (∃ t : ℤ, x = 35 * t + 17) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_integer_fractions_l2366_236657


namespace NUMINAMATH_CALUDE_snow_probability_l2366_236641

theorem snow_probability (p : ℝ) (h : p = 2/3) :
  3 * p^2 * (1 - p) = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l2366_236641


namespace NUMINAMATH_CALUDE_total_birds_in_marsh_l2366_236667

def geese : ℕ := 58
def ducks : ℕ := 37
def herons : ℕ := 23
def kingfishers : ℕ := 46
def swans : ℕ := 15

theorem total_birds_in_marsh : geese + ducks + herons + kingfishers + swans = 179 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_in_marsh_l2366_236667


namespace NUMINAMATH_CALUDE_erased_number_proof_l2366_236612

theorem erased_number_proof (b : ℕ) (x : ℕ) : 
  3 ≤ b →
  (b - 2) * (b + 3) / 2 - x = 1015 * (b - 3) / 19 →
  x = 805 :=
sorry

end NUMINAMATH_CALUDE_erased_number_proof_l2366_236612


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l2366_236626

theorem sum_of_cubes_of_roots (r s t : ℝ) : 
  (r - (27 : ℝ)^(1/3 : ℝ)) * (r - (64 : ℝ)^(1/3 : ℝ)) * (r - (125 : ℝ)^(1/3 : ℝ)) = 1/2 →
  (s - (27 : ℝ)^(1/3 : ℝ)) * (s - (64 : ℝ)^(1/3 : ℝ)) * (s - (125 : ℝ)^(1/3 : ℝ)) = 1/2 →
  (t - (27 : ℝ)^(1/3 : ℝ)) * (t - (64 : ℝ)^(1/3 : ℝ)) * (t - (125 : ℝ)^(1/3 : ℝ)) = 1/2 →
  r ≠ s → r ≠ t → s ≠ t →
  r^3 + s^3 + t^3 = 214.5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l2366_236626


namespace NUMINAMATH_CALUDE_binary_110101_to_base7_l2366_236671

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its base-7 representation (as a list of digits). -/
def nat_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) :=
    if m = 0 then acc
    else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem binary_110101_to_base7 :
  nat_to_base7 (binary_to_nat [true, false, true, false, true, true]) = [1, 0, 4] := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_to_base7_l2366_236671


namespace NUMINAMATH_CALUDE_max_display_sum_l2366_236652

def is_valid_hour (h : ℕ) : Prop := h ≥ 0 ∧ h ≤ 23

def is_valid_minute (m : ℕ) : Prop := m ≥ 0 ∧ m ≤ 59

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def display_sum (h m : ℕ) : ℕ :=
  digit_sum h + digit_sum m

theorem max_display_sum :
  ∃ (h m : ℕ), is_valid_hour h ∧ is_valid_minute m ∧
  ∀ (h' m' : ℕ), is_valid_hour h' → is_valid_minute m' →
  display_sum h' m' ≤ display_sum h m ∧
  display_sum h m = 24 :=
sorry

end NUMINAMATH_CALUDE_max_display_sum_l2366_236652


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l2366_236649

theorem angle_sum_in_circle (y : ℚ) : 
  (6 * y + 3 * y + y + 4 * y = 360) → y = 180 / 7 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l2366_236649


namespace NUMINAMATH_CALUDE_dina_dolls_count_l2366_236611

/-- The number of dolls Ivy has -/
def ivy_dolls : ℕ := 30

/-- The number of collector's edition dolls Ivy has -/
def ivy_collector_dolls : ℕ := 20

/-- The ratio of collector's edition dolls to total dolls for Ivy -/
def collector_ratio : ℚ := 2/3

/-- The number of dolls Dina has -/
def dina_dolls : ℕ := 2 * ivy_dolls

theorem dina_dolls_count : dina_dolls = 60 := by
  sorry

end NUMINAMATH_CALUDE_dina_dolls_count_l2366_236611


namespace NUMINAMATH_CALUDE_admission_difference_l2366_236628

/-- Represents the admission plan for a university -/
structure AdmissionPlan where
  firstTier : ℕ
  secondTier : ℕ
  thirdTier : ℕ
  ratio_condition : firstTier * 5 = secondTier * 2 ∧ firstTier * 3 = thirdTier * 2

/-- Theorem stating the difference between second-tier and first-tier admissions -/
theorem admission_difference (plan : AdmissionPlan) (h : plan.thirdTier = 1500) :
  plan.secondTier - plan.firstTier = 1500 := by
  sorry

#check admission_difference

end NUMINAMATH_CALUDE_admission_difference_l2366_236628


namespace NUMINAMATH_CALUDE_fraction_subtraction_simplification_l2366_236694

theorem fraction_subtraction_simplification :
  8 / 21 - 10 / 63 = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_simplification_l2366_236694


namespace NUMINAMATH_CALUDE_b_four_lt_b_seven_l2366_236678

def b (α : ℕ → ℕ) : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1 + 1 / (b α n + 1 / α (n + 1))

theorem b_four_lt_b_seven (α : ℕ → ℕ) : b α 4 < b α 7 := by
  sorry

end NUMINAMATH_CALUDE_b_four_lt_b_seven_l2366_236678


namespace NUMINAMATH_CALUDE_expected_ones_is_half_l2366_236698

/-- The probability of rolling a 1 on a standard die -/
def prob_one : ℚ := 1/6

/-- The probability of not rolling a 1 on a standard die -/
def prob_not_one : ℚ := 1 - prob_one

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 1's when rolling three standard dice -/
def expected_ones : ℚ := 
  0 * (prob_not_one^num_dice) +
  1 * (num_dice.choose 1 * prob_one * prob_not_one^2) +
  2 * (num_dice.choose 2 * prob_one^2 * prob_not_one) +
  3 * prob_one^num_dice

theorem expected_ones_is_half : expected_ones = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expected_ones_is_half_l2366_236698


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_50_and_294_l2366_236624

theorem smallest_n_divisible_by_50_and_294 :
  ∃ (n : ℕ), n > 0 ∧ 50 ∣ n^2 ∧ 294 ∣ n^3 ∧
  ∀ (m : ℕ), m > 0 ∧ 50 ∣ m^2 ∧ 294 ∣ m^3 → n ≤ m :=
by
  use 210
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_50_and_294_l2366_236624


namespace NUMINAMATH_CALUDE_jessie_score_is_30_l2366_236613

-- Define the scoring system
def correct_points : ℚ := 2
def incorrect_points : ℚ := -0.5
def unanswered_points : ℚ := 0

-- Define Jessie's answers
def correct_answers : ℕ := 16
def incorrect_answers : ℕ := 4
def unanswered_questions : ℕ := 10

-- Define Jessie's score calculation
def jessie_score : ℚ :=
  (correct_answers : ℚ) * correct_points +
  (incorrect_answers : ℚ) * incorrect_points +
  (unanswered_questions : ℚ) * unanswered_points

-- Theorem to prove
theorem jessie_score_is_30 : jessie_score = 30 := by
  sorry

end NUMINAMATH_CALUDE_jessie_score_is_30_l2366_236613


namespace NUMINAMATH_CALUDE_quiz_sum_l2366_236677

theorem quiz_sum (x y : ℕ+) (h1 : x - y = 4) (h2 : x * y = 104) : x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_quiz_sum_l2366_236677


namespace NUMINAMATH_CALUDE_even_quadratic_function_range_l2366_236604

/-- A quadratic function that is even -/
def EvenQuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a c : ℝ, ∀ x : ℝ, f x = a * x^2 + c

theorem even_quadratic_function_range
  (f : ℝ → ℝ)
  (hf : EvenQuadraticFunction f)
  (h1 : 1 ≤ f 1 ∧ f 1 ≤ 2)
  (h2 : 3 ≤ f 2 ∧ f 2 ≤ 4) :
  14/3 ≤ f 3 ∧ f 3 ≤ 9 := by
sorry

end NUMINAMATH_CALUDE_even_quadratic_function_range_l2366_236604


namespace NUMINAMATH_CALUDE_mathematics_letter_probability_l2366_236697

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of unique letters in 'MATHEMATICS' -/
def unique_letters : ℕ := 8

/-- The probability of selecting a letter from the alphabet that appears in 'MATHEMATICS' -/
def probability : ℚ := unique_letters / alphabet_size

theorem mathematics_letter_probability : probability = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_mathematics_letter_probability_l2366_236697


namespace NUMINAMATH_CALUDE_prob_same_group_l2366_236621

/-- The number of interest groups -/
def num_groups : ℕ := 3

/-- The probability of a student joining any specific group -/
def prob_join_group : ℚ := 1 / num_groups

/-- The total number of possible outcomes for two students joining groups -/
def total_outcomes : ℕ := num_groups * num_groups

/-- The number of outcomes where both students join the same group -/
def same_group_outcomes : ℕ := num_groups

theorem prob_same_group :
  (same_group_outcomes : ℚ) / total_outcomes = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_prob_same_group_l2366_236621


namespace NUMINAMATH_CALUDE_distance_covered_l2366_236669

theorem distance_covered (walk_speed run_speed : ℝ) (total_time : ℝ) (h1 : walk_speed = 4)
    (h2 : run_speed = 8) (h3 : total_time = 0.75) : ℝ :=
  let total_distance := 8
  let half_distance := total_distance / 2
  let walk_time := half_distance / walk_speed
  let run_time := half_distance / run_speed
  have time_equation : walk_time + run_time = total_time := by sorry
  have distance_equation : total_distance = walk_speed * walk_time + run_speed * run_time := by sorry
  total_distance

#check distance_covered

end NUMINAMATH_CALUDE_distance_covered_l2366_236669


namespace NUMINAMATH_CALUDE_deepak_age_l2366_236635

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 5 / 7 →
  arun_age + 6 = 36 →
  deepak_age = 42 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l2366_236635


namespace NUMINAMATH_CALUDE_all_terms_are_integers_l2366_236668

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => (1 + (sequence_a (n + 1))^2) / (sequence_a n)

theorem all_terms_are_integers :
  ∀ n : ℕ, ∃ k : ℤ, sequence_a n = k :=
by sorry

end NUMINAMATH_CALUDE_all_terms_are_integers_l2366_236668


namespace NUMINAMATH_CALUDE_compare_sqrt_sums_l2366_236638

theorem compare_sqrt_sums (a : ℝ) (h : a > 0) :
  Real.sqrt a + Real.sqrt (a + 3) < Real.sqrt (a + 1) + Real.sqrt (a + 2) := by
sorry

end NUMINAMATH_CALUDE_compare_sqrt_sums_l2366_236638


namespace NUMINAMATH_CALUDE_flower_calculation_l2366_236618

/- Define the initial quantities -/
def initial_roses : ℕ := 36
def initial_chocolates : ℕ := 5
def initial_cupcakes : ℕ := 10
def initial_sunflowers : ℕ := 24

/- Define the trading events -/
def trade_day5 : ℕ × ℕ := (12, 6)  -- (roses, sunflowers)
def trade_day6 : ℕ × ℕ := (12, 20)  -- (roses, cupcakes)
def trade_day7 : ℕ := 15  -- daffodils

/- Define the wilting rates -/
def wilt_rate_day5 : ℚ := 1/10
def wilt_rate_day6_roses : ℚ := 1/5
def wilt_rate_day6_sunflowers : ℚ := 3/10
def wilt_rate_day7_roses : ℚ := 1/4
def wilt_rate_day7_sunflowers : ℚ := 3/20
def wilt_rate_day7_daffodils : ℚ := 1/5

/- Define the function to calculate the number of unwilted flowers -/
def calculate_unwilted_flowers (initial_roses initial_sunflowers : ℕ) 
  (trade_day5 trade_day6 : ℕ × ℕ) (trade_day7 : ℕ)
  (wilt_rate_day5 wilt_rate_day6_roses wilt_rate_day6_sunflowers 
   wilt_rate_day7_roses wilt_rate_day7_sunflowers wilt_rate_day7_daffodils : ℚ) :
  ℕ × ℕ × ℕ := sorry

/- Theorem statement -/
theorem flower_calculation :
  calculate_unwilted_flowers initial_roses initial_sunflowers
    trade_day5 trade_day6 trade_day7
    wilt_rate_day5 wilt_rate_day6_roses wilt_rate_day6_sunflowers
    wilt_rate_day7_roses wilt_rate_day7_sunflowers wilt_rate_day7_daffodils
  = (34, 18, 12) := by sorry

end NUMINAMATH_CALUDE_flower_calculation_l2366_236618


namespace NUMINAMATH_CALUDE_locus_equation_l2366_236683

-- Define the focus point F
def F : ℝ × ℝ := (2, 0)

-- Define the directrix line l: x + 3 = 0
def l (x : ℝ) : Prop := x + 3 = 0

-- Define the distance condition for point M
def distance_condition (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  let dist_to_F := Real.sqrt ((x - F.1)^2 + (y - F.2)^2)
  let dist_to_l := |x + 3|
  dist_to_F + 1 = dist_to_l

-- State the theorem
theorem locus_equation :
  ∀ M : ℝ × ℝ, distance_condition M ↔ M.2^2 = 8 * M.1 :=
sorry

end NUMINAMATH_CALUDE_locus_equation_l2366_236683


namespace NUMINAMATH_CALUDE_binary_representation_of_89_l2366_236699

def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem binary_representation_of_89 :
  decimal_to_binary 89 = [1, 0, 1, 1, 0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_89_l2366_236699
