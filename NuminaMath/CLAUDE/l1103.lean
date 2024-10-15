import Mathlib

namespace NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l1103_110306

theorem students_in_both_band_and_chorus 
  (total : ℕ) 
  (band : ℕ) 
  (chorus : ℕ) 
  (band_or_chorus : ℕ) 
  (h1 : total = 300)
  (h2 : band = 100)
  (h3 : chorus = 120)
  (h4 : band_or_chorus = 195) :
  band + chorus - band_or_chorus = 25 :=
by sorry

end NUMINAMATH_CALUDE_students_in_both_band_and_chorus_l1103_110306


namespace NUMINAMATH_CALUDE_floor_expression_equals_eight_l1103_110382

def n : ℕ := 1004

theorem floor_expression_equals_eight :
  ⌊(1005^3 : ℚ) / (1003 * 1004) - (1003^3 : ℚ) / (1004 * 1005)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_expression_equals_eight_l1103_110382


namespace NUMINAMATH_CALUDE_cosine_sum_simplification_l1103_110362

theorem cosine_sum_simplification :
  Real.cos (π / 15) + Real.cos (4 * π / 15) + Real.cos (14 * π / 15) = (Real.sqrt 21 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_sum_simplification_l1103_110362


namespace NUMINAMATH_CALUDE_b_share_correct_l1103_110325

/-- The share of the total payment for worker b -/
def b_share (a_days b_days c_days d_days total_payment : ℚ) : ℚ :=
  (1 / b_days) / ((1 / a_days) + (1 / b_days) + (1 / c_days) + (1 / d_days)) * total_payment

/-- Theorem stating that b's share is correct given the problem conditions -/
theorem b_share_correct :
  b_share 6 8 12 15 2400 = (1 / 8) / (53 / 120) * 2400 := by
  sorry

#eval b_share 6 8 12 15 2400

end NUMINAMATH_CALUDE_b_share_correct_l1103_110325


namespace NUMINAMATH_CALUDE_additional_apples_needed_l1103_110327

def apples_needed (pies : ℕ) (apples_per_pie : ℕ) (available_apples : ℕ) : ℕ :=
  pies * apples_per_pie - available_apples

theorem additional_apples_needed : 
  apples_needed 10 8 50 = 30 := by
  sorry

end NUMINAMATH_CALUDE_additional_apples_needed_l1103_110327


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1103_110318

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 15*x + 6 = 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 15*x + 6 = (x - r₁) * (x - r₂) ∧ 
  (1 / r₁ + 1 / r₂ = 5 / 2) := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l1103_110318


namespace NUMINAMATH_CALUDE_students_pets_difference_fourth_grade_classrooms_difference_l1103_110331

theorem students_pets_difference : ℕ → ℕ → ℕ → ℕ → ℕ
  | num_classrooms, students_per_class, rabbits_per_class, hamsters_per_class =>
    let total_students := num_classrooms * students_per_class
    let total_rabbits := num_classrooms * rabbits_per_class
    let total_hamsters := num_classrooms * hamsters_per_class
    let total_pets := total_rabbits + total_hamsters
    total_students - total_pets

theorem fourth_grade_classrooms_difference :
  students_pets_difference 5 20 2 1 = 85 := by
  sorry

end NUMINAMATH_CALUDE_students_pets_difference_fourth_grade_classrooms_difference_l1103_110331


namespace NUMINAMATH_CALUDE_game_ends_in_six_rounds_l1103_110319

/-- Represents a player in the token game -/
inductive Player : Type
| A
| B
| C

/-- Represents the state of the game at any given round -/
structure GameState :=
  (tokens : Player → ℕ)

/-- Determines if the game has ended (any player has 0 tokens) -/
def game_ended (state : GameState) : Prop :=
  ∃ p : Player, state.tokens p = 0

/-- Simulates one round of the game -/
def play_round (state : GameState) : GameState :=
  sorry

/-- The initial state of the game -/
def initial_state : GameState :=
  { tokens := λ p => match p with
    | Player.A => 16
    | Player.B => 14
    | Player.C => 12 }

/-- Theorem stating that the game ends after exactly 6 rounds -/
theorem game_ends_in_six_rounds :
  let final_state := (play_round^[6]) initial_state
  game_ended final_state ∧ ¬game_ended ((play_round^[5]) initial_state) :=
sorry

end NUMINAMATH_CALUDE_game_ends_in_six_rounds_l1103_110319


namespace NUMINAMATH_CALUDE_max_value_inequality_l1103_110366

theorem max_value_inequality (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (a_ge : a ≥ -1/2)
  (b_ge : b ≥ -3/2)
  (c_ge : c ≥ -2) :
  Real.sqrt (4*a + 2) + Real.sqrt (4*b + 6) + Real.sqrt (4*c + 8) ≤ 2 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1103_110366


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l1103_110395

theorem arithmetic_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum1 : a 1 + a 3 = 10) 
  (h_sum2 : a 4 + a 6 = 5/4) : 
  ∃ q : ℚ, q = 1/2 ∧ ∀ n : ℕ, a (n + 1) = a n + q := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_ratio_l1103_110395


namespace NUMINAMATH_CALUDE_min_value_theorem_l1103_110309

/-- A line that bisects a circle -/
structure BisectingLine where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : ∀ x y : ℝ, a * x + b * y - 2 = 0 → 
    (x - 3)^2 + (y - 2)^2 = 25 → (x - 3)^2 + (y - 2)^2 ≤ 25

/-- The theorem stating the minimum value of 3/a + 2/b -/
theorem min_value_theorem (l : BisectingLine) : 
  (∀ k : BisectingLine, 3 / l.a + 2 / l.b ≤ 3 / k.a + 2 / k.b) → 
  3 / l.a + 2 / l.b = 25 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1103_110309


namespace NUMINAMATH_CALUDE_store_earnings_theorem_l1103_110313

/-- Represents the earnings from selling bottled drinks in a country store. -/
def store_earnings (cola_price juice_price water_price : ℚ) 
                   (cola_sold juice_sold water_sold : ℕ) : ℚ :=
  cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold

/-- Theorem stating that the store earned $88 from selling bottled drinks. -/
theorem store_earnings_theorem : 
  store_earnings 3 1.5 1 15 12 25 = 88 := by
  sorry

end NUMINAMATH_CALUDE_store_earnings_theorem_l1103_110313


namespace NUMINAMATH_CALUDE_equal_triangle_areas_l1103_110369

-- Define the trapezoid ABCD
structure Trapezoid (A B C D : ℝ × ℝ) : Prop where
  parallel : (A.2 - D.2) / (A.1 - D.1) = (B.2 - C.2) / (B.1 - C.1)

-- Define a point inside a polygon
def PointInside (P : ℝ × ℝ) (polygon : List (ℝ × ℝ)) : Prop := sorry

-- Define parallel lines
def Parallel (P₁ Q₁ P₂ Q₂ : ℝ × ℝ) : Prop :=
  (P₁.2 - Q₁.2) / (P₁.1 - Q₁.1) = (P₂.2 - Q₂.2) / (P₂.1 - Q₂.1)

-- Define the area of a triangle
def TriangleArea (P Q R : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem equal_triangle_areas 
  (A B C D M N : ℝ × ℝ) 
  (trap : Trapezoid A B C D)
  (m_inside : PointInside M [A, B, C, D])
  (n_inside : PointInside N [B, M, C])
  (am_cn_parallel : Parallel A M C N)
  (bm_dn_parallel : Parallel B M D N) :
  TriangleArea A B N = TriangleArea C D M := by
  sorry

end NUMINAMATH_CALUDE_equal_triangle_areas_l1103_110369


namespace NUMINAMATH_CALUDE_average_of_shifted_data_l1103_110393

/-- Given four positive real numbers with a specific variance, prove that the average of these numbers plus 3 is 5 -/
theorem average_of_shifted_data (x₁ x₂ x₃ x₄ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0)
  (h_var : (x₁^2 + x₂^2 + x₃^2 + x₄^2 - 16) / 4 = (x₁^2 + x₂^2 + x₃^2 + x₄^2) / 4 - ((x₁ + x₂ + x₃ + x₄) / 4)^2) :
  ((x₁ + 3) + (x₂ + 3) + (x₃ + 3) + (x₄ + 3)) / 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_shifted_data_l1103_110393


namespace NUMINAMATH_CALUDE_units_digit_of_A_is_1_l1103_110389

-- Define the sequence of powers of 3
def powerOf3 : ℕ → ℕ
| 0 => 1
| n + 1 => 3 * powerOf3 n

-- Define A
def A : ℕ := 2 * (3 + 1) * (powerOf3 2 + 1) * (powerOf3 4 + 1) + 1

-- Theorem statement
theorem units_digit_of_A_is_1 : A % 10 = 1 := by
  sorry


end NUMINAMATH_CALUDE_units_digit_of_A_is_1_l1103_110389


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l1103_110397

theorem largest_integer_negative_quadratic :
  ∃ (n : ℤ), n^2 - 13*n + 40 < 0 ∧
  ∀ (m : ℤ), m^2 - 13*m + 40 < 0 → m ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l1103_110397


namespace NUMINAMATH_CALUDE_product_xyz_l1103_110301

theorem product_xyz (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xy : x * y = 27 * Real.rpow 3 (1/3))
  (h_xz : x * z = 45 * Real.rpow 3 (1/3))
  (h_yz : y * z = 18 * Real.rpow 3 (1/3))
  (h_x_2y : x = 2 * y) : 
  x * y * z = 108 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_l1103_110301


namespace NUMINAMATH_CALUDE_positive_expression_l1103_110355

theorem positive_expression (x y z : ℝ) 
  (hx : 0 < x ∧ x < 2) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 0 < z ∧ z < 1) : 
  0 < y + x^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l1103_110355


namespace NUMINAMATH_CALUDE_sum_of_same_sign_values_l1103_110352

theorem sum_of_same_sign_values (a b : ℝ) : 
  (abs a = 3) → (abs b = 1) → (a * b > 0) → (a + b = 4 ∨ a + b = -4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_same_sign_values_l1103_110352


namespace NUMINAMATH_CALUDE_largest_four_digit_congruence_l1103_110386

theorem largest_four_digit_congruence :
  ∃ (n : ℕ), 
    n ≤ 9999 ∧ 
    n ≥ 1000 ∧ 
    45 * n ≡ 180 [MOD 315] ∧
    ∀ (m : ℕ), m ≤ 9999 ∧ m ≥ 1000 ∧ 45 * m ≡ 180 [MOD 315] → m ≤ n ∧
    n = 9993 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruence_l1103_110386


namespace NUMINAMATH_CALUDE_book_reading_ratio_l1103_110330

theorem book_reading_ratio (total_pages : ℕ) (pages_day1 : ℕ) (pages_left : ℕ) :
  total_pages = 360 →
  pages_day1 = 50 →
  pages_left = 210 →
  ∃ (pages_day2 : ℕ),
    pages_day2 + pages_day1 + pages_left = total_pages ∧
    pages_day2 = 2 * pages_day1 :=
by sorry

end NUMINAMATH_CALUDE_book_reading_ratio_l1103_110330


namespace NUMINAMATH_CALUDE_impossible_tiling_after_replacement_l1103_110399

/-- Represents a tile type -/
inductive Tile
| TwoByTwo
| OneByFour

/-- Represents a tiling of a rectangular grid -/
def Tiling := List Tile

/-- Represents a rectangular grid -/
structure Grid :=
(rows : Nat)
(cols : Nat)

/-- Checks if a tiling is valid for a given grid -/
def isValidTiling (g : Grid) (t : Tiling) : Prop :=
  -- Definition omitted
  sorry

/-- Checks if a grid can be tiled with 2x2 and 1x4 tiles -/
def canBeTiled (g : Grid) : Prop :=
  ∃ t : Tiling, isValidTiling g t

/-- Represents the operation of replacing one 2x2 tile with a 1x4 tile -/
def replaceTile (t : Tiling) : Tiling :=
  -- Definition omitted
  sorry

/-- Main theorem: If a grid can be tiled, replacing one 2x2 tile with a 1x4 tile makes it impossible to tile -/
theorem impossible_tiling_after_replacement (g : Grid) :
  canBeTiled g → ¬(∃ t : Tiling, isValidTiling g (replaceTile t)) :=
by
  sorry

end NUMINAMATH_CALUDE_impossible_tiling_after_replacement_l1103_110399


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1103_110398

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  ArithmeticSequence a → ArithmeticSequence b →
  (a 1 + b 1 = 7) → (a 3 + b 3 = 21) →
  (a 5 + b 5 = 35) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1103_110398


namespace NUMINAMATH_CALUDE_rajan_investment_is_20000_l1103_110312

/-- Represents the investment scenario with Rajan, Rakesh, and Mukesh --/
structure InvestmentScenario where
  rajan_investment : ℕ
  rakesh_investment : ℕ
  mukesh_investment : ℕ
  total_profit : ℕ
  rajan_profit : ℕ

/-- The investment scenario satisfies the given conditions --/
def satisfies_conditions (scenario : InvestmentScenario) : Prop :=
  scenario.rakesh_investment = 25000 ∧
  scenario.mukesh_investment = 15000 ∧
  scenario.total_profit = 4600 ∧
  scenario.rajan_profit = 2400 ∧
  (scenario.rajan_investment * 12 : ℚ) / 
    (scenario.rajan_investment * 12 + scenario.rakesh_investment * 4 + scenario.mukesh_investment * 8) = 
    (scenario.rajan_profit : ℚ) / scenario.total_profit

/-- Theorem stating that if the scenario satisfies the conditions, Rajan's investment is 20000 --/
theorem rajan_investment_is_20000 (scenario : InvestmentScenario) :
  satisfies_conditions scenario → scenario.rajan_investment = 20000 := by
  sorry

#check rajan_investment_is_20000

end NUMINAMATH_CALUDE_rajan_investment_is_20000_l1103_110312


namespace NUMINAMATH_CALUDE_pairwise_sum_product_inequality_l1103_110376

theorem pairwise_sum_product_inequality 
  (x : Fin 64 → ℝ) 
  (h_pos : ∀ i, x i > 0) 
  (h_strict_mono : StrictMono x) : 
  (x 63 * x 64) / (x 0 * x 1) > (x 63 + x 64) / (x 0 + x 1) := by
  sorry

end NUMINAMATH_CALUDE_pairwise_sum_product_inequality_l1103_110376


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1103_110396

/-- A polynomial with integer coefficients -/
def IntPolynomial (n : ℕ) := Fin n → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial n) : ℕ := n

/-- Irreducibility of a polynomial -/
def irreducible (p : IntPolynomial n) : Prop := sorry

/-- The modulus of a complex number is not greater than 1 -/
def modulusNotGreaterThanOne (z : ℂ) : Prop := Complex.abs z ≤ 1

/-- The roots of a polynomial -/
def roots (p : IntPolynomial n) : Set ℂ := sorry

/-- The statement of the theorem -/
theorem polynomial_factorization 
  (n : ℕ+) 
  (f : IntPolynomial n.val) 
  (h_irred : irreducible f) 
  (h_an : f (Fin.last n.val) ≠ 0)
  (h_roots : ∀ z ∈ roots f, modulusNotGreaterThanOne z) :
  ∃ (m : ℕ+) (g : IntPolynomial m.val), 
    ∃ (h : IntPolynomial (n.val + m.val)), 
      h = sorry ∧ 
      (∀ i, h i = if i.val < n.val then f i else if i.val < n.val + m.val then g (i - n.val) else 0) ∧
      h = λ i => if i.val = n.val + m.val - 1 then 1 else if i.val = n.val + m.val then -1 else 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1103_110396


namespace NUMINAMATH_CALUDE_decimal_difference_l1103_110365

-- Define the repeating decimal 0.727272...
def repeating_decimal : ℚ := 72 / 99

-- Define the terminating decimal 0.72
def terminating_decimal : ℚ := 72 / 100

-- Theorem statement
theorem decimal_difference :
  repeating_decimal - terminating_decimal = 2 / 275 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l1103_110365


namespace NUMINAMATH_CALUDE_unique_n_congruence_l1103_110385

theorem unique_n_congruence : ∃! n : ℤ, 3 ≤ n ∧ n ≤ 9 ∧ n ≡ 12473 [ZMOD 7] ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_congruence_l1103_110385


namespace NUMINAMATH_CALUDE_average_of_numbers_is_ten_l1103_110375

def numbers : List ℝ := [6, 8, 9, 11, 16]

theorem average_of_numbers_is_ten :
  (List.sum numbers) / (List.length numbers) = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_is_ten_l1103_110375


namespace NUMINAMATH_CALUDE_range_of_a_satisfying_condition_l1103_110305

/-- The universal set U is the set of real numbers. -/
def U : Set ℝ := Set.univ

/-- Set A is defined as {x | (x - 2)(x - 9) < 0}. -/
def A : Set ℝ := {x | (x - 2) * (x - 9) < 0}

/-- Set B is defined as {x | -2 - x ≤ 0 ≤ 5 - x}. -/
def B : Set ℝ := {x | -2 - x ≤ 0 ∧ 0 ≤ 5 - x}

/-- Set C is defined as {x | a ≤ x ≤ 2 - a}, where a is a real number. -/
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 2 - a}

/-- The theorem states that given the conditions, the range of values for a that satisfies C ∪ (∁ₘB) = R is (-∞, -3]. -/
theorem range_of_a_satisfying_condition :
  ∀ a : ℝ, (C a ∪ (U \ B) = U) ↔ a ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_satisfying_condition_l1103_110305


namespace NUMINAMATH_CALUDE_tenth_largest_number_l1103_110363

/-- Given a list of digits, generate all possible three-digit numbers -/
def generateThreeDigitNumbers (digits : List Nat) : List Nat :=
  sorry

/-- Sort a list of numbers in descending order -/
def sortDescending (numbers : List Nat) : List Nat :=
  sorry

theorem tenth_largest_number : 
  let digits : List Nat := [5, 3, 1, 9]
  let threeDigitNumbers := generateThreeDigitNumbers digits
  let sortedNumbers := sortDescending threeDigitNumbers
  List.get! sortedNumbers 9 = 531 := by
  sorry

end NUMINAMATH_CALUDE_tenth_largest_number_l1103_110363


namespace NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l1103_110339

theorem order_of_logarithmic_expressions (x : ℝ) 
  (h1 : x ∈ Set.Ioo (Real.exp (-1)) 1)
  (a b c : ℝ) 
  (ha : a = Real.log x)
  (hb : b = 2 * Real.log x)
  (hc : c = (Real.log x) ^ 3) : 
  b < a ∧ a < c := by
sorry

end NUMINAMATH_CALUDE_order_of_logarithmic_expressions_l1103_110339


namespace NUMINAMATH_CALUDE_inequality_proof_l1103_110367

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x / (y + z)) * (b + c) + (y / (z + x)) * (c + a) + (z / (x + y)) * (a + b) ≥ 
  Real.sqrt (3 * (a * b + b * c + c * a)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1103_110367


namespace NUMINAMATH_CALUDE_prob_red_then_white_l1103_110308

/-- The probability of drawing a red marble first and a white marble second without replacement
    from a bag containing 3 red marbles and 5 white marbles is 15/56. -/
theorem prob_red_then_white (red : ℕ) (white : ℕ) (total : ℕ) (h1 : red = 3) (h2 : white = 5) 
  (h3 : total = red + white) :
  (red / total) * (white / (total - 1)) = 15 / 56 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_white_l1103_110308


namespace NUMINAMATH_CALUDE_weight_replacement_l1103_110322

theorem weight_replacement (n : ℕ) (avg_increase w_new : ℝ) :
  n = 7 →
  avg_increase = 6.2 →
  w_new = 119.4 →
  ∃ w_old : ℝ, w_old = w_new - n * avg_increase ∧ w_old = 76 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l1103_110322


namespace NUMINAMATH_CALUDE_brian_stones_l1103_110324

theorem brian_stones (total : ℕ) (white black grey green : ℕ) : 
  total = 100 → 
  white + black = total → 
  grey = 40 → 
  green = 60 → 
  white * green = black * grey → 
  white > black → 
  white = 60 := by sorry

end NUMINAMATH_CALUDE_brian_stones_l1103_110324


namespace NUMINAMATH_CALUDE_alexanders_pictures_l1103_110311

theorem alexanders_pictures (total_pencils : ℕ) 
  (new_galleries : ℕ) (pictures_per_new_gallery : ℕ) 
  (pencils_per_picture : ℕ) (pencils_per_exhibition : ℕ) : 
  total_pencils = 88 →
  new_galleries = 5 →
  pictures_per_new_gallery = 2 →
  pencils_per_picture = 4 →
  pencils_per_exhibition = 2 →
  (total_pencils - 
    (new_galleries * pictures_per_new_gallery * pencils_per_picture) - 
    ((new_galleries + 1) * pencils_per_exhibition)) / pencils_per_picture = 9 :=
by sorry

end NUMINAMATH_CALUDE_alexanders_pictures_l1103_110311


namespace NUMINAMATH_CALUDE_range_of_f_l1103_110356

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.sin (ω * x)
noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (2 * x + φ) + 1

theorem range_of_f (ω : ℝ) (h_ω : ω > 0) (φ : ℝ) 
  (h_symmetry : ∀ x : ℝ, ∃ c : ℝ, f ω (c - x) = f ω (c + x) ∧ g φ (c - x) = g φ (c + x)) :
  Set.range (f ω) = Set.Icc (-3) 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l1103_110356


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_multiple_l1103_110387

/-- An arithmetic sequence with common difference d -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) - a n = d

/-- The constant multiple of a sequence -/
def ConstantMultiple (a : ℕ → ℝ) (c : ℝ) : ℕ → ℝ :=
  fun n => c * a n

theorem arithmetic_sequence_constant_multiple
  (a : ℕ → ℝ) (d c : ℝ) (hc : c ≠ 0) (ha : ArithmeticSequence a d) :
  ArithmeticSequence (ConstantMultiple a c) (c * d) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_multiple_l1103_110387


namespace NUMINAMATH_CALUDE_cone_shape_l1103_110302

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Defines the set of points satisfying φ ≤ c -/
def ConeSet (c : ℝ) : Set SphericalPoint :=
  {p : SphericalPoint | p.φ ≤ c}

/-- Theorem: The set of points satisfying φ ≤ c forms a cone -/
theorem cone_shape (c : ℝ) (h : 0 ≤ c ∧ c ≤ π) :
  ∃ (cone : Set SphericalPoint), ConeSet c = cone :=
sorry

end NUMINAMATH_CALUDE_cone_shape_l1103_110302


namespace NUMINAMATH_CALUDE_complex_sum_equality_l1103_110333

theorem complex_sum_equality : 
  let z₁ : ℂ := -1/2 + 3/4 * I
  let z₂ : ℂ := 7/3 - 5/6 * I
  z₁ + z₂ = 11/6 - 1/12 * I := by
sorry

end NUMINAMATH_CALUDE_complex_sum_equality_l1103_110333


namespace NUMINAMATH_CALUDE_factorization_correctness_l1103_110315

theorem factorization_correctness (x : ℝ) : 3 * x^2 - 2*x - 1 = (3*x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_correctness_l1103_110315


namespace NUMINAMATH_CALUDE_gcd_7163_209_l1103_110372

theorem gcd_7163_209 :
  let a := 7163
  let b := 209
  let c := 57
  let d := 38
  let e := 19
  a = b * 34 + c →
  b = c * 3 + d →
  c = d * 1 + e →
  d = e * 2 →
  Nat.gcd a b = e :=
by sorry

end NUMINAMATH_CALUDE_gcd_7163_209_l1103_110372


namespace NUMINAMATH_CALUDE_park_hikers_l1103_110357

theorem park_hikers (total : ℕ) (difference : ℕ) (hikers : ℕ) (bikers : ℕ) : 
  total = 676 → 
  difference = 178 → 
  total = hikers + bikers → 
  hikers = bikers + difference → 
  hikers = 427 := by
sorry

end NUMINAMATH_CALUDE_park_hikers_l1103_110357


namespace NUMINAMATH_CALUDE_least_boxes_for_candy_packing_l1103_110321

/-- Given that N is a non-zero perfect cube and 45 is a factor of N,
    prove that the least number of boxes needed to pack N pieces of candy,
    with 45 pieces per box, is 75. -/
theorem least_boxes_for_candy_packing (N : ℕ) : 
  N ≠ 0 ∧ 
  (∃ k : ℕ, N = k^3) ∧ 
  (∃ m : ℕ, N = 45 * m) ∧
  (∀ M : ℕ, M ≠ 0 ∧ (∃ j : ℕ, M = j^3) ∧ (∃ n : ℕ, M = 45 * n) → N ≤ M) →
  N / 45 = 75 := by
sorry

end NUMINAMATH_CALUDE_least_boxes_for_candy_packing_l1103_110321


namespace NUMINAMATH_CALUDE_m_range_l1103_110373

-- Define the propositions
def p (m : ℝ) : Prop := ∀ x, |x| + |x - 1| > m
def q (m : ℝ) : Prop := ∀ x y, x < y → (-(5 - 2*m)^x) > (-(5 - 2*m)^y)

-- Define the theorem
theorem m_range :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Set.Icc 1 2 ∧ m ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1103_110373


namespace NUMINAMATH_CALUDE_round_table_seating_arrangements_l1103_110360

def num_people : ℕ := 6
def num_specific_people : ℕ := 2

theorem round_table_seating_arrangements :
  let num_units := num_people - num_specific_people + 1
  (num_specific_people.factorial) * ((num_units - 1).factorial) = 48 := by
  sorry

end NUMINAMATH_CALUDE_round_table_seating_arrangements_l1103_110360


namespace NUMINAMATH_CALUDE_right_triangle_height_l1103_110336

theorem right_triangle_height (base height hypotenuse : ℝ) : 
  base = 4 →
  base + height + hypotenuse = 12 →
  base^2 + height^2 = hypotenuse^2 →
  height = 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_height_l1103_110336


namespace NUMINAMATH_CALUDE_helly_theorem_2d_l1103_110349

-- Define a type for points in the plane
variable (Point : Type)

-- Define a type for convex sets in the plane
variable (ConvexSet : Type)

-- Define a function to check if a point is in a convex set
variable (isIn : Point → ConvexSet → Prop)

-- Define a function to check if a set is convex
variable (isConvex : ConvexSet → Prop)

-- Define the theorem
theorem helly_theorem_2d 
  (n : ℕ) 
  (h_n : n ≥ 4) 
  (A : Fin n → ConvexSet) 
  (h_convex : ∀ i, isConvex (A i)) 
  (h_intersection : ∀ i j k, ∃ p, isIn p (A i) ∧ isIn p (A j) ∧ isIn p (A k)) :
  ∃ p, ∀ i, isIn p (A i) :=
sorry

end NUMINAMATH_CALUDE_helly_theorem_2d_l1103_110349


namespace NUMINAMATH_CALUDE_max_greece_value_l1103_110391

/-- Represents a mapping from letters to digits -/
def LetterMap := Char → Nat

/-- Check if a LetterMap is valid according to the problem conditions -/
def isValidMapping (m : LetterMap) : Prop :=
  (∀ c₁ c₂, c₁ ≠ c₂ → m c₁ ≠ m c₂) ∧ 
  (∀ c, m c ≤ 9) ∧
  m 'G' ≠ 0 ∧ m 'E' ≠ 0 ∧ m 'V' ≠ 0 ∧ m 'I' ≠ 0

/-- Convert a string of letters to a number using the given mapping -/
def stringToNumber (m : LetterMap) (s : String) : Nat :=
  s.foldl (fun acc c => acc * 10 + m c) 0

/-- Check if the equation holds for a given mapping -/
def equationHolds (m : LetterMap) : Prop :=
  (stringToNumber m "VER" - stringToNumber m "IA") = 
  (m 'G')^((m 'R')^(m 'E')) * (stringToNumber m "GRE" + stringToNumber m "ECE")

/-- The main theorem to be proved -/
theorem max_greece_value (m : LetterMap) :
  isValidMapping m →
  equationHolds m →
  (∀ m', isValidMapping m' → equationHolds m' → 
    stringToNumber m' "GREECE" ≤ stringToNumber m "GREECE") →
  stringToNumber m "GREECE" = 196646 := by
  sorry

end NUMINAMATH_CALUDE_max_greece_value_l1103_110391


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1103_110343

theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 26)
  (h3 : r - p = 32) :
  (p + q) / 2 = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1103_110343


namespace NUMINAMATH_CALUDE_smallest_2016_div_2017_correct_l1103_110341

/-- The smallest natural number that starts with 2016 and is divisible by 2017 -/
def smallest_2016_div_2017 : ℕ := 20162001

/-- A number starts with 2016 if it's greater than or equal to 2016 * 10^4 and less than 2017 * 10^4 -/
def starts_with_2016 (n : ℕ) : Prop :=
  2016 * 10^4 ≤ n ∧ n < 2017 * 10^4

theorem smallest_2016_div_2017_correct :
  starts_with_2016 smallest_2016_div_2017 ∧
  smallest_2016_div_2017 % 2017 = 0 ∧
  ∀ n : ℕ, n < smallest_2016_div_2017 →
    ¬(starts_with_2016 n ∧ n % 2017 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_2016_div_2017_correct_l1103_110341


namespace NUMINAMATH_CALUDE_fifteen_team_league_games_l1103_110377

/-- The number of games played in a league where each team plays every other team once -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league with 15 teams, where each team plays every other team once,
    the total number of games played is 105 -/
theorem fifteen_team_league_games :
  games_played 15 = 105 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_team_league_games_l1103_110377


namespace NUMINAMATH_CALUDE_M_mod_49_l1103_110370

/-- M is the 92-digit number formed by concatenating integers from 1 to 50 -/
def M : ℕ := sorry

/-- The sum of digits from 1 to 50 -/
def sum_digits : ℕ := (50 * (1 + 50)) / 2

theorem M_mod_49 : M % 49 = 18 := by sorry

end NUMINAMATH_CALUDE_M_mod_49_l1103_110370


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l1103_110353

theorem quadratic_equation_result (y : ℝ) (h : 6 * y^2 + 7 = 2 * y + 10) : 
  (12 * y - 4)^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l1103_110353


namespace NUMINAMATH_CALUDE_number_puzzle_l1103_110323

theorem number_puzzle (x : ℝ) : (1/2 : ℝ) * x + (1/3 : ℝ) * x = (1/4 : ℝ) * x + 7 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1103_110323


namespace NUMINAMATH_CALUDE_average_of_last_three_l1103_110388

theorem average_of_last_three (numbers : Fin 6 → ℝ) 
  (h1 : (numbers 0 + numbers 1 + numbers 2 + numbers 3 + numbers 4 + numbers 5) / 6 = 30)
  (h2 : (numbers 0 + numbers 1 + numbers 2 + numbers 3) / 4 = 25)
  (h3 : numbers 3 = 25) :
  (numbers 3 + numbers 4 + numbers 5) / 3 = 35 := by
sorry

end NUMINAMATH_CALUDE_average_of_last_three_l1103_110388


namespace NUMINAMATH_CALUDE_debbie_number_l1103_110334

def alice_skips (n : ℕ) : Bool :=
  n % 4 = 3

def barbara_says (n : ℕ) : Bool :=
  alice_skips n ∧ ¬(n % 12 = 7)

def candice_says (n : ℕ) : Bool :=
  alice_skips n ∧ barbara_says n ∧ ¬(n % 24 = 11)

def debbie_says (n : ℕ) : Bool :=
  1 ≤ n ∧ n ≤ 1200 ∧ ¬(alice_skips n) ∧ ¬(barbara_says n) ∧ ¬(candice_says n)

theorem debbie_number : ∃! n : ℕ, debbie_says n ∧ n = 1187 := by
  sorry

end NUMINAMATH_CALUDE_debbie_number_l1103_110334


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l1103_110368

theorem gcd_power_two_minus_one (a b : ℕ+) :
  Nat.gcd (2^a.val - 1) (2^b.val - 1) = 2^(Nat.gcd a.val b.val) - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l1103_110368


namespace NUMINAMATH_CALUDE_monotonic_quadratic_range_l1103_110329

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem monotonic_quadratic_range (a : ℝ) :
  (∀ x ∈ Set.Icc 2 3, Monotone (fun x => f a x)) →
  a ∈ Set.Iic 2 ∪ Set.Ici 3 :=
by sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_range_l1103_110329


namespace NUMINAMATH_CALUDE_edwards_lawn_mowing_earnings_l1103_110335

/-- Edward's lawn mowing business earnings and expenses --/
theorem edwards_lawn_mowing_earnings 
  (spring_earnings : ℕ) 
  (summer_earnings : ℕ) 
  (supplies_cost : ℕ) 
  (h1 : spring_earnings = 2)
  (h2 : summer_earnings = 27)
  (h3 : supplies_cost = 5) :
  spring_earnings + summer_earnings - supplies_cost = 24 :=
by sorry

end NUMINAMATH_CALUDE_edwards_lawn_mowing_earnings_l1103_110335


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l1103_110328

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ (k : ℕ), 2^k ∣ (10^1000 - 4^500) ∧ 
  ∀ (m : ℕ), 2^m ∣ (10^1000 - 4^500) → m ≤ k := by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l1103_110328


namespace NUMINAMATH_CALUDE_min_sum_a_b_l1103_110379

theorem min_sum_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + a*x + 2*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 2*b*x + a = 0) :
  ∀ c d : ℝ, c > 0 → d > 0 →
  (∃ x : ℝ, x^2 + c*x + 2*d = 0) →
  (∃ x : ℝ, x^2 + 2*d*x + c = 0) →
  c + d ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_a_b_l1103_110379


namespace NUMINAMATH_CALUDE_total_cost_theorem_l1103_110300

/-- Calculates the total cost of purchasing two laptops with accessories --/
def total_cost (first_laptop_price : ℝ) (second_laptop_multiplier : ℝ) 
  (second_laptop_discount : ℝ) (hard_drive_price : ℝ) (mouse_price : ℝ) 
  (software_subscription_price : ℝ) (insurance_rate : ℝ) : ℝ :=
  let first_laptop_total := first_laptop_price + hard_drive_price + mouse_price + 
    software_subscription_price + (insurance_rate * first_laptop_price)
  let second_laptop_price := first_laptop_price * second_laptop_multiplier
  let second_laptop_discounted := second_laptop_price * (1 - second_laptop_discount)
  let second_laptop_total := second_laptop_discounted + hard_drive_price + mouse_price + 
    (2 * software_subscription_price) + (insurance_rate * second_laptop_discounted)
  first_laptop_total + second_laptop_total

/-- Theorem stating the total cost of purchasing both laptops with accessories --/
theorem total_cost_theorem : 
  total_cost 500 3 0.15 80 20 120 0.1 = 2512.5 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l1103_110300


namespace NUMINAMATH_CALUDE_prism_volume_l1103_110317

theorem prism_volume (x y z : ℝ) 
  (h1 : x * y = 24) 
  (h2 : y * z = 8) 
  (h3 : x * z = 3) : 
  x * y * z = 24 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1103_110317


namespace NUMINAMATH_CALUDE_problem_solution_l1103_110350

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - x
def g (x : ℝ) : ℝ := 2*x - 3

-- Define the interval [0, 2]
def interval : Set ℝ := Set.Icc 0 2

theorem problem_solution :
  -- 1. Tangent line equation
  (∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 2*x - y - 2 = 0) ∧
  (∀ x, x ≠ 1 → (f x - f 1) / (x - 1) < 2) ∧
  (∀ x, x ≠ 1 → (f x - f 1) / (x - 1) > 2) ∧
  
  -- 2. Maximum value on the interval
  (∀ x ∈ interval, f x ≤ 6) ∧
  (∃ x ∈ interval, f x = 6) ∧
  
  -- 3. Existence of unique x₀
  (∃! x₀, f x₀ = g x₀) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1103_110350


namespace NUMINAMATH_CALUDE_complement_P_intersect_Q_range_of_a_when_P_subset_Q_l1103_110392

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}
def Q : Set ℝ := {x | x^2 - 3*x ≤ 10}

-- Statement for the first part of the problem
theorem complement_P_intersect_Q :
  (Set.univ \ P 3) ∩ Q = Set.Icc (-2) 4 := by sorry

-- Statement for the second part of the problem
theorem range_of_a_when_P_subset_Q :
  {a : ℝ | P a ∩ Q = P a} = Set.Iic 2 := by sorry

end NUMINAMATH_CALUDE_complement_P_intersect_Q_range_of_a_when_P_subset_Q_l1103_110392


namespace NUMINAMATH_CALUDE_leftSideSeats_l1103_110358

/-- Represents the seating arrangement in a bus -/
structure BusSeats where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeat : ℕ
  peoplePerSeat : ℕ
  totalCapacity : ℕ

/-- The bus seating arrangement satisfies the given conditions -/
def validBusSeats (bus : BusSeats) : Prop :=
  bus.rightSeats = bus.leftSeats - 3 ∧
  bus.peoplePerSeat = 3 ∧
  bus.backSeat = 9 ∧
  bus.totalCapacity = 90

/-- The theorem stating that the number of seats on the left side is 15 -/
theorem leftSideSeats (bus : BusSeats) (h : validBusSeats bus) : 
  bus.leftSeats = 15 := by
  sorry

#check leftSideSeats

end NUMINAMATH_CALUDE_leftSideSeats_l1103_110358


namespace NUMINAMATH_CALUDE_people_on_boats_l1103_110348

theorem people_on_boats (num_boats : ℕ) (people_per_boat : ℕ) :
  num_boats = 5 → people_per_boat = 3 → num_boats * people_per_boat = 15 := by
  sorry

end NUMINAMATH_CALUDE_people_on_boats_l1103_110348


namespace NUMINAMATH_CALUDE_imaginary_unit_cubed_l1103_110344

theorem imaginary_unit_cubed (i : ℂ) (h : i^2 = -1) : i^3 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_cubed_l1103_110344


namespace NUMINAMATH_CALUDE_cubic_less_than_square_l1103_110320

theorem cubic_less_than_square (x : ℚ) : 
  (x = 3/4 → x^3 < x^2) ∧ 
  (x = 5/3 → x^3 ≥ x^2) ∧ 
  (x = 1 → x^3 ≥ x^2) ∧ 
  (x = 3/2 → x^3 ≥ x^2) ∧ 
  (x = 21/20 → x^3 ≥ x^2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_less_than_square_l1103_110320


namespace NUMINAMATH_CALUDE_water_sip_calculation_l1103_110326

/-- Proves that given a 2-liter bottle of water consumed in 250 minutes with sips taken every 5 minutes, each sip is 40 ml. -/
theorem water_sip_calculation (bottle_volume : ℕ) (total_time : ℕ) (sip_interval : ℕ) :
  bottle_volume = 2000 →
  total_time = 250 →
  sip_interval = 5 →
  (bottle_volume / (total_time / sip_interval) : ℚ) = 40 := by
  sorry

#check water_sip_calculation

end NUMINAMATH_CALUDE_water_sip_calculation_l1103_110326


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l1103_110345

/-- Given 4 persons, if replacing one person with a new person weighing 129 kg
    increases the average weight by 8.5 kg, then the weight of the replaced person was 95 kg. -/
theorem weight_of_replaced_person
  (initial_count : Nat)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : initial_count = 4)
  (h2 : weight_increase = 8.5)
  (h3 : new_person_weight = 129) :
  new_person_weight - (initial_count : ℝ) * weight_increase = 95 := by
sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l1103_110345


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1103_110359

theorem max_value_of_expression (x : ℝ) (h : 0 ≤ x ∧ x ≤ 25) :
  Real.sqrt (x + 64) + Real.sqrt (25 - x) + 2 * Real.sqrt x ≤ 19 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1103_110359


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1103_110351

theorem trigonometric_identity (α : ℝ) :
  2 * (Real.sin (3 * π - 2 * α))^2 * (Real.cos (5 * π + 2 * α))^2 =
  1/4 - 1/4 * Real.sin (5/2 * π - 8 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1103_110351


namespace NUMINAMATH_CALUDE_reflected_ray_equation_l1103_110380

/-- The equation of a line passing through two given points is correct. -/
theorem reflected_ray_equation (x y : ℝ) :
  let p1 : ℝ × ℝ := (-1, -3)  -- Symmetric point of (-1, 3) with respect to x-axis
  let p2 : ℝ × ℝ := (4, 6)    -- Given point that the reflected ray passes through
  9 * x - 5 * y - 6 = 0 ↔ (y - p1.2) * (p2.1 - p1.1) = (x - p1.1) * (p2.2 - p1.2) :=
by sorry

end NUMINAMATH_CALUDE_reflected_ray_equation_l1103_110380


namespace NUMINAMATH_CALUDE_arithmetic_series_sum_10_70_1_7th_l1103_110316

def arithmeticSeriesSum (a₁ aₙ d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_series_sum_10_70_1_7th : 
  arithmeticSeriesSum 10 70 (1/7) = 16840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_series_sum_10_70_1_7th_l1103_110316


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1103_110384

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - x < 0}
def N : Set ℝ := {x | |x| < 2}

-- Define propositions p and q
def p (a : ℝ) : Prop := a ∈ M
def q (a : ℝ) : Prop := a ∈ N

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬p a) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l1103_110384


namespace NUMINAMATH_CALUDE_quadratic_expansion_sum_l1103_110342

theorem quadratic_expansion_sum (d : ℝ) (h : d ≠ 0) : 
  ∃ (a b c : ℤ), (15 * d^2 + 15 + 7 * d) + (3 * d + 9)^2 = a * d^2 + b * d + c ∧ a + b + c = 181 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expansion_sum_l1103_110342


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1103_110390

-- Problem 1
theorem problem_1 : -4 * 9 = -36 := by sorry

-- Problem 2
theorem problem_2 : 10 - 14 - (-5) = 1 := by sorry

-- Problem 3
theorem problem_3 : -3 * (-1/3)^3 = 1/9 := by sorry

-- Problem 4
theorem problem_4 : -56 + (-8) * (1/8) = -57 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1103_110390


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l1103_110371

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes,
    with each box containing at least one ball. -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 150 ways to distribute 5 distinguishable balls into 3 indistinguishable boxes,
    with each box containing at least one ball. -/
theorem distribute_five_balls_three_boxes :
  distribute_balls 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l1103_110371


namespace NUMINAMATH_CALUDE_ellipse_midpoint_theorem_l1103_110338

/-- Defines an ellipse with semi-major axis a and semi-minor axis b -/
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}

/-- Defines a line with slope m passing through point (x₀, y₀) -/
def Line (m x₀ y₀ : ℝ) := {p : ℝ × ℝ | p.2 = m * (p.1 - x₀) + y₀}

theorem ellipse_midpoint_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C := Ellipse a b
  let L := Line (4/5) 3 0
  (0, 4) ∈ C ∧ 
  (a^2 - b^2) / a^2 = 9/25 →
  ∃ p q : ℝ × ℝ, p ∈ C ∧ p ∈ L ∧ q ∈ C ∧ q ∈ L ∧ 
  (p.1 + q.1) / 2 = 3/2 ∧ (p.2 + q.2) / 2 = -6/5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_midpoint_theorem_l1103_110338


namespace NUMINAMATH_CALUDE_grape_juice_mixture_l1103_110354

/-- Given an initial mixture with 10% grape juice, adding 20 gallons of pure grape juice
    to create a new mixture with 40% grape juice, prove that the initial mixture
    must have been 40 gallons. -/
theorem grape_juice_mixture (initial_volume : ℝ) : 
  (0.1 * initial_volume + 20) / (initial_volume + 20) = 0.4 → initial_volume = 40 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_mixture_l1103_110354


namespace NUMINAMATH_CALUDE_yuna_has_biggest_number_l1103_110374

def yoongi_number : ℕ := 4
def yuna_number : ℕ := 5
def jungkook_number : ℕ := 6 - 3

theorem yuna_has_biggest_number :
  yuna_number > yoongi_number ∧ yuna_number > jungkook_number :=
by sorry

end NUMINAMATH_CALUDE_yuna_has_biggest_number_l1103_110374


namespace NUMINAMATH_CALUDE_r_eq_m_times_phi_l1103_110307

/-- The algorithm for writing numbers on intersecting circles -/
def writeNumbers (m : ℕ) (n : ℕ) : Set (ℕ × ℕ) := sorry

/-- The number of appearances of a number on the circles -/
def r (n : ℕ) (m : ℕ) : ℕ := sorry

/-- Euler's totient function -/
def φ : ℕ → ℕ := sorry

/-- Theorem stating the relationship between r(n,m) and φ(n) -/
theorem r_eq_m_times_phi (n : ℕ) (m : ℕ) :
  r n m = m * φ n := by sorry

end NUMINAMATH_CALUDE_r_eq_m_times_phi_l1103_110307


namespace NUMINAMATH_CALUDE_profit_maximization_l1103_110364

/-- Represents the daily sales volume as a function of the selling price. -/
def sales_volume (x : ℝ) : ℝ := -20 * x + 1600

/-- Represents the daily profit as a function of the selling price. -/
def profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

theorem profit_maximization (x : ℝ) (h1 : x ≥ 45) (h2 : x < 80) :
  profit x ≤ 8000 ∧ profit 60 = 8000 :=
by sorry

#check profit_maximization

end NUMINAMATH_CALUDE_profit_maximization_l1103_110364


namespace NUMINAMATH_CALUDE_square_starts_with_self_l1103_110361

def starts_with (a b : ℕ) : Prop :=
  ∃ k, a = b * 10^k + (a % 10^k)

theorem square_starts_with_self (N : ℕ) :
  (N > 0) → (starts_with (N^2) N) → ∃ k, N = 10^(k-1) :=
sorry

end NUMINAMATH_CALUDE_square_starts_with_self_l1103_110361


namespace NUMINAMATH_CALUDE_smallest_abs_z_l1103_110332

theorem smallest_abs_z (z : ℂ) (h : Complex.abs (z - 9) + Complex.abs (z - 6*I) = 15) : 
  ∃ (min_abs_z : ℝ), min_abs_z = 3.6 ∧ ∀ w : ℂ, Complex.abs (w - 9) + Complex.abs (w - 6*I) = 15 → Complex.abs w ≥ min_abs_z :=
sorry

end NUMINAMATH_CALUDE_smallest_abs_z_l1103_110332


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1103_110337

def U : Set Int := {-1, 0, 1, 2, 3}
def P : Set Int := {0, 1, 2}
def Q : Set Int := {-1, 0}

theorem complement_union_theorem :
  (U \ P) ∪ Q = {-1, 0, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1103_110337


namespace NUMINAMATH_CALUDE_problem_solution_l1103_110303

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem problem_solution (x y : ℝ) :
  (∀ x, (f x)^2 - (g x)^2 = -4) ∧
  (f x * f y = 4 ∧ g x * g y = 8 → g (x + y) / g (x - y) = 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1103_110303


namespace NUMINAMATH_CALUDE_range_f_is_closed_interval_l1103_110304

/-- The quadratic function f(x) = -x^2 + 4x + 1 -/
def f (x : ℝ) : ℝ := -x^2 + 4*x + 1

/-- The closed interval [0, 3] -/
def I : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

/-- The range of f over the interval I -/
def range_f : Set ℝ := { y | ∃ x ∈ I, f x = y }

theorem range_f_is_closed_interval :
  range_f = { y | 1 ≤ y ∧ y ≤ 5 } := by sorry

end NUMINAMATH_CALUDE_range_f_is_closed_interval_l1103_110304


namespace NUMINAMATH_CALUDE_debate_panel_probability_l1103_110314

def total_members : ℕ := 20
def boys : ℕ := 8
def girls : ℕ := 12
def panel_size : ℕ := 4

theorem debate_panel_probability :
  let total_combinations := Nat.choose total_members panel_size
  let all_boys := Nat.choose boys panel_size
  let all_girls := Nat.choose girls panel_size
  let prob_complement := (all_boys + all_girls : ℚ) / total_combinations
  1 - prob_complement = 856 / 969 := by sorry

end NUMINAMATH_CALUDE_debate_panel_probability_l1103_110314


namespace NUMINAMATH_CALUDE_parabola_bound_l1103_110381

theorem parabola_bound (a b c : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |a * x^2 - b * x + c| < 1) →
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |(a + b) * x^2 + c| < 1) := by
sorry

end NUMINAMATH_CALUDE_parabola_bound_l1103_110381


namespace NUMINAMATH_CALUDE_division_simplification_l1103_110346

theorem division_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (12 * x^2 * y) / (-6 * x * y) = -2 * x :=
by sorry

end NUMINAMATH_CALUDE_division_simplification_l1103_110346


namespace NUMINAMATH_CALUDE_equation_solution_l1103_110394

theorem equation_solution : ∃ c : ℝ, (c - 15) / 3 = (2 * c - 3) / 5 ∧ c = -66 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1103_110394


namespace NUMINAMATH_CALUDE_parabola_vertex_l1103_110347

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := y = -9 * (x - 7)^2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (7, 0)

/-- Theorem: The vertex of the parabola y = -9(x-7)^2 is at the point (7, 0) -/
theorem parabola_vertex :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = vertex :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1103_110347


namespace NUMINAMATH_CALUDE_books_left_after_donation_l1103_110340

/-- Calculates the total number of books left after donation --/
def booksLeftAfterDonation (
  mysteryShelvesCount : ℕ)
  (mysteryBooksPerShelf : ℕ)
  (pictureBooksShelvesCount : ℕ)
  (pictureBooksPerShelf : ℕ)
  (autobiographyShelvesCount : ℕ)
  (autobiographyBooksPerShelf : ℝ)
  (cookbookShelvesCount : ℕ)
  (cookbookBooksPerShelf : ℝ)
  (mysteryBooksDonated : ℕ)
  (pictureBooksdonated : ℕ)
  (autobiographiesDonated : ℕ)
  (cookbooksDonated : ℕ) : ℝ :=
  let totalBooksBeforeDonation :=
    (mysteryShelvesCount * mysteryBooksPerShelf : ℝ) +
    (pictureBooksShelvesCount * pictureBooksPerShelf : ℝ) +
    (autobiographyShelvesCount : ℝ) * autobiographyBooksPerShelf +
    (cookbookShelvesCount : ℝ) * cookbookBooksPerShelf
  let totalBooksDonated :=
    (mysteryBooksDonated + pictureBooksdonated + autobiographiesDonated + cookbooksDonated : ℝ)
  totalBooksBeforeDonation - totalBooksDonated

theorem books_left_after_donation :
  booksLeftAfterDonation 3 9 5 12 4 8.5 2 11.5 7 8 3 5 = 121 := by
  sorry

end NUMINAMATH_CALUDE_books_left_after_donation_l1103_110340


namespace NUMINAMATH_CALUDE_probability_two_ties_l1103_110378

/-- The probability of selecting 2 ties from a boutique with shirts, pants, and ties -/
theorem probability_two_ties (shirts pants ties : ℕ) : 
  shirts = 4 → pants = 8 → ties = 18 → 
  (ties : ℚ) / (shirts + pants + ties) * ((ties - 1) : ℚ) / (shirts + pants + ties - 1) = 51 / 145 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_ties_l1103_110378


namespace NUMINAMATH_CALUDE_min_value_of_f_l1103_110383

theorem min_value_of_f (x : Real) (h : x ∈ Set.Icc (π/4) (5*π/12)) : 
  let f := fun (x : Real) => (Real.sin x)^2 - 2*(Real.cos x)^2 / (Real.sin x * Real.cos x)
  ∃ (m : Real), m = -1 ∧ ∀ y ∈ Set.Icc (π/4) (5*π/12), f y ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1103_110383


namespace NUMINAMATH_CALUDE_inequality_sequence_properties_l1103_110310

/-- Definition of the nth inequality in the sequence -/
def nth_inequality (n : ℕ+) (x : ℝ) : Prop :=
  x + (2*n*(2*n-1))/x < 4*n - 1

/-- Definition of the solution set for the nth inequality -/
def nth_solution_set (n : ℕ+) (x : ℝ) : Prop :=
  (2*n - 1 : ℝ) < x ∧ x < 2*n

/-- Definition of the special inequality with parameter a -/
def special_inequality (a : ℕ+) (x : ℝ) : Prop :=
  x + (12*a)/(x+1) < 4*a + 2

/-- Definition of the solution set for the special inequality -/
def special_solution_set (a : ℕ+) (x : ℝ) : Prop :=
  2 < x ∧ x < 4*a - 1

/-- Main theorem statement -/
theorem inequality_sequence_properties :
  ∀ (n : ℕ+),
    (∀ (x : ℝ), nth_inequality n x ↔ nth_solution_set n x) ∧
    (∀ (a : ℕ+) (x : ℝ), special_inequality a x ↔ special_solution_set a x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_sequence_properties_l1103_110310
