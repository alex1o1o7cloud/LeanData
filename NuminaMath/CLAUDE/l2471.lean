import Mathlib

namespace inscribed_rectangle_sides_l2471_247187

/-- A rectangle inscribed in a triangle -/
structure InscribedRectangle where
  -- Triangle dimensions
  triangleBase : ℝ
  triangleHeight : ℝ
  -- Rectangle side ratio
  rectRatio : ℝ
  -- Rectangle sides
  rectShortSide : ℝ
  rectLongSide : ℝ
  -- Conditions
  triangleBase_pos : 0 < triangleBase
  triangleHeight_pos : 0 < triangleHeight
  rectRatio_pos : 0 < rectRatio
  rectShortSide_pos : 0 < rectShortSide
  rectLongSide_pos : 0 < rectLongSide
  ratio_cond : rectLongSide / rectShortSide = 9 / 5
  inscribed_cond : rectLongSide ≤ triangleBase
  proportion_cond : (triangleHeight - rectShortSide) / triangleHeight = rectLongSide / triangleBase

/-- The sides of the inscribed rectangle are 10 and 18 -/
theorem inscribed_rectangle_sides (r : InscribedRectangle) 
    (h1 : r.triangleBase = 48) 
    (h2 : r.triangleHeight = 16) 
    (h3 : r.rectRatio = 9/5) : 
    r.rectShortSide = 10 ∧ r.rectLongSide = 18 := by
  sorry

end inscribed_rectangle_sides_l2471_247187


namespace max_x_plus_y_on_circle_l2471_247101

theorem max_x_plus_y_on_circle :
  let S := {(x, y) : ℝ × ℝ | x^2 + y^2 - 3*y - 1 = 0}
  ∃ (max : ℝ), ∀ (p : ℝ × ℝ), p ∈ S → p.1 + p.2 ≤ max ∧
  ∃ (q : ℝ × ℝ), q ∈ S ∧ q.1 + q.2 = max ∧
  max = (Real.sqrt 26 + 3) / 2 :=
sorry

end max_x_plus_y_on_circle_l2471_247101


namespace existence_of_twin_prime_divisors_l2471_247143

theorem existence_of_twin_prime_divisors :
  ∃ (n : ℕ) (p₁ p₂ : ℕ), 
    Odd n ∧ 
    0 < n ∧
    Prime p₁ ∧ 
    Prime p₂ ∧ 
    (2^n - 1) % p₁ = 0 ∧ 
    (2^n - 1) % p₂ = 0 ∧ 
    p₁ - p₂ = 2 := by
  sorry

end existence_of_twin_prime_divisors_l2471_247143


namespace tenPeopleCircularArrangements_l2471_247165

/-- The number of unique circular arrangements of n people around a table,
    where rotations are considered the same. -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem stating that the number of unique circular arrangements
    of 10 people is equal to 9! -/
theorem tenPeopleCircularArrangements :
  circularArrangements 10 = Nat.factorial 9 := by
  sorry

end tenPeopleCircularArrangements_l2471_247165


namespace hyperbolas_M_value_l2471_247157

/-- Two hyperbolas with the same asymptotes -/
def hyperbolas_same_asymptotes (M : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧
  (∀ (x y : ℝ), x^2/9 - y^2/16 = 1 → y = k*x ∨ y = -k*x) ∧
  (∀ (x y : ℝ), y^2/25 - x^2/M = 1 → y = k*x ∨ y = -k*x)

/-- The value of M for which the hyperbolas have the same asymptotes -/
theorem hyperbolas_M_value :
  hyperbolas_same_asymptotes (225/16) :=
sorry

end hyperbolas_M_value_l2471_247157


namespace yuna_division_l2471_247188

theorem yuna_division (x : ℚ) : 8 * x = 56 → 42 / x = 6 := by
  sorry

end yuna_division_l2471_247188


namespace plot_length_is_sixty_l2471_247185

/-- Proves that the length of a rectangular plot is 60 metres given the specified conditions -/
theorem plot_length_is_sixty (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_metre : ℝ) (total_cost : ℝ) :
  length = breadth + 20 →
  perimeter = 2 * length + 2 * breadth →
  cost_per_metre = 26.50 →
  total_cost = 5300 →
  perimeter = total_cost / cost_per_metre →
  length = 60 := by
  sorry

end plot_length_is_sixty_l2471_247185


namespace beidou_satellite_altitude_scientific_notation_l2471_247100

theorem beidou_satellite_altitude_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 21500000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.15 ∧ n = 7 := by
  sorry

end beidou_satellite_altitude_scientific_notation_l2471_247100


namespace joggers_regain_sight_main_proof_l2471_247170

/-- The time it takes for two joggers to regain sight of each other after being obscured by a circular stadium --/
theorem joggers_regain_sight (steven_speed linda_speed : ℝ) 
  (path_distance stadium_diameter : ℝ) (initial_distance : ℝ) : ℝ :=
  let t : ℝ := 225
  sorry

/-- The main theorem that proves the time is 225 seconds --/
theorem main_proof : joggers_regain_sight 4 2 300 200 300 = 225 := by
  sorry

end joggers_regain_sight_main_proof_l2471_247170


namespace rotated_line_slope_l2471_247171

theorem rotated_line_slope (m : ℝ) (θ : ℝ) :
  m = -Real.sqrt 3 →
  θ = π / 3 →
  (m * Real.cos θ + Real.sin θ) / (Real.cos θ - m * Real.sin θ) = Real.sqrt 3 := by
  sorry

end rotated_line_slope_l2471_247171


namespace rabbit_speed_problem_l2471_247136

theorem rabbit_speed_problem (rabbit_speed : ℕ) (x : ℕ) : 
  rabbit_speed = 45 →
  2 * (2 * rabbit_speed + x) = 188 →
  x = 4 := by
sorry

end rabbit_speed_problem_l2471_247136


namespace percent_of_number_l2471_247198

/-- 0.1 percent of 12,356 is equal to 12.356 -/
theorem percent_of_number : (0.1 / 100) * 12356 = 12.356 := by sorry

end percent_of_number_l2471_247198


namespace overlap_area_is_0_15_l2471_247150

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by its vertices -/
structure Square where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- A triangle defined by its vertices -/
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- The area of the overlapping region between a square and a triangle -/
def overlapArea (s : Square) (t : Triangle) : ℝ := sorry

/-- The theorem stating the area of overlap between the specific square and triangle -/
theorem overlap_area_is_0_15 :
  let s := Square.mk
    (Point.mk 0 0)
    (Point.mk 2 0)
    (Point.mk 2 2)
    (Point.mk 0 2)
  let t := Triangle.mk
    (Point.mk 3 0)
    (Point.mk 1 2)
    (Point.mk 2 1)
  overlapArea s t = 0.15 := by sorry

end overlap_area_is_0_15_l2471_247150


namespace largest_root_of_cubic_equation_l2471_247186

theorem largest_root_of_cubic_equation :
  let f (x : ℝ) := 4 * x^3 - 17 * x^2 + x + 10
  ∃ (max_root : ℝ), max_root = (25 + Real.sqrt 545) / 8 ∧
    f max_root = 0 ∧
    ∀ (y : ℝ), f y = 0 → y ≤ max_root :=
by sorry

end largest_root_of_cubic_equation_l2471_247186


namespace count_squarish_numbers_l2471_247120

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def first_two_digits (n : ℕ) : ℕ := n / 10000

def middle_two_digits (n : ℕ) : ℕ := (n / 100) % 100

def last_two_digits (n : ℕ) : ℕ := n % 100

def has_no_zero_digit (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 6 → (n / 10^d) % 10 ≠ 0

def is_squarish (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧
  is_perfect_square n ∧
  has_no_zero_digit n ∧
  is_perfect_square (first_two_digits n) ∧
  is_perfect_square (middle_two_digits n) ∧
  is_perfect_square (last_two_digits n)

theorem count_squarish_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_squarish n) ∧ s.card = 2 :=
sorry

end count_squarish_numbers_l2471_247120


namespace transformed_area_l2471_247104

/-- Given a region T in the plane with area 9, prove that when transformed
    by the matrix [[3, 0], [8, 3]], the resulting region T' has an area of 81. -/
theorem transformed_area (T : Set (Fin 2 → ℝ)) (harea : MeasureTheory.volume T = 9) :
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 8, 3]
  let T' := (fun p ↦ M.mulVec p) '' T
  MeasureTheory.volume T' = 81 := by
sorry

end transformed_area_l2471_247104


namespace runner_stops_in_D_l2471_247103

/-- Represents the quarters of the circular track -/
inductive Quarter : Type
| A : Quarter
| B : Quarter
| C : Quarter
| D : Quarter

/-- The circular track -/
structure Track :=
  (circumference : ℝ)
  (start : Quarter)

/-- Determines the quarter where a runner stops after running a given distance -/
def stop_quarter (t : Track) (distance : ℝ) : Quarter :=
  sorry

/-- The main theorem -/
theorem runner_stops_in_D (t : Track) (distance : ℝ) :
  t.circumference = 40 ∧ t.start = Quarter.D ∧ distance = 1600 →
  stop_quarter t distance = Quarter.D :=
sorry

end runner_stops_in_D_l2471_247103


namespace orange_crates_pigeonhole_l2471_247126

theorem orange_crates_pigeonhole :
  ∀ (crate_contents : Fin 150 → ℕ),
  (∀ i, 130 ≤ crate_contents i ∧ crate_contents i ≤ 150) →
  ∃ n : ℕ, 130 ≤ n ∧ n ≤ 150 ∧ (Finset.filter (λ i => crate_contents i = n) Finset.univ).card ≥ 8 :=
by sorry

end orange_crates_pigeonhole_l2471_247126


namespace probability_at_least_one_one_l2471_247161

-- Define the number of sides on each die
def num_sides : ℕ := 8

-- Define the probability of at least one die showing 1
def prob_at_least_one_one : ℚ := 15 / 64

-- Theorem statement
theorem probability_at_least_one_one :
  let total_outcomes := num_sides * num_sides
  let outcomes_without_one := (num_sides - 1) * (num_sides - 1)
  let favorable_outcomes := total_outcomes - outcomes_without_one
  (favorable_outcomes : ℚ) / total_outcomes = prob_at_least_one_one := by
  sorry

end probability_at_least_one_one_l2471_247161


namespace cost_price_calculation_l2471_247116

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 500)
  (h2 : profit_percentage = 25) :
  selling_price / (1 + profit_percentage / 100) = 400 := by
sorry

end cost_price_calculation_l2471_247116


namespace bella_earrings_l2471_247155

/-- Given three friends Bella, Monica, and Rachel, with the following conditions:
    1. Bella has 25% of Monica's earrings
    2. Monica has twice as many earrings as Rachel
    3. The total number of earrings among the three friends is 70
    Prove that Bella has 10 earrings. -/
theorem bella_earrings (bella monica rachel : ℕ) : 
  bella = (25 : ℕ) * monica / 100 →
  monica = 2 * rachel →
  bella + monica + rachel = 70 →
  bella = 10 := by
sorry

end bella_earrings_l2471_247155


namespace jason_initial_cards_l2471_247134

/-- The number of Pokemon cards Jason gave away -/
def cards_given_away : ℕ := 9

/-- The number of Pokemon cards Jason has left -/
def cards_left : ℕ := 4

/-- The initial number of Pokemon cards Jason had -/
def initial_cards : ℕ := cards_given_away + cards_left

theorem jason_initial_cards : initial_cards = 13 := by
  sorry

end jason_initial_cards_l2471_247134


namespace coin_collection_value_l2471_247127

theorem coin_collection_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℚ) :
  total_coins = 15 →
  sample_coins = 5 →
  sample_value = 12 →
  (total_coins : ℚ) * (sample_value / sample_coins) = 36 :=
by sorry

end coin_collection_value_l2471_247127


namespace sticker_count_l2471_247199

theorem sticker_count (stickers_per_page : ℕ) (total_pages : ℕ) : 
  stickers_per_page = 10 → total_pages = 22 → stickers_per_page * total_pages = 220 :=
by sorry

end sticker_count_l2471_247199


namespace triangle_angle_c_value_l2471_247158

theorem triangle_angle_c_value (A B C x : ℝ) : 
  A = 45 ∧ B = 3 * x ∧ C = (1 / 2) * B ∧ A + B + C = 180 → C = 45 := by
  sorry

end triangle_angle_c_value_l2471_247158


namespace linear_function_through_0_3_l2471_247195

/-- A linear function passing through (0,3) -/
def LinearFunctionThrough0_3 (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y - f 0) ∧ f 0 = 3

theorem linear_function_through_0_3 (f : ℝ → ℝ) (hf : LinearFunctionThrough0_3 f) :
  ∃ m : ℝ, ∀ x : ℝ, f x = m * x + 3 :=
sorry

end linear_function_through_0_3_l2471_247195


namespace inverse_proportion_constant_l2471_247189

/-- Given two points A(3,m) and B(3m-1,2) on the graph of y = k/x, prove that k = 2 -/
theorem inverse_proportion_constant (k m : ℝ) : 
  (3 * m = k) ∧ (2 * (3 * m - 1) = k) → k = 2 := by
  sorry

end inverse_proportion_constant_l2471_247189


namespace eddies_sister_pies_per_day_l2471_247130

theorem eddies_sister_pies_per_day :
  let eddie_pies_per_day : ℕ := 3
  let mother_pies_per_day : ℕ := 8
  let total_days : ℕ := 7
  let total_pies : ℕ := 119
  ∃ (sister_pies_per_day : ℕ),
    sister_pies_per_day * total_days + eddie_pies_per_day * total_days + mother_pies_per_day * total_days = total_pies ∧
    sister_pies_per_day = 6 :=
by sorry

end eddies_sister_pies_per_day_l2471_247130


namespace painters_time_equation_l2471_247197

/-- The time it takes for two painters to paint a room together, given their individual rates and a lunch break -/
theorem painters_time_equation (doug_rate : ℚ) (dave_rate : ℚ) (t : ℚ) 
  (h_doug : doug_rate = 1 / 5)
  (h_dave : dave_rate = 1 / 7)
  (h_positive : t > 0) :
  (doug_rate + dave_rate) * (t - 1) = 1 ↔ t = 47 / 12 :=
by sorry

end painters_time_equation_l2471_247197


namespace paco_cookies_bought_l2471_247113

-- Define the initial number of cookies
def initial_cookies : ℕ := 13

-- Define the number of cookies eaten
def cookies_eaten : ℕ := 2

-- Define the additional cookies compared to eaten ones
def additional_cookies : ℕ := 34

-- Define the function to calculate the number of cookies bought
def cookies_bought (initial : ℕ) (eaten : ℕ) (additional : ℕ) : ℕ :=
  additional + eaten

-- Theorem statement
theorem paco_cookies_bought :
  cookies_bought initial_cookies cookies_eaten additional_cookies = 36 := by
  sorry

end paco_cookies_bought_l2471_247113


namespace max_value_expression_l2471_247125

theorem max_value_expression (x y z : ℝ) (h1 : x + 2*y + z = 7) (h2 : y ≥ 0) :
  ∃ M : ℝ, M = (10.5 : ℝ) ∧ ∀ x' y' z' : ℝ, x' + 2*y' + z' = 7 → y' ≥ 0 →
    x'*y' + x'*z' + y'*z' + y'^2 ≤ M :=
by sorry

end max_value_expression_l2471_247125


namespace license_plate_count_l2471_247118

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The total number of characters in the license plate -/
def total_chars : ℕ := 8

/-- The number of digits in the license plate -/
def num_plate_digits : ℕ := 5

/-- The number of letters in the license plate -/
def num_plate_letters : ℕ := 3

/-- The number of possible starting positions for the letter block -/
def letter_block_positions : ℕ := 6

/-- Calculates the total number of distinct license plates -/
def total_license_plates : ℕ :=
  letter_block_positions * num_digits ^ num_plate_digits * num_letters ^ num_plate_letters

theorem license_plate_count :
  total_license_plates = 10545600000 := by sorry

end license_plate_count_l2471_247118


namespace problem_statement_l2471_247169

theorem problem_statement (x y : ℝ) : 
  |x - 2| + (y + 3)^2 = 0 → (x + y)^2020 = 1 := by sorry

end problem_statement_l2471_247169


namespace first_four_eq_last_four_l2471_247147

/-- A sequence of 0s and 1s -/
def BinarySeq := List Bool

/-- Checks if two segments of 5 terms are different -/
def differentSegments (s : BinarySeq) (i j : Nat) : Prop :=
  i < j ∧ j + 4 < s.length ∧
  (List.take 5 (List.drop i s) ≠ List.take 5 (List.drop j s))

/-- The condition that any two consecutive segments of 5 terms are different -/
def validSequence (s : BinarySeq) : Prop :=
  ∀ i j, i < j → j + 4 < s.length → differentSegments s i j

/-- S is the longest sequence satisfying the condition -/
def longestValidSequence (S : BinarySeq) : Prop :=
  validSequence S ∧ ∀ s, validSequence s → s.length ≤ S.length

theorem first_four_eq_last_four (S : BinarySeq) (h : longestValidSequence S) :
  S.take 4 = (S.reverse.take 4).reverse :=
sorry

end first_four_eq_last_four_l2471_247147


namespace class_representation_ratio_l2471_247132

theorem class_representation_ratio (boys girls : ℕ) 
  (h1 : boys + girls > 0)  -- ensure non-empty class
  (h2 : (boys : ℚ) / (boys + girls : ℚ) = 3/4 * (girls : ℚ) / (boys + girls : ℚ)) :
  (boys : ℚ) / (boys + girls : ℚ) = 3/7 := by
sorry

end class_representation_ratio_l2471_247132


namespace initial_mixture_volume_l2471_247102

/-- Prove that the initial volume of a milk-water mixture is 155 liters -/
theorem initial_mixture_volume (milk : ℝ) (water : ℝ) : 
  milk / water = 3 / 2 →  -- Initial ratio of milk to water
  milk / (water + 62) = 3 / 4 →  -- New ratio after adding 62 liters of water
  milk + water = 155 :=  -- Initial volume of the mixture
by sorry

end initial_mixture_volume_l2471_247102


namespace perpendicular_lines_slope_product_l2471_247193

/-- Two lines in a 2D plane are perpendicular iff the product of their slopes is -1 -/
theorem perpendicular_lines_slope_product (k₁ k₂ l₁ l₂ : ℝ) (hk₁ : k₁ ≠ 0) (hk₂ : k₂ ≠ 0) :
  (∀ x y₁ y₂ : ℝ, y₁ = k₁ * x + l₁ ∧ y₂ = k₂ * x + l₂) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, y₁ = k₁ * x₁ + l₁ ∧ y₂ = k₂ * x₂ + l₂ → 
    ((x₂ - x₁) * (k₁ * x₁ + l₁ - (k₂ * x₂ + l₂)) + (x₂ - x₁) * (y₂ - y₁) = 0)) ↔
  k₁ * k₂ = -1 :=
sorry

end perpendicular_lines_slope_product_l2471_247193


namespace count_two_repeating_digits_l2471_247109

/-- A four-digit number is a natural number between 1000 and 9999 inclusive. -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that counts the occurrences of each digit in a four-digit number. -/
def DigitCount (n : ℕ) : ℕ → ℕ := sorry

/-- A four-digit number has exactly two repeating digits if exactly one digit appears twice
    and the other two digits appear once each. -/
def HasExactlyTwoRepeatingDigits (n : ℕ) : Prop :=
  FourDigitNumber n ∧ ∃! d : ℕ, DigitCount n d = 2

/-- The count of four-digit numbers with exactly two repeating digits. -/
def CountTwoRepeatingDigits : ℕ := sorry

/-- The main theorem stating that the count of four-digit numbers with exactly two repeating digits is 2736. -/
theorem count_two_repeating_digits :
  CountTwoRepeatingDigits = 2736 := by sorry

end count_two_repeating_digits_l2471_247109


namespace range_of_m_l2471_247105

theorem range_of_m (m : ℝ) : 
  (∃ (x₁ x₂ x₃ x₄ : ℤ), 
    (∀ x : ℤ, (3 * ↑x - m > 0 ∧ ↑x - 1 ≤ 5) ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄))) →
  (6 ≤ m ∧ m < 9) :=
by sorry

end range_of_m_l2471_247105


namespace clementine_baked_72_cookies_l2471_247110

/-- The number of cookies Clementine baked -/
def clementine_cookies : ℕ := 72

/-- The number of cookies Jake baked -/
def jake_cookies : ℕ := 2 * clementine_cookies

/-- The number of cookies Tory baked -/
def tory_cookies : ℕ := (clementine_cookies + jake_cookies) / 2

/-- The price of each cookie in dollars -/
def cookie_price : ℕ := 2

/-- The total amount of money made from selling cookies in dollars -/
def total_money : ℕ := 648

theorem clementine_baked_72_cookies :
  clementine_cookies = 72 ∧
  jake_cookies = 2 * clementine_cookies ∧
  tory_cookies = (clementine_cookies + jake_cookies) / 2 ∧
  cookie_price = 2 ∧
  total_money = 648 ∧
  total_money = cookie_price * (clementine_cookies + jake_cookies + tory_cookies) :=
by sorry

end clementine_baked_72_cookies_l2471_247110


namespace quadratic_inequality_l2471_247153

theorem quadratic_inequality (y : ℝ) : y^2 - 8*y + 12 < 0 ↔ 2 < y ∧ y < 6 := by
  sorry

end quadratic_inequality_l2471_247153


namespace range_of_a_l2471_247184

-- Define propositions p and q
def p (a : ℝ) : Prop := -3 < a ∧ a < 1
def q (a : ℝ) : Prop := a > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) → ((-3 < a ∧ a ≤ 0) ∨ a ≥ 1) :=
by sorry

end range_of_a_l2471_247184


namespace power_sum_equality_l2471_247179

theorem power_sum_equality (x : ℝ) : x^3 * x + x^2 * x^2 = 2 * x^4 := by
  sorry

end power_sum_equality_l2471_247179


namespace bowling_ball_weight_l2471_247111

theorem bowling_ball_weight (b k : ℝ) 
  (h1 : 5 * b = 3 * k) 
  (h2 : 4 * k = 120) : 
  b = 18 := by
sorry

end bowling_ball_weight_l2471_247111


namespace probability_of_specific_pairing_l2471_247106

theorem probability_of_specific_pairing (n : ℕ) (h : n = 25) :
  let total_students := n
  let available_partners := n - 1
  (1 : ℚ) / available_partners = 1 / 24 :=
by
  sorry

end probability_of_specific_pairing_l2471_247106


namespace parabola_vertex_and_point_l2471_247181

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y (p : Parabola) (x : ℚ) : ℚ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_vertex_and_point (p : Parabola) :
  p.y 2 = 1 → p.y 0 = 5 → p.a + p.b - p.c = -8 := by
  sorry

end parabola_vertex_and_point_l2471_247181


namespace direct_proportion_through_point_one_three_l2471_247141

/-- A direct proportion function passing through (1, 3) has k = 3 -/
theorem direct_proportion_through_point_one_three (k : ℝ) : 
  (∀ x y : ℝ, y = k * x) → -- Direct proportion function
  (3 : ℝ) = k * (1 : ℝ) →  -- Passes through (1, 3)
  k = 3 := by sorry

end direct_proportion_through_point_one_three_l2471_247141


namespace smallest_positive_largest_negative_expression_l2471_247194

theorem smallest_positive_largest_negative_expression :
  ∃ (a b c d : ℚ),
    (∀ n : ℚ, n > 0 → a ≤ n) ∧
    (∀ n : ℚ, n < 0 → n ≤ b) ∧
    (∀ n : ℚ, n ≠ 0 → |c| ≤ |n|) ∧
    (d⁻¹ = d) ∧
    (a - b + c^2 - |d| = 1) := by
  sorry

end smallest_positive_largest_negative_expression_l2471_247194


namespace train_crossing_time_l2471_247140

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_length : ℝ) (signal_crossing_time : ℝ) (platform_length : ℝ) :
  train_length = 300 →
  signal_crossing_time = 18 →
  platform_length = 400 →
  ∃ (platform_crossing_time : ℝ), abs (platform_crossing_time - 42) < 0.1 := by
  sorry


end train_crossing_time_l2471_247140


namespace geometric_sequence_property_l2471_247156

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Theorem: In a geometric sequence, if a_3 * a_4 = 6, then a_2 * a_5 = 6 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_prod : a 3 * a 4 = 6) : a 2 * a 5 = 6 := by
  sorry

end geometric_sequence_property_l2471_247156


namespace winning_strategy_works_l2471_247123

/-- Represents the game state with blue and white balls --/
structure GameState where
  blue : ℕ
  white : ℕ

/-- Represents a player's move --/
inductive Move
  | TakeBlue
  | TakeWhite

/-- Applies a move to the game state --/
def applyMove (state : GameState) (move : Move) : GameState :=
  match move with
  | Move.TakeBlue => { blue := state.blue - 3, white := state.white }
  | Move.TakeWhite => { blue := state.blue, white := state.white - 2 }

/-- Checks if the game is over (no balls left) --/
def isGameOver (state : GameState) : Prop :=
  state.blue = 0 ∧ state.white = 0

/-- Represents the winning strategy --/
def winningStrategy (state : GameState) : Prop :=
  3 * state.white = 2 * state.blue

/-- The main theorem to prove --/
theorem winning_strategy_works (initialState : GameState)
  (h_initial : initialState.blue = 15 ∧ initialState.white = 12) :
  ∃ (firstMove : Move),
    let stateAfterFirstMove := applyMove initialState firstMove
    winningStrategy stateAfterFirstMove ∧
    (∀ (opponentMove : Move),
      let stateAfterOpponent := applyMove stateAfterFirstMove opponentMove
      ∃ (response : Move),
        let stateAfterResponse := applyMove stateAfterOpponent response
        winningStrategy stateAfterResponse ∨ isGameOver stateAfterResponse) :=
sorry

end winning_strategy_works_l2471_247123


namespace trajectory_is_line_segment_l2471_247107

/-- The trajectory of a point M satisfying |MF₁| + |MF₂| = 8 is a line segment -/
theorem trajectory_is_line_segment (M : ℝ × ℝ) : 
  let F₁ : ℝ × ℝ := (-4, 0)
  let F₂ : ℝ × ℝ := (4, 0)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  (dist M F₁ + dist M F₂ = 8) →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * F₂.1 + (1 - t) * F₁.1, t * F₂.2 + (1 - t) * F₁.2) :=
by sorry

end trajectory_is_line_segment_l2471_247107


namespace isosceles_right_triangle_l2471_247183

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the altitude from a point to a line segment
def altitude (p : ℝ × ℝ) (q r : ℝ × ℝ) : ℝ := sorry

-- Define the angle at a vertex
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem isosceles_right_triangle (t : Triangle) :
  altitude t.A t.B t.C ≥ length t.B t.C →
  altitude t.B t.A t.C ≥ length t.A t.C →
  angle t.B t.A t.C = 90 ∧ angle t.A t.B t.C = 45 ∧ angle t.A t.C t.B = 45 :=
by sorry

end isosceles_right_triangle_l2471_247183


namespace opposite_absolute_value_and_square_l2471_247178

theorem opposite_absolute_value_and_square (x y : ℝ) :
  |x + y - 2| + (2*x - 3*y + 5)^2 = 0 → x = 1/5 ∧ y = 9/5 := by
  sorry

end opposite_absolute_value_and_square_l2471_247178


namespace identify_roles_l2471_247162

-- Define the possible roles
inductive Role
  | Knight
  | Liar
  | Normal

-- Define the statements made by each individual
def statement_A : Prop := ∃ x, x = Role.Normal
def statement_B : Prop := statement_A
def statement_C : Prop := ¬∃ x, x = Role.Normal

-- Define the properties of each role
def always_true (r : Role) : Prop := r = Role.Knight
def always_false (r : Role) : Prop := r = Role.Liar
def can_be_either (r : Role) : Prop := r = Role.Normal

-- The main theorem
theorem identify_roles :
  ∃! (role_A role_B role_C : Role),
    -- Each person has a unique role
    role_A ≠ role_B ∧ role_B ≠ role_C ∧ role_A ≠ role_C ∧
    -- One of each role exists
    (always_true role_A ∨ always_true role_B ∨ always_true role_C) ∧
    (always_false role_A ∨ always_false role_B ∨ always_false role_C) ∧
    (can_be_either role_A ∨ can_be_either role_B ∨ can_be_either role_C) ∧
    -- Statements are consistent with roles
    ((always_true role_A → statement_A) ∧ (always_false role_A → ¬statement_A) ∧ (can_be_either role_A → True)) ∧
    ((always_true role_B → statement_B) ∧ (always_false role_B → ¬statement_B) ∧ (can_be_either role_B → True)) ∧
    ((always_true role_C → statement_C) ∧ (always_false role_C → ¬statement_C) ∧ (can_be_either role_C → True)) ∧
    -- The solution
    always_false role_A ∧ always_true role_B ∧ can_be_either role_C :=
by sorry


end identify_roles_l2471_247162


namespace delta_y_over_delta_x_l2471_247148

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 4

-- Define the theorem
theorem delta_y_over_delta_x (Δx : ℝ) (Δy : ℝ) (h1 : f 1 = -2) (h2 : f (1 + Δx) = -2 + Δy) :
  Δy / Δx = 4 + 2 * Δx := by
  sorry

end delta_y_over_delta_x_l2471_247148


namespace rhombus_perimeter_l2471_247177

theorem rhombus_perimeter (x₁ x₂ : ℝ) : 
  x₁^2 - 14*x₁ + 48 = 0 →
  x₂^2 - 14*x₂ + 48 = 0 →
  x₁ ≠ x₂ →
  let s := Real.sqrt ((x₁^2 + x₂^2) / 4)
  4 * s = 20 := by sorry

end rhombus_perimeter_l2471_247177


namespace f_range_l2471_247192

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((1 - x) / (1 + x))

theorem f_range : ∀ x : ℝ, f x = -3 * π / 4 ∨ f x = π / 4 := by
  sorry

end f_range_l2471_247192


namespace vector_operation_proof_l2471_247164

def vector_a : Fin 2 → ℝ := ![2, 4]
def vector_b : Fin 2 → ℝ := ![-1, 1]

theorem vector_operation_proof :
  (2 • vector_a - vector_b) = ![5, 7] := by sorry

end vector_operation_proof_l2471_247164


namespace one_thirds_in_nine_thirds_l2471_247124

theorem one_thirds_in_nine_thirds : (9 : ℚ) / 3 / (1 : ℚ) / 3 = 9 := by
  sorry

end one_thirds_in_nine_thirds_l2471_247124


namespace triangle_area_l2471_247172

/-- The area of a triangle with vertices at (2, 1), (2, 7), and (8, 4) is 18 square units -/
theorem triangle_area : ℝ := by
  -- Define the vertices of the triangle
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (2, 7)
  let C : ℝ × ℝ := (8, 4)

  -- Calculate the area using the formula: Area = (1/2) * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
  let area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

  -- Prove that the calculated area equals 18
  sorry

end triangle_area_l2471_247172


namespace correct_book_arrangements_l2471_247145

/-- The number of ways to arrange 11 books (3 Arabic, 4 German, 4 Spanish) on a shelf, keeping the Arabic books together -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 11
  let arabic_books : ℕ := 3
  let german_books : ℕ := 4
  let spanish_books : ℕ := 4
  let arabic_unit : ℕ := 1
  let total_units : ℕ := arabic_unit + german_books + spanish_books
  (Nat.factorial total_units) * (Nat.factorial arabic_books)

theorem correct_book_arrangements :
  book_arrangements = 2177280 :=
by sorry

end correct_book_arrangements_l2471_247145


namespace root_difference_of_cubic_l2471_247121

theorem root_difference_of_cubic (x : ℝ → ℝ) :
  (∀ t, 81 * (x t)^3 - 162 * (x t)^2 + 81 * (x t) - 8 = 0) →
  (∃ a d, ∀ t, x t = a + d * t) →
  (∃ t₁ t₂, ∀ t, x t₁ ≤ x t ∧ x t ≤ x t₂) →
  x t₂ - x t₁ = 4 * Real.sqrt 6 / 9 := by
sorry

end root_difference_of_cubic_l2471_247121


namespace total_cards_l2471_247151

theorem total_cards (mao_cards : ℕ) (li_cards : ℕ) 
  (h1 : mao_cards = 23) (h2 : li_cards = 20) : 
  mao_cards + li_cards = 43 := by
  sorry

end total_cards_l2471_247151


namespace simplify_expression_l2471_247173

theorem simplify_expression (a : ℝ) (h : a > 1) :
  (1 - a) * Real.sqrt (1 / (a - 1)) = -Real.sqrt (a - 1) := by
  sorry

end simplify_expression_l2471_247173


namespace nancy_carrots_l2471_247152

/-- The number of carrots Nancy picked the next day -/
def carrots_picked_next_day (initial_carrots : ℕ) (thrown_out : ℕ) (total_carrots : ℕ) : ℕ :=
  total_carrots - (initial_carrots - thrown_out)

/-- Proof that Nancy picked 21 carrots the next day -/
theorem nancy_carrots :
  carrots_picked_next_day 12 2 31 = 21 := by
  sorry

end nancy_carrots_l2471_247152


namespace y_value_proof_l2471_247196

theorem y_value_proof : ∀ y : ℝ, (1/3 - 1/4 = 4/y) → y = 48 := by
  sorry

end y_value_proof_l2471_247196


namespace candy_distribution_l2471_247108

theorem candy_distribution (n : ℕ) (h : n = 30) :
  (min (n % 4) ((4 - n % 4) % 4)) = 2 := by
  sorry

end candy_distribution_l2471_247108


namespace function_decreasing_interval_l2471_247137

def f (a b x : ℝ) : ℝ := x^3 - 3*a*x + b

theorem function_decreasing_interval
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : ∃ x1, f a b x1 = 6 ∧ ∀ x, f a b x ≤ 6)
  (h3 : ∃ x2, f a b x2 = 2 ∧ ∀ x, f a b x ≥ 2) :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1,
    x < y → f a b x > f a b y :=
by sorry

end function_decreasing_interval_l2471_247137


namespace x_square_plus_reciprocal_l2471_247154

theorem x_square_plus_reciprocal (x : ℝ) (h : 31 = x^6 + 1/x^6) : 
  x^2 + 1/x^2 = (34 : ℝ)^(1/3) := by
  sorry

end x_square_plus_reciprocal_l2471_247154


namespace chemist_problem_solution_l2471_247160

/-- Represents the purity of a salt solution as a real number between 0 and 1 -/
def Purity := { p : ℝ // 0 ≤ p ∧ p ≤ 1 }

/-- The chemist's problem setup -/
structure ChemistProblem where
  solution1 : Purity
  solution2 : Purity
  total_amount : ℝ
  final_purity : Purity
  amount_solution1 : ℝ
  h1 : solution1.val = 0.3
  h2 : total_amount = 60
  h3 : final_purity.val = 0.5
  h4 : amount_solution1 = 40

theorem chemist_problem_solution (p : ChemistProblem) : p.solution2.val = 0.9 := by
  sorry

end chemist_problem_solution_l2471_247160


namespace andrew_fruit_purchase_l2471_247159

/-- Calculates the total amount paid for fruits given the quantities and prices -/
def totalAmountPaid (grapeQuantity mangoQuantity grapePrice mangoPrice : ℕ) : ℕ :=
  grapeQuantity * grapePrice + mangoQuantity * mangoPrice

/-- Theorem stating that Andrew paid 975 for his fruit purchase -/
theorem andrew_fruit_purchase : 
  totalAmountPaid 6 9 74 59 = 975 := by
  sorry

end andrew_fruit_purchase_l2471_247159


namespace max_green_socks_l2471_247175

/-- Represents the count of socks in a basket -/
structure SockBasket where
  green : ℕ
  yellow : ℕ
  total_bound : green + yellow ≤ 2025

/-- The probability of selecting two green socks without replacement -/
def prob_two_green (b : SockBasket) : ℚ :=
  (b.green * (b.green - 1)) / ((b.green + b.yellow) * (b.green + b.yellow - 1))

/-- Theorem stating the maximum number of green socks possible -/
theorem max_green_socks (b : SockBasket) 
  (h : prob_two_green b = 1/3) : 
  b.green ≤ 990 ∧ ∃ b' : SockBasket, b'.green = 990 ∧ prob_two_green b' = 1/3 :=
sorry

end max_green_socks_l2471_247175


namespace lcm_of_180_504_169_l2471_247174

def a : ℕ := 180
def b : ℕ := 504
def c : ℕ := 169

theorem lcm_of_180_504_169 : 
  Nat.lcm (Nat.lcm a b) c = 2^3 * 3^2 * 5 * 7 * 13^2 := by sorry

end lcm_of_180_504_169_l2471_247174


namespace min_a3_and_a2b2_l2471_247139

theorem min_a3_and_a2b2 (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0)
  (hseq1 : a₂ = a₁ + b₁ ∧ a₃ = a₁ + 2*b₁)
  (hseq2 : b₂ = b₁ * a₁ ∧ b₃ = b₁ * a₁^2)
  (heq : a₃ = b₃) :
  (∀ a₁' a₂' a₃' b₁' b₂' b₃' : ℝ, 
    (a₁' > 0 ∧ a₂' > 0 ∧ a₃' > 0 ∧ b₁' > 0 ∧ b₂' > 0 ∧ b₃' > 0) →
    (a₂' = a₁' + b₁' ∧ a₃' = a₁' + 2*b₁') →
    (b₂' = b₁' * a₁' ∧ b₃' = b₁' * a₁'^2) →
    (a₃' = b₃') →
    a₃' ≥ 3 * Real.sqrt 6 / 2) ∧
  (a₃ = 3 * Real.sqrt 6 / 2 → a₂ * b₂ = 15 * Real.sqrt 6 / 8) :=
sorry

end min_a3_and_a2b2_l2471_247139


namespace geometric_sequence_ratio_l2471_247138

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

-- Define the conditions and theorem
theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a3 : a 3 = 2) 
  (h_a4a6 : a 4 * a 6 = 16) :
  (a 9 - a 11) / (a 5 - a 7) = 4 := by
  sorry

end geometric_sequence_ratio_l2471_247138


namespace three_digit_sum_l2471_247163

theorem three_digit_sum (a b c : Nat) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a < 10 ∧ b < 10 ∧ c < 10 →
  (1730 + a) % 9 = 0 →
  (1730 + b) % 11 = 0 →
  (1730 + c) % 6 = 0 →
  a + b + c = 19 := by
sorry

end three_digit_sum_l2471_247163


namespace solve_for_b_l2471_247142

/-- Given a system of equations and its solution, prove the value of b. -/
theorem solve_for_b (a : ℝ) : 
  (∃ x y : ℝ, a * x - 2 * y = 1 ∧ 2 * x + b * y = 5) →
  (∃ x y : ℝ, x = 1 ∧ y = a ∧ a * x - 2 * y = 1 ∧ 2 * x + b * y = 5) →
  b = -3 :=
by
  sorry

end solve_for_b_l2471_247142


namespace factorization_equality_l2471_247180

theorem factorization_equality (a b : ℝ) : 12 * b^3 - 3 * a^2 * b = 3 * b * (2*b + a) * (2*b - a) := by
  sorry

end factorization_equality_l2471_247180


namespace chess_tournament_participants_perfect_square_l2471_247135

theorem chess_tournament_participants_perfect_square 
  (B : ℕ) -- number of boys
  (G : ℕ) -- number of girls
  (total_points : ℕ → ℕ → ℕ) -- function that calculates total points given boys and girls
  (h1 : ∀ x y, total_points x y = x * y) -- each participant plays once with every other
  (h2 : ∀ x y, 2 * (x * y) = x * (x - 1) + y * (y - 1)) -- half points from boys
  : ∃ k : ℕ, B + G = k^2 :=
sorry

end chess_tournament_participants_perfect_square_l2471_247135


namespace hanna_has_zero_erasers_l2471_247149

/-- The number of erasers Tanya has -/
def tanya_total : ℕ := 30

/-- The number of red erasers Tanya has -/
def tanya_red : ℕ := tanya_total / 2

/-- The number of blue erasers Tanya has -/
def tanya_blue : ℕ := tanya_total / 3

/-- The number of yellow erasers Tanya has -/
def tanya_yellow : ℕ := tanya_total - tanya_red - tanya_blue

/-- Rachel's erasers in terms of Tanya's red erasers -/
def rachel_erasers : ℤ := tanya_red / 3 - 5

/-- Hanna's erasers in terms of Rachel's -/
def hanna_erasers : ℤ := 3 * rachel_erasers

theorem hanna_has_zero_erasers :
  tanya_yellow = 2 * tanya_blue → hanna_erasers = 0 := by sorry

end hanna_has_zero_erasers_l2471_247149


namespace polynomial_division_result_l2471_247122

-- Define the polynomials f and d
def f (x : ℝ) : ℝ := 3 * x^4 - 9 * x^3 + 6 * x^2 + 2 * x - 5
def d (x : ℝ) : ℝ := x^2 - 2 * x + 1

-- State the theorem
theorem polynomial_division_result :
  ∃ (q r : ℝ → ℝ), 
    (∀ x, f x = q x * d x + r x) ∧ 
    (∀ x, r x = 14) ∧
    (q 1 + r (-1) = 17) := by
  sorry

end polynomial_division_result_l2471_247122


namespace number_of_elements_in_set_l2471_247117

theorem number_of_elements_in_set (S : ℝ) (n : ℕ) 
  (h1 : (S + 26) / n = 5)
  (h2 : (S + 36) / n = 6) :
  n = 10 := by
  sorry

end number_of_elements_in_set_l2471_247117


namespace negative_fraction_comparison_l2471_247168

theorem negative_fraction_comparison : -5/6 > -6/7 := by
  sorry

end negative_fraction_comparison_l2471_247168


namespace angle_in_second_quadrant_l2471_247190

theorem angle_in_second_quadrant (α : Real) (x : Real) :
  -- α is in the second quadrant
  π / 2 < α ∧ α < π →
  -- p(x, √5) is on the terminal side of α
  ∃ (r : Real), r > 0 ∧ x = r * Real.cos α ∧ Real.sqrt 5 = r * Real.sin α →
  -- cos α = (√2/4)x
  Real.cos α = (Real.sqrt 2 / 4) * x →
  -- x = -√3
  x = -Real.sqrt 3 := by
sorry

end angle_in_second_quadrant_l2471_247190


namespace base5_123_to_base10_l2471_247182

/-- Converts a base-5 number to base-10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 5 * acc + d) 0

/-- The base-5 representation of 123 --/
def base5_123 : List Nat := [1, 2, 3]

theorem base5_123_to_base10 :
  base5ToBase10 base5_123 = 38 := by
  sorry

end base5_123_to_base10_l2471_247182


namespace inequality_proof_l2471_247115

theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) :
  1 / (a * b^2) > 1 / (a^2 * b) := by
  sorry

end inequality_proof_l2471_247115


namespace rectangular_plot_breadth_l2471_247146

/-- Proves that the breadth of a rectangular plot is 14 meters, given that its length is thrice its breadth and its area is 588 square meters. -/
theorem rectangular_plot_breadth (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * breadth → 
  area = length * breadth → 
  area = 588 → 
  breadth = 14 := by
sorry

end rectangular_plot_breadth_l2471_247146


namespace power_equality_l2471_247133

theorem power_equality (x : ℝ) : (1/8 : ℝ) * 2^36 = 8^x → x = 11 := by
  sorry

end power_equality_l2471_247133


namespace sine_cosine_sum_l2471_247128

/-- Given an angle α whose terminal side passes through the point (-3a, 4a) where a > 0,
    prove that sin α + 2cos α = -2/5 -/
theorem sine_cosine_sum (a : ℝ) (α : ℝ) (h1 : a > 0) 
    (h2 : Real.cos α = -3 * a / (5 * a)) (h3 : Real.sin α = 4 * a / (5 * a)) : 
    Real.sin α + 2 * Real.cos α = -2/5 := by
  sorry

end sine_cosine_sum_l2471_247128


namespace complex_number_ratio_l2471_247166

theorem complex_number_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_eq : ((x - y)^2 + (x - z)^2 + (y - z)^2) * (x + y + z) = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = (3.5 : ℂ) := by
sorry

end complex_number_ratio_l2471_247166


namespace range_of_m_l2471_247167

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 2/x + 1/y = 1) :
  (∀ x y, x > 0 → y > 0 → 2/x + 1/y = 1 → x + 2*y > m^2 + 2*m) ↔ -4 < m ∧ m < 2 := by
  sorry

end range_of_m_l2471_247167


namespace lara_flowers_theorem_l2471_247129

/-- The number of flowers Lara bought -/
def total_flowers : ℕ := sorry

/-- The number of flowers Lara gave to her mom -/
def flowers_to_mom : ℕ := 15

/-- The number of flowers Lara gave to her grandma -/
def flowers_to_grandma : ℕ := flowers_to_mom + 6

/-- The number of flowers Lara put in the vase -/
def flowers_in_vase : ℕ := 16

/-- Theorem stating the total number of flowers Lara bought -/
theorem lara_flowers_theorem : 
  total_flowers = flowers_to_mom + flowers_to_grandma + flowers_in_vase ∧ 
  total_flowers = 52 := by sorry

end lara_flowers_theorem_l2471_247129


namespace perimeter_values_finite_l2471_247191

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (AB BC CD AD : ℕ+)

-- Define the conditions for our specific quadrilateral
def ValidQuadrilateral (q : Quadrilateral) : Prop :=
  q.AB = 3 ∧ q.CD = 2 * q.AD

-- Define the perimeter
def Perimeter (q : Quadrilateral) : ℕ :=
  q.AB + q.BC + q.CD + q.AD

-- Define the right angle condition using Pythagorean theorem
def RightAngles (q : Quadrilateral) : Prop :=
  q.BC ^ 2 + (q.CD - q.AB) ^ 2 = q.AD ^ 2

-- Main theorem
theorem perimeter_values_finite :
  {p : ℕ | p < 3025 ∧ ∃ q : Quadrilateral, ValidQuadrilateral q ∧ RightAngles q ∧ Perimeter q = p}.Finite :=
sorry

end perimeter_values_finite_l2471_247191


namespace cupcakes_eaten_equals_packaged_l2471_247131

/-- Proves that the number of cupcakes Todd ate is equal to the number of cupcakes used for packaging -/
theorem cupcakes_eaten_equals_packaged (initial_cupcakes : ℕ) (num_packages : ℕ) (cupcakes_per_package : ℕ)
  (h1 : initial_cupcakes = 71)
  (h2 : num_packages = 4)
  (h3 : cupcakes_per_package = 7) :
  initial_cupcakes - (initial_cupcakes - num_packages * cupcakes_per_package) = num_packages * cupcakes_per_package :=
by sorry

end cupcakes_eaten_equals_packaged_l2471_247131


namespace car_value_reduction_l2471_247112

-- Define the original price of the car
def original_price : ℝ := 4000

-- Define the reduction rate
def reduction_rate : ℝ := 0.30

-- Define the current value of the car
def current_value : ℝ := original_price * (1 - reduction_rate)

-- Theorem to prove
theorem car_value_reduction : current_value = 2800 := by
  sorry

end car_value_reduction_l2471_247112


namespace divisibility_condition_l2471_247144

theorem divisibility_condition (n : ℕ) (h : n ≥ 2) :
  (20^n + 19^n) % (20^(n-2) + 19^(n-2)) = 0 ↔ n = 3 := by
  sorry

end divisibility_condition_l2471_247144


namespace binomial_30_3_l2471_247176

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l2471_247176


namespace movie_spending_ratio_l2471_247114

/-- Proves that the ratio of movie spending to weekly allowance is 1:2 --/
theorem movie_spending_ratio (weekly_allowance car_wash_earnings final_amount : ℕ) :
  weekly_allowance = 8 →
  car_wash_earnings = 8 →
  final_amount = 12 →
  (weekly_allowance + car_wash_earnings - final_amount) / weekly_allowance = 1 / 2 := by
  sorry

end movie_spending_ratio_l2471_247114


namespace automobile_dealer_revenue_l2471_247119

/-- Represents the revenue calculation for an automobile dealer's sale --/
theorem automobile_dealer_revenue :
  ∀ (num_suvs : ℕ),
    num_suvs + (num_suvs + 50) + (2 * num_suvs) = 150 →
    20000 * (num_suvs + 50) + 30000 * (2 * num_suvs) + 40000 * num_suvs = 4000000 :=
by
  sorry

end automobile_dealer_revenue_l2471_247119
