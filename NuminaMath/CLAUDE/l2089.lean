import Mathlib

namespace mikes_training_time_l2089_208960

/-- Proves that Mike trained for 1 hour per day during the first week -/
theorem mikes_training_time (x : ℝ) : 
  (7 * x + 7 * (x + 3) = 35) → x = 1 := by
  sorry

end mikes_training_time_l2089_208960


namespace roots_product_l2089_208916

-- Define the logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := 
  (lg x)^2 + (lg 5 + lg 7) * (lg x) + (lg 5) * (lg 7) = 0

-- Theorem statement
theorem roots_product (m n : ℝ) : 
  equation m ∧ equation n ∧ m ≠ n → m * n = 1/35 := by
  sorry

end roots_product_l2089_208916


namespace expression_one_l2089_208925

theorem expression_one : 5 * (-2)^2 - (-2)^3 / 4 = 22 := by sorry

end expression_one_l2089_208925


namespace expression_C_is_negative_l2089_208974

-- Define the variables with their approximate values
def A : ℝ := -4.2
def B : ℝ := 2.3
def C : ℝ := -0.5
def D : ℝ := 3.4
def E : ℝ := -1.8

-- Theorem stating that the expression (D/B) * C is negative
theorem expression_C_is_negative : (D / B) * C < 0 := by
  sorry

end expression_C_is_negative_l2089_208974


namespace base_10_to_base_7_l2089_208938

theorem base_10_to_base_7 : 
  ∃ (a b c d : ℕ), 
    784 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
    a = 2 ∧ b = 2 ∧ c = 0 ∧ d = 0 :=
by sorry

end base_10_to_base_7_l2089_208938


namespace certain_number_problem_l2089_208910

theorem certain_number_problem (x y : ℝ) (h1 : 0.25 * x = 0.15 * y - 15) (h2 : x = 840) : y = 1500 := by
  sorry

end certain_number_problem_l2089_208910


namespace stuffed_animals_ratio_l2089_208928

/-- Proves the ratio of Kenley's stuffed animals to McKenna's is 2:1 --/
theorem stuffed_animals_ratio :
  let mcKenna : ℕ := 34
  let total : ℕ := 175
  let kenley : ℕ := (total - mcKenna - 5) / 2
  (kenley : ℚ) / mcKenna = 2 := by sorry

end stuffed_animals_ratio_l2089_208928


namespace max_distance_ellipse_circle_l2089_208920

/-- The maximum distance between points on an ellipse and a moving circle --/
theorem max_distance_ellipse_circle (a b R : ℝ) (ha : 0 < b) (hab : b < a) (hR : b < R) (hRa : R < a) :
  let ellipse := {p : ℝ × ℝ | (p.1 / a)^2 + (p.2 / b)^2 = 1}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = R^2}
  ∃ (A B : ℝ × ℝ), A ∈ ellipse ∧ B ∈ circle ∧
    (∀ (C : ℝ × ℝ), C ∈ ellipse → (A.1 - C.1) * (B.2 - A.2) = (A.2 - C.2) * (B.1 - A.1)) ∧
    (∀ (D : ℝ × ℝ), D ∈ circle → (B.1 - D.1) * (A.2 - B.2) = (B.2 - D.2) * (A.1 - B.1)) ∧
    ∀ (A' B' : ℝ × ℝ), A' ∈ ellipse → B' ∈ circle →
      (∀ (C : ℝ × ℝ), C ∈ ellipse → (A'.1 - C.1) * (B'.2 - A'.2) = (A'.2 - C.2) * (B'.1 - A'.1)) →
      (∀ (D : ℝ × ℝ), D ∈ circle → (B'.1 - D.1) * (A'.2 - B'.2) = (B'.2 - D.2) * (A'.1 - B'.1)) →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) :=
by
  sorry

#check max_distance_ellipse_circle

end max_distance_ellipse_circle_l2089_208920


namespace irrational_plus_five_iff_irrational_l2089_208907

theorem irrational_plus_five_iff_irrational (a : ℝ) :
  Irrational a ↔ Irrational (a + 5) :=
sorry

end irrational_plus_five_iff_irrational_l2089_208907


namespace right_triangle_area_l2089_208985

theorem right_triangle_area (a b c : ℝ) (h1 : a = 24) (h2 : c = 25) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 84 := by
sorry

end right_triangle_area_l2089_208985


namespace red_apples_count_l2089_208926

/-- The number of apples produced by each tree -/
def applesPerTree : ℕ := 20

/-- The percentage of red apples on the first tree -/
def firstTreeRedPercentage : ℚ := 40 / 100

/-- The percentage of red apples on the second tree -/
def secondTreeRedPercentage : ℚ := 50 / 100

/-- The total number of red apples from both trees -/
def totalRedApples : ℕ := 18

theorem red_apples_count :
  ⌊(firstTreeRedPercentage * applesPerTree : ℚ)⌋ +
  ⌊(secondTreeRedPercentage * applesPerTree : ℚ)⌋ = totalRedApples :=
sorry

end red_apples_count_l2089_208926


namespace distance_between_4th_and_30th_red_l2089_208941

/-- Represents the color of a light -/
inductive LightColor
| Red
| Green

/-- Represents the cyclic pattern of lights -/
def lightPattern : List LightColor := 
  [LightColor.Red, LightColor.Red, LightColor.Red, 
   LightColor.Green, LightColor.Green, LightColor.Green, LightColor.Green]

/-- The distance between each light in inches -/
def lightDistance : ℕ := 8

/-- Calculates the position of the nth red light -/
def nthRedLightPosition (n : ℕ) : ℕ := sorry

/-- Calculates the distance between two positions in feet -/
def distanceInFeet (pos1 pos2 : ℕ) : ℚ := sorry

/-- Theorem: The distance between the 4th and 30th red light is 41.33 feet -/
theorem distance_between_4th_and_30th_red : 
  distanceInFeet (nthRedLightPosition 4) (nthRedLightPosition 30) = 41.33 := by sorry

end distance_between_4th_and_30th_red_l2089_208941


namespace factory_production_l2089_208984

/-- The total number of cars made by a factory over two days, given the production on the first day and that the second day's production is twice the first day's. -/
def total_cars (first_day_production : ℕ) : ℕ :=
  first_day_production + 2 * first_day_production

/-- Theorem stating that the total number of cars made over two days is 180,
    given that 60 cars were made on the first day. -/
theorem factory_production : total_cars 60 = 180 := by
  sorry

end factory_production_l2089_208984


namespace total_books_is_80_l2089_208967

/-- Calculates the total number of books bought given the conditions -/
def total_books (total_price : ℕ) (math_book_price : ℕ) (history_book_price : ℕ) (math_books_bought : ℕ) : ℕ :=
  let history_books_bought := (total_price - math_book_price * math_books_bought) / history_book_price
  math_books_bought + history_books_bought

/-- Proves that the total number of books bought is 80 under the given conditions -/
theorem total_books_is_80 :
  total_books 390 4 5 10 = 80 := by
  sorry

end total_books_is_80_l2089_208967


namespace modulus_of_z_l2089_208998

theorem modulus_of_z (z : ℂ) (h : (2 * z) / (1 - z) = Complex.I) : Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end modulus_of_z_l2089_208998


namespace f_bounded_implies_k_eq_three_l2089_208965

/-- The function f(x) = -4x³ + kx --/
def f (k : ℝ) (x : ℝ) : ℝ := -4 * x^3 + k * x

/-- The theorem stating that if f(x) ≤ 1 for all x in [-1, 1], then k = 3 --/
theorem f_bounded_implies_k_eq_three (k : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f k x ≤ 1) → k = 3 := by
  sorry

end f_bounded_implies_k_eq_three_l2089_208965


namespace three_pair_prob_standard_deck_l2089_208955

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- Represents a "three pair" hand in poker -/
structure ThreePair :=
  (triplet_rank : Nat)
  (triplet_suit : Nat)
  (pair_rank : Nat)
  (pair_suit : Nat)

/-- The number of ways to choose 5 cards from a deck -/
def choose_five (d : Deck) : Nat :=
  Nat.choose d.cards 5

/-- The number of valid "three pair" hands -/
def count_three_pairs (d : Deck) : Nat :=
  d.ranks * d.suits * (d.ranks - 1) * d.suits

/-- The probability of getting a "three pair" hand -/
def three_pair_probability (d : Deck) : ℚ :=
  count_three_pairs d / choose_five d

/-- Theorem: The probability of a "three pair" in a standard deck is 2,496 / 2,598,960 -/
theorem three_pair_prob_standard_deck :
  three_pair_probability (Deck.mk 52 13 4) = 2496 / 2598960 := by
  sorry

end three_pair_prob_standard_deck_l2089_208955


namespace smallest_prime_dividing_sum_l2089_208976

theorem smallest_prime_dividing_sum : ∃ (p : ℕ), 
  Nat.Prime p ∧ 
  p ∣ (2^11 + 7^13) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (2^11 + 7^13) → p ≤ q :=
by sorry

end smallest_prime_dividing_sum_l2089_208976


namespace unique_prime_sum_of_fourth_powers_l2089_208927

theorem unique_prime_sum_of_fourth_powers (p a b c : ℕ) : 
  Prime p ∧ Prime a ∧ Prime b ∧ Prime c ∧ p = a^4 + b^4 + c^4 - 3 → p = 719 :=
sorry

end unique_prime_sum_of_fourth_powers_l2089_208927


namespace product_pqr_l2089_208937

theorem product_pqr (p q r : ℤ) 
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h_sum : p + q + r = 26)
  (h_eq : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 360 / (p * q * r) = 1) :
  p * q * r = 576 := by
  sorry

end product_pqr_l2089_208937


namespace economics_test_absentees_l2089_208964

theorem economics_test_absentees (total_students : Nat) 
  (correct_q1 : Nat) (correct_q2 : Nat) (correct_both : Nat) :
  total_students = 30 →
  correct_q1 = 25 →
  correct_q2 = 22 →
  correct_both = 22 →
  correct_both = correct_q2 →
  total_students - correct_q2 = 8 :=
by
  sorry

end economics_test_absentees_l2089_208964


namespace golden_ratio_properties_l2089_208962

theorem golden_ratio_properties : ∃ a : ℝ, 
  (a = (Real.sqrt 5 - 1) / 2) ∧ 
  (a^2 + a - 1 = 0) ∧ 
  (a^3 - 2*a + 2015 = 2014) := by
  sorry

end golden_ratio_properties_l2089_208962


namespace used_car_selection_l2089_208986

theorem used_car_selection (num_cars : ℕ) (num_clients : ℕ) (selections_per_car : ℕ) :
  num_cars = 10 →
  num_clients = 15 →
  selections_per_car = 3 →
  (num_cars * selections_per_car) / num_clients = 2 := by
  sorry

end used_car_selection_l2089_208986


namespace solutions_equation1_solutions_equation2_l2089_208933

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := 2 * x^2 - 5 * x + 1 = 0
def equation2 (x : ℝ) : Prop := (2 * x - 1)^2 - x^2 = 0

-- Theorem for the solutions of equation1
theorem solutions_equation1 :
  ∃ x₁ x₂ : ℝ, x₁ = (5 + Real.sqrt 17) / 4 ∧
              x₂ = (5 - Real.sqrt 17) / 4 ∧
              equation1 x₁ ∧
              equation1 x₂ ∧
              ∀ x : ℝ, equation1 x → (x = x₁ ∨ x = x₂) :=
sorry

-- Theorem for the solutions of equation2
theorem solutions_equation2 :
  ∃ x₁ x₂ : ℝ, x₁ = 1/3 ∧
              x₂ = 1 ∧
              equation2 x₁ ∧
              equation2 x₂ ∧
              ∀ x : ℝ, equation2 x → (x = x₁ ∨ x = x₂) :=
sorry

end solutions_equation1_solutions_equation2_l2089_208933


namespace jelly_bean_ratio_l2089_208952

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

def total_jelly_beans : ℕ := 4000
def red_jelly_beans : ℕ := (3 * total_jelly_beans) / 4
def coconut_flavored_jelly_beans : ℕ := 750

theorem jelly_bean_ratio :
  Ratio.mk coconut_flavored_jelly_beans red_jelly_beans = Ratio.mk 1 4 := by
  sorry

end jelly_bean_ratio_l2089_208952


namespace sum_of_squares_of_roots_l2089_208909

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (10 * x₁^2 - 14 * x₁ - 24 = 0) → 
  (10 * x₂^2 - 14 * x₂ - 24 = 0) → 
  (x₁ ≠ x₂) →
  x₁^2 + x₂^2 = 169/25 := by
sorry

end sum_of_squares_of_roots_l2089_208909


namespace sum_powers_equality_l2089_208996

theorem sum_powers_equality (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  a^5 + b^5 = c^5 + d^5 ∧ 
  ∃ (a b c d : ℝ), (a + b = c + d) ∧ (a^3 + b^3 = c^3 + d^3) ∧ (a^4 + b^4 ≠ c^4 + d^4) := by
  sorry

end sum_powers_equality_l2089_208996


namespace problem_solution_l2089_208979

theorem problem_solution (a e : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * e) : e = 49 := by
  sorry

end problem_solution_l2089_208979


namespace problem_solution_l2089_208943

theorem problem_solution (a b : ℝ) : 
  let A := 2 * a^2 + 3 * a * b - 2 * a - (1/3 : ℝ)
  let B := -a^2 + (1/2 : ℝ) * a * b + (2/3 : ℝ)
  (a + 1)^2 + |b + 2| = 0 → 4 * A - (3 * A - 2 * B) = 11 :=
by sorry

end problem_solution_l2089_208943


namespace point_movement_l2089_208951

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Moves a point on the number line by a given distance -/
def movePoint (p : Point) (distance : ℝ) : Point :=
  ⟨p.value + distance⟩

theorem point_movement :
  let A : Point := ⟨-4⟩
  let B : Point := movePoint A 6
  B.value = 2 := by
  sorry

end point_movement_l2089_208951


namespace cube_root_of_neg_125_l2089_208903

theorem cube_root_of_neg_125 : ∃ x : ℝ, x^3 = -125 ∧ x = -5 := by sorry

end cube_root_of_neg_125_l2089_208903


namespace collinear_points_x_value_l2089_208987

/-- Given three collinear points A(-1, 1), B(2, -4), and C(x, -9), prove that x = 5 -/
theorem collinear_points_x_value : 
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (2, -4)
  let C : ℝ × ℝ := (x, -9)
  (∀ t : ℝ, (1 - t) * A.1 + t * B.1 = C.1 ∧ (1 - t) * A.2 + t * B.2 = C.2) →
  x = 5 := by
sorry


end collinear_points_x_value_l2089_208987


namespace solution_set_for_a_equals_one_range_of_a_l2089_208923

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |2*x - 1|

-- Theorem 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Theorem 2
theorem range_of_a :
  ∀ a : ℝ, (∀ x ∈ {x : ℝ | 1/2 ≤ x ∧ x ≤ 1}, f a x ≤ |2*x + 1|) →
  -1 ≤ a ∧ a ≤ 5/2 := by sorry

end solution_set_for_a_equals_one_range_of_a_l2089_208923


namespace markov_equation_solution_l2089_208978

theorem markov_equation_solution (m n p : ℕ) : 
  m^2 + n^2 + p^2 = m * n * p → 
  ∃ m₁ n₁ p₁ : ℕ, m = 3 * m₁ ∧ n = 3 * n₁ ∧ p = 3 * p₁ ∧ 
  m₁^2 + n₁^2 + p₁^2 = 3 * m₁ * n₁ * p₁ := by
sorry

end markov_equation_solution_l2089_208978


namespace max_piles_for_660_stones_l2089_208922

/-- Represents the stone splitting game -/
structure StoneSplittingGame where
  initial_stones : ℕ
  max_piles : ℕ

/-- Checks if a list of pile sizes is valid according to the game rules -/
def is_valid_configuration (piles : List ℕ) : Prop :=
  ∀ i j, i < piles.length → j < piles.length → 
    2 * piles[i]! > piles[j]! ∧ 2 * piles[j]! > piles[i]!

/-- Theorem stating the maximum number of piles for 660 stones -/
theorem max_piles_for_660_stones (game : StoneSplittingGame) 
  (h1 : game.initial_stones = 660)
  (h2 : game.max_piles = 30) :
  ∃ (piles : List ℕ), 
    piles.length = game.max_piles ∧ 
    piles.sum = game.initial_stones ∧
    is_valid_configuration piles ∧
    ∀ (other_piles : List ℕ), 
      other_piles.sum = game.initial_stones → 
      is_valid_configuration other_piles →
      other_piles.length ≤ game.max_piles :=
sorry


end max_piles_for_660_stones_l2089_208922


namespace eleventh_sum_14_l2089_208935

/-- Given a natural number, returns the sum of its digits -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if the given natural number has digits that sum to 14 -/
def sum_to_14 (n : ℕ) : Prop := digit_sum n = 14

/-- Returns the nth positive integer whose digits sum to 14 -/
def nth_sum_14 (n : ℕ) : ℕ := sorry

theorem eleventh_sum_14 : nth_sum_14 11 = 149 := by sorry

end eleventh_sum_14_l2089_208935


namespace melissa_bonus_points_l2089_208942

/-- Given a player's regular points per game, number of games played, and total score,
    calculate the bonus points per game. -/
def bonusPointsPerGame (regularPointsPerGame : ℕ) (numGames : ℕ) (totalScore : ℕ) : ℕ :=
  ((totalScore - regularPointsPerGame * numGames) / numGames : ℕ)

/-- Theorem stating that for the given conditions, the bonus points per game is 82. -/
theorem melissa_bonus_points :
  bonusPointsPerGame 109 79 15089 = 82 := by
  sorry

#eval bonusPointsPerGame 109 79 15089

end melissa_bonus_points_l2089_208942


namespace system_solution_transformation_l2089_208988

theorem system_solution_transformation (x y : ℝ) : 
  (2 * x + 3 * y = 19 ∧ 3 * x + 4 * y = 26) → 
  (2 * (2 * x + 4) + 3 * (y + 3) = 19 ∧ 3 * (2 * x + 4) + 4 * (y + 3) = 26) → 
  (x = 2 ∧ y = 5) → 
  (x = -1 ∧ y = 2) := by
sorry

end system_solution_transformation_l2089_208988


namespace sum_remainder_nine_l2089_208902

theorem sum_remainder_nine (n : ℤ) : ((9 - n) + (n + 4)) % 9 = 4 := by
  sorry

end sum_remainder_nine_l2089_208902


namespace ones_digit_of_34_power_power_4_cycle_seventeen_power_seventeen_odd_main_theorem_l2089_208917

theorem ones_digit_of_34_power (n : ℕ) : n > 0 → (34^n) % 10 = (4^n) % 10 := by sorry

theorem power_4_cycle (n : ℕ) : (4^n) % 10 = if n % 2 = 0 then 6 else 4 := by sorry

theorem seventeen_power_seventeen_odd : 17^17 % 2 = 1 := by sorry

theorem main_theorem : (34^(34*(17^17))) % 10 = 4 := by sorry

end ones_digit_of_34_power_power_4_cycle_seventeen_power_seventeen_odd_main_theorem_l2089_208917


namespace line_parallel_perpendicular_l2089_208991

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular 
  (a b : Line) (α : Plane) :
  parallel a b → perpendicular a α → perpendicular b α :=
by sorry

end line_parallel_perpendicular_l2089_208991


namespace milk_delivery_proof_l2089_208999

/-- The amount of milk in liters delivered to Minjeong's house in a week -/
def milk_per_week (bottles_per_day : ℕ) (liters_per_bottle : ℚ) (days_in_week : ℕ) : ℚ :=
  (bottles_per_day : ℚ) * liters_per_bottle * (days_in_week : ℚ)

/-- Proof that 4.2 liters of milk are delivered to Minjeong's house in a week -/
theorem milk_delivery_proof :
  milk_per_week 3 (2/10) 7 = 21/5 := by
  sorry

end milk_delivery_proof_l2089_208999


namespace quadratic_equation_solution_l2089_208908

theorem quadratic_equation_solution (x : ℝ) (h1 : x^2 - 4*x = 0) (h2 : x ≠ 0) : x = 4 := by
  sorry

end quadratic_equation_solution_l2089_208908


namespace xy_value_l2089_208972

theorem xy_value (x y : ℝ) (h : |x + 2*y| + (y - 3)^2 = 0) : x^y = -216 := by
  sorry

end xy_value_l2089_208972


namespace arithmetic_sequence_proof_l2089_208945

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n - 1

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (a 1) + n * (n - 1)

-- Define b_n
def b (n : ℕ) : ℚ := (-1)^(n-1) * (4 * n) / (a n * a (n+1))

-- Define T_n (sum of first n terms of b_n)
def T (n : ℕ) : ℚ :=
  if n % 2 = 0 then (2 * n) / (2 * n + 1)
  else (2 * n + 2) / (2 * n + 1)

-- Theorem statement
theorem arithmetic_sequence_proof :
  (∀ n : ℕ, S (n+1) - S n = 2) ∧  -- Common difference is 2
  (S 2)^2 = S 1 * S 4 ∧           -- S_1, S_2, S_4 form a geometric sequence
  (∀ n : ℕ, a n = 2 * n - 1) ∧    -- General formula for a_n
  (∀ n : ℕ, T n = if n % 2 = 0 then (2 * n) / (2 * n + 1) else (2 * n + 2) / (2 * n + 1)) :=
by sorry

end arithmetic_sequence_proof_l2089_208945


namespace train_capacity_l2089_208944

/-- Proves that given a train with 4 carriages, each initially having 25 seats
    and can accommodate 10 more passengers, the total number of passengers
    that would fill up 3 such trains is 420. -/
theorem train_capacity (initial_seats : Nat) (additional_seats : Nat) 
  (carriages_per_train : Nat) (number_of_trains : Nat) :
  initial_seats = 25 →
  additional_seats = 10 →
  carriages_per_train = 4 →
  number_of_trains = 3 →
  (initial_seats + additional_seats) * carriages_per_train * number_of_trains = 420 := by
  sorry

#eval (25 + 10) * 4 * 3  -- Should output 420

end train_capacity_l2089_208944


namespace ratio_subtraction_l2089_208948

theorem ratio_subtraction (a b : ℚ) (h : a / b = 4 / 7) :
  (a - b) / b = -3 / 7 := by sorry

end ratio_subtraction_l2089_208948


namespace vincent_earnings_l2089_208977

/-- Represents Vincent's bookstore earnings over a period of days -/
def bookstore_earnings (fantasy_price : ℕ) (fantasy_sold : ℕ) (literature_sold : ℕ) (days : ℕ) : ℕ :=
  let literature_price := fantasy_price / 2
  let daily_earnings := fantasy_price * fantasy_sold + literature_price * literature_sold
  daily_earnings * days

/-- Theorem stating that Vincent's earnings after 5 days will be $180 -/
theorem vincent_earnings : bookstore_earnings 4 5 8 5 = 180 := by
  sorry

end vincent_earnings_l2089_208977


namespace sum_of_digits_n_l2089_208973

/-- The least 6-digit number that leaves a remainder of 2 when divided by 4, 610, and 15 -/
def n : ℕ := 102482

/-- Condition: n is at least 100000 (6-digit number) -/
axiom n_six_digits : n ≥ 100000

/-- Condition: n leaves remainder 2 when divided by 4 -/
axiom n_mod_4 : n % 4 = 2

/-- Condition: n leaves remainder 2 when divided by 610 -/
axiom n_mod_610 : n % 610 = 2

/-- Condition: n leaves remainder 2 when divided by 15 -/
axiom n_mod_15 : n % 15 = 2

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ :=
  if m = 0 then 0 else m % 10 + sum_of_digits (m / 10)

/-- Theorem: The sum of digits of n is 17 -/
theorem sum_of_digits_n : sum_of_digits n = 17 := by
  sorry

end sum_of_digits_n_l2089_208973


namespace largest_integer_with_remainder_l2089_208924

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 6 = 4 ∧ ∀ m : ℕ, m < 100 → m % 6 = 4 → m ≤ n :=
by sorry

end largest_integer_with_remainder_l2089_208924


namespace initial_journey_speed_l2089_208947

/-- Proves that the speed of the initial journey is 63 mph given the conditions -/
theorem initial_journey_speed (d : ℝ) (v : ℝ) (h1 : v > 0) : 
  (2 * d) / (d / v + 2 * (d / v)) = 42 → v = 63 := by
  sorry

end initial_journey_speed_l2089_208947


namespace walking_speed_calculation_l2089_208959

theorem walking_speed_calculation (total_distance : ℝ) (running_speed : ℝ) (total_time : ℝ) 
  (h1 : total_distance = 4)
  (h2 : running_speed = 8)
  (h3 : total_time = 0.75)
  (h4 : ∃ (walking_time running_time : ℝ), 
    walking_time + running_time = total_time ∧ 
    walking_time * walking_speed = running_time * running_speed ∧
    walking_time * walking_speed = total_distance / 2) :
  walking_speed = 4 := by
  sorry

end walking_speed_calculation_l2089_208959


namespace base4_to_decimal_conversion_l2089_208989

/-- Converts a base-4 digit to its decimal value -/
def base4ToDecimal (digit : Nat) : Nat :=
  match digit with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | _ => 0  -- Default case, should not occur in valid input

/-- Converts a list of base-4 digits to its decimal representation -/
def base4ListToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + (base4ToDecimal d) * (4 ^ (digits.length - 1 - i))) 0

theorem base4_to_decimal_conversion :
  base4ListToDecimal [0, 1, 3, 2, 0, 1, 3, 2] = 7710 := by
  sorry

#eval base4ListToDecimal [0, 1, 3, 2, 0, 1, 3, 2]

end base4_to_decimal_conversion_l2089_208989


namespace rice_bags_sold_l2089_208921

/-- A trader sells rice bags and restocks. This theorem proves the number of bags sold. -/
theorem rice_bags_sold (initial_stock : ℕ) (restocked : ℕ) (final_stock : ℕ) 
  (h1 : initial_stock = 55)
  (h2 : restocked = 132)
  (h3 : final_stock = 164) :
  initial_stock + restocked - final_stock = 23 := by
  sorry

end rice_bags_sold_l2089_208921


namespace max_cherries_proof_l2089_208971

/-- Represents the number of fruits Alice can buy -/
structure FruitPurchase where
  apples : ℕ
  bananas : ℕ
  cherries : ℕ

/-- Checks if a purchase satisfies all conditions -/
def isValidPurchase (p : FruitPurchase) : Prop :=
  p.apples ≥ 1 ∧ p.bananas ≥ 1 ∧ p.cherries ≥ 1 ∧
  2 * p.apples + 5 * p.bananas + 10 * p.cherries = 100

/-- The maximum number of cherries Alice can purchase -/
def maxCherries : ℕ := 8

theorem max_cherries_proof :
  (∃ p : FruitPurchase, isValidPurchase p ∧ p.cherries = maxCherries) ∧
  (∀ p : FruitPurchase, isValidPurchase p → p.cherries ≤ maxCherries) :=
sorry

end max_cherries_proof_l2089_208971


namespace largest_multiple_under_500_l2089_208956

theorem largest_multiple_under_500 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 :=
by sorry

end largest_multiple_under_500_l2089_208956


namespace debate_team_groups_l2089_208990

/-- The number of boys on the debate team -/
def num_boys : ℕ := 28

/-- The number of girls on the debate team -/
def num_girls : ℕ := 4

/-- The minimum number of boys required in each group -/
def min_boys_per_group : ℕ := 2

/-- The minimum number of girls required in each group -/
def min_girls_per_group : ℕ := 1

/-- The maximum number of groups that can be formed -/
def max_groups : ℕ := 4

theorem debate_team_groups :
  (num_girls ≥ max_groups * min_girls_per_group) ∧
  (num_boys ≥ max_groups * min_boys_per_group) ∧
  (∀ n : ℕ, n > max_groups → 
    (num_girls < n * min_girls_per_group) ∨ 
    (num_boys < n * min_boys_per_group)) :=
sorry

end debate_team_groups_l2089_208990


namespace slope_of_line_l2089_208969

-- Define a line with equation y = 3x + 1
def line (x : ℝ) : ℝ := 3 * x + 1

-- State the theorem: the slope of this line is 3
theorem slope_of_line :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (line x₂ - line x₁) / (x₂ - x₁) = 3) := by
sorry

end slope_of_line_l2089_208969


namespace polynomial_root_comparison_l2089_208936

theorem polynomial_root_comparison (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h1 : a₁ ≤ a₂) (h2 : a₂ ≤ a₃) 
  (h3 : b₁ ≤ b₂) (h4 : b₂ ≤ b₃) 
  (h5 : a₁ + a₂ + a₃ = b₁ + b₂ + b₃) 
  (h6 : a₁*a₂ + a₂*a₃ + a₁*a₃ = b₁*b₂ + b₂*b₃ + b₁*b₃) 
  (h7 : a₁ ≤ b₁) : 
  a₃ ≤ b₃ := by
sorry

end polynomial_root_comparison_l2089_208936


namespace calculate_expression_l2089_208957

theorem calculate_expression : (-1 : ℝ)^200 - (-1/2 : ℝ)^0 + (3⁻¹ : ℝ) * 6 = 2 := by
  sorry

end calculate_expression_l2089_208957


namespace factors_of_1320_l2089_208953

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def count_factors (factorization : List (ℕ × ℕ)) : ℕ := sorry

theorem factors_of_1320 :
  let factorization := prime_factorization 1320
  count_factors factorization = 24 := by sorry

end factors_of_1320_l2089_208953


namespace minimum_guests_l2089_208954

theorem minimum_guests (total_food : ℕ) (max_per_guest : ℕ) (h1 : total_food = 327) (h2 : max_per_guest = 2) :
  ∃ (min_guests : ℕ), min_guests = 164 ∧ min_guests * max_per_guest ≥ total_food ∧ (min_guests - 1) * max_per_guest < total_food :=
by
  sorry

end minimum_guests_l2089_208954


namespace triangle_abc_properties_l2089_208915

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
prove the values of a, b, and the area of the triangle.
-/
theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  c = 3 →
  C = π / 3 →
  Real.sin B = 2 * Real.sin A →
  (a = Real.sqrt 3 ∧ b = 2 * Real.sqrt 3) ∧
  (1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2) := by
  sorry

end triangle_abc_properties_l2089_208915


namespace quadratic_range_l2089_208913

/-- The quadratic function f(x) = x^2 - 2x - 3 -/
def f (x : ℝ) := x^2 - 2*x - 3

/-- The theorem states that for x in [-2, 2], the range of f(x) is [-4, 5] -/
theorem quadratic_range :
  ∀ x ∈ Set.Icc (-2 : ℝ) 2,
  ∃ y ∈ Set.Icc (-4 : ℝ) 5,
  f x = y ∧
  (∀ z, f z ∈ Set.Icc (-4 : ℝ) 5) :=
sorry

end quadratic_range_l2089_208913


namespace andrews_dog_foreign_objects_l2089_208939

/-- Calculates the total number of foreign objects on a dog given the number of burrs,
    the ratio of ticks to burrs, and the ratio of fleas to ticks. -/
def total_foreign_objects (burrs : ℕ) (ticks_to_burrs_ratio : ℕ) (fleas_to_ticks_ratio : ℕ) : ℕ :=
  let ticks := burrs * ticks_to_burrs_ratio
  let fleas := ticks * fleas_to_ticks_ratio
  burrs + ticks + fleas

/-- Theorem stating that for a dog with 12 burrs, 6 times as many ticks as burrs,
    and 3 times as many fleas as ticks, the total number of foreign objects is 300. -/
theorem andrews_dog_foreign_objects : 
  total_foreign_objects 12 6 3 = 300 := by
  sorry

end andrews_dog_foreign_objects_l2089_208939


namespace prob_sum_gt_8_is_correct_l2089_208930

/-- The probability of getting a sum greater than 8 when tossing two dice -/
def prob_sum_gt_8 : ℚ :=
  5 / 18

/-- The total number of possible outcomes when tossing two dice -/
def total_outcomes : ℕ := 36

/-- The number of ways to get a sum of 8 or less when tossing two dice -/
def ways_sum_le_8 : ℕ := 26

/-- Theorem: The probability of getting a sum greater than 8 when tossing two dice is 5/18 -/
theorem prob_sum_gt_8_is_correct : prob_sum_gt_8 = 1 - (ways_sum_le_8 : ℚ) / total_outcomes :=
by sorry

end prob_sum_gt_8_is_correct_l2089_208930


namespace x_fourth_plus_inverse_x_fourth_l2089_208919

theorem x_fourth_plus_inverse_x_fourth (x : ℝ) (h : x^2 + 1/x^2 = 2) : x^4 + 1/x^4 = 2 := by
  sorry

end x_fourth_plus_inverse_x_fourth_l2089_208919


namespace last_locker_opened_l2089_208963

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Represents the process of toggling lockers -/
def toggle_lockers (n : ℕ) (k : ℕ) : List LockerState → List LockerState :=
  sorry

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- Finds the largest perfect square less than or equal to a given number -/
def largest_perfect_square_le (n : ℕ) : ℕ :=
  sorry

theorem last_locker_opened (num_lockers : ℕ) (num_lockers_eq : num_lockers = 500) :
  largest_perfect_square_le num_lockers = 484 :=
sorry

end last_locker_opened_l2089_208963


namespace monic_quadratic_with_complex_root_l2089_208981

theorem monic_quadratic_with_complex_root :
  ∃ (a b : ℝ), ∀ (x : ℂ),
    (x^2 + a*x + b = 0 ↔ x = -3 - Complex.I * Real.sqrt 7 ∨ x = -3 + Complex.I * Real.sqrt 7) ∧
    (a = 6 ∧ b = 16) := by
  sorry

end monic_quadratic_with_complex_root_l2089_208981


namespace fly_distance_from_ceiling_l2089_208905

theorem fly_distance_from_ceiling (x y z : ℝ) :
  x = 2 ∧ y = 6 ∧ Real.sqrt (x^2 + y^2 + z^2) = 10 → z = 2 * Real.sqrt 15 := by
  sorry

end fly_distance_from_ceiling_l2089_208905


namespace cubic_expression_value_l2089_208906

theorem cubic_expression_value (m : ℝ) (h : m^2 - m - 1 = 0) : 
  2*m^3 - 3*m^2 - m + 9 = 8 := by
  sorry

end cubic_expression_value_l2089_208906


namespace arithmetic_mean_property_l2089_208946

def number_set : List Nat := [9, 9999, 99999999, 999999999999, 9999999999999999, 99999999999999999999]

def arithmetic_mean (xs : List Nat) : Nat :=
  xs.sum / xs.length

def has_18_digits (n : Nat) : Prop :=
  n ≥ 10^17 ∧ n < 10^18

def all_digits_distinct (n : Nat) : Prop :=
  ∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10

def does_not_contain_4 (n : Nat) : Prop :=
  ∀ i, (n / 10^i) % 10 ≠ 4

theorem arithmetic_mean_property :
  let mean := arithmetic_mean number_set
  has_18_digits mean ∧ all_digits_distinct mean ∧ does_not_contain_4 mean :=
sorry

end arithmetic_mean_property_l2089_208946


namespace quadratic_equations_solutions_l2089_208911

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2/3 ∧ x₂ = 2 ∧ 3*x₁^2 - 8*x₁ + 4 = 0 ∧ 3*x₂^2 - 8*x₂ + 4 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 4/3 ∧ y₂ = -2 ∧ (2*y₁ - 1)^2 = (y₁ - 3)^2 ∧ (2*y₂ - 1)^2 = (y₂ - 3)^2) :=
by sorry

end quadratic_equations_solutions_l2089_208911


namespace geometric_sequence_value_l2089_208966

theorem geometric_sequence_value (a : ℝ) : 
  (∃ (r : ℝ), 1 * r = a ∧ a * r = (1/16 : ℝ)) → 
  (a = (1/4 : ℝ) ∨ a = -(1/4 : ℝ)) := by
  sorry

end geometric_sequence_value_l2089_208966


namespace laptop_savings_weeks_l2089_208975

theorem laptop_savings_weeks (laptop_cost : ℕ) (birthday_money : ℕ) (weekly_earnings : ℕ) 
  (h1 : laptop_cost = 800)
  (h2 : birthday_money = 140)
  (h3 : weekly_earnings = 20) :
  ∃ (weeks : ℕ), birthday_money + weekly_earnings * weeks = laptop_cost ∧ weeks = 33 := by
  sorry

end laptop_savings_weeks_l2089_208975


namespace inequality_proof_l2089_208929

theorem inequality_proof (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 - x) / (y^2 + y) + (y^2 - y) / (x^2 + x) > 2/3 := by
  sorry

end inequality_proof_l2089_208929


namespace two_cyclists_problem_l2089_208968

/-- Two cyclists problem -/
theorem two_cyclists_problem (north_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  north_speed = 30 →
  time = 0.7142857142857143 →
  distance = 50 →
  ∃ (south_speed : ℝ), south_speed = 40 ∧ distance = (north_speed + south_speed) * time :=
by sorry

end two_cyclists_problem_l2089_208968


namespace greatest_abcba_div_by_11_and_3_l2089_208994

def is_abcba (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), 
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

theorem greatest_abcba_div_by_11_and_3 : 
  (∀ n : ℕ, is_abcba n → n % 11 = 0 → n % 3 = 0 → n ≤ 96569) ∧ 
  is_abcba 96569 ∧ 
  96569 % 11 = 0 ∧ 
  96569 % 3 = 0 :=
sorry

end greatest_abcba_div_by_11_and_3_l2089_208994


namespace equation_solution_l2089_208992

theorem equation_solution (y : ℝ) (h : y ≠ 0) : 
  (2 / y + (3 / y) / (6 / y) = 1.5) → y = 2 := by
sorry

end equation_solution_l2089_208992


namespace opposite_reciprocal_expression_value_l2089_208931

theorem opposite_reciprocal_expression_value :
  ∀ (a b c : ℤ) (m n : ℚ),
    a = -b →                          -- a and b are opposite numbers
    c = -1 →                          -- c is the smallest negative integer in absolute value
    m * n = 1 →                       -- m and n are reciprocal numbers
    (a + b) / 3 + c^2 - 4 * m * n = -3 :=
by
  sorry

end opposite_reciprocal_expression_value_l2089_208931


namespace unequal_grandchildren_probability_l2089_208914

/-- The number of grandchildren -/
def n : ℕ := 12

/-- The probability of a child being male or female -/
def p : ℚ := 1/2

/-- The probability of having an unequal number of grandsons and granddaughters -/
def unequal_probability : ℚ := 793/1024

theorem unequal_grandchildren_probability :
  (1 : ℚ) - (n.choose (n/2) : ℚ) / (2^n : ℚ) = unequal_probability :=
sorry

end unequal_grandchildren_probability_l2089_208914


namespace real_part_of_i_times_one_plus_i_l2089_208982

theorem real_part_of_i_times_one_plus_i (i : ℂ) :
  i * i = -1 →
  Complex.re (i * (1 + i)) = -1 :=
by sorry

end real_part_of_i_times_one_plus_i_l2089_208982


namespace gcd_of_45_75_105_l2089_208918

theorem gcd_of_45_75_105 : Nat.gcd 45 (Nat.gcd 75 105) = 15 := by sorry

end gcd_of_45_75_105_l2089_208918


namespace geometric_sequence_common_ratio_l2089_208940

/-- Given a geometric sequence {a_n} with common ratio q and S_n as the sum of its first n terms,
    if S_5, S_4, and S_6 form an arithmetic sequence, then q = -2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- {a_n} is a geometric sequence with common ratio q
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- S_n is the sum of first n terms
  2 * S 4 = S 5 + S 6 →  -- S_5, S_4, and S_6 form an arithmetic sequence
  q = -2 :=
by sorry

end geometric_sequence_common_ratio_l2089_208940


namespace cos_supplementary_angles_l2089_208934

theorem cos_supplementary_angles (α β : Real) (h : α + β = Real.pi) : 
  Real.cos α = Real.cos β := by
  sorry

end cos_supplementary_angles_l2089_208934


namespace classmates_not_invited_l2089_208983

/-- A simple graph representing friendships among classmates -/
structure FriendshipGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  symm : ∀ {a b}, (a, b) ∈ edges → (b, a) ∈ edges
  irrefl : ∀ a, (a, a) ∉ edges

/-- The set of vertices reachable within n steps from a given vertex -/
def reachableWithin (G : FriendshipGraph) (start : Nat) (n : Nat) : Finset Nat :=
  sorry

/-- The main theorem -/
theorem classmates_not_invited (G : FriendshipGraph) (mark : Nat) : 
  G.vertices.card = 25 →
  mark ∈ G.vertices →
  (G.vertices \ reachableWithin G mark 3).card = 5 := by
  sorry

end classmates_not_invited_l2089_208983


namespace smallest_n_square_and_cube_l2089_208997

theorem smallest_n_square_and_cube : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (j : ℕ), 5 * n = j^3) ∧
  (∀ (m : ℕ), m > 0 → 
    ((∃ (k : ℕ), 4 * m = k^2) ∧ (∃ (j : ℕ), 5 * m = j^3)) → 
    m ≥ 500) :=
by sorry

end smallest_n_square_and_cube_l2089_208997


namespace ratio_of_45_to_9_l2089_208961

theorem ratio_of_45_to_9 (certain_number : ℕ) (h : certain_number = 45) : 
  certain_number / 9 = 5 := by
  sorry

end ratio_of_45_to_9_l2089_208961


namespace total_money_is_900_l2089_208912

/-- The amount of money Sam has -/
def sam_money : ℕ := 200

/-- The amount of money Billy has -/
def billy_money : ℕ := 3 * sam_money - 150

/-- The amount of money Lila has -/
def lila_money : ℕ := billy_money - sam_money

/-- The total amount of money they have together -/
def total_money : ℕ := sam_money + billy_money + lila_money

theorem total_money_is_900 : total_money = 900 := by
  sorry

end total_money_is_900_l2089_208912


namespace smallest_seven_digit_binary_l2089_208950

theorem smallest_seven_digit_binary : ∀ n : ℕ, n > 0 → (
  (Nat.log 2 n + 1 = 7) ↔ n ≥ 64 ∧ ∀ m : ℕ, m > 0 ∧ m < 64 → Nat.log 2 m + 1 < 7
) := by sorry

end smallest_seven_digit_binary_l2089_208950


namespace total_miles_four_weeks_eq_272_l2089_208932

/-- Calculates the total miles Vins rides in a four-week period -/
def total_miles_four_weeks : ℕ :=
  let library_distance : ℕ := 6
  let school_distance : ℕ := 5
  let friend_distance : ℕ := 8
  let extra_return_distance : ℕ := 1
  let friend_shortcut : ℕ := 2
  let library_days_per_week : ℕ := 3
  let school_days_per_week : ℕ := 2
  let friend_visits_per_four_weeks : ℕ := 2
  let weeks : ℕ := 4

  let library_miles_per_week := (library_distance + library_distance + extra_return_distance) * library_days_per_week
  let school_miles_per_week := (school_distance + school_distance + extra_return_distance) * school_days_per_week
  let friend_miles_per_four_weeks := (friend_distance + friend_distance - friend_shortcut) * friend_visits_per_four_weeks

  (library_miles_per_week + school_miles_per_week) * weeks + friend_miles_per_four_weeks

theorem total_miles_four_weeks_eq_272 : total_miles_four_weeks = 272 := by
  sorry

end total_miles_four_weeks_eq_272_l2089_208932


namespace chord_equation_l2089_208995

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents an ellipse -/
structure Ellipse :=
  (a : ℝ)
  (b : ℝ)

/-- Checks if a point is inside an ellipse -/
def isInside (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) < 1

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Checks if a point bisects a chord of an ellipse -/
def bisectsChord (p : Point) (e : Ellipse) : Prop :=
  sorry  -- Definition of bisecting a chord

theorem chord_equation (e : Ellipse) (m : Point) :
  e.a = 4 →
  e.b = 2 →
  m.x = 2 →
  m.y = 1 →
  isInside m e →
  bisectsChord m e →
  ∃ l : Line, l.a = 1 ∧ l.b = 2 ∧ l.c = -4 :=
sorry

end chord_equation_l2089_208995


namespace unique_solution_when_a_is_three_fourths_l2089_208980

/-- The equation has exactly one solution when a = 3/4 -/
theorem unique_solution_when_a_is_three_fourths (x a : ℝ) :
  (∃! x, (x^2 - a)^2 + 2*(x^2 - a) + (x - a) + 2 = 0) ↔ a = 3/4 := by
  sorry

end unique_solution_when_a_is_three_fourths_l2089_208980


namespace sum_of_possible_sums_l2089_208970

theorem sum_of_possible_sums (n : ℕ) (h : n = 9) : 
  (n * (n * (n + 1) / 2) - (n * (n + 1) / 2)) = 360 := by
  sorry

#check sum_of_possible_sums

end sum_of_possible_sums_l2089_208970


namespace ceva_theorem_l2089_208900

/-- Represents a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (X Y Z : Point)

/-- Represents a line segment -/
structure LineSegment :=
  (A B : Point)

/-- Represents the intersection point of three lines -/
def intersectionPoint (l1 l2 l3 : LineSegment) : Point := sorry

/-- Calculates the ratio of distances from a point to two other points -/
def distanceRatio (P A B : Point) : ℝ := sorry

theorem ceva_theorem (T : Triangle) (X' Y' Z' : Point) (P : Point) :
  let XX' := LineSegment.mk T.X X'
  let YY' := LineSegment.mk T.Y Y'
  let ZZ' := LineSegment.mk T.Z Z'
  P = intersectionPoint XX' YY' ZZ' →
  (distanceRatio P T.X X' + distanceRatio P T.Y Y' + distanceRatio P T.Z Z' = 100) →
  (distanceRatio P T.X X' * distanceRatio P T.Y Y' * distanceRatio P T.Z Z' = 102) := by
  sorry

end ceva_theorem_l2089_208900


namespace crocodile_count_l2089_208958

theorem crocodile_count (total : ℕ) (alligators : ℕ) (vipers : ℕ) 
  (h1 : total = 50)
  (h2 : alligators = 23)
  (h3 : vipers = 5)
  (h4 : ∃ crocodiles : ℕ, total = crocodiles + alligators + vipers) :
  ∃ crocodiles : ℕ, crocodiles = 22 ∧ total = crocodiles + alligators + vipers :=
by sorry

end crocodile_count_l2089_208958


namespace interest_rate_calculation_l2089_208901

-- Define the simple interest calculation function
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

-- State the theorem
theorem interest_rate_calculation (principal time interest : ℚ) 
  (h1 : principal = 8925)
  (h2 : time = 5)
  (h3 : interest = 4016.25)
  (h4 : simple_interest principal (9 : ℚ) time = interest) :
  ∃ (rate : ℚ), simple_interest principal rate time = interest ∧ rate = 9 := by
  sorry


end interest_rate_calculation_l2089_208901


namespace quadratic_inequality_empty_solution_range_l2089_208949

theorem quadratic_inequality_empty_solution_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 3 * x + a < 0) ↔ a < -3/2 := by sorry

end quadratic_inequality_empty_solution_range_l2089_208949


namespace regression_line_intercept_l2089_208993

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- The regression line passes through a given point -/
def passes_through (line : RegressionLine) (x y : ℝ) : Prop :=
  y = line.slope * x + line.intercept

theorem regression_line_intercept (b : ℝ) (x₀ y₀ : ℝ) :
  let line := RegressionLine.mk b ((y₀ : ℝ) - b * x₀)
  passes_through line x₀ y₀ ∧ line.slope = 1.23 ∧ x₀ = 4 ∧ y₀ = 5 →
  line.intercept = 0.08 := by
  sorry

end regression_line_intercept_l2089_208993


namespace unique_number_property_l2089_208904

theorem unique_number_property : ∃! x : ℝ, x / 2 = x - 2 := by sorry

end unique_number_property_l2089_208904
