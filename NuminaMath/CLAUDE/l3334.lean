import Mathlib

namespace NUMINAMATH_CALUDE_partner_A_share_l3334_333492

/-- Represents a partner's investment in a partnership --/
structure Investment where
  capital_ratio : ℚ
  time_ratio : ℚ

/-- Calculates the share of profit for a given investment --/
def calculate_share (inv : Investment) (total_capital_time : ℚ) (total_profit : ℚ) : ℚ :=
  (inv.capital_ratio * inv.time_ratio) / total_capital_time * total_profit

/-- Theorem stating that partner A's share of the profit is 100 --/
theorem partner_A_share :
  let a := Investment.mk (1/6) (1/6)
  let b := Investment.mk (1/3) (1/3)
  let c := Investment.mk (1/2) 1
  let total_capital_time := (1/6 * 1/6) + (1/3 * 1/3) + (1/2 * 1)
  let total_profit := 2300
  calculate_share a total_capital_time total_profit = 100 := by
  sorry

end NUMINAMATH_CALUDE_partner_A_share_l3334_333492


namespace NUMINAMATH_CALUDE_batsman_average_is_37_l3334_333490

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  averageIncrease : ℕ
  lastInningScore : ℕ

/-- Calculates the new average score after the latest inning -/
def newAverage (b : Batsman) : ℚ :=
  (b.totalRuns + b.lastInningScore) / b.innings

/-- Theorem: Given the conditions, prove that the new average is 37 -/
theorem batsman_average_is_37 (b : Batsman)
    (h1 : b.innings = 17)
    (h2 : b.lastInningScore = 85)
    (h3 : b.averageIncrease = 3)
    (h4 : newAverage b = (b.totalRuns / (b.innings - 1) + b.averageIncrease)) :
    newAverage b = 37 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_is_37_l3334_333490


namespace NUMINAMATH_CALUDE_shop_prices_existence_l3334_333479

theorem shop_prices_existence (S : ℕ) (h : S ≥ 100) :
  ∃ (a b c P : ℕ), 
    a > b ∧ b > c ∧ 
    a + b + c = S ∧
    a * b * c = P ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    ∃ (a' b' c' : ℕ), (a' ≠ a ∨ b' ≠ b ∨ c' ≠ c) ∧
      a' > b' ∧ b' > c' ∧
      a' + b' + c' = S ∧
      a' * b' * c' = P ∧
      a' > 0 ∧ b' > 0 ∧ c' > 0 :=
by sorry

end NUMINAMATH_CALUDE_shop_prices_existence_l3334_333479


namespace NUMINAMATH_CALUDE_marcos_strawberry_weight_l3334_333467

/-- Marco and his dad went strawberry picking. This theorem proves the weight of Marco's strawberries. -/
theorem marcos_strawberry_weight
  (total_weight : ℕ)
  (dads_weight : ℕ)
  (h1 : total_weight = 40)
  (h2 : dads_weight = 32)
  : total_weight - dads_weight = 8 := by
  sorry

end NUMINAMATH_CALUDE_marcos_strawberry_weight_l3334_333467


namespace NUMINAMATH_CALUDE_large_square_area_l3334_333445

theorem large_square_area (s : ℝ) (h1 : s > 0) (h2 : 2 * s^2 = 14) : (3 * s)^2 = 63 := by
  sorry

end NUMINAMATH_CALUDE_large_square_area_l3334_333445


namespace NUMINAMATH_CALUDE_exists_non_unique_f_l3334_333439

theorem exists_non_unique_f : ∃ (f : ℕ → ℕ), 
  (∀ n : ℕ, f (f n) = 4 * n + 9) ∧ 
  (∀ k : ℕ, f (2^(k-1)) = 2^k + 3) ∧ 
  (∃ n : ℕ, f n ≠ 2 * n + 3) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_unique_f_l3334_333439


namespace NUMINAMATH_CALUDE_kayak_production_sum_l3334_333437

theorem kayak_production_sum (a : ℕ) (r : ℕ) (n : ℕ) : 
  a = 9 → r = 3 → n = 5 → a * (r^n - 1) / (r - 1) = 1089 := by
  sorry

end NUMINAMATH_CALUDE_kayak_production_sum_l3334_333437


namespace NUMINAMATH_CALUDE_hawk_crow_percentage_l3334_333488

theorem hawk_crow_percentage (num_crows : ℕ) (total_birds : ℕ) (percentage : ℚ) : 
  num_crows = 30 →
  total_birds = 78 →
  total_birds = num_crows + (num_crows * (1 + percentage / 100)) →
  percentage = 60 := by
sorry

end NUMINAMATH_CALUDE_hawk_crow_percentage_l3334_333488


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l3334_333441

/-- Proves that given two rectangles with equal area, where one rectangle measures 8 inches by 15 inches
    and the other is 30 inches wide, the length of the second rectangle is 4 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_width : ℝ) 
    (h1 : carol_length = 8)
    (h2 : carol_width = 15)
    (h3 : jordan_width = 30)
    (h4 : carol_length * carol_width = jordan_width * jordan_length) :
    jordan_length = 4 :=
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l3334_333441


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3334_333457

def P : Set (ℝ × ℝ) := {p | p.1 + p.2 = 3}
def Q : Set (ℝ × ℝ) := {q | q.1 - q.2 = 5}

theorem intersection_of_P_and_Q :
  P ∩ Q = {(4, -1)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3334_333457


namespace NUMINAMATH_CALUDE_roxanne_change_l3334_333404

/-- Calculates the change Roxanne should receive after buying lemonade and sandwiches -/
theorem roxanne_change (lemonade_price : ℝ) (sandwich_price : ℝ) (lemonade_quantity : ℕ) (sandwich_quantity : ℕ) (paid_amount : ℝ) : 
  lemonade_price = 2 →
  sandwich_price = 2.5 →
  lemonade_quantity = 2 →
  sandwich_quantity = 2 →
  paid_amount = 20 →
  paid_amount - (lemonade_price * lemonade_quantity + sandwich_price * sandwich_quantity) = 11 := by
sorry

end NUMINAMATH_CALUDE_roxanne_change_l3334_333404


namespace NUMINAMATH_CALUDE_no_solution_exists_l3334_333463

theorem no_solution_exists : ¬∃ (f c₁ c₂ : ℕ), 
  (f > 0) ∧ (c₁ > 0) ∧ (c₂ > 0) ∧ 
  (∃ k : ℕ, f = k * (c₁ + c₂)) ∧
  (f + 5 = 2 * ((c₁ + 5) + (c₂ + 5))) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3334_333463


namespace NUMINAMATH_CALUDE_equation_solution_l3334_333452

theorem equation_solution :
  ∃ x : ℚ, x ≠ 2 ∧ (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2 ∧ x = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3334_333452


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3334_333415

def repeating_decimal_3 : ℚ := 1/3
def repeating_decimal_04 : ℚ := 4/99
def repeating_decimal_005 : ℚ := 5/999

theorem sum_of_repeating_decimals : 
  repeating_decimal_3 + repeating_decimal_04 + repeating_decimal_005 = 742/999 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3334_333415


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l3334_333484

/-- Given the purchase of blankets with known and unknown rates, prove the unknown rate -/
theorem unknown_blanket_rate (total_blankets : ℕ) (known_rate1 known_rate2 avg_rate : ℚ) 
  (count1 count2 count_unknown : ℕ) :
  total_blankets = count1 + count2 + count_unknown →
  count1 = 1 →
  count2 = 5 →
  count_unknown = 2 →
  known_rate1 = 100 →
  known_rate2 = 150 →
  avg_rate = 150 →
  (count1 * known_rate1 + count2 * known_rate2 + count_unknown * ((total_blankets * avg_rate - count1 * known_rate1 - count2 * known_rate2) / count_unknown)) / total_blankets = avg_rate →
  (total_blankets * avg_rate - count1 * known_rate1 - count2 * known_rate2) / count_unknown = 175 :=
by sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_l3334_333484


namespace NUMINAMATH_CALUDE_factor_expression_l3334_333433

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3334_333433


namespace NUMINAMATH_CALUDE_chess_tournament_schedules_l3334_333494

/-- Represents the number of players from each school -/
def num_players : ℕ := 4

/-- Represents the number of games each player plays against each opponent -/
def games_per_opponent : ℕ := 3

/-- Represents the number of games played simultaneously in each round -/
def games_per_round : ℕ := 3

/-- Calculates the total number of games in the tournament -/
def total_games : ℕ := num_players * num_players * games_per_opponent

/-- Calculates the number of rounds in the tournament -/
def num_rounds : ℕ := total_games / games_per_round

/-- Theorem stating the number of distinct ways to schedule the tournament -/
theorem chess_tournament_schedules :
  (Nat.factorial num_rounds) / (Nat.factorial games_per_round) =
  (Nat.factorial 16) / (Nat.factorial 3) :=
sorry

end NUMINAMATH_CALUDE_chess_tournament_schedules_l3334_333494


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3334_333443

-- Define the sets A and B
def A : Set ℝ := {x | x / (x - 1) < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3334_333443


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3334_333497

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3334_333497


namespace NUMINAMATH_CALUDE_current_library_books_l3334_333409

def library_books (initial : ℕ) (first_purchase : ℕ) (second_purchase : ℕ) (donation : ℕ) : ℕ :=
  initial + first_purchase + second_purchase - donation

theorem current_library_books :
  library_books 500 300 400 200 = 1000 := by sorry

end NUMINAMATH_CALUDE_current_library_books_l3334_333409


namespace NUMINAMATH_CALUDE_CD_length_l3334_333487

-- Define the points
variable (A B C D : ℝ × ℝ)

-- Define the conditions
def AD_perp_AB : (A.1 - D.1) * (B.1 - A.1) + (A.2 - D.2) * (B.2 - A.2) = 0 := sorry
def BC_perp_AB : (B.1 - C.1) * (B.1 - A.1) + (B.2 - C.2) * (B.2 - A.2) = 0 := sorry
def CD_perp_AC : (C.1 - D.1) * (C.1 - A.1) + (C.2 - D.2) * (C.2 - A.2) = 0 := sorry

def AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 4 := sorry
def BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 3 := sorry

-- Theorem to prove
theorem CD_length : 
  Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 20/3 :=
by sorry

end NUMINAMATH_CALUDE_CD_length_l3334_333487


namespace NUMINAMATH_CALUDE_sum_of_roots_satisfies_equation_l3334_333453

-- Define the polynomial
def polynomial (a b c x : ℝ) : ℝ := x^4 + a*x^2 + b*x + c

-- Define the equation for the sum of two roots
def sum_of_roots_equation (a b c u : ℝ) : ℝ := u^6 + 2*a*u^4 + (a^2 - 4*c)*u^2 - b^2

-- Theorem statement
theorem sum_of_roots_satisfies_equation (a b c : ℝ) :
  ∃ (x₁ x₂ : ℝ), polynomial a b c x₁ = 0 ∧ polynomial a b c x₂ = 0 ∧
  (∃ (u : ℝ), u = x₁ + x₂ ∧ sum_of_roots_equation a b c u = 0) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_satisfies_equation_l3334_333453


namespace NUMINAMATH_CALUDE_lucy_shells_count_l3334_333465

/-- The number of shells Lucy initially had -/
def initial_shells : ℕ := 68

/-- The number of additional shells Lucy found -/
def additional_shells : ℕ := 21

/-- The total number of shells Lucy has now -/
def total_shells : ℕ := initial_shells + additional_shells

theorem lucy_shells_count : total_shells = 89 := by
  sorry

end NUMINAMATH_CALUDE_lucy_shells_count_l3334_333465


namespace NUMINAMATH_CALUDE_units_digit_of_sum_factorials_l3334_333466

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_sum_factorials :
  units_digit (sum_of_factorials 50) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_factorials_l3334_333466


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l3334_333471

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 5

-- Define the interval
def I : Set ℝ := Set.Icc (-1) 1

-- State the theorem
theorem max_min_values_of_f :
  (∃ (x : ℝ), x ∈ I ∧ f x = 5) ∧
  (∀ (x : ℝ), x ∈ I → f x ≤ 5) ∧
  (∃ (x : ℝ), x ∈ I ∧ f x = 1) ∧
  (∀ (x : ℝ), x ∈ I → f x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l3334_333471


namespace NUMINAMATH_CALUDE_f_negative_2011_l3334_333486

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 2

theorem f_negative_2011 (a b : ℝ) :
  f a b 2011 = 10 → f a b (-2011) = -14 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_2011_l3334_333486


namespace NUMINAMATH_CALUDE_kylie_coins_from_brother_l3334_333469

/-- The number of coins Kylie got from her piggy bank -/
def piggy_bank_coins : ℕ := 15

/-- The number of coins Kylie got from her father -/
def father_coins : ℕ := 8

/-- The number of coins Kylie gave to her friend Laura -/
def coins_given_away : ℕ := 21

/-- The number of coins Kylie had left -/
def coins_left : ℕ := 15

/-- The number of coins Kylie got from her brother -/
def brother_coins : ℕ := 13

theorem kylie_coins_from_brother :
  piggy_bank_coins + brother_coins + father_coins - coins_given_away = coins_left :=
by sorry

end NUMINAMATH_CALUDE_kylie_coins_from_brother_l3334_333469


namespace NUMINAMATH_CALUDE_school_supplies_cost_l3334_333455

/-- Calculates the total cost of school supplies with discounts applied --/
theorem school_supplies_cost 
  (haley_paper_price : ℝ) 
  (haley_paper_quantity : ℕ)
  (sister_paper_price : ℝ)
  (sister_paper_quantity : ℕ)
  (paper_discount : ℝ)
  (haley_pen_price : ℝ)
  (haley_pen_quantity : ℕ)
  (sister_pen_price : ℝ)
  (sister_pen_quantity : ℕ)
  (pen_discount : ℝ)
  (h1 : haley_paper_price = 3.75)
  (h2 : haley_paper_quantity = 2)
  (h3 : sister_paper_price = 4.50)
  (h4 : sister_paper_quantity = 3)
  (h5 : paper_discount = 0.5)
  (h6 : haley_pen_price = 1.45)
  (h7 : haley_pen_quantity = 5)
  (h8 : sister_pen_price = 1.65)
  (h9 : sister_pen_quantity = 7)
  (h10 : pen_discount = 0.25)
  : ℝ := by
  sorry

end NUMINAMATH_CALUDE_school_supplies_cost_l3334_333455


namespace NUMINAMATH_CALUDE_law_of_sines_l3334_333425

/-- The Law of Sines for a triangle ABC -/
theorem law_of_sines (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (0 < A) ∧ (A < π) →
  (0 < B) ∧ (B < π) →
  (0 < C) ∧ (C < π) →
  (A + B + C = π) →
  (a / Real.sin A = b / Real.sin B) ∧
  (b / Real.sin B = c / Real.sin C) :=
sorry

end NUMINAMATH_CALUDE_law_of_sines_l3334_333425


namespace NUMINAMATH_CALUDE_massager_vibration_rate_l3334_333478

theorem massager_vibration_rate (lowest_rate : ℝ) : 
  (∃ (highest_rate : ℝ),
    highest_rate = lowest_rate * 1.6 ∧ 
    (5 * 60) * highest_rate = 768000) →
  lowest_rate = 1600 := by
sorry

end NUMINAMATH_CALUDE_massager_vibration_rate_l3334_333478


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3334_333408

theorem complex_modulus_problem (z : ℂ) (h : (1 - I) * z = 1 + 2*I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3334_333408


namespace NUMINAMATH_CALUDE_alien_arms_count_l3334_333499

/-- The number of arms an alien has -/
def alien_arms : ℕ := sorry

/-- The number of legs an alien has -/
def alien_legs : ℕ := 8

/-- The number of arms a Martian has -/
def martian_arms : ℕ := 2 * alien_arms

/-- The number of legs a Martian has -/
def martian_legs : ℕ := alien_legs / 2

theorem alien_arms_count : alien_arms = 3 :=
  by
    have h1 : 5 * (alien_arms + alien_legs) = 5 * (martian_arms + martian_legs) + 5 := by sorry
    sorry

end NUMINAMATH_CALUDE_alien_arms_count_l3334_333499


namespace NUMINAMATH_CALUDE_jake_papayas_l3334_333419

/-- The number of papayas Jake's brother can eat in one week -/
def brother_papayas : ℕ := 5

/-- The number of papayas Jake's father can eat in one week -/
def father_papayas : ℕ := 4

/-- The total number of papayas needed for 4 weeks -/
def total_papayas : ℕ := 48

/-- The number of weeks -/
def num_weeks : ℕ := 4

/-- Theorem: Jake can eat 3 papayas in one week -/
theorem jake_papayas : 
  ∃ (j : ℕ), j = 3 ∧ num_weeks * (j + brother_papayas + father_papayas) = total_papayas :=
by sorry

end NUMINAMATH_CALUDE_jake_papayas_l3334_333419


namespace NUMINAMATH_CALUDE_max_value_of_sum_cube_roots_l3334_333422

theorem max_value_of_sum_cube_roots (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_100 : a + b + c + d = 100) : 
  let S := (a / (b + 7)) ^ (1/3) + (b / (c + 7)) ^ (1/3) + 
           (c / (d + 7)) ^ (1/3) + (d / (a + 7)) ^ (1/3)
  S ≤ 8 / 7 ^ (1/3) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_cube_roots_l3334_333422


namespace NUMINAMATH_CALUDE_berry_difference_l3334_333468

/-- The number of strawberries in a box -/
def strawberries_per_box : ℕ := 12

/-- The cost of a box of strawberries in dollars -/
def strawberry_box_cost : ℕ := 2

/-- The number of blueberries in a box -/
def blueberries_per_box : ℕ := 48

/-- The cost of a box of blueberries in dollars -/
def blueberry_box_cost : ℕ := 3

/-- The amount Sareen can spend in dollars -/
def sareen_budget : ℕ := 12

/-- The number of strawberries Sareen can buy -/
def m : ℕ := (sareen_budget / strawberry_box_cost) * strawberries_per_box

/-- The number of blueberries Sareen can buy -/
def n : ℕ := (sareen_budget / blueberry_box_cost) * blueberries_per_box

theorem berry_difference : n - m = 120 := by
  sorry

end NUMINAMATH_CALUDE_berry_difference_l3334_333468


namespace NUMINAMATH_CALUDE_daps_equiv_48_dips_l3334_333440

/-- Conversion rate between daps and dops -/
def daps_to_dops : ℚ := 4 / 5

/-- Conversion rate between dops and dips -/
def dops_to_dips : ℚ := 8 / 3

/-- The number of daps equivalent to 48 dips -/
def daps_equiv_to_48_dips : ℚ := 22.5

theorem daps_equiv_48_dips :
  daps_equiv_to_48_dips = 48 * dops_to_dips * daps_to_dops := by
  sorry

end NUMINAMATH_CALUDE_daps_equiv_48_dips_l3334_333440


namespace NUMINAMATH_CALUDE_inscribed_rectangle_perimeter_is_12_l3334_333438

/-- A right triangle with legs of length 6 containing an inscribed rectangle -/
structure RightTriangleWithInscribedRectangle where
  /-- The length of each leg of the right triangle -/
  leg_length : ℝ
  /-- The inscribed rectangle shares an angle with the triangle -/
  shares_angle : Bool
  /-- The inscribed rectangle is contained within the triangle -/
  is_inscribed : Bool

/-- The perimeter of the inscribed rectangle -/
def inscribed_rectangle_perimeter (t : RightTriangleWithInscribedRectangle) : ℝ := 12

/-- Theorem: The perimeter of the inscribed rectangle is 12 -/
theorem inscribed_rectangle_perimeter_is_12 (t : RightTriangleWithInscribedRectangle)
  (h1 : t.leg_length = 6)
  (h2 : t.shares_angle = true)
  (h3 : t.is_inscribed = true) :
  inscribed_rectangle_perimeter t = 12 := by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_perimeter_is_12_l3334_333438


namespace NUMINAMATH_CALUDE_root_magnitude_theorem_l3334_333417

theorem root_magnitude_theorem (A B C D : ℝ) 
  (h1 : ∀ x : ℂ, x^2 + A*x + B = 0 → Complex.abs x < 1)
  (h2 : ∀ x : ℂ, x^2 + C*x + D = 0 → Complex.abs x < 1) :
  ∀ x : ℂ, x^2 + (A+C)/2*x + (B+D)/2 = 0 → Complex.abs x < 1 :=
sorry

end NUMINAMATH_CALUDE_root_magnitude_theorem_l3334_333417


namespace NUMINAMATH_CALUDE_triangle_existence_uniqueness_l3334_333407

/-- A point in 2D Euclidean space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Incircle of a triangle -/
structure Incircle where
  center : Point
  radius : ℝ

/-- Excircle of a triangle -/
structure Excircle where
  center : Point
  radius : ℝ

/-- Predicate to check if a point lies on a line segment -/
def lies_on_segment (P Q R : Point) : Prop := sorry

/-- Predicate to check if a point is the tangency point of a circle and a line -/
def is_tangency_point (P : Point) (C : Incircle ⊕ Excircle) (L M : Point) : Prop := sorry

/-- Theorem stating the existence and uniqueness of a triangle given specific tangency points -/
theorem triangle_existence_uniqueness 
  (T_a T_aa T_c T_ac : Point) 
  (h_distinct : T_a ≠ T_aa ∧ T_c ≠ T_ac) 
  (h_not_collinear : ¬ lies_on_segment T_a T_c T_aa) : 
  ∃! (ABC : Triangle) (k : Incircle) (k' : Excircle), 
    is_tangency_point T_a (Sum.inl k) ABC.B ABC.C ∧ 
    is_tangency_point T_aa (Sum.inr k') ABC.B ABC.C ∧
    is_tangency_point T_c (Sum.inl k) ABC.A ABC.B ∧
    is_tangency_point T_ac (Sum.inr k') ABC.A ABC.B := by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_uniqueness_l3334_333407


namespace NUMINAMATH_CALUDE_hidden_dots_sum_l3334_333473

/-- The sum of numbers on a single die --/
def single_die_sum : ℕ := 21

/-- The total number of dice --/
def total_dice : ℕ := 4

/-- The number of visible faces --/
def visible_faces : ℕ := 10

/-- The sum of visible numbers --/
def visible_sum : ℕ := 37

theorem hidden_dots_sum :
  single_die_sum * total_dice - visible_sum = 47 := by
  sorry

end NUMINAMATH_CALUDE_hidden_dots_sum_l3334_333473


namespace NUMINAMATH_CALUDE_max_annual_profit_l3334_333461

noncomputable section

def fixed_cost : ℝ := 2.6

def additional_investment (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 40 then 10 * x^2 + 300 * x
  else (901 * x^2 - 9450 * x + 10000) / x

def selling_price : ℝ := 0.9

def annual_profit (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 40 then selling_price * x - additional_investment x - fixed_cost
  else (selling_price * x * x - (901 * x^2 - 9450 * x + 10000)) / x - fixed_cost

theorem max_annual_profit :
  ∃ (x : ℝ), x = 100 ∧ annual_profit x = 8990 ∧
  ∀ (y : ℝ), y ≥ 0 → annual_profit y ≤ annual_profit x :=
sorry

end NUMINAMATH_CALUDE_max_annual_profit_l3334_333461


namespace NUMINAMATH_CALUDE_ant_final_position_l3334_333459

/-- Represents the vertices of the rectangle --/
inductive Vertex : Type
  | A : Vertex
  | B : Vertex
  | C : Vertex
  | D : Vertex

/-- Represents a single movement of the ant --/
def next_vertex : Vertex → Vertex
  | Vertex.A => Vertex.B
  | Vertex.B => Vertex.C
  | Vertex.C => Vertex.D
  | Vertex.D => Vertex.A

/-- Represents multiple movements of the ant --/
def ant_position (start : Vertex) (moves : Nat) : Vertex :=
  match moves with
  | 0 => start
  | n + 1 => next_vertex (ant_position start n)

/-- The main theorem to prove --/
theorem ant_final_position :
  ant_position Vertex.A 2018 = Vertex.C := by
  sorry

end NUMINAMATH_CALUDE_ant_final_position_l3334_333459


namespace NUMINAMATH_CALUDE_one_point_six_million_scientific_notation_l3334_333424

theorem one_point_six_million_scientific_notation :
  (1.6 : ℝ) * (1000000 : ℝ) = (1.6 : ℝ) * (10 : ℝ) ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_one_point_six_million_scientific_notation_l3334_333424


namespace NUMINAMATH_CALUDE_equation_solutions_l3334_333476

-- Define the equations
def equation1 (x : ℝ) : Prop := (2*x - 1)^2 - 4*x = 0
def equation2 (x : ℝ) : Prop := (2*x - 3)^2 = x^2

-- State the theorem
theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = 1 + Real.sqrt 3 / 2 ∧ x2 = 1 - Real.sqrt 3 / 2 ∧ equation1 x1 ∧ equation1 x2) ∧
  (∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = 1 ∧ equation2 x1 ∧ equation2 x2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3334_333476


namespace NUMINAMATH_CALUDE_trigonometric_sum_equals_half_l3334_333474

theorem trigonometric_sum_equals_half : 
  Real.sin (120 * π / 180) ^ 2 + Real.cos (180 * π / 180) + Real.tan (45 * π / 180) - 
  Real.cos (-330 * π / 180) ^ 2 + Real.sin (-210 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equals_half_l3334_333474


namespace NUMINAMATH_CALUDE_cubic_polynomial_problem_l3334_333426

-- Define the cubic equation
def cubic_equation (x : ℝ) : ℝ := x^3 + 4*x^2 + 7*x + 10

-- Define the roots a, b, c
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

-- Define P(x)
noncomputable def P : ℝ → ℝ := sorry

-- Theorem statement
theorem cubic_polynomial_problem :
  (cubic_equation a = 0) ∧ 
  (cubic_equation b = 0) ∧ 
  (cubic_equation c = 0) ∧
  (P a = 2*(b + c)) ∧
  (P b = 2*(a + c)) ∧
  (P c = 2*(a + b)) ∧
  (P (a + b + c) = -20) →
  ∀ x, P x = (4*x^3 + 16*x^2 + 55*x - 16) / 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_problem_l3334_333426


namespace NUMINAMATH_CALUDE_polynomial_integer_roots_l3334_333493

def polynomial (x a : ℤ) : ℤ := x^3 + 5*x^2 + a*x + 12

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, polynomial x a = 0

def valid_a_values : Set ℤ := {-18, 16, -20, 12, -16, 8, -11, 5, -4, 2, 0, -1}

theorem polynomial_integer_roots :
  ∀ a : ℤ, has_integer_root a ↔ a ∈ valid_a_values := by sorry

end NUMINAMATH_CALUDE_polynomial_integer_roots_l3334_333493


namespace NUMINAMATH_CALUDE_exam_arrangements_l3334_333410

/- Define the number of subjects -/
def num_subjects : ℕ := 6

/- Define the condition that Chinese must be first -/
def chinese_first : ℕ := 1

/- Define the number of subjects excluding Chinese, Math, and English -/
def other_subjects : ℕ := 3

/- Define the number of spaces available for Math and English -/
def available_spaces : ℕ := 4

/- Define the function to calculate the number of arrangements -/
def num_arrangements : ℕ :=
  chinese_first * (Nat.factorial other_subjects) * (available_spaces * (available_spaces - 1) / 2)

/- Theorem statement -/
theorem exam_arrangements :
  num_arrangements = 72 :=
sorry

end NUMINAMATH_CALUDE_exam_arrangements_l3334_333410


namespace NUMINAMATH_CALUDE_max_value_problem_l3334_333447

theorem max_value_problem (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  11 * x + 3 * y + 8 * z ≤ Real.sqrt (2292.25 / 225) :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_l3334_333447


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3334_333456

/-- Represents factorization from left to right -/
def is_factorization_left_to_right (f : ℝ → ℝ) (g : ℝ → ℝ → ℝ) : Prop :=
  ∀ x, f x = g (x + 2) (x - 2)

/-- The equation m^2 - 4 = (m + 2)(m - 2) represents factorization from left to right -/
theorem quadratic_factorization :
  is_factorization_left_to_right (λ m => m^2 - 4) (λ a b => a * b) :=
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3334_333456


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l3334_333421

theorem part_to_whole_ratio (N P : ℚ) (h1 : N = 240) (h2 : (1/5) * N + 6 = P - 6) :
  (P - 6) / N = 9 / 40 := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l3334_333421


namespace NUMINAMATH_CALUDE_families_without_pets_l3334_333498

theorem families_without_pets (total : ℕ) (cats : ℕ) (dogs : ℕ) (both : ℕ) 
  (h1 : total = 40)
  (h2 : cats = 18)
  (h3 : dogs = 24)
  (h4 : both = 10) :
  total - (cats + dogs - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_families_without_pets_l3334_333498


namespace NUMINAMATH_CALUDE_mike_total_hours_l3334_333428

/-- Calculate the total hours worked given hours per day and number of days -/
def total_hours (hours_per_day : ℕ) (days : ℕ) : ℕ := hours_per_day * days

/-- Proof that Mike worked 15 hours in total -/
theorem mike_total_hours : total_hours 3 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_mike_total_hours_l3334_333428


namespace NUMINAMATH_CALUDE_percent_composition_l3334_333431

theorem percent_composition (z : ℝ) (hz : z ≠ 0) :
  (42 / 100) * z = (60 / 100) * ((70 / 100) * z) := by
  sorry

end NUMINAMATH_CALUDE_percent_composition_l3334_333431


namespace NUMINAMATH_CALUDE_second_divisor_problem_l3334_333470

theorem second_divisor_problem (initial : ℝ) (first_divisor : ℝ) (final_result : ℝ) (x : ℝ) :
  initial = 8900 →
  first_divisor = 6 →
  final_result = 370.8333333333333 →
  (initial / first_divisor) / x = final_result →
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_problem_l3334_333470


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3334_333435

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    prove that under certain conditions, its equation is x²/4 - y²/6 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (C : ℝ × ℝ → Prop) 
  (hC : ∀ x y, C (x, y) ↔ x^2 / a^2 - y^2 / b^2 = 1)
  (F : ℝ × ℝ) (hF : F.1 > 0 ∧ F.2 = 0) -- Right focus
  (B : ℝ × ℝ) (hB : B.1 = 0) -- B is on the imaginary axis
  (A : ℝ × ℝ) (hA : C A) -- A is on the hyperbola
  (hAF : ∃ t : ℝ, A = B + t • (F - B)) -- A is on BF
  (hBA : ∃ k : ℝ, k • (A - B) = 2 • (F - A)) -- BA = 2AF
  (hBF : (F.1 - B.1)^2 + (F.2 - B.2)^2 = 16) -- |BF| = 4
  : ∀ x y, C (x, y) ↔ x^2 / 4 - y^2 / 6 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3334_333435


namespace NUMINAMATH_CALUDE_excellent_grade_percentage_l3334_333450

theorem excellent_grade_percentage (total : ℕ) (excellent : ℕ) (h1 : total = 360) (h2 : excellent = 72) :
  (excellent : ℚ) / (total : ℚ) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_excellent_grade_percentage_l3334_333450


namespace NUMINAMATH_CALUDE_area_of_PRQ_l3334_333430

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)
  (xy_length : Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 15)
  (xz_length : Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) = 14)
  (yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 7)

-- Define the circumcenter P
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the incenter Q
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the point R
def R (t : Triangle) : ℝ × ℝ := sorry

-- Define the condition for R being tangent to XZ, YZ, and the circumcircle
def is_tangent (t : Triangle) (r : ℝ × ℝ) : Prop := sorry

-- Define the area of a triangle given three points
def triangle_area (a b c : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_PRQ (t : Triangle) 
  (h : is_tangent t (R t)) : 
  triangle_area (circumcenter t) (incenter t) (R t) = 245 / 72 := by
  sorry

end NUMINAMATH_CALUDE_area_of_PRQ_l3334_333430


namespace NUMINAMATH_CALUDE_seashells_given_correct_l3334_333444

/-- The number of seashells Tom gave to Jessica -/
def seashells_given : ℕ :=
  5 - 3

theorem seashells_given_correct : seashells_given = 2 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_correct_l3334_333444


namespace NUMINAMATH_CALUDE_greatest_n_value_l3334_333480

theorem greatest_n_value (n : ℤ) (h : 101 * n^2 ≤ 3600) : n ≤ 5 ∧ ∃ (m : ℤ), m > 5 → 101 * m^2 > 3600 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_l3334_333480


namespace NUMINAMATH_CALUDE_davids_math_marks_l3334_333475

/-- Represents the marks obtained in each subject -/
structure SubjectMarks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculates the average marks given the total marks and number of subjects -/
def average (total : ℕ) (subjects : ℕ) : ℚ :=
  (total : ℚ) / (subjects : ℚ)

/-- Theorem stating that David's marks in Mathematics are 35 -/
theorem davids_math_marks (marks : SubjectMarks) 
    (h1 : marks.english = 36)
    (h2 : marks.physics = 42)
    (h3 : marks.chemistry = 57)
    (h4 : marks.biology = 55)
    (h5 : average (marks.english + marks.mathematics + marks.physics + marks.chemistry + marks.biology) 5 = 45) :
    marks.mathematics = 35 := by
  sorry

#check davids_math_marks

end NUMINAMATH_CALUDE_davids_math_marks_l3334_333475


namespace NUMINAMATH_CALUDE_equation_solutions_l3334_333464

/-- The equation we're solving -/
def equation (x : ℂ) : Prop :=
  x ≠ -2 ∧ (15*x - x^2)/(x + 2) * (x + (15 - x)/(x + 2)) = 48

/-- The set of solutions to the equation -/
def solutions : Set ℂ :=
  { x | x = 12 + 2*Real.sqrt 38 ∨
        x = 12 - 2*Real.sqrt 38 ∨
        x = -1/2 + Complex.I*(Real.sqrt 95)/2 ∨
        x = -1/2 - Complex.I*(Real.sqrt 95)/2 }

/-- Theorem stating that the solutions are correct and complete -/
theorem equation_solutions :
  ∀ x, equation x ↔ x ∈ solutions :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3334_333464


namespace NUMINAMATH_CALUDE_units_digit_of_42_cubed_plus_24_cubed_l3334_333482

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_42_cubed_plus_24_cubed :
  unitsDigit (42^3 + 24^3) = 2 := by
  sorry


end NUMINAMATH_CALUDE_units_digit_of_42_cubed_plus_24_cubed_l3334_333482


namespace NUMINAMATH_CALUDE_conical_frustum_volume_l3334_333432

/-- Right prism with equilateral triangle base -/
structure RightPrism where
  height : ℝ
  base_side : ℝ

/-- Conical frustum within the right prism -/
def ConicalFrustum (p : RightPrism) : Type :=
  Unit

/-- Volume of the conical frustum -/
def volume (p : RightPrism) (f : ConicalFrustum p) : ℝ :=
  sorry

/-- Theorem: Volume of conical frustum in given right prism -/
theorem conical_frustum_volume (p : RightPrism) (f : ConicalFrustum p)
    (h1 : p.height = 3)
    (h2 : p.base_side = 1) :
    volume p f = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_conical_frustum_volume_l3334_333432


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l3334_333406

theorem line_intercepts_sum (c : ℝ) : 
  (∃ x y : ℝ, 3 * x + 5 * y + c = 0 ∧ x + y = 30) → c = -56.25 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l3334_333406


namespace NUMINAMATH_CALUDE_download_time_proof_l3334_333414

/-- Proves that the download time for a 360 GB program at 50 MB/s is 2 hours -/
theorem download_time_proof (download_speed : ℝ) (program_size : ℝ) (mb_per_gb : ℝ) :
  download_speed = 50 ∧ program_size = 360 ∧ mb_per_gb = 1000 →
  (program_size * mb_per_gb) / (download_speed * 3600) = 2 := by
  sorry

end NUMINAMATH_CALUDE_download_time_proof_l3334_333414


namespace NUMINAMATH_CALUDE_math_statements_l3334_333420

theorem math_statements :
  (8^0 = 1) ∧
  (|-8| = 8) ∧
  (-(-8) = 8) ∧
  (¬(Real.sqrt 8 = 2 * Real.sqrt 2 ∨ Real.sqrt 8 = -2 * Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_math_statements_l3334_333420


namespace NUMINAMATH_CALUDE_quadratic_real_roots_range_l3334_333489

theorem quadratic_real_roots_range (m : ℝ) :
  (∃ x : ℝ, x^2 - x - m = 0) ↔ m ≥ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_range_l3334_333489


namespace NUMINAMATH_CALUDE_quadratic_equations_common_root_l3334_333442

theorem quadratic_equations_common_root (p q r s : ℝ) 
  (hq : q ≠ -1) (hs : s ≠ -1) : 
  (∃ (a b : ℝ), (a^2 + p*a + q = 0 ∧ a^2 + r*a + s = 0) ∧ 
   (b^2 + p*b + q = 0 ∧ (1/b)^2 + r*(1/b) + s = 0)) ↔ 
  (p*r = (q+1)*(s+1) ∧ p*(q+1)*s = r*(s+1)*q) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equations_common_root_l3334_333442


namespace NUMINAMATH_CALUDE_characterization_of_solutions_l3334_333411

/-- Sum of digits of a non-negative integer -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- The set of solutions -/
def solution_set : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19}

/-- Main theorem: n ≤ 2s(n) iff n is in the solution set -/
theorem characterization_of_solutions (n : ℕ) :
  n ≤ 2 * sum_of_digits n ↔ n ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_characterization_of_solutions_l3334_333411


namespace NUMINAMATH_CALUDE_tree_spacing_l3334_333400

theorem tree_spacing (yard_length : ℕ) (num_trees : ℕ) (distance : ℕ) : 
  yard_length = 273 ∧ num_trees = 14 → distance * (num_trees - 1) = yard_length → distance = 21 := by
  sorry

end NUMINAMATH_CALUDE_tree_spacing_l3334_333400


namespace NUMINAMATH_CALUDE_ac_unit_final_price_l3334_333429

/-- Calculates the final price of an air-conditioning unit after a series of price changes. -/
def final_price (initial_price : ℝ) : ℝ :=
  let price1 := initial_price * (1 - 0.12)  -- February
  let price2 := price1 * (1 + 0.08)         -- March
  let price3 := price2 * (1 - 0.10)         -- April
  let price4 := price3 * (1 + 0.05)         -- June
  let price5 := price4 * (1 - 0.07)         -- August
  let price6 := price5 * (1 + 0.06)         -- October
  let price7 := price6 * (1 - 0.15)         -- November
  price7

/-- Theorem stating that the final price of the air-conditioning unit is approximately $353.71. -/
theorem ac_unit_final_price : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |final_price 470 - 353.71| < ε :=
sorry

end NUMINAMATH_CALUDE_ac_unit_final_price_l3334_333429


namespace NUMINAMATH_CALUDE_map_scale_l3334_333449

/-- Given a map where 15 centimeters represents 90 kilometers,
    prove that 20 centimeters represents 120 kilometers. -/
theorem map_scale (map_cm : ℝ) (map_km : ℝ) (length_cm : ℝ) :
  map_cm = 15 ∧ map_km = 90 ∧ length_cm = 20 →
  (length_cm / map_cm) * map_km = 120 := by
sorry

end NUMINAMATH_CALUDE_map_scale_l3334_333449


namespace NUMINAMATH_CALUDE_amount_after_two_years_l3334_333434

/-- The annual growth rate -/
def r : ℚ := 1 / 8

/-- The initial amount -/
def initial_amount : ℚ := 76800

/-- The amount after n years -/
def amount_after (n : ℕ) : ℚ := initial_amount * (1 + r) ^ n

/-- Theorem: The amount after two years is 97200 -/
theorem amount_after_two_years : amount_after 2 = 97200 := by
  sorry

end NUMINAMATH_CALUDE_amount_after_two_years_l3334_333434


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l3334_333483

noncomputable def y (x : ℝ) : ℝ := Real.rpow (x - Real.log x - 1) (1/3)

theorem function_satisfies_equation (x : ℝ) (h : x > 0) :
  Real.log x + (y x)^3 - 3 * x * (y x)^2 * (deriv y x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l3334_333483


namespace NUMINAMATH_CALUDE_binomial_inequality_l3334_333403

theorem binomial_inequality (n : ℕ) : 2 ≤ (1 + 1 / n)^n ∧ (1 + 1 / n)^n < 3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_inequality_l3334_333403


namespace NUMINAMATH_CALUDE_jeans_price_calculation_l3334_333401

/-- The price of jeans after discount and tax -/
def jeans_final_price (socks_price t_shirt_price jeans_price : ℝ)
  (jeans_discount t_shirt_discount tax_rate : ℝ) : ℝ :=
  let jeans_discounted := jeans_price * (1 - jeans_discount)
  let taxable_amount := jeans_discounted + t_shirt_price * (1 - t_shirt_discount)
  jeans_discounted * (1 + tax_rate)

/-- The problem statement -/
theorem jeans_price_calculation :
  let socks_price := 5
  let t_shirt_price := socks_price + 10
  let jeans_price := 2 * t_shirt_price
  let jeans_discount := 0.15
  let t_shirt_discount := 0.10
  let tax_rate := 0.08
  jeans_final_price socks_price t_shirt_price jeans_price
    jeans_discount t_shirt_discount tax_rate = 27.54 := by
  sorry

end NUMINAMATH_CALUDE_jeans_price_calculation_l3334_333401


namespace NUMINAMATH_CALUDE_f_increasing_f_comparison_l3334_333418

noncomputable section

-- Define the function f with the given property
def f_property (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, a ≠ b → a * f a + b * f b > a * f b + b * f a

-- Theorem 1: f is monotonically increasing
theorem f_increasing (f : ℝ → ℝ) (hf : f_property f) :
  Monotone f := by sorry

-- Theorem 2: f(x+y) > f(6) under given conditions
theorem f_comparison (f : ℝ → ℝ) (hf : f_property f) (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h_eq : 4/x + 9/y = 4) :
  f (x + y) > f 6 := by sorry

end NUMINAMATH_CALUDE_f_increasing_f_comparison_l3334_333418


namespace NUMINAMATH_CALUDE_fraction_equality_l3334_333413

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / (a + 37) = 875 / 1000 → a = 259 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3334_333413


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l3334_333446

def solutions : Set (ℤ × ℤ) := {(6, 9), (7, 3), (8, 1), (9, 0), (11, -1), (17, -2), (4, -15), (3, -9), (2, -7), (1, -6), (-1, -5), (-7, -4)}

theorem diophantine_equation_solutions :
  {(x, y) : ℤ × ℤ | x * y + 3 * x - 5 * y = -3} = solutions := by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l3334_333446


namespace NUMINAMATH_CALUDE_no_roots_composition_l3334_333481

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b * x + c

/-- Theorem: If f(x) = x has no real roots, then f(f(x)) = x has no real roots -/
theorem no_roots_composition (b c : ℝ) 
  (h : ∀ x : ℝ, f b c x ≠ x) : 
  ∀ x : ℝ, f b c (f b c x) ≠ x := by
  sorry

end NUMINAMATH_CALUDE_no_roots_composition_l3334_333481


namespace NUMINAMATH_CALUDE_janice_starting_sentences_janice_started_with_258_sentences_l3334_333477

/-- Calculates the number of sentences Janice started with today -/
theorem janice_starting_sentences 
  (typing_speed : ℕ) 
  (typing_duration1 typing_duration2 typing_duration3 : ℕ)
  (erased_sentences : ℕ)
  (total_sentences : ℕ) : ℕ :=
  let total_duration := typing_duration1 + typing_duration2 + typing_duration3
  let typed_sentences := total_duration * typing_speed
  let net_typed_sentences := typed_sentences - erased_sentences
  total_sentences - net_typed_sentences

/-- Proves that Janice started with 258 sentences today -/
theorem janice_started_with_258_sentences : 
  janice_starting_sentences 6 20 15 18 40 536 = 258 := by
  sorry

end NUMINAMATH_CALUDE_janice_starting_sentences_janice_started_with_258_sentences_l3334_333477


namespace NUMINAMATH_CALUDE_unique_intersection_implies_a_equals_three_l3334_333458

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x - 2

def intersection_count (f g : ℝ → ℝ) : ℕ := sorry

theorem unique_intersection_implies_a_equals_three :
  ∀ a : ℝ, intersection_count f (g a) = 1 → a = 3 := by sorry

end NUMINAMATH_CALUDE_unique_intersection_implies_a_equals_three_l3334_333458


namespace NUMINAMATH_CALUDE_directrix_of_given_parabola_l3334_333472

/-- A parabola with equation y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ := sorry

/-- The parabola y = 4x^2 - 3 -/
def given_parabola : Parabola := { a := 4, b := -3 }

theorem directrix_of_given_parabola :
  directrix given_parabola = -19/16 := by sorry

end NUMINAMATH_CALUDE_directrix_of_given_parabola_l3334_333472


namespace NUMINAMATH_CALUDE_fractional_equation_solution_condition_l3334_333495

theorem fractional_equation_solution_condition (m : ℝ) : 
  (∃ x : ℝ, x ≠ 2 ∧ (m + x) / (2 - x) - 3 = 0) ↔ m ≠ -2 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_condition_l3334_333495


namespace NUMINAMATH_CALUDE_max_third_altitude_is_seven_l3334_333427

/-- A scalene triangle with two known altitudes -/
structure ScaleneTriangle where
  /-- The length of the first known altitude -/
  altitude1 : ℝ
  /-- The length of the second known altitude -/
  altitude2 : ℝ
  /-- The triangle is scalene -/
  scalene : True
  /-- The known altitudes have lengths 5 and 15 -/
  altitudes_given : altitude1 = 5 ∧ altitude2 = 15

/-- The maximum possible integer length of the third altitude -/
def max_third_altitude (triangle : ScaleneTriangle) : ℕ :=
  7

/-- Theorem stating that the maximum possible integer length of the third altitude is 7 -/
theorem max_third_altitude_is_seven (triangle : ScaleneTriangle) :
  max_third_altitude triangle = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_third_altitude_is_seven_l3334_333427


namespace NUMINAMATH_CALUDE_megatech_budget_allocation_l3334_333491

theorem megatech_budget_allocation :
  let total_budget : ℝ := 100
  let microphotonics : ℝ := 10
  let home_electronics : ℝ := 24
  let food_additives : ℝ := 15
  let industrial_lubricants : ℝ := 8
  let basic_astrophysics_degrees : ℝ := 50.4
  let total_degrees : ℝ := 360
  let basic_astrophysics : ℝ := (basic_astrophysics_degrees / total_degrees) * total_budget
  let genetically_modified_microorganisms : ℝ := total_budget - (microphotonics + home_electronics + food_additives + industrial_lubricants + basic_astrophysics)
  genetically_modified_microorganisms = 29 := by
sorry

end NUMINAMATH_CALUDE_megatech_budget_allocation_l3334_333491


namespace NUMINAMATH_CALUDE_line_equation_proof_l3334_333448

/-- A line passing through point (1, -2) with slope 3 has the equation 3x - y - 5 = 0 -/
theorem line_equation_proof (x y : ℝ) : 
  (y - (-2) = 3 * (x - 1)) ↔ (3 * x - y - 5 = 0) := by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3334_333448


namespace NUMINAMATH_CALUDE_binomial_9_choose_3_l3334_333405

theorem binomial_9_choose_3 : Nat.choose 9 3 = 84 := by
  sorry

end NUMINAMATH_CALUDE_binomial_9_choose_3_l3334_333405


namespace NUMINAMATH_CALUDE_half_shading_sufficient_l3334_333485

/-- Represents a square grid --/
structure Grid :=
  (size : ℕ)
  (total_cells : ℕ)

/-- Represents the minimum number of cells to be shaded --/
def min_shaded_cells (g : Grid) : ℕ := g.total_cells / 2

/-- Theorem stating that shading half the cells is sufficient --/
theorem half_shading_sufficient (g : Grid) (h : g.size = 12) (h' : g.total_cells = 144) :
  ∃ (shaded : ℕ), shaded = min_shaded_cells g ∧ 
  shaded ≤ g.total_cells ∧
  shaded ≥ g.total_cells / 2 :=
sorry

#check half_shading_sufficient

end NUMINAMATH_CALUDE_half_shading_sufficient_l3334_333485


namespace NUMINAMATH_CALUDE_sum_and_equality_problem_l3334_333416

theorem sum_and_equality_problem (a b c : ℚ) :
  a + b + c = 120 ∧ (a + 8 = b - 3) ∧ (b - 3 = 3 * c) →
  b = 56 + 4/7 := by
sorry

end NUMINAMATH_CALUDE_sum_and_equality_problem_l3334_333416


namespace NUMINAMATH_CALUDE_remainder_congruence_l3334_333402

theorem remainder_congruence (x : ℤ) 
  (h1 : (2 + x) % 8 = 9 % 8)
  (h2 : (3 + x) % 27 = 4 % 27)
  (h3 : (11 + x) % 1331 = 49 % 1331) :
  x % 198 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_congruence_l3334_333402


namespace NUMINAMATH_CALUDE_cone_radius_l3334_333436

/-- Given a cone with slant height 10 cm and curved surface area 157.07963267948966 cm², 
    the radius of the base is 5 cm. -/
theorem cone_radius (slant_height : ℝ) (curved_surface_area : ℝ) :
  slant_height = 10 ∧ 
  curved_surface_area = 157.07963267948966 ∧
  curved_surface_area = Real.pi * (5 : ℝ) * slant_height :=
by sorry

end NUMINAMATH_CALUDE_cone_radius_l3334_333436


namespace NUMINAMATH_CALUDE_exists_distinct_diagonal_products_l3334_333462

/-- A type representing the vertices of a nonagon -/
inductive Vertex : Type
| v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9

/-- A function type representing an arrangement of numbers on the nonagon vertices -/
def Arrangement := Vertex → Fin 9

/-- The set of all diagonals in a nonagon -/
def Diagonals : Set (Vertex × Vertex) := sorry

/-- Calculate the product of numbers at the ends of a diagonal -/
def diagonalProduct (arr : Arrangement) (d : Vertex × Vertex) : Nat := sorry

/-- Theorem stating that there exists an arrangement with all distinct diagonal products -/
theorem exists_distinct_diagonal_products :
  ∃ (arr : Arrangement), Function.Injective (diagonalProduct arr) := by sorry

end NUMINAMATH_CALUDE_exists_distinct_diagonal_products_l3334_333462


namespace NUMINAMATH_CALUDE_min_value_trigonometric_fraction_l3334_333460

theorem min_value_trigonometric_fraction (a b : ℝ) (θ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hθ : θ ∈ Set.Ioo 0 (π / 2)) :
  a / Real.sin θ + b / Real.cos θ ≥ (Real.rpow a (2/3) + Real.rpow b (2/3))^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_fraction_l3334_333460


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_equation_l3334_333451

/-- Given distinct non-zero real numbers x, y, and z, and a real number r,
    if x^2(y-z), y^2(z-x), and z^2(x-y) form a geometric progression with common ratio r,
    then r satisfies the equation r^2 + r + 1 = 0 -/
theorem geometric_progression_ratio_equation (x y z r : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (hprogression : ∃ (a : ℝ), a ≠ 0 ∧ 
    x^2 * (y - z) = a ∧ 
    y^2 * (z - x) = a * r ∧ 
    z^2 * (x - y) = a * r^2) :
  r^2 + r + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_equation_l3334_333451


namespace NUMINAMATH_CALUDE_log_power_difference_l3334_333412

theorem log_power_difference (x : ℝ) (h1 : x < 1) 
  (h2 : (Real.log x / Real.log 10)^2 - Real.log (x^2) / Real.log 10 = 48) :
  (Real.log x / Real.log 10)^5 - Real.log (x^5) / Real.log 10 = -7746 := by
  sorry

end NUMINAMATH_CALUDE_log_power_difference_l3334_333412


namespace NUMINAMATH_CALUDE_multiple_births_quintuplets_l3334_333423

theorem multiple_births_quintuplets (total_babies : ℕ) 
  (h_total : total_babies = 1500)
  (h_triplets_quadruplets : ∃ (t q : ℕ), t = 3 * q)
  (h_twins_triplets : ∃ (w t : ℕ), w = 2 * t)
  (h_quintuplets_quadruplets : ∃ (q qu : ℕ), q = qu / 2)
  (h_sum : ∃ (w t q qu : ℕ), 2 * w + 3 * t + 4 * q + 5 * qu = total_babies) :
  ∃ (quintuplets : ℕ), quintuplets = 1500 / 11 ∧ 
    quintuplets * 5 = total_babies * 5 / 11 :=
by sorry

end NUMINAMATH_CALUDE_multiple_births_quintuplets_l3334_333423


namespace NUMINAMATH_CALUDE_cloth_cost_price_l3334_333454

/-- Given a cloth sale scenario, prove the cost price per meter -/
theorem cloth_cost_price
  (total_meters : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ)
  (h1 : total_meters = 85)
  (h2 : selling_price = 8925)
  (h3 : profit_per_meter = 35) :
  (selling_price - total_meters * profit_per_meter) / total_meters = 70 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l3334_333454


namespace NUMINAMATH_CALUDE_smallest_integer_inequality_l3334_333496

theorem smallest_integer_inequality (x y z : ℝ) :
  ∃ (n : ℕ), (x^2 + y^2 + z^2)^2 ≤ n * (x^4 + y^4 + z^4) ∧
  ∀ (m : ℕ), m < n → ∃ (a b c : ℝ), (a^2 + b^2 + c^2)^2 > m * (a^4 + b^4 + c^4) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_inequality_l3334_333496
