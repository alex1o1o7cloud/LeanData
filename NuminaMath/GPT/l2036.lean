import Mathlib

namespace selling_price_correct_l2036_203675

noncomputable def discount1 (price : ℝ) : ℝ := price * 0.85
noncomputable def discount2 (price : ℝ) : ℝ := price * 0.90
noncomputable def discount3 (price : ℝ) : ℝ := price * 0.95

noncomputable def final_price (initial_price : ℝ) : ℝ :=
  discount3 (discount2 (discount1 initial_price))

theorem selling_price_correct : final_price 3600 = 2616.30 := by
  sorry

end selling_price_correct_l2036_203675


namespace correct_option_e_l2036_203622

theorem correct_option_e : 15618 = 1 + 5^6 - 1 * 8 :=
by sorry

end correct_option_e_l2036_203622


namespace alphabet_letter_count_l2036_203642

def sequence_count : Nat :=
  let total_sequences := 2^7
  let sequences_per_letter := 1 + 7 -- 1 correct sequence + 7 single-bit alterations
  total_sequences / sequences_per_letter

theorem alphabet_letter_count : sequence_count = 16 :=
  by
    -- Proof placeholder
    sorry

end alphabet_letter_count_l2036_203642


namespace percentage_of_total_spent_on_other_items_l2036_203640

-- Definitions for the given problem conditions

def total_amount (a : ℝ) := a
def spent_on_clothing (a : ℝ) := 0.50 * a
def spent_on_food (a : ℝ) := 0.10 * a
def spent_on_other_items (a x_clothing x_food : ℝ) := a - x_clothing - x_food
def tax_on_clothing (x_clothing : ℝ) := 0.04 * x_clothing
def tax_on_food := 0
def tax_on_other_items (x_other_items : ℝ) := 0.08 * x_other_items
def total_tax (a : ℝ) := 0.052 * a

-- The theorem we need to prove
theorem percentage_of_total_spent_on_other_items (a x_clothing x_food x_other_items : ℝ)
    (h1 : x_clothing = spent_on_clothing a)
    (h2 : x_food = spent_on_food a)
    (h3 : x_other_items = spent_on_other_items a x_clothing x_food)
    (h4 : tax_on_clothing x_clothing + tax_on_food + tax_on_other_items x_other_items = total_tax a) :
    0.40 * a = x_other_items :=
sorry

end percentage_of_total_spent_on_other_items_l2036_203640


namespace min_a_value_l2036_203681

theorem min_a_value (a : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x ≤ 1/2) → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end min_a_value_l2036_203681


namespace sum_of_integers_l2036_203641

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 180) : x + y = 28 := 
by
  sorry

end sum_of_integers_l2036_203641


namespace initial_eggs_proof_l2036_203620

-- Definitions based on the conditions provided
def initial_eggs := 7
def added_eggs := 4
def total_eggs := 11

-- The statement to be proved
theorem initial_eggs_proof : initial_eggs + added_eggs = total_eggs :=
by
  -- Placeholder for proof
  sorry

end initial_eggs_proof_l2036_203620


namespace largest_number_is_sqrt_7_l2036_203606

noncomputable def largest_root (d e f : ℝ) : ℝ :=
if d ≥ e ∧ d ≥ f then d else if e ≥ d ∧ e ≥ f then e else f

theorem largest_number_is_sqrt_7 :
  ∃ (d e f : ℝ), (d + e + f = 3) ∧ (d * e + d * f + e * f = -14) ∧ (d * e * f = 21) ∧ (largest_root d e f = Real.sqrt 7) :=
sorry

end largest_number_is_sqrt_7_l2036_203606


namespace paving_cost_correct_l2036_203623

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_m : ℝ := 300
def area (length : ℝ) (width : ℝ) : ℝ := length * width
def cost (area : ℝ) (rate : ℝ) : ℝ := area * rate

theorem paving_cost_correct :
  cost (area length width) rate_per_sq_m = 6187.50 :=
by
  sorry

end paving_cost_correct_l2036_203623


namespace series_sum_to_4_l2036_203697

theorem series_sum_to_4 (x : ℝ) (hx : ∑' n : ℕ, (n + 1) * x^n = 4) : x = 1 / 2 := 
sorry

end series_sum_to_4_l2036_203697


namespace negation_equiv_l2036_203603

-- Given problem conditions
def exists_real_x_lt_0 : Prop := ∃ x : ℝ, x^2 + 1 < 0

-- Mathematically equivalent proof problem statement
theorem negation_equiv :
  ¬exists_real_x_lt_0 ↔ ∀ x : ℝ, x^2 + 1 ≥ 0 :=
by
  sorry

end negation_equiv_l2036_203603


namespace arithmetic_sequence_common_difference_l2036_203659

theorem arithmetic_sequence_common_difference (a d : ℕ) (n : ℕ) :
  a = 5 →
  (a + (n - 1) * d = 50) →
  (n * (a + (a + (n - 1) * d)) / 2 = 275) →
  d = 5 := 
by
  intros ha ha_n hs_n
  sorry

end arithmetic_sequence_common_difference_l2036_203659


namespace area_triangle_CIN_l2036_203637

variables (A B C D M N I : Type*)

-- Definitions and assumptions
-- ABCD is a square
def is_square (ABCD : Type*) (side : ℝ) : Prop := sorry
-- M is the midpoint of AB
def midpoint_AB (M A B : Type*) : Prop := sorry
-- N is the midpoint of BC
def midpoint_BC (N B C : Type*) : Prop := sorry
-- Lines CM and DN intersect at I
def lines_intersect_at (C M D N I : Type*) : Prop := sorry

-- Goal
theorem area_triangle_CIN (ABCD : Type*) (side : ℝ) (M N C I : Type*) 
  (h1 : is_square ABCD side)
  (h2 : midpoint_AB M A B)
  (h3 : midpoint_BC N B C)
  (h4 : lines_intersect_at C M D N I) :
  sorry := sorry

end area_triangle_CIN_l2036_203637


namespace pow_evaluation_l2036_203601

theorem pow_evaluation (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end pow_evaluation_l2036_203601


namespace find_abc_l2036_203633

theorem find_abc (a b c : ℕ) (h1 : c = b^2) (h2 : (a + b + c)^3 - a^3 - b^3 - c^3 = 210) : a + b + c = 3 := 
by
  sorry

end find_abc_l2036_203633


namespace find_radius_of_inscribed_sphere_l2036_203609

variables (a b c s : ℝ)

theorem find_radius_of_inscribed_sphere
  (h1 : a + b + c = 18)
  (h2 : 2 * (a * b + b * c + c * a) = 216)
  (h3 : a^2 + b^2 + c^2 = 108) :
  s = 3 * Real.sqrt 3 :=
by
  sorry

end find_radius_of_inscribed_sphere_l2036_203609


namespace profit_calculation_l2036_203605

-- Definitions from conditions
def initial_shares := 20
def cost_per_share := 3
def sold_shares := 10
def sale_price_per_share := 4
def remaining_shares_value_multiplier := 2

-- Calculations based on conditions
def initial_cost := initial_shares * cost_per_share
def revenue_from_sold_shares := sold_shares * sale_price_per_share
def remaining_shares := initial_shares - sold_shares
def value_of_remaining_shares := remaining_shares * (cost_per_share * remaining_shares_value_multiplier)
def total_value := revenue_from_sold_shares + value_of_remaining_shares
def expected_profit := total_value - initial_cost

-- The problem statement to be proven
theorem profit_calculation : expected_profit = 40 := by
  -- Proof steps go here
  sorry

end profit_calculation_l2036_203605


namespace find_dividend_l2036_203618

-- Definitions based on conditions from the problem
def divisor : ℕ := 13
def quotient : ℕ := 17
def remainder : ℕ := 1

-- Statement of the proof problem
theorem find_dividend (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

-- Proof statement ensuring dividend is as expected
example : find_dividend divisor quotient remainder = 222 :=
by 
  sorry

end find_dividend_l2036_203618


namespace proposition_holds_for_all_positive_odd_numbers_l2036_203676

theorem proposition_holds_for_all_positive_odd_numbers
  (P : ℕ → Prop)
  (h1 : P 1)
  (h2 : ∀ k, k ≥ 1 → P k → P (k + 2)) :
  ∀ n, n % 2 = 1 → n ≥ 1 → P n :=
by
  sorry

end proposition_holds_for_all_positive_odd_numbers_l2036_203676


namespace original_gain_percentage_is_5_l2036_203695

def costPrice : ℝ := 200
def newCostPrice : ℝ := costPrice * 0.95
def desiredProfitRatio : ℝ := 0.10
def newSellingPrice : ℝ := newCostPrice * (1 + desiredProfitRatio)
def originalSellingPrice : ℝ := newSellingPrice + 1

theorem original_gain_percentage_is_5 :
  ((originalSellingPrice - costPrice) / costPrice) * 100 = 5 :=
by 
  sorry

end original_gain_percentage_is_5_l2036_203695


namespace total_amount_l2036_203680

theorem total_amount (x y z : ℝ) (hx : y = 45 / 0.45)
  (hy : z = (45 / 0.45) * 0.30)
  (hx_total : y = 45) :
  x + y + z = 175 :=
by
  -- Proof is omitted as per instructions
  sorry

end total_amount_l2036_203680


namespace shapes_fit_exactly_l2036_203658

-- Conditions: Shapes are drawn on a piece of paper and folded along a central bold line
def shapes_drawn_on_paper := true
def paper_folded_along_central_line := true

-- Define the main proof problem
theorem shapes_fit_exactly : shapes_drawn_on_paper ∧ paper_folded_along_central_line → 
  number_of_shapes_fitting_exactly_on_top = 3 :=
by
  intros h
  sorry

end shapes_fit_exactly_l2036_203658


namespace circle_condition_l2036_203636

theorem circle_condition (k : ℝ) : (∃ x y : ℝ, x^2 + y^2 + 4 * x + 2 * y + 4 * k + 1 = 0) → (k < 1) :=
by
  sorry

end circle_condition_l2036_203636


namespace jennifer_money_left_l2036_203651

variable (initial_amount : ℝ) (spent_sandwich_rate : ℝ) (spent_museum_rate : ℝ) (spent_book_rate : ℝ)

def money_left := initial_amount - (spent_sandwich_rate * initial_amount + spent_museum_rate * initial_amount + spent_book_rate * initial_amount)

theorem jennifer_money_left (h_initial : initial_amount = 150)
  (h_sandwich_rate : spent_sandwich_rate = 1/5)
  (h_museum_rate : spent_museum_rate = 1/6)
  (h_book_rate : spent_book_rate = 1/2) :
  money_left initial_amount spent_sandwich_rate spent_museum_rate spent_book_rate = 20 :=
by
  sorry

end jennifer_money_left_l2036_203651


namespace stock_price_after_two_years_l2036_203674

def initial_price : ℝ := 120

def first_year_increase (p : ℝ) : ℝ := p * 2

def second_year_decrease (p : ℝ) : ℝ := p * 0.30

def final_price (initial : ℝ) : ℝ :=
  let after_first_year := first_year_increase initial
  after_first_year - second_year_decrease after_first_year

theorem stock_price_after_two_years : final_price initial_price = 168 :=
by
  sorry

end stock_price_after_two_years_l2036_203674


namespace floor_neg_seven_over_four_l2036_203692

theorem floor_neg_seven_over_four : (Int.floor (-7 / 4 : ℚ)) = -2 := 
by
  sorry

end floor_neg_seven_over_four_l2036_203692


namespace number_of_possible_x_values_l2036_203696
noncomputable def triangle_sides_possible_values (x : ℕ) : Prop :=
  27 < x ∧ x < 63

theorem number_of_possible_x_values : 
  ∃ n, n = (62 - 28 + 1) ∧ ( ∀ x : ℕ, triangle_sides_possible_values x ↔ 28 ≤ x ∧ x ≤ 62) :=
sorry

end number_of_possible_x_values_l2036_203696


namespace minimum_value_of_polynomial_l2036_203645

def polynomial (x : ℝ) : ℝ := (12 - x) * (10 - x) * (12 + x) * (10 + x)

theorem minimum_value_of_polynomial : ∃ x : ℝ, polynomial x = -484 :=
by
  sorry

end minimum_value_of_polynomial_l2036_203645


namespace percentage_milk_in_B_l2036_203664

theorem percentage_milk_in_B :
  ∀ (A B C : ℕ),
  A = 1200 → B + C = A → B + 150 = C - 150 →
  (B:ℝ) / (A:ℝ) * 100 = 37.5 :=
by
  intros A B C hA hBC hE
  sorry

end percentage_milk_in_B_l2036_203664


namespace solution_set_of_inequality_l2036_203634

theorem solution_set_of_inequality (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x + 3| ≥ a) ↔ a ≤ 5 :=
sorry

end solution_set_of_inequality_l2036_203634


namespace circumcircle_diameter_l2036_203682

-- Given that the perimeter of triangle ABC is equal to 3 times the sum of the sines of its angles
-- and the Law of Sines holds for this triangle, we need to prove the diameter of the circumcircle is 3.
theorem circumcircle_diameter (a b c : ℝ) (A B C : ℝ) (R : ℝ)
  (h_perimeter : a + b + c = 3 * (Real.sin A + Real.sin B + Real.sin C))
  (h_law_of_sines : a / Real.sin A = 2 * R ∧ b / Real.sin B = 2 * R ∧ c / Real.sin C = 2 * R) :
  2 * R = 3 := 
by
  sorry

end circumcircle_diameter_l2036_203682


namespace exist_ints_a_b_l2036_203657

theorem exist_ints_a_b (n : ℕ) : (∃ a b : ℤ, (n : ℤ) + a^2 = b^2) ↔ ¬ n % 4 = 2 := 
by
  sorry

end exist_ints_a_b_l2036_203657


namespace prime_factorization_sum_l2036_203648

theorem prime_factorization_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : 13 * x^7 = 17 * y^11) : 
  a * e + b * f = 18 :=
by
  -- Let a and b be prime factors of x
  let a : ℕ := 17 -- prime factor found in the solution
  let e : ℕ := 1 -- exponent found for 17
  let b : ℕ := 0 -- no second prime factor
  let f : ℕ := 0 -- corresponding exponent

  sorry

end prime_factorization_sum_l2036_203648


namespace min_value_inverse_sum_l2036_203610

theorem min_value_inverse_sum {m n : ℝ} (h1 : -2 * m - 2 * n + 1 = 0) (h2 : m * n > 0) : 
  (1 / m + 1 / n) ≥ 8 :=
sorry

end min_value_inverse_sum_l2036_203610


namespace collapsed_buildings_l2036_203625

theorem collapsed_buildings (initial_collapse : ℕ) (collapse_one : initial_collapse = 4)
                            (collapse_double : ∀ n m, m = 2 * n) : (4 + 8 + 16 + 32 = 60) :=
by
  sorry

end collapsed_buildings_l2036_203625


namespace sum_of_numbers_l2036_203693

-- Definitions that come directly from the conditions
def product_condition (A B : ℕ) : Prop := A * B = 9375
def quotient_condition (A B : ℕ) : Prop := A / B = 15

-- Theorem that proves the sum of A and B is 400, based on the given conditions
theorem sum_of_numbers (A B : ℕ) (h1 : product_condition A B) (h2 : quotient_condition A B) : A + B = 400 :=
sorry

end sum_of_numbers_l2036_203693


namespace two_x_equals_y_l2036_203639

theorem two_x_equals_y (x y : ℝ) (h1 : (x + y) / 3 = 1) (h2 : x + 2 * y = 5) : 2 * x = y := 
by
  sorry

end two_x_equals_y_l2036_203639


namespace total_cards_proof_l2036_203613

-- Define the standard size of a deck of playing cards
def standard_deck_size : Nat := 52

-- Define the number of complete decks the shopkeeper has
def complete_decks : Nat := 6

-- Define the number of additional cards the shopkeeper has
def additional_cards : Nat := 7

-- Define the total number of cards from the complete decks
def total_deck_cards : Nat := complete_decks * standard_deck_size

-- Define the total number of all cards the shopkeeper has
def total_cards : Nat := total_deck_cards + additional_cards

-- The theorem statement that we need to prove
theorem total_cards_proof : total_cards = 319 := by
  sorry

end total_cards_proof_l2036_203613


namespace largest_root_polynomial_intersection_l2036_203654

/-
Given a polynomial P(x) = x^6 - 15x^5 + 74x^4 - 130x^3 + a * x^2 + b * x
and a line L(x) = c * x - 24,
such that P(x) stays above L(x) except at three distinct values of x where they intersect,
and one of the intersections is a root of triple multiplicity.
Prove that the largest value of x for which P(x) = L(x) is 6.
-/
theorem largest_root_polynomial_intersection (a b c : ℝ) (P L : ℝ → ℝ) (x : ℝ) :
  P x = x^6 - 15*x^5 + 74*x^4 - 130*x^3 + a*x^2 + b*x →
  L x = c*x - 24 →
  (∀ x, P x ≥ L x) ∨ (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ P x1 = L x1 ∧ P x2 = L x2 ∧ P x3 = L x3 ∧
  (∃ x0 : ℝ, x1 = x0 ∧ x2 = x0 ∧ x3 = x0 ∧ ∃ k : ℕ, k = 3)) →
  x = 6 :=
sorry

end largest_root_polynomial_intersection_l2036_203654


namespace fg_value_l2036_203619

def g (x : ℤ) : ℤ := 4 * x - 3
def f (x : ℤ) : ℤ := 6 * x + 2

theorem fg_value : f (g 5) = 104 := by
  sorry

end fg_value_l2036_203619


namespace least_possible_b_l2036_203653

open Nat

/-- 
  Given conditions:
  a and b are consecutive Fibonacci numbers with a > b,
  and their sum is 100 degrees.
  We need to prove that the least possible value of b is 21 degrees.
-/
theorem least_possible_b (a b : ℕ) (h1 : fib a = fib (b + 1))
  (h2 : a > b) (h3 : a + b = 100) : b = 21 :=
sorry

end least_possible_b_l2036_203653


namespace parabola_min_distance_a_l2036_203643

noncomputable def directrix_distance (P : Real × Real) (a : Real) : Real :=
abs (P.2 + 1 / (4 * a))

noncomputable def distance (P Q : Real × Real) : Real :=
Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem parabola_min_distance_a (a : Real) :
  (∀ (P : Real × Real), P.2 = a * P.1^2 → 
    distance P (2, 0) + directrix_distance P a = Real.sqrt 5) ↔ 
    a = 1 / 4 ∨ a = -1 / 4 :=
by
  sorry

end parabola_min_distance_a_l2036_203643


namespace dog_catches_fox_at_distance_l2036_203673

def initial_distance : ℝ := 30
def dog_leap_distance : ℝ := 2
def fox_leap_distance : ℝ := 1
def dog_leaps_per_time_unit : ℝ := 2
def fox_leaps_per_time_unit : ℝ := 3

noncomputable def dog_speed : ℝ := dog_leaps_per_time_unit * dog_leap_distance
noncomputable def fox_speed : ℝ := fox_leaps_per_time_unit * fox_leap_distance
noncomputable def relative_speed : ℝ := dog_speed - fox_speed
noncomputable def time_to_catch := initial_distance / relative_speed
noncomputable def distance_dog_runs := time_to_catch * dog_speed

theorem dog_catches_fox_at_distance :
  distance_dog_runs = 120 :=
  by sorry

end dog_catches_fox_at_distance_l2036_203673


namespace linear_function_m_l2036_203667

theorem linear_function_m (m : ℤ) (h₁ : |m| = 1) (h₂ : m + 1 ≠ 0) : m = 1 := by
  sorry

end linear_function_m_l2036_203667


namespace max_value_l2036_203668

noncomputable def f (x y : ℝ) : ℝ := 8 * x ^ 2 + 9 * x * y + 18 * y ^ 2 + 2 * x + 3 * y
noncomputable def g (x y : ℝ) : Prop := 4 * x ^ 2 + 9 * y ^ 2 = 8

theorem max_value : ∃ x y : ℝ, g x y ∧ f x y = 26 :=
by
  sorry

end max_value_l2036_203668


namespace complex_number_addition_identity_l2036_203660

-- Definitions of the conditions
def imaginary_unit (i : ℂ) := i^2 = -1

def complex_fraction_decomposition (a b : ℝ) (i : ℂ) := 
  (1 + i) / (1 - i) = a + b * i

-- The statement of the problem
theorem complex_number_addition_identity :
  ∃ (a b : ℝ) (i : ℂ), imaginary_unit i ∧ complex_fraction_decomposition a b i ∧ (a + b = 1) :=
sorry

end complex_number_addition_identity_l2036_203660


namespace choir_row_lengths_l2036_203628

theorem choir_row_lengths (x : ℕ) : 
  ((x ∈ [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) ∧ (90 % x = 0)) → (x = 5 ∨ x = 6 ∨ x = 9 ∨ x = 10 ∨ x = 15) :=
by
  intro h
  cases h
  sorry

end choir_row_lengths_l2036_203628


namespace tile_chessboard_2n_l2036_203631

theorem tile_chessboard_2n (n : ℕ) (board : Fin (2^n) → Fin (2^n) → Prop) (i j : Fin (2^n)) 
  (h : board i j = false) : ∃ tile : Fin (2^n) → Fin (2^n) → Bool, 
  (∀ i j, board i j = true ↔ tile i j = true) :=
sorry

end tile_chessboard_2n_l2036_203631


namespace jenna_more_than_four_times_martha_l2036_203672

noncomputable def problems : ℝ := 20
noncomputable def martha_problems : ℝ := 2
noncomputable def angela_problems : ℝ := 9
noncomputable def jenna_problems : ℝ := 6  -- We calculated J = 6 from the conditions
noncomputable def mark_problems : ℝ := jenna_problems / 2

theorem jenna_more_than_four_times_martha :
  (jenna_problems - 4 * martha_problems = 2) :=
by
  sorry

end jenna_more_than_four_times_martha_l2036_203672


namespace remaining_angle_obtuse_cut_by_right_is_acute_remaining_angle_straight_cut_by_acute_is_obtuse_l2036_203687

-- Definitions of angle types
def obtuse_angle (θ : ℝ) := θ > 90 ∧ θ < 180
def right_angle (θ : ℝ) := θ = 90
def acute_angle (θ : ℝ) := θ > 0 ∧ θ < 90
def straight_angle (θ : ℝ) := θ = 180

-- Proposition 1: Remaining angle when an obtuse angle is cut by a right angle is acute
theorem remaining_angle_obtuse_cut_by_right_is_acute (θ : ℝ) (φ : ℝ) 
    (h1 : obtuse_angle θ) (h2 : right_angle φ) : acute_angle (θ - φ) :=
  sorry

-- Proposition 2: Remaining angle when a straight angle is cut by an acute angle is obtuse
theorem remaining_angle_straight_cut_by_acute_is_obtuse (α : ℝ) (β : ℝ) 
    (h1 : straight_angle α) (h2 : acute_angle β) : obtuse_angle (α - β) :=
  sorry

end remaining_angle_obtuse_cut_by_right_is_acute_remaining_angle_straight_cut_by_acute_is_obtuse_l2036_203687


namespace harriet_forward_speed_proof_l2036_203646

def harriet_forward_time : ℝ := 3 -- forward time in hours
def harriet_return_speed : ℝ := 150 -- return speed in km/h
def harriet_total_time : ℝ := 5 -- total trip time in hours

noncomputable def harriet_forward_speed : ℝ :=
  let distance := harriet_return_speed * (harriet_total_time - harriet_forward_time)
  distance / harriet_forward_time

theorem harriet_forward_speed_proof : harriet_forward_speed = 100 := by
  sorry

end harriet_forward_speed_proof_l2036_203646


namespace polygon_triangle_division_l2036_203644

theorem polygon_triangle_division (n k : ℕ) (h : k * 3 = n * 3 - 6) : k ≥ n - 2 := sorry

end polygon_triangle_division_l2036_203644


namespace trigonometric_identity_proof_l2036_203652

variable (α β : Real)

theorem trigonometric_identity_proof :
  4.28 * Real.sin (β / 2 - Real.pi / 2) ^ 2 - Real.cos (α - 3 * Real.pi / 2) ^ 2 = 
  Real.cos (α + β) * Real.cos (α - β) :=
by
  sorry

end trigonometric_identity_proof_l2036_203652


namespace differentiable_increasing_necessary_but_not_sufficient_l2036_203635

variable {f : ℝ → ℝ}

theorem differentiable_increasing_necessary_but_not_sufficient (h_diff : ∀ x : ℝ, DifferentiableAt ℝ f x) :
  (∀ x : ℝ, 0 < deriv f x) → ∀ x : ℝ, MonotoneOn f (Set.univ : Set ℝ) ∧ ¬ (∀ x : ℝ, MonotoneOn f (Set.univ : Set ℝ) → ∀ x : ℝ, 0 < deriv f x) := 
sorry

end differentiable_increasing_necessary_but_not_sufficient_l2036_203635


namespace find_x_minus_y_l2036_203626

variables (x y : ℚ)

theorem find_x_minus_y
  (h1 : 3 * x - 4 * y = 17)
  (h2 : x + 3 * y = 1) :
  x - y = 69 / 13 := 
sorry

end find_x_minus_y_l2036_203626


namespace debra_probability_l2036_203698

theorem debra_probability :
  let p_tail := 1 / 2
  let p_head := 1 / 2
  let p_THTHT := p_tail * p_head * p_tail * p_head * p_tail
  let P := (1 / 4) / (3 / 4)
  (p_THTHT * P) = 1 / 96 :=
by
  -- Definitions of p_tail, p_head, p_THTHT, and P
  let p_tail := 1 / 2
  let p_head := 1 / 2
  let p_THTHT := p_tail * p_head * p_tail * p_head * p_tail
  let P := (1 / 4) / (3 / 4)
  -- Placeholder for proof computation
  sorry

end debra_probability_l2036_203698


namespace triangle_ratio_l2036_203665

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ)
  (hA : A = 2 * Real.pi / 3)
  (h_a : a = Real.sqrt 3 * c)
  (h_angle_sum : A + B + C = Real.pi)
  (h_law_of_sines : a / Real.sin A = c / Real.sin C) :
  b / c = 1 :=
sorry

end triangle_ratio_l2036_203665


namespace calculate_N_l2036_203670

theorem calculate_N (h : (25 / 100) * N = (55 / 100) * 3010) : N = 6622 :=
by
  sorry

end calculate_N_l2036_203670


namespace slope_of_CD_l2036_203685

-- Given circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 15 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 48 = 0

-- Define the line whose slope needs to be found
def line (x y : ℝ) : Prop := 22*x - 12*y - 33 = 0

-- State the proof problem
theorem slope_of_CD : ∀ x y : ℝ, circle1 x y → circle2 x y → line x y ∧ (∃ m : ℝ, m = 11/6) :=
by sorry

end slope_of_CD_l2036_203685


namespace gcd_50421_35343_l2036_203662

theorem gcd_50421_35343 : Int.gcd 50421 35343 = 23 := by
  sorry

end gcd_50421_35343_l2036_203662


namespace cows_count_24_l2036_203602

-- Declare the conditions as given in the problem.
variables (D C : Nat)

-- Define the total number of legs and heads and the given condition.
def total_legs := 2 * D + 4 * C
def total_heads := D + C
axiom condition : total_legs = 2 * total_heads + 48

-- The goal is to prove that the number of cows C is 24.
theorem cows_count_24 : C = 24 :=
by
  sorry

end cows_count_24_l2036_203602


namespace blackboard_problem_l2036_203600

theorem blackboard_problem (n : ℕ) (h_pos : 0 < n) :
  ∃ x, (∀ (t : ℕ), t < n - 1 → ∃ a b : ℕ, a + b + 2 * (t + 1) = n + 1 ∧ a > 0 ∧ b > 0) → 
  x ≥ 2 ^ ((4 * n ^ 2 - 4) / 3) :=
by
  sorry

end blackboard_problem_l2036_203600


namespace sum_of_possible_values_of_a_l2036_203656

theorem sum_of_possible_values_of_a :
  (∀ r s : ℤ, r + s = a ∧ r * s = 3 * a) → ∃ a : ℤ, (a = 12) :=
by
  sorry

end sum_of_possible_values_of_a_l2036_203656


namespace Lizzie_group_difference_l2036_203615

theorem Lizzie_group_difference
  (lizzie_group_members : ℕ)
  (total_members : ℕ)
  (lizzie_more_than_other : lizzie_group_members > total_members - lizzie_group_members)
  (lizzie_members_eq : lizzie_group_members = 54)
  (total_members_eq : total_members = 91)
  : lizzie_group_members - (total_members - lizzie_group_members) = 17 := 
sorry

end Lizzie_group_difference_l2036_203615


namespace not_p_or_not_q_implies_p_and_q_and_p_or_q_l2036_203650

variable (p q : Prop)

theorem not_p_or_not_q_implies_p_and_q_and_p_or_q (h : ¬ (¬ p ∨ ¬ q)) : (p ∧ q) ∧ (p ∨ q) :=
sorry

end not_p_or_not_q_implies_p_and_q_and_p_or_q_l2036_203650


namespace hazel_walked_distance_l2036_203691

theorem hazel_walked_distance
  (first_hour_distance : ℕ)
  (second_hour_distance : ℕ)
  (h1 : first_hour_distance = 2)
  (h2 : second_hour_distance = 2 * first_hour_distance) :
  (first_hour_distance + second_hour_distance = 6) :=
by {
  sorry
}

end hazel_walked_distance_l2036_203691


namespace general_formula_sum_of_b_l2036_203679

variable {a : ℕ → ℕ} (b : ℕ → ℕ)
variable (n : ℕ)

def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n : ℕ, a (n+2) = q * a (n+1)

def initial_conditions (a : ℕ → ℕ) : Prop :=
  a 1 + a 2 = 9 ∧ a 2 + a 3 = 18

theorem general_formula (q : ℕ) (h1 : is_geometric_sequence a q) (h2 : initial_conditions a) :
  a n = 3 * 2^(n - 1) :=
sorry

def b_n (a : ℕ → ℕ) (n : ℕ) : ℕ := a n + 2 * n

def sum_b (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum b

theorem sum_of_b (h1 : ∀ m : ℕ, b m = a m + 2 * m) (h2 : initial_conditions a) :
  sum_b b n = 3 * 2^n + n * (n + 1) - 3 :=
sorry

end general_formula_sum_of_b_l2036_203679


namespace friends_recycled_pounds_l2036_203624

-- Definitions for the given conditions
def pounds_per_point : ℕ := 4
def paige_recycled : ℕ := 14
def total_points : ℕ := 4

-- The proof statement
theorem friends_recycled_pounds :
  ∃ p_friends : ℕ, 
  (paige_recycled / pounds_per_point) + (p_friends / pounds_per_point) = total_points 
  → p_friends = 4 := 
sorry

end friends_recycled_pounds_l2036_203624


namespace find_Y_l2036_203678

theorem find_Y :
  ∃ Y : ℤ, (19 + Y / 151) * 151 = 2912 ∧ Y = 43 :=
by
  use 43
  sorry

end find_Y_l2036_203678


namespace equation_solutions_l2036_203655

theorem equation_solutions
  (a : ℝ) :
  (∃ x : ℝ, (1 < a ∧ a < 2) ∧ (x = (1 - a) / a ∨ x = -1)) ∨
  (a = 2 ∧ (∃ x : ℝ, x = -1 ∨ x = -1/2)) ∨
  (a > 2 ∧ (∃ x : ℝ, x = (1 - a) / a ∨ x = -1 ∨ x = 1 - a)) ∨
  (0 ≤ a ∧ a ≤ 1 ∧ (∃ x : ℝ, x = -1)) ∨
  (a < 0 ∧ (∃ x : ℝ, x = (1 - a) / a ∨ x = -1)) := sorry

end equation_solutions_l2036_203655


namespace correct_calculation_l2036_203666

theorem correct_calculation (a b : ℝ) : 
  ¬(a * a^3 = a^3) ∧ ¬((a^2)^3 = a^5) ∧ (-a^2 * b)^2 = a^4 * b^2 ∧ ¬(a^3 / a = a^3) :=
by {
  sorry
}

end correct_calculation_l2036_203666


namespace kevin_eggs_l2036_203677

theorem kevin_eggs : 
  ∀ (bonnie george cheryl kevin : ℕ),
  bonnie = 13 → 
  george = 9 → 
  cheryl = 56 → 
  cheryl = bonnie + george + kevin + 29 →
  kevin = 5 :=
by
  intros bonnie george cheryl kevin h_bonnie h_george h_cheryl h_eqn
  subst h_bonnie
  subst h_george
  subst h_cheryl
  simp at h_eqn
  sorry

end kevin_eggs_l2036_203677


namespace first_year_with_sum_15_l2036_203689

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) + (n % 1000 / 100) + (n % 100 / 10) + (n % 10)

theorem first_year_with_sum_15 : ∃ y > 2100, sum_of_digits y = 15 :=
  sorry

end first_year_with_sum_15_l2036_203689


namespace angle_with_same_terminal_side_l2036_203699

-- Given conditions in the problem: angles to choose from
def angles : List ℕ := [60, 70, 100, 130]

-- Definition of the equivalence relation (angles having the same terminal side)
def same_terminal_side (θ α : ℕ) : Prop :=
  ∃ k : ℤ, θ = α + k * 360

-- Proof goal: 420° has the same terminal side as one of the angles in the list
theorem angle_with_same_terminal_side :
  ∃ α ∈ angles, same_terminal_side 420 α :=
sorry  -- proof not required

end angle_with_same_terminal_side_l2036_203699


namespace sin_cos_alpha_l2036_203647

open Real

theorem sin_cos_alpha (α : ℝ) (h1 : sin (2 * α) = -sqrt 2 / 2) (h2 : α ∈ Set.Ioc (3 * π / 2) (2 * π)) :
  sin α + cos α = sqrt 2 / 2 :=
sorry

end sin_cos_alpha_l2036_203647


namespace sum_of_three_integers_with_product_of_5_cubed_l2036_203608

theorem sum_of_three_integers_with_product_of_5_cubed :
  ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  a * b * c = 5^3 ∧ 
  a + b + c = 31 :=
by
  sorry

end sum_of_three_integers_with_product_of_5_cubed_l2036_203608


namespace height_of_prism_l2036_203690

-- Definitions based on conditions
def Volume : ℝ := 120
def edge1 : ℝ := 3
def edge2 : ℝ := 4
def BaseArea : ℝ := edge1 * edge2

-- Define the problem statement
theorem height_of_prism (h : ℝ) : (BaseArea * h / 2 = Volume) → (h = 20) :=
by
  intro h_value
  have Volume_equiv : h = 2 * Volume / BaseArea := sorry
  sorry

end height_of_prism_l2036_203690


namespace ellipse_hyperbola_tangent_m_eq_l2036_203629

variable (x y m : ℝ)

def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 2)^2 = 1
def curves_tangent (x m : ℝ) : Prop := ∃ y, ellipse x y ∧ hyperbola x y m

theorem ellipse_hyperbola_tangent_m_eq :
  (∃ x, curves_tangent x (12/13)) ↔ true := 
by
  sorry

end ellipse_hyperbola_tangent_m_eq_l2036_203629


namespace hyperbola_asymptotes_l2036_203688

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (x^2 - (y^2 / 9) = 1) → (y = 3 * x ∨ y = -3 * x) :=
by
  -- conditions and theorem to prove
  sorry

end hyperbola_asymptotes_l2036_203688


namespace boys_attended_dance_l2036_203627

variables (d b : ℕ)

theorem boys_attended_dance (h1 : b = 2 * d) (h2 : b = (d - 1) + 8) : b = 14 :=
by sorry

end boys_attended_dance_l2036_203627


namespace complement_of_intersection_l2036_203638

def U : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {1, 2, 3}
def intersection : Set ℕ := M ∩ N
def complement : Set ℕ := U \ intersection

theorem complement_of_intersection (U M N : Set ℕ) :
  U = {0, 1, 2, 3} →
  M = {0, 1, 2} →
  N = {1, 2, 3} →
  (U \ (M ∩ N)) = {0, 3} := by
  intro hU hM hN
  simp [hU, hM, hN]
  sorry

end complement_of_intersection_l2036_203638


namespace suresh_investment_correct_l2036_203671

noncomputable def suresh_investment
  (ramesh_investment : ℝ)
  (total_profit : ℝ)
  (ramesh_profit_share : ℝ)
  : ℝ := sorry

theorem suresh_investment_correct
  (ramesh_investment : ℝ := 40000)
  (total_profit : ℝ := 19000)
  (ramesh_profit_share : ℝ := 11875)
  : suresh_investment ramesh_investment total_profit ramesh_profit_share = 24000 := sorry

end suresh_investment_correct_l2036_203671


namespace tan_ratio_l2036_203649

theorem tan_ratio (p q : ℝ) 
  (h1: Real.sin (p + q) = 5 / 8)
  (h2: Real.sin (p - q) = 3 / 8) : Real.tan p / Real.tan q = 4 := 
by
  sorry

end tan_ratio_l2036_203649


namespace simplify_expression_l2036_203604

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : ( (3 * x + 6 - 5 * x) / 3 ) = ( (-2 * x) / 3 + 2 ) :=
by
  sorry

end simplify_expression_l2036_203604


namespace proof_f_g_l2036_203669

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5*x - 1
def g (x : ℝ) : ℝ := 2*x + 3

theorem proof_f_g (x : ℝ) : f (g 2) - g (f 2) = 258 :=
by
  sorry

end proof_f_g_l2036_203669


namespace roots_of_quadratic_implies_values_l2036_203684

theorem roots_of_quadratic_implies_values (a b : ℝ) :
  (∃ x : ℝ, x^2 + 2 * (1 + a) * x + (3 * a^2 + 4 * a * b + 4 * b^2 + 2) = 0) →
  a = 1 ∧ b = -1/2 :=
by
  sorry

end roots_of_quadratic_implies_values_l2036_203684


namespace zero_function_l2036_203612

noncomputable def f : ℝ → ℝ := sorry

theorem zero_function :
  (∀ x y : ℝ, f x + f y = f (f x * f y)) → (∀ x : ℝ, f x = 0) :=
by
  intro h
  sorry

end zero_function_l2036_203612


namespace ladybugs_with_spots_l2036_203694

theorem ladybugs_with_spots (total_ladybugs : ℕ) (ladybugs_without_spots : ℕ) : total_ladybugs = 67082 ∧ ladybugs_without_spots = 54912 → total_ladybugs - ladybugs_without_spots = 12170 := by
  sorry

end ladybugs_with_spots_l2036_203694


namespace find_number_l2036_203632

noncomputable def some_number : ℝ :=
  0.27712 / 9.237333333333334

theorem find_number :
  (69.28 * 0.004) / some_number = 9.237333333333334 :=
by 
  sorry

end find_number_l2036_203632


namespace complete_the_square_1_complete_the_square_2_complete_the_square_3_l2036_203630

theorem complete_the_square_1 (x : ℝ) : 
  (x^2 - 2 * x + 3) = (x - 1)^2 + 2 :=
sorry

theorem complete_the_square_2 (x : ℝ) : 
  (3 * x^2 + 6 * x - 1) = 3 * (x + 1)^2 - 4 :=
sorry

theorem complete_the_square_3 (x : ℝ) : 
  (-2 * x^2 + 3 * x - 2) = -2 * (x - 3 / 4)^2 - 7 / 8 :=
sorry

end complete_the_square_1_complete_the_square_2_complete_the_square_3_l2036_203630


namespace complement_union_and_complement_intersect_l2036_203683

-- Definitions of sets according to the problem conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }

-- The correct answers derived in the solution
def complement_union_A_B : Set ℝ := { x | x ≤ 2 ∨ 10 ≤ x }
def complement_A_intersect_B : Set ℝ := { x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) }

-- Statement of the mathematically equivalent proof problem
theorem complement_union_and_complement_intersect:
  (Set.compl (A ∪ B) = complement_union_A_B) ∧ 
  ((Set.compl A) ∩ B = complement_A_intersect_B) :=
  by 
    sorry

end complement_union_and_complement_intersect_l2036_203683


namespace solid_has_identical_views_is_sphere_or_cube_l2036_203663

-- Define the conditions for orthographic projections being identical
def identical_views_in_orthographic_projections (solid : Type) : Prop :=
  sorry -- Assume the logic for checking identical orthographic projections is defined

-- Define the types for sphere and cube
structure Sphere : Type := 
  (radius : ℝ)

structure Cube : Type := 
  (side_length : ℝ)

-- The main statement to prove
theorem solid_has_identical_views_is_sphere_or_cube (solid : Type) 
  (h : identical_views_in_orthographic_projections solid) : 
  solid = Sphere ∨ solid = Cube :=
by 
  sorry -- The detailed proof is omitted

end solid_has_identical_views_is_sphere_or_cube_l2036_203663


namespace count_lineups_not_last_l2036_203611

theorem count_lineups_not_last (n : ℕ) (htallest_not_last : n = 5) :
  ∃ (k : ℕ), k = 96 :=
by { sorry }

end count_lineups_not_last_l2036_203611


namespace sum_of_squares_of_consecutive_integers_l2036_203617

theorem sum_of_squares_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x^2 + (x + 1)^2 = 1625 := by
  sorry

end sum_of_squares_of_consecutive_integers_l2036_203617


namespace intersection_of_A_and_B_union_of_A_and_B_complement_of_A_with_respect_to_U_l2036_203661

open Set

universe u

def U : Set ℝ := univ

def A : Set ℝ := { x | x ≥ 1 ∨ x ≤ -3 }

def B : Set ℝ := { x | -4 < x ∧ x < 0 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -4 < x ∧ x ≤ -3 } :=
by sorry

theorem union_of_A_and_B :
  A ∪ B = { x | x < 0 ∨ x ≥ 1 } :=
by sorry

theorem complement_of_A_with_respect_to_U :
  U \ A = { x | -3 < x ∧ x < 1 } :=
by sorry

end intersection_of_A_and_B_union_of_A_and_B_complement_of_A_with_respect_to_U_l2036_203661


namespace ratio_lcm_gcf_280_476_l2036_203621

theorem ratio_lcm_gcf_280_476 : 
  let a := 280
  let b := 476
  let lcm_ab := Nat.lcm a b
  let gcf_ab := Nat.gcd a b
  lcm_ab / gcf_ab = 170 := by
  sorry

end ratio_lcm_gcf_280_476_l2036_203621


namespace son_l2036_203614

theorem son's_age (S M : ℕ) (h1 : M = S + 20) (h2 : M + 2 = 2 * (S + 2)) : S = 18 := by
  sorry

end son_l2036_203614


namespace greatest_identical_snack_bags_l2036_203616

-- Defining the quantities of each type of snack
def granola_bars : Nat := 24
def dried_fruit : Nat := 36
def nuts : Nat := 60

-- Statement of the problem: greatest number of identical snack bags Serena can make without any food left over.
theorem greatest_identical_snack_bags :
  Nat.gcd (Nat.gcd granola_bars dried_fruit) nuts = 12 :=
sorry

end greatest_identical_snack_bags_l2036_203616


namespace solve_quadratic_eq_l2036_203607

theorem solve_quadratic_eq (x y : ℝ) :
  (x = 3 ∧ y = 1) ∨ (x = -1 ∧ y = 1) ∨ (x = 3 ∧ y = -1) ∨ (x = -3 ∧ y = -1) ∨ (x = -1 ∧ y = -5) ↔
  x ^ 2 - x * y + y ^ 2 - x + 3 * y - 7 = 0 := sorry

end solve_quadratic_eq_l2036_203607


namespace abs_gt_two_l2036_203686

theorem abs_gt_two (x : ℝ) : |x| > 2 → x > 2 ∨ x < -2 :=
by
  intros
  sorry

end abs_gt_two_l2036_203686
