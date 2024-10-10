import Mathlib

namespace point_outside_circle_iff_a_in_range_l30_3024

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 - x + y + a = 0

-- Define what it means for a point to be outside the circle
def point_outside_circle (x y a : ℝ) : Prop := x^2 + y^2 - x + y + a > 0

-- Theorem statement
theorem point_outside_circle_iff_a_in_range :
  ∀ a : ℝ, point_outside_circle 2 1 a ↔ -4 < a ∧ a < 1/2 := by sorry

end point_outside_circle_iff_a_in_range_l30_3024


namespace final_donut_count_l30_3015

def donutsRemaining (initial : ℕ) : ℕ :=
  let afterBill := initial - 2
  let afterSecretary := afterBill - 4
  let afterManager := afterSecretary - (afterSecretary / 10)
  let afterFirstGroup := afterManager - (afterManager / 3)
  afterFirstGroup - (afterFirstGroup / 2)

theorem final_donut_count :
  donutsRemaining 50 = 14 :=
by sorry

end final_donut_count_l30_3015


namespace wills_jogging_time_l30_3052

/-- Calculates the jogging time given initial calories, burn rate, and final calories -/
def joggingTime (initialCalories : ℕ) (burnRate : ℕ) (finalCalories : ℕ) : ℕ :=
  (initialCalories - finalCalories) / burnRate

/-- Theorem stating that Will's jogging time is 30 minutes -/
theorem wills_jogging_time :
  let initialCalories : ℕ := 900
  let burnRate : ℕ := 10
  let finalCalories : ℕ := 600
  joggingTime initialCalories burnRate finalCalories = 30 := by
  sorry

end wills_jogging_time_l30_3052


namespace value_of_k_l30_3027

theorem value_of_k : ∃ k : ℚ, (k + 2) / 4 - (2 * k - 1) / 6 = 1 ∧ k = -4 := by
  sorry

end value_of_k_l30_3027


namespace max_value_of_g_l30_3090

def g (x : ℝ) : ℝ := 5 * x - x^5

theorem max_value_of_g :
  ∃ (max : ℝ), max = 4 ∧
  ∀ x : ℝ, 0 ≤ x → x ≤ Real.sqrt 5 → g x ≤ max :=
sorry

end max_value_of_g_l30_3090


namespace vector_subtraction_l30_3017

/-- Given two vectors in ℝ³, prove that their difference is equal to a specific vector. -/
theorem vector_subtraction (a b : ℝ × ℝ × ℝ) :
  a = (1, -2, 1) →
  b = (1, 0, 2) →
  a - b = (0, -2, -1) := by
  sorry

end vector_subtraction_l30_3017


namespace cells_after_3_hours_l30_3016

/-- Represents the number of cells after a given number of divisions -/
def cells (n : ℕ) : ℕ := 2^n

/-- The number of divisions that occur in 3 hours -/
def divisions_in_3_hours : ℕ := 3 * 2

theorem cells_after_3_hours : cells divisions_in_3_hours = 64 := by
  sorry

#eval cells divisions_in_3_hours

end cells_after_3_hours_l30_3016


namespace hotel_air_conditioning_l30_3034

theorem hotel_air_conditioning (total_rooms : ℚ) : 
  total_rooms > 0 →
  (3 / 4 : ℚ) * total_rooms + (1 / 4 : ℚ) * total_rooms = total_rooms →
  (3 / 5 : ℚ) * total_rooms = total_rooms * (3 / 5 : ℚ) →
  (2 / 3 : ℚ) * ((3 / 5 : ℚ) * total_rooms) = (2 / 5 : ℚ) * total_rooms →
  let rented_rooms := (3 / 4 : ℚ) * total_rooms
  let non_rented_rooms := total_rooms - rented_rooms
  let ac_rooms := (3 / 5 : ℚ) * total_rooms
  let rented_ac_rooms := (2 / 5 : ℚ) * total_rooms
  let non_rented_ac_rooms := ac_rooms - rented_ac_rooms
  (non_rented_ac_rooms / non_rented_rooms) * 100 = 80 := by
sorry


end hotel_air_conditioning_l30_3034


namespace complex_modulus_one_l30_3023

theorem complex_modulus_one (a : ℝ) :
  let z : ℂ := (a - 1) + a * Complex.I
  Complex.abs z = 1 → a = 1 := by
sorry

end complex_modulus_one_l30_3023


namespace largest_inscribed_equilateral_triangle_area_l30_3096

/-- The area of the largest equilateral triangle inscribed in a circle of radius 10 -/
theorem largest_inscribed_equilateral_triangle_area (r : ℝ) (h : r = 10) :
  let s := r * (3 / Real.sqrt 3)
  let area := (s^2 * Real.sqrt 3) / 4
  area = 75 * Real.sqrt 3 := by
  sorry

end largest_inscribed_equilateral_triangle_area_l30_3096


namespace intersection_sum_l30_3078

theorem intersection_sum (a b : ℝ) : 
  (3 = (1/3) * 6 + a) ∧ (6 = (1/3) * 3 + b) → a + b = 6 := by
  sorry

end intersection_sum_l30_3078


namespace parallelogram_side_length_l30_3041

theorem parallelogram_side_length 
  (s : ℝ) 
  (side1 : ℝ) 
  (side2 : ℝ) 
  (angle : ℝ) 
  (area : ℝ) 
  (h1 : side1 = 3 * s) 
  (h2 : side2 = s) 
  (h3 : angle = π / 3) -- 60 degrees in radians
  (h4 : area = 9 * Real.sqrt 3) 
  (h5 : area = side1 * side2 * Real.sin angle) : 
  s = Real.sqrt 3 := by
sorry

end parallelogram_side_length_l30_3041


namespace percentage_problem_l30_3089

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 800 →
  0.4 * N = (P / 100) * 650 + 190 →
  P = 20 := by sorry

end percentage_problem_l30_3089


namespace sine_sum_problem_l30_3046

theorem sine_sum_problem (α : Real) (h1 : α ∈ Set.Ioo 0 π) (h2 : Real.tan (α - π/4) = 1/3) :
  Real.sin (π/4 + α) = (3 * Real.sqrt 10) / 10 := by
  sorry

end sine_sum_problem_l30_3046


namespace equation_solutions_l30_3084

def equation (x : ℝ) : Prop :=
  1 / (x^2 + 14*x - 36) + 1 / (x^2 + 5*x - 14) + 1 / (x^2 - 16*x - 36) = 0

theorem equation_solutions :
  {x : ℝ | equation x} = {9, -4, 12, 3} := by sorry

end equation_solutions_l30_3084


namespace height_difference_ruby_xavier_l30_3077

-- Constants and conversion factors
def inch_to_cm : ℝ := 2.54
def m_to_cm : ℝ := 100

-- Given heights and relationships
def janet_height_inch : ℝ := 62.75
def charlene_height_factor : ℝ := 1.5
def pablo_charlene_diff_m : ℝ := 1.85
def ruby_pablo_diff_cm : ℝ := 0.5
def xavier_charlene_diff_m : ℝ := 2.13
def paul_xavier_diff_cm : ℝ := 97.75
def paul_ruby_diff_m : ℝ := 0.5

-- Theorem statement
theorem height_difference_ruby_xavier :
  let janet_height_cm := janet_height_inch * inch_to_cm
  let charlene_height_cm := charlene_height_factor * janet_height_cm
  let pablo_height_cm := charlene_height_cm + pablo_charlene_diff_m * m_to_cm
  let ruby_height_cm := pablo_height_cm - ruby_pablo_diff_cm
  let xavier_height_cm := charlene_height_cm + xavier_charlene_diff_m * m_to_cm
  let paul_height_cm := ruby_height_cm + paul_ruby_diff_m * m_to_cm
  let height_diff_cm := xavier_height_cm - ruby_height_cm
  let height_diff_inch := height_diff_cm / inch_to_cm
  ∃ ε > 0, |height_diff_inch - 18.78| < ε :=
by
  sorry

end height_difference_ruby_xavier_l30_3077


namespace work_completion_time_l30_3044

/-- The time taken for two workers to complete three times a piece of work -/
def time_to_complete_work (aarti_rate : ℚ) (bina_rate : ℚ) : ℚ :=
  3 / (aarti_rate + bina_rate)

/-- Theorem stating that Aarti and Bina will take approximately 9.23 days to complete three times the work -/
theorem work_completion_time :
  let aarti_rate : ℚ := 1 / 5
  let bina_rate : ℚ := 1 / 8
  abs (time_to_complete_work aarti_rate bina_rate - 9.23) < 0.01 := by
  sorry

end work_completion_time_l30_3044


namespace find_set_C_l30_3008

def A : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 6 = 0}

theorem find_set_C : 
  ∃ C : Set ℝ, 
    (C = {0, 2, 3}) ∧ 
    (∀ a : ℝ, a ∈ C ↔ (A ∪ B a = A)) :=
by sorry

end find_set_C_l30_3008


namespace positive_combination_l30_3049

theorem positive_combination (x y : ℝ) (h1 : x + y > 0) (h2 : 4 * x + y > 0) : 
  8 * x + 5 * y > 0 := by
  sorry

end positive_combination_l30_3049


namespace no_fraction_value_l30_3047

-- Define the No operator
def No : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * No n

-- State the theorem
theorem no_fraction_value :
  (No 2022) / (No 2023) = 1 / 2023 := by sorry

end no_fraction_value_l30_3047


namespace polynomial_factorization_l30_3038

theorem polynomial_factorization (y : ℝ) :
  1 + 5*y^2 + 25*y^4 + 125*y^6 + 625*y^8 = 
  (5*y^2 + ((5+Real.sqrt 5)*y)/2 + 1) * 
  (5*y^2 + ((5-Real.sqrt 5)*y)/2 + 1) * 
  (5*y^2 - ((5+Real.sqrt 5)*y)/2 + 1) * 
  (5*y^2 - ((5-Real.sqrt 5)*y)/2 + 1) := by
  sorry

end polynomial_factorization_l30_3038


namespace polynomial_simplification_l30_3091

theorem polynomial_simplification (x : ℝ) :
  (x^3 + 4*x^2 - 7*x + 11) + (-4*x^4 - x^3 + x^2 + 7*x + 3) = -4*x^4 + 5*x^2 + 14 :=
by sorry

end polynomial_simplification_l30_3091


namespace problem_statement_l30_3057

theorem problem_statement (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end problem_statement_l30_3057


namespace three_digit_number_property_l30_3003

theorem three_digit_number_property : 
  ∃ (N : ℕ), 
    (100 ≤ N ∧ N < 1000) ∧ 
    (N % 11 = 0) ∧ 
    (N / 11 = (N / 100)^2 + ((N / 10) % 10)^2 + (N % 10)^2) ∧
    (N = 550 ∨ N = 803) ∧
    (∀ (M : ℕ), 
      (100 ≤ M ∧ M < 1000) ∧ 
      (M % 11 = 0) ∧ 
      (M / 11 = (M / 100)^2 + ((M / 10) % 10)^2 + (M % 10)^2) →
      (M = 550 ∨ M = 803)) :=
by sorry

end three_digit_number_property_l30_3003


namespace coin_game_expected_value_l30_3025

/-- A modified coin game with three outcomes --/
structure CoinGame where
  prob_heads : ℝ
  prob_tails : ℝ
  prob_edge : ℝ
  payoff_heads : ℝ
  payoff_tails : ℝ
  payoff_edge : ℝ

/-- Calculate the expected value of the coin game --/
def expected_value (game : CoinGame) : ℝ :=
  game.prob_heads * game.payoff_heads +
  game.prob_tails * game.payoff_tails +
  game.prob_edge * game.payoff_edge

/-- Theorem stating the expected value of the specific coin game --/
theorem coin_game_expected_value :
  let game : CoinGame := {
    prob_heads := 1/4,
    prob_tails := 1/2,
    prob_edge := 1/4,
    payoff_heads := 4,
    payoff_tails := -3,
    payoff_edge := 0
  }
  expected_value game = -1/2 := by
  sorry


end coin_game_expected_value_l30_3025


namespace mooncake_packing_l30_3097

theorem mooncake_packing :
  ∃ (x y : ℕ), 
    9 * x + 4 * y = 35 ∧ 
    (∀ (a b : ℕ), 9 * a + 4 * b = 35 → x + y ≤ a + b) ∧
    x + y = 5 := by
  sorry

end mooncake_packing_l30_3097


namespace unreachable_target_l30_3070

/-- A permutation of the first 100 natural numbers -/
def Permutation := Fin 100 → Fin 100

/-- The initial sequence 1, 2, 3, ..., 99, 100 -/
def initial : Permutation := fun i => i + 1

/-- The target sequence 100, 99, 98, ..., 2, 1 -/
def target : Permutation := fun i => 100 - i

/-- A valid swap in the sequence -/
def validSwap (p : Permutation) (i j : Fin 100) : Prop :=
  ∃ k, i < k ∧ k < j ∧ j = i + 2 ∧
    (∀ m, m ≠ i ∧ m ≠ j → p m = p m) ∧
    p i = p j ∧ p j = p i

/-- A sequence that can be obtained from the initial sequence using valid swaps -/
inductive reachable : Permutation → Prop
  | init : reachable initial
  | swap : ∀ {p q : Permutation}, reachable p → validSwap p i j → q = p ∘ (Equiv.swap i j) → reachable q

theorem unreachable_target : ¬ reachable target := by sorry

end unreachable_target_l30_3070


namespace train_length_l30_3032

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 180 → time = 7 → speed * time * (5 / 18) = 350 := by
  sorry

end train_length_l30_3032


namespace rotten_oranges_percentage_l30_3083

/-- Proves that the percentage of rotten oranges is 15% given the problem conditions -/
theorem rotten_oranges_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_bananas_percentage : ℚ)
  (good_fruits_percentage : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_bananas_percentage = 4 / 100)
  (h4 : good_fruits_percentage = 894 / 1000)
  : (90 : ℚ) / total_oranges = 15 / 100 := by
  sorry

#check rotten_oranges_percentage

end rotten_oranges_percentage_l30_3083


namespace simplify_and_evaluate_fraction_l30_3066

theorem simplify_and_evaluate_fraction (a : ℝ) (h : a = 5) :
  (a^2 - 4) / a^2 / (1 - 2/a) = 7/5 := by
  sorry

end simplify_and_evaluate_fraction_l30_3066


namespace interior_angles_sum_l30_3029

theorem interior_angles_sum (n : ℕ) : 
  (180 * (n - 2) = 1800) → (180 * ((n + 4) - 2) = 2520) := by
  sorry

end interior_angles_sum_l30_3029


namespace speedster_convertibles_count_l30_3035

/-- Represents the inventory of an automobile company -/
structure Inventory where
  total : ℕ
  speedsters : ℕ
  speedsterConvertibles : ℕ

/-- Conditions of the inventory -/
def inventoryConditions (i : Inventory) : Prop :=
  i.speedsters = (2 * i.total) / 3 ∧
  i.speedsterConvertibles = (4 * i.speedsters) / 5 ∧
  i.total - i.speedsters = 50

/-- Theorem stating that under the given conditions, there are 80 Speedster convertibles -/
theorem speedster_convertibles_count (i : Inventory) :
  inventoryConditions i → i.speedsterConvertibles = 80 := by
  sorry

end speedster_convertibles_count_l30_3035


namespace wire_length_difference_l30_3040

theorem wire_length_difference (total_length first_part : ℕ) 
  (h1 : total_length = 180)
  (h2 : first_part = 106) :
  first_part - (total_length - first_part) = 32 := by
  sorry

end wire_length_difference_l30_3040


namespace magnitude_of_perpendicular_vector_l30_3075

/-- Given two planar vectors a and b, where a is perpendicular to b,
    prove that the magnitude of b is √5 --/
theorem magnitude_of_perpendicular_vector
  (a b : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b.1 = -2)
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  Real.sqrt (b.1^2 + b.2^2) = Real.sqrt 5 := by
  sorry

end magnitude_of_perpendicular_vector_l30_3075


namespace quadratic_roots_relation_l30_3012

theorem quadratic_roots_relation (b c : ℚ) : 
  (∃ r s : ℚ, (4 * r^2 - 6 * r - 8 = 0) ∧ 
               (4 * s^2 - 6 * s - 8 = 0) ∧ 
               ((r + 3)^2 + b * (r + 3) + c = 0) ∧ 
               ((s + 3)^2 + b * (s + 3) + c = 0)) →
  c = 23 / 2 := by
sorry

end quadratic_roots_relation_l30_3012


namespace coefficient_sum_l30_3054

theorem coefficient_sum (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ x : ℝ, (5*x - 2)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 729 := by
sorry

end coefficient_sum_l30_3054


namespace value_calculation_l30_3056

theorem value_calculation (x : ℝ) (y : ℝ) (h1 : x = 50.0) (h2 : y = 0.20 * x - 4) : y = 6.0 := by
  sorry

end value_calculation_l30_3056


namespace socks_cost_prove_socks_cost_l30_3011

def initial_amount : ℕ := 100
def shirt_cost : ℕ := 24
def final_amount : ℕ := 65

theorem socks_cost : ℕ :=
  initial_amount - shirt_cost - final_amount

theorem prove_socks_cost : socks_cost = 11 := by
  sorry

end socks_cost_prove_socks_cost_l30_3011


namespace probability_theorem_l30_3031

/-- The probability of selecting three distinct integers between 1 and 100 (inclusive) 
    such that their product is odd and a multiple of 5 -/
def probability_odd_multiple_of_five : ℚ := 3 / 125

/-- The set of integers from 1 to 100, inclusive -/
def integer_set : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 100}

/-- A function that determines if a natural number is odd -/
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- A function that determines if a natural number is a multiple of 5 -/
def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

/-- The main theorem stating that the probability of selecting three distinct integers 
    between 1 and 100 (inclusive) such that their product is odd and a multiple of 5 
    is equal to 3/125 -/
theorem probability_theorem : 
  ∀ (a b c : ℕ), a ∈ integer_set → b ∈ integer_set → c ∈ integer_set → 
  a ≠ b → b ≠ c → a ≠ c →
  (is_odd a ∧ is_odd b ∧ is_odd c ∧ (is_multiple_of_five a ∨ is_multiple_of_five b ∨ is_multiple_of_five c)) →
  probability_odd_multiple_of_five = 3 / 125 := by
  sorry

end probability_theorem_l30_3031


namespace stock_yield_proof_l30_3021

/-- Proves that the calculated yield matches the quoted yield for a given stock --/
theorem stock_yield_proof (quoted_yield : ℚ) (stock_price : ℚ) 
  (h1 : quoted_yield = 8 / 100)
  (h2 : stock_price = 225) : 
  let dividend := quoted_yield * stock_price
  ((dividend / stock_price) * 100 : ℚ) = quoted_yield * 100 := by
  sorry

end stock_yield_proof_l30_3021


namespace perimeter_of_triangle_MNO_l30_3098

/-- A right prism with equilateral triangular bases -/
structure RightPrism :=
  (height : ℝ)
  (base_side : ℝ)

/-- Points on the edges of the prism -/
structure PrismPoints (prism : RightPrism) :=
  (M : ℝ × ℝ × ℝ)
  (N : ℝ × ℝ × ℝ)
  (O : ℝ × ℝ × ℝ)

/-- The perimeter of triangle MNO in the prism -/
def triangle_perimeter (prism : RightPrism) (points : PrismPoints prism) : ℝ :=
  sorry

/-- Theorem stating the perimeter of triangle MNO -/
theorem perimeter_of_triangle_MNO (prism : RightPrism) (points : PrismPoints prism) 
  (h1 : prism.height = 20)
  (h2 : prism.base_side = 10)
  (h3 : points.M = (5, 0, 0))
  (h4 : points.N = (5, 5*Real.sqrt 3, 0))
  (h5 : points.O = (5, 0, 10)) :
  triangle_perimeter prism points = 5 + 10 * Real.sqrt 5 :=
sorry

end perimeter_of_triangle_MNO_l30_3098


namespace two_numbers_with_given_means_l30_3072

theorem two_numbers_with_given_means (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a * b = 5) →
  (2 * a * b / (a + b) = 5/3) →
  ((a = 1 ∧ b = 5) ∨ (a = 5 ∧ b = 1)) :=
by sorry

end two_numbers_with_given_means_l30_3072


namespace correct_factorization_l30_3018

theorem correct_factorization (a b : ℤ) :
  (∃ k : ℤ, (X + 6) * (X - 2) = X^2 + k*X + b) ∧
  (∃ m : ℤ, (X - 8) * (X + 4) = X^2 + a*X + m) →
  (X + 2) * (X - 6) = X^2 + a*X + b :=
by sorry

end correct_factorization_l30_3018


namespace decorative_window_area_ratio_l30_3033

-- Define the window structure
structure DecorativeWindow where
  ab : ℝ  -- Width of the rectangle (diameter of semicircles)
  ad : ℝ  -- Length of the rectangle
  h_ab_positive : ab > 0
  h_ratio : ad / ab = 4 / 3

-- Define the theorem
theorem decorative_window_area_ratio (w : DecorativeWindow) (h_ab : w.ab = 36) :
  (w.ad * w.ab) / (π * (w.ab / 2)^2) = 16 / (3 * π) := by
  sorry

end decorative_window_area_ratio_l30_3033


namespace negative_two_less_than_negative_one_l30_3045

theorem negative_two_less_than_negative_one : -2 < -1 := by
  sorry

end negative_two_less_than_negative_one_l30_3045


namespace dilution_proof_l30_3039

/-- Proves that adding 6 ounces of water to 12 ounces of a 60% alcohol solution results in a 40% alcohol solution. -/
theorem dilution_proof (initial_volume : ℝ) (initial_concentration : ℝ) (target_concentration : ℝ) 
  (water_added : ℝ) (h1 : initial_volume = 12) (h2 : initial_concentration = 0.6) 
  (h3 : target_concentration = 0.4) (h4 : water_added = 6) : 
  (initial_volume * initial_concentration) / (initial_volume + water_added) = target_concentration :=
by sorry

end dilution_proof_l30_3039


namespace jeff_ninja_stars_l30_3009

/-- The number of ninja throwing stars each person has -/
structure NinjaStars where
  eric : ℕ
  chad : ℕ
  jeff : ℕ

/-- The conditions of the problem -/
def ninja_star_problem (stars : NinjaStars) : Prop :=
  stars.eric = 4 ∧
  stars.chad = 2 * stars.eric ∧
  stars.eric + stars.chad + stars.jeff = 16 ∧
  stars.chad = (2 * stars.eric) - 2

theorem jeff_ninja_stars :
  ∃ (stars : NinjaStars), ninja_star_problem stars ∧ stars.jeff = 6 := by
  sorry

end jeff_ninja_stars_l30_3009


namespace total_birds_after_breeding_l30_3061

/-- Represents the types of birds on the farm -/
inductive BirdType
  | Hen
  | Duck
  | Goose
  | Pigeon

/-- Represents the count and breeding information for each bird type -/
structure BirdInfo where
  count : ℕ
  maleRatio : ℚ
  femaleRatio : ℚ
  offspringPerFemale : ℕ
  breedingSuccessRate : ℚ

/-- Calculates the total number of birds after the breeding season -/
def totalBirdsAfterBreeding (birdCounts : BirdType → BirdInfo) (pigeonHatchRate : ℚ) : ℕ :=
  sorry

/-- The main theorem stating the total number of birds after breeding -/
theorem total_birds_after_breeding :
  let birdCounts : BirdType → BirdInfo
    | BirdType.Hen => ⟨40, 2/9, 7/9, 7, 85/100⟩
    | BirdType.Duck => ⟨20, 1/4, 3/4, 9, 75/100⟩
    | BirdType.Goose => ⟨10, 3/11, 8/11, 5, 90/100⟩
    | BirdType.Pigeon => ⟨30, 1/2, 1/2, 2, 80/100⟩
  totalBirdsAfterBreeding birdCounts (80/100) = 442 := by
  sorry

end total_birds_after_breeding_l30_3061


namespace modern_growth_pattern_l30_3080

/-- Represents the different types of population growth patterns --/
inductive PopulationGrowthPattern
  | Traditional
  | Modern
  | Primitive
  | TransitionPrimitiveToTraditional

/-- Represents the level of a demographic rate --/
inductive RateLevel
  | Low
  | Medium
  | High

/-- Represents a country --/
structure Country where
  birthRate : RateLevel
  deathRate : RateLevel
  naturalGrowthRate : RateLevel

/-- Determines the population growth pattern of a country --/
def determineGrowthPattern (c : Country) : PopulationGrowthPattern :=
  sorry

theorem modern_growth_pattern (ourCountry : Country) 
  (h1 : ourCountry.birthRate = RateLevel.Low)
  (h2 : ourCountry.deathRate = RateLevel.Low)
  (h3 : ourCountry.naturalGrowthRate = RateLevel.Low) :
  determineGrowthPattern ourCountry = PopulationGrowthPattern.Modern :=
sorry

end modern_growth_pattern_l30_3080


namespace ratio_solution_set_l30_3093

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the solution set of f(x) ≥ 0
def solution_set_f (f : ℝ → ℝ) : Set ℝ := {x | f x ≥ 0}

-- Define the solution set of g(x) ≥ 0
def solution_set_g (g : ℝ → ℝ) : Set ℝ := {x | g x ≥ 0}

-- Define the solution set of f(x)/g(x) > 0
def solution_set_ratio (f g : ℝ → ℝ) : Set ℝ := {x | f x / g x > 0}

-- State the theorem
theorem ratio_solution_set 
  (h1 : solution_set_f f = Set.Icc 1 2) 
  (h2 : solution_set_g g = ∅) : 
  solution_set_ratio f g = Set.Ioi 2 ∪ Set.Iio 1 := by
  sorry

end ratio_solution_set_l30_3093


namespace product_evaluation_l30_3064

theorem product_evaluation :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) = 7^128 - 5^128 := by
  sorry

end product_evaluation_l30_3064


namespace lattice_triangle_circumcircle_diameter_bound_l30_3081

/-- A triangle with vertices on lattice points -/
structure LatticeTriangle where
  vertices : Fin 3 → ℤ × ℤ

/-- The side lengths of a LatticeTriangle -/
def side_lengths (t : LatticeTriangle) : Fin 3 → ℝ := sorry

/-- The diameter of the circumcircle of a LatticeTriangle -/
def circumcircle_diameter (t : LatticeTriangle) : ℝ := sorry

/-- Theorem: The diameter of the circumcircle of a triangle with lattice point vertices
    does not exceed the product of its side lengths -/
theorem lattice_triangle_circumcircle_diameter_bound (t : LatticeTriangle) :
  circumcircle_diameter t ≤ (side_lengths t 0) * (side_lengths t 1) * (side_lengths t 2) := by
  sorry

end lattice_triangle_circumcircle_diameter_bound_l30_3081


namespace sequence_formula_l30_3067

def S (n : ℕ+) (a : ℕ+ → ℚ) : ℚ := 1 - n * a n

theorem sequence_formula (a : ℕ+ → ℚ) 
  (h : ∀ n : ℕ+, S n a = 1 - n * a n) : 
  ∀ n : ℕ+, a n = 1 / (n * (n + 1)) := by
  sorry

end sequence_formula_l30_3067


namespace incorrect_calculation_l30_3022

theorem incorrect_calculation (a : ℝ) : (2 * a)^3 ≠ 6 * a^3 := by
  sorry

end incorrect_calculation_l30_3022


namespace decrement_value_theorem_l30_3000

theorem decrement_value_theorem (n : ℕ) (original_mean new_mean : ℝ) 
  (h1 : n = 50)
  (h2 : original_mean = 200)
  (h3 : new_mean = 185) :
  let decrement := (n * original_mean - n * new_mean) / n
  decrement = 15 := by
sorry

end decrement_value_theorem_l30_3000


namespace jerry_water_usage_l30_3043

/-- Calculates the total water usage for Jerry's household in July --/
def total_water_usage (drinking_cooking : ℕ) (shower_usage : ℕ) (num_showers : ℕ) 
  (pool_length : ℕ) (pool_width : ℕ) (pool_height : ℕ) : ℕ :=
  drinking_cooking + (shower_usage * num_showers) + (pool_length * pool_width * pool_height)

theorem jerry_water_usage :
  total_water_usage 100 20 15 10 10 6 = 1400 := by
  sorry

end jerry_water_usage_l30_3043


namespace least_subtraction_for_divisibility_problem_solution_l30_3058

theorem least_subtraction_for_divisibility (n m k : ℕ) (h : n - k ≡ 0 [MOD m]) : 
  ∀ j < k, ¬(n - j ≡ 0 [MOD m]) → k = n % m :=
sorry

-- The specific problem instance
def original_number : ℕ := 1852745
def divisor : ℕ := 251
def subtrahend : ℕ := 130

theorem problem_solution :
  (original_number - subtrahend) % divisor = 0 ∧
  ∀ j < subtrahend, (original_number - j) % divisor ≠ 0 :=
sorry

end least_subtraction_for_divisibility_problem_solution_l30_3058


namespace product_of_diff_squares_l30_3086

theorem product_of_diff_squares (a b c d : ℕ+) 
  (ha : ∃ (x y : ℕ+), a = x^2 - y^2)
  (hb : ∃ (z w : ℕ+), b = z^2 - w^2)
  (hc : ∃ (p q : ℕ+), c = p^2 - q^2)
  (hd : ∃ (r s : ℕ+), d = r^2 - s^2) :
  ∃ (u v : ℕ+), (a * b * c * d : ℕ) = u^2 - v^2 :=
sorry

end product_of_diff_squares_l30_3086


namespace least_bananas_l30_3085

theorem least_bananas (b₁ b₂ b₃ : ℕ) : 
  (∃ (A B C : ℕ), 
    A = b₁ / 2 + b₂ / 3 + 5 * b₃ / 12 ∧
    B = b₁ / 4 + 2 * b₂ / 3 + 5 * b₃ / 12 ∧
    C = b₁ / 4 + b₂ / 3 + b₃ / 6 ∧
    A = 4 * k ∧ B = 3 * k ∧ C = 2 * k ∧
    (∀ m, m < b₁ + b₂ + b₃ → 
      ¬(∃ (A' B' C' : ℕ), 
        A' = m / 2 + (b₁ + b₂ + b₃ - m) / 3 + 5 * (b₁ + b₂ + b₃ - m) / 12 ∧
        B' = m / 4 + 2 * (b₁ + b₂ + b₃ - m) / 3 + 5 * (b₁ + b₂ + b₃ - m) / 12 ∧
        C' = m / 4 + (b₁ + b₂ + b₃ - m) / 3 + (b₁ + b₂ + b₃ - m) / 6 ∧
        A' = 4 * k' ∧ B' = 3 * k' ∧ C' = 2 * k'))) →
  b₁ + b₂ + b₃ = 276 :=
by sorry

end least_bananas_l30_3085


namespace fraction_evaluation_l30_3065

theorem fraction_evaluation : (3 : ℚ) / (2 - 4 / (-5)) = 15 / 14 := by sorry

end fraction_evaluation_l30_3065


namespace spinner_probability_l30_3076

theorem spinner_probability (p_A p_B p_C p_D p_E : ℚ) : 
  p_A = 1/5 →
  p_B = 1/3 →
  p_C = p_D →
  p_E = 2 * p_C →
  p_A + p_B + p_C + p_D + p_E = 1 →
  p_C = 7/60 := by
sorry

end spinner_probability_l30_3076


namespace vector_decomposition_l30_3053

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![6, 5, -14]
def p : Fin 3 → ℝ := ![1, 1, 4]
def q : Fin 3 → ℝ := ![0, -3, 2]
def r : Fin 3 → ℝ := ![2, 1, -1]

/-- Theorem: x can be expressed as a linear combination of p, q, and r -/
theorem vector_decomposition :
  x = (-2 : ℝ) • p + (-1 : ℝ) • q + (4 : ℝ) • r := by
  sorry

end vector_decomposition_l30_3053


namespace x_power_243_minus_inverse_l30_3095

theorem x_power_243_minus_inverse (x : ℝ) (h : x - 1/x = Real.sqrt 3) : 
  x^243 - 1/x^243 = 6 * Real.sqrt 3 := by
  sorry

end x_power_243_minus_inverse_l30_3095


namespace total_earnings_proof_l30_3063

/-- Calculates the total earnings for a three-day fundraiser car wash activity. -/
def total_earnings (friday_earnings : ℕ) : ℕ :=
  let saturday_earnings := 2 * friday_earnings + 7
  let sunday_earnings := friday_earnings + 78
  friday_earnings + saturday_earnings + sunday_earnings

/-- Proves that the total earnings over three days is 673, given the specified conditions. -/
theorem total_earnings_proof :
  total_earnings 147 = 673 := by
  sorry

#eval total_earnings 147

end total_earnings_proof_l30_3063


namespace percentage_problem_l30_3059

theorem percentage_problem (x : ℝ) : (350 / 100) * x = 140 → x = 40 := by
  sorry

end percentage_problem_l30_3059


namespace danivan_drugstore_inventory_l30_3010

def calculate_final_inventory (starting_inventory : ℕ) (daily_sales : List ℕ) (deliveries : List ℕ) : ℕ :=
  let daily_changes := List.zipWith (λ s d => d - s) daily_sales deliveries
  starting_inventory + daily_changes.sum

theorem danivan_drugstore_inventory : 
  let starting_inventory : ℕ := 4500
  let daily_sales : List ℕ := [1277, 2124, 679, 854, 535, 1073, 728]
  let deliveries : List ℕ := [2250, 0, 980, 750, 0, 1345, 0]
  calculate_final_inventory starting_inventory daily_sales deliveries = 2555 := by
  sorry

#eval calculate_final_inventory 4500 [1277, 2124, 679, 854, 535, 1073, 728] [2250, 0, 980, 750, 0, 1345, 0]

end danivan_drugstore_inventory_l30_3010


namespace sum_of_powers_equals_negative_two_l30_3030

theorem sum_of_powers_equals_negative_two :
  -1^2010 + (-1)^2011 + 1^2012 - 1^2013 = -2 := by
  sorry

end sum_of_powers_equals_negative_two_l30_3030


namespace two_six_minus_one_prime_divisors_l30_3092

theorem two_six_minus_one_prime_divisors :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧
  (∀ (r : Nat), Nat.Prime r → r ∣ (2^6 - 1) → r = p ∨ r = q) ∧
  p + q = 10 := by
  sorry

end two_six_minus_one_prime_divisors_l30_3092


namespace contractor_payment_proof_l30_3062

/-- Calculates the total amount received by a contractor given the contract terms and absence information. -/
def contractor_payment (total_days : ℕ) (payment_per_day : ℚ) (fine_per_day : ℚ) (absent_days : ℕ) : ℚ :=
  (total_days - absent_days : ℕ) * payment_per_day - absent_days * fine_per_day

/-- Proves that the contractor receives Rs. 555 given the specified conditions. -/
theorem contractor_payment_proof :
  contractor_payment 30 25 (15/2) 6 = 555 := by
  sorry

end contractor_payment_proof_l30_3062


namespace orthocentre_constructible_l30_3037

/-- A point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line in the plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- A circle in the plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- A triangle in the plane -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Definition of a circumcircle of a triangle -/
def isCircumcircle (c : Circle) (t : Triangle) : Prop :=
  sorry

/-- Definition of a circumcentre of a triangle -/
def isCircumcentre (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Definition of an orthocentre of a triangle -/
def isOrthocentre (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Definition of constructible using only a straightedge -/
def isStraightedgeConstructible (p : Point) (given : Set Point) : Prop :=
  sorry

/-- The main theorem -/
theorem orthocentre_constructible (t : Triangle) (c : Circle) (o : Point) :
  isCircumcircle c t → isCircumcentre o t →
  ∃ h : Point, isOrthocentre h t ∧ 
    isStraightedgeConstructible h {t.A, t.B, t.C, o} :=
  sorry

end orthocentre_constructible_l30_3037


namespace smallest_x_for_g_equality_l30_3013

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem smallest_x_for_g_equality (g : ℝ → ℝ) : 
  (∀ (x : ℝ), x > 0 → g (4 * x) = 4 * g x) →
  (∀ (x : ℝ), 2 ≤ x ∧ x ≤ 4 → g x = 1 - |x - 3|) →
  (∀ (x : ℝ), x ≥ 0 ∧ g x = g 2048 → x ≥ 2) ∧
  g 2 = g 2048 :=
by sorry

end smallest_x_for_g_equality_l30_3013


namespace saturday_attendance_l30_3055

theorem saturday_attendance (price : ℝ) (total_earnings : ℝ) : 
  price = 10 →
  total_earnings = 300 →
  ∃ (saturday : ℕ),
    saturday * price + (saturday / 2) * price = total_earnings ∧
    saturday = 20 := by
  sorry

end saturday_attendance_l30_3055


namespace petya_ran_less_than_two_minutes_l30_3007

/-- Represents the race between Petya and Vasya -/
structure Race where
  distance : ℝ
  petyaSpeed : ℝ
  petyaTime : ℝ
  vasyaStartDelay : ℝ

/-- Conditions of the race -/
def raceConditions (r : Race) : Prop :=
  r.distance > 0 ∧
  r.petyaSpeed > 0 ∧
  r.petyaTime > 0 ∧
  r.vasyaStartDelay = 1 ∧
  r.distance = r.petyaSpeed * r.petyaTime ∧
  r.petyaTime < r.distance / (2 * r.petyaSpeed) + r.vasyaStartDelay

/-- Theorem: Under the given conditions, Petya ran the distance in less than two minutes -/
theorem petya_ran_less_than_two_minutes (r : Race) (h : raceConditions r) : r.petyaTime < 2 := by
  sorry

end petya_ran_less_than_two_minutes_l30_3007


namespace right_triangle_hypotenuse_l30_3036

theorem right_triangle_hypotenuse (L M N : ℝ) : 
  -- LMN is a right triangle with right angle at M
  -- sin N = 3/5
  -- LM = 18
  Real.sin N = 3/5 → LM = 18 → LN = 30 := by
  sorry

end right_triangle_hypotenuse_l30_3036


namespace vehicle_distance_after_3_minutes_l30_3026

/-- The distance between two vehicles after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

theorem vehicle_distance_after_3_minutes :
  let truck_speed : ℝ := 65
  let car_speed : ℝ := 85
  let time : ℝ := 3 / 60
  distance_between_vehicles truck_speed car_speed time = 1 := by sorry

end vehicle_distance_after_3_minutes_l30_3026


namespace fraction_equivalence_l30_3060

theorem fraction_equivalence : (15 : ℚ) / 25 = 3 / 5 := by
  sorry

end fraction_equivalence_l30_3060


namespace physics_score_l30_3069

/-- Represents the scores in physics, chemistry, and mathematics -/
structure Scores where
  physics : ℕ
  chemistry : ℕ
  mathematics : ℕ

/-- The average score of all three subjects is 60 -/
def average_all (s : Scores) : Prop :=
  (s.physics + s.chemistry + s.mathematics) / 3 = 60

/-- The average score of physics and mathematics is 90 -/
def average_physics_math (s : Scores) : Prop :=
  (s.physics + s.mathematics) / 2 = 90

/-- The average score of physics and chemistry is 70 -/
def average_physics_chem (s : Scores) : Prop :=
  (s.physics + s.chemistry) / 2 = 70

/-- Theorem stating that given the conditions, the physics score is 140 -/
theorem physics_score (s : Scores) 
  (h1 : average_all s)
  (h2 : average_physics_math s)
  (h3 : average_physics_chem s) :
  s.physics = 140 := by
  sorry

end physics_score_l30_3069


namespace two_numbers_sum_2014_l30_3004

theorem two_numbers_sum_2014 : ∃ (x y : ℕ), x > y ∧ x + y = 2014 ∧ 3 * (x / 100) = y + 6 ∧ y = 51 := by
  sorry

end two_numbers_sum_2014_l30_3004


namespace triangle_abc_properties_l30_3094

open Real

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  sin A / a = sin B / b →
  sin B / b = sin C / c →
  2 * sin A - sin B = 2 * sin C * cos B →
  c = 2 →
  C = π / 3 ∧ ∀ x, (2 * a - b = x) → -2 < x ∧ x < 4 :=
by sorry

end triangle_abc_properties_l30_3094


namespace wire_length_ratio_l30_3088

theorem wire_length_ratio :
  let edge_length : ℕ := 8
  let large_cube_wire_length : ℕ := 12 * edge_length
  let large_cube_volume : ℕ := edge_length ^ 3
  let unit_cube_wire_length : ℕ := 12
  let total_unit_cubes : ℕ := large_cube_volume
  let total_unit_cube_wire_length : ℕ := total_unit_cubes * unit_cube_wire_length
  (large_cube_wire_length : ℚ) / total_unit_cube_wire_length = 1 / 64 := by
  sorry

end wire_length_ratio_l30_3088


namespace range_of_inequality_l30_3074

-- Define an even function that is monotonically increasing on [0, +∞)
def f (x : ℝ) : ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f (-x) = f x
axiom f_monotone : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

-- Theorem statement
theorem range_of_inequality :
  ∀ x : ℝ, f (2 * x - 1) ≤ f 3 ↔ -1 ≤ x ∧ x ≤ 2 := by sorry

end range_of_inequality_l30_3074


namespace arithmetic_sequence_5_to_119_l30_3087

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Proof that the arithmetic sequence from 5 to 119 with common difference 3 has 39 terms -/
theorem arithmetic_sequence_5_to_119 :
  arithmeticSequenceLength 5 119 3 = 39 := by
  sorry

end arithmetic_sequence_5_to_119_l30_3087


namespace power_of_product_l30_3042

theorem power_of_product (a : ℝ) : (3 * a) ^ 3 = 27 * a ^ 3 := by
  sorry

end power_of_product_l30_3042


namespace james_distance_traveled_l30_3005

/-- Calculates the distance traveled given speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: James' distance traveled -/
theorem james_distance_traveled :
  distance_traveled 80.0 16.0 = 1280.0 := by
  sorry

end james_distance_traveled_l30_3005


namespace largest_multiple_of_12_less_than_neg_95_l30_3082

theorem largest_multiple_of_12_less_than_neg_95 : 
  ∀ n : ℤ, n * 12 < -95 → n * 12 ≤ -96 :=
by
  sorry

end largest_multiple_of_12_less_than_neg_95_l30_3082


namespace convex_n_gon_interior_angles_ratio_l30_3051

theorem convex_n_gon_interior_angles_ratio (n : ℕ) : 
  n ≥ 3 →
  ∃ x : ℝ, x > 0 ∧
    (∀ k : ℕ, k ≤ n → k * x < 180) ∧
    n * (n + 1) / 2 * x = (n - 2) * 180 →
  n = 3 ∨ n = 4 :=
sorry

end convex_n_gon_interior_angles_ratio_l30_3051


namespace matching_shoes_probability_l30_3050

theorem matching_shoes_probability (n : ℕ) (h : n = 12) :
  let total_shoes := 2 * n
  let total_combinations := (total_shoes.choose 2 : ℚ)
  let matching_pairs := n
  matching_pairs / total_combinations = 1 / 46 :=
by sorry

end matching_shoes_probability_l30_3050


namespace balloon_radius_increase_l30_3001

theorem balloon_radius_increase (C₁ C₂ r₁ r₂ Δr : ℝ) : 
  C₁ = 20 → 
  C₂ = 25 → 
  C₁ = 2 * Real.pi * r₁ → 
  C₂ = 2 * Real.pi * r₂ → 
  Δr = r₂ - r₁ → 
  Δr = 5 / (2 * Real.pi) := by
sorry

end balloon_radius_increase_l30_3001


namespace system_solution_l30_3028

-- Define the system of equations
def system (x y z w : ℝ) : Prop :=
  (x + y = z^2 + w^2 + 6*z*w) ∧
  (x + z = y^2 + w^2 + 6*y*w) ∧
  (x + w = y^2 + z^2 + 6*y*z) ∧
  (y + z = x^2 + w^2 + 6*x*w) ∧
  (y + w = x^2 + z^2 + 6*x*z) ∧
  (z + w = x^2 + y^2 + 6*x*y)

-- Define the set of solutions
def solution_set : Set (ℝ × ℝ × ℝ × ℝ) :=
  {(0, 0, 0, 0), (1/4, 1/4, 1/4, 1/4), (-1/4, -1/4, 3/4, -1/4), (-1/2, -1/2, 5/2, -1/2)}

-- Define cyclic permutations
def cyclic_perm (x y z w : ℝ) : Set (ℝ × ℝ × ℝ × ℝ) :=
  {(x, y, z, w), (y, z, w, x), (z, w, x, y), (w, x, y, z)}

-- Define the full solution set including cyclic permutations
def full_solution_set : Set (ℝ × ℝ × ℝ × ℝ) :=
  ⋃ (s : ℝ × ℝ × ℝ × ℝ) (hs : s ∈ solution_set), cyclic_perm s.1 s.2.1 s.2.2.1 s.2.2.2

-- Theorem statement
theorem system_solution :
  ∀ (x y z w : ℝ), system x y z w ↔ (x, y, z, w) ∈ full_solution_set :=
sorry

end system_solution_l30_3028


namespace circle_m_range_l30_3019

-- Define the equation of the circle
def circle_equation (x y m : ℝ) : Prop :=
  x + y^2 - x + y + m = 0

-- State the theorem
theorem circle_m_range :
  ∀ m : ℝ, (∃ x y : ℝ, circle_equation x y m) → m < (1/2) :=
by sorry

end circle_m_range_l30_3019


namespace second_cart_travel_distance_l30_3006

/-- Distance traveled by the first cart in n seconds -/
def first_cart_distance (n : ℕ) : ℕ := n * (6 + (n - 1) * 4)

/-- Distance traveled by the second cart in n seconds -/
def second_cart_distance (n : ℕ) : ℕ := n * (7 + (n - 1) * 9 / 2)

/-- Time taken by the first cart to reach the bottom -/
def total_time : ℕ := 35

/-- Time difference between the start of the two carts -/
def start_delay : ℕ := 2

theorem second_cart_travel_distance :
  second_cart_distance (total_time - start_delay) = 4983 := by
  sorry

end second_cart_travel_distance_l30_3006


namespace amandas_flowers_l30_3073

theorem amandas_flowers (amanda_flowers : ℕ) (peter_flowers : ℕ) : 
  peter_flowers = 3 * amanda_flowers →
  peter_flowers - 15 = 45 →
  amanda_flowers = 20 := by
sorry

end amandas_flowers_l30_3073


namespace website_earnings_l30_3099

/-- Calculates daily earnings for a website given monthly visits, days in a month, and earnings per visit -/
def daily_earnings (monthly_visits : ℕ) (days_in_month : ℕ) (earnings_per_visit : ℚ) : ℚ :=
  (monthly_visits : ℚ) / (days_in_month : ℚ) * earnings_per_visit

/-- Proves that given 30000 monthly visits in a 30-day month with $0.01 earnings per visit, daily earnings are $10 -/
theorem website_earnings : daily_earnings 30000 30 (1/100) = 10 := by
  sorry

end website_earnings_l30_3099


namespace committee_selection_ways_l30_3068

/-- The number of ways to choose two committees from a club -/
def choose_committees (total_members : ℕ) (exec_size : ℕ) (aux_size : ℕ) : ℕ :=
  Nat.choose total_members exec_size * Nat.choose (total_members - exec_size) aux_size

/-- Theorem stating the number of ways to choose committees from a 30-member club -/
theorem committee_selection_ways :
  choose_committees 30 5 3 = 327764800 := by
  sorry

end committee_selection_ways_l30_3068


namespace factor_a_squared_minus_16_l30_3020

theorem factor_a_squared_minus_16 (a : ℝ) : a^2 - 16 = (a - 4) * (a + 4) := by
  sorry

end factor_a_squared_minus_16_l30_3020


namespace arithmetic_sequence_10th_term_l30_3002

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_10th_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 2)
  (h_a4 : a 4 = 6) :
  a 10 = 14 := by
  sorry

end arithmetic_sequence_10th_term_l30_3002


namespace vector_difference_magnitude_l30_3014

def OA : ℝ × ℝ := (2, 2)
def OB : ℝ × ℝ := (5, 3)

def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

theorem vector_difference_magnitude : 
  Real.sqrt ((2 * OA.1 - OB.1)^2 + (2 * OA.2 - OB.2)^2) = Real.sqrt 2 := by
  sorry

end vector_difference_magnitude_l30_3014


namespace not_divisible_by_power_of_two_l30_3071

theorem not_divisible_by_power_of_two (p : ℕ) (hp : p > 1) :
  ¬(2^p ∣ 3^p + 1) := by
sorry

end not_divisible_by_power_of_two_l30_3071


namespace farm_chicken_count_l30_3048

/-- The number of chicken coops on the farm -/
def num_coops : ℕ := 9

/-- The number of chickens in each coop -/
def chickens_per_coop : ℕ := 60

/-- The total number of chickens on the farm -/
def total_chickens : ℕ := num_coops * chickens_per_coop

theorem farm_chicken_count : total_chickens = 540 := by
  sorry

end farm_chicken_count_l30_3048


namespace infinite_series_sum_l30_3079

theorem infinite_series_sum : 
  ∑' n : ℕ, (1 / ((2*n+1)^2 - (2*n-1)^2)) * (1 / (2*n-1)^2 - 1 / (2*n+1)^2) = 1 := by
  sorry

end infinite_series_sum_l30_3079
