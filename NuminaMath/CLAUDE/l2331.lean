import Mathlib

namespace nail_sizes_l2331_233145

theorem nail_sizes (fraction_2d : ℝ) (fraction_2d_or_4d : ℝ) (fraction_4d : ℝ) :
  fraction_2d = 0.25 →
  fraction_2d_or_4d = 0.75 →
  fraction_4d = fraction_2d_or_4d - fraction_2d →
  fraction_4d = 0.50 := by
sorry

end nail_sizes_l2331_233145


namespace third_number_in_multiplication_l2331_233161

theorem third_number_in_multiplication (p n : ℕ) : 
  (p = 125 * 243 * n / 405) → 
  (1000 ≤ p) → 
  (p < 10000) → 
  (∀ m : ℕ, m < n → 125 * 243 * m / 405 < 1000) →
  n = 14 := by
sorry

end third_number_in_multiplication_l2331_233161


namespace equal_probabilities_l2331_233123

/-- Represents a box containing balls of different colors -/
structure Box where
  red : ℕ
  green : ℕ

/-- Represents the state of both boxes -/
structure BoxState where
  red_box : Box
  green_box : Box

/-- Initial state of the boxes -/
def initial_state : BoxState :=
  { red_box := { red := 100, green := 0 },
    green_box := { red := 0, green := 100 } }

/-- State after transferring 8 red balls from red box to green box -/
def after_first_transfer (state : BoxState) : BoxState :=
  { red_box := { red := state.red_box.red - 8, green := state.red_box.green },
    green_box := { red := state.green_box.red + 8, green := state.green_box.green } }

/-- State after transferring 8 balls from green box to red box -/
def after_second_transfer (state : BoxState) : BoxState :=
  { red_box := { red := state.red_box.red + 8, green := state.red_box.green },
    green_box := { red := state.green_box.red - 8, green := state.green_box.green } }

/-- Final state after all transfers -/
def final_state : BoxState :=
  after_second_transfer (after_first_transfer initial_state)

/-- Probability of drawing a specific color ball from a box -/
def prob_draw (box : Box) (color : String) : ℚ :=
  match color with
  | "red" => box.red / (box.red + box.green)
  | "green" => box.green / (box.red + box.green)
  | _ => 0

theorem equal_probabilities :
  prob_draw final_state.red_box "green" = prob_draw final_state.green_box "red" := by
  sorry

end equal_probabilities_l2331_233123


namespace typing_time_proof_l2331_233114

def typing_speed : ℕ := 38
def paper_length : ℕ := 4560
def minutes_per_hour : ℕ := 60

theorem typing_time_proof :
  (paper_length / typing_speed : ℚ) / minutes_per_hour = 2 := by
  sorry

end typing_time_proof_l2331_233114


namespace hamburger_sales_proof_l2331_233113

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The average number of hamburgers sold per day -/
def average_daily_sales : ℕ := 9

/-- The total number of hamburgers sold in a week -/
def total_weekly_sales : ℕ := days_in_week * average_daily_sales

theorem hamburger_sales_proof : total_weekly_sales = 63 := by
  sorry

end hamburger_sales_proof_l2331_233113


namespace anya_erasers_difference_l2331_233104

theorem anya_erasers_difference (andrea_erasers : ℕ) (anya_ratio : ℚ) : 
  andrea_erasers = 6 → 
  anya_ratio = 4.5 → 
  (anya_ratio * andrea_erasers : ℚ) - andrea_erasers = 21 := by
sorry

end anya_erasers_difference_l2331_233104


namespace ab_value_l2331_233185

theorem ab_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a^2 + b^2 = 9) (h2 : a^4 + b^4 = 65) : a * b = 2 * Real.sqrt 2 := by
  sorry

end ab_value_l2331_233185


namespace max_value_expression_l2331_233141

/-- For positive real numbers a and b, and angle θ where 0 ≤ θ ≤ π/2,
    the maximum value of 2(a - x)(x + cos(θ)√(x^2 + b^2)) is a^2 + cos^2(θ)b^2 -/
theorem max_value_expression (a b : ℝ) (θ : ℝ) 
    (ha : a > 0) (hb : b > 0) (hθ : 0 ≤ θ ∧ θ ≤ π/2) :
  (⨆ x, 2 * (a - x) * (x + Real.cos θ * Real.sqrt (x^2 + b^2))) = a^2 + Real.cos θ^2 * b^2 := by
  sorry

end max_value_expression_l2331_233141


namespace fgh_supermarket_difference_l2331_233162

theorem fgh_supermarket_difference (total : ℕ) (us : ℕ) (h1 : total = 70) (h2 : us = 42) (h3 : us < total) :
  us - (total - us) = 14 := by
  sorry

end fgh_supermarket_difference_l2331_233162


namespace function_value_at_negative_l2331_233116

/-- Given a function f(x) = ax³ + bx - c/x + 2, if f(2023) = 6, then f(-2023) = -2 -/
theorem function_value_at_negative (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^3 + b * x - c / x + 2
  f 2023 = 6 → f (-2023) = -2 := by sorry

end function_value_at_negative_l2331_233116


namespace selection_for_38_classes_6_routes_l2331_233142

/-- The number of ways for a given number of classes to each choose one of a given number of routes. -/
def number_of_selections (num_classes : ℕ) (num_routes : ℕ) : ℕ := num_routes ^ num_classes

/-- Theorem stating that the number of ways for 38 classes to each choose one of 6 routes is 6^38. -/
theorem selection_for_38_classes_6_routes : number_of_selections 38 6 = 6^38 := by
  sorry

#eval number_of_selections 38 6

end selection_for_38_classes_6_routes_l2331_233142


namespace max_sum_squares_l2331_233147

theorem max_sum_squares (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + 2*b + 3*c = 1) :
  ∃ (max : ℝ), max = 1 ∧ a^2 + b^2 + c^2 ≤ max ∧ ∃ (a' b' c' : ℝ), 
    0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + 2*b' + 3*c' = 1 ∧ a'^2 + b'^2 + c'^2 = max :=
sorry

end max_sum_squares_l2331_233147


namespace vertex_of_parabola_l2331_233121

/-- The function f(x) = 3(x-1)^2 + 2 -/
def f (x : ℝ) : ℝ := 3 * (x - 1)^2 + 2

/-- The vertex of the parabola defined by f -/
def vertex : ℝ × ℝ := (1, 2)

theorem vertex_of_parabola :
  ∀ x : ℝ, f x ≥ f (vertex.1) ∧ f (vertex.1) = vertex.2 := by
  sorry

end vertex_of_parabola_l2331_233121


namespace other_endpoint_of_line_segment_l2331_233131

/-- Given a line segment with midpoint (3, 4) and one endpoint (-2, -5), 
    the other endpoint is (8, 13) -/
theorem other_endpoint_of_line_segment 
  (midpoint : ℝ × ℝ) 
  (endpoint1 : ℝ × ℝ) 
  (endpoint2 : ℝ × ℝ) : 
  midpoint = (3, 4) → 
  endpoint1 = (-2, -5) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧ 
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) → 
  endpoint2 = (8, 13) := by
sorry

end other_endpoint_of_line_segment_l2331_233131


namespace freds_salary_l2331_233154

theorem freds_salary (mikes_current_salary : ℝ) (mikes_salary_ratio : ℝ) (salary_increase_percent : ℝ) :
  mikes_current_salary = 15400 ∧
  mikes_salary_ratio = 10 ∧
  salary_increase_percent = 40 →
  (mikes_current_salary / (1 + salary_increase_percent / 100) / mikes_salary_ratio) = 1100 :=
by sorry

end freds_salary_l2331_233154


namespace volume_of_sphere_wedge_l2331_233108

/-- Given a sphere with circumference 18π inches cut into six congruent wedges,
    prove that the volume of one wedge is 162π cubic inches. -/
theorem volume_of_sphere_wedge :
  ∀ (r : ℝ), 
    r > 0 →
    2 * Real.pi * r = 18 * Real.pi →
    (4 / 3 * Real.pi * r^3) / 6 = 162 * Real.pi := by
  sorry

end volume_of_sphere_wedge_l2331_233108


namespace polynomial_simplification_l2331_233110

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + 5 * x^3 - 3 * x + 7) + (-x^4 + 4 * x^2 - 5 * x + 2) =
  x^4 + 5 * x^3 + 4 * x^2 - 8 * x + 9 := by
  sorry

end polynomial_simplification_l2331_233110


namespace snow_probability_l2331_233159

theorem snow_probability (p : ℝ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 5) :
  1 - (1 - p)^n = 1023/1024 := by
  sorry

end snow_probability_l2331_233159


namespace two_color_line_exists_l2331_233103

/-- Represents a color in the grid -/
inductive Color
| Red
| Blue
| Green
| Yellow

/-- Represents a point in the 2D grid -/
structure Point where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point in the grid -/
def Coloring := Point → Color

/-- Predicate to check if four points form a unit square -/
def isUnitSquare (p1 p2 p3 p4 : Point) : Prop :=
  (p1.x = p2.x ∧ p1.y + 1 = p2.y) ∧
  (p2.x + 1 = p3.x ∧ p2.y = p3.y) ∧
  (p3.x = p4.x ∧ p3.y - 1 = p4.y) ∧
  (p4.x - 1 = p1.x ∧ p4.y = p1.y)

/-- Predicate to check if a coloring is valid (adjacent nodes in unit squares have different colors) -/
def isValidColoring (c : Coloring) : Prop :=
  ∀ p1 p2 p3 p4 : Point, isUnitSquare p1 p2 p3 p4 →
    c p1 ≠ c p2 ∧ c p1 ≠ c p3 ∧ c p1 ≠ c p4 ∧
    c p2 ≠ c p3 ∧ c p2 ≠ c p4 ∧
    c p3 ≠ c p4

/-- Predicate to check if a line (horizontal or vertical) uses only two colors -/
def lineUsesTwoColors (c : Coloring) : Prop :=
  (∃ y : ℤ, ∃ c1 c2 : Color, ∀ x : ℤ, c ⟨x, y⟩ = c1 ∨ c ⟨x, y⟩ = c2) ∨
  (∃ x : ℤ, ∃ c1 c2 : Color, ∀ y : ℤ, c ⟨x, y⟩ = c1 ∨ c ⟨x, y⟩ = c2)

theorem two_color_line_exists (c : Coloring) (h : isValidColoring c) : lineUsesTwoColors c := by
  sorry

end two_color_line_exists_l2331_233103


namespace average_score_theorem_l2331_233105

def max_score : ℕ := 900
def amar_percent : ℕ := 64
def bhavan_percent : ℕ := 36
def chetan_percent : ℕ := 44
def num_boys : ℕ := 3

theorem average_score_theorem :
  let amar_score := max_score * amar_percent / 100
  let bhavan_score := max_score * bhavan_percent / 100
  let chetan_score := max_score * chetan_percent / 100
  let total_score := amar_score + bhavan_score + chetan_score
  (total_score / num_boys : ℚ) = 432 := by sorry

end average_score_theorem_l2331_233105


namespace magnitude_of_complex_number_l2331_233148

theorem magnitude_of_complex_number : Complex.abs (5/6 + 2*Complex.I) = 13/6 := by
  sorry

end magnitude_of_complex_number_l2331_233148


namespace sum_K_floor_quotient_100_l2331_233106

/-- K(x) is the number of irreducible fractions a/b where 1 ≤ a < x and 1 ≤ b < x -/
def K (x : ℕ) : ℕ :=
  (Finset.range (x - 1)).sum (λ k => Nat.totient k)

/-- The sum of K(⌊100/k⌋) for k from 1 to 100 equals 9801 -/
theorem sum_K_floor_quotient_100 :
  (Finset.range 100).sum (λ k => K (100 / (k + 1))) = 9801 := by
  sorry

end sum_K_floor_quotient_100_l2331_233106


namespace fraction_zero_implies_x_equals_one_l2331_233152

theorem fraction_zero_implies_x_equals_one (x : ℝ) : 
  (x^2 - 1) / (x + 1) = 0 → x = 1 := by
  sorry

end fraction_zero_implies_x_equals_one_l2331_233152


namespace exists_sum_all_odd_digits_l2331_233171

/-- A function that returns true if all digits of a natural number are odd -/
def allDigitsOdd (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

/-- A function that reverses the digits of a three-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 100 + (n / 10 % 10) * 10 + (n / 100)

/-- Theorem stating that there exists a three-digit number A such that
    A + reverseDigits(A) has all odd digits -/
theorem exists_sum_all_odd_digits :
  ∃ A : ℕ, 100 ≤ A ∧ A < 1000 ∧ allDigitsOdd (A + reverseDigits A) :=
sorry

end exists_sum_all_odd_digits_l2331_233171


namespace four_card_selection_three_suits_l2331_233170

theorem four_card_selection_three_suits (deck_size : Nat) (suits : Nat) (cards_per_suit : Nat) 
  (selection_size : Nat) (suits_represented : Nat) (cards_from_main_suit : Nat) :
  deck_size = suits * cards_per_suit →
  selection_size = 4 →
  suits = 4 →
  cards_per_suit = 13 →
  suits_represented = 3 →
  cards_from_main_suit = 2 →
  (suits.choose 1) * (suits - 1).choose (suits_represented - 1) * 
  (cards_per_suit.choose cards_from_main_suit) * 
  (cards_per_suit.choose 1) * (cards_per_suit.choose 1) = 158184 := by
sorry

end four_card_selection_three_suits_l2331_233170


namespace red_parrots_count_l2331_233111

theorem red_parrots_count (total : ℕ) (green_fraction : ℚ) (blue_fraction : ℚ) 
  (h_total : total = 160)
  (h_green : green_fraction = 5/8)
  (h_blue : blue_fraction = 1/4)
  (h_sum : green_fraction + blue_fraction < 1) :
  total - (green_fraction * total).num - (blue_fraction * total).num = 20 := by
  sorry

end red_parrots_count_l2331_233111


namespace tennis_players_count_l2331_233150

theorem tennis_players_count (total : ℕ) (baseball : ℕ) (both : ℕ) (no_sport : ℕ) :
  total = 310 →
  baseball = 255 →
  both = 94 →
  no_sport = 11 →
  ∃ tennis : ℕ, tennis = 138 ∧ total = tennis + baseball - both + no_sport :=
by sorry

end tennis_players_count_l2331_233150


namespace adams_earnings_l2331_233199

/-- Adam's lawn mowing earnings problem -/
theorem adams_earnings (rate : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ) 
  (h1 : rate = 9)
  (h2 : total_lawns = 12)
  (h3 : forgotten_lawns = 8) :
  (total_lawns - forgotten_lawns) * rate = 36 := by
  sorry

#check adams_earnings

end adams_earnings_l2331_233199


namespace complex_magnitude_equality_l2331_233115

theorem complex_magnitude_equality (n : ℝ) :
  n > 0 ∧ Complex.abs (2 + n * Complex.I) = 4 * Real.sqrt 5 ↔ n = 2 * Real.sqrt 19 := by
  sorry

end complex_magnitude_equality_l2331_233115


namespace cats_given_away_l2331_233158

/-- Proves that the number of cats given away is 14, given the initial and remaining cat counts -/
theorem cats_given_away (initial_cats : ℝ) (remaining_cats : ℕ) 
  (h1 : initial_cats = 17.0) (h2 : remaining_cats = 3) : 
  initial_cats - remaining_cats = 14 := by
  sorry

end cats_given_away_l2331_233158


namespace shell_calculation_l2331_233143

theorem shell_calculation (initial : Real) (add1 : Real) (add2 : Real) (subtract : Real) (final : Real) :
  initial = 5.2 ∧ add1 = 15.7 ∧ add2 = 17.5 ∧ subtract = 4.3 ∧ final = 102.3 →
  final = 3 * ((initial + add1 + add2 - subtract)) :=
by sorry

end shell_calculation_l2331_233143


namespace negation_of_all_squares_nonnegative_l2331_233177

theorem negation_of_all_squares_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end negation_of_all_squares_nonnegative_l2331_233177


namespace intersection_implies_sum_l2331_233173

-- Define the sets N and M
def N (q : ℝ) : Set ℝ := {x | x^2 + 6*x - q = 0}
def M (p : ℝ) : Set ℝ := {x | x^2 - p*x + 6 = 0}

-- State the theorem
theorem intersection_implies_sum (p q : ℝ) :
  M p ∩ N q = {2} → p + q = 21 := by
  sorry

end intersection_implies_sum_l2331_233173


namespace donna_weekly_episodes_l2331_233191

/-- The number of episodes Donna can watch on a weekday -/
def weekday_episodes : ℕ := 8

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days_per_week : ℕ := 2

/-- The multiplier for weekend episodes compared to weekday episodes -/
def weekend_multiplier : ℕ := 3

/-- The total number of episodes Donna can watch in a week -/
def total_episodes_per_week : ℕ :=
  weekday_episodes * weekdays_per_week +
  (weekday_episodes * weekend_multiplier) * weekend_days_per_week

theorem donna_weekly_episodes :
  total_episodes_per_week = 88 := by
  sorry


end donna_weekly_episodes_l2331_233191


namespace dagger_example_l2331_233151

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (q / n)

-- Theorem statement
theorem dagger_example : dagger (3/7) (11/4) = 132/7 := by
  sorry

end dagger_example_l2331_233151


namespace range_of_a_l2331_233109

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + x - 2 > 0
def q (x a : ℝ) : Prop := x > a

-- Define what it means for q to be sufficient but not necessary for p
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, q x a → p x) ∧ ¬(∀ x, p x → q x a)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  sufficient_not_necessary a → a ∈ Set.Ici 1 :=
by sorry

end range_of_a_l2331_233109


namespace compound_interest_proof_l2331_233101

/-- Given a principal amount for which the simple interest over 2 years at 10% rate is $600,
    prove that the compound interest over 2 years at 10% rate is $630 --/
theorem compound_interest_proof (P : ℝ) : 
  P * 0.1 * 2 = 600 → P * (1 + 0.1)^2 - P = 630 := by
  sorry

end compound_interest_proof_l2331_233101


namespace grocery_stock_problem_l2331_233174

theorem grocery_stock_problem (asparagus_bundles : ℕ) (asparagus_price : ℚ)
  (grape_boxes : ℕ) (grape_price : ℚ)
  (apple_price : ℚ) (total_worth : ℚ) :
  asparagus_bundles = 60 →
  asparagus_price = 3 →
  grape_boxes = 40 →
  grape_price = (5/2) →
  apple_price = (1/2) →
  total_worth = 630 →
  (total_worth - (asparagus_bundles * asparagus_price + grape_boxes * grape_price)) / apple_price = 700 := by
  sorry

end grocery_stock_problem_l2331_233174


namespace mrs_kaplan_pizza_slices_l2331_233163

theorem mrs_kaplan_pizza_slices :
  ∀ (bobby_pizzas : ℕ) (slices_per_pizza : ℕ) (kaplan_fraction : ℚ),
    bobby_pizzas = 2 →
    slices_per_pizza = 6 →
    kaplan_fraction = 1 / 4 →
    (↑bobby_pizzas * ↑slices_per_pizza : ℚ) * kaplan_fraction = 3 :=
by
  sorry

end mrs_kaplan_pizza_slices_l2331_233163


namespace ali_baba_treasure_max_value_l2331_233117

/-- The maximum value problem for Ali Baba's treasure --/
theorem ali_baba_treasure_max_value :
  let f : ℝ → ℝ → ℝ := λ x y => 20 * x + 60 * y
  let S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 ≤ 100 ∧ p.1 + 5 * p.2 ≤ 200}
  ∃ (x y : ℝ), (x, y) ∈ S ∧ f x y = 3000 ∧ ∀ (x' y' : ℝ), (x', y') ∈ S → f x' y' ≤ 3000 :=
by sorry


end ali_baba_treasure_max_value_l2331_233117


namespace closed_set_A_l2331_233144

def f (x : ℚ) : ℚ := (1 + x) / (1 - x)

def A : Set ℚ := {2, -3, -1/2, 1/3}

theorem closed_set_A :
  (2 ∈ A) ∧
  (∀ x ∈ A, f x ∈ A) ∧
  (∀ S : Set ℚ, 2 ∈ S → (∀ x ∈ S, f x ∈ S) → A ⊆ S) :=
sorry

end closed_set_A_l2331_233144


namespace calculate_expression_l2331_233124

theorem calculate_expression : 
  2 / (-1/4) - |(-Real.sqrt 18)| + (1/5)⁻¹ = -3 * Real.sqrt 2 - 3 := by
  sorry

end calculate_expression_l2331_233124


namespace arithmetic_sequence_common_difference_l2331_233181

/-- An arithmetic sequence with the given property has a common difference of 3. -/
theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h : a 2015 = a 2013 + 6)  -- The given condition
  : ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 :=
by sorry

end arithmetic_sequence_common_difference_l2331_233181


namespace perpendicular_vectors_imply_b_l2331_233169

/-- Two 2D vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- The direction vector of the first line -/
def v : ℝ × ℝ := (4, -9)

/-- The direction vector of the second line -/
def w (b : ℝ) : ℝ × ℝ := (b, 3)

/-- Theorem: If the direction vectors v and w(b) are perpendicular, then b = 27/4 -/
theorem perpendicular_vectors_imply_b (b : ℝ) :
  perpendicular v (w b) → b = 27/4 := by
  sorry

end perpendicular_vectors_imply_b_l2331_233169


namespace function_equality_l2331_233137

theorem function_equality (x : ℝ) : x = Real.log (Real.exp x) := by
  sorry

end function_equality_l2331_233137


namespace simplify_expression_l2331_233122

theorem simplify_expression : Real.sqrt ((-2)^6) - (-1)^0 = 7 := by
  sorry

end simplify_expression_l2331_233122


namespace power_of_power_l2331_233138

theorem power_of_power (a : ℝ) : (a ^ 3) ^ 4 = a ^ 12 := by
  sorry

end power_of_power_l2331_233138


namespace alani_earnings_l2331_233192

/-- Calculates the earnings for baby-sitting given the base earnings, base hours, and actual hours worked. -/
def calculate_earnings (base_earnings : ℚ) (base_hours : ℚ) (actual_hours : ℚ) : ℚ :=
  (base_earnings / base_hours) * actual_hours

/-- Proves that Alani will earn $75 for 5 hours of baby-sitting given her rate of $45 for 3 hours. -/
theorem alani_earnings : calculate_earnings 45 3 5 = 75 := by
  sorry

end alani_earnings_l2331_233192


namespace first_train_speed_calculation_l2331_233180

/-- The speed of the first train in kmph -/
def first_train_speed : ℝ := 72

/-- The speed of the second train in kmph -/
def second_train_speed : ℝ := 36

/-- The length of the first train in meters -/
def first_train_length : ℝ := 200

/-- The length of the second train in meters -/
def second_train_length : ℝ := 300

/-- The time taken for the first train to cross the second train in seconds -/
def crossing_time : ℝ := 49.9960003199744

theorem first_train_speed_calculation :
  first_train_speed = 
    (first_train_length + second_train_length) / crossing_time * 3600 / 1000 + second_train_speed :=
by sorry

end first_train_speed_calculation_l2331_233180


namespace fold_length_is_ten_rectangle_fold_length_l2331_233187

/-- Represents a folded rectangle with specific properties -/
structure FoldedRectangle where
  short_side : ℝ
  long_side : ℝ
  fold_length : ℝ
  congruent_triangles : Prop

/-- The folded rectangle satisfies the problem conditions -/
def satisfies_conditions (r : FoldedRectangle) : Prop :=
  r.short_side = 12 ∧
  r.long_side = r.short_side * 3/2 ∧
  r.congruent_triangles

/-- The theorem to be proved -/
theorem fold_length_is_ten 
  (r : FoldedRectangle) 
  (h : satisfies_conditions r) : 
  r.fold_length = 10 := by
  sorry

/-- The main theorem restated in terms of the problem -/
theorem rectangle_fold_length :
  ∃ (r : FoldedRectangle), 
    satisfies_conditions r ∧ 
    r.fold_length = 10 := by
  sorry

end fold_length_is_ten_rectangle_fold_length_l2331_233187


namespace attendance_decrease_l2331_233183

/-- Proves that given a projected 25 percent increase in attendance and actual attendance being 64 percent of the projected attendance, the actual percent decrease in attendance is 20 percent. -/
theorem attendance_decrease (P : ℝ) (P_positive : P > 0) : 
  let projected_attendance := 1.25 * P
  let actual_attendance := 0.64 * projected_attendance
  let percent_decrease := (P - actual_attendance) / P * 100
  percent_decrease = 20 := by
  sorry

end attendance_decrease_l2331_233183


namespace planes_perp_to_line_are_parallel_line_in_plane_perp_to_other_plane_implies_planes_perp_l2331_233136

-- Define basic geometric objects
variable (P Q R : Plane) (L M : Line)

-- Define geometric relationships
def perpendicular (L : Line) (P : Plane) : Prop := sorry
def parallel (P Q : Plane) : Prop := sorry
def contains (P : Plane) (L : Line) : Prop := sorry

-- Theorem 1: Two planes perpendicular to the same line are parallel to each other
theorem planes_perp_to_line_are_parallel 
  (h1 : perpendicular L P) (h2 : perpendicular L Q) : parallel P Q := by sorry

-- Theorem 2: If a line within a plane is perpendicular to another plane, 
-- then these two planes are perpendicular to each other
theorem line_in_plane_perp_to_other_plane_implies_planes_perp 
  (h1 : contains P L) (h2 : perpendicular L Q) : perpendicular P Q := by sorry

end planes_perp_to_line_are_parallel_line_in_plane_perp_to_other_plane_implies_planes_perp_l2331_233136


namespace arithmetic_sequence_property_l2331_233132

/-- Given an arithmetic sequence {a_n} where a_2 + a_8 = 16, prove that a_5 = 8 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : a 2 + a 8 = 16) : 
  a 5 = 8 := by
sorry

end arithmetic_sequence_property_l2331_233132


namespace min_distance_to_line_l2331_233156

theorem min_distance_to_line (x y : ℝ) :
  8 * x + 15 * y = 120 →
  x ≥ 0 →
  ∃ (min : ℝ), min = 120 / 17 ∧ ∀ (x' y' : ℝ), 8 * x' + 15 * y' = 120 → x' ≥ 0 → Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min :=
by
  sorry

end min_distance_to_line_l2331_233156


namespace expression_bounds_l2331_233112

-- Define the constraint function
def constraint (x y : ℝ) : Prop := (|x| - 3)^2 + (|y| - 2)^2 = 1

-- Define the expression to be minimized/maximized
def expression (x y : ℝ) : ℝ := |x + 2| + |y + 3|

-- Theorem statement
theorem expression_bounds :
  (∃ x y : ℝ, constraint x y) →
  (∃ min max : ℝ,
    (∀ x y : ℝ, constraint x y → expression x y ≥ min) ∧
    (∃ x y : ℝ, constraint x y ∧ expression x y = min) ∧
    (∀ x y : ℝ, constraint x y → expression x y ≤ max) ∧
    (∃ x y : ℝ, constraint x y ∧ expression x y = max) ∧
    min = 2 - Real.sqrt 2 ∧
    max = 10 + Real.sqrt 2) :=
sorry

end expression_bounds_l2331_233112


namespace omega_range_l2331_233140

theorem omega_range (ω : ℝ) (f : ℝ → ℝ) : 
  ω > 0 → 
  (∀ x, f x = Real.sin (ω * x + π / 4)) →
  (∀ x y, π / 2 < x → x < y → y < π → f y < f x) →
  1 / 2 ≤ ω ∧ ω ≤ 5 / 4 := by
sorry

end omega_range_l2331_233140


namespace set_membership_implies_value_l2331_233125

theorem set_membership_implies_value (m : ℚ) : 
  let A : Set ℚ := {m + 2, 2 * m^2 + m}
  3 ∈ A → m = -3/2 := by
  sorry

end set_membership_implies_value_l2331_233125


namespace three_digit_addition_l2331_233164

theorem three_digit_addition (A B : Nat) : A < 10 → B < 10 → 
  600 + 10 * A + 5 + 100 + 10 * B = 748 → B = 3 := by
  sorry

end three_digit_addition_l2331_233164


namespace freds_allowance_l2331_233100

theorem freds_allowance (allowance : ℝ) : 
  (allowance / 2 + 6 = 14) → allowance = 16 := by
  sorry

end freds_allowance_l2331_233100


namespace corporation_total_employees_l2331_233179

/-- The total number of employees in a corporation -/
def total_employees (part_time full_time contractors interns consultants : ℕ) : ℕ :=
  part_time + full_time + contractors + interns + consultants

/-- Theorem: The corporation employs 66907 workers in total -/
theorem corporation_total_employees :
  total_employees 2047 63109 1500 333 918 = 66907 := by
  sorry

end corporation_total_employees_l2331_233179


namespace square_fold_angle_l2331_233107

/-- A square with side length 1 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1))

/-- The angle formed by two lines after folding a square along its diagonal -/
def dihedral_angle (s : Square) : ℝ := sorry

/-- Theorem: The dihedral angle formed by folding a square along its diagonal is 60° -/
theorem square_fold_angle (s : Square) : dihedral_angle s = 60 * π / 180 := by sorry

end square_fold_angle_l2331_233107


namespace periodic_properties_l2331_233102

-- Define a periodic function
def Periodic (f : ℝ → ℝ) : Prop :=
  ∃ T > 0, ∀ x, f (x + T) = f x

-- Define a non-periodic function
def NonPeriodic (g : ℝ → ℝ) : Prop :=
  ¬ Periodic g

theorem periodic_properties
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : Periodic f) (hg : NonPeriodic g) :
  Periodic (fun x ↦ (f x)^2) ∧
  NonPeriodic (fun x ↦ Real.sqrt (g x)) ∧
  Periodic (g ∘ f) :=
sorry

end periodic_properties_l2331_233102


namespace quadratic_two_distinct_roots_l2331_233197

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
   (k - 1) * x^2 - 2 * x + 1 = 0 ∧ 
   (k - 1) * y^2 - 2 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ 1) :=
sorry

end quadratic_two_distinct_roots_l2331_233197


namespace sin_double_angle_on_unit_circle_l2331_233196

/-- Given a point B on the unit circle with coordinates (-3/5, 4/5), 
    prove that sin(2α) = -24/25, where α is the angle formed by OA and OB, 
    and O is the origin and A is the point (1,0) on the unit circle. -/
theorem sin_double_angle_on_unit_circle 
  (B : ℝ × ℝ) 
  (h_B_on_circle : B.1^2 + B.2^2 = 1) 
  (h_B_coords : B = (-3/5, 4/5)) 
  (α : ℝ) 
  (h_α_def : α = Real.arccos B.1) : 
  Real.sin (2 * α) = -24/25 := by
sorry

end sin_double_angle_on_unit_circle_l2331_233196


namespace arithmetic_geometric_mean_ratio_l2331_233157

theorem arithmetic_geometric_mean_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  let A := (x + y) / 2
  let G := Real.sqrt (x * y)
  A / G = 5 / 4 → x / y = 4 := by
  sorry

end arithmetic_geometric_mean_ratio_l2331_233157


namespace circle_division_relationship_l2331_233166

theorem circle_division_relationship (a k : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x y : ℝ, x^2 + y^2 = 1 → (x = a ∨ x = -a ∨ y = k * x)) →
  a^2 * (k^2 + 1) ≥ 1 := by
  sorry

end circle_division_relationship_l2331_233166


namespace equation_equivalent_to_two_lines_l2331_233190

-- Define the equation
def equation (x y : ℝ) : Prop := (x - 2*y)^2 = x^2 + y^2

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = 0
def line2 (x y : ℝ) : Prop := y = (4/3) * x

-- Theorem statement
theorem equation_equivalent_to_two_lines :
  ∀ x y : ℝ, equation x y ↔ (line1 x y ∨ line2 x y) :=
sorry

end equation_equivalent_to_two_lines_l2331_233190


namespace min_occupied_seats_150_l2331_233184

/-- Given a row of seats, returns the minimum number of occupied seats required
    to ensure the next person must sit next to someone -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for 150 seats, the minimum number of occupied seats
    required to ensure the next person must sit next to someone is 90 -/
theorem min_occupied_seats_150 : min_occupied_seats 150 = 90 := by
  sorry

end min_occupied_seats_150_l2331_233184


namespace g_is_even_l2331_233165

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define g as the sum of f(x) and f(-x)
def g (x : ℝ) : ℝ := f x + f (-x)

-- Theorem stating that g is an even function
theorem g_is_even : ∀ x : ℝ, g f x = g f (-x) := by
  sorry

end g_is_even_l2331_233165


namespace greatest_integer_difference_l2331_233118

-- Define the sets A, B, and C
def A : Set ℝ := {-6, -5, -4, -3}
def B : Set ℝ := {2/3, 3/4, 7/9, 2.5}
def C : Set ℝ := {5, 5.5, 6, 6.5}

-- Define the theorem
theorem greatest_integer_difference (a b c : ℝ) 
  (ha : a ∈ A) (hb : b ∈ B) (hc : c ∈ C) :
  ∃ (d : ℤ), d = 5 ∧ 
  ∀ (a' b' c' : ℝ), a' ∈ A → b' ∈ B → c' ∈ C → 
  (Int.floor (|c' - Real.sqrt b' - (a' + Real.sqrt b')|) : ℤ) ≤ d :=
sorry

end greatest_integer_difference_l2331_233118


namespace single_elimination_tournament_games_l2331_233126

/-- Number of games required to determine a champion in a single-elimination tournament -/
def games_required (num_players : ℕ) : ℕ := num_players - 1

/-- Theorem: In a single-elimination tournament with 512 players, 511 games are required to determine the champion -/
theorem single_elimination_tournament_games (num_players : ℕ) (h : num_players = 512) :
  games_required num_players = 511 := by
  sorry

end single_elimination_tournament_games_l2331_233126


namespace expression_factorization_l2331_233134

theorem expression_factorization (a b c d p q r s : ℝ) :
  (a * p + b * q + c * r + d * s)^2 +
  (a * q - b * p + c * s - d * r)^2 +
  (a * r - b * s - c * p + d * q)^2 +
  (a * s + b * r - c * q - d * p)^2 =
  (a^2 + b^2 + c^2 + d^2) * (p^2 + q^2 + r^2 + s^2) := by
  sorry

end expression_factorization_l2331_233134


namespace divisibility_by_two_l2331_233188

theorem divisibility_by_two (a b : ℕ) (h : 2 ∣ (a * b)) : ¬(¬(2 ∣ a) ∧ ¬(2 ∣ b)) := by
  sorry

end divisibility_by_two_l2331_233188


namespace robin_cupcakes_proof_l2331_233198

/-- Represents the number of cupcakes Robin initially made -/
def initial_cupcakes : ℕ := 42

/-- Represents the number of cupcakes Robin sold -/
def sold_cupcakes : ℕ := 22

/-- Represents the number of cupcakes Robin made later -/
def new_cupcakes : ℕ := 39

/-- Represents the final number of cupcakes Robin had -/
def final_cupcakes : ℕ := 59

/-- Proves that the initial number of cupcakes is correct given the conditions -/
theorem robin_cupcakes_proof : 
  initial_cupcakes - sold_cupcakes + new_cupcakes = final_cupcakes :=
by sorry

end robin_cupcakes_proof_l2331_233198


namespace least_k_for_inequality_l2331_233149

theorem least_k_for_inequality : 
  ∃ k : ℕ+, (∀ a : ℝ, a ∈ Set.Icc 0 1 → ∀ n : ℕ+, (a^(k:ℝ) * (1 - a)^(n:ℝ) < 1 / ((n:ℝ) + 1)^3)) ∧ 
  (∀ k' : ℕ+, k' < k → ∃ a : ℝ, a ∈ Set.Icc 0 1 ∧ ∃ n : ℕ+, a^(k':ℝ) * (1 - a)^(n:ℝ) ≥ 1 / ((n:ℝ) + 1)^3) ∧
  k = 4 :=
sorry

end least_k_for_inequality_l2331_233149


namespace shortest_rope_length_l2331_233168

theorem shortest_rope_length (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a : ℝ) / 4 = (b : ℝ) / 5 ∧ (b : ℝ) / 5 = (c : ℝ) / 6 →
  a + c = b + 100 →
  a = 80 := by
sorry

end shortest_rope_length_l2331_233168


namespace x_eq_2_sufficient_not_necessary_l2331_233139

-- Define the vectors a and b
def a (x : ℝ) : Fin 2 → ℝ := fun i => if i = 0 then x else 1
def b (x : ℝ) : Fin 2 → ℝ := fun i => if i = 0 then 4 else x

-- Define the parallel condition
def are_parallel (x : ℝ) : Prop := ∃ k : ℝ, ∀ i : Fin 2, a x i = k * b x i

-- Statement: x = 2 is sufficient but not necessary for a and b to be parallel
theorem x_eq_2_sufficient_not_necessary :
  (are_parallel 2) ∧ (∃ y : ℝ, y ≠ 2 ∧ are_parallel y) := by sorry

end x_eq_2_sufficient_not_necessary_l2331_233139


namespace line_hyperbola_intersection_l2331_233146

/-- The line y = kx + 2 intersects the hyperbola x^2 - y^2 = 2 at exactly one point if and only if k = ±1 or k = ±√3 -/
theorem line_hyperbola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 2 ∧ p.1^2 - p.2^2 = 2) ↔ 
  (k = 1 ∨ k = -1 ∨ k = Real.sqrt 3 ∨ k = -Real.sqrt 3) :=
sorry

end line_hyperbola_intersection_l2331_233146


namespace ten_player_tournament_matches_l2331_233178

/-- The number of matches in a round-robin tournament -/
def roundRobinMatches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A 10-player round-robin tournament has 45 matches -/
theorem ten_player_tournament_matches :
  roundRobinMatches 10 = 45 := by
  sorry

end ten_player_tournament_matches_l2331_233178


namespace parabola_line_intersection_l2331_233129

/-- A point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- A parabola defined by a focus point and a directrix line -/
structure Parabola :=
  (focus : Point) (directrix : Line)

/-- Represents the intersection points between a line and a parabola -/
inductive Intersection
  | NoIntersection
  | OnePoint (p : Point)
  | TwoPoints (p1 p2 : Point)

/-- 
Given a point F (focus), a line L, and a line D (directrix) in a plane,
there exists a construction method to find the intersection points (if any)
between L and the parabola defined by focus F and directrix D.
-/
theorem parabola_line_intersection
  (F : Point) (L D : Line) :
  ∃ (construct : Point → Line → Line → Intersection),
    construct F L D = Intersection.NoIntersection ∨
    (∃ p, construct F L D = Intersection.OnePoint p) ∨
    (∃ p1 p2, construct F L D = Intersection.TwoPoints p1 p2) :=
sorry

end parabola_line_intersection_l2331_233129


namespace intersection_theorem_union_theorem_complement_union_theorem_l2331_233182

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 3 < x ∧ x < 10}

-- State the theorems to be proved
theorem intersection_theorem : A ∩ B = {x | 3 < x ∧ x < 7} := by sorry

theorem union_theorem : A ∪ B = {x | 2 ≤ x ∧ x < 10} := by sorry

theorem complement_union_theorem : (Set.univ \ A) ∪ (Set.univ \ B) = {x | x ≤ 3 ∨ x ≥ 7} := by sorry

end intersection_theorem_union_theorem_complement_union_theorem_l2331_233182


namespace side_angle_relation_l2331_233155

theorem side_angle_relation (A B C : ℝ) (a b c : ℝ) :
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a / Real.sin A = b / Real.sin B) →
  (a / Real.sin A = c / Real.sin C) →
  (a > b ↔ Real.sin A > Real.sin B) :=
by sorry

end side_angle_relation_l2331_233155


namespace lamp_lit_area_l2331_233189

/-- Given a square plot with a lamp on one corner, if the light reaches 21 m
    and the lit area is 346.36 m², then the side length of the square plot is 21 m. -/
theorem lamp_lit_area (light_reach : ℝ) (lit_area : ℝ) (side_length : ℝ) : 
  light_reach = 21 →
  lit_area = 346.36 →
  lit_area = (1/4) * Real.pi * light_reach^2 →
  side_length = light_reach →
  side_length = 21 := by sorry

end lamp_lit_area_l2331_233189


namespace function_equation_solution_l2331_233195

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x ^ 2 + f y) = x * f x + y) →
  (∀ x : ℝ, f x = x ∨ f x = -x) :=
by sorry

end function_equation_solution_l2331_233195


namespace fifteen_shaded_cubes_l2331_233176

/-- Represents a 3x3x3 cube constructed from smaller cubes -/
structure LargeCube where
  size : Nat
  total_cubes : Nat
  shaded_per_face : Nat

/-- Calculates the number of uniquely shaded cubes in the large cube -/
def count_shaded_cubes (cube : LargeCube) : Nat :=
  sorry

/-- Theorem stating that the number of uniquely shaded cubes is 15 -/
theorem fifteen_shaded_cubes (cube : LargeCube) 
  (h1 : cube.size = 3) 
  (h2 : cube.total_cubes = 27) 
  (h3 : cube.shaded_per_face = 3) : 
  count_shaded_cubes cube = 15 := by
  sorry

end fifteen_shaded_cubes_l2331_233176


namespace total_legs_in_farm_l2331_233194

theorem total_legs_in_farm (total_animals : Nat) (num_ducks : Nat) (duck_legs : Nat) (dog_legs : Nat) :
  total_animals = 8 →
  num_ducks = 4 →
  duck_legs = 2 →
  dog_legs = 4 →
  (num_ducks * duck_legs + (total_animals - num_ducks) * dog_legs) = 24 := by
  sorry

end total_legs_in_farm_l2331_233194


namespace m_less_than_one_sufficient_not_necessary_l2331_233167

-- Define the function f(x) = x^2 + 2x + m
def f (x m : ℝ) : ℝ := x^2 + 2*x + m

-- Define what it means for f to have a root
def has_root (m : ℝ) : Prop := ∃ x : ℝ, f x m = 0

-- Statement: "m < 1" is a sufficient but not necessary condition for f to have a root
theorem m_less_than_one_sufficient_not_necessary :
  (∀ m : ℝ, m < 1 → has_root m) ∧ 
  (∃ m : ℝ, ¬(m < 1) ∧ has_root m) :=
sorry

end m_less_than_one_sufficient_not_necessary_l2331_233167


namespace sqrt_720_simplification_l2331_233128

theorem sqrt_720_simplification : Real.sqrt 720 = 12 * Real.sqrt 5 := by
  sorry

end sqrt_720_simplification_l2331_233128


namespace polygon_sides_from_interior_angle_l2331_233135

theorem polygon_sides_from_interior_angle (interior_angle : ℝ) : 
  interior_angle = 140 → (360 / (180 - interior_angle) : ℝ) = 9 := by
  sorry

end polygon_sides_from_interior_angle_l2331_233135


namespace contrapositive_exponential_l2331_233133

theorem contrapositive_exponential (a b : ℝ) :
  (∀ a b, a > b → 2^a > 2^b) ↔ (∀ a b, 2^a ≤ 2^b → a ≤ b) := by
  sorry

end contrapositive_exponential_l2331_233133


namespace solution_set_equality_l2331_233172

theorem solution_set_equality (x : ℝ) : 
  (1 / ((x + 1) * (x - 1)) ≤ 0) ↔ (-1 < x ∧ x < 1) := by
  sorry

end solution_set_equality_l2331_233172


namespace even_sine_function_phi_l2331_233175

/-- Given a function f(x) = sin((x + φ) / 3) where φ ∈ [0, 2π],
    prove that if f is even, then φ = 3π/2 -/
theorem even_sine_function_phi (φ : ℝ) (h1 : φ ∈ Set.Icc 0 (2 * Real.pi)) :
  (∀ x, Real.sin ((x + φ) / 3) = Real.sin ((-x + φ) / 3)) →
  φ = 3 * Real.pi / 2 := by sorry

end even_sine_function_phi_l2331_233175


namespace bowtie_equation_solution_l2331_233153

-- Define the ⋈ operation
noncomputable def bowtie (a b : ℝ) : ℝ := a * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem bowtie_equation_solution :
  ∀ y : ℝ, bowtie 3 y = 27 → y = 72 := by
sorry

end bowtie_equation_solution_l2331_233153


namespace eight_four_two_power_l2331_233186

theorem eight_four_two_power : 8^8 * 4^4 / 2^28 = 16 := by sorry

end eight_four_two_power_l2331_233186


namespace complex_modulus_problem_l2331_233130

theorem complex_modulus_problem (z : ℂ) (i : ℂ) (h1 : i^2 = -1) (h2 : (1 - i) * z = 1) : 
  Complex.abs (4 * z - 3) = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l2331_233130


namespace product_expansion_sum_l2331_233127

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x : ℝ, (4 * x^2 - 6 * x + 5) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + 2 * c + d = -44 := by
sorry

end product_expansion_sum_l2331_233127


namespace counter_value_l2331_233119

/-- Given a counter with 'a' beads in the tens place and 'b' beads in the ones place,
    the number represented by this counter is equal to 10a + b. -/
theorem counter_value (a b : ℕ) : 10 * a + b = 10 * a + b := by
  sorry

end counter_value_l2331_233119


namespace solution_system_l2331_233160

theorem solution_system (x y m n : ℤ) : 
  x = 2 ∧ y = -3 ∧ x + y = m ∧ 2 * x - y = n → m - n = -8 := by
  sorry

end solution_system_l2331_233160


namespace sampling_is_systematic_l2331_233120

/-- Represents a sampling method -/
inductive SamplingMethod
  | Stratified
  | Lottery
  | Random
  | Systematic

/-- Represents a class in the freshman year -/
structure FreshmanClass where
  students : Fin 56

/-- Represents the freshman year -/
structure FreshmanYear where
  classes : Fin 35 → FreshmanClass

/-- Defines the sampling method used in the problem -/
def samplingMethodUsed (year : FreshmanYear) : SamplingMethod :=
  SamplingMethod.Systematic

/-- Theorem stating that the sampling method used is systematic sampling -/
theorem sampling_is_systematic (year : FreshmanYear) :
  samplingMethodUsed year = SamplingMethod.Systematic :=
sorry

end sampling_is_systematic_l2331_233120


namespace obese_employee_is_male_prob_l2331_233193

-- Define the company's employee structure
structure Company where
  male_ratio : ℚ
  female_ratio : ℚ
  male_obese_ratio : ℚ
  female_obese_ratio : ℚ

-- Define the probability function
def prob_obese_is_male (c : Company) : ℚ :=
  (c.male_ratio * c.male_obese_ratio) / 
  (c.male_ratio * c.male_obese_ratio + c.female_ratio * c.female_obese_ratio)

-- Theorem statement
theorem obese_employee_is_male_prob 
  (c : Company) 
  (h1 : c.male_ratio = 3/5) 
  (h2 : c.female_ratio = 2/5)
  (h3 : c.male_obese_ratio = 1/5)
  (h4 : c.female_obese_ratio = 1/10) :
  prob_obese_is_male c = 3/4 := by
  sorry

end obese_employee_is_male_prob_l2331_233193
