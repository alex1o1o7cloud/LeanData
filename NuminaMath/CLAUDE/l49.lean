import Mathlib

namespace square_sum_product_l49_4991

theorem square_sum_product (x : ℝ) :
  (Real.sqrt (9 + x) + Real.sqrt (16 - x) = 8) →
  ((9 + x) * (16 - x) = 380.25) := by
  sorry

end square_sum_product_l49_4991


namespace fibonacci_mod_4_2022_l49_4973

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def b (n : ℕ) : ℕ := fibonacci n % 4

theorem fibonacci_mod_4_2022 : b 2022 = 0 := by
  sorry

end fibonacci_mod_4_2022_l49_4973


namespace inequalities_proof_l49_4999

theorem inequalities_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  2/(a-1) + 1/(b-2) ≥ 2 ∧ 2*a + b ≥ 8 := by
  sorry

end inequalities_proof_l49_4999


namespace tangent_line_y_intercept_l49_4914

/-- The y-intercept of the tangent line to y = x^3 + 11 at (1,12) is 9 -/
theorem tangent_line_y_intercept : 
  let f (x : ℝ) := x^3 + 11
  let P : ℝ × ℝ := (1, 12)
  let m := (deriv f) P.1
  let tangent_line (x : ℝ) := m * (x - P.1) + P.2
  tangent_line 0 = 9 := by sorry

end tangent_line_y_intercept_l49_4914


namespace existence_of_point_l49_4906

theorem existence_of_point :
  ∃ (x₀ y₀ z₀ : ℝ),
    (x₀ + y₀ + z₀ ≠ 0) ∧
    (0 < x₀^2 + y₀^2 + z₀^2) ∧
    (x₀^2 + y₀^2 + z₀^2 < 1 / 1999) ∧
    (1.999 < (x₀^2 + y₀^2 + z₀^2) / (x₀ + y₀ + z₀)) ∧
    ((x₀^2 + y₀^2 + z₀^2) / (x₀ + y₀ + z₀) < 2) :=
by sorry

end existence_of_point_l49_4906


namespace exam_results_l49_4990

theorem exam_results (total_students : ℕ) 
  (percent_8_or_more : ℚ) (percent_5_or_less : ℚ) :
  total_students = 40 →
  percent_8_or_more = 20 / 100 →
  percent_5_or_less = 45 / 100 →
  (1 : ℚ) - percent_8_or_more - percent_5_or_less = 35 / 100 := by
  sorry

end exam_results_l49_4990


namespace point_Q_y_coordinate_product_l49_4994

theorem point_Q_y_coordinate_product : ∀ (y₁ y₂ : ℝ),
  (∃ (Q : ℝ × ℝ), 
    Q.1 = 4 ∧ 
    ((Q.1 - 1)^2 + (Q.2 - (-3))^2) = 10^2 ∧
    (Q.2 = y₁ ∨ Q.2 = y₂) ∧
    y₁ ≠ y₂) →
  y₁ * y₂ = -82 := by
sorry

end point_Q_y_coordinate_product_l49_4994


namespace cube_edge_length_l49_4993

/-- Given a cube with volume 3375 cm³, prove that the total length of its edges is 180 cm. -/
theorem cube_edge_length (V : ℝ) (h : V = 3375) : 
  12 * (V ^ (1/3 : ℝ)) = 180 :=
sorry

end cube_edge_length_l49_4993


namespace sum_congruence_mod_9_l49_4981

theorem sum_congruence_mod_9 : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end sum_congruence_mod_9_l49_4981


namespace committee_formation_proof_l49_4904

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_formation_proof :
  let total_students : ℕ := 8
  let committee_size : ℕ := 5
  let always_included : ℕ := 2
  let remaining_students : ℕ := total_students - always_included
  let students_to_choose : ℕ := committee_size - always_included
  choose remaining_students students_to_choose = 20 := by
  sorry

end committee_formation_proof_l49_4904


namespace quadratic_reciprocity_legendre_symbol_two_l49_4995

-- Define the Legendre symbol
noncomputable def legendre_symbol (a p : ℕ) : ℤ := sorry

-- Quadratic reciprocity law
theorem quadratic_reciprocity (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hodd_p : Odd p) (hodd_q : Odd q) :
  legendre_symbol p q * legendre_symbol q p = (-1) ^ ((p - 1) * (q - 1) / 4) := by sorry

-- Legendre symbol of 2
theorem legendre_symbol_two (m : ℕ) (hm : Nat.Prime m) (hodd_m : Odd m) :
  legendre_symbol 2 m = (-1) ^ ((m^2 - 1) / 8) := by sorry

end quadratic_reciprocity_legendre_symbol_two_l49_4995


namespace big_n_conference_teams_l49_4998

/-- The number of games in a round-robin tournament with n teams -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of teams in the BIG N conference -/
theorem big_n_conference_teams : ∃ (n : ℕ), n > 0 ∧ num_games n = 21 :=
  sorry

end big_n_conference_teams_l49_4998


namespace deck_width_proof_l49_4924

/-- Proves that for a rectangular pool of 20 feet by 22 feet, surrounded by a deck of uniform width,
    if the total area of the pool and deck is 728 square feet, then the width of the deck is 3 feet. -/
theorem deck_width_proof (w : ℝ) : 
  (20 + 2*w) * (22 + 2*w) = 728 → w = 3 :=
by sorry

end deck_width_proof_l49_4924


namespace division_problem_l49_4955

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 217 →
  divisor = 4 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 54 := by
sorry

end division_problem_l49_4955


namespace rectangle_area_comparison_l49_4943

theorem rectangle_area_comparison (a b : ℝ) (ha : a = 8) (hb : b = 15) : 
  let d := Real.sqrt (a^2 + b^2)
  let new_rectangle_area := (d + b) * (d - b)
  let square_area := (a + b)^2
  new_rectangle_area ≠ square_area := by
sorry

end rectangle_area_comparison_l49_4943


namespace four_digit_numbers_extrema_l49_4960

theorem four_digit_numbers_extrema :
  let sum_of_numbers : ℕ := 106656
  let is_valid_number : (ℕ → Bool) :=
    λ n => n ≥ 1000 ∧ n ≤ 9999 ∧ 
           (let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
            digits.all (· ≠ 0) ∧ digits.Nodup)
  let valid_numbers := (List.range 10000).filter is_valid_number
  sum_of_numbers = valid_numbers.sum →
  (∀ n ∈ valid_numbers, n ≤ 9421 ∧ n ≥ 1249) ∧
  9421 ∈ valid_numbers ∧ 1249 ∈ valid_numbers :=
by sorry

end four_digit_numbers_extrema_l49_4960


namespace x_squared_over_y_squared_equals_two_l49_4920

theorem x_squared_over_y_squared_equals_two
  (x y z : ℝ) 
  (x_pos : 0 < x) 
  (y_pos : 0 < y) 
  (z_pos : 0 < z) 
  (all_different : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h : y^2 / (x^2 - z^2) = (x^2 + y^2) / z^2)
  (h2 : (x^2 + y^2) / z^2 = x^2 / y^2) :
  x^2 / y^2 = 2 := by
sorry

end x_squared_over_y_squared_equals_two_l49_4920


namespace max_customers_interviewed_l49_4939

theorem max_customers_interviewed (total : ℕ) (impulsive : ℕ) (ad_influence_ratio : ℚ) (consultant_ratio : ℚ) : 
  total ≤ 50 ∧
  impulsive = 7 ∧
  ad_influence_ratio = 3/4 ∧
  consultant_ratio = 1/3 ∧
  (∃ k : ℕ, total - impulsive = 4 * k) ∧
  (ad_influence_ratio * (total - impulsive)).isInt ∧
  (consultant_ratio * ad_influence_ratio * (total - impulsive)).isInt →
  total ≤ 47 ∧ 
  (∃ max_total : ℕ, max_total = 47 ∧ 
    max_total ≤ 50 ∧
    (∃ k : ℕ, max_total - impulsive = 4 * k) ∧
    (ad_influence_ratio * (max_total - impulsive)).isInt ∧
    (consultant_ratio * ad_influence_ratio * (max_total - impulsive)).isInt) :=
by sorry

end max_customers_interviewed_l49_4939


namespace p_q_contradictory_l49_4907

-- Define proposition p
def p : Prop := ∀ a : ℝ, a > 0 → a^2 ≠ 0

-- Define proposition q
def q : Prop := ∀ a : ℝ, a ≤ 0 → a^2 = 0

-- Theorem stating that p and q are contradictory
theorem p_q_contradictory : p ↔ ¬q := by
  sorry


end p_q_contradictory_l49_4907


namespace symmetric_point_coordinates_l49_4944

structure Point where
  x : ℝ
  y : ℝ

def translate_left (p : Point) (d : ℝ) : Point :=
  ⟨p.x - d, p.y⟩

def symmetric_origin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem symmetric_point_coordinates :
  let A : Point := ⟨1, 2⟩
  let B : Point := translate_left A 2
  let C : Point := symmetric_origin B
  C = ⟨1, -2⟩ := by sorry

end symmetric_point_coordinates_l49_4944


namespace seven_eighths_of_64_l49_4911

theorem seven_eighths_of_64 : (7 / 8 : ℚ) * 64 = 56 := by
  sorry

end seven_eighths_of_64_l49_4911


namespace max_reciprocal_sum_of_zeros_l49_4927

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then k * x^2 + 2 * x - 1
  else if x > 1 then k * x + 1
  else 0  -- Define f(x) as 0 for x ≤ 0 to make it total

/-- Theorem stating the maximum value of 1/x₁ + 1/x₂ -/
theorem max_reciprocal_sum_of_zeros (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 ∧ 1/x₁ + 1/x₂ ≤ 9/4) ∧
  (∃ k₀ : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k₀ x₁ = 0 ∧ f k₀ x₂ = 0 ∧ 1/x₁ + 1/x₂ = 9/4) :=
by sorry


end max_reciprocal_sum_of_zeros_l49_4927


namespace imaginary_part_of_fraction_l49_4912

noncomputable def i : ℂ := Complex.I

theorem imaginary_part_of_fraction (z : ℂ) : z = 2016 / (1 + i) → Complex.im z = -1008 := by
  sorry

end imaginary_part_of_fraction_l49_4912


namespace max_number_after_two_moves_l49_4908

def initial_number : ℕ := 4597

def swap_adjacent_digits (n : ℕ) (i : ℕ) : ℕ := 
  sorry

def subtract_100 (n : ℕ) : ℕ := 
  sorry

def make_move (n : ℕ) (i : ℕ) : ℕ := 
  subtract_100 (swap_adjacent_digits n i)

def max_after_moves (n : ℕ) (moves : ℕ) : ℕ := 
  sorry

theorem max_number_after_two_moves : 
  max_after_moves initial_number 2 = 4659 := by
  sorry

end max_number_after_two_moves_l49_4908


namespace phi_value_l49_4910

theorem phi_value (φ : Real) (h1 : 0 < φ) (h2 : φ < Real.pi / 2) 
  (h3 : Real.sqrt 2 * Real.cos (20 * Real.pi / 180) = Real.sin φ - Real.cos φ) : 
  φ = 25 * Real.pi / 180 := by
sorry

end phi_value_l49_4910


namespace num_category_B_prob_both_categories_l49_4974

/- Define the number of category A housekeepers -/
def category_A : ℕ := 12

/- Define the total number of housekeepers selected for training -/
def selected_total : ℕ := 20

/- Define the number of category B housekeepers selected for training -/
def selected_B : ℕ := 16

/- Define the number of category A housekeepers available for hiring -/
def available_A : ℕ := 3

/- Define the number of category B housekeepers available for hiring -/
def available_B : ℕ := 2

/- Theorem for the number of category B housekeepers -/
theorem num_category_B : ∃ x : ℕ, 
  (category_A * selected_B) / (selected_total - selected_B) = x :=
sorry

/- Theorem for the probability of hiring from both categories -/
theorem prob_both_categories : 
  (available_A * available_B) / ((available_A + available_B) * (available_A + available_B - 1) / 2) = 3/5 :=
sorry

end num_category_B_prob_both_categories_l49_4974


namespace three_fourths_of_45_l49_4956

theorem three_fourths_of_45 : (3 : ℚ) / 4 * 45 = 33 + 3 / 4 := by
  sorry

end three_fourths_of_45_l49_4956


namespace equation_transformation_l49_4946

theorem equation_transformation (a b : ℝ) : 
  (∀ x, x^2 - 6*x - 5 = 0 ↔ (x + a)^2 = b) → a + b = 11 := by
  sorry

end equation_transformation_l49_4946


namespace problem_solution_l49_4972

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Define the theorem
theorem problem_solution :
  -- Given conditions
  (∀ x : ℝ, (f 2 x ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 5)) →
  -- Part 1: Prove that a = 2
  (∃! a : ℝ, ∀ x : ℝ, (f a x ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 5)) ∧
  -- Part 2: Prove that the minimum value of f(x) + f(x+5) is 5
  (∀ x : ℝ, f 2 x + f 2 (x + 5) ≥ 5) ∧
  (∃ x : ℝ, f 2 x + f 2 (x + 5) = 5) :=
by sorry

end problem_solution_l49_4972


namespace parallel_range_perpendicular_min_abs_product_l49_4903

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + a^2 * y + 1 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a^2 + 1) * x - b * y + 3 = 0

-- Define parallel lines
def parallel (a b : ℝ) : Prop := ∀ x y, l₁ a x y ↔ l₂ a b x y

-- Define perpendicular lines
def perpendicular (a b : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂, 
  l₁ a x₁ y₁ → l₂ a b x₂ y₂ → (x₁ - x₂) * (y₁ - y₂) = 0

-- Statement 1: If l₁ ∥ l₂, then b ∈ (-∞, -6) ∪ (-6, 0]
theorem parallel_range (a b : ℝ) : 
  parallel a b → b < -6 ∨ (-6 < b ∧ b ≤ 0) :=
sorry

-- Statement 2: If l₁ ⟂ l₂, then the minimum value of |ab| is 2
theorem perpendicular_min_abs_product (a b : ℝ) :
  perpendicular a b → |a * b| ≥ 2 :=
sorry

end parallel_range_perpendicular_min_abs_product_l49_4903


namespace sum_of_roots_l49_4954

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 - 3*x^2 + 5*x - 17 = 0)
  (hy : y^3 - 3*y^2 + 5*y + 11 = 0) : 
  x + y = 2 := by sorry

end sum_of_roots_l49_4954


namespace quadratic_always_negative_l49_4905

theorem quadratic_always_negative (k : ℝ) :
  (∀ x : ℝ, (5 - k) * x^2 - 2 * (1 - k) * x + (2 - 2 * k) < 0) ↔ k > 9 := by
  sorry

end quadratic_always_negative_l49_4905


namespace quadratic_radical_rule_l49_4926

theorem quadratic_radical_rule (n : ℕ+) : 
  Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2)) := by
  sorry

end quadratic_radical_rule_l49_4926


namespace total_area_after_expansion_l49_4916

/-- Theorem: Total area of two houses after expansion -/
theorem total_area_after_expansion (small_house large_house expansion : ℕ) 
  (h1 : small_house = 5200)
  (h2 : large_house = 7300)
  (h3 : expansion = 3500) :
  small_house + large_house + expansion = 16000 := by
  sorry

#check total_area_after_expansion

end total_area_after_expansion_l49_4916


namespace ball_selection_problem_l49_4953

/-- The number of red balls in the box -/
def red_balls : ℕ := 12

/-- The number of blue balls in the box -/
def blue_balls : ℕ := 7

/-- The total number of balls in the box -/
def total_balls : ℕ := red_balls + blue_balls

/-- The number of ways to select 3 red balls and 2 blue balls -/
def ways_to_select : ℕ := Nat.choose red_balls 3 * Nat.choose blue_balls 2

/-- The probability of drawing 2 blue balls first, then 1 red ball -/
def prob_draw : ℚ :=
  (Nat.choose blue_balls 2 * Nat.choose red_balls 1) / Nat.choose total_balls 3

/-- The final result -/
theorem ball_selection_problem :
  ways_to_select * prob_draw = 388680 / 323 := by sorry

end ball_selection_problem_l49_4953


namespace gcd_of_B_is_five_l49_4982

/-- The set of all numbers that can be represented as the sum of five consecutive positive integers -/
def B : Set ℕ := {n : ℕ | ∃ y : ℕ, y > 0 ∧ n = (y - 2) + (y - 1) + y + (y + 1) + (y + 2)}

/-- The greatest common divisor of all numbers in B is 5 -/
theorem gcd_of_B_is_five : ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d) ∧ d = 5 := by
  sorry

end gcd_of_B_is_five_l49_4982


namespace exists_equal_digit_sum_l49_4966

-- Define an arithmetic progression
def arithmeticProgression (a₀ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₀ + n * d

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem exists_equal_digit_sum (a₀ : ℕ) (d : ℕ) (h : d ≠ 0) :
  ∃ (m n : ℕ), m ≠ n ∧ 
    sumOfDigits (arithmeticProgression a₀ d m) = sumOfDigits (arithmeticProgression a₀ d n) := by
  sorry


end exists_equal_digit_sum_l49_4966


namespace simplify_expression_l49_4968

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) :
  ((x + y)^2 - y * (2*x + y) - 6*x) / (2*x) = x/2 - 3 := by
  sorry

end simplify_expression_l49_4968


namespace sold_below_cost_price_l49_4921

def cost_price : ℚ := 5625
def profit_percentage : ℚ := 16 / 100
def additional_amount : ℚ := 1800

def selling_price_with_profit : ℚ := cost_price * (1 + profit_percentage)
def actual_selling_price : ℚ := selling_price_with_profit - additional_amount

def percentage_below_cost : ℚ := (cost_price - actual_selling_price) / cost_price * 100

theorem sold_below_cost_price : percentage_below_cost = 16 := by sorry

end sold_below_cost_price_l49_4921


namespace concentric_circles_chords_l49_4948

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle between two adjacent chords is 60°, then the minimum number of such chords
    needed to complete a full circle is 3. -/
theorem concentric_circles_chords (angle_between_chords : ℝ) (n : ℕ) :
  angle_between_chords = 60 →
  (n : ℝ) * (180 - angle_between_chords) = 360 →
  n = 3 :=
sorry

end concentric_circles_chords_l49_4948


namespace smallest_a_is_9_l49_4989

-- Define the arithmetic sequence
def is_arithmetic_sequence (a b c : ℕ) : Prop := b - a = c - b

-- Define the function f
def f (a b c : ℕ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem smallest_a_is_9 
  (a b c : ℕ) 
  (r s : ℝ) 
  (h_arith : is_arithmetic_sequence a b c)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a < b ∧ b < c)
  (h_f_r : f a b c r = s)
  (h_f_s : f a b c s = r)
  (h_rs : r * s = 2017)
  (h_distinct : r ≠ s) :
  ∀ a' : ℕ, (∃ b' c' : ℕ, 
    is_arithmetic_sequence a' b' c' ∧ 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧
    a' < b' ∧ b' < c' ∧
    (∃ r' s' : ℝ, f a' b' c' r' = s' ∧ f a' b' c' s' = r' ∧ r' * s' = 2017 ∧ r' ≠ s')) →
  a' ≥ 9 :=
sorry

end smallest_a_is_9_l49_4989


namespace geometric_series_sum_l49_4935

/-- Sum of a finite geometric series -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the first 6 terms of the geometric series with first term 1/4 and common ratio 1/4 -/
theorem geometric_series_sum :
  geometricSum (1/4 : ℚ) (1/4 : ℚ) 6 = 4095/12288 := by
  sorry

end geometric_series_sum_l49_4935


namespace sum_of_a_and_b_l49_4945

theorem sum_of_a_and_b (a b : ℝ) (ha : |a| = 4) (hb : |b| = 7) (hab : a < b) :
  a + b = 3 ∨ a + b = 11 := by
  sorry

end sum_of_a_and_b_l49_4945


namespace ralph_peanuts_l49_4985

-- Define the initial number of peanuts
def initial_peanuts : ℕ := 74

-- Define the number of peanuts lost
def peanuts_lost : ℕ := 59

-- Theorem to prove
theorem ralph_peanuts : initial_peanuts - peanuts_lost = 15 := by
  sorry

end ralph_peanuts_l49_4985


namespace exists_multiple_with_odd_digit_sum_l49_4940

/-- Sum of digits of a natural number in decimal notation -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- Theorem: For any natural number M, there exists a multiple of M with an odd sum of digits -/
theorem exists_multiple_with_odd_digit_sum (M : ℕ) : 
  ∃ k : ℕ, M ∣ k ∧ isOdd (sumOfDigits k) := by sorry

end exists_multiple_with_odd_digit_sum_l49_4940


namespace weight_lifting_equivalence_l49_4913

/-- Given that Max originally lifts two 30-pound weights 10 times, this theorem proves
    that he needs to lift two 25-pound weights 12 times to match the original total weight. -/
theorem weight_lifting_equivalence :
  let original_weight : ℕ := 30
  let original_reps : ℕ := 10
  let new_weight : ℕ := 25
  let total_weight : ℕ := 2 * original_weight * original_reps
  ∃ (n : ℕ), 2 * new_weight * n = total_weight ∧ n = 12 :=
by sorry

end weight_lifting_equivalence_l49_4913


namespace expression_value_l49_4928

theorem expression_value (a b : ℝ) (h : a^2 + 2*a*b + b^2 = 0) :
  a*(a + 4*b) - (a + 2*b)*(a - 2*b) = 0 := by
sorry

end expression_value_l49_4928


namespace parabola_properties_l49_4969

def parabola (x : ℝ) : ℝ := (x + 2)^2 - 1

theorem parabola_properties :
  (∀ x y : ℝ, parabola x = y → y = (x + 2)^2 - 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → parabola x₁ < parabola x₂) ∧
  (∀ x : ℝ, parabola x ≥ parabola (-2)) ∧
  (parabola (-2) = -1) ∧
  (∀ x₁ x₂ : ℝ, x₁ < -2 ∧ -2 < x₂ → parabola x₁ = parabola x₂ → x₁ + x₂ = -4) ∧
  (∀ x : ℝ, x > -2 → ∀ h : ℝ, h > 0 → parabola (x + h) > parabola x) :=
by sorry

end parabola_properties_l49_4969


namespace books_per_shelf_l49_4923

/-- Given four shelves with books and a round-trip distance, 
    prove the number of books on each shelf. -/
theorem books_per_shelf 
  (num_shelves : ℕ) 
  (round_trip_distance : ℕ) 
  (h1 : num_shelves = 4)
  (h2 : round_trip_distance = 3200)
  (h3 : ∃ (books_per_shelf : ℕ), 
    num_shelves * books_per_shelf = round_trip_distance / 2) :
  ∃ (books_per_shelf : ℕ), books_per_shelf = 400 :=
by
  sorry

end books_per_shelf_l49_4923


namespace fruits_in_box_l49_4977

/-- The number of fruits in a box after adding persimmons and apples -/
theorem fruits_in_box (persimmons apples : ℕ) : persimmons = 2 → apples = 7 → persimmons + apples = 9 := by
  sorry

end fruits_in_box_l49_4977


namespace amoeba_bacteria_ratio_l49_4971

-- Define the initial number of amoebas and bacteria
def initial_amoeba : ℕ := sorry
def initial_bacteria : ℕ := sorry

-- Define the number of days
def days : ℕ := 100

-- Define the function for the number of amoebas on day n
def amoeba (n : ℕ) : ℕ := 2^(n-1) * initial_amoeba

-- Define the function for the number of bacteria on day n after predation
def bacteria_after_predation (n : ℕ) : ℕ := 2^(n-1) * (initial_bacteria - initial_amoeba)

theorem amoeba_bacteria_ratio :
  bacteria_after_predation days = 0 → initial_amoeba = initial_bacteria := by sorry

end amoeba_bacteria_ratio_l49_4971


namespace books_lost_l49_4915

/-- Given that Sandy has 10 books, Tim has 33 books, and they now have 19 books together,
    prove that Benny lost 24 books. -/
theorem books_lost (sandy_books tim_books total_books_now : ℕ)
  (h1 : sandy_books = 10)
  (h2 : tim_books = 33)
  (h3 : total_books_now = 19) :
  sandy_books + tim_books - total_books_now = 24 := by
  sorry

end books_lost_l49_4915


namespace isosceles_triangle_sides_l49_4942

/-- An isosceles triangle with perimeter 20 -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  perimeter_eq : base + 2 * leg = 20

/-- The lengths of the sides when each leg is twice the base -/
def legsTwiceBase (t : IsoscelesTriangle) : Prop :=
  t.leg = 2 * t.base ∧ t.base = 4 ∧ t.leg = 8

/-- The lengths of the sides when one side is 6 -/
def oneSideSix (t : IsoscelesTriangle) : Prop :=
  (t.base = 6 ∧ t.leg = 7) ∨ (t.base = 8 ∧ t.leg = 6)

theorem isosceles_triangle_sides :
  (∀ t : IsoscelesTriangle, t.leg = 2 * t.base → legsTwiceBase t) ∧
  (∀ t : IsoscelesTriangle, (t.base = 6 ∨ t.leg = 6) → oneSideSix t) := by sorry

end isosceles_triangle_sides_l49_4942


namespace delaney_travel_time_l49_4901

/-- The time (in minutes) when the bus leaves, relative to midnight -/
def bus_departure_time : ℕ := 8 * 60

/-- The time (in minutes) when Delaney left home, relative to midnight -/
def delaney_departure_time : ℕ := 7 * 60 + 50

/-- The time (in minutes) that Delaney missed the bus by -/
def missed_by : ℕ := 20

/-- The time (in minutes) it takes Delaney to reach the pick-up point -/
def travel_time : ℕ := bus_departure_time + missed_by - delaney_departure_time

theorem delaney_travel_time : travel_time = 30 := by
  sorry

end delaney_travel_time_l49_4901


namespace base_ten_to_seven_l49_4961

theorem base_ten_to_seven : 2023 = 5 * 7^3 + 6 * 7^2 + 2 * 7^1 + 0 * 7^0 := by
  sorry

end base_ten_to_seven_l49_4961


namespace three_rug_overlap_l49_4967

theorem three_rug_overlap (A B C X Y Z : ℝ) 
  (h1 : A + B + C = 200) 
  (h2 : X + Y + Z = 140) 
  (h3 : Y = 22) 
  (h4 : X + 2*Y + 3*Z = A + B + C) : 
  Z = 19 := by
sorry

end three_rug_overlap_l49_4967


namespace smallest_divisible_by_1_to_10_l49_4925

theorem smallest_divisible_by_1_to_10 : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end smallest_divisible_by_1_to_10_l49_4925


namespace negation_of_proposition_l49_4951

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by sorry

end negation_of_proposition_l49_4951


namespace parallel_tangents_sum_bound_l49_4988

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k + 4/k) * Real.log x + (4 - x^2) / x

theorem parallel_tangents_sum_bound (k : ℝ) (x₁ x₂ : ℝ) (h_k : k ≥ 4) 
  (h_distinct : x₁ ≠ x₂) (h_positive : x₁ > 0 ∧ x₂ > 0) 
  (h_parallel : (deriv (f k)) x₁ = (deriv (f k)) x₂) :
  x₁ + x₂ > 16/5 := by
  sorry

end parallel_tangents_sum_bound_l49_4988


namespace quadratic_real_roots_condition_l49_4964

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + m = 0) → m ≤ 9/4 := by
  sorry

end quadratic_real_roots_condition_l49_4964


namespace B_power_2017_l49_4996

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_2017 : B^2017 = B := by sorry

end B_power_2017_l49_4996


namespace circle_circumference_area_relation_l49_4958

theorem circle_circumference_area_relation : 
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 12 → π * d < 10 * (π * d^2 / 4) := by
  sorry

#check circle_circumference_area_relation

end circle_circumference_area_relation_l49_4958


namespace lunch_group_size_l49_4922

/-- The number of people having lunch, including Benny -/
def num_people : ℕ := 3

/-- The cost of one lunch special in dollars -/
def lunch_cost : ℕ := 8

/-- The total bill in dollars -/
def total_bill : ℕ := 24

theorem lunch_group_size :
  num_people * lunch_cost = total_bill :=
by sorry

end lunch_group_size_l49_4922


namespace abc_equation_l49_4949

theorem abc_equation (a b c p : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq1 : a + 2/b = p)
  (h_eq2 : b + 2/c = p)
  (h_eq3 : c + 2/a = p) :
  a * b * c + 2 * p = 0 := by
sorry

end abc_equation_l49_4949


namespace age_difference_l49_4983

theorem age_difference (A B C : ℤ) (h1 : C = A - 12) : A + B - (B + C) = 12 := by
  sorry

end age_difference_l49_4983


namespace power_two_equality_l49_4965

theorem power_two_equality (m : ℕ) : 2^m = 2 * 16^2 * 4^3 * 8 → m = 18 := by
  sorry

end power_two_equality_l49_4965


namespace xyz_value_l49_4976

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9)
  (h3 : x + y + z = 3) :
  x * y * z = 5 := by
  sorry

end xyz_value_l49_4976


namespace solution_set_f_greater_than_7_range_of_m_for_solution_exists_l49_4962

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 2| + |2*x - 3|

-- Theorem for the solution set of f(x) > 7
theorem solution_set_f_greater_than_7 :
  {x : ℝ | f x > 7} = {x : ℝ | x < -3/2 ∨ x > 2} :=
sorry

-- Theorem for the range of m
theorem range_of_m_for_solution_exists :
  {m : ℝ | ∃ x, f x ≤ |3*m - 2|} = {m : ℝ | m ≤ -1 ∨ m ≥ 7/3} :=
sorry

end solution_set_f_greater_than_7_range_of_m_for_solution_exists_l49_4962


namespace second_customer_regular_hours_l49_4931

/-- Represents the hourly rates and customer data for an online service -/
structure OnlineService where
  regularRate : ℝ
  premiumRate : ℝ
  customer1PremiumHours : ℝ
  customer1RegularHours : ℝ
  customer1TotalCharge : ℝ
  customer2PremiumHours : ℝ
  customer2TotalCharge : ℝ

/-- Calculates the number of regular hours for the second customer -/
def calculateCustomer2RegularHours (service : OnlineService) : ℝ :=
  -- Implementation not required for the statement
  sorry

/-- Theorem stating that the second customer spent 3 regular hours -/
theorem second_customer_regular_hours (service : OnlineService) 
  (h1 : service.customer1PremiumHours = 2)
  (h2 : service.customer1RegularHours = 9)
  (h3 : service.customer1TotalCharge = 28)
  (h4 : service.customer2PremiumHours = 3)
  (h5 : service.customer2TotalCharge = 27) :
  calculateCustomer2RegularHours service = 3 := by
  sorry

#eval "Lean 4 statement generated successfully."

end second_customer_regular_hours_l49_4931


namespace function_properties_l49_4900

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin x * Real.cos x + Real.sin x ^ 2 - 3 / 2

theorem function_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (A B C a b c : ℝ),
    f C = 0 →
    c = 3 →
    2 * Real.sin A - Real.sin B = 0 →
    a ^ 2 + b ^ 2 - 2 * a * b * Real.cos C = c ^ 2 →
    a = Real.sqrt 3 ∧ b = 2 * Real.sqrt 3) :=
by sorry

end function_properties_l49_4900


namespace xy_ratio_values_l49_4997

theorem xy_ratio_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : 2 * x^2 + 2 * y^2 = 5 * x * y) : 
  (x + y) / (x - y) = 3 ∨ (x + y) / (x - y) = -3 := by
  sorry

end xy_ratio_values_l49_4997


namespace parabola_with_directrix_neg_two_l49_4975

/-- A parabola is defined by its directrix and focus. -/
structure Parabola where
  directrix : ℝ  -- y-coordinate of the directrix
  focus : ℝ × ℝ  -- (x, y) coordinates of the focus

/-- The standard equation of a parabola with a vertical axis of symmetry. -/
def standardEquation (p : Parabola) : Prop :=
  ∀ x y : ℝ, (x ^ 2 = 4 * p.focus.2 * y) ↔ (y - p.focus.2) ^ 2 = (x - p.focus.1) ^ 2 + (y - p.directrix) ^ 2

theorem parabola_with_directrix_neg_two (p : Parabola) 
  (h : p.directrix = -2) : 
  standardEquation p ↔ ∀ x y : ℝ, x ^ 2 = 8 * y :=
sorry

end parabola_with_directrix_neg_two_l49_4975


namespace min_product_of_three_min_product_is_neg_720_l49_4959

def S : Finset Int := {-10, -7, -3, 0, 2, 4, 8, 9}

theorem min_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
  x ≠ y → y ≠ z → x ≠ z → 
  a * b * c ≥ x * y * z :=
by
  sorry

theorem min_product_is_neg_720 : 
  ∃ a b c : Int, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a * b * c = -720 ∧
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
   x ≠ y → y ≠ z → x ≠ z → 
   x * y * z ≥ -720) :=
by
  sorry

end min_product_of_three_min_product_is_neg_720_l49_4959


namespace job_completion_time_l49_4937

theorem job_completion_time (x : ℝ) : 
  x > 0 → -- A's completion time is positive
  4 * (1/x + 1/20) = 1 - 0.5333333333333333 → -- Condition from working together
  x = 15 := by
    sorry

end job_completion_time_l49_4937


namespace dollar_op_neg_three_neg_four_l49_4929

def dollar_op (x y : Int) : Int := x * (y + 1) + x * y

theorem dollar_op_neg_three_neg_four : dollar_op (-3) (-4) = 21 := by
  sorry

end dollar_op_neg_three_neg_four_l49_4929


namespace find_m_value_l49_4980

theorem find_m_value (m : ℤ) : 
  (∃ (a b : ℝ), ∀ (x : ℝ), (2/3 * (m + 4) * x^(|m| - 3) + 6 > 0) ↔ (a * x + b > 0)) →
  (m + 4 ≠ 0) →
  m = 4 := by
sorry

end find_m_value_l49_4980


namespace march_greatest_drop_l49_4963

/-- Represents the months in the first half of 2021 -/
inductive Month
| january
| february
| march
| april
| may
| june

/-- The price change for each month -/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.january  => -3.00
  | Month.february => 1.50
  | Month.march    => -4.50
  | Month.april    => 2.00
  | Month.may      => -1.00
  | Month.june     => 0.50

/-- The month with the greatest price drop -/
def greatest_drop : Month := Month.march

theorem march_greatest_drop :
  ∀ m : Month, price_change greatest_drop ≤ price_change m :=
by sorry

end march_greatest_drop_l49_4963


namespace speaking_orders_eq_720_l49_4919

/-- The number of different speaking orders for selecting 4 students from a group of 7 students,
    including students A and B, with the requirement that at least one of A or B must participate. -/
def speakingOrders : ℕ :=
  Nat.descFactorial 7 4 - Nat.descFactorial 5 4

/-- Theorem stating that the number of speaking orders is 720. -/
theorem speaking_orders_eq_720 : speakingOrders = 720 := by
  sorry

end speaking_orders_eq_720_l49_4919


namespace fraction_calculation_l49_4979

theorem fraction_calculation : (3 / 4 + 2 + 1 / 3) / (1 + 1 / 2) = 37 / 18 := by
  sorry

end fraction_calculation_l49_4979


namespace inverse_functions_l49_4930

/-- A function type representing the described graphs --/
inductive FunctionGraph
  | Parabola
  | StraightLine
  | HorizontalLine
  | Semicircle
  | CubicFunction

/-- Predicate to determine if a function graph has an inverse --/
def has_inverse (f : FunctionGraph) : Prop :=
  match f with
  | FunctionGraph.StraightLine => true
  | FunctionGraph.Semicircle => true
  | _ => false

/-- Theorem stating which function graphs have inverses --/
theorem inverse_functions (f : FunctionGraph) :
  has_inverse f ↔ (f = FunctionGraph.StraightLine ∨ f = FunctionGraph.Semicircle) :=
sorry

end inverse_functions_l49_4930


namespace sam_read_100_pages_l49_4917

def minimum_assigned : ℕ := 25

def harrison_extra : ℕ := 10

def pam_extra : ℕ := 15

def sam_multiplier : ℕ := 2

def harrison_pages : ℕ := minimum_assigned + harrison_extra

def pam_pages : ℕ := harrison_pages + pam_extra

def sam_pages : ℕ := sam_multiplier * pam_pages

theorem sam_read_100_pages : sam_pages = 100 := by
  sorry

end sam_read_100_pages_l49_4917


namespace intersection_equals_interval_l49_4934

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 1) / Real.log 10}
def B : Set ℝ := {x | -x^2 + 2 ≤ 2}

-- Define the open interval (1, 2]
def openClosedInterval : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem intersection_equals_interval : A ∩ B = openClosedInterval := by sorry

end intersection_equals_interval_l49_4934


namespace tree_height_problem_l49_4952

theorem tree_height_problem (h1 h2 : ℝ) : 
  h1 = h2 + 20 →  -- One tree is 20 feet taller than the other
  h2 / h1 = 5 / 7 →  -- The heights are in the ratio 5:7
  h1 = 70 := by  -- The height of the taller tree is 70 feet
sorry

end tree_height_problem_l49_4952


namespace basic_structures_are_sequential_conditional_loop_modular_not_basic_structure_l49_4992

/-- The set of basic algorithm structures -/
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop
  | Modular

/-- The set of basic algorithm structures contains exactly Sequential, Conditional, and Loop -/
def basic_structures : Set AlgorithmStructure :=
  {AlgorithmStructure.Sequential, AlgorithmStructure.Conditional, AlgorithmStructure.Loop}

/-- The theorem stating that the basic structures are exactly Sequential, Conditional, and Loop -/
theorem basic_structures_are_sequential_conditional_loop :
  basic_structures = {AlgorithmStructure.Sequential, AlgorithmStructure.Conditional, AlgorithmStructure.Loop} :=
by sorry

/-- The theorem stating that Modular is not a basic structure -/
theorem modular_not_basic_structure :
  AlgorithmStructure.Modular ∉ basic_structures :=
by sorry

end basic_structures_are_sequential_conditional_loop_modular_not_basic_structure_l49_4992


namespace path_area_l49_4970

/-- The area of a ring-shaped path around a circular lawn -/
theorem path_area (r : ℝ) (w : ℝ) (h1 : r = 35) (h2 : w = 7) :
  (π * (r + w)^2 - π * r^2) = 539 * π :=
sorry

end path_area_l49_4970


namespace chord_length_l49_4933

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by sorry

end chord_length_l49_4933


namespace cube_side_length_l49_4987

-- Define the constants
def paint_cost_per_kg : ℝ := 60
def area_covered_per_kg : ℝ := 20
def total_paint_cost : ℝ := 1800
def num_cube_sides : ℕ := 6

-- Define the theorem
theorem cube_side_length :
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    side_length^2 * num_cube_sides * paint_cost_per_kg / area_covered_per_kg = total_paint_cost ∧
    side_length = 10 := by
  sorry

end cube_side_length_l49_4987


namespace necklace_profit_l49_4947

/-- Calculate the profit from selling necklaces -/
theorem necklace_profit
  (charms_per_necklace : ℕ)
  (charm_cost : ℕ)
  (selling_price : ℕ)
  (necklaces_sold : ℕ)
  (h1 : charms_per_necklace = 10)
  (h2 : charm_cost = 15)
  (h3 : selling_price = 200)
  (h4 : necklaces_sold = 30) :
  (selling_price - charms_per_necklace * charm_cost) * necklaces_sold = 1500 :=
by sorry

end necklace_profit_l49_4947


namespace problem_solution_l49_4932

theorem problem_solution : (42 / (9 - 2 + 3)) * 7 = 29.4 := by
  sorry

end problem_solution_l49_4932


namespace quadratic_equation_solution_l49_4918

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 9
  ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -3 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end quadratic_equation_solution_l49_4918


namespace periodic_function_proof_l49_4936

open Real

theorem periodic_function_proof (a : ℚ) (b d c : ℝ) 
  (f : ℝ → ℝ) 
  (h_range : ∀ x, f x ∈ Set.Icc (-1) 1)
  (h_eq : ∀ x, f (x + a + b) - f (x + b) = c * (x + 2 * a + ⌊x⌋ - 2 * ⌊x + a⌋ - ⌊b⌋) + d) :
  ∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x :=
sorry

end periodic_function_proof_l49_4936


namespace equation_solution_l49_4950

theorem equation_solution : 
  {x : ℝ | (5 - 2*x)^(x + 1) = 1} = {-1, 2, 3} := by sorry

end equation_solution_l49_4950


namespace platform_length_platform_length_is_200_l49_4902

/-- Given a train traveling at 72 kmph that crosses a platform in 30 seconds and a man in 20 seconds, 
    the length of the platform is 200 meters. -/
theorem platform_length 
  (train_speed : ℝ) 
  (time_platform : ℝ) 
  (time_man : ℝ) 
  (h1 : train_speed = 72) 
  (h2 : time_platform = 30) 
  (h3 : time_man = 20) : ℝ := by
  
  -- Convert train speed from kmph to m/s
  let train_speed_ms := train_speed * 1000 / 3600

  -- Calculate length of train
  let train_length := train_speed_ms * time_man

  -- Calculate total distance (train + platform)
  let total_distance := train_speed_ms * time_platform

  -- Calculate platform length
  let platform_length := total_distance - train_length

  -- Prove that platform_length = 200
  sorry

/-- The length of the platform is 200 meters -/
theorem platform_length_is_200 : platform_length 72 30 20 rfl rfl rfl = 200 := by sorry

end platform_length_platform_length_is_200_l49_4902


namespace perfect_square_prob_l49_4938

/-- A function that represents the number of ways to roll a 10-sided die n times
    such that the product of the rolls is a perfect square -/
def b : ℕ → ℕ
  | 0 => 1
  | n + 1 => 10^n + 2 * b n

/-- The probability of rolling a 10-sided die 4 times and getting a product
    that is a perfect square -/
def prob_perfect_square : ℚ :=
  b 4 / 10^4

theorem perfect_square_prob :
  prob_perfect_square = 316 / 2500 := by
  sorry

end perfect_square_prob_l49_4938


namespace factorization_equality_l49_4984

theorem factorization_equality (x y : ℝ) : 
  y^2 + x*y - 3*x - y - 6 = (y - 3) * (y + 2 + x) := by
  sorry

end factorization_equality_l49_4984


namespace impossibleToMakeAllMultiplesOfTen_l49_4909

/-- Represents an 8x8 grid of integers -/
def Grid := Fin 8 → Fin 8 → ℤ

/-- Represents an operation on the grid -/
inductive Operation
| threeByThree (i j : Fin 8) : Operation
| fourByFour (i j : Fin 8) : Operation

/-- Apply an operation to a grid -/
def applyOperation (g : Grid) (op : Operation) : Grid :=
  sorry

/-- Check if all numbers in the grid are multiples of 10 -/
def allMultiplesOfTen (g : Grid) : Prop :=
  ∀ i j, ∃ k, g i j = 10 * k

/-- The main theorem -/
theorem impossibleToMakeAllMultiplesOfTen :
  ∃ (g : Grid),
    (∀ i j, g i j ≥ 0) ∧
    ¬∃ (ops : List Operation), allMultiplesOfTen (ops.foldl applyOperation g) :=
  sorry

end impossibleToMakeAllMultiplesOfTen_l49_4909


namespace max_sum_semi_axes_l49_4978

/-- The maximum sum of semi-axes of an ellipse and hyperbola with the same foci -/
theorem max_sum_semi_axes (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x y : ℝ, x^2/25 + y^2/m^2 = 1) →
  (∃ x y : ℝ, x^2/7 - y^2/n^2 = 1) →
  (25 - m^2 = 7 + n^2) →
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → 
    (∃ x y : ℝ, x^2/25 + y^2/m'^2 = 1) →
    (∃ x y : ℝ, x^2/7 - y^2/n'^2 = 1) →
    (25 - m'^2 = 7 + n'^2) →
    m + n ≥ m' + n') →
  m + n = 6 :=
by sorry

end max_sum_semi_axes_l49_4978


namespace octal_sum_units_digit_l49_4986

/-- The units digit of the sum of two octal numbers -/
def octal_units_digit_sum (a b : ℕ) : ℕ :=
  (a % 8 + b % 8) % 8

/-- Theorem: The units digit of 53₈ + 64₈ in base 8 is 7 -/
theorem octal_sum_units_digit :
  octal_units_digit_sum 53 64 = 7 := by
  sorry

end octal_sum_units_digit_l49_4986


namespace mersenne_last_two_digits_l49_4941

/-- The exponent used in the Mersenne prime -/
def p : ℕ := 82589933

/-- The Mersenne number -/
def mersenne_number : ℕ := 2^p - 1

/-- The last two digits of a number -/
def last_two_digits (n : ℕ) : ℕ := n % 100

theorem mersenne_last_two_digits : last_two_digits mersenne_number = 91 := by
  sorry

end mersenne_last_two_digits_l49_4941


namespace odd_plus_one_even_implies_f_four_zero_l49_4957

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_plus_one_even_implies_f_four_zero (f : ℝ → ℝ) 
  (h_odd : is_odd f) (h_even : is_even (fun x ↦ f (x + 1))) : f 4 = 0 := by
  sorry

end odd_plus_one_even_implies_f_four_zero_l49_4957
