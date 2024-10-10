import Mathlib

namespace vacation_probability_l721_72163

theorem vacation_probability (prob_A prob_B : ℝ) 
  (h1 : prob_A = 1/4)
  (h2 : prob_B = 1/5)
  (h3 : 0 ≤ prob_A ∧ prob_A ≤ 1)
  (h4 : 0 ≤ prob_B ∧ prob_B ≤ 1) :
  1 - (1 - prob_A) * (1 - prob_B) = 2/5 := by
sorry

end vacation_probability_l721_72163


namespace contrapositive_real_roots_l721_72196

theorem contrapositive_real_roots (m : ℝ) :
  (¬(∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0) ↔
  (m > 0 → ∃ x : ℝ, x^2 + x - m = 0) :=
by sorry

end contrapositive_real_roots_l721_72196


namespace right_triangle_perimeter_l721_72176

theorem right_triangle_perimeter (area : ℝ) (leg : ℝ) (h1 : area = 120) (h2 : leg = 24) :
  ∃ (other_leg hypotenuse : ℝ),
    area = (1 / 2) * leg * other_leg ∧
    hypotenuse ^ 2 = leg ^ 2 + other_leg ^ 2 ∧
    leg + other_leg + hypotenuse = 60 := by
  sorry

end right_triangle_perimeter_l721_72176


namespace three_digit_number_divisible_by_11_l721_72179

theorem three_digit_number_divisible_by_11 :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 6 ∧ (n / 100) % 10 = 3 ∧ n % 11 = 0 ∧ n = 396 := by
  sorry

end three_digit_number_divisible_by_11_l721_72179


namespace z_in_first_quadrant_l721_72139

def z : ℂ := (3 - Complex.I) * (1 + Complex.I)

theorem z_in_first_quadrant : z.re > 0 ∧ z.im > 0 := by
  sorry

end z_in_first_quadrant_l721_72139


namespace grapes_problem_l721_72170

theorem grapes_problem (bryce_grapes : ℚ) : 
  (∃ (carter_grapes : ℚ), 
    bryce_grapes = carter_grapes + 7 ∧ 
    carter_grapes = bryce_grapes / 3) → 
  bryce_grapes = 21 / 2 := by
sorry

end grapes_problem_l721_72170


namespace trig_identities_l721_72106

theorem trig_identities (α : Real) (h : Real.sin α = 2 * Real.cos α) : 
  ((2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4) ∧
  (Real.sin α ^ 2 + Real.sin α * Real.cos α - 2 * Real.cos α ^ 2 = 4/5) := by
  sorry

end trig_identities_l721_72106


namespace house_problem_l721_72142

theorem house_problem (total garage pool both : ℕ) 
  (h_total : total = 65)
  (h_garage : garage = 50)
  (h_pool : pool = 40)
  (h_both : both = 35) :
  total - (garage + pool - both) = 10 := by
  sorry

end house_problem_l721_72142


namespace max_value_implies_m_equals_20_l721_72151

/-- The function f(x) = -x^3 + 6x^2 - m --/
def f (x m : ℝ) : ℝ := -x^3 + 6*x^2 - m

/-- The maximum value of f(x) is 12 --/
def max_value : ℝ := 12

theorem max_value_implies_m_equals_20 :
  (∃ x₀ : ℝ, ∀ x : ℝ, f x m ≤ f x₀ m) ∧ (∃ x₁ : ℝ, f x₁ m = max_value) → m = 20 := by
  sorry

end max_value_implies_m_equals_20_l721_72151


namespace max_erasable_digits_l721_72171

/-- Represents the number of digits in the original number -/
def total_digits : ℕ := 1000

/-- Represents the sum of digits we want to maintain after erasure -/
def target_sum : ℕ := 2018

/-- Represents the repetitive pattern in the original number -/
def pattern : List ℕ := [2, 0, 1, 8]

/-- Represents the sum of digits in one repetition of the pattern -/
def pattern_sum : ℕ := pattern.sum

/-- Represents the number of complete repetitions of the pattern in the original number -/
def repetitions : ℕ := total_digits / pattern.length

theorem max_erasable_digits : 
  ∃ (erasable : ℕ), 
    erasable = total_digits - (target_sum / pattern_sum * pattern.length + target_sum % pattern_sum) ∧
    erasable = 741 := by sorry

end max_erasable_digits_l721_72171


namespace brand_a_most_cost_effective_l721_72169

/-- Represents a chocolate bar brand with its price and s'mores per bar -/
structure ChocolateBar where
  price : ℝ
  smoresPerBar : ℕ

/-- Calculates the cost of chocolate bars for a given number of s'mores -/
def calculateCost (bar : ChocolateBar) (numSmores : ℕ) : ℝ :=
  let numBars := (numSmores + bar.smoresPerBar - 1) / bar.smoresPerBar
  let cost := numBars * bar.price
  if numBars ≥ 10 then cost * 0.85 else cost

/-- Proves that Brand A is the most cost-effective option for Ron's scout camp -/
theorem brand_a_most_cost_effective :
  let numScouts : ℕ := 15
  let smoresPerScout : ℕ := 2
  let brandA := ChocolateBar.mk 1.50 3
  let brandB := ChocolateBar.mk 2.10 4
  let brandC := ChocolateBar.mk 3.00 6
  let totalSmores := numScouts * smoresPerScout
  let costA := calculateCost brandA totalSmores
  let costB := calculateCost brandB totalSmores
  let costC := calculateCost brandC totalSmores
  (costA < costB ∧ costA < costC) ∧ costA = 12.75 := by
  sorry

end brand_a_most_cost_effective_l721_72169


namespace square_root_of_25_l721_72137

theorem square_root_of_25 : ∀ x : ℝ, x^2 = 25 ↔ x = 5 ∨ x = -5 := by sorry

end square_root_of_25_l721_72137


namespace p_h_neg_three_equals_eight_l721_72130

-- Define the function h
def h (x : ℝ) : ℝ := 2 * x^2 - 10

-- Define the theorem
theorem p_h_neg_three_equals_eight 
  (p : ℝ → ℝ) -- p is a function from reals to reals
  (h_def : ∀ x, h x = 2 * x^2 - 10) -- definition of h
  (p_h_three : p (h 3) = 8) -- given condition
  : p (h (-3)) = 8 := by
  sorry

end p_h_neg_three_equals_eight_l721_72130


namespace square_difference_formula_l721_72158

theorem square_difference_formula : 15^2 - 2*(15*5) + 5^2 = 100 := by
  sorry

end square_difference_formula_l721_72158


namespace f_of_g_composition_l721_72133

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x + 1

theorem f_of_g_composition : f (1 + g 2) = 8 := by
  sorry

end f_of_g_composition_l721_72133


namespace polynomial_degree_example_l721_72107

/-- The degree of a polynomial (3x^5 + 2x^4 - x^2 + 5)(4x^{11} - 8x^8 + 3x^5 - 10) - (x^3 + 7)^6 -/
theorem polynomial_degree_example : 
  let p₁ : Polynomial ℝ := X^5 * 3 + X^4 * 2 - X^2 + 5
  let p₂ : Polynomial ℝ := X^11 * 4 - X^8 * 8 + X^5 * 3 - 10
  let p₃ : Polynomial ℝ := (X^3 + 7)^6
  (p₁ * p₂ - p₃).degree = 18 := by
  sorry

end polynomial_degree_example_l721_72107


namespace circle_through_three_points_l721_72175

/-- A circle passing through three points -/
structure Circle3Points where
  O : ℝ × ℝ
  M₁ : ℝ × ℝ
  M₂ : ℝ × ℝ

/-- The equation of a circle in standard form -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Theorem: The circle passing through O(0,0), M₁(1,1), and M₂(4,2) 
    has the equation (x-4)² + (y+3)² = 25, with center (4, -3) and radius 5 -/
theorem circle_through_three_points :
  let c := Circle3Points.mk (0, 0) (1, 1) (4, 2)
  ∃ (h k r : ℝ),
    h = 4 ∧ k = -3 ∧ r = 5 ∧
    (∀ (x y : ℝ), CircleEquation h k r x y ↔
      ((x = c.O.1 ∧ y = c.O.2) ∨
       (x = c.M₁.1 ∧ y = c.M₁.2) ∨
       (x = c.M₂.1 ∧ y = c.M₂.2))) :=
by sorry

end circle_through_three_points_l721_72175


namespace parabola_through_point_l721_72156

/-- A parabola passing through the point (4, -2) has either the equation y² = x or x² = -8y -/
theorem parabola_through_point (P : ℝ × ℝ) (h : P = (4, -2)) :
  (∃ (x y : ℝ), y^2 = x ∧ P = (x, y)) ∨ (∃ (x y : ℝ), x^2 = -8*y ∧ P = (x, y)) := by
  sorry

end parabola_through_point_l721_72156


namespace library_books_count_l721_72194

/-- The number of bookshelves in the library -/
def num_bookshelves : ℕ := 28

/-- The number of floors in each bookshelf -/
def floors_per_bookshelf : ℕ := 6

/-- The number of books left on a floor after taking two books -/
def books_left_after_taking_two : ℕ := 20

/-- The total number of books in the library -/
def total_books : ℕ := num_bookshelves * floors_per_bookshelf * (books_left_after_taking_two + 2)

theorem library_books_count : total_books = 3696 := by
  sorry

end library_books_count_l721_72194


namespace dot_product_equals_eight_l721_72189

def a : Fin 2 → ℝ := ![0, 4]
def b : Fin 2 → ℝ := ![2, 2]

theorem dot_product_equals_eight :
  (Finset.univ.sum (λ i => a i * b i)) = 8 := by
  sorry

end dot_product_equals_eight_l721_72189


namespace gift_budget_calculation_l721_72178

theorem gift_budget_calculation (total_budget : ℕ) (num_friends : ℕ) (friend_gift_cost : ℕ) 
  (num_family : ℕ) : 
  total_budget = 200 → 
  num_friends = 12 → 
  friend_gift_cost = 15 → 
  num_family = 4 → 
  (total_budget - num_friends * friend_gift_cost) / num_family = 5 := by
sorry

end gift_budget_calculation_l721_72178


namespace marcus_baseball_cards_l721_72144

theorem marcus_baseball_cards 
  (initial_cards : ℝ) 
  (cards_from_carter : ℝ) 
  (h1 : initial_cards = 210.0) 
  (h2 : cards_from_carter = 58.0) : 
  initial_cards + cards_from_carter = 268.0 := by
sorry

end marcus_baseball_cards_l721_72144


namespace shop_profit_calculation_l721_72112

/-- Proves that the mean profit for the first 15 days is 285 Rs, given the conditions of the problem. -/
theorem shop_profit_calculation (total_days : ℕ) (mean_profit : ℚ) (last_half_mean : ℚ) :
  total_days = 30 →
  mean_profit = 350 →
  last_half_mean = 415 →
  (total_days * mean_profit - (total_days / 2) * last_half_mean) / (total_days / 2) = 285 := by
  sorry

end shop_profit_calculation_l721_72112


namespace square_side_length_l721_72131

-- Define the circumference of the largest inscribed circle
def circle_circumference : ℝ := 37.69911184307752

-- Define π as a constant (approximation)
def π : ℝ := 3.141592653589793

-- Theorem statement
theorem square_side_length (circle_circumference : ℝ) (π : ℝ) :
  let radius := circle_circumference / (2 * π)
  let diameter := 2 * radius
  diameter = 12 := by sorry

end square_side_length_l721_72131


namespace inequality_proof_l721_72109

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c ≤ 3) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) ≥ 3 / 2 := by
  sorry

end inequality_proof_l721_72109


namespace remainder_problem_l721_72166

theorem remainder_problem (y : ℤ) (h : y % 264 = 42) : y % 22 = 20 := by
  sorry

end remainder_problem_l721_72166


namespace unique_solution_cubic_equation_l721_72102

theorem unique_solution_cubic_equation :
  ∀ x y : ℕ+, x^3 - y^3 = x * y + 41 ↔ x = 5 ∧ y = 4 := by
sorry

end unique_solution_cubic_equation_l721_72102


namespace smallest_middle_term_l721_72150

theorem smallest_middle_term (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c → 
  (∃ d : ℝ, a = b - d ∧ c = b + d) → 
  a * b * c = 216 → 
  b ≥ 6 := by
sorry

end smallest_middle_term_l721_72150


namespace intersection_complement_theorem_l721_72193

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def N : Set ℝ := {y | ∃ x, y = x^2 + 1}

-- State the theorem
theorem intersection_complement_theorem :
  M ∩ (U \ N) = {x : ℝ | -1 ≤ x ∧ x < 1} := by sorry

end intersection_complement_theorem_l721_72193


namespace restaurant_gratuity_calculation_l721_72129

def calculate_gratuity (base_price : ℝ) (tax_rate : ℝ) (discount_rate : ℝ) (gratuity_rate : ℝ) : ℝ :=
  let discounted_price := base_price * (1 - discount_rate)
  let price_after_tax := discounted_price * (1 + tax_rate)
  price_after_tax * gratuity_rate

theorem restaurant_gratuity_calculation :
  let striploin_gratuity := calculate_gratuity 80 0.10 0.05 0.15
  let wine_gratuity := calculate_gratuity 10 0.15 0 0.20
  let dessert_gratuity := calculate_gratuity 12 0.05 0.10 0.10
  let water_gratuity := calculate_gratuity 3 0 0 0.05
  striploin_gratuity + wine_gratuity + dessert_gratuity + water_gratuity = 16.12 := by
  sorry

end restaurant_gratuity_calculation_l721_72129


namespace brianna_book_savings_l721_72138

theorem brianna_book_savings (m : ℚ) (p : ℚ) : 
  (1/4 : ℚ) * m = (1/2 : ℚ) * p → m - p = (1/2 : ℚ) * m :=
by
  sorry

end brianna_book_savings_l721_72138


namespace cloak_purchase_change_l721_72116

/-- Represents the price of an invisibility cloak and the change received in different scenarios --/
structure CloakPurchase where
  silver_paid : ℕ
  gold_change : ℕ

/-- Proves that buying an invisibility cloak for 14 gold coins results in a change of 10 silver coins --/
theorem cloak_purchase_change 
  (purchase1 : CloakPurchase)
  (purchase2 : CloakPurchase)
  (h1 : purchase1.silver_paid = 20 ∧ purchase1.gold_change = 4)
  (h2 : purchase2.silver_paid = 15 ∧ purchase2.gold_change = 1)
  (gold_paid : ℕ)
  (h3 : gold_paid = 14)
  : ∃ (silver_change : ℕ), silver_change = 10 := by
  sorry

end cloak_purchase_change_l721_72116


namespace uma_money_fraction_l721_72146

theorem uma_money_fraction (rita sam tina unknown : ℚ) : 
  rita > 0 ∧ sam > 0 ∧ tina > 0 ∧ unknown > 0 →
  rita / 6 = sam / 5 ∧ rita / 6 = tina / 7 ∧ rita / 6 = unknown / 8 →
  (rita / 6 + sam / 5 + tina / 7 + unknown / 8) / (rita + sam + tina + unknown) = 2 / 13 := by
sorry

end uma_money_fraction_l721_72146


namespace fourDigitNumbers_eq_14_l721_72161

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of 4-digit numbers formed using digits 2 and 3, where each number must include at least one occurrence of both digits -/
def fourDigitNumbers : ℕ :=
  choose 4 1 + choose 4 2 + choose 4 3

theorem fourDigitNumbers_eq_14 : fourDigitNumbers = 14 := by sorry

end fourDigitNumbers_eq_14_l721_72161


namespace t_value_l721_72186

theorem t_value : 
  let t := 2 / (1 - Real.rpow 2 (1/3))
  t = -2 * (1 + Real.rpow 2 (1/3) + Real.sqrt 2) := by sorry

end t_value_l721_72186


namespace square_plus_abs_eq_zero_l721_72122

theorem square_plus_abs_eq_zero (x y : ℝ) :
  x^2 + |y + 8| = 0 → x = 0 ∧ y = -8 := by
  sorry

end square_plus_abs_eq_zero_l721_72122


namespace inscribed_squares_ratio_l721_72119

/-- Represents a right triangle with sides a, b, and c -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2

/-- Represents a square inscribed in the triangle with one vertex at the right angle -/
def corner_square (t : RightTriangle) := 
  { x : ℝ // x > 0 ∧ x / t.a = x / t.b }

/-- Represents a square inscribed in the triangle with one side on the hypotenuse -/
def hypotenuse_square (t : RightTriangle) := 
  { y : ℝ // y > 0 ∧ y / t.a = y / t.b }

/-- The main theorem to be proved -/
theorem inscribed_squares_ratio 
  (t : RightTriangle) 
  (h : t.a = 5 ∧ t.b = 12 ∧ t.c = 13) :
  ∀ (x : corner_square t) (y : hypotenuse_square t), 
  x.val = y.val := by sorry

end inscribed_squares_ratio_l721_72119


namespace curve_asymptotes_sum_l721_72183

/-- A curve with equation y = x / (x^3 + Ax^2 + Bx + C) where A, B, and C are integers -/
structure Curve where
  A : ℤ
  B : ℤ
  C : ℤ

/-- The denominator of the curve equation -/
def Curve.denominator (c : Curve) (x : ℝ) : ℝ :=
  x^3 + c.A * x^2 + c.B * x + c.C

/-- A curve has a vertical asymptote at x = a if its denominator is zero at x = a -/
def has_vertical_asymptote (c : Curve) (a : ℝ) : Prop :=
  c.denominator a = 0

theorem curve_asymptotes_sum (c : Curve) 
  (h1 : has_vertical_asymptote c (-1))
  (h2 : has_vertical_asymptote c 2)
  (h3 : has_vertical_asymptote c 3) :
  c.A + c.B + c.C = -3 := by
  sorry

end curve_asymptotes_sum_l721_72183


namespace power_function_value_l721_72168

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x^a

-- State the theorem
theorem power_function_value 
  (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 4 = 1/2) : 
  f (1/16) = 4 := by
sorry

end power_function_value_l721_72168


namespace greatest_prime_factor_of_341_l721_72172

theorem greatest_prime_factor_of_341 : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ 341 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 341 → q ≤ p :=
by
  -- The proof would go here
  sorry

end greatest_prime_factor_of_341_l721_72172


namespace rowing_round_trip_time_l721_72174

/-- Proves that the total time to row to a place and back is 1 hour, given the specified conditions -/
theorem rowing_round_trip_time
  (rowing_speed : ℝ)
  (current_speed : ℝ)
  (distance : ℝ)
  (h1 : rowing_speed = 5)
  (h2 : current_speed = 1)
  (h3 : distance = 2.4)
  : (distance / (rowing_speed + current_speed)) + (distance / (rowing_speed - current_speed)) = 1 := by
  sorry

end rowing_round_trip_time_l721_72174


namespace dispatch_methods_count_l721_72114

def num_male_servants : ℕ := 5
def num_female_servants : ℕ := 4
def num_total_servants : ℕ := num_male_servants + num_female_servants
def num_selected : ℕ := 3
def num_areas : ℕ := 3

theorem dispatch_methods_count :
  (Nat.choose num_total_servants num_selected - 
   Nat.choose num_male_servants num_selected - 
   Nat.choose num_female_servants num_selected) * 
  (Nat.factorial num_selected) = 420 := by
  sorry

end dispatch_methods_count_l721_72114


namespace rhombus_area_l721_72181

/-- The area of a rhombus with side length √117 and diagonals differing by 10 units is 72 square units. -/
theorem rhombus_area (s : ℝ) (d₁ d₂ : ℝ) (h₁ : s = Real.sqrt 117) (h₂ : d₂ - d₁ = 10) 
  (h₃ : s^2 = (d₁/2)^2 + (d₂/2)^2) : d₁ * d₂ / 2 = 72 := by
  sorry

#check rhombus_area

end rhombus_area_l721_72181


namespace christen_peeled_21_potatoes_l721_72147

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  initial_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  time_before_christen : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def christenPeeledPotatoes (scenario : PotatoPeeling) : ℕ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that Christen peeled 21 potatoes in the given scenario -/
theorem christen_peeled_21_potatoes :
  let scenario : PotatoPeeling := {
    initial_potatoes := 60,
    homer_rate := 4,
    christen_rate := 6,
    time_before_christen := 6
  }
  christenPeeledPotatoes scenario = 21 := by
  sorry

end christen_peeled_21_potatoes_l721_72147


namespace angle_inclination_range_l721_72155

theorem angle_inclination_range (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ((-a - 2 + 1) * (Real.sqrt 3 / 3 * a + 1) > 0)) :
  ∃ α : ℝ, (2 * Real.pi / 3 < α) ∧ (α < 3 * Real.pi / 4) ∧ 
  (a = Real.tan α) :=
sorry

end angle_inclination_range_l721_72155


namespace prob_both_white_l721_72103

/-- Represents an urn with white and black balls -/
structure Urn :=
  (white : ℕ)
  (black : ℕ)

/-- Calculates the probability of drawing a white ball from an urn -/
def prob_white (u : Urn) : ℚ :=
  u.white / (u.white + u.black)

/-- The first urn -/
def urn1 : Urn := ⟨2, 10⟩

/-- The second urn -/
def urn2 : Urn := ⟨8, 4⟩

/-- Theorem: The probability of drawing white balls from both urns is 1/9 -/
theorem prob_both_white : prob_white urn1 * prob_white urn2 = 1 / 9 := by
  sorry

end prob_both_white_l721_72103


namespace company_production_days_l721_72105

/-- Given a company's production data, prove the number of past days. -/
theorem company_production_days (n : ℕ) : 
  (∀ (P : ℕ), P = 80 * n) →  -- Average daily production for past n days
  (∀ (new_total : ℕ), new_total = 80 * n + 220) →  -- Total including today's production
  (∀ (new_avg : ℝ), new_avg = (80 * n + 220) / (n + 1)) →  -- New average
  (new_avg = 95) →  -- New average is 95
  n = 8 := by sorry

end company_production_days_l721_72105


namespace quadratic_root_problem_l721_72104

theorem quadratic_root_problem (m : ℝ) : 
  ((0 : ℝ) = 0 → (m - 2) * 0^2 + 4 * 0 + 2 - |m| = 0) ∧ 
  (m - 2 ≠ 0) → 
  m = -2 := by
  sorry

end quadratic_root_problem_l721_72104


namespace theater_hall_seats_l721_72149

/-- Represents a theater hall with three categories of seats. -/
structure TheaterHall where
  totalSeats : ℕ
  categoryIPrice : ℕ
  categoryIIPrice : ℕ
  categoryIIIPrice : ℕ
  freeTickets : ℕ
  revenueDifference : ℕ

/-- Checks if the theater hall satisfies all given conditions. -/
def validTheaterHall (hall : TheaterHall) : Prop :=
  (hall.totalSeats % 5 = 0) ∧
  (hall.categoryIPrice = 220) ∧
  (hall.categoryIIPrice = 200) ∧
  (hall.categoryIIIPrice = 180) ∧
  (hall.freeTickets = 150) ∧
  (hall.revenueDifference = 4320)

/-- Theorem stating that a valid theater hall has 360 seats. -/
theorem theater_hall_seats (hall : TheaterHall) 
  (h : validTheaterHall hall) : hall.totalSeats = 360 := by
  sorry

#check theater_hall_seats

end theater_hall_seats_l721_72149


namespace square_circle_difference_l721_72110

-- Define the square and circle
def square_diagonal : ℝ := 8
def circle_diameter : ℝ := 8

-- Theorem statement
theorem square_circle_difference :
  let square_side := (square_diagonal ^ 2 / 2).sqrt
  let square_area := square_side ^ 2
  let square_perimeter := 4 * square_side
  let circle_radius := circle_diameter / 2
  let circle_area := π * circle_radius ^ 2
  let circle_perimeter := 2 * π * circle_radius
  (circle_area - square_area = 16 * π - 32) ∧
  (circle_perimeter - square_perimeter = 8 * π - 16 * Real.sqrt 2) :=
by sorry

end square_circle_difference_l721_72110


namespace circle_area_from_circumference_l721_72162

/-- Given a circle with circumference 24 cm, its area is 144/π square centimeters. -/
theorem circle_area_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 24 → π * r^2 = 144 / π := by
  sorry

end circle_area_from_circumference_l721_72162


namespace eric_running_time_l721_72141

/-- Given Eric's trip to the park and back, prove the time he ran before jogging. -/
theorem eric_running_time (total_time_to_park : ℕ) (running_time : ℕ) : 
  total_time_to_park = running_time + 10 →
  90 = 3 * total_time_to_park →
  running_time = 20 := by
sorry

end eric_running_time_l721_72141


namespace sodium_hypochlorite_weight_approx_l721_72113

/-- The atomic weight of sodium in g/mol -/
def sodium_weight : ℝ := 22.99

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.45

/-- The molecular weight of sodium hypochlorite (NaOCl) in g/mol -/
def sodium_hypochlorite_weight : ℝ := sodium_weight + oxygen_weight + chlorine_weight

/-- The given molecular weight of a certain substance -/
def given_weight : ℝ := 74

/-- Theorem stating that the molecular weight of sodium hypochlorite is approximately equal to the given weight -/
theorem sodium_hypochlorite_weight_approx : 
  ∃ ε > 0, |sodium_hypochlorite_weight - given_weight| < ε :=
sorry

end sodium_hypochlorite_weight_approx_l721_72113


namespace power_product_equality_l721_72120

theorem power_product_equality : 3^3 * 2^2 * 7^2 * 11 = 58212 := by
  sorry

end power_product_equality_l721_72120


namespace quadratic_maximum_l721_72160

theorem quadratic_maximum (s : ℝ) : -7 * s^2 + 56 * s + 20 ≤ 132 ∧ ∃ t : ℝ, -7 * t^2 + 56 * t + 20 = 132 := by
  sorry

end quadratic_maximum_l721_72160


namespace adam_initial_money_l721_72108

/-- The cost of the airplane in dollars -/
def airplane_cost : ℚ := 4.28

/-- The change Adam received in dollars -/
def change_received : ℚ := 0.72

/-- Adam's initial amount of money in dollars -/
def initial_money : ℚ := airplane_cost + change_received

theorem adam_initial_money : initial_money = 5 := by
  sorry

end adam_initial_money_l721_72108


namespace student_B_visited_C_l721_72132

structure Student :=
  (name : String)
  (visited : Finset String)

def University : Type := String

theorem student_B_visited_C (studentA studentB studentC : Student) 
  (univA univB univC : University) :
  studentA.name = "A" →
  studentB.name = "B" →
  studentC.name = "C" →
  univA = "A" →
  univB = "B" →
  univC = "C" →
  studentA.visited.card > studentB.visited.card →
  univA ∉ studentA.visited →
  univB ∉ studentB.visited →
  ∃ (u : University), u ∈ studentA.visited ∧ u ∈ studentB.visited ∧ u ∈ studentC.visited →
  univC ∈ studentB.visited :=
by sorry

end student_B_visited_C_l721_72132


namespace tank_filling_l721_72173

theorem tank_filling (original_buckets : ℕ) (capacity_reduction : ℚ) : 
  original_buckets = 10 →
  capacity_reduction = 2/5 →
  ∃ (new_buckets : ℕ), new_buckets ≥ 25 ∧ 
    (new_buckets : ℚ) * capacity_reduction = original_buckets := by
  sorry

end tank_filling_l721_72173


namespace shower_has_three_walls_l721_72121

/-- Represents the properties of a shower with tiled walls -/
structure Shower :=
  (width_tiles : ℕ)
  (height_tiles : ℕ)
  (total_tiles : ℕ)

/-- Calculates the number of walls in a shower -/
def number_of_walls (s : Shower) : ℚ :=
  s.total_tiles / (s.width_tiles * s.height_tiles)

/-- Theorem: The shower has 3 walls -/
theorem shower_has_three_walls (s : Shower) 
  (h1 : s.width_tiles = 8)
  (h2 : s.height_tiles = 20)
  (h3 : s.total_tiles = 480) : 
  number_of_walls s = 3 := by
  sorry

#eval number_of_walls { width_tiles := 8, height_tiles := 20, total_tiles := 480 }

end shower_has_three_walls_l721_72121


namespace ice_cream_sales_for_games_l721_72192

theorem ice_cream_sales_for_games (game_cost : ℕ) (ice_cream_price : ℕ) : 
  game_cost = 60 → ice_cream_price = 5 → (2 * game_cost) / ice_cream_price = 24 := by
  sorry

#check ice_cream_sales_for_games

end ice_cream_sales_for_games_l721_72192


namespace cone_volume_l721_72187

/-- Given a cone with slant height 3 and lateral surface area 3π, its volume is (2√2π)/3 -/
theorem cone_volume (l : ℝ) (A_L : ℝ) (h : ℝ) (r : ℝ) (V : ℝ) : 
  l = 3 →
  A_L = 3 * Real.pi →
  A_L = Real.pi * r * l →
  l^2 = h^2 + r^2 →
  V = (1/3) * Real.pi * r^2 * h →
  V = (2 * Real.sqrt 2 * Real.pi) / 3 := by
sorry

end cone_volume_l721_72187


namespace geometric_sequence_sum_l721_72126

def geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) :
  geometric_sequence a r →
  a 6 = 1 →
  a 7 = 0.25 →
  a 3 + a 4 = 80 :=
by
  sorry

end geometric_sequence_sum_l721_72126


namespace x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l721_72154

theorem x_eq_one_sufficient_not_necessary_for_x_squared_eq_one :
  (∃ x : ℝ, x ^ 2 = 1 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → x ^ 2 = 1) :=
by sorry

end x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l721_72154


namespace system_solution_l721_72117

theorem system_solution :
  ∃! (x y : ℤ), 16*x + 24*y = 32 ∧ 24*x + 16*y = 48 :=
by
  -- The proof goes here
  sorry

end system_solution_l721_72117


namespace set_operations_l721_72134

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 6}

-- Define the theorem
theorem set_operations :
  (A ∩ B = {x : ℝ | 3 ≤ x ∧ x < 5}) ∧
  (A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 6}) ∧
  ((Aᶜ : Set ℝ) ∩ B = {x : ℝ | 5 ≤ x ∧ x ≤ 6}) ∧
  ((A ∩ B)ᶜ = {x : ℝ | x < 3 ∨ x ≥ 5}) := by
  sorry

end set_operations_l721_72134


namespace range_of_a_when_proposition_is_false_l721_72135

theorem range_of_a_when_proposition_is_false :
  (¬∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) → (0 < a ∧ a < 1) :=
by sorry

end range_of_a_when_proposition_is_false_l721_72135


namespace fried_chicken_dinner_orders_l721_72190

/-- Represents the number of pieces of chicken used in different order types -/
structure ChickenPieces where
  pasta : Nat
  barbecue : Nat
  friedDinner : Nat

/-- Represents the number of orders for each type -/
structure Orders where
  pasta : Nat
  barbecue : Nat
  friedDinner : Nat

/-- Calculates the total number of chicken pieces used -/
def totalChickenPieces (cp : ChickenPieces) (o : Orders) : Nat :=
  cp.pasta * o.pasta + cp.barbecue * o.barbecue + cp.friedDinner * o.friedDinner

/-- The main theorem to prove -/
theorem fried_chicken_dinner_orders
  (cp : ChickenPieces)
  (o : Orders)
  (h1 : cp.pasta = 2)
  (h2 : cp.barbecue = 3)
  (h3 : cp.friedDinner = 8)
  (h4 : o.pasta = 6)
  (h5 : o.barbecue = 3)
  (h6 : totalChickenPieces cp o = 37) :
  o.friedDinner = 2 := by
  sorry

end fried_chicken_dinner_orders_l721_72190


namespace incorrect_statement_l721_72198

theorem incorrect_statement : ¬(0 > |(-1)|) ∧ (-(-3) = 3) ∧ (|2| = |-2|) ∧ (-2 > -3) := by
  sorry

end incorrect_statement_l721_72198


namespace spongebob_daily_earnings_l721_72167

/-- Spongebob's earnings for a day of work at the burger shop -/
def spongebob_earnings (num_burgers : ℕ) (price_burger : ℚ) (num_fries : ℕ) (price_fries : ℚ) : ℚ :=
  num_burgers * price_burger + num_fries * price_fries

/-- Theorem: Spongebob's earnings for the day are $78 -/
theorem spongebob_daily_earnings : 
  spongebob_earnings 30 2 12 (3/2) = 78 := by
sorry

end spongebob_daily_earnings_l721_72167


namespace rectangle_perimeter_l721_72148

/-- Given a triangle with sides 9, 12, and 15 units and a rectangle with width 6 units
    and area equal to the triangle's area, the perimeter of the rectangle is 30 units. -/
theorem rectangle_perimeter (triangle_side1 triangle_side2 triangle_side3 rectangle_width : ℝ) :
  triangle_side1 = 9 ∧ triangle_side2 = 12 ∧ triangle_side3 = 15 ∧ rectangle_width = 6 ∧
  (1/2 * triangle_side1 * triangle_side2 = rectangle_width * (1/2 * triangle_side1 * triangle_side2 / rectangle_width)) →
  2 * (rectangle_width + (1/2 * triangle_side1 * triangle_side2 / rectangle_width)) = 30 :=
by sorry

end rectangle_perimeter_l721_72148


namespace mary_remaining_money_l721_72182

def initial_money : ℕ := 58
def pie_cost : ℕ := 6

theorem mary_remaining_money :
  initial_money - pie_cost = 52 := by sorry

end mary_remaining_money_l721_72182


namespace difference_in_base8_l721_72118

/-- Converts a base 8 number represented as a list of digits to its decimal equivalent -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a decimal number to its base 8 representation as a list of digits -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec convert (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else convert (m / 8) ((m % 8) :: acc)
    convert n []

theorem difference_in_base8 :
  let a := base8ToDecimal [1, 2, 3, 4]
  let b := base8ToDecimal [7, 6, 5]
  decimalToBase8 (a - b) = [2, 2, 5] :=
by sorry

end difference_in_base8_l721_72118


namespace fair_attendance_l721_72159

/-- The total attendance at a fair over three years -/
def total_attendance (this_year : ℕ) : ℕ :=
  let next_year := 2 * this_year
  let last_year := next_year - 200
  last_year + this_year + next_year

/-- Theorem: The total attendance over three years is 2800 -/
theorem fair_attendance : total_attendance 600 = 2800 := by
  sorry

end fair_attendance_l721_72159


namespace binary_110101_equals_53_l721_72127

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 110101₂ -/
def binary_110101 : List Bool := [true, false, true, false, true, true]

theorem binary_110101_equals_53 : binary_to_decimal binary_110101 = 53 := by
  sorry

end binary_110101_equals_53_l721_72127


namespace no_cube_in_range_l721_72165

theorem no_cube_in_range : ¬ ∃ n : ℕ, 4 ≤ n ∧ n ≤ 12 ∧ ∃ k : ℕ, n^2 + 3*n + 1 = k^3 := by
  sorry

end no_cube_in_range_l721_72165


namespace semicircle_perimeter_approx_l721_72136

/-- The perimeter of a semicircle with radius 9 is approximately 46.26 -/
theorem semicircle_perimeter_approx :
  let r : ℝ := 9
  let π_approx : ℝ := 3.14
  let semicircle_perimeter := r * π_approx + 2 * r
  ∃ ε > 0, abs (semicircle_perimeter - 46.26) < ε :=
by
  sorry

end semicircle_perimeter_approx_l721_72136


namespace square_difference_ratio_l721_72195

theorem square_difference_ratio : 
  (1632^2 - 1629^2) / (1635^2 - 1626^2) = 1/3 := by
  sorry

end square_difference_ratio_l721_72195


namespace trail_mix_weight_l721_72128

/-- The weight of peanuts in pounds -/
def peanuts : ℝ := 0.17

/-- The weight of chocolate chips in pounds -/
def chocolate_chips : ℝ := 0.17

/-- The weight of raisins in pounds -/
def raisins : ℝ := 0.08

/-- The weight of dried apricots in pounds -/
def dried_apricots : ℝ := 0.12

/-- The weight of sunflower seeds in pounds -/
def sunflower_seeds : ℝ := 0.09

/-- The weight of coconut flakes in pounds -/
def coconut_flakes : ℝ := 0.15

/-- The total weight of trail mix in pounds -/
def total_weight : ℝ := peanuts + chocolate_chips + raisins + dried_apricots + sunflower_seeds + coconut_flakes

theorem trail_mix_weight : total_weight = 0.78 := by
  sorry

end trail_mix_weight_l721_72128


namespace new_average_weight_l721_72123

/-- Given a bowling team with the following properties:
  * The original team has 7 players
  * The original average weight is 103 kg
  * Two new players join the team
  * One new player weighs 110 kg
  * The other new player weighs 60 kg
  
  Prove that the new average weight of the team is 99 kg -/
theorem new_average_weight 
  (original_players : Nat) 
  (original_avg_weight : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) 
  (h1 : original_players = 7)
  (h2 : original_avg_weight = 103)
  (h3 : new_player1_weight = 110)
  (h4 : new_player2_weight = 60) :
  let total_weight := original_players * original_avg_weight + new_player1_weight + new_player2_weight
  let new_total_players := original_players + 2
  total_weight / new_total_players = 99 := by
sorry

end new_average_weight_l721_72123


namespace lynne_cat_books_l721_72157

def books_about_cats (x : ℕ) : Prop :=
  ∃ (total_spent : ℕ),
    let books_solar_system := 2
    let magazines := 3
    let book_cost := 7
    let magazine_cost := 4
    total_spent = 75 ∧
    total_spent = x * book_cost + books_solar_system * book_cost + magazines * magazine_cost

theorem lynne_cat_books : ∃ x : ℕ, books_about_cats x ∧ x = 7 := by
  sorry

end lynne_cat_books_l721_72157


namespace square_root_problem_l721_72191

theorem square_root_problem (x : ℝ) : (Real.sqrt x / 11 = 4) → x = 1936 := by
  sorry

end square_root_problem_l721_72191


namespace jason_borrowed_amount_l721_72101

/-- Calculates the total earnings for a given number of hours based on the described payment structure -/
def jasonEarnings (hours : ℕ) : ℕ :=
  let fullCycles := hours / 9
  let remainingHours := hours % 9
  let earningsPerCycle := (List.range 9).sum
  fullCycles * earningsPerCycle + (List.range remainingHours).sum

theorem jason_borrowed_amount :
  jasonEarnings 27 = 135 := by
  sorry

end jason_borrowed_amount_l721_72101


namespace casey_owns_five_hoodies_l721_72180

/-- The number of hoodies Fiona and Casey own together -/
def total_hoodies : ℕ := 8

/-- The number of hoodies Fiona owns -/
def fiona_hoodies : ℕ := 3

/-- The number of hoodies Casey owns -/
def casey_hoodies : ℕ := total_hoodies - fiona_hoodies

theorem casey_owns_five_hoodies : casey_hoodies = 5 := by
  sorry

end casey_owns_five_hoodies_l721_72180


namespace original_number_l721_72145

theorem original_number : ∃ x : ℝ, 10 * x = x + 81 ∧ x = 9 := by sorry

end original_number_l721_72145


namespace craigs_walk_distance_l721_72197

theorem craigs_walk_distance (distance_school_to_david : ℝ) (distance_david_to_home : ℝ)
  (h1 : distance_school_to_david = 0.27)
  (h2 : distance_david_to_home = 0.73) :
  distance_school_to_david + distance_david_to_home = 1.00 := by
  sorry

end craigs_walk_distance_l721_72197


namespace no_real_roots_l721_72185

theorem no_real_roots :
  ∀ x : ℝ, ¬(Real.sqrt (x + 9) - Real.sqrt (x - 5) + 2 = 0) :=
by sorry

end no_real_roots_l721_72185


namespace min_draws_for_twelve_balls_l721_72152

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  black : Nat

/-- Represents the minimum number of balls needed to guarantee at least n balls of a single color -/
def minDrawsForColor (counts : BallCounts) (n : Nat) : Nat :=
  sorry

/-- The main theorem stating the minimum number of draws required -/
theorem min_draws_for_twelve_balls (counts : BallCounts) 
  (h_red : counts.red = 30)
  (h_green : counts.green = 22)
  (h_yellow : counts.yellow = 18)
  (h_blue : counts.blue = 15)
  (h_black : counts.black = 10) :
  minDrawsForColor counts 12 = 55 := by
  sorry

end min_draws_for_twelve_balls_l721_72152


namespace division_problem_l721_72143

theorem division_problem (remainder quotient divisor dividend : ℕ) : 
  remainder = 5 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + 3 →
  dividend = divisor * quotient + remainder →
  dividend = 113 := by
sorry

end division_problem_l721_72143


namespace common_root_sum_k_l721_72184

theorem common_root_sum_k : ∃ (k₁ k₂ : ℝ),
  (∃ x : ℝ, x^2 - 4*x + 3 = 0 ∧ x^2 - 6*x + k₁ = 0) ∧
  (∃ x : ℝ, x^2 - 4*x + 3 = 0 ∧ x^2 - 6*x + k₂ = 0) ∧
  k₁ ≠ k₂ ∧
  k₁ + k₂ = 14 :=
by sorry

end common_root_sum_k_l721_72184


namespace x_minus_y_values_l721_72125

theorem x_minus_y_values (x y : ℝ) (h1 : x^2 = 4) (h2 : |y| = 5) (h3 : x * y < 0) :
  x - y = -7 ∨ x - y = 7 := by
  sorry

end x_minus_y_values_l721_72125


namespace initial_number_equation_l721_72124

theorem initial_number_equation : ∃ x : ℝ, 3 * (2 * x + 13) = 93 :=
by sorry

end initial_number_equation_l721_72124


namespace sum_squared_equals_400_l721_72188

variable (a b c : ℝ)

theorem sum_squared_equals_400 
  (h1 : a^2 + b^2 + c^2 = 390) 
  (h2 : a*b + b*c + c*a = 5) : 
  (a + b + c)^2 = 400 := by
sorry

end sum_squared_equals_400_l721_72188


namespace division_remainder_proof_l721_72177

theorem division_remainder_proof :
  let dividend : ℕ := 165
  let divisor : ℕ := 18
  let quotient : ℕ := 9
  let remainder : ℕ := dividend - divisor * quotient
  remainder = 3 := by
sorry

end division_remainder_proof_l721_72177


namespace alex_coin_distribution_distribution_satisfies_conditions_l721_72111

/-- The minimum number of additional coins needed -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for Alex's scenario -/
theorem alex_coin_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 105) :
  min_additional_coins num_friends initial_coins = 15 := by
  sorry

/-- Proof that the distribution satisfies the conditions -/
theorem distribution_satisfies_conditions (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 105) :
  ∀ i j, i ≠ j → i ≤ num_friends → j ≤ num_friends → 
  (i : ℕ) ≠ (j : ℕ) ∧ (i : ℕ) ≥ 1 ∧ (j : ℕ) ≥ 1 := by
  sorry

end alex_coin_distribution_distribution_satisfies_conditions_l721_72111


namespace arithmetic_sequence_100th_term_unique_index_298_l721_72199

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℕ := 1 + (n - 1) * 3

/-- The theorem stating that the 100th term of the arithmetic sequence is 298 -/
theorem arithmetic_sequence_100th_term :
  arithmetic_sequence 100 = 298 := by sorry

/-- The theorem stating that 100 is the unique index for which the term is 298 -/
theorem unique_index_298 :
  ∀ n : ℕ, arithmetic_sequence n = 298 ↔ n = 100 := by sorry

end arithmetic_sequence_100th_term_unique_index_298_l721_72199


namespace x_squared_plus_four_y_squared_lt_one_l721_72153

theorem x_squared_plus_four_y_squared_lt_one
  (x y : ℝ)
  (x_pos : x > 0)
  (y_pos : y > 0)
  (h : x^3 + y^3 = x - y) :
  x^2 + 4*y^2 < 1 :=
by sorry

end x_squared_plus_four_y_squared_lt_one_l721_72153


namespace stock_percentage_l721_72115

/-- Given the income, price per unit, and total investment of a stock,
    calculate the percentage of the stock. -/
theorem stock_percentage
  (income : ℝ)
  (price_per_unit : ℝ)
  (total_investment : ℝ)
  (h1 : income = 900)
  (h2 : price_per_unit = 102)
  (h3 : total_investment = 4590)
  : (income / total_investment) * 100 = (900 : ℝ) / 4590 * 100 := by
  sorry

end stock_percentage_l721_72115


namespace three_lines_triangle_l721_72140

/-- A line in the 2D plane represented by its equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if three lines intersect at a single point -/
def intersect_at_point (l1 l2 l3 : Line) : Prop :=
  ∃ x y : ℝ, l1.a * x + l1.b * y = l1.c ∧
             l2.a * x + l2.b * y = l2.c ∧
             l3.a * x + l3.b * y = l3.c

/-- The set of possible values for m -/
def possible_m_values : Set ℝ :=
  {m : ℝ | m = 4 ∨ m = -1/6 ∨ m = 1 ∨ m = -2/3}

theorem three_lines_triangle (m : ℝ) :
  let l1 : Line := ⟨4, 1, 4⟩
  let l2 : Line := ⟨m, 1, 0⟩
  let l3 : Line := ⟨2, -3*m, 4⟩
  (parallel l1 l2 ∨ parallel l1 l3 ∨ parallel l2 l3 ∨ intersect_at_point l1 l2 l3) →
  m ∈ possible_m_values :=
sorry

end three_lines_triangle_l721_72140


namespace book_arrangement_count_l721_72100

theorem book_arrangement_count : ℕ := by
  -- Define the total number of books
  let total_books : ℕ := 6
  -- Define the number of identical copies for each book type
  let identical_copies1 : ℕ := 3
  let identical_copies2 : ℕ := 2
  let unique_book : ℕ := 1

  -- Assert that the sum of all book types equals the total number of books
  have h_total : identical_copies1 + identical_copies2 + unique_book = total_books := by sorry

  -- Define the number of distinct arrangements
  let arrangements : ℕ := Nat.factorial total_books / (Nat.factorial identical_copies1 * Nat.factorial identical_copies2)

  -- Prove that the number of distinct arrangements is 60
  have h_result : arrangements = 60 := by sorry

  -- Return the result
  exact 60

end book_arrangement_count_l721_72100


namespace value_of_expression_l721_72164

theorem value_of_expression (x : ℝ) (h : x = -3) : 3 * x^2 + 2 * x = 21 := by
  sorry

end value_of_expression_l721_72164
